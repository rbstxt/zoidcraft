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

/// Static catalog entry for native Zoidium plugins (Spec 04 §5.2–5.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PluginCatalogEntry {
    id: &'static str,
    name: &'static str,
    category: &'static str,
    default_enabled: bool,
}

const KNOWN_PLUGINS: &[PluginCatalogEntry] = &[
    // Utilities
    PluginCatalogEntry {
        id: "easing-plus",
        name: "Easing+",
        category: "utilities",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "expressions-plus",
        name: "Expressions+",
        category: "utilities",
        default_enabled: true,
    },
    PluginCatalogEntry {
        id: "shader-plus",
        name: "Shader+",
        category: "utilities",
        default_enabled: true,
    },
    PluginCatalogEntry {
        id: "player-plus",
        name: "Player+",
        category: "utilities",
        default_enabled: true,
    },
    PluginCatalogEntry {
        id: "asset-manager",
        name: "Assets",
        category: "utilities",
        default_enabled: true,
    },
    PluginCatalogEntry {
        id: "precomp-plus",
        name: "Precomp+",
        category: "utilities",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "gif-support",
        name: "Animated GIF",
        category: "utilities",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "graph-plus",
        name: "Graph Editor 2",
        category: "utilities",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "motion-path",
        name: "Motion Paths",
        category: "utilities",
        default_enabled: true,
    },
    PluginCatalogEntry {
        id: "mp4-export",
        name: "MP4 Export",
        category: "utilities",
        default_enabled: true,
    },
    // Scene Extensions
    PluginCatalogEntry {
        id: "light-plus",
        name: "Light+",
        category: "scene",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "geometry-plus",
        name: "Geometry+",
        category: "scene",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "repeater",
        name: "Effector",
        category: "scene",
        default_enabled: true,
    },
    PluginCatalogEntry {
        id: "effector-plus",
        name: "Effector+",
        category: "scene",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "material-plus",
        name: "Material+",
        category: "scene",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "text-plus",
        name: "Text+",
        category: "scene",
        default_enabled: true,
    },
    PluginCatalogEntry {
        id: "expresstext-plus",
        name: "ExpressText+",
        category: "scene",
        default_enabled: true,
    },
    PluginCatalogEntry {
        id: "particles-plus",
        name: "Particles+",
        category: "scene",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "trapcode-suite",
        name: "Rowbyte & Red Giant Suite",
        category: "scene",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "optical-flares",
        name: "Optical Flares",
        category: "scene",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "camera-plus",
        name: "Camera+",
        category: "scene",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "shape-plus",
        name: "Shape+",
        category: "scene",
        default_enabled: true,
    },
    PluginCatalogEntry {
        id: "constraints",
        name: "Constraints",
        category: "scene",
        default_enabled: true,
    },
    // Adjustment Extensions
    PluginCatalogEntry {
        id: "native-fx",
        name: "Native FX",
        category: "adjustment",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "node-effect",
        name: "Node Effect",
        category: "adjustment",
        default_enabled: true,
    },
    PluginCatalogEntry {
        id: "magic-looks",
        name: "Magic Looks",
        category: "adjustment",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "openzoid-legacy",
        name: "OpenZoid Legacy",
        category: "adjustment",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "smart-motion-blur-lite",
        name: "ZSMB (beta 1.0)",
        category: "adjustment",
        default_enabled: false,
    },
    PluginCatalogEntry {
        id: "sglow-hq-v2-preview",
        name: "S_Glow (beta 1.0)",
        category: "adjustment",
        default_enabled: false,
    },
    // Experimental
    PluginCatalogEntry {
        id: "layer-input",
        name: "Layer Input",
        category: "experimental",
        default_enabled: false,
    },
];

// ---------------------------------------------------------------------------
// Editable Session Document (kept synchronized with `zc_project::Project`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackKind {
    Video,
    Audio,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionKeyframe {
    pub frame: f64,
    pub value: Value,
    pub tween: u16,
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
            keyframes: vec![SessionKeyframe {
                frame: 0.0,
                value,
                tween: 1,
            }],
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
        }
    }

    fn reset(&mut self) {
        self.animated = false;
        self.expression = None;
        self.keyframes = vec![SessionKeyframe {
            frame: 0.0,
            value: self.default_value.clone(),
            tween: 1,
        }];
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionObject3D {
    pub id: String,
    pub parent_id: Option<String>,
    pub kind: String,
    pub name: String,
    pub properties: BTreeMap<String, SessionProperty>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionEffect {
    pub id: String,
    pub kind: String,
    pub name: String,
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
                    SessionKeyframe {
                        frame: 0.0,
                        value: json!(self.time_start),
                        tween: 1,
                    },
                    SessionKeyframe {
                        frame: self.length,
                        value: json!(self.time_end),
                        tween: 1,
                    },
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

fn property_track_to_session(track: &zc_project::PropertyTrack) -> SessionProperty {
    let default_value = prop_value_to_json(&track.value);
    let keyframes = if track.keyframes.is_empty() {
        vec![SessionKeyframe {
            frame: 0.0,
            value: default_value.clone(),
            tween: 1,
        }]
    } else {
        track
            .keyframes
            .iter()
            .map(|kf| SessionKeyframe {
                frame: kf.frame,
                value: prop_value_to_json(&kf.value),
                tween: kf.tween as u16,
            })
            .collect()
    };
    SessionProperty {
        animated: track.animated,
        expression: track.expression.clone(),
        default_value,
        keyframes,
    }
}

fn session_property_to_track(
    prop: &SessionProperty,
    fallback: Option<&zc_project::PropertyTrack>,
) -> zc_project::PropertyTrack {
    let value = prop
        .keyframes
        .first()
        .map(|k| json_to_prop_value(&k.value))
        .unwrap_or_else(|| json_to_prop_value(&prop.default_value));
    let keyframes = prop
        .keyframes
        .iter()
        .map(|kf| {
            let (cp, cont) = fallback
                .and_then(|t| {
                    t.keyframes
                        .iter()
                        .find(|existing| (existing.frame - kf.frame).abs() < 1e-6)
                })
                .map_or(([[-10.0, 0.0], [10.0, 0.0]], true), |existing| {
                    (existing.control_points, existing.continuous_tangent)
                });
            zc_project::PropertyKeyframe::from_cm3_tween(
                kf.frame,
                json_to_prop_value(&kf.value),
                u32::from(kf.tween),
                cp,
                cont,
            )
        })
        .collect();
    zc_project::PropertyTrack {
        animated: prop.animated,
        expression: prop.expression.clone(),
        static_behavior: fallback
            .map_or(zc_project::StaticBehavior::Constant, |t| t.static_behavior),
        value,
        keyframes,
        channels: fallback.map_or_else(Vec::new, |t| t.channels.clone()),
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

fn object_3d_kind_to_str(kind: &zc_project::Object3DKind) -> String {
    match kind {
        zc_project::Object3DKind::Group { .. } => "group".to_string(),
        zc_project::Object3DKind::Text3D(_) => "text_3d".to_string(),
        zc_project::Object3DKind::Shape3D(s) => match s {
            zc_project::Shape3DKind::Box { .. } => "shape.box".to_string(),
            zc_project::Shape3DKind::Cylinder { .. } => "shape.cylinder".to_string(),
            zc_project::Shape3DKind::Rectangle { .. } => "shape.rectangle".to_string(),
            zc_project::Shape3DKind::Circle { .. } => "shape.circle".to_string(),
            zc_project::Shape3DKind::Sphere { .. } => "shape.sphere".to_string(),
            zc_project::Shape3DKind::Donut { .. } => "shape.donut".to_string(),
            zc_project::Shape3DKind::Wire { .. } => "shape.wire".to_string(),
            zc_project::Shape3DKind::CustomGeometry { .. } => "shape.custom".to_string(),
            _ => "zoidium:geometry-plus/shape".to_string(),
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

        let obj_id = obj.id.to_string();
        out.push(SessionObject3D {
            id: obj_id.clone(),
            parent_id: parent_id.clone(),
            kind: object_3d_kind_to_str(&obj.kind),
            name: obj.name.clone(),
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
    for (k, track) in &fx.params {
        properties.insert(k.clone(), property_track_to_session(track));
    }
    SessionEffect {
        id: fx.id.to_string(),
        kind: effect_kind_to_session_id(&fx.kind),
        name: fx.name.clone(),
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
        // Standard CM3 linear clip time uses StaticBehavior::Linear with 2 linear keyframes.
        let is_standard_linear = !time_curve.animated
            || (time_curve.static_behavior == zc_project::StaticBehavior::Linear
                && time_curve.keyframes.len() == 2
                && time_curve.expression.is_none());
        (t0, t1, !is_standard_linear)
    } else {
        (0.0, default_end, time_curve.animated)
    }
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
                    "time".to_string(),
                    property_track_to_session(&vc.time_curve),
                );
                let mut objects_3d = Vec::new();
                if let zc_project::LayerKind::Scene3D { objects, .. } = &vc.layer.kind {
                    flatten_objects_3d(objects, None, &mut objects_3d);
                }
                let effects = vc.layer.effects.iter().map(effect_to_session).collect();
                let mut clip = SessionClip {
                    id: vc.id.to_string(),
                    kind: layer_kind_to_session_kind(&vc.layer.kind),
                    name: vc.name.clone(),
                    start: vc.start,
                    length: vc.length,
                    time_start,
                    time_end,
                    time_animated,
                    media_length: None,
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
                let mut clip = SessionClip {
                    id: ac.id.to_string(),
                    kind: "audio".to_string(),
                    name: ac.name.clone(),
                    start: ac.start,
                    length: ac.length,
                    time_start,
                    time_end,
                    time_animated,
                    media_length: None,
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
                enabled: at.enabled,
                solo: at.solo,
                locked: false,
                clips,
            });
        }

        Self {
            project_name: project.name.clone(),
            resolution: project.sequence.resolution,
            fps,
            sequence_length: project.sequence.length_frames,
            video_tracks,
            audio_tracks,
            sequence_properties: BTreeMap::new(),
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
            if let Some(rest) = kind.strip_prefix("zoidium:")
                && let Some((plugin_id, _)) = rest.split_once('/')
                && !plugin_id.is_empty()
            {
                deps.insert(plugin_id.to_string());
            }
        }

        for track in self.all_tracks() {
            for clip in &track.clips {
                inspect_kind(&clip.kind, &mut deps);
                for obj in &clip.objects_3d {
                    inspect_kind(&obj.kind, &mut deps);
                    if obj.properties.values().any(|p| p.expression.is_some()) {
                        deps.insert("expressions-plus".to_string());
                    }
                }
                for fx in &clip.effects {
                    inspect_kind(&fx.kind, &mut deps);
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
        project_json_bytes
            .saturating_add(asset_blob_bytes)
            .saturating_add(doc_bytes)
            .saturating_add(sel_bytes)
    }
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
        let mut asset_blobs = BTreeMap::new();
        for (key, rec) in &project.assets {
            if let Some(bytes) = rec.bytes.as_ref() {
                asset_blobs.insert(key.clone(), Arc::new(bytes.clone()));
            }
        }
        Self {
            project,
            plugins,
            selection: SelectionState::default(),
            playback: PlaybackState::default(),
            history: UndoStack::new(),
            registry: CommandRegistry::new_default(),
            document,
            asset_blobs,
            next_id: 1,
        }
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

    /// Captures a complete undo snapshot of the current session (including any embedded asset bytes).
    #[must_use]
    pub fn snapshot(&self) -> SessionSnapshot {
        SessionSnapshot {
            project: self.project.clone(),
            document: self.document.clone(),
            selection: self.selection.clone(),
        }
    }

    /// Captures a lightweight undo snapshot for history storage: any embedded asset binary
    /// buffers are deduplicated into `self.asset_blobs` and stripped from the snapshot clone
    /// so repeated edits never duplicate megabytes of media in undo memory.
    fn capture_history_snapshot(&mut self) -> SessionSnapshot {
        for (key, rec) in &self.project.assets {
            if let Some(bytes) = rec.bytes.as_ref() {
                let needs_update = match self.asset_blobs.get(key) {
                    Some(existing) => existing.as_slice() != bytes.as_slice(),
                    None => true,
                };
                if needs_update {
                    self.asset_blobs
                        .insert(key.clone(), Arc::new(bytes.clone()));
                }
            }
        }
        let mut lightweight_project = self.project.clone();
        for rec in lightweight_project.assets.values_mut() {
            rec.bytes = None;
        }
        SessionSnapshot {
            project: lightweight_project,
            document: self.document.clone(),
            selection: self.selection.clone(),
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
        Ok(self.history.commit_transaction(changed)?)
    }

    /// Cancels the active transaction and restores the exact pre-transaction state without
    /// pushing an undo entry.
    pub fn cancel_transaction(&mut self) -> Result<(), SessionError> {
        let baseline = self.history.cancel_transaction()?;
        self.restore_snapshot(baseline);
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
            let result = self.execute_inner(command)?;
            if let Some(before_snapshot) = before {
                self.history.push(label, before_snapshot);
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
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        format!("00000000-0000-4000-8000-{id:012x}")
    }

    fn restore_snapshot(&mut self, mut snapshot: SessionSnapshot) {
        for (key, rec) in &mut snapshot.project.assets {
            if rec.bytes.is_none()
                && let Some(blob) = self.asset_blobs.get(key)
            {
                rec.bytes = Some((**blob).clone());
            }
        }
        self.project = snapshot.project;
        self.document = snapshot.document;
        self.selection = snapshot.selection;
        self.sync_document_to_project();
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

        let fps_u32 = self.project.sequence.fps.max(1);
        let resolution = self.document.resolution;

        // Index existing tracks and clips by UUID string so preserved metadata survives sync.
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

        let mut synced_vtracks = Vec::with_capacity(self.document.video_tracks.len());
        for strack in &self.document.video_tracks {
            let mut vt =
                existing_vtracks
                    .remove(&strack.id)
                    .unwrap_or_else(|| zc_project::VideoTrack {
                        id: parse_uuid_or_new(&strack.id),
                        name: strack.name.clone(),
                        enabled: strack.enabled,
                        solo: strack.solo,
                        locked: strack.locked,
                        is_3d: false,
                        clips: Vec::new(),
                        preserved: BTreeMap::new(),
                    });
            vt.name = strack.name.clone();
            vt.enabled = strack.enabled;
            vt.solo = strack.solo;
            vt.locked = strack.locked;
            vt.clips = strack
                .clips
                .iter()
                .map(|sclip| {
                    sync_video_clip_from_session(
                        sclip,
                        existing_vclips.remove(&sclip.id),
                        resolution,
                        fps_u32,
                    )
                })
                .collect();
            synced_vtracks.push(vt);
        }
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

        let mut synced_atracks = Vec::with_capacity(self.document.audio_tracks.len());
        for strack in &self.document.audio_tracks {
            let mut at =
                existing_atracks
                    .remove(&strack.id)
                    .unwrap_or_else(|| zc_project::AudioTrack {
                        id: parse_uuid_or_new(&strack.id),
                        name: strack.name.clone(),
                        enabled: strack.enabled,
                        muted: false,
                        solo: strack.solo,
                        clips: Vec::new(),
                        preserved: BTreeMap::new(),
                    });
            at.name = strack.name.clone();
            at.enabled = strack.enabled;
            at.solo = strack.solo;
            at.clips = strack
                .clips
                .iter()
                .map(|sclip| {
                    sync_audio_clip_from_session(sclip, existing_aclips.remove(&sclip.id), fps_u32)
                })
                .collect();
            synced_atracks.push(at);
        }
        self.project.sequence.audio_tracks = synced_atracks;

        for dep in self.document.collect_plugin_dependencies() {
            self.project.plugin_requirements.insert(dep);
        }
    }

    /// Spec 02 §2.5 Step B: enforces strict non-overlap on `track` around `placed_clip_id`.
    /// Splits straddled left neighbors, right-trims partially overlapped left neighbors,
    /// removes engulfed neighbors, and left-trims partially overlapped right neighbors
    /// (updating both keyframes and non-animated media-time endpoints).
    fn resolve_track_overlaps(&mut self, kind: TrackKind, track_idx: usize, placed_clip_id: &str) {
        let fps = self.document.fps.max(1.0);

        let (placed_start, placed_end) = {
            let track = match kind {
                TrackKind::Video => &self.document.video_tracks[track_idx],
                TrackKind::Audio => &self.document.audio_tracks[track_idx],
            };
            let Some(placed) = track.clips.iter().find(|c| c.id == placed_clip_id) else {
                return;
            };
            (placed.start, placed.start + placed.length)
        };

        let existing_clips = {
            let track = match kind {
                TrackKind::Video => &mut self.document.video_tracks[track_idx],
                TrackKind::Audio => &mut self.document.audio_tracks[track_idx],
            };
            std::mem::take(&mut track.clips)
        };

        let mut kept: Vec<SessionClip> = Vec::with_capacity(existing_clips.len() + 1);
        let mut removed_for_selection: Vec<SessionClip> = Vec::new();

        for mut other in existing_clips {
            if other.id == placed_clip_id {
                kept.push(other);
                continue;
            }
            let other_start = other.start;
            let other_end = other.start + other.length;

            // Completely disjoint: no overlap.
            if other_end <= placed_start || other_start >= placed_end {
                kept.push(other);
                continue;
            }

            // Completely engulfed by placed clip.
            if other_start >= placed_start && other_end <= placed_end {
                removed_for_selection.push(other);
                continue;
            }

            // Straddle split: other starts strictly before placed_start and ends strictly after placed_end.
            if other_start < placed_start && other_end > placed_end {
                let mut right_rem = other.clone();
                let right_id = self.alloc_uuid_id();
                right_rem.id = right_id;
                let cut_right = placed_end - other_start;
                right_rem.start = placed_end;
                right_rem.length = other_end - placed_end;
                right_rem.shift_all_keyframes(-cut_right);
                if !right_rem.time_animated {
                    right_rem.time_start = other.eval_time_at(cut_right);
                    right_rem.sync_time_property();
                }
                self.remap_cloned_clip_children(&mut right_rem);

                let new_left_len = placed_start - other_start;
                if !other.time_animated {
                    other.time_end = other.eval_time_at(new_left_len);
                }
                other.length = new_left_len;
                other.sync_time_property();

                kept.push(other);
                kept.push(right_rem);
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
                    kept.push(other);
                }
                continue;
            }

            // Right neighbor partially overlapped on its left edge (other_start >= placed_start && other_end > placed_end).
            let cut = placed_end - other_start;
            let new_len = other_end - placed_end;
            if new_len < 1e-6 {
                removed_for_selection.push(other);
            } else {
                if !other.time_animated {
                    let old_rate = if other.length > 0.0 {
                        (other.time_end - other.time_start) / other.length
                    } else {
                        1.0 / fps
                    };
                    other.time_start += cut * old_rate;
                }
                other.start = placed_end;
                other.length = new_len;
                other.shift_all_keyframes(-cut);
                other.sync_time_property();
                kept.push(other);
            }
        }

        kept.sort_by(|a, b| a.start.total_cmp(&b.start));
        match kind {
            TrackKind::Video => self.document.video_tracks[track_idx].clips = kept,
            TrackKind::Audio => self.document.audio_tracks[track_idx].clips = kept,
        }
        for removed in &removed_for_selection {
            self.prune_clip_selection(removed);
        }
    }

    fn remap_cloned_clip_children(&mut self, clip: &mut SessionClip) {
        let mut obj_remap = BTreeMap::new();
        for obj in &mut clip.objects_3d {
            let new_id = self.alloc_uuid_id();
            obj_remap.insert(obj.id.clone(), new_id.clone());
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
                self.next_id = 1;
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
                self.document.fps = fps;
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
                let current = self.capture_history_snapshot();
                match self.history.undo(current) {
                    Some((label, prev)) => {
                        self.restore_snapshot(prev);
                        Ok(json!({
                            "undone": true,
                            "label": label,
                            "can_undo": self.history.can_undo(),
                            "can_redo": self.history.can_redo(),
                        }))
                    }
                    None => Ok(json!({
                        "undone": false,
                        "can_undo": false,
                        "can_redo": self.history.can_redo(),
                    })),
                }
            }
            Command::Redo => {
                if self.history.is_in_transaction() {
                    return Err(SessionError::History(
                        HistoryError::TransactionAlreadyActive,
                    ));
                }
                let current = self.capture_history_snapshot();
                match self.history.redo(current) {
                    Some((label, next)) => {
                        self.restore_snapshot(next);
                        Ok(json!({
                            "redone": true,
                            "label": label,
                            "can_undo": self.history.can_undo(),
                            "can_redo": self.history.can_redo(),
                        }))
                    }
                    None => Ok(json!({
                        "redone": false,
                        "can_undo": self.history.can_undo(),
                        "can_redo": false,
                    })),
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
                    self.document.video_tracks.remove(idx)
                } else if let Some(idx) = self
                    .document
                    .audio_tracks
                    .iter()
                    .position(|t| t.id == track_id)
                {
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

                let clip_id = self.alloc_uuid_id();
                let clip_name = name.unwrap_or_else(|| kind.clone());
                let fps = self.document.fps.max(1.0);
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
                    properties: BTreeMap::new(),
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
                // Spec 02 §2.6: clamp start >= 0, media time_start >= 0 when adjusting time,
                // and minimum clip length == 1 frame.
                let mut min_delta = -clip.start;
                if adjust_time {
                    min_delta = min_delta.max(-clip.time_start * fps);
                }
                let max_delta = (clip.length - 1.0).max(0.0);
                let applied_delta = delta_frames.clamp(min_delta, max_delta);
                let old_length = clip.length;
                clip.start += applied_delta;
                clip.length -= applied_delta;

                if rate_stretch && old_length > 0.0 {
                    clip.scale_all_keyframes(clip.length / old_length);
                    clip.sync_time_property();
                } else if adjust_time {
                    clip.shift_all_keyframes(-applied_delta);
                    clip.time_start = (clip.time_start + applied_delta / fps).max(0.0);
                    clip.sync_time_property();
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
                let mut clamped_delta = delta_frames;
                if adjust_time && let Some(media_len) = clip.media_length {
                    let max_delta = media_len - clip.time_end * fps;
                    clamped_delta = clamped_delta.min(max_delta);
                }
                let new_length = (old_length + clamped_delta).max(1.0);
                let applied_delta = new_length - old_length;
                clip.length = new_length;

                if rate_stretch && old_length > 0.0 {
                    clip.scale_all_keyframes(new_length / old_length);
                    clip.sync_time_property();
                } else if adjust_time {
                    clip.time_end = (clip.time_end + applied_delta / fps).max(clip.time_start);
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
                let new_obj_id = self.alloc_uuid_id();
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(SessionError::TrackLocked(track.id.clone()));
                }
                let clip = &mut track.clips[clip_idx];
                if let Some(pid) = parent_id.as_deref()
                    && !clip.objects_3d.iter().any(|o| o.id == pid)
                {
                    return Err(SessionError::NotFound(format!(
                        "parent 3D object '{pid}' in clip '{clip_id}'"
                    )));
                }
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

                clip.objects_3d.push(SessionObject3D {
                    id: new_obj_id.clone(),
                    parent_id: parent_id.clone(),
                    kind: kind.clone(),
                    name: obj_name.clone(),
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
                let new_obj_id = self.alloc_uuid_id();
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

                let mut cloned = clip.objects_3d[pos].clone();
                cloned.id = new_obj_id.clone();
                clip.objects_3d.insert(pos + 1, cloned);
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

                // Spec 01 §4.7: reject dropping an ancestor into itself or any of its descendants.
                if let Some(target_parent) = parent_id.as_deref() {
                    if target_parent == object_id {
                        return Err(SessionError::CycleDetected(format!(
                            "cannot parent 3D object '{object_id}' to itself"
                        )));
                    }
                    if !clip.objects_3d.iter().any(|o| o.id == target_parent) {
                        return Err(SessionError::NotFound(format!(
                            "parent 3D object '{target_parent}' in clip '{clip_id}'"
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
                let new_fx_id = self.alloc_uuid_id();
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(SessionError::TrackLocked(track.id.clone()));
                }
                let clip = &mut track.clips[clip_idx];
                let fx_name = name.unwrap_or_else(|| kind.clone());
                let mut properties = BTreeMap::new();
                properties.insert(
                    "enabled".to_string(),
                    SessionProperty::static_value(json!(1)),
                );
                clip.effects.push(SessionEffect {
                    id: new_fx_id.clone(),
                    kind: kind.clone(),
                    name: fx_name.clone(),
                    properties,
                });
                Ok(json!({
                    "clip_id": clip_id,
                    "effect_id": new_fx_id,
                    "kind": kind,
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
                let props = self.document.find_property_map_mut(&target_id)?;
                let entry = props
                    .entry(property.clone())
                    .or_insert_with(|| SessionProperty::static_value(value.clone()));
                if entry.keyframes.is_empty() {
                    entry.keyframes.push(SessionKeyframe {
                        frame: 0.0,
                        value: value.clone(),
                        tween: 1,
                    });
                } else {
                    entry.keyframes[0].value = value.clone();
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
                let props = self.document.find_property_map_mut(&target_id)?;
                let entry = props
                    .entry(property.clone())
                    .or_insert_with(|| SessionProperty::static_value(Value::Null));

                // Spec 01 §4.2 stopwatch behavior.
                if entry.animated {
                    let collapsed_val = entry
                        .keyframes
                        .first()
                        .map_or_else(|| entry.default_value.clone(), |k| k.value.clone());
                    entry.animated = false;
                    entry.expression = None;
                    entry.keyframes = vec![SessionKeyframe {
                        frame: 0.0,
                        value: collapsed_val,
                        tween: 1,
                    }];
                } else {
                    entry.animated = true;
                    if entry.keyframes.len() == 1 {
                        entry.keyframes[0].frame = frame;
                    } else if entry.keyframes.is_empty() {
                        entry.keyframes.push(SessionKeyframe {
                            frame,
                            value: entry.default_value.clone(),
                            tween: 1,
                        });
                    }
                }
                Ok(json!({
                    "target_id": target_id,
                    "property": property,
                    "animated": entry.animated,
                    "keyframes": entry.keyframes.len(),
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
                    entry.keyframes.push(SessionKeyframe {
                        frame,
                        value: value.clone(),
                        tween: resolved_tween,
                    });
                    entry.keyframes.sort_by(|a, b| a.frame.total_cmp(&b.frame));
                }
                Ok(json!({
                    "target_id": target_id,
                    "property": property,
                    "frame": frame,
                    "value": value,
                    "tween": resolved_tween,
                    "keyframe_count": entry.keyframes.len(),
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
                let props = self.document.find_property_map_mut(&target_id)?;
                let entry = props
                    .entry(property.clone())
                    .or_insert_with(|| SessionProperty::static_value(Value::Null));
                let cleaned = expression
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
                if cleaned.is_some() {
                    entry.animated = true;
                }
                entry.expression = cleaned.clone();
                Ok(json!({
                    "target_id": target_id,
                    "property": property,
                    "expression": cleaned,
                }))
            }
            Command::ResetPropertyList { target_id } => {
                let props = self.document.find_property_map_mut(&target_id)?;
                let count = props.len();
                for prop in props.values_mut() {
                    prop.reset();
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
                Ok(json!({
                    "plugin_id": plugin_id,
                    "enabled": self.plugins.is_enabled(&plugin_id),
                    "vanilla_cm3_mode": self.plugins.is_vanilla_cm3_mode(),
                }))
            }
            Command::DisableAllOptionalPlugins => {
                let active_deps = self.active_plugin_dependencies();
                let mut disabled = Vec::new();
                let mut blocked = Vec::new();
                for entry in KNOWN_PLUGINS {
                    if self.plugins.is_enabled(entry.id) {
                        match self.plugins.disable(entry.id, &active_deps) {
                            Ok(_) => disabled.push(entry.id),
                            Err(err) => blocked.push(json!({
                                "plugin_id": entry.id,
                                "reason": err.to_string(),
                            })),
                        }
                    }
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
                let plugins: Vec<Value> = KNOWN_PLUGINS
                    .iter()
                    .map(|p| {
                        json!({
                            "id": p.id,
                            "name": p.name,
                            "category": p.category,
                            "default_enabled": p.default_enabled,
                            "enabled": self.plugins.is_enabled(p.id),
                            "in_use": active_deps.contains(p.id),
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

fn build_object_3d_from_session(
    sobj: &SessionObject3D,
    existing: Option<zc_project::Object3D>,
    children: Vec<zc_project::Object3D>,
) -> zc_project::Object3D {
    let mut obj = existing.unwrap_or_else(|| {
        let kind = match sobj.kind.as_str() {
            "group" => zc_project::Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: Vec::new(),
            },
            "text_3d" => zc_project::Object3DKind::Text3D(Box::default()),
            "light" => zc_project::Object3DKind::Light(Box::new(zc_project::LightData::spot())),
            "camera" => zc_project::Object3DKind::Camera(Box::default()),
            "particles" => zc_project::Object3DKind::Particles(Box::default()),
            "model" => zc_project::Object3DKind::Model(Box::default()),
            "shape.sphere" => zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Sphere {
                radius: 5.0,
                width_segments: 16,
                height_segments: 16,
            }),
            "shape.cylinder" => {
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Cylinder {
                    top_radius: 5.0,
                    bottom_radius: 5.0,
                    height: 10.0,
                    detail: 16,
                    open_ended: false,
                })
            }
            "shape.rectangle" => {
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Rectangle {
                    size: [10.0, 10.0],
                })
            }
            "shape.circle" => zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Circle {
                radius: 5.0,
                detail: 24,
            }),
            "shape.donut" => zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Donut {
                radius: 5.0,
                tube: 2.0,
                radial_segments: 12,
                tubular_segments: 24,
            }),
            _ => zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Box {
                size: [10.0, 10.0, 10.0],
            }),
        };
        let mut created = zc_project::Object3D::new(&sobj.name, kind);
        created.id = parse_uuid_or_new(&sobj.id);
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
    if let zc_project::Object3DKind::Group {
        children: ref mut slot,
        ..
    } = obj.kind
    {
        *slot = children;
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
    depth: usize,
) -> Vec<zc_project::Object3D> {
    if depth > 64 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for sobj in flat {
        if sobj.parent_id.as_deref() == parent_id {
            let children = rebuild_object_tree(flat, Some(&sobj.id), existing, depth + 1);
            let prev = existing.remove(&sobj.id);
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
        let mut created = zc_project::create_effect_by_id(&sfx.kind).unwrap_or_else(|| {
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
        });
        created.id = parse_uuid_or_new(&sfx.id);
        created
    });
    fx.name = sfx.name.clone();
    for (k, prop) in &sfx.properties {
        if k == "enabled" {
            fx.enabled = session_property_to_track(prop, Some(&fx.enabled));
        } else {
            let prev = fx.params.get(k);
            let track = session_property_to_track(prop, prev);
            fx.params.insert(k.clone(), track);
        }
    }
    fx
}

fn sync_video_clip_from_session(
    sclip: &SessionClip,
    existing: Option<zc_project::VideoClip>,
    resolution: [u32; 2],
    fps: u32,
) -> zc_project::VideoClip {
    let mut vc = existing.unwrap_or_else(|| {
        let layer_kind = match sclip.kind.as_str() {
            "video" => zc_project::LayerKind::Video { media: None },
            "adjustment" => zc_project::LayerKind::Adjustment,
            "composite" => zc_project::LayerKind::Composite { layers: Vec::new() },
            "image" => zc_project::LayerKind::Image { image: None },
            "generator" => zc_project::LayerKind::Generator,
            "shape" => zc_project::LayerKind::Shape2D { shapes: Vec::new() },
            "text_2d" => zc_project::LayerKind::Text2D {
                text: "text".to_owned(),
                font: None,
                size: 64,
                shapes: Vec::new(),
            },
            "preset_shape" => zc_project::LayerKind::PresetShape2D {
                preset: zc_project::PresetShapeKind::Rectangle,
                shapes: Vec::new(),
            },
            "scene" => zc_project::LayerKind::Scene3D {
                objects: Vec::new(),
                shadow_mode: zc_project::CompatMode::Current,
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
    vc.layer.resolution = resolution;

    if let Some(prop) = sclip.properties.get("time") {
        vc.time_curve = session_property_to_track(prop, Some(&vc.time_curve));
    } else if !sclip.time_animated {
        vc.time_curve = zc_project::PropertyTrack {
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
        };
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

    if let zc_project::LayerKind::Scene3D { objects, .. } = &mut vc.layer.kind {
        let mut existing_objs = BTreeMap::new();
        collect_existing_objects(std::mem::take(objects), &mut existing_objs);
        *objects = rebuild_object_tree(&sclip.objects_3d, None, &mut existing_objs, 0);
    }

    let mut existing_fx: BTreeMap<String, zc_project::EffectInstance> = vc
        .layer
        .effects
        .drain(..)
        .map(|fx| (fx.id.to_string(), fx))
        .collect();
    vc.layer.effects = sclip
        .effects
        .iter()
        .map(|sfx| sync_effect_from_session(sfx, existing_fx.remove(&sfx.id)))
        .collect();

    vc
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
    }
    if let Some(p) = sclip.properties.get("volume") {
        ac.audio.volume = session_property_to_track(p, Some(&ac.audio.volume));
    }
    if let Some(p) = sclip.properties.get("pan") {
        ac.audio.pan = session_property_to_track(p, Some(&ac.audio.pan));
    }
    ac
}

fn collect_plugin_deps_from_json(value: &Value, deps: &mut BTreeSet<String>) {
    match value {
        Value::String(s) => {
            if let Some(rest) = s.strip_prefix("zoidium:")
                && let Some((plugin_id, _)) = rest.split_once('/')
                && !plugin_id.is_empty()
            {
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
            Some(&json!(45))
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
        // Default project has 1 video track with 1 Scene clip and length 300.
        assert_eq!(session.document().video_tracks.len(), 1);
        assert_eq!(session.document().video_tracks[0].clips.len(), 1);
        assert!((session.document().sequence_length - 300.0).abs() < 1e-9);

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
}
