use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

use crate::control::ControlError;
use crate::history::{ByteSized, HistoryError};
use crate::selection::{KeyframeRef, SelectionError, SelectionState};

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

/// Builds a [`SessionError::TrackLocked`] error displaying the human-readable track name
/// (falling back to the track ID only when the track has no name).
#[must_use]
pub(crate) fn track_locked_error(name: &str, id: &str) -> SessionError {
    let trimmed = name.trim();
    let label = if trimmed.is_empty() { id } else { trimmed };
    SessionError::TrackLocked(label.to_owned())
}

pub(crate) fn json_value_type_name(val: &Value) -> &'static str {
    match val {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

pub(crate) fn is_valid_namespaced_plugin_effect_id(kind: &str) -> bool {
    let trimmed = kind.trim();
    let lower = trimmed.to_ascii_lowercase();
    let Some(rest) = lower
        .strip_prefix("zoidium:")
        .or_else(|| lower.strip_prefix("zoidcraft:"))
    else {
        return false;
    };
    let Some((ns, effect_id)) = rest.split_once('/') else {
        return false;
    };
    !ns.trim().is_empty() && !effect_id.trim().is_empty()
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

pub(crate) const fn default_control_points() -> [[f64; 2]; 2] {
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
    pub(crate) fn from_prop_value(v: &zc_project::PropertyValue) -> Self {
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
    pub(crate) fn new(frame: f64, value: Value, tween: u16) -> Self {
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
    pub(crate) fn static_value(value: Value) -> Self {
        Self {
            animated: false,
            expression: None,
            default_value: value.clone(),
            keyframes: vec![SessionKeyframe::new(0.0, value, 1)],
        }
    }

    pub(crate) fn static_value_with_kind(value: Value, kind: SessionValueKind) -> Self {
        let mut kf = SessionKeyframe::new(0.0, value.clone(), 1);
        kf.value_kind = Some(kind);
        Self {
            animated: false,
            expression: None,
            default_value: value,
            keyframes: vec![kf],
        }
    }

    pub(crate) fn shift_keyframes(&mut self, offset: f64) {
        for kf in &mut self.keyframes {
            kf.frame += offset;
        }
    }

    pub(crate) fn scale_keyframes(&mut self, scale: f64) {
        for kf in &mut self.keyframes {
            kf.frame *= scale;
            kf.control_points[0][0] *= scale;
            kf.control_points[1][0] *= scale;
        }
    }

    pub(crate) fn reset(&mut self) {
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
    pub(crate) fn eval_time_at(&self, local_frame: f64) -> f64 {
        if self.length <= 0.0 {
            return self.time_start;
        }
        let u = (local_frame / self.length).clamp(0.0, 1.0);
        self.time_start + (self.time_end - self.time_start) * u
    }

    pub(crate) fn sync_time_property(&mut self) {
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

    pub(crate) fn shift_all_keyframes(&mut self, offset: f64) {
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

    pub(crate) fn scale_all_keyframes(&mut self, scale: f64) {
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
    /// Active Precomp+ composition UUID (`None` edits the root `project.sequence`;
    /// `Some(id)` edits `project.compositions[&id]` in place).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_composition_id: Option<zc_project::Uuid>,
    /// Preserves the declaration order and original `SequenceControl::id` for each `ctl_*`
    /// entry in `sequence_properties` so duplicate-named controls never collide or reorder.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sequence_control_order: Vec<(String, String)>,
    /// Child effects of `EffectKind::EffectGroup` instances, indexed by parent `SessionEffect.id`.
    /// Supports arbitrary nesting when a child effect is itself an `EffectGroup`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub effect_group_children: BTreeMap<String, Vec<SessionEffect>>,
    /// Preserved `CustomProperty.metadata` maps (e.g. Shader+ kinds, dropdown option lists,
    /// conditional visibility, and rename rules), indexed by
    /// `target_id -> property_name -> metadata_map`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub property_metadata: BTreeMap<String, BTreeMap<String, BTreeMap<String, Value>>>,
    /// Track-level matte settings (`trackMatte`, `matteType`, `matteInvert`) indexed by `track_id`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub track_matte: BTreeMap<String, BTreeMap<String, Value>>,
}

pub(crate) fn effect_tree_contains(
    group_children: &BTreeMap<String, Vec<SessionEffect>>,
    root_id: &str,
    target_id: &str,
    depth: usize,
) -> bool {
    if depth >= 16 {
        return false;
    }
    let Some(children) = group_children.get(root_id) else {
        return false;
    };
    for child in children {
        if child.id == target_id
            || effect_tree_contains(group_children, &child.id, target_id, depth + 1)
        {
            return true;
        }
    }
    false
}

impl SessionDocument {
    /// Returns the direct child effects of an `EffectGroup` by parent effect ID.
    #[must_use]
    pub fn effect_children(&self, group_effect_id: &str) -> &[SessionEffect] {
        self.effect_group_children
            .get(group_effect_id)
            .map_or(&[], Vec::as_slice)
    }

    /// Finds an effect by ID across both top-level clip stacks and nested `EffectGroup` children.
    #[must_use]
    pub fn find_effect(&self, effect_id: &str) -> Option<&SessionEffect> {
        for track in self.all_tracks() {
            for clip in &track.clips {
                if let Some(fx) = clip.effects.iter().find(|e| e.id == effect_id) {
                    return Some(fx);
                }
            }
        }
        for children in self.effect_group_children.values() {
            if let Some(fx) = children.iter().find(|e| e.id == effect_id) {
                return Some(fx);
            }
        }
        None
    }

    /// Looks up preserved `CustomProperty.metadata` for `(target_id, property)`.
    #[must_use]
    pub fn property_metadata(
        &self,
        target_id: &str,
        property: &str,
    ) -> Option<&BTreeMap<String, Value>> {
        self.property_metadata.get(target_id)?.get(property)
    }

    /// Sets or clears preserved `CustomProperty.metadata` for `(target_id, property)`.
    pub fn set_property_metadata(
        &mut self,
        target_id: impl Into<String>,
        property: impl Into<String>,
        metadata: BTreeMap<String, Value>,
    ) {
        let tid = target_id.into();
        let prop = property.into();
        if metadata.is_empty() {
            if let Some(map) = self.property_metadata.get_mut(&tid) {
                map.remove(&prop);
                if map.is_empty() {
                    self.property_metadata.remove(&tid);
                }
            }
        } else {
            self.property_metadata
                .entry(tid)
                .or_default()
                .insert(prop, metadata);
        }
    }
    pub(crate) fn all_tracks(&self) -> impl Iterator<Item = &SessionTrack> {
        self.video_tracks.iter().chain(self.audio_tracks.iter())
    }

    pub(crate) fn all_tracks_mut(&mut self) -> impl Iterator<Item = &mut SessionTrack> {
        self.video_tracks
            .iter_mut()
            .chain(self.audio_tracks.iter_mut())
    }

    /// Finds a video or audio track by ID.
    #[must_use]
    pub fn find_track(&self, track_id: &str) -> Option<&SessionTrack> {
        self.all_tracks().find(|t| t.id == track_id)
    }

    pub(crate) fn find_track_mut(&mut self, track_id: &str) -> Option<&mut SessionTrack> {
        self.all_tracks_mut().find(|t| t.id == track_id)
    }

    pub(crate) fn find_clip_location(&self, clip_id: &str) -> Option<(TrackKind, usize, usize)> {
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

    pub(crate) fn find_clip_mut(&mut self, clip_id: &str) -> Option<(&mut SessionTrack, usize)> {
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

    pub(crate) fn find_property_map_mut(
        &mut self,
        target_id: &str,
    ) -> Result<&mut BTreeMap<String, SessionProperty>, SessionError> {
        if target_id == "sequence" {
            return Ok(&mut self.sequence_properties);
        }

        if self
            .effect_group_children
            .values()
            .any(|children| children.iter().any(|e| e.id == target_id))
        {
            let mut owning_locked: Option<(String, String)> = None;
            let mut found_in_group = false;
            for track in self.video_tracks.iter().chain(self.audio_tracks.iter()) {
                for clip in &track.clips {
                    if clip.effects.iter().any(|fx| {
                        effect_tree_contains(&self.effect_group_children, &fx.id, target_id, 0)
                    }) {
                        found_in_group = true;
                        if track.locked {
                            owning_locked = Some((track.name.clone(), track.id.clone()));
                        }
                        break;
                    }
                }
                if found_in_group {
                    break;
                }
            }
            if let Some((tname, tid)) = owning_locked {
                return Err(track_locked_error(&tname, &tid));
            }
            for children in self.effect_group_children.values_mut() {
                if let Some(fx) = children.iter_mut().find(|e| e.id == target_id) {
                    return Ok(&mut fx.properties);
                }
            }
        }

        for track in self
            .video_tracks
            .iter_mut()
            .chain(self.audio_tracks.iter_mut())
        {
            for clip in &mut track.clips {
                if clip.id == target_id {
                    if track.locked {
                        return Err(track_locked_error(&track.name, &track.id));
                    }
                    return Ok(&mut clip.properties);
                }
                if let Some(obj) = clip.objects_3d.iter_mut().find(|o| o.id == target_id) {
                    if track.locked {
                        return Err(track_locked_error(&track.name, &track.id));
                    }
                    return Ok(&mut obj.properties);
                }
                if let Some(fx) = clip.effects.iter_mut().find(|e| e.id == target_id) {
                    if track.locked {
                        return Err(track_locked_error(&track.name, &track.id));
                    }
                    return Ok(&mut fx.properties);
                }
            }
        }

        Err(SessionError::NotFound(format!("target '{target_id}'")))
    }

    pub(crate) fn recompute_sequence_length_if_extended(&mut self) {
        let max_end = self
            .all_tracks()
            .flat_map(|t| t.clips.iter())
            .map(|c| c.start + c.length)
            .fold(0.0_f64, f64::max);
        if max_end > self.sequence_length {
            self.sequence_length = max_end;
        }
    }
}

/// Atomic snapshot stored inside [`UndoStack`] so Undo/Redo restores both project data
/// and editor selection together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub project: zc_project::Project,
    pub document: SessionDocument,
    pub selection: SelectionState,
    /// Enabled plugin IDs captured atomically with this snapshot so Undo/Redo restores
    /// plugin auto-enablement (e.g. `precomp-plus` on `PrecomposeClips`/`CreateComposition`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) enabled_plugins: Option<BTreeSet<String>>,
    /// Content-addressed SHA-256 blob references (`asset_key -> content_sha256`) for
    /// assets whose binary buffers were deduplicated out of `project.assets` into
    /// `Session.asset_blobs`. Assets absent from this map had `bytes == None` at
    /// snapshot time and remain `None` on restore.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) asset_blob_refs: BTreeMap<String, String>,
}

fn estimate_json_value_bytes(v: &Value) -> usize {
    let base = std::mem::size_of::<Value>();
    match v {
        Value::Null | Value::Bool(_) | Value::Number(_) => base,
        Value::String(s) => base.saturating_add(s.len()),
        Value::Array(items) => items.iter().fold(base, |acc, item| {
            acc.saturating_add(estimate_json_value_bytes(item))
        }),
        Value::Object(map) => map.iter().fold(base, |acc, (k, val)| {
            acc.saturating_add(k.len())
                .saturating_add(estimate_json_value_bytes(val))
                .saturating_add(32)
        }),
    }
}

fn estimate_preserved_bytes(map: &BTreeMap<String, Value>) -> usize {
    map.iter().fold(0usize, |acc, (k, v)| {
        acc.saturating_add(k.len())
            .saturating_add(estimate_json_value_bytes(v))
            .saturating_add(48)
    })
}

fn estimate_prop_value_bytes(v: &zc_project::PropertyValue) -> usize {
    let base = std::mem::size_of::<zc_project::PropertyValue>();
    match v {
        zc_project::PropertyValue::Text(s)
        | zc_project::PropertyValue::List(s)
        | zc_project::PropertyValue::Shader(s) => base.saturating_add(s.len()),
        zc_project::PropertyValue::Asset(opt) => {
            base.saturating_add(opt.as_ref().map_or(0, String::len))
        }
        zc_project::PropertyValue::Gradient(stops) => base.saturating_add(
            stops
                .len()
                .saturating_mul(std::mem::size_of::<zc_project::GradientStop>()),
        ),
        zc_project::PropertyValue::Curve(val) | zc_project::PropertyValue::Json(val) => {
            base.saturating_add(estimate_json_value_bytes(val))
        }
        _ => base,
    }
}

fn estimate_prop_track_bytes(track: &zc_project::PropertyTrack) -> usize {
    let mut total = std::mem::size_of::<zc_project::PropertyTrack>()
        .saturating_add(track.expression.as_ref().map_or(0, String::len))
        .saturating_add(estimate_prop_value_bytes(&track.value));
    for kf in &track.keyframes {
        total = total
            .saturating_add(std::mem::size_of::<zc_project::PropertyKeyframe>())
            .saturating_add(estimate_prop_value_bytes(&kf.value));
    }
    for ch in &track.channels {
        total = total.saturating_add(estimate_prop_track_bytes(ch));
    }
    total
}

fn estimate_custom_properties_bytes(props: &[zc_project::CustomProperty]) -> usize {
    props.iter().fold(0usize, |acc, cp| {
        acc.saturating_add(std::mem::size_of::<zc_project::CustomProperty>())
            .saturating_add(cp.name.len())
            .saturating_add(estimate_prop_track_bytes(&cp.track))
            .saturating_add(estimate_preserved_bytes(&cp.metadata))
    })
}

fn estimate_material_bytes(mat: &zc_project::MaterialSpec) -> usize {
    let base = std::mem::size_of::<zc_project::MaterialSpec>();
    match mat {
        zc_project::MaterialSpec::SingleColor { color, .. } => {
            base.saturating_add(estimate_prop_track_bytes(color))
        }
        zc_project::MaterialSpec::Texture(t) => base
            .saturating_add(std::mem::size_of::<zc_project::TextureMaterial>())
            .saturating_add(t.texture.as_ref().map_or(0, String::len))
            .saturating_add(estimate_prop_track_bytes(&t.repeat))
            .saturating_add(estimate_prop_track_bytes(&t.offset))
            .saturating_add(estimate_prop_track_bytes(&t.center))
            .saturating_add(estimate_prop_track_bytes(&t.rotation)),
        zc_project::MaterialSpec::Video(v) => base
            .saturating_add(std::mem::size_of::<zc_project::VideoMaterial>())
            .saturating_add(v.video.as_ref().map_or(0, String::len))
            .saturating_add(estimate_prop_track_bytes(&v.color))
            .saturating_add(estimate_prop_track_bytes(&v.opacity)),
        zc_project::MaterialSpec::CustomPbr(p) => base
            .saturating_add(std::mem::size_of::<zc_project::CustomPbrMaterial>())
            .saturating_add(p.texture.as_ref().map_or(0, String::len))
            .saturating_add(p.normal_map.as_ref().map_or(0, String::len))
            .saturating_add(estimate_prop_track_bytes(&p.color))
            .saturating_add(estimate_prop_track_bytes(&p.emissive))
            .saturating_add(estimate_prop_track_bytes(&p.roughness))
            .saturating_add(estimate_prop_track_bytes(&p.metalness))
            .saturating_add(estimate_prop_track_bytes(&p.opacity))
            .saturating_add(estimate_prop_track_bytes(&p.repeat))
            .saturating_add(estimate_prop_track_bytes(&p.offset))
            .saturating_add(estimate_prop_track_bytes(&p.center))
            .saturating_add(estimate_prop_track_bytes(&p.rotation)),
        zc_project::MaterialSpec::CustomShader {
            frag_shader,
            custom_properties,
        } => base
            .saturating_add(frag_shader.len())
            .saturating_add(estimate_custom_properties_bytes(custom_properties)),
        zc_project::MaterialSpec::PbrPlus(p) => base
            .saturating_add(std::mem::size_of::<zc_project::PbrPlusMaterial>())
            .saturating_add(p.base_color_map.as_ref().map_or(0, String::len))
            .saturating_add(p.emissive_map.as_ref().map_or(0, String::len))
            .saturating_add(p.roughness_map.as_ref().map_or(0, String::len))
            .saturating_add(p.metalness_map.as_ref().map_or(0, String::len))
            .saturating_add(p.normal_map.as_ref().map_or(0, String::len))
            .saturating_add(p.alpha_map.as_ref().map_or(0, String::len))
            .saturating_add(p.ao_texture.as_ref().map_or(0, String::len))
            .saturating_add(estimate_prop_track_bytes(&p.base_color))
            .saturating_add(estimate_prop_track_bytes(&p.emissive))
            .saturating_add(estimate_prop_track_bytes(&p.roughness))
            .saturating_add(estimate_prop_track_bytes(&p.metalness))
            .saturating_add(estimate_prop_track_bytes(&p.opacity))
            .saturating_add(estimate_prop_track_bytes(&p.emissive_intensity))
            .saturating_add(estimate_prop_track_bytes(&p.repeat))
            .saturating_add(estimate_prop_track_bytes(&p.offset))
            .saturating_add(estimate_prop_track_bytes(&p.center))
            .saturating_add(estimate_prop_track_bytes(&p.rotation))
            .saturating_add(
                p.normal_scale_track
                    .as_ref()
                    .map_or(0, estimate_prop_track_bytes),
            )
            .saturating_add(
                p.reflection_intensity_track
                    .as_ref()
                    .map_or(0, estimate_prop_track_bytes),
            )
            .saturating_add(estimate_preserved_bytes(&p.preserved)),
        zc_project::MaterialSpec::Matcap(m) => base
            .saturating_add(std::mem::size_of::<zc_project::MatcapMaterial>())
            .saturating_add(m.texture.as_ref().map_or(0, String::len))
            .saturating_add(estimate_prop_track_bytes(&m.color))
            .saturating_add(estimate_prop_track_bytes(&m.brightness))
            .saturating_add(estimate_prop_track_bytes(&m.opacity)),
        zc_project::MaterialSpec::ImagePlus(i) => base
            .saturating_add(std::mem::size_of::<zc_project::ImagePlusMaterial>())
            .saturating_add(i.texture.as_ref().map_or(0, String::len))
            .saturating_add(estimate_prop_track_bytes(&i.opacity))
            .saturating_add(estimate_prop_track_bytes(&i.repeat))
            .saturating_add(estimate_prop_track_bytes(&i.offset))
            .saturating_add(estimate_prop_track_bytes(&i.center))
            .saturating_add(estimate_prop_track_bytes(&i.rotation)),
        zc_project::MaterialSpec::NodeMaterial { graph, .. } => {
            base.saturating_add(estimate_json_value_bytes(graph))
        }
        zc_project::MaterialSpec::Unknown {
            material_type,
            payload,
        } => base
            .saturating_add(material_type.len())
            .saturating_add(estimate_json_value_bytes(payload)),
    }
}

fn estimate_effect_bytes(fx: &zc_project::EffectInstance) -> usize {
    let mut total = std::mem::size_of::<zc_project::EffectInstance>()
        .saturating_add(fx.name.len())
        .saturating_add(estimate_prop_track_bytes(&fx.enabled))
        .saturating_add(estimate_preserved_bytes(&fx.preserved));
    for (k, v) in &fx.params {
        total = total
            .saturating_add(k.len())
            .saturating_add(estimate_prop_track_bytes(v))
            .saturating_add(48);
    }
    match &fx.kind {
        zc_project::EffectKind::Standard(id) => total.saturating_add(id.len()),
        zc_project::EffectKind::Plugin {
            plugin_id,
            effect_id,
        } => total
            .saturating_add(plugin_id.len())
            .saturating_add(effect_id.len()),
        zc_project::EffectKind::CustomShader {
            frag_shader,
            custom_properties,
        } => total
            .saturating_add(frag_shader.len())
            .saturating_add(estimate_custom_properties_bytes(custom_properties)),
        zc_project::EffectKind::EffectGroup {
            custom_properties,
            children,
        } => children.iter().fold(
            total.saturating_add(estimate_custom_properties_bytes(custom_properties)),
            |acc, child| acc.saturating_add(estimate_effect_bytes(child)),
        ),
    }
}

fn estimate_object3d_bytes(obj: &zc_project::Object3D) -> usize {
    let mut total = std::mem::size_of::<zc_project::Object3D>()
        .saturating_add(obj.name.len())
        .saturating_add(estimate_prop_track_bytes(&obj.enabled))
        .saturating_add(estimate_prop_track_bytes(&obj.position))
        .saturating_add(estimate_prop_track_bytes(&obj.rotation))
        .saturating_add(estimate_prop_track_bytes(&obj.scale))
        .saturating_add(estimate_preserved_bytes(&obj.preserved));
    if let Some(mat) = &obj.material {
        total = total.saturating_add(estimate_material_bytes(mat));
    }
    for (k, mat) in &obj.material_slots {
        total = total
            .saturating_add(k.len())
            .saturating_add(estimate_material_bytes(mat))
            .saturating_add(48);
    }
    if let Some(tt) = &obj.constraints.track_to {
        total = total
            .saturating_add(tt.target_id.len())
            .saturating_add(estimate_prop_track_bytes(&tt.influence));
    }
    if let Some(fl) = &obj.constraints.follow {
        total = total
            .saturating_add(fl.target_id.len())
            .saturating_add(estimate_prop_track_bytes(&fl.position_offset))
            .saturating_add(estimate_prop_track_bytes(&fl.rotation_offset))
            .saturating_add(estimate_prop_track_bytes(&fl.scale_offset));
    }
    match &obj.kind {
        zc_project::Object3DKind::Group {
            custom_properties,
            children,
            ..
        } => {
            total = total.saturating_add(estimate_custom_properties_bytes(custom_properties));
            for child in children {
                total = total.saturating_add(estimate_object3d_bytes(child));
            }
        }
        zc_project::Object3DKind::Text3D(t) => {
            total = total
                .saturating_add(std::mem::size_of::<zc_project::Text3DData>())
                .saturating_add(t.text.len())
                .saturating_add(t.font.as_ref().map_or(0, String::len));
        }
        zc_project::Object3DKind::Light(l) => {
            total = total
                .saturating_add(std::mem::size_of::<zc_project::LightData>())
                .saturating_add(estimate_prop_track_bytes(&l.color))
                .saturating_add(estimate_prop_track_bytes(&l.ground_color))
                .saturating_add(estimate_prop_track_bytes(&l.target))
                .saturating_add(estimate_prop_track_bytes(&l.intensity))
                .saturating_add(estimate_prop_track_bytes(&l.angle));
        }
        zc_project::Object3DKind::Camera(c) => {
            total = total
                .saturating_add(std::mem::size_of::<zc_project::CameraData>())
                .saturating_add(estimate_prop_track_bytes(&c.fov))
                .saturating_add(estimate_prop_track_bytes(&c.zoom))
                .saturating_add(estimate_prop_track_bytes(&c.shake_amplitude))
                .saturating_add(estimate_prop_track_bytes(&c.shake_speed))
                .saturating_add(estimate_prop_track_bytes(&c.focal_length))
                .saturating_add(estimate_prop_track_bytes(&c.film_offset_x))
                .saturating_add(estimate_prop_track_bytes(&c.film_offset_y));
        }
        zc_project::Object3DKind::Particles(p) => {
            total = total
                .saturating_add(std::mem::size_of::<zc_project::ParticlesData>())
                .saturating_add(p.sprite_preset.as_ref().map_or(0, String::len))
                .saturating_add(estimate_prop_track_bytes(&p.time));
        }
        zc_project::Object3DKind::Model(m) => {
            total = total
                .saturating_add(std::mem::size_of::<zc_project::ModelData>())
                .saturating_add(m.asset_id.as_ref().map_or(0, String::len));
            for (k, v) in &m.model_properties {
                total = total
                    .saturating_add(k.len())
                    .saturating_add(estimate_prop_track_bytes(v))
                    .saturating_add(48);
            }
        }
        zc_project::Object3DKind::Dummy {
            original_type,
            payload,
        } => {
            total = total
                .saturating_add(original_type.as_ref().map_or(0, String::len))
                .saturating_add(estimate_json_value_bytes(payload));
        }
        zc_project::Object3DKind::Shape3D(_) => {}
    }
    total
}

fn estimate_layer_bytes(layer: &zc_project::Layer) -> usize {
    let mut total = std::mem::size_of::<zc_project::Layer>()
        .saturating_add(layer.name.len())
        .saturating_add(layer.blending.len())
        .saturating_add(estimate_prop_track_bytes(&layer.position))
        .saturating_add(estimate_prop_track_bytes(&layer.scale))
        .saturating_add(estimate_prop_track_bytes(&layer.rotation))
        .saturating_add(estimate_prop_track_bytes(&layer.anchor))
        .saturating_add(estimate_prop_track_bytes(&layer.opacity))
        .saturating_add(estimate_preserved_bytes(&layer.preserved));
    for fx in &layer.effects {
        total = total.saturating_add(estimate_effect_bytes(fx));
    }
    match &layer.kind {
        zc_project::LayerKind::Scene3D { objects, .. } => {
            for obj in objects {
                total = total.saturating_add(estimate_object3d_bytes(obj));
            }
        }
        zc_project::LayerKind::Composite { layers } => {
            for child in layers {
                total = total.saturating_add(estimate_layer_bytes(child));
            }
        }
        zc_project::LayerKind::Image { image } => {
            total = total.saturating_add(image.as_ref().map_or(0, String::len));
        }
        zc_project::LayerKind::Video { media } => {
            total = total.saturating_add(media.as_ref().map_or(0, String::len));
        }
        zc_project::LayerKind::Text2D {
            text, font, shapes, ..
        } => {
            total = total
                .saturating_add(text.len())
                .saturating_add(font.as_ref().map_or(0, String::len));
            for shape in shapes {
                total = total
                    .saturating_add(std::mem::size_of::<zc_project::Shape2DNode>())
                    .saturating_add(estimate_preserved_bytes(&shape.preserved));
            }
        }
        zc_project::LayerKind::Shape2D { shapes }
        | zc_project::LayerKind::PresetShape2D { shapes, .. } => {
            for shape in shapes {
                total = total
                    .saturating_add(std::mem::size_of::<zc_project::Shape2DNode>())
                    .saturating_add(estimate_preserved_bytes(&shape.preserved));
            }
        }
        zc_project::LayerKind::Plugin {
            plugin_type,
            payload,
        } => {
            total = total
                .saturating_add(plugin_type.len())
                .saturating_add(estimate_json_value_bytes(payload));
        }
        _ => {}
    }
    total
}

fn estimate_sequence_bytes(seq: &zc_project::Sequence) -> usize {
    let mut total = std::mem::size_of::<zc_project::Sequence>()
        .saturating_add(seq.name.len())
        .saturating_add(estimate_prop_track_bytes(&seq.motion_blur.samples))
        .saturating_add(estimate_prop_track_bytes(&seq.motion_blur.shutter))
        .saturating_add(estimate_preserved_bytes(&seq.preserved));
    for m in &seq.markers {
        total = total
            .saturating_add(std::mem::size_of::<zc_project::Marker>())
            .saturating_add(m.label.len());
    }
    for ctl in &seq.controls {
        total = total
            .saturating_add(std::mem::size_of::<zc_project::SequenceControl>())
            .saturating_add(ctl.id.len())
            .saturating_add(ctl.name.len())
            .saturating_add(estimate_prop_track_bytes(&ctl.track))
            .saturating_add(estimate_preserved_bytes(&ctl.preserved));
    }
    for vt in &seq.video_tracks {
        total = total
            .saturating_add(std::mem::size_of::<zc_project::VideoTrack>())
            .saturating_add(vt.name.len())
            .saturating_add(estimate_preserved_bytes(&vt.preserved));
        for vc in &vt.clips {
            total = total
                .saturating_add(std::mem::size_of::<zc_project::VideoClip>())
                .saturating_add(vc.name.len())
                .saturating_add(vc.media.as_ref().map_or(0, String::len))
                .saturating_add(estimate_prop_track_bytes(&vc.time_curve))
                .saturating_add(estimate_layer_bytes(&vc.layer))
                .saturating_add(estimate_preserved_bytes(&vc.preserved));
        }
    }
    for at in &seq.audio_tracks {
        total = total
            .saturating_add(std::mem::size_of::<zc_project::AudioTrack>())
            .saturating_add(at.name.len())
            .saturating_add(estimate_preserved_bytes(&at.preserved));
        for ac in &at.clips {
            total = total
                .saturating_add(std::mem::size_of::<zc_project::AudioClip>())
                .saturating_add(ac.name.len())
                .saturating_add(ac.media.as_ref().map_or(0, String::len))
                .saturating_add(estimate_prop_track_bytes(&ac.time_curve))
                .saturating_add(estimate_prop_track_bytes(&ac.audio.volume))
                .saturating_add(estimate_prop_track_bytes(&ac.audio.pan))
                .saturating_add(estimate_preserved_bytes(&ac.preserved));
        }
    }
    for masks in seq.layer_masks.values() {
        for mask in masks {
            total = total
                .saturating_add(std::mem::size_of::<zc_project::LayerMask>())
                .saturating_add(mask.id.len())
                .saturating_add(mask.name.len())
                .saturating_add(estimate_preserved_bytes(&mask.preserved));
        }
    }
    total =
        total
            .saturating_add(seq.scene_backdrops.len().saturating_mul(
                std::mem::size_of::<zc_project::SceneBackdrop>().saturating_add(32),
            ))
            .saturating_add(seq.layer_3d_props.len().saturating_mul(
                std::mem::size_of::<zc_project::Layer3DProps>().saturating_add(32),
            ));
    for cam_objs in seq.camera_layer_objects.values() {
        for obj in cam_objs {
            total = total.saturating_add(estimate_object3d_bytes(obj));
        }
    }
    total = total
        .saturating_add(seq.bypassed_effects.len().saturating_mul(32))
        .saturating_add(seq.precomp_offsets.len().saturating_mul(48))
        .saturating_add(
            seq.clip_links
                .values()
                .map(|v| v.len().saturating_mul(16).saturating_add(32))
                .fold(0usize, usize::saturating_add),
        );
    total
}

fn estimate_project_bytes(project: &zc_project::Project) -> usize {
    let mut total = std::mem::size_of::<zc_project::Project>()
        .saturating_add(project.name.len())
        .saturating_add(estimate_sequence_bytes(&project.sequence))
        .saturating_add(estimate_preserved_bytes(&project.preserved));
    for (k, record) in &project.assets {
        total = total
            .saturating_add(std::mem::size_of::<zc_project::AssetRecord>())
            .saturating_add(k.len())
            .saturating_add(record.id.len())
            .saturating_add(record.filename.len())
            .saturating_add(record.url.as_ref().map_or(0, String::len))
            .saturating_add(record.sha256.len())
            .saturating_add(record.bytes.as_ref().map_or(0, Vec::len))
            .saturating_add(estimate_preserved_bytes(&record.preserved))
            .saturating_add(48);
    }
    for media in &project.media {
        total = total
            .saturating_add(std::mem::size_of::<zc_project::MediaItem>())
            .saturating_add(media.name.len())
            .saturating_add(media.asset_id.as_ref().map_or(0, String::len))
            .saturating_add(estimate_preserved_bytes(&media.preserved));
    }
    for comp in project.compositions.values() {
        total = total
            .saturating_add(estimate_sequence_bytes(comp))
            .saturating_add(48);
    }
    for req in &project.plugin_requirements {
        total = total.saturating_add(req.len()).saturating_add(32);
    }
    total
}

fn estimate_session_property_bytes(prop: &SessionProperty) -> usize {
    let mut total = std::mem::size_of::<SessionProperty>()
        .saturating_add(prop.expression.as_ref().map_or(0, String::len))
        .saturating_add(estimate_json_value_bytes(&prop.default_value));
    for kf in &prop.keyframes {
        total = total
            .saturating_add(std::mem::size_of::<SessionKeyframe>())
            .saturating_add(estimate_json_value_bytes(&kf.value));
    }
    total
}

fn estimate_session_document_bytes(doc: &SessionDocument) -> usize {
    let mut total = std::mem::size_of::<SessionDocument>().saturating_add(doc.project_name.len());
    for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
        total = total
            .saturating_add(std::mem::size_of::<SessionTrack>())
            .saturating_add(track.id.len())
            .saturating_add(track.name.len());
        for clip in &track.clips {
            total = total
                .saturating_add(std::mem::size_of::<SessionClip>())
                .saturating_add(clip.id.len())
                .saturating_add(clip.kind.len())
                .saturating_add(clip.name.len())
                .saturating_add(clip.source_clip_id.as_ref().map_or(0, String::len));
            for (k, p) in &clip.properties {
                total = total
                    .saturating_add(k.len())
                    .saturating_add(estimate_session_property_bytes(p))
                    .saturating_add(48);
            }
            for obj in &clip.objects_3d {
                total = total
                    .saturating_add(std::mem::size_of::<SessionObject3D>())
                    .saturating_add(obj.id.len())
                    .saturating_add(obj.kind.len())
                    .saturating_add(obj.name.len())
                    .saturating_add(obj.parent_id.as_ref().map_or(0, String::len))
                    .saturating_add(obj.source_id.as_ref().map_or(0, String::len));
                for (k, p) in &obj.properties {
                    total = total
                        .saturating_add(k.len())
                        .saturating_add(estimate_session_property_bytes(p))
                        .saturating_add(48);
                }
            }
            for fx in &clip.effects {
                total = total
                    .saturating_add(std::mem::size_of::<SessionEffect>())
                    .saturating_add(fx.id.len())
                    .saturating_add(fx.kind.len())
                    .saturating_add(fx.name.len())
                    .saturating_add(fx.source_id.as_ref().map_or(0, String::len));
                for (k, p) in &fx.properties {
                    total = total
                        .saturating_add(k.len())
                        .saturating_add(estimate_session_property_bytes(p))
                        .saturating_add(48);
                }
            }
        }
    }
    for (k, p) in &doc.sequence_properties {
        total = total
            .saturating_add(k.len())
            .saturating_add(estimate_session_property_bytes(p))
            .saturating_add(48);
    }
    for (k, id) in &doc.sequence_control_order {
        total = total
            .saturating_add(k.len())
            .saturating_add(id.len())
            .saturating_add(48);
    }
    for (parent_id, children) in &doc.effect_group_children {
        total = total.saturating_add(parent_id.len()).saturating_add(48);
        for fx in children {
            total = total
                .saturating_add(std::mem::size_of::<SessionEffect>())
                .saturating_add(fx.id.len())
                .saturating_add(fx.kind.len())
                .saturating_add(fx.name.len())
                .saturating_add(fx.source_id.as_ref().map_or(0, String::len));
            for (k, p) in &fx.properties {
                total = total
                    .saturating_add(k.len())
                    .saturating_add(estimate_session_property_bytes(p))
                    .saturating_add(48);
            }
        }
    }
    for (tid, props) in &doc.property_metadata {
        total = total.saturating_add(tid.len()).saturating_add(48);
        for (pk, meta) in props {
            total = total
                .saturating_add(pk.len())
                .saturating_add(estimate_preserved_bytes(meta))
                .saturating_add(48);
        }
    }
    for (tid, map) in &doc.track_matte {
        total = total
            .saturating_add(tid.len())
            .saturating_add(estimate_preserved_bytes(map))
            .saturating_add(48);
    }
    total
}

fn estimate_selection_bytes(sel: &SelectionState) -> usize {
    let mut total = std::mem::size_of::<SelectionState>()
        .saturating_add(sel.clip_base_type.as_ref().map_or(0, String::len))
        .saturating_add(sel.object_3d_base_type.as_ref().map_or(0, String::len))
        .saturating_add(sel.effect_base_type.as_ref().map_or(0, String::len));
    for id in sel
        .clip_ids
        .iter()
        .chain(sel.object_3d_ids.iter())
        .chain(sel.effect_ids.iter())
    {
        total = total
            .saturating_add(std::mem::size_of::<String>())
            .saturating_add(id.len());
    }
    for kf in &sel.keyframes {
        total = total
            .saturating_add(std::mem::size_of::<KeyframeRef>())
            .saturating_add(kf.target_id.len())
            .saturating_add(kf.property.len())
            .saturating_add(kf.channel.as_ref().map_or(0, String::len));
    }
    total
}

impl ByteSized for SessionSnapshot {
    fn estimated_bytes(&self) -> usize {
        let project_bytes = estimate_project_bytes(&self.project);
        let doc_bytes = estimate_session_document_bytes(&self.document);
        let sel_bytes = estimate_selection_bytes(&self.selection);
        let plugin_bytes = self.enabled_plugins.as_ref().map_or(0, |set| {
            set.iter()
                .map(|s| s.len().saturating_add(32))
                .fold(0usize, usize::saturating_add)
        });
        let ref_bytes = self
            .asset_blob_refs
            .iter()
            .map(|(k, v)| k.len().saturating_add(v.len()).saturating_add(48))
            .fold(0usize, usize::saturating_add);
        project_bytes
            .saturating_add(doc_bytes)
            .saturating_add(sel_bytes)
            .saturating_add(plugin_bytes)
            .saturating_add(ref_bytes)
    }
}
