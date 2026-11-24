pub mod commands;
pub mod control;
pub mod history;
pub mod selection;

use std::collections::{BTreeMap, BTreeSet};

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
    pub properties: BTreeMap<String, SessionProperty>,
    pub objects_3d: Vec<SessionObject3D>,
    pub effects: Vec<SessionEffect>,
}

impl SessionClip {
    fn shift_all_keyframes(&mut self, offset: f64) {
        for prop in self.properties.values_mut() {
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
        for prop in self.properties.values_mut() {
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

impl SessionDocument {
    fn from_project(project: &zc_project::Project) -> Self {
        let mut doc = Self {
            project_name: String::from("Untitled"),
            resolution: [1920, 1080],
            fps: 30.0,
            sequence_length: 300.0,
            video_tracks: Vec::new(),
            audio_tracks: Vec::new(),
            sequence_properties: BTreeMap::new(),
        };

        if let Ok(val) = serde_json::to_value(project) {
            if let Some(name) = val
                .get("name")
                .or_else(|| val.get("project_name"))
                .or_else(|| val.pointer("/manifest/project_name"))
                .and_then(Value::as_str)
            {
                doc.project_name = name.to_string();
            }

            if let Some(seq) = val.get("sequence") {
                if let Some(res) = seq
                    .get("resolution")
                    .or_else(|| seq.pointer("/properties/resolution"))
                    .and_then(Value::as_array)
                    && res.len() == 2
                    && let (Some(w), Some(h)) = (res[0].as_u64(), res[1].as_u64())
                    && w > 0
                    && h > 0
                {
                    doc.resolution = [w.min(7680) as u32, h.min(7680) as u32];
                }
                if let Some(rate) = seq
                    .get("rate")
                    .or_else(|| seq.get("fps"))
                    .or_else(|| seq.pointer("/properties/rate"))
                    .and_then(Value::as_f64)
                    && rate.is_finite()
                    && rate > 0.0
                {
                    doc.fps = rate;
                }
                if let Some(len) = seq
                    .get("length")
                    .or_else(|| seq.get("duration_frames"))
                    .and_then(Value::as_f64)
                    && len.is_finite()
                    && len >= 0.0
                {
                    doc.sequence_length = len;
                }
            }
        }

        doc
    }

    fn all_tracks(&self) -> impl Iterator<Item = &SessionTrack> {
        self.video_tracks.iter().chain(self.audio_tracks.iter())
    }

    fn all_tracks_mut(&mut self) -> impl Iterator<Item = &mut SessionTrack> {
        self.video_tracks
            .iter_mut()
            .chain(self.audio_tracks.iter_mut())
    }

    fn find_track(&self, track_id: &str) -> Option<&SessionTrack> {
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
        let project_bytes = serde_json::to_vec(&self.project).map_or(0, |b| b.len());
        let doc_bytes = serde_json::to_vec(&self.document).map_or(0, |b| b.len());
        let sel_bytes = serde_json::to_vec(&self.selection).map_or(0, |b| b.len());
        project_bytes
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
        let document = SessionDocument::from_project(&project);
        Self {
            project,
            plugins,
            selection: SelectionState::default(),
            playback: PlaybackState::default(),
            history: UndoStack::with_limits(max_entries, max_bytes),
            registry: CommandRegistry::new_default(),
            document,
            next_id: 1,
        }
    }

    /// Builds a session around an existing project and plugin registry.
    #[must_use]
    pub fn from_project(project: zc_project::Project, plugins: zc_plugin::PluginRegistry) -> Self {
        let document = SessionDocument::from_project(&project);
        Self {
            project,
            plugins,
            selection: SelectionState::default(),
            playback: PlaybackState::default(),
            history: UndoStack::new(),
            registry: CommandRegistry::new_default(),
            document,
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

    /// Captures a complete undo snapshot of the current session.
    #[must_use]
    pub fn snapshot(&self) -> SessionSnapshot {
        SessionSnapshot {
            project: self.project.clone(),
            document: self.document.clone(),
            selection: self.selection.clone(),
        }
    }

    /// Begins a live edit transaction (e.g. slider scrub or clip drag) that will commit
    /// as one single undo step on [`Self::commit_transaction`] or roll back cleanly on
    /// [`Self::cancel_transaction`].
    pub fn begin_transaction(&mut self, label: impl Into<String>) -> Result<(), SessionError> {
        let baseline = self.snapshot();
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
                Some(self.snapshot())
            };
            let result = self.execute_inner(command)?;
            if let Some(before_snapshot) = before {
                self.history.push(label, before_snapshot);
            } else {
                self.history.record_transaction_edit();
            }
            self.try_sync_to_project();
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

    fn restore_snapshot(&mut self, snapshot: SessionSnapshot) {
        self.project = snapshot.project;
        self.document = snapshot.document;
        self.selection = snapshot.selection;
        let max_frame = (self.document.sequence_length - 1.0).max(0.0);
        self.playback.current_frame = self.playback.current_frame.clamp(0.0, max_frame);
        self.playback.bump_epoch();
    }

    /// Best-effort synchronization of top-level sequence metadata back into `zc_project::Project`
    /// using its `Serialize`/`Deserialize` representation.
    fn try_sync_to_project(&mut self) {
        let Ok(mut val) = serde_json::to_value(&self.project) else {
            return;
        };

        if let Some(obj) = val.as_object_mut() {
            if obj.contains_key("name") {
                obj.insert(
                    "name".to_string(),
                    Value::String(self.document.project_name.clone()),
                );
            }
            if obj.contains_key("project_name") {
                obj.insert(
                    "project_name".to_string(),
                    Value::String(self.document.project_name.clone()),
                );
            }
            if let Some(manifest) = obj.get_mut("manifest").and_then(Value::as_object_mut)
                && manifest.contains_key("project_name")
            {
                manifest.insert(
                    "project_name".to_string(),
                    Value::String(self.document.project_name.clone()),
                );
            }
            if let Some(seq) = obj.get_mut("sequence").and_then(Value::as_object_mut) {
                if seq.contains_key("resolution") {
                    seq.insert("resolution".to_string(), json!(self.document.resolution));
                }
                if seq.contains_key("rate") {
                    seq.insert("rate".to_string(), json!(self.document.fps));
                } else if seq.contains_key("fps") {
                    seq.insert("fps".to_string(), json!(self.document.fps));
                }
                if seq.contains_key("length") {
                    seq.insert("length".to_string(), json!(self.document.sequence_length));
                } else if seq.contains_key("duration_frames") {
                    seq.insert(
                        "duration_frames".to_string(),
                        json!(self.document.sequence_length),
                    );
                }
            }
        }

        if let Ok(updated) = serde_json::from_value::<zc_project::Project>(val) {
            self.project = updated;
        }
    }

    fn execute_inner(&mut self, command: Command) -> Result<Value, SessionError> {
        match command {
            // --- Project ---
            Command::NewProject => {
                self.project = zc_project::Project::new_cm3_default();
                self.document = SessionDocument::from_project(&self.project);
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
                let current = self.snapshot();
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
                let current = self.snapshot();
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
                if let Some(track) = self.document.find_track(&track_id) {
                    if track.locked {
                        return Err(SessionError::TrackLocked(track_id));
                    }
                } else {
                    return Err(SessionError::NotFound(format!("track '{track_id}'")));
                }

                let clip_id = self.alloc_uuid_id();
                let clip_name = name.unwrap_or_else(|| kind.clone());
                let clip = SessionClip {
                    id: clip_id.clone(),
                    kind: kind.clone(),
                    name: clip_name.clone(),
                    start,
                    length,
                    properties: BTreeMap::new(),
                    objects_3d: Vec::new(),
                    effects: Vec::new(),
                };

                let track = self
                    .document
                    .find_track_mut(&track_id)
                    .ok_or_else(|| SessionError::NotFound(format!("track '{track_id}'")))?;
                track.clips.push(clip);
                track.clips.sort_by(|a, b| a.start.total_cmp(&b.start));
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
                let dest_track = self
                    .document
                    .find_track(&dest_track_id)
                    .ok_or_else(|| SessionError::NotFound(format!("track '{dest_track_id}'")))?;
                if dest_track.locked {
                    return Err(SessionError::TrackLocked(dest_track_id));
                }
                if dest_track.kind != src_kind {
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

                let dest = self
                    .document
                    .find_track_mut(&dest_track_id)
                    .ok_or_else(|| SessionError::NotFound(format!("track '{dest_track_id}'")))?;
                dest.clips.push(clip);
                dest.clips.sort_by(|a, b| a.start.total_cmp(&b.start));
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
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(SessionError::TrackLocked(track.id.clone()));
                }
                let clip = &mut track.clips[clip_idx];
                // Spec 02 §2.6: clamp start >= 0 and minimum clip length == 1 frame.
                let min_delta = -clip.start;
                let max_delta = (clip.length - 1.0).max(0.0);
                let applied_delta = delta_frames.clamp(min_delta, max_delta);
                let old_length = clip.length;
                clip.start += applied_delta;
                clip.length -= applied_delta;

                if rate_stretch && old_length > 0.0 {
                    clip.scale_all_keyframes(clip.length / old_length);
                } else {
                    clip.shift_all_keyframes(-applied_delta);
                }

                let start = clip.start;
                let length = clip.length;
                track.clips.sort_by(|a, b| a.start.total_cmp(&b.start));
                self.playback.bump_epoch();
                Ok(json!({
                    "clip_id": clip_id,
                    "start": start,
                    "length": length,
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
                let (track, clip_idx) = self
                    .document
                    .find_clip_mut(&clip_id)
                    .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
                if track.locked {
                    return Err(SessionError::TrackLocked(track.id.clone()));
                }
                let clip = &mut track.clips[clip_idx];
                let old_length = clip.length;
                let new_length = (old_length + delta_frames).max(1.0);
                clip.length = new_length;

                if rate_stretch && old_length > 0.0 {
                    clip.scale_all_keyframes(new_length / old_length);
                }

                let start = clip.start;
                self.document.recompute_sequence_length_if_extended();
                self.playback.bump_epoch();
                Ok(json!({
                    "clip_id": clip_id,
                    "start": start,
                    "length": new_length,
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

                let mut right_clip = orig.clone();
                let orig_start = orig.start;
                let orig_length = orig.length;
                let right_clip_id = self.alloc_uuid_id();
                right_clip.id = right_clip_id.clone();
                right_clip.start = orig_start + local_offset;
                right_clip.length = orig_length - local_offset;
                // Spec 02 §2.7: shift all dynamic keyframes on right half by -local_offset.
                right_clip.shift_all_keyframes(-local_offset);
                // Reassign distinct IDs to cloned child objects and effects.
                let mut obj_id_remap = BTreeMap::new();
                for obj in &mut right_clip.objects_3d {
                    let new_id = self.alloc_uuid_id();
                    obj_id_remap.insert(obj.id.clone(), new_id.clone());
                    obj.id = new_id;
                }
                for obj in &mut right_clip.objects_3d {
                    if let Some(pid) = obj.parent_id.as_ref()
                        && let Some(mapped) = obj_id_remap.get(pid)
                    {
                        obj.parent_id = Some(mapped.clone());
                    }
                }
                for fx in &mut right_clip.effects {
                    fx.id = self.alloc_uuid_id();
                }

                let track_mut = match kind {
                    TrackKind::Video => &mut self.document.video_tracks[track_idx],
                    TrackKind::Audio => &mut self.document.audio_tracks[track_idx],
                };
                track_mut.clips[clip_idx].length = local_offset;
                let right_start = right_clip.start;
                let right_length = right_clip.length;
                track_mut.clips.insert(clip_idx + 1, right_clip);
                self.playback.bump_epoch();
                Ok(json!({
                    "left_clip_id": clip_id,
                    "left_length": local_offset,
                    "right_clip_id": right_clip_id,
                    "right_start": right_start,
                    "right_length": right_length,
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
                let mut obj_remap = BTreeMap::new();
                for obj in &mut cloned.objects_3d {
                    let new_id = self.alloc_uuid_id();
                    obj_remap.insert(obj.id.clone(), new_id.clone());
                    obj.id = new_id;
                }
                for obj in &mut cloned.objects_3d {
                    if let Some(pid) = obj.parent_id.as_ref()
                        && let Some(mapped) = obj_remap.get(pid)
                    {
                        obj.parent_id = Some(mapped.clone());
                    }
                }
                for fx in &mut cloned.effects {
                    fx.id = self.alloc_uuid_id();
                }

                let new_start = cloned.start;
                let track_mut = match kind {
                    TrackKind::Video => &mut self.document.video_tracks[track_idx],
                    TrackKind::Audio => &mut self.document.audio_tracks[track_idx],
                };
                track_mut.clips.push(cloned);
                track_mut.clips.sort_by(|a, b| a.start.total_cmp(&b.start));
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
            if let Some(req) = map.get("required_plugins").and_then(Value::as_array) {
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
        let right_clip = session.document().video_tracks[0]
            .clips
            .iter()
            .find(|c| c.id == right_id)
            .expect("right clip exists");
        assert!((right_clip.start - 50.0).abs() < 1e-9);
        assert!((right_clip.length - 60.0).abs() < 1e-9);
        let shifted_frame = right_clip.properties["opacity"].keyframes[0].frame;
        assert!((shifted_frame - 20.0).abs() < 1e-9);
    }
}
