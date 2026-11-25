#![forbid(unsafe_code)]

pub mod commands;
pub mod control;
mod document;
mod execute;
pub mod history;
mod plugin_defaults;
mod plugin_deps;
mod pro_commands;
mod property_conv;
mod pz_clipboard;
pub mod selection;
mod sync_from_project;
mod sync_objects_to_project;
mod sync_to_project;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::{Value, json};

pub use commands::{ClipSnapshot, Command, CommandMeta, CommandRegistry};
pub use control::{
    ControlError, ControlRequest, ControlResponse, ControlTarget, dispatch_json_line,
    dispatch_json_line_str, dispatch_json_lines, dispatch_request,
};
pub use document::{
    SessionClip, SessionDocument, SessionEffect, SessionError, SessionKeyframe, SessionObject3D,
    SessionProperty, SessionSnapshot, SessionTrack, SessionValueKind, TrackKind,
};
pub use history::{
    ByteSized, DEFAULT_MAX_BYTES, DEFAULT_MAX_ENTRIES, HistoryEntry, HistoryError, UndoStack,
};
pub use selection::{
    KeyframeRef, PlaybackState, SelectionDomain, SelectionError, SelectionState,
    TIME_FORMAT_FRAME_NUMBER, TIME_FORMAT_TIMECODE_FRAMES, TIME_FORMAT_TIMECODE_MS,
};

pub(crate) use document::track_locked_error;
use plugin_deps::collect_plugin_deps_from_project;
pub(crate) use property_conv::{
    eval_session_property_at, prop_f64_static, prop_str_static, property_track_to_session,
    session_property_to_track,
};
pub(crate) use sync_from_project::parse_uuid_or_new;
pub(crate) use sync_to_project::regenerate_cloned_layer_ids;

fn is_valid_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// O(1) edge/midpoint byte sample check (at most 192 bytes compared) to detect direct
/// same-length in-place mutations of `AssetRecord::bytes` without hashing multi-megabyte media.
fn bytes_edge_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    const SAMPLE: usize = 64;
    if a.len() <= SAMPLE * 3 {
        return a == b;
    }
    let mid = (a.len() / 2).saturating_sub(SAMPLE / 2);
    let tail = a.len() - SAMPLE;
    a[..SAMPLE] == b[..SAMPLE]
        && a[mid..mid + SAMPLE] == b[mid..mid + SAMPLE]
        && a[tail..] == b[tail..]
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

pub(crate) fn seed_id_allocator(
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
    pub(crate) document: SessionDocument,
    /// Shared immutable asset binary blobs deduplicated outside undo snapshots so
    /// undo history never duplicates multi-megabyte embedded media in memory.
    pub(crate) asset_blobs: BTreeMap<String, Arc<Vec<u8>>>,
    /// Set of UUIDs already present in the project or allocated by this session so
    /// reopening a saved project can never collide with existing IDs.
    pub(crate) used_ids: BTreeSet<String>,
    /// Plugin IDs most recently derived from `self.document` so deleting plugin-backed
    /// items releases those document-derived requirements without dropping explicit external locks.
    pub(crate) prev_doc_plugin_deps: BTreeSet<String>,
    /// Explicit `project.plugin_requirements` present at load time that were not derived from
    /// `self.document`, preserved across document edits and undo/redo.
    pub(crate) external_plugin_requirements: BTreeSet<String>,
    /// Monotonic counter bumped on non-history session state mutations (e.g. plugin toggles).
    pub(crate) plugin_revision: u64,
    /// Keyframe shifts to apply when cloning `layer_masks` for newly split clips on sync.
    pub(crate) pending_clip_side_table_shifts: BTreeMap<String, f64>,
    /// Set by command handlers that mutate `self.project` directly (and rebuild `self.document`
    /// from `self.project` if needed) so `Session::execute` can skip a redundant
    /// `sync_document_to_project()` pass.
    pub(crate) doc_already_synced_with_project: bool,
    pub(crate) next_id: u64,
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
            pending_clip_side_table_shifts: BTreeMap::new(),
            doc_already_synced_with_project: false,
            next_id,
        }
    }

    /// Returns the currently active [`zc_project::Sequence`] (root `project.sequence` when
    /// `active_composition_id` is `None`, or `project.compositions[&id]` when editing a precomp).
    #[must_use]
    pub fn active_sequence(&self) -> &zc_project::Sequence {
        if let Some(comp_id) = self.document.active_composition_id
            && let Some(comp) = self.project.compositions.get(&comp_id)
        {
            return comp;
        }
        &self.project.sequence
    }

    /// Mutable view of the currently active [`zc_project::Sequence`].
    pub fn active_sequence_mut(&mut self) -> &mut zc_project::Sequence {
        if let Some(comp_id) = self.document.active_composition_id
            && self.project.compositions.contains_key(&comp_id)
        {
            return self
                .project
                .compositions
                .get_mut(&comp_id)
                .expect("checked contains_key");
        }
        &mut self.project.sequence
    }

    /// Flushes `self.document` into the active target sequence in `self.project`.
    pub fn sync_to_project(&mut self) {
        self.sync_document_to_project();
    }

    /// Flushes `self.document` into the active target sequence and returns a cloned [`zc_project::Project`].
    #[must_use]
    pub fn to_project(&mut self) -> zc_project::Project {
        self.sync_document_to_project();
        self.project.clone()
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
            enabled_plugins: Some(self.plugins.enabled_ids().clone()),
            asset_blob_refs: BTreeMap::new(),
        }
    }

    /// Captures a lightweight undo snapshot for history storage without copying or re-hashing
    /// embedded media buffers on each edit: temporarily moves raw asset `Vec<u8>` buffers out of
    /// `self.project.assets`, interns them by content SHA-256 in `self.asset_blobs` (reusing
    /// `rec.sha256` in O(1) whenever it is a valid 64-hex digest matching `rec.size`), clones
    /// the byte-free metadata `Project`, and moves the live buffers back into `self.project.assets`.
    fn capture_history_snapshot(&mut self) -> SessionSnapshot {
        let mut taken: Vec<(String, Vec<u8>)> = Vec::new();
        let mut asset_blob_refs: BTreeMap<String, String> = BTreeMap::new();

        for (key, rec) in &mut self.project.assets {
            if let Some(bytes) = rec.bytes.take() {
                let bytes_len = bytes.len() as u64;
                let cached_digest = if is_valid_sha256_hex(&rec.sha256) && rec.size == bytes_len {
                    match self.asset_blobs.get(&rec.sha256) {
                        Some(existing) if bytes_edge_eq(existing, &bytes) => {
                            Some(rec.sha256.clone())
                        }
                        Some(_) => None,
                        None => Some(rec.sha256.clone()),
                    }
                } else {
                    None
                };
                let digest = if let Some(digest) = cached_digest {
                    digest
                } else {
                    let computed = zc_project::sha256_hex(&bytes);
                    rec.sha256 = computed.clone();
                    rec.size = bytes_len;
                    computed
                };
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
            enabled_plugins: Some(self.plugins.enabled_ids().clone()),
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
            let (src_locked, src_track_name, src_track_id, clip_len) = match src_kind {
                TrackKind::Video => {
                    let t = &self.document.video_tracks[src_track_idx];
                    (
                        t.locked,
                        t.name.clone(),
                        t.id.clone(),
                        t.clips[src_clip_idx].length,
                    )
                }
                TrackKind::Audio => {
                    let t = &self.document.audio_tracks[src_track_idx];
                    (
                        t.locked,
                        t.name.clone(),
                        t.id.clone(),
                        t.clips[src_clip_idx].length,
                    )
                }
            };
            if src_locked {
                return Err(track_locked_error(&src_track_name, &src_track_id));
            }
            let (dest_kind, dest_track_idx) = if let Some(idx) = self
                .document
                .video_tracks
                .iter()
                .position(|t| t.id == *dest_track_id)
            {
                let t = &self.document.video_tracks[idx];
                if t.locked {
                    return Err(track_locked_error(&t.name, &t.id));
                }
                (TrackKind::Video, idx)
            } else if let Some(idx) = self
                .document
                .audio_tracks
                .iter()
                .position(|t| t.id == *dest_track_id)
            {
                let t = &self.document.audio_tracks[idx];
                if t.locked {
                    return Err(track_locked_error(&t.name, &t.id));
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
        deps
    }

    /// Returns a deterministic JSON inspection payload for GUI, CLI, tests, and MCP tools.
    #[must_use]
    pub fn inspect_state(&self) -> Value {
        let compositions: Vec<Value> = self
            .project
            .compositions
            .iter()
            .map(|(id, seq)| {
                json!({
                    "id": id.to_string(),
                    "name": seq.name,
                    "width": seq.resolution[0],
                    "height": seq.resolution[1],
                    "fps": seq.fps,
                })
            })
            .collect();
        json!({
            "project_name": self.document.project_name,
            "resolution": self.document.resolution,
            "fps": self.document.fps,
            "sequence_length": self.document.sequence_length,
            "active_composition_id": self.document.active_composition_id.map(|id| id.to_string()),
            "compositions": compositions,
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
        let records_undo = command.is_undoable();
        if records_undo {
            let before = if self.history.is_in_transaction() {
                None
            } else {
                Some(self.capture_history_snapshot())
            };
            self.doc_already_synced_with_project = false;
            let result = match self.execute_inner(command) {
                Ok(val) => val,
                Err(err) => {
                    self.doc_already_synced_with_project = false;
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
            if self.doc_already_synced_with_project {
                self.doc_already_synced_with_project = false;
                self.refresh_plugin_requirements_from_document();
            } else {
                self.sync_document_to_project();
            }
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

    pub(crate) fn alloc_uuid_id(&mut self) -> String {
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
    pub(crate) fn record_snapshot_ids(&mut self) {
        let (ids, next) = seed_id_allocator(&self.project, &self.document);
        self.used_ids.extend(ids);
        if next > self.next_id {
            self.next_id = next;
        }
    }

    pub(crate) fn restore_snapshot(&mut self, mut snapshot: SessionSnapshot) {
        let prev_active_comp = self.document.active_composition_id;
        // Rehydrate ONLY assets explicitly recorded in `asset_blob_refs` (preserving
        // exact content-addressed versions and leaving originally-`None` assets as `None`).
        // When the live project already holds the exact same content-addressed buffer, move it
        // in O(1) instead of cloning multi-megabyte media payloads on every Undo/Redo.
        for (asset_key, blob_digest) in &snapshot.asset_blob_refs {
            if let Some(rec) = snapshot.project.assets.get_mut(asset_key)
                && rec.bytes.is_none()
                && let Some(blob) = self.asset_blobs.get(blob_digest)
            {
                let blob_len = blob.len() as u64;
                let reused = self
                    .project
                    .assets
                    .get_mut(asset_key)
                    .filter(|cur| {
                        cur.sha256 == *blob_digest
                            && cur.size == blob_len
                            && cur.bytes.as_deref().is_some_and(|b| bytes_edge_eq(b, blob))
                    })
                    .and_then(|cur| cur.bytes.take());
                rec.bytes = Some(reused.unwrap_or_else(|| (**blob).clone()));
                rec.sha256 = blob_digest.clone();
                rec.size = blob_len;
            }
        }
        self.project = snapshot.project;
        self.document = snapshot.document;
        self.selection = snapshot.selection;
        if let Some(enabled_plugins) = snapshot.enabled_plugins {
            let prev_enabled = self.plugins.enabled_ids().clone();
            let mut settings = self.plugins.export_settings();
            settings.enabled_plugins = enabled_plugins;
            self.plugins.apply_settings(&settings);
            if *self.plugins.enabled_ids() != prev_enabled {
                self.plugin_revision = self.plugin_revision.saturating_add(1);
            }
        }
        // Preserve pure view navigation (`active_composition_id`) across unrelated Undo/Redo
        // whenever the currently viewed composition still exists in the restored project.
        if prev_active_comp != self.document.active_composition_id {
            let target_comp =
                prev_active_comp.filter(|id| self.project.compositions.contains_key(id));
            if target_comp != self.document.active_composition_id {
                let target_seq = match target_comp {
                    None => &self.project.sequence,
                    Some(id) => self
                        .project
                        .compositions
                        .get(&id)
                        .unwrap_or(&self.project.sequence),
                };
                self.document = SessionDocument::from_sequence(
                    &self.project.name,
                    &self.project.assets,
                    &self.project.media,
                    target_seq,
                    target_comp,
                );
                self.selection.clear_all();
            }
        }
        self.prev_doc_plugin_deps = self.document.collect_plugin_dependencies();
        self.record_snapshot_ids();
        let max_frame = (self.document.sequence_length - 1.0).max(0.0);
        self.playback.current_frame = self.playback.current_frame.clamp(0.0, max_frame);
        self.playback.bump_epoch();
    }
}

impl ControlTarget for Session {
    fn dispatch_command(&mut self, command: Command) -> Result<Value, ControlError> {
        self.execute(command).map_err(ControlError::from)
    }
}
