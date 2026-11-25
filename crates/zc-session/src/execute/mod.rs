mod objects_effects;
mod properties;

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use crate::commands::{Command, CommandMeta};
use crate::document::{
    SessionClip, SessionDocument, SessionError, SessionProperty, SessionTrack, SessionValueKind,
    TrackKind, is_valid_namespaced_plugin_effect_id, track_locked_error,
};
use crate::history::HistoryError;
use crate::pro_commands::{scale_property_track_keyframes, shift_property_track_keyframes};
use crate::selection::PlaybackState;
use crate::sync_from_project::{default_clip_properties, parse_uuid_or_new};
use crate::{Session, seed_id_allocator};

impl Session {
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
    pub(crate) fn resolve_track_overlaps_for_group(
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
                            right_rem.id = right_id.clone();
                            self.pending_clip_side_table_shifts
                                .insert(right_id, -cut_right);
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

    pub(crate) fn remap_cloned_clip_children(&mut self, clip: &mut SessionClip) {
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

    pub(crate) fn execute_inner(&mut self, command: Command) -> Result<Value, SessionError> {
        match command {
            // --- Project ---
            Command::NewProject => {
                self.project = zc_project::Project::new_cm3_default();
                self.document = SessionDocument::from_project(&self.project);
                self.asset_blobs.clear();
                self.pending_clip_side_table_shifts.clear();
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
                self.active_sequence_mut().exact_fps = None;
                self.playback.bump_epoch();
                Ok(json!({ "fps": self.document.fps }))
            }
            Command::SetSequenceLength { frames } => {
                if !frames.is_finite() || frames <= 0.0 || frames > 1_000_000.0 {
                    return Err(SessionError::InvalidArgument(format!(
                        "sequence length {frames} must be a finite number in (0, 1000000]"
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
                    let t = &self.document.video_tracks[idx];
                    if t.locked {
                        return Err(track_locked_error(&t.name, &t.id));
                    }
                    self.document.video_tracks.remove(idx)
                } else if let Some(idx) = self
                    .document
                    .audio_tracks
                    .iter()
                    .position(|t| t.id == track_id)
                {
                    let t = &self.document.audio_tracks[idx];
                    if t.locked {
                        return Err(track_locked_error(&t.name, &t.id));
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
                    let t = &self.document.video_tracks[idx];
                    if t.locked {
                        return Err(track_locked_error(&t.name, &t.id));
                    }
                    (TrackKind::Video, idx)
                } else if let Some(idx) = self
                    .document
                    .audio_tracks
                    .iter()
                    .position(|t| t.id == track_id)
                {
                    let t = &self.document.audio_tracks[idx];
                    if t.locked {
                        return Err(track_locked_error(&t.name, &t.id));
                    }
                    (TrackKind::Audio, idx)
                } else {
                    return Err(SessionError::NotFound(format!("track '{track_id}'")));
                };

                let raw_kind = kind.trim();
                let raw_lower = raw_kind.to_ascii_lowercase();
                let is_audio_clip = raw_lower == "audio";
                let valid_video_kind = matches!(
                    raw_lower.as_str(),
                    "video"
                        | "image"
                        | "adjustment"
                        | "composite"
                        | "scene"
                        | "scene_3d"
                        | "scene3d"
                        | "generator"
                        | "color"
                        | "solid"
                        | "shape"
                        | "shape2d"
                        | "shape_2d"
                        | "shape_layer"
                        | "text"
                        | "text_2d"
                        | "text2d"
                        | "preset_shape"
                        | "presetshape"
                        | "camera"
                        | "camera_layer"
                        | "precomp"
                        | "shape_plus"
                ) || is_valid_namespaced_plugin_effect_id(raw_kind);
                if !is_audio_clip && !valid_video_kind {
                    return Err(SessionError::InvalidArgument(format!(
                        "unknown clip kind '{kind}'"
                    )));
                }
                if (track_kind == TrackKind::Audio) != is_audio_clip {
                    return Err(SessionError::InvalidOperation(format!(
                        "clip kind '{kind}' is incompatible with {track_kind:?} track '{track_id}'"
                    )));
                }

                let kind = match raw_lower.as_str() {
                    "shape2d" | "shape_2d" => "shape".to_owned(),
                    "text2d" => "text_2d".to_owned(),
                    "precomp" => "zoidium:precomp-plus/precomp".to_owned(),
                    "shape_plus" | "zoidium:shape-plus/layer" => {
                        "zoidium:shape-plus/shape-layer".to_owned()
                    }
                    "camera" | "camera_layer" => "zoidium:camera-plus/camera-layer".to_owned(),
                    "presetshape" => "preset_shape".to_owned(),
                    _ => raw_kind.to_owned(),
                };
                // Auto-enable required built-in plugin atomically inside this undo snapshot.
                if let Some(plugin_id) = kind
                    .strip_prefix("zoidium:")
                    .or_else(|| kind.strip_prefix("zoidcraft:"))
                    .and_then(|rest| rest.split_once('/').map(|(p, _)| p))
                    .filter(|p| !p.is_empty())
                {
                    let _ = self.enable_plugin_requirement(plugin_id);
                }
                let clip_id = self.alloc_uuid_id();
                let clip_name = name.unwrap_or_else(|| kind.clone());
                let fps = self.document.fps.max(1.0);
                let seq_bg = self.active_sequence().background_color;
                let mut properties = default_clip_properties(track_kind, &kind, seq_bg);
                if kind == "zoidium:precomp-plus/precomp" {
                    let comp_uuid = parse_uuid_or_new(&self.alloc_uuid_id());
                    let v1_uuid = parse_uuid_or_new(&self.alloc_uuid_id());
                    let a1_uuid = parse_uuid_or_new(&self.alloc_uuid_id());
                    let comp = zc_project::Sequence {
                        id: comp_uuid,
                        name: clip_name.clone(),
                        resolution: self.document.resolution,
                        fps: fps.round().clamp(1.0, 240.0) as u32,
                        length_frames: self.document.sequence_length.max(length),
                        background_color: seq_bg,
                        video_tracks: vec![zc_project::VideoTrack {
                            id: v1_uuid,
                            name: "V1".to_owned(),
                            enabled: true,
                            solo: false,
                            locked: false,
                            is_3d: false,
                            clips: Vec::new(),
                            preserved: BTreeMap::new(),
                        }],
                        audio_tracks: vec![zc_project::AudioTrack {
                            id: a1_uuid,
                            name: "A1".to_owned(),
                            enabled: true,
                            muted: false,
                            solo: false,
                            locked: false,
                            clips: Vec::new(),
                            preserved: BTreeMap::new(),
                        }],
                        ..zc_project::Sequence::default()
                    };
                    self.project.compositions.insert(comp_uuid, comp);
                    properties.insert(
                        "compositionId".to_owned(),
                        SessionProperty::static_value_with_kind(
                            json!(comp_uuid.to_string()),
                            SessionValueKind::Text,
                        ),
                    );
                }
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
                    return Err(track_locked_error(&src_track.name, &src_track.id));
                }

                let dest_track_id = target_track_id.unwrap_or_else(|| src_track.id.clone());
                let (dest_kind, dest_track_idx) = if let Some(idx) = self
                    .document
                    .video_tracks
                    .iter()
                    .position(|t| t.id == dest_track_id)
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
                    .position(|t| t.id == dest_track_id)
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
                    return Err(track_locked_error(&track.name, &track.id));
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
                if kind == TrackKind::Video
                    && let Ok(clip_uuid) = zc_project::Uuid::parse_str(&clip_id)
                {
                    let seq = self.active_sequence_mut();
                    if let Some(layer_id) = seq
                        .video_tracks
                        .iter()
                        .flat_map(|t| t.clips.iter())
                        .find(|c| c.id == clip_uuid)
                        .map(|c| c.layer.id)
                        && let Some(masks) = seq.layer_masks.get_mut(&layer_id)
                    {
                        if rate_stretch && old_length > 0.0 {
                            let scale = length / old_length;
                            for m in masks {
                                scale_property_track_keyframes(&mut m.path, scale);
                            }
                        } else {
                            for m in masks {
                                shift_property_track_keyframes(&mut m.path, -applied_delta);
                            }
                        }
                    }
                }
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
                    return Err(track_locked_error(&track.name, &track.id));
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
                if kind == TrackKind::Video
                    && rate_stretch
                    && old_length > 0.0
                    && let Ok(clip_uuid) = zc_project::Uuid::parse_str(&clip_id)
                {
                    let scale = new_length / old_length;
                    let seq = self.active_sequence_mut();
                    if let Some(layer_id) = seq
                        .video_tracks
                        .iter()
                        .flat_map(|t| t.clips.iter())
                        .find(|c| c.id == clip_uuid)
                        .map(|c| c.layer.id)
                        && let Some(masks) = seq.layer_masks.get_mut(&layer_id)
                    {
                        for m in masks {
                            scale_property_track_keyframes(&mut m.path, scale);
                        }
                    }
                }
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
                    return Err(track_locked_error(&track.name, &track.id));
                }
                let orig = &track.clips[clip_idx];
                let local_offset = at_frame - orig.start;
                // Spec 02 §2.7: strictly require 0 < local_offset < clip.length.
                if local_offset <= 0.0 || local_offset >= orig.length {
                    return Err(SessionError::InvalidOperation(
                        "Move playhead inside the selected clip to split".to_owned(),
                    ));
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
                self.pending_clip_side_table_shifts
                    .insert(right_clip_id.clone(), -local_offset);
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
                    return Err(track_locked_error(&track.name, &track.id));
                }

                let mut cloned = track.clips[clip_idx].clone();
                let new_clip_id = self.alloc_uuid_id();
                let cloned_kind = cloned.kind.clone();
                cloned.source_clip_id = cloned
                    .source_clip_id
                    .clone()
                    .or_else(|| Some(cloned.id.clone()));
                cloned.id = new_clip_id.clone();
                cloned.start += cloned.length;
                self.remap_cloned_clip_children(&mut cloned);

                let new_start = cloned.start;
                let new_end = new_start + cloned.length;
                let range_overlaps_track = |t: &SessionTrack| {
                    t.clips.iter().any(|c| {
                        let c_end = c.start + c.length;
                        c_end > new_start + 1e-6 && c.start < new_end - 1e-6
                    })
                };
                let dest_track_idx = {
                    let tracks = match kind {
                        TrackKind::Video => &self.document.video_tracks,
                        TrackKind::Audio => &self.document.audio_tracks,
                    };
                    if !range_overlaps_track(&tracks[track_idx]) {
                        track_idx
                    } else if let Some(idx) = ((track_idx + 1)..tracks.len())
                        .chain(0..track_idx)
                        .find(|&i| !tracks[i].locked && !range_overlaps_track(&tracks[i]))
                    {
                        idx
                    } else {
                        let new_track_id = self.alloc_uuid_id();
                        match kind {
                            TrackKind::Video => {
                                let name = format!("V{}", self.document.video_tracks.len() + 1);
                                self.document.video_tracks.push(SessionTrack {
                                    id: new_track_id,
                                    kind: TrackKind::Video,
                                    name,
                                    enabled: true,
                                    solo: false,
                                    locked: false,
                                    is_3d: false,
                                    clips: Vec::new(),
                                });
                                self.document.video_tracks.len() - 1
                            }
                            TrackKind::Audio => {
                                let name = format!("A{}", self.document.audio_tracks.len() + 1);
                                self.document.audio_tracks.push(SessionTrack {
                                    id: new_track_id,
                                    kind: TrackKind::Audio,
                                    name,
                                    enabled: true,
                                    solo: false,
                                    locked: false,
                                    is_3d: false,
                                    clips: Vec::new(),
                                });
                                self.document.audio_tracks.len() - 1
                            }
                        }
                    }
                };
                let dest_track_id = match kind {
                    TrackKind::Video => {
                        let t = &mut self.document.video_tracks[dest_track_idx];
                        t.clips.push(cloned);
                        t.clips.sort_by(|a, b| a.start.total_cmp(&b.start));
                        t.id.clone()
                    }
                    TrackKind::Audio => {
                        let t = &mut self.document.audio_tracks[dest_track_idx];
                        t.clips.push(cloned);
                        t.clips.sort_by(|a, b| a.start.total_cmp(&b.start));
                        t.id.clone()
                    }
                };

                // Select newly created duplicate(s) (preserving multi-duplicate selection inside transactions).
                if self.history.is_in_transaction()
                    && self.selection.clip_ids.len() > 1
                    && let Some(pos) = self.selection.clip_ids.iter().position(|id| id == &clip_id)
                {
                    self.selection.clip_ids[pos] = new_clip_id.clone();
                    self.selection.clear_objects_3d();
                    self.selection.clear_effects();
                    self.selection.clear_keyframes();
                } else {
                    let base_type = self.selection.clip_base_type.clone().or(Some(cloned_kind));
                    let _ = self.selection.select_clip(
                        new_clip_id.clone(),
                        base_type.as_deref(),
                        false,
                    );
                }

                self.document.recompute_sequence_length_if_extended();
                self.playback.bump_epoch();
                Ok(json!({
                    "clip_id": new_clip_id,
                    "source_clip_id": clip_id,
                    "track_id": dest_track_id,
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
                    return Err(track_locked_error(&track_mut.name, &track_mut.id));
                }
                let removed = track_mut.clips.remove(clip_idx);
                self.prune_clip_selection(&removed);
                self.playback.bump_epoch();
                Ok(json!({ "deleted_clip_id": removed.id }))
            }

            // --- Scene & Effects ---
            cmd @ (Command::AddObject3D { .. }
            | Command::DeleteObject3D { .. }
            | Command::DuplicateObject3D { .. }
            | Command::ReorderObject3D { .. }
            | Command::AddEffect { .. }
            | Command::DeleteEffect { .. }
            | Command::DuplicateEffect { .. }
            | Command::ReorderEffect { .. }) => self.exec_scene_or_effect(cmd),

            // --- Properties & Keyframes ---
            cmd @ (Command::SetPropertyValue { .. }
            | Command::TogglePropertyAnimated { .. }
            | Command::SetKeyframe { .. }
            | Command::DeleteKeyframe { .. }
            | Command::SetExpression { .. }
            | Command::RemoveProperty { .. }
            | Command::ResetPropertyList { .. }) => self.exec_property_command(cmd),

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
                    "vanilla_cm3_mode": self.plugins.is_vanilla_cm3_mode() && active_deps.is_empty(),
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

            // --- Pro Commands (Shape+, LayerMask, Precomp+, Preserved/Preset/SplitLetters, Markers, Controls, Clipboard, Assets) ---
            Command::UpdateShapePlusPayload { clip_id, payload } => {
                self.exec_update_shape_plus_payload(clip_id, payload)
            }
            Command::AddLayerMask { clip_id, mask } => self.exec_add_layer_mask(clip_id, mask),
            Command::UpdateLayerMask {
                clip_id,
                mask_id,
                mask,
            } => self.exec_update_layer_mask(clip_id, mask_id, mask),
            Command::RemoveLayerMask { clip_id, mask_id } => {
                self.exec_remove_layer_mask(clip_id, mask_id)
            }
            Command::ReorderLayerMask {
                clip_id,
                mask_id,
                to_index,
            } => self.exec_reorder_layer_mask(clip_id, mask_id, to_index),
            Command::DuplicateLayerMask { clip_id, mask_id } => {
                self.exec_duplicate_layer_mask(clip_id, mask_id)
            }
            Command::CreateComposition {
                name,
                resolution,
                fps,
                length_frames,
            } => self.exec_create_composition(name, resolution, fps, length_frames),
            Command::RenameComposition { comp_id, name } => {
                self.exec_rename_composition(comp_id, name)
            }
            Command::DuplicateComposition { comp_id } => self.exec_duplicate_composition(comp_id),
            Command::DeleteComposition { comp_id } => self.exec_delete_composition(comp_id),
            Command::PrecomposeClips { clip_ids, name } => {
                self.exec_precompose_clips(clip_ids, name)
            }
            Command::SetActiveComposition { comp_id } => self.exec_set_active_composition(comp_id),
            Command::SetPrecompClipTarget {
                clip_id,
                composition_id,
            } => self.exec_set_precomp_clip_target(clip_id, composition_id),
            Command::UpdateLayerPreserved {
                clip_id,
                key,
                value,
            } => self.exec_update_layer_preserved(clip_id, key, value),
            Command::SetPresetShapeKind {
                clip_id,
                shape_kind,
            } => self.exec_set_preset_shape_kind(clip_id, shape_kind),
            Command::SplitText3DIntoLetters { clip_id, object_id } => {
                self.exec_split_text_3d_into_letters(clip_id, object_id)
            }
            Command::AddMarker { frame, name, color } => self.exec_add_marker(frame, name, color),
            Command::UpdateMarker {
                index,
                frame,
                name,
                color,
            } => self.exec_update_marker(index, frame, name, color),
            Command::DeleteMarker { index } => self.exec_delete_marker(index),
            Command::AddSequenceControl {
                name,
                kind,
                value,
                min,
                max,
                step,
            } => self.exec_add_sequence_control(name, kind, value, min, max, step),
            Command::UpdateSequenceControl {
                control_id,
                name,
                value,
                min,
                max,
                step,
            } => self.exec_update_sequence_control(control_id, name, value, min, max, step),
            Command::DeleteSequenceControl { control_id } => {
                self.exec_delete_sequence_control(control_id)
            }
            Command::ReorderSequenceControl {
                control_id,
                to_index,
            } => self.exec_reorder_sequence_control(control_id, to_index),
            Command::PasteClips {
                clips,
                target_frame,
            } => self.exec_paste_clips(clips, target_frame),
            Command::FreezeClipFrame { clip_id, frame } => {
                self.exec_freeze_clip_frame(clip_id, frame)
            }
            Command::ReverseClip { clip_id } => self.exec_reverse_clip(clip_id),
            Command::RelinkAsset {
                asset_id,
                filename,
                bytes,
            } => self.exec_relink_asset(asset_id, filename, bytes),
            Command::PurgeUnusedAssets => self.exec_purge_unused_assets(),
            Command::PastePzJson {
                json,
                hint,
                target_clip_id,
                parent_object_id,
                target_object_id,
                target_property,
                at_frame,
            } => self.exec_paste_pz_json(
                &json,
                hint.as_deref(),
                target_clip_id.as_deref(),
                parent_object_id.as_deref(),
                target_object_id.as_deref(),
                target_property.as_deref(),
                at_frame,
            ),
            Command::CopyPzJson {
                kind,
                clip_id,
                target_ids,
                property,
            } => {
                self.exec_copy_pz_json(&kind, clip_id.as_deref(), &target_ids, property.as_deref())
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
