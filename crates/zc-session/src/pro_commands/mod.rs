mod freeze_expr;
mod helpers;

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde_json::{Value, json};
use zc_project::Uuid;

use crate::{
    ClipSnapshot, Session, SessionClip, SessionDocument, SessionError, SessionKeyframe,
    SessionObject3D, SessionProperty, SessionTrack, SessionValueKind, TrackKind,
    eval_session_property_at, parse_uuid_or_new, prop_f64_static, prop_str_static,
    regenerate_cloned_layer_ids,
};

use freeze_expr::{
    FreezeExprState, FreezeTimeExprContext, offset_property_track_expressions_time_origin,
    reverse_session_keyframes,
};
pub(crate) use helpers::{
    clone_layer_masks_shifted, scale_property_track_keyframes, shift_property_track_keyframes,
};
use helpers::{
    collect_sequence_asset_refs, find_video_clip_layer_id_checked, find_video_clip_mut_checked,
    proportional_char_advance_ratio, refresh_sequence_controls_in_document,
    regenerate_sequence_uuids, sequence_references_composition, sync_marker_colors_vec,
    write_marker_colors_vec,
};

impl Session {
    /// Captures serializable [`ClipSnapshot`] copies of the specified clips from the active
    /// composition (including full project layers, `LayerMask` lists, and `Precomp+` offsets)
    /// for clipboard copy/paste via [`crate::Command::PasteClips`].
    pub fn copy_clips(&mut self, clip_ids: &[Uuid]) -> Result<Vec<ClipSnapshot>, SessionError> {
        if clip_ids.is_empty() {
            return Err(SessionError::InvalidArgument(
                "clip_ids cannot be empty".to_owned(),
            ));
        }
        self.sync_document_to_project();
        let mut out = Vec::with_capacity(clip_ids.len());
        let seq = self.active_sequence();
        for &cid in clip_ids {
            let cid_str = cid.to_string();
            let (kind, track_idx, clip_idx) = self
                .document
                .find_clip_location(&cid_str)
                .ok_or_else(|| SessionError::NotFound(format!("clip '{cid}'")))?;
            match kind {
                TrackKind::Video => {
                    let s_track = &self.document.video_tracks[track_idx];
                    let s_clip = s_track.clips[clip_idx].clone();
                    let vc = seq
                        .video_tracks
                        .iter()
                        .flat_map(|t| t.clips.iter())
                        .find(|c| c.id == cid)
                        .cloned();
                    let layer_masks = vc
                        .as_ref()
                        .and_then(|c| seq.layer_masks.get(&c.layer.id))
                        .cloned()
                        .unwrap_or_default();
                    let precomp_offset = vc
                        .as_ref()
                        .and_then(|c| seq.precomp_offsets.get(&c.layer.id).copied());
                    out.push(ClipSnapshot {
                        track_kind: TrackKind::Video,
                        track_id: Some(s_track.id.clone()),
                        clip: s_clip,
                        video_clip: vc,
                        audio_clip: None,
                        layer_masks,
                        precomp_offset,
                    });
                }
                TrackKind::Audio => {
                    let s_track = &self.document.audio_tracks[track_idx];
                    let s_clip = s_track.clips[clip_idx].clone();
                    let ac = seq
                        .audio_tracks
                        .iter()
                        .flat_map(|t| t.clips.iter())
                        .find(|c| c.id == cid)
                        .cloned();
                    out.push(ClipSnapshot {
                        track_kind: TrackKind::Audio,
                        track_id: Some(s_track.id.clone()),
                        clip: s_clip,
                        video_clip: None,
                        audio_clip: ac,
                        layer_masks: Vec::new(),
                        precomp_offset: None,
                    });
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn exec_update_shape_plus_payload(
        &mut self,
        clip_id: Uuid,
        payload: Value,
    ) -> Result<Value, SessionError> {
        let seq = self.active_sequence_mut();
        let vc = find_video_clip_mut_checked(seq, clip_id)?;
        match &mut vc.layer.kind {
            zc_project::LayerKind::Plugin {
                plugin_type,
                payload: slot,
            } if plugin_type.contains("shape-plus") => {
                *slot = payload;
            }
            _ => {
                return Err(SessionError::InvalidOperation(format!(
                    "clip '{clip_id}' is not a Shape+ layer"
                )));
            }
        }
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({ "clip_id": clip_id.to_string() }))
    }

    pub(crate) fn exec_add_layer_mask(
        &mut self,
        clip_id: Uuid,
        mut mask: zc_project::LayerMask,
    ) -> Result<Value, SessionError> {
        let layer_id = find_video_clip_layer_id_checked(self.active_sequence(), clip_id)?;
        let existing_ids: BTreeSet<String> = self
            .active_sequence()
            .layer_masks
            .get(&layer_id)
            .map(|v| v.iter().map(|m| m.id.clone()).collect())
            .unwrap_or_default();
        if mask.id.trim().is_empty() || existing_ids.contains(&mask.id) {
            mask.id = format!("mask-{}", self.alloc_uuid_id());
        }
        let mask_id = mask.id.clone();
        let entry = self
            .active_sequence_mut()
            .layer_masks
            .entry(layer_id)
            .or_default();
        entry.push(mask);
        let count = entry.len();
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "layer_id": layer_id.to_string(),
            "mask_id": mask_id,
            "count": count,
        }))
    }

    pub(crate) fn exec_update_layer_mask(
        &mut self,
        clip_id: Uuid,
        mask_id: String,
        mut mask: zc_project::LayerMask,
    ) -> Result<Value, SessionError> {
        let layer_id = find_video_clip_layer_id_checked(self.active_sequence(), clip_id)?;
        let masks = self
            .active_sequence_mut()
            .layer_masks
            .get_mut(&layer_id)
            .ok_or_else(|| {
                SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
            })?;
        let pos = masks.iter().position(|m| m.id == mask_id).ok_or_else(|| {
            SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
        })?;
        if mask.id.trim().is_empty() {
            mask.id = mask_id.clone();
        }
        let final_id = mask.id.clone();
        masks[pos] = mask;
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "mask_id": final_id,
        }))
    }

    pub(crate) fn exec_remove_layer_mask(
        &mut self,
        clip_id: Uuid,
        mask_id: String,
    ) -> Result<Value, SessionError> {
        let layer_id = find_video_clip_layer_id_checked(self.active_sequence(), clip_id)?;
        let seq = self.active_sequence_mut();
        let masks = seq.layer_masks.get_mut(&layer_id).ok_or_else(|| {
            SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
        })?;
        let pos = masks.iter().position(|m| m.id == mask_id).ok_or_else(|| {
            SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
        })?;
        masks.remove(pos);
        let remaining = masks.len();
        if remaining == 0 {
            seq.layer_masks.remove(&layer_id);
        }
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "removed_mask_id": mask_id,
            "remaining": remaining,
        }))
    }

    pub(crate) fn exec_reorder_layer_mask(
        &mut self,
        clip_id: Uuid,
        mask_id: String,
        to_index: usize,
    ) -> Result<Value, SessionError> {
        let layer_id = find_video_clip_layer_id_checked(self.active_sequence(), clip_id)?;
        let masks = self
            .active_sequence_mut()
            .layer_masks
            .get_mut(&layer_id)
            .ok_or_else(|| {
                SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
            })?;
        let pos = masks.iter().position(|m| m.id == mask_id).ok_or_else(|| {
            SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
        })?;
        let item = masks.remove(pos);
        let insert_at = to_index.min(masks.len());
        masks.insert(insert_at, item);
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "mask_id": mask_id,
            "index": insert_at,
        }))
    }

    pub(crate) fn exec_duplicate_layer_mask(
        &mut self,
        clip_id: Uuid,
        mask_id: String,
    ) -> Result<Value, SessionError> {
        let layer_id = find_video_clip_layer_id_checked(self.active_sequence(), clip_id)?;
        let new_id = format!("mask-{}", self.alloc_uuid_id());
        let masks = self
            .active_sequence_mut()
            .layer_masks
            .get_mut(&layer_id)
            .ok_or_else(|| {
                SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
            })?;
        let pos = masks.iter().position(|m| m.id == mask_id).ok_or_else(|| {
            SessionError::NotFound(format!("mask '{mask_id}' on clip '{clip_id}'"))
        })?;
        let mut cloned = masks[pos].clone();
        cloned.id = new_id.clone();
        cloned.name = format!("{} Copy", cloned.name);
        masks.insert(pos + 1, cloned);
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "mask_id": new_id,
            "source_mask_id": mask_id,
        }))
    }

    pub(crate) fn exec_create_composition(
        &mut self,
        name: String,
        resolution: [u32; 2],
        fps: u32,
        length_frames: u32,
    ) -> Result<Value, SessionError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(SessionError::InvalidArgument(
                "composition name cannot be empty".to_owned(),
            ));
        }
        if resolution[0] == 0 || resolution[1] == 0 || resolution[0] > 7680 || resolution[1] > 7680
        {
            return Err(SessionError::InvalidArgument(format!(
                "resolution {}x{} out of supported range [1..=7680]",
                resolution[0], resolution[1]
            )));
        }
        if fps == 0 || fps > 240 {
            return Err(SessionError::InvalidArgument(format!(
                "fps {fps} out of supported range [1..=240]"
            )));
        }
        if length_frames == 0 {
            return Err(SessionError::InvalidArgument(
                "length_frames must be >= 1".to_owned(),
            ));
        }
        let comp_id = parse_uuid_or_new(&self.alloc_uuid_id());
        let v1_id = parse_uuid_or_new(&self.alloc_uuid_id());
        let a1_id = parse_uuid_or_new(&self.alloc_uuid_id());
        let comp = zc_project::Sequence {
            id: comp_id,
            name: trimmed.to_owned(),
            resolution,
            fps,
            length_frames: f64::from(length_frames),
            video_tracks: vec![zc_project::VideoTrack {
                id: v1_id,
                name: "V1".to_owned(),
                enabled: true,
                solo: false,
                locked: false,
                is_3d: false,
                clips: Vec::new(),
                preserved: BTreeMap::new(),
            }],
            audio_tracks: vec![zc_project::AudioTrack {
                id: a1_id,
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
        self.project.compositions.insert(comp_id, comp);
        self.enable_plugin_requirement("precomp-plus");
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "comp_id": comp_id.to_string(),
            "name": trimmed,
            "resolution": resolution,
            "fps": fps,
            "length_frames": length_frames,
        }))
    }

    pub(crate) fn exec_rename_composition(
        &mut self,
        comp_id: Uuid,
        name: String,
    ) -> Result<Value, SessionError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(SessionError::InvalidArgument(
                "composition name cannot be empty".to_owned(),
            ));
        }
        let comp = self
            .project
            .compositions
            .get_mut(&comp_id)
            .ok_or_else(|| SessionError::NotFound(format!("composition '{comp_id}'")))?;
        comp.name = trimmed.to_owned();
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "comp_id": comp_id.to_string(),
            "name": trimmed,
        }))
    }

    pub(crate) fn exec_duplicate_composition(
        &mut self,
        comp_id: Uuid,
    ) -> Result<Value, SessionError> {
        let mut cloned = self
            .project
            .compositions
            .get(&comp_id)
            .cloned()
            .ok_or_else(|| SessionError::NotFound(format!("composition '{comp_id}'")))?;
        let mut alloc = || parse_uuid_or_new(&self.alloc_uuid_id());
        regenerate_sequence_uuids(&mut cloned, &mut alloc);
        cloned.name = format!("{} Copy", cloned.name);
        let new_id = cloned.id;
        let new_name = cloned.name.clone();
        self.project.compositions.insert(new_id, cloned);
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "comp_id": new_id.to_string(),
            "source_comp_id": comp_id.to_string(),
            "name": new_name,
        }))
    }

    pub(crate) fn exec_delete_composition(&mut self, comp_id: Uuid) -> Result<Value, SessionError> {
        if !self.project.compositions.contains_key(&comp_id) {
            return Err(SessionError::NotFound(format!("composition '{comp_id}'")));
        }
        // Flush any pending edits in the active sequence before scanning references.
        self.sync_document_to_project();
        let in_use =
            sequence_references_composition(&self.project.sequence, comp_id)
                || self.project.compositions.iter().any(|(&id, seq)| {
                    id != comp_id && sequence_references_composition(seq, comp_id)
                });
        if in_use {
            return Err(SessionError::InvalidOperation(format!(
                "cannot delete composition '{comp_id}' because it is still referenced by a PrecompRef clip"
            )));
        }
        self.project.compositions.remove(&comp_id);
        self.project.composition_ids.retain(|_, v| *v != comp_id);
        if self.document.active_composition_id == Some(comp_id) {
            self.document = SessionDocument::from_sequence(
                &self.project.name,
                &self.project.assets,
                &self.project.media,
                &self.project.sequence,
                None,
            );
            self.selection.clear_all();
        }
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({ "deleted_comp_id": comp_id.to_string() }))
    }

    pub(crate) fn exec_precompose_clips(
        &mut self,
        clip_ids: Vec<Uuid>,
        name: String,
    ) -> Result<Value, SessionError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(SessionError::InvalidArgument(
                "composition name cannot be empty".to_owned(),
            ));
        }
        if clip_ids.is_empty() {
            return Err(SessionError::InvalidArgument(
                "precompose requires at least one clip_id".to_owned(),
            ));
        }
        let unique_ids: BTreeSet<Uuid> = clip_ids.iter().copied().collect();
        // Flush current document edits into active sequence before extracting clips.
        self.sync_document_to_project();

        let (resolution, fps, parent_exact_fps, bg_color, mut child_controls) = {
            let seq = self.active_sequence();
            // Validate every requested clip exists and its track is unlocked.
            let mut found = BTreeSet::new();
            for vt in &seq.video_tracks {
                for vc in &vt.clips {
                    if unique_ids.contains(&vc.id) {
                        if vt.locked {
                            return Err(crate::track_locked_error(&vt.name, &vt.id.to_string()));
                        }
                        found.insert(vc.id);
                    }
                }
            }
            for at in &seq.audio_tracks {
                for ac in &at.clips {
                    if unique_ids.contains(&ac.id) {
                        if at.locked {
                            return Err(crate::track_locked_error(&at.name, &at.id.to_string()));
                        }
                        found.insert(ac.id);
                    }
                }
            }
            for &cid in &unique_ids {
                if !found.contains(&cid) {
                    return Err(SessionError::NotFound(format!("clip '{cid}'")));
                }
            }
            (
                seq.resolution,
                seq.fps.max(1),
                seq.exact_fps,
                seq.background_color,
                seq.controls.clone(),
            )
        };
        let effective_fps = parent_exact_fps.unwrap_or_else(|| f64::from(fps)).max(1.0);

        // Compute bounding span [min_start, max_end], lowest affected video track index,
        // and solo state before extracting selected clips.
        let mut min_start = f64::INFINITY;
        let mut max_end = 0.0_f64;
        let mut target_vtrack_idx: Option<usize> = None;
        let mut any_selected_solo = false;
        let parent_has_solo;
        {
            let seq = self.active_sequence();
            parent_has_solo = seq.video_tracks.iter().any(|vt| vt.solo);
            for (ti, vt) in seq.video_tracks.iter().enumerate() {
                for vc in &vt.clips {
                    if unique_ids.contains(&vc.id) {
                        min_start = min_start.min(vc.start);
                        max_end = max_end.max(vc.start + vc.length);
                        if target_vtrack_idx.is_none() {
                            target_vtrack_idx = Some(ti);
                        }
                        if vt.solo {
                            any_selected_solo = true;
                        }
                    }
                }
            }
            for at in &seq.audio_tracks {
                for ac in &at.clips {
                    if unique_ids.contains(&ac.id) {
                        min_start = min_start.min(ac.start);
                        max_end = max_end.max(ac.start + ac.length);
                    }
                }
            }
        }
        if !min_start.is_finite() {
            min_start = 0.0;
        }
        let span = (max_end - min_start).max(1.0);

        let comp_id = parse_uuid_or_new(&self.alloc_uuid_id());
        let precomp_clip_id = parse_uuid_or_new(&self.alloc_uuid_id());
        let precomp_layer_id = parse_uuid_or_new(&self.alloc_uuid_id());

        let mut new_vtracks = Vec::new();
        let mut new_atracks = Vec::new();
        let mut moved_layer_ids = Vec::new();

        // Extract selected clips from active sequence and move their side-tables into the new comp.
        let mut moved_layer_masks = BTreeMap::new();
        let mut moved_precomp_offsets = BTreeMap::new();
        let mut moved_camera_objs = BTreeMap::new();
        let mut moved_backdrops = BTreeMap::new();
        let mut moved_layer_3d = BTreeMap::new();

        let track_alloc_count_v = self.active_sequence().video_tracks.len();
        let track_alloc_count_a = self.active_sequence().audio_tracks.len();
        let fresh_vtrack_ids: Vec<Uuid> = (0..=track_alloc_count_v)
            .map(|_| parse_uuid_or_new(&self.alloc_uuid_id()))
            .collect();
        let fresh_atrack_ids: Vec<Uuid> = (0..=track_alloc_count_a)
            .map(|_| parse_uuid_or_new(&self.alloc_uuid_id()))
            .collect();

        {
            let seq = self.active_sequence_mut();
            for (idx, vt) in seq.video_tracks.iter_mut().enumerate() {
                let mut extracted = Vec::new();
                vt.clips.retain(|vc| {
                    !unique_ids.contains(&vc.id) || {
                        let mut moved = vc.clone();
                        moved.start = (moved.start - min_start).max(0.0);
                        moved_layer_ids.push(moved.layer.id);
                        extracted.push(moved);
                        false
                    }
                });
                if !extracted.is_empty() {
                    new_vtracks.push(zc_project::VideoTrack {
                        id: fresh_vtrack_ids[idx],
                        name: format!("V{}", new_vtracks.len() + 1),
                        enabled: vt.enabled,
                        solo: vt.solo,
                        locked: false,
                        is_3d: vt.is_3d,
                        clips: extracted,
                        preserved: BTreeMap::new(),
                    });
                }
            }
            for (idx, at) in seq.audio_tracks.iter_mut().enumerate() {
                let mut extracted = Vec::new();
                at.clips.retain(|ac| {
                    !unique_ids.contains(&ac.id) || {
                        let mut moved = ac.clone();
                        moved.start = (moved.start - min_start).max(0.0);
                        extracted.push(moved);
                        false
                    }
                });
                if !extracted.is_empty() {
                    new_atracks.push(zc_project::AudioTrack {
                        id: fresh_atrack_ids[idx],
                        name: format!("A{}", new_atracks.len() + 1),
                        enabled: at.enabled,
                        muted: at.muted,
                        solo: at.solo,
                        locked: false,
                        clips: extracted,
                        preserved: BTreeMap::new(),
                    });
                }
            }
            for lid in &moved_layer_ids {
                if let Some(m) = seq.layer_masks.remove(lid) {
                    moved_layer_masks.insert(*lid, m);
                }
                if let Some(o) = seq.precomp_offsets.remove(lid) {
                    moved_precomp_offsets.insert(*lid, o);
                }
                if let Some(c) = seq.camera_layer_objects.remove(lid) {
                    moved_camera_objs.insert(*lid, c);
                }
                if let Some(b) = seq.scene_backdrops.remove(lid) {
                    moved_backdrops.insert(*lid, b);
                }
                if let Some(p) = seq.layer_3d_props.remove(lid) {
                    moved_layer_3d.insert(*lid, p);
                }
            }

            // Place the replacement PrecompRef clip without overlapping unselected clips,
            // falling onto a disabled track, shifting z-order across unrelated tracks, or
            // losing solo visibility when parent tracks are soloed.
            let precomp_end = min_start + span;
            let track_has_unselected_overlap = |vt: &zc_project::VideoTrack| {
                vt.clips.iter().any(|c| {
                    let c_end = c.start + c.length;
                    c_end > min_start + 1e-6 && c.start < precomp_end - 1e-6
                })
            };
            let needs_solo = parent_has_solo && any_selected_solo;
            let vt_idx = if let Some(ti) = target_vtrack_idx {
                let target_vt = &seq.video_tracks[ti];
                let can_reuse = target_vt.enabled
                    && !target_vt.locked
                    && !track_has_unselected_overlap(target_vt)
                    && (!needs_solo || target_vt.solo || target_vt.clips.is_empty());
                if can_reuse {
                    if needs_solo {
                        seq.video_tracks[ti].solo = true;
                    }
                    ti
                } else {
                    seq.video_tracks.insert(
                        ti,
                        zc_project::VideoTrack {
                            id: fresh_vtrack_ids[track_alloc_count_v],
                            name: format!("V{}", seq.video_tracks.len() + 1),
                            enabled: true,
                            solo: needs_solo,
                            locked: false,
                            is_3d: false,
                            clips: Vec::new(),
                            preserved: BTreeMap::new(),
                        },
                    );
                    ti
                }
            } else if let Some(ti) = seq.video_tracks.iter().position(|t| {
                t.enabled
                    && !t.locked
                    && !track_has_unselected_overlap(t)
                    && (!parent_has_solo || t.solo)
            }) {
                ti
            } else {
                seq.video_tracks.push(zc_project::VideoTrack {
                    id: fresh_vtrack_ids[track_alloc_count_v],
                    name: format!("V{}", seq.video_tracks.len() + 1),
                    enabled: true,
                    solo: needs_solo,
                    locked: false,
                    is_3d: false,
                    clips: Vec::new(),
                    preserved: BTreeMap::new(),
                });
                seq.video_tracks.len() - 1
            };

            let mut layer = zc_project::Layer::new(
                trimmed,
                zc_project::LayerKind::PrecompRef {
                    composition_id: comp_id,
                },
                resolution,
            );
            layer.id = precomp_layer_id;
            let mut time_curve = zc_project::PropertyTrack::linear_time_curve(span, fps);
            // Match `exact_fps` (e.g. 29.97 / 23.976 / 59.94) so the replacement PrecompRef
            // clip advances 1:1 with child frames without fractional-FPS playback drift.
            let exact_end_sec = span / effective_fps.max(1.0);
            if let Some(last_kf) = time_curve.keyframes.last_mut() {
                last_kf.value = zc_project::PropertyValue::Number(exact_end_sec);
            }
            let precomp_clip = zc_project::VideoClip {
                id: precomp_clip_id,
                name: trimmed.to_owned(),
                start: min_start,
                length: span,
                link: None,
                media: None,
                time_curve,
                layer,
                preserved: BTreeMap::new(),
            };
            seq.video_tracks[vt_idx].clips.push(precomp_clip);
            seq.video_tracks[vt_idx]
                .clips
                .sort_by(|a, b| a.start.total_cmp(&b.start));
        }

        // Preserve parent sequence controls inside the child composition so moved
        // expressions referencing controls by name or ID evaluate identically.
        for ctl in &mut child_controls {
            shift_property_track_keyframes(&mut ctl.track, -min_start);
            if min_start.abs() > 1e-9 {
                offset_property_track_expressions_time_origin(
                    &mut ctl.track,
                    min_start,
                    effective_fps,
                );
            }
        }

        if new_vtracks.is_empty() {
            new_vtracks.push(zc_project::VideoTrack {
                id: parse_uuid_or_new(&self.alloc_uuid_id()),
                name: "V1".to_owned(),
                enabled: true,
                solo: false,
                locked: false,
                is_3d: false,
                clips: Vec::new(),
                preserved: BTreeMap::new(),
            });
        }
        if new_atracks.is_empty() {
            new_atracks.push(zc_project::AudioTrack {
                id: parse_uuid_or_new(&self.alloc_uuid_id()),
                name: "A1".to_owned(),
                enabled: true,
                muted: false,
                solo: false,
                locked: false,
                clips: Vec::new(),
                preserved: BTreeMap::new(),
            });
        }

        let comp = zc_project::Sequence {
            id: comp_id,
            name: trimmed.to_owned(),
            resolution,
            fps,
            exact_fps: parent_exact_fps,
            length_frames: span,
            background_color: bg_color,
            controls: child_controls,
            video_tracks: new_vtracks,
            audio_tracks: new_atracks,
            layer_masks: moved_layer_masks,
            precomp_offsets: moved_precomp_offsets,
            camera_layer_objects: moved_camera_objs,
            scene_backdrops: moved_backdrops,
            layer_3d_props: moved_layer_3d,
            ..zc_project::Sequence::default()
        };
        self.project.compositions.insert(comp_id, comp);
        self.enable_plugin_requirement("precomp-plus");

        let active_id = self.document.active_composition_id;
        let active_seq = match active_id {
            Some(cid) => self
                .project
                .compositions
                .get(&cid)
                .unwrap_or(&self.project.sequence),
            None => &self.project.sequence,
        };
        self.document = SessionDocument::from_sequence(
            &self.project.name,
            &self.project.assets,
            &self.project.media,
            active_seq,
            active_id,
        );
        self.document.recompute_sequence_length_if_extended();
        self.active_sequence_mut().length_frames = self.document.sequence_length;
        self.selection.clear_all();
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;

        Ok(json!({
            "comp_id": comp_id.to_string(),
            "clip_id": precomp_clip_id.to_string(),
            "name": trimmed,
            "start": min_start,
            "length": span,
        }))
    }

    pub(crate) fn exec_set_active_composition(
        &mut self,
        comp_id: Option<Uuid>,
    ) -> Result<Value, SessionError> {
        // Flush any edits in the currently active composition before switching.
        self.sync_document_to_project();
        if let Some(id) = comp_id
            && !self.project.compositions.contains_key(&id)
        {
            return Err(SessionError::NotFound(format!("composition '{id}'")));
        }
        if comp_id.is_some() {
            self.enable_plugin_requirement("precomp-plus");
        }
        let target_seq = match comp_id {
            None => &self.project.sequence,
            Some(id) => self
                .project
                .compositions
                .get(&id)
                .ok_or_else(|| SessionError::NotFound(format!("composition '{id}'")))?,
        };
        let target_name = target_seq.name.clone();
        self.document = SessionDocument::from_sequence(
            &self.project.name,
            &self.project.assets,
            &self.project.media,
            target_seq,
            comp_id,
        );
        self.selection.clear_all();
        let max_frame = (self.document.sequence_length - 1.0).max(0.0);
        self.playback.current_frame = self.playback.current_frame.clamp(0.0, max_frame);
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "active_composition_id": comp_id.map(|u| u.to_string()),
            "name": target_name,
        }))
    }

    pub(crate) fn exec_set_precomp_clip_target(
        &mut self,
        clip_id: Uuid,
        composition_id: Uuid,
    ) -> Result<Value, SessionError> {
        if !self.project.compositions.contains_key(&composition_id) {
            return Err(SessionError::NotFound(format!(
                "composition '{composition_id}'"
            )));
        }
        if self.document.active_composition_id == Some(composition_id) {
            return Err(SessionError::CycleDetected(format!(
                "composition '{composition_id}' cannot reference itself"
            )));
        }
        let cid_str = clip_id.to_string();
        let (track, clip_idx) = self
            .document
            .find_clip_mut(&cid_str)
            .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
        if track.locked {
            return Err(crate::track_locked_error(&track.name, &track.id));
        }
        if track.kind != TrackKind::Video
            || track.clips[clip_idx].kind != "zoidium:precomp-plus/precomp"
        {
            return Err(SessionError::InvalidOperation(format!(
                "clip '{clip_id}' is not a PrecompRef clip"
            )));
        }
        track.clips[clip_idx].properties.insert(
            "compositionId".to_owned(),
            SessionProperty::static_value_with_kind(
                json!(composition_id.to_string()),
                SessionValueKind::Text,
            ),
        );
        let vc = find_video_clip_mut_checked(self.active_sequence_mut(), clip_id)?;
        vc.layer.kind = zc_project::LayerKind::PrecompRef { composition_id };
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": cid_str,
            "composition_id": composition_id.to_string(),
        }))
    }

    pub(crate) fn exec_update_layer_preserved(
        &mut self,
        clip_id: Uuid,
        key: String,
        value: Option<Value>,
    ) -> Result<Value, SessionError> {
        if key.trim().is_empty() {
            return Err(SessionError::InvalidArgument(
                "preserved key cannot be empty".to_owned(),
            ));
        }
        let vc = find_video_clip_mut_checked(self.active_sequence_mut(), clip_id)?;
        match &value {
            Some(v) => {
                vc.layer.preserved.insert(key.clone(), v.clone());
            }
            None => {
                vc.layer.preserved.remove(&key);
            }
        }
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "key": key,
            "value": value,
        }))
    }

    pub(crate) fn exec_set_preset_shape_kind(
        &mut self,
        clip_id: Uuid,
        shape_kind: zc_project::PresetShapeKind,
    ) -> Result<Value, SessionError> {
        let vc = find_video_clip_mut_checked(self.active_sequence_mut(), clip_id)?;
        match &mut vc.layer.kind {
            zc_project::LayerKind::PresetShape2D { preset, .. } => {
                *preset = shape_kind;
            }
            _ => {
                return Err(SessionError::InvalidOperation(format!(
                    "clip '{clip_id}' is not a PresetShape2D layer"
                )));
            }
        }
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "clip_id": clip_id.to_string(),
            "shape_kind": shape_kind,
        }))
    }

    pub(crate) fn exec_split_text_3d_into_letters(
        &mut self,
        clip_id: Uuid,
        object_id: Uuid,
    ) -> Result<Value, SessionError> {
        let cid_str = clip_id.to_string();
        let oid_str = object_id.to_string();

        let (orig_obj, orig_pos_idx) = {
            let (track, clip_idx) = self
                .document
                .find_clip_mut(&cid_str)
                .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
            if track.locked {
                return Err(crate::track_locked_error(&track.name, &track.id));
            }
            let clip = &track.clips[clip_idx];
            if track.kind != TrackKind::Video || clip.kind != "scene" {
                return Err(SessionError::InvalidOperation(format!(
                    "clip '{clip_id}' is not a Scene3D clip"
                )));
            }
            let pos = clip
                .objects_3d
                .iter()
                .position(|o| o.id == oid_str)
                .ok_or_else(|| {
                    SessionError::NotFound(format!("3D object '{object_id}' in clip '{clip_id}'"))
                })?;
            let obj = &clip.objects_3d[pos];
            if !matches!(obj.kind.as_str(), "text" | "text_3d" | "text3d") {
                return Err(SessionError::InvalidOperation(format!(
                    "3D object '{object_id}' is kind '{}', expected 'text_3d'",
                    obj.kind
                )));
            }
            (obj.clone(), pos)
        };

        let text = prop_str_static(&orig_obj.properties, "text").unwrap_or_default();
        let height = prop_f64_static(&orig_obj.properties, "height")
            .unwrap_or(20.0)
            .max(0.05);
        let spacing = prop_f64_static(&orig_obj.properties, "spacing").unwrap_or(0.0);
        let line_height = prop_f64_static(&orig_obj.properties, "lineHeight")
            .unwrap_or(1.2)
            .max(0.1);
        let position_mode = prop_f64_static(&orig_obj.properties, "positionMode")
            .map_or(0_i64, |v| v.round() as i64);
        let font_lower = prop_str_static(&orig_obj.properties, "font")
            .unwrap_or_default()
            .to_ascii_lowercase();
        let is_monospace = font_lower.contains("mono")
            || font_lower.contains("source-code-pro")
            || font_lower.contains("source code pro");

        // Compute per-line proportional character advance positions, preserving `\n` line breaks.
        let lines: Vec<&str> = text
            .split('\n')
            .map(|line| line.strip_suffix('\r').unwrap_or(line))
            .collect();
        let line_count = lines.len();
        let line_step = height * line_height;
        let vertical_base = match position_mode {
            3 => 0.0,                                                     // Top
            4 => (line_count.saturating_sub(1) as f64) * line_step,       // Bottom
            _ => (line_count.saturating_sub(1) as f64) * 0.5 * line_step, // Center & others
        };

        let mut placed_chars: Vec<(char, f64, f64)> = Vec::new();
        for (line_idx, line) in lines.iter().enumerate() {
            let line_chars: Vec<char> = line.chars().collect();
            let mut line_placed: Vec<(char, f64)> = Vec::new();
            let mut cursor = 0.0_f64;
            for (i, &ch) in line_chars.iter().enumerate() {
                let adv = proportional_char_advance_ratio(ch, is_monospace) * height;
                if !ch.is_whitespace() {
                    line_placed.push((ch, cursor + adv * 0.5));
                }
                cursor += adv;
                if i + 1 < line_chars.len() {
                    cursor += spacing;
                }
            }
            let line_width = cursor;
            let align_offset = match position_mode {
                1 => 0.0,               // Left
                2 => -line_width,       // Right
                _ => -line_width * 0.5, // Center (and Top/Bottom/Front/Back)
            };
            let line_y = vertical_base - (line_idx as f64) * line_step;
            for (ch, center_x) in line_placed {
                placed_chars.push((ch, center_x + align_offset, line_y));
            }
        }
        if placed_chars.is_empty() {
            return Err(SessionError::InvalidOperation(
                "cannot split Text3D with no non-whitespace characters".to_owned(),
            ));
        }

        let group_id = self.alloc_uuid_id();
        let mut group_props = BTreeMap::new();
        for key in ["enabled", "position", "rotation", "scale", "eulerOrder"] {
            if let Some(p) = orig_obj.properties.get(key) {
                group_props.insert(key.to_owned(), p.clone());
            }
        }
        group_props
            .entry("enabled".to_owned())
            .or_insert_with(|| SessionProperty::static_value(json!(1)));
        group_props
            .entry("position".to_owned())
            .or_insert_with(|| SessionProperty::static_value(json!([0.0, 0.0, 0.0])));
        group_props
            .entry("rotation".to_owned())
            .or_insert_with(|| SessionProperty::static_value(json!([0.0, 0.0, 0.0])));
        group_props
            .entry("scale".to_owned())
            .or_insert_with(|| SessionProperty::static_value(json!([1.0, 1.0, 1.0])));

        let group_obj = SessionObject3D {
            id: group_id.clone(),
            parent_id: orig_obj.parent_id.clone(),
            kind: "group".to_owned(),
            name: orig_obj.name.clone(),
            source_id: None,
            properties: group_props,
        };

        let mut letter_objs = Vec::with_capacity(placed_chars.len());
        let mut letter_ids = Vec::with_capacity(placed_chars.len());
        for (ch, pos_x, pos_y) in placed_chars {
            let letter_id = self.alloc_uuid_id();
            letter_ids.push(letter_id.clone());
            let mut lprops = orig_obj.properties.clone();
            lprops.insert(
                "text".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!(ch.to_string()),
                    SessionValueKind::Text,
                ),
            );
            lprops.insert(
                "position".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([pos_x, pos_y, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            lprops.insert(
                "rotation".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            lprops.insert(
                "scale".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([1.0, 1.0, 1.0]),
                    SessionValueKind::Vec3,
                ),
            );
            lprops.insert(
                "positionMode".to_owned(),
                SessionProperty::static_value_with_kind(json!(0), SessionValueKind::Option),
            );
            letter_objs.push(SessionObject3D {
                id: letter_id,
                parent_id: Some(group_id.clone()),
                kind: "text_3d".to_owned(),
                name: ch.to_string(),
                source_id: None,
                properties: lprops,
            });
        }

        let (track, clip_idx) = self
            .document
            .find_clip_mut(&cid_str)
            .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
        let clip = &mut track.clips[clip_idx];
        clip.objects_3d.remove(orig_pos_idx);
        clip.objects_3d.insert(orig_pos_idx, group_obj);
        for (offset, lo) in letter_objs.into_iter().enumerate() {
            clip.objects_3d.insert(orig_pos_idx + 1 + offset, lo);
        }
        self.selection.prune_object_3d(&oid_str);
        let _ = self
            .selection
            .select_object_3d(group_id.clone(), "group", false);
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": cid_str,
            "group_id": group_id,
            "letter_ids": letter_ids,
            "count": letter_ids.len(),
        }))
    }

    pub(crate) fn exec_add_marker(
        &mut self,
        frame: f64,
        name: String,
        color: Option<String>,
    ) -> Result<Value, SessionError> {
        if !frame.is_finite() || frame < 0.0 {
            return Err(SessionError::InvalidArgument(
                "marker frame must be finite and >= 0".to_owned(),
            ));
        }
        let seq = self.active_sequence_mut();
        let mut colors = sync_marker_colors_vec(seq);
        let idx = seq.markers.len();
        seq.markers.push(zc_project::Marker {
            frame,
            label: name.clone(),
        });
        colors.push(color.as_ref().map_or(Value::Null, |c| json!(c)));
        write_marker_colors_vec(seq, colors);
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "index": idx,
            "frame": frame,
            "name": name,
            "color": color,
        }))
    }

    pub(crate) fn exec_update_marker(
        &mut self,
        index: usize,
        frame: f64,
        name: String,
        color: Option<String>,
    ) -> Result<Value, SessionError> {
        if !frame.is_finite() || frame < 0.0 {
            return Err(SessionError::InvalidArgument(
                "marker frame must be finite and >= 0".to_owned(),
            ));
        }
        let seq = self.active_sequence_mut();
        if index >= seq.markers.len() {
            return Err(SessionError::NotFound(format!("marker index {index}")));
        }
        let mut colors = sync_marker_colors_vec(seq);
        seq.markers[index].frame = frame;
        seq.markers[index].label = name.clone();
        colors[index] = color.as_ref().map_or(Value::Null, |c| json!(c));
        write_marker_colors_vec(seq, colors);
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "index": index,
            "frame": frame,
            "name": name,
            "color": color,
        }))
    }

    pub(crate) fn exec_delete_marker(&mut self, index: usize) -> Result<Value, SessionError> {
        let seq = self.active_sequence_mut();
        if index >= seq.markers.len() {
            return Err(SessionError::NotFound(format!("marker index {index}")));
        }
        let mut colors = sync_marker_colors_vec(seq);
        let removed = seq.markers.remove(index);
        if index < colors.len() {
            colors.remove(index);
        }
        write_marker_colors_vec(seq, colors);
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "deleted_index": index,
            "frame": removed.frame,
            "name": removed.label,
        }))
    }

    pub(crate) fn exec_add_sequence_control(
        &mut self,
        name: String,
        kind: zc_project::SequenceControlKind,
        value: zc_project::PropertyValue,
        min: Option<f64>,
        max: Option<f64>,
        step: Option<f64>,
    ) -> Result<Value, SessionError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(SessionError::InvalidArgument(
                "sequence control name cannot be empty".to_owned(),
            ));
        }
        if self.active_sequence().controls.len() >= 256 {
            return Err(SessionError::InvalidOperation(
                "maximum of 256 sequence controls reached".to_owned(),
            ));
        }
        let control_id = format!("ctl-{}", self.alloc_uuid_id());
        let mut preserved = BTreeMap::new();
        if let Some(v) = min.filter(|x| x.is_finite()) {
            preserved.insert("min".to_owned(), json!(v));
        }
        if let Some(v) = max.filter(|x| x.is_finite()) {
            preserved.insert("max".to_owned(), json!(v));
        }
        if let Some(v) = step.filter(|x| x.is_finite()) {
            preserved.insert("step".to_owned(), json!(v));
        }
        self.active_sequence_mut()
            .controls
            .push(zc_project::SequenceControl {
                id: control_id.clone(),
                name: trimmed.to_owned(),
                kind,
                track: zc_project::PropertyTrack::constant(value),
                preserved,
            });
        refresh_sequence_controls_in_document(self);
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "control_id": control_id,
            "name": trimmed,
            "kind": kind,
        }))
    }

    pub(crate) fn exec_update_sequence_control(
        &mut self,
        control_id: String,
        name: String,
        value: zc_project::PropertyValue,
        min: Option<f64>,
        max: Option<f64>,
        step: Option<f64>,
    ) -> Result<Value, SessionError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(SessionError::InvalidArgument(
                "sequence control name cannot be empty".to_owned(),
            ));
        }
        let ctl = self
            .active_sequence_mut()
            .controls
            .iter_mut()
            .find(|c| c.id == control_id)
            .ok_or_else(|| SessionError::NotFound(format!("sequence control '{control_id}'")))?;
        ctl.name = trimmed.to_owned();
        ctl.track.value = value.clone();
        if let Some(first) = ctl.track.keyframes.first_mut() {
            first.value = value;
        }
        match min.filter(|x| x.is_finite()) {
            Some(v) => {
                ctl.preserved.insert("min".to_owned(), json!(v));
            }
            None => {
                ctl.preserved.remove("min");
            }
        }
        match max.filter(|x| x.is_finite()) {
            Some(v) => {
                ctl.preserved.insert("max".to_owned(), json!(v));
            }
            None => {
                ctl.preserved.remove("max");
            }
        }
        match step.filter(|x| x.is_finite()) {
            Some(v) => {
                ctl.preserved.insert("step".to_owned(), json!(v));
            }
            None => {
                ctl.preserved.remove("step");
            }
        }
        refresh_sequence_controls_in_document(self);
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "control_id": control_id,
            "name": trimmed,
        }))
    }

    pub(crate) fn exec_delete_sequence_control(
        &mut self,
        control_id: String,
    ) -> Result<Value, SessionError> {
        let seq = self.active_sequence_mut();
        let pos = seq
            .controls
            .iter()
            .position(|c| c.id == control_id)
            .ok_or_else(|| SessionError::NotFound(format!("sequence control '{control_id}'")))?;
        seq.controls.remove(pos);
        refresh_sequence_controls_in_document(self);
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({ "deleted_control_id": control_id }))
    }

    pub(crate) fn exec_reorder_sequence_control(
        &mut self,
        control_id: String,
        to_index: usize,
    ) -> Result<Value, SessionError> {
        let seq = self.active_sequence_mut();
        let pos = seq
            .controls
            .iter()
            .position(|c| c.id == control_id)
            .ok_or_else(|| SessionError::NotFound(format!("sequence control '{control_id}'")))?;
        let item = seq.controls.remove(pos);
        let insert_at = to_index.min(seq.controls.len());
        seq.controls.insert(insert_at, item);
        refresh_sequence_controls_in_document(self);
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "control_id": control_id,
            "index": insert_at,
        }))
    }

    pub(crate) fn exec_paste_clips(
        &mut self,
        clips: Vec<ClipSnapshot>,
        target_frame: f64,
    ) -> Result<Value, SessionError> {
        if !target_frame.is_finite() || target_frame < 0.0 {
            return Err(SessionError::InvalidArgument(
                "target_frame must be finite and >= 0".to_owned(),
            ));
        }
        if clips.is_empty() {
            return Err(SessionError::InvalidArgument(
                "clips cannot be empty".to_owned(),
            ));
        }
        // Flush current document to active sequence so any existing clips/layers are intact.
        self.sync_document_to_project();

        let min_start = clips
            .iter()
            .map(|c| c.clip.start)
            .fold(f64::INFINITY, f64::min);
        let base_start = if min_start.is_finite() {
            min_start
        } else {
            0.0
        };

        let mut pasted_ids = Vec::with_capacity(clips.len());
        let mut affected_tracks: BTreeMap<(u8, usize), BTreeSet<String>> = BTreeMap::new();
        let mut pending_side_tables: Vec<(Uuid, Vec<zc_project::LayerMask>, Option<f64>)> =
            Vec::new();

        for snap in clips {
            let new_start = (target_frame + (snap.clip.start - base_start)).max(0.0);
            let track_idx = match snap.track_kind {
                TrackKind::Video => {
                    let matched = snap.track_id.as_deref().and_then(|tid| {
                        self.document
                            .video_tracks
                            .iter()
                            .position(|t| t.id == tid && !t.locked)
                    });
                    if let Some(idx) = matched {
                        idx
                    } else if let Some(idx) =
                        self.document.video_tracks.iter().position(|t| !t.locked)
                    {
                        idx
                    } else {
                        let tid = self.alloc_uuid_id();
                        let name = format!("V{}", self.document.video_tracks.len() + 1);
                        self.document.video_tracks.push(SessionTrack {
                            id: tid,
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
                }
                TrackKind::Audio => {
                    let matched = snap.track_id.as_deref().and_then(|tid| {
                        self.document
                            .audio_tracks
                            .iter()
                            .position(|t| t.id == tid && !t.locked)
                    });
                    if let Some(idx) = matched {
                        idx
                    } else if let Some(idx) =
                        self.document.audio_tracks.iter().position(|t| !t.locked)
                    {
                        idx
                    } else {
                        let tid = self.alloc_uuid_id();
                        let name = format!("A{}", self.document.audio_tracks.len() + 1);
                        self.document.audio_tracks.push(SessionTrack {
                            id: tid,
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
            };

            let mut sclip: SessionClip = snap.clip;
            let new_clip_id = self.alloc_uuid_id();
            sclip.id = new_clip_id.clone();
            sclip.source_clip_id = None;
            sclip.start = new_start;
            self.remap_cloned_clip_children(&mut sclip);

            // Seed matching project clip + side tables so sync_document_to_project preserves all lossless fields.
            let new_clip_uuid = parse_uuid_or_new(&new_clip_id);
            match snap.track_kind {
                TrackKind::Video => {
                    if let Some(mut vc) = snap.video_clip {
                        vc.id = new_clip_uuid;
                        vc.link = None;
                        vc.start = new_start;
                        let mut alloc = || parse_uuid_or_new(&self.alloc_uuid_id());
                        regenerate_cloned_layer_ids(&mut vc.layer, true, &mut alloc);
                        let new_layer_id = vc.layer.id;
                        let seq = self.active_sequence_mut();
                        if !snap.layer_masks.is_empty() {
                            seq.layer_masks.insert(new_layer_id, snap.layer_masks);
                        }
                        if let Some(off) = snap.precomp_offset {
                            seq.precomp_offsets.insert(new_layer_id, off);
                        }
                        if let Some(vt) = seq.video_tracks.get_mut(track_idx) {
                            vt.clips.push(vc);
                        } else if let Some(vt) = seq.video_tracks.first_mut() {
                            vt.clips.push(vc);
                        }
                    } else if !snap.layer_masks.is_empty() || snap.precomp_offset.is_some() {
                        pending_side_tables.push((
                            new_clip_uuid,
                            snap.layer_masks,
                            snap.precomp_offset,
                        ));
                    }
                    self.document.video_tracks[track_idx].clips.push(sclip);
                    affected_tracks
                        .entry((0, track_idx))
                        .or_default()
                        .insert(new_clip_id.clone());
                }
                TrackKind::Audio => {
                    if let Some(mut ac) = snap.audio_clip {
                        ac.id = new_clip_uuid;
                        ac.link = None;
                        ac.start = new_start;
                        let seq = self.active_sequence_mut();
                        if let Some(at) = seq.audio_tracks.get_mut(track_idx) {
                            at.clips.push(ac);
                        } else if let Some(at) = seq.audio_tracks.first_mut() {
                            at.clips.push(ac);
                        }
                    }
                    self.document.audio_tracks[track_idx].clips.push(sclip);
                    affected_tracks
                        .entry((1, track_idx))
                        .or_default()
                        .insert(new_clip_id.clone());
                }
            }
            pasted_ids.push(new_clip_id);
        }

        for ((kind_tag, track_idx), group_ids) in affected_tracks {
            let kind = if kind_tag == 0 {
                TrackKind::Video
            } else {
                TrackKind::Audio
            };
            self.resolve_track_overlaps_for_group(kind, track_idx, &group_ids);
        }
        if !pending_side_tables.is_empty() {
            // Materialize newly pasted video clips that lacked a pre-built `snap.video_clip`
            // so their `layer.id` exists and receives `snap.layer_masks` & `snap.precomp_offset`.
            self.sync_document_to_project();
            let mut needs_shape_plus = false;
            let mut needs_precomp_plus = false;
            for (clip_uuid, masks, precomp_offset) in pending_side_tables {
                let layer_id_opt = self
                    .active_sequence()
                    .video_tracks
                    .iter()
                    .flat_map(|vt| vt.clips.iter())
                    .find(|c| c.id == clip_uuid)
                    .map(|c| c.layer.id);
                if let Some(layer_id) = layer_id_opt {
                    let seq = self.active_sequence_mut();
                    if !masks.is_empty() {
                        seq.layer_masks.insert(layer_id, masks);
                        needs_shape_plus = true;
                    }
                    if let Some(off) = precomp_offset {
                        seq.precomp_offsets.insert(layer_id, off);
                        needs_precomp_plus = true;
                    }
                }
            }
            if needs_shape_plus {
                let _ = self.enable_plugin_requirement("shape-plus");
            }
            if needs_precomp_plus {
                let _ = self.enable_plugin_requirement("precomp-plus");
            }
        }
        self.document.recompute_sequence_length_if_extended();
        self.playback.bump_epoch();
        Ok(json!({ "clip_ids": pasted_ids }))
    }

    pub(crate) fn exec_freeze_clip_frame(
        &mut self,
        clip_id: Uuid,
        frame: f64,
    ) -> Result<Value, SessionError> {
        if !frame.is_finite() {
            return Err(SessionError::InvalidArgument(
                "freeze frame must be finite".to_owned(),
            ));
        }
        let cid_str = clip_id.to_string();
        let (local_frame, base_time, time_prop_snapshot, time_expr) = {
            let (track, clip_idx) = self
                .document
                .find_clip_mut(&cid_str)
                .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
            if track.locked {
                return Err(crate::track_locked_error(&track.name, &track.id));
            }
            let clip = &track.clips[clip_idx];
            let local_frame = (frame - clip.start).clamp(0.0, clip.length);
            let base_time = if clip.time_animated {
                clip.properties
                    .get("time")
                    .and_then(|p| eval_session_property_at(p, local_frame).as_f64())
                    .unwrap_or_else(|| clip.eval_time_at(local_frame))
            } else {
                clip.eval_time_at(local_frame)
            };
            let time_prop = clip.properties.get("time").cloned();
            let expr = time_prop
                .as_ref()
                .and_then(|p| p.expression.as_deref())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            (local_frame, base_time, time_prop, expr)
        };

        let frozen_time = if let Some(expr_src) = time_expr {
            let config = if self.plugins.is_enabled("expressions-plus") {
                zc_expr::EvalConfig::expressions_plus()
            } else {
                zc_expr::EvalConfig::vanilla()
            };
            self.sync_document_to_project();
            let active_seq = self.active_sequence();
            let fps = active_seq
                .exact_fps
                .unwrap_or_else(|| f64::from(active_seq.fps.max(1)))
                .max(1.0);
            let active_vclip = active_seq
                .video_tracks
                .iter()
                .flat_map(|vt| vt.clips.iter())
                .find(|c| c.id == clip_id);
            let state = FreezeExprState {
                project: &self.project,
                sequence: active_seq,
                abs_frame: frame,
                fps,
                config: &config,
                control_cache: RefCell::new(HashMap::new()),
                active_stack: RefCell::new(HashSet::new()),
            };
            let self_prop_id = zc_expr::HandleId::new(format!("prop:clip:{clip_id}:time"));
            let ctx = FreezeTimeExprContext {
                state: &state,
                frame: local_frame,
                time: base_time,
                base_value: zc_expr::Value::Number(base_time),
                self_prop_id,
                self_track: active_vclip.map(|c| &c.time_curve),
                time_prop_fallback: time_prop_snapshot.as_ref(),
                active_clip_id: Some(clip_id),
                active_layer_id: active_vclip.map(|c| c.layer.id),
                active_object_id: None,
                active_effect_id: None,
            };
            let outcome = zc_expr::eval_with_fallback(&expr_src, &ctx, &config);
            if outcome.error.is_none() {
                let evaluated = outcome.value.to_number();
                if evaluated.is_finite() {
                    evaluated
                } else {
                    base_time
                }
            } else {
                base_time
            }
        } else {
            base_time
        };

        let (track, clip_idx) = self
            .document
            .find_clip_mut(&cid_str)
            .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
        let clip = &mut track.clips[clip_idx];
        clip.time_start = frozen_time;
        clip.time_end = frozen_time;
        clip.time_animated = true;
        clip.properties.insert(
            "time".to_owned(),
            SessionProperty {
                animated: false,
                expression: None,
                default_value: json!(frozen_time),
                keyframes: vec![SessionKeyframe::new(0.0, json!(frozen_time), 1)],
            },
        );
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": cid_str,
            "frozen_time": frozen_time,
        }))
    }

    pub(crate) fn exec_reverse_clip(&mut self, clip_id: Uuid) -> Result<Value, SessionError> {
        let cid_str = clip_id.to_string();
        let (track, clip_idx) = self
            .document
            .find_clip_mut(&cid_str)
            .ok_or_else(|| SessionError::NotFound(format!("clip '{clip_id}'")))?;
        if track.locked {
            return Err(crate::track_locked_error(&track.name, &track.id));
        }
        let clip = &mut track.clips[clip_idx];
        if !clip.time_animated {
            std::mem::swap(&mut clip.time_start, &mut clip.time_end);
            clip.sync_time_property();
        } else {
            let len = clip.length;
            if let Some(tp) = clip.properties.get_mut("time") {
                reverse_session_keyframes(&mut tp.keyframes, len);
            }
            std::mem::swap(&mut clip.time_start, &mut clip.time_end);
        }
        let ts = clip.time_start;
        let te = clip.time_end;
        self.playback.bump_epoch();
        Ok(json!({
            "clip_id": cid_str,
            "time_start": ts,
            "time_end": te,
        }))
    }

    pub(crate) fn exec_relink_asset(
        &mut self,
        asset_id: String,
        filename: String,
        bytes: Vec<u8>,
    ) -> Result<Value, SessionError> {
        let trimmed = filename.trim();
        if trimmed.is_empty() {
            return Err(SessionError::InvalidArgument(
                "relink filename cannot be empty".to_owned(),
            ));
        }
        let rec = self
            .project
            .assets
            .get_mut(&asset_id)
            .ok_or_else(|| SessionError::NotFound(format!("asset '{asset_id}'")))?;
        let sha256 = zc_project::sha256_hex(&bytes);
        let size = bytes.len() as u64;
        rec.filename = trimmed.to_owned();
        rec.sha256 = sha256.clone();
        rec.size = size;
        rec.missing = false;
        rec.external = false;
        rec.bytes = Some(bytes);
        self.playback.bump_epoch();
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "asset_id": asset_id,
            "filename": trimmed,
            "sha256": sha256,
            "size": size,
        }))
    }

    pub(crate) fn exec_purge_unused_assets(&mut self) -> Result<Value, SessionError> {
        self.sync_document_to_project();
        let known_assets: BTreeSet<String> = self.project.assets.keys().cloned().collect();
        let mut referenced: BTreeSet<String> = BTreeSet::new();
        collect_sequence_asset_refs(&self.project.sequence, &known_assets, &mut referenced);
        for comp in self.project.compositions.values() {
            collect_sequence_asset_refs(comp, &known_assets, &mut referenced);
        }
        let mut purged: Vec<String> = Vec::new();
        self.project.assets.retain(|id, _| {
            if referenced.contains(id) {
                true
            } else {
                purged.push(id.clone());
                false
            }
        });
        let purged_set: BTreeSet<&str> = purged.iter().map(String::as_str).collect();
        self.project.media.retain(|m| {
            m.asset_id
                .as_deref()
                .is_none_or(|aid| !purged_set.contains(aid))
        });
        self.doc_already_synced_with_project = true;
        Ok(json!({
            "purged_count": purged.len(),
            "purged_asset_ids": purged,
        }))
    }
}
