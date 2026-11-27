use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use crate::Session;
use crate::document::{
    SessionClip, SessionEffect, SessionObject3D, SessionProperty, SessionValueKind,
};
use crate::plugin_defaults::default_shape2d_rectangle_path;
use crate::pro_commands::clone_layer_masks_shifted;
use crate::property_conv::{
    find_shape2d_fill_mut, find_shape2d_stroke_size_mut, parse_preserved_bool_u64,
    parse_preserved_rgba, prop_f64_static, prop_str_static, prop_vec3_static, prop_vec4_static,
    property_track_matches_session, session_property_to_track,
};
use crate::sync_from_project::parse_uuid_or_new;
use crate::sync_objects_to_project::{collect_existing_objects, rebuild_object_tree};

struct ClipSyncContext<'a> {
    resolution: [u32; 2],
    fps: u32,
    seq_bg: [f32; 4],
    existing_scene_backdrops: &'a BTreeMap<zc_project::Uuid, zc_project::SceneBackdrop>,
    existing_layer_3d_props: &'a BTreeMap<zc_project::Uuid, zc_project::Layer3DProps>,
    existing_camera_layer_objects: &'a BTreeMap<zc_project::Uuid, Vec<zc_project::Object3D>>,
    effect_group_children: &'a BTreeMap<String, Vec<SessionEffect>>,
    property_metadata: &'a BTreeMap<String, BTreeMap<String, BTreeMap<String, Value>>>,
}

#[derive(Default)]
struct ClipSideTableUpdates {
    precomp_offsets: Vec<(zc_project::Uuid, f64)>,
    scene_backdrops: Vec<(zc_project::Uuid, zc_project::SceneBackdrop)>,
    layer_3d_props: Vec<(zc_project::Uuid, zc_project::Layer3DProps)>,
    camera_layer_objects: Vec<(zc_project::Uuid, Vec<zc_project::Object3D>)>,
}

impl Session {
    /// Synchronizes all edits from `self.document` directly into `self.project` in place
    /// without lossy JSON round-tripping and without disturbing embedded asset binary buffers.
    pub(crate) fn sync_document_to_project(&mut self) {
        self.project.name = self.document.project_name.clone();
        let active_comp_id = self
            .document
            .active_composition_id
            .filter(|id| self.project.compositions.contains_key(id));

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

        let pending_shifts = std::mem::take(&mut self.pending_clip_side_table_shifts);
        let mut new_compositions_to_insert: Vec<(zc_project::Uuid, zc_project::Sequence)> =
            Vec::new();

        {
            let target_seq: &mut zc_project::Sequence = match active_comp_id {
                Some(comp_id) => self
                    .project
                    .compositions
                    .get_mut(&comp_id)
                    .expect("filtered active_comp_id"),
                None => &mut self.project.sequence,
            };

            target_seq.resolution = self.document.resolution;
            let next_fps = self.document.fps.round().clamp(1.0, 240.0) as u32;
            if target_seq.fps != next_fps {
                target_seq.fps = next_fps;
                target_seq.exact_fps = None;
            }
            target_seq.length_frames = self.document.sequence_length;

            // Sync sequence-level motion blur & Expressions+ sequence controls.
            if let Some(v) =
                prop_f64_static(&self.document.sequence_properties, "motionBlur.enabled")
            {
                target_seq.motion_blur.enabled = v.round() as i64 != 0;
            }
            if let Some(p) = self.document.sequence_properties.get("motionBlur.samples") {
                target_seq.motion_blur.samples =
                    session_property_to_track(p, Some(&target_seq.motion_blur.samples));
            }
            if let Some(p) = self.document.sequence_properties.get("motionBlur.shutter") {
                target_seq.motion_blur.shutter =
                    session_property_to_track(p, Some(&target_seq.motion_blur.shutter));
            }
            let mut existing_ctls: Vec<Option<zc_project::SequenceControl>> =
                target_seq.controls.drain(..).map(Some).collect();
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
            target_seq.controls = synced_ctls;

            let fps_u32 = target_seq.fps.max(1);
            let resolution = self.document.resolution;
            let seq_bg = target_seq.background_color;
            let seq_len = target_seq.length_frames.max(1.0);

            // Index existing tracks and clips by UUID string so preserved metadata and
            // source-clip templates (for Split / Duplicate / Overwrite straddle splits) survive sync.
            let mut existing_vtracks: BTreeMap<String, zc_project::VideoTrack> = target_seq
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
            let needs_vclip_templates = self
                .document
                .video_tracks
                .iter()
                .any(|st| st.clips.iter().any(|sc| sc.source_clip_id.is_some()));
            let vclip_templates = needs_vclip_templates.then(|| existing_vclips.clone());

            let mut cloned_layer_pairs: Vec<(zc_project::Uuid, zc_project::Uuid, f64)> = Vec::new();
            let mut side_updates = ClipSideTableUpdates::default();
            let mut synced_vtracks = Vec::with_capacity(self.document.video_tracks.len());
            let mut live_clip_ids: BTreeSet<zc_project::Uuid> = BTreeSet::new();
            let sync_ctx = ClipSyncContext {
                resolution,
                fps: fps_u32,
                seq_bg,
                existing_scene_backdrops: &target_seq.scene_backdrops,
                existing_layer_3d_props: &target_seq.layer_3d_props,
                existing_camera_layer_objects: &target_seq.camera_layer_objects,
                effect_group_children: &self.document.effect_group_children,
                property_metadata: &self.document.property_metadata,
            };
            let track_matte_map = &self.document.track_matte;
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
                if let Some(matte_props) = track_matte_map.get(&strack.id) {
                    for (canon, snake) in [
                        ("trackMatte", "track_matte"),
                        ("matteType", "matte_type"),
                        ("matteInvert", "matte_invert"),
                    ] {
                        if let Some(val) = matte_props.get(canon) {
                            if val.is_null() || val.as_str().is_some_and(str::is_empty) {
                                vt.preserved.remove(canon);
                                vt.preserved.remove(snake);
                            } else {
                                vt.preserved.remove(snake);
                                vt.preserved.insert(canon.to_owned(), val.clone());
                            }
                        } else {
                            vt.preserved.remove(canon);
                            vt.preserved.remove(snake);
                        }
                    }
                } else {
                    for k in [
                        "trackMatte",
                        "track_matte",
                        "matteType",
                        "matte_type",
                        "matteInvert",
                        "matte_invert",
                    ] {
                        vt.preserved.remove(k);
                    }
                }
                vt.clips = strack
                    .clips
                    .iter_mut()
                    .map(|sclip| {
                        let base = existing_vclips.remove(&sclip.id).or_else(|| {
                            vclip_templates.as_ref().and_then(|templates| {
                                sclip
                                    .source_clip_id
                                    .as_deref()
                                    .and_then(|src_id| templates.get(src_id))
                                    .map(|tpl| {
                                        let old_layer_id = tpl.layer.id;
                                        let mut cloned = tpl.clone();
                                        cloned.id = parse_uuid_or_new(&sclip.id);
                                        cloned.link = None;
                                        regenerate_cloned_layer_ids(
                                            &mut cloned.layer,
                                            true,
                                            &mut alloc_uuid,
                                        );
                                        let shift =
                                            pending_shifts.get(&sclip.id).copied().unwrap_or(0.0);
                                        cloned_layer_pairs.push((
                                            old_layer_id,
                                            cloned.layer.id,
                                            shift,
                                        ));
                                        cloned
                                    })
                            })
                        });
                        sclip.source_clip_id = None;
                        let mut synced = sync_video_clip_from_session(
                            sclip,
                            base,
                            &sync_ctx,
                            &mut side_updates,
                            &mut alloc_uuid,
                        );
                        if let zc_project::LayerKind::PrecompRef {
                            ref mut composition_id,
                        } = synced.layer.kind
                        {
                            if *composition_id == zc_project::Uuid::nil() {
                                let new_comp_id = alloc_uuid();
                                let v1_id = alloc_uuid();
                                let a1_id = alloc_uuid();
                                let comp = zc_project::Sequence {
                                    id: new_comp_id,
                                    name: synced.name.clone(),
                                    resolution,
                                    fps: fps_u32,
                                    length_frames: seq_len,
                                    background_color: seq_bg,
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
                                *composition_id = new_comp_id;
                                sclip.properties.insert(
                                    "compositionId".to_owned(),
                                    SessionProperty::static_value_with_kind(
                                        json!(new_comp_id.to_string()),
                                        SessionValueKind::Text,
                                    ),
                                );
                                new_compositions_to_insert.push((new_comp_id, comp));
                            }
                            if let Some(off) = prop_f64_static(&sclip.properties, "precompOffset") {
                                side_updates.precomp_offsets.push((synced.layer.id, off));
                            }
                        }
                        live_clip_ids.insert(synced.id);
                        synced
                    })
                    .collect();
                synced_vtracks.push(vt);
            }
            target_seq.video_tracks = synced_vtracks;

            // Propagate sequence side-tables (`layer_masks`, `precomp_offsets`, etc.) from source
            // clip layers onto split / duplicated clip layers (shifting mask keyframes on split!).
            for (old_lid, new_lid, shift) in cloned_layer_pairs {
                if let Some(masks) = target_seq.layer_masks.get(&old_lid).cloned() {
                    target_seq
                        .layer_masks
                        .insert(new_lid, clone_layer_masks_shifted(&masks, shift));
                }
                if let Some(&offset) = target_seq.precomp_offsets.get(&old_lid) {
                    target_seq.precomp_offsets.insert(new_lid, offset);
                }
                if let Some(bd) = target_seq.scene_backdrops.get(&old_lid).cloned() {
                    target_seq.scene_backdrops.insert(new_lid, bd);
                }
                if let Some(p3d) = target_seq.layer_3d_props.get(&old_lid).cloned() {
                    target_seq.layer_3d_props.insert(new_lid, p3d);
                }
                if let Some(cam_objs) = target_seq.camera_layer_objects.get(&old_lid).cloned() {
                    target_seq.camera_layer_objects.insert(new_lid, cam_objs);
                }
            }
            for (lid, off) in side_updates.precomp_offsets {
                target_seq.precomp_offsets.insert(lid, off);
            }
            for (lid, bd) in side_updates.scene_backdrops {
                target_seq.scene_backdrops.insert(lid, bd);
            }
            for (lid, p3d) in side_updates.layer_3d_props {
                target_seq.layer_3d_props.insert(lid, p3d);
            }
            for (lid, cam_objs) in side_updates.camera_layer_objects {
                target_seq.camera_layer_objects.insert(lid, cam_objs);
            }

            let mut existing_atracks: BTreeMap<String, zc_project::AudioTrack> = target_seq
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
            let needs_aclip_templates = self
                .document
                .audio_tracks
                .iter()
                .any(|st| st.clips.iter().any(|sc| sc.source_clip_id.is_some()));
            let aclip_templates = needs_aclip_templates.then(|| existing_aclips.clone());

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
                            aclip_templates.as_ref().and_then(|templates| {
                                sclip
                                    .source_clip_id
                                    .as_deref()
                                    .and_then(|src_id| templates.get(src_id))
                                    .map(|tpl| {
                                        let mut cloned = tpl.clone();
                                        cloned.id = parse_uuid_or_new(&sclip.id);
                                        cloned.link = None;
                                        cloned
                                    })
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
            target_seq.audio_tracks = synced_atracks;

            // Prune stale clip UUIDs (e.g. engulfed or deleted clips) from clip_links.
            target_seq.clip_links.retain(|_, members| {
                members.retain(|id| live_clip_ids.contains(id));
                !members.is_empty()
            });
        }

        for (cid, comp) in new_compositions_to_insert {
            self.project.compositions.insert(cid, comp);
        }
        self.next_id = next_id_counter;
        self.used_ids = used_ids_local;
        self.refresh_plugin_requirements_from_document();
    }
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

pub(crate) fn regenerate_cloned_layer_ids(
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

fn clamp_track_f64_range(track: &mut zc_project::PropertyTrack, min: f64, max: f64) {
    if let zc_project::PropertyValue::Number(ref mut n) = track.value {
        *n = n.clamp(min, max);
    }
    for kf in &mut track.keyframes {
        if let zc_project::PropertyValue::Number(ref mut n) = kf.value {
            *n = n.clamp(min, max);
        }
    }
}

fn parse_compat_mode_from_props(
    props: &BTreeMap<String, SessionProperty>,
    key: &str,
) -> Option<zc_project::CompatMode> {
    if let Some(s) = prop_str_static(props, key) {
        return Some(zc_project::CompatMode::from_token(&s));
    }
    prop_f64_static(props, key).map(|n| {
        if n.round() as i64 != 0 {
            zc_project::CompatMode::Legacy
        } else {
            zc_project::CompatMode::Current
        }
    })
}

fn sync_camera_data_from_session(
    props: &BTreeMap<String, SessionProperty>,
    c: &mut zc_project::CameraData,
) {
    if let Some(ck) = prop_f64_static(props, "camera.kind") {
        c.kind = if ck.round() as i64 == 1 {
            zc_project::CameraKind::Orthographic
        } else {
            zc_project::CameraKind::Perspective
        };
    }
    if let Some(p) = props.get("fov") {
        c.fov = session_property_to_track(p, Some(&c.fov));
    }
    if let Some(v) = prop_f64_static(props, "near") {
        c.near = v.max(1e-4);
    }
    if let Some(v) = prop_f64_static(props, "far") {
        c.far = v.max(c.near + 0.1);
    }
    if let Some(p) = props.get("zoom") {
        c.zoom = session_property_to_track(p, Some(&c.zoom));
    }
    if let Some(v) = prop_f64_static(props, "active") {
        c.active = v.round() as i64 != 0;
    }
    if let Some(p) = props.get("focalLength") {
        c.focal_length = session_property_to_track(p, Some(&c.focal_length));
    }
    if let Some(v) = prop_f64_static(props, "filmGate")
        && (v - c.film_gate).abs() > 1e-12
    {
        c.film_gate = v.clamp(1.0, 500.0);
    }
    if let Some(p) = props.get("filmOffsetX") {
        c.film_offset_x = session_property_to_track(p, Some(&c.film_offset_x));
    }
    if let Some(p) = props.get("filmOffsetY") {
        c.film_offset_y = session_property_to_track(p, Some(&c.film_offset_y));
    }
    if props.keys().any(|k| k.starts_with("motionBlur.")) {
        let mb = c.motion_blur.get_or_insert_with(Default::default);
        if let Some(v) = prop_f64_static(props, "motionBlur.enabled") {
            mb.enabled = v.round() as i64 != 0;
        }
        if let Some(v) = prop_f64_static(props, "motionBlur.samples") {
            mb.samples = v.round().clamp(1.0, 64.0) as u32;
        }
        if let Some(p) = props.get("motionBlur.shutter") {
            mb.shutter = session_property_to_track(p, Some(&mb.shutter));
        }
    }
    if let Some(p) = props.get("shakeAmplitude") {
        c.shake_amplitude = session_property_to_track(p, Some(&c.shake_amplitude));
    }
    if let Some(p) = props.get("shakeSpeed") {
        c.shake_speed = session_property_to_track(p, Some(&c.shake_speed));
    }
    if let Some(v) = prop_f64_static(props, "shake.mode") {
        c.shake.mode = v.round().clamp(0.0, 1.0) as u32;
    }
    if let Some(v) = prop_vec4_static(props, "shake.amplitude") {
        c.shake.amplitude = v;
    }
    if let Some(v) = prop_vec4_static(props, "shake.frequency") {
        c.shake.frequency = v;
    }
    if let Some(v) = prop_vec4_static(props, "shake.phase") {
        c.shake.phase = v;
    }
    if let Some(v) = prop_vec4_static(props, "shake.noise") {
        c.shake.noise = v;
    }
    if let Some(v) = prop_vec4_static(props, "shake.smooth") {
        c.shake.smooth = v;
    }
    if props.contains_key("dofEnabled")
        || props.contains_key("dofFocusDistance")
        || props.contains_key("dofAperture")
        || props.contains_key("dofFocusAreaWidth")
        || props.contains_key("dofNearBlur")
        || props.contains_key("dofFarBlur")
    {
        let was_none = c.dof.is_none();
        let dof = c.dof.get_or_insert_with(Default::default);
        if let Some(p) = props.get("dofEnabled") {
            if let Some(v) = prop_f64_static(props, "dofEnabled") {
                dof.enabled = v.round() as i64 != 0;
            }
            if dof.enabled_track.is_some() || p.animated || p.expression.is_some() {
                dof.enabled_track = Some(session_property_to_track(p, dof.enabled_track.as_ref()));
            }
        } else if was_none {
            dof.enabled = true;
        }
        if let Some(p) = props.get("dofFocusDistance") {
            dof.focus_distance = session_property_to_track(p, Some(&dof.focus_distance));
        }
        if let Some(p) = props.get("dofAperture") {
            let changed = !property_track_matches_session(&dof.aperture, p);
            dof.aperture = session_property_to_track(p, Some(&dof.aperture));
            if changed {
                clamp_track_f64_range(&mut dof.aperture, 0.0, 40.0);
            }
        }
        if let Some(p) = props.get("dofFocusAreaWidth") {
            dof.focus_area_width = session_property_to_track(p, Some(&dof.focus_area_width));
        }
        if let Some(p) = props.get("dofNearBlur") {
            dof.near_blur = session_property_to_track(p, Some(&dof.near_blur));
        }
        if let Some(p) = props.get("dofFarBlur") {
            dof.far_blur = session_property_to_track(p, Some(&dof.far_blur));
        }
    }
    if props.keys().any(|k| k.starts_with("vibrate.")) {
        let vib = c.vibrate.get_or_insert_with(Default::default);
        if let Some(v) = prop_f64_static(props, "vibrate.enabled") {
            vib.enabled = v.round() as i64 != 0;
        }
        if let Some(v) = prop_f64_static(props, "vibrate.seed") {
            vib.seed = v.round().max(0.0) as u32;
        }
        if let Some(p) = props.get("vibrate.positionAmplitude") {
            vib.position_amplitude = session_property_to_track(p, Some(&vib.position_amplitude));
        }
        if let Some(p) = props.get("vibrate.rotationAmplitude") {
            vib.rotation_amplitude = session_property_to_track(p, Some(&vib.rotation_amplitude));
        }
        if let Some(p) = props.get("vibrate.frequency") {
            vib.frequency = session_property_to_track(p, Some(&vib.frequency));
        }
        if let Some(v) = prop_f64_static(props, "vibrate.regularPulse") {
            vib.regular_pulse = v.round() as i64 != 0;
        }
        if let Some(v) = prop_f64_static(props, "vibrate.relative") {
            vib.relative = v.round() as i64 != 0;
        }
        if let Some(v) = prop_f64_static(props, "vibrate.enablePosition") {
            vib.enable_position = v.round() as i64 != 0;
        }
        if let Some(v) = prop_f64_static(props, "vibrate.enableRotation") {
            vib.enable_rotation = v.round() as i64 != 0;
        }
        if let Some(p) = props.get("vibrate.positionFrequency") {
            vib.position_frequency = session_property_to_track(p, Some(&vib.position_frequency));
        }
        if let Some(p) = props.get("vibrate.rotationFrequency") {
            vib.rotation_frequency = session_property_to_track(p, Some(&vib.rotation_frequency));
        }
    }
}

fn find_primary_camera_object_mut(
    objects: &mut [zc_project::Object3D],
) -> Option<&mut zc_project::Object3D> {
    for obj in objects {
        if matches!(obj.kind, zc_project::Object3DKind::Camera(_)) {
            return Some(obj);
        }
        if let zc_project::Object3DKind::Group { children, .. } = &mut obj.kind
            && let Some(found) = find_primary_camera_object_mut(children)
        {
            return Some(found);
        }
    }
    None
}

fn apply_material_custom_prop_metadata(
    mat: &mut zc_project::MaterialSpec,
    prefix: &str,
    obj_meta: &BTreeMap<String, BTreeMap<String, Value>>,
) {
    if let zc_project::MaterialSpec::CustomShader {
        custom_properties, ..
    } = mat
    {
        for cp in custom_properties.iter_mut() {
            let key = format!("{prefix}custom.{}", cp.name);
            if let Some(meta) = obj_meta.get(&key) {
                cp.metadata = meta.clone();
            }
        }
    }
}

fn post_process_synced_objects_3d(
    objects: &mut [zc_project::Object3D],
    sobj_map: &BTreeMap<String, &SessionObject3D>,
    prop_meta: &BTreeMap<String, BTreeMap<String, BTreeMap<String, Value>>>,
) {
    for obj in objects.iter_mut() {
        let obj_id = obj.id.to_string();
        if let Some(sobj) = sobj_map.get(&obj_id) {
            // Sync `zoidium_shadow_mode` (`zoidiumShadowMode` / `shadowMode`).
            if let Some(mode) = parse_compat_mode_from_props(&sobj.properties, "zoidiumShadowMode")
                .or_else(|| parse_compat_mode_from_props(&sobj.properties, "shadowMode"))
            {
                obj.zoidium_shadow_mode = mode;
            }
            // Sync `iesProfile` on lights (preserving across both standard and Light+/Trapcode lights).
            if matches!(obj.kind, zc_project::Object3DKind::Light(_))
                && let Some(prop) = sobj.properties.get("iesProfile")
            {
                let val = prop
                    .keyframes
                    .first()
                    .map(|k| &k.value)
                    .unwrap_or(&prop.default_value);
                if val.is_null() || val.as_str().is_some_and(|s| s.trim().is_empty()) {
                    obj.preserved.remove("iesProfile");
                    obj.preserved.remove("property.iesProfile");
                } else {
                    obj.preserved.insert("iesProfile".to_owned(), val.clone());
                    obj.preserved
                        .insert("property.iesProfile".to_owned(), val.clone());
                }
            }
            // Sync `path_closed` on 3D Wire shapes.
            if matches!(
                obj.kind,
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Wire { .. })
            ) && let Some(v) = prop_f64_static(&sobj.properties, "path_closed")
            {
                let closed_u64 = u64::from(v.round() as i64 != 0);
                if closed_u64 != 0
                    || obj.preserved.contains_key("path_closed")
                    || obj.preserved.contains_key("property.path_closed")
                {
                    obj.preserved
                        .insert("path_closed".to_owned(), json!(closed_u64));
                    obj.preserved
                        .insert("property.path_closed".to_owned(), json!(closed_u64));
                }
            }
        }
        // Sync preserved `CustomProperty.metadata` onto 3D Groups and CustomShader materials.
        if let Some(obj_meta) = prop_meta.get(&obj_id) {
            if let zc_project::Object3DKind::Group {
                custom_properties, ..
            } = &mut obj.kind
            {
                for cp in custom_properties.iter_mut() {
                    if let Some(meta) = obj_meta
                        .get(&cp.name)
                        .or_else(|| obj_meta.get(&format!("custom.{}", cp.name)))
                    {
                        cp.metadata = meta.clone();
                    }
                }
            }
            if let Some(mat) = obj.material.as_mut() {
                apply_material_custom_prop_metadata(mat, "material.", obj_meta);
            }
            if let Some(mat) = obj.material_slots.get_mut("front") {
                apply_material_custom_prop_metadata(mat, "frontMaterial.", obj_meta);
            }
            if let Some(mat) = obj.material_slots.get_mut("side") {
                apply_material_custom_prop_metadata(mat, "sideMaterial.", obj_meta);
            }
        }
        if let zc_project::Object3DKind::Group { children, .. } = &mut obj.kind {
            post_process_synced_objects_3d(children, sobj_map, prop_meta);
        }
    }
}

fn sync_effect_from_session(
    sfx: &SessionEffect,
    existing: Option<zc_project::EffectInstance>,
    group_children: &BTreeMap<String, Vec<SessionEffect>>,
    prop_meta: &BTreeMap<String, BTreeMap<String, BTreeMap<String, Value>>>,
    depth: usize,
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
                    .or_else(|| sfx.kind.strip_prefix("zoidcraft:"))
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
    let fx_meta = prop_meta.get(&sfx.id);
    if let zc_project::EffectKind::CustomShader {
        ref mut frag_shader,
        ref mut custom_properties,
    } = fx.kind
    {
        if let Some(code) = prop_str_static(&sfx.properties, "fragShader") {
            *frag_shader = code;
        }
        // Preserve exact original uniform declaration order, per-axis Bezier channels, and
        // `CustomProperty.metadata` (Shader+ kinds/dropdowns/visibility/rename rules),
        // and never duplicate custom shader uniforms into `fx.params`.
        let old_cps = std::mem::take(custom_properties);
        let mut handled_names = BTreeSet::new();
        for mut cp in old_cps {
            if let Some(prop) = sfx.properties.get(&cp.name) {
                cp.track = session_property_to_track(prop, Some(&cp.track));
                if let Some(meta) = fx_meta.and_then(|m| m.get(&cp.name)) {
                    cp.metadata = meta.clone();
                }
                handled_names.insert(cp.name.clone());
                custom_properties.push(cp);
            }
        }
        for (k, prop) in &sfx.properties {
            if k == "enabled" || k == "fragShader" || handled_names.contains(k) {
                continue;
            }
            let track = session_property_to_track(prop, None);
            let metadata = fx_meta.and_then(|m| m.get(k)).cloned().unwrap_or_default();
            handled_names.insert(k.clone());
            custom_properties.push(zc_project::CustomProperty {
                name: k.clone(),
                prop_type: None,
                track,
                metadata,
            });
        }
        fx.params
            .retain(|k, _| k != "enabled" && k != "fragShader" && !handled_names.contains(k));
    } else if let zc_project::EffectKind::EffectGroup {
        ref mut custom_properties,
        ref mut children,
    } = fx.kind
    {
        let mut handled_names = BTreeSet::new();
        for cp in custom_properties.iter_mut() {
            if let Some(prop) = sfx.properties.get(&cp.name) {
                cp.track = session_property_to_track(prop, Some(&cp.track));
                if let Some(meta) = fx_meta.and_then(|m| m.get(&cp.name)) {
                    cp.metadata = meta.clone();
                }
                handled_names.insert(cp.name.clone());
            }
        }
        fx.params.retain(|k, _| {
            k != "enabled" && !handled_names.contains(k) && sfx.properties.contains_key(k)
        });
        for (k, prop) in &sfx.properties {
            if k == "enabled" || handled_names.contains(k) {
                continue;
            }
            if let Some(meta) = fx_meta.and_then(|m| m.get(k)) {
                custom_properties.push(zc_project::CustomProperty {
                    name: k.clone(),
                    prop_type: None,
                    track: session_property_to_track(prop, None),
                    metadata: meta.clone(),
                });
                fx.params.remove(k);
                handled_names.insert(k.clone());
            } else {
                let prev = fx.params.get(k);
                let track = session_property_to_track(prop, prev);
                fx.params.insert(k.clone(), track);
            }
        }
        if depth < 16
            && let Some(session_children) = group_children.get(&sfx.id)
        {
            let mut existing_children: BTreeMap<String, zc_project::EffectInstance> = children
                .drain(..)
                .map(|child| (child.id.to_string(), child))
                .collect();
            *children = session_children
                .iter()
                .map(|child_sfx| {
                    let base_child = existing_children.remove(&child_sfx.id);
                    sync_effect_from_session(
                        child_sfx,
                        base_child,
                        group_children,
                        prop_meta,
                        depth + 1,
                    )
                })
                .collect();
        }
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
    ctx: &ClipSyncContext<'_>,
    side_updates: &mut ClipSideTableUpdates,
    alloc_uuid: &mut impl FnMut() -> zc_project::Uuid,
) -> zc_project::VideoClip {
    let resolution = ctx.resolution;
    let fps = ctx.fps;
    let seq_bg = ctx.seq_bg;
    let was_existing = existing.is_some();
    let mut vc = existing.unwrap_or_else(|| {
        let layer_kind = match sclip.kind.as_str() {
            "video" => zc_project::LayerKind::Video { media: None },
            "adjustment" => zc_project::LayerKind::Adjustment,
            "composite" => zc_project::LayerKind::Composite { layers: Vec::new() },
            "image" => zc_project::LayerKind::Image { image: None },
            "generator" => zc_project::LayerKind::Generator,
            "shape" | "shape2d" | "shape_2d" => zc_project::LayerKind::Shape2D {
                shapes: vec![zc_project::Shape2DNode {
                    id: alloc_uuid(),
                    name: "Shape".to_owned(),
                    kind: zc_project::Shape2DKind::Path {
                        path: default_shape2d_rectangle_path(),
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
            "text" | "text_2d" | "text2d" => zc_project::LayerKind::Text2D {
                text: "text".to_owned(),
                font: None,
                size: 72,
                shapes: vec![zc_project::Shape2DNode {
                    id: alloc_uuid(),
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
                    id: alloc_uuid(),
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
            "zoidium:precomp-plus/precomp" | "precomp" => {
                let comp_id = prop_str_static(&sclip.properties, "compositionId")
                    .and_then(|s| zc_project::Uuid::parse_str(&s).ok())
                    .unwrap_or_else(zc_project::Uuid::nil);
                zc_project::LayerKind::PrecompRef {
                    composition_id: comp_id,
                }
            }
            "zoidium:shape-plus/shape-layer" | "shape_plus" => zc_project::LayerKind::Plugin {
                plugin_type: "zoidium:shape-plus/shape-layer".to_owned(),
                payload: zc_render::shape_plus::default_shape_plus_payload(),
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
        zc_project::LayerKind::Scene3D {
            objects,
            shadow_mode,
        } => {
            if let Some(mode) = parse_compat_mode_from_props(&sclip.properties, "shadowMode")
                .or_else(|| parse_compat_mode_from_props(&sclip.properties, "zoidiumShadowMode"))
            {
                *shadow_mode = mode;
            }
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

            if let Some(prev_bd) = ctx.existing_scene_backdrops.get(&vc.layer.id) {
                let next_opacity = sclip.properties.get("backdropOpacity").map_or_else(
                    || prev_bd.opacity.clone(),
                    |p| session_property_to_track(p, Some(&prev_bd.opacity)),
                );
                side_updates.scene_backdrops.push((
                    vc.layer.id,
                    zc_project::SceneBackdrop {
                        mode: scene_bg_mode.max(0) as u32,
                        color: [r, g, b],
                        opacity: next_opacity,
                    },
                ));
            } else if scene_bg_mode != 0
                && let Some(op_prop) = sclip.properties.get("backdropOpacity")
                && (op_prop.animated || op_prop.expression.is_some())
            {
                side_updates.scene_backdrops.push((
                    vc.layer.id,
                    zc_project::SceneBackdrop {
                        mode: scene_bg_mode.max(0) as u32,
                        color: [r, g, b],
                        opacity: session_property_to_track(op_prop, None),
                    },
                ));
            }

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
            let needs_obj_templates = sclip.objects_3d.iter().any(|sobj| sobj.source_id.is_some());
            let obj_templates = needs_obj_templates.then(|| existing_objs.clone());
            *objects = rebuild_object_tree(
                &sclip.objects_3d,
                &mut existing_objs,
                obj_templates.as_ref(),
            );
            let sobj_map: BTreeMap<String, &SessionObject3D> = sclip
                .objects_3d
                .iter()
                .map(|sobj| (sobj.id.clone(), sobj))
                .collect();
            post_process_synced_objects_3d(objects, &sobj_map, ctx.property_metadata);
            for sobj in &mut sclip.objects_3d {
                sobj.source_id = None;
            }
        }
        zc_project::LayerKind::CameraLayer { camera } => {
            let prev_camera = camera.as_ref().clone();
            sync_camera_data_from_session(&sclip.properties, camera);
            let clip_camera_edited = *camera.as_ref() != prev_camera;

            if !sclip.objects_3d.is_empty()
                || ctx.existing_camera_layer_objects.contains_key(&vc.layer.id)
            {
                let mut existing_cam_objs = BTreeMap::new();
                if let Some(prev_objs) = ctx.existing_camera_layer_objects.get(&vc.layer.id) {
                    collect_existing_objects(prev_objs.clone(), &mut existing_cam_objs);
                }
                let needs_templates = sclip.objects_3d.iter().any(|sobj| sobj.source_id.is_some());
                let obj_templates = needs_templates.then(|| existing_cam_objs.clone());
                let mut rebuilt = rebuild_object_tree(
                    &sclip.objects_3d,
                    &mut existing_cam_objs,
                    obj_templates.as_ref(),
                );
                let sobj_map: BTreeMap<String, &SessionObject3D> = sclip
                    .objects_3d
                    .iter()
                    .map(|sobj| (sobj.id.clone(), sobj))
                    .collect();
                post_process_synced_objects_3d(&mut rebuilt, &sobj_map, ctx.property_metadata);
                if let Some(cam_obj) = find_primary_camera_object_mut(&mut rebuilt) {
                    if let Some(pos_prop) = sclip.properties.get("cameraPosition")
                        && !property_track_matches_session(&cam_obj.position, pos_prop)
                    {
                        cam_obj.position =
                            session_property_to_track(pos_prop, Some(&cam_obj.position));
                    }
                    if let Some(rot_prop) = sclip.properties.get("cameraRotation")
                        && !property_track_matches_session(&cam_obj.rotation, rot_prop)
                    {
                        cam_obj.rotation =
                            session_property_to_track(rot_prop, Some(&cam_obj.rotation));
                    }
                    if let zc_project::Object3DKind::Camera(ref mut obj_cam) = cam_obj.kind {
                        if clip_camera_edited {
                            sync_camera_data_from_session(&sclip.properties, obj_cam);
                        } else {
                            **camera = obj_cam.as_ref().clone();
                        }
                    }
                }
                side_updates
                    .camera_layer_objects
                    .push((vc.layer.id, rebuilt));
                for sobj in &mut sclip.objects_3d {
                    sobj.source_id = None;
                }
            } else {
                // Keep layer 2D/3D transform aligned if `cameraPosition` / `cameraRotation`
                // were edited directly on a CameraLayer without a `camera_layer_objects` tree.
                if let Some(cpos) = prop_vec3_static(&sclip.properties, "cameraPosition") {
                    let cur_xy = match &vc.layer.position.value {
                        zc_project::PropertyValue::Vec2(v) => *v,
                        _ => [0.0, 0.0],
                    };
                    if (cpos[0] - cur_xy[0]).abs() > 1e-9 || (cpos[1] - cur_xy[1]).abs() > 1e-9 {
                        vc.layer.position = zc_project::PropertyTrack::vec2([cpos[0], cpos[1]]);
                    }
                    if !sclip.properties.contains_key("depth") {
                        let mut p3d = ctx
                            .existing_layer_3d_props
                            .get(&vc.layer.id)
                            .cloned()
                            .unwrap_or_default();
                        p3d.depth = zc_project::PropertyTrack::number(cpos[2]);
                        side_updates.layer_3d_props.push((vc.layer.id, p3d));
                    }
                }
                if let Some(crot) = prop_vec3_static(&sclip.properties, "cameraRotation") {
                    let cur_rz = vc.layer.rotation.value.as_f64().unwrap_or(0.0);
                    if (crot[2] - cur_rz).abs() > 1e-9 {
                        vc.layer.rotation = zc_project::PropertyTrack::number(crot[2]);
                    }
                    if !sclip.properties.contains_key("rotationX")
                        && !sclip.properties.contains_key("rotationY")
                    {
                        let mut p3d = side_updates
                            .layer_3d_props
                            .iter()
                            .rev()
                            .find(|(id, _)| *id == vc.layer.id)
                            .map(|(_, p)| p.clone())
                            .or_else(|| ctx.existing_layer_3d_props.get(&vc.layer.id).cloned())
                            .unwrap_or_default();
                        p3d.rotation_x = zc_project::PropertyTrack::number(crot[0]);
                        p3d.rotation_y = zc_project::PropertyTrack::number(crot[1]);
                        side_updates.layer_3d_props.push((vc.layer.id, p3d));
                    }
                }
            }
        }
        _ => {}
    }

    // Sync Camera+ `layer_3d_props` (`depth`, `rotationX`, `rotationY`, `motionBlurAmount`)
    // for any video layer that exposes or edits 3D layer transform / motion-blur props.
    let existing_p3d = ctx.existing_layer_3d_props.get(&vc.layer.id);
    if sclip.properties.contains_key("depth")
        || sclip.properties.contains_key("rotationX")
        || sclip.properties.contains_key("rotationY")
        || sclip.properties.contains_key("motionBlurAmount")
    {
        let mut p3d = side_updates
            .layer_3d_props
            .iter()
            .rev()
            .find(|(id, _)| *id == vc.layer.id)
            .map(|(_, p)| p.clone())
            .or_else(|| existing_p3d.cloned())
            .unwrap_or_default();
        if let Some(p) = sclip.properties.get("depth") {
            let depth_changed =
                existing_p3d.is_none_or(|prev| !property_track_matches_session(&prev.depth, p));
            if depth_changed || !matches!(vc.layer.kind, zc_project::LayerKind::CameraLayer { .. })
            {
                p3d.depth = session_property_to_track(p, Some(&p3d.depth));
            }
        }
        if let Some(p) = sclip.properties.get("rotationX") {
            let rx_changed = existing_p3d
                .is_none_or(|prev| !property_track_matches_session(&prev.rotation_x, p));
            if rx_changed || !matches!(vc.layer.kind, zc_project::LayerKind::CameraLayer { .. }) {
                p3d.rotation_x = session_property_to_track(p, Some(&p3d.rotation_x));
            }
        }
        if let Some(p) = sclip.properties.get("rotationY") {
            let ry_changed = existing_p3d
                .is_none_or(|prev| !property_track_matches_session(&prev.rotation_y, p));
            if ry_changed || !matches!(vc.layer.kind, zc_project::LayerKind::CameraLayer { .. }) {
                p3d.rotation_y = session_property_to_track(p, Some(&p3d.rotation_y));
            }
        }
        if let Some(p) = sclip.properties.get("motionBlurAmount") {
            p3d.motion_blur_amount = session_property_to_track(p, Some(&p3d.motion_blur_amount));
        }
        // For CameraLayer without `camera_layer_objects`, also allow `cameraPosition.z` /
        // `cameraRotation.xy` edits when `depth`/`rotationX`/`rotationY` weren't directly changed.
        if matches!(vc.layer.kind, zc_project::LayerKind::CameraLayer { .. })
            && sclip.objects_3d.is_empty()
            && !ctx.existing_camera_layer_objects.contains_key(&vc.layer.id)
        {
            if let Some(cpos) = prop_vec3_static(&sclip.properties, "cameraPosition") {
                let depth_unchanged = existing_p3d.is_some_and(|prev| {
                    sclip
                        .properties
                        .get("depth")
                        .is_none_or(|p| property_track_matches_session(&prev.depth, p))
                }) || sclip.properties.get("depth").is_none_or(|p| {
                    p.keyframes
                        .first()
                        .and_then(|k| k.value.as_f64())
                        .is_some_and(|d| (d - 100.0).abs() < 1e-9)
                });
                if depth_unchanged
                    && (cpos[2] - p3d.depth.value.as_f64().unwrap_or(100.0)).abs() > 1e-9
                {
                    p3d.depth = zc_project::PropertyTrack::number(cpos[2]);
                }
            }
            if let Some(crot) = prop_vec3_static(&sclip.properties, "cameraRotation") {
                let rx_unchanged = existing_p3d.is_some_and(|prev| {
                    sclip
                        .properties
                        .get("rotationX")
                        .is_none_or(|p| property_track_matches_session(&prev.rotation_x, p))
                }) || sclip.properties.get("rotationX").is_none_or(|p| {
                    p.keyframes
                        .first()
                        .and_then(|k| k.value.as_f64())
                        .is_some_and(|v| v.abs() < 1e-9)
                });
                let ry_unchanged = existing_p3d.is_some_and(|prev| {
                    sclip
                        .properties
                        .get("rotationY")
                        .is_none_or(|p| property_track_matches_session(&prev.rotation_y, p))
                }) || sclip.properties.get("rotationY").is_none_or(|p| {
                    p.keyframes
                        .first()
                        .and_then(|k| k.value.as_f64())
                        .is_some_and(|v| v.abs() < 1e-9)
                });
                if rx_unchanged
                    && (crot[0] - p3d.rotation_x.value.as_f64().unwrap_or(0.0)).abs() > 1e-9
                {
                    p3d.rotation_x = zc_project::PropertyTrack::number(crot[0]);
                }
                if ry_unchanged
                    && (crot[1] - p3d.rotation_y.value.as_f64().unwrap_or(0.0)).abs() > 1e-9
                {
                    p3d.rotation_y = zc_project::PropertyTrack::number(crot[1]);
                }
            }
        }
        side_updates.layer_3d_props.push((vc.layer.id, p3d));
    }

    // Sync clip/layer-level Track Matte properties (`trackMatte`, `matteType`, `matteInvert`).
    for (canon, snake) in [
        ("trackMatte", "track_matte"),
        ("matteType", "matte_type"),
        ("matteInvert", "matte_invert"),
    ] {
        if let Some(prop) = sclip.properties.get(canon) {
            let val = prop
                .keyframes
                .first()
                .map(|k| &k.value)
                .unwrap_or(&prop.default_value);
            if val.is_null() || val.as_str().is_some_and(str::is_empty) {
                vc.layer.preserved.remove(canon);
                vc.layer.preserved.remove(snake);
                vc.preserved.remove(canon);
                vc.preserved.remove(snake);
            } else {
                vc.layer.preserved.remove(snake);
                vc.layer.preserved.insert(canon.to_owned(), val.clone());
            }
        }
    }

    let mut existing_fx: BTreeMap<String, zc_project::EffectInstance> = vc
        .layer
        .effects
        .drain(..)
        .map(|fx| (fx.id.to_string(), fx))
        .collect();
    let needs_fx_templates = sclip.effects.iter().any(|sfx| sfx.source_id.is_some());
    let fx_templates = needs_fx_templates.then(|| existing_fx.clone());
    vc.layer.effects = sclip
        .effects
        .iter_mut()
        .map(|sfx| {
            let base = existing_fx.remove(&sfx.id).or_else(|| {
                fx_templates.as_ref().and_then(|templates| {
                    sfx.source_id
                        .as_deref()
                        .and_then(|src_id| templates.get(src_id))
                        .map(|tpl| {
                            let mut cloned = tpl.clone();
                            cloned.id = parse_uuid_or_new(&sfx.id);
                            regenerate_effect_ids(&mut cloned, false, alloc_uuid);
                            cloned
                        })
                })
            });
            sfx.source_id = None;
            sync_effect_from_session(
                sfx,
                base,
                ctx.effect_group_children,
                ctx.property_metadata,
                0,
            )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::Command;

    #[test]
    fn scene_backdrops_expose_and_sync_from_scene_and_sequence_edits() {
        let mut proj = zc_project::Project::new_cm3_default();
        let scene_clip_id = proj.sequence.video_tracks[0].clips[0].id.to_string();
        let scene_layer_id = proj.sequence.video_tracks[0].clips[0].layer.id;
        proj.sequence.scene_backdrops.insert(
            scene_layer_id,
            zc_project::SceneBackdrop {
                mode: 1,
                color: [0.1, 0.2, 0.4],
                opacity: zc_project::PropertyTrack::number(0.8),
            },
        );

        let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
        let clip_props = &session.document().video_tracks[0].clips[0].properties;
        assert_eq!(prop_f64_static(clip_props, "sceneBackground"), Some(1.0));
        assert_eq!(
            prop_vec3_static(clip_props, "backgroundColor"),
            Some([0.1, 0.2, 0.4])
        );
        assert_eq!(prop_f64_static(clip_props, "backdropOpacity"), Some(0.8));
        assert_eq!(
            prop_vec3_static(&session.document().sequence_properties, "backgroundColor"),
            Some([0.1, 0.2, 0.4])
        );

        // Editing sequence backgroundColor updates the primary solid scene_backdrop immediately.
        session
            .execute(Command::SetPropertyValue {
                target_id: "sequence".to_owned(),
                property: "backgroundColor".to_owned(),
                value: json!([0.85, 0.15, 0.25]),
            })
            .expect("edit sequence backgroundColor");
        let bd = &session.project.sequence.scene_backdrops[&scene_layer_id];
        assert_eq!(bd.mode, 1);
        assert!((bd.color[0] - 0.85).abs() < 1e-6);
        assert!((bd.color[1] - 0.15).abs() < 1e-6);
        assert!((bd.color[2] - 0.25).abs() < 1e-6);

        // Editing Scene3D clip backdropOpacity & backgroundColor updates scene_backdrops directly.
        session
            .execute(Command::SetPropertyValue {
                target_id: scene_clip_id.clone(),
                property: "backgroundColor".to_owned(),
                value: json!([0.2, 0.9, 0.3]),
            })
            .expect("edit scene backgroundColor");
        session
            .execute(Command::SetPropertyValue {
                target_id: scene_clip_id,
                property: "backdropOpacity".to_owned(),
                value: json!(0.5),
            })
            .expect("edit scene backdropOpacity");
        let bd2 = &session.project.sequence.scene_backdrops[&scene_layer_id];
        assert_eq!(bd2.color, [0.2, 0.9, 0.3]);
        assert_eq!(bd2.opacity.value.as_f64(), Some(0.5));
    }

    #[test]
    fn effect_group_children_round_trip_and_support_property_edits() {
        let mut proj = zc_project::Project::new_cm3_default();
        let mut child_blur =
            zc_project::create_effect_by_id("boxblur").expect("boxblur effect template");
        child_blur.name = "Inner Blur".to_owned();
        child_blur.enabled = zc_project::PropertyTrack::option(0);
        let child_blur_id = child_blur.id.to_string();

        let mut child_shader = zc_project::create_custom_shader(
            "Inner Shader",
            "uniform sampler2D tDiffuse; varying vec2 vUvScaled; uniform float Gain; void main() { gl_FragColor = texture2D(tDiffuse, vUvScaled) * Gain; }",
        );
        let child_shader_id = child_shader.id.to_string();
        if let zc_project::EffectKind::CustomShader {
            ref mut custom_properties,
            ..
        } = child_shader.kind
        {
            custom_properties.push(zc_project::CustomProperty {
                name: "Gain".to_owned(),
                prop_type: Some(0),
                track: zc_project::PropertyTrack::number(1.25),
                metadata: BTreeMap::new(),
            });
        }

        let mut group = zc_project::create_effect_group("Macro FX");
        let group_id = group.id.to_string();
        if let zc_project::EffectKind::EffectGroup {
            ref mut children, ..
        } = group.kind
        {
            children.push(child_blur);
            children.push(child_shader);
        }
        proj.sequence.video_tracks[0].clips[0]
            .layer
            .effects
            .push(group);

        let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
        let children = session.document().effect_children(&group_id);
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].id, child_blur_id);
        assert_eq!(children[0].name, "Inner Blur");
        assert_eq!(
            prop_f64_static(&children[0].properties, "enabled"),
            Some(0.0)
        );
        assert_eq!(children[1].id, child_shader_id);
        assert_eq!(prop_f64_static(&children[1].properties, "Gain"), Some(1.25));

        // Edit child effect enabled state, property value, and inline name directly via Session.
        session
            .execute(Command::SetPropertyValue {
                target_id: child_blur_id.clone(),
                property: "enabled".to_owned(),
                value: json!(1),
            })
            .expect("enable child blur");
        session
            .execute(Command::SetPropertyValue {
                target_id: child_shader_id.clone(),
                property: "Gain".to_owned(),
                value: json!(2.5),
            })
            .expect("edit child shader uniform");
        session
            .execute(Command::SetPropertyValue {
                target_id: child_shader_id.clone(),
                property: "name".to_owned(),
                value: json!("Boosted Shader"),
            })
            .expect("rename child shader");

        // Verify synced back to Project preserving exact child order, enabled state, and properties.
        let reloaded =
            Session::from_project(session.project.clone(), zc_plugin::PluginRegistry::new());
        let fx_proj = &reloaded.project.sequence.video_tracks[0].clips[0]
            .layer
            .effects[0];
        let zc_project::EffectKind::EffectGroup {
            children: proj_children,
            ..
        } = &fx_proj.kind
        else {
            panic!("expected EffectGroup");
        };
        assert_eq!(proj_children.len(), 2);
        assert_eq!(proj_children[0].id.to_string(), child_blur_id);
        assert_eq!(proj_children[0].enabled.value.as_f64(), Some(1.0));
        assert_eq!(proj_children[1].id.to_string(), child_shader_id);
        assert_eq!(proj_children[1].name, "Boosted Shader");
        let zc_project::EffectKind::CustomShader {
            custom_properties, ..
        } = &proj_children[1].kind
        else {
            panic!("expected CustomShader child");
        };
        assert_eq!(custom_properties[0].track.value.as_f64(), Some(2.5));
    }

    #[test]
    fn custom_property_metadata_persists_shader_plus_kinds_dropdowns_visibility_and_rename() {
        let mut proj = zc_project::Project::new_cm3_default();
        let mut shader = zc_project::create_custom_shader(
            "Shader+ FX",
            "uniform sampler2D tDiffuse; varying vec2 vUvScaled; uniform int Mode; uniform float Mix; void main() { gl_FragColor = texture2D(tDiffuse, vUvScaled) * Mix; }",
        );
        let shader_id = shader.id.to_string();

        let mut dropdown_meta = BTreeMap::new();
        dropdown_meta.insert(
            "type".to_owned(),
            json!({
                "zoidiumShader": {
                    "kind": "staticDropdown",
                    "items": ["Normal", "Add", "Screen"],
                    "visibility": { "when": { "prop": "Mix", "op": ">", "value": 0.0 } },
                    "rename": [{ "when": { "prop": "Mix", "op": ">", "value": 0.5 }, "label": "Strong Blend" }]
                }
            }),
        );
        let mut layer_meta = BTreeMap::new();
        layer_meta.insert(
            "type".to_owned(),
            json!({
                "zoidiumShader": { "kind": "layerSource" },
                "_zoidiumShaderLayerSource": true
            }),
        );

        if let zc_project::EffectKind::CustomShader {
            ref mut custom_properties,
            ..
        } = shader.kind
        {
            custom_properties.push(zc_project::CustomProperty {
                name: "Mode".to_owned(),
                prop_type: Some(2),
                track: zc_project::PropertyTrack::option(1),
                metadata: dropdown_meta.clone(),
            });
            custom_properties.push(zc_project::CustomProperty {
                name: "Mix".to_owned(),
                prop_type: Some(0),
                track: zc_project::PropertyTrack::number(0.75),
                metadata: BTreeMap::new(),
            });
            custom_properties.push(zc_project::CustomProperty {
                name: "SourceTex".to_owned(),
                prop_type: None,
                track: zc_project::PropertyTrack::text("track:v1"),
                metadata: layer_meta.clone(),
            });
        }
        proj.sequence.video_tracks[0].clips[0]
            .layer
            .effects
            .push(shader);

        let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
        assert_eq!(
            session.document().property_metadata(&shader_id, "Mode"),
            Some(&dropdown_meta)
        );
        assert_eq!(
            session
                .document()
                .property_metadata(&shader_id, "SourceTex"),
            Some(&layer_meta)
        );

        // Add a brand-new Shader+ `timeSample` uniform and attach its metadata via `__meta.`.
        session
            .execute(Command::SetPropertyValue {
                target_id: shader_id.clone(),
                property: "PrevFrame".to_owned(),
                value: json!(-1.0),
            })
            .expect("add timeSample prop");
        session
            .execute(Command::SetPropertyValue {
                target_id: shader_id.clone(),
                property: "__meta.PrevFrame".to_owned(),
                value: json!({
                    "type": {
                        "zoidiumShader": {
                            "kind": "timeSample",
                            "unit": "frames"
                        }
                    }
                }),
            })
            .expect("set timeSample metadata");

        // Reload through full Project -> Session round-trip and assert metadata survived intact.
        let reloaded =
            Session::from_project(session.project.clone(), zc_plugin::PluginRegistry::new());
        let fx_proj = &reloaded.project.sequence.video_tracks[0].clips[0]
            .layer
            .effects[0];
        let zc_project::EffectKind::CustomShader {
            custom_properties, ..
        } = &fx_proj.kind
        else {
            panic!("expected CustomShader");
        };
        let mode_cp = custom_properties
            .iter()
            .find(|cp| cp.name == "Mode")
            .expect("Mode prop");
        assert_eq!(mode_cp.metadata, dropdown_meta);
        let prev_cp = custom_properties
            .iter()
            .find(|cp| cp.name == "PrevFrame")
            .expect("PrevFrame prop");
        assert_eq!(
            prev_cp.metadata["type"]["zoidiumShader"]["kind"],
            json!("timeSample")
        );
    }

    #[test]
    fn camera_layer_layer3d_track_matte_ies_wire_and_shadow_mode_round_trip() {
        let mut proj = zc_project::Project::new_cm3_default();
        let scene_layer_id = proj.sequence.video_tracks[0].clips[0].layer.id;
        let track_id = proj.sequence.video_tracks[0].id.to_string();
        proj.sequence.video_tracks[0]
            .preserved
            .insert("trackMatte".to_owned(), json!("v2"));
        proj.sequence.video_tracks[0]
            .preserved
            .insert("matteType".to_owned(), json!(1));
        proj.sequence.layer_3d_props.insert(
            scene_layer_id,
            zc_project::Layer3DProps {
                depth: zc_project::PropertyTrack::number(-25.0),
                rotation_x: zc_project::PropertyTrack::number(15.0),
                rotation_y: zc_project::PropertyTrack::number(-30.0),
                motion_blur_amount: zc_project::PropertyTrack::number(1.5),
            },
        );
        let scene_clip = &mut proj.sequence.video_tracks[0].clips[0];
        if let zc_project::LayerKind::Scene3D {
            ref mut objects,
            ref mut shadow_mode,
        } = scene_clip.layer.kind
        {
            *shadow_mode = zc_project::CompatMode::Legacy;
            let mut light = zc_project::Object3D::new(
                "IES Light",
                zc_project::Object3DKind::Light(Box::new(zc_project::LightData::for_kind(
                    zc_project::LightKind::Point,
                ))),
            );
            light.zoidium_shadow_mode = zc_project::CompatMode::Legacy;
            light
                .preserved
                .insert("iesProfile".to_owned(), json!("studio_cone.ies"));
            let mut wire = zc_project::Object3D::new(
                "Wire Loop",
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Wire {
                    size: [10.0, 10.0, 10.0],
                    thickness: 0.5,
                    loops: 2,
                    inner_segments: 8,
                    outer_segments: 16,
                }),
            );
            wire.preserved.insert("path_closed".to_owned(), json!(1));
            objects.push(light);
            objects.push(wire);
        }

        // Add a CameraLayer clip with Vibrate, DOF, and motion blur.
        let cam_data = zc_project::CameraData {
            fov: zc_project::PropertyTrack::number(48.0),
            vibrate: Some(zc_project::CameraVibrateParams {
                enabled: true,
                position_amplitude: zc_project::PropertyTrack::vec3([2.0, 3.0, 1.0]),
                frequency: zc_project::PropertyTrack::number(4.5),
                ..zc_project::CameraVibrateParams::default()
            }),
            dof: Some(zc_project::CameraDofParams {
                enabled: true,
                focus_distance: zc_project::PropertyTrack::number(120.0),
                aperture: zc_project::PropertyTrack::number(5.5),
                near_blur: zc_project::PropertyTrack::number(140.0),
                far_blur: zc_project::PropertyTrack::number(180.0),
                ..zc_project::CameraDofParams::default()
            }),
            motion_blur: Some(zc_project::CameraMotionBlurParams {
                enabled: true,
                samples: 12,
                shutter: zc_project::PropertyTrack::number(180.0),
            }),
            ..zc_project::CameraData::default()
        };
        let cam_clip = zc_project::VideoClip {
            id: zc_project::Uuid::new_v4(),
            name: "Cam Layer".to_owned(),
            start: 0.0,
            length: 120.0,
            link: None,
            media: None,
            time_curve: zc_project::PropertyTrack::linear_time_curve(120.0, 30),
            layer: zc_project::Layer::new(
                "Cam Layer",
                zc_project::LayerKind::CameraLayer {
                    camera: Box::new(cam_data),
                },
                [1920, 1080],
            ),
            preserved: BTreeMap::new(),
        };
        let cam_clip_id = cam_clip.id.to_string();
        let cam_layer_id = cam_clip.layer.id;
        proj.sequence.video_tracks.push(zc_project::VideoTrack {
            id: zc_project::Uuid::new_v4(),
            name: "CamTrack".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: true,
            clips: vec![cam_clip],
            preserved: BTreeMap::new(),
        });

        let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
        let scene_clip_id = session.document().video_tracks[0].clips[0].id.clone();

        // Check Track Matte exposed on document and editable via SetPropertyValue.
        assert_eq!(
            session.document().track_matte[&track_id]["trackMatte"],
            json!("v2")
        );
        session
            .execute(Command::SetPropertyValue {
                target_id: track_id.clone(),
                property: "matteType".to_owned(),
                value: json!(2),
            })
            .expect("edit track matteType");
        session
            .execute(Command::SetPropertyValue {
                target_id: scene_clip_id.clone(),
                property: "motionBlurAmount".to_owned(),
                value: json!(2.25),
            })
            .expect("edit motionBlurAmount");
        session
            .execute(Command::SetPropertyValue {
                target_id: cam_clip_id.clone(),
                property: "fov".to_owned(),
                value: json!(35.0),
            })
            .expect("edit CameraLayer fov");
        session
            .execute(Command::SetPropertyValue {
                target_id: cam_clip_id,
                property: "dofFocusDistance".to_owned(),
                value: json!(200.0),
            })
            .expect("edit CameraLayer DOF");

        // Reload and verify every single field survived round-trip.
        let reloaded =
            Session::from_project(session.project.clone(), zc_plugin::PluginRegistry::new());
        assert_eq!(
            reloaded.project.sequence.video_tracks[0].preserved["matteType"],
            json!(2)
        );
        assert_eq!(
            reloaded.project.sequence.layer_3d_props[&scene_layer_id]
                .motion_blur_amount
                .value
                .as_f64(),
            Some(2.25)
        );
        let zc_project::LayerKind::Scene3D {
            objects,
            shadow_mode,
        } = &reloaded.project.sequence.video_tracks[0].clips[0]
            .layer
            .kind
        else {
            panic!("expected Scene3D");
        };
        assert_eq!(*shadow_mode, zc_project::CompatMode::Legacy);
        let light = objects
            .iter()
            .find(|o| o.name == "IES Light")
            .expect("IES Light");
        assert_eq!(light.zoidium_shadow_mode, zc_project::CompatMode::Legacy);
        assert_eq!(light.preserved["iesProfile"], json!("studio_cone.ies"));
        let wire = objects
            .iter()
            .find(|o| o.name == "Wire Loop")
            .expect("Wire Loop");
        assert_eq!(wire.preserved["path_closed"], json!(1));

        let cam_layer = &reloaded.project.sequence.video_tracks[1].clips[0].layer;
        assert_eq!(cam_layer.id, cam_layer_id);
        let zc_project::LayerKind::CameraLayer { camera } = &cam_layer.kind else {
            panic!("expected CameraLayer");
        };
        assert_eq!(camera.fov.value.as_f64(), Some(35.0));
        assert_eq!(
            camera.dof.as_ref().unwrap().focus_distance.value.as_f64(),
            Some(200.0)
        );
        assert_eq!(
            camera.vibrate.as_ref().unwrap().frequency.value.as_f64(),
            Some(4.5)
        );
        assert_eq!(camera.motion_blur.as_ref().unwrap().samples, 12);
    }
}
