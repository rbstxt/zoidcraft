use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use crate::Session;
use crate::document::{SessionClip, SessionEffect, SessionProperty, SessionValueKind};
use crate::plugin_defaults::default_shape2d_rectangle_path;
use crate::pro_commands::clone_layer_masks_shifted;
use crate::property_conv::{
    find_shape2d_fill_mut, find_shape2d_stroke_size_mut, parse_preserved_bool_u64,
    parse_preserved_rgba, prop_f64_static, prop_str_static, session_property_to_track,
};
use crate::sync_from_project::parse_uuid_or_new;
use crate::sync_objects_to_project::{collect_existing_objects, rebuild_object_tree};

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
            let mut precomp_offset_updates: Vec<(zc_project::Uuid, f64)> = Vec::new();
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
                            resolution,
                            fps_u32,
                            seq_bg,
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
                                precomp_offset_updates.push((synced.layer.id, off));
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
            for (lid, off) in precomp_offset_updates {
                target_seq.precomp_offsets.insert(lid, off);
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
            let needs_obj_templates = sclip.objects_3d.iter().any(|sobj| sobj.source_id.is_some());
            let obj_templates = needs_obj_templates.then(|| existing_objs.clone());
            *objects = rebuild_object_tree(
                &sclip.objects_3d,
                &mut existing_objs,
                obj_templates.as_ref(),
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
