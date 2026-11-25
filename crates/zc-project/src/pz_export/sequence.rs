use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::model::{
    AudioClip, AudioTrack, DrawOp, Layer, LayerKind, LayerMask, MaskMode, Object3D, Object3DKind,
    PresetShapeKind, PropertyTrack, Sequence, SequenceControl, SequenceControlKind, Shape2DKind,
    Shape2DNode, VideoClip, VideoTrack,
};

use super::effects::serialize_effect;
use super::objects3d::{SceneConstraintExportCtx, serialize_object3d, serialize_object3d_with_ctx};
use super::tracks::{
    finite_f64, remap_json_value, remap_str, serialize_dynamic_group, serialize_dynamic_track,
    serialize_keyframe, serialize_property_track, serialize_property_value,
};

pub(super) fn serialize_markers_track(seq: &Sequence) -> Value {
    let marker_colors = seq.preserved.get("markerColors").and_then(Value::as_array);
    let marker_kfs: Vec<Value> = seq
        .markers
        .iter()
        .enumerate()
        .map(|(idx, m)| {
            let mut kf = Map::new();
            kf.insert("frame".to_owned(), json!(finite_f64(m.frame)));
            kf.insert("value".to_owned(), Value::String(m.label.clone()));
            kf.insert("tween".to_owned(), json!(0));
            if let Some(col) = marker_colors
                .and_then(|arr| arr.get(idx))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                kf.insert("color".to_owned(), Value::String(col.to_owned()));
            }
            Value::Object(kf)
        })
        .collect();
    json!({
        "animated": true,
        "keyframes": marker_kfs
    })
}

pub(super) fn serialize_sequence(
    seq: &Sequence,
    comp_str_ids: &BTreeMap<Uuid, String>,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();

    for (k, v) in &seq.preserved {
        if k.starts_with("property.")
            || matches!(
                k.as_str(),
                "clipLinks.seed" | "zoidiumControls.nextId" | "comp_id" | "compSource"
            )
        {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    let mut props = Map::new();
    for (k, v) in &seq.preserved {
        if let Some(prop_key) = k.strip_prefix("property.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(prop_key.to_owned(), cloned);
        }
    }

    props.insert("name".to_owned(), Value::String(seq.name.clone()));
    props.insert(
        "resolution".to_owned(),
        json!([seq.resolution[0].max(1), seq.resolution[1].max(1)]),
    );
    let rate = seq
        .exact_fps
        .filter(|r| r.is_finite() && *r > 0.0)
        .unwrap_or_else(|| f64::from(seq.fps.max(1)));
    props.insert("rate".to_owned(), json!(rate));
    props.insert(
        "backgroundColor".to_owned(),
        json!([
            finite_f64(f64::from(seq.background_color[0])),
            finite_f64(f64::from(seq.background_color[1])),
            finite_f64(f64::from(seq.background_color[2])),
            finite_f64(f64::from(seq.background_color[3])),
        ]),
    );
    props.insert(
        "motionBlur".to_owned(),
        json!(u32::from(seq.motion_blur.enabled)),
    );
    props.insert(
        "motionBlurSamples".to_owned(),
        serialize_property_track(&seq.motion_blur.samples, remap),
    );
    props.insert(
        "motionBlurShutter".to_owned(),
        serialize_property_track(&seq.motion_blur.shutter, remap),
    );
    if seq.motion_blur.track_frame != 0 {
        props.insert(
            "motionBlurTrackFrame".to_owned(),
            serialize_property_track(&PropertyTrack::option(seq.motion_blur.track_frame), remap),
        );
    }
    if (seq.motion_blur.sensitivity - 100.0).abs() > 1e-9 {
        props.insert(
            "motionBlurSensitivity".to_owned(),
            serialize_property_track(&PropertyTrack::number(seq.motion_blur.sensitivity), remap),
        );
    }

    if !seq.markers.is_empty() {
        props.insert("markers".to_owned(), serialize_markers_track(seq));
    }

    if !seq.controls.is_empty() {
        let controls_json = serialize_sequence_controls(seq, remap);
        if let Ok(encoded) = serde_json::to_string(&controls_json) {
            props.insert("zoidiumControls".to_owned(), Value::String(encoded));
        }
    }

    obj.insert("properties".to_owned(), Value::Object(props));
    obj.insert("length".to_owned(), json!(finite_f64(seq.length_frames)));

    let video_tracks: Vec<Value> = seq
        .video_tracks
        .iter()
        .map(|vt| serialize_video_track(vt, seq, comp_str_ids, remap))
        .collect();
    obj.insert("videoTracks".to_owned(), Value::Array(video_tracks));

    let audio_tracks: Vec<Value> = seq
        .audio_tracks
        .iter()
        .map(|at| serialize_audio_track(at, remap))
        .collect();
    obj.insert("audioTracks".to_owned(), Value::Array(audio_tracks));

    if let Some(clip_links) = build_clip_links_json(seq) {
        obj.insert("clipLinks".to_owned(), clip_links);
    }

    Value::Object(obj)
}

pub(super) fn parse_valid_control_id(id: &str) -> Option<u64> {
    let digits = id.strip_prefix('c')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok()
}

pub(super) const fn sequence_control_kind_label(kind: SequenceControlKind) -> &'static str {
    match kind {
        SequenceControlKind::Slider => "Slider",
        SequenceControlKind::Angle => "Angle",
        SequenceControlKind::Checkbox => "Checkbox",
        SequenceControlKind::Color => "Color",
        SequenceControlKind::Point2D => "Point 2D",
        SequenceControlKind::Point3D => "Point 3D",
    }
}

pub(super) fn serialize_sequence_controls(
    seq: &Sequence,
    remap: &BTreeMap<String, String>,
) -> Value {
    // Pass 1: preserve existing valid unique `c<N>` IDs (matching `/^c\d+$/`).
    let mut reserved_ids: BTreeSet<String> = BTreeSet::new();
    let mut assigned_ids: Vec<Option<String>> = Vec::with_capacity(seq.controls.len());
    let mut max_num = 0u64;
    for ctl in &seq.controls {
        if let Some(n) = parse_valid_control_id(&ctl.id)
            && reserved_ids.insert(ctl.id.clone())
        {
            max_num = max_num.max(n);
            assigned_ids.push(Some(ctl.id.clone()));
        } else {
            assigned_ids.push(None);
        }
    }

    // Pass 2: assign sequential `c<next>` to invalid, duplicate, or session `ctl_*` IDs.
    let mut next_candidate = max_num.saturating_add(1).max(1);
    let mut final_ids: Vec<String> = Vec::with_capacity(seq.controls.len());
    for slot in assigned_ids {
        if let Some(existing) = slot {
            final_ids.push(existing);
        } else {
            while reserved_ids.contains(&format!("c{next_candidate}")) {
                next_candidate = next_candidate.saturating_add(1);
            }
            let id = format!("c{next_candidate}");
            reserved_ids.insert(id.clone());
            max_num = max_num.max(next_candidate);
            next_candidate = next_candidate.saturating_add(1);
            final_ids.push(id);
        }
    }

    // Normalize names so every control has a unique non-empty trimmed name.
    let mut used_names: BTreeSet<String> = BTreeSet::new();
    let mut final_names: Vec<String> = Vec::with_capacity(seq.controls.len());
    for ctl in &seq.controls {
        let trimmed = ctl.name.trim();
        let base = if trimmed.is_empty() {
            sequence_control_kind_label(ctl.kind)
        } else {
            trimmed
        };
        let mut candidate = base.to_owned();
        let mut suffix = 2u64;
        while !used_names.insert(candidate.clone()) {
            candidate = format!("{base} {suffix}");
            suffix = suffix.saturating_add(1);
        }
        final_names.push(candidate);
    }

    let preserved_next = seq
        .preserved
        .get("zoidiumControls.nextId")
        .and_then(Value::as_u64)
        .unwrap_or(1);
    let next_id = preserved_next
        .max(max_num.saturating_add(1))
        .max(seq.controls.len() as u64 + 1)
        .max(1);

    let mut seen_raw_ids: BTreeSet<String> = BTreeSet::new();
    let controls: Vec<Value> = seq
        .controls
        .iter()
        .enumerate()
        .map(|(idx, ctl)| {
            let trimmed_raw = ctl.id.trim();
            let preserve_raw_id = (!trimmed_raw.is_empty()
                && trimmed_raw != final_ids[idx]
                && seen_raw_ids.insert(trimmed_raw.to_owned()))
            .then_some(trimmed_raw);
            if trimmed_raw == final_ids[idx] {
                seen_raw_ids.insert(trimmed_raw.to_owned());
            }
            serialize_sequence_control(
                ctl,
                &final_ids[idx],
                preserve_raw_id,
                &final_names[idx],
                remap,
            )
        })
        .collect();

    json!({
        "version": 1,
        "nextId": next_id,
        "controls": controls
    })
}

pub(super) fn serialize_sequence_control(
    ctl: &SequenceControl,
    normalized_id: &str,
    preserved_raw_id: Option<&str>,
    normalized_name: &str,
    remap: &BTreeMap<String, String>,
) -> Value {
    let kind_str = sequence_control_kind_label(ctl.kind);
    let (cm3_type, default_val): (u32, Value) = match ctl.kind {
        SequenceControlKind::Slider => (0, json!(0)),
        SequenceControlKind::Angle => (0, json!(0)),
        SequenceControlKind::Checkbox => (6, json!(0)),
        SequenceControlKind::Color => (4, json!([1, 1, 1])),
        SequenceControlKind::Point2D => (1, json!([0, 0])),
        SequenceControlKind::Point3D => (2, json!([0, 0, 0])),
    };

    let mut type_obj = ctl
        .preserved
        .get("type")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    type_obj.remove("name");
    type_obj.insert("custom".to_owned(), Value::Bool(true));
    type_obj.insert("dynamic".to_owned(), Value::Bool(true));
    type_obj.insert("type".to_owned(), json!(cm3_type));
    type_obj.entry("value".to_owned()).or_insert(default_val);
    type_obj.insert(
        "zoidiumControlKind".to_owned(),
        Value::String(kind_str.to_owned()),
    );
    type_obj.insert(
        "zoidiumControlId".to_owned(),
        Value::String(normalized_id.to_owned()),
    );
    if let Some(raw_id) = preserved_raw_id {
        type_obj.insert(
            "zoidcraftControlId".to_owned(),
            Value::String(raw_id.to_owned()),
        );
    } else {
        type_obj.remove("zoidcraftControlId");
    }

    match ctl.kind {
        SequenceControlKind::Slider => {
            type_obj.entry("min".to_owned()).or_insert_with(|| json!(0));
            type_obj
                .entry("max".to_owned())
                .or_insert_with(|| json!(100));
            type_obj
                .entry("step".to_owned())
                .or_insert_with(|| json!(0.01));
        }
        SequenceControlKind::Angle => {
            type_obj
                .entry("scaleFactor".to_owned())
                .or_insert_with(|| json!(std::f64::consts::PI / 180.0));
            type_obj
                .entry("step".to_owned())
                .or_insert_with(|| json!(1));
            type_obj
                .entry("decimals".to_owned())
                .or_insert_with(|| json!(1));
        }
        SequenceControlKind::Checkbox => {
            type_obj
                .entry("items".to_owned())
                .or_insert_with(|| Value::String("off;on".to_owned()));
        }
        SequenceControlKind::Color
        | SequenceControlKind::Point2D
        | SequenceControlKind::Point3D => {}
    }

    for meta_key in ["min", "max", "step", "scaleFactor", "decimals", "items"] {
        if let Some(mv) = ctl.preserved.get(meta_key) {
            type_obj.insert(meta_key.to_owned(), mv.clone());
        }
    }

    let mut obj = Map::new();
    for (k, v) in &ctl.preserved {
        if matches!(
            k.as_str(),
            "type" | "min" | "max" | "step" | "scaleFactor" | "decimals" | "items"
        ) {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    obj.insert("type".to_owned(), Value::Object(type_obj));
    obj.insert("properties".to_owned(), json!({ "name": normalized_name }));

    // Merge dynamic track representation directly onto the control object (matches Zoidium `controls.js`).
    let is_group_kind = matches!(
        ctl.kind,
        SequenceControlKind::Color | SequenceControlKind::Point2D | SequenceControlKind::Point3D
    );
    let track_val = serialize_dynamic_track(&ctl.track, is_group_kind, remap);
    if let Value::Object(track_map) = track_val {
        for (k, v) in track_map {
            obj.insert(k, v);
        }
    }

    Value::Object(obj)
}

pub(super) fn build_clip_links_json(seq: &Sequence) -> Option<Value> {
    if seq.clip_links.is_empty() && !seq.preserved.contains_key("clipLinks.seed") {
        return None;
    }

    let mut clip_addr: BTreeMap<Uuid, (u32, usize, usize)> = BTreeMap::new();
    for (ti, vt) in seq.video_tracks.iter().enumerate() {
        for (ci, clip) in vt.clips.iter().enumerate() {
            clip_addr.insert(clip.id, (1, ti, ci));
        }
    }
    for (ti, at) in seq.audio_tracks.iter().enumerate() {
        for (ci, clip) in at.clips.iter().enumerate() {
            clip_addr.insert(clip.id, (2, ti, ci));
        }
    }

    let mut links_map = Map::new();
    let mut max_link_id = 0u32;
    for (&link_id, uuids) in &seq.clip_links {
        let mut addrs = Vec::new();
        for uuid in uuids {
            if let Some(&(track_kind, track_idx, clip_idx)) = clip_addr.get(uuid) {
                // Canonical 5-element CM3 / Zoidium clip link address: [0, 1|2, trackIdx, 0, clipIdx]
                addrs.push(json!([0, track_kind, track_idx, 0, clip_idx]));
            }
        }
        if addrs.len() >= 2 {
            max_link_id = max_link_id.max(link_id);
            links_map.insert(link_id.to_string(), json!({ "clips": addrs }));
        }
    }

    if links_map.is_empty() && !seq.preserved.contains_key("clipLinks.seed") {
        return None;
    }

    let seed = seq
        .preserved
        .get("clipLinks.seed")
        .and_then(Value::as_u64)
        .unwrap_or(u64::from(max_link_id.saturating_add(1)).max(1));

    Some(json!({
        "seed": seed,
        "links": links_map
    }))
}

pub(super) fn serialize_video_track(
    track: &VideoTrack,
    seq: &Sequence,
    comp_str_ids: &BTreeMap<Uuid, String>,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();
    for (k, v) in &track.preserved {
        if k == "_zoidiumLayerInputId" {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    obj.insert("type".to_owned(), json!(0));
    obj.insert("name".to_owned(), Value::String(track.name.clone()));
    obj.insert("enabled".to_owned(), Value::Bool(track.enabled));
    if track.solo {
        obj.insert("zoidiumSolo".to_owned(), Value::Bool(true));
    }
    if track.locked {
        obj.insert("zoidiumLocked".to_owned(), Value::Bool(true));
    }
    if track.is_3d {
        obj.insert("track3d".to_owned(), Value::Bool(true));
    }
    if let Some(layer_input_id) = seq.track_layer_input_ids.get(&track.id)
        && !layer_input_id.is_empty()
    {
        obj.insert(
            "_zoidiumLayerInputId".to_owned(),
            Value::String(layer_input_id.clone()),
        );
    }

    let clips: Vec<Value> = track
        .clips
        .iter()
        .map(|clip| serialize_video_clip(clip, seq, comp_str_ids, remap))
        .collect();
    obj.insert("clips".to_owned(), Value::Array(clips));

    Value::Object(obj)
}

pub(super) fn serialize_audio_track(track: &AudioTrack, remap: &BTreeMap<String, String>) -> Value {
    let mut obj = Map::new();
    for (k, v) in &track.preserved {
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    obj.insert("type".to_owned(), json!(1));
    obj.insert("name".to_owned(), Value::String(track.name.clone()));
    obj.insert("enabled".to_owned(), Value::Bool(track.enabled));
    if track.muted != !track.enabled {
        obj.insert("muted".to_owned(), Value::Bool(track.muted));
    }
    if track.solo {
        obj.insert("zoidiumSolo".to_owned(), Value::Bool(true));
    }
    if track.locked {
        obj.insert("zoidiumLocked".to_owned(), Value::Bool(true));
    }

    let clips: Vec<Value> = track
        .clips
        .iter()
        .map(|clip| serialize_audio_clip(clip, remap))
        .collect();
    obj.insert("clips".to_owned(), Value::Array(clips));

    Value::Object(obj)
}

pub(crate) fn serialize_video_clip(
    clip: &VideoClip,
    seq: &Sequence,
    comp_str_ids: &BTreeMap<Uuid, String>,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();
    for (k, v) in &clip.preserved {
        if k.starts_with("property.") {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    obj.insert("start".to_owned(), json!(finite_f64(clip.start)));
    obj.insert("length".to_owned(), json!(finite_f64(clip.length)));
    if let Some(link) = clip.link {
        obj.insert("link".to_owned(), json!(link));
    }

    let mut props = Map::new();
    for (k, v) in &clip.preserved {
        if let Some(pk) = k.strip_prefix("property.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(pk.to_owned(), cloned);
        }
    }
    props.insert("name".to_owned(), Value::String(clip.name.clone()));
    if let Some(media) = &clip.media {
        props.insert("media".to_owned(), Value::String(remap_str(media, remap)));
    }
    props.insert(
        "time".to_owned(),
        serialize_property_track(&clip.time_curve, remap),
    );

    obj.insert("properties".to_owned(), Value::Object(props));
    obj.insert(
        "object".to_owned(),
        serialize_layer(&clip.layer, seq, comp_str_ids, remap),
    );

    Value::Object(obj)
}

pub(crate) fn serialize_audio_clip(clip: &AudioClip, remap: &BTreeMap<String, String>) -> Value {
    let mut obj = Map::new();
    for (k, v) in &clip.preserved {
        if k.starts_with("property.") {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    obj.insert("start".to_owned(), json!(finite_f64(clip.start)));
    obj.insert("length".to_owned(), json!(finite_f64(clip.length)));
    if let Some(link) = clip.link {
        obj.insert("link".to_owned(), json!(link));
    }

    let mut props = Map::new();
    for (k, v) in &clip.preserved {
        if let Some(pk) = k.strip_prefix("property.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(pk.to_owned(), cloned);
        }
    }
    props.insert("name".to_owned(), Value::String(clip.name.clone()));
    if let Some(media) = &clip.media {
        props.insert("media".to_owned(), Value::String(remap_str(media, remap)));
    }
    props.insert(
        "time".to_owned(),
        serialize_property_track(&clip.time_curve, remap),
    );
    obj.insert("properties".to_owned(), Value::Object(props));

    let mut audio_obj = Map::new();
    let mut audio_props = Map::new();
    for (k, v) in &clip.audio.preserved {
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        if let Some(ok) = k.strip_prefix("object.") {
            audio_obj.insert(ok.to_owned(), cloned);
        } else {
            audio_props.insert(k.clone(), cloned);
        }
    }
    audio_props.insert("name".to_owned(), Value::String(clip.audio.name.clone()));
    audio_props.insert(
        "volume".to_owned(),
        serialize_property_track(&clip.audio.volume, remap),
    );
    audio_props.insert(
        "pan".to_owned(),
        serialize_property_track(&clip.audio.pan, remap),
    );
    audio_obj.insert("properties".to_owned(), Value::Object(audio_props));
    obj.insert("object".to_owned(), Value::Object(audio_obj));

    Value::Object(obj)
}

pub(super) fn serialize_layer(
    layer: &Layer,
    seq: &Sequence,
    comp_str_ids: &BTreeMap<Uuid, String>,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = if let LayerKind::Plugin { payload, .. } = &layer.kind {
        let mut base = match payload {
            Value::Object(map) => map.clone(),
            _ => Map::new(),
        };
        let mut base_val = Value::Object(base);
        remap_json_value(&mut base_val, remap);
        if let Value::Object(m) = base_val {
            base = m;
        } else {
            base = Map::new();
        }
        base
    } else {
        Map::new()
    };

    let is_plugin_layer = matches!(&layer.kind, LayerKind::Plugin { .. });
    for (k, v) in &layer.preserved {
        if k.starts_with("property.")
            || k.starts_with("childProperty.")
            || matches!(k.as_str(), "compId" | "offset")
        {
            continue;
        }
        if is_plugin_layer && obj.contains_key(k) {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    let mut props = obj
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (k, v) in &layer.preserved {
        if let Some(pk) = k.strip_prefix("property.") {
            if is_plugin_layer && props.contains_key(pk) {
                continue;
            }
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(pk.to_owned(), cloned);
        }
    }

    let mut child_props = props
        .get("childProperties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (k, v) in &layer.preserved {
        if let Some(ck) = k.strip_prefix("childProperty.") {
            if is_plugin_layer && child_props.contains_key(ck) {
                continue;
            }
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            child_props.insert(ck.to_owned(), cloned);
        }
    }

    props.insert("name".to_owned(), Value::String(layer.name.clone()));
    props.insert(
        "resolution".to_owned(),
        json!([layer.resolution[0].max(1), layer.resolution[1].max(1)]),
    );
    props.insert(
        "position".to_owned(),
        serialize_property_track(&layer.position, remap),
    );
    props.insert(
        "scale".to_owned(),
        serialize_property_track(&layer.scale, remap),
    );
    props.insert(
        "rotation".to_owned(),
        serialize_property_track(&layer.rotation, remap),
    );
    props.insert(
        "anchor".to_owned(),
        serialize_property_track(&layer.anchor, remap),
    );
    props.insert(
        "opacity".to_owned(),
        serialize_property_track(&layer.opacity, remap),
    );
    props.insert("blending".to_owned(), Value::String(layer.blending.clone()));

    if let Some(p3d) = seq.layer_3d_props.get(&layer.id) {
        props.insert(
            "depth".to_owned(),
            serialize_property_track(&p3d.depth, remap),
        );
        props.insert(
            "rotationX".to_owned(),
            serialize_property_track(&p3d.rotation_x, remap),
        );
        props.insert(
            "rotationY".to_owned(),
            serialize_property_track(&p3d.rotation_y, remap),
        );
        props.insert(
            "motionBlurAmount".to_owned(),
            serialize_property_track(&p3d.motion_blur_amount, remap),
        );
    }

    match &layer.kind {
        LayerKind::Video { .. } => {
            obj.insert("type".to_owned(), json!(0));
        }
        LayerKind::Adjustment => {
            obj.insert("type".to_owned(), json!(1));
        }
        LayerKind::Composite { layers } => {
            obj.insert("type".to_owned(), json!(2));
            // CRITICAL INVARIANT: ZoidCraft stores bottom-to-top; CM3 Composite.objects is top-to-bottom.
            let children: Vec<Value> = layers
                .iter()
                .rev()
                .map(|child| serialize_layer(child, seq, comp_str_ids, remap))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(children));
        }
        LayerKind::Image { image } => {
            obj.insert("type".to_owned(), json!(3));
            child_props.insert(
                "image".to_owned(),
                image
                    .as_ref()
                    .map_or(Value::Null, |k| Value::String(remap_str(k, remap))),
            );
        }
        LayerKind::Scene3D {
            objects,
            shadow_mode,
        } => {
            obj.insert("type".to_owned(), json!(4));
            obj.insert(
                "zoidiumShadowMode".to_owned(),
                Value::String(shadow_mode.as_str().to_owned()),
            );
            if let Some(bd) = seq.scene_backdrops.get(&layer.id) {
                child_props.insert("zoidiumBackdropMode".to_owned(), json!(bd.mode));
                child_props.insert(
                    "zoidiumBackdropColor".to_owned(),
                    json!([
                        finite_f64(bd.color[0]),
                        finite_f64(bd.color[1]),
                        finite_f64(bd.color[2])
                    ]),
                );
                child_props.insert(
                    "zoidiumBackdropOpacity".to_owned(),
                    serialize_property_value(&bd.opacity.value, remap),
                );
            }
            let constraint_ctx = SceneConstraintExportCtx::from_objects(objects);
            let objs: Vec<Value> = objects
                .iter()
                .map(|o| serialize_object3d_with_ctx(o, remap, Some(&constraint_ctx)))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(objs));
        }
        LayerKind::Generator => {
            obj.insert("type".to_owned(), json!(5));
        }
        LayerKind::Shape2D { shapes } => {
            obj.insert("type".to_owned(), json!(6));
            let arr: Vec<Value> = shapes
                .iter()
                .map(|s| serialize_shape2d_node(s, remap))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(arr));
        }
        LayerKind::Text2D {
            text,
            font,
            size,
            shapes,
        } => {
            obj.insert("type".to_owned(), json!(7));
            child_props.insert("text".to_owned(), Value::String(text.clone()));
            child_props.insert(
                "font".to_owned(),
                font.as_ref()
                    .map_or(Value::Null, |f| Value::String(remap_str(f, remap))),
            );
            child_props.insert("size".to_owned(), json!(*size));
            let arr: Vec<Value> = shapes
                .iter()
                .map(|s| serialize_shape2d_node(s, remap))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(arr));
        }
        LayerKind::PresetShape2D { preset, shapes } => {
            obj.insert("type".to_owned(), json!(8));
            let preset_id = match preset {
                PresetShapeKind::Ellipse => 0,
                PresetShapeKind::Rectangle => 1,
            };
            child_props.insert("preset".to_owned(), json!(preset_id));
            let arr: Vec<Value> = shapes
                .iter()
                .map(|s| serialize_shape2d_node(s, remap))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(arr));
        }
        LayerKind::CameraLayer { camera } => {
            obj.insert("type".to_owned(), json!(9));
            let objs: Vec<Value> = if let Some(cam_objs) = seq.camera_layer_objects.get(&layer.id)
                && !cam_objs.is_empty()
            {
                let constraint_ctx = SceneConstraintExportCtx::from_objects(cam_objs);
                cam_objs
                    .iter()
                    .map(|o| serialize_object3d_with_ctx(o, remap, Some(&constraint_ctx)))
                    .collect()
            } else {
                let mut cam_obj = Object3D::new("Camera", Object3DKind::Camera(camera.clone()));
                cam_obj.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);
                vec![serialize_object3d(&cam_obj, remap)]
            };
            obj.insert("objects".to_owned(), Value::Array(objs));
        }
        LayerKind::PrecompRef { composition_id } => {
            obj.insert("type".to_owned(), json!(2));
            let comp_str = comp_str_ids
                .get(composition_id)
                .cloned()
                .unwrap_or_else(|| format!("comp-{composition_id}"));
            obj.insert("compId".to_owned(), Value::String(comp_str));
            let offset = seq.precomp_offsets.get(&layer.id).copied().unwrap_or(0.0);
            obj.insert("offset".to_owned(), json!(finite_f64(offset)));
            obj.insert("objects".to_owned(), Value::Array(Vec::new()));
        }
        LayerKind::Plugin { plugin_type, .. } => {
            if let Ok(num) = plugin_type.parse::<u32>() {
                obj.insert("type".to_owned(), json!(num));
            } else {
                obj.insert("type".to_owned(), Value::String(plugin_type.clone()));
            }
        }
    }

    if !child_props.is_empty() {
        props.insert("childProperties".to_owned(), Value::Object(child_props));
    }
    obj.insert("properties".to_owned(), Value::Object(props));

    if let Some(masks) = seq.layer_masks.get(&layer.id)
        && !masks.is_empty()
    {
        let arr: Vec<Value> = masks
            .iter()
            .map(|m| serialize_layer_mask(m, remap))
            .collect();
        obj.insert("zoidiumMasks".to_owned(), Value::Array(arr));
        obj.remove("masks");
    } else {
        obj.remove("zoidiumMasks");
        obj.remove("masks");
    }

    let effects: Vec<Value> = layer
        .effects
        .iter()
        .map(|fx| serialize_effect(fx, seq, remap))
        .collect();
    obj.insert("effects".to_owned(), Value::Array(effects));

    Value::Object(obj)
}

pub(crate) fn serialize_layer_mask(mask: &LayerMask, remap: &BTreeMap<String, String>) -> Value {
    let mut obj = Map::new();
    for (k, v) in &mask.preserved {
        if k.starts_with("property.") {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    obj.insert("id".to_owned(), Value::String(mask.id.clone()));
    obj.insert("kind".to_owned(), Value::String("path".to_owned()));
    obj.insert(
        "mode".to_owned(),
        Value::String(
            match mask.mode {
                MaskMode::Add => "add",
                MaskMode::Subtract => "subtract",
                MaskMode::Intersect => "intersect",
            }
            .to_owned(),
        ),
    );
    obj.insert(
        "feather".to_owned(),
        json!(finite_f64(mask.feather).clamp(0.0, 500.0)),
    );
    obj.insert("expansion".to_owned(), json!(finite_f64(mask.expansion)));
    obj.insert(
        "opacity".to_owned(),
        json!(finite_f64(mask.opacity).clamp(0.0, 1.0)),
    );
    obj.insert("invert".to_owned(), json!(u32::from(mask.invert)));
    obj.insert("enabled".to_owned(), json!(u32::from(mask.enabled)));
    obj.insert("locked".to_owned(), Value::Bool(mask.locked));
    if !mask.outline_color.is_empty() {
        obj.insert(
            "outlineColor".to_owned(),
            Value::String(mask.outline_color.clone()),
        );
    } else {
        obj.remove("outlineColor");
    }
    if !mask.vertex_ids.is_empty() {
        obj.insert(
            "vertexIds".to_owned(),
            Value::Array(mask.vertex_ids.iter().cloned().map(Value::String).collect()),
        );
    }
    if !mask.vertex_modes.is_empty() {
        let mut vm = Map::new();
        for (k, v) in &mask.vertex_modes {
            vm.insert(k.clone(), Value::String(v.clone()));
        }
        obj.insert("vertexModes".to_owned(), Value::Object(vm));
    }

    let mut props = Map::new();
    for (k, v) in &mask.preserved {
        if let Some(pk) = k.strip_prefix("property.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(pk.to_owned(), cloned);
        }
    }
    props.insert("name".to_owned(), Value::String(mask.name.clone()));
    props.insert("enabled".to_owned(), json!(u32::from(mask.enabled)));
    props.insert("closed".to_owned(), json!(u32::from(mask.closed)));
    props.insert(
        "path".to_owned(),
        serialize_mask_path_track(&mask.path, remap),
    );
    obj.insert("properties".to_owned(), Value::Object(props));

    Value::Object(obj)
}

pub(super) fn serialize_mask_path_track(
    track: &PropertyTrack,
    remap: &BTreeMap<String, String>,
) -> Value {
    if !track.channels.is_empty() {
        return serialize_dynamic_group(track, remap);
    }

    let mut obj = Map::new();
    obj.insert("animated".to_owned(), Value::Bool(track.animated));
    if let Some(expr) = &track.expression {
        obj.insert("expression".to_owned(), Value::String(expr.clone()));
    }
    let kfs: Vec<Value> = if !track.keyframes.is_empty() {
        track
            .keyframes
            .iter()
            .map(|kf| serialize_keyframe(kf, remap))
            .collect()
    } else {
        vec![json!({
            "frame": 0,
            "value": serialize_property_value(&track.value, remap),
            "tween": 1
        })]
    };
    obj.insert("keyframes".to_owned(), Value::Array(kfs));
    Value::Object(obj)
}

pub(super) fn serialize_shape2d_node(
    node: &Shape2DNode,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();
    for (k, v) in &node.preserved {
        if k.starts_with("property.") {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        obj.insert(k.clone(), cloned);
    }

    let mut props = Map::new();
    for (k, v) in &node.preserved {
        if let Some(pk) = k.strip_prefix("property.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(pk.to_owned(), cloned);
        }
    }
    props.insert("name".to_owned(), Value::String(node.name.clone()));
    props.insert(
        "position".to_owned(),
        serialize_property_track(&node.position, remap),
    );
    props.insert(
        "rotation".to_owned(),
        serialize_property_track(&node.rotation, remap),
    );
    props.insert(
        "scale".to_owned(),
        serialize_property_track(&node.scale, remap),
    );
    obj.insert("properties".to_owned(), Value::Object(props));

    match &node.kind {
        Shape2DKind::Group { children } => {
            obj.insert("type".to_owned(), json!(0));
            let arr: Vec<Value> = children
                .iter()
                .map(|c| serialize_shape2d_node(c, remap))
                .collect();
            obj.insert("objects".to_owned(), Value::Array(arr));
        }
        Shape2DKind::Path { path } => {
            obj.insert("type".to_owned(), json!(1));
            obj.insert("path".to_owned(), path.clone());
        }
    }

    let ops: Vec<Value> = node
        .operations
        .iter()
        .map(|op| serialize_draw_op(op, remap))
        .collect();
    obj.insert("operations".to_owned(), Value::Array(ops));

    Value::Object(obj)
}

pub(super) fn serialize_draw_op(op: &DrawOp, remap: &BTreeMap<String, String>) -> Value {
    match op {
        DrawOp::Fill { color } => json!({
            "type": 0,
            "properties": {
                "color": serialize_property_track(color, remap)
            }
        }),
        DrawOp::Stroke {
            color,
            size,
            opacity,
        } => json!({
            "type": 1,
            "properties": {
                "color": serialize_property_track(color, remap),
                "size": serialize_property_track(size, remap),
                "opacity": serialize_property_track(opacity, remap)
            }
        }),
    }
}
