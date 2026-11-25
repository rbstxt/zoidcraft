use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};
use uuid::Uuid;

use crate::model::{
    AudioClip, AudioClipData, AudioTrack, CameraData, CompatMode, CustomProperty, DrawOp,
    EffectInstance, EffectKind, Layer, Layer3DProps, LayerKind, LayerMask, Marker, MaskMode,
    MaterialSpec, Object3D, Object3DKind, PresetShapeKind, PropertyTrack, PropertyValue,
    SceneBackdrop, Sequence, SequenceControl, SequenceControlKind, Shape2DKind, Shape2DNode,
    StaticBehavior, VideoClip, VideoTrack, cm3_blend_mode_from_index, normalize_blend_mode_token,
};

use super::effects::parse_effect;
use super::objects3d::{parse_camera_data, parse_object3d, remap_scene_constraint_targets};
use super::tracks::{
    build_fallback_time_curve, kf_frame, kf_value, parse_property_track, prop_f64, prop_str,
    prop_vec2, prop_vec3, rebase_time_curve_for_clipped_prefix, sanitize_display_name,
    value_as_bool, value_as_f64, value_as_f64_array, value_as_u32,
};

pub(super) fn parse_compositions(
    media_val: Option<&Value>,
    root_seq_val: Option<&Value>,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> (BTreeMap<Uuid, Sequence>, BTreeMap<String, Uuid>) {
    let mut compositions = BTreeMap::new();
    let mut comp_ids = BTreeMap::new();
    let Some(arr) = media_val.and_then(Value::as_array) else {
        return (compositions, comp_ids);
    };

    // Pass 1: discover all Precomp+ media entries and assign UUIDs so nested precomps can resolve.
    let mut comp_entries: Vec<(String, Uuid, &Map<String, Value>)> = Vec::new();
    for (idx, item) in arr.iter().enumerate() {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let explicit_comp_id = obj
            .get("comp")
            .and_then(|c| c.get("id"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        let is_legacy_comp = obj.get("isComp").and_then(Value::as_bool) == Some(true);
        if explicit_comp_id.is_none() && !is_legacy_comp {
            continue;
        }
        let comp_id = explicit_comp_id
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("comp-{idx}"));
        let uuid = Uuid::new_v4();
        comp_ids.insert(comp_id.clone(), uuid);
        comp_entries.push((comp_id, uuid, obj));
    }

    if comp_entries.is_empty() {
        return (compositions, comp_ids);
    }
    plugins.insert("precomp-plus".to_owned());

    // Inherit root sequence properties (resolution, rate, motionBlur) without root controls/markers.
    let base_props = root_seq_val
        .and_then(|s| s.get("properties"))
        .and_then(Value::as_object)
        .cloned()
        .map(|mut p| {
            p.remove("zoidiumControls");
            p.remove("markers");
            p
        })
        .unwrap_or_default();

    // Pass 2: parse each composition's `data[]` tracks into a `Sequence`.
    for (comp_id, uuid, obj) in comp_entries {
        let mut seq_props = base_props.clone();
        let comp_props = obj.get("properties").and_then(Value::as_object);
        let comp_name = comp_props
            .and_then(|p| prop_str(p, "name"))
            .or_else(|| {
                obj.get("title")
                    .and_then(Value::as_str)
                    .map(sanitize_display_name)
            })
            .or_else(|| {
                obj.get("name")
                    .and_then(Value::as_str)
                    .map(sanitize_display_name)
            })
            .unwrap_or_else(|| comp_id.clone());
        seq_props.insert("name".to_owned(), Value::String(comp_name));
        if let Some(bg) = comp_props.and_then(|p| {
            p.get("backgroundColor")
                .or_else(|| p.get("zoidcraftBackgroundColor"))
        }) {
            seq_props.insert("backgroundColor".to_owned(), bg.clone());
        }
        if let Some(markers) = comp_props.and_then(|p| p.get("markers")) {
            seq_props.insert("markers".to_owned(), markers.clone());
        }

        let mut video_tracks = Vec::new();
        let mut audio_tracks = Vec::new();
        if let Some(tracks) = obj.get("data").and_then(Value::as_array) {
            for track in tracks {
                let is_audio = track.get("type").and_then(value_as_u32) == Some(1);
                if is_audio {
                    audio_tracks.push(track.clone());
                } else {
                    video_tracks.push(track.clone());
                }
            }
        }

        let mut synth = Map::new();
        synth.insert("properties".to_owned(), Value::Object(seq_props));
        if let Some(len) = obj.get("compLength") {
            synth.insert("length".to_owned(), len.clone());
        }
        synth.insert("videoTracks".to_owned(), Value::Array(video_tracks));
        synth.insert("audioTracks".to_owned(), Value::Array(audio_tracks));
        if let Some(cl) = obj.get("comp").and_then(|c| c.get("clipLinks"))
            && !cl.is_null()
        {
            synth.insert("clipLinks".to_owned(), cl.clone());
        }
        if let Some(mc) = obj.get("comp").and_then(|c| c.get("markerColors"))
            && !mc.is_null()
        {
            synth.insert("markerColors".to_owned(), mc.clone());
        }

        let synth_val = Value::Object(synth);
        let mut comp_seq = parse_sequence(Some(&synth_val), &comp_ids, plugins, warnings);
        comp_seq.id = uuid;
        comp_seq
            .preserved
            .insert("comp_id".to_owned(), Value::String(comp_id));
        if let Some(src) = obj.get("comp").and_then(|c| c.get("compSource")) {
            comp_seq
                .preserved
                .insert("compSource".to_owned(), src.clone());
        }
        compositions.insert(uuid, comp_seq);
    }

    (compositions, comp_ids)
}

#[derive(Default)]
pub(crate) struct SequenceSideTables {
    pub(crate) layer_masks: BTreeMap<Uuid, Vec<LayerMask>>,
    pub(crate) precomp_offsets: BTreeMap<Uuid, f64>,
    pub(crate) camera_layer_objects: BTreeMap<Uuid, Vec<Object3D>>,
    pub(crate) scene_backdrops: BTreeMap<Uuid, SceneBackdrop>,
    pub(crate) layer_3d_props: BTreeMap<Uuid, Layer3DProps>,
    pub(crate) track_layer_input_ids: BTreeMap<Uuid, String>,
    pub(crate) bypassed_effects: BTreeSet<Uuid>,
}

pub(super) fn parse_sequence(
    seq_val: Option<&Value>,
    comp_ids: &BTreeMap<String, Uuid>,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> Sequence {
    let mut seq = Sequence::default();
    let Some(obj) = seq_val.and_then(Value::as_object) else {
        return seq;
    };

    let props = obj.get("properties").and_then(Value::as_object);
    if let Some(props) = props {
        if let Some(name) = prop_str(props, "name") {
            seq.name = name;
        }
        if let Some(res) = prop_vec2(props, "resolution") {
            seq.resolution = [
                res[0].round().max(1.0) as u32,
                res[1].round().max(1.0) as u32,
            ];
        }
        if let Some(rate) = prop_f64(props, "rate") {
            seq.fps = (rate.round() as u32).clamp(1, 240);
            if rate.is_finite() && rate > 0.0 && (rate - f64::from(seq.fps)).abs() > 1e-6 {
                seq.exact_fps = Some(rate);
            }
        }
        if let Some(mb) = props.get("motionBlur") {
            seq.motion_blur.enabled = value_as_u32(mb).unwrap_or(0) != 0;
        }
        if let Some(samples) = props.get("motionBlurSamples") {
            seq.motion_blur.samples = parse_property_track(samples, PropertyValue::Number(16.0));
        }
        if let Some(shutter) = props.get("motionBlurShutter") {
            seq.motion_blur.shutter = parse_property_track(shutter, PropertyValue::Number(0.5));
        }
        if let Some(tf_val) = props.get("motionBlurTrackFrame").filter(|v| !v.is_null()) {
            let track = parse_property_track(tf_val, PropertyValue::Option(0));
            let parsed_tf = match track.value {
                PropertyValue::Option(v) => Some(v),
                PropertyValue::Number(v) if v.is_finite() => Some(v.round().max(0.0) as u32),
                PropertyValue::Bool(b) => Some(u32::from(b)),
                _ => value_as_u32(tf_val),
            };
            if let Some(tf) = parsed_tf {
                seq.motion_blur.track_frame = tf;
                plugins.insert("camera-plus".to_owned());
            }
        }
        if let Some(sens_val) = props.get("motionBlurSensitivity").filter(|v| !v.is_null()) {
            let track = parse_property_track(sens_val, PropertyValue::Number(0.0));
            if let Some(sens) = track.value.as_f64() {
                seq.motion_blur.sensitivity = sens;
                plugins.insert("camera-plus".to_owned());
            }
        }
        if let Some(bg_val) = props
            .get("backgroundColor")
            .or_else(|| props.get("zoidcraftBackgroundColor"))
            .or_else(|| obj.get("backgroundColor"))
            .or_else(|| obj.get("zoidcraftBackgroundColor"))
            && let Some(bg) = parse_rgba_f32(bg_val)
        {
            seq.background_color = bg;
        }
        if let Some(markers_val) = props.get("markers") {
            let (markers, kf_colors) = parse_markers(markers_val);
            seq.markers = markers;
            if kf_colors.iter().any(|c| !c.is_null()) {
                seq.preserved
                    .insert("markerColors".to_owned(), Value::Array(kf_colors));
            }
        }
        if let Some(mc) = props
            .get("markerColors")
            .or_else(|| props.get("zoidcraftMarkerColors"))
            .filter(|v| v.is_array())
        {
            seq.preserved.insert("markerColors".to_owned(), mc.clone());
        }
        if let Some(controls_val) = props.get("zoidiumControls") {
            let (controls, next_id) = parse_sequence_controls(controls_val);
            if let Some(nid) = next_id {
                seq.preserved
                    .insert("zoidiumControls.nextId".to_owned(), Value::from(nid));
            }
            if !controls.is_empty() {
                plugins.insert("expressions-plus".to_owned());
                seq.controls = controls;
            }
        }
        for (k, v) in props {
            if !matches!(
                k.as_str(),
                "name"
                    | "resolution"
                    | "rate"
                    | "zoidcraftBackgroundColor"
                    | "motionBlur"
                    | "motionBlurSamples"
                    | "motionBlurShutter"
                    | "motionBlurTrackFrame"
                    | "motionBlurSensitivity"
                    | "markers"
                    | "markerColors"
                    | "zoidcraftMarkerColors"
                    | "zoidiumControls"
            ) {
                seq.preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    } else if let Some(bg_val) = obj
        .get("backgroundColor")
        .or_else(|| obj.get("zoidcraftBackgroundColor"))
        && let Some(bg) = parse_rgba_f32(bg_val)
    {
        seq.background_color = bg;
    }

    seq.length_frames = obj.get("length").and_then(value_as_f64).unwrap_or(0.0);

    let mut side_tables = SequenceSideTables::default();
    let mut video_clip_index_maps: Vec<Vec<Option<usize>>> = Vec::new();
    // CM3 videoTracks are already bottom-to-top (`0 -> N-1`).
    if let Some(tracks) = obj.get("videoTracks").and_then(Value::as_array) {
        for (idx, track_val) in tracks.iter().enumerate() {
            let (track, index_map) = parse_video_track(
                track_val,
                idx,
                seq.resolution,
                seq.fps,
                comp_ids,
                &mut side_tables,
                plugins,
                warnings,
            );
            seq.video_tracks.push(track);
            video_clip_index_maps.push(index_map);
        }
    }

    let mut audio_clip_index_maps: Vec<Vec<Option<usize>>> = Vec::new();
    if let Some(tracks) = obj.get("audioTracks").and_then(Value::as_array) {
        for (idx, track_val) in tracks.iter().enumerate() {
            let (track, index_map) = parse_audio_track(track_val, idx, seq.fps, warnings);
            seq.audio_tracks.push(track);
            audio_clip_index_maps.push(index_map);
        }
    }

    seq.layer_masks = side_tables.layer_masks;
    seq.precomp_offsets = side_tables.precomp_offsets;
    seq.camera_layer_objects = side_tables.camera_layer_objects;
    seq.scene_backdrops = side_tables.scene_backdrops;
    seq.layer_3d_props = side_tables.layer_3d_props;
    seq.track_layer_input_ids = side_tables.track_layer_input_ids;
    seq.bypassed_effects = side_tables.bypassed_effects;

    if !seq.length_frames.is_finite() || seq.length_frames <= 0.0 {
        let mut max_end = 0.0_f64;
        for track in &seq.video_tracks {
            for clip in &track.clips {
                max_end = max_end.max(clip.start + clip.length);
            }
        }
        for track in &seq.audio_tracks {
            for clip in &track.clips {
                max_end = max_end.max(clip.start + clip.length);
            }
        }
        seq.length_frames = max_end;
    }

    if let Some(clip_links_val) = obj.get("clipLinks") {
        resolve_clip_links(
            clip_links_val,
            &mut seq,
            &video_clip_index_maps,
            &audio_clip_index_maps,
        );
    }

    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "properties" | "length" | "videoTracks" | "audioTracks" | "clipLinks"
        ) {
            seq.preserved.insert(k.clone(), v.clone());
        }
    }

    seq
}

pub(super) fn parse_rgba_f32(val: &Value) -> Option<[f32; 4]> {
    if let Some(obj) = val.as_object() {
        if let Some(inner) = obj.get("value") {
            return parse_rgba_f32(inner);
        }
        if let Some(kfs) = obj.get("keyframes").and_then(Value::as_array)
            && let Some(first_val) = kfs.first().and_then(kf_value)
        {
            return parse_rgba_f32(first_val);
        }
    }
    if let Some(arr) = val.as_array() {
        if arr.len() >= 4 {
            let r = value_as_f64(&arr[0])?;
            let g = value_as_f64(&arr[1])?;
            let b = value_as_f64(&arr[2])?;
            let a = value_as_f64(&arr[3])?;
            return Some([r as f32, g as f32, b as f32, a as f32]);
        }
        if arr.len() == 3 {
            let r = value_as_f64(&arr[0])?;
            let g = value_as_f64(&arr[1])?;
            let b = value_as_f64(&arr[2])?;
            return Some([r as f32, g as f32, b as f32, 1.0]);
        }
    }
    if let Some(s) = val.as_str() {
        let hex = s.trim().strip_prefix('#')?;
        if hex.len() == 6 || hex.len() == 8 {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            let a = if hex.len() == 8 {
                u8::from_str_radix(&hex[6..8], 16).ok()?
            } else {
                255
            };
            return Some([
                f32::from(r) / 255.0,
                f32::from(g) / 255.0,
                f32::from(b) / 255.0,
                f32::from(a) / 255.0,
            ]);
        }
    }
    None
}

pub(super) fn parse_markers(val: &Value) -> (Vec<Marker>, Vec<Value>) {
    let mut out = Vec::new();
    let mut colors = Vec::new();
    let Some(kfs) = val.get("keyframes").and_then(Value::as_array) else {
        return (out, colors);
    };
    for (idx, kf) in kfs.iter().enumerate() {
        let frame = kf_frame(kf).unwrap_or(0.0);
        let label = match kf_value(kf) {
            Some(Value::String(s)) => sanitize_display_name(s),
            Some(Value::Number(n)) => n.to_string(),
            Some(Value::Bool(b)) => b.to_string(),
            _ => (idx + 1).to_string(),
        };
        let color = kf
            .get("color")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map_or(Value::Null, |s| Value::String(s.to_owned()));
        out.push(Marker { frame, label });
        colors.push(color);
    }
    (out, colors)
}

pub(super) fn parse_sequence_controls(val: &Value) -> (Vec<SequenceControl>, Option<u64>) {
    // Unwrap `{ value: "<json>" }` if serialized as a wrapped static property object.
    let unwrapped = if let Some(inner) = val.as_object().and_then(|o| o.get("value")) {
        if val.get("controls").is_none() {
            inner
        } else {
            val
        }
    } else {
        val
    };

    let parsed_owned;
    let obj_val = if let Some(s) = unwrapped.as_str() {
        if s.trim().is_empty() {
            return (Vec::new(), None);
        }
        match serde_json::from_str::<Value>(s) {
            Ok(v) => {
                parsed_owned = v;
                &parsed_owned
            }
            Err(_) => return (Vec::new(), None),
        }
    } else {
        unwrapped
    };

    let next_id = obj_val.get("nextId").and_then(Value::as_u64);
    let Some(arr) = obj_val.get("controls").and_then(Value::as_array) else {
        return (Vec::new(), next_id);
    };

    let mut out = Vec::new();
    for item in arr {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let type_obj = obj.get("type").and_then(Value::as_object);
        let props_obj = obj.get("properties").and_then(Value::as_object);

        let id = type_obj
            .and_then(|t| t.get("zoidcraftControlId"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .or_else(|| {
                type_obj
                    .and_then(|t| t.get("zoidiumControlId"))
                    .and_then(Value::as_str)
            })
            .or_else(|| obj.get("id").and_then(Value::as_str))
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let name = props_obj
            .and_then(|p| prop_str(p, "name"))
            .or_else(|| {
                obj.get("name")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .unwrap_or_else(|| "Control".to_owned());

        let kind_str = type_obj
            .and_then(|t| t.get("zoidiumControlKind"))
            .and_then(Value::as_str)
            .or_else(|| obj.get("kind").and_then(Value::as_str))
            .or_else(|| obj.get("type").and_then(Value::as_str))
            .unwrap_or("slider");

        let kind = match kind_str.trim().to_ascii_lowercase().as_str() {
            "angle" => SequenceControlKind::Angle,
            "checkbox" => SequenceControlKind::Checkbox,
            "color" => SequenceControlKind::Color,
            "point 2d" | "point2d" | "point_2d" | "vector2" => SequenceControlKind::Point2D,
            "point 3d" | "point3d" | "point_3d" | "vector3" => SequenceControlKind::Point3D,
            _ => SequenceControlKind::Slider,
        };

        let default_val = match kind {
            SequenceControlKind::Color => PropertyValue::Color([1.0, 1.0, 1.0]),
            SequenceControlKind::Point2D => PropertyValue::Vec2([0.0, 0.0]),
            SequenceControlKind::Point3D => PropertyValue::Vec3([0.0, 0.0, 0.0]),
            SequenceControlKind::Checkbox => PropertyValue::Option(0),
            _ => PropertyValue::Number(0.0),
        };

        let track = if obj.contains_key("keyframes") || obj.contains_key("objects") {
            parse_property_track(item, default_val)
        } else if let Some(v) = obj.get("value").or_else(|| obj.get("property")) {
            parse_property_track(v, default_val)
        } else if let Some(v) = type_obj.and_then(|t| t.get("value")) {
            parse_property_track(v, default_val)
        } else {
            PropertyTrack::constant(default_val)
        };

        let mut preserved = BTreeMap::new();
        if let Some(to) = type_obj {
            for meta_key in ["min", "max", "step", "scaleFactor", "decimals", "items"] {
                if let Some(mv) = to.get(meta_key) {
                    preserved.insert(meta_key.to_owned(), mv.clone());
                }
            }
            preserved.insert("type".to_owned(), Value::Object(to.clone()));
        }
        for (k, v) in obj {
            if !matches!(
                k.as_str(),
                "id" | "name"
                    | "kind"
                    | "type"
                    | "value"
                    | "property"
                    | "properties"
                    | "animated"
                    | "expression"
                    | "keyframes"
                    | "objects"
            ) {
                preserved.insert(k.clone(), v.clone());
            }
        }
        out.push(SequenceControl {
            id,
            name,
            kind,
            track,
            preserved,
        });
    }
    (out, next_id)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn parse_video_track(
    val: &Value,
    index: usize,
    seq_res: [u32; 2],
    fps: u32,
    comp_ids: &BTreeMap<String, Uuid>,
    side_tables: &mut SequenceSideTables,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> (VideoTrack, Vec<Option<usize>>) {
    let track_id = Uuid::new_v4();
    let Some(obj) = val.as_object() else {
        return (
            VideoTrack {
                id: track_id,
                name: format!("Video {}", index + 1),
                enabled: true,
                solo: false,
                locked: false,
                is_3d: false,
                clips: Vec::new(),
                preserved: BTreeMap::new(),
            },
            Vec::new(),
        );
    };

    let enabled = obj.get("enabled").and_then(value_as_bool).unwrap_or(true);
    let solo = obj
        .get("zoidiumSolo")
        .or_else(|| obj.get("solo"))
        .and_then(value_as_bool)
        .unwrap_or(false);
    let locked = obj
        .get("zoidiumLocked")
        .or_else(|| obj.get("locked"))
        .and_then(value_as_bool)
        .unwrap_or(false);
    let is_3d = obj.get("track3d").and_then(value_as_bool).unwrap_or(false);
    if is_3d {
        plugins.insert("camera-plus".to_owned());
    }
    if let Some(layer_input_id) = obj
        .get("_zoidiumLayerInputId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        side_tables
            .track_layer_input_ids
            .insert(track_id, layer_input_id.to_owned());
    }
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("Video {}", index + 1));

    let mut clips = Vec::new();
    let mut clip_index_map = Vec::new();
    if let Some(arr) = obj.get("clips").and_then(Value::as_array) {
        clip_index_map.reserve(arr.len());
        for clip_val in arr {
            if let Some(clip) = parse_video_clip(
                clip_val,
                seq_res,
                fps,
                comp_ids,
                side_tables,
                plugins,
                warnings,
            ) {
                clip_index_map.push(Some(clips.len()));
                clips.push(clip);
            } else {
                clip_index_map.push(None);
            }
        }
    }

    let mut preserved = BTreeMap::new();
    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type"
                | "name"
                | "enabled"
                | "solo"
                | "locked"
                | "zoidiumSolo"
                | "zoidiumLocked"
                | "track3d"
                | "clips"
        ) {
            preserved.insert(k.clone(), v.clone());
        }
    }

    (
        VideoTrack {
            id: track_id,
            name,
            enabled,
            solo,
            locked,
            is_3d,
            clips,
            preserved,
        },
        clip_index_map,
    )
}

pub(super) fn parse_audio_track(
    val: &Value,
    index: usize,
    fps: u32,
    warnings: &mut Vec<String>,
) -> (AudioTrack, Vec<Option<usize>>) {
    let Some(obj) = val.as_object() else {
        return (
            AudioTrack {
                id: Uuid::new_v4(),
                name: format!("Audio {}", index + 1),
                enabled: true,
                muted: false,
                solo: false,
                locked: false,
                clips: Vec::new(),
                preserved: BTreeMap::new(),
            },
            Vec::new(),
        );
    };

    let enabled = obj.get("enabled").and_then(value_as_bool).unwrap_or(true);
    // Zoidium persists audio track mute as `enabled: false` (no separate `muted` field).
    let muted = obj.get("muted").and_then(value_as_bool).unwrap_or(!enabled);
    let solo = obj
        .get("zoidiumSolo")
        .or_else(|| obj.get("solo"))
        .and_then(value_as_bool)
        .unwrap_or(false);
    let locked = obj
        .get("zoidiumLocked")
        .or_else(|| obj.get("locked"))
        .and_then(value_as_bool)
        .unwrap_or(false);
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("Audio {}", index + 1));

    let mut clips = Vec::new();
    let mut clip_index_map = Vec::new();
    if let Some(arr) = obj.get("clips").and_then(Value::as_array) {
        clip_index_map.reserve(arr.len());
        for clip_val in arr {
            if let Some(clip) = parse_audio_clip(clip_val, fps, warnings) {
                clip_index_map.push(Some(clips.len()));
                clips.push(clip);
            } else {
                clip_index_map.push(None);
            }
        }
    }

    let mut preserved = BTreeMap::new();
    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type"
                | "name"
                | "enabled"
                | "muted"
                | "solo"
                | "locked"
                | "zoidiumSolo"
                | "zoidiumLocked"
                | "clips"
        ) {
            preserved.insert(k.clone(), v.clone());
        }
    }

    (
        AudioTrack {
            id: Uuid::new_v4(),
            name,
            enabled,
            muted,
            solo,
            locked,
            clips,
            preserved,
        },
        clip_index_map,
    )
}

pub(super) fn shift_property_track(track: &mut PropertyTrack, delta: f64) {
    for kf in &mut track.keyframes {
        kf.frame += delta;
    }
    for ch in &mut track.channels {
        shift_property_track(ch, delta);
    }
}

pub(super) fn shift_json_keyframes(val: &mut Value, delta: f64) {
    match val {
        Value::Array(arr) => {
            for item in arr {
                shift_json_keyframes(item, delta);
            }
        }
        Value::Object(map) => {
            if let Some(kfs) = map.get_mut("keyframes").and_then(Value::as_array_mut) {
                for kf in kfs {
                    if let Some(kf_obj) = kf.as_object_mut() {
                        if let Some(f) = kf_obj.get("frame").and_then(value_as_f64) {
                            kf_obj.insert("frame".to_owned(), Value::from(f + delta));
                        } else if let Some(f) = kf_obj.get("t").and_then(value_as_f64) {
                            kf_obj.insert("t".to_owned(), Value::from(f + delta));
                        } else if let Some(f) = kf_obj.get("f").and_then(value_as_f64) {
                            kf_obj.insert("f".to_owned(), Value::from(f + delta));
                        }
                    }
                }
            }
            for (k, v) in map.iter_mut() {
                if k != "keyframes" {
                    shift_json_keyframes(v, delta);
                }
            }
        }
        _ => {}
    }
}

pub(super) fn shift_custom_properties(props: &mut [CustomProperty], delta: f64) {
    for cp in props {
        shift_property_track(&mut cp.track, delta);
    }
}

pub(super) fn shift_material(mat: &mut MaterialSpec, delta: f64) {
    match mat {
        MaterialSpec::SingleColor { color, .. } => shift_property_track(color, delta),
        MaterialSpec::Texture(m) => {
            shift_property_track(&mut m.repeat, delta);
            shift_property_track(&mut m.offset, delta);
            shift_property_track(&mut m.center, delta);
            shift_property_track(&mut m.rotation, delta);
        }
        MaterialSpec::Video(m) => {
            shift_property_track(&mut m.color, delta);
            shift_property_track(&mut m.opacity, delta);
        }
        MaterialSpec::CustomPbr(m) => {
            shift_property_track(&mut m.color, delta);
            shift_property_track(&mut m.emissive, delta);
            shift_property_track(&mut m.roughness, delta);
            shift_property_track(&mut m.metalness, delta);
            shift_property_track(&mut m.opacity, delta);
            shift_property_track(&mut m.repeat, delta);
            shift_property_track(&mut m.offset, delta);
            shift_property_track(&mut m.center, delta);
            shift_property_track(&mut m.rotation, delta);
        }
        MaterialSpec::CustomShader {
            custom_properties, ..
        } => shift_custom_properties(custom_properties, delta),
        MaterialSpec::PbrPlus(m) => {
            shift_property_track(&mut m.base_color, delta);
            shift_property_track(&mut m.emissive, delta);
            shift_property_track(&mut m.emissive_intensity, delta);
            shift_property_track(&mut m.roughness, delta);
            shift_property_track(&mut m.metalness, delta);
            shift_property_track(&mut m.opacity, delta);
            shift_property_track(&mut m.repeat, delta);
            shift_property_track(&mut m.offset, delta);
            shift_property_track(&mut m.center, delta);
            shift_property_track(&mut m.rotation, delta);
            if let Some(t) = m.normal_scale_track.as_mut() {
                shift_property_track(t, delta);
            }
            if let Some(t) = m.reflection_intensity_track.as_mut() {
                shift_property_track(t, delta);
            }
            for v in m.preserved.values_mut() {
                shift_json_keyframes(v, delta);
            }
        }
        MaterialSpec::Matcap(m) => {
            shift_property_track(&mut m.color, delta);
            shift_property_track(&mut m.brightness, delta);
            shift_property_track(&mut m.opacity, delta);
        }
        MaterialSpec::ImagePlus(m) => {
            shift_property_track(&mut m.opacity, delta);
            shift_property_track(&mut m.repeat, delta);
            shift_property_track(&mut m.offset, delta);
            shift_property_track(&mut m.center, delta);
            shift_property_track(&mut m.rotation, delta);
        }
        MaterialSpec::NodeMaterial { graph, .. } => shift_json_keyframes(graph, delta),
        MaterialSpec::Unknown { payload, .. } => shift_json_keyframes(payload, delta),
    }
}

pub(super) fn shift_camera_data(cam: &mut CameraData, delta: f64) {
    shift_property_track(&mut cam.fov, delta);
    shift_property_track(&mut cam.zoom, delta);
    shift_property_track(&mut cam.focal_length, delta);
    shift_property_track(&mut cam.film_offset_x, delta);
    shift_property_track(&mut cam.film_offset_y, delta);
    shift_property_track(&mut cam.shake_amplitude, delta);
    shift_property_track(&mut cam.shake_speed, delta);
    if let Some(dof) = cam.dof.as_mut() {
        if let Some(t) = dof.enabled_track.as_mut() {
            shift_property_track(t, delta);
        }
        shift_property_track(&mut dof.focus_distance, delta);
        shift_property_track(&mut dof.aperture, delta);
        shift_property_track(&mut dof.focus_area_width, delta);
        shift_property_track(&mut dof.near_blur, delta);
        shift_property_track(&mut dof.far_blur, delta);
    }
    if let Some(vib) = cam.vibrate.as_mut() {
        shift_property_track(&mut vib.position_amplitude, delta);
        shift_property_track(&mut vib.rotation_amplitude, delta);
        shift_property_track(&mut vib.frequency, delta);
        shift_property_track(&mut vib.position_frequency, delta);
        shift_property_track(&mut vib.rotation_frequency, delta);
    }
    if let Some(mb) = cam.motion_blur.as_mut() {
        shift_property_track(&mut mb.shutter, delta);
    }
}

pub(super) fn shift_object_3d(obj: &mut Object3D, delta: f64) {
    shift_property_track(&mut obj.enabled, delta);
    shift_property_track(&mut obj.position, delta);
    shift_property_track(&mut obj.rotation, delta);
    shift_property_track(&mut obj.scale, delta);
    if let Some(mat) = obj.material.as_mut() {
        shift_material(mat, delta);
    }
    for mat in obj.material_slots.values_mut() {
        shift_material(mat, delta);
    }
    if let Some(track_to) = obj.constraints.track_to.as_mut() {
        shift_property_track(&mut track_to.influence, delta);
    }
    if let Some(follow) = obj.constraints.follow.as_mut() {
        shift_property_track(&mut follow.position_offset, delta);
        shift_property_track(&mut follow.rotation_offset, delta);
        shift_property_track(&mut follow.scale_offset, delta);
    }
    match &mut obj.kind {
        Object3DKind::Group {
            custom_properties,
            children,
            ..
        } => {
            shift_custom_properties(custom_properties, delta);
            for child in children {
                shift_object_3d(child, delta);
            }
        }
        Object3DKind::Text3D(t) => {
            shift_property_track(&mut t.character_index_offset, delta);
            shift_property_track(&mut t.character_count_override, delta);
        }
        Object3DKind::Light(l) => {
            shift_property_track(&mut l.color, delta);
            shift_property_track(&mut l.ground_color, delta);
            shift_property_track(&mut l.target, delta);
            shift_property_track(&mut l.intensity, delta);
            shift_property_track(&mut l.angle, delta);
        }
        Object3DKind::Camera(cam) => shift_camera_data(cam, delta),
        Object3DKind::Particles(p) => shift_property_track(&mut p.time, delta),
        Object3DKind::Model(m) => {
            for track in m.model_properties.values_mut() {
                shift_property_track(track, delta);
            }
        }
        Object3DKind::Dummy { payload, .. } => shift_json_keyframes(payload, delta),
        Object3DKind::Shape3D(_) => {}
    }
    for v in obj.preserved.values_mut() {
        shift_json_keyframes(v, delta);
    }
}

pub(super) fn shift_shape_2d(node: &mut Shape2DNode, delta: f64) {
    shift_property_track(&mut node.position, delta);
    shift_property_track(&mut node.rotation, delta);
    shift_property_track(&mut node.scale, delta);
    for op in &mut node.operations {
        match op {
            DrawOp::Fill { color } => shift_property_track(color, delta),
            DrawOp::Stroke {
                color,
                size,
                opacity,
            } => {
                shift_property_track(color, delta);
                shift_property_track(size, delta);
                shift_property_track(opacity, delta);
            }
        }
    }
    if let Shape2DKind::Group { children } = &mut node.kind {
        for child in children {
            shift_shape_2d(child, delta);
        }
    }
}

pub(super) fn shift_effect(fx: &mut EffectInstance, delta: f64) {
    shift_property_track(&mut fx.enabled, delta);
    for track in fx.params.values_mut() {
        shift_property_track(track, delta);
    }
    match &mut fx.kind {
        EffectKind::CustomShader {
            custom_properties, ..
        } => shift_custom_properties(custom_properties, delta),
        EffectKind::EffectGroup {
            custom_properties,
            children,
        } => {
            shift_custom_properties(custom_properties, delta);
            for child in children {
                shift_effect(child, delta);
            }
        }
        EffectKind::Standard(_) | EffectKind::Plugin { .. } => {}
    }
}

pub(super) fn shift_layer(layer: &mut Layer, delta: f64, side_tables: &mut SequenceSideTables) {
    shift_property_track(&mut layer.position, delta);
    shift_property_track(&mut layer.scale, delta);
    shift_property_track(&mut layer.rotation, delta);
    shift_property_track(&mut layer.anchor, delta);
    shift_property_track(&mut layer.opacity, delta);
    for fx in &mut layer.effects {
        shift_effect(fx, delta);
    }
    if let Some(masks) = side_tables.layer_masks.get_mut(&layer.id) {
        for mask in masks {
            shift_property_track(&mut mask.path, delta);
        }
    }
    if let Some(cam_objs) = side_tables.camera_layer_objects.get_mut(&layer.id) {
        for obj in cam_objs {
            shift_object_3d(obj, delta);
        }
    }
    if let Some(backdrop) = side_tables.scene_backdrops.get_mut(&layer.id) {
        shift_property_track(&mut backdrop.opacity, delta);
    }
    if let Some(p3d) = side_tables.layer_3d_props.get_mut(&layer.id) {
        shift_property_track(&mut p3d.depth, delta);
        shift_property_track(&mut p3d.rotation_x, delta);
        shift_property_track(&mut p3d.rotation_y, delta);
        shift_property_track(&mut p3d.motion_blur_amount, delta);
    }
    if let Some(tp2d) = layer.preserved.get_mut("zoidiumTextPlus2D") {
        shift_json_keyframes(tp2d, delta);
    }
    match &mut layer.kind {
        LayerKind::Composite { layers } => {
            for child in layers {
                shift_layer(child, delta, side_tables);
            }
        }
        LayerKind::Scene3D { objects, .. } => {
            for obj in objects {
                shift_object_3d(obj, delta);
            }
        }
        LayerKind::Shape2D { shapes }
        | LayerKind::Text2D { shapes, .. }
        | LayerKind::PresetShape2D { shapes, .. } => {
            for shape in shapes {
                shift_shape_2d(shape, delta);
            }
        }
        LayerKind::CameraLayer { camera } => shift_camera_data(camera, delta),
        LayerKind::Plugin { payload, .. } => shift_json_keyframes(payload, delta),
        LayerKind::Video { .. }
        | LayerKind::Adjustment
        | LayerKind::Image { .. }
        | LayerKind::Generator
        | LayerKind::PrecompRef { .. } => {}
    }
}

pub(super) fn shift_audio_clip_data(audio: &mut AudioClipData, delta: f64) {
    shift_property_track(&mut audio.volume, delta);
    shift_property_track(&mut audio.pan, delta);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_video_clip(
    val: &Value,
    seq_res: [u32; 2],
    fps: u32,
    comp_ids: &BTreeMap<String, Uuid>,
    side_tables: &mut SequenceSideTables,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> Option<VideoClip> {
    let obj = val.as_object();
    let raw_start = obj
        .and_then(|o| o.get("start"))
        .and_then(value_as_f64)
        .unwrap_or(0.0);
    let raw_length = obj
        .and_then(|o| o.get("length"))
        .and_then(value_as_f64)
        .unwrap_or(0.0);
    let (start, length) = if raw_start < 0.0 {
        let remaining = raw_length + raw_start;
        if remaining <= 0.0 {
            warnings.push(format!(
                "dropped video clip ending at or before frame 0 (start={raw_start}, length={raw_length})"
            ));
            return None;
        }
        warnings.push(format!(
            "normalized negative video clip start ({raw_start}) to 0.0"
        ));
        (0.0, remaining)
    } else {
        (raw_start, raw_length.max(0.0))
    };
    let link = obj.and_then(|o| o.get("link")).and_then(value_as_u32);
    let props = obj
        .and_then(|o| o.get("properties"))
        .and_then(Value::as_object);

    let name = props
        .and_then(|p| prop_str(p, "name"))
        .or_else(|| {
            obj.and_then(|o| o.get("name"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| "Clip".to_owned());
    let media = props
        .and_then(|p| p.get("media"))
        .or_else(|| obj.and_then(|o| o.get("media")))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let mut time_curve = props.and_then(|p| p.get("time")).map_or_else(
        || build_fallback_time_curve(obj, raw_length, fps),
        |v| parse_property_track(v, PropertyValue::Number(0.0)),
    );
    if raw_start < 0.0 {
        rebase_time_curve_for_clipped_prefix(&mut time_curve, -raw_start, length, fps);
    }
    time_curve.static_behavior = StaticBehavior::Linear;

    let mut layer = obj.and_then(|o| o.get("object")).map_or_else(
        || Layer::new("Layer", LayerKind::Generator, seq_res),
        |layer_val| {
            parse_layer(
                layer_val,
                seq_res,
                media.clone(),
                comp_ids,
                side_tables,
                plugins,
                warnings,
            )
        },
    );
    if raw_start < 0.0 {
        shift_layer(&mut layer, raw_start, side_tables);
    }

    let mut preserved = BTreeMap::new();
    if let Some(p) = props {
        for (k, v) in p {
            if !matches!(k.as_str(), "name" | "media" | "time") {
                preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }
    if let Some(o) = obj {
        for (k, v) in o {
            if !matches!(
                k.as_str(),
                "start" | "length" | "link" | "properties" | "object"
            ) {
                preserved.insert(k.clone(), v.clone());
            }
        }
    }

    Some(VideoClip {
        id: Uuid::new_v4(),
        name,
        start,
        length,
        link,
        media,
        time_curve,
        layer,
        preserved,
    })
}

pub(crate) fn parse_audio_clip(
    val: &Value,
    fps: u32,
    warnings: &mut Vec<String>,
) -> Option<AudioClip> {
    let obj = val.as_object();
    let raw_start = obj
        .and_then(|o| o.get("start"))
        .and_then(value_as_f64)
        .unwrap_or(0.0);
    let raw_length = obj
        .and_then(|o| o.get("length"))
        .and_then(value_as_f64)
        .unwrap_or(0.0);
    let (start, length) = if raw_start < 0.0 {
        let remaining = raw_length + raw_start;
        if remaining <= 0.0 {
            warnings.push(format!(
                "dropped audio clip ending at or before frame 0 (start={raw_start}, length={raw_length})"
            ));
            return None;
        }
        warnings.push(format!(
            "normalized negative audio clip start ({raw_start}) to 0.0"
        ));
        (0.0, remaining)
    } else {
        (raw_start, raw_length.max(0.0))
    };
    let link = obj.and_then(|o| o.get("link")).and_then(value_as_u32);
    let props = obj
        .and_then(|o| o.get("properties"))
        .and_then(Value::as_object);

    let name = props
        .and_then(|p| prop_str(p, "name"))
        .or_else(|| {
            obj.and_then(|o| o.get("name"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| "Audio Clip".to_owned());
    let media = props
        .and_then(|p| p.get("media"))
        .or_else(|| obj.and_then(|o| o.get("media")))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let mut time_curve = props.and_then(|p| p.get("time")).map_or_else(
        || build_fallback_time_curve(obj, raw_length, fps),
        |v| parse_property_track(v, PropertyValue::Number(0.0)),
    );
    if raw_start < 0.0 {
        rebase_time_curve_for_clipped_prefix(&mut time_curve, -raw_start, length, fps);
    }
    time_curve.static_behavior = StaticBehavior::Linear;

    let mut audio = AudioClipData::default();
    if let Some(audio_obj) = obj.and_then(|o| o.get("object")).and_then(Value::as_object) {
        if let Some(audio_props) = audio_obj.get("properties").and_then(Value::as_object) {
            if let Some(n) = prop_str(audio_props, "name") {
                audio.name = n;
            }
            if let Some(v) = audio_props.get("volume") {
                audio.volume = parse_property_track(v, PropertyValue::Number(1.0));
            }
            if let Some(p) = audio_props.get("pan") {
                audio.pan = parse_property_track(p, PropertyValue::Number(0.0));
            }
            for (k, v) in audio_props {
                if !matches!(k.as_str(), "name" | "volume" | "pan") {
                    audio.preserved.insert(k.clone(), v.clone());
                }
            }
        }
        // Preserve non-`properties` fields on `audio_clip.object` (P2-4).
        for (k, v) in audio_obj {
            if k != "properties" {
                audio.preserved.insert(format!("object.{k}"), v.clone());
            }
        }
    }
    if raw_start < 0.0 {
        shift_audio_clip_data(&mut audio, raw_start);
    }

    let mut preserved = BTreeMap::new();
    if let Some(p) = props {
        for (k, v) in p {
            if !matches!(k.as_str(), "name" | "media" | "time") {
                preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }
    if let Some(o) = obj {
        for (k, v) in o {
            if !matches!(
                k.as_str(),
                "start" | "length" | "link" | "properties" | "object"
            ) {
                preserved.insert(k.clone(), v.clone());
            }
        }
    }

    Some(AudioClip {
        id: Uuid::new_v4(),
        name,
        start,
        length,
        link,
        media,
        time_curve,
        audio,
        preserved,
    })
}

pub(super) fn parse_blending_token(val: &Value) -> Option<String> {
    if let Some(s) = val.as_str() {
        return (!s.trim().is_empty()).then(|| normalize_blend_mode_token(s));
    }
    if let Some(idx) = val.as_u64().and_then(|n| usize::try_from(n).ok()) {
        return cm3_blend_mode_from_index(idx).map(ToOwned::to_owned);
    }
    if let Some(inner) = val.as_object().and_then(|o| o.get("value")) {
        return parse_blending_token(inner);
    }
    None
}

pub(crate) fn parse_layer_masks(val: &Value) -> Vec<LayerMask> {
    let Some(arr) = val.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(arr.len());
    for (idx, item) in arr.iter().enumerate() {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let props = obj.get("properties").and_then(Value::as_object);
        let id = obj
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("mask-{}", idx + 1));
        let name = props
            .and_then(|p| prop_str(p, "name"))
            .or_else(|| {
                obj.get("name")
                    .and_then(Value::as_str)
                    .map(sanitize_display_name)
            })
            .unwrap_or_else(|| format!("Mask {}", idx + 1));
        let top_enabled = obj.get("enabled").and_then(value_as_bool).unwrap_or(true);
        let prop_enabled = props
            .and_then(|p| p.get("enabled"))
            .and_then(value_as_bool)
            .unwrap_or(true);
        let enabled = top_enabled && prop_enabled;
        let mode = obj
            .get("mode")
            .or_else(|| props.and_then(|p| p.get("mode")))
            .and_then(Value::as_str)
            .map_or(MaskMode::Add, MaskMode::from_token);
        let invert = obj
            .get("invert")
            .or_else(|| obj.get("inverted"))
            .or_else(|| props.and_then(|p| p.get("invert").or_else(|| p.get("inverted"))))
            .and_then(value_as_bool)
            .unwrap_or(false);
        let feather = obj
            .get("feather")
            .or_else(|| props.and_then(|p| p.get("feather")))
            .and_then(value_as_f64)
            .unwrap_or(0.0);
        let expansion = obj
            .get("expansion")
            .or_else(|| props.and_then(|p| p.get("expansion")))
            .and_then(value_as_f64)
            .unwrap_or(0.0);
        let opacity = obj
            .get("opacity")
            .or_else(|| props.and_then(|p| p.get("opacity")))
            .and_then(value_as_f64)
            .unwrap_or(1.0);
        let closed = props
            .and_then(|p| p.get("closed"))
            .or_else(|| obj.get("closed"))
            .and_then(value_as_bool)
            .unwrap_or_else(|| {
                obj.contains_key("vertices")
                    || obj.contains_key("points")
                    || props.is_some_and(|p| p.contains_key("vertices") || p.contains_key("points"))
            });
        let locked = obj.get("locked").and_then(value_as_bool).unwrap_or(false);
        let outline_color = obj
            .get("outlineColor")
            .or_else(|| obj.get("outline_color"))
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default()
            .to_owned();
        let vertex_ids: Vec<String> = obj
            .get("vertexIds")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let mut vertex_modes = BTreeMap::new();
        if let Some(vm) = obj.get("vertexModes").and_then(Value::as_object) {
            for (k, v) in vm {
                if let Some(s) = v.as_str() {
                    vertex_modes.insert(k.clone(), s.to_owned());
                }
            }
        }
        // Mask path is either `{ objects: [6*N scalar tracks] }`, `{ animated, keyframes: [{ frame, value: [6*N numbers] }] }`,
        // or shorthand `vertices` / `points` array `[[x, y], ...]` / `[{ x, y, ... }, ...]`.
        let path = props
            .and_then(|p| p.get("path"))
            .or_else(|| obj.get("path"))
            .map(parse_mask_path_track)
            .or_else(|| {
                obj.get("vertices")
                    .or_else(|| obj.get("points"))
                    .or_else(|| props.and_then(|p| p.get("vertices").or_else(|| p.get("points"))))
                    .and_then(parse_mask_vertices_shorthand)
            })
            .unwrap_or_else(|| {
                PropertyTrack::constant(PropertyValue::Json(Value::Array(Vec::new())))
            });
        let mut preserved = BTreeMap::new();
        for (k, v) in obj {
            if !matches!(
                k.as_str(),
                "id" | "name"
                    | "kind"
                    | "mode"
                    | "feather"
                    | "expansion"
                    | "opacity"
                    | "invert"
                    | "inverted"
                    | "enabled"
                    | "locked"
                    | "outlineColor"
                    | "outline_color"
                    | "vertexIds"
                    | "vertexModes"
                    | "properties"
            ) {
                preserved.insert(k.clone(), v.clone());
            }
        }
        if let Some(p) = props {
            for (k, v) in p {
                if !matches!(k.as_str(), "name" | "enabled" | "closed" | "path") {
                    preserved.insert(format!("property.{k}"), v.clone());
                }
            }
        }
        out.push(LayerMask {
            id,
            name,
            enabled,
            mode,
            invert,
            feather,
            expansion,
            opacity,
            closed,
            locked,
            outline_color,
            vertex_ids,
            vertex_modes,
            path,
            preserved,
        });
    }
    out
}

pub(super) fn parse_mask_path_track(val: &Value) -> PropertyTrack {
    if let Some(obj) = val.as_object()
        && let Some(channels_arr) = obj.get("objects").and_then(Value::as_array)
        && obj.get("keyframes").is_none()
    {
        let channels: Vec<PropertyTrack> = channels_arr
            .iter()
            .map(|c| parse_property_track(c, PropertyValue::Number(0.0)))
            .collect();
        let animated = channels.iter().any(|c| c.animated)
            || obj
                .get("animated")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let expression = obj
            .get("expression")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let combined_arr: Vec<Value> = channels
            .iter()
            .map(|c| Value::from(c.value.as_f64().unwrap_or(0.0)))
            .collect();
        return PropertyTrack {
            animated,
            expression,
            static_behavior: StaticBehavior::Constant,
            value: PropertyValue::Json(Value::Array(combined_arr)),
            keyframes: Vec::new(),
            channels,
        };
    }
    parse_property_track(val, PropertyValue::Json(Value::Array(Vec::new())))
}

pub(super) fn parse_mask_vertices_shorthand(val: &Value) -> Option<PropertyTrack> {
    let arr = val.as_array()?;
    let mut flat = Vec::new();
    if arr.iter().all(Value::is_number) {
        for n in arr {
            flat.push(Value::from(value_as_f64(n).unwrap_or(0.0)));
        }
    } else {
        for v in arr {
            if let Some(pair) = v.as_array()
                && pair.len() >= 2
            {
                let x = value_as_f64(&pair[0]).unwrap_or(0.0);
                let y = value_as_f64(&pair[1]).unwrap_or(0.0);
                let in_x = pair.get(2).and_then(value_as_f64).unwrap_or(0.0);
                let in_y = pair.get(3).and_then(value_as_f64).unwrap_or(0.0);
                let out_x = pair.get(4).and_then(value_as_f64).unwrap_or(0.0);
                let out_y = pair.get(5).and_then(value_as_f64).unwrap_or(0.0);
                flat.extend([
                    Value::from(x),
                    Value::from(y),
                    Value::from(in_x),
                    Value::from(in_y),
                    Value::from(out_x),
                    Value::from(out_y),
                ]);
            } else if let Some(vo) = v.as_object() {
                let x = vo.get("x").and_then(value_as_f64).unwrap_or(0.0);
                let y = vo.get("y").and_then(value_as_f64).unwrap_or(0.0);
                let in_x = vo
                    .get("inX")
                    .or_else(|| vo.get("ix"))
                    .and_then(value_as_f64)
                    .unwrap_or(0.0);
                let in_y = vo
                    .get("inY")
                    .or_else(|| vo.get("iy"))
                    .and_then(value_as_f64)
                    .unwrap_or(0.0);
                let out_x = vo
                    .get("outX")
                    .or_else(|| vo.get("ox"))
                    .and_then(value_as_f64)
                    .unwrap_or(0.0);
                let out_y = vo
                    .get("outY")
                    .or_else(|| vo.get("oy"))
                    .and_then(value_as_f64)
                    .unwrap_or(0.0);
                flat.extend([
                    Value::from(x),
                    Value::from(y),
                    Value::from(in_x),
                    Value::from(in_y),
                    Value::from(out_x),
                    Value::from(out_y),
                ]);
            }
        }
    }
    Some(PropertyTrack::constant(PropertyValue::Json(Value::Array(
        flat,
    ))))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_layer(
    val: &Value,
    seq_res: [u32; 2],
    clip_media: Option<String>,
    comp_ids: &BTreeMap<String, Uuid>,
    side_tables: &mut SequenceSideTables,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> Layer {
    let Some(obj) = val.as_object() else {
        return Layer::new("Layer", LayerKind::Generator, seq_res);
    };

    let props = obj.get("properties").and_then(Value::as_object);
    let child_props = props
        .and_then(|p| p.get("childProperties"))
        .and_then(Value::as_object)
        .or_else(|| obj.get("childProperties").and_then(Value::as_object));

    let resolution = props
        .and_then(|p| prop_vec2(p, "resolution"))
        .map_or(seq_res, |r| {
            [r[0].round().max(1.0) as u32, r[1].round().max(1.0) as u32]
        });

    let layer_type = obj.get("type");
    let mut parsed_camera_objects: Option<Vec<Object3D>> = None;
    let mut parsed_precomp_offset: Option<f64> = None;
    let mut parsed_backdrop: Option<SceneBackdrop> = None;

    let (default_name, kind) = if let Some(type_str) = layer_type.and_then(Value::as_str) {
        if type_str == "zoidium:shape-plus/shape-layer" {
            plugins.insert("shape-plus".to_owned());
            (
                "Shape+",
                LayerKind::Plugin {
                    plugin_type: type_str.to_owned(),
                    payload: val.clone(),
                },
            )
        } else {
            if let Some((plugin_id, _)) = type_str
                .strip_prefix("zoidium:")
                .and_then(|r| r.split_once('/'))
                .filter(|(p, b)| !p.is_empty() && !b.is_empty())
            {
                plugins.insert(plugin_id.to_owned());
            }
            warnings.push(format!("preserved unknown layer type {type_str:?}"));
            (
                "Plugin Layer",
                LayerKind::Plugin {
                    plugin_type: type_str.to_owned(),
                    payload: val.clone(),
                },
            )
        }
    } else {
        match layer_type.and_then(value_as_u32) {
            Some(0) => ("Layer", LayerKind::Video { media: clip_media }),
            Some(1) => ("Adjustment", LayerKind::Adjustment),
            Some(2) => {
                if let Some(comp_id_str) = obj
                    .get("compId")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                {
                    plugins.insert("precomp-plus".to_owned());
                    let composition_id = if let Some(&uuid) = comp_ids.get(comp_id_str) {
                        uuid
                    } else {
                        warnings.push(format!(
                            "precomp layer references unknown composition id {comp_id_str:?}"
                        ));
                        Uuid::new_v4()
                    };
                    parsed_precomp_offset =
                        Some(obj.get("offset").and_then(value_as_f64).unwrap_or(0.0));
                    ("Precomp", LayerKind::PrecompRef { composition_id })
                } else {
                    // CRITICAL INVARIANT: CM3 Composite.objects are top-to-bottom (`0` = top, `M-1` = bottom).
                    // Reverse into ZoidCraft's unified bottom-to-top order (`0` = bottom, `M-1` = top).
                    let mut children = Vec::new();
                    if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                        for child_val in arr.iter().rev() {
                            children.push(parse_layer(
                                child_val,
                                resolution,
                                None,
                                comp_ids,
                                side_tables,
                                plugins,
                                warnings,
                            ));
                        }
                    }
                    ("Composite", LayerKind::Composite { layers: children })
                }
            }
            Some(3) => {
                let image = child_props
                    .and_then(|cp| cp.get("image"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                ("Image", LayerKind::Image { image })
            }
            Some(4) => {
                let shadow_mode = obj
                    .get("zoidiumShadowMode")
                    .or_else(|| props.and_then(|p| p.get("zoidiumShadowMode")))
                    .and_then(Value::as_str)
                    .map_or(CompatMode::Legacy, CompatMode::from_token);
                if let Some(cp) = child_props
                    && (cp.contains_key("zoidiumBackdropMode")
                        || cp.contains_key("zoidiumBackdropColor")
                        || cp.contains_key("zoidiumBackdropOpacity"))
                {
                    let mut bd = SceneBackdrop::default();
                    if let Some(m) = cp.get("zoidiumBackdropMode").and_then(value_as_u32) {
                        bd.mode = m;
                    }
                    if let Some(c) = prop_vec3(cp, "zoidiumBackdropColor") {
                        bd.color = c;
                    }
                    if let Some(o) = cp.get("zoidiumBackdropOpacity") {
                        bd.opacity = parse_property_track(o, PropertyValue::Number(1.0));
                    }
                    parsed_backdrop = Some(bd);
                }
                let mut objects = Vec::new();
                if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                    for item in arr {
                        objects.push(parse_object3d(item, plugins, warnings));
                    }
                }
                remap_scene_constraint_targets(&mut objects);
                (
                    "Scene",
                    LayerKind::Scene3D {
                        objects,
                        shadow_mode,
                    },
                )
            }
            Some(5) => ("Layer", LayerKind::Generator),
            Some(6) => {
                let shapes = parse_shapes2d(obj.get("objects"));
                ("Shape", LayerKind::Shape2D { shapes })
            }
            Some(7) => {
                let text = child_props
                    .and_then(|cp| prop_str(cp, "text"))
                    .unwrap_or_else(|| "text".to_owned());
                let font = child_props
                    .and_then(|cp| cp.get("font"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                let size = child_props
                    .and_then(|cp| prop_f64(cp, "size").or_else(|| prop_f64(cp, "fontSize")))
                    .map_or(100, |v| v.round().max(1.0) as u32);
                let mut shapes = parse_shapes2d(obj.get("objects"));
                if shapes.is_empty()
                    && let Some(color_val) = child_props.and_then(|cp| cp.get("color"))
                {
                    let color =
                        parse_property_track(color_val, PropertyValue::Color([1.0, 1.0, 1.0]));
                    shapes.push(Shape2DNode {
                        id: Uuid::new_v4(),
                        name: "Text Fill".to_owned(),
                        kind: Shape2DKind::Group {
                            children: Vec::new(),
                        },
                        position: PropertyTrack::vec2([0.0, 0.0]),
                        rotation: PropertyTrack::number(0.0),
                        scale: PropertyTrack::vec2([1.0, 1.0]),
                        operations: vec![DrawOp::Fill { color }],
                        preserved: BTreeMap::new(),
                    });
                }
                if obj.contains_key("zoidiumTextPlus2D") {
                    plugins.insert("text-plus".to_owned());
                }
                (
                    "Text",
                    LayerKind::Text2D {
                        text,
                        font,
                        size,
                        shapes,
                    },
                )
            }
            Some(8) => {
                let preset = match child_props
                    .and_then(|cp| cp.get("preset"))
                    .and_then(value_as_u32)
                    .unwrap_or(0)
                {
                    1 => PresetShapeKind::Rectangle,
                    _ => PresetShapeKind::Ellipse,
                };
                let shapes = parse_shapes2d(obj.get("objects"));
                ("Preset shape", LayerKind::PresetShape2D { preset, shapes })
            }
            Some(9) => {
                plugins.insert("camera-plus".to_owned());
                let mut objects = Vec::new();
                if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                    for item in arr {
                        objects.push(parse_object3d(item, plugins, warnings));
                    }
                }
                remap_scene_constraint_targets(&mut objects);
                let primary_cam = objects
                    .iter()
                    .find_map(|o| match &o.kind {
                        Object3DKind::Camera(cam) => Some((**cam).clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| parse_camera_data(props, 1, plugins));
                if !objects.is_empty() {
                    parsed_camera_objects = Some(objects);
                }
                (
                    "Camera",
                    LayerKind::CameraLayer {
                        camera: Box::new(primary_cam),
                    },
                )
            }
            _ => {
                let type_label = layer_type
                    .map(Value::to_string)
                    .unwrap_or_else(|| "unknown".to_owned());
                warnings.push(format!("preserved unknown layer type {type_label}"));
                (
                    "Plugin Layer",
                    LayerKind::Plugin {
                        plugin_type: type_label,
                        payload: val.clone(),
                    },
                )
            }
        }
    };

    let mut layer = Layer::new(default_name, kind, resolution);
    if let Some(objs) = parsed_camera_objects {
        side_tables.camera_layer_objects.insert(layer.id, objs);
    }
    if let Some(offset) = parsed_precomp_offset {
        side_tables.precomp_offsets.insert(layer.id, offset);
    }
    if let Some(bd) = parsed_backdrop {
        side_tables.scene_backdrops.insert(layer.id, bd);
    }

    if let Some(props) = props {
        if let Some(n) = prop_str(props, "name") {
            layer.name = n;
        }
        if let Some(v) = props.get("position") {
            layer.position = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
        }
        if let Some(v) = props.get("scale") {
            layer.scale = parse_property_track(v, PropertyValue::Vec2([1.0, 1.0]));
        }
        if let Some(v) = props.get("rotation") {
            layer.rotation = parse_property_track(v, PropertyValue::Number(0.0));
        }
        if let Some(v) = props.get("anchor") {
            layer.anchor = parse_property_track(v, PropertyValue::Vec2([0.0, 0.0]));
        }
        if let Some(v) = props.get("opacity") {
            layer.opacity = parse_property_track(v, PropertyValue::Number(1.0));
        }
        if let Some(b) = props.get("blending").and_then(parse_blending_token) {
            layer.blending = b;
        }
        if props.contains_key("depth")
            || props.contains_key("rotationX")
            || props.contains_key("rotationY")
            || props.contains_key("motionBlurAmount")
        {
            plugins.insert("camera-plus".to_owned());
            let mut p3d = Layer3DProps::default();
            if let Some(v) = props.get("depth") {
                p3d.depth = parse_property_track(v, PropertyValue::Number(0.0));
            }
            if let Some(v) = props.get("rotationX") {
                p3d.rotation_x = parse_property_track(v, PropertyValue::Number(0.0));
            }
            if let Some(v) = props.get("rotationY") {
                p3d.rotation_y = parse_property_track(v, PropertyValue::Number(0.0));
            }
            if let Some(v) = props.get("motionBlurAmount") {
                p3d.motion_blur_amount = parse_property_track(v, PropertyValue::Number(1.0));
            }
            side_tables.layer_3d_props.insert(layer.id, p3d);
        }
        for (k, v) in props {
            if !matches!(
                k.as_str(),
                "name"
                    | "childProperties"
                    | "resolution"
                    | "position"
                    | "scale"
                    | "rotation"
                    | "anchor"
                    | "opacity"
                    | "blending"
                    | "zoidiumShadowMode"
                    | "depth"
                    | "rotationX"
                    | "rotationY"
                    | "motionBlurAmount"
            ) {
                layer.preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }

    if let Some(cp) = child_props {
        for (k, v) in cp {
            let known_child_key = match &layer.kind {
                LayerKind::Image { .. } => k == "image",
                LayerKind::Text2D { .. } => matches!(k.as_str(), "text" | "font" | "size"),
                LayerKind::PresetShape2D { .. } => k == "preset",
                LayerKind::Scene3D { .. } => matches!(
                    k.as_str(),
                    "zoidiumBackdropMode" | "zoidiumBackdropColor" | "zoidiumBackdropOpacity"
                ),
                _ => false,
            };
            if !known_child_key {
                layer
                    .preserved
                    .insert(format!("childProperty.{k}"), v.clone());
            }
        }
    }

    if matches!(layer.kind, LayerKind::Generator)
        && !layer.preserved.contains_key("childProperty.color")
        && let Some(c) = props
            .and_then(|p| p.get("color"))
            .or_else(|| obj.get("color"))
    {
        layer
            .preserved
            .insert("childProperty.color".to_owned(), c.clone());
    }

    if let Some(masks_val) = obj.get("zoidiumMasks").or_else(|| obj.get("masks")) {
        let masks = parse_layer_masks(masks_val);
        if !masks.is_empty() {
            plugins.insert("shape-plus".to_owned());
            side_tables.layer_masks.insert(layer.id, masks);
        }
    }

    if let Some(effects) = obj.get("effects").and_then(Value::as_array) {
        for eff_val in effects {
            layer
                .effects
                .push(parse_effect(eff_val, side_tables, plugins, warnings));
        }
    }

    let plugin_payload_obj = match &layer.kind {
        LayerKind::Plugin {
            payload: Value::Object(map),
            ..
        } => Some(map),
        _ => None,
    };
    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type"
                | "properties"
                | "childProperties"
                | "objects"
                | "effects"
                | "zoidiumShadowMode"
                | "zoidiumMasks"
                | "masks"
        ) && !plugin_payload_obj.is_some_and(|map| map.contains_key(k))
        {
            layer.preserved.insert(k.clone(), v.clone());
        }
    }

    layer
}

pub(super) fn parse_shapes2d(val: Option<&Value>) -> Vec<Shape2DNode> {
    let Some(arr) = val.and_then(Value::as_array) else {
        return Vec::new();
    };
    arr.iter().map(parse_shape2d_node).collect()
}

pub(super) fn parse_shape2d_node(val: &Value) -> Shape2DNode {
    let Some(obj) = val.as_object() else {
        return Shape2DNode {
            id: Uuid::new_v4(),
            name: "Shape".to_owned(),
            kind: Shape2DKind::Group {
                children: Vec::new(),
            },
            position: PropertyTrack::vec2([0.0, 0.0]),
            rotation: PropertyTrack::number(0.0),
            scale: PropertyTrack::vec2([1.0, 1.0]),
            operations: Vec::new(),
            preserved: BTreeMap::new(),
        };
    };

    let props = obj.get("properties").and_then(Value::as_object);
    let shape_type = obj.get("type").and_then(value_as_u32).unwrap_or(0);
    let kind = if shape_type == 1 {
        let raw_path = obj.get("path").cloned().unwrap_or(Value::Null);
        let path = if raw_path.is_null() {
            // CM3 Shape2D `type: 1` rectangle node with `size: [w, h]` or `width`/`height`
            // (without explicit SVG/path commands): synthesize closed rectangle path.
            let rect_size = props
                .and_then(|p| prop_vec2(p, "size"))
                .or_else(|| obj.get("size").and_then(value_as_f64_array::<2>))
                .or_else(|| {
                    let w = props
                        .and_then(|p| prop_f64(p, "width"))
                        .or_else(|| obj.get("width").and_then(value_as_f64))?;
                    let h = props
                        .and_then(|p| prop_f64(p, "height"))
                        .or_else(|| obj.get("height").and_then(value_as_f64))
                        .unwrap_or(w);
                    Some([w, h])
                });
            if let Some([w, h]) = rect_size {
                let hw = (w.abs() * 0.5).max(0.5);
                let hh = (h.abs() * 0.5).max(0.5);
                serde_json::json!({
                    "commands": [
                        { "type": "M", "x": -hw, "y": -hh },
                        { "type": "L", "x": hw, "y": -hh },
                        { "type": "L", "x": hw, "y": hh },
                        { "type": "L", "x": -hw, "y": hh },
                        { "type": "Z" }
                    ]
                })
            } else {
                raw_path
            }
        } else {
            raw_path
        };
        Shape2DKind::Path { path }
    } else {
        Shape2DKind::Group {
            children: parse_shapes2d(obj.get("objects")),
        }
    };

    let name = props
        .and_then(|p| prop_str(p, "name"))
        .unwrap_or_else(|| if shape_type == 1 { "Path" } else { "Group" }.to_owned());
    let position = props.and_then(|p| p.get("position")).map_or_else(
        || PropertyTrack::vec2([0.0, 0.0]),
        |v| parse_property_track(v, PropertyValue::Vec2([0.0, 0.0])),
    );
    let rotation = props.and_then(|p| p.get("rotation")).map_or_else(
        || PropertyTrack::number(0.0),
        |v| parse_property_track(v, PropertyValue::Number(0.0)),
    );
    let scale = props.and_then(|p| p.get("scale")).map_or_else(
        || PropertyTrack::vec2([1.0, 1.0]),
        |v| parse_property_track(v, PropertyValue::Vec2([1.0, 1.0])),
    );

    let mut operations = Vec::new();
    if let Some(ops) = obj.get("operations").and_then(Value::as_array) {
        for op in ops {
            if let Some(parsed) = parse_draw_op(op) {
                operations.push(parsed);
            }
        }
    }
    if operations.is_empty()
        && let Some(color_val) = props
            .and_then(|p| {
                p.get("color")
                    .or_else(|| p.get("fill"))
                    .or_else(|| p.get("fillColor"))
            })
            .or_else(|| obj.get("color"))
    {
        operations.push(DrawOp::Fill {
            color: parse_property_track(color_val, PropertyValue::Color([1.0, 1.0, 1.0])),
        });
    }

    let mut preserved = BTreeMap::new();
    if let Some(p) = props {
        for (k, v) in p {
            if !matches!(k.as_str(), "name" | "position" | "rotation" | "scale") {
                preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }
    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type" | "properties" | "objects" | "path" | "operations"
        ) {
            preserved.insert(k.clone(), v.clone());
        }
    }

    Shape2DNode {
        id: Uuid::new_v4(),
        name,
        kind,
        position,
        rotation,
        scale,
        operations,
        preserved,
    }
}

/// Parses `draw.fill` (`0`) and `draw.stroke` (`1`).
/// Strictly maps color channels by array index `[0]=R, [1]=G, [2]=B` (ignoring CM3's swapped UI labels).
pub(super) fn parse_draw_op(val: &Value) -> Option<DrawOp> {
    let obj = val.as_object()?;
    let op_type = obj.get("type").and_then(value_as_u32)?;
    let props = obj.get("properties").and_then(Value::as_object);

    match op_type {
        0 => {
            let color = props.and_then(|p| p.get("color")).map_or_else(
                || PropertyTrack::color([1.0, 1.0, 1.0]),
                |v| parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0])),
            );
            Some(DrawOp::Fill { color })
        }
        1 => {
            let color = props.and_then(|p| p.get("color")).map_or_else(
                || PropertyTrack::color([0.0, 0.0, 0.0]),
                |v| parse_property_track(v, PropertyValue::Color([0.0, 0.0, 0.0])),
            );
            let size = props.and_then(|p| p.get("size")).map_or_else(
                || PropertyTrack::number(1.0),
                |v| parse_property_track(v, PropertyValue::Number(1.0)),
            );
            let opacity = props.and_then(|p| p.get("opacity")).map_or_else(
                || PropertyTrack::number(1.0),
                |v| parse_property_track(v, PropertyValue::Number(1.0)),
            );
            Some(DrawOp::Stroke {
                color,
                size,
                opacity,
            })
        }
        _ => None,
    }
}

pub(super) fn parse_clip_link_address(parts: &[Value]) -> Option<(u64, usize, usize)> {
    match parts.len() {
        // Canonical CM3 / Zoidium 5-element address: [0, 1|2, trackIdx, 0, clipIdx]
        5 => {
            let track_kind = parts[1].as_u64()?;
            let track_idx = usize::try_from(parts[2].as_u64()?).ok()?;
            let clip_idx = usize::try_from(parts[4].as_u64()?).ok()?;
            Some((track_kind, track_idx, clip_idx))
        }
        // Legacy pre-core22 3-element address: [trackType(0=video|1=audio), trackIdx, clipIdx]
        3 => {
            let legacy_kind = parts[0].as_u64()?;
            let track_kind = match legacy_kind {
                0 => 1,
                1 => 2,
                _ => return None,
            };
            let track_idx = usize::try_from(parts[1].as_u64()?).ok()?;
            let clip_idx = usize::try_from(parts[2].as_u64()?).ok()?;
            Some((track_kind, track_idx, clip_idx))
        }
        // Backwards-compatible synthetic 4-element address: [0, 1|2, trackIdx, clipIdx]
        4 => {
            let track_kind = parts[1].as_u64()?;
            let track_idx = usize::try_from(parts[2].as_u64()?).ok()?;
            let clip_idx = usize::try_from(parts[3].as_u64()?).ok()?;
            Some((track_kind, track_idx, clip_idx))
        }
        _ => None,
    }
}

pub(super) fn resolve_clip_links(
    clip_links_val: &Value,
    seq: &mut Sequence,
    video_clip_index_maps: &[Vec<Option<usize>>],
    audio_clip_index_maps: &[Vec<Option<usize>>],
) {
    if let Some(seed) = clip_links_val.get("seed") {
        seq.preserved
            .insert("clipLinks.seed".to_owned(), seed.clone());
    }

    // Reset any stale or orphaned `clip.link` values before resolving active link groups.
    for vt in &mut seq.video_tracks {
        for clip in &mut vt.clips {
            clip.link = None;
        }
    }
    for at in &mut seq.audio_tracks {
        for clip in &mut at.clips {
            clip.link = None;
        }
    }

    let Some(links) = clip_links_val.get("links").and_then(Value::as_object) else {
        return;
    };

    for (link_key, entry) in links {
        let Ok(link_id) = link_key.parse::<u32>() else {
            continue;
        };
        let Some(clips) = entry.get("clips").and_then(Value::as_array) else {
            continue;
        };
        // Collect resolved member locations first so singleton groups (< 2 valid clips) are pruned cleanly.
        let mut members: Vec<(u64, usize, usize, Uuid)> = Vec::new();
        let mut seen_uuids: BTreeSet<Uuid> = BTreeSet::new();
        for addr in clips {
            let Some(parts) = addr.as_array() else {
                continue;
            };
            let Some((track_kind, track_idx, raw_clip_idx)) = parse_clip_link_address(parts) else {
                continue;
            };
            match track_kind {
                1 => {
                    let Some(kept_clip_idx) = video_clip_index_maps
                        .get(track_idx)
                        .and_then(|m| m.get(raw_clip_idx))
                        .copied()
                        .flatten()
                    else {
                        continue;
                    };
                    if let Some(clip) = seq
                        .video_tracks
                        .get(track_idx)
                        .and_then(|t| t.clips.get(kept_clip_idx))
                        && seen_uuids.insert(clip.id)
                    {
                        members.push((1, track_idx, kept_clip_idx, clip.id));
                    }
                }
                2 => {
                    let Some(kept_clip_idx) = audio_clip_index_maps
                        .get(track_idx)
                        .and_then(|m| m.get(raw_clip_idx))
                        .copied()
                        .flatten()
                    else {
                        continue;
                    };
                    if let Some(clip) = seq
                        .audio_tracks
                        .get(track_idx)
                        .and_then(|t| t.clips.get(kept_clip_idx))
                        && seen_uuids.insert(clip.id)
                    {
                        members.push((2, track_idx, kept_clip_idx, clip.id));
                    }
                }
                _ => {}
            }
        }

        // Zoidium object-history.js:17-31: prune singleton link groups (< 2 resolved clips).
        if members.len() < 2 {
            continue;
        }

        let mut uuids = Vec::with_capacity(members.len());
        for (track_kind, track_idx, kept_clip_idx, clip_id) in members {
            match track_kind {
                1 => {
                    if let Some(clip) = seq
                        .video_tracks
                        .get_mut(track_idx)
                        .and_then(|t| t.clips.get_mut(kept_clip_idx))
                        && clip.link.is_none()
                    {
                        clip.link = Some(link_id);
                    }
                }
                2 => {
                    if let Some(clip) = seq
                        .audio_tracks
                        .get_mut(track_idx)
                        .and_then(|t| t.clips.get_mut(kept_clip_idx))
                        && clip.link.is_none()
                    {
                        clip.link = Some(link_id);
                    }
                }
                _ => {}
            }
            uuids.push(clip_id);
        }
        seq.clip_links.insert(link_id, uuids);
    }
}
