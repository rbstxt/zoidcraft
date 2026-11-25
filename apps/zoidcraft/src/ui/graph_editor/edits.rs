//! Keyframe editing operations, copy/paste JSON and target resolution.

use egui::{Pos2, Shape, Stroke, pos2};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use zc_session::{Command, Session, SessionProperty};

use super::*;

pub fn keyframe_exists_in_doc(
    doc: &zc_session::SessionDocument,
    target_id: &str,
    property: &str,
    local_frame: f64,
) -> bool {
    let matches_prop = |props: &BTreeMap<String, SessionProperty>| -> bool {
        props.get(property).is_some_and(|p| {
            p.keyframes
                .iter()
                .any(|k| (k.frame - local_frame).abs() < 1e-4)
        })
    };
    if target_id == "sequence" {
        return matches_prop(&doc.sequence_properties);
    }
    for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
        for clip in &track.clips {
            if clip.id == target_id {
                return matches_prop(&clip.properties);
            }
            for obj in &clip.objects_3d {
                if obj.id == target_id {
                    return matches_prop(&obj.properties);
                }
            }
            for fx in &clip.effects {
                if fx.id == target_id {
                    return matches_prop(&fx.properties);
                }
            }
        }
    }
    false
}

pub fn prune_stale_selected_keys(session: &Session, state: &mut GraphEditorState) {
    if state.selected_keys.is_empty() {
        return;
    }
    let doc = session.document();
    state
        .selected_keys
        .retain(|gk| keyframe_exists_in_doc(doc, &gk.target_id, &gk.property, gk.local_frame()));
}

pub fn expand_linked_axis_selection(
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    anchor: &GraphKeyId,
) {
    if anchor.channel_index.is_none() {
        return;
    }
    for track in tracks {
        if track.target_id == anchor.target_id
            && track.property == anchor.property
            && state.is_track_visible(&track.track_key)
        {
            for kf in &track.keyframes {
                if (kf.local_frame - anchor.local_frame()).abs() < 1e-4 {
                    let sibling = GraphKeyId::new(
                        &track.target_id,
                        &track.property,
                        track.channel_index,
                        kf.local_frame,
                    );
                    if !state.selected_keys.contains(&sibling) {
                        state.selected_keys.push(sibling);
                    }
                }
            }
        }
    }
}

pub fn set_continuous_on_selected(
    session: &mut Session,
    state: &mut GraphEditorState,
    continuous: bool,
    action: &mut GraphEditorAction,
) {
    if state.selected_keys.is_empty() {
        return;
    }
    if session
        .begin_transaction(if continuous {
            "Auto Tangents"
        } else {
            "Break Tangents"
        })
        .is_ok()
    {
        let mut changed = false;
        for kid in &state.selected_keys {
            let entry = state.handles.entry(kid.clone()).or_default();
            entry.continuous = continuous;
            if session
                .set_keyframe_bezier_handles(
                    &kid.target_id,
                    &kid.property,
                    kid.local_frame(),
                    [entry.in_tangent, entry.out_tangent],
                    entry.continuous,
                    None,
                )
                .is_ok()
            {
                changed = true;
            }
        }
        let _ = session.commit_transaction(changed);
        if changed {
            action.curves_modified = true;
        }
    }
}

pub fn insert_key_at_playhead(
    session: &mut Session,
    state: &GraphEditorState,
    tracks: &[CurveTrack],
    action: &mut GraphEditorAction,
) {
    let cur_abs = session.playback.current_frame.round();
    let mut visited: BTreeSet<(String, String)> = BTreeSet::new();
    if session.begin_transaction("Insert Graph Keyframes").is_ok() {
        let mut changed = false;
        for track in tracks {
            if !state.is_track_visible(&track.track_key) || track.is_expression {
                continue;
            }
            if !visited.insert((track.target_id.clone(), track.property.clone())) {
                continue;
            }
            let local_f = (cur_abs - track.frame_offset).max(0.0);
            let value = if track.channel_index.is_some() {
                let mut vec_vals = Vec::new();
                for ch_track in tracks.iter().filter(|t| {
                    t.target_id == track.target_id
                        && t.property == track.property
                        && t.channel_index.is_some()
                }) {
                    let ch_idx = ch_track.channel_index.unwrap_or(0);
                    if vec_vals.len() <= ch_idx {
                        vec_vals.resize(ch_idx + 1, Value::from(0.0));
                    }
                    let v = ch_track.evaluate_at(local_f, &state.handles);
                    vec_vals[ch_idx] = Value::from(v);
                }
                Value::Array(vec_vals)
            } else {
                Value::from(track.evaluate_at(local_f, &state.handles))
            };
            if session
                .execute(Command::SetKeyframe {
                    target_id: track.target_id.clone(),
                    property: track.property.clone(),
                    frame: local_f,
                    value,
                    tween: None,
                })
                .is_ok()
            {
                changed = true;
            }
        }
        let _ = session.commit_transaction(changed);
        if changed {
            action.curves_modified = true;
        }
    }
}

pub fn delete_selected_graph_keys(
    session: &mut Session,
    state: &mut GraphEditorState,
    action: &mut GraphEditorAction,
) {
    prune_stale_selected_keys(session, state);
    if state.selected_keys.is_empty() {
        return;
    }
    let mut seen: BTreeSet<(String, String, u64)> = BTreeSet::new();
    let mut deleted: Vec<(String, String, f64)> = Vec::new();
    if session.begin_transaction("Delete Graph Keyframes").is_ok() {
        for gk in &state.selected_keys {
            let dedup = (gk.target_id.clone(), gk.property.clone(), gk.frame_bits);
            if !seen.insert(dedup) {
                continue;
            }
            let f = gk.local_frame();
            if session
                .execute(Command::DeleteKeyframe {
                    target_id: gk.target_id.clone(),
                    property: gk.property.clone(),
                    frame: f,
                })
                .is_ok()
            {
                deleted.push((gk.target_id.clone(), gk.property.clone(), f));
            }
        }
        let changed = !deleted.is_empty();
        let _ = session.commit_transaction(changed);
        if changed {
            state.selected_keys.retain(|gk| {
                !deleted.iter().any(|(t, p, f)| {
                    gk.target_id == *t && gk.property == *p && (gk.local_frame() - *f).abs() < 1e-6
                })
            });
            state.handles.retain(|gk, _| {
                !deleted.iter().any(|(t, p, f)| {
                    gk.target_id == *t && gk.property == *p && (gk.local_frame() - *f).abs() < 1e-6
                })
            });
            action.curves_modified = true;
        }
    }
}

pub fn copy_selected_graph_keys(state: &mut GraphEditorState, tracks: &[CurveTrack]) {
    let mut raw = Vec::new();
    let mut min_abs = f64::INFINITY;
    for kid in &state.selected_keys {
        if let Some((track, kf)) = find_track_keyframe(tracks, kid) {
            min_abs = min_abs.min(track.frame_offset + kf.local_frame);
            raw.push((track, kf));
        }
    }
    if raw.is_empty() || !min_abs.is_finite() {
        return;
    }
    state.copied_keys = raw
        .into_iter()
        .map(|(track, kf)| CopiedGraphKey {
            target_id: track.target_id.clone(),
            property: track.property.clone(),
            channel_index: track.channel_index,
            rel_frame: (track.frame_offset + kf.local_frame) - min_abs,
            value: kf.value,
            full_value: kf.full_value.clone(),
            tween: kf.tween,
            control_points: kf.control_points,
            continuous_tangent: kf.continuous_tangent,
        })
        .collect();
}

pub fn paste_graph_keys_at_playhead(
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    action: &mut GraphEditorAction,
) {
    if state.copied_keys.is_empty() {
        return;
    }
    let playhead = session.playback.current_frame.round();
    if session.begin_transaction("Paste Graph Keyframes").is_ok() {
        let mut changed = false;
        let mut new_sel = Vec::new();
        for ck in &state.copied_keys {
            let offset = tracks
                .iter()
                .find(|t| t.target_id == ck.target_id && t.property == ck.property)
                .map_or(0.0, |t| t.frame_offset);
            let local_f = (playhead + ck.rel_frame - offset).max(0.0);
            let value = write_channel_value(&ck.full_value, ck.channel_index, ck.value);
            if session
                .execute(Command::SetKeyframe {
                    target_id: ck.target_id.clone(),
                    property: ck.property.clone(),
                    frame: local_f,
                    value,
                    tween: Some(ck.tween),
                })
                .is_ok()
            {
                let _ = session.set_keyframe_bezier_handles(
                    &ck.target_id,
                    &ck.property,
                    local_f,
                    ck.control_points,
                    ck.continuous_tangent,
                    Some(ck.tween),
                );
                changed = true;
                new_sel.push(GraphKeyId::new(
                    &ck.target_id,
                    &ck.property,
                    ck.channel_index,
                    local_f,
                ));
            }
        }
        let _ = session.commit_transaction(changed);
        if changed {
            state.selected_keys = new_sel;
            action.curves_modified = true;
        }
    }
}

pub fn resolve_active_curve_target_with_channel(
    state: &GraphEditorState,
    tracks: &[CurveTrack],
) -> Option<(String, String, f64, Option<usize>)> {
    if let Some(sel) = state.selected_keys.first() {
        let offset = tracks
            .iter()
            .find(|t| t.target_id == sel.target_id && t.property == sel.property)
            .map_or(0.0, |t| t.frame_offset);
        return Some((
            sel.target_id.clone(),
            sel.property.clone(),
            offset,
            sel.channel_index,
        ));
    }
    if let Some(ref solo) = state.solo_track
        && let Some(t) = tracks.iter().find(|t| &t.track_key == solo)
    {
        return Some((
            t.target_id.clone(),
            t.property.clone(),
            t.frame_offset,
            t.channel_index,
        ));
    }
    if let Some((ref tid, ref prop)) = state.focused_property {
        let matching = tracks
            .iter()
            .find(|t| {
                &t.target_id == tid && &t.property == prop && state.is_track_visible(&t.track_key)
            })
            .or_else(|| {
                tracks
                    .iter()
                    .find(|t| &t.target_id == tid && &t.property == prop)
            });
        let offset = matching.map_or(0.0, |t| t.frame_offset);
        let ch = matching.and_then(|t| t.channel_index);
        return Some((tid.clone(), prop.clone(), offset, ch));
    }
    tracks
        .iter()
        .find(|t| state.is_track_visible(&t.track_key))
        .or_else(|| tracks.first())
        .map(|t| {
            (
                t.target_id.clone(),
                t.property.clone(),
                t.frame_offset,
                t.channel_index,
            )
        })
}

pub fn resolve_active_curve_target(
    state: &GraphEditorState,
    tracks: &[CurveTrack],
) -> Option<(String, String, f64)> {
    resolve_active_curve_target_with_channel(state, tracks)
        .map(|(tid, prop, offset, _)| (tid, prop, offset))
}

pub fn find_session_property<'a>(
    doc: &'a zc_session::SessionDocument,
    target_id: &str,
    prop_name: &str,
) -> Option<&'a SessionProperty> {
    if target_id == "sequence" {
        return doc.sequence_properties.get(prop_name);
    }
    for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
        for clip in &track.clips {
            if clip.id == target_id {
                return clip.properties.get(prop_name);
            }
            if let Some(obj) = clip.objects_3d.iter().find(|o| o.id == target_id) {
                return obj.properties.get(prop_name);
            }
            if let Some(fx) = clip.effects.iter().find(|e| e.id == target_id) {
                return fx.properties.get(prop_name);
            }
        }
    }
    None
}

pub fn numeric_array_len(v: &Value) -> Option<usize> {
    let arr = v.as_array()?;
    if (2..=4).contains(&arr.len()) && arr.iter().all(Value::is_number) {
        Some(arr.len())
    } else {
        None
    }
}

pub fn vector_property_dim(prop: &SessionProperty) -> Option<usize> {
    if let Some(len) = prop
        .keyframes
        .iter()
        .find_map(|kf| numeric_array_len(&kf.value))
    {
        return Some(len);
    }
    numeric_array_len(&prop.default_value)
}

pub fn extract_vec_components(v: &Value, dim: usize, fallback: &[f64]) -> Vec<f64> {
    let mut out = fallback.to_vec();
    out.resize(dim, 0.0);
    if let Some(arr) = v.as_array() {
        for (i, item) in arr.iter().take(dim).enumerate() {
            if let Some(n) = item.as_f64()
                && n.is_finite()
            {
                out[i] = n;
            }
        }
    }
    out
}

pub fn eval_vector_prop_at(prop: &SessionProperty, local_frame: f64, dim: usize) -> Vec<f64> {
    let zero = vec![0.0; dim];
    let default_vec = extract_vec_components(&prop.default_value, dim, &zero);
    if prop.keyframes.is_empty() {
        return default_vec;
    }
    if prop.keyframes.len() == 1 || local_frame <= prop.keyframes[0].frame {
        return extract_vec_components(&prop.keyframes[0].value, dim, &default_vec);
    }
    let last = &prop.keyframes[prop.keyframes.len() - 1];
    if local_frame >= last.frame {
        return extract_vec_components(&last.value, dim, &default_vec);
    }
    let idx = prop
        .keyframes
        .partition_point(|k| k.frame <= local_frame)
        .clamp(1, prop.keyframes.len() - 1);
    let k0 = &prop.keyframes[idx - 1];
    let k1 = &prop.keyframes[idx];
    let span = (k1.frame - k0.frame).max(1e-9);
    let u = ((local_frame - k0.frame) / span).clamp(0.0, 1.0);
    let v0 = extract_vec_components(&k0.value, dim, &default_vec);
    let v1 = extract_vec_components(&k1.value, dim, &default_vec);
    (0..dim).map(|i| v0[i] + (v1[i] - v0[i]) * u).collect()
}

pub fn normalize_pz_keyframe_object(
    obj: &serde_json::Map<String, Value>,
) -> Option<serde_json::Map<String, Value>> {
    let frame = obj
        .get("frame")
        .or_else(|| obj.get("t"))
        .or_else(|| obj.get("f"))
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())?;
    let value = obj
        .get("value")
        .or_else(|| obj.get("v"))
        .or_else(|| obj.get("val"))?
        .clone();
    let mut out = obj.clone();
    out.remove("t");
    out.remove("f");
    out.remove("v");
    out.remove("val");
    out.insert("frame".to_owned(), Value::from(frame));
    out.insert("value".to_owned(), value);
    if !out.contains_key("tween")
        && let Some(e) = out.remove("e")
    {
        out.insert("tween".to_owned(), e);
    }
    Some(out)
}

/// Normalizes PanZoid shorthand keyframe JSON (`t`/`v`/`e` -> `frame`/`value`/`tween`) and,
/// when pasting scalar keyframes onto a `Vec2`/`Vec3`/`Vec4`/`Color` property, lifts each scalar
/// into the active vector channel while preserving/interpolating all other vector components.
pub fn prepare_curve_paste_json(
    existing_prop: Option<&SessionProperty>,
    active_channel: Option<usize>,
    raw_json: &str,
    at_frame: Option<f64>,
) -> (String, Option<f64>) {
    let Ok(parsed) = serde_json::from_str::<Value>(raw_json) else {
        return (raw_json.to_owned(), at_frame);
    };
    let vec_dim = existing_prop.and_then(vector_property_dim);

    match parsed {
        Value::Array(items) => {
            let Some(mut normalized) = items
                .iter()
                .map(|item| item.as_object().and_then(normalize_pz_keyframe_object))
                .collect::<Option<Vec<_>>>()
            else {
                return (raw_json.to_owned(), at_frame);
            };
            if normalized.is_empty() {
                return (raw_json.to_owned(), at_frame);
            }

            let all_scalar = normalized
                .iter()
                .all(|k| k.get("value").is_some_and(Value::is_number));
            if let (Some(prop), Some(dim)) = (existing_prop, vec_dim)
                && all_scalar
            {
                let ch = active_channel.filter(|&c| c < dim).unwrap_or(0);
                let min_f = normalized
                    .iter()
                    .filter_map(|k| k.get("frame").and_then(Value::as_f64))
                    .fold(f64::INFINITY, f64::min);
                let delta = match at_frame {
                    Some(target_f) if min_f.is_finite() => target_f - min_f,
                    _ => 0.0,
                };
                for kf in &mut normalized {
                    let raw_f = kf.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
                    let dest_f = (raw_f + delta).max(0.0);
                    let scalar = kf.get("value").and_then(Value::as_f64).unwrap_or(0.0);
                    let mut vec_val = eval_vector_prop_at(prop, dest_f, dim);
                    vec_val[ch] = scalar;
                    kf.insert("frame".to_owned(), Value::from(dest_f));
                    kf.insert("value".to_owned(), serde_json::json!(vec_val));
                }
                let out_arr = Value::Array(normalized.into_iter().map(Value::Object).collect());
                return (out_arr.to_string(), None);
            }

            let out_arr = Value::Array(normalized.into_iter().map(Value::Object).collect());
            (out_arr.to_string(), at_frame)
        }
        Value::Object(mut obj) => {
            if let Some(kfs_arr) = obj.get("keyframes").and_then(Value::as_array)
                && let Some(mut normalized) = kfs_arr
                    .iter()
                    .map(|item| item.as_object().and_then(normalize_pz_keyframe_object))
                    .collect::<Option<Vec<_>>>()
            {
                let all_scalar = !normalized.is_empty()
                    && normalized
                        .iter()
                        .all(|k| k.get("value").is_some_and(Value::is_number));
                if let (Some(prop), Some(dim)) = (existing_prop, vec_dim)
                    && all_scalar
                {
                    let ch = active_channel.filter(|&c| c < dim).unwrap_or(0);
                    for kf in &mut normalized {
                        let f = kf
                            .get("frame")
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0)
                            .max(0.0);
                        let scalar = kf.get("value").and_then(Value::as_f64).unwrap_or(0.0);
                        let mut vec_val = eval_vector_prop_at(prop, f, dim);
                        vec_val[ch] = scalar;
                        kf.insert("frame".to_owned(), Value::from(f));
                        kf.insert("value".to_owned(), serde_json::json!(vec_val));
                    }
                    if let Some(scalar_default) = obj.get("value").and_then(Value::as_f64) {
                        let mut base_default = eval_vector_prop_at(prop, 0.0, dim);
                        base_default[ch] = scalar_default;
                        obj.insert("value".to_owned(), serde_json::json!(base_default));
                    }
                }
                obj.insert(
                    "keyframes".to_owned(),
                    Value::Array(normalized.into_iter().map(Value::Object).collect()),
                );
                return (Value::Object(obj).to_string(), at_frame);
            }
            (raw_json.to_owned(), at_frame)
        }
        _ => (raw_json.to_owned(), at_frame),
    }
}

pub fn copy_curve_track_as_pz_json(
    ctx: &egui::Context,
    session: &mut Session,
    state: &GraphEditorState,
    tracks: &[CurveTrack],
) -> Option<String> {
    let (target_id, property, _) = resolve_active_curve_target(state, tracks)?;
    let json = session
        .copy_property_as_pz_json(&target_id, &property)
        .ok()?;
    crate::ui::clipboard::write_system_clipboard(ctx, &json);
    Some(json)
}

pub fn try_paste_pz_keyframes_json(
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    text: &str,
    action: &mut GraphEditorAction,
) -> bool {
    let trimmed = text.trim();
    if !(trimmed.starts_with('{') || trimmed.starts_with('[')) {
        return false;
    }
    let Some((target_id, property, frame_offset, active_channel)) =
        resolve_active_curve_target_with_channel(state, tracks)
    else {
        return false;
    };
    let at_frame = if trimmed.starts_with('[') {
        Some(
            (session.playback.current_frame - frame_offset)
                .max(0.0)
                .round(),
        )
    } else {
        None
    };
    let existing_prop = find_session_property(session.document(), &target_id, &property).cloned();
    let (prepared_json, prepared_at_frame) =
        prepare_curve_paste_json(existing_prop.as_ref(), active_channel, trimmed, at_frame);
    if session
        .execute(Command::PastePzJson {
            json: prepared_json,
            hint: Some("keyframes".to_owned()),
            target_clip_id: None,
            parent_object_id: None,
            target_object_id: Some(target_id),
            target_property: Some(property),
            at_frame: prepared_at_frame,
        })
        .is_ok()
    {
        state.selected_keys.clear();
        state.handles.clear();
        action.curves_modified = true;
        return true;
    }
    false
}

pub fn handle_graph_paste_text(
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    text: Option<&str>,
    action: &mut GraphEditorAction,
) -> bool {
    if let Some(raw) = text {
        let trimmed = raw.trim();
        let looks_like_css = trimmed.to_ascii_lowercase().contains("cubic-bezier");
        if looks_like_css && let Some(parsed) = parse_cubic_bezier_str(trimmed) {
            apply_parsed_easing_curve(session, state, tracks, parsed, action);
            return true;
        }
        if !trimmed.is_empty()
            && !crate::ui::clipboard::matches_last_internal_copy(raw)
            && try_paste_pz_keyframes_json(session, state, tracks, trimmed, action)
        {
            return true;
        }
        if !state.copied_keys.is_empty() {
            paste_graph_keys_at_playhead(session, state, tracks, action);
            return true;
        }
        if !trimmed.is_empty()
            && try_paste_pz_keyframes_json(session, state, tracks, trimmed, action)
        {
            return true;
        }
        if let Some(parsed) = parse_cubic_bezier_str(trimmed) {
            apply_parsed_easing_curve(session, state, tracks, parsed, action);
            return true;
        }
    } else if !state.copied_keys.is_empty() {
        paste_graph_keys_at_playhead(session, state, tracks, action);
        return true;
    } else if let Some(ref clip) = state.easing_clipboard.clone()
        && let Some(parsed) = parse_cubic_bezier_str(clip)
    {
        apply_parsed_easing_curve(session, state, tracks, parsed, action);
        return true;
    }
    false
}

pub fn paint_dashed_polyline(
    painter: &egui::Painter,
    pts: &[Pos2],
    stroke: Stroke,
    dash: f32,
    gap: f32,
) {
    if pts.len() < 2 {
        return;
    }
    let period = (dash + gap).max(1.0);
    let mut phase = 0.0_f32;
    for seg in pts.windows(2) {
        let a = seg[0];
        let b = seg[1];
        let d = b - a;
        let len = d.length();
        if len <= 1e-3 {
            continue;
        }
        let dir = d / len;
        let mut cursor = 0.0_f32;
        while cursor < len {
            let in_dash = phase < dash;
            let remain_in_state = if in_dash {
                dash - phase
            } else {
                period - phase
            };
            let step = remain_in_state.min(len - cursor);
            if in_dash && step > 0.25 {
                painter.line_segment([a + dir * cursor, a + dir * (cursor + step)], stroke);
            }
            cursor += step;
            phase = (phase + step) % period;
        }
    }
}

pub fn paint_dashed_circle(painter: &egui::Painter, center: Pos2, radius: f32, stroke: Stroke) {
    const DASHES: usize = 8;
    let step = std::f32::consts::TAU / DASHES as f32;
    let arc = step * 0.65;
    for i in 0..DASHES {
        let a0 = i as f32 * step;
        let a1 = a0 + arc;
        let am = 0.5 * (a0 + a1);
        let p0 = pos2(center.x + a0.cos() * radius, center.y + a0.sin() * radius);
        let pm = pos2(center.x + am.cos() * radius, center.y + am.sin() * radius);
        let p1 = pos2(center.x + a1.cos() * radius, center.y + a1.sin() * radius);
        painter.add(Shape::line(vec![p0, pm, p1], stroke));
    }
}

/// Parses CSS `cubic-bezier(x1, y1, x2, y2)` (case-insensitive), `[x1, y1, x2, y2]`,
/// or bare 4-number strings into `[x1, y1, x2, y2]` with `x1, x2` clamped to `[0, 1]`.
#[must_use]
pub fn parse_cubic_bezier_str(raw: &str) -> Option<[f64; 4]> {
    let trimmed = raw.trim().trim_end_matches(';').trim();
    let lower = trimmed.to_ascii_lowercase();
    let body = if let Some(idx) = lower.find("cubic-bezier(") {
        let after = &trimmed[idx + "cubic-bezier(".len()..];
        let close = after.find(')')?;
        &after[..close]
    } else if let Some(inner) = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        inner
    } else {
        trimmed
    };
    let nums: Vec<f64> = body
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|tok| !tok.is_empty())
        .filter_map(|tok| tok.parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .collect();
    if nums.len() == 4 {
        Some([
            nums[0].clamp(0.0, 1.0),
            nums[1],
            nums[2].clamp(0.0, 1.0),
            nums[3],
        ])
    } else {
        None
    }
}
