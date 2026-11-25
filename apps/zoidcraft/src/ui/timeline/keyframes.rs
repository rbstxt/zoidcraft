//! Dope-sheet keyframe lanes, rows, and splitter.

use super::*;

#[derive(Debug, Clone)]
pub(crate) struct KeyframeLaneItem {
    pub(crate) local_frame: f64,
    pub(crate) value: Value,
    pub(crate) tween: u16,
    pub(crate) control_points: [[f64; 2]; 2],
    pub(crate) continuous_tangent: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct KeyframeLaneRow {
    pub(crate) target_id: String,
    pub(crate) property: String,
    pub(crate) display_label: String,
    pub(crate) frame_offset: f64,
    pub(crate) locked: bool,
    pub(crate) keyframes: Vec<KeyframeLaneItem>,
}

pub(crate) fn keyframe_selection_hash(session: &Session) -> u64 {
    fn mix_str(hash: &mut u64, s: &str) {
        for &b in s.as_bytes() {
            *hash ^= u64::from(b);
            *hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        *hash ^= 0xff;
        *hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    if let Some(first_clip) = session.selection.clip_ids.first() {
        mix_str(&mut hash, first_clip);
    }
    hash ^= 0x11;
    hash = hash.wrapping_mul(0x0100_0000_01b3);
    for id in &session.selection.object_3d_ids {
        mix_str(&mut hash, id);
    }
    hash ^= 0x22;
    hash = hash.wrapping_mul(0x0100_0000_01b3);
    for id in &session.selection.effect_ids {
        mix_str(&mut hash, id);
    }
    hash
}

pub(crate) fn collect_keyframe_rows(session: &Session) -> Vec<KeyframeLaneRow> {
    let doc = session.document();
    let mut rows = Vec::new();
    let has_obj_sel = !session.selection.object_3d_ids.is_empty();
    let has_fx_sel = !session.selection.effect_ids.is_empty();

    if let Some(clip_id) = session.selection.clip_ids.first() {
        for track in doc.video_tracks.iter().chain(doc.audio_tracks.iter()) {
            if let Some(clip) = track.clips.iter().find(|c| &c.id == clip_id) {
                if !has_obj_sel && !has_fx_sel {
                    push_property_rows(
                        &mut rows,
                        &clip.id,
                        "",
                        clip.start,
                        track.locked,
                        &clip.properties,
                    );
                }
                if has_obj_sel || !has_fx_sel {
                    for obj in &clip.objects_3d {
                        if !has_obj_sel || session.selection.object_3d_ids.contains(&obj.id) {
                            let prefix = format!("{}.", obj.name);
                            push_property_rows(
                                &mut rows,
                                &obj.id,
                                &prefix,
                                clip.start,
                                track.locked,
                                &obj.properties,
                            );
                        }
                    }
                }
                if has_fx_sel || !has_obj_sel {
                    for fx in &clip.effects {
                        if !has_fx_sel || session.selection.effect_ids.contains(&fx.id) {
                            let prefix = format!("{}.", fx.name);
                            push_property_rows(
                                &mut rows,
                                &fx.id,
                                &prefix,
                                clip.start,
                                track.locked,
                                &fx.properties,
                            );
                        }
                    }
                }
                if !has_obj_sel && !has_fx_sel && track.kind == TrackKind::Video {
                    collect_mask_and_shape_plus_keyframe_rows(&mut rows, session, clip);
                }
                break;
            }
        }
    }

    if !has_obj_sel && !has_fx_sel {
        push_property_rows(
            &mut rows,
            "sequence",
            "seq.",
            0.0,
            false,
            &doc.sequence_properties,
        );
    }
    rows
}

pub(crate) fn collect_mask_and_shape_plus_keyframe_rows(
    out: &mut Vec<KeyframeLaneRow>,
    session: &Session,
    clip: &SessionClip,
) {
    let Ok(clip_uuid) = zc_project::Uuid::parse_str(&clip.id) else {
        return;
    };
    let seq = session.active_sequence();
    let Some(vc) = seq
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == clip_uuid)
    else {
        return;
    };

    // 1. Animated LayerMask properties (path keyframes + preserved animated mask props).
    if let Some(masks) = seq.layer_masks.get(&vc.layer.id) {
        for mask in masks {
            let mask_name = if mask.name.trim().is_empty() {
                mask.id.as_str()
            } else {
                mask.name.as_str()
            };
            if (mask.path.animated || mask.path.keyframes.len() > 1)
                && !mask.path.keyframes.is_empty()
            {
                let keyframes = mask
                    .path
                    .keyframes
                    .iter()
                    .map(|k| KeyframeLaneItem {
                        local_frame: k.frame,
                        value: serde_json::to_value(&k.value).unwrap_or(Value::Null),
                        tween: k.tween as u16,
                        control_points: k.control_points,
                        continuous_tangent: k.continuous_tangent,
                    })
                    .collect();
                out.push(KeyframeLaneRow {
                    target_id: format!("mask:{}:{}", clip.id, mask.id),
                    property: "path".to_string(),
                    display_label: format!("Mask:{mask_name}.path"),
                    frame_offset: clip.start,
                    locked: true,
                    keyframes,
                });
            }
            for (prop_key, prop_val) in &mask.preserved {
                if let Some(kfs) = extract_json_keyframes(prop_val) {
                    out.push(KeyframeLaneRow {
                        target_id: format!("mask:{}:{}", clip.id, mask.id),
                        property: prop_key.clone(),
                        display_label: format!("Mask:{mask_name}.{prop_key}"),
                        frame_offset: clip.start,
                        locked: true,
                        keyframes: kfs,
                    });
                }
            }
        }
    }

    // 2. Animated Shape+ payload tracks (and preserved Shape+ / layer animated tracks).
    if let zc_project::LayerKind::Plugin {
        plugin_type,
        payload,
    } = &vc.layer.kind
        && plugin_type.contains("shape-plus")
    {
        collect_shape_plus_payload_rows(out, &clip.id, clip.start, payload, 0);
    }
    for (key, val) in &vc.layer.preserved {
        if key == "shapePlus" {
            collect_shape_plus_payload_rows(out, &clip.id, clip.start, val, 0);
        } else if let Some(kfs) = extract_json_keyframes(val) {
            out.push(KeyframeLaneRow {
                target_id: format!("preserved:{}", clip.id),
                property: key.clone(),
                display_label: format!("Layer.{key}"),
                frame_offset: clip.start,
                locked: true,
                keyframes: kfs,
            });
        }
    }
}

pub(crate) fn collect_shape_plus_payload_rows(
    out: &mut Vec<KeyframeLaneRow>,
    clip_id: &str,
    frame_offset: f64,
    payload: &Value,
    depth: usize,
) {
    if depth > 16 {
        return;
    }
    let Some(contents) = payload.get("contents").and_then(Value::as_array) else {
        return;
    };
    for item in contents {
        let item_id = item.get("id").and_then(Value::as_str).unwrap_or("item");
        let item_kind = item.get("kind").and_then(Value::as_str).unwrap_or("shape");
        let item_name = item
            .get("properties")
            .and_then(|p| p.get("name"))
            .and_then(|v| {
                v.as_str()
                    .or_else(|| v.get("value").and_then(Value::as_str))
            })
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(item_kind);

        if let Some(props) = item.get("properties").and_then(Value::as_object) {
            for (prop_name, prop_val) in props {
                if let Some(kfs) = extract_json_keyframes(prop_val) {
                    out.push(KeyframeLaneRow {
                        target_id: format!("shape_plus:{clip_id}:{item_id}"),
                        property: prop_name.clone(),
                        display_label: format!("Shape+:{item_name}.{prop_name}"),
                        frame_offset,
                        locked: true,
                        keyframes: kfs,
                    });
                }
            }
        }
        collect_shape_plus_payload_rows(out, clip_id, frame_offset, item, depth + 1);
    }
}

pub(crate) fn extract_json_keyframes(val: &Value) -> Option<Vec<KeyframeLaneItem>> {
    let obj = val.as_object()?;
    let kfs_arr = obj.get("keyframes").and_then(Value::as_array)?;
    if kfs_arr.is_empty() {
        return None;
    }
    let is_animated = obj
        .get("animated")
        .and_then(Value::as_bool)
        .unwrap_or(kfs_arr.len() > 1);
    if !is_animated && kfs_arr.len() <= 1 {
        return None;
    }
    let mut items: Vec<KeyframeLaneItem> = kfs_arr
        .iter()
        .filter_map(|k| {
            let f = k.get("frame").and_then(Value::as_f64)?;
            let v = k.get("value").cloned().unwrap_or(Value::Null);
            let tween = k.get("tween").and_then(Value::as_u64).unwrap_or(1) as u16;
            Some(KeyframeLaneItem {
                local_frame: f,
                value: v,
                tween,
                control_points: [[-10.0, 0.0], [10.0, 0.0]],
                continuous_tangent: true,
            })
        })
        .collect();
    if items.is_empty() {
        return None;
    }
    items.sort_by(|a, b| a.local_frame.total_cmp(&b.local_frame));
    Some(items)
}

pub(crate) fn push_property_rows(
    out: &mut Vec<KeyframeLaneRow>,
    target_id: &str,
    prefix: &str,
    frame_offset: f64,
    locked: bool,
    props: &BTreeMap<String, SessionProperty>,
) {
    for (name, prop) in props {
        if !prop.animated || prop.keyframes.is_empty() {
            continue;
        }
        let keyframes = prop
            .keyframes
            .iter()
            .map(|k| KeyframeLaneItem {
                local_frame: k.frame,
                value: k.value.clone(),
                tween: k.tween,
                control_points: k.control_points,
                continuous_tangent: k.continuous_tangent,
            })
            .collect();
        out.push(KeyframeLaneRow {
            target_id: target_id.to_string(),
            property: name.clone(),
            display_label: format!("{prefix}{name}"),
            frame_offset,
            locked,
            keyframes,
        });
    }
}

// ---------------------------------------------------------------------------
// Snapping & Overlap Helpers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn paint_and_interact_keyframes(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut TimelineViewState,
    rows: &[KeyframeLaneRow],
    header_rect: Rect,
    lanes_rect: Rect,
    fps: f64,
    action: &mut TimelineAction,
) {
    let total_kf_h = rows.len() as f32 * KEYFRAME_ROW_HEIGHT;
    let max_scroll = (total_kf_h - lanes_rect.height()).max(0.0);
    state.keyframe_scroll_y = clamp_scroll_y(state.keyframe_scroll_y, max_scroll);
    let (vscroll_consumed, vscroll_hovered) = interact_vertical_scrollbar(
        ui,
        state,
        VScrollPane::Keyframes,
        lanes_rect,
        max_scroll,
        total_kf_h,
        false,
    );
    let interact_lanes_rect = if max_scroll > 0.5 {
        Rect::from_min_max(
            lanes_rect.min,
            pos2(
                (lanes_rect.right() - VSCROLLBAR_HIT_WIDTH).max(lanes_rect.left()),
                lanes_rect.bottom(),
            ),
        )
    } else {
        lanes_rect
    };

    let header_painter = ui.painter().with_clip_rect(header_rect);
    header_painter.rect_filled(header_rect, CornerRadius::ZERO, COLOR_KF_EMPTY_BG);
    let lane_painter = ui.painter().with_clip_rect(lanes_rect);
    lane_painter.rect_filled(lanes_rect, CornerRadius::ZERO, COLOR_KF_EMPTY_BG);
    // Allow frame-0 keyframe diamonds (`r = 5.5 px`) to render intact without left-edge clipping
    // while keeping 10 px right padding on lane labels so labels never collide with frame-0 diamonds.
    let diamond_clip = Rect::from_min_max(
        pos2(
            (lanes_rect.left() - 7.0).max(header_rect.right() - 8.0),
            lanes_rect.top(),
        ),
        lanes_rect.max,
    );
    let diamond_painter = ui.painter().with_clip_rect(diamond_clip);
    let label_clip_rect = Rect::from_min_max(
        header_rect.min,
        pos2(
            (header_rect.right() - 10.0).max(header_rect.left()),
            header_rect.bottom(),
        ),
    );
    let label_painter = ui.painter().with_clip_rect(label_clip_rect);

    let lane_resp = ui.interact(
        interact_lanes_rect,
        ui.scope_id().with("timeline_kf_lanes"),
        Sense::click_and_drag(),
    );

    let modifiers = ui.input(|i| i.modifiers);
    let mut clicked_any_diamond = vscroll_consumed;

    // First pass: paint row backgrounds, labels (clipped with right padding), and header border
    // BEFORE diamonds so frame-0 diamonds sit cleanly intact on top of the 1 px seam.
    for (idx, row) in rows.iter().enumerate() {
        let y0 = lanes_rect.top() + idx as f32 * KEYFRAME_ROW_HEIGHT - state.keyframe_scroll_y;
        let y1 = y0 + KEYFRAME_ROW_HEIGHT;
        if y1 < lanes_rect.top() || y0 > lanes_rect.bottom() {
            continue;
        }

        let label_bg = if idx % 2 == 0 {
            COLOR_KF_LABEL_EVEN
        } else {
            COLOR_KF_LABEL_ODD
        };
        let lane_bg = if idx % 2 == 0 {
            COLOR_KF_LANE_EVEN
        } else {
            COLOR_KF_LANE_ODD
        };

        let row_header_rect =
            Rect::from_min_max(pos2(header_rect.left(), y0), pos2(header_rect.right(), y1));
        let row_lane_rect =
            Rect::from_min_max(pos2(lanes_rect.left(), y0), pos2(lanes_rect.right(), y1));

        header_painter.rect_filled(row_header_rect, CornerRadius::ZERO, label_bg);
        label_painter.text(
            pos2(row_header_rect.left() + 6.0, row_header_rect.center().y),
            Align2::LEFT_CENTER,
            &row.display_label,
            FontId::monospace(12.0),
            COLOR_KF_TEXT,
        );

        // Right-click on keyframe property label opens it in Graph Editor.
        let label_resp = ui.interact(
            row_header_rect.intersect(header_rect),
            ui.scope_id()
                .with(("kf_label", idx, &row.target_id, &row.property)),
            Sense::click(),
        );
        if label_resp.secondary_clicked() {
            action.open_graph_editor_for = Some((row.target_id.clone(), row.property.clone()));
        }

        lane_painter.rect_filled(row_lane_rect, CornerRadius::ZERO, lane_bg);
    }

    if !rows.is_empty() {
        let rows_h = (total_kf_h - state.keyframe_scroll_y).clamp(0.0, header_rect.height());
        if rows_h > 0.0 {
            header_painter.rect_filled(
                Rect::from_min_size(
                    pos2(header_rect.right() - 1.0, header_rect.top()),
                    vec2(1.0, rows_h),
                ),
                CornerRadius::ZERO,
                COLOR_TRACK_BORDER,
            );
        }
    }

    // Second pass: paint and interact keyframe diamonds intact (including frame 0!).
    for (idx, row) in rows.iter().enumerate() {
        let y0 = lanes_rect.top() + idx as f32 * KEYFRAME_ROW_HEIGHT - state.keyframe_scroll_y;
        let y1 = y0 + KEYFRAME_ROW_HEIGHT;
        if y1 < lanes_rect.top() || y0 > lanes_rect.bottom() {
            continue;
        }

        let hue_deg = (string_hash(&row.property).wrapping_mul(40) % 360) as f32;
        let fill_unselected = hsl_to_color32(hue_deg, 0.40, 0.45);
        let stroke_color = hsl_to_color32(hue_deg, 0.40, 0.60);

        let drag_delta = match &state.drag_state {
            TimelineDragState::DraggingKeyframe(drag) => drag.preview_delta_frames,
            _ => 0.0,
        };

        for (kf_idx, item) in row.keyframes.iter().enumerate() {
            let local_frame = item.local_frame;
            let is_selected = session.selection.keyframes.iter().any(|sel| {
                sel.target_id == row.target_id
                    && sel.property == row.property
                    && sel.matches_frame(local_frame)
            });

            let display_abs_frame = (row.frame_offset
                + local_frame
                + if is_selected && !row.locked {
                    drag_delta
                } else {
                    0.0
                })
            .max(0.0);
            let cx = lanes_rect.left()
                + (frame_to_px(display_abs_frame, fps, state.zoom) - state.scroll_x) as f32;
            let cy = (y0 + y1) * 0.5;
            if cx < lanes_rect.left() - 10.0 || cx > lanes_rect.right() + 10.0 {
                continue;
            }

            let r = 5.5_f32;
            let diamond_hit = Rect::from_center_size(pos2(cx, cy), vec2(13.0, 13.0));
            let kf_resp = ui.interact(
                diamond_hit.intersect(diamond_clip),
                ui.scope_id()
                    .with(("kf_diamond", &row.target_id, &row.property, kf_idx)),
                Sense::click_and_drag(),
            );

            if kf_resp.hovered() {
                state.hover = TimelineHover::Keyframe {
                    target_id: row.target_id.clone(),
                    property: row.property.clone(),
                };
                ui.ctx().set_cursor_icon(CursorIcon::Move);
            }

            // Never toggle selection on BOTH drag_started_by and clicked_by:
            // clicked_by handles single/Shift-toggle selection; drag_started_by only prepares drag.
            if kf_resp.clicked_by(PointerButton::Primary) {
                clicked_any_diamond = true;
                state.focused = true;
                if row.locked
                    && (row.target_id.starts_with("mask:")
                        || row.target_id.starts_with("shape_plus:")
                        || row.target_id.starts_with("preserved:"))
                {
                    let target_f = (row.frame_offset + local_frame).round().max(0.0);
                    let _ = session.execute(Command::SetFrame { frame: target_f });
                    action.playhead_scrubbed = true;
                } else {
                    let kf_ref = KeyframeRef::new(&row.target_id, &row.property, None, local_frame);
                    session.selection.select_keyframe(kf_ref, modifiers.shift);
                }
            } else if kf_resp.drag_started_by(PointerButton::Primary) {
                clicked_any_diamond = true;
                state.focused = true;
                if !modifiers.shift && !is_selected {
                    let kf_ref = KeyframeRef::new(&row.target_id, &row.property, None, local_frame);
                    session.selection.select_keyframe(kf_ref, false);
                }
                if !row.locked && (!modifiers.shift || is_selected) {
                    let dragged =
                        collect_selected_keyframe_infos(rows, &session.selection.keyframes);
                    if !dragged.is_empty() {
                        let pointer_x = ui.input(|i| i.pointer.interact_pos()).map_or(cx, |p| p.x);
                        let content_x = f64::from(pointer_x - lanes_rect.left()) + state.scroll_x;
                        let anchor_abs = row.frame_offset + local_frame;
                        let anchor_px = frame_to_px(anchor_abs, fps, state.zoom);
                        state.drag_state = TimelineDragState::DraggingKeyframe(KeyframeDragState {
                            anchor_orig_abs_frame: anchor_abs,
                            grab_offset_px: anchor_px - content_x,
                            preview_delta_frames: 0.0,
                            keyframes: dragged,
                            moved: false,
                        });
                    }
                }
            }

            let pts = vec![
                pos2(cx, cy - r),
                pos2(cx + r, cy),
                pos2(cx, cy + r),
                pos2(cx - r, cy),
            ];
            let fill = if is_selected {
                COLOR_KF_SELECTED
            } else {
                fill_unselected
            };
            diamond_painter.add(Shape::convex_polygon(
                pts,
                fill,
                Stroke::new(1.0, stroke_color),
            ));
        }
    }

    // Empty keyframe lane interactions: right-click scrubs playhead, left-drag starts marquee.
    if !clicked_any_diamond {
        if lane_resp.secondary_clicked()
            && let Some(pos) = lane_resp.interact_pointer_pos()
        {
            let content_x = (f64::from(pos.x - lanes_rect.left()) + state.scroll_x).max(0.0);
            let frame = px_to_frame(content_x, fps, state.zoom).round().max(0.0);
            let _ = session.execute(Command::SetFrame { frame });
            action.playhead_scrubbed = true;
        } else if lane_resp.clicked_by(PointerButton::Primary) && !modifiers.shift {
            session.selection.clear_keyframes();
        } else if lane_resp.drag_started_by(PointerButton::Primary)
            && matches!(state.drag_state, TimelineDragState::None)
            && let Some(origin) = lane_resp.interact_pointer_pos()
        {
            state.drag_state = TimelineDragState::MarqueeSelect {
                origin,
                current: origin,
                additive: modifiers.shift,
                in_keyframes: true,
            };
        }
    }

    // Draggable vertical scrollbar pinned strictly to `lanes_rect.right()` regardless of `scroll_x`.
    if max_scroll > 0.5 && lanes_rect.height() >= 16.0 {
        paint_vertical_scroll_indicator(
            &lane_painter,
            lanes_rect,
            state.keyframe_scroll_y,
            max_scroll,
            total_kf_h,
            false,
            vscroll_hovered,
            state
                .vscrollbar_drag
                .is_some_and(|(p, _)| p == VScrollPane::Keyframes),
        );
    }
}

pub(crate) fn paint_and_interact_kf_splitter(
    ui: &mut Ui,
    state: &mut TimelineViewState,
    rect: Rect,
    current_kf_h: f32,
    max_kf_h: f32,
) {
    let hit_rect = rect.expand2(vec2(0.0, 4.0));
    let resp = ui.interact(
        hit_rect,
        ui.scope_id().with("timeline_kf_splitter"),
        Sense::click_and_drag(),
    );

    if resp.hovered() || state.kf_splitter_drag.is_some() {
        ui.ctx().set_cursor_icon(CursorIcon::ResizeVertical);
    }

    let start_drag = resp.drag_started_by(PointerButton::Primary)
        || (resp.hovered()
            && state.kf_splitter_drag.is_none()
            && ui.input(|i| i.pointer.primary_pressed()));
    if start_drag
        && let Some(pos) = resp
            .interact_pointer_pos()
            .or_else(|| ui.input(|i| i.pointer.latest_pos()))
    {
        state.kf_splitter_drag = Some((pos.y, current_kf_h));
        state.keyframe_area_custom = true;
    }

    if let Some((start_y, start_h)) = state.kf_splitter_drag {
        if let Some(pos) = ui.input(|i| i.pointer.latest_pos()) {
            let raw_h = start_h + (pos.y - start_y);
            state.keyframe_area_height = raw_h.clamp(KEYFRAME_ROW_HEIGHT.min(max_kf_h), max_kf_h);
            state.keyframe_area_custom = true;
        }
        if !ui.input(|i| i.pointer.primary_down()) {
            state.kf_splitter_drag = None;
        }
    }

    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, COLOR_KF_SPLITTER_BG);
}

pub(crate) fn collect_selected_keyframe_infos(
    rows: &[KeyframeLaneRow],
    selected: &[KeyframeRef],
) -> Vec<DraggedKeyframeInfo> {
    let mut infos = Vec::new();
    for row in rows {
        if row.locked {
            continue;
        }
        for item in &row.keyframes {
            if selected.iter().any(|s| {
                s.target_id == row.target_id
                    && s.property == row.property
                    && s.matches_frame(item.local_frame)
            }) {
                infos.push(DraggedKeyframeInfo {
                    target_id: row.target_id.clone(),
                    property: row.property.clone(),
                    frame_offset: row.frame_offset,
                    orig_local_frame: item.local_frame,
                    value: item.value.clone(),
                    tween: item.tween,
                    control_points: item.control_points,
                    continuous_tangent: item.continuous_tangent,
                });
            }
        }
    }
    infos
}
