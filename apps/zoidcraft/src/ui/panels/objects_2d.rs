use super::edit_tab::show_preset_shape_kind_row;
use super::effects_inspector::oversized_text_summary;
use super::property_rows::{
    show_session_color_prop, show_session_number_prop, show_session_option_prop_ex,
    show_session_text_prop_ex,
};
use super::shared_widgets::{
    empty_panel_message, find_project_video_clip, humanize_prop_label, paint_crisp_panel_border,
    panel_top_header, show_font_selector_row,
};
use super::state::{
    WidgetTxTracker, apply_widget_tx_command, header_compact_action_button,
    paste_pz_json_from_clipboard,
};
use super::*;

// ---------------------------------------------------------------------------
// 4. Non-3D Objects Tab (Shape+ 21-Kind Tree + SVG, PresetShape2D, Text2D + Text+ 2D)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) fn show_non_scene_objects_tab(
    ui: &mut Ui,
    panel_rect: Rect,
    viewport_rect: Rect,
    clip: &SessionClip,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) -> bool {
    let is_shape_plus = clip.kind == "zoidium:shape-plus/shape-layer"
        || find_project_video_clip(session, &clip.id).is_some_and(|vc| {
            matches!(
                &vc.layer.kind,
                LayerKind::Plugin { plugin_type, .. }
                    if plugin_type == "zoidium:shape-plus/shape-layer"
            )
        });
    if is_shape_plus {
        show_shape_plus_objects_tab(
            ui,
            panel_rect,
            viewport_rect,
            clip,
            session,
            theme,
            state,
            actions,
        );
        return true;
    }

    if clip.kind == "text" || clip.kind == "text_2d" {
        show_text2d_objects_tab(
            ui,
            panel_rect,
            viewport_rect,
            clip,
            session,
            theme,
            state,
            actions,
        );
        return true;
    }

    if clip.kind == "preset_shape" || clip.kind == "shape" {
        ScrollArea::vertical()
            .id_salt("preset_shape_objects_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                panel_top_header(ui, &clip.name, theme);
                let local_frame = session.playback.current_frame - clip.start;
                if clip.kind == "preset_shape" {
                    show_preset_shape_kind_row(ui, session, &clip.id, viewport_rect, state, theme);
                }
                show_session_color_prop(
                    ui,
                    session,
                    &clip.id,
                    "fillColor",
                    "Fill color",
                    [1.0, 1.0, 1.0],
                    local_frame,
                    clip.start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_color_prop(
                    ui,
                    session,
                    &clip.id,
                    "strokeColor",
                    "Stroke color",
                    [0.0, 0.0, 0.0],
                    local_frame,
                    clip.start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    &clip.id,
                    "strokeWidth",
                    "Stroke width",
                    0.0,
                    ScrubConfig::new(0.0, 200.0, 1.0, 1),
                    local_frame,
                    clip.start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            });
        return true;
    }

    false
}

#[allow(clippy::too_many_arguments)]
fn show_shape_plus_objects_tab(
    ui: &mut Ui,
    panel_rect: Rect,
    viewport_rect: Rect,
    clip: &SessionClip,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let Some(cid) = Uuid::parse_str(&clip.id).ok() else {
        empty_panel_message(ui, "invalid Shape+ clip id", theme);
        return;
    };
    let mut payload = find_project_video_clip(session, &clip.id)
        .and_then(|vc| match &vc.layer.kind {
            LayerKind::Plugin { payload, .. } => Some(payload.clone()),
            _ => None,
        })
        .unwrap_or_else(default_shape_plus_payload);

    let (top_rect, bot_rect) = begin_split_panel(
        ui,
        Id::unique("shape_plus_split"),
        panel_rect,
        &mut state.objects_split,
    );
    ui.painter()
        .rect_filled(top_rect, CornerRadius::ZERO, BG_PANEL);
    ui.painter()
        .rect_filled(bot_rect, CornerRadius::ZERO, BG_PANEL);

    let mut payload_dirty = false;

    // Top Pane: Clean "Shape+" header line + compact 2nd action row (Import SVG / Paste SVG / Paste JSON)
    show_split_pane(
        ui,
        Id::unique("shape_plus_split_p0"),
        top_rect.shrink(2.0),
        |ui, _| {
            ui.label(
                egui::RichText::new("Shape+")
                    .font(theme.font(FONT_TITLE))
                    .color(TEXT_STRONG),
            );
            ui.add_space(2.0);
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(4.0, 3.0);
                if ui.small_button("Import SVG...").clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("SVG Vector", &["svg"])
                        .pick_file()
                {
                    match std::fs::read_to_string(&path)
                        .map_err(|e| e.to_string())
                        .and_then(|txt| import_svg_to_shape_plus(&txt))
                    {
                        Ok(imported) => {
                            payload = imported;
                            payload_dirty = true;
                            state.shape_plus_svg_error = None;
                        }
                        Err(err) => {
                            state.shape_plus_svg_error = Some(err);
                        }
                    }
                }
                if ui.small_button("Paste SVG").clicked() {
                    state.shape_plus_svg_open = !state.shape_plus_svg_open;
                }
                if ui
                    .small_button("Paste JSON")
                    .on_hover_text("Paste Shape+ JSON from clipboard (PZ compatible)")
                    .clicked()
                    && paste_pz_json_from_clipboard(
                        session,
                        state,
                        actions,
                        "shape_plus",
                        Some(clip.id.clone()),
                        None,
                        None,
                        None,
                        "Shape+ JSON",
                    )
                    && let Some(updated) =
                        find_project_video_clip(session, &clip.id).and_then(|vc| {
                            match &vc.layer.kind {
                                LayerKind::Plugin { payload, .. } => Some(payload.clone()),
                                _ => None,
                            }
                        })
                {
                    payload = updated;
                }
            });

            if state.shape_plus_svg_open {
                ui.add_space(4.0);
                ui.add(
                    TextEdit::multiline(&mut state.shape_plus_svg_paste)
                        .hint_text("<svg>...</svg>")
                        .desired_rows(3)
                        .desired_width(ui.available_width())
                        .font(egui::FontId::monospace(11.0)),
                );
                ui.horizontal(|ui| {
                    if cm3_prop_button(
                        ui,
                        "Import Pasted SVG",
                        ButtonVariant::Primary,
                        false,
                        theme,
                    )
                    .clicked()
                    {
                        match import_svg_to_shape_plus(&state.shape_plus_svg_paste) {
                            Ok(imported) => {
                                payload = imported;
                                payload_dirty = true;
                                state.shape_plus_svg_open = false;
                                state.shape_plus_svg_error = None;
                            }
                            Err(err) => state.shape_plus_svg_error = Some(err),
                        }
                    }
                    if let Some(err) = &state.shape_plus_svg_error {
                        ui.label(
                            egui::RichText::new(err)
                                .font(theme.font(FONT_COMPACT))
                                .color(DANGER_TEXT),
                        );
                    }
                });
            }

            // + Add Item (all 21 Shape+ kinds) with clean integer-row scrollable popup
            ui.add_space(4.0);
            let kind_labels: Vec<&str> = ALL_SHAPE_PLUS_KINDS
                .iter()
                .map(|k| shape_plus_kind_title(k).unwrap_or(*k))
                .collect();
            let dd_popup_id = Id::unique("shape_plus_add_kind_popup");
            let mut dd_open = ui
                .data_mut(|d| d.get_temp::<bool>(dd_popup_id))
                .unwrap_or(false);
            let mut dd_btn_rect = Rect::NOTHING;
            let mut dd_just_opened = false;

            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                let cur_idx = state
                    .shape_plus_add_kind
                    .min(kind_labels.len().saturating_sub(1));
                let cur_label = kind_labels.get(cur_idx).copied().unwrap_or("Rectangle");
                let dd_w = (ui.available_width() - 62.0).clamp(110.0, 200.0);
                let (raw_slot, _) = ui.allocate_exact_size(Vec2::new(dd_w, 23.0), Sense::hover());
                let btn_rect = Rect::from_min_size(
                    Pos2::new(raw_slot.left().round(), (raw_slot.top() + 1.0).round()),
                    Vec2::new(dd_w.round(), 21.0),
                );
                dd_btn_rect = btn_rect;
                let btn_resp = ui
                    .interact(btn_rect, dd_popup_id.with("btn"), Sense::click())
                    .on_hover_cursor(CursorIcon::PointingHand);
                ui.painter()
                    .rect_filled(btn_rect, CornerRadius::ZERO, BG_INPUT);
                paint_crisp_panel_border(
                    ui.painter(),
                    btn_rect,
                    if dd_open || btn_resp.hovered() {
                        theme.accent()
                    } else {
                        BORDER_CONTROL
                    },
                );
                ui.painter()
                    .with_clip_rect(Rect::from_min_max(
                        btn_rect.left_top(),
                        Pos2::new(
                            (btn_rect.right() - 18.0).max(btn_rect.left()),
                            btn_rect.bottom(),
                        ),
                    ))
                    .text(
                        Pos2::new(btn_rect.left() + 5.0, btn_rect.center().y),
                        Align2::LEFT_CENTER,
                        cur_label,
                        theme.font(FONT_SMALL),
                        theme.value(),
                    );
                paint_cm3_icon(
                    ui.painter(),
                    Rect::from_center_size(
                        Pos2::new(btn_rect.right() - 8.5, btn_rect.center().y),
                        Vec2::splat(12.0),
                    ),
                    Cm3Icon::ChevronDown,
                    theme.value(),
                );
                if btn_resp.clicked() {
                    dd_open = !dd_open;
                    dd_just_opened = dd_open;
                }
                if cm3_prop_button(ui, "+ Add", ButtonVariant::Primary, false, theme).clicked() {
                    let kind = ALL_SHAPE_PLUS_KINDS[state
                        .shape_plus_add_kind
                        .min(ALL_SHAPE_PLUS_KINDS.len() - 1)];
                    if let Ok(new_id) = add_shape_plus_item_smart(
                        &mut payload,
                        state.shape_plus_selected.as_deref(),
                        kind,
                    ) {
                        state.shape_plus_selected = Some(new_id);
                        payload_dirty = true;
                    }
                }
            });

            if dd_open {
                let row_h = 24.0_f32;
                let visible_rows = kind_labels.len().min(12);
                let popup_h = visible_rows as f32 * row_h + 8.0;
                let popup_w = dd_btn_rect.width().max(180.0);
                let px = dd_btn_rect.left().clamp(
                    viewport_rect.left() + 8.0,
                    (viewport_rect.right() - popup_w - 8.0).max(viewport_rect.left() + 8.0),
                );
                let below_y = dd_btn_rect.bottom() + 2.0;
                let py = if below_y + popup_h <= viewport_rect.bottom() - 8.0 {
                    below_y
                } else {
                    (dd_btn_rect.top() - popup_h - 2.0).clamp(
                        viewport_rect.top() + 8.0,
                        (viewport_rect.bottom() - popup_h - 8.0).max(viewport_rect.top() + 8.0),
                    )
                };
                let mut close_now = ui.input(|i| i.key_pressed(Key::Escape));
                let area_resp = Area::new(dd_popup_id)
                    .order(Order::Foreground)
                    .fixed_pos(Pos2::new(px.round(), py.round()))
                    .show(ui.ctx(), |ui| {
                        Frame::NONE
                            .fill(BG_PANEL)
                            .stroke(Stroke::new(1.0, theme.accent()))
                            .inner_margin(egui::Margin::same(4))
                            .show(ui, |ui| {
                                ui.set_width(popup_w - 8.0);
                                ScrollArea::vertical()
                                    .id_salt((dd_popup_id, "scroll"))
                                    .max_height(visible_rows as f32 * row_h)
                                    .scroll_bar_visibility(ScrollBarVisibility::AlwaysVisible)
                                    .auto_shrink([false, false])
                                    .show(ui, |ui| {
                                        ui.spacing_mut().item_spacing.y = 0.0;
                                        for (idx, lbl) in kind_labels.iter().enumerate() {
                                            let is_sel = idx == state.shape_plus_add_kind;
                                            let (r, resp) = ui.allocate_exact_size(
                                                Vec2::new(ui.available_width(), row_h),
                                                Sense::click(),
                                            );
                                            let resp =
                                                resp.on_hover_cursor(CursorIcon::PointingHand);
                                            if is_sel {
                                                ui.painter().rect_filled(
                                                    r,
                                                    CornerRadius::ZERO,
                                                    theme.accent(),
                                                );
                                            } else if resp.hovered() {
                                                ui.painter().rect_filled(
                                                    r,
                                                    CornerRadius::ZERO,
                                                    BG_DROPDOWN_HOVER,
                                                );
                                            }
                                            ui.painter().text(
                                                Pos2::new(r.left() + 6.0, r.center().y),
                                                Align2::LEFT_CENTER,
                                                *lbl,
                                                theme.font(FONT_SMALL),
                                                if is_sel { TEXT_BRIGHT } else { TEXT_PRIMARY },
                                            );
                                            if resp.clicked() {
                                                state.shape_plus_add_kind = idx;
                                                close_now = true;
                                            }
                                        }
                                    });
                            });
                    });
                let clicked_outside = !dd_just_opened
                    && ui.input(|i| {
                        i.pointer.primary_pressed()
                            && i.pointer.interact_pos().is_some_and(|p| {
                                !area_resp.response.rect.contains(p) && !dd_btn_rect.contains(p)
                            })
                    });
                if close_now || clicked_outside {
                    dd_open = false;
                }
            }
            ui.data_mut(|d| d.insert_temp(dd_popup_id, dd_open));

            ui.separator();

            // Render hierarchical Shape+ Contents tree
            let doc = parse_shape_plus_document(&payload);
            let mut tree_action: Option<(String, &'static str, usize)> = None;
            ScrollArea::vertical()
                .id_salt("shape_plus_tree_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    fn draw_items(
                        ui: &mut Ui,
                        items: &[zc_render::shape_plus::ShapePlusItem],
                        depth: usize,
                        state: &mut PanelsState,
                        theme: &Cm3Theme,
                        action: &mut Option<(String, &'static str, usize)>,
                    ) {
                        let count = items.len();
                        for (idx, item) in items.iter().enumerate() {
                            let is_sel =
                                state.shape_plus_selected.as_deref() == Some(item.id.as_str());
                            let enabled = item
                                .properties
                                .get("enabled")
                                .and_then(Value::as_f64)
                                .unwrap_or(1.0)
                                >= 0.5;
                            let name = item
                                .properties
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_else(|| {
                                    shape_plus_kind_title(&item.kind).unwrap_or(&item.kind)
                                });
                            let (row_rect, row_resp) = ui.allocate_exact_size(
                                Vec2::new(ui.available_width(), 24.0),
                                Sense::click(),
                            );
                            let row_resp = row_resp.on_hover_cursor(CursorIcon::PointingHand);
                            let pointer_in_row = ui.input(|i| {
                                i.pointer.hover_pos().is_some_and(|p| row_rect.contains(p))
                            });
                            if is_sel {
                                ui.painter().rect_filled(
                                    row_rect,
                                    CornerRadius::ZERO,
                                    theme.accent(),
                                );
                            } else if row_resp.hovered() || pointer_in_row {
                                ui.painter()
                                    .rect_filled(row_rect, CornerRadius::ZERO, BG_HOVER);
                            }
                            ui.painter().line_segment(
                                [
                                    Pos2::new(row_rect.left(), row_rect.bottom() - 0.5),
                                    Pos2::new(row_rect.right(), row_rect.bottom() - 0.5),
                                ],
                                Stroke::new(1.0, BORDER_ROW_SOFT),
                            );

                            let mut sub_clicked = false;
                            let mut cursor_x = row_rect.left() + 4.0 + (depth as f32) * 14.0;
                            if item.kind == "group" {
                                let collapsed = state.shape_plus_collapsed.contains(&item.id);
                                let tri_rect = Rect::from_min_size(
                                    Pos2::new(cursor_x, (row_rect.center().y - 8.0).round()),
                                    Vec2::splat(16.0),
                                );
                                let tri_resp = ui
                                    .interact(
                                        tri_rect,
                                        Id::unique(("sp_grp_col", &item.id)),
                                        Sense::click(),
                                    )
                                    .on_hover_cursor(CursorIcon::PointingHand)
                                    .on_hover_text(if collapsed {
                                        "Expand group"
                                    } else {
                                        "Collapse group"
                                    });
                                paint_cm3_icon(
                                    ui.painter(),
                                    tri_rect.shrink(2.0),
                                    if collapsed {
                                        Cm3Icon::ChevronRight
                                    } else {
                                        Cm3Icon::ChevronDown
                                    },
                                    if is_sel { TEXT_BRIGHT } else { TEXT_MUTED },
                                );
                                if tri_resp.clicked() {
                                    sub_clicked = true;
                                    if collapsed {
                                        state.shape_plus_collapsed.remove(&item.id);
                                    } else {
                                        state.shape_plus_collapsed.insert(item.id.clone());
                                    }
                                }
                                cursor_x += 18.0;
                            }

                            let eye_rect = Rect::from_min_size(
                                Pos2::new(cursor_x, (row_rect.center().y - 9.0).round()),
                                Vec2::splat(18.0),
                            );
                            let mut eye_ui =
                                ui.new_child(egui::UiBuilder::new().max_rect(eye_rect));
                            if paint_visibility_eye_button(
                                &mut eye_ui,
                                Id::unique(("sp_vis", &item.id)),
                                enabled,
                                theme,
                            ) {
                                sub_clicked = true;
                                *action = Some((item.id.clone(), "toggle", idx));
                            }
                            cursor_x += 22.0;

                            // Right-aligned compact action buttons (×, D, ↓, ↑)
                            let mut right_x = row_rect.right() - 4.0;
                            let btn_y = (row_rect.center().y - 8.5).round();
                            let draw_row_btn = |ui: &mut Ui,
                                                right_x: &mut f32,
                                                btn_id: Id,
                                                lbl: &str,
                                                tip: &str|
                             -> bool {
                                let w = 17.0;
                                *right_x -= w;
                                let r = Rect::from_min_size(
                                    Pos2::new(*right_x, btn_y),
                                    Vec2::new(w, 17.0),
                                );
                                *right_x -= 2.0;
                                let (clicked, _) =
                                    header_compact_action_button(ui, btn_id, r, lbl, tip, theme);
                                clicked
                            };
                            if draw_row_btn(
                                ui,
                                &mut right_x,
                                Id::unique(("sp_del", &item.id)),
                                "×",
                                "Delete",
                            ) {
                                sub_clicked = true;
                                *action = Some((item.id.clone(), "del", idx));
                            }
                            if draw_row_btn(
                                ui,
                                &mut right_x,
                                Id::unique(("sp_dup", &item.id)),
                                "D",
                                "Duplicate",
                            ) {
                                sub_clicked = true;
                                *action = Some((item.id.clone(), "dup", idx));
                            }
                            if idx + 1 < count
                                && draw_row_btn(
                                    ui,
                                    &mut right_x,
                                    Id::unique(("sp_dn", &item.id)),
                                    "↓",
                                    "Down",
                                )
                            {
                                sub_clicked = true;
                                *action = Some((item.id.clone(), "down", idx + 1));
                            }
                            if idx > 0
                                && draw_row_btn(
                                    ui,
                                    &mut right_x,
                                    Id::unique(("sp_up", &item.id)),
                                    "↑",
                                    "Up",
                                )
                            {
                                sub_clicked = true;
                                *action = Some((item.id.clone(), "up", idx - 1));
                            }

                            let label_rect = Rect::from_min_max(
                                Pos2::new(cursor_x, row_rect.top()),
                                Pos2::new((right_x - 4.0).max(cursor_x + 8.0), row_rect.bottom()),
                            );
                            ui.painter().with_clip_rect(label_rect).text(
                                Pos2::new(label_rect.left(), label_rect.center().y),
                                Align2::LEFT_CENTER,
                                name,
                                theme.font(FONT_SMALL),
                                if !enabled {
                                    TEXT_DISABLED
                                } else if is_sel {
                                    TEXT_BRIGHT
                                } else {
                                    TEXT_PRIMARY
                                },
                            );

                            if row_resp.clicked() && !sub_clicked {
                                state.shape_plus_selected = Some(item.id.clone());
                            }
                            if item.kind == "group"
                                && !state.shape_plus_collapsed.contains(&item.id)
                            {
                                draw_items(ui, &item.contents, depth + 1, state, theme, action);
                            }
                        }
                    }
                    draw_items(ui, &doc.contents, 0, state, theme, &mut tree_action);
                });

            if let Some((item_id, op, target_idx)) = tree_action {
                match op {
                    "del" => {
                        if remove_shape_plus_item(&mut payload, &item_id).unwrap_or(false) {
                            if state.shape_plus_selected.as_deref() == Some(item_id.as_str()) {
                                state.shape_plus_selected = None;
                            }
                            payload_dirty = true;
                        }
                    }
                    "dup" => {
                        if let Ok(Some(new_id)) = duplicate_shape_plus_item(&mut payload, &item_id)
                        {
                            state.shape_plus_selected = Some(new_id);
                            payload_dirty = true;
                        }
                    }
                    "up" | "down" => {
                        if reorder_shape_plus_item(&mut payload, &item_id, target_idx)
                            .unwrap_or(false)
                        {
                            payload_dirty = true;
                        }
                    }
                    "toggle" => {
                        if let Some(item_obj) = find_shape_item_mut(&mut payload, &item_id)
                            && let Some(props) = item_obj
                                .get_mut("properties")
                                .and_then(Value::as_object_mut)
                        {
                            let cur = props.get("enabled").and_then(Value::as_f64).unwrap_or(1.0);
                            props.insert(
                                "enabled".to_owned(),
                                json!(if cur >= 0.5 { 0 } else { 1 }),
                            );
                            payload_dirty = true;
                        }
                    }
                    _ => {}
                }
            }
        },
    );
    paint_crisp_panel_border(ui.painter(), top_rect, BORDER_PANEL);

    // Bottom Pane: Selected Shape+ Item Property Inspector
    let mut shape_tx_tracker = WidgetTxTracker::default();
    show_split_pane(
        ui,
        Id::unique("shape_plus_split_p1"),
        bot_rect.shrink(4.0),
        |ui, _| {
            let Some(sel_id) = state.shape_plus_selected.clone() else {
                empty_panel_message(ui, "select a Shape+ item", theme);
                return;
            };
            let Some(item_obj) = find_shape_item_mut(&mut payload, &sel_id) else {
                empty_panel_message(ui, "select a Shape+ item", theme);
                return;
            };
            let kind = item_obj
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            ui.label(
                egui::RichText::new(format!(
                    "{} Properties",
                    shape_plus_kind_title(&kind).unwrap_or(&kind)
                ))
                .font(theme.font(FONT_TITLE))
                .color(TEXT_STRONG),
            );
            ui.separator();

            ScrollArea::vertical()
                .id_salt(("shape_plus_item_inspector", &sel_id))
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let Some(props) = item_obj
                        .get_mut("properties")
                        .and_then(Value::as_object_mut)
                    else {
                        return;
                    };
                    let mut keys: Vec<String> = props.keys().cloned().collect();
                    keys.sort_by(|a, b| {
                        shape_plus_prop_sort_key(a)
                            .cmp(&shape_plus_prop_sort_key(b))
                            .then_with(|| a.cmp(b))
                    });
                    for key in keys {
                        if key == "enabled" || key == "path" {
                            continue;
                        }
                        let Some(val) = props.get(&key).cloned() else {
                            continue;
                        };
                        let label = humanize_prop_label(&key);
                        match val {
                            Value::String(mut s) => {
                                if let Some(opts) = shape_plus_string_enum_options(&kind, &key) {
                                    let mut idx = opts
                                        .iter()
                                        .position(|o| o.eq_ignore_ascii_case(&s))
                                        .unwrap_or(0);
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 6.0;
                                        ui.label(
                                            egui::RichText::new(&label)
                                                .font(theme.font(FONT_SMALL))
                                                .color(TEXT_PRIMARY),
                                        );
                                        if cm3_dropdown(
                                            ui,
                                            Id::unique(("sp_prop_dd", &sel_id, &key)),
                                            &mut idx,
                                            opts,
                                            false,
                                            viewport_rect,
                                            theme,
                                        ) {
                                            props.insert(key, json!(opts[idx]));
                                            payload_dirty = true;
                                        }
                                    });
                                } else if let Some(summary) = oversized_text_summary(&s) {
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 6.0;
                                        ui.label(
                                            egui::RichText::new(&label)
                                                .font(theme.font(FONT_SMALL))
                                                .color(TEXT_PRIMARY),
                                        );
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(&summary)
                                                    .font(theme.font(FONT_SMALL))
                                                    .color(TEXT_MUTED),
                                            )
                                            .truncate(),
                                        )
                                        .on_hover_text(
                                            "Text exceeds 8,192 characters; inline editing disabled.",
                                        );
                                    });
                                } else {
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 6.0;
                                        ui.label(
                                            egui::RichText::new(&label)
                                                .font(theme.font(FONT_SMALL))
                                                .color(TEXT_PRIMARY),
                                        );
                                        if ui
                                            .add(
                                                TextEdit::singleline(&mut s)
                                                    .font(theme.font(FONT_SMALL))
                                                    .desired_width(130.0),
                                            )
                                            .changed()
                                        {
                                            props.insert(key, json!(s));
                                            payload_dirty = true;
                                        }
                                    });
                                }
                            }
                            Value::Number(n) => {
                                let mut v = n.as_f64().unwrap_or(0.0);
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(&label)
                                            .font(theme.font(FONT_SMALL))
                                            .color(TEXT_PRIMARY),
                                    );
                                    let resp = ui.add(egui::DragValue::new(&mut v).speed(0.5));
                                    if shape_tx_tracker.observe(&resp) {
                                        props.insert(key, json!(v));
                                        payload_dirty = true;
                                    }
                                });
                            }
                            Value::Array(arr) if arr.len() == 2 => {
                                let mut x = arr[0].as_f64().unwrap_or(0.0);
                                let mut y = arr[1].as_f64().unwrap_or(0.0);
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(&label)
                                            .font(theme.font(FONT_SMALL))
                                            .color(TEXT_PRIMARY),
                                    );
                                    let cx = ui.add(egui::DragValue::new(&mut x).speed(1.0));
                                    let cy = ui.add(egui::DragValue::new(&mut y).speed(1.0));
                                    let ch_x = shape_tx_tracker.observe(&cx);
                                    let ch_y = shape_tx_tracker.observe(&cy);
                                    if ch_x || ch_y {
                                        props.insert(key, json!([x, y]));
                                        payload_dirty = true;
                                    }
                                });
                            }
                            Value::Array(arr) if arr.len() == 3 => {
                                let mut rgb = [
                                    arr[0].as_f64().unwrap_or(1.0) as f32,
                                    arr[1].as_f64().unwrap_or(1.0) as f32,
                                    arr[2].as_f64().unwrap_or(1.0) as f32,
                                ];
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(&label)
                                            .font(theme.font(FONT_SMALL))
                                            .color(TEXT_PRIMARY),
                                    );
                                    let resp = ui.color_edit_button_rgb(&mut rgb);
                                    if shape_tx_tracker.observe(&resp) {
                                        props.insert(
                                            key,
                                            json!([
                                                f64::from(rgb[0]),
                                                f64::from(rgb[1]),
                                                f64::from(rgb[2])
                                            ]),
                                        );
                                        payload_dirty = true;
                                    }
                                });
                            }
                            _ => {}
                        }
                    }
                });
        },
    );
    paint_crisp_panel_border(ui.painter(), bot_rect, BORDER_PANEL);

    if payload_dirty {
        shape_tx_tracker.changed = true;
    }
    apply_widget_tx_command(
        ui,
        session,
        state,
        &clip.id,
        "shape_plus_payload",
        "edit shape+ property",
        shape_tx_tracker,
        |session| {
            let _ = session.execute(Command::UpdateShapePlusPayload {
                clip_id: cid,
                payload,
            });
        },
    );
}

/// Paints a crisp vector visibility eye icon (never relying on missing emoji glyphs).
pub(super) fn paint_visibility_eye_button(
    ui: &mut Ui,
    id: Id,
    visible: bool,
    theme: &Cm3Theme,
) -> bool {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(18.0, 18.0), Sense::click());
    let resp = resp
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text(if visible { "Hide item" } else { "Show item" });
    if resp.hovered() {
        ui.painter()
            .rect_filled(rect, CornerRadius::same(2), BG_HOVER);
    }
    let c = rect.center();
    let col = if visible {
        if resp.hovered() {
            TEXT_BRIGHT
        } else {
            theme.value()
        }
    } else if resp.hovered() {
        TEXT_PRIMARY
    } else {
        TEXT_DISABLED
    };
    let rx = 6.0_f32;
    let ry = 3.6_f32;
    let mut upper = Vec::with_capacity(9);
    let mut lower = Vec::with_capacity(9);
    for i in 0..=8 {
        let t = (i as f32 / 8.0) * 2.0 - 1.0;
        let x = c.x + t * rx;
        let arch = (1.0 - t * t) * ry;
        upper.push(Pos2::new(x, c.y - arch));
        lower.push(Pos2::new(x, c.y + arch));
    }
    ui.painter().add(Shape::line(upper, Stroke::new(1.3, col)));
    ui.painter().add(Shape::line(lower, Stroke::new(1.3, col)));
    if visible {
        ui.painter().circle_filled(c, 2.0, col);
    } else {
        ui.painter().line_segment(
            [
                Pos2::new(c.x - 5.5, c.y + 4.5),
                Pos2::new(c.x + 5.5, c.y - 4.5),
            ],
            Stroke::new(1.4, col),
        );
    }
    let _ = id;
    resp.clicked()
}

pub(super) fn shape_plus_prop_sort_key(key: &str) -> u8 {
    match key {
        "name" => 0,
        "mode" => 1,
        "position" => 2,
        "size" => 3,
        "points" => 4,
        "outerRadius" => 5,
        "innerRadius" => 6,
        "roundness" | "outerRoundness" => 7,
        "innerRoundness" => 8,
        "scale" => 9,
        "rotation" => 10,
        "anchor" => 11,
        "skew" => 12,
        "skewAxis" => 13,
        "color" => 14,
        "strokeWidth" => 15,
        "lineCap" | "cap" => 16,
        "lineJoin" | "join" => 17,
        "miterLimit" => 18,
        "opacity" => 19,
        "composite" => 20,
        "fillRule" | "fill_rule" => 21,
        _ => 100,
    }
}

pub(super) fn shape_plus_string_enum_options(
    kind: &str,
    key: &str,
) -> Option<&'static [&'static str]> {
    match (kind, key) {
        ("polystar", "mode") | (_, "polystarType" | "starType") => Some(&["star", "polygon"]),
        ("trim", "mode") => Some(&["simultaneous", "individually"]),
        ("mergePaths", "mode") => Some(&["add", "subtract", "intersect", "exclude"]),
        ("zigZag" | "wigglePaths", "points") => Some(&["corner", "smooth"]),
        (_, "composite") => Some(&["below", "above"]),
        (_, "fillRule" | "fill_rule") => Some(&["nonzero", "evenodd"]),
        (_, "lineCap" | "cap") => Some(&["butt", "round", "square"]),
        (_, "lineJoin" | "join") => Some(&["miter", "round", "bevel"]),
        _ => None,
    }
}

fn find_enclosing_shape_group<'a>(payload: &'a Value, id: &str) -> Option<&'a str> {
    fn rec<'a>(arr: &'a [Value], id: &str, parent_group: Option<&'a str>) -> Option<&'a str> {
        for item in arr {
            let item_id = item.get("id").and_then(Value::as_str);
            let item_kind = item.get("kind").and_then(Value::as_str);
            if item_id == Some(id) {
                return if item_kind == Some("group") {
                    item_id
                } else {
                    parent_group
                };
            }
            if let Some(children) = item.get("contents").and_then(Value::as_array) {
                let next_parent = if item_kind == Some("group") {
                    item_id
                } else {
                    parent_group
                };
                if let Some(found) = rec(children, id, next_parent) {
                    return Some(found);
                }
            }
        }
        None
    }
    payload
        .get("contents")
        .and_then(Value::as_array)
        .and_then(|c| rec(c, id, None))
}

pub(super) fn add_shape_plus_item_smart(
    payload: &mut Value,
    selected_id: Option<&str>,
    kind: &str,
) -> Result<String, String> {
    let target_group: Option<String> = selected_id
        .and_then(|sid| find_enclosing_shape_group(payload, sid))
        .map(str::to_owned);
    let is_geometry_shape = matches!(kind, "rectangle" | "ellipse" | "polystar" | "path");
    if let Some(gid) = target_group {
        add_shape_plus_item(payload, Some(&gid), kind)
    } else if is_geometry_shape {
        // Wrap root-inserted shapes inside a new Group with a default Fill so they render immediately
        let group_id = add_shape_plus_item(payload, None, "group")?;
        let shape_id = add_shape_plus_item(payload, Some(&group_id), kind)?;
        let _ = add_shape_plus_item(payload, Some(&group_id), "fill");
        Ok(shape_id)
    } else {
        add_shape_plus_item(payload, None, kind)
    }
}

fn find_shape_item_mut<'a>(payload: &'a mut Value, id: &str) -> Option<&'a mut Value> {
    fn rec<'a>(arr: &'a mut [Value], id: &str) -> Option<&'a mut Value> {
        for item in arr {
            if item.get("id").and_then(Value::as_str) == Some(id) {
                return Some(item);
            }
            if let Some(children) = item.get_mut("contents").and_then(Value::as_array_mut)
                && let Some(found) = rec(children, id)
            {
                return Some(found);
            }
        }
        None
    }
    payload
        .get_mut("contents")
        .and_then(Value::as_array_mut)
        .and_then(|c| rec(c, id))
}

#[allow(clippy::too_many_arguments)]
fn show_text2d_objects_tab(
    ui: &mut Ui,
    _panel_rect: Rect,
    viewport_rect: Rect,
    clip: &SessionClip,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let local_frame = session.playback.current_frame - clip.start;
    let cid = Uuid::parse_str(&clip.id).ok();

    ScrollArea::vertical()
        .id_salt("text2d_objects_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            panel_top_header(ui, &clip.name, theme);
            show_session_text_prop_ex(
                ui,
                session,
                &clip.id,
                "text",
                "Text",
                "Text",
                true,
                local_frame,
                state,
                theme,
            );
            show_font_selector_row(
                ui,
                session,
                &clip.id,
                "text2d_obj_font",
                local_frame,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_number_prop(
                ui,
                session,
                &clip.id,
                "fontSize",
                "Font size",
                72.0,
                ScrubConfig::new(4.0, 1000.0, 1.0, 0),
                local_frame,
                clip.start,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_color_prop(
                ui,
                session,
                &clip.id,
                "color",
                "Fill color",
                [1.0, 1.0, 1.0],
                local_frame,
                clip.start,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_color_prop(
                ui,
                session,
                &clip.id,
                "strokeColor",
                "Stroke color",
                [0.0, 0.0, 0.0],
                local_frame,
                clip.start,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_number_prop(
                ui,
                session,
                &clip.id,
                "strokeWidth",
                "Stroke width",
                0.0,
                ScrubConfig::new(0.0, 100.0, 0.5, 1),
                local_frame,
                clip.start,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_option_prop_ex(
                ui,
                session,
                &clip.id,
                "align",
                "Alignment",
                &["Left", "Center", "Right"],
                1,
                false,
                local_frame,
                clip.start,
                viewport_rect,
                state,
                theme,
                actions,
            );

            // Text+ 2D 3-Stack Character Animators (`zoidiumTextPlus2D`)
            if let Some(clip_uuid) = cid {
                cm3_collapsible_section(
                    ui,
                    Id::unique(("text2d_plus_animators", &clip.id)),
                    "Text+ 2D Character Animators",
                    true,
                    theme,
                    |ui| {
                        let mut root = find_project_video_clip(session, &clip.id)
                            .and_then(|vc| vc.layer.preserved.get("zoidiumTextPlus2D").cloned())
                            .filter(Value::is_object)
                            .unwrap_or_else(default_text_plus_2d_root);

                        // 3-Stack tabs: Char 1 | Char 2 | Char 3 (compact padding, zero right-edge clipping)
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            ui.spacing_mut().button_padding = Vec2::new(6.0, 2.0);
                            for (idx, lbl) in ["Char 1", "Char 2", "Char 3"].iter().enumerate() {
                                if ui
                                    .selectable_label(
                                        state.text2d_animator_tab == idx,
                                        egui::RichText::new(*lbl).font(theme.font(FONT_SMALL)),
                                    )
                                    .clicked()
                                {
                                    state.text2d_animator_tab = idx;
                                }
                            }
                        });
                        ui.add_space(4.0);

                        if let Some(obj) = root.as_object_mut() {
                            obj.entry("schemaVersion").or_insert_with(|| json!(1));
                        }

                        let slot_obj = text_plus_2d_slot_mut(&mut root, state.text2d_animator_tab);

                        let mut dirty = false;
                        if let Some(slot) = slot_obj {
                            const MODES: &[&str] = &[
                                "None",
                                "Character Transform",
                                "Gradient Transform",
                                "Character Shake",
                                "Selected Transform",
                                "Selected Shake",
                                "Random Scatter",
                            ];
                            const DELAY_ORDERS: &[&str] = &["Start", "End", "Middle", "Random"];
                            let mut mode = tp2d_read_mode(slot);
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new("Mode")
                                        .font(theme.font(FONT_SMALL))
                                        .color(TEXT_PRIMARY),
                                );
                                if cm3_dropdown(
                                    ui,
                                    Id::unique((
                                        "text2d_anim_mode",
                                        &clip.id,
                                        state.text2d_animator_tab,
                                    )),
                                    &mut mode,
                                    MODES,
                                    false,
                                    viewport_rect,
                                    theme,
                                ) {
                                    slot.insert("mode".to_owned(), json!(mode));
                                    dirty = true;
                                }
                            });

                            // Character range / indices input for Selected Transform (4) & Selected Shake (5)
                            if matches!(mode, 4 | 5) {
                                let mut chars_str = tp2d_read_str(slot, "characterNumbers", "1");
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new("Characters")
                                            .font(theme.font(FONT_COMPACT))
                                            .color(TEXT_MUTED),
                                    );
                                    if ui
                                        .add(
                                            TextEdit::singleline(&mut chars_str)
                                                .hint_text("1, 3, 5")
                                                .font(theme.font(FONT_COMPACT))
                                                .desired_width(
                                                    (ui.available_width() - 6.0).max(60.0),
                                                ),
                                        )
                                        .on_hover_text("1-based character indices (e.g. 1, 2, 4)")
                                        .changed()
                                    {
                                        slot.insert(
                                            "characterNumbers".to_owned(),
                                            json!(chars_str),
                                        );
                                        dirty = true;
                                    }
                                });
                            }

                            let tx_cell = std::cell::Cell::new(WidgetTxTracker::default());
                            let drag_scalar = |ui: &mut Ui,
                                               lbl: &str,
                                               v: &mut f64,
                                               speed: f64,
                                               range: Option<(f64, f64)>|
                             -> bool {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(lbl)
                                            .font(theme.font(FONT_COMPACT))
                                            .color(TEXT_MUTED),
                                    );
                                    let mut dv = egui::DragValue::new(v).speed(speed);
                                    if let Some((lo, hi)) = range {
                                        dv = dv.range(lo..=hi);
                                    }
                                    let resp = ui.add(dv);
                                    let mut tr = tx_cell.get();
                                    let ch = tr.observe(&resp);
                                    tx_cell.set(tr);
                                    ch
                                })
                                .inner
                            };

                            let drag_vec2 = |ui: &mut Ui,
                                             lbl: &str,
                                             v: &mut [f64; 2],
                                             speed: f64|
                             -> bool {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 4.0;
                                    ui.label(
                                        egui::RichText::new(lbl)
                                            .font(theme.font(FONT_COMPACT))
                                            .color(TEXT_MUTED),
                                    );
                                    let cx = ui.add(
                                        egui::DragValue::new(&mut v[0]).speed(speed).prefix("X "),
                                    );
                                    let cy = ui.add(
                                        egui::DragValue::new(&mut v[1]).speed(speed).prefix("Y "),
                                    );
                                    let mut tr = tx_cell.get();
                                    let ch_x = tr.observe(&cx);
                                    let ch_y = tr.observe(&cy);
                                    tx_cell.set(tr);
                                    ch_x || ch_y
                                })
                                .inner
                            };

                            match mode {
                                // Character Transform (1) & Selected Transform (4)
                                1 | 4 => {
                                    let mut pos = tp2d_read_vec2(
                                        slot,
                                        "charPosition",
                                        ("offsetX", "offsetY"),
                                        [0.0, 0.0],
                                    );
                                    if drag_vec2(ui, "Position", &mut pos, 1.0) {
                                        tp2d_write_vec2(slot, "charPosition", pos);
                                        dirty = true;
                                    }
                                    let mut scale = tp2d_read_vec2(
                                        slot,
                                        "charScale",
                                        ("scaleX", "scaleY"),
                                        [1.0, 1.0],
                                    );
                                    if drag_vec2(ui, "Scale", &mut scale, 0.02) {
                                        tp2d_write_vec2(slot, "charScale", scale);
                                        dirty = true;
                                    }
                                    let mut rot =
                                        tp2d_read_f64(slot, "charRotation", &["rotation"], 0.0);
                                    if drag_scalar(ui, "Rotation (rad)", &mut rot, 0.02, None) {
                                        tp2d_write_f64(slot, "charRotation", rot);
                                        dirty = true;
                                    }
                                    let mut opacity =
                                        tp2d_read_f64(slot, "charOpacity", &["opacity"], 1.0);
                                    if drag_scalar(
                                        ui,
                                        "Opacity",
                                        &mut opacity,
                                        0.01,
                                        Some((0.0, 1.0)),
                                    ) {
                                        tp2d_write_f64(slot, "charOpacity", opacity);
                                        dirty = true;
                                    }
                                    let mut delay =
                                        tp2d_read_f64(slot, "delayPerCharacter", &["delay"], 0.0);
                                    if drag_scalar(
                                        ui,
                                        "Char delay",
                                        &mut delay,
                                        0.25,
                                        Some((0.0, 1000.0)),
                                    ) {
                                        tp2d_write_f64(slot, "delayPerCharacter", delay);
                                        dirty = true;
                                    }
                                    let mut order = tp2d_read_order(slot, "delayOrder");
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new("Delay order")
                                                .font(theme.font(FONT_COMPACT))
                                                .color(TEXT_MUTED),
                                        );
                                        if cm3_dropdown(
                                            ui,
                                            Id::unique((
                                                "text2d_delay_order",
                                                &clip.id,
                                                state.text2d_animator_tab,
                                            )),
                                            &mut order,
                                            DELAY_ORDERS,
                                            false,
                                            viewport_rect,
                                            theme,
                                        ) {
                                            slot.insert("delayOrder".to_owned(), json!(order));
                                            dirty = true;
                                        }
                                    });
                                    let mut seed =
                                        tp2d_read_f64(slot, "randomSeed", &["seed"], 1.0);
                                    if drag_scalar(ui, "Random seed", &mut seed, 1.0, None) {
                                        tp2d_write_f64(slot, "randomSeed", seed);
                                        dirty = true;
                                    }
                                }
                                // Gradient Transform (2)
                                2 => {
                                    let mut p0 = tp2d_read_vec2(
                                        slot,
                                        "firstPosition",
                                        ("offsetX", "offsetY"),
                                        [0.0, 0.0],
                                    );
                                    if drag_vec2(ui, "First pos", &mut p0, 1.0) {
                                        tp2d_write_vec2(slot, "firstPosition", p0);
                                        dirty = true;
                                    }
                                    let mut p1 = tp2d_read_vec2(
                                        slot,
                                        "lastPosition",
                                        ("offsetX", "offsetY"),
                                        [0.0, 0.0],
                                    );
                                    if drag_vec2(ui, "Last pos", &mut p1, 1.0) {
                                        tp2d_write_vec2(slot, "lastPosition", p1);
                                        dirty = true;
                                    }
                                    let mut s0 = tp2d_read_vec2(
                                        slot,
                                        "firstScale",
                                        ("scaleX", "scaleY"),
                                        [1.0, 1.0],
                                    );
                                    if drag_vec2(ui, "First scale", &mut s0, 0.02) {
                                        tp2d_write_vec2(slot, "firstScale", s0);
                                        dirty = true;
                                    }
                                    let mut s1 = tp2d_read_vec2(
                                        slot,
                                        "lastScale",
                                        ("scaleX", "scaleY"),
                                        [1.0, 1.0],
                                    );
                                    if drag_vec2(ui, "Last scale", &mut s1, 0.02) {
                                        tp2d_write_vec2(slot, "lastScale", s1);
                                        dirty = true;
                                    }
                                    let mut r0 = tp2d_read_f64(slot, "firstRotation", &[], 0.0);
                                    if drag_scalar(ui, "First rot (rad)", &mut r0, 0.02, None) {
                                        tp2d_write_f64(slot, "firstRotation", r0);
                                        dirty = true;
                                    }
                                    let mut r1 =
                                        tp2d_read_f64(slot, "lastRotation", &["rotation"], 0.0);
                                    if drag_scalar(ui, "Last rot (rad)", &mut r1, 0.02, None) {
                                        tp2d_write_f64(slot, "lastRotation", r1);
                                        dirty = true;
                                    }
                                    let mut o0 = tp2d_read_f64(slot, "firstOpacity", &[], 1.0);
                                    if drag_scalar(
                                        ui,
                                        "First opacity",
                                        &mut o0,
                                        0.01,
                                        Some((0.0, 1.0)),
                                    ) {
                                        tp2d_write_f64(slot, "firstOpacity", o0);
                                        dirty = true;
                                    }
                                    let mut o1 =
                                        tp2d_read_f64(slot, "lastOpacity", &["opacity"], 1.0);
                                    if drag_scalar(
                                        ui,
                                        "Last opacity",
                                        &mut o1,
                                        0.01,
                                        Some((0.0, 1.0)),
                                    ) {
                                        tp2d_write_f64(slot, "lastOpacity", o1);
                                        dirty = true;
                                    }
                                }
                                // Character Shake (3) & Selected Shake (5)
                                3 | 5 => {
                                    let mut amp =
                                        tp2d_read_f64(slot, "shakeAmplitude", &["amount"], 0.25);
                                    if drag_scalar(ui, "Amplitude (rad)", &mut amp, 0.02, None) {
                                        tp2d_write_f64(slot, "shakeAmplitude", amp);
                                        dirty = true;
                                    }
                                    let mut period =
                                        tp2d_read_f64(slot, "shakePeriod", &["speed"], 24.0);
                                    if drag_scalar(
                                        ui,
                                        "Period (frames)",
                                        &mut period,
                                        0.5,
                                        Some((0.1, 10_000.0)),
                                    ) {
                                        tp2d_write_f64(slot, "shakePeriod", period);
                                        dirty = true;
                                    }
                                    let mut phase = tp2d_read_f64(slot, "shakePhase", &[], 0.0);
                                    if drag_scalar(ui, "Phase", &mut phase, 0.25, None) {
                                        tp2d_write_f64(slot, "shakePhase", phase);
                                        dirty = true;
                                    }
                                    let mut delay = tp2d_read_f64(
                                        slot,
                                        "shakeDelayPerCharacter",
                                        &["delay"],
                                        0.0,
                                    );
                                    if drag_scalar(
                                        ui,
                                        "Char delay",
                                        &mut delay,
                                        0.25,
                                        Some((0.0, 1000.0)),
                                    ) {
                                        tp2d_write_f64(slot, "shakeDelayPerCharacter", delay);
                                        dirty = true;
                                    }
                                    let mut order = tp2d_read_order(slot, "shakeDelayOrder");
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new("Delay order")
                                                .font(theme.font(FONT_COMPACT))
                                                .color(TEXT_MUTED),
                                        );
                                        if cm3_dropdown(
                                            ui,
                                            Id::unique((
                                                "text2d_shake_delay_order",
                                                &clip.id,
                                                state.text2d_animator_tab,
                                            )),
                                            &mut order,
                                            DELAY_ORDERS,
                                            false,
                                            viewport_rect,
                                            theme,
                                        ) {
                                            slot.insert("shakeDelayOrder".to_owned(), json!(order));
                                            dirty = true;
                                        }
                                    });
                                    let mut seed =
                                        tp2d_read_f64(slot, "shakeRandomSeed", &["seed"], 1.0);
                                    if drag_scalar(ui, "Random seed", &mut seed, 1.0, None) {
                                        tp2d_write_f64(slot, "shakeRandomSeed", seed);
                                        dirty = true;
                                    }
                                }
                                // Random Scatter (6)
                                6 => {
                                    let mut amount = tp2d_read_f64(slot, "amount", &[], 1.0);
                                    if drag_scalar(
                                        ui,
                                        "Amount",
                                        &mut amount,
                                        0.02,
                                        Some((0.0, 100.0)),
                                    ) {
                                        tp2d_write_f64(slot, "amount", amount);
                                        dirty = true;
                                    }
                                    let mut pmin = tp2d_read_vec2(
                                        slot,
                                        "positionMin",
                                        ("offsetX", "offsetY"),
                                        [-10.0, -10.0],
                                    );
                                    if drag_vec2(ui, "Pos min", &mut pmin, 1.0) {
                                        tp2d_write_vec2(slot, "positionMin", pmin);
                                        dirty = true;
                                    }
                                    let mut pmax = tp2d_read_vec2(
                                        slot,
                                        "positionMax",
                                        ("offsetX", "offsetY"),
                                        [10.0, 10.0],
                                    );
                                    if drag_vec2(ui, "Pos max", &mut pmax, 1.0) {
                                        tp2d_write_vec2(slot, "positionMax", pmax);
                                        dirty = true;
                                    }
                                    let mut rmin = tp2d_read_f64(slot, "rotationMin", &[], 0.0);
                                    if drag_scalar(ui, "Rot min (rad)", &mut rmin, 0.02, None) {
                                        tp2d_write_f64(slot, "rotationMin", rmin);
                                        dirty = true;
                                    }
                                    let mut rmax =
                                        tp2d_read_f64(slot, "rotationMax", &["rotation"], 0.0);
                                    if drag_scalar(ui, "Rot max (rad)", &mut rmax, 0.02, None) {
                                        tp2d_write_f64(slot, "rotationMax", rmax);
                                        dirty = true;
                                    }
                                    let mut smin = tp2d_read_vec2(
                                        slot,
                                        "scaleMin",
                                        ("scaleX", "scaleY"),
                                        [1.0, 1.0],
                                    );
                                    if drag_vec2(ui, "Scale min", &mut smin, 0.02) {
                                        tp2d_write_vec2(slot, "scaleMin", smin);
                                        dirty = true;
                                    }
                                    let mut smax = tp2d_read_vec2(
                                        slot,
                                        "scaleMax",
                                        ("scaleX", "scaleY"),
                                        [1.0, 1.0],
                                    );
                                    if drag_vec2(ui, "Scale max", &mut smax, 0.02) {
                                        tp2d_write_vec2(slot, "scaleMax", smax);
                                        dirty = true;
                                    }
                                    let mut seed = tp2d_read_f64(slot, "seed", &[], 1.0);
                                    if drag_scalar(ui, "Seed", &mut seed, 1.0, None) {
                                        tp2d_write_f64(slot, "seed", seed);
                                        dirty = true;
                                    }
                                }
                                _ => {}
                            }
                            if dirty {
                                let mut tr = tx_cell.get();
                                tr.changed = true;
                                tx_cell.set(tr);
                            }
                            apply_widget_tx_command(
                                ui,
                                session,
                                state,
                                &clip.id,
                                "text2d_animator",
                                "edit text2d animator",
                                tx_cell.get(),
                                |session| {
                                    let _ = session.execute(Command::UpdateLayerPreserved {
                                        clip_id: clip_uuid,
                                        key: "zoidiumTextPlus2D".to_owned(),
                                        value: Some(root),
                                    });
                                },
                            );
                        }
                    },
                );
            }
        });
}

pub(super) fn default_text_plus_2d_root() -> Value {
    json!({
        "schemaVersion": 1,
        "characterProperties": { "mode": 0 },
        "extraCharacterProperties": [{ "mode": 0 }, { "mode": 0 }]
    })
}

pub(super) fn text_plus_2d_slot_mut(
    root: &mut Value,
    tab_idx: usize,
) -> Option<&mut serde_json::Map<String, Value>> {
    match tab_idx {
        0 => root.as_object_mut().and_then(|o| {
            o.entry("characterProperties")
                .or_insert_with(|| json!({ "mode": 0 }))
                .as_object_mut()
        }),
        n => {
            let extra_idx = n - 1;
            root.as_object_mut().and_then(|o| {
                let arr = o
                    .entry("extraCharacterProperties")
                    .or_insert_with(|| json!([{ "mode": 0 }, { "mode": 0 }]))
                    .as_array_mut()?;
                while arr.len() <= extra_idx {
                    arr.push(json!({ "mode": 0 }));
                }
                arr[extra_idx].as_object_mut()
            })
        }
    }
}

fn tp2d_unwrap_scalar_f64(v: &Value) -> Option<f64> {
    if let Some(n) = v.as_f64() {
        return n.is_finite().then_some(n);
    }
    if let Some(inner) = v.get("value") {
        return tp2d_unwrap_scalar_f64(inner);
    }
    if let Some(arr) = v.as_array()
        && let Some(first) = arr.first()
    {
        return tp2d_unwrap_scalar_f64(first);
    }
    None
}

fn tp2d_read_mode(slot: &serde_json::Map<String, Value>) -> usize {
    let Some(v) = slot.get("mode") else {
        return 0;
    };
    if let Some(s) = v
        .as_str()
        .or_else(|| v.get("value").and_then(Value::as_str))
    {
        return match s.trim().to_ascii_lowercase().as_str() {
            "1" | "character-transform" | "character transform" => 1,
            "2" | "gradient-transform" | "gradient transform" => 2,
            "3" | "character-shake" | "character shake" => 3,
            "4" | "selected-transform" | "selected character transform" => 4,
            "5" | "selected-shake" | "selected character shake" => 5,
            "6" | "random-scatter" | "random scatter" => 6,
            _ => 0,
        };
    }
    (tp2d_unwrap_scalar_f64(v).unwrap_or(0.0).round() as i64).clamp(0, 6) as usize
}

fn tp2d_read_order(slot: &serde_json::Map<String, Value>, key: &str) -> usize {
    let Some(v) = slot.get(key) else {
        return 0;
    };
    if let Some(s) = v
        .as_str()
        .or_else(|| v.get("value").and_then(Value::as_str))
    {
        return match s.trim().to_ascii_lowercase().as_str() {
            "1" | "end" | "reverse" => 1,
            "2" | "middle" | "center out" | "center-out" => 2,
            "3" | "random" => 3,
            _ => 0,
        };
    }
    (tp2d_unwrap_scalar_f64(v).unwrap_or(0.0).round() as i64).clamp(0, 3) as usize
}

fn tp2d_read_str(slot: &serde_json::Map<String, Value>, key: &str, default: &str) -> String {
    let Some(v) = slot.get(key) else {
        return default.to_owned();
    };
    if let Some(s) = v.as_str() {
        return s.to_owned();
    }
    if let Some(s) = v.get("value").and_then(Value::as_str) {
        return s.to_owned();
    }
    default.to_owned()
}

pub(super) fn tp2d_read_f64(
    slot: &serde_json::Map<String, Value>,
    canonical_key: &str,
    legacy_keys: &[&str],
    default: f64,
) -> f64 {
    if let Some(n) = slot.get(canonical_key).and_then(tp2d_unwrap_scalar_f64) {
        return n;
    }
    for &lk in legacy_keys {
        if let Some(n) = slot.get(lk).and_then(tp2d_unwrap_scalar_f64) {
            return n;
        }
    }
    default
}

pub(super) fn tp2d_write_f64(
    slot: &mut serde_json::Map<String, Value>,
    canonical_key: &str,
    val: f64,
) {
    slot.insert(canonical_key.to_owned(), json!(val));
}

fn tp2d_unwrap_vec2(v: &Value, default: [f64; 2]) -> Option<[f64; 2]> {
    if let Some(arr) = v.as_array()
        && arr.len() >= 2
    {
        return Some([
            arr[0]
                .as_f64()
                .filter(|x| x.is_finite())
                .unwrap_or(default[0]),
            arr[1]
                .as_f64()
                .filter(|x| x.is_finite())
                .unwrap_or(default[1]),
        ]);
    }
    if let Some(objs) = v.get("objects").and_then(Value::as_array)
        && objs.len() >= 2
    {
        return Some([
            tp2d_unwrap_scalar_f64(&objs[0]).unwrap_or(default[0]),
            tp2d_unwrap_scalar_f64(&objs[1]).unwrap_or(default[1]),
        ]);
    }
    if let Some(inner) = v.get("value") {
        return tp2d_unwrap_vec2(inner, default);
    }
    None
}

pub(super) fn tp2d_read_vec2(
    slot: &serde_json::Map<String, Value>,
    canonical_key: &str,
    legacy_xy: (&str, &str),
    default: [f64; 2],
) -> [f64; 2] {
    if let Some(v) = slot
        .get(canonical_key)
        .and_then(|val| tp2d_unwrap_vec2(val, default))
    {
        return v;
    }
    let lx = slot.get(legacy_xy.0).and_then(tp2d_unwrap_scalar_f64);
    let ly = slot.get(legacy_xy.1).and_then(tp2d_unwrap_scalar_f64);
    if lx.is_some() || ly.is_some() {
        return [lx.unwrap_or(default[0]), ly.unwrap_or(default[1])];
    }
    default
}

pub(super) fn tp2d_write_vec2(
    slot: &mut serde_json::Map<String, Value>,
    canonical_key: &str,
    val: [f64; 2],
) {
    slot.insert(canonical_key.to_owned(), json!([val[0], val[1]]));
}
