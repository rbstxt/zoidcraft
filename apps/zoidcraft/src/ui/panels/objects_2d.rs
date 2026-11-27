use super::edit_tab::show_preset_shape_kind_row;
use super::effects_inspector::oversized_text_summary;
use super::property_rows::{
    lookup_session_prop_ref, show_session_color_prop, show_session_number_prop,
    show_session_option_prop_ex, show_session_text_prop_ex,
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
                show_clip_track_matte_rows(ui, session, &clip.id, viewport_rect, state, theme);
            });
        return true;
    }

    if matches!(
        clip.kind.as_str(),
        "video" | "image" | "composite" | "adjustment"
    ) {
        ScrollArea::vertical()
            .id_salt(("visual_clip_objects_scroll", &clip.id))
            .auto_shrink([false, false])
            .show(ui, |ui| {
                panel_top_header(ui, &clip.name, theme);
                show_clip_track_matte_rows(ui, session, &clip.id, viewport_rect, state, theme);
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

    let local_frame = session.playback.current_frame - clip.start;
    let clip_start = clip.start;
    let easing_plus_enabled = session.plugins.is_enabled("easing-plus");

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
    let mut pending_seek_frame: Option<f64> = None;
    let mut pending_convert_id: Option<String> = None;

    let rename_id = Id::unique(("sp_inline_rename", &clip.id));
    let mut rename_state: Option<(String, String, bool)> = ui
        .data_mut(|d| d.get_temp::<Option<(String, String, bool)>>(rename_id))
        .flatten();

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
                #[cfg(target_arch = "wasm32")]
                if ui.small_button("Import SVG...").clicked() {
                    actions.push(PanelAction::ImportWebSvg(cid));
                }
                #[cfg(not(target_arch = "wasm32"))]
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

            // Render hierarchical Shape+ Contents tree with paint swatches, scope hints,
            // Convert to Path, and inline rename.
            let doc = parse_shape_plus_document(&payload);
            let scope = compute_shape_plus_paint_scope(&doc.contents, local_frame);
            let mut tree_action: Option<ShapePlusTreeAction> = None;
            ScrollArea::vertical()
                .id_salt("shape_plus_tree_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    #[allow(clippy::too_many_arguments)]
                    fn draw_items(
                        ui: &mut Ui,
                        items: &[zc_render::shape_plus::ShapePlusItem],
                        depth: usize,
                        local_frame: f64,
                        scope: &BTreeMap<String, ShapePlusScopeEntry>,
                        state: &mut PanelsState,
                        rename_state: &mut Option<(String, String, bool)>,
                        theme: &Cm3Theme,
                        action: &mut Option<ShapePlusTreeAction>,
                    ) {
                        let count = items.len();
                        for (idx, item) in items.iter().enumerate() {
                            let is_sel =
                                state.shape_plus_selected.as_deref() == Some(item.id.as_str());
                            let enabled = sp_item_enabled(item, local_frame);
                            let default_title =
                                shape_plus_kind_title(&item.kind).unwrap_or(&item.kind);
                            let name = sp_eval_str(
                                item.properties.get("name"),
                                local_frame,
                                default_title,
                            );
                            let scope_hint = shape_plus_scope_hint(scope, item);
                            let is_renaming = rename_state
                                .as_ref()
                                .is_some_and(|(rid, _, _)| rid == &item.id);

                            let (row_rect, row_resp) = ui.allocate_exact_size(
                                Vec2::new(ui.available_width(), 24.0),
                                Sense::click(),
                            );
                            let row_resp = row_resp.on_hover_cursor(CursorIcon::PointingHand);
                            let row_resp = if let Some(hint) = scope_hint {
                                let tip = if item.kind == "textAnimator" {
                                    "A text animator affects text items above it in the same group."
                                } else {
                                    "A fill or stroke draws the paths above it in the same group."
                                };
                                row_resp.on_hover_text(format!("{hint} — {tip}"))
                            } else {
                                row_resp
                            };
                            let pointer_in_row = ui.input(|i| {
                                i.pointer.hover_pos().is_some_and(|p| row_rect.contains(p))
                            });
                            let row_bg = if is_sel {
                                theme.accent()
                            } else if row_resp.hovered() || pointer_in_row {
                                BG_HOVER
                            } else {
                                BG_PANEL
                            };
                            if row_bg != BG_PANEL {
                                ui.painter()
                                    .rect_filled(row_rect, CornerRadius::ZERO, row_bg);
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
                                *action = Some(ShapePlusTreeAction::ToggleEnabled(item.id.clone()));
                            }
                            cursor_x += 22.0;

                            // Right-aligned compact action buttons (×, D, ↓, ↑, plus P for Convert to Path)
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
                                *action = Some(ShapePlusTreeAction::Delete(item.id.clone()));
                            }
                            if draw_row_btn(
                                ui,
                                &mut right_x,
                                Id::unique(("sp_dup", &item.id)),
                                "D",
                                "Duplicate",
                            ) {
                                sub_clicked = true;
                                *action = Some(ShapePlusTreeAction::Duplicate(item.id.clone()));
                            }
                            if idx + 1 < count
                                && draw_row_btn(
                                    ui,
                                    &mut right_x,
                                    Id::unique(("sp_dn", &item.id)),
                                    "↓",
                                    "Move Down",
                                )
                            {
                                sub_clicked = true;
                                *action =
                                    Some(ShapePlusTreeAction::Reorder(item.id.clone(), idx + 1));
                            }
                            if idx > 0
                                && draw_row_btn(
                                    ui,
                                    &mut right_x,
                                    Id::unique(("sp_up", &item.id)),
                                    "↑",
                                    "Move Up",
                                )
                            {
                                sub_clicked = true;
                                *action =
                                    Some(ShapePlusTreeAction::Reorder(item.id.clone(), idx - 1));
                            }
                            if is_shape_plus_convertible_kind(&item.kind) {
                                let conv_tip = if item.kind == "text" {
                                    "Create Shapes from Text"
                                } else {
                                    "Convert to Path"
                                };
                                if draw_row_btn(
                                    ui,
                                    &mut right_x,
                                    Id::unique(("sp_conv", &item.id)),
                                    "P",
                                    conv_tip,
                                ) {
                                    sub_clicked = true;
                                    *action =
                                        Some(ShapePlusTreeAction::ConvertToPath(item.id.clone()));
                                }
                            }

                            let label_rect = Rect::from_min_max(
                                Pos2::new(cursor_x, row_rect.top()),
                                Pos2::new((right_x - 4.0).max(cursor_x + 8.0), row_rect.bottom()),
                            );

                            if is_renaming {
                                if let Some((_, buf, just_started)) = rename_state.as_mut() {
                                    let edit_rect = Rect::from_min_max(
                                        Pos2::new(label_rect.left(), row_rect.top() + 2.0),
                                        Pos2::new(label_rect.right(), row_rect.bottom() - 2.0),
                                    );
                                    let mut edit_ui =
                                        ui.new_child(egui::UiBuilder::new().max_rect(edit_rect));
                                    let te_resp = edit_ui.add(
                                        TextEdit::singleline(buf)
                                            .font(theme.font(FONT_SMALL))
                                            .desired_width(edit_rect.width().max(24.0)),
                                    );
                                    if *just_started {
                                        te_resp.request_focus();
                                        *just_started = false;
                                    }
                                    let escape = ui.input(|i| i.key_pressed(Key::Escape));
                                    let enter = ui.input(|i| i.key_pressed(Key::Enter));
                                    if escape {
                                        *rename_state = None;
                                    } else if enter || te_resp.lost_focus() {
                                        let trimmed = buf.trim().to_owned();
                                        if !trimmed.is_empty() && trimmed != name {
                                            *action = Some(ShapePlusTreeAction::RenameCommit(
                                                item.id.clone(),
                                                trimmed,
                                            ));
                                        }
                                        *rename_state = None;
                                    }
                                }
                                sub_clicked = true;
                            } else {
                                let name_color = if !enabled {
                                    TEXT_DISABLED
                                } else if is_sel {
                                    TEXT_BRIGHT
                                } else {
                                    TEXT_PRIMARY
                                };
                                let name_galley = ui.painter().layout_no_wrap(
                                    name.clone(),
                                    theme.font(FONT_SMALL),
                                    name_color,
                                );
                                let name_w = name_galley.size().x;
                                let text_y = label_rect.center().y - name_galley.size().y * 0.5;
                                ui.painter().with_clip_rect(label_rect).galley(
                                    Pos2::new(label_rect.left(), text_y),
                                    name_galley,
                                    name_color,
                                );

                                let mut after_x = label_rect.left() + name_w + 6.0;
                                if is_shape_plus_paint_kind(&item.kind)
                                    && after_x + 10.0 <= label_rect.right()
                                {
                                    let swatch_rect = Rect::from_min_size(
                                        Pos2::new(
                                            after_x.round(),
                                            (row_rect.center().y - 5.0).round(),
                                        ),
                                        Vec2::splat(10.0),
                                    );
                                    paint_shape_plus_tree_swatch(
                                        ui.painter(),
                                        swatch_rect,
                                        &item.kind,
                                        &item.properties,
                                        local_frame,
                                        row_bg,
                                    );
                                    after_x += 16.0;
                                }

                                if let Some(hint) = scope_hint
                                    && after_x + 12.0 < label_rect.right()
                                {
                                    let hint_rect = Rect::from_min_max(
                                        Pos2::new(after_x, label_rect.top()),
                                        Pos2::new(label_rect.right(), label_rect.bottom()),
                                    );
                                    let hint_col = if is_sel {
                                        color32_with_alpha(TEXT_BRIGHT, 175)
                                    } else {
                                        TEXT_MUTED
                                    };
                                    ui.painter().with_clip_rect(hint_rect).text(
                                        Pos2::new(hint_rect.left(), hint_rect.center().y),
                                        Align2::LEFT_CENTER,
                                        hint,
                                        theme.font(FONT_COMPACT),
                                        hint_col,
                                    );
                                }
                            }

                            if row_resp.double_clicked() && !sub_clicked {
                                state.shape_plus_selected = Some(item.id.clone());
                                *rename_state = Some((item.id.clone(), name.clone(), true));
                            } else if row_resp.clicked() && !sub_clicked {
                                state.shape_plus_selected = Some(item.id.clone());
                            }

                            // Row right-click context menu
                            let ctx_id = item.id.clone();
                            let ctx_kind = item.kind.clone();
                            let ctx_name = name.clone();
                            row_resp.context_menu(|ui| {
                                if ui.button("Rename").clicked() {
                                    *rename_state = Some((ctx_id.clone(), ctx_name.clone(), true));
                                    ui.close();
                                }
                                if is_shape_plus_convertible_kind(&ctx_kind) {
                                    let lbl = if ctx_kind == "text" {
                                        "Create Shapes from Text"
                                    } else {
                                        "Convert to Path"
                                    };
                                    if ui.button(lbl).clicked() {
                                        *action = Some(ShapePlusTreeAction::ConvertToPath(
                                            ctx_id.clone(),
                                        ));
                                        ui.close();
                                    }
                                }
                                if ui.button("Duplicate").clicked() {
                                    *action = Some(ShapePlusTreeAction::Duplicate(ctx_id.clone()));
                                    ui.close();
                                }
                                if ui
                                    .button(if enabled { "Disable" } else { "Enable" })
                                    .clicked()
                                {
                                    *action =
                                        Some(ShapePlusTreeAction::ToggleEnabled(ctx_id.clone()));
                                    ui.close();
                                }
                                if idx > 0 && ui.button("Move Up").clicked() {
                                    *action =
                                        Some(ShapePlusTreeAction::Reorder(ctx_id.clone(), idx - 1));
                                    ui.close();
                                }
                                if idx + 1 < count && ui.button("Move Down").clicked() {
                                    *action =
                                        Some(ShapePlusTreeAction::Reorder(ctx_id.clone(), idx + 1));
                                    ui.close();
                                }
                                ui.separator();
                                if ui
                                    .button(egui::RichText::new("Delete").color(DANGER_TEXT))
                                    .clicked()
                                {
                                    *action = Some(ShapePlusTreeAction::Delete(ctx_id.clone()));
                                    ui.close();
                                }
                            });

                            if item.kind == "group"
                                && !state.shape_plus_collapsed.contains(&item.id)
                            {
                                draw_items(
                                    ui,
                                    &item.contents,
                                    depth + 1,
                                    local_frame,
                                    scope,
                                    state,
                                    rename_state,
                                    theme,
                                    action,
                                );
                            }
                        }
                    }
                    draw_items(
                        ui,
                        &doc.contents,
                        0,
                        local_frame,
                        &scope,
                        state,
                        &mut rename_state,
                        theme,
                        &mut tree_action,
                    );
                });

            if let Some(act) = tree_action {
                match act {
                    ShapePlusTreeAction::Delete(item_id) => {
                        if remove_shape_plus_item(&mut payload, &item_id).unwrap_or(false) {
                            if state.shape_plus_selected.as_deref() == Some(item_id.as_str()) {
                                state.shape_plus_selected = None;
                            }
                            payload_dirty = true;
                        }
                    }
                    ShapePlusTreeAction::Duplicate(item_id) => {
                        if let Ok(Some(new_id)) = duplicate_shape_plus_item(&mut payload, &item_id)
                        {
                            state.shape_plus_selected = Some(new_id);
                            payload_dirty = true;
                        }
                    }
                    ShapePlusTreeAction::Reorder(item_id, target_idx) => {
                        if reorder_shape_plus_item(&mut payload, &item_id, target_idx)
                            .unwrap_or(false)
                        {
                            payload_dirty = true;
                        }
                    }
                    ShapePlusTreeAction::ToggleEnabled(item_id) => {
                        if let Some(item_obj) = find_shape_item_mut(&mut payload, &item_id)
                            && let Some(props) = item_obj
                                .get_mut("properties")
                                .and_then(Value::as_object_mut)
                        {
                            let cur_on = sp_eval_f64(props.get("enabled"), local_frame, 1.0) >= 0.5;
                            let was_bool = props.get("enabled").is_some_and(Value::is_boolean);
                            let next_val = if was_bool {
                                json!(!cur_on)
                            } else {
                                json!(i32::from(!cur_on))
                            };
                            props.insert("enabled".to_owned(), next_val);
                            payload_dirty = true;
                        }
                    }
                    ShapePlusTreeAction::RenameCommit(item_id, new_name) => {
                        if let Some(item_obj) = find_shape_item_mut(&mut payload, &item_id)
                            && let Some(props) = item_obj
                                .get_mut("properties")
                                .and_then(Value::as_object_mut)
                        {
                            if let Some(existing) = props.get_mut("name") {
                                if existing.is_object() {
                                    sp_prop_set_value(existing, local_frame, json!(new_name), 0);
                                } else {
                                    *existing = json!(new_name);
                                }
                            } else {
                                props.insert("name".to_owned(), json!(new_name));
                            }
                            payload_dirty = true;
                        }
                    }
                    ShapePlusTreeAction::ConvertToPath(item_id) => {
                        pending_convert_id = Some(item_id);
                    }
                }
            }
        },
    );
    ui.data_mut(|d| d.insert_temp(rename_id, rename_state));
    paint_crisp_panel_border(ui.painter(), top_rect, BORDER_PANEL);

    // Bottom Pane: Preset Variables (if present) + Full Animated Shape+ Item Inspector
    let expr_open_id = Id::unique(("sp_expr_open_set", &clip.id));
    let mut expr_open_set: BTreeSet<(String, String)> = ui
        .data_mut(|d| d.get_temp::<BTreeSet<(String, String)>>(expr_open_id))
        .unwrap_or_default();

    let mut shape_tx_tracker = WidgetTxTracker::default();
    show_split_pane(
        ui,
        Id::unique("shape_plus_split_p1"),
        bot_rect.shrink(4.0),
        |ui, _| {
            ScrollArea::vertical()
                .id_salt((
                    "shape_plus_item_inspector",
                    state.shape_plus_selected.as_deref().unwrap_or("none"),
                ))
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let matte_active = read_clip_track_matte_mode(session, &clip.id) > 0;
                    cm3_collapsible_section(
                        ui,
                        Id::unique(("sp_track_matte_sec", &clip.id)),
                        "Track Matte",
                        matte_active,
                        theme,
                        |ui| {
                            show_clip_track_matte_rows(
                                ui,
                                session,
                                &clip.id,
                                viewport_rect,
                                state,
                                theme,
                            );
                        },
                    );
                    ui.add_space(4.0);

                    // Optional Preset Variables section
                    if payload.get("presetVariables").is_some_and(|pv| {
                        pv.as_object().is_some_and(|m| !m.is_empty())
                            || pv.as_array().is_some_and(|a| !a.is_empty())
                    }) {
                        cm3_collapsible_section(
                            ui,
                            Id::unique(("sp_preset_vars_sec", &clip.id)),
                            "Preset Variables",
                            true,
                            theme,
                            |ui| {
                                if let Some(pv) = payload.get_mut("presetVariables")
                                    && show_shape_plus_preset_variables(
                                        ui,
                                        pv,
                                        viewport_rect,
                                        theme,
                                        &mut shape_tx_tracker,
                                    )
                                {
                                    payload_dirty = true;
                                }
                            },
                        );
                        ui.add_space(4.0);
                    }

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
                    let vertex_count_hint = item_obj
                        .get("vertexIds")
                        .and_then(Value::as_array)
                        .map(Vec::len);

                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "{} Properties",
                                shape_plus_kind_title(&kind).unwrap_or(&kind)
                            ))
                            .font(theme.font(FONT_TITLE))
                            .color(TEXT_STRONG),
                        );
                        if is_shape_plus_convertible_kind(&kind) {
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    let btn_lbl = if kind == "text" {
                                        "Shapes from Text"
                                    } else {
                                        "Convert to Path"
                                    };
                                    if ui
                                        .small_button(btn_lbl)
                                        .on_hover_text(
                                            "Convert this item into editable Bezier Path geometry",
                                        )
                                        .clicked()
                                    {
                                        pending_convert_id = Some(sel_id.clone());
                                    }
                                },
                            );
                        }
                    });
                    ui.separator();

                    let Some(props) = item_obj
                        .get_mut("properties")
                        .and_then(Value::as_object_mut)
                    else {
                        return;
                    };

                    let is_stroke_kind = matches!(kind.as_str(), "stroke" | "gradientStroke");
                    let mut keys: Vec<String> = props.keys().cloned().collect();
                    keys.sort_by(|a, b| {
                        shape_plus_prop_sort_key(a)
                            .cmp(&shape_plus_prop_sort_key(b))
                            .then_with(|| a.cmp(b))
                    });

                    for key in keys {
                        if key == "enabled" {
                            continue;
                        }
                        if is_stroke_kind && is_stroke_dash_subkey(&key) {
                            continue;
                        }
                        if kind == "path" && key == "closed" && props.contains_key("path") {
                            // Rendered inline on the Path row so topology controls stay grouped.
                            continue;
                        }
                        let Some(val_snapshot) = props.get(&key).cloned() else {
                            continue;
                        };
                        let prop_kind = classify_shape_plus_prop(&kind, &key, &val_snapshot);
                        let label = humanize_prop_label(&key);

                        match prop_kind {
                            ShapePlusPropKind::Path => {
                                let closed_val =
                                    sp_eval_f64(props.get("closed"), local_frame, 1.0) >= 0.5;
                                let mut closed_mut = closed_val;
                                let flat = sp_eval_f64_vec(Some(&val_snapshot), local_frame);
                                let v_count = vertex_count_hint.unwrap_or(flat.len() / 6);
                                let mut prop_val = val_snapshot;
                                let cur_json = json!(flat);

                                if show_sp_path_row(
                                    ui,
                                    &sel_id,
                                    &key,
                                    v_count,
                                    &mut closed_mut,
                                    &mut prop_val,
                                    cur_json,
                                    local_frame,
                                    clip_start,
                                    easing_plus_enabled,
                                    viewport_rect,
                                    theme,
                                    &mut expr_open_set,
                                    &mut pending_seek_frame,
                                ) {
                                    props.insert(key, prop_val);
                                    payload_dirty = true;
                                }
                                if closed_mut != closed_val {
                                    let was_bool =
                                        props.get("closed").is_some_and(Value::is_boolean);
                                    props.insert(
                                        "closed".to_owned(),
                                        if was_bool {
                                            json!(closed_mut)
                                        } else {
                                            json!(i32::from(closed_mut))
                                        },
                                    );
                                    payload_dirty = true;
                                }
                            }
                            ShapePlusPropKind::GradientStops => {
                                let mut prop_val = val_snapshot;
                                if show_sp_gradient_stops_editor(
                                    ui,
                                    &sel_id,
                                    &key,
                                    &label,
                                    &mut prop_val,
                                    local_frame,
                                    clip_start,
                                    easing_plus_enabled,
                                    viewport_rect,
                                    theme,
                                    &mut shape_tx_tracker,
                                    &mut expr_open_set,
                                    &mut pending_seek_frame,
                                ) {
                                    props.insert(key, prop_val);
                                    payload_dirty = true;
                                }
                            }
                            ShapePlusPropKind::Dashes => {
                                let mut prop_val = val_snapshot;
                                if show_sp_dashes_array_editor(
                                    ui,
                                    &label,
                                    &mut prop_val,
                                    local_frame,
                                    theme,
                                    &mut shape_tx_tracker,
                                ) {
                                    props.insert(key, prop_val);
                                    payload_dirty = true;
                                }
                            }
                            ShapePlusPropKind::Bool => {
                                let was_bool = val_snapshot.is_boolean()
                                    || val_snapshot.get("value").is_some_and(Value::is_boolean);
                                let mut checked =
                                    sp_eval_f64(Some(&val_snapshot), local_frame, 0.0) >= 0.5;
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 6.0;
                                    ui.label(
                                        egui::RichText::new(&label)
                                            .font(theme.font(FONT_SMALL))
                                            .color(TEXT_PRIMARY),
                                    );
                                    let toggle_label = if checked { "On" } else { "Off" };
                                    if ui.checkbox(&mut checked, toggle_label).changed() {
                                        let new_val = if was_bool {
                                            json!(checked)
                                        } else {
                                            json!(i32::from(checked))
                                        };
                                        let mut prop_val = val_snapshot;
                                        sp_prop_set_value(&mut prop_val, local_frame, new_val, 0);
                                        props.insert(key, prop_val);
                                        payload_dirty = true;
                                    }
                                });
                            }
                            ShapePlusPropKind::String => {
                                let mut s = sp_eval_str(Some(&val_snapshot), local_frame, "");
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
                                            let mut prop_val = val_snapshot;
                                            sp_prop_set_value(
                                                &mut prop_val,
                                                local_frame,
                                                json!(opts[idx]),
                                                0,
                                            );
                                            props.insert(key, prop_val);
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
                                } else if key == "sourceText" {
                                    ui.label(
                                        egui::RichText::new(&label)
                                            .font(theme.font(FONT_SMALL))
                                            .color(TEXT_PRIMARY),
                                    );
                                    if ui
                                        .add(
                                            TextEdit::multiline(&mut s)
                                                .font(theme.font(FONT_SMALL))
                                                .desired_rows(2)
                                                .desired_width(ui.available_width()),
                                        )
                                        .changed()
                                    {
                                        let mut prop_val = val_snapshot;
                                        sp_prop_set_value(&mut prop_val, local_frame, json!(s), 0);
                                        props.insert(key, prop_val);
                                        payload_dirty = true;
                                    }
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
                                                    .desired_width(
                                                        (ui.available_width() - 8.0)
                                                            .clamp(80.0, 160.0),
                                                    ),
                                            )
                                            .changed()
                                        {
                                            let mut prop_val = val_snapshot;
                                            sp_prop_set_value(
                                                &mut prop_val,
                                                local_frame,
                                                json!(s),
                                                0,
                                            );
                                            props.insert(key, prop_val);
                                            payload_dirty = true;
                                        }
                                    });
                                }
                            }
                            ShapePlusPropKind::Number => {
                                let mut prop_val = val_snapshot;
                                let meta = shape_plus_number_meta(&kind, &key);
                                if show_sp_number_row(
                                    ui,
                                    &sel_id,
                                    &key,
                                    &label,
                                    meta,
                                    &mut prop_val,
                                    local_frame,
                                    clip_start,
                                    easing_plus_enabled,
                                    viewport_rect,
                                    theme,
                                    &mut shape_tx_tracker,
                                    &mut expr_open_set,
                                    &mut pending_seek_frame,
                                ) {
                                    props.insert(key, prop_val);
                                    payload_dirty = true;
                                }
                            }
                            ShapePlusPropKind::Vec2 => {
                                let mut prop_val = val_snapshot;
                                if show_sp_vec2_row(
                                    ui,
                                    &sel_id,
                                    &key,
                                    &label,
                                    &mut prop_val,
                                    local_frame,
                                    clip_start,
                                    easing_plus_enabled,
                                    viewport_rect,
                                    theme,
                                    &mut shape_tx_tracker,
                                    &mut expr_open_set,
                                    &mut pending_seek_frame,
                                ) {
                                    props.insert(key, prop_val);
                                    payload_dirty = true;
                                }
                            }
                            ShapePlusPropKind::Rgb => {
                                let mut prop_val = val_snapshot;
                                if show_sp_rgb_row(
                                    ui,
                                    &sel_id,
                                    &key,
                                    &label,
                                    &mut prop_val,
                                    local_frame,
                                    clip_start,
                                    easing_plus_enabled,
                                    viewport_rect,
                                    theme,
                                    &mut shape_tx_tracker,
                                    &mut expr_open_set,
                                    &mut pending_seek_frame,
                                ) {
                                    props.insert(key, prop_val);
                                    payload_dirty = true;
                                }
                            }
                            ShapePlusPropKind::Unknown => {}
                        }
                    }

                    // Collapsible Dashes section on Stroke / Gradient Stroke
                    if is_stroke_kind {
                        let has_dashes = sp_stroke_has_active_dashes(props, local_frame);
                        ui.add_space(4.0);
                        cm3_collapsible_section(
                            ui,
                            Id::unique(("sp_stroke_dashes_sec", &sel_id)),
                            "Dashes",
                            has_dashes,
                            theme,
                            |ui| {
                                if show_sp_stroke_dashes_section(
                                    ui,
                                    &sel_id,
                                    props,
                                    local_frame,
                                    clip_start,
                                    easing_plus_enabled,
                                    viewport_rect,
                                    theme,
                                    &mut shape_tx_tracker,
                                    &mut expr_open_set,
                                    &mut pending_seek_frame,
                                ) {
                                    payload_dirty = true;
                                }
                            },
                        );
                    }
                });
        },
    );
    ui.data_mut(|d| d.insert_temp(expr_open_id, expr_open_set));
    paint_crisp_panel_border(ui.painter(), bot_rect, BORDER_PANEL);

    // Execute deferred Convert to Path with optional project font asset bytes
    if let Some(convert_id) = pending_convert_id {
        let (primary_font, fallback_font) =
            resolve_shape_plus_item_font_bytes(session, &payload, &convert_id, local_frame);
        match zc_render::shape_plus::convert_shape_plus_item_to_path(
            &mut payload,
            &convert_id,
            local_frame,
            primary_font,
            fallback_font,
        ) {
            Ok(new_ids) => {
                if let Some(first_id) = new_ids.first() {
                    state.shape_plus_selected = Some(first_id.clone());
                }
                payload_dirty = true;
                let msg = if new_ids.len() == 1 {
                    "Converted item to Path".to_owned()
                } else {
                    format!("Converted item to {} Path items", new_ids.len())
                };
                actions.push(PanelAction::StatusMessage(msg));
            }
            Err(err) => {
                actions.push(PanelAction::StatusMessage(format!(
                    "Convert to Path failed: {err}"
                )));
            }
        }
    }

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

    if let Some(target_frame) = pending_seek_frame {
        let _ = session.execute(Command::SetFrame {
            frame: target_frame,
        });
    }
}

#[derive(Debug, Clone)]
enum ShapePlusTreeAction {
    Delete(String),
    Duplicate(String),
    Reorder(String, usize),
    ToggleEnabled(String),
    RenameCommit(String, String),
    ConvertToPath(String),
}

// ---------------------------------------------------------------------------
// Shape+ Paint Scope, Swatches, Keyframe/Expression & Editor Helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ShapePlusScopeEntry {
    pub enabled: bool,
    pub painted: bool,
    pub paths: usize,
    pub texts: usize,
}

pub(super) fn is_shape_plus_geometry_kind(kind: &str) -> bool {
    matches!(kind, "rectangle" | "ellipse" | "polystar" | "path" | "text")
}

pub(super) fn is_shape_plus_paint_kind(kind: &str) -> bool {
    matches!(kind, "fill" | "stroke" | "gradientFill" | "gradientStroke")
}

pub(super) fn is_shape_plus_convertible_kind(kind: &str) -> bool {
    matches!(kind, "rectangle" | "ellipse" | "polystar" | "text")
}

fn sp_item_enabled(item: &zc_render::shape_plus::ShapePlusItem, frame: f64) -> bool {
    sp_eval_f64(item.properties.get("enabled"), frame, 1.0) >= 0.5
}

fn count_enabled_texts(items: &[zc_render::shape_plus::ShapePlusItem], frame: f64) -> usize {
    let mut count = 0usize;
    for item in items {
        if !sp_item_enabled(item, frame) {
            continue;
        }
        if item.kind == "text" {
            count += 1;
        } else if item.kind == "group" {
            count += count_enabled_texts(&item.contents, frame);
        }
    }
    count
}

/// Computes per-group paint & text-animator scope matching Zoidium `contents-ops.js` `paintScope`.
pub(super) fn compute_shape_plus_paint_scope(
    items: &[zc_render::shape_plus::ShapePlusItem],
    frame: f64,
) -> BTreeMap<String, ShapePlusScopeEntry> {
    fn walk(
        list: &[zc_render::shape_plus::ShapePlusItem],
        parent_enabled: bool,
        frame: f64,
        out: &mut BTreeMap<String, ShapePlusScopeEntry>,
    ) -> Vec<String> {
        let mut above: Vec<String> = Vec::new();
        let mut texts = 0usize;
        for item in list {
            let on = parent_enabled && sp_item_enabled(item, frame);
            if item.kind == "group" {
                if on {
                    texts += count_enabled_texts(&item.contents, frame);
                }
                let child_above = walk(&item.contents, on, frame, out);
                above.extend(child_above);
            } else if is_shape_plus_geometry_kind(&item.kind) {
                if item.kind == "text" && on {
                    texts += 1;
                }
                out.insert(
                    item.id.clone(),
                    ShapePlusScopeEntry {
                        enabled: on,
                        painted: false,
                        paths: 0,
                        texts: 0,
                    },
                );
                if on {
                    above.push(item.id.clone());
                }
            } else if item.kind == "textAnimator" {
                out.insert(
                    item.id.clone(),
                    ShapePlusScopeEntry {
                        enabled: on,
                        painted: false,
                        paths: 0,
                        texts: if on { texts } else { 0 },
                    },
                );
            } else if is_shape_plus_paint_kind(&item.kind) {
                out.insert(
                    item.id.clone(),
                    ShapePlusScopeEntry {
                        enabled: on,
                        painted: false,
                        paths: if on { above.len() } else { 0 },
                        texts: 0,
                    },
                );
                if on {
                    for gid in &above {
                        if let Some(entry) = out.get_mut(gid) {
                            entry.painted = true;
                        }
                    }
                }
            }
        }
        above
    }

    let mut out = BTreeMap::new();
    let _ = walk(items, true, frame, &mut out);
    out
}

/// Returns the dim tree scope hint (`"no fill or stroke"`, `"no paths above"`, or `"no text above"`).
pub(super) fn shape_plus_scope_hint(
    scope: &BTreeMap<String, ShapePlusScopeEntry>,
    item: &zc_render::shape_plus::ShapePlusItem,
) -> Option<&'static str> {
    let entry = scope.get(&item.id)?;
    if !entry.enabled {
        return None;
    }
    if item.kind == "textAnimator" {
        return (entry.texts == 0).then_some("no text above");
    }
    if is_shape_plus_geometry_kind(&item.kind) {
        return (!entry.painted).then_some("no fill or stroke");
    }
    if is_shape_plus_paint_kind(&item.kind) {
        return (entry.paths == 0).then_some("no paths above");
    }
    None
}

fn paint_shape_plus_tree_swatch(
    painter: &Painter,
    rect: Rect,
    kind: &str,
    props: &serde_json::Map<String, Value>,
    frame: f64,
    row_bg: Color32,
) {
    match kind {
        "fill" => {
            let rgb = sp_eval_rgb(props.get("color"), frame, [1.0, 1.0, 1.0]);
            let col = sp_rgb_to_color32(rgb);
            painter.rect_filled(rect, CornerRadius::same(2), col);
            painter.rect_stroke(
                rect,
                CornerRadius::same(2),
                Stroke::new(1.0, Color32::BLACK),
                StrokeKind::Inside,
            );
        }
        "stroke" => {
            let rgb = sp_eval_rgb(props.get("color"), frame, [1.0, 1.0, 1.0]);
            let col = sp_rgb_to_color32(rgb);
            painter.rect_stroke(
                rect.shrink(1.0),
                CornerRadius::same(1),
                Stroke::new(2.5, col),
                StrokeKind::Inside,
            );
            painter.rect_stroke(
                rect,
                CornerRadius::same(2),
                Stroke::new(1.0, Color32::BLACK),
                StrokeKind::Inside,
            );
        }
        "gradientFill" | "gradientStroke" => {
            let stops = sp_eval_gradient_stops(props.get("stops"), frame);
            painter.rect_filled(rect, CornerRadius::ZERO, BG_SUNKEN);
            paint_gradient_mesh(painter, rect, &stops, 8);
            if kind == "gradientStroke" {
                let inner = rect.shrink(3.0);
                painter.rect_filled(inner, CornerRadius::ZERO, row_bg);
                painter.rect_stroke(
                    inner,
                    CornerRadius::ZERO,
                    Stroke::new(0.75, Color32::BLACK),
                    StrokeKind::Outside,
                );
            }
            painter.rect_stroke(
                rect,
                CornerRadius::ZERO,
                Stroke::new(1.0, Color32::BLACK),
                StrokeKind::Inside,
            );
        }
        _ => {}
    }
}

fn sp_rgb_to_color32(rgb: [f64; 3]) -> Color32 {
    Color32::from_rgb(
        (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
    )
}

fn sp_rgba_to_color32(rgba: [f64; 4]) -> Color32 {
    Color32::from_rgba_unmultiplied(
        (rgba[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgba[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgba[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgba[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    )
}

fn sp_eval_f64(val: Option<&Value>, frame: f64, default: f64) -> f64 {
    zc_render::shape_plus::eval_prop_f64(val, frame, default, None)
}

fn sp_eval_segment_factor(k0: &Value, k1: &Value, frame: f64) -> Option<f64> {
    let f0 = k0.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
    let f1 = k1.get("frame").and_then(Value::as_f64).unwrap_or(f0);
    let span = f1 - f0;
    if span <= 1e-12 || frame <= f0 {
        return Some(0.0);
    }
    if frame >= f1 {
        return Some(1.0);
    }
    let tween_code = k0
        .get("tween")
        .and_then(Value::as_u64)
        .map_or(1_u16, |t| t.min(u64::from(u16::MAX)) as u16);
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(tween_code);
    if mode == InterpolationMode::Discrete || easing == zc_anim::EasingKind::Hold {
        return Some(0.0);
    }
    let u = ((frame - f0) / span).clamp(0.0, 1.0);
    if let Some(arr) = k0
        .get("bezier")
        .or_else(|| k0.get("controlPoints"))
        .and_then(Value::as_array)
        && arr.len() == 4
        && let (Some(x1), Some(y1), Some(x2), Some(y2)) = (
            arr[0].as_f64(),
            arr[1].as_f64(),
            arr[2].as_f64(),
            arr[3].as_f64(),
        )
    {
        return Some(zc_anim::eval_cubic_bezier(x1, y1, x2, y2, u));
    }
    if mode == InterpolationMode::Bezier {
        return None;
    }
    Some(eval_easing(easing, u))
}

fn sp_eval_segment_scalar(k0: &Value, v0: f64, k1: &Value, v1: f64, frame: f64) -> f64 {
    if let Some(e) = sp_eval_segment_factor(k0, k1, frame) {
        return v0 + (v1 - v0) * e;
    }
    let f0 = k0.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
    let f1 = k1.get("frame").and_then(Value::as_f64).unwrap_or(f0);
    let span = (f1 - f0).max(1e-12);
    let out_t = k0
        .get("outTangent")
        .and_then(Value::as_array)
        .filter(|a| a.len() >= 2)
        .and_then(|a| Some([a[0].as_f64()?, a[1].as_f64()?]));
    let in_t = k1
        .get("inTangent")
        .and_then(Value::as_array)
        .filter(|a| a.len() >= 2)
        .and_then(|a| Some([a[0].as_f64()?, a[1].as_f64()?]));
    if out_t.is_some() || in_t.is_some() {
        return eval_segment_bezier(
            BezierSegment {
                start_frame: f0,
                start_value: v0,
                out_tangent: out_t.unwrap_or([span / 3.0, (v1 - v0) / 3.0]),
                end_frame: f1,
                end_value: v1,
                in_tangent: in_t.unwrap_or([-span / 3.0, -(v1 - v0) / 3.0]),
            },
            frame,
        );
    }
    let tween_code = k0
        .get("tween")
        .and_then(Value::as_u64)
        .map_or(1_u16, |t| t.min(u64::from(u16::MAX)) as u16);
    let (_, easing) = InterpolationMode::from_cm3_tween_code(tween_code);
    let u = ((frame - f0) / span).clamp(0.0, 1.0);
    v0 + (v1 - v0) * eval_easing(easing, u)
}

fn sp_eval_f64_vec(val: Option<&Value>, frame: f64) -> Vec<f64> {
    let Some(v) = val else {
        return Vec::new();
    };
    if let Some(arr) = v.as_array() {
        return arr
            .iter()
            .map(|x| x.as_f64().filter(|n| n.is_finite()).unwrap_or(0.0))
            .collect();
    }
    if let Some(objs) = v.get("objects").and_then(Value::as_array) {
        return objs
            .iter()
            .map(|o| sp_eval_f64(Some(o), frame, 0.0))
            .collect();
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        let mut sorted: Vec<(f64, Vec<f64>, &Value)> = kfs
            .iter()
            .filter_map(|k| {
                let f = k.get("frame").and_then(Value::as_f64)?;
                let arr = k.get("value").and_then(Value::as_array)?;
                let vals = arr
                    .iter()
                    .map(|x| x.as_f64().filter(|n| n.is_finite()).unwrap_or(0.0))
                    .collect();
                Some((f, vals, k))
            })
            .collect();
        if !sorted.is_empty() {
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
            if frame <= sorted[0].0 {
                return sorted[0].1.clone();
            }
            if frame >= sorted[sorted.len() - 1].0 {
                return sorted[sorted.len() - 1].1.clone();
            }
            for w in sorted.windows(2) {
                if frame <= w[1].0 {
                    let len = w[0].1.len().min(w[1].1.len());
                    return (0..len)
                        .map(|i| {
                            sp_eval_segment_scalar(w[0].2, w[0].1[i], w[1].2, w[1].1[i], frame)
                        })
                        .collect();
                }
            }
        }
    }
    if let Some(inner) = v.get("value") {
        return sp_eval_f64_vec(Some(inner), frame);
    }
    Vec::new()
}

fn sp_eval_vec2(val: Option<&Value>, frame: f64, default: [f64; 2]) -> [f64; 2] {
    let arr = sp_eval_f64_vec(val, frame);
    if arr.len() >= 2 {
        [arr[0], arr[1]]
    } else {
        default
    }
}

fn sp_eval_rgb(val: Option<&Value>, frame: f64, default: [f64; 3]) -> [f64; 3] {
    let Some(v) = val else {
        return default;
    };
    if let Some(s) = v.as_str() {
        let c = sp_parse_css_rgba(s);
        return [c[0], c[1], c[2]];
    }
    let arr = sp_eval_f64_vec(Some(v), frame);
    if arr.len() >= 3 {
        let scale = if arr[0] > 1.0 || arr[1] > 1.0 || arr[2] > 1.0 {
            1.0 / 255.0
        } else {
            1.0
        };
        return [
            (arr[0] * scale).clamp(0.0, 1.0),
            (arr[1] * scale).clamp(0.0, 1.0),
            (arr[2] * scale).clamp(0.0, 1.0),
        ];
    }
    if let Some(inner) = v.get("value") {
        return sp_eval_rgb(Some(inner), frame, default);
    }
    default
}

fn sp_eval_str(val: Option<&Value>, frame: f64, default: &str) -> String {
    let Some(v) = val else {
        return default.to_owned();
    };
    if let Some(s) = v.as_str() {
        return s.to_owned();
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        let mut pts: Vec<(f64, &str)> = kfs
            .iter()
            .filter_map(|k| {
                let f = k.get("frame").and_then(Value::as_f64)?;
                let s = k.get("value").and_then(Value::as_str)?;
                Some((f, s))
            })
            .collect();
        if !pts.is_empty() {
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut prev = pts[0].1;
            for (kf_f, s) in pts {
                if kf_f > frame {
                    break;
                }
                prev = s;
            }
            return prev.to_owned();
        }
    }
    if let Some(inner) = v.get("value") {
        return sp_eval_str(Some(inner), frame, default);
    }
    default.to_owned()
}

fn sp_parse_css_rgba(css: &str) -> [f64; 4] {
    let trimmed = css.trim();
    if let Ok(c) = Rgba::from_hex(trimmed) {
        return [
            f64::from(c.0[0]),
            f64::from(c.0[1]),
            f64::from(c.0[2]),
            f64::from(c.0[3]),
        ];
    }
    if let Some(inner) = trimmed
        .strip_prefix("rgba(")
        .or_else(|| trimmed.strip_prefix("rgb("))
        .and_then(|s| s.strip_suffix(')'))
    {
        let parts: Vec<f64> = inner
            .split(',')
            .filter_map(|p| p.trim().parse::<f64>().ok())
            .collect();
        if parts.len() >= 3 {
            let scale = if parts[0] > 1.0 || parts[1] > 1.0 || parts[2] > 1.0 {
                1.0 / 255.0
            } else {
                1.0
            };
            return [
                (parts[0] * scale).clamp(0.0, 1.0),
                (parts[1] * scale).clamp(0.0, 1.0),
                (parts[2] * scale).clamp(0.0, 1.0),
                parts.get(3).copied().unwrap_or(1.0).clamp(0.0, 1.0),
            ];
        }
    }
    [1.0, 1.0, 1.0, 1.0]
}

fn sp_format_css_rgba(rgba: [f64; 4]) -> String {
    let r = (rgba[0].clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (rgba[1].clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (rgba[2].clamp(0.0, 1.0) * 255.0).round() as u8;
    let a = rgba[3].clamp(0.0, 1.0);
    if (a - 1.0).abs() < 1e-4 {
        format!("rgba({r},{g},{b},1)")
    } else {
        format!("rgba({r},{g},{b},{a:.3})")
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SpGradientStop {
    position: f64,
    color: [f64; 4],
}

fn sp_default_gradient_stops() -> Vec<SpGradientStop> {
    vec![
        SpGradientStop {
            position: 0.0,
            color: [1.0, 77.0 / 255.0, 26.0 / 255.0, 1.0],
        },
        SpGradientStop {
            position: 1.0,
            color: [26.0 / 255.0, 102.0 / 255.0, 1.0, 1.0],
        },
    ]
}

fn sp_parse_gradient_stops_array(arr: &[Value]) -> Vec<SpGradientStop> {
    let mut out = Vec::new();
    for item in arr.iter().take(16) {
        let (pos, col_val) = if let Some(pair) = item.as_array()
            && pair.len() >= 2
        {
            (pair[0].as_f64(), Some(&pair[1]))
        } else if let Some(obj) = item.as_object() {
            (
                obj.get("position")
                    .or_else(|| obj.get("offset"))
                    .and_then(Value::as_f64),
                obj.get("color"),
            )
        } else {
            (None, None)
        };
        let (Some(p), Some(cv)) = (pos, col_val) else {
            continue;
        };
        if !p.is_finite() {
            continue;
        }
        let rgba = if let Some(s) = cv.as_str() {
            Some(sp_parse_css_rgba(s))
        } else if let Some(c_arr) = cv.as_array()
            && c_arr.len() >= 3
        {
            let r = c_arr[0].as_f64().unwrap_or(0.0);
            let g = c_arr[1].as_f64().unwrap_or(0.0);
            let b = c_arr[2].as_f64().unwrap_or(0.0);
            let a = c_arr.get(3).and_then(Value::as_f64).unwrap_or(1.0);
            let scale = if r > 1.0 || g > 1.0 || b > 1.0 {
                1.0 / 255.0
            } else {
                1.0
            };
            Some([
                (r * scale).clamp(0.0, 1.0),
                (g * scale).clamp(0.0, 1.0),
                (b * scale).clamp(0.0, 1.0),
                a.clamp(0.0, 1.0),
            ])
        } else {
            None
        };
        if let Some(color) = rgba {
            out.push(SpGradientStop {
                position: p.clamp(0.0, 1.0),
                color,
            });
        }
    }
    out.sort_by(|a, b| a.position.total_cmp(&b.position));
    out
}

fn sp_eval_gradient_stops(val: Option<&Value>, frame: f64) -> Vec<SpGradientStop> {
    let Some(v) = val else {
        return sp_default_gradient_stops();
    };
    if let Some(arr) = v.as_array() {
        let parsed = sp_parse_gradient_stops_array(arr);
        return if parsed.len() >= 2 {
            parsed
        } else {
            sp_default_gradient_stops()
        };
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        let mut sorted: Vec<(f64, Vec<SpGradientStop>, &Value)> = kfs
            .iter()
            .filter_map(|k| {
                let f = k.get("frame").and_then(Value::as_f64)?;
                let arr = k.get("value").and_then(Value::as_array)?;
                let s = sp_parse_gradient_stops_array(arr);
                (s.len() >= 2).then_some((f, s, k))
            })
            .collect();
        if !sorted.is_empty() {
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
            if frame <= sorted[0].0 {
                return sorted[0].1.clone();
            }
            if frame >= sorted[sorted.len() - 1].0 {
                return sorted[sorted.len() - 1].1.clone();
            }
            for w in sorted.windows(2) {
                if frame <= w[1].0 {
                    let t = sp_eval_segment_factor(w[0].2, w[1].2, frame).unwrap_or_else(|| {
                        let span = (w[1].0 - w[0].0).max(1e-9);
                        ((frame - w[0].0) / span).clamp(0.0, 1.0)
                    });
                    let len = w[0].1.len().min(w[1].1.len());
                    return (0..len)
                        .map(|i| SpGradientStop {
                            position: w[0].1[i].position
                                + (w[1].1[i].position - w[0].1[i].position) * t,
                            color: [
                                w[0].1[i].color[0] + (w[1].1[i].color[0] - w[0].1[i].color[0]) * t,
                                w[0].1[i].color[1] + (w[1].1[i].color[1] - w[0].1[i].color[1]) * t,
                                w[0].1[i].color[2] + (w[1].1[i].color[2] - w[0].1[i].color[2]) * t,
                                w[0].1[i].color[3] + (w[1].1[i].color[3] - w[0].1[i].color[3]) * t,
                            ],
                        })
                        .collect();
                }
            }
        }
    }
    if let Some(inner) = v.get("value") {
        return sp_eval_gradient_stops(Some(inner), frame);
    }
    sp_default_gradient_stops()
}

fn sp_stops_to_json(stops: &[SpGradientStop]) -> Value {
    let arr: Vec<Value> = stops
        .iter()
        .map(|s| {
            json!({
                "position": s.position.clamp(0.0, 1.0),
                "color": sp_format_css_rgba(s.color),
            })
        })
        .collect();
    Value::Array(arr)
}

fn sp_sample_gradient(stops: &[SpGradientStop], t: f64) -> [f64; 4] {
    if stops.is_empty() {
        return [1.0, 1.0, 1.0, 1.0];
    }
    let mut a = &stops[0];
    let mut b = stops.last().unwrap_or(a);
    for w in stops.windows(2) {
        if t <= w[1].position {
            a = &w[0];
            b = &w[1];
            break;
        }
    }
    let f = ((t - a.position) / (b.position - a.position).max(1e-9)).clamp(0.0, 1.0);
    [
        a.color[0] + (b.color[0] - a.color[0]) * f,
        a.color[1] + (b.color[1] - a.color[1]) * f,
        a.color[2] + (b.color[2] - a.color[2]) * f,
        a.color[3] + (b.color[3] - a.color[3]) * f,
    ]
}

fn paint_gradient_mesh(painter: &Painter, rect: Rect, stops: &[SpGradientStop], segs: usize) {
    let n = segs.max(2);
    let mut mesh = Mesh::default();
    for i in 0..n {
        let t0 = i as f64 / n as f64;
        let t1 = (i + 1) as f64 / n as f64;
        let c0 = sp_rgba_to_color32(sp_sample_gradient(stops, t0));
        let c1 = sp_rgba_to_color32(sp_sample_gradient(stops, t1));
        let x0 = rect.left() + rect.width() * (t0 as f32);
        let x1 = rect.left() + rect.width() * (t1 as f32);
        let tl = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(x0, rect.top()), c0);
        let tr = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(x1, rect.top()), c1);
        let br = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(x1, rect.bottom()), c1);
        let bl = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(x0, rect.bottom()), c0);
        mesh.add_triangle(tl, tr, br);
        mesh.add_triangle(tl, br, bl);
    }
    painter.add(Shape::mesh(mesh));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ShapePlusPropKind {
    String,
    Bool,
    Number,
    Vec2,
    Rgb,
    GradientStops,
    Dashes,
    Path,
    Unknown,
}

fn is_known_bool_key(key: &str) -> bool {
    matches!(
        key,
        "closed"
            | "enabled"
            | "reverse"
            | "clip"
            | "wigglyOn"
            | "lockDimensions"
            | "selectorRandomize"
    ) || key.starts_with("enabled_")
}

pub(super) fn classify_shape_plus_prop(kind: &str, key: &str, val: &Value) -> ShapePlusPropKind {
    if key == "path" && kind == "path" {
        return ShapePlusPropKind::Path;
    }
    if key == "stops" {
        return ShapePlusPropKind::GradientStops;
    }
    if key == "dashes" {
        return ShapePlusPropKind::Dashes;
    }
    if matches!(key, "color" | "fillColor" | "strokeColor") {
        return ShapePlusPropKind::Rgb;
    }
    if val.is_boolean() {
        return ShapePlusPropKind::Bool;
    }
    if is_known_bool_key(key) {
        return ShapePlusPropKind::Bool;
    }
    if let Some(obj) = val.as_object() {
        if let Some(objs) = obj.get("objects").and_then(Value::as_array) {
            return match objs.len() {
                1 => ShapePlusPropKind::Number,
                2 => ShapePlusPropKind::Vec2,
                3 => ShapePlusPropKind::Rgb,
                _ => ShapePlusPropKind::Unknown,
            };
        }
        if let Some(inner) = obj.get("value") {
            return classify_shape_plus_prop(kind, key, inner);
        }
        if let Some(first_val) = obj
            .get("keyframes")
            .and_then(Value::as_array)
            .and_then(|kfs| kfs.first())
            .and_then(|k| k.get("value"))
        {
            return classify_shape_plus_prop(kind, key, first_val);
        }
    }
    match val {
        Value::String(_) => ShapePlusPropKind::String,
        Value::Bool(_) => ShapePlusPropKind::Bool,
        Value::Number(_) => ShapePlusPropKind::Number,
        Value::Array(arr) if arr.len() == 2 && arr.iter().all(Value::is_number) => {
            ShapePlusPropKind::Vec2
        }
        Value::Array(arr) if arr.len() == 3 && arr.iter().all(Value::is_number) => {
            ShapePlusPropKind::Rgb
        }
        _ => ShapePlusPropKind::Unknown,
    }
}

pub(super) fn sp_prop_is_animated(val: &Value) -> bool {
    let Some(obj) = val.as_object() else {
        return false;
    };
    if let Some(b) = obj.get("animated").and_then(Value::as_bool) {
        return b;
    }
    if let Some(kfs) = obj.get("keyframes").and_then(Value::as_array) {
        return !kfs.is_empty();
    }
    if let Some(objs) = obj.get("objects").and_then(Value::as_array) {
        return objs.iter().any(sp_prop_is_animated);
    }
    false
}

pub(super) fn sp_prop_expression(val: &Value) -> Option<&str> {
    val.as_object()
        .and_then(|o| o.get("expression"))
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
}

pub(super) fn sp_prop_keyframe_frames(val: &Value) -> Vec<f64> {
    let Some(obj) = val.as_object() else {
        return Vec::new();
    };
    let mut frames: Vec<f64> = Vec::new();
    if let Some(kfs) = obj.get("keyframes").and_then(Value::as_array) {
        for kf in kfs {
            if let Some(f) = kf.get("frame").and_then(Value::as_f64)
                && f.is_finite()
            {
                frames.push(f);
            }
        }
    } else if let Some(objs) = obj.get("objects").and_then(Value::as_array) {
        for sub in objs {
            frames.extend(sp_prop_keyframe_frames(sub));
        }
    }
    frames.sort_by(f64::total_cmp);
    frames.dedup_by(|a, b| (*b - *a).abs() < 1e-4);
    frames
}

fn sp_prop_keyframe_tween_at(val: &Value, frame: f64) -> Option<u16> {
    let obj = val.as_object()?;
    if let Some(kfs) = obj.get("keyframes").and_then(Value::as_array) {
        for kf in kfs {
            if let Some(f) = kf.get("frame").and_then(Value::as_f64)
                && (f - frame).abs() < 1e-4
            {
                return Some(
                    kf.get("tween")
                        .and_then(Value::as_u64)
                        .map_or(1_u16, |t| t.min(u64::from(u16::MAX)) as u16),
                );
            }
        }
    } else if let Some(objs) = obj.get("objects").and_then(Value::as_array) {
        for sub in objs {
            if let Some(tw) = sp_prop_keyframe_tween_at(sub, frame) {
                return Some(tw);
            }
        }
    }
    None
}

/// Normalizes a raw value or CM3 `{ objects: [...] }` dynamic group into canonical
/// `{ value, animated, keyframes, [expression] }` object form, preserving any keyframes.
fn sp_ensure_prop_object<'a>(
    val: &'a mut Value,
    current_val: &Value,
    default_tween: u16,
) -> &'a mut serde_json::Map<String, Value> {
    if val.is_object() && val.get("objects").is_some() {
        // Convert CM3 component `objects` array losslessly into combined vector keyframes
        let frames = sp_prop_keyframe_frames(val);
        let animated = sp_prop_is_animated(val);
        let expr = sp_prop_expression(val).map(str::to_owned);
        let mut kfs = Vec::new();
        for f in frames {
            let sample = sp_eval_f64_vec(Some(val), f);
            let tw = sp_prop_keyframe_tween_at(val, f).unwrap_or(default_tween);
            kfs.push(json!({
                "frame": f,
                "value": sample,
                "tween": tw,
            }));
        }
        let mut map = serde_json::Map::new();
        map.insert("value".to_owned(), current_val.clone());
        map.insert("animated".to_owned(), json!(animated));
        map.insert("keyframes".to_owned(), Value::Array(kfs));
        if let Some(e) = expr {
            map.insert("expression".to_owned(), json!(e));
        }
        *val = Value::Object(map);
    } else if !val.is_object() {
        *val = json!({
            "value": val.clone(),
            "animated": false,
            "keyframes": [],
        });
    }
    val.as_object_mut().expect("ensured object")
}

fn sp_sort_keyframes(kfs: &mut [Value]) {
    kfs.sort_by(|a, b| {
        let fa = a.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
        let fb = b.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
        fa.total_cmp(&fb)
    });
}

/// Toggles `animated` on a Shape+ property JSON value at `frame`.
/// Turning ON inserts a keyframe at `frame`; turning OFF bakes `current_val` into `"value"`
/// and clears `"keyframes"` so `eval_prop_*` evaluates the static value cleanly.
pub(super) fn sp_prop_toggle_animated(
    val: &mut Value,
    frame: f64,
    current_val: Value,
    default_tween: u16,
) {
    let was_animated = sp_prop_is_animated(val);
    let obj = sp_ensure_prop_object(val, &current_val, default_tween);
    if was_animated {
        obj.insert("animated".to_owned(), json!(false));
        obj.insert("value".to_owned(), current_val);
        obj.insert("keyframes".to_owned(), json!([]));
    } else {
        obj.insert("animated".to_owned(), json!(true));
        obj.insert("value".to_owned(), current_val.clone());
        obj.insert(
            "keyframes".to_owned(),
            json!([{ "frame": frame, "value": current_val, "tween": default_tween }]),
        );
    }
}

/// Writes an edited value to a Shape+ property JSON value:
/// - When `animated: true`: updates or inserts a keyframe at `frame` (keeping keyframes sorted).
/// - When `animated: false`: updates the static value cleanly (clearing stale keyframes if wrapped).
pub(super) fn sp_prop_set_value(val: &mut Value, frame: f64, new_val: Value, default_tween: u16) {
    if sp_prop_is_animated(val) {
        let obj = sp_ensure_prop_object(val, &new_val, default_tween);
        obj.insert("animated".to_owned(), json!(true));
        obj.insert("value".to_owned(), new_val.clone());
        let kfs = obj
            .entry("keyframes".to_owned())
            .or_insert_with(|| json!([]))
            .as_array_mut();
        if let Some(arr) = kfs {
            if let Some(existing) = arr.iter_mut().find(|k| {
                k.get("frame")
                    .and_then(Value::as_f64)
                    .is_some_and(|f| (f - frame).abs() < 1e-4)
            }) {
                if let Some(kobj) = existing.as_object_mut() {
                    kobj.insert("value".to_owned(), new_val);
                }
            } else {
                arr.push(json!({
                    "frame": frame,
                    "value": new_val,
                    "tween": default_tween,
                }));
                sp_sort_keyframes(arr);
            }
        }
    } else if val.is_object() {
        let obj = sp_ensure_prop_object(val, &new_val, default_tween);
        obj.insert("animated".to_owned(), json!(false));
        obj.insert("value".to_owned(), new_val);
        obj.insert("keyframes".to_owned(), json!([]));
    } else {
        *val = new_val;
    }
}

/// Inserts or removes a keyframe at `frame` on an animated Shape+ property.
pub(super) fn sp_prop_toggle_keyframe(
    val: &mut Value,
    frame: f64,
    current_val: Value,
    default_tween: u16,
) {
    let obj = sp_ensure_prop_object(val, &current_val, default_tween);
    let kfs = obj
        .entry("keyframes".to_owned())
        .or_insert_with(|| json!([]))
        .as_array_mut();
    let Some(arr) = kfs else {
        return;
    };
    if let Some(pos) = arr.iter().position(|k| {
        k.get("frame")
            .and_then(Value::as_f64)
            .is_some_and(|f| (f - frame).abs() < 1e-4)
    }) {
        arr.remove(pos);
        if arr.is_empty() {
            obj.insert("animated".to_owned(), json!(false));
            obj.insert("value".to_owned(), current_val);
        }
    } else {
        arr.push(json!({
            "frame": frame,
            "value": current_val.clone(),
            "tween": default_tween,
        }));
        sp_sort_keyframes(arr);
        obj.insert("animated".to_owned(), json!(true));
        obj.insert("value".to_owned(), current_val);
    }
}

/// Updates the `tween` field of the keyframe at `frame` (if present).
pub(super) fn sp_prop_set_keyframe_tween(val: &mut Value, frame: f64, new_tween: u16) -> bool {
    let Some(arr) = val
        .as_object_mut()
        .and_then(|o| o.get_mut("keyframes"))
        .and_then(Value::as_array_mut)
    else {
        return false;
    };
    if let Some(kf) = arr.iter_mut().find(|k| {
        k.get("frame")
            .and_then(Value::as_f64)
            .is_some_and(|f| (f - frame).abs() < 1e-4)
    }) && let Some(kobj) = kf.as_object_mut()
    {
        kobj.insert("tween".to_owned(), json!(new_tween));
        return true;
    }
    false
}

/// Sets or clears `"expression"` on a Shape+ property JSON value.
pub(super) fn sp_prop_set_expression(val: &mut Value, current_val: Value, expr: Option<&str>) {
    match expr.filter(|s| !s.trim().is_empty()) {
        Some(src) => {
            let obj = sp_ensure_prop_object(val, &current_val, 1);
            obj.insert("expression".to_owned(), json!(src));
        }
        None => {
            if let Some(obj) = val.as_object_mut() {
                obj.remove("expression");
                let animated = obj
                    .get("animated")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let has_kfs = obj
                    .get("keyframes")
                    .and_then(Value::as_array)
                    .is_some_and(|a| !a.is_empty());
                if !animated
                    && !has_kfs
                    && obj.get("objects").is_none()
                    && let Some(inner) = obj.get("value").cloned()
                {
                    *val = inner;
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SpNumberMeta {
    /// Stored in radians internally (`geometry.rs`/`raster.rs`), displayed in degrees (`°`).
    is_radians: bool,
    suffix: &'static str,
    speed: f64,
    range: Option<(f64, f64)>,
}

fn shape_plus_number_meta(kind: &str, key: &str) -> SpNumberMeta {
    match (kind, key) {
        (_, "rotation" | "angle" | "twistAngle" | "skew" | "skewAxis" | "highlightAngle") => {
            SpNumberMeta {
                is_radians: true,
                suffix: "°",
                speed: 1.0,
                range: None,
            }
        }
        ("trim", "offset") | (_, "temporalPhase" | "spatialPhase" | "wavePhase") => SpNumberMeta {
            is_radians: false,
            suffix: "°",
            speed: 1.0,
            range: None,
        },
        (
            "group" | "fill" | "stroke" | "gradientFill" | "gradientStroke" | "repeater",
            "opacity" | "startOpacity" | "endOpacity",
        ) => SpNumberMeta {
            is_radians: false,
            suffix: "",
            speed: 0.01,
            range: Some((0.0, 1.0)),
        },
        ("trim", "start" | "end") => SpNumberMeta {
            is_radians: false,
            suffix: "%",
            speed: 0.5,
            range: Some((0.0, 100.0)),
        },
        ("puckerBloat", "amount") => SpNumberMeta {
            is_radians: false,
            suffix: "%",
            speed: 0.5,
            range: Some((-100.0, 100.0)),
        },
        ("polystar", "points") => SpNumberMeta {
            is_radians: false,
            suffix: "",
            speed: 0.2,
            range: Some((3.0, 128.0)),
        },
        ("zigZag", "ridges") => SpNumberMeta {
            is_radians: false,
            suffix: "",
            speed: 0.2,
            range: Some((1.0, 64.0)),
        },
        ("repeater", "copies") => SpNumberMeta {
            is_radians: false,
            suffix: "",
            speed: 0.1,
            range: Some((0.0, 128.0)),
        },
        (_, "width" | "strokeWidth" | "radius" | "roundness" | "outerRadius" | "innerRadius") => {
            SpNumberMeta {
                is_radians: false,
                suffix: "",
                speed: 0.5,
                range: Some((0.0, 10_000.0)),
            }
        }
        (_, "miterLimit") => SpNumberMeta {
            is_radians: false,
            suffix: "",
            speed: 0.1,
            range: Some((1.0, 100.0)),
        },
        (_, "fontSize") => SpNumberMeta {
            is_radians: false,
            suffix: "",
            speed: 1.0,
            range: Some((1.0, 2000.0)),
        },
        (
            _,
            "correlation" | "highlightLength" | "outerRoundness" | "innerRoundness" | "waveAmount"
            | "startLength" | "endLength" | "taperStartLength" | "taperEndLength",
        ) => SpNumberMeta {
            is_radians: false,
            suffix: "%",
            speed: 0.5,
            range: None,
        },
        _ => SpNumberMeta {
            is_radians: false,
            suffix: "",
            speed: 0.5,
            range: None,
        },
    }
}

fn show_sp_fx_button(ui: &mut Ui, has_expr: bool, is_open: bool, theme: &Cm3Theme) -> bool {
    let active_col = Color32::from_rgb(0xe0, 0xa8, 0x4a);
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(22.0, 18.0), Sense::click());
    let resp = resp
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text(if has_expr {
            "Edit expression (active)"
        } else {
            "Add / edit expression"
        });
    let bg = if is_open {
        theme.accent()
    } else if resp.hovered() {
        BG_HOVER
    } else {
        BG_BUTTON
    };
    ui.painter().rect_filled(rect, CornerRadius::same(2), bg);
    paint_crisp_panel_border(
        ui.painter(),
        rect,
        if has_expr {
            active_col
        } else if resp.hovered() || is_open {
            theme.accent()
        } else {
            BORDER_CONTROL
        },
    );
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        "fx",
        theme.font(FONT_COMPACT),
        if has_expr {
            active_col
        } else if is_open || resp.hovered() {
            TEXT_BRIGHT
        } else {
            TEXT_MUTED
        },
    );
    resp.clicked()
}

#[allow(clippy::too_many_arguments)]
fn show_sp_kf_and_expr_subrows(
    ui: &mut Ui,
    sel_id: &str,
    key: &str,
    prop_val: &mut Value,
    current_val: &Value,
    default_tween: u16,
    interpolated: bool,
    local_frame: f64,
    clip_start: f64,
    easing_plus_enabled: bool,
    viewport_rect: Rect,
    theme: &Cm3Theme,
    expr_open_set: &mut BTreeSet<(String, String)>,
    pending_seek_frame: &mut Option<f64>,
) -> bool {
    let mut dirty = false;
    let animated = sp_prop_is_animated(prop_val);
    if animated {
        let frames = sp_prop_keyframe_frames(prop_val);
        let has_prev = frames.iter().any(|&f| f < local_frame - 1e-4);
        let has_next = frames.iter().any(|&f| f > local_frame + 1e-4);
        let exact_tween = sp_prop_keyframe_tween_at(prop_val, local_frame);
        let kf_state = KeyframeBarState {
            has_prev,
            is_keyed: exact_tween.is_some(),
            has_next,
            interpolated,
            tween: exact_tween.unwrap_or(default_tween),
            easing_plus_enabled,
            viewport_rect,
        };
        let ev = ui
            .horizontal(|ui| {
                ui.add_space(24.0);
                cm3_keyframe_bar(ui, Id::unique(("sp_kf_bar", sel_id, key)), kf_state, theme)
            })
            .inner;
        if ev.prev_kf
            && let Some(&prev) = frames.iter().rev().find(|&&f| f < local_frame - 1e-4)
        {
            *pending_seek_frame = Some(prev + clip_start);
        }
        if ev.next_kf
            && let Some(&next) = frames.iter().find(|&&f| f > local_frame + 1e-4)
        {
            *pending_seek_frame = Some(next + clip_start);
        }
        if ev.toggle_kf {
            sp_prop_toggle_keyframe(prop_val, local_frame, current_val.clone(), default_tween);
            dirty = true;
        }
        if ev.toggle_interp
            && let Some(cur_tw) = exact_tween
        {
            let toggled = if (cur_tw & 0xFF) == 0 {
                (1 << 8) | 1
            } else {
                cur_tw ^ (1 << 8)
            };
            if sp_prop_set_keyframe_tween(prop_val, local_frame, toggled) {
                dirty = true;
            }
        }
        if let Some(ease_idx) = ev.set_easing
            && let Some(cur_tw) = exact_tween
        {
            let new_tw = if ease_idx == 0 {
                0
            } else {
                (cur_tw & 0xFF00) | u16::from(ease_idx)
            };
            if sp_prop_set_keyframe_tween(prop_val, local_frame, new_tw) {
                dirty = true;
            }
        }
    }

    let expr_key = (sel_id.to_owned(), key.to_owned());
    if expr_open_set.contains(&expr_key) {
        let mut expr_buf = sp_prop_expression(prop_val).unwrap_or("").to_owned();
        ui.horizontal(|ui| {
            ui.add_space(24.0);
            ui.spacing_mut().item_spacing.x = 4.0;
            let resp = ui.add(
                TextEdit::singleline(&mut expr_buf)
                    .id(Id::unique(("sp_expr_edit", sel_id, key)))
                    .hint_text("Expression (e.g. Math.sin(time * 2) * 50)")
                    .font(egui::FontId::monospace(11.0))
                    .desired_width((ui.available_width() - 32.0).max(80.0)),
            );
            if resp.changed() {
                sp_prop_set_expression(prop_val, current_val.clone(), Some(&expr_buf));
                dirty = true;
            }
            if ui
                .small_button("×")
                .on_hover_text("Clear expression")
                .clicked()
            {
                sp_prop_set_expression(prop_val, current_val.clone(), None);
                expr_open_set.remove(&expr_key);
                dirty = true;
            }
        });
    }

    dirty
}

#[allow(clippy::too_many_arguments)]
fn show_sp_number_row(
    ui: &mut Ui,
    sel_id: &str,
    key: &str,
    label: &str,
    meta: SpNumberMeta,
    prop_val: &mut Value,
    local_frame: f64,
    clip_start: f64,
    easing_plus_enabled: bool,
    viewport_rect: Rect,
    theme: &Cm3Theme,
    tx_tracker: &mut WidgetTxTracker,
    expr_open_set: &mut BTreeSet<(String, String)>,
    pending_seek_frame: &mut Option<f64>,
) -> bool {
    let mut dirty = false;
    let stored = sp_eval_f64(Some(prop_val), local_frame, 0.0);
    let mut display = if meta.is_radians {
        stored.to_degrees()
    } else {
        stored
    };
    let animated = sp_prop_is_animated(prop_val);
    let has_expr = sp_prop_expression(prop_val).is_some();
    let expr_key = (sel_id.to_owned(), key.to_owned());
    let expr_open = expr_open_set.contains(&expr_key);

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let sw = cm3_stopwatch_button(ui, key, animated, has_expr, theme);
        if sw.toggled {
            sp_prop_toggle_animated(prop_val, local_frame, json!(stored), 1);
            dirty = true;
        }
        if sw.alt_expression && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
        ui.label(
            egui::RichText::new(label)
                .font(theme.font(FONT_SMALL))
                .color(TEXT_PRIMARY),
        );
        let mut dv = egui::DragValue::new(&mut display).speed(meta.speed);
        if !meta.suffix.is_empty() {
            dv = dv.suffix(meta.suffix);
        }
        if let Some((lo, hi)) = meta.range {
            dv = dv.range(lo..=hi);
        }
        let resp = ui.add(dv);
        if tx_tracker.observe(&resp) {
            let next_stored = if meta.is_radians {
                display.to_radians()
            } else {
                display
            };
            sp_prop_set_value(prop_val, local_frame, json!(next_stored), 1);
            dirty = true;
        }
        if show_sp_fx_button(ui, has_expr, expr_open, theme) && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
    });

    let cur_stored = sp_eval_f64(Some(prop_val), local_frame, stored);
    if show_sp_kf_and_expr_subrows(
        ui,
        sel_id,
        key,
        prop_val,
        &json!(cur_stored),
        1,
        true,
        local_frame,
        clip_start,
        easing_plus_enabled,
        viewport_rect,
        theme,
        expr_open_set,
        pending_seek_frame,
    ) {
        dirty = true;
    }
    dirty
}

#[allow(clippy::too_many_arguments)]
fn show_sp_vec2_row(
    ui: &mut Ui,
    sel_id: &str,
    key: &str,
    label: &str,
    prop_val: &mut Value,
    local_frame: f64,
    clip_start: f64,
    easing_plus_enabled: bool,
    viewport_rect: Rect,
    theme: &Cm3Theme,
    tx_tracker: &mut WidgetTxTracker,
    expr_open_set: &mut BTreeSet<(String, String)>,
    pending_seek_frame: &mut Option<f64>,
) -> bool {
    let mut dirty = false;
    let cur = sp_eval_vec2(Some(prop_val), local_frame, [0.0, 0.0]);
    let mut x = cur[0];
    let mut y = cur[1];
    let animated = sp_prop_is_animated(prop_val);
    let has_expr = sp_prop_expression(prop_val).is_some();
    let expr_key = (sel_id.to_owned(), key.to_owned());
    let expr_open = expr_open_set.contains(&expr_key);
    let speed = if key == "scale" { 0.02 } else { 1.0 };

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let sw = cm3_stopwatch_button(ui, key, animated, has_expr, theme);
        if sw.toggled {
            sp_prop_toggle_animated(prop_val, local_frame, json!([cur[0], cur[1]]), 1);
            dirty = true;
        }
        if sw.alt_expression && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
        ui.label(
            egui::RichText::new(label)
                .font(theme.font(FONT_SMALL))
                .color(TEXT_PRIMARY),
        );
        let cx = ui.add(egui::DragValue::new(&mut x).speed(speed).prefix("X "));
        let cy = ui.add(egui::DragValue::new(&mut y).speed(speed).prefix("Y "));
        let ch_x = tx_tracker.observe(&cx);
        let ch_y = tx_tracker.observe(&cy);
        if ch_x || ch_y {
            sp_prop_set_value(prop_val, local_frame, json!([x, y]), 1);
            dirty = true;
        }
        if show_sp_fx_button(ui, has_expr, expr_open, theme) && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
    });

    let cur_now = sp_eval_vec2(Some(prop_val), local_frame, [x, y]);
    if show_sp_kf_and_expr_subrows(
        ui,
        sel_id,
        key,
        prop_val,
        &json!([cur_now[0], cur_now[1]]),
        1,
        true,
        local_frame,
        clip_start,
        easing_plus_enabled,
        viewport_rect,
        theme,
        expr_open_set,
        pending_seek_frame,
    ) {
        dirty = true;
    }
    dirty
}

#[allow(clippy::too_many_arguments)]
fn show_sp_rgb_row(
    ui: &mut Ui,
    sel_id: &str,
    key: &str,
    label: &str,
    prop_val: &mut Value,
    local_frame: f64,
    clip_start: f64,
    easing_plus_enabled: bool,
    viewport_rect: Rect,
    theme: &Cm3Theme,
    tx_tracker: &mut WidgetTxTracker,
    expr_open_set: &mut BTreeSet<(String, String)>,
    pending_seek_frame: &mut Option<f64>,
) -> bool {
    let mut dirty = false;
    let cur = sp_eval_rgb(Some(prop_val), local_frame, [1.0, 1.0, 1.0]);
    let mut rgb = [cur[0] as f32, cur[1] as f32, cur[2] as f32];
    let animated = sp_prop_is_animated(prop_val);
    let has_expr = sp_prop_expression(prop_val).is_some();
    let expr_key = (sel_id.to_owned(), key.to_owned());
    let expr_open = expr_open_set.contains(&expr_key);

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let sw = cm3_stopwatch_button(ui, key, animated, has_expr, theme);
        if sw.toggled {
            sp_prop_toggle_animated(prop_val, local_frame, json!([cur[0], cur[1], cur[2]]), 1);
            dirty = true;
        }
        if sw.alt_expression && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
        ui.label(
            egui::RichText::new(label)
                .font(theme.font(FONT_SMALL))
                .color(TEXT_PRIMARY),
        );
        let resp = ui.color_edit_button_rgb(&mut rgb);
        if tx_tracker.observe(&resp) {
            sp_prop_set_value(
                prop_val,
                local_frame,
                json!([f64::from(rgb[0]), f64::from(rgb[1]), f64::from(rgb[2])]),
                1,
            );
            dirty = true;
        }
        if show_sp_fx_button(ui, has_expr, expr_open, theme) && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
    });

    let cur_now = sp_eval_rgb(Some(prop_val), local_frame, cur);
    if show_sp_kf_and_expr_subrows(
        ui,
        sel_id,
        key,
        prop_val,
        &json!([cur_now[0], cur_now[1], cur_now[2]]),
        1,
        true,
        local_frame,
        clip_start,
        easing_plus_enabled,
        viewport_rect,
        theme,
        expr_open_set,
        pending_seek_frame,
    ) {
        dirty = true;
    }
    dirty
}

#[allow(clippy::too_many_arguments)]
fn show_sp_path_row(
    ui: &mut Ui,
    sel_id: &str,
    key: &str,
    vertex_count: usize,
    closed: &mut bool,
    prop_val: &mut Value,
    current_val: Value,
    local_frame: f64,
    clip_start: f64,
    easing_plus_enabled: bool,
    viewport_rect: Rect,
    theme: &Cm3Theme,
    expr_open_set: &mut BTreeSet<(String, String)>,
    pending_seek_frame: &mut Option<f64>,
) -> bool {
    let mut dirty = false;
    let animated = sp_prop_is_animated(prop_val);
    let has_expr = sp_prop_expression(prop_val).is_some();
    let expr_key = (sel_id.to_owned(), key.to_owned());
    let expr_open = expr_open_set.contains(&expr_key);
    let kf_count = sp_prop_keyframe_frames(prop_val).len();

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let sw = cm3_stopwatch_button(ui, key, animated, has_expr, theme);
        if sw.toggled {
            sp_prop_toggle_animated(prop_val, local_frame, current_val.clone(), 1);
            dirty = true;
        }
        if sw.alt_expression && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
        ui.label(
            egui::RichText::new("Path")
                .font(theme.font(FONT_SMALL))
                .color(TEXT_PRIMARY),
        );
        let summary = if animated {
            format!("{vertex_count}v · {kf_count}k")
        } else {
            format!(
                "{vertex_count} {}",
                if vertex_count == 1 {
                    "vertex"
                } else {
                    "vertices"
                }
            )
        };
        ui.label(
            egui::RichText::new(summary)
                .font(theme.font(FONT_COMPACT))
                .color(theme.value()),
        );
        ui.checkbox(closed, "Closed");
        if show_sp_fx_button(ui, has_expr, expr_open, theme) && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
    });

    if show_sp_kf_and_expr_subrows(
        ui,
        sel_id,
        key,
        prop_val,
        &current_val,
        1,
        true,
        local_frame,
        clip_start,
        easing_plus_enabled,
        viewport_rect,
        theme,
        expr_open_set,
        pending_seek_frame,
    ) {
        dirty = true;
    }
    dirty
}

#[allow(clippy::too_many_arguments)]
fn show_sp_gradient_stops_editor(
    ui: &mut Ui,
    sel_id: &str,
    key: &str,
    label: &str,
    prop_val: &mut Value,
    local_frame: f64,
    clip_start: f64,
    easing_plus_enabled: bool,
    viewport_rect: Rect,
    theme: &Cm3Theme,
    tx_tracker: &mut WidgetTxTracker,
    expr_open_set: &mut BTreeSet<(String, String)>,
    pending_seek_frame: &mut Option<f64>,
) -> bool {
    let mut dirty = false;
    let mut stops = sp_eval_gradient_stops(Some(prop_val), local_frame);
    let animated = sp_prop_is_animated(prop_val);
    let has_expr = sp_prop_expression(prop_val).is_some();
    let expr_key = (sel_id.to_owned(), key.to_owned());
    let expr_open = expr_open_set.contains(&expr_key);

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let sw = cm3_stopwatch_button(ui, key, animated, has_expr, theme);
        if sw.toggled {
            sp_prop_toggle_animated(prop_val, local_frame, sp_stops_to_json(&stops), 0);
            dirty = true;
        }
        if sw.alt_expression && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
        ui.label(
            egui::RichText::new(label)
                .font(theme.font(FONT_SMALL))
                .color(TEXT_PRIMARY),
        );
        if show_sp_fx_button(ui, has_expr, expr_open, theme) && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
    });

    // Live gradient preview bar + stop position markers
    let bar_w = (ui.available_width() - 8.0).clamp(120.0, 260.0);
    let (bar_rect, _) = ui.allocate_exact_size(Vec2::new(bar_w, 20.0), Sense::hover());
    ui.painter()
        .rect_filled(bar_rect, CornerRadius::ZERO, BG_SUNKEN);
    paint_gradient_mesh(ui.painter(), bar_rect, &stops, 24);
    paint_crisp_panel_border(ui.painter(), bar_rect, BORDER_CONTROL);
    for stop in &stops {
        let sx = bar_rect.left() + (stop.position.clamp(0.0, 1.0) as f32) * bar_rect.width();
        let tick =
            Rect::from_center_size(Pos2::new(sx, bar_rect.bottom() - 3.0), Vec2::new(5.0, 6.0));
        ui.painter()
            .rect_filled(tick, CornerRadius::ZERO, sp_rgba_to_color32(stop.color));
        ui.painter().rect_stroke(
            tick,
            CornerRadius::ZERO,
            Stroke::new(1.0, Color32::WHITE),
            StrokeKind::Inside,
        );
    }

    ui.add_space(3.0);
    let mut stops_changed = false;
    let mut remove_idx: Option<usize> = None;
    let can_remove = stops.len() > 2;

    for (idx, stop) in stops.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            ui.label(
                egui::RichText::new(format!("#{}", idx + 1))
                    .font(theme.font(FONT_COMPACT))
                    .color(TEXT_MUTED),
            );
            let pos_resp = ui.add(
                egui::DragValue::new(&mut stop.position)
                    .range(0.0..=1.0)
                    .speed(0.005)
                    .fixed_decimals(3),
            );
            if tx_tracker.observe(&pos_resp) {
                stops_changed = true;
            }
            let mut rgb = [
                stop.color[0] as f32,
                stop.color[1] as f32,
                stop.color[2] as f32,
            ];
            let col_resp = ui.color_edit_button_rgb(&mut rgb);
            if tx_tracker.observe(&col_resp) {
                stop.color[0] = f64::from(rgb[0]);
                stop.color[1] = f64::from(rgb[1]);
                stop.color[2] = f64::from(rgb[2]);
                stops_changed = true;
            }
            let alpha_resp = ui.add(
                egui::DragValue::new(&mut stop.color[3])
                    .range(0.0..=1.0)
                    .speed(0.01)
                    .prefix("A ")
                    .fixed_decimals(2),
            );
            if tx_tracker.observe(&alpha_resp) {
                stops_changed = true;
            }
            if can_remove
                && ui
                    .small_button("×")
                    .on_hover_text("Remove gradient stop")
                    .clicked()
            {
                remove_idx = Some(idx);
            }
        });
    }

    if let Some(idx) = remove_idx
        && stops.len() > 2
    {
        stops.remove(idx);
        stops_changed = true;
    }

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        if stops.len() < 16
            && ui
                .small_button("+ Add Stop")
                .on_hover_text("Insert an interpolated gradient stop (max 16)")
                .clicked()
        {
            // Place new stop at the midpoint of the largest gap between stops
            let mut best_t = 0.5;
            let mut best_gap = -1.0_f64;
            for w in stops.windows(2) {
                let gap = w[1].position - w[0].position;
                if gap > best_gap {
                    best_gap = gap;
                    best_t = (w[0].position + w[1].position) * 0.5;
                }
            }
            let color = sp_sample_gradient(&stops, best_t);
            stops.push(SpGradientStop {
                position: best_t,
                color,
            });
            stops.sort_by(|a, b| a.position.total_cmp(&b.position));
            stops_changed = true;
        }
        if ui
            .small_button("Reverse")
            .on_hover_text("Reverse gradient stops")
            .clicked()
        {
            for s in &mut stops {
                s.position = (1.0 - s.position).clamp(0.0, 1.0);
            }
            stops.sort_by(|a, b| a.position.total_cmp(&b.position));
            stops_changed = true;
        }
        if stops.len() >= 2
            && ui
                .small_button("Distribute")
                .on_hover_text("Evenly distribute stop positions across 0..1")
                .clicked()
        {
            stops.sort_by(|a, b| a.position.total_cmp(&b.position));
            let n = stops.len();
            for (i, s) in stops.iter_mut().enumerate() {
                s.position = i as f64 / (n - 1) as f64;
            }
            stops_changed = true;
        }
    });

    if stops_changed {
        stops.sort_by(|a, b| a.position.total_cmp(&b.position));
        sp_prop_set_value(prop_val, local_frame, sp_stops_to_json(&stops), 0);
        dirty = true;
    }

    let cur_json = sp_stops_to_json(&stops);
    if show_sp_kf_and_expr_subrows(
        ui,
        sel_id,
        key,
        prop_val,
        &cur_json,
        0,
        false,
        local_frame,
        clip_start,
        easing_plus_enabled,
        viewport_rect,
        theme,
        expr_open_set,
        pending_seek_frame,
    ) {
        dirty = true;
    }
    dirty
}

fn is_stroke_dash_subkey(key: &str) -> bool {
    matches!(
        key,
        "dashes"
            | "dashPairs"
            | "dash1"
            | "dash2"
            | "dash3"
            | "gap1"
            | "gap2"
            | "gap3"
            | "dashLength"
            | "dashGap"
            | "dashOffset"
    )
}

fn sp_stroke_has_active_dashes(props: &serde_json::Map<String, Value>, frame: f64) -> bool {
    if sp_eval_f64(props.get("dashPairs"), frame, 0.0) >= 0.5 {
        return true;
    }
    if let Some(d) = props.get("dashes")
        && !sp_eval_f64_vec(Some(d), frame).is_empty()
    {
        return true;
    }
    sp_eval_f64(props.get("dashOffset"), frame, 0.0).abs() > 1e-6
}

fn sp_sync_dashes_to_stroke_props(
    props: &mut serde_json::Map<String, Value>,
    pairs: &[[f64; 2]],
    frame: f64,
) {
    let clamped = pairs.len().min(3);
    props.insert("dashPairs".to_owned(), json!(clamped));
    for (i, &[dash, gap]) in pairs.iter().take(3).enumerate() {
        let dk = format!("dash{}", i + 1);
        let gk = format!("gap{}", i + 1);
        if let Some(existing) = props.get_mut(&dk) {
            sp_prop_set_value(existing, frame, json!(dash), 1);
        } else {
            props.insert(dk, json!(dash));
        }
        if let Some(existing) = props.get_mut(&gk) {
            sp_prop_set_value(existing, frame, json!(gap), 1);
        } else {
            props.insert(gk, json!(gap));
        }
    }
    if props.contains_key("dashes") {
        let mut flat = Vec::with_capacity(pairs.len() * 2);
        for &[d, g] in pairs {
            flat.push(d);
            flat.push(g);
        }
        if let Some(existing) = props.get_mut("dashes") {
            sp_prop_set_value(existing, frame, json!(flat), 1);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn show_sp_stroke_dashes_section(
    ui: &mut Ui,
    sel_id: &str,
    props: &mut serde_json::Map<String, Value>,
    local_frame: f64,
    clip_start: f64,
    easing_plus_enabled: bool,
    viewport_rect: Rect,
    theme: &Cm3Theme,
    tx_tracker: &mut WidgetTxTracker,
    expr_open_set: &mut BTreeSet<(String, String)>,
    pending_seek_frame: &mut Option<f64>,
) -> bool {
    let mut dirty = false;
    let mut pairs: Vec<[f64; 2]> = if let Some(d_val) = props.get("dashes") {
        let flat = sp_eval_f64_vec(Some(d_val), local_frame);
        flat.chunks(2)
            .map(|chunk| {
                let d = chunk[0].max(0.0);
                let g = chunk.get(1).copied().unwrap_or(d).max(0.0);
                [d, g]
            })
            .collect()
    } else {
        let n = sp_eval_f64(props.get("dashPairs"), local_frame, 0.0)
            .round()
            .clamp(0.0, 3.0) as usize;
        (0..n)
            .map(|i| {
                let d =
                    sp_eval_f64(props.get(&format!("dash{}", i + 1)), local_frame, 20.0).max(0.0);
                let g =
                    sp_eval_f64(props.get(&format!("gap{}", i + 1)), local_frame, 10.0).max(0.0);
                [d, g]
            })
            .collect()
    };

    let mut pairs_changed = false;
    let mut remove_idx: Option<usize> = None;
    for (idx, pair) in pairs.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            ui.label(
                egui::RichText::new(format!("Dash {}", idx + 1))
                    .font(theme.font(FONT_COMPACT))
                    .color(TEXT_MUTED),
            );
            let d_resp = ui.add(
                egui::DragValue::new(&mut pair[0])
                    .range(0.0..=10_000.0)
                    .speed(0.5)
                    .prefix("D "),
            );
            let g_resp = ui.add(
                egui::DragValue::new(&mut pair[1])
                    .range(0.0..=10_000.0)
                    .speed(0.5)
                    .prefix("G "),
            );
            let ch_d = tx_tracker.observe(&d_resp);
            let ch_g = tx_tracker.observe(&g_resp);
            if ch_d || ch_g {
                pairs_changed = true;
            }
            if ui
                .small_button("×")
                .on_hover_text("Remove dash/gap pair")
                .clicked()
            {
                remove_idx = Some(idx);
            }
        });
    }

    if let Some(idx) = remove_idx {
        pairs.remove(idx);
        pairs_changed = true;
    }

    if pairs.len() < 3
        && ui
            .small_button("+ Add Dash")
            .on_hover_text("Add a dash/gap pair (up to 3 pairs)")
            .clicked()
    {
        pairs.push([20.0, 10.0]);
        pairs_changed = true;
    }

    if pairs_changed {
        sp_sync_dashes_to_stroke_props(props, &pairs, local_frame);
        dirty = true;
    }

    // Animated Dash Offset row
    let mut off_val = props
        .get("dashOffset")
        .cloned()
        .unwrap_or_else(|| json!(0.0));
    if show_sp_number_row(
        ui,
        sel_id,
        "dashOffset",
        "Dash Offset",
        SpNumberMeta {
            is_radians: false,
            suffix: "",
            speed: 0.5,
            range: None,
        },
        &mut off_val,
        local_frame,
        clip_start,
        easing_plus_enabled,
        viewport_rect,
        theme,
        tx_tracker,
        expr_open_set,
        pending_seek_frame,
    ) {
        props.insert("dashOffset".to_owned(), off_val);
        dirty = true;
    }

    dirty
}

fn show_sp_dashes_array_editor(
    ui: &mut Ui,
    label: &str,
    prop_val: &mut Value,
    local_frame: f64,
    theme: &Cm3Theme,
    tx_tracker: &mut WidgetTxTracker,
) -> bool {
    let mut nums = sp_eval_f64_vec(Some(prop_val), local_frame);
    let mut changed = false;
    let mut remove_idx: Option<usize> = None;

    ui.label(
        egui::RichText::new(label)
            .font(theme.font(FONT_SMALL))
            .color(TEXT_PRIMARY),
    );
    for (idx, v) in nums.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            let tag = if idx.is_multiple_of(2) { "Dash" } else { "Gap" };
            ui.label(
                egui::RichText::new(format!("{tag} {}", idx / 2 + 1))
                    .font(theme.font(FONT_COMPACT))
                    .color(TEXT_MUTED),
            );
            let resp = ui.add(egui::DragValue::new(v).range(0.0..=10_000.0).speed(0.5));
            if tx_tracker.observe(&resp) {
                changed = true;
            }
            if ui.small_button("×").on_hover_text("Remove entry").clicked() {
                remove_idx = Some(idx);
            }
        });
    }
    if let Some(idx) = remove_idx {
        nums.remove(idx);
        changed = true;
    }
    if ui.small_button("+ Add Dash").clicked() {
        nums.push(if nums.len().is_multiple_of(2) {
            20.0
        } else {
            10.0
        });
        changed = true;
    }
    if changed {
        sp_prop_set_value(prop_val, local_frame, json!(nums), 1);
    }
    changed
}

fn show_shape_plus_preset_variables(
    ui: &mut Ui,
    pv: &mut Value,
    viewport_rect: Rect,
    theme: &Cm3Theme,
    tx_tracker: &mut WidgetTxTracker,
) -> bool {
    let mut dirty = false;
    if let Some(obj) = pv.as_object_mut() {
        let keys: Vec<String> = obj.keys().cloned().collect();
        for key in keys {
            let Some(entry) = obj.get_mut(&key) else {
                continue;
            };
            let display = entry
                .get("label")
                .or_else(|| entry.get("name"))
                .and_then(Value::as_str)
                .map_or_else(|| humanize_prop_label(&key), str::to_owned);
            if show_one_preset_variable_row(
                ui,
                &key,
                &display,
                entry,
                viewport_rect,
                theme,
                tx_tracker,
            ) {
                dirty = true;
            }
        }
    } else if let Some(arr) = pv.as_array_mut() {
        for (idx, entry) in arr.iter_mut().enumerate() {
            let id_key = format!("var_{idx}");
            let display = entry
                .get("label")
                .or_else(|| entry.get("name"))
                .or_else(|| entry.get("id"))
                .and_then(Value::as_str)
                .map_or_else(|| format!("Variable {}", idx + 1), str::to_owned);
            if show_one_preset_variable_row(
                ui,
                &id_key,
                &display,
                entry,
                viewport_rect,
                theme,
                tx_tracker,
            ) {
                dirty = true;
            }
        }
    }
    dirty
}

fn show_one_preset_variable_row(
    ui: &mut Ui,
    id_key: &str,
    label: &str,
    entry: &mut Value,
    viewport_rect: Rect,
    theme: &Cm3Theme,
    tx_tracker: &mut WidgetTxTracker,
) -> bool {
    let is_desc = entry.is_object() && entry.get("value").is_some();
    let min_val = entry.get("min").and_then(Value::as_f64);
    let max_val = entry.get("max").and_then(Value::as_f64);
    let step_val = entry.get("step").and_then(Value::as_f64).unwrap_or(0.5);
    let declared_type = entry
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let options: Option<Vec<String>> = entry
        .get("options")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    v.as_str()
                        .or_else(|| v.get("label").and_then(Value::as_str))
                        .map(str::to_owned)
                })
                .collect()
        })
        .filter(|v: &Vec<String>| !v.is_empty());

    let target = if is_desc {
        entry.get_mut("value").expect("checked value")
    } else {
        entry
    };

    if let Some(opts_owned) = options
        && let Some(cur_s) = target.as_str()
    {
        let opts_ref: Vec<&str> = opts_owned.iter().map(String::as_str).collect();
        let mut idx = opts_ref
            .iter()
            .position(|o| o.eq_ignore_ascii_case(cur_s))
            .unwrap_or(0);
        let mut ch = false;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            ui.label(
                egui::RichText::new(label)
                    .font(theme.font(FONT_SMALL))
                    .color(TEXT_PRIMARY),
            );
            if cm3_dropdown(
                ui,
                Id::unique(("sp_pvar_dd", id_key)),
                &mut idx,
                &opts_ref,
                false,
                viewport_rect,
                theme,
            ) {
                *target = json!(opts_owned[idx]);
                ch = true;
            }
        });
        return ch;
    }

    match target.clone() {
        Value::Bool(mut b) => {
            let mut ch = false;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                ui.label(
                    egui::RichText::new(label)
                        .font(theme.font(FONT_SMALL))
                        .color(TEXT_PRIMARY),
                );
                if ui.checkbox(&mut b, "").changed() {
                    *target = json!(b);
                    ch = true;
                }
            });
            ch
        }
        Value::Number(n) => {
            let mut v = n.as_f64().unwrap_or(0.0);
            let mut ch = false;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                ui.label(
                    egui::RichText::new(label)
                        .font(theme.font(FONT_SMALL))
                        .color(TEXT_PRIMARY),
                );
                let mut dv = egui::DragValue::new(&mut v).speed(step_val.max(0.01));
                if let (Some(lo), Some(hi)) = (min_val, max_val) {
                    dv = dv.range(lo..=hi);
                }
                let resp = ui.add(dv);
                if tx_tracker.observe(&resp) {
                    *target = json!(v);
                    ch = true;
                }
            });
            ch
        }
        Value::Array(arr) if arr.len() == 2 && arr.iter().all(Value::is_number) => {
            let mut x = arr[0].as_f64().unwrap_or(0.0);
            let mut y = arr[1].as_f64().unwrap_or(0.0);
            let mut ch = false;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.label(
                    egui::RichText::new(label)
                        .font(theme.font(FONT_SMALL))
                        .color(TEXT_PRIMARY),
                );
                let rx = ui.add(egui::DragValue::new(&mut x).speed(step_val).prefix("X "));
                let ry = ui.add(egui::DragValue::new(&mut y).speed(step_val).prefix("Y "));
                let cx = tx_tracker.observe(&rx);
                let cy = tx_tracker.observe(&ry);
                if cx || cy {
                    *target = json!([x, y]);
                    ch = true;
                }
            });
            ch
        }
        Value::Array(arr)
            if (arr.len() == 3 || declared_type == "color")
                && arr.len() >= 3
                && arr.iter().take(3).all(Value::is_number) =>
        {
            let mut rgb = [
                arr[0].as_f64().unwrap_or(1.0) as f32,
                arr[1].as_f64().unwrap_or(1.0) as f32,
                arr[2].as_f64().unwrap_or(1.0) as f32,
            ];
            let mut ch = false;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                ui.label(
                    egui::RichText::new(label)
                        .font(theme.font(FONT_SMALL))
                        .color(TEXT_PRIMARY),
                );
                let resp = ui.color_edit_button_rgb(&mut rgb);
                if tx_tracker.observe(&resp) {
                    *target = json!([f64::from(rgb[0]), f64::from(rgb[1]), f64::from(rgb[2])]);
                    ch = true;
                }
            });
            ch
        }
        Value::String(mut s) => {
            let mut ch = false;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                ui.label(
                    egui::RichText::new(label)
                        .font(theme.font(FONT_SMALL))
                        .color(TEXT_PRIMARY),
                );
                if declared_type == "color" {
                    let c = sp_parse_css_rgba(&s);
                    let mut rgb = [c[0] as f32, c[1] as f32, c[2] as f32];
                    let resp = ui.color_edit_button_rgb(&mut rgb);
                    if tx_tracker.observe(&resp) {
                        *target = json!(sp_format_css_rgba([
                            f64::from(rgb[0]),
                            f64::from(rgb[1]),
                            f64::from(rgb[2]),
                            c[3],
                        ]));
                        ch = true;
                    }
                } else if ui
                    .add(
                        TextEdit::singleline(&mut s)
                            .font(theme.font(FONT_SMALL))
                            .desired_width((ui.available_width() - 8.0).clamp(70.0, 150.0)),
                    )
                    .changed()
                {
                    *target = json!(s);
                    ch = true;
                }
            });
            ch
        }
        _ => false,
    }
}

fn find_shape_item_ref<'a>(payload: &'a Value, id: &str) -> Option<&'a Value> {
    fn rec<'a>(arr: &'a [Value], id: &str) -> Option<&'a Value> {
        for item in arr {
            if item.get("id").and_then(Value::as_str) == Some(id) {
                return Some(item);
            }
            if let Some(children) = item.get("contents").and_then(Value::as_array)
                && let Some(found) = rec(children, id)
            {
                return Some(found);
            }
        }
        None
    }
    payload
        .get("contents")
        .and_then(Value::as_array)
        .and_then(|c| rec(c, id))
}

fn lookup_project_font_bytes<'a>(session: &'a Session, font_ref: &str) -> Option<&'a [u8]> {
    let trimmed = font_ref.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(asset) = session.project.assets.get(trimmed) {
        return asset.bytes.as_deref();
    }
    for asset in session.project.assets.values() {
        if asset.id == trimmed || asset.filename.eq_ignore_ascii_case(trimmed) {
            return asset.bytes.as_deref();
        }
        let stem = std::path::Path::new(&asset.filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&asset.filename);
        if stem.eq_ignore_ascii_case(trimmed) {
            return asset.bytes.as_deref();
        }
    }
    None
}

fn resolve_shape_plus_item_font_bytes<'a>(
    session: &'a Session,
    payload: &Value,
    item_id: &str,
    frame: f64,
) -> (Option<&'a [u8]>, Option<&'a [u8]>) {
    let Some(item) = find_shape_item_ref(payload, item_id) else {
        return (None, None);
    };
    if item.get("kind").and_then(Value::as_str) != Some("text") {
        return (None, None);
    }
    let props = item.get("properties").and_then(Value::as_object);
    let primary_ref = props
        .and_then(|p| p.get("fontFamily").or_else(|| p.get("font")))
        .map(|v| sp_eval_str(Some(v), frame, ""))
        .unwrap_or_default();
    let fallback_ref = props
        .and_then(|p| p.get("fallbackFont"))
        .map(|v| sp_eval_str(Some(v), frame, ""))
        .unwrap_or_default();
    (
        lookup_project_font_bytes(session, &primary_ref),
        lookup_project_font_bytes(session, &fallback_ref),
    )
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
        "path" => 1,
        "mode" | "gradientType" => 2,
        "position" => 3,
        "size" => 4,
        "points" => 5,
        "outerRadius" => 6,
        "innerRadius" => 7,
        "roundness" | "outerRoundness" => 8,
        "innerRoundness" => 9,
        "scale" => 10,
        "rotation" | "angle" | "twistAngle" => 11,
        "anchor" => 12,
        "skew" => 13,
        "skewAxis" => 14,
        "color" | "stops" => 15,
        "width" | "strokeWidth" => 16,
        "lineCap" | "cap" => 17,
        "lineJoin" | "join" => 18,
        "miterLimit" => 19,
        "opacity" => 20,
        "composite" => 21,
        "fillRule" | "fill_rule" => 22,
        _ => 100,
    }
}

pub(super) fn shape_plus_string_enum_options(
    kind: &str,
    key: &str,
) -> Option<&'static [&'static str]> {
    match (kind, key) {
        ("polystar", "mode") | (_, "polystarType" | "starType") => Some(&["star", "polygon"]),
        ("trim", "mode") => Some(&["simultaneous", "individual"]),
        ("mergePaths", "mode") => Some(&["merge", "add", "subtract", "intersect", "exclude"]),
        ("zigZag" | "wigglePaths", "points") => Some(&["corner", "smooth"]),
        (_, "composite") => Some(&["below", "above"]),
        (_, "fillRule" | "fill_rule") => Some(&["nonzero", "even-odd"]),
        (_, "lineCap" | "cap") => Some(&["butt", "round", "square"]),
        (_, "lineJoin" | "join") => Some(&["miter", "round", "bevel"]),
        (_, "gradientType") => Some(&["linear", "radial"]),
        (_, "alignment") => Some(&["left", "center", "right"]),
        (_, "textMode") => Some(&["point", "box", "paragraph"]),
        (_, "caps") => Some(&["normal", "all", "small"]),
        (_, "selectorUnits" | "taperUnits") => Some(&["percentage", "index"]),
        (_, "waveUnits") => Some(&["cycles", "length"]),
        (_, "selectorBasedOn" | "wigglyBasedOn") => {
            Some(&["characters", "charactersExcludingSpaces", "words", "lines"])
        }
        (_, "selectorMode" | "wigglyMode") => {
            Some(&["add", "subtract", "intersect", "min", "max", "difference"])
        }
        (_, "selectorShape") => Some(&[
            "square", "rampUp", "rampDown", "triangle", "round", "smooth",
        ]),
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
            show_clip_track_matte_rows(ui, session, &clip.id, viewport_rect, state, theme);

            // Text+ 2D 3-Stack Character Animators (`zoidiumTextPlus2D`)
            if let Some(clip_uuid) = cid {
                let easing_plus_enabled = session.plugins.is_enabled("easing-plus");
                let expr_open_id = Id::unique(("tp2d_expr_open_set", &clip.id));
                let mut expr_open_set: BTreeSet<(String, String)> = ui
                    .data_mut(|d| d.get_temp::<BTreeSet<(String, String)>>(expr_open_id))
                    .unwrap_or_default();
                let mut pending_seek_frame: Option<f64> = None;

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

                        let slot_scope_id =
                            format!("{}#tp2d_{}", clip.id, state.text2d_animator_tab);
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

                            let mut tx_tracker = WidgetTxTracker::default();
                            match mode {
                                // Character Transform (1) & Selected Transform (4)
                                1 | 4 => {
                                    dirty |= show_tp2d_vec2_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "charPosition",
                                        ("offsetX", "offsetY"),
                                        "Position",
                                        [0.0, 0.0],
                                        1.0,
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_vec2_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "charScale",
                                        ("scaleX", "scaleY"),
                                        "Scale",
                                        [1.0, 1.0],
                                        0.02,
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "charRotation",
                                        &["rotation"],
                                        "Rotation",
                                        0.0,
                                        SpNumberMeta {
                                            is_radians: true,
                                            suffix: "°",
                                            speed: 1.0,
                                            range: None,
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "charOpacity",
                                        &["opacity"],
                                        "Opacity",
                                        1.0,
                                        SpNumberMeta {
                                            is_radians: false,
                                            suffix: "",
                                            speed: 0.01,
                                            range: Some((0.0, 1.0)),
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "delayPerCharacter",
                                        &["delay"],
                                        "Char delay",
                                        0.0,
                                        SpNumberMeta {
                                            is_radians: false,
                                            suffix: "",
                                            speed: 0.25,
                                            range: Some((0.0, 1000.0)),
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_option_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "delayOrder",
                                        "Delay order",
                                        DELAY_ORDERS,
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "randomSeed",
                                        &["seed"],
                                        "Random seed",
                                        1.0,
                                        SpNumberMeta {
                                            is_radians: false,
                                            suffix: "",
                                            speed: 1.0,
                                            range: None,
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                }
                                // Gradient Transform (2)
                                2 => {
                                    dirty |= show_tp2d_vec2_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "firstPosition",
                                        ("offsetX", "offsetY"),
                                        "First pos",
                                        [0.0, 0.0],
                                        1.0,
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_vec2_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "lastPosition",
                                        ("offsetX", "offsetY"),
                                        "Last pos",
                                        [0.0, 0.0],
                                        1.0,
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_vec2_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "firstScale",
                                        ("scaleX", "scaleY"),
                                        "First scale",
                                        [1.0, 1.0],
                                        0.02,
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_vec2_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "lastScale",
                                        ("scaleX", "scaleY"),
                                        "Last scale",
                                        [1.0, 1.0],
                                        0.02,
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "firstRotation",
                                        &[],
                                        "First rot",
                                        0.0,
                                        SpNumberMeta {
                                            is_radians: true,
                                            suffix: "°",
                                            speed: 1.0,
                                            range: None,
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "lastRotation",
                                        &["rotation"],
                                        "Last rot",
                                        0.0,
                                        SpNumberMeta {
                                            is_radians: true,
                                            suffix: "°",
                                            speed: 1.0,
                                            range: None,
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "firstOpacity",
                                        &[],
                                        "First opacity",
                                        1.0,
                                        SpNumberMeta {
                                            is_radians: false,
                                            suffix: "",
                                            speed: 0.01,
                                            range: Some((0.0, 1.0)),
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "lastOpacity",
                                        &["opacity"],
                                        "Last opacity",
                                        1.0,
                                        SpNumberMeta {
                                            is_radians: false,
                                            suffix: "",
                                            speed: 0.01,
                                            range: Some((0.0, 1.0)),
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                }
                                // Character Shake (3) & Selected Shake (5)
                                3 | 5 => {
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "shakeAmplitude",
                                        &["amount"],
                                        "Amplitude",
                                        0.25,
                                        SpNumberMeta {
                                            is_radians: true,
                                            suffix: "°",
                                            speed: 1.0,
                                            range: None,
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "shakePeriod",
                                        &["speed"],
                                        "Period (frames)",
                                        24.0,
                                        SpNumberMeta {
                                            is_radians: false,
                                            suffix: "",
                                            speed: 0.5,
                                            range: Some((0.1, 10_000.0)),
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "shakePhase",
                                        &[],
                                        "Phase",
                                        0.0,
                                        SpNumberMeta {
                                            is_radians: false,
                                            suffix: "",
                                            speed: 0.25,
                                            range: None,
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "shakeDelayPerCharacter",
                                        &["delay"],
                                        "Char delay",
                                        0.0,
                                        SpNumberMeta {
                                            is_radians: false,
                                            suffix: "",
                                            speed: 0.25,
                                            range: Some((0.0, 1000.0)),
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_option_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "shakeDelayOrder",
                                        "Delay order",
                                        DELAY_ORDERS,
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "shakeRandomSeed",
                                        &["seed"],
                                        "Random seed",
                                        1.0,
                                        SpNumberMeta {
                                            is_radians: false,
                                            suffix: "",
                                            speed: 1.0,
                                            range: None,
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                }
                                // Random Scatter (6)
                                6 => {
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "amount",
                                        &[],
                                        "Amount",
                                        1.0,
                                        SpNumberMeta {
                                            is_radians: false,
                                            suffix: "",
                                            speed: 0.02,
                                            range: Some((0.0, 100.0)),
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_vec2_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "positionMin",
                                        ("offsetX", "offsetY"),
                                        "Pos min",
                                        [-10.0, -10.0],
                                        1.0,
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_vec2_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "positionMax",
                                        ("offsetX", "offsetY"),
                                        "Pos max",
                                        [10.0, 10.0],
                                        1.0,
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "rotationMin",
                                        &[],
                                        "Rot min",
                                        0.0,
                                        SpNumberMeta {
                                            is_radians: true,
                                            suffix: "°",
                                            speed: 1.0,
                                            range: None,
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "rotationMax",
                                        &["rotation"],
                                        "Rot max",
                                        0.0,
                                        SpNumberMeta {
                                            is_radians: true,
                                            suffix: "°",
                                            speed: 1.0,
                                            range: None,
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_vec2_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "scaleMin",
                                        ("scaleX", "scaleY"),
                                        "Scale min",
                                        [1.0, 1.0],
                                        0.02,
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_vec2_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "scaleMax",
                                        ("scaleX", "scaleY"),
                                        "Scale max",
                                        [1.0, 1.0],
                                        0.02,
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                    dirty |= show_tp2d_number_row(
                                        ui,
                                        slot,
                                        &slot_scope_id,
                                        "seed",
                                        &[],
                                        "Seed",
                                        1.0,
                                        SpNumberMeta {
                                            is_radians: false,
                                            suffix: "",
                                            speed: 1.0,
                                            range: None,
                                        },
                                        local_frame,
                                        clip.start,
                                        easing_plus_enabled,
                                        viewport_rect,
                                        theme,
                                        &mut tx_tracker,
                                        &mut expr_open_set,
                                        &mut pending_seek_frame,
                                    );
                                }
                                _ => {}
                            }
                            if dirty {
                                tx_tracker.changed = true;
                            }
                            apply_widget_tx_command(
                                ui,
                                session,
                                state,
                                &clip.id,
                                "text2d_animator",
                                "edit text2d animator",
                                tx_tracker,
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
                ui.data_mut(|d| d.insert_temp(expr_open_id, expr_open_set));
                if let Some(target_f) = pending_seek_frame {
                    let _ = session.execute(Command::SetFrame { frame: target_f });
                }
            }
        });
}

// ---------------------------------------------------------------------------
// Track Matte Controls (Clip / Layer Inspector: Video, Image, Shape, Text, Scene)
// ---------------------------------------------------------------------------

const TRACK_MATTE_LABELS: [&str; 5] = ["None", "Alpha", "Alpha Inverted", "Luma", "Luma Inverted"];

const TRACK_MATTE_TOKENS: [Option<&str>; 5] = [
    None,
    Some("alpha"),
    Some("alpha_inverted"),
    Some("luma"),
    Some("luma_inverted"),
];

fn parse_track_matte_mode_idx(val: &Value) -> Option<usize> {
    if let Some(n) = val.as_u64() {
        return (n <= 4).then_some(n as usize);
    }
    if let Some(n) = val.as_i64() {
        return (0..=4).contains(&n).then_some(n as usize);
    }
    if let Some(f) = val.as_f64().filter(|x| x.is_finite()) {
        let r = f.round() as i64;
        return (0..=4).contains(&r).then_some(r as usize);
    }
    if let Some(s) = val.as_str() {
        let norm: String = s
            .trim()
            .to_ascii_lowercase()
            .chars()
            .filter(|c| !matches!(c, ' ' | '-' | '_'))
            .collect();
        return match norm.as_str() {
            "" | "0" | "none" | "off" => Some(0),
            "1" | "alpha" | "alphamatte" => Some(1),
            "2" | "alphainverted" | "alphainv" | "invertedalpha" => Some(2),
            "3" | "luma" | "lumamatte" | "luminance" => Some(3),
            "4" | "lumainverted" | "lumainv" | "invertedluma" | "luminanceinverted" => Some(4),
            _ => None,
        };
    }
    val.get("value").and_then(parse_track_matte_mode_idx)
}

/// Reads the effective Track Matte mode index (`0..=4`) for a visual clip.
pub(super) fn read_clip_track_matte_mode(session: &Session, clip_id: &str) -> usize {
    for prop_key in ["matteType", "trackMatte"] {
        if let Some(prop) = lookup_session_prop_ref(session, clip_id, prop_key) {
            let v = prop
                .keyframes
                .first()
                .map(|k| &k.value)
                .unwrap_or(&prop.default_value);
            if let Some(idx) = parse_track_matte_mode_idx(v)
                && (idx > 0 || prop_key == "matteType")
            {
                return idx;
            }
        }
    }
    if let Some(vc) = find_project_video_clip(session, clip_id) {
        for map in [&vc.preserved, &vc.layer.preserved] {
            for key in ["trackMatte", "matteType", "track_matte", "matte_type"] {
                if let Some(v) = map.get(key)
                    && let Some(idx) = parse_track_matte_mode_idx(v)
                {
                    return idx;
                }
            }
        }
    }
    0
}

/// Reads an explicit matte source video-track index for a visual clip (`None` = Auto / Track Above).
pub(super) fn read_clip_matte_source_index(session: &Session, clip_id: &str) -> Option<usize> {
    let vc = find_project_video_clip(session, clip_id)?;
    let seq = session.active_sequence();
    for map in [&vc.preserved, &vc.layer.preserved] {
        for key in ["matteTrackIndex", "trackMatteIndex", "matte_track_index"] {
            if let Some(val) = map.get(key) {
                if let Some(n) = val.as_u64() {
                    return usize::try_from(n).ok();
                }
                if let Some(n) = val.as_i64()
                    && n >= 0
                {
                    return usize::try_from(n).ok();
                }
                if let Some(f) = val.as_f64()
                    && f.is_finite()
                    && f >= 0.0
                {
                    return Some(f.round() as usize);
                }
                if let Some(s) = val.as_str() {
                    let trimmed = s.trim();
                    if let Ok(idx) = trimmed.parse::<usize>() {
                        return Some(idx);
                    }
                    if let Some(pos) = seq
                        .video_tracks
                        .iter()
                        .position(|t| t.id.to_string() == trimmed || t.name == trimmed)
                    {
                        return Some(pos);
                    }
                }
            }
        }
    }
    None
}

/// Writes Track Matte mode (`matteType` + `trackMatte`) and optional explicit source track index
/// (`matteTrackIndex`) onto a clip in a single undo transaction.
pub(super) fn set_clip_track_matte(
    session: &mut Session,
    clip_id: &str,
    mode_idx: usize,
    source_track_idx: Option<usize>,
) {
    let clamped = mode_idx.min(TRACK_MATTE_LABELS.len() - 1);
    let started_tx = if !session.history.is_in_transaction() {
        session.begin_transaction("Set track matte").is_ok()
    } else {
        false
    };
    let _ = session.execute(Command::SetPropertyValue {
        target_id: clip_id.to_owned(),
        property: "matteType".to_owned(),
        value: json!(clamped),
    });
    let _ = session.execute(Command::SetPropertyValue {
        target_id: clip_id.to_owned(),
        property: "trackMatte".to_owned(),
        value: TRACK_MATTE_TOKENS[clamped].map_or(Value::Null, |tok| json!(tok)),
    });
    if let Ok(clip_uuid) = Uuid::parse_str(clip_id) {
        let matte_idx_val = if clamped == 0 {
            None
        } else {
            source_track_idx.map(|i| json!(i))
        };
        let _ = session.execute(Command::UpdateLayerPreserved {
            clip_id: clip_uuid,
            key: "matteTrackIndex".to_owned(),
            value: matte_idx_val,
        });
    }
    if started_tx && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    }
}

/// Renders `"Track Matte"` mode selector (`None` / `Alpha` / `Alpha Inverted` / `Luma` /
/// `Luma Inverted`) and `"Matte source"` video-track selector for any visual clip.
pub(super) fn show_clip_track_matte_rows(
    ui: &mut Ui,
    session: &mut Session,
    clip_id: &str,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    let own_track_idx = session
        .document()
        .video_tracks
        .iter()
        .position(|t| t.clips.iter().any(|c| c.id == clip_id));
    let Some(own_track_idx) = own_track_idx else {
        return;
    };

    let mut mode_idx = read_clip_track_matte_mode(session, clip_id);
    let current_source = read_clip_matte_source_index(session, clip_id)
        .filter(|&i| i < session.document().video_tracks.len() && i != own_track_idx);

    let mode_changed = cm3_property_row_aligned(
        ui,
        "Track Matte",
        true,
        state.prop_columns,
        theme,
        |_| {},
        |ui| {
            cm3_dropdown(
                ui,
                Id::unique(("clip_track_matte_mode", clip_id)),
                &mut mode_idx,
                &TRACK_MATTE_LABELS,
                false,
                viewport_rect,
                theme,
            )
        },
    );
    if mode_changed {
        set_clip_track_matte(session, clip_id, mode_idx, current_source);
    }

    if mode_idx > 0 {
        let mut source_choices: Vec<(Option<usize>, String)> =
            vec![(None, "Auto (Track Above)".to_owned())];
        for (idx, tr) in session.document().video_tracks.iter().enumerate() {
            if idx == own_track_idx {
                continue;
            }
            let lbl = if tr.name.trim().is_empty() {
                format!("V{}", idx + 1)
            } else {
                format!("V{} — {}", idx + 1, tr.name)
            };
            source_choices.push((Some(idx), lbl));
        }
        let mut src_pick = source_choices
            .iter()
            .position(|(opt_idx, _)| *opt_idx == current_source)
            .unwrap_or(0);
        let src_labels: Vec<&str> = source_choices.iter().map(|(_, s)| s.as_str()).collect();
        let src_changed = cm3_property_row_aligned(
            ui,
            "Matte source",
            true,
            state.prop_columns,
            theme,
            |_| {},
            |ui| {
                cm3_dropdown(
                    ui,
                    Id::unique(("clip_track_matte_source", clip_id)),
                    &mut src_pick,
                    &src_labels,
                    false,
                    viewport_rect,
                    theme,
                )
            },
        );
        if src_changed {
            let next_src = source_choices
                .get(src_pick)
                .and_then(|(opt_idx, _)| *opt_idx);
            set_clip_track_matte(session, clip_id, mode_idx, next_src);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn show_tp2d_number_row(
    ui: &mut Ui,
    slot: &mut serde_json::Map<String, Value>,
    scope_id: &str,
    canonical_key: &str,
    legacy_keys: &[&str],
    label: &str,
    default_val: f64,
    meta: SpNumberMeta,
    local_frame: f64,
    clip_start: f64,
    easing_plus_enabled: bool,
    viewport_rect: Rect,
    theme: &Cm3Theme,
    tx_tracker: &mut WidgetTxTracker,
    expr_open_set: &mut BTreeSet<(String, String)>,
    pending_seek_frame: &mut Option<f64>,
) -> bool {
    let mut prop_val = slot
        .get(canonical_key)
        .cloned()
        .unwrap_or_else(|| json!(tp2d_read_f64(slot, canonical_key, legacy_keys, default_val)));
    let changed = show_sp_number_row(
        ui,
        scope_id,
        canonical_key,
        label,
        meta,
        &mut prop_val,
        local_frame,
        clip_start,
        easing_plus_enabled,
        viewport_rect,
        theme,
        tx_tracker,
        expr_open_set,
        pending_seek_frame,
    );
    if changed {
        slot.insert(canonical_key.to_owned(), prop_val);
    }
    changed
}

#[allow(clippy::too_many_arguments)]
fn show_tp2d_vec2_row(
    ui: &mut Ui,
    slot: &mut serde_json::Map<String, Value>,
    scope_id: &str,
    canonical_key: &str,
    legacy_xy: (&str, &str),
    label: &str,
    default_val: [f64; 2],
    speed: f64,
    local_frame: f64,
    clip_start: f64,
    easing_plus_enabled: bool,
    viewport_rect: Rect,
    theme: &Cm3Theme,
    tx_tracker: &mut WidgetTxTracker,
    expr_open_set: &mut BTreeSet<(String, String)>,
    pending_seek_frame: &mut Option<f64>,
) -> bool {
    let mut prop_val = slot.get(canonical_key).cloned().unwrap_or_else(|| {
        let v = tp2d_read_vec2(slot, canonical_key, legacy_xy, default_val);
        json!([v[0], v[1]])
    });
    let mut dirty = false;
    let cur = sp_eval_vec2(Some(&prop_val), local_frame, default_val);
    let mut x = cur[0];
    let mut y = cur[1];
    let animated = sp_prop_is_animated(&prop_val);
    let has_expr = sp_prop_expression(&prop_val).is_some();
    let expr_key = (scope_id.to_owned(), canonical_key.to_owned());
    let expr_open = expr_open_set.contains(&expr_key);

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let sw = cm3_stopwatch_button(ui, canonical_key, animated, has_expr, theme);
        if sw.toggled {
            sp_prop_toggle_animated(&mut prop_val, local_frame, json!([cur[0], cur[1]]), 1);
            dirty = true;
        }
        if sw.alt_expression && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
        ui.label(
            egui::RichText::new(label)
                .font(theme.font(FONT_SMALL))
                .color(TEXT_PRIMARY),
        );
        let cx = ui.add(egui::DragValue::new(&mut x).speed(speed).prefix("X "));
        let cy = ui.add(egui::DragValue::new(&mut y).speed(speed).prefix("Y "));
        let ch_x = tx_tracker.observe(&cx);
        let ch_y = tx_tracker.observe(&cy);
        if ch_x || ch_y {
            sp_prop_set_value(&mut prop_val, local_frame, json!([x, y]), 1);
            dirty = true;
        }
        if show_sp_fx_button(ui, has_expr, expr_open, theme) && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
    });

    let cur_now = sp_eval_vec2(Some(&prop_val), local_frame, [x, y]);
    if show_sp_kf_and_expr_subrows(
        ui,
        scope_id,
        canonical_key,
        &mut prop_val,
        &json!([cur_now[0], cur_now[1]]),
        1,
        true,
        local_frame,
        clip_start,
        easing_plus_enabled,
        viewport_rect,
        theme,
        expr_open_set,
        pending_seek_frame,
    ) {
        dirty = true;
    }
    if dirty {
        slot.insert(canonical_key.to_owned(), prop_val);
    }
    dirty
}

fn tp2d_eval_order_at_frame(val: Option<&Value>, frame: f64, max_opts: usize) -> usize {
    let Some(v) = val else {
        return 0;
    };
    let max_idx = max_opts.saturating_sub(1);
    let has_kfs = v
        .get("keyframes")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty());
    if !has_kfs
        && let Some(s) = v
            .as_str()
            .or_else(|| v.get("value").and_then(Value::as_str))
    {
        return match s.trim().to_ascii_lowercase().as_str() {
            "1" | "end" | "reverse" => 1.min(max_idx),
            "2" | "middle" | "center out" | "center-out" => 2.min(max_idx),
            "3" | "random" => 3.min(max_idx),
            _ => 0,
        };
    }
    let idx = sp_eval_f64(Some(v), frame, 0.0).round() as i64;
    idx.clamp(0, max_idx as i64) as usize
}

#[allow(clippy::too_many_arguments)]
fn show_tp2d_option_row(
    ui: &mut Ui,
    slot: &mut serde_json::Map<String, Value>,
    scope_id: &str,
    canonical_key: &str,
    label: &str,
    options: &[&str],
    local_frame: f64,
    clip_start: f64,
    easing_plus_enabled: bool,
    viewport_rect: Rect,
    theme: &Cm3Theme,
    expr_open_set: &mut BTreeSet<(String, String)>,
    pending_seek_frame: &mut Option<f64>,
) -> bool {
    let mut prop_val = slot.get(canonical_key).cloned().unwrap_or_else(|| json!(0));
    let mut dirty = false;
    let mut idx = tp2d_eval_order_at_frame(Some(&prop_val), local_frame, options.len());
    let animated = sp_prop_is_animated(&prop_val);
    let has_expr = sp_prop_expression(&prop_val).is_some();
    let expr_key = (scope_id.to_owned(), canonical_key.to_owned());
    let expr_open = expr_open_set.contains(&expr_key);

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let sw = cm3_stopwatch_button(ui, canonical_key, animated, has_expr, theme);
        if sw.toggled {
            sp_prop_toggle_animated(&mut prop_val, local_frame, json!(idx), 0);
            dirty = true;
        }
        if sw.alt_expression && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
        ui.label(
            egui::RichText::new(label)
                .font(theme.font(FONT_SMALL))
                .color(TEXT_PRIMARY),
        );
        if cm3_dropdown(
            ui,
            Id::unique(("tp2d_opt", scope_id, canonical_key)),
            &mut idx,
            options,
            false,
            viewport_rect,
            theme,
        ) {
            sp_prop_set_value(&mut prop_val, local_frame, json!(idx), 0);
            dirty = true;
        }
        if show_sp_fx_button(ui, has_expr, expr_open, theme) && !expr_open_set.remove(&expr_key) {
            expr_open_set.insert(expr_key.clone());
        }
    });

    let cur_idx = tp2d_eval_order_at_frame(Some(&prop_val), local_frame, options.len());
    if show_sp_kf_and_expr_subrows(
        ui,
        scope_id,
        canonical_key,
        &mut prop_val,
        &json!(cur_idx),
        0,
        false,
        local_frame,
        clip_start,
        easing_plus_enabled,
        viewport_rect,
        theme,
        expr_open_set,
        pending_seek_frame,
    ) {
        dirty = true;
    }
    if dirty {
        slot.insert(canonical_key.to_owned(), prop_val);
    }
    dirty
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

#[cfg(test)]
mod shape_plus_ui_tests {
    use super::*;

    #[test]
    fn shape_plus_paint_scope_hints_follow_group_stack_rules() {
        let payload = json!({
            "schemaVersion": 1,
            "renderVersion": 1,
            "nextId": 10,
            "contents": [
                {
                    "id": "g1",
                    "kind": "group",
                    "properties": { "name": "Group 1", "enabled": 1 },
                    "contents": [
                        { "id": "p_top", "kind": "fill", "properties": { "enabled": 1 } },
                        { "id": "r1", "kind": "rectangle", "properties": { "enabled": 1 } },
                        { "id": "f1", "kind": "fill", "properties": { "enabled": 1 } },
                        { "id": "e_unpainted", "kind": "ellipse", "properties": { "enabled": 1 } },
                        { "id": "f_disabled", "kind": "stroke", "properties": { "enabled": 0 } }
                    ]
                },
                {
                    "id": "ta_orphan",
                    "kind": "textAnimator",
                    "properties": { "enabled": 1 }
                },
                {
                    "id": "t1",
                    "kind": "text",
                    "properties": { "enabled": 1 }
                },
                {
                    "id": "ta_ok",
                    "kind": "textAnimator",
                    "properties": { "enabled": 1 }
                },
                {
                    "id": "f_root",
                    "kind": "fill",
                    "properties": { "enabled": 1 }
                }
            ]
        });
        let doc = parse_shape_plus_document(&payload);
        let scope = compute_shape_plus_paint_scope(&doc.contents, 0.0);

        let g1 = &doc.contents[0];
        // p_top is above every geometry item -> "no paths above"
        assert_eq!(
            shape_plus_scope_hint(&scope, &g1.contents[0]),
            Some("no paths above")
        );
        // r1 is painted by f1 below it -> None
        assert_eq!(shape_plus_scope_hint(&scope, &g1.contents[1]), None);
        // f1 sees r1 above it -> None
        assert_eq!(shape_plus_scope_hint(&scope, &g1.contents[2]), None);
        // e_unpainted is painted by root f_root (group paths propagate up to parent paints!)
        assert_eq!(shape_plus_scope_hint(&scope, &g1.contents[3]), None);
        // f_disabled is disabled -> None
        assert_eq!(shape_plus_scope_hint(&scope, &g1.contents[4]), None);
        // ta_orphan has no text above it -> "no text above"
        assert_eq!(
            shape_plus_scope_hint(&scope, &doc.contents[1]),
            Some("no text above")
        );
        // t1 is painted by f_root -> None
        assert_eq!(shape_plus_scope_hint(&scope, &doc.contents[2]), None);
        // ta_ok has t1 above it -> None
        assert_eq!(shape_plus_scope_hint(&scope, &doc.contents[3]), None);
        // f_root sees g1's paths and t1 -> None
        assert_eq!(shape_plus_scope_hint(&scope, &doc.contents[4]), None);

        // Without root fill, e_unpainted has only a disabled stroke below it -> "no fill or stroke"
        let mut payload_no_root_fill = payload.clone();
        payload_no_root_fill["contents"]
            .as_array_mut()
            .unwrap()
            .pop();
        let doc2 = parse_shape_plus_document(&payload_no_root_fill);
        let scope2 = compute_shape_plus_paint_scope(&doc2.contents, 0.0);
        assert_eq!(
            shape_plus_scope_hint(&scope2, &doc2.contents[0].contents[3]),
            Some("no fill or stroke")
        );
    }

    #[test]
    fn shape_plus_keyframe_toggle_insert_sort_and_eval_roundtrip() {
        let mut prop = json!(10.0);
        assert!(!sp_prop_is_animated(&prop));

        // Toggle stopwatch ON at frame 0 -> inserts keyframe at frame 0
        sp_prop_toggle_animated(&mut prop, 0.0, json!(10.0), 1);
        assert!(sp_prop_is_animated(&prop));
        assert_eq!(sp_prop_keyframe_frames(&prop), vec![0.0]);

        // Edit at frame 30 -> appends keyframe at 30
        sp_prop_set_value(&mut prop, 30.0, json!(40.0), 1);
        assert_eq!(sp_prop_keyframe_frames(&prop), vec![0.0, 30.0]);

        // Edit at frame 10 -> inserts in sorted order between 0 and 30
        sp_prop_set_value(&mut prop, 10.0, json!(20.0), 1);
        assert_eq!(sp_prop_keyframe_frames(&prop), vec![0.0, 10.0, 30.0]);

        // Midpoint between 10 (20.0) and 30 (40.0) evaluates to 30.0 via zc_render::shape_plus::eval_prop_f64
        let mid = zc_render::shape_plus::eval_prop_f64(Some(&prop), 20.0, 0.0, None);
        assert!((mid - 30.0).abs() < 1e-6, "expected 30.0, got {mid}");

        // Toggle keyframe OFF at frame 10 removes that keyframe
        sp_prop_toggle_keyframe(&mut prop, 10.0, json!(20.0), 1);
        assert_eq!(sp_prop_keyframe_frames(&prop), vec![0.0, 30.0]);

        // Toggle stopwatch OFF at frame 15 bakes evaluated value (25.0) and clears keyframes
        let baked = zc_render::shape_plus::eval_prop_f64(Some(&prop), 15.0, 0.0, None);
        assert!((baked - 25.0).abs() < 1e-6);
        sp_prop_toggle_animated(&mut prop, 15.0, json!(baked), 1);
        assert!(!sp_prop_is_animated(&prop));
        assert!(sp_prop_keyframe_frames(&prop).is_empty());
        assert!(
            (zc_render::shape_plus::eval_prop_f64(Some(&prop), 0.0, 0.0, None) - 25.0).abs() < 1e-6
        );
    }

    #[test]
    fn shape_plus_expression_set_clear_and_preserves_animation() {
        let mut prop = json!(5.0);
        sp_prop_set_expression(&mut prop, json!(5.0), Some("Math.sin(time) * 50"));
        assert_eq!(sp_prop_expression(&prop), Some("Math.sin(time) * 50"));
        assert!(!sp_prop_is_animated(&prop));

        // Clearing expression on unanimated property unwraps back to plain value
        sp_prop_set_expression(&mut prop, json!(5.0), None);
        assert_eq!(sp_prop_expression(&prop), None);
        assert_eq!(prop, json!(5.0));

        // Setting & clearing expression on an animated property preserves keyframes
        sp_prop_toggle_animated(&mut prop, 0.0, json!(5.0), 1);
        sp_prop_set_value(&mut prop, 20.0, json!(15.0), 1);
        sp_prop_set_expression(&mut prop, json!(5.0), Some("value + 1"));
        assert_eq!(sp_prop_expression(&prop), Some("value + 1"));
        assert!(sp_prop_is_animated(&prop));
        sp_prop_set_expression(&mut prop, json!(5.0), None);
        assert_eq!(sp_prop_expression(&prop), None);
        assert!(sp_prop_is_animated(&prop));
        assert_eq!(sp_prop_keyframe_frames(&prop), vec![0.0, 20.0]);
    }

    #[test]
    fn shape_plus_angle_unit_meta_matches_geometry_and_raster() {
        assert!(shape_plus_number_meta("group", "rotation").is_radians);
        assert!(shape_plus_number_meta("polystar", "rotation").is_radians);
        assert!(shape_plus_number_meta("twist", "angle").is_radians);
        assert!(shape_plus_number_meta("group", "skew").is_radians);
        assert!(shape_plus_number_meta("group", "skewAxis").is_radians);
        assert!(shape_plus_number_meta("gradientFill", "highlightAngle").is_radians);

        let trim_off = shape_plus_number_meta("trim", "offset");
        assert!(!trim_off.is_radians);
        assert_eq!(trim_off.suffix, "°");

        let rep_off = shape_plus_number_meta("repeater", "offset");
        assert!(!rep_off.is_radians);
        assert_eq!(rep_off.suffix, "");
    }

    #[test]
    fn shape_plus_prop_classification_covers_all_types() {
        assert_eq!(
            classify_shape_plus_prop("path", "path", &json!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
            ShapePlusPropKind::Path
        );
        assert_eq!(
            classify_shape_plus_prop("path", "closed", &json!(true)),
            ShapePlusPropKind::Bool
        );
        assert_eq!(
            classify_shape_plus_prop("gradientFill", "stops", &json!([])),
            ShapePlusPropKind::GradientStops
        );
        assert_eq!(
            classify_shape_plus_prop("stroke", "dashes", &json!([10.0, 5.0])),
            ShapePlusPropKind::Dashes
        );
        assert_eq!(
            classify_shape_plus_prop(
                "rectangle",
                "size",
                &json!({ "animated": true, "keyframes": [{ "frame": 0.0, "value": [100.0, 50.0], "tween": 1 }] })
            ),
            ShapePlusPropKind::Vec2
        );
        assert_eq!(
            classify_shape_plus_prop(
                "fill",
                "color",
                &json!({ "value": [1.0, 0.5, 0.25], "animated": false, "keyframes": [] })
            ),
            ShapePlusPropKind::Rgb
        );
    }

    #[test]
    fn text_plus_2d_animated_delay_order_and_track_matte_roundtrip() {
        let mut delay_prop = json!(0.0);
        sp_prop_toggle_animated(&mut delay_prop, 0.0, json!(0.0), 1);
        sp_prop_set_value(&mut delay_prop, 20.0, json!(10.0), 1);
        assert!(sp_prop_is_animated(&delay_prop));
        assert!((sp_eval_f64(Some(&delay_prop), 10.0, 0.0) - 5.0).abs() < 1e-6);

        let mut order_prop = json!("end");
        assert_eq!(tp2d_eval_order_at_frame(Some(&order_prop), 0.0, 4), 1);
        sp_prop_toggle_animated(&mut order_prop, 0.0, json!(1), 0);
        sp_prop_set_value(&mut order_prop, 15.0, json!(3), 0);
        assert_eq!(tp2d_eval_order_at_frame(Some(&order_prop), 5.0, 4), 1);
        assert_eq!(tp2d_eval_order_at_frame(Some(&order_prop), 15.0, 4), 3);

        // Track Matte write + read round-trip on a clip
        let mut session = Session::new();
        let t0 = session.document().video_tracks[0].id.clone();
        let _ = session.execute(Command::AddVideoTrack {
            name: Some("Matte Track".to_owned()),
        });
        let clip_id = session
            .execute(Command::AddClip {
                track_id: t0,
                kind: "text".to_owned(),
                name: Some("Title".to_owned()),
                start: 0.0,
                length: 60.0,
            })
            .expect("add text clip")["clip_id"]
            .as_str()
            .expect("clip_id")
            .to_owned();

        assert_eq!(read_clip_track_matte_mode(&session, &clip_id), 0);
        assert_eq!(read_clip_matte_source_index(&session, &clip_id), None);

        set_clip_track_matte(&mut session, &clip_id, 3, Some(1));
        assert_eq!(read_clip_track_matte_mode(&session, &clip_id), 3);
        assert_eq!(read_clip_matte_source_index(&session, &clip_id), Some(1));

        set_clip_track_matte(&mut session, &clip_id, 0, None);
        assert_eq!(read_clip_track_matte_mode(&session, &clip_id), 0);
        assert_eq!(read_clip_matte_source_index(&session, &clip_id), None);
    }
}
