use super::effects_inspector::{
    collect_shader_image_asset_choices, oversized_text_summary, show_effect_inspector,
    summarize_complex_json_value,
};
use super::media_tab::layout_single_line_ellipsized;
use super::objects_2d::paint_visibility_eye_button;
use super::objects_tab::show_panel_fuzzy_picker;
use super::property_rows::{eval_prop_u32, find_clip_cached, panel_property_row};
use super::shared_widgets::{
    empty_panel_message, find_project_video_clip, humanize_prop_label, paint_crisp_panel_border,
    property_header_with_actions,
};
use super::state::{
    InlineRenameState, TreeDragState, commit_inspector_tx_if_open, copy_pz_json_to_clipboard,
    header_compact_action_button, paste_pz_json_from_clipboard,
};
use super::*;

// ---------------------------------------------------------------------------
// Tab 5: Effects (fx) — 40/60 Auto-Split Effect Stack + Dynamic Inspector
// ---------------------------------------------------------------------------

const fn effect_category_label(cat: EffectCategory) -> &'static str {
    match cat {
        EffectCategory::GlowAndLight => "Glow & Light",
        EffectCategory::BlurAndSharpen => "Blur & Sharpen",
        EffectCategory::ColorAndTone => "Color & Tone",
        EffectCategory::KeyingAndMasking => "Keying & Masking",
        EffectCategory::OverlayAndComposite => "Overlay & Composite",
        EffectCategory::DistortAndTransform => "Distort & Transform",
        EffectCategory::StylizeAndGlitch => "Stylize & Glitch",
        EffectCategory::Temporal => "Temporal",
        EffectCategory::NodeAndGrading => "Node & Grading",
    }
}

pub(super) fn build_effect_picker_items(session: &Session) -> Vec<FuzzyItem> {
    let mut items = Vec::with_capacity(64);

    // Custom Shader & Effect Group (CM3 native)
    items.push(FuzzyItem {
        id: "customshader".into(),
        category: "Custom & Groups".into(),
        name: "Custom Shader".into(),
        desc: "GLSL ES 1.00 fragment shader effect".into(),
    });
    items.push(FuzzyItem {
        id: "group".into(),
        category: "Custom & Groups".into(),
        name: "Effect Group".into(),
        desc: "Macro container grouping multiple effects".into(),
    });

    // All 44 standard CM3 effects
    for desc in standard_effect_catalog() {
        items.push(FuzzyItem {
            id: desc.id.to_owned(),
            category: effect_category_label(desc.category).to_owned(),
            name: desc.display_name.to_owned(),
            desc: format!("Standard ({} pass)", desc.passes),
        });
    }

    // Zoidium plugin effects gated dynamically by enabled plugins (covers all 36 catalog effects,
    // including smart-motion-blur-lite and layerinput without duplicate aliases).
    for desc in zoidium_effect_catalog() {
        let allowed = desc
            .plugin_id
            .is_none_or(|pid| session.plugins.is_enabled(pid));
        if allowed {
            let pid = desc.plugin_id.unwrap_or("zoidium");
            let plugin_label = session
                .plugins
                .get(pid)
                .map_or("Extension", |p| p.name.as_str());
            items.push(FuzzyItem {
                id: format!("zoidium:{pid}/{}", desc.id),
                category: format!("Plugin: {}", effect_category_label(desc.category)),
                name: desc.display_name.to_owned(),
                desc: format!("from {plugin_label}"),
            });
        }
    }

    // Declarative Shader Pack effects gated dynamically by enabled ShaderPack plugins
    // (AfterClip, AlipFX Shader Pack 4, CCFX Shader Pack), grouped under their pack names.
    for pack_fx in session.plugins.enabled_shader_pack_effects() {
        let pack_label = session
            .plugins
            .get(pack_fx.plugin_id)
            .map_or(pack_fx.plugin_id, |p| p.name.as_str());
        items.push(FuzzyItem {
            id: pack_fx.namespaced_id(),
            category: format!("Shader Pack: {pack_label}"),
            name: pack_fx.name.to_owned(),
            desc: format!("{} ({})", pack_fx.description, pack_fx.category),
        });
    }

    items
}

pub(super) fn show_effects_tab(
    ui: &mut Ui,
    panel_rect: Rect,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let Some(clip_id) = session.selection.clip_ids.first().cloned() else {
        empty_panel_message(ui, "select a clip", theme);
        return;
    };
    let rev_before = session.revision();
    let Some(clip) = find_clip_cached(session, state, &clip_id) else {
        empty_panel_message(ui, "select a clip", theme);
        return;
    };

    // Expand the Add Effect picker over the FULL sidebar height when open
    if state.add_effect_open {
        let items = build_effect_picker_items(session);
        match show_panel_fuzzy_picker(
            ui,
            panel_rect,
            Id::unique("add_effect_picker"),
            state.picker_open_seq,
            &items,
            &mut state.add_effect_picker,
            theme,
        ) {
            FuzzyPickerEvent::Cancelled => {
                state.add_effect_open = false;
            }
            FuzzyPickerEvent::Selected(kind) => {
                commit_inspector_tx_if_open(session, state);
                let shader_pack_fx = session.plugins.find_shader_pack_effect(&kind);
                let display = shader_pack_fx
                    .map(|fx| fx.name.to_owned())
                    .or_else(|| items.iter().find(|i| i.id == kind).map(|i| i.name.clone()));
                let started_tx = kind == "customshader"
                    && session.begin_transaction("add custom shader").is_ok();
                if let Ok(res) = session.execute(Command::AddEffect {
                    clip_id: clip_id.clone(),
                    kind: kind.clone(),
                    name: display,
                }) && let Some(fx_id) = res.get("effect_id").and_then(Value::as_str)
                {
                    if kind == "customshader" {
                        let _ = session.execute(Command::SetPropertyValue {
                            target_id: fx_id.to_owned(),
                            property: "fragShader".to_owned(),
                            value: json!(DEFAULT_CUSTOM_SHADER_GLSL),
                        });
                    }
                    let selected_kind = res.get("kind").and_then(Value::as_str).unwrap_or(&kind);
                    let _ = session.selection.select_effect(fx_id, selected_kind, false);
                    ui.ctx().data_mut(|d| {
                        d.insert_temp(
                            Id::unique(("effects_list_selection_anchor", &clip_id)),
                            fx_id.to_owned(),
                        );
                    });
                }
                if started_tx && session.history.is_in_transaction() {
                    let _ = session.commit_transaction(true);
                }
                state.add_effect_open = false;
            }
            FuzzyPickerEvent::None => {}
        }
        return;
    }

    let (top_rect, bot_rect) = begin_split_panel(
        ui,
        Id::unique("effects_40_60_split"),
        panel_rect,
        &mut state.effects_split,
    );
    ui.input(|i| {
        if (i.pointer.primary_pressed() || i.pointer.secondary_pressed())
            && let Some(pos) = i.pointer.interact_pos()
        {
            if top_rect.contains(pos) {
                state.effects_focus_bottom = false;
            } else if bot_rect.contains(pos) {
                state.effects_focus_bottom = true;
            }
        }
    });
    let top_inner = top_rect.shrink(1.0);
    let bot_inner = bot_rect.shrink(1.0);
    ui.painter()
        .rect_filled(top_rect, CornerRadius::ZERO, BG_PANEL);
    ui.painter()
        .rect_filled(bot_rect, CornerRadius::ZERO, BG_PANEL);

    show_split_pane(
        ui,
        Id::unique("effects_40_60_split_p0"),
        top_inner,
        |ui, _top_rect| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;

            // 28px #222222 non-collapsible header bar (27px body + 1px #242424 bottom border),
            // 16px #cccccc "Effects" title, Paste button, and right-aligned `+`.
            // Keeping the primary list non-collapsible prevents collapsing into a 240px blank split pane.
            let (hdr_rect, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), Sense::hover());
            ui.painter()
                .rect_filled(hdr_rect, CornerRadius::ZERO, BG_TITLE);
            let hdr_div_y = (hdr_rect.bottom() - 1.0).round();
            ui.painter().rect_filled(
                Rect::from_min_max(
                    Pos2::new(hdr_rect.left(), hdr_div_y),
                    Pos2::new(hdr_rect.right(), hdr_div_y + 1.0),
                ),
                CornerRadius::ZERO,
                BORDER_ROW_SOFT,
            );
            let add_icon_rect = Rect::from_min_size(
                Pos2::new(hdr_rect.right() - 24.0, (hdr_rect.center().y - 8.0).round()),
                Vec2::splat(15.0),
            );
            let add_rect = add_icon_rect.expand(3.0);
            let paste_w = (ui
                .painter()
                .layout_no_wrap("Paste".to_owned(), theme.font(FONT_COMPACT), TEXT_PRIMARY)
                .size()
                .x
                .ceil()
                + 12.0)
                .max(44.0);
            let paste_rect = Rect::from_min_size(
                Pos2::new(
                    (add_rect.left() - paste_w - 4.0).max(hdr_rect.left() + 68.0),
                    (hdr_rect.center().y - 9.0).round(),
                ),
                Vec2::new(paste_w, 18.0),
            );
            let (paste_clicked, _paste_hovered) = header_compact_action_button(
                ui,
                Id::unique("effects_paste_btn"),
                paste_rect,
                "Paste",
                "Paste Effect / Shader JSON or GLSL (Cmd+V)",
                theme,
            );
            let add_resp = ui
                .interact(add_rect, Id::unique("effects_add_btn"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text("Add effect");
            if add_resp.hovered() {
                ui.painter().rect_stroke(
                    add_rect,
                    CornerRadius::ZERO,
                    Stroke::new(2.0, theme.accent()),
                    StrokeKind::Inside,
                );
            }
            let fx_title = if clip.effects.is_empty() {
                "Effects".to_owned()
            } else {
                format!("Effects ({})", clip.effects.len())
            };
            ui.painter()
                .with_clip_rect(Rect::from_min_max(
                    Pos2::new(hdr_rect.left() + 8.0, hdr_rect.top()),
                    Pos2::new(
                        (paste_rect.left() - 4.0).max(hdr_rect.left() + 12.0),
                        hdr_rect.bottom(),
                    ),
                ))
                .text(
                    Pos2::new(hdr_rect.left() + 10.0, hdr_rect.center().y),
                    Align2::LEFT_CENTER,
                    fx_title,
                    theme.font(FONT_BODY),
                    TEXT_PRIMARY,
                );
            paint_cm3_icon(
                ui.painter(),
                add_icon_rect,
                Cm3Icon::Add,
                if add_resp.hovered() {
                    TEXT_BRIGHT
                } else {
                    TEXT_PRIMARY
                },
            );
            if add_resp.clicked() {
                state.picker_open_seq = state.picker_open_seq.wrapping_add(1);
                state.add_effect_open = true;
                state.add_effect_picker = FuzzyPickerState {
                    just_opened: true,
                    ..FuzzyPickerState::default()
                };
            } else if paste_clicked {
                paste_pz_json_from_clipboard(
                    session,
                    state,
                    actions,
                    "effects",
                    Some(clip_id.clone()),
                    None,
                    None,
                    None,
                    "effect JSON or GLSL",
                );
            }

            // Effect stack rows
            ScrollArea::vertical()
                .id_salt("effects_list_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = Vec2::ZERO;
                    if clip.effects.is_empty() {
                        return;
                    }

                    let mut reorder_req: Option<(String, usize)> = None;
                    let local_frame = session.playback.current_frame - clip.start;
                    let anchor_key = Id::unique(("effects_list_selection_anchor", &clip_id));

                    // Pre-compute visible flattened order (top-level effects + expanded EffectGroup
                    // children) so Shift-click range selection works across all visible rows.
                    let group_children_per_fx: Vec<Vec<EffectGroupChildRow>> = clip
                        .effects
                        .iter()
                        .map(|fx| {
                            if fx.kind == "group" {
                                collect_effect_group_child_rows(
                                    session,
                                    &clip_id,
                                    &fx.id,
                                    &state.collapsed_effect_groups,
                                )
                            } else {
                                Vec::new()
                            }
                        })
                        .collect();
                    let mut visible_effects: Vec<(String, String)> =
                        Vec::with_capacity(clip.effects.len());
                    for (i, fx) in clip.effects.iter().enumerate() {
                        visible_effects.push((fx.id.clone(), fx.kind.clone()));
                        if !group_children_per_fx[i].is_empty()
                            && !state.collapsed_effect_groups.contains(&fx.id)
                        {
                            for child in &group_children_per_fx[i] {
                                visible_effects.push((child.id.clone(), child.kind_id.clone()));
                            }
                        }
                    }

                    for (idx, fx) in clip.effects.iter().enumerate() {
                        let selected = session.selection.effect_ids.contains(&fx.id);
                        let enabled =
                            eval_prop_u32(fx.properties.get("enabled"), local_frame, 1) != 0;

                        let row_w = ui.available_width();
                        let (row_rect, row_resp) =
                            ui.allocate_exact_size(Vec2::new(row_w, 27.0), Sense::click_and_drag());
                        let pointer_in_row = ui
                            .input(|i| i.pointer.hover_pos().is_some_and(|p| row_rect.contains(p)));

                        if selected {
                            ui.painter()
                                .rect_filled(row_rect, CornerRadius::ZERO, theme.accent());
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

                        // If this top-level effect is an EffectGroup with children, draw expand/collapse tree box.
                        let group_children = &group_children_per_fx[idx];
                        let has_group_children = !group_children.is_empty();
                        let group_expanded =
                            has_group_children && !state.collapsed_effect_groups.contains(&fx.id);
                        let mut toggled_group_expand = false;
                        if has_group_children {
                            let tri_center =
                                Pos2::new(row_rect.left() + 18.0, row_rect.center().y + 1.0);
                            let tri_rect = Rect::from_center_size(tri_center, Vec2::splat(15.0));
                            let tri_resp = ui
                                .interact(
                                    tri_rect,
                                    Id::unique(("fx_grp_toggle", &fx.id)),
                                    Sense::click(),
                                )
                                .on_hover_cursor(CursorIcon::PointingHand);
                            paint_cm3_tree_box(
                                ui.painter(),
                                tri_center,
                                group_expanded,
                                if tri_resp.hovered() || selected {
                                    TEXT_BRIGHT
                                } else {
                                    TEXT_MUTED
                                },
                            );
                            if tri_resp.clicked() {
                                toggled_group_expand = true;
                                if group_expanded {
                                    state.collapsed_effect_groups.insert(fx.id.clone());
                                } else {
                                    state.collapsed_effect_groups.remove(&fx.id);
                                }
                            }
                        }

                        // Label or Inline Rename input (35px left indent; Reset/Dup/Remove live in Inspector header)
                        let label_left = row_rect.left() + 35.0;
                        let label_right = (row_rect.right() - 8.0).max(label_left);
                        let label_rect = Rect::from_min_max(
                            Pos2::new(label_left, row_rect.top()),
                            Pos2::new(label_right, row_rect.bottom()),
                        );

                        let is_renaming = state
                            .rename_state
                            .as_ref()
                            .is_some_and(|r| r.target_id == fx.id);
                        if is_renaming {
                            if let Some(rename) = state.rename_state.as_mut() {
                                let edit_id = Id::unique(("fx_rename", &fx.id));
                                let mut commit = false;
                                let mut cancel = false;
                                ui.input(|i| {
                                    if i.key_pressed(Key::Enter) {
                                        commit = true;
                                    } else if i.key_pressed(Key::Escape) {
                                        cancel = true;
                                    }
                                });
                                let mut rename_ui =
                                    ui.new_child(egui::UiBuilder::new().max_rect(label_rect));
                                let resp = Frame::NONE
                                    .fill(BG_INPUT)
                                    .stroke(Stroke::new(1.0, theme.accent()))
                                    .inner_margin(egui::Margin::symmetric(3, 1))
                                    .show(&mut rename_ui, |ui| {
                                        ui.add(
                                            TextEdit::singleline(&mut rename.buffer)
                                                .id(edit_id)
                                                .frame(Frame::NONE)
                                                .font(theme.font(FONT_BODY))
                                                .text_color(theme.value())
                                                .desired_width(label_rect.width()),
                                        )
                                    })
                                    .inner;
                                if rename.just_started {
                                    resp.request_focus();
                                    rename.just_started = false;
                                }
                                if cancel {
                                    state.rename_state = None;
                                } else if commit || resp.lost_focus() {
                                    let trimmed = rename.buffer.trim().to_owned();
                                    if !trimmed.is_empty() && trimmed != fx.name {
                                        let _ = session.execute(Command::SetPropertyValue {
                                            target_id: fx.id.clone(),
                                            property: "name".to_owned(),
                                            value: json!(trimmed),
                                        });
                                    }
                                    state.rename_state = None;
                                }
                            }
                        } else {
                            ui.painter().with_clip_rect(label_rect).text(
                                Pos2::new(label_rect.left(), label_rect.center().y),
                                Align2::LEFT_CENTER,
                                &fx.name,
                                theme.font(FONT_BODY),
                                if !enabled {
                                    TEXT_DISABLED
                                } else if selected {
                                    TEXT_BRIGHT
                                } else {
                                    TEXT_PRIMARY
                                },
                            );
                        }

                        if row_resp.clicked() && !toggled_group_expand {
                            commit_inspector_tx_if_open(session, state);
                            let (shift, ctrl_cmd) = ui.input(|i| {
                                (i.modifiers.shift, i.modifiers.ctrl || i.modifiers.command)
                            });
                            apply_effect_list_click(
                                ui.ctx(),
                                session,
                                &visible_effects,
                                anchor_key,
                                &fx.id,
                                &fx.kind,
                                shift,
                                ctrl_cmd,
                            );
                        }

                        // Right-click context menu (Rename, Copy/Paste PZ JSON, Duplicate, Remove)
                        let ctx_fx_id = fx.id.clone();
                        let ctx_fx_name = fx.name.clone();
                        row_resp.context_menu(|ui| {
                            if ui.button("Rename").clicked() {
                                state.rename_state = Some(InlineRenameState {
                                    target_id: ctx_fx_id.clone(),
                                    buffer: ctx_fx_name.clone(),
                                    just_started: true,
                                });
                                ui.close();
                            }
                            if ui.button("Copy JSON").clicked() {
                                let ids = if selected && session.selection.effect_ids.len() > 1 {
                                    session.selection.effect_ids.clone()
                                } else {
                                    vec![ctx_fx_id.clone()]
                                };
                                let count = ids.len();
                                let res = session.copy_effects_as_pz_json(&clip_id, &ids);
                                copy_pz_json_to_clipboard(
                                    ui.ctx(),
                                    res,
                                    &format!(
                                        "Copied {count} effect{} JSON to clipboard",
                                        if count == 1 { "" } else { "s" }
                                    ),
                                    actions,
                                );
                                ui.close();
                            }
                            if ui.button("Paste JSON").clicked() {
                                paste_pz_json_from_clipboard(
                                    session,
                                    state,
                                    actions,
                                    "effects",
                                    Some(clip_id.clone()),
                                    None,
                                    None,
                                    None,
                                    "effect JSON or GLSL",
                                );
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("Duplicate").clicked() {
                                let _ = session.execute(Command::DuplicateEffect {
                                    clip_id: clip_id.clone(),
                                    effect_id: ctx_fx_id.clone(),
                                });
                                ui.close();
                            }
                            if ui.button("Remove").clicked() {
                                commit_inspector_tx_if_open(session, state);
                                delete_effect_and_select_adjacent(session, &clip_id, &ctx_fx_id);
                                ui.close();
                            }
                        });

                        // Drag-to-reorder effect stack (5px vertical threshold)
                        if row_resp.drag_started()
                            && let Some(pos) = ui.input(|i| i.pointer.interact_pos())
                        {
                            state.effect_drag = Some(TreeDragState {
                                dragging_id: fx.id.clone(),
                                start_y: pos.y,
                                active: false,
                            });
                        }
                        if let Some(drag) = state.effect_drag.as_mut()
                            && let Some(pos) = ui.input(|i| i.pointer.latest_pos())
                        {
                            if !drag.active && (pos.y - drag.start_y).abs() >= 5.0 {
                                drag.active = true;
                            }
                            if drag.active && drag.dragging_id != fx.id && row_rect.contains(pos) {
                                let before = pos.y < row_rect.center().y;
                                let bar_y = if before {
                                    row_rect.top()
                                } else {
                                    row_rect.bottom() - 3.0
                                };
                                ui.painter().rect_filled(
                                    Rect::from_min_size(
                                        Pos2::new(row_rect.left(), bar_y),
                                        Vec2::new(row_rect.width(), 3.0),
                                    ),
                                    CornerRadius::ZERO,
                                    Color32::from_white_alpha(64),
                                );
                                if !ui.input(|i| i.pointer.primary_down()) {
                                    let target_idx = if before { idx } else { idx + 1 };
                                    reorder_req = Some((drag.dragging_id.clone(), target_idx));
                                }
                            }
                        }

                        // Render expanded EffectGroup nested child rows underneath parent
                        if group_expanded {
                            let mut pending_toggle_child: Option<String> = None;
                            let mut pending_delete_child: Option<String> = None;
                            for child in group_children {
                                let child_selected =
                                    session.selection.effect_ids.contains(&child.id);
                                let (c_rect, c_resp) = ui.allocate_exact_size(
                                    Vec2::new(ui.available_width(), 24.0),
                                    Sense::click(),
                                );
                                let c_ptr_in = ui.input(|i| {
                                    i.pointer.hover_pos().is_some_and(|p| c_rect.contains(p))
                                });
                                if child_selected {
                                    ui.painter().rect_filled(
                                        c_rect,
                                        CornerRadius::ZERO,
                                        theme.accent(),
                                    );
                                } else if c_resp.hovered() || c_ptr_in {
                                    ui.painter()
                                        .rect_filled(c_rect, CornerRadius::ZERO, BG_HOVER);
                                }
                                ui.painter().line_segment(
                                    [
                                        Pos2::new(c_rect.left(), c_rect.bottom() - 0.5),
                                        Pos2::new(c_rect.right(), c_rect.bottom() - 0.5),
                                    ],
                                    Stroke::new(1.0, BORDER_ROW_SOFT),
                                );

                                // Right-aligned enable toggle button (eye icon)
                                let eye_rect = Rect::from_min_size(
                                    Pos2::new(
                                        c_rect.right() - 24.0,
                                        (c_rect.center().y - 9.0).round(),
                                    ),
                                    Vec2::splat(18.0),
                                );
                                let mut eye_ui =
                                    ui.new_child(egui::UiBuilder::new().max_rect(eye_rect));
                                let eye_clicked = paint_visibility_eye_button(
                                    &mut eye_ui,
                                    Id::unique(("fx_grp_child_en", &child.id)),
                                    child.enabled,
                                    theme,
                                );
                                if eye_clicked {
                                    pending_toggle_child = Some(child.id.clone());
                                }

                                let indent = 35.0 + (child.depth as f32) * 14.0;
                                let mut child_toggled_expand = false;
                                if child.has_children {
                                    let tri_c = Pos2::new(
                                        c_rect.left() + indent - 12.0,
                                        c_rect.center().y + 1.0,
                                    );
                                    let tri_r = Rect::from_center_size(tri_c, Vec2::splat(14.0));
                                    let tri_resp = ui
                                        .interact(
                                            tri_r,
                                            Id::unique(("fx_grp_sub_toggle", &child.id)),
                                            Sense::click(),
                                        )
                                        .on_hover_cursor(CursorIcon::PointingHand);
                                    paint_cm3_tree_box(
                                        ui.painter(),
                                        tri_c,
                                        !state.collapsed_effect_groups.contains(&child.id),
                                        if tri_resp.hovered() || child_selected {
                                            TEXT_BRIGHT
                                        } else {
                                            TEXT_MUTED
                                        },
                                    );
                                    if tri_resp.clicked() {
                                        child_toggled_expand = true;
                                        if state.collapsed_effect_groups.contains(&child.id) {
                                            state.collapsed_effect_groups.remove(&child.id);
                                        } else {
                                            state.collapsed_effect_groups.insert(child.id.clone());
                                        }
                                    }
                                }

                                let branch = if child.is_last { "└─ " } else { "├─ " };
                                let child_label = format!("{branch}{}", child.name);
                                let c_label_left = c_rect.left() + indent;
                                let c_label_right =
                                    (eye_rect.left() - 4.0).max(c_label_left + 12.0);
                                let c_label_rect = Rect::from_min_max(
                                    Pos2::new(c_label_left, c_rect.top()),
                                    Pos2::new(c_label_right, c_rect.bottom()),
                                );
                                let c_color = if !child.enabled {
                                    TEXT_DISABLED
                                } else if child_selected {
                                    TEXT_BRIGHT
                                } else {
                                    TEXT_PRIMARY
                                };
                                let c_galley = layout_single_line_ellipsized(
                                    ui.painter(),
                                    &child_label,
                                    theme.font(FONT_SMALL),
                                    c_color,
                                    c_label_rect.width(),
                                );
                                ui.painter().with_clip_rect(c_label_rect).galley(
                                    Pos2::new(
                                        c_label_rect.left(),
                                        c_label_rect.center().y - c_galley.size().y * 0.5,
                                    ),
                                    c_galley,
                                    c_color,
                                );

                                if c_resp.clicked() && !eye_clicked && !child_toggled_expand {
                                    commit_inspector_tx_if_open(session, state);
                                    let (shift, ctrl_cmd) = ui.input(|i| {
                                        (i.modifiers.shift, i.modifiers.ctrl || i.modifiers.command)
                                    });
                                    apply_effect_list_click(
                                        ui.ctx(),
                                        session,
                                        &visible_effects,
                                        anchor_key,
                                        &child.id,
                                        &child.kind_id,
                                        shift,
                                        ctrl_cmd,
                                    );
                                }

                                let ctx_cid = child.id.clone();
                                let ctx_enabled = child.enabled;
                                c_resp.context_menu(|ui| {
                                    if ui
                                        .button(if ctx_enabled { "Disable" } else { "Enable" })
                                        .clicked()
                                    {
                                        pending_toggle_child = Some(ctx_cid.clone());
                                        ui.close();
                                    }
                                    if ui.button("Copy JSON").clicked() {
                                        let res = session.copy_effects_as_pz_json(
                                            &clip_id,
                                            std::slice::from_ref(&ctx_cid),
                                        );
                                        copy_pz_json_to_clipboard(
                                            ui.ctx(),
                                            res,
                                            "Copied effect JSON to clipboard",
                                            actions,
                                        );
                                        ui.close();
                                    }
                                    ui.separator();
                                    if ui.button("Remove").clicked() {
                                        pending_delete_child = Some(ctx_cid.clone());
                                        ui.close();
                                    }
                                });
                            }
                            if let Some(cid) = pending_toggle_child {
                                mutate_effect_group_child(
                                    session,
                                    &clip_id,
                                    &cid,
                                    "toggle grouped effect",
                                    |eff| {
                                        let cur = effect_track_bool(&eff.enabled, true);
                                        let next = if cur { 0 } else { 1 };
                                        eff.enabled.value = PropertyValue::Option(next);
                                        if let Some(kf) = eff.enabled.keyframes.first_mut() {
                                            kf.value = PropertyValue::Option(next);
                                        }
                                    },
                                );
                            }
                            if let Some(cid) = pending_delete_child {
                                commit_inspector_tx_if_open(session, state);
                                delete_effect_group_child(session, &clip_id, &cid);
                            }
                        }
                    }

                    if !ui.input(|i| i.pointer.primary_down()) {
                        if let Some((fx_id, mut target_idx)) = reorder_req {
                            if let Some(src_idx) = clip.effects.iter().position(|e| e.id == fx_id)
                                && src_idx < target_idx
                            {
                                target_idx -= 1;
                            }
                            let _ = session.execute(Command::ReorderEffect {
                                clip_id: clip_id.clone(),
                                effect_id: fx_id,
                                index: target_idx,
                            });
                        }
                        state.effect_drag = None;
                    }
                });
        },
    );
    // Crisp single 1px #333333 border around top pane (layout.rs paints the focus accent only when SideInspector is focused)
    paint_crisp_panel_border(ui.painter(), top_rect, BORDER_PANEL);
    // Only re-fetch clip snapshot if pane 0 actually mutated the session (avoids a second full
    // clip clone on every 60Hz frame).
    let clip_for_inspector = if session.revision() != rev_before {
        find_clip_cached(session, state, &clip_id).unwrap_or(clip)
    } else {
        clip
    };
    show_split_pane(
        ui,
        Id::unique("effects_40_60_split_p1"),
        bot_inner,
        |ui, bot_rect| {
            // Pane 1 (60%): Dynamic Effect Parameter Inspector
            show_effect_inspector(
                ui,
                bot_rect,
                viewport_rect,
                &clip_for_inspector,
                session,
                theme,
                state,
                actions,
            );
        },
    );
    // Crisp single 1px #333333 border around bottom pane (layout.rs paints the focus accent only when SideInspector is focused)
    paint_crisp_panel_border(ui.painter(), bot_rect, BORDER_PANEL);
}

fn sync_effect_base_type(session: &mut Session, visible_order: &[(String, String)]) {
    let mut shared_kind: Option<&str> = None;
    let mut mixed = false;
    for sel_id in &session.selection.effect_ids {
        if let Some((_, kind)) = visible_order.iter().find(|(id, _)| id == sel_id) {
            match shared_kind {
                None => shared_kind = Some(kind.as_str()),
                Some(prev) if prev != kind.as_str() => {
                    mixed = true;
                    break;
                }
                _ => {}
            }
        }
    }
    session.selection.effect_base_type = if mixed {
        None
    } else {
        shared_kind.map(str::to_owned)
    };
}

#[allow(clippy::too_many_arguments)]
fn apply_effect_list_click(
    ctx: &egui::Context,
    session: &mut Session,
    visible_order: &[(String, String)],
    anchor_key: Id,
    clicked_id: &str,
    clicked_kind: &str,
    shift: bool,
    ctrl_cmd: bool,
) {
    if shift {
        let clicked_pos = visible_order.iter().position(|(id, _)| id == clicked_id);
        let stored_anchor = ctx.data(|d| d.get_temp::<String>(anchor_key));
        let anchor_id = stored_anchor
            .filter(|aid| visible_order.iter().any(|(id, _)| id == aid))
            .or_else(|| {
                session
                    .selection
                    .effect_ids
                    .last()
                    .filter(|aid| visible_order.iter().any(|(id, _)| id == aid.as_str()))
                    .cloned()
            });
        let anchor_pos = anchor_id
            .as_deref()
            .and_then(|aid| visible_order.iter().position(|(id, _)| id == aid));

        if let (Some(a_idx), Some(c_idx)) = (anchor_pos, clicked_pos) {
            let lo = a_idx.min(c_idx);
            let hi = a_idx.max(c_idx);
            let range = &visible_order[lo..=hi];
            if !ctrl_cmd {
                session.selection.effect_ids = range.iter().map(|(id, _)| id.clone()).collect();
            } else {
                for (id, _) in range {
                    if !session.selection.effect_ids.contains(id) {
                        session.selection.effect_ids.push(id.clone());
                    }
                }
            }
            sync_effect_base_type(session, visible_order);
            session.selection.clear_objects_3d();
            session.selection.clear_keyframes();
            if let Some(aid) = anchor_id {
                ctx.data_mut(|d| d.insert_temp(anchor_key, aid));
            }
            return;
        }

        let _ = session
            .selection
            .select_effect(clicked_id, clicked_kind, false);
        ctx.data_mut(|d| d.insert_temp(anchor_key, clicked_id.to_owned()));
        return;
    }

    if ctrl_cmd {
        if let Some(pos) = session
            .selection
            .effect_ids
            .iter()
            .position(|id| id == clicked_id)
        {
            session.selection.effect_ids.remove(pos);
        } else {
            session.selection.effect_ids.push(clicked_id.to_owned());
            session.selection.clear_objects_3d();
            session.selection.clear_keyframes();
            ctx.data_mut(|d| d.insert_temp(anchor_key, clicked_id.to_owned()));
        }
        sync_effect_base_type(session, visible_order);
        return;
    }

    let _ = session
        .selection
        .select_effect(clicked_id, clicked_kind, false);
    ctx.data_mut(|d| d.insert_temp(anchor_key, clicked_id.to_owned()));
}

pub(super) const DEFAULT_CUSTOM_SHADER_GLSL: &str = r#"uniform sampler2D tDiffuse;
uniform vec2 resolution;
varying vec2 vUv;
varying vec2 vUvScaled;

void main() {
    gl_FragColor = texture2D(tDiffuse, vUvScaled);
}
"#;

#[derive(Debug, Clone)]
pub(super) struct EffectGroupChildRow {
    pub(super) id: String,
    pub(super) kind_id: String,
    pub(super) name: String,
    pub(super) enabled: bool,
    pub(super) depth: usize,
    pub(super) is_last: bool,
    pub(super) has_children: bool,
}

fn effect_instance_kind_id(kind: &EffectKind) -> String {
    match kind {
        EffectKind::Standard(id) => id.clone(),
        EffectKind::CustomShader { .. } => "customshader".to_owned(),
        EffectKind::EffectGroup { .. } => "group".to_owned(),
        EffectKind::Plugin {
            plugin_id,
            effect_id,
        } => format!("zoidium:{plugin_id}/{effect_id}"),
    }
}

fn effect_track_bool(track: &PropertyTrack, default: bool) -> bool {
    match &track.value {
        PropertyValue::Bool(b) => *b,
        PropertyValue::Option(o) => *o != 0,
        PropertyValue::Number(n) => *n != 0.0,
        _ => default,
    }
}

pub(super) fn collect_effect_group_child_rows(
    session: &Session,
    clip_id: &str,
    group_effect_id: &str,
    collapsed: &BTreeSet<String>,
) -> Vec<EffectGroupChildRow> {
    fn push_children(
        children: &[EffectInstance],
        depth: usize,
        collapsed: &BTreeSet<String>,
        out: &mut Vec<EffectGroupChildRow>,
    ) {
        let count = children.len();
        for (i, child) in children.iter().enumerate() {
            let cid = child.id.to_string();
            let sub_children = match &child.kind {
                EffectKind::EffectGroup { children, .. } => children.as_slice(),
                _ => &[],
            };
            let has_children = !sub_children.is_empty();
            out.push(EffectGroupChildRow {
                id: cid.clone(),
                kind_id: effect_instance_kind_id(&child.kind),
                name: child.name.clone(),
                enabled: effect_track_bool(&child.enabled, true),
                depth,
                is_last: i + 1 == count,
                has_children,
            });
            if has_children && !collapsed.contains(&cid) {
                push_children(sub_children, depth + 1, collapsed, out);
            }
        }
    }

    let Some(vc) = find_project_video_clip(session, clip_id) else {
        return Vec::new();
    };
    let Some(group_fx) = vc
        .layer
        .effects
        .iter()
        .find(|e| e.id.to_string() == group_effect_id)
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let EffectKind::EffectGroup { children, .. } = &group_fx.kind {
        push_children(children, 1, collapsed, &mut out);
    }
    out
}

pub(super) fn find_effect_group_child(
    session: &Session,
    clip_id: &str,
    child_id: &str,
) -> Option<EffectInstance> {
    fn search(effects: &[EffectInstance], target: &str) -> Option<EffectInstance> {
        for fx in effects {
            if fx.id.to_string() == target {
                return Some(fx.clone());
            }
            if let EffectKind::EffectGroup { children, .. } = &fx.kind
                && let Some(found) = search(children, target)
            {
                return Some(found);
            }
        }
        None
    }

    let vc = find_project_video_clip(session, clip_id)?;
    for fx in &vc.layer.effects {
        if let EffectKind::EffectGroup { children, .. } = &fx.kind
            && let Some(found) = search(children, child_id)
        {
            return Some(found);
        }
    }
    None
}

fn mutate_effect_group_child(
    session: &mut Session,
    clip_id: &str,
    child_id: &str,
    label: &str,
    f: impl FnOnce(&mut EffectInstance),
) -> bool {
    fn find_mut<'a>(
        effects: &'a mut [EffectInstance],
        target: &str,
    ) -> Option<&'a mut EffectInstance> {
        for fx in effects {
            if fx.id.to_string() == target {
                return Some(fx);
            }
            if let EffectKind::EffectGroup { children, .. } = &mut fx.kind
                && let Some(found) = find_mut(children, target)
            {
                return Some(found);
            }
        }
        None
    }

    let Ok(clip_uuid) = Uuid::parse_str(clip_id) else {
        return false;
    };
    let started_tx = if !session.history.is_in_transaction() {
        session.begin_transaction(label).is_ok()
    } else {
        false
    };
    let mut mutated = false;
    'outer: for track in &mut session.active_sequence_mut().video_tracks {
        for vc in &mut track.clips {
            if vc.id == clip_uuid {
                for fx in &mut vc.layer.effects {
                    if let EffectKind::EffectGroup { children, .. } = &mut fx.kind
                        && let Some(target) = find_mut(children, child_id)
                    {
                        f(target);
                        mutated = true;
                        break 'outer;
                    }
                }
            }
        }
    }
    if mutated {
        session.playback.bump_epoch();
    }
    if started_tx {
        let _ = session.commit_transaction(mutated);
    }
    mutated
}

fn regenerate_grouped_effect_ids(eff: &mut EffectInstance) {
    eff.id = Uuid::new_v4();
    if let EffectKind::EffectGroup { children, .. } = &mut eff.kind {
        for child in children.iter_mut() {
            regenerate_grouped_effect_ids(child);
        }
    }
}

/// Resets an Effect Group child's parameters back to its template defaults inside an undoable transaction.
pub(super) fn reset_effect_group_child(
    session: &mut Session,
    clip_id: &str,
    child_id: &str,
) -> bool {
    mutate_effect_group_child(session, clip_id, child_id, "reset grouped effect", |eff| {
        eff.enabled = PropertyTrack::option(1);
        let template = match &eff.kind {
            EffectKind::Standard(id) => create_effect_by_id(id),
            EffectKind::Plugin {
                plugin_id,
                effect_id,
            } => create_effect_by_id(&format!("zoidium:{plugin_id}/{effect_id}"))
                .or_else(|| create_effect_by_id(effect_id)),
            EffectKind::CustomShader { .. } | EffectKind::EffectGroup { .. } => None,
        };
        if let Some(tpl) = template {
            eff.params = tpl.params;
        } else {
            for tr in eff.params.values_mut() {
                tr.animated = false;
                tr.expression = None;
                tr.keyframes.clear();
                tr.channels.clear();
            }
            match &mut eff.kind {
                EffectKind::CustomShader {
                    custom_properties, ..
                }
                | EffectKind::EffectGroup {
                    custom_properties, ..
                } => {
                    for cp in custom_properties.iter_mut() {
                        cp.track.animated = false;
                        cp.track.expression = None;
                        cp.track.keyframes.clear();
                        cp.track.channels.clear();
                    }
                }
                _ => {}
            }
        }
    })
}

/// Clones an Effect Group child (regenerating UUIDs recursively) directly after the original inside
/// the same Effect Group, selecting the newly created clone in an undoable transaction.
pub(super) fn duplicate_effect_group_child(
    session: &mut Session,
    clip_id: &str,
    child_id: &str,
) -> Option<String> {
    fn dup_rec(effects: &mut Vec<EffectInstance>, target: &str) -> Option<(String, String)> {
        if let Some(pos) = effects.iter().position(|e| e.id.to_string() == target) {
            let mut cloned = effects[pos].clone();
            regenerate_grouped_effect_ids(&mut cloned);
            let new_id = cloned.id.to_string();
            let new_kind = effect_instance_kind_id(&cloned.kind);
            effects.insert(pos + 1, cloned);
            return Some((new_id, new_kind));
        }
        for fx in effects.iter_mut() {
            if let EffectKind::EffectGroup { children, .. } = &mut fx.kind
                && let Some(found) = dup_rec(children, target)
            {
                return Some(found);
            }
        }
        None
    }

    let clip_uuid = Uuid::parse_str(clip_id).ok()?;
    let started_tx = if !session.history.is_in_transaction() {
        session
            .begin_transaction("duplicate grouped effect")
            .is_ok()
    } else {
        false
    };
    let mut created: Option<(String, String)> = None;
    'outer: for track in &mut session.active_sequence_mut().video_tracks {
        for vc in &mut track.clips {
            if vc.id == clip_uuid {
                for fx in &mut vc.layer.effects {
                    if let EffectKind::EffectGroup { children, .. } = &mut fx.kind
                        && let Some(res) = dup_rec(children, child_id)
                    {
                        created = Some(res);
                        break 'outer;
                    }
                }
            }
        }
    }
    if let Some((ref new_id, ref new_kind)) = created {
        session.playback.bump_epoch();
        let _ = session.selection.select_effect(new_id, new_kind, false);
    }
    if started_tx {
        let _ = session.commit_transaction(created.is_some());
    }
    created.map(|(id, _)| id)
}

/// Deletes a top-level effect on `clip_id`. When the deleted effect was selected and other
/// effects remain on that clip, automatically selects the next (or previous) remaining effect
/// instead of clearing selection to empty.
pub(super) fn delete_effect_and_select_adjacent(
    session: &mut Session,
    clip_id: &str,
    effect_id: &str,
) {
    let was_selected = session
        .selection
        .effect_ids
        .iter()
        .any(|id| id == effect_id);
    let adjacent = if was_selected {
        session
            .document()
            .video_tracks
            .iter()
            .flat_map(|t| t.clips.iter())
            .find(|c| c.id == clip_id)
            .and_then(|c| {
                let pos = c.effects.iter().position(|e| e.id == effect_id)?;
                let next_fx = if pos + 1 < c.effects.len() {
                    Some(&c.effects[pos + 1])
                } else if pos > 0 {
                    Some(&c.effects[pos - 1])
                } else {
                    None
                }?;
                Some((next_fx.id.clone(), next_fx.kind.clone()))
            })
    } else {
        None
    };

    let _ = session.execute(Command::DeleteEffect {
        clip_id: clip_id.to_owned(),
        effect_id: effect_id.to_owned(),
    });

    if was_selected
        && session.selection.effect_ids.is_empty()
        && let Some((next_id, next_kind)) = adjacent
    {
        let _ = session.selection.select_effect(&next_id, &next_kind, false);
    }
}

/// Reads the `custom_properties` of an `EffectGroup` (top-level or nested) from `session.project`.
pub(super) fn find_effect_group_custom_properties(
    session: &Session,
    clip_id: &str,
    group_effect_id: &str,
) -> Vec<CustomProperty> {
    fn search(effects: &[EffectInstance], target: &str) -> Option<Vec<CustomProperty>> {
        for fx in effects {
            if fx.id.to_string() == target {
                if let EffectKind::EffectGroup {
                    custom_properties, ..
                } = &fx.kind
                {
                    return Some(custom_properties.clone());
                }
                return None;
            }
            if let EffectKind::EffectGroup { children, .. } = &fx.kind
                && let Some(found) = search(children, target)
            {
                return Some(found);
            }
        }
        None
    }
    let Some(vc) = find_project_video_clip(session, clip_id) else {
        return Vec::new();
    };
    search(&vc.layer.effects, group_effect_id).unwrap_or_default()
}

/// Updates one `CustomProperty`'s value on an `EffectGroup` (top-level or nested) inside an undoable transaction.
pub(super) fn update_effect_group_custom_property(
    session: &mut Session,
    clip_id: &str,
    group_effect_id: &str,
    prop_name: &str,
    new_val: PropertyValue,
) -> bool {
    fn find_mut<'a>(
        effects: &'a mut [EffectInstance],
        target: &str,
    ) -> Option<&'a mut EffectInstance> {
        for fx in effects {
            if fx.id.to_string() == target {
                return Some(fx);
            }
            if let EffectKind::EffectGroup { children, .. } = &mut fx.kind
                && let Some(found) = find_mut(children, target)
            {
                return Some(found);
            }
        }
        None
    }

    let Ok(clip_uuid) = Uuid::parse_str(clip_id) else {
        return false;
    };
    let started_tx = if !session.history.is_in_transaction() {
        session
            .begin_transaction("edit effect group custom property")
            .is_ok()
    } else {
        false
    };
    let mut mutated = false;
    'outer: for track in &mut session.active_sequence_mut().video_tracks {
        for vc in &mut track.clips {
            if vc.id == clip_uuid
                && let Some(target) = find_mut(&mut vc.layer.effects, group_effect_id)
                && let EffectKind::EffectGroup {
                    custom_properties, ..
                } = &mut target.kind
                && let Some(cp) = custom_properties.iter_mut().find(|c| c.name == prop_name)
            {
                cp.track.value = new_val.clone();
                if let Some(kf) = cp.track.keyframes.first_mut() {
                    kf.value = new_val;
                }
                mutated = true;
                break 'outer;
            }
        }
    }
    if mutated {
        session.playback.bump_epoch();
    }
    if started_tx {
        let _ = session.commit_transaction(mutated);
    }
    mutated
}

#[allow(clippy::too_many_arguments)]
pub(super) fn render_effect_group_custom_properties(
    ui: &mut Ui,
    session: &mut Session,
    clip_id: &str,
    group_effect_id: &str,
    custom_props: &[CustomProperty],
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let mut updates: Vec<(String, PropertyValue)> = Vec::new();
    for cp in custom_props {
        let lbl = humanize_prop_label(&cp.name);
        match &cp.track.value {
            PropertyValue::Number(v) => {
                let mut cur = *v;
                let ev = panel_property_row(
                    ui,
                    &lbl,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_scrub_number(
                            ui,
                            Id::unique(("fx_grp_cp_num", group_effect_id, &cp.name)),
                            &mut cur,
                            ScrubConfig::new(-100_000.0, 100_000.0, 0.05, 2),
                            theme,
                        )
                    },
                );
                if ev.changed {
                    updates.push((cp.name.clone(), PropertyValue::Number(cur)));
                }
            }
            PropertyValue::Bool(b) => {
                let mut idx = usize::from(*b);
                let changed = panel_property_row(
                    ui,
                    &lbl,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_dropdown(
                            ui,
                            Id::unique(("fx_grp_cp_bool", group_effect_id, &cp.name)),
                            &mut idx,
                            &["off", "on"],
                            false,
                            viewport_rect,
                            theme,
                        )
                    },
                );
                if changed {
                    updates.push((cp.name.clone(), PropertyValue::Bool(idx != 0)));
                }
            }
            PropertyValue::Option(o) => {
                let mut cur = f64::from(*o);
                let ev = panel_property_row(
                    ui,
                    &lbl,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_scrub_number(
                            ui,
                            Id::unique(("fx_grp_cp_opt", group_effect_id, &cp.name)),
                            &mut cur,
                            ScrubConfig::new(0.0, 64.0, 1.0, 0),
                            theme,
                        )
                    },
                );
                if ev.changed {
                    updates.push((cp.name.clone(), PropertyValue::Option(cur.round() as u32)));
                }
            }
            PropertyValue::Vec2(v) => {
                let mut cur = *v;
                let ev = panel_property_row(
                    ui,
                    &lbl,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_vector2(
                            ui,
                            Id::unique(("fx_grp_cp_v2", group_effect_id, &cp.name)),
                            &mut cur,
                            ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 2),
                            false,
                            theme,
                        )
                    },
                );
                if ev.changed {
                    updates.push((cp.name.clone(), PropertyValue::Vec2(cur)));
                }
            }
            PropertyValue::Vec3(v) => {
                let mut cur = *v;
                let ev = panel_property_row(
                    ui,
                    &lbl,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_vector3(
                            ui,
                            Id::unique(("fx_grp_cp_v3", group_effect_id, &cp.name)),
                            &mut cur,
                            ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 2),
                            false,
                            theme,
                        )
                    },
                );
                if ev.changed {
                    updates.push((cp.name.clone(), PropertyValue::Vec3(cur)));
                }
            }
            PropertyValue::Color(c) => {
                let mut rgba = Rgba::new(c[0] as f32, c[1] as f32, c[2] as f32, 1.0);
                let cev = panel_property_row(
                    ui,
                    &lbl,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_hsva_color_picker(
                            ui,
                            Id::unique(("fx_grp_cp_col", group_effect_id, &cp.name)),
                            &mut rgba,
                            false,
                            false,
                            viewport_rect,
                            theme,
                        )
                    },
                );
                if cev.changed {
                    updates.push((
                        cp.name.clone(),
                        PropertyValue::Color([
                            f64::from(rgba.0[0]),
                            f64::from(rgba.0[1]),
                            f64::from(rgba.0[2]),
                        ]),
                    ));
                }
                if cev.eyedropper_clicked {
                    actions.push(PanelAction::EyedropperActivate {
                        target_id: group_effect_id.to_owned(),
                        property: cp.name.clone(),
                    });
                }
            }
            PropertyValue::ColorRgba(c) => {
                let mut rgba = Rgba::new(c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32);
                let cev = panel_property_row(
                    ui,
                    &lbl,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_hsva_color_picker(
                            ui,
                            Id::unique(("fx_grp_cp_rgba", group_effect_id, &cp.name)),
                            &mut rgba,
                            true,
                            false,
                            viewport_rect,
                            theme,
                        )
                    },
                );
                if cev.changed {
                    updates.push((
                        cp.name.clone(),
                        PropertyValue::ColorRgba([
                            f64::from(rgba.0[0]),
                            f64::from(rgba.0[1]),
                            f64::from(rgba.0[2]),
                            f64::from(rgba.0[3]),
                        ]),
                    ));
                }
                if cev.eyedropper_clicked {
                    actions.push(PanelAction::EyedropperActivate {
                        target_id: group_effect_id.to_owned(),
                        property: cp.name.clone(),
                    });
                }
            }
            PropertyValue::Text(s) => {
                let mut buf = s.clone();
                let changed = panel_property_row(
                    ui,
                    &lbl,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        ui.add(
                            TextEdit::singleline(&mut buf)
                                .font(theme.font(FONT_SMALL))
                                .desired_width(ui.available_width()),
                        )
                        .changed()
                    },
                );
                if changed {
                    updates.push((cp.name.clone(), PropertyValue::Text(buf)));
                }
            }
            PropertyValue::Asset(opt_id) => {
                let cur_id = opt_id.as_deref().unwrap_or("");
                let (labels, ids, mut sel) = collect_shader_image_asset_choices(session, cur_id);
                let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
                let changed = panel_property_row(
                    ui,
                    &lbl,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_dropdown(
                            ui,
                            Id::unique(("fx_grp_cp_asset", group_effect_id, &cp.name)),
                            &mut sel,
                            &label_refs,
                            false,
                            viewport_rect,
                            theme,
                        )
                    },
                );
                if changed && let Some(chosen) = ids.get(sel) {
                    let next = if chosen.is_empty() {
                        None
                    } else {
                        Some(chosen.clone())
                    };
                    updates.push((cp.name.clone(), PropertyValue::Asset(next)));
                }
            }
            PropertyValue::Gradient(stops) => {
                panel_property_row(
                    ui,
                    &lbl,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        ui.label(
                            egui::RichText::new(format!("Gradient ({} stops)", stops.len()))
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_MUTED),
                        );
                    },
                );
            }
            PropertyValue::Json(v) | PropertyValue::Curve(v) => {
                let summary = summarize_complex_json_value(v);
                panel_property_row(
                    ui,
                    &lbl,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        ui.label(
                            egui::RichText::new(summary)
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_MUTED),
                        );
                    },
                );
            }
            PropertyValue::Vec4(c) => {
                let mut rgba = Rgba::new(c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32);
                let cev = panel_property_row(
                    ui,
                    &lbl,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_hsva_color_picker(
                            ui,
                            Id::unique(("fx_grp_cp_v4", group_effect_id, &cp.name)),
                            &mut rgba,
                            true,
                            false,
                            viewport_rect,
                            theme,
                        )
                    },
                );
                if cev.changed {
                    updates.push((
                        cp.name.clone(),
                        PropertyValue::Vec4([
                            f64::from(rgba.0[0]),
                            f64::from(rgba.0[1]),
                            f64::from(rgba.0[2]),
                            f64::from(rgba.0[3]),
                        ]),
                    ));
                }
                if cev.eyedropper_clicked {
                    actions.push(PanelAction::EyedropperActivate {
                        target_id: group_effect_id.to_owned(),
                        property: cp.name.clone(),
                    });
                }
            }
            PropertyValue::List(s) | PropertyValue::Shader(s) => {
                let preview = if s.len() > 36 {
                    format!("{}…", s.chars().take(36).collect::<String>())
                } else {
                    s.clone()
                };
                panel_property_row(
                    ui,
                    &lbl,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        ui.label(
                            egui::RichText::new(preview)
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_MUTED),
                        );
                    },
                );
            }
        }
    }
    for (name, val) in updates {
        update_effect_group_custom_property(session, clip_id, group_effect_id, &name, val);
    }
}

fn delete_effect_group_child(session: &mut Session, clip_id: &str, child_id: &str) -> bool {
    fn remove_rec(
        effects: &mut Vec<EffectInstance>,
        target: &str,
        fallback_parent: Option<(String, String)>,
    ) -> (bool, Option<(String, String)>) {
        if let Some(pos) = effects.iter().position(|e| e.id.to_string() == target) {
            effects.remove(pos);
            let adj = if pos < effects.len() {
                Some((
                    effects[pos].id.to_string(),
                    effect_instance_kind_id(&effects[pos].kind),
                ))
            } else if pos > 0 {
                Some((
                    effects[pos - 1].id.to_string(),
                    effect_instance_kind_id(&effects[pos - 1].kind),
                ))
            } else {
                fallback_parent
            };
            return (true, adj);
        }
        for fx in effects.iter_mut() {
            let parent_info = Some((fx.id.to_string(), effect_instance_kind_id(&fx.kind)));
            if let EffectKind::EffectGroup { children, .. } = &mut fx.kind {
                let (removed, adj) = remove_rec(children, target, parent_info);
                if removed {
                    return (true, adj);
                }
            }
        }
        (false, None)
    }

    let Ok(clip_uuid) = Uuid::parse_str(clip_id) else {
        return false;
    };
    let was_selected = session.selection.effect_ids.iter().any(|id| id == child_id);
    let started_tx = if !session.history.is_in_transaction() {
        session.begin_transaction("delete grouped effect").is_ok()
    } else {
        false
    };
    let mut removed = false;
    let mut adjacent: Option<(String, String)> = None;
    'outer: for track in &mut session.active_sequence_mut().video_tracks {
        for vc in &mut track.clips {
            if vc.id == clip_uuid {
                for fx in &mut vc.layer.effects {
                    let parent_info = Some((fx.id.to_string(), effect_instance_kind_id(&fx.kind)));
                    if let EffectKind::EffectGroup { children, .. } = &mut fx.kind {
                        let (did_remove, adj) = remove_rec(children, child_id, parent_info);
                        if did_remove {
                            removed = true;
                            adjacent = adj;
                            break 'outer;
                        }
                    }
                }
            }
        }
    }
    if removed {
        session.selection.prune_effect(child_id);
        if was_selected
            && session.selection.effect_ids.is_empty()
            && let Some((adj_id, adj_kind)) = adjacent
        {
            let _ = session.selection.select_effect(&adj_id, &adj_kind, false);
        }
    }
    if started_tx {
        let _ = session.commit_transaction(removed);
    }
    removed
}

#[allow(clippy::too_many_arguments)]
pub(super) fn show_nested_effect_group_child_inspector(
    ui: &mut Ui,
    inspector_rect: Rect,
    viewport_rect: Rect,
    clip: &SessionClip,
    child_fx: &EffectInstance,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let child_id = child_fx.id.to_string();
    cm3_column_adjust_splitter(
        ui,
        Id::unique("fx_child_inspector_col_splitter"),
        inspector_rect,
        &mut state.prop_columns,
    );

    ScrollArea::vertical()
        .id_salt(("fx_child_inspector_scroll", &child_id))
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            let (root_open, hdr_act) = property_header_with_actions(
                ui,
                Id::unique(("fx_child_root_hdr", &child_id)),
                &child_fx.name,
                state.prop_columns.label_ratio,
                theme,
            );
            if hdr_act.copy_json {
                let res =
                    session.copy_effects_as_pz_json(&clip.id, std::slice::from_ref(&child_id));
                copy_pz_json_to_clipboard(
                    ui.ctx(),
                    res,
                    "Copied grouped effect JSON to clipboard",
                    actions,
                );
            }
            if hdr_act.paste_json {
                paste_pz_json_from_clipboard(
                    session,
                    state,
                    actions,
                    "effects",
                    Some(clip.id.clone()),
                    None,
                    None,
                    None,
                    "effect JSON or GLSL",
                );
            }
            if hdr_act.reset {
                commit_inspector_tx_if_open(session, state);
                reset_effect_group_child(session, &clip.id, &child_id);
            }
            if hdr_act.duplicate {
                commit_inspector_tx_if_open(session, state);
                duplicate_effect_group_child(session, &clip.id, &child_id);
            }
            if hdr_act.remove {
                commit_inspector_tx_if_open(session, state);
                delete_effect_group_child(session, &clip.id, &child_id);
                return;
            }
            if !root_open {
                return;
            }
            state.inspector_root_indent = true;

            // Enabled toggle
            let mut en_idx = usize::from(effect_track_bool(&child_fx.enabled, true));
            let en_changed = panel_property_row(
                ui,
                "Enabled",
                true,
                state,
                theme,
                |_| {},
                |ui| {
                    cm3_dropdown(
                        ui,
                        Id::unique(("fx_child_enabled", &child_id)),
                        &mut en_idx,
                        &["off", "on"],
                        false,
                        viewport_rect,
                        theme,
                    )
                },
            );
            if en_changed {
                let next_val = en_idx as u32;
                mutate_effect_group_child(
                    session,
                    &clip.id,
                    &child_id,
                    "toggle grouped effect",
                    |eff| {
                        eff.enabled.value = PropertyValue::Option(next_val);
                        if let Some(kf) = eff.enabled.keyframes.first_mut() {
                            kf.value = PropertyValue::Option(next_val);
                        }
                    },
                );
            }

            if let EffectKind::EffectGroup {
                children,
                custom_properties,
            } = &child_fx.kind
            {
                panel_property_row(
                    ui,
                    "Group type",
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        ui.label(
                            egui::RichText::new(format!("Nested Group ({})", children.len()))
                                .font(theme.font(FONT_BODY))
                                .color(TEXT_MUTED),
                        );
                    },
                );
                if !custom_properties.is_empty() {
                    render_effect_group_custom_properties(
                        ui,
                        session,
                        &clip.id,
                        &child_id,
                        custom_properties,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
            }

            let mut param_updates: Vec<(String, PropertyValue)> = Vec::new();
            for (key, track) in &child_fx.params {
                if key == "enabled" {
                    continue;
                }
                let lbl = humanize_prop_label(key);
                match &track.value {
                    PropertyValue::Number(v) => {
                        let mut cur = *v;
                        let ev = panel_property_row(
                            ui,
                            &lbl,
                            true,
                            state,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_scrub_number(
                                    ui,
                                    Id::unique(("fx_child_num", &child_id, key)),
                                    &mut cur,
                                    ScrubConfig::new(-100_000.0, 100_000.0, 0.05, 2),
                                    theme,
                                )
                            },
                        );
                        if ev.changed {
                            param_updates.push((key.clone(), PropertyValue::Number(cur)));
                        }
                    }
                    PropertyValue::Bool(b) => {
                        let mut idx = usize::from(*b);
                        let changed = panel_property_row(
                            ui,
                            &lbl,
                            true,
                            state,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_dropdown(
                                    ui,
                                    Id::unique(("fx_child_bool", &child_id, key)),
                                    &mut idx,
                                    &["off", "on"],
                                    false,
                                    viewport_rect,
                                    theme,
                                )
                            },
                        );
                        if changed {
                            param_updates.push((key.clone(), PropertyValue::Bool(idx != 0)));
                        }
                    }
                    PropertyValue::Option(o) => {
                        let mut cur = f64::from(*o);
                        let ev = panel_property_row(
                            ui,
                            &lbl,
                            true,
                            state,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_scrub_number(
                                    ui,
                                    Id::unique(("fx_child_opt", &child_id, key)),
                                    &mut cur,
                                    ScrubConfig::new(0.0, 64.0, 1.0, 0),
                                    theme,
                                )
                            },
                        );
                        if ev.changed {
                            param_updates
                                .push((key.clone(), PropertyValue::Option(cur.round() as u32)));
                        }
                    }
                    PropertyValue::Vec2(v) => {
                        let mut cur = *v;
                        let ev = panel_property_row(
                            ui,
                            &lbl,
                            true,
                            state,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_vector2(
                                    ui,
                                    Id::unique(("fx_child_v2", &child_id, key)),
                                    &mut cur,
                                    ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 2),
                                    false,
                                    theme,
                                )
                            },
                        );
                        if ev.changed {
                            param_updates.push((key.clone(), PropertyValue::Vec2(cur)));
                        }
                    }
                    PropertyValue::Vec3(v) => {
                        let mut cur = *v;
                        let ev = panel_property_row(
                            ui,
                            &lbl,
                            true,
                            state,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_vector3(
                                    ui,
                                    Id::unique(("fx_child_v3", &child_id, key)),
                                    &mut cur,
                                    ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 2),
                                    false,
                                    theme,
                                )
                            },
                        );
                        if ev.changed {
                            param_updates.push((key.clone(), PropertyValue::Vec3(cur)));
                        }
                    }
                    PropertyValue::Color(c) => {
                        let mut rgba = Rgba::new(c[0] as f32, c[1] as f32, c[2] as f32, 1.0);
                        let cev = panel_property_row(
                            ui,
                            &lbl,
                            true,
                            state,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_hsva_color_picker(
                                    ui,
                                    Id::unique(("fx_child_col", &child_id, key)),
                                    &mut rgba,
                                    false,
                                    false,
                                    viewport_rect,
                                    theme,
                                )
                            },
                        );
                        if cev.changed {
                            param_updates.push((
                                key.clone(),
                                PropertyValue::Color([
                                    f64::from(rgba.0[0]),
                                    f64::from(rgba.0[1]),
                                    f64::from(rgba.0[2]),
                                ]),
                            ));
                        }
                        if cev.eyedropper_clicked {
                            actions.push(PanelAction::EyedropperActivate {
                                target_id: child_id.clone(),
                                property: key.clone(),
                            });
                        }
                    }
                    PropertyValue::ColorRgba(c) => {
                        let mut rgba =
                            Rgba::new(c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32);
                        let cev = panel_property_row(
                            ui,
                            &lbl,
                            true,
                            state,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_hsva_color_picker(
                                    ui,
                                    Id::unique(("fx_child_rgba", &child_id, key)),
                                    &mut rgba,
                                    true,
                                    false,
                                    viewport_rect,
                                    theme,
                                )
                            },
                        );
                        if cev.changed {
                            param_updates.push((
                                key.clone(),
                                PropertyValue::ColorRgba([
                                    f64::from(rgba.0[0]),
                                    f64::from(rgba.0[1]),
                                    f64::from(rgba.0[2]),
                                    f64::from(rgba.0[3]),
                                ]),
                            ));
                        }
                        if cev.eyedropper_clicked {
                            actions.push(PanelAction::EyedropperActivate {
                                target_id: child_id.clone(),
                                property: key.clone(),
                            });
                        }
                    }
                    PropertyValue::Text(s) => {
                        if let Some(summary) = oversized_text_summary(s) {
                            panel_property_row(
                                ui,
                                &lbl,
                                true,
                                state,
                                theme,
                                |_| {},
                                |ui| {
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
                                },
                            );
                        } else {
                            let mut buf = s.clone();
                            let changed = panel_property_row(
                                ui,
                                &lbl,
                                true,
                                state,
                                theme,
                                |_| {},
                                |ui| {
                                    ui.add(
                                        TextEdit::singleline(&mut buf)
                                            .font(theme.font(FONT_SMALL))
                                            .desired_width(ui.available_width()),
                                    )
                                    .changed()
                                },
                            );
                            if changed {
                                param_updates.push((key.clone(), PropertyValue::Text(buf)));
                            }
                        }
                    }
                    _ => {}
                }
            }
            if !param_updates.is_empty() {
                mutate_effect_group_child(
                    session,
                    &clip.id,
                    &child_id,
                    "edit grouped effect param",
                    |eff| {
                        for (k, val) in param_updates {
                            if let Some(tr) = eff.params.get_mut(&k) {
                                tr.value = val.clone();
                                if !tr.animated
                                    && let Some(kf) = tr.keyframes.first_mut()
                                {
                                    kf.value = val;
                                }
                            }
                        }
                    },
                );
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_pack_picker_entries_gate_on_enabled_packs() {
        let mut session = Session::new();

        // Disabled by default: none of the 34 Shader Pack effects appear in the picker.
        let initial_items = build_effect_picker_items(&session);
        assert!(
            !initial_items
                .iter()
                .any(|item| item.category.starts_with("Shader Pack:")),
            "disabled Shader Packs must not appear in the Effects picker"
        );

        // Enable `afterclip`: its Shader Pack effects appear grouped under its pack name.
        session
            .execute(Command::EnablePlugin {
                plugin_id: "afterclip".to_owned(),
            })
            .expect("enable afterclip");
        let afterclip_items = build_effect_picker_items(&session);
        let afterclip_pack_count = afterclip_items
            .iter()
            .filter(|item| item.id.starts_with("zoidcraft:afterclip/"))
            .count();
        assert!(
            afterclip_pack_count > 0,
            "enabled afterclip pack must populate Effects picker"
        );
        assert!(
            afterclip_items
                .iter()
                .filter(|item| item.id.starts_with("zoidcraft:afterclip/"))
                .all(|item| item.category.starts_with("Shader Pack:"))
        );
        assert!(
            !afterclip_items
                .iter()
                .any(|item| item.id.starts_with("zoidcraft:ccfx-shader-pack/"))
        );

        // Enable the other two Shader Packs and verify all 34 declarative effects appear.
        session
            .execute(Command::EnablePlugin {
                plugin_id: "alipfx-shader-pack-4".to_owned(),
            })
            .expect("enable alipfx-shader-pack-4");
        session
            .execute(Command::EnablePlugin {
                plugin_id: "ccfx-shader-pack".to_owned(),
            })
            .expect("enable ccfx-shader-pack");
        let all_items = build_effect_picker_items(&session);
        let total_pack_items = all_items
            .iter()
            .filter(|item| item.category.starts_with("Shader Pack:"))
            .count();
        assert_eq!(
            total_pack_items, 34,
            "all 34 Shader Pack effects across the 3 packs must appear when enabled"
        );

        // Disabling a pack removes only that pack's effects from the picker.
        session
            .execute(Command::DisablePlugin {
                plugin_id: "ccfx-shader-pack".to_owned(),
            })
            .expect("disable ccfx-shader-pack");
        let after_disable = build_effect_picker_items(&session);
        assert!(
            !after_disable
                .iter()
                .any(|item| item.id.starts_with("zoidcraft:ccfx-shader-pack/"))
        );
        assert!(
            after_disable
                .iter()
                .any(|item| item.id.starts_with("zoidcraft:afterclip/"))
        );
    }

    #[test]
    fn effect_group_child_reset_and_duplicate_work_undoably() {
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let clip_uuid = Uuid::parse_str(&clip_id).expect("clip uuid");

        let mut child = create_effect_by_id("gaussianblur").expect("gaussianblur template");
        let child_id = child.id.to_string();
        let default_delta = child
            .params
            .get("delta")
            .and_then(|tr| tr.value.as_f64())
            .expect("default delta");
        if let Some(tr) = child.params.get_mut("delta") {
            tr.value = PropertyValue::Number(default_delta + 42.0);
        }

        let mut group = zc_project::create_effect_group("FX Group");
        let group_id = group.id.to_string();
        if let EffectKind::EffectGroup {
            ref mut children, ..
        } = group.kind
        {
            children.push(child);
        }
        for track in &mut session.active_sequence_mut().video_tracks {
            for vc in &mut track.clips {
                if vc.id == clip_uuid {
                    vc.layer.effects.push(group.clone());
                }
            }
        }

        // Reset restores the child's parameters back to template defaults.
        assert!(reset_effect_group_child(&mut session, &clip_id, &child_id));
        let reset_child =
            find_effect_group_child(&session, &clip_id, &child_id).expect("reset child");
        assert_eq!(
            reset_child
                .params
                .get("delta")
                .and_then(|tr| tr.value.as_f64()),
            Some(default_delta)
        );

        // Duplicate clones the child directly inside the same EffectGroup with a fresh UUID.
        let dup_id =
            duplicate_effect_group_child(&mut session, &clip_id, &child_id).expect("dup child");
        assert_ne!(dup_id, child_id);
        let rows = collect_effect_group_child_rows(&session, &clip_id, &group_id, &BTreeSet::new());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].id, child_id);
        assert_eq!(rows[1].id, dup_id);
        assert_eq!(session.selection.effect_ids, vec![dup_id]);
    }
}
