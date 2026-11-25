use super::media_tab::layout_single_line_ellipsized;
use super::property_rows::{
    apply_color_picker_to_session, apply_scrub_to_session, eval_prop_f64, eval_prop_str,
    eval_prop_u32, eval_prop_vec2, eval_prop_vec3, find_clip_cached, normalized_kf_frame,
    show_session_color_prop, show_session_number_prop, show_session_text_prop_ex,
    show_session_vec2_prop,
};
use super::shared_widgets::{empty_panel_message, find_project_video_clip, show_font_selector_row};
use super::state::{
    InlineRenameState, WidgetTxTracker, apply_widget_tx_command, copy_pz_json_to_clipboard,
    paste_pz_json_from_clipboard, touch_inspector_tx_owner,
};
use super::*;

// ---------------------------------------------------------------------------
// Tab 3: Edit (edit2) — Selected Clip Inspector
// ---------------------------------------------------------------------------

pub(super) fn show_edit_tab(
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

    let Some(clip_snapshot) = find_clip_cached(session, state, &clip_id) else {
        empty_panel_message(ui, "select a clip", theme);
        return;
    };

    cm3_column_adjust_splitter(
        ui,
        Id::unique("edit_col_splitter"),
        panel_rect,
        &mut state.prop_columns,
    );

    let local_frame = session.playback.current_frame - clip_snapshot.start;
    let clip_start = clip_snapshot.start;
    let layer_title: &str = match clip_snapshot.kind.as_str() {
        "scene" => "Scene",
        "text" | "text_2d" => "Text",
        "shape" | "preset_shape" | "zoidium:shape-plus/shape-layer" => "Shape",
        "audio" => "Audio",
        "adjustment" => "Adjustment",
        "composite" => "Composite",
        "video" => "Video",
        "image" => "Image",
        _ => clip_snapshot.name.as_str(),
    };

    ScrollArea::vertical()
        .id_salt(("edit_tab_scroll", &clip_id))
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;

            let selected_clip_count = session.selection.clip_ids.len();
            if selected_clip_count > 1 {
                let banner_resp = Frame::NONE
                    .fill(BG_TITLE)
                    .inner_margin(egui::Margin {
                        left: 10,
                        right: 8,
                        top: 5,
                        bottom: 5,
                    })
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.label(
                            egui::RichText::new(format!("{selected_clip_count} clips selected"))
                                .font(theme.font(FONT_SMALL))
                                .color(theme.value()),
                        );
                    });
                let br = banner_resp.response.rect;
                ui.painter().rect_filled(
                    Rect::from_min_size(br.left_top(), Vec2::new(3.0, br.height())),
                    CornerRadius::ZERO,
                    theme.accent(),
                );
                ui.painter().rect_filled(
                    Rect::from_min_max(
                        Pos2::new(br.left(), (br.bottom() - 1.0).round()),
                        Pos2::new(br.right(), br.bottom().round()),
                    ),
                    CornerRadius::ZERO,
                    BORDER_ROW_SOFT,
                );
            }

            // Root collapsible #222222 bar titled with just the clip name (NO "Clip: " prefix)
            cm3_collapsible_section_with_action(
                ui,
                Id::unique(("edit_clip_root", &clip_id)),
                &clip_snapshot.name,
                true,
                false,
                state.prop_columns.label_ratio,
                None,
                theme,
                |ui| {
                    // Nested #2a2a2a "Properties" section (Media + Time)
                    let (reset_props, _) = cm3_collapsible_section_with_action(
                        ui,
                        Id::unique(("edit_clip_props", &clip_id)),
                        "Properties",
                        true,
                        true,
                        state.prop_columns.label_ratio,
                        Some((Cm3Icon::Reset, "Reset properties")),
                        theme,
                        |ui| {
                            // Media (read-only filename or "(none)" in #999999)
                            let media_label = eval_prop_str(
                                clip_snapshot.properties.get("assetId"),
                                local_frame,
                                "",
                            );
                            let resolved_media = if media_label.is_empty() {
                                "(none)".to_owned()
                            } else {
                                session
                                    .project
                                    .assets
                                    .get(&media_label)
                                    .map(|a| a.filename.clone())
                                    .unwrap_or(media_label)
                            };
                            cm3_property_row_aligned(
                                ui,
                                "Media",
                                true,
                                state.prop_columns,
                                theme,
                                |_| {},
                                |ui| {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(&resolved_media)
                                                .font(theme.font(FONT_BODY))
                                                .color(TEXT_DISABLED),
                                        )
                                        .truncate(),
                                    );
                                },
                            );

                            // Time (0.000, with stopwatch)
                            show_session_number_prop(
                                ui,
                                session,
                                &clip_id,
                                "time",
                                "Time",
                                (local_frame / session.document().fps.max(1.0)).max(0.0),
                                ScrubConfig::new(0.0, 100_000.0, 0.05, 3),
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );
                        },
                    );
                    if reset_props {
                        let time_animated = clip_snapshot
                            .properties
                            .get("time")
                            .is_some_and(|p| p.animated);
                        let started_tx = session.begin_transaction("reset properties").is_ok();
                        if time_animated {
                            let _ = session.execute(Command::TogglePropertyAnimated {
                                target_id: clip_id.clone(),
                                property: "time".to_owned(),
                                frame: normalized_kf_frame(session, local_frame),
                            });
                        }
                        let _ = session.execute(Command::SetPropertyValue {
                            target_id: clip_id.clone(),
                            property: "time".to_owned(),
                            value: json!(0.0),
                        });
                        if started_tx && session.history.is_in_transaction() {
                            let _ = session.commit_transaction(true);
                        }
                    }

                    // Nested #2a2a2a layer section ("Scene" / "Text" / "Shape" / "Audio" etc.)
                    let (reset_layer, _) = cm3_collapsible_section_with_action(
                        ui,
                        Id::unique(("edit_layer_root", &clip_id)),
                        layer_title,
                        true,
                        true,
                        state.prop_columns.label_ratio,
                        Some((Cm3Icon::Reset, "Reset layer")),
                        theme,
                        |ui| {
                            if clip_snapshot.kind == "audio" {
                                show_session_number_prop(
                                    ui,
                                    session,
                                    &clip_id,
                                    "volume",
                                    "Volume",
                                    1.0,
                                    ScrubConfig::new(0.0, 4.0, 0.05, 2),
                                    local_frame,
                                    clip_start,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                                show_session_number_prop(
                                    ui,
                                    session,
                                    &clip_id,
                                    "pan",
                                    "Pan",
                                    0.0,
                                    ScrubConfig::new(-1.0, 1.0, 0.05, 2),
                                    local_frame,
                                    clip_start,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                                return;
                            }

                            // Suppress the bottom divider drawn by cm3_collapsible_section_with_action
                            // directly beneath the nested layer ("Scene") header before Masks.
                            let hdr_div_y = (ui.cursor().top() - 1.0).round();
                            ui.painter().rect_filled(
                                Rect::from_min_max(
                                    Pos2::new(panel_rect.left() + 1.0, hdr_div_y),
                                    Pos2::new(panel_rect.right() - 1.0, hdr_div_y + 1.0),
                                ),
                                CornerRadius::ZERO,
                                BG_PANEL,
                            );

                            // Shape+ Layer Masks (`zoidiumMasks`) full interactive section
                            show_edit_layer_masks_section(
                                ui,
                                session,
                                &clip_id,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );

                            // Precomp+ Target Composition selector (when editing a Precomp clip)
                            show_edit_precomp_target_section(
                                ui,
                                session,
                                &clip_id,
                                viewport_rect,
                                state,
                                theme,
                            );

                            // Scene3D / visual layer backdrop & resolution controls (no stopwatch)
                            if clip_snapshot.kind == "scene" {
                                // Scene Background dropdown ("Transparent" / "Color")
                                let mut bg_mode_idx = eval_prop_u32(
                                    clip_snapshot.properties.get("sceneBackground"),
                                    local_frame,
                                    0,
                                ) as usize;
                                let bg_changed = cm3_property_row_aligned(
                                    ui,
                                    "Scene Background",
                                    true,
                                    state.prop_columns,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        cm3_dropdown(
                                            ui,
                                            Id::unique(("clip_scene_bg_mode", &clip_id)),
                                            &mut bg_mode_idx,
                                            &["Transparent", "Color"],
                                            false,
                                            viewport_rect,
                                            theme,
                                        )
                                    },
                                );
                                if bg_changed {
                                    let _ = session.execute(Command::SetPropertyValue {
                                        target_id: clip_id.clone(),
                                        property: "sceneBackground".to_owned(),
                                        value: json!(bg_mode_idx),
                                    });
                                }

                                // Backdrop Color swatch (no stopwatch, exact 28px pitch)
                                let bg_rgb = eval_prop_vec3(
                                    clip_snapshot.properties.get("backgroundColor"),
                                    local_frame,
                                    [0.0, 0.0, 0.0],
                                );
                                let mut bg_rgba = Rgba::new(
                                    bg_rgb[0] as f32,
                                    bg_rgb[1] as f32,
                                    bg_rgb[2] as f32,
                                    1.0,
                                );
                                let backdrop_before_y = ui.cursor().top();
                                let cev = cm3_property_row_aligned(
                                    ui,
                                    "Backdrop",
                                    true,
                                    state.prop_columns,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        cm3_hsva_color_picker(
                                            ui,
                                            Id::unique(("clip_backdrop_color", &clip_id)),
                                            &mut bg_rgba,
                                            false,
                                            false,
                                            viewport_rect,
                                            theme,
                                        )
                                    },
                                );
                                let backdrop_after_y = ui.cursor().top();
                                if backdrop_after_y - backdrop_before_y < 27.5 {
                                    let old_div_y = (backdrop_after_y - 1.0).round();
                                    let row_w = ui.available_width();
                                    let row_left = ui.cursor().left().round();
                                    ui.painter().rect_filled(
                                        Rect::from_min_max(
                                            Pos2::new(row_left, old_div_y),
                                            Pos2::new(row_left + row_w, old_div_y + 1.0),
                                        ),
                                        CornerRadius::ZERO,
                                        BG_PANEL,
                                    );
                                    let (new_div, _) = ui
                                        .allocate_exact_size(Vec2::new(row_w, 1.0), Sense::hover());
                                    ui.painter().rect_filled(
                                        new_div,
                                        CornerRadius::ZERO,
                                        BORDER_ROW_SOFT,
                                    );
                                }
                                touch_inspector_tx_owner(state, &clip_id, "backgroundColor");
                                apply_color_picker_to_session(
                                    session,
                                    &clip_id,
                                    "backgroundColor",
                                    false,
                                    local_frame,
                                    json!([
                                        f64::from(bg_rgba.0[0]),
                                        f64::from(bg_rgba.0[1]),
                                        f64::from(bg_rgba.0[2])
                                    ]),
                                    cev,
                                    state,
                                );
                                if cev.eyedropper_clicked {
                                    actions.push(PanelAction::EyedropperActivate {
                                        target_id: clip_id.clone(),
                                        property: "backgroundColor".to_owned(),
                                    });
                                }

                                // Backdrop Opacity (1.00, no stopwatch)
                                touch_inspector_tx_owner(state, &clip_id, "backdropOpacity");
                                let mut bg_op = eval_prop_f64(
                                    clip_snapshot.properties.get("backdropOpacity"),
                                    local_frame,
                                    1.0,
                                );
                                let op_ev = cm3_property_row_aligned(
                                    ui,
                                    "Bg alpha",
                                    true,
                                    state.prop_columns,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        cm3_scrub_number(
                                            ui,
                                            Id::unique(("clip_backdrop_opacity", &clip_id)),
                                            &mut bg_op,
                                            ScrubConfig::new(0.0, 1.0, 0.05, 2),
                                            theme,
                                        )
                                    },
                                );
                                apply_scrub_to_session(
                                    session,
                                    &clip_id,
                                    "backdropOpacity",
                                    false,
                                    local_frame,
                                    json!(bg_op),
                                    op_ev,
                                    state,
                                );

                                // Resolution (1920 x 1080 with "x" separator, no stopwatch)
                                touch_inspector_tx_owner(state, &clip_id, "resolution");
                                let [seq_w, seq_h] = session.document().resolution;
                                let mut layer_res = eval_prop_vec2(
                                    clip_snapshot.properties.get("resolution"),
                                    local_frame,
                                    [f64::from(seq_w), f64::from(seq_h)],
                                );
                                let res_ev = cm3_property_row_aligned(
                                    ui,
                                    "Resolution",
                                    true,
                                    state.prop_columns,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        cm3_vector2_with_sep(
                                            ui,
                                            Id::unique(("clip_resolution", &clip_id)),
                                            &mut layer_res,
                                            &ScrubConfig::new(16.0, 7680.0, 10.0, 0),
                                            false,
                                            "x",
                                            theme,
                                        )
                                    },
                                );
                                apply_scrub_to_session(
                                    session,
                                    &clip_id,
                                    "resolution",
                                    false,
                                    local_frame,
                                    json!(layer_res),
                                    res_ev,
                                    state,
                                );
                            }

                            // Kind-specific 2D text / shape properties
                            match clip_snapshot.kind.as_str() {
                                "text" | "text_2d" => {
                                    show_session_text_prop_ex(
                                        ui,
                                        session,
                                        &clip_id,
                                        "text",
                                        "Text",
                                        "text",
                                        true,
                                        local_frame,
                                        state,
                                        theme,
                                    );
                                    show_font_selector_row(
                                        ui,
                                        session,
                                        &clip_id,
                                        "text2d_edit_font",
                                        local_frame,
                                        viewport_rect,
                                        state,
                                        theme,
                                        actions,
                                    );
                                    show_session_number_prop(
                                        ui,
                                        session,
                                        &clip_id,
                                        "fontSize",
                                        "Font size",
                                        72.0,
                                        ScrubConfig::new(4.0, 1000.0, 1.0, 0),
                                        local_frame,
                                        clip_start,
                                        viewport_rect,
                                        state,
                                        theme,
                                        actions,
                                    );
                                    show_session_color_prop(
                                        ui,
                                        session,
                                        &clip_id,
                                        "color",
                                        "Color",
                                        [1.0, 1.0, 1.0],
                                        local_frame,
                                        clip_start,
                                        viewport_rect,
                                        state,
                                        theme,
                                        actions,
                                    );
                                }
                                "shape" | "preset_shape" => {
                                    if clip_snapshot.kind == "preset_shape" {
                                        show_preset_shape_kind_row(
                                            ui,
                                            session,
                                            &clip_id,
                                            viewport_rect,
                                            state,
                                            theme,
                                        );
                                    }
                                    show_session_color_prop(
                                        ui,
                                        session,
                                        &clip_id,
                                        "fillColor",
                                        "Fill color",
                                        [1.0, 1.0, 1.0],
                                        local_frame,
                                        clip_start,
                                        viewport_rect,
                                        state,
                                        theme,
                                        actions,
                                    );
                                    show_session_number_prop(
                                        ui,
                                        session,
                                        &clip_id,
                                        "strokeWidth",
                                        "Stroke width",
                                        0.0,
                                        ScrubConfig::new(0.0, 200.0, 1.0, 1),
                                        local_frame,
                                        clip_start,
                                        viewport_rect,
                                        state,
                                        theme,
                                        actions,
                                    );
                                    show_session_color_prop(
                                        ui,
                                        session,
                                        &clip_id,
                                        "strokeColor",
                                        "Stroke color",
                                        [1.0, 1.0, 1.0],
                                        local_frame,
                                        clip_start,
                                        viewport_rect,
                                        state,
                                        theme,
                                        actions,
                                    );
                                }
                                _ => {}
                            }

                            // Transform & Compositing rows (Position, Anchor, Scale, Rotation, Opacity, Blending mode)
                            show_session_vec2_prop(
                                ui,
                                session,
                                &clip_id,
                                "position",
                                "Position",
                                [0.0, 0.0],
                                ScrubConfig::new(-10_000.0, 10_000.0, 1.0, 2),
                                false,
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );
                            show_session_vec2_prop(
                                ui,
                                session,
                                &clip_id,
                                "anchor",
                                "Anchor",
                                [0.0, 0.0],
                                ScrubConfig::new(-10_000.0, 10_000.0, 1.0, 2),
                                false,
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );
                            show_session_vec2_prop(
                                ui,
                                session,
                                &clip_id,
                                "scale",
                                "Scale",
                                [1.0, 1.0],
                                ScrubConfig::new(-100.0, 100.0, 0.05, 2),
                                true,
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );
                            show_session_number_prop(
                                ui,
                                session,
                                &clip_id,
                                "rotation",
                                "Rotation",
                                0.0,
                                ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 1)
                                    .with_scale_factor(std::f64::consts::PI / 180.0),
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );
                            show_session_number_prop(
                                ui,
                                session,
                                &clip_id,
                                "opacity",
                                "Opacity",
                                1.0,
                                ScrubConfig::new(0.0, 1.0, 0.05, 2),
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );

                            // Blending mode dropdown (no stopwatch)
                            let cur_blend_token = eval_prop_str(
                                clip_snapshot.properties.get("blending"),
                                local_frame,
                                "BLEND_SRC_OVER",
                            );
                            let mut blend_idx = BlendMode::ALL
                                .iter()
                                .position(|b| b.token() == cur_blend_token)
                                .unwrap_or(3);
                            let blend_labels: Vec<&str> =
                                BlendMode::ALL.iter().map(|b| b.display_name()).collect();
                            let changed = cm3_property_row_aligned(
                                ui,
                                "Blending mode",
                                true,
                                state.prop_columns,
                                theme,
                                |_| {},
                                |ui| {
                                    cm3_dropdown(
                                        ui,
                                        Id::unique(("clip_blend", &clip_id)),
                                        &mut blend_idx,
                                        &blend_labels,
                                        false,
                                        viewport_rect,
                                        theme,
                                    )
                                },
                            );
                            if changed {
                                let token = BlendMode::ALL[blend_idx].token();
                                let _ = session.execute(Command::SetPropertyValue {
                                    target_id: clip_id.clone(),
                                    property: "blending".to_owned(),
                                    value: json!(token),
                                });
                            }
                        },
                    );
                    if reset_layer {
                        let _ = session.execute(Command::ResetPropertyList {
                            target_id: clip_id.clone(),
                        });
                    }
                },
            );
        });
}

// ---------------------------------------------------------------------------
// 3. Shape+ Layer Masks (`zoidiumMasks`) & Precomp+ Target Picker (Edit Tab)
// ---------------------------------------------------------------------------

/// Creates a centered 50%×50% default rectangular mask in sequence pixel coordinates
/// `(0..seq_w, 0..seq_h)` so all 4 vertices are clearly visible inside the viewport.
pub(super) fn default_layer_mask(index: usize, seq_w: u32, seq_h: u32) -> LayerMask {
    let w = f64::from(seq_w.max(64));
    let h = f64::from(seq_h.max(64));
    let x0 = (w * 0.25).round();
    let x1 = (w * 0.75).round();
    let y0 = (h * 0.25).round();
    let y1 = (h * 0.75).round();
    let default_rect_path = json!([
        x0, y0, 0.0, 0.0, 0.0, 0.0, x1, y0, 0.0, 0.0, 0.0, 0.0, x1, y1, 0.0, 0.0, 0.0, 0.0, x0, y1,
        0.0, 0.0, 0.0, 0.0
    ]);
    let palette = ["#58a6ff", "#f78166", "#56d364", "#d2a8ff", "#e3b341"];
    LayerMask {
        id: format!("mask_{index}"),
        name: format!("Mask {index}"),
        enabled: true,
        mode: MaskMode::Add,
        invert: false,
        feather: 0.0,
        expansion: 0.0,
        opacity: 1.0,
        closed: true,
        locked: false,
        outline_color: palette[(index.saturating_sub(1)) % palette.len()].to_owned(),
        vertex_ids: vec![
            "v1".to_owned(),
            "v2".to_owned(),
            "v3".to_owned(),
            "v4".to_owned(),
        ],
        vertex_modes: BTreeMap::new(),
        path: PropertyTrack::constant(PropertyValue::Json(default_rect_path)),
        preserved: BTreeMap::new(),
    }
}

fn show_edit_layer_masks_section(
    ui: &mut Ui,
    session: &mut Session,
    clip_id: &str,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let clip_uuid = Uuid::parse_str(clip_id).ok();
    let layer_id = find_project_video_clip(session, clip_id).map(|vc| vc.layer.id);
    let masks: Vec<LayerMask> = layer_id
        .and_then(|lid| session.active_sequence().layer_masks.get(&lid).cloned())
        .unwrap_or_default();

    let mask_w = ui.available_width();
    let (masks_hdr, _) = ui.allocate_exact_size(Vec2::new(mask_w, 24.0), Sense::hover());
    ui.painter().text(
        Pos2::new(masks_hdr.left() + 15.0, masks_hdr.top() + 10.5),
        Align2::LEFT_CENTER,
        format!("Masks ({})", masks.len()),
        theme.font(FONT_BODY),
        TEXT_PRIMARY,
    );

    let (add_clicked, copy_clicked, paste_clicked) = Frame::NONE
        .inner_margin(egui::Margin {
            left: 14,
            right: 10,
            top: 2,
            bottom: 6,
        })
        .show(ui, |ui| {
            let mut add = false;
            let mut copy = false;
            let mut paste = false;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                add = cm3_prop_button(ui, "+ Add mask", ButtonVariant::Normal, false, theme)
                    .clicked();
                copy = cm3_prop_button(ui, "Copy", ButtonVariant::Normal, false, theme)
                    .on_hover_text("Copy layer masks JSON to clipboard (PZ compatible)")
                    .clicked();
                paste = cm3_prop_button(ui, "Paste", ButtonVariant::Normal, false, theme)
                    .on_hover_text("Paste layer masks JSON from clipboard (PZ compatible)")
                    .clicked();
            });
            (add, copy, paste)
        })
        .inner;
    if add_clicked && let Some(cid) = clip_uuid {
        let [seq_w, seq_h] = session.document().resolution;
        let new_mask = default_layer_mask(masks.len() + 1, seq_w, seq_h);
        let new_id = new_mask.id.clone();
        if session
            .execute(Command::AddLayerMask {
                clip_id: cid,
                mask: new_mask,
            })
            .is_ok()
        {
            state.expanded_masks.insert(new_id);
        }
    }
    if copy_clicked {
        let res = session.copy_masks_as_pz_json(clip_id);
        copy_pz_json_to_clipboard(
            ui.ctx(),
            res,
            "Copied layer masks JSON to clipboard",
            actions,
        );
    }
    if paste_clicked {
        paste_pz_json_from_clipboard(
            session,
            state,
            actions,
            "masks",
            Some(clip_id.to_owned()),
            None,
            None,
            None,
            "layer masks JSON",
        );
    }

    // Expandable mask cards
    let total = masks.len();
    for (idx, mut mask) in masks.into_iter().enumerate() {
        let Some(cid) = clip_uuid else {
            break;
        };
        let expanded = state.expanded_masks.contains(&mask.id);
        let mut changed = false;
        let mut tx_tracker = WidgetTxTracker::default();

        Frame::NONE
            .fill(BG_SUNKEN)
            .stroke(Stroke::new(1.0, BORDER_ROW_SOFT))
            .inner_margin(egui::Margin::symmetric(8, 5))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 5.0;
                    if cm3_icon_button(
                        ui,
                        if expanded {
                            Cm3Icon::ChevronDown
                        } else {
                            Cm3Icon::ChevronRight
                        },
                        14.0,
                        TEXT_MUTED,
                        if expanded {
                            "Collapse mask"
                        } else {
                            "Expand mask"
                        },
                        theme,
                    ) {
                        if expanded {
                            state.expanded_masks.remove(&mask.id);
                        } else {
                            state.expanded_masks.insert(mask.id.clone());
                        }
                    }
                    if ui
                        .checkbox(&mut mask.enabled, "")
                        .on_hover_text("Toggle mask visibility")
                        .changed()
                    {
                        changed = true;
                    }
                    let btn_count = 2 + usize::from(idx + 1 < total) + usize::from(idx > 0);
                    let buttons_w = btn_count as f32 * 24.0 + 4.0;
                    let name_w = (ui.available_width() - buttons_w).max(32.0);
                    let mask_rename_id = format!("mask_rename:{}", mask.id);
                    let is_renaming = state
                        .rename_state
                        .as_ref()
                        .is_some_and(|r| r.target_id == mask_rename_id);
                    if is_renaming {
                        if let Some(rename) = state.rename_state.as_mut() {
                            let mut commit = false;
                            let mut cancel = false;
                            ui.input(|i| {
                                if i.key_pressed(Key::Enter) {
                                    commit = true;
                                } else if i.key_pressed(Key::Escape) {
                                    cancel = true;
                                }
                            });
                            let resp = ui.add(
                                TextEdit::singleline(&mut rename.buffer)
                                    .font(theme.font(FONT_SMALL))
                                    .desired_width(name_w),
                            );
                            if rename.just_started {
                                resp.request_focus();
                                rename.just_started = false;
                            }
                            if cancel {
                                mark_floating_editor_escape_consumed(ui.ctx());
                                state.rename_state = None;
                            } else if commit || resp.lost_focus() {
                                let trimmed = rename.buffer.trim();
                                if !trimmed.is_empty() && trimmed != mask.name {
                                    mask.name = trimmed.to_owned();
                                    changed = true;
                                }
                                state.rename_state = None;
                            }
                        }
                    } else {
                        let (name_rect, name_resp) =
                            ui.allocate_exact_size(Vec2::new(name_w, 20.0), Sense::click());
                        let name_resp = name_resp
                            .on_hover_cursor(CursorIcon::PointingHand)
                            .on_hover_text(format!("{} (double-click to rename)", mask.name));
                        let name_galley = layout_single_line_ellipsized(
                            ui.painter(),
                            &mask.name,
                            theme.font(FONT_SMALL),
                            TEXT_PRIMARY,
                            (name_w - 4.0).max(16.0),
                        );
                        ui.painter().with_clip_rect(name_rect).galley(
                            Pos2::new(
                                name_rect.left() + 2.0,
                                name_rect.center().y - name_galley.size().y * 0.5,
                            ),
                            name_galley,
                            TEXT_PRIMARY,
                        );
                        if name_resp.double_clicked() {
                            state.rename_state = Some(InlineRenameState {
                                target_id: mask_rename_id,
                                buffer: mask.name.clone(),
                                just_started: true,
                            });
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("×").on_hover_text("Delete mask").clicked() {
                            let _ = session.execute(Command::RemoveLayerMask {
                                clip_id: cid,
                                mask_id: mask.id.clone(),
                            });
                        }
                        if ui
                            .small_button("D")
                            .on_hover_text("Duplicate mask")
                            .clicked()
                        {
                            let _ = session.execute(Command::DuplicateLayerMask {
                                clip_id: cid,
                                mask_id: mask.id.clone(),
                            });
                        }
                        if idx + 1 < total
                            && ui.small_button("↓").on_hover_text("Move down").clicked()
                        {
                            let _ = session.execute(Command::ReorderLayerMask {
                                clip_id: cid,
                                mask_id: mask.id.clone(),
                                to_index: idx + 1,
                            });
                        }
                        if idx > 0 && ui.small_button("↑").on_hover_text("Move up").clicked() {
                            let _ = session.execute(Command::ReorderLayerMask {
                                clip_id: cid,
                                mask_id: mask.id.clone(),
                                to_index: idx - 1,
                            });
                        }
                    });
                });

                if expanded {
                    ui.add_space(4.0);
                    let mut mode_idx = match mask.mode {
                        MaskMode::Add => 0,
                        MaskMode::Subtract => 1,
                        MaskMode::Intersect => 2,
                    };
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 8.0;
                        ui.label(
                            egui::RichText::new("Mode")
                                .font(theme.font(FONT_COMPACT))
                                .color(TEXT_MUTED),
                        );
                        if cm3_dropdown(
                            ui,
                            Id::unique(("mask_mode", &mask.id)),
                            &mut mode_idx,
                            &["Add", "Subtract", "Intersect"],
                            false,
                            viewport_rect,
                            theme,
                        ) {
                            mask.mode = match mode_idx {
                                1 => MaskMode::Subtract,
                                2 => MaskMode::Intersect,
                                _ => MaskMode::Add,
                            };
                            changed = true;
                        }
                        ui.add_space(4.0);
                        if ui.checkbox(&mut mask.invert, "Invert").changed() {
                            changed = true;
                        }
                        ui.add_space(4.0);
                        if ui.checkbox(&mut mask.locked, "Lock").changed() {
                            changed = true;
                        }
                    });
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Feather")
                                .font(theme.font(FONT_COMPACT))
                                .color(TEXT_MUTED),
                        );
                        let feather_resp = ui.add(
                            egui::DragValue::new(&mut mask.feather)
                                .range(0.0..=500.0)
                                .speed(0.5),
                        );
                        if tx_tracker.observe(&feather_resp) {
                            changed = true;
                        }
                        ui.label(
                            egui::RichText::new("Expand")
                                .font(theme.font(FONT_COMPACT))
                                .color(TEXT_MUTED),
                        );
                        let expand_resp = ui.add(
                            egui::DragValue::new(&mut mask.expansion)
                                .range(-500.0..=500.0)
                                .speed(0.5),
                        );
                        if tx_tracker.observe(&expand_resp) {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Opacity")
                                .font(theme.font(FONT_COMPACT))
                                .color(TEXT_MUTED),
                        );
                        let opacity_resp =
                            ui.add(egui::Slider::new(&mut mask.opacity, 0.0..=1.0).max_decimals(2));
                        if tx_tracker.observe(&opacity_resp) {
                            changed = true;
                        }
                        let mut hex = parse_hex_rgb_f32(&mask.outline_color);
                        let col_resp = ui.color_edit_button_rgb(&mut hex);
                        if tx_tracker.observe(&col_resp) {
                            mask.outline_color = format!(
                                "#{:02x}{:02x}{:02x}",
                                (hex[0] * 255.0).round() as u8,
                                (hex[1] * 255.0).round() as u8,
                                (hex[2] * 255.0).round() as u8,
                            );
                            changed = true;
                        }
                    });
                }
            });

        if changed {
            tx_tracker.changed = true;
        }
        let mask_owner_key = format!("mask_{}", mask.id);
        let mask_id = mask.id.clone();
        apply_widget_tx_command(
            ui,
            session,
            state,
            clip_id,
            &mask_owner_key,
            "edit layer mask",
            tx_tracker,
            |session| {
                let _ = session.execute(Command::UpdateLayerMask {
                    clip_id: cid,
                    mask_id,
                    mask,
                });
            },
        );
    }

    let (mask_div, _) = ui.allocate_exact_size(Vec2::new(mask_w, 1.0), Sense::hover());
    ui.painter()
        .rect_filled(mask_div, CornerRadius::ZERO, BORDER_ROW_SOFT);
}

fn parse_hex_rgb_f32(hex: &str) -> [f32; 3] {
    let s = hex.trim().trim_start_matches('#');
    if s.len() == 6
        && let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&s[0..2], 16),
            u8::from_str_radix(&s[2..4], 16),
            u8::from_str_radix(&s[4..6], 16),
        )
    {
        return [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
    }
    [0.34, 0.65, 1.0]
}

fn show_edit_precomp_target_section(
    ui: &mut Ui,
    session: &mut Session,
    clip_id: &str,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    let Some(vc) = find_project_video_clip(session, clip_id) else {
        return;
    };
    let LayerKind::PrecompRef { composition_id } = vc.layer.kind else {
        return;
    };
    let Some(cid) = Uuid::parse_str(clip_id).ok() else {
        return;
    };
    let mut comps: Vec<(Uuid, String)> = session
        .project
        .compositions
        .iter()
        .map(|(id, s)| (*id, s.name.clone()))
        .collect();
    comps.sort_by(|a, b| a.1.cmp(&b.1));
    if comps.is_empty() {
        return;
    }
    let labels: Vec<&str> = comps.iter().map(|(_, n)| n.as_str()).collect();
    let mut sel_idx = comps
        .iter()
        .position(|(id, _)| *id == composition_id)
        .unwrap_or(0);
    let changed = cm3_property_row_aligned(
        ui,
        "Target comp",
        true,
        state.prop_columns,
        theme,
        |_| {},
        |ui| {
            ui.horizontal(|ui| {
                let c = cm3_dropdown(
                    ui,
                    Id::unique(("precomp_target_dd", clip_id)),
                    &mut sel_idx,
                    &labels,
                    false,
                    viewport_rect,
                    theme,
                );
                if ui.small_button("Open").clicked() {
                    let _ = session.execute(Command::SetActiveComposition {
                        comp_id: Some(composition_id),
                    });
                }
                c
            })
            .inner
        },
    );
    if changed {
        let _ = session.execute(Command::SetPrecompClipTarget {
            clip_id: cid,
            composition_id: comps[sel_idx].0,
        });
    }
}

pub(super) fn show_preset_shape_kind_row(
    ui: &mut Ui,
    session: &mut Session,
    clip_id: &str,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    let Some(cid) = Uuid::parse_str(clip_id).ok() else {
        return;
    };
    let cur = find_project_video_clip(session, clip_id)
        .and_then(|vc| match vc.layer.kind {
            LayerKind::PresetShape2D { preset, .. } => Some(preset),
            _ => None,
        })
        .unwrap_or(PresetShapeKind::Ellipse);
    let mut idx = match cur {
        PresetShapeKind::Ellipse => 0,
        PresetShapeKind::Rectangle => 1,
    };
    let changed = cm3_property_row_aligned(
        ui,
        "Shape kind",
        true,
        state.prop_columns,
        theme,
        |_| {},
        |ui| {
            cm3_dropdown(
                ui,
                Id::unique(("preset_shape_kind", clip_id)),
                &mut idx,
                &["Ellipse", "Rectangle"],
                false,
                viewport_rect,
                theme,
            )
        },
    );
    if changed {
        let shape_kind = if idx == 1 {
            PresetShapeKind::Rectangle
        } else {
            PresetShapeKind::Ellipse
        };
        let _ = session.execute(Command::SetPresetShapeKind {
            clip_id: cid,
            shape_kind,
        });
    }
}
