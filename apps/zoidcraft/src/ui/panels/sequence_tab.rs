use super::media_tab::layout_single_line_ellipsized;
use super::objects_2d::{read_clip_track_matte_mode, show_clip_track_matte_rows};
use super::property_rows::{
    eval_prop_u32, find_clip_cached, lookup_session_prop_ref, normalized_kf_frame,
    open_expression_window, show_session_number_prop,
};
use super::shared_widgets::humanize_prop_label;
use super::state::{
    WidgetTxTracker, apply_widget_tx_command, begin_inspector_tx, clear_inspector_tx_state,
    close_existing_tx_before_start, owns_inspector_tx, run_atomic_commit_outside_foreign_tx,
    touch_inspector_tx_owner,
};
use super::*;

// ---------------------------------------------------------------------------
// Tab 2: Sequence (sequence)
// ---------------------------------------------------------------------------

/// Applies an undoable update to the active sequence's RGBA `background_color` and bumps the
/// playback/render cache epoch so the viewport (and checkerboard overlay) redraw immediately.
pub(super) fn set_sequence_background_color_undoable(
    session: &mut Session,
    bg_color: [f32; 4],
    label: &str,
) {
    let started_tx = if !session.history.is_in_transaction() {
        session.begin_transaction(label).is_ok()
    } else {
        false
    };
    session.active_sequence_mut().background_color = bg_color;
    session.playback.bump_epoch();
    if started_tx && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    }
}

/// Returns responsive motion-blur labels (`"Samples"`/`"Shutter"` normally, or distinct
/// compact `"Samp."`/`"Shutt."` on narrow `800×540` panels) so they never collapse to `"S..."`.
pub(super) fn sequence_motion_blur_labels(panel_width: f32) -> (&'static str, &'static str) {
    if panel_width < 245.0 {
        ("Samp.", "Shutt.")
    } else {
        ("Samples", "Shutter")
    }
}

/// Computes the Sequence header `label_ratio` so `"Sequence"` never clips against `↻`.
pub(super) fn sequence_header_label_ratio(avail_w: f32, base_ratio: f32) -> f32 {
    let w = avail_w.max(120.0);
    let min_ratio = (112.0 / w).min(0.86);
    base_ratio.max(min_ratio)
}

pub(super) fn show_sequence_tab(
    ui: &mut Ui,
    panel_rect: Rect,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    cm3_column_adjust_splitter(
        ui,
        Id::unique("seq_col_splitter"),
        panel_rect,
        &mut state.prop_columns,
    );

    let frame = session.playback.current_frame;
    ScrollArea::vertical()
        .id_salt("sequence_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;

            let seq_avail_w = ui.available_width().max(120.0);
            let seq_hdr_ratio =
                sequence_header_label_ratio(seq_avail_w, state.prop_columns.label_ratio);
            let (reset_seq, _) = cm3_collapsible_section_with_action(
                ui,
                Id::unique("seq_general"),
                "Sequence",
                true,
                false,
                seq_hdr_ratio,
                Some((Cm3Icon::Reset, "Reset sequence")),
                theme,
                |ui| {
                    // Resolution: 1920 x 1080 (0 decimals, "x" separator)
                    touch_inspector_tx_owner(state, "sequence", "resolution");
                    let [rw, rh] = session.document().resolution;
                    let mut res_vec = [f64::from(rw), f64::from(rh)];
                    let ev = cm3_property_row_aligned(
                        ui,
                        "Resolution",
                        true,
                        state.prop_columns,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_vector2_with_sep(
                                ui,
                                Id::unique("seq_resolution"),
                                &mut res_vec,
                                &ScrubConfig::new(16.0, 7680.0, 10.0, 0),
                                false,
                                "x",
                                theme,
                            )
                        },
                    );
                    let owns_res_tx = owns_inspector_tx(state, "sequence", "resolution");
                    let res_cmd = || Command::SetResolution {
                        width: res_vec[0].round().clamp(16.0, 7680.0) as u32,
                        height: res_vec[1].round().clamp(16.0, 7680.0) as u32,
                    };
                    if ev.cancelled {
                        if owns_res_tx {
                            if session.history.is_in_transaction() {
                                let _ = session.cancel_transaction();
                            }
                            clear_inspector_tx_state(state);
                        }
                    } else if ev.started && ev.committed {
                        if owns_res_tx {
                            if ev.changed {
                                let _ = session.execute(res_cmd());
                                state.inspector_tx_has_edits = true;
                            }
                            if session.history.is_in_transaction() {
                                let _ = session.commit_transaction(true);
                            }
                            clear_inspector_tx_state(state);
                        } else {
                            close_existing_tx_before_start(session, state);
                            if ev.changed {
                                let _ = session.execute(res_cmd());
                            }
                        }
                        begin_inspector_tx(
                            session,
                            state,
                            "sequence",
                            "resolution",
                            "set resolution",
                        );
                    } else {
                        if ev.started && !ev.committed {
                            close_existing_tx_before_start(session, state);
                            begin_inspector_tx(
                                session,
                                state,
                                "sequence",
                                "resolution",
                                "set resolution",
                            );
                        }
                        let owns_res_tx = owns_inspector_tx(state, "sequence", "resolution");
                        if ev.changed && ev.committed && !ev.started && !owns_res_tx {
                            run_atomic_commit_outside_foreign_tx(session, state, |session| {
                                let _ = session.execute(res_cmd());
                            });
                        } else {
                            if ev.changed || (ev.committed && owns_res_tx) {
                                let _ = session.execute(res_cmd());
                                if owns_res_tx {
                                    state.inspector_tx_has_edits = true;
                                }
                            }
                            if ev.committed && owns_res_tx {
                                if session.history.is_in_transaction() {
                                    let _ = session.commit_transaction(true);
                                }
                                clear_inspector_tx_state(state);
                            }
                        }
                    }

                    // Frame rate: integer 30 (0 decimals)
                    touch_inspector_tx_owner(state, "sequence", "fps");
                    let mut fps = session.document().fps;
                    let ev = cm3_property_row_aligned(
                        ui,
                        "Frame rate",
                        true,
                        state.prop_columns,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_scrub_number(
                                ui,
                                Id::unique("seq_fps"),
                                &mut fps,
                                ScrubConfig::new(1.0, 240.0, 1.0, 0),
                                theme,
                            )
                        },
                    );
                    let owns_fps_tx = owns_inspector_tx(state, "sequence", "fps");
                    if ev.cancelled {
                        if owns_fps_tx {
                            if session.history.is_in_transaction() {
                                let _ = session.cancel_transaction();
                            }
                            clear_inspector_tx_state(state);
                        }
                    } else if ev.started && ev.committed {
                        if owns_fps_tx {
                            if ev.changed {
                                let _ = session.execute(Command::SetFps { fps });
                                state.inspector_tx_has_edits = true;
                            }
                            if session.history.is_in_transaction() {
                                let _ = session.commit_transaction(true);
                            }
                            clear_inspector_tx_state(state);
                        } else {
                            close_existing_tx_before_start(session, state);
                            if ev.changed {
                                let _ = session.execute(Command::SetFps { fps });
                            }
                        }
                        begin_inspector_tx(session, state, "sequence", "fps", "set fps");
                    } else {
                        if ev.started && !ev.committed {
                            close_existing_tx_before_start(session, state);
                            begin_inspector_tx(session, state, "sequence", "fps", "set fps");
                        }
                        let owns_fps_tx = owns_inspector_tx(state, "sequence", "fps");
                        if ev.changed && ev.committed && !ev.started && !owns_fps_tx {
                            run_atomic_commit_outside_foreign_tx(session, state, |session| {
                                let _ = session.execute(Command::SetFps { fps });
                            });
                        } else {
                            if ev.changed || (ev.committed && owns_fps_tx) {
                                let _ = session.execute(Command::SetFps { fps });
                                if owns_fps_tx {
                                    state.inspector_tx_has_edits = true;
                                }
                            }
                            if ev.committed && owns_fps_tx {
                                if session.history.is_in_transaction() {
                                    let _ = session.commit_transaction(true);
                                }
                                clear_inspector_tx_state(state);
                            }
                        }
                    }

                    // Background color (RGBA swatch/picker) + Transparent toggle (alpha 0.0 vs 1.0)
                    touch_inspector_tx_owner(state, "sequence", "backgroundColor");
                    let bg = session.active_sequence().background_color;
                    let mut bg_rgba = Rgba::new(bg[0], bg[1], bg[2], bg[3]);
                    let bg_ev = cm3_property_row_aligned(
                        ui,
                        "Background",
                        true,
                        state.prop_columns,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_hsva_color_picker(
                                ui,
                                Id::unique("seq_background_color"),
                                &mut bg_rgba,
                                true,
                                false,
                                viewport_rect,
                                theme,
                            )
                        },
                    );
                    if bg_ev.eyedropper_clicked {
                        actions.push(PanelAction::EyedropperActivate {
                            target_id: "sequence".to_owned(),
                            property: "backgroundColor".to_owned(),
                        });
                    }
                    let owns_bg_tx = owns_inspector_tx(state, "sequence", "backgroundColor");
                    let new_bg = [bg_rgba.0[0], bg_rgba.0[1], bg_rgba.0[2], bg_rgba.0[3]];
                    if bg_ev.cancelled {
                        if owns_bg_tx {
                            if session.history.is_in_transaction() {
                                let _ = session.cancel_transaction();
                            }
                            clear_inspector_tx_state(state);
                        }
                    } else if bg_ev.started && bg_ev.committed {
                        if owns_bg_tx {
                            if bg_ev.changed {
                                session.active_sequence_mut().background_color = new_bg;
                                session.playback.bump_epoch();
                                state.inspector_tx_has_edits = true;
                            }
                            if session.history.is_in_transaction() {
                                let _ = session.commit_transaction(true);
                            }
                            clear_inspector_tx_state(state);
                        } else {
                            close_existing_tx_before_start(session, state);
                            if bg_ev.changed {
                                set_sequence_background_color_undoable(
                                    session,
                                    new_bg,
                                    "set background color",
                                );
                            }
                        }
                        begin_inspector_tx(
                            session,
                            state,
                            "sequence",
                            "backgroundColor",
                            "set background color",
                        );
                    } else {
                        if bg_ev.started && !bg_ev.committed {
                            close_existing_tx_before_start(session, state);
                            begin_inspector_tx(
                                session,
                                state,
                                "sequence",
                                "backgroundColor",
                                "set background color",
                            );
                        }
                        let owns_bg_tx = owns_inspector_tx(state, "sequence", "backgroundColor");
                        if bg_ev.changed && bg_ev.committed && !bg_ev.started && !owns_bg_tx {
                            run_atomic_commit_outside_foreign_tx(session, state, |session| {
                                set_sequence_background_color_undoable(
                                    session,
                                    new_bg,
                                    "set background color",
                                );
                            });
                        } else {
                            if bg_ev.changed || (bg_ev.committed && owns_bg_tx) {
                                session.active_sequence_mut().background_color = new_bg;
                                session.playback.bump_epoch();
                                if owns_bg_tx {
                                    state.inspector_tx_has_edits = true;
                                }
                            }
                            if bg_ev.committed && owns_bg_tx {
                                if session.history.is_in_transaction() {
                                    let _ = session.commit_transaction(true);
                                }
                                clear_inspector_tx_state(state);
                            }
                        }
                    }

                    let cur_bg = session.active_sequence().background_color;
                    let mut transparent_idx = usize::from(cur_bg[3] <= 0.001);
                    let transparent_changed = cm3_property_row_aligned(
                        ui,
                        "Transparent",
                        true,
                        state.prop_columns,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_dropdown(
                                ui,
                                Id::unique("seq_bg_transparent"),
                                &mut transparent_idx,
                                &["off", "on"],
                                false,
                                viewport_rect,
                                theme,
                            )
                        },
                    );
                    if transparent_changed {
                        let next_alpha = if transparent_idx != 0 { 0.0 } else { 1.0 };
                        let next_bg = [cur_bg[0], cur_bg[1], cur_bg[2], next_alpha];
                        run_atomic_commit_outside_foreign_tx(session, state, |session| {
                            set_sequence_background_color_undoable(
                                session,
                                next_bg,
                                if transparent_idx != 0 {
                                    "enable transparent background"
                                } else {
                                    "disable transparent background"
                                },
                            );
                        });
                    }

                    // Markers: < ◇ > navigator row (operates on the active sequence / precomp)
                    let (prev_marker_frame, exact_marker_idx, next_marker_frame) =
                        find_active_sequence_marker_neighbors(session, frame);

                    let total_w = ui.available_width().max(120.0);
                    let (markers_rect, _) =
                        ui.allocate_exact_size(Vec2::new(total_w, 27.0), Sense::hover());
                    let narrow_markers = total_w < 210.0;
                    let label_indent: f32 = if narrow_markers { 18.0 } else { 36.0 };
                    let ctrl_reserve: f32 = if narrow_markers { 64.0 } else { 74.0 };
                    let min_split = (label_indent + 58.0).min((total_w - ctrl_reserve).max(64.0));
                    let max_split = (total_w - ctrl_reserve).max(min_split);
                    let split_x = (markers_rect.left()
                        + (total_w * state.prop_columns.label_ratio).clamp(min_split, max_split))
                    .round();
                    let label_max_w =
                        (split_x - 5.0 - (markers_rect.left() + label_indent)).max(16.0);
                    let label_clip = Rect::from_min_max(
                        Pos2::new(markers_rect.left() + label_indent, markers_rect.top()),
                        Pos2::new(
                            (split_x - 5.0).max(markers_rect.left() + label_indent),
                            markers_rect.bottom(),
                        ),
                    );
                    let markers_galley = layout_single_line_ellipsized(
                        ui.painter(),
                        "Markers",
                        theme.font(FONT_BODY),
                        TEXT_PRIMARY,
                        label_max_w,
                    );
                    ui.painter().with_clip_rect(label_clip).galley(
                        Pos2::new(
                            markers_rect.left() + label_indent,
                            markers_rect.center().y - markers_galley.size().y * 0.5,
                        ),
                        markers_galley,
                        TEXT_PRIMARY,
                    );
                    let mut go_prev = false;
                    let mut toggle_marker = false;
                    let mut go_next = false;
                    let col2_rect = Rect::from_min_max(
                        Pos2::new(split_x, markers_rect.top() + 3.0),
                        Pos2::new(markers_rect.right(), markers_rect.bottom() + 3.0),
                    );
                    let mut col2_ui = ui.new_child(
                        egui::UiBuilder::new()
                            .max_rect(col2_rect)
                            .layout(egui::Layout::left_to_right(egui::Align::Center)),
                    );
                    col2_ui.set_clip_rect(col2_ui.clip_rect().intersect(col2_rect));
                    col2_ui.spacing_mut().item_spacing.x = if narrow_markers { 4.0 } else { 7.0 };
                    if cm3_icon_button(
                        &mut col2_ui,
                        Cm3Icon::PrevKf,
                        17.0,
                        TEXT_PRIMARY,
                        "Previous marker",
                        theme,
                    ) {
                        go_prev = true;
                    }
                    let (icon, tip) = if exact_marker_idx.is_some() {
                        (Cm3Icon::KeyframeSolid, "Remove marker at current frame")
                    } else {
                        (Cm3Icon::KeyframeHollow, "Add marker at current frame")
                    };
                    if cm3_icon_button(&mut col2_ui, icon, 17.0, TEXT_PRIMARY, tip, theme) {
                        toggle_marker = true;
                    }
                    if cm3_icon_button(
                        &mut col2_ui,
                        Cm3Icon::NextKf,
                        17.0,
                        TEXT_PRIMARY,
                        "Next marker",
                        theme,
                    ) {
                        go_next = true;
                    }
                    if go_prev && let Some(target_f) = prev_marker_frame {
                        let _ = session.execute(Command::SetFrame { frame: target_f });
                    }
                    if go_next && let Some(target_f) = next_marker_frame {
                        let _ = session.execute(Command::SetFrame { frame: target_f });
                    }
                    if toggle_marker {
                        toggle_active_sequence_marker(session, frame);
                    }

                    // 1px #242424 divider directly between Markers and Multisample motion blur
                    let (gap_div, _) =
                        ui.allocate_exact_size(Vec2::new(total_w, 1.0), Sense::hover());
                    ui.painter()
                        .rect_filled(gap_div, CornerRadius::ZERO, BORDER_ROW_SOFT);

                    // Multisample motion blur (compact dropdown "off"/"on", NO stopwatch)
                    let mut mb_idx = eval_prop_u32(
                        lookup_session_prop_ref(session, "sequence", "motionBlur.enabled"),
                        frame,
                        0,
                    ) as usize;
                    let mb_changed = cm3_property_row_aligned(
                        ui,
                        "Multisample motion blur",
                        true,
                        state.prop_columns,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_dropdown(
                                ui,
                                Id::unique("seq_mb_enabled"),
                                &mut mb_idx,
                                &["off", "on"],
                                false,
                                viewport_rect,
                                theme,
                            )
                        },
                    );
                    if mb_changed {
                        let _ = session.execute(Command::SetPropertyValue {
                            target_id: "sequence".to_owned(),
                            property: "motionBlur.enabled".to_owned(),
                            value: json!(mb_idx),
                        });
                    }

                    // Samples (16, with stopwatch) & Shutter (0.50, with stopwatch)
                    let (samples_lbl, shutter_lbl) = sequence_motion_blur_labels(seq_avail_w);
                    let prev_seq_ratio = state.prop_columns.label_ratio;
                    if seq_avail_w < 245.0 {
                        state.prop_columns.label_ratio =
                            prev_seq_ratio.max((106.0 / seq_avail_w).min(0.56));
                    }
                    show_session_number_prop(
                        ui,
                        session,
                        "sequence",
                        "motionBlur.samples",
                        samples_lbl,
                        16.0,
                        ScrubConfig::new(1.0, 64.0, 1.0, 0),
                        frame,
                        0.0,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                    show_session_number_prop(
                        ui,
                        session,
                        "sequence",
                        "motionBlur.shutter",
                        shutter_lbl,
                        0.5,
                        ScrubConfig::new(0.0, 2.0, 0.05, 2),
                        frame,
                        0.0,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                    state.prop_columns.label_ratio = prev_seq_ratio;

                    // Selected visual clip Track Matte controls (video / image / shape / text / scene)
                    if let Some(sel_clip_id) = session.selection.clip_ids.first().cloned()
                        && let Some(sel_clip) = find_clip_cached(session, state, &sel_clip_id)
                        && sel_clip.kind != "audio"
                    {
                        let matte_active = read_clip_track_matte_mode(session, &sel_clip_id) > 0;
                        cm3_collapsible_section_with_action(
                            ui,
                            Id::unique(("seq_selected_track_matte", &sel_clip_id)),
                            "Track Matte",
                            matte_active,
                            true,
                            state.prop_columns.label_ratio,
                            None,
                            theme,
                            |ui| {
                                show_clip_track_matte_rows(
                                    ui,
                                    session,
                                    &sel_clip_id,
                                    viewport_rect,
                                    state,
                                    theme,
                                );
                            },
                        );
                    }

                    // Controls nested section (Expressions+): full SequenceControlKind builder
                    // (Slider, Angle, Checkbox, Color, Point2D, Point3D) + legacy ctl_* compatibility.
                    if session.plugins.is_enabled("expressions-plus") {
                        show_sequence_controls_section(
                            ui,
                            session,
                            frame,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                },
            );

            if reset_seq {
                let started_tx = session.begin_transaction("reset sequence").is_ok();
                let _ = session.execute(Command::SetResolution {
                    width: 1920,
                    height: 1080,
                });
                let _ = session.execute(Command::SetFps { fps: 30.0 });
                session.active_sequence_mut().background_color = [0.0, 0.0, 0.0, 1.0];
                session.playback.bump_epoch();
                let _ = session.execute(Command::ResetPropertyList {
                    target_id: "sequence".to_owned(),
                });
                if started_tx && session.history.is_in_transaction() {
                    let _ = session.commit_transaction(true);
                }
            }
        });
}

// ---------------------------------------------------------------------------
// 2. Expressions+ Sequence Controls Builder (Sequence Tab)
// ---------------------------------------------------------------------------

/// Sanitizes a sequence control name into a clean JS-safe identifier (no spaces or punctuation),
/// e.g. `"Slider 1"` -> `"slider_1"` or `"ctl_slider_1"` when empty/invalid.
pub(super) fn sanitize_control_identifier(raw: &str, fallback: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_underscore = false;
    for ch in raw.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_underscore = false;
        } else if (ch == '_' || ch == '-' || ch.is_whitespace())
            && !prev_underscore
            && !out.is_empty()
        {
            out.push('_');
            prev_underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() {
        return fallback.to_owned();
    }
    if out.as_bytes()[0].is_ascii_digit() {
        format!("ctl_{out}")
    } else {
        out
    }
}

#[allow(clippy::too_many_arguments)]
fn show_sequence_controls_section(
    ui: &mut Ui,
    session: &mut Session,
    frame: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    const KIND_LABELS: &[&str] = &["Slider", "Angle", "Checkbox", "Color", "Point2D", "Point3D"];
    const KIND_SLUGS: &[&str] = &["slider", "angle", "check", "color", "point2d", "point3d"];
    const KINDS: [SequenceControlKind; 6] = [
        SequenceControlKind::Slider,
        SequenceControlKind::Angle,
        SequenceControlKind::Checkbox,
        SequenceControlKind::Color,
        SequenceControlKind::Point2D,
        SequenceControlKind::Point3D,
    ];

    cm3_collapsible_section_with_action(
        ui,
        Id::unique("seq_expr_controls"),
        "Controls",
        true,
        true,
        state.prop_columns.label_ratio,
        None,
        theme,
        |ui| {
            // Responsive add-control builder for all 6 SequenceControlKind types + min/max/step.
            // Stacks cleanly when panel width < 240px (e.g. 800x540) so `+` and `Step` never clip.
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(6, 4))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 4.0;
                    let form_w = ui.available_width();
                    let narrow_form = form_w < 240.0;
                    let mut add_clicked = false;
                    if narrow_form {
                        ui.add(
                            TextEdit::singleline(&mut state.new_seq_control_name)
                                .hint_text("ctl_slider_1")
                                .font(theme.font(FONT_SMALL))
                                .desired_width(form_w.max(80.0)),
                        );
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            cm3_dropdown(
                                ui,
                                Id::unique("seq_new_ctl_kind"),
                                &mut state.new_seq_control_kind,
                                KIND_LABELS,
                                false,
                                viewport_rect,
                                theme,
                            );
                            if cm3_icon_button(
                                ui,
                                Cm3Icon::Add,
                                16.0,
                                TEXT_PRIMARY,
                                "Add sequence control",
                                theme,
                            ) {
                                add_clicked = true;
                            }
                        });
                    } else {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            let name_w = (form_w - 118.0).clamp(72.0, 140.0);
                            ui.add(
                                TextEdit::singleline(&mut state.new_seq_control_name)
                                    .hint_text("ctl_slider_1")
                                    .font(theme.font(FONT_SMALL))
                                    .desired_width(name_w),
                            );
                            cm3_dropdown(
                                ui,
                                Id::unique("seq_new_ctl_kind"),
                                &mut state.new_seq_control_kind,
                                KIND_LABELS,
                                false,
                                viewport_rect,
                                theme,
                            );
                            if cm3_icon_button(
                                ui,
                                Cm3Icon::Add,
                                16.0,
                                TEXT_PRIMARY,
                                "Add sequence control",
                                theme,
                            ) {
                                add_clicked = true;
                            }
                        });
                    }
                    if add_clicked {
                        let kind_idx = state.new_seq_control_kind.min(KINDS.len() - 1);
                        let kind = KINDS[kind_idx];
                        let fallback_name =
                            format!("{} {}", KIND_LABELS[kind_idx], state.seq_control_counter);
                        let trimmed = state.new_seq_control_name.trim();
                        let name = if trimmed.is_empty() {
                            fallback_name
                        } else {
                            trimmed.to_owned()
                        };
                        state.seq_control_counter += 1;
                        state.new_seq_control_name =
                            format!("{} {}", KIND_LABELS[kind_idx], state.seq_control_counter);
                        let default_val = match kind {
                            SequenceControlKind::Slider | SequenceControlKind::Angle => {
                                PropertyValue::Number(0.0)
                            }
                            SequenceControlKind::Checkbox => PropertyValue::Bool(false),
                            SequenceControlKind::Color => PropertyValue::Color([1.0, 1.0, 1.0]),
                            SequenceControlKind::Point2D => PropertyValue::Vec2([0.0, 0.0]),
                            SequenceControlKind::Point3D => PropertyValue::Vec3([0.0, 0.0, 0.0]),
                        };
                        let min = state.new_seq_control_min.trim().parse::<f64>().ok();
                        let max = state.new_seq_control_max.trim().parse::<f64>().ok();
                        let step = state.new_seq_control_step.trim().parse::<f64>().ok();
                        let _ = session.execute(Command::AddSequenceControl {
                            name,
                            kind,
                            value: default_val,
                            min,
                            max,
                            step,
                        });
                    }
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(
                                "Use control(\"Name\") or controls.<name> in expressions",
                            )
                            .font(theme.font(FONT_COMPACT))
                            .color(TEXT_MUTED),
                        )
                        .wrap(),
                    );
                    if state.new_seq_control_kind <= 1 {
                        if narrow_form {
                            let field_w = ((form_w - 64.0) * 0.5).clamp(32.0, 64.0);
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 4.0;
                                ui.label(
                                    egui::RichText::new("Min")
                                        .font(theme.font(FONT_COMPACT))
                                        .color(TEXT_MUTED),
                                );
                                ui.add(
                                    TextEdit::singleline(&mut state.new_seq_control_min)
                                        .font(theme.font(FONT_COMPACT))
                                        .desired_width(field_w),
                                );
                                ui.label(
                                    egui::RichText::new("Max")
                                        .font(theme.font(FONT_COMPACT))
                                        .color(TEXT_MUTED),
                                );
                                ui.add(
                                    TextEdit::singleline(&mut state.new_seq_control_max)
                                        .font(theme.font(FONT_COMPACT))
                                        .desired_width(field_w),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 4.0;
                                ui.label(
                                    egui::RichText::new("Step")
                                        .font(theme.font(FONT_COMPACT))
                                        .color(TEXT_MUTED),
                                );
                                ui.add(
                                    TextEdit::singleline(&mut state.new_seq_control_step)
                                        .font(theme.font(FONT_COMPACT))
                                        .desired_width(field_w),
                                );
                            });
                        } else {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 4.0;
                                ui.label(
                                    egui::RichText::new("Min")
                                        .font(theme.font(FONT_COMPACT))
                                        .color(TEXT_MUTED),
                                );
                                ui.add(
                                    TextEdit::singleline(&mut state.new_seq_control_min)
                                        .font(theme.font(FONT_COMPACT))
                                        .desired_width(40.0),
                                );
                                ui.label(
                                    egui::RichText::new("Max")
                                        .font(theme.font(FONT_COMPACT))
                                        .color(TEXT_MUTED),
                                );
                                ui.add(
                                    TextEdit::singleline(&mut state.new_seq_control_max)
                                        .font(theme.font(FONT_COMPACT))
                                        .desired_width(40.0),
                                );
                                ui.label(
                                    egui::RichText::new("Step")
                                        .font(theme.font(FONT_COMPACT))
                                        .color(TEXT_MUTED),
                                );
                                ui.add(
                                    TextEdit::singleline(&mut state.new_seq_control_step)
                                        .font(theme.font(FONT_COMPACT))
                                        .desired_width(36.0),
                                );
                            });
                        }
                    }
                });

            // Render project SequenceControl items
            let _ = KIND_SLUGS;
            let controls = session.active_sequence().controls.clone();
            let count = controls.len();
            for (idx, ctl) in controls.into_iter().enumerate() {
                let mut min_v = ctl.preserved.get("min").and_then(Value::as_f64);
                let mut max_v = ctl.preserved.get("max").and_then(Value::as_f64);
                let mut step_v = ctl.preserved.get("step").and_then(Value::as_f64);
                let mut name_buf = ctl.name.clone();
                let mut updated_val: Option<PropertyValue> = None;
                let mut name_changed = false;
                let mut meta_changed = false;
                let mut tx_tracker = WidgetTxTracker::default();
                let mirrored_key = session
                    .document()
                    .sequence_control_order
                    .iter()
                    .find(|(_, id)| id == &ctl.id)
                    .map(|(k, _)| k.clone())
                    .unwrap_or_else(|| {
                        if ctl.name.starts_with("ctl_") {
                            ctl.name.clone()
                        } else {
                            format!("ctl_{}", ctl.name)
                        }
                    });
                let mirrored_prop = lookup_session_prop_ref(session, "sequence", &mirrored_key);
                let ctl_animated = mirrored_prop.is_some_and(|p| p.animated);
                let ctl_expr = mirrored_prop.and_then(|p| p.expression.clone());
                let ctl_has_expr = ctl_expr.is_some();
                let mut toggle_ctl_anim = false;
                let mut open_ctl_expr = false;

                Frame::NONE
                    .inner_margin(egui::Margin::symmetric(6, 3))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 3.0;
                            let sw_ev = cm3_stopwatch_button(
                                ui,
                                &mirrored_key,
                                ctl_animated,
                                ctl_has_expr,
                                theme,
                            );
                            if sw_ev.toggled {
                                toggle_ctl_anim = true;
                            }
                            if sw_ev.alt_expression {
                                open_ctl_expr = true;
                            }
                            let fx_color = if ctl_has_expr {
                                theme.value()
                            } else {
                                TEXT_MUTED
                            };
                            if cm3_icon_button(
                                ui,
                                Cm3Icon::Code,
                                15.0,
                                fx_color,
                                "Edit control expression",
                                theme,
                            ) {
                                open_ctl_expr = true;
                            }
                            let btn_count =
                                1usize + usize::from(idx + 1 < count) + usize::from(idx > 0);
                            let btns_reserve = btn_count as f32 * 24.0 + 4.0;
                            let name_w = (ui.available_width() - btns_reserve).max(56.0);
                            if ui
                                .add(
                                    TextEdit::singleline(&mut name_buf)
                                        .font(theme.font(FONT_SMALL))
                                        .desired_width(name_w),
                                )
                                .lost_focus()
                                && !name_buf.trim().is_empty()
                            {
                                let trimmed = name_buf.trim().to_owned();
                                if trimmed != ctl.name {
                                    name_buf = trimmed;
                                    name_changed = true;
                                }
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .small_button("×")
                                        .on_hover_text("Delete control")
                                        .clicked()
                                    {
                                        let _ = session.execute(Command::DeleteSequenceControl {
                                            control_id: ctl.id.clone(),
                                        });
                                    }
                                    if idx + 1 < count
                                        && ui.small_button("↓").on_hover_text("Move down").clicked()
                                    {
                                        let _ = session.execute(Command::ReorderSequenceControl {
                                            control_id: ctl.id.clone(),
                                            to_index: idx + 1,
                                        });
                                    }
                                    if idx > 0
                                        && ui.small_button("↑").on_hover_text("Move up").clicked()
                                    {
                                        let _ = session.execute(Command::ReorderSequenceControl {
                                            control_id: ctl.id.clone(),
                                            to_index: idx - 1,
                                        });
                                    }
                                },
                            );
                        });
                        // Typed value editor
                        match ctl.kind {
                            SequenceControlKind::Slider => {
                                let mut v = ctl.track.value.as_f64().unwrap_or(0.0);
                                let mut lo = min_v.unwrap_or(0.0);
                                let mut hi = max_v.unwrap_or(100.0).max(lo + 1e-3);
                                let mut st = step_v.unwrap_or(0.1).max(0.001);
                                ui.spacing_mut().slider_width =
                                    (ui.available_width() - 68.0).max(64.0);
                                let resp = ui.add(
                                    egui::Slider::new(&mut v, lo..=hi)
                                        .step_by(st)
                                        .clamping(egui::SliderClamping::Never),
                                );
                                if tx_tracker.observe(&resp) {
                                    updated_val = Some(PropertyValue::Number(v));
                                }
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 4.0;
                                    ui.label(
                                        egui::RichText::new("Min")
                                            .font(theme.font(FONT_COMPACT))
                                            .color(TEXT_MUTED),
                                    );
                                    let r_lo = ui.add(
                                        egui::DragValue::new(&mut lo).speed(0.5).max_decimals(3),
                                    );
                                    ui.label(
                                        egui::RichText::new("Max")
                                            .font(theme.font(FONT_COMPACT))
                                            .color(TEXT_MUTED),
                                    );
                                    let r_hi = ui.add(
                                        egui::DragValue::new(&mut hi).speed(0.5).max_decimals(3),
                                    );
                                    ui.label(
                                        egui::RichText::new("Step")
                                            .font(theme.font(FONT_COMPACT))
                                            .color(TEXT_MUTED),
                                    );
                                    let r_st = ui.add(
                                        egui::DragValue::new(&mut st)
                                            .range(0.001..=1000.0)
                                            .speed(0.05)
                                            .max_decimals(3),
                                    );
                                    if tx_tracker.observe(&r_lo)
                                        || tx_tracker.observe(&r_hi)
                                        || tx_tracker.observe(&r_st)
                                    {
                                        min_v = Some(lo);
                                        max_v = Some(hi.max(lo + 1e-3));
                                        step_v = Some(st.max(0.001));
                                        meta_changed = true;
                                    }
                                });
                            }
                            SequenceControlKind::Angle => {
                                let mut v = ctl.track.value.as_f64().unwrap_or(0.0);
                                let lo = min_v.unwrap_or(-360.0);
                                let hi = max_v.unwrap_or(360.0);
                                let st = step_v.unwrap_or(0.5);
                                let resp = ui.add(
                                    egui::DragValue::new(&mut v)
                                        .range(lo..=hi)
                                        .speed(st)
                                        .suffix("°"),
                                );
                                if tx_tracker.observe(&resp) {
                                    updated_val = Some(PropertyValue::Number(v));
                                }
                            }
                            SequenceControlKind::Checkbox => {
                                let mut b = match &ctl.track.value {
                                    PropertyValue::Bool(v) => *v,
                                    PropertyValue::Number(n) => *n >= 0.5,
                                    _ => false,
                                };
                                if ui.checkbox(&mut b, "Enabled").changed() {
                                    updated_val = Some(PropertyValue::Bool(b));
                                }
                            }
                            SequenceControlKind::Color => {
                                let c = ctl.track.value.as_vec3().unwrap_or([1.0, 1.0, 1.0]);
                                let mut rgb = [c[0] as f32, c[1] as f32, c[2] as f32];
                                let resp = ui.color_edit_button_rgb(&mut rgb);
                                if tx_tracker.observe(&resp) {
                                    updated_val = Some(PropertyValue::Color([
                                        f64::from(rgb[0]),
                                        f64::from(rgb[1]),
                                        f64::from(rgb[2]),
                                    ]));
                                }
                            }
                            SequenceControlKind::Point2D => {
                                let mut p = ctl.track.value.as_vec2().unwrap_or([0.0, 0.0]);
                                ui.horizontal(|ui| {
                                    let cx = ui.add(
                                        egui::DragValue::new(&mut p[0]).speed(1.0).prefix("X "),
                                    );
                                    let cy = ui.add(
                                        egui::DragValue::new(&mut p[1]).speed(1.0).prefix("Y "),
                                    );
                                    let ch_x = tx_tracker.observe(&cx);
                                    let ch_y = tx_tracker.observe(&cy);
                                    if ch_x || ch_y {
                                        updated_val = Some(PropertyValue::Vec2(p));
                                    }
                                });
                            }
                            SequenceControlKind::Point3D => {
                                let mut p = ctl.track.value.as_vec3().unwrap_or([0.0, 0.0, 0.0]);
                                ui.horizontal(|ui| {
                                    let cx = ui.add(
                                        egui::DragValue::new(&mut p[0]).speed(1.0).prefix("X "),
                                    );
                                    let cy = ui.add(
                                        egui::DragValue::new(&mut p[1]).speed(1.0).prefix("Y "),
                                    );
                                    let cz = ui.add(
                                        egui::DragValue::new(&mut p[2]).speed(1.0).prefix("Z "),
                                    );
                                    let ch_x = tx_tracker.observe(&cx);
                                    let ch_y = tx_tracker.observe(&cy);
                                    let ch_z = tx_tracker.observe(&cz);
                                    if ch_x || ch_y || ch_z {
                                        updated_val = Some(PropertyValue::Vec3(p));
                                    }
                                });
                            }
                        }
                    });

                if toggle_ctl_anim {
                    let _ = session.execute(Command::TogglePropertyAnimated {
                        target_id: "sequence".to_owned(),
                        property: mirrored_key.clone(),
                        frame: normalized_kf_frame(session, frame),
                    });
                }
                if open_ctl_expr {
                    let init_code = ctl_expr.unwrap_or_else(|| match &ctl.track.value {
                        PropertyValue::Number(n) => format!("{n}"),
                        PropertyValue::Bool(b) => if *b { "1" } else { "0" }.to_owned(),
                        PropertyValue::Vec2(v) => format!("[{}, {}]", v[0], v[1]),
                        PropertyValue::Vec3(v) | PropertyValue::Color(v) => {
                            format!("[{}, {}, {}]", v[0], v[1], v[2])
                        }
                        _ => "0".to_owned(),
                    });
                    if !ctl_has_expr {
                        let _ = session.execute(Command::SetExpression {
                            target_id: "sequence".to_owned(),
                            property: mirrored_key.clone(),
                            expression: Some(init_code.clone()),
                        });
                    }
                    open_expression_window(
                        "sequence",
                        &mirrored_key,
                        init_code,
                        state,
                        viewport_rect,
                    );
                }

                if name_changed || meta_changed || updated_val.is_some() {
                    tx_tracker.changed = true;
                }
                let ctl_owner_key = format!("seq_ctl_{}", ctl.id);
                apply_widget_tx_command(
                    ui,
                    session,
                    state,
                    "sequence",
                    &ctl_owner_key,
                    "edit sequence control",
                    tx_tracker,
                    |session| {
                        let _ = session.execute(Command::UpdateSequenceControl {
                            control_id: ctl.id,
                            name: name_buf.trim().to_owned(),
                            value: updated_val.unwrap_or(ctl.track.value),
                            min: min_v,
                            max: max_v,
                            step: step_v,
                        });
                    },
                );
            }

            // Legacy ctl_* sequence_properties compatibility (skip keys already backed by
            // `active_sequence().controls` so typed SequenceControls never render twice!)
            let mirrored_ctl_keys = collect_mirrored_sequence_control_keys(session);
            let ctl_keys: Vec<String> = session
                .document()
                .sequence_properties
                .keys()
                .filter(|k| k.starts_with("ctl_") && !mirrored_ctl_keys.contains(k.as_str()))
                .cloned()
                .collect();
            for key in ctl_keys {
                let display_lbl = key.strip_prefix("ctl_").unwrap_or(&key);
                let label = humanize_prop_label(display_lbl);
                show_session_number_prop(
                    ui,
                    session,
                    "sequence",
                    &key,
                    &label,
                    0.0,
                    ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 2),
                    frame,
                    0.0,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            }
        },
    );
}

/// Returns the set of `sequence_properties` keys already backed by `active_sequence().controls`.
pub(super) fn collect_mirrored_sequence_control_keys(session: &Session) -> BTreeSet<String> {
    let mut base_prefixes: Vec<(String, String)> = Vec::new();
    let mut keys: BTreeSet<String> = BTreeSet::new();
    for ctl in &session.active_sequence().controls {
        let base = if ctl.name.starts_with("ctl_") {
            ctl.name.clone()
        } else {
            format!("ctl_{}", ctl.name)
        };
        keys.insert(base.clone());
        if !ctl.id.is_empty() {
            keys.insert(format!("{base}__{}", ctl.id));
        }
        base_prefixes.push((format!("{base}__"), ctl.id.clone()));
    }
    for prop_key in session.document().sequence_properties.keys() {
        if base_prefixes
            .iter()
            .any(|(prefix, _)| prop_key.starts_with(prefix))
        {
            keys.insert(prop_key.clone());
        }
    }
    keys
}

/// Finds `(previous_marker_frame, exact_marker_idx, next_marker_frame)` on the currently active
/// sequence or nested precomp around `frame`.
pub(super) fn find_active_sequence_marker_neighbors(
    session: &Session,
    frame: f64,
) -> (Option<f64>, Option<usize>, Option<f64>) {
    let mut prev_marker_frame: Option<f64> = None;
    let mut next_marker_frame: Option<f64> = None;
    let mut exact_marker_idx: Option<usize> = None;
    for (idx, m) in session.active_sequence().markers.iter().enumerate() {
        if (m.frame - frame).abs() < 0.5 {
            exact_marker_idx = Some(idx);
        } else if m.frame < frame - 0.5 {
            prev_marker_frame = Some(match prev_marker_frame {
                Some(best) => best.max(m.frame),
                None => m.frame,
            });
        } else if m.frame > frame + 0.5 {
            next_marker_frame = Some(match next_marker_frame {
                Some(best) => best.min(m.frame),
                None => m.frame,
            });
        }
    }
    (prev_marker_frame, exact_marker_idx, next_marker_frame)
}

/// Adds or removes a marker at `frame` on the currently active sequence / nested precomp
/// inside an undoable transaction.
pub(super) fn toggle_active_sequence_marker(session: &mut Session, frame: f64) {
    let (_, exact_marker_idx, _) = find_active_sequence_marker_neighbors(session, frame);
    let label = if exact_marker_idx.is_some() {
        "Remove marker"
    } else {
        "Add marker"
    };
    let started_tx = if !session.history.is_in_transaction() {
        session.begin_transaction(label).is_ok()
    } else {
        false
    };
    if let Some(idx) = exact_marker_idx {
        session.active_sequence_mut().markers.remove(idx);
    } else {
        let f = frame.round().max(0.0);
        session.active_sequence_mut().markers.push(Marker {
            frame: f,
            label: format!("Marker {}", f as u64),
        });
        session
            .active_sequence_mut()
            .markers
            .sort_by(|a, b| a.frame.total_cmp(&b.frame));
    }
    if started_tx && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    }
}

#[cfg(test)]
mod sequence_marker_tests {
    use super::{find_active_sequence_marker_neighbors, toggle_active_sequence_marker};
    use zc_project::Uuid;
    use zc_session::{Command, Session};

    #[test]
    fn active_precomp_markers_do_not_mutate_root_sequence() {
        let mut session = Session::new();
        let created = session
            .execute(Command::CreateComposition {
                name: "Precomp Markers".to_owned(),
                resolution: [1280, 720],
                fps: 30,
                length_frames: 120,
            })
            .expect("create precomp");
        let comp_id =
            Uuid::parse_str(created["comp_id"].as_str().expect("comp_id")).expect("comp uuid");

        session
            .execute(Command::SetActiveComposition {
                comp_id: Some(comp_id),
            })
            .expect("activate precomp");

        toggle_active_sequence_marker(&mut session, 24.0);
        assert!(
            session.project.sequence.markers.is_empty(),
            "root sequence markers must remain untouched when editing a precomp"
        );
        assert_eq!(session.active_sequence().markers.len(), 1);
        assert_eq!(
            find_active_sequence_marker_neighbors(&session, 24.0),
            (None, Some(0), None)
        );
        assert_eq!(
            find_active_sequence_marker_neighbors(&session, 10.0),
            (None, None, Some(24.0))
        );

        toggle_active_sequence_marker(&mut session, 24.0);
        assert!(session.active_sequence().markers.is_empty());
    }
}
