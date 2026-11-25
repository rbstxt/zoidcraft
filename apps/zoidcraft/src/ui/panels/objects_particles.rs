use super::property_rows::{
    eval_prop_str, lookup_session_prop_ref, show_session_number_prop, show_session_option_prop_ex,
    show_session_vec3_prop,
};
use super::shared_widgets::show_session_asset_prop;
use super::state::{WidgetTxTracker, apply_widget_tx_command};
use super::*;

// ---------------------------------------------------------------------------
// 7. Full 5-Tab 3D Particles Inspector + 54-Sprite Picker + Over-Life Stops
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) fn show_particles_3d_inspector(
    ui: &mut Ui,
    session: &mut Session,
    obj_id: &str,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    const TABS: [&str; 5] = ["Emitter", "Position", "Motion", "Rotation", "Appearance"];
    Frame::NONE
        .inner_margin(egui::Margin::symmetric(6, 4))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 3.0;
                for (i, lbl) in TABS.iter().enumerate() {
                    if ui
                        .selectable_label(state.particles_tab == i, *lbl)
                        .clicked()
                    {
                        state.particles_tab = i;
                    }
                }
            });
        });

    match state.particles_tab {
        0 => {
            // Emitter
            show_session_number_prop(
                ui,
                session,
                obj_id,
                "particles.time",
                "Time",
                0.0,
                ScrubConfig::new(-1000.0, 10_000.0, 0.05, 2),
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
                obj_id,
                "particles.number",
                "Max count",
                100.0,
                ScrubConfig::new(1.0, 100_000.0, 1.0, 0),
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
                obj_id,
                "particles.rate",
                "Emission rate",
                25.0,
                ScrubConfig::new(0.0, 10_000.0, 0.5, 2),
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
                obj_id,
                "particles.lifetime",
                "Lifetime (s)",
                2.0,
                ScrubConfig::new(0.01, 600.0, 0.1, 2),
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        1 => {
            // Position
            show_session_option_prop_ex(
                ui,
                session,
                obj_id,
                "particles.pdist",
                "Distribution",
                &["Box", "Sphere", "Disc"],
                0,
                false,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_vec3_prop(
                ui,
                session,
                obj_id,
                "particles.ipos",
                "Init position",
                [0.0, 0.0, 0.0],
                ScrubConfig::new(-10_000.0, 10_000.0, 0.5, 2),
                false,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_vec3_prop(
                ui,
                session,
                obj_id,
                "particles.pspread",
                "Position spread",
                [10.0, 10.0, 10.0],
                ScrubConfig::new(0.0, 10_000.0, 0.5, 2),
                false,
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
                obj_id,
                "particles.iradius",
                "Init radius",
                5.0,
                ScrubConfig::new(0.0, 10_000.0, 0.25, 2),
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
                obj_id,
                "particles.rspread",
                "Radius spread",
                2.0,
                ScrubConfig::new(0.0, 10_000.0, 0.25, 2),
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        2 => {
            // Motion
            show_session_option_prop_ex(
                ui,
                session,
                obj_id,
                "particles.vdist",
                "Vel distribution",
                &["Box", "Sphere", "Radial"],
                0,
                false,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_vec3_prop(
                ui,
                session,
                obj_id,
                "particles.ivel",
                "Init velocity",
                [0.0, 10.0, 0.0],
                ScrubConfig::new(-10_000.0, 10_000.0, 0.5, 2),
                false,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_vec3_prop(
                ui,
                session,
                obj_id,
                "particles.vspread",
                "Velocity spread",
                [5.0, 5.0, 5.0],
                ScrubConfig::new(0.0, 10_000.0, 0.5, 2),
                false,
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
                obj_id,
                "particles.irvel",
                "Radial velocity",
                0.0,
                ScrubConfig::new(-10_000.0, 10_000.0, 0.25, 2),
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
                obj_id,
                "particles.rvspread",
                "Radial spread",
                0.0,
                ScrubConfig::new(0.0, 10_000.0, 0.25, 2),
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_vec3_prop(
                ui,
                session,
                obj_id,
                "particles.accel",
                "Acceleration",
                [0.0, -9.8, 0.0],
                ScrubConfig::new(-10_000.0, 10_000.0, 0.5, 2),
                false,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        3 => {
            // Rotation
            show_session_number_prop(
                ui,
                session,
                obj_id,
                "particles.iang",
                "Init angle",
                0.0,
                ScrubConfig::new(-3600.0, 3600.0, 1.0, 1),
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
                obj_id,
                "particles.aspread",
                "Angle spread",
                0.0,
                ScrubConfig::new(0.0, 360.0, 1.0, 1),
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
                obj_id,
                "particles.angvel",
                "Angular vel",
                0.0,
                ScrubConfig::new(-3600.0, 3600.0, 1.0, 1),
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
                obj_id,
                "particles.avspread",
                "Ang vel spread",
                0.0,
                ScrubConfig::new(0.0, 3600.0, 1.0, 1),
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        _ => {
            // Appearance: Blending, Texture, Searchable 54-Sprite Picker, Color-over-Life, Size-over-Life
            show_session_option_prop_ex(
                ui,
                session,
                obj_id,
                "particles.blending",
                "Blending",
                &["Normal", "Additive", "Subtractive", "Multiply"],
                1,
                false,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_asset_prop(
                ui,
                session,
                obj_id,
                "particles.texture",
                "Custom texture",
                false,
                local_frame,
                viewport_rect,
                state,
                theme,
                actions,
            );

            // Searchable 54-sprite picker (Particles+ plugin, collapsed by default so
            // Color over Life and Size over Life are immediately visible without scrolling)
            let particles_plus_enabled = session.plugins.is_enabled("particles-plus");
            let cur_sprite = eval_prop_str(
                lookup_session_prop_ref(session, obj_id, "particles.spritePreset"),
                local_frame,
                "soft_glow",
            );
            let sprite_hdr_rect =
                Rect::from_min_size(ui.cursor().min, Vec2::new(ui.available_width(), 27.0));
            cm3_collapsible_section(
                ui,
                Id::unique((obj_id, "particles_sprite_picker")),
                &format!("Built-in Sprites ({})", all_particle_sprite_names().len()),
                false,
                theme,
                |ui| {
                    if !particles_plus_enabled {
                        Frame::NONE
                            .inner_margin(egui::Margin::symmetric(8, 6))
                            .show(ui, |ui| {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(
                                            "Enable the Particles+ plugin in Plugins to browse all 54 procedural sprites.",
                                        )
                                        .font(theme.font(FONT_SMALL))
                                        .color(TEXT_MUTED),
                                    )
                                    .wrap(),
                                );
                            });
                        return;
                    }
                    Frame::NONE
                        .inner_margin(egui::Margin::symmetric(6, 4))
                        .show(ui, |ui| {
                            ui.add(
                                TextEdit::singleline(&mut state.particles_sprite_filter)
                                    .hint_text("Filter 54 sprites...")
                                    .font(theme.font(FONT_SMALL))
                                    .desired_width(ui.available_width()),
                            );
                            ui.add_space(4.0);
                            let q = state.particles_sprite_filter.trim().to_ascii_lowercase();
                            ScrollArea::vertical()
                                .id_salt("particles_sprite_grid")
                                .max_height(130.0)
                                .show(ui, |ui| {
                                    ui.horizontal_wrapped(|ui| {
                                        ui.spacing_mut().item_spacing = Vec2::splat(3.0);
                                        for &sp in all_particle_sprite_names() {
                                            if !q.is_empty()
                                                && !sp.to_ascii_lowercase().contains(&q)
                                            {
                                                continue;
                                            }
                                            let active = sp == cur_sprite;
                                            if ui
                                                .selectable_label(
                                                    active,
                                                    egui::RichText::new(sp)
                                                        .font(theme.font(FONT_COMPACT)),
                                                )
                                                .clicked()
                                            {
                                                let _ =
                                                    session.execute(Command::SetPropertyValue {
                                                        target_id: obj_id.to_owned(),
                                                        property: "particles.spritePreset"
                                                            .to_owned(),
                                                        value: json!(sp),
                                                    });
                                            }
                                        }
                                    });
                                });
                        });
                },
            );
            ui.interact(
                sprite_hdr_rect,
                Id::unique((obj_id, "particles_sprite_picker_tip")),
                Sense::hover(),
            )
            .on_hover_text("Browse 54 procedural sprites (Particles+)");

            // Color-over-Life stop editor (`particles.color` JSON)
            show_particle_color_over_life_editor(ui, session, state, obj_id, local_frame, theme);
            // Size-over-Life stop editor (`particles.size` JSON)
            show_particle_size_over_life_editor(ui, session, state, obj_id, local_frame, theme);
        }
    }
}

fn show_particle_color_over_life_editor(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut PanelsState,
    obj_id: &str,
    local_frame: f64,
    theme: &Cm3Theme,
) {
    cm3_collapsible_section(
        ui,
        Id::unique((obj_id, "particles_color_life")),
        "Color over Life",
        false,
        theme,
        |ui| {
            let raw = eval_prop_str(
                lookup_session_prop_ref(session, obj_id, "particles.color"),
                local_frame,
                "",
            );
            let mut stops: Vec<(f64, [f32; 4])> = serde_json::from_str::<Value>(&raw)
                .ok()
                .and_then(|v| v.as_array().cloned())
                .map(|arr| {
                    arr.into_iter()
                        .filter_map(|item| {
                            let t = item
                                .get("t")
                                .or_else(|| item.get("pos"))
                                .and_then(Value::as_f64)?;
                            let c = item.get("color").and_then(Value::as_array)?;
                            Some((
                                t,
                                [
                                    c.first()?.as_f64()? as f32,
                                    c.get(1)?.as_f64()? as f32,
                                    c.get(2)?.as_f64()? as f32,
                                    c.get(3).and_then(Value::as_f64).unwrap_or(1.0) as f32,
                                ],
                            ))
                        })
                        .collect()
                })
                .filter(|v: &Vec<_>| v.len() >= 2)
                .unwrap_or_else(|| vec![(0.0, [1.0, 0.9, 0.6, 1.0]), (1.0, [1.0, 0.3, 0.1, 0.0])]);

            let mut dirty = false;
            let mut remove_idx = None;
            let mut tx_tracker = WidgetTxTracker::default();
            let stops_len = stops.len();
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(6, 4))
                .show(ui, |ui| {
                    for (si, (t, rgba)) in stops.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            let col_resp = ui.color_edit_button_rgba_unmultiplied(rgba);
                            if tx_tracker.observe(&col_resp) {
                                dirty = true;
                            }
                            let t_resp =
                                ui.add(egui::DragValue::new(t).range(0.0..=1.0).speed(0.01));
                            if tx_tracker.observe(&t_resp) {
                                dirty = true;
                            }
                            if stops_len > 2 && ui.small_button("×").clicked() {
                                remove_idx = Some(si);
                            }
                        });
                    }
                    if let Some(ri) = remove_idx {
                        stops.remove(ri);
                        dirty = true;
                    }
                    if ui.small_button("+ Add Color Stop").clicked() {
                        stops.push((0.5, [1.0, 1.0, 1.0, 1.0]));
                        dirty = true;
                    }
                });
            if dirty {
                tx_tracker.changed = true;
            }
            apply_widget_tx_command(
                ui,
                session,
                state,
                obj_id,
                "particles.color",
                "edit particles.color",
                tx_tracker,
                |session| {
                    stops.sort_by(|a, b| a.0.total_cmp(&b.0));
                    let encoded = Value::Array(
                        stops
                            .into_iter()
                            .map(|(t, c)| {
                                json!({
                                    "t": t,
                                    "color": [f64::from(c[0]), f64::from(c[1]), f64::from(c[2]), f64::from(c[3])]
                                })
                            })
                            .collect(),
                    )
                    .to_string();
                    let _ = session.execute(Command::SetPropertyValue {
                        target_id: obj_id.to_owned(),
                        property: "particles.color".to_owned(),
                        value: json!(encoded),
                    });
                },
            );
        },
    );
}

fn show_particle_size_over_life_editor(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut PanelsState,
    obj_id: &str,
    local_frame: f64,
    theme: &Cm3Theme,
) {
    cm3_collapsible_section(
        ui,
        Id::unique((obj_id, "particles_size_life")),
        "Size over Life",
        false,
        theme,
        |ui| {
            let raw = eval_prop_str(
                lookup_session_prop_ref(session, obj_id, "particles.size"),
                local_frame,
                "",
            );
            let mut stops: Vec<(f64, f64)> = serde_json::from_str::<Value>(&raw)
                .ok()
                .and_then(|v| v.as_array().cloned())
                .map(|arr| {
                    arr.into_iter()
                        .filter_map(|item| {
                            let t = item
                                .get("t")
                                .or_else(|| item.get("pos"))
                                .and_then(Value::as_f64)?;
                            let s = item
                                .get("size")
                                .or_else(|| item.get("value"))
                                .and_then(Value::as_f64)?;
                            Some((t, s))
                        })
                        .collect()
                })
                .filter(|v: &Vec<_>| v.len() >= 2)
                .unwrap_or_else(|| vec![(0.0, 1.0), (1.0, 0.0)]);

            let mut dirty = false;
            let mut remove_idx = None;
            let mut tx_tracker = WidgetTxTracker::default();
            let stops_len = stops.len();
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(6, 4))
                .show(ui, |ui| {
                    for (si, (t, sz)) in stops.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            let t_resp = ui.add(
                                egui::DragValue::new(t)
                                    .range(0.0..=1.0)
                                    .speed(0.01)
                                    .prefix("t "),
                            );
                            if tx_tracker.observe(&t_resp) {
                                dirty = true;
                            }
                            let sz_resp = ui.add(
                                egui::DragValue::new(sz)
                                    .range(0.0..=100.0)
                                    .speed(0.05)
                                    .prefix("sz "),
                            );
                            if tx_tracker.observe(&sz_resp) {
                                dirty = true;
                            }
                            if stops_len > 2 && ui.small_button("×").clicked() {
                                remove_idx = Some(si);
                            }
                        });
                    }
                    if let Some(ri) = remove_idx {
                        stops.remove(ri);
                        dirty = true;
                    }
                    if ui.small_button("+ Add Size Stop").clicked() {
                        stops.push((0.5, 1.0));
                        dirty = true;
                    }
                });
            if dirty {
                tx_tracker.changed = true;
            }
            apply_widget_tx_command(
                ui,
                session,
                state,
                obj_id,
                "particles.size",
                "edit particles.size",
                tx_tracker,
                |session| {
                    stops.sort_by(|a, b| a.0.total_cmp(&b.0));
                    let encoded = Value::Array(
                        stops
                            .into_iter()
                            .map(|(t, s)| json!({ "t": t, "size": s }))
                            .collect(),
                    )
                    .to_string();
                    let _ = session.execute(Command::SetPropertyValue {
                        target_id: obj_id.to_owned(),
                        property: "particles.size".to_owned(),
                        value: json!(encoded),
                    });
                },
            );
        },
    );
}
