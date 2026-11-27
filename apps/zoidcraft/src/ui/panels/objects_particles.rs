use super::property_rows::{
    eval_prop_str, eval_prop_vec3, lookup_session_prop_ref, show_session_number_prop,
    show_session_option_prop_ex, show_session_vec3_prop,
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
            show_particle_color_over_life_editor(
                ui,
                session,
                state,
                obj_id,
                local_frame,
                viewport_rect,
                theme,
                actions,
            );
            // Size-over-Life stop editor (`particles.size` JSON)
            show_particle_size_over_life_editor(ui, session, state, obj_id, local_frame, theme);
        }
    }
}

/// Parses CSS hex (`#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`) or `rgb(...)` / `rgba(...)` color
/// strings into normalized `[r, g, b, a]` components in `0.0..=1.0`.
pub(super) fn parse_particle_color_css(css: &str) -> Option<[f32; 4]> {
    let trimmed = css.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(c) = Rgba::from_hex(trimmed) {
        return Some(c.0);
    }
    if let Some(hex) = trimmed.strip_prefix('#') {
        if hex.len() == 3 || hex.len() == 4 {
            let mut bytes = [0u8; 4];
            bytes[3] = 255;
            for (i, ch) in hex.chars().enumerate() {
                let d = ch.to_digit(16)? as u8;
                bytes[i] = d * 17;
            }
            return Some([
                f32::from(bytes[0]) / 255.0,
                f32::from(bytes[1]) / 255.0,
                f32::from(bytes[2]) / 255.0,
                f32::from(bytes[3]) / 255.0,
            ]);
        }
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    let inner = lower
        .strip_prefix("rgba(")
        .or_else(|| lower.strip_prefix("rgb("))?
        .strip_suffix(')')?;
    let parts: Vec<f64> = inner
        .split(',')
        .filter_map(|p| p.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .collect();
    if parts.len() < 3 {
        return None;
    }
    let rgb_unit_float = parts[..3].iter().any(|&v| v > 0.0 && v < 1.0);
    let rgb_scale = if rgb_unit_float { 1.0 } else { 1.0 / 255.0 };
    let r = (parts[0] * rgb_scale).clamp(0.0, 1.0) as f32;
    let g = (parts[1] * rgb_scale).clamp(0.0, 1.0) as f32;
    let b = (parts[2] * rgb_scale).clamp(0.0, 1.0) as f32;
    let a_raw = parts.get(3).copied().unwrap_or(1.0);
    let a = if a_raw > 1.0 {
        (a_raw / 255.0).clamp(0.0, 1.0) as f32
    } else {
        a_raw.clamp(0.0, 1.0) as f32
    };
    Some([r, g, b, a])
}

fn parse_particle_color_value(val: &Value) -> Option<[f32; 4]> {
    if let Some(s) = val.as_str() {
        return parse_particle_color_css(s);
    }
    if let Some(c) = val.as_array()
        && c.len() >= 3
    {
        let r_raw = c[0].as_f64().filter(|v| v.is_finite())?;
        let g_raw = c[1].as_f64().filter(|v| v.is_finite())?;
        let b_raw = c[2].as_f64().filter(|v| v.is_finite())?;
        let a_raw = c
            .get(3)
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
            .unwrap_or(1.0);
        let rgb_scale = if r_raw > 1.0 || g_raw > 1.0 || b_raw > 1.0 {
            1.0 / 255.0
        } else {
            1.0
        };
        let a = if a_raw > 1.0 {
            (a_raw / 255.0).clamp(0.0, 1.0) as f32
        } else {
            a_raw.clamp(0.0, 1.0) as f32
        };
        return Some([
            (r_raw * rgb_scale).clamp(0.0, 1.0) as f32,
            (g_raw * rgb_scale).clamp(0.0, 1.0) as f32,
            (b_raw * rgb_scale).clamp(0.0, 1.0) as f32,
            a,
        ]);
    }
    None
}

/// Parses `particles.color` JSON supporting both inspector schema
/// (`{"t", "color": [r, g, b, a]}`) and `sync_from_project` `GradientStop` schema
/// (`{"position", "color": "#rrggbb" | "rgba(...)"}`).
pub(super) fn parse_particle_color_stops(raw: &str) -> Vec<(f64, [f32; 4])> {
    let mut stops: Vec<(f64, [f32; 4])> = serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|v| v.as_array().cloned())
        .map(|arr| {
            arr.into_iter()
                .filter_map(|item| {
                    let t = item
                        .get("t")
                        .or_else(|| item.get("position"))
                        .or_else(|| item.get("pos"))
                        .and_then(Value::as_f64)
                        .filter(|v| v.is_finite())?
                        .clamp(0.0, 1.0);
                    let rgba = parse_particle_color_value(item.get("color")?)?;
                    Some((t, rgba))
                })
                .collect()
        })
        .unwrap_or_default();
    stops.sort_by(|a, b| a.0.total_cmp(&b.0));
    match stops.len() {
        0 => vec![(0.0, [1.0, 0.9, 0.6, 1.0]), (1.0, [1.0, 0.3, 0.1, 0.0])],
        1 => {
            let c = stops[0].1;
            vec![(0.0, c), (1.0, c)]
        }
        _ => stops,
    }
}

/// Parses `particles.size` JSON supporting both inspector schema (`{"t", "size"}`) and
/// `sync_from_project` / `GradientStop` schema (`{"position", "value" | "color": "1.5"}`).
pub(super) fn parse_particle_size_stops(raw: &str) -> Vec<(f64, f64)> {
    let parse_num = |val: &Value| -> Option<f64> {
        val.as_f64()
            .or_else(|| val.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
            .filter(|v| v.is_finite())
            .map(|v| v.max(0.0))
    };
    let mut stops: Vec<(f64, f64)> = serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|v| v.as_array().cloned())
        .map(|arr| {
            arr.into_iter()
                .filter_map(|item| {
                    let t = item
                        .get("t")
                        .or_else(|| item.get("position"))
                        .or_else(|| item.get("pos"))
                        .and_then(Value::as_f64)
                        .filter(|v| v.is_finite())?
                        .clamp(0.0, 1.0);
                    let s = item
                        .get("size")
                        .or_else(|| item.get("value"))
                        .or_else(|| item.get("color"))
                        .and_then(parse_num)?;
                    Some((t, s))
                })
                .collect()
        })
        .unwrap_or_default();
    stops.sort_by(|a, b| a.0.total_cmp(&b.0));
    match stops.len() {
        0 => vec![(0.0, 1.0), (1.0, 0.0)],
        1 => {
            let sz = stops[0].1;
            vec![(0.0, sz), (1.0, sz)]
        }
        _ => stops,
    }
}

fn encode_particle_color_stops(stops: &[(f64, [f32; 4])]) -> String {
    let mut sorted = stops.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    Value::Array(
        sorted
            .into_iter()
            .map(|(t, c)| {
                json!({
                    "t": t,
                    "color": [f64::from(c[0]), f64::from(c[1]), f64::from(c[2]), f64::from(c[3])]
                })
            })
            .collect(),
    )
    .to_string()
}

#[allow(clippy::too_many_arguments)]
fn show_particle_color_over_life_editor(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut PanelsState,
    obj_id: &str,
    local_frame: f64,
    viewport_rect: Rect,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
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
            let mut stops = parse_particle_color_stops(&raw);

            // Apply any completed viewport eyedropper pick targeting a specific color stop.
            let mut picked_updates: Vec<(usize, String, [f64; 3])> = Vec::new();
            for si in 0..stops.len() {
                let pick_key = format!("particles.color.__pick_{si}");
                if let Some(prop) = lookup_session_prop_ref(session, obj_id, &pick_key) {
                    let rgb = eval_prop_vec3(Some(prop), local_frame, [1.0, 1.0, 1.0]);
                    picked_updates.push((si, pick_key, rgb));
                }
            }
            if !picked_updates.is_empty() {
                for (si, _, rgb) in &picked_updates {
                    if let Some((_, rgba)) = stops.get_mut(*si) {
                        rgba[0] = rgb[0].clamp(0.0, 1.0) as f32;
                        rgba[1] = rgb[1].clamp(0.0, 1.0) as f32;
                        rgba[2] = rgb[2].clamp(0.0, 1.0) as f32;
                    }
                }
                let encoded = encode_particle_color_stops(&stops);
                let _ = session.execute(Command::SetPropertyValue {
                    target_id: obj_id.to_owned(),
                    property: "particles.color".to_owned(),
                    value: json!(encoded),
                });
                for (_, pick_key, _) in picked_updates {
                    let _ = session.execute(Command::RemoveProperty {
                        target_id: obj_id.to_owned(),
                        property: pick_key,
                    });
                }
            }

            let mut dirty = false;
            let mut remove_idx = None;
            let mut tx_tracker = WidgetTxTracker::default();
            let stops_len = stops.len();
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(6, 4))
                .show(ui, |ui| {
                    for (si, (t, rgba)) in stops.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            let mut stop_rgba = Rgba::new(rgba[0], rgba[1], rgba[2], rgba[3]);
                            let cev = cm3_hsva_color_picker(
                                ui,
                                Id::unique((obj_id, "particles_color_stop", si)),
                                &mut stop_rgba,
                                true,
                                false,
                                viewport_rect,
                                theme,
                            );
                            if cev.started {
                                tx_tracker.drag_started = true;
                            }
                            if cev.committed || cev.cancelled {
                                tx_tracker.drag_stopped = true;
                            }
                            if cev.changed || cev.committed {
                                *rgba = stop_rgba.0;
                                dirty = true;
                            }
                            if cev.eyedropper_clicked {
                                actions.push(PanelAction::EyedropperActivate {
                                    target_id: obj_id.to_owned(),
                                    property: format!("particles.color.__pick_{si}"),
                                });
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
                    let encoded = encode_particle_color_stops(&stops);
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
            let mut stops = parse_particle_size_stops(&raw);

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

#[cfg(test)]
mod particle_stop_parser_tests {
    use super::{parse_particle_color_css, parse_particle_color_stops, parse_particle_size_stops};

    #[test]
    fn color_over_life_parses_both_inspector_and_project_schemas() {
        // Inspector schema: {"t", "color": [r, g, b, a]}
        let inspector_json =
            r#"[{"t":0.0,"color":[0.2,0.4,0.8,1.0]},{"t":1.0,"color":[1.0,0.0,0.5,0.25]}]"#;
        let stops = parse_particle_color_stops(inspector_json);
        assert_eq!(stops.len(), 2);
        assert!((stops[0].0 - 0.0).abs() < 1e-6);
        assert!((stops[0].1[0] - 0.2).abs() < 1e-5);
        assert!((stops[0].1[2] - 0.8).abs() < 1e-5);
        assert!((stops[1].0 - 1.0).abs() < 1e-6);
        assert!((stops[1].1[3] - 0.25).abs() < 1e-5);

        // sync_from_project GradientStop schema: {"position", "color": "#rrggbb" | "rgba(...)"}
        let synced_json = r##"[{"position":0.0,"color":"#3366cc"},{"position":0.5,"color":"rgba(255,128,0,0.5)"},{"position":1.0,"color":"rgb(0,255,0)"}]"##;
        let synced = parse_particle_color_stops(synced_json);
        assert_eq!(synced.len(), 3);
        assert!((synced[0].0 - 0.0).abs() < 1e-6);
        assert!((synced[0].1[0] - (51.0_f32 / 255.0)).abs() < 1e-4);
        assert!((synced[0].1[2] - (204.0_f32 / 255.0)).abs() < 1e-4);
        assert!((synced[1].0 - 0.5).abs() < 1e-6);
        assert!((synced[1].1[0] - 1.0).abs() < 1e-4);
        assert!((synced[1].1[1] - (128.0_f32 / 255.0)).abs() < 1e-4);
        assert!((synced[1].1[3] - 0.5).abs() < 1e-4);
        assert!((synced[2].0 - 1.0).abs() < 1e-6);
        assert!((synced[2].1[1] - 1.0).abs() < 1e-4);

        // Short hex & unit float rgba
        let short_hex = parse_particle_color_css("#f08").expect("short hex");
        assert!((short_hex[0] - 1.0).abs() < 1e-4);
        assert!((short_hex[1] - 0.0).abs() < 1e-4);
        assert!((short_hex[2] - (136.0_f32 / 255.0)).abs() < 1e-4);
    }

    #[test]
    fn size_over_life_parses_both_inspector_and_project_schemas() {
        // Inspector schema: {"t", "size"}
        let inspector_json = r#"[{"t":0.0,"size":2.5},{"t":0.75,"size":0.5},{"t":1.0,"size":0.0}]"#;
        let stops = parse_particle_size_stops(inspector_json);
        assert_eq!(stops.len(), 3);
        assert!((stops[0].0 - 0.0).abs() < 1e-6);
        assert!((stops[0].1 - 2.5).abs() < 1e-6);
        assert!((stops[1].0 - 0.75).abs() < 1e-6);
        assert!((stops[1].1 - 0.5).abs() < 1e-6);

        // Project/sync schema: {"position", "value"} and GradientStop {"position", "color": "3.25"}
        let synced_json = r#"[{"position":0.0,"value":4.0},{"position":0.5,"color":"1.75"},{"position":1.0,"value":"0.25"}]"#;
        let synced = parse_particle_size_stops(synced_json);
        assert_eq!(synced.len(), 3);
        assert!((synced[0].1 - 4.0).abs() < 1e-6);
        assert!((synced[1].0 - 0.5).abs() < 1e-6);
        assert!((synced[1].1 - 1.75).abs() < 1e-6);
        assert!((synced[2].1 - 0.25).abs() < 1e-6);
    }
}
