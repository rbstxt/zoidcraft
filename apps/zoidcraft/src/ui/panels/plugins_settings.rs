use super::shared_widgets::{
    paint_crisp_border_2px, paint_crisp_panel_border, paint_small_chevron, panel_top_header,
    section_banner, warning_banner,
};
use super::*;

// ---------------------------------------------------------------------------
// Tab 7: Restore — Autosave & Project Snapshot Restore
// ---------------------------------------------------------------------------

/// Converts internal dot/snake command IDs (e.g. `"timeline.trim_clip_left"`, `"scene.add_object_3d"`)
/// into clean human-readable Restore History labels (`"Trim clip left"`, `"Add 3D object"`).
pub(super) fn humanize_history_label(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "Edit".to_owned();
    }
    match trimmed {
        "project.new" => return "New project".to_owned(),
        "project.set_name" => return "Rename project".to_owned(),
        "project.set_resolution" => return "Set resolution".to_owned(),
        "project.set_fps" => return "Set frame rate".to_owned(),
        "project.set_sequence_length" => return "Set sequence length".to_owned(),
        "scene.add_object_3d" => return "Add 3D object".to_owned(),
        "scene.delete_object_3d" => return "Delete 3D object".to_owned(),
        "scene.duplicate_object_3d" => return "Duplicate 3D object".to_owned(),
        "scene.reorder_object_3d" => return "Reorder 3D object".to_owned(),
        "scene.split_text_3d_into_letters" => return "Split 3D text into letters".to_owned(),
        "clipboard.paste_pz_json" => return "Paste .pz JSON".to_owned(),
        "property.set_value" => return "Set property".to_owned(),
        "property.toggle_animated" => return "Toggle keyframe animation".to_owned(),
        "property.set_keyframe" => return "Set keyframe".to_owned(),
        "property.delete_keyframe" => return "Delete keyframe".to_owned(),
        "property.set_expression" => return "Set expression".to_owned(),
        "property.remove" => return "Remove property".to_owned(),
        "property.reset_list" => return "Reset properties".to_owned(),
        _ => {}
    }

    // If raw looks like `namespace.snake_case`, strip the namespace prefix and humanize.
    let body = if let Some((ns, rest)) = trimmed.split_once('.')
        && !ns.contains(' ')
        && !rest.is_empty()
        && ns.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        rest
    } else {
        trimmed
    };

    let words: Vec<String> = body
        .split(['_', ' '])
        .filter(|w| !w.is_empty())
        .enumerate()
        .map(|(i, w)| {
            let lower = w.to_ascii_lowercase();
            match lower.as_str() {
                "3d" => "3D".to_owned(),
                "2d" => "2D".to_owned(),
                "fps" => "FPS".to_owned(),
                "pz" => ".pz".to_owned(),
                "json" => "JSON".to_owned(),
                "glsl" => "GLSL".to_owned(),
                _ if i == 0 => {
                    let mut chars = w.chars();
                    match chars.next() {
                        Some(first) => {
                            let mut s = first.to_uppercase().to_string();
                            s.push_str(chars.as_str());
                            s
                        }
                        None => String::new(),
                    }
                }
                _ => w.to_owned(),
            }
        })
        .collect();
    if words.is_empty() {
        trimmed.to_owned()
    } else {
        words.join(" ")
    }
}

pub(super) fn show_restore_tab(
    ui: &mut Ui,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    ScrollArea::vertical()
        .id_salt("restore_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;
            panel_top_header(ui, "Restore", theme);

            let project_name = session.document().project_name.clone();
            let [rw, rh] = session.document().resolution;
            let fps = session.document().fps;
            let undo_len = session.history.undo_len();
            let redo_len = session.history.redo_len();

            Frame::NONE
                .inner_margin(egui::Margin::symmetric(10, 10))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 8.0;

                    ui.label(
                        egui::RichText::new("Project snapshots & autosave")
                            .font(theme.font(FONT_BODY))
                            .color(TEXT_STRONG),
                    );
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(
                                "Undo history snapshots are tracked automatically during editing. \
                                 Step back to a previous state or reset the workspace to the default starter project.",
                            )
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_OPTION_DESC),
                        )
                        .wrap(),
                    );

                    Frame::NONE
                        .fill(BG_TITLE)
                        .stroke(Stroke::new(1.0, BORDER_ROW_SOFT))
                        .inner_margin(egui::Margin::symmetric(8, 8))
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            ui.spacing_mut().item_spacing.y = 4.0;
                            ui.label(
                                egui::RichText::new(format!("Active project: {project_name}"))
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_PRIMARY),
                            );
                            ui.label(
                                egui::RichText::new(format!("Format: {rw}×{rh} @ {fps:.0} fps"))
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_MUTED),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "History states: {undo_len} undo · {redo_len} redo"
                                ))
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_MUTED),
                            );
                        });

                    let undo_tip = session
                        .history
                        .peek_undo_label()
                        .map_or_else(
                            || "Nothing to undo".to_owned(),
                            |l| format!("Undo: {}", humanize_history_label(l)),
                        );
                    let redo_tip = session
                        .history
                        .peek_redo_label()
                        .map_or_else(
                            || "Nothing to redo".to_owned(),
                            |l| format!("Redo: {}", humanize_history_label(l)),
                        );
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        let half_w = ((ui.available_width() - 6.0) * 0.5).max(60.0);
                        ui.add_enabled_ui(undo_len > 0, |ui| {
                            if ui
                                .add_sized(
                                    Vec2::new(half_w, 24.0),
                                    egui::Button::new(
                                        egui::RichText::new(format!("Undo ({undo_len})"))
                                            .font(theme.font(FONT_SMALL))
                                            .color(if undo_len > 0 {
                                                TEXT_PRIMARY
                                            } else {
                                                TEXT_DISABLED
                                            }),
                                    )
                                    .fill(BG_BUTTON)
                                    .stroke(Stroke::new(1.0, BORDER_CONTROL)),
                                )
                                .on_hover_text(undo_tip)
                                .clicked()
                            {
                                let _ = session.execute(Command::Undo);
                                actions.push(PanelAction::StatusMessage(
                                    "Restored previous snapshot".to_owned(),
                                ));
                            }
                        });
                        ui.add_enabled_ui(redo_len > 0, |ui| {
                            if ui
                                .add_sized(
                                    Vec2::new(half_w, 24.0),
                                    egui::Button::new(
                                        egui::RichText::new(format!("Redo ({redo_len})"))
                                            .font(theme.font(FONT_SMALL))
                                            .color(if redo_len > 0 {
                                                TEXT_PRIMARY
                                            } else {
                                                TEXT_DISABLED
                                            }),
                                    )
                                    .fill(BG_BUTTON)
                                    .stroke(Stroke::new(1.0, BORDER_CONTROL)),
                                )
                                .on_hover_text(redo_tip)
                                .clicked()
                            {
                                let _ = session.execute(Command::Redo);
                                actions.push(PanelAction::StatusMessage(
                                    "Reapplied next snapshot".to_owned(),
                                ));
                            }
                        });
                    });

                    // Compact scrollable recent history activity list
                    if !state.history_log.is_empty() {
                        ui.label(
                            egui::RichText::new("Recent history")
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_STRONG),
                        );
                        Frame::NONE
                            .fill(BG_SUNKEN)
                            .stroke(Stroke::new(1.0, BORDER_ROW_SOFT))
                            .inner_margin(egui::Margin::symmetric(6, 4))
                            .show(ui, |ui| {
                                ui.set_min_width(ui.available_width());
                                ScrollArea::vertical()
                                    .id_salt("restore_recent_history_scroll")
                                    .max_height(110.0)
                                    .auto_shrink([false, true])
                                    .show(ui, |ui| {
                                        ui.spacing_mut().item_spacing.y = 2.0;
                                        let active_rev = session.history.revision();
                                        for (rev, label) in &state.history_log {
                                            let is_cur = *rev == active_rev;
                                            let prefix = if is_cur { "• " } else { "  " };
                                            let pretty = humanize_history_label(label);
                                            ui.add(
                                                egui::Label::new(
                                                    egui::RichText::new(format!(
                                                        "{prefix}#{rev} {pretty}"
                                                    ))
                                                    .font(theme.font(FONT_COMPACT))
                                                    .color(if is_cur {
                                                        theme.value()
                                                    } else {
                                                        TEXT_MUTED
                                                    }),
                                                )
                                                .truncate(),
                                            );
                                        }
                                    });
                            });
                    }

                    if cm3_prop_button(ui, "Export .pz...", ButtonVariant::Normal, true, theme)
                        .clicked()
                    {
                        state.pending_project_actions.push(ProjectAction::ExportPz);
                    }

                    if cm3_prop_button(
                        ui,
                        "Reset to default project",
                        ButtonVariant::Danger,
                        true,
                        theme,
                    )
                    .clicked()
                    {
                        actions.push(PanelAction::NewProject);
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    {
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("Disk autosave snapshots")
                            .font(theme.font(FONT_BODY))
                            .color(TEXT_STRONG),
                    );
                    if cm3_prop_button(
                        ui,
                        "Save Recovery Snapshot Now",
                        ButtonVariant::Primary,
                        true,
                        theme,
                    )
                    .clicked()
                    {
                        actions.push(PanelAction::SaveRecoverySnapshotNow);
                    }

                    if state.recovery_snapshots.is_empty() {
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(
                                    "No disk recovery snapshots yet. Modified projects autosave every 60 seconds.",
                                )
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_OPTION_DESC),
                            )
                            .wrap(),
                        );
                    } else {
                        for snap in &state.recovery_snapshots {
                            let ts_str = format_unix_timestamp_utc(snap.timestamp_secs);
                            let size_str = format_snapshot_bytes(snap.size_bytes);
                            Frame::NONE
                                .fill(BG_TITLE)
                                .stroke(Stroke::new(1.0, BORDER_ROW_SOFT))
                                .inner_margin(egui::Margin::symmetric(8, 6))
                                .show(ui, |ui| {
                                    ui.set_min_width(ui.available_width());
                                    ui.spacing_mut().item_spacing.y = 4.0;
                                    ui.label(
                                        egui::RichText::new(&snap.filename)
                                            .font(theme.font(FONT_SMALL))
                                            .color(TEXT_PRIMARY),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!("{ts_str} · {size_str}"))
                                            .font(theme.font(FONT_COMPACT))
                                            .color(TEXT_MUTED),
                                    );
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 6.0;
                                        if cm3_prop_button(
                                            ui,
                                            "Restore",
                                            ButtonVariant::Primary,
                                            false,
                                            theme,
                                        )
                                        .clicked()
                                        {
                                            actions.push(PanelAction::RestoreRecoverySnapshot(
                                                snap.path.clone(),
                                            ));
                                        }
                                        if cm3_prop_button(
                                            ui,
                                            "Delete",
                                            ButtonVariant::Danger,
                                            false,
                                            theme,
                                        )
                                        .clicked()
                                        {
                                            actions.push(PanelAction::DeleteRecoverySnapshot(
                                                snap.path.clone(),
                                            ));
                                        }
                                    });
                                });
                        }
                    }
                    }
                });
        });
}

#[cfg(not(target_arch = "wasm32"))]
fn format_snapshot_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn format_unix_timestamp_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = rem / 3600;
    let minute = (rem % 3600) / 60;
    let second = rem % 60;

    // Howard Hinnant's civil-from-days algorithm (UTC, proleptic Gregorian).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    format!("{year:04}-{m:02}-{d:02} {hour:02}:{minute:02}:{second:02} UTC")
}

// ---------------------------------------------------------------------------
// Tab 7: Plugins (Zoidium Extension Manager, Spec 04 §5.1)
// ---------------------------------------------------------------------------

fn zoidium_plugin_summary<'a>(id: &str, fallback: &'a str) -> &'a str {
    match id {
        // Utilities (exact Zoidium 08_plugins.png summaries)
        "easing-plus" => "Extended Bezier Easing",
        "expressions-plus" => "Modern expressions and property links",
        "shader-plus" => "Custom shader property editors",
        "player-plus" => "Playback Quality Controls",
        "asset-manager" => "Project panel for every asset",
        "precomp-plus" => "Pre-compose and nested compositions",
        "gif-support" => "Animated GIF image assets",
        "graph-plus" => "Rebuilt keyframe graph editor",
        "motion-path" => "Editable motion paths in the preview",
        "mp4-export" => "H.264 MP4 video export",
        // Scene Extensions
        "light-plus" => "Spot, point, directional & hemisphere lights",
        "geometry-plus" => "Rounded boxes & procedural 3D shapes",
        "repeater" => "Repeaters, twist, warp & Voronoi fracture",
        "effector-plus" => "Legacy deformer compatibility loader",
        "material-plus" => "Node, PBR+, Matcap & Image+ materials",
        "text-plus" => "2D/3D text animators & advanced bevels",
        "expresstext-plus" => "Per-letter 3D text effector stack",
        "particles-plus" => "54 procedural particle sprites",
        "trapcode-suite" => "3D particles, lattice & studio lights",
        "optical-flares" => "3D projected lens flares with occlusion",
        "camera-plus" => "Physical camera, DoF & 3D tracks",
        "shape-plus" => "Animated vector layers & pen masks",
        "constraints" => "Track To & Follow 3D constraints",
        // Adjustment Extensions
        "native-fx" => "Temporal FX, curves, drop shadow & motion blur",
        "openzoid-legacy" => "JPEG Damage, VHS, Datamosh & ASCII FX",
        "magic-looks" => "Color grading, film looks & 3D LUTs",
        "node-effect" => "Visual GPU shader node graph",
        "smart-motion-blur-lite" => "Fast optical-flow motion blur",
        "sglow-hq-v2-preview" => "Multi-scale glow with RGB spread",
        // Experimental
        "layer-input" => "Cross-track texture sampling",
        _ => fallback,
    }
}

fn is_cm3_compatible_plugin(plugin_id: &str) -> bool {
    matches!(
        plugin_id,
        "easing-plus"
            | "shader-plus"
            | "player-plus"
            | "asset-manager"
            | "graph-plus"
            | "motion-path"
            | "mp4-export"
            | "node-effect"
            | "effector-plus"
            | "particles-plus"
    )
}

fn paint_outlined_plugin_badge_at(
    painter: &Painter,
    pos: Pos2,
    text: &str,
    bg: Color32,
    border: Color32,
    fg: Color32,
    theme: &Cm3Theme,
) -> f32 {
    let galley = painter.layout_no_wrap(text.to_owned(), theme.font(11.5), fg);
    let size = Vec2::new((galley.size().x + 13.0).round(), 16.0);
    let rect = Rect::from_min_size(Pos2::new(pos.x.round(), pos.y.round()), size);
    painter.rect_filled(rect, CornerRadius::ZERO, bg);
    paint_crisp_panel_border(painter, rect, border);
    painter.galley(
        Pos2::new(
            (rect.center().x - galley.size().x * 0.5).round(),
            rect.center().y - galley.size().y * 0.5 - 0.5,
        ),
        galley,
        fg,
    );
    size.x
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PillToggleState {
    Off,
    Partial,
    On,
}

fn paint_pill_toggle(
    ui: &mut Ui,
    toggle_state: PillToggleState,
    locked: bool,
    theme: &Cm3Theme,
) -> egui::Response {
    // 36x20 rounded pill switch matching Zoidium 08_plugins.png
    // ON: #384668 track + #506084 rim + #e4e4e4 right thumb (14x14)
    // PARTIAL: #333333 track + #6d6d6d rim + 18x4 #9a9a9a dash
    // OFF: #202020 track + #4d4d4d rim + #9a9a9a left thumb (14x14)
    let (raw_rect, resp) = ui.allocate_exact_size(Vec2::new(36.0, 20.0), Sense::click());
    let rect = Rect::from_min_size(
        Pos2::new(raw_rect.left().round(), raw_rect.top().round()),
        Vec2::new(36.0, 20.0),
    );
    let resp = resp.on_hover_cursor(if locked {
        CursorIcon::NotAllowed
    } else {
        CursorIcon::PointingHand
    });
    let radius = CornerRadius::same(10);
    let (track_fill, rim_color) = match toggle_state {
        PillToggleState::On => (
            if locked {
                color32_with_alpha(theme.accent(), 150)
            } else {
                theme.accent()
            },
            if resp.hovered() && !locked {
                theme.value()
            } else {
                Color32::from_rgb(0x50, 0x60, 0x84)
            },
        ),
        PillToggleState::Partial => (
            BG_HOVER,
            if resp.hovered() && !locked {
                theme.value()
            } else {
                TEXT_FAINT
            },
        ),
        PillToggleState::Off => (
            BG_INPUT,
            if resp.hovered() && !locked {
                theme.value()
            } else {
                Color32::from_rgb(0x4d, 0x4d, 0x4d)
            },
        ),
    };
    ui.painter().rect_filled(rect, radius, track_fill);
    ui.painter().rect_stroke(
        rect,
        radius,
        Stroke::new(1.0, rim_color),
        StrokeKind::Inside,
    );
    match toggle_state {
        PillToggleState::On => {
            ui.painter().circle_filled(
                Pos2::new(rect.right() - 10.0, rect.center().y),
                7.0,
                TEXT_BRIGHT,
            );
        }
        PillToggleState::Off => {
            ui.painter().circle_filled(
                Pos2::new(rect.left() + 10.0, rect.center().y),
                7.0,
                TEXT_MUTED,
            );
        }
        PillToggleState::Partial => {
            let dash = Rect::from_min_size(
                Pos2::new(rect.left() + 9.0, rect.top() + 8.0),
                Vec2::new(18.0, 4.0),
            );
            ui.painter()
                .rect_filled(dash, CornerRadius::same(2), TEXT_MUTED);
        }
    }
    resp
}

/// Match quality rank for the Plugins search box (lower is higher priority):
/// 0 = exact title/id match, 1 = title prefix match, 2 = title substring match,
/// 3 = plugin id substring match, 4 = summary/tagline description match.
fn plugin_search_match_rank(
    desc: &PluginDescriptor,
    summary: &str,
    query_lower: &str,
) -> Option<u8> {
    if query_lower.is_empty() {
        return Some(0);
    }
    let name_low = desc.name.to_ascii_lowercase();
    let id_low = desc.id.to_ascii_lowercase();
    if name_low == query_lower || id_low == query_lower {
        return Some(0);
    }
    if name_low.starts_with(query_lower) {
        return Some(1);
    }
    if name_low.contains(query_lower) {
        return Some(2);
    }
    if id_low.contains(query_lower) {
        return Some(3);
    }
    if summary.to_ascii_lowercase().contains(query_lower)
        || desc.tagline.to_ascii_lowercase().contains(query_lower)
    {
        return Some(4);
    }
    None
}

pub(super) struct RankedPluginCategory {
    pub(super) best_rank: u8,
    pub(super) cat_ord: usize,
    pub(super) category: PluginCategory,
    pub(super) all_in_cat: Vec<PluginDescriptor>,
    pub(super) descriptors: Vec<PluginDescriptor>,
}

/// Filters and orders plugin categories + descriptors for the Plugins search box so
/// exact/title matches (e.g. `"Native FX"` for `"native"`) always rank above
/// internal description/tagline matches (`"MP4 Export"`, `"Node Effect"`).
pub(super) fn rank_filtered_plugin_categories(
    session: &Session,
    query_lower: &str,
) -> Vec<(PluginCategory, Vec<PluginDescriptor>, Vec<PluginDescriptor>)> {
    let mut ranked: Vec<RankedPluginCategory> = Vec::new();
    for (cat_ord, category) in [
        PluginCategory::Utilities,
        PluginCategory::Scene,
        PluginCategory::Adjustment,
        PluginCategory::Experimental,
    ]
    .into_iter()
    .enumerate()
    {
        let all_in_cat: Vec<PluginDescriptor> = session
            .plugins
            .visible_in_category(category)
            .into_iter()
            .cloned()
            .collect();
        if all_in_cat.is_empty() {
            continue;
        }
        let mut matched: Vec<(u8, usize, PluginDescriptor)> = all_in_cat
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                let summary = zoidium_plugin_summary(&d.id, &d.tagline);
                plugin_search_match_rank(d, summary, query_lower).map(|rank| (rank, i, d.clone()))
            })
            .collect();
        if matched.is_empty() {
            continue;
        }
        if !query_lower.is_empty() {
            matched.sort_by_key(|(rank, idx, _)| (*rank, *idx));
        }
        let best_rank = matched.first().map_or(u8::MAX, |(r, _, _)| *r);
        let descriptors: Vec<PluginDescriptor> = matched.into_iter().map(|(_, _, d)| d).collect();
        ranked.push(RankedPluginCategory {
            best_rank,
            cat_ord,
            category,
            all_in_cat,
            descriptors,
        });
    }
    if !query_lower.is_empty() {
        ranked.sort_by_key(|entry| (entry.best_rank, entry.cat_ord));
    }
    ranked
        .into_iter()
        .map(|entry| (entry.category, entry.all_in_cat, entry.descriptors))
        .collect()
}

pub(super) fn show_plugins_tab(
    ui: &mut Ui,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
) {
    let active_deps = session.active_plugin_dependencies();

    ScrollArea::vertical()
        .id_salt("plugins_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;

            // Edge-to-edge 43px #222222 header bar ("Plugins")
            panel_top_header(ui, "Plugins", theme);

            // 34px single-frame "type to filter" search box (w=301, right: 16)
            Frame::NONE
                .inner_margin(egui::Margin {
                    left: 4,
                    right: 16,
                    top: 5,
                    bottom: 0,
                })
                .show(ui, |ui| {
                    let filter_id = Id::unique("plugins_filter_box");
                    let focused = ui.memory(|m| m.has_focus(filter_id));
                    let box_w = ui.available_width();
                    ui.allocate_ui_with_layout(
                        Vec2::new(box_w, 34.0),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            let box_rect =
                                Rect::from_min_size(ui.cursor().min, Vec2::new(box_w, 34.0));
                            ui.painter()
                                .rect_filled(box_rect, CornerRadius::ZERO, BG_PANEL);
                            paint_crisp_border_2px(
                                ui.painter(),
                                box_rect,
                                if focused {
                                    theme.accent()
                                } else {
                                    BORDER_FILTER
                                },
                            );
                            Frame::NONE
                                .inner_margin(egui::Margin {
                                    left: 8,
                                    right: 8,
                                    top: 7,
                                    bottom: 5,
                                })
                                .show(ui, |ui| {
                                    ui.add(
                                        TextEdit::singleline(&mut state.plugin_filter)
                                            .id(filter_id)
                                            .frame(Frame::NONE)
                                            .font(theme.font(FONT_BODY))
                                            .text_color(TEXT_PRIMARY)
                                            .desired_width((box_w - 16.0).max(40.0)),
                                    );
                                });
                            if state.plugin_filter.is_empty() {
                                ui.painter().text(
                                    Pos2::new(box_rect.left() + 8.0, box_rect.center().y + 1.0),
                                    Align2::LEFT_CENTER,
                                    "type to filter",
                                    theme.font(FONT_BODY),
                                    TEXT_FAINT,
                                );
                            }
                        },
                    );
                });

            // Dismissible notice banner with "x" button
            if let Some(notice) = state.plugin_notice.clone() {
                let mut dismiss = false;
                let resp = Frame::NONE
                    .fill(WARNING_SURFACE)
                    .inner_margin(egui::Margin {
                        left: 10,
                        right: 6,
                        top: 6,
                        bottom: 6,
                    })
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.set_max_width((ui.available_width() - 22.0).max(60.0));
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(&notice)
                                            .font(theme.font(FONT_SMALL))
                                            .color(DANGER_TEXT),
                                    )
                                    .wrap(),
                                );
                            });
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Min),
                                |ui| {
                                    if cm3_icon_button(
                                        ui,
                                        Cm3Icon::Close,
                                        12.0,
                                        DANGER_TEXT,
                                        "Dismiss notice",
                                        theme,
                                    ) {
                                        dismiss = true;
                                    }
                                },
                            );
                        });
                    });
                let r = resp.response.rect;
                let bar = Rect::from_min_size(r.left_top(), Vec2::new(3.0, r.height()));
                ui.painter()
                    .rect_filled(bar, CornerRadius::ZERO, WARNING_BAR);
                if dismiss {
                    state.plugin_notice = None;
                }
            }

            let q = state.plugin_filter.trim().to_ascii_lowercase();

            for (category, all_in_cat, descriptors) in rank_filtered_plugin_categories(session, &q)
            {
                let enabled_count = all_in_cat
                    .iter()
                    .filter(|d| session.plugins.is_enabled(&d.id))
                    .count();

                let cat_id = Id::unique(("plugin_cat", category.display_name()));
                let default_open = !category.collapsed_by_default();
                let mut open = if !q.is_empty() {
                    true
                } else {
                    ui.data_mut(|d| d.get_temp::<bool>(cat_id))
                        .unwrap_or(default_open)
                };

                // Flat #2a2a2a 40px category header (1px #242424 bottom divider), #7e8fb9 chevron,
                // category name (ellipsis-truncated), muted "X of Y on" counter, right-aligned 36x20 pill
                let cat_w = ui.available_width();
                let (cat_rect, _) =
                    ui.allocate_exact_size(Vec2::new(cat_w, 40.0), Sense::hover());
                ui.painter()
                    .rect_filled(cat_rect, CornerRadius::ZERO, BG_PANEL);
                let cat_div_y = (cat_rect.bottom() - 1.0).round();
                ui.painter().rect_filled(
                    Rect::from_min_max(
                        Pos2::new(cat_rect.left(), cat_div_y),
                        Pos2::new(cat_rect.right(), cat_div_y + 1.0),
                    ),
                    CornerRadius::ZERO,
                    BORDER_ROW_SOFT,
                );

                // Toggle click zone on right (36x20 pill with compact right padding so
                // "Scene Extensions" displays cleanly without ellipsizing)
                let cat_toggle_rect = Rect::from_min_size(
                    Pos2::new(
                        (cat_rect.right() - 44.0).round(),
                        (cat_rect.center().y - 10.0).round(),
                    ),
                    Vec2::new(36.0, 20.0),
                );
                let cat_hdr_click_rect = Rect::from_min_max(
                    cat_rect.left_top(),
                    Pos2::new(
                        (cat_toggle_rect.left() - 6.0).max(cat_rect.left()),
                        cat_rect.bottom(),
                    ),
                );
                let cat_resp = ui
                    .interact(cat_hdr_click_rect, cat_id.with("hdr"), Sense::click())
                    .on_hover_cursor(CursorIcon::PointingHand);
                if cat_resp.clicked() {
                    open = !open;
                }
                ui.data_mut(|d| d.insert_temp(cat_id, open));

                // #7e8fb9 chevron at left + 10.0, center.y + 0.5
                paint_small_chevron(
                    ui.painter(),
                    Pos2::new(cat_rect.left() + 10.0, cat_rect.center().y + 0.5),
                    open,
                    theme.value(),
                );

                let count_text = if cat_rect.width() < 260.0 {
                    format!("{}/{}", enabled_count, all_in_cat.len())
                } else {
                    format!("{} of {} on", enabled_count, all_in_cat.len())
                };
                let count_galley = ui.painter().layout_no_wrap(
                    count_text,
                    theme.font(FONT_COMPACT),
                    TEXT_MUTED,
                );
                let count_right = (cat_toggle_rect.left() - 8.0).max(cat_rect.left() + 80.0);
                let count_left = (count_right - count_galley.size().x).max(cat_rect.left() + 80.0);

                let name_x = cat_rect.left() + 20.0;
                let max_name_w = (count_left - 6.0 - name_x).max(24.0);
                let raw_name = category.display_name();
                let mut name_galley = ui.painter().layout_no_wrap(
                    raw_name.to_owned(),
                    theme.font(FONT_BODY),
                    TEXT_PRIMARY,
                );
                let mut name_truncated = false;
                if name_galley.size().x > max_name_w {
                    name_truncated = true;
                    let chars: Vec<char> = raw_name.chars().collect();
                    let mut len = chars.len();
                    while len > 0 {
                        len -= 1;
                        let candidate: String =
                            chars[..len].iter().collect::<String>() + "...";
                        let g = ui.painter().layout_no_wrap(
                            candidate,
                            theme.font(FONT_BODY),
                            TEXT_PRIMARY,
                        );
                        if g.size().x <= max_name_w || len == 0 {
                            name_galley = g;
                            break;
                        }
                    }
                }
                let actual_count_x = if name_truncated {
                    count_left
                } else {
                    (name_x + name_galley.size().x + 8.0).min(count_left)
                };
                ui.painter().galley(
                    Pos2::new(name_x, cat_rect.center().y - name_galley.size().y * 0.5),
                    name_galley,
                    TEXT_PRIMARY,
                );
                ui.painter().galley(
                    Pos2::new(
                        actual_count_x,
                        cat_rect.center().y - count_galley.size().y * 0.5,
                    ),
                    count_galley,
                    TEXT_MUTED,
                );

                // Category-level pill switch using new_child (NEVER rewinds parent cursor!)
                let cat_pill_state = if enabled_count == all_in_cat.len() {
                    PillToggleState::On
                } else if enabled_count == 0 {
                    PillToggleState::Off
                } else {
                    PillToggleState::Partial
                };
                let mut cat_pill_ui =
                    ui.new_child(egui::UiBuilder::new().max_rect(cat_toggle_rect));
                let cat_toggle_resp =
                    paint_pill_toggle(&mut cat_pill_ui, cat_pill_state, false, theme);
                if cat_toggle_resp.clicked() {
                    let enable_all = enabled_count < all_in_cat.len();
                    let mut blocked = 0usize;
                    for d in &all_in_cat {
                        if d.always_enabled {
                            continue;
                        }
                        if enable_all && !session.plugins.is_enabled(&d.id) {
                            let _ = session.execute(Command::EnablePlugin {
                                plugin_id: d.id.clone(),
                            });
                        } else if !enable_all
                            && session.plugins.is_enabled(&d.id)
                            && session
                                .execute(Command::DisablePlugin {
                                    plugin_id: d.id.clone(),
                                })
                                .is_err()
                        {
                            blocked += 1;
                        }
                    }
                    if blocked > 0 {
                        state.plugin_notice = Some(format!(
                            "{blocked} plugin(s) remained enabled because they are in use by the active project."
                        ));
                    } else {
                        state.plugin_notice = None;
                    }
                }

                if open {
                    if category == PluginCategory::Experimental {
                        warning_banner(ui, zc_plugin::EXPERIMENTAL_WARNING, theme);
                    }

                    for desc in descriptors {
                        let enabled = session.plugins.is_enabled(&desc.id);
                        let in_use = active_deps.contains(&desc.id);
                        let locked = desc.always_enabled || (enabled && in_use);
                        let cm3_compat = is_cm3_compatible_plugin(&desc.id);
                        let summary = zoidium_plugin_summary(&desc.id, &desc.tagline);

                        // Full-bleed flat #2a2a2a row with dynamic height so badges and
                        // multi-line summaries (e.g. Shader FX Pack 4) never clip.
                        let row_w = ui.available_width();
                        let text_w = (row_w - 64.0).max(80.0);
                        let badge_label = if cm3_compat { "Standard" } else { "Extended" };
                        let badge_text_w = ui
                            .painter()
                            .layout_no_wrap(
                                badge_label.to_owned(),
                                theme.font(FONT_BADGE),
                                TEXT_PRIMARY,
                            )
                            .size()
                            .x
                            + 12.0;
                        let total_badges_w = badge_text_w + if in_use { 48.0 } else { 0.0 };
                        let single_title_galley = ui.painter().layout_no_wrap(
                            desc.name.clone(),
                            theme.font(FONT_BODY),
                            TEXT_PRIMARY,
                        );
                        let badges_inline =
                            single_title_galley.size().x + 6.0 + total_badges_w <= text_w;
                        let title_galley = if badges_inline {
                            single_title_galley
                        } else {
                            ui.painter().layout(
                                desc.name.clone(),
                                theme.font(FONT_BODY),
                                TEXT_PRIMARY,
                                text_w,
                            )
                        };
                        let title_h = title_galley.size().y;
                        let header_h = if badges_inline {
                            title_h.max(20.0)
                        } else {
                            title_h + 21.0
                        };

                        let mut summary_job = egui::text::LayoutJob::default();
                        summary_job.wrap.max_width = text_w;
                        summary_job.append(
                            summary,
                            0.0,
                            egui::TextFormat {
                                font_id: theme.font(FONT_SMALL),
                                color: TEXT_MUTED,
                                line_height: Some(18.0),
                                ..Default::default()
                            },
                        );
                        let summary_galley = ui.painter().layout_job(summary_job);
                        let summary_h = summary_galley.size().y;

                        // Suppress duplicate per-row EXPERIMENTAL_WARNING when category banner already shows it
                        let row_warning = desc.warning.as_ref().filter(|w| {
                            !(category == PluginCategory::Experimental
                                && w.as_str() == zc_plugin::EXPERIMENTAL_WARNING)
                        });
                        let warn_galley = row_warning.map(|w| {
                            ui.painter().layout(
                                w.clone(),
                                theme.font(FONT_COMPACT),
                                DANGER_TEXT,
                                text_w,
                            )
                        });
                        let warn_extra_h =
                            warn_galley.as_ref().map_or(0.0, |g| g.size().y + 6.0);
                        let top_pad = 7.0;
                        let summary_offset = top_pad + header_h + 4.0;
                        let row_h =
                            (summary_offset + summary_h + 8.0 + warn_extra_h).max(55.0);

                        let (row_rect, _) =
                            ui.allocate_exact_size(Vec2::new(row_w, row_h), Sense::hover());
                        ui.painter()
                            .rect_filled(row_rect, CornerRadius::ZERO, BG_PANEL);
                        let row_div_y = (row_rect.bottom() - 1.0).round();
                        ui.painter().rect_filled(
                            Rect::from_min_max(
                                Pos2::new(row_rect.left(), row_div_y),
                                Pos2::new(row_rect.right(), row_div_y + 1.0),
                            ),
                            CornerRadius::ZERO,
                            BORDER_ROW_SOFT,
                        );

                        // Deterministic title, badge, and summary painting (never clipped)
                        let text_left = row_rect.left() + 5.0;
                        let text_clip = Rect::from_min_max(
                            Pos2::new(text_left, row_rect.top()),
                            Pos2::new(
                                (row_rect.right() - 56.0).max(text_left + 80.0),
                                row_rect.bottom() - 1.0,
                            ),
                        );
                        let text_painter = ui.painter().with_clip_rect(text_clip);

                        let title_y = row_rect.top() + top_pad;
                        let title_w = title_galley.size().x;
                        text_painter.galley(
                            Pos2::new(text_left, title_y),
                            title_galley,
                            TEXT_PRIMARY,
                        );

                        let (badge_x, badge_y) = if badges_inline {
                            (text_left + title_w + 6.0, title_y + 2.5)
                        } else {
                            (text_left, title_y + title_h + 3.0)
                        };
                        let badge_w = if cm3_compat {
                            paint_outlined_plugin_badge_at(
                                &text_painter,
                                Pos2::new(badge_x, badge_y),
                                "Standard",
                                Color32::from_rgb(0x2d, 0x30, 0x37),
                                theme.value(),
                                Color32::from_rgb(0xb2, 0xbc, 0xd5),
                                theme,
                            )
                        } else {
                            paint_outlined_plugin_badge_at(
                                &text_painter,
                                Pos2::new(badge_x, badge_y),
                                "Extended",
                                Color32::from_rgb(0x38, 0x28, 0x28),
                                WARNING_BAR,
                                DANGER_TEXT,
                                theme,
                            )
                        };
                        if in_use {
                            let in_use_rect = Rect::from_min_size(
                                Pos2::new((badge_x + badge_w + 6.0).round(), badge_y.round()),
                                Vec2::new(42.0, 16.0),
                            );
                            paint_badge(
                                &text_painter,
                                in_use_rect,
                                "IN USE",
                                theme.accent(),
                                TEXT_BRIGHT,
                            );
                        }

                        let summary_y = row_rect.top() + summary_offset;
                        text_painter.galley(
                            Pos2::new(text_left, summary_y),
                            summary_galley,
                            TEXT_MUTED,
                        );
                        if let Some(wg) = warn_galley {
                            text_painter.galley(
                                Pos2::new(text_left, summary_y + summary_h + 4.0),
                                wg,
                                DANGER_TEXT,
                            );
                        }

                        // Right pill switch aligned to x=329..365 (-52.0), vertically centered
                        let pill_rect = Rect::from_min_size(
                            Pos2::new(
                                (row_rect.right() - 52.0).round(),
                                (row_rect.center().y - 11.5).round(),
                            ),
                            Vec2::new(36.0, 20.0),
                        );
                        let mut pill_ui =
                            ui.new_child(egui::UiBuilder::new().max_rect(pill_rect));
                        let pill_state = if enabled {
                            PillToggleState::On
                        } else {
                            PillToggleState::Off
                        };
                        let toggle_resp =
                            paint_pill_toggle(&mut pill_ui, pill_state, locked, theme);
                        if toggle_resp.clicked() && !locked {
                            let cmd = if enabled {
                                Command::DisablePlugin {
                                    plugin_id: desc.id.clone(),
                                }
                            } else {
                                Command::EnablePlugin {
                                    plugin_id: desc.id.clone(),
                                }
                            };
                            match session.execute(cmd) {
                                Ok(_) => state.plugin_notice = None,
                                Err(err) => {
                                    state.plugin_notice = Some(err.to_string());
                                }
                            }
                        }
                    }
                }
            }

            // Subtle bottom footer for Vanilla CM3 Mode & Reset Defaults
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(10, 10))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let half_w = ((ui.available_width() - 6.0) * 0.5).max(80.0);
                        ui.allocate_ui(Vec2::new(half_w, 26.0), |ui| {
                            if cm3_prop_button(
                                ui,
                                "Core Mode",
                                ButtonVariant::Normal,
                                true,
                                theme,
                            )
                            .on_hover_text(
                                "Disable all optional plugins to enter Core Mode",
                            )
                            .clicked()
                            {
                                match session.execute(Command::DisableAllOptionalPlugins) {
                                    Ok(val) => {
                                        let blocked_count = val
                                            .get("blocked")
                                            .and_then(Value::as_array)
                                            .map_or(0, Vec::len);
                                        if blocked_count > 0 {
                                            state.plugin_notice = Some(format!(
                                                "{blocked_count} plugin(s) remained enabled because they are in use by the active project."
                                            ));
                                        } else {
                                            state.plugin_notice =
                                                Some("Switched to Core Mode.".to_owned());
                                        }
                                    }
                                    Err(err) => {
                                        state.plugin_notice = Some(err.to_string());
                                    }
                                }
                            }
                        });
                        ui.add_space(6.0);
                        ui.allocate_ui(Vec2::new(half_w, 26.0), |ui| {
                            if cm3_prop_button(
                                ui,
                                "Reset Defaults",
                                ButtonVariant::Normal,
                                true,
                                theme,
                            )
                            .on_hover_text(
                                "Reset plugins to defaults while preserving in-use project plugins",
                            )
                            .clicked()
                            {
                                session.plugins.reset_to_defaults();
                                let mut kept_in_use = 0usize;
                                for dep_id in &active_deps {
                                    if !session.plugins.is_enabled(dep_id)
                                        && session.plugins.enable(dep_id).is_ok()
                                    {
                                        kept_in_use += 1;
                                    }
                                }
                                if kept_in_use > 0 {
                                    state.plugin_notice = Some(format!(
                                        "Reset plugins to defaults ({kept_in_use} in-use plugin(s) remained enabled)."
                                    ));
                                } else {
                                    state.plugin_notice = None;
                                }
                            }
                        });
                    });
                });
        });
}

// ---------------------------------------------------------------------------
// Tab 11: Settings (Flat Zoidium Settings Matching 07_settings_clean.png)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ThemeColorPopupState {
    open: bool,
    hsv: Hsv,
    hex_buf: String,
}

/// Theme main color picker without a dead viewport eyedropper button.
fn show_theme_color_picker(
    ui: &mut Ui,
    id: Id,
    rgb: &mut Rgb,
    viewport_rect: Rect,
    theme: &Cm3Theme,
) -> bool {
    let popup_id = id.with("theme_color_popup");
    let mut st = ui
        .data_mut(|d| d.get_temp::<ThemeColorPopupState>(popup_id))
        .unwrap_or_else(|| ThemeColorPopupState {
            open: false,
            hsv: rgb.to_hsv(),
            hex_buf: rgb.with_alpha(1.0).to_hex_rgb(),
        });
    let mut changed = false;
    let mut swatch_rect = Rect::NOTHING;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(30.0, 20.0), Sense::click());
        swatch_rect = rect;
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, rgb_to_color32(*rgb));
        ui.painter().rect_stroke(
            rect,
            CornerRadius::ZERO,
            Stroke::new(
                1.0,
                if st.open || resp.hovered() {
                    theme.accent()
                } else {
                    BORDER_CONTROL
                },
            ),
            StrokeKind::Inside,
        );
        if resp.on_hover_cursor(CursorIcon::PointingHand).clicked() {
            st.open = !st.open;
            if st.open {
                st.hsv = rgb.to_hsv();
                st.hex_buf = rgb.with_alpha(1.0).to_hex_rgb();
            }
        }

        // Inline hex input (single-frame, no dead eyedropper)
        let hex_id = id.with("theme_hex_inline");
        if !ui.memory(|m| m.has_focus(hex_id)) && !st.open {
            st.hex_buf = rgb.with_alpha(1.0).to_hex_rgb();
        }
        let hex_resp = Frame::NONE
            .fill(BG_INPUT)
            .stroke(Stroke::new(1.0, BORDER_CONTROL))
            .inner_margin(egui::Margin::symmetric(4, 2))
            .show(ui, |ui| {
                ui.add(
                    TextEdit::singleline(&mut st.hex_buf)
                        .id(hex_id)
                        .frame(Frame::NONE)
                        .font(theme.font(FONT_SMALL))
                        .text_color(theme.value())
                        .desired_width(78.0),
                )
            })
            .inner;
        if hex_resp.lost_focus()
            || (hex_resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter)))
        {
            if let Ok(parsed) = Rgba::from_hex(&st.hex_buf) {
                *rgb = parsed.rgb();
                st.hsv = rgb.to_hsv();
                st.hex_buf = rgb.with_alpha(1.0).to_hex_rgb();
                changed = true;
            }
            if hex_resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                ui.memory_mut(|m| m.surrender_focus(hex_id));
            }
        }
    });

    if st.open && swatch_rect != Rect::NOTHING {
        let popup_size = Vec2::new(230.0, 195.0);
        let mut x = swatch_rect.left();
        let mut y = swatch_rect.bottom() + 2.0;
        if y + popup_size.y > viewport_rect.bottom() - 4.0 {
            y = (swatch_rect.top() - popup_size.y - 2.0).max(viewport_rect.top() + 4.0);
        }
        if x + popup_size.x > viewport_rect.right() - 4.0 {
            x = (viewport_rect.right() - popup_size.x - 4.0).max(viewport_rect.left() + 4.0);
        }
        let area_resp = Area::new(popup_id)
            .order(Order::Foreground)
            .fixed_pos(Pos2::new(x, y))
            .interactable(true)
            .show(ui.ctx(), |ui| {
                Frame::NONE
                    .fill(BG_PANEL)
                    .stroke(Stroke::new(1.0, theme.accent()))
                    .inner_margin(egui::Margin::same(6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            // 180x180 SV square
                            let (sv_rect, sv_resp) =
                                ui.allocate_exact_size(Vec2::splat(180.0), Sense::click_and_drag());
                            let pure = rgb_to_color32(Rgb::from_hsv(Hsv {
                                h: st.hsv.h,
                                s: 1.0,
                                v: 1.0,
                            }));
                            let mut mesh = Mesh::default();
                            mesh.colored_vertex(sv_rect.left_top(), Color32::WHITE);
                            mesh.colored_vertex(sv_rect.right_top(), pure);
                            mesh.colored_vertex(sv_rect.right_bottom(), Color32::BLACK);
                            mesh.colored_vertex(sv_rect.left_bottom(), Color32::BLACK);
                            mesh.add_triangle(0, 1, 2);
                            mesh.add_triangle(0, 2, 3);
                            ui.painter().add(Shape::mesh(mesh));
                            if (sv_resp.clicked() || sv_resp.dragged())
                                && let Some(p) = ui.input(|i| i.pointer.interact_pos())
                            {
                                st.hsv.s =
                                    ((p.x - sv_rect.left()) / sv_rect.width()).clamp(0.0, 1.0);
                                st.hsv.v = (1.0 - (p.y - sv_rect.top()) / sv_rect.height())
                                    .clamp(0.0, 1.0);
                                *rgb = Rgb::from_hsv(st.hsv);
                                st.hex_buf = rgb.with_alpha(1.0).to_hex_rgb();
                                changed = true;
                            }
                            let ret = Pos2::new(
                                sv_rect.left() + st.hsv.s * sv_rect.width(),
                                sv_rect.top() + (1.0 - st.hsv.v) * sv_rect.height(),
                            );
                            ui.painter().circle_stroke(
                                ret,
                                4.0,
                                Stroke::new(1.5, Color32::from_gray(0xcc)),
                            );

                            // 22x180 Hue bar
                            let (hue_rect, hue_resp) = ui.allocate_exact_size(
                                Vec2::new(22.0, 180.0),
                                Sense::click_and_drag(),
                            );
                            let stops = [
                                Color32::from_rgb(255, 0, 0),
                                Color32::from_rgb(255, 255, 0),
                                Color32::from_rgb(0, 255, 0),
                                Color32::from_rgb(0, 255, 255),
                                Color32::from_rgb(0, 0, 255),
                                Color32::from_rgb(255, 0, 255),
                                Color32::from_rgb(255, 0, 0),
                            ];
                            let mut hmesh = Mesh::default();
                            let seg_h = hue_rect.height() / 6.0;
                            for i in 0..6 {
                                let y0 = hue_rect.top() + seg_h * i as f32;
                                let y1 = hue_rect.top() + seg_h * (i + 1) as f32;
                                let base = hmesh.vertices.len() as u32;
                                hmesh.colored_vertex(Pos2::new(hue_rect.left(), y0), stops[i]);
                                hmesh.colored_vertex(Pos2::new(hue_rect.right(), y0), stops[i]);
                                hmesh.colored_vertex(Pos2::new(hue_rect.right(), y1), stops[i + 1]);
                                hmesh.colored_vertex(Pos2::new(hue_rect.left(), y1), stops[i + 1]);
                                hmesh.add_triangle(base, base + 1, base + 2);
                                hmesh.add_triangle(base, base + 2, base + 3);
                            }
                            ui.painter().add(Shape::mesh(hmesh));
                            if (hue_resp.clicked() || hue_resp.dragged())
                                && let Some(p) = ui.input(|i| i.pointer.interact_pos())
                            {
                                st.hsv.h =
                                    ((p.y - hue_rect.top()) / hue_rect.height()).clamp(0.0, 0.9999);
                                *rgb = Rgb::from_hsv(st.hsv);
                                st.hex_buf = rgb.with_alpha(1.0).to_hex_rgb();
                                changed = true;
                            }
                            let hy = hue_rect.top() + st.hsv.h * hue_rect.height();
                            ui.painter().line_segment(
                                [
                                    Pos2::new(hue_rect.left(), hy),
                                    Pos2::new(hue_rect.right(), hy),
                                ],
                                Stroke::new(2.0, Color32::WHITE),
                            );
                        });
                    });
            });

        let close_popup = ui.input(|i| {
            i.key_pressed(Key::Escape)
                || (i.pointer.primary_pressed()
                    && i.pointer.interact_pos().is_some_and(|p| {
                        !area_resp.response.rect.contains(p) && !swatch_rect.contains(p)
                    }))
        });
        if close_popup {
            st.open = false;
        }
    }

    ui.data_mut(|d| d.insert_temp(popup_id, st));
    changed
}

fn settings_flat_row_margin(ui: &mut Ui, margin: egui::Margin, add_content: impl FnOnce(&mut Ui)) {
    let resp = Frame::NONE
        .fill(BG_PANEL)
        .inner_margin(margin)
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            add_content(ui);
        });
    let r = resp.response.rect;
    let div_y = (r.bottom() - 1.0).round();
    ui.painter().rect_filled(
        Rect::from_min_max(
            Pos2::new(r.left(), div_y),
            Pos2::new(r.right(), div_y + 1.0),
        ),
        CornerRadius::ZERO,
        BORDER_ROW_SOFT,
    );
}

fn settings_flat_row_pad(ui: &mut Ui, pad_v: i8, add_content: impl FnOnce(&mut Ui)) {
    settings_flat_row_margin(
        ui,
        egui::Margin {
            left: 6,
            right: 5,
            top: pad_v,
            bottom: pad_v,
        },
        add_content,
    );
}

/// Fixed-width `fixed_w × 21 px` CM3 Settings dropdown with `14 px` (`FONT_SMALL`) interior text
/// matching exact Zoidium `07_settings_clean.png` geometry (`222 / 78 / 230 / 119 px`).
fn settings_fixed_dropdown(
    ui: &mut Ui,
    id: Id,
    selected_idx: &mut usize,
    options: &[&str],
    fixed_w: f32,
    viewport_rect: Rect,
    theme: &Cm3Theme,
) -> bool {
    if options.is_empty() {
        return false;
    }
    if *selected_idx >= options.len() {
        *selected_idx = 0;
    }
    let popup_id = id.with("settings_dd_popup");
    let mut open = ui
        .data_mut(|d| d.get_temp::<bool>(popup_id))
        .unwrap_or(false);
    let mut changed = false;

    let current_label = options[*selected_idx];
    let label_galley = ui.painter().layout_no_wrap(
        current_label.to_owned(),
        theme.font(FONT_SMALL),
        theme.value(),
    );
    let desired_w = fixed_w.max(label_galley.size().x + 26.0);
    let width = desired_w.min(ui.available_width().max(44.0)).round();
    let (raw_btn_rect, btn_resp) = ui.allocate_exact_size(Vec2::new(width, 21.0), Sense::click());
    let btn_rect = Rect::from_min_size(
        Pos2::new(raw_btn_rect.left().round(), raw_btn_rect.top().round()),
        Vec2::new(width, 21.0),
    );
    let btn_resp = btn_resp.on_hover_cursor(CursorIcon::PointingHand);

    ui.painter()
        .rect_filled(btn_rect, CornerRadius::ZERO, BG_INPUT);
    paint_crisp_panel_border(
        ui.painter(),
        btn_rect,
        if open || btn_resp.hovered() {
            theme.accent()
        } else {
            BORDER_CONTROL
        },
    );

    let clip_painter = ui.painter().with_clip_rect(Rect::from_min_max(
        btn_rect.left_top(),
        Pos2::new(
            (btn_rect.right() - 19.0).max(btn_rect.left()),
            btn_rect.bottom(),
        ),
    ));
    clip_painter.galley(
        Pos2::new(
            btn_rect.left() + 5.0,
            btn_rect.center().y - label_galley.size().y * 0.5,
        ),
        label_galley,
        theme.value(),
    );
    let caret_rect = Rect::from_center_size(
        Pos2::new(btn_rect.right() - 8.5, btn_rect.center().y),
        Vec2::splat(11.0),
    );
    paint_cm3_icon(
        ui.painter(),
        caret_rect,
        Cm3Icon::ChevronDown,
        theme.value(),
    );

    if btn_resp.clicked() {
        open = !open;
    }

    if open {
        let row_h = 22.0;
        let popup_h = (options.len() as f32 * row_h + 4.0).min(240.0);
        let mut x = btn_rect.left().round();
        let mut y = (btn_rect.bottom() + 1.0).round();
        if y + popup_h > viewport_rect.bottom() - 4.0 {
            y = (btn_rect.top() - popup_h - 1.0).max(viewport_rect.top() + 4.0);
        }
        if x + width > viewport_rect.right() - 4.0 {
            x = (viewport_rect.right() - width - 4.0).max(viewport_rect.left() + 4.0);
        }
        let esc = ui.input(|i| i.key_pressed(Key::Escape));
        if esc {
            open = false;
        }
        let area_resp = Area::new(popup_id)
            .order(Order::Foreground)
            .fixed_pos(Pos2::new(x, y))
            .interactable(true)
            .show(ui.ctx(), |ui| {
                Frame::NONE
                    .fill(BG_INPUT)
                    .stroke(Stroke::new(1.0, theme.accent()))
                    .inner_margin(egui::Margin::same(2))
                    .show(ui, |ui| {
                        ui.set_min_width(width - 4.0);
                        ui.spacing_mut().item_spacing.y = 0.0;
                        ScrollArea::vertical().max_height(236.0).show(ui, |ui| {
                            for (idx, opt) in options.iter().enumerate() {
                                let (r, resp) = ui.allocate_exact_size(
                                    Vec2::new((width - 4.0).max(40.0), row_h),
                                    Sense::click(),
                                );
                                let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
                                if idx == *selected_idx {
                                    ui.painter()
                                        .rect_filled(r, CornerRadius::ZERO, theme.accent());
                                } else if resp.hovered() {
                                    ui.painter().rect_filled(
                                        r,
                                        CornerRadius::ZERO,
                                        BG_DROPDOWN_HOVER,
                                    );
                                }
                                ui.painter().with_clip_rect(r).text(
                                    Pos2::new(r.left() + 4.0, r.center().y),
                                    Align2::LEFT_CENTER,
                                    *opt,
                                    theme.font(FONT_SMALL),
                                    if idx == *selected_idx || resp.hovered() {
                                        TEXT_BRIGHT
                                    } else {
                                        TEXT_PRIMARY
                                    },
                                );
                                if resp.clicked() {
                                    if *selected_idx != idx {
                                        *selected_idx = idx;
                                        changed = true;
                                    }
                                    open = false;
                                }
                            }
                        });
                    });
            });
        let clicked_outside = ui.input(|i| {
            i.pointer.primary_pressed()
                && i.pointer
                    .interact_pos()
                    .is_some_and(|p| !area_resp.response.rect.contains(p) && !btn_rect.contains(p))
        });
        if clicked_outside {
            open = false;
        }
    }

    ui.data_mut(|d| d.insert_temp(popup_id, open));
    changed
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct EditorLayoutPreset {
    pub(super) name: &'static str,
    pub(super) description: &'static str,
    pub(super) inspector_width_ratio: f32,
    pub(super) viewport_height_ratio: f32,
    pub(super) objects_split_ratio: f32,
    pub(super) effects_split_ratio: f32,
    pub(super) prop_label_ratio: f32,
    pub(super) objects_list_open: bool,
}

pub(super) const EDITOR_LAYOUT_PRESETS: [EditorLayoutPreset; 4] = [
    EditorLayoutPreset {
        name: "Classic",
        description: "Balanced inspector, viewport, and timeline proportions.",
        inspector_width_ratio: 0.31,
        viewport_height_ratio: 0.65,
        objects_split_ratio: 0.40,
        effects_split_ratio: 0.40,
        prop_label_ratio: 0.44,
        objects_list_open: true,
    },
    EditorLayoutPreset {
        name: "Compact",
        description: "Tighter inspector labels with compact object and effects panes.",
        inspector_width_ratio: 0.26,
        viewport_height_ratio: 0.62,
        objects_split_ratio: 0.30,
        effects_split_ratio: 0.30,
        prop_label_ratio: 0.38,
        objects_list_open: true,
    },
    EditorLayoutPreset {
        name: "Wide Viewport",
        description: "Maximizes canvas space with a slim inspector and collapsed object list.",
        inspector_width_ratio: 0.24,
        viewport_height_ratio: 0.74,
        objects_split_ratio: 0.26,
        effects_split_ratio: 0.28,
        prop_label_ratio: 0.40,
        objects_list_open: false,
    },
    EditorLayoutPreset {
        name: "Color & FX",
        description: "Expanded inspector and effects proportions for grading and shader work.",
        inspector_width_ratio: 0.36,
        viewport_height_ratio: 0.68,
        objects_split_ratio: 0.34,
        effects_split_ratio: 0.48,
        prop_label_ratio: 0.46,
        objects_list_open: true,
    },
];

pub(super) const EDITOR_LAYOUT_LABELS: [&str; 4] = [
    EDITOR_LAYOUT_PRESETS[0].name,
    EDITOR_LAYOUT_PRESETS[1].name,
    EDITOR_LAYOUT_PRESETS[2].name,
    EDITOR_LAYOUT_PRESETS[3].name,
];

/// Applies one of the 4 [`EDITOR_LAYOUT_PRESETS`] to `state`, updating concrete split ratios
/// and panel visibility immediately.
pub(super) fn apply_editor_layout_preset(
    index: usize,
    state: &mut PanelsState,
) -> &'static EditorLayoutPreset {
    let clamped = index.min(EDITOR_LAYOUT_PRESETS.len() - 1);
    let preset = &EDITOR_LAYOUT_PRESETS[clamped];
    state.settings_editor_layout = clamped;
    state.objects_split.ratio = preset.objects_split_ratio;
    state.effects_split.ratio = preset.effects_split_ratio;
    state.prop_columns.label_ratio = preset.prop_label_ratio;
    state.objects_list_open = preset.objects_list_open;
    preset
}

pub(super) fn show_settings_tab(
    ui: &mut Ui,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &mut Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    // Hydrate persisted editor layout preset once per egui context lifetime.
    let layout_persist_id = Id::unique("zc_editor_layout_preset");
    let layout_ratios_id = Id::unique("zc_editor_layout_ratios");
    let layout_hydrated_id = Id::unique("zc_editor_layout_hydrated");
    if !ui
        .ctx()
        .data(|d| d.get_temp::<bool>(layout_hydrated_id).unwrap_or(false))
    {
        ui.ctx()
            .data_mut(|d| d.insert_temp(layout_hydrated_id, true));
        if let Some(saved_idx) = ui
            .ctx()
            .data_mut(|d| d.get_persisted::<usize>(layout_persist_id))
        {
            let preset = apply_editor_layout_preset(saved_idx, state);
            ui.ctx().data_mut(|d| {
                d.insert_persisted(
                    layout_ratios_id,
                    (preset.inspector_width_ratio, preset.viewport_height_ratio),
                );
            });
        }
    }

    ScrollArea::vertical()
        .id_salt("settings_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;

            // Edge-to-edge 43px #222222 header bar ("Settings")
            panel_top_header(ui, "Settings", theme);

            // 1a. Editor layout control row (single-line ellipsized label + guaranteed gap before dropdown)
            let mut layout_changed = false;
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 5,
                    top: 6,
                    bottom: 6,
                },
                |ui| {
                    let total_w = ui.available_width();
                    let dd_w = 222.0_f32.min((total_w * 0.52).max(80.0));
                    let label_w = (total_w - dd_w - 8.0).max(48.0);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 8.0;
                        ui.allocate_ui_with_layout(
                            Vec2::new(label_w, 26.0),
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new("Editor layout")
                                            .font(theme.font(FONT_BODY))
                                            .color(TEXT_PRIMARY),
                                    )
                                    .truncate(),
                                );
                            },
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if settings_fixed_dropdown(
                                ui,
                                Id::unique("settings_editor_layout"),
                                &mut state.settings_editor_layout,
                                &EDITOR_LAYOUT_LABELS,
                                dd_w,
                                viewport_rect,
                                theme,
                            ) {
                                layout_changed = true;
                            }
                        });
                    });
                },
            );
            if layout_changed {
                let preset = apply_editor_layout_preset(state.settings_editor_layout, state);
                let idx = state.settings_editor_layout;
                ui.ctx().data_mut(|d| {
                    d.insert_persisted(layout_persist_id, idx);
                    d.insert_persisted(
                        layout_ratios_id,
                        (preset.inspector_width_ratio, preset.viewport_height_ratio),
                    );
                });
                actions.push(PanelAction::StatusMessage(format!(
                    "Editor layout applied: {}",
                    preset.name
                )));
            }

            // 1b. Editor layout description row
            let active_layout = &EDITOR_LAYOUT_PRESETS
                [state.settings_editor_layout.min(EDITOR_LAYOUT_PRESETS.len() - 1)];
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 4,
                    top: 6,
                    bottom: 5,
                },
                |ui| {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(active_layout.description)
                                .font(theme.font(FONT_BODY))
                                .color(TEXT_PRIMARY),
                        )
                        .wrap(),
                    );
                },
            );

            // 2. Default frame rate: boxed #202020 1px #1b1b1b 90x21 numeric input (32px total)
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 5,
                    top: 5,
                    bottom: 5,
                },
                |ui| {
                    let fps_id = PanelsState::default_fps_input_id();
                    let was_edited = state.settings_default_fps_buf.is_some();
                    let mut fps_buf = state.settings_default_fps_buf.take()
                        .unwrap_or_else(|| state.app_settings.default_fps.to_string());
                    ui.horizontal(|ui| {
                        let avail_w = ui.available_width();
                        let box_w = (avail_w * 0.34).clamp(52.0, 90.0);
                        let label_w = (avail_w - box_w - 8.0).max(40.0);
                        ui.allocate_ui_with_layout(
                            Vec2::new(label_w, 21.0),
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new("Default frame rate")
                                            .font(theme.font(FONT_BODY))
                                            .color(TEXT_PRIMARY),
                                    )
                                    .truncate(),
                                );
                            },
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let focused = ui.memory(|m| m.has_focus(fps_id));
                            let resp = ui
                                .allocate_ui_with_layout(
                                    Vec2::new(box_w, 21.0),
                                    egui::Layout::left_to_right(egui::Align::Center),
                                    |ui| {
                                        let box_rect = Rect::from_min_size(
                                            ui.cursor().min,
                                            Vec2::new(box_w, 21.0),
                                        );
                                        ui.painter()
                                            .rect_filled(box_rect, CornerRadius::ZERO, BG_INPUT);
                                        paint_crisp_panel_border(
                                            ui.painter(),
                                            box_rect,
                                            if focused {
                                                theme.accent()
                                            } else {
                                                BORDER_CONTROL
                                            },
                                        );
                                        Frame::NONE
                                            .inner_margin(egui::Margin::symmetric(5, 1))
                                            .show(ui, |ui| {
                                                ui.add(
                                                    TextEdit::singleline(&mut fps_buf)
                                                    .id(fps_id)
                                                    .frame(Frame::NONE)
                                                    .font(theme.font(FONT_SMALL))
                                                    .text_color(theme.value())
                                                    .desired_width(box_w - 10.0),
                                                )
                                            })
                                            .inner
                                    },
                                )
                                .inner;
                            if was_edited || resp.changed() {
                                state.settings_default_fps_buf = Some(fps_buf);
                            }
                            if resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                                resp.surrender_focus();
                            }
                        });
                    });
                },
            );

            // 3a. Preview cache control row (single-line ellipsized label + guaranteed gap before value & "MiB")
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 9,
                    right: 5,
                    top: 6,
                    bottom: 5,
                },
                |ui| {
                    let mut cache = f64::from(state.app_settings.preview_cache_mib);
                    let mut cache_ev = ScrubEvent::default();
                    ui.horizontal(|ui| {
                        let avail_w = ui.available_width();
                        let right_reserve = 92.0_f32.min((avail_w * 0.48).max(70.0));
                        let label_w = (avail_w - right_reserve - 8.0).max(48.0);
                        ui.allocate_ui_with_layout(
                            Vec2::new(label_w, 21.0),
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new("Preview cache")
                                            .font(theme.font(FONT_BODY))
                                            .color(TEXT_PRIMARY),
                                    )
                                    .truncate(),
                                );
                            },
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new("MiB")
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_FAINT),
                            );
                            ui.add_space(8.0);
                            cache_ev = cm3_scrub_number(
                                ui,
                                Id::unique("settings_cache"),
                                &mut cache,
                                ScrubConfig::new(0.0, 16_384.0, 64.0, 0),
                                theme,
                            );
                        });
                    });
                    if cache_ev.changed || cache_ev.committed {
                        state.app_settings.preview_cache_mib =
                            cache.round().clamp(0.0, 16_384.0) as u32;
                    }
                },
            );

            // 3b. Preview cache telemetry row (60px total, left=9 indent, 23px line-height pitch)
            let [rw, rh] = session.document().resolution;
            let frame_bytes = u64::from(rw.max(1)) * u64::from(rh.max(1)) * 4;
            let est_frames = if state.app_settings.preview_cache_mib == 0 {
                0
            } else {
                (u64::from(state.app_settings.preview_cache_mib) * 1024 * 1024)
                    / frame_bytes.max(1)
            };
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 9,
                    right: 4,
                    top: 7,
                    bottom: 7,
                },
                |ui| {
                    let used_bytes = ui.ctx().data(|d| {
                        d.get_temp::<u64>(Id::unique("zc_preview_cache_used_bytes"))
                            .unwrap_or(0)
                    });
                    let used_mib = used_bytes as f64 / (1024.0 * 1024.0);
                    let used_label = if used_bytes == 0 {
                        "0 MiB".to_owned()
                    } else if used_mib < 10.0 {
                        format!("{used_mib:.1} MiB")
                    } else {
                        format!("{used_mib:.0} MiB")
                    };
                    let mut job = egui::text::LayoutJob::default();
                    job.append(
                        &format!("Using {used_label} · about {est_frames} frames at {rw}×{rh}"),
                        0.0,
                        egui::TextFormat {
                            font_id: theme.font(FONT_BODY),
                            color: TEXT_MUTED,
                            line_height: Some(23.0),
                            ..Default::default()
                        },
                    );
                    ui.add(egui::Label::new(job).wrap());
                },
            );

            // 3c. Preview cache description row (149px total, left=9 indent, 22.4px line-height pitch)
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 9,
                    right: 4,
                    top: 8,
                    bottom: 7,
                },
                |ui| {
                    let mut job = egui::text::LayoutJob::default();
                    job.append(
                        "Finished preview frames use up to this much GPU memory. Set 0 to turn off caching. Temporal input textures use up to another 256 MiB, plus active samples and rendering buffers.",
                        0.0,
                        egui::TextFormat {
                            font_id: theme.font(FONT_BODY),
                            color: TEXT_MUTED,
                            line_height: Some(22.4),
                            ..Default::default()
                        },
                    );
                    ui.add(egui::Label::new(job).wrap());
                },
            );

            // 4. Gradient editor (33px total, 78x21 dropdown)
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 5,
                    top: 6,
                    bottom: 6,
                },
                |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Gradient editor")
                                .font(theme.font(FONT_BODY))
                                .color(TEXT_PRIMARY),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            settings_fixed_dropdown(
                                ui,
                                Id::unique("settings_gradient_editor"),
                                &mut state.settings_gradient_editor,
                                &["Modern", "Classic"],
                                78.0,
                                viewport_rect,
                                theme,
                            );
                        });
                    });
                    ui.ctx().data_mut(|d| {
                        d.insert_temp(
                            Id::unique("zc_gradient_editor_style"),
                            state.settings_gradient_editor,
                        );
                    });
                },
            );

            // 5. Editor font (51px total, 230x21 dropdown TOP-aligned at top_margin = 6px)
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 5,
                    top: 5,
                    bottom: 6,
                },
                |ui| {
                    let mut font_idx = EditorFontChoice::ALL
                        .iter()
                        .position(|&f| f == theme.editor_font)
                        .unwrap_or(0);
                    let font_labels: Vec<&str> =
                        EditorFontChoice::ALL.iter().map(|f| f.label()).collect();
                    let mut changed = false;
                    let total_w = ui.available_width();
                    let dd_w = 230.0_f32.min((total_w - 60.0).max(90.0));
                    let label_w = (total_w - dd_w - 6.0).max(52.0);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        ui.allocate_ui_with_layout(
                            Vec2::new(label_w, 40.0),
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new("Editor font")
                                            .font(theme.font(FONT_BODY))
                                            .color(TEXT_PRIMARY),
                                    )
                                    .wrap(),
                                );
                            },
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                            Frame::NONE
                                .inner_margin(egui::Margin {
                                    left: 0,
                                    right: 0,
                                    top: 1,
                                    bottom: 0,
                                })
                                .show(ui, |ui| {
                                    changed = settings_fixed_dropdown(
                                        ui,
                                        Id::unique("settings_editor_font"),
                                        &mut font_idx,
                                        &font_labels,
                                        dd_w,
                                        viewport_rect,
                                        theme,
                                    );
                                });
                        });
                    });
                    if changed {
                        theme.editor_font = EditorFontChoice::ALL[font_idx];
                    }
                },
            );

            // 6. Color profile (32px total: 119x21 dropdown + 5px gap + 54x21 "Edit..." button)
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 5,
                    top: 5,
                    bottom: 6,
                },
                |ui| {
                    ui.spacing_mut().item_spacing.y = 6.0;
                    let mut prof_idx = ThemeProfileChoice::ALL
                        .iter()
                        .position(|&p| p == theme.profile)
                        .unwrap_or(0);
                    let labels: Vec<&str> =
                        ThemeProfileChoice::ALL.iter().map(|p| p.label()).collect();
                    let mut prof_changed = false;

                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 5.0;
                        let profile_label = if ui.available_width() < 270.0 {
                            "Profile"
                        } else {
                            "Color profile"
                        };
                        ui.label(
                            egui::RichText::new(profile_label)
                                .font(theme.font(FONT_BODY))
                                .color(TEXT_PRIMARY),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.spacing_mut().item_spacing.x = 5.0;
                            // Compact 46x21 "Edit..." button so the Color profile dropdown keeps full width at 900x600
                            let edit_w = 46.0;
                            let (raw_edit_rect, edit_resp) =
                                ui.allocate_exact_size(Vec2::new(edit_w, 21.0), Sense::click());
                            let edit_rect = Rect::from_min_size(
                                Pos2::new(raw_edit_rect.left().round(), raw_edit_rect.top().round()),
                                Vec2::new(edit_w, 21.0),
                            );
                            let edit_resp = edit_resp.on_hover_cursor(CursorIcon::PointingHand);
                            ui.painter().rect_filled(
                                edit_rect,
                                CornerRadius::ZERO,
                                if state.settings_edit_colors || edit_resp.hovered() {
                                    BG_HOVER
                                } else {
                                    BG_TITLE
                                },
                            );
                            paint_crisp_panel_border(
                                ui.painter(),
                                edit_rect,
                                if state.settings_edit_colors || edit_resp.hovered() {
                                    theme.accent()
                                } else {
                                    BORDER_CONTROL
                                },
                            );
                            ui.painter().text(
                                edit_rect.center(),
                                Align2::CENTER_CENTER,
                                "Edit...",
                                theme.font(FONT_BADGE),
                                theme.value(),
                            );
                            if edit_resp.clicked() {
                                state.settings_edit_colors = !state.settings_edit_colors;
                            }

                            prof_changed = settings_fixed_dropdown(
                                ui,
                                Id::unique("settings_theme_profile"),
                                &mut prof_idx,
                                &labels,
                                119.0,
                                viewport_rect,
                                theme,
                            );
                        });
                    });
                    if prof_changed {
                        theme.set_profile(ThemeProfileChoice::ALL[prof_idx]);
                    }

                    // Inline custom color / hue / sat editor revealed ONLY when "Edit..." is toggled
                    if state.settings_edit_colors {
                        let mut color_cols = state.prop_columns;
                        color_cols.label_ratio = color_cols.label_ratio.max(0.45);
                        let mut main_rgb = theme.custom_main_rgb;
                        let color_changed = cm3_property_row_aligned(
                            ui,
                            "Main color",
                            false,
                            color_cols,
                            theme,
                            |_| {},
                            |ui| {
                                show_theme_color_picker(
                                    ui,
                                    Id::unique("settings_main_color"),
                                    &mut main_rgb,
                                    viewport_rect,
                                    theme,
                                )
                            },
                        );
                        if color_changed {
                            theme.set_custom_main(main_rgb);
                        }

                        let mut hue = f64::from(theme.custom_hue_deg);
                        let hev = cm3_property_row_aligned(
                            ui,
                            "Hue shift",
                            false,
                            color_cols,
                            theme,
                            |_| {},
                            |ui| {
                                let mut ev = ScrubEvent::default();
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 4.0;
                                    ev = cm3_scrub_number(
                                        ui,
                                        Id::unique("settings_hue"),
                                        &mut hue,
                                        ScrubConfig::new(-180.0, 180.0, 1.0, 0),
                                        theme,
                                    );
                                    ui.label(
                                        egui::RichText::new("deg")
                                            .font(theme.font(FONT_COMPACT))
                                            .color(TEXT_FAINT),
                                    );
                                });
                                ev
                            },
                        );
                        if hev.changed || hev.committed {
                            theme.set_custom_hue_sat(hue as f32, theme.custom_sat_percent);
                        }

                        let mut sat = f64::from(theme.custom_sat_percent);
                        let sev = cm3_property_row_aligned(
                            ui,
                            "Saturation",
                            false,
                            color_cols,
                            theme,
                            |_| {},
                            |ui| {
                                let mut ev = ScrubEvent::default();
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 4.0;
                                    ev = cm3_scrub_number(
                                        ui,
                                        Id::unique("settings_sat"),
                                        &mut sat,
                                        ScrubConfig::new(0.0, 200.0, 1.0, 0),
                                        theme,
                                    );
                                    ui.label(
                                        egui::RichText::new("%")
                                            .font(theme.font(FONT_COMPACT))
                                            .color(TEXT_FAINT),
                                    );
                                });
                                ev
                            },
                        );
                        if sev.changed || sev.committed {
                            theme.set_custom_hue_sat(theme.custom_hue_deg, sat as f32);
                        }

                        let slot_specs = [
                            (0usize, "Focus", theme.palette.focus),
                            (1usize, "Link", theme.palette.link),
                            (2usize, "Clip", theme.palette.clip),
                            (3usize, "Clip Sel", theme.palette.clip_selected),
                            (4usize, "Sel Box", theme.palette.clip_selection),
                        ];
                        for (slot, label, cur_rgb) in slot_specs {
                            let mut rgb = cur_rgb;
                            let changed = cm3_property_row_aligned(
                                ui,
                                label,
                                false,
                                color_cols,
                                theme,
                                |_| {},
                                |ui| {
                                    show_theme_color_picker(
                                        ui,
                                        Id::unique(("settings_slot_color", slot)),
                                        &mut rgb,
                                        viewport_rect,
                                        theme,
                                    )
                                },
                            );
                            if changed {
                                theme.set_slot_override(slot, rgb);
                            }
                        }

                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            if ui.small_button("Reset Slots").clicked() {
                                theme.clear_slot_overrides();
                            }
                            if ui.small_button("Copy JSON").clicked() {
                                let json_str = theme.export_theme_json();
                                write_system_clipboard(ui.ctx(), &json_str);
                                state.debug_copied_notice =
                                    Some("Copied theme JSON to clipboard.".to_owned());
                            }
                            if ui.small_button("Paste JSON").clicked()
                                && let Some(clip_txt) = read_system_clipboard()
                                && theme.import_theme_json(&clip_txt)
                            {
                                state.debug_copied_notice =
                                    Some("Imported theme JSON from clipboard.".to_owned());
                            }
                        });
                    }
                },
            );

            // 6b. MEDIA TOOLS (FFMPEG): desktop external binary configuration (No FFmpeg Inside)
            #[cfg(not(target_arch = "wasm32"))]
            {
                let (ff_hdr, _) =
                    ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), Sense::hover());
                ui.painter()
                    .rect_filled(ff_hdr, CornerRadius::ZERO, BG_TITLE);
                let ff_div_y = (ff_hdr.bottom() - 1.0).round();
                ui.painter().rect_filled(
                    Rect::from_min_max(
                        Pos2::new(ff_hdr.left(), ff_div_y),
                        Pos2::new(ff_hdr.right(), ff_div_y + 1.0),
                    ),
                    CornerRadius::ZERO,
                    BG_SUNKEN,
                );
                ui.painter().text(
                    Pos2::new(ff_hdr.left() + 6.0, ff_hdr.center().y),
                    Align2::LEFT_CENTER,
                    "MEDIA TOOLS (FFMPEG)",
                    theme.font(FONT_BANNER),
                    TEXT_FAINT,
                );

                settings_flat_row_margin(
                    ui,
                    egui::Margin {
                        left: 6,
                        right: 6,
                        top: 6,
                        bottom: 8,
                    },
                    |ui| {
                        ui.spacing_mut().item_spacing.y = 5.0;
                        let ff_ok = state.export_status.ffmpeg_available;
                        let status_color = if ff_ok {
                            Color32::from_rgb(76, 198, 122)
                        } else {
                            DANGER_TEXT
                        };
                        ui.label(
                            egui::RichText::new(format!(
                                "FFmpeg: {}",
                                state.export_status.ffmpeg_version
                            ))
                            .font(theme.font(FONT_SMALL))
                            .color(status_color),
                        );
                        let path_display = state
                            .app_settings
                            .ffmpeg_path
                            .as_deref()
                            .map(|p| format!("Custom: {p}"))
                            .or_else(|| {
                                state
                                    .export_status
                                    .ffmpeg_resolved_path
                                    .as_deref()
                                    .map(|p| format!("Auto: {p}"))
                            })
                            .unwrap_or_else(|| "Auto-detect (not found)".to_owned());
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(path_display)
                                    .font(theme.font(FONT_COMPACT))
                                    .color(TEXT_MUTED),
                            )
                            .truncate(),
                        );
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            if cm3_prop_button(
                                ui,
                                if ff_ok { "Setup..." } else { "Install FFmpeg..." },
                                if ff_ok {
                                    ButtonVariant::Normal
                                } else {
                                    ButtonVariant::Primary
                                },
                                false,
                                theme,
                            )
                            .clicked()
                            {
                                actions.push(PanelAction::OpenFfmpegSetup);
                            }
                            if cm3_prop_button(ui, "Browse...", ButtonVariant::Normal, false, theme)
                                .clicked()
                            {
                                actions.push(PanelAction::BrowseFfmpegBinary);
                            }
                            if state.app_settings.ffmpeg_path.is_some()
                                && cm3_prop_button(ui, "Reset", ButtonVariant::Normal, false, theme)
                                    .clicked()
                            {
                                actions.push(PanelAction::ClearFfmpegBinary);
                            }
                            if cm3_prop_button(ui, "Re-check", ButtonVariant::Normal, false, theme)
                                .clicked()
                            {
                                actions.push(PanelAction::ReprobeFfmpeg);
                            }
                        });
                    },
                );
            }

            // 7. DEBUG INFORMATION: full-width 28px #222222 uppercase banner (27px body + 1px #1d1d1d divider)
            let (dbg_hdr, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), Sense::hover());
            ui.painter()
                .rect_filled(dbg_hdr, CornerRadius::ZERO, BG_TITLE);
            let dbg_div_y = (dbg_hdr.bottom() - 1.0).round();
            ui.painter().rect_filled(
                Rect::from_min_max(
                    Pos2::new(dbg_hdr.left(), dbg_div_y),
                    Pos2::new(dbg_hdr.right(), dbg_div_y + 1.0),
                ),
                CornerRadius::ZERO,
                BG_SUNKEN,
            );
            ui.painter().text(
                Pos2::new(dbg_hdr.left() + 6.0, dbg_hdr.center().y),
                Align2::LEFT_CENTER,
                "DEBUG INFORMATION",
                theme.font(FONT_BANNER),
                TEXT_FAINT,
            );

            // 7a. Native desktop debug description + Copy Debug Info button
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 6,
                    top: 6,
                    bottom: 8,
                },
                |ui| {
                    ui.spacing_mut().item_spacing.y = 6.0;
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(
                                "Copies your OS, ZoidCraft version, enabled plugins, and undo state for a bug report. Project content is not included; review before sharing.",
                            )
                            .font(theme.font(FONT_BODY))
                            .color(TEXT_PRIMARY),
                        )
                        .wrap(),
                    );
                    if cm3_prop_button(ui, "Copy Debug Info", ButtonVariant::Normal, true, theme)
                        .clicked()
                    {
                        let info = build_debug_info_report(session, &state.app_settings);
                        write_system_clipboard(ui.ctx(), &info);
                        state.debug_copied_notice =
                            Some("Copied debug diagnostics to clipboard.".to_owned());
                    }
                    if let Some(msg) = &state.debug_copied_notice {
                        ui.label(
                            egui::RichText::new(msg)
                                .font(theme.font(FONT_SMALL))
                                .color(theme.value()),
                        );
                    }
                },
            );

            // 7b. Additional ZoidCraft controls below the fold (accessible on scroll)
            settings_flat_row_pad(ui, 8, |ui| {
                ui.spacing_mut().item_spacing.y = 6.0;
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Show FPS overlay")
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_PRIMARY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let st = if state.app_settings.show_fps_overlay {
                            PillToggleState::On
                        } else {
                            PillToggleState::Off
                        };
                        if paint_pill_toggle(ui, st, false, theme).clicked() {
                            state.app_settings.show_fps_overlay =
                                !state.app_settings.show_fps_overlay;
                        }
                    });
                });

                let mut max_entries = state.app_settings.undo_max_entries as f64;
                let mut ev_e = ScrubEvent::default();
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Undo max steps")
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_PRIMARY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ev_e = cm3_scrub_number(
                            ui,
                            Id::unique("settings_undo_entries"),
                            &mut max_entries,
                            ScrubConfig::new(10.0, 2000.0, 10.0, 0),
                            theme,
                        );
                    });
                });

                let mut max_mib = state.app_settings.undo_max_mib as f64;
                let mut ev_b = ScrubEvent::default();
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Undo memory (MiB)")
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_PRIMARY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ev_b = cm3_scrub_number(
                            ui,
                            Id::unique("settings_undo_mib"),
                            &mut max_mib,
                            ScrubConfig::new(4.0, 1024.0, 8.0, 0),
                            theme,
                        );
                    });
                });

                if ev_e.committed || ev_b.committed {
                    state.app_settings.undo_max_entries =
                        max_entries.round().clamp(10.0, 2000.0) as usize;
                    state.app_settings.undo_max_mib = max_mib.round().clamp(4.0, 1024.0) as usize;
                    session.history.set_limits(
                        state.app_settings.undo_max_entries,
                        state.app_settings.undo_max_mib * 1024 * 1024,
                    );
                }

                let enabled_opt = session
                    .plugins
                    .enabled_ids()
                    .iter()
                    .filter(|id| id.as_str() != "core")
                    .count();
                let [rw, rh] = session.document().resolution;
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(format!(
                            "ZoidCraft {} · {}×{} @ {:.0} fps · {} plugins active · undo {}/{}",
                            env!("CARGO_PKG_VERSION"),
                            rw,
                            rh,
                            session.document().fps,
                            enabled_opt,
                            session.history.undo_len(),
                            session.history.max_entries()
                        ))
                        .font(theme.font(FONT_COMPACT))
                        .color(TEXT_OPTION_DESC),
                    )
                    .wrap(),
                );
            });
        });
}

// ---------------------------------------------------------------------------
// Tab 10: About (about)
// ---------------------------------------------------------------------------

pub(super) fn show_about_tab(ui: &mut Ui, session: &Session, theme: &Cm3Theme) {
    ScrollArea::vertical()
        .id_salt("about_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;
            panel_top_header(ui, "About", theme);

            Frame::NONE
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 4.0;
                    ui.label(
                        egui::RichText::new("ZoidCraft")
                            .font(theme.font(FONT_TITLE))
                            .color(TEXT_BRIGHT),
                    );
                    ui.label(
                        egui::RichText::new(format!("Version {}", env!("CARGO_PKG_VERSION")))
                            .font(theme.font(FONT_SMALL))
                            .color(theme.value()),
                    );
                    ui.add_space(6.0);
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(
                                "Native Rust + wgpu motion-graphics & 3D video editor \
                                 with modular built-in extensions.",
                            )
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_PRIMARY),
                        )
                        .wrap(),
                    );
                    ui.add_space(8.0);

                    let enabled_opt = session
                        .plugins
                        .enabled_ids()
                        .iter()
                        .filter(|id| id.as_str() != "core")
                        .count();
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(format!("Active optional plugins: {enabled_opt}"))
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_MUTED),
                        )
                        .wrap(),
                    );
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(format!(
                                "Undo depth: {}/{}",
                                session.history.undo_len(),
                                session.history.max_entries()
                            ))
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_MUTED),
                        )
                        .wrap(),
                    );
                    ui.add_space(8.0);

                    section_banner(ui, "Key Shortcuts", theme);
                    ui.add_space(4.0);
                    #[cfg(target_os = "macos")]
                    let shortcuts: [(&str, &str); 8] = [
                        ("Space", "Play / Pause"),
                        ("Home / End", "Jump to start / end"),
                        ("Left / Right", "Step -1 / +1 frame (Shift: +-5)"),
                        ("J / K / L", "Shuttle reverse / stop / forward"),
                        ("C / T", "Editing Camera / 2D Transform"),
                        ("Cmd+Z / Cmd+Y", "Undo / Redo"),
                        ("Cmd+G", "Open Graph Editor"),
                        ("Backquote (`)", "Maximize / restore focused panel"),
                    ];
                    #[cfg(not(target_os = "macos"))]
                    let shortcuts: [(&str, &str); 8] = [
                        ("Space", "Play / Pause"),
                        ("Home / End", "Jump to start / end"),
                        ("Left / Right", "Step -1 / +1 frame (Shift: +-5)"),
                        ("J / K / L", "Shuttle reverse / stop / forward"),
                        ("C / T", "Editing Camera / 2D Transform"),
                        ("Ctrl+Z / Ctrl+Y", "Undo / Redo"),
                        ("Ctrl+G", "Open Graph Editor"),
                        ("Backquote (`)", "Maximize / restore focused panel"),
                    ];
                    let avail_w = ui.available_width();
                    let key_col_w = 112.0_f32.min((avail_w * 0.42).max(84.0));
                    let desc_col_w = (avail_w - key_col_w - 8.0).max(80.0);
                    for (key, desc) in shortcuts {
                        ui.horizontal_top(|ui| {
                            ui.spacing_mut().item_spacing.x = 8.0;
                            ui.allocate_ui_with_layout(
                                Vec2::new(key_col_w, 0.0),
                                egui::Layout::top_down(egui::Align::LEFT),
                                |ui| {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(key)
                                                .font(theme.font(FONT_SMALL))
                                                .color(theme.value()),
                                        )
                                        .wrap(),
                                    );
                                },
                            );
                            ui.allocate_ui_with_layout(
                                Vec2::new(desc_col_w, 0.0),
                                egui::Layout::top_down(egui::Align::LEFT),
                                |ui| {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(desc)
                                                .font(theme.font(FONT_SMALL))
                                                .color(TEXT_PRIMARY),
                                        )
                                        .wrap(),
                                    );
                                },
                            );
                        });
                        ui.add_space(2.0);
                    }
                });
        });
}

pub(super) fn build_debug_info_report(session: &Session, app_settings: &AppSettings) -> String {
    let [rw, rh] = session.document().resolution;
    let enabled_plugins: Vec<String> = session
        .plugins
        .enabled_ids()
        .iter()
        .filter(|id| id.as_str() != "core")
        .cloned()
        .collect();
    format!(
        "ZoidCraft {}\nOS: {} ({})\nSequence: {}x{} @ {:.0} fps\nEnabled plugins ({}): {}\nUndo history: {}/{} (limit {} MiB)\nPreview cache: {} MiB\nFPS overlay: {}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        rw,
        rh,
        session.document().fps,
        enabled_plugins.len(),
        if enabled_plugins.is_empty() {
            "(none)".to_owned()
        } else {
            enabled_plugins.join(", ")
        },
        session.history.undo_len(),
        session.history.max_entries(),
        app_settings.undo_max_mib,
        app_settings.preview_cache_mib,
        if app_settings.show_fps_overlay {
            "on"
        } else {
            "off"
        },
    )
}

#[cfg(test)]
mod settings_layout_tests {
    use super::{EDITOR_LAYOUT_LABELS, EDITOR_LAYOUT_PRESETS, apply_editor_layout_preset};
    use crate::ui::panels::PanelsState;

    #[test]
    fn editor_layout_presets_apply_distinct_split_ratios_and_visibility() {
        assert_eq!(
            EDITOR_LAYOUT_LABELS,
            ["Classic", "Compact", "Wide Viewport", "Color & FX"]
        );
        assert_eq!(EDITOR_LAYOUT_PRESETS.len(), 4);

        let mut state = PanelsState::default();
        let wide = apply_editor_layout_preset(2, &mut state);
        assert_eq!(wide.name, "Wide Viewport");
        assert_eq!(state.settings_editor_layout, 2);
        assert!(!state.objects_list_open);
        assert!((state.objects_split.ratio - wide.objects_split_ratio).abs() < 1e-6);
        assert!((state.effects_split.ratio - wide.effects_split_ratio).abs() < 1e-6);
        assert!((state.prop_columns.label_ratio - wide.prop_label_ratio).abs() < 1e-6);

        let color_fx = apply_editor_layout_preset(3, &mut state);
        assert_eq!(color_fx.name, "Color & FX");
        assert!(state.objects_list_open);
        assert!(color_fx.effects_split_ratio > wide.effects_split_ratio);

        // Out-of-range index clamps cleanly to last preset
        let clamped = apply_editor_layout_preset(99, &mut state);
        assert_eq!(clamped.name, "Color & FX");
        assert_eq!(state.settings_editor_layout, 3);
    }
}
