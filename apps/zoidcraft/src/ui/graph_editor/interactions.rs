//! Graph editor window, channel list, ruler, drag handling and hit testing.

use crate::ui::gestures::{ScrollDevice, unoccluded_hover_pos};
use crate::ui::theme::{Cm3Icon, paint_cm3_icon};
use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, PointerButton, Pos2, Rect, ScrollArea,
    Sense, Shape, Stroke, StrokeKind, Ui, Window, pos2, vec2,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use zc_anim::{EasingKind, eval_easing};
use zc_color::{Hsl, Rgb, ThemePalette};
use zc_session::{Command, Session};

use super::*;

// ---------------------------------------------------------------------------
// Public Entry Points (Floating Window & Docked Panel)
// ---------------------------------------------------------------------------

/// Renders the Graph Editor inside a floating, resizable `egui::Window` (Spec 02 §4).
pub fn show_graph_editor_window(
    ctx: &egui::Context,
    session: &mut Session,
    state: &mut GraphEditorState,
    palette: &ThemePalette,
) -> GraphEditorAction {
    if !state.open {
        state.cancel_active_interaction(session);
        return GraphEditorAction::default();
    }
    let graph_plus = session.plugins.is_enabled("graph-plus");
    let title = if graph_plus {
        "Graph Editor 2"
    } else {
        "Graph Editor"
    };
    let mut open = state.open;
    let mut action = GraphEditorAction::default();
    let mut body_shown = false;
    let win_id = egui::Id::NULL.with("zoidcraft_graph_editor_window");
    let border_color = if state.window_focused {
        rgb_color(palette.focus)
    } else {
        COLOR_WINDOW_BORDER
    };

    let screen = ctx.input(|i| i.viewport_rect());
    let default_w =
        (if graph_plus { 820.0_f32 } else { 640.0_f32 }).min((screen.width() - 48.0).max(480.0));
    let default_h =
        (if graph_plus { 520.0_f32 } else { 400.0_f32 }).min((screen.height() - 96.0).max(300.0));
    // Spawn centered over the viewport region (right of the ~300px sidebar, clear of top HUD),
    // clamped cleanly inside screen bounds instead of overlapping the sidebar at (16, 46).
    let viewport_center_x = (screen.left() + 300.0 + screen.right()) * 0.5;
    let spawn_x = (viewport_center_x - default_w * 0.5)
        .max(screen.left() + 24.0)
        .min((screen.right() - default_w - 16.0).max(screen.left() + 16.0))
        .round();
    let spawn_y = (screen.top() + 72.0)
        .min((screen.bottom() - default_h - 24.0).max(screen.top() + 32.0))
        .round();

    let mut window = Window::new(title)
        .id(win_id)
        .title_bar(false)
        .movable(!state.maximized)
        .default_pos(pos2(spawn_x, spawn_y))
        .default_size(vec2(default_w, default_h))
        .min_size(if state.collapsed {
            vec2(260.0, 30.0)
        } else {
            vec2(480.0, 300.0)
        })
        .resizable(!state.collapsed && !state.maximized)
        .collapsible(false)
        .frame(
            egui::Frame::NONE
                .fill(COLOR_BG)
                .corner_radius(CornerRadius::ZERO)
                .stroke(Stroke::new(1.0, border_color))
                .inner_margin(egui::Margin::ZERO),
        );

    if state.maximized {
        let max_rect = ctx.input(|i| i.viewport_rect()).shrink(4.0);
        window = window
            .fixed_pos(max_rect.min)
            .fixed_size(if state.collapsed {
                vec2(max_rect.width(), 30.0)
            } else {
                max_rect.size()
            });
    } else if state.collapsed {
        window = window.fixed_size(vec2(320.0, 30.0));
    }

    window.show(ctx, |ui| {
        let area_layer_id = ui.layer_id();
        if ctx.input(|i| i.pointer.primary_pressed())
            && let Some(pos) = ctx.input(|i| i.pointer.interact_pos())
        {
            state.window_focused = ctx.layer_id_at(pos) == Some(area_layer_id);
        }

        // Sharp-cornered CM3 titlebar: --zui-bg-title-focused when focused, plus Collapse / Maximize / Close.
        ui.spacing_mut().item_spacing = vec2(0.0, 0.0);
        let (title_rect, _) =
            ui.allocate_exact_size(vec2(ui.available_width(), 30.0), Sense::hover());
        let title_bg = if state.window_focused {
            rgb_color(palette.bg_title_focused())
        } else {
            COLOR_BG
        };
        ui.painter()
            .rect_filled(title_rect, CornerRadius::ZERO, title_bg);
        ui.painter().rect_filled(
            Rect::from_min_size(
                pos2(title_rect.left(), (title_rect.bottom() - 1.0).round()),
                vec2(title_rect.width(), 1.0),
            ),
            CornerRadius::ZERO,
            COLOR_BORDER,
        );
        ui.painter().text(
            pos2(title_rect.left() + 9.0, title_rect.center().y),
            Align2::LEFT_CENTER,
            title,
            FontId::monospace(16.0),
            COLOR_TITLE_TEXT,
        );

        let title_btn =
            |ui: &mut Ui, rect: Rect, salt: &str, icon: Cm3Icon, tooltip: &str| -> bool {
                let resp = ui
                    .interact(rect, win_id.with(salt), Sense::click())
                    .on_hover_cursor(CursorIcon::PointingHand)
                    .on_hover_text(tooltip);
                if resp.hovered() {
                    ui.painter().rect_filled(
                        rect,
                        CornerRadius::ZERO,
                        Color32::from_rgb(0x33, 0x33, 0x33),
                    );
                }
                let ic_rect = Rect::from_center_size(rect.center(), vec2(12.0, 12.0));
                paint_cm3_icon(
                    ui.painter(),
                    ic_rect,
                    icon,
                    if resp.hovered() {
                        COLOR_TEXT_BRIGHT
                    } else {
                        Color32::from_rgb(0x9a, 0x9a, 0x9a)
                    },
                );
                resp.clicked()
            };

        let collapse_rect = Rect::from_min_size(
            pos2(title_rect.right() - 80.0, title_rect.top() + 2.0),
            vec2(26.0, 26.0),
        );
        if title_btn(
            ui,
            collapse_rect,
            "collapse",
            Cm3Icon::Collapse,
            if state.collapsed {
                "Expand Graph Editor"
            } else {
                "Collapse Graph Editor"
            },
        ) {
            state.collapsed = !state.collapsed;
        }

        let max_rect = Rect::from_min_size(
            pos2(title_rect.right() - 54.0, title_rect.top() + 2.0),
            vec2(26.0, 26.0),
        );
        if title_btn(
            ui,
            max_rect,
            "maximize",
            Cm3Icon::Maximize,
            if state.maximized {
                "Restore Graph Editor"
            } else {
                "Maximize Graph Editor"
            },
        ) {
            state.maximized = !state.maximized;
            if state.maximized {
                state.collapsed = false;
            }
        }

        let close_rect = Rect::from_min_size(
            pos2(title_rect.right() - 28.0, title_rect.top() + 2.0),
            vec2(26.0, 26.0),
        );
        if title_btn(
            ui,
            close_rect,
            "close",
            Cm3Icon::Close,
            "Close Graph Editor (Esc)",
        ) {
            open = false;
        }

        if !state.collapsed {
            ui.spacing_mut().item_spacing = vec2(5.0, 3.0);
            body_shown = true;
            action = show_graph_editor_panel(ui, session, state, palette);
        }
    });

    state.open = open;
    let primary_down = ctx.input(|i| i.pointer.primary_down());
    let has_primary_drag = matches!(
        state.drag_state,
        GraphDragState::DraggingKeys { .. }
            | GraphDragState::DraggingHandle { .. }
            | GraphDragState::DraggingSpeedHandle { .. }
            | GraphDragState::DraggingEasingPad { .. }
            | GraphDragState::Marquee { .. }
    );
    if !state.open || !body_shown || (!primary_down && has_primary_drag) {
        state.cancel_active_interaction(session);
    }
    action
}

/// Renders the dual-mode Graph Editor directly inside any docked panel or window `Ui`.
pub fn show_graph_editor_panel(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut GraphEditorState,
    palette: &ThemePalette,
) -> GraphEditorAction {
    let mut action = GraphEditorAction::default();
    let graph_plus = session.plugins.is_enabled("graph-plus");
    let easing_plus = session.plugins.is_enabled("easing-plus");
    if !easing_plus && matches!(state.drag_state, GraphDragState::DraggingEasingPad { .. }) {
        state.cancel_active_interaction(session);
    }
    let tracks = collect_curve_tracks(session, state, graph_plus);
    prune_stale_selected_keys(session, state);
    if let Some(ref solo) = state.solo_track
        && !tracks.iter().any(|t| &t.track_key == solo)
    {
        state.solo_track = None;
    }
    if !matches!(
        state.drag_state,
        GraphDragState::DraggingHandle { .. } | GraphDragState::DraggingSpeedHandle { .. }
    ) {
        sync_handles_from_tracks(&mut state.handles, &tracks);
    }

    let full_rect = ui.available_rect_before_wrap();
    ui.painter()
        .rect_filled(full_rect, CornerRadius::ZERO, COLOR_BG);

    let est_canvas_size = vec2(
        (full_rect.width() - TRACK_LIST_WIDTH).max(200.0),
        (full_rect.height() - 150.0).max(160.0),
    );
    let fps = session.document().fps.max(1.0);

    // Top toolbar: Graph Editor 2 big preset tiles (68x54) + Custom easings (50x30) + tool tiles (38x32),
    // or compact vanilla CM3 toolbar strip.
    if graph_plus {
        show_g2_toolbar(
            ui,
            session,
            state,
            &tracks,
            full_rect.width(),
            est_canvas_size,
            palette,
            &mut action,
        );
    } else {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(3.0, 3.0);
            ui.spacing_mut().button_padding = vec2(4.0, 2.0);
            if show_plot_mode_toggle(ui, state, palette) {
                let doc = session.document().clone();
                state.fit_curves_for_mode(&tracks, &doc, fps, est_canvas_size);
            }
            ui.separator();
            if toolbar_chip(ui, "Hold", false, palette) {
                apply_tween_to_selected(session, state, &tracks, 0, &mut action);
            }
            if toolbar_chip(ui, "Linear", false, palette) {
                apply_tween_to_selected(session, state, &tracks, 1, &mut action);
            }
            if toolbar_chip(ui, "Ease", false, palette) {
                apply_tween_to_selected(session, state, &tracks, 7, &mut action);
            }
            if toolbar_chip(ui, "Bezier", false, palette) {
                toggle_bezier_on_selected(session, state, &tracks, &mut action);
            }
            if toolbar_chip(ui, "Link/Break", false, palette) {
                toggle_continuous_on_selected(session, state, &mut action);
            }
            if toolbar_chip(ui, "Flat", false, palette) {
                reset_handles_on_selected(session, state, &mut action);
            }
            ui.separator();
            if toolbar_chip(ui, "Fit (\\)", false, palette) {
                let doc = session.document().clone();
                state.fit_curves_for_mode(&tracks, &doc, fps, est_canvas_size);
            }
            if toolbar_chip(ui, "Snap", state.snap_enabled, palette) {
                state.snap_enabled = !state.snap_enabled;
            }
            if state.focused_property.is_some() {
                let follow_label = if state.follow_selection {
                    "Follow"
                } else {
                    "Show All"
                };
                if toolbar_chip(ui, follow_label, state.follow_selection, palette) {
                    state.follow_selection = !state.follow_selection;
                    if state.follow_selection {
                        state.focused_property = None;
                    }
                }
            }
            if toolbar_chip(ui, "33 Easings", state.show_all_easings_popup, palette) {
                state.show_all_easings_popup = !state.show_all_easings_popup;
            }
            ui.separator();
            if toolbar_chip(ui, "Copy Curve JSON", false, palette) {
                let _ = copy_curve_track_as_pz_json(ui.ctx(), session, state, &tracks);
            }
            if toolbar_chip(ui, "Paste Curve JSON", false, palette)
                && let Some(sys_text) = crate::ui::clipboard::read_system_clipboard()
            {
                let _ =
                    try_paste_pz_keyframes_json(session, state, &tracks, &sys_text, &mut action);
            }
        });
    }

    ui.separator();

    // Easing preset strip (CM3 33 easings when expanded).
    if state.show_all_easings_popup {
        show_cm3_33_easings_strip(ui, session, state, &tracks, palette, &mut action);
        ui.separator();
    }

    // Reserve bottom space for Easing+ panel (if enabled) and Graph Editor 2 status bar.
    let status_h = if graph_plus { STATUS_BAR_HEIGHT } else { 0.0 };
    let easing_plus_h = if easing_plus { 188.0 } else { 0.0 };
    let body_avail = ui.available_rect_before_wrap();
    let main_h = (body_avail.height() - easing_plus_h - status_h).max(120.0);
    let main_rect = Rect::from_min_size(body_avail.min, vec2(body_avail.width(), main_h));
    let _ = ui.allocate_rect(main_rect, Sense::hover());

    // Split left channel track list & right 2D curve canvas (Vanilla CM3 uses 33% left width).
    let left_w = if graph_plus {
        TRACK_LIST_WIDTH.min((main_rect.width() * 0.35).max(0.0))
    } else {
        let max_w = (main_rect.width() * 0.5).max(0.0);
        (main_rect.width() * 0.33).clamp(110.0_f32.min(max_w), max_w)
    };
    let track_list_rect = Rect::from_min_size(main_rect.min, vec2(left_w, main_rect.height()));
    let right_rect = Rect::from_min_max(
        pos2(track_list_rect.right(), main_rect.top()),
        main_rect.max,
    );

    show_channel_track_list(ui, session, state, &tracks, track_list_rect, graph_plus);

    let ruler_rect =
        Rect::from_min_size(right_rect.min, vec2(right_rect.width(), TOP_RULER_HEIGHT));
    let canvas_rect =
        Rect::from_min_max(pos2(right_rect.left(), ruler_rect.bottom()), right_rect.max);

    // Graph Editor keyboard shortcuts & Cmd/Ctrl+C/V clipboard when focused/hovered and no text input has focus.
    let graph_hovered = unoccluded_hover_pos(ui, full_rect).is_some();
    let canvas_hovered = unoccluded_hover_pos(ui, main_rect).is_some();
    if !ui.ctx().egui_wants_keyboard_input() && (state.window_focused || graph_hovered) {
        let (has_copy, has_paste_key, pasted_text) = ui.input(|i| {
            let cmd = i.modifiers.command || i.modifiers.ctrl;
            let mut ev_copy = false;
            let mut ev_paste: Option<String> = None;
            for ev in &i.events {
                match ev {
                    egui::Event::Copy => ev_copy = true,
                    egui::Event::Paste(text) => ev_paste = Some(text.clone()),
                    _ => {}
                }
            }
            (
                ev_copy || (cmd && i.key_pressed(egui::Key::C)),
                cmd && i.key_pressed(egui::Key::V),
                ev_paste,
            )
        });

        if has_copy {
            if !state.selected_keys.is_empty() {
                copy_selected_graph_keys(state, &tracks);
                let _ = copy_curve_track_as_pz_json(ui.ctx(), session, state, &tracks);
            }
            if easing_plus || state.selected_keys.is_empty() {
                let [x1, y1, x2, y2] = state.easing_plus_curve;
                let css = format!("cubic-bezier({x1:.4}, {y1:.4}, {x2:.4}, {y2:.4})");
                state.easing_clipboard = Some(css.clone());
                if state.selected_keys.is_empty()
                    && (easing_plus
                        || copy_curve_track_as_pz_json(ui.ctx(), session, state, &tracks).is_none())
                {
                    crate::ui::clipboard::write_system_clipboard(ui.ctx(), &css);
                }
            }
        }

        if pasted_text.is_some() || has_paste_key {
            let effective_text = pasted_text.or_else(crate::ui::clipboard::read_system_clipboard);
            let _ = handle_graph_paste_text(
                session,
                state,
                &tracks,
                effective_text.as_deref(),
                &mut action,
            );
        }
    }

    if !ui.ctx().egui_wants_keyboard_input() && canvas_hovered {
        let (k_f, k_bslash, k_h, k_v, k_s, k_i, k_t, k_r, k_k, k_n, k_del) = ui.input(|i| {
            let no_mod = !i.modifiers.ctrl && !i.modifiers.command && !i.modifiers.alt;
            (
                no_mod && i.key_pressed(egui::Key::F),
                no_mod && i.key_pressed(egui::Key::Backslash),
                no_mod && i.key_pressed(egui::Key::H),
                no_mod && i.key_pressed(egui::Key::V),
                no_mod && i.key_pressed(egui::Key::S),
                no_mod && i.key_pressed(egui::Key::I),
                no_mod && i.key_pressed(egui::Key::T),
                no_mod && i.key_pressed(egui::Key::R),
                no_mod && i.key_pressed(egui::Key::K),
                no_mod && i.key_pressed(egui::Key::N),
                no_mod && (i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)),
            )
        });
        if k_f {
            state.frame_selected_keys(&tracks, canvas_rect.size());
            if state.plot_mode == GraphPlotMode::Speed {
                let doc = session.document().clone();
                state.fit_speed_y(&tracks, &doc, fps, canvas_rect.size());
            }
        } else if k_bslash {
            let doc = session.document().clone();
            state.fit_curves_for_mode(&tracks, &doc, fps, canvas_rect.size());
        } else if k_h && graph_plus {
            state.pan_tool = true;
        } else if k_v && graph_plus {
            state.pan_tool = false;
        } else if k_s {
            state.snap_enabled = !state.snap_enabled;
        } else if k_i {
            toggle_bezier_on_selected(session, state, &tracks, &mut action);
        } else if k_t {
            toggle_continuous_on_selected(session, state, &mut action);
        } else if k_r {
            reset_handles_on_selected(session, state, &mut action);
        } else if k_k && graph_plus {
            insert_key_at_playhead(session, state, &tracks, &mut action);
        } else if k_n && graph_plus {
            state.normalize_curves = !state.normalize_curves;
        } else if k_del && !state.selected_keys.is_empty() {
            delete_selected_graph_keys(session, state, &mut action);
        }
    }

    show_graph_ruler(ui, session, state, ruler_rect, graph_plus, &mut action);
    show_curve_canvas(
        ui,
        session,
        state,
        &tracks,
        canvas_rect,
        palette,
        graph_plus,
        &mut action,
    );

    // Optional Easing+ bottom workstation (flexible curve pad + 2x2 inputs + 190 px presets grid).
    let mut next_top = main_rect.bottom();
    if easing_plus {
        let pad_rect = Rect::from_min_size(
            pos2(body_avail.left(), next_top + 4.0),
            vec2(body_avail.width(), (easing_plus_h - 4.0).max(120.0)),
        );
        show_easing_plus_panel(ui, session, state, &tracks, pad_rect, palette, &mut action);
        next_top += easing_plus_h;
    }

    // Graph Editor 2 bottom `.g2-status` bar.
    if graph_plus {
        let status_rect = Rect::from_min_max(
            pos2(
                body_avail.left(),
                next_top.min(body_avail.bottom() - status_h),
            ),
            body_avail.max,
        );
        show_g2_status_bar(ui, session, state, &tracks, status_rect, palette);
    }

    action
}

// ---------------------------------------------------------------------------
// Left Channel / Property Track List
// ---------------------------------------------------------------------------

/// Abbreviates common property tokens (e.g. `.position` -> `.pos`, `.rotation` -> `.rot`)
/// before middle-eliding so channels like `Camera.position.X` and `Camera.rotation.X` remain
/// cleanly distinguishable even in narrow Speed Mode legends.
#[must_use]
pub fn elide_channel_name(name: &str, max_chars: usize) -> String {
    if name.chars().count() <= max_chars {
        return name.to_owned();
    }

    // Step 1: abbreviate common dot-separated property segments while keeping axis suffixes intact.
    let mut parts: Vec<String> = name
        .split('.')
        .map(|seg| match seg.to_ascii_lowercase().as_str() {
            "position" => "pos".to_owned(),
            "rotation" => "rot".to_owned(),
            "scale" => "scl".to_owned(),
            "opacity" => "op".to_owned(),
            "amount" => "amt".to_owned(),
            "intensity" => "int".to_owned(),
            "strength" => "str".to_owned(),
            "radius" => "rad".to_owned(),
            "anchor" => "anc".to_owned(),
            _ => seg.to_owned(),
        })
        .collect();
    let mut candidate = parts.join(".");
    if candidate.chars().count() <= max_chars {
        return candidate;
    }

    // Step 2: if still too long and prefixed with `Camera.`, shorten `Camera` -> `Cam`.
    if parts.len() >= 2 && parts[0].eq_ignore_ascii_case("camera") {
        parts[0] = "Cam".to_owned();
        candidate = parts.join(".");
        if candidate.chars().count() <= max_chars {
            return candidate;
        }
    }

    let chars: Vec<char> = candidate.chars().collect();
    if max_chars <= 3 {
        return chars[chars.len().saturating_sub(max_chars)..]
            .iter()
            .collect();
    }
    // Preserve trailing axis suffix (`.X`, `.Y`, `.Z`, `.W`, `.R`, `.G`, `.B`, `.A`) plus the
    // abbreviated property token immediately before it (e.g. `pos.X`, `rot.X`).
    let suffix_len = if chars.len() >= 2 && chars[chars.len() - 2] == '.' {
        let ideal_tail = 5.min(max_chars.saturating_sub(2));
        ideal_tail.max(2).min(chars.len())
    } else {
        (max_chars / 2).clamp(2.min(chars.len()), chars.len())
    };
    let head_len = max_chars.saturating_sub(suffix_len + 2);
    if head_len == 0 {
        let keep = max_chars.saturating_sub(2);
        let tail: String = chars[chars.len().saturating_sub(keep)..].iter().collect();
        return format!("..{tail}");
    }
    let head: String = chars[..head_len].iter().collect();
    let tail: String = chars[chars.len() - suffix_len..].iter().collect();
    format!("{head}..{tail}")
}

pub fn show_channel_track_list(
    ui: &mut Ui,
    session: &Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    rect: Rect,
    graph_plus: bool,
) {
    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, COLOR_TRACK_LIST_BG);
    ui.painter().rect_filled(
        Rect::from_min_size(
            pos2((rect.right() - 1.0).round(), rect.top()),
            vec2(1.0, rect.height()),
        ),
        CornerRadius::ZERO,
        COLOR_BORDER,
    );

    ui.scope_builder(egui::UiBuilder::new().max_rect(rect.shrink(4.0)), |ui| {
        ScrollArea::vertical()
            .id_salt("graph_editor_track_list")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if tracks.is_empty() {
                    ui.colored_label(COLOR_LABEL_MUTED, "No animated curves");
                    return;
                }
                // Reserve 12 px right-side gutter so the vertical scrollbar never overlaps
                // or intercepts clicks on Solo (`O`) buttons.
                const SCROLLBAR_GUTTER_PX: f32 = 12.0;
                for track in tracks {
                    let visible = state.is_track_visible(&track.track_key);
                    let soloed = state.solo_track.as_deref() == Some(track.track_key.as_str());
                    let local_f = session.playback.current_frame - track.frame_offset;
                    let fps = session.document().fps.max(1.0);
                    let live_val = match state.plot_mode {
                        GraphPlotMode::Value => eval_track_live_value(
                            track,
                            session.document(),
                            &state.handles,
                            local_f,
                        ),
                        GraphPlotMode::Speed => {
                            if track.is_expression {
                                let eps = 0.05;
                                let v0 = eval_track_live_value(
                                    track,
                                    session.document(),
                                    &state.handles,
                                    local_f - eps,
                                );
                                let v1 = eval_track_live_value(
                                    track,
                                    session.document(),
                                    &state.handles,
                                    local_f + eps,
                                );
                                ((v1 - v0) / (2.0 * eps)) * fps
                            } else {
                                eval_curve_velocity(track, local_f, fps, &state.handles)
                            }
                        }
                    };

                    // Row order: [swatch] [fx?] [name] [value] [Eye] [Solo] [scrollbar gutter].
                    let row_w = (ui.available_width() - SCROLLBAR_GUTTER_PX).max(40.0);
                    let (row_rect, _) = ui.allocate_exact_size(vec2(row_w, 20.0), Sense::hover());
                    let p = ui.painter();
                    let cy = row_rect.center().y;

                    // 1. 12 x 4 px channel color swatch.
                    let bar =
                        Rect::from_center_size(pos2(row_rect.left() + 7.0, cy), vec2(12.0, 4.0));
                    p.rect_filled(bar, CornerRadius::same(1), track.color);

                    // Allocate right-hand buttons from right edge inward: [Solo] [Eye] [value].
                    let mut right_cursor = row_rect.right();

                    // Solo button rendered on ALL tracks (both vector axes and scalar properties).
                    right_cursor -= 16.0;
                    let solo_rect =
                        Rect::from_center_size(pos2(right_cursor + 8.0, cy), vec2(16.0, 16.0));
                    let solo_resp = ui
                        .interact(
                            solo_rect,
                            ui.scope_id().with(("g_solo", &track.track_key)),
                            Sense::click(),
                        )
                        .on_hover_cursor(CursorIcon::PointingHand)
                        .on_hover_text(if soloed { "Unsolo curve" } else { "Solo curve" });
                    if solo_resp.hovered() {
                        p.rect_filled(
                            solo_rect,
                            CornerRadius::ZERO,
                            Color32::from_rgb(0x36, 0x36, 0x36),
                        );
                    }
                    let solo_fg = if soloed {
                        Color32::from_rgb(0xd4, 0x48, 0x48)
                    } else if solo_resp.hovered() {
                        COLOR_TEXT_BRIGHT
                    } else {
                        Color32::from_rgb(0xac, 0xac, 0xac)
                    };
                    paint_cm3_icon(p, solo_rect.shrink(2.5), Cm3Icon::SoloCircle, solo_fg);
                    if soloed {
                        p.circle_filled(solo_rect.center(), 2.0, solo_fg);
                    }
                    if solo_resp.clicked() {
                        state.solo_track = if soloed {
                            None
                        } else {
                            Some(track.track_key.clone())
                        };
                    }
                    right_cursor -= 2.0;

                    // Eye visibility toggle.
                    right_cursor -= 16.0;
                    let eye_rect =
                        Rect::from_center_size(pos2(right_cursor + 8.0, cy), vec2(16.0, 16.0));
                    let eye_resp = ui
                        .interact(
                            eye_rect,
                            ui.scope_id().with(("g_eye", &track.track_key)),
                            Sense::click(),
                        )
                        .on_hover_cursor(CursorIcon::PointingHand)
                        .on_hover_text(if visible { "Hide curve" } else { "Show curve" });
                    if eye_resp.hovered() {
                        p.rect_filled(
                            eye_rect,
                            CornerRadius::ZERO,
                            Color32::from_rgb(0x36, 0x36, 0x36),
                        );
                    }
                    let eye_fg = if !visible {
                        Color32::from_rgb(0x66, 0x66, 0x66)
                    } else if eye_resp.hovered() {
                        COLOR_TEXT_BRIGHT
                    } else {
                        Color32::from_rgb(0xac, 0xac, 0xac)
                    };
                    paint_cm3_icon(p, eye_rect.shrink(1.5), Cm3Icon::Eye, eye_fg);
                    if !visible {
                        p.line_segment(
                            [
                                pos2(eye_rect.left() + 2.5, eye_rect.bottom() - 2.5),
                                pos2(eye_rect.right() - 2.5, eye_rect.top() + 2.5),
                            ],
                            Stroke::new(1.3, Color32::from_rgb(0xa9, 0x24, 0x24)),
                        );
                    }
                    if eye_resp.clicked() {
                        if state.hidden_tracks.contains(&track.track_key) {
                            state.hidden_tracks.remove(&track.track_key);
                        } else {
                            state.hidden_tracks.insert(track.track_key.clone());
                        }
                    }
                    right_cursor -= 4.0;

                    // Live value readout (Graph Editor 2 or Speed mode) positioned left of Eye/Solo
                    // with clear horizontal separation from the channel name.
                    if graph_plus || state.plot_mode == GraphPlotMode::Speed {
                        let val_str = match state.plot_mode {
                            GraphPlotMode::Value => format!("{live_val:.2}"),
                            GraphPlotMode::Speed => format!("{live_val:.1} u/s"),
                        };
                        let val_w = (val_str.len() as f32 * 6.3 + 4.0).clamp(36.0, 74.0);
                        let val_left = (right_cursor - val_w).max(row_rect.left() + 64.0);
                        let val_rect = Rect::from_min_max(
                            pos2(val_left, row_rect.top()),
                            pos2(right_cursor, row_rect.bottom()),
                        );
                        ui.painter().with_clip_rect(val_rect).text(
                            pos2(val_rect.right(), cy),
                            Align2::RIGHT_CENTER,
                            val_str,
                            FontId::monospace(10.0),
                            Color32::from_rgb(0x7e, 0x8f, 0xb9),
                        );
                        right_cursor = val_rect.left() - 6.0;
                    }

                    // Expression `fx` badge when this track is driven by an expression.
                    let mut name_left = row_rect.left() + 17.0;
                    if track.is_expression {
                        let fx_rect =
                            Rect::from_center_size(pos2(name_left + 10.0, cy), vec2(18.0, 13.0));
                        p.rect(
                            fx_rect,
                            CornerRadius::same(2),
                            Color32::from_rgb(0x3a, 0x2d, 0x52),
                            Stroke::new(1.0, Color32::from_rgb(0xc7, 0x92, 0xea)),
                            StrokeKind::Inside,
                        );
                        p.text(
                            fx_rect.center(),
                            Align2::CENTER_CENTER,
                            "fx",
                            FontId::monospace(9.0),
                            Color32::from_rgb(0xe0, 0xc0, 0xff),
                        );
                        if let Some(ref expr_src) = track.expression {
                            let _ = ui
                                .interact(
                                    fx_rect,
                                    ui.scope_id().with(("g_fx_badge", &track.track_key)),
                                    Sense::hover(),
                                )
                                .on_hover_text(format!("Expression:\n{expr_src}"));
                        }
                        name_left = fx_rect.right() + 4.0;
                    }

                    // Middle-elided channel / property name preserving `.X`/`.Y`/`.Z` suffixes.
                    let name_rect = Rect::from_min_max(
                        pos2(name_left, row_rect.top()),
                        pos2(right_cursor.max(name_left), row_rect.bottom()),
                    );
                    let max_chars = ((name_rect.width() / 6.8).floor() as usize).max(3);
                    let shown_name = elide_channel_name(&track.display_name, max_chars);
                    let label_color = if visible {
                        COLOR_TEXT
                    } else {
                        COLOR_LABEL_MUTED
                    };
                    ui.painter().with_clip_rect(name_rect).text(
                        pos2(name_rect.left(), cy),
                        Align2::LEFT_CENTER,
                        &shown_name,
                        FontId::monospace(11.0),
                        label_color,
                    );
                    if shown_name != track.display_name {
                        let _ = ui
                            .interact(
                                name_rect,
                                ui.scope_id().with(("g_name_tip", &track.track_key)),
                                Sense::hover(),
                            )
                            .on_hover_text(&track.display_name);
                    }
                }
            });
    });
}

// ---------------------------------------------------------------------------
// Top Frame Ruler (Scrubbing Synced with ViewX / ZoomX)
// ---------------------------------------------------------------------------

pub fn handle_graph_wheel_and_pinch(
    ui: &Ui,
    state: &mut GraphEditorState,
    rect: Rect,
    hovered: bool,
    is_ruler: bool,
) {
    if !hovered {
        return;
    }
    let Some(anchor) = unoccluded_hover_pos(ui, rect) else {
        return;
    };
    let g = ui.input(|i| state.gesture_classifier.sample(i));

    // Note: `zoom_x` and `zoom_y` are units per CSS pixel (smaller = zoomed IN).
    // Dividing by `fx` / `fy` (> 1.0 = zoom in) shrinks units/px while preserving `anchor`.
    let apply_zoom = |state: &mut GraphEditorState, fx: f64, fy: f64| {
        let anchor_f = state.screen_x_to_frame(anchor.x, rect);
        let anchor_v = state.screen_y_to_value(anchor.y, rect);
        if (fx - 1.0).abs() > 1e-6 {
            state.zoom_x = (state.zoom_x / fx).clamp(1e-4, 500.0);
            state.center_x = anchor_f - f64::from(anchor.x - rect.center().x) * state.zoom_x;
        }
        if !is_ruler && (fy - 1.0).abs() > 1e-6 {
            state.zoom_y = (state.zoom_y / fy).clamp(1e-6, 10000.0);
            state.center_y = anchor_v + f64::from(anchor.y - rect.center().y) * state.zoom_y;
        }
    };

    // 1. True trackpad pinch (`Event::Zoom` / multi-touch).
    if g.has_pinch() {
        let z = f64::from(g.pinch_zoom).clamp(0.5, 2.0);
        if is_ruler {
            apply_zoom(state, z, 1.0);
        } else if g.modifiers.alt {
            apply_zoom(state, 1.0, z);
        } else {
            apply_zoom(state, z, z);
        }
        return;
    }

    // 2. Ctrl/Cmd + Wheel (egui 0.37 folds this into `zoom_delta()` and zeroes `smooth_scroll_delta`).
    if g.has_mod_wheel_zoom() {
        let z = f64::from(g.mod_wheel_zoom).clamp(0.5, 2.0);
        if is_ruler {
            apply_zoom(state, z, 1.0);
        } else if g.modifiers.alt {
            apply_zoom(state, 1.0, z);
        } else {
            apply_zoom(state, z, z);
        }
        return;
    }

    // 3. Shift + Wheel -> pan X only.
    if g.modifiers.shift && g.has_scroll() {
        state.center_x -= f64::from(g.scroll_delta.x + g.scroll_delta.y) * state.zoom_x;
        return;
    }

    // 4. Alt + Wheel -> zoom Y (or zoom X on the ruler).
    if g.modifiers.alt && (g.scroll_delta.y.abs() > 0.01 || g.raw_wheel_points.y.abs() > 0.01) {
        let dy = if g.scroll_delta.y.abs() > 0.01 {
            g.scroll_delta.y
        } else {
            g.raw_wheel_points.y
        };
        let z = (f64::from(dy) * 0.0025).clamp(-0.4, 0.4).exp();
        if is_ruler {
            apply_zoom(state, z, 1.0);
        } else {
            apply_zoom(state, 1.0, z);
        }
        return;
    }

    if !g.has_scroll() {
        return;
    }

    // 5. Trackpad two-finger scroll -> smooth 2D pan (X only on the ruler).
    if g.device == ScrollDevice::Trackpad {
        state.center_x -= f64::from(g.scroll_delta.x) * state.zoom_x;
        if !is_ruler {
            state.center_y += f64::from(g.scroll_delta.y) * state.zoom_y;
        }
        return;
    }

    // 6. Discrete mouse wheel -> horizontal tilt pans X, vertical notches zoom X around pointer.
    if g.scroll_delta.x.abs() > 0.01 {
        state.center_x -= f64::from(g.scroll_delta.x) * state.zoom_x;
    }
    if g.scroll_delta.y.abs() > 0.01 {
        let z = (f64::from(g.scroll_delta.y) * 0.0025)
            .clamp(-0.4, 0.4)
            .exp();
        apply_zoom(state, z, 1.0);
    }
}

pub fn show_graph_ruler(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut GraphEditorState,
    rect: Rect,
    graph_plus: bool,
    action: &mut GraphEditorAction,
) {
    let resp = ui.interact(
        rect,
        ui.scope_id().with("graph_editor_ruler"),
        Sense::click_and_drag(),
    );
    if resp.hovered() {
        ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
    }
    handle_graph_wheel_and_pinch(ui, state, rect, resp.hovered(), true);

    if matches!(state.drag_state, GraphDragState::None)
        && (resp.drag_started_by(PointerButton::Primary)
            || resp.drag_started_by(PointerButton::Secondary)
            || resp.clicked_by(PointerButton::Primary))
    {
        state.drag_state = GraphDragState::ScrubbingPlayhead;
    }

    if matches!(state.drag_state, GraphDragState::ScrubbingPlayhead) {
        if let Some(pos) = ui.input(|i| i.pointer.latest_pos()) {
            let max_f = (session.document().sequence_length - 1.0).max(0.0);
            let target = state
                .screen_x_to_frame(pos.x, rect)
                .round()
                .clamp(0.0, max_f);
            let _ = session.execute(Command::SetFrame { frame: target });
            action.playhead_scrubbed = true;
        }
        if !ui.input(|i| i.pointer.primary_down() || i.pointer.secondary_down()) {
            state.drag_state = GraphDragState::None;
        }
    }

    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, CornerRadius::ZERO, COLOR_RULER_BG);

    let (major_x, minor_x) = adaptive_grid_step(state.zoom_x);
    let min_f = state.screen_x_to_frame(rect.left(), rect);
    let max_f = state.screen_x_to_frame(rect.right(), rect);
    let first = (min_f / minor_x).floor() as i64;
    let last = (max_f / minor_x).ceil() as i64;

    for idx in first..=last {
        let f = idx as f64 * minor_x;
        let x = state.frame_to_screen_x(f, rect).round();
        let is_major = idx % 5 == 0;
        let top = if is_major {
            (rect.top() + 4.0).round()
        } else {
            (rect.bottom() - 8.0).round()
        };
        painter.rect_filled(
            Rect::from_min_max(pos2(x, top), pos2(x + 1.0, rect.bottom())),
            CornerRadius::ZERO,
            if is_major {
                COLOR_GRID_ZERO
            } else {
                COLOR_GRID_MAJOR
            },
        );
        if is_major {
            let label = if major_x >= 1.0 {
                format!("{f:.0}")
            } else {
                format!("{f:.2}")
            };
            painter.text(
                pos2(x + 4.0, rect.top() + 4.0),
                Align2::LEFT_TOP,
                label,
                FontId::monospace(11.0),
                COLOR_LABEL_MUTED,
            );
        }
    }

    let ph_x = state
        .frame_to_screen_x(session.playback.current_frame, rect)
        .round();
    let ph_color = if graph_plus {
        COLOR_PLAYHEAD
    } else {
        COLOR_PLAYHEAD_VANILLA
    };
    painter.rect_filled(
        Rect::from_min_size(pos2(ph_x, rect.top()), vec2(1.0, rect.height())),
        CornerRadius::ZERO,
        ph_color,
    );
    painter.rect_filled(
        Rect::from_min_size(
            pos2(rect.left(), (rect.bottom() - 1.0).round()),
            vec2(rect.width(), 1.0),
        ),
        CornerRadius::ZERO,
        COLOR_BORDER,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn process_graph_drag(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut GraphEditorState,
    tracks: &[CurveTrack],
    rect: Rect,
    palette: &ThemePalette,
    pointer_pos: Option<Pos2>,
    primary_down: bool,
    modifiers: egui::Modifiers,
    action: &mut GraphEditorAction,
) {
    state.snap_guide_frame = None;
    let drag = std::mem::take(&mut state.drag_state);

    match drag {
        GraphDragState::DraggingKeys {
            start_pos,
            anchor_orig_abs_frame,
            anchor_orig_value,
            mut preview_df,
            mut preview_dv,
            keys,
            mut moved,
        } => {
            if let Some(pos) = pointer_pos {
                let delta_px = pos - start_pos;
                if delta_px.x.abs() + delta_px.y.abs() > 3.0 {
                    moved = true;
                }
                let mut raw_df = f64::from(delta_px.x) * state.zoom_x;
                let mut raw_dv = if state.plot_mode == GraphPlotMode::Speed {
                    0.0
                } else {
                    -f64::from(delta_px.y) * state.zoom_y
                };

                // Shift locks to dominant axis (horizontal frame-only or vertical value-only).
                if modifiers.shift {
                    if delta_px.x.abs() >= delta_px.y.abs() {
                        raw_dv = 0.0;
                    } else {
                        raw_df = 0.0;
                    }
                } else if modifiers.ctrl || modifiers.command {
                    raw_df = 0.0;
                }

                // Frame snapping within 8 px to playhead & other non-dragged keyframes.
                if raw_df.abs() <= 1e-9 {
                    preview_df = 0.0;
                } else {
                    let mut target_abs = (anchor_orig_abs_frame + raw_df).round();
                    if state.snap_enabled && !modifiers.alt {
                        let mut snap_targets = vec![session.playback.current_frame.round()];
                        for t in tracks {
                            if !state.is_track_visible(&t.track_key) {
                                continue;
                            }
                            for k in &t.keyframes {
                                let kid = GraphKeyId::new(
                                    &t.target_id,
                                    &t.property,
                                    t.channel_index,
                                    k.local_frame,
                                );
                                if !keys.iter().any(|dk| dk.id == kid) {
                                    snap_targets.push(t.frame_offset + k.local_frame);
                                }
                            }
                        }
                        let cand_x = state.frame_to_screen_x(target_abs, rect);
                        let mut best = SNAP_PX;
                        let mut snapped = None;
                        for st in snap_targets {
                            let d = (state.frame_to_screen_x(st, rect) - cand_x).abs();
                            if d < best {
                                best = d;
                                snapped = Some(st);
                            }
                        }
                        if let Some(st) = snapped {
                            target_abs = st;
                            state.snap_guide_frame = Some(st);
                        }
                    }

                    let min_abs = keys
                        .iter()
                        .map(|dk| dk.frame_offset + dk.orig_local_frame)
                        .fold(f64::INFINITY, f64::min);
                    let mut df = target_abs - anchor_orig_abs_frame;
                    if min_abs.is_finite() {
                        df = df.max(-min_abs);
                    }
                    preview_df = df;
                }
                let _ = anchor_orig_value;
                preview_dv = raw_dv;
            }

            if primary_down {
                ui.ctx().set_cursor_icon(CursorIcon::Move);
                state.drag_state = GraphDragState::DraggingKeys {
                    start_pos,
                    anchor_orig_abs_frame,
                    anchor_orig_value,
                    preview_df,
                    preview_dv,
                    keys,
                    moved,
                };
            } else if moved
                && (preview_df.abs() > 1e-6 || preview_dv.abs() > 1e-6)
                && session.begin_transaction("Edit Graph Keyframes").is_ok()
            {
                // Merge multi-channel edits per (target_id, property, new_frame) so dragging X and Y
                // together updates ALL dragged axes rather than overwriting earlier channels.
                struct MergedKeyEdit {
                    target_id: String,
                    property: String,
                    new_frame: f64,
                    value: Value,
                    tween: u16,
                    control_points: [[f64; 2]; 2],
                    continuous_tangent: bool,
                    members: Vec<(GraphKeyId, GraphKeyId)>,
                }

                let mut merged_edits: BTreeMap<(String, String, u64), MergedKeyEdit> =
                    BTreeMap::new();
                for dk in &keys {
                    let new_frame = dk.orig_local_frame + preview_df;
                    let new_val_scalar = dk.orig_value + preview_dv;
                    let new_id = GraphKeyId::new(
                        &dk.id.target_id,
                        &dk.id.property,
                        dk.id.channel_index,
                        new_frame,
                    );
                    let group_key = (
                        dk.id.target_id.clone(),
                        dk.id.property.clone(),
                        canonical_bits(new_frame),
                    );
                    merged_edits
                        .entry(group_key)
                        .and_modify(|edit| {
                            edit.value = write_channel_value(
                                &edit.value,
                                dk.id.channel_index,
                                new_val_scalar,
                            );
                            edit.members.push((dk.id.clone(), new_id.clone()));
                        })
                        .or_insert_with(|| MergedKeyEdit {
                            target_id: dk.id.target_id.clone(),
                            property: dk.id.property.clone(),
                            new_frame,
                            value: write_channel_value(
                                &dk.orig_full_value,
                                dk.id.channel_index,
                                new_val_scalar,
                            ),
                            tween: dk.tween,
                            control_points: dk.control_points,
                            continuous_tangent: dk.continuous_tangent,
                            members: vec![(dk.id.clone(), new_id)],
                        });
                }

                let mut new_sel = Vec::new();
                let mut updated_handles: Vec<(GraphKeyId, BezierHandlePair)> = Vec::new();
                let mut old_ids_to_clear: Vec<GraphKeyId> = Vec::new();
                let mut changed = false;

                for edit in merged_edits.into_values() {
                    if session
                        .execute(Command::SetKeyframe {
                            target_id: edit.target_id.clone(),
                            property: edit.property.clone(),
                            frame: edit.new_frame,
                            value: edit.value,
                            tween: Some(edit.tween),
                        })
                        .is_ok()
                    {
                        let _ = session.set_keyframe_bezier_handles(
                            &edit.target_id,
                            &edit.property,
                            edit.new_frame,
                            edit.control_points,
                            edit.continuous_tangent,
                            Some(edit.tween),
                        );
                        changed = true;
                        let hp = BezierHandlePair {
                            in_tangent: edit.control_points[0],
                            out_tangent: edit.control_points[1],
                            continuous: edit.continuous_tangent,
                        };
                        for (old_id, new_id) in edit.members {
                            let preserved = state.handles.get(&old_id).copied().unwrap_or(hp);
                            old_ids_to_clear.push(old_id);
                            updated_handles.push((new_id.clone(), preserved));
                            new_sel.push(new_id);
                        }
                    }
                }

                if changed {
                    for old_id in old_ids_to_clear {
                        state.handles.remove(&old_id);
                    }
                    for (new_id, hp) in updated_handles {
                        state.handles.insert(new_id, hp);
                    }
                    // Delete old frames once per moved (target_id, property, orig_local_frame).
                    if preview_df.abs() > 1e-6 {
                        let mut deleted_keys = BTreeSet::new();
                        for dk in &keys {
                            let del_key = (
                                dk.id.target_id.clone(),
                                dk.id.property.clone(),
                                canonical_bits(dk.orig_local_frame),
                            );
                            if !deleted_keys.insert(del_key) {
                                continue;
                            }
                            let still_used = keys.iter().any(|o| {
                                o.id.target_id == dk.id.target_id
                                    && o.id.property == dk.id.property
                                    && ((o.orig_local_frame + preview_df) - dk.orig_local_frame)
                                        .abs()
                                        < 1e-6
                            });
                            if !still_used {
                                let _ = session.execute(Command::DeleteKeyframe {
                                    target_id: dk.id.target_id.clone(),
                                    property: dk.id.property.clone(),
                                    frame: dk.orig_local_frame,
                                });
                            }
                        }
                    }
                    state.selected_keys = new_sel;
                }
                let _ = session.commit_transaction(changed);
                if changed {
                    action.curves_modified = true;
                }
            }
        }
        GraphDragState::DraggingHandle {
            key,
            is_out,
            prev_local_frame,
            next_local_frame,
            key_abs_frame,
            key_value,
            grab_start_pos,
            orig_tangent,
            mut changed,
        } => {
            if let Some(pos) = pointer_pos {
                let delta_px = pos - grab_start_pos;
                if changed || delta_px.length() > 0.5 || modifiers.alt {
                    let raw_dx = orig_tangent[0] + f64::from(delta_px.x) * state.zoom_x;
                    let raw_dy = orig_tangent[1] - f64::from(delta_px.y) * state.zoom_y;

                    // Clamp handle X strictly within adjacent segment [t0, t1].
                    let clamped_dx = if is_out {
                        let max_dx = (next_local_frame - key.local_frame()).max(0.0);
                        raw_dx.clamp(0.0, max_dx)
                    } else {
                        let min_dx = -(key.local_frame() - prev_local_frame).max(0.0);
                        raw_dx.clamp(min_dx, 0.0)
                    };

                    let updated_pair = {
                        let entry = state.handles.entry(key.clone()).or_default();
                        if modifiers.alt {
                            entry.continuous = false;
                        }
                        if is_out {
                            entry.out_tangent = [clamped_dx, raw_dy];
                            if entry.continuous {
                                let min_in = -(key.local_frame() - prev_local_frame).max(0.0);
                                entry.in_tangent = [(-clamped_dx).clamp(min_in, 0.0), -raw_dy];
                            }
                        } else {
                            entry.in_tangent = [clamped_dx, raw_dy];
                            if entry.continuous {
                                let max_out = (next_local_frame - key.local_frame()).max(0.0);
                                entry.out_tangent = [(-clamped_dx).clamp(0.0, max_out), -raw_dy];
                            }
                        }
                        *entry
                    };

                    // Persist tangent handles directly to Session / Project so rendering,
                    // export, undo/redo, and save/reload immediately reflect the edited curve.
                    if session
                        .set_keyframe_bezier_handles(
                            &key.target_id,
                            &key.property,
                            key.local_frame(),
                            [updated_pair.in_tangent, updated_pair.out_tangent],
                            updated_pair.continuous,
                            None,
                        )
                        .is_ok()
                    {
                        changed = true;
                        action.curves_modified = true;
                    }

                    // Auto-enable Bezier interpolation bit on the edited segment (and opposite
                    // segment too when tangent continuity is linked).
                    ensure_bezier_tween_for_handle(session, tracks, &key, is_out, action);
                    if updated_pair.continuous {
                        ensure_bezier_tween_for_handle(session, tracks, &key, !is_out, action);
                    }
                }
            }

            if primary_down {
                ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
                state.drag_state = GraphDragState::DraggingHandle {
                    key,
                    is_out,
                    prev_local_frame,
                    next_local_frame,
                    key_abs_frame,
                    key_value,
                    grab_start_pos,
                    orig_tangent,
                    changed,
                };
            } else if session.history.is_in_transaction() {
                let _ = session.commit_transaction(changed);
            }
        }
        GraphDragState::DraggingSpeedHandle {
            key,
            is_out,
            prev_local_frame,
            next_local_frame,
            key_abs_frame,
            fps,
            mut changed,
        } => {
            if let Some(pos) = pointer_pos {
                let target_frame = state.screen_x_to_frame(pos.x, rect);
                let target_speed = state.screen_y_to_value(pos.y, rect);
                let raw_dx = target_frame - key_abs_frame;
                let out_span = (next_local_frame - key.local_frame()).max(1e-3);
                let in_span = (key.local_frame() - prev_local_frame).max(1e-3);
                let span = if is_out { out_span } else { in_span };
                let tangent = speed_handle_to_tangent(raw_dx, target_speed, fps, is_out, span);

                let updated_pair = {
                    let entry = state.handles.entry(key.clone()).or_default();
                    if modifiers.alt {
                        entry.continuous = false;
                    }
                    if is_out {
                        entry.out_tangent = tangent;
                        if entry.continuous && prev_local_frame < key.local_frame() {
                            entry.in_tangent = speed_handle_to_tangent(
                                -tangent[0],
                                target_speed,
                                fps,
                                false,
                                in_span,
                            );
                        }
                    } else {
                        entry.in_tangent = tangent;
                        if entry.continuous && next_local_frame > key.local_frame() {
                            entry.out_tangent = speed_handle_to_tangent(
                                -tangent[0],
                                target_speed,
                                fps,
                                true,
                                out_span,
                            );
                        }
                    }
                    *entry
                };

                if session
                    .set_keyframe_bezier_handles(
                        &key.target_id,
                        &key.property,
                        key.local_frame(),
                        [updated_pair.in_tangent, updated_pair.out_tangent],
                        updated_pair.continuous,
                        None,
                    )
                    .is_ok()
                {
                    changed = true;
                    action.curves_modified = true;
                }
                ensure_bezier_tween_for_handle(session, tracks, &key, is_out, action);
                if updated_pair.continuous {
                    ensure_bezier_tween_for_handle(session, tracks, &key, !is_out, action);
                }
            }

            if primary_down {
                ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
                state.drag_state = GraphDragState::DraggingSpeedHandle {
                    key,
                    is_out,
                    prev_local_frame,
                    next_local_frame,
                    key_abs_frame,
                    fps,
                    changed,
                };
            } else if session.history.is_in_transaction() {
                let _ = session.commit_transaction(changed);
            }
        }
        GraphDragState::Marquee {
            origin,
            mut current,
            additive,
        } => {
            if let Some(pos) = pointer_pos {
                current = pos;
            }
            let sel_rect = Rect::from_two_pos(origin, current);
            ui.painter().with_clip_rect(rect).rect(
                sel_rect,
                CornerRadius::ZERO,
                rgb_alpha(palette.focus, 50),
                Stroke::new(1.0, rgb_color(palette.focus)),
                StrokeKind::Inside,
            );
            if primary_down {
                state.drag_state = GraphDragState::Marquee {
                    origin,
                    current,
                    additive,
                };
            } else if (current - origin).length() > 3.0 {
                if !additive {
                    state.selected_keys.clear();
                }
                let fps = session.document().fps.max(1.0);
                for track in tracks {
                    if !state.is_track_visible(&track.track_key) || track.is_expression {
                        continue;
                    }
                    for (idx, kf) in track.keyframes.iter().enumerate() {
                        let y_val = match state.plot_mode {
                            GraphPlotMode::Value => kf.value,
                            GraphPlotMode::Speed => {
                                speed_keyframe_anchor_velocity(state, track, idx, fps)
                            }
                        };
                        let pt = pos2(
                            state.frame_to_screen_x(track.frame_offset + kf.local_frame, rect),
                            state.value_to_screen_y(y_val, rect),
                        );
                        if sel_rect.contains(pt) {
                            let kid = GraphKeyId::new(
                                &track.target_id,
                                &track.property,
                                track.channel_index,
                                kf.local_frame,
                            );
                            if !state.selected_keys.contains(&kid) {
                                state.selected_keys.push(kid);
                            }
                        }
                    }
                }
            }
        }
        other => {
            state.drag_state = other;
        }
    }
}
pub fn show_plot_mode_toggle(
    ui: &mut Ui,
    state: &mut GraphEditorState,
    palette: &ThemePalette,
) -> bool {
    let mut switched = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        let is_val = state.plot_mode == GraphPlotMode::Value;
        let is_spd = state.plot_mode == GraphPlotMode::Speed;
        let seg_btn = |ui: &mut Ui, label: &str, active: bool, tooltip: &str| -> bool {
            let (rect, resp) = ui.allocate_exact_size(vec2(46.0, 22.0), Sense::click());
            let resp = resp
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text(tooltip);
            let p = ui.painter();
            p.rect(
                rect,
                CornerRadius::ZERO,
                if active {
                    rgb_color(palette.focus)
                } else if resp.hovered() {
                    COLOR_TILE_HOVER
                } else {
                    COLOR_TILE_BG
                },
                Stroke::new(
                    1.0,
                    if active {
                        rgb_color(palette.focus)
                    } else {
                        COLOR_BORDER
                    },
                ),
                StrokeKind::Inside,
            );
            p.text(
                rect.center(),
                Align2::CENTER_CENTER,
                label,
                FontId::monospace(10.5),
                if active {
                    COLOR_TEXT_BRIGHT
                } else {
                    COLOR_TEXT
                },
            );
            resp.clicked()
        };
        if seg_btn(
            ui,
            "Value",
            is_val,
            "Value Graph (property value over time)",
        ) && !is_val
        {
            state.plot_mode = GraphPlotMode::Value;
            switched = true;
        }
        if seg_btn(
            ui,
            "Speed",
            is_spd,
            "Speed Graph (velocity derivative dv/dt in units/sec)",
        ) && !is_spd
        {
            state.plot_mode = GraphPlotMode::Speed;
            switched = true;
        }
    });
    switched
}

pub type SpeedHandleHit = (GraphKeyId, bool, f64, f64, f64);

pub fn hit_test_speed_handle(
    state: &GraphEditorState,
    tracks: &[CurveTrack],
    rect: Rect,
    fps: f64,
    pointer: Pos2,
) -> Option<SpeedHandleHit> {
    let mut best_dist = 12.0_f32;
    let mut hit = None;
    for track in tracks {
        if !state.is_track_visible(&track.track_key) || track.is_expression {
            continue;
        }
        for (idx, kf) in track.keyframes.iter().enumerate() {
            let key_id = GraphKeyId::new(
                &track.target_id,
                &track.property,
                track.channel_index,
                kf.local_frame,
            );
            if !state.selected_keys.contains(&key_id) {
                continue;
            }
            let abs_f = track.frame_offset + kf.local_frame;
            let hp = state.handles.get(&key_id).copied().unwrap_or_default();
            let prev_f = idx
                .checked_sub(1)
                .and_then(|p| track.keyframes.get(p))
                .map_or(kf.local_frame, |k| k.local_frame);
            let next_f = track
                .keyframes
                .get(idx + 1)
                .map_or(kf.local_frame, |k| k.local_frame);

            if let Some(eff_in) = effective_in_tangent(state, track, idx, hp)
                && let Some(prev) = idx.checked_sub(1).and_then(|p| track.keyframes.get(p))
            {
                let in_vel =
                    eval_segment_velocity(track, prev, kf, kf.local_frame, fps, &state.handles);
                let pos = speed_handle_screen_pos(state, rect, abs_f, in_vel, eff_in[0], false);
                let d = pointer.distance(pos);
                if d < best_dist {
                    best_dist = d;
                    hit = Some((key_id.clone(), false, prev_f, next_f, abs_f));
                }
            }
            if let Some(eff_out) = effective_out_tangent(state, track, idx, hp)
                && let Some(next) = track.keyframes.get(idx + 1)
            {
                let out_vel =
                    eval_segment_velocity(track, kf, next, kf.local_frame, fps, &state.handles);
                let pos = speed_handle_screen_pos(state, rect, abs_f, out_vel, eff_out[0], true);
                let d = pointer.distance(pos);
                if d < best_dist {
                    best_dist = d;
                    hit = Some((key_id, true, prev_f, next_f, abs_f));
                }
            }
        }
    }
    hit
}

pub fn hit_test_speed_keyframe<'a>(
    state: &GraphEditorState,
    tracks: &'a [CurveTrack],
    rect: Rect,
    fps: f64,
    pointer: Pos2,
) -> Option<(&'a CurveTrack, usize)> {
    let mut best_dist = 11.0_f32;
    let mut hit = None;
    for track in tracks {
        if !state.is_track_visible(&track.track_key) || track.is_expression {
            continue;
        }
        for (idx, kf) in track.keyframes.iter().enumerate() {
            let vel = speed_keyframe_anchor_velocity(state, track, idx, fps);
            let pt = pos2(
                state.frame_to_screen_x(track.frame_offset + kf.local_frame, rect),
                state.value_to_screen_y(vel, rect),
            );
            let d = pointer.distance(pt);
            if d < best_dist {
                best_dist = d;
                hit = Some((track, idx));
            }
        }
    }
    hit
}

pub fn hit_test_keyframe<'a>(
    state: &GraphEditorState,
    tracks: &'a [CurveTrack],
    rect: Rect,
    pointer: Pos2,
) -> Option<(&'a CurveTrack, usize)> {
    let mut best_dist = 11.0_f32;
    let mut hit = None;
    for track in tracks {
        if !state.is_track_visible(&track.track_key) || track.is_expression {
            continue;
        }
        for (idx, kf) in track.keyframes.iter().enumerate() {
            let pt = pos2(
                state.frame_to_screen_x(track.frame_offset + kf.local_frame, rect),
                state.value_to_screen_y(kf.value, rect),
            );
            let d = pointer.distance(pt);
            if d < best_dist {
                best_dist = d;
                hit = Some((track, idx));
            }
        }
    }
    hit
}

pub fn compute_handle_screen_pos(
    state: &GraphEditorState,
    rect: Rect,
    abs_frame: f64,
    value: f64,
    rel_tangent: [f64; 2],
    is_out: bool,
) -> Pos2 {
    let origin = pos2(
        state.frame_to_screen_x(abs_frame, rect),
        state.value_to_screen_y(value, rect),
    );
    let raw = pos2(
        state.frame_to_screen_x(abs_frame + rel_tangent[0], rect),
        state.value_to_screen_y(value + rel_tangent[1], rect),
    );
    let delta = raw - origin;
    let len = delta.length();
    if len < 1e-4 {
        let dir = if is_out { 1.0 } else { -1.0 };
        pos2(origin.x + dir * MIN_HANDLE_SCREEN_PX, origin.y)
    } else if len < MIN_HANDLE_SCREEN_PX {
        origin + (delta / len) * MIN_HANDLE_SCREEN_PX
    } else {
        raw
    }
}

/// Computes adaptive 1-2-5-10 `(major_step, minor_step)` for grid lines (Spec 02 §4.2).
#[must_use]
pub fn adaptive_grid_step(zoom: f64) -> (f64, f64) {
    let target = (zoom * 64.0).max(1e-9);
    let power = 10.0_f64.powf(target.log10().floor());
    let factor = [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .find(|&v| v * power >= target)
        .unwrap_or(10.0);
    let major = factor * power;
    (major, major / 5.0)
}

pub fn paint_easing_thumbnail(
    painter: &egui::Painter,
    plot: Rect,
    kind: EasingKind,
    color: Color32,
) {
    if kind == EasingKind::Hold {
        let p0 = pos2(plot.left(), plot.bottom());
        let pm = pos2(plot.right(), plot.bottom());
        let p1 = pos2(plot.right(), plot.top());
        painter.add(Shape::line(vec![p0, pm, p1], Stroke::new(1.5, color)));
        return;
    }
    let steps = 24_usize;
    let mut pts = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let u = i as f64 / steps as f64;
        let e = eval_easing(kind, u).clamp(-0.35, 1.35);
        let x = plot.left() + (u as f32) * plot.width();
        let y = plot.bottom() - (e as f32) * plot.height();
        pts.push(pos2(x, y));
    }
    painter.add(Shape::line(pts, Stroke::new(1.5, color)));
}

pub fn toolbar_chip(ui: &mut Ui, label: &str, active: bool, palette: &ThemePalette) -> bool {
    let fill = if active {
        rgb_color(palette.focus)
    } else {
        COLOR_TILE_BG
    };
    let btn = egui::Button::new(
        egui::RichText::new(label)
            .monospace()
            .size(11.0)
            .color(COLOR_TEXT_BRIGHT),
    )
    .fill(fill)
    .corner_radius(CornerRadius::ZERO);
    ui.add(btn).clicked()
}

pub fn find_track_keyframe<'a>(
    tracks: &'a [CurveTrack],
    kid: &GraphKeyId,
) -> Option<(&'a CurveTrack, &'a CurveKeyframe)> {
    let track = tracks.iter().find(|t| t.track_key == kid.track_key())?;
    let kf = track
        .keyframes
        .iter()
        .find(|k| (k.local_frame - kid.local_frame()).abs() < 1e-6)?;
    Some((track, kf))
}

pub fn collect_dragged_graph_keys(
    tracks: &[CurveTrack],
    selected: &[GraphKeyId],
) -> Vec<DraggedGraphKey> {
    let mut out = Vec::new();
    for kid in selected {
        if let Some((track, kf)) = find_track_keyframe(tracks, kid) {
            out.push(DraggedGraphKey {
                id: kid.clone(),
                frame_offset: track.frame_offset,
                orig_local_frame: kf.local_frame,
                orig_value: kf.value,
                orig_full_value: kf.full_value.clone(),
                tween: kf.tween,
                control_points: kf.control_points,
                continuous_tangent: kf.continuous_tangent,
            });
        }
    }
    out
}

pub fn write_channel_value(orig: &Value, channel: Option<usize>, scalar: f64) -> Value {
    match (orig, channel) {
        (Value::Array(arr), Some(ch)) if ch < arr.len() => {
            let mut next = arr.clone();
            next[ch] = Value::from(scalar);
            Value::Array(next)
        }
        _ => Value::from(scalar),
    }
}

pub fn toggle_key_selection(list: &mut Vec<GraphKeyId>, id: GraphKeyId) {
    if let Some(pos) = list.iter().position(|existing| existing == &id) {
        list.remove(pos);
    } else {
        list.push(id);
    }
}

pub fn channel_track_key(target_id: &str, property: &str, channel: Option<usize>) -> String {
    match channel {
        Some(ch) => format!("{target_id}:{property}:{ch}"),
        None => format!("{target_id}:{property}"),
    }
}

pub fn canonical_bits(frame: f64) -> u64 {
    (frame + 0.0).to_bits()
}

pub fn string_hash_u32(s: &str) -> u32 {
    let mut h = 0_u32;
    for b in s.bytes() {
        h = h.wrapping_mul(31).wrapping_add(u32::from(b));
    }
    h
}

pub fn short_easing_name(kind: EasingKind) -> &'static str {
    match kind {
        EasingKind::Hold => "hold",
        EasingKind::Linear => "linear",
        EasingKind::QuadIn => "quad in",
        EasingKind::QuadOut => "quad out",
        EasingKind::QuadInOut => "quad io",
        EasingKind::CubicIn => "cubic in",
        EasingKind::CubicOut => "cubic out",
        EasingKind::CubicInOut => "cubic io",
        EasingKind::QuartIn => "quart in",
        EasingKind::QuartOut => "quart out",
        EasingKind::QuartInOut => "quart io",
        EasingKind::QuintIn => "quint in",
        EasingKind::QuintOut => "quint out",
        EasingKind::QuintInOut => "quint io",
        EasingKind::SineIn => "sine in",
        EasingKind::SineOut => "sine out",
        EasingKind::SineInOut => "sine io",
        EasingKind::ExpoIn => "expo in",
        EasingKind::ExpoOut => "expo out",
        EasingKind::ExpoInOut => "expo io",
        EasingKind::CircIn => "circ in",
        EasingKind::CircOut => "circ out",
        EasingKind::CircInOut => "circ io",
        EasingKind::ElasticIn => "elas in",
        EasingKind::ElasticOut => "elas out",
        EasingKind::ElasticInOut => "elas io",
        EasingKind::BackIn => "back in",
        EasingKind::BackOut => "back out",
        EasingKind::BackInOut => "back io",
        EasingKind::BounceIn => "bnc in",
        EasingKind::BounceOut => "bnc out",
        EasingKind::BounceInOut => "bnc io",
        EasingKind::CubicOutIn => "cubic oi",
        EasingKind::CustomBezier { .. } => "bezier",
    }
}

pub fn rgb_color(rgb: Rgb) -> Color32 {
    let [r, g, b] = rgb.to_u8();
    Color32::from_rgb(r, g, b)
}

pub fn rgb_alpha(rgb: Rgb, a: u8) -> Color32 {
    let [r, g, b] = rgb.to_u8();
    Color32::from_rgba_unmultiplied(r, g, b, a)
}

pub fn hsl_color(hue_deg: f32, sat: f32, light: f32) -> Color32 {
    rgb_color(Rgb::from_hsl(Hsl {
        h: (hue_deg / 360.0).rem_euclid(1.0),
        s: sat.clamp(0.0, 1.0),
        l: light.clamp(0.0, 1.0),
    }))
}
