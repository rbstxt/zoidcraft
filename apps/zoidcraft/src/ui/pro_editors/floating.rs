//! Floating Pro Editor window geometry, screen clamping and Node Graph window shell.

use crate::ui::controls::{
    ButtonVariant, FloatingWindowManager, FloatingWindowSpec, cm3_floating_window, cm3_prop_button,
};
use crate::ui::theme::{
    BG_PANEL, BG_SUNKEN, BORDER_PANEL, Cm3Theme, DANGER_TEXT, FONT_COMPACT, FONT_SMALL, TEXT_FAINT,
    TEXT_MUTED,
};
use egui::{Color32, CornerRadius, Id, Key, Pos2, Rect, Sense, Stroke, Vec2};
use zc_session::Session;

use super::*;

/// Computes a centered floating editor window `(pos, size)` clamped 100% inside `screen`
/// so the title bar, toolbar, canvas, inspector, and bottom status bar are never clipped.
#[must_use]
pub fn compute_clamped_editor_window_rect(screen: Rect, max_w: f32, max_h: f32) -> (Pos2, Vec2) {
    let margin = 40.0_f32;
    let avail_w = (screen.width() - margin * 2.0).max(240.0);
    let avail_h = (screen.height() - margin * 2.0).max(200.0);
    let w = avail_w.min(max_w).min((screen.width() - 16.0).max(240.0));
    let h = avail_h.min(max_h).min((screen.height() - 16.0).max(200.0));
    let max_x = (screen.right() - w - 8.0).max(screen.left() + 8.0);
    let max_y = (screen.bottom() - h - 8.0).max(screen.top() + 8.0);
    let x = (screen.left() + ((screen.width() - w) * 0.5).max(margin))
        .clamp(screen.left() + 8.0, max_x);
    let y =
        (screen.top() + ((screen.height() - h) * 0.5).max(margin)).clamp(screen.top() + 8.0, max_y);
    (
        Pos2::new(x.round(), y.round()),
        Vec2::new(w.round(), h.round()),
    )
}

/// Strictly clamps an open floating window's size and position inside `screen` (with an 8px
/// margin) so edge resizes or drags can NEVER push the `- □ ×` titlebar controls off-screen.
pub fn clamp_floating_window_in_screen(
    floating_windows: &mut FloatingWindowManager,
    spec: &FloatingWindowSpec,
    screen: Rect,
) {
    let Some(win) = floating_windows.windows.get_mut(&spec.id) else {
        return;
    };
    if !win.open || win.maximized {
        return;
    }
    let margin = 8.0_f32;
    let avail_w = (screen.width() - margin * 2.0).max(160.0);
    let avail_h = (screen.height() - margin * 2.0).max(80.0);
    let min_w = spec.min_size.x.max(180.0).min(avail_w);
    let min_h = spec.min_size.y.max(80.0).min(avail_h);

    win.size.x = win.size.x.clamp(min_w, avail_w);
    win.size.y = win.size.y.clamp(min_h, avail_h);

    let eff_size = if win.collapsed {
        Vec2::new(260.0_f32.min(avail_w), 30.0)
    } else {
        win.size
    };

    let min_x = screen.left() + margin;
    let max_x = (screen.right() - eff_size.x - margin).max(min_x);
    let min_y = screen.top() + margin;
    let max_y = (screen.bottom() - eff_size.y - margin).max(min_y);

    win.pos.x = win.pos.x.clamp(min_x, max_x);
    win.pos.y = win.pos.y.clamp(min_y, max_y);

    if !win.collapsed {
        // Guard against east/south-east resize drags overshooting the viewport right/bottom edge.
        let max_w_from_pos = (screen.right() - margin - win.pos.x).max(min_w);
        let max_h_from_pos = (screen.bottom() - margin - win.pos.y).max(min_h);
        win.size.x = win.size.x.min(max_w_from_pos);
        win.size.y = win.size.y.min(max_h_from_pos);
    }
}

/// Marks `Key::Escape` as consumed by a floating editor on this frame so lower windows
/// and global shortcuts do not react to the same press.
pub fn mark_floating_editor_escape_consumed(ctx: &egui::Context) {
    ctx.data_mut(|d| {
        d.insert_temp(Id::unique("zc_transient_popover_consumed_esc"), true);
    });
    ctx.input_mut(|i| {
        i.consume_key(egui::Modifiers::NONE, Key::Escape);
    });
}

/// Returns `true` and consumes `Key::Escape` when `window_id` is the topmost/focused
/// open floating pro editor and no transient popover already consumed `Escape` this frame.
pub fn consume_escape_for_floating_editor(
    ctx: &egui::Context,
    floating_windows: &FloatingWindowManager,
    window_id: &str,
) -> bool {
    if !floating_windows.is_open(window_id) {
        return false;
    }
    let popover_consumed = ctx.data(|d| {
        d.get_temp::<bool>(Id::unique("zc_transient_popover_consumed_esc"))
            .unwrap_or(false)
    });
    if popover_consumed {
        return false;
    }
    let topmost_id = floating_windows
        .focused_id
        .as_deref()
        .filter(|id| floating_windows.is_open(id))
        .or_else(|| {
            floating_windows
                .windows
                .iter()
                .filter(|(_, w)| w.open)
                .max_by_key(|(_, w)| w.z_order)
                .map(|(k, _)| k.as_str())
        });
    if topmost_id != Some(window_id) {
        return false;
    }
    let esc_consumed = ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, Key::Escape));
    if !esc_consumed {
        return false;
    }
    if let Some(focused_widget) = ctx.memory(|m| m.focused()) {
        ctx.memory_mut(|m| m.surrender_focus(focused_widget));
    }
    ctx.data_mut(|d| {
        d.insert_temp(Id::unique("zc_transient_popover_consumed_esc"), true);
    });
    true
}

pub fn ensure_pro_window_in_screen(
    floating_windows: &mut FloatingWindowManager,
    spec: &FloatingWindowSpec,
    screen: Rect,
    just_opened: &mut bool,
) {
    let (default_pos, default_size) =
        compute_clamped_editor_window_rect(screen, spec.default_size.x, spec.default_size.y);
    if !floating_windows.is_open(&spec.id) {
        floating_windows.open_or_focus(spec, screen);
        *just_opened = true;
    }
    if let Some(win) = floating_windows.windows.get_mut(&spec.id)
        && *just_opened
    {
        win.size = default_size;
        win.pos = default_pos;
        win.restore_size = default_size;
        win.restore_pos = default_pos;
        win.collapsed = false;
        win.maximized = false;
        *just_opened = false;
    }
    clamp_floating_window_in_screen(floating_windows, spec, screen);
}

/// Renders the floating Node Graph Editor window when `state.open` is true.
pub fn show_node_graph_editor_window(
    ctx: &egui::Context,
    viewport_rect: Rect,
    floating_windows: &mut FloatingWindowManager,
    state: &mut NodeGraphEditorState,
    session: &mut Session,
    theme: &Cm3Theme,
) {
    if !state.open {
        return;
    }

    let ctx_screen = ctx.input(|i| i.viewport_rect());
    let screen = if ctx_screen.width() > 100.0 && ctx_screen.height() > 100.0 {
        ctx_screen
    } else {
        viewport_rect
    };
    let (_, default_size) = compute_clamped_editor_window_rect(screen, 860.0, 560.0);

    let spec = FloatingWindowSpec::new("pro_node_graph_editor", "Node Graph Editor")
        .with_subtitle(if state.target_label.is_empty() {
            state.target_kind.label().to_string()
        } else {
            format!("{} — {}", state.target_kind.label(), state.target_label)
        })
        .with_default_size(default_size);

    ensure_pro_window_in_screen(floating_windows, &spec, screen, &mut state.just_opened);

    let mut close_requested = false;
    let mut should_commit = false;

    let ev = cm3_floating_window(ctx, floating_windows, &spec, screen, theme, |ui| {
        // --- Top Toolbar ---
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            if cm3_prop_button(
                ui,
                if state.palette_open {
                    "Close Palette"
                } else {
                    "+ Add Node"
                },
                if state.palette_open {
                    ButtonVariant::Primary
                } else {
                    ButtonVariant::Normal
                },
                false,
                theme,
            )
            .clicked()
            {
                state.palette_open = !state.palette_open;
            }

            // Template selector buttons
            let templates: &[&str] = match state.target_kind {
                NodeGraphTargetKind::NodeEffect => {
                    &["Passthrough", "Glow + Vignette", "Noise Displace"]
                }
                NodeGraphTargetKind::NodeMaterial => &["Noise Ramp", "Marble PBR", "Brushed Metal"],
            };
            egui::ComboBox::from_id_salt(Id::unique("node_graph_templates"))
                .selected_text("Templates")
                .width(120.0)
                .show_ui(ui, |ui| {
                    for (idx, name) in templates.iter().enumerate() {
                        if ui.selectable_label(false, *name).clicked() {
                            state.apply_template(idx);
                            if state.auto_apply {
                                should_commit = true;
                            }
                        }
                    }
                });

            if cm3_prop_button(ui, "Fit View", ButtonVariant::Normal, false, theme).clicked() {
                state.needs_initial_fit = true;
            }

            ui.label(
                egui::RichText::new(format!("Zoom: {:.0}%", state.zoom * 100.0))
                    .font(theme.font(FONT_SMALL))
                    .color(TEXT_MUTED),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if cm3_prop_button(ui, "Close", ButtonVariant::Normal, false, theme).clicked() {
                    close_requested = true;
                }
                let apply_label = if cfg!(target_os = "macos") {
                    if state.dirty {
                        "Apply * (Cmd+Enter)"
                    } else {
                        "Apply (Cmd+Enter)"
                    }
                } else if state.dirty {
                    "Apply * (Ctrl+Enter)"
                } else {
                    "Apply (Ctrl+Enter)"
                };
                if cm3_prop_button(ui, apply_label, ButtonVariant::Primary, false, theme).clicked()
                {
                    should_commit = true;
                }
                ui.checkbox(&mut state.auto_apply, "Auto-Apply");
            });
        });

        if ui.input(|i| (i.modifiers.ctrl || i.modifiers.command) && i.key_pressed(Key::Enter)) {
            should_commit = true;
        }

        ui.add_space(4.0);

        // --- Main 3-Column Split: Optional Palette | Canvas | Node Inspector ---
        let body_height = (ui.available_height() - 28.0).max(200.0);
        let palette_w = if state.palette_open { 200.0 } else { 0.0 };
        let inspector_w = 248.0;
        let total_w = ui.available_width();
        let canvas_w =
            (total_w - palette_w - inspector_w - if state.palette_open { 12.0 } else { 6.0 })
                .max(240.0);

        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;

            // Optional Left Palette (explicit top-down layout so items stack cleanly from top)
            if state.palette_open {
                ui.allocate_ui_with_layout(
                    Vec2::new(palette_w, body_height),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        egui::Frame::NONE
                            .fill(BG_SUNKEN)
                            .stroke(Stroke::new(1.0, BORDER_PANEL))
                            .corner_radius(CornerRadius::same(4))
                            .inner_margin(egui::Margin::same(6))
                            .show(ui, |ui| {
                                ui.set_width((palette_w - 14.0).max(100.0));
                                ui.set_min_height((body_height - 14.0).max(100.0));
                                ui.set_max_height((body_height - 14.0).max(100.0));
                                if show_node_palette_ui(ui, state, theme) && state.auto_apply {
                                    should_commit = true;
                                }
                            });
                    },
                );
            }

            // Center Interactive Graph Canvas
            let (canvas_rect, _) =
                ui.allocate_exact_size(Vec2::new(canvas_w, body_height), Sense::hover());
            if state.needs_initial_fit {
                fit_node_graph_view(state, canvas_rect.size());
                state.needs_initial_fit = false;
            }
            if show_node_canvas_ui(ui, canvas_rect, state, theme) && state.auto_apply {
                should_commit = true;
            }

            // Right Node Inspector Sidebar (pinned to TOP with top-down vertical layout)
            ui.allocate_ui_with_layout(
                Vec2::new(inspector_w, body_height),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    egui::Frame::NONE
                        .fill(BG_PANEL)
                        .stroke(Stroke::new(1.0, BORDER_PANEL))
                        .corner_radius(CornerRadius::same(4))
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            ui.set_width((inspector_w - 18.0).max(120.0));
                            ui.set_min_height((body_height - 18.0).max(100.0));
                            ui.set_max_height((body_height - 18.0).max(100.0));
                            if show_node_inspector_ui(ui, state, theme) && state.auto_apply {
                                should_commit = true;
                            }
                        });
                },
            );
        });

        ui.add_space(4.0);

        // --- Bottom Compile Status Bar ---
        {
            const HINT_TEXT: &str = "Drag sockets • Click wire to disconnect";
            let (badge_color, status_text) = match &state.compile_status {
                NodeCompileStatus::Ok(msg) => (Color32::from_rgb(76, 198, 122), msg.as_str()),
                NodeCompileStatus::Error(err) => (DANGER_TEXT, err.as_str()),
            };
            let (bar_rect, bar_resp) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 20.0), Sense::hover());
            let hint_galley = ui.painter().layout_no_wrap(
                HINT_TEXT.to_owned(),
                theme.font(FONT_COMPACT),
                TEXT_FAINT,
            );
            let full_status_galley = ui.painter().layout_no_wrap(
                status_text.to_owned(),
                theme.font(FONT_SMALL),
                badge_color,
            );
            let (left_max_w, show_hint) =
                compute_node_status_bar_layout(bar_rect.width(), hint_galley.size().x);

            let dot_center = Pos2::new(bar_rect.left() + 5.0, bar_rect.center().y);
            ui.painter().circle_filled(dot_center, 4.0, badge_color);

            let left_x = bar_rect.left() + 14.0;
            let (status_galley, truncated) = if full_status_galley.size().x > left_max_w {
                let chars: Vec<char> = status_text.chars().collect();
                let mut len = chars.len();
                let mut chosen = full_status_galley;
                while len > 0 {
                    len -= 1;
                    let candidate =
                        format!("{}…", chars[..len].iter().collect::<String>().trim_end());
                    let g =
                        ui.painter()
                            .layout_no_wrap(candidate, theme.font(FONT_SMALL), badge_color);
                    if g.size().x <= left_max_w || len == 0 {
                        chosen = g;
                        break;
                    }
                }
                (chosen, true)
            } else {
                (full_status_galley, false)
            };
            let status_clip = Rect::from_min_max(
                Pos2::new(left_x, bar_rect.top()),
                Pos2::new(
                    (left_x + left_max_w).min(bar_rect.right()),
                    bar_rect.bottom(),
                ),
            );
            ui.painter().with_clip_rect(status_clip).galley(
                Pos2::new(left_x, bar_rect.center().y - status_galley.size().y * 0.5),
                status_galley,
                badge_color,
            );
            if truncated {
                bar_resp.on_hover_text(status_text);
            }

            if show_hint {
                let hint_x =
                    (bar_rect.right() - hint_galley.size().x).max(status_clip.right() + 12.0);
                ui.painter().with_clip_rect(bar_rect).galley(
                    Pos2::new(hint_x, bar_rect.center().y - hint_galley.size().y * 0.5),
                    hint_galley,
                    TEXT_FAINT,
                );
            }
        }
    });

    clamp_floating_window_in_screen(floating_windows, &spec, screen);

    if ev.closed && ctx.input(|i| i.key_pressed(Key::Escape)) {
        mark_floating_editor_escape_consumed(ctx);
    } else if !ev.closed
        && consume_escape_for_floating_editor(ctx, floating_windows, "pro_node_graph_editor")
    {
        close_requested = true;
    }

    if should_commit {
        state.commit_to_session(session);
    }
    if close_requested || ev.closed {
        state.open = false;
        floating_windows.close("pro_node_graph_editor");
    }
}

/// Computes `(left_status_max_width, show_right_hint)` for the Node Graph status bar so the
/// left compile status text and right-aligned interaction hint never collide (minimum 12px gap).
pub fn compute_node_status_bar_layout(total_w: f32, right_hint_w: f32) -> (f32, bool) {
    const LEFT_PREFIX: f32 = 16.0;
    const GAP: f32 = 12.0;
    const MIN_LEFT_FOR_HINT: f32 = 140.0;
    let show_hint = total_w >= LEFT_PREFIX + MIN_LEFT_FOR_HINT + GAP + right_hint_w;
    let left_max_w = if show_hint {
        (total_w - LEFT_PREFIX - GAP - right_hint_w).max(40.0)
    } else {
        (total_w - LEFT_PREFIX).max(40.0)
    };
    (left_max_w, show_hint)
}

pub fn fit_node_graph_view(state: &mut NodeGraphEditorState, canvas_size: Vec2) {
    if state.nodes.is_empty() {
        state.pan = Vec2::new(40.0, 40.0);
        state.zoom = 1.0;
        return;
    }
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for node in &state.nodes {
        let spec = find_node_spec(state.target_kind, &node.kind);
        let sz = node_size_graph_space(spec);
        min_x = min_x.min(node.pos[0]);
        min_y = min_y.min(node.pos[1]);
        max_x = max_x.max(node.pos[0] + sz.x);
        max_y = max_y.max(node.pos[1] + sz.y);
    }
    let pad = 24.0_f32;
    let graph_w = (max_x - min_x).max(1.0);
    let graph_h = (max_y - min_y).max(1.0);
    let avail_w = (canvas_size.x - pad * 2.0).max(80.0);
    let avail_h = (canvas_size.y - pad * 2.0).max(80.0);
    let zoom = (avail_w / graph_w).min(avail_h / graph_h).clamp(0.35, 1.0);
    let scaled_w = graph_w * zoom;
    let scaled_h = graph_h * zoom;
    let offset_x = ((canvas_size.x - scaled_w) * 0.5).max(pad);
    let offset_y = ((canvas_size.y - scaled_h) * 0.5).max(pad);
    state.zoom = zoom;
    state.pan = Vec2::new(offset_x - min_x * zoom, offset_y - min_y * zoom);
}
