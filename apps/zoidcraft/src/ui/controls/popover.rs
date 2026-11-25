//! Transient popover escape coordination and floating window chrome.

use crate::ui::theme::{
    BG_HOVER, BG_PANEL, BG_TITLE, BORDER_CONTROL, BORDER_PANEL, Cm3Icon, Cm3Theme, FONT_SMALL,
    FONT_TITLE, TEXT_BRIGHT, TEXT_FAINT, TEXT_MUTED, TEXT_STRONG, color32_with_alpha,
    paint_cm3_icon,
};
use egui::{
    Align2, Area, Color32, CornerRadius, CursorIcon, Frame, Id, Key, Order, Painter, Pos2, Rect,
    Sense, Stroke, StrokeKind, Ui, UiBuilder, Vec2,
};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// 0. Transient Popover Escape Coordination (LIFO Dismissal Priority)
// ---------------------------------------------------------------------------

pub fn transient_popover_consumed_esc_bool_id() -> Id {
    Id::unique("zc_transient_popover_consumed_esc")
}

pub fn transient_popover_consumed_esc_pass_id() -> Id {
    Id::unique("zc_transient_popover_consumed_esc_pass")
}

pub fn transient_popover_open_pass_id() -> Id {
    Id::unique("zc_transient_popover_open_pass")
}

/// Resets the per-frame `bool` Escape-consumed flag at the start of each frame so sibling
/// windows (e.g. pro editors reading `zc_transient_popover_consumed_esc` as `bool`) stay
/// synchronized across frames.
pub fn reset_transient_popover_escape_flag(ctx: &egui::Context) {
    ctx.data_mut(|d| d.insert_temp(transient_popover_consumed_esc_bool_id(), false));
}

/// Consumes `Key::Escape` from egui's input state for the current pass so no underlying
/// floating window or global shortcut handler also fires on the same keypress.
pub fn consume_escape_key_event(ctx: &egui::Context) {
    ctx.input_mut(|i| {
        let _ = i.consume_key(egui::Modifiers::NONE, Key::Escape);
        i.events.retain(|ev| {
            !matches!(
                ev,
                egui::Event::Key {
                    key: Key::Escape,
                    pressed: true,
                    ..
                }
            )
        });
    });
}

/// Records that a foreground transient popover (color picker, dropdown, easing menu) is open
/// on the current egui pass so shortcut dispatchers at the top of the next pass yield `Escape`.
pub fn mark_transient_popover_open(ctx: &egui::Context) {
    let pass = ctx.cumulative_pass_nr();
    ctx.data_mut(|d| d.insert_temp(transient_popover_open_pass_id(), pass));
}

/// Clears the open-popover pass marker when a popover closes via non-Escape interaction
/// without clobbering another popover that opened during the current pass.
pub fn clear_transient_popover_open_if_stale(ctx: &egui::Context) {
    let cur = ctx.cumulative_pass_nr();
    ctx.data_mut(|d| {
        if d.get_temp::<u64>(transient_popover_open_pass_id())
            .is_some_and(|open_pass| open_pass < cur)
        {
            d.remove::<u64>(transient_popover_open_pass_id());
        }
    });
}

/// Marks that a foreground transient popover consumed `Key::Escape` during the current pass,
/// strips `Key::Escape` from input, and clears the open-popover marker.
pub fn consume_transient_popover_escape(ctx: &egui::Context) {
    let cur = ctx.cumulative_pass_nr();
    ctx.data_mut(|d| {
        d.insert_temp(transient_popover_consumed_esc_bool_id(), true);
        d.insert_temp(transient_popover_consumed_esc_pass_id(), cur);
        d.remove::<u64>(transient_popover_open_pass_id());
    });
    consume_escape_key_event(ctx);
}

/// Returns `true` if a foreground transient popover is currently open (either on this pass
/// or still open from the immediately preceding pass before widgets run).
#[must_use]
pub fn is_transient_popover_open(ctx: &egui::Context) -> bool {
    let cur = ctx.cumulative_pass_nr();
    ctx.data(|d| d.get_temp::<u64>(transient_popover_open_pass_id()))
        .is_some_and(|open_pass| cur <= open_pass.saturating_add(1))
}

/// Returns `true` if a foreground transient popover already consumed `Key::Escape` during
/// the current egui pass.
#[must_use]
pub fn transient_popover_consumed_escape(ctx: &egui::Context) -> bool {
    let cur = ctx.cumulative_pass_nr();
    ctx.data(|d| {
        d.get_temp::<u64>(transient_popover_consumed_esc_pass_id())
            .is_some_and(|consumed_pass| consumed_pass == cur)
            || d.get_temp::<bool>(transient_popover_consumed_esc_bool_id())
                .unwrap_or(false)
    })
}
// ---------------------------------------------------------------------------
// 9. Zoidium Floating Window Manager (Spec 01 §4.8)
// ---------------------------------------------------------------------------

/// Cascade anchor for newly opened floating windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CascadeOrigin {
    #[default]
    Right,
    Left,
    Center,
}

/// Static configuration for one floating window.
#[derive(Debug, Clone)]
pub struct FloatingWindowSpec {
    pub id: String,
    pub title: String,
    pub subtitle: String,
    pub default_size: Vec2,
    pub min_size: Vec2,
    pub cascade: CascadeOrigin,
}

impl FloatingWindowSpec {
    #[must_use]
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            subtitle: String::new(),
            default_size: Vec2::new(340.0, 420.0),
            min_size: Vec2::new(240.0, 120.0),
            cascade: CascadeOrigin::Right,
        }
    }

    #[must_use]
    pub fn with_subtitle(mut self, subtitle: impl Into<String>) -> Self {
        self.subtitle = subtitle.into();
        self
    }

    #[must_use]
    pub const fn with_default_size(mut self, size: Vec2) -> Self {
        self.default_size = size;
        self
    }
}

/// Mutable geometry & focus state for a floating window.
#[derive(Debug, Clone)]
pub struct FloatingWindowState {
    pub open: bool,
    pub pos: Pos2,
    pub size: Vec2,
    pub collapsed: bool,
    pub maximized: bool,
    pub restore_pos: Pos2,
    pub restore_size: Vec2,
    pub dragging: bool,
    pub z_order: u64,
}

/// Manages Z-order, focus, and 6-step cascade placement for all Zoidium floating windows.
#[derive(Debug, Clone, Default)]
pub struct FloatingWindowManager {
    pub windows: BTreeMap<String, FloatingWindowState>,
    pub focused_id: Option<String>,
    pub z_counter: u64,
    pub cascade_step: usize,
}

impl FloatingWindowManager {
    /// Opens or focuses a floating window, applying 6-step cascade if newly created.
    pub fn open_or_focus(&mut self, spec: &FloatingWindowSpec, viewport: Rect) {
        self.z_counter = self.z_counter.saturating_add(1);
        let z = self.z_counter;
        if let Some(win) = self.windows.get_mut(&spec.id) {
            win.open = true;
            win.z_order = z;
            self.focused_id = Some(spec.id.clone());
            return;
        }

        let step = (self.cascade_step % 6) as f32;
        self.cascade_step = (self.cascade_step + 1) % 6;
        let size = Vec2::new(
            spec.default_size.x.max(spec.min_size.x.max(180.0)),
            spec.default_size.y.max(spec.min_size.y.max(80.0)),
        );
        let pos = match spec.cascade {
            CascadeOrigin::Right => Pos2::new(
                viewport.right() - size.x - 380.0 - 36.0 * step,
                viewport.top() + 70.0 + 36.0 * step,
            ),
            CascadeOrigin::Left => Pos2::new(
                viewport.left() + 70.0 + 36.0 * step,
                viewport.top() + 70.0 + 36.0 * step,
            ),
            CascadeOrigin::Center => Pos2::new(
                viewport.center().x - size.x * 0.5 + 24.0 * step,
                viewport.center().y - size.y * 0.5 + 24.0 * step,
            ),
        };
        let clamped = clamp_window_pos(pos, size, viewport);
        self.windows.insert(
            spec.id.clone(),
            FloatingWindowState {
                open: true,
                pos: clamped,
                size,
                collapsed: false,
                maximized: false,
                restore_pos: clamped,
                restore_size: size,
                dragging: false,
                z_order: z,
            },
        );
        self.focused_id = Some(spec.id.clone());
    }

    /// Closes a floating window and transfers focus to the next highest open window.
    pub fn close(&mut self, id: &str) {
        if let Some(win) = self.windows.get_mut(id) {
            win.open = false;
            win.dragging = false;
        }
        if self.focused_id.as_deref() == Some(id) {
            self.focused_id = self
                .windows
                .iter()
                .filter(|(_, w)| w.open)
                .max_by_key(|(_, w)| w.z_order)
                .map(|(k, _)| k.clone());
        }
    }

    /// Brings an open floating window to the front.
    pub fn focus(&mut self, id: &str) {
        self.z_counter = self.z_counter.saturating_add(1);
        if let Some(win) = self.windows.get_mut(id) {
            win.z_order = self.z_counter;
            self.focused_id = Some(id.to_owned());
        }
    }

    /// Clears floating-window focus when clicking back into a main workspace pane.
    pub fn clear_focus(&mut self) {
        self.focused_id = None;
    }

    /// Returns `true` if `id` is currently open.
    #[must_use]
    pub fn is_open(&self, id: &str) -> bool {
        self.windows.get(id).is_some_and(|w| w.open)
    }

    /// Returns `true` if any floating window is currently open.
    #[must_use]
    pub fn has_any_open(&self) -> bool {
        self.windows.values().any(|w| w.open)
    }
}

/// Outcome of rendering [`cm3_floating_window`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FloatingWindowEvent {
    pub closed: bool,
    pub focused: bool,
}

pub fn clamp_window_pos(pos: Pos2, size: Vec2, viewport: Rect) -> Pos2 {
    // Keep at least 80px width and 32px height reachable (Spec 01 §4.8)
    let min_x = viewport.left() + 8.0 - size.x + 80.0;
    let max_x = (viewport.right() - 80.0).max(min_x);
    let min_y = viewport.top() + 8.0;
    let max_y = (viewport.bottom() - 32.0).max(min_y);
    Pos2::new(pos.x.clamp(min_x, max_x), pos.y.clamp(min_y, max_y))
}

/// Renders a Zoidium Floating Window (`30px` title bar, `#384668` focus border, `0.92` drag opacity,
/// 5 edge resize grips, double-click collapse, maximize, `Escape` close; Spec 01 §4.8).
pub fn cm3_floating_window(
    ctx: &egui::Context,
    manager: &mut FloatingWindowManager,
    spec: &FloatingWindowSpec,
    viewport: Rect,
    theme: &Cm3Theme,
    add_body: impl FnOnce(&mut Ui),
) -> FloatingWindowEvent {
    let mut ev = FloatingWindowEvent::default();
    let Some(mut win) = manager.windows.get(&spec.id).cloned() else {
        return ev;
    };
    if !win.open {
        return ev;
    }

    let is_focused = manager.focused_id.as_deref() == Some(spec.id.as_str());
    let min_w = spec.min_size.x.max(180.0);
    let min_h = spec.min_size.y.max(80.0);

    // Escape closes the focused floating window unless a text input or foreground transient
    // popover (color picker / dropdown / easing menu) owns Escape (Spec 01 §4.8).
    if is_focused
        && !ctx.egui_wants_keyboard_input()
        && !is_transient_popover_open(ctx)
        && !transient_popover_consumed_escape(ctx)
        && ctx.input(|i| i.key_pressed(Key::Escape))
    {
        manager.close(&spec.id);
        consume_escape_key_event(ctx);
        ev.closed = true;
        return ev;
    }

    let (rect_pos, rect_size) = if win.maximized {
        (
            Pos2::new(viewport.left() + 8.0, viewport.top() + 8.0),
            Vec2::new(
                (viewport.width() - 16.0).max(min_w),
                (viewport.height() - 16.0).max(min_h),
            ),
        )
    } else if win.collapsed {
        let collapsed_size = Vec2::new(260.0, 30.0);
        (
            clamp_window_pos(win.pos, collapsed_size, viewport),
            collapsed_size,
        )
    } else {
        (
            clamp_window_pos(win.pos, win.size, viewport),
            Vec2::new(win.size.x.max(min_w), win.size.y.max(min_h)),
        )
    };

    let win_id = Id::unique(("zoidcraft_floating_window", &spec.id));
    let drag_alpha = if win.dragging { 235 } else { 255 }; // 0.92 opacity while dragging
    let body_bg = color32_with_alpha(BG_PANEL, drag_alpha);
    let title_bg_opaque = if is_focused {
        theme.bg_title_focused()
    } else {
        BG_TITLE
    };
    let title_bg = color32_with_alpha(title_bg_opaque, drag_alpha);
    let title_fg = if is_focused {
        theme.contrast_fg(title_bg_opaque)
    } else {
        TEXT_STRONG
    };
    let border_color = if is_focused {
        theme.accent()
    } else {
        BORDER_PANEL
    };

    let mut request_focus = false;
    let mut request_close = false;

    let area_resp = Area::new(win_id)
        .order(Order::Middle)
        .fixed_pos(rect_pos)
        .interactable(true)
        .show(ctx, |ui| {
            let outer_rect = Rect::from_min_size(rect_pos, rect_size);

            // Soft multi-pass drop shadow (0 8px 28px rgba(0,0,0,0.55))
            let shadow_layers = [(10.0, 18), (6.0, 34), (3.0, 56)];
            for (expand, alpha) in shadow_layers {
                let sr = outer_rect.expand(expand).translate(Vec2::new(0.0, 4.0));
                ui.painter()
                    .rect_filled(sr, CornerRadius::ZERO, Color32::from_black_alpha(alpha));
            }

            // Detect primary press anywhere inside outer_rect on this window's layer (including body widgets).
            let area_layer_id = ui.layer_id();
            let press_pos = ctx.input(|i| {
                if i.pointer.primary_pressed() {
                    i.pointer.interact_pos()
                } else {
                    None
                }
            });
            if press_pos.is_some_and(|p| {
                outer_rect.contains(p) && ctx.layer_id_at(p) == Some(area_layer_id)
            }) {
                request_focus = true;
            }

            let bg_resp = ui.interact(outer_rect, win_id.with("bg"), Sense::click());
            if bg_resp.clicked() || bg_resp.is_pointer_button_down_on() {
                request_focus = true;
            }

            ui.painter()
                .rect_filled(outer_rect, CornerRadius::ZERO, body_bg);

            // --- 30px Title Bar ---
            let title_rect =
                Rect::from_min_size(outer_rect.left_top(), Vec2::new(rect_size.x, 30.0));
            ui.painter()
                .rect_filled(title_rect, CornerRadius::ZERO, title_bg);
            ui.painter().line_segment(
                [title_rect.left_bottom(), title_rect.right_bottom()],
                Stroke::new(1.0, BORDER_CONTROL),
            );

            // Title bar action buttons (26x26 each: Collapse, Maximize, Close)
            let btn_size = Vec2::splat(26.0);
            let close_rect = Rect::from_min_size(
                Pos2::new(title_rect.right() - 28.0, title_rect.top() + 2.0),
                btn_size,
            );
            let max_rect = Rect::from_min_size(
                Pos2::new(close_rect.left() - 26.0, title_rect.top() + 2.0),
                btn_size,
            );
            let col_rect = Rect::from_min_size(
                Pos2::new(max_rect.left() - 26.0, title_rect.top() + 2.0),
                btn_size,
            );

            let drag_rect = Rect::from_min_max(
                title_rect.left_top(),
                Pos2::new(col_rect.left(), title_rect.bottom()),
            );
            let drag_resp = ui
                .interact(
                    drag_rect,
                    win_id.with("title_drag"),
                    Sense::click_and_drag(),
                )
                .on_hover_cursor(CursorIcon::Move);

            if drag_resp.drag_started() {
                request_focus = true;
                if !win.maximized {
                    win.dragging = true;
                }
            }
            if drag_resp.dragged() && !win.maximized {
                win.pos = clamp_window_pos(win.pos + drag_resp.drag_delta(), rect_size, viewport);
            }
            if drag_resp.drag_stopped() {
                win.dragging = false;
            }
            if drag_resp.double_clicked() {
                win.collapsed = !win.collapsed;
                if win.collapsed {
                    win.maximized = false;
                }
                request_focus = true;
            }
            if drag_resp.clicked() {
                request_focus = true;
            }

            // Title & optional subtitle
            let title_clip = ui.painter().with_clip_rect(drag_rect);
            let title_galley =
                title_clip.layout_no_wrap(spec.title.clone(), theme.font(FONT_TITLE), title_fg);
            let title_pos = Pos2::new(
                title_rect.left() + 9.0,
                title_rect.center().y - title_galley.size().y * 0.5,
            );
            let title_right = title_pos.x + title_galley.size().x;
            title_clip.galley(title_pos, title_galley, title_fg);

            if !win.collapsed && !spec.subtitle.is_empty() {
                title_clip.text(
                    Pos2::new(title_right + 8.0, title_rect.center().y),
                    Align2::LEFT_CENTER,
                    &spec.subtitle,
                    theme.font(FONT_SMALL),
                    theme.value(),
                );
            }

            // Render title bar buttons
            let draw_title_btn = |ui: &mut Ui, r: Rect, icon: Cm3Icon, tip: &str, key: &str| {
                let resp = ui
                    .interact(r, win_id.with(key), Sense::click())
                    .on_hover_cursor(CursorIcon::PointingHand)
                    .on_hover_text(tip);
                if resp.hovered() {
                    ui.painter().rect_filled(r, CornerRadius::ZERO, BG_HOVER);
                }
                let ic = Rect::from_center_size(r.center(), Vec2::splat(12.0));
                paint_cm3_icon(
                    ui.painter(),
                    ic,
                    icon,
                    if resp.hovered() {
                        TEXT_BRIGHT
                    } else {
                        TEXT_MUTED
                    },
                );
                resp.clicked()
            };

            if draw_title_btn(ui, col_rect, Cm3Icon::Collapse, "collapse", "col") {
                win.collapsed = !win.collapsed;
                if win.collapsed {
                    win.maximized = false;
                }
                request_focus = true;
            }
            if draw_title_btn(ui, max_rect, Cm3Icon::Maximize, "maximize / restore", "max") {
                if win.maximized {
                    win.maximized = false;
                    win.pos = win.restore_pos;
                    win.size = win.restore_size;
                } else {
                    win.restore_pos = win.pos;
                    win.restore_size = win.size;
                    win.maximized = true;
                    win.collapsed = false;
                }
                request_focus = true;
            }
            if draw_title_btn(ui, close_rect, Cm3Icon::Close, "close (Esc)", "close") {
                request_close = true;
            }

            // --- Window Body ---
            if !win.collapsed {
                let body_rect = Rect::from_min_max(
                    Pos2::new(outer_rect.left() + 1.0, title_rect.bottom()),
                    Pos2::new(outer_rect.right() - 1.0, outer_rect.bottom() - 1.0),
                );
                ui.scope_builder(
                    UiBuilder::new()
                        .max_rect(body_rect)
                        .id_salt(win_id.with("body")),
                    |ui| {
                        ui.set_clip_rect(ui.clip_rect().intersect(body_rect));
                        Frame::NONE
                            .inner_margin(egui::Margin::same(8))
                            .show(ui, |ui| {
                                add_body(ui);
                            });
                    },
                );

                // --- 5 Edge Resize Grips on outer rim (E, W, S, SE, SW; hidden when maximized) ---
                if !win.maximized {
                    // East outer rim
                    let e_rect = Rect::from_min_max(
                        Pos2::new(outer_rect.right() - 2.0, title_rect.bottom()),
                        Pos2::new(outer_rect.right() + 6.0, outer_rect.bottom() - 8.0),
                    );
                    let e_resp = ui
                        .interact(e_rect, win_id.with("rz_e"), Sense::drag())
                        .on_hover_cursor(CursorIcon::ResizeHorizontal);
                    if e_resp.dragged() {
                        win.size.x = (win.size.x + e_resp.drag_delta().x).max(min_w);
                        request_focus = true;
                    }

                    // West outer rim (anchors right edge)
                    let w_rect = Rect::from_min_max(
                        Pos2::new(outer_rect.left() - 6.0, title_rect.bottom()),
                        Pos2::new(outer_rect.left() + 2.0, outer_rect.bottom() - 8.0),
                    );
                    let w_resp = ui
                        .interact(w_rect, win_id.with("rz_w"), Sense::drag())
                        .on_hover_cursor(CursorIcon::ResizeHorizontal);
                    if w_resp.dragged() {
                        let right_edge = win.pos.x + win.size.x;
                        let new_w = (win.size.x - w_resp.drag_delta().x).max(min_w);
                        win.pos.x = right_edge - new_w;
                        win.size.x = new_w;
                        request_focus = true;
                    }

                    // South outer rim
                    let s_rect = Rect::from_min_max(
                        Pos2::new(outer_rect.left() + 8.0, outer_rect.bottom() - 2.0),
                        Pos2::new(outer_rect.right() - 8.0, outer_rect.bottom() + 6.0),
                    );
                    let s_resp = ui
                        .interact(s_rect, win_id.with("rz_s"), Sense::drag())
                        .on_hover_cursor(CursorIcon::ResizeVertical);
                    if s_resp.dragged() {
                        win.size.y = (win.size.y + s_resp.drag_delta().y).max(min_h);
                        request_focus = true;
                    }

                    // South-East outer rim corner + 6x6 #6d6d6d L-bracket
                    let se_rect = Rect::from_min_max(
                        Pos2::new(outer_rect.right() - 8.0, outer_rect.bottom() - 8.0),
                        Pos2::new(outer_rect.right() + 6.0, outer_rect.bottom() + 6.0),
                    );
                    let se_resp = ui
                        .interact(se_rect, win_id.with("rz_se"), Sense::drag())
                        .on_hover_cursor(CursorIcon::ResizeNwSe);
                    if se_resp.dragged() {
                        let d = se_resp.drag_delta();
                        win.size.x = (win.size.x + d.x).max(min_w);
                        win.size.y = (win.size.y + d.y).max(min_h);
                        request_focus = true;
                    }
                    let l_stroke = Stroke::new(2.0, TEXT_FAINT);
                    let p_br = Pos2::new(outer_rect.right() - 3.0, outer_rect.bottom() - 3.0);
                    ui.painter()
                        .line_segment([Pos2::new(p_br.x - 6.0, p_br.y), p_br], l_stroke);
                    ui.painter()
                        .line_segment([Pos2::new(p_br.x, p_br.y - 6.0), p_br], l_stroke);

                    // South-West outer rim corner (anchors right edge)
                    let sw_rect = Rect::from_min_max(
                        Pos2::new(outer_rect.left() - 6.0, outer_rect.bottom() - 8.0),
                        Pos2::new(outer_rect.left() + 8.0, outer_rect.bottom() + 6.0),
                    );
                    let sw_resp = ui
                        .interact(sw_rect, win_id.with("rz_sw"), Sense::drag())
                        .on_hover_cursor(CursorIcon::ResizeNeSw);
                    if sw_resp.dragged() {
                        let d = sw_resp.drag_delta();
                        let right_edge = win.pos.x + win.size.x;
                        let new_w = (win.size.x - d.x).max(min_w);
                        win.pos.x = right_edge - new_w;
                        win.size.x = new_w;
                        win.size.y = (win.size.y + d.y).max(min_h);
                        request_focus = true;
                    }
                }
            }

            // Outer 1px focus/unfocused border
            ui.painter().rect_stroke(
                outer_rect,
                CornerRadius::ZERO,
                Stroke::new(1.0, border_color),
                StrokeKind::Inside,
            );
        });

    if request_close {
        manager.close(&spec.id);
        ev.closed = true;
    } else {
        manager.windows.insert(spec.id.clone(), win);
        if request_focus {
            manager.focus(&spec.id);
            ctx.move_to_top(area_resp.response.layer_id);
            ev.focused = true;
        } else if is_focused {
            ctx.move_to_top(area_resp.response.layer_id);
        }
    }
    ev
}

// ---------------------------------------------------------------------------
// Shared Viewport Popup Clamping, Pixel Snapping & Tree-Depth Helpers
// ---------------------------------------------------------------------------

thread_local! {
    pub static CM3_TREE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub static CM3_ROW_HAS_DROPDOWN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub static CM3_ROW_HAS_COLOR_PICKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub struct TreeDepthGuard {
    pub prev: usize,
}

impl TreeDepthGuard {
    pub fn push(next: usize) -> Self {
        let prev = CM3_TREE_DEPTH.with(|d| {
            let old = d.get();
            d.set(next);
            old
        });
        Self { prev }
    }
}

impl Drop for TreeDepthGuard {
    fn drop(&mut self) {
        CM3_TREE_DEPTH.with(|d| d.set(self.prev));
    }
}

#[inline]
pub fn snap_rect_px(r: Rect) -> Rect {
    Rect::from_min_max(
        Pos2::new(r.left().round(), r.top().round()),
        Pos2::new(r.right().round(), r.bottom().round()),
    )
}

#[inline]
pub fn paint_crisp_border_1px(painter: &Painter, rect: Rect, color: Color32) {
    let snapped = snap_rect_px(rect);
    let min_x = snapped.min.x;
    let min_y = snapped.min.y;
    let max_x = snapped.max.x;
    let max_y = snapped.max.y;
    if max_x - min_x < 1.0 || max_y - min_y < 1.0 {
        return;
    }
    // Top edge: [min_x..max_x, min_y..min_y+1]
    painter.rect_filled(
        Rect::from_min_max(Pos2::new(min_x, min_y), Pos2::new(max_x, min_y + 1.0)),
        CornerRadius::ZERO,
        color,
    );
    // Bottom edge: [min_x..max_x, max_y-1..max_y]
    if max_y - min_y > 1.0 {
        painter.rect_filled(
            Rect::from_min_max(Pos2::new(min_x, max_y - 1.0), Pos2::new(max_x, max_y)),
            CornerRadius::ZERO,
            color,
        );
    }
    // Left & right inner edges: [min_y+1..max_y-1]
    if max_y - min_y > 2.0 {
        painter.rect_filled(
            Rect::from_min_max(
                Pos2::new(min_x, min_y + 1.0),
                Pos2::new(min_x + 1.0, max_y - 1.0),
            ),
            CornerRadius::ZERO,
            color,
        );
        if max_x - min_x > 1.0 {
            painter.rect_filled(
                Rect::from_min_max(
                    Pos2::new(max_x - 1.0, min_y + 1.0),
                    Pos2::new(max_x, max_y - 1.0),
                ),
                CornerRadius::ZERO,
                color,
            );
        }
    }
}

pub fn clamp_popup_pos(anchor: Rect, popup_size: Vec2, viewport: Rect) -> Pos2 {
    let mut x = anchor.left().round();
    let mut y = anchor.bottom().round() + 1.0;

    if y + popup_size.y > viewport.bottom() - 4.0 {
        let flipped_y = anchor.top().round() - popup_size.y - 1.0;
        if flipped_y >= viewport.top() + 4.0 {
            y = flipped_y;
        } else {
            y = (viewport.bottom() - popup_size.y - 4.0).max(viewport.top() + 4.0);
        }
    }
    if x + popup_size.x > viewport.right() - 4.0 {
        x = (viewport.right() - popup_size.x - 4.0).max(viewport.left() + 4.0);
    }
    if x < viewport.left() + 4.0 {
        x = viewport.left() + 4.0;
    }
    Pos2::new(x.round(), y.round())
}
