//! Pixel-exact & interaction-exact CM3 property widgets and Zoidium Floating Window manager
//! implementing Spec 01 §4.

use std::collections::BTreeMap;

use egui::{
    Align2, Area, Color32, CornerRadius, CursorIcon, Frame, Id, Key, Mesh, Order, Painter, Pos2,
    Rect, Response, ScrollArea, Sense, Shape, Stroke, StrokeKind, TextEdit, Ui, UiBuilder, Vec2,
};
use zc_anim::EasingKind;
use zc_color::{Hsv, Rgb, Rgba};
use zc_project::GradientStop;

use super::theme::{
    BG_BACK_BUTTON, BG_BUTTON, BG_BUTTON_PRESSED, BG_DROPDOWN_HOVER, BG_HOVER, BG_INPUT,
    BG_OPTION_ACTIVE, BG_OPTION_HOVER, BG_PANEL, BG_TITLE, BORDER_BUTTON_HOVER, BORDER_CONTROL,
    BORDER_FILTER, BORDER_OPTION_ACTIVE, BORDER_PANEL, BORDER_ROW_SOFT, Cm3Icon, Cm3Theme,
    DANGER_TEXT, FONT_BANNER, FONT_BODY, FONT_EASING, FONT_SMALL, FONT_TITLE, TEXT_BRIGHT,
    TEXT_BUTTON, TEXT_DISABLED, TEXT_FAINT, TEXT_MUTED, TEXT_OPTION_DESC, TEXT_PRIMARY,
    TEXT_STRONG, color32_with_alpha, paint_cm3_icon, paint_easing_preview, rgb_to_color32,
    rgba_to_color32,
};

// ---------------------------------------------------------------------------
// 0. Transient Popover Escape Coordination (LIFO Dismissal Priority)
// ---------------------------------------------------------------------------

fn transient_popover_consumed_esc_bool_id() -> Id {
    Id::unique("zc_transient_popover_consumed_esc")
}

fn transient_popover_consumed_esc_pass_id() -> Id {
    Id::unique("zc_transient_popover_consumed_esc_pass")
}

fn transient_popover_open_pass_id() -> Id {
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
fn consume_escape_key_event(ctx: &egui::Context) {
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
fn clear_transient_popover_open_if_stale(ctx: &egui::Context) {
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
// 1. Numeric Scrub Input (Exact CM3 Spec 01 §4.3 State Machine)
// ---------------------------------------------------------------------------

/// Configuration for [`cm3_scrub_number`], [`cm3_vector2`], and [`cm3_vector3`].
#[derive(Debug, Clone, Copy)]
pub struct ScrubConfig {
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub decimals: usize,
    /// Multiplied when writing to storage (`stored = ui * scale_factor`).
    pub scale_factor: f64,
    pub read_only: bool,
}

impl Default for ScrubConfig {
    fn default() -> Self {
        Self {
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
            step: 1.0,
            decimals: 2,
            scale_factor: 1.0,
            read_only: false,
        }
    }
}

impl ScrubConfig {
    #[must_use]
    pub const fn new(min: f64, max: f64, step: f64, decimals: usize) -> Self {
        Self {
            min,
            max,
            step,
            decimals,
            scale_factor: 1.0,
            read_only: false,
        }
    }

    #[must_use]
    pub const fn with_scale_factor(mut self, scale_factor: f64) -> Self {
        self.scale_factor = scale_factor;
        self
    }

    #[must_use]
    pub const fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }
}

/// Lifecycle events emitted by [`cm3_scrub_number`] for single-transaction undo grouping.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScrubEvent {
    /// Emitted once on gesture start (`mousedown` or entering text edit).
    pub started: bool,
    /// Emitted every frame while live-scrubbing or stepping.
    pub changed: bool,
    /// Emitted once when the gesture completes with a modified value.
    pub committed: bool,
    /// Emitted when cancelled (`Escape`) or ended with no net change.
    pub cancelled: bool,
}

impl ScrubEvent {
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            started: self.started || other.started,
            changed: self.changed || other.changed,
            committed: self.committed || other.committed,
            cancelled: self.cancelled || other.cancelled,
        }
    }
}

#[derive(Debug, Clone)]
enum ScrubMode {
    Idle,
    PendingDrag {
        start_pos: Pos2,
        last_pos: Pos2,
        initial_display: f64,
        start_value: f64,
        accum_display: f64,
        alt_at_start: bool,
    },
    Scrubbing {
        last_pos: Pos2,
        initial_display: f64,
        start_value: f64,
        accum_display: f64,
        alt_at_start: bool,
    },
    TextEdit {
        buffer: String,
        initial_display: f64,
        start_value: f64,
        just_entered: bool,
        alt_at_start: bool,
    },
}

#[derive(Debug, Clone)]
struct ScrubMemory {
    mode: ScrubMode,
}

impl Default for ScrubMemory {
    fn default() -> Self {
        Self {
            mode: ScrubMode::Idle,
        }
    }
}

fn round_decimals(val: f64, decimals: usize) -> f64 {
    let factor = 10.0_f64.powi(decimals.min(9) as i32);
    (val * factor).round() / factor
}

fn clamp_and_round(val: f64, cfg: ScrubConfig) -> f64 {
    let (lo, hi) = if cfg.min <= cfg.max {
        (cfg.min, cfg.max)
    } else {
        (cfg.max, cfg.min)
    };
    let mut clamped = if val.is_finite() { val } else { 0.0 };
    if lo.is_finite() && clamped < lo {
        clamped = lo;
    }
    if hi.is_finite() && clamped > hi {
        clamped = hi;
    }
    round_decimals(clamped, cfg.decimals)
}

fn clamp_stored_bounds(val: f64, cfg: ScrubConfig) -> f64 {
    let safe_scale = if cfg.scale_factor.is_finite() && cfg.scale_factor.abs() > 1e-12 {
        cfg.scale_factor
    } else {
        1.0
    };
    let mut a = cfg.min * safe_scale;
    let mut b = cfg.max * safe_scale;
    if a.is_nan() {
        a = f64::NEG_INFINITY;
    }
    if b.is_nan() {
        b = f64::INFINITY;
    }
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let mut out = if val.is_finite() { val } else { 0.0 };
    if lo.is_finite() && out < lo {
        out = lo;
    }
    if hi.is_finite() && out > hi {
        out = hi;
    }
    out
}

/// Exact CM3 numeric scrub + click-to-type control (Spec 01 §4.3).
pub fn cm3_scrub_number(
    ui: &mut Ui,
    id: Id,
    value: &mut f64,
    config: ScrubConfig,
    theme: &Cm3Theme,
) -> ScrubEvent {
    cm3_scrub_number_inner(ui, id, value, config, theme).0
}

/// Internal numeric scrub implementation returning `(event, text_edit_commit_info)` where
/// `text_edit_commit_info` is `Some((pre_edit_start_value, alt_at_start))` on a committed `TextEdit`.
fn cm3_scrub_number_inner(
    ui: &mut Ui,
    id: Id,
    value: &mut f64,
    config: ScrubConfig,
    theme: &Cm3Theme,
) -> (ScrubEvent, Option<(f64, bool)>) {
    cm3_scrub_number_styled(ui, id, value, config, theme, FONT_BODY, 3.0)
}

fn cm3_scrub_number_styled(
    ui: &mut Ui,
    id: Id,
    value: &mut f64,
    config: ScrubConfig,
    theme: &Cm3Theme,
    font_size: f32,
    left_inset: f32,
) -> (ScrubEvent, Option<(f64, bool)>) {
    let safe_scale = if config.scale_factor.is_finite() && config.scale_factor.abs() > 1e-12 {
        config.scale_factor
    } else {
        1.0
    };

    // Read-only state: #999999 text, default cursor, non-interactive.
    // If read_only flipped on mid-gesture (PendingDrag or Scrubbing), emit cancelled and reset to Idle.
    if config.read_only {
        let mut ev = ScrubEvent::default();
        if let Some(prev_mem) = ui.data_mut(|d| d.get_temp::<ScrubMemory>(id)) {
            match prev_mem.mode {
                ScrubMode::PendingDrag { start_value, .. }
                | ScrubMode::Scrubbing { start_value, .. } => {
                    *value = start_value;
                    ev.cancelled = true;
                    ui.data_mut(|d| d.insert_temp(id, ScrubMemory::default()));
                }
                ScrubMode::TextEdit { .. } => {
                    ui.memory_mut(|m| m.surrender_focus(id.with("text_edit")));
                    ui.data_mut(|d| d.insert_temp(id, ScrubMemory::default()));
                }
                ScrubMode::Idle => {}
            }
        }
        let display_val = clamp_and_round(*value / safe_scale, config);
        let formatted = format!("{:.*}", config.decimals, display_val);
        let galley = ui
            .painter()
            .layout_no_wrap(formatted, theme.font(font_size), TEXT_DISABLED);
        let desired = Vec2::new(galley.size().x.max(14.0) + left_inset, 24.0);
        let (rect, _) = ui.allocate_exact_size(desired, Sense::hover());
        ui.painter().galley(
            Pos2::new(
                rect.left() + left_inset,
                rect.center().y - galley.size().y * 0.5,
            ),
            galley,
            TEXT_DISABLED,
        );
        return (ev, None);
    }

    let display_val = clamp_and_round(*value / safe_scale, config);
    let mut mem: ScrubMemory = ui
        .data_mut(|d| d.get_temp::<ScrubMemory>(id))
        .unwrap_or_default();
    let mut ev = ScrubEvent::default();
    let mut text_commit: Option<(f64, bool)> = None;

    match &mut mem.mode {
        ScrubMode::TextEdit {
            buffer,
            initial_display,
            start_value,
            just_entered,
            alt_at_start,
        } => {
            let edit_id = id.with("text_edit");
            let mut commit_edit = false;
            let mut cancel_edit = false;
            let mut step_delta = 0.0_f64;

            if ui.memory(|m| m.has_focus(edit_id)) {
                ui.input(|i| {
                    if i.key_pressed(Key::Escape) {
                        cancel_edit = true;
                    } else if i.key_pressed(Key::Enter) {
                        commit_edit = true;
                    } else if i.key_pressed(Key::ArrowUp) {
                        step_delta += config.step;
                    } else if i.key_pressed(Key::ArrowDown) {
                        step_delta -= config.step;
                    } else if i.key_pressed(Key::PageUp) {
                        step_delta += 10.0 * config.step;
                    } else if i.key_pressed(Key::PageDown) {
                        step_delta -= 10.0 * config.step;
                    }
                });
            }

            if step_delta.abs() > 0.0 {
                let parsed = buffer.trim().parse::<f64>().unwrap_or(*initial_display);
                let stepped = clamp_and_round(parsed + step_delta, config);
                *buffer = format!("{:.*}", config.decimals, stepped);
            }

            let desired_w = (buffer.len().max(5) as f32 * 9.5 + 12.0).clamp(48.0, 140.0);
            let frame_resp = Frame::NONE
                .fill(BG_INPUT)
                .stroke(Stroke::new(1.0, theme.accent()))
                .inner_margin(egui::Margin::symmetric(2, 1))
                .show(ui, |ui| {
                    let resp = ui.add(
                        TextEdit::singleline(buffer)
                            .id(edit_id)
                            .frame(Frame::NONE)
                            .font(theme.font(FONT_BODY))
                            .text_color(theme.value())
                            .desired_width(desired_w),
                    );
                    if *just_entered {
                        resp.request_focus();
                        *just_entered = false;
                    }
                    resp
                });

            let resp = frame_resp.inner;
            if cancel_edit {
                // Never hold an open transaction in TextEdit: revert buffer without emitting cancel/change.
                *buffer = format!("{:.*}", config.decimals, *initial_display);
                ev.changed = false;
                ev.committed = false;
                ev.cancelled = false;
                ui.memory_mut(|m| m.surrender_focus(edit_id));
                mem.mode = ScrubMode::Idle;
            } else if commit_edit || resp.lost_focus() {
                let final_display = buffer
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite())
                    .map_or(*initial_display, |v| clamp_and_round(v, config));
                if (final_display - *initial_display).abs() > 1e-12 {
                    *value = final_display * safe_scale;
                    ev.changed = true;
                    ev.committed = true;
                    ev.started = false;
                    ev.cancelled = false;
                    text_commit = Some((*start_value, *alt_at_start));
                } else {
                    *buffer = format!("{:.*}", config.decimals, *initial_display);
                    ev.changed = false;
                    ev.committed = false;
                    ev.cancelled = false;
                }
                ui.memory_mut(|m| m.surrender_focus(edit_id));
                mem.mode = ScrubMode::Idle;
            }
        }
        ScrubMode::Idle => {
            let formatted = format!("{:.*}", config.decimals, display_val);
            let font = theme.font(font_size);
            let base_galley =
                ui.painter()
                    .layout_no_wrap(formatted.clone(), font.clone(), theme.value());
            let desired = Vec2::new(base_galley.size().x.max(14.0) + left_inset, 24.0);
            let (_, rect) = ui.allocate_space(desired);
            let resp = ui
                .interact(rect, id.with("hit"), Sense::click_and_drag())
                .on_hover_cursor(CursorIcon::ResizeHorizontal);

            // Clean flat Zoidium value text; highlight text on hover.
            let (galley, text_color) = if resp.hovered() {
                (
                    ui.painter()
                        .layout_no_wrap(formatted, font, theme.value_strong()),
                    theme.value_strong(),
                )
            } else {
                (base_galley, theme.value())
            };
            ui.painter().galley(
                Pos2::new(
                    rect.left() + left_inset,
                    rect.center().y - galley.size().y * 0.5,
                ),
                galley,
                text_color,
            );

            if resp.is_pointer_button_down_on()
                && ui.input(|i| i.pointer.primary_down())
                && let Some(pos) = ui.input(|i| i.pointer.interact_pos())
            {
                let alt = ui.input(|i| i.modifiers.alt);
                mem.mode = ScrubMode::PendingDrag {
                    start_pos: pos,
                    last_pos: pos,
                    initial_display: display_val,
                    start_value: *value,
                    accum_display: display_val,
                    alt_at_start: alt,
                };
                ev.started = true;
            }
        }
        ScrubMode::PendingDrag {
            start_pos,
            last_pos,
            initial_display,
            start_value,
            accum_display,
            alt_at_start,
        } => {
            let formatted = format!("{:.*}", config.decimals, display_val);
            let galley = ui.painter().layout_no_wrap(
                formatted.clone(),
                theme.font(font_size),
                theme.value_strong(),
            );
            let desired = Vec2::new(galley.size().x.max(14.0) + left_inset, 24.0);
            let (_, rect) = ui.allocate_space(desired);
            let _resp = ui
                .interact(rect, id.with("hit"), Sense::click_and_drag())
                .on_hover_cursor(CursorIcon::ResizeHorizontal);
            ui.painter().galley(
                Pos2::new(
                    rect.left() + left_inset,
                    rect.center().y - galley.size().y * 0.5,
                ),
                galley,
                theme.value_strong(),
            );
            ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);

            let (down, pos_opt, esc) = ui.input(|i| {
                (
                    i.pointer.primary_down(),
                    i.pointer.latest_pos(),
                    i.key_pressed(Key::Escape),
                )
            });

            if esc {
                *value = *start_value;
                ev.cancelled = true;
                mem.mode = ScrubMode::Idle;
            } else if !down {
                // Released before 2 px threshold -> immediately cancel PendingDrag transaction
                // before entering TextEdit so no session transaction is left open.
                ev.cancelled = true;
                mem.mode = ScrubMode::TextEdit {
                    buffer: formatted,
                    initial_display: *initial_display,
                    start_value: *start_value,
                    just_entered: true,
                    alt_at_start: *alt_at_start,
                };
            } else if let Some(pos) = pos_opt {
                let d2 = (pos.x - start_pos.x) - (pos.y - start_pos.y);
                if d2.abs() > 2.0 {
                    let init = *initial_display;
                    let sv = *start_value;
                    let accum = *accum_display;
                    let alt = *alt_at_start;
                    *last_pos = pos;
                    mem.mode = ScrubMode::Scrubbing {
                        last_pos: pos,
                        initial_display: init,
                        start_value: sv,
                        accum_display: accum,
                        alt_at_start: alt,
                    };
                }
            }
        }
        ScrubMode::Scrubbing {
            last_pos,
            initial_display,
            start_value,
            accum_display,
            ..
        } => {
            let (down, pos_opt, shift, ctrl, esc) = ui.input(|i| {
                (
                    i.pointer.primary_down(),
                    i.pointer.latest_pos(),
                    i.modifiers.shift,
                    i.modifiers.ctrl || i.modifiers.command,
                    i.key_pressed(Key::Escape),
                )
            });

            ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);

            if esc {
                *value = *start_value;
                ev.cancelled = true;
                mem.mode = ScrubMode::Idle;
            } else if let Some(pos) = pos_opt {
                let delta = f64::from((pos.x - last_pos.x) - (pos.y - last_pos.y));
                *last_pos = pos;
                if delta.abs() > 0.0 {
                    let prev_rounded = clamp_and_round(*value / safe_scale, config);
                    let m_shift = if shift { 10.0 } else { 1.0 };
                    let m_ctrl = if ctrl { 0.1 } else { 1.0 };
                    let (lo, hi) = if config.min <= config.max {
                        (config.min, config.max)
                    } else {
                        (config.max, config.min)
                    };
                    *accum_display = (*accum_display
                        + delta * config.step * 0.01 * m_shift * m_ctrl)
                        .clamp(lo, hi);
                    let rounded = round_decimals(*accum_display, config.decimals);
                    *value = rounded * safe_scale;
                    if (rounded - prev_rounded).abs() > 1e-12 {
                        ev.changed = true;
                    }
                }
                if !down {
                    let final_display = clamp_and_round(*accum_display, config);
                    if (final_display - *initial_display).abs() > 1e-12 {
                        *value = final_display * safe_scale;
                        ev.committed = true;
                    } else {
                        *value = *start_value;
                        ev.cancelled = true;
                    }
                    mem.mode = ScrubMode::Idle;
                }
            } else if !down {
                let final_display = clamp_and_round(*accum_display, config);
                if (final_display - *initial_display).abs() > 1e-12 {
                    *value = final_display * safe_scale;
                    ev.committed = true;
                } else {
                    *value = *start_value;
                    ev.cancelled = true;
                }
                mem.mode = ScrubMode::Idle;
            }

            let cur_disp = clamp_and_round(*value / safe_scale, config);
            let formatted = format!("{:.*}", config.decimals, cur_disp);
            let galley =
                ui.painter()
                    .layout_no_wrap(formatted, theme.font(font_size), theme.value_strong());
            let desired = Vec2::new(galley.size().x.max(14.0) + left_inset, 24.0);
            let (_, rect) = ui.allocate_space(desired);
            let _resp = ui
                .interact(rect, id.with("hit"), Sense::click_and_drag())
                .on_hover_cursor(CursorIcon::ResizeHorizontal);
            ui.painter().galley(
                Pos2::new(
                    rect.left() + left_inset,
                    rect.center().y - galley.size().y * 0.5,
                ),
                galley,
                theme.value_strong(),
            );
        }
    }

    ui.data_mut(|d| d.insert_temp(id, mem));
    (ev, text_commit)
}

#[allow(clippy::too_many_arguments)]
fn measure_vector_inline_width(
    ui: &Ui,
    values: &[f64],
    config: ScrubConfig,
    sep: &str,
    theme: &Cm3Theme,
    font_size: f32,
    inset: f32,
    spacing: f32,
) -> f32 {
    let safe_scale = if config.scale_factor.is_finite() && config.scale_factor.abs() > 1e-12 {
        config.scale_factor
    } else {
        1.0
    };
    let font = theme.font(font_size);
    let sep_w = ui
        .painter()
        .layout_no_wrap(sep.to_owned(), font.clone(), TEXT_DISABLED)
        .size()
        .x;
    let mut total = 0.0_f32;
    for (idx, &v) in values.iter().enumerate() {
        if idx > 0 {
            total += spacing + sep_w + spacing;
        }
        let disp = clamp_and_round(v / safe_scale, config);
        let s = format!("{:.*}", config.decimals, disp);
        let w = ui
            .painter()
            .layout_no_wrap(s, font.clone(), theme.value())
            .size()
            .x
            .max(14.0)
            + inset;
        total += w;
    }
    total
}

/// Chooses `(font_size, left_inset, item_spacing)` so all inline vector components fit inside
/// `avail_w` without clipping the rightmost component off the inspector edge.
fn choose_vector_inline_style(
    ui: &Ui,
    values: &[f64],
    config: ScrubConfig,
    sep: &str,
    theme: &Cm3Theme,
) -> (f32, f32, f32) {
    let avail_w = ui.available_width().max(60.0);
    // Prefer exact default styling when everything fits cleanly; step down smoothly for dense
    // Vec3 values or narrow viewports so the rightmost digit never clips off the right border.
    for &(font_size, inset, spacing) in &[
        (FONT_BODY, 2.0, 2.0),
        (14.5, 1.5, 1.5),
        (13.0, 1.0, 1.0),
        (11.5, 0.5, 0.5),
    ] {
        if measure_vector_inline_width(ui, values, config, sep, theme, font_size, inset, spacing)
            <= avail_w - 2.0
        {
            return (font_size, inset, spacing);
        }
    }
    (11.0, 0.5, 0.5)
}

// ---------------------------------------------------------------------------
// 2. Vector2 / Vector3 Inputs with Link Ratio & Alt Temporary Unlink (§4.4)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct VectorLinkMemory {
    active_axis: Option<usize>,
    ratios: [f64; 3],
    initial_values: [f64; 3],
    unlinked: bool,
}

/// Inline CM3 Vector2 control with custom `#999999` separator (`"x"` or `","`) and proportional link ratio.
pub fn cm3_vector2_with_sep(
    ui: &mut Ui,
    id: Id,
    values: &mut [f64; 2],
    cfg: &ScrubConfig,
    link_enabled: bool,
    sep: &str,
    theme: &Cm3Theme,
) -> ScrubEvent {
    let config = *cfg;
    let mut link_mem: VectorLinkMemory = ui
        .data_mut(|d| d.get_temp::<VectorLinkMemory>(id.with("link")))
        .unwrap_or_default();
    let alt_held = ui.input(|i| i.modifiers.alt);
    let mut total = ScrubEvent::default();
    let sep_color = Color32::from_rgb(0x99, 0x99, 0x99);
    let (font_size, left_inset, item_spacing) =
        choose_vector_inline_style(ui, values, config, sep, theme);

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = item_spacing;
        let mut started_axis: Option<usize> = None;
        for axis in 0..2 {
            if axis > 0 {
                ui.label(
                    egui::RichText::new(sep)
                        .font(theme.font(font_size))
                        .color(sep_color),
                );
            }
            let (ev, text_commit) = cm3_scrub_number_styled(
                ui,
                id.with(axis),
                &mut values[axis],
                config,
                theme,
                font_size,
                left_inset,
            );
            if ev.started {
                started_axis = Some(axis);
                link_mem.active_axis = Some(axis);
                link_mem.initial_values[0] = values[0];
                link_mem.initial_values[1] = values[1];
                link_mem.unlinked = alt_held || !link_enabled;
                if !link_mem.unlinked {
                    if values[axis].abs() > 1e-9 {
                        let denom = values[axis];
                        link_mem.ratios[0] = values[0] / denom;
                        link_mem.ratios[1] = values[1] / denom;
                    } else if values.iter().all(|v| v.abs() <= 1e-9) {
                        // Uniform proportional scale when starting from exact zero.
                        link_mem.ratios = [1.0; 3];
                    } else {
                        // Undefined ratio (other axis non-zero while active axis is ~0): unlink cleanly.
                        link_mem.ratios = [1.0; 3];
                        link_mem.unlinked = true;
                    }
                }
            }
            if let Some((start_value, alt_at_start)) = text_commit {
                if ev.committed && link_enabled && !alt_at_start && !alt_held {
                    let other = 1 - axis;
                    if start_value.abs() > 1e-9 {
                        let factor = values[axis] / start_value;
                        values[other] = clamp_stored_bounds(values[other] * factor, config);
                    } else if values[other].abs() <= 1e-9 {
                        values[other] = clamp_stored_bounds(values[axis], config);
                    }
                }
                // If an earlier axis started PendingDrag on this same frame, refresh its snapshot
                // so the new drag begins from the post-TextEdit committed values.
                if ev.committed
                    && let Some(prev_axis) = started_axis
                    && prev_axis != axis
                {
                    link_mem.initial_values[0] = values[0];
                    link_mem.initial_values[1] = values[1];
                    if !link_mem.unlinked {
                        if values[prev_axis].abs() > 1e-9 {
                            let denom = values[prev_axis];
                            link_mem.ratios[0] = values[0] / denom;
                            link_mem.ratios[1] = values[1] / denom;
                        } else if values.iter().all(|v| v.abs() <= 1e-9) {
                            link_mem.ratios = [1.0; 3];
                        } else {
                            link_mem.ratios = [1.0; 3];
                            link_mem.unlinked = true;
                        }
                    }
                    let safe_scale =
                        if config.scale_factor.is_finite() && config.scale_factor.abs() > 1e-12 {
                            config.scale_factor
                        } else {
                            1.0
                        };
                    let new_disp = clamp_and_round(values[prev_axis] / safe_scale, config);
                    ui.data_mut(|d| {
                        if let Some(mut sm) = d.get_temp::<ScrubMemory>(id.with(prev_axis))
                            && let ScrubMode::PendingDrag {
                                initial_display,
                                start_value,
                                accum_display,
                                ..
                            } = &mut sm.mode
                        {
                            *initial_display = new_disp;
                            *start_value = values[prev_axis];
                            *accum_display = new_disp;
                            d.insert_temp(id.with(prev_axis), sm);
                        }
                    });
                }
            } else if (ev.changed || ev.committed)
                && link_mem.active_axis == Some(axis)
                && !link_mem.unlinked
                && link_enabled
            {
                let other = 1 - axis;
                values[other] = clamp_stored_bounds(values[axis] * link_mem.ratios[other], config);
            }
            if ev.cancelled
                && link_mem.active_axis == Some(axis)
                && !link_mem.unlinked
                && link_enabled
            {
                let other = 1 - axis;
                values[other] = link_mem.initial_values[other];
            }
            if (ev.committed || ev.cancelled) && link_mem.active_axis == Some(axis) {
                link_mem.active_axis = None;
            }
            total = total.merge(ev);
        }
    });

    ui.data_mut(|d| d.insert_temp(id.with("link"), link_mem));
    total
}

/// Inline CM3 Vector2 control with `#999999` comma separator and proportional `link_ratio`.
pub fn cm3_vector2(
    ui: &mut Ui,
    id: Id,
    values: &mut [f64; 2],
    config: ScrubConfig,
    link_ratio: bool,
    theme: &Cm3Theme,
) -> ScrubEvent {
    cm3_vector2_with_sep(ui, id, values, &config, link_ratio, ",", theme)
}

/// Inline CM3 Vector3 control with `#999999` comma separators and proportional `link_ratio`.
pub fn cm3_vector3(
    ui: &mut Ui,
    id: Id,
    values: &mut [f64; 3],
    config: ScrubConfig,
    link_ratio: bool,
    theme: &Cm3Theme,
) -> ScrubEvent {
    let mut link_mem: VectorLinkMemory = ui
        .data_mut(|d| d.get_temp::<VectorLinkMemory>(id.with("link")))
        .unwrap_or_default();
    let alt_held = ui.input(|i| i.modifiers.alt);
    let mut total = ScrubEvent::default();
    let avail_w = ui.available_width().max(60.0);
    let (font_size, left_inset, item_spacing) =
        choose_vector_inline_style(ui, values, config, ",", theme);
    let fits_inline = measure_vector_inline_width(
        ui,
        values,
        config,
        ",",
        theme,
        font_size,
        left_inset,
        item_spacing,
    ) <= avail_w - 2.0;
    let (wrap_font, wrap_inset, wrap_spacing) = if fits_inline {
        (font_size, left_inset, item_spacing)
    } else {
        let s0 = choose_vector_inline_style(ui, &values[0..2], config, ",", theme);
        let s1 = choose_vector_inline_style(ui, &values[2..3], config, ",", theme);
        (s0.0.min(s1.0), s0.1.min(s1.1), s0.2.min(s1.2))
    };

    let mut started_axis: Option<usize> = None;
    let mut render_axis = |ui: &mut Ui,
                           axis: usize,
                           show_leading_comma: bool,
                           font_sz: f32,
                           inset: f32| {
        if show_leading_comma {
            ui.label(
                egui::RichText::new(",")
                    .font(theme.font(font_sz))
                    .color(TEXT_DISABLED),
            );
        }
        let (ev, text_commit) = cm3_scrub_number_styled(
            ui,
            id.with(axis),
            &mut values[axis],
            config,
            theme,
            font_sz,
            inset,
        );
        if ev.started {
            started_axis = Some(axis);
            link_mem.active_axis = Some(axis);
            link_mem.initial_values = *values;
            link_mem.unlinked = alt_held || !link_ratio;
            if !link_mem.unlinked {
                if values[axis].abs() > 1e-9 {
                    let denom = values[axis];
                    for (ratio, &val) in link_mem.ratios.iter_mut().zip(values.iter()) {
                        *ratio = val / denom;
                    }
                } else if values.iter().all(|v| v.abs() <= 1e-9) {
                    link_mem.ratios = [1.0; 3];
                } else {
                    link_mem.ratios = [1.0; 3];
                    link_mem.unlinked = true;
                }
            }
        }
        if let Some((start_value, alt_at_start)) = text_commit {
            if ev.committed && link_ratio && !alt_at_start && !alt_held {
                let v = values[axis];
                if start_value.abs() > 1e-9 {
                    let factor = v / start_value;
                    for (other, slot) in values.iter_mut().enumerate() {
                        if other != axis {
                            *slot = clamp_stored_bounds(*slot * factor, config);
                        }
                    }
                } else if values
                    .iter()
                    .enumerate()
                    .all(|(other, val)| other == axis || val.abs() <= 1e-9)
                {
                    for (other, slot) in values.iter_mut().enumerate() {
                        if other != axis {
                            *slot = clamp_stored_bounds(v, config);
                        }
                    }
                }
            }
            // Refresh an earlier axis's same-frame PendingDrag snapshot after a TextEdit commit.
            if ev.committed
                && let Some(prev_axis) = started_axis
                && prev_axis != axis
            {
                link_mem.initial_values = *values;
                if !link_mem.unlinked {
                    if values[prev_axis].abs() > 1e-9 {
                        let denom = values[prev_axis];
                        for (ratio, &val) in link_mem.ratios.iter_mut().zip(values.iter()) {
                            *ratio = val / denom;
                        }
                    } else if values.iter().all(|v| v.abs() <= 1e-9) {
                        link_mem.ratios = [1.0; 3];
                    } else {
                        link_mem.ratios = [1.0; 3];
                        link_mem.unlinked = true;
                    }
                }
                let safe_scale =
                    if config.scale_factor.is_finite() && config.scale_factor.abs() > 1e-12 {
                        config.scale_factor
                    } else {
                        1.0
                    };
                let new_disp = clamp_and_round(values[prev_axis] / safe_scale, config);
                ui.data_mut(|d| {
                    if let Some(mut sm) = d.get_temp::<ScrubMemory>(id.with(prev_axis))
                        && let ScrubMode::PendingDrag {
                            initial_display,
                            start_value,
                            accum_display,
                            ..
                        } = &mut sm.mode
                    {
                        *initial_display = new_disp;
                        *start_value = values[prev_axis];
                        *accum_display = new_disp;
                        d.insert_temp(id.with(prev_axis), sm);
                    }
                });
            }
        } else if (ev.changed || ev.committed)
            && link_mem.active_axis == Some(axis)
            && !link_mem.unlinked
            && link_ratio
        {
            let v = values[axis];
            for (other, (slot, &ratio)) in values.iter_mut().zip(link_mem.ratios.iter()).enumerate()
            {
                if other != axis {
                    *slot = clamp_stored_bounds(v * ratio, config);
                }
            }
        }
        if ev.cancelled && link_mem.active_axis == Some(axis) && !link_mem.unlinked && link_ratio {
            for (other, (slot, &init_val)) in values
                .iter_mut()
                .zip(link_mem.initial_values.iter())
                .enumerate()
            {
                if other != axis {
                    *slot = init_val;
                }
            }
        }
        if (ev.committed || ev.cancelled) && link_mem.active_axis == Some(axis) {
            link_mem.active_axis = None;
        }
        total = total.merge(ev);
    };

    if fits_inline {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = item_spacing;
            for axis in 0..3 {
                render_axis(ui, axis, axis > 0, font_size, left_inset);
            }
        });
    } else {
        // Automatic 2+1 wrap when panel is very narrow (e.g. 220px): (X, Y) on row 1, (Z) on row 2.
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = wrap_spacing;
                render_axis(ui, 0, false, wrap_font, wrap_inset);
                render_axis(ui, 1, true, wrap_font, wrap_inset);
            });
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = wrap_spacing;
                render_axis(ui, 2, false, wrap_font, wrap_inset);
            });
        });
    }

    ui.data_mut(|d| d.insert_temp(id.with("link"), link_mem));
    total
}

/// Adaptive CM3 Vector4 control (`X, Y, Z, W`) with compact inline scaling and automatic
/// 2×2 grid wrap when `ui.available_width()` is narrower than the readable inline width,
/// ensuring all 4 components stay 100% readable without right-edge clipping.
pub fn cm3_vector4(
    ui: &mut Ui,
    id: Id,
    values: &mut [f64; 4],
    config: ScrubConfig,
    theme: &Cm3Theme,
) -> ScrubEvent {
    let avail_w = ui.available_width().max(60.0);
    let inline_fit = [
        (FONT_BODY, 2.0, 2.0),
        (14.5, 1.5, 1.5),
        (13.0, 1.0, 1.0),
        (11.5, 0.5, 1.0),
    ]
    .into_iter()
    .find(|&(font_size, inset, spacing)| {
        measure_vector_inline_width(ui, values, config, ",", theme, font_size, inset, spacing)
            <= avail_w - 2.0
    });

    let mut total = ScrubEvent::default();
    if let Some((font_size, left_inset, item_spacing)) = inline_fit {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = item_spacing;
            for (axis, slot) in values.iter_mut().enumerate() {
                if axis > 0 {
                    ui.label(
                        egui::RichText::new(",")
                            .font(theme.font(font_size))
                            .color(TEXT_DISABLED),
                    );
                }
                let (ev, _) = cm3_scrub_number_styled(
                    ui,
                    id.with(axis),
                    slot,
                    config,
                    theme,
                    font_size,
                    left_inset,
                );
                total = total.merge(ev);
            }
        });
    } else {
        // Automatic 2×2 wrap: (X, Y) on row 1, (Z, W) on row 2 with uniform style.
        let s0 = choose_vector_inline_style(ui, &values[0..2], config, ",", theme);
        let s1 = choose_vector_inline_style(ui, &values[2..4], config, ",", theme);
        let font_size = s0.0.min(s1.0);
        let left_inset = s0.1.min(s1.1);
        let item_spacing = s0.2.min(s1.2);
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            for row in 0..2 {
                let start = row * 2;
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = item_spacing;
                    for offset in 0..2 {
                        let axis = start + offset;
                        if offset > 0 {
                            ui.label(
                                egui::RichText::new(",")
                                    .font(theme.font(font_size))
                                    .color(TEXT_DISABLED),
                            );
                        }
                        let (ev, _) = cm3_scrub_number_styled(
                            ui,
                            id.with(axis),
                            &mut values[axis],
                            config,
                            theme,
                            font_size,
                            left_inset,
                        );
                        total = total.merge(ev);
                    }
                });
            }
        });
    }
    total
}

// ---------------------------------------------------------------------------
// 3. Stopwatch Button & Keyframe Control Bar (Spec 01 §4.2)
// ---------------------------------------------------------------------------

/// Events returned by [`cm3_stopwatch_button`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StopwatchEvent {
    /// Normal click toggled animation on/off.
    pub toggled: bool,
    /// `Alt+Click` toggled expression mode.
    pub alt_expression: bool,
}

/// Exact `16 x 16 px` CM3 stopwatch icon centered in a `21 x 21 px` button
/// (`#cccccc` inactive, theme value accent when animated, amber when expression-driven).
pub fn cm3_stopwatch_button(
    ui: &mut Ui,
    prop_name: &str,
    animated: bool,
    has_expression: bool,
    theme: &Cm3Theme,
) -> StopwatchEvent {
    let _ = prop_name;
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(21.0), Sense::click());
    let resp = resp.on_hover_cursor(CursorIcon::PointingHand);

    let icon_rect = Rect::from_min_size(
        Pos2::new(
            (rect.center().x - 9.0).round(),
            (rect.center().y - 8.0).round(),
        ),
        Vec2::splat(16.0),
    );

    if resp.hovered() {
        ui.painter().rect_stroke(
            rect,
            CornerRadius::ZERO,
            Stroke::new(2.0, theme.accent()),
            StrokeKind::Inside,
        );
    }

    let icon_color = if has_expression {
        Color32::from_rgb(0xe0, 0xa8, 0x4a)
    } else if animated {
        theme.value()
    } else if resp.hovered() {
        TEXT_BRIGHT
    } else {
        TEXT_PRIMARY
    };
    paint_cm3_icon(ui.painter(), icon_rect, Cm3Icon::Stopwatch, icon_color);

    let resp = resp.on_hover_text(if has_expression {
        "toggle animation (expression active; Alt+Click to edit)"
    } else {
        "toggle animation (Alt+Click for expression)"
    });

    let mut ev = StopwatchEvent::default();
    if resp.clicked() {
        if ui.input(|i| i.modifiers.alt) {
            ev.alt_expression = true;
        } else {
            ev.toggled = true;
        }
    }
    ev
}

/// Display inputs for [`cm3_keyframe_bar`].
#[derive(Debug, Clone, Copy)]
pub struct KeyframeBarState {
    pub has_prev: bool,
    pub is_keyed: bool,
    pub has_next: bool,
    pub interpolated: bool,
    /// Raw CM3 16-bit tween field (`low byte = ease 0..=32`, `bit 8 = Bezier`).
    pub tween: u16,
    /// When `true` and Bezier mode is set, displays `"Ez+"` button.
    pub easing_plus_enabled: bool,
    /// Viewport bounds for popup clamping/flipping.
    pub viewport_rect: Rect,
}

/// Actions emitted by [`cm3_keyframe_bar`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KeyframeBarEvent {
    pub prev_kf: bool,
    pub toggle_kf: bool,
    pub next_kf: bool,
    pub toggle_interp: bool,
    pub set_easing: Option<u8>,
    pub open_easing_plus: bool,
}

#[derive(Debug, Clone, Default)]
struct EasingPopupState {
    open: bool,
    highlighted: usize,
}

/// Exact `27 px`-tall CM3 Keyframe Control Bar with 33-easing dropdown popup (Spec 01 §4.2).
pub fn cm3_keyframe_bar(
    ui: &mut Ui,
    id: Id,
    state: KeyframeBarState,
    theme: &Cm3Theme,
) -> KeyframeBarEvent {
    let mut ev = KeyframeBarEvent::default();
    let popup_id = id.with("easing_popup");
    let mut popup_state: EasingPopupState = ui
        .data_mut(|d| d.get_temp::<EasingPopupState>(popup_id))
        .unwrap_or_default();

    let is_bezier = ((state.tween >> 8) & 1) == 1;
    let ease_idx = (state.tween & 0xFF).min(32) as u8;
    let easing = EasingKind::from_cm3_index(ease_idx).unwrap_or(EasingKind::Linear);
    let mut easing_btn_rect = Rect::NOTHING;

    ui.horizontal(|ui| {
        ui.set_min_height(27.0);
        ui.spacing_mut().item_spacing.x = 5.0;

        // 1. Previous Keyframe (17x17)
        let prev_color = if state.has_prev {
            TEXT_PRIMARY
        } else {
            TEXT_FAINT
        };
        if cm3_icon_button_enabled(
            ui,
            Cm3Icon::PrevKf,
            17.0,
            prev_color,
            "previous keyframe",
            state.has_prev,
            theme,
        ) {
            ev.prev_kf = true;
        }

        // 2. Keyframe Diamond Toggle (hollow vs solid)
        let (kf_icon, kf_tip) = if state.is_keyed {
            (Cm3Icon::KeyframeSolid, "delete keyframe")
        } else {
            (Cm3Icon::KeyframeHollow, "add keyframe")
        };
        if cm3_icon_button(ui, kf_icon, 17.0, TEXT_PRIMARY, kf_tip, theme) {
            ev.toggle_kf = true;
        }

        // 3. Next Keyframe (17x17)
        let next_color = if state.has_next {
            TEXT_PRIMARY
        } else {
            TEXT_FAINT
        };
        if cm3_icon_button_enabled(
            ui,
            Cm3Icon::NextKf,
            17.0,
            next_color,
            "next keyframe",
            state.has_next,
            theme,
        ) {
            ev.next_kf = true;
        }

        // 4. Interpolation Toggle & Easing Dropdown (only when keyed & interpolated)
        if state.is_keyed && state.interpolated {
            let (raw_interp_rect, interp_resp) =
                ui.allocate_exact_size(Vec2::new(37.0, 27.0), Sense::click());
            let interp_rect = snap_rect_px(raw_interp_rect);
            let interp_resp = interp_resp
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text(if is_bezier {
                    "interpolation: cubic bezier (click for standard)"
                } else {
                    "interpolation: standard easing (click for bezier)"
                });
            ui.painter()
                .rect_filled(interp_rect, CornerRadius::ZERO, BG_INPUT);
            let interp_border = if interp_resp.hovered() {
                theme.accent()
            } else {
                BORDER_CONTROL
            };
            paint_crisp_border_1px(ui.painter(), interp_rect, interp_border);
            paint_cm3_icon(
                ui.painter(),
                interp_rect.shrink2(Vec2::new(2.0, 2.0)),
                if is_bezier {
                    Cm3Icon::InterpBezier
                } else {
                    Cm3Icon::InterpStandard
                },
                theme.value(),
            );
            if interp_resp.clicked() {
                ev.toggle_interp = true;
            }

            if state.easing_plus_enabled && is_bezier {
                let (raw_ez_rect, ez_resp) =
                    ui.allocate_exact_size(Vec2::new(42.0, 27.0), Sense::click());
                let ez_rect = snap_rect_px(raw_ez_rect);
                let ez_resp = ez_resp
                    .on_hover_cursor(CursorIcon::PointingHand)
                    .on_hover_text("Open Easing+ Bezier Curve Editor");
                ui.painter()
                    .rect_filled(ez_rect, CornerRadius::ZERO, BG_INPUT);
                paint_crisp_border_1px(
                    ui.painter(),
                    ez_rect,
                    if ez_resp.hovered() {
                        theme.accent()
                    } else {
                        BORDER_CONTROL
                    },
                );
                ui.painter().text(
                    ez_rect.center(),
                    Align2::CENTER_CENTER,
                    "Ez+",
                    theme.font(FONT_EASING),
                    theme.value(),
                );
                if ez_resp.clicked() {
                    ev.open_easing_plus = true;
                }
            } else {
                let (raw_e_rect, e_resp) =
                    ui.allocate_exact_size(Vec2::new(37.0, 27.0), Sense::click());
                let e_rect = snap_rect_px(raw_e_rect);
                easing_btn_rect = e_rect;
                let e_resp = e_resp
                    .on_hover_cursor(CursorIcon::PointingHand)
                    .on_hover_text(format!("easing: {}", easing.display_name()));
                ui.painter()
                    .rect_filled(e_rect, CornerRadius::ZERO, BG_INPUT);
                paint_crisp_border_1px(
                    ui.painter(),
                    e_rect,
                    if popup_state.open || e_resp.hovered() {
                        theme.accent()
                    } else {
                        BORDER_CONTROL
                    },
                );
                paint_easing_preview(
                    ui.painter(),
                    e_rect.shrink2(Vec2::new(1.0, 2.0)),
                    easing,
                    theme.value(),
                );
                if e_resp.clicked() {
                    popup_state.open = !popup_state.open;
                    popup_state.highlighted = usize::from(ease_idx);
                }
            }
        } else {
            popup_state.open = false;
        }
    });

    // Floating 33-easing dropdown popup (250px wide, max 300px tall, #202020 bg, 1px #384668)
    if popup_state.open && easing_btn_rect != Rect::NOTHING {
        let popup_size = Vec2::new(250.0, 300.0);
        let pos = clamp_popup_pos(easing_btn_rect, popup_size, state.viewport_rect);

        let mut close_popup = false;
        let mut closed_by_escape = false;
        let mut key_navigated = false;
        let pointer_moved = ui.input(|i| i.pointer.delta() != Vec2::ZERO);
        ui.input(|i| {
            if i.key_pressed(Key::Escape) {
                close_popup = true;
                closed_by_escape = true;
            } else if i.key_pressed(Key::ArrowDown) {
                popup_state.highlighted = (popup_state.highlighted + 1) % EasingKind::ALL_CM3.len();
                key_navigated = true;
            } else if i.key_pressed(Key::ArrowUp) {
                popup_state.highlighted = (popup_state.highlighted + EasingKind::ALL_CM3.len() - 1)
                    % EasingKind::ALL_CM3.len();
                key_navigated = true;
            } else if i.key_pressed(Key::Enter) {
                ev.set_easing = Some(popup_state.highlighted as u8);
                close_popup = true;
            }
        });

        let area_resp = Area::new(popup_id)
            .order(Order::Foreground)
            .fixed_pos(pos)
            .interactable(true)
            .show(ui.ctx(), |ui| {
                Frame::NONE
                    .fill(BG_INPUT)
                    .stroke(Stroke::new(1.0, theme.accent()))
                    .inner_margin(egui::Margin::same(2))
                    .show(ui, |ui| {
                        ui.set_min_width(popup_size.x - 6.0);
                        ui.set_max_width(popup_size.x - 6.0);
                        ScrollArea::vertical()
                            .id_salt(popup_id.with("scroll"))
                            .max_height(popup_size.y - 6.0)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                ui.spacing_mut().item_spacing = Vec2::new(0.0, 1.0);
                                let menu_active_bg = theme.menu_active();
                                let menu_active_fg = theme.contrast_fg(menu_active_bg);
                                for (idx, kind) in EasingKind::ALL_CM3.iter().copied().enumerate() {
                                    let row_size = Vec2::new(ui.available_width(), 27.0);
                                    let (row_rect, row_resp) =
                                        ui.allocate_exact_size(row_size, Sense::click());
                                    if row_resp.hovered() && pointer_moved {
                                        popup_state.highlighted = idx;
                                    }
                                    let is_selected = idx == usize::from(ease_idx);
                                    let is_hi = idx == popup_state.highlighted;
                                    if key_navigated && is_hi {
                                        row_resp.scroll_to_me(None);
                                    }
                                    if is_hi {
                                        ui.painter().rect_filled(
                                            row_rect,
                                            CornerRadius::ZERO,
                                            menu_active_bg,
                                        );
                                    } else if is_selected {
                                        ui.painter().rect_filled(
                                            row_rect,
                                            CornerRadius::ZERO,
                                            BG_DROPDOWN_HOVER,
                                        );
                                    }

                                    let icon_rect = Rect::from_min_size(
                                        Pos2::new(row_rect.left() + 4.0, row_rect.top() + 2.0),
                                        Vec2::new(35.0, 23.0),
                                    );
                                    paint_easing_preview(
                                        ui.painter(),
                                        icon_rect,
                                        kind,
                                        if is_hi {
                                            menu_active_fg
                                        } else if is_selected {
                                            TEXT_BRIGHT
                                        } else {
                                            theme.value()
                                        },
                                    );
                                    ui.painter().with_clip_rect(row_rect).text(
                                        Pos2::new(icon_rect.right() + 8.0, row_rect.center().y),
                                        Align2::LEFT_CENTER,
                                        kind.display_name(),
                                        theme.font(FONT_SMALL),
                                        if is_hi {
                                            menu_active_fg
                                        } else if is_selected {
                                            TEXT_BRIGHT
                                        } else {
                                            TEXT_PRIMARY
                                        },
                                    );

                                    if row_resp.clicked() {
                                        ev.set_easing = Some(idx as u8);
                                        close_popup = true;
                                    }
                                }
                            });
                    });
            });

        // Click outside closes popup
        let clicked_outside = ui.input(|i| {
            i.pointer.primary_pressed()
                && i.pointer.interact_pos().is_some_and(|p| {
                    !area_resp.response.rect.contains(p) && !easing_btn_rect.contains(p)
                })
        });
        if close_popup || clicked_outside {
            popup_state.open = false;
            if closed_by_escape {
                consume_transient_popover_escape(ui.ctx());
            } else {
                clear_transient_popover_open_if_stale(ui.ctx());
            }
        }
    }

    if popup_state.open {
        mark_transient_popover_open(ui.ctx());
    }

    ui.data_mut(|d| d.insert_temp(popup_id, popup_state));
    ev
}

// ---------------------------------------------------------------------------
// 4. HSVA Color Picker + Swatch + Eyedropper (Spec 01 §4.5)
// ---------------------------------------------------------------------------

/// Events emitted by [`cm3_hsva_color_picker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ColorPickerEvent {
    pub started: bool,
    pub changed: bool,
    pub committed: bool,
    pub cancelled: bool,
    pub eyedropper_clicked: bool,
}

#[derive(Debug, Clone)]
struct ColorPopupState {
    open: bool,
    initial: Rgba,
    hsv: Hsv,
    alpha: f32,
    hex_buf: String,
    dirty: bool,
}

fn default_recent_swatches() -> Vec<Rgb> {
    // Rows 0-2 (0..18): 18 full-saturation rainbow hues (0..340 deg in 20 deg steps)
    // Row 3 (18..24): 6 grayscales white -> black
    // Rows 4-5 (24..36): 12 recent/varied seed colors
    let mut swatches = Vec::with_capacity(36);
    for i in 0..18 {
        let h = i as f32 / 18.0;
        swatches.push(Rgb::from_hsv(Hsv { h, s: 1.0, v: 1.0 }));
    }
    for v in [255_u8, 204, 153, 102, 51, 0] {
        swatches.push(Rgb::from_u8(v, v, v));
    }
    let seed_colors = [
        Rgb::from_u8(0xe0, 0x4f, 0x5f),
        Rgb::from_u8(0xf2, 0x8f, 0x3b),
        Rgb::from_u8(0xf7, 0xd0, 0x60),
        Rgb::from_u8(0x70, 0xc2, 0x9d),
        Rgb::from_u8(0x4a, 0x90, 0xe2),
        Rgb::from_u8(0x7e, 0x8f, 0xb9),
        Rgb::from_u8(0x8e, 0x6c, 0xbf),
        Rgb::from_u8(0xd9, 0x6b, 0xa8),
        Rgb::from_u8(0x38, 0x46, 0x68),
        Rgb::from_u8(0x2d, 0x6a, 0x4f),
        Rgb::from_u8(0x9b, 0x22, 0x26),
        Rgb::from_u8(0xee, 0x9b, 0x00),
    ];
    swatches.extend_from_slice(&seed_colors);
    swatches
}

fn push_recent_swatch(ui: &mut Ui, rgb: Rgb) {
    let key = Id::unique("zoidcraft_recent_color_swatches_v2");
    let mut list = ui
        .data_mut(|d| d.get_persisted::<Vec<Rgb>>(key))
        .filter(|v| v.len() == 36)
        .unwrap_or_else(default_recent_swatches);
    let target = rgb.to_u8();
    // Preserve fixed 24-swatch palette (rows 0-3); rotate only the 12 recent slots (24..36).
    let mut recent = list.split_off(24);
    if let Some(pos) = recent.iter().position(|c| c.to_u8() == target) {
        recent.remove(pos);
    }
    recent.insert(0, rgb);
    recent.truncate(12);
    list.extend(recent);
    ui.data_mut(|d| d.insert_persisted(key, list));
}

fn read_recent_swatches(ui: &mut Ui) -> Vec<Rgb> {
    let key = Id::unique("zoidcraft_recent_color_swatches_v2");
    ui.data_mut(|d| d.get_persisted::<Vec<Rgb>>(key))
        .filter(|v| v.len() == 36)
        .unwrap_or_else(default_recent_swatches)
}

fn format_picker_css(c: Rgba, has_alpha: bool) -> String {
    let [r, g, b, _] = c.to_u8();
    if has_alpha {
        let a = (c.a().clamp(0.0, 1.0) * 1000.0).round() / 1000.0;
        let a_str = if (a - a.round()).abs() < 1e-4 {
            format!("{}", a.round() as i32)
        } else {
            let s = format!("{a:.3}");
            s.trim_end_matches('0').trim_end_matches('.').to_owned()
        };
        format!("rgba({r}, {g}, {b}, {a_str})")
    } else {
        format!("rgb({r}, {g}, {b})")
    }
}

fn parse_picker_color(input: &str) -> Option<Rgba> {
    let trimmed = input.trim();
    if let Ok(c) = Rgba::from_hex(trimmed) {
        return Some(c);
    }
    let lower = trimmed.to_ascii_lowercase();
    let inner = lower
        .strip_prefix("rgba(")
        .or_else(|| lower.strip_prefix("rgb("))
        .and_then(|s| s.strip_suffix(')'))?;
    let parts: Vec<f32> = inner
        .split(',')
        .map(|p| p.trim().parse::<f32>().ok())
        .collect::<Option<Vec<f32>>>()?;
    if (parts.len() == 3 || parts.len() == 4) && parts.iter().all(|v| v.is_finite()) {
        let r = (parts[0] / 255.0).clamp(0.0, 1.0);
        let g = (parts[1] / 255.0).clamp(0.0, 1.0);
        let b = (parts[2] / 255.0).clamp(0.0, 1.0);
        let a = parts.get(3).copied().unwrap_or(1.0).clamp(0.0, 1.0);
        return Some(Rgba::new(r, g, b, a));
    }
    None
}

fn paint_checkerboard(painter: &Painter, rect: Rect, cell: f32) {
    let c0 = Color32::from_gray(0x50);
    let c1 = Color32::from_gray(0x30);
    let cols = (rect.width() / cell).ceil() as usize;
    let rows = (rect.height() / cell).ceil() as usize;
    for r in 0..rows {
        for c in 0..cols {
            let x0 = rect.left() + c as f32 * cell;
            let y0 = rect.top() + r as f32 * cell;
            let tile = Rect::from_min_max(
                Pos2::new(x0, y0),
                Pos2::new(
                    (x0 + cell).min(rect.right()),
                    (y0 + cell).min(rect.bottom()),
                ),
            );
            painter.rect_filled(
                tile,
                CornerRadius::ZERO,
                if (r + c) % 2 == 0 { c0 } else { c1 },
            );
        }
    }
}

fn paint_sv_square(painter: &Painter, rect: Rect, hue: f32) {
    let pure = rgb_to_color32(Rgb::from_hsv(Hsv {
        h: hue,
        s: 1.0,
        v: 1.0,
    }));
    // Pass 1: Horizontal saturation gradient (WHITE on left -> pure hue on right).
    let mut sat_mesh = Mesh::default();
    let tl = sat_mesh.vertices.len() as u32;
    sat_mesh.colored_vertex(rect.left_top(), Color32::WHITE);
    let tr = sat_mesh.vertices.len() as u32;
    sat_mesh.colored_vertex(rect.right_top(), pure);
    let br = sat_mesh.vertices.len() as u32;
    sat_mesh.colored_vertex(rect.right_bottom(), pure);
    let bl = sat_mesh.vertices.len() as u32;
    sat_mesh.colored_vertex(rect.left_bottom(), Color32::WHITE);
    sat_mesh.add_triangle(tl, tr, br);
    sat_mesh.add_triangle(tl, br, bl);
    painter.add(Shape::mesh(sat_mesh));

    // Pass 2: Vertical value alpha overlay (TRANSPARENT on top -> opaque BLACK on bottom).
    let mut val_mesh = Mesh::default();
    let tl = val_mesh.vertices.len() as u32;
    val_mesh.colored_vertex(rect.left_top(), Color32::TRANSPARENT);
    let tr = val_mesh.vertices.len() as u32;
    val_mesh.colored_vertex(rect.right_top(), Color32::TRANSPARENT);
    let br = val_mesh.vertices.len() as u32;
    val_mesh.colored_vertex(rect.right_bottom(), Color32::BLACK);
    let bl = val_mesh.vertices.len() as u32;
    val_mesh.colored_vertex(rect.left_bottom(), Color32::BLACK);
    val_mesh.add_triangle(tl, tr, br);
    val_mesh.add_triangle(tl, br, bl);
    painter.add(Shape::mesh(val_mesh));
}

fn paint_hue_bar(painter: &Painter, rect: Rect) {
    let stops = [
        Color32::from_rgb(255, 0, 0),
        Color32::from_rgb(255, 255, 0),
        Color32::from_rgb(0, 255, 0),
        Color32::from_rgb(0, 255, 255),
        Color32::from_rgb(0, 0, 255),
        Color32::from_rgb(255, 0, 255),
        Color32::from_rgb(255, 0, 0),
    ];
    let mut mesh = Mesh::default();
    let seg_h = rect.height() / 6.0;
    for i in 0..6 {
        let y0 = rect.top() + seg_h * i as f32;
        let y1 = rect.top() + seg_h * (i + 1) as f32;
        let tl = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(rect.left(), y0), stops[i]);
        let tr = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(rect.right(), y0), stops[i]);
        let br = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(rect.right(), y1), stops[i + 1]);
        let bl = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(rect.left(), y1), stops[i + 1]);
        mesh.add_triangle(tl, tr, br);
        mesh.add_triangle(tl, br, bl);
    }
    painter.add(Shape::mesh(mesh));
}

fn paint_alpha_bar(painter: &Painter, rect: Rect, rgb: Rgb) {
    paint_checkerboard(painter, rect, 6.0);
    let [r, g, b] = rgb.to_u8();
    let top_c = Color32::from_rgba_unmultiplied(r, g, b, 255);
    let bot_c = Color32::from_rgba_unmultiplied(r, g, b, 0);
    let mut mesh = Mesh::default();
    let tl = mesh.vertices.len() as u32;
    mesh.colored_vertex(rect.left_top(), top_c);
    let tr = mesh.vertices.len() as u32;
    mesh.colored_vertex(rect.right_top(), top_c);
    let br = mesh.vertices.len() as u32;
    mesh.colored_vertex(rect.right_bottom(), bot_c);
    let bl = mesh.vertices.len() as u32;
    mesh.colored_vertex(rect.left_bottom(), bot_c);
    mesh.add_triangle(tl, tr, br);
    mesh.add_triangle(tl, br, bl);
    painter.add(Shape::mesh(mesh));
}

/// CM3 `30x20` swatch + `17x17` eyedropper + `482x262`/`512x262` HSVA popup picker (Spec 01 §4.5).
pub fn cm3_hsva_color_picker(
    ui: &mut Ui,
    id: Id,
    color: &mut Rgba,
    has_alpha: bool,
    read_only: bool,
    viewport_rect: Rect,
    theme: &Cm3Theme,
) -> ColorPickerEvent {
    CM3_ROW_HAS_COLOR_PICKER.with(|f| f.set(true));
    let mut ev = ColorPickerEvent::default();
    let popup_id = id.with("hsva_popup");
    let mut state = ui
        .data_mut(|d| d.get_temp::<ColorPopupState>(popup_id))
        .unwrap_or_else(|| ColorPopupState {
            open: false,
            initial: *color,
            hsv: color.rgb().to_hsv(),
            alpha: color.a(),
            hex_buf: format_picker_css(*color, has_alpha),
            dirty: false,
        });

    // If read_only (e.g. expression toggled on) flips true while the popup is open,
    // immediately close, restore initial color if dirty, and emit cancelled so any open tx closes.
    if read_only && state.open {
        let was_dirty = state.dirty;
        state.open = false;
        if was_dirty {
            *color = state.initial;
            state.hsv = state.initial.rgb().to_hsv();
            state.alpha = state.initial.a();
            state.hex_buf = format_picker_css(*color, has_alpha);
        }
        state.dirty = false;
        ev.cancelled = true;
        ev.changed = was_dirty;
        ui.memory_mut(|m| m.surrender_focus(popup_id.with("hex_input")));
        clear_transient_popover_open_if_stale(ui.ctx());
    }

    let mut swatch_rect = Rect::NOTHING;

    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 9.0;
            let (raw_rect, resp) = ui.allocate_exact_size(
                Vec2::new(30.0, 20.0),
                if read_only {
                    Sense::hover()
                } else {
                    Sense::click()
                },
            );
            let rect = snap_rect_px(raw_rect);
            swatch_rect = rect;
            if has_alpha {
                paint_checkerboard(ui.painter(), rect, 5.0);
            }
            let fill = if has_alpha {
                rgba_to_color32(*color)
            } else {
                rgb_to_color32(color.rgb())
            };
            ui.painter().rect_filled(rect, CornerRadius::ZERO, fill);
            let border = if state.open || (!read_only && resp.hovered()) {
                theme.accent()
            } else {
                BORDER_CONTROL
            };
            paint_crisp_border_1px(ui.painter(), rect, border);

            if !read_only {
                let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
                if resp.clicked() {
                    if !state.open {
                        state.open = true;
                        state.initial = *color;
                        state.hsv = color.rgb().to_hsv();
                        state.alpha = if has_alpha { color.a() } else { 1.0 };
                        state.hex_buf = format_picker_css(*color, has_alpha);
                        state.dirty = false;
                        ev.started = true;
                    } else {
                        state.open = false;
                        clear_transient_popover_open_if_stale(ui.ctx());
                        if state.dirty {
                            push_recent_swatch(ui, color.rgb());
                            ev.committed = true;
                        } else {
                            ev.cancelled = true;
                        }
                    }
                }

                // Eyedropper (17x17 px, 9px right of swatch)
                if cm3_icon_button(
                    ui,
                    Cm3Icon::Eyedropper,
                    17.0,
                    TEXT_PRIMARY,
                    "pick color from viewport",
                    theme,
                ) {
                    ev.eyedropper_clicked = true;
                }
            }
        });
    });

    if state.open && !read_only && swatch_rect != Rect::NOTHING {
        // Exact Zoidium popup size: 1px border + 5px padding = 6px margin per side around 470x250 / 500x250 content.
        let popup_size = Vec2::new(if has_alpha { 512.0 } else { 482.0 }, 262.0);
        let pos = clamp_popup_pos(swatch_rect, popup_size, viewport_rect);
        let mut commit_now = false;
        let mut cancel_now = false;
        let mut hex_enter_handled = false;

        let recent = read_recent_swatches(ui);

        let area_resp = Area::new(popup_id)
            .order(Order::Foreground)
            .fixed_pos(pos)
            .interactable(true)
            .show(ui.ctx(), |ui| {
                let outer_rect = Rect::from_min_size(pos, popup_size);
                let _ = ui.allocate_space(popup_size);
                ui.painter()
                    .rect_filled(outer_rect, CornerRadius::ZERO, BG_PANEL);

                let content_rect = outer_rect.shrink(6.0);
                let mut content_ui = ui.new_child(
                    UiBuilder::new()
                        .id_salt(popup_id.with("content"))
                        .max_rect(content_rect)
                        .layout(egui::Layout::left_to_right(egui::Align::TOP)),
                );
                content_ui.spacing_mut().item_spacing = Vec2::new(5.0, 0.0);

                // 1. 250x250 SV square (borderless on #2a2a2a)
                let (sv_rect, sv_resp) =
                    content_ui.allocate_exact_size(Vec2::splat(250.0), Sense::click_and_drag());
                let sv_rect = snap_rect_px(sv_rect);
                paint_sv_square(content_ui.painter(), sv_rect, state.hsv.h);
                let sv_pressed_or_dragged = sv_resp.clicked()
                    || sv_resp.dragged()
                    || (sv_resp.is_pointer_button_down_on()
                        && content_ui.input(|i| i.pointer.primary_down()));
                if sv_pressed_or_dragged
                    && let Some(p) = sv_resp.interact_pointer_pos().or_else(|| {
                        content_ui
                            .input(|i| i.pointer.interact_pos().or_else(|| i.pointer.latest_pos()))
                    })
                {
                    state.hsv.s = ((p.x - sv_rect.left()) / sv_rect.width()).clamp(0.0, 1.0);
                    state.hsv.v = (1.0 - (p.y - sv_rect.top()) / sv_rect.height()).clamp(0.0, 1.0);
                    let rgb = Rgb::from_hsv(state.hsv);
                    *color = rgb.with_alpha(if has_alpha { state.alpha } else { 1.0 });
                    state.hex_buf = format_picker_css(*color, has_alpha);
                    state.dirty = true;
                    ev.changed = true;
                }
                let reticle = Pos2::new(
                    (sv_rect.left() + state.hsv.s * sv_rect.width()).round(),
                    (sv_rect.top() + (1.0 - state.hsv.v) * sv_rect.height()).round(),
                );
                // Clip reticle strictly inside the SV square so corner positions like
                // rgb(255,255,255) never overflow onto the popup border or inspector above.
                ui.painter().with_clip_rect(sv_rect).circle_stroke(
                    reticle,
                    4.0,
                    Stroke::new(1.0, Color32::from_gray(0xaa)),
                );

                // 2. 25x250 Hue bar (borderless, un-squashed 25x6 1px #aaaaaa cursor centered on hy)
                let (hue_rect, hue_resp) =
                    content_ui.allocate_exact_size(Vec2::new(25.0, 250.0), Sense::click_and_drag());
                let hue_rect = snap_rect_px(hue_rect);
                paint_hue_bar(content_ui.painter(), hue_rect);
                let hue_pressed_or_dragged = hue_resp.clicked()
                    || hue_resp.dragged()
                    || (hue_resp.is_pointer_button_down_on()
                        && content_ui.input(|i| i.pointer.primary_down()));
                if hue_pressed_or_dragged
                    && let Some(p) = hue_resp.interact_pointer_pos().or_else(|| {
                        content_ui
                            .input(|i| i.pointer.interact_pos().or_else(|| i.pointer.latest_pos()))
                    })
                {
                    state.hsv.h = ((p.y - hue_rect.top()) / hue_rect.height()).clamp(0.0, 0.9999);
                    let rgb = Rgb::from_hsv(state.hsv);
                    *color = rgb.with_alpha(if has_alpha { state.alpha } else { 1.0 });
                    state.hex_buf = format_picker_css(*color, has_alpha);
                    state.dirty = true;
                    ev.changed = true;
                }
                let hy = (hue_rect.top() + state.hsv.h * hue_rect.height()).round();
                let h_cursor = Rect::from_min_max(
                    Pos2::new(hue_rect.left(), hy - 3.0),
                    Pos2::new(hue_rect.right(), hy + 3.0),
                );
                paint_crisp_border_1px(
                    &ui.painter().with_clip_rect(hue_rect),
                    h_cursor,
                    Color32::from_gray(0xaa),
                );

                // 3. Optional 25x250 Alpha bar (borderless, un-squashed 25x6 1px #aaaaaa cursor)
                if has_alpha {
                    let (a_rect, a_resp) = content_ui
                        .allocate_exact_size(Vec2::new(25.0, 250.0), Sense::click_and_drag());
                    let a_rect = snap_rect_px(a_rect);
                    paint_alpha_bar(content_ui.painter(), a_rect, Rgb::from_hsv(state.hsv));
                    let alpha_pressed_or_dragged = a_resp.clicked()
                        || a_resp.dragged()
                        || (a_resp.is_pointer_button_down_on()
                            && content_ui.input(|i| i.pointer.primary_down()));
                    if alpha_pressed_or_dragged
                        && let Some(p) = a_resp.interact_pointer_pos().or_else(|| {
                            content_ui.input(|i| {
                                i.pointer.interact_pos().or_else(|| i.pointer.latest_pos())
                            })
                        })
                    {
                        state.alpha =
                            (1.0 - (p.y - a_rect.top()) / a_rect.height()).clamp(0.0, 1.0);
                        *color = Rgb::from_hsv(state.hsv).with_alpha(state.alpha);
                        state.hex_buf = format_picker_css(*color, has_alpha);
                        state.dirty = true;
                        ev.changed = true;
                    }
                    let ay = (a_rect.top() + (1.0 - state.alpha) * a_rect.height()).round();
                    let a_cursor = Rect::from_min_max(
                        Pos2::new(a_rect.left(), ay - 3.0),
                        Pos2::new(a_rect.right(), ay + 3.0),
                    );
                    paint_crisp_border_1px(
                        &ui.painter().with_clip_rect(a_rect),
                        a_cursor,
                        Color32::from_gray(0xaa),
                    );
                }

                // 4. Right panel (185x250 px): 25px preview + 5px gap + 185px 6x6 grid + 14px gap + 21px input
                let right_rect = Rect::from_min_size(
                    Pos2::new(content_ui.cursor().left().round(), content_rect.top()),
                    Vec2::new(185.0, 250.0),
                );
                let mut right_ui = content_ui.new_child(
                    UiBuilder::new()
                        .id_salt(popup_id.with("right_col"))
                        .max_rect(right_rect)
                        .layout(egui::Layout::top_down(egui::Align::LEFT)),
                );
                right_ui.spacing_mut().item_spacing = Vec2::ZERO;

                // Split comparison preview bar (185 x 25 px, borderless)
                let (cmp_rect, cmp_resp) =
                    right_ui.allocate_exact_size(Vec2::new(185.0, 25.0), Sense::click());
                let cmp_rect = snap_rect_px(cmp_rect);
                let mid_x = cmp_rect.center().x.round();
                let left_half =
                    Rect::from_min_max(cmp_rect.left_top(), Pos2::new(mid_x, cmp_rect.bottom()));
                let right_half =
                    Rect::from_min_max(Pos2::new(mid_x, cmp_rect.top()), cmp_rect.right_bottom());
                if has_alpha {
                    paint_checkerboard(right_ui.painter(), cmp_rect, 5.0);
                }
                right_ui.painter().rect_filled(
                    left_half,
                    CornerRadius::ZERO,
                    rgba_to_color32(state.initial),
                );
                right_ui.painter().rect_filled(
                    right_half,
                    CornerRadius::ZERO,
                    rgba_to_color32(*color),
                );
                if cmp_resp.clicked()
                    && cmp_resp
                        .interact_pointer_pos()
                        .is_some_and(|p| left_half.contains(p))
                {
                    *color = state.initial;
                    state.hsv = state.initial.rgb().to_hsv();
                    state.alpha = state.initial.a();
                    state.hex_buf = format_picker_css(*color, has_alpha);
                    state.dirty = false;
                    ev.changed = true;
                }

                // 5 px gap between preview bar and 6x6 swatch grid
                right_ui.add_space(5.0);

                // 36 swatches (6 x 6 grid, 27.5x27.5 alternating 28/27px tiles, exact 4px gaps = 185x185 px)
                let (_, raw_grid_rect) = right_ui.allocate_space(Vec2::new(185.0, 185.0));
                let grid_rect = snap_rect_px(raw_grid_rect);
                for row in 0..6 {
                    let y0 = (grid_rect.top() + row as f32 * 31.5).round();
                    let y1 = (grid_rect.top() + row as f32 * 31.5 + 27.5).round();
                    for col in 0..6 {
                        let idx = row * 6 + col;
                        let sw_rgb = recent.get(idx).copied().unwrap_or(Rgb::BLACK);
                        let x0 = (grid_rect.left() + col as f32 * 31.5).round();
                        let x1 = (grid_rect.left() + col as f32 * 31.5 + 27.5).round();
                        let s_rect = Rect::from_min_max(Pos2::new(x0, y0), Pos2::new(x1, y1));
                        let s_resp = right_ui
                            .interact(s_rect, popup_id.with(("swatch", idx)), Sense::click())
                            .on_hover_cursor(CursorIcon::PointingHand);
                        right_ui.painter().rect_filled(
                            s_rect,
                            CornerRadius::ZERO,
                            rgb_to_color32(sw_rgb),
                        );
                        if s_resp.hovered() {
                            paint_crisp_border_1px(right_ui.painter(), s_rect, TEXT_BRIGHT);
                        }
                        if s_resp.clicked() {
                            state.hsv = sw_rgb.to_hsv();
                            *color = sw_rgb.with_alpha(if has_alpha { state.alpha } else { 1.0 });
                            state.hex_buf = format_picker_css(*color, has_alpha);
                            state.dirty = true;
                            ev.changed = true;
                        }
                    }
                }

                // 14 px gap above bottom CSS rgb()/rgba() input
                right_ui.add_space(14.0);

                // Bottom CSS input (exact 185 x 21 px, FONT_BODY 16px, 4px left padding, flush with SV bottom)
                let sub_control_changed = ev.changed;
                let hex_id = popup_id.with("hex_input");
                let hex_focused_now = right_ui.memory(|m| m.has_focus(hex_id));
                let hex_enter_pressed = right_ui.input(|i| i.key_pressed(Key::Enter));
                let (_, raw_input_rect) = right_ui.allocate_space(Vec2::new(185.0, 21.0));
                let input_rect = snap_rect_px(raw_input_rect);
                right_ui
                    .painter()
                    .rect_filled(input_rect, CornerRadius::ZERO, BG_INPUT);
                paint_crisp_border_1px(
                    right_ui.painter(),
                    input_rect,
                    if hex_focused_now {
                        theme.accent()
                    } else {
                        BORDER_CONTROL
                    },
                );
                let inner_input_rect = Rect::from_min_max(
                    Pos2::new(input_rect.left() + 4.0, input_rect.top() + 1.0),
                    Pos2::new(input_rect.right() - 4.0, input_rect.bottom() - 1.0),
                );
                let hex_resp = {
                    let mut input_ui = right_ui.new_child(
                        UiBuilder::new()
                            .id_salt(popup_id.with("hex_child"))
                            .max_rect(inner_input_rect)
                            .layout(egui::Layout::left_to_right(egui::Align::Center)),
                    );
                    input_ui.set_clip_rect(input_ui.clip_rect().intersect(inner_input_rect));
                    input_ui.add(
                        TextEdit::singleline(&mut state.hex_buf)
                            .id(hex_id)
                            .frame(Frame::NONE)
                            .font(theme.font(FONT_BODY))
                            .text_color(theme.value())
                            .desired_width(inner_input_rect.width()),
                    )
                };
                let enter_in_hex = hex_enter_pressed
                    && (hex_resp.has_focus() || hex_resp.lost_focus() || hex_focused_now);
                if enter_in_hex {
                    hex_enter_handled = true;
                    if let Some(parsed) = parse_picker_color(&state.hex_buf) {
                        *color = if has_alpha {
                            parsed
                        } else {
                            parsed.rgb().with_alpha(1.0)
                        };
                        state.hsv = color.rgb().to_hsv();
                        state.alpha = color.a();
                        state.hex_buf = format_picker_css(*color, has_alpha);
                        state.dirty = true;
                        ev.changed = true;
                        right_ui.memory_mut(|m| m.surrender_focus(hex_id));
                    } else {
                        // Invalid color text: keep focus in the field and do NOT close popup.
                        hex_resp.request_focus();
                    }
                } else if hex_resp.lost_focus() && !sub_control_changed {
                    if let Some(parsed) = parse_picker_color(&state.hex_buf) {
                        *color = if has_alpha {
                            parsed
                        } else {
                            parsed.rgb().with_alpha(1.0)
                        };
                        state.hsv = color.rgb().to_hsv();
                        state.alpha = color.a();
                        state.hex_buf = format_picker_css(*color, has_alpha);
                        state.dirty = true;
                        ev.changed = true;
                    } else {
                        state.hex_buf = format_picker_css(*color, has_alpha);
                    }
                }

                // Crisp 1 px outer popup border (#384668 / theme.accent())
                paint_crisp_border_1px(ui.painter(), outer_rect, theme.accent());
            });

        let hex_focused = ui.memory(|m| m.has_focus(popup_id.with("hex_input")));
        ui.input(|i| {
            if i.key_pressed(Key::Escape) {
                cancel_now = true;
            } else if (!hex_focused && !hex_enter_handled && i.key_pressed(Key::Enter))
                || (i.pointer.primary_pressed()
                    && i.pointer.interact_pos().is_some_and(|p| {
                        !area_resp.response.rect.contains(p) && !swatch_rect.contains(p)
                    }))
            {
                commit_now = true;
            }
        });

        if cancel_now {
            *color = state.initial;
            state.open = false;
            ev.cancelled = true;
            ui.memory_mut(|m| m.surrender_focus(popup_id.with("hex_input")));
            consume_transient_popover_escape(ui.ctx());
        } else if commit_now {
            state.open = false;
            ui.memory_mut(|m| m.surrender_focus(popup_id.with("hex_input")));
            clear_transient_popover_open_if_stale(ui.ctx());
            if state.dirty {
                push_recent_swatch(ui, color.rgb());
                ev.committed = true;
            } else {
                ev.cancelled = true;
            }
        }
    }

    if state.open && !read_only {
        mark_transient_popover_open(ui.ctx());
    }

    ui.data_mut(|d| d.insert_temp(popup_id, state));
    ev
}

// ---------------------------------------------------------------------------
// 5. Gradient Bar Control (Spec 01 §4.6)
// ---------------------------------------------------------------------------

/// Events emitted by [`cm3_gradient_bar`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GradientBarEvent {
    pub changed: bool,
    pub committed: bool,
    pub cancelled: bool,
}

#[derive(Debug, Clone, Default)]
struct GradientBarState {
    selected_stop: usize,
    dragging_stop: Option<usize>,
    drag_dirty: bool,
}

fn parse_css_rgba(css: &str) -> Rgba {
    let trimmed = css.trim();
    if let Ok(c) = Rgba::from_hex(trimmed) {
        return c;
    }
    if let Some(inner) = trimmed
        .strip_prefix("rgba(")
        .or_else(|| trimmed.strip_prefix("rgb("))
        .and_then(|s| s.strip_suffix(')'))
    {
        let parts: Vec<f32> = inner
            .split(',')
            .filter_map(|p| p.trim().parse::<f32>().ok())
            .collect();
        if parts.len() >= 3 {
            let r = (parts[0] / 255.0).clamp(0.0, 1.0);
            let g = (parts[1] / 255.0).clamp(0.0, 1.0);
            let b = (parts[2] / 255.0).clamp(0.0, 1.0);
            let a = parts.get(3).copied().unwrap_or(1.0).clamp(0.0, 1.0);
            return Rgba::new(r, g, b, a);
        }
    }
    Rgba::WHITE
}

fn format_css_rgba(c: Rgba) -> String {
    let [r, g, b, _] = c.to_u8();
    format!("rgba({r},{g},{b},{:.3})", c.a().clamp(0.0, 1.0))
}

fn sample_gradient(stops: &[GradientStop], t: f64) -> Rgba {
    if stops.is_empty() {
        return Rgba::WHITE;
    }
    let mut sorted: Vec<(f64, Rgba)> = stops
        .iter()
        .map(|s| (s.position.clamp(0.0, 1.0), parse_css_rgba(&s.color)))
        .collect();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

    if t <= sorted[0].0 {
        return sorted[0].1;
    }
    if let Some(last) = sorted.last()
        && t >= last.0
    {
        return last.1;
    }
    for pair in sorted.windows(2) {
        let (p0, c0) = pair[0];
        let (p1, c1) = pair[1];
        if t >= p0 && t <= p1 {
            let span = (p1 - p0).max(1e-6);
            let u = ((t - p0) / span) as f32;
            let inv = 1.0 - u;
            return Rgba::new(
                c0.0[0] * inv + c1.0[0] * u,
                c0.0[1] * inv + c1.0[1] * u,
                c0.0[2] * inv + c1.0[2] * u,
                c0.0[3] * inv + c1.0[3] * u,
            );
        }
    }
    sorted[0].1
}

/// Exact CM3 `245 px` gradient bar (`30 px` preview + `15 px` stop lane, Spec 01 §4.6).
pub fn cm3_gradient_bar(
    ui: &mut Ui,
    id: Id,
    stops: &mut Vec<GradientStop>,
    viewport_rect: Rect,
    theme: &Cm3Theme,
) -> GradientBarEvent {
    if stops.is_empty() {
        *stops = GradientStop::white_default();
    }

    let mut ev = GradientBarEvent::default();
    let mut state: GradientBarState = ui
        .data_mut(|d| d.get_temp::<GradientBarState>(id))
        .unwrap_or_default();
    if state.selected_stop >= stops.len() {
        state.selected_stop = stops.len().saturating_sub(1);
    }
    let prev_selected = state.selected_stop;
    let mut stop_changed = false;

    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 3.0;
        let width = ui.available_width().clamp(160.0, 245.0);

        // 30px gradient preview bar
        let (bar_rect, bar_resp) = ui.allocate_exact_size(Vec2::new(width, 30.0), Sense::click());
        paint_checkerboard(ui.painter(), bar_rect, 6.0);

        // Render smooth 24-segment gradient mesh
        const SEGS: usize = 24;
        let mut mesh = Mesh::default();
        for i in 0..SEGS {
            let t0 = i as f64 / SEGS as f64;
            let t1 = (i + 1) as f64 / SEGS as f64;
            let c0 = rgba_to_color32(sample_gradient(stops, t0));
            let c1 = rgba_to_color32(sample_gradient(stops, t1));
            let x0 = bar_rect.left() + bar_rect.width() * (t0 as f32);
            let x1 = bar_rect.left() + bar_rect.width() * (t1 as f32);
            let tl = mesh.vertices.len() as u32;
            mesh.colored_vertex(Pos2::new(x0, bar_rect.top()), c0);
            let tr = mesh.vertices.len() as u32;
            mesh.colored_vertex(Pos2::new(x1, bar_rect.top()), c1);
            let br = mesh.vertices.len() as u32;
            mesh.colored_vertex(Pos2::new(x1, bar_rect.bottom()), c1);
            let bl = mesh.vertices.len() as u32;
            mesh.colored_vertex(Pos2::new(x0, bar_rect.bottom()), c0);
            mesh.add_triangle(tl, tr, br);
            mesh.add_triangle(tl, br, bl);
        }
        ui.painter().add(Shape::mesh(mesh));
        ui.painter().rect_stroke(
            bar_rect,
            CornerRadius::ZERO,
            Stroke::new(1.0, BORDER_CONTROL),
            StrokeKind::Inside,
        );

        // Click preview bar to insert interpolated stop
        if bar_resp.clicked()
            && let Some(pos) = bar_resp.interact_pointer_pos()
        {
            let t = f64::from(((pos.x - bar_rect.left()) / bar_rect.width()).clamp(0.0, 1.0));
            let sampled = sample_gradient(stops, t);
            stops.push(GradientStop {
                position: t,
                color: format_css_rgba(sampled),
            });
            state.selected_stop = stops.len() - 1;
            stop_changed = true;
            ev.changed = true;
            ev.committed = true;
        }

        // 15px stop lane with 10x10 draggable stop squares
        let (lane_rect, _) = ui.allocate_exact_size(Vec2::new(width, 15.0), Sense::hover());
        let mut remove_idx = None;
        for (idx, stop) in stops.iter_mut().enumerate() {
            let cx = lane_rect.left() + (stop.position.clamp(0.0, 1.0) as f32) * lane_rect.width();
            let stop_rect =
                Rect::from_center_size(Pos2::new(cx, lane_rect.center().y), Vec2::splat(10.0));
            let s_resp = ui.interact(stop_rect, id.with(("stop", idx)), Sense::click_and_drag());
            let fill = rgba_to_color32(parse_css_rgba(&stop.color));
            ui.painter()
                .rect_filled(stop_rect, CornerRadius::ZERO, fill);
            let border = if idx == state.selected_stop {
                theme.accent_strong()
            } else {
                BORDER_CONTROL
            };
            ui.painter().rect_stroke(
                stop_rect,
                CornerRadius::ZERO,
                Stroke::new(1.0, border),
                StrokeKind::Inside,
            );

            if s_resp.clicked() || s_resp.drag_started() {
                state.selected_stop = idx;
            }
            if s_resp.drag_started() {
                state.dragging_stop = Some(idx);
                state.drag_dirty = false;
            }
            if s_resp.dragged()
                && state.dragging_stop == Some(idx)
                && let Some(p) = ui.input(|i| i.pointer.interact_pos())
            {
                let t = f64::from(((p.x - lane_rect.left()) / lane_rect.width()).clamp(0.0, 1.0));
                if (stop.position - t).abs() > 1e-4 {
                    stop.position = t;
                    state.drag_dirty = true;
                    ev.changed = true;
                }
            }
            if s_resp.drag_stopped() && state.dragging_stop == Some(idx) {
                state.dragging_stop = None;
                if state.drag_dirty {
                    state.drag_dirty = false;
                    ev.committed = true;
                }
            }
            if s_resp.secondary_clicked() {
                remove_idx = Some(idx);
            }
        }

        if let Some(idx) = remove_idx
            && stops.len() > 1
        {
            stops.remove(idx);
            if idx < state.selected_stop {
                state.selected_stop -= 1;
            } else if state.selected_stop >= stops.len() {
                state.selected_stop = stops.len() - 1;
            }
            state.dragging_stop = None;
            state.drag_dirty = false;
            stop_changed = true;
            ev.changed = true;
            ev.committed = true;
        }

        if state.selected_stop != prev_selected {
            stop_changed = true;
        }

        // Inline color editor for selected stop (single picker ID; commit & close popup when stop changes)
        let stop_picker_id = id.with("stop_color");
        if stop_changed {
            let stop_popup_id = stop_picker_id.with("hsva_popup");
            let mut was_open = false;
            let mut was_open_dirty = false;
            ui.data_mut(|d| {
                if let Some(mut st) = d.get_temp::<ColorPopupState>(stop_popup_id)
                    && st.open
                {
                    was_open = true;
                    was_open_dirty = st.dirty;
                    st.open = false;
                    st.dirty = false;
                    d.insert_temp(stop_popup_id, st);
                }
            });
            if was_open_dirty {
                ev.committed = true;
            } else if was_open && !ev.changed && !ev.committed {
                ev.cancelled = true;
            }
        }
        if let Some(sel) = stops.get_mut(state.selected_stop) {
            let mut stop_rgba = parse_css_rgba(&sel.color);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("stop {}", state.selected_stop + 1))
                        .font(theme.font(FONT_SMALL))
                        .color(TEXT_MUTED),
                );
                let cev = cm3_hsva_color_picker(
                    ui,
                    stop_picker_id,
                    &mut stop_rgba,
                    true,
                    false,
                    viewport_rect,
                    theme,
                );
                if cev.cancelled {
                    sel.color = format_css_rgba(stop_rgba);
                    ev.changed = true;
                    ev.cancelled = true;
                    ev.committed = false;
                } else if cev.changed || cev.committed {
                    sel.color = format_css_rgba(stop_rgba);
                    ev.changed = true;
                    if cev.committed {
                        ev.committed = true;
                    }
                }
            });
        }
    });

    ui.data_mut(|d| d.insert_temp(id, state));
    ev
}

// ---------------------------------------------------------------------------
// 6. Buttons, Icon Buttons, Dropdown & Collapsible Section (§4.6 & §4.7)
// ---------------------------------------------------------------------------

/// Visual variant for [`cm3_prop_button`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ButtonVariant {
    #[default]
    Normal,
    Primary,
    Danger,
    Back,
}

/// Full-width or inline CM3 `.propbutton` (`#434343` bg, `2px solid #545454` hover, `#384668` focus).
pub fn cm3_prop_button(
    ui: &mut Ui,
    label: &str,
    variant: ButtonVariant,
    full_width: bool,
    theme: &Cm3Theme,
) -> Response {
    let font = theme.font(FONT_BODY);
    let measure = ui
        .painter()
        .layout_no_wrap(label.to_owned(), font.clone(), TEXT_BUTTON);
    let desired_w = if full_width {
        ui.available_width().max(24.0)
    } else {
        (measure.size().x + 18.0).min(ui.available_width().max(24.0))
    };
    let desired = Vec2::new(desired_w, (measure.size().y + 10.0).max(28.0));
    let (rect, resp) = ui.allocate_exact_size(desired, Sense::click());
    let resp = resp.on_hover_cursor(CursorIcon::PointingHand);

    let bg = if resp.is_pointer_button_down_on() {
        match variant {
            ButtonVariant::Primary => theme.accent_strong(),
            _ => BG_BUTTON_PRESSED,
        }
    } else {
        match variant {
            ButtonVariant::Primary => theme.accent(),
            ButtonVariant::Back => BG_BACK_BUTTON,
            ButtonVariant::Normal | ButtonVariant::Danger => BG_BUTTON,
        }
    };

    let border = if resp.has_focus() {
        Stroke::new(2.0, theme.accent())
    } else if resp.hovered() {
        Stroke::new(2.0, BORDER_BUTTON_HOVER)
    } else {
        Stroke::NONE
    };

    ui.painter().rect_filled(rect, CornerRadius::ZERO, bg);
    if border != Stroke::NONE {
        ui.painter()
            .rect_stroke(rect, CornerRadius::ZERO, border, StrokeKind::Inside);
    }

    let text_color = match variant {
        ButtonVariant::Primary => theme.contrast_fg(bg),
        ButtonVariant::Danger => DANGER_TEXT,
        ButtonVariant::Normal | ButtonVariant::Back => {
            if resp.hovered() {
                TEXT_BRIGHT
            } else {
                TEXT_BUTTON
            }
        }
    };
    let galley = if text_color == TEXT_BUTTON {
        measure
    } else {
        ui.painter()
            .layout_no_wrap(label.to_owned(), font, text_color)
    };
    let text_x = if galley.size().x + 8.0 <= rect.width() {
        rect.center().x - galley.size().x * 0.5
    } else {
        rect.left() + 6.0
    };
    ui.painter()
        .with_clip_rect(rect.shrink2(Vec2::new(4.0, 1.0)))
        .galley(
            Pos2::new(text_x, rect.center().y - galley.size().y * 0.5),
            galley,
            text_color,
        );

    resp
}

fn cm3_icon_button_enabled(
    ui: &mut Ui,
    icon: Cm3Icon,
    icon_px: f32,
    color: Color32,
    tooltip: &str,
    enabled: bool,
    theme: &Cm3Theme,
) -> bool {
    let box_px = icon_px + 4.0;
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(box_px), sense);
    let resp = if enabled {
        resp.on_hover_cursor(CursorIcon::PointingHand)
    } else {
        resp
    };
    if enabled && resp.hovered() {
        ui.painter().rect_stroke(
            rect,
            CornerRadius::ZERO,
            Stroke::new(2.0, theme.accent()),
            StrokeKind::Inside,
        );
    }
    let icon_rect = Rect::from_min_size(
        Pos2::new(
            (rect.center().x - icon_px * 0.5).round(),
            (rect.center().y - icon_px * 0.5).round(),
        ),
        Vec2::splat(icon_px),
    );
    paint_cm3_icon(ui.painter(), icon_rect, icon, color);
    let resp = resp.on_hover_text(tooltip);
    enabled && resp.clicked()
}

/// Compact CM3 action icon button (`17x17` or toolbar `25x25`) with `2px` focus/hover border.
pub fn cm3_icon_button(
    ui: &mut Ui,
    icon: Cm3Icon,
    icon_px: f32,
    color: Color32,
    tooltip: &str,
    theme: &Cm3Theme,
) -> bool {
    cm3_icon_button_enabled(ui, icon, icon_px, color, tooltip, true, theme)
}

#[derive(Debug, Clone, Default)]
struct DropdownPopupState {
    open: bool,
    highlighted: usize,
    last_option_count: usize,
    open_epoch: u64,
}

/// Truncates `text` with a trailing ellipsis (`…`) so it fits strictly within `max_width`.
fn layout_truncated_dropdown_label(
    painter: &Painter,
    text: &str,
    font: egui::FontId,
    color: Color32,
    max_width: f32,
) -> std::sync::Arc<egui::Galley> {
    if max_width <= 6.0 {
        return painter.layout_no_wrap(String::new(), font, color);
    }
    let chars: Vec<char> = text.chars().collect();
    let mut lo = 0usize;
    let mut hi = chars.len();
    let mut best = painter.layout_no_wrap("…".to_owned(), font.clone(), color);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        let mut candidate: String = chars[..mid].iter().collect();
        candidate.push('…');
        let g = painter.layout_no_wrap(candidate, font.clone(), color);
        if g.size().x <= max_width {
            best = g;
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    best
}

/// Exact CM3 `.pz-inputbox` dropdown select (`#202020` bg, `1px solid #1b1b1b`, `#7e8fb9` text).
pub fn cm3_dropdown(
    ui: &mut Ui,
    id: Id,
    selected_idx: &mut usize,
    options: &[&str],
    read_only: bool,
    viewport_rect: Rect,
    theme: &Cm3Theme,
) -> bool {
    CM3_ROW_HAS_DROPDOWN.with(|f| f.set(true));
    if options.is_empty() {
        return false;
    }
    if *selected_idx >= options.len() {
        *selected_idx = 0;
    }

    let current_label = options[*selected_idx];
    if read_only {
        ui.label(
            egui::RichText::new(current_label)
                .font(theme.font(FONT_BODY))
                .color(TEXT_DISABLED),
        );
        return false;
    }

    let popup_id = id.with("dropdown_popup");
    let mut popup_state = ui
        .data_mut(|d| d.get_temp::<DropdownPopupState>(popup_id))
        .unwrap_or(DropdownPopupState {
            open: false,
            highlighted: *selected_idx,
            last_option_count: options.len(),
            open_epoch: 0,
        });
    if popup_state.last_option_count != options.len() {
        popup_state.last_option_count = options.len();
        popup_state.open_epoch = popup_state.open_epoch.wrapping_add(1);
    }
    let mut changed = false;

    let font = theme.font(13.3);
    let mut max_opt_w = 0.0_f32;
    let mut current_galley = None;
    for (idx, opt) in options.iter().enumerate() {
        let galley = ui
            .painter()
            .layout_no_wrap((*opt).to_owned(), font.clone(), theme.value());
        max_opt_w = max_opt_w.max(galley.size().x);
        if idx == *selected_idx {
            current_galley = Some(galley);
        }
    }
    let label_galley = current_galley.unwrap_or_else(|| {
        ui.painter()
            .layout_no_wrap(current_label.to_owned(), font.clone(), theme.value())
    });
    let max_w = ui.available_width().max(46.0);
    // Reserve 5px left inset + 23px right chrome (28px total) when room exists so selected
    // labels never crowd or clip against the chevron.
    let width = (max_opt_w + 28.0).clamp(46.0, max_w).round();
    let (_, raw_slot_rect) = ui.allocate_space(Vec2::new(width, 23.0));
    let btn_rect = snap_rect_px(Rect::from_min_size(
        Pos2::new(raw_slot_rect.left(), raw_slot_rect.top() + 2.0),
        Vec2::new(width, 21.0),
    ));
    let btn_resp = ui
        .interact(btn_rect, id.with("btn"), Sense::click())
        .on_hover_cursor(CursorIcon::PointingHand);

    ui.painter()
        .rect_filled(btn_rect, CornerRadius::ZERO, BG_INPUT);
    paint_crisp_border_1px(
        ui.painter(),
        btn_rect,
        if popup_state.open || btn_resp.hovered() {
            theme.accent()
        } else {
            BORDER_CONTROL
        },
    );

    let label_left = btn_rect.left() + 5.0;
    let label_right = (btn_rect.right() - 19.0).max(label_left);
    let label_avail_w = (label_right - label_left).max(0.0);
    let rendered_galley = if label_galley.size().x <= label_avail_w + 0.5 {
        label_galley
    } else {
        layout_truncated_dropdown_label(
            ui.painter(),
            current_label,
            font.clone(),
            theme.value(),
            label_avail_w,
        )
    };
    let label_clip = ui.clip_rect().intersect(Rect::from_min_max(
        btn_rect.left_top(),
        Pos2::new(label_right, btn_rect.bottom()),
    ));
    ui.painter().with_clip_rect(label_clip).galley(
        Pos2::new(
            label_left,
            (btn_rect.center().y - rendered_galley.size().y * 0.5).round(),
        ),
        rendered_galley,
        theme.value(),
    );

    let caret_rect = Rect::from_center_size(
        Pos2::new(btn_rect.right() - 9.0, btn_rect.center().y),
        Vec2::splat(11.0),
    );
    paint_cm3_icon(
        ui.painter(),
        caret_rect,
        Cm3Icon::ChevronDown,
        theme.value(),
    );

    if btn_resp.clicked() {
        popup_state.open = !popup_state.open;
        popup_state.highlighted = *selected_idx;
        if popup_state.open {
            popup_state.open_epoch = popup_state.open_epoch.wrapping_add(1);
            popup_state.last_option_count = options.len();
        }
    }

    if popup_state.open {
        popup_state.highlighted = popup_state.highlighted.min(options.len().saturating_sub(1));
        let row_h = 24.0;
        let total_rows_h = options.len() as f32 * row_h;
        let needs_scroll = total_rows_h > 276.0;
        let inner_h = total_rows_h.min(276.0);
        let popup_h = inner_h + 4.0;
        let scrollbar_pad = if needs_scroll { 14.0 } else { 0.0 };
        let max_popup_w = (viewport_rect.width() - 8.0).max(width).min(280.0);
        let popup_w = (max_opt_w + 22.0 + scrollbar_pad)
            .max(width)
            .min(max_popup_w)
            .round();
        let popup_size = Vec2::new(popup_w, popup_h);
        let pos = clamp_popup_pos(btn_rect, popup_size, viewport_rect);
        let mut close_now = false;
        let mut closed_by_escape = false;
        let mut key_navigated = false;
        let pointer_moved = ui.input(|i| i.pointer.delta() != Vec2::ZERO);

        ui.input(|i| {
            if i.key_pressed(Key::Escape) {
                close_now = true;
                closed_by_escape = true;
            } else if i.key_pressed(Key::ArrowDown) {
                popup_state.highlighted = (popup_state.highlighted + 1) % options.len();
                key_navigated = true;
            } else if i.key_pressed(Key::ArrowUp) {
                popup_state.highlighted =
                    (popup_state.highlighted + options.len() - 1) % options.len();
                key_navigated = true;
            } else if i.key_pressed(Key::Enter) {
                if *selected_idx != popup_state.highlighted {
                    *selected_idx = popup_state.highlighted;
                    changed = true;
                }
                close_now = true;
            }
        });

        // Key the popup Area by (open_epoch, options.len()) so egui never reuses stale
        // cached Area/ScrollArea geometry when dynamic option lists grow or shrink mid-session.
        let area_id = popup_id.with(("area", popup_state.open_epoch, options.len()));
        let area_resp = Area::new(area_id)
            .order(Order::Foreground)
            .fixed_pos(pos)
            .interactable(true)
            .show(ui.ctx(), |ui| {
                let inner_w = (popup_size.x - 4.0).max(20.0);
                let frame_resp = Frame::NONE
                    .fill(BG_INPUT)
                    .inner_margin(egui::Margin::same(2))
                    .show(ui, |ui| {
                        ui.set_min_width(inner_w);
                        ui.set_max_width(inner_w);
                        ScrollArea::vertical()
                            .id_salt(area_id.with("scroll"))
                            .min_scrolled_height(inner_h)
                            .max_height(inner_h)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                ui.spacing_mut().item_spacing = Vec2::ZERO;
                                let menu_active_bg = theme.menu_active();
                                let menu_active_fg = theme.contrast_fg(menu_active_bg);
                                for (idx, opt) in options.iter().enumerate() {
                                    let (r_rect, r_resp) = ui.allocate_exact_size(
                                        Vec2::new(ui.available_width(), row_h),
                                        Sense::click(),
                                    );
                                    if r_resp.hovered() && pointer_moved {
                                        popup_state.highlighted = idx;
                                    }
                                    let selected = idx == *selected_idx;
                                    let is_hi = idx == popup_state.highlighted;
                                    if key_navigated && is_hi {
                                        r_resp.scroll_to_me(None);
                                    }
                                    if is_hi {
                                        ui.painter().rect_filled(
                                            r_rect,
                                            CornerRadius::ZERO,
                                            menu_active_bg,
                                        );
                                    } else if selected {
                                        ui.painter().rect_filled(
                                            r_rect,
                                            CornerRadius::ZERO,
                                            BG_DROPDOWN_HOVER,
                                        );
                                    }
                                    ui.painter()
                                        .with_clip_rect(r_rect.shrink2(Vec2::new(2.0, 0.0)))
                                        .text(
                                            Pos2::new(r_rect.left() + 4.0, r_rect.center().y),
                                            Align2::LEFT_CENTER,
                                            *opt,
                                            font.clone(),
                                            if is_hi {
                                                menu_active_fg
                                            } else if selected {
                                                TEXT_BRIGHT
                                            } else {
                                                TEXT_PRIMARY
                                            },
                                        );
                                    if r_resp.clicked() {
                                        if *selected_idx != idx {
                                            *selected_idx = idx;
                                            changed = true;
                                        }
                                        close_now = true;
                                    }
                                }
                            });
                    });
                paint_crisp_border_1px(ui.painter(), frame_resp.response.rect, theme.accent());
            });

        let outside = ui.input(|i| {
            i.pointer.primary_pressed()
                && i.pointer
                    .interact_pos()
                    .is_some_and(|p| !area_resp.response.rect.contains(p) && !btn_rect.contains(p))
        });
        if close_now || outside {
            popup_state.open = false;
            if closed_by_escape {
                consume_transient_popover_escape(ui.ctx());
            } else {
                clear_transient_popover_open_if_stale(ui.ctx());
            }
        }
    }

    if popup_state.open {
        mark_transient_popover_open(ui.ctx());
    }

    ui.data_mut(|d| d.insert_temp(popup_id, popup_state));
    changed
}

/// Crisp `7x7 px` framed square collapse glyph (`1px` hollow square with a center
/// `-` dash when open or `+` when collapsed) matching Zoidium tree headers.
pub fn paint_cm3_tree_box(painter: &Painter, center: Pos2, open: bool, color: Color32) {
    let x0 = (center.x - 3.5).round();
    let y0 = (center.y - 3.5).round();
    let outer = Rect::from_min_max(Pos2::new(x0, y0), Pos2::new(x0 + 7.0, y0 + 7.0));
    paint_crisp_border_1px(painter, outer, color);
    // Horizontal center dash (3x1 px)
    painter.rect_filled(
        Rect::from_min_max(Pos2::new(x0 + 2.0, y0 + 3.0), Pos2::new(x0 + 5.0, y0 + 4.0)),
        CornerRadius::ZERO,
        color,
    );
    if !open {
        // Vertical center stem (1x3 px)
        painter.rect_filled(
            Rect::from_min_max(Pos2::new(x0 + 3.0, y0 + 2.0), Pos2::new(x0 + 4.0, y0 + 5.0)),
            CornerRadius::ZERO,
            color,
        );
    }
}

/// CM3 collapsible section (`27 px` root / `28 px` nested or action header, `16 px` body title, `7x7` framed square collapse glyph)
/// with automatic tree nesting indents (`+15 px` per level), optional `nested` surface styling,
/// and Column-2-aligned action icon (e.g. Reset / `+`).
/// Returns `(action_clicked, body_result)`.
#[allow(clippy::too_many_arguments)]
pub fn cm3_collapsible_section_with_action<R>(
    ui: &mut Ui,
    id: Id,
    title: &str,
    default_open: bool,
    nested: bool,
    col_ratio: f32,
    action_icon: Option<(Cm3Icon, &str)>,
    theme: &Cm3Theme,
    add_body: impl FnOnce(&mut Ui) -> R,
) -> (bool, Option<R>) {
    ui.spacing_mut().item_spacing.y = 0.0;
    let mut open = ui
        .data_mut(|d| d.get_temp::<bool>(id))
        .unwrap_or(default_open);

    let depth = CM3_TREE_DEPTH.with(|d| d.get());
    let effective_depth = if nested { depth.max(1) } else { depth };
    let indent_px = effective_depth as f32 * 15.0;
    let hdr_h = if nested || action_icon.is_some() {
        28.0
    } else {
        27.0
    };

    let (_, raw_hdr_rect) = ui.allocate_space(Vec2::new(ui.available_width(), hdr_h));
    let hdr_rect = snap_rect_px(raw_hdr_rect);
    let hdr_resp = ui
        .interact(hdr_rect, id.with("hdr"), Sense::click())
        .on_hover_cursor(CursorIcon::PointingHand);

    let title_x = hdr_rect.left() + 21.0 + indent_px;
    let action_rect = action_icon.map(|_| {
        let max_x = (hdr_rect.right() - 20.0).max(title_x + 4.0);
        let action_x = ((hdr_rect.left() + hdr_rect.width() * col_ratio).round() + 1.0)
            .clamp(title_x + 4.0, max_x);
        let action_y = (hdr_rect.center().y - 10.0).round();
        Rect::from_min_size(Pos2::new(action_x, action_y), Vec2::splat(20.0))
    });

    let mut action_clicked = false;
    let mut action_hovered = false;
    if let (Some((icon, tooltip)), Some(a_rect)) = (action_icon, action_rect) {
        let a_resp = ui
            .interact(a_rect, id.with("action"), Sense::click())
            .on_hover_cursor(CursorIcon::PointingHand)
            .on_hover_text(tooltip);
        action_hovered = a_resp.hovered();
        action_clicked = a_resp.clicked();
        let _ = icon;
    }

    let bg = if nested { BG_PANEL } else { BG_TITLE };
    ui.painter().rect_filled(hdr_rect, CornerRadius::ZERO, bg);
    let div_y = (hdr_rect.bottom() - 1.0).round();
    let div_rect = Rect::from_min_max(
        Pos2::new(hdr_rect.left(), div_y),
        Pos2::new(hdr_rect.right(), div_y + 1.0),
    );
    ui.painter()
        .rect_filled(div_rect, CornerRadius::ZERO, BORDER_ROW_SOFT);

    // Crisp 7x7 framed square collapse glyph (`1px #9a9a9a` hollow square with `-` / `+`)
    let box_color = if hdr_resp.hovered() && !action_hovered {
        TEXT_BRIGHT
    } else {
        TEXT_MUTED
    };
    paint_cm3_tree_box(
        ui.painter(),
        Pos2::new(
            hdr_rect.left() + 10.5 + indent_px,
            hdr_rect.center().y + 0.5,
        ),
        open,
        box_color,
    );

    let title_right = action_rect.map_or(hdr_rect.right() - 4.0, |r| (r.left() - 4.0).max(title_x));
    let title_clip = Rect::from_min_max(
        Pos2::new(title_x - 1.0, hdr_rect.top()),
        Pos2::new(title_right.max(title_x), hdr_rect.bottom()),
    );
    ui.painter().with_clip_rect(title_clip).text(
        Pos2::new(title_x, hdr_rect.center().y),
        Align2::LEFT_CENTER,
        title,
        theme.font(FONT_BODY),
        TEXT_PRIMARY,
    );

    if let (Some((icon, _)), Some(a_rect)) = (action_icon, action_rect) {
        let icon_color = if action_hovered {
            TEXT_BRIGHT
        } else {
            TEXT_PRIMARY
        };
        paint_cm3_icon(ui.painter(), a_rect, icon, icon_color);
    }

    if hdr_resp.clicked() && !action_clicked {
        open = !open;
    }
    ui.data_mut(|d| d.insert_temp(id, open));

    let body = if open {
        let _depth_guard = TreeDepthGuard::push(effective_depth + 1);
        Some(add_body(ui))
    } else {
        None
    };
    (action_clicked, body)
}

/// CM3 collapsible section with `7x7` framed square collapse glyph (Spec 01 §4.7).
pub fn cm3_collapsible_section<R>(
    ui: &mut Ui,
    id: Id,
    title: &str,
    default_open: bool,
    theme: &Cm3Theme,
    add_body: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    let (_, body) = cm3_collapsible_section_with_action(
        ui,
        id,
        title,
        default_open,
        false,
        PropertyColumnState::default().label_ratio,
        None,
        theme,
        add_body,
    );
    body
}

// ---------------------------------------------------------------------------
// 7. Inline Fuzzy Picker (Add Object / Add Effect, Spec 01 §4.7)
// ---------------------------------------------------------------------------

/// One selectable entry in [`cm3_fuzzy_picker`].
#[derive(Debug, Clone)]
pub struct FuzzyItem {
    pub id: String,
    pub category: String,
    pub name: String,
    pub desc: String,
}

/// Persistent filter & active-row state for [`cm3_fuzzy_picker`].
#[derive(Debug, Clone, Default)]
pub struct FuzzyPickerState {
    pub query: String,
    pub active_index: usize,
    pub just_opened: bool,
}

/// Output event from [`cm3_fuzzy_picker`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FuzzyPickerEvent {
    None,
    Cancelled,
    Selected(String),
}

/// Fuzzy similarity score in `[0.0, 1.0]` combining contiguous substring & subsequence match.
#[must_use]
pub fn fuzzy_match_field(query: &str, field: &str) -> f32 {
    let q = query.trim().to_ascii_lowercase();
    if q.is_empty() {
        return 1.0;
    }
    let f = field.to_ascii_lowercase();
    if f.is_empty() {
        return 0.0;
    }
    if f == q {
        return 1.0;
    }
    if f.starts_with(&q) {
        return 0.95;
    }
    if f.contains(&q) {
        return 0.85;
    }
    // Ordered character subsequence ratio
    let mut qi = q.chars().peekable();
    let mut matched = 0usize;
    let mut consecutive = 0usize;
    let mut best_run = 0usize;
    for ch in f.chars() {
        if qi.peek().copied() == Some(ch) {
            qi.next();
            matched += 1;
            consecutive += 1;
            best_run = best_run.max(consecutive);
        } else {
            consecutive = 0;
        }
    }
    let q_len = q.chars().count().max(1);
    if matched < q_len {
        return (matched as f32 / q_len as f32) * 0.4;
    }
    let density = (best_run as f32 / q_len as f32).clamp(0.0, 1.0);
    0.55 + 0.30 * density
}

/// Weighted Spec 01 §4.7 fuzzy score (`0.6 * name + 0.4 * desc`, threshold `0.45`).
#[must_use]
pub fn fuzzy_score(query: &str, name: &str, desc: &str) -> f32 {
    if query.trim().is_empty() {
        return 1.0;
    }
    let name_s = fuzzy_match_field(query, name);
    let desc_s = fuzzy_match_field(query, desc);
    (0.6 * name_s + 0.4 * desc_s).max(name_s * 0.85)
}

/// Exact Spec 01 §4.7 3-row inline Add Object / Add Effect fuzzy picker.
pub fn cm3_fuzzy_picker(
    ui: &mut Ui,
    id: Id,
    items: &[FuzzyItem],
    state: &mut FuzzyPickerState,
    theme: &Cm3Theme,
) -> FuzzyPickerEvent {
    let mut event = FuzzyPickerEvent::None;

    // 1. Full-width Cancel button
    if cm3_prop_button(ui, "cancel", ButtonVariant::Normal, true, theme).clicked() {
        return FuzzyPickerEvent::Cancelled;
    }

    // 2. Auto-focused filter box (.pz-filterbox: #2a2a2a bg, 2px solid #363636 / #384668 focus)
    let filter_id = id.with("filter_box");
    let focused = ui.memory(|m| m.has_focus(filter_id));
    Frame::NONE
        .fill(BG_PANEL)
        .stroke(Stroke::new(
            2.0,
            if focused {
                theme.accent()
            } else {
                BORDER_FILTER
            },
        ))
        .inner_margin(egui::Margin::symmetric(5, 4))
        .show(ui, |ui| {
            let resp = ui.add(
                TextEdit::singleline(&mut state.query)
                    .id(filter_id)
                    .frame(Frame::NONE)
                    .hint_text("type to filter")
                    .font(theme.font(FONT_BODY))
                    .text_color(TEXT_PRIMARY)
                    .desired_width(ui.available_width()),
            );
            if state.just_opened {
                resp.request_focus();
                state.just_opened = false;
            }
        });

    // Build filtered list (threshold 0.45 when query non-empty)
    let filtered: Vec<&FuzzyItem> = if state.query.trim().is_empty() {
        items.iter().collect()
    } else {
        let mut scored: Vec<(f32, &FuzzyItem)> = items
            .iter()
            .filter_map(|it| {
                let s = fuzzy_score(&state.query, &it.name, &it.desc);
                (s >= 0.45).then_some((s, it))
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored.into_iter().map(|(_, it)| it).collect()
    };

    if filtered.is_empty() || state.active_index >= filtered.len() {
        state.active_index = 0;
    }

    // Keyboard navigation (Escape, ArrowUp, ArrowDown, Enter)
    let mut key_navigated = false;
    let pointer_moved = ui.input(|i| i.pointer.delta() != Vec2::ZERO);
    ui.input(|i| {
        if i.key_pressed(Key::Escape) {
            event = FuzzyPickerEvent::Cancelled;
        } else if !filtered.is_empty() {
            if i.key_pressed(Key::ArrowDown) {
                state.active_index = (state.active_index + 1) % filtered.len();
                key_navigated = true;
            } else if i.key_pressed(Key::ArrowUp) {
                state.active_index = (state.active_index + filtered.len() - 1) % filtered.len();
                key_navigated = true;
            } else if i.key_pressed(Key::Enter) {
                event = FuzzyPickerEvent::Selected(filtered[state.active_index].id.clone());
            }
        }
    });
    if event == FuzzyPickerEvent::Cancelled {
        consume_transient_popover_escape(ui.ctx());
    }
    if event != FuzzyPickerEvent::None {
        return event;
    }

    // 3. Scrollable .pz-options list with category banners
    ScrollArea::vertical()
        .id_salt(id.with("options_scroll"))
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;
            if filtered.is_empty() {
                ui.add_space(12.0);
                ui.label(
                    egui::RichText::new("no matching items")
                        .font(theme.font(FONT_SMALL))
                        .color(TEXT_FAINT),
                );
                return;
            }

            let mut last_category: Option<&str> = None;
            let show_categories = state.query.trim().is_empty();

            for (idx, item) in filtered.iter().enumerate() {
                if show_categories && last_category != Some(item.category.as_str()) {
                    last_category = Some(item.category.as_str());
                    let (cat_rect, _) = ui
                        .allocate_exact_size(Vec2::new(ui.available_width(), 24.0), Sense::hover());
                    ui.painter()
                        .rect_filled(cat_rect, CornerRadius::ZERO, BG_TITLE);
                    ui.painter()
                        .with_clip_rect(cat_rect.shrink2(Vec2::new(4.0, 0.0)))
                        .text(
                            Pos2::new(cat_rect.left() + 5.0, cat_rect.center().y),
                            Align2::LEFT_CENTER,
                            &item.category,
                            theme.font(FONT_BANNER),
                            TEXT_FAINT,
                        );
                }

                let has_desc = !item.desc.is_empty();
                let row_h = if has_desc { 42.0 } else { 28.0 };
                let (row_rect, row_resp) =
                    ui.allocate_exact_size(Vec2::new(ui.available_width(), row_h), Sense::click());
                let row_resp = row_resp.on_hover_cursor(CursorIcon::PointingHand);
                let is_active = idx == state.active_index;
                if key_navigated && is_active {
                    row_resp.scroll_to_me(None);
                }

                if is_active {
                    ui.painter()
                        .rect_filled(row_rect, CornerRadius::ZERO, BG_OPTION_ACTIVE);
                    ui.painter().rect_stroke(
                        row_rect,
                        CornerRadius::ZERO,
                        Stroke::new(1.0, BORDER_OPTION_ACTIVE),
                        StrokeKind::Inside,
                    );
                } else if row_resp.hovered() {
                    ui.painter()
                        .rect_filled(row_rect, CornerRadius::ZERO, BG_OPTION_HOVER);
                }

                let row_clip = ui
                    .painter()
                    .with_clip_rect(row_rect.shrink2(Vec2::new(4.0, 0.0)));
                row_clip.text(
                    Pos2::new(
                        row_rect.left() + 8.0,
                        if has_desc {
                            row_rect.top() + 13.0
                        } else {
                            row_rect.center().y
                        },
                    ),
                    Align2::LEFT_CENTER,
                    &item.name,
                    theme.font(FONT_BODY),
                    if is_active { TEXT_BRIGHT } else { TEXT_PRIMARY },
                );

                if has_desc {
                    row_clip.text(
                        Pos2::new(row_rect.left() + 8.0, row_rect.top() + 31.0),
                        Align2::LEFT_CENTER,
                        &item.desc,
                        theme.font(FONT_SMALL),
                        TEXT_OPTION_DESC,
                    );
                }

                if row_resp.hovered() && pointer_moved {
                    state.active_index = idx;
                }
                if row_resp.clicked() {
                    event = FuzzyPickerEvent::Selected(item.id.clone());
                }
            }
        });

    event
}

// ---------------------------------------------------------------------------
// 8. Resizable 2-Column Property Row Helper (Spec 01 §4.1)
// ---------------------------------------------------------------------------

/// Shared column width state for CM3 property trees (`layout = 0`, default `48%` label / `52%` value).
#[derive(Debug, Clone, Copy)]
pub struct PropertyColumnState {
    pub label_ratio: f32,
}

impl Default for PropertyColumnState {
    fn default() -> Self {
        Self { label_ratio: 0.48 }
    }
}

/// Interactive full-height column splitter inside a property panel (`8 px` hitbox, `1 px` visual line).
pub fn cm3_column_adjust_splitter(
    ui: &mut Ui,
    id: Id,
    panel_rect: Rect,
    col_state: &mut PropertyColumnState,
) {
    let split_x = (panel_rect.left()
        + (panel_rect.width() * col_state.label_ratio)
            .clamp(72.0, (panel_rect.width() - 72.0).max(72.0)))
    .round();
    let hit = Rect::from_min_max(
        Pos2::new(split_x - 4.0, panel_rect.top()),
        Pos2::new(split_x + 4.0, panel_rect.bottom()),
    );
    let resp = ui
        .interact(hit, id, Sense::drag())
        .on_hover_cursor(CursorIcon::ResizeHorizontal);
    if resp.hovered() || resp.dragged() {
        let line_color = if resp.dragged() {
            BORDER_OPTION_ACTIVE
        } else {
            BORDER_BUTTON_HOVER
        };
        ui.painter().line_segment(
            [
                Pos2::new(split_x + 0.5, panel_rect.top()),
                Pos2::new(split_x + 0.5, panel_rect.bottom()),
            ],
            Stroke::new(1.0, line_color),
        );
    }
    if resp.dragged() {
        let new_w = (split_x + resp.drag_delta().x - panel_rect.left())
            .clamp(50.0, (panel_rect.width() - 72.0).max(50.0));
        col_state.label_ratio = (new_w / panel_rect.width().max(1.0)).clamp(0.15, 0.80);
    }
}

#[allow(clippy::too_many_arguments)]
fn cm3_property_row_inner<R>(
    ui: &mut Ui,
    label: &str,
    col_state: PropertyColumnState,
    base_indent_px: f32,
    _reserve_stopwatch_gutter: bool,
    theme: &Cm3Theme,
    left_extra: impl FnOnce(&mut Ui),
    right_value: impl FnOnce(&mut Ui) -> R,
) -> R {
    ui.spacing_mut().item_spacing.y = 0.0;
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        let total_w = ui.available_width().max(120.0);
        let row_left = ui.cursor().left().round();
        let row_top = ui.cursor().top().round();
        let row_right = (row_left + total_w).round();
        let split_x = (row_left
            + (total_w * col_state.label_ratio).clamp(72.0, (total_w - 72.0).max(72.0)))
        .round();
        let col1_right = (split_x - 8.0).max(row_left + 40.0);

        // Column 2 runs first (disjoint horizontal rect [split_x .. row_right]) so Column 1
        // knows whether this row contains a dropdown or color picker for exact vertical centering.
        let col2_max_rect = Rect::from_min_max(
            Pos2::new(split_x, row_top),
            Pos2::new(row_right, (row_top + 1000.0).max(ui.max_rect().bottom())),
        );
        let prev_dropdown_flag = CM3_ROW_HAS_DROPDOWN.with(|f| f.replace(false));
        let prev_color_flag = CM3_ROW_HAS_COLOR_PICKER.with(|f| f.replace(false));
        let (out, col2_used_h, row_has_dropdown, row_has_color_picker) = {
            let mut col2_ui = ui.new_child(
                UiBuilder::new()
                    .id_salt("col2")
                    .max_rect(col2_max_rect)
                    .layout(egui::Layout::top_down(egui::Align::LEFT)),
            );
            col2_ui.spacing_mut().item_spacing.y = 0.0;
            let col2_clip = ui.clip_rect().intersect(Rect::from_min_max(
                Pos2::new(split_x, ui.clip_rect().top()),
                Pos2::new(row_right, ui.clip_rect().bottom()),
            ));
            col2_ui.set_clip_rect(col2_clip);
            col2_ui.add_space(2.0);
            let result = right_value(&mut col2_ui);
            let used_h = (col2_ui.min_rect().bottom() - row_top).max(0.0);
            let has_dd = CM3_ROW_HAS_DROPDOWN.with(|f| f.replace(prev_dropdown_flag));
            let has_cp = CM3_ROW_HAS_COLOR_PICKER.with(|f| f.replace(prev_color_flag));
            (result, used_h, has_dd, has_cp)
        };

        // Column 1: fixed rect [row_left .. split_x - 11.0] so short labels never pull Column 2 left.
        // Center labels at row_top + 14.0 on 27-28px rows (dropdown, color picker, stopwatch) and
        // row_top + 13.0 on 26px plain rows, while keeping stopwatch icon alignment unchanged.
        let base_col1_h = if row_has_dropdown || row_has_color_picker {
            28.0
        } else {
            26.0
        };
        let col1_rect = Rect::from_min_max(
            Pos2::new(row_left, row_top),
            Pos2::new(col1_right, row_top + base_col1_h),
        );
        let has_left_widget = {
            let mut col1_ui = ui.new_child(
                UiBuilder::new()
                    .id_salt("col1")
                    .max_rect(col1_rect)
                    .layout(egui::Layout::left_to_right(egui::Align::Center)),
            );
            col1_ui.set_clip_rect(ui.clip_rect().intersect(col1_rect));
            col1_ui.spacing_mut().item_spacing = Vec2::new(5.0, 0.0);
            if base_indent_px > 0.0 {
                col1_ui.add_space(base_indent_px);
            }
            let gutter_start = col1_ui.cursor().left();
            left_extra(&mut col1_ui);
            let drew_left = col1_ui.cursor().left() > gutter_start + 0.5;
            if drew_left && !row_has_dropdown && !row_has_color_picker {
                let label_rect = Rect::from_min_max(
                    Pos2::new(col1_ui.cursor().left(), row_top),
                    Pos2::new(col1_right, row_top + 28.0),
                );
                let mut label_ui = ui.new_child(
                    UiBuilder::new()
                        .id_salt("col1_lbl")
                        .max_rect(label_rect)
                        .layout(egui::Layout::left_to_right(egui::Align::Center)),
                );
                label_ui.set_clip_rect(ui.clip_rect().intersect(label_rect));
                label_ui.add(
                    egui::Label::new(
                        egui::RichText::new(label)
                            .font(theme.font(FONT_BODY))
                            .color(TEXT_PRIMARY),
                    )
                    .truncate(),
                );
            } else {
                col1_ui.add(
                    egui::Label::new(
                        egui::RichText::new(label)
                            .font(theme.font(FONT_BODY))
                            .color(TEXT_PRIMARY),
                    )
                    .truncate(),
                );
            }
            drew_left
        };

        // Match Zoidium row pitch:
        // - Dropdown rows: 28 px content + 1 px divider = 29 px pitch
        // - Stopwatch / color picker rows: 27 px content + 1 px divider = 28 px pitch
        // - Plain rows: 26 px content + 1 px divider = 27 px pitch
        let min_content_h = if row_has_dropdown {
            28.0
        } else if has_left_widget || row_has_color_picker {
            27.0
        } else {
            26.0
        };
        let content_h = if col2_used_h > min_content_h {
            (col2_used_h + 2.0).ceil()
        } else {
            min_content_h
        };
        let _ = ui.allocate_space(Vec2::new(total_w, content_h));

        // Crisp integer-snapped 1 px #242424 bottom divider.
        let (_, div_rect) = ui.allocate_space(Vec2::new(total_w, 1.0));
        let div_y = div_rect.top().round();
        let div_snap = Rect::from_min_max(
            Pos2::new(row_left, div_y),
            Pos2::new(row_right, div_y + 1.0),
        );
        ui.painter()
            .rect_filled(div_snap, CornerRadius::ZERO, BORDER_ROW_SOFT);
        out
    })
    .inner
}

/// Renders one CM3 2-column property row (`27..29 px` total pitch, Spec 01 §4.1).
pub fn cm3_property_row<R>(
    ui: &mut Ui,
    label: &str,
    col_state: PropertyColumnState,
    indent_depth: usize,
    theme: &Cm3Theme,
    left_extra: impl FnOnce(&mut Ui),
    right_value: impl FnOnce(&mut Ui) -> R,
) -> R {
    let base_indent_px = 21.0 + indent_depth as f32 * 15.0;
    cm3_property_row_inner(
        ui,
        label,
        col_state,
        base_indent_px,
        false,
        theme,
        left_extra,
        right_value,
    )
}

/// Renders one CM3 2-column property row aligned to the active tree nesting depth
/// (`21 px + depth * 15 px`: `36 px` at Depth 1, `51 px` at Depth 2) with crisp Column 2 alignment.
pub fn cm3_property_row_aligned<R>(
    ui: &mut Ui,
    label: &str,
    reserve_stopwatch_gutter: bool,
    col_state: PropertyColumnState,
    theme: &Cm3Theme,
    left_extra: impl FnOnce(&mut Ui),
    add_control: impl FnOnce(&mut Ui) -> R,
) -> R {
    let depth = CM3_TREE_DEPTH.with(|d| d.get());
    let base_indent_px = 21.0 + depth as f32 * 15.0;
    cm3_property_row_inner(
        ui,
        label,
        col_state,
        base_indent_px,
        reserve_stopwatch_gutter,
        theme,
        left_extra,
        add_control,
    )
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
    z_counter: u64,
    cascade_step: usize,
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

fn clamp_window_pos(pos: Pos2, size: Vec2, viewport: Rect) -> Pos2 {
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
    static CM3_TREE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CM3_ROW_HAS_DROPDOWN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static CM3_ROW_HAS_COLOR_PICKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct TreeDepthGuard {
    prev: usize,
}

impl TreeDepthGuard {
    fn push(next: usize) -> Self {
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
fn snap_rect_px(r: Rect) -> Rect {
    Rect::from_min_max(
        Pos2::new(r.left().round(), r.top().round()),
        Pos2::new(r.right().round(), r.bottom().round()),
    )
}

#[inline]
fn paint_crisp_border_1px(painter: &Painter, rect: Rect, color: Color32) {
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

fn clamp_popup_pos(anchor: Rect, popup_size: Vec2, viewport: Rect) -> Pos2 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn run_frame(ctx: &egui::Context, events: Vec<egui::Event>, f: impl FnMut(&mut Ui)) {
        let mut raw = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
            events,
            ..Default::default()
        };
        raw.focused = true;
        let mut f = f;
        let mut out = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(Frame::NONE)
                .show(ui, |ui| {
                    f(ui);
                });
        });
        out.textures_delta.clear();
    }

    #[test]
    fn default_swatches_order_and_css_roundtrip() {
        let sw = default_recent_swatches();
        assert_eq!(sw.len(), 36);
        // Row 0 col 0 is pure red (h = 0)
        assert_eq!(sw[0].to_u8(), [255, 0, 0]);
        // Row 3 (18..24) is white -> black grayscale
        assert_eq!(sw[18].to_u8(), [255, 255, 255]);
        assert_eq!(sw[19].to_u8(), [204, 204, 204]);
        assert_eq!(sw[20].to_u8(), [153, 153, 153]);
        assert_eq!(sw[21].to_u8(), [102, 102, 102]);
        assert_eq!(sw[22].to_u8(), [51, 51, 51]);
        assert_eq!(sw[23].to_u8(), [0, 0, 0]);

        let black = Rgba::new(0.0, 0.0, 0.0, 1.0);
        assert_eq!(format_picker_css(black, false), "rgb(0, 0, 0)");
        assert_eq!(format_picker_css(black, true), "rgba(0, 0, 0, 1)");
        let semi = Rgba::new(1.0, 0.0, 0.5, 0.5);
        let css = format_picker_css(semi, true);
        assert!(css.starts_with("rgba(255, 0, 128, 0.5"));
        let parsed_css = parse_picker_color("rgb(12, 34, 56)").expect("rgb parse");
        assert_eq!(parsed_css.to_u8(), [12, 34, 56, 255]);
        let parsed_hex = parse_picker_color("#112233").expect("hex parse");
        assert_eq!(parsed_hex.to_u8(), [0x11, 0x22, 0x33, 255]);
        assert!(parse_picker_color("not-a-color").is_none());
    }

    #[test]
    fn scrub_pending_drag_to_text_edit_cancels_transaction_and_commits_cleanly() {
        let ctx = egui::Context::default();
        let theme = Cm3Theme::default();
        let id = Id::unique("test_scrub_tx");
        let cfg = ScrubConfig::new(0.0, 100.0, 1.0, 2);
        let mut val = 10.0_f64;

        // Frame 1: layout to discover hit rect
        let mut hit_center = Pos2::ZERO;
        run_frame(&ctx, vec![], |ui| {
            let start = ui.cursor().min;
            let ev = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
            assert_eq!(ev, ScrubEvent::default());
            hit_center = Pos2::new(start.x + 8.0, start.y + 10.0);
        });

        // Frame 2: pointer down -> PendingDrag, ev.started = true
        let mut ev2 = ScrubEvent::default();
        run_frame(
            &ctx,
            vec![
                egui::Event::PointerMoved(hit_center),
                egui::Event::PointerButton {
                    pos: hit_center,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            |ui| {
                ev2 = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
            },
        );
        assert!(ev2.started && !ev2.cancelled && !ev2.committed);

        // Frame 3: pointer up without dragging -> immediately cancels PendingDrag tx & enters TextEdit
        let mut ev3 = ScrubEvent::default();
        run_frame(
            &ctx,
            vec![egui::Event::PointerButton {
                pos: hit_center,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
            |ui| {
                ev3 = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
            },
        );
        assert!(ev3.cancelled && !ev3.started && !ev3.committed && !ev3.changed);

        // Frame 4: TextEdit renders and requests focus
        let mut ev4 = ScrubEvent::default();
        run_frame(&ctx, vec![], |ui| {
            ev4 = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
        });
        assert_eq!(ev4, ScrubEvent::default());

        // Frame 5: Escape in TextEdit -> reverts and emits NO events (no open tx)
        let mut ev5 = ScrubEvent::default();
        run_frame(
            &ctx,
            vec![egui::Event::Key {
                key: Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            |ui| {
                ev5 = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
            },
        );
        assert_eq!(ev5, ScrubEvent::default());
        assert!((val - 10.0).abs() < 1e-9);
    }

    #[test]
    fn scrub_read_only_mid_pending_drag_emits_cancel() {
        let ctx = egui::Context::default();
        let theme = Cm3Theme::default();
        let id = Id::unique("test_scrub_ro");
        let cfg = ScrubConfig::new(0.0, 100.0, 1.0, 2);
        let mut val = 25.0_f64;

        let mut hit_center = Pos2::ZERO;
        run_frame(&ctx, vec![], |ui| {
            let start = ui.cursor().min;
            let _ = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
            hit_center = Pos2::new(start.x + 8.0, start.y + 10.0);
        });
        run_frame(
            &ctx,
            vec![
                egui::Event::PointerMoved(hit_center),
                egui::Event::PointerButton {
                    pos: hit_center,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            |ui| {
                let ev = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
                assert!(ev.started);
            },
        );

        let mut ev_ro = ScrubEvent::default();
        run_frame(&ctx, vec![], |ui| {
            ev_ro = cm3_scrub_number(ui, id, &mut val, cfg.with_read_only(true), &theme);
        });
        assert!(ev_ro.cancelled);
        assert!((val - 25.0).abs() < 1e-9);
    }

    #[test]
    fn vector2_text_edit_commit_scales_linked_axis_proportionally() {
        let ctx = egui::Context::default();
        let theme = Cm3Theme::default();
        let id = Id::unique("test_vec2_link");
        let cfg = ScrubConfig::new(0.0, 100.0, 1.0, 2);
        let mut vals = [2.0_f64, 6.0_f64];

        // Enter TextEdit on axis 0 via temp memory (simulating pointer-down -> pointer-up -> typed "5.00")
        ctx.data_mut(|d| {
            d.insert_temp(
                id.with(0usize),
                ScrubMemory {
                    mode: ScrubMode::TextEdit {
                        buffer: "5.00".to_owned(),
                        initial_display: 2.0,
                        start_value: 2.0,
                        just_entered: false,
                        alt_at_start: false,
                    },
                },
            );
        });
        ctx.memory_mut(|m| m.request_focus(id.with(0usize).with("text_edit")));

        // Frame to settle focus
        run_frame(&ctx, vec![], |ui| {
            let _ = cm3_vector2_with_sep(ui, id, &mut vals, &cfg, true, ",", &theme);
        });

        // Press Enter to commit 5.00 on axis 0 -> axis 1 should scale 6.0 * (5.0 / 2.0) = 15.0
        let mut ev = ScrubEvent::default();
        run_frame(
            &ctx,
            vec![egui::Event::Key {
                key: Key::Enter,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            |ui| {
                ev = cm3_vector2_with_sep(ui, id, &mut vals, &cfg, true, ",", &theme);
            },
        );
        assert!(ev.changed && ev.committed && !ev.started && !ev.cancelled);
        assert!((vals[0] - 5.0).abs() < 1e-6);
        assert!((vals[1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn property_row_col2_aligns_identically_for_short_and_long_labels() {
        let ctx = egui::Context::default();
        let theme = Cm3Theme::default();
        let col_state = PropertyColumnState::default();
        let mut short_x = 0.0_f32;
        let mut long_x = 0.0_f32;

        run_frame(&ctx, vec![], |ui| {
            ui.set_width(321.0);
            cm3_property_row_aligned(
                ui,
                "Time",
                true,
                col_state,
                &theme,
                |_| {},
                |ui| {
                    short_x = ui.cursor().left();
                },
            );
            cm3_property_row_aligned(
                ui,
                "Multisample motion blur",
                true,
                col_state,
                &theme,
                |_| {},
                |ui| {
                    long_x = ui.cursor().left();
                },
            );
        });
        assert!(
            (short_x - long_x).abs() < 1e-3,
            "Column 2 misaligned: short={short_x}, long={long_x}"
        );
    }

    #[test]
    fn gradient_bar_stop_cancel_and_switch_events() {
        let ctx = egui::Context::default();
        let theme = Cm3Theme::default();
        let id = Id::unique("test_grad_bar");
        let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let mut stops = vec![
            GradientStop {
                position: 0.0,
                color: "rgba(255,0,0,1.000)".to_owned(),
            },
            GradientStop {
                position: 1.0,
                color: "rgba(0,0,255,1.000)".to_owned(),
            },
        ];

        // Open stop color picker popup in dirty state with pre-drag initial = red, current = green
        let stop_popup_id = id.with("stop_color").with("hsva_popup");
        stops[0].color = "rgba(0,255,0,1.000)".to_owned();
        ctx.data_mut(|d| {
            d.insert_temp(
                stop_popup_id,
                ColorPopupState {
                    open: true,
                    initial: Rgba::new(1.0, 0.0, 0.0, 1.0),
                    hsv: Rgb::from_u8(0, 255, 0).to_hsv(),
                    alpha: 1.0,
                    hex_buf: "rgba(0, 255, 0, 1)".to_owned(),
                    dirty: true,
                },
            );
        });

        // Press Escape -> restores red, emits changed=true, cancelled=true, committed=false
        let mut ev = GradientBarEvent::default();
        run_frame(
            &ctx,
            vec![egui::Event::Key {
                key: Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            |ui| {
                ev = cm3_gradient_bar(ui, id, &mut stops, vp, &theme);
            },
        );
        assert!(ev.changed && ev.cancelled && !ev.committed);
        assert_eq!(parse_css_rgba(&stops[0].color).to_u8(), [255, 0, 0, 255]);
    }

    #[test]
    fn hsva_color_picker_read_only_mid_open_cancels_and_restores() {
        let ctx = egui::Context::default();
        let theme = Cm3Theme::default();
        let id = Id::unique("test_color_ro_cancel");
        let popup_id = id.with("hsva_popup");
        let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let initial = Rgba::new(0.2, 0.4, 0.8, 1.0);
        let mut color = Rgba::new(0.9, 0.1, 0.1, 1.0);

        // Case 1: popup is open and dirty, then read_only flips true -> restores initial, cancels, changed=true
        ctx.data_mut(|d| {
            d.insert_temp(
                popup_id,
                ColorPopupState {
                    open: true,
                    initial,
                    hsv: color.rgb().to_hsv(),
                    alpha: 1.0,
                    hex_buf: format_picker_css(color, false),
                    dirty: true,
                },
            );
        });
        let mut ev = ColorPickerEvent::default();
        run_frame(&ctx, vec![], |ui| {
            ev = cm3_hsva_color_picker(ui, id, &mut color, false, true, vp, &theme);
        });
        assert!(ev.cancelled && ev.changed && !ev.committed && !ev.started);
        assert_eq!(color.to_u8(), initial.to_u8());
        let st = ctx
            .data_mut(|d| d.get_temp::<ColorPopupState>(popup_id))
            .expect("persisted state");
        assert!(!st.open && !st.dirty);

        // Case 2: popup is open and clean (not dirty), read_only flips true -> cancels, changed=false
        ctx.data_mut(|d| {
            d.insert_temp(
                popup_id,
                ColorPopupState {
                    open: true,
                    initial,
                    hsv: initial.rgb().to_hsv(),
                    alpha: 1.0,
                    hex_buf: format_picker_css(initial, false),
                    dirty: false,
                },
            );
        });
        let mut ev_clean = ColorPickerEvent::default();
        run_frame(&ctx, vec![], |ui| {
            ev_clean = cm3_hsva_color_picker(ui, id, &mut color, false, true, vp, &theme);
        });
        assert!(ev_clean.cancelled && !ev_clean.changed && !ev_clean.committed);
        let st2 = ctx
            .data_mut(|d| d.get_temp::<ColorPopupState>(popup_id))
            .expect("persisted state");
        assert!(!st2.open && !st2.dirty);
    }

    #[test]
    fn gradient_bar_clean_stop_switch_emits_no_spurious_commit() {
        let ctx = egui::Context::default();
        let theme = Cm3Theme::default();
        let id = Id::unique("test_grad_clean_switch");
        let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let mut stops = vec![
            GradientStop {
                position: 0.0,
                color: "rgba(255,0,0,1.000)".to_owned(),
            },
            GradientStop {
                position: 1.0,
                color: "rgba(0,0,255,1.000)".to_owned(),
            },
        ];

        // Frame 1: layout to discover stop 1 rect (right end of 245px lane)
        let mut lane_origin = Pos2::ZERO;
        run_frame(&ctx, vec![], |ui| {
            ui.set_width(245.0);
            lane_origin = ui.cursor().min;
            let _ = cm3_gradient_bar(ui, id, &mut stops, vp, &theme);
        });

        // Open stop 0's color picker in clean (dirty=false) state
        let stop_popup_id = id.with("stop_color").with("hsva_popup");
        ctx.data_mut(|d| {
            d.insert_temp(
                stop_popup_id,
                ColorPopupState {
                    open: true,
                    initial: Rgba::new(1.0, 0.0, 0.0, 1.0),
                    hsv: Rgb::from_u8(255, 0, 0).to_hsv(),
                    alpha: 1.0,
                    hex_buf: "rgba(255, 0, 0, 1)".to_owned(),
                    dirty: false,
                },
            );
        });

        // Click stop 1 (at x = lane_origin.x + 245.0, y = lane_origin.y + 30.0 + 3.0 + 7.5)
        let stop1_pos = Pos2::new(lane_origin.x + 244.0, lane_origin.y + 40.5);
        let mut ev_down = GradientBarEvent::default();
        run_frame(
            &ctx,
            vec![
                egui::Event::PointerMoved(stop1_pos),
                egui::Event::PointerButton {
                    pos: stop1_pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            |ui| {
                ui.set_width(245.0);
                ev_down = cm3_gradient_bar(ui, id, &mut stops, vp, &theme);
            },
        );
        let mut ev_up = GradientBarEvent::default();
        run_frame(
            &ctx,
            vec![egui::Event::PointerButton {
                pos: stop1_pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
            |ui| {
                ui.set_width(245.0);
                ev_up = cm3_gradient_bar(ui, id, &mut stops, vp, &theme);
            },
        );
        assert!(!ev_down.committed && !ev_up.committed);
        let st = ctx
            .data_mut(|d| d.get_temp::<ColorPopupState>(stop_popup_id))
            .expect("stop popup state");
        assert!(!st.open);
    }

    #[test]
    fn gradient_bar_reverted_stop_switch_emits_cancel_to_close_tx() {
        let ctx = egui::Context::default();
        let theme = Cm3Theme::default();
        let id = Id::unique("test_grad_reverted_switch");
        let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let mut stops = vec![
            GradientStop {
                position: 0.0,
                color: "rgba(255,0,0,1.000)".to_owned(),
            },
            GradientStop {
                position: 1.0,
                color: "rgba(0,0,255,1.000)".to_owned(),
            },
        ];

        // Frame 1: layout to discover stop 1 rect (right end of 245px lane)
        let mut lane_origin = Pos2::ZERO;
        run_frame(&ctx, vec![], |ui| {
            ui.set_width(245.0);
            lane_origin = ui.cursor().min;
            let _ = cm3_gradient_bar(ui, id, &mut stops, vp, &theme);
        });

        // Stop 0's color picker is open and clean (e.g. after reverting via comparison bar)
        let stop_popup_id = id.with("stop_color").with("hsva_popup");
        ctx.data_mut(|d| {
            d.insert_temp(
                stop_popup_id,
                ColorPopupState {
                    open: true,
                    initial: Rgba::new(1.0, 0.0, 0.0, 1.0),
                    hsv: Rgb::from_u8(255, 0, 0).to_hsv(),
                    alpha: 1.0,
                    hex_buf: "rgba(255, 0, 0, 1)".to_owned(),
                    dirty: false,
                },
            );
        });

        // Switch to stop 1 in a single click frame so `stop_changed` fires while `st.open && !st.dirty`
        let stop1_pos = Pos2::new(lane_origin.x + 244.0, lane_origin.y + 40.5);
        let mut ev = GradientBarEvent::default();
        run_frame(
            &ctx,
            vec![
                egui::Event::PointerMoved(stop1_pos),
                egui::Event::PointerButton {
                    pos: stop1_pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerButton {
                    pos: stop1_pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            |ui| {
                ui.set_width(245.0);
                ev = cm3_gradient_bar(ui, id, &mut stops, vp, &theme);
            },
        );
        assert!(ev.cancelled && !ev.committed && !ev.changed, "ev={ev:?}");
        let bar_st = ctx
            .data_mut(|d| d.get_temp::<GradientBarState>(id))
            .expect("gradient bar state");
        assert_eq!(bar_st.selected_stop, 1);
        let st = ctx
            .data_mut(|d| d.get_temp::<ColorPopupState>(stop_popup_id))
            .expect("stop popup state");
        assert!(!st.open);
    }

    #[test]
    fn scrub_event_merge_preserves_same_frame_text_commit_and_pending_drag_start() {
        let a = ScrubEvent {
            started: false,
            changed: true,
            committed: true,
            cancelled: false,
        };
        let b = ScrubEvent {
            started: true,
            changed: false,
            committed: false,
            cancelled: false,
        };
        let merged = a.merge(b);
        assert!(merged.started && merged.changed && merged.committed && !merged.cancelled);
        assert_eq!(b.merge(a), merged);
    }

    #[test]
    fn hsv_color_picker_single_click_in_sv_square_picks_immediately() {
        let ctx = egui::Context::default();
        let theme = Cm3Theme::default();
        let id = Id::unique("test_hsv_sv_single_click");
        let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let mut color = Rgba::new(1.0, 1.0, 1.0, 1.0); // white: S=0, V=1

        // Frame 1: measure swatch position.
        let mut swatch_center = Pos2::ZERO;
        run_frame(&ctx, vec![], |ui| {
            let start = ui.cursor().min;
            let _ = cm3_hsva_color_picker(ui, id, &mut color, false, false, vp, &theme);
            swatch_center = Pos2::new(start.x + 15.0, start.y + 12.0);
        });

        // Frame 2: click swatch to open HSVA popup.
        let mut open_ev = ColorPickerEvent::default();
        run_frame(
            &ctx,
            vec![
                egui::Event::PointerMoved(swatch_center),
                egui::Event::PointerButton {
                    pos: swatch_center,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerButton {
                    pos: swatch_center,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            |ui| {
                open_ev = cm3_hsva_color_picker(ui, id, &mut color, false, false, vp, &theme);
            },
        );
        assert!(open_ev.started);

        // Frame 3: render open popup once so layout settles.
        run_frame(&ctx, vec![], |ui| {
            let _ = cm3_hsva_color_picker(ui, id, &mut color, false, false, vp, &theme);
        });

        // Frame 4: single pointer-down inside the lower-right of the 250x250 SV square (no drag!).
        let popup_pos = clamp_popup_pos(
            Rect::from_min_size(
                Pos2::new(swatch_center.x - 15.0, swatch_center.y - 10.0),
                Vec2::new(30.0, 20.0),
            ),
            Vec2::new(482.0, 262.0),
            vp,
        );
        let sv_click = Pos2::new(popup_pos.x + 6.0 + 200.0, popup_pos.y + 6.0 + 125.0);
        let mut pick_ev = ColorPickerEvent::default();
        run_frame(
            &ctx,
            vec![
                egui::Event::PointerMoved(sv_click),
                egui::Event::PointerButton {
                    pos: sv_click,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            |ui| {
                pick_ev = cm3_hsva_color_picker(ui, id, &mut color, false, false, vp, &theme);
            },
        );
        assert!(
            pick_ev.changed,
            "single click press in SV square must pick immediately without requiring a drag"
        );
        let hsv = color.rgb().to_hsv();
        assert!(
            hsv.s > 0.6 && hsv.v < 0.7,
            "expected high saturation / mid value from SV click, got hsv={hsv:?}"
        );
    }

    #[test]
    fn dropdown_dynamic_option_count_reanchors_flush_to_button() {
        let ctx = egui::Context::default();
        let theme = Cm3Theme::default();
        let id = Id::unique("test_dd_dynamic_opts");
        let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let mut selected = 0usize;
        let opts_5 = ["Basic", "Lambert", "Phong", "Standard", "Physical"];
        let opts_9 = [
            "Basic", "Lambert", "Phong", "Standard", "Physical", "Toon", "Matcap", "Normal",
            "Depth",
        ];

        // Position button near bottom of viewport so the 9-option popup flips above cleanly.
        let mut btn_center = Pos2::ZERO;
        run_frame(&ctx, vec![], |ui| {
            ui.add_space(430.0);
            let top_left = ui.cursor().min;
            let _ = cm3_dropdown(ui, id, &mut selected, &opts_5, false, vp, &theme);
            btn_center = Pos2::new(top_left.x + 30.0, top_left.y + 12.0);
        });

        // Open with 5 options.
        run_frame(
            &ctx,
            vec![
                egui::Event::PointerMoved(btn_center),
                egui::Event::PointerButton {
                    pos: btn_center,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerButton {
                    pos: btn_center,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            |ui| {
                ui.add_space(430.0);
                let _ = cm3_dropdown(ui, id, &mut selected, &opts_5, false, vp, &theme);
            },
        );
        let popup_id = id.with("dropdown_popup");
        let st1 = ctx
            .data_mut(|d| d.get_temp::<DropdownPopupState>(popup_id))
            .expect("popup state after 5 options");
        assert!(st1.open);
        assert_eq!(st1.last_option_count, 5);
        let epoch1 = st1.open_epoch;

        // Grow options dynamically from 5 to 9 mid-session (e.g. Material+ enabled).
        run_frame(&ctx, vec![], |ui| {
            ui.add_space(430.0);
            let _ = cm3_dropdown(ui, id, &mut selected, &opts_9, false, vp, &theme);
        });
        let st2 = ctx
            .data_mut(|d| d.get_temp::<DropdownPopupState>(popup_id))
            .expect("popup state after 9 options");
        assert_eq!(st2.last_option_count, 9);
        assert_ne!(
            st2.open_epoch, epoch1,
            "growing option count must bump open_epoch so egui never reuses stale cached Area height"
        );
    }

    #[test]
    fn vector3_fits_within_default_property_column_without_clipping() {
        let ctx = egui::Context::default();
        let theme = Cm3Theme::default();
        let col_state = PropertyColumnState::default();
        let id = Id::unique("test_vec3_fit");
        let cfg = ScrubConfig::new(-10000.0, 10000.0, 1.0, 2);
        let mut vals = [100.0_f64, 100.0, 100.0];
        let mut row_right = 0.0_f32;
        let mut used_right = 0.0_f32;

        run_frame(&ctx, vec![], |ui| {
            ui.allocate_ui(Vec2::new(324.0, 120.0), |ui| {
                row_right = ui.max_rect().right();
                cm3_property_row_aligned(
                    ui,
                    "Scale",
                    true,
                    col_state,
                    &theme,
                    |_| {},
                    |ui| {
                        let _ = cm3_vector3(ui, id, &mut vals, cfg, false, &theme);
                        used_right = ui.min_rect().right();
                    },
                );
            });
        });
        assert!(
            used_right <= row_right + 1.0,
            "Vec3 (100.00, 100.00, 100.00) must fit inside default inspector column without clipping: used_right={used_right}, row_right={row_right}"
        );
    }

    #[test]
    fn vector4_fits_inline_and_wraps_to_2x2_when_narrow_without_clipping() {
        let ctx = egui::Context::default();
        super::super::theme::install_cm3_fonts(&ctx);
        let theme = Cm3Theme::default();
        let col_state = PropertyColumnState::default();
        let cfg = ScrubConfig::new(-10000.0, 10000.0, 0.5, 1);

        // 1. Standard 1280px inspector column (~330px total -> ~198px Col 2) fits 4 components cleanly.
        let mut vals = [12.0_f64, 12.0, 40.0, 20.0];
        let mut row_right = 0.0_f32;
        let mut used_right = 0.0_f32;
        run_frame(&ctx, vec![], |ui| {
            ui.allocate_ui(Vec2::new(330.0, 120.0), |ui| {
                row_right = ui.max_rect().right();
                cm3_property_row_aligned(
                    ui,
                    "Amplitude",
                    true,
                    col_state,
                    &theme,
                    |_| {},
                    |ui| {
                        let _ = cm3_vector4(
                            ui,
                            Id::unique("test_vec4_fit_std"),
                            &mut vals,
                            cfg,
                            &theme,
                        );
                        used_right = ui.min_rect().right();
                    },
                );
            });
        });
        assert!(
            used_right <= row_right + 1.0,
            "Vec4 [12.0, 12.0, 40.0, 20.0] must fit inside default inspector column without clipping: used_right={used_right}, row_right={row_right}"
        );

        // 2. Narrow column (~200px total -> ~120px Col 2) with wide 2-decimal numbers wraps into
        //    a neat 2x2 grid (height >= 44px) and NEVER clips past row_right.
        let mut wide_vals = [120.25_f64, 120.50, 400.75, 200.00];
        let wide_cfg = ScrubConfig::new(-10000.0, 10000.0, 0.1, 2);
        let mut narrow_row_right = 0.0_f32;
        let mut narrow_used_right = 0.0_f32;
        let mut narrow_used_h = 0.0_f32;
        run_frame(&ctx, vec![], |ui| {
            ui.allocate_ui(Vec2::new(200.0, 160.0), |ui| {
                narrow_row_right = ui.max_rect().right();
                cm3_property_row_aligned(
                    ui,
                    "Noise",
                    true,
                    col_state,
                    &theme,
                    |_| {},
                    |ui| {
                        let top = ui.cursor().top();
                        let _ = cm3_vector4(
                            ui,
                            Id::unique("test_vec4_fit_narrow"),
                            &mut wide_vals,
                            wide_cfg,
                            &theme,
                        );
                        narrow_used_right = ui.min_rect().right();
                        narrow_used_h = ui.min_rect().bottom() - top;
                    },
                );
            });
        });
        assert!(
            narrow_used_right <= narrow_row_right + 1.0,
            "Wrapped 2x2 Vec4 must never clip right edge: used_right={narrow_used_right}, row_right={narrow_row_right}"
        );
        assert!(
            narrow_used_h >= 40.0,
            "Narrow Vec4 should wrap cleanly into 2 rows (2x2 grid), got height={narrow_used_h}"
        );
    }

    #[test]
    fn dropdown_reserves_chevron_padding_and_ellipsizes_when_constrained() {
        let ctx = egui::Context::default();
        super::super::theme::install_cm3_fonts(&ctx);
        let theme = Cm3Theme::default();
        let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let opts = ["Classic Dark", "High Contrast", "Ocean Slate"];
        let mut selected = 0usize;

        // Roomy container: dropdown allocates max_opt_w + 28px so full label never touches chevron.
        let mut full_box_w = 0.0_f32;
        let mut classic_w = 0.0_f32;
        run_frame(&ctx, vec![], |ui| {
            classic_w = ui
                .painter()
                .layout_no_wrap("Classic Dark".to_owned(), theme.font(13.3), theme.value())
                .size()
                .x;
            ui.allocate_ui(Vec2::new(240.0, 40.0), |ui| {
                let start = ui.cursor().left();
                let _ = cm3_dropdown(
                    ui,
                    Id::unique("test_dd_roomy"),
                    &mut selected,
                    &opts,
                    false,
                    vp,
                    &theme,
                );
                full_box_w = ui.min_rect().right() - start;
            });
        });
        // Label starts at +5px and clips at right - 19px => available label width = full_box_w - 24px.
        assert!(
            classic_w <= full_box_w - 24.0,
            "roomy dropdown ({full_box_w}px) must leave full clearance for 'Classic Dark' ({classic_w}px) before chevron"
        );

        // Constrained container: helper produces clean ellipsis fitting strictly within max_width.
        run_frame(&ctx, vec![], |ui| {
            let max_w = (classic_w * 0.65).max(24.0);
            let g = layout_truncated_dropdown_label(
                ui.painter(),
                "Classic Dark",
                theme.font(13.3),
                theme.value(),
                max_w,
            );
            assert!(
                g.text().ends_with('…'),
                "expected ellipsis, got '{}'",
                g.text()
            );
            assert!(
                g.size().x <= max_w + 0.5,
                "truncated label ({}px) must fit strictly within {}px before chevron",
                g.size().x,
                max_w
            );
        });
    }

    #[test]
    fn property_column_default_ratio_rebalanced_and_usable_at_220px() {
        let col = PropertyColumnState::default();
        assert!(
            (col.label_ratio - 0.48).abs() < 1e-6,
            "default PropertyColumnState::label_ratio must be 0.48, got {}",
            col.label_ratio
        );

        let ctx = egui::Context::default();
        super::super::theme::install_cm3_fonts(&ctx);
        let theme = Cm3Theme::default();
        let cfg = ScrubConfig::new(-10000.0, 10000.0, 1.0, 2);
        let mut vals = [100.0_f64, 100.0, 100.0];
        let mut row_right = 0.0_f32;
        let mut col2_left = 0.0_f32;
        let mut used_right = 0.0_f32;

        run_frame(&ctx, vec![], |ui| {
            ui.allocate_ui(Vec2::new(220.0, 120.0), |ui| {
                row_right = ui.max_rect().right();
                cm3_property_row_aligned(
                    ui,
                    "Position",
                    true,
                    col,
                    &theme,
                    |_| {},
                    |ui| {
                        col2_left = ui.cursor().left();
                        let _ = cm3_vector3(
                            ui,
                            Id::unique("test_vec3_220px"),
                            &mut vals,
                            cfg,
                            false,
                            &theme,
                        );
                        used_right = ui.min_rect().right();
                    },
                );
            });
        });

        let col2_w = row_right - col2_left;
        assert!(
            col2_w >= 100.0,
            "220px panel must preserve >=100px value column, got {col2_w}"
        );
        assert!(
            used_right <= row_right + 1.0,
            "220px panel Vec3 must not clip past right border: used_right={used_right}, row_right={row_right}"
        );
    }

    #[test]
    fn floating_windows_and_color_picker_dismiss_in_strict_lifo_order_on_escape() {
        let ctx = egui::Context::default();
        let theme = Cm3Theme::default();
        let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));
        let picker_id = Id::unique("test_lifo_color_picker");
        let mut color = Rgba::new(1.0, 1.0, 1.0, 1.0); // top-left white corner (QA6-01 #5)

        let mut mgr = FloatingWindowManager::default();
        let spec_a = FloatingWindowSpec::new("expr_editor", "Expression Editor");
        let spec_b = FloatingWindowSpec::new("shader_editor", "Shader Editor");
        mgr.open_or_focus(&spec_a, vp);
        mgr.open_or_focus(&spec_b, vp);
        assert_eq!(mgr.focused_id.as_deref(), Some("shader_editor"));

        // Open color picker popover on top of both floating windows.
        ctx.data_mut(|d| {
            d.insert_temp(
                picker_id.with("hsva_popup"),
                ColorPopupState {
                    open: true,
                    initial: color,
                    hsv: color.rgb().to_hsv(),
                    alpha: 1.0,
                    hex_buf: "rgb(255, 255, 255)".to_owned(),
                    dirty: false,
                },
            );
        });

        let esc_event = || {
            vec![egui::Event::Key {
                key: Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }]
        };

        // 1st Escape: closes ONLY the color picker; both floating windows stay open.
        let mut picker_ev = ColorPickerEvent::default();
        let mut ev_b = FloatingWindowEvent::default();
        let mut ev_a = FloatingWindowEvent::default();
        reset_transient_popover_escape_flag(&ctx);
        run_frame(&ctx, esc_event(), |ui| {
            picker_ev = cm3_hsva_color_picker(ui, picker_id, &mut color, false, false, vp, &theme);
            ev_b = cm3_floating_window(ui.ctx(), &mut mgr, &spec_b, vp, &theme, |_| {});
            ev_a = cm3_floating_window(ui.ctx(), &mut mgr, &spec_a, vp, &theme, |_| {});
        });
        assert!(picker_ev.cancelled, "1st Escape must dismiss color picker");
        assert!(
            !ev_b.closed && mgr.is_open("shader_editor"),
            "Shader Editor must stay open when color picker consumed Escape"
        );
        assert!(
            !ev_a.closed && mgr.is_open("expr_editor"),
            "Expression Editor must stay open when color picker consumed Escape"
        );

        // 2nd Escape: closes ONLY the topmost focused window (Shader Editor), transferring focus to Expression Editor.
        reset_transient_popover_escape_flag(&ctx);
        run_frame(&ctx, esc_event(), |ui| {
            ev_b = cm3_floating_window(ui.ctx(), &mut mgr, &spec_b, vp, &theme, |_| {});
            ev_a = cm3_floating_window(ui.ctx(), &mut mgr, &spec_a, vp, &theme, |_| {});
        });
        assert!(ev_b.closed && !mgr.is_open("shader_editor"));
        assert!(
            !ev_a.closed && mgr.is_open("expr_editor"),
            "Expression Editor must NOT close on the same Escape press that closed Shader Editor"
        );
        assert_eq!(mgr.focused_id.as_deref(), Some("expr_editor"));

        // 3rd Escape: closes Expression Editor.
        reset_transient_popover_escape_flag(&ctx);
        run_frame(&ctx, esc_event(), |ui| {
            ev_a = cm3_floating_window(ui.ctx(), &mut mgr, &spec_a, vp, &theme, |_| {});
        });
        assert!(ev_a.closed && !mgr.is_open("expr_editor"));
    }
}
