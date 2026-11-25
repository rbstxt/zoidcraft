//! Numeric scrub input and vector link widgets.

use crate::ui::theme::{
    BG_INPUT, Cm3Icon, Cm3Theme, FONT_BODY, TEXT_BRIGHT, TEXT_DISABLED, TEXT_PRIMARY,
    paint_cm3_icon,
};
use egui::{
    Color32, CornerRadius, CursorIcon, Frame, Id, Key, Pos2, Rect, Sense, Stroke, StrokeKind,
    TextEdit, Ui, Vec2,
};

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
pub enum ScrubMode {
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
pub struct ScrubMemory {
    pub mode: ScrubMode,
}

impl Default for ScrubMemory {
    fn default() -> Self {
        Self {
            mode: ScrubMode::Idle,
        }
    }
}

pub fn round_decimals(val: f64, decimals: usize) -> f64 {
    let factor = 10.0_f64.powi(decimals.min(9) as i32);
    (val * factor).round() / factor
}

pub fn clamp_and_round(val: f64, cfg: ScrubConfig) -> f64 {
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

pub fn clamp_stored_bounds(val: f64, cfg: ScrubConfig) -> f64 {
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
pub fn cm3_scrub_number_inner(
    ui: &mut Ui,
    id: Id,
    value: &mut f64,
    config: ScrubConfig,
    theme: &Cm3Theme,
) -> (ScrubEvent, Option<(f64, bool)>) {
    cm3_scrub_number_styled(ui, id, value, config, theme, FONT_BODY, 3.0)
}

pub fn cm3_scrub_number_styled(
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

pub fn cached_monospace_char_width(ui: &Ui, font_size: f32, theme: &Cm3Theme) -> f32 {
    let font = theme.font(font_size);
    let cache_id = Id::unique((
        "zc_mono_char_w",
        font.size.to_bits(),
        theme.editor_font,
        ui.pixels_per_point().to_bits(),
    ));
    if let Some(w) = ui.data(|d| d.get_temp::<f32>(cache_id)) {
        return w;
    }
    let w = ui
        .painter()
        .layout_no_wrap("0".to_owned(), font, theme.value())
        .size()
        .x
        .max(1.0);
    ui.data_mut(|d| d.insert_temp(cache_id, w));
    w
}

pub struct CharCounter(usize);

impl std::fmt::Write for CharCounter {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0 += s.chars().count();
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn measure_vector_inline_width(
    ui: &Ui,
    values: &[f64],
    config: ScrubConfig,
    sep: &str,
    theme: &Cm3Theme,
    font_size: f32,
    inset: f32,
    spacing: f32,
) -> f32 {
    use std::fmt::Write as _;

    let safe_scale = if config.scale_factor.is_finite() && config.scale_factor.abs() > 1e-12 {
        config.scale_factor
    } else {
        1.0
    };
    let char_w = cached_monospace_char_width(ui, font_size, theme);
    let sep_w = sep.chars().count() as f32 * char_w;
    let mut total = 0.0_f32;
    for (idx, &v) in values.iter().enumerate() {
        if idx > 0 {
            total += spacing + sep_w + spacing;
        }
        let disp = clamp_and_round(v / safe_scale, config);
        let mut counter = CharCounter(0);
        let _ = write!(&mut counter, "{:.*}", config.decimals, disp);
        let w = (counter.0 as f32 * char_w).max(14.0) + inset;
        total += w;
    }
    total
}

/// Chooses `(font_size, left_inset, item_spacing)` so all inline vector components fit inside
/// `avail_w` without clipping the rightmost component off the inspector edge.
pub fn choose_vector_inline_style(
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
pub struct VectorLinkMemory {
    pub active_axis: Option<usize>,
    pub ratios: [f64; 3],
    pub initial_values: [f64; 3],
    pub unlinked: bool,
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
