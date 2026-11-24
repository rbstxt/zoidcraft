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
    rgba_to_color32, stopwatch_active_color,
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
enum ScrubMode {
    Idle,
    PendingDrag {
        start_pos: Pos2,
        last_pos: Pos2,
        initial_display: f64,
        accum_display: f64,
        alt_at_start: bool,
    },
    Scrubbing {
        last_pos: Pos2,
        initial_display: f64,
        accum_display: f64,
        alt_at_start: bool,
    },
    TextEdit {
        buffer: String,
        initial_display: f64,
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
    let clamped = val.clamp(cfg.min, cfg.max);
    round_decimals(clamped, cfg.decimals)
}

/// Exact CM3 numeric scrub + click-to-type control (Spec 01 §4.3).
pub fn cm3_scrub_number(
    ui: &mut Ui,
    id: Id,
    value: &mut f64,
    config: ScrubConfig,
    theme: &Cm3Theme,
) -> ScrubEvent {
    let safe_scale = if config.scale_factor.abs() > 1e-12 {
        config.scale_factor
    } else {
        1.0
    };
    let display_val = clamp_and_round(*value / safe_scale, config);

    // Read-only state: #999999 text, default cursor, non-interactive.
    if config.read_only {
        let formatted = format!("{:.*}", config.decimals, display_val);
        let galley = ui
            .painter()
            .layout_no_wrap(formatted, theme.font(FONT_BODY), TEXT_DISABLED);
        let desired = Vec2::new((galley.size().x + 6.0).max(26.0), 24.0);
        let (rect, _) = ui.allocate_exact_size(desired, Sense::hover());
        ui.painter().galley(
            Pos2::new(rect.left() + 3.0, rect.center().y - galley.size().y * 0.5),
            galley,
            TEXT_DISABLED,
        );
        return ScrubEvent::default();
    }

    let mut mem: ScrubMemory = ui
        .data_mut(|d| d.get_temp::<ScrubMemory>(id))
        .unwrap_or_default();
    let mut ev = ScrubEvent::default();

    match &mut mem.mode {
        ScrubMode::TextEdit {
            buffer,
            initial_display,
            just_entered,
            ..
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
                *value = stepped * safe_scale;
                ev.changed = true;
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
                *value = *initial_display * safe_scale;
                ev.cancelled = true;
                ui.memory_mut(|m| m.surrender_focus(edit_id));
                mem.mode = ScrubMode::Idle;
            } else if commit_edit || resp.lost_focus() {
                let final_display = buffer
                    .trim()
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite())
                    .map_or(*initial_display, |v| clamp_and_round(v, config));
                *value = final_display * safe_scale;
                if (final_display - *initial_display).abs() > 1e-12 {
                    ev.committed = true;
                } else {
                    ev.cancelled = true;
                }
                ui.memory_mut(|m| m.surrender_focus(edit_id));
                mem.mode = ScrubMode::Idle;
            }
        }
        ScrubMode::Idle => {
            let formatted = format!("{:.*}", config.decimals, display_val);
            let galley =
                ui.painter()
                    .layout_no_wrap(formatted, theme.font(FONT_BODY), theme.value());
            let desired = Vec2::new((galley.size().x + 6.0).max(26.0), 24.0);
            let (rect, resp) = ui.allocate_exact_size(desired, Sense::click_and_drag());
            let resp = resp.on_hover_cursor(CursorIcon::Move);

            if resp.hovered() {
                ui.painter().rect_stroke(
                    rect,
                    CornerRadius::ZERO,
                    Stroke::new(1.0, color32_with_alpha(theme.accent(), 140)),
                    StrokeKind::Inside,
                );
            }
            ui.painter().galley(
                Pos2::new(rect.left() + 3.0, rect.center().y - galley.size().y * 0.5),
                galley,
                theme.value(),
            );

            if resp.is_pointer_button_down_on()
                && let Some(pos) = ui.input(|i| i.pointer.interact_pos())
            {
                let alt = ui.input(|i| i.modifiers.alt);
                mem.mode = ScrubMode::PendingDrag {
                    start_pos: pos,
                    last_pos: pos,
                    initial_display: display_val,
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
            accum_display,
            alt_at_start,
        } => {
            let formatted = format!("{:.*}", config.decimals, display_val);
            let galley = ui.painter().layout_no_wrap(
                formatted.clone(),
                theme.font(FONT_BODY),
                theme.value_strong(),
            );
            let desired = Vec2::new((galley.size().x + 6.0).max(26.0), 24.0);
            let (rect, _) = ui.allocate_exact_size(desired, Sense::click_and_drag());
            ui.painter().galley(
                Pos2::new(rect.left() + 3.0, rect.center().y - galley.size().y * 0.5),
                galley,
                theme.value_strong(),
            );
            ui.ctx().set_cursor_icon(CursorIcon::Move);

            let (down, pos_opt, esc) = ui.input(|i| {
                (
                    i.pointer.primary_down(),
                    i.pointer.latest_pos(),
                    i.key_pressed(Key::Escape),
                )
            });

            if esc {
                *value = *initial_display * safe_scale;
                ev.cancelled = true;
                mem.mode = ScrubMode::Idle;
            } else if !down {
                // Released before 2 px threshold -> discard drag transaction and enter TextEdit.
                ev.cancelled = true;
                mem.mode = ScrubMode::TextEdit {
                    buffer: formatted,
                    initial_display: *initial_display,
                    just_entered: true,
                    alt_at_start: *alt_at_start,
                };
            } else if let Some(pos) = pos_opt {
                let d2 = (pos.x - start_pos.x) - (pos.y - start_pos.y);
                if d2.abs() > 2.0 {
                    let init = *initial_display;
                    let accum = *accum_display;
                    let alt = *alt_at_start;
                    *last_pos = pos;
                    mem.mode = ScrubMode::Scrubbing {
                        last_pos: pos,
                        initial_display: init,
                        accum_display: accum,
                        alt_at_start: alt,
                    };
                }
            }
        }
        ScrubMode::Scrubbing {
            last_pos,
            initial_display,
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

            ui.ctx().set_cursor_icon(CursorIcon::Move);

            if esc {
                *value = *initial_display * safe_scale;
                ev.cancelled = true;
                mem.mode = ScrubMode::Idle;
            } else if let Some(pos) = pos_opt {
                let delta = f64::from((pos.x - last_pos.x) - (pos.y - last_pos.y));
                *last_pos = pos;
                if delta.abs() > 0.0 {
                    let m_shift = if shift { 10.0 } else { 1.0 };
                    let m_ctrl = if ctrl { 0.1 } else { 1.0 };
                    *accum_display = (*accum_display
                        + delta * config.step * 0.01 * m_shift * m_ctrl)
                        .clamp(config.min, config.max);
                    let rounded = round_decimals(*accum_display, config.decimals);
                    *value = rounded * safe_scale;
                    ev.changed = true;
                }
                if !down {
                    let final_display = clamp_and_round(*accum_display, config);
                    *value = final_display * safe_scale;
                    if (final_display - *initial_display).abs() > 1e-12 {
                        ev.committed = true;
                    } else {
                        ev.cancelled = true;
                    }
                    mem.mode = ScrubMode::Idle;
                }
            } else if !down {
                let final_display = clamp_and_round(*accum_display, config);
                *value = final_display * safe_scale;
                if (final_display - *initial_display).abs() > 1e-12 {
                    ev.committed = true;
                } else {
                    ev.cancelled = true;
                }
                mem.mode = ScrubMode::Idle;
            }

            let cur_disp = clamp_and_round(*value / safe_scale, config);
            let formatted = format!("{:.*}", config.decimals, cur_disp);
            let galley =
                ui.painter()
                    .layout_no_wrap(formatted, theme.font(FONT_BODY), theme.value_strong());
            let desired = Vec2::new((galley.size().x + 6.0).max(26.0), 24.0);
            let (rect, _) = ui.allocate_exact_size(desired, Sense::hover());
            ui.painter().rect_filled(rect, CornerRadius::ZERO, BG_INPUT);
            ui.painter().rect_stroke(
                rect,
                CornerRadius::ZERO,
                Stroke::new(1.0, theme.accent()),
                StrokeKind::Inside,
            );
            ui.painter().galley(
                Pos2::new(rect.left() + 3.0, rect.center().y - galley.size().y * 0.5),
                galley,
                theme.value_strong(),
            );
        }
    }

    ui.data_mut(|d| d.insert_temp(id, mem));
    ev
}

// ---------------------------------------------------------------------------
// 2. Vector2 / Vector3 Inputs with Link Ratio & Alt Temporary Unlink (§4.4)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct VectorLinkMemory {
    active_axis: Option<usize>,
    ratios: [f64; 3],
    unlinked: bool,
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
    let mut link_mem: VectorLinkMemory = ui
        .data_mut(|d| d.get_temp::<VectorLinkMemory>(id.with("link")))
        .unwrap_or_default();
    let alt_held = ui.input(|i| i.modifiers.alt);
    let mut total = ScrubEvent::default();

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for axis in 0..2 {
            if axis > 0 {
                ui.label(
                    egui::RichText::new(",")
                        .font(theme.font(FONT_BODY))
                        .color(TEXT_DISABLED),
                );
            }
            let ev = cm3_scrub_number(ui, id.with(axis), &mut values[axis], config, theme);
            if ev.started {
                link_mem.active_axis = Some(axis);
                link_mem.unlinked = alt_held || !link_ratio;
                if !link_mem.unlinked && values[axis].abs() > 1e-9 {
                    let denom = values[axis];
                    link_mem.ratios[0] = values[0] / denom;
                    link_mem.ratios[1] = values[1] / denom;
                }
            }
            if (ev.changed || ev.committed)
                && link_mem.active_axis == Some(axis)
                && !link_mem.unlinked
                && link_ratio
            {
                let other = 1 - axis;
                values[other] = (values[axis] * link_mem.ratios[other]).clamp(
                    config.min * config.scale_factor,
                    config.max * config.scale_factor,
                );
            }
            if ev.committed || ev.cancelled {
                link_mem.active_axis = None;
            }
            total = total.merge(ev);
        }
    });

    ui.data_mut(|d| d.insert_temp(id.with("link"), link_mem));
    total
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

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for axis in 0..3 {
            if axis > 0 {
                ui.label(
                    egui::RichText::new(",")
                        .font(theme.font(FONT_BODY))
                        .color(TEXT_DISABLED),
                );
            }
            let ev = cm3_scrub_number(ui, id.with(axis), &mut values[axis], config, theme);
            if ev.started {
                link_mem.active_axis = Some(axis);
                link_mem.unlinked = alt_held || !link_ratio;
                if !link_mem.unlinked && values[axis].abs() > 1e-9 {
                    let denom = values[axis];
                    for (ratio, &val) in link_mem.ratios.iter_mut().zip(values.iter()) {
                        *ratio = val / denom;
                    }
                }
            }
            if (ev.changed || ev.committed)
                && link_mem.active_axis == Some(axis)
                && !link_mem.unlinked
                && link_ratio
            {
                let v = values[axis];
                for (other, (slot, &ratio)) in
                    values.iter_mut().zip(link_mem.ratios.iter()).enumerate()
                {
                    if other != axis {
                        *slot = (v * ratio).clamp(
                            config.min * config.scale_factor,
                            config.max * config.scale_factor,
                        );
                    }
                }
            }
            if ev.committed || ev.cancelled {
                link_mem.active_axis = None;
            }
            total = total.merge(ev);
        }
    });

    ui.data_mut(|d| d.insert_temp(id.with("link"), link_mem));
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

/// Exact `17 x 17 px` CM3 stopwatch button (`#2a2a2a` inactive, hashed HSL active).
pub fn cm3_stopwatch_button(
    ui: &mut Ui,
    prop_name: &str,
    animated: bool,
    has_expression: bool,
    theme: &Cm3Theme,
) -> StopwatchEvent {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(21.0), Sense::click());
    let resp = resp.on_hover_cursor(CursorIcon::PointingHand);

    let icon_rect = Rect::from_center_size(rect.center(), Vec2::splat(17.0));
    let active_color = stopwatch_active_color(prop_name);

    if resp.hovered() {
        ui.painter().rect_stroke(
            rect,
            CornerRadius::ZERO,
            Stroke::new(2.0, theme.accent()),
            StrokeKind::Inside,
        );
    }

    let icon_color = if animated || has_expression {
        active_color
    } else if resp.hovered() {
        TEXT_MUTED
    } else {
        // Recessed CM3 inactive stopwatch (#2a2a2a fill over subtle dark rim)
        ui.painter()
            .circle_filled(icon_rect.center(), 6.0, BG_PANEL);
        Color32::from_rgb(0x4c, 0x4c, 0x4c)
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
        if cm3_icon_button(
            ui,
            Cm3Icon::PrevKf,
            17.0,
            prev_color,
            "previous keyframe",
            theme,
        ) && state.has_prev
        {
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
        if cm3_icon_button(
            ui,
            Cm3Icon::NextKf,
            17.0,
            next_color,
            "next keyframe",
            theme,
        ) && state.has_next
        {
            ev.next_kf = true;
        }

        // 4. Interpolation Toggle & Easing Dropdown (only when keyed & interpolated)
        if state.is_keyed && state.interpolated {
            let (interp_rect, interp_resp) =
                ui.allocate_exact_size(Vec2::new(37.0, 27.0), Sense::click());
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
            ui.painter().rect_stroke(
                interp_rect,
                CornerRadius::ZERO,
                Stroke::new(1.0, interp_border),
                StrokeKind::Inside,
            );
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
                let (ez_rect, ez_resp) =
                    ui.allocate_exact_size(Vec2::new(42.0, 27.0), Sense::click());
                let ez_resp = ez_resp
                    .on_hover_cursor(CursorIcon::PointingHand)
                    .on_hover_text("Open Easing+ Bezier Curve Editor");
                ui.painter()
                    .rect_filled(ez_rect, CornerRadius::ZERO, BG_INPUT);
                ui.painter().rect_stroke(
                    ez_rect,
                    CornerRadius::ZERO,
                    Stroke::new(
                        1.0,
                        if ez_resp.hovered() {
                            theme.accent()
                        } else {
                            BORDER_CONTROL
                        },
                    ),
                    StrokeKind::Inside,
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
                let (e_rect, e_resp) =
                    ui.allocate_exact_size(Vec2::new(37.0, 27.0), Sense::click());
                easing_btn_rect = e_rect;
                let e_resp = e_resp
                    .on_hover_cursor(CursorIcon::PointingHand)
                    .on_hover_text(format!("easing: {}", easing.display_name()));
                ui.painter()
                    .rect_filled(e_rect, CornerRadius::ZERO, BG_INPUT);
                ui.painter().rect_stroke(
                    e_rect,
                    CornerRadius::ZERO,
                    Stroke::new(
                        1.0,
                        if popup_state.open || e_resp.hovered() {
                            theme.accent()
                        } else {
                            BORDER_CONTROL
                        },
                    ),
                    StrokeKind::Inside,
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
        ui.input(|i| {
            if i.key_pressed(Key::Escape) {
                close_popup = true;
            } else if i.key_pressed(Key::ArrowDown) {
                popup_state.highlighted = (popup_state.highlighted + 1) % EasingKind::ALL_CM3.len();
            } else if i.key_pressed(Key::ArrowUp) {
                popup_state.highlighted = (popup_state.highlighted + EasingKind::ALL_CM3.len() - 1)
                    % EasingKind::ALL_CM3.len();
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
                                for (idx, kind) in EasingKind::ALL_CM3.iter().copied().enumerate() {
                                    let row_size = Vec2::new(ui.available_width(), 27.0);
                                    let (row_rect, row_resp) =
                                        ui.allocate_exact_size(row_size, Sense::click());
                                    if row_resp.hovered() {
                                        popup_state.highlighted = idx;
                                    }
                                    let is_selected = idx == usize::from(ease_idx);
                                    let is_hi = idx == popup_state.highlighted;
                                    if is_hi {
                                        ui.painter().rect_filled(
                                            row_rect,
                                            CornerRadius::ZERO,
                                            theme.menu_active(),
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
                                        if is_hi || is_selected {
                                            TEXT_BRIGHT
                                        } else {
                                            theme.value()
                                        },
                                    );
                                    ui.painter().text(
                                        Pos2::new(icon_rect.right() + 8.0, row_rect.center().y),
                                        Align2::LEFT_CENTER,
                                        kind.display_name(),
                                        theme.font(FONT_SMALL),
                                        if is_hi || is_selected {
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
        }
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
    // 12 varied colors + 6 grayscales + 18 full-saturation rainbow hues = 36 swatches (Spec 01 §4.5)
    let mut swatches = Vec::with_capacity(36);
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
    for i in 0..6 {
        let v = (i as f32 / 5.0 * 255.0).round() as u8;
        swatches.push(Rgb::from_u8(v, v, v));
    }
    for i in 0..18 {
        let h = i as f32 / 18.0;
        swatches.push(Rgb::from_hsv(Hsv { h, s: 1.0, v: 1.0 }));
    }
    swatches
}

fn push_recent_swatch(ui: &mut Ui, rgb: Rgb) {
    let key = Id::unique("zoidcraft_recent_color_swatches");
    let mut list = ui
        .data_mut(|d| d.get_persisted::<Vec<Rgb>>(key))
        .unwrap_or_else(default_recent_swatches);
    let target = rgb.to_u8();
    if let Some(pos) = list.iter().position(|c| c.to_u8() == target) {
        list.remove(pos);
    }
    list.insert(0, rgb);
    list.truncate(36);
    ui.data_mut(|d| d.insert_persisted(key, list));
}

fn read_recent_swatches(ui: &mut Ui) -> Vec<Rgb> {
    let key = Id::unique("zoidcraft_recent_color_swatches");
    ui.data_mut(|d| d.get_persisted::<Vec<Rgb>>(key))
        .unwrap_or_else(default_recent_swatches)
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
    let mut mesh = Mesh::default();
    // 2x2 quad: top-left white, top-right pure hue, bottom-left black, bottom-right black
    let tl = mesh.vertices.len() as u32;
    mesh.colored_vertex(rect.left_top(), Color32::WHITE);
    let tr = mesh.vertices.len() as u32;
    mesh.colored_vertex(rect.right_top(), pure);
    let br = mesh.vertices.len() as u32;
    mesh.colored_vertex(rect.right_bottom(), Color32::BLACK);
    let bl = mesh.vertices.len() as u32;
    mesh.colored_vertex(rect.left_bottom(), Color32::BLACK);
    mesh.add_triangle(tl, tr, br);
    mesh.add_triangle(tl, br, bl);
    painter.add(Shape::mesh(mesh));
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

/// CM3 `30x20` swatch + `17x17` eyedropper + `470x250`/`500x250` HSVA popup picker (Spec 01 §4.5).
pub fn cm3_hsva_color_picker(
    ui: &mut Ui,
    id: Id,
    color: &mut Rgba,
    has_alpha: bool,
    read_only: bool,
    viewport_rect: Rect,
    theme: &Cm3Theme,
) -> ColorPickerEvent {
    let mut ev = ColorPickerEvent::default();
    let popup_id = id.with("hsva_popup");
    let mut state = ui
        .data_mut(|d| d.get_temp::<ColorPopupState>(popup_id))
        .unwrap_or_else(|| ColorPopupState {
            open: false,
            initial: *color,
            hsv: color.rgb().to_hsv(),
            alpha: color.a(),
            hex_buf: if has_alpha {
                color.to_hex_rgba()
            } else {
                color.to_hex_rgb()
            },
            dirty: false,
        });

    let mut swatch_rect = Rect::NOTHING;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 10.0;
        let (rect, resp) = ui.allocate_exact_size(
            Vec2::new(30.0, 20.0),
            if read_only {
                Sense::hover()
            } else {
                Sense::click()
            },
        );
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
        ui.painter().rect_stroke(
            rect,
            CornerRadius::ZERO,
            Stroke::new(1.0, border),
            StrokeKind::Inside,
        );

        if !read_only {
            let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
            if resp.clicked() {
                if !state.open {
                    state.open = true;
                    state.initial = *color;
                    state.hsv = color.rgb().to_hsv();
                    state.alpha = if has_alpha { color.a() } else { 1.0 };
                    state.hex_buf = if has_alpha {
                        color.to_hex_rgba()
                    } else {
                        color.to_hex_rgb()
                    };
                    state.dirty = false;
                    ev.started = true;
                } else {
                    state.open = false;
                    if state.dirty {
                        push_recent_swatch(ui, color.rgb());
                        ev.committed = true;
                    } else {
                        ev.cancelled = true;
                    }
                }
            }

            // Eyedropper (17x17 px, 10px right of swatch)
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

    if state.open && !read_only && swatch_rect != Rect::NOTHING {
        let popup_size = Vec2::new(if has_alpha { 500.0 } else { 470.0 }, 260.0);
        let pos = clamp_popup_pos(swatch_rect, popup_size, viewport_rect);
        let mut commit_now = false;
        let mut cancel_now = false;

        let recent = read_recent_swatches(ui);

        let area_resp = Area::new(popup_id)
            .order(Order::Foreground)
            .fixed_pos(pos)
            .interactable(true)
            .show(ui.ctx(), |ui| {
                Frame::NONE
                    .fill(BG_PANEL)
                    .stroke(Stroke::new(1.0, theme.accent()))
                    .inner_margin(egui::Margin::same(5))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing = Vec2::new(5.0, 0.0);

                            // 1. 250x250 SV square
                            let (sv_rect, sv_resp) =
                                ui.allocate_exact_size(Vec2::splat(250.0), Sense::click_and_drag());
                            paint_sv_square(ui.painter(), sv_rect, state.hsv.h);
                            ui.painter().rect_stroke(
                                sv_rect,
                                CornerRadius::ZERO,
                                Stroke::new(1.0, BORDER_CONTROL),
                                StrokeKind::Inside,
                            );
                            if (sv_resp.clicked() || sv_resp.dragged())
                                && let Some(p) = ui.input(|i| i.pointer.interact_pos())
                            {
                                state.hsv.s =
                                    ((p.x - sv_rect.left()) / sv_rect.width()).clamp(0.0, 1.0);
                                state.hsv.v = (1.0 - (p.y - sv_rect.top()) / sv_rect.height())
                                    .clamp(0.0, 1.0);
                                let rgb = Rgb::from_hsv(state.hsv);
                                *color = rgb.with_alpha(if has_alpha { state.alpha } else { 1.0 });
                                state.hex_buf = if has_alpha {
                                    color.to_hex_rgba()
                                } else {
                                    color.to_hex_rgb()
                                };
                                state.dirty = true;
                                ev.changed = true;
                            }
                            let reticle = Pos2::new(
                                sv_rect.left() + state.hsv.s * sv_rect.width(),
                                sv_rect.top() + (1.0 - state.hsv.v) * sv_rect.height(),
                            );
                            ui.painter().circle_stroke(
                                reticle,
                                4.0,
                                Stroke::new(1.5, Color32::from_gray(0xaa)),
                            );

                            // 2. 25x250 Hue bar
                            let (hue_rect, hue_resp) = ui.allocate_exact_size(
                                Vec2::new(25.0, 250.0),
                                Sense::click_and_drag(),
                            );
                            paint_hue_bar(ui.painter(), hue_rect);
                            ui.painter().rect_stroke(
                                hue_rect,
                                CornerRadius::ZERO,
                                Stroke::new(1.0, BORDER_CONTROL),
                                StrokeKind::Inside,
                            );
                            if (hue_resp.clicked() || hue_resp.dragged())
                                && let Some(p) = ui.input(|i| i.pointer.interact_pos())
                            {
                                state.hsv.h =
                                    ((p.y - hue_rect.top()) / hue_rect.height()).clamp(0.0, 0.9999);
                                let rgb = Rgb::from_hsv(state.hsv);
                                *color = rgb.with_alpha(if has_alpha { state.alpha } else { 1.0 });
                                state.hex_buf = if has_alpha {
                                    color.to_hex_rgba()
                                } else {
                                    color.to_hex_rgb()
                                };
                                state.dirty = true;
                                ev.changed = true;
                            }
                            let hy = hue_rect.top() + state.hsv.h * hue_rect.height();
                            let h_cursor = Rect::from_min_max(
                                Pos2::new(hue_rect.left(), (hy - 3.0).max(hue_rect.top())),
                                Pos2::new(hue_rect.right(), (hy + 3.0).min(hue_rect.bottom())),
                            );
                            ui.painter().rect_stroke(
                                h_cursor,
                                CornerRadius::ZERO,
                                Stroke::new(1.5, Color32::from_gray(0xaa)),
                                StrokeKind::Inside,
                            );

                            // 3. Optional 25x250 Alpha bar
                            if has_alpha {
                                let (a_rect, a_resp) = ui.allocate_exact_size(
                                    Vec2::new(25.0, 250.0),
                                    Sense::click_and_drag(),
                                );
                                paint_alpha_bar(ui.painter(), a_rect, Rgb::from_hsv(state.hsv));
                                ui.painter().rect_stroke(
                                    a_rect,
                                    CornerRadius::ZERO,
                                    Stroke::new(1.0, BORDER_CONTROL),
                                    StrokeKind::Inside,
                                );
                                if (a_resp.clicked() || a_resp.dragged())
                                    && let Some(p) = ui.input(|i| i.pointer.interact_pos())
                                {
                                    state.alpha = (1.0 - (p.y - a_rect.top()) / a_rect.height())
                                        .clamp(0.0, 1.0);
                                    *color = Rgb::from_hsv(state.hsv).with_alpha(state.alpha);
                                    state.hex_buf = color.to_hex_rgba();
                                    state.dirty = true;
                                    ev.changed = true;
                                }
                                let ay = a_rect.top() + (1.0 - state.alpha) * a_rect.height();
                                let a_cursor = Rect::from_min_max(
                                    Pos2::new(a_rect.left(), (ay - 3.0).max(a_rect.top())),
                                    Pos2::new(a_rect.right(), (ay + 3.0).min(a_rect.bottom())),
                                );
                                ui.painter().rect_stroke(
                                    a_cursor,
                                    CornerRadius::ZERO,
                                    Stroke::new(1.5, Color32::from_gray(0xaa)),
                                    StrokeKind::Inside,
                                );
                            }

                            // 4. Right panel (185 px): comparison bar, 36 swatches, CSS hex input
                            ui.vertical(|ui| {
                                ui.set_width(185.0);
                                ui.spacing_mut().item_spacing = Vec2::new(4.0, 5.0);

                                // Split comparison bar (25px)
                                let (cmp_rect, cmp_resp) =
                                    ui.allocate_exact_size(Vec2::new(185.0, 25.0), Sense::click());
                                let mid_x = cmp_rect.center().x;
                                let left_half = Rect::from_min_max(
                                    cmp_rect.left_top(),
                                    Pos2::new(mid_x, cmp_rect.bottom()),
                                );
                                let right_half = Rect::from_min_max(
                                    Pos2::new(mid_x, cmp_rect.top()),
                                    cmp_rect.right_bottom(),
                                );
                                if has_alpha {
                                    paint_checkerboard(ui.painter(), cmp_rect, 5.0);
                                }
                                ui.painter().rect_filled(
                                    left_half,
                                    CornerRadius::ZERO,
                                    rgba_to_color32(state.initial),
                                );
                                ui.painter().rect_filled(
                                    right_half,
                                    CornerRadius::ZERO,
                                    rgba_to_color32(*color),
                                );
                                ui.painter().rect_stroke(
                                    cmp_rect,
                                    CornerRadius::ZERO,
                                    Stroke::new(1.0, BORDER_CONTROL),
                                    StrokeKind::Inside,
                                );
                                if cmp_resp.clicked()
                                    && cmp_resp
                                        .interact_pointer_pos()
                                        .is_some_and(|p| left_half.contains(p))
                                {
                                    *color = state.initial;
                                    state.hsv = state.initial.rgb().to_hsv();
                                    state.alpha = state.initial.a();
                                    state.hex_buf = if has_alpha {
                                        color.to_hex_rgba()
                                    } else {
                                        color.to_hex_rgb()
                                    };
                                    state.dirty = false;
                                    ev.changed = true;
                                }

                                // 36 swatches (6 x 6 grid, 27x27 tiles, 4px gap)
                                for row in 0..6 {
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 4.0;
                                        for col in 0..6 {
                                            let idx = row * 6 + col;
                                            let sw_rgb =
                                                recent.get(idx).copied().unwrap_or(Rgb::BLACK);
                                            let (s_rect, s_resp) = ui.allocate_exact_size(
                                                Vec2::new(27.5, 25.0),
                                                Sense::click(),
                                            );
                                            ui.painter().rect_filled(
                                                s_rect,
                                                CornerRadius::ZERO,
                                                rgb_to_color32(sw_rgb),
                                            );
                                            ui.painter().rect_stroke(
                                                s_rect,
                                                CornerRadius::ZERO,
                                                Stroke::new(
                                                    1.0,
                                                    if s_resp.hovered() {
                                                        TEXT_BRIGHT
                                                    } else {
                                                        BORDER_CONTROL
                                                    },
                                                ),
                                                StrokeKind::Inside,
                                            );
                                            if s_resp.clicked() {
                                                state.hsv = sw_rgb.to_hsv();
                                                *color = sw_rgb.with_alpha(if has_alpha {
                                                    state.alpha
                                                } else {
                                                    1.0
                                                });
                                                state.hex_buf = if has_alpha {
                                                    color.to_hex_rgba()
                                                } else {
                                                    color.to_hex_rgb()
                                                };
                                                state.dirty = true;
                                                ev.changed = true;
                                            }
                                        }
                                    });
                                }

                                // Bottom CSS hex input
                                let hex_id = popup_id.with("hex_input");
                                let hex_resp = Frame::NONE
                                    .fill(BG_INPUT)
                                    .stroke(Stroke::new(1.0, BORDER_CONTROL))
                                    .inner_margin(egui::Margin::symmetric(4, 2))
                                    .show(ui, |ui| {
                                        ui.add(
                                            TextEdit::singleline(&mut state.hex_buf)
                                                .id(hex_id)
                                                .font(theme.font(FONT_SMALL))
                                                .text_color(theme.value())
                                                .desired_width(175.0),
                                        )
                                    })
                                    .inner;
                                if hex_resp.lost_focus()
                                    || (hex_resp.has_focus()
                                        && ui.input(|i| i.key_pressed(Key::Enter)))
                                {
                                    if let Ok(parsed) = Rgba::from_hex(&state.hex_buf) {
                                        *color = if has_alpha {
                                            parsed
                                        } else {
                                            parsed.rgb().with_alpha(1.0)
                                        };
                                        state.hsv = color.rgb().to_hsv();
                                        state.alpha = color.a();
                                        state.dirty = true;
                                        ev.changed = true;
                                    }
                                    if hex_resp.has_focus()
                                        && ui.input(|i| i.key_pressed(Key::Enter))
                                    {
                                        ui.memory_mut(|m| m.surrender_focus(hex_id));
                                    }
                                }
                            });
                        });
                    });
            });

        let hex_focused = ui.memory(|m| m.has_focus(popup_id.with("hex_input")));
        ui.input(|i| {
            if i.key_pressed(Key::Escape) {
                cancel_now = true;
            } else if (!hex_focused && i.key_pressed(Key::Enter))
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
        } else if commit_now {
            state.open = false;
            if state.dirty {
                push_recent_swatch(ui, color.rgb());
                ev.committed = true;
            } else {
                ev.cancelled = true;
            }
        }
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
}

#[derive(Debug, Clone, Default)]
struct GradientBarState {
    selected_stop: usize,
    dragging_stop: Option<usize>,
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
            }
            if s_resp.dragged()
                && state.dragging_stop == Some(idx)
                && let Some(p) = ui.input(|i| i.pointer.interact_pos())
            {
                let t = f64::from(((p.x - lane_rect.left()) / lane_rect.width()).clamp(0.0, 1.0));
                if (stop.position - t).abs() > 1e-4 {
                    stop.position = t;
                    ev.changed = true;
                }
            }
            if s_resp.drag_stopped() && state.dragging_stop == Some(idx) {
                state.dragging_stop = None;
                ev.committed = true;
            }
            if s_resp.secondary_clicked() {
                remove_idx = Some(idx);
            }
        }

        if let Some(idx) = remove_idx
            && stops.len() > 1
        {
            stops.remove(idx);
            state.selected_stop = state.selected_stop.min(stops.len() - 1);
            ev.changed = true;
            ev.committed = true;
        }

        // Inline color editor for selected stop
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
                    id.with(("stop_color", state.selected_stop)),
                    &mut stop_rgba,
                    true,
                    false,
                    viewport_rect,
                    theme,
                );
                if cev.changed || cev.committed {
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
    let galley = ui.painter().layout_no_wrap(
        label.to_owned(),
        theme.font(FONT_BODY),
        match variant {
            ButtonVariant::Primary => TEXT_BRIGHT,
            ButtonVariant::Danger => DANGER_TEXT,
            ButtonVariant::Normal | ButtonVariant::Back => TEXT_BUTTON,
        },
    );
    let desired_w = if full_width {
        ui.available_width().max(24.0)
    } else {
        (galley.size().x + 18.0).min(ui.available_width().max(24.0))
    };
    let desired = Vec2::new(desired_w, (galley.size().y + 10.0).max(28.0));
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
        ButtonVariant::Primary => TEXT_BRIGHT,
        ButtonVariant::Danger => DANGER_TEXT,
        ButtonVariant::Normal | ButtonVariant::Back => {
            if resp.hovered() {
                TEXT_BRIGHT
            } else {
                TEXT_BUTTON
            }
        }
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

/// Compact CM3 action icon button (`17x17` or toolbar `25x25`) with `2px` focus/hover border.
pub fn cm3_icon_button(
    ui: &mut Ui,
    icon: Cm3Icon,
    icon_px: f32,
    color: Color32,
    tooltip: &str,
    theme: &Cm3Theme,
) -> bool {
    let box_px = icon_px + 4.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(box_px), Sense::click());
    let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
    if resp.hovered() {
        ui.painter().rect_stroke(
            rect,
            CornerRadius::ZERO,
            Stroke::new(2.0, theme.accent()),
            StrokeKind::Inside,
        );
    }
    let icon_rect = Rect::from_center_size(rect.center(), Vec2::splat(icon_px));
    paint_cm3_icon(ui.painter(), icon_rect, icon, color);
    resp.on_hover_text(tooltip).clicked()
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
    let mut open = ui
        .data_mut(|d| d.get_temp::<bool>(popup_id))
        .unwrap_or(false);
    let mut changed = false;

    let width = ui.available_width().clamp(90.0, 220.0);
    let (btn_rect, btn_resp) = ui.allocate_exact_size(Vec2::new(width, 25.0), Sense::click());
    let btn_resp = btn_resp.on_hover_cursor(CursorIcon::PointingHand);

    ui.painter()
        .rect_filled(btn_rect, CornerRadius::ZERO, BG_INPUT);
    ui.painter().rect_stroke(
        btn_rect,
        CornerRadius::ZERO,
        Stroke::new(
            1.0,
            if open || btn_resp.hovered() {
                theme.accent()
            } else {
                BORDER_CONTROL
            },
        ),
        StrokeKind::Inside,
    );

    let clip_painter = ui.painter().with_clip_rect(Rect::from_min_max(
        btn_rect.left_top(),
        Pos2::new(
            (btn_rect.right() - 18.0).max(btn_rect.left()),
            btn_rect.bottom(),
        ),
    ));
    clip_painter.text(
        Pos2::new(btn_rect.left() + 6.0, btn_rect.center().y),
        Align2::LEFT_CENTER,
        current_label,
        theme.font(FONT_BODY),
        theme.value(),
    );

    let caret_rect = Rect::from_center_size(
        Pos2::new(btn_rect.right() - 9.0, btn_rect.center().y),
        Vec2::splat(10.0),
    );
    paint_cm3_icon(ui.painter(), caret_rect, Cm3Icon::ChevronDown, TEXT_MUTED);

    if btn_resp.clicked() {
        open = !open;
    }

    if open {
        let row_h = 24.0;
        let popup_h = (options.len() as f32 * row_h + 6.0).min(280.0);
        let popup_size = Vec2::new(width.max(150.0), popup_h);
        let pos = clamp_popup_pos(btn_rect, popup_size, viewport_rect);
        let mut close_now = false;

        ui.input(|i| {
            if i.key_pressed(Key::Escape) {
                close_now = true;
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
                            .max_height(popup_h - 6.0)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                ui.spacing_mut().item_spacing = Vec2::ZERO;
                                for (idx, opt) in options.iter().enumerate() {
                                    let (r_rect, r_resp) = ui.allocate_exact_size(
                                        Vec2::new(ui.available_width(), row_h),
                                        Sense::click(),
                                    );
                                    let selected = idx == *selected_idx;
                                    if r_resp.hovered() {
                                        ui.painter().rect_filled(
                                            r_rect,
                                            CornerRadius::ZERO,
                                            theme.menu_active(),
                                        );
                                    } else if selected {
                                        ui.painter().rect_filled(
                                            r_rect,
                                            CornerRadius::ZERO,
                                            BG_DROPDOWN_HOVER,
                                        );
                                    }
                                    ui.painter().text(
                                        Pos2::new(r_rect.left() + 8.0, r_rect.center().y),
                                        Align2::LEFT_CENTER,
                                        *opt,
                                        theme.font(FONT_SMALL),
                                        if r_resp.hovered() || selected {
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
            });

        let outside = ui.input(|i| {
            i.pointer.primary_pressed()
                && i.pointer
                    .interact_pos()
                    .is_some_and(|p| !area_resp.response.rect.contains(p) && !btn_rect.contains(p))
        });
        if close_now || outside {
            open = false;
        }
    }

    ui.data_mut(|d| d.insert_temp(popup_id, open));
    changed
}

/// CM3 collapsible section with `50 ms` linear chevron rotation (`0 deg <-> -90 deg`, Spec 01 §4.7).
pub fn cm3_collapsible_section<R>(
    ui: &mut Ui,
    id: Id,
    title: &str,
    default_open: bool,
    theme: &Cm3Theme,
    add_body: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    let mut open = ui
        .data_mut(|d| d.get_temp::<bool>(id))
        .unwrap_or(default_open);

    // 50 ms linear animation matching Spec 01 §7
    let t = ui.ctx().animate_bool_with_time(id.with("anim"), open, 0.05);

    let (hdr_rect, hdr_resp) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), Sense::click());
    let hdr_resp = hdr_resp.on_hover_cursor(CursorIcon::PointingHand);
    ui.painter()
        .rect_filled(hdr_rect, CornerRadius::ZERO, BG_TITLE);
    ui.painter().line_segment(
        [hdr_rect.left_bottom(), hdr_rect.right_bottom()],
        Stroke::new(1.0, BORDER_ROW_SOFT),
    );

    // Rotated chevron triangle: -90 deg (pointing right) when t=0, 0 deg (pointing down) when t=1
    let angle = -std::f32::consts::FRAC_PI_2 * (1.0 - t);
    let center = Pos2::new(hdr_rect.left() + 12.0, hdr_rect.center().y);
    let rot = |dx: f32, dy: f32| -> Pos2 {
        let (s, c) = angle.sin_cos();
        Pos2::new(center.x + dx * c - dy * s, center.y + dx * s + dy * c)
    };
    let tri = vec![rot(-4.0, -2.5), rot(4.0, -2.5), rot(0.0, 3.5)];
    let chev_color = if hdr_resp.hovered() {
        TEXT_PRIMARY
    } else {
        TEXT_MUTED
    };
    ui.painter()
        .add(Shape::convex_polygon(tri, chev_color, Stroke::NONE));

    ui.painter().text(
        Pos2::new(hdr_rect.left() + 24.0, hdr_rect.center().y),
        Align2::LEFT_CENTER,
        title,
        theme.font(FONT_TITLE),
        TEXT_STRONG,
    );

    if hdr_resp.clicked() {
        open = !open;
    }
    ui.data_mut(|d| d.insert_temp(id, open));

    if open { Some(add_body(ui)) } else { None }
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
    ui.input(|i| {
        if i.key_pressed(Key::Escape) {
            event = FuzzyPickerEvent::Cancelled;
        } else if !filtered.is_empty() {
            if i.key_pressed(Key::ArrowDown) {
                state.active_index = (state.active_index + 1) % filtered.len();
            } else if i.key_pressed(Key::ArrowUp) {
                state.active_index = (state.active_index + filtered.len() - 1) % filtered.len();
            } else if i.key_pressed(Key::Enter) {
                event = FuzzyPickerEvent::Selected(filtered[state.active_index].id.clone());
            }
        }
    });
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
                    ui.painter().text(
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

                ui.painter().text(
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
                    ui.painter().text(
                        Pos2::new(row_rect.left() + 8.0, row_rect.top() + 31.0),
                        Align2::LEFT_CENTER,
                        &item.desc,
                        theme.font(FONT_SMALL),
                        TEXT_OPTION_DESC,
                    );
                }

                if row_resp.hovered() {
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

/// Shared column width state for CM3 property trees (`layout = 0`, default `50%`).
#[derive(Debug, Clone, Copy)]
pub struct PropertyColumnState {
    pub label_ratio: f32,
}

impl Default for PropertyColumnState {
    fn default() -> Self {
        Self { label_ratio: 0.48 }
    }
}

/// Interactive full-height `5 px` column splitter inside a property panel (Spec 01 §4.1).
pub fn cm3_column_adjust_splitter(
    ui: &mut Ui,
    id: Id,
    panel_rect: Rect,
    col_state: &mut PropertyColumnState,
) {
    let split_x = panel_rect.left() + (panel_rect.width() * col_state.label_ratio).max(10.0);
    let hit = Rect::from_min_max(
        Pos2::new(split_x - 2.5, panel_rect.top()),
        Pos2::new(split_x + 2.5, panel_rect.bottom()),
    );
    let resp = ui
        .interact(hit, id, Sense::drag())
        .on_hover_cursor(CursorIcon::ResizeHorizontal);
    if resp.dragged() {
        let new_w = (split_x + resp.drag_delta().x - panel_rect.left())
            .clamp(50.0, (panel_rect.width() - 60.0).max(50.0));
        col_state.label_ratio = (new_w / panel_rect.width().max(1.0)).clamp(0.15, 0.80);
    }
}

/// Renders one CM3 2-column property row (`1px solid #242424` divider, `5px` gap, Spec 01 §4.1).
pub fn cm3_property_row<R>(
    ui: &mut Ui,
    label: &str,
    col_state: PropertyColumnState,
    indent_depth: usize,
    theme: &Cm3Theme,
    left_extra: impl FnOnce(&mut Ui),
    right_value: impl FnOnce(&mut Ui) -> R,
) -> R {
    let total_w = ui.available_width().max(120.0);
    let label_w = (total_w * col_state.label_ratio).clamp(72.0, (total_w - 60.0).max(72.0));
    let indent_px = indent_depth as f32 * 15.0;

    let mut out: Option<R> = None;
    let row_resp = ui.horizontal(|ui| {
        ui.set_min_width(total_w);
        ui.spacing_mut().item_spacing.x = 5.0;

        // Column 1: optional stopwatch + label
        ui.allocate_ui_with_layout(
            Vec2::new(label_w, 26.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                if indent_px > 0.0 {
                    ui.add_space(indent_px);
                }
                left_extra(ui);
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(label)
                            .font(theme.font(FONT_BODY))
                            .color(TEXT_PRIMARY),
                    )
                    .truncate(),
                );
            },
        );

        // Column 2: value controls
        let val_w = (total_w - label_w - 10.0).max(40.0);
        ui.allocate_ui_with_layout(
            Vec2::new(val_w, 26.0),
            egui::Layout::top_down(egui::Align::LEFT),
            |ui| {
                out = Some(right_value(ui));
            },
        );
    });

    let r = row_resp.response.rect;
    ui.painter().line_segment(
        [
            Pos2::new(r.left(), r.bottom() + 1.0),
            Pos2::new(r.right(), r.bottom() + 1.0),
        ],
        Stroke::new(1.0, BORDER_ROW_SOFT),
    );
    ui.add_space(2.0);
    out.expect("property row value closure always runs")
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

    // Escape closes the focused floating window (Spec 01 §4.8)
    if is_focused && ctx.input(|i| i.key_pressed(Key::Escape)) {
        manager.close(&spec.id);
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
        (win.pos, Vec2::new(260.0, 30.0))
    } else {
        (
            clamp_window_pos(win.pos, win.size, viewport),
            Vec2::new(win.size.x.max(min_w), win.size.y.max(min_h)),
        )
    };

    let win_id = Id::unique(("zoidcraft_floating_window", &spec.id));
    let drag_alpha = if win.dragging { 235 } else { 255 }; // 0.92 opacity while dragging
    let body_bg = color32_with_alpha(BG_PANEL, drag_alpha);
    let title_bg = color32_with_alpha(
        if is_focused {
            theme.bg_title_focused()
        } else {
            BG_TITLE
        },
        drag_alpha,
    );
    let border_color = if is_focused {
        theme.accent()
    } else {
        BORDER_PANEL
    };

    let mut request_focus = false;
    let mut request_close = false;

    Area::new(win_id)
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
                title_clip.layout_no_wrap(spec.title.clone(), theme.font(FONT_TITLE), TEXT_STRONG);
            let title_pos = Pos2::new(
                title_rect.left() + 9.0,
                title_rect.center().y - title_galley.size().y * 0.5,
            );
            let title_right = title_pos.x + title_galley.size().x;
            title_clip.galley(title_pos, title_galley, TEXT_STRONG);

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

                // --- 5 Edge Resize Grips (E, W, S, SE, SW; hidden when maximized) ---
                if !win.maximized {
                    // East
                    let e_rect = Rect::from_min_max(
                        Pos2::new(outer_rect.right() - 4.0, title_rect.bottom()),
                        Pos2::new(outer_rect.right() + 6.0, outer_rect.bottom() - 12.0),
                    );
                    let e_resp = ui
                        .interact(e_rect, win_id.with("rz_e"), Sense::drag())
                        .on_hover_cursor(CursorIcon::ResizeHorizontal);
                    if e_resp.dragged() {
                        win.size.x = (win.size.x + e_resp.drag_delta().x).max(min_w);
                        request_focus = true;
                    }

                    // West (anchors right edge)
                    let w_rect = Rect::from_min_max(
                        Pos2::new(outer_rect.left() - 6.0, title_rect.bottom()),
                        Pos2::new(outer_rect.left() + 4.0, outer_rect.bottom() - 12.0),
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

                    // South
                    let s_rect = Rect::from_min_max(
                        Pos2::new(outer_rect.left() + 12.0, outer_rect.bottom() - 4.0),
                        Pos2::new(outer_rect.right() - 12.0, outer_rect.bottom() + 6.0),
                    );
                    let s_resp = ui
                        .interact(s_rect, win_id.with("rz_s"), Sense::drag())
                        .on_hover_cursor(CursorIcon::ResizeVertical);
                    if s_resp.dragged() {
                        win.size.y = (win.size.y + s_resp.drag_delta().y).max(min_h);
                        request_focus = true;
                    }

                    // South-East (20x20 grip + 6x6 #6d6d6d L-bracket)
                    let se_rect = Rect::from_min_max(
                        Pos2::new(outer_rect.right() - 13.0, outer_rect.bottom() - 13.0),
                        Pos2::new(outer_rect.right() + 7.0, outer_rect.bottom() + 7.0),
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

                    // South-West (anchors right edge)
                    let sw_rect = Rect::from_min_max(
                        Pos2::new(outer_rect.left() - 7.0, outer_rect.bottom() - 13.0),
                        Pos2::new(outer_rect.left() + 13.0, outer_rect.bottom() + 7.0),
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
            ev.focused = true;
        }
    }
    ev
}

// ---------------------------------------------------------------------------
// Shared Viewport Popup Clamping / Flipping Helper
// ---------------------------------------------------------------------------

fn clamp_popup_pos(anchor: Rect, popup_size: Vec2, viewport: Rect) -> Pos2 {
    let mut x = anchor.left();
    let mut y = anchor.bottom() + 2.0;

    if y + popup_size.y > viewport.bottom() - 4.0 {
        let flipped_y = anchor.top() - popup_size.y - 2.0;
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
    Pos2::new(x, y)
}
