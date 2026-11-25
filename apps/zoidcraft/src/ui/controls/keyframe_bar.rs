//! Stopwatch button and keyframe navigation bar.

use crate::ui::theme::{
    BG_DROPDOWN_HOVER, BG_INPUT, BORDER_CONTROL, Cm3Icon, Cm3Theme, FONT_EASING, FONT_SMALL,
    TEXT_BRIGHT, TEXT_FAINT, TEXT_PRIMARY, paint_cm3_icon, paint_easing_preview,
};
use egui::{
    Align2, Area, CornerRadius, CursorIcon, Frame, Id, Key, Order, Pos2, Rect, ScrollArea, Sense,
    Stroke, Ui, Vec2,
};
use zc_anim::EasingKind;

use super::*;

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
pub struct EasingPopupState {
    pub open: bool,
    pub highlighted: usize,
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
