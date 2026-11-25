//! Property buttons, icon buttons, dropdowns and collapsible sections.

use crate::ui::theme::{
    BG_BACK_BUTTON, BG_BUTTON, BG_BUTTON_PRESSED, BG_DROPDOWN_HOVER, BG_INPUT, BG_PANEL, BG_TITLE,
    BORDER_BUTTON_HOVER, BORDER_CONTROL, BORDER_ROW_SOFT, Cm3Icon, Cm3Theme, DANGER_TEXT,
    FONT_BODY, TEXT_BRIGHT, TEXT_BUTTON, TEXT_DISABLED, TEXT_MUTED, TEXT_PRIMARY, paint_cm3_icon,
};
use egui::{
    Align2, Area, Color32, CornerRadius, CursorIcon, Frame, Id, Key, Order, Painter, Pos2, Rect,
    Response, ScrollArea, Sense, Stroke, StrokeKind, Ui, Vec2,
};

use super::*;

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

pub fn cm3_icon_button_enabled(
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DropdownWidthCacheKey {
    pub option_count: usize,
    pub options_hash: u64,
    pub font_size_bits: u32,
    pub editor_font: crate::ui::theme::EditorFontChoice,
    pub ppp_bits: u32,
}

#[derive(Debug, Clone, Default)]
pub struct DropdownPopupState {
    pub open: bool,
    pub highlighted: usize,
    pub last_option_count: usize,
    pub open_epoch: u64,
    pub cached_max_opt_w: Option<(DropdownWidthCacheKey, f32)>,
}

pub fn hash_dropdown_options(options: &[&str]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for opt in options {
        for &b in opt.as_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

pub fn cached_dropdown_mono_glyph_widths(painter: &Painter, font: &egui::FontId) -> (f32, f32) {
    let ctx = painter.ctx();
    let cache_id = Id::unique((
        "zc_dd_label_metrics",
        font.size.to_bits(),
        ctx.pixels_per_point().to_bits(),
    ));
    if let Some(metrics) = ctx.data(|d| d.get_temp::<(f32, f32)>(cache_id)) {
        return metrics;
    }
    let char_w = painter
        .layout_no_wrap("M".to_owned(), font.clone(), Color32::WHITE)
        .size()
        .x
        .max(1.0);
    let ellipsis_w = painter
        .layout_no_wrap("…".to_owned(), font.clone(), Color32::WHITE)
        .size()
        .x
        .max(1.0);
    let metrics = (char_w, ellipsis_w);
    ctx.data_mut(|d| d.insert_temp(cache_id, metrics));
    metrics
}

/// Truncates `text` with a trailing ellipsis (`…`) so it fits strictly within `max_width`
/// using cached monospace glyph-width estimation + single-pass verification.
pub fn layout_truncated_dropdown_label(
    painter: &Painter,
    text: &str,
    font: egui::FontId,
    color: Color32,
    max_width: f32,
) -> std::sync::Arc<egui::Galley> {
    if max_width <= 6.0 {
        return painter.layout_no_wrap(String::new(), font, color);
    }
    let (char_w, ellipsis_w) = cached_dropdown_mono_glyph_widths(painter, &font);
    if ellipsis_w > max_width {
        return painter.layout_no_wrap("…".to_owned(), font, color);
    }
    let char_count = text.chars().count();
    let mut prefix_chars = ((max_width - ellipsis_w) / char_w).floor().max(0.0) as usize;
    prefix_chars = prefix_chars.min(char_count.saturating_sub(1));

    let make_candidate = |n: usize| -> String {
        let byte_end = text
            .char_indices()
            .nth(n)
            .map_or(text.len(), |(idx, _)| idx);
        let mut s = String::with_capacity(byte_end + 3);
        s.push_str(&text[..byte_end]);
        s.push('…');
        s
    };

    let mut best = painter.layout_no_wrap(make_candidate(prefix_chars), font.clone(), color);
    if best.size().x > max_width {
        while prefix_chars > 0 && best.size().x > max_width {
            prefix_chars -= 1;
            best = painter.layout_no_wrap(make_candidate(prefix_chars), font.clone(), color);
        }
        return best;
    }

    while prefix_chars + 1 < char_count && best.size().x + char_w * 0.5 <= max_width {
        let next = painter.layout_no_wrap(make_candidate(prefix_chars + 1), font.clone(), color);
        if next.size().x <= max_width {
            best = next;
            prefix_chars += 1;
        } else {
            break;
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
            cached_max_opt_w: None,
        });
    if popup_state.last_option_count != options.len() {
        popup_state.last_option_count = options.len();
        popup_state.open_epoch = popup_state.open_epoch.wrapping_add(1);
    }
    let mut changed = false;

    let font = theme.font(13.3);
    let max_opt_key = DropdownWidthCacheKey {
        option_count: options.len(),
        options_hash: hash_dropdown_options(options),
        font_size_bits: font.size.to_bits(),
        editor_font: theme.editor_font,
        ppp_bits: ui.pixels_per_point().to_bits(),
    };
    let mut current_galley = None;
    let max_opt_w = if let Some((cached_key, cached_w)) = popup_state.cached_max_opt_w
        && cached_key == max_opt_key
    {
        cached_w
    } else {
        let mut computed_max = 0.0_f32;
        for (idx, opt) in options.iter().enumerate() {
            let galley =
                ui.painter()
                    .layout_no_wrap((*opt).to_owned(), font.clone(), theme.value());
            computed_max = computed_max.max(galley.size().x);
            if idx == *selected_idx {
                current_galley = Some(galley);
            }
        }
        popup_state.cached_max_opt_w = Some((max_opt_key, computed_max));
        computed_max
    };
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
