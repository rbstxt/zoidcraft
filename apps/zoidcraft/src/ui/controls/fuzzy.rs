//! Fuzzy match scoring and fuzzy picker popup.

use crate::ui::theme::{
    BG_OPTION_ACTIVE, BG_OPTION_HOVER, BG_PANEL, BG_TITLE, BORDER_FILTER, BORDER_OPTION_ACTIVE,
    Cm3Theme, FONT_BANNER, FONT_BODY, FONT_SMALL, TEXT_BRIGHT, TEXT_FAINT, TEXT_OPTION_DESC,
    TEXT_PRIMARY,
};
use egui::{
    Align2, CornerRadius, CursorIcon, Frame, Id, Key, Pos2, ScrollArea, Sense, Stroke, StrokeKind,
    TextEdit, Ui, Vec2,
};

use super::*;

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
