//! Two-column property rows and column splitter.

use crate::ui::theme::{
    BORDER_BUTTON_HOVER, BORDER_OPTION_ACTIVE, BORDER_ROW_SOFT, Cm3Theme, FONT_BODY, TEXT_PRIMARY,
};
use egui::{CornerRadius, CursorIcon, Id, Pos2, Rect, Sense, Stroke, Ui, UiBuilder, Vec2};

use super::*;

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
pub fn cm3_property_row_inner<R>(
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
