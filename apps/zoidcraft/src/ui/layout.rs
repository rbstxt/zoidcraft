//! Pixel-exact recursive binary Split Panel engine, CM3 workspace shell,
//! animated Elevator rail, File & Playback toolbars, Audio Level Meter,
//! and global shortcut dispatcher (Spec 01 §1 & §6).

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

use egui::{
    Align, Align2, Area, Color32, CornerRadius, CursorIcon, Frame, Id, Key, Layout, Mesh, Order,
    Painter, Pos2, Rect, ScrollArea, Sense, Shape, Stroke, StrokeKind, TextEdit, Ui, UiBuilder,
    Vec2,
};
use zc_session::{Command, Session};

use super::panels::{ExportSettings, PanelAction, PanelsState, SideTab, show_side_panel};
use super::shortcuts::{selected_clip_is_scene3d, toggle_editing_camera_synced};
use super::theme::{
    AUDIO_PEAK_YELLOW, BG_DROPDOWN_HOVER, BG_INPUT, BG_PANEL, BG_SUNKEN, BG_TITLE, BG_TOOLBAR,
    BORDER_BUTTON_HOVER, BORDER_CONTROL, BORDER_OPTION_ACTIVE, BORDER_PANEL, BORDER_RAIL_FOCUS,
    Cm3Icon, Cm3Theme, EditorFontChoice, FONT_BODY, FONT_COMPACT, FONT_SMALL, FONT_TAB,
    TEXT_BRIGHT, TEXT_DISABLED, TEXT_MUTED, TEXT_PRIMARY, TEXT_STRONG, TOGGLE_ACTIVE,
    TOOLBAR_ICON_INACTIVE, VOID_BLACK, paint_cm3_icon,
};

// ---------------------------------------------------------------------------
// 0. Pixel-Snapping & Crisp 1px Border Helpers
// ---------------------------------------------------------------------------

/// Rounds a rectangle's edges to exact integer logical pixels while preserving non-inversion.
#[inline]
fn snap_rect(rect: Rect) -> Rect {
    let min_x = rect.min.x.round();
    let min_y = rect.min.y.round();
    let max_x = rect.max.x.round().max(min_x);
    let max_y = rect.max.y.round().max(min_y);
    Rect::from_min_max(Pos2::new(min_x, min_y), Pos2::new(max_x, max_y))
}

/// Appends an unfeathered integer-snapped solid quad into `mesh`.
#[inline]
fn push_crisp_quad(mesh: &mut Mesh, rect: Rect, color: Color32) {
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return;
    }
    let idx = mesh.vertices.len() as u32;
    mesh.colored_vertex(rect.left_top(), color);
    mesh.colored_vertex(rect.right_top(), color);
    mesh.colored_vertex(rect.right_bottom(), color);
    mesh.colored_vertex(rect.left_bottom(), color);
    mesh.add_triangle(idx, idx + 1, idx + 2);
    mesh.add_triangle(idx, idx + 2, idx + 3);
}

/// Paints an exact unfeathered filled rectangle via raw triangle mesh (zero epaint feathering).
#[inline]
fn paint_crisp_rect_filled(painter: &Painter, rect: Rect, color: Color32) {
    let r = snap_rect(rect);
    if r.width() <= 0.0 || r.height() <= 0.0 {
        return;
    }
    let mut mesh = Mesh::default();
    push_crisp_quad(&mut mesh, r, color);
    painter.add(Shape::mesh(mesh));
}

/// Paints a razor-sharp 1px border using 4 integer-snapped unfeathered edge quads inside `rect`.
#[inline]
fn paint_panel_border(painter: &Painter, rect: Rect, color: Color32) {
    let r = snap_rect(rect);
    if r.width() < 1.0 || r.height() < 1.0 {
        return;
    }
    let mut mesh = Mesh::default();
    let top_max_y = (r.min.y + 1.0).min(r.max.y);
    let bot_min_y = (r.max.y - 1.0).max(top_max_y);
    // Top 1px strip
    push_crisp_quad(
        &mut mesh,
        Rect::from_min_max(r.min, Pos2::new(r.max.x, top_max_y)),
        color,
    );
    // Bottom 1px strip (when height > 1px)
    if bot_min_y < r.max.y {
        push_crisp_quad(
            &mut mesh,
            Rect::from_min_max(Pos2::new(r.min.x, bot_min_y), r.max),
            color,
        );
    }
    // Left & right 1px strips between top and bottom (non-overlapping corners)
    if top_max_y < bot_min_y {
        let left_max_x = (r.min.x + 1.0).min(r.max.x);
        push_crisp_quad(
            &mut mesh,
            Rect::from_min_max(
                Pos2::new(r.min.x, top_max_y),
                Pos2::new(left_max_x, bot_min_y),
            ),
            color,
        );
        let right_min_x = (r.max.x - 1.0).max(left_max_x);
        if right_min_x < r.max.x {
            push_crisp_quad(
                &mut mesh,
                Rect::from_min_max(
                    Pos2::new(right_min_x, top_max_y),
                    Pos2::new(r.max.x, bot_min_y),
                ),
                color,
            );
        }
    }
    if !mesh.vertices.is_empty() {
        painter.add(Shape::mesh(mesh));
    }
}

/// Paints a solid fountain-pen nib icon pointing bottom-left (Zoidium L5 Shape/Pen tool).
fn paint_shape_pen_icon(painter: &Painter, rect: Rect, color: Color32) {
    let side = rect.width().min(rect.height()).max(1.0);
    let r = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let pt = |nx: f32, ny: f32| Pos2::new(r.left() + r.width() * nx, r.top() + r.height() * ny);
    let nib = vec![
        pt(0.16, 0.84),
        pt(0.28, 0.38),
        pt(0.66, 0.05),
        pt(0.95, 0.34),
        pt(0.62, 0.72),
    ];
    painter.add(Shape::convex_polygon(nib, color, Stroke::NONE));
}

/// Paints an outlined fountain-pen nib with solid center dot and tip slit (Zoidium L6 Bezier Pen tool).
fn paint_bezier_pen_icon(painter: &Painter, rect: Rect, color: Color32) {
    let side = rect.width().min(rect.height()).max(1.0);
    let r = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let pt = |nx: f32, ny: f32| Pos2::new(r.left() + r.width() * nx, r.top() + r.height() * ny);
    let stroke = Stroke::new((side * 0.072).clamp(1.3, 1.8), color);
    let nib = vec![
        pt(0.19, 0.81),
        pt(0.31, 0.39),
        pt(0.66, 0.08),
        pt(0.92, 0.34),
        pt(0.60, 0.69),
    ];
    painter.add(Shape::closed_line(nib, stroke));
    let center = pt(0.51, 0.44);
    painter.circle_filled(center, side * 0.125, color);
    painter.line_segment([pt(0.19, 0.81), pt(0.45, 0.51)], stroke);
}

/// Paints a compact solid diagonal pencil icon pointing bottom-left (Zoidium L7 Freehand Draw tool).
fn paint_draw_pencil_icon(painter: &Painter, rect: Rect, color: Color32) {
    let side = rect.width().min(rect.height()).max(1.0);
    let r = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let pt = |nx: f32, ny: f32| Pos2::new(r.left() + r.width() * nx, r.top() + r.height() * ny);
    let dp = |s: f32, w: f32| pt(0.16 + s + w, 0.84 - s + w);
    let hw = 0.095;
    let tip = vec![dp(0.0, 0.0), dp(0.16, -hw), dp(0.16, hw)];
    painter.add(Shape::convex_polygon(tip, color, Stroke::NONE));
    let shaft = vec![dp(0.15, -hw), dp(0.58, -hw), dp(0.58, hw), dp(0.15, hw)];
    painter.add(Shape::convex_polygon(shaft, color, Stroke::NONE));
}

/// Paints a crisp procedural motion-path trajectory icon (spline + 4x4 square endpoints + hollow center node).
fn paint_motion_path_icon(painter: &Painter, rect: Rect, color: Color32) {
    let side = rect.width().min(rect.height()).max(1.0);
    let r = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let pt = |nx: f32, ny: f32| Pos2::new(r.left() + r.width() * nx, r.top() + r.height() * ny);
    let sw = (side * 0.048).clamp(1.1, 1.35);
    let stroke = Stroke::new(sw, color);
    let mut curve = Vec::with_capacity(13);
    for i in 0..=12 {
        let t = i as f32 / 12.0;
        let inv = 1.0 - t;
        let x = inv * inv * inv * 0.20
            + 3.0 * inv * inv * t * 0.26
            + 3.0 * inv * t * t * 0.78
            + t * t * t * 0.84;
        let y = inv * inv * inv * 0.76
            + 3.0 * inv * inv * t * 0.34
            + 3.0 * inv * t * t * 0.66
            + t * t * t * 0.24;
        curve.push(pt(x, y));
    }
    painter.add(Shape::line(curve, stroke));
    let sq_size = Vec2::splat((side * 0.16).round().max(3.0));
    painter.rect_filled(
        Rect::from_center_size(pt(0.20, 0.76), sq_size),
        CornerRadius::ZERO,
        color,
    );
    painter.rect_filled(
        Rect::from_center_size(pt(0.84, 0.24), sq_size),
        CornerRadius::ZERO,
        color,
    );
    // epaint circle_stroke draws outside `radius`, so 0.075*25 + 1.2 ≈ 3.1px outer radius (6x6 ring).
    painter.circle_stroke(pt(0.50, 0.50), side * 0.075, stroke);
}

// ---------------------------------------------------------------------------
// 1. Recursive Binary Split Panel Engine (Spec 01 §1.1)
// ---------------------------------------------------------------------------

/// Exact CM3 visual splitter gap (`3 px`).
pub const SPLITTER_SIZE: f32 = 3.0;
/// Invisible pointer hover/drag hitbox thickness for split panel dividers (`8 px`).
pub const SPLITTER_HITBOX_SIZE: f32 = 8.0;
const HALF_SPLITTER: f32 = SPLITTER_SIZE * 0.5;

/// Computes the `8 px` invisible pointer hover/drag hitbox centered on `split_rect`
/// and clamped strictly inside `container`.
#[must_use]
pub fn compute_splitter_hitbox(split_rect: Rect, container: Rect, horiz: bool) -> Rect {
    let half_hit = SPLITTER_HITBOX_SIZE * 0.5;
    if horiz {
        let cx = split_rect.center().x;
        Rect::from_min_max(
            Pos2::new((cx - half_hit).max(container.left()), split_rect.top()),
            Pos2::new((cx + half_hit).min(container.right()), split_rect.bottom()),
        )
    } else {
        let cy = split_rect.center().y;
        Rect::from_min_max(
            Pos2::new(split_rect.left(), (cy - half_hit).max(container.top())),
            Pos2::new(split_rect.right(), (cy + half_hit).min(container.bottom())),
        )
    }
}

/// Split orientation matching Spec 01 §1.1 (`Vertical = 0`, `Horizontal = 1`, `Auto = -1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitOrientation {
    /// Top (`Pane 0`) over Bottom (`Pane 1`), cursor `ns-resize`.
    Vertical,
    /// Left (`Pane 0`) beside Right (`Pane 1`), cursor `ew-resize`.
    Horizontal,
    /// Chooses `Horizontal` when `width > height`, else `Vertical`.
    Auto,
}

/// Persistent ratio and minimum-size constraints for one binary Split Panel node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitPanelState {
    pub ratio: f32,
    pub orientation: SplitOrientation,
    pub min_pane_0: f32,
    pub min_pane_1: f32,
}

impl SplitPanelState {
    #[must_use]
    pub const fn new(
        ratio: f32,
        orientation: SplitOrientation,
        min_pane_0: f32,
        min_pane_1: f32,
    ) -> Self {
        Self {
            ratio,
            orientation,
            min_pane_0,
            min_pane_1,
        }
    }

    #[must_use]
    pub fn resolved_horizontal(&self, rect: Rect) -> bool {
        match self.orientation {
            SplitOrientation::Horizontal => true,
            SplitOrientation::Vertical => false,
            SplitOrientation::Auto => rect.width() > rect.height(),
        }
    }

    /// Computes `(pane_0_rect, splitter_rect, pane_1_rect)` using exact Spec 01 §1.1 math
    /// and clamping precedence (`Pane 1`'s minimum wins if container shrinks below sum),
    /// strictly guarding tiny containers against inverted rects.
    #[must_use]
    pub fn compute_rects(&self, container: Rect) -> (Rect, Rect, Rect) {
        let container = snap_rect(container);
        let horiz = self.resolved_horizontal(container);
        let extent = if horiz {
            container.width().max(0.0)
        } else {
            container.height().max(0.0)
        };
        let split_w = SPLITTER_SIZE.min(extent).round();
        let half_split = split_w * 0.5;
        let r = self.ratio.clamp(0.0, 1.0);
        let max_start = (extent - split_w).max(0.0);

        let mut split_start = (extent * r - half_split).round().clamp(0.0, max_start);
        let mut split_end = split_start + split_w;

        // Clamping precedence (Spec 01 §1.1):
        // 1) Clamp split_start >= min_pane_0 (shifting split_end = split_start + split_w)
        let min0 = self.min_pane_0.round();
        let min1 = self.min_pane_1.round();
        if split_start < min0 {
            split_start = min0.min(max_start).max(0.0);
            split_end = (split_start + split_w).min(extent);
        }
        // 2) Clamp extent - split_end >= min_pane_1 (Pane 1's minimum wins)
        if extent - split_end < min1 {
            split_end = (extent - min1).round().clamp(split_w, extent);
            split_start = (split_end - split_w).max(0.0);
        }
        // Final integer-snapped non-inversion invariant: 0 <= split_start <= split_end <= extent
        split_start = split_start.round().clamp(0.0, max_start);
        split_end = (split_start + split_w).clamp(split_start, extent);

        if horiz {
            let x0 = container.left();
            let p0 = Rect::from_min_max(
                Pos2::new(x0, container.top()),
                Pos2::new(x0 + split_start, container.bottom()),
            );
            let split = Rect::from_min_max(
                Pos2::new(x0 + split_start, container.top()),
                Pos2::new(x0 + split_end, container.bottom()),
            );
            let p1 = Rect::from_min_max(
                Pos2::new(x0 + split_end, container.top()),
                container.right_bottom(),
            );
            (p0, split, p1)
        } else {
            let y0 = container.top();
            let p0 = Rect::from_min_max(
                Pos2::new(container.left(), y0),
                Pos2::new(container.right(), y0 + split_start),
            );
            let split = Rect::from_min_max(
                Pos2::new(container.left(), y0 + split_start),
                Pos2::new(container.right(), y0 + split_end),
            );
            let p1 = Rect::from_min_max(
                Pos2::new(container.left(), y0 + split_end),
                container.right_bottom(),
            );
            (p0, split, p1)
        }
    }
}

/// Per-drag anchor stored in egui temp memory so dragging a splitter tracks exact pointer
/// displacement from `press_origin` with zero Frame-1 `drag_delta()` snap and zero drift.
#[derive(Debug, Clone, Copy, Default)]
struct SplitDragAnchor {
    origin_axis: f32,
    start_p0_extent: f32,
}

fn update_split_ratio_from_anchor(
    rect: Rect,
    horiz: bool,
    state: &mut SplitPanelState,
    anchor: SplitDragAnchor,
    cur_axis: f32,
) -> bool {
    let extent = if horiz {
        rect.width().max(1.0)
    } else {
        rect.height().max(1.0)
    };
    let max_p0 = (extent - state.min_pane_1 - SPLITTER_SIZE).max(0.0);
    let min_p0 = state.min_pane_0.min(max_p0);
    let new_p0 = (anchor.start_p0_extent + (cur_axis - anchor.origin_axis)).clamp(min_p0, max_p0);
    let new_ratio = ((new_p0 + HALF_SPLITTER) / extent).clamp(0.0, 1.0);
    if (state.ratio - new_ratio).abs() > 1e-6 {
        state.ratio = new_ratio;
        true
    } else {
        false
    }
}

fn handle_split_drag_response(
    ui: &Ui,
    anchor_id: Id,
    rect: Rect,
    horiz: bool,
    p0: Rect,
    state: &mut SplitPanelState,
    resp: &egui::Response,
) -> bool {
    if !resp.dragged() {
        ui.ctx()
            .data_mut(|d| d.remove_temp::<SplitDragAnchor>(anchor_id));
        return false;
    }

    let cur_p0_extent = if horiz { p0.width() } else { p0.height() };
    let (press_origin, interact_pos) = ui
        .ctx()
        .input(|i| (i.pointer.press_origin(), i.pointer.interact_pos()));

    let anchor = if resp.drag_started() {
        press_origin.or(interact_pos).map(|origin| {
            let a = SplitDragAnchor {
                origin_axis: if horiz { origin.x } else { origin.y },
                start_p0_extent: cur_p0_extent,
            };
            ui.ctx().data_mut(|d| d.insert_temp(anchor_id, a));
            a
        })
    } else {
        ui.ctx()
            .data(|d| d.get_temp::<SplitDragAnchor>(anchor_id))
            .or_else(|| {
                let origin = press_origin.or(interact_pos)?;
                let a = SplitDragAnchor {
                    origin_axis: if horiz { origin.x } else { origin.y },
                    start_p0_extent: cur_p0_extent,
                };
                ui.ctx().data_mut(|d| d.insert_temp(anchor_id, a));
                Some(a)
            })
    };

    if let (Some(anchor), Some(pos)) = (anchor, interact_pos) {
        let cur_axis = if horiz { pos.x } else { pos.y };
        update_split_ratio_from_anchor(rect, horiz, state, anchor, cur_axis)
    } else if !resp.drag_started() {
        // Fallback when pointer position is unavailable: apply incremental delta only after frame 0
        // so a pre-click cursor jump can never snap the pane to its minimum extent.
        let extent = if horiz {
            rect.width().max(1.0)
        } else {
            rect.height().max(1.0)
        };
        let delta = if horiz {
            resp.drag_delta().x
        } else {
            resp.drag_delta().y
        };
        let max_p0 = (extent - state.min_pane_1 - SPLITTER_SIZE).max(0.0);
        let min_p0 = state.min_pane_0.min(max_p0);
        let new_p0 = (cur_p0_extent + delta).clamp(min_p0, max_p0);
        let new_ratio = ((new_p0 + HALF_SPLITTER) / extent).clamp(0.0, 1.0);
        if (state.ratio - new_ratio).abs() > 1e-6 {
            state.ratio = new_ratio;
            true
        } else {
            false
        }
    } else {
        false
    }
}

fn paint_splitter_bar(ui: &Ui, split_rect: Rect, horiz: bool, hovered: bool, dragged: bool) {
    ui.painter()
        .rect_filled(split_rect, CornerRadius::ZERO, VOID_BLACK);
    if hovered || dragged {
        let line_color = if dragged {
            BORDER_OPTION_ACTIVE
        } else {
            BORDER_BUTTON_HOVER
        };
        if horiz {
            let lx = split_rect.center().x.floor();
            paint_crisp_rect_filled(
                ui.painter(),
                Rect::from_min_max(
                    Pos2::new(lx, split_rect.top()),
                    Pos2::new(lx + 1.0, split_rect.bottom()),
                ),
                line_color,
            );
        } else {
            let ly = split_rect.center().y.floor();
            paint_crisp_rect_filled(
                ui.painter(),
                Rect::from_min_max(
                    Pos2::new(split_rect.left(), ly),
                    Pos2::new(split_rect.right(), ly + 1.0),
                ),
                line_color,
            );
        }
    }
}

/// Applies any active splitter drag BEFORE laying out child panes (zero-lag resize),
/// returning `(pane_0_rect, pane_1_rect)`. Must be paired with [`finish_split_panel`]
/// after both panes are drawn so the `8 px` splitter hitbox sits above child panes on the
/// same layer and never loses hover/drag to adjacent scrollbars or canvas regions.
fn pre_layout_split_panel(
    ui: &Ui,
    id: Id,
    rect: Rect,
    state: &mut SplitPanelState,
) -> (Rect, Rect) {
    let splitter_id = id.with("splitter");
    let anchor_id = splitter_id.with("drag_anchor");
    let horiz = state.resolved_horizontal(rect);
    if ui.ctx().is_being_dragged(splitter_id)
        && let Some(anchor) = ui.ctx().data(|d| d.get_temp::<SplitDragAnchor>(anchor_id))
        && let Some(pos) = ui.ctx().input(|i| i.pointer.interact_pos())
    {
        let cur_axis = if horiz { pos.x } else { pos.y };
        let _ = update_split_ratio_from_anchor(rect, horiz, state, anchor, cur_axis);
    }
    let (p0, _split, p1) = state.compute_rects(rect);
    (p0, p1)
}

/// Registers the `8 px` splitter hover/drag hitbox AFTER child panes have been drawn
/// and paints the `3 px` divider + crisp `1 px` hover/active line.
fn finish_split_panel(ui: &mut Ui, id: Id, rect: Rect, state: &mut SplitPanelState) {
    let splitter_id = id.with("splitter");
    let anchor_id = splitter_id.with("drag_anchor");
    let horiz = state.resolved_horizontal(rect);
    let (p0, split_rect, _p1) = state.compute_rects(rect);
    let hit_rect = compute_splitter_hitbox(split_rect, rect, horiz);

    let resp = ui
        .interact(hit_rect, splitter_id, Sense::drag())
        .on_hover_cursor(if horiz {
            CursorIcon::ResizeHorizontal
        } else {
            CursorIcon::ResizeVertical
        });
    if handle_split_drag_response(ui, anchor_id, rect, horiz, p0, state, &resp) {
        ui.ctx().request_repaint();
    }

    let (_p0, new_split_rect, _p1) = state.compute_rects(rect);
    paint_splitter_bar(ui, new_split_rect, horiz, resp.hovered(), resp.dragged());
}

/// Draws and handles dragging for a CM3 Split Panel bar (`8 px` invisible hover/drag hitbox
/// with crisp `1 px` hover/active line over the `3 px` visual gap), returning `(pane_0_rect, pane_1_rect)`.
pub fn begin_split_panel(
    ui: &mut Ui,
    id: Id,
    rect: Rect,
    state: &mut SplitPanelState,
) -> (Rect, Rect) {
    let splitter_id = id.with("splitter");
    let anchor_id = splitter_id.with("drag_anchor");
    let horiz = state.resolved_horizontal(rect);
    let (p0, split_rect, _p1) = state.compute_rects(rect);
    let hit_rect = compute_splitter_hitbox(split_rect, rect, horiz);

    let resp = ui
        .interact(hit_rect, splitter_id, Sense::drag())
        .on_hover_cursor(if horiz {
            CursorIcon::ResizeHorizontal
        } else {
            CursorIcon::ResizeVertical
        });
    let _ = handle_split_drag_response(ui, anchor_id, rect, horiz, p0, state, &resp);

    // Recompute rects AFTER updating ratio and paint splitter at its live zero-lag position.
    let (p0, new_split_rect, p1) = state.compute_rects(rect);
    paint_splitter_bar(ui, new_split_rect, horiz, resp.hovered(), resp.dragged());
    (p0, p1)
}

/// Renders one clipped child pane returned by [`begin_split_panel`].
pub fn show_split_pane(
    ui: &mut Ui,
    id: Id,
    pane_rect: Rect,
    draw_pane: impl FnOnce(&mut Ui, Rect),
) {
    ui.scope_builder(UiBuilder::new().max_rect(pane_rect).id_salt(id), |ui| {
        ui.set_clip_rect(ui.clip_rect().intersect(pane_rect));
        draw_pane(ui, pane_rect);
    });
}

/// Renders a recursive binary CM3 Split Panel with a draggable `3 px` splitter bar.
#[allow(dead_code)]
pub fn show_split_panel(
    ui: &mut Ui,
    id: Id,
    rect: Rect,
    state: &mut SplitPanelState,
    draw_pane_0: impl FnOnce(&mut Ui, Rect),
    draw_pane_1: impl FnOnce(&mut Ui, Rect),
) {
    let (p0, p1) = begin_split_panel(ui, id, rect, state);
    show_split_pane(ui, id.with("pane0"), p0, draw_pane_0);
    show_split_pane(ui, id.with("pane1"), p1, draw_pane_1);
}

// ---------------------------------------------------------------------------
// 2. Aspect-Fit Viewport Rect & Audio Level Meter (Spec 01 §1.5 & §1.7)
// ---------------------------------------------------------------------------

/// Computes the exact centered aspect-fit canvas rect inside `viewport_rect` (Spec 01 §1.5).
#[must_use]
pub fn compute_viewport_canvas_rect(viewport_rect: Rect, resolution: [u32; 2]) -> Rect {
    let w = f32::from(resolution[0].max(1) as u16).max(1.0);
    let h = f32::from(resolution[1].max(1) as u16).max(1.0);
    let vw = viewport_rect.width().max(1.0);
    let vh = viewport_rect.height().max(1.0);
    let s = (vw / w).min(vh / h);
    let cw = (w * s).round().max(1.0);
    let ch = (h * s).round().max(1.0);
    let left = viewport_rect.left() + (vw - cw) * 0.5;
    let top = viewport_rect.top() + (vh - ch) * 0.5;
    Rect::from_min_size(Pos2::new(left, top), Vec2::new(cw, ch))
}

/// Stateful stereo Audio Level Meter holding RMS & nonlinear peak-hold decay (Spec 01 §1.7).
#[derive(Debug, Clone, Copy, Default)]
pub struct AudioMeterState {
    /// Current combined 1024-sample normalized RMS `[0.0, 1.0]`.
    pub rms: f32,
    /// Decaying combined peak-hold level `[0.0, 1.0]`.
    pub peak_hold: f32,
    /// Left-channel normalized RMS `[0.0, 1.0]`.
    pub rms_l: f32,
    /// Right-channel normalized RMS `[0.0, 1.0]`.
    pub rms_r: f32,
    /// Left-channel decaying peak-hold `[0.0, 1.0]`.
    pub peak_hold_l: f32,
    /// Right-channel decaying peak-hold `[0.0, 1.0]`.
    pub peak_hold_r: f32,
}

impl AudioMeterState {
    /// Updates meter with latest `(rms, max_sample)` applying Spec 01 §1.7 nonlinear decay:
    /// `peak_{t+1} = max(max_sample, peak_t - 0.01 / exp(peak_t))`.
    pub fn update(&mut self, rms: f32, max_sample: f32) {
        let r = rms.clamp(0.0, 1.0);
        let m = max_sample.clamp(0.0, 1.0);
        self.rms = r;
        self.rms_l = r;
        self.rms_r = r;
        let decayed = (self.peak_hold - 0.01 / self.peak_hold.exp()).max(0.0);
        self.peak_hold = m.max(decayed);
        self.peak_hold_l = self.peak_hold;
        self.peak_hold_r = self.peak_hold;
    }

    /// Updates stereo L/R audio level meter with `dt`-scaled nonlinear peak-hold decay.
    pub fn update_stereo(&mut self, rms_l: f32, rms_r: f32, dt: f32) {
        let l = rms_l.clamp(0.0, 1.0);
        let r = rms_r.clamp(0.0, 1.0);
        let frames = (dt.clamp(0.0, 0.25) * 60.0).max(0.0);
        self.rms_l = l;
        self.rms_r = r;
        self.rms = l.max(r);
        let decayed_l = (self.peak_hold_l - (0.01 * frames) / self.peak_hold_l.exp()).max(0.0);
        let decayed_r = (self.peak_hold_r - (0.01 * frames) / self.peak_hold_r.exp()).max(0.0);
        self.peak_hold_l = l.max(decayed_l);
        self.peak_hold_r = r.max(decayed_r);
        self.peak_hold = self.peak_hold_l.max(self.peak_hold_r);
    }
}

fn paint_meter_channel(ui: &mut Ui, bar: Rect, rms: f32, peak_hold: f32) {
    let bar = snap_rect(bar);
    if bar.width() <= 1.0 || bar.height() <= 2.0 {
        return;
    }
    // Bottom-to-top RGB gradient calibrated to Zoidium's silent & active meter:
    // #009000 (0%) -> #ffff00 (88%) -> #ff0000 (100%)
    let y_split = (bar.bottom() - bar.height() * 0.88).round();
    let green = Color32::from_rgb(0x00, 0x90, 0x00);
    let yellow = Color32::from_rgb(0xff, 0xff, 0x00);
    let red = Color32::from_rgb(0xff, 0x00, 0x00);

    let mut mesh = Mesh::default();
    let bl = mesh.vertices.len() as u32;
    mesh.colored_vertex(bar.left_bottom(), green);
    let br = mesh.vertices.len() as u32;
    mesh.colored_vertex(bar.right_bottom(), green);
    let mr = mesh.vertices.len() as u32;
    mesh.colored_vertex(Pos2::new(bar.right(), y_split), yellow);
    let ml = mesh.vertices.len() as u32;
    mesh.colored_vertex(Pos2::new(bar.left(), y_split), yellow);
    mesh.add_triangle(bl, br, mr);
    mesh.add_triangle(bl, mr, ml);

    let tl = mesh.vertices.len() as u32;
    mesh.colored_vertex(bar.left_top(), red);
    let tr = mesh.vertices.len() as u32;
    mesh.colored_vertex(bar.right_top(), red);
    mesh.add_triangle(ml, mr, tr);
    mesh.add_triangle(ml, tr, tl);
    ui.painter().add(Shape::mesh(mesh));

    // Top-down dark mask (rgb(16,16,16) @ alpha 241) scaled to (1 - rms) so silent meter matches
    // Zoidium's dark-tinted silent gradient (29,16,16) -> (29,29,16) @ 88% -> (16,23,16)
    let mask_h = (bar.height() * (1.0 - rms.clamp(0.0, 1.0))).round();
    if mask_h >= 1.0 {
        let mask_rect = Rect::from_min_size(bar.left_top(), Vec2::new(bar.width(), mask_h));
        ui.painter().rect_filled(
            mask_rect,
            CornerRadius::ZERO,
            Color32::from_rgba_unmultiplied(16, 16, 16, 241),
        );
    }

    // Always paint crisp unfeathered 1px #cece62 peak-hold line (rests at bar.bottom() - 1.0 when silent)
    let clamped_peak = peak_hold.clamp(0.0, 1.0);
    let py = ((bar.bottom() - 1.0) - (bar.height() - 1.0).max(0.0) * clamped_peak)
        .round()
        .clamp(bar.top(), bar.bottom() - 1.0);
    paint_crisp_rect_filled(
        ui.painter(),
        Rect::from_min_size(Pos2::new(bar.left(), py), Vec2::new(bar.width(), 1.0)),
        AUDIO_PEAK_YELLOW,
    );
}

/// Paints the exact `44 px` CM3 Audio Level Meter (Spec 01 §1.7).
pub fn paint_audio_meter(
    ui: &mut Ui,
    rect: Rect,
    focused: bool,
    meter: &AudioMeterState,
    theme: &Cm3Theme,
) {
    let rect = snap_rect(rect);
    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, BG_TOOLBAR);

    let inner = snap_rect(rect.shrink(5.0));
    if inner.width() > 2.0 && inner.height() > 2.0 {
        let (rms_l, peak_l, rms_r, peak_r) = if meter.rms_l == 0.0
            && meter.rms_r == 0.0
            && meter.peak_hold_l == 0.0
            && meter.peak_hold_r == 0.0
            && (meter.rms > 0.0 || meter.peak_hold > 0.0)
        {
            (meter.rms, meter.peak_hold, meter.rms, meter.peak_hold)
        } else {
            (
                meter.rms_l,
                meter.peak_hold_l,
                meter.rms_r,
                meter.peak_hold_r,
            )
        };
        paint_meter_channel(ui, inner, rms_l.max(rms_r), peak_l.max(peak_r));
    }

    let border = if focused {
        theme.accent()
    } else {
        BORDER_PANEL
    };
    paint_panel_border(ui.painter(), rect, border);
}

// ---------------------------------------------------------------------------
// 3. Focus, Player+, Actions & Workspace State
// ---------------------------------------------------------------------------

/// Active focused leaf panel inside the CM3 workspace (drives `1px` `#384668` focus border).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PanelFocus {
    #[default]
    None,
    Toolbar,
    ElevatorRail,
    SideInspector,
    Viewport,
    PlaybackToolbar,
    Timeline,
    AudioMeter,
}

/// Preview resolution divider for Zoidium `player-plus` plugin (Spec 04 §5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreviewQuality {
    #[default]
    Full,
    Half,
    Quarter,
    Eighth,
}

impl PreviewQuality {
    pub const ALL: [Self; 4] = [Self::Full, Self::Half, Self::Quarter, Self::Eighth];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Full => "Full",
            Self::Half => "Half",
            Self::Quarter => "Quarter",
            Self::Eighth => "1/8",
        }
    }

    /// Compact quality label used in narrow viewports (`800×540`) so `"Quarter"` never clips.
    #[must_use]
    pub const fn compact_label(self) -> &'static str {
        match self {
            Self::Full => "Full",
            Self::Half => "1/2",
            Self::Quarter => "1/4",
            Self::Eighth => "1/8",
        }
    }

    #[must_use]
    pub const fn divisor(self) -> u32 {
        match self {
            Self::Full => 1,
            Self::Half => 2,
            Self::Quarter => 4,
            Self::Eighth => 8,
        }
    }
}

/// Responsive geometry metrics for the Playback & Viewport bottom toolbar.
#[derive(Debug, Clone, Copy)]
pub struct PlaybackToolbarMetrics {
    pub btn_size: f32,
    pub btn_gap: f32,
    pub pill_gap: f32,
    pub icon_size: f32,
    pub edge_pad: f32,
    pub right_pad: f32,
    pub group_gap: f32,
    pub pill_font: f32,
    pub speed_dd_w: f32,
    pub q_w: f32,
    pub compact_quality_labels: bool,
    pub pill_left_inset: f32,
    pub pill_chevron_reserve: f32,
    pub chevron_size: f32,
    pub chevron_right_inset: f32,
}

/// Runtime state for `player-plus` viewport toolbar controls.
#[derive(Debug, Clone, Copy)]
pub struct PlayerPlusState {
    pub drawing_paused: bool,
    pub single_render_requested: bool,
    pub preview_quality: PreviewQuality,
    /// When enabled, dynamically steps down preview scale during playback when frames exceed budget.
    pub adaptive_quality: bool,
    /// Dynamic adaptive resolution divisor (`1`, `2`, `4`, `8`); resets to `1` on pause.
    pub adaptive_divisor: u32,
    /// Upscale filtering mode: `false` = Smooth (`Linear`), `true` = Sharp (`Nearest`).
    pub sharp_upscale: bool,
    /// Render effects at full sequence scale even when preview quality is reduced.
    pub exact_effects: bool,
    /// When `false` (default), playback steps through uncached frames without skipping so the
    /// preview cache builds a contiguous green bar before playing back at real-time speed.
    /// When `true`, playback always advances in real time and drops uncached frames if needed.
    pub drop_frames: bool,
}

impl Default for PlayerPlusState {
    fn default() -> Self {
        Self {
            drawing_paused: false,
            single_render_requested: false,
            preview_quality: PreviewQuality::Full,
            adaptive_quality: false,
            adaptive_divisor: 1,
            sharp_upscale: false,
            exact_effects: false,
            drop_frames: false,
        }
    }
}

/// High-level UI actions bubbled out of [`Cm3Workspace::show`] to `main.rs` / host shell.
#[derive(Debug, Clone)]
pub enum UiAction {
    NewProject,
    OpenProject,
    SaveProject {
        include_assets: bool,
    },
    ImportMedia,
    #[cfg(target_arch = "wasm32")]
    RelinkWebAsset(String),
    #[cfg(target_arch = "wasm32")]
    ImportWebSvg(zc_project::Uuid),
    OpenGraphEditor,
    PickAssetForProperty {
        target_id: String,
        property: String,
        is_lut: bool,
    },
    OpenGraphForProperty {
        target_id: String,
        property: String,
    },
    StartExport(ExportSettings),
    AddToExportQueue(ExportSettings),
    RunExportQueue,
    CancelExport,
    #[cfg(not(target_arch = "wasm32"))]
    OpenOutputFolder(PathBuf),
    #[cfg(not(target_arch = "wasm32"))]
    OpenFfmpegSetup,
    #[cfg(not(target_arch = "wasm32"))]
    BrowseFfmpegBinary,
    #[cfg(not(target_arch = "wasm32"))]
    ClearFfmpegBinary,
    #[cfg(not(target_arch = "wasm32"))]
    ReprobeFfmpeg,
    #[cfg(not(target_arch = "wasm32"))]
    SaveRecoverySnapshotNow,
    #[cfg(not(target_arch = "wasm32"))]
    RestoreRecoverySnapshot(PathBuf),
    #[cfg(not(target_arch = "wasm32"))]
    DeleteRecoverySnapshot(PathBuf),
    EyedropperActivate {
        target_id: String,
        property: String,
    },
    SingleFrameRender,
    StatusMessage(String),
}

/// Complete CM3 + Zoidium root workspace layout controller.
pub struct Cm3Workspace {
    /// Root horizontal split: Left Column (30%) | Right Workspace (70%).
    pub root_split: SplitPanelState,
    /// Right workspace vertical split: Viewport (65%) / Playback + Timeline (35%).
    pub right_split: SplitPanelState,
    /// Bottom-right horizontal split: Timeline | Audio Meter (45px fixed strip).
    pub timeline_meter_split: SplitPanelState,
    /// Active left-rail tab.
    pub active_tab: SideTab,
    /// Whether the Elevator sidebar drawer is expanded (`300 px`) vs collapsed (`60 px`).
    pub rail_expanded: bool,
    /// Current animated width of the Elevator rail (`60.0..=300.0`).
    rail_width_px: f32,
    /// Roving keyboard-focused tab index inside the Elevator rail.
    rail_focus_index: usize,
    /// Drag-hover auto-switch dwell timer `(tab, hover_start_time_sec)`.
    rail_drag_dwell: Option<(SideTab, f64)>,
    /// Currently focused leaf panel (drives `#384668` panel focus border & `` ` `` maximize).
    pub focused_panel: PanelFocus,
    /// If `Some`, that leaf panel is maximized across the entire workspace (`` ` `` shortcut).
    pub maximized_panel: Option<PanelFocus>,
    /// Inline editable project name buffer for the top-left Zoidium toolbar.
    project_name_buf: String,
    /// Whether the top-left project name field currently has an active edit session.
    project_name_editing: bool,
    /// Timestamp (seconds) when the project first became dirty, for the >5min save reminder accent dot.
    dirty_since_sec: Option<f64>,
    /// Audio level meter state (RMS + peak hold).
    pub audio_meter: AudioMeterState,
    /// Player+ toolbar state.
    pub player_plus: PlayerPlusState,
    /// Player+ viewport checkerboard transparency grid toggle.
    pub transparency_grid: bool,
    /// Motion Path toolbar toggle state.
    pub motion_path_visible: bool,
    /// Active viewport shape/vector drawing tool (`0` = shape pen, `1` = bezier pen, `2` = pencil).
    pub viewport_draw_tool: Option<u8>,
    /// Tracks whether the bottom inspector sub-pane (vs top list sub-pane) is focused on split tabs.
    split_inspector_focus_bottom: bool,
    /// Whether the Graph Editor floating window is currently open (drives Graph toolbar button border).
    pub graph_open: bool,
    /// Whether the Player+ playback speed / render-pause popup menu is open.
    speed_menu_open: bool,
    /// All side-panel inspector states.
    pub panels: PanelsState,
    /// When `true` (e.g. a modal confirmation dialog is open), suppresses workspace shortcuts.
    pub suppress_shortcuts: bool,
    /// When `true`, the host shell already ran `shortcuts::dispatch_shortcuts_with_focus`, so
    /// `handle_global_shortcuts` only processes workspace-exclusive keys (`Backtick` maximize).
    pub external_shortcuts_dispatched: bool,
    /// Cached `"Paused"` label widths keyed by `(font_scale_bits, editor_font, ppp_bits)`
    /// so the playback toolbar avoids re-shaping `"Paused"` twice on every 60Hz frame.
    cached_paused_label_widths: Option<((u32, EditorFontChoice, u32), f32, f32)>,
}

impl Default for Cm3Workspace {
    fn default() -> Self {
        Self::new()
    }
}

impl Cm3Workspace {
    /// Creates a new CM3 workspace initialized with roomy, responsive split ratios.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root_split: SplitPanelState::new(0.31, SplitOrientation::Horizontal, 270.0, 420.0),
            right_split: SplitPanelState::new(0.65, SplitOrientation::Vertical, 120.0, 120.0),
            timeline_meter_split: SplitPanelState::new(
                1.0,
                SplitOrientation::Horizontal,
                240.0,
                44.0,
            ),
            active_tab: SideTab::Media,
            rail_expanded: false,
            rail_width_px: 60.0,
            rail_focus_index: 0,
            rail_drag_dwell: None,
            focused_panel: PanelFocus::None,
            maximized_panel: None,
            project_name_buf: "project".to_owned(),
            project_name_editing: false,
            dirty_since_sec: None,
            audio_meter: AudioMeterState::default(),
            player_plus: PlayerPlusState::default(),
            transparency_grid: false,
            motion_path_visible: false,
            viewport_draw_tool: None,
            split_inspector_focus_bottom: false,
            graph_open: false,
            speed_menu_open: false,
            panels: PanelsState::default(),
            suppress_shortcuts: false,
            external_shortcuts_dispatched: false,
            cached_paused_label_widths: None,
        }
    }

    /// Responsive minimum sidebar width: keeps inspector controls cleanly usable (`>= 300px`)
    /// at standard/wide windows (`>= 1000px`) while scaling smoothly down to `270px` for `800×540`.
    #[must_use]
    pub fn compute_root_min_sidebar_width(window_width: f32) -> f32 {
        if window_width >= 1000.0 {
            300.0
        } else if window_width <= 800.0 {
            270.0
        } else {
            let t = (window_width - 800.0) / 200.0;
            (270.0 + t * 30.0).round()
        }
    }

    fn clicked_in_pane(ui: &Ui, rect: Rect) -> bool {
        let layer_id = ui.layer_id();
        let press_pos = ui.ctx().input(|i| {
            if i.pointer.primary_pressed() {
                i.pointer.interact_pos()
            } else {
                None
            }
        });
        press_pos.is_some_and(|p| rect.contains(p) && ui.ctx().layer_id_at(p) == Some(layer_id))
    }

    /// Renders the complete pixel-exact CM3 workspace inside `ui.max_rect()` and returns
    /// any host-level [`UiAction`]s triggered this frame.
    ///
    /// - `draw_viewport(ui, rect, is_focused)` plugs in `viewport.rs` (3D/2D compositor canvas).
    /// - `draw_timeline(ui, rect, is_focused)` plugs in `timeline.rs` (ruler, dope sheet, tracks).
    pub fn show(
        &mut self,
        ui: &mut Ui,
        session: &mut Session,
        theme: &mut Cm3Theme,
        mut draw_viewport: impl FnMut(&mut Ui, Rect, bool, &mut Session),
        mut draw_timeline: impl FnMut(&mut Ui, Rect, bool, &mut Session),
    ) -> Vec<UiAction> {
        let mut actions = Vec::new();
        let full_rect = ui.max_rect();

        // Flush integer-snapped root rect so panel 1px #333333 borders sit flush at window edges
        // and the active elevator tab sits flush at x=0 (no spurious 1px outer black perimeter).
        let root_rect = Rect::from_min_max(
            Pos2::new(full_rect.min.x.round(), full_rect.min.y.round()),
            Pos2::new(full_rect.max.x.round(), full_rect.max.y.round()),
        );
        ui.painter()
            .rect_filled(root_rect, CornerRadius::ZERO, VOID_BLACK);

        // Global shortcuts (strictly guarded against active text inputs & floating windows)
        self.handle_global_shortcuts(ui.ctx(), session, &mut actions);

        // Maximized leaf panel mode (CM3 Backquote `` ` `` shortcut, Spec 01 §6.2)
        if let Some(max_panel) = self.maximized_panel {
            match max_panel {
                PanelFocus::Viewport => {
                    self.show_viewport_frame(ui, root_rect, session, theme, &mut draw_viewport);
                    return actions;
                }
                PanelFocus::Timeline => {
                    self.show_timeline_frame(ui, root_rect, session, theme, &mut draw_timeline);
                    return actions;
                }
                PanelFocus::SideInspector => {
                    self.show_left_lower_pane(
                        ui,
                        root_rect,
                        root_rect,
                        session,
                        theme,
                        &mut actions,
                    );
                    return actions;
                }
                PanelFocus::None
                | PanelFocus::Toolbar
                | PanelFocus::ElevatorRail
                | PanelFocus::PlaybackToolbar
                | PanelFocus::AudioMeter => {
                    // Unsupported focus targets clear maximize and fall through without blanking.
                    self.maximized_panel = None;
                }
            }
        }

        // Keep minimum sidebar width responsive (>= 300px at >= 1000px, scaling smoothly down to 270px at 800x540).
        self.root_split.min_pane_0 = Self::compute_root_min_sidebar_width(root_rect.width());

        // Root Horizontal Split: 30% Left Column | 70% Right Workspace
        let (left_rect, right_rect) = pre_layout_split_panel(
            ui,
            Id::unique("cm3_root_split"),
            root_rect,
            &mut self.root_split,
        );

        show_split_pane(
            ui,
            Id::unique("cm3_root_split_p0"),
            left_rect,
            |ui, left_rect| {
                let left_rect = snap_rect(left_rect);
                // Fixed 40px File & History Toolbar at top + 3px black gap + Elevator/Inspector below
                let toolbar_h = 40.0_f32.min(left_rect.height()).round();
                let toolbar_rect = snap_rect(Rect::from_min_size(
                    left_rect.left_top(),
                    Vec2::new(left_rect.width(), toolbar_h),
                ));
                let lower_rect = snap_rect(Rect::from_min_max(
                    Pos2::new(
                        left_rect.left(),
                        (toolbar_rect.bottom() + SPLITTER_SIZE).min(left_rect.bottom()),
                    ),
                    left_rect.right_bottom(),
                ));

                self.show_file_history_toolbar(ui, toolbar_rect, session, theme, &mut actions);
                self.show_left_lower_pane(ui, lower_rect, root_rect, session, theme, &mut actions);
            },
        );

        show_split_pane(
            ui,
            Id::unique("cm3_root_split_p1"),
            right_rect,
            |ui, right_rect| {
                // Vertical Split: 65% Viewport / 35% (40px Playback Toolbar + Timeline/Audio Meter).
                // The 8px hitbox sits strictly on the viewport/playback-toolbar seam (40+ px above
                // the timeline ruler) and is registered post-pane so child widgets cannot steal it.
                let (vp_rect, bottom_rect) = pre_layout_split_panel(
                    ui,
                    Id::unique("cm3_right_split"),
                    right_rect,
                    &mut self.right_split,
                );
                show_split_pane(
                    ui,
                    Id::unique("cm3_right_split_p0"),
                    vp_rect,
                    |ui, vp_rect| {
                        self.show_viewport_frame(ui, vp_rect, session, theme, &mut draw_viewport);
                    },
                );
                show_split_pane(
                    ui,
                    Id::unique("cm3_right_split_p1"),
                    bottom_rect,
                    |ui, bottom_rect| {
                        let bottom_rect = snap_rect(bottom_rect);
                        let pb_h = 40.0_f32.min(bottom_rect.height()).round();
                        let pb_rect = snap_rect(Rect::from_min_size(
                            bottom_rect.left_top(),
                            Vec2::new(bottom_rect.width(), pb_h),
                        ));
                        let tl_row_rect = snap_rect(Rect::from_min_max(
                            Pos2::new(
                                bottom_rect.left(),
                                (pb_rect.bottom() + SPLITTER_SIZE).min(bottom_rect.bottom()),
                            ),
                            bottom_rect.right_bottom(),
                        ));

                        self.show_playback_toolbar(ui, pb_rect, session, theme, &mut actions);

                        // Fixed 44px Audio Level Meter right strip with non-draggable 3px divider
                        let avail_w = tl_row_rect.width().max(0.0);
                        let meter_w = 44.0_f32.min(avail_w).round();
                        let div_w = SPLITTER_SIZE.min((avail_w - meter_w).max(0.0)).round();
                        let tl_w = (avail_w - meter_w - div_w).max(0.0).round();
                        let tl_rect = snap_rect(Rect::from_min_size(
                            tl_row_rect.left_top(),
                            Vec2::new(tl_w, tl_row_rect.height()),
                        ));
                        let div_rect = snap_rect(Rect::from_min_size(
                            Pos2::new(tl_rect.right(), tl_row_rect.top()),
                            Vec2::new(div_w, tl_row_rect.height()),
                        ));
                        let meter_rect = snap_rect(Rect::from_min_size(
                            Pos2::new(div_rect.right(), tl_row_rect.top()),
                            Vec2::new(meter_w, tl_row_rect.height()),
                        ));
                        ui.painter()
                            .rect_filled(div_rect, CornerRadius::ZERO, VOID_BLACK);

                        show_split_pane(
                            ui,
                            Id::unique("cm3_timeline_meter_p0"),
                            tl_rect,
                            |ui, tl_rect| {
                                self.show_timeline_frame(
                                    ui,
                                    tl_rect,
                                    session,
                                    theme,
                                    &mut draw_timeline,
                                );
                            },
                        );
                        show_split_pane(
                            ui,
                            Id::unique("cm3_timeline_meter_p1"),
                            meter_rect,
                            |ui, meter_rect| {
                                if Self::clicked_in_pane(ui, meter_rect) {
                                    self.focused_panel = PanelFocus::AudioMeter;
                                    self.panels.floating_windows.clear_focus();
                                }
                                paint_audio_meter(
                                    ui,
                                    meter_rect,
                                    self.focused_panel == PanelFocus::AudioMeter,
                                    &self.audio_meter,
                                    theme,
                                );
                            },
                        );
                    },
                );
                finish_split_panel(
                    ui,
                    Id::unique("cm3_right_split"),
                    right_rect,
                    &mut self.right_split,
                );
            },
        );
        finish_split_panel(
            ui,
            Id::unique("cm3_root_split"),
            root_rect,
            &mut self.root_split,
        );

        actions
    }

    // -----------------------------------------------------------------------
    // Top-Left 40px File & History Toolbar (Spec 01 §1.3 + Zoidium 3-Col)
    // -----------------------------------------------------------------------

    fn show_file_history_toolbar(
        &mut self,
        ui: &mut Ui,
        rect: Rect,
        session: &mut Session,
        theme: &Cm3Theme,
        actions: &mut Vec<UiAction>,
    ) {
        let rect = snap_rect(rect);
        let focused = self.focused_panel == PanelFocus::Toolbar;
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, BG_TOOLBAR);

        if Self::clicked_in_pane(ui, rect) {
            self.focused_panel = PanelFocus::Toolbar;
            self.panels.floating_windows.clear_focus();
        }

        let inner = snap_rect(rect.shrink(1.0));
        let compact = inner.width() < 310.0;
        let btn_sz = if compact { 26.0_f32 } else { 29.0_f32 };
        let btn_gap = if compact { 3.0_f32 } else { 4.0_f32 };
        let sep_w = if compact { 7.0_f32 } else { 10.0_f32 };
        let icon_sz = if compact { 22.0_f32 } else { 25.0_f32 };
        let edge_pad = if compact { 4.0_f32 } else { 5.0_f32 };
        let name_gap = if compact { 5.0_f32 } else { 6.0_f32 };
        // 5 buttons (New, Open, Save | Undo, Redo) + 3 inner gaps + Undo/Redo separator
        let buttons_total_w = 5.0 * btn_sz + 3.0 * btn_gap + sep_w;
        let avail_for_name =
            (inner.width() - edge_pad - name_gap - buttons_total_w - edge_pad).max(0.0);
        let show_name_col = avail_for_name >= 56.0;

        let now = ui.input(|i| i.time);
        let is_dirty = session.is_dirty();
        if is_dirty {
            self.dirty_since_sec.get_or_insert(now);
        } else {
            self.dirty_since_sec = None;
        }
        let save_overdue = self
            .dirty_since_sec
            .is_some_and(|start| now - start >= 300.0);

        // Column 1: Zoidium inline project name input (always visible down to narrow widths)
        let name_edit_id = Id::unique("zoidium_toolbar_project_name");
        if !self.project_name_editing {
            let trimmed = session.document().project_name.trim();
            self.project_name_buf = if trimmed.is_empty() {
                "project".to_owned()
            } else {
                session.document().project_name.clone()
            };
        }

        let mut cursor_x = if show_name_col {
            let max_name_w = if compact { 112.0_f32 } else { 184.0_f32 };
            let name_w = avail_for_name.min(max_name_w).round();
            let name_h = 29.0_f32.min((inner.height() - 6.0).max(20.0)).round();
            let name_y = (inner.top() + (inner.height() - name_h) * 0.5).round();
            let name_rect = snap_rect(Rect::from_min_size(
                Pos2::new((inner.left() + edge_pad).round(), name_y),
                Vec2::new(name_w, name_h),
            ));
            let name_focused = ui.memory(|m| m.has_focus(name_edit_id));
            // Always paint BG_INPUT (#202020) fill & crisp 1px BORDER_CONTROL (#1b1b1b) border even when unfocused
            ui.painter()
                .rect_filled(name_rect, CornerRadius::ZERO, BG_INPUT);
            paint_panel_border(
                ui.painter(),
                name_rect,
                if name_focused {
                    theme.accent()
                } else {
                    BORDER_CONTROL
                },
            );
            let (esc_pressed, enter_pressed) =
                ui.input(|i| (i.key_pressed(Key::Escape), i.key_pressed(Key::Enter)));
            let edit_rect = Rect::from_min_max(
                Pos2::new(name_rect.left() + 6.0, name_rect.top() + 3.0),
                Pos2::new(
                    (name_rect.right() - 4.0).max(name_rect.left() + 7.0),
                    (name_rect.bottom() - 2.0).max(name_rect.top() + 5.0),
                ),
            );
            let resp = ui
                .scope_builder(
                    UiBuilder::new()
                        .max_rect(edit_rect)
                        .layout(Layout::left_to_right(Align::Center)),
                    |ui| {
                        ui.spacing_mut().interact_size.y = 0.0;
                        ui.add(
                            TextEdit::singleline(&mut self.project_name_buf)
                                .id(name_edit_id)
                                .frame(Frame::NONE)
                                .font(theme.font(if compact { FONT_SMALL } else { FONT_BODY }))
                                .text_color(theme.value())
                                .desired_width((name_w - 10.0).max(20.0)),
                        )
                    },
                )
                .inner;
            if resp.gained_focus() {
                self.project_name_editing = true;
                self.focused_panel = PanelFocus::Toolbar;
                self.panels.floating_windows.clear_focus();
            }
            if self.project_name_editing && esc_pressed {
                let fallback = session.document().project_name.trim();
                self.project_name_buf = if fallback.is_empty() {
                    "project".to_owned()
                } else {
                    session.document().project_name.clone()
                };
                self.project_name_editing = false;
                ui.memory_mut(|m| m.surrender_focus(name_edit_id));
            } else if self.project_name_editing
                && (resp.lost_focus() || (resp.has_focus() && enter_pressed))
            {
                let trimmed = self.project_name_buf.trim().to_owned();
                if !trimmed.is_empty() && trimmed != session.document().project_name {
                    let _ = session.execute(Command::SetProjectName { name: trimmed });
                } else {
                    let fallback = session.document().project_name.trim();
                    self.project_name_buf = if fallback.is_empty() {
                        "project".to_owned()
                    } else {
                        session.document().project_name.clone()
                    };
                }
                self.project_name_editing = false;
                ui.memory_mut(|m| m.surrender_focus(name_edit_id));
            } else if resp.has_focus() {
                self.project_name_editing = true;
            }
            (name_rect.right() + name_gap).round()
        } else {
            (inner.center().x - buttons_total_w * 0.5)
                .max(inner.left() + 2.0)
                .round()
        };

        // Column 2: New, Load, Save [dirty dot], divider gap, Undo, Redo
        let btn_y = (inner.top() + (inner.height() - btn_sz) * 0.5).round();
        let disabled_icon_tint = Color32::from_rgb(0x52, 0x52, 0x52);

        let toolbar_btn = |cursor_x: &mut f32,
                           icon: Cm3Icon,
                           tip: &str,
                           enabled: bool,
                           dirty_dot: bool,
                           advance_gap: f32|
         -> bool {
            let b_rect = snap_rect(Rect::from_min_size(
                Pos2::new(cursor_x.round(), btn_y),
                Vec2::splat(btn_sz),
            ));
            *cursor_x += btn_sz + advance_gap;
            let sense = if enabled {
                Sense::click()
            } else {
                Sense::hover()
            };
            let mut resp = ui
                .interact(b_rect, Id::unique(("ftb", tip)), sense)
                .on_hover_text(tip);
            if enabled {
                resp = resp.on_hover_cursor(CursorIcon::PointingHand);
            }
            if enabled && resp.hovered() && !resp.has_focus() {
                ui.painter().rect_filled(
                    b_rect,
                    CornerRadius::ZERO,
                    Color32::from_rgba_unmultiplied(0xff, 0xff, 0xff, 10),
                );
            }
            // 2px theme.accent() border ONLY on keyboard/active focus (not passive hover)
            if enabled && resp.has_focus() {
                ui.painter().rect_stroke(
                    b_rect,
                    CornerRadius::ZERO,
                    Stroke::new(2.0, theme.accent()),
                    StrokeKind::Inside,
                );
            }
            let icon_rect = Rect::from_center_size(b_rect.center(), Vec2::splat(icon_sz));
            let tint = if enabled {
                TOOLBAR_ICON_INACTIVE
            } else {
                disabled_icon_tint
            };
            paint_cm3_icon(ui.painter(), icon_rect, icon, tint);

            // Zoidium 8px save reminder dot at top-right of Save (radius 4px, center right-7.5, top+6.5)
            if dirty_dot {
                let dot_center = Pos2::new(b_rect.right() - 7.5, b_rect.top() + 6.5);
                let dot_color = if save_overdue {
                    theme.accent()
                } else {
                    TEXT_MUTED
                };
                ui.painter().circle_filled(dot_center, 4.0, dot_color);
            }
            enabled && resp.clicked()
        };

        if toolbar_btn(
            &mut cursor_x,
            Cm3Icon::New,
            "new (ctrl-m)",
            true,
            false,
            btn_gap,
        ) {
            actions.push(UiAction::NewProject);
        }
        if toolbar_btn(
            &mut cursor_x,
            Cm3Icon::Load,
            "open (ctrl-o)",
            true,
            false,
            btn_gap,
        ) {
            actions.push(UiAction::OpenProject);
        }
        if toolbar_btn(
            &mut cursor_x,
            Cm3Icon::Save,
            "save (ctrl-s)",
            true,
            true,
            sep_w,
        ) {
            let include_assets = !ui.input(|i| i.modifiers.shift);
            actions.push(UiAction::SaveProject { include_assets });
        }
        if toolbar_btn(
            &mut cursor_x,
            Cm3Icon::Undo,
            "undo (ctrl-z)",
            session.history.can_undo(),
            false,
            btn_gap,
        ) {
            let _ = session.execute(Command::Undo);
        }
        if toolbar_btn(
            &mut cursor_x,
            Cm3Icon::Redo,
            "redo (ctrl-y)",
            session.history.can_redo(),
            false,
            0.0,
        ) {
            let _ = session.execute(Command::Redo);
        }

        let border = if focused {
            theme.accent()
        } else {
            BORDER_PANEL
        };
        paint_panel_border(ui.painter(), rect, border);
    }

    // -----------------------------------------------------------------------
    // Left Lower Pane: Animated 60px <-> 300px Elevator Rail + Side Inspector
    // -----------------------------------------------------------------------

    fn show_left_lower_pane(
        &mut self,
        ui: &mut Ui,
        rect: Rect,
        viewport_rect: Rect,
        session: &mut Session,
        theme: &mut Cm3Theme,
        actions: &mut Vec<UiAction>,
    ) {
        let rect = snap_rect(rect);
        let viewport_rect = snap_rect(viewport_rect);
        let assets_enabled = session.plugins.is_enabled("asset-manager");
        if !assets_enabled && self.active_tab == SideTab::Assets {
            self.active_tab = SideTab::Media;
        }

        // Animate rail width: 200ms expand, 50ms collapse (Spec 01 §1.4 & §7)
        let max_rail_w = 300.0_f32.min(viewport_rect.width().max(60.0)).round();
        let target_w = if self.rail_expanded { max_rail_w } else { 60.0 };
        let dt = ui.input(|i| i.stable_dt).clamp(0.0, 0.1);
        let duration = if self.rail_expanded { 0.20 } else { 0.05 };
        let speed = 240.0 / duration;
        if (self.rail_width_px - target_w).abs() > 0.5 {
            if self.rail_width_px < target_w {
                self.rail_width_px = (self.rail_width_px + speed * dt).min(target_w);
            } else {
                self.rail_width_px = (self.rail_width_px - speed * dt).max(target_w);
            }
            ui.ctx().request_repaint();
        } else {
            self.rail_width_px = target_w;
        }

        // Side Inspector always sits flush at left: 60px so expanded drawer overlays without reflowing!
        let rail_base_w = 60.0_f32.min(rect.width()).round();
        let inspector_rect = snap_rect(Rect::from_min_max(
            Pos2::new((rect.left() + rail_base_w).min(rect.right()), rect.top()),
            rect.right_bottom(),
        ));

        let is_split_side_tab = match self.active_tab {
            SideTab::Objects => session
                .selection
                .clip_ids
                .first()
                .and_then(|id| {
                    session
                        .document()
                        .video_tracks
                        .iter()
                        .chain(session.document().audio_tracks.iter())
                        .flat_map(|t| t.clips.iter())
                        .find(|c| &c.id == id)
                })
                .is_some_and(|c| c.kind == "scene"),
            SideTab::Effects => session.selection.clip_ids.first().is_some_and(|id| {
                session
                    .document()
                    .video_tracks
                    .iter()
                    .chain(session.document().audio_tracks.iter())
                    .flat_map(|t| t.clips.iter())
                    .any(|c| &c.id == id)
            }),
            _ => false,
        };

        if Self::clicked_in_pane(ui, inspector_rect) {
            self.focused_panel = PanelFocus::SideInspector;
            self.panels.floating_windows.clear_focus();
            if is_split_side_tab {
                let split_state = match self.active_tab {
                    SideTab::Objects => self.panels.objects_split,
                    _ => self.panels.effects_split,
                };
                let (_top_r, _sp_r, bot_r) = split_state.compute_rects(inspector_rect);
                if let Some(pos) = ui.ctx().input(|i| i.pointer.interact_pos()) {
                    self.split_inspector_focus_bottom = bot_r.contains(pos);
                }
            }
        }

        ui.painter()
            .rect_filled(inspector_rect, CornerRadius::ZERO, BG_PANEL);

        // Objects & Effects split-pane tabs own their own sibling 1px panel borders & edge-to-edge 3px splitter.
        let inspector_inner = if is_split_side_tab {
            inspector_rect
        } else {
            snap_rect(inspector_rect.shrink(1.0))
        };
        let mut panel_actions = Vec::new();
        ui.scope_builder(
            UiBuilder::new()
                .max_rect(inspector_inner)
                .id_salt("side_inspector_content"),
            |ui| {
                ui.set_clip_rect(ui.clip_rect().intersect(inspector_inner));
                show_side_panel(
                    ui,
                    self.active_tab,
                    inspector_inner,
                    viewport_rect,
                    session,
                    theme,
                    &mut self.panels,
                    &mut panel_actions,
                );
            },
        );
        for act in panel_actions {
            actions.push(match act {
                PanelAction::NewProject => UiAction::NewProject,
                PanelAction::ImportMedia => UiAction::ImportMedia,
                #[cfg(target_arch = "wasm32")]
                PanelAction::RelinkWebAsset(id) => UiAction::RelinkWebAsset(id),
                #[cfg(target_arch = "wasm32")]
                PanelAction::ImportWebSvg(id) => UiAction::ImportWebSvg(id),
                PanelAction::StartExport(cfg) => UiAction::StartExport(cfg),
                PanelAction::AddToExportQueue(cfg) => UiAction::AddToExportQueue(cfg),
                PanelAction::RunExportQueue => UiAction::RunExportQueue,
                PanelAction::CancelExport => UiAction::CancelExport,
                #[cfg(not(target_arch = "wasm32"))]
                PanelAction::OpenOutputFolder(path) => UiAction::OpenOutputFolder(path),
                #[cfg(not(target_arch = "wasm32"))]
                PanelAction::OpenFfmpegSetup => UiAction::OpenFfmpegSetup,
                #[cfg(not(target_arch = "wasm32"))]
                PanelAction::BrowseFfmpegBinary => UiAction::BrowseFfmpegBinary,
                #[cfg(not(target_arch = "wasm32"))]
                PanelAction::ClearFfmpegBinary => UiAction::ClearFfmpegBinary,
                #[cfg(not(target_arch = "wasm32"))]
                PanelAction::ReprobeFfmpeg => UiAction::ReprobeFfmpeg,
                #[cfg(not(target_arch = "wasm32"))]
                PanelAction::SaveRecoverySnapshotNow => UiAction::SaveRecoverySnapshotNow,
                #[cfg(not(target_arch = "wasm32"))]
                PanelAction::RestoreRecoverySnapshot(path) => {
                    UiAction::RestoreRecoverySnapshot(path)
                }
                #[cfg(not(target_arch = "wasm32"))]
                PanelAction::DeleteRecoverySnapshot(path) => UiAction::DeleteRecoverySnapshot(path),
                PanelAction::EyedropperActivate {
                    target_id,
                    property,
                } => UiAction::EyedropperActivate {
                    target_id,
                    property,
                },
                PanelAction::OpenGraphEditor => UiAction::OpenGraphEditor,
                PanelAction::PickAssetForProperty {
                    target_id,
                    property,
                    is_lut,
                } => UiAction::PickAssetForProperty {
                    target_id,
                    property,
                    is_lut,
                },
                PanelAction::OpenGraphForProperty {
                    target_id,
                    property,
                } => UiAction::OpenGraphForProperty {
                    target_id,
                    property,
                },
                PanelAction::StatusMessage(msg) => UiAction::StatusMessage(msg),
            });
        }

        if is_split_side_tab {
            if self.focused_panel == PanelFocus::SideInspector {
                let split_state = match self.active_tab {
                    SideTab::Objects => self.panels.objects_split,
                    _ => self.panels.effects_split,
                };
                let (top_r, _sp_r, bot_r) = split_state.compute_rects(inspector_rect);
                let focused_sub_rect = if self.split_inspector_focus_bottom {
                    bot_r
                } else {
                    top_r
                };
                paint_panel_border(ui.painter(), focused_sub_rect, theme.accent());
            }
        } else {
            let insp_border = if self.focused_panel == PanelFocus::SideInspector {
                theme.accent()
            } else {
                BORDER_PANEL
            };
            paint_panel_border(ui.painter(), inspector_rect, insp_border);
        }

        // Render Elevator Rail: overlay Area when expanded/animating, inline 60px strip when collapsed.
        if self.rail_width_px > 60.5 {
            let base_strip = snap_rect(Rect::from_min_size(
                rect.left_top(),
                Vec2::new(rail_base_w, rect.height()),
            ));
            ui.painter()
                .rect_filled(base_strip, CornerRadius::ZERO, VOID_BLACK);

            let drawer_rect = snap_rect(Rect::from_min_size(
                rect.left_top(),
                Vec2::new(self.rail_width_px.min(max_rail_w).round(), rect.height()),
            ));

            // Collapse expanded drawer on clicks outside rail or Escape.
            if self.rail_expanded {
                let wants_kb = ui.ctx().egui_wants_keyboard_input();
                let should_collapse = ui.ctx().input(|i| {
                    (i.pointer.primary_pressed()
                        && i.pointer
                            .interact_pos()
                            .is_some_and(|p| !drawer_rect.contains(p)))
                        || (!wants_kb && i.key_pressed(Key::Escape))
                });
                if should_collapse {
                    self.rail_expanded = false;
                }
            }

            Area::new(Id::unique("cm3_elevator_drawer_overlay"))
                .order(Order::Foreground)
                .fixed_pos(drawer_rect.left_top())
                .interactable(true)
                .show(ui.ctx(), |ui| {
                    // Drawer overlay shadow: 5px 0 10px rgba(0,0,0,0.5)
                    for (dx, alpha) in [(2.0, 95), (6.0, 55), (10.0, 22)] {
                        let sh = Rect::from_min_size(
                            Pos2::new(drawer_rect.right(), drawer_rect.top()),
                            Vec2::new(dx, drawer_rect.height()),
                        );
                        ui.painter().rect_filled(
                            sh,
                            CornerRadius::ZERO,
                            Color32::from_black_alpha(alpha),
                        );
                    }
                    ui.scope_builder(
                        UiBuilder::new()
                            .max_rect(drawer_rect)
                            .id_salt("cm3_elevator_drawer_content"),
                        |ui| {
                            ui.set_clip_rect(drawer_rect);
                            self.show_elevator_rail(ui, drawer_rect, assets_enabled, theme);
                        },
                    );
                });
        } else {
            let rail_rect = snap_rect(Rect::from_min_size(
                rect.left_top(),
                Vec2::new(rail_base_w, rect.height()),
            ));
            self.show_elevator_rail(ui, rail_rect, assets_enabled, theme);
        }
    }

    /// Computes the ordered sidebar tabs visible for the current plugin state (hiding `Assets`
    /// when `asset-manager` is disabled, e.g. in Core Mode).
    #[must_use]
    pub fn visible_side_tabs(assets_enabled: bool) -> Vec<SideTab> {
        SideTab::ALL
            .iter()
            .copied()
            .filter(|tab| assets_enabled || *tab != SideTab::Assets)
            .collect()
    }

    /// Computes per-slot vertical height so the menu toggle and all visible sidebar tabs fit
    /// cleanly even at compact heights like `900×600`, while preserving exact `50px` slots at
    /// standard heights (`>= 720px`).
    #[must_use]
    pub fn compute_elevator_slot_height(rail_height: f32, visible_tab_count: usize) -> f32 {
        let total_slots = (visible_tab_count + 1).max(1) as f32;
        (rail_height / total_slots).floor().clamp(34.0, 50.0)
    }

    /// Computes a fixed-width Player+ Speed/Pause pill that accommodates `"Paused"` + chevron
    /// with zero clipping even in compact `900×600` layout and zero horizontal jump on play/pause.
    #[must_use]
    pub fn compute_speed_pill_width(compact: bool, paused_label_width: f32) -> f32 {
        let base: f32 = if compact { 80.0 } else { 88.0 };
        // 6px left inset + full `"Paused"` label + 20px right breathing room & chevron.
        let needed = paused_label_width.ceil() + 26.0;
        base.max(needed).min(112.0).round()
    }

    /// Computes responsive sizing for the 3-group Playback & Viewport toolbar across wide,
    /// compact (`900×600`), narrow (`800×540` ~525px), and ultra-narrow (`>= 418px`) widths
    /// so `Loop`, `Graph`, `Paused`, and `Full` are never clipped off the right edge.
    #[must_use]
    pub fn compute_playback_toolbar_metrics(
        inner_width: f32,
        paused_w_small: f32,
        paused_w_compact: f32,
    ) -> PlaybackToolbarMetrics {
        if inner_width >= 690.0 {
            PlaybackToolbarMetrics {
                btn_size: 29.0,
                btn_gap: 2.0,
                pill_gap: 2.0,
                icon_size: 25.0,
                edge_pad: 5.0,
                right_pad: 8.0,
                group_gap: 6.0,
                pill_font: FONT_SMALL,
                speed_dd_w: Self::compute_speed_pill_width(false, paused_w_small),
                q_w: 54.0,
                compact_quality_labels: false,
                pill_left_inset: 6.0,
                pill_chevron_reserve: 17.0,
                chevron_size: 11.0,
                chevron_right_inset: 8.5,
            }
        } else if inner_width >= 565.0 {
            PlaybackToolbarMetrics {
                btn_size: 25.0,
                btn_gap: 1.0,
                pill_gap: 1.0,
                icon_size: 21.0,
                edge_pad: 3.0,
                right_pad: 4.0,
                group_gap: 4.0,
                pill_font: FONT_SMALL,
                speed_dd_w: Self::compute_speed_pill_width(true, paused_w_small),
                q_w: 50.0,
                compact_quality_labels: false,
                pill_left_inset: 6.0,
                pill_chevron_reserve: 17.0,
                chevron_size: 11.0,
                chevron_right_inset: 8.5,
            }
        } else if inner_width >= 490.0 {
            PlaybackToolbarMetrics {
                btn_size: 22.0,
                btn_gap: 1.0,
                pill_gap: 1.0,
                icon_size: 18.0,
                edge_pad: 2.0,
                right_pad: 3.0,
                group_gap: 3.0,
                pill_font: FONT_COMPACT,
                speed_dd_w: (paused_w_compact.ceil() + 20.0).clamp(62.0, 74.0).round(),
                q_w: 38.0,
                compact_quality_labels: true,
                pill_left_inset: 4.0,
                pill_chevron_reserve: 14.0,
                chevron_size: 9.0,
                chevron_right_inset: 7.0,
            }
        } else {
            PlaybackToolbarMetrics {
                btn_size: 19.0,
                btn_gap: 0.0,
                pill_gap: 1.0,
                icon_size: 16.0,
                edge_pad: 2.0,
                right_pad: 2.0,
                group_gap: 2.0,
                pill_font: FONT_COMPACT,
                speed_dd_w: (paused_w_compact.ceil() + 18.0).clamp(56.0, 66.0).round(),
                q_w: 32.0,
                compact_quality_labels: true,
                pill_left_inset: 4.0,
                pill_chevron_reserve: 13.0,
                chevron_size: 9.0,
                chevron_right_inset: 6.5,
            }
        }
    }

    fn show_elevator_rail(
        &mut self,
        ui: &mut Ui,
        rail_rect: Rect,
        assets_enabled: bool,
        theme: &Cm3Theme,
    ) {
        let rail_rect = snap_rect(rail_rect);
        ui.painter()
            .rect_filled(rail_rect, CornerRadius::ZERO, VOID_BLACK);
        let rail_focused = self.focused_panel == PanelFocus::ElevatorRail;
        let visible_tabs = Self::visible_side_tabs(assets_enabled);
        let tab_count = visible_tabs.len();
        if tab_count > 0 && self.rail_focus_index >= tab_count {
            self.rail_focus_index = tab_count - 1;
        }
        let slot_h = Self::compute_elevator_slot_height(rail_rect.height(), tab_count);
        let icon_sz = (slot_h - 16.0).clamp(22.0, 30.0).round();
        let tab_font_sz = if slot_h < 42.0 {
            (FONT_TAB * (slot_h / 50.0)).clamp(15.0, FONT_TAB)
        } else {
            FONT_TAB
        };

        // Keyboard navigation when rail has focus (Spec 01 §1.4)
        let mut rail_key_navigated = false;
        if rail_focused
            && tab_count > 0
            && !self.suppress_shortcuts
            && !ui.ctx().egui_wants_keyboard_input()
        {
            ui.input(|i| {
                if i.key_pressed(Key::ArrowDown) {
                    self.rail_focus_index = (self.rail_focus_index + 1) % tab_count;
                    rail_key_navigated = true;
                } else if i.key_pressed(Key::ArrowUp) {
                    self.rail_focus_index = (self.rail_focus_index + tab_count - 1) % tab_count;
                    rail_key_navigated = true;
                } else if i.key_pressed(Key::Enter) {
                    self.active_tab = visible_tabs[self.rail_focus_index];
                    self.rail_expanded = false;
                }
            });
        }

        // Top Menu Toggle button (scales smoothly with slot_h so all 12 items fit at 900×600)
        let menu_rect = snap_rect(Rect::from_min_size(
            rail_rect.left_top(),
            Vec2::new(rail_rect.width(), slot_h.min(rail_rect.height())),
        ));
        let menu_resp = ui
            .interact(menu_rect, Id::unique("elevator_menu_btn"), Sense::click())
            .on_hover_cursor(CursorIcon::PointingHand)
            .on_hover_text(if self.rail_expanded {
                "collapse sidebar"
            } else {
                "expand sidebar"
            });
        if menu_resp.hovered() {
            ui.painter()
                .rect_filled(menu_rect, CornerRadius::ZERO, BG_TITLE);
        }
        // Accent border on menu button ONLY on focus (not on hover)
        if menu_resp.has_focus() {
            ui.painter().rect_stroke(
                menu_rect,
                CornerRadius::ZERO,
                Stroke::new(2.0, theme.accent()),
                StrokeKind::Inside,
            );
        }
        let menu_icon_rect = Rect::from_center_size(
            Pos2::new(
                (menu_rect.left() + 30.0).round(),
                menu_rect.center().y.round(),
            ),
            Vec2::splat(icon_sz),
        );
        paint_cm3_icon(ui.painter(), menu_icon_rect, Cm3Icon::Menu, TEXT_STRONG);
        if menu_resp.clicked() {
            self.rail_expanded = !self.rail_expanded;
            self.focused_panel = PanelFocus::ElevatorRail;
            self.panels.floating_windows.clear_focus();
        }

        // Scrollable Tab Strip (scales cleanly to fit all tabs at 900×600, scrolls if even shorter)
        let strip_rect = snap_rect(Rect::from_min_max(
            Pos2::new(rail_rect.left(), menu_rect.bottom()),
            rail_rect.right_bottom(),
        ));
        let now = ui.input(|i| i.time);
        // Only trigger drag-hover auto-switch on real drag payloads, never raw primary_down().
        let has_drag_payload = egui::DragAndDrop::has_any_payload(ui.ctx());
        let mut hovered_tab_this_frame: Option<SideTab> = None;
        let accent = theme.accent();
        let contrast = theme.contrast_fg(accent);
        let selected_fg = if contrast == TEXT_BRIGHT {
            TEXT_STRONG
        } else {
            contrast
        };

        ui.scope_builder(
            UiBuilder::new()
                .max_rect(strip_rect)
                .id_salt("elevator_tab_strip"),
            |ui| {
                ui.set_clip_rect(ui.clip_rect().intersect(strip_rect));
                ScrollArea::vertical()
                    .id_salt("elevator_tabs_scroll")
                    .scroll_bar_visibility(
                        egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded,
                    )
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing = Vec2::ZERO;
                        for (idx, tab) in visible_tabs.iter().copied().enumerate() {
                            let (row_rect, row_resp) = ui.allocate_exact_size(
                                Vec2::new(strip_rect.width(), slot_h),
                                Sense::click(),
                            );
                            let draw_row = snap_rect(row_rect);
                            let row_resp = row_resp
                                .on_hover_cursor(CursorIcon::PointingHand)
                                .on_hover_text(tab.title());

                            if rail_key_navigated && rail_focused && self.rail_focus_index == idx {
                                row_resp.scroll_to_me(None);
                            }

                            let selected = self.active_tab == tab;
                            if selected {
                                ui.painter()
                                    .rect_filled(draw_row, CornerRadius::ZERO, accent);
                            } else if row_resp.hovered() {
                                ui.painter()
                                    .rect_filled(draw_row, CornerRadius::ZERO, BG_TITLE);
                            }

                            if rail_focused && self.rail_focus_index == idx {
                                ui.painter().rect_stroke(
                                    draw_row,
                                    CornerRadius::ZERO,
                                    Stroke::new(2.0, BORDER_RAIL_FOCUS),
                                    StrokeKind::Inside,
                                );
                            }

                            let fg = if selected { selected_fg } else { TEXT_STRONG };
                            let icon_rect = Rect::from_center_size(
                                Pos2::new(
                                    (draw_row.left() + 30.0).round(),
                                    draw_row.center().y.round(),
                                ),
                                Vec2::splat(icon_sz),
                            );
                            paint_cm3_icon(ui.painter(), icon_rect, tab.icon(), fg);

                            // Tab label when expanded at left + 62px
                            if strip_rect.width() > 64.0 {
                                let label_x = (draw_row.left() + 62.0).round();
                                ui.painter().with_clip_rect(draw_row).text(
                                    Pos2::new(label_x, draw_row.center().y),
                                    Align2::LEFT_CENTER,
                                    tab.title(),
                                    theme.font(tab_font_sz),
                                    fg,
                                );
                            }

                            if row_resp.clicked() {
                                self.active_tab = tab;
                                self.rail_focus_index = idx;
                                self.rail_expanded = false;
                                // Split-pane tabs (Objects, Effects) remain unfocused until a sub-pane is clicked.
                                self.focused_panel =
                                    if matches!(tab, SideTab::Objects | SideTab::Effects) {
                                        PanelFocus::None
                                    } else {
                                        PanelFocus::SideInspector
                                    };
                                self.split_inspector_focus_bottom = false;
                                self.panels.floating_windows.clear_focus();
                            }

                            if row_resp.hovered() && has_drag_payload && !selected {
                                hovered_tab_this_frame = Some(tab);
                            }
                        }
                    });
            },
        );

        // 200ms drag-hover auto-switch timer (Spec 01 §1.4)
        match (hovered_tab_this_frame, self.rail_drag_dwell) {
            (Some(tab), Some((prev_tab, start_t))) if tab == prev_tab => {
                if now - start_t >= 0.20 {
                    self.active_tab = tab;
                    self.rail_drag_dwell = None;
                } else {
                    ui.ctx().request_repaint();
                }
            }
            (Some(tab), _) => {
                self.rail_drag_dwell = Some((tab, now));
                ui.ctx().request_repaint();
            }
            (None, _) => {
                self.rail_drag_dwell = None;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Right Top: Viewport Frame (#1d1d1d background, 1px focus border)
    // -----------------------------------------------------------------------

    fn show_viewport_frame(
        &mut self,
        ui: &mut Ui,
        rect: Rect,
        session: &mut Session,
        theme: &Cm3Theme,
        draw_viewport: &mut impl FnMut(&mut Ui, Rect, bool, &mut Session),
    ) {
        let rect = snap_rect(rect);
        let focused = self.focused_panel == PanelFocus::Viewport;
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, BG_SUNKEN);

        let inner = snap_rect(rect.shrink(1.0));
        if Self::clicked_in_pane(ui, rect) {
            self.focused_panel = PanelFocus::Viewport;
            self.panels.floating_windows.clear_focus();
        }

        ui.scope_builder(
            UiBuilder::new()
                .max_rect(inner)
                .id_salt("cm3_viewport_pane"),
            |ui| {
                ui.set_clip_rect(ui.clip_rect().intersect(inner));
                draw_viewport(ui, inner, focused, session);
            },
        );

        let border = if focused {
            theme.accent()
        } else {
            BORDER_PANEL
        };
        paint_panel_border(ui.painter(), rect, border);
    }

    // -----------------------------------------------------------------------
    // Right Middle: 40px Playback & Viewport Toolbar (Zoidium 3-Group Layout)
    // -----------------------------------------------------------------------

    fn show_playback_toolbar(
        &mut self,
        ui: &mut Ui,
        rect: Rect,
        session: &mut Session,
        theme: &Cm3Theme,
        actions: &mut Vec<UiAction>,
    ) {
        let rect = snap_rect(rect);
        let focused = self.focused_panel == PanelFocus::PlaybackToolbar;
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, BG_TOOLBAR);

        if Self::clicked_in_pane(ui, rect) {
            self.focused_panel = PanelFocus::PlaybackToolbar;
            self.panels.floating_windows.clear_focus();
        }

        let inner = snap_rect(rect.shrink(1.0));
        let player_plus = session.plugins.is_enabled("player-plus");
        let motion_path = session.plugins.is_enabled("motion-path");
        let scene_selected = selected_clip_is_scene3d(session);
        let editing_camera_active = session
            .selection
            .is_editing_camera_active(&session.playback, scene_selected);

        // Responsive pitch so all 3 groups (including Loop, Graph, Speed/Pause, Quality)
        // fit cleanly with zero clipping across wide, 900×600, 800×540, and 420px min-width layouts.
        let paused_cache_key = (
            theme.font_scale.to_bits(),
            theme.editor_font,
            ui.pixels_per_point().to_bits(),
        );
        let (paused_label_w_small, paused_label_w_compact) = if let Some((key, small_w, compact_w)) =
            self.cached_paused_label_widths
            && key == paused_cache_key
        {
            (small_w, compact_w)
        } else {
            let small_w = ui
                .painter()
                .layout_no_wrap("Paused".to_owned(), theme.font(FONT_SMALL), theme.value())
                .size()
                .x;
            let compact_w = ui
                .painter()
                .layout_no_wrap("Paused".to_owned(), theme.font(FONT_COMPACT), theme.value())
                .size()
                .x;
            self.cached_paused_label_widths = Some((paused_cache_key, small_w, compact_w));
            (small_w, compact_w)
        };
        let m = Self::compute_playback_toolbar_metrics(
            inner.width(),
            paused_label_w_small,
            paused_label_w_compact,
        );
        let btn_size = m.btn_size;
        let btn_gap = m.btn_gap;
        let pill_gap = m.pill_gap;
        let btn_step = btn_size + btn_gap;
        let icon_size = m.icon_size;
        let edge_pad = m.edge_pad;
        let right_pad = m.right_pad;
        let group_gap = m.group_gap;
        let speed_dd_w = m.speed_dd_w;
        let q_w = m.q_w;
        let by = (inner.top() + (inner.height() - btn_size) * 0.5).floor();
        let disabled_tint = Color32::from_rgb(0x52, 0x52, 0x52);

        #[derive(Clone, Copy)]
        enum PbIcon {
            Std(Cm3Icon),
            ShapePen,
            BezierPen,
            DrawPencil,
            MotionPath,
        }

        let pb_icon_btn_enabled = |cx: &mut f32,
                                   icon: PbIcon,
                                   id_key: &str,
                                   tip: &str,
                                   tint: Color32,
                                   accent_border: bool,
                                   enabled: bool|
         -> bool {
            let b_rect = snap_rect(Rect::from_min_size(
                Pos2::new(cx.round(), by),
                Vec2::splat(btn_size),
            ));
            *cx += btn_step;
            let sense = if enabled {
                Sense::click()
            } else {
                Sense::hover()
            };
            let mut resp = ui
                .interact(b_rect, Id::unique(("pbtb", id_key)), sense)
                .on_hover_text(tip);
            if enabled {
                resp = resp.on_hover_cursor(CursorIcon::PointingHand);
            }
            if enabled && resp.hovered() && !resp.has_focus() && !accent_border {
                ui.painter().rect_filled(
                    b_rect,
                    CornerRadius::ZERO,
                    Color32::from_rgba_unmultiplied(0xff, 0xff, 0xff, 10),
                );
            }
            // 2px theme.accent() border ONLY on focus or active open state (never passive hover)
            if enabled && (resp.has_focus() || accent_border) {
                ui.painter().rect_stroke(
                    b_rect,
                    CornerRadius::ZERO,
                    Stroke::new(2.0, theme.accent()),
                    StrokeKind::Inside,
                );
            }
            let ic = Rect::from_center_size(b_rect.center(), Vec2::splat(icon_size));
            match icon {
                PbIcon::Std(icon) => paint_cm3_icon(ui.painter(), ic, icon, tint),
                PbIcon::ShapePen => paint_shape_pen_icon(ui.painter(), ic, tint),
                PbIcon::BezierPen => paint_bezier_pen_icon(ui.painter(), ic, tint),
                PbIcon::DrawPencil => paint_draw_pencil_icon(ui.painter(), ic, tint),
                PbIcon::MotionPath => paint_motion_path_icon(ui.painter(), ic, tint),
            }
            enabled && resp.clicked()
        };

        let pb_icon_btn =
            |cx: &mut f32, icon: PbIcon, tip: &str, tint: Color32, accent_border: bool| -> bool {
                pb_icon_btn_enabled(cx, icon, tip, tip, tint, accent_border, true)
            };

        // --- LEFT group ---
        let mut left_x = (inner.left() + edge_pad).round();
        if player_plus {
            if pb_icon_btn(
                &mut left_x,
                PbIcon::Std(Cm3Icon::Pause),
                "Player+: pause viewport drawing",
                if self.player_plus.drawing_paused {
                    TOGGLE_ACTIVE
                } else {
                    TEXT_MUTED
                },
                false,
            ) {
                self.player_plus.drawing_paused = !self.player_plus.drawing_paused;
            }
            if pb_icon_btn(
                &mut left_x,
                PbIcon::Std(Cm3Icon::Reset),
                "Player+: render single frame",
                TEXT_MUTED,
                false,
            ) {
                self.player_plus.single_render_requested = true;
                actions.push(UiAction::SingleFrameRender);
            }
            let checker_enabled = !editing_camera_active;
            if pb_icon_btn_enabled(
                &mut left_x,
                PbIcon::Std(Cm3Icon::TransparencyGrid),
                "transparency_grid",
                if checker_enabled {
                    "Player+: toggle transparency grid"
                } else {
                    "Checkerboard applies in Preview mode (exit Editing Camera)"
                },
                if !checker_enabled {
                    disabled_tint
                } else if self.transparency_grid {
                    theme.value()
                } else {
                    TEXT_MUTED
                },
                checker_enabled && self.transparency_grid,
                checker_enabled,
            ) {
                self.transparency_grid = !self.transparency_grid;
            }
        }
        if pb_icon_btn(
            &mut left_x,
            PbIcon::Std(Cm3Icon::Camera),
            "editing camera (c)",
            if editing_camera_active {
                TOGGLE_ACTIVE
            } else {
                TOOLBAR_ICON_INACTIVE
            },
            false,
        ) {
            let _ = toggle_editing_camera_synced(session);
        }
        if pb_icon_btn(
            &mut left_x,
            PbIcon::Std(Cm3Icon::Transform),
            "2D layer transform controls (t)",
            if session.playback.transform_controls_2d {
                TOGGLE_ACTIVE
            } else {
                TOOLBAR_ICON_INACTIVE
            },
            false,
        ) {
            session.playback.toggle_transform_controls_2d();
        }
        if pb_icon_btn(
            &mut left_x,
            PbIcon::ShapePen,
            "shape pen tool",
            if self.viewport_draw_tool == Some(0) {
                TOGGLE_ACTIVE
            } else {
                TOOLBAR_ICON_INACTIVE
            },
            false,
        ) {
            self.viewport_draw_tool = if self.viewport_draw_tool == Some(0) {
                None
            } else {
                Some(0)
            };
        }
        if pb_icon_btn(
            &mut left_x,
            PbIcon::BezierPen,
            "bezier path edit tool",
            if self.viewport_draw_tool == Some(1) {
                TOGGLE_ACTIVE
            } else {
                TOOLBAR_ICON_INACTIVE
            },
            false,
        ) {
            self.viewport_draw_tool = if self.viewport_draw_tool == Some(1) {
                None
            } else {
                Some(1)
            };
        }
        if pb_icon_btn(
            &mut left_x,
            PbIcon::DrawPencil,
            "freehand draw tool",
            if self.viewport_draw_tool == Some(2) {
                TOGGLE_ACTIVE
            } else {
                TOOLBAR_ICON_INACTIVE
            },
            false,
        ) {
            self.viewport_draw_tool = if self.viewport_draw_tool == Some(2) {
                None
            } else {
                Some(2)
            };
        }
        if motion_path
            && pb_icon_btn(
                &mut left_x,
                PbIcon::MotionPath,
                "motion paths overlay",
                if self.motion_path_visible {
                    TOGGLE_ACTIVE
                } else {
                    TOOLBAR_ICON_INACTIVE
                },
                false,
            )
        {
            self.motion_path_visible = !self.motion_path_visible;
        }
        let left_end = (left_x - btn_gap).max(inner.left());

        // Precompute RIGHT group fixed width so CENTER clamps without ever pushing RIGHT off-screen,
        // and so Loop/Graph NEVER jump horizontally when toggling play/pause or cycling quality.
        let right_edge = (inner.right() - right_pad).round();
        let base_right_w = btn_size + btn_gap + btn_size; // Loop + Graph
        let pp_extra_w = pill_gap + speed_dd_w + pill_gap + q_w;
        let transport_w = 5.0 * btn_size + 4.0 * btn_gap;
        let max_right_avail =
            (right_edge - (left_end + group_gap + transport_w + group_gap)).max(0.0);
        let show_pp_right = player_plus && max_right_avail >= base_right_w + pp_extra_w;
        let right_w = if show_pp_right {
            base_right_w + pp_extra_w
        } else {
            base_right_w
        };

        // --- CENTER group (horizontally centered on inner.center().x, clamped between LEFT & RIGHT) ---
        let min_center_x = (left_end + group_gap).round();
        let max_center_x = (right_edge - right_w - group_gap - transport_w)
            .round()
            .max(min_center_x);
        let mut center_x = (inner.center().x - transport_w * 0.5)
            .round()
            .clamp(min_center_x, max_center_x);
        if pb_icon_btn(
            &mut center_x,
            PbIcon::Std(Cm3Icon::Start),
            "jump to start (Home)",
            TOOLBAR_ICON_INACTIVE,
            false,
        ) {
            let _ = session.execute(Command::SetFrame { frame: 0.0 });
        }
        if pb_icon_btn(
            &mut center_x,
            PbIcon::Std(Cm3Icon::PrevFrame),
            "previous frame (Left)",
            TOOLBAR_ICON_INACTIVE,
            false,
        ) {
            let _ = session.execute(Command::StepFrame { delta: -1.0 });
        }
        let playing = session.playback.is_playing();
        if pb_icon_btn(
            &mut center_x,
            PbIcon::Std(if playing {
                Cm3Icon::Pause
            } else {
                Cm3Icon::Play
            }),
            if playing {
                "pause (Space)"
            } else {
                "play (Space)"
            },
            if playing {
                TOGGLE_ACTIVE
            } else {
                TOOLBAR_ICON_INACTIVE
            },
            false,
        ) {
            let _ = session.execute(Command::TogglePlay);
        }
        if pb_icon_btn(
            &mut center_x,
            PbIcon::Std(Cm3Icon::NextFrame),
            "next frame (Right)",
            TOOLBAR_ICON_INACTIVE,
            false,
        ) {
            let _ = session.execute(Command::StepFrame { delta: 1.0 });
        }
        if pb_icon_btn(
            &mut center_x,
            PbIcon::Std(Cm3Icon::End),
            "jump to end (End)",
            TOOLBAR_ICON_INACTIVE,
            false,
        ) {
            let end_f = (session.document().sequence_length - 1.0).max(0.0);
            let _ = session.execute(Command::SetFrame { frame: end_f });
        }
        let transport_end = center_x - btn_gap;

        // --- RIGHT group (fixed-width Speed/Pause pill + fixed-width Quality -> zero layout jump) ---
        let mut rx = (right_edge - right_w)
            .round()
            .max(transport_end + group_gap);

        if rx + btn_size <= right_edge + 0.5
            && pb_icon_btn(
                &mut rx,
                PbIcon::Std(Cm3Icon::Loop),
                "loop playback (Ctrl+L)",
                if session.playback.looping {
                    TOGGLE_ACTIVE
                } else {
                    TOOLBAR_ICON_INACTIVE
                },
                false,
            )
        {
            let _ = session.execute(Command::ToggleLoop);
        }
        if rx + btn_size <= right_edge + 0.5
            && pb_icon_btn(
                &mut rx,
                PbIcon::Std(Cm3Icon::Graph),
                "graph editor (Ctrl+G)",
                TEXT_PRIMARY,
                self.graph_open,
            )
        {
            self.graph_open = !self.graph_open;
            actions.push(UiAction::OpenGraphEditor);
        }

        if show_pp_right {
            let ctrl_h = (btn_size - 2.0).clamp(19.0, 27.0);
            let ctrl_y = (inner.top() + (inner.height() - ctrl_h) * 0.5).floor();
            let q_label = if m.compact_quality_labels {
                self.player_plus.preview_quality.compact_label()
            } else {
                self.player_plus.preview_quality.label()
            };

            // Align gap between Graph button and Speed/Pause pill when btn_gap < pill_gap.
            rx += (pill_gap - btn_gap).max(0.0);

            // Always-visible fixed-width Speed / Render-Pause pill (both playing and paused)
            let speed_rect = snap_rect(Rect::from_min_size(
                Pos2::new(rx.round(), ctrl_y),
                Vec2::new(speed_dd_w, ctrl_h),
            ));
            rx += speed_dd_w + pill_gap;
            let pill_label = if self.player_plus.drawing_paused {
                "Held".to_owned()
            } else if !playing {
                "Paused".to_owned()
            } else {
                let s = session.playback.speed;
                if (s - s.round()).abs() < 1e-3 || ((s * 10.0) - (s * 10.0).round()).abs() < 1e-3 {
                    format!("{s:.1}x")
                } else {
                    format!("{s:.2}x")
                }
            };
            let p_resp = ui
                .interact(
                    speed_rect,
                    Id::unique("pp_speed_pause_dropdown"),
                    Sense::click(),
                )
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text("Player+: playback speed presets & render pause");
            let bg = if self.player_plus.drawing_paused {
                TOGGLE_ACTIVE
            } else {
                BG_INPUT
            };
            let fg = if self.player_plus.drawing_paused {
                TEXT_BRIGHT
            } else {
                theme.value()
            };
            ui.painter().rect_filled(speed_rect, CornerRadius::ZERO, bg);
            if p_resp.hovered() && !p_resp.has_focus() && !self.player_plus.drawing_paused {
                ui.painter().rect_filled(
                    speed_rect,
                    CornerRadius::ZERO,
                    Color32::from_rgba_unmultiplied(0xff, 0xff, 0xff, 10),
                );
            }
            paint_panel_border(
                ui.painter(),
                speed_rect,
                if self.speed_menu_open || p_resp.has_focus() {
                    theme.accent()
                } else {
                    BORDER_CONTROL
                },
            );
            let pill_text_clip = Rect::from_min_max(
                speed_rect.left_top(),
                Pos2::new(
                    (speed_rect.right() - m.pill_chevron_reserve).max(speed_rect.left()),
                    speed_rect.bottom(),
                ),
            );
            ui.painter().with_clip_rect(pill_text_clip).text(
                Pos2::new(speed_rect.left() + m.pill_left_inset, speed_rect.center().y),
                Align2::LEFT_CENTER,
                &pill_label,
                theme.font(m.pill_font),
                fg,
            );
            let chevron_rect = Rect::from_center_size(
                Pos2::new(
                    speed_rect.right() - m.chevron_right_inset,
                    speed_rect.center().y - 0.5,
                ),
                Vec2::splat(m.chevron_size),
            );
            paint_cm3_icon(ui.painter(), chevron_rect, Cm3Icon::ChevronDown, fg);
            if p_resp.clicked() {
                self.speed_menu_open = !self.speed_menu_open;
                if !self.speed_menu_open {
                    ui.memory_mut(|m| m.surrender_focus(p_resp.id));
                }
            }

            if self.speed_menu_open {
                let presets: [f64; 5] = [0.25, 0.5, 1.0, 1.5, 2.0];
                let row_h = 24.0_f32;
                let sep_h = 5.0_f32;
                let popup_w = 136.0_f32;
                let popup_h = (presets.len() + 1) as f32 * row_h + sep_h + 4.0;
                let screen_rect = ui.ctx().input(|i| i.viewport_rect());
                let mut popup_x = speed_rect.left().round();
                if popup_x + popup_w > screen_rect.right() - 4.0 {
                    popup_x = (screen_rect.right() - popup_w - 4.0).max(screen_rect.left() + 4.0);
                }
                let mut popup_y = speed_rect.top().round() - popup_h - 2.0;
                if popup_y < screen_rect.top() + 4.0 {
                    popup_y = (speed_rect.bottom().round() + 2.0)
                        .min((screen_rect.bottom() - popup_h - 4.0).max(screen_rect.top() + 4.0));
                }
                let mut close_menu = ui.input(|i| i.key_pressed(Key::Escape));
                let area_resp = Area::new(Id::unique("pp_speed_menu_popup"))
                    .order(Order::Foreground)
                    .fixed_pos(Pos2::new(popup_x, popup_y))
                    .interactable(true)
                    .show(ui.ctx(), |ui| {
                        let inner_w = popup_w - 4.0;
                        let frame_resp = Frame::NONE
                            .fill(BG_INPUT)
                            .inner_margin(egui::Margin::same(2))
                            .show(ui, |ui| {
                                ui.set_min_width(inner_w);
                                ui.set_max_width(inner_w);
                                ui.spacing_mut().item_spacing = Vec2::ZERO;
                                let menu_active_bg = theme.menu_active();
                                let menu_active_fg = theme.contrast_fg(menu_active_bg);
                                for &spd in &presets {
                                    let (r_rect, r_resp) = ui.allocate_exact_size(
                                        Vec2::new(inner_w, row_h),
                                        Sense::click(),
                                    );
                                    let is_cur =
                                        playing && (session.playback.speed - spd).abs() < 1e-3;
                                    if r_resp.hovered() {
                                        ui.painter().rect_filled(
                                            r_rect,
                                            CornerRadius::ZERO,
                                            menu_active_bg,
                                        );
                                    } else if is_cur {
                                        ui.painter().rect_filled(
                                            r_rect,
                                            CornerRadius::ZERO,
                                            BG_DROPDOWN_HOVER,
                                        );
                                    }
                                    let label = if (spd - spd.round()).abs() < 1e-3
                                        || ((spd * 10.0) - (spd * 10.0).round()).abs() < 1e-3
                                    {
                                        format!("{spd:.1}x")
                                    } else {
                                        format!("{spd:.2}x")
                                    };
                                    ui.painter().text(
                                        Pos2::new(r_rect.left() + 8.0, r_rect.center().y),
                                        Align2::LEFT_CENTER,
                                        label,
                                        theme.font(FONT_SMALL),
                                        if r_resp.hovered() {
                                            menu_active_fg
                                        } else if is_cur {
                                            TEXT_BRIGHT
                                        } else {
                                            TEXT_PRIMARY
                                        },
                                    );
                                    if r_resp.clicked() {
                                        let _ = session.execute(Command::SetSpeed { speed: spd });
                                        close_menu = true;
                                    }
                                }

                                // Separator line
                                let (sep_rect, _) = ui
                                    .allocate_exact_size(Vec2::new(inner_w, sep_h), Sense::hover());
                                let sy = sep_rect.center().y.round();
                                ui.painter().rect_filled(
                                    Rect::from_min_max(
                                        Pos2::new(sep_rect.left() + 4.0, sy),
                                        Pos2::new(sep_rect.right() - 4.0, sy + 1.0),
                                    ),
                                    CornerRadius::ZERO,
                                    BORDER_PANEL,
                                );

                                // Pause Render toggle
                                let (pr_rect, pr_resp) = ui
                                    .allocate_exact_size(Vec2::new(inner_w, row_h), Sense::click());
                                if pr_resp.hovered() {
                                    ui.painter().rect_filled(
                                        pr_rect,
                                        CornerRadius::ZERO,
                                        menu_active_bg,
                                    );
                                } else if self.player_plus.drawing_paused {
                                    ui.painter().rect_filled(
                                        pr_rect,
                                        CornerRadius::ZERO,
                                        BG_DROPDOWN_HOVER,
                                    );
                                }
                                let pr_label = if self.player_plus.drawing_paused {
                                    "✓ Pause Render"
                                } else {
                                    "Pause Render"
                                };
                                ui.painter().text(
                                    Pos2::new(pr_rect.left() + 8.0, pr_rect.center().y),
                                    Align2::LEFT_CENTER,
                                    pr_label,
                                    theme.font(FONT_SMALL),
                                    if pr_resp.hovered() {
                                        menu_active_fg
                                    } else if self.player_plus.drawing_paused {
                                        TEXT_BRIGHT
                                    } else {
                                        TEXT_PRIMARY
                                    },
                                );
                                if pr_resp.clicked() {
                                    self.player_plus.drawing_paused =
                                        !self.player_plus.drawing_paused;
                                    close_menu = true;
                                }
                            });
                        paint_panel_border(ui.painter(), frame_resp.response.rect, theme.accent());
                    });

                let clicked_outside = ui.input(|i| {
                    i.pointer.primary_pressed()
                        && i.pointer.interact_pos().is_some_and(|p| {
                            !area_resp.response.rect.contains(p) && !speed_rect.contains(p)
                        })
                });
                if close_menu || clicked_outside {
                    self.speed_menu_open = false;
                    // Ensure closed speed dropdown never retains egui keyboard focus so
                    // Delete/Backspace and transport shortcuts continue firing immediately.
                    ui.memory_mut(|m| m.surrender_focus(p_resp.id));
                }
            } else if p_resp.has_focus() {
                ui.memory_mut(|m| m.surrender_focus(p_resp.id));
            }

            // Fixed-width quality text trigger ("Full" / "Half" / "Quarter" / "1/8")
            let q_rect = snap_rect(Rect::from_min_size(
                Pos2::new(rx.round(), ctrl_y),
                Vec2::new(q_w, ctrl_h),
            ));
            let q_resp = ui
                .interact(q_rect, Id::unique("pp_quality"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text(
                    "Player+: Left-click to cycle quality · Right-click for adaptive/upscale options",
                );
            if q_resp.hovered() && !q_resp.has_focus() {
                ui.painter().rect_filled(
                    q_rect,
                    CornerRadius::ZERO,
                    Color32::from_rgba_unmultiplied(0xff, 0xff, 0xff, 10),
                );
            }
            if q_resp.has_focus() {
                ui.painter().rect_stroke(
                    q_rect,
                    CornerRadius::ZERO,
                    Stroke::new(2.0, theme.accent()),
                    StrokeKind::Inside,
                );
            }
            ui.painter().text(
                Pos2::new(q_rect.center().x, q_rect.center().y - 0.5),
                Align2::CENTER_CENTER,
                q_label,
                theme.font(m.pill_font),
                theme.value(),
            );
            if q_resp.clicked() {
                let idx = PreviewQuality::ALL
                    .iter()
                    .position(|&q| q == self.player_plus.preview_quality)
                    .unwrap_or(0);
                self.player_plus.preview_quality =
                    PreviewQuality::ALL[(idx + 1) % PreviewQuality::ALL.len()];
                self.player_plus.adaptive_divisor = 1;
                ui.memory_mut(|m| m.surrender_focus(q_resp.id));
            }
            q_resp.context_menu(|ui| {
                ui.set_min_width(168.0);
                ui.label(egui::RichText::new("Preview Quality").strong());
                for q in PreviewQuality::ALL {
                    if ui
                        .selectable_label(self.player_plus.preview_quality == q, q.label())
                        .clicked()
                    {
                        self.player_plus.preview_quality = q;
                        self.player_plus.adaptive_divisor = 1;
                        ui.close();
                    }
                }
                ui.separator();
                if ui
                    .checkbox(&mut self.player_plus.adaptive_quality, "Adaptive quality")
                    .changed()
                    && !self.player_plus.adaptive_quality
                {
                    self.player_plus.adaptive_divisor = 1;
                }
                ui.checkbox(
                    &mut self.player_plus.sharp_upscale,
                    "Sharp upscale (Nearest)",
                );
                ui.checkbox(&mut self.player_plus.exact_effects, "Exact effects")
                    .on_hover_text(
                        "Evaluate effects at full sequence resolution (disables preview downsampling)",
                    );
                ui.checkbox(&mut self.player_plus.drop_frames, "Drop frames (realtime)")
                    .on_hover_text(
                        "Skip uncached frames to keep real-time playback speed instead of caching every frame",
                    );
            });
        } else {
            self.speed_menu_open = false;
        }

        let border = if focused {
            theme.accent()
        } else {
            BORDER_PANEL
        };
        paint_panel_border(ui.painter(), rect, border);
    }

    // -----------------------------------------------------------------------
    // Right Bottom: Timeline Panel Frame
    // -----------------------------------------------------------------------

    fn show_timeline_frame(
        &mut self,
        ui: &mut Ui,
        rect: Rect,
        session: &mut Session,
        theme: &Cm3Theme,
        draw_timeline: &mut impl FnMut(&mut Ui, Rect, bool, &mut Session),
    ) {
        let rect = snap_rect(rect);
        let focused = self.focused_panel == PanelFocus::Timeline;
        ui.painter().rect_filled(rect, CornerRadius::ZERO, BG_PANEL);

        let inner = snap_rect(rect.shrink(1.0));
        if Self::clicked_in_pane(ui, rect) {
            self.focused_panel = PanelFocus::Timeline;
            self.panels.floating_windows.clear_focus();
        }

        ui.scope_builder(
            UiBuilder::new()
                .max_rect(inner)
                .id_salt("cm3_timeline_pane"),
            |ui| {
                ui.set_clip_rect(ui.clip_rect().intersect(inner));
                draw_timeline(ui, inner, focused, session);
            },
        );

        let border = if focused {
            theme.accent()
        } else {
            BORDER_PANEL
        };
        paint_panel_border(ui.painter(), rect, border);
    }

    // -----------------------------------------------------------------------
    // Global CM3 Keyboard Shortcut Dispatcher (Spec 01 §6)
    // -----------------------------------------------------------------------

    fn handle_global_shortcuts(
        &mut self,
        ctx: &egui::Context,
        session: &mut Session,
        actions: &mut Vec<UiAction>,
    ) {
        // Strict input-focus & modal guard (Spec 01 §6.1)
        if self.suppress_shortcuts
            || ctx.egui_wants_keyboard_input()
            || ctx.text_edit_focused()
            || self.project_name_editing
        {
            return;
        }

        // Inside floating windows, only Undo / Redo / Save forward to CM3 (Spec 01 §4.8)
        let in_floating = self.panels.floating_windows.focused_id.is_some();
        let in_shader_editor =
            self.panels.floating_windows.focused_id.as_deref() == Some("shader_editor");

        // When host shell already dispatched shortcuts via `shortcuts::dispatch_shortcuts_with_focus`,
        // only handle workspace-exclusive Backtick maximize so suppressed keys (e.g. Space during
        // pointer drag or Inspector Delete) are never re-fired here.
        if self.external_shortcuts_dispatched {
            if !in_floating {
                ctx.input(|i| {
                    let ctrl = i.modifiers.ctrl || i.modifiers.command;
                    let alt = i.modifiers.alt;
                    if !ctrl && !alt && i.key_pressed(Key::Backtick) {
                        if self.maximized_panel.is_some() {
                            self.maximized_panel = None;
                        } else if matches!(
                            self.focused_panel,
                            PanelFocus::Viewport | PanelFocus::Timeline | PanelFocus::SideInspector
                        ) {
                            self.maximized_panel = Some(self.focused_panel);
                        }
                    }
                });
            }
            return;
        }

        ctx.input(|i| {
            let ctrl = i.modifiers.ctrl || i.modifiers.command;
            let shift = i.modifiers.shift;
            let alt = i.modifiers.alt;

            // Project & History (also forwarded from floating windows; avoid Ctrl+S clash with Shader Editor)
            if ctrl && !alt && !shift && i.key_pressed(Key::Z) {
                let _ = session.execute(Command::Undo);
                return;
            }
            if ctrl && !alt && (i.key_pressed(Key::Y) || (shift && i.key_pressed(Key::Z))) {
                let _ = session.execute(Command::Redo);
                return;
            }
            if ctrl && !alt && !in_shader_editor && i.key_pressed(Key::S) {
                actions.push(UiAction::SaveProject {
                    include_assets: !shift,
                });
                return;
            }

            if in_floating {
                return;
            }

            let non_repeat_space = i.events.iter().any(|ev| {
                matches!(
                    ev,
                    egui::Event::Key {
                        key: Key::Space,
                        pressed: true,
                        repeat: false,
                        modifiers,
                        ..
                    } if !modifiers.ctrl && !modifiers.command && !modifiers.shift && !modifiers.alt
                )
            });

            if ctrl && !shift && !alt && (i.key_pressed(Key::N) || i.key_pressed(Key::M)) {
                actions.push(UiAction::NewProject);
            } else if ctrl && !shift && !alt && i.key_pressed(Key::O) {
                actions.push(UiAction::OpenProject);
            } else if ctrl && !shift && !alt && i.key_pressed(Key::L) {
                let _ = session.execute(Command::ToggleLoop);
            } else if ctrl && !shift && !alt && i.key_pressed(Key::G) {
                self.graph_open = !self.graph_open;
                actions.push(UiAction::OpenGraphEditor);
            } else if !ctrl && !alt && i.key_pressed(Key::Backtick) {
                if self.maximized_panel.is_some() {
                    self.maximized_panel = None;
                } else if matches!(
                    self.focused_panel,
                    PanelFocus::Viewport | PanelFocus::Timeline | PanelFocus::SideInspector
                ) {
                    self.maximized_panel = Some(self.focused_panel);
                }
            } else if non_repeat_space && !i.pointer.primary_down() {
                let _ = session.execute(Command::TogglePlay);
            } else if !ctrl && !shift && !alt && i.key_pressed(Key::Home) {
                let _ = session.execute(Command::SetFrame { frame: 0.0 });
            } else if !ctrl && !shift && !alt && i.key_pressed(Key::End) {
                let end_f = (session.document().sequence_length - 1.0).max(0.0);
                let _ = session.execute(Command::SetFrame { frame: end_f });
            } else if !ctrl && !alt && i.key_pressed(Key::ArrowLeft) {
                let delta = if shift { -5.0 } else { -1.0 };
                let _ = session.execute(Command::StepFrame { delta });
            } else if !ctrl && !alt && i.key_pressed(Key::ArrowRight) {
                let delta = if shift { 5.0 } else { 1.0 };
                let _ = session.execute(Command::StepFrame { delta });
            } else if !ctrl && !shift && !alt && i.key_pressed(Key::J) {
                let speed = super::shortcuts::next_shuttle_reverse_speed(session.playback.speed);
                let _ = session.execute(Command::SetSpeed { speed });
            } else if !ctrl && !shift && !alt && i.key_pressed(Key::K) {
                let _ = session.execute(Command::SetSpeed { speed: 0.0 });
                session.playback.last_speed = 1.0;
            } else if !ctrl && !shift && !alt && i.key_pressed(Key::L) {
                let speed = super::shortcuts::next_shuttle_forward_speed(session.playback.speed);
                let _ = session.execute(Command::SetSpeed { speed });
            } else if !ctrl
                && !shift
                && !alt
                && i.key_pressed(Key::C)
                && self.focused_panel != PanelFocus::Timeline
            {
                let _ = toggle_editing_camera_synced(session);
            } else if !ctrl && !shift && !alt && i.key_pressed(Key::T) {
                session.playback.toggle_transform_controls_2d();
            }
        });
    }
}

#[allow(dead_code)]
fn _keep_theme_imports_used() -> (Color32, Color32, Color32, Frame, Shape) {
    (
        TEXT_PRIMARY,
        TEXT_DISABLED,
        TEXT_BRIGHT,
        Frame::NONE,
        Shape::Noop,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elevator_rail_fits_all_tabs_at_900x600_and_hides_assets_in_core_mode() {
        // Full mode: 11 tabs + 1 menu button = 12 slots; at 900x600 lower rail height is 557px.
        let full_tabs = Cm3Workspace::visible_side_tabs(true);
        assert_eq!(full_tabs.len(), 11);
        assert!(full_tabs.contains(&SideTab::Assets));
        assert!(full_tabs.contains(&SideTab::About));
        assert!(full_tabs.contains(&SideTab::Settings));

        let slot_h_600 = Cm3Workspace::compute_elevator_slot_height(557.0, full_tabs.len());
        assert!(
            slot_h_600 * (full_tabs.len() + 1) as f32 <= 557.0,
            "all 12 rail slots (including About & Settings) must fit within 900x600 height: slot_h={slot_h_600}"
        );
        // At 1280x720 (rail height 677px), slots stay exact 50px.
        let slot_h_720 = Cm3Workspace::compute_elevator_slot_height(677.0, full_tabs.len());
        assert!((slot_h_720 - 50.0).abs() < 1e-3);

        // Core Mode (asset-manager disabled): Assets tab is hidden.
        let core_tabs = Cm3Workspace::visible_side_tabs(false);
        assert_eq!(core_tabs.len(), 10);
        assert!(!core_tabs.contains(&SideTab::Assets));
    }

    #[test]
    fn responsive_900x600_workspace_keeps_sidebar_and_toolbars_visible() {
        let ctx = egui::Context::default();
        let mut ws = Cm3Workspace::new();
        let mut session = Session::new();
        let mut theme = Cm3Theme::default();

        // Enforce sensible minimum width at 900x600.
        let root_900 = Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 600.0));
        let (left_900, _sp, right_900) = ws.root_split.compute_rects(root_900);
        assert!(
            left_900.width() >= 270.0,
            "left sidebar must stay >= 270px at 900x600, got {}",
            left_900.width()
        );
        assert!(right_900.width() >= 580.0);

        // Render full workspace at 900x600 without panics or zero-size clipping.
        let raw = egui::RawInput {
            screen_rect: Some(root_900),
            ..Default::default()
        };
        let mut out = ctx.run_ui(raw, |ui| {
            let _ = ws.show(
                ui,
                &mut session,
                &mut theme,
                |_ui, _r, _f, _s| {},
                |_ui, _r, _f, _s| {},
            );
        });
        out.textures_delta.clear();

        // Core Mode redirect: if Assets tab was active and asset-manager is disabled, switches away.
        ws.active_tab = SideTab::Assets;
        let no_deps = std::collections::BTreeSet::new();
        session
            .plugins
            .disable("asset-manager", &no_deps)
            .expect("disable asset-manager");
        let raw2 = egui::RawInput {
            screen_rect: Some(root_900),
            ..Default::default()
        };
        let mut out2 = ctx.run_ui(raw2, |ui| {
            let _ = ws.show(
                ui,
                &mut session,
                &mut theme,
                |_ui, _r, _f, _s| {},
                |_ui, _r, _f, _s| {},
            );
        });
        out2.textures_delta.clear();
        assert_ne!(
            ws.active_tab,
            SideTab::Assets,
            "disabling asset-manager must leave the Assets tab"
        );
    }

    #[test]
    fn speed_pause_pill_fits_paused_label_without_clipping_at_900x600() {
        let ctx = egui::Context::default();
        super::super::theme::install_cm3_fonts(&ctx);
        let theme = Cm3Theme::default();
        let mut paused_w = 0.0_f32;
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
            paused_w = ui
                .painter()
                .layout_no_wrap("Paused".to_owned(), theme.font(FONT_SMALL), theme.value())
                .size()
                .x;
        });
        out.textures_delta.clear();

        let compact_w = Cm3Workspace::compute_speed_pill_width(true, paused_w);
        let normal_w = Cm3Workspace::compute_speed_pill_width(false, paused_w);
        assert!(compact_w >= 80.0);
        assert!(normal_w >= 88.0);
        // Text starts at +6px and clips at right - 17px => available text span = width - 23px.
        assert!(
            paused_w + 2.0 <= compact_w - 23.0,
            "compact Speed/Pause pill ({compact_w}px) must fit full 'Paused' ({paused_w}px) + chevron without clipping 'd'"
        );
        // Verify full 3-group compact playback toolbar still fits inside 900x600 right pane (~617px inner).
        let left_w = 9.0 * 25.0 + 8.0 * 1.0;
        let center_w = 5.0 * 25.0 + 4.0 * 1.0;
        let right_w = (25.0 + 1.0 + 25.0) + (1.0 + compact_w + 1.0 + 50.0);
        let total_needed = 3.0 + left_w + 4.0 + center_w + 4.0 + right_w + 4.0;
        assert!(
            total_needed <= 617.0,
            "compact playback toolbar with widened Speed/Pause pill must fit 900x600: needed={total_needed}"
        );
    }

    #[test]
    fn splitter_hitbox_is_8px_and_clamped_to_container() {
        let container = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 540.0));
        let h_split = SplitPanelState::new(0.35, SplitOrientation::Horizontal, 270.0, 420.0);
        let (_p0, split_rect, _p1) = h_split.compute_rects(container);
        assert!((split_rect.width() - SPLITTER_SIZE).abs() < 1e-3);
        let h_hit = compute_splitter_hitbox(split_rect, container, true);
        assert!(
            (h_hit.width() - SPLITTER_HITBOX_SIZE).abs() < 1e-3,
            "horizontal splitter hitbox must be 8px, got {}",
            h_hit.width()
        );
        assert!(h_hit.left() >= container.left() && h_hit.right() <= container.right());

        let v_split = SplitPanelState::new(0.65, SplitOrientation::Vertical, 120.0, 120.0);
        let (_vp0, v_split_rect, _vp1) = v_split.compute_rects(container);
        let v_hit = compute_splitter_hitbox(v_split_rect, container, false);
        assert!(
            (v_hit.height() - SPLITTER_HITBOX_SIZE).abs() < 1e-3,
            "vertical splitter hitbox must be 8px, got {}",
            v_hit.height()
        );
        assert!(v_hit.top() >= container.top() && v_hit.bottom() <= container.bottom());
    }

    #[test]
    fn playback_toolbar_fits_all_controls_at_800x540_and_min_pane_width() {
        let ctx = egui::Context::default();
        super::super::theme::install_cm3_fonts(&ctx);
        let theme = Cm3Theme::default();
        let mut paused_small = 0.0_f32;
        let mut paused_compact = 0.0_f32;
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
            paused_small = ui
                .painter()
                .layout_no_wrap("Paused".to_owned(), theme.font(FONT_SMALL), theme.value())
                .size()
                .x;
            paused_compact = ui
                .painter()
                .layout_no_wrap("Paused".to_owned(), theme.font(FONT_COMPACT), theme.value())
                .size()
                .x;
        });
        out.textures_delta.clear();

        // Check both 800x540 default right pane (inner ~525px) and min_pane_1=420px (inner 418px).
        for inner_w in [525.0_f32, 418.0_f32] {
            let m = Cm3Workspace::compute_playback_toolbar_metrics(
                inner_w,
                paused_small,
                paused_compact,
            );
            let left_w = 9.0 * m.btn_size + 8.0 * m.btn_gap;
            let center_w = 5.0 * m.btn_size + 4.0 * m.btn_gap;
            let right_w = (m.btn_size + m.btn_gap + m.btn_size)
                + (m.pill_gap + m.speed_dd_w + m.pill_gap + m.q_w);
            let total_needed =
                m.edge_pad + left_w + m.group_gap + center_w + m.group_gap + right_w + m.right_pad;
            assert!(
                total_needed <= inner_w,
                "all playback controls (Graph, Paused, Full) must fit inside inner_w={inner_w}: needed={total_needed}"
            );
            let text_span = m.speed_dd_w - m.pill_left_inset - m.pill_chevron_reserve;
            assert!(
                paused_compact <= text_span + 0.5,
                "Paused label ({paused_compact}px) must fit inside pill text span ({text_span}px) at inner_w={inner_w}"
            );
        }
    }

    #[test]
    fn vertical_sidebar_splitter_1px_drag_tracks_smoothly_without_snap() {
        let ctx = egui::Context::default();
        let mut ws = Cm3Workspace::new();
        let mut session = Session::new();
        let mut theme = Cm3Theme::default();
        let root_rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));

        // Frame 0: establish initial layout and place pointer far to the right (x=800),
        // reproducing the exact pre-click jump condition that used to snap sidebar to 271px.
        let mut raw0 = egui::RawInput {
            screen_rect: Some(root_rect),
            ..Default::default()
        };
        raw0.events
            .push(egui::Event::PointerMoved(Pos2::new(800.0, 360.0)));
        let mut out0 = ctx.run_ui(raw0, |ui| {
            let _ = ws.show(
                ui,
                &mut session,
                &mut theme,
                |_ui, _r, _f, _s| {},
                |_ui, _r, _f, _s| {},
            );
        });
        out0.textures_delta.clear();

        let (initial_left, initial_split, _initial_right) = ws.root_split.compute_rects(root_rect);
        let initial_w = initial_left.width();
        let split_x = initial_split.center().x;
        assert!(
            (initial_w - 395.0).abs() <= 2.0,
            "expected initial sidebar width ~395px at 1280px, got {initial_w}"
        );

        // Frame 1: pointer jumps from x=800 to the splitter border and presses primary button.
        let mut raw1 = egui::RawInput {
            screen_rect: Some(root_rect),
            ..Default::default()
        };
        raw1.events
            .push(egui::Event::PointerMoved(Pos2::new(split_x, 360.0)));
        raw1.events.push(egui::Event::PointerButton {
            pos: Pos2::new(split_x, 360.0),
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        let mut out1 = ctx.run_ui(raw1, |ui| {
            let _ = ws.show(
                ui,
                &mut session,
                &mut theme,
                |_ui, _r, _f, _s| {},
                |_ui, _r, _f, _s| {},
            );
        });
        out1.textures_delta.clear();

        let (after_press_left, _, _) = ws.root_split.compute_rects(root_rect);
        assert!(
            (after_press_left.width() - initial_w).abs() <= 1.0,
            "pressing splitter after cursor jump must NOT snap sidebar width: initial={initial_w}, after_press={}",
            after_press_left.width()
        );

        // Frame 2: drag right by exactly 1px.
        let mut raw2 = egui::RawInput {
            screen_rect: Some(root_rect),
            ..Default::default()
        };
        raw2.events
            .push(egui::Event::PointerMoved(Pos2::new(split_x + 1.0, 360.0)));
        let mut out2 = ctx.run_ui(raw2, |ui| {
            let _ = ws.show(
                ui,
                &mut session,
                &mut theme,
                |_ui, _r, _f, _s| {},
                |_ui, _r, _f, _s| {},
            );
        });
        out2.textures_delta.clear();

        let (after_1px_left, _, _) = ws.root_split.compute_rects(root_rect);
        assert!(
            (after_1px_left.width() - (initial_w + 1.0)).abs() <= 1.0,
            "1px drag on vertical splitter must move smoothly from {initial_w} to ~{}, got {}",
            initial_w + 1.0,
            after_1px_left.width()
        );

        // Frame 3: release pointer and verify responsive min sidebar width >= 300px at >= 1000px.
        let mut raw3 = egui::RawInput {
            screen_rect: Some(root_rect),
            ..Default::default()
        };
        raw3.events.push(egui::Event::PointerButton {
            pos: Pos2::new(split_x + 1.0, 360.0),
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        });
        let mut out3 = ctx.run_ui(raw3, |ui| {
            let _ = ws.show(
                ui,
                &mut session,
                &mut theme,
                |_ui, _r, _f, _s| {},
                |_ui, _r, _f, _s| {},
            );
        });
        out3.textures_delta.clear();

        assert_eq!(Cm3Workspace::compute_root_min_sidebar_width(1440.0), 300.0);
        assert_eq!(Cm3Workspace::compute_root_min_sidebar_width(1000.0), 300.0);
        assert_eq!(Cm3Workspace::compute_root_min_sidebar_width(800.0), 270.0);
    }

    #[test]
    fn horizontal_viewport_timeline_splitter_drags_down_and_up_without_ruler_conflict() {
        let ctx = egui::Context::default();
        let mut ws = Cm3Workspace::new();
        let mut session = Session::new();
        let mut theme = Cm3Theme::default();
        let root_rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));

        // Frame 0: initial layout with pointer up near viewport top (y=80).
        let mut raw0 = egui::RawInput {
            screen_rect: Some(root_rect),
            ..Default::default()
        };
        raw0.events
            .push(egui::Event::PointerMoved(Pos2::new(800.0, 80.0)));
        let mut timeline_rect_seen = Rect::NOTHING;
        let mut out0 = ctx.run_ui(raw0, |ui| {
            let _ = ws.show(
                ui,
                &mut session,
                &mut theme,
                |vp_ui, vp_r, _f, _s| {
                    // Simulate viewport full-rect click_and_drag widget to ensure it does not steal splitter hitbox.
                    let _ =
                        vp_ui.interact(vp_r, Id::unique("test_vp_drag"), Sense::click_and_drag());
                },
                |_tl_ui, tl_r, _f, _s| {
                    timeline_rect_seen = tl_r;
                },
            );
        });
        out0.textures_delta.clear();

        let (_left, _root_sp, right_rect) = ws.root_split.compute_rects(root_rect);
        let (init_vp, init_h_split, _init_bot) = ws.right_split.compute_rects(right_rect);
        let h_hit = compute_splitter_hitbox(init_h_split, right_rect, false);
        assert!(
            h_hit.bottom() + 35.0 < timeline_rect_seen.top(),
            "horizontal splitter 8px hitbox (bottom={}) must sit cleanly on viewport/toolbar seam well above timeline ruler (top={})",
            h_hit.bottom(),
            timeline_rect_seen.top()
        );

        let init_vp_h = init_vp.height();
        let grab_x = right_rect.center().x;
        // Grab inside the 8px hitbox slightly into the viewport side (1.5px above the 3px gap)
        // to prove post-pane splitter registration wins over viewport click_and_drag.
        let grab_y = init_h_split.top() - 1.5;

        // Frame 1: press on horizontal splitter after cursor jump from y=80.
        let mut raw1 = egui::RawInput {
            screen_rect: Some(root_rect),
            ..Default::default()
        };
        raw1.events
            .push(egui::Event::PointerMoved(Pos2::new(grab_x, grab_y)));
        raw1.events.push(egui::Event::PointerButton {
            pos: Pos2::new(grab_x, grab_y),
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        let mut out1 = ctx.run_ui(raw1, |ui| {
            let _ = ws.show(
                ui,
                &mut session,
                &mut theme,
                |vp_ui, vp_r, _f, _s| {
                    let _ =
                        vp_ui.interact(vp_r, Id::unique("test_vp_drag"), Sense::click_and_drag());
                },
                |_tl_ui, _tl_r, _f, _s| {},
            );
        });
        out1.textures_delta.clear();

        let (vp_after_press, _, _) = ws.right_split.compute_rects(right_rect);
        assert!(
            (vp_after_press.height() - init_vp_h).abs() <= 1.0,
            "pressing horizontal splitter must not jump viewport height: init={init_vp_h}, after_press={}",
            vp_after_press.height()
        );

        // Frame 2: drag DOWN by +40px.
        let mut raw2 = egui::RawInput {
            screen_rect: Some(root_rect),
            ..Default::default()
        };
        raw2.events
            .push(egui::Event::PointerMoved(Pos2::new(grab_x, grab_y + 40.0)));
        let mut out2 = ctx.run_ui(raw2, |ui| {
            let _ = ws.show(
                ui,
                &mut session,
                &mut theme,
                |vp_ui, vp_r, _f, _s| {
                    let _ =
                        vp_ui.interact(vp_r, Id::unique("test_vp_drag"), Sense::click_and_drag());
                },
                |_tl_ui, _tl_r, _f, _s| {},
            );
        });
        out2.textures_delta.clear();

        let (vp_after_down, _, _) = ws.right_split.compute_rects(right_rect);
        assert!(
            (vp_after_down.height() - (init_vp_h + 40.0)).abs() <= 1.5,
            "dragging horizontal splitter down +40px must grow viewport to ~{}, got {}",
            init_vp_h + 40.0,
            vp_after_down.height()
        );

        // Frame 3: drag BACK UP by -60px (net -20px from start).
        let mut raw3 = egui::RawInput {
            screen_rect: Some(root_rect),
            ..Default::default()
        };
        raw3.events
            .push(egui::Event::PointerMoved(Pos2::new(grab_x, grab_y - 20.0)));
        let mut out3 = ctx.run_ui(raw3, |ui| {
            let _ = ws.show(
                ui,
                &mut session,
                &mut theme,
                |vp_ui, vp_r, _f, _s| {
                    let _ =
                        vp_ui.interact(vp_r, Id::unique("test_vp_drag"), Sense::click_and_drag());
                },
                |_tl_ui, _tl_r, _f, _s| {},
            );
        });
        out3.textures_delta.clear();

        let (vp_after_up, _, _) = ws.right_split.compute_rects(right_rect);
        assert!(
            (vp_after_up.height() - (init_vp_h - 20.0)).abs() <= 1.5,
            "dragging horizontal splitter back up -60px must move viewport to ~{}, got {}",
            init_vp_h - 20.0,
            vp_after_up.height()
        );
    }

    #[test]
    fn player_plus_defaults_to_caching_every_frame_without_dropping() {
        let state = PlayerPlusState::default();
        assert!(
            !state.drop_frames,
            "PlayerPlusState must default drop_frames to false so RAM preview caches every frame cleanly"
        );
    }
}
