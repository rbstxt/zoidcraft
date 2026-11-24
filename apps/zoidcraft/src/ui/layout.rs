//! Pixel-exact recursive binary Split Panel engine, CM3 workspace shell,
//! animated Elevator rail, File & Playback toolbars, Audio Level Meter,
//! and global shortcut dispatcher (Spec 01 §1 & §6).

use egui::{
    Align, Align2, Area, Color32, CornerRadius, CursorIcon, Frame, Id, Key, Layout, Mesh, Order,
    Painter, Pos2, Rect, ScrollArea, Sense, Shape, Stroke, StrokeKind, TextEdit, Ui, UiBuilder,
    Vec2,
};
use zc_session::{Command, Session};

use super::panels::{ExportSettings, PanelAction, PanelsState, SideTab, show_side_panel};
use super::theme::{
    AUDIO_PEAK_YELLOW, BG_INPUT, BG_PANEL, BG_SUNKEN, BG_TITLE, BG_TOOLBAR, BORDER_CONTROL,
    BORDER_PANEL, BORDER_RAIL_FOCUS, Cm3Icon, Cm3Theme, FONT_BODY, FONT_SMALL, FONT_TAB,
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

/// Exact CM3 splitter thickness (`3 px` visual gap and pointer hitbox).
pub const SPLITTER_SIZE: f32 = 3.0;
const HALF_SPLITTER: f32 = SPLITTER_SIZE * 0.5;

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

/// Draws and handles dragging for a `3 px` CM3 Split Panel bar, returning `(pane_0_rect, pane_1_rect)`.
pub fn begin_split_panel(
    ui: &mut Ui,
    id: Id,
    rect: Rect,
    state: &mut SplitPanelState,
) -> (Rect, Rect) {
    let horiz = state.resolved_horizontal(rect);
    let (p0, split_rect, _p1) = state.compute_rects(rect);

    let resp = ui
        .interact(split_rect, id.with("splitter"), Sense::drag())
        .on_hover_cursor(if horiz {
            CursorIcon::ResizeHorizontal
        } else {
            CursorIcon::ResizeVertical
        });
    if resp.dragged() {
        let extent = if horiz {
            rect.width().max(1.0)
        } else {
            rect.height().max(1.0)
        };
        let cur_p0_extent = if horiz { p0.width() } else { p0.height() };
        let delta = if horiz {
            resp.drag_delta().x
        } else {
            resp.drag_delta().y
        };
        let max_p0 = (extent - state.min_pane_1 - SPLITTER_SIZE).max(0.0);
        let min_p0 = state.min_pane_0.min(max_p0);
        let new_p0 = (cur_p0_extent + delta).clamp(min_p0, max_p0);
        state.ratio = ((new_p0 + HALF_SPLITTER) / extent).clamp(0.0, 1.0);
    }

    // Recompute rects AFTER updating ratio and paint splitter at its live zero-lag position.
    let (p0, new_split_rect, p1) = state.compute_rects(rect);
    ui.painter()
        .rect_filled(new_split_rect, CornerRadius::ZERO, VOID_BLACK);
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

/// Runtime state for `player-plus` viewport toolbar controls.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlayerPlusState {
    pub drawing_paused: bool,
    pub single_render_requested: bool,
    pub preview_quality: PreviewQuality,
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
    CancelExport,
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
    viewport_draw_tool: Option<u8>,
    /// Tracks whether the bottom inspector sub-pane (vs top list sub-pane) is focused on split tabs.
    split_inspector_focus_bottom: bool,
    /// Whether the Graph Editor floating window is currently open (drives Graph toolbar button border).
    pub graph_open: bool,
    /// All side-panel inspector states.
    pub panels: PanelsState,
    /// When `true` (e.g. a modal confirmation dialog is open), suppresses workspace shortcuts.
    pub suppress_shortcuts: bool,
}

impl Default for Cm3Workspace {
    fn default() -> Self {
        Self::new()
    }
}

impl Cm3Workspace {
    /// Creates a new CM3 workspace initialized with exact Spec 01 §1.2 ratios.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root_split: SplitPanelState::new(0.30, SplitOrientation::Horizontal, 200.0, 260.0),
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
            motion_path_visible: true,
            viewport_draw_tool: None,
            split_inspector_focus_bottom: false,
            graph_open: false,
            panels: PanelsState::default(),
            suppress_shortcuts: false,
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

        // Root Horizontal Split: 30% Left Column | 70% Right Workspace
        let (left_rect, right_rect) = begin_split_panel(
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
                // Vertical Split: 65% Viewport / 35% (40px Playback Toolbar + Timeline/Audio Meter)
                let (vp_rect, bottom_rect) = begin_split_panel(
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
            },
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
        // 5 buttons (29px each) + 4 gaps (5px) + 15px undo/redo separator = 180px
        let buttons_total_w = 5.0 * 29.0 + 4.0 * 5.0 + 15.0;
        // Exact CSS grid: minmax(100px, 1fr) auto minmax(0, 1fr)
        let show_name_col = inner.width() >= 105.0 + buttons_total_w;

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

        // Column 1: Zoidium inline project name input (min 100x29 at x=inner.left()+5, y=inner.top()+5)
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
            let free_w = (inner.width() - buttons_total_w).max(0.0);
            let col1_w = (free_w * 0.5)
                .max(105.0)
                .min((inner.width() - buttons_total_w).max(105.0))
                .round();
            let name_w = (col1_w - 5.0)
                .max(100.0)
                .min((inner.width() - buttons_total_w - 5.0).max(100.0))
                .round();
            let name_rect = snap_rect(Rect::from_min_size(
                Pos2::new((inner.left() + 5.0).round(), (inner.top() + 5.0).round()),
                Vec2::new(name_w, 29.0),
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
                Pos2::new(name_rect.left() + 6.0, name_rect.top() + 4.0),
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
                                .font(theme.font(FONT_BODY))
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
            name_rect.right().round()
        } else {
            (inner.center().x - buttons_total_w * 0.5)
                .max(inner.left() + 2.0)
                .round()
        };

        // Column 2: New, Load, Save [dirty dot], divider gap, Undo, Redo
        let btn_y = (inner.top() + 5.0).round();

        let toolbar_btn = |cursor_x: &mut f32,
                           icon: Cm3Icon,
                           tip: &str,
                           enabled: bool,
                           dirty_dot: bool|
         -> bool {
            let b_rect = snap_rect(Rect::from_min_size(
                Pos2::new(cursor_x.round(), btn_y),
                Vec2::splat(29.0),
            ));
            *cursor_x += 34.0; // 29px + 5px right margin
            let resp = ui
                .interact(b_rect, Id::unique(("ftb", tip)), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text(tip);
            if resp.hovered() && !resp.has_focus() {
                ui.painter().rect_filled(
                    b_rect,
                    CornerRadius::ZERO,
                    Color32::from_rgba_unmultiplied(0xff, 0xff, 0xff, 10),
                );
            }
            // 2px theme.accent() border ONLY on keyboard/active focus (not passive hover)
            if resp.has_focus() {
                ui.painter().rect_stroke(
                    b_rect,
                    CornerRadius::ZERO,
                    Stroke::new(2.0, theme.accent()),
                    StrokeKind::Inside,
                );
            }
            let icon_rect = Rect::from_center_size(b_rect.center(), Vec2::splat(25.0));
            // Render all 5 toolbar icons at full #acacac unconditionally (do not dim Undo/Redo)
            paint_cm3_icon(ui.painter(), icon_rect, icon, TOOLBAR_ICON_INACTIVE);

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

        if toolbar_btn(&mut cursor_x, Cm3Icon::New, "new (ctrl-m)", true, false) {
            actions.push(UiAction::NewProject);
        }
        if toolbar_btn(&mut cursor_x, Cm3Icon::Load, "open (ctrl-o)", true, false) {
            actions.push(UiAction::OpenProject);
        }
        if toolbar_btn(&mut cursor_x, Cm3Icon::Save, "save (ctrl-s)", true, true) {
            let include_assets = !ui.input(|i| i.modifiers.shift);
            actions.push(UiAction::SaveProject { include_assets });
        }
        cursor_x += 15.0; // separator before Undo/Redo
        if toolbar_btn(
            &mut cursor_x,
            Cm3Icon::Undo,
            "undo (ctrl-z)",
            session.history.can_undo(),
            false,
        ) {
            let _ = session.execute(Command::Undo);
        }
        if toolbar_btn(
            &mut cursor_x,
            Cm3Icon::Redo,
            "redo (ctrl-y)",
            session.history.can_redo(),
            false,
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
                PanelAction::StartExport(cfg) => UiAction::StartExport(cfg),
                PanelAction::CancelExport => UiAction::CancelExport,
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
                            self.show_elevator_rail(ui, drawer_rect, theme);
                        },
                    );
                });
        } else {
            let rail_rect = snap_rect(Rect::from_min_size(
                rect.left_top(),
                Vec2::new(rail_base_w, rect.height()),
            ));
            self.show_elevator_rail(ui, rail_rect, theme);
        }
    }

    fn show_elevator_rail(&mut self, ui: &mut Ui, rail_rect: Rect, theme: &Cm3Theme) {
        let rail_rect = snap_rect(rail_rect);
        ui.painter()
            .rect_filled(rail_rect, CornerRadius::ZERO, VOID_BLACK);
        let rail_focused = self.focused_panel == PanelFocus::ElevatorRail;
        let tab_count = SideTab::ALL.len();
        if tab_count > 0 && self.rail_focus_index >= tab_count {
            self.rail_focus_index = tab_count - 1;
        }

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
                    self.active_tab = SideTab::ALL[self.rail_focus_index];
                    self.rail_expanded = false;
                }
            });
        }

        // Top 50px Menu Toggle button (30x30 icon horizontally centered at x = left + 15px)
        let menu_rect = snap_rect(Rect::from_min_size(
            rail_rect.left_top(),
            Vec2::new(rail_rect.width(), 50.0_f32.min(rail_rect.height())),
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
        let menu_icon_rect = Rect::from_min_size(
            Pos2::new(
                (menu_rect.left() + 15.0).round(),
                (menu_rect.top() + 10.0).round(),
            ),
            Vec2::splat(30.0),
        );
        paint_cm3_icon(ui.painter(), menu_icon_rect, Cm3Icon::Menu, TEXT_STRONG);
        if menu_resp.clicked() {
            self.rail_expanded = !self.rail_expanded;
            self.focused_panel = PanelFocus::ElevatorRail;
            self.panels.floating_windows.clear_focus();
        }

        // Scrollable Tab Strip (top: 50px to bottom, 50px rows, all 11 Zoidium tabs)
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
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing = Vec2::ZERO;
                        for (idx, tab) in SideTab::ALL.iter().copied().enumerate() {
                            let (row_rect, row_resp) = ui.allocate_exact_size(
                                Vec2::new(strip_rect.width(), 50.0),
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
                            // 30x30 icon centered in 60px rail at x = left + 15px, y = top + 11px
                            let icon_rect = Rect::from_min_size(
                                Pos2::new(
                                    (draw_row.left() + 15.0).round(),
                                    (draw_row.top() + 11.0).round(),
                                ),
                                Vec2::splat(30.0),
                            );
                            paint_cm3_icon(ui.painter(), icon_rect, tab.icon(), fg);

                            // 22px label when expanded at left + 62px
                            if strip_rect.width() > 64.0 {
                                let label_x = (draw_row.left() + 62.0).round();
                                ui.painter().with_clip_rect(draw_row).text(
                                    Pos2::new(label_x, draw_row.center().y),
                                    Align2::LEFT_CENTER,
                                    tab.title(),
                                    theme.font(FONT_TAB),
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
        let by = (inner.top() + (inner.height() - 29.0) * 0.5).floor();
        const BTN_SIZE: f32 = 29.0;
        const BTN_GAP: f32 = 2.0;
        const BTN_STEP: f32 = BTN_SIZE + BTN_GAP; // 31px

        #[derive(Clone, Copy)]
        enum PbIcon {
            Std(Cm3Icon),
            ShapePen,
            BezierPen,
            DrawPencil,
            MotionPath,
        }

        let pb_icon_btn =
            |cx: &mut f32, icon: PbIcon, tip: &str, tint: Color32, accent_border: bool| -> bool {
                let b_rect = snap_rect(Rect::from_min_size(
                    Pos2::new(cx.round(), by),
                    Vec2::splat(BTN_SIZE),
                ));
                *cx += BTN_STEP;
                let resp = ui
                    .interact(b_rect, Id::unique(("pbtb", tip)), Sense::click())
                    .on_hover_cursor(CursorIcon::PointingHand)
                    .on_hover_text(tip);
                if resp.hovered() && !resp.has_focus() && !accent_border {
                    ui.painter().rect_filled(
                        b_rect,
                        CornerRadius::ZERO,
                        Color32::from_rgba_unmultiplied(0xff, 0xff, 0xff, 10),
                    );
                }
                // 2px theme.accent() border ONLY on focus or active open state (never passive hover)
                if resp.has_focus() || accent_border {
                    ui.painter().rect_stroke(
                        b_rect,
                        CornerRadius::ZERO,
                        Stroke::new(2.0, theme.accent()),
                        StrokeKind::Inside,
                    );
                }
                let ic = Rect::from_center_size(b_rect.center(), Vec2::splat(25.0));
                match icon {
                    PbIcon::Std(icon) => paint_cm3_icon(ui.painter(), ic, icon, tint),
                    PbIcon::ShapePen => paint_shape_pen_icon(ui.painter(), ic, tint),
                    PbIcon::BezierPen => paint_bezier_pen_icon(ui.painter(), ic, tint),
                    PbIcon::DrawPencil => paint_draw_pencil_icon(ui.painter(), ic, tint),
                    PbIcon::MotionPath => paint_motion_path_icon(ui.painter(), ic, tint),
                }
                resp.clicked()
            };

        // --- LEFT group (starting inner.left() + 5px, 29x29 buttons, 2px gap -> 31px pitch) ---
        let mut left_x = (inner.left() + 5.0).round();
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
            if pb_icon_btn(
                &mut left_x,
                PbIcon::Std(Cm3Icon::TransparencyGrid),
                "Player+: toggle transparency grid",
                if self.transparency_grid {
                    theme.value()
                } else {
                    TEXT_MUTED
                },
                self.transparency_grid,
            ) {
                self.transparency_grid = !self.transparency_grid;
            }
        }
        if pb_icon_btn(
            &mut left_x,
            PbIcon::Std(Cm3Icon::Camera),
            "editing camera (c)",
            if session.playback.editing_camera {
                TOGGLE_ACTIVE
            } else {
                TOOLBAR_ICON_INACTIVE
            },
            false,
        ) {
            session.playback.toggle_editing_camera();
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
        let left_end = (left_x - BTN_GAP).max(inner.left());

        // --- CENTER group (horizontally centered on inner.center().x, 29x29 buttons, 2px gap) ---
        let transport_w = 5.0 * BTN_SIZE + 4.0 * BTN_GAP; // 153px
        let mut center_x = (inner.center().x - transport_w * 0.5)
            .round()
            .max(left_end + 6.0);
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
        let transport_end = center_x - BTN_GAP;

        // --- RIGHT group (right-aligned from inner.right() - 11px, matching Zoidium R0..R4) ---
        let is_playing = session.playback.is_playing();
        let right_edge = (inner.right() - 11.0).round();
        let right_avail = (right_edge - (transport_end + 6.0)).max(0.0);
        let q_label = self.player_plus.preview_quality.label();
        let q_w = (q_label.len() as f32 * 8.5 + 20.0).ceil().max(54.0);
        let paused_dd_w = 82.0_f32;
        let show_paused_dd = player_plus && !is_playing;
        let base_right_w = BTN_SIZE + BTN_GAP + BTN_SIZE; // Loop + Graph = 60px
        let pp_extra_w = if show_paused_dd {
            BTN_GAP + paused_dd_w + q_w
        } else {
            BTN_GAP + q_w
        };
        let show_pp_right = player_plus && right_avail >= base_right_w + pp_extra_w;
        let right_w = if show_pp_right {
            base_right_w + pp_extra_w
        } else {
            base_right_w
        };
        let mut rx = (right_edge - right_w).round().max(transport_end + 6.0);

        if rx + BTN_SIZE <= right_edge + 0.5
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
        if rx + BTN_SIZE <= right_edge + 0.5
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
            let ctrl_h = 27.0_f32;
            let ctrl_y = (inner.top() + (inner.height() - ctrl_h) * 0.5).floor();

            // 82x27 "Paused ▾" dropdown right to the left of "Full" whenever
            // player-plus is enabled and playback is paused (matching Zoidium startup toolbar).
            if show_paused_dd {
                let paused_rect = snap_rect(Rect::from_min_size(
                    Pos2::new(rx.round(), ctrl_y),
                    Vec2::new(paused_dd_w, ctrl_h),
                ));
                rx += paused_dd_w;
                let p_resp = ui
                    .interact(
                        paused_rect,
                        Id::unique("pp_paused_dropdown"),
                        Sense::click(),
                    )
                    .on_hover_cursor(CursorIcon::PointingHand)
                    .on_hover_text(if self.player_plus.drawing_paused {
                        "Player+: paused preview (drawing held; click to resume)"
                    } else {
                        "Player+: paused preview quality / hold mode"
                    });
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
                ui.painter()
                    .rect_filled(paused_rect, CornerRadius::ZERO, bg);
                if p_resp.hovered() && !p_resp.has_focus() && !self.player_plus.drawing_paused {
                    ui.painter().rect_filled(
                        paused_rect,
                        CornerRadius::ZERO,
                        Color32::from_rgba_unmultiplied(0xff, 0xff, 0xff, 10),
                    );
                }
                paint_panel_border(
                    ui.painter(),
                    paused_rect,
                    if p_resp.has_focus() {
                        theme.accent()
                    } else {
                        BORDER_CONTROL
                    },
                );
                ui.painter().text(
                    Pos2::new(paused_rect.left() + 8.0, paused_rect.center().y),
                    Align2::LEFT_CENTER,
                    "Paused",
                    theme.font(FONT_SMALL),
                    fg,
                );
                let chevron_rect = Rect::from_center_size(
                    Pos2::new(paused_rect.right() - 9.5, paused_rect.center().y - 0.5),
                    Vec2::splat(13.0),
                );
                paint_cm3_icon(ui.painter(), chevron_rect, Cm3Icon::ChevronDown, fg);
                if p_resp.clicked() {
                    self.player_plus.drawing_paused = !self.player_plus.drawing_paused;
                }
            }

            // Borderless quality text trigger ("Full" / "Half" / "Quarter", theme.value() #7e8fb9)
            let q_rect = snap_rect(Rect::from_min_size(
                Pos2::new(rx.round(), ctrl_y),
                Vec2::new(q_w, ctrl_h),
            ));
            let q_resp = ui
                .interact(q_rect, Id::unique("pp_quality"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text("Player+: cycle preview resolution quality");
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
                theme.font(FONT_SMALL),
                theme.value(),
            );
            if q_resp.clicked() {
                let idx = PreviewQuality::ALL
                    .iter()
                    .position(|&q| q == self.player_plus.preview_quality)
                    .unwrap_or(0);
                self.player_plus.preview_quality =
                    PreviewQuality::ALL[(idx + 1) % PreviewQuality::ALL.len()];
            }
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
        if self.suppress_shortcuts || ctx.egui_wants_keyboard_input() {
            return;
        }

        // Inside floating windows, only Undo / Redo / Save forward to CM3 (Spec 01 §4.8)
        let in_floating = self.panels.floating_windows.focused_id.is_some();
        let in_shader_editor =
            self.panels.floating_windows.focused_id.as_deref() == Some("shader_editor");

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
            } else if !ctrl && !shift && !alt && i.key_pressed(Key::Space) {
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
                let speed = session.playback.speed - 0.25;
                let _ = session.execute(Command::SetSpeed { speed });
            } else if !ctrl && !shift && !alt && i.key_pressed(Key::K) {
                let _ = session.execute(Command::SetSpeed { speed: 0.0 });
            } else if !ctrl
                && !shift
                && !alt
                && i.key_pressed(Key::L)
                && self.focused_panel != PanelFocus::Timeline
            {
                let speed = session.playback.speed + 0.25;
                let _ = session.execute(Command::SetSpeed { speed });
            } else if !ctrl
                && !shift
                && !alt
                && i.key_pressed(Key::C)
                && self.focused_panel != PanelFocus::Timeline
            {
                session.playback.toggle_editing_camera();
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
