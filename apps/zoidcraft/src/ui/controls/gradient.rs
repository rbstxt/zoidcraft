//! Gradient bar widget and gradient sampling helpers.

use crate::ui::theme::{BORDER_CONTROL, Cm3Theme, FONT_SMALL, TEXT_MUTED, rgba_to_color32};
use egui::{CornerRadius, Id, Mesh, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, Ui, Vec2};
use zc_color::Rgba;
use zc_project::GradientStop;

use super::*;

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
pub struct GradientBarState {
    pub selected_stop: usize,
    pub dragging_stop: Option<usize>,
    pub drag_dirty: bool,
}

pub fn parse_css_rgba(css: &str) -> Rgba {
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

pub fn format_css_rgba(c: Rgba) -> String {
    let [r, g, b, _] = c.to_u8();
    format!("rgba({r},{g},{b},{:.3})", c.a().clamp(0.0, 1.0))
}

pub fn sample_gradient(stops: &[GradientStop], t: f64) -> Rgba {
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
