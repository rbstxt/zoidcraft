//! HSVA color picker popover.

use crate::ui::theme::{
    BG_INPUT, BG_PANEL, BORDER_CONTROL, Cm3Icon, Cm3Theme, FONT_BODY, FONT_COMPACT, TEXT_BRIGHT,
    TEXT_MUTED, TEXT_PRIMARY, rgb_to_color32, rgba_to_color32,
};
use egui::{
    Area, Color32, CornerRadius, CursorIcon, Frame, Id, Key, Mesh, Order, Painter, Pos2, Rect,
    Sense, Shape, Stroke, TextEdit, Ui, UiBuilder, Vec2,
};
use zc_color::{Hsv, Rgb, Rgba};

use super::*;

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
pub struct ColorPopupState {
    pub open: bool,
    pub initial: Rgba,
    pub hsv: Hsv,
    pub alpha: f32,
    pub hex_buf: String,
    pub dirty: bool,
}

pub fn default_recent_swatches() -> Vec<Rgb> {
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

pub fn push_recent_swatch(ui: &mut Ui, rgb: Rgb) {
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

pub fn read_recent_swatches(ui: &mut Ui) -> Vec<Rgb> {
    let key = Id::unique("zoidcraft_recent_color_swatches_v2");
    ui.data_mut(|d| d.get_persisted::<Vec<Rgb>>(key))
        .filter(|v| v.len() == 36)
        .unwrap_or_else(default_recent_swatches)
}

const EYEDROPPER_LAST_SAMPLED_KEY: &str = "zc_eyedropper_last_sampled_color_v1";

/// Stores the most recently sampled eyedropper color in egui memory so any color picker
/// across panels or floating pro editors can reuse it immediately.
pub fn store_sampled_eyedropper_color(ctx: &egui::Context, color: Rgba) {
    ctx.data_mut(|d| {
        d.insert_temp(Id::unique(EYEDROPPER_LAST_SAMPLED_KEY), color);
    });
}

/// Reads the most recently sampled eyedropper color from shared egui memory.
#[must_use]
pub fn read_last_sampled_eyedropper_color(ctx: &egui::Context) -> Option<Rgba> {
    ctx.data(|d| d.get_temp::<Rgba>(Id::unique(EYEDROPPER_LAST_SAMPLED_KEY)))
}

/// Delivers a sampled color to a specific [`cm3_hsva_color_picker`] instance ID so it is applied
/// automatically on its next frame.
pub fn deliver_eyedropper_sample(ctx: &egui::Context, picker_id: Id, color: Rgba) {
    store_sampled_eyedropper_color(ctx, color);
    ctx.data_mut(|d| {
        d.insert_temp(picker_id.with("eyedropper_delivered_color"), color);
    });
}

/// Consumes any pending delivered eyedropper sample for `picker_id`.
pub fn take_eyedropper_sample(ctx: &egui::Context, picker_id: Id) -> Option<Rgba> {
    let slot = picker_id.with("eyedropper_delivered_color");
    ctx.data_mut(|d| {
        let val = d.get_temp::<Rgba>(slot);
        if val.is_some() {
            d.remove_temp::<Rgba>(slot);
        }
        val
    })
}

/// Opens the built-in interactive eyedropper sampler popover for `picker_id`.
/// Useful for callers that wire `ev.eyedropper_clicked` without viewport pixel capture.
pub fn request_eyedropper_sampler(ctx: &egui::Context, picker_id: Id) {
    ctx.data_mut(|d| {
        d.insert_temp(picker_id.with("eyedropper_sampler_open"), true);
    });
}

/// Marks that an external caller (e.g. viewport pixel eyedropper) already took ownership of the
/// eyedropper interaction for `picker_id` so the built-in fallback popover stays closed.
pub fn dismiss_eyedropper_sampler(ctx: &egui::Context, picker_id: Id) {
    ctx.data_mut(|d| {
        d.insert_temp(picker_id.with("eyedropper_sampler_open"), false);
    });
}

pub fn format_picker_css(c: Rgba, has_alpha: bool) -> String {
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

pub fn parse_picker_color(input: &str) -> Option<Rgba> {
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

pub fn paint_checkerboard(painter: &Painter, rect: Rect, cell: f32) {
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

pub fn paint_sv_square(painter: &Painter, rect: Rect, hue: f32) {
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

pub fn paint_hue_bar(painter: &Painter, rect: Rect) {
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

pub fn paint_alpha_bar(painter: &Painter, rect: Rect, rgb: Rgb) {
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
    let sampler_open_id = id.with("eyedropper_sampler_open");
    let mut sampler_open = ui
        .ctx()
        .data(|d| d.get_temp::<bool>(sampler_open_id))
        .unwrap_or(false);
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

    // Apply any externally delivered eyedropper sample for this picker ID.
    if !read_only && let Some(delivered) = take_eyedropper_sample(ui.ctx(), id) {
        *color = if has_alpha {
            delivered
        } else {
            delivered.rgb().with_alpha(1.0)
        };
        state.hsv = color.rgb().to_hsv();
        state.alpha = if has_alpha { color.a() } else { 1.0 };
        state.hex_buf = format_picker_css(*color, has_alpha);
        state.dirty = false;
        sampler_open = false;
        push_recent_swatch(ui, color.rgb());
        ev.changed = true;
        ev.committed = true;
    }

    // If read_only (e.g. expression toggled on) flips true while the popup is open,
    // immediately close, restore initial color if dirty, and emit cancelled so any open tx closes.
    if read_only && (state.open || sampler_open) {
        let was_dirty = state.dirty;
        state.open = false;
        sampler_open = false;
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
    let mut eyedropper_rect = Rect::NOTHING;
    let mut opened_sampler_this_frame = false;

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
                    sampler_open = false;
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
                let eye_origin = ui.cursor().min;
                eyedropper_rect = Rect::from_min_size(eye_origin, Vec2::splat(17.0));
                if cm3_icon_button(
                    ui,
                    Cm3Icon::Eyedropper,
                    17.0,
                    if sampler_open {
                        theme.accent()
                    } else {
                        TEXT_PRIMARY
                    },
                    "pick color from viewport or sampler",
                    theme,
                ) {
                    ev.eyedropper_clicked = true;
                    sampler_open = !sampler_open;
                    opened_sampler_this_frame = sampler_open;
                    if sampler_open {
                        state.open = false;
                    } else {
                        clear_transient_popover_open_if_stale(ui.ctx());
                    }
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

    // Built-in interactive eyedropper sampler popover (recent colors, active palette,
    // shared last-sampled color, and interactive continuous spectrum strip).
    if sampler_open && !read_only && swatch_rect != Rect::NOTHING {
        let sampler_area_id = id.with("eyedropper_sampler_area");
        let last_sampled = read_last_sampled_eyedropper_color(ui.ctx());
        let swatches = read_recent_swatches(ui);
        let has_last = last_sampled.is_some();
        let popup_size = Vec2::new(236.0, if has_last { 164.0 } else { 140.0 });
        let pos = clamp_popup_pos(swatch_rect, popup_size, viewport_rect);
        let mut picked_rgb: Option<Rgb> = None;
        let mut hovered_rgb: Option<Rgb> = None;
        let mut close_sampler = false;

        let area_resp = Area::new(sampler_area_id)
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
                        .id_salt(sampler_area_id.with("content"))
                        .max_rect(content_rect)
                        .layout(egui::Layout::top_down(egui::Align::LEFT)),
                );
                content_ui.spacing_mut().item_spacing = Vec2::new(4.0, 4.0);

                // Header: Title + live preview hex + close
                content_ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Eyedropper Sampler")
                            .font(theme.font(FONT_COMPACT))
                            .color(TEXT_BRIGHT),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("×").clicked() {
                            close_sampler = true;
                        }
                        let [r, g, b] = color.rgb().to_u8();
                        ui.label(
                            egui::RichText::new(format!("#{r:02x}{g:02x}{b:02x}"))
                                .font(theme.font(FONT_COMPACT))
                                .color(TEXT_MUTED),
                        );
                    });
                });

                // Optional shared last-sampled pill
                if let Some(last) = last_sampled {
                    let lrgb = last.rgb();
                    let (row_r, row_resp) = content_ui
                        .allocate_exact_size(Vec2::new(content_rect.width(), 18.0), Sense::click());
                    let sw_r = Rect::from_min_size(
                        Pos2::new(row_r.left(), row_r.top() + 2.0),
                        Vec2::new(26.0, 14.0),
                    );
                    content_ui.painter().rect_filled(
                        sw_r,
                        CornerRadius::ZERO,
                        rgb_to_color32(lrgb),
                    );
                    paint_crisp_border_1px(
                        content_ui.painter(),
                        sw_r,
                        if row_resp.hovered() {
                            TEXT_BRIGHT
                        } else {
                            BORDER_CONTROL
                        },
                    );
                    let [r, g, b] = lrgb.to_u8();
                    content_ui.painter().text(
                        Pos2::new(sw_r.right() + 6.0, row_r.center().y),
                        egui::Align2::LEFT_CENTER,
                        format!("Last sampled  #{r:02x}{g:02x}{b:02x}"),
                        theme.font(FONT_COMPACT),
                        if row_resp.hovered() {
                            TEXT_BRIGHT
                        } else {
                            TEXT_PRIMARY
                        },
                    );
                    let row_resp = row_resp.on_hover_cursor(CursorIcon::PointingHand);
                    if row_resp.hovered() {
                        hovered_rgb = Some(lrgb);
                    }
                    if row_resp.clicked() {
                        picked_rgb = Some(lrgb);
                    }
                }

                // Continuous horizontal spectrum strip (click or drag to sample any hue/brightness)
                let (strip_raw, strip_resp) = content_ui.allocate_exact_size(
                    Vec2::new(content_rect.width(), 22.0),
                    Sense::click_and_drag(),
                );
                let strip_rect = snap_rect_px(strip_raw);
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
                let seg_w = strip_rect.width() / 6.0;
                for i in 0..6 {
                    let x0 = strip_rect.left() + seg_w * i as f32;
                    let x1 = strip_rect.left() + seg_w * (i + 1) as f32;
                    let tl = mesh.vertices.len() as u32;
                    mesh.colored_vertex(Pos2::new(x0, strip_rect.top()), stops[i]);
                    let tr = mesh.vertices.len() as u32;
                    mesh.colored_vertex(Pos2::new(x1, strip_rect.top()), stops[i + 1]);
                    let br = mesh.vertices.len() as u32;
                    mesh.colored_vertex(Pos2::new(x1, strip_rect.bottom()), stops[i + 1]);
                    let bl = mesh.vertices.len() as u32;
                    mesh.colored_vertex(Pos2::new(x0, strip_rect.bottom()), stops[i]);
                    mesh.add_triangle(tl, tr, br);
                    mesh.add_triangle(tl, br, bl);
                }
                content_ui.painter().add(Shape::mesh(mesh));
                paint_crisp_border_1px(content_ui.painter(), strip_rect, BORDER_CONTROL);
                let strip_resp = strip_resp.on_hover_cursor(CursorIcon::Crosshair);
                if let Some(p) = strip_resp
                    .hover_pos()
                    .or_else(|| strip_resp.interact_pointer_pos())
                    && strip_rect.contains(p)
                {
                    let h = ((p.x - strip_rect.left()) / strip_rect.width().max(1.0))
                        .clamp(0.0, 0.9999);
                    let v = (1.0
                        - ((p.y - strip_rect.top()) / strip_rect.height().max(1.0)) * 0.45)
                        .clamp(0.2, 1.0);
                    let sampled = Rgb::from_hsv(Hsv { h, s: 0.9, v });
                    hovered_rgb = Some(sampled);
                    if strip_resp.clicked() || strip_resp.drag_stopped() {
                        picked_rgb = Some(sampled);
                    }
                }

                // Recent colors row (12 slots from swatches[24..36])
                content_ui.label(
                    egui::RichText::new("Recent & Active Palette")
                        .font(theme.font(FONT_COMPACT))
                        .color(TEXT_MUTED),
                );
                let draw_swatch_row =
                    |ui: &mut Ui,
                     row_tag: &str,
                     slice: &[Rgb],
                     hovered_rgb: &mut Option<Rgb>,
                     picked_rgb: &mut Option<Rgb>| {
                        let count = slice.len().max(1);
                        let gap = 2.0_f32;
                        let total_w = content_rect.width();
                        let cell_w =
                            ((total_w - gap * (count as f32 - 1.0)) / count as f32).max(10.0);
                        let (row_rect, _) =
                            ui.allocate_exact_size(Vec2::new(total_w, 18.0), Sense::hover());
                        for (idx, &sw_rgb) in slice.iter().enumerate() {
                            let x0 = (row_rect.left() + idx as f32 * (cell_w + gap)).round();
                            let s_rect = Rect::from_min_size(
                                Pos2::new(x0, row_rect.top()),
                                Vec2::new(cell_w.round().max(8.0), 18.0),
                            );
                            let s_resp = ui
                                .interact(
                                    s_rect,
                                    sampler_area_id.with((row_tag, idx)),
                                    Sense::click(),
                                )
                                .on_hover_cursor(CursorIcon::PointingHand);
                            ui.painter().rect_filled(
                                s_rect,
                                CornerRadius::ZERO,
                                rgb_to_color32(sw_rgb),
                            );
                            if s_resp.hovered() {
                                paint_crisp_border_1px(ui.painter(), s_rect, TEXT_BRIGHT);
                                *hovered_rgb = Some(sw_rgb);
                            }
                            if s_resp.clicked() {
                                *picked_rgb = Some(sw_rgb);
                            }
                        }
                    };
                if swatches.len() >= 36 {
                    draw_swatch_row(
                        &mut content_ui,
                        "recent",
                        &swatches[24..36],
                        &mut hovered_rgb,
                        &mut picked_rgb,
                    );
                    draw_swatch_row(
                        &mut content_ui,
                        "hues",
                        &swatches[0..12],
                        &mut hovered_rgb,
                        &mut picked_rgb,
                    );
                    draw_swatch_row(
                        &mut content_ui,
                        "grays",
                        &swatches[12..24],
                        &mut hovered_rgb,
                        &mut picked_rgb,
                    );
                }

                if let Some(hrgb) = hovered_rgb {
                    let chip_rect = Rect::from_min_size(
                        Pos2::new(outer_rect.right() - 78.0, outer_rect.top() + 7.0),
                        Vec2::new(12.0, 12.0),
                    );
                    ui.painter()
                        .rect_filled(chip_rect, CornerRadius::ZERO, rgb_to_color32(hrgb));
                    paint_crisp_border_1px(ui.painter(), chip_rect, TEXT_BRIGHT);
                }

                paint_crisp_border_1px(ui.painter(), outer_rect, theme.accent());
            });

        if let Some(rgb) = picked_rgb {
            state.hsv = rgb.to_hsv();
            *color = rgb.with_alpha(if has_alpha { state.alpha } else { 1.0 });
            state.hex_buf = format_picker_css(*color, has_alpha);
            state.dirty = false;
            sampler_open = false;
            push_recent_swatch(ui, rgb);
            store_sampled_eyedropper_color(ui.ctx(), *color);
            clear_transient_popover_open_if_stale(ui.ctx());
            ev.changed = true;
            ev.committed = true;
        } else {
            let mut cancel_sampler = false;
            ui.input(|i| {
                if i.key_pressed(Key::Escape) {
                    cancel_sampler = true;
                } else if !opened_sampler_this_frame
                    && i.pointer.primary_pressed()
                    && i.pointer.interact_pos().is_some_and(|p| {
                        !area_resp.response.rect.contains(p)
                            && !swatch_rect.contains(p)
                            && !eyedropper_rect.contains(p)
                    })
                {
                    close_sampler = true;
                }
            });
            if cancel_sampler {
                sampler_open = false;
                ev.cancelled = true;
                consume_transient_popover_escape(ui.ctx());
            } else if close_sampler {
                sampler_open = false;
                clear_transient_popover_open_if_stale(ui.ctx());
            }
        }
    }

    if (state.open || sampler_open) && !read_only {
        mark_transient_popover_open(ui.ctx());
    }

    ui.ctx()
        .data_mut(|d| d.insert_temp(sampler_open_id, sampler_open));
    ui.data_mut(|d| d.insert_temp(popup_id, state));
    ev
}
