use super::shared_widgets::{
    compute_asset_usage_counts_cached, paint_crisp_border_2px, panel_top_header,
};
use super::*;

// ---------------------------------------------------------------------------
// Tab 1: Media (project)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MediaCardIcon {
    Scene3D,
    Adjustment,
    Text,
    PresetShape,
    VectorShape,
    ShapePlus,
    Composite,
    Camera,
    Precomp,
}

struct MediaPreset {
    kind: &'static str,
    title: &'static str,
    subtitle: &'static str,
    icon: MediaCardIcon,
    required_plugin: Option<&'static str>,
}

const MEDIA_PRESETS: &[MediaPreset] = &[
    MediaPreset {
        kind: "scene",
        title: "3D Scene",
        subtitle: "3D camera, lights & meshes",
        icon: MediaCardIcon::Scene3D,
        required_plugin: None,
    },
    MediaPreset {
        kind: "adjustment",
        title: "Adjustment",
        subtitle: "Post-processing effect layer",
        icon: MediaCardIcon::Adjustment,
        required_plugin: None,
    },
    MediaPreset {
        kind: "text_2d",
        title: "Text",
        subtitle: "2D typographic layer",
        icon: MediaCardIcon::Text,
        required_plugin: None,
    },
    MediaPreset {
        kind: "preset_shape",
        title: "Preset shape",
        subtitle: "Parametric 2D preset",
        icon: MediaCardIcon::PresetShape,
        required_plugin: None,
    },
    MediaPreset {
        kind: "shape",
        title: "Shape",
        subtitle: "Custom vector shape",
        icon: MediaCardIcon::VectorShape,
        required_plugin: None,
    },
    MediaPreset {
        kind: "composite",
        title: "Composite",
        subtitle: "Nested layer composite",
        icon: MediaCardIcon::Composite,
        required_plugin: None,
    },
    MediaPreset {
        kind: "zoidium:camera-plus/camera-layer",
        title: "Camera Layer",
        subtitle: "Camera+ shared 3D camera",
        icon: MediaCardIcon::Camera,
        required_plugin: Some("camera-plus"),
    },
    MediaPreset {
        kind: "zoidium:shape-plus/shape-layer",
        title: "Shape+",
        subtitle: "Animated 2D vector layer",
        icon: MediaCardIcon::ShapePlus,
        required_plugin: Some("shape-plus"),
    },
    MediaPreset {
        kind: "zoidium:precomp-plus/precomp",
        title: "Precomp",
        subtitle: "Nested sequence composition",
        icon: MediaCardIcon::Precomp,
        required_plugin: Some("precomp-plus"),
    },
];

fn paint_media_card_icon(
    painter: &Painter,
    rect: Rect,
    icon: MediaCardIcon,
    color: Color32,
    card_bg: Color32,
) {
    let side = rect.width().min(rect.height()).max(1.0);
    let r = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let pt = |nx: f32, ny: f32| Pos2::new(r.left() + r.width() * nx, r.top() + r.height() * ny);
    match icon {
        MediaCardIcon::Scene3D => paint_cm3_icon(
            painter,
            Rect::from_center_size(rect.center(), Vec2::splat((side * 1.0625).round())),
            Cm3Icon::Objects,
            color,
        ),
        MediaCardIcon::Adjustment => paint_cm3_icon(
            painter,
            Rect::from_center_size(
                Pos2::new(rect.center().x, rect.center().y + 0.5),
                Vec2::splat((side * 1.104).round()),
            ),
            Cm3Icon::Fx,
            color,
        ),
        MediaCardIcon::Camera => paint_cm3_icon(painter, r, Cm3Icon::Camera, color),
        MediaCardIcon::Precomp => paint_cm3_icon(painter, r, Cm3Icon::Sequence, color),
        MediaCardIcon::Text => {
            // Bold geometric "T" glyph matching Zoidium Media card (39x39 inside 48x48)
            let bar = Rect::from_min_max(pt(0.10, 0.10), pt(0.90, 0.28));
            let stem = Rect::from_min_max(pt(0.40, 0.26), pt(0.60, 0.90));
            painter.rect_filled(bar, CornerRadius::ZERO, color);
            painter.rect_filled(stem, CornerRadius::ZERO, color);
        }
        MediaCardIcon::PresetShape => {
            // Geometric triangle overlapping a circle for parametric Preset shapes
            painter.circle_filled(pt(0.64, 0.64), side * 0.28, color);
            let cutout = vec![pt(0.36, 0.06), pt(0.75, 0.82), pt(-0.02, 0.82)];
            painter.add(Shape::convex_polygon(
                cutout,
                card_bg,
                Stroke::new(2.0, card_bg),
            ));
            let tri = vec![pt(0.36, 0.12), pt(0.70, 0.78), pt(0.02, 0.78)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
        MediaCardIcon::VectorShape => {
            // Bezier path + anchor nodes + tangent handles for custom vector Shape
            let sw = (side * 0.065).clamp(1.8, 3.2);
            let mut curve_pts = Vec::with_capacity(17);
            let p0 = pt(0.14, 0.78);
            let p1 = pt(0.16, 0.18);
            let p2 = pt(0.84, 0.82);
            let p3 = pt(0.86, 0.22);
            for i in 0..=16 {
                let t = i as f32 / 16.0;
                let u = 1.0 - t;
                let x = u * u * u * p0.x
                    + 3.0 * u * u * t * p1.x
                    + 3.0 * u * t * t * p2.x
                    + t * t * t * p3.x;
                let y = u * u * u * p0.y
                    + 3.0 * u * u * t * p1.y
                    + 3.0 * u * t * t * p2.y
                    + t * t * t * p3.y;
                curve_pts.push(Pos2::new(x, y));
            }
            painter.add(Shape::line(curve_pts, Stroke::new(sw, color)));
            // Tangent handles
            let handle_stroke = Stroke::new((sw * 0.55).max(1.2), color32_with_alpha(color, 165));
            painter.line_segment([p0, pt(0.18, 0.34)], handle_stroke);
            painter.line_segment([p3, pt(0.82, 0.66)], handle_stroke);
            painter.circle_filled(pt(0.18, 0.34), side * 0.055, color);
            painter.circle_filled(pt(0.82, 0.66), side * 0.055, color);
            // Square bezier anchor nodes
            for anchor in [p0, pt(0.50, 0.50), p3] {
                let box_r = Rect::from_center_size(anchor, Vec2::splat(side * 0.16));
                painter.rect_filled(box_r, CornerRadius::ZERO, card_bg);
                painter.rect_stroke(
                    box_r,
                    CornerRadius::ZERO,
                    Stroke::new(sw * 0.8, color),
                    StrokeKind::Inside,
                );
            }
        }
        MediaCardIcon::ShapePlus => {
            // Stacked rounded vector shapes with a crisp '+' badge in top-right for Shape+
            let sw = (side * 0.06).clamp(1.6, 3.0);
            let back = Rect::from_min_max(pt(0.10, 0.24), pt(0.62, 0.76));
            painter.rect_stroke(
                back,
                CornerRadius::same(4),
                Stroke::new(sw, color32_with_alpha(color, 180)),
                StrokeKind::Inside,
            );
            let front = Rect::from_min_max(pt(0.24, 0.38), pt(0.76, 0.90));
            painter.rect_filled(front, CornerRadius::same(4), card_bg);
            painter.rect_filled(front.shrink(sw * 0.6), CornerRadius::same(3), color);
            // Crisp '+' badge at upper-right
            let plus_c = pt(0.78, 0.22);
            let arm = side * 0.14;
            let thick = (side * 0.07).clamp(2.0, 3.5);
            painter.rect_filled(
                Rect::from_center_size(plus_c, Vec2::new(arm * 2.0, thick)),
                CornerRadius::ZERO,
                color,
            );
            painter.rect_filled(
                Rect::from_center_size(plus_c, Vec2::new(thick, arm * 2.0)),
                CornerRadius::ZERO,
                color,
            );
        }
        MediaCardIcon::Composite => {
            // 3 stacked isometric diamonds (54x48 px at card scale) matching Zoidium Composite card
            let scale = side / 48.0;
            let cr = Rect::from_center_size(rect.center(), Vec2::new(54.0 * scale, 48.0 * scale));
            let cpt = |nx: f32, ny: f32| {
                Pos2::new(cr.left() + cr.width() * nx, cr.top() + cr.height() * ny)
            };
            let sw = (5.0 * scale).clamp(2.4, 5.5);
            let top = vec![
                cpt(0.50, 0.04),
                cpt(0.98, 0.30),
                cpt(0.50, 0.56),
                cpt(0.02, 0.30),
            ];
            painter.add(Shape::convex_polygon(top, color, Stroke::NONE));
            let mid = vec![cpt(0.04, 0.48), cpt(0.50, 0.74), cpt(0.96, 0.48)];
            painter.add(Shape::line(mid, Stroke::new(sw, color)));
            let bot = vec![cpt(0.04, 0.68), cpt(0.50, 0.94), cpt(0.96, 0.68)];
            painter.add(Shape::line(bot, Stroke::new(sw, color)));
        }
    }
}

fn paint_audio_media_icon(painter: &Painter, rect: Rect, color: Color32) {
    let side = rect.width().min(rect.height()).max(1.0);
    let r = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let pt = |nx: f32, ny: f32| Pos2::new(r.left() + r.width() * nx, r.top() + r.height() * ny);
    let sw = (side * 0.065).clamp(1.6, 3.0);

    // Speaker body + cone on the left
    let body = Rect::from_min_max(pt(0.10, 0.36), pt(0.28, 0.64));
    painter.rect_filled(body, CornerRadius::same(1), color);
    let cone = vec![
        pt(0.26, 0.36),
        pt(0.50, 0.16),
        pt(0.50, 0.84),
        pt(0.26, 0.64),
    ];
    painter.add(Shape::convex_polygon(cone, color, Stroke::NONE));

    // Crisp 3-bar soundwave / equalizer bars on the right
    let bars = [(0.62, 0.34, 0.66), (0.74, 0.22, 0.78), (0.86, 0.30, 0.70)];
    for (x, y0, y1) in bars {
        painter.line_segment([pt(x, y0), pt(x, y1)], Stroke::new(sw, color));
    }
}

/// Paints a crisp landscape photo-frame icon (outer border, sun, and twin mountain peaks)
/// for imported image assets (`.png`/`.jpg`/`.webp`/etc.) in Project Media.
fn paint_image_media_icon(painter: &Painter, rect: Rect, color: Color32, card_bg: Color32) {
    let side = rect.width().min(rect.height()).max(1.0);
    let r = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let pt = |nx: f32, ny: f32| Pos2::new(r.left() + r.width() * nx, r.top() + r.height() * ny);
    let sw = (side * 0.06).clamp(1.6, 2.8);

    // Outer landscape picture frame
    let frame = Rect::from_min_max(pt(0.08, 0.16), pt(0.92, 0.84));
    painter.rect_filled(frame, CornerRadius::same(2), card_bg);
    painter.rect_stroke(
        frame,
        CornerRadius::same(2),
        Stroke::new(sw, color),
        StrokeKind::Inside,
    );

    // Sun disc in upper-left
    painter.circle_filled(pt(0.30, 0.36), (side * 0.085).max(1.8), color);

    // Back mountain peak (right-center)
    let back_peak = vec![pt(0.42, 0.78), pt(0.67, 0.40), pt(0.87, 0.78)];
    painter.add(Shape::convex_polygon(
        back_peak,
        color32_with_alpha(color, 185),
        Stroke::NONE,
    ));

    // Front mountain peak (left-center)
    let front_peak = vec![pt(0.13, 0.78), pt(0.39, 0.48), pt(0.62, 0.78)];
    painter.add(Shape::convex_polygon(front_peak, color, Stroke::NONE));
}

pub(super) fn layout_single_line_ellipsized(
    painter: &Painter,
    text: &str,
    font: egui::FontId,
    color: Color32,
    max_width: f32,
) -> std::sync::Arc<egui::Galley> {
    let full = painter.layout_no_wrap(text.to_owned(), font.clone(), color);
    if full.size().x <= max_width || max_width <= 8.0 {
        return full;
    }
    let chars: Vec<char> = text.chars().collect();
    let mut len = chars.len();
    let mut chosen = full;
    while len > 0 {
        len -= 1;
        let cand = format!("{}…", chars[..len].iter().collect::<String>().trim_end());
        let g = painter.layout_no_wrap(cand, font.clone(), color);
        if g.size().x <= max_width || len == 0 {
            chosen = g;
            break;
        }
    }
    chosen
}

pub(super) fn clip_kind_for_asset(kind: AssetKind, filename: &str) -> &'static str {
    let ext = filename
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match kind {
        AssetKind::Image => "image",
        AssetKind::Av => {
            if matches!(ext.as_str(), "mp4" | "webm" | "mkv" | "mov" | "avi") {
                "video"
            } else {
                "audio"
            }
        }
        _ => match ext.as_str() {
            "mp4" | "webm" | "mkv" | "mov" | "avi" => "video",
            "mp3" | "wav" | "ogg" | "flac" | "m4a" | "aac" | "opus" => "audio",
            _ => "image",
        },
    }
}

struct MediaLibraryEntry {
    name: String,
    asset_id: Option<String>,
    clip_kind: &'static str,
    missing: bool,
}

enum MediaGridItem<'a> {
    Preset(&'a MediaPreset),
    Imported(MediaLibraryEntry),
}

pub(super) fn show_media_tab(
    ui: &mut Ui,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    ScrollArea::vertical()
        .id_salt("media_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;

            // Top search box: 34px tall, 4px left/right margin, 5px top/bottom margin,
            // #2a2a2a fill, crisp 2px solid #363636 border, single-frame (Frame::NONE)
            Frame::NONE
                .inner_margin(egui::Margin {
                    left: 4,
                    right: 4,
                    top: 5,
                    bottom: 5,
                })
                .show(ui, |ui| {
                    let search_id = Id::unique("media_search_box");
                    let focused = ui.memory(|m| m.has_focus(search_id));
                    let box_w = ui.available_width();
                    ui.allocate_ui_with_layout(
                        Vec2::new(box_w, 34.0),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            let box_rect =
                                Rect::from_min_size(ui.cursor().min, Vec2::new(box_w, 34.0));
                            ui.painter()
                                .rect_filled(box_rect, CornerRadius::ZERO, BG_PANEL);
                            paint_crisp_border_2px(
                                ui.painter(),
                                box_rect,
                                if focused {
                                    theme.accent()
                                } else {
                                    BORDER_FILTER
                                },
                            );
                            Frame::NONE
                                .inner_margin(egui::Margin::symmetric(7, 6))
                                .show(ui, |ui| {
                                    ui.add(
                                        TextEdit::singleline(&mut state.media_search)
                                            .id(search_id)
                                            .frame(Frame::NONE)
                                            .font(theme.font(FONT_BODY))
                                            .text_color(TEXT_PRIMARY)
                                            .desired_width((box_w - 14.0).max(40.0)),
                                    );
                                });
                            if state.media_search.is_empty() {
                                ui.painter().text(
                                    Pos2::new(box_rect.left() + 8.0, box_rect.center().y),
                                    Align2::LEFT_CENTER,
                                    "search media",
                                    theme.font(FONT_BODY),
                                    Color32::from_rgb(0x75, 0x75, 0x75),
                                );
                            }
                        },
                    );
                });

            // Precomp+ Compositions manager (Main + nested compositions)
            if session.plugins.is_enabled("precomp-plus")
                || !session.project.compositions.is_empty()
            {
                show_compositions_manager_section(ui, session, theme, state);
            }

            // 100% edge-to-edge 30px #222222 "Project media" bar with borderless "+ Import" inside
            let (bar_rect, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 30.0), Sense::hover());
            ui.painter()
                .rect_filled(bar_rect, CornerRadius::ZERO, BG_TITLE);
            let import_galley = ui.painter().layout_no_wrap(
                "Import".to_owned(),
                theme.font(FONT_BODY),
                TEXT_PRIMARY,
            );
            let plus_size = 16.0;
            let plus_gap = 7.0;
            let right_pad = 8.0;
            let import_w = plus_size + plus_gap + import_galley.size().x + right_pad + 8.0;
            let import_rect = Rect::from_min_max(
                Pos2::new(
                    (bar_rect.right() - import_w).max(bar_rect.left()),
                    bar_rect.top(),
                ),
                bar_rect.right_bottom(),
            );
            // Keep title cleanly clipped/shortened before "+ Import" so it never overlaps at 900x600
            let title_left = bar_rect.left() + 5.0;
            let title_max_w = (import_rect.left() - title_left - 6.0).max(16.0);
            let full_title_galley = ui.painter().layout_no_wrap(
                "Project media".to_owned(),
                theme.font(FONT_BODY),
                TEXT_PRIMARY,
            );
            let title_galley = if full_title_galley.size().x <= title_max_w {
                full_title_galley
            } else {
                let short_galley = ui.painter().layout_no_wrap(
                    "Media".to_owned(),
                    theme.font(FONT_BODY),
                    TEXT_PRIMARY,
                );
                if short_galley.size().x <= title_max_w {
                    short_galley
                } else {
                    let chars: Vec<char> = "Media".chars().collect();
                    let mut len = chars.len();
                    let mut chosen = short_galley;
                    while len > 0 {
                        len -= 1;
                        let cand = format!("{}…", chars[..len].iter().collect::<String>());
                        let g =
                            ui.painter()
                                .layout_no_wrap(cand, theme.font(FONT_BODY), TEXT_PRIMARY);
                        if g.size().x <= title_max_w || len == 0 {
                            chosen = g;
                            break;
                        }
                    }
                    chosen
                }
            };
            let title_clip = Rect::from_min_max(
                Pos2::new(title_left, bar_rect.top()),
                Pos2::new(
                    (import_rect.left() - 4.0).max(title_left),
                    bar_rect.bottom(),
                ),
            );
            ui.painter().with_clip_rect(title_clip).galley(
                Pos2::new(
                    title_left,
                    bar_rect.center().y + 1.0 - title_galley.size().y * 0.5,
                ),
                title_galley,
                TEXT_PRIMARY,
            );
            let import_resp = ui
                .interact(import_rect, Id::unique("media_import_btn"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text("Import image, video, audio, or font media");
            if import_resp.hovered() {
                ui.painter()
                    .rect_filled(import_rect, CornerRadius::ZERO, BG_HOVER);
            }
            let import_color = if import_resp.hovered() {
                TEXT_BRIGHT
            } else {
                TOOLBAR_ICON_INACTIVE
            };
            let label_right = bar_rect.right() - right_pad;
            let label_left = label_right - import_galley.size().x;
            ui.painter().text(
                Pos2::new(label_right, bar_rect.center().y + 1.0),
                Align2::RIGHT_CENTER,
                "Import",
                theme.font(FONT_BODY),
                import_color,
            );
            let plus_cx = (label_left - plus_gap - plus_size * 0.5).round();
            let plus_cy = (bar_rect.center().y + 1.0).round();
            ui.painter().rect_filled(
                Rect::from_min_size(
                    Pos2::new(plus_cx - 8.0, plus_cy - 1.0),
                    Vec2::new(16.0, 2.0),
                ),
                CornerRadius::ZERO,
                import_color,
            );
            ui.painter().rect_filled(
                Rect::from_min_size(
                    Pos2::new(plus_cx - 1.0, plus_cy - 8.0),
                    Vec2::new(2.0, 16.0),
                ),
                CornerRadius::ZERO,
                import_color,
            );
            if import_resp.clicked() {
                actions.push(PanelAction::ImportMedia);
            }

            // Build unified list: built-in presets + imported media in the same 2-column grid
            let q = state.media_search.trim().to_ascii_lowercase();
            let mut items: Vec<MediaGridItem<'_>> = MEDIA_PRESETS
                .iter()
                .filter(|p| {
                    p.required_plugin
                        .is_none_or(|pid| session.plugins.is_enabled(pid))
                })
                .filter(|p| {
                    q.is_empty()
                        || p.title.to_ascii_lowercase().contains(&q)
                        || p.subtitle.to_ascii_lowercase().contains(&q)
                })
                .map(MediaGridItem::Preset)
                .collect();

            let mut seen_asset_ids: BTreeSet<String> = BTreeSet::new();
            let mut seen_names: BTreeSet<String> = BTreeSet::new();

            for m in &session.project.media {
                let asset_rec = m
                    .asset_id
                    .as_deref()
                    .and_then(|aid| session.project.assets.get(aid));
                if let Some(aid) = &m.asset_id {
                    seen_asset_ids.insert(aid.clone());
                }
                seen_names.insert(m.name.to_ascii_lowercase());
                let filename = asset_rec
                    .map(|a| a.filename.as_str())
                    .unwrap_or(m.name.as_str());
                let kind = asset_rec.map_or(m.kind, |a| a.kind);
                let clip_kind = clip_kind_for_asset(kind, filename);
                if !q.is_empty()
                    && !m.name.to_ascii_lowercase().contains(&q)
                    && !clip_kind.contains(&q)
                {
                    continue;
                }
                items.push(MediaGridItem::Imported(MediaLibraryEntry {
                    name: m.name.clone(),
                    asset_id: m.asset_id.clone(),
                    clip_kind,
                    missing: asset_rec.is_some_and(|a| a.missing),
                }));
            }

            for a in session.project.assets.values() {
                if seen_asset_ids.contains(&a.id)
                    || seen_names.contains(&a.filename.to_ascii_lowercase())
                {
                    continue;
                }
                let clip_kind = clip_kind_for_asset(a.kind, &a.filename);
                if !q.is_empty()
                    && !a.filename.to_ascii_lowercase().contains(&q)
                    && !clip_kind.contains(&q)
                {
                    continue;
                }
                items.push(MediaGridItem::Imported(MediaLibraryEntry {
                    name: a.filename.clone(),
                    asset_id: Some(a.id.clone()),
                    clip_kind,
                    missing: a.missing,
                }));
            }

            // 2-column grid: 4px outer side margin, 3px gap, ~155x111 flat #333333 borderless cards
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(4, 4))
                .show(ui, |ui| {
                    let cols = if ui.available_width() >= 140.0 { 2 } else { 1 };
                    let gap = 3.0;
                    let card_w =
                        ((ui.available_width() - gap * (cols - 1) as f32) / cols as f32).max(72.0);
                    let card_h = 111.5;
                    let card_hover_bg = Color32::from_rgb(0x3d, 0x3d, 0x3d);

                    for chunk in items.chunks(cols) {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = gap;
                            for item in chunk {
                                let (rect, resp) = ui
                                    .allocate_exact_size(Vec2::new(card_w, card_h), Sense::click());
                                let bg = if resp.hovered() {
                                    card_hover_bg
                                } else {
                                    BG_HOVER
                                };
                                ui.painter().rect_filled(rect, CornerRadius::ZERO, bg);

                                let narrow_card = card_w < 96.0;
                                let icon_rect = Rect::from_center_size(
                                    Pos2::new(
                                        rect.center().x,
                                        rect.top() + if narrow_card { 42.0 } else { 45.5 },
                                    ),
                                    Vec2::splat(if narrow_card { 40.0 } else { 48.0 }),
                                );
                                let label_clip = Rect::from_min_max(
                                    Pos2::new(
                                        rect.left() + 5.0,
                                        rect.bottom() - if narrow_card { 24.0 } else { 26.0 },
                                    ),
                                    Pos2::new(rect.right() - 5.0, rect.bottom() - 3.0),
                                );
                                let card_font =
                                    theme.font(if narrow_card { FONT_COMPACT } else { FONT_BODY });

                                match item {
                                    MediaGridItem::Preset(preset) => {
                                        let resp = resp
                                            .on_hover_cursor(CursorIcon::PointingHand)
                                            .on_hover_text(format!(
                                                "{} — {}",
                                                preset.title, preset.subtitle
                                            ));
                                        paint_media_card_icon(
                                            ui.painter(),
                                            icon_rect,
                                            preset.icon,
                                            TEXT_PRIMARY,
                                            bg,
                                        );
                                        let galley = layout_single_line_ellipsized(
                                            ui.painter(),
                                            preset.title,
                                            card_font.clone(),
                                            TEXT_PRIMARY,
                                            label_clip.width().max(24.0),
                                        );
                                        let label_y = label_clip.center().y - galley.size().y * 0.5;
                                        ui.painter().with_clip_rect(label_clip).galley(
                                            Pos2::new(label_clip.left(), label_y),
                                            galley,
                                            TEXT_PRIMARY,
                                        );
                                        if resp.clicked() {
                                            add_preset_clip(session, preset.kind, preset.title);
                                        }
                                    }
                                    MediaGridItem::Imported(entry) => {
                                        let resp = resp
                                            .on_hover_cursor(CursorIcon::PointingHand)
                                            .on_hover_text(format!(
                                                "{} — click to add {} clip to timeline",
                                                entry.name, entry.clip_kind
                                            ));
                                        let text_col = if entry.missing {
                                            DANGER_TEXT
                                        } else {
                                            TEXT_PRIMARY
                                        };
                                        if entry.clip_kind == "audio" {
                                            paint_audio_media_icon(
                                                ui.painter(),
                                                icon_rect,
                                                text_col,
                                            );
                                        } else if entry.clip_kind == "image" {
                                            paint_image_media_icon(
                                                ui.painter(),
                                                icon_rect,
                                                text_col,
                                                bg,
                                            );
                                        } else {
                                            let icon = match entry.clip_kind {
                                                "video" => Cm3Icon::Sequence,
                                                _ => Cm3Icon::Project,
                                            };
                                            paint_cm3_icon(ui.painter(), icon_rect, icon, text_col);
                                        }
                                        let galley = layout_single_line_ellipsized(
                                            ui.painter(),
                                            &entry.name,
                                            card_font.clone(),
                                            text_col,
                                            label_clip.width().max(24.0),
                                        );
                                        let label_y = label_clip.center().y - galley.size().y * 0.5;
                                        ui.painter().with_clip_rect(label_clip).galley(
                                            Pos2::new(label_clip.left(), label_y),
                                            galley,
                                            text_col,
                                        );
                                        if resp.clicked() {
                                            add_clip_with_asset(
                                                session,
                                                entry.clip_kind,
                                                &entry.name,
                                                entry.asset_id.as_deref(),
                                            );
                                        }
                                    }
                                }
                            }
                        });
                        ui.add_space(gap);
                    }
                });
        });
}

pub(super) fn add_preset_clip(session: &mut Session, kind: &str, title: &str) {
    add_clip_with_asset(session, kind, title, None);
}

fn add_clip_with_asset(session: &mut Session, kind: &str, title: &str, asset_id: Option<&str>) {
    let is_audio = kind == "audio";
    let desired_start = session.playback.current_frame.max(0.0).round();
    let length = session.active_sequence().length_frames.clamp(1.0, 150.0);

    // Place the new clip at the current playhead (`desired_start`) on the first unlocked
    // matching track with no overlapping clip in `[desired_start, desired_start + length)`.
    // If every existing unlocked track overlaps at `desired_start`, create a new track above
    // via `Command::AddVideoTrack` / `Command::AddAudioTrack` so the clip is immediately
    // visible in both the viewport and timeline instead of pushed off-screen!
    let chosen_track_id: Option<String> = {
        let tracks = if is_audio {
            &session.document().audio_tracks
        } else {
            &session.document().video_tracks
        };
        tracks
            .iter()
            .find(|t| {
                !t.locked
                    && !t.clips.iter().any(|c| {
                        c.start < desired_start + length - 1e-6
                            && desired_start < c.start + c.length - 1e-6
                    })
            })
            .map(|t| t.id.clone())
    };

    let started_tx = session.begin_transaction(format!("add {title}")).is_ok();

    let (track_id, start) = if let Some(tid) = chosen_track_id {
        (tid, desired_start)
    } else {
        let create_cmd = if is_audio {
            Command::AddAudioTrack { name: None }
        } else {
            Command::AddVideoTrack { name: None }
        };
        if let Ok(res) = session.execute(create_cmd)
            && let Some(tid) = res.get("track_id").and_then(Value::as_str)
        {
            (tid.to_owned(), desired_start)
        } else {
            if started_tx && session.history.is_in_transaction() {
                let _ = session.cancel_transaction();
            }
            return;
        }
    };

    if let Ok(res) = session.execute(Command::AddClip {
        track_id,
        kind: kind.to_owned(),
        name: Some(title.to_owned()),
        start,
        length,
    }) && let Some(cid) = res.get("clip_id").and_then(Value::as_str)
    {
        if let Some(aid) = asset_id {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: cid.to_owned(),
                property: "assetId".to_owned(),
                value: json!(aid),
            });
        }

        // Seed default Camera + Spot Light + 3D Text inside new 3D Scene clips so they
        // are immediately visible in the viewport.
        if kind == "scene" {
            let _ = session.execute(Command::AddObject3D {
                clip_id: cid.to_owned(),
                parent_id: None,
                kind: "camera".to_owned(),
                name: Some("Camera".to_owned()),
            });
            if let Ok(light_res) = session.execute(Command::AddObject3D {
                clip_id: cid.to_owned(),
                parent_id: None,
                kind: "spot_light".to_owned(),
                name: Some("Spot Light".to_owned()),
            }) && let Some(lid) = light_res.get("object_id").and_then(Value::as_str)
            {
                let _ = session.execute(Command::SetPropertyValue {
                    target_id: lid.to_owned(),
                    property: "position".to_owned(),
                    value: json!([0.0, 10.0, 0.0]),
                });
            }
            let _ = session.execute(Command::AddObject3D {
                clip_id: cid.to_owned(),
                parent_id: None,
                kind: "text".to_owned(),
                name: Some("Text".to_owned()),
            });
        }

        let _ = session.selection.select_clip(cid, Some(kind), false);
    }

    if started_tx && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    }
}

// ---------------------------------------------------------------------------
// 1. Precomp+ Compositions Manager (Media Tab)
// ---------------------------------------------------------------------------

fn show_compositions_manager_section(
    ui: &mut Ui,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
) {
    let active_comp = session.document().active_composition_id;
    let mut comps: Vec<(Uuid, String, [u32; 2], u32)> = session
        .project
        .compositions
        .iter()
        .map(|(id, seq)| (*id, seq.name.clone(), seq.resolution, seq.fps))
        .collect();
    comps.sort_by(|a, b| a.1.cmp(&b.1));

    let (add_comp, _) = cm3_collapsible_section_with_action(
        ui,
        Id::unique("media_compositions_section"),
        "Compositions",
        true,
        false,
        state.prop_columns.label_ratio,
        Some((Cm3Icon::Add, "New composition")),
        theme,
        |ui| {
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(6, 4))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 4.0;
                    // Breadcrumb / Back to Main + Pre-compose Selected actions
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        if active_comp.is_some()
                            && cm3_prop_button(
                                ui,
                                "← Back to Main",
                                ButtonVariant::Primary,
                                false,
                                theme,
                            )
                            .clicked()
                        {
                            let _ =
                                session.execute(Command::SetActiveComposition { comp_id: None });
                        }
                        let selected_clip_uuids: Vec<Uuid> = session
                            .selection
                            .clip_ids
                            .iter()
                            .filter_map(|s| Uuid::parse_str(s).ok())
                            .collect();
                        if cm3_prop_button(
                            ui,
                            "Pre-compose Selected",
                            ButtonVariant::Normal,
                            selected_clip_uuids.is_empty(),
                            theme,
                        )
                        .clicked()
                        {
                            let name =
                                format!("Precomp {}", session.project.compositions.len() + 1);
                            let _ = session.execute(Command::PrecomposeClips {
                                clip_ids: selected_clip_uuids,
                                name,
                            });
                        }
                    });

                    // Main composition row
                    let main_active = active_comp.is_none();
                    ui.horizontal(|ui| {
                        let lbl = format!(
                            "Main ({}×{} @ {}fps)",
                            session.project.sequence.resolution[0],
                            session.project.sequence.resolution[1],
                            session.project.sequence.fps
                        );
                        if ui
                            .selectable_label(
                                main_active,
                                egui::RichText::new(lbl).font(theme.font(FONT_SMALL)).color(
                                    if main_active {
                                        TEXT_BRIGHT
                                    } else {
                                        TEXT_PRIMARY
                                    },
                                ),
                            )
                            .clicked()
                            && !main_active
                        {
                            let _ =
                                session.execute(Command::SetActiveComposition { comp_id: None });
                        }
                    });

                    // Nested compositions
                    for (cid, cname, cres, cfps) in &comps {
                        let is_active = active_comp == Some(*cid);
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 3.0;
                            if state.renaming_comp_id == Some(*cid) {
                                let resp = ui.add(
                                    TextEdit::singleline(&mut state.renaming_comp_buf)
                                        .desired_width(110.0)
                                        .font(theme.font(FONT_SMALL)),
                                );
                                if ui.small_button("OK").clicked()
                                    || (resp.lost_focus()
                                        && ui.input(|i| i.key_pressed(Key::Enter)))
                                {
                                    let trimmed = state.renaming_comp_buf.trim().to_owned();
                                    if !trimmed.is_empty() {
                                        let _ = session.execute(Command::RenameComposition {
                                            comp_id: *cid,
                                            name: trimmed,
                                        });
                                    }
                                    state.renaming_comp_id = None;
                                }
                            } else {
                                let lbl = format!("{cname} ({}×{} @ {cfps}fps)", cres[0], cres[1]);
                                let actions_w = 58.0_f32;
                                let label_w = (ui.available_width() - actions_w - 6.0).max(48.0);
                                ui.allocate_ui_with_layout(
                                    Vec2::new(label_w, 18.0),
                                    egui::Layout::left_to_right(egui::Align::Center),
                                    |ui| {
                                        let resp = ui
                                            .add(
                                                egui::Button::new(
                                                    egui::RichText::new(&lbl)
                                                        .font(theme.font(FONT_SMALL))
                                                        .color(if is_active {
                                                            TEXT_BRIGHT
                                                        } else {
                                                            TEXT_PRIMARY
                                                        }),
                                                )
                                                .fill(if is_active {
                                                    theme.accent()
                                                } else {
                                                    Color32::TRANSPARENT
                                                })
                                                .stroke(Stroke::NONE)
                                                .truncate(),
                                            )
                                            .on_hover_text(&lbl);
                                        if resp.clicked() {
                                            let _ =
                                                session.execute(Command::SetActiveComposition {
                                                    comp_id: Some(*cid),
                                                });
                                        }
                                    },
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if cm3_icon_button(
                                            ui,
                                            Cm3Icon::Remove,
                                            13.0,
                                            TEXT_MUTED,
                                            "Delete comp",
                                            theme,
                                        ) {
                                            let _ = session.execute(Command::DeleteComposition {
                                                comp_id: *cid,
                                            });
                                        }
                                        if cm3_icon_button(
                                            ui,
                                            Cm3Icon::Duplicate,
                                            13.0,
                                            TEXT_MUTED,
                                            "Duplicate comp",
                                            theme,
                                        ) {
                                            let _ =
                                                session.execute(Command::DuplicateComposition {
                                                    comp_id: *cid,
                                                });
                                        }
                                        if cm3_icon_button(
                                            ui,
                                            Cm3Icon::Edit2,
                                            13.0,
                                            TEXT_MUTED,
                                            "Rename comp",
                                            theme,
                                        ) {
                                            state.renaming_comp_id = Some(*cid);
                                            state.renaming_comp_buf = cname.clone();
                                        }
                                    },
                                );
                            }
                        });
                    }
                });
        },
    );
    if add_comp {
        let idx = session.project.compositions.len() + 1;
        let name = format!("{} {idx}", state.new_comp_name.trim());
        let res = session.project.sequence.resolution;
        let fps = session.project.sequence.fps;
        let len = (session.project.sequence.length_frames.round() as u32).max(150);
        let _ = session.execute(Command::CreateComposition {
            name,
            resolution: res,
            fps,
            length_frames: len,
        });
    }
}

// ---------------------------------------------------------------------------
// Tab 9: Assets — Project Assets & Starter Templates
// ---------------------------------------------------------------------------

pub(super) fn show_assets_tab(
    ui: &mut Ui,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    ScrollArea::vertical()
        .id_salt("assets_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;
            panel_top_header(ui, "Assets", theme);

            // Single-frame "type to filter" search box (34px tall, 4px side margin, 2px #363636)
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(4, 6))
                .show(ui, |ui| {
                    let filter_id = Id::unique("assets_filter_box");
                    let focused = ui.memory(|m| m.has_focus(filter_id));
                    let box_w = ui.available_width();
                    ui.allocate_ui_with_layout(
                        Vec2::new(box_w, 34.0),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            let box_rect =
                                Rect::from_min_size(ui.cursor().min, Vec2::new(box_w, 34.0));
                            ui.painter()
                                .rect_filled(box_rect, CornerRadius::ZERO, BG_PANEL);
                            ui.painter().rect_stroke(
                                box_rect,
                                CornerRadius::ZERO,
                                Stroke::new(
                                    2.0,
                                    if focused {
                                        theme.accent()
                                    } else {
                                        BORDER_FILTER
                                    },
                                ),
                                StrokeKind::Inside,
                            );
                            Frame::NONE
                                .inner_margin(egui::Margin::symmetric(8, 6))
                                .show(ui, |ui| {
                                    ui.add(
                                        TextEdit::singleline(&mut state.assets_filter)
                                            .id(filter_id)
                                            .frame(Frame::NONE)
                                            .hint_text(
                                                egui::RichText::new("type to filter")
                                                    .color(TEXT_EMPTY),
                                            )
                                            .font(theme.font(FONT_BODY))
                                            .text_color(TEXT_PRIMARY)
                                            .desired_width((box_w - 16.0).max(40.0)),
                                    );
                                });
                        },
                    );
                });

            let q = state.assets_filter.trim().to_ascii_lowercase();

            let asset_usage = compute_asset_usage_counts_cached(session, state);
            let unused_asset_count = session
                .project
                .assets
                .keys()
                .filter(|id| asset_usage.get(*id).copied().unwrap_or(0) == 0)
                .count();

            // Project assets bar (responsive layout preventing title/button overlap at 800x540)
            let (bar_rect, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 27.0), Sense::hover());
            ui.painter()
                .rect_filled(bar_rect, CornerRadius::ZERO, BG_TITLE);
            ui.painter().line_segment(
                [bar_rect.left_bottom(), bar_rect.right_bottom()],
                Stroke::new(1.0, BORDER_ROW_SOFT),
            );
            let narrow_assets_bar = bar_rect.width() < 235.0;
            let import_lbl = "+ Import";
            let purge_lbl = if narrow_assets_bar {
                "Purge"
            } else {
                "Purge Unused"
            };
            let import_w = ui
                .painter()
                .layout_no_wrap(import_lbl.to_owned(), theme.font(FONT_SMALL), TEXT_PRIMARY)
                .size()
                .x
                .ceil()
                + 14.0;
            let purge_w = ui
                .painter()
                .layout_no_wrap(purge_lbl.to_owned(), theme.font(FONT_COMPACT), TEXT_MUTED)
                .size()
                .x
                .ceil()
                + 10.0;
            let imp_rect = Rect::from_min_max(
                Pos2::new(
                    (bar_rect.right() - import_w).max(bar_rect.left() + 80.0),
                    bar_rect.top(),
                ),
                bar_rect.right_bottom(),
            );
            let purge_rect = Rect::from_min_max(
                Pos2::new(
                    (imp_rect.left() - purge_w - 4.0).max(bar_rect.left() + 52.0),
                    bar_rect.top(),
                ),
                Pos2::new(
                    (imp_rect.left() - 4.0).max(bar_rect.left() + 56.0),
                    bar_rect.bottom(),
                ),
            );
            let title_left = bar_rect.left() + 8.0;
            let title_max_w = (purge_rect.left() - title_left - 6.0).max(24.0);
            let title_text = if bar_rect.width() < 210.0 {
                "Assets"
            } else {
                "Project assets"
            };
            let title_galley = layout_single_line_ellipsized(
                ui.painter(),
                title_text,
                theme.font(FONT_SMALL),
                TEXT_STRONG,
                title_max_w,
            );
            let title_clip = Rect::from_min_max(
                Pos2::new(title_left, bar_rect.top()),
                Pos2::new((purge_rect.left() - 4.0).max(title_left), bar_rect.bottom()),
            );
            ui.painter().with_clip_rect(title_clip).galley(
                Pos2::new(
                    title_left,
                    bar_rect.center().y - title_galley.size().y * 0.5,
                ),
                title_galley,
                TEXT_STRONG,
            );
            let purge_resp = ui
                .interact(purge_rect, Id::unique("assets_purge_btn"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text("Remove assets not referenced by any clip, object, or effect");
            ui.painter().with_clip_rect(purge_rect).text(
                Pos2::new(purge_rect.right() - 4.0, bar_rect.center().y),
                Align2::RIGHT_CENTER,
                purge_lbl,
                theme.font(FONT_COMPACT),
                if purge_resp.hovered() {
                    DANGER_TEXT
                } else {
                    TEXT_MUTED
                },
            );
            if purge_resp.clicked() {
                if unused_asset_count == 0 {
                    actions.push(PanelAction::StatusMessage(
                        "No unused assets to purge".to_owned(),
                    ));
                } else if let Ok(res) = session.execute(Command::PurgeUnusedAssets) {
                    let removed = res
                        .get("removed_count")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    actions.push(PanelAction::StatusMessage(format!(
                        "Purged {removed} unused asset(s)"
                    )));
                }
            }
            let imp_resp = ui
                .interact(imp_rect, Id::unique("assets_import_btn"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand);
            ui.painter().with_clip_rect(imp_rect).text(
                Pos2::new(bar_rect.right() - 8.0, bar_rect.center().y),
                Align2::RIGHT_CENTER,
                import_lbl,
                theme.font(FONT_SMALL),
                if imp_resp.hovered() {
                    TEXT_BRIGHT
                } else {
                    theme.value()
                },
            );
            if imp_resp.clicked() {
                actions.push(PanelAction::ImportMedia);
            }

            let project_assets: Vec<(String, String, &'static str, u64, bool, usize)> = session
                .project
                .assets
                .values()
                .filter(|a| q.is_empty() || a.filename.to_ascii_lowercase().contains(&q))
                .map(|a| {
                    let count = asset_usage.get(&a.id).copied().unwrap_or(0);
                    (
                        a.id.clone(),
                        a.filename.clone(),
                        clip_kind_for_asset(a.kind, &a.filename),
                        a.size,
                        a.missing,
                        count,
                    )
                })
                .collect();

            if project_assets.is_empty() {
                Frame::NONE
                    .inner_margin(egui::Margin::symmetric(10, 8))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("No imported assets in project.")
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_EMPTY),
                        );
                    });
            } else {
                for (aid, fname, clip_kind, size, missing, use_count) in project_assets {
                    let row_h = if missing { 44.0 } else { 36.0 };
                    let (row_rect, resp) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), row_h),
                        Sense::click(),
                    );
                    let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
                    if resp.hovered() {
                        ui.painter()
                            .rect_filled(row_rect, CornerRadius::ZERO, BG_HOVER);
                    }
                    ui.painter().line_segment(
                        [row_rect.left_bottom(), row_rect.right_bottom()],
                        Stroke::new(1.0, BORDER_ROW_SOFT),
                    );
                    let usage_badge = if use_count == 0 {
                        "unused".to_owned()
                    } else {
                        format!("used {use_count}×")
                    };
                    let meta = if missing {
                        format!("MISSING · {usage_badge}")
                    } else if size > 0 {
                        format!("{clip_kind} · {} KiB · {usage_badge}", size.div_ceil(1024))
                    } else {
                        format!("{clip_kind} · {usage_badge}")
                    };
                    let text_right = if missing {
                        (row_rect.right() - 78.0).max(row_rect.left() + 30.0)
                    } else {
                        (row_rect.right() - 8.0).max(row_rect.left() + 30.0)
                    };
                    let text_max_w = (text_right - (row_rect.left() + 10.0)).max(20.0);
                    let text_clip = Rect::from_min_max(
                        Pos2::new(row_rect.left() + 10.0, row_rect.top()),
                        Pos2::new(text_right, row_rect.bottom()),
                    );
                    let fname_col = if missing { DANGER_TEXT } else { TEXT_PRIMARY };
                    let fname_galley = layout_single_line_ellipsized(
                        ui.painter(),
                        &fname,
                        theme.font(FONT_SMALL),
                        fname_col,
                        text_max_w,
                    );
                    ui.painter().with_clip_rect(text_clip).galley(
                        Pos2::new(
                            row_rect.left() + 10.0,
                            row_rect.top() + 13.0 - fname_galley.size().y * 0.5,
                        ),
                        fname_galley,
                        fname_col,
                    );
                    let meta_col = if missing { DANGER_TEXT } else { TEXT_MUTED };
                    let meta_galley = layout_single_line_ellipsized(
                        ui.painter(),
                        &meta,
                        theme.font(FONT_COMPACT),
                        meta_col,
                        text_max_w,
                    );
                    ui.painter().with_clip_rect(text_clip).galley(
                        Pos2::new(
                            row_rect.left() + 10.0,
                            row_rect.top() + 27.0 - meta_galley.size().y * 0.5,
                        ),
                        meta_galley,
                        meta_col,
                    );
                    let mut relinked = false;
                    if missing {
                        let relink_rect = Rect::from_min_size(
                            Pos2::new(row_rect.right() - 72.0, row_rect.center().y - 11.0),
                            Vec2::new(64.0, 22.0),
                        );
                        let relink_resp = ui
                            .interact(
                                relink_rect,
                                Id::unique(("asset_relink", &aid)),
                                Sense::click(),
                            )
                            .on_hover_cursor(CursorIcon::PointingHand);
                        ui.painter().rect_filled(
                            relink_rect,
                            CornerRadius::same(2),
                            if relink_resp.hovered() {
                                theme.accent()
                            } else {
                                BG_BUTTON
                            },
                        );
                        ui.painter().rect_stroke(
                            relink_rect,
                            CornerRadius::same(2),
                            Stroke::new(1.0, BORDER_CONTROL),
                            StrokeKind::Inside,
                        );
                        ui.painter().text(
                            relink_rect.center(),
                            Align2::CENTER_CENTER,
                            "Relink...",
                            theme.font(FONT_COMPACT),
                            TEXT_BRIGHT,
                        );
                        if relink_resp.clicked() {
                            relinked = true;
                            if let Some(path) = rfd::FileDialog::new()
                                .set_title(format!("Relink missing asset: {fname}"))
                                .pick_file()
                                && let Ok(bytes) = std::fs::read(&path)
                            {
                                let new_name = path
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or(&fname)
                                    .to_owned();
                                let _ = session.execute(Command::RelinkAsset {
                                    asset_id: aid.clone(),
                                    filename: new_name.clone(),
                                    bytes,
                                });
                                actions.push(PanelAction::StatusMessage(format!(
                                    "Relinked asset `{new_name}`"
                                )));
                            }
                        }
                    }
                    if !relinked && resp.clicked() {
                        add_clip_with_asset(session, clip_kind, &fname, Some(&aid));
                    }
                }
            }

            // Built-in starter templates / presets list
            let (tpl_hdr, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 27.0), Sense::hover());
            ui.painter()
                .rect_filled(tpl_hdr, CornerRadius::ZERO, BG_TITLE);
            ui.painter().line_segment(
                [tpl_hdr.left_bottom(), tpl_hdr.right_bottom()],
                Stroke::new(1.0, BORDER_ROW_SOFT),
            );
            ui.painter().text(
                Pos2::new(tpl_hdr.left() + 10.0, tpl_hdr.center().y),
                Align2::LEFT_CENTER,
                "Starter templates & presets",
                theme.font(FONT_SMALL),
                TEXT_STRONG,
            );

            let filtered_presets: Vec<&MediaPreset> = MEDIA_PRESETS
                .iter()
                .filter(|p| {
                    p.required_plugin
                        .is_none_or(|pid| session.plugins.is_enabled(pid))
                        && (q.is_empty()
                            || p.title.to_ascii_lowercase().contains(&q)
                            || p.subtitle.to_ascii_lowercase().contains(&q))
                })
                .collect();
            for preset in filtered_presets {
                let (row_rect, resp) =
                    ui.allocate_exact_size(Vec2::new(ui.available_width(), 44.0), Sense::click());
                let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
                let row_bg = if resp.hovered() { BG_HOVER } else { BG_PANEL };
                ui.painter()
                    .rect_filled(row_rect, CornerRadius::ZERO, row_bg);
                ui.painter().line_segment(
                    [row_rect.left_bottom(), row_rect.right_bottom()],
                    Stroke::new(1.0, BORDER_ROW_SOFT),
                );
                let icon_rect = Rect::from_min_size(
                    Pos2::new(row_rect.left() + 10.0, row_rect.center().y - 11.0),
                    Vec2::splat(22.0),
                );
                paint_media_card_icon(ui.painter(), icon_rect, preset.icon, TEXT_PRIMARY, row_bg);
                let tx = icon_rect.right() + 10.0;
                let max_text_w = (row_rect.right() - tx - 8.0).max(24.0);
                let text_clip = Rect::from_min_max(
                    Pos2::new(tx, row_rect.top()),
                    Pos2::new((row_rect.right() - 6.0).max(tx), row_rect.bottom()),
                );
                let title_col = if resp.hovered() {
                    TEXT_BRIGHT
                } else {
                    TEXT_PRIMARY
                };
                let title_galley = layout_single_line_ellipsized(
                    ui.painter(),
                    preset.title,
                    theme.font(FONT_SMALL),
                    title_col,
                    max_text_w,
                );
                ui.painter().with_clip_rect(text_clip).galley(
                    Pos2::new(tx, row_rect.top() + 14.0 - title_galley.size().y * 0.5),
                    title_galley,
                    title_col,
                );
                let sub_galley = layout_single_line_ellipsized(
                    ui.painter(),
                    preset.subtitle,
                    theme.font(FONT_COMPACT),
                    TEXT_OPTION_DESC,
                    max_text_w,
                );
                ui.painter().with_clip_rect(text_clip).galley(
                    Pos2::new(tx, row_rect.top() + 31.0 - sub_galley.size().y * 0.5),
                    sub_galley,
                    TEXT_OPTION_DESC,
                );
                if resp.clicked() {
                    add_preset_clip(session, preset.kind, preset.title);
                }
            }
        });
}
