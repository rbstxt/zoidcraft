use super::media_tab::layout_single_line_ellipsized;
use super::shared_widgets::{paint_crisp_panel_border, panel_top_header, warning_banner};
use super::*;

// ---------------------------------------------------------------------------
// Tab 6: Export (export) — Device & Frame Export (Spec 04 §4.1)
// ---------------------------------------------------------------------------

pub(super) fn container_label(c: ExportContainer) -> &'static str {
    match c {
        ExportContainer::Mp4 => "MP4 (.mp4)",
        ExportContainer::WebM => "WebM (.webm)",
        ExportContainer::Mkv => "MKV (.mkv)",
        ExportContainer::PngSequence => "PNG Sequence (.png)",
        ExportContainer::Gif => "GIF (.gif)",
        ExportContainer::Wav => "WAV Audio (.wav)",
        ExportContainer::M4a => "M4A Audio (.m4a)",
    }
}

fn video_codec_label(c: ExportVideoCodec) -> &'static str {
    match c {
        ExportVideoCodec::H264 => "H.264 / AVC",
        ExportVideoCodec::H265 => "H.265 / HEVC",
        ExportVideoCodec::Vp9 => "VP9",
        ExportVideoCodec::Vp8 => "VP8 (legacy)",
        ExportVideoCodec::Av1 => "AV1",
        ExportVideoCodec::Png => "PNG (lossless)",
    }
}

fn audio_codec_label(c: ExportAudioCodec) -> &'static str {
    match c {
        ExportAudioCodec::AacLc => "AAC-LC",
        ExportAudioCodec::Opus => "Opus",
        ExportAudioCodec::PcmS16Le => "PCM (s16le)",
        ExportAudioCodec::None => "None",
    }
}

pub(crate) fn available_export_containers(mp4_enabled: bool) -> &'static [ExportContainer] {
    if mp4_enabled {
        &ExportContainer::ALL
    } else {
        &[
            ExportContainer::WebM,
            ExportContainer::Mkv,
            ExportContainer::PngSequence,
            ExportContainer::Gif,
            ExportContainer::Wav,
            ExportContainer::M4a,
        ]
    }
}

pub(crate) fn allowed_video_codecs(container: ExportContainer) -> &'static [ExportVideoCodec] {
    match container {
        ExportContainer::Mp4 => &[
            ExportVideoCodec::H264,
            ExportVideoCodec::H265,
            ExportVideoCodec::Vp9,
            ExportVideoCodec::Av1,
        ],
        ExportContainer::WebM => &[
            ExportVideoCodec::Vp9,
            ExportVideoCodec::Vp8,
            ExportVideoCodec::Av1,
        ],
        ExportContainer::Mkv => &[
            ExportVideoCodec::H264,
            ExportVideoCodec::H265,
            ExportVideoCodec::Vp9,
            ExportVideoCodec::Vp8,
            ExportVideoCodec::Av1,
        ],
        ExportContainer::PngSequence => &[ExportVideoCodec::Png],
        ExportContainer::Gif | ExportContainer::Wav | ExportContainer::M4a => &[],
    }
}

pub(crate) fn allowed_audio_codecs(container: ExportContainer) -> &'static [ExportAudioCodec] {
    match container {
        ExportContainer::Mp4 => &[
            ExportAudioCodec::AacLc,
            ExportAudioCodec::Opus,
            ExportAudioCodec::None,
        ],
        ExportContainer::WebM => &[ExportAudioCodec::Opus, ExportAudioCodec::None],
        ExportContainer::Mkv => &[
            ExportAudioCodec::AacLc,
            ExportAudioCodec::Opus,
            ExportAudioCodec::PcmS16Le,
            ExportAudioCodec::None,
        ],
        ExportContainer::PngSequence | ExportContainer::Gif => &[ExportAudioCodec::None],
        ExportContainer::Wav => &[ExportAudioCodec::PcmS16Le],
        ExportContainer::M4a => &[ExportAudioCodec::AacLc],
    }
}

pub(crate) fn supports_transparent_alpha(
    container: ExportContainer,
    video_codec: ExportVideoCodec,
) -> bool {
    matches!(
        container,
        ExportContainer::PngSequence | ExportContainer::Gif
    ) || (matches!(container, ExportContainer::WebM | ExportContainer::Mkv)
        && video_codec == ExportVideoCodec::Vp9)
}

pub(crate) fn sanitize_export_settings(settings: &mut ExportSettings, mp4_enabled: bool) {
    let available = available_export_containers(mp4_enabled);
    if !available.contains(&settings.container) {
        settings.container = available[0];
    }
    let valid_vcodecs = allowed_video_codecs(settings.container);
    if !valid_vcodecs.is_empty() && !valid_vcodecs.contains(&settings.video_codec) {
        settings.video_codec = valid_vcodecs[0];
    }
    let valid_acodecs = allowed_audio_codecs(settings.container);
    if !valid_acodecs.contains(&settings.audio_codec) {
        settings.audio_codec = valid_acodecs[0];
    }
    if !supports_transparent_alpha(settings.container, settings.video_codec) {
        settings.transparent_alpha = false;
    }
    let max_crf: u32 = match settings.video_codec {
        ExportVideoCodec::H264 | ExportVideoCodec::H265 => 51,
        _ => 63,
    };
    settings.crf = settings.crf.min(max_crf);
}

fn paint_export_desktop_icon(painter: &Painter, rect: Rect, color: Color32, cutout: Color32) {
    // 54x40 desktop monitor + PC tower icon matching Zoidium 06_export_clean.png
    let pt = |nx: f32, ny: f32| {
        Pos2::new(
            rect.left() + rect.width() * nx,
            rect.top() + rect.height() * ny,
        )
    };

    // Monitor bezel & inner screen
    let mon = Rect::from_min_max(pt(0.00, 0.00), pt(0.72, 0.70));
    painter.rect_filled(mon, CornerRadius::same(2), color);
    let screen = Rect::from_min_max(pt(0.07, 0.10), pt(0.65, 0.62));
    painter.rect_filled(screen, CornerRadius::ZERO, cutout);
    // Monitor stand neck & wider rounded foot
    let neck = Rect::from_min_max(pt(0.30, 0.70), pt(0.42, 0.80));
    painter.rect_filled(neck, CornerRadius::ZERO, color);
    let base = Rect::from_min_max(pt(0.14, 0.80), pt(0.58, 0.96));
    painter.rect_filled(base, CornerRadius::same(4), color);

    // Desktop PC tower on right with single top disc slot
    let tower = Rect::from_min_max(pt(0.70, 0.12), pt(1.00, 0.92));
    painter.rect_filled(tower, CornerRadius::same(2), color);
    let bay = Rect::from_min_max(pt(0.76, 0.24), pt(0.94, 0.30));
    painter.rect_filled(bay, CornerRadius::ZERO, cutout);
}

fn paint_export_frame_icon(painter: &Painter, rect: Rect, color: Color32, cutout: Color32) {
    // 52x42 framed landscape photo icon matching Zoidium 06_export_clean.png
    let pt = |nx: f32, ny: f32| {
        Pos2::new(
            rect.left() + rect.width() * nx,
            rect.top() + rect.height() * ny,
        )
    };

    let outer = Rect::from_min_max(pt(0.00, 0.00), pt(1.00, 1.00));
    painter.rect_filled(outer, CornerRadius::same(1), color);
    let inner = Rect::from_min_max(pt(0.11, 0.14), pt(0.89, 0.86));
    painter.rect_filled(inner, CornerRadius::ZERO, cutout);

    // Sun circle (upper right-center)
    painter.circle_filled(pt(0.67, 0.34), rect.height() * 0.085, color);

    // Left smaller mountain peak
    let m1 = vec![pt(0.11, 0.86), pt(0.28, 0.48), pt(0.47, 0.86)];
    painter.add(Shape::convex_polygon(m1, color, Stroke::NONE));

    // Right wider overlapping mountain peak
    let m2 = vec![pt(0.36, 0.86), pt(0.56, 0.56), pt(0.82, 0.86)];
    painter.add(Shape::convex_polygon(m2, color, Stroke::NONE));
}

fn paint_export_pz_icon(painter: &Painter, rect: Rect, color: Color32, cutout: Color32) {
    // 52x42 portable project archive box + downward export arrow icon
    let pt = |nx: f32, ny: f32| {
        Pos2::new(
            rect.left() + rect.width() * nx,
            rect.top() + rect.height() * ny,
        )
    };

    // Archive box body
    let outer = Rect::from_min_max(pt(0.08, 0.10), pt(0.92, 0.94));
    painter.rect_filled(outer, CornerRadius::same(2), color);
    let inner = Rect::from_min_max(pt(0.16, 0.30), pt(0.84, 0.86));
    painter.rect_filled(inner, CornerRadius::ZERO, cutout);
    // Top lid divider
    let lid = Rect::from_min_max(pt(0.04, 0.06), pt(0.96, 0.24));
    painter.rect_filled(lid, CornerRadius::same(2), color);
    // Downward export arrow inside box
    let shaft = Rect::from_min_max(pt(0.44, 0.36), pt(0.56, 0.62));
    painter.rect_filled(shaft, CornerRadius::ZERO, color);
    let head = vec![pt(0.32, 0.58), pt(0.68, 0.58), pt(0.50, 0.78)];
    painter.add(Shape::convex_polygon(head, color, Stroke::NONE));
}

/// Full-column-width CM3-styled dropdown used by the Device Render panel so all export
/// dropdowns (`Container`, `Video codec`, `Rate control`, `Preset`, `Audio codec`, `Sample rate`)
/// share one unified width and align cleanly on the right edge.
fn export_uniform_dropdown(
    ui: &mut Ui,
    id: Id,
    selected_idx: &mut usize,
    options: &[&str],
    viewport_rect: Rect,
    theme: &Cm3Theme,
) -> bool {
    if options.is_empty() {
        return false;
    }
    if *selected_idx >= options.len() {
        *selected_idx = 0;
    }
    let popup_id = id.with("export_dd_popup");
    let mut open = ui
        .data_mut(|d| d.get_temp::<bool>(popup_id))
        .unwrap_or(false);
    let mut changed = false;

    let width = ui.available_width().max(64.0).round();
    let (_, raw_slot) = ui.allocate_space(Vec2::new(width, 23.0));
    let btn_rect = Rect::from_min_size(
        Pos2::new(raw_slot.left().round(), (raw_slot.top() + 1.0).round()),
        Vec2::new(width, 21.0),
    );
    let btn_resp = ui
        .interact(btn_rect, id.with("btn"), Sense::click())
        .on_hover_cursor(CursorIcon::PointingHand);
    ui.painter()
        .rect_filled(btn_rect, CornerRadius::ZERO, BG_INPUT);
    paint_crisp_panel_border(
        ui.painter(),
        btn_rect,
        if open || btn_resp.hovered() {
            theme.accent()
        } else {
            BORDER_CONTROL
        },
    );
    let font = theme.font(13.3);
    let text_max_w = (btn_rect.width() - 24.0).max(16.0);
    let lbl_galley = layout_single_line_ellipsized(
        ui.painter(),
        options[*selected_idx],
        font.clone(),
        theme.value(),
        text_max_w,
    );
    let text_clip = Rect::from_min_max(
        btn_rect.left_top(),
        Pos2::new(
            (btn_rect.right() - 18.0).max(btn_rect.left()),
            btn_rect.bottom(),
        ),
    );
    ui.painter().with_clip_rect(text_clip).galley(
        Pos2::new(
            btn_rect.left() + 5.0,
            btn_rect.center().y - lbl_galley.size().y * 0.5,
        ),
        lbl_galley,
        theme.value(),
    );
    let caret_rect = Rect::from_center_size(
        Pos2::new(btn_rect.right() - 8.5, btn_rect.center().y),
        Vec2::splat(12.0),
    );
    paint_cm3_icon(
        ui.painter(),
        caret_rect,
        Cm3Icon::ChevronDown,
        theme.value(),
    );

    let mut just_opened = false;
    if btn_resp.clicked() {
        open = !open;
        if open {
            just_opened = true;
        }
    }

    if open {
        let row_h = 22.0_f32;
        let max_opt_w = options
            .iter()
            .map(|o| {
                ui.painter()
                    .layout_no_wrap((*o).to_owned(), font.clone(), theme.value())
                    .size()
                    .x
            })
            .fold(0.0_f32, f32::max);
        let popup_w = (max_opt_w + 18.0)
            .max(btn_rect.width())
            .min((viewport_rect.width() - 16.0).max(96.0))
            .round();
        let popup_h = ((options.len() as f32) * row_h + 4.0)
            .min((viewport_rect.height() - 16.0).max(60.0))
            .round();
        let x = btn_rect.left().clamp(
            viewport_rect.left() + 8.0,
            (viewport_rect.right() - popup_w - 8.0).max(viewport_rect.left() + 8.0),
        );
        let below_y = btn_rect.bottom() + 2.0;
        let y = if below_y + popup_h <= viewport_rect.bottom() - 8.0 {
            below_y
        } else {
            (btn_rect.top() - popup_h - 2.0).clamp(
                viewport_rect.top() + 8.0,
                (viewport_rect.bottom() - popup_h - 8.0).max(viewport_rect.top() + 8.0),
            )
        };
        let mut close_now = ui.input(|i| i.key_pressed(Key::Escape));
        if close_now {
            mark_floating_editor_escape_consumed(ui.ctx());
        }
        let area_resp = Area::new(popup_id)
            .order(Order::Foreground)
            .fixed_pos(Pos2::new(x.round(), y.round()))
            .interactable(true)
            .movable(false)
            .show(ui.ctx(), |ui| {
                Frame::NONE
                    .fill(BG_INPUT)
                    .stroke(Stroke::new(1.0, theme.accent()))
                    .inner_margin(egui::Margin::same(2))
                    .show(ui, |ui| {
                        let inner_w = (popup_w - 4.0).max(24.0);
                        ui.set_min_width(inner_w);
                        ui.set_max_width(inner_w);
                        ui.spacing_mut().item_spacing = Vec2::ZERO;
                        let active_bg = theme.menu_active();
                        let active_fg = theme.contrast_fg(active_bg);
                        for (idx, opt) in options.iter().enumerate() {
                            let (r_rect, r_resp) =
                                ui.allocate_exact_size(Vec2::new(inner_w, row_h), Sense::click());
                            let r_resp = r_resp.on_hover_cursor(CursorIcon::PointingHand);
                            let is_sel = idx == *selected_idx;
                            if r_resp.hovered() {
                                ui.painter()
                                    .rect_filled(r_rect, CornerRadius::ZERO, active_bg);
                            } else if is_sel {
                                ui.painter().rect_filled(
                                    r_rect,
                                    CornerRadius::ZERO,
                                    BG_DROPDOWN_HOVER,
                                );
                            }
                            ui.painter()
                                .with_clip_rect(r_rect.shrink2(Vec2::new(4.0, 0.0)))
                                .text(
                                    Pos2::new(r_rect.left() + 5.0, r_rect.center().y),
                                    Align2::LEFT_CENTER,
                                    *opt,
                                    font.clone(),
                                    if r_resp.hovered() {
                                        active_fg
                                    } else if is_sel {
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
        let clicked_outside = !just_opened
            && ui.input(|i| {
                i.pointer.primary_pressed()
                    && i.pointer.interact_pos().is_some_and(|p| {
                        !area_resp.response.rect.contains(p) && !btn_rect.contains(p)
                    })
            });
        if close_now || clicked_outside {
            open = false;
        }
    }
    ui.data_mut(|d| d.insert_temp(popup_id, open));
    changed
}

pub(super) fn show_export_tab(
    ui: &mut Ui,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let mp4_enabled = session.plugins.is_enabled("mp4-export");
    let available_containers = available_export_containers(mp4_enabled);

    // Auto-coerce container, video codec, audio codec, alpha, and CRF bounds
    sanitize_export_settings(&mut state.export_settings, mp4_enabled);

    ScrollArea::vertical()
        .id_salt("export_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;

            // Edge-to-edge 42px #222222 header bar ("Export project")
            panel_top_header(ui, "Export project", theme);

            // Landing screen (default None): full-width clickable #2a2a2a cards whose height
            // dynamically grows to fit wrapped subtitle lines at narrow widths (e.g. 800x540).
            let Some(active_mode) = state.export_view_mode else {
                let compute_card_h = |ui: &Ui, title: &str, subtitle: &str| -> f32 {
                    let avail_w = ui.available_width().max(120.0);
                    let text_w = (avail_w - 88.0).max(40.0);
                    let title_h = ui
                        .painter()
                        .layout(
                            title.to_owned(),
                            theme.font(FONT_BODY),
                            TEXT_PRIMARY,
                            text_w,
                        )
                        .size()
                        .y;
                    let sub_h = ui
                        .painter()
                        .layout(
                            subtitle.to_owned(),
                            theme.font(FONT_SMALL),
                            TEXT_OPTION_DESC,
                            text_w,
                        )
                        .size()
                        .y;
                    (9.0 + title_h + 3.0 + sub_h + 10.0).ceil().max(77.0)
                };
                for (card_idx, (mode, title, subtitle)) in [
                    (
                        ExportMode::Device,
                        "Device render",
                        "Render your video on your device.",
                    ),
                    (
                        ExportMode::Frame,
                        "Single frame capture",
                        "Save a single frame as an image.",
                    ),
                ]
                .into_iter()
                .enumerate()
                {
                    let card_h = compute_card_h(ui, title, subtitle);
                    let (row_rect, _) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), card_h),
                        Sense::hover(),
                    );
                    let card_id = Id::unique(("export_landing_card", card_idx));
                    let hovered = ui.rect_contains_pointer(row_rect);
                    let fill = if hovered { BG_HOVER } else { BG_PANEL };
                    ui.painter().rect_filled(row_rect, CornerRadius::ZERO, fill);
                    let div_y = (row_rect.bottom() - 1.0).round();
                    ui.painter().rect_filled(
                        Rect::from_min_max(
                            Pos2::new(row_rect.left(), div_y),
                            Pos2::new(row_rect.right(), div_y + 1.0),
                        ),
                        CornerRadius::ZERO,
                        BORDER_ROW_SOFT,
                    );

                    // Procedural vector icon centered vertically on the left
                    if mode == ExportMode::Device {
                        let icon_rect = Rect::from_min_size(
                            Pos2::new(row_rect.left() + 11.0, (row_rect.center().y - 20.0).round()),
                            Vec2::new(54.0, 40.0),
                        );
                        paint_export_desktop_icon(ui.painter(), icon_rect, TEXT_PRIMARY, fill);
                    } else {
                        let icon_rect = Rect::from_min_size(
                            Pos2::new(row_rect.left() + 12.0, (row_rect.center().y - 21.0).round()),
                            Vec2::new(52.0, 42.0),
                        );
                        paint_export_frame_icon(ui.painter(), icon_rect, TEXT_PRIMARY, fill);
                    }

                    let text_left = row_rect.left() + 78.0;
                    let text_rect = Rect::from_min_max(
                        Pos2::new(text_left, row_rect.top() + 9.0),
                        Pos2::new(
                            (row_rect.right() - 10.0).max(text_left + 20.0),
                            (row_rect.bottom() - 4.0).max(row_rect.top() + 24.0),
                        ),
                    );
                    let mut text_ui = ui.new_child(egui::UiBuilder::new().max_rect(text_rect));
                    text_ui.set_clip_rect(text_ui.clip_rect().intersect(text_rect));
                    text_ui.spacing_mut().item_spacing.y = 3.0;
                    text_ui.add(
                        egui::Label::new(
                            egui::RichText::new(title)
                                .font(theme.font(FONT_BODY))
                                .color(if hovered { TEXT_BRIGHT } else { TEXT_PRIMARY }),
                        )
                        .wrap()
                        .selectable(false),
                    );
                    text_ui.add(
                        egui::Label::new(
                            egui::RichText::new(subtitle)
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_OPTION_DESC),
                        )
                        .wrap()
                        .selectable(false),
                    );

                    let resp = ui
                        .interact(row_rect, card_id, Sense::click())
                        .on_hover_cursor(CursorIcon::PointingHand);
                    if resp.clicked() {
                        state.export_settings.mode = mode;
                        state.export_view_mode = Some(mode);
                    }
                }

                // Portable `.pz` project archive export row
                {
                    let pz_title = "Export .pz...";
                    let pz_sub = "Save the project as a portable .pz archive.";
                    let card_h = compute_card_h(ui, pz_title, pz_sub);
                    let (row_rect, _) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), card_h),
                        Sense::hover(),
                    );
                    let hovered = ui.rect_contains_pointer(row_rect);
                    let fill = if hovered { BG_HOVER } else { BG_PANEL };
                    ui.painter().rect_filled(row_rect, CornerRadius::ZERO, fill);
                    let div_y = (row_rect.bottom() - 1.0).round();
                    ui.painter().rect_filled(
                        Rect::from_min_max(
                            Pos2::new(row_rect.left(), div_y),
                            Pos2::new(row_rect.right(), div_y + 1.0),
                        ),
                        CornerRadius::ZERO,
                        BORDER_ROW_SOFT,
                    );

                    let icon_rect = Rect::from_min_size(
                        Pos2::new(row_rect.left() + 12.0, (row_rect.center().y - 21.0).round()),
                        Vec2::new(52.0, 42.0),
                    );
                    paint_export_pz_icon(ui.painter(), icon_rect, TEXT_PRIMARY, fill);

                    let text_left = row_rect.left() + 78.0;
                    let text_rect = Rect::from_min_max(
                        Pos2::new(text_left, row_rect.top() + 9.0),
                        Pos2::new(
                            (row_rect.right() - 10.0).max(text_left + 20.0),
                            (row_rect.bottom() - 4.0).max(row_rect.top() + 24.0),
                        ),
                    );
                    let mut text_ui = ui.new_child(egui::UiBuilder::new().max_rect(text_rect));
                    text_ui.set_clip_rect(text_ui.clip_rect().intersect(text_rect));
                    text_ui.spacing_mut().item_spacing.y = 3.0;
                    text_ui.add(
                        egui::Label::new(
                            egui::RichText::new(pz_title)
                                .font(theme.font(FONT_BODY))
                                .color(if hovered { TEXT_BRIGHT } else { TEXT_PRIMARY }),
                        )
                        .wrap()
                        .selectable(false),
                    );
                    text_ui.add(
                        egui::Label::new(
                            egui::RichText::new(pz_sub)
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_OPTION_DESC),
                        )
                        .wrap()
                        .selectable(false),
                    );

                    let resp = ui
                        .interact(row_rect, Id::unique("export_landing_pz_card"), Sense::click())
                        .on_hover_cursor(CursorIcon::PointingHand);
                    if resp.clicked() {
                        state.pending_project_actions.push(ProjectAction::ExportPz);
                    }
                }
                if state.export_status.exporting {
                    Frame::NONE
                        .inner_margin(egui::Margin::symmetric(8, 8))
                        .show(ui, |ui| {
                            ui.spacing_mut().item_spacing.y = 6.0;
                            let p = state.export_status.progress.clamp(0.0, 1.0);
                            let (bar_rect, _) = ui.allocate_exact_size(
                                Vec2::new(ui.available_width(), 20.0),
                                Sense::hover(),
                            );
                            ui.painter()
                                .rect_filled(bar_rect, CornerRadius::ZERO, BG_TITLE);
                            let fill_rect = Rect::from_min_size(
                                bar_rect.left_top(),
                                Vec2::new(bar_rect.width() * p, bar_rect.height()),
                            );
                            ui.painter()
                                .rect_filled(fill_rect, CornerRadius::ZERO, theme.accent());
                            ui.painter().text(
                                bar_rect.center(),
                                Align2::CENTER_CENTER,
                                format_export_progress_label(&state.export_status),
                                theme.font(FONT_SMALL),
                                TEXT_BRIGHT,
                            );
                            if !state.export_status.status_message.is_empty() {
                                ui.label(
                                    egui::RichText::new(&state.export_status.status_message)
                                        .font(theme.font(FONT_SMALL))
                                        .color(TEXT_MUTED),
                                );
                            }
                            if cm3_prop_button(
                                ui,
                                "Cancel export",
                                ButtonVariant::Danger,
                                true,
                                theme,
                            )
                            .clicked()
                            {
                                actions.push(PanelAction::CancelExport);
                            }
                        });
                } else if let Some(completed_path) =
                    state.export_status.last_completed_output.clone()
                {
                    Frame::NONE
                        .inner_margin(egui::Margin::symmetric(8, 6))
                        .show(ui, |ui| {
                            if cm3_prop_button(
                                ui,
                                "Open Output Folder",
                                ButtonVariant::Normal,
                                true,
                                theme,
                            )
                            .clicked()
                            {
                                actions.push(PanelAction::OpenOutputFolder(completed_path));
                            }
                        });
                }
                if !state.export_queue.is_empty() {
                    Frame::NONE
                        .inner_margin(egui::Margin::symmetric(8, 8))
                        .show(ui, |ui| {
                            show_export_queue_section(ui, theme, state, actions);
                        });
                }
                return;
            };

            state.export_settings.mode = active_mode;
            let is_device = active_mode == ExportMode::Device;
            let is_audio_only = is_device && state.export_settings.container.is_audio_only();
            let needs_ffmpeg = is_device && state.export_settings.container.requires_ffmpeg();
            let mut export_cols = state.prop_columns;
            export_cols.label_ratio = export_cols.label_ratio.max(0.47);

            Frame::NONE
                .inner_margin(egui::Margin::symmetric(8, 8))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 4.0;

                    // Clean "← Back" button (hidden while exporting so progress & Cancel stay visible) + mode title
                    ui.horizontal(|ui| {
                        if !state.export_status.exporting {
                            if cm3_prop_button(ui, "← Back", ButtonVariant::Back, false, theme)
                                .clicked()
                            {
                                state.export_view_mode = None;
                            }
                            ui.add_space(6.0);
                        }
                        ui.label(
                            egui::RichText::new(if is_device {
                                "Device render"
                            } else {
                                "Single frame capture"
                            })
                            .font(theme.font(FONT_BODY))
                            .color(TEXT_PRIMARY),
                        );
                    });
                    ui.add_space(4.0);

                    // FFmpeg status bar (shown for video device render modes)
                    if needs_ffmpeg {
                        let ff_color = if state.export_status.ffmpeg_available {
                            TEXT_PRIMARY
                        } else {
                            DANGER_TEXT
                        };
                        Frame::NONE
                            .fill(BG_TITLE)
                            .stroke(Stroke::new(1.0, BORDER_CONTROL))
                            .inner_margin(egui::Margin::symmetric(8, 5))
                            .show(ui, |ui| {
                                ui.set_min_width(ui.available_width());
                                ui.label(
                                    egui::RichText::new(format!(
                                        "FFmpeg: {}",
                                        state.export_status.ffmpeg_version
                                    ))
                                    .font(theme.font(FONT_SMALL))
                                    .color(ff_color),
                                );
                            });
                        ui.add_space(4.0);
                    }

                    if is_device {
                        // Container (MP4 gated on `mp4-export` plugin)
                        let container_labels: Vec<&str> = available_containers
                            .iter()
                            .map(|&c| container_label(c))
                            .collect();
                        let mut c_idx = available_containers
                            .iter()
                            .position(|&c| c == state.export_settings.container)
                            .unwrap_or(0);
                        let changed = cm3_property_row_aligned(
                            ui,
                            "Container",
                            false,
                            export_cols,
                            theme,
                            |_| {},
                            |ui| {
                                export_uniform_dropdown(
                                    ui,
                                    Id::unique("exp_container"),
                                    &mut c_idx,
                                    &container_labels,
                                    viewport_rect,
                                    theme,
                                )
                            },
                        );
                        if changed {
                            state.export_settings.container = available_containers[c_idx];
                            sanitize_export_settings(&mut state.export_settings, mp4_enabled);
                        }

                        // Video codec and rate control are hidden for PNG Sequence, GIF, WAV, and M4A
                        if state.export_settings.container.uses_video_codec() {
                            let v_list = allowed_video_codecs(state.export_settings.container);
                            let v_labels: Vec<&str> =
                                v_list.iter().map(|&c| video_codec_label(c)).collect();
                            let mut vc_idx = v_list
                                .iter()
                                .position(|&c| c == state.export_settings.video_codec)
                                .unwrap_or(0);
                            if cm3_property_row_aligned(
                                ui,
                                "Video codec",
                                false,
                                export_cols,
                                theme,
                                |_| {},
                                |ui| {
                                    export_uniform_dropdown(
                                        ui,
                                        Id::unique("exp_vcodec"),
                                        &mut vc_idx,
                                        &v_labels,
                                        viewport_rect,
                                        theme,
                                    )
                                },
                            ) {
                                state.export_settings.video_codec = v_list[vc_idx];
                                sanitize_export_settings(&mut state.export_settings, mp4_enabled);
                            }

                            // Rate control
                            let mut rc_idx = ExportRateControl::ALL
                                .iter()
                                .position(|&r| r == state.export_settings.rate_control)
                                .unwrap_or(0);
                            if cm3_property_row_aligned(
                                ui,
                                "Rate control",
                                false,
                                export_cols,
                                theme,
                                |_| {},
                                |ui| {
                                    export_uniform_dropdown(
                                        ui,
                                        Id::unique("exp_rc"),
                                        &mut rc_idx,
                                        &ExportRateControl::LABELS,
                                        viewport_rect,
                                        theme,
                                    )
                                },
                            ) {
                                state.export_settings.rate_control =
                                    ExportRateControl::ALL[rc_idx];
                            }

                            let cur_max_crf: u32 = match state.export_settings.video_codec {
                                ExportVideoCodec::H264 | ExportVideoCodec::H265 => 51,
                                _ => 63,
                            };
                            match state.export_settings.rate_control {
                                ExportRateControl::Crf => {
                                    let mut crf =
                                        f64::from(state.export_settings.crf.min(cur_max_crf));
                                    let ev = cm3_property_row_aligned(
                                        ui,
                                        "CRF",
                                        false,
                                        export_cols,
                                        theme,
                                        |_| {},
                                        |ui| {
                                            cm3_scrub_number(
                                                ui,
                                                Id::unique("exp_crf"),
                                                &mut crf,
                                                ScrubConfig::new(
                                                    0.0,
                                                    f64::from(cur_max_crf),
                                                    1.0,
                                                    0,
                                                ),
                                                theme,
                                            )
                                        },
                                    );
                                    if ev.changed || ev.committed {
                                        state.export_settings.crf =
                                            crf.round().clamp(0.0, f64::from(cur_max_crf)) as u32;
                                    }
                                }
                                ExportRateControl::Bitrate => {
                                    let mut mbps = state.export_settings.bitrate_mbps;
                                    let ev = cm3_property_row_aligned(
                                        ui,
                                        "Bitrate Mbps",
                                        false,
                                        export_cols,
                                        theme,
                                        |_| {},
                                        |ui| {
                                            cm3_scrub_number(
                                                ui,
                                                Id::unique("exp_bitrate"),
                                                &mut mbps,
                                                ScrubConfig::new(0.5, 200.0, 0.5, 1),
                                                theme,
                                            )
                                        },
                                    );
                                    if ev.changed || ev.committed {
                                        state.export_settings.bitrate_mbps = mbps;
                                    }
                                }
                                ExportRateControl::Cm3Preset => {
                                    let mut p_idx = ExportCm3Preset::ALL
                                        .iter()
                                        .position(|&p| p == state.export_settings.cm3_preset)
                                        .unwrap_or(2);
                                    if cm3_property_row_aligned(
                                        ui,
                                        "Preset",
                                        false,
                                        export_cols,
                                        theme,
                                        |_| {},
                                        |ui| {
                                            export_uniform_dropdown(
                                                ui,
                                                Id::unique("exp_cm3_preset"),
                                                &mut p_idx,
                                                &ExportCm3Preset::LABELS,
                                                viewport_rect,
                                                theme,
                                            )
                                        },
                                    ) {
                                        state.export_settings.cm3_preset =
                                            ExportCm3Preset::ALL[p_idx];
                                    }
                                }
                            }
                        }

                        // Transparent alpha (PNG Sequence, GIF, or VP9 in WebM/MKV)
                        let show_alpha = supports_transparent_alpha(
                            state.export_settings.container,
                            state.export_settings.video_codec,
                        );
                        if show_alpha {
                            let mut alpha_idx = usize::from(state.export_settings.transparent_alpha);
                            if cm3_property_row_aligned(
                                ui,
                                "Alpha channel",
                                false,
                                export_cols,
                                theme,
                                |_| {},
                                |ui| {
                                    export_uniform_dropdown(
                                        ui,
                                        Id::unique("exp_alpha"),
                                        &mut alpha_idx,
                                        &["off", "on"],
                                        viewport_rect,
                                        theme,
                                    )
                                },
                            ) {
                                state.export_settings.transparent_alpha = alpha_idx != 0;
                            }
                        }

                        // Audio controls (hidden for PNG Sequence and GIF)
                        if state.export_settings.container.supports_audio() {
                            let a_list = allowed_audio_codecs(state.export_settings.container);
                            let a_labels: Vec<&str> =
                                a_list.iter().map(|&c| audio_codec_label(c)).collect();
                            let mut ac_idx = a_list
                                .iter()
                                .position(|&a| a == state.export_settings.audio_codec)
                                .unwrap_or(0);
                            if cm3_property_row_aligned(
                                ui,
                                "Audio codec",
                                false,
                                export_cols,
                                theme,
                                |_| {},
                                |ui| {
                                    export_uniform_dropdown(
                                        ui,
                                        Id::unique("exp_acodec"),
                                        &mut ac_idx,
                                        &a_labels,
                                        viewport_rect,
                                        theme,
                                    )
                                },
                            ) {
                                state.export_settings.audio_codec = a_list[ac_idx];
                            }

                            if matches!(
                                state.export_settings.audio_codec,
                                ExportAudioCodec::AacLc | ExportAudioCodec::Opus
                            ) {
                                let mut kbps = f64::from(state.export_settings.audio_bitrate_kbps);
                                let ev = cm3_property_row_aligned(
                                    ui,
                                    "Audio kbps",
                                    false,
                                    export_cols,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        cm3_scrub_number(
                                            ui,
                                            Id::unique("exp_abitrate"),
                                            &mut kbps,
                                            ScrubConfig::new(32.0, 512.0, 16.0, 0),
                                            theme,
                                        )
                                    },
                                );
                                if ev.changed || ev.committed {
                                    state.export_settings.audio_bitrate_kbps =
                                        kbps.round().clamp(32.0, 512.0) as u32;
                                }
                            }

                            if state.export_settings.audio_codec != ExportAudioCodec::None {
                                const RATES: [u32; 4] = [22_050, 44_100, 48_000, 96_000];
                                const RATE_LABELS: [&str; 4] =
                                    ["22050 Hz", "44100 Hz", "48000 Hz", "96000 Hz"];
                                let mut sr_idx = RATES
                                    .iter()
                                    .position(|&r| r == state.export_settings.sample_rate_hz)
                                    .unwrap_or(2);
                                if cm3_property_row_aligned(
                                    ui,
                                    "Sample rate",
                                    false,
                                    export_cols,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        export_uniform_dropdown(
                                            ui,
                                            Id::unique("exp_sample_rate"),
                                            &mut sr_idx,
                                            &RATE_LABELS,
                                            viewport_rect,
                                            theme,
                                        )
                                    },
                                ) {
                                    state.export_settings.sample_rate_hz = RATES[sr_idx];
                                }
                            }
                        }
                    }

                    // Resolution (shown for Video, PNG Sequence, GIF, and Single Frame; hidden for audio-only)
                    if !is_audio_only {
                        let mut res = [
                            f64::from(state.export_settings.width),
                            f64::from(state.export_settings.height),
                        ];
                        let ev = cm3_property_row_aligned(
                            ui,
                            "Resolution",
                            false,
                            export_cols,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_vector2_with_sep(
                                    ui,
                                    Id::unique("exp_res"),
                                    &mut res,
                                    &ScrubConfig::new(16.0, 7680.0, 10.0, 0),
                                    false,
                                    "x",
                                    theme,
                                )
                            },
                        );
                        if ev.changed || ev.committed {
                            state.export_settings.width = res[0].round().clamp(16.0, 7680.0) as u32;
                            state.export_settings.height =
                                res[1].round().clamp(16.0, 7680.0) as u32;
                        }
                    }

                    // Frame rate (video/GIF/PNG Sequence) & Range (all Device exports including audio-only)
                    if is_device {
                        if !is_audio_only {
                            let mut fps = state.export_settings.fps;
                            let ev = cm3_property_row_aligned(
                                ui,
                                "Frame rate",
                                false,
                                export_cols,
                                theme,
                                |_| {},
                                |ui| {
                                    cm3_scrub_number(
                                        ui,
                                        Id::unique("exp_fps"),
                                        &mut fps,
                                        ScrubConfig::new(1.0, 240.0, 1.0, 0),
                                        theme,
                                    )
                                },
                            );
                            if ev.changed || ev.committed {
                                state.export_settings.fps = fps;
                            }
                        }

                        let mut range = [
                            state.export_settings.start_frame,
                            state.export_settings.end_frame,
                        ];
                        let ev = cm3_property_row_aligned(
                            ui,
                            "Frame range",
                            false,
                            export_cols,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_vector2(
                                    ui,
                                    Id::unique("exp_range"),
                                    &mut range,
                                    ScrubConfig::new(0.0, 1_000_000.0, 1.0, 0),
                                    false,
                                    theme,
                                )
                            },
                        );
                        if ev.changed || ev.committed {
                            state.export_settings.start_frame = range[0].max(0.0);
                            state.export_settings.end_frame = range[1].max(range[0] + 1.0);
                        }
                    }

                    ui.add_space(4.0);
                    if cm3_prop_button(ui, "Sync from sequence", ButtonVariant::Normal, true, theme)
                        .clicked()
                    {
                        let [w, h] = session.document().resolution;
                        state.export_settings.width = w;
                        state.export_settings.height = h;
                        state.export_settings.fps = session.document().fps;
                        state.export_settings.start_frame = 0.0;
                        state.export_settings.end_frame = session.document().sequence_length;
                    }

                    ui.add_space(8.0);

                    // Live Progress Bar (#222222 track, #384668 accent fill, Spec 01 §2.5)
                    if state.export_status.exporting || state.export_status.progress > 0.0 {
                        let p = state.export_status.progress.clamp(0.0, 1.0);
                        let (bar_rect, _) = ui.allocate_exact_size(
                            Vec2::new(ui.available_width(), 20.0),
                            Sense::hover(),
                        );
                        ui.painter()
                            .rect_filled(bar_rect, CornerRadius::ZERO, BG_TITLE);
                        let fill_rect = Rect::from_min_size(
                            bar_rect.left_top(),
                            Vec2::new(bar_rect.width() * p, bar_rect.height()),
                        );
                        ui.painter()
                            .rect_filled(fill_rect, CornerRadius::ZERO, theme.accent());
                        ui.painter().rect_stroke(
                            bar_rect,
                            CornerRadius::ZERO,
                            Stroke::new(1.0, BORDER_CONTROL),
                            StrokeKind::Inside,
                        );
                        ui.painter().text(
                            bar_rect.center(),
                            Align2::CENTER_CENTER,
                            format_export_progress_label(&state.export_status),
                            theme.font(FONT_SMALL),
                            TEXT_BRIGHT,
                        );
                        ui.add_space(4.0);
                    }

                    // Always display non-empty status_message (even when progress == 0.0, e.g. errors)
                    if !state.export_status.status_message.is_empty() {
                        let lower = state.export_status.status_message.to_ascii_lowercase();
                        let is_err = lower.contains("error")
                            || lower.contains("fail")
                            || lower.contains("missing")
                            || lower.contains("invalid");
                        if is_err {
                            warning_banner(ui, &state.export_status.status_message, theme);
                        } else {
                            ui.label(
                                egui::RichText::new(&state.export_status.status_message)
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_MUTED),
                            );
                        }
                        ui.add_space(6.0);
                    }

                    if state.export_status.exporting {
                        if cm3_prop_button(ui, "Cancel export", ButtonVariant::Danger, true, theme)
                            .clicked()
                        {
                            actions.push(PanelAction::CancelExport);
                        }
                    } else {
                        let label = if is_device {
                            match state.export_settings.container {
                                ExportContainer::PngSequence => "Start PNG sequence render",
                                ExportContainer::Gif => "Start GIF render",
                                ExportContainer::Wav => "Start WAV audio export",
                                ExportContainer::M4a => "Start M4A audio export",
                                ExportContainer::Mp4
                                | ExportContainer::WebM
                                | ExportContainer::Mkv => "Start video render",
                            }
                        } else {
                            "Export current frame (PNG)"
                        };
                        let can_start = !needs_ffmpeg || state.export_status.ffmpeg_available;
                        if can_start {
                            if cm3_prop_button(ui, label, ButtonVariant::Primary, true, theme)
                                .clicked()
                            {
                                actions
                                    .push(PanelAction::StartExport(state.export_settings.clone()));
                            }
                            if is_device {
                                ui.add_space(4.0);
                                if cm3_prop_button(
                                    ui,
                                    "Add to Queue",
                                    ButtonVariant::Normal,
                                    true,
                                    theme,
                                )
                                .clicked()
                                {
                                    actions.push(PanelAction::AddToExportQueue(
                                        state.export_settings.clone(),
                                    ));
                                }
                            }
                        } else {
                            let (btn_rect, resp) = ui.allocate_exact_size(
                                Vec2::new(ui.available_width(), 28.0),
                                Sense::hover(),
                            );
                            ui.painter()
                                .rect_filled(btn_rect, CornerRadius::ZERO, BG_BUTTON);
                            ui.painter().rect_stroke(
                                btn_rect,
                                CornerRadius::ZERO,
                                Stroke::new(1.0, BORDER_CONTROL),
                                StrokeKind::Inside,
                            );
                            ui.painter().text(
                                btn_rect.center(),
                                Align2::CENTER_CENTER,
                                label,
                                theme.font(FONT_BODY),
                                TEXT_DISABLED,
                            );
                            resp.on_hover_cursor(CursorIcon::NotAllowed)
                                .on_hover_text("FFmpeg is not available; install FFmpeg or export PNG Sequence, WAV Audio, or Single Frame.");
                        }

                        if let Some(completed_path) =
                            state.export_status.last_completed_output.clone()
                        {
                            ui.add_space(4.0);
                            if cm3_prop_button(
                                ui,
                                "Open Output Folder",
                                ButtonVariant::Normal,
                                true,
                                theme,
                            )
                            .clicked()
                            {
                                actions.push(PanelAction::OpenOutputFolder(completed_path));
                            }
                        }
                    }

                    if !state.export_queue.is_empty() {
                        ui.add_space(10.0);
                        show_export_queue_section(ui, theme, state, actions);
                    }
                });
        });
}

fn format_export_progress_label(status: &ExportStatus) -> String {
    let pct = status.progress.clamp(0.0, 1.0) * 100.0;
    match (status.live_fps, status.eta_secs) {
        (Some(fps), Some(eta)) if fps.is_finite() && eta.is_finite() => {
            format!("{pct:.0}% · {fps:.1} fps · ETA {}", format_eta_clock(eta))
        }
        (Some(fps), None) if fps.is_finite() => format!("{pct:.0}% · {fps:.1} fps"),
        _ => format!("{pct:.0}%"),
    }
}

fn format_eta_clock(secs: f64) -> String {
    let total = secs.max(0.0).round() as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

fn show_export_queue_section(
    ui: &mut Ui,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let queued_count = state
        .export_queue
        .iter()
        .filter(|j| j.status == ExportJobStatus::Queued)
        .count();
    let finished_count = state
        .export_queue
        .iter()
        .filter(|j| {
            matches!(
                j.status,
                ExportJobStatus::Completed
                    | ExportJobStatus::Failed(_)
                    | ExportJobStatus::Cancelled
            )
        })
        .count();

    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("Export Queue ({})", state.export_queue.len()))
                .font(theme.font(FONT_BODY))
                .color(TEXT_STRONG),
        );
    });
    ui.add_space(4.0);

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        if queued_count > 0
            && !state.export_status.exporting
            && cm3_prop_button(ui, "Run Queue", ButtonVariant::Primary, false, theme).clicked()
        {
            actions.push(PanelAction::RunExportQueue);
        }
        if finished_count > 0
            && cm3_prop_button(ui, "Clear Finished", ButtonVariant::Normal, false, theme).clicked()
        {
            state
                .export_queue
                .retain(|j| matches!(j.status, ExportJobStatus::Queued | ExportJobStatus::Running));
        }
    });
    ui.add_space(6.0);

    let mut swap_pair: Option<(usize, usize)> = None;
    let mut remove_idx: Option<usize> = None;
    let total_jobs = state.export_queue.len();

    for (idx, job) in state.export_queue.iter().enumerate() {
        let is_running = job.status == ExportJobStatus::Running;
        let (status_label, status_color) = match &job.status {
            ExportJobStatus::Queued => ("Queued", TEXT_MUTED),
            ExportJobStatus::Running => ("Running", theme.value()),
            ExportJobStatus::Completed => ("Done", Color32::from_rgb(76, 198, 122)),
            ExportJobStatus::Failed(_) => ("Failed", DANGER_TEXT),
            ExportJobStatus::Cancelled => ("Cancelled", TEXT_MUTED),
        };
        let filename = job
            .output_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("output");

        Frame::NONE
            .fill(BG_TITLE)
            .stroke(Stroke::new(
                1.0,
                if is_running {
                    theme.accent()
                } else {
                    BORDER_ROW_SOFT
                },
            ))
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.spacing_mut().item_spacing.y = 3.0;
                ui.horizontal(|ui| {
                    let badge_galley = ui.painter().layout_no_wrap(
                        status_label.to_owned(),
                        theme.font(FONT_COMPACT),
                        status_color,
                    );
                    let badge_w = (badge_galley.size().x.ceil() + 8.0).max(52.0);
                    let label_max_w = (ui.available_width() - badge_w - 6.0).max(36.0);
                    ui.allocate_ui_with_layout(
                        Vec2::new(label_max_w, 16.0),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(&job.label)
                                        .font(theme.font(FONT_SMALL))
                                        .color(TEXT_PRIMARY),
                                )
                                .truncate(),
                            )
                            .on_hover_text(&job.label);
                        },
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(status_label)
                                .font(theme.font(FONT_COMPACT))
                                .color(status_color),
                        );
                    });
                });
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(filename)
                            .font(theme.font(FONT_COMPACT))
                            .color(TEXT_MUTED),
                    )
                    .truncate(),
                );
                if let ExportJobStatus::Failed(err) = &job.status {
                    ui.label(
                        egui::RichText::new(err)
                            .font(theme.font(FONT_COMPACT))
                            .color(DANGER_TEXT),
                    );
                }
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if !is_running {
                        if idx > 0
                            && state.export_queue[idx - 1].status != ExportJobStatus::Running
                            && cm3_prop_button(ui, "↑", ButtonVariant::Normal, false, theme)
                                .clicked()
                        {
                            swap_pair = Some((idx, idx - 1));
                        }
                        if idx + 1 < total_jobs
                            && state.export_queue[idx + 1].status != ExportJobStatus::Running
                            && cm3_prop_button(ui, "↓", ButtonVariant::Normal, false, theme)
                                .clicked()
                        {
                            swap_pair = Some((idx, idx + 1));
                        }
                        if cm3_prop_button(ui, "Remove", ButtonVariant::Danger, false, theme)
                            .clicked()
                        {
                            remove_idx = Some(idx);
                        }
                    }
                    if job.status == ExportJobStatus::Completed
                        && cm3_prop_button(ui, "Open Folder", ButtonVariant::Normal, false, theme)
                            .clicked()
                    {
                        actions.push(PanelAction::OpenOutputFolder(job.output_path.clone()));
                    }
                });
            });
        ui.add_space(4.0);
    }

    if let Some((a, b)) = swap_pair {
        state.export_queue.swap(a, b);
    }
    if let Some(idx) = remove_idx
        && idx < state.export_queue.len()
        && state.export_queue[idx].status != ExportJobStatus::Running
    {
        state.export_queue.remove(idx);
    }
}
