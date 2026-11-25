use super::property_rows::{
    EXPR_EDITOR_DEFAULT_SIZE, clear_property_expression, eval_prop_str, eval_prop_u32,
    eval_prop_value, insert_expression_snippet, lookup_session_prop_ref, panel_property_row,
};
use super::sequence_tab::sanitize_control_identifier;
use super::state::{
    begin_inspector_tx, clear_inspector_tx_state, close_existing_tx_before_start,
    header_compact_action_button, owns_inspector_tx, run_atomic_commit_outside_foreign_tx,
};
use super::*;

pub(super) fn apply_paired_scalar_scrub(
    session: &mut Session,
    target_id: &str,
    owner_key: &str,
    a: (&str, f64),
    b: (&str, f64),
    ev: ScrubEvent,
    state: &mut PanelsState,
) {
    let owns_tx = owns_inspector_tx(state, target_id, owner_key);
    if ev.cancelled {
        if owns_tx {
            if session.history.is_in_transaction() {
                let _ = session.cancel_transaction();
            }
            clear_inspector_tx_state(state);
        }
        return;
    }
    if ev.started && ev.committed {
        if owns_tx {
            if ev.changed {
                let _ = session.execute(Command::SetPropertyValue {
                    target_id: target_id.to_owned(),
                    property: a.0.to_owned(),
                    value: json!(a.1),
                });
                let _ = session.execute(Command::SetPropertyValue {
                    target_id: target_id.to_owned(),
                    property: b.0.to_owned(),
                    value: json!(b.1),
                });
                state.inspector_tx_has_edits = true;
            }
            if session.history.is_in_transaction() {
                let _ = session.commit_transaction(true);
            }
            clear_inspector_tx_state(state);
        } else {
            close_existing_tx_before_start(session, state);
            if ev.changed {
                let opened_atomic = session
                    .begin_transaction(format!("edit {owner_key}"))
                    .is_ok();
                let _ = session.execute(Command::SetPropertyValue {
                    target_id: target_id.to_owned(),
                    property: a.0.to_owned(),
                    value: json!(a.1),
                });
                let _ = session.execute(Command::SetPropertyValue {
                    target_id: target_id.to_owned(),
                    property: b.0.to_owned(),
                    value: json!(b.1),
                });
                if opened_atomic && session.history.is_in_transaction() {
                    let _ = session.commit_transaction(true);
                }
            }
        }
        begin_inspector_tx(
            session,
            state,
            target_id,
            owner_key,
            format!("edit {owner_key}"),
        );
        return;
    }
    if ev.started && !ev.committed {
        close_existing_tx_before_start(session, state);
        begin_inspector_tx(
            session,
            state,
            target_id,
            owner_key,
            format!("edit {owner_key}"),
        );
    }
    let owns_tx = owns_inspector_tx(state, target_id, owner_key);
    if ev.changed && ev.committed && !ev.started && !owns_tx {
        run_atomic_commit_outside_foreign_tx(session, state, |session| {
            let opened_atomic = session
                .begin_transaction(format!("edit {owner_key}"))
                .is_ok();
            let _ = session.execute(Command::SetPropertyValue {
                target_id: target_id.to_owned(),
                property: a.0.to_owned(),
                value: json!(a.1),
            });
            let _ = session.execute(Command::SetPropertyValue {
                target_id: target_id.to_owned(),
                property: b.0.to_owned(),
                value: json!(b.1),
            });
            if opened_atomic && session.history.is_in_transaction() {
                let _ = session.commit_transaction(true);
            }
        });
        return;
    }
    if ev.changed || (ev.committed && owns_tx) {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: target_id.to_owned(),
            property: a.0.to_owned(),
            value: json!(a.1),
        });
        let _ = session.execute(Command::SetPropertyValue {
            target_id: target_id.to_owned(),
            property: b.0.to_owned(),
            value: json!(b.1),
        });
        if owns_tx {
            state.inspector_tx_has_edits = true;
        }
    }
    if ev.committed && owns_tx {
        if session.history.is_in_transaction() {
            let _ = session.commit_transaction(true);
        }
        clear_inspector_tx_state(state);
    }
}

/// Returns `true` when `ext` is a supported OS system font extension (`.ttf`, `.otf`, `.ttc`).
pub(super) fn is_supported_system_font_ext(ext: &str) -> bool {
    ext.eq_ignore_ascii_case("ttf")
        || ext.eq_ignore_ascii_case("otf")
        || ext.eq_ignore_ascii_case("ttc")
}

/// Scans `dirs` for `.ttf` / `.otf` / `.ttc` fonts, deriving clean display names from file stems,
/// deduplicating case-insensitively, and sorting alphabetically.
pub(super) fn discover_system_fonts_in_dirs(
    dirs: Vec<std::path::PathBuf>,
) -> Vec<(String, std::path::PathBuf)> {
    use std::path::PathBuf;
    // Keyed by lowercase display name -> (display_name, path) for case-insensitive deduplication.
    let mut seen = BTreeMap::<String, (String, PathBuf)>::new();
    let mut stack = dirs;
    let mut visited_dirs = 0usize;
    while let Some(dir) = stack.pop() {
        visited_dirs += 1;
        if visited_dirs > 256 {
            break;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if stack.len() < 128 {
                    stack.push(path);
                }
                continue;
            }
            let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                continue;
            };
            if !is_supported_system_font_ext(ext) {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let display = stem.trim();
            if display.is_empty() || display.starts_with('.') {
                continue;
            }
            let key = display.to_ascii_lowercase();
            let is_ttc = ext.eq_ignore_ascii_case("ttc");
            match seen.get_mut(&key) {
                None => {
                    seen.insert(key, (display.to_owned(), path));
                }
                Some((_, existing_path)) => {
                    // Prefer standalone .ttf/.otf over .ttc when both share the exact same stem.
                    let existing_is_ttc = existing_path
                        .extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| e.eq_ignore_ascii_case("ttc"));
                    if existing_is_ttc && !is_ttc {
                        *existing_path = path;
                    }
                }
            }
        }
    }
    let mut out: Vec<(String, PathBuf)> = seen.into_values().collect();
    out.sort_by(|a, b| {
        a.0.to_ascii_lowercase()
            .cmp(&b.0.to_ascii_lowercase())
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

#[allow(clippy::too_many_arguments)]
/// Discovers installed `.ttf` / `.otf` / `.ttc` system fonts once per process (sorted, deduplicated),
/// with precomputed ASCII-lowercased display names for zero-allocation filtering.
fn installed_system_fonts() -> &'static [(String, String, std::path::PathBuf)] {
    use std::path::PathBuf;
    use std::sync::OnceLock;
    static CACHE: OnceLock<Vec<(String, String, PathBuf)>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut dirs: Vec<PathBuf> = Vec::new();
        #[cfg(target_os = "macos")]
        {
            dirs.push(PathBuf::from("/System/Library/Fonts"));
            dirs.push(PathBuf::from("/System/Library/Fonts/Supplemental"));
            dirs.push(PathBuf::from("/Library/Fonts"));
            if let Ok(home) = std::env::var("HOME") {
                dirs.push(PathBuf::from(home).join("Library/Fonts"));
            }
        }
        #[cfg(target_os = "windows")]
        {
            if let Ok(win) = std::env::var("WINDIR") {
                dirs.push(PathBuf::from(win).join("Fonts"));
            } else {
                dirs.push(PathBuf::from(r"C:\Windows\Fonts"));
            }
            if let Ok(local) = std::env::var("LOCALAPPDATA") {
                dirs.push(PathBuf::from(local).join(r"Microsoft\Windows\Fonts"));
            }
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            dirs.push(PathBuf::from("/usr/share/fonts"));
            dirs.push(PathBuf::from("/usr/local/share/fonts"));
            if let Ok(home) = std::env::var("HOME") {
                dirs.push(PathBuf::from(home).join(".local/share/fonts"));
            }
        }

        discover_system_fonts_in_dirs(dirs)
            .into_iter()
            .map(|(display, path)| {
                let lower = display.to_ascii_lowercase();
                (display, lower, path)
            })
            .collect()
    })
}

const BUILTIN_FONT_CHOICES: [(&str, &str, &str); 3] = [
    ("Geist-Regular.ttf", "Geist Regular", "geist regular"),
    ("GeistMono-Regular.ttf", "Geist Mono", "geist mono"),
    (
        "SourceCodePro-Regular.ttf",
        "Source Code Pro",
        "source code pro",
    ),
];

enum FontChoice {
    Builtin(&'static str, &'static str, &'static str),
    ProjectAsset(String, String, String),
    SystemFont(&'static str, &'static str, &'static std::path::Path),
}

impl FontChoice {
    fn label(&self) -> &str {
        match self {
            Self::Builtin(_, l, _) => l,
            Self::ProjectAsset(_, l, _) => l.as_str(),
            Self::SystemFont(l, _, _) => l,
        }
    }

    fn lower_label(&self) -> &str {
        match self {
            Self::Builtin(_, _, low) => low,
            Self::ProjectAsset(_, _, low) => low.as_str(),
            Self::SystemFont(_, low, _) => low,
        }
    }
}

fn is_font_asset_filename(filename: &str) -> bool {
    let bytes = filename.as_bytes();
    let ends_ci = |ext: &[u8]| {
        bytes.len() >= ext.len() && bytes[bytes.len() - ext.len()..].eq_ignore_ascii_case(ext)
    };
    ends_ci(b".ttf") || ends_ci(b".otf") || ends_ci(b".ttc") || ends_ci(b".woff")
}

/// Resolves the active font display label without allocating a `Vec<FontChoice>` when the
/// font selector dropdown is closed.
fn resolve_current_font_display_label(session: &Session, cur_font: &str) -> String {
    for &(key, label, _) in &BUILTIN_FONT_CHOICES {
        if key.eq_ignore_ascii_case(cur_font) {
            return label.to_owned();
        }
    }
    for asset in session.project.assets.values() {
        if !is_font_asset_filename(&asset.filename) {
            continue;
        }
        let stem = std::path::Path::new(&asset.filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&asset.filename);
        if asset.id == cur_font || stem.eq_ignore_ascii_case(cur_font) {
            return stem.to_owned();
        }
    }
    for (display, _, path) in installed_system_fonts() {
        if display.eq_ignore_ascii_case(cur_font) || path.to_str().is_some_and(|p| p == cur_font) {
            return display.clone();
        }
    }
    "Geist Regular".to_owned()
}

/// Shared Font selector for both `Text3D` and `Text2D` (bundled OFL fonts + project font assets
/// + installed OS `.ttf`/`.otf` system fonts + custom font file picker).
#[allow(clippy::too_many_arguments)]
pub(super) fn show_font_selector_row(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    id_salt: &str,
    local_frame: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let cur_font = eval_prop_str(
        lookup_session_prop_ref(session, target_id, "font"),
        local_frame,
        "Geist-Regular.ttf",
    );
    let current_label = resolve_current_font_display_label(session, &cur_font);

    let popup_id = Id::unique((target_id, id_salt, "font_search_popup"));
    let search_id = popup_id.with("search_input");
    let mut popup_open = ui
        .data_mut(|d| d.get_temp::<bool>(popup_id))
        .unwrap_or(false);
    let mut just_opened_popup = false;
    let mut btn_rect_saved = Rect::NOTHING;

    panel_property_row(
        ui,
        "Font",
        true,
        state,
        theme,
        |_| {},
        |ui| {
            let avail_w = ui.available_width();
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                let dd_w = (avail_w - 24.0).max(56.0);
                let (raw_slot, _) = ui.allocate_exact_size(Vec2::new(dd_w, 23.0), Sense::hover());
                let btn_rect = Rect::from_min_size(
                    Pos2::new(raw_slot.left().round(), (raw_slot.top() + 1.0).round()),
                    Vec2::new(dd_w.round(), 21.0),
                );
                btn_rect_saved = btn_rect;
                let btn_resp = ui
                    .interact(btn_rect, popup_id.with("btn"), Sense::click())
                    .on_hover_cursor(CursorIcon::PointingHand);
                ui.painter()
                    .rect_filled(btn_rect, CornerRadius::ZERO, BG_INPUT);
                paint_crisp_panel_border(
                    ui.painter(),
                    btn_rect,
                    if popup_open || btn_resp.hovered() {
                        theme.accent()
                    } else {
                        BORDER_CONTROL
                    },
                );
                let text_clip = Rect::from_min_max(
                    btn_rect.left_top(),
                    Pos2::new(
                        (btn_rect.right() - 18.0).max(btn_rect.left()),
                        btn_rect.bottom(),
                    ),
                );
                ui.painter().with_clip_rect(text_clip).text(
                    Pos2::new(btn_rect.left() + 5.0, btn_rect.center().y),
                    Align2::LEFT_CENTER,
                    &current_label,
                    theme.font(FONT_SMALL),
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
                if btn_resp.clicked() {
                    popup_open = !popup_open;
                    if popup_open {
                        just_opened_popup = true;
                    }
                }
                if cm3_icon_button(
                    ui,
                    Cm3Icon::Load,
                    17.0,
                    TEXT_PRIMARY,
                    "Choose custom font file...",
                    theme,
                ) {
                    actions.push(PanelAction::PickAssetForProperty {
                        target_id: target_id.to_owned(),
                        property: "font".to_owned(),
                        is_lut: false,
                    });
                }
            });
        },
    );

    if just_opened_popup {
        state.font_picker_search.clear();
    }

    let mut chosen_font: Option<FontChoice> = None;
    if popup_open {
        let sys_fonts = installed_system_fonts();
        let mut choices: Vec<FontChoice> = Vec::with_capacity(3 + sys_fonts.len());
        for &(key, label, lower) in &BUILTIN_FONT_CHOICES {
            choices.push(FontChoice::Builtin(key, label, lower));
        }

        let mut project_fonts: Vec<(String, String)> = session
            .project
            .assets
            .values()
            .filter(|asset| is_font_asset_filename(&asset.filename))
            .map(|asset| (asset.id.clone(), asset.filename.clone()))
            .collect();
        project_fonts.sort_by(|a, b| a.1.cmp(&b.1));

        let mut project_stems: BTreeSet<String> = BTreeSet::new();
        for (aid, fname) in project_fonts {
            let stem = std::path::Path::new(&fname)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&fname)
                .to_owned();
            let stem_lower = stem.to_ascii_lowercase();
            project_stems.insert(stem_lower.clone());
            choices.push(FontChoice::ProjectAsset(aid, stem, stem_lower));
        }

        for (display, display_lower, path) in sys_fonts {
            if project_stems.is_empty() || !project_stems.contains(display_lower.as_str()) {
                choices.push(FontChoice::SystemFont(
                    display.as_str(),
                    display_lower.as_str(),
                    path.as_path(),
                ));
            }
        }

        let mut font_idx = choices
            .iter()
            .position(|c| match c {
                FontChoice::Builtin(k, _, _) => k.eq_ignore_ascii_case(&cur_font),
                FontChoice::ProjectAsset(aid, lbl, _) => {
                    aid == &cur_font || lbl.eq_ignore_ascii_case(&cur_font)
                }
                FontChoice::SystemFont(lbl, _, path) => {
                    lbl.eq_ignore_ascii_case(&cur_font)
                        || path.to_str().is_some_and(|p| p == cur_font)
                }
            })
            .unwrap_or(0);
        let mut font_changed = false;

        let popup_w = 290.0_f32
            .max(btn_rect_saved.width())
            .min((viewport_rect.width() - 16.0).max(180.0));
        let popup_h = 316.0_f32.min((viewport_rect.height() - 16.0).max(140.0));
        let x = btn_rect_saved.left().clamp(
            viewport_rect.left() + 8.0,
            (viewport_rect.right() - popup_w - 8.0).max(viewport_rect.left() + 8.0),
        );
        let below_y = btn_rect_saved.bottom() + 2.0;
        let y = if below_y + popup_h <= viewport_rect.bottom() - 8.0 {
            below_y
        } else {
            (btn_rect_saved.top() - popup_h - 2.0).clamp(
                viewport_rect.top() + 8.0,
                (viewport_rect.bottom() - popup_h - 8.0).max(viewport_rect.top() + 8.0),
            )
        };
        let mut close_now = ui.input(|i| i.key_pressed(Key::Escape));
        if close_now {
            mark_floating_editor_escape_consumed(ui.ctx());
        }

        let list_h = (popup_h - 52.0).max(80.0);
        let area_resp = Area::new(popup_id)
            .order(Order::Foreground)
            .fixed_pos(Pos2::new(x.round(), y.round()))
            .interactable(true)
            .movable(false)
            .show(ui.ctx(), |ui| {
                Frame::NONE
                    .fill(BG_PANEL)
                    .stroke(Stroke::new(1.0, theme.accent()))
                    .inner_margin(egui::Margin::same(6))
                    .show(ui, |ui| {
                        ui.set_min_width(popup_w - 12.0);
                        ui.set_max_width(popup_w - 12.0);
                        ui.set_max_height(popup_h - 12.0);
                        let search_resp = ui.add(
                            TextEdit::singleline(&mut state.font_picker_search)
                                .id(search_id)
                                .hint_text("Search fonts (e.g. Impact, Geist, Hiragino)...")
                                .font(theme.font(FONT_SMALL))
                                .desired_width(ui.available_width()),
                        );
                        if just_opened_popup {
                            search_resp.request_focus();
                        }
                        ui.add_space(4.0);
                        let q = state.font_picker_search.trim().to_ascii_lowercase();
                        let filtered: Vec<usize> = if q.is_empty() {
                            (0..choices.len()).collect()
                        } else {
                            choices
                                .iter()
                                .enumerate()
                                .filter(|(_, c)| c.lower_label().contains(&q))
                                .map(|(idx, _)| idx)
                                .collect()
                        };
                        let first_match = filtered.first().copied();
                        let scroll = ScrollArea::vertical()
                            .id_salt((popup_id, "list"))
                            .min_scrolled_height(list_h)
                            .max_height(list_h)
                            .scroll_bar_visibility(ScrollBarVisibility::AlwaysVisible)
                            .auto_shrink([false, false]);
                        if filtered.is_empty() {
                            scroll.show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new("No matching fonts")
                                        .font(theme.font(FONT_SMALL))
                                        .color(TEXT_FAINT),
                                );
                            });
                        } else {
                            scroll.show_rows(ui, 24.0, filtered.len(), |ui, row_range| {
                                ui.spacing_mut().item_spacing.y = 1.0;
                                for &idx in &filtered[row_range] {
                                    let lbl = choices[idx].label();
                                    let is_sel = idx == font_idx;
                                    let (row_rect, row_resp) = ui.allocate_exact_size(
                                        Vec2::new(ui.available_width(), 24.0),
                                        Sense::click(),
                                    );
                                    let row_resp =
                                        row_resp.on_hover_cursor(CursorIcon::PointingHand);
                                    if is_sel {
                                        ui.painter().rect_filled(
                                            row_rect,
                                            CornerRadius::ZERO,
                                            theme.accent(),
                                        );
                                    } else if row_resp.hovered() {
                                        ui.painter().rect_filled(
                                            row_rect,
                                            CornerRadius::ZERO,
                                            BG_DROPDOWN_HOVER,
                                        );
                                    }
                                    ui.painter()
                                        .with_clip_rect(row_rect.shrink2(Vec2::new(4.0, 0.0)))
                                        .text(
                                            Pos2::new(row_rect.left() + 6.0, row_rect.center().y),
                                            Align2::LEFT_CENTER,
                                            lbl,
                                            theme.font(FONT_SMALL),
                                            if is_sel { TEXT_BRIGHT } else { TEXT_PRIMARY },
                                        );
                                    if row_resp.clicked() {
                                        font_idx = idx;
                                        font_changed = true;
                                        close_now = true;
                                    }
                                }
                            });
                        }
                        if ui.input(|i| i.key_pressed(Key::Enter))
                            && let Some(idx) = first_match
                        {
                            font_idx = idx;
                            font_changed = true;
                            close_now = true;
                        }
                    });
            });
        if area_resp.response.rect.is_positive()
            && ui.ctx().layer_id_at(area_resp.response.rect.center())
                != Some(area_resp.response.layer_id)
        {
            ui.ctx().request_repaint();
        }
        // Prevent any mouse-wheel / trackpad scroll over the open font popup from leaking
        // into the parent inspector ScrollArea behind it.
        if ui.input(|i| {
            i.pointer
                .hover_pos()
                .is_some_and(|p| area_resp.response.rect.contains(p))
        }) {
            ui.input_mut(|i| {
                i.smooth_scroll_delta = Vec2::ZERO;
            });
        }
        let clicked_outside = !just_opened_popup
            && ui.input(|i| {
                i.pointer.primary_pressed()
                    && i.pointer.interact_pos().is_some_and(|p| {
                        !area_resp.response.rect.contains(p) && !btn_rect_saved.contains(p)
                    })
            });
        if close_now || clicked_outside {
            popup_open = false;
        }
        if font_changed && font_idx < choices.len() {
            chosen_font = Some(choices.swap_remove(font_idx));
        }
    }
    ui.data_mut(|d| d.insert_temp(popup_id, popup_open));

    if let Some(choice) = chosen_font {
        let resolved_value: Option<String> = match choice {
            FontChoice::Builtin(key, _, _) => Some(key.to_owned()),
            FontChoice::ProjectAsset(aid, _, _) => Some(aid),
            FontChoice::SystemFont(display, _, path) => {
                // Embed the selected OS system font into `session.project.assets` so the
                // project renders cleanly and stays portable across `.zcraft` / `.pz` export.
                match std::fs::read(path) {
                    Ok(bytes) if !bytes.is_empty() && bytes.len() <= 32 * 1024 * 1024 => {
                        let filename = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(display)
                            .to_owned();
                        let existing_id = session
                            .project
                            .assets
                            .values()
                            .find(|a| a.filename == filename && a.bytes.is_some())
                            .map(|a| a.id.clone());
                        let asset_id = existing_id.unwrap_or_else(|| {
                            let id = session.project.insert_binary_asset(
                                AssetKind::Font,
                                filename.clone(),
                                bytes,
                            );
                            session.project.media.push(zc_project::MediaItem {
                                id: zc_project::Uuid::new_v4(),
                                name: filename,
                                asset_id: Some(id.clone()),
                                kind: AssetKind::Font,
                                preserved: BTreeMap::new(),
                            });
                            id
                        });
                        Some(asset_id)
                    }
                    // Large system font collections (e.g. >32 MiB CJK `.ttc`): reference by path
                    // so `zc-render`'s direct `.ttc` loader renders immediately without bloating `.zcraft`.
                    Ok(bytes) if !bytes.is_empty() => Some(path.to_string_lossy().into_owned()),
                    _ => None,
                }
            }
        };
        if let Some(val) = resolved_value {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: target_id.to_owned(),
                property: "font".to_owned(),
                value: json!(val),
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn show_session_asset_prop(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    key: &str,
    lbl: &str,
    is_lut: bool,
    local_frame: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let cur_id = eval_prop_str(
        lookup_session_prop_ref(session, target_id, key),
        local_frame,
        "",
    );

    let mut assets: Vec<(String, String)> = session
        .project
        .assets
        .values()
        .map(|a| (a.id.clone(), a.filename.clone()))
        .collect();
    assets.sort_by(|a, b| a.1.cmp(&b.1));

    let mut option_labels: Vec<String> = Vec::with_capacity(assets.len() + 1);
    option_labels.push("(none)".to_owned());
    for (_, fname) in &assets {
        option_labels.push(fname.clone());
    }
    let mut selected_idx = if cur_id.is_empty() {
        0
    } else {
        assets
            .iter()
            .position(|(id, _)| id == &cur_id)
            .map_or(0, |i| i + 1)
    };
    let label_refs: Vec<&str> = option_labels.iter().map(String::as_str).collect();

    let mut dd_changed = false;
    let mut clear_clicked = false;
    panel_property_row(
        ui,
        lbl,
        true,
        state,
        theme,
        |_| {},
        |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                dd_changed = cm3_dropdown(
                    ui,
                    Id::unique((target_id, key, "obj_asset_dd")),
                    &mut selected_idx,
                    &label_refs,
                    false,
                    viewport_rect,
                    theme,
                );
                if cm3_icon_button(
                    ui,
                    Cm3Icon::Load,
                    17.0,
                    TEXT_PRIMARY,
                    "Choose asset file...",
                    theme,
                ) {
                    actions.push(PanelAction::PickAssetForProperty {
                        target_id: target_id.to_owned(),
                        property: key.to_owned(),
                        is_lut,
                    });
                }
                if !cur_id.is_empty()
                    && cm3_icon_button(ui, Cm3Icon::Close, 15.0, TEXT_PRIMARY, "Clear asset", theme)
                {
                    clear_clicked = true;
                }
            });
        },
    );
    if clear_clicked {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: target_id.to_owned(),
            property: key.to_owned(),
            value: Value::Null,
        });
    } else if dd_changed {
        let new_val = if selected_idx == 0 {
            Value::Null
        } else {
            json!(assets[selected_idx - 1].0)
        };
        let _ = session.execute(Command::SetPropertyValue {
            target_id: target_id.to_owned(),
            property: key.to_owned(),
            value: new_val,
        });
    }
}

/// Computes Shader Editor `(pos, size)` clear of the left Inspector (`x >= 390.0`) and
/// constrained completely inside `screen`.
#[must_use]
pub(super) fn compute_shader_editor_window_rect(screen: Rect) -> (Pos2, Vec2) {
    let min_left_clear = screen.left() + 390.0;
    let max_right_avail = (screen.right() - min_left_clear - 16.0).max(320.0);
    let w = max_right_avail
        .min(560.0)
        .min((screen.width() - 16.0).max(280.0));
    let h = (screen.height() - 80.0)
        .clamp(280.0, 460.0)
        .min((screen.height() - 16.0).max(200.0));
    let max_x = (screen.right() - w - 8.0).max(screen.left() + 8.0);
    let max_y = (screen.bottom() - h - 8.0).max(screen.top() + 8.0);
    let x = min_left_clear.clamp(screen.left() + 8.0, max_x);
    let y = (screen.top() + 48.0).clamp(screen.top() + 8.0, max_y);
    (
        Pos2::new(x.round(), y.round()),
        Vec2::new(w.round(), h.round()),
    )
}

/// Strips raw Rust `Debug` noise (e.g., `Error { kind: ..., meta: Span { start: ..., end: ... } }`)
/// from GLSL/Naga compiler errors into a concise human-readable message.
#[must_use]
pub(super) fn format_shader_compile_error(raw: &str, source: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "Shader compile error: unknown syntax error.".to_owned();
    }

    // Parse Naga GLSL `Error { kind: ..., meta: Span { start: N, end: M } }` entries
    if trimmed.contains("Error {") && trimmed.contains("kind:") {
        let mut messages: Vec<String> = Vec::new();
        let mut rest = trimmed;
        while let Some(idx) = rest.find("Error {") {
            rest = &rest[idx + "Error {".len()..];
            let kind_str = if let Some(k_idx) = rest.find("kind:") {
                let after_k = rest[k_idx + "kind:".len()..].trim_start();
                let end_k = after_k
                    .find(", meta:")
                    .or_else(|| after_k.find('}'))
                    .unwrap_or(after_k.len());
                after_k[..end_k].trim()
            } else {
                ""
            };
            let cleaned_kind = clean_naga_error_kind(kind_str);
            // Try to extract byte span to compute line/column when it maps into user source
            let loc_hint = extract_span_start(rest).and_then(|start_byte| {
                // Only map directly when the span lies within the user source length
                if start_byte < source.len() {
                    let prefix = &source[..start_byte];
                    let line = prefix.bytes().filter(|&b| b == b'\n').count() + 1;
                    let col = prefix
                        .rsplit('\n')
                        .next()
                        .map_or(1, |s| s.chars().count() + 1);
                    Some(format!("line {line}, col {col}"))
                } else {
                    None
                }
            });
            let entry = match loc_hint {
                Some(loc) if !cleaned_kind.is_empty() => format!("{cleaned_kind} ({loc})"),
                _ if !cleaned_kind.is_empty() => cleaned_kind,
                _ => "Invalid GLSL syntax".to_owned(),
            };
            if !messages.contains(&entry) {
                messages.push(entry);
            }
            if messages.len() >= 3 {
                break;
            }
        }
        if !messages.is_empty() {
            return format!("Shader error: {}", messages.join("; "));
        }
    }

    // Strip redundant leading prefixes & multi-line verbose noise
    let clean = trimmed
        .strip_prefix("GLSL 450 parse error:")
        .or_else(|| trimmed.strip_prefix("Naga shader validation error:"))
        .unwrap_or(trimmed)
        .trim();
    let first_line = clean
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or(clean);
    format!("Shader error: {first_line}")
}

fn extract_span_start(fragment: &str) -> Option<usize> {
    let span_idx = fragment.find("Span {")?;
    let after_span = &fragment[span_idx + "Span {".len()..];
    let start_idx = after_span.find("start:")?;
    let digits: String = after_span[start_idx + "start:".len()..]
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<usize>().ok()
}

fn clean_naga_error_kind(kind: &str) -> String {
    let k = kind.trim();
    if let Some(inner) = k
        .strip_prefix("SemanticError(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return inner.trim_matches('"').to_owned();
    }
    if let Some(inner) = k
        .strip_prefix("UnknownVariable(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let name =
            extract_quoted_token(inner).unwrap_or_else(|| inner.trim_matches('"').to_owned());
        return format!("Unknown variable `{name}`");
    }
    if let Some(inner) = k
        .strip_prefix("UnknownField(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let name =
            extract_quoted_token(inner).unwrap_or_else(|| inner.trim_matches('"').to_owned());
        return format!("Unknown field `{name}`");
    }
    if let Some(inner) = k
        .strip_prefix("UnknownType(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let name =
            extract_quoted_token(inner).unwrap_or_else(|| inner.trim_matches('"').to_owned());
        return format!("Unknown type `{name}`");
    }
    if let Some(inner) = k
        .strip_prefix("UnknownFunction(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let name =
            extract_quoted_token(inner).unwrap_or_else(|| inner.trim_matches('"').to_owned());
        return format!("Unknown function `{name}`");
    }
    if let Some(inner) = k
        .strip_prefix("VariableAlreadyDeclared(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let name =
            extract_quoted_token(inner).unwrap_or_else(|| inner.trim_matches('"').to_owned());
        return format!("Variable `{name}` already declared");
    }
    if let Some(inner) = k
        .strip_prefix("InvalidToken(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let tok = extract_quoted_token(inner).unwrap_or_else(|| inner.to_owned());
        return format!("Unexpected token `{tok}`");
    }
    if let Some(inner) = k
        .strip_prefix("ExpectedToken(")
        .and_then(|s| s.strip_suffix(')'))
    {
        let cleaned = inner
            .replace("Word(", "")
            .replace(')', "")
            .replace('"', "`");
        return format!("Syntax error (expected {cleaned})");
    }
    if let Some(inner) = k
        .strip_prefix("PreprocessorError(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return format!("Preprocessor error: {}", inner.trim_matches('"'));
    }
    if k == "EndOfFile" {
        return "Unexpected end of shader (check missing `;` or `}`)".to_owned();
    }
    // Generic balanced formatting for `VariantName("value")` or `VariantName(...)`
    if let Some(open_idx) = k.find('(')
        && let Some(inner) = k[open_idx + 1..].strip_suffix(')')
    {
        let variant = humanize_prop_label(&k[..open_idx]);
        if let Some(tok) = extract_quoted_token(inner) {
            return format!("{variant} `{tok}`");
        }
        let cleaned_inner = inner
            .replace("Word(", "")
            .replace(['(', ')'], "")
            .replace('"', "`");
        return format!("{variant}: {cleaned_inner}");
    }
    k.replace("Word(", "")
        .replace(['(', ')'], "")
        .replace('"', "`")
}

fn extract_quoted_token(s: &str) -> Option<String> {
    let start = s.find('"')? + 1;
    let end = s[start..].find('"')? + start;
    Some(s[start..end].to_owned())
}

// ---------------------------------------------------------------------------
// Inspector Floating Windows (Custom Shader GLSL Editor + Expression Editor)
// ---------------------------------------------------------------------------

pub(super) fn show_inspector_floating_windows(
    ctx: &egui::Context,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
) {
    let ctx_screen = ctx.input(|i| i.viewport_rect());
    let screen = if ctx_screen.width() > 100.0 && ctx_screen.height() > 100.0 {
        ctx_screen
    } else {
        viewport_rect
    };

    // 1. Custom Shader GLSL Editor (Spec 01 §4.6 Shader+ zoomable code editor)
    if let Some(mut editor) = state.shader_editor.take() {
        if !target_exists_in_session(session, &editor.effect_id) {
            state.floating_windows.close("shader_editor");
        } else {
            let (shader_pos, shader_size) = compute_shader_editor_window_rect(screen);
            let spec = FloatingWindowSpec::new("shader_editor", "Shader Editor")
                .with_subtitle(editor.effect_name.clone())
                .with_default_size(shader_size);
            if !state.floating_windows.is_open("shader_editor") {
                state.floating_windows.open_or_focus(&spec, screen);
                state.shader_editor_just_opened = true;
            }
            if let Some(win) = state.floating_windows.windows.get_mut("shader_editor")
                && state.shader_editor_just_opened
            {
                win.pos = shader_pos;
                win.size = shader_size;
                win.restore_pos = shader_pos;
                win.restore_size = shader_size;
                win.collapsed = false;
                win.maximized = false;
                state.shader_editor_just_opened = false;
            }
            clamp_floating_window_in_screen(&mut state.floating_windows, &spec, screen);
            let is_focused = state.floating_windows.focused_id.as_deref() == Some("shader_editor");
            let mut apply_code = false;
            let ev = cm3_floating_window(
                ctx,
                &mut state.floating_windows,
                &spec,
                screen,
                theme,
                |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("GLSL ES 1.00")
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_MUTED),
                        );
                        if cm3_prop_button(ui, "A-", ButtonVariant::Normal, false, theme).clicked()
                        {
                            editor.font_px = (editor.font_px - 2.0).clamp(10.0, 24.0);
                        }
                        if cm3_prop_button(ui, "A+", ButtonVariant::Normal, false, theme).clicked()
                        {
                            editor.font_px = (editor.font_px + 2.0).clamp(10.0, 24.0);
                        }
                        egui::ComboBox::from_id_salt(Id::unique("shader_starter_templates"))
                            .selected_text("Templates")
                            .width(116.0)
                            .show_ui(ui, |ui| {
                                for (name, src) in SHADER_PLUS_STARTER_TEMPLATES {
                                    if ui.selectable_label(false, *name).clicked() {
                                        editor.code = (*src).to_owned();
                                        editor.error_message = None;
                                    }
                                }
                            });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let apply_lbl = if cfg!(target_os = "macos") {
                                "Apply (Cmd+Enter)"
                            } else {
                                "Apply (Ctrl+Enter)"
                            };
                            if cm3_prop_button(ui, apply_lbl, ButtonVariant::Primary, false, theme)
                                .clicked()
                            {
                                apply_code = true;
                            }
                        });
                    });

                    if is_focused
                        && ui.input(|i| {
                            (i.modifiers.ctrl || i.modifiers.command) && i.key_pressed(Key::Enter)
                        })
                    {
                        apply_code = true;
                    }

                    if let Some(err) = &editor.error_message {
                        warning_banner(ui, err, theme);
                    }

                    ui.add_space(4.0);
                    let scroll_out = ScrollArea::vertical()
                        .id_salt("shader_code_scroll")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.add(
                                TextEdit::multiline(&mut editor.code)
                                    .font(egui::FontId::monospace(editor.font_px))
                                    .text_color(TEXT_BRIGHT)
                                    .desired_width(ui.available_width())
                                    .desired_rows(18)
                                    .code_editor(),
                            );
                        });
                    if crate::ui::gestures::unoccluded_hover_pos(ui, scroll_out.inner_rect)
                        .is_some()
                    {
                        let zoom = ui.input(|i| i.zoom_delta());
                        if zoom.is_finite() && (zoom - 1.0).abs() > 1e-4 {
                            editor.font_px = (editor.font_px * zoom).clamp(10.0, 24.0);
                        }
                    }
                },
            );

            if apply_code {
                let mut compile_opts = CompileOptions::default();
                for ((fx_id, prop_key), &kind_idx) in &state.shader_prop_kinds {
                    if fx_id == &editor.effect_id && kind_idx == 9 {
                        let v = eval_prop_u32(
                            lookup_session_prop_ref(session, &editor.effect_id, prop_key),
                            session.playback.current_frame,
                            0,
                        );
                        compile_opts.defines.insert(prop_key.clone(), v.to_string());
                    }
                }
                match translate_glsl_es100_with_options(&editor.code, &compile_opts) {
                    Ok(_) => {
                        match session.execute(Command::SetPropertyValue {
                            target_id: editor.effect_id.clone(),
                            property: editor.property.clone(),
                            value: json!(editor.code),
                        }) {
                            Ok(_) => editor.error_message = None,
                            Err(err) => {
                                editor.error_message =
                                    Some(format!("Failed to apply shader: {err}"));
                            }
                        }
                    }
                    Err(err) => {
                        editor.error_message =
                            Some(format_shader_compile_error(&err.to_string(), &editor.code));
                    }
                }
            }
            clamp_floating_window_in_screen(&mut state.floating_windows, &spec, screen);
            let mut close_shader = ev.closed;
            if ev.closed && ctx.input(|i| i.key_pressed(Key::Escape)) {
                mark_floating_editor_escape_consumed(ctx);
            } else if !ev.closed
                && consume_escape_for_floating_editor(ctx, &state.floating_windows, "shader_editor")
            {
                state.floating_windows.close("shader_editor");
                close_shader = true;
            }
            if !close_shader {
                state.shader_editor = Some(editor);
            }
        }
    }

    // 2. Expression Editor Floating Window
    if let Some(mut expr) = state.expr_editor.take() {
        if !target_exists_in_session(session, &expr.target_id) {
            state.floating_windows.close("expression_editor");
        } else {
            let (default_pos, default_size) = compute_clamped_editor_window_rect(
                screen,
                EXPR_EDITOR_DEFAULT_SIZE.x,
                EXPR_EDITOR_DEFAULT_SIZE.y,
            );
            let mut spec = FloatingWindowSpec::new("expression_editor", "Expression Editor")
                .with_subtitle(expr.property.clone())
                .with_default_size(default_size);
            spec.min_size = Vec2::new(340.0, 260.0);
            if !state.floating_windows.is_open("expression_editor") {
                state.floating_windows.open_or_focus(&spec, screen);
                state.expr_editor_just_opened = true;
            }
            if let Some(win) = state.floating_windows.windows.get_mut("expression_editor")
                && state.expr_editor_just_opened
            {
                win.size = default_size;
                win.pos = default_pos;
                win.restore_size = default_size;
                win.restore_pos = default_pos;
                win.collapsed = false;
                win.maximized = false;
                state.expr_editor_just_opened = false;
            }
            clamp_floating_window_in_screen(&mut state.floating_windows, &spec, screen);
            let is_focused =
                state.floating_windows.focused_id.as_deref() == Some("expression_editor");
            let mut commit_expr = false;
            let mut clear_expr = false;
            let mut close_expr = false;
            let live_diag = evaluate_expression_live_preview_cached(
                session,
                state,
                &expr.target_id,
                &expr.property,
                &expr.code,
            );
            let has_syntax_err = live_diag.is_err();
            let ev = cm3_floating_window(
                ctx,
                &mut state.floating_windows,
                &spec,
                screen,
                theme,
                |ui| {
                    // Quick snippet insertion bar
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = Vec2::new(4.0, 3.0);
                        ui.label(
                            egui::RichText::new("Snippets:")
                                .font(theme.font(FONT_COMPACT))
                                .color(TEXT_MUTED),
                        );
                        for (lbl, snippet) in [
                            ("sin/time", "Math.sin(time * 2.0) * 0.5 + 0.5"),
                            ("wiggle", "wiggle(2, 15)"),
                            ("loopOut", "loopOut(\"cycle\")"),
                            ("smoothstep", "smooth(0, 2, time)"),
                            ("control(...)", "control(\"ctl_slider_1\")"),
                            ("audioLevel", "audioLevel(0, 1) * 100"),
                        ] {
                            if ui
                                .add(
                                    egui::Button::new(
                                        egui::RichText::new(lbl)
                                            .font(theme.font(FONT_COMPACT))
                                            .color(TEXT_PRIMARY),
                                    )
                                    .fill(BG_BUTTON)
                                    .stroke(Stroke::new(1.0, BORDER_CONTROL)),
                                )
                                .clicked()
                            {
                                expr.code = insert_expression_snippet(&expr.code, snippet);
                                expr.error_message = None;
                            }
                        }
                    });
                    ui.add_space(4.0);

                    if let Some(err) = &expr.error_message {
                        warning_banner(ui, err, theme);
                        ui.add_space(4.0);
                    }

                    // Reserve fixed footer space so Live/Diagnostic + Apply/Clear/Close NEVER clip.
                    let footer_h = 62.0;
                    let editor_h = (ui.available_height() - footer_h - 6.0).max(56.0);
                    ScrollArea::vertical()
                        .id_salt("expr_editor_code_scroll")
                        .max_height(editor_h)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.add_sized(
                                Vec2::new(ui.available_width(), editor_h.max(56.0)),
                                TextEdit::multiline(&mut expr.code)
                                    .font(egui::FontId::monospace(12.5))
                                    .text_color(theme.value())
                                    .code_editor(),
                            );
                        });
                    if is_focused
                        && ui.input(|i| {
                            (i.modifiers.ctrl || i.modifiers.command) && i.key_pressed(Key::Enter)
                        })
                    {
                        commit_expr = true;
                    }

                    // Fixed bottom footer (diagnostic + actions)
                    ui.add_space(4.0);
                    ui.separator();
                    ui.add_space(3.0);
                    match &live_diag {
                        Ok(val_str) => {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(format!(
                                        "Live @ frame {:.1}: {val_str}",
                                        session.playback.current_frame
                                    ))
                                    .font(theme.font(FONT_SMALL))
                                    .color(Color32::from_rgb(76, 198, 122)),
                                )
                                .truncate(),
                            );
                        }
                        Err(diag) => {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(format!("Diagnostic: {diag}"))
                                        .font(theme.font(FONT_SMALL))
                                        .color(DANGER_TEXT),
                                )
                                .truncate(),
                            )
                            .on_hover_text(diag);
                        }
                    }
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        let apply_resp = ui
                            .add_enabled_ui(!has_syntax_err, |ui| {
                                cm3_prop_button(ui, "Apply", ButtonVariant::Primary, false, theme)
                            })
                            .inner;
                        if has_syntax_err {
                            apply_resp.on_disabled_hover_text("Fix syntax error before applying");
                        } else if apply_resp.clicked() {
                            commit_expr = true;
                        }
                        if cm3_prop_button(
                            ui,
                            "Clear Expression",
                            ButtonVariant::Danger,
                            false,
                            theme,
                        )
                        .clicked()
                        {
                            clear_expr = true;
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if cm3_prop_button(ui, "Close", ButtonVariant::Normal, false, theme)
                                .clicked()
                            {
                                close_expr = true;
                            }
                        });
                    });
                },
            );
            clamp_floating_window_in_screen(&mut state.floating_windows, &spec, screen);
            if ev.closed && ctx.input(|i| i.key_pressed(Key::Escape)) {
                mark_floating_editor_escape_consumed(ctx);
            } else if !ev.closed
                && !close_expr
                && consume_escape_for_floating_editor(
                    ctx,
                    &state.floating_windows,
                    "expression_editor",
                )
            {
                close_expr = true;
            }

            if commit_expr {
                if let Err(diag) = &live_diag {
                    expr.error_message = Some(format!("Fix syntax error before applying: {diag}"));
                } else {
                    match session.execute(Command::SetExpression {
                        target_id: expr.target_id.clone(),
                        property: expr.property.clone(),
                        expression: Some(expr.code.clone()),
                    }) {
                        Ok(_) => expr.error_message = None,
                        Err(err) => {
                            expr.error_message = Some(format!("Failed to apply expression: {err}"));
                        }
                    }
                }
            } else if clear_expr {
                match clear_property_expression(session, &expr.target_id, &expr.property) {
                    Ok(()) => {
                        state.floating_windows.close("expression_editor");
                    }
                    Err(err) => {
                        expr.error_message = Some(format!("Failed to clear expression: {err}"));
                        clear_expr = false;
                    }
                }
            } else if close_expr {
                state.floating_windows.close("expression_editor");
            }
            if !ev.closed && !clear_expr && !close_expr {
                state.expr_editor = Some(expr);
            }
        }
    }

    // 3. Pro Interactive Editor Windows (Node Graph Editor & Color Looks Setup)
    show_pro_editor_windows(
        ctx,
        viewport_rect,
        &mut state.floating_windows,
        &mut state.node_graph_editor,
        &mut state.looks_editor,
        session,
        theme,
    );
}

fn target_exists_in_session(session: &Session, target_id: &str) -> bool {
    if target_id == "sequence" {
        return true;
    }
    for track in session
        .document()
        .video_tracks
        .iter()
        .chain(session.document().audio_tracks.iter())
    {
        for clip in &track.clips {
            if clip.id == target_id
                || clip.objects_3d.iter().any(|o| o.id == target_id)
                || clip.effects.iter().any(|e| e.id == target_id)
            {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Small UI Helpers
// ---------------------------------------------------------------------------

/// Paints a razor-sharp integer-snapped 1px border using 4 filled strips (zero AA feathering).
pub(super) fn paint_crisp_panel_border(painter: &Painter, rect: Rect, color: Color32) {
    let min_x = rect.min.x.round();
    let min_y = rect.min.y.round();
    let max_x = rect.max.x.round().max(min_x);
    let max_y = rect.max.y.round().max(min_y);
    if max_x - min_x < 1.0 || max_y - min_y < 1.0 {
        return;
    }
    painter.rect_filled(
        Rect::from_min_max(Pos2::new(min_x, min_y), Pos2::new(max_x, min_y + 1.0)),
        CornerRadius::ZERO,
        color,
    );
    painter.rect_filled(
        Rect::from_min_max(Pos2::new(min_x, max_y - 1.0), Pos2::new(max_x, max_y)),
        CornerRadius::ZERO,
        color,
    );
    if max_y - min_y > 2.0 {
        painter.rect_filled(
            Rect::from_min_max(
                Pos2::new(min_x, min_y + 1.0),
                Pos2::new(min_x + 1.0, max_y - 1.0),
            ),
            CornerRadius::ZERO,
            color,
        );
        painter.rect_filled(
            Rect::from_min_max(
                Pos2::new(max_x - 1.0, min_y + 1.0),
                Pos2::new(max_x, max_y - 1.0),
            ),
            CornerRadius::ZERO,
            color,
        );
    }
}

/// Paints a razor-sharp integer-snapped 2px border using 4 filled strips (zero AA feathering).
pub(super) fn paint_crisp_border_2px(painter: &Painter, rect: Rect, color: Color32) {
    let min_x = rect.min.x.round();
    let min_y = rect.min.y.round();
    let max_x = rect.max.x.round().max(min_x);
    let max_y = rect.max.y.round().max(min_y);
    if max_x - min_x < 2.0 || max_y - min_y < 2.0 {
        return;
    }
    painter.rect_filled(
        Rect::from_min_max(Pos2::new(min_x, min_y), Pos2::new(max_x, min_y + 2.0)),
        CornerRadius::ZERO,
        color,
    );
    painter.rect_filled(
        Rect::from_min_max(Pos2::new(min_x, max_y - 2.0), Pos2::new(max_x, max_y)),
        CornerRadius::ZERO,
        color,
    );
    if max_y - min_y > 4.0 {
        painter.rect_filled(
            Rect::from_min_max(
                Pos2::new(min_x, min_y + 2.0),
                Pos2::new(min_x + 2.0, max_y - 2.0),
            ),
            CornerRadius::ZERO,
            color,
        );
        painter.rect_filled(
            Rect::from_min_max(
                Pos2::new(max_x - 2.0, min_y + 2.0),
                Pos2::new(max_x, max_y - 2.0),
            ),
            CornerRadius::ZERO,
            color,
        );
    }
}

/// Edge-to-edge 43px #222222 top header bar (42px body + 1px #242424 bottom divider) and 18px
/// #cccccc title matching Zoidium Export, Restore, Plugins, Assets, About, and Settings panels.
pub(super) fn panel_top_header(ui: &mut Ui, title: &str, theme: &Cm3Theme) {
    let (hdr_rect, _) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 43.0), Sense::hover());
    ui.painter()
        .rect_filled(hdr_rect, CornerRadius::ZERO, BG_TITLE);
    let div_y = (hdr_rect.bottom() - 1.0).round();
    ui.painter().rect_filled(
        Rect::from_min_max(
            Pos2::new(hdr_rect.left(), div_y),
            Pos2::new(hdr_rect.right(), div_y + 1.0),
        ),
        CornerRadius::ZERO,
        BORDER_ROW_SOFT,
    );
    ui.painter().text(
        Pos2::new(hdr_rect.left() + 9.0, hdr_rect.center().y + 0.5),
        Align2::LEFT_CENTER,
        title,
        theme.font(FONT_TITLE),
        TEXT_PRIMARY,
    );
}

/// Compact 10x6 collapse chevron triangle (pointing down when open, right when collapsed).
pub(super) fn paint_small_chevron(painter: &Painter, center: Pos2, open: bool, color: Color32) {
    let pts = if open {
        vec![
            Pos2::new(center.x - 5.0, center.y - 3.0),
            Pos2::new(center.x + 5.0, center.y - 3.0),
            Pos2::new(center.x, center.y + 3.0),
        ]
    } else {
        vec![
            Pos2::new(center.x - 3.0, center.y - 5.0),
            Pos2::new(center.x + 3.0, center.y),
            Pos2::new(center.x - 3.0, center.y + 5.0),
        ]
    };
    painter.add(Shape::convex_polygon(pts, color, Stroke::NONE));
}

/// Converts camelCase / snake_case / dot-separated property keys into Sentence case UI labels.
pub(super) fn humanize_prop_label(key: &str) -> String {
    let trimmed = key.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    match trimmed {
        "fov" => return "FOV".to_owned(),
        "lut" => return "LUT".to_owned(),
        "blending" => return "Blending mode".to_owned(),
        "frameRate" => return "Frame rate".to_owned(),
        _ => {}
    }

    let mut words = String::with_capacity(trimmed.len() + 6);
    let mut prev_lower = false;
    let mut prev_space = true;
    for ch in trimmed.chars() {
        if ch == '_' || ch == '.' || ch == '-' {
            if !prev_space {
                words.push(' ');
                prev_space = true;
            }
            prev_lower = false;
        } else if ch.is_ascii_uppercase() {
            if prev_lower && !prev_space {
                words.push(' ');
            }
            words.push(ch.to_ascii_lowercase());
            prev_lower = false;
            prev_space = false;
        } else {
            words.push(ch);
            prev_lower = ch.is_ascii_lowercase() || ch.is_ascii_digit();
            prev_space = ch.is_whitespace();
        }
    }

    let mut chars = words.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_ascii_uppercase(), chars.as_str()),
        None => String::new(),
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct PropertyHeaderActions {
    pub(super) copy_json: bool,
    pub(super) paste_json: bool,
    pub(super) reset: bool,
    pub(super) duplicate: bool,
    pub(super) remove: bool,
}

/// Collapsible inspector root header with right-aligned Copy/Paste JSON + Reset (`↺`),
/// Duplicate (`⧉`), and Remove (`×`) buttons and clean title ellipsizing.
pub(super) fn property_header_with_actions(
    ui: &mut Ui,
    id: Id,
    title: &str,
    _col_ratio: f32,
    theme: &Cm3Theme,
) -> (bool, PropertyHeaderActions) {
    let mut open = ui.data_mut(|d| d.get_temp::<bool>(id)).unwrap_or(true);
    let mut act = PropertyHeaderActions::default();

    let (hdr_rect, hdr_resp) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 27.0), Sense::click());
    ui.painter()
        .rect_filled(hdr_rect, CornerRadius::ZERO, BG_TITLE);
    ui.painter().line_segment(
        [
            Pos2::new(hdr_rect.left(), hdr_rect.bottom() - 0.5),
            Pos2::new(hdr_rect.right(), hdr_rect.bottom() - 0.5),
        ],
        Stroke::new(1.0, BORDER_ROW_SOFT),
    );

    // Reserve right-aligned action cluster first so title text never collides with buttons
    let icon_y = (hdr_rect.center().y - 7.5).round();
    let rm_x = (hdr_rect.right() - 21.0).round();
    let dup_x = rm_x - 21.0;
    let reset_x = dup_x - 21.0;
    let reset_icon_rect = Rect::from_min_size(Pos2::new(reset_x, icon_y), Vec2::splat(15.0));
    let dup_icon_rect = Rect::from_min_size(Pos2::new(dup_x, icon_y), Vec2::splat(15.0));
    let rm_icon_rect = Rect::from_min_size(Pos2::new(rm_x, icon_y), Vec2::splat(15.0));

    // Auto-sized crisp "Copy" / "Paste" buttons (+10px horizontal padding, 4px gap) matching Material header
    let copy_lbl = "Copy";
    let paste_lbl = "Paste";
    let copy_w = (ui
        .painter()
        .layout_no_wrap(copy_lbl.to_owned(), theme.font(FONT_COMPACT), TEXT_PRIMARY)
        .size()
        .x
        .ceil()
        + 10.0)
        .max(36.0);
    let paste_w = (ui
        .painter()
        .layout_no_wrap(paste_lbl.to_owned(), theme.font(FONT_COMPACT), TEXT_PRIMARY)
        .size()
        .x
        .ceil()
        + 10.0)
        .max(40.0);
    let btn_y = (hdr_rect.center().y - 9.0).round();
    let paste_x = (reset_x - 5.0 - paste_w).round();
    let copy_x = (paste_x - 4.0 - copy_w).round();
    let copy_rect = Rect::from_min_size(Pos2::new(copy_x, btn_y), Vec2::new(copy_w, 18.0));
    let paste_rect = Rect::from_min_size(Pos2::new(paste_x, btn_y), Vec2::new(paste_w, 18.0));

    let (copy_clicked, copy_hovered) = header_compact_action_button(
        ui,
        id.with("copy_json"),
        copy_rect,
        copy_lbl,
        "Copy JSON (Cmd+C)",
        theme,
    );
    let (paste_clicked, paste_hovered) = header_compact_action_button(
        ui,
        id.with("paste_json"),
        paste_rect,
        paste_lbl,
        "Paste JSON (Cmd+V)",
        theme,
    );

    let reset_resp = ui
        .interact(
            reset_icon_rect.expand(2.5),
            id.with("reset"),
            Sense::click(),
        )
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text("Reset");
    let dup_resp = ui
        .interact(dup_icon_rect.expand(2.5), id.with("dup"), Sense::click())
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text("Duplicate");
    let rm_resp = ui
        .interact(rm_icon_rect.expand(2.5), id.with("rm"), Sense::click())
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text("Remove");

    let any_btn_hovered = copy_hovered
        || paste_hovered
        || reset_resp.hovered()
        || dup_resp.hovered()
        || rm_resp.hovered();
    paint_cm3_tree_box(
        ui.painter(),
        Pos2::new(hdr_rect.left() + 10.5, hdr_rect.center().y + 1.0),
        open,
        if hdr_resp.hovered() && !any_btn_hovered {
            TEXT_BRIGHT
        } else {
            TEXT_MUTED
        },
    );

    let title_left = hdr_rect.left() + 21.0;
    let title_right = (copy_x - 6.0).max(title_left + 16.0);
    let max_title_w = (title_right - title_left).max(16.0);
    let mut title_galley =
        ui.painter()
            .layout_no_wrap(title.to_owned(), theme.font(FONT_BODY), TEXT_PRIMARY);
    if title_galley.size().x > max_title_w {
        let chars: Vec<char> = title.chars().collect();
        let mut len = chars.len();
        while len > 0 {
            len -= 1;
            let candidate = format!("{}…", chars[..len].iter().collect::<String>().trim_end());
            let g = ui
                .painter()
                .layout_no_wrap(candidate, theme.font(FONT_BODY), TEXT_PRIMARY);
            if g.size().x <= max_title_w || len == 0 {
                title_galley = g;
                break;
            }
        }
    }
    let title_clip = Rect::from_min_max(
        Pos2::new(title_left - 1.0, hdr_rect.top()),
        Pos2::new(title_right, hdr_rect.bottom()),
    );
    ui.painter().with_clip_rect(title_clip).galley(
        Pos2::new(
            title_left,
            hdr_rect.center().y - title_galley.size().y * 0.5,
        ),
        title_galley,
        TEXT_PRIMARY,
    );

    paint_cm3_icon(
        ui.painter(),
        reset_icon_rect,
        Cm3Icon::Reset,
        if reset_resp.hovered() {
            TEXT_BRIGHT
        } else {
            TEXT_PRIMARY
        },
    );
    paint_cm3_icon(
        ui.painter(),
        dup_icon_rect,
        Cm3Icon::Duplicate,
        if dup_resp.hovered() {
            TEXT_BRIGHT
        } else {
            TEXT_PRIMARY
        },
    );
    paint_cm3_icon(
        ui.painter(),
        rm_icon_rect,
        Cm3Icon::Close,
        if rm_resp.hovered() {
            DANGER_TEXT
        } else {
            TEXT_PRIMARY
        },
    );

    act.copy_json = copy_clicked;
    act.paste_json = paste_clicked;
    act.reset = reset_resp.clicked();
    act.duplicate = dup_resp.clicked();
    act.remove = rm_resp.clicked();
    if hdr_resp.clicked()
        && !act.copy_json
        && !act.paste_json
        && !act.reset
        && !act.duplicate
        && !act.remove
    {
        open = !open;
    }
    ui.data_mut(|d| d.insert_temp(id, open));
    (open, act)
}

pub(super) fn section_banner(ui: &mut Ui, title: &str, theme: &Cm3Theme) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 24.0), Sense::hover());
    ui.painter().rect_filled(rect, CornerRadius::ZERO, BG_TITLE);
    ui.painter().text(
        Pos2::new(rect.left() + 6.0, rect.center().y),
        Align2::LEFT_CENTER,
        title,
        theme.font(FONT_BANNER),
        TEXT_FAINT,
    );
}

pub(super) fn warning_banner(ui: &mut Ui, message: &str, theme: &Cm3Theme) {
    let resp = Frame::NONE
        .fill(WARNING_SURFACE)
        .inner_margin(egui::Margin {
            left: 10,
            right: 8,
            top: 6,
            bottom: 6,
        })
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.label(
                egui::RichText::new(message)
                    .font(theme.font(FONT_SMALL))
                    .color(DANGER_TEXT),
            );
        });
    let r = resp.response.rect;
    let bar = Rect::from_min_size(r.left_top(), Vec2::new(3.0, r.height()));
    ui.painter()
        .rect_filled(bar, CornerRadius::ZERO, WARNING_BAR);
}

pub(super) fn empty_panel_message(ui: &mut Ui, message: &str, theme: &Cm3Theme) {
    let avail = ui.available_rect_before_wrap();
    ui.painter().text(
        Pos2::new(avail.center().x.round(), (avail.center().y + 1.0).round()),
        Align2::CENTER_CENTER,
        message,
        theme.font(FONT_BODY),
        TEXT_EMPTY,
    );
}

#[allow(dead_code)]
fn _keep_rgb_import_used(c: Rgb) -> Color32 {
    rgb_to_color32(c)
}

// ===========================================================================
// Wave 4B Professional Inspector & Builder Modules
// ===========================================================================

pub(super) const SHADER_PLUS_STARTER_TEMPLATES: &[(&str, &str)] = &[
    (
        "Chromatic Pulse",
        r#"precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 resolution;
uniform float time;
varying vec2 vUv;
varying vec2 vUvScaled;

void main() {
    float shift = 0.004 * sin(time * 3.0);
    float r = texture2D(tDiffuse, vUvScaled + vec2(shift, 0.0)).r;
    float g = texture2D(tDiffuse, vUvScaled).g;
    float b = texture2D(tDiffuse, vUvScaled - vec2(shift, 0.0)).b;
    float a = texture2D(tDiffuse, vUvScaled).a;
    gl_FragColor = vec4(r, g, b, a);
}
"#,
    ),
    (
        "Procedural Plasma",
        r#"precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 resolution;
uniform float time;
varying vec2 vUv;
varying vec2 vUvScaled;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    vec2 p = vUv * 6.0;
    float v = sin(p.x + time) + sin(p.y - time * 1.2) + sin((p.x + p.y) * 0.7 + time);
    vec3 col = 0.5 + 0.5 * cos(v + vec3(0.0, 2.0, 4.0));
    gl_FragColor = vec4(mix(src.rgb, col, 0.45), src.a);
}
"#,
    ),
    (
        "CRT Scanlines",
        r#"precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 resolution;
varying vec2 vUv;
varying vec2 vUvScaled;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    float line = 0.85 + 0.15 * sin(vUv.y * resolution.y * 1.5);
    gl_FragColor = vec4(src.rgb * line, src.a);
}
"#,
    ),
    (
        "Radial Vignette",
        r#"precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 resolution;
varying vec2 vUv;
varying vec2 vUvScaled;

void main() {
    vec4 src = texture2D(tDiffuse, vUvScaled);
    float d = length(vUv - vec2(0.5, 0.5)) * 1.414;
    float vig = smoothstep(1.0, 0.35, d);
    gl_FragColor = vec4(src.rgb * vig, src.a);
}
"#,
    ),
];

pub(super) fn find_project_video_clip<'a>(
    session: &'a Session,
    clip_id: &str,
) -> Option<&'a zc_project::VideoClip> {
    let cid = Uuid::parse_str(clip_id).ok()?;
    for vt in &session.active_sequence().video_tracks {
        if let Some(vc) = vt.clips.iter().find(|c| c.id == cid) {
            return Some(vc);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// 10. Asset Usage Count Scanner & Live Expression Diagnostics
// ---------------------------------------------------------------------------

fn compute_asset_usage_counts(session: &Session) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let known: BTreeSet<String> = session.project.assets.keys().cloned().collect();
    if known.is_empty() {
        return counts;
    }
    fn scan_json(val: &Value, known: &BTreeSet<String>, counts: &mut BTreeMap<String, usize>) {
        match val {
            Value::String(s) => {
                if known.contains(s) {
                    *counts.entry(s.clone()).or_insert(0) += 1;
                }
            }
            Value::Array(arr) => {
                for v in arr {
                    scan_json(v, known, counts);
                }
            }
            Value::Object(map) => {
                for v in map.values() {
                    scan_json(v, known, counts);
                }
            }
            _ => {}
        }
    }

    // Scan active session document (clips, objects_3d, effects)
    for track in session
        .document()
        .video_tracks
        .iter()
        .chain(session.document().audio_tracks.iter())
    {
        for clip in &track.clips {
            for prop in clip.properties.values() {
                scan_json(&prop.default_value, &known, &mut counts);
            }
            for obj in &clip.objects_3d {
                for prop in obj.properties.values() {
                    scan_json(&prop.default_value, &known, &mut counts);
                }
            }
            for fx in &clip.effects {
                for prop in fx.properties.values() {
                    scan_json(&prop.default_value, &known, &mut counts);
                }
            }
        }
    }
    // Also scan project audio clips & compositions media fields
    for seq in
        std::iter::once(&session.project.sequence).chain(session.project.compositions.values())
    {
        for at in &seq.audio_tracks {
            for ac in &at.clips {
                if let Some(m) = &ac.media
                    && known.contains(m)
                {
                    *counts.entry(m.clone()).or_insert(0) += 1;
                }
            }
        }
    }
    counts
}

pub(super) fn compute_asset_usage_counts_cached(
    session: &Session,
    state: &mut PanelsState,
) -> std::sync::Arc<BTreeMap<String, usize>> {
    let rev = session.revision();
    if let Some((cached_rev, counts)) = &state.cached_asset_usage
        && *cached_rev == rev
    {
        return std::sync::Arc::clone(counts);
    }
    let counts = std::sync::Arc::new(compute_asset_usage_counts(session));
    state.cached_asset_usage = Some((rev, std::sync::Arc::clone(&counts)));
    counts
}

struct BorrowedExprContext<'a> {
    frame: f64,
    time: f64,
    fps: f64,
    value: zc_expr::Value,
    properties: &'a BTreeMap<String, zc_expr::Value>,
}

impl zc_expr::ExprContext for BorrowedExprContext<'_> {
    fn frame(&self) -> f64 {
        self.frame
    }

    fn time(&self) -> f64 {
        self.time
    }

    fn fps(&self) -> f64 {
        self.fps
    }

    fn value(&self) -> zc_expr::Value {
        self.value.clone()
    }

    fn properties(&self) -> BTreeMap<String, zc_expr::Value> {
        if self.properties.is_empty() {
            BTreeMap::new()
        } else {
            self.properties.clone()
        }
    }
}

fn build_expr_controls_map(session: &Session) -> BTreeMap<String, zc_expr::Value> {
    let mut properties = BTreeMap::new();
    for ctl in &session.active_sequence().controls {
        let ev = match &ctl.track.value {
            PropertyValue::Number(n) => Some(zc_expr::Value::Number(*n)),
            PropertyValue::Bool(b) => Some(zc_expr::Value::Bool(*b)),
            PropertyValue::Vec2(v) => Some(zc_expr::Value::Array(vec![
                zc_expr::Value::Number(v[0]),
                zc_expr::Value::Number(v[1]),
            ])),
            PropertyValue::Vec3(v) | PropertyValue::Color(v) => Some(zc_expr::Value::Array(vec![
                zc_expr::Value::Number(v[0]),
                zc_expr::Value::Number(v[1]),
                zc_expr::Value::Number(v[2]),
            ])),
            PropertyValue::ColorRgba(v) => Some(zc_expr::Value::Array(vec![
                zc_expr::Value::Number(v[0]),
                zc_expr::Value::Number(v[1]),
                zc_expr::Value::Number(v[2]),
                zc_expr::Value::Number(v[3]),
            ])),
            _ => None,
        };
        if let Some(val) = ev {
            properties.insert(ctl.name.clone(), val.clone());
            let clean = sanitize_control_identifier(&ctl.name, &ctl.name);
            properties.entry(clean).or_insert(val);
        }
    }
    properties
}

pub(super) fn compile_expr_cached(
    state: &mut PanelsState,
    trimmed: &str,
    mode: zc_expr::ExprMode,
) -> Result<std::sync::Arc<zc_expr::CompiledExpr>, String> {
    let is_plus = matches!(mode, zc_expr::ExprMode::ExpressionsPlus);
    let key = (is_plus, trimmed.to_owned());
    if let Some(cached) = state.cached_compiled_exprs.get(&key) {
        return cached.clone();
    }
    let result = zc_expr::compile(trimmed, mode)
        .map(std::sync::Arc::new)
        .map_err(|e| e.to_string());
    if state.cached_compiled_exprs.len() >= 128 {
        state.cached_compiled_exprs.clear();
    }
    state.cached_compiled_exprs.insert(key, result.clone());
    result
}

fn evaluate_session_expression_raw_with_state(
    session: &Session,
    mut state: Option<&mut PanelsState>,
    target_id: &str,
    property: &str,
    code: &str,
    base_val_override: Option<Value>,
) -> Result<zc_expr::Value, String> {
    let trimmed = code.trim();
    if trimmed.is_empty() {
        return Err("Expression is empty".to_owned());
    }
    let mode = if session.plugins.is_enabled("expressions-plus") {
        zc_expr::ExprMode::ExpressionsPlus
    } else {
        zc_expr::ExprMode::VanillaCm3
    };
    let compiled = if let Some(st) = state.as_deref_mut() {
        compile_expr_cached(st, trimmed, mode)?
    } else {
        std::sync::Arc::new(zc_expr::compile(trimmed, mode).map_err(|e| e.to_string())?)
    };
    let fps = session.document().fps.max(1.0);
    let frame = session.playback.current_frame;
    let cur_val = base_val_override
        .or_else(|| eval_prop_value(lookup_session_prop_ref(session, target_id, property), frame))
        .map(|v| json_to_expr_value(&v))
        .unwrap_or(zc_expr::Value::Number(0.0));
    let owned_controls;
    let controls_ref: &BTreeMap<String, zc_expr::Value> = if let Some(st) = state {
        let rev = session.revision();
        if !st
            .cached_expr_controls
            .as_ref()
            .is_some_and(|(r, _)| *r == rev)
        {
            st.cached_expr_controls = Some((rev, build_expr_controls_map(session)));
        }
        &st.cached_expr_controls.as_ref().expect("just populated").1
    } else {
        owned_controls = build_expr_controls_map(session);
        &owned_controls
    };
    let ctx = BorrowedExprContext {
        frame,
        time: frame / fps,
        fps,
        value: cur_val,
        properties: controls_ref,
    };
    let cfg = zc_expr::EvalConfig {
        mode,
        max_steps: 200_000,
        max_recursion: 64,
    };
    compiled.evaluate(&ctx, &cfg).map_err(|e| e.to_string())
}

fn evaluate_session_expression_raw(
    session: &Session,
    target_id: &str,
    property: &str,
    code: &str,
    base_val_override: Option<Value>,
) -> Result<zc_expr::Value, String> {
    evaluate_session_expression_raw_with_state(
        session,
        None,
        target_id,
        property,
        code,
        base_val_override,
    )
}

fn expr_value_to_f64(v: &zc_expr::Value) -> Option<f64> {
    match v {
        zc_expr::Value::Number(n) if n.is_finite() => Some(*n),
        zc_expr::Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        zc_expr::Value::Array(arr) => arr.first().and_then(expr_value_to_f64),
        _ => None,
    }
}

pub(super) fn eval_session_expr_f64(
    session: &Session,
    state: &mut PanelsState,
    target_id: &str,
    property: &str,
    code: &str,
    base: f64,
) -> Option<f64> {
    let out = evaluate_session_expression_raw_with_state(
        session,
        Some(state),
        target_id,
        property,
        code,
        Some(json!(base)),
    )
    .ok()?;
    expr_value_to_f64(&out)
}

pub(super) fn eval_session_expr_vec2(
    session: &Session,
    state: &mut PanelsState,
    target_id: &str,
    property: &str,
    code: &str,
    base: [f64; 2],
) -> Option<[f64; 2]> {
    let out = evaluate_session_expression_raw_with_state(
        session,
        Some(state),
        target_id,
        property,
        code,
        Some(json!(base)),
    )
    .ok()?;
    match &out {
        zc_expr::Value::Array(arr) => Some([
            arr.first().and_then(expr_value_to_f64).unwrap_or(base[0]),
            arr.get(1).and_then(expr_value_to_f64).unwrap_or(base[1]),
        ]),
        _ => expr_value_to_f64(&out).map(|s| [s, s]),
    }
}

pub(super) fn eval_session_expr_vec3(
    session: &Session,
    state: &mut PanelsState,
    target_id: &str,
    property: &str,
    code: &str,
    base: [f64; 3],
) -> Option<[f64; 3]> {
    let out = evaluate_session_expression_raw_with_state(
        session,
        Some(state),
        target_id,
        property,
        code,
        Some(json!(base)),
    )
    .ok()?;
    match &out {
        zc_expr::Value::Array(arr) => Some([
            arr.first().and_then(expr_value_to_f64).unwrap_or(base[0]),
            arr.get(1).and_then(expr_value_to_f64).unwrap_or(base[1]),
            arr.get(2).and_then(expr_value_to_f64).unwrap_or(base[2]),
        ]),
        _ => expr_value_to_f64(&out).map(|s| [s, s, s]),
    }
}

pub(super) fn eval_session_expr_vec4(
    session: &Session,
    state: &mut PanelsState,
    target_id: &str,
    property: &str,
    code: &str,
    base: [f64; 4],
) -> Option<[f64; 4]> {
    let out = evaluate_session_expression_raw_with_state(
        session,
        Some(state),
        target_id,
        property,
        code,
        Some(json!(base)),
    )
    .ok()?;
    match &out {
        zc_expr::Value::Array(arr) => Some([
            arr.first().and_then(expr_value_to_f64).unwrap_or(base[0]),
            arr.get(1).and_then(expr_value_to_f64).unwrap_or(base[1]),
            arr.get(2).and_then(expr_value_to_f64).unwrap_or(base[2]),
            arr.get(3).and_then(expr_value_to_f64).unwrap_or(base[3]),
        ]),
        _ => expr_value_to_f64(&out).map(|s| [s, s, s, s]),
    }
}

fn evaluate_expression_live_preview_cached(
    session: &Session,
    state: &mut PanelsState,
    target_id: &str,
    property: &str,
    code: &str,
) -> Result<String, String> {
    let out = evaluate_session_expression_raw_with_state(
        session,
        Some(state),
        target_id,
        property,
        code,
        None,
    )?;
    Ok(format_expr_value(&out))
}

#[cfg(test)]
pub(super) fn evaluate_expression_live_preview(
    session: &Session,
    target_id: &str,
    property: &str,
    code: &str,
) -> Result<String, String> {
    let out = evaluate_session_expression_raw(session, target_id, property, code, None)?;
    Ok(format_expr_value(&out))
}

fn json_to_expr_value(v: &Value) -> zc_expr::Value {
    match v {
        Value::Bool(b) => zc_expr::Value::Bool(*b),
        Value::Number(n) => zc_expr::Value::Number(n.as_f64().unwrap_or(0.0)),
        Value::String(s) => zc_expr::Value::String(s.clone()),
        Value::Array(arr) => zc_expr::Value::Array(arr.iter().map(json_to_expr_value).collect()),
        _ => zc_expr::Value::Number(0.0),
    }
}

fn format_expr_value(v: &zc_expr::Value) -> String {
    match v {
        zc_expr::Value::Number(n) => {
            if (n.fract()).abs() < 1e-6 {
                format!("{}", n.round() as i64)
            } else {
                format!("{n:.4}")
            }
        }
        zc_expr::Value::Bool(b) => b.to_string(),
        zc_expr::Value::String(s) => format!("\"{s}\""),
        zc_expr::Value::Array(arr) => {
            let parts: Vec<String> = arr.iter().map(format_expr_value).collect();
            format!("[{}]", parts.join(", "))
        }
        zc_expr::Value::Null => "null".to_owned(),
        zc_expr::Value::Undefined => "undefined".to_owned(),
        zc_expr::Value::Object(_) => "[Object]".to_owned(),
    }
}
