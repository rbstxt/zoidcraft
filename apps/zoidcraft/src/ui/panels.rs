//! Complete CM3 left-rail inspector panels (Media, Sequence, Edit, Objects, Effects,
//! Export, Plugins, Settings, About) wired directly to `zc_session::Session`.

use std::collections::BTreeSet;

use egui::{
    Align2, Color32, CornerRadius, CursorIcon, Frame, Id, Key, Pos2, Rect, ScrollArea, Sense,
    Stroke, StrokeKind, TextEdit, Ui, Vec2,
};
use serde_json::{Value, json};
use zc_color::{BlendMode, Rgb, Rgba};
use zc_plugin::PluginCategory;
use zc_project::{
    EffectCategory, GradientStop, PropertyValue, create_effect_by_id, standard_effect_catalog,
    zoidium_effect_catalog,
};
use zc_session::{Command, Session, SessionClip, SessionObject3D, SessionProperty};

use super::controls::{
    ButtonVariant, FloatingWindowManager, FloatingWindowSpec, FuzzyItem, FuzzyPickerEvent,
    FuzzyPickerState, KeyframeBarEvent, KeyframeBarState, PropertyColumnState, ScrubConfig,
    ScrubEvent, cm3_collapsible_section, cm3_column_adjust_splitter, cm3_dropdown,
    cm3_floating_window, cm3_fuzzy_picker, cm3_gradient_bar, cm3_hsva_color_picker,
    cm3_icon_button, cm3_keyframe_bar, cm3_prop_button, cm3_property_row, cm3_scrub_number,
    cm3_stopwatch_button, cm3_vector2, cm3_vector3,
};
use super::layout::{SplitOrientation, SplitPanelState, begin_split_panel, show_split_pane};
use super::theme::{
    BG_HOVER, BG_INPUT, BG_PANEL, BG_TITLE, BORDER_CONTROL, BORDER_FILTER, BORDER_ROW_SOFT,
    Cm3Icon, Cm3Theme, DANGER_TEXT, FONT_BANNER, FONT_BODY, FONT_COMPACT, FONT_SMALL, FONT_TITLE,
    TEXT_BRIGHT, TEXT_DISABLED, TEXT_EMPTY, TEXT_FAINT, TEXT_MUTED, TEXT_OPTION_DESC, TEXT_PRIMARY,
    TEXT_STRONG, ThemeProfileChoice, WARNING_BAR, WARNING_SURFACE, color32_with_alpha, paint_badge,
    paint_cm3_icon, rgb_to_color32,
};

// ---------------------------------------------------------------------------
// Left-Rail Tab Enum (Exact CM3 Order + Zoidium Tabs Above About, Spec 01 §1.4)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SideTab {
    #[default]
    Media,
    Sequence,
    Edit,
    Objects,
    Effects,
    Export,
    Plugins,
    Settings,
    About,
}

impl SideTab {
    /// Exact CM3 order with Zoidium Plugins & Settings inserted immediately above About.
    pub const ALL: [Self; 9] = [
        Self::Media,
        Self::Sequence,
        Self::Edit,
        Self::Objects,
        Self::Effects,
        Self::Export,
        Self::Plugins,
        Self::Settings,
        Self::About,
    ];

    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Self::Media => "Media",
            Self::Sequence => "Sequence",
            Self::Edit => "Edit",
            Self::Objects => "Objects",
            Self::Effects => "Effects",
            Self::Export => "Export",
            Self::Plugins => "Plugins",
            Self::Settings => "Settings",
            Self::About => "About",
        }
    }

    #[must_use]
    pub const fn icon(self) -> Cm3Icon {
        match self {
            Self::Media => Cm3Icon::Project,
            Self::Sequence => Cm3Icon::Sequence,
            Self::Edit => Cm3Icon::Edit2,
            Self::Objects => Cm3Icon::Objects,
            Self::Effects => Cm3Icon::Fx,
            Self::Export => Cm3Icon::Export,
            Self::Plugins => Cm3Icon::Plugins,
            Self::Settings => Cm3Icon::Settings,
            Self::About => Cm3Icon::About,
        }
    }
}

// ---------------------------------------------------------------------------
// Export Settings & Status (Spec 04 §4.1)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportMode {
    #[default]
    Device,
    Frame,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportContainer {
    #[default]
    Mp4,
    WebM,
    Mkv,
    PngSequence,
}

impl ExportContainer {
    pub const ALL: [Self; 4] = [Self::Mp4, Self::WebM, Self::Mkv, Self::PngSequence];
    pub const LABELS: [&'static str; 4] = [
        "MP4 (.mp4)",
        "WebM (.webm)",
        "MKV (.mkv)",
        "PNG Sequence (.png)",
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportVideoCodec {
    #[default]
    H264,
    H265,
    Vp9,
    Vp8,
    Av1,
    Png,
}

impl ExportVideoCodec {
    pub const ALL: [Self; 6] = [
        Self::H264,
        Self::H265,
        Self::Vp9,
        Self::Vp8,
        Self::Av1,
        Self::Png,
    ];
    pub const LABELS: [&'static str; 6] = [
        "H.264 (libx264)",
        "H.265 / HEVC (libx265)",
        "VP9 (libvpx-vp9)",
        "VP8 (legacy)",
        "AV1 (libsvtav1)",
        "PNG (lossless)",
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportRateControl {
    #[default]
    Crf,
    Bitrate,
    Cm3Preset,
}

impl ExportRateControl {
    pub const ALL: [Self; 3] = [Self::Crf, Self::Bitrate, Self::Cm3Preset];
    pub const LABELS: [&'static str; 3] = ["Quality (CRF)", "Bitrate (Mbps)", "CM3 Quality Preset"];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportCm3Preset {
    Fastest,
    Fast,
    #[default]
    Balanced,
    Good,
    Extreme,
}

impl ExportCm3Preset {
    pub const ALL: [Self; 5] = [
        Self::Fastest,
        Self::Fast,
        Self::Balanced,
        Self::Good,
        Self::Extreme,
    ];
    pub const LABELS: [&'static str; 5] = ["Fastest", "Fast", "Balanced", "Good", "Extreme"];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportAudioCodec {
    #[default]
    AacLc,
    Opus,
    PcmS16Le,
    None,
}

impl ExportAudioCodec {
    pub const ALL: [Self; 4] = [Self::AacLc, Self::Opus, Self::PcmS16Le, Self::None];
    pub const LABELS: [&'static str; 4] = ["AAC-LC", "Opus", "PCM (s16le)", "None"];
}

/// Complete Export tab configuration matching Spec 04 §4.1.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportSettings {
    pub mode: ExportMode,
    pub container: ExportContainer,
    pub video_codec: ExportVideoCodec,
    pub rate_control: ExportRateControl,
    pub crf: u32,
    pub bitrate_mbps: f64,
    pub cm3_preset: ExportCm3Preset,
    pub audio_codec: ExportAudioCodec,
    pub audio_bitrate_kbps: u32,
    pub sample_rate_hz: u32,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub start_frame: f64,
    pub end_frame: f64,
    pub transparent_alpha: bool,
    pub datamosh_mode: usize,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            mode: ExportMode::Device,
            container: ExportContainer::Mp4,
            video_codec: ExportVideoCodec::H264,
            rate_control: ExportRateControl::Crf,
            crf: 18,
            bitrate_mbps: 12.0,
            cm3_preset: ExportCm3Preset::Balanced,
            audio_codec: ExportAudioCodec::AacLc,
            audio_bitrate_kbps: 192,
            sample_rate_hz: 48_000,
            width: 1920,
            height: 1080,
            fps: 30.0,
            start_frame: 0.0,
            end_frame: 300.0,
            transparent_alpha: false,
            datamosh_mode: 0,
        }
    }
}

/// Live FFmpeg probe & active export job telemetry shown in the Export tab.
#[derive(Debug, Clone)]
pub struct ExportStatus {
    pub ffmpeg_available: bool,
    pub ffmpeg_version: String,
    pub exporting: bool,
    /// Progress fraction in `0.0..=1.0`.
    pub progress: f32,
    pub status_message: String,
}

impl Default for ExportStatus {
    fn default() -> Self {
        Self {
            ffmpeg_available: true,
            ffmpeg_version: "ready (probe on export)".to_owned(),
            exporting: false,
            progress: 0.0,
            status_message: String::new(),
        }
    }
}

/// Persistent editor preferences shown in the Settings tab.
#[derive(Debug, Clone)]
pub struct AppSettings {
    pub preview_cache_mib: u32,
    pub undo_max_entries: usize,
    pub undo_max_mib: usize,
    pub show_fps_overlay: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            preview_cache_mib: 512,
            undo_max_entries: 200,
            undo_max_mib: 64,
            show_fps_overlay: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Actions Emitted by Side Panels to the App Shell
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum PanelAction {
    ImportMedia,
    StartExport(ExportSettings),
    CancelExport,
    EyedropperActivate { target_id: String, property: String },
    OpenGraphEditor,
    StatusMessage(String),
}

// ---------------------------------------------------------------------------
// Transient Side-Panel UI State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TreeDropMode {
    Before,
    After,
    Inside,
}

#[derive(Debug, Clone)]
struct TreeDragState {
    dragging_id: String,
    start_y: f32,
    active: bool,
}

#[derive(Debug, Clone)]
struct InlineRenameState {
    target_id: String,
    buffer: String,
    just_started: bool,
}

#[derive(Debug, Clone)]
struct ShaderEditorTarget {
    effect_id: String,
    effect_name: String,
    code: String,
    font_px: f32,
    error_message: Option<String>,
}

#[derive(Debug, Clone)]
struct ExprEditorTarget {
    target_id: String,
    property: String,
    code: String,
}

/// Stateful UI context for all left-rail panels and inspector sub-windows.
pub struct PanelsState {
    pub media_search: String,
    pub objects_split: SplitPanelState,
    pub effects_split: SplitPanelState,
    pub prop_columns: PropertyColumnState,
    pub add_object_open: bool,
    pub add_object_picker: FuzzyPickerState,
    pub add_effect_open: bool,
    pub add_effect_picker: FuzzyPickerState,
    pub collapsed_objects: BTreeSet<String>,
    pub export_settings: ExportSettings,
    pub export_status: ExportStatus,
    pub app_settings: AppSettings,
    pub plugin_notice: Option<String>,
    pub floating_windows: FloatingWindowManager,
    pub new_seq_control_name: String,
    pub new_shader_prop_name: String,
    pub new_shader_prop_kind: usize,
    object_drag: Option<TreeDragState>,
    effect_drag: Option<TreeDragState>,
    rename_state: Option<InlineRenameState>,
    shader_editor: Option<ShaderEditorTarget>,
    expr_editor: Option<ExprEditorTarget>,
}

impl Default for PanelsState {
    fn default() -> Self {
        Self {
            media_search: String::new(),
            objects_split: SplitPanelState::new(0.40, SplitOrientation::Auto, 80.0, 100.0),
            effects_split: SplitPanelState::new(0.40, SplitOrientation::Auto, 80.0, 100.0),
            prop_columns: PropertyColumnState::default(),
            add_object_open: false,
            add_object_picker: FuzzyPickerState::default(),
            add_effect_open: false,
            add_effect_picker: FuzzyPickerState::default(),
            collapsed_objects: BTreeSet::new(),
            export_settings: ExportSettings::default(),
            export_status: ExportStatus::default(),
            app_settings: AppSettings::default(),
            plugin_notice: None,
            floating_windows: FloatingWindowManager::default(),
            new_seq_control_name: String::from("ctl_value"),
            new_shader_prop_name: String::from("Amount"),
            new_shader_prop_kind: 0,
            object_drag: None,
            effect_drag: None,
            rename_state: None,
            shader_editor: None,
            expr_editor: None,
        }
    }
}

/// Renders the currently active CM3 / Zoidium side panel inside `panel_rect`.
#[allow(clippy::too_many_arguments)]
pub fn show_side_panel(
    ui: &mut Ui,
    tab: SideTab,
    panel_rect: Rect,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &mut Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    match tab {
        SideTab::Media => show_media_tab(ui, session, theme, state, actions),
        SideTab::Sequence => {
            show_sequence_tab(
                ui,
                panel_rect,
                viewport_rect,
                session,
                theme,
                state,
                actions,
            );
        }
        SideTab::Edit => {
            show_edit_tab(
                ui,
                panel_rect,
                viewport_rect,
                session,
                theme,
                state,
                actions,
            );
        }
        SideTab::Objects => {
            show_objects_tab(
                ui,
                panel_rect,
                viewport_rect,
                session,
                theme,
                state,
                actions,
            );
        }
        SideTab::Effects => {
            show_effects_tab(
                ui,
                panel_rect,
                viewport_rect,
                session,
                theme,
                state,
                actions,
            );
        }
        SideTab::Export => {
            show_export_tab(ui, viewport_rect, session, theme, state, actions);
        }
        SideTab::Plugins => show_plugins_tab(ui, session, theme, state),
        SideTab::Settings => show_settings_tab(ui, viewport_rect, session, theme, state),
        SideTab::About => show_about_tab(ui, session, theme),
    }

    // Floating editors spawned from inspectors (Custom Shader GLSL editor, Expression editor)
    show_inspector_floating_windows(ui.ctx(), viewport_rect, session, theme, state);
}

// ---------------------------------------------------------------------------
// Tab 1: Media (project)
// ---------------------------------------------------------------------------

struct MediaPreset {
    kind: &'static str,
    title: &'static str,
    subtitle: &'static str,
    required_plugin: Option<&'static str>,
}

const MEDIA_PRESETS: &[MediaPreset] = &[
    MediaPreset {
        kind: "scene",
        title: "3D Scene",
        subtitle: "3D camera, lights & meshes",
        required_plugin: None,
    },
    MediaPreset {
        kind: "adjustment",
        title: "Adjustment",
        subtitle: "Post-processing effect layer",
        required_plugin: None,
    },
    MediaPreset {
        kind: "text",
        title: "Text",
        subtitle: "2D typographic layer",
        required_plugin: None,
    },
    MediaPreset {
        kind: "preset_shape",
        title: "Preset shape",
        subtitle: "Parametric 2D preset",
        required_plugin: None,
    },
    MediaPreset {
        kind: "shape",
        title: "Shape",
        subtitle: "Custom vector shape",
        required_plugin: None,
    },
    MediaPreset {
        kind: "composite",
        title: "Composite",
        subtitle: "Nested layer composite",
        required_plugin: None,
    },
    MediaPreset {
        kind: "zoidium:camera-plus/camera",
        title: "Camera Layer",
        subtitle: "Camera+ shared 3D camera",
        required_plugin: Some("camera-plus"),
    },
    MediaPreset {
        kind: "zoidium:shape-plus/shape-layer",
        title: "Shape+ Layer",
        subtitle: "Animated AE-style vector layer",
        required_plugin: Some("shape-plus"),
    },
    MediaPreset {
        kind: "zoidium:precomp-plus/precomp",
        title: "Precomp",
        subtitle: "Nested sequence composition",
        required_plugin: Some("precomp-plus"),
    },
];

fn show_media_tab(
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
            Frame::NONE
                .inner_margin(egui::Margin::same(6))
                .show(ui, |ui| {
                    // Search filter box
                    let search_id = Id::unique("media_search_box");
                    let focused = ui.memory(|m| m.has_focus(search_id));
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
                            ui.add(
                                TextEdit::singleline(&mut state.media_search)
                                    .id(search_id)
                                    .hint_text("search media & presets")
                                    .font(theme.font(FONT_BODY))
                                    .text_color(TEXT_PRIMARY)
                                    .desired_width(ui.available_width()),
                            );
                        });

                    ui.add_space(6.0);
                    if cm3_prop_button(ui, "import media...", ButtonVariant::Normal, true, theme)
                        .clicked()
                    {
                        actions.push(PanelAction::ImportMedia);
                    }

                    ui.add_space(8.0);
                    section_banner(ui, "Built-in Presets", theme);
                    ui.add_space(4.0);

                    let q = state.media_search.trim().to_ascii_lowercase();
                    let presets: Vec<&MediaPreset> = MEDIA_PRESETS
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
                        .collect();

                    let cols = if ui.available_width() > 300.0 { 2 } else { 1 };
                    for chunk in presets.chunks(cols) {
                        ui.horizontal(|ui| {
                            let gap = 5.0;
                            let card_w = ((ui.available_width() - gap * (cols - 1) as f32)
                                / cols as f32)
                                .max(100.0);
                            for preset in chunk {
                                let (rect, resp) =
                                    ui.allocate_exact_size(Vec2::new(card_w, 54.0), Sense::click());
                                let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
                                let bg = if resp.hovered() {
                                    theme.menu_active()
                                } else {
                                    BG_HOVER
                                };
                                ui.painter().rect_filled(rect, CornerRadius::ZERO, bg);
                                ui.painter().rect_stroke(
                                    rect,
                                    CornerRadius::ZERO,
                                    Stroke::new(
                                        1.0,
                                        if resp.hovered() {
                                            theme.accent()
                                        } else {
                                            BORDER_CONTROL
                                        },
                                    ),
                                    StrokeKind::Inside,
                                );
                                ui.painter().text(
                                    Pos2::new(rect.left() + 8.0, rect.top() + 18.0),
                                    Align2::LEFT_CENTER,
                                    preset.title,
                                    theme.font(FONT_BODY),
                                    if resp.hovered() {
                                        TEXT_BRIGHT
                                    } else {
                                        TEXT_PRIMARY
                                    },
                                );
                                ui.painter().text(
                                    Pos2::new(rect.left() + 8.0, rect.top() + 38.0),
                                    Align2::LEFT_CENTER,
                                    preset.subtitle,
                                    theme.font(FONT_COMPACT),
                                    TEXT_OPTION_DESC,
                                );
                                if resp.clicked() {
                                    add_preset_clip(session, preset.kind, preset.title);
                                }
                            }
                        });
                        ui.add_space(4.0);
                    }

                    // Project Media list
                    ui.add_space(8.0);
                    section_banner(ui, "Project Media", theme);
                    ui.add_space(4.0);

                    let media_items: Vec<(String, String)> = session
                        .project
                        .media
                        .iter()
                        .filter(|m| q.is_empty() || m.name.to_ascii_lowercase().contains(&q))
                        .map(|m| (m.name.clone(), format!("{:?}", m.kind)))
                        .collect();
                    let asset_items: Vec<(String, u64, bool)> = session
                        .project
                        .assets
                        .values()
                        .filter(|a| q.is_empty() || a.filename.to_ascii_lowercase().contains(&q))
                        .map(|a| (a.filename.clone(), a.size, a.missing))
                        .collect();

                    if media_items.is_empty() && asset_items.is_empty() {
                        ui.label(
                            egui::RichText::new("no imported project media")
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_EMPTY),
                        );
                    } else {
                        for (name, kind_label) in media_items {
                            let (r, resp) = ui.allocate_exact_size(
                                Vec2::new(ui.available_width(), 28.0),
                                Sense::click(),
                            );
                            ui.painter().rect_filled(
                                r,
                                CornerRadius::ZERO,
                                if resp.hovered() { BG_HOVER } else { BG_TITLE },
                            );
                            ui.painter().text(
                                Pos2::new(r.left() + 8.0, r.center().y),
                                Align2::LEFT_CENTER,
                                &name,
                                theme.font(FONT_BODY),
                                TEXT_PRIMARY,
                            );
                            ui.painter().text(
                                Pos2::new(r.right() - 8.0, r.center().y),
                                Align2::RIGHT_CENTER,
                                &kind_label,
                                theme.font(FONT_COMPACT),
                                TEXT_FAINT,
                            );
                            if resp.clicked() {
                                add_preset_clip(session, "image", &name);
                            }
                        }
                        for (filename, size, missing) in asset_items {
                            let (r, _) = ui.allocate_exact_size(
                                Vec2::new(ui.available_width(), 26.0),
                                Sense::hover(),
                            );
                            ui.painter().rect_filled(r, CornerRadius::ZERO, BG_TITLE);
                            ui.painter().text(
                                Pos2::new(r.left() + 8.0, r.center().y),
                                Align2::LEFT_CENTER,
                                &filename,
                                theme.font(FONT_SMALL),
                                if missing { DANGER_TEXT } else { TEXT_PRIMARY },
                            );
                            let meta = if missing {
                                "missing".to_owned()
                            } else {
                                format!("{} KiB", size.div_ceil(1024))
                            };
                            ui.painter().text(
                                Pos2::new(r.right() - 8.0, r.center().y),
                                Align2::RIGHT_CENTER,
                                meta,
                                theme.font(FONT_COMPACT),
                                TEXT_FAINT,
                            );
                        }
                    }
                });
        });
}

fn add_preset_clip(session: &mut Session, kind: &str, title: &str) {
    let track_id = if let Some(first) = session.document().video_tracks.first() {
        first.id.clone()
    } else if let Ok(res) = session.execute(Command::AddVideoTrack { name: None })
        && let Some(tid) = res.get("track_id").and_then(Value::as_str)
    {
        tid.to_owned()
    } else {
        return;
    };

    let start = session.playback.current_frame.max(0.0).round();
    if let Ok(res) = session.execute(Command::AddClip {
        track_id,
        kind: kind.to_owned(),
        name: Some(title.to_owned()),
        start,
        length: 150.0,
    }) && let Some(cid) = res.get("clip_id").and_then(Value::as_str)
    {
        let _ = session.selection.select_clip(cid, Some(kind), false);
    }
}

// ---------------------------------------------------------------------------
// Tab 2: Sequence (sequence)
// ---------------------------------------------------------------------------

fn show_sequence_tab(
    ui: &mut Ui,
    panel_rect: Rect,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    cm3_column_adjust_splitter(
        ui,
        Id::unique("seq_col_splitter"),
        panel_rect,
        &mut state.prop_columns,
    );

    let frame = session.playback.current_frame;
    ScrollArea::vertical()
        .id_salt("sequence_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            cm3_collapsible_section(
                ui,
                Id::unique("seq_general"),
                "Sequence",
                true,
                theme,
                |ui| {
                    // Resolution X / Y
                    let [rw, rh] = session.document().resolution;
                    let mut res_vec = [f64::from(rw), f64::from(rh)];
                    let ev = cm3_property_row(
                        ui,
                        "resolution",
                        state.prop_columns,
                        0,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_vector2(
                                ui,
                                Id::unique("seq_resolution"),
                                &mut res_vec,
                                ScrubConfig::new(16.0, 7680.0, 10.0, 0),
                                false,
                                theme,
                            )
                        },
                    );
                    if ev.started {
                        let _ = session.begin_transaction("set resolution");
                    }
                    if ev.changed || ev.committed {
                        let _ = session.execute(Command::SetResolution {
                            width: res_vec[0].round().clamp(16.0, 7680.0) as u32,
                            height: res_vec[1].round().clamp(16.0, 7680.0) as u32,
                        });
                    }
                    if ev.committed && session.history.is_in_transaction() {
                        let _ = session.commit_transaction(true);
                    } else if ev.cancelled && session.history.is_in_transaction() {
                        let _ = session.cancel_transaction();
                    }

                    // Frame Rate (FPS)
                    let mut fps = session.document().fps;
                    let ev = cm3_property_row(
                        ui,
                        "frame rate",
                        state.prop_columns,
                        0,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_scrub_number(
                                ui,
                                Id::unique("seq_fps"),
                                &mut fps,
                                ScrubConfig::new(1.0, 240.0, 1.0, 2),
                                theme,
                            )
                        },
                    );
                    if ev.started {
                        let _ = session.begin_transaction("set fps");
                    }
                    if ev.changed || ev.committed {
                        let _ = session.execute(Command::SetFps { fps });
                    }
                    if ev.committed && session.history.is_in_transaction() {
                        let _ = session.commit_transaction(true);
                    } else if ev.cancelled && session.history.is_in_transaction() {
                        let _ = session.cancel_transaction();
                    }

                    // Sequence Duration (frames)
                    let mut length = session.document().sequence_length;
                    let ev = cm3_property_row(
                        ui,
                        "duration (frames)",
                        state.prop_columns,
                        0,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_scrub_number(
                                ui,
                                Id::unique("seq_length"),
                                &mut length,
                                ScrubConfig::new(1.0, 1_000_000.0, 10.0, 0),
                                theme,
                            )
                        },
                    );
                    if ev.started {
                        let _ = session.begin_transaction("set sequence length");
                    }
                    if ev.changed || ev.committed {
                        let _ = session.execute(Command::SetSequenceLength { frames: length });
                    }
                    if ev.committed && session.history.is_in_transaction() {
                        let _ = session.commit_transaction(true);
                    } else if ev.cancelled && session.history.is_in_transaction() {
                        let _ = session.cancel_transaction();
                    }
                },
            );

            // Motion Blur section
            cm3_collapsible_section(ui, Id::unique("seq_mb"), "Motion Blur", true, theme, |ui| {
                show_session_option_prop(
                    ui,
                    session,
                    "sequence",
                    "motionBlur.enabled",
                    "enabled",
                    &["off", "on"],
                    0,
                    frame,
                    0.0,
                    viewport_rect,
                    state,
                    theme,
                );
                show_session_number_prop(
                    ui,
                    session,
                    "sequence",
                    "motionBlur.samples",
                    "samples",
                    16.0,
                    ScrubConfig::new(1.0, 64.0, 1.0, 0),
                    frame,
                    0.0,
                    viewport_rect,
                    state,
                    theme,
                );
                show_session_number_prop(
                    ui,
                    session,
                    "sequence",
                    "motionBlur.shutter",
                    "shutter",
                    0.5,
                    ScrubConfig::new(0.0, 2.0, 0.05, 2),
                    frame,
                    0.0,
                    viewport_rect,
                    state,
                    theme,
                );
            });

            // Expressions+ Sequence Controls (only when `expressions-plus` plugin is enabled)
            if session.plugins.is_enabled("expressions-plus") {
                cm3_collapsible_section(
                    ui,
                    Id::unique("seq_expr_controls"),
                    "Sequence Controls (Expressions+)",
                    true,
                    theme,
                    |ui| {
                        ui.horizontal(|ui| {
                            Frame::NONE
                                .fill(BG_INPUT)
                                .stroke(Stroke::new(1.0, BORDER_CONTROL))
                                .inner_margin(egui::Margin::symmetric(4, 2))
                                .show(ui, |ui| {
                                    ui.add(
                                        TextEdit::singleline(&mut state.new_seq_control_name)
                                            .font(theme.font(FONT_SMALL))
                                            .text_color(theme.value())
                                            .desired_width(130.0),
                                    );
                                });
                            if cm3_prop_button(ui, "+ slider", ButtonVariant::Normal, false, theme)
                                .clicked()
                            {
                                let raw = state.new_seq_control_name.trim();
                                let prop_key = if raw.starts_with("ctl_") {
                                    raw.to_owned()
                                } else if raw.is_empty() {
                                    "ctl_slider".to_owned()
                                } else {
                                    format!("ctl_{raw}")
                                };
                                let _ = session.execute(Command::SetPropertyValue {
                                    target_id: "sequence".to_owned(),
                                    property: prop_key,
                                    value: json!(0.0),
                                });
                            }
                        });

                        let ctl_keys: Vec<String> = session
                            .document()
                            .sequence_properties
                            .keys()
                            .filter(|k| k.starts_with("ctl_"))
                            .cloned()
                            .collect();
                        if ctl_keys.is_empty() {
                            ui.label(
                                egui::RichText::new("no sequence controls yet")
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_EMPTY),
                            );
                        } else {
                            for key in ctl_keys {
                                show_session_number_prop(
                                    ui,
                                    session,
                                    "sequence",
                                    &key,
                                    &key,
                                    0.0,
                                    ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 2),
                                    frame,
                                    0.0,
                                    viewport_rect,
                                    state,
                                    theme,
                                );
                            }
                        }
                    },
                );
            }
            let _ = actions;
        });
}

// ---------------------------------------------------------------------------
// Tab 3: Edit (edit2) — Selected Clip Inspector
// ---------------------------------------------------------------------------

fn show_edit_tab(
    ui: &mut Ui,
    panel_rect: Rect,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let Some(clip_id) = session.selection.clip_ids.first().cloned() else {
        empty_panel_message(ui, "select a clip", theme);
        return;
    };

    let Some(clip_snapshot) = find_clip_clone(session, &clip_id) else {
        empty_panel_message(ui, "select a clip", theme);
        return;
    };

    cm3_column_adjust_splitter(
        ui,
        Id::unique("edit_col_splitter"),
        panel_rect,
        &mut state.prop_columns,
    );

    let local_frame = session.playback.current_frame - clip_snapshot.start;
    let clip_start = clip_snapshot.start;

    ScrollArea::vertical()
        .id_salt("edit_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // Property list header with Reset button (Spec 01 §4.1)
            property_header_with_reset(
                ui,
                &format!("Clip: {}", clip_snapshot.name),
                &clip_id,
                session,
                theme,
            );

            // Start & Length
            let mut start = clip_snapshot.start;
            let ev = cm3_property_row(
                ui,
                "start",
                state.prop_columns,
                0,
                theme,
                |_| {},
                |ui| {
                    cm3_scrub_number(
                        ui,
                        Id::unique(("clip_start", &clip_id)),
                        &mut start,
                        ScrubConfig::new(0.0, 1_000_000.0, 1.0, 0),
                        theme,
                    )
                },
            );
            if ev.started {
                let _ = session.begin_transaction("move clip");
            }
            if ev.changed || ev.committed {
                let _ = session.execute(Command::MoveClip {
                    clip_id: clip_id.clone(),
                    target_track_id: None,
                    start: start.max(0.0),
                });
            }
            if ev.committed && session.history.is_in_transaction() {
                let _ = session.commit_transaction(true);
            } else if ev.cancelled && session.history.is_in_transaction() {
                let _ = session.cancel_transaction();
            }

            let mut length = clip_snapshot.length;
            let ev = cm3_property_row(
                ui,
                "length",
                state.prop_columns,
                0,
                theme,
                |_| {},
                |ui| {
                    cm3_scrub_number(
                        ui,
                        Id::unique(("clip_len", &clip_id)),
                        &mut length,
                        ScrubConfig::new(1.0, 1_000_000.0, 1.0, 0),
                        theme,
                    )
                },
            );
            if ev.started {
                let _ = session.begin_transaction("trim clip");
            }
            if ev.changed || ev.committed {
                let delta = length - clip_snapshot.length;
                if delta.abs() > 1e-6 {
                    let _ = session.execute(Command::TrimClipRight {
                        clip_id: clip_id.clone(),
                        delta_frames: delta,
                        rate_stretch: false,
                    });
                }
            }
            if ev.committed && session.history.is_in_transaction() {
                let _ = session.commit_transaction(true);
            } else if ev.cancelled && session.history.is_in_transaction() {
                let _ = session.cancel_transaction();
            }

            // Transform & Compositing Section
            cm3_collapsible_section(
                ui,
                Id::unique("edit_transform"),
                "Transform",
                true,
                theme,
                |ui| {
                    show_session_vec2_prop(
                        ui,
                        session,
                        &clip_id,
                        "position",
                        "position",
                        [0.0, 0.0],
                        ScrubConfig::new(-10_000.0, 10_000.0, 1.0, 1),
                        false,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                    );
                    show_session_vec2_prop(
                        ui,
                        session,
                        &clip_id,
                        "scale",
                        "scale",
                        [1.0, 1.0],
                        ScrubConfig::new(-100.0, 100.0, 0.05, 3),
                        true,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                    );
                    show_session_number_prop(
                        ui,
                        session,
                        &clip_id,
                        "rotation",
                        "rotation",
                        0.0,
                        ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 1)
                            .with_scale_factor(std::f64::consts::PI / 180.0),
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                    );
                    show_session_number_prop(
                        ui,
                        session,
                        &clip_id,
                        "opacity",
                        "opacity",
                        1.0,
                        ScrubConfig::new(0.0, 1.0, 0.05, 2),
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                    );

                    // Blending Mode (All 43 CM3 BlendModes)
                    let cur_blend_token = eval_prop_str(
                        clip_snapshot.properties.get("blending"),
                        local_frame,
                        "BLEND_SRC_OVER",
                    );
                    let mut blend_idx = BlendMode::ALL
                        .iter()
                        .position(|b| b.token() == cur_blend_token)
                        .unwrap_or(3);
                    let blend_labels: Vec<&str> =
                        BlendMode::ALL.iter().map(|b| b.display_name()).collect();
                    let changed = cm3_property_row(
                        ui,
                        "blending",
                        state.prop_columns,
                        0,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_dropdown(
                                ui,
                                Id::unique(("clip_blend", &clip_id)),
                                &mut blend_idx,
                                &blend_labels,
                                false,
                                viewport_rect,
                                theme,
                            )
                        },
                    );
                    if changed {
                        let token = BlendMode::ALL[blend_idx].token();
                        let _ = session.execute(Command::SetPropertyValue {
                            target_id: clip_id.clone(),
                            property: "blending".to_owned(),
                            value: json!(token),
                        });
                    }

                    // Time remap property
                    show_session_number_prop(
                        ui,
                        session,
                        &clip_id,
                        "time",
                        "time",
                        (local_frame / session.document().fps.max(1.0)).max(0.0),
                        ScrubConfig::new(0.0, 100_000.0, 0.05, 3),
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                    );
                },
            );

            // Layer-specific properties
            match clip_snapshot.kind.as_str() {
                "text" => {
                    cm3_collapsible_section(
                        ui,
                        Id::unique("edit_text"),
                        "Text Layer",
                        true,
                        theme,
                        |ui| {
                            show_session_text_prop(
                                ui,
                                session,
                                &clip_id,
                                "text",
                                "text",
                                "text",
                                local_frame,
                                state,
                                theme,
                            );
                            show_session_number_prop(
                                ui,
                                session,
                                &clip_id,
                                "fontSize",
                                "font size",
                                72.0,
                                ScrubConfig::new(4.0, 1000.0, 1.0, 0),
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                            );
                            show_session_color_prop(
                                ui,
                                session,
                                &clip_id,
                                "color",
                                "color",
                                [1.0, 1.0, 1.0],
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );
                        },
                    );
                }
                "shape" | "preset_shape" => {
                    cm3_collapsible_section(
                        ui,
                        Id::unique("edit_shape"),
                        "Shape Layer",
                        true,
                        theme,
                        |ui| {
                            show_session_color_prop(
                                ui,
                                session,
                                &clip_id,
                                "fillColor",
                                "fill color",
                                [1.0, 1.0, 1.0],
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );
                            show_session_number_prop(
                                ui,
                                session,
                                &clip_id,
                                "strokeWidth",
                                "stroke width",
                                0.0,
                                ScrubConfig::new(0.0, 200.0, 1.0, 1),
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                            );
                        },
                    );
                }
                "scene" => {
                    cm3_collapsible_section(
                        ui,
                        Id::unique("edit_scene"),
                        "3D Scene",
                        true,
                        theme,
                        |ui| {
                            show_session_color_prop(
                                ui,
                                session,
                                &clip_id,
                                "backgroundColor",
                                "background",
                                [0.0, 0.0, 0.0],
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );
                            show_session_option_prop(
                                ui,
                                session,
                                &clip_id,
                                "shadows",
                                "shadows",
                                &["off", "on"],
                                1,
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                            );
                        },
                    );
                }
                _ => {}
            }
        });
}

// ---------------------------------------------------------------------------
// Tab 4: Objects (objects) — 40/60 Auto-Split 3D Tree + Inspector
// ---------------------------------------------------------------------------

fn build_object_picker_items(session: &Session) -> Vec<FuzzyItem> {
    let mut items = vec![
        // Standard CM3 3D Shapes
        FuzzyItem {
            id: "box".into(),
            category: "3D Shapes".into(),
            name: "Box".into(),
            desc: "3D rectangular prism".into(),
        },
        FuzzyItem {
            id: "cylinder".into(),
            category: "3D Shapes".into(),
            name: "Cylinder".into(),
            desc: "Cylinder or cone primitive".into(),
        },
        FuzzyItem {
            id: "rectangle".into(),
            category: "3D Shapes".into(),
            name: "Rectangle".into(),
            desc: "Flat 3D plane".into(),
        },
        FuzzyItem {
            id: "circle".into(),
            category: "3D Shapes".into(),
            name: "Circle".into(),
            desc: "Flat 3D disc".into(),
        },
        FuzzyItem {
            id: "sphere".into(),
            category: "3D Shapes".into(),
            name: "Sphere".into(),
            desc: "UV sphere mesh".into(),
        },
        FuzzyItem {
            id: "donut".into(),
            category: "3D Shapes".into(),
            name: "Donut".into(),
            desc: "Torus ring primitive".into(),
        },
        FuzzyItem {
            id: "wire".into(),
            category: "3D Shapes".into(),
            name: "Wire".into(),
            desc: "3D polyline tube".into(),
        },
        // Standard CM3 Scene Objects
        FuzzyItem {
            id: "text".into(),
            category: "Scene Objects".into(),
            name: "Text".into(),
            desc: "Extruded 3D text mesh".into(),
        },
        FuzzyItem {
            id: "group".into(),
            category: "Scene Objects".into(),
            name: "Group".into(),
            desc: "Hierarchical transform container".into(),
        },
        FuzzyItem {
            id: "particles".into(),
            category: "Scene Objects".into(),
            name: "Particles".into(),
            desc: "3D particle emitter".into(),
        },
        FuzzyItem {
            id: "model".into(),
            category: "Scene Objects".into(),
            name: "3D Model".into(),
            desc: "Imported OBJ/GLTF mesh".into(),
        },
        // Lights & Cameras
        FuzzyItem {
            id: "spot_light".into(),
            category: "Lights & Cameras".into(),
            name: "Spot Light".into(),
            desc: "Cone light with shadows".into(),
        },
        FuzzyItem {
            id: "point_light".into(),
            category: "Lights & Cameras".into(),
            name: "Point Light".into(),
            desc: "Omnidirectional point light".into(),
        },
        FuzzyItem {
            id: "directional_light".into(),
            category: "Lights & Cameras".into(),
            name: "Directional Light".into(),
            desc: "Parallel sun-like light".into(),
        },
        FuzzyItem {
            id: "hemisphere_light".into(),
            category: "Lights & Cameras".into(),
            name: "Hemisphere Light".into(),
            desc: "Sky & ground ambient light".into(),
        },
        FuzzyItem {
            id: "camera".into(),
            category: "Lights & Cameras".into(),
            name: "Camera".into(),
            desc: "Perspective or orthographic camera".into(),
        },
    ];

    if session.plugins.is_enabled("geometry-plus") {
        for (id, name, desc) in [
            (
                "zoidium:geometry-plus/rounded_box",
                "Rounded Box",
                "Beveled procedural box",
            ),
            (
                "zoidium:geometry-plus/icosahedron",
                "Icosahedron",
                "Low-poly 20-face polyhedron",
            ),
            (
                "zoidium:geometry-plus/octahedron",
                "Octahedron",
                "Low-poly 8-face polyhedron",
            ),
            (
                "zoidium:geometry-plus/dodecahedron",
                "Dodecahedron",
                "Low-poly 12-face polyhedron",
            ),
            (
                "zoidium:geometry-plus/tetrahedron",
                "Tetrahedron",
                "Low-poly 4-face polyhedron",
            ),
            (
                "zoidium:geometry-plus/capsule",
                "Capsule",
                "Procedural pill mesh",
            ),
            (
                "zoidium:geometry-plus/torus_knot",
                "Torus Knot",
                "Procedural mathematical knot",
            ),
            (
                "zoidium:geometry-plus/superquadric",
                "Superquadric",
                "Parametric super-ellipsoid",
            ),
        ] {
            items.push(FuzzyItem {
                id: id.into(),
                category: "Geometry+".into(),
                name: name.into(),
                desc: desc.into(),
            });
        }
    }

    if session.plugins.is_enabled("light-plus") {
        items.push(FuzzyItem {
            id: "zoidium:light-plus/light_rig".into(),
            category: "Light+".into(),
            name: "Light+ Rig".into(),
            desc: "Extended Spot/Point/Directional/Hemi chooser".into(),
        });
    }

    if session.plugins.is_enabled("repeater") {
        for (id, name, desc) in [
            (
                "zoidium:repeater/repeater",
                "Repeater",
                "Procedural child instancer",
            ),
            (
                "zoidium:repeater/linear_repeater",
                "Linear Repeater",
                "Linear array instancer",
            ),
            (
                "zoidium:repeater/random_repeater",
                "Random Repeater",
                "Seeded random instancer",
            ),
            (
                "zoidium:repeater/echo_repeater",
                "Echo Repeater",
                "Temporal ghost instancer",
            ),
            (
                "zoidium:repeater/twist",
                "Twist Deformer",
                "Axial twist mesh deformer",
            ),
            (
                "zoidium:repeater/warp",
                "Warp Deformer",
                "Bend & shear mesh deformer",
            ),
            (
                "zoidium:repeater/voronoi_fracture",
                "Voronoi Fracture",
                "3D cell fracture deformer",
            ),
        ] {
            items.push(FuzzyItem {
                id: id.into(),
                category: "Effector".into(),
                name: name.into(),
                desc: desc.into(),
            });
        }
    }

    if session.plugins.is_enabled("camera-plus") {
        items.push(FuzzyItem {
            id: "zoidium:camera-plus/camera_plus".into(),
            category: "Camera+".into(),
            name: "Camera+".into(),
            desc: "Film gate, DOF, motion blur & vibrate".into(),
        });
    }

    if session.plugins.is_enabled("optical-flares") {
        items.push(FuzzyItem {
            id: "zoidium:optical-flares/flare".into(),
            category: "Optical Flares".into(),
            name: "Optical Flare".into(),
            desc: "3D depth-occluded lens flare".into(),
        });
    }

    if session.plugins.is_enabled("trapcode-suite") {
        for (id, name, desc) in [
            (
                "zoidium:trapcode-suite/particular",
                "Particular",
                "Deterministic 3D fluid particle sim",
            ),
            (
                "zoidium:trapcode-suite/form",
                "Form",
                "Deformable 3D particle lattice",
            ),
            (
                "zoidium:trapcode-suite/plexus",
                "Plexus",
                "Procedural point & edge cloud",
            ),
            (
                "zoidium:trapcode-suite/area_light",
                "Area Light (LTC)",
                "Rectangular soft area light",
            ),
            (
                "zoidium:trapcode-suite/ies_light",
                "IES Photometric Light",
                "LM-63 photometric light",
            ),
        ] {
            items.push(FuzzyItem {
                id: id.into(),
                category: "Rowbyte & Red Giant Suite".into(),
                name: name.into(),
                desc: desc.into(),
            });
        }
    }

    items
}

fn show_objects_tab(
    ui: &mut Ui,
    panel_rect: Rect,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let Some(clip_id) = session.selection.clip_ids.first().cloned() else {
        empty_panel_message(ui, "select a clip", theme);
        return;
    };
    let Some(clip) = find_clip_clone(session, &clip_id) else {
        empty_panel_message(ui, "select a clip", theme);
        return;
    };

    let (top_rect, bot_rect) = begin_split_panel(
        ui,
        Id::unique("objects_40_60_split"),
        panel_rect,
        &mut state.objects_split,
    );
    show_split_pane(
        ui,
        Id::unique("objects_40_60_split_p0"),
        top_rect,
        |ui, _top_rect| {
            // Pane 0 (40%): Object Tree or Inline Fuzzy Add Picker
            if state.add_object_open {
                let items = build_object_picker_items(session);
                match cm3_fuzzy_picker(
                    ui,
                    Id::unique("add_object_picker"),
                    &items,
                    &mut state.add_object_picker,
                    theme,
                ) {
                    FuzzyPickerEvent::Cancelled => {
                        state.add_object_open = false;
                    }
                    FuzzyPickerEvent::Selected(kind) => {
                        let parent_id = session.selection.object_3d_ids.first().and_then(|sel| {
                            clip.objects_3d
                                .iter()
                                .find(|o| o.id == *sel && is_container_object(&o.kind))
                                .map(|o| o.id.clone())
                        });
                        let display = items.iter().find(|i| i.id == kind).map(|i| i.name.clone());
                        if let Ok(res) = session.execute(Command::AddObject3D {
                            clip_id: clip_id.clone(),
                            parent_id,
                            kind: kind.clone(),
                            name: display,
                        }) && let Some(oid) = res.get("object_id").and_then(Value::as_str)
                        {
                            let _ = session.selection.select_object_3d(oid, &kind, false);
                        }
                        state.add_object_open = false;
                    }
                    FuzzyPickerEvent::None => {}
                }
                return;
            }

            // Header bar with `+` add button
            ui.horizontal(|ui| {
                let hdr_rect =
                    Rect::from_min_size(ui.cursor().min, Vec2::new(ui.available_width(), 28.0));
                ui.painter()
                    .rect_filled(hdr_rect, CornerRadius::ZERO, BG_TITLE);
                ui.allocate_ui_with_layout(
                    Vec2::new(ui.available_width(), 28.0),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new("Objects")
                                .font(theme.font(FONT_TITLE))
                                .color(TEXT_STRONG),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add_space(4.0);
                            if cm3_icon_button(
                                ui,
                                Cm3Icon::Add,
                                17.0,
                                TEXT_PRIMARY,
                                "add 3D object",
                                theme,
                            ) {
                                state.add_object_open = true;
                                state.add_object_picker = FuzzyPickerState {
                                    query: String::new(),
                                    active_index: 0,
                                    just_opened: true,
                                };
                            }
                        });
                    },
                );
            });

            // Hierarchical 3D object tree
            ScrollArea::vertical()
                .id_salt("objects_tree_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = Vec2::ZERO;
                    if clip.objects_3d.is_empty() {
                        empty_panel_message(ui, "no 3D objects — click + to add", theme);
                        return;
                    }

                    let flat_tree = flatten_object_tree(&clip.objects_3d, &state.collapsed_objects);
                    let mut drop_request: Option<(String, String, TreeDropMode, bool)> = None;

                    for (obj_idx, depth, has_children) in flat_tree {
                        let obj = &clip.objects_3d[obj_idx];
                        let selected = session.selection.object_3d_ids.contains(&obj.id);
                        let is_group = is_container_object(&obj.kind);

                        let row_w = ui.available_width();
                        let (row_rect, row_resp) =
                            ui.allocate_exact_size(Vec2::new(row_w, 26.0), Sense::click_and_drag());

                        if selected {
                            ui.painter()
                                .rect_filled(row_rect, CornerRadius::ZERO, theme.accent());
                        } else if row_resp.hovered() {
                            ui.painter()
                                .rect_filled(row_rect, CornerRadius::ZERO, BG_HOVER);
                        }
                        ui.painter().line_segment(
                            [row_rect.left_bottom(), row_rect.right_bottom()],
                            Stroke::new(1.0, BORDER_ROW_SOFT),
                        );

                        let indent = 6.0 + depth as f32 * 15.0;

                        // Collapse triangle for containers
                        if has_children || is_group {
                            let tri_rect = Rect::from_min_size(
                                Pos2::new(row_rect.left() + indent, row_rect.top() + 3.0),
                                Vec2::splat(20.0),
                            );
                            let collapsed = state.collapsed_objects.contains(&obj.id);
                            let tri_resp = ui
                                .interact(
                                    tri_rect,
                                    Id::unique(("obj_collapse", &obj.id)),
                                    Sense::click(),
                                )
                                .on_hover_cursor(CursorIcon::PointingHand);
                            paint_cm3_icon(
                                ui.painter(),
                                tri_rect.shrink(4.0),
                                if collapsed {
                                    Cm3Icon::ChevronRight
                                } else {
                                    Cm3Icon::ChevronDown
                                },
                                TEXT_MUTED,
                            );
                            if tri_resp.clicked() {
                                if collapsed {
                                    state.collapsed_objects.remove(&obj.id);
                                } else {
                                    state.collapsed_objects.insert(obj.id.clone());
                                }
                            }
                        }

                        // Action buttons at right (duplicate, remove — 17x17 each)
                        let rm_rect = Rect::from_min_size(
                            Pos2::new(row_rect.right() - 24.0, row_rect.top() + 3.0),
                            Vec2::splat(20.0),
                        );
                        let dup_rect = Rect::from_min_size(
                            Pos2::new(rm_rect.left() - 22.0, row_rect.top() + 3.0),
                            Vec2::splat(20.0),
                        );

                        let dup_resp = ui
                            .interact(dup_rect, Id::unique(("obj_dup", &obj.id)), Sense::click())
                            .on_hover_cursor(CursorIcon::PointingHand)
                            .on_hover_text("duplicate");
                        let rm_resp = ui
                            .interact(rm_rect, Id::unique(("obj_rm", &obj.id)), Sense::click())
                            .on_hover_cursor(CursorIcon::PointingHand)
                            .on_hover_text("delete");
                        paint_cm3_icon(
                            ui.painter(),
                            dup_rect.shrink(1.5),
                            Cm3Icon::Duplicate,
                            if dup_resp.hovered() {
                                TEXT_BRIGHT
                            } else {
                                TEXT_PRIMARY
                            },
                        );
                        paint_cm3_icon(
                            ui.painter(),
                            rm_rect.shrink(1.5),
                            Cm3Icon::Remove,
                            if rm_resp.hovered() {
                                DANGER_TEXT
                            } else {
                                TEXT_PRIMARY
                            },
                        );

                        if dup_resp.clicked() {
                            let _ = session.execute(Command::DuplicateObject3D {
                                clip_id: clip_id.clone(),
                                object_id: obj.id.clone(),
                            });
                        }
                        if rm_resp.clicked() {
                            let _ = session.execute(Command::DeleteObject3D {
                                clip_id: clip_id.clone(),
                                object_id: obj.id.clone(),
                            });
                        }

                        // Label or Inline Rename input
                        let label_left = row_rect.left() + indent + 20.0;
                        let label_rect = Rect::from_min_max(
                            Pos2::new(label_left, row_rect.top()),
                            Pos2::new((dup_rect.left() - 4.0).max(label_left), row_rect.bottom()),
                        );

                        let is_renaming = state
                            .rename_state
                            .as_ref()
                            .is_some_and(|r| r.target_id == obj.id);
                        if is_renaming {
                            if let Some(rename) = state.rename_state.as_mut() {
                                let edit_id = Id::unique(("obj_rename", &obj.id));
                                let mut commit = false;
                                let mut cancel = false;
                                ui.input(|i| {
                                    if i.key_pressed(Key::Enter) {
                                        commit = true;
                                    } else if i.key_pressed(Key::Escape) {
                                        cancel = true;
                                    }
                                });
                                let resp = ui
                                    .scope_builder(
                                        egui::UiBuilder::new().max_rect(label_rect),
                                        |ui| {
                                            ui.add(
                                                TextEdit::singleline(&mut rename.buffer)
                                                    .id(edit_id)
                                                    .font(theme.font(FONT_BODY))
                                                    .text_color(theme.value())
                                                    .desired_width(label_rect.width()),
                                            )
                                        },
                                    )
                                    .inner;
                                if rename.just_started {
                                    resp.request_focus();
                                    rename.just_started = false;
                                }
                                if cancel {
                                    state.rename_state = None;
                                } else if commit || resp.lost_focus() {
                                    let trimmed = rename.buffer.trim().to_owned();
                                    if !trimmed.is_empty() && trimmed != obj.name {
                                        let _ = session.execute(Command::SetPropertyValue {
                                            target_id: obj.id.clone(),
                                            property: "name".to_owned(),
                                            value: json!(trimmed),
                                        });
                                    }
                                    state.rename_state = None;
                                }
                            }
                        } else {
                            ui.painter().with_clip_rect(label_rect).text(
                                Pos2::new(label_rect.left(), label_rect.center().y),
                                Align2::LEFT_CENTER,
                                &obj.name,
                                theme.font(FONT_BODY),
                                if selected { TEXT_BRIGHT } else { TEXT_PRIMARY },
                            );
                        }

                        // Click / Shift+Click selection with same-type guard (Spec 01 §4.7)
                        if row_resp.clicked() && !dup_resp.clicked() && !rm_resp.clicked() {
                            let shift = ui.input(|i| i.modifiers.shift);
                            if let Err(err) = session
                                .selection
                                .select_object_3d(&obj.id, &obj.kind, shift)
                            {
                                actions.push(PanelAction::StatusMessage(err.to_string()));
                            }
                        }

                        // Right-click inline rename (Spec 01 §4.7)
                        if row_resp.secondary_clicked() {
                            state.rename_state = Some(InlineRenameState {
                                target_id: obj.id.clone(),
                                buffer: obj.name.clone(),
                                just_started: true,
                            });
                        }

                        // 5 px vertical drag-to-reorder & nest-inside (Spec 01 §4.7)
                        if row_resp.drag_started()
                            && let Some(pos) = ui.input(|i| i.pointer.interact_pos())
                        {
                            state.object_drag = Some(TreeDragState {
                                dragging_id: obj.id.clone(),
                                start_y: pos.y,
                                active: false,
                            });
                        }

                        if let Some(drag) = state.object_drag.as_mut()
                            && let Some(pos) = ui.input(|i| i.pointer.latest_pos())
                        {
                            if !drag.active && (pos.y - drag.start_y).abs() >= 5.0 {
                                drag.active = true;
                            }
                            if drag.active && drag.dragging_id != obj.id && row_rect.contains(pos) {
                                let n = ((pos.y - row_rect.top()) / row_rect.height().max(1.0))
                                    .clamp(0.0, 1.0);
                                let t = if is_group { 1.0 } else { 0.0 };
                                let s = 1.0 / (t + 2.0);
                                let mode = if n < s {
                                    TreeDropMode::Before
                                } else if 1.0 - n < s {
                                    TreeDropMode::After
                                } else if is_group {
                                    TreeDropMode::Inside
                                } else {
                                    TreeDropMode::After
                                };

                                let indicator_color = Color32::from_white_alpha(64); // #fff @ 0.25
                                match mode {
                                    TreeDropMode::Before => {
                                        let bar = Rect::from_min_size(
                                            row_rect.left_top(),
                                            Vec2::new(row_rect.width(), 3.0),
                                        );
                                        ui.painter().rect_filled(
                                            bar,
                                            CornerRadius::ZERO,
                                            indicator_color,
                                        );
                                    }
                                    TreeDropMode::After => {
                                        let bar = Rect::from_min_size(
                                            Pos2::new(row_rect.left(), row_rect.bottom() - 3.0),
                                            Vec2::new(row_rect.width(), 3.0),
                                        );
                                        ui.painter().rect_filled(
                                            bar,
                                            CornerRadius::ZERO,
                                            indicator_color,
                                        );
                                    }
                                    TreeDropMode::Inside => {
                                        ui.painter().rect_filled(
                                            row_rect,
                                            CornerRadius::ZERO,
                                            indicator_color,
                                        );
                                    }
                                }

                                if !ui.input(|i| i.pointer.primary_down()) {
                                    let alt = ui.input(|i| i.modifiers.alt);
                                    drop_request =
                                        Some((drag.dragging_id.clone(), obj.id.clone(), mode, alt));
                                }
                            }
                        }
                    }

                    // Finalize drag on release
                    if !ui.input(|i| i.pointer.primary_down()) {
                        if let Some((src_id, dst_id, mode, alt_clone)) = drop_request {
                            apply_object_drop(session, &clip, &src_id, &dst_id, mode, alt_clone);
                        }
                        state.object_drag = None;
                    }
                });
        },
    );
    show_split_pane(
        ui,
        Id::unique("objects_40_60_split_p1"),
        bot_rect,
        |ui, bot_rect| {
            // Pane 1 (60%): Selected 3D Object Property Inspector
            show_object_inspector(
                ui,
                bot_rect,
                viewport_rect,
                &clip,
                session,
                theme,
                state,
                actions,
            );
        },
    );
}

fn is_container_object(kind: &str) -> bool {
    matches!(
        kind,
        "group"
            | "zoidium:repeater/repeater"
            | "zoidium:repeater/linear_repeater"
            | "zoidium:repeater/random_repeater"
            | "zoidium:repeater/echo_repeater"
            | "zoidium:repeater/twist"
            | "zoidium:repeater/warp"
            | "zoidium:repeater/voronoi_fracture"
    )
}

fn flatten_object_tree(
    objects: &[SessionObject3D],
    collapsed: &BTreeSet<String>,
) -> Vec<(usize, usize, bool)> {
    let mut out = Vec::with_capacity(objects.len());
    fn visit(
        objects: &[SessionObject3D],
        parent_id: Option<&str>,
        depth: usize,
        collapsed: &BTreeSet<String>,
        out: &mut Vec<(usize, usize, bool)>,
    ) {
        for (idx, obj) in objects.iter().enumerate() {
            if obj.parent_id.as_deref() == parent_id {
                let has_children = objects
                    .iter()
                    .any(|child| child.parent_id.as_deref() == Some(obj.id.as_str()));
                out.push((idx, depth, has_children));
                if has_children && !collapsed.contains(&obj.id) {
                    visit(objects, Some(obj.id.as_str()), depth + 1, collapsed, out);
                }
            }
        }
    }
    visit(objects, None, 0, collapsed, &mut out);
    out
}

fn apply_object_drop(
    session: &mut Session,
    clip: &SessionClip,
    src_id: &str,
    dst_id: &str,
    mode: TreeDropMode,
    alt_clone: bool,
) {
    let Some(dst_idx) = clip.objects_3d.iter().position(|o| o.id == dst_id) else {
        return;
    };
    let dst = &clip.objects_3d[dst_idx];
    let (new_parent, insert_idx) = match mode {
        TreeDropMode::Inside => (Some(dst.id.clone()), clip.objects_3d.len()),
        TreeDropMode::Before => (dst.parent_id.clone(), dst_idx),
        TreeDropMode::After => (dst.parent_id.clone(), dst_idx + 1),
    };

    if alt_clone {
        if let Ok(res) = session.execute(Command::DuplicateObject3D {
            clip_id: clip.id.clone(),
            object_id: src_id.to_owned(),
        }) && let Some(cloned_id) = res.get("object_id").and_then(Value::as_str)
        {
            let _ = session.execute(Command::ReorderObject3D {
                clip_id: clip.id.clone(),
                object_id: cloned_id.to_owned(),
                parent_id: new_parent,
                index: insert_idx,
            });
        }
    } else {
        let _ = session.execute(Command::ReorderObject3D {
            clip_id: clip.id.clone(),
            object_id: src_id.to_owned(),
            parent_id: new_parent,
            index: insert_idx,
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn show_object_inspector(
    ui: &mut Ui,
    inspector_rect: Rect,
    viewport_rect: Rect,
    clip: &SessionClip,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let Some(obj_id) = session.selection.object_3d_ids.first().cloned() else {
        empty_panel_message(ui, "select an object", theme);
        return;
    };
    let Some(obj) = clip.objects_3d.iter().find(|o| o.id == obj_id).cloned() else {
        empty_panel_message(ui, "select an object", theme);
        return;
    };

    cm3_column_adjust_splitter(
        ui,
        Id::unique("obj_inspector_col_splitter"),
        inspector_rect,
        &mut state.prop_columns,
    );

    let local_frame = session.playback.current_frame - clip.start;
    let clip_start = clip.start;

    ScrollArea::vertical()
        .id_salt("obj_inspector_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            property_header_with_reset(
                ui,
                &format!("{} ({})", obj.name, obj.kind),
                &obj.id,
                session,
                theme,
            );

            // Transform Section
            cm3_collapsible_section(
                ui,
                Id::unique("obj_transform"),
                "Transform",
                true,
                theme,
                |ui| {
                    show_session_option_prop(
                        ui,
                        session,
                        &obj.id,
                        "enabled",
                        "enabled",
                        &["off", "on"],
                        1,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                    );
                    show_session_vec3_prop(
                        ui,
                        session,
                        &obj.id,
                        "position",
                        "position",
                        [0.0, 0.0, 0.0],
                        ScrubConfig::new(-100_000.0, 100_000.0, 1.0, 2),
                        false,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                    );
                    show_session_vec3_prop(
                        ui,
                        session,
                        &obj.id,
                        "rotation",
                        "rotation",
                        [0.0, 0.0, 0.0],
                        ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 1)
                            .with_scale_factor(std::f64::consts::PI / 180.0),
                        false,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                    );
                    show_session_vec3_prop(
                        ui,
                        session,
                        &obj.id,
                        "scale",
                        "scale",
                        [1.0, 1.0, 1.0],
                        ScrubConfig::new(-1000.0, 1000.0, 0.05, 3),
                        true,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                    );
                },
            );

            // Kind-specific sections
            if obj.kind == "text" {
                cm3_collapsible_section(
                    ui,
                    Id::unique("obj_text3d"),
                    "3D Text",
                    true,
                    theme,
                    |ui| {
                        show_session_text_prop(
                            ui,
                            session,
                            &obj.id,
                            "text",
                            "text",
                            "text",
                            local_frame,
                            state,
                            theme,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            "height",
                            "height",
                            4.0,
                            ScrubConfig::new(0.1, 500.0, 0.1, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            "thickness",
                            "thickness",
                            1.0,
                            ScrubConfig::new(0.0, 100.0, 0.1, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                        );
                        show_session_option_prop(
                            ui,
                            session,
                            &obj.id,
                            "bevelEnabled",
                            "bevel",
                            &["off", "on"],
                            1,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                        );
                        if session.plugins.is_enabled("text-plus") {
                            show_session_number_prop(
                                ui,
                                session,
                                &obj.id,
                                "bevelTension",
                                "bevel tension (Text+)",
                                0.5,
                                ScrubConfig::new(0.0, 1.0, 0.05, 2),
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                            );
                        }
                    },
                );
            }

            if obj.kind.contains("light") {
                cm3_collapsible_section(ui, Id::unique("obj_light"), "Light", true, theme, |ui| {
                    show_session_color_prop(
                        ui,
                        session,
                        &obj.id,
                        "color",
                        "color",
                        [1.0, 1.0, 1.0],
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                    show_session_number_prop(
                        ui,
                        session,
                        &obj.id,
                        "intensity",
                        "intensity",
                        1.0,
                        ScrubConfig::new(0.0, 100.0, 0.1, 2),
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                    );
                    show_session_number_prop(
                        ui,
                        session,
                        &obj.id,
                        "distance",
                        "range",
                        100.0,
                        ScrubConfig::new(0.0, 10_000.0, 1.0, 1),
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                    );
                });
            } else if obj.kind.contains("camera") {
                cm3_collapsible_section(
                    ui,
                    Id::unique("obj_camera"),
                    "Camera",
                    true,
                    theme,
                    |ui| {
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            "fov",
                            "fov (deg)",
                            45.0,
                            ScrubConfig::new(1.0, 179.0, 0.5, 1),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            "near",
                            "near",
                            0.1,
                            ScrubConfig::new(0.01, 1000.0, 0.1, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            "far",
                            "far",
                            5000.0,
                            ScrubConfig::new(1.0, 100_000.0, 10.0, 0),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                        );
                        if session.plugins.is_enabled("camera-plus") {
                            show_session_number_prop(
                                ui,
                                session,
                                &obj.id,
                                "dofFocusDistance",
                                "DOF focus",
                                80.0,
                                ScrubConfig::new(0.1, 5000.0, 1.0, 1),
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                            );
                            show_session_number_prop(
                                ui,
                                session,
                                &obj.id,
                                "dofAperture",
                                "DOF aperture",
                                2.8,
                                ScrubConfig::new(0.5, 32.0, 0.1, 2),
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                            );
                        }
                    },
                );
            } else {
                // Material section for mesh/geometry objects
                cm3_collapsible_section(
                    ui,
                    Id::unique("obj_material"),
                    "Material",
                    true,
                    theme,
                    |ui| {
                        let mat_options: Vec<&str> = if session.plugins.is_enabled("material-plus")
                        {
                            vec![
                                "single color",
                                "texture",
                                "video",
                                "custom PBR",
                                "custom shader",
                                "PBR+ (Material+)",
                                "Matcap (Material+)",
                                "Image+ (Material+)",
                                "Node Material",
                            ]
                        } else {
                            vec![
                                "single color",
                                "texture",
                                "video",
                                "custom PBR",
                                "custom shader",
                            ]
                        };
                        show_session_option_prop(
                            ui,
                            session,
                            &obj.id,
                            "material.type",
                            "material type",
                            &mat_options,
                            3,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                        );
                        show_session_color_prop(
                            ui,
                            session,
                            &obj.id,
                            "material.color",
                            "color",
                            [1.0, 1.0, 1.0],
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            "material.roughness",
                            "roughness",
                            0.5,
                            ScrubConfig::new(0.0, 1.0, 0.05, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            "material.metalness",
                            "metalness",
                            0.5,
                            ScrubConfig::new(0.0, 1.0, 0.05, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            "material.opacity",
                            "opacity",
                            1.0,
                            ScrubConfig::new(0.0, 1.0, 0.05, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                        );
                    },
                );
            }

            // Constraints section (when `constraints` plugin is enabled)
            if session.plugins.is_enabled("constraints") {
                cm3_collapsible_section(
                    ui,
                    Id::unique("obj_constraints"),
                    "Constraints",
                    false,
                    theme,
                    |ui| {
                        show_session_option_prop(
                            ui,
                            session,
                            &obj.id,
                            "constraint.trackTo.enabled",
                            "track to",
                            &["off", "on"],
                            0,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            "constraint.trackTo.influence",
                            "track influence",
                            1.0,
                            ScrubConfig::new(0.0, 1.0, 0.05, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                        );
                        show_session_option_prop(
                            ui,
                            session,
                            &obj.id,
                            "constraint.follow.enabled",
                            "follow",
                            &["off", "on"],
                            0,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                        );
                    },
                );
            }
        });
}

// ---------------------------------------------------------------------------
// Tab 5: Effects (fx) — 40/60 Auto-Split Effect Stack + Dynamic Inspector
// ---------------------------------------------------------------------------

const fn effect_category_label(cat: EffectCategory) -> &'static str {
    match cat {
        EffectCategory::GlowAndLight => "Glow & Light",
        EffectCategory::BlurAndSharpen => "Blur & Sharpen",
        EffectCategory::ColorAndTone => "Color & Tone",
        EffectCategory::KeyingAndMasking => "Keying & Masking",
        EffectCategory::OverlayAndComposite => "Overlay & Composite",
        EffectCategory::DistortAndTransform => "Distort & Transform",
        EffectCategory::StylizeAndGlitch => "Stylize & Glitch",
        EffectCategory::Temporal => "Temporal",
        EffectCategory::NodeAndGrading => "Node & Grading",
    }
}

fn build_effect_picker_items(session: &Session) -> Vec<FuzzyItem> {
    let mut items = Vec::with_capacity(64);

    // Custom Shader & Effect Group (CM3 native)
    items.push(FuzzyItem {
        id: "customshader".into(),
        category: "Custom & Groups".into(),
        name: "Custom Shader".into(),
        desc: "GLSL ES 1.00 fragment shader effect".into(),
    });
    items.push(FuzzyItem {
        id: "effectgroup".into(),
        category: "Custom & Groups".into(),
        name: "Effect Group".into(),
        desc: "Macro container grouping multiple effects".into(),
    });

    // All 44 standard CM3 effects
    for desc in standard_effect_catalog() {
        items.push(FuzzyItem {
            id: desc.id.to_owned(),
            category: effect_category_label(desc.category).to_owned(),
            name: desc.display_name.to_owned(),
            desc: format!("CM3 standard ({} pass)", desc.passes),
        });
    }

    // Zoidium plugin effects gated dynamically by enabled plugins
    for desc in zoidium_effect_catalog() {
        let allowed = desc
            .plugin_id
            .is_none_or(|pid| session.plugins.is_enabled(pid));
        if allowed {
            let pid = desc.plugin_id.unwrap_or("zoidium");
            items.push(FuzzyItem {
                id: format!("zoidium:{pid}/{}", desc.id),
                category: format!("Plugin: {}", effect_category_label(desc.category)),
                name: desc.display_name.to_owned(),
                desc: format!("from {pid}"),
            });
        }
    }

    // ZSMB lightweight plugin effect when enabled
    if session.plugins.is_enabled("smart-motion-blur-lite") {
        items.push(FuzzyItem {
            id: "zoidium:smart-motion-blur-lite/zsmb".into(),
            category: "Plugin: Blur & Sharpen".into(),
            name: "ZSMB (Smart Motion Blur Lite)".into(),
            desc: "Single-neighbor optical-flow motion blur".into(),
        });
    }

    // Layer Input experimental effect when enabled
    if session.plugins.is_enabled("layer-input") {
        items.push(FuzzyItem {
            id: "zoidium:layer-input/layer_input".into(),
            category: "Experimental".into(),
            name: "Layer Input".into(),
            desc: "Cross-track texture composite".into(),
        });
    }

    items
}

fn show_effects_tab(
    ui: &mut Ui,
    panel_rect: Rect,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let Some(clip_id) = session.selection.clip_ids.first().cloned() else {
        empty_panel_message(ui, "select a clip", theme);
        return;
    };
    let Some(clip) = find_clip_clone(session, &clip_id) else {
        empty_panel_message(ui, "select a clip", theme);
        return;
    };

    let (top_rect, bot_rect) = begin_split_panel(
        ui,
        Id::unique("effects_40_60_split"),
        panel_rect,
        &mut state.effects_split,
    );
    show_split_pane(
        ui,
        Id::unique("effects_40_60_split_p0"),
        top_rect,
        |ui, _top_rect| {
            // Pane 0 (40%): Effect Stack or Inline Fuzzy Add Effect Picker
            if state.add_effect_open {
                let items = build_effect_picker_items(session);
                match cm3_fuzzy_picker(
                    ui,
                    Id::unique("add_effect_picker"),
                    &items,
                    &mut state.add_effect_picker,
                    theme,
                ) {
                    FuzzyPickerEvent::Cancelled => {
                        state.add_effect_open = false;
                    }
                    FuzzyPickerEvent::Selected(kind) => {
                        let display = items.iter().find(|i| i.id == kind).map(|i| i.name.clone());
                        if let Ok(res) = session.execute(Command::AddEffect {
                            clip_id: clip_id.clone(),
                            kind: kind.clone(),
                            name: display,
                        }) && let Some(fx_id) = res.get("effect_id").and_then(Value::as_str)
                        {
                            // Initialize default Custom Shader source if adding Custom Shader
                            if kind == "customshader" {
                                let _ = session.execute(Command::SetPropertyValue {
                                    target_id: fx_id.to_owned(),
                                    property: "fragShader".to_owned(),
                                    value: json!(DEFAULT_CUSTOM_SHADER_GLSL),
                                });
                            }
                            let _ = session.selection.select_effect(fx_id, &kind, false);
                        }
                        state.add_effect_open = false;
                    }
                    FuzzyPickerEvent::None => {}
                }
                return;
            }

            // Header bar with `+` add button
            ui.horizontal(|ui| {
                let hdr_rect =
                    Rect::from_min_size(ui.cursor().min, Vec2::new(ui.available_width(), 28.0));
                ui.painter()
                    .rect_filled(hdr_rect, CornerRadius::ZERO, BG_TITLE);
                ui.allocate_ui_with_layout(
                    Vec2::new(ui.available_width(), 28.0),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new("Effects")
                                .font(theme.font(FONT_TITLE))
                                .color(TEXT_STRONG),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add_space(4.0);
                            if cm3_icon_button(
                                ui,
                                Cm3Icon::Add,
                                17.0,
                                TEXT_PRIMARY,
                                "add effect",
                                theme,
                            ) {
                                state.add_effect_open = true;
                                state.add_effect_picker = FuzzyPickerState {
                                    query: String::new(),
                                    active_index: 0,
                                    just_opened: true,
                                };
                            }
                        });
                    },
                );
            });

            // Effect stack rows
            ScrollArea::vertical()
                .id_salt("effects_list_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = Vec2::ZERO;
                    if clip.effects.is_empty() {
                        empty_panel_message(ui, "no effects — click + to add", theme);
                        return;
                    }

                    let mut reorder_req: Option<(String, usize)> = None;
                    let local_frame = session.playback.current_frame - clip.start;

                    for (idx, fx) in clip.effects.iter().enumerate() {
                        let selected = session.selection.effect_ids.contains(&fx.id);
                        let enabled =
                            eval_prop_u32(fx.properties.get("enabled"), local_frame, 1) != 0;

                        let row_w = ui.available_width();
                        let (row_rect, row_resp) =
                            ui.allocate_exact_size(Vec2::new(row_w, 26.0), Sense::click_and_drag());

                        if selected {
                            ui.painter()
                                .rect_filled(row_rect, CornerRadius::ZERO, theme.accent());
                        } else if row_resp.hovered() {
                            ui.painter()
                                .rect_filled(row_rect, CornerRadius::ZERO, BG_HOVER);
                        }
                        ui.painter().line_segment(
                            [row_rect.left_bottom(), row_rect.right_bottom()],
                            Stroke::new(1.0, BORDER_ROW_SOFT),
                        );

                        // Right action buttons: enable eye, duplicate, remove
                        let rm_rect = Rect::from_min_size(
                            Pos2::new(row_rect.right() - 24.0, row_rect.top() + 3.0),
                            Vec2::splat(20.0),
                        );
                        let dup_rect = Rect::from_min_size(
                            Pos2::new(rm_rect.left() - 22.0, row_rect.top() + 3.0),
                            Vec2::splat(20.0),
                        );
                        let eye_rect = Rect::from_min_size(
                            Pos2::new(dup_rect.left() - 22.0, row_rect.top() + 3.0),
                            Vec2::splat(20.0),
                        );

                        let eye_resp = ui
                            .interact(eye_rect, Id::unique(("fx_eye", &fx.id)), Sense::click())
                            .on_hover_cursor(CursorIcon::PointingHand)
                            .on_hover_text(if enabled {
                                "disable effect"
                            } else {
                                "enable effect"
                            });
                        let dup_resp = ui
                            .interact(dup_rect, Id::unique(("fx_dup", &fx.id)), Sense::click())
                            .on_hover_cursor(CursorIcon::PointingHand)
                            .on_hover_text("duplicate effect");
                        let rm_resp = ui
                            .interact(rm_rect, Id::unique(("fx_rm", &fx.id)), Sense::click())
                            .on_hover_cursor(CursorIcon::PointingHand)
                            .on_hover_text("delete effect");

                        paint_cm3_icon(
                            ui.painter(),
                            eye_rect.shrink(2.0),
                            Cm3Icon::Eye,
                            if enabled { TEXT_PRIMARY } else { TEXT_FAINT },
                        );
                        paint_cm3_icon(
                            ui.painter(),
                            dup_rect.shrink(1.5),
                            Cm3Icon::Duplicate,
                            if dup_resp.hovered() {
                                TEXT_BRIGHT
                            } else {
                                TEXT_PRIMARY
                            },
                        );
                        paint_cm3_icon(
                            ui.painter(),
                            rm_rect.shrink(1.5),
                            Cm3Icon::Remove,
                            if rm_resp.hovered() {
                                DANGER_TEXT
                            } else {
                                TEXT_PRIMARY
                            },
                        );

                        if eye_resp.clicked() {
                            let _ = session.execute(Command::SetPropertyValue {
                                target_id: fx.id.clone(),
                                property: "enabled".to_owned(),
                                value: json!(if enabled { 0 } else { 1 }),
                            });
                        }
                        if dup_resp.clicked() {
                            let _ = session.execute(Command::DuplicateEffect {
                                clip_id: clip_id.clone(),
                                effect_id: fx.id.clone(),
                            });
                        }
                        if rm_resp.clicked() {
                            let _ = session.execute(Command::DeleteEffect {
                                clip_id: clip_id.clone(),
                                effect_id: fx.id.clone(),
                            });
                        }

                        // Label
                        let label_rect = Rect::from_min_max(
                            Pos2::new(row_rect.left() + 8.0, row_rect.top()),
                            Pos2::new(
                                (eye_rect.left() - 4.0).max(row_rect.left() + 8.0),
                                row_rect.bottom(),
                            ),
                        );
                        ui.painter().with_clip_rect(label_rect).text(
                            Pos2::new(label_rect.left(), label_rect.center().y),
                            Align2::LEFT_CENTER,
                            &fx.name,
                            theme.font(FONT_BODY),
                            if !enabled {
                                TEXT_DISABLED
                            } else if selected {
                                TEXT_BRIGHT
                            } else {
                                TEXT_PRIMARY
                            },
                        );

                        if row_resp.clicked()
                            && !eye_resp.clicked()
                            && !dup_resp.clicked()
                            && !rm_resp.clicked()
                        {
                            let shift = ui.input(|i| i.modifiers.shift);
                            if let Err(err) =
                                session.selection.select_effect(&fx.id, &fx.kind, shift)
                            {
                                actions.push(PanelAction::StatusMessage(err.to_string()));
                            }
                        }

                        // Drag-to-reorder effect stack (5px vertical threshold)
                        if row_resp.drag_started()
                            && let Some(pos) = ui.input(|i| i.pointer.interact_pos())
                        {
                            state.effect_drag = Some(TreeDragState {
                                dragging_id: fx.id.clone(),
                                start_y: pos.y,
                                active: false,
                            });
                        }
                        if let Some(drag) = state.effect_drag.as_mut()
                            && let Some(pos) = ui.input(|i| i.pointer.latest_pos())
                        {
                            if !drag.active && (pos.y - drag.start_y).abs() >= 5.0 {
                                drag.active = true;
                            }
                            if drag.active && drag.dragging_id != fx.id && row_rect.contains(pos) {
                                let before = pos.y < row_rect.center().y;
                                let bar_y = if before {
                                    row_rect.top()
                                } else {
                                    row_rect.bottom() - 3.0
                                };
                                ui.painter().rect_filled(
                                    Rect::from_min_size(
                                        Pos2::new(row_rect.left(), bar_y),
                                        Vec2::new(row_rect.width(), 3.0),
                                    ),
                                    CornerRadius::ZERO,
                                    Color32::from_white_alpha(64),
                                );
                                if !ui.input(|i| i.pointer.primary_down()) {
                                    let target_idx = if before { idx } else { idx + 1 };
                                    reorder_req = Some((drag.dragging_id.clone(), target_idx));
                                }
                            }
                        }
                    }

                    if !ui.input(|i| i.pointer.primary_down()) {
                        if let Some((fx_id, target_idx)) = reorder_req {
                            let _ = session.execute(Command::ReorderEffect {
                                clip_id: clip_id.clone(),
                                effect_id: fx_id,
                                index: target_idx,
                            });
                        }
                        state.effect_drag = None;
                    }
                });
        },
    );
    show_split_pane(
        ui,
        Id::unique("effects_40_60_split_p1"),
        bot_rect,
        |ui, bot_rect| {
            // Pane 1 (60%): Dynamic Effect Parameter Inspector
            show_effect_inspector(
                ui,
                bot_rect,
                viewport_rect,
                &clip,
                session,
                theme,
                state,
                actions,
            );
        },
    );
}

const DEFAULT_CUSTOM_SHADER_GLSL: &str = r#"uniform sampler2D tDiffuse;
uniform vec2 resolution;
varying vec2 vUv;
varying vec2 vUvScaled;

void main() {
    gl_FragColor = texture2D(tDiffuse, vUvScaled);
}
"#;

#[allow(clippy::too_many_arguments)]
fn show_effect_inspector(
    ui: &mut Ui,
    inspector_rect: Rect,
    viewport_rect: Rect,
    clip: &SessionClip,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let Some(fx_id) = session.selection.effect_ids.first().cloned() else {
        empty_panel_message(ui, "select an effect", theme);
        return;
    };
    let Some(fx) = clip.effects.iter().find(|e| e.id == fx_id).cloned() else {
        empty_panel_message(ui, "select an effect", theme);
        return;
    };

    cm3_column_adjust_splitter(
        ui,
        Id::unique("fx_inspector_col_splitter"),
        inspector_rect,
        &mut state.prop_columns,
    );

    let local_frame = session.playback.current_frame - clip.start;
    let clip_start = clip.start;

    ScrollArea::vertical()
        .id_salt("fx_inspector_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            property_header_with_reset(ui, &fx.name, &fx.id, session, theme);

            // Always show Enabled toggle
            show_session_option_prop(
                ui,
                session,
                &fx.id,
                "enabled",
                "enabled",
                &["off", "on"],
                1,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
            );

            // Custom Shader specialized inspector + Shader+ controls
            if fx.kind == "customshader" {
                let cur_code = eval_prop_str(
                    fx.properties.get("fragShader"),
                    local_frame,
                    DEFAULT_CUSTOM_SHADER_GLSL,
                );
                cm3_property_row(
                    ui,
                    "fragment shader",
                    state.prop_columns,
                    0,
                    theme,
                    |_| {},
                    |ui| {
                        ui.horizontal(|ui| {
                            if cm3_icon_button(
                                ui,
                                Cm3Icon::Code,
                                17.0,
                                theme.value(),
                                "open shader code editor",
                                theme,
                            ) || cm3_prop_button(
                                ui,
                                "edit shader",
                                ButtonVariant::Normal,
                                false,
                                theme,
                            )
                            .clicked()
                            {
                                state.shader_editor = Some(ShaderEditorTarget {
                                    effect_id: fx.id.clone(),
                                    effect_name: fx.name.clone(),
                                    code: cur_code.clone(),
                                    font_px: 14.0,
                                    error_message: None,
                                });
                                let spec =
                                    FloatingWindowSpec::new("shader_editor", "Shader Editor")
                                        .with_subtitle(fx.name.clone())
                                        .with_default_size(Vec2::new(560.0, 440.0));
                                state.floating_windows.open_or_focus(&spec, viewport_rect);
                            }
                        });
                    },
                );

                // Shader+ Custom Property builder & live uniform list
                let shader_plus = session.plugins.is_enabled("shader-plus");
                let sec_title = if shader_plus {
                    "Custom Properties (Shader+)"
                } else {
                    "Custom Properties"
                };
                cm3_collapsible_section(
                    ui,
                    Id::unique("shader_custom_props"),
                    sec_title,
                    true,
                    theme,
                    |ui| {
                        let prop_kinds: &[&str] = if shader_plus {
                            &[
                                "Number / Slider",
                                "Vector2",
                                "Vector3",
                                "Color (RGB)",
                                "Color + Alpha",
                                "Toggle",
                                "Angle (deg)",
                                "Seed",
                            ]
                        } else {
                            &["Number", "Vector2", "Vector3", "Color (RGB)"]
                        };
                        if state.new_shader_prop_kind >= prop_kinds.len() {
                            state.new_shader_prop_kind = 0;
                        }
                        ui.horizontal(|ui| {
                            Frame::NONE
                                .fill(BG_INPUT)
                                .stroke(Stroke::new(1.0, BORDER_CONTROL))
                                .inner_margin(egui::Margin::symmetric(4, 2))
                                .show(ui, |ui| {
                                    ui.add(
                                        TextEdit::singleline(&mut state.new_shader_prop_name)
                                            .font(theme.font(FONT_SMALL))
                                            .text_color(theme.value())
                                            .desired_width(100.0),
                                    );
                                });
                            cm3_dropdown(
                                ui,
                                Id::unique("shader_new_prop_kind"),
                                &mut state.new_shader_prop_kind,
                                prop_kinds,
                                false,
                                viewport_rect,
                                theme,
                            );
                            if cm3_icon_button(
                                ui,
                                Cm3Icon::Add,
                                17.0,
                                TEXT_PRIMARY,
                                "add uniform property",
                                theme,
                            ) {
                                let name = state.new_shader_prop_name.trim();
                                if !name.is_empty() {
                                    let default_val = match state.new_shader_prop_kind {
                                        1 => json!([0.0, 0.0]),
                                        2 => json!([0.0, 0.0, 0.0]),
                                        3 => json!([1.0, 1.0, 1.0]),
                                        4 => json!([1.0, 1.0, 1.0, 1.0]),
                                        5 => json!(1),
                                        _ => json!(0.5),
                                    };
                                    let _ = session.execute(Command::SetPropertyValue {
                                        target_id: fx.id.clone(),
                                        property: format!("u_{name}"),
                                        value: default_val,
                                    });
                                }
                            }
                        });

                        let custom_keys: Vec<(String, Value)> = fx
                            .properties
                            .iter()
                            .filter(|(k, _)| k.starts_with("u_"))
                            .map(|(k, p)| {
                                let val = p
                                    .keyframes
                                    .first()
                                    .map_or_else(|| p.default_value.clone(), |kf| kf.value.clone());
                                (k.clone(), val)
                            })
                            .collect();

                        for (key, val) in custom_keys {
                            let label = key.strip_prefix("u_").unwrap_or(&key);
                            render_dynamic_json_param(
                                ui,
                                session,
                                &fx.id,
                                &key,
                                label,
                                &val,
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );
                        }
                    },
                );
                return;
            }

            // Standard CM3 & Zoidium plugin effects: look up canonical schema via `create_effect_by_id`
            let base_id = fx.kind.rsplit('/').next().unwrap_or(fx.kind.as_str());
            if let Some(template) = create_effect_by_id(base_id) {
                for (param_key, track) in &template.params {
                    render_template_param(
                        ui,
                        session,
                        &fx.id,
                        param_key,
                        &track.value,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
            } else {
                // Fallback: render any dynamic properties present on the SessionEffect
                let keys: Vec<(String, Value)> = fx
                    .properties
                    .iter()
                    .filter(|(k, _)| k.as_str() != "enabled")
                    .map(|(k, p)| {
                        let val = p
                            .keyframes
                            .first()
                            .map_or_else(|| p.default_value.clone(), |kf| kf.value.clone());
                        (k.clone(), val)
                    })
                    .collect();
                if keys.is_empty() {
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("no additional parameters")
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_FAINT),
                    );
                } else {
                    for (key, val) in keys {
                        render_dynamic_json_param(
                            ui,
                            session,
                            &fx.id,
                            &key,
                            &key,
                            &val,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                }
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn render_template_param(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    key: &str,
    default_val: &PropertyValue,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    match default_val {
        PropertyValue::Number(def) => {
            let cfg = ScrubConfig::new(-10_000.0, 10_000.0, 0.05, 3);
            show_session_number_prop(
                ui,
                session,
                target_id,
                key,
                key,
                *def,
                cfg,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
            );
        }
        PropertyValue::Vec2(def) => {
            show_session_vec2_prop(
                ui,
                session,
                target_id,
                key,
                key,
                *def,
                ScrubConfig::new(-10_000.0, 10_000.0, 0.05, 3),
                key.contains("scale") || key.contains("size") || key.contains("multiplier"),
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
            );
        }
        PropertyValue::Vec3(def) => {
            show_session_vec3_prop(
                ui,
                session,
                target_id,
                key,
                key,
                *def,
                ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 3),
                false,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
            );
        }
        PropertyValue::Color(def) => {
            show_session_color_prop(
                ui,
                session,
                target_id,
                key,
                key,
                *def,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::Option(def) => {
            let opts: &[&str] = match key {
                "wrap" => &["clamp", "tile", "reflect"],
                "repeat" => &["tile", "reflect"],
                "mask" => &["result", "mask"],
                "gradientType" => &["linear", "radial", "angular", "reflected", "diamond"],
                "direction" => &["vertical", "horizontal"],
                "affect" => &["shadows", "highlights"],
                "mode" => &["mode 0", "mode 1", "mode 2"],
                "cameraType" => &["perspective", "orthographic"],
                _ => &["off", "on"],
            };
            show_session_option_prop(
                ui,
                session,
                target_id,
                key,
                key,
                opts,
                *def,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
            );
        }
        PropertyValue::List(def) if key == "blending" => {
            let prop_opt = lookup_session_prop(session, target_id, key);
            let cur = eval_prop_str(prop_opt.as_ref(), local_frame, def);
            let mut idx = BlendMode::ALL
                .iter()
                .position(|b| b.token() == cur)
                .unwrap_or(3);
            let labels: Vec<&str> = BlendMode::ALL.iter().map(|b| b.display_name()).collect();
            let changed = cm3_property_row(
                ui,
                key,
                state.prop_columns,
                0,
                theme,
                |_| {},
                |ui| {
                    cm3_dropdown(
                        ui,
                        Id::unique((target_id, key)),
                        &mut idx,
                        &labels,
                        false,
                        viewport_rect,
                        theme,
                    )
                },
            );
            if changed {
                let _ = session.execute(Command::SetPropertyValue {
                    target_id: target_id.to_owned(),
                    property: key.to_owned(),
                    value: json!(BlendMode::ALL[idx].token()),
                });
            }
        }
        PropertyValue::Gradient(def) => {
            let prop_opt = lookup_session_prop(session, target_id, key);
            let mut stops = prop_opt
                .as_ref()
                .and_then(|p| p.keyframes.first())
                .and_then(|kf| serde_json::from_value::<Vec<GradientStop>>(kf.value.clone()).ok())
                .unwrap_or_else(|| def.clone());
            let ev = cm3_property_row(
                ui,
                key,
                state.prop_columns,
                0,
                theme,
                |_| {},
                |ui| {
                    cm3_gradient_bar(
                        ui,
                        Id::unique((target_id, key)),
                        &mut stops,
                        viewport_rect,
                        theme,
                    )
                },
            );
            if ev.committed
                && let Ok(v) = serde_json::to_value(&stops)
            {
                let _ = session.execute(Command::SetPropertyValue {
                    target_id: target_id.to_owned(),
                    property: key.to_owned(),
                    value: v,
                });
            }
        }
        PropertyValue::Text(def) | PropertyValue::List(def) | PropertyValue::Shader(def) => {
            show_session_text_prop(
                ui,
                session,
                target_id,
                key,
                key,
                def,
                local_frame,
                state,
                theme,
            );
        }
        PropertyValue::Asset(def) => {
            let prop_opt = lookup_session_prop(session, target_id, key);
            let cur = eval_prop_str(
                prop_opt.as_ref(),
                local_frame,
                def.as_deref().unwrap_or("(none)"),
            );
            cm3_property_row(
                ui,
                key,
                state.prop_columns,
                0,
                theme,
                |_| {},
                |ui| {
                    ui.horizontal(|ui| {
                        if cm3_icon_button(
                            ui,
                            Cm3Icon::Load,
                            17.0,
                            TEXT_PRIMARY,
                            "choose file...",
                            theme,
                        ) {
                            actions.push(PanelAction::ImportMedia);
                        }
                        if cur != "(none)"
                            && cm3_icon_button(
                                ui,
                                Cm3Icon::Remove,
                                17.0,
                                TEXT_PRIMARY,
                                "clear asset",
                                theme,
                            )
                        {
                            let _ = session.execute(Command::SetPropertyValue {
                                target_id: target_id.to_owned(),
                                property: key.to_owned(),
                                value: Value::Null,
                            });
                        }
                        let shown = if cur.chars().count() > 20 {
                            let trunc: String = cur.chars().take(17).collect();
                            format!("{trunc}...")
                        } else {
                            cur.clone()
                        };
                        ui.label(
                            egui::RichText::new(shown)
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_DISABLED),
                        )
                        .on_hover_text(cur);
                    });
                },
            );
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn render_dynamic_json_param(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    key: &str,
    label: &str,
    val: &Value,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    match val {
        Value::Number(n) => {
            show_session_number_prop(
                ui,
                session,
                target_id,
                key,
                label,
                n.as_f64().unwrap_or(0.0),
                ScrubConfig::new(-10_000.0, 10_000.0, 0.05, 3),
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
            );
        }
        Value::Array(arr) if arr.len() == 2 => {
            let def = [
                arr[0].as_f64().unwrap_or(0.0),
                arr[1].as_f64().unwrap_or(0.0),
            ];
            show_session_vec2_prop(
                ui,
                session,
                target_id,
                key,
                label,
                def,
                ScrubConfig::new(-10_000.0, 10_000.0, 0.05, 3),
                false,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
            );
        }
        Value::Array(arr) if arr.len() == 3 => {
            let def = [
                arr[0].as_f64().unwrap_or(0.0),
                arr[1].as_f64().unwrap_or(0.0),
                arr[2].as_f64().unwrap_or(0.0),
            ];
            if label.to_ascii_lowercase().contains("color") {
                show_session_color_prop(
                    ui,
                    session,
                    target_id,
                    key,
                    label,
                    def,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            } else {
                show_session_vec3_prop(
                    ui,
                    session,
                    target_id,
                    key,
                    label,
                    def,
                    ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 3),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                );
            }
        }
        Value::String(s) => {
            show_session_text_prop(
                ui,
                session,
                target_id,
                key,
                label,
                s,
                local_frame,
                state,
                theme,
            );
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tab 6: Export (export) — Device & Frame Export (Spec 04 §4.1)
// ---------------------------------------------------------------------------

fn show_export_tab(
    ui: &mut Ui,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    ScrollArea::vertical()
        .id_salt("export_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            Frame::NONE
                .inner_margin(egui::Margin::same(6))
                .show(ui, |ui| {
                    // Device / Frame sub-mode switcher
                    ui.horizontal(|ui| {
                        let dev_var = if state.export_settings.mode == ExportMode::Device {
                            ButtonVariant::Primary
                        } else {
                            ButtonVariant::Normal
                        };
                        let frm_var = if state.export_settings.mode == ExportMode::Frame {
                            ButtonVariant::Primary
                        } else {
                            ButtonVariant::Normal
                        };
                        if cm3_prop_button(ui, "Device Video", dev_var, false, theme).clicked() {
                            state.export_settings.mode = ExportMode::Device;
                        }
                        if cm3_prop_button(ui, "Single Frame", frm_var, false, theme).clicked() {
                            state.export_settings.mode = ExportMode::Frame;
                        }
                    });

                    ui.add_space(6.0);

                    // FFmpeg status bar
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

                    ui.add_space(6.0);

                    if state.export_settings.mode == ExportMode::Device {
                        // Container
                        let mut c_idx = ExportContainer::ALL
                            .iter()
                            .position(|&c| c == state.export_settings.container)
                            .unwrap_or(0);
                        let changed = cm3_property_row(
                            ui,
                            "container",
                            state.prop_columns,
                            0,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_dropdown(
                                    ui,
                                    Id::unique("exp_container"),
                                    &mut c_idx,
                                    &ExportContainer::LABELS,
                                    false,
                                    viewport_rect,
                                    theme,
                                )
                            },
                        );
                        if changed {
                            state.export_settings.container = ExportContainer::ALL[c_idx];
                            match state.export_settings.container {
                                ExportContainer::Mp4 => {
                                    state.export_settings.video_codec = ExportVideoCodec::H264;
                                    state.export_settings.audio_codec = ExportAudioCodec::AacLc;
                                }
                                ExportContainer::WebM => {
                                    state.export_settings.video_codec = ExportVideoCodec::Vp9;
                                    state.export_settings.audio_codec = ExportAudioCodec::Opus;
                                }
                                ExportContainer::Mkv => {
                                    state.export_settings.video_codec = ExportVideoCodec::H264;
                                    state.export_settings.audio_codec = ExportAudioCodec::PcmS16Le;
                                }
                                ExportContainer::PngSequence => {
                                    state.export_settings.video_codec = ExportVideoCodec::Png;
                                    state.export_settings.audio_codec = ExportAudioCodec::None;
                                }
                            }
                        }

                        // Video Codec
                        let mut vc_idx = ExportVideoCodec::ALL
                            .iter()
                            .position(|&c| c == state.export_settings.video_codec)
                            .unwrap_or(0);
                        if cm3_property_row(
                            ui,
                            "video codec",
                            state.prop_columns,
                            0,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_dropdown(
                                    ui,
                                    Id::unique("exp_vcodec"),
                                    &mut vc_idx,
                                    &ExportVideoCodec::LABELS,
                                    false,
                                    viewport_rect,
                                    theme,
                                )
                            },
                        ) {
                            state.export_settings.video_codec = ExportVideoCodec::ALL[vc_idx];
                        }

                        // Rate Control
                        let mut rc_idx = ExportRateControl::ALL
                            .iter()
                            .position(|&r| r == state.export_settings.rate_control)
                            .unwrap_or(0);
                        if cm3_property_row(
                            ui,
                            "rate control",
                            state.prop_columns,
                            0,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_dropdown(
                                    ui,
                                    Id::unique("exp_rc"),
                                    &mut rc_idx,
                                    &ExportRateControl::LABELS,
                                    false,
                                    viewport_rect,
                                    theme,
                                )
                            },
                        ) {
                            state.export_settings.rate_control = ExportRateControl::ALL[rc_idx];
                        }

                        match state.export_settings.rate_control {
                            ExportRateControl::Crf => {
                                let mut crf = f64::from(state.export_settings.crf);
                                let ev = cm3_property_row(
                                    ui,
                                    "quality (CRF)",
                                    state.prop_columns,
                                    0,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        cm3_scrub_number(
                                            ui,
                                            Id::unique("exp_crf"),
                                            &mut crf,
                                            ScrubConfig::new(0.0, 63.0, 1.0, 0),
                                            theme,
                                        )
                                    },
                                );
                                if ev.changed || ev.committed {
                                    state.export_settings.crf = crf.round().clamp(0.0, 63.0) as u32;
                                }
                            }
                            ExportRateControl::Bitrate => {
                                let mut mbps = state.export_settings.bitrate_mbps;
                                let ev = cm3_property_row(
                                    ui,
                                    "bitrate (Mbps)",
                                    state.prop_columns,
                                    0,
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
                                if cm3_property_row(
                                    ui,
                                    "preset",
                                    state.prop_columns,
                                    0,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        cm3_dropdown(
                                            ui,
                                            Id::unique("exp_cm3_preset"),
                                            &mut p_idx,
                                            &ExportCm3Preset::LABELS,
                                            false,
                                            viewport_rect,
                                            theme,
                                        )
                                    },
                                ) {
                                    state.export_settings.cm3_preset = ExportCm3Preset::ALL[p_idx];
                                }
                            }
                        }

                        // Audio Codec
                        let mut ac_idx = ExportAudioCodec::ALL
                            .iter()
                            .position(|&a| a == state.export_settings.audio_codec)
                            .unwrap_or(0);
                        if cm3_property_row(
                            ui,
                            "audio codec",
                            state.prop_columns,
                            0,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_dropdown(
                                    ui,
                                    Id::unique("exp_acodec"),
                                    &mut ac_idx,
                                    &ExportAudioCodec::LABELS,
                                    false,
                                    viewport_rect,
                                    theme,
                                )
                            },
                        ) {
                            state.export_settings.audio_codec = ExportAudioCodec::ALL[ac_idx];
                        }
                    }

                    // Resolution, FPS, Range
                    let mut res = [
                        f64::from(state.export_settings.width),
                        f64::from(state.export_settings.height),
                    ];
                    let ev = cm3_property_row(
                        ui,
                        "resolution",
                        state.prop_columns,
                        0,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_vector2(
                                ui,
                                Id::unique("exp_res"),
                                &mut res,
                                ScrubConfig::new(16.0, 7680.0, 10.0, 0),
                                false,
                                theme,
                            )
                        },
                    );
                    if ev.changed || ev.committed {
                        state.export_settings.width = res[0].round().clamp(16.0, 7680.0) as u32;
                        state.export_settings.height = res[1].round().clamp(16.0, 7680.0) as u32;
                    }

                    let mut fps = state.export_settings.fps;
                    let ev = cm3_property_row(
                        ui,
                        "frame rate",
                        state.prop_columns,
                        0,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_scrub_number(
                                ui,
                                Id::unique("exp_fps"),
                                &mut fps,
                                ScrubConfig::new(1.0, 240.0, 1.0, 2),
                                theme,
                            )
                        },
                    );
                    if ev.changed || ev.committed {
                        state.export_settings.fps = fps;
                    }

                    let mut range = [
                        state.export_settings.start_frame,
                        state.export_settings.end_frame,
                    ];
                    let ev = cm3_property_row(
                        ui,
                        "range (frames)",
                        state.prop_columns,
                        0,
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

                    if cm3_prop_button(ui, "sync from sequence", ButtonVariant::Normal, true, theme)
                        .clicked()
                    {
                        let [w, h] = session.document().resolution;
                        state.export_settings.width = w;
                        state.export_settings.height = h;
                        state.export_settings.fps = session.document().fps;
                        state.export_settings.start_frame = 0.0;
                        state.export_settings.end_frame = session.document().sequence_length;
                    }

                    ui.add_space(10.0);

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
                            format!("{:.0}%", p * 100.0),
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
                        ui.add_space(6.0);
                    }

                    if state.export_status.exporting {
                        if cm3_prop_button(ui, "cancel export", ButtonVariant::Danger, true, theme)
                            .clicked()
                        {
                            actions.push(PanelAction::CancelExport);
                        }
                    } else {
                        let label = if state.export_settings.mode == ExportMode::Device {
                            "start video render"
                        } else {
                            "export current frame (PNG)"
                        };
                        if cm3_prop_button(ui, label, ButtonVariant::Primary, true, theme).clicked()
                        {
                            actions.push(PanelAction::StartExport(state.export_settings.clone()));
                        }
                    }
                });
        });
}

// ---------------------------------------------------------------------------
// Tab 7: Plugins (Zoidium Extension Manager, Spec 04 §5.1)
// ---------------------------------------------------------------------------

fn show_plugins_tab(ui: &mut Ui, session: &mut Session, theme: &Cm3Theme, state: &mut PanelsState) {
    let active_deps = session.active_plugin_dependencies();
    let is_vanilla = session.plugins.is_vanilla_cm3_mode();

    ScrollArea::vertical()
        .id_salt("plugins_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            Frame::NONE
                .inner_margin(egui::Margin::same(6))
                .show(ui, |ui| {
                    ui.set_max_width(ui.available_width());
                    // Header actions: Vanilla CM3 Mode & Reset Defaults
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(if is_vanilla {
                                "Mode: Vanilla CM3"
                            } else {
                                "Mode: Zoidium Extensions"
                            })
                            .font(theme.font(FONT_SMALL))
                            .color(if is_vanilla {
                                TEXT_MUTED
                            } else {
                                theme.value()
                            }),
                        );
                    });
                    ui.add_space(4.0);

                    if cm3_prop_button(
                        ui,
                        "Vanilla CM3 Mode",
                        ButtonVariant::Normal,
                        true,
                        theme,
                    )
                    .on_hover_text("Disable all optional Zoidium plugins to restore Vanilla CM3")
                    .clicked()
                    {
                        match session.execute(Command::DisableAllOptionalPlugins) {
                            Ok(val) => {
                                let blocked_count = val
                                    .get("blocked")
                                    .and_then(Value::as_array)
                                    .map_or(0, Vec::len);
                                if blocked_count > 0 {
                                    state.plugin_notice = Some(format!(
                                        "{blocked_count} plugin(s) remained enabled because they are in use by the active project."
                                    ));
                                } else {
                                    state.plugin_notice =
                                        Some("Switched to Vanilla CM3 Mode.".to_owned());
                                }
                            }
                            Err(err) => {
                                state.plugin_notice = Some(err.to_string());
                            }
                        }
                    }
                    ui.add_space(3.0);
                    if cm3_prop_button(
                        ui,
                        "Reset Plugins to Defaults",
                        ButtonVariant::Normal,
                        true,
                        theme,
                    )
                    .clicked()
                    {
                        session.plugins.reset_to_defaults();
                        state.plugin_notice = None;
                    }

                    if let Some(notice) = state.plugin_notice.clone() {
                        ui.add_space(6.0);
                        warning_banner(ui, &notice, theme);
                    }
                    ui.add_space(6.0);

                    for category in [
                        PluginCategory::Utilities,
                        PluginCategory::Scene,
                        PluginCategory::Adjustment,
                        PluginCategory::Experimental,
                    ] {
                        let descriptors: Vec<_> = session
                            .plugins
                            .visible_in_category(category)
                            .into_iter()
                            .cloned()
                            .collect();
                        if descriptors.is_empty() {
                            continue;
                        }

                        let default_open = !category.collapsed_by_default();
                        cm3_collapsible_section(
                            ui,
                            Id::unique(("plugin_cat", category.display_name())),
                            category.display_name(),
                            default_open,
                            theme,
                            |ui| {
                                if category == PluginCategory::Experimental {
                                    warning_banner(
                                        ui,
                                        zc_plugin::EXPERIMENTAL_WARNING,
                                        theme,
                                    );
                                    ui.add_space(4.0);
                                }

                                for desc in descriptors {
                                    let enabled = session.plugins.is_enabled(&desc.id);
                                    let in_use = active_deps.contains(&desc.id);
                                    let locked = desc.always_enabled || (enabled && in_use);

                                    Frame::NONE
                                        .fill(BG_TITLE)
                                        .stroke(Stroke::new(1.0, BORDER_ROW_SOFT))
                                        .inner_margin(egui::Margin::symmetric(8, 6))
                                        .show(ui, |ui| {
                                            ui.set_min_width(ui.available_width());
                                            ui.horizontal(|ui| {
                                                // Accent dot
                                                let [r, g, b] = desc.accent_color;
                                                let (dot_rect, _) = ui.allocate_exact_size(
                                                    Vec2::splat(10.0),
                                                    Sense::hover(),
                                                );
                                                ui.painter().circle_filled(
                                                    dot_rect.center(),
                                                    4.0,
                                                    Color32::from_rgb(r, g, b),
                                                );

                                                ui.label(
                                                    egui::RichText::new(&desc.name)
                                                        .font(theme.font(FONT_BODY))
                                                        .color(TEXT_STRONG),
                                                );

                                                if in_use {
                                                    let (b_rect, _) = ui.allocate_exact_size(
                                                        Vec2::new(48.0, 16.0),
                                                        Sense::hover(),
                                                    );
                                                    paint_badge(
                                                        ui.painter(),
                                                        b_rect,
                                                        "IN USE",
                                                        theme.accent(),
                                                        TEXT_BRIGHT,
                                                    );
                                                }

                                                ui.with_layout(
                                                    egui::Layout::right_to_left(egui::Align::Center),
                                                    |ui| {
                                                        // 16x16 Zoidium checkbox tinted with accent
                                                        let (cb_rect, cb_resp) =
                                                            ui.allocate_exact_size(
                                                                Vec2::splat(18.0),
                                                                Sense::click(),
                                                            );
                                                        let cb_resp = cb_resp.on_hover_cursor(
                                                            if locked {
                                                                CursorIcon::NotAllowed
                                                            } else {
                                                                CursorIcon::PointingHand
                                                            },
                                                        );
                                                        let fill = if enabled {
                                                            if locked {
                                                                color32_with_alpha(
                                                                    theme.accent(),
                                                                    150,
                                                                )
                                                            } else {
                                                                theme.accent()
                                                            }
                                                        } else {
                                                            BG_INPUT
                                                        };
                                                        ui.painter().rect_filled(
                                                            cb_rect,
                                                            CornerRadius::ZERO,
                                                            fill,
                                                        );
                                                        ui.painter().rect_stroke(
                                                            cb_rect,
                                                            CornerRadius::ZERO,
                                                            Stroke::new(1.0, BORDER_CONTROL),
                                                            StrokeKind::Inside,
                                                        );
                                                        if enabled {
                                                            let c = cb_rect.center();
                                                            ui.painter().line_segment(
                                                                [
                                                                    Pos2::new(c.x - 4.0, c.y),
                                                                    Pos2::new(c.x - 1.0, c.y + 3.0),
                                                                ],
                                                                Stroke::new(2.0, TEXT_BRIGHT),
                                                            );
                                                            ui.painter().line_segment(
                                                                [
                                                                    Pos2::new(c.x - 1.0, c.y + 3.0),
                                                                    Pos2::new(c.x + 5.0, c.y - 4.0),
                                                                ],
                                                                Stroke::new(2.0, TEXT_BRIGHT),
                                                            );
                                                        }

                                                        if cb_resp.clicked() {
                                                            let cmd = if enabled {
                                                                Command::DisablePlugin {
                                                                    plugin_id: desc.id.clone(),
                                                                }
                                                            } else {
                                                                Command::EnablePlugin {
                                                                    plugin_id: desc.id.clone(),
                                                                }
                                                            };
                                                            match session.execute(cmd) {
                                                                Ok(_) => state.plugin_notice = None,
                                                                Err(err) => {
                                                                    state.plugin_notice =
                                                                        Some(err.to_string());
                                                                }
                                                            }
                                                        }
                                                    },
                                                );
                                            });

                                            ui.add(
                                                egui::Label::new(
                                                    egui::RichText::new(&desc.tagline)
                                                        .font(theme.font(FONT_SMALL))
                                                        .color(TEXT_OPTION_DESC),
                                                )
                                                .wrap(),
                                            );
                                            if let Some(warn) = &desc.warning {
                                                ui.add(
                                                    egui::Label::new(
                                                        egui::RichText::new(warn)
                                                            .font(theme.font(FONT_COMPACT))
                                                            .color(DANGER_TEXT),
                                                    )
                                                    .wrap(),
                                                );
                                            }
                                        });
                                    ui.add_space(3.0);
                                }
                            },
                        );
                    }
                });
        });
}

// ---------------------------------------------------------------------------
// Tab 8: Settings (Theme Presets, HSL Derivation, Cache & Undo Limits)
// ---------------------------------------------------------------------------

fn show_settings_tab(
    ui: &mut Ui,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &mut Cm3Theme,
    state: &mut PanelsState,
) {
    ScrollArea::vertical()
        .id_salt("settings_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let theme_snap = theme.clone();
            cm3_collapsible_section(
                ui,
                Id::unique("settings_theme"),
                "Theme & Accent Palette",
                true,
                &theme_snap,
                |ui| {
                    // Profile preset selector
                    let mut prof_idx = ThemeProfileChoice::ALL
                        .iter()
                        .position(|&p| p == theme.profile)
                        .unwrap_or(0);
                    let labels: Vec<&str> =
                        ThemeProfileChoice::ALL.iter().map(|p| p.label()).collect();
                    let changed = cm3_property_row(
                        ui,
                        "theme profile",
                        state.prop_columns,
                        0,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_dropdown(
                                ui,
                                Id::unique("settings_theme_profile"),
                                &mut prof_idx,
                                &labels,
                                false,
                                viewport_rect,
                                theme,
                            )
                        },
                    );
                    if changed {
                        theme.set_profile(ThemeProfileChoice::ALL[prof_idx]);
                    }

                    // Custom Main Color picker (Spec 01 §2.6 A)
                    let mut main_rgba = theme.custom_main_rgb.with_alpha(1.0);
                    let cev = cm3_property_row(
                        ui,
                        "main color",
                        state.prop_columns,
                        0,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_hsva_color_picker(
                                ui,
                                Id::unique("settings_main_color"),
                                &mut main_rgba,
                                false,
                                false,
                                viewport_rect,
                                theme,
                            )
                        },
                    );
                    if cev.changed || cev.committed {
                        theme.set_custom_main(main_rgba.rgb());
                    }

                    // Hue & Saturation shift controls (Spec 01 §2.6 B)
                    let mut hue = f64::from(theme.custom_hue_deg);
                    let hev = cm3_property_row(
                        ui,
                        "hue shift (deg)",
                        state.prop_columns,
                        0,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_scrub_number(
                                ui,
                                Id::unique("settings_hue"),
                                &mut hue,
                                ScrubConfig::new(-180.0, 180.0, 1.0, 0),
                                theme,
                            )
                        },
                    );
                    if hev.changed || hev.committed {
                        theme.set_custom_hue_sat(hue as f32, theme.custom_sat_percent);
                    }

                    let mut sat = f64::from(theme.custom_sat_percent);
                    let sev = cm3_property_row(
                        ui,
                        "saturation (%)",
                        state.prop_columns,
                        0,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_scrub_number(
                                ui,
                                Id::unique("settings_sat"),
                                &mut sat,
                                ScrubConfig::new(0.0, 200.0, 1.0, 0),
                                theme,
                            )
                        },
                    );
                    if sev.changed || sev.committed {
                        theme.set_custom_hue_sat(theme.custom_hue_deg, sat as f32);
                    }

                    // Live 5-color Classic / derived palette preview strip
                    Frame::NONE
                        .inner_margin(egui::Margin::symmetric(6, 6))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let entries = [
                                    ("focus", theme.accent()),
                                    ("link", theme.value()),
                                    ("clip", theme.clip()),
                                    ("sel", theme.clip_selected()),
                                    ("box", theme.clip_selection()),
                                ];
                                for (lbl, col) in entries {
                                    let (r, _) = ui
                                        .allocate_exact_size(Vec2::new(42.0, 24.0), Sense::hover());
                                    ui.painter().rect_filled(r, CornerRadius::ZERO, col);
                                    ui.painter().rect_stroke(
                                        r,
                                        CornerRadius::ZERO,
                                        Stroke::new(1.0, BORDER_CONTROL),
                                        StrokeKind::Inside,
                                    );
                                    ui.painter().text(
                                        r.center(),
                                        Align2::CENTER_CENTER,
                                        lbl,
                                        theme.font(FONT_COMPACT),
                                        TEXT_BRIGHT,
                                    );
                                }
                            });
                        });
                },
            );

            // Performance & Memory section
            cm3_collapsible_section(
                ui,
                Id::unique("settings_perf"),
                "Performance & History",
                true,
                theme,
                |ui| {
                    let mut cache = f64::from(state.app_settings.preview_cache_mib);
                    let ev = cm3_property_row(
                        ui,
                        "preview cache (MiB)",
                        state.prop_columns,
                        0,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_scrub_number(
                                ui,
                                Id::unique("settings_cache"),
                                &mut cache,
                                ScrubConfig::new(0.0, 16_384.0, 64.0, 0),
                                theme,
                            )
                        },
                    );
                    if ev.changed || ev.committed {
                        state.app_settings.preview_cache_mib =
                            cache.round().clamp(0.0, 16_384.0) as u32;
                    }

                    let mut max_entries = state.app_settings.undo_max_entries as f64;
                    let ev_e = cm3_property_row(
                        ui,
                        "undo max steps",
                        state.prop_columns,
                        0,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_scrub_number(
                                ui,
                                Id::unique("settings_undo_entries"),
                                &mut max_entries,
                                ScrubConfig::new(10.0, 2000.0, 10.0, 0),
                                theme,
                            )
                        },
                    );
                    let mut max_mib = state.app_settings.undo_max_mib as f64;
                    let ev_b = cm3_property_row(
                        ui,
                        "undo memory (MiB)",
                        state.prop_columns,
                        0,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_scrub_number(
                                ui,
                                Id::unique("settings_undo_mib"),
                                &mut max_mib,
                                ScrubConfig::new(4.0, 1024.0, 8.0, 0),
                                theme,
                            )
                        },
                    );
                    if ev_e.committed || ev_b.committed {
                        state.app_settings.undo_max_entries =
                            max_entries.round().clamp(10.0, 2000.0) as usize;
                        state.app_settings.undo_max_mib =
                            max_mib.round().clamp(4.0, 1024.0) as usize;
                        session.history.set_limits(
                            state.app_settings.undo_max_entries,
                            state.app_settings.undo_max_mib * 1024 * 1024,
                        );
                    }
                },
            );
        });
}

// ---------------------------------------------------------------------------
// Tab 9: About (about)
// ---------------------------------------------------------------------------

fn show_about_tab(ui: &mut Ui, session: &Session, theme: &Cm3Theme) {
    ScrollArea::vertical()
        .id_salt("about_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            Frame::NONE
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new("ZoidCraft")
                            .font(theme.font(FONT_TITLE))
                            .color(TEXT_BRIGHT),
                    );
                    ui.label(
                        egui::RichText::new(format!("Version {}", env!("CARGO_PKG_VERSION")))
                            .font(theme.font(FONT_SMALL))
                            .color(theme.value()),
                    );
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(
                            "Clean-room native Rust + wgpu motion-graphics & 3D video editor \
                             implementing ClipMaker 3 workflows and toggleable Zoidium extensions.",
                        )
                        .font(theme.font(FONT_SMALL))
                        .color(TEXT_PRIMARY),
                    );
                    ui.add_space(10.0);

                    let enabled_opt = session
                        .plugins
                        .enabled_ids()
                        .iter()
                        .filter(|id| id.as_str() != "core")
                        .count();
                    ui.label(
                        egui::RichText::new(format!(
                            "Active optional plugins: {enabled_opt}  |  Undo depth: {}/{}",
                            session.history.undo_len(),
                            session.history.max_entries()
                        ))
                        .font(theme.font(FONT_SMALL))
                        .color(TEXT_MUTED),
                    );
                    ui.add_space(10.0);

                    section_banner(ui, "Key Shortcuts", theme);
                    ui.add_space(4.0);
                    for (key, desc) in [
                        ("Space", "Play / Pause"),
                        ("Home / End", "Jump to start / end"),
                        ("Left / Right", "Step -1 / +1 frame (Shift: +-5)"),
                        ("J / K / L", "Shuttle reverse / stop / forward"),
                        ("C / T", "Editing Camera / 2D Transform"),
                        ("Ctrl+Z / Ctrl+Y", "Undo / Redo"),
                        ("Ctrl+G", "Open Graph Editor"),
                        ("Backquote (`)", "Maximize / restore focused panel"),
                    ] {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("{key:<15}"))
                                    .font(theme.font(FONT_SMALL))
                                    .color(theme.value()),
                            );
                            ui.label(
                                egui::RichText::new(desc)
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_PRIMARY),
                            );
                        });
                    }
                });
        });
}

// ---------------------------------------------------------------------------
// Inspector Floating Windows (Custom Shader GLSL Editor + Expression Editor)
// ---------------------------------------------------------------------------

fn show_inspector_floating_windows(
    ctx: &egui::Context,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
) {
    // 1. Custom Shader GLSL Editor (Spec 01 §4.6 Shader+ zoomable code editor)
    if let Some(mut editor) = state.shader_editor.take() {
        let spec = FloatingWindowSpec::new("shader_editor", "Shader Editor")
            .with_subtitle(editor.effect_name.clone())
            .with_default_size(Vec2::new(560.0, 440.0));
        let mut apply_code = false;
        let ev = cm3_floating_window(
            ctx,
            &mut state.floating_windows,
            &spec,
            viewport_rect,
            theme,
            |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("GLSL ES 1.00")
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_MUTED),
                    );
                    if cm3_prop_button(ui, "A-", ButtonVariant::Normal, false, theme).clicked() {
                        editor.font_px = (editor.font_px - 2.0).clamp(10.0, 24.0);
                    }
                    if cm3_prop_button(ui, "A+", ButtonVariant::Normal, false, theme).clicked() {
                        editor.font_px = (editor.font_px + 2.0).clamp(10.0, 24.0);
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if cm3_prop_button(
                            ui,
                            "Apply (Ctrl+Enter)",
                            ButtonVariant::Primary,
                            false,
                            theme,
                        )
                        .clicked()
                        {
                            apply_code = true;
                        }
                    });
                });

                if ui.input(|i| {
                    (i.modifiers.ctrl || i.modifiers.command)
                        && (i.key_pressed(Key::Enter) || i.key_pressed(Key::S))
                }) {
                    apply_code = true;
                }

                if let Some(err) = &editor.error_message {
                    warning_banner(ui, err, theme);
                }

                ui.add_space(4.0);
                ScrollArea::vertical()
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
            },
        );

        if apply_code {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: editor.effect_id.clone(),
                property: "fragShader".to_owned(),
                value: json!(editor.code),
            });
            editor.error_message = None;
        }
        if !ev.closed {
            state.shader_editor = Some(editor);
        }
    }

    // 2. Expression Editor Floating Window
    if let Some(mut expr) = state.expr_editor.take() {
        let spec = FloatingWindowSpec::new("expression_editor", "Expression Editor")
            .with_subtitle(expr.property.clone())
            .with_default_size(Vec2::new(420.0, 240.0));
        let mut commit_expr = false;
        let mut clear_expr = false;
        let ev = cm3_floating_window(
            ctx,
            &mut state.floating_windows,
            &spec,
            viewport_rect,
            theme,
            |ui| {
                ui.add(
                    TextEdit::multiline(&mut expr.code)
                        .font(theme.font(FONT_SMALL))
                        .text_color(theme.value())
                        .desired_width(ui.available_width())
                        .desired_rows(6)
                        .code_editor(),
                );
                if ui.input(|i| {
                    (i.modifiers.ctrl || i.modifiers.command) && i.key_pressed(Key::Enter)
                }) {
                    commit_expr = true;
                }
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if cm3_prop_button(ui, "Apply", ButtonVariant::Primary, false, theme).clicked()
                    {
                        commit_expr = true;
                    }
                    if cm3_prop_button(ui, "Clear Expression", ButtonVariant::Danger, false, theme)
                        .clicked()
                    {
                        clear_expr = true;
                    }
                });
            },
        );
        if commit_expr {
            let _ = session.execute(Command::SetExpression {
                target_id: expr.target_id.clone(),
                property: expr.property.clone(),
                expression: Some(expr.code.clone()),
            });
        } else if clear_expr {
            let _ = session.execute(Command::SetExpression {
                target_id: expr.target_id.clone(),
                property: expr.property.clone(),
                expression: None,
            });
        }
        if !ev.closed && !clear_expr {
            state.expr_editor = Some(expr);
        }
    }
}

// ---------------------------------------------------------------------------
// Shared Property Binding Helpers (Stopwatch + Keyframe Bar + Session Execute)
// ---------------------------------------------------------------------------

fn find_clip_clone(session: &Session, clip_id: &str) -> Option<SessionClip> {
    for track in session
        .document()
        .video_tracks
        .iter()
        .chain(session.document().audio_tracks.iter())
    {
        if let Some(clip) = track.clips.iter().find(|c| c.id == clip_id) {
            return Some(clip.clone());
        }
    }
    None
}

fn lookup_session_prop(
    session: &Session,
    target_id: &str,
    property: &str,
) -> Option<SessionProperty> {
    if target_id == "sequence" {
        return session
            .document()
            .sequence_properties
            .get(property)
            .cloned();
    }
    for track in session
        .document()
        .video_tracks
        .iter()
        .chain(session.document().audio_tracks.iter())
    {
        for clip in &track.clips {
            if clip.id == target_id {
                return clip.properties.get(property).cloned();
            }
            if let Some(obj) = clip.objects_3d.iter().find(|o| o.id == target_id) {
                return obj.properties.get(property).cloned();
            }
            if let Some(fx) = clip.effects.iter().find(|e| e.id == target_id) {
                return fx.properties.get(property).cloned();
            }
        }
    }
    None
}

fn eval_prop_value(prop: Option<&SessionProperty>, local_frame: f64) -> Option<&Value> {
    let p = prop?;
    if p.keyframes.is_empty() {
        return Some(&p.default_value);
    }
    if p.keyframes.len() == 1 || !p.animated {
        return Some(&p.keyframes[0].value);
    }
    // Nearest keyframe at or before local_frame
    let mut best = &p.keyframes[0];
    for kf in &p.keyframes {
        if kf.frame <= local_frame + 1e-6 {
            best = kf;
        } else {
            break;
        }
    }
    Some(&best.value)
}

fn eval_prop_f64(prop: Option<&SessionProperty>, local_frame: f64, fallback: f64) -> f64 {
    eval_prop_value(prop, local_frame)
        .and_then(Value::as_f64)
        .unwrap_or(fallback)
}

fn eval_prop_u32(prop: Option<&SessionProperty>, local_frame: f64, fallback: u32) -> u32 {
    eval_prop_value(prop, local_frame)
        .and_then(Value::as_u64)
        .map_or(fallback, |v| v as u32)
}

fn eval_prop_vec2(
    prop: Option<&SessionProperty>,
    local_frame: f64,
    fallback: [f64; 2],
) -> [f64; 2] {
    eval_prop_value(prop, local_frame)
        .and_then(Value::as_array)
        .filter(|a| a.len() >= 2)
        .map_or(fallback, |a| {
            [
                a[0].as_f64().unwrap_or(fallback[0]),
                a[1].as_f64().unwrap_or(fallback[1]),
            ]
        })
}

fn eval_prop_vec3(
    prop: Option<&SessionProperty>,
    local_frame: f64,
    fallback: [f64; 3],
) -> [f64; 3] {
    eval_prop_value(prop, local_frame)
        .and_then(Value::as_array)
        .filter(|a| a.len() >= 3)
        .map_or(fallback, |a| {
            [
                a[0].as_f64().unwrap_or(fallback[0]),
                a[1].as_f64().unwrap_or(fallback[1]),
                a[2].as_f64().unwrap_or(fallback[2]),
            ]
        })
}

fn eval_prop_str(prop: Option<&SessionProperty>, local_frame: f64, fallback: &str) -> String {
    eval_prop_value(prop, local_frame)
        .and_then(Value::as_str)
        .map_or_else(|| fallback.to_owned(), str::to_owned)
}

fn build_kf_bar_state(
    prop: Option<&SessionProperty>,
    local_frame: f64,
    interpolated: bool,
    easing_plus_enabled: bool,
    viewport_rect: Rect,
) -> KeyframeBarState {
    let Some(p) = prop else {
        return KeyframeBarState {
            has_prev: false,
            is_keyed: false,
            has_next: false,
            interpolated,
            tween: 1,
            easing_plus_enabled,
            viewport_rect,
        };
    };
    let has_prev = p.keyframes.iter().any(|k| k.frame < local_frame - 1e-4);
    let has_next = p.keyframes.iter().any(|k| k.frame > local_frame + 1e-4);
    let exact = p
        .keyframes
        .iter()
        .find(|k| (k.frame - local_frame).abs() < 1e-4);
    KeyframeBarState {
        has_prev,
        is_keyed: exact.is_some(),
        has_next,
        interpolated,
        tween: exact.map_or(1, |k| k.tween),
        easing_plus_enabled,
        viewport_rect,
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_kf_bar_event(
    session: &mut Session,
    target_id: &str,
    prop_name: &str,
    prop: Option<&SessionProperty>,
    local_frame: f64,
    clip_start: f64,
    current_val: Value,
    ev: KeyframeBarEvent,
) {
    if ev.prev_kf
        && let Some(p) = prop
        && let Some(prev) = p
            .keyframes
            .iter()
            .rev()
            .find(|k| k.frame < local_frame - 1e-4)
    {
        let _ = session.execute(Command::SetFrame {
            frame: prev.frame + clip_start,
        });
    }
    if ev.next_kf
        && let Some(p) = prop
        && let Some(next) = p.keyframes.iter().find(|k| k.frame > local_frame + 1e-4)
    {
        let _ = session.execute(Command::SetFrame {
            frame: next.frame + clip_start,
        });
    }
    if ev.toggle_kf {
        let is_keyed = prop.is_some_and(|p| {
            p.keyframes
                .iter()
                .any(|k| (k.frame - local_frame).abs() < 1e-4)
        });
        if is_keyed {
            let _ = session.execute(Command::DeleteKeyframe {
                target_id: target_id.to_owned(),
                property: prop_name.to_owned(),
                frame: local_frame,
            });
        } else {
            let _ = session.execute(Command::SetKeyframe {
                target_id: target_id.to_owned(),
                property: prop_name.to_owned(),
                frame: local_frame,
                value: current_val.clone(),
                tween: Some(1),
            });
        }
    }
    if ev.toggle_interp
        && let Some(p) = prop
        && let Some(kf) = p
            .keyframes
            .iter()
            .find(|k| (k.frame - local_frame).abs() < 1e-4)
    {
        let toggled = kf.tween ^ (1 << 8);
        let _ = session.execute(Command::SetKeyframe {
            target_id: target_id.to_owned(),
            property: prop_name.to_owned(),
            frame: local_frame,
            value: kf.value.clone(),
            tween: Some(toggled),
        });
    }
    if let Some(ease_idx) = ev.set_easing
        && let Some(p) = prop
        && let Some(kf) = p
            .keyframes
            .iter()
            .find(|k| (k.frame - local_frame).abs() < 1e-4)
    {
        let new_tween = (kf.tween & 0xFF00) | u16::from(ease_idx);
        let _ = session.execute(Command::SetKeyframe {
            target_id: target_id.to_owned(),
            property: prop_name.to_owned(),
            frame: local_frame,
            value: kf.value.clone(),
            tween: Some(new_tween),
        });
    }
}

fn apply_scrub_to_session(
    session: &mut Session,
    target_id: &str,
    prop_name: &str,
    animated: bool,
    local_frame: f64,
    value: Value,
    ev: ScrubEvent,
) {
    if ev.started {
        let _ = session.begin_transaction(format!("edit {prop_name}"));
    }
    if ev.changed || ev.committed {
        let cmd = if animated {
            Command::SetKeyframe {
                target_id: target_id.to_owned(),
                property: prop_name.to_owned(),
                frame: local_frame,
                value,
                tween: None,
            }
        } else {
            Command::SetPropertyValue {
                target_id: target_id.to_owned(),
                property: prop_name.to_owned(),
                value,
            }
        };
        let _ = session.execute(cmd);
    }
    if ev.committed && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    } else if ev.cancelled && session.history.is_in_transaction() {
        let _ = session.cancel_transaction();
    }
}

#[allow(clippy::too_many_arguments)]
fn show_session_number_prop(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    label: &str,
    default_val: f64,
    config: ScrubConfig,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    let prop = lookup_session_prop(session, target_id, prop_key);
    let animated = prop.as_ref().is_some_and(|p| p.animated);
    let has_expr = prop.as_ref().is_some_and(|p| p.expression.is_some());
    let mut val = eval_prop_f64(prop.as_ref(), local_frame, default_val);
    let easing_plus = session.plugins.is_enabled("easing-plus");

    let mut sw_ev = Default::default();
    let mut kf_ev = KeyframeBarEvent::default();
    let mut open_expr = false;

    let scrub_ev = cm3_property_row(
        ui,
        label,
        state.prop_columns,
        0,
        theme,
        |ui| {
            sw_ev = cm3_stopwatch_button(ui, prop_key, animated, has_expr, theme);
        },
        |ui| {
            if animated && !has_expr {
                let bar_state = build_kf_bar_state(
                    prop.as_ref(),
                    local_frame,
                    true,
                    easing_plus,
                    viewport_rect,
                );
                kf_ev = cm3_keyframe_bar(
                    ui,
                    Id::unique((target_id, prop_key, "kf")),
                    bar_state,
                    theme,
                );
            }
            let ev = cm3_scrub_number(
                ui,
                Id::unique((target_id, prop_key)),
                &mut val,
                config.with_read_only(config.read_only || has_expr),
                theme,
            );
            if has_expr {
                ui.horizontal(|ui| {
                    if cm3_icon_button(
                        ui,
                        Cm3Icon::Code,
                        17.0,
                        theme.value(),
                        "edit expression",
                        theme,
                    ) {
                        open_expr = true;
                    }
                    if let Some(expr) = prop.as_ref().and_then(|p| p.expression.as_deref()) {
                        ui.label(
                            egui::RichText::new(expr)
                                .font(theme.font(FONT_COMPACT))
                                .color(TEXT_MUTED),
                        );
                    }
                });
            }
            ev
        },
    );

    handle_stopwatch_event(
        session,
        target_id,
        prop_key,
        has_expr,
        local_frame,
        json!(val),
        sw_ev,
        state,
        viewport_rect,
    );
    if open_expr {
        open_expression_window(
            target_id,
            prop_key,
            prop.as_ref()
                .and_then(|p| p.expression.clone())
                .unwrap_or_else(|| format!("{val}")),
            state,
            viewport_rect,
        );
    }
    apply_kf_bar_event(
        session,
        target_id,
        prop_key,
        prop.as_ref(),
        local_frame,
        clip_start,
        json!(val),
        kf_ev,
    );
    apply_scrub_to_session(
        session,
        target_id,
        prop_key,
        animated,
        local_frame,
        json!(val),
        scrub_ev,
    );
}

#[allow(clippy::too_many_arguments)]
fn show_session_vec2_prop(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    label: &str,
    default_val: [f64; 2],
    config: ScrubConfig,
    link_ratio: bool,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    let prop = lookup_session_prop(session, target_id, prop_key);
    let animated = prop.as_ref().is_some_and(|p| p.animated);
    let has_expr = prop.as_ref().is_some_and(|p| p.expression.is_some());
    let mut val = eval_prop_vec2(prop.as_ref(), local_frame, default_val);
    let easing_plus = session.plugins.is_enabled("easing-plus");

    let mut sw_ev = Default::default();
    let mut kf_ev = KeyframeBarEvent::default();

    let scrub_ev = cm3_property_row(
        ui,
        label,
        state.prop_columns,
        0,
        theme,
        |ui| {
            sw_ev = cm3_stopwatch_button(ui, prop_key, animated, has_expr, theme);
        },
        |ui| {
            if animated && !has_expr {
                let bar_state = build_kf_bar_state(
                    prop.as_ref(),
                    local_frame,
                    true,
                    easing_plus,
                    viewport_rect,
                );
                kf_ev = cm3_keyframe_bar(
                    ui,
                    Id::unique((target_id, prop_key, "kf")),
                    bar_state,
                    theme,
                );
            }
            cm3_vector2(
                ui,
                Id::unique((target_id, prop_key)),
                &mut val,
                config.with_read_only(config.read_only || has_expr),
                link_ratio,
                theme,
            )
        },
    );

    handle_stopwatch_event(
        session,
        target_id,
        prop_key,
        has_expr,
        local_frame,
        json!(val),
        sw_ev,
        state,
        viewport_rect,
    );
    apply_kf_bar_event(
        session,
        target_id,
        prop_key,
        prop.as_ref(),
        local_frame,
        clip_start,
        json!(val),
        kf_ev,
    );
    apply_scrub_to_session(
        session,
        target_id,
        prop_key,
        animated,
        local_frame,
        json!(val),
        scrub_ev,
    );
}

#[allow(clippy::too_many_arguments)]
fn show_session_vec3_prop(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    label: &str,
    default_val: [f64; 3],
    config: ScrubConfig,
    link_ratio: bool,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    let prop = lookup_session_prop(session, target_id, prop_key);
    let animated = prop.as_ref().is_some_and(|p| p.animated);
    let has_expr = prop.as_ref().is_some_and(|p| p.expression.is_some());
    let mut val = eval_prop_vec3(prop.as_ref(), local_frame, default_val);
    let easing_plus = session.plugins.is_enabled("easing-plus");

    let mut sw_ev = Default::default();
    let mut kf_ev = KeyframeBarEvent::default();

    let scrub_ev = cm3_property_row(
        ui,
        label,
        state.prop_columns,
        0,
        theme,
        |ui| {
            sw_ev = cm3_stopwatch_button(ui, prop_key, animated, has_expr, theme);
        },
        |ui| {
            if animated && !has_expr {
                let bar_state = build_kf_bar_state(
                    prop.as_ref(),
                    local_frame,
                    true,
                    easing_plus,
                    viewport_rect,
                );
                kf_ev = cm3_keyframe_bar(
                    ui,
                    Id::unique((target_id, prop_key, "kf")),
                    bar_state,
                    theme,
                );
            }
            cm3_vector3(
                ui,
                Id::unique((target_id, prop_key)),
                &mut val,
                config.with_read_only(config.read_only || has_expr),
                link_ratio,
                theme,
            )
        },
    );

    handle_stopwatch_event(
        session,
        target_id,
        prop_key,
        has_expr,
        local_frame,
        json!(val),
        sw_ev,
        state,
        viewport_rect,
    );
    apply_kf_bar_event(
        session,
        target_id,
        prop_key,
        prop.as_ref(),
        local_frame,
        clip_start,
        json!(val),
        kf_ev,
    );
    apply_scrub_to_session(
        session,
        target_id,
        prop_key,
        animated,
        local_frame,
        json!(val),
        scrub_ev,
    );
}

#[allow(clippy::too_many_arguments)]
fn show_session_color_prop(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    label: &str,
    default_rgb: [f64; 3],
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let prop = lookup_session_prop(session, target_id, prop_key);
    let animated = prop.as_ref().is_some_and(|p| p.animated);
    let has_expr = prop.as_ref().is_some_and(|p| p.expression.is_some());
    let rgb_f64 = eval_prop_vec3(prop.as_ref(), local_frame, default_rgb);
    let mut rgba = Rgba::new(rgb_f64[0] as f32, rgb_f64[1] as f32, rgb_f64[2] as f32, 1.0);
    let easing_plus = session.plugins.is_enabled("easing-plus");

    let mut sw_ev = Default::default();
    let mut kf_ev = KeyframeBarEvent::default();

    let cev = cm3_property_row(
        ui,
        label,
        state.prop_columns,
        0,
        theme,
        |ui| {
            sw_ev = cm3_stopwatch_button(ui, prop_key, animated, has_expr, theme);
        },
        |ui| {
            if animated && !has_expr {
                let bar_state = build_kf_bar_state(
                    prop.as_ref(),
                    local_frame,
                    true,
                    easing_plus,
                    viewport_rect,
                );
                kf_ev = cm3_keyframe_bar(
                    ui,
                    Id::unique((target_id, prop_key, "kf")),
                    bar_state,
                    theme,
                );
            }
            cm3_hsva_color_picker(
                ui,
                Id::unique((target_id, prop_key)),
                &mut rgba,
                false,
                has_expr,
                viewport_rect,
                theme,
            )
        },
    );

    let out_val = json!([
        f64::from(rgba.0[0]),
        f64::from(rgba.0[1]),
        f64::from(rgba.0[2])
    ]);
    handle_stopwatch_event(
        session,
        target_id,
        prop_key,
        has_expr,
        local_frame,
        out_val.clone(),
        sw_ev,
        state,
        viewport_rect,
    );
    apply_kf_bar_event(
        session,
        target_id,
        prop_key,
        prop.as_ref(),
        local_frame,
        clip_start,
        out_val.clone(),
        kf_ev,
    );
    let scrub_equiv = ScrubEvent {
        started: cev.started,
        changed: cev.changed,
        committed: cev.committed,
        cancelled: cev.cancelled,
    };
    apply_scrub_to_session(
        session,
        target_id,
        prop_key,
        animated,
        local_frame,
        out_val,
        scrub_equiv,
    );
    if cev.eyedropper_clicked {
        actions.push(PanelAction::EyedropperActivate {
            target_id: target_id.to_owned(),
            property: prop_key.to_owned(),
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn show_session_option_prop(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    label: &str,
    options: &[&str],
    default_idx: u32,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    let prop = lookup_session_prop(session, target_id, prop_key);
    let animated = prop.as_ref().is_some_and(|p| p.animated);
    let has_expr = prop.as_ref().is_some_and(|p| p.expression.is_some());
    let mut idx = eval_prop_u32(prop.as_ref(), local_frame, default_idx) as usize;
    let mut sw_ev = Default::default();
    let mut kf_ev = KeyframeBarEvent::default();

    let changed = cm3_property_row(
        ui,
        label,
        state.prop_columns,
        0,
        theme,
        |ui| {
            sw_ev = cm3_stopwatch_button(ui, prop_key, animated, has_expr, theme);
        },
        |ui| {
            if animated && !has_expr {
                let bar_state =
                    build_kf_bar_state(prop.as_ref(), local_frame, false, false, viewport_rect);
                kf_ev = cm3_keyframe_bar(
                    ui,
                    Id::unique((target_id, prop_key, "kf")),
                    bar_state,
                    theme,
                );
            }
            cm3_dropdown(
                ui,
                Id::unique((target_id, prop_key)),
                &mut idx,
                options,
                has_expr,
                viewport_rect,
                theme,
            )
        },
    );

    handle_stopwatch_event(
        session,
        target_id,
        prop_key,
        has_expr,
        local_frame,
        json!(idx),
        sw_ev,
        state,
        viewport_rect,
    );
    apply_kf_bar_event(
        session,
        target_id,
        prop_key,
        prop.as_ref(),
        local_frame,
        clip_start,
        json!(idx),
        kf_ev,
    );
    if changed {
        let cmd = if animated {
            Command::SetKeyframe {
                target_id: target_id.to_owned(),
                property: prop_key.to_owned(),
                frame: local_frame,
                value: json!(idx),
                tween: Some(0),
            }
        } else {
            Command::SetPropertyValue {
                target_id: target_id.to_owned(),
                property: prop_key.to_owned(),
                value: json!(idx),
            }
        };
        let _ = session.execute(cmd);
    }
}

#[allow(clippy::too_many_arguments)]
fn show_session_text_prop(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    label: &str,
    default_text: &str,
    local_frame: f64,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    let prop = lookup_session_prop(session, target_id, prop_key);
    let current = eval_prop_str(prop.as_ref(), local_frame, default_text);
    let buf_id = Id::unique((target_id, prop_key, "text_buf"));
    let mut buf = ui
        .data_mut(|d| d.get_temp::<String>(buf_id))
        .unwrap_or_else(|| current.clone());
    let edit_id = Id::unique((target_id, prop_key, "text_edit"));
    if !ui.memory(|m| m.has_focus(edit_id)) {
        buf = current.clone();
    }

    let mut commit = false;
    cm3_property_row(
        ui,
        label,
        state.prop_columns,
        0,
        theme,
        |_| {},
        |ui| {
            let focused = ui.memory(|m| m.has_focus(edit_id));
            let resp = Frame::NONE
                .fill(BG_INPUT)
                .stroke(Stroke::new(
                    1.0,
                    if focused {
                        theme.accent()
                    } else {
                        BORDER_CONTROL
                    },
                ))
                .inner_margin(egui::Margin::symmetric(4, 2))
                .show(ui, |ui| {
                    ui.add(
                        TextEdit::singleline(&mut buf)
                            .id(edit_id)
                            .font(theme.font(FONT_BODY))
                            .text_color(theme.value())
                            .desired_width(ui.available_width().min(200.0)),
                    )
                })
                .inner;
            if resp.lost_focus() || (resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter))) {
                commit = true;
                ui.memory_mut(|m| m.surrender_focus(edit_id));
            }
        },
    );

    if commit && buf != current {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: target_id.to_owned(),
            property: prop_key.to_owned(),
            value: json!(buf),
        });
    }
    ui.data_mut(|d| d.insert_temp(buf_id, buf));
}

#[allow(clippy::too_many_arguments)]
fn handle_stopwatch_event(
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    has_expr: bool,
    local_frame: f64,
    current_val: Value,
    ev: super::controls::StopwatchEvent,
    state: &mut PanelsState,
    viewport_rect: Rect,
) {
    if ev.toggled {
        let _ = session.execute(Command::TogglePropertyAnimated {
            target_id: target_id.to_owned(),
            property: prop_key.to_owned(),
            frame: local_frame,
        });
    } else if ev.alt_expression {
        if has_expr {
            let _ = session.execute(Command::SetExpression {
                target_id: target_id.to_owned(),
                property: prop_key.to_owned(),
                expression: None,
            });
        } else {
            let init_expr = current_val.to_string();
            let _ = session.execute(Command::SetExpression {
                target_id: target_id.to_owned(),
                property: prop_key.to_owned(),
                expression: Some(init_expr.clone()),
            });
            open_expression_window(target_id, prop_key, init_expr, state, viewport_rect);
        }
    }
}

fn open_expression_window(
    target_id: &str,
    prop_key: &str,
    code: String,
    state: &mut PanelsState,
    viewport_rect: Rect,
) {
    state.expr_editor = Some(ExprEditorTarget {
        target_id: target_id.to_owned(),
        property: prop_key.to_owned(),
        code,
    });
    let spec = FloatingWindowSpec::new("expression_editor", "Expression Editor")
        .with_subtitle(prop_key.to_owned())
        .with_default_size(Vec2::new(420.0, 240.0));
    state.floating_windows.open_or_focus(&spec, viewport_rect);
}

// ---------------------------------------------------------------------------
// Small UI Helpers
// ---------------------------------------------------------------------------

fn property_header_with_reset(
    ui: &mut Ui,
    title: &str,
    target_id: &str,
    session: &mut Session,
    theme: &Cm3Theme,
) {
    let (hdr_rect, _) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 30.0), Sense::hover());
    ui.painter()
        .rect_filled(hdr_rect, CornerRadius::ZERO, BG_TITLE);
    ui.painter().line_segment(
        [hdr_rect.left_bottom(), hdr_rect.right_bottom()],
        Stroke::new(1.0, BORDER_ROW_SOFT),
    );
    ui.painter().text(
        Pos2::new(hdr_rect.left() + 8.0, hdr_rect.center().y),
        Align2::LEFT_CENTER,
        title,
        theme.font(FONT_TITLE),
        TEXT_STRONG,
    );

    let reset_rect = Rect::from_min_size(
        Pos2::new(hdr_rect.right() - 26.0, hdr_rect.top() + 5.0),
        Vec2::splat(20.0),
    );
    let reset_resp = ui
        .interact(
            reset_rect,
            Id::unique(("reset_list", target_id)),
            Sense::click(),
        )
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text("reset");
    if reset_resp.hovered() {
        ui.painter().rect_stroke(
            reset_rect,
            CornerRadius::ZERO,
            Stroke::new(2.0, theme.accent()),
            StrokeKind::Inside,
        );
    }
    paint_cm3_icon(
        ui.painter(),
        reset_rect.shrink(1.5),
        Cm3Icon::Reset,
        if reset_resp.hovered() {
            TEXT_BRIGHT
        } else {
            TEXT_PRIMARY
        },
    );
    if reset_resp.clicked() {
        let _ = session.execute(Command::ResetPropertyList {
            target_id: target_id.to_owned(),
        });
    }
}

fn section_banner(ui: &mut Ui, title: &str, theme: &Cm3Theme) {
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

fn warning_banner(ui: &mut Ui, message: &str, theme: &Cm3Theme) {
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

fn empty_panel_message(ui: &mut Ui, message: &str, theme: &Cm3Theme) {
    let avail = ui.available_rect_before_wrap();
    ui.painter().text(
        avail.center(),
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
