//! Complete CM3 & Zoidium left-rail inspector panels (Media, Sequence, Edit, Objects,
//! Effects, Export, Restore, Plugins, Assets, About, Settings) wired directly to
//! `zc_session::Session`.

use std::collections::{BTreeMap, BTreeSet};

use egui::{
    Align2, Area, Color32, CornerRadius, CursorIcon, Frame, Id, Key, Mesh, Order, Painter, Pos2,
    Rect, ScrollArea, Sense, Shape, Stroke, StrokeKind, TextEdit, Ui, Vec2,
};
use serde_json::{Value, json};
use zc_anim::{BezierSegment, InterpolationMode, eval_easing, eval_segment_bezier};
use zc_color::{BlendMode, Hsv, Rgb, Rgba};
use zc_plugin::PluginCategory;
use zc_project::{
    AssetKind, EffectCategory, GradientStop, Marker, PropertyValue, create_effect_by_id,
    standard_effect_catalog, zoidium_effect_catalog,
};
use zc_session::{Command, Session, SessionClip, SessionObject3D, SessionProperty};
use zc_shader::translate_glsl_es100;

use super::controls::{
    ButtonVariant, FloatingWindowManager, FloatingWindowSpec, FuzzyItem, FuzzyPickerEvent,
    FuzzyPickerState, KeyframeBarEvent, KeyframeBarState, PropertyColumnState, ScrubConfig,
    ScrubEvent, cm3_collapsible_section, cm3_collapsible_section_with_action,
    cm3_column_adjust_splitter, cm3_dropdown, cm3_floating_window, cm3_fuzzy_picker,
    cm3_gradient_bar, cm3_hsva_color_picker, cm3_icon_button, cm3_keyframe_bar, cm3_prop_button,
    cm3_property_row, cm3_property_row_aligned, cm3_scrub_number, cm3_stopwatch_button,
    cm3_vector2, cm3_vector2_with_sep, cm3_vector3, paint_cm3_tree_box,
};
use super::layout::{SplitOrientation, SplitPanelState, begin_split_panel, show_split_pane};
use super::theme::{
    BG_BUTTON, BG_DROPDOWN_HOVER, BG_HOVER, BG_INPUT, BG_PANEL, BG_SUNKEN, BG_TITLE,
    BORDER_CONTROL, BORDER_FILTER, BORDER_PANEL, BORDER_ROW_SOFT, Cm3Icon, Cm3Theme, DANGER_TEXT,
    EditorFontChoice, FONT_BADGE, FONT_BANNER, FONT_BODY, FONT_COMPACT, FONT_EASING, FONT_SMALL,
    FONT_TITLE, TEXT_BRIGHT, TEXT_DISABLED, TEXT_EMPTY, TEXT_FAINT, TEXT_MUTED, TEXT_OPTION_DESC,
    TEXT_PRIMARY, TEXT_STRONG, TOOLBAR_ICON_INACTIVE, ThemeProfileChoice, VOID_BLACK, WARNING_BAR,
    WARNING_SURFACE, color32_with_alpha, paint_badge, paint_cm3_icon, rgb_to_color32,
};

// ---------------------------------------------------------------------------
// Left-Rail Tab Enum (Exact Zoidium 11-Tab Order, Settings After About)
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
    Restore,
    Plugins,
    Assets,
    About,
    Settings,
}

impl SideTab {
    /// Exact Zoidium 11-tab elevator rail order (`Settings` after `About`).
    pub const ALL: [Self; 11] = [
        Self::Media,
        Self::Sequence,
        Self::Edit,
        Self::Objects,
        Self::Effects,
        Self::Export,
        Self::Restore,
        Self::Plugins,
        Self::Assets,
        Self::About,
        Self::Settings,
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
            Self::Restore => "Restore",
            Self::Plugins => "Plugins",
            Self::Assets => "Assets",
            Self::About => "About",
            Self::Settings => "Settings",
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
            Self::Restore => Cm3Icon::TabRestore,
            Self::Plugins => Cm3Icon::Plugins,
            Self::Assets => Cm3Icon::TabAssets,
            Self::About => Cm3Icon::About,
            Self::Settings => Cm3Icon::Settings,
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
    pub const LABELS: [&'static str; 3] = ["Quality (CRF)", "Bitrate (Mbps)", "Quality Preset"];
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
    NewProject,
    StartExport(ExportSettings),
    CancelExport,
    EyedropperActivate {
        target_id: String,
        property: String,
    },
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
    property: String,
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
    error_message: Option<String>,
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
    pub export_view_mode: Option<ExportMode>,
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
    plugin_filter: String,
    assets_filter: String,
    shader_prop_kinds: BTreeMap<(String, String), usize>,
    settings_editor_layout: usize,
    settings_gradient_editor: usize,
    settings_default_fps_buf: String,
    settings_edit_colors: bool,
    seq_control_counter: usize,
    objects_list_open: bool,
    effects_list_open: bool,
    #[allow(dead_code)]
    objects_focus_bottom: bool,
    #[allow(dead_code)]
    effects_focus_bottom: bool,
    expanded_vec_props: BTreeSet<(String, String)>,
    inspector_root_indent: bool,
    inspector_tx_owner: Option<(String, String)>,
    inspector_tx_label: Option<String>,
    inspector_tx_has_edits: bool,
    inspector_tx_owner_seen: bool,
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
            export_view_mode: None,
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
            plugin_filter: String::new(),
            assets_filter: String::new(),
            shader_prop_kinds: BTreeMap::new(),
            settings_editor_layout: 0,
            settings_gradient_editor: 0,
            settings_default_fps_buf: String::from("30"),
            settings_edit_colors: false,
            seq_control_counter: 1,
            objects_list_open: true,
            effects_list_open: true,
            objects_focus_bottom: false,
            effects_focus_bottom: false,
            expanded_vec_props: BTreeSet::new(),
            inspector_root_indent: false,
            inspector_tx_owner: None,
            inspector_tx_label: None,
            inspector_tx_has_edits: false,
            inspector_tx_owner_seen: false,
        }
    }
}

fn clear_inspector_tx_state(state: &mut PanelsState) {
    state.inspector_tx_owner = None;
    state.inspector_tx_label = None;
    state.inspector_tx_has_edits = false;
}

fn commit_inspector_tx_if_open(session: &mut Session, state: &mut PanelsState) {
    let had_owner = state.inspector_tx_owner.is_some();
    let had_edits = state.inspector_tx_has_edits;
    clear_inspector_tx_state(state);
    if had_owner && session.history.is_in_transaction() {
        if had_edits {
            let _ = session.commit_transaction(true);
        } else {
            let _ = session.cancel_transaction();
        }
    }
}

fn close_existing_tx_before_start(session: &mut Session, state: &mut PanelsState) {
    if session.history.is_in_transaction() {
        if state.inspector_tx_owner.is_some() && !state.inspector_tx_has_edits {
            let _ = session.cancel_transaction();
        } else {
            let _ = session.commit_transaction(true);
        }
        clear_inspector_tx_state(state);
    } else if state.inspector_tx_owner.is_some() {
        clear_inspector_tx_state(state);
    }
}

fn begin_inspector_tx(
    session: &mut Session,
    state: &mut PanelsState,
    target_id: &str,
    owner_key: &str,
    label: impl Into<String>,
) -> bool {
    let label = label.into();
    if session.begin_transaction(label.clone()).is_ok() {
        state.inspector_tx_owner = Some((target_id.to_owned(), owner_key.to_owned()));
        state.inspector_tx_label = Some(label);
        state.inspector_tx_has_edits = false;
        state.inspector_tx_owner_seen = true;
        true
    } else {
        false
    }
}

fn run_atomic_commit_outside_foreign_tx(
    session: &mut Session,
    state: &mut PanelsState,
    commit_fn: impl FnOnce(&mut Session),
) {
    if session.history.is_in_transaction() && state.inspector_tx_owner.is_some() {
        let saved_owner = state.inspector_tx_owner.take();
        let saved_label = state.inspector_tx_label.take();
        let had_edits = state.inspector_tx_has_edits;
        let was_seen = state.inspector_tx_owner_seen;
        state.inspector_tx_has_edits = false;

        if had_edits {
            let _ = session.commit_transaction(true);
        } else {
            let _ = session.cancel_transaction();
        }

        commit_fn(session);

        if !had_edits && let Some(owner) = saved_owner {
            let label = saved_label.unwrap_or_else(|| format!("edit {}", owner.1));
            if session.begin_transaction(label.clone()).is_ok() {
                state.inspector_tx_owner = Some(owner);
                state.inspector_tx_label = Some(label);
                state.inspector_tx_has_edits = false;
                state.inspector_tx_owner_seen = was_seen;
            }
        }
    } else {
        commit_fn(session);
    }
}

fn owns_inspector_tx(state: &PanelsState, target_id: &str, prop_key: &str) -> bool {
    state
        .inspector_tx_owner
        .as_ref()
        .is_some_and(|(t, p)| t == target_id && p == prop_key)
}

fn touch_inspector_tx_owner(state: &mut PanelsState, target_id: &str, prop_key: &str) {
    if owns_inspector_tx(state, target_id, prop_key) {
        state.inspector_tx_owner_seen = true;
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
    state.inspector_tx_owner_seen = false;
    state.inspector_root_indent = false;
    if !session.history.is_in_transaction() {
        clear_inspector_tx_state(state);
    }

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
        SideTab::Restore => show_restore_tab(ui, session, theme, actions),
        SideTab::Plugins => show_plugins_tab(ui, session, theme, state),
        SideTab::Assets => show_assets_tab(ui, session, theme, state, actions),
        SideTab::About => show_about_tab(ui, session, theme),
        SideTab::Settings => show_settings_tab(ui, viewport_rect, session, theme, state),
    }

    // If an inspector widget owning an open undo transaction was unmounted mid-edit
    // (e.g. collapsed section, tab switch, selection change, or read-only flip), close cleanly.
    if state.inspector_tx_owner.is_some() && !state.inspector_tx_owner_seen {
        commit_inspector_tx_if_open(session, state);
    }

    // Floating editors spawned from inspectors (Custom Shader GLSL editor, Expression editor)
    show_inspector_floating_windows(ui.ctx(), viewport_rect, session, theme, state);
}

// ---------------------------------------------------------------------------
// Tab 1: Media (project)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MediaCardIcon {
    Scene3D,
    Adjustment,
    Text,
    Shape,
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
        icon: MediaCardIcon::Shape,
        required_plugin: None,
    },
    MediaPreset {
        kind: "shape",
        title: "Shape",
        subtitle: "Custom vector shape",
        icon: MediaCardIcon::Shape,
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
        icon: MediaCardIcon::Shape,
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
        MediaCardIcon::Shape => {
            // Triangle overlapping a circle matching Zoidium Preset shape / Shape / Shape+ (45x38)
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

fn clip_kind_for_asset(kind: AssetKind, filename: &str) -> &'static str {
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

            // 100% edge-to-edge 30px #222222 "Project media" bar with borderless "+ Import" inside
            let (bar_rect, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 30.0), Sense::hover());
            ui.painter()
                .rect_filled(bar_rect, CornerRadius::ZERO, BG_TITLE);
            ui.painter().text(
                Pos2::new(bar_rect.left() + 5.0, bar_rect.center().y + 1.0),
                Align2::LEFT_CENTER,
                "Project media",
                theme.font(FONT_BODY),
                TEXT_PRIMARY,
            );
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

                                let icon_rect = Rect::from_center_size(
                                    Pos2::new(rect.center().x, rect.top() + 45.5),
                                    Vec2::splat(48.0),
                                );
                                let label_clip = Rect::from_min_max(
                                    Pos2::new(rect.left() + 5.0, rect.bottom() - 28.0),
                                    Pos2::new(rect.right() - 5.0, rect.bottom() - 2.0),
                                );

                                match item {
                                    MediaGridItem::Preset(preset) => {
                                        let resp = resp
                                            .on_hover_cursor(CursorIcon::PointingHand)
                                            .on_hover_text(preset.subtitle);
                                        paint_media_card_icon(
                                            ui.painter(),
                                            icon_rect,
                                            preset.icon,
                                            TEXT_PRIMARY,
                                            bg,
                                        );
                                        ui.painter().with_clip_rect(label_clip).text(
                                            Pos2::new(rect.left() + 5.0, rect.bottom() - 4.0),
                                            Align2::LEFT_BOTTOM,
                                            preset.title,
                                            theme.font(FONT_BODY),
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
                                                "Click to add {} clip to timeline",
                                                entry.clip_kind
                                            ));
                                        let icon = match entry.clip_kind {
                                            "video" => Cm3Icon::Sequence,
                                            "audio" => Cm3Icon::Edit2,
                                            _ => Cm3Icon::Project,
                                        };
                                        paint_cm3_icon(
                                            ui.painter(),
                                            icon_rect,
                                            icon,
                                            if entry.missing {
                                                DANGER_TEXT
                                            } else {
                                                TEXT_PRIMARY
                                            },
                                        );
                                        ui.painter().with_clip_rect(label_clip).text(
                                            Pos2::new(rect.left() + 5.0, rect.bottom() - 4.0),
                                            Align2::LEFT_BOTTOM,
                                            &entry.name,
                                            theme.font(FONT_BODY),
                                            if entry.missing {
                                                DANGER_TEXT
                                            } else {
                                                TEXT_PRIMARY
                                            },
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

fn find_non_overlapping_start(clips: &[SessionClip], desired: f64, length: f64) -> f64 {
    let mut start = desired.max(0.0).round();
    loop {
        if let Some(blocker) = clips
            .iter()
            .find(|c| c.start < start + length - 1e-6 && start < c.start + c.length - 1e-6)
        {
            start = (blocker.start + blocker.length).ceil();
        } else {
            return start;
        }
    }
}

fn add_preset_clip(session: &mut Session, kind: &str, title: &str) {
    add_clip_with_asset(session, kind, title, None);
}

fn add_clip_with_asset(session: &mut Session, kind: &str, title: &str, asset_id: Option<&str>) {
    let is_audio = kind == "audio";
    let desired_start = session.playback.current_frame.max(0.0).round();
    let length = 150.0;

    // Pick a matching unlocked track with no overlap at playhead, or place at earliest
    // non-overlapping start on the first unlocked track (never blindly overwriting Track 0).
    let chosen: Option<(String, f64)> = {
        let tracks = if is_audio {
            &session.document().audio_tracks
        } else {
            &session.document().video_tracks
        };
        if let Some(free_track) = tracks.iter().find(|t| {
            !t.locked
                && !t.clips.iter().any(|c| {
                    c.start < desired_start + length - 1e-6
                        && desired_start < c.start + c.length - 1e-6
                })
        }) {
            Some((free_track.id.clone(), desired_start))
        } else {
            tracks.iter().find(|t| !t.locked).map(|t| {
                let start = find_non_overlapping_start(&t.clips, desired_start, length);
                (t.id.clone(), start)
            })
        }
    };

    let started_tx = session.begin_transaction(format!("add {title}")).is_ok();

    let (track_id, start) = if let Some(pair) = chosen {
        pair
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
            ui.spacing_mut().item_spacing.y = 0.0;

            let (reset_seq, _) = cm3_collapsible_section_with_action(
                ui,
                Id::unique("seq_general"),
                "Sequence",
                true,
                false,
                state.prop_columns.label_ratio,
                Some((Cm3Icon::Reset, "Reset sequence")),
                theme,
                |ui| {
                    // Resolution: 1920 x 1080 (0 decimals, "x" separator)
                    touch_inspector_tx_owner(state, "sequence", "resolution");
                    let [rw, rh] = session.document().resolution;
                    let mut res_vec = [f64::from(rw), f64::from(rh)];
                    let ev = cm3_property_row_aligned(
                        ui,
                        "Resolution",
                        true,
                        state.prop_columns,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_vector2_with_sep(
                                ui,
                                Id::unique("seq_resolution"),
                                &mut res_vec,
                                &ScrubConfig::new(16.0, 7680.0, 10.0, 0),
                                false,
                                "x",
                                theme,
                            )
                        },
                    );
                    let owns_res_tx = owns_inspector_tx(state, "sequence", "resolution");
                    let res_cmd = || Command::SetResolution {
                        width: res_vec[0].round().clamp(16.0, 7680.0) as u32,
                        height: res_vec[1].round().clamp(16.0, 7680.0) as u32,
                    };
                    if ev.cancelled {
                        if owns_res_tx {
                            if session.history.is_in_transaction() {
                                let _ = session.cancel_transaction();
                            }
                            clear_inspector_tx_state(state);
                        }
                    } else if ev.started && ev.committed {
                        if owns_res_tx {
                            if ev.changed {
                                let _ = session.execute(res_cmd());
                                state.inspector_tx_has_edits = true;
                            }
                            if session.history.is_in_transaction() {
                                let _ = session.commit_transaction(true);
                            }
                            clear_inspector_tx_state(state);
                        } else {
                            close_existing_tx_before_start(session, state);
                            if ev.changed {
                                let _ = session.execute(res_cmd());
                            }
                        }
                        begin_inspector_tx(
                            session,
                            state,
                            "sequence",
                            "resolution",
                            "set resolution",
                        );
                    } else {
                        if ev.started && !ev.committed {
                            close_existing_tx_before_start(session, state);
                            begin_inspector_tx(
                                session,
                                state,
                                "sequence",
                                "resolution",
                                "set resolution",
                            );
                        }
                        let owns_res_tx = owns_inspector_tx(state, "sequence", "resolution");
                        if ev.changed && ev.committed && !ev.started && !owns_res_tx {
                            run_atomic_commit_outside_foreign_tx(session, state, |session| {
                                let _ = session.execute(res_cmd());
                            });
                        } else {
                            if ev.changed || (ev.committed && owns_res_tx) {
                                let _ = session.execute(res_cmd());
                                if owns_res_tx {
                                    state.inspector_tx_has_edits = true;
                                }
                            }
                            if ev.committed && owns_res_tx {
                                if session.history.is_in_transaction() {
                                    let _ = session.commit_transaction(true);
                                }
                                clear_inspector_tx_state(state);
                            }
                        }
                    }

                    // Frame rate: integer 30 (0 decimals)
                    touch_inspector_tx_owner(state, "sequence", "fps");
                    let mut fps = session.document().fps;
                    let ev = cm3_property_row_aligned(
                        ui,
                        "Frame rate",
                        true,
                        state.prop_columns,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_scrub_number(
                                ui,
                                Id::unique("seq_fps"),
                                &mut fps,
                                ScrubConfig::new(1.0, 240.0, 1.0, 0),
                                theme,
                            )
                        },
                    );
                    let owns_fps_tx = owns_inspector_tx(state, "sequence", "fps");
                    if ev.cancelled {
                        if owns_fps_tx {
                            if session.history.is_in_transaction() {
                                let _ = session.cancel_transaction();
                            }
                            clear_inspector_tx_state(state);
                        }
                    } else if ev.started && ev.committed {
                        if owns_fps_tx {
                            if ev.changed {
                                let _ = session.execute(Command::SetFps { fps });
                                state.inspector_tx_has_edits = true;
                            }
                            if session.history.is_in_transaction() {
                                let _ = session.commit_transaction(true);
                            }
                            clear_inspector_tx_state(state);
                        } else {
                            close_existing_tx_before_start(session, state);
                            if ev.changed {
                                let _ = session.execute(Command::SetFps { fps });
                            }
                        }
                        begin_inspector_tx(session, state, "sequence", "fps", "set fps");
                    } else {
                        if ev.started && !ev.committed {
                            close_existing_tx_before_start(session, state);
                            begin_inspector_tx(session, state, "sequence", "fps", "set fps");
                        }
                        let owns_fps_tx = owns_inspector_tx(state, "sequence", "fps");
                        if ev.changed && ev.committed && !ev.started && !owns_fps_tx {
                            run_atomic_commit_outside_foreign_tx(session, state, |session| {
                                let _ = session.execute(Command::SetFps { fps });
                            });
                        } else {
                            if ev.changed || (ev.committed && owns_fps_tx) {
                                let _ = session.execute(Command::SetFps { fps });
                                if owns_fps_tx {
                                    state.inspector_tx_has_edits = true;
                                }
                            }
                            if ev.committed && owns_fps_tx {
                                if session.history.is_in_transaction() {
                                    let _ = session.commit_transaction(true);
                                }
                                clear_inspector_tx_state(state);
                            }
                        }
                    }

                    // Markers: < ◇ > navigator row (no divider directly below; 33px gap + divider before Motion blur)
                    let mut prev_marker_frame: Option<f64> = None;
                    let mut next_marker_frame: Option<f64> = None;
                    let mut exact_marker_idx: Option<usize> = None;
                    for (idx, m) in session.project.sequence.markers.iter().enumerate() {
                        if (m.frame - frame).abs() < 0.5 {
                            exact_marker_idx = Some(idx);
                        } else if m.frame < frame - 0.5 {
                            prev_marker_frame = Some(match prev_marker_frame {
                                Some(best) => best.max(m.frame),
                                None => m.frame,
                            });
                        } else if m.frame > frame + 0.5 {
                            next_marker_frame = Some(match next_marker_frame {
                                Some(best) => best.min(m.frame),
                                None => m.frame,
                            });
                        }
                    }

                    let total_w = ui.available_width().max(120.0);
                    let (markers_rect, _) =
                        ui.allocate_exact_size(Vec2::new(total_w, 27.0), Sense::hover());
                    let split_x = (markers_rect.left()
                        + (total_w * state.prop_columns.label_ratio)
                            .clamp(72.0, (total_w - 60.0).max(72.0)))
                    .round();
                    let label_clip = Rect::from_min_max(
                        Pos2::new(markers_rect.left() + 36.0, markers_rect.top()),
                        Pos2::new(
                            (split_x - 5.0).max(markers_rect.left() + 36.0),
                            markers_rect.bottom(),
                        ),
                    );
                    ui.painter().with_clip_rect(label_clip).text(
                        Pos2::new(markers_rect.left() + 36.0, markers_rect.center().y),
                        Align2::LEFT_CENTER,
                        "Markers",
                        theme.font(FONT_BODY),
                        TEXT_PRIMARY,
                    );
                    let mut go_prev = false;
                    let mut toggle_marker = false;
                    let mut go_next = false;
                    let col2_rect = Rect::from_min_max(
                        Pos2::new(split_x, markers_rect.top() + 3.0),
                        Pos2::new(markers_rect.right(), markers_rect.bottom() + 3.0),
                    );
                    let mut col2_ui = ui.new_child(
                        egui::UiBuilder::new()
                            .max_rect(col2_rect)
                            .layout(egui::Layout::left_to_right(egui::Align::Center)),
                    );
                    col2_ui.set_clip_rect(col2_ui.clip_rect().intersect(col2_rect));
                    col2_ui.spacing_mut().item_spacing.x = 7.0;
                    if cm3_icon_button(
                        &mut col2_ui,
                        Cm3Icon::PrevKf,
                        17.0,
                        TEXT_PRIMARY,
                        "Previous marker",
                        theme,
                    ) {
                        go_prev = true;
                    }
                    let (icon, tip) = if exact_marker_idx.is_some() {
                        (Cm3Icon::KeyframeSolid, "Remove marker at current frame")
                    } else {
                        (Cm3Icon::KeyframeHollow, "Add marker at current frame")
                    };
                    if cm3_icon_button(&mut col2_ui, icon, 17.0, TEXT_PRIMARY, tip, theme) {
                        toggle_marker = true;
                    }
                    if cm3_icon_button(
                        &mut col2_ui,
                        Cm3Icon::NextKf,
                        17.0,
                        TEXT_PRIMARY,
                        "Next marker",
                        theme,
                    ) {
                        go_next = true;
                    }
                    if go_prev && let Some(target_f) = prev_marker_frame {
                        let _ = session.execute(Command::SetFrame { frame: target_f });
                    }
                    if go_next && let Some(target_f) = next_marker_frame {
                        let _ = session.execute(Command::SetFrame { frame: target_f });
                    }
                    if toggle_marker {
                        let label = if exact_marker_idx.is_some() {
                            "Remove marker"
                        } else {
                            "Add marker"
                        };
                        let started_tx = if !session.history.is_in_transaction() {
                            session.begin_transaction(label).is_ok()
                        } else {
                            false
                        };
                        if let Some(idx) = exact_marker_idx {
                            session.project.sequence.markers.remove(idx);
                        } else {
                            let f = frame.round().max(0.0);
                            session.project.sequence.markers.push(Marker {
                                frame: f,
                                label: format!("Marker {}", f as u64),
                            });
                            session
                                .project
                                .sequence
                                .markers
                                .sort_by(|a, b| a.frame.total_cmp(&b.frame));
                        }
                        if started_tx {
                            let _ = session.commit_transaction(true);
                        }
                    }

                    // 33px blank gap + 1px #242424 divider directly above Multisample motion blur
                    ui.add_space(33.0);
                    let (gap_div, _) =
                        ui.allocate_exact_size(Vec2::new(total_w, 1.0), Sense::hover());
                    ui.painter()
                        .rect_filled(gap_div, CornerRadius::ZERO, BORDER_ROW_SOFT);

                    // Multisample motion blur (compact dropdown "off"/"on", NO stopwatch)
                    let mb_prop = lookup_session_prop(session, "sequence", "motionBlur.enabled");
                    let mut mb_idx = eval_prop_u32(mb_prop.as_ref(), frame, 0) as usize;
                    let mb_changed = cm3_property_row_aligned(
                        ui,
                        "Multisample motion blur",
                        true,
                        state.prop_columns,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_dropdown(
                                ui,
                                Id::unique("seq_mb_enabled"),
                                &mut mb_idx,
                                &["off", "on"],
                                false,
                                viewport_rect,
                                theme,
                            )
                        },
                    );
                    if mb_changed {
                        let _ = session.execute(Command::SetPropertyValue {
                            target_id: "sequence".to_owned(),
                            property: "motionBlur.enabled".to_owned(),
                            value: json!(mb_idx),
                        });
                    }

                    // Samples (16, with stopwatch) & Shutter (0.50, with stopwatch)
                    show_session_number_prop(
                        ui,
                        session,
                        "sequence",
                        "motionBlur.samples",
                        "Samples",
                        16.0,
                        ScrubConfig::new(1.0, 64.0, 1.0, 0),
                        frame,
                        0.0,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                    show_session_number_prop(
                        ui,
                        session,
                        "sequence",
                        "motionBlur.shutter",
                        "Shutter",
                        0.5,
                        ScrubConfig::new(0.0, 2.0, 0.05, 2),
                        frame,
                        0.0,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );

                    // Controls nested section (Expressions+): clicking + adds a new slider control;
                    // empty below when no controls exist (no permanent ctl_value input or placeholder)
                    if session.plugins.is_enabled("expressions-plus") {
                        let (add_ctl, _) = cm3_collapsible_section_with_action(
                            ui,
                            Id::unique("seq_expr_controls"),
                            "Controls",
                            true,
                            true,
                            state.prop_columns.label_ratio,
                            Some((Cm3Icon::Add, "Add control")),
                            theme,
                            |ui| {
                                let ctl_keys: Vec<String> = session
                                    .document()
                                    .sequence_properties
                                    .keys()
                                    .filter(|k| k.starts_with("ctl_"))
                                    .cloned()
                                    .collect();
                                for key in ctl_keys {
                                    let display_lbl = key.strip_prefix("ctl_").unwrap_or(&key);
                                    let label = humanize_prop_label(display_lbl);
                                    show_session_number_prop(
                                        ui,
                                        session,
                                        "sequence",
                                        &key,
                                        &label,
                                        0.0,
                                        ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 2),
                                        frame,
                                        0.0,
                                        viewport_rect,
                                        state,
                                        theme,
                                        actions,
                                    );
                                }
                            },
                        );
                        if add_ctl {
                            let existing = &session.document().sequence_properties;
                            let mut idx = state.seq_control_counter.max(1);
                            let prop_key = loop {
                                let candidate = format!("ctl_slider_{idx}");
                                if !existing.contains_key(&candidate) {
                                    state.seq_control_counter = idx + 1;
                                    break candidate;
                                }
                                idx += 1;
                            };
                            let _ = session.execute(Command::SetPropertyValue {
                                target_id: "sequence".to_owned(),
                                property: prop_key,
                                value: json!(0.0),
                            });
                        }
                    }
                },
            );

            if reset_seq {
                let started_tx = session.begin_transaction("reset sequence").is_ok();
                let _ = session.execute(Command::SetResolution {
                    width: 1920,
                    height: 1080,
                });
                let _ = session.execute(Command::SetFps { fps: 30.0 });
                let _ = session.execute(Command::ResetPropertyList {
                    target_id: "sequence".to_owned(),
                });
                if started_tx && session.history.is_in_transaction() {
                    let _ = session.commit_transaction(true);
                }
            }
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
    let layer_title: &str = match clip_snapshot.kind.as_str() {
        "scene" => "Scene",
        "text" | "text_2d" => "Text",
        "shape" | "preset_shape" | "zoidium:shape-plus/shape-layer" => "Shape",
        "audio" => "Audio",
        "adjustment" => "Adjustment",
        "composite" => "Composite",
        "video" => "Video",
        "image" => "Image",
        _ => clip_snapshot.name.as_str(),
    };

    ScrollArea::vertical()
        .id_salt("edit_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;

            // Root collapsible #222222 bar titled with just the clip name (NO "Clip: " prefix)
            cm3_collapsible_section_with_action(
                ui,
                Id::unique(("edit_clip_root", &clip_id)),
                &clip_snapshot.name,
                true,
                false,
                state.prop_columns.label_ratio,
                None,
                theme,
                |ui| {
                    // Nested #2a2a2a "Properties" section (Media + Time)
                    let (reset_props, _) = cm3_collapsible_section_with_action(
                        ui,
                        Id::unique(("edit_clip_props", &clip_id)),
                        "Properties",
                        true,
                        true,
                        state.prop_columns.label_ratio,
                        Some((Cm3Icon::Reset, "Reset properties")),
                        theme,
                        |ui| {
                            // Media (read-only filename or "(none)" in #999999)
                            let media_label = eval_prop_str(
                                clip_snapshot.properties.get("assetId"),
                                local_frame,
                                "",
                            );
                            let resolved_media = if media_label.is_empty() {
                                "(none)".to_owned()
                            } else {
                                session
                                    .project
                                    .assets
                                    .get(&media_label)
                                    .map(|a| a.filename.clone())
                                    .unwrap_or(media_label)
                            };
                            cm3_property_row_aligned(
                                ui,
                                "Media",
                                true,
                                state.prop_columns,
                                theme,
                                |_| {},
                                |ui| {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(&resolved_media)
                                                .font(theme.font(FONT_BODY))
                                                .color(TEXT_DISABLED),
                                        )
                                        .truncate(),
                                    );
                                },
                            );

                            // Time (0.000, with stopwatch)
                            show_session_number_prop(
                                ui,
                                session,
                                &clip_id,
                                "time",
                                "Time",
                                (local_frame / session.document().fps.max(1.0)).max(0.0),
                                ScrubConfig::new(0.0, 100_000.0, 0.05, 3),
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );
                        },
                    );
                    if reset_props {
                        let time_animated = clip_snapshot
                            .properties
                            .get("time")
                            .is_some_and(|p| p.animated);
                        let started_tx = session.begin_transaction("reset properties").is_ok();
                        if time_animated {
                            let _ = session.execute(Command::TogglePropertyAnimated {
                                target_id: clip_id.clone(),
                                property: "time".to_owned(),
                                frame: normalized_kf_frame(session, local_frame),
                            });
                        }
                        let _ = session.execute(Command::SetPropertyValue {
                            target_id: clip_id.clone(),
                            property: "time".to_owned(),
                            value: json!(0.0),
                        });
                        if started_tx && session.history.is_in_transaction() {
                            let _ = session.commit_transaction(true);
                        }
                    }

                    // Nested #2a2a2a layer section ("Scene" / "Text" / "Shape" / "Audio" etc.)
                    let (reset_layer, _) = cm3_collapsible_section_with_action(
                        ui,
                        Id::unique(("edit_layer_root", &clip_id)),
                        layer_title,
                        true,
                        true,
                        state.prop_columns.label_ratio,
                        Some((Cm3Icon::Reset, "Reset layer")),
                        theme,
                        |ui| {
                            if clip_snapshot.kind == "audio" {
                                show_session_number_prop(
                                    ui,
                                    session,
                                    &clip_id,
                                    "volume",
                                    "Volume",
                                    1.0,
                                    ScrubConfig::new(0.0, 4.0, 0.05, 2),
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
                                    "pan",
                                    "Pan",
                                    0.0,
                                    ScrubConfig::new(-1.0, 1.0, 0.05, 2),
                                    local_frame,
                                    clip_start,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                                return;
                            }

                            // Suppress the bottom divider drawn by cm3_collapsible_section_with_action
                            // directly beneath the nested layer ("Scene") header before Masks.
                            let hdr_div_y = (ui.cursor().top() - 1.0).round();
                            ui.painter().rect_filled(
                                Rect::from_min_max(
                                    Pos2::new(panel_rect.left() + 1.0, hdr_div_y),
                                    Pos2::new(panel_rect.right() - 1.0, hdr_div_y + 1.0),
                                ),
                                CornerRadius::ZERO,
                                BG_PANEL,
                            );

                            // Masks header + subtle borderless "+ Add mask" (#000000, 55px total block: 24 + 30 + 1px divider)
                            let mask_w = ui.available_width();
                            let (masks_hdr, _) =
                                ui.allocate_exact_size(Vec2::new(mask_w, 24.0), Sense::hover());
                            ui.painter().text(
                                Pos2::new(masks_hdr.left() + 15.0, masks_hdr.top() + 10.5),
                                Align2::LEFT_CENTER,
                                "Masks",
                                theme.font(FONT_BODY),
                                TEXT_PRIMARY,
                            );
                            let (add_mask_rect, add_mask_resp) =
                                ui.allocate_exact_size(Vec2::new(mask_w, 30.0), Sense::click());
                            let add_mask_resp =
                                add_mask_resp.on_hover_cursor(CursorIcon::PointingHand);
                            let add_mask_color = if add_mask_resp.hovered() {
                                TEXT_PRIMARY
                            } else {
                                VOID_BLACK
                            };
                            ui.painter().text(
                                Pos2::new(add_mask_rect.left() + 25.0, add_mask_rect.top() + 10.0),
                                Align2::LEFT_CENTER,
                                "+ Add mask",
                                theme.font(FONT_EASING),
                                add_mask_color,
                            );
                            let (mask_div, _) =
                                ui.allocate_exact_size(Vec2::new(mask_w, 1.0), Sense::hover());
                            ui.painter()
                                .rect_filled(mask_div, CornerRadius::ZERO, BORDER_ROW_SOFT);
                            if add_mask_resp.clicked() {
                                let _ = session.execute(Command::AddEffect {
                                    clip_id: clip_id.clone(),
                                    kind: "mask".to_owned(),
                                    name: Some("Mask".to_owned()),
                                });
                            }

                            // Scene3D / visual layer backdrop & resolution controls (no stopwatch)
                            if clip_snapshot.kind == "scene" {
                                // Scene Background dropdown ("Transparent" / "Color")
                                let mut bg_mode_idx = eval_prop_u32(
                                    clip_snapshot.properties.get("sceneBackground"),
                                    local_frame,
                                    0,
                                ) as usize;
                                let bg_changed = cm3_property_row_aligned(
                                    ui,
                                    "Scene Background",
                                    true,
                                    state.prop_columns,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        cm3_dropdown(
                                            ui,
                                            Id::unique(("clip_scene_bg_mode", &clip_id)),
                                            &mut bg_mode_idx,
                                            &["Transparent", "Color"],
                                            false,
                                            viewport_rect,
                                            theme,
                                        )
                                    },
                                );
                                if bg_changed {
                                    let _ = session.execute(Command::SetPropertyValue {
                                        target_id: clip_id.clone(),
                                        property: "sceneBackground".to_owned(),
                                        value: json!(bg_mode_idx),
                                    });
                                }

                                // Backdrop Color swatch (no stopwatch, exact 28px pitch)
                                let bg_rgb = eval_prop_vec3(
                                    clip_snapshot.properties.get("backgroundColor"),
                                    local_frame,
                                    [0.0, 0.0, 0.0],
                                );
                                let mut bg_rgba = Rgba::new(
                                    bg_rgb[0] as f32,
                                    bg_rgb[1] as f32,
                                    bg_rgb[2] as f32,
                                    1.0,
                                );
                                let backdrop_before_y = ui.cursor().top();
                                let cev = cm3_property_row_aligned(
                                    ui,
                                    "Backdrop Color",
                                    true,
                                    state.prop_columns,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        cm3_hsva_color_picker(
                                            ui,
                                            Id::unique(("clip_backdrop_color", &clip_id)),
                                            &mut bg_rgba,
                                            false,
                                            false,
                                            viewport_rect,
                                            theme,
                                        )
                                    },
                                );
                                let backdrop_after_y = ui.cursor().top();
                                if backdrop_after_y - backdrop_before_y < 27.5 {
                                    let old_div_y = (backdrop_after_y - 1.0).round();
                                    let row_w = ui.available_width();
                                    let row_left = ui.cursor().left().round();
                                    ui.painter().rect_filled(
                                        Rect::from_min_max(
                                            Pos2::new(row_left, old_div_y),
                                            Pos2::new(row_left + row_w, old_div_y + 1.0),
                                        ),
                                        CornerRadius::ZERO,
                                        BG_PANEL,
                                    );
                                    let (new_div, _) = ui
                                        .allocate_exact_size(Vec2::new(row_w, 1.0), Sense::hover());
                                    ui.painter().rect_filled(
                                        new_div,
                                        CornerRadius::ZERO,
                                        BORDER_ROW_SOFT,
                                    );
                                }
                                touch_inspector_tx_owner(state, &clip_id, "backgroundColor");
                                apply_color_picker_to_session(
                                    session,
                                    &clip_id,
                                    "backgroundColor",
                                    false,
                                    local_frame,
                                    json!([
                                        f64::from(bg_rgba.0[0]),
                                        f64::from(bg_rgba.0[1]),
                                        f64::from(bg_rgba.0[2])
                                    ]),
                                    cev,
                                    state,
                                );
                                if cev.eyedropper_clicked {
                                    actions.push(PanelAction::EyedropperActivate {
                                        target_id: clip_id.clone(),
                                        property: "backgroundColor".to_owned(),
                                    });
                                }

                                // Backdrop Opacity (1.00, no stopwatch)
                                touch_inspector_tx_owner(state, &clip_id, "backdropOpacity");
                                let mut bg_op = eval_prop_f64(
                                    clip_snapshot.properties.get("backdropOpacity"),
                                    local_frame,
                                    1.0,
                                );
                                let op_ev = cm3_property_row_aligned(
                                    ui,
                                    "Backdrop Opacity",
                                    true,
                                    state.prop_columns,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        cm3_scrub_number(
                                            ui,
                                            Id::unique(("clip_backdrop_opacity", &clip_id)),
                                            &mut bg_op,
                                            ScrubConfig::new(0.0, 1.0, 0.05, 2),
                                            theme,
                                        )
                                    },
                                );
                                apply_scrub_to_session(
                                    session,
                                    &clip_id,
                                    "backdropOpacity",
                                    false,
                                    local_frame,
                                    json!(bg_op),
                                    op_ev,
                                    state,
                                );

                                // Resolution (1920 x 1080 with "x" separator, no stopwatch)
                                touch_inspector_tx_owner(state, &clip_id, "resolution");
                                let [seq_w, seq_h] = session.document().resolution;
                                let mut layer_res = eval_prop_vec2(
                                    clip_snapshot.properties.get("resolution"),
                                    local_frame,
                                    [f64::from(seq_w), f64::from(seq_h)],
                                );
                                let res_ev = cm3_property_row_aligned(
                                    ui,
                                    "Resolution",
                                    true,
                                    state.prop_columns,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        cm3_vector2_with_sep(
                                            ui,
                                            Id::unique(("clip_resolution", &clip_id)),
                                            &mut layer_res,
                                            &ScrubConfig::new(16.0, 7680.0, 10.0, 0),
                                            false,
                                            "x",
                                            theme,
                                        )
                                    },
                                );
                                apply_scrub_to_session(
                                    session,
                                    &clip_id,
                                    "resolution",
                                    false,
                                    local_frame,
                                    json!(layer_res),
                                    res_ev,
                                    state,
                                );
                            }

                            // Kind-specific 2D text / shape properties
                            match clip_snapshot.kind.as_str() {
                                "text" | "text_2d" => {
                                    show_session_text_prop(
                                        ui,
                                        session,
                                        &clip_id,
                                        "text",
                                        "Text",
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
                                        "Font size",
                                        72.0,
                                        ScrubConfig::new(4.0, 1000.0, 1.0, 0),
                                        local_frame,
                                        clip_start,
                                        viewport_rect,
                                        state,
                                        theme,
                                        actions,
                                    );
                                    show_session_color_prop(
                                        ui,
                                        session,
                                        &clip_id,
                                        "color",
                                        "Color",
                                        [1.0, 1.0, 1.0],
                                        local_frame,
                                        clip_start,
                                        viewport_rect,
                                        state,
                                        theme,
                                        actions,
                                    );
                                }
                                "shape" | "preset_shape" => {
                                    show_session_color_prop(
                                        ui,
                                        session,
                                        &clip_id,
                                        "fillColor",
                                        "Fill color",
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
                                        "Stroke width",
                                        0.0,
                                        ScrubConfig::new(0.0, 200.0, 1.0, 1),
                                        local_frame,
                                        clip_start,
                                        viewport_rect,
                                        state,
                                        theme,
                                        actions,
                                    );
                                }
                                _ => {}
                            }

                            // Transform & Compositing rows (Position, Scale, Rotation, Opacity, Blending mode)
                            show_session_vec2_prop(
                                ui,
                                session,
                                &clip_id,
                                "position",
                                "Position",
                                [0.0, 0.0],
                                ScrubConfig::new(-10_000.0, 10_000.0, 1.0, 2),
                                false,
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );
                            show_session_vec2_prop(
                                ui,
                                session,
                                &clip_id,
                                "scale",
                                "Scale",
                                [1.0, 1.0],
                                ScrubConfig::new(-100.0, 100.0, 0.05, 2),
                                true,
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
                                "rotation",
                                "Rotation",
                                0.0,
                                ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 1)
                                    .with_scale_factor(std::f64::consts::PI / 180.0),
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
                                "opacity",
                                "Opacity",
                                1.0,
                                ScrubConfig::new(0.0, 1.0, 0.05, 2),
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );

                            // Blending mode dropdown (no stopwatch)
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
                            let changed = cm3_property_row_aligned(
                                ui,
                                "Blending mode",
                                true,
                                state.prop_columns,
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
                        },
                    );
                    if reset_layer {
                        let _ = session.execute(Command::ResetPropertyList {
                            target_id: clip_id.clone(),
                        });
                    }
                },
            );
        });
}

// ---------------------------------------------------------------------------
// Tab 4: Objects (objects) — 40/60 Auto-Split 3D Tree + Inspector
// ---------------------------------------------------------------------------

fn build_object_picker_items(session: &Session) -> Vec<FuzzyItem> {
    let mut items = vec![
        // Core 10 CM3/Zoidium Objects (exact Zoidium OBJECTS 10 ordering & descriptions)
        FuzzyItem {
            id: "box".into(),
            category: "Objects".into(),
            name: "Shape".into(),
            desc: "Primitive object such as a box or sphere.".into(),
        },
        FuzzyItem {
            id: "text".into(),
            category: "Objects".into(),
            name: "Text".into(),
            desc: "Custom 3D text.".into(),
        },
        FuzzyItem {
            id: "spot_light".into(),
            category: "Objects".into(),
            name: "Light".into(),
            desc: "Creates a Spot Light with shadows in a scene.".into(),
        },
        FuzzyItem {
            id: "group".into(),
            category: "Objects".into(),
            name: "Group".into(),
            desc: "Group multiple objects together.".into(),
        },
        FuzzyItem {
            id: "camera".into(),
            category: "Objects".into(),
            name: "Camera".into(),
            desc: "Perspective or orthographic camera.".into(),
        },
        FuzzyItem {
            id: "particles".into(),
            category: "Objects".into(),
            name: "Particles".into(),
            desc: "3D particle emitter.".into(),
        },
        FuzzyItem {
            id: "model".into(),
            category: "Objects".into(),
            name: "3D Model".into(),
            desc: "Imported OBJ/GLTF mesh.".into(),
        },
        FuzzyItem {
            id: "sphere".into(),
            category: "Objects".into(),
            name: "Sphere".into(),
            desc: "UV sphere mesh.".into(),
        },
        FuzzyItem {
            id: "cylinder".into(),
            category: "Objects".into(),
            name: "Cylinder".into(),
            desc: "Cylinder or cone primitive.".into(),
        },
        FuzzyItem {
            id: "donut".into(),
            category: "Objects".into(),
            name: "Donut".into(),
            desc: "Torus ring primitive.".into(),
        },
        // Additional CM3 Primitives & Lights
        FuzzyItem {
            id: "rectangle".into(),
            category: "More Objects".into(),
            name: "Rectangle".into(),
            desc: "Flat 3D plane.".into(),
        },
        FuzzyItem {
            id: "circle".into(),
            category: "More Objects".into(),
            name: "Circle".into(),
            desc: "Flat 3D disc.".into(),
        },
        FuzzyItem {
            id: "wire".into(),
            category: "More Objects".into(),
            name: "Wire".into(),
            desc: "3D polyline tube.".into(),
        },
        FuzzyItem {
            id: "point_light".into(),
            category: "More Objects".into(),
            name: "Point Light".into(),
            desc: "Omnidirectional point light.".into(),
        },
        FuzzyItem {
            id: "directional_light".into(),
            category: "More Objects".into(),
            name: "Directional Light".into(),
            desc: "Parallel sun-like light.".into(),
        },
        FuzzyItem {
            id: "hemisphere_light".into(),
            category: "More Objects".into(),
            name: "Hemisphere Light".into(),
            desc: "Sky & ground ambient light.".into(),
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
            category: "Lens Flares".into(),
            name: "Lens Flare".into(),
            desc: "3D depth-occluded lens flare".into(),
        });
    }

    if session.plugins.is_enabled("trapcode-suite") {
        for (id, name, desc) in [
            (
                "zoidium:trapcode-suite/particular",
                "3D Particle Emitter",
                "Deterministic 3D fluid particle sim",
            ),
            (
                "zoidium:trapcode-suite/form",
                "Particle Lattice",
                "Deformable 3D particle lattice",
            ),
            (
                "zoidium:trapcode-suite/plexus",
                "Point Cloud Network",
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
                category: "Particle & Light Suite".into(),
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
        empty_panel_message(ui, "select a 3D scene clip", theme);
        return;
    };
    let Some(clip) = find_clip_clone(session, &clip_id) else {
        empty_panel_message(ui, "select a 3D scene clip", theme);
        return;
    };
    if clip.kind != "scene" {
        empty_panel_message(ui, "select a 3D scene clip", theme);
        return;
    }

    let (top_rect, bot_rect) = begin_split_panel(
        ui,
        Id::unique("objects_40_60_split"),
        panel_rect,
        &mut state.objects_split,
    );
    ui.input(|i| {
        if (i.pointer.primary_pressed() || i.pointer.secondary_pressed())
            && let Some(pos) = i.pointer.interact_pos()
        {
            if top_rect.contains(pos) {
                state.objects_focus_bottom = false;
            } else if bot_rect.contains(pos) {
                state.objects_focus_bottom = true;
            }
        }
    });
    let top_inner = top_rect.shrink(1.0);
    let bot_inner = bot_rect.shrink(1.0);
    ui.painter()
        .rect_filled(top_rect, CornerRadius::ZERO, BG_PANEL);
    ui.painter()
        .rect_filled(bot_rect, CornerRadius::ZERO, BG_PANEL);

    show_split_pane(
        ui,
        Id::unique("objects_40_60_split_p0"),
        top_inner,
        |ui, _top_rect| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;
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
                        commit_inspector_tx_if_open(session, state);
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

            // 28px #222222 header bar (27px body + 1px #242424 bottom border), 7x7 framed square glyph,
            // 16px #cccccc "3D Objects" title, and right-aligned `+`
            let (hdr_rect, hdr_resp) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), Sense::click());
            ui.painter()
                .rect_filled(hdr_rect, CornerRadius::ZERO, BG_TITLE);
            let hdr_div_y = (hdr_rect.bottom() - 1.0).round();
            ui.painter().rect_filled(
                Rect::from_min_max(
                    Pos2::new(hdr_rect.left(), hdr_div_y),
                    Pos2::new(hdr_rect.right(), hdr_div_y + 1.0),
                ),
                CornerRadius::ZERO,
                BORDER_ROW_SOFT,
            );
            paint_cm3_tree_box(
                ui.painter(),
                Pos2::new(hdr_rect.left() + 10.5, hdr_rect.center().y + 1.0),
                state.objects_list_open,
                if hdr_resp.hovered() {
                    TEXT_BRIGHT
                } else {
                    TEXT_MUTED
                },
            );
            ui.painter().text(
                Pos2::new(hdr_rect.left() + 21.0, hdr_rect.center().y),
                Align2::LEFT_CENTER,
                "3D Objects",
                theme.font(FONT_BODY),
                TEXT_PRIMARY,
            );
            let add_icon_rect = Rect::from_min_size(
                Pos2::new(hdr_rect.right() - 24.0, (hdr_rect.center().y - 8.0).round()),
                Vec2::splat(15.0),
            );
            let add_rect = add_icon_rect.expand(3.0);
            let add_resp = ui
                .interact(add_rect, Id::unique("objects_add_btn"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text("Add 3D object");
            if add_resp.hovered() {
                ui.painter().rect_stroke(
                    add_rect,
                    CornerRadius::ZERO,
                    Stroke::new(2.0, theme.accent()),
                    StrokeKind::Inside,
                );
            }
            paint_cm3_icon(
                ui.painter(),
                add_icon_rect,
                Cm3Icon::Add,
                if add_resp.hovered() {
                    TEXT_BRIGHT
                } else {
                    TEXT_PRIMARY
                },
            );
            if add_resp.clicked() {
                state.add_object_open = true;
                state.add_object_picker = FuzzyPickerState {
                    just_opened: true,
                    ..FuzzyPickerState::default()
                };
            } else if hdr_resp.clicked() {
                state.objects_list_open = !state.objects_list_open;
            }

            if !state.objects_list_open {
                return;
            }

            // Hierarchical 3D object tree
            ScrollArea::vertical()
                .id_salt("objects_tree_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = Vec2::ZERO;
                    if clip.objects_3d.is_empty() {
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
                            ui.allocate_exact_size(Vec2::new(row_w, 27.0), Sense::click_and_drag());
                        let pointer_in_row = ui
                            .input(|i| i.pointer.hover_pos().is_some_and(|p| row_rect.contains(p)));

                        if selected {
                            ui.painter()
                                .rect_filled(row_rect, CornerRadius::ZERO, theme.accent());
                        } else if row_resp.hovered() || pointer_in_row {
                            ui.painter()
                                .rect_filled(row_rect, CornerRadius::ZERO, BG_HOVER);
                        }
                        ui.painter().line_segment(
                            [
                                Pos2::new(row_rect.left(), row_rect.bottom() - 0.5),
                                Pos2::new(row_rect.right(), row_rect.bottom() - 0.5),
                            ],
                            Stroke::new(1.0, BORDER_ROW_SOFT),
                        );

                        // Top-level row name indent: 36px from panel inner left
                        let label_left = row_rect.left() + 36.0 + depth as f32 * 15.0;

                        // Collapse triangle for containers (positioned left of label)
                        if has_children || is_group {
                            let tri_rect = Rect::from_min_size(
                                Pos2::new(
                                    (label_left - 20.0).max(row_rect.left() + 4.0),
                                    row_rect.top() + 3.0,
                                ),
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

                        // Label or Inline Rename input (Duplicate & Remove live in Inspector header)
                        let label_right = (row_rect.right() - 8.0).max(label_left);
                        let label_rect = Rect::from_min_max(
                            Pos2::new(label_left, row_rect.top()),
                            Pos2::new(label_right, row_rect.bottom()),
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
                                let mut rename_ui =
                                    ui.new_child(egui::UiBuilder::new().max_rect(label_rect));
                                let resp = Frame::NONE
                                    .fill(BG_INPUT)
                                    .stroke(Stroke::new(1.0, theme.accent()))
                                    .inner_margin(egui::Margin::symmetric(3, 1))
                                    .show(&mut rename_ui, |ui| {
                                        ui.add(
                                            TextEdit::singleline(&mut rename.buffer)
                                                .id(edit_id)
                                                .frame(Frame::NONE)
                                                .font(theme.font(FONT_BODY))
                                                .text_color(theme.value())
                                                .desired_width(label_rect.width()),
                                        )
                                    })
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
                        if row_resp.clicked() {
                            commit_inspector_tx_if_open(session, state);
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
    // Crisp single 1px #333333 border around top pane (layout.rs paints the focus accent only when SideInspector is focused)
    paint_crisp_panel_border(ui.painter(), top_rect, BORDER_PANEL);
    // Re-read clip snapshot after pane 0 mutations (AddObject3D, duplicate, reorder, rename)
    // so the inspector immediately reflects newly created/selected objects without flashing empty.
    let clip_for_inspector = find_clip_clone(session, &clip_id).unwrap_or(clip);
    show_split_pane(
        ui,
        Id::unique("objects_40_60_split_p1"),
        bot_inner,
        |ui, bot_rect| {
            // Pane 1 (60%): Selected 3D Object Property Inspector
            show_object_inspector(
                ui,
                bot_rect,
                viewport_rect,
                &clip_for_inspector,
                session,
                theme,
                state,
                actions,
            );
        },
    );
    // Crisp single 1px #333333 border around bottom pane (layout.rs paints the focus accent only when SideInspector is focused)
    paint_crisp_panel_border(ui.painter(), bot_rect, BORDER_PANEL);
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
    let Some(src_idx) = clip.objects_3d.iter().position(|o| o.id == src_id) else {
        return;
    };
    let Some(dst_idx) = clip.objects_3d.iter().position(|o| o.id == dst_id) else {
        return;
    };
    let dst = &clip.objects_3d[dst_idx];
    let (new_parent, mut target_idx) = match mode {
        TreeDropMode::Inside => (Some(dst.id.clone()), clip.objects_3d.len()),
        TreeDropMode::Before => (dst.parent_id.clone(), dst_idx),
        TreeDropMode::After => (dst.parent_id.clone(), dst_idx + 1),
    };
    if !alt_clone && src_idx < target_idx {
        target_idx -= 1;
    }

    if alt_clone {
        let started_tx = session.begin_transaction("clone object").is_ok();
        if let Ok(res) = session.execute(Command::DuplicateObject3D {
            clip_id: clip.id.clone(),
            object_id: src_id.to_owned(),
        }) && let Some(cloned_id) = res.get("object_id").and_then(Value::as_str)
        {
            let _ = session.execute(Command::ReorderObject3D {
                clip_id: clip.id.clone(),
                object_id: cloned_id.to_owned(),
                parent_id: new_parent,
                index: target_idx,
            });
        }
        if started_tx && session.history.is_in_transaction() {
            let _ = session.commit_transaction(true);
        }
    } else {
        let _ = session.execute(Command::ReorderObject3D {
            clip_id: clip.id.clone(),
            object_id: src_id.to_owned(),
            parent_id: new_parent,
            index: target_idx,
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
            ui.spacing_mut().item_spacing.y = 0.0;
            let (root_open, hdr_act) = property_header_with_actions(
                ui,
                Id::unique(("obj_root_hdr", &obj.id)),
                &obj.name,
                state.prop_columns.label_ratio,
                theme,
            );
            if hdr_act.reset {
                let _ = session.execute(Command::ResetPropertyList {
                    target_id: obj.id.clone(),
                });
            }
            if hdr_act.duplicate {
                let _ = session.execute(Command::DuplicateObject3D {
                    clip_id: clip.id.clone(),
                    object_id: obj.id.clone(),
                });
            }
            if hdr_act.remove {
                commit_inspector_tx_if_open(session, state);
                let _ = session.execute(Command::DeleteObject3D {
                    clip_id: clip.id.clone(),
                    object_id: obj.id.clone(),
                });
                return;
            }
            if !root_open {
                return;
            }
            state.inspector_root_indent = true;

            let is_camera = obj.kind.contains("camera");
            let is_light = obj.kind.contains("light");
            let is_text = obj.kind == "text" || obj.kind == "text_3d";
            let is_particles = obj.kind == "particles";

            // Build reusable target-object dropdown list for Track to / Follow constraints
            let constraint_targets: Vec<(String, String)> = clip
                .objects_3d
                .iter()
                .filter(|other| other.id != obj.id)
                .map(|other| (other.id.clone(), other.name.clone()))
                .collect();

            if is_camera {
                // Camera inspector (exact Zoidium 11_objects_camera.png flat layout)
                show_session_vec3_prop(
                    ui,
                    session,
                    &obj.id,
                    "position",
                    "Position",
                    [0.0, 0.0, 80.0],
                    ScrubConfig::new(-100_000.0, 100_000.0, 1.0, 2),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec3_prop(
                    ui,
                    session,
                    &obj.id,
                    "rotation",
                    "Rotation",
                    [0.0, 0.0, 0.0],
                    ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 2)
                        .with_scale_factor(std::f64::consts::PI / 180.0),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_option_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "eulerOrder",
                    "Rotation order",
                    &["XYZ", "YZX", "ZXY", "XZY", "YXZ", "ZYX"],
                    0,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );

                if session.plugins.is_enabled("constraints") {
                    show_constraint_target_rows(
                        ui,
                        session,
                        &obj.id,
                        &constraint_targets,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }

                // All 8 procedural Camera Shake properties (11_objects_camera.png)
                show_session_number_prop(
                    ui,
                    session,
                    &obj.id,
                    "shakeAmplitude",
                    "Shake amount",
                    0.0,
                    ScrubConfig::new(0.0, 1.0, 0.01, 3),
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
                    "shakeSpeed",
                    "Shake speed",
                    30.0,
                    ScrubConfig::new(0.0, 1000.0, 0.5, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_option_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "shake.mode",
                    "Shake mode",
                    &["rotate", "move"],
                    0,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec4_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "shake.amplitude",
                    "Amplitude",
                    [12.0, 12.0, 40.0, 20.0],
                    ScrubConfig::new(0.0, 1000.0, 0.5, 1),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec4_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "shake.frequency",
                    "Frequency",
                    [0.60, 0.80, 0.70, 0.90],
                    ScrubConfig::new(0.0, 100.0, 0.05, 2),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec4_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "shake.phase",
                    "Phase",
                    [0.0, 11.0, 20.0, 31.0],
                    ScrubConfig::new(-1000.0, 1000.0, 0.5, 1),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec4_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "shake.noise",
                    "Noise",
                    [0.90, 0.90, 0.90, 0.55],
                    ScrubConfig::new(0.0, 10.0, 0.05, 2),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec4_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "shake.smooth",
                    "Smooth",
                    [1.0, 1.0, 1.0, 1.0],
                    ScrubConfig::new(0.0, 10.0, 1.0, 0),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );

                // Camera projection & Camera+ optics below shake rows
                show_session_option_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "camera.kind",
                    "Camera type",
                    &["Perspective", "Orthographic"],
                    0,
                    false,
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
                    "fov",
                    "FOV",
                    60.0,
                    ScrubConfig::new(1.0, 179.0, 0.5, 1),
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
                    "near",
                    "Near",
                    0.1,
                    ScrubConfig::new(0.01, 1000.0, 0.1, 2),
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
                    "far",
                    "Far",
                    5000.0,
                    ScrubConfig::new(1.0, 100_000.0, 10.0, 0),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
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
                        actions,
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
                        actions,
                    );
                }
                return;
            }

            if is_light {
                // Light inspector: flat, no Rotation/Scale; conditional by light.kind
                show_session_option_prop(
                    ui,
                    session,
                    &obj.id,
                    "enabled",
                    "Enabled",
                    &["off", "on"],
                    1,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                let default_lkind = match obj.kind.as_str() {
                    "point_light" => 1,
                    "directional_light" => 2,
                    "hemisphere_light" => 3,
                    _ => 0,
                };
                show_session_option_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "light.kind",
                    "Light type",
                    &["Spot", "Point", "Directional", "Hemisphere"],
                    default_lkind,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                let lkind = eval_prop_u32(
                    lookup_session_prop(session, &obj.id, "light.kind").as_ref(),
                    local_frame,
                    default_lkind,
                );
                if lkind != 3 {
                    show_session_vec3_prop(
                        ui,
                        session,
                        &obj.id,
                        "position",
                        "Position",
                        [0.0, 10.0, 0.0],
                        ScrubConfig::new(-100_000.0, 100_000.0, 1.0, 2),
                        false,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                if lkind == 0 || lkind == 2 {
                    show_session_vec3_prop(
                        ui,
                        session,
                        &obj.id,
                        "target",
                        "Target",
                        [0.0, 0.0, 0.0],
                        ScrubConfig::new(-100_000.0, 100_000.0, 1.0, 2),
                        false,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                show_session_color_prop(
                    ui,
                    session,
                    &obj.id,
                    "color",
                    if lkind == 3 { "Sky color" } else { "Color" },
                    [1.0, 1.0, 1.0],
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                if lkind == 3 {
                    show_session_color_prop(
                        ui,
                        session,
                        &obj.id,
                        "groundColor",
                        "Ground color",
                        [1.0, 1.0, 1.0],
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                show_session_number_prop(
                    ui,
                    session,
                    &obj.id,
                    "intensity",
                    "Intensity",
                    1.0,
                    ScrubConfig::new(0.0, 100.0, 0.1, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                if lkind == 0 {
                    show_session_number_prop(
                        ui,
                        session,
                        &obj.id,
                        "angle",
                        "Angle",
                        60.0,
                        ScrubConfig::new(0.0, 90.0, 0.5, 1),
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
                        "penumbra",
                        "Penumbra",
                        0.0,
                        ScrubConfig::new(0.0, 1.0, 0.05, 2),
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                if lkind == 0 || lkind == 1 {
                    show_session_number_prop(
                        ui,
                        session,
                        &obj.id,
                        "distance",
                        "Range",
                        0.0,
                        ScrubConfig::new(0.0, 10_000.0, 1.0, 1),
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
                        "decay",
                        "Decay",
                        1.0,
                        ScrubConfig::new(0.0, 10.0, 0.1, 2),
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                if lkind != 3 {
                    show_session_option_prop_ex(
                        ui,
                        session,
                        &obj.id,
                        "castShadow",
                        "Cast shadow",
                        &["off", "on"],
                        0,
                        false,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                if session.plugins.is_enabled("constraints") {
                    show_constraint_target_rows(
                        ui,
                        session,
                        &obj.id,
                        &constraint_targets,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                return;
            }

            // Standard flat object properties (Text3D, Shapes, Model, Group, Particles)
            show_session_option_prop(
                ui,
                session,
                &obj.id,
                "enabled",
                "Enabled",
                &["off", "on"],
                1,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_vec3_prop(
                ui,
                session,
                &obj.id,
                "position",
                "Position",
                [0.0, 0.0, 0.0],
                ScrubConfig::new(-100_000.0, 100_000.0, 1.0, 2),
                false,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_vec3_prop(
                ui,
                session,
                &obj.id,
                "rotation",
                "Rotation",
                [0.0, 0.0, 0.0],
                ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 2)
                    .with_scale_factor(std::f64::consts::PI / 180.0),
                false,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
            show_session_vec3_prop(
                ui,
                session,
                &obj.id,
                "scale",
                "Scale",
                [1.0, 1.0, 1.0],
                ScrubConfig::new(-1000.0, 1000.0, 0.05, 3),
                true,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );

            if is_text {
                // Non-animated 3D Text geometry properties matching CM3 Spec 03 §3.4
                show_session_text_prop(
                    ui,
                    session,
                    &obj.id,
                    "text",
                    "Text",
                    "text",
                    local_frame,
                    state,
                    theme,
                );
                // Font selector (preset + project font assets + file picker)
                let cur_font = eval_prop_str(
                    lookup_session_prop(session, &obj.id, "font").as_ref(),
                    local_frame,
                    "/assets/fonts/2d/bebas.ttf",
                );
                let mut font_choices: Vec<(String, String)> = vec![
                    (
                        "/assets/fonts/2d/bebas.ttf".to_owned(),
                        "Bebas Neue".to_owned(),
                    ),
                    ("Geist-Regular.ttf".to_owned(), "Geist".to_owned()),
                ];
                for asset in session.project.assets.values() {
                    let lower = asset.filename.to_ascii_lowercase();
                    if lower.ends_with(".ttf")
                        || lower.ends_with(".otf")
                        || lower.ends_with(".woff")
                    {
                        font_choices.push((asset.id.clone(), asset.filename.clone()));
                    }
                }
                let font_labels: Vec<&str> =
                    font_choices.iter().map(|(_, lbl)| lbl.as_str()).collect();
                let mut font_idx = font_choices
                    .iter()
                    .position(|(val, _)| val == &cur_font)
                    .unwrap_or(0);
                let mut font_changed = false;
                panel_property_row(
                    ui,
                    "Font",
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            font_changed = cm3_dropdown(
                                ui,
                                Id::unique((&obj.id, "text3d_font")),
                                &mut font_idx,
                                &font_labels,
                                false,
                                viewport_rect,
                                theme,
                            );
                            if cm3_icon_button(
                                ui,
                                Cm3Icon::Load,
                                17.0,
                                TEXT_PRIMARY,
                                "Choose custom font file...",
                                theme,
                            ) {
                                actions.push(PanelAction::PickAssetForProperty {
                                    target_id: obj.id.clone(),
                                    property: "font".to_owned(),
                                    is_lut: false,
                                });
                            }
                        });
                    },
                );
                if font_changed {
                    let _ = session.execute(Command::SetPropertyValue {
                        target_id: obj.id.clone(),
                        property: "font".to_owned(),
                        value: json!(font_choices[font_idx].0),
                    });
                }

                // Combined non-animated Size ([height, thickness], default [20, 3], 0 decimals)
                touch_inspector_tx_owner(state, &obj.id, "text3d_size");
                let h_val = eval_prop_f64(
                    lookup_session_prop(session, &obj.id, "height").as_ref(),
                    local_frame,
                    20.0,
                );
                let t_val = eval_prop_f64(
                    lookup_session_prop(session, &obj.id, "thickness").as_ref(),
                    local_frame,
                    3.0,
                );
                let mut size_vec = [h_val, t_val];
                let size_ev = panel_property_row(
                    ui,
                    "Size",
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_vector2(
                            ui,
                            Id::unique((&obj.id, "text3d_size")),
                            &mut size_vec,
                            ScrubConfig::new(0.1, 500.0, 1.0, 0),
                            false,
                            theme,
                        )
                    },
                );
                apply_paired_scalar_scrub(
                    session,
                    &obj.id,
                    "text3d_size",
                    ("height", size_vec[0]),
                    ("thickness", size_vec[1].max(0.0)),
                    size_ev,
                    state,
                );

                // Detail (non-animated scalar, default 5)
                touch_inspector_tx_owner(state, &obj.id, "detail");
                let mut detail_val = eval_prop_f64(
                    lookup_session_prop(session, &obj.id, "detail").as_ref(),
                    local_frame,
                    5.0,
                );
                let detail_ev = panel_property_row(
                    ui,
                    "Detail",
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_scrub_number(
                            ui,
                            Id::unique((&obj.id, "detail")),
                            &mut detail_val,
                            ScrubConfig::new(1.0, 200.0, 1.0, 0),
                            theme,
                        )
                    },
                );
                apply_scrub_to_session(
                    session,
                    &obj.id,
                    "detail",
                    false,
                    local_frame,
                    json!(detail_val.round().clamp(1.0, 200.0)),
                    detail_ev,
                    state,
                );

                // Bevel (non-animated off/on, default 0)
                show_session_option_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "bevelEnabled",
                    "Bevel",
                    &["off", "on"],
                    0,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                let bevel_on = eval_prop_u32(
                    lookup_session_prop(session, &obj.id, "bevelEnabled").as_ref(),
                    local_frame,
                    0,
                ) != 0;
                if bevel_on {
                    touch_inspector_tx_owner(state, &obj.id, "text3d_bevel_size");
                    let bt = eval_prop_f64(
                        lookup_session_prop(session, &obj.id, "bevelThickness").as_ref(),
                        local_frame,
                        0.1,
                    );
                    let bs = eval_prop_f64(
                        lookup_session_prop(session, &obj.id, "bevelSize").as_ref(),
                        local_frame,
                        0.5,
                    );
                    let mut bevel_vec = [bt, bs];
                    let bevel_ev = panel_property_row(
                        ui,
                        "Bevel size",
                        true,
                        state,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_vector2(
                                ui,
                                Id::unique((&obj.id, "text3d_bevel_size")),
                                &mut bevel_vec,
                                ScrubConfig::new(0.0, 100.0, 0.05, 2),
                                false,
                                theme,
                            )
                        },
                    );
                    apply_paired_scalar_scrub(
                        session,
                        &obj.id,
                        "text3d_bevel_size",
                        ("bevelThickness", bevel_vec[0]),
                        ("bevelSize", bevel_vec[1]),
                        bevel_ev,
                        state,
                    );
                }

                // Center point (positionMode)
                show_session_option_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "positionMode",
                    "Center point",
                    &["center", "left", "right", "top", "bottom", "front", "back"],
                    0,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );

                if session.plugins.is_enabled("text-plus") {
                    show_session_number_prop(
                        ui,
                        session,
                        &obj.id,
                        "bevelTension",
                        "Bevel tension",
                        0.5,
                        ScrubConfig::new(0.0, 1.0, 0.05, 2),
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
            } else if is_particles {
                show_session_number_prop(
                    ui,
                    session,
                    &obj.id,
                    "particles.number",
                    "Number",
                    100.0,
                    ScrubConfig::new(1.0, 100_000.0, 1.0, 0),
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
                    "particles.rate",
                    "Rate",
                    25.0,
                    ScrubConfig::new(0.0, 10_000.0, 0.5, 2),
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
                    "particles.lifetime",
                    "Lifetime",
                    2.0,
                    ScrubConfig::new(0.01, 600.0, 0.1, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            } else {
                // Shape / Model / Group geometry properties
                match obj.kind.as_str() {
                    "box" | "zoidium:geometry-plus/rounded_box" => {
                        show_session_vec3_prop(
                            ui,
                            session,
                            &obj.id,
                            "box_size",
                            "Size",
                            [10.0, 10.0, 10.0],
                            ScrubConfig::new(0.01, 10_000.0, 0.5, 2),
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                    "cylinder" => {
                        show_session_vec3_prop(
                            ui,
                            session,
                            &obj.id,
                            "cylinder_size",
                            "Size",
                            [10.0, 10.0, 20.0],
                            ScrubConfig::new(0.0, 10_000.0, 0.5, 2),
                            false,
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
                            "cylinder_detail",
                            "Detail",
                            16.0,
                            ScrubConfig::new(3.0, 128.0, 1.0, 0),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_option_prop_ex(
                            ui,
                            session,
                            &obj.id,
                            "cylinder_openEnded",
                            "Open ended",
                            &["off", "on"],
                            1,
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                    "rectangle" => {
                        show_session_vec2_prop(
                            ui,
                            session,
                            &obj.id,
                            "rect_size",
                            "Size",
                            [10.0, 10.0],
                            ScrubConfig::new(0.01, 10_000.0, 0.5, 2),
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                    "circle" => {
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            "circle_size",
                            "Radius",
                            10.0,
                            ScrubConfig::new(0.01, 10_000.0, 0.5, 2),
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
                            "circle_detail",
                            "Detail",
                            16.0,
                            ScrubConfig::new(3.0, 128.0, 1.0, 0),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                    "sphere" => {
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            "sphere_size",
                            "Radius",
                            10.0,
                            ScrubConfig::new(0.01, 10_000.0, 0.5, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_vec2_prop(
                            ui,
                            session,
                            &obj.id,
                            "sphere_detail",
                            "Detail",
                            [10.0, 10.0],
                            ScrubConfig::new(3.0, 128.0, 1.0, 0),
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                    "donut" => {
                        show_session_vec2_prop(
                            ui,
                            session,
                            &obj.id,
                            "donut_size",
                            "Size",
                            [10.0, 10.0],
                            ScrubConfig::new(0.01, 10_000.0, 0.5, 2),
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_vec2_prop(
                            ui,
                            session,
                            &obj.id,
                            "donut_detail",
                            "Detail",
                            [16.0, 16.0],
                            ScrubConfig::new(3.0, 128.0, 1.0, 0),
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                    "wire" => {
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            "path_thickness",
                            "Thickness",
                            1.0,
                            ScrubConfig::new(0.01, 500.0, 0.1, 2),
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
                            "path_detail",
                            "Detail",
                            16.0,
                            ScrubConfig::new(2.0, 256.0, 1.0, 0),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                    "model" => {
                        show_session_asset_prop(
                            ui,
                            session,
                            &obj.id,
                            "customGeometry",
                            "Model",
                            false,
                            local_frame,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                    "group" => {
                        show_session_option_prop_ex(
                            ui,
                            session,
                            &obj.id,
                            "reflectionVisibility",
                            "Reflection",
                            &["normal", "reflection only", "both"],
                            0,
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                    _ => {}
                }
            }

            // Constraints for non-camera/non-light objects
            if session.plugins.is_enabled("constraints") {
                show_constraint_target_rows(
                    ui,
                    session,
                    &obj.id,
                    &constraint_targets,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            }

            // Material section for all mesh/geometry objects (Text3D, Shapes, Models)
            let is_mesh_object =
                !is_light && !is_camera && !is_particles && !is_container_object(&obj.kind);
            if is_mesh_object {
                state.inspector_root_indent = false;
                cm3_collapsible_section_with_action(
                    ui,
                    Id::unique(("obj_material", &obj.id)),
                    "Material",
                    true,
                    true,
                    state.prop_columns.label_ratio,
                    None,
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
                        show_session_option_prop_ex(
                            ui,
                            session,
                            &obj.id,
                            "material.type",
                            "Material",
                            &mat_options,
                            0,
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        let mat_type = eval_prop_u32(
                            lookup_session_prop(session, &obj.id, "material.type").as_ref(),
                            local_frame,
                            0,
                        );
                        match mat_type {
                            0 => {
                                // Single color: Color + Render side
                                show_session_color_prop(
                                    ui,
                                    session,
                                    &obj.id,
                                    "material.color",
                                    "Color",
                                    [1.0, 1.0, 1.0],
                                    local_frame,
                                    clip_start,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                                show_session_option_prop_ex(
                                    ui,
                                    session,
                                    &obj.id,
                                    "material.side",
                                    "Render side",
                                    &["front", "back", "double"],
                                    0,
                                    false,
                                    local_frame,
                                    clip_start,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                            }
                            1 => {
                                // Texture: Image + Render side
                                show_session_asset_prop(
                                    ui,
                                    session,
                                    &obj.id,
                                    "material.texture",
                                    "Image",
                                    false,
                                    local_frame,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                                show_session_option_prop_ex(
                                    ui,
                                    session,
                                    &obj.id,
                                    "material.side",
                                    "Render side",
                                    &["front", "back", "double"],
                                    0,
                                    false,
                                    local_frame,
                                    clip_start,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                            }
                            2 => {
                                // Video: Video + Color + Opacity
                                show_session_asset_prop(
                                    ui,
                                    session,
                                    &obj.id,
                                    "material.texture",
                                    "Video",
                                    false,
                                    local_frame,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                                show_session_color_prop(
                                    ui,
                                    session,
                                    &obj.id,
                                    "material.color",
                                    "Color",
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
                                    "material.opacity",
                                    "Opacity",
                                    1.0,
                                    ScrubConfig::new(0.0, 1.0, 0.05, 2),
                                    local_frame,
                                    clip_start,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                            }
                            4 => {
                                // Custom shader material: Edit shader button
                                let cur_code = eval_prop_str(
                                    lookup_session_prop(session, &obj.id, "material.fragShader")
                                        .as_ref(),
                                    local_frame,
                                    DEFAULT_CUSTOM_SHADER_GLSL,
                                );
                                cm3_property_row_aligned(
                                    ui,
                                    "Fragment shader",
                                    true,
                                    state.prop_columns,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        if cm3_prop_button(
                                            ui,
                                            "Edit shader",
                                            ButtonVariant::Normal,
                                            false,
                                            theme,
                                        )
                                        .clicked()
                                        {
                                            state.shader_editor = Some(ShaderEditorTarget {
                                                effect_id: obj.id.clone(),
                                                property: "material.fragShader".to_owned(),
                                                effect_name: obj.name.clone(),
                                                code: cur_code.clone(),
                                                font_px: 14.0,
                                                error_message: None,
                                            });
                                            let spec = FloatingWindowSpec::new(
                                                "shader_editor",
                                                "Shader Editor",
                                            )
                                            .with_subtitle(obj.name.clone())
                                            .with_default_size(Vec2::new(560.0, 440.0));
                                            state
                                                .floating_windows
                                                .open_or_focus(&spec, viewport_rect);
                                        }
                                    },
                                );
                            }
                            _ => {
                                // Custom PBR & Material+ modes
                                show_session_color_prop(
                                    ui,
                                    session,
                                    &obj.id,
                                    "material.color",
                                    "Color",
                                    [1.0, 1.0, 1.0],
                                    local_frame,
                                    clip_start,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                                show_session_color_prop(
                                    ui,
                                    session,
                                    &obj.id,
                                    "material.emissive",
                                    "Emissive",
                                    [0.0, 0.0, 0.0],
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
                                    "Roughness",
                                    0.5,
                                    ScrubConfig::new(0.0, 1.0, 0.05, 2),
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
                                    "material.metalness",
                                    "Metalness",
                                    0.5,
                                    ScrubConfig::new(0.0, 1.0, 0.05, 2),
                                    local_frame,
                                    clip_start,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                                show_session_asset_prop(
                                    ui,
                                    session,
                                    &obj.id,
                                    "material.texture",
                                    "Texture",
                                    false,
                                    local_frame,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                                show_session_number_prop(
                                    ui,
                                    session,
                                    &obj.id,
                                    "material.opacity",
                                    "Opacity",
                                    1.0,
                                    ScrubConfig::new(0.0, 1.0, 0.05, 2),
                                    local_frame,
                                    clip_start,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                                show_session_option_prop_ex(
                                    ui,
                                    session,
                                    &obj.id,
                                    "material.side",
                                    "Render side",
                                    &["front", "back", "double"],
                                    0,
                                    false,
                                    local_frame,
                                    clip_start,
                                    viewport_rect,
                                    state,
                                    theme,
                                    actions,
                                );
                            }
                        }
                    },
                );
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn show_constraint_target_rows(
    ui: &mut Ui,
    session: &mut Session,
    obj_id: &str,
    targets: &[(String, String)],
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let mut labels: Vec<String> = Vec::with_capacity(targets.len() + 1);
    labels.push("None".to_owned());
    for (_, name) in targets {
        labels.push(name.clone());
    }
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();

    // Track to target dropdown
    let track_enabled = eval_prop_u32(
        lookup_session_prop(session, obj_id, "constraint.trackTo.enabled").as_ref(),
        local_frame,
        0,
    ) != 0;
    let track_target = eval_prop_str(
        lookup_session_prop(session, obj_id, "constraint.trackTo.targetId").as_ref(),
        local_frame,
        "",
    );
    let mut track_idx = if !track_enabled || track_target.is_empty() {
        0
    } else {
        targets
            .iter()
            .position(|(id, _)| id == &track_target)
            .map_or(0, |i| i + 1)
    };
    let track_changed = panel_property_row(
        ui,
        "Track to",
        true,
        state,
        theme,
        |_| {},
        |ui| {
            cm3_dropdown(
                ui,
                Id::unique((obj_id, "constraint_track_to_target")),
                &mut track_idx,
                &label_refs,
                false,
                viewport_rect,
                theme,
            )
        },
    );
    if track_changed {
        let started_tx = session.begin_transaction("set track to").is_ok();
        if track_idx == 0 {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.trackTo.enabled".to_owned(),
                value: json!(0),
            });
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.trackTo.targetId".to_owned(),
                value: json!(""),
            });
        } else {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.trackTo.enabled".to_owned(),
                value: json!(1),
            });
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.trackTo.targetId".to_owned(),
                value: json!(targets[track_idx - 1].0),
            });
        }
        if started_tx && session.history.is_in_transaction() {
            let _ = session.commit_transaction(true);
        }
    }
    if track_idx > 0 {
        show_session_number_prop(
            ui,
            session,
            obj_id,
            "constraint.trackTo.influence",
            "Track influence",
            1.0,
            ScrubConfig::new(0.0, 1.0, 0.05, 2),
            local_frame,
            clip_start,
            viewport_rect,
            state,
            theme,
            actions,
        );
    }

    // Follow target dropdown
    let follow_enabled = eval_prop_u32(
        lookup_session_prop(session, obj_id, "constraint.follow.enabled").as_ref(),
        local_frame,
        0,
    ) != 0;
    let follow_target = eval_prop_str(
        lookup_session_prop(session, obj_id, "constraint.follow.targetId").as_ref(),
        local_frame,
        "",
    );
    let mut follow_idx = if !follow_enabled || follow_target.is_empty() {
        0
    } else {
        targets
            .iter()
            .position(|(id, _)| id == &follow_target)
            .map_or(0, |i| i + 1)
    };
    let follow_changed = panel_property_row(
        ui,
        "Follow",
        true,
        state,
        theme,
        |_| {},
        |ui| {
            cm3_dropdown(
                ui,
                Id::unique((obj_id, "constraint_follow_target")),
                &mut follow_idx,
                &label_refs,
                false,
                viewport_rect,
                theme,
            )
        },
    );
    if follow_changed {
        let started_tx = session.begin_transaction("set follow").is_ok();
        if follow_idx == 0 {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.follow.enabled".to_owned(),
                value: json!(0),
            });
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.follow.targetId".to_owned(),
                value: json!(""),
            });
        } else {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.follow.enabled".to_owned(),
                value: json!(1),
            });
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.follow.targetId".to_owned(),
                value: json!(targets[follow_idx - 1].0),
            });
        }
        if started_tx && session.history.is_in_transaction() {
            let _ = session.commit_transaction(true);
        }
    }
}

fn apply_paired_scalar_scrub(
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

#[allow(clippy::too_many_arguments)]
fn show_session_asset_prop(
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
    let prop_opt = lookup_session_prop(session, target_id, key);
    let cur_id = eval_prop_str(prop_opt.as_ref(), local_frame, "");

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
        id: "group".into(),
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
            desc: format!("Standard ({} pass)", desc.passes),
        });
    }

    // Zoidium plugin effects gated dynamically by enabled plugins
    for desc in zoidium_effect_catalog() {
        let allowed = desc
            .plugin_id
            .is_none_or(|pid| session.plugins.is_enabled(pid));
        if allowed {
            let pid = desc.plugin_id.unwrap_or("zoidium");
            let plugin_label = session
                .plugins
                .get(pid)
                .map_or("Extension", |p| p.name.as_str());
            items.push(FuzzyItem {
                id: format!("zoidium:{pid}/{}", desc.id),
                category: format!("Plugin: {}", effect_category_label(desc.category)),
                name: desc.display_name.to_owned(),
                desc: format!("from {plugin_label}"),
            });
        }
    }

    // Lightweight motion blur plugin effect when enabled
    if session.plugins.is_enabled("smart-motion-blur-lite") {
        items.push(FuzzyItem {
            id: "zoidium:smart-motion-blur-lite/zsmb".into(),
            category: "Plugin: Blur & Sharpen".into(),
            name: "Smart Motion Blur Lite".into(),
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
    ui.input(|i| {
        if (i.pointer.primary_pressed() || i.pointer.secondary_pressed())
            && let Some(pos) = i.pointer.interact_pos()
        {
            if top_rect.contains(pos) {
                state.effects_focus_bottom = false;
            } else if bot_rect.contains(pos) {
                state.effects_focus_bottom = true;
            }
        }
    });
    let top_inner = top_rect.shrink(1.0);
    let bot_inner = bot_rect.shrink(1.0);
    ui.painter()
        .rect_filled(top_rect, CornerRadius::ZERO, BG_PANEL);
    ui.painter()
        .rect_filled(bot_rect, CornerRadius::ZERO, BG_PANEL);

    show_split_pane(
        ui,
        Id::unique("effects_40_60_split_p0"),
        top_inner,
        |ui, _top_rect| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;
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
                        commit_inspector_tx_if_open(session, state);
                        let display = items.iter().find(|i| i.id == kind).map(|i| i.name.clone());
                        let started_tx = kind == "customshader"
                            && session.begin_transaction("add custom shader").is_ok();
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
                        if started_tx && session.history.is_in_transaction() {
                            let _ = session.commit_transaction(true);
                        }
                        state.add_effect_open = false;
                    }
                    FuzzyPickerEvent::None => {}
                }
                return;
            }

            // 28px #222222 header bar (27px body + 1px #242424 bottom border), 7x7 framed square glyph,
            // 16px #cccccc "Effects" title, and right-aligned `+`
            let (hdr_rect, hdr_resp) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), Sense::click());
            ui.painter()
                .rect_filled(hdr_rect, CornerRadius::ZERO, BG_TITLE);
            let hdr_div_y = (hdr_rect.bottom() - 1.0).round();
            ui.painter().rect_filled(
                Rect::from_min_max(
                    Pos2::new(hdr_rect.left(), hdr_div_y),
                    Pos2::new(hdr_rect.right(), hdr_div_y + 1.0),
                ),
                CornerRadius::ZERO,
                BORDER_ROW_SOFT,
            );
            paint_cm3_tree_box(
                ui.painter(),
                Pos2::new(hdr_rect.left() + 10.5, hdr_rect.center().y + 1.0),
                state.effects_list_open,
                if hdr_resp.hovered() {
                    TEXT_BRIGHT
                } else {
                    TEXT_MUTED
                },
            );
            ui.painter().text(
                Pos2::new(hdr_rect.left() + 21.0, hdr_rect.center().y),
                Align2::LEFT_CENTER,
                "Effects",
                theme.font(FONT_BODY),
                TEXT_PRIMARY,
            );
            let add_icon_rect = Rect::from_min_size(
                Pos2::new(hdr_rect.right() - 24.0, (hdr_rect.center().y - 8.0).round()),
                Vec2::splat(15.0),
            );
            let add_rect = add_icon_rect.expand(3.0);
            let add_resp = ui
                .interact(add_rect, Id::unique("effects_add_btn"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text("Add effect");
            if add_resp.hovered() {
                ui.painter().rect_stroke(
                    add_rect,
                    CornerRadius::ZERO,
                    Stroke::new(2.0, theme.accent()),
                    StrokeKind::Inside,
                );
            }
            paint_cm3_icon(
                ui.painter(),
                add_icon_rect,
                Cm3Icon::Add,
                if add_resp.hovered() {
                    TEXT_BRIGHT
                } else {
                    TEXT_PRIMARY
                },
            );
            if add_resp.clicked() {
                state.add_effect_open = true;
                state.add_effect_picker = FuzzyPickerState {
                    just_opened: true,
                    ..FuzzyPickerState::default()
                };
            } else if hdr_resp.clicked() {
                state.effects_list_open = !state.effects_list_open;
            }

            if !state.effects_list_open {
                return;
            }

            // Effect stack rows
            ScrollArea::vertical()
                .id_salt("effects_list_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = Vec2::ZERO;
                    if clip.effects.is_empty() {
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
                            ui.allocate_exact_size(Vec2::new(row_w, 27.0), Sense::click_and_drag());
                        let pointer_in_row = ui
                            .input(|i| i.pointer.hover_pos().is_some_and(|p| row_rect.contains(p)));

                        if selected {
                            ui.painter()
                                .rect_filled(row_rect, CornerRadius::ZERO, theme.accent());
                        } else if row_resp.hovered() || pointer_in_row {
                            ui.painter()
                                .rect_filled(row_rect, CornerRadius::ZERO, BG_HOVER);
                        }
                        ui.painter().line_segment(
                            [
                                Pos2::new(row_rect.left(), row_rect.bottom() - 0.5),
                                Pos2::new(row_rect.right(), row_rect.bottom() - 0.5),
                            ],
                            Stroke::new(1.0, BORDER_ROW_SOFT),
                        );

                        // Label or Inline Rename input (35px left indent; Reset/Dup/Remove live in Inspector header)
                        let label_left = row_rect.left() + 35.0;
                        let label_right = (row_rect.right() - 8.0).max(label_left);
                        let label_rect = Rect::from_min_max(
                            Pos2::new(label_left, row_rect.top()),
                            Pos2::new(label_right, row_rect.bottom()),
                        );

                        let is_renaming = state
                            .rename_state
                            .as_ref()
                            .is_some_and(|r| r.target_id == fx.id);
                        if is_renaming {
                            if let Some(rename) = state.rename_state.as_mut() {
                                let edit_id = Id::unique(("fx_rename", &fx.id));
                                let mut commit = false;
                                let mut cancel = false;
                                ui.input(|i| {
                                    if i.key_pressed(Key::Enter) {
                                        commit = true;
                                    } else if i.key_pressed(Key::Escape) {
                                        cancel = true;
                                    }
                                });
                                let mut rename_ui =
                                    ui.new_child(egui::UiBuilder::new().max_rect(label_rect));
                                let resp = Frame::NONE
                                    .fill(BG_INPUT)
                                    .stroke(Stroke::new(1.0, theme.accent()))
                                    .inner_margin(egui::Margin::symmetric(3, 1))
                                    .show(&mut rename_ui, |ui| {
                                        ui.add(
                                            TextEdit::singleline(&mut rename.buffer)
                                                .id(edit_id)
                                                .frame(Frame::NONE)
                                                .font(theme.font(FONT_BODY))
                                                .text_color(theme.value())
                                                .desired_width(label_rect.width()),
                                        )
                                    })
                                    .inner;
                                if rename.just_started {
                                    resp.request_focus();
                                    rename.just_started = false;
                                }
                                if cancel {
                                    state.rename_state = None;
                                } else if commit || resp.lost_focus() {
                                    let trimmed = rename.buffer.trim().to_owned();
                                    if !trimmed.is_empty() && trimmed != fx.name {
                                        let _ = session.execute(Command::SetPropertyValue {
                                            target_id: fx.id.clone(),
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
                        }

                        if row_resp.clicked() {
                            commit_inspector_tx_if_open(session, state);
                            let shift = ui.input(|i| i.modifiers.shift);
                            if let Err(err) =
                                session.selection.select_effect(&fx.id, &fx.kind, shift)
                            {
                                actions.push(PanelAction::StatusMessage(err.to_string()));
                            }
                        }

                        // Right-click inline rename (Spec 01 §4.7)
                        if row_resp.secondary_clicked() {
                            state.rename_state = Some(InlineRenameState {
                                target_id: fx.id.clone(),
                                buffer: fx.name.clone(),
                                just_started: true,
                            });
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
                        if let Some((fx_id, mut target_idx)) = reorder_req {
                            if let Some(src_idx) = clip.effects.iter().position(|e| e.id == fx_id)
                                && src_idx < target_idx
                            {
                                target_idx -= 1;
                            }
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
    // Crisp single 1px #333333 border around top pane (layout.rs paints the focus accent only when SideInspector is focused)
    paint_crisp_panel_border(ui.painter(), top_rect, BORDER_PANEL);
    // Re-read clip snapshot after pane 0 mutations (AddEffect, duplicate, reorder, rename)
    // so the inspector reflects the newly selected effect immediately.
    let clip_for_inspector = find_clip_clone(session, &clip_id).unwrap_or(clip);
    show_split_pane(
        ui,
        Id::unique("effects_40_60_split_p1"),
        bot_inner,
        |ui, bot_rect| {
            // Pane 1 (60%): Dynamic Effect Parameter Inspector
            show_effect_inspector(
                ui,
                bot_rect,
                viewport_rect,
                &clip_for_inspector,
                session,
                theme,
                state,
                actions,
            );
        },
    );
    // Crisp single 1px #333333 border around bottom pane (layout.rs paints the focus accent only when SideInspector is focused)
    paint_crisp_panel_border(ui.painter(), bot_rect, BORDER_PANEL);
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
        empty_panel_message(ui, "select an object", theme);
        return;
    };
    let Some(fx) = clip.effects.iter().find(|e| e.id == fx_id).cloned() else {
        empty_panel_message(ui, "select an object", theme);
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
            ui.spacing_mut().item_spacing.y = 0.0;
            let (root_open, hdr_act) = property_header_with_actions(
                ui,
                Id::unique(("fx_root_hdr", &fx.id)),
                &fx.name,
                state.prop_columns.label_ratio,
                theme,
            );
            if hdr_act.reset {
                let _ = session.execute(Command::ResetPropertyList {
                    target_id: fx.id.clone(),
                });
            }
            if hdr_act.duplicate {
                let _ = session.execute(Command::DuplicateEffect {
                    clip_id: clip.id.clone(),
                    effect_id: fx.id.clone(),
                });
            }
            if hdr_act.remove {
                commit_inspector_tx_if_open(session, state);
                let _ = session.execute(Command::DeleteEffect {
                    clip_id: clip.id.clone(),
                    effect_id: fx.id.clone(),
                });
                return;
            }
            if !root_open {
                return;
            }
            state.inspector_root_indent = true;

            // Always show Enabled toggle
            show_session_option_prop(
                ui,
                session,
                &fx.id,
                "enabled",
                "Enabled",
                &["off", "on"],
                1,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );

            // Effect Group dedicated inspector
            if fx.kind == "group" {
                panel_property_row(
                    ui,
                    "Group type",
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        ui.label(
                            egui::RichText::new("Effect Group")
                                .font(theme.font(FONT_BODY))
                                .color(TEXT_MUTED),
                        );
                    },
                );
                let extra_keys: Vec<(String, Value)> = fx
                    .properties
                    .iter()
                    .filter(|(k, _)| k.as_str() != "enabled")
                    .filter_map(|(k, p)| {
                        let val = p
                            .keyframes
                            .first()
                            .map_or_else(|| p.default_value.clone(), |kf| kf.value.clone());
                        if val.is_null() {
                            None
                        } else {
                            Some((k.clone(), val))
                        }
                    })
                    .collect();
                for (key, val) in extra_keys {
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
                return;
            }

            // Custom Shader specialized inspector + Shader+ controls
            if fx.kind == "customshader" {
                let cur_code = eval_prop_str(
                    fx.properties.get("fragShader"),
                    local_frame,
                    DEFAULT_CUSTOM_SHADER_GLSL,
                );
                let open_shader = panel_property_row(
                    ui,
                    "Fragment shader",
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        let mut clicked = false;
                        ui.horizontal(|ui| {
                            if cm3_icon_button(
                                ui,
                                Cm3Icon::Code,
                                17.0,
                                theme.value(),
                                "Open shader code editor",
                                theme,
                            ) || cm3_prop_button(
                                ui,
                                "Edit shader",
                                ButtonVariant::Normal,
                                false,
                                theme,
                            )
                            .clicked()
                            {
                                clicked = true;
                            }
                        });
                        clicked
                    },
                );
                if open_shader {
                    state.shader_editor = Some(ShaderEditorTarget {
                        effect_id: fx.id.clone(),
                        property: "fragShader".to_owned(),
                        effect_name: fx.name.clone(),
                        code: cur_code.clone(),
                        font_px: 14.0,
                        error_message: None,
                    });
                    let spec = FloatingWindowSpec::new("shader_editor", "Shader Editor")
                        .with_subtitle(fx.name.clone())
                        .with_default_size(Vec2::new(560.0, 440.0));
                    state.floating_windows.open_or_focus(&spec, viewport_rect);
                }

                // Shader+ Custom Property builder & live uniform list
                let shader_plus = session.plugins.is_enabled("shader-plus");
                let sec_title = if shader_plus {
                    "Custom Properties (Shader+)"
                } else {
                    "Custom Properties"
                };
                state.inspector_root_indent = false;
                cm3_collapsible_section(
                    ui,
                    Id::unique("shader_custom_props"),
                    sec_title,
                    true,
                    theme,
                    |ui| {
                        let prop_kinds: &[&str] = if shader_plus {
                            &[
                                "Number",
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
                        Frame::NONE
                            .inner_margin(egui::Margin::symmetric(6, 4))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 4.0;
                                    let name_w = (ui.available_width() - 148.0).clamp(56.0, 110.0);
                                    Frame::NONE
                                        .fill(BG_INPUT)
                                        .stroke(Stroke::new(1.0, BORDER_CONTROL))
                                        .inner_margin(egui::Margin::symmetric(4, 2))
                                        .show(ui, |ui| {
                                            ui.add(
                                                TextEdit::singleline(
                                                    &mut state.new_shader_prop_name,
                                                )
                                                .frame(Frame::NONE)
                                                .font(theme.font(FONT_SMALL))
                                                .text_color(theme.value())
                                                .desired_width(name_w),
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
                                            let prop_key = if name.starts_with("u_") {
                                                name.to_owned()
                                            } else {
                                                format!("u_{name}")
                                            };
                                            let kind_idx = state.new_shader_prop_kind;
                                            let default_val = match kind_idx {
                                                1 => json!([0.0, 0.0]),
                                                2 => json!([0.0, 0.0, 0.0]),
                                                3 => json!([1.0, 1.0, 1.0]),
                                                4 => json!([1.0, 1.0, 1.0, 1.0]),
                                                5 => json!(true),
                                                6 => json!(0.0),
                                                7 => json!(1),
                                                _ => json!(0.5),
                                            };
                                            state.shader_prop_kinds.insert(
                                                (fx.id.clone(), prop_key.clone()),
                                                kind_idx,
                                            );
                                            let _ = session.execute(Command::SetPropertyValue {
                                                target_id: fx.id.clone(),
                                                property: prop_key,
                                                value: default_val,
                                            });
                                        }
                                    }
                                });
                            });

                        let custom_keys: Vec<(String, Value)> = fx
                            .properties
                            .iter()
                            .filter(|(k, _)| k.starts_with("u_"))
                            .filter_map(|(k, p)| {
                                let val = p
                                    .keyframes
                                    .first()
                                    .map_or_else(|| p.default_value.clone(), |kf| kf.value.clone());
                                if val.is_null() {
                                    None
                                } else {
                                    Some((k.clone(), val))
                                }
                            })
                            .collect();

                        for (key, val) in custom_keys {
                            let label = key.strip_prefix("u_").unwrap_or(&key);
                            let kind_hint = state
                                .shader_prop_kinds
                                .get(&(fx.id.clone(), key.clone()))
                                .copied();
                            render_custom_shader_uniform(
                                ui,
                                session,
                                &fx.id,
                                &key,
                                label,
                                &val,
                                kind_hint,
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
                        base_id,
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
                    .filter_map(|(k, p)| {
                        let val = p
                            .keyframes
                            .first()
                            .map_or_else(|| p.default_value.clone(), |kf| kf.value.clone());
                        if val.is_null() {
                            None
                        } else {
                            Some((k.clone(), val))
                        }
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

fn template_option_labels(effect_kind: &str, key: &str) -> &'static [&'static str] {
    match (effect_kind, key) {
        ("mirror", "mode") => &[
            "left",
            "right",
            "top",
            "bottom",
            "left + top",
            "right + top",
            "left + bottom",
            "right + bottom",
        ],
        ("static", "blending") => &["none", "add", "subtract", "multiply"],
        ("sglow", "combine") => &["multiply", "add", "screen", "difference", "overlay"],
        ("sglow", "quality") => &["low", "medium", "high"],
        ("sglow", "show") => &["result", "glow only", "threshold"],
        ("sglow", "edgeMode") => &["transparent", "clamp", "reflect"],
        ("sglow", "opacity") => &["normal", "affect alpha"],
        ("sglow", "blurEngine") => &["gaussian", "box", "dual kawase"],
        ("sglow", "timeMode") => &["sequence", "manual"],
        ("echo", "operator") => &[
            "add",
            "maximum",
            "minimum",
            "screen",
            "composite in front",
            "composite in back",
            "blend",
        ],
        ("ascii", "colorMode") => &["band", "footage", "duotone", "mono", "invert"],
        ("ascii", "fontFamily") => &["monospace", "sans", "serif"],
        ("ascii", "charset") => &["standard", "blocks", "dense", "minimal"],
        ("datamosh", "motionFamily") => &[
            "block",
            "horizontal",
            "vertical",
            "zoom",
            "rotate",
            "wave",
            "swap",
            "random",
        ],
        ("datamosh", "lookMode") => &["classic", "smear", "echo"],
        ("smartmotionblur", "view") => &["result", "motion vectors", "confidence"],
        ("smartmotionblur", "analysisFrames") => &["prev & next", "prev", "next"],
        ("smartmotionblur", "quality") => &["low", "medium", "high"],
        ("shutter", "mode") => &["linear", "double sided", "radial"],
        ("mask", "mode") => &["rectangle", "ellipse"],
        ("shake", "shake.mode") | ("shake", "mode") => &["rotate", "move"],
        ("overlay", "repeat") => &["off", "on"],
        ("duplicate", "repeat") => &["tile", "reflect"],
        ("timeoffset", "units") => &["frames", "seconds"],
        ("timeremap", "frameBlending") => &["off", "mix", "optical flow"],
        (_, "wrap") => &["clamp", "tile", "reflect"],
        (_, "mask") => &["result", "mask"],
        (_, "gradientType") => &["linear", "radial", "angular", "reflected", "diamond"],
        (_, "direction") => &["vertical", "horizontal"],
        (_, "affect") => &["shadows", "highlights"],
        (_, "cameraType") => &["perspective", "orthographic"],
        _ => &["off", "on"],
    }
}

fn template_number_scrub_config(effect_kind: &str, key: &str) -> ScrubConfig {
    // Angle parameters stored in radians, displayed & scrubbed in degrees
    if matches!(key, "rotation" | "angle")
        || (effect_kind == "directionalblur" && key == "direction")
    {
        return ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 1)
            .with_scale_factor(std::f64::consts::PI / 180.0);
    }

    match (effect_kind, key) {
        (
            _,
            "opacity"
            | "sourceOpacity"
            | "affectAlpha"
            | "glowFromAlpha"
            | "glowUnderSource"
            | "lightBackground"
            | "threshold"
            | "blendThreshold"
            | "soften"
            | "blendSoften"
            | "feather"
            | "covered"
            | "fade"
            | "decay"
            | "dither"
            | "density"
            | "splitBalance"
            | "vignetteStrength"
            | "vignetteRadius"
            | "grain"
            | "diffusion"
            | "chromaticAberration"
            | "trackingJitter"
            | "chromaBleed"
            | "noise"
            | "headSwitch"
            | "scanlines"
            | "atmosphereDetail"
            | "blurAmount",
        ) => ScrubConfig::new(0.0, 1.0, 0.01, 3),
        ("crop", "left" | "top" | "right" | "bottom") => ScrubConfig::new(0.0, 1.0, 0.01, 3),
        ("posterize", "regions") => ScrubConfig::new(2.0, 256.0, 1.0, 0),
        ("echo", "numberOfEchoes") | ("radialblurspin", "samples") => {
            ScrubConfig::new(1.0, 64.0, 1.0, 0)
        }
        ("posterizetime", "frameRate") => ScrubConfig::new(1.0, 240.0, 1.0, 1),
        ("jpegdamage", "quality") => ScrubConfig::new(1.0, 100.0, 1.0, 0),
        ("jpegdamage", "blockSize")
        | ("datamosh", "blockSize")
        | ("tracery", "gridSize")
        | ("ascii", "cellSize") => ScrubConfig::new(2.0, 256.0, 1.0, 0),
        ("smartmotionblur", "motionSensitivity" | "vectorSmoothness") => {
            ScrubConfig::new(0.0, 100.0, 1.0, 1)
        }
        ("smartmotionblur", "maxDisplacement") => ScrubConfig::new(1.0, 2048.0, 1.0, 0),
        ("sglow", "glowWidth") => ScrubConfig::new(0.0, 1000.0, 1.0, 1),
        ("fisheye", "amount") | ("pulse", "amplitude") | ("shake", "zoom") => {
            ScrubConfig::new(-1000.0, 1000.0, 1.0, 1)
        }
        ("brightnesscontrast", "brightness")
        | ("exposure", "exposure")
        | ("magiclooks", "exposure" | "temperature" | "tint") => {
            ScrubConfig::new(-10.0, 10.0, 0.05, 2)
        }
        ("brightnesscontrast", "contrast")
        | ("grayscale", "saturation")
        | ("magiclooks", "contrast" | "saturation" | "strength")
        | ("bloom", "amount" | "strength" | "radius")
        | ("anamorphicflare", "amount" | "strength" | "radius") => {
            ScrubConfig::new(0.0, 10.0, 0.05, 2)
        }
        (_, "seed" | "atmosphereSeed") => ScrubConfig::new(0.0, 100_000.0, 1.0, 2),
        (_, "delta" | "blur" | "softness") => ScrubConfig::new(0.0, 500.0, 0.1, 2),
        (_, "time") => ScrubConfig::new(-100_000.0, 100_000.0, 0.05, 3),
        _ => ScrubConfig::new(-10_000.0, 10_000.0, 0.05, 3),
    }
}

#[allow(clippy::too_many_arguments)]
fn render_template_param(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    effect_kind: &str,
    key: &str,
    default_val: &PropertyValue,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let display_label = humanize_prop_label(key);
    let lbl = display_label.as_str();
    match default_val {
        PropertyValue::Number(def) => {
            let cfg = template_number_scrub_config(effect_kind, key);
            show_session_number_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                *def,
                cfg,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::Vec2(def) => {
            show_session_vec2_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                *def,
                ScrubConfig::new(-10_000.0, 10_000.0, 0.05, 3),
                key.contains("scale") || key.contains("size") || key.contains("multiplier"),
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::Vec3(def) => {
            let cfg = if key == "cameraRotation" || key == "imageRotation" {
                ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 1)
                    .with_scale_factor(std::f64::consts::PI / 180.0)
            } else {
                ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 3)
            };
            show_session_vec3_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                *def,
                cfg,
                false,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::Vec4(def) => {
            let (cfg, animatable) = match key {
                "shake.amplitude" | "shake.phase" => {
                    (ScrubConfig::new(-10_000.0, 10_000.0, 0.5, 1), false)
                }
                "shake.frequency" | "shake.noise" => (ScrubConfig::new(0.0, 100.0, 0.05, 2), false),
                "shake.smooth" => (ScrubConfig::new(0.0, 10.0, 1.0, 0), false),
                _ => (ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 2), true),
            };
            show_session_vec4_prop_ex(
                ui,
                session,
                target_id,
                key,
                lbl,
                *def,
                cfg,
                animatable,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::Color(def) => {
            show_session_color_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                *def,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::ColorRgba(def) => {
            show_session_color_rgba_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
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
            let opts = template_option_labels(effect_kind, key);
            let static_option = matches!(
                key,
                "gradientType"
                    | "wrap"
                    | "cameraType"
                    | "quality"
                    | "blurEngine"
                    | "edgeMode"
                    | "shake.mode"
                    | "mode"
                    | "combine"
                    | "show"
                    | "timeMode"
                    | "operator"
                    | "colorMode"
                    | "fontFamily"
                    | "charset"
                    | "motionFamily"
                    | "lookMode"
                    | "view"
                    | "analysisFrames"
                    | "units"
                    | "frameBlending"
                    | "direction"
                    | "affect"
            );
            show_session_option_prop_ex(
                ui,
                session,
                target_id,
                key,
                lbl,
                opts,
                *def,
                !static_option,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
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
            let changed = panel_property_row(
                ui,
                lbl,
                true,
                state,
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
            touch_inspector_tx_owner(state, target_id, key);
            let prop_opt = lookup_session_prop(session, target_id, key);
            let animated = prop_opt.as_ref().is_some_and(|p| p.animated);
            let mut stops = eval_prop_value(prop_opt.as_ref(), local_frame)
                .and_then(|v| serde_json::from_value::<Vec<GradientStop>>(v).ok())
                .unwrap_or_else(|| def.clone());
            let ev = panel_property_row(
                ui,
                lbl,
                true,
                state,
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
            let owns_grad_tx = owns_inspector_tx(state, target_id, key);
            if ev.cancelled {
                if owns_grad_tx {
                    if session.history.is_in_transaction() {
                        let _ = session.cancel_transaction();
                    }
                    clear_inspector_tx_state(state);
                }
            } else {
                if ev.changed && !ev.committed && !owns_grad_tx {
                    close_existing_tx_before_start(session, state);
                    begin_inspector_tx(session, state, target_id, key, format!("edit {key}"));
                }
                let owns_grad_tx = owns_inspector_tx(state, target_id, key);
                if ev.changed
                    && let Ok(v) = serde_json::to_value(&stops)
                {
                    let exec_grad = |session: &mut Session| {
                        let cmd = if animated {
                            Command::SetKeyframe {
                                target_id: target_id.to_owned(),
                                property: key.to_owned(),
                                frame: resolve_kf_write_frame(
                                    session,
                                    prop_opt.as_ref(),
                                    local_frame,
                                ),
                                value: v,
                                tween: None,
                            }
                        } else {
                            Command::SetPropertyValue {
                                target_id: target_id.to_owned(),
                                property: key.to_owned(),
                                value: v,
                            }
                        };
                        let _ = session.execute(cmd);
                    };
                    if ev.committed && !owns_grad_tx {
                        run_atomic_commit_outside_foreign_tx(session, state, exec_grad);
                    } else {
                        exec_grad(session);
                        if owns_grad_tx {
                            state.inspector_tx_has_edits = true;
                        }
                    }
                }
                if ev.committed && owns_grad_tx {
                    if session.history.is_in_transaction() {
                        let _ = session.commit_transaction(true);
                    }
                    clear_inspector_tx_state(state);
                }
            }
        }
        PropertyValue::Curve(def) => {
            show_session_curve_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                def,
                local_frame,
                state,
                theme,
            );
        }
        PropertyValue::Text(def) | PropertyValue::List(def) | PropertyValue::Shader(def) => {
            show_session_text_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                def,
                local_frame,
                state,
                theme,
            );
        }
        PropertyValue::Asset(def) => {
            let is_lut = key == "lut";
            let prop_opt = lookup_session_prop(session, target_id, key);
            let cur_id =
                eval_prop_str(prop_opt.as_ref(), local_frame, def.as_deref().unwrap_or(""));

            // Build sorted project asset list for dropdown selection
            let mut assets: Vec<(String, String)> = session
                .project
                .assets
                .values()
                .map(|a| (a.id.clone(), a.filename.clone()))
                .collect();
            assets.sort_by(|a, b| {
                if is_lut {
                    let a_cube = a.1.to_ascii_lowercase().ends_with(".cube");
                    let b_cube = b.1.to_ascii_lowercase().ends_with(".cube");
                    b_cube.cmp(&a_cube).then_with(|| a.1.cmp(&b.1))
                } else {
                    a.1.cmp(&b.1)
                }
            });

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
                            Id::unique((target_id, key, "asset_dd")),
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
                            if is_lut {
                                "Load .cube LUT file..."
                            } else {
                                "Choose asset file..."
                            },
                            theme,
                        ) {
                            actions.push(PanelAction::PickAssetForProperty {
                                target_id: target_id.to_owned(),
                                property: key.to_owned(),
                                is_lut,
                            });
                        }
                        if !cur_id.is_empty()
                            && cm3_icon_button(
                                ui,
                                Cm3Icon::Remove,
                                17.0,
                                TEXT_PRIMARY,
                                "Clear asset",
                                theme,
                            )
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
        PropertyValue::Bool(def) => {
            show_session_option_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                &["off", "on"],
                u32::from(*def),
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::Json(_) => {
            let btn_label = match effect_kind {
                "nodeeffect" => "Open Node Editor",
                "magiclooks" => "Open Color Looks Editor",
                _ => "Edit JSON",
            };
            panel_property_row(
                ui,
                lbl,
                true,
                state,
                theme,
                |_| {},
                |ui| {
                    if cm3_prop_button(ui, btn_label, ButtonVariant::Normal, false, theme).clicked()
                    {
                        actions.push(PanelAction::StatusMessage(format!(
                            "{btn_label}: configured via effect properties"
                        )));
                    }
                },
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_custom_shader_uniform(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    key: &str,
    label: &str,
    val: &Value,
    kind_hint: Option<usize>,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let lower = label.to_ascii_lowercase();
    let mut delete_uniform = false;

    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_max_width((ui.available_width() - 24.0).max(80.0));
            match val {
                Value::Bool(b) => {
                    show_session_option_prop(
                        ui,
                        session,
                        target_id,
                        key,
                        label,
                        &["off", "on"],
                        u32::from(*b),
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                Value::Number(n) => {
                    let is_toggle = kind_hint == Some(5);
                    let is_angle =
                        kind_hint == Some(6) || lower.contains("angle") || lower.contains("rot");
                    let is_seed = kind_hint == Some(7) || lower.contains("seed");
                    if is_toggle {
                        let def = u32::from(n.as_f64().unwrap_or(0.0) >= 0.5);
                        show_session_option_prop(
                            ui,
                            session,
                            target_id,
                            key,
                            label,
                            &["off", "on"],
                            def,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    } else if is_angle {
                        show_session_number_prop(
                            ui,
                            session,
                            target_id,
                            key,
                            label,
                            n.as_f64().unwrap_or(0.0),
                            ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 1)
                                .with_scale_factor(std::f64::consts::PI / 180.0),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    } else if is_seed {
                        show_session_number_prop(
                            ui,
                            session,
                            target_id,
                            key,
                            label,
                            n.as_f64().unwrap_or(1.0),
                            ScrubConfig::new(0.0, 100_000.0, 1.0, 0),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    } else {
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
                            actions,
                        );
                    }
                }
                Value::Array(arr) if arr.len() == 4 => {
                    let def = [
                        arr[0].as_f64().unwrap_or(1.0),
                        arr[1].as_f64().unwrap_or(1.0),
                        arr[2].as_f64().unwrap_or(1.0),
                        arr[3].as_f64().unwrap_or(1.0),
                    ];
                    show_session_color_rgba_prop(
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
                }
                Value::Array(arr)
                    if arr.len() == 3 && (kind_hint == Some(3) || lower.contains("color")) =>
                {
                    let def = [
                        arr[0].as_f64().unwrap_or(1.0),
                        arr[1].as_f64().unwrap_or(1.0),
                        arr[2].as_f64().unwrap_or(1.0),
                    ];
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
                }
                _ => {
                    render_dynamic_json_param(
                        ui,
                        session,
                        target_id,
                        key,
                        label,
                        val,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
            }
        });

        if cm3_icon_button(
            ui,
            Cm3Icon::Remove,
            15.0,
            TEXT_MUTED,
            "remove custom uniform",
            theme,
        ) {
            delete_uniform = true;
        }
    });

    if delete_uniform {
        state
            .shader_prop_kinds
            .remove(&(target_id.to_owned(), key.to_owned()));
        let _ = session.execute(Command::RemoveProperty {
            target_id: target_id.to_owned(),
            property: key.to_owned(),
        });
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
        Value::Bool(b) => {
            show_session_option_prop(
                ui,
                session,
                target_id,
                key,
                label,
                &["off", "on"],
                u32::from(*b),
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
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
                actions,
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
                actions,
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
                    actions,
                );
            }
        }
        Value::Array(arr) if arr.len() == 4 => {
            let def = [
                arr[0].as_f64().unwrap_or(0.0),
                arr[1].as_f64().unwrap_or(0.0),
                arr[2].as_f64().unwrap_or(0.0),
                arr[3].as_f64().unwrap_or(1.0),
            ];
            if label.to_ascii_lowercase().contains("color") {
                show_session_color_rgba_prop(
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
                show_session_vec4_prop(
                    ui,
                    session,
                    target_id,
                    key,
                    label,
                    def,
                    ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
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

fn container_label(c: ExportContainer) -> &'static str {
    match c {
        ExportContainer::Mp4 => "MP4 (.mp4)",
        ExportContainer::WebM => "WebM (.webm)",
        ExportContainer::Mkv => "MKV (.mkv)",
        ExportContainer::PngSequence => "PNG Sequence (.png)",
    }
}

fn video_codec_label(c: ExportVideoCodec) -> &'static str {
    match c {
        ExportVideoCodec::H264 => "H.264 (libx264)",
        ExportVideoCodec::H265 => "H.265 / HEVC (libx265)",
        ExportVideoCodec::Vp9 => "VP9 (libvpx-vp9)",
        ExportVideoCodec::Vp8 => "VP8 (legacy)",
        ExportVideoCodec::Av1 => "AV1 (libsvtav1)",
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

fn allowed_video_codecs(container: ExportContainer) -> &'static [ExportVideoCodec] {
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
    }
}

fn allowed_audio_codecs(container: ExportContainer) -> &'static [ExportAudioCodec] {
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
        ExportContainer::PngSequence => &[ExportAudioCodec::None],
    }
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

fn show_export_tab(
    ui: &mut Ui,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let mp4_enabled = session.plugins.is_enabled("mp4-export");
    let available_containers: &[ExportContainer] = if mp4_enabled {
        &ExportContainer::ALL
    } else {
        &[
            ExportContainer::WebM,
            ExportContainer::Mkv,
            ExportContainer::PngSequence,
        ]
    };

    // Auto-coerce container, video codec, audio codec, alpha, and CRF bounds
    if !available_containers.contains(&state.export_settings.container) {
        state.export_settings.container = available_containers[0];
    }
    let valid_vcodecs = allowed_video_codecs(state.export_settings.container);
    if !valid_vcodecs.contains(&state.export_settings.video_codec) {
        state.export_settings.video_codec = valid_vcodecs[0];
    }
    let valid_acodecs = allowed_audio_codecs(state.export_settings.container);
    if !valid_acodecs.contains(&state.export_settings.audio_codec) {
        state.export_settings.audio_codec = valid_acodecs[0];
    }
    let alpha_supported = state.export_settings.container == ExportContainer::PngSequence
        || (state.export_settings.video_codec == ExportVideoCodec::Vp9
            && state.export_settings.container != ExportContainer::Mp4);
    if !alpha_supported {
        state.export_settings.transparent_alpha = false;
    }
    let max_crf: u32 = match state.export_settings.video_codec {
        ExportVideoCodec::H264 | ExportVideoCodec::H265 => 51,
        _ => 63,
    };
    state.export_settings.crf = state.export_settings.crf.min(max_crf);

    ScrollArea::vertical()
        .id_salt("export_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;

            // Edge-to-edge 42px #222222 header bar ("Export project")
            panel_top_header(ui, "Export project", theme);

            // Landing screen (default None): 2 full-width 77px clickable #2a2a2a rows (76px body + 1px divider)
            let Some(active_mode) = state.export_view_mode else {
                for (mode, title, subtitle) in [
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
                ] {
                    let (row_rect, resp) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), 77.0),
                        Sense::click(),
                    );
                    let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
                    let fill = if resp.hovered() { BG_HOVER } else { BG_PANEL };
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

                    // Procedural vector icon on left: 54x40 at (+11, +18) (Device) or 52x42 at (+12, +17) (Frame)
                    if mode == ExportMode::Device {
                        let icon_rect = Rect::from_min_size(
                            Pos2::new(row_rect.left() + 11.0, (row_rect.top() + 18.0).round()),
                            Vec2::new(54.0, 40.0),
                        );
                        paint_export_desktop_icon(ui.painter(), icon_rect, TEXT_PRIMARY, fill);
                    } else {
                        let icon_rect = Rect::from_min_size(
                            Pos2::new(row_rect.left() + 12.0, (row_rect.top() + 17.0).round()),
                            Vec2::new(52.0, 42.0),
                        );
                        paint_export_frame_icon(ui.painter(), icon_rect, TEXT_PRIMARY, fill);
                    }

                    let text_left = row_rect.left() + 78.0;
                    let text_rect = Rect::from_min_max(
                        Pos2::new(text_left, row_rect.top() + 9.0),
                        Pos2::new(
                            (row_rect.right() - 10.0).max(text_left + 20.0),
                            row_rect.bottom() - 4.0,
                        ),
                    );
                    let mut text_ui = ui.new_child(egui::UiBuilder::new().max_rect(text_rect));
                    text_ui.spacing_mut().item_spacing.y = 3.0;
                    text_ui.label(
                        egui::RichText::new(title)
                            .font(theme.font(FONT_BODY))
                            .color(if resp.hovered() {
                                TEXT_BRIGHT
                            } else {
                                TEXT_PRIMARY
                            }),
                    );
                    text_ui.add(
                        egui::Label::new(
                            egui::RichText::new(subtitle)
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_OPTION_DESC),
                        )
                        .wrap(),
                    );

                    if resp.clicked() {
                        state.export_settings.mode = mode;
                        state.export_view_mode = Some(mode);
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
                                format!("{:.0}%", p * 100.0),
                                theme.font(FONT_SMALL),
                                TEXT_BRIGHT,
                            );
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
                }
                return;
            };

            state.export_settings.mode = active_mode;
            let is_device = active_mode == ExportMode::Device;
            let is_png_seq =
                is_device && state.export_settings.container == ExportContainer::PngSequence;
            let needs_ffmpeg = is_device && !is_png_seq;

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
                            state.prop_columns,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_dropdown(
                                    ui,
                                    Id::unique("exp_container"),
                                    &mut c_idx,
                                    &container_labels,
                                    false,
                                    viewport_rect,
                                    theme,
                                )
                            },
                        );
                        if changed {
                            let next = available_containers[c_idx];
                            state.export_settings.container = next;
                            let next_v = allowed_video_codecs(next);
                            if !next_v.contains(&state.export_settings.video_codec) {
                                state.export_settings.video_codec = next_v[0];
                            }
                            let next_a = allowed_audio_codecs(next);
                            if !next_a.contains(&state.export_settings.audio_codec) {
                                state.export_settings.audio_codec = next_a[0];
                            }
                            if next == ExportContainer::Mp4
                                || (next != ExportContainer::PngSequence
                                    && state.export_settings.video_codec != ExportVideoCodec::Vp9)
                            {
                                state.export_settings.transparent_alpha = false;
                            }
                        }

                        // Video codec, rate control, and audio controls are hidden for PNG Sequence
                        if state.export_settings.container != ExportContainer::PngSequence {
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
                                state.prop_columns,
                                theme,
                                |_| {},
                                |ui| {
                                    cm3_dropdown(
                                        ui,
                                        Id::unique("exp_vcodec"),
                                        &mut vc_idx,
                                        &v_labels,
                                        false,
                                        viewport_rect,
                                        theme,
                                    )
                                },
                            ) {
                                state.export_settings.video_codec = v_list[vc_idx];
                                let new_max = match state.export_settings.video_codec {
                                    ExportVideoCodec::H264 | ExportVideoCodec::H265 => 51,
                                    _ => 63,
                                };
                                state.export_settings.crf = state.export_settings.crf.min(new_max);
                                if state.export_settings.video_codec != ExportVideoCodec::Vp9 {
                                    state.export_settings.transparent_alpha = false;
                                }
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
                                state.prop_columns,
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
                                        "Quality (CRF)",
                                        false,
                                        state.prop_columns,
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
                                        "Bitrate (Mbps)",
                                        false,
                                        state.prop_columns,
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
                                        state.prop_columns,
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
                                        state.export_settings.cm3_preset =
                                            ExportCm3Preset::ALL[p_idx];
                                    }
                                }
                            }
                        }

                        // Transparent alpha (PNG Sequence or VP9 in WebM/MKV)
                        let show_alpha =
                            state.export_settings.container == ExportContainer::PngSequence
                                || (state.export_settings.video_codec == ExportVideoCodec::Vp9
                                    && state.export_settings.container != ExportContainer::Mp4);
                        if show_alpha {
                            let mut alpha_idx = usize::from(state.export_settings.transparent_alpha);
                            if cm3_property_row_aligned(
                                ui,
                                "Transparent alpha",
                                false,
                                state.prop_columns,
                                theme,
                                |_| {},
                                |ui| {
                                    cm3_dropdown(
                                        ui,
                                        Id::unique("exp_alpha"),
                                        &mut alpha_idx,
                                        &["off", "on"],
                                        false,
                                        viewport_rect,
                                        theme,
                                    )
                                },
                            ) {
                                state.export_settings.transparent_alpha = alpha_idx != 0;
                            }
                        }

                        // Audio controls (hidden for PNG Sequence)
                        if state.export_settings.container != ExportContainer::PngSequence {
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
                                state.prop_columns,
                                theme,
                                |_| {},
                                |ui| {
                                    cm3_dropdown(
                                        ui,
                                        Id::unique("exp_acodec"),
                                        &mut ac_idx,
                                        &a_labels,
                                        false,
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
                                    "Audio bitrate (kbps)",
                                    false,
                                    state.prop_columns,
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
                                    state.prop_columns,
                                    theme,
                                    |_| {},
                                    |ui| {
                                        cm3_dropdown(
                                            ui,
                                            Id::unique("exp_sample_rate"),
                                            &mut sr_idx,
                                            &RATE_LABELS,
                                            false,
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

                    // Resolution (both Device & Single Frame, "x" separator)
                    let mut res = [
                        f64::from(state.export_settings.width),
                        f64::from(state.export_settings.height),
                    ];
                    let ev = cm3_property_row_aligned(
                        ui,
                        "Resolution",
                        false,
                        state.prop_columns,
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
                        state.export_settings.height = res[1].round().clamp(16.0, 7680.0) as u32;
                    }

                    // Frame rate (0 decimals) & Range only apply to Device Video / PNG Sequence
                    if is_device {
                        let mut fps = state.export_settings.fps;
                        let ev = cm3_property_row_aligned(
                            ui,
                            "Frame rate",
                            false,
                            state.prop_columns,
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

                        let mut range = [
                            state.export_settings.start_frame,
                            state.export_settings.end_frame,
                        ];
                        let ev = cm3_property_row_aligned(
                            ui,
                            "Range (frames)",
                            false,
                            state.prop_columns,
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
                            format!("{:.0}%", p * 100.0),
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
                            if is_png_seq {
                                "Start PNG sequence render"
                            } else {
                                "Start video render"
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
                                .on_hover_text("FFmpeg is not available; install FFmpeg or export PNG Sequence / Single Frame.");
                        }
                    }
                });
        });
}

// ---------------------------------------------------------------------------
// Tab 7: Restore — Autosave & Project Snapshot Restore
// ---------------------------------------------------------------------------

fn show_restore_tab(
    ui: &mut Ui,
    session: &mut Session,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    ScrollArea::vertical()
        .id_salt("restore_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;
            panel_top_header(ui, "Restore", theme);

            let project_name = session.document().project_name.clone();
            let [rw, rh] = session.document().resolution;
            let fps = session.document().fps;
            let undo_len = session.history.undo_len();
            let redo_len = session.history.redo_len();

            Frame::NONE
                .inner_margin(egui::Margin::symmetric(10, 10))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 8.0;

                    ui.label(
                        egui::RichText::new("Project snapshots & autosave")
                            .font(theme.font(FONT_BODY))
                            .color(TEXT_STRONG),
                    );
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(
                                "Undo history snapshots are tracked automatically during editing. \
                                 Step back to a previous state or reset the workspace to the default starter project.",
                            )
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_OPTION_DESC),
                        )
                        .wrap(),
                    );

                    Frame::NONE
                        .fill(BG_TITLE)
                        .stroke(Stroke::new(1.0, BORDER_ROW_SOFT))
                        .inner_margin(egui::Margin::symmetric(8, 8))
                        .show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            ui.spacing_mut().item_spacing.y = 4.0;
                            ui.label(
                                egui::RichText::new(format!("Active project: {project_name}"))
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_PRIMARY),
                            );
                            ui.label(
                                egui::RichText::new(format!("Format: {rw}×{rh} @ {fps:.0} fps"))
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_MUTED),
                            );
                            ui.label(
                                egui::RichText::new(format!(
                                    "History states: {undo_len} undo · {redo_len} redo"
                                ))
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_MUTED),
                            );
                        });

                    if undo_len > 0
                        && cm3_prop_button(
                            ui,
                            "Restore previous snapshot (Undo)",
                            ButtonVariant::Normal,
                            true,
                            theme,
                        )
                        .clicked()
                    {
                        let _ = session.execute(Command::Undo);
                        actions.push(PanelAction::StatusMessage(
                            "Restored previous snapshot".to_owned(),
                        ));
                    }

                    if cm3_prop_button(
                        ui,
                        "Reset to default project",
                        ButtonVariant::Danger,
                        true,
                        theme,
                    )
                    .clicked()
                    {
                        actions.push(PanelAction::NewProject);
                    }
                });
        });
}

// ---------------------------------------------------------------------------
// Tab 9: Assets — Project Assets & Starter Templates
// ---------------------------------------------------------------------------

fn show_assets_tab(
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

            // Project assets bar
            let (bar_rect, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 27.0), Sense::hover());
            ui.painter()
                .rect_filled(bar_rect, CornerRadius::ZERO, BG_TITLE);
            ui.painter().line_segment(
                [bar_rect.left_bottom(), bar_rect.right_bottom()],
                Stroke::new(1.0, BORDER_ROW_SOFT),
            );
            ui.painter().text(
                Pos2::new(bar_rect.left() + 10.0, bar_rect.center().y),
                Align2::LEFT_CENTER,
                "Project assets",
                theme.font(FONT_SMALL),
                TEXT_STRONG,
            );
            let imp_rect = Rect::from_min_max(
                Pos2::new(
                    (bar_rect.right() - 78.0).max(bar_rect.left()),
                    bar_rect.top(),
                ),
                bar_rect.right_bottom(),
            );
            let imp_resp = ui
                .interact(imp_rect, Id::unique("assets_import_btn"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand);
            ui.painter().text(
                Pos2::new(bar_rect.right() - 10.0, bar_rect.center().y),
                Align2::RIGHT_CENTER,
                "+ Import",
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

            let project_assets: Vec<(String, String, &'static str, u64, bool)> = session
                .project
                .assets
                .values()
                .filter(|a| q.is_empty() || a.filename.to_ascii_lowercase().contains(&q))
                .map(|a| {
                    (
                        a.id.clone(),
                        a.filename.clone(),
                        clip_kind_for_asset(a.kind, &a.filename),
                        a.size,
                        a.missing,
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
                for (aid, fname, clip_kind, size, missing) in project_assets {
                    let (row_rect, resp) = ui
                        .allocate_exact_size(Vec2::new(ui.available_width(), 34.0), Sense::click());
                    let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
                    if resp.hovered() {
                        ui.painter()
                            .rect_filled(row_rect, CornerRadius::ZERO, BG_HOVER);
                    }
                    ui.painter().line_segment(
                        [row_rect.left_bottom(), row_rect.right_bottom()],
                        Stroke::new(1.0, BORDER_ROW_SOFT),
                    );
                    let meta = if missing {
                        "missing".to_owned()
                    } else if size > 0 {
                        format!("{clip_kind} · {} KiB", size.div_ceil(1024))
                    } else {
                        clip_kind.to_owned()
                    };
                    ui.painter().text(
                        Pos2::new(row_rect.left() + 10.0, row_rect.center().y),
                        Align2::LEFT_CENTER,
                        &fname,
                        theme.font(FONT_SMALL),
                        if missing { DANGER_TEXT } else { TEXT_PRIMARY },
                    );
                    ui.painter().text(
                        Pos2::new(row_rect.right() - 10.0, row_rect.center().y),
                        Align2::RIGHT_CENTER,
                        meta,
                        theme.font(FONT_COMPACT),
                        TEXT_MUTED,
                    );
                    if resp.clicked() {
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
                ui.painter().text(
                    Pos2::new(tx, row_rect.top() + 14.0),
                    Align2::LEFT_CENTER,
                    preset.title,
                    theme.font(FONT_SMALL),
                    if resp.hovered() {
                        TEXT_BRIGHT
                    } else {
                        TEXT_PRIMARY
                    },
                );
                ui.painter().text(
                    Pos2::new(tx, row_rect.top() + 31.0),
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
}

// ---------------------------------------------------------------------------
// Tab 7: Plugins (Zoidium Extension Manager, Spec 04 §5.1)
// ---------------------------------------------------------------------------

fn zoidium_plugin_summary<'a>(id: &str, fallback: &'a str) -> &'a str {
    match id {
        // Utilities (exact Zoidium 08_plugins.png summaries)
        "easing-plus" => "Extended Bezier Easing",
        "expressions-plus" => "Modern expressions and property links",
        "shader-plus" => "Custom shader property editors",
        "player-plus" => "Playback Quality Controls",
        "asset-manager" => "Project panel for every asset",
        "precomp-plus" => "Pre-compose and nested compositions",
        "gif-support" => "Animated GIF image assets",
        "graph-plus" => "Rebuilt keyframe graph editor",
        "motion-path" => "Editable motion paths in the preview",
        "mp4-export" => "H.264 MP4 video export",
        // Scene Extensions
        "light-plus" => "Spot, point, directional & hemisphere lights",
        "geometry-plus" => "Rounded boxes & procedural 3D shapes",
        "repeater" => "Repeaters, twist, warp & Voronoi fracture",
        "effector-plus" => "Legacy deformer compatibility loader",
        "material-plus" => "Node, PBR+, Matcap & Image+ materials",
        "text-plus" => "2D/3D text animators & advanced bevels",
        "expresstext-plus" => "Per-letter 3D text effector stack",
        "particles-plus" => "54 procedural particle sprites",
        "trapcode-suite" => "3D particles, lattice & studio lights",
        "optical-flares" => "3D projected lens flares with occlusion",
        "camera-plus" => "Physical camera, DoF & 3D tracks",
        "shape-plus" => "Animated vector layers & pen masks",
        "constraints" => "Track To & Follow 3D constraints",
        // Adjustment Extensions
        "native-fx" => "Temporal FX, curves, drop shadow & motion blur",
        "openzoid-legacy" => "JPEG Damage, VHS, Datamosh & ASCII FX",
        "magic-looks" => "Color grading, film looks & 3D LUTs",
        "node-effect" => "Visual GPU shader node graph",
        "smart-motion-blur-lite" => "Fast optical-flow motion blur",
        "sglow-hq-v2-preview" => "Multi-scale glow with RGB spread",
        // Experimental
        "layer-input" => "Cross-track texture sampling",
        _ => fallback,
    }
}

fn is_cm3_compatible_plugin(plugin_id: &str) -> bool {
    matches!(
        plugin_id,
        "easing-plus"
            | "shader-plus"
            | "player-plus"
            | "asset-manager"
            | "graph-plus"
            | "motion-path"
            | "mp4-export"
            | "node-effect"
            | "effector-plus"
            | "particles-plus"
    )
}

fn paint_outlined_plugin_badge_at(
    painter: &Painter,
    pos: Pos2,
    text: &str,
    bg: Color32,
    border: Color32,
    fg: Color32,
    theme: &Cm3Theme,
) -> f32 {
    let galley = painter.layout_no_wrap(text.to_owned(), theme.font(11.5), fg);
    let size = Vec2::new((galley.size().x + 13.0).round(), 16.0);
    let rect = Rect::from_min_size(Pos2::new(pos.x.round(), pos.y.round()), size);
    painter.rect_filled(rect, CornerRadius::ZERO, bg);
    paint_crisp_panel_border(painter, rect, border);
    painter.galley(
        Pos2::new(
            (rect.center().x - galley.size().x * 0.5).round(),
            rect.center().y - galley.size().y * 0.5 - 0.5,
        ),
        galley,
        fg,
    );
    size.x
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PillToggleState {
    Off,
    Partial,
    On,
}

fn paint_pill_toggle(
    ui: &mut Ui,
    toggle_state: PillToggleState,
    locked: bool,
    theme: &Cm3Theme,
) -> egui::Response {
    // 36x20 rounded pill switch matching Zoidium 08_plugins.png
    // ON: #384668 track + #506084 rim + #e4e4e4 right thumb (14x14)
    // PARTIAL: #333333 track + #6d6d6d rim + 18x4 #9a9a9a dash
    // OFF: #202020 track + #4d4d4d rim + #9a9a9a left thumb (14x14)
    let (raw_rect, resp) = ui.allocate_exact_size(Vec2::new(36.0, 20.0), Sense::click());
    let rect = Rect::from_min_size(
        Pos2::new(raw_rect.left().round(), raw_rect.top().round()),
        Vec2::new(36.0, 20.0),
    );
    let resp = resp.on_hover_cursor(if locked {
        CursorIcon::NotAllowed
    } else {
        CursorIcon::PointingHand
    });
    let radius = CornerRadius::same(10);
    let (track_fill, rim_color) = match toggle_state {
        PillToggleState::On => (
            if locked {
                color32_with_alpha(theme.accent(), 150)
            } else {
                theme.accent()
            },
            if resp.hovered() && !locked {
                theme.value()
            } else {
                Color32::from_rgb(0x50, 0x60, 0x84)
            },
        ),
        PillToggleState::Partial => (
            BG_HOVER,
            if resp.hovered() && !locked {
                theme.value()
            } else {
                TEXT_FAINT
            },
        ),
        PillToggleState::Off => (
            BG_INPUT,
            if resp.hovered() && !locked {
                theme.value()
            } else {
                Color32::from_rgb(0x4d, 0x4d, 0x4d)
            },
        ),
    };
    ui.painter().rect_filled(rect, radius, track_fill);
    ui.painter().rect_stroke(
        rect,
        radius,
        Stroke::new(1.0, rim_color),
        StrokeKind::Inside,
    );
    match toggle_state {
        PillToggleState::On => {
            ui.painter().circle_filled(
                Pos2::new(rect.right() - 10.0, rect.center().y),
                7.0,
                TEXT_BRIGHT,
            );
        }
        PillToggleState::Off => {
            ui.painter().circle_filled(
                Pos2::new(rect.left() + 10.0, rect.center().y),
                7.0,
                TEXT_MUTED,
            );
        }
        PillToggleState::Partial => {
            let dash = Rect::from_min_size(
                Pos2::new(rect.left() + 9.0, rect.top() + 8.0),
                Vec2::new(18.0, 4.0),
            );
            ui.painter()
                .rect_filled(dash, CornerRadius::same(2), TEXT_MUTED);
        }
    }
    resp
}

fn show_plugins_tab(ui: &mut Ui, session: &mut Session, theme: &Cm3Theme, state: &mut PanelsState) {
    let active_deps = session.active_plugin_dependencies();

    ScrollArea::vertical()
        .id_salt("plugins_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;

            // Edge-to-edge 43px #222222 header bar ("Plugins")
            panel_top_header(ui, "Plugins", theme);

            // 34px single-frame "type to filter" search box (w=301, right: 16)
            Frame::NONE
                .inner_margin(egui::Margin {
                    left: 4,
                    right: 16,
                    top: 5,
                    bottom: 0,
                })
                .show(ui, |ui| {
                    let filter_id = Id::unique("plugins_filter_box");
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
                                .inner_margin(egui::Margin {
                                    left: 8,
                                    right: 8,
                                    top: 7,
                                    bottom: 5,
                                })
                                .show(ui, |ui| {
                                    ui.add(
                                        TextEdit::singleline(&mut state.plugin_filter)
                                            .id(filter_id)
                                            .frame(Frame::NONE)
                                            .font(theme.font(FONT_BODY))
                                            .text_color(TEXT_PRIMARY)
                                            .desired_width((box_w - 16.0).max(40.0)),
                                    );
                                });
                            if state.plugin_filter.is_empty() {
                                ui.painter().text(
                                    Pos2::new(box_rect.left() + 8.0, box_rect.center().y + 1.0),
                                    Align2::LEFT_CENTER,
                                    "type to filter",
                                    theme.font(FONT_BODY),
                                    TEXT_FAINT,
                                );
                            }
                        },
                    );
                });

            // Dismissible notice banner with "x" button
            if let Some(notice) = state.plugin_notice.clone() {
                let mut dismiss = false;
                let resp = Frame::NONE
                    .fill(WARNING_SURFACE)
                    .inner_margin(egui::Margin {
                        left: 10,
                        right: 6,
                        top: 6,
                        bottom: 6,
                    })
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.set_max_width((ui.available_width() - 22.0).max(60.0));
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(&notice)
                                            .font(theme.font(FONT_SMALL))
                                            .color(DANGER_TEXT),
                                    )
                                    .wrap(),
                                );
                            });
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Min),
                                |ui| {
                                    if cm3_icon_button(
                                        ui,
                                        Cm3Icon::Close,
                                        12.0,
                                        DANGER_TEXT,
                                        "Dismiss notice",
                                        theme,
                                    ) {
                                        dismiss = true;
                                    }
                                },
                            );
                        });
                    });
                let r = resp.response.rect;
                let bar = Rect::from_min_size(r.left_top(), Vec2::new(3.0, r.height()));
                ui.painter()
                    .rect_filled(bar, CornerRadius::ZERO, WARNING_BAR);
                if dismiss {
                    state.plugin_notice = None;
                }
            }

            let q = state.plugin_filter.trim().to_ascii_lowercase();

            for category in [
                PluginCategory::Utilities,
                PluginCategory::Scene,
                PluginCategory::Adjustment,
                PluginCategory::Experimental,
            ] {
                let all_in_cat: Vec<_> = session
                    .plugins
                    .visible_in_category(category)
                    .into_iter()
                    .cloned()
                    .collect();
                if all_in_cat.is_empty() {
                    continue;
                }
                let enabled_count = all_in_cat
                    .iter()
                    .filter(|d| session.plugins.is_enabled(&d.id))
                    .count();
                let descriptors: Vec<_> = all_in_cat
                    .iter()
                    .filter(|d| {
                        let summary = zoidium_plugin_summary(&d.id, &d.tagline);
                        q.is_empty()
                            || d.name.to_ascii_lowercase().contains(&q)
                            || summary.to_ascii_lowercase().contains(&q)
                            || d.tagline.to_ascii_lowercase().contains(&q)
                            || d.id.to_ascii_lowercase().contains(&q)
                    })
                    .cloned()
                    .collect();
                if descriptors.is_empty() {
                    continue;
                }

                let cat_id = Id::unique(("plugin_cat", category.display_name()));
                let default_open = !category.collapsed_by_default();
                let mut open = ui
                    .data_mut(|d| d.get_temp::<bool>(cat_id))
                    .unwrap_or(default_open);

                // Flat #2a2a2a 40px category header (1px #242424 bottom divider), #7e8fb9 chevron,
                // category name (ellipsis-truncated), muted "X of Y on" counter, right-aligned 36x20 pill
                let cat_w = ui.available_width();
                let (cat_rect, _) =
                    ui.allocate_exact_size(Vec2::new(cat_w, 40.0), Sense::hover());
                ui.painter()
                    .rect_filled(cat_rect, CornerRadius::ZERO, BG_PANEL);
                let cat_div_y = (cat_rect.bottom() - 1.0).round();
                ui.painter().rect_filled(
                    Rect::from_min_max(
                        Pos2::new(cat_rect.left(), cat_div_y),
                        Pos2::new(cat_rect.right(), cat_div_y + 1.0),
                    ),
                    CornerRadius::ZERO,
                    BORDER_ROW_SOFT,
                );

                // Toggle click zone on right (36x20 pill aligned to x=329..365)
                let cat_toggle_rect = Rect::from_min_size(
                    Pos2::new(
                        (cat_rect.right() - 52.0).round(),
                        (cat_rect.center().y - 10.0).round(),
                    ),
                    Vec2::new(36.0, 20.0),
                );
                let cat_hdr_click_rect = Rect::from_min_max(
                    cat_rect.left_top(),
                    Pos2::new(
                        (cat_toggle_rect.left() - 8.0).max(cat_rect.left()),
                        cat_rect.bottom(),
                    ),
                );
                let cat_resp = ui
                    .interact(cat_hdr_click_rect, cat_id.with("hdr"), Sense::click())
                    .on_hover_cursor(CursorIcon::PointingHand);
                if cat_resp.clicked() {
                    open = !open;
                }
                ui.data_mut(|d| d.insert_temp(cat_id, open));

                // #7e8fb9 chevron at left + 11.0, center.y + 0.5
                paint_small_chevron(
                    ui.painter(),
                    Pos2::new(cat_rect.left() + 11.0, cat_rect.center().y + 0.5),
                    open,
                    theme.value(),
                );

                let count_text = format!("{} of {} on", enabled_count, all_in_cat.len());
                let count_galley = ui.painter().layout_no_wrap(
                    count_text,
                    theme.font(FONT_SMALL),
                    TEXT_MUTED,
                );
                let count_right = (cat_toggle_rect.left() - 18.0).max(cat_rect.left() + 80.0);
                let count_left = (count_right - count_galley.size().x).max(cat_rect.left() + 80.0);

                let name_x = cat_rect.left() + 22.0;
                let max_name_w = (count_left - 12.0 - name_x).max(24.0);
                let raw_name = category.display_name();
                let mut name_galley = ui.painter().layout_no_wrap(
                    raw_name.to_owned(),
                    theme.font(FONT_BODY),
                    TEXT_PRIMARY,
                );
                let mut name_truncated = false;
                if name_galley.size().x > max_name_w {
                    name_truncated = true;
                    let chars: Vec<char> = raw_name.chars().collect();
                    let mut len = chars.len();
                    while len > 0 {
                        len -= 1;
                        let candidate: String =
                            chars[..len].iter().collect::<String>() + "...";
                        let g = ui.painter().layout_no_wrap(
                            candidate,
                            theme.font(FONT_BODY),
                            TEXT_PRIMARY,
                        );
                        if g.size().x <= max_name_w || len == 0 {
                            name_galley = g;
                            break;
                        }
                    }
                }
                let actual_count_x = if name_truncated {
                    count_left
                } else {
                    (name_x + name_galley.size().x + 8.0).min(count_left)
                };
                ui.painter().galley(
                    Pos2::new(name_x, cat_rect.center().y - name_galley.size().y * 0.5),
                    name_galley,
                    TEXT_PRIMARY,
                );
                ui.painter().galley(
                    Pos2::new(
                        actual_count_x,
                        cat_rect.center().y - count_galley.size().y * 0.5,
                    ),
                    count_galley,
                    TEXT_MUTED,
                );

                // Category-level pill switch using new_child (NEVER rewinds parent cursor!)
                let cat_pill_state = if enabled_count == all_in_cat.len() {
                    PillToggleState::On
                } else if enabled_count == 0 {
                    PillToggleState::Off
                } else {
                    PillToggleState::Partial
                };
                let mut cat_pill_ui =
                    ui.new_child(egui::UiBuilder::new().max_rect(cat_toggle_rect));
                let cat_toggle_resp =
                    paint_pill_toggle(&mut cat_pill_ui, cat_pill_state, false, theme);
                if cat_toggle_resp.clicked() {
                    let enable_all = enabled_count < all_in_cat.len();
                    let mut blocked = 0usize;
                    for d in &all_in_cat {
                        if d.always_enabled {
                            continue;
                        }
                        if enable_all && !session.plugins.is_enabled(&d.id) {
                            let _ = session.execute(Command::EnablePlugin {
                                plugin_id: d.id.clone(),
                            });
                        } else if !enable_all
                            && session.plugins.is_enabled(&d.id)
                            && session
                                .execute(Command::DisablePlugin {
                                    plugin_id: d.id.clone(),
                                })
                                .is_err()
                        {
                            blocked += 1;
                        }
                    }
                    if blocked > 0 {
                        state.plugin_notice = Some(format!(
                            "{blocked} plugin(s) remained enabled because they are in use by the active project."
                        ));
                    } else {
                        state.plugin_notice = None;
                    }
                }

                if open {
                    if category == PluginCategory::Experimental {
                        warning_banner(ui, zc_plugin::EXPERIMENTAL_WARNING, theme);
                    }

                    for desc in descriptors {
                        let enabled = session.plugins.is_enabled(&desc.id);
                        let in_use = active_deps.contains(&desc.id);
                        let locked = desc.always_enabled || (enabled && in_use);
                        let cm3_compat = is_cm3_compatible_plugin(&desc.id);
                        let summary = zoidium_plugin_summary(&desc.id, &desc.tagline);

                        // Full-bleed flat #2a2a2a row with exact Zoidium 55px (1-line) / 70px (2-line) cadence
                        let row_w = ui.available_width();
                        let text_w = (row_w - 64.0).max(80.0);
                        let mut summary_job = egui::text::LayoutJob::default();
                        summary_job.wrap.max_width = text_w;
                        summary_job.append(
                            summary,
                            0.0,
                            egui::TextFormat {
                                font_id: theme.font(FONT_SMALL),
                                color: TEXT_MUTED,
                                line_height: Some(18.0),
                                ..Default::default()
                            },
                        );
                        let summary_galley = ui.painter().layout_job(summary_job);
                        let summary_lines = summary_galley.rows.len();
                        let warn_galley = desc.warning.as_ref().map(|w| {
                            ui.painter().layout(
                                w.clone(),
                                theme.font(FONT_COMPACT),
                                DANGER_TEXT,
                                text_w,
                            )
                        });
                        let warn_extra_h =
                            warn_galley.as_ref().map_or(0.0, |g| g.size().y + 6.0);
                        let row_h = if summary_lines <= 1 { 55.0 } else { 70.0 } + warn_extra_h;

                        let (row_rect, _) =
                            ui.allocate_exact_size(Vec2::new(row_w, row_h), Sense::hover());
                        ui.painter()
                            .rect_filled(row_rect, CornerRadius::ZERO, BG_PANEL);
                        let row_div_y = (row_rect.bottom() - 1.0).round();
                        ui.painter().rect_filled(
                            Rect::from_min_max(
                                Pos2::new(row_rect.left(), row_div_y),
                                Pos2::new(row_rect.right(), row_div_y + 1.0),
                            ),
                            CornerRadius::ZERO,
                            BORDER_ROW_SOFT,
                        );

                        // Deterministic pixel-exact title, badge, and summary painting
                        let text_left = row_rect.left() + 5.0;
                        let text_clip = Rect::from_min_max(
                            Pos2::new(text_left, row_rect.top()),
                            Pos2::new(
                                (row_rect.right() - 56.0).max(text_left + 80.0),
                                row_rect.bottom() - 1.0,
                            ),
                        );
                        let text_painter = ui.painter().with_clip_rect(text_clip);

                        let title_y = row_rect.top() + if summary_lines <= 1 { 7.5 } else { 5.5 };
                        let badge_y =
                            row_rect.top() + if summary_lines <= 1 { 11.0 } else { 9.0 };
                        let title_galley = text_painter.layout_no_wrap(
                            desc.name.clone(),
                            theme.font(FONT_BODY),
                            TEXT_PRIMARY,
                        );
                        let title_w = title_galley.size().x;
                        text_painter.galley(
                            Pos2::new(text_left, title_y),
                            title_galley,
                            TEXT_PRIMARY,
                        );

                        let badge_x = text_left + title_w + 6.0;
                        let badge_w = if cm3_compat {
                            paint_outlined_plugin_badge_at(
                                &text_painter,
                                Pos2::new(badge_x, badge_y),
                                "Standard",
                                Color32::from_rgb(0x2d, 0x30, 0x37),
                                theme.value(),
                                Color32::from_rgb(0xb2, 0xbc, 0xd5),
                                theme,
                            )
                        } else {
                            paint_outlined_plugin_badge_at(
                                &text_painter,
                                Pos2::new(badge_x, badge_y),
                                "Extended",
                                Color32::from_rgb(0x38, 0x28, 0x28),
                                WARNING_BAR,
                                DANGER_TEXT,
                                theme,
                            )
                        };
                        if in_use {
                            let in_use_rect = Rect::from_min_size(
                                Pos2::new((badge_x + badge_w + 6.0).round(), badge_y.round()),
                                Vec2::new(42.0, 16.0),
                            );
                            paint_badge(
                                &text_painter,
                                in_use_rect,
                                "IN USE",
                                theme.accent(),
                                TEXT_BRIGHT,
                            );
                        }

                        let summary_y =
                            row_rect.top() + if summary_lines <= 1 { 30.0 } else { 29.0 };
                        let summary_h = summary_galley.size().y;
                        text_painter.galley(
                            Pos2::new(text_left, summary_y),
                            summary_galley,
                            TEXT_MUTED,
                        );
                        if let Some(wg) = warn_galley {
                            text_painter.galley(
                                Pos2::new(text_left, summary_y + summary_h + 4.0),
                                wg,
                                DANGER_TEXT,
                            );
                        }

                        // Right pill switch aligned to x=329..365 (-52.0), vertically centered
                        let pill_rect = Rect::from_min_size(
                            Pos2::new(
                                (row_rect.right() - 52.0).round(),
                                (row_rect.center().y - 11.5).round(),
                            ),
                            Vec2::new(36.0, 20.0),
                        );
                        let mut pill_ui =
                            ui.new_child(egui::UiBuilder::new().max_rect(pill_rect));
                        let pill_state = if enabled {
                            PillToggleState::On
                        } else {
                            PillToggleState::Off
                        };
                        let toggle_resp =
                            paint_pill_toggle(&mut pill_ui, pill_state, locked, theme);
                        if toggle_resp.clicked() && !locked {
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
                                    state.plugin_notice = Some(err.to_string());
                                }
                            }
                        }
                    }
                }
            }

            // Subtle bottom footer for Vanilla CM3 Mode & Reset Defaults
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(10, 10))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let half_w = ((ui.available_width() - 6.0) * 0.5).max(80.0);
                        ui.allocate_ui(Vec2::new(half_w, 26.0), |ui| {
                            if cm3_prop_button(
                                ui,
                                "Core Mode",
                                ButtonVariant::Normal,
                                true,
                                theme,
                            )
                            .on_hover_text(
                                "Disable all optional plugins to enter Core Mode",
                            )
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
                                                Some("Switched to Core Mode.".to_owned());
                                        }
                                    }
                                    Err(err) => {
                                        state.plugin_notice = Some(err.to_string());
                                    }
                                }
                            }
                        });
                        ui.add_space(6.0);
                        ui.allocate_ui(Vec2::new(half_w, 26.0), |ui| {
                            if cm3_prop_button(
                                ui,
                                "Reset Defaults",
                                ButtonVariant::Normal,
                                true,
                                theme,
                            )
                            .on_hover_text(
                                "Reset plugins to defaults while preserving in-use project plugins",
                            )
                            .clicked()
                            {
                                session.plugins.reset_to_defaults();
                                let mut kept_in_use = 0usize;
                                for dep_id in &active_deps {
                                    if !session.plugins.is_enabled(dep_id)
                                        && session.plugins.enable(dep_id).is_ok()
                                    {
                                        kept_in_use += 1;
                                    }
                                }
                                if kept_in_use > 0 {
                                    state.plugin_notice = Some(format!(
                                        "Reset plugins to defaults ({kept_in_use} in-use plugin(s) remained enabled)."
                                    ));
                                } else {
                                    state.plugin_notice = None;
                                }
                            }
                        });
                    });
                });
        });
}

// ---------------------------------------------------------------------------
// Tab 11: Settings (Flat Zoidium Settings Matching 07_settings_clean.png)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ThemeColorPopupState {
    open: bool,
    hsv: Hsv,
    hex_buf: String,
}

/// Theme main color picker without a dead viewport eyedropper button.
fn show_theme_color_picker(
    ui: &mut Ui,
    id: Id,
    rgb: &mut Rgb,
    viewport_rect: Rect,
    theme: &Cm3Theme,
) -> bool {
    let popup_id = id.with("theme_color_popup");
    let mut st = ui
        .data_mut(|d| d.get_temp::<ThemeColorPopupState>(popup_id))
        .unwrap_or_else(|| ThemeColorPopupState {
            open: false,
            hsv: rgb.to_hsv(),
            hex_buf: rgb.with_alpha(1.0).to_hex_rgb(),
        });
    let mut changed = false;
    let mut swatch_rect = Rect::NOTHING;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(30.0, 20.0), Sense::click());
        swatch_rect = rect;
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, rgb_to_color32(*rgb));
        ui.painter().rect_stroke(
            rect,
            CornerRadius::ZERO,
            Stroke::new(
                1.0,
                if st.open || resp.hovered() {
                    theme.accent()
                } else {
                    BORDER_CONTROL
                },
            ),
            StrokeKind::Inside,
        );
        if resp.on_hover_cursor(CursorIcon::PointingHand).clicked() {
            st.open = !st.open;
            if st.open {
                st.hsv = rgb.to_hsv();
                st.hex_buf = rgb.with_alpha(1.0).to_hex_rgb();
            }
        }

        // Inline hex input (single-frame, no dead eyedropper)
        let hex_id = id.with("theme_hex_inline");
        if !ui.memory(|m| m.has_focus(hex_id)) && !st.open {
            st.hex_buf = rgb.with_alpha(1.0).to_hex_rgb();
        }
        let hex_resp = Frame::NONE
            .fill(BG_INPUT)
            .stroke(Stroke::new(1.0, BORDER_CONTROL))
            .inner_margin(egui::Margin::symmetric(4, 2))
            .show(ui, |ui| {
                ui.add(
                    TextEdit::singleline(&mut st.hex_buf)
                        .id(hex_id)
                        .frame(Frame::NONE)
                        .font(theme.font(FONT_SMALL))
                        .text_color(theme.value())
                        .desired_width(78.0),
                )
            })
            .inner;
        if hex_resp.lost_focus()
            || (hex_resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter)))
        {
            if let Ok(parsed) = Rgba::from_hex(&st.hex_buf) {
                *rgb = parsed.rgb();
                st.hsv = rgb.to_hsv();
                st.hex_buf = rgb.with_alpha(1.0).to_hex_rgb();
                changed = true;
            }
            if hex_resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                ui.memory_mut(|m| m.surrender_focus(hex_id));
            }
        }
    });

    if st.open && swatch_rect != Rect::NOTHING {
        let popup_size = Vec2::new(230.0, 195.0);
        let mut x = swatch_rect.left();
        let mut y = swatch_rect.bottom() + 2.0;
        if y + popup_size.y > viewport_rect.bottom() - 4.0 {
            y = (swatch_rect.top() - popup_size.y - 2.0).max(viewport_rect.top() + 4.0);
        }
        if x + popup_size.x > viewport_rect.right() - 4.0 {
            x = (viewport_rect.right() - popup_size.x - 4.0).max(viewport_rect.left() + 4.0);
        }
        let area_resp = Area::new(popup_id)
            .order(Order::Foreground)
            .fixed_pos(Pos2::new(x, y))
            .interactable(true)
            .show(ui.ctx(), |ui| {
                Frame::NONE
                    .fill(BG_PANEL)
                    .stroke(Stroke::new(1.0, theme.accent()))
                    .inner_margin(egui::Margin::same(6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            // 180x180 SV square
                            let (sv_rect, sv_resp) =
                                ui.allocate_exact_size(Vec2::splat(180.0), Sense::click_and_drag());
                            let pure = rgb_to_color32(Rgb::from_hsv(Hsv {
                                h: st.hsv.h,
                                s: 1.0,
                                v: 1.0,
                            }));
                            let mut mesh = Mesh::default();
                            mesh.colored_vertex(sv_rect.left_top(), Color32::WHITE);
                            mesh.colored_vertex(sv_rect.right_top(), pure);
                            mesh.colored_vertex(sv_rect.right_bottom(), Color32::BLACK);
                            mesh.colored_vertex(sv_rect.left_bottom(), Color32::BLACK);
                            mesh.add_triangle(0, 1, 2);
                            mesh.add_triangle(0, 2, 3);
                            ui.painter().add(Shape::mesh(mesh));
                            if (sv_resp.clicked() || sv_resp.dragged())
                                && let Some(p) = ui.input(|i| i.pointer.interact_pos())
                            {
                                st.hsv.s =
                                    ((p.x - sv_rect.left()) / sv_rect.width()).clamp(0.0, 1.0);
                                st.hsv.v = (1.0 - (p.y - sv_rect.top()) / sv_rect.height())
                                    .clamp(0.0, 1.0);
                                *rgb = Rgb::from_hsv(st.hsv);
                                st.hex_buf = rgb.with_alpha(1.0).to_hex_rgb();
                                changed = true;
                            }
                            let ret = Pos2::new(
                                sv_rect.left() + st.hsv.s * sv_rect.width(),
                                sv_rect.top() + (1.0 - st.hsv.v) * sv_rect.height(),
                            );
                            ui.painter().circle_stroke(
                                ret,
                                4.0,
                                Stroke::new(1.5, Color32::from_gray(0xcc)),
                            );

                            // 22x180 Hue bar
                            let (hue_rect, hue_resp) = ui.allocate_exact_size(
                                Vec2::new(22.0, 180.0),
                                Sense::click_and_drag(),
                            );
                            let stops = [
                                Color32::from_rgb(255, 0, 0),
                                Color32::from_rgb(255, 255, 0),
                                Color32::from_rgb(0, 255, 0),
                                Color32::from_rgb(0, 255, 255),
                                Color32::from_rgb(0, 0, 255),
                                Color32::from_rgb(255, 0, 255),
                                Color32::from_rgb(255, 0, 0),
                            ];
                            let mut hmesh = Mesh::default();
                            let seg_h = hue_rect.height() / 6.0;
                            for i in 0..6 {
                                let y0 = hue_rect.top() + seg_h * i as f32;
                                let y1 = hue_rect.top() + seg_h * (i + 1) as f32;
                                let base = hmesh.vertices.len() as u32;
                                hmesh.colored_vertex(Pos2::new(hue_rect.left(), y0), stops[i]);
                                hmesh.colored_vertex(Pos2::new(hue_rect.right(), y0), stops[i]);
                                hmesh.colored_vertex(Pos2::new(hue_rect.right(), y1), stops[i + 1]);
                                hmesh.colored_vertex(Pos2::new(hue_rect.left(), y1), stops[i + 1]);
                                hmesh.add_triangle(base, base + 1, base + 2);
                                hmesh.add_triangle(base, base + 2, base + 3);
                            }
                            ui.painter().add(Shape::mesh(hmesh));
                            if (hue_resp.clicked() || hue_resp.dragged())
                                && let Some(p) = ui.input(|i| i.pointer.interact_pos())
                            {
                                st.hsv.h =
                                    ((p.y - hue_rect.top()) / hue_rect.height()).clamp(0.0, 0.9999);
                                *rgb = Rgb::from_hsv(st.hsv);
                                st.hex_buf = rgb.with_alpha(1.0).to_hex_rgb();
                                changed = true;
                            }
                            let hy = hue_rect.top() + st.hsv.h * hue_rect.height();
                            ui.painter().line_segment(
                                [
                                    Pos2::new(hue_rect.left(), hy),
                                    Pos2::new(hue_rect.right(), hy),
                                ],
                                Stroke::new(2.0, Color32::WHITE),
                            );
                        });
                    });
            });

        let close_popup = ui.input(|i| {
            i.key_pressed(Key::Escape)
                || (i.pointer.primary_pressed()
                    && i.pointer.interact_pos().is_some_and(|p| {
                        !area_resp.response.rect.contains(p) && !swatch_rect.contains(p)
                    }))
        });
        if close_popup {
            st.open = false;
        }
    }

    ui.data_mut(|d| d.insert_temp(popup_id, st));
    changed
}

fn settings_flat_row_margin(ui: &mut Ui, margin: egui::Margin, add_content: impl FnOnce(&mut Ui)) {
    let resp = Frame::NONE
        .fill(BG_PANEL)
        .inner_margin(margin)
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            add_content(ui);
        });
    let r = resp.response.rect;
    let div_y = (r.bottom() - 1.0).round();
    ui.painter().rect_filled(
        Rect::from_min_max(
            Pos2::new(r.left(), div_y),
            Pos2::new(r.right(), div_y + 1.0),
        ),
        CornerRadius::ZERO,
        BORDER_ROW_SOFT,
    );
}

fn settings_flat_row_pad(ui: &mut Ui, pad_v: i8, add_content: impl FnOnce(&mut Ui)) {
    settings_flat_row_margin(
        ui,
        egui::Margin {
            left: 6,
            right: 5,
            top: pad_v,
            bottom: pad_v,
        },
        add_content,
    );
}

/// Fixed-width `fixed_w × 21 px` CM3 Settings dropdown with `14 px` (`FONT_SMALL`) interior text
/// matching exact Zoidium `07_settings_clean.png` geometry (`222 / 78 / 230 / 119 px`).
fn settings_fixed_dropdown(
    ui: &mut Ui,
    id: Id,
    selected_idx: &mut usize,
    options: &[&str],
    fixed_w: f32,
    viewport_rect: Rect,
    theme: &Cm3Theme,
) -> bool {
    if options.is_empty() {
        return false;
    }
    if *selected_idx >= options.len() {
        *selected_idx = 0;
    }
    let popup_id = id.with("settings_dd_popup");
    let mut open = ui
        .data_mut(|d| d.get_temp::<bool>(popup_id))
        .unwrap_or(false);
    let mut changed = false;

    let width = fixed_w.min(ui.available_width().max(44.0)).round();
    let (raw_btn_rect, btn_resp) = ui.allocate_exact_size(Vec2::new(width, 21.0), Sense::click());
    let btn_rect = Rect::from_min_size(
        Pos2::new(raw_btn_rect.left().round(), raw_btn_rect.top().round()),
        Vec2::new(width, 21.0),
    );
    let btn_resp = btn_resp.on_hover_cursor(CursorIcon::PointingHand);

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

    let current_label = options[*selected_idx];
    let label_galley = ui.painter().layout_no_wrap(
        current_label.to_owned(),
        theme.font(FONT_SMALL),
        theme.value(),
    );
    let clip_painter = ui.painter().with_clip_rect(Rect::from_min_max(
        btn_rect.left_top(),
        Pos2::new(
            (btn_rect.right() - 16.0).max(btn_rect.left()),
            btn_rect.bottom(),
        ),
    ));
    clip_painter.galley(
        Pos2::new(
            btn_rect.left() + 5.0,
            btn_rect.center().y - label_galley.size().y * 0.5,
        ),
        label_galley,
        theme.value(),
    );
    let caret_rect = Rect::from_center_size(
        Pos2::new(btn_rect.right() - 8.5, btn_rect.center().y),
        Vec2::splat(11.0),
    );
    paint_cm3_icon(
        ui.painter(),
        caret_rect,
        Cm3Icon::ChevronDown,
        theme.value(),
    );

    if btn_resp.clicked() {
        open = !open;
    }

    if open {
        let row_h = 22.0;
        let popup_h = (options.len() as f32 * row_h + 4.0).min(240.0);
        let mut x = btn_rect.left().round();
        let mut y = (btn_rect.bottom() + 1.0).round();
        if y + popup_h > viewport_rect.bottom() - 4.0 {
            y = (btn_rect.top() - popup_h - 1.0).max(viewport_rect.top() + 4.0);
        }
        if x + width > viewport_rect.right() - 4.0 {
            x = (viewport_rect.right() - width - 4.0).max(viewport_rect.left() + 4.0);
        }
        let esc = ui.input(|i| i.key_pressed(Key::Escape));
        if esc {
            open = false;
        }
        let area_resp = Area::new(popup_id)
            .order(Order::Foreground)
            .fixed_pos(Pos2::new(x, y))
            .interactable(true)
            .show(ui.ctx(), |ui| {
                Frame::NONE
                    .fill(BG_INPUT)
                    .stroke(Stroke::new(1.0, theme.accent()))
                    .inner_margin(egui::Margin::same(2))
                    .show(ui, |ui| {
                        ui.set_min_width(width - 4.0);
                        ui.spacing_mut().item_spacing.y = 0.0;
                        ScrollArea::vertical().max_height(236.0).show(ui, |ui| {
                            for (idx, opt) in options.iter().enumerate() {
                                let (r, resp) = ui.allocate_exact_size(
                                    Vec2::new((width - 4.0).max(40.0), row_h),
                                    Sense::click(),
                                );
                                let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
                                if idx == *selected_idx {
                                    ui.painter()
                                        .rect_filled(r, CornerRadius::ZERO, theme.accent());
                                } else if resp.hovered() {
                                    ui.painter().rect_filled(
                                        r,
                                        CornerRadius::ZERO,
                                        BG_DROPDOWN_HOVER,
                                    );
                                }
                                ui.painter().with_clip_rect(r).text(
                                    Pos2::new(r.left() + 4.0, r.center().y),
                                    Align2::LEFT_CENTER,
                                    *opt,
                                    theme.font(FONT_SMALL),
                                    if idx == *selected_idx || resp.hovered() {
                                        TEXT_BRIGHT
                                    } else {
                                        TEXT_PRIMARY
                                    },
                                );
                                if resp.clicked() {
                                    if *selected_idx != idx {
                                        *selected_idx = idx;
                                        changed = true;
                                    }
                                    open = false;
                                }
                            }
                        });
                    });
            });
        let clicked_outside = ui.input(|i| {
            i.pointer.primary_pressed()
                && i.pointer
                    .interact_pos()
                    .is_some_and(|p| !area_resp.response.rect.contains(p) && !btn_rect.contains(p))
        });
        if clicked_outside {
            open = false;
        }
    }

    ui.data_mut(|d| d.insert_temp(popup_id, open));
    changed
}

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
            ui.spacing_mut().item_spacing = Vec2::ZERO;

            // Edge-to-edge 43px #222222 header bar ("Settings")
            panel_top_header(ui, "Settings", theme);

            // 1a. Editor layout control row (51px total; dropdown TOP-aligned at top_margin = 6px)
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 5,
                    top: 5,
                    bottom: 6,
                },
                |ui| {
                    let total_w = ui.available_width();
                    let dd_w = 222.0_f32.min((total_w - 66.0).max(90.0));
                    let label_w = (total_w - dd_w - 6.0).max(52.0);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        ui.allocate_ui_with_layout(
                            Vec2::new(label_w, 40.0),
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new("Editor layout")
                                            .font(theme.font(FONT_BODY))
                                            .color(TEXT_PRIMARY),
                                    )
                                    .wrap(),
                                );
                            },
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                            Frame::NONE
                                .inner_margin(egui::Margin {
                                    left: 0,
                                    right: 0,
                                    top: 1,
                                    bottom: 0,
                                })
                                .show(ui, |ui| {
                                    settings_fixed_dropdown(
                                        ui,
                                        Id::unique("settings_editor_layout"),
                                        &mut state.settings_editor_layout,
                                        &["Classic"],
                                        dd_w,
                                        viewport_rect,
                                        theme,
                                    );
                                });
                        });
                    });
                },
            );

            // 1b. Editor layout description row (71px total, 32-char wrap width)
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 4,
                    top: 6,
                    bottom: 5,
                },
                |ui| {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(
                                "Changing the editor layout takes effect after ZoidCraft is reloaded.",
                            )
                            .font(theme.font(FONT_BODY))
                            .color(TEXT_PRIMARY),
                        )
                        .wrap(),
                    );
                },
            );

            // 2. Default frame rate: boxed #202020 1px #1b1b1b 90x21 numeric input (32px total)
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 5,
                    top: 5,
                    bottom: 5,
                },
                |ui| {
                    let fps_id = Id::unique("settings_default_fps");
                    if !ui.memory(|m| m.has_focus(fps_id)) {
                        state.settings_default_fps_buf = format!("{:.0}", session.document().fps);
                    }
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Default frame rate")
                                .font(theme.font(FONT_BODY))
                                .color(TEXT_PRIMARY),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let focused = ui.memory(|m| m.has_focus(fps_id));
                            let box_w = 90.0;
                            let resp = ui
                                .allocate_ui_with_layout(
                                    Vec2::new(box_w, 21.0),
                                    egui::Layout::left_to_right(egui::Align::Center),
                                    |ui| {
                                        let box_rect = Rect::from_min_size(
                                            ui.cursor().min,
                                            Vec2::new(box_w, 21.0),
                                        );
                                        ui.painter()
                                            .rect_filled(box_rect, CornerRadius::ZERO, BG_INPUT);
                                        paint_crisp_panel_border(
                                            ui.painter(),
                                            box_rect,
                                            if focused {
                                                theme.accent()
                                            } else {
                                                BORDER_CONTROL
                                            },
                                        );
                                        Frame::NONE
                                            .inner_margin(egui::Margin::symmetric(5, 1))
                                            .show(ui, |ui| {
                                                ui.add(
                                                    TextEdit::singleline(
                                                        &mut state.settings_default_fps_buf,
                                                    )
                                                    .id(fps_id)
                                                    .frame(Frame::NONE)
                                                    .font(theme.font(FONT_SMALL))
                                                    .text_color(theme.value())
                                                    .desired_width(box_w - 10.0),
                                                )
                                            })
                                            .inner
                                    },
                                )
                                .inner;
                            if resp.lost_focus()
                                || (resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter)))
                            {
                                if let Ok(parsed) =
                                    state.settings_default_fps_buf.trim().parse::<f64>()
                                {
                                    let clamped = parsed.clamp(1.0, 240.0).round();
                                    let _ = session.execute(Command::SetFps { fps: clamped });
                                    state.settings_default_fps_buf = format!("{clamped:.0}");
                                }
                                if resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                                    ui.memory_mut(|m| m.surrender_focus(fps_id));
                                }
                            }
                        });
                    });
                },
            );

            // 3a. Preview cache control row (33px total, left=9 indent, 10px gap before 14px #6d6d6d "MiB")
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 9,
                    right: 5,
                    top: 6,
                    bottom: 5,
                },
                |ui| {
                    let mut cache = f64::from(state.app_settings.preview_cache_mib);
                    let mut cache_ev = ScrubEvent::default();
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Preview cache")
                                .font(theme.font(FONT_BODY))
                                .color(TEXT_PRIMARY),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new("MiB")
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_FAINT),
                            );
                            ui.add_space(10.0);
                            cache_ev = cm3_scrub_number(
                                ui,
                                Id::unique("settings_cache"),
                                &mut cache,
                                ScrubConfig::new(0.0, 16_384.0, 64.0, 0),
                                theme,
                            );
                        });
                    });
                    if cache_ev.changed || cache_ev.committed {
                        state.app_settings.preview_cache_mib =
                            cache.round().clamp(0.0, 16_384.0) as u32;
                    }
                },
            );

            // 3b. Preview cache telemetry row (60px total, left=9 indent, 23px line-height pitch)
            let [rw, rh] = session.document().resolution;
            let frame_bytes = u64::from(rw.max(1)) * u64::from(rh.max(1)) * 4;
            let est_frames = if state.app_settings.preview_cache_mib == 0 {
                0
            } else {
                (u64::from(state.app_settings.preview_cache_mib) * 1024 * 1024)
                    / frame_bytes.max(1)
            };
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 9,
                    right: 4,
                    top: 7,
                    bottom: 7,
                },
                |ui| {
                    let mut job = egui::text::LayoutJob::default();
                    job.append(
                        &format!("Using 0 MiB · about {est_frames} frames at {rw}×{rh}"),
                        0.0,
                        egui::TextFormat {
                            font_id: theme.font(FONT_BODY),
                            color: TEXT_MUTED,
                            line_height: Some(23.0),
                            ..Default::default()
                        },
                    );
                    ui.add(egui::Label::new(job).wrap());
                },
            );

            // 3c. Preview cache description row (149px total, left=9 indent, 22.4px line-height pitch)
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 9,
                    right: 4,
                    top: 8,
                    bottom: 7,
                },
                |ui| {
                    let mut job = egui::text::LayoutJob::default();
                    job.append(
                        "Finished preview frames use up to this much GPU memory. Set 0 to turn off caching. Temporal input textures use up to another 256 MiB, plus active samples and rendering buffers.",
                        0.0,
                        egui::TextFormat {
                            font_id: theme.font(FONT_BODY),
                            color: TEXT_MUTED,
                            line_height: Some(22.4),
                            ..Default::default()
                        },
                    );
                    ui.add(egui::Label::new(job).wrap());
                },
            );

            // 4. Gradient editor (33px total, 78x21 dropdown)
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 5,
                    top: 6,
                    bottom: 6,
                },
                |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Gradient editor")
                                .font(theme.font(FONT_BODY))
                                .color(TEXT_PRIMARY),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            settings_fixed_dropdown(
                                ui,
                                Id::unique("settings_gradient_editor"),
                                &mut state.settings_gradient_editor,
                                &["Modern", "Classic"],
                                78.0,
                                viewport_rect,
                                theme,
                            );
                        });
                    });
                },
            );

            // 5. Editor font (51px total, 230x21 dropdown TOP-aligned at top_margin = 6px)
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 5,
                    top: 5,
                    bottom: 6,
                },
                |ui| {
                    let mut font_idx = EditorFontChoice::ALL
                        .iter()
                        .position(|&f| f == theme.editor_font)
                        .unwrap_or(0);
                    let font_labels: Vec<&str> =
                        EditorFontChoice::ALL.iter().map(|f| f.label()).collect();
                    let mut changed = false;
                    let total_w = ui.available_width();
                    let dd_w = 230.0_f32.min((total_w - 60.0).max(90.0));
                    let label_w = (total_w - dd_w - 6.0).max(52.0);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        ui.allocate_ui_with_layout(
                            Vec2::new(label_w, 40.0),
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new("Editor font")
                                            .font(theme.font(FONT_BODY))
                                            .color(TEXT_PRIMARY),
                                    )
                                    .wrap(),
                                );
                            },
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                            Frame::NONE
                                .inner_margin(egui::Margin {
                                    left: 0,
                                    right: 0,
                                    top: 1,
                                    bottom: 0,
                                })
                                .show(ui, |ui| {
                                    changed = settings_fixed_dropdown(
                                        ui,
                                        Id::unique("settings_editor_font"),
                                        &mut font_idx,
                                        &font_labels,
                                        dd_w,
                                        viewport_rect,
                                        theme,
                                    );
                                });
                        });
                    });
                    if changed {
                        theme.editor_font = EditorFontChoice::ALL[font_idx];
                    }
                },
            );

            // 6. Color profile (32px total: 119x21 dropdown + 5px gap + 54x21 "Edit..." button)
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 5,
                    top: 5,
                    bottom: 6,
                },
                |ui| {
                    ui.spacing_mut().item_spacing.y = 6.0;
                    let mut prof_idx = ThemeProfileChoice::ALL
                        .iter()
                        .position(|&p| p == theme.profile)
                        .unwrap_or(0);
                    let labels: Vec<&str> =
                        ThemeProfileChoice::ALL.iter().map(|p| p.label()).collect();
                    let mut prof_changed = false;

                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 5.0;
                        ui.label(
                            egui::RichText::new("Color profile")
                                .font(theme.font(FONT_BODY))
                                .color(TEXT_PRIMARY),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.spacing_mut().item_spacing.x = 5.0;
                            // Fixed 54x21 "Edit..." button
                            let (raw_edit_rect, edit_resp) =
                                ui.allocate_exact_size(Vec2::new(54.0, 21.0), Sense::click());
                            let edit_rect = Rect::from_min_size(
                                Pos2::new(raw_edit_rect.left().round(), raw_edit_rect.top().round()),
                                Vec2::new(54.0, 21.0),
                            );
                            let edit_resp = edit_resp.on_hover_cursor(CursorIcon::PointingHand);
                            ui.painter().rect_filled(
                                edit_rect,
                                CornerRadius::ZERO,
                                if state.settings_edit_colors || edit_resp.hovered() {
                                    BG_HOVER
                                } else {
                                    BG_TITLE
                                },
                            );
                            paint_crisp_panel_border(
                                ui.painter(),
                                edit_rect,
                                if state.settings_edit_colors || edit_resp.hovered() {
                                    theme.accent()
                                } else {
                                    BORDER_CONTROL
                                },
                            );
                            ui.painter().text(
                                edit_rect.center(),
                                Align2::CENTER_CENTER,
                                "Edit...",
                                theme.font(FONT_BADGE),
                                theme.value(),
                            );
                            if edit_resp.clicked() {
                                state.settings_edit_colors = !state.settings_edit_colors;
                            }

                            prof_changed = settings_fixed_dropdown(
                                ui,
                                Id::unique("settings_theme_profile"),
                                &mut prof_idx,
                                &labels,
                                119.0,
                                viewport_rect,
                                theme,
                            );
                        });
                    });
                    if prof_changed {
                        theme.set_profile(ThemeProfileChoice::ALL[prof_idx]);
                    }

                    // Inline custom color / hue / sat editor revealed ONLY when "Edit..." is toggled
                    if state.settings_edit_colors {
                        let mut main_rgb = theme.custom_main_rgb;
                        let color_changed = cm3_property_row_aligned(
                            ui,
                            "Main color",
                            false,
                            state.prop_columns,
                            theme,
                            |_| {},
                            |ui| {
                                show_theme_color_picker(
                                    ui,
                                    Id::unique("settings_main_color"),
                                    &mut main_rgb,
                                    viewport_rect,
                                    theme,
                                )
                            },
                        );
                        if color_changed {
                            theme.set_custom_main(main_rgb);
                        }

                        let mut hue = f64::from(theme.custom_hue_deg);
                        let hev = cm3_property_row_aligned(
                            ui,
                            "Hue shift (deg)",
                            false,
                            state.prop_columns,
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
                        let sev = cm3_property_row_aligned(
                            ui,
                            "Saturation (%)",
                            false,
                            state.prop_columns,
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

                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            let entries = [
                                ("focus", theme.accent()),
                                ("link", theme.value()),
                                ("clip", theme.clip()),
                                ("sel", theme.clip_selected()),
                                ("box", theme.clip_selection()),
                            ];
                            for (lbl, col) in entries {
                                let (r, _) = ui
                                    .allocate_exact_size(Vec2::new(42.0, 22.0), Sense::hover());
                                ui.painter().rect_filled(r, CornerRadius::ZERO, col);
                                paint_crisp_panel_border(ui.painter(), r, BORDER_CONTROL);
                                ui.painter().text(
                                    r.center(),
                                    Align2::CENTER_CENTER,
                                    lbl,
                                    theme.font(FONT_COMPACT),
                                    TEXT_BRIGHT,
                                );
                            }
                        });
                    }
                },
            );

            // 7. DEBUG INFORMATION: full-width 28px #222222 uppercase banner (27px body + 1px #1d1d1d divider)
            let (dbg_hdr, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), Sense::hover());
            ui.painter()
                .rect_filled(dbg_hdr, CornerRadius::ZERO, BG_TITLE);
            let dbg_div_y = (dbg_hdr.bottom() - 1.0).round();
            ui.painter().rect_filled(
                Rect::from_min_max(
                    Pos2::new(dbg_hdr.left(), dbg_div_y),
                    Pos2::new(dbg_hdr.right(), dbg_div_y + 1.0),
                ),
                CornerRadius::ZERO,
                BG_SUNKEN,
            );
            ui.painter().text(
                Pos2::new(dbg_hdr.left() + 6.0, dbg_hdr.center().y),
                Align2::LEFT_CENTER,
                "DEBUG INFORMATION",
                theme.font(FONT_BANNER),
                TEXT_FAINT,
            );

            // 7a. Full 8-line 16px #cccccc debug description row (171px total, 32-char wrap width)
            settings_flat_row_margin(
                ui,
                egui::Margin {
                    left: 6,
                    right: 4,
                    top: 6,
                    bottom: 5,
                },
                |ui| {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(
                                "Saves your OS, browser, enabled plugins, and recent crash evidence for a bug report. Crash data survives a reload. Project content is not included; review before sharing. Desktop users can use the Debug menu if this panel is unavailable.",
                            )
                            .font(theme.font(FONT_BODY))
                            .color(TEXT_PRIMARY),
                        )
                        .wrap(),
                    );
                },
            );

            // 7b. Additional ZoidCraft controls below the fold (accessible on scroll)
            settings_flat_row_pad(ui, 8, |ui| {
                ui.spacing_mut().item_spacing.y = 6.0;
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Show FPS overlay")
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_PRIMARY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let st = if state.app_settings.show_fps_overlay {
                            PillToggleState::On
                        } else {
                            PillToggleState::Off
                        };
                        if paint_pill_toggle(ui, st, false, theme).clicked() {
                            state.app_settings.show_fps_overlay =
                                !state.app_settings.show_fps_overlay;
                        }
                    });
                });

                let mut max_entries = state.app_settings.undo_max_entries as f64;
                let mut ev_e = ScrubEvent::default();
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Undo max steps")
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_PRIMARY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ev_e = cm3_scrub_number(
                            ui,
                            Id::unique("settings_undo_entries"),
                            &mut max_entries,
                            ScrubConfig::new(10.0, 2000.0, 10.0, 0),
                            theme,
                        );
                    });
                });

                let mut max_mib = state.app_settings.undo_max_mib as f64;
                let mut ev_b = ScrubEvent::default();
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Undo memory (MiB)")
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_PRIMARY),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ev_b = cm3_scrub_number(
                            ui,
                            Id::unique("settings_undo_mib"),
                            &mut max_mib,
                            ScrubConfig::new(4.0, 1024.0, 8.0, 0),
                            theme,
                        );
                    });
                });

                if ev_e.committed || ev_b.committed {
                    state.app_settings.undo_max_entries =
                        max_entries.round().clamp(10.0, 2000.0) as usize;
                    state.app_settings.undo_max_mib = max_mib.round().clamp(4.0, 1024.0) as usize;
                    session.history.set_limits(
                        state.app_settings.undo_max_entries,
                        state.app_settings.undo_max_mib * 1024 * 1024,
                    );
                }

                let enabled_opt = session
                    .plugins
                    .enabled_ids()
                    .iter()
                    .filter(|id| id.as_str() != "core")
                    .count();
                let [rw, rh] = session.document().resolution;
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(format!(
                            "ZoidCraft {} · {}×{} @ {:.0} fps · {} plugins active · undo {}/{}",
                            env!("CARGO_PKG_VERSION"),
                            rw,
                            rh,
                            session.document().fps,
                            enabled_opt,
                            session.history.undo_len(),
                            session.history.max_entries()
                        ))
                        .font(theme.font(FONT_COMPACT))
                        .color(TEXT_OPTION_DESC),
                    )
                    .wrap(),
                );
            });
        });
}

// ---------------------------------------------------------------------------
// Tab 10: About (about)
// ---------------------------------------------------------------------------

fn show_about_tab(ui: &mut Ui, session: &Session, theme: &Cm3Theme) {
    ScrollArea::vertical()
        .id_salt("about_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;
            panel_top_header(ui, "About", theme);

            Frame::NONE
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 4.0;
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
                    ui.add_space(6.0);
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(
                                "Native Rust + wgpu motion-graphics & 3D video editor \
                                 with modular built-in extensions.",
                            )
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_PRIMARY),
                        )
                        .wrap(),
                    );
                    ui.add_space(8.0);

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
                    ui.add_space(8.0);

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
        if !target_exists_in_session(session, &editor.effect_id) {
            state.floating_windows.close("shader_editor");
        } else {
            let spec = FloatingWindowSpec::new("shader_editor", "Shader Editor")
                .with_subtitle(editor.effect_name.clone())
                .with_default_size(Vec2::new(560.0, 440.0));
            let is_focused = state.floating_windows.focused_id.as_deref() == Some("shader_editor");
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
                        if cm3_prop_button(ui, "A-", ButtonVariant::Normal, false, theme).clicked()
                        {
                            editor.font_px = (editor.font_px - 2.0).clamp(10.0, 24.0);
                        }
                        if cm3_prop_button(ui, "A+", ButtonVariant::Normal, false, theme).clicked()
                        {
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
                match translate_glsl_es100(&editor.code) {
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
                        editor.error_message = Some(format!("Shader compile error: {err}"));
                    }
                }
            }
            if !ev.closed {
                state.shader_editor = Some(editor);
            }
        }
    }

    // 2. Expression Editor Floating Window
    if let Some(mut expr) = state.expr_editor.take() {
        if !target_exists_in_session(session, &expr.target_id) {
            state.floating_windows.close("expression_editor");
        } else {
            let spec = FloatingWindowSpec::new("expression_editor", "Expression Editor")
                .with_subtitle(expr.property.clone())
                .with_default_size(Vec2::new(420.0, 240.0));
            let is_focused =
                state.floating_windows.focused_id.as_deref() == Some("expression_editor");
            let mut commit_expr = false;
            let mut clear_expr = false;
            let ev = cm3_floating_window(
                ctx,
                &mut state.floating_windows,
                &spec,
                viewport_rect,
                theme,
                |ui| {
                    if let Some(err) = &expr.error_message {
                        warning_banner(ui, err, theme);
                        ui.add_space(4.0);
                    }
                    ui.add(
                        TextEdit::multiline(&mut expr.code)
                            .font(theme.font(FONT_SMALL))
                            .text_color(theme.value())
                            .desired_width(ui.available_width())
                            .desired_rows(6)
                            .code_editor(),
                    );
                    if is_focused
                        && ui.input(|i| {
                            (i.modifiers.ctrl || i.modifiers.command) && i.key_pressed(Key::Enter)
                        })
                    {
                        commit_expr = true;
                    }
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if cm3_prop_button(ui, "Apply", ButtonVariant::Primary, false, theme)
                            .clicked()
                        {
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
                    });
                },
            );
            if commit_expr {
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
            } else if clear_expr {
                match session.execute(Command::SetExpression {
                    target_id: expr.target_id.clone(),
                    property: expr.property.clone(),
                    expression: None,
                }) {
                    Ok(_) => {
                        state.floating_windows.close("expression_editor");
                    }
                    Err(err) => {
                        expr.error_message = Some(format!("Failed to clear expression: {err}"));
                        clear_expr = false;
                    }
                }
            }
            if !ev.closed && !clear_expr {
                state.expr_editor = Some(expr);
            }
        }
    }
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

#[allow(clippy::too_many_arguments)]
fn eval_scalar_segment(
    a_frame: f64,
    a_val: f64,
    a_out_tangent: [f64; 2],
    b_frame: f64,
    b_val: f64,
    b_in_tangent: [f64; 2],
    tween: u16,
    local_frame: f64,
) -> f64 {
    let span = b_frame - a_frame;
    if span <= 1e-9 {
        return b_val;
    }
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(tween);
    match mode {
        InterpolationMode::Discrete => a_val,
        InterpolationMode::Interpolated => {
            let t = ((local_frame - a_frame) / span).clamp(0.0, 1.0);
            let eased = eval_easing(easing, t);
            a_val + (b_val - a_val) * eased
        }
        InterpolationMode::Bezier => eval_segment_bezier(
            BezierSegment {
                start_frame: a_frame,
                start_value: a_val,
                out_tangent: a_out_tangent,
                end_frame: b_frame,
                end_value: b_val,
                in_tangent: b_in_tangent,
            },
            local_frame,
        ),
    }
}

fn eval_prop_value(prop: Option<&SessionProperty>, local_frame: f64) -> Option<Value> {
    let p = prop?;
    if p.keyframes.is_empty() {
        return Some(p.default_value.clone());
    }
    if p.keyframes.len() == 1 || !p.animated {
        return Some(p.keyframes[0].value.clone());
    }
    if local_frame <= p.keyframes[0].frame + 1e-6 {
        return Some(p.keyframes[0].value.clone());
    }
    let last = &p.keyframes[p.keyframes.len() - 1];
    if local_frame >= last.frame - 1e-6 {
        return Some(last.value.clone());
    }

    for pair in p.keyframes.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if local_frame >= a.frame - 1e-6 && local_frame <= b.frame + 1e-6 {
            let (mode, _) = InterpolationMode::from_cm3_tween_code(b.tween);
            if mode == InterpolationMode::Discrete {
                return Some(a.value.clone());
            }
            match (&a.value, &b.value) {
                (Value::Number(na), Value::Number(nb)) => {
                    if let (Some(va), Some(vb)) = (na.as_f64(), nb.as_f64()) {
                        let v = eval_scalar_segment(
                            a.frame,
                            va,
                            a.control_points[1],
                            b.frame,
                            vb,
                            b.control_points[0],
                            b.tween,
                            local_frame,
                        );
                        return Some(json!(v));
                    }
                }
                (Value::Array(aa), Value::Array(ab)) if aa.len() == ab.len() => {
                    let mut out = Vec::with_capacity(aa.len());
                    let mut all_num = true;
                    for (ea, eb) in aa.iter().zip(ab.iter()) {
                        if let (Some(va), Some(vb)) = (ea.as_f64(), eb.as_f64()) {
                            let v = eval_scalar_segment(
                                a.frame,
                                va,
                                a.control_points[1],
                                b.frame,
                                vb,
                                b.control_points[0],
                                b.tween,
                                local_frame,
                            );
                            out.push(json!(v));
                        } else {
                            all_num = false;
                            break;
                        }
                    }
                    if all_num {
                        return Some(Value::Array(out));
                    }
                }
                _ => {}
            }
            return Some(a.value.clone());
        }
    }

    Some(last.value.clone())
}

fn eval_prop_f64(prop: Option<&SessionProperty>, local_frame: f64, fallback: f64) -> f64 {
    eval_prop_value(prop, local_frame)
        .and_then(|v| v.as_f64())
        .unwrap_or(fallback)
}

fn eval_prop_u32(prop: Option<&SessionProperty>, local_frame: f64, fallback: u32) -> u32 {
    eval_prop_value(prop, local_frame)
        .and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_f64().map(|f| f.round().max(0.0) as u64))
                .or_else(|| v.as_bool().map(u64::from))
        })
        .map_or(fallback, |v| v as u32)
}

fn eval_prop_vec2(
    prop: Option<&SessionProperty>,
    local_frame: f64,
    fallback: [f64; 2],
) -> [f64; 2] {
    eval_prop_value(prop, local_frame)
        .and_then(|v| {
            let a = v.as_array()?;
            if a.len() >= 2 {
                Some([
                    a[0].as_f64().unwrap_or(fallback[0]),
                    a[1].as_f64().unwrap_or(fallback[1]),
                ])
            } else {
                None
            }
        })
        .unwrap_or(fallback)
}

fn eval_prop_vec3(
    prop: Option<&SessionProperty>,
    local_frame: f64,
    fallback: [f64; 3],
) -> [f64; 3] {
    eval_prop_value(prop, local_frame)
        .and_then(|v| {
            let a = v.as_array()?;
            if a.len() >= 3 {
                Some([
                    a[0].as_f64().unwrap_or(fallback[0]),
                    a[1].as_f64().unwrap_or(fallback[1]),
                    a[2].as_f64().unwrap_or(fallback[2]),
                ])
            } else {
                None
            }
        })
        .unwrap_or(fallback)
}

fn eval_prop_vec4(
    prop: Option<&SessionProperty>,
    local_frame: f64,
    fallback: [f64; 4],
) -> [f64; 4] {
    eval_prop_value(prop, local_frame)
        .and_then(|v| {
            let a = v.as_array()?;
            if a.len() >= 4 {
                Some([
                    a[0].as_f64().unwrap_or(fallback[0]),
                    a[1].as_f64().unwrap_or(fallback[1]),
                    a[2].as_f64().unwrap_or(fallback[2]),
                    a[3].as_f64().unwrap_or(fallback[3]),
                ])
            } else {
                None
            }
        })
        .unwrap_or(fallback)
}

fn eval_prop_str(prop: Option<&SessionProperty>, local_frame: f64, fallback: &str) -> String {
    eval_prop_value(prop, local_frame)
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_else(|| fallback.to_owned())
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

fn normalized_kf_frame(session: &Session, local_frame: f64) -> f64 {
    if session.playback.speed.abs() > 1e-6 {
        local_frame
    } else {
        local_frame.round()
    }
}

#[allow(clippy::too_many_arguments)]
fn panel_property_row<R>(
    ui: &mut Ui,
    label: &str,
    reserve_stopwatch_gutter: bool,
    state: &PanelsState,
    theme: &Cm3Theme,
    left_extra: impl FnOnce(&mut Ui),
    right_value: impl FnOnce(&mut Ui) -> R,
) -> R {
    if state.inspector_root_indent {
        cm3_property_row(
            ui,
            label,
            state.prop_columns,
            1,
            theme,
            left_extra,
            right_value,
        )
    } else {
        cm3_property_row_aligned(
            ui,
            label,
            reserve_stopwatch_gutter,
            state.prop_columns,
            theme,
            left_extra,
            right_value,
        )
    }
}

fn resolve_kf_write_frame(
    session: &Session,
    prop: Option<&SessionProperty>,
    local_frame: f64,
) -> f64 {
    if let Some(p) = prop
        && let Some(kf) = p
            .keyframes
            .iter()
            .find(|k| (k.frame - local_frame).abs() < 1e-4)
    {
        return kf.frame;
    }
    normalized_kf_frame(session, local_frame)
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
    actions: &mut Vec<PanelAction>,
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
        let exact_kf = prop.and_then(|p| {
            p.keyframes
                .iter()
                .find(|k| (k.frame - local_frame).abs() < 1e-4)
        });
        if let Some(kf) = exact_kf {
            let _ = session.execute(Command::DeleteKeyframe {
                target_id: target_id.to_owned(),
                property: prop_name.to_owned(),
                frame: kf.frame,
            });
        } else {
            let _ = session.execute(Command::SetKeyframe {
                target_id: target_id.to_owned(),
                property: prop_name.to_owned(),
                frame: normalized_kf_frame(session, local_frame),
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
        let toggled = if (kf.tween & 0xFF) == 0 {
            (1 << 8) | 1
        } else {
            kf.tween ^ (1 << 8)
        };
        let _ = session.execute(Command::SetKeyframe {
            target_id: target_id.to_owned(),
            property: prop_name.to_owned(),
            frame: kf.frame,
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
        let new_tween = if ease_idx == 0 {
            0
        } else {
            (kf.tween & 0xFF00) | u16::from(ease_idx)
        };
        let _ = session.execute(Command::SetKeyframe {
            target_id: target_id.to_owned(),
            property: prop_name.to_owned(),
            frame: kf.frame,
            value: kf.value.clone(),
            tween: Some(new_tween),
        });
    }
    if ev.open_easing_plus {
        actions.push(PanelAction::OpenGraphForProperty {
            target_id: target_id.to_owned(),
            property: prop_name.to_owned(),
        });
    }
}

fn execute_prop_value_command(
    session: &mut Session,
    target_id: &str,
    prop_name: &str,
    animated: bool,
    local_frame: f64,
    value: Value,
) {
    let cmd = if animated {
        let prop = lookup_session_prop(session, target_id, prop_name);
        Command::SetKeyframe {
            target_id: target_id.to_owned(),
            property: prop_name.to_owned(),
            frame: resolve_kf_write_frame(session, prop.as_ref(), local_frame),
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

#[allow(clippy::too_many_arguments)]
fn apply_scrub_to_session(
    session: &mut Session,
    target_id: &str,
    prop_name: &str,
    animated: bool,
    local_frame: f64,
    value: Value,
    ev: ScrubEvent,
    state: &mut PanelsState,
) {
    apply_scrub_to_session_with_owner_key(
        session,
        target_id,
        prop_name,
        prop_name,
        animated,
        local_frame,
        value,
        ev,
        state,
    );
}

#[allow(clippy::too_many_arguments)]
fn apply_scrub_to_session_with_owner_key(
    session: &mut Session,
    target_id: &str,
    prop_name: &str,
    tx_owner_key: &str,
    animated: bool,
    local_frame: f64,
    value: Value,
    ev: ScrubEvent,
    state: &mut PanelsState,
) {
    let owns_tx = owns_inspector_tx(state, target_id, tx_owner_key);
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
                execute_prop_value_command(
                    session,
                    target_id,
                    prop_name,
                    animated,
                    local_frame,
                    value,
                );
                state.inspector_tx_has_edits = true;
            }
            if session.history.is_in_transaction() {
                let _ = session.commit_transaction(true);
            }
            clear_inspector_tx_state(state);
        } else {
            close_existing_tx_before_start(session, state);
            if ev.changed {
                execute_prop_value_command(
                    session,
                    target_id,
                    prop_name,
                    animated,
                    local_frame,
                    value,
                );
            }
        }
        begin_inspector_tx(
            session,
            state,
            target_id,
            tx_owner_key,
            format!("edit {prop_name}"),
        );
        return;
    }
    if ev.started && !ev.committed {
        close_existing_tx_before_start(session, state);
        begin_inspector_tx(
            session,
            state,
            target_id,
            tx_owner_key,
            format!("edit {prop_name}"),
        );
    }
    let owns_tx = owns_inspector_tx(state, target_id, tx_owner_key);
    if ev.changed && ev.committed && !ev.started && !owns_tx {
        run_atomic_commit_outside_foreign_tx(session, state, |session| {
            execute_prop_value_command(session, target_id, prop_name, animated, local_frame, value);
        });
        return;
    }
    if ev.changed || (ev.committed && owns_tx) {
        execute_prop_value_command(session, target_id, prop_name, animated, local_frame, value);
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

/// Applies color-picker edits lazily: never opens a session undo transaction on popup open
/// (`cev.started`) before the first actual `cev.changed`, and guards cancel/commit by tx ownership.
#[allow(clippy::too_many_arguments)]
fn apply_color_picker_to_session(
    session: &mut Session,
    target_id: &str,
    prop_name: &str,
    animated: bool,
    local_frame: f64,
    value: Value,
    cev: super::controls::ColorPickerEvent,
    state: &mut PanelsState,
) {
    let owns_tx = owns_inspector_tx(state, target_id, prop_name);
    if cev.cancelled {
        if owns_tx {
            if session.history.is_in_transaction() {
                let _ = session.cancel_transaction();
            }
            clear_inspector_tx_state(state);
        }
        return;
    }
    if cev.changed && !cev.committed && !owns_tx {
        close_existing_tx_before_start(session, state);
        begin_inspector_tx(
            session,
            state,
            target_id,
            prop_name,
            format!("edit {prop_name}"),
        );
    }
    let owns_tx = owns_inspector_tx(state, target_id, prop_name);
    if cev.changed {
        if cev.committed && !owns_tx {
            run_atomic_commit_outside_foreign_tx(session, state, |session| {
                execute_prop_value_command(
                    session,
                    target_id,
                    prop_name,
                    animated,
                    local_frame,
                    value,
                );
            });
        } else {
            execute_prop_value_command(session, target_id, prop_name, animated, local_frame, value);
            if owns_tx {
                state.inspector_tx_has_edits = true;
            }
        }
    }
    if cev.committed && owns_tx {
        if session.history.is_in_transaction() {
            let _ = session.commit_transaction(true);
        }
        clear_inspector_tx_state(state);
    }
}

fn show_expression_inline_chip(
    ui: &mut Ui,
    prop: Option<&SessionProperty>,
    theme: &Cm3Theme,
) -> bool {
    let mut open_expr = false;
    if let Some(expr) = prop.and_then(|p| p.expression.as_deref()) {
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
            ui.add(
                egui::Label::new(
                    egui::RichText::new(expr)
                        .font(theme.font(FONT_COMPACT))
                        .color(TEXT_MUTED),
                )
                .truncate(),
            );
        });
    }
    open_expr
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
    actions: &mut Vec<PanelAction>,
) {
    touch_inspector_tx_owner(state, target_id, prop_key);
    let prop = lookup_session_prop(session, target_id, prop_key);
    let animated = prop.as_ref().is_some_and(|p| p.animated);
    let existing_expr = prop.as_ref().and_then(|p| p.expression.clone());
    let has_expr = existing_expr.is_some();
    // Clip `time` initializes as a non-animated 2-keyframe linear ramp [0 -> start, length -> end].
    let mut val = if prop_key == "time"
        && let Some(p) = prop.as_ref()
        && !p.animated
        && p.keyframes.len() == 2
        && let (Some(v0), Some(v1)) = (p.keyframes[0].value.as_f64(), p.keyframes[1].value.as_f64())
    {
        let f0 = p.keyframes[0].frame;
        let f1 = p.keyframes[1].frame;
        let span = (f1 - f0).max(1e-6);
        let t = ((local_frame - f0) / span).clamp(0.0, 1.0);
        v0 + (v1 - v0) * t
    } else {
        eval_prop_f64(prop.as_ref(), local_frame, default_val)
    };
    let easing_plus = session.plugins.is_enabled("easing-plus");

    let mut sw_ev = Default::default();
    let mut kf_ev = KeyframeBarEvent::default();
    let mut open_expr = false;

    let scrub_ev = panel_property_row(
        ui,
        label,
        false,
        state,
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
            if has_expr && show_expression_inline_chip(ui, prop.as_ref(), theme) {
                open_expr = true;
            }
            ev
        },
    );

    handle_stopwatch_event(
        session,
        target_id,
        prop_key,
        existing_expr.as_deref(),
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
            existing_expr.unwrap_or_else(|| format!("{val}")),
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
        actions,
    );
    apply_scrub_to_session(
        session,
        target_id,
        prop_key,
        animated,
        local_frame,
        json!(val),
        scrub_ev,
        state,
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
    actions: &mut Vec<PanelAction>,
) {
    touch_inspector_tx_owner(state, target_id, prop_key);
    let prop = lookup_session_prop(session, target_id, prop_key);
    let animated = prop.as_ref().is_some_and(|p| p.animated);
    let existing_expr = prop.as_ref().and_then(|p| p.expression.clone());
    let has_expr = existing_expr.is_some();
    let mut val = eval_prop_vec2(prop.as_ref(), local_frame, default_val);
    let easing_plus = session.plugins.is_enabled("easing-plus");
    let expand_key = (target_id.to_owned(), prop_key.to_owned());
    let mut expanded = state.expanded_vec_props.contains(&expand_key);

    let mut sw_ev = Default::default();
    let mut kf_ev = KeyframeBarEvent::default();
    let mut open_expr = false;

    let scrub_ev = panel_property_row(
        ui,
        label,
        false,
        state,
        theme,
        |ui| {
            // Paint interactive 7x7 tree box 14px left of stopwatch (zero shift to stopwatch!)
            let box_center = Pos2::new(ui.cursor().left() - 10.5, ui.max_rect().center().y + 2.0);
            let box_rect = Rect::from_center_size(box_center, Vec2::splat(13.0));
            let box_resp = ui
                .interact(
                    box_rect,
                    Id::unique((target_id, prop_key, "vec_exp")),
                    Sense::click(),
                )
                .on_hover_cursor(CursorIcon::PointingHand);
            paint_cm3_tree_box(
                ui.painter(),
                box_center,
                expanded,
                if box_resp.hovered() {
                    TEXT_BRIGHT
                } else {
                    TEXT_MUTED
                },
            );
            if box_resp.clicked() {
                expanded = !expanded;
            }
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
            let ev = cm3_vector2(
                ui,
                Id::unique((target_id, prop_key)),
                &mut val,
                config.with_read_only(config.read_only || has_expr),
                link_ratio,
                theme,
            );
            if has_expr && show_expression_inline_chip(ui, prop.as_ref(), theme) {
                open_expr = true;
            }
            ev
        },
    );

    if expanded {
        state.expanded_vec_props.insert(expand_key);
    } else {
        state.expanded_vec_props.remove(&expand_key);
    }

    handle_stopwatch_event(
        session,
        target_id,
        prop_key,
        existing_expr.as_deref(),
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
            existing_expr.unwrap_or_else(|| json!(val).to_string()),
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
        actions,
    );
    apply_scrub_to_session(
        session,
        target_id,
        prop_key,
        animated,
        local_frame,
        json!(val),
        scrub_ev,
        state,
    );

    // Expanded individual X / Y axis rows
    if expanded {
        for (axis, axis_lbl) in [(0usize, "X"), (1usize, "Y")] {
            let axis_owner_key = format!("{prop_key}::axis{axis}");
            touch_inspector_tx_owner(state, target_id, &axis_owner_key);
            let ev = panel_property_row(
                ui,
                axis_lbl,
                true,
                state,
                theme,
                |ui| {
                    ui.add_space(15.0);
                },
                |ui| {
                    cm3_scrub_number(
                        ui,
                        Id::unique((target_id, prop_key, "sub_axis", axis)),
                        &mut val[axis],
                        config.with_read_only(config.read_only || has_expr),
                        theme,
                    )
                },
            );
            apply_scrub_to_session_with_owner_key(
                session,
                target_id,
                prop_key,
                &axis_owner_key,
                animated,
                local_frame,
                json!(val),
                ev,
                state,
            );
        }
    }
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
    actions: &mut Vec<PanelAction>,
) {
    touch_inspector_tx_owner(state, target_id, prop_key);
    let prop = lookup_session_prop(session, target_id, prop_key);
    let animated = prop.as_ref().is_some_and(|p| p.animated);
    let existing_expr = prop.as_ref().and_then(|p| p.expression.clone());
    let has_expr = existing_expr.is_some();
    let mut val = eval_prop_vec3(prop.as_ref(), local_frame, default_val);
    let easing_plus = session.plugins.is_enabled("easing-plus");
    let expand_key = (target_id.to_owned(), prop_key.to_owned());
    let mut expanded = state.expanded_vec_props.contains(&expand_key);

    let mut sw_ev = Default::default();
    let mut kf_ev = KeyframeBarEvent::default();
    let mut open_expr = false;

    let scrub_ev = panel_property_row(
        ui,
        label,
        false,
        state,
        theme,
        |ui| {
            // Paint interactive 7x7 tree box 14px left of stopwatch (zero shift to stopwatch!)
            let box_center = Pos2::new(ui.cursor().left() - 10.5, ui.max_rect().center().y + 2.0);
            let box_rect = Rect::from_center_size(box_center, Vec2::splat(13.0));
            let box_resp = ui
                .interact(
                    box_rect,
                    Id::unique((target_id, prop_key, "vec_exp")),
                    Sense::click(),
                )
                .on_hover_cursor(CursorIcon::PointingHand);
            paint_cm3_tree_box(
                ui.painter(),
                box_center,
                expanded,
                if box_resp.hovered() {
                    TEXT_BRIGHT
                } else {
                    TEXT_MUTED
                },
            );
            if box_resp.clicked() {
                expanded = !expanded;
            }
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
            let ev = cm3_vector3(
                ui,
                Id::unique((target_id, prop_key)),
                &mut val,
                config.with_read_only(config.read_only || has_expr),
                link_ratio,
                theme,
            );
            if has_expr && show_expression_inline_chip(ui, prop.as_ref(), theme) {
                open_expr = true;
            }
            ev
        },
    );

    if expanded {
        state.expanded_vec_props.insert(expand_key);
    } else {
        state.expanded_vec_props.remove(&expand_key);
    }

    handle_stopwatch_event(
        session,
        target_id,
        prop_key,
        existing_expr.as_deref(),
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
            existing_expr.unwrap_or_else(|| json!(val).to_string()),
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
        actions,
    );
    apply_scrub_to_session(
        session,
        target_id,
        prop_key,
        animated,
        local_frame,
        json!(val),
        scrub_ev,
        state,
    );

    // Expanded individual X / Y / Z axis rows
    if expanded {
        for (axis, axis_lbl) in [(0usize, "X"), (1usize, "Y"), (2usize, "Z")] {
            let axis_owner_key = format!("{prop_key}::axis{axis}");
            touch_inspector_tx_owner(state, target_id, &axis_owner_key);
            let ev = panel_property_row(
                ui,
                axis_lbl,
                true,
                state,
                theme,
                |ui| {
                    ui.add_space(15.0);
                },
                |ui| {
                    cm3_scrub_number(
                        ui,
                        Id::unique((target_id, prop_key, "sub_axis", axis)),
                        &mut val[axis],
                        config.with_read_only(config.read_only || has_expr),
                        theme,
                    )
                },
            );
            apply_scrub_to_session_with_owner_key(
                session,
                target_id,
                prop_key,
                &axis_owner_key,
                animated,
                local_frame,
                json!(val),
                ev,
                state,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn show_session_vec4_prop(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    label: &str,
    default_val: [f64; 4],
    config: ScrubConfig,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    show_session_vec4_prop_ex(
        ui,
        session,
        target_id,
        prop_key,
        label,
        default_val,
        config,
        true,
        local_frame,
        clip_start,
        viewport_rect,
        state,
        theme,
        actions,
    );
}

#[allow(clippy::too_many_arguments)]
fn show_session_vec4_prop_ex(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    label: &str,
    default_val: [f64; 4],
    config: ScrubConfig,
    animatable: bool,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    touch_inspector_tx_owner(state, target_id, prop_key);
    let prop = lookup_session_prop(session, target_id, prop_key);
    let animated = animatable && prop.as_ref().is_some_and(|p| p.animated);
    let existing_expr = prop.as_ref().and_then(|p| p.expression.clone());
    let has_expr = animatable && existing_expr.is_some();
    let mut val = eval_prop_vec4(prop.as_ref(), local_frame, default_val);
    let easing_plus = session.plugins.is_enabled("easing-plus");

    let mut sw_ev = Default::default();
    let mut kf_ev = KeyframeBarEvent::default();
    let mut open_expr = false;

    let scrub_ev = panel_property_row(
        ui,
        label,
        !animatable,
        state,
        theme,
        |ui| {
            if animatable {
                sw_ev = cm3_stopwatch_button(ui, prop_key, animated, has_expr, theme);
            }
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
            let cfg = config.with_read_only(config.read_only || has_expr);
            let mut total = ScrubEvent::default();
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                for (axis, slot) in val.iter_mut().enumerate() {
                    if axis > 0 {
                        ui.label(
                            egui::RichText::new(",")
                                .font(theme.font(FONT_BODY))
                                .color(TEXT_DISABLED),
                        );
                    }
                    let ev = cm3_scrub_number(
                        ui,
                        Id::unique((target_id, prop_key, axis)),
                        slot,
                        cfg,
                        theme,
                    );
                    total = total.merge(ev);
                }
            });
            if has_expr && show_expression_inline_chip(ui, prop.as_ref(), theme) {
                open_expr = true;
            }
            total
        },
    );

    if animatable {
        handle_stopwatch_event(
            session,
            target_id,
            prop_key,
            existing_expr.as_deref(),
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
                existing_expr.unwrap_or_else(|| json!(val).to_string()),
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
            actions,
        );
    }
    apply_scrub_to_session(
        session,
        target_id,
        prop_key,
        animated,
        local_frame,
        json!(val),
        scrub_ev,
        state,
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
    touch_inspector_tx_owner(state, target_id, prop_key);
    let prop = lookup_session_prop(session, target_id, prop_key);
    let animated = prop.as_ref().is_some_and(|p| p.animated);
    let existing_expr = prop.as_ref().and_then(|p| p.expression.clone());
    let has_expr = existing_expr.is_some();
    let rgb_f64 = eval_prop_vec3(prop.as_ref(), local_frame, default_rgb);
    let mut rgba = Rgba::new(rgb_f64[0] as f32, rgb_f64[1] as f32, rgb_f64[2] as f32, 1.0);
    let easing_plus = session.plugins.is_enabled("easing-plus");

    let mut sw_ev = Default::default();
    let mut kf_ev = KeyframeBarEvent::default();
    let mut open_expr = false;

    let cev = panel_property_row(
        ui,
        label,
        false,
        state,
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
            let ev = cm3_hsva_color_picker(
                ui,
                Id::unique((target_id, prop_key)),
                &mut rgba,
                false,
                has_expr,
                viewport_rect,
                theme,
            );
            if has_expr && show_expression_inline_chip(ui, prop.as_ref(), theme) {
                open_expr = true;
            }
            ev
        },
    );

    let out_val = json!([
        f64::from(rgba.0[0]),
        f64::from(rgba.0[1]),
        f64::from(rgba.0[2])
    ]);
    apply_color_picker_to_session(
        session,
        target_id,
        prop_key,
        animated,
        local_frame,
        out_val.clone(),
        cev,
        state,
    );
    handle_stopwatch_event(
        session,
        target_id,
        prop_key,
        existing_expr.as_deref(),
        local_frame,
        out_val.clone(),
        sw_ev,
        state,
        viewport_rect,
    );
    if open_expr {
        open_expression_window(
            target_id,
            prop_key,
            existing_expr.unwrap_or_else(|| out_val.to_string()),
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
        out_val,
        kf_ev,
        actions,
    );
    if cev.eyedropper_clicked {
        actions.push(PanelAction::EyedropperActivate {
            target_id: target_id.to_owned(),
            property: prop_key.to_owned(),
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn show_session_color_rgba_prop(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    label: &str,
    default_rgba: [f64; 4],
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    touch_inspector_tx_owner(state, target_id, prop_key);
    let prop = lookup_session_prop(session, target_id, prop_key);
    let animated = prop.as_ref().is_some_and(|p| p.animated);
    let existing_expr = prop.as_ref().and_then(|p| p.expression.clone());
    let has_expr = existing_expr.is_some();
    let rgba_f64 = eval_prop_vec4(prop.as_ref(), local_frame, default_rgba);
    let mut rgba = Rgba::new(
        rgba_f64[0] as f32,
        rgba_f64[1] as f32,
        rgba_f64[2] as f32,
        rgba_f64[3] as f32,
    );
    let easing_plus = session.plugins.is_enabled("easing-plus");

    let mut sw_ev = Default::default();
    let mut kf_ev = KeyframeBarEvent::default();
    let mut open_expr = false;

    let cev = panel_property_row(
        ui,
        label,
        false,
        state,
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
            let ev = cm3_hsva_color_picker(
                ui,
                Id::unique((target_id, prop_key)),
                &mut rgba,
                true,
                has_expr,
                viewport_rect,
                theme,
            );
            if has_expr && show_expression_inline_chip(ui, prop.as_ref(), theme) {
                open_expr = true;
            }
            ev
        },
    );

    let out_val = json!([
        f64::from(rgba.0[0]),
        f64::from(rgba.0[1]),
        f64::from(rgba.0[2]),
        f64::from(rgba.0[3])
    ]);
    apply_color_picker_to_session(
        session,
        target_id,
        prop_key,
        animated,
        local_frame,
        out_val.clone(),
        cev,
        state,
    );
    handle_stopwatch_event(
        session,
        target_id,
        prop_key,
        existing_expr.as_deref(),
        local_frame,
        out_val.clone(),
        sw_ev,
        state,
        viewport_rect,
    );
    if open_expr {
        open_expression_window(
            target_id,
            prop_key,
            existing_expr.unwrap_or_else(|| out_val.to_string()),
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
        out_val,
        kf_ev,
        actions,
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
    actions: &mut Vec<PanelAction>,
) {
    show_session_option_prop_ex(
        ui,
        session,
        target_id,
        prop_key,
        label,
        options,
        default_idx,
        true,
        local_frame,
        clip_start,
        viewport_rect,
        state,
        theme,
        actions,
    );
}

#[allow(clippy::too_many_arguments)]
fn show_session_option_prop_ex(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    label: &str,
    options: &[&str],
    default_idx: u32,
    animatable: bool,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let prop = lookup_session_prop(session, target_id, prop_key);
    let animated = animatable && prop.as_ref().is_some_and(|p| p.animated);
    let existing_expr = prop.as_ref().and_then(|p| p.expression.clone());
    let has_expr = animatable && existing_expr.is_some();
    let mut idx = eval_prop_u32(prop.as_ref(), local_frame, default_idx) as usize;
    let mut sw_ev = Default::default();
    let mut kf_ev = KeyframeBarEvent::default();
    let mut open_expr = false;

    let changed = panel_property_row(
        ui,
        label,
        !animatable,
        state,
        theme,
        |ui| {
            if animatable {
                sw_ev = cm3_stopwatch_button(ui, prop_key, animated, has_expr, theme);
            }
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
            let ch = cm3_dropdown(
                ui,
                Id::unique((target_id, prop_key)),
                &mut idx,
                options,
                has_expr,
                viewport_rect,
                theme,
            );
            if has_expr && show_expression_inline_chip(ui, prop.as_ref(), theme) {
                open_expr = true;
            }
            ch
        },
    );

    if animatable {
        handle_stopwatch_event(
            session,
            target_id,
            prop_key,
            existing_expr.as_deref(),
            local_frame,
            json!(idx),
            sw_ev,
            state,
            viewport_rect,
        );
        if open_expr {
            open_expression_window(
                target_id,
                prop_key,
                existing_expr.unwrap_or_else(|| json!(idx).to_string()),
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
            json!(idx),
            kf_ev,
            actions,
        );
    }
    if changed {
        let cmd = if animated {
            Command::SetKeyframe {
                target_id: target_id.to_owned(),
                property: prop_key.to_owned(),
                frame: resolve_kf_write_frame(session, prop.as_ref(), local_frame),
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

#[derive(Debug, Clone, Default)]
struct CurveEditorUiState {
    channel_idx: usize,
    dragging_pt: Option<usize>,
    drag_changed: bool,
}

#[allow(clippy::too_many_arguments)]
fn show_session_curve_prop(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    label: &str,
    default_curve: &Value,
    local_frame: f64,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    touch_inspector_tx_owner(state, target_id, prop_key);
    const CHANNELS: [(&str, &str, Color32); 5] = [
        ("rgb", "RGB", Color32::from_rgb(0xdc, 0xdc, 0xdc)),
        ("r", "R", Color32::from_rgb(0xe8, 0x62, 0x62)),
        ("g", "G", Color32::from_rgb(0x6b, 0xd4, 0x7e)),
        ("b", "B", Color32::from_rgb(0x62, 0x9b, 0xf2)),
        ("a", "A", Color32::from_rgb(0xb8, 0xb8, 0xb8)),
    ];

    let prop = lookup_session_prop(session, target_id, prop_key);
    let mut curve_val = eval_prop_value(prop.as_ref(), local_frame)
        .filter(|v| v.is_object())
        .unwrap_or_else(|| default_curve.clone());
    let id = Id::unique((target_id, prop_key, "curve_ed"));
    let mut ui_st: CurveEditorUiState = ui
        .data_mut(|d| d.get_temp::<CurveEditorUiState>(id))
        .unwrap_or_default();
    if ui_st.channel_idx >= CHANNELS.len() {
        ui_st.channel_idx = 0;
    }

    let (ch_key, _, ch_color) = CHANNELS[ui_st.channel_idx];
    let mut pts: Vec<[f64; 2]> = curve_val
        .get(ch_key)
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    let a = p.as_array()?;
                    Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?])
                })
                .collect()
        })
        .filter(|v: &Vec<[f64; 2]>| v.len() >= 2)
        .unwrap_or_else(|| vec![[0.0, 0.0], [1.0, 1.0]]);
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]));

    let mut changed = false;
    let mut committed = false;
    let mut channel_switched_during_drag = false;

    panel_property_row(
        ui,
        label,
        true,
        state,
        theme,
        |_| {},
        |ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 4.0;
                // Channel selector tabs + Reset button
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    for (idx, (_, title, col)) in CHANNELS.iter().enumerate() {
                        let active = idx == ui_st.channel_idx;
                        let (r, resp) =
                            ui.allocate_exact_size(Vec2::new(28.0, 20.0), Sense::click());
                        ui.painter().rect_filled(
                            r,
                            CornerRadius::ZERO,
                            if active { theme.accent() } else { BG_INPUT },
                        );
                        ui.painter().rect_stroke(
                            r,
                            CornerRadius::ZERO,
                            Stroke::new(1.0, if active { *col } else { BORDER_CONTROL }),
                            StrokeKind::Inside,
                        );
                        ui.painter().text(
                            r.center(),
                            Align2::CENTER_CENTER,
                            *title,
                            theme.font(FONT_COMPACT),
                            if active { TEXT_BRIGHT } else { *col },
                        );
                        if resp.on_hover_cursor(CursorIcon::PointingHand).clicked() {
                            if ui_st.dragging_pt.is_some() {
                                channel_switched_during_drag = true;
                            }
                            ui_st.channel_idx = idx;
                            ui_st.dragging_pt = None;
                            ui_st.drag_changed = false;
                        }
                    }
                    if cm3_icon_button(
                        ui,
                        Cm3Icon::Reset,
                        14.0,
                        TEXT_MUTED,
                        "reset channel curve",
                        theme,
                    ) {
                        pts = vec![[0.0, 0.0], [1.0, 1.0]];
                        changed = true;
                        committed = true;
                    }
                });

                // Interactive curve canvas
                let canvas_w = ui.available_width().clamp(120.0, 200.0);
                let (rect, resp) =
                    ui.allocate_exact_size(Vec2::new(canvas_w, 96.0), Sense::click_and_drag());
                ui.painter().rect_filled(rect, CornerRadius::ZERO, BG_INPUT);
                ui.painter().rect_stroke(
                    rect,
                    CornerRadius::ZERO,
                    Stroke::new(1.0, BORDER_CONTROL),
                    StrokeKind::Inside,
                );
                // 4x4 grid + identity diagonal
                let grid_stroke = Stroke::new(1.0, Color32::from_rgb(0x2c, 0x2c, 0x2c));
                for i in 1..4 {
                    let frac = i as f32 / 4.0;
                    let gx = rect.left() + rect.width() * frac;
                    let gy = rect.top() + rect.height() * frac;
                    ui.painter().line_segment(
                        [Pos2::new(gx, rect.top()), Pos2::new(gx, rect.bottom())],
                        grid_stroke,
                    );
                    ui.painter().line_segment(
                        [Pos2::new(rect.left(), gy), Pos2::new(rect.right(), gy)],
                        grid_stroke,
                    );
                }
                ui.painter().line_segment(
                    [rect.left_bottom(), rect.right_top()],
                    Stroke::new(1.0, Color32::from_rgb(0x3a, 0x3a, 0x3a)),
                );

                let to_screen = |p: [f64; 2]| -> Pos2 {
                    Pos2::new(
                        rect.left() + (p[0].clamp(0.0, 1.0) as f32) * rect.width(),
                        rect.bottom() - (p[1].clamp(0.0, 1.0) as f32) * rect.height(),
                    )
                };
                let from_screen = |pos: Pos2| -> [f64; 2] {
                    [
                        f64::from(((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0)),
                        f64::from(((rect.bottom() - pos.y) / rect.height()).clamp(0.0, 1.0)),
                    ]
                };

                // Handle pointer interactions (drag existing point, click to add, right-click to remove)
                if resp.drag_started()
                    && let Some(pos) = ui.input(|i| i.pointer.interact_pos())
                    && let Some((best_i, _)) = pts
                        .iter()
                        .enumerate()
                        .map(|(i, p)| (i, to_screen(*p).distance(pos)))
                        .filter(|(_, d)| *d <= 10.0)
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                {
                    ui_st.dragging_pt = Some(best_i);
                    ui_st.drag_changed = false;
                }
                if resp.dragged()
                    && let Some(pt_idx) = ui_st.dragging_pt
                    && let Some(pos) = ui.input(|i| i.pointer.interact_pos())
                    && pt_idx < pts.len()
                {
                    let mut np = from_screen(pos);
                    if pt_idx == 0 {
                        np[0] = 0.0;
                    } else if pt_idx + 1 == pts.len() {
                        np[0] = 1.0;
                    } else {
                        let min_x = (pts[pt_idx - 1][0] + 0.01).min(0.99);
                        let max_x = (pts[pt_idx + 1][0] - 0.01).max(min_x);
                        np[0] = np[0].clamp(min_x, max_x);
                    }
                    if (pts[pt_idx][0] - np[0]).abs() > 1e-4
                        || (pts[pt_idx][1] - np[1]).abs() > 1e-4
                    {
                        pts[pt_idx] = np;
                        changed = true;
                        ui_st.drag_changed = true;
                    }
                }
                if resp.drag_stopped() && ui_st.dragging_pt.is_some() {
                    if ui_st.drag_changed {
                        committed = true;
                    }
                    ui_st.dragging_pt = None;
                    ui_st.drag_changed = false;
                }
                if resp.clicked()
                    && ui_st.dragging_pt.is_none()
                    && let Some(pos) = resp.interact_pointer_pos()
                {
                    let near_existing = pts.iter().any(|p| to_screen(*p).distance(pos) <= 10.0);
                    if !near_existing && pts.len() < 16 {
                        pts.push(from_screen(pos));
                        pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
                        changed = true;
                        committed = true;
                    }
                }
                if resp.secondary_clicked()
                    && let Some(pos) = resp.interact_pointer_pos()
                    && pts.len() > 2
                    && let Some((best_i, _)) = pts
                        .iter()
                        .enumerate()
                        .skip(1)
                        .take(pts.len() - 2)
                        .map(|(i, p)| (i, to_screen(*p).distance(pos)))
                        .filter(|(_, d)| *d <= 10.0)
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                {
                    pts.remove(best_i);
                    changed = true;
                    committed = true;
                }

                // Draw curve polyline & point handles
                let line_pts: Vec<Pos2> = pts.iter().map(|p| to_screen(*p)).collect();
                ui.painter()
                    .add(Shape::line(line_pts.clone(), Stroke::new(1.5, ch_color)));
                for (i, sp) in line_pts.iter().enumerate() {
                    let h = Rect::from_center_size(*sp, Vec2::splat(7.0));
                    ui.painter().rect_filled(
                        h,
                        CornerRadius::ZERO,
                        if ui_st.dragging_pt == Some(i) {
                            TEXT_BRIGHT
                        } else {
                            ch_color
                        },
                    );
                    ui.painter().rect_stroke(
                        h,
                        CornerRadius::ZERO,
                        Stroke::new(1.0, BORDER_CONTROL),
                        StrokeKind::Inside,
                    );
                }
            });
        },
    );

    let owns_curve_tx = owns_inspector_tx(state, target_id, prop_key);
    if channel_switched_during_drag && owns_curve_tx {
        commit_inspector_tx_if_open(session, state);
    }
    let owns_curve_tx = owns_inspector_tx(state, target_id, prop_key);
    if changed && !committed && !owns_curve_tx {
        close_existing_tx_before_start(session, state);
        begin_inspector_tx(
            session,
            state,
            target_id,
            prop_key,
            format!("edit {prop_key}"),
        );
    }
    let owns_curve_tx = owns_inspector_tx(state, target_id, prop_key);
    if changed {
        if let Some(obj) = curve_val.as_object_mut() {
            obj.insert(ch_key.to_owned(), json!(pts));
        }
        let exec_curve = |session: &mut Session| {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: target_id.to_owned(),
                property: prop_key.to_owned(),
                value: curve_val,
            });
        };
        if committed && !owns_curve_tx {
            run_atomic_commit_outside_foreign_tx(session, state, exec_curve);
        } else {
            exec_curve(session);
            if owns_curve_tx {
                state.inspector_tx_has_edits = true;
            }
        }
    }
    if committed && owns_curve_tx {
        if session.history.is_in_transaction() {
            let _ = session.commit_transaction(true);
        }
        clear_inspector_tx_state(state);
    }

    ui.data_mut(|d| d.insert_temp(id, ui_st));
}

#[derive(Debug, Clone, Default)]
struct TextPropEditState {
    buf: String,
    editing: bool,
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
    let animated = prop.as_ref().is_some_and(|p| p.animated);
    let current = eval_prop_str(prop.as_ref(), local_frame, default_text);
    let state_id = Id::unique((target_id, prop_key, "text_edit_state"));
    let edit_id = Id::unique((target_id, prop_key, "text_edit"));

    let mut edit_st = ui
        .data_mut(|d| d.get_temp::<TextPropEditState>(state_id))
        .unwrap_or_else(|| TextPropEditState {
            buf: current.clone(),
            editing: false,
        });
    // Only sync from session when NOT actively editing so focus-loss frame retains typed buffer.
    if !edit_st.editing {
        edit_st.buf = current.clone();
    }

    let (esc_pressed, enter_pressed) =
        ui.input(|i| (i.key_pressed(Key::Escape), i.key_pressed(Key::Enter)));
    let mut commit = false;
    let mut cancel = false;

    panel_property_row(
        ui,
        label,
        true,
        state,
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
                        TextEdit::singleline(&mut edit_st.buf)
                            .id(edit_id)
                            .frame(Frame::NONE)
                            .font(theme.font(FONT_BODY))
                            .text_color(theme.value())
                            .desired_width(ui.available_width().min(200.0)),
                    )
                })
                .inner;

            if resp.gained_focus() {
                edit_st.editing = true;
            }
            if edit_st.editing {
                if esc_pressed {
                    cancel = true;
                    ui.memory_mut(|m| m.surrender_focus(edit_id));
                } else if resp.lost_focus() || (resp.has_focus() && enter_pressed) {
                    commit = true;
                    ui.memory_mut(|m| m.surrender_focus(edit_id));
                }
            }
        },
    );

    if cancel {
        edit_st.buf = current;
        edit_st.editing = false;
    } else if commit {
        if edit_st.buf != current {
            let cmd = if animated {
                Command::SetKeyframe {
                    target_id: target_id.to_owned(),
                    property: prop_key.to_owned(),
                    frame: resolve_kf_write_frame(session, prop.as_ref(), local_frame),
                    value: json!(edit_st.buf),
                    tween: Some(0),
                }
            } else {
                Command::SetPropertyValue {
                    target_id: target_id.to_owned(),
                    property: prop_key.to_owned(),
                    value: json!(edit_st.buf),
                }
            };
            let _ = session.execute(cmd);
        }
        edit_st.editing = false;
    }

    ui.data_mut(|d| d.insert_temp(state_id, edit_st));
}

#[allow(clippy::too_many_arguments)]
fn handle_stopwatch_event(
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    existing_expr: Option<&str>,
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
            frame: normalized_kf_frame(session, local_frame),
        });
    } else if ev.alt_expression {
        if let Some(expr_code) = existing_expr {
            // Re-open Expression Editor prefilled with existing expression (never delete on Alt+Click)
            open_expression_window(
                target_id,
                prop_key,
                expr_code.to_owned(),
                state,
                viewport_rect,
            );
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
        error_message: None,
    });
    let spec = FloatingWindowSpec::new("expression_editor", "Expression Editor")
        .with_subtitle(prop_key.to_owned())
        .with_default_size(Vec2::new(420.0, 240.0));
    state.floating_windows.open_or_focus(&spec, viewport_rect);
}

// ---------------------------------------------------------------------------
// Small UI Helpers
// ---------------------------------------------------------------------------

/// Paints a razor-sharp integer-snapped 1px border using 4 filled strips (zero AA feathering).
fn paint_crisp_panel_border(painter: &Painter, rect: Rect, color: Color32) {
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
fn paint_crisp_border_2px(painter: &Painter, rect: Rect, color: Color32) {
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
fn panel_top_header(ui: &mut Ui, title: &str, theme: &Cm3Theme) {
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
fn paint_small_chevron(painter: &Painter, center: Pos2, open: bool, color: Color32) {
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
fn humanize_prop_label(key: &str) -> String {
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
struct PropertyHeaderActions {
    reset: bool,
    duplicate: bool,
    remove: bool,
}

/// Collapsible inspector root header with Column-2-aligned Reset (`↻`), Duplicate (`❐`),
/// and Remove (`✕`) buttons matching Zoidium `11_objects_camera.png` (21px pitch).
fn property_header_with_actions(
    ui: &mut Ui,
    id: Id,
    title: &str,
    col_ratio: f32,
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

    let col2_x = (hdr_rect.left()
        + (hdr_rect.width() * col_ratio).clamp(72.0, (hdr_rect.width() - 68.0).max(72.0)))
    .round();
    let icon_y = (hdr_rect.center().y - 7.5).round();
    let reset_icon_rect = Rect::from_min_size(Pos2::new(col2_x, icon_y), Vec2::splat(15.0));
    let dup_icon_rect = Rect::from_min_size(Pos2::new(col2_x + 21.0, icon_y), Vec2::splat(15.0));
    let rm_icon_rect = Rect::from_min_size(Pos2::new(col2_x + 42.0, icon_y), Vec2::splat(15.0));

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

    let any_btn_hovered = reset_resp.hovered() || dup_resp.hovered() || rm_resp.hovered();
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

    let title_clip = Rect::from_min_max(
        Pos2::new(hdr_rect.left() + 20.0, hdr_rect.top()),
        Pos2::new(
            (col2_x - 4.0).max(hdr_rect.left() + 20.0),
            hdr_rect.bottom(),
        ),
    );
    ui.painter().with_clip_rect(title_clip).text(
        Pos2::new(hdr_rect.left() + 21.0, hdr_rect.center().y),
        Align2::LEFT_CENTER,
        title,
        theme.font(FONT_BODY),
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

    act.reset = reset_resp.clicked();
    act.duplicate = dup_resp.clicked();
    act.remove = rm_resp.clicked();
    if hdr_resp.clicked() && !act.reset && !act.duplicate && !act.remove {
        open = !open;
    }
    ui.data_mut(|d| d.insert_temp(id, open));
    (open, act)
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

#[cfg(test)]
mod tests {
    use super::*;
    use zc_session::SessionKeyframe;

    #[test]
    fn eval_prop_value_uses_incoming_b_tween_semantics() {
        let prop = SessionProperty {
            animated: true,
            default_value: json!(1.0),
            expression: None,
            keyframes: vec![
                SessionKeyframe {
                    frame: 0.0,
                    value: json!(1.0),
                    tween: 1, // Linear on k0
                    control_points: [[-10.0, 0.0], [10.0, 0.0]],
                    continuous_tangent: true,
                    value_kind: None,
                },
                SessionKeyframe {
                    frame: 20.0,
                    value: json!(0.0),
                    tween: 0, // Hold (Discrete) on k1 governs segment [0, 20]
                    control_points: [[-10.0, 0.0], [10.0, 0.0]],
                    continuous_tangent: true,
                    value_kind: None,
                },
            ],
        };
        let v = eval_prop_f64(Some(&prop), 10.0, -1.0);
        assert!(
            (v - 1.0).abs() < 1e-6,
            "expected Hold value 1.0 governed by b.tween, got {v}"
        );
    }

    #[test]
    fn apply_scrub_to_session_handles_cancel_atomic_commit_and_stale_tx() {
        let mut session = Session::new();
        let mut state = PanelsState::default();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();

        // 1. Atomic TextEdit commit (changed && committed, no prior tx) leaves no open tx
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "opacity",
            false,
            0.0,
            json!(0.5),
            ScrubEvent {
                started: false,
                changed: true,
                committed: true,
                cancelled: false,
            },
            &mut state,
        );
        assert!(!session.history.is_in_transaction());
        let prop = lookup_session_prop(&session, &clip_id, "opacity");
        assert!((eval_prop_f64(prop.as_ref(), 0.0, 1.0) - 0.5).abs() < 1e-6);

        // 2. Start + change + cancel rolls back cleanly with no leaked transaction
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "opacity",
            false,
            0.0,
            json!(0.2),
            ScrubEvent {
                started: true,
                changed: true,
                committed: false,
                cancelled: false,
            },
            &mut state,
        );
        assert!(session.history.is_in_transaction());
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "opacity",
            false,
            0.0,
            json!(0.2),
            ScrubEvent {
                started: false,
                changed: false,
                committed: false,
                cancelled: true,
            },
            &mut state,
        );
        assert!(!session.history.is_in_transaction());
        let prop = lookup_session_prop(&session, &clip_id, "opacity");
        assert!((eval_prop_f64(prop.as_ref(), 0.0, 1.0) - 0.5).abs() < 1e-6);

        // 3. Stale open transaction is closed cleanly when a new started event arrives
        session.begin_transaction("stale").unwrap();
        assert!(session.history.is_in_transaction());
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "opacity",
            false,
            0.0,
            json!(0.8),
            ScrubEvent {
                started: true,
                changed: true,
                committed: false,
                cancelled: false,
            },
            &mut state,
        );
        assert!(session.history.is_in_transaction());
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "opacity",
            false,
            0.0,
            json!(0.8),
            ScrubEvent {
                started: false,
                changed: false,
                committed: true,
                cancelled: false,
            },
            &mut state,
        );
        assert!(!session.history.is_in_transaction());
        let prop = lookup_session_prop(&session, &clip_id, "opacity");
        assert!((eval_prop_f64(prop.as_ref(), 0.0, 1.0) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn inspector_tx_ownership_guard_same_frame_vector_and_lazy_color_picker() {
        let mut session = Session::new();
        let mut state = PanelsState::default();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();

        // 1. Ownership guard: Widget A starts a tx; Widget B's cancel/commit must NOT touch it.
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "opacity",
            false,
            0.0,
            json!(0.4),
            ScrubEvent {
                started: true,
                changed: true,
                committed: false,
                cancelled: false,
            },
            &mut state,
        );
        assert!(session.history.is_in_transaction());
        assert!(owns_inspector_tx(&state, &clip_id, "opacity"));

        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "rotation",
            false,
            0.0,
            json!(10.0),
            ScrubEvent {
                started: false,
                changed: false,
                committed: false,
                cancelled: true,
            },
            &mut state,
        );
        assert!(session.history.is_in_transaction());
        assert!(owns_inspector_tx(&state, &clip_id, "opacity"));

        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "rotation",
            false,
            0.0,
            json!(10.0),
            ScrubEvent {
                started: false,
                changed: false,
                committed: true,
                cancelled: false,
            },
            &mut state,
        );
        assert!(session.history.is_in_transaction());
        assert!(owns_inspector_tx(&state, &clip_id, "opacity"));

        // Commit Widget A cleanly.
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "opacity",
            false,
            0.0,
            json!(0.4),
            ScrubEvent {
                started: false,
                changed: false,
                committed: true,
                cancelled: false,
            },
            &mut state,
        );
        assert!(!session.history.is_in_transaction());
        assert!(state.inspector_tx_owner.is_none());

        // 2. Same-frame started && changed && committed on multi-axis vector commits axis 0
        // atomically AND opens a fresh owned tx for axis 1.
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "position",
            false,
            0.0,
            json!([12.0, 0.0]),
            ScrubEvent {
                started: true,
                changed: true,
                committed: true,
                cancelled: false,
            },
            &mut state,
        );
        assert!(session.history.is_in_transaction());
        assert!(owns_inspector_tx(&state, &clip_id, "position"));
        // Cancelling axis 1's fresh drag must preserve axis 0's committed [12.0, 0.0]!
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "position",
            false,
            0.0,
            json!([12.0, 99.0]),
            ScrubEvent {
                started: false,
                changed: true,
                committed: false,
                cancelled: false,
            },
            &mut state,
        );
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "position",
            false,
            0.0,
            json!([12.0, 99.0]),
            ScrubEvent {
                started: false,
                changed: false,
                committed: false,
                cancelled: true,
            },
            &mut state,
        );
        assert!(!session.history.is_in_transaction());
        let pos = eval_prop_vec2(
            lookup_session_prop(&session, &clip_id, "position").as_ref(),
            0.0,
            [0.0, 0.0],
        );
        assert!((pos[0] - 12.0).abs() < 1e-6 && pos[1].abs() < 1e-6);

        // 3. Lazy color picker: opening unchanged popup does NOT hold an open tx, so clicking
        // another control + cancelling the unchanged picker never rolls back that control.
        apply_color_picker_to_session(
            &mut session,
            &clip_id,
            "backgroundColor",
            false,
            0.0,
            json!([0.0, 0.0, 0.0]),
            super::super::controls::ColorPickerEvent {
                started: true,
                changed: false,
                committed: false,
                cancelled: false,
                eyedropper_clicked: false,
            },
            &mut state,
        );
        assert!(!session.history.is_in_transaction());
        let _ = session.execute(Command::SetPropertyValue {
            target_id: clip_id.clone(),
            property: "opacity".to_owned(),
            value: json!(0.75),
        });
        apply_color_picker_to_session(
            &mut session,
            &clip_id,
            "backgroundColor",
            false,
            0.0,
            json!([0.0, 0.0, 0.0]),
            super::super::controls::ColorPickerEvent {
                started: false,
                changed: false,
                committed: false,
                cancelled: true,
                eyedropper_clicked: false,
            },
            &mut state,
        );
        let op = eval_prop_f64(
            lookup_session_prop(&session, &clip_id, "opacity").as_ref(),
            0.0,
            1.0,
        );
        assert!((op - 0.75).abs() < 1e-6);
    }

    #[test]
    fn kf_bar_hold_to_bezier_toggle_and_epsilon_delete() {
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        session
            .execute(Command::TogglePropertyAnimated {
                target_id: clip_id.clone(),
                property: "opacity".to_owned(),
                frame: 0.0,
            })
            .unwrap();
        session
            .execute(Command::SetKeyframe {
                target_id: clip_id.clone(),
                property: "opacity".to_owned(),
                frame: 10.0,
                value: json!(0.25),
                tween: Some(0), // Hold
            })
            .unwrap();

        let prop = lookup_session_prop(&session, &clip_id, "opacity");
        let mut actions = Vec::new();
        // Query slightly offset (5e-5, within 1e-4 but > 1e-6) and toggle Hold -> Bezier
        apply_kf_bar_event(
            &mut session,
            &clip_id,
            "opacity",
            prop.as_ref(),
            10.00005,
            0.0,
            json!(0.25),
            KeyframeBarEvent {
                toggle_interp: true,
                ..Default::default()
            },
            &mut actions,
        );
        let prop_after = lookup_session_prop(&session, &clip_id, "opacity").unwrap();
        assert_eq!(prop_after.keyframes.len(), 2, "must not duplicate keyframe");
        let kf = prop_after
            .keyframes
            .iter()
            .find(|k| (k.frame - 10.0).abs() < 1e-6)
            .unwrap();
        let (mode, _) = InterpolationMode::from_cm3_tween_code(kf.tween);
        assert_eq!(mode, InterpolationMode::Bezier);

        // Delete using slightly offset local_frame must still delete exact kf.frame
        apply_kf_bar_event(
            &mut session,
            &clip_id,
            "opacity",
            Some(&prop_after),
            10.00005,
            0.0,
            json!(0.25),
            KeyframeBarEvent {
                toggle_kf: true,
                ..Default::default()
            },
            &mut actions,
        );
        let prop_deleted = lookup_session_prop(&session, &clip_id, "opacity").unwrap();
        assert_eq!(prop_deleted.keyframes.len(), 1);
    }

    #[test]
    fn transform_wrap_has_all_three_modes_and_objects_picker_matches_zoidium() {
        assert_eq!(
            template_option_labels("transform", "wrap"),
            &["clamp", "tile", "reflect"]
        );

        let session = Session::new();
        let items = build_object_picker_items(&session);
        let core: Vec<_> = items.iter().filter(|i| i.category == "Objects").collect();
        assert_eq!(core.len(), 10);
        assert_eq!(core[0].name, "Shape");
        assert_eq!(core[1].name, "Text");
        assert_eq!(core[2].name, "Light");
        assert!(
            items
                .iter()
                .all(|i| i.id != "zoidium:geometry-plus/superquadric")
        );
    }

    #[test]
    fn alt_drag_clone_is_single_undo_transaction() {
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        session
            .execute(Command::AddObject3D {
                clip_id: clip_id.clone(),
                parent_id: None,
                kind: "box".to_owned(),
                name: Some("Box".to_owned()),
            })
            .unwrap();
        let clip = session.document().video_tracks[0].clips[0].clone();
        assert!(clip.objects_3d.len() >= 2);
        let src_id = clip.objects_3d[0].id.clone();
        let dst_id = clip.objects_3d[1].id.clone();
        let undo_before = session.history.undo_len();
        let count_before = clip.objects_3d.len();

        apply_object_drop(
            &mut session,
            &clip,
            &src_id,
            &dst_id,
            TreeDropMode::After,
            true,
        );
        assert_eq!(session.history.undo_len(), undo_before + 1);
        assert_eq!(
            session.document().video_tracks[0].clips[0].objects_3d.len(),
            count_before + 1
        );

        session.execute(Command::Undo).unwrap();
        assert_eq!(
            session.document().video_tracks[0].clips[0].objects_3d.len(),
            count_before
        );
    }

    #[test]
    fn expanded_sub_axis_text_edit_commit_preserves_parent_pending_drag_tx() {
        let mut session = Session::new();
        let mut state = PanelsState::default();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();

        // Parent inline vector row starts PendingDrag on pointer-down (0 edits so far).
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "position",
            false,
            0.0,
            json!([0.0, 0.0]),
            ScrubEvent {
                started: true,
                changed: false,
                committed: false,
                cancelled: false,
            },
            &mut state,
        );
        assert!(session.history.is_in_transaction());
        assert!(owns_inspector_tx(&state, &clip_id, "position"));
        assert!(!state.inspector_tx_has_edits);

        // Same frame: expanded X sub-axis blurs its TextEdit with a committed value.
        // Because its tx_owner_key ("position::axis0") is distinct from parent ("position"),
        // it must NOT prematurely close parent's PendingDrag tx AND must commit outside it.
        apply_scrub_to_session_with_owner_key(
            &mut session,
            &clip_id,
            "position",
            "position::axis0",
            false,
            0.0,
            json!([42.0, 0.0]),
            ScrubEvent {
                started: false,
                changed: true,
                committed: true,
                cancelled: false,
            },
            &mut state,
        );
        assert!(session.history.is_in_transaction());
        assert!(owns_inspector_tx(&state, &clip_id, "position"));
        assert!(!state.inspector_tx_has_edits);

        // Same frame: an unchanged sub-axis Y blur must also leave parent's PendingDrag tx open.
        apply_scrub_to_session_with_owner_key(
            &mut session,
            &clip_id,
            "position",
            "position::axis1",
            false,
            0.0,
            json!([42.0, 0.0]),
            ScrubEvent {
                started: false,
                changed: false,
                committed: true,
                cancelled: false,
            },
            &mut state,
        );
        assert!(session.history.is_in_transaction());
        assert!(owns_inspector_tx(&state, &clip_id, "position"));

        // Next frame: parent transitions PendingDrag -> TextEdit and cancels its tx.
        // Sub-axis X's committed value [42.0, 0.0] must be preserved!
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "position",
            false,
            0.0,
            json!([42.0, 0.0]),
            ScrubEvent {
                started: false,
                changed: false,
                committed: false,
                cancelled: true,
            },
            &mut state,
        );
        assert!(!session.history.is_in_transaction());
        assert!(state.inspector_tx_owner.is_none());
        let pos = eval_prop_vec2(
            lookup_session_prop(&session, &clip_id, "position").as_ref(),
            0.0,
            [0.0, 0.0],
        );
        assert!((pos[0] - 42.0).abs() < 1e-6 && pos[1].abs() < 1e-6);
    }

    #[test]
    fn upward_click_from_text_edit_to_pending_drag_preserves_committed_edit() {
        let mut session = Session::new();
        let mut state = PanelsState::default();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();

        // 1. Scrub TextEdit (lower widget "rotation") -> upward click onto upper widget ("opacity").
        let undo_before = session.history.undo_len();
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "opacity",
            false,
            0.0,
            json!(1.0),
            ScrubEvent {
                started: true,
                changed: false,
                committed: false,
                cancelled: false,
            },
            &mut state,
        );
        assert!(session.history.is_in_transaction());
        assert!(owns_inspector_tx(&state, &clip_id, "opacity"));

        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "rotation",
            false,
            0.0,
            json!(45.0),
            ScrubEvent {
                started: false,
                changed: true,
                committed: true,
                cancelled: false,
            },
            &mut state,
        );
        assert!(session.history.is_in_transaction());
        assert!(owns_inspector_tx(&state, &clip_id, "opacity"));
        assert!(!state.inspector_tx_has_edits);

        // Pointer-up on upper widget: PendingDrag -> TextEdit cancels opacity's tx.
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "opacity",
            false,
            0.0,
            json!(1.0),
            ScrubEvent {
                started: false,
                changed: false,
                committed: false,
                cancelled: true,
            },
            &mut state,
        );
        assert!(!session.history.is_in_transaction());
        assert_eq!(session.history.undo_len(), undo_before + 1);
        let rot = eval_prop_f64(
            lookup_session_prop(&session, &clip_id, "rotation").as_ref(),
            0.0,
            0.0,
        );
        assert!((rot - 45.0).abs() < 1e-6);

        // 2. Atomic color-picker commit (hex/swatch) while upper widget holds unedited PendingDrag.
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "opacity",
            false,
            0.0,
            json!(1.0),
            ScrubEvent {
                started: true,
                changed: false,
                committed: false,
                cancelled: false,
            },
            &mut state,
        );
        apply_color_picker_to_session(
            &mut session,
            &clip_id,
            "backgroundColor",
            false,
            0.0,
            json!([0.25, 0.5, 0.75]),
            super::super::controls::ColorPickerEvent {
                started: false,
                changed: true,
                committed: true,
                cancelled: false,
                eyedropper_clicked: false,
            },
            &mut state,
        );
        assert!(session.history.is_in_transaction());
        assert!(owns_inspector_tx(&state, &clip_id, "opacity"));
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "opacity",
            false,
            0.0,
            json!(1.0),
            ScrubEvent {
                started: false,
                changed: false,
                committed: false,
                cancelled: true,
            },
            &mut state,
        );
        assert!(!session.history.is_in_transaction());
        let bg = eval_prop_vec3(
            lookup_session_prop(&session, &clip_id, "backgroundColor").as_ref(),
            0.0,
            [0.0, 0.0, 0.0],
        );
        assert!(
            (bg[0] - 0.25).abs() < 1e-6
                && (bg[1] - 0.5).abs() < 1e-6
                && (bg[2] - 0.75).abs() < 1e-6
        );

        // 3. Paired scalar atomic commit while upper widget holds unedited PendingDrag.
        let undo_before_paired = session.history.undo_len();
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "opacity",
            false,
            0.0,
            json!(1.0),
            ScrubEvent {
                started: true,
                changed: false,
                committed: false,
                cancelled: false,
            },
            &mut state,
        );
        apply_paired_scalar_scrub(
            &mut session,
            &clip_id,
            "paired_test",
            ("propA", 3.0),
            ("propB", 7.0),
            ScrubEvent {
                started: false,
                changed: true,
                committed: true,
                cancelled: false,
            },
            &mut state,
        );
        assert!(session.history.is_in_transaction());
        assert!(owns_inspector_tx(&state, &clip_id, "opacity"));
        apply_scrub_to_session(
            &mut session,
            &clip_id,
            "opacity",
            false,
            0.0,
            json!(1.0),
            ScrubEvent {
                started: false,
                changed: false,
                committed: false,
                cancelled: true,
            },
            &mut state,
        );
        assert!(!session.history.is_in_transaction());
        assert_eq!(session.history.undo_len(), undo_before_paired + 1);
        let a = eval_prop_f64(
            lookup_session_prop(&session, &clip_id, "propA").as_ref(),
            0.0,
            0.0,
        );
        let b = eval_prop_f64(
            lookup_session_prop(&session, &clip_id, "propB").as_ref(),
            0.0,
            0.0,
        );
        assert!((a - 3.0).abs() < 1e-6 && (b - 7.0).abs() < 1e-6);
    }
}
