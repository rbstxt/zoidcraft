use super::shared_widgets::paint_crisp_panel_border;
use super::*;

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
    Gif,
    Wav,
    M4a,
}

impl ExportContainer {
    pub const ALL: [Self; 7] = [
        Self::Mp4,
        Self::WebM,
        Self::Mkv,
        Self::PngSequence,
        Self::Gif,
        Self::Wav,
        Self::M4a,
    ];
    pub const LABELS: [&'static str; 7] = [
        "MP4 (.mp4)",
        "WebM (.webm)",
        "MKV (.mkv)",
        "PNG Sequence (.png)",
        "GIF (.gif)",
        "WAV Audio (.wav)",
        "M4A Audio (.m4a)",
    ];

    /// Returns `true` for audio-only containers (`Wav`, `M4a`) that skip video rendering.
    #[must_use]
    pub const fn is_audio_only(self) -> bool {
        matches!(self, Self::Wav | Self::M4a)
    }

    /// Returns `true` when this container exposes video codec and rate-control options.
    #[must_use]
    pub const fn uses_video_codec(self) -> bool {
        matches!(self, Self::Mp4 | Self::WebM | Self::Mkv)
    }

    /// Returns `true` when this container supports an audio stream.
    #[must_use]
    pub const fn supports_audio(self) -> bool {
        !matches!(self, Self::PngSequence | Self::Gif)
    }

    /// Returns `true` when this container requires FFmpeg to encode.
    #[must_use]
    pub const fn requires_ffmpeg(self) -> bool {
        !matches!(self, Self::PngSequence | Self::Wav)
    }
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
        "H.264 / AVC",
        "H.265 / HEVC",
        "VP9",
        "VP8 (legacy)",
        "AV1",
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

/// Execution state for a queued export job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportJobStatus {
    Queued,
    Running,
    Completed,
    Failed(String),
    Cancelled,
}

/// One queued export job shown in the Export tab's multi-job queue.
#[derive(Debug, Clone)]
pub struct ExportQueueJob {
    pub id: u64,
    pub label: String,
    pub output_path: PathBuf,
    pub settings: ExportSettings,
    pub status: ExportJobStatus,
}

/// Metadata for a disk autosave recovery `.zcraft` snapshot shown in the Restore tab.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoverySnapshotInfo {
    pub filename: String,
    pub timestamp_secs: u64,
    pub size_bytes: u64,
    pub path: PathBuf,
}

/// Live FFmpeg probe & active export job telemetry shown in the Export tab.
#[derive(Debug, Clone)]
pub struct ExportStatus {
    pub ffmpeg_available: bool,
    pub ffmpeg_version: String,
    pub ffmpeg_resolved_path: Option<String>,
    pub exporting: bool,
    /// Progress fraction in `0.0..=1.0`.
    pub progress: f32,
    pub status_message: String,
    /// Measured live rendering throughput in frames/second while exporting.
    pub live_fps: Option<f64>,
    /// Estimated remaining seconds while exporting.
    pub eta_secs: Option<f64>,
    /// Output file or folder from the most recently completed export.
    pub last_completed_output: Option<PathBuf>,
}

impl Default for ExportStatus {
    fn default() -> Self {
        Self {
            ffmpeg_available: true,
            ffmpeg_version: "ready (probe on export)".to_owned(),
            ffmpeg_resolved_path: None,
            exporting: false,
            progress: 0.0,
            status_message: String::new(),
            live_fps: None,
            eta_secs: None,
            last_completed_output: None,
        }
    }
}

/// Live update-checker telemetry mirrored into the Settings tab.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateUiStatus {
    pub checking: bool,
    pub available: bool,
    pub latest_version: Option<String>,
    pub status_message: String,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for UpdateUiStatus {
    fn default() -> Self {
        Self {
            checking: false,
            available: false,
            latest_version: None,
            status_message: "Not checked yet".to_owned(),
        }
    }
}

/// Persistent editor preferences shown in the Settings tab.
#[derive(Debug, Clone, PartialEq)]
pub struct AppSettings {
    pub default_fps: f64,
    pub preview_cache_mib: u32,
    pub undo_max_entries: usize,
    pub undo_max_mib: usize,
    pub show_fps_overlay: bool,
    /// Optional user-selected `ffmpeg` executable path (`None` = auto-detect).
    pub ffmpeg_path: Option<String>,
    /// Optional user-selected `ffprobe` executable path (`None` = auto-detect / sibling).
    pub ffprobe_path: Option<String>,
    /// Suppresses the non-blocking startup FFmpeg setup banner when `true`.
    pub ffmpeg_setup_dismissed: bool,
    /// When true, checks GitHub Releases for a newer desktop version on startup.
    pub check_updates_on_startup: bool,
    /// Optional release version (e.g. `"0.2.7"`) skipped by the user for update banners.
    pub skipped_update_version: Option<String>,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            default_fps: 30.0,
            preview_cache_mib: 512,
            undo_max_entries: 200,
            undo_max_mib: 64,
            show_fps_overlay: true,
            ffmpeg_path: None,
            ffprobe_path: None,
            ffmpeg_setup_dismissed: false,
            check_updates_on_startup: true,
            skipped_update_version: None,
        }
    }
}

impl AppSettings {
    pub(crate) fn undo_limits(&self) -> (usize, usize) {
        (
            self.undo_max_entries.clamp(1, 2000),
            self.undo_max_mib.clamp(4, 1024) * 1024 * 1024,
        )
    }
}

// ---------------------------------------------------------------------------
// Actions Emitted by Side Panels to the App Shell
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectAction {
    ExportPz,
}

#[derive(Debug, Clone)]
pub enum PanelAction {
    ImportMedia,
    #[cfg(target_arch = "wasm32")]
    RelinkWebAsset(String),
    #[cfg(target_arch = "wasm32")]
    ImportWebSvg(zc_project::Uuid),
    NewProject,
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
    CheckForUpdates,
    #[cfg(not(target_arch = "wasm32"))]
    OpenUpdateDetails,
    #[cfg(not(target_arch = "wasm32"))]
    ClearSkippedUpdate,
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
pub(super) enum TreeDropMode {
    Before,
    After,
    Inside,
}

#[derive(Debug, Clone)]
pub(super) struct TreeDragState {
    pub(super) dragging_id: String,
    pub(super) start_y: f32,
    pub(super) active: bool,
}

#[derive(Debug, Clone)]
pub(super) struct InlineRenameState {
    pub(super) target_id: String,
    pub(super) buffer: String,
    pub(super) just_started: bool,
}

#[derive(Debug, Clone)]
pub(super) struct ShaderEditorTarget {
    pub(super) effect_id: String,
    pub(super) property: String,
    pub(super) effect_name: String,
    pub(super) code: String,
    pub(super) font_px: f32,
    pub(super) error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct ExprEditorTarget {
    pub(super) target_id: String,
    pub(super) property: String,
    pub(super) code: String,
    pub(super) error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct StopwatchContextMenuState {
    pub(super) target_id: String,
    pub(super) property: String,
    pub(super) existing_expr: Option<String>,
    pub(super) local_frame: f64,
    pub(super) current_val: Value,
    pub(super) pos: Pos2,
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
    pub export_queue: Vec<ExportQueueJob>,
    pub export_queue_running: bool,
    pub export_view_mode: Option<ExportMode>,
    /// Single-frame image format: `0 = PNG`, `1 = JPG`, `2 = WebP`.
    pub export_frame_format: usize,
    /// Aspect-ratio mismatch fit mode: `0 = Stretch`, `1 = Crop`, `2 = Black fill`.
    pub export_aspect_mode: usize,
    /// Latest live export preview thumbnail `(width, height, straight RGBA8)` updated during export.
    pub export_preview_rgba: Option<(u32, u32, Vec<u8>)>,
    /// Asset Manager sort mode: `0 = Name`, `1 = Size`, `2 = Usage`, `3 = Type`.
    pub assets_sort_mode: usize,
    pub recovery_snapshots: Vec<RecoverySnapshotInfo>,
    pub app_settings: AppSettings,
    #[cfg(not(target_arch = "wasm32"))]
    pub update_status: UpdateUiStatus,
    pub plugin_notice: Option<String>,
    pub floating_windows: FloatingWindowManager,
    pub new_seq_control_name: String,
    pub new_shader_prop_name: String,
    pub new_shader_prop_kind: usize,
    pub pending_project_actions: Vec<ProjectAction>,
    pub node_graph_editor: NodeGraphEditorState,
    pub looks_editor: LooksEditorState,
    pub effect_setup_editor: EffectSetupState,
    pub lens_flare_options: LensFlareOptionsState,
    pub particle_designer: ParticleDesignerState,
    pub(super) object_drag: Option<TreeDragState>,
    pub(super) effect_drag: Option<TreeDragState>,
    pub(super) rename_state: Option<InlineRenameState>,
    pub(super) shader_editor: Option<ShaderEditorTarget>,
    pub(super) expr_editor: Option<ExprEditorTarget>,
    pub(super) expr_editor_just_opened: bool,
    pub(super) stopwatch_ctx_menu: Option<StopwatchContextMenuState>,
    pub(super) collapsed_effect_groups: BTreeSet<String>,
    pub(super) history_log: VecDeque<(u64, String)>,
    pub(super) last_history_rev: u64,
    pub(super) plugin_filter: String,
    pub(super) assets_filter: String,
    pub(super) shader_prop_kinds: BTreeMap<(String, String), usize>,
    pub(super) settings_editor_layout: usize,
    pub(super) settings_gradient_editor: usize,
    pub(super) settings_default_fps_buf: Option<String>,
    pub(super) settings_edit_colors: bool,
    pub(super) seq_control_counter: usize,
    pub(super) new_seq_control_kind: usize,
    pub(super) new_seq_control_min: String,
    pub(super) new_seq_control_max: String,
    pub(super) new_seq_control_step: String,
    pub(super) particles_tab: usize,
    pub(super) particles_sprite_filter: String,
    pub(super) shape_plus_selected: Option<String>,
    pub(super) shape_plus_collapsed: BTreeSet<String>,
    pub(super) shape_plus_add_kind: usize,
    pub(super) shape_plus_svg_paste: String,
    pub(super) shape_plus_svg_open: bool,
    pub(super) shape_plus_svg_error: Option<String>,
    pub(super) expanded_masks: BTreeSet<String>,
    pub(super) new_comp_name: String,
    pub(super) renaming_comp_id: Option<Uuid>,
    pub(super) renaming_comp_buf: String,
    pub(super) text2d_animator_tab: usize,
    pub(super) objects_list_open: bool,
    #[allow(dead_code)]
    pub(super) objects_focus_bottom: bool,
    #[allow(dead_code)]
    pub(super) effects_focus_bottom: bool,
    pub(super) expanded_vec_props: BTreeSet<(String, String)>,
    pub(super) inspector_root_indent: bool,
    pub(super) inspector_tx_owner: Option<(String, String)>,
    pub(super) inspector_tx_label: Option<String>,
    pub(super) inspector_tx_has_edits: bool,
    pub(super) inspector_tx_owner_seen: bool,
    pub(super) picker_open_seq: u64,
    pub(super) font_picker_search: String,
    pub(super) shader_editor_just_opened: bool,
    pub(super) debug_copied_notice: Option<String>,
    /// Cached asset reference counts keyed by `session.revision()`.
    pub(super) cached_asset_usage: Option<(u64, std::sync::Arc<BTreeMap<String, usize>>)>,
    /// Bounded cache of compiled expressions keyed by `(expressions_plus, trimmed_source)`.
    pub(super) cached_compiled_exprs:
        BTreeMap<(bool, String), Result<std::sync::Arc<zc_expr::CompiledExpr>, String>>,
    /// Cached sequence control values for expression evaluation keyed by `session.revision()`.
    pub(super) cached_expr_controls: Option<(u64, BTreeMap<String, zc_expr::Value>)>,
    /// Cached selected clip snapshot keyed by `(session.revision(), clip_id)` so Objects/Effects/Edit
    /// tabs avoid deep-cloning `SessionClip` twice per 60Hz frame when the document hasn't changed.
    pub(super) cached_selected_clip: Option<(u64, String, std::sync::Arc<SessionClip>)>,
}

impl Default for PanelsState {
    fn default() -> Self {
        Self {
            media_search: String::new(),
            objects_split: SplitPanelState::new(0.40, SplitOrientation::Auto, 80.0, 100.0),
            effects_split: SplitPanelState::new(0.40, SplitOrientation::Auto, 80.0, 100.0),
            prop_columns: PropertyColumnState { label_ratio: 0.44 },
            add_object_open: false,
            add_object_picker: FuzzyPickerState::default(),
            add_effect_open: false,
            add_effect_picker: FuzzyPickerState::default(),
            collapsed_objects: BTreeSet::new(),
            export_settings: ExportSettings::default(),
            export_status: ExportStatus::default(),
            export_queue: Vec::new(),
            export_queue_running: false,
            export_view_mode: None,
            export_frame_format: 0,
            export_aspect_mode: 0,
            export_preview_rgba: None,
            assets_sort_mode: 0,
            recovery_snapshots: Vec::new(),
            app_settings: AppSettings::default(),
            #[cfg(not(target_arch = "wasm32"))]
            update_status: UpdateUiStatus::default(),
            plugin_notice: None,
            floating_windows: FloatingWindowManager::default(),
            new_seq_control_name: String::from("ctl_slider_1"),
            new_shader_prop_name: String::from("Amount"),
            new_shader_prop_kind: 0,
            pending_project_actions: Vec::new(),
            node_graph_editor: NodeGraphEditorState::default(),
            looks_editor: LooksEditorState::default(),
            effect_setup_editor: EffectSetupState::default(),
            lens_flare_options: LensFlareOptionsState::default(),
            particle_designer: ParticleDesignerState::default(),
            object_drag: None,
            effect_drag: None,
            rename_state: None,
            shader_editor: None,
            expr_editor: None,
            expr_editor_just_opened: false,
            stopwatch_ctx_menu: None,
            collapsed_effect_groups: BTreeSet::new(),
            history_log: VecDeque::new(),
            last_history_rev: 0,
            plugin_filter: String::new(),
            assets_filter: String::new(),
            shader_prop_kinds: BTreeMap::new(),
            settings_editor_layout: 0,
            settings_gradient_editor: 0,
            settings_default_fps_buf: None,
            settings_edit_colors: false,
            seq_control_counter: 1,
            new_seq_control_kind: 0,
            new_seq_control_min: String::from("0"),
            new_seq_control_max: String::from("100"),
            new_seq_control_step: String::from("0.1"),
            particles_tab: 0,
            particles_sprite_filter: String::new(),
            shape_plus_selected: None,
            shape_plus_collapsed: BTreeSet::new(),
            shape_plus_add_kind: 1,
            shape_plus_svg_paste: String::new(),
            shape_plus_svg_open: false,
            shape_plus_svg_error: None,
            expanded_masks: BTreeSet::new(),
            new_comp_name: String::from("Precomp"),
            renaming_comp_id: None,
            renaming_comp_buf: String::new(),
            text2d_animator_tab: 0,
            objects_list_open: true,
            objects_focus_bottom: false,
            effects_focus_bottom: false,
            expanded_vec_props: BTreeSet::new(),
            inspector_root_indent: false,
            inspector_tx_owner: None,
            inspector_tx_label: None,
            inspector_tx_has_edits: false,
            inspector_tx_owner_seen: false,
            picker_open_seq: 0,
            font_picker_search: String::new(),
            shader_editor_just_opened: false,
            debug_copied_notice: None,
            cached_asset_usage: None,
            cached_compiled_exprs: BTreeMap::new(),
            cached_expr_controls: None,
            cached_selected_clip: None,
        }
    }
}

impl PanelsState {
    pub(super) fn default_fps_input_id() -> Id {
        Id::unique("settings_default_fps")
    }

    pub(crate) fn commit_default_fps_on_blur(
        &mut self,
        ctx: &egui::Context,
        session: &mut Session,
        settings_visible: bool,
    ) {
        let input_id = Self::default_fps_input_id();
        if settings_visible && ctx.memory(|m| m.has_focus(input_id)) {
            return;
        }
        let Some(buffer) = self.settings_default_fps_buf.take() else {
            return;
        };
        ctx.memory_mut(|m| m.surrender_focus(input_id));
        let Ok(fps) = buffer.trim().parse::<f64>() else {
            return;
        };
        if !fps.is_finite() {
            return;
        }
        let clamped = fps.clamp(1.0, 240.0);
        self.app_settings.default_fps = clamped;
        if session.document().fps != clamped.round()
            || session.active_sequence().exact_fps.is_some()
        {
            let _ = session.execute(Command::SetFps { fps: clamped });
        }
    }

    pub fn take_project_actions(&mut self) -> Vec<ProjectAction> {
        std::mem::take(&mut self.pending_project_actions)
    }
}

pub(super) fn clear_inspector_tx_state(state: &mut PanelsState) {
    state.inspector_tx_owner = None;
    state.inspector_tx_label = None;
    state.inspector_tx_has_edits = false;
}

pub(super) fn commit_inspector_tx_if_open(session: &mut Session, state: &mut PanelsState) {
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

pub(super) fn close_existing_tx_before_start(session: &mut Session, state: &mut PanelsState) {
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

pub(super) fn begin_inspector_tx(
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

/// Tracks continuous egui widget drag / color-picker edit lifecycles so sliders, DragValues,
/// and color editors group 60Hz edits into a single undo transaction via `PanelsState`.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct WidgetTxTracker {
    pub(super) drag_started: bool,
    pub(super) drag_stopped: bool,
    pub(super) changed: bool,
}

impl WidgetTxTracker {
    pub(super) fn observe(&mut self, resp: &egui::Response) -> bool {
        if resp.drag_started() {
            self.drag_started = true;
        }
        if resp.drag_stopped() {
            self.drag_stopped = true;
        }
        if resp.changed() {
            self.changed = true;
            true
        } else {
            false
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_widget_tx_command(
    ui: &Ui,
    session: &mut Session,
    state: &mut PanelsState,
    target_id: &str,
    owner_key: &str,
    label: &str,
    tracker: WidgetTxTracker,
    execute_cmd: impl FnOnce(&mut Session),
) {
    touch_inspector_tx_owner(state, target_id, owner_key);
    let primary_down = ui.input(|i| i.pointer.primary_down());
    let mut owns_tx = owns_inspector_tx(state, target_id, owner_key);
    if tracker.drag_started || (tracker.changed && !owns_tx && primary_down) {
        close_existing_tx_before_start(session, state);
        owns_tx = begin_inspector_tx(session, state, target_id, owner_key, label);
    }
    if tracker.changed {
        execute_cmd(session);
        if owns_tx {
            state.inspector_tx_has_edits = true;
        }
    }
    if owns_tx && (tracker.drag_stopped || !primary_down) {
        if session.history.is_in_transaction() {
            if state.inspector_tx_has_edits {
                let _ = session.commit_transaction(true);
            } else {
                let _ = session.cancel_transaction();
            }
        }
        clear_inspector_tx_state(state);
    }
}

pub(super) fn run_atomic_commit_outside_foreign_tx(
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

pub(super) fn owns_inspector_tx(state: &PanelsState, target_id: &str, prop_key: &str) -> bool {
    state
        .inspector_tx_owner
        .as_ref()
        .is_some_and(|(t, p)| t == target_id && p == prop_key)
}

pub(super) fn touch_inspector_tx_owner(state: &mut PanelsState, target_id: &str, prop_key: &str) {
    if owns_inspector_tx(state, target_id, prop_key) {
        state.inspector_tx_owner_seen = true;
    }
}

/// Executes `Command::PastePzJson` on `json` and emits clear toast feedback into `actions`.
#[allow(clippy::too_many_arguments)]
pub(super) fn paste_pz_json_text(
    session: &mut Session,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
    json: String,
    hint: &str,
    target_clip_id: Option<String>,
    parent_object_id: Option<String>,
    target_object_id: Option<String>,
    target_property: Option<String>,
) -> bool {
    commit_inspector_tx_if_open(session, state);
    // When pasting into the Layer Masks section, normalize mask objects that serialize with
    // `"kind": "path"` so `zc-project`'s clipboard classifier doesn't mistake them for Shape+
    // vector path items before running `parse_layer_masks`.
    let json = if hint == "masks" {
        fn strip_mask_path_kind(v: &mut Value) {
            if let Some(obj) = v.as_object_mut() {
                let is_mask_shape = obj.get("kind").and_then(Value::as_str) == Some("path")
                    && (obj.contains_key("feather")
                        || obj.contains_key("expansion")
                        || obj.contains_key("invert")
                        || obj.contains_key("outlineColor")
                        || obj.contains_key("vertexIds")
                        || obj.contains_key("mode"));
                if is_mask_shape {
                    obj.remove("kind");
                }
                for key in ["masks", "zoidiumMasks", "data"] {
                    if let Some(child) = obj.get_mut(key) {
                        strip_mask_path_kind(child);
                    }
                }
            } else if let Some(arr) = v.as_array_mut() {
                for elem in arr {
                    strip_mask_path_kind(elem);
                }
            }
        }
        match serde_json::from_str::<Value>(&json) {
            Ok(mut val) => {
                strip_mask_path_kind(&mut val);
                val.to_string()
            }
            Err(_) => json,
        }
    } else {
        json
    };
    match session.execute(Command::PastePzJson {
        json,
        hint: Some(hint.to_owned()),
        target_clip_id,
        parent_object_id,
        target_object_id,
        target_property,
        at_frame: None,
    }) {
        Ok(val) => {
            let kind = val.get("kind").and_then(Value::as_str).unwrap_or(hint);
            let default_count = if kind == "material" { 1 } else { 0 };
            let count = val
                .get("count")
                .and_then(Value::as_u64)
                .unwrap_or(default_count);
            if count == 0 && kind != "material" {
                let err_msg = match kind {
                    "shape_plus" => "Paste failed: no valid Shape+ items found in JSON".to_owned(),
                    "effects" => "Paste failed: no valid effects found in JSON".to_owned(),
                    "objects3d" => "Paste failed: no valid 3D objects found in JSON".to_owned(),
                    "masks" => "Paste failed: no valid layer masks found in JSON".to_owned(),
                    "clips" => "Paste failed: no valid clips found in JSON".to_owned(),
                    "keyframes" => "Paste failed: no valid keyframes found in JSON".to_owned(),
                    other => format!("Paste failed: no valid {other} items found"),
                };
                actions.push(PanelAction::StatusMessage(err_msg));
                return false;
            }
            let mut msg = match kind {
                "effects" => format!("Pasted {count} effect{}", if count == 1 { "" } else { "s" }),
                "objects3d" => {
                    format!(
                        "Pasted {count} 3D object{}",
                        if count == 1 { "" } else { "s" }
                    )
                }
                "material" => "Pasted 3D material".to_owned(),
                "masks" => format!("Pasted {count} mask{}", if count == 1 { "" } else { "s" }),
                "shape_plus" => {
                    format!(
                        "Pasted {count} Shape+ item{}",
                        if count == 1 { "" } else { "s" }
                    )
                }
                "clips" => format!("Pasted {count} clip{}", if count == 1 { "" } else { "s" }),
                "keyframes" => {
                    format!(
                        "Pasted {count} keyframe{}",
                        if count == 1 { "" } else { "s" }
                    )
                }
                other => format!("Pasted {other}"),
            };
            if let Some(plugins) = val.get("enabled_plugins").and_then(Value::as_array) {
                let names: Vec<&str> = plugins.iter().filter_map(Value::as_str).collect();
                if !names.is_empty() {
                    msg.push_str(&format!(" (enabled: {})", names.join(", ")));
                }
            }
            actions.push(PanelAction::StatusMessage(msg));
            true
        }
        Err(err) => {
            actions.push(PanelAction::StatusMessage(format!("Paste failed: {err}")));
            false
        }
    }
}

/// Reads the OS clipboard and executes `Command::PastePzJson` with clear toast feedback.
#[allow(clippy::too_many_arguments)]
pub(super) fn paste_pz_json_from_clipboard(
    session: &mut Session,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
    hint: &str,
    target_clip_id: Option<String>,
    parent_object_id: Option<String>,
    target_object_id: Option<String>,
    target_property: Option<String>,
    empty_label: &str,
) -> bool {
    let Some(json) = read_system_clipboard() else {
        actions.push(PanelAction::StatusMessage(format!(
            "Clipboard is empty (no {empty_label})"
        )));
        return false;
    };
    paste_pz_json_text(
        session,
        state,
        actions,
        json,
        hint,
        target_clip_id,
        parent_object_id,
        target_object_id,
        target_property,
    )
}

/// Writes a serialized PZ JSON payload to the OS system clipboard and reports toast feedback.
pub(super) fn copy_pz_json_to_clipboard(
    ctx: &egui::Context,
    result: Result<String, zc_session::SessionError>,
    success_msg: &str,
    actions: &mut Vec<PanelAction>,
) {
    match result {
        Ok(json) => {
            write_system_clipboard(ctx, &json);
            actions.push(PanelAction::StatusMessage(success_msg.to_owned()));
        }
        Err(err) => {
            actions.push(PanelAction::StatusMessage(format!("Copy failed: {err}")));
        }
    }
}

/// Renders a crisp compact action button inside a 27-28px panel/section header bar.
/// Returns `(clicked, hovered)`.
pub(super) fn header_compact_action_button(
    ui: &mut Ui,
    id: Id,
    rect: Rect,
    label: &str,
    tooltip: &str,
    theme: &Cm3Theme,
) -> (bool, bool) {
    let resp = ui
        .interact(rect, id, Sense::click())
        .on_hover_cursor(CursorIcon::PointingHand)
        .on_hover_text(tooltip);
    let hovered = resp.hovered();
    ui.painter().rect_filled(
        rect,
        CornerRadius::ZERO,
        if hovered { BG_HOVER } else { BG_BUTTON },
    );
    paint_crisp_panel_border(
        ui.painter(),
        rect,
        if hovered {
            theme.accent()
        } else {
            BORDER_CONTROL
        },
    );
    ui.painter().with_clip_rect(rect).text(
        rect.center(),
        Align2::CENTER_CENTER,
        label,
        theme.font(FONT_COMPACT),
        if hovered { TEXT_BRIGHT } else { TEXT_PRIMARY },
    );
    (resp.clicked(), hovered)
}
