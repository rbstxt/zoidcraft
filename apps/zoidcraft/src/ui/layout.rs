//! Pixel-exact recursive binary Split Panel engine, CM3 workspace shell,
//! animated Elevator rail, File & Playback toolbars, Audio Level Meter,
//! and global shortcut dispatcher (Spec 01 §1 & §6).

use egui::{
    Align2, Color32, CornerRadius, CursorIcon, Frame, Id, Key, Mesh, Pos2, Rect, ScrollArea, Sense,
    Shape, Stroke, StrokeKind, TextEdit, Ui, UiBuilder, Vec2,
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
    /// and clamping precedence (`Pane 1`'s minimum wins if container shrinks below sum).
    #[must_use]
    pub fn compute_rects(&self, container: Rect) -> (Rect, Rect, Rect) {
        let horiz = self.resolved_horizontal(container);
        let extent = if horiz {
            container.width().max(SPLITTER_SIZE)
        } else {
            container.height().max(SPLITTER_SIZE)
        };
        let r = self.ratio.clamp(0.0, 1.0);

        let mut pane0_end = extent * r - HALF_SPLITTER;
        let mut pane1_start = extent * r + HALF_SPLITTER;

        // Clamping precedence (Spec 01 §1.1):
        // 1) Clamp pane0_end >= min_pane_0 (shifting pane1_start = pane0_end + 3)
        if pane0_end < self.min_pane_0 {
            pane0_end = self.min_pane_0;
            pane1_start = pane0_end + SPLITTER_SIZE;
        }
        // 2) Clamp extent - pane1_start >= min_pane_1 (shifting pane0_end = pane1_start - 3)
        if extent - pane1_start < self.min_pane_1 {
            pane1_start = (extent - self.min_pane_1).max(0.0);
            pane0_end = (pane1_start - SPLITTER_SIZE).max(0.0);
        }

        if horiz {
            let x0 = container.left();
            let p0 = Rect::from_min_max(
                Pos2::new(x0, container.top()),
                Pos2::new(x0 + pane0_end, container.bottom()),
            );
            let split = Rect::from_min_max(
                Pos2::new(x0 + pane0_end, container.top()),
                Pos2::new(x0 + pane1_start, container.bottom()),
            );
            let p1 = Rect::from_min_max(
                Pos2::new(x0 + pane1_start, container.top()),
                container.right_bottom(),
            );
            (p0, split, p1)
        } else {
            let y0 = container.top();
            let p0 = Rect::from_min_max(
                Pos2::new(container.left(), y0),
                Pos2::new(container.right(), y0 + pane0_end),
            );
            let split = Rect::from_min_max(
                Pos2::new(container.left(), y0 + pane0_end),
                Pos2::new(container.right(), y0 + pane1_start),
            );
            let p1 = Rect::from_min_max(
                Pos2::new(container.left(), y0 + pane1_start),
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

    // 3 px black splitter bar & live drag hitbox
    ui.painter()
        .rect_filled(split_rect, CornerRadius::ZERO, VOID_BLACK);
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
        let new_p0 = (cur_p0_extent + delta).clamp(
            state.min_pane_0,
            (extent - state.min_pane_1 - SPLITTER_SIZE).max(state.min_pane_0),
        );
        state.ratio = ((new_p0 + HALF_SPLITTER) / extent).clamp(0.0, 1.0);
    }

    // Recompute rects if ratio moved this frame for crisp zero-lag drag
    let (p0, _, p1) = state.compute_rects(rect);
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

/// Stateful Audio Level Meter holding RMS & nonlinear peak-hold decay (Spec 01 §1.7).
#[derive(Debug, Clone, Copy, Default)]
pub struct AudioMeterState {
    /// Current 1024-sample normalized RMS `[0.0, 1.0]`.
    pub rms: f32,
    /// Decaying peak-hold level `[0.0, 1.0]`.
    pub peak_hold: f32,
}

impl AudioMeterState {
    /// Updates meter with latest `(rms, max_sample)` applying Spec 01 §1.7 nonlinear decay:
    /// `peak_{t+1} = max(max_sample, peak_t - 0.01 / exp(peak_t))`.
    pub fn update(&mut self, rms: f32, max_sample: f32) {
        self.rms = rms.clamp(0.0, 1.0);
        let decayed = (self.peak_hold - 0.01 / self.peak_hold.exp()).max(0.0);
        self.peak_hold = max_sample.clamp(0.0, 1.0).max(decayed);
    }
}

/// Paints the exact `45 px` CM3 Audio Level Meter (Spec 01 §1.7).
pub fn paint_audio_meter(
    ui: &mut Ui,
    rect: Rect,
    focused: bool,
    meter: &AudioMeterState,
    theme: &Cm3Theme,
) {
    ui.painter()
        .rect_filled(rect, CornerRadius::ZERO, BG_TOOLBAR);

    let inner = rect.shrink(4.0);
    if inner.width() > 2.0 && inner.height() > 2.0 {
        // Bottom-to-top vertical gradient: green (0%) -> yellow (90%) -> red (100%)
        let y_split = inner.bottom() - inner.height() * 0.90;
        let green = Color32::from_rgb(0x1f, 0xa8, 0x3b);
        let yellow = Color32::from_rgb(0xe8, 0xd9, 0x2a);
        let red = Color32::from_rgb(0xd8, 0x26, 0x26);

        let mut mesh = Mesh::default();
        // Lower 90%: green -> yellow
        let bl = mesh.vertices.len() as u32;
        mesh.colored_vertex(inner.left_bottom(), green);
        let br = mesh.vertices.len() as u32;
        mesh.colored_vertex(inner.right_bottom(), green);
        let mr = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(inner.right(), y_split), yellow);
        let ml = mesh.vertices.len() as u32;
        mesh.colored_vertex(Pos2::new(inner.left(), y_split), yellow);
        mesh.add_triangle(bl, br, mr);
        mesh.add_triangle(bl, mr, ml);

        // Top 10%: yellow -> red
        let tl = mesh.vertices.len() as u32;
        mesh.colored_vertex(inner.left_top(), red);
        let tr = mesh.vertices.len() as u32;
        mesh.colored_vertex(inner.right_top(), red);
        mesh.add_triangle(ml, mr, tr);
        mesh.add_triangle(ml, tr, tl);
        ui.painter().add(Shape::mesh(mesh));

        // Top-down dark mask (#111111 @ 95% opacity) scaled to (1 - rms)
        let mask_h = inner.height() * (1.0 - meter.rms.clamp(0.0, 1.0));
        if mask_h > 0.5 {
            let mask_rect = Rect::from_min_size(inner.left_top(), Vec2::new(inner.width(), mask_h));
            ui.painter().rect_filled(
                mask_rect,
                CornerRadius::ZERO,
                Color32::from_rgba_unmultiplied(0x11, 0x11, 0x11, 242),
            );
        }

        // 1px peak-hold line (#cece62) at peak * 100% from bottom
        if meter.peak_hold > 0.005 {
            let py = inner.bottom() - inner.height() * meter.peak_hold.clamp(0.0, 1.0);
            ui.painter().line_segment(
                [Pos2::new(inner.left(), py), Pos2::new(inner.right(), py)],
                Stroke::new(1.0, AUDIO_PEAK_YELLOW),
            );
        }
    }

    let border = if focused {
        theme.accent()
    } else {
        BORDER_PANEL
    };
    ui.painter().rect_stroke(
        rect,
        CornerRadius::ZERO,
        Stroke::new(1.0, border),
        StrokeKind::Inside,
    );
}

// ---------------------------------------------------------------------------
// 3. Focus, Player+, Actions & Workspace State
// ---------------------------------------------------------------------------

/// Active focused leaf panel inside the CM3 workspace (drives `1px` `#384668` focus border).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PanelFocus {
    #[default]
    Viewport,
    Toolbar,
    ElevatorRail,
    SideInspector,
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
    SaveProject { include_assets: bool },
    ImportMedia,
    OpenGraphEditor,
    StartExport(ExportSettings),
    CancelExport,
    EyedropperActivate { target_id: String, property: String },
    SingleFrameRender,
    StatusMessage(String),
}

/// Complete CM3 + Zoidium root workspace layout controller.
pub struct Cm3Workspace {
    /// Root horizontal split: Left Column (30%) | Right Workspace (70%).
    pub root_split: SplitPanelState,
    /// Right workspace vertical split: Viewport (65%) / Playback + Timeline (35%).
    pub right_split: SplitPanelState,
    /// Bottom-right horizontal split: Timeline | Audio Meter (45px fixed via ratio 1.0).
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
    /// Audio level meter state (RMS + peak hold).
    pub audio_meter: AudioMeterState,
    /// Player+ toolbar state.
    pub player_plus: PlayerPlusState,
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
                45.0,
            ),
            active_tab: SideTab::Objects,
            rail_expanded: false,
            rail_width_px: 60.0,
            rail_focus_index: 3,
            rail_drag_dwell: None,
            focused_panel: PanelFocus::Viewport,
            maximized_panel: None,
            project_name_buf: String::new(),
            audio_meter: AudioMeterState::default(),
            player_plus: PlayerPlusState::default(),
            panels: PanelsState::default(),
            suppress_shortcuts: false,
        }
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

        // Outer 1px black perimeter frame (Spec 01 §1.1)
        ui.painter()
            .rect_filled(full_rect, CornerRadius::ZERO, VOID_BLACK);
        let root_rect = full_rect.shrink(1.0);

        // Global shortcuts (strictly guarded against active text inputs & floating windows)
        self.handle_global_shortcuts(ui.ctx(), session, &mut actions);

        // Maximized leaf panel mode (CM3 Backquote `` ` `` shortcut, Spec 01 §6.2)
        if let Some(max_panel) = self.maximized_panel {
            match max_panel {
                PanelFocus::Viewport => {
                    self.show_viewport_frame(ui, root_rect, session, theme, &mut draw_viewport);
                }
                PanelFocus::Timeline => {
                    self.show_timeline_frame(ui, root_rect, session, theme, &mut draw_timeline);
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
                }
                _ => {
                    self.maximized_panel = None;
                }
            }
            return actions;
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
                // Fixed 40px File & History Toolbar at top + 3px black gap + Elevator/Inspector below
                let toolbar_h = 40.0_f32.min(left_rect.height());
                let toolbar_rect = Rect::from_min_size(
                    left_rect.left_top(),
                    Vec2::new(left_rect.width(), toolbar_h),
                );
                let lower_rect = Rect::from_min_max(
                    Pos2::new(
                        left_rect.left(),
                        (toolbar_rect.bottom() + SPLITTER_SIZE).min(left_rect.bottom()),
                    ),
                    left_rect.right_bottom(),
                );

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
                        let pb_h = 40.0_f32.min(bottom_rect.height());
                        let pb_rect = Rect::from_min_size(
                            bottom_rect.left_top(),
                            Vec2::new(bottom_rect.width(), pb_h),
                        );
                        let tl_row_rect = Rect::from_min_max(
                            Pos2::new(
                                bottom_rect.left(),
                                (pb_rect.bottom() + SPLITTER_SIZE).min(bottom_rect.bottom()),
                            ),
                            bottom_rect.right_bottom(),
                        );

                        self.show_playback_toolbar(ui, pb_rect, session, theme, &mut actions);

                        // Horizontal Split (ratio 1.0): Timeline | 45px Audio Level Meter
                        let (tl_rect, meter_rect) = begin_split_panel(
                            ui,
                            Id::unique("cm3_timeline_meter_split"),
                            tl_row_rect,
                            &mut self.timeline_meter_split,
                        );
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
                                let meter_resp = ui.interact(
                                    meter_rect,
                                    Id::unique("audio_meter_focus"),
                                    Sense::click(),
                                );
                                if meter_resp.clicked() {
                                    self.focused_panel = PanelFocus::AudioMeter;
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
        let focused = self.focused_panel == PanelFocus::Toolbar;
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, BG_TOOLBAR);

        let bg_resp = ui.interact(rect, Id::unique("file_toolbar_bg"), Sense::click());
        if bg_resp.clicked() {
            self.focused_panel = PanelFocus::Toolbar;
        }

        let inner = rect.shrink(5.0);
        let buttons_total_w = 5.0 * 29.0 + 4.0 * 5.0 + 15.0; // 5 buttons + gaps + 15px separator
        let show_name_col = inner.width() >= buttons_total_w + 105.0;

        // Column 1: Zoidium inline project name input (height 29px, #7e8fb9 value text)
        let name_edit_id = Id::unique("zoidium_toolbar_project_name");
        if !ui.memory(|m| m.has_focus(name_edit_id)) {
            self.project_name_buf = session.document().project_name.clone();
        }
        if show_name_col {
            let name_w = ((inner.width() - buttons_total_w) * 0.5 - 6.0).max(90.0);
            let name_rect = Rect::from_min_size(
                Pos2::new(inner.left(), inner.center().y - 14.5),
                Vec2::new(name_w, 29.0),
            );
            let name_focused = ui.memory(|m| m.has_focus(name_edit_id));
            ui.painter().rect_filled(
                name_rect,
                CornerRadius::ZERO,
                if name_focused { BG_INPUT } else { BG_TOOLBAR },
            );
            if name_focused {
                ui.painter().rect_stroke(
                    name_rect,
                    CornerRadius::ZERO,
                    Stroke::new(1.0, theme.accent()),
                    StrokeKind::Inside,
                );
            }
            let resp = ui
                .scope_builder(
                    UiBuilder::new().max_rect(name_rect.shrink2(Vec2::new(4.0, 2.0))),
                    |ui| {
                        ui.add(
                            TextEdit::singleline(&mut self.project_name_buf)
                                .id(name_edit_id)
                                .font(theme.font(FONT_BODY))
                                .text_color(theme.value())
                                .desired_width(name_w - 8.0),
                        )
                    },
                )
                .inner;
            if resp.lost_focus() || (resp.has_focus() && ui.input(|i| i.key_pressed(Key::Enter))) {
                let trimmed = self.project_name_buf.trim().to_owned();
                if !trimmed.is_empty() && trimmed != session.document().project_name {
                    let _ = session.execute(Command::SetProjectName { name: trimmed });
                }
                ui.memory_mut(|m| m.surrender_focus(name_edit_id));
            }
        }

        // Column 2: Centered 5 action buttons (New, Open, Save [dirty dot], sep, Undo, Redo)
        let start_x = if show_name_col {
            inner.center().x - buttons_total_w * 0.5
        } else {
            (inner.center().x - buttons_total_w * 0.5).max(inner.left())
        };
        let mut cursor_x = start_x;
        let btn_y = inner.center().y - 14.5;

        let toolbar_btn = |cursor_x: &mut f32,
                           icon: Cm3Icon,
                           tip: &str,
                           enabled: bool,
                           dirty_dot: bool|
         -> bool {
            let b_rect = Rect::from_min_size(Pos2::new(*cursor_x, btn_y), Vec2::splat(29.0));
            *cursor_x += 34.0; // 29px + 5px right margin
            let resp = ui
                .interact(b_rect, Id::unique(("ftb", tip)), Sense::click())
                .on_hover_cursor(if enabled {
                    CursorIcon::PointingHand
                } else {
                    CursorIcon::Default
                })
                .on_hover_text(tip);
            if enabled && resp.hovered() {
                ui.painter().rect_stroke(
                    b_rect,
                    CornerRadius::ZERO,
                    Stroke::new(2.0, theme.accent()),
                    StrokeKind::Inside,
                );
            }
            let icon_rect = Rect::from_center_size(b_rect.center(), Vec2::splat(25.0));
            let tint = if enabled {
                TOOLBAR_ICON_INACTIVE
            } else {
                Color32::from_rgb(0x4a, 0x4a, 0x4a)
            };
            paint_cm3_icon(ui.painter(), icon_rect, icon, tint);

            // Zoidium 8x8 unsaved-changes indicator dot at top-right of Save (Spec 01 §1.3)
            if dirty_dot {
                let dot_center = Pos2::new(b_rect.right() - 5.0, b_rect.top() + 5.0);
                ui.painter().circle_filled(dot_center, 4.0, TEXT_MUTED);
            }
            enabled && resp.clicked()
        };

        if toolbar_btn(&mut cursor_x, Cm3Icon::New, "new (ctrl-m)", true, false) {
            actions.push(UiAction::NewProject);
        }
        if toolbar_btn(&mut cursor_x, Cm3Icon::Load, "open (ctrl-o)", true, false) {
            actions.push(UiAction::OpenProject);
        }
        if toolbar_btn(
            &mut cursor_x,
            Cm3Icon::Save,
            "save (ctrl-s)",
            true,
            session.is_dirty(),
        ) {
            let include_assets = !ui.input(|i| i.modifiers.shift);
            actions.push(UiAction::SaveProject { include_assets });
        }
        cursor_x += 15.0; // 15px separator
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
        ui.painter().rect_stroke(
            rect,
            CornerRadius::ZERO,
            Stroke::new(1.0, border),
            StrokeKind::Inside,
        );
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
        // Animate rail width: 200ms expand, 50ms collapse (Spec 01 §1.4 & §7)
        let target_w = if self.rail_expanded {
            300.0_f32.min(rect.width())
        } else {
            60.0
        };
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

        // Side Inspector always sits at left: 60px so expanded drawer overlays without reflowing!
        let inspector_rect = Rect::from_min_max(
            Pos2::new((rect.left() + 60.0).min(rect.right()), rect.top()),
            rect.right_bottom(),
        );

        let insp_bg = ui.interact(
            inspector_rect,
            Id::unique("side_inspector_focus"),
            Sense::click(),
        );
        if insp_bg.clicked() || insp_bg.is_pointer_button_down_on() {
            self.focused_panel = PanelFocus::SideInspector;
            if self.rail_expanded {
                self.rail_expanded = false;
            }
        }

        ui.painter()
            .rect_filled(inspector_rect, CornerRadius::ZERO, BG_PANEL);

        let mut panel_actions = Vec::new();
        ui.scope_builder(
            UiBuilder::new()
                .max_rect(inspector_rect.shrink(1.0))
                .id_salt("side_inspector_content"),
            |ui| {
                ui.set_clip_rect(ui.clip_rect().intersect(inspector_rect.shrink(1.0)));
                show_side_panel(
                    ui,
                    self.active_tab,
                    inspector_rect.shrink(1.0),
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
                PanelAction::StatusMessage(msg) => UiAction::StatusMessage(msg),
            });
        }

        let insp_border = if self.focused_panel == PanelFocus::SideInspector {
            theme.accent()
        } else {
            BORDER_PANEL
        };
        ui.painter().rect_stroke(
            inspector_rect,
            CornerRadius::ZERO,
            Stroke::new(1.0, insp_border),
            StrokeKind::Inside,
        );

        // Render Elevator Rail LAST so expanded 300px drawer overlays the inspector cleanly
        let rail_rect = Rect::from_min_size(
            rect.left_top(),
            Vec2::new(self.rail_width_px.min(rect.width()), rect.height()),
        );
        if self.rail_width_px > 61.0 {
            // Drawer overlay shadow: 5px 0 10px rgba(0,0,0,0.5)
            for (dx, alpha) in [(2.0, 95), (6.0, 55), (10.0, 22)] {
                let sh = Rect::from_min_size(
                    Pos2::new(rail_rect.right(), rail_rect.top()),
                    Vec2::new(dx, rail_rect.height()),
                );
                ui.painter()
                    .rect_filled(sh, CornerRadius::ZERO, Color32::from_black_alpha(alpha));
            }
        }
        self.show_elevator_rail(ui, rail_rect, theme);
    }

    fn show_elevator_rail(&mut self, ui: &mut Ui, rail_rect: Rect, theme: &Cm3Theme) {
        ui.painter()
            .rect_filled(rail_rect, CornerRadius::ZERO, VOID_BLACK);
        let rail_focused = self.focused_panel == PanelFocus::ElevatorRail;

        // Keyboard navigation when rail has focus (Spec 01 §1.4)
        if rail_focused && !self.suppress_shortcuts && !ui.ctx().egui_wants_keyboard_input() {
            ui.input(|i| {
                if i.key_pressed(Key::ArrowDown) {
                    self.rail_focus_index = (self.rail_focus_index + 1) % SideTab::ALL.len();
                } else if i.key_pressed(Key::ArrowUp) {
                    self.rail_focus_index =
                        (self.rail_focus_index + SideTab::ALL.len() - 1) % SideTab::ALL.len();
                } else if i.key_pressed(Key::Enter) {
                    self.active_tab = SideTab::ALL[self.rail_focus_index];
                    self.rail_expanded = false;
                }
            });
        }

        // Top 50px Menu Toggle button (`menu` icon 30x30 at left: 13px, top: 4px)
        let menu_rect = Rect::from_min_size(
            rail_rect.left_top(),
            Vec2::new(rail_rect.width(), 50.0_f32.min(rail_rect.height())),
        );
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
            ui.painter().rect_stroke(
                menu_rect,
                CornerRadius::ZERO,
                Stroke::new(2.0, theme.accent()),
                StrokeKind::Inside,
            );
        }
        let menu_icon_rect = Rect::from_min_size(
            Pos2::new(menu_rect.left() + 13.0, menu_rect.top() + 10.0),
            Vec2::splat(30.0),
        );
        paint_cm3_icon(ui.painter(), menu_icon_rect, Cm3Icon::Menu, TEXT_STRONG);
        if menu_resp.clicked() {
            self.rail_expanded = !self.rail_expanded;
            self.focused_panel = PanelFocus::ElevatorRail;
        }

        // Scrollable Tab Strip (top: 50px to bottom, 50px rows)
        let strip_rect = Rect::from_min_max(
            Pos2::new(rail_rect.left(), menu_rect.bottom()),
            rail_rect.right_bottom(),
        );
        let now = ui.input(|i| i.time);
        let pointer_dragging = ui.input(|i| i.pointer.primary_down());
        let mut hovered_tab_this_frame: Option<SideTab> = None;

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
                            let row_resp = row_resp
                                .on_hover_cursor(CursorIcon::PointingHand)
                                .on_hover_text(tab.title());

                            let selected = self.active_tab == tab;
                            if selected {
                                ui.painter().rect_filled(
                                    row_rect,
                                    CornerRadius::ZERO,
                                    theme.accent(),
                                );
                            } else if row_resp.hovered() {
                                ui.painter()
                                    .rect_filled(row_rect, CornerRadius::ZERO, BG_TITLE);
                            }

                            if rail_focused && self.rail_focus_index == idx {
                                ui.painter().rect_stroke(
                                    row_rect,
                                    CornerRadius::ZERO,
                                    Stroke::new(2.0, BORDER_RAIL_FOCUS),
                                    StrokeKind::Inside,
                                );
                            }

                            // 30x30 icon at left: 13px, top: 9px
                            let icon_rect = Rect::from_min_size(
                                Pos2::new(row_rect.left() + 13.0, row_rect.top() + 9.0),
                                Vec2::splat(30.0),
                            );
                            paint_cm3_icon(ui.painter(), icon_rect, tab.icon(), TEXT_STRONG);

                            // 22px label clipped horizontally when collapsed (margin-right 30px from icon)
                            if strip_rect.width() > 64.0 {
                                let label_x = icon_rect.right() + 24.0;
                                ui.painter().with_clip_rect(row_rect).text(
                                    Pos2::new(label_x, row_rect.center().y),
                                    Align2::LEFT_CENTER,
                                    tab.title(),
                                    theme.font(FONT_TAB),
                                    TEXT_STRONG,
                                );
                            }

                            if row_resp.clicked() {
                                self.active_tab = tab;
                                self.rail_focus_index = idx;
                                self.rail_expanded = false;
                                self.focused_panel = PanelFocus::SideInspector;
                            }

                            if row_resp.hovered() && pointer_dragging && !selected {
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
        let focused = self.focused_panel == PanelFocus::Viewport;
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, BG_SUNKEN);

        let inner = rect.shrink(1.0);
        if ui.input(|i| {
            i.pointer.primary_pressed()
                && i.pointer.interact_pos().is_some_and(|p| inner.contains(p))
        }) {
            self.focused_panel = PanelFocus::Viewport;
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
        ui.painter().rect_stroke(
            rect,
            CornerRadius::ZERO,
            Stroke::new(1.0, border),
            StrokeKind::Inside,
        );
    }

    // -----------------------------------------------------------------------
    // Right Middle: 40px Playback & Viewport Toolbar (Spec 01 §1.6)
    // -----------------------------------------------------------------------

    fn show_playback_toolbar(
        &mut self,
        ui: &mut Ui,
        rect: Rect,
        session: &mut Session,
        theme: &Cm3Theme,
        actions: &mut Vec<UiAction>,
    ) {
        let focused = self.focused_panel == PanelFocus::PlaybackToolbar;
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, BG_TOOLBAR);

        let bg_resp = ui.interact(rect, Id::unique("playback_toolbar_bg"), Sense::click());
        if bg_resp.clicked() {
            self.focused_panel = PanelFocus::PlaybackToolbar;
        }

        let inner = rect.shrink(5.0);
        let player_plus = session.plugins.is_enabled("player-plus");
        // 9 standard icon buttons + 2 separators (15px each)
        let std_w = 9.0 * 29.0 + 8.0 * 5.0 + 2.0 * 15.0;
        let extra_w = if player_plus { 150.0 } else { 0.0 };
        let total_w = std_w + extra_w;
        let mut cx = (inner.center().x - total_w * 0.5).max(inner.left());
        let by = inner.center().y - 14.5;

        let pb_btn = |cx: &mut f32, icon: Cm3Icon, tip: &str, active: bool| -> bool {
            let b_rect = Rect::from_min_size(Pos2::new(*cx, by), Vec2::splat(29.0));
            *cx += 34.0;
            let resp = ui
                .interact(b_rect, Id::unique(("pbtb", tip)), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text(tip);
            if resp.hovered() {
                ui.painter().rect_stroke(
                    b_rect,
                    CornerRadius::ZERO,
                    Stroke::new(2.0, theme.accent()),
                    StrokeKind::Inside,
                );
            }
            let ic = Rect::from_center_size(b_rect.center(), Vec2::splat(25.0));
            let color = if active {
                TOGGLE_ACTIVE // #8a2828 active toggle fill (Spec 01 §1.6)
            } else {
                TOOLBAR_ICON_INACTIVE // #acacac inactive fill
            };
            paint_cm3_icon(ui.painter(), ic, icon, color);
            resp.clicked()
        };

        // 1. Editing Camera (C)
        if pb_btn(
            &mut cx,
            Cm3Icon::Camera,
            "editing camera (c)",
            session.playback.editing_camera,
        ) {
            session.playback.toggle_editing_camera();
        }
        // 2. Layer Transform Controls (T)
        if pb_btn(
            &mut cx,
            Cm3Icon::Transform,
            "2D layer transform controls (t)",
            session.playback.transform_controls_2d,
        ) {
            session.playback.toggle_transform_controls_2d();
        }
        // 3. Separator (15px)
        cx += 15.0;
        // 4. Start (Home)
        if pb_btn(&mut cx, Cm3Icon::Start, "jump to start (Home)", false) {
            let _ = session.execute(Command::SetFrame { frame: 0.0 });
        }
        // 5. Previous Frame (ArrowLeft)
        if pb_btn(&mut cx, Cm3Icon::PrevFrame, "previous frame (Left)", false) {
            let _ = session.execute(Command::StepFrame { delta: -1.0 });
        }
        // 6. Play / Pause (Space)
        let playing = session.playback.is_playing();
        if pb_btn(
            &mut cx,
            if playing {
                Cm3Icon::Pause
            } else {
                Cm3Icon::Play
            },
            if playing {
                "pause (Space)"
            } else {
                "play (Space)"
            },
            playing,
        ) {
            let _ = session.execute(Command::TogglePlay);
        }
        // 7. Next Frame (ArrowRight)
        if pb_btn(&mut cx, Cm3Icon::NextFrame, "next frame (Right)", false) {
            let _ = session.execute(Command::StepFrame { delta: 1.0 });
        }
        // 8. End (End)
        if pb_btn(&mut cx, Cm3Icon::End, "jump to end (End)", false) {
            let end_f = (session.document().sequence_length - 1.0).max(0.0);
            let _ = session.execute(Command::SetFrame { frame: end_f });
        }
        // 9. Separator (15px)
        cx += 15.0;
        // 10. Loop (Ctrl+L)
        if pb_btn(
            &mut cx,
            Cm3Icon::Loop,
            "loop playback (Ctrl+L)",
            session.playback.looping,
        ) {
            let _ = session.execute(Command::ToggleLoop);
        }
        // 11. Graph Editor (Ctrl+G)
        if pb_btn(&mut cx, Cm3Icon::Graph, "graph editor (Ctrl+G)", false) {
            actions.push(UiAction::OpenGraphEditor);
        }

        // Optional Player+ controls (Pause Drawing, Single Render, Preview Quality)
        if player_plus && cx + 120.0 <= inner.right() {
            cx += 10.0;
            let pause_rect = Rect::from_min_size(Pos2::new(cx, by + 3.0), Vec2::new(52.0, 23.0));
            cx += 56.0;
            let p_resp = ui
                .interact(pause_rect, Id::unique("pp_pause_draw"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text("Player+: pause viewport drawing");
            ui.painter().rect_filled(
                pause_rect,
                CornerRadius::ZERO,
                if self.player_plus.drawing_paused {
                    TOGGLE_ACTIVE
                } else {
                    BG_INPUT
                },
            );
            ui.painter().rect_stroke(
                pause_rect,
                CornerRadius::ZERO,
                Stroke::new(1.0, BORDER_CONTROL),
                StrokeKind::Inside,
            );
            ui.painter().text(
                pause_rect.center(),
                Align2::CENTER_CENTER,
                if self.player_plus.drawing_paused {
                    "Hold"
                } else {
                    "Live"
                },
                theme.font(FONT_SMALL),
                TEXT_BRIGHT,
            );
            if p_resp.clicked() {
                self.player_plus.drawing_paused = !self.player_plus.drawing_paused;
            }

            if self.player_plus.drawing_paused {
                let one_rect = Rect::from_min_size(Pos2::new(cx, by + 3.0), Vec2::new(28.0, 23.0));
                cx += 32.0;
                let o_resp = ui
                    .interact(one_rect, Id::unique("pp_single_render"), Sense::click())
                    .on_hover_cursor(CursorIcon::PointingHand)
                    .on_hover_text("Player+: render single frame");
                ui.painter()
                    .rect_filled(one_rect, CornerRadius::ZERO, BG_INPUT);
                ui.painter().rect_stroke(
                    one_rect,
                    CornerRadius::ZERO,
                    Stroke::new(1.0, BORDER_CONTROL),
                    StrokeKind::Inside,
                );
                ui.painter().text(
                    one_rect.center(),
                    Align2::CENTER_CENTER,
                    "1F",
                    theme.font(FONT_SMALL),
                    theme.value(),
                );
                if o_resp.clicked() {
                    self.player_plus.single_render_requested = true;
                    actions.push(UiAction::SingleFrameRender);
                }
            }

            let q_rect = Rect::from_min_size(Pos2::new(cx, by + 3.0), Vec2::new(46.0, 23.0));
            let q_resp = ui
                .interact(q_rect, Id::unique("pp_quality"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text("Player+: cycle preview resolution quality");
            ui.painter()
                .rect_filled(q_rect, CornerRadius::ZERO, BG_INPUT);
            ui.painter().rect_stroke(
                q_rect,
                CornerRadius::ZERO,
                Stroke::new(1.0, BORDER_CONTROL),
                StrokeKind::Inside,
            );
            ui.painter().text(
                q_rect.center(),
                Align2::CENTER_CENTER,
                self.player_plus.preview_quality.label(),
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
        ui.painter().rect_stroke(
            rect,
            CornerRadius::ZERO,
            Stroke::new(1.0, border),
            StrokeKind::Inside,
        );
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
        let focused = self.focused_panel == PanelFocus::Timeline;
        ui.painter().rect_filled(rect, CornerRadius::ZERO, BG_PANEL);

        let inner = rect.shrink(1.0);
        if ui.input(|i| {
            i.pointer.primary_pressed()
                && i.pointer.interact_pos().is_some_and(|p| inner.contains(p))
        }) {
            self.focused_panel = PanelFocus::Timeline;
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
        ui.painter().rect_stroke(
            rect,
            CornerRadius::ZERO,
            Stroke::new(1.0, border),
            StrokeKind::Inside,
        );
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

        ctx.input(|i| {
            let ctrl = i.modifiers.ctrl || i.modifiers.command;
            let shift = i.modifiers.shift;
            let alt = i.modifiers.alt;

            // Project & History (also forwarded from floating windows)
            if ctrl && !alt && !shift && i.key_pressed(Key::Z) {
                let _ = session.execute(Command::Undo);
                return;
            }
            if ctrl && !alt && (i.key_pressed(Key::Y) || (shift && i.key_pressed(Key::Z))) {
                let _ = session.execute(Command::Redo);
                return;
            }
            if ctrl && !alt && i.key_pressed(Key::S) {
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
                actions.push(UiAction::OpenGraphEditor);
            } else if !ctrl && !alt && i.key_pressed(Key::Backtick) {
                if self.maximized_panel.is_some() {
                    self.maximized_panel = None;
                } else {
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
