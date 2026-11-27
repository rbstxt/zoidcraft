//! Desktop "No FFmpeg Inside" on-demand FFmpeg setup banner, modal dialog,
//! background installer worker, custom binary picker, and live re-probe controller.

use std::collections::VecDeque;

use super::*;
use crate::ui::controls::{ButtonVariant, cm3_prop_button};
use crate::ui::theme::{BORDER_CONTROL, DANGER_TEXT, FONT_BODY, TEXT_MUTED, TEXT_STRONG};

const MAX_INSTALL_LOG_LINES: usize = 120;
const INSTALL_SHUTDOWN_JOIN_TIMEOUT: Duration = Duration::from_millis(250);

pub(crate) enum FfmpegInstallEvent {
    Line(String),
    Finished(Result<zc_export::FfmpegInstallOutcome, String>),
}

pub(crate) struct ActiveFfmpegInstallJob {
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) rx: Receiver<FfmpegInstallEvent>,
    pub(crate) handle: Option<JoinHandle<()>>,
}

impl ActiveFfmpegInstallJob {
    pub(crate) fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    pub(crate) fn cancel_and_join_bounded(&mut self, timeout: Duration) {
        self.request_cancel();
        let Some(handle) = self.handle.take() else {
            return;
        };
        let deadline = Instant::now() + timeout;
        while !handle.is_finished() {
            let now = Instant::now();
            if now >= deadline {
                drop(handle);
                return;
            }
            thread::sleep(Duration::from_millis(5).min(deadline.saturating_duration_since(now)));
        }
        let _ = handle.join();
    }
}

impl Drop for ActiveFfmpegInstallJob {
    fn drop(&mut self) {
        self.cancel_and_join_bounded(INSTALL_SHUTDOWN_JOIN_TIMEOUT);
    }
}

#[derive(Default)]
pub(crate) struct FfmpegSetupState {
    pub(crate) open: bool,
    pub(crate) banner_visible: bool,
    pub(crate) plan: Option<zc_export::FfmpegInstallPlan>,
    pub(crate) active_job: Option<ActiveFfmpegInstallJob>,
    pub(crate) log_lines: VecDeque<String>,
    pub(crate) last_error: Option<String>,
    pub(crate) last_success: Option<String>,
    pub(crate) copied_notice: Option<String>,
}

impl FfmpegSetupState {
    pub(crate) fn is_running(&self) -> bool {
        self.active_job.is_some()
    }

    fn push_log_line(&mut self, line: String) {
        if self.log_lines.len() >= MAX_INSTALL_LOG_LINES {
            self.log_lines.pop_front();
        }
        self.log_lines.push_back(line);
    }
}

pub(crate) fn open_url_in_os_browser(url: &str) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let res = ProcessCommand::new("open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let res = ProcessCommand::new("cmd")
        .args(["/C", "start", "", url])
        .spawn();
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    let res = ProcessCommand::new("xdg-open").arg(url).spawn();

    res.map(|_| ()).map_err(|err| err.to_string())
}

impl ZoidCraftApp {
    /// Applies persisted custom `ffmpeg` / `ffprobe` paths from `AppSettings` to `zc-render`'s
    /// live media tool resolver (zero restart required).
    pub(crate) fn apply_ffmpeg_preferences_to_resolver(&self) {
        let settings = &self.workspace.panels.app_settings;
        let ffmpeg = settings
            .ffmpeg_path
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        let ffprobe = settings
            .ffprobe_path
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        zc_render::video::set_custom_media_tool_paths(ffmpeg, ffprobe);
    }

    /// Clears cached binary paths and starts a fresh non-blocking `ffmpeg -version` probe.
    pub(crate) fn trigger_ffmpeg_reprobe(&mut self) {
        self.apply_ffmpeg_preferences_to_resolver();
        zc_render::video::refresh_media_tool_cache();
        self.workspace.panels.export_status.ffmpeg_available = false;
        self.workspace.panels.export_status.ffmpeg_version = "checking...".to_owned();
        let (tx, rx) = mpsc::channel::<Result<String, ExportError>>();
        thread::spawn(move || {
            let _ = tx.send(probe_ffmpeg(None));
        });
        self.ffmpeg_probe_rx = Some(rx);
    }

    pub(crate) fn open_ffmpeg_setup_modal(&mut self) {
        self.ffmpeg_setup.plan = Some(zc_export::detect_ffmpeg_install_plan());
        self.ffmpeg_setup.open = true;
        self.ffmpeg_setup.banner_visible = false;
        self.ffmpeg_setup.copied_notice = None;
        if !self.ffmpeg_setup.is_running() {
            self.ffmpeg_setup.last_error = None;
        }
    }

    pub(crate) fn start_ffmpeg_install(&mut self) {
        if self.ffmpeg_setup.is_running() {
            return;
        }
        let plan = self
            .ffmpeg_setup
            .plan
            .clone()
            .unwrap_or_else(zc_export::detect_ffmpeg_install_plan);
        self.ffmpeg_setup.plan = Some(plan.clone());
        if !plan.can_run_automatically {
            self.ffmpeg_setup.last_error = Some(format!(
                "Automatic installation is not available ({}). Copy the command above or use Locate Binary.",
                plan.manager_name
            ));
            return;
        }

        self.ffmpeg_setup.log_lines.clear();
        self.ffmpeg_setup.last_error = None;
        self.ffmpeg_setup.last_success = None;
        self.ffmpeg_setup.copied_notice = None;

        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = Arc::clone(&cancel);
        let (tx, rx) = mpsc::channel::<FfmpegInstallEvent>();
        let tx_lines = tx.clone();

        let spawned = thread::Builder::new()
            .name("zc-ffmpeg-install".to_owned())
            .spawn(move || {
                let res = zc_export::run_ffmpeg_install(&plan, &cancel_worker, move |line| {
                    let _ = tx_lines.send(FfmpegInstallEvent::Line(line));
                })
                .map_err(|err| err.to_string());
                let _ = tx.send(FfmpegInstallEvent::Finished(res));
            });

        match spawned {
            Ok(handle) => {
                self.ffmpeg_setup.active_job = Some(ActiveFfmpegInstallJob {
                    cancel,
                    rx,
                    handle: Some(handle),
                });
            }
            Err(err) => {
                self.ffmpeg_setup.last_error =
                    Some(format!("Failed to start installer thread: {err}"));
            }
        }
    }

    pub(crate) fn cancel_ffmpeg_install(&mut self) {
        if let Some(job) = &self.ffmpeg_setup.active_job {
            job.request_cancel();
            self.ffmpeg_setup
                .push_log_line("Cancelling installation...".to_owned());
        }
    }

    pub(crate) fn poll_ffmpeg_install(&mut self, ctx: &egui::Context) {
        let mut finished: Option<Result<zc_export::FfmpegInstallOutcome, String>> = None;
        let mut disconnected = false;
        let mut new_lines: Vec<String> = Vec::new();

        if let Some(job) = &self.ffmpeg_setup.active_job {
            loop {
                match job.rx.try_recv() {
                    Ok(FfmpegInstallEvent::Line(line)) => new_lines.push(line),
                    Ok(FfmpegInstallEvent::Finished(res)) => finished = Some(res),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }

        for line in new_lines {
            self.ffmpeg_setup.push_log_line(line);
        }

        if finished.is_none() && disconnected && self.ffmpeg_setup.active_job.is_some() {
            finished = Some(Err("installer thread exited unexpectedly".to_owned()));
        }

        if self.ffmpeg_setup.active_job.is_some() && finished.is_none() {
            ctx.request_repaint_after(Duration::from_millis(50));
            return;
        }

        if let Some(res) = finished {
            if let Some(mut job) = self.ffmpeg_setup.active_job.take()
                && let Some(handle) = job.handle.take()
            {
                let _ = handle.join();
            }
            match res {
                Ok(outcome) => {
                    self.apply_verified_ffmpeg_outcome(outcome, false);
                }
                Err(err) => {
                    self.ffmpeg_setup.last_error = Some(err.clone());
                    self.show_toast(format!("FFmpeg installation did not complete: {err}"));
                }
            }
        }
    }

    fn apply_verified_ffmpeg_outcome(
        &mut self,
        outcome: zc_export::FfmpegInstallOutcome,
        save_custom_override: bool,
    ) {
        if save_custom_override {
            self.workspace.panels.app_settings.ffmpeg_path =
                Some(outcome.ffmpeg_path.to_string_lossy().into_owned());
            self.workspace.panels.app_settings.ffprobe_path = outcome
                .ffprobe_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned());
            self.apply_ffmpeg_preferences_to_resolver();
        } else {
            zc_render::video::refresh_media_tool_cache();
        }

        self.workspace.panels.export_status.ffmpeg_available = true;
        self.workspace.panels.export_status.ffmpeg_version = outcome.version.clone();
        self.workspace.panels.export_status.ffmpeg_resolved_path =
            Some(outcome.ffmpeg_path.to_string_lossy().into_owned());
        self.ffmpeg_setup.banner_visible = false;
        self.ffmpeg_setup.last_error = None;
        self.ffmpeg_setup.last_success = Some(format!(
            "Ready: {} ({})",
            outcome.version,
            outcome.ffmpeg_path.display()
        ));
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.invalidate();
        }
        self.force_single_render = true;
        self.show_toast(format!("FFmpeg ready: {}", outcome.version));
    }

    pub(crate) fn handle_browse_ffmpeg_binary(&mut self) {
        self.pause_playback_for_modal_dialog();
        #[cfg(windows)]
        let dialog = rfd::FileDialog::new()
            .set_title("Locate FFmpeg executable")
            .add_filter("FFmpeg Executable", &["exe"]);
        #[cfg(not(windows))]
        let dialog = rfd::FileDialog::new().set_title("Locate FFmpeg executable");
        let chosen = dialog.pick_file();
        self.last_tick = Instant::now();
        let Some(path) = chosen else {
            return;
        };

        match zc_export::verify_custom_ffmpeg_binary(&path) {
            Ok(outcome) => {
                self.apply_verified_ffmpeg_outcome(outcome, true);
            }
            Err(err) => {
                let msg = format!("Invalid FFmpeg executable ({}): {err}", path.display());
                self.ffmpeg_setup.last_error = Some(msg.clone());
                self.show_toast(msg);
            }
        }
    }

    pub(crate) fn handle_clear_ffmpeg_binary(&mut self) {
        self.workspace.panels.app_settings.ffmpeg_path = None;
        self.workspace.panels.app_settings.ffprobe_path = None;
        self.workspace.panels.export_status.ffmpeg_resolved_path = None;
        self.ffmpeg_setup.last_error = None;
        self.ffmpeg_setup.last_success = None;
        self.trigger_ffmpeg_reprobe();
        self.show_toast("Reset FFmpeg path to auto-detect");
    }

    /// Renders a non-blocking bottom-center banner on launch (or video import) when FFmpeg is
    /// unavailable, giving 1-click access to the setup modal without blocking the editor.
    pub(crate) fn show_ffmpeg_setup_banner(&mut self, ctx: &egui::Context) {
        if !self.ffmpeg_setup.banner_visible
            || self.ffmpeg_setup.open
            || self.workspace.panels.export_status.ffmpeg_available
            || self.startup_recovery_banner.is_some()
            || self.pending_eyedropper.is_some()
        {
            return;
        }

        let mut install_clicked = false;
        let mut skip_clicked = false;
        let (banner_w, clamp_bounds) = self
            .last_viewport_rect
            .and_then(|vp| compute_viewport_banner_bounds(vp, 520.0))
            .map_or((460.0, None), |(w, b)| (w, Some(b)));
        let inner_w = (banner_w - 18.0).max(140.0);
        let narrow = banner_w < 380.0;
        let text_font = self
            .theme
            .font(if narrow { FONT_COMPACT } else { FONT_SMALL });

        let mut area = Area::new(Id::unique("zc_ffmpeg_setup_banner")).order(Order::Middle);
        if let Some(bounds) = clamp_bounds {
            let left = (bounds.center().x - banner_w * 0.5)
                .clamp(
                    bounds.left(),
                    (bounds.right() - banner_w).max(bounds.left()),
                )
                .round();
            area = area
                .constrain_to(bounds)
                .pivot(Align2::LEFT_BOTTOM)
                .fixed_pos(Pos2::new(left, bounds.bottom().round()));
        } else {
            area = area.anchor(Align2::CENTER_BOTTOM, Vec2::new(0.0, -14.0));
        }

        area.show(ctx, |ui| {
            Frame::NONE
                .fill(BG_TITLE)
                .stroke(Stroke::new(1.0, self.theme.accent()))
                .inner_margin(egui::Margin::symmetric(8, 5))
                .show(ui, |ui| {
                    ui.set_min_width(inner_w);
                    ui.set_max_width(inner_w);
                    ui.allocate_ui_with_layout(
                        Vec2::new(inner_w, 24.0),
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            ui.spacing_mut().item_spacing.x = 6.0;
                            let skip_btn = egui::Button::new(
                                egui::RichText::new("Skip")
                                    .font(text_font.clone())
                                    .color(TEXT_PRIMARY),
                            )
                            .fill(BG_BUTTON)
                            .stroke(Stroke::new(1.0, BORDER_PANEL))
                            .corner_radius(CornerRadius::same(2))
                            .min_size(Vec2::new(0.0, 22.0));
                            if ui.add(skip_btn).clicked() {
                                skip_clicked = true;
                            }

                            let install_btn = egui::Button::new(
                                egui::RichText::new("Install FFmpeg...")
                                    .font(text_font.clone())
                                    .color(TEXT_BRIGHT),
                            )
                            .fill(Color32::from_rgb(0x2d, 0x67, 0xc4))
                            .stroke(Stroke::new(1.0, Color32::from_rgb(0x52, 0x8b, 0xe4)))
                            .corner_radius(CornerRadius::same(2))
                            .min_size(Vec2::new(0.0, 22.0));
                            if ui.add(install_btn).clicked() {
                                install_clicked = true;
                            }

                            let label_w = ui.available_width().max(40.0);
                            ui.add_sized(
                                [label_w, 20.0],
                                egui::Label::new(
                                    egui::RichText::new(
                                        "FFmpeg not found — needed for video import & MP4/WebM export",
                                    )
                                    .font(text_font)
                                    .color(TEXT_BRIGHT),
                                )
                                .truncate(),
                            )
                            .on_hover_text(
                                "ZoidCraft works without FFmpeg for 3D, PNG, and WAV. Install FFmpeg to unlock video import and MP4/WebM/MKV/GIF export.",
                            );
                        },
                    );
                });
        });

        if install_clicked {
            self.open_ffmpeg_setup_modal();
        } else if skip_clicked {
            self.ffmpeg_setup.banner_visible = false;
            self.workspace.panels.app_settings.ffmpeg_setup_dismissed = true;
        }
    }

    /// Renders the centered FFmpeg Setup modal when opened on demand or from the setup banner.
    pub(crate) fn show_ffmpeg_setup_modal(&mut self, ctx: &egui::Context) {
        if !self.ffmpeg_setup.open {
            return;
        }

        let plan = self
            .ffmpeg_setup
            .plan
            .clone()
            .unwrap_or_else(zc_export::detect_ffmpeg_install_plan);
        let is_running = self.ffmpeg_setup.is_running();
        let ffmpeg_available = self.workspace.panels.export_status.ffmpeg_available;
        let ffmpeg_version = self.workspace.panels.export_status.ffmpeg_version.clone();
        let resolved_path = self
            .workspace
            .panels
            .export_status
            .ffmpeg_resolved_path
            .clone();

        let mut start_install = false;
        let mut cancel_install = false;
        let mut browse_binary = false;
        let mut reprobe = false;
        let mut open_download = false;
        let mut copy_command = false;
        let mut close_modal = false;

        let screen = ctx.input(|i| i.viewport_rect());
        let modal_w = (screen.width() - 40.0).clamp(320.0, 520.0);

        Area::new(Id::unique("zc_ffmpeg_setup_scrim"))
            .order(Order::Foreground)
            .fixed_pos(screen.min)
            .show(ctx, |ui| {
                let _ = ui.allocate_rect(screen, egui::Sense::click());
                ui.painter().rect_filled(
                    screen,
                    CornerRadius::ZERO,
                    Color32::from_rgba_unmultiplied(0, 0, 0, 140),
                );
            });

        egui::Window::new("FFmpeg Setup")
            .id(Id::unique("zc_ffmpeg_setup_modal"))
            .order(Order::Foreground)
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .min_width(modal_w)
            .max_width(modal_w)
            .frame(
                Frame::NONE
                    .fill(BG_PANEL)
                    .stroke(Stroke::new(1.0, self.theme.accent()))
                    .inner_margin(egui::Margin::same(14)),
            )
            .show(ctx, |ui| {
                ui.spacing_mut().item_spacing.y = 8.0;

                ui.add(
                    egui::Label::new(
                        egui::RichText::new(
                            "ZoidCraft keeps FFmpeg external (No FFmpeg Inside). 3D scenes, effects, PNG/JPEG/WebP capture, and WAV audio work out of the box. Install FFmpeg to enable video clip decoding and MP4 / WebM / MKV / GIF / M4A export.",
                        )
                        .font(self.theme.font(FONT_SMALL))
                        .color(TEXT_PRIMARY),
                    )
                    .wrap(),
                );

                // Status card
                Frame::NONE
                    .fill(BG_TITLE)
                    .stroke(Stroke::new(1.0, BORDER_CONTROL))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.spacing_mut().item_spacing.y = 3.0;
                        let (badge, color) = if ffmpeg_available {
                            (format!("Status: Available ({ffmpeg_version})"), Color32::from_rgb(76, 198, 122))
                        } else if is_running {
                            ("Status: Installing FFmpeg...".to_owned(), self.theme.value())
                        } else {
                            (format!("Status: Not found ({ffmpeg_version})"), DANGER_TEXT)
                        };
                        ui.label(
                            egui::RichText::new(badge)
                                .font(self.theme.font(FONT_BODY))
                                .color(color),
                        );
                        if let Some(ref p) = resolved_path {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(format!("Binary: {p}"))
                                        .font(self.theme.font(FONT_COMPACT))
                                        .color(TEXT_MUTED),
                                )
                                .truncate(),
                            );
                        }
                    });

                // Detected installer + transparent command preview
                Frame::NONE
                    .fill(BG_SUNKEN)
                    .stroke(Stroke::new(1.0, BORDER_CONTROL))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.spacing_mut().item_spacing.y = 4.0;
                        ui.label(
                            egui::RichText::new(format!("Recommended installer: {}", plan.manager_name))
                                .font(self.theme.font(FONT_SMALL))
                                .color(TEXT_STRONG),
                        );
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 6.0;
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(&plan.display_command)
                                        .monospace()
                                        .size(12.0)
                                        .color(TEXT_BRIGHT),
                                )
                                .wrap(),
                            );
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if cm3_prop_button(ui, "Copy", ButtonVariant::Normal, false, &self.theme)
                                    .clicked()
                                {
                                    copy_command = true;
                                }
                            });
                        });
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(&plan.manual_hint)
                                    .font(self.theme.font(FONT_COMPACT))
                                    .color(TEXT_MUTED),
                            )
                            .wrap(),
                        );
                        if let Some(ref notice) = self.ffmpeg_setup.copied_notice {
                            ui.label(
                                egui::RichText::new(notice)
                                    .font(self.theme.font(FONT_COMPACT))
                                    .color(self.theme.value()),
                            );
                        }
                    });

                // Primary action row
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    if is_running {
                        if cm3_prop_button(
                            ui,
                            "Cancel Install",
                            ButtonVariant::Danger,
                            false,
                            &self.theme,
                        )
                        .clicked()
                        {
                            cancel_install = true;
                        }
                    } else if plan.can_run_automatically
                        && cm3_prop_button(
                            ui,
                            &format!("Install via {}", plan.manager_name),
                            ButtonVariant::Primary,
                            false,
                            &self.theme,
                        )
                        .clicked()
                    {
                        start_install = true;
                    }

                    if !is_running {
                        if cm3_prop_button(
                            ui,
                            "Locate Binary...",
                            ButtonVariant::Normal,
                            false,
                            &self.theme,
                        )
                        .clicked()
                        {
                            browse_binary = true;
                        }
                        if cm3_prop_button(
                            ui,
                            "Download Page",
                            ButtonVariant::Normal,
                            false,
                            &self.theme,
                        )
                        .clicked()
                        {
                            open_download = true;
                        }
                        if cm3_prop_button(
                            ui,
                            "Re-check",
                            ButtonVariant::Normal,
                            false,
                            &self.theme,
                        )
                        .clicked()
                        {
                            reprobe = true;
                        }
                    }
                });

                // Live bounded installer output log
                if !self.ffmpeg_setup.log_lines.is_empty() {
                    Frame::NONE
                        .fill(BG_SUNKEN)
                        .stroke(Stroke::new(1.0, BORDER_CONTROL))
                        .inner_margin(egui::Margin::same(6))
                        .show(ui, |ui| {
                            egui::ScrollArea::vertical()
                                .id_salt("zc_ffmpeg_install_log")
                                .max_height(130.0)
                                .stick_to_bottom(true)
                                .show(ui, |ui| {
                                    ui.set_min_width(ui.available_width());
                                    for line in &self.ffmpeg_setup.log_lines {
                                        ui.label(
                                            egui::RichText::new(line)
                                                .monospace()
                                                .size(11.0)
                                                .color(TEXT_MUTED),
                                        );
                                    }
                                });
                        });
                }

                if let Some(ref err) = self.ffmpeg_setup.last_error {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(err)
                                .font(self.theme.font(FONT_SMALL))
                                .color(DANGER_TEXT),
                        )
                        .wrap(),
                    );
                }
                if let Some(ref ok) = self.ffmpeg_setup.last_success {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(ok)
                                .font(self.theme.font(FONT_SMALL))
                                .color(Color32::from_rgb(76, 198, 122)),
                        )
                        .wrap(),
                    );
                }

                // Footer row: startup preference + Close
                ui.horizontal(|ui| {
                    ui.checkbox(
                        &mut self.workspace.panels.app_settings.ffmpeg_setup_dismissed,
                        egui::RichText::new("Don't prompt on startup")
                            .font(self.theme.font(FONT_SMALL))
                            .color(TEXT_MUTED),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let close_lbl = if ffmpeg_available { "Done" } else { "Close" };
                        if cm3_prop_button(ui, close_lbl, ButtonVariant::Normal, false, &self.theme)
                            .clicked()
                        {
                            close_modal = true;
                        }
                    });
                });
            });

        if copy_command {
            ui::clipboard::write_system_clipboard(ctx, &plan.display_command);
            self.ffmpeg_setup.copied_notice = Some("Copied command to clipboard.".to_owned());
        }
        if start_install {
            self.start_ffmpeg_install();
        }
        if cancel_install {
            self.cancel_ffmpeg_install();
        }
        if browse_binary {
            self.handle_browse_ffmpeg_binary();
        }
        if open_download && let Err(err) = open_url_in_os_browser(plan.download_url) {
            self.ffmpeg_setup.last_error = Some(format!("Failed to open browser: {err}"));
        }
        if reprobe {
            self.trigger_ffmpeg_reprobe();
        }
        if close_modal {
            self.ffmpeg_setup.open = false;
        }
    }
}
