//! Desktop update checker, GitHub Releases metadata parser, platform installer
//! selector, non-blocking top-center update banner, and Update Available modal.
//!
//! Uses zero external HTTP crates (OS-native `curl` on Unix, PowerShell
//! `Invoke-WebRequest` on Windows) and strictly forbids unsafe code.

use super::*;
use crate::ui::controls::{ButtonVariant, cm3_prop_button};
use crate::ui::theme::{BORDER_CONTROL, DANGER_TEXT, FONT_BODY, TEXT_MUTED, TEXT_STRONG};

pub(crate) const GITHUB_LATEST_RELEASE_API: &str =
    "https://api.github.com/repos/rbstxt/zoidcraft/releases/latest";
pub(crate) const GITHUB_RELEASES_PAGE: &str = "https://github.com/rbstxt/zoidcraft/releases";
pub(crate) const UPDATE_CHECK_TIMEOUT_SECS: u64 = 8;

const MAX_NOTES_LINES: usize = 16;
const MAX_NOTES_CHARS: usize = 800;

/// Strict `MAJOR.MINOR.PATCH` semantic version (prerelease tags excluded).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct SemVer {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
}

/// Parses a stable semantic version string such as `"0.2.7"`, `"v0.2.7"`, or
/// `"V1.2.3+build.4"`. Rejects prerelease suffixes (`-alpha`, `-beta`, `-rc`, etc.).
#[must_use]
pub(crate) fn parse_semver(raw: &str) -> Option<SemVer> {
    let trimmed = raw.trim();
    let stripped = trimmed
        .strip_prefix('v')
        .or_else(|| trimmed.strip_prefix('V'))
        .unwrap_or(trimmed);
    let without_build = match stripped.split_once('+') {
        Some((base, _build)) => base,
        None => stripped,
    };
    if without_build.contains('-') {
        return None;
    }

    let mut parts = without_build.split('.');
    let major_s = parts.next()?;
    let minor_s = parts.next()?;
    let patch_s = parts.next()?;
    if parts.next().is_some() {
        return None;
    }

    for part in [major_s, minor_s, patch_s] {
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
    }

    Some(SemVer {
        major: major_s.parse::<u64>().ok()?,
        minor: minor_s.parse::<u64>().ok()?,
        patch: patch_s.parse::<u64>().ok()?,
    })
}

/// Returns `true` iff both `current` and `candidate` parse cleanly as stable
/// semantic versions and `candidate > current`.
#[must_use]
pub(crate) fn is_newer_version(current: &str, candidate: &str) -> bool {
    matches!(
        (parse_semver(current), parse_semver(candidate)),
        (Some(cur), Some(cand)) if cand > cur
    )
}

/// Validated metadata for an available GitHub release update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReleaseUpdateInfo {
    pub version: String,
    pub tag_name: String,
    pub release_name: String,
    pub release_url: String,
    pub download_url: String,
    pub download_label: String,
    pub asset_name: Option<String>,
    pub notes_summary: String,
    pub published_at: Option<String>,
}

/// Validates that `url` is a safe `https://` URL free of whitespace, control
/// characters, and shell metacharacters before handing it to the OS browser.
#[must_use]
pub(crate) fn is_safe_https_url(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("https://") else {
        return false;
    };
    if rest.is_empty() || rest.starts_with('/') {
        return false;
    }
    !url.chars().any(|c| {
        c.is_whitespace()
            || c.is_control()
            || matches!(
                c,
                '"' | '\'' | '`' | '&' | '|' | ';' | '<' | '>' | '$' | '\n' | '\r'
            )
    })
}

/// Selects the best matching platform installer/archive asset from `(name, url)` pairs
/// published by `.github/workflows/release.yml`, rejecting any unsafe download URL.
#[must_use]
pub(crate) fn select_platform_download_asset(
    assets: &[(String, String)],
    os: &str,
    arch: &str,
) -> Option<(String, String)> {
    let safe_assets: Vec<(&str, &str, String)> = assets
        .iter()
        .filter(|(name, url)| !name.trim().is_empty() && is_safe_https_url(url))
        .map(|(name, url)| (name.as_str(), url.as_str(), name.to_ascii_lowercase()))
        .collect();

    let pick = |predicate: fn(&str) -> bool| -> Option<(String, String)> {
        safe_assets
            .iter()
            .find(|(_, _, lower)| predicate(lower))
            .map(|(name, url, _)| ((*name).to_owned(), (*url).to_owned()))
    };

    let os_lower = os.to_ascii_lowercase();
    let arch_lower = arch.to_ascii_lowercase();

    match (os_lower.as_str(), arch_lower.as_str()) {
        ("macos", "aarch64") => {
            pick(|n| n.ends_with(".dmg") && (n.contains("arm64") || n.contains("aarch64"))).or_else(
                || {
                    pick(|n| {
                        n.ends_with(".tar.gz") && (n.contains("arm64") || n.contains("aarch64"))
                    })
                },
            )
        }
        ("macos", "x86_64") => {
            pick(|n| n.ends_with(".dmg") && (n.contains("x86_64") || n.contains("x64"))).or_else(
                || pick(|n| n.ends_with(".tar.gz") && (n.contains("x86_64") || n.contains("x64"))),
            )
        }
        ("windows", _) => pick(|n| n.ends_with("setup.exe"))
            .or_else(|| pick(|n| n.ends_with(".exe")))
            .or_else(|| {
                pick(|n| {
                    n.ends_with("portable.zip")
                        && !n.contains("macos")
                        && !n.contains("darwin")
                        && !n.contains("linux")
                })
            })
            .or_else(|| {
                pick(|n| n.ends_with(".zip") && (n.contains("windows") || n.contains("win")))
            })
            .or_else(|| {
                pick(|n| {
                    n.ends_with(".zip")
                        && !n.contains("macos")
                        && !n.contains("darwin")
                        && !n.contains("linux")
                })
            }),
        ("linux", _) => pick(|n| n.ends_with(".deb") && n.contains("linux"))
            .or_else(|| pick(|n| n.ends_with(".deb")))
            .or_else(|| pick(|n| n.ends_with(".tar.gz") && n.contains("linux")))
            .or_else(|| {
                pick(|n| {
                    n.ends_with(".tar.gz")
                        && !n.contains("macos")
                        && !n.contains("darwin")
                        && !n.contains("windows")
                        && !n.contains("win")
                })
            }),
        _ => None,
    }
}

fn platform_download_label(os: &str) -> String {
    match os.to_ascii_lowercase().as_str() {
        "macos" => "Download for macOS".to_owned(),
        "windows" => "Download for Windows".to_owned(),
        "linux" => "Download for Linux".to_owned(),
        _ => "Download Update".to_owned(),
    }
}

fn extract_published_date(raw: Option<&str>) -> Option<String> {
    let s = raw?.trim();
    let date = s.get(..10)?;
    let bytes = date.as_bytes();
    if bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes[..4].iter().all(u8::is_ascii_digit)
        && bytes[5..7].iter().all(u8::is_ascii_digit)
        && bytes[8..10].iter().all(u8::is_ascii_digit)
    {
        Some(date.to_owned())
    } else {
        None
    }
}

fn summarize_release_notes(raw: &str) -> String {
    let normalized = raw.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines: Vec<&str> = Vec::new();
    let mut truncated_by_lines = false;
    let mut prev_blank = true;

    for raw_line in normalized.split('\n') {
        let line = raw_line.trim_end();
        let is_blank = line.trim().is_empty();
        if is_blank {
            if prev_blank {
                continue;
            }
            prev_blank = true;
            if lines.len() < MAX_NOTES_LINES {
                lines.push("");
            }
            continue;
        }
        prev_blank = false;
        if lines.len() >= MAX_NOTES_LINES {
            truncated_by_lines = true;
            break;
        }
        lines.push(line);
    }

    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }

    let mut joined = lines.join("\n");
    let char_count = joined.chars().count();
    if char_count > MAX_NOTES_CHARS || (truncated_by_lines && char_count + 2 > MAX_NOTES_CHARS) {
        let mut clipped: String = joined.chars().take(MAX_NOTES_CHARS - 1).collect();
        let trimmed_len = clipped.trim_end().len();
        clipped.truncate(trimmed_len);
        clipped.push('…');
        return clipped;
    }

    if truncated_by_lines && !joined.is_empty() {
        joined.push_str("\n…");
    }
    joined
}

fn truncate_single_line(text: &str, max_chars: usize) -> String {
    let clean = text.trim();
    if clean.chars().count() <= max_chars {
        return clean.to_owned();
    }
    let mut out: String = clean.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Parses a GitHub Releases `/releases/latest` JSON payload and returns
/// `Ok(Some(ReleaseUpdateInfo))` iff it represents a newer stable release.
pub(crate) fn parse_github_release_json(
    body: &str,
    current_version: &str,
    os: &str,
    arch: &str,
) -> Result<Option<ReleaseUpdateInfo>, String> {
    let json: Value =
        serde_json::from_str(body).map_err(|err| format!("Invalid release metadata: {err}"))?;
    if !json.is_object() {
        return Err("Invalid release metadata: expected JSON object".to_owned());
    }

    if json.get("draft").and_then(Value::as_bool).unwrap_or(false)
        || json
            .get("prerelease")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        return Ok(None);
    }

    let tag_name = json
        .get("tag_name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "Release metadata missing tag_name".to_owned())?;

    let Some(parsed_tag) = parse_semver(tag_name) else {
        return Ok(None);
    };
    if !is_newer_version(current_version, tag_name) {
        return Ok(None);
    }

    let version = format!(
        "{}.{}.{}",
        parsed_tag.major, parsed_tag.minor, parsed_tag.patch
    );
    let release_name = json
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("ZoidCraft v{version}"));

    let release_url = json
        .get("html_url")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|u| is_safe_https_url(u))
        .map(str::to_owned)
        .unwrap_or_else(|| GITHUB_RELEASES_PAGE.to_owned());

    let assets: Vec<(String, String)> = json
        .get("assets")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let name = item.get("name").and_then(Value::as_str)?.trim();
                    let url = item
                        .get("browser_download_url")
                        .and_then(Value::as_str)?
                        .trim();
                    if name.is_empty() || url.is_empty() {
                        None
                    } else {
                        Some((name.to_owned(), url.to_owned()))
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let (download_url, download_label, asset_name) =
        if let Some((name, url)) = select_platform_download_asset(&assets, os, arch) {
            (url, platform_download_label(os), Some(name))
        } else {
            (release_url.clone(), "Open Release Page".to_owned(), None)
        };

    let notes_summary =
        summarize_release_notes(json.get("body").and_then(Value::as_str).unwrap_or_default());
    let published_at = extract_published_date(json.get("published_at").and_then(Value::as_str));

    Ok(Some(ReleaseUpdateInfo {
        version,
        tag_name: tag_name.to_owned(),
        release_name,
        release_url,
        download_url,
        download_label,
        asset_name,
        notes_summary,
        published_at,
    }))
}

#[cfg(not(target_os = "windows"))]
fn fetch_latest_release_json(user_agent: &str) -> Result<String, String> {
    let output = ProcessCommand::new("curl")
        .args([
            "-fsSL",
            "--max-time",
            &UPDATE_CHECK_TIMEOUT_SECS.to_string(),
            "-H",
            "Accept: application/vnd.github+json",
            "-H",
            &format!("User-Agent: {user_agent}"),
            GITHUB_LATEST_RELEASE_API,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("Unable to run curl: {err}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr.lines().next().unwrap_or("").trim();
        if detail.is_empty() {
            return Err(format!("HTTP request failed ({})", output.status));
        }
        return Err(truncate_single_line(detail, 120));
    }

    String::from_utf8(output.stdout).map_err(|_| "Response was not valid UTF-8".to_owned())
}

#[cfg(target_os = "windows")]
fn fetch_latest_release_json(user_agent: &str) -> Result<String, String> {
    use std::os::windows::process::CommandExt;

    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let safe_ua = user_agent.replace('\'', "''");
    let script = format!(
        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; \
         $ProgressPreference = 'SilentlyContinue'; \
         (Invoke-WebRequest -UseBasicParsing -TimeoutSec {UPDATE_CHECK_TIMEOUT_SECS} \
         -Headers @{{Accept='application/vnd.github+json'; 'User-Agent'='{safe_ua}'}} \
         -Uri '{GITHUB_LATEST_RELEASE_API}').Content"
    );
    let output = ProcessCommand::new("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            &script,
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("Unable to run PowerShell: {err}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr.lines().next().unwrap_or("").trim();
        if detail.is_empty() {
            return Err(format!("HTTP request failed ({})", output.status));
        }
        return Err(truncate_single_line(detail, 120));
    }

    String::from_utf8(output.stdout).map_err(|_| "Response was not valid UTF-8".to_owned())
}

type UpdateCheckResult = Result<Option<ReleaseUpdateInfo>, String>;

/// Runtime state for background update checks, banner dismissal, and modal display.
#[derive(Default)]
pub(crate) struct UpdateCheckState {
    pub(crate) dialog_open: bool,
    pub(crate) banner_dismissed_version: Option<String>,
    pub(crate) pending_rx: Option<Receiver<UpdateCheckResult>>,
    pub(crate) pending_manual: bool,
    pub(crate) available_update: Option<ReleaseUpdateInfo>,
    pub(crate) last_error: Option<String>,
    pub(crate) checked_once: bool,
}

impl ZoidCraftApp {
    /// Starts a non-blocking background GitHub Releases check.
    ///
    /// If a check is already running, promotes `pending_manual` if `manual` is true
    /// without spawning a duplicate request.
    pub(crate) fn trigger_update_check(&mut self, manual: bool) {
        if self.update_check.pending_rx.is_some() {
            self.update_check.pending_manual |= manual;
            return;
        }

        self.update_check.pending_manual = manual;
        self.update_check.checked_once = true;
        if manual {
            self.update_check.last_error = None;
        }
        self.workspace.panels.update_status.checking = true;
        self.workspace.panels.update_status.status_message = "Checking for updates...".to_owned();

        let (tx, rx) = mpsc::channel::<UpdateCheckResult>();
        let user_agent = format!("ZoidCraft/{}", env!("CARGO_PKG_VERSION"));
        let current_version = env!("CARGO_PKG_VERSION");
        let os = std::env::consts::OS;
        let arch = std::env::consts::ARCH;

        let spawned = thread::Builder::new()
            .name("zc-update-check".to_owned())
            .spawn(move || {
                let res = fetch_latest_release_json(&user_agent)
                    .and_then(|body| parse_github_release_json(&body, current_version, os, arch));
                let _ = tx.send(res);
            });

        match spawned {
            Ok(_) => {
                self.update_check.pending_rx = Some(rx);
            }
            Err(err) => {
                let msg = format!("Failed to start update check: {err}");
                self.workspace.panels.update_status.checking = false;
                self.workspace.panels.update_status.available = false;
                self.workspace.panels.update_status.status_message = if manual {
                    format!("Check failed: {msg}")
                } else {
                    "Offline / check skipped".to_owned()
                };
                if manual {
                    self.update_check.last_error = Some(msg.clone());
                    self.show_toast(msg);
                }
            }
        }
    }

    /// Polls any in-flight background update check without blocking the UI thread.
    pub(crate) fn poll_update_check(&mut self, ctx: &egui::Context) {
        let outcome = match self.update_check.pending_rx.as_ref() {
            Some(rx) => match rx.try_recv() {
                Ok(res) => res,
                Err(TryRecvError::Empty) => {
                    ctx.request_repaint_after(Duration::from_millis(120));
                    return;
                }
                Err(TryRecvError::Disconnected) => {
                    Err("Update check thread exited unexpectedly".to_owned())
                }
            },
            None => return,
        };

        self.update_check.pending_rx = None;
        let manual = std::mem::take(&mut self.update_check.pending_manual);

        match outcome {
            Ok(Some(info)) => {
                let was_skipped = self
                    .workspace
                    .panels
                    .app_settings
                    .skipped_update_version
                    .as_deref()
                    == Some(info.version.as_str());
                self.update_check.available_update = Some(info.clone());
                self.update_check.last_error = None;
                self.workspace.panels.update_status.checking = false;
                self.workspace.panels.update_status.available = true;
                self.workspace.panels.update_status.latest_version = Some(info.version.clone());
                self.workspace.panels.update_status.status_message =
                    format!("Update available: v{}", info.version);

                if manual {
                    self.update_check.banner_dismissed_version = None;
                    if was_skipped {
                        self.update_check.dialog_open = true;
                    }
                    self.show_toast(format!("Update available: v{}", info.version));
                }
            }
            Ok(None) => {
                self.update_check.available_update = None;
                self.update_check.last_error = None;
                self.update_check.dialog_open = false;
                self.workspace.panels.update_status.checking = false;
                self.workspace.panels.update_status.available = false;
                self.workspace.panels.update_status.latest_version = None;
                self.workspace.panels.update_status.status_message =
                    format!("Up to date (v{})", env!("CARGO_PKG_VERSION"));
                if manual {
                    self.show_toast(format!(
                        "ZoidCraft v{} is up to date",
                        env!("CARGO_PKG_VERSION")
                    ));
                }
            }
            Err(err) => {
                self.workspace.panels.update_status.checking = false;
                self.workspace.panels.update_status.available = false;
                self.workspace.panels.update_status.status_message = if manual {
                    format!("Check failed: {err}")
                } else {
                    "Offline / check skipped".to_owned()
                };
                if manual {
                    self.update_check.last_error = Some(err.clone());
                    self.show_toast(format!("Update check failed: {err}"));
                }
            }
        }
    }

    /// Opens the Update Available details modal when an update is already known,
    /// otherwise triggers a manual update check.
    pub(crate) fn open_update_details_modal(&mut self) {
        if self.update_check.available_update.is_some() {
            self.update_check.dialog_open = true;
        } else {
            self.trigger_update_check(true);
        }
    }

    /// Clears any persisted skipped update version and re-enables the update banner.
    pub(crate) fn clear_skipped_update_version(&mut self) {
        self.workspace.panels.app_settings.skipped_update_version = None;
        self.update_check.banner_dismissed_version = None;
        self.show_toast("Cleared skipped update version".to_owned());
    }

    /// Renders a non-intrusive top-center viewport banner when a newer stable release
    /// is available and has not been skipped or dismissed for this session.
    pub(crate) fn show_update_banner(&mut self, ctx: &egui::Context) {
        let Some(info) = self.update_check.available_update.clone() else {
            return;
        };
        if self.update_check.dialog_open
            || self.ffmpeg_setup.open
            || self.startup_recovery_banner.is_some()
            || self.pending_eyedropper.is_some()
            || self
                .workspace
                .panels
                .app_settings
                .skipped_update_version
                .as_deref()
                == Some(info.version.as_str())
            || self.update_check.banner_dismissed_version.as_deref() == Some(info.version.as_str())
        {
            return;
        }

        let mut download_clicked = false;
        let mut details_clicked = false;
        let mut skip_clicked = false;
        let mut dismiss_clicked = false;

        let (banner_w, clamp_bounds) = self
            .last_viewport_rect
            .and_then(|vp| compute_viewport_banner_bounds(vp, 540.0))
            .map_or((480.0, None), |(w, b)| (w, Some(b)));
        let inner_w = (banner_w - 18.0).max(160.0);
        let narrow = banner_w < 400.0;
        let text_font = self
            .theme
            .font(if narrow { FONT_COMPACT } else { FONT_SMALL });

        let mut area = Area::new(Id::unique("zc_update_available_banner")).order(Order::Middle);
        if let Some(bounds) = clamp_bounds {
            let left = (bounds.center().x - banner_w * 0.5)
                .clamp(
                    bounds.left(),
                    (bounds.right() - banner_w).max(bounds.left()),
                )
                .round();
            area = area
                .constrain_to(bounds)
                .fixed_pos(Pos2::new(left, bounds.top().round()));
        } else {
            area = area.anchor(Align2::CENTER_TOP, Vec2::new(0.0, 10.0));
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

                            let dismiss_btn = egui::Button::new(
                                egui::RichText::new("×")
                                    .font(self.theme.font(FONT_SMALL))
                                    .color(TEXT_PRIMARY),
                            )
                            .fill(BG_BUTTON)
                            .stroke(Stroke::new(1.0, BORDER_PANEL))
                            .corner_radius(CornerRadius::same(2))
                            .min_size(Vec2::new(22.0, 22.0));
                            if ui
                                .add(dismiss_btn)
                                .on_hover_text("Dismiss for this session")
                                .clicked()
                            {
                                dismiss_clicked = true;
                            }

                            let skip_btn = egui::Button::new(
                                egui::RichText::new("Skip")
                                    .font(text_font.clone())
                                    .color(TEXT_PRIMARY),
                            )
                            .fill(BG_BUTTON)
                            .stroke(Stroke::new(1.0, BORDER_PANEL))
                            .corner_radius(CornerRadius::same(2))
                            .min_size(Vec2::new(0.0, 22.0));
                            if ui
                                .add(skip_btn)
                                .on_hover_text(format!("Skip v{} until a newer release", info.version))
                                .clicked()
                            {
                                skip_clicked = true;
                            }

                            let details_btn = egui::Button::new(
                                egui::RichText::new("Details")
                                    .font(text_font.clone())
                                    .color(TEXT_PRIMARY),
                            )
                            .fill(BG_BUTTON)
                            .stroke(Stroke::new(1.0, BORDER_PANEL))
                            .corner_radius(CornerRadius::same(2))
                            .min_size(Vec2::new(0.0, 22.0));
                            if ui.add(details_btn).clicked() {
                                details_clicked = true;
                            }

                            let download_btn = egui::Button::new(
                                egui::RichText::new("Download")
                                    .font(text_font.clone())
                                    .color(TEXT_BRIGHT),
                            )
                            .fill(Color32::from_rgb(0x2d, 0x67, 0xc4))
                            .stroke(Stroke::new(1.0, Color32::from_rgb(0x52, 0x8b, 0xe4)))
                            .corner_radius(CornerRadius::same(2))
                            .min_size(Vec2::new(0.0, 22.0));
                            if ui
                                .add(download_btn)
                                .on_hover_text(&info.download_label)
                                .clicked()
                            {
                                download_clicked = true;
                            }

                            let label_w = ui.available_width().max(40.0);
                            let summary = if narrow {
                                format!("v{} available", info.version)
                            } else {
                                format!("ZoidCraft v{} available", info.version)
                            };
                            let tooltip = format!(
                                "{} (current: v{}). Click Details for release notes or Download to get the installer.",
                                info.release_name,
                                env!("CARGO_PKG_VERSION")
                            );
                            ui.add_sized(
                                [label_w, 20.0],
                                egui::Label::new(
                                    egui::RichText::new(summary)
                                        .font(text_font)
                                        .color(TEXT_BRIGHT),
                                )
                                .truncate(),
                            )
                            .on_hover_text(tooltip);
                        },
                    );
                });
        });

        if download_clicked {
            if let Err(err) = open_url_in_os_browser(&info.download_url) {
                self.show_toast(format!("Failed to open browser: {err}"));
            }
        } else if details_clicked {
            self.update_check.dialog_open = true;
        } else if skip_clicked {
            self.workspace.panels.app_settings.skipped_update_version = Some(info.version.clone());
            self.show_toast(format!("Skipped update v{}", info.version));
        } else if dismiss_clicked {
            self.update_check.banner_dismissed_version = Some(info.version);
        }
    }

    /// Renders the centered Update Available details modal with release notes and
    /// direct platform download actions.
    pub(crate) fn show_update_modal(&mut self, ctx: &egui::Context) {
        if !self.update_check.dialog_open {
            return;
        }
        let Some(info) = self.update_check.available_update.clone() else {
            self.update_check.dialog_open = false;
            return;
        };

        let mut open_download = false;
        let mut open_release_notes = false;
        let mut skip_version = false;
        let mut close_modal = ctx.input(|i| i.key_pressed(Key::Escape));

        let screen = ctx.input(|i| i.viewport_rect());
        let modal_w = (screen.width() - 40.0).clamp(320.0, 520.0);

        Area::new(Id::unique("zc_update_modal_scrim"))
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

        egui::Window::new("Update Available")
            .id(Id::unique("zc_update_modal"))
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

                // Version & package summary card
                Frame::NONE
                    .fill(BG_TITLE)
                    .stroke(Stroke::new(1.0, BORDER_CONTROL))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.spacing_mut().item_spacing.y = 3.0;
                        ui.label(
                            egui::RichText::new(&info.release_name)
                                .font(self.theme.font(FONT_BODY))
                                .color(TEXT_BRIGHT),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "Installed: v{}   →   Latest: v{}",
                                env!("CARGO_PKG_VERSION"),
                                info.version
                            ))
                            .font(self.theme.font(FONT_SMALL))
                            .color(self.theme.value()),
                        );
                        if let Some(ref date) = info.published_at {
                            ui.label(
                                egui::RichText::new(format!("Published: {date}"))
                                    .font(self.theme.font(FONT_COMPACT))
                                    .color(TEXT_MUTED),
                            );
                        }
                        if let Some(ref asset) = info.asset_name {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(format!("Package: {asset}"))
                                        .font(self.theme.font(FONT_COMPACT))
                                        .color(TEXT_MUTED),
                                )
                                .truncate(),
                            );
                        }
                    });

                // Scrollable release notes excerpt
                Frame::NONE
                    .fill(BG_SUNKEN)
                    .stroke(Stroke::new(1.0, BORDER_CONTROL))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.spacing_mut().item_spacing.y = 4.0;
                        ui.label(
                            egui::RichText::new("Release Notes")
                                .font(self.theme.font(FONT_SMALL))
                                .color(TEXT_STRONG),
                        );
                        if info.notes_summary.is_empty() {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(
                                        "No release notes were included. Open the GitHub release page for full details.",
                                    )
                                    .font(self.theme.font(FONT_COMPACT))
                                    .color(TEXT_MUTED),
                                )
                                .wrap(),
                            );
                        } else {
                            egui::ScrollArea::vertical()
                                .id_salt("zc_update_release_notes")
                                .max_height(180.0)
                                .show(ui, |ui| {
                                    ui.set_min_width(ui.available_width());
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(&info.notes_summary)
                                                .font(self.theme.font(FONT_SMALL))
                                                .color(TEXT_PRIMARY),
                                        )
                                        .wrap(),
                                    );
                                });
                        }
                    });

                if let Some(ref err) = self.update_check.last_error {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(err)
                                .font(self.theme.font(FONT_SMALL))
                                .color(DANGER_TEXT),
                        )
                        .wrap(),
                    );
                }

                // Action buttons row
                let skip_label = format!("Skip v{}", info.version);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    if cm3_prop_button(
                        ui,
                        &info.download_label,
                        ButtonVariant::Primary,
                        false,
                        &self.theme,
                    )
                    .clicked()
                    {
                        open_download = true;
                    }
                    if cm3_prop_button(
                        ui,
                        "Release Notes",
                        ButtonVariant::Normal,
                        false,
                        &self.theme,
                    )
                    .clicked()
                    {
                        open_release_notes = true;
                    }
                    if cm3_prop_button(ui, &skip_label, ButtonVariant::Normal, false, &self.theme)
                        .clicked()
                    {
                        skip_version = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if cm3_prop_button(ui, "Close", ButtonVariant::Normal, false, &self.theme)
                            .clicked()
                        {
                            close_modal = true;
                        }
                    });
                });
            });

        if open_download && let Err(err) = open_url_in_os_browser(&info.download_url) {
            self.update_check.last_error = Some(format!("Failed to open browser: {err}"));
        }
        if open_release_notes && let Err(err) = open_url_in_os_browser(&info.release_url) {
            self.update_check.last_error = Some(format!("Failed to open browser: {err}"));
        }
        if skip_version {
            self.workspace.panels.app_settings.skipped_update_version = Some(info.version.clone());
            self.update_check.banner_dismissed_version = None;
            self.update_check.dialog_open = false;
            self.show_toast(format!("Skipped update v{}", info.version));
        } else if close_modal {
            self.update_check.dialog_open = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_release_assets() -> Vec<(String, String)> {
        vec![
            (
                "ZoidCraft-macOS-arm64.dmg".to_owned(),
                "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/ZoidCraft-macOS-arm64.dmg".to_owned(),
            ),
            (
                "zoidcraft-macOS-arm64.tar.gz".to_owned(),
                "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/zoidcraft-macOS-arm64.tar.gz".to_owned(),
            ),
            (
                "ZoidCraft-macOS-x86_64.dmg".to_owned(),
                "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/ZoidCraft-macOS-x86_64.dmg".to_owned(),
            ),
            (
                "zoidcraft-macOS-x86_64.tar.gz".to_owned(),
                "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/zoidcraft-macOS-x86_64.tar.gz".to_owned(),
            ),
            (
                "ZoidCraft-Windows-x86_64-Setup.exe".to_owned(),
                "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/ZoidCraft-Windows-x86_64-Setup.exe".to_owned(),
            ),
            (
                "ZoidCraft-Windows-x86_64-portable.zip".to_owned(),
                "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/ZoidCraft-Windows-x86_64-portable.zip".to_owned(),
            ),
            (
                "ZoidCraft-Linux-x86_64.deb".to_owned(),
                "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/ZoidCraft-Linux-x86_64.deb".to_owned(),
            ),
            (
                "zoidcraft-Linux-x86_64.tar.gz".to_owned(),
                "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/zoidcraft-Linux-x86_64.tar.gz".to_owned(),
            ),
            (
                "SHA256SUMS.txt".to_owned(),
                "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/SHA256SUMS.txt".to_owned(),
            ),
        ]
    }

    #[test]
    fn parse_semver_handles_valid_and_invalid_inputs() {
        assert_eq!(
            parse_semver("0.2.6"),
            Some(SemVer {
                major: 0,
                minor: 2,
                patch: 6
            })
        );
        assert_eq!(
            parse_semver("v0.2.7"),
            Some(SemVer {
                major: 0,
                minor: 2,
                patch: 7
            })
        );
        assert_eq!(
            parse_semver("  V12.34.56+20261128  "),
            Some(SemVer {
                major: 12,
                minor: 34,
                patch: 56
            })
        );

        // Prereleases and malformed versions must be rejected.
        for bad in [
            "",
            "   ",
            "0.2",
            "0.2.7.1",
            "vv0.2.7",
            "v0.2.7-alpha",
            "0.2.7-beta.1",
            "1.0.0-rc.1+build.9",
            "0.+2.7",
            "0.-2.7",
            "abc.def.ghi",
        ] {
            assert_eq!(parse_semver(bad), None, "expected None for {bad:?}");
        }
    }

    #[test]
    fn is_newer_version_compares_numeric_components() {
        assert!(is_newer_version("0.2.6", "0.2.7"));
        assert!(is_newer_version("0.2.6", "v0.2.10"));
        assert!(is_newer_version("0.2.6", "v0.3.0"));
        assert!(is_newer_version("0.2.6", "1.0.0"));

        assert!(!is_newer_version("0.2.6", "0.2.6"));
        assert!(!is_newer_version("0.2.6", "v0.2.6"));
        assert!(!is_newer_version("0.2.6", "0.2.5"));
        assert!(!is_newer_version("0.2.6", "0.1.9"));
        assert!(!is_newer_version("0.2.6", "v0.2.7-rc.1"));
        assert!(!is_newer_version("invalid", "0.2.7"));
    }

    #[test]
    fn is_safe_https_url_rejects_non_https_and_shell_metacharacters() {
        assert!(is_safe_https_url(GITHUB_RELEASES_PAGE));
        assert!(is_safe_https_url(
            "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/ZoidCraft-macOS-arm64.dmg"
        ));

        for bad in [
            "",
            "https://",
            "https:///path",
            "http://github.com/rbstxt/zoidcraft/releases",
            "javascript:alert(1)",
            "https://example.com/a b",
            "https://example.com/a\"b",
            "https://example.com/a'b",
            "https://example.com/a`id`",
            "https://example.com/a&calc",
            "https://example.com/a|sh",
            "https://example.com/a;ls",
            "https://example.com/a<b",
            "https://example.com/a>b",
            "https://example.com/$HOME",
            "https://example.com/a\nb",
            "https://example.com/a\rb",
            "https://example.com/a\x07b",
        ] {
            assert!(!is_safe_https_url(bad), "expected unsafe URL: {bad:?}");
        }
    }

    #[test]
    fn select_platform_download_asset_matches_primary_installers() {
        let assets = sample_release_assets();

        assert_eq!(
            select_platform_download_asset(&assets, "macos", "aarch64").map(|(n, _)| n),
            Some("ZoidCraft-macOS-arm64.dmg".to_owned())
        );
        assert_eq!(
            select_platform_download_asset(&assets, "macos", "x86_64").map(|(n, _)| n),
            Some("ZoidCraft-macOS-x86_64.dmg".to_owned())
        );
        assert_eq!(
            select_platform_download_asset(&assets, "windows", "x86_64").map(|(n, _)| n),
            Some("ZoidCraft-Windows-x86_64-Setup.exe".to_owned())
        );
        assert_eq!(
            select_platform_download_asset(&assets, "linux", "x86_64").map(|(n, _)| n),
            Some("ZoidCraft-Linux-x86_64.deb".to_owned())
        );
    }

    #[test]
    fn select_platform_download_asset_falls_back_to_archives_and_rejects_unsafe_urls() {
        let archives_only: Vec<(String, String)> = sample_release_assets()
            .into_iter()
            .filter(|(n, _)| !n.ends_with(".dmg") && !n.ends_with(".exe") && !n.ends_with(".deb"))
            .collect();

        assert_eq!(
            select_platform_download_asset(&archives_only, "macos", "aarch64").map(|(n, _)| n),
            Some("zoidcraft-macOS-arm64.tar.gz".to_owned())
        );
        assert_eq!(
            select_platform_download_asset(&archives_only, "macos", "x86_64").map(|(n, _)| n),
            Some("zoidcraft-macOS-x86_64.tar.gz".to_owned())
        );
        assert_eq!(
            select_platform_download_asset(&archives_only, "windows", "x86_64").map(|(n, _)| n),
            Some("ZoidCraft-Windows-x86_64-portable.zip".to_owned())
        );
        assert_eq!(
            select_platform_download_asset(&archives_only, "linux", "x86_64").map(|(n, _)| n),
            Some("zoidcraft-Linux-x86_64.tar.gz".to_owned())
        );

        // Unsafe DMG URL must be skipped in favor of safe tar.gz.
        let unsafe_dmg = vec![
            (
                "ZoidCraft-macOS-arm64.dmg".to_owned(),
                "http://insecure.example/ZoidCraft-macOS-arm64.dmg".to_owned(),
            ),
            (
                "ZoidCraft-macOS-arm64-evil.dmg".to_owned(),
                "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/a;rm".to_owned(),
            ),
            (
                "zoidcraft-macOS-arm64.tar.gz".to_owned(),
                "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/zoidcraft-macOS-arm64.tar.gz".to_owned(),
            ),
        ];
        assert_eq!(
            select_platform_download_asset(&unsafe_dmg, "macos", "aarch64").map(|(n, _)| n),
            Some("zoidcraft-macOS-arm64.tar.gz".to_owned())
        );

        // Linux must not pick a macOS tarball when no Linux asset exists.
        let macos_only = vec![(
            "zoidcraft-macOS-arm64.tar.gz".to_owned(),
            "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/zoidcraft-macOS-arm64.tar.gz".to_owned(),
        )];
        assert_eq!(
            select_platform_download_asset(&macos_only, "linux", "x86_64"),
            None
        );
    }

    #[test]
    fn parse_github_release_json_handles_updates_filters_and_fallbacks() {
        let payload = json!({
            "tag_name": "v0.2.7",
            "name": "ZoidCraft 0.2.7",
            "draft": false,
            "prerelease": false,
            "html_url": "https://github.com/rbstxt/zoidcraft/releases/tag/v0.2.7",
            "published_at": "2026-11-28T10:15:30Z",
            "body": "### Highlights\r\n- Faster timeline playback\r\n- Update notifications",
            "assets": [
                {
                    "name": "ZoidCraft-macOS-arm64.dmg",
                    "browser_download_url": "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/ZoidCraft-macOS-arm64.dmg"
                },
                {
                    "name": "ZoidCraft-Windows-x86_64-Setup.exe",
                    "browser_download_url": "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/ZoidCraft-Windows-x86_64-Setup.exe"
                }
            ]
        })
        .to_string();

        let info = parse_github_release_json(&payload, "0.2.6", "macos", "aarch64")
            .expect("valid release json")
            .expect("newer version detected");
        assert_eq!(info.version, "0.2.7");
        assert_eq!(info.tag_name, "v0.2.7");
        assert_eq!(info.release_name, "ZoidCraft 0.2.7");
        assert_eq!(
            info.release_url,
            "https://github.com/rbstxt/zoidcraft/releases/tag/v0.2.7"
        );
        assert_eq!(
            info.download_url,
            "https://github.com/rbstxt/zoidcraft/releases/download/v0.2.7/ZoidCraft-macOS-arm64.dmg"
        );
        assert_eq!(info.download_label, "Download for macOS");
        assert_eq!(
            info.asset_name.as_deref(),
            Some("ZoidCraft-macOS-arm64.dmg")
        );
        assert_eq!(info.published_at.as_deref(), Some("2026-11-28"));
        assert!(info.notes_summary.contains("Faster timeline playback"));

        // Same or older version returns Ok(None).
        assert_eq!(
            parse_github_release_json(&payload, "0.2.7", "macos", "aarch64").expect("ok"),
            None
        );
        assert_eq!(
            parse_github_release_json(&payload, "0.3.0", "macos", "aarch64").expect("ok"),
            None
        );

        // Missing platform asset falls back cleanly to html_url.
        let linux_fallback = parse_github_release_json(&payload, "0.2.6", "linux", "x86_64")
            .expect("valid json")
            .expect("newer release");
        assert_eq!(linux_fallback.download_url, linux_fallback.release_url);
        assert_eq!(linux_fallback.download_label, "Open Release Page");
        assert_eq!(linux_fallback.asset_name, None);

        // Draft and prerelease payloads are ignored.
        let draft_json = json!({
            "tag_name": "v0.3.0",
            "draft": true,
            "prerelease": false
        })
        .to_string();
        assert_eq!(
            parse_github_release_json(&draft_json, "0.2.6", "macos", "aarch64").expect("ok"),
            None
        );

        let pre_json = json!({
            "tag_name": "v0.3.0",
            "draft": false,
            "prerelease": true
        })
        .to_string();
        assert_eq!(
            parse_github_release_json(&pre_json, "0.2.6", "macos", "aarch64").expect("ok"),
            None
        );

        // Long release notes are cleanly bounded without panicking on UTF-8.
        let long_body = (1..=30)
            .map(|i| format!("Line {i:02}: ✨ {}", "abcdef1234 ".repeat(6)))
            .collect::<Vec<_>>()
            .join("\n");
        let long_json = json!({
            "tag_name": "v0.2.8",
            "draft": false,
            "prerelease": false,
            "html_url": "http://unsafe.example/ignored",
            "body": long_body,
            "assets": []
        })
        .to_string();
        let long_info = parse_github_release_json(&long_json, "0.2.6", "macos", "aarch64")
            .expect("valid json")
            .expect("newer release");
        assert_eq!(long_info.release_url, GITHUB_RELEASES_PAGE);
        assert_eq!(long_info.download_url, GITHUB_RELEASES_PAGE);
        assert!(long_info.notes_summary.chars().count() <= MAX_NOTES_CHARS);
        assert!(long_info.notes_summary.ends_with('…'));

        // Malformed JSON returns an error.
        assert!(parse_github_release_json("{not valid json", "0.2.6", "macos", "aarch64").is_err());
    }
}
