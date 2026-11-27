//! On-demand, non-bundled ("No FFmpeg Inside") FFmpeg installation planner and runner.
//!
//! Detects the best native OS package manager (`Homebrew` on macOS, `WinGet`/`Scoop` on Windows,
//! `Linuxbrew` or `pkexec` + `apt`/`dnf`/`pacman`/`zypper`/`apk` on Linux), presents the exact
//! command transparently before running, streams bounded installer output, supports cooperative
//! cancellation with process-tree cleanup, and refreshes `zc_render::video` discovery caches
//! without requiring an application restart.

#![forbid(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{BufRead, BufReader};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::process::{Command, Stdio};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use crate::ExportError;
#[cfg(not(target_arch = "wasm32"))]
use crate::ffmpeg::{configure_process_group, kill_process_tree_and_reap, probe_ffmpeg};

/// Official FFmpeg download portal shown as a fallback when no automatic package manager is found.
pub const FFMPEG_DOWNLOAD_URL: &str = "https://ffmpeg.org/download.html";

/// Maximum bytes kept per streamed installer output line to prevent unbounded memory growth.
#[cfg(not(target_arch = "wasm32"))]
const MAX_INSTALL_LINE_CHARS: usize = 400;

/// Maximum trailing log lines retained for error diagnostics.
#[cfg(not(target_arch = "wasm32"))]
const MAX_ERROR_TAIL_LINES: usize = 16;

/// Concrete, inspectable installation plan for the current operating system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfmpegInstallPlan {
    /// Human-readable installer name (e.g. `"Homebrew"`, `"WinGet"`, `"Manual"`).
    pub manager_name: String,
    /// `true` when ZoidCraft can execute `program` + `args` directly from the GUI/CLI.
    pub can_run_automatically: bool,
    /// Resolved executable path to spawn when `can_run_automatically` is `true`.
    pub program: Option<PathBuf>,
    /// Exact CLI arguments passed to `program`.
    pub args: Vec<String>,
    /// Exact shell command preview displayed to the user (and copyable with one click).
    pub display_command: String,
    /// Supplemental guidance shown when automatic installation is unavailable or fails.
    pub manual_hint: String,
    /// Official FFmpeg download URL.
    pub download_url: &'static str,
}

/// Verified FFmpeg binary paths & version returned after installation or manual binary selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfmpegInstallOutcome {
    pub version: String,
    pub ffmpeg_path: PathBuf,
    pub ffprobe_path: Option<PathBuf>,
}

/// Detects the most reliable FFmpeg installation plan for the current machine.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn detect_ffmpeg_install_plan() -> FfmpegInstallPlan {
    detect_ffmpeg_install_plan_for_os(std::env::consts::OS, resolve_system_tool)
}

#[cfg(not(target_arch = "wasm32"))]
fn is_executable_file(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_system_tool(name: &str) -> Option<PathBuf> {
    #[cfg(windows)]
    let candidate_names: Vec<String> = {
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".exe")
            || lower.ends_with(".cmd")
            || lower.ends_with(".bat")
            || lower.ends_with(".ps1")
        {
            vec![name.to_owned()]
        } else {
            vec![
                format!("{name}.exe"),
                format!("{name}.cmd"),
                format!("{name}.ps1"),
            ]
        }
    };
    #[cfg(not(windows))]
    let candidate_names: [String; 1] = [name.to_owned()];

    if let Some(path_env) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_env) {
            for candidate_name in &candidate_names {
                let candidate = dir.join(candidate_name);
                if is_executable_file(&candidate) {
                    return Some(candidate);
                }
            }
        }
    }

    let mut fallback_dirs: Vec<PathBuf> = Vec::new();
    #[cfg(target_os = "macos")]
    {
        fallback_dirs.push(PathBuf::from("/opt/homebrew/bin"));
        fallback_dirs.push(PathBuf::from("/usr/local/bin"));
        fallback_dirs.push(PathBuf::from("/opt/local/bin"));
        fallback_dirs.push(PathBuf::from("/usr/bin"));
    }
    #[cfg(target_os = "linux")]
    {
        fallback_dirs.push(PathBuf::from("/usr/bin"));
        fallback_dirs.push(PathBuf::from("/usr/local/bin"));
        fallback_dirs.push(PathBuf::from("/home/linuxbrew/.linuxbrew/bin"));
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
            fallback_dirs.push(PathBuf::from(local_app_data).join("Microsoft\\WindowsApps"));
        }
        if let Some(user_profile) = std::env::var_os("USERPROFILE") {
            fallback_dirs.push(PathBuf::from(user_profile).join("scoop\\shims"));
        }
        fallback_dirs.push(PathBuf::from(r"C:\ProgramData\chocolatey\bin"));
    }

    for dir in fallback_dirs {
        for candidate_name in &candidate_names {
            let candidate = dir.join(candidate_name);
            if is_executable_file(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

/// Pure, testable OS plan builder parameterized by OS name and tool resolver.
pub fn detect_ffmpeg_install_plan_for_os<F>(os: &str, mut find_tool: F) -> FfmpegInstallPlan
where
    F: FnMut(&str) -> Option<PathBuf>,
{
    match os {
        "macos" => {
            if let Some(brew) = find_tool("brew") {
                return FfmpegInstallPlan {
                    manager_name: "Homebrew".to_owned(),
                    can_run_automatically: true,
                    program: Some(brew),
                    args: vec!["install".to_owned(), "ffmpeg".to_owned()],
                    display_command: "brew install ffmpeg".to_owned(),
                    manual_hint: "Installs official Homebrew FFmpeg & ffprobe bottles.".to_owned(),
                    download_url: FFMPEG_DOWNLOAD_URL,
                };
            }
            if find_tool("port").is_some() {
                return FfmpegInstallPlan {
                    manager_name: "MacPorts".to_owned(),
                    can_run_automatically: false,
                    program: None,
                    args: Vec::new(),
                    display_command: "sudo port install ffmpeg".to_owned(),
                    manual_hint:
                        "Run this command in Terminal, or pick an existing ffmpeg binary with Locate Binary."
                            .to_owned(),
                    download_url: FFMPEG_DOWNLOAD_URL,
                };
            }
            FfmpegInstallPlan {
                manager_name: "Manual (macOS)".to_owned(),
                can_run_automatically: false,
                program: None,
                args: Vec::new(),
                display_command: "brew install ffmpeg".to_owned(),
                manual_hint: "Homebrew was not found. Install Homebrew (https://brew.sh) then run `brew install ffmpeg`, or download a macOS FFmpeg build and choose Locate Binary.".to_owned(),
                download_url: FFMPEG_DOWNLOAD_URL,
            }
        }
        "windows" => {
            if let Some(winget) = find_tool("winget") {
                return FfmpegInstallPlan {
                    manager_name: "WinGet".to_owned(),
                    can_run_automatically: true,
                    program: Some(winget),
                    args: vec![
                        "install".to_owned(),
                        "--id".to_owned(),
                        "Gyan.FFmpeg".to_owned(),
                        "-e".to_owned(),
                        "--accept-package-agreements".to_owned(),
                        "--accept-source-agreements".to_owned(),
                    ],
                    display_command: "winget install --id Gyan.FFmpeg -e".to_owned(),
                    manual_hint:
                        "Installs Gyan.FFmpeg (ffmpeg + ffprobe) via Windows Package Manager."
                            .to_owned(),
                    download_url: FFMPEG_DOWNLOAD_URL,
                };
            }
            if let Some(scoop) = find_tool("scoop")
                .or_else(|| find_tool("scoop.cmd"))
                .or_else(|| find_tool("scoop.ps1"))
                .or_else(|| find_tool("scoop.exe"))
            {
                let ext = scoop
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_ascii_lowercase);
                let (program, args) = match ext.as_deref() {
                    Some("cmd" | "bat") => {
                        let cmd_exe = find_tool("cmd").unwrap_or_else(|| PathBuf::from("cmd.exe"));
                        (
                            cmd_exe,
                            vec![
                                "/C".to_owned(),
                                scoop.to_string_lossy().into_owned(),
                                "install".to_owned(),
                                "ffmpeg".to_owned(),
                            ],
                        )
                    }
                    Some("ps1") => {
                        let ps = find_tool("pwsh")
                            .or_else(|| find_tool("powershell"))
                            .unwrap_or_else(|| PathBuf::from("powershell.exe"));
                        (
                            ps,
                            vec![
                                "-NoProfile".to_owned(),
                                "-ExecutionPolicy".to_owned(),
                                "Bypass".to_owned(),
                                "-File".to_owned(),
                                scoop.to_string_lossy().into_owned(),
                                "install".to_owned(),
                                "ffmpeg".to_owned(),
                            ],
                        )
                    }
                    _ => (scoop, vec!["install".to_owned(), "ffmpeg".to_owned()]),
                };
                return FfmpegInstallPlan {
                    manager_name: "Scoop".to_owned(),
                    can_run_automatically: true,
                    program: Some(program),
                    args,
                    display_command: "scoop install ffmpeg".to_owned(),
                    manual_hint: "Installs FFmpeg & ffprobe in your user Scoop directory."
                        .to_owned(),
                    download_url: FFMPEG_DOWNLOAD_URL,
                };
            }
            if find_tool("choco").is_some() {
                return FfmpegInstallPlan {
                    manager_name: "Chocolatey".to_owned(),
                    can_run_automatically: false,
                    program: None,
                    args: Vec::new(),
                    display_command: "choco install ffmpeg -y".to_owned(),
                    manual_hint:
                        "Run this command in an Administrator terminal, or use Locate Binary."
                            .to_owned(),
                    download_url: FFMPEG_DOWNLOAD_URL,
                };
            }
            FfmpegInstallPlan {
                manager_name: "Manual (Windows)".to_owned(),
                can_run_automatically: false,
                program: None,
                args: Vec::new(),
                display_command: "winget install --id Gyan.FFmpeg -e".to_owned(),
                manual_hint: "Download a Windows FFmpeg build from ffmpeg.org, extract ffmpeg.exe, and choose Locate Binary.".to_owned(),
                download_url: FFMPEG_DOWNLOAD_URL,
            }
        }
        _ => {
            if let Some(brew) = find_tool("brew") {
                return FfmpegInstallPlan {
                    manager_name: "Homebrew".to_owned(),
                    can_run_automatically: true,
                    program: Some(brew),
                    args: vec!["install".to_owned(), "ffmpeg".to_owned()],
                    display_command: "brew install ffmpeg".to_owned(),
                    manual_hint: "Installs FFmpeg without requiring root privileges.".to_owned(),
                    download_url: FFMPEG_DOWNLOAD_URL,
                };
            }
            let pkexec = find_tool("pkexec");
            let linux_managers: [(&str, &str, &[&str], &str); 5] = [
                (
                    "apt-get",
                    "APT",
                    &["install", "-y", "ffmpeg"],
                    "sudo apt-get update && sudo apt-get install -y ffmpeg",
                ),
                (
                    "dnf",
                    "DNF",
                    &["install", "-y", "ffmpeg"],
                    "sudo dnf install -y ffmpeg",
                ),
                (
                    "pacman",
                    "Pacman",
                    &["-S", "--noconfirm", "ffmpeg"],
                    "sudo pacman -S --noconfirm ffmpeg",
                ),
                (
                    "zypper",
                    "Zypper",
                    &["install", "-y", "ffmpeg"],
                    "sudo zypper install -y ffmpeg",
                ),
                ("apk", "APK", &["add", "ffmpeg"], "sudo apk add ffmpeg"),
            ];
            for (tool, label, pm_args, sudo_cmd) in linux_managers {
                if let Some(pm_path) = find_tool(tool) {
                    if let Some(ref pk) = pkexec {
                        if tool == "apt-get" {
                            let sh_bin = find_tool("sh").map_or_else(
                                || "sh".to_owned(),
                                |p| p.to_string_lossy().into_owned(),
                            );
                            let shell_cmd =
                                "apt-get update && apt-get install -y ffmpeg".to_owned();
                            return FfmpegInstallPlan {
                                manager_name: format!("{label} (via pkexec)"),
                                can_run_automatically: true,
                                program: Some(pk.clone()),
                                args: vec![sh_bin, "-c".to_owned(), shell_cmd.clone()],
                                display_command: format!("pkexec sh -c \"{shell_cmd}\""),
                                manual_hint: format!(
                                    "Prompts for system authorization, or run `{sudo_cmd}` in a terminal."
                                ),
                                download_url: FFMPEG_DOWNLOAD_URL,
                            };
                        }
                        let mut args = vec![pm_path.to_string_lossy().into_owned()];
                        args.extend(pm_args.iter().map(|s| (*s).to_owned()));
                        return FfmpegInstallPlan {
                            manager_name: format!("{label} (via pkexec)"),
                            can_run_automatically: true,
                            program: Some(pk.clone()),
                            args,
                            display_command: format!("pkexec {tool} {}", pm_args.join(" ")),
                            manual_hint: format!(
                                "Prompts for system authorization, or run `{sudo_cmd}` in a terminal."
                            ),
                            download_url: FFMPEG_DOWNLOAD_URL,
                        };
                    }
                    return FfmpegInstallPlan {
                        manager_name: label.to_owned(),
                        can_run_automatically: false,
                        program: None,
                        args: Vec::new(),
                        display_command: sudo_cmd.to_owned(),
                        manual_hint:
                            "Run this command in a terminal, then click Re-check or Locate Binary."
                                .to_owned(),
                        download_url: FFMPEG_DOWNLOAD_URL,
                    };
                }
            }
            FfmpegInstallPlan {
                manager_name: "Manual".to_owned(),
                can_run_automatically: false,
                program: None,
                args: Vec::new(),
                display_command: "sudo apt install ffmpeg".to_owned(),
                manual_hint:
                    "Install FFmpeg with your system package manager or choose Locate Binary."
                        .to_owned(),
                download_url: FFMPEG_DOWNLOAD_URL,
            }
        }
    }
}

/// Validates a user-selected `ffmpeg` binary path by running `ffmpeg -version` with a bounded timeout.
#[cfg(not(target_arch = "wasm32"))]
pub fn verify_custom_ffmpeg_binary(path: &Path) -> Result<FfmpegInstallOutcome, ExportError> {
    if !path.is_file() {
        return Err(ExportError::FfmpegNotFound(format!(
            "selected path is not a file: {}",
            path.display()
        )));
    }
    let version = probe_ffmpeg(Some(path))?;
    let ffprobe_path = zc_render::video::derive_sibling_ffprobe_path(path);
    Ok(FfmpegInstallOutcome {
        version,
        ffmpeg_path: path.to_path_buf(),
        ffprobe_path,
    })
}

/// Executes an automatic [`FfmpegInstallPlan`], streaming output lines to `on_line`, honoring
/// cooperative `cancel` (terminating the installer process tree cleanly), refreshing
/// `zc_render::video` discovery caches, and verifying `ffmpeg -version` on completion.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_ffmpeg_install<F>(
    plan: &FfmpegInstallPlan,
    cancel: &AtomicBool,
    mut on_line: F,
) -> Result<FfmpegInstallOutcome, ExportError>
where
    F: FnMut(String),
{
    if cancel.load(Ordering::Acquire) {
        return Err(ExportError::Cancelled);
    }
    let Some(program) = plan.program.as_ref().filter(|_| plan.can_run_automatically) else {
        return Err(ExportError::InvalidSettings(format!(
            "automatic installation is unavailable for '{}'; run `{}` manually",
            plan.manager_name, plan.display_command
        )));
    };

    on_line(format!("$ {}", plan.display_command));

    let mut cmd = Command::new(program);
    cmd.args(&plan.args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Prevent Homebrew auto-update hangs and disable interactive prompts where supported.
    cmd.env("HOMEBREW_NO_AUTO_UPDATE", "1");
    cmd.env("HOMEBREW_NO_ENV_HINTS", "1");
    cmd.env("DEBIAN_FRONTEND", "noninteractive");
    configure_process_group(&mut cmd);

    let mut child = cmd.spawn().map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            ExportError::FfmpegNotFound(format!(
                "installer '{}' not found: {}",
                plan.manager_name,
                program.display()
            ))
        } else {
            ExportError::Io(err)
        }
    })?;

    let (line_tx, line_rx) = mpsc::channel::<String>();
    let spawn_reader = |reader: Box<dyn std::io::Read + Send>, tx: mpsc::Sender<String>| {
        std::thread::spawn(move || {
            let mut buf = BufReader::new(reader);
            let mut raw = String::new();
            while buf.read_line(&mut raw).unwrap_or(0) > 0 {
                let trimmed = raw.trim_end_matches(['\r', '\n']);
                if !trimmed.is_empty() {
                    let line: String = trimmed.chars().take(MAX_INSTALL_LINE_CHARS).collect();
                    let _ = tx.send(line);
                }
                raw.clear();
            }
        })
    };

    let out_handle = child
        .stdout
        .take()
        .map(|s| spawn_reader(Box::new(s), line_tx.clone()));
    let err_handle = child
        .stderr
        .take()
        .map(|s| spawn_reader(Box::new(s), line_tx.clone()));
    drop(line_tx);

    let mut tail_lines: VecDeque<String> = VecDeque::with_capacity(MAX_ERROR_TAIL_LINES);
    let mut drain_lines = |rx: &mpsc::Receiver<String>, cb: &mut F| {
        while let Ok(line) = rx.try_recv() {
            if tail_lines.len() >= MAX_ERROR_TAIL_LINES {
                tail_lines.pop_front();
            }
            tail_lines.push_back(line.clone());
            cb(line);
        }
    };

    let status = loop {
        drain_lines(&line_rx, &mut on_line);
        if cancel.load(Ordering::Acquire) {
            kill_process_tree_and_reap(&mut child);
            if let Some(h) = out_handle {
                let _ = h.join();
            }
            if let Some(h) = err_handle {
                let _ = h.join();
            }
            return Err(ExportError::Cancelled);
        }
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(err) => {
                kill_process_tree_and_reap(&mut child);
                if let Some(h) = out_handle {
                    let _ = h.join();
                }
                if let Some(h) = err_handle {
                    let _ = h.join();
                }
                return Err(ExportError::Io(err));
            }
        }
    };

    if let Some(h) = out_handle {
        let _ = h.join();
    }
    if let Some(h) = err_handle {
        let _ = h.join();
    }
    drain_lines(&line_rx, &mut on_line);

    if !status.success() {
        let summary = tail_lines.into_iter().collect::<Vec<_>>().join("\n");
        return Err(ExportError::FfmpegFailed {
            exit_code: status.code(),
            stderr: if summary.is_empty() {
                format!("{} exited with status {status}", plan.manager_name)
            } else {
                summary
            },
        });
    }

    // Clear discovery caches so newly installed binaries are picked up immediately.
    zc_render::video::refresh_media_tool_cache();
    let version = probe_ffmpeg(None)?;
    let ffmpeg_path = zc_render::video::resolve_ffmpeg_binary_path();
    let ffprobe_candidate = zc_render::video::resolve_ffprobe_binary_path();
    let ffprobe_path = ffprobe_candidate.is_file().then_some(ffprobe_candidate);
    on_line(format!("Verified: {version} ({})", ffmpeg_path.display()));

    Ok(FfmpegInstallOutcome {
        version,
        ffmpeg_path,
        ffprobe_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_plan_selects_expected_manager_per_os() {
        let mac_brew = detect_ffmpeg_install_plan_for_os("macos", |tool| {
            (tool == "brew").then(|| PathBuf::from("/opt/homebrew/bin/brew"))
        });
        assert_eq!(mac_brew.manager_name, "Homebrew");
        assert!(mac_brew.can_run_automatically);
        assert_eq!(mac_brew.display_command, "brew install ffmpeg");

        let mac_none = detect_ffmpeg_install_plan_for_os("macos", |_| None);
        assert!(!mac_none.can_run_automatically);
        assert_eq!(mac_none.download_url, FFMPEG_DOWNLOAD_URL);

        let win_winget = detect_ffmpeg_install_plan_for_os("windows", |tool| {
            (tool == "winget").then(|| PathBuf::from(r"C:\Windows\System32\winget.exe"))
        });
        assert_eq!(win_winget.manager_name, "WinGet");
        assert!(win_winget.can_run_automatically);
        assert!(win_winget.args.contains(&"Gyan.FFmpeg".to_owned()));

        let win_scoop = detect_ffmpeg_install_plan_for_os("windows", |tool| {
            (tool == "scoop").then(|| PathBuf::from(r"C:\Users\me\scoop\shims\scoop.exe"))
        });
        assert_eq!(win_scoop.manager_name, "Scoop");
        assert!(win_scoop.can_run_automatically);

        let linux_pkexec_apt = detect_ffmpeg_install_plan_for_os("linux", |tool| match tool {
            "pkexec" => Some(PathBuf::from("/usr/bin/pkexec")),
            "apt-get" => Some(PathBuf::from("/usr/bin/apt-get")),
            _ => None,
        });
        assert!(linux_pkexec_apt.can_run_automatically);
        assert_eq!(linux_pkexec_apt.manager_name, "APT (via pkexec)");
        assert_eq!(
            linux_pkexec_apt.program,
            Some(PathBuf::from("/usr/bin/pkexec"))
        );
        assert_eq!(
            linux_pkexec_apt.args,
            vec![
                "sh".to_owned(),
                "-c".to_owned(),
                "apt-get update && apt-get install -y ffmpeg".to_owned(),
            ]
        );
        assert_eq!(
            linux_pkexec_apt.display_command,
            "pkexec sh -c \"apt-get update && apt-get install -y ffmpeg\""
        );

        let linux_no_pkexec = detect_ffmpeg_install_plan_for_os("linux", |tool| {
            (tool == "pacman").then(|| PathBuf::from("/usr/bin/pacman"))
        });
        assert!(!linux_no_pkexec.can_run_automatically);
        assert_eq!(
            linux_no_pkexec.display_command,
            "sudo pacman -S --noconfirm ffmpeg"
        );
    }

    #[test]
    fn windows_scoop_detects_cmd_ps1_and_exe_shims() {
        let scoop_cmd = detect_ffmpeg_install_plan_for_os("windows", |tool| match tool {
            "scoop.cmd" => Some(PathBuf::from(r"C:\Users\me\scoop\shims\scoop.cmd")),
            "cmd" => Some(PathBuf::from(r"C:\Windows\System32\cmd.exe")),
            _ => None,
        });
        assert_eq!(scoop_cmd.manager_name, "Scoop");
        assert!(scoop_cmd.can_run_automatically);
        assert_eq!(
            scoop_cmd.program,
            Some(PathBuf::from(r"C:\Windows\System32\cmd.exe"))
        );
        assert_eq!(
            scoop_cmd.args,
            vec![
                "/C".to_owned(),
                r"C:\Users\me\scoop\shims\scoop.cmd".to_owned(),
                "install".to_owned(),
                "ffmpeg".to_owned(),
            ]
        );
        assert_eq!(scoop_cmd.display_command, "scoop install ffmpeg");

        let scoop_ps1 = detect_ffmpeg_install_plan_for_os("windows", |tool| match tool {
            "scoop.ps1" => Some(PathBuf::from(r"C:\Users\me\scoop\shims\scoop.ps1")),
            _ => None,
        });
        assert_eq!(scoop_ps1.manager_name, "Scoop");
        assert!(scoop_ps1.can_run_automatically);
        assert_eq!(scoop_ps1.program, Some(PathBuf::from("powershell.exe")));
        assert_eq!(
            scoop_ps1.args,
            vec![
                "-NoProfile".to_owned(),
                "-ExecutionPolicy".to_owned(),
                "Bypass".to_owned(),
                "-File".to_owned(),
                r"C:\Users\me\scoop\shims\scoop.ps1".to_owned(),
                "install".to_owned(),
                "ffmpeg".to_owned(),
            ]
        );

        let scoop_exe = detect_ffmpeg_install_plan_for_os("windows", |tool| {
            (tool == "scoop.exe").then(|| PathBuf::from(r"C:\Users\me\scoop\shims\scoop.exe"))
        });
        assert_eq!(scoop_exe.manager_name, "Scoop");
        assert!(scoop_exe.can_run_automatically);
        assert_eq!(
            scoop_exe.program,
            Some(PathBuf::from(r"C:\Users\me\scoop\shims\scoop.exe"))
        );
        assert_eq!(
            scoop_exe.args,
            vec!["install".to_owned(), "ffmpeg".to_owned()]
        );
    }

    #[test]
    fn linux_apt_pkexec_uses_resolved_sh_and_updates_before_install() {
        let plan = detect_ffmpeg_install_plan_for_os("linux", |tool| match tool {
            "pkexec" => Some(PathBuf::from("/usr/bin/pkexec")),
            "apt-get" => Some(PathBuf::from("/usr/bin/apt-get")),
            "sh" => Some(PathBuf::from("/bin/sh")),
            _ => None,
        });
        assert!(plan.can_run_automatically);
        assert_eq!(plan.program, Some(PathBuf::from("/usr/bin/pkexec")));
        assert_eq!(
            plan.args,
            vec![
                "/bin/sh".to_owned(),
                "-c".to_owned(),
                "apt-get update && apt-get install -y ffmpeg".to_owned(),
            ]
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn run_ffmpeg_install_honors_pre_cancelled_token_and_rejects_manual_plan() {
        let manual = detect_ffmpeg_install_plan_for_os("macos", |_| None);
        let cancel = AtomicBool::new(false);
        let err = run_ffmpeg_install(&manual, &cancel, |_| {}).unwrap_err();
        assert!(matches!(err, ExportError::InvalidSettings(_)));

        let auto = detect_ffmpeg_install_plan_for_os("macos", |tool| {
            (tool == "brew").then(|| PathBuf::from("/opt/homebrew/bin/brew"))
        });
        cancel.store(true, Ordering::Release);
        let err = run_ffmpeg_install(&auto, &cancel, |_| {}).unwrap_err();
        assert!(matches!(err, ExportError::Cancelled));

        let bad_custom = verify_custom_ffmpeg_binary(Path::new("/nonexistent/zoidcraft_ffmpeg"));
        assert!(matches!(bad_custom, Err(ExportError::FfmpegNotFound(_))));
    }
}
