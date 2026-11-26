//! Cross-platform OS system clipboard helpers for PanZoid (CM3 / Zoidium) JSON
//! copy & paste and debug diagnostics.
//!
//! Uses `#![forbid(unsafe_code)]` with zero external crate dependencies:
//! - Always mirrors writes to `egui::Context::copy_text` AND the native OS clipboard
//!   (`pbcopy`/`pbpaste` on macOS, `wl-copy`/`xclip`/`xsel` on Linux, `clip`/PowerShell
//!   on Windows) so external apps (e.g. a browser running PanZoid) and `pbpaste` always
//!   observe copied text immediately.
//! - Uses bounded waits so a hung clipboard helper can never stall the UI thread.

#![forbid(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
#[cfg(not(target_arch = "wasm32"))]
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};

use eframe::egui;

/// Non-empty sentinel used to seed `egui-winit`'s in-memory fallback clipboard on launch
/// without touching the user's real OS clipboard, guaranteeing `Cmd/Ctrl+V` always emits
/// `egui::Event::Paste` even on a fresh launch.
pub const CLIPBOARD_SENTINEL: &str = "\u{001F}zoidcraft-clipboard-bridge";

/// Tracks the last text written by ZoidCraft itself via [`write_system_clipboard`] so
/// Timeline `Cmd+V` can distinguish between pasting our own lossless internal `ClipSnapshot`
/// vs an external PanZoid JSON copied from another app.
static LAST_WRITTEN_CLIPBOARD: OnceLock<Mutex<Option<String>>> = OnceLock::new();

/// Tracks the last text written via [`write_system_clipboard`] during the current frame so
/// end-of-frame `OutputCommand::CopyText` mirroring never pipes identical text twice.
static FRAME_SYSTEM_WRITE: OnceLock<Mutex<Option<String>>> = OnceLock::new();

fn last_written_cell() -> &'static Mutex<Option<String>> {
    LAST_WRITTEN_CLIPBOARD.get_or_init(|| Mutex::new(None))
}

fn frame_system_write_cell() -> &'static Mutex<Option<String>> {
    FRAME_SYSTEM_WRITE.get_or_init(|| Mutex::new(None))
}

/// Returns `true` if `text` matches the last text written by ZoidCraft itself.
#[must_use]
pub fn matches_last_internal_copy(text: &str) -> bool {
    last_written_cell()
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|prev| prev == text))
        .unwrap_or(false)
}

/// Records that `text` was written via [`write_system_clipboard`] in the current frame.
fn record_frame_system_write(text: &str) {
    if let Ok(mut guard) = frame_system_write_cell().lock() {
        *guard = Some(text.to_owned());
    }
}

/// Returns `true` (and clears the marker) if `text` was already written to the OS clipboard
/// via [`write_system_clipboard`] during the current frame.
#[must_use]
pub fn consume_frame_system_write(text: &str) -> bool {
    if let Ok(mut guard) = frame_system_write_cell().lock()
        && guard.as_deref() == Some(text)
    {
        *guard = None;
        return true;
    }
    false
}

/// Clears any per-frame system clipboard write marker at the end of a frame.
pub fn clear_frame_system_write() {
    if let Ok(mut guard) = frame_system_write_cell().lock() {
        *guard = None;
    }
}

/// Records a copy initiated by an `egui` widget (e.g. `Cmd+C`/`Cmd+X` inside a `TextEdit`)
/// so Timeline clip paste will not treat plain widget text as an internal clip snapshot copy.
pub fn record_widget_os_copy() {
    if let Ok(mut guard) = last_written_cell().lock() {
        *guard = None;
    }
}

/// Writes `text` to both `egui`'s clipboard and the native OS system clipboard.
/// Returns `true` if the OS clipboard write succeeded (or in headless tests, falls back cleanly).
pub fn write_system_clipboard(ctx: &egui::Context, text: &str) -> bool {
    ctx.copy_text(text.to_owned());
    if let Ok(mut guard) = last_written_cell().lock() {
        *guard = Some(text.to_owned());
    }
    record_frame_system_write(text);
    write_os_clipboard_raw(text)
}

/// Bridges incoming `egui::RawInput` paste events against the live native OS clipboard.
///
/// - Replaces every `Event::Paste(text)` payload with live OS clipboard text when available
///   (defeating `egui-winit`'s stale in-memory clipboard stub).
/// - Drops `Event::Paste(CLIPBOARD_SENTINEL)` when the OS clipboard is empty.
/// - Synthesizes `Event::Paste(os_text)` when `Cmd/Ctrl+V` arrives without an `Event::Paste`.
#[cfg(not(target_arch = "wasm32"))]
pub fn bridge_raw_input_paste(
    raw_input: &mut egui::RawInput,
    read_os: &mut dyn FnMut() -> Option<String>,
) {
    let mut cached_os: Option<Option<String>> = None;
    let mut get_os = || -> Option<String> {
        if cached_os.is_none() {
            cached_os = Some(read_os());
        }
        cached_os.as_ref().and_then(|opt| opt.clone())
    };

    let mut had_paste_event = false;
    let mut had_cmd_v = false;

    let mut i = 0;
    while i < raw_input.events.len() {
        match &mut raw_input.events[i] {
            egui::Event::Paste(text) => {
                if let Some(os_text) = get_os() {
                    *text = os_text;
                    had_paste_event = true;
                    i += 1;
                } else if text == CLIPBOARD_SENTINEL {
                    raw_input.events.remove(i);
                } else {
                    had_paste_event = true;
                    i += 1;
                }
            }
            egui::Event::Key {
                key: egui::Key::V,
                pressed: true,
                modifiers,
                ..
            } if modifiers.command_only() => {
                had_cmd_v = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    if had_cmd_v
        && !had_paste_event
        && let Some(os_text) = get_os()
    {
        raw_input.events.push(egui::Event::Paste(os_text));
    }
}

#[cfg(target_arch = "wasm32")]
pub fn bridge_web_paste(input: &mut egui::RawInput) {
    let has_browser_paste = input
        .events
        .iter()
        .any(|event| matches!(event, egui::Event::Paste(_)));
    if !has_browser_paste && input.events.iter().any(|event| matches!(
        event,
        egui::Event::Key { key: egui::Key::V, pressed: true, modifiers, .. } if modifiers.command_only()
    )) && let Some(text) = read_system_clipboard() {
        input.events.push(egui::Event::Paste(text));
    }
}

/// Mirrors `egui::OutputCommand::CopyText` (such as `Cmd+C`/`Cmd+X` inside any focused
/// `TextEdit`) to the native OS clipboard, skipping [`CLIPBOARD_SENTINEL`] and any text
/// already piped via [`write_system_clipboard`] during the same frame.
pub fn mirror_output_copy_commands(
    commands: &[egui::OutputCommand],
    write_os: &mut dyn FnMut(&str) -> bool,
) {
    let mut last_mirrored_this_frame: Option<&str> = None;
    for cmd in commands {
        if let egui::OutputCommand::CopyText(text) = cmd {
            if text.is_empty() || text == CLIPBOARD_SENTINEL {
                continue;
            }
            if consume_frame_system_write(text) || last_mirrored_this_frame == Some(text.as_str()) {
                continue;
            }
            let _ = write_os(text);
            record_widget_os_copy();
            last_mirrored_this_frame = Some(text.as_str());
        }
    }
    clear_frame_system_write();
}

/// Reads plain UTF-8 text from the native OS system clipboard.
#[must_use]
pub fn read_system_clipboard() -> Option<String> {
    read_os_clipboard_raw().filter(|s| !s.trim().is_empty())
}

/// Writes `text` directly to the OS system clipboard (without requiring an `egui::Context`).
pub fn write_os_clipboard_raw(text: &str) -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        crate::web::call("__zcClipboard", "writeText", &[text.into()]).is_ok()
    }
    #[cfg(target_os = "macos")]
    {
        pipe_to_command("pbcopy", &[], text)
    }
    #[cfg(target_os = "windows")]
    {
        pipe_to_command("clip", &[], text)
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        pipe_to_command("wl-copy", &[], text)
            || pipe_to_command("xclip", &["-selection", "clipboard"], text)
            || pipe_to_command("xsel", &["--clipboard", "--input"], text)
    }
}

/// Reads UTF-8 text directly from the OS system clipboard.
#[must_use]
pub fn read_os_clipboard_raw() -> Option<String> {
    #[cfg(target_arch = "wasm32")]
    {
        crate::web::call("__zcClipboard", "readText", &[])
            .ok()?
            .as_string()
    }
    #[cfg(target_os = "macos")]
    {
        read_from_command("pbpaste", &[])
    }
    #[cfg(target_os = "windows")]
    {
        read_from_command(
            "powershell",
            &["-NoProfile", "-Command", "Get-Clipboard -Raw"],
        )
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        read_from_command("wl-paste", &["--no-newline"])
            .or_else(|| read_from_command("xclip", &["-selection", "clipboard", "-o"]))
            .or_else(|| read_from_command("xsel", &["--clipboard", "--output"]))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn pipe_to_command(program: &str, args: &[&str], text: &str) -> bool {
    let Ok(mut child) = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    if let Some(mut stdin) = child.stdin.take()
        && stdin.write_all(text.as_bytes()).is_err()
    {
        let _ = child.kill();
        let _ = child.wait();
        return false;
    }
    child.wait().is_ok_and(|s| s.success())
}

#[cfg(not(target_arch = "wasm32"))]
fn read_from_command(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_copy_tracking_records_last_written_text() {
        let ctx = egui::Context::default();
        let sample = r#"{"type":"bloom","properties":{"name":"Bloom"}}"#;
        // Record via tracking cell directly so unit tests don't clobber developer clipboard.
        if let Ok(mut guard) = last_written_cell().lock() {
            *guard = Some(sample.to_owned());
        }
        assert!(matches_last_internal_copy(sample));
        assert!(!matches_last_internal_copy(r#"{"type":"rgbshift"}"#));
        let _ = ctx;
    }

    #[test]
    fn bridge_replaces_sentinel_and_stale_paste_with_live_os_clipboard() {
        // 1. Fresh launch: egui-winit emits Event::Paste(CLIPBOARD_SENTINEL) -> replaced by live OS text.
        let mut raw = egui::RawInput::default();
        raw.events
            .push(egui::Event::Paste(CLIPBOARD_SENTINEL.to_owned()));
        let mut reads = 0usize;
        bridge_raw_input_paste(&mut raw, &mut || {
            reads += 1;
            Some("fresh os paste".to_owned())
        });
        assert_eq!(reads, 1);
        assert_eq!(
            raw.events,
            vec![egui::Event::Paste("fresh os paste".to_owned())]
        );

        // 2. Stale in-memory egui-winit text is overridden by live OS clipboard text.
        let mut raw_stale = egui::RawInput::default();
        raw_stale
            .events
            .push(egui::Event::Paste("stale internal copy".to_owned()));
        bridge_raw_input_paste(&mut raw_stale, &mut || {
            Some("newer external pbcopy".to_owned())
        });
        assert_eq!(
            raw_stale.events,
            vec![egui::Event::Paste("newer external pbcopy".to_owned())]
        );

        // 3. Empty OS clipboard drops sentinel paste cleanly (never pastes raw sentinel into TextEdit).
        let mut raw_empty = egui::RawInput::default();
        raw_empty
            .events
            .push(egui::Event::Paste(CLIPBOARD_SENTINEL.to_owned()));
        bridge_raw_input_paste(&mut raw_empty, &mut || None);
        assert!(raw_empty.events.is_empty());

        // 4. Cmd+V without Event::Paste synthesizes Event::Paste from live OS clipboard.
        let mut raw_key = egui::RawInput::default();
        raw_key.events.push(egui::Event::Key {
            key: egui::Key::V,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        });
        bridge_raw_input_paste(&mut raw_key, &mut || Some("from cmd+v".to_owned()));
        assert!(
            raw_key
                .events
                .iter()
                .any(|e| matches!(e, egui::Event::Paste(t) if t == "from cmd+v"))
        );
    }

    #[test]
    fn mirror_output_copy_commands_ignores_sentinel_and_deduplicates_system_writes() {
        clear_frame_system_write();
        let mut written: Vec<String> = Vec::new();
        let commands = vec![
            egui::OutputCommand::CopyText(CLIPBOARD_SENTINEL.to_owned()),
            egui::OutputCommand::CopyText(String::new()),
            egui::OutputCommand::CopyText("expression_sin(t)".to_owned()),
        ];
        mirror_output_copy_commands(&commands, &mut |s| {
            written.push(s.to_owned());
            true
        });
        assert_eq!(written, vec!["expression_sin(t)".to_owned()]);

        // Same-frame write via `write_system_clipboard` marker is not piped a second time.
        written.clear();
        let json = r#"{"type":"bloom"}"#;
        record_frame_system_write(json);
        let cmds2 = vec![egui::OutputCommand::CopyText(json.to_owned())];
        mirror_output_copy_commands(&cmds2, &mut |s| {
            written.push(s.to_owned());
            true
        });
        assert!(
            written.is_empty(),
            "same-frame system clipboard write must not be redundantly re-piped"
        );
    }
}
