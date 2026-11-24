//! Complete ClipMaker 3 keyboard shortcut dispatcher (Spec 01 §6).
//!
//! All shortcuts are strictly suppressed whenever `ctx.wants_keyboard_input()` is `true`.

use egui::{Context, Key, Modifiers};
use zc_session::{Command, Session};

use super::graph_editor::GraphEditorState;
use super::timeline::TimelineViewState;
use super::viewport::ViewportState;

/// Actions returned by [`dispatch_shortcuts`] that require host-shell coordination
/// (native file dialogs, floating window visibility, status toast reporting).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShortcutOutcome {
    pub handled_any: bool,
    pub new_project_requested: bool,
    pub open_project_requested: bool,
    pub save_project_requested: bool,
    pub save_without_assets: bool,
    pub toggle_graph_editor: bool,
    pub cancel_or_close: bool,
    pub last_error: Option<String>,
}

/// Dispatches all global and editor ClipMaker 3 keyboard shortcuts for the current frame.
///
/// Strictly returns an empty [`ShortcutOutcome`] without consuming any keys whenever
/// `ctx.wants_keyboard_input()` is `true` (e.g. text inputs, numeric editors, inline rename).
pub fn dispatch_shortcuts(
    ctx: &Context,
    session: &mut Session,
    timeline: &mut TimelineViewState,
    graph_editor: &mut GraphEditorState,
    viewport: &mut ViewportState,
) -> ShortcutOutcome {
    let mut outcome = ShortcutOutcome::default();
    if ctx.egui_wants_keyboard_input() {
        return outcome;
    }

    let (modifiers, pressed) = ctx.input(|i| {
        let mut m = i.modifiers;
        for ev in &i.events {
            if let egui::Event::Key {
                pressed: true,
                modifiers: km,
                ..
            } = ev
            {
                m = *km;
                break;
            }
        }
        let p = PressedSnapshot {
            space: i.key_pressed(Key::Space),
            home: i.key_pressed(Key::Home),
            end: i.key_pressed(Key::End),
            arrow_left: i.key_pressed(Key::ArrowLeft),
            arrow_right: i.key_pressed(Key::ArrowRight),
            j: i.key_pressed(Key::J),
            k: i.key_pressed(Key::K),
            l: i.key_pressed(Key::L),
            c: i.key_pressed(Key::C),
            t: i.key_pressed(Key::T),
            z: i.key_pressed(Key::Z),
            y: i.key_pressed(Key::Y),
            s: i.key_pressed(Key::S),
            o: i.key_pressed(Key::O),
            m: i.key_pressed(Key::M),
            n: i.key_pressed(Key::N),
            g: i.key_pressed(Key::G),
            d: i.key_pressed(Key::D),
            delete: i.key_pressed(Key::Delete),
            backspace: i.key_pressed(Key::Backspace),
            minus: i.key_pressed(Key::Minus),
            equals: i.key_pressed(Key::Equals) || i.key_pressed(Key::Plus),
            backslash: i.key_pressed(Key::Backslash),
            escape: i.key_pressed(Key::Escape),
        };
        (m, p)
    });

    // --- Escape: cancel active drag / rename or request closing focused overlay ---
    if pressed.escape && no_mods(modifiers) {
        let cancelled = timeline.cancel_active_interaction(session)
            || graph_editor.cancel_active_interaction()
            || viewport.cancel_active_interaction(session);
        if !cancelled && graph_editor.open {
            graph_editor.open = false;
        }
        outcome.cancel_or_close = true;
        outcome.handled_any = true;
        return outcome;
    }

    // --- File & History (Ctrl/Cmd shortcuts) ---
    if cmd_shift(modifiers) && pressed.z {
        exec(session, Command::Redo, &mut outcome);
        return outcome;
    }
    if cmd_only(modifiers) && pressed.z {
        exec(session, Command::Undo, &mut outcome);
        return outcome;
    }
    if cmd_only(modifiers) && pressed.y {
        exec(session, Command::Redo, &mut outcome);
        return outcome;
    }
    if cmd_shift(modifiers) && pressed.s {
        outcome.save_project_requested = true;
        outcome.save_without_assets = true;
        outcome.handled_any = true;
        return outcome;
    }
    if cmd_only(modifiers) && pressed.s {
        outcome.save_project_requested = true;
        outcome.save_without_assets = false;
        outcome.handled_any = true;
        return outcome;
    }
    if cmd_only(modifiers) && pressed.o {
        outcome.open_project_requested = true;
        outcome.handled_any = true;
        return outcome;
    }
    if cmd_only(modifiers) && (pressed.m || pressed.n) {
        outcome.new_project_requested = true;
        outcome.handled_any = true;
        return outcome;
    }
    if cmd_only(modifiers) && pressed.l {
        exec(session, Command::ToggleLoop, &mut outcome);
        return outcome;
    }
    if cmd_only(modifiers) && pressed.g {
        graph_editor.open = !graph_editor.open;
        outcome.toggle_graph_editor = true;
        outcome.handled_any = true;
        return outcome;
    }
    if cmd_only(modifiers) && pressed.d {
        duplicate_selected(session, &mut outcome);
        return outcome;
    }

    // --- Playback Transport ---
    if no_mods(modifiers) && pressed.space {
        exec(session, Command::TogglePlay, &mut outcome);
        return outcome;
    }
    if no_mods(modifiers) && pressed.home {
        exec(session, Command::SetFrame { frame: 0.0 }, &mut outcome);
        return outcome;
    }
    if no_mods(modifiers) && pressed.end {
        let last_frame = (session.document().sequence_length - 1.0).max(0.0);
        exec(
            session,
            Command::SetFrame { frame: last_frame },
            &mut outcome,
        );
        return outcome;
    }
    if pressed.arrow_left && (no_mods(modifiers) || shift_only(modifiers)) {
        let delta = if modifiers.shift { -10.0 } else { -1.0 };
        exec(session, Command::StepFrame { delta }, &mut outcome);
        return outcome;
    }
    if pressed.arrow_right && (no_mods(modifiers) || shift_only(modifiers)) {
        let delta = if modifiers.shift { 10.0 } else { 1.0 };
        exec(session, Command::StepFrame { delta }, &mut outcome);
        return outcome;
    }
    if no_mods(modifiers) && pressed.j {
        let next = session.playback.speed - 0.25;
        exec(session, Command::SetSpeed { speed: next }, &mut outcome);
        return outcome;
    }
    if no_mods(modifiers) && pressed.k {
        exec(session, Command::SetSpeed { speed: 0.0 }, &mut outcome);
        return outcome;
    }
    if no_mods(modifiers) && pressed.l {
        let next = session.playback.speed + 0.25;
        exec(session, Command::SetSpeed { speed: next }, &mut outcome);
        return outcome;
    }

    // --- Viewport Mode Toggles ---
    if no_mods(modifiers) && pressed.c {
        session.playback.toggle_editing_camera();
        outcome.handled_any = true;
        return outcome;
    }
    if no_mods(modifiers) && pressed.t {
        session.playback.toggle_transform_controls_2d();
        outcome.handled_any = true;
        return outcome;
    }

    // --- Timeline Clip Editing & Zoom ---
    if no_mods(modifiers) && pressed.s {
        split_selected_clips_at_playhead(session, &mut outcome);
        return outcome;
    }
    if no_mods(modifiers) && (pressed.delete || pressed.backspace) {
        delete_selected(session, &mut outcome);
        return outcome;
    }
    if (no_mods(modifiers) || shift_only(modifiers)) && pressed.equals {
        timeline.zoom_at_playhead(true, session.playback.current_frame, session.document().fps);
        outcome.handled_any = true;
        return outcome;
    }
    if no_mods(modifiers) && pressed.minus {
        timeline.zoom_at_playhead(
            false,
            session.playback.current_frame,
            session.document().fps,
        );
        outcome.handled_any = true;
        return outcome;
    }
    if no_mods(modifiers) && pressed.backslash {
        timeline.fit_sequence(session.document().sequence_length, session.document().fps);
        outcome.handled_any = true;
        return outcome;
    }

    outcome
}

// ---------------------------------------------------------------------------
// Compound Selection Command Helpers (Single Undo Transactions)
// ---------------------------------------------------------------------------

fn split_selected_clips_at_playhead(session: &mut Session, outcome: &mut ShortcutOutcome) {
    let clip_ids = session.selection.clip_ids.clone();
    if clip_ids.is_empty() {
        return;
    }
    let at_frame = session.playback.current_frame.round();
    if session.begin_transaction("Split Clips").is_ok() {
        let mut any_split = false;
        for clip_id in clip_ids {
            match session.execute(Command::SplitClip { clip_id, at_frame }) {
                Ok(_) => any_split = true,
                Err(err) => outcome.last_error = Some(err.to_string()),
            }
        }
        let _ = session.commit_transaction(any_split);
        outcome.handled_any = true;
    }
}

fn duplicate_selected(session: &mut Session, outcome: &mut ShortcutOutcome) {
    if !session.selection.object_3d_ids.is_empty()
        && let Some(clip_id) = session.selection.clip_ids.first().cloned()
    {
        let ids = session.selection.object_3d_ids.clone();
        if session.begin_transaction("Duplicate 3D Objects").is_ok() {
            let mut changed = false;
            for object_id in ids {
                if session
                    .execute(Command::DuplicateObject3D {
                        clip_id: clip_id.clone(),
                        object_id,
                    })
                    .is_ok()
                {
                    changed = true;
                }
            }
            let _ = session.commit_transaction(changed);
            outcome.handled_any = true;
        }
        return;
    }

    if !session.selection.effect_ids.is_empty()
        && let Some(clip_id) = session.selection.clip_ids.first().cloned()
    {
        let ids = session.selection.effect_ids.clone();
        if session.begin_transaction("Duplicate Effects").is_ok() {
            let mut changed = false;
            for effect_id in ids {
                if session
                    .execute(Command::DuplicateEffect {
                        clip_id: clip_id.clone(),
                        effect_id,
                    })
                    .is_ok()
                {
                    changed = true;
                }
            }
            let _ = session.commit_transaction(changed);
            outcome.handled_any = true;
        }
        return;
    }

    let clip_ids = session.selection.clip_ids.clone();
    if !clip_ids.is_empty() && session.begin_transaction("Duplicate Clips").is_ok() {
        let mut changed = false;
        for clip_id in clip_ids {
            if session.execute(Command::DuplicateClip { clip_id }).is_ok() {
                changed = true;
            }
        }
        let _ = session.commit_transaction(changed);
        outcome.handled_any = true;
    }
}

fn delete_selected(session: &mut Session, outcome: &mut ShortcutOutcome) {
    // Priority order: selected keyframes -> 3D objects -> effects -> clips.
    if !session.selection.keyframes.is_empty() {
        let kfs = session.selection.keyframes.clone();
        if session.begin_transaction("Delete Keyframes").is_ok() {
            let mut changed = false;
            for kf in kfs {
                if session
                    .execute(Command::DeleteKeyframe {
                        target_id: kf.target_id,
                        property: kf.property,
                        frame: kf.frame,
                    })
                    .is_ok()
                {
                    changed = true;
                }
            }
            let _ = session.commit_transaction(changed);
            outcome.handled_any = true;
        }
        return;
    }

    if !session.selection.object_3d_ids.is_empty()
        && let Some(clip_id) = session.selection.clip_ids.first().cloned()
    {
        let ids = session.selection.object_3d_ids.clone();
        if session.begin_transaction("Delete 3D Objects").is_ok() {
            let mut changed = false;
            for object_id in ids {
                if session
                    .execute(Command::DeleteObject3D {
                        clip_id: clip_id.clone(),
                        object_id,
                    })
                    .is_ok()
                {
                    changed = true;
                }
            }
            let _ = session.commit_transaction(changed);
            outcome.handled_any = true;
        }
        return;
    }

    if !session.selection.effect_ids.is_empty()
        && let Some(clip_id) = session.selection.clip_ids.first().cloned()
    {
        let ids = session.selection.effect_ids.clone();
        if session.begin_transaction("Delete Effects").is_ok() {
            let mut changed = false;
            for effect_id in ids {
                if session
                    .execute(Command::DeleteEffect {
                        clip_id: clip_id.clone(),
                        effect_id,
                    })
                    .is_ok()
                {
                    changed = true;
                }
            }
            let _ = session.commit_transaction(changed);
            outcome.handled_any = true;
        }
        return;
    }

    let clip_ids = session.selection.clip_ids.clone();
    if !clip_ids.is_empty() && session.begin_transaction("Delete Clips").is_ok() {
        let mut changed = false;
        for clip_id in clip_ids {
            if session.execute(Command::DeleteClip { clip_id }).is_ok() {
                changed = true;
            }
        }
        let _ = session.commit_transaction(changed);
        outcome.handled_any = true;
    }
}

fn exec(session: &mut Session, cmd: Command, outcome: &mut ShortcutOutcome) {
    if let Err(err) = session.execute(cmd) {
        outcome.last_error = Some(err.to_string());
    }
    outcome.handled_any = true;
}

#[derive(Debug, Default)]
struct PressedSnapshot {
    space: bool,
    home: bool,
    end: bool,
    arrow_left: bool,
    arrow_right: bool,
    j: bool,
    k: bool,
    l: bool,
    c: bool,
    t: bool,
    z: bool,
    y: bool,
    s: bool,
    o: bool,
    m: bool,
    n: bool,
    g: bool,
    d: bool,
    delete: bool,
    backspace: bool,
    minus: bool,
    equals: bool,
    backslash: bool,
    escape: bool,
}

fn no_mods(m: Modifiers) -> bool {
    !m.command && !m.ctrl && !m.shift && !m.alt
}

fn shift_only(m: Modifiers) -> bool {
    !m.command && !m.ctrl && m.shift && !m.alt
}

fn cmd_only(m: Modifiers) -> bool {
    (m.command || m.ctrl) && !m.shift && !m.alt
}

fn cmd_shift(m: Modifiers) -> bool {
    (m.command || m.ctrl) && m.shift && !m.alt
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn run_key_event(
        ctx: &Context,
        key: Key,
        modifiers: Modifiers,
        session: &mut Session,
        timeline: &mut TimelineViewState,
        graph: &mut GraphEditorState,
        viewport: &mut ViewportState,
    ) -> ShortcutOutcome {
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        });
        let mut outcome = ShortcutOutcome::default();
        let mut full = ctx.run_ui(raw, |ui| {
            outcome = dispatch_shortcuts(ui.ctx(), session, timeline, graph, viewport);
        });
        full.textures_delta.clear();
        outcome
    }

    #[test]
    fn shortcuts_dispatch_playback_zoom_and_toggles() {
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        // Space toggles play.
        let out = run_key_event(
            &ctx,
            Key::Space,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out.handled_any);
        assert!(session.playback.is_playing());

        // Shift+ArrowRight steps +10 frames (and pauses playback).
        let _ = run_key_event(
            &ctx,
            Key::ArrowRight,
            Modifiers::SHIFT,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(!session.playback.is_playing());
        assert!((session.playback.current_frame - 10.0).abs() < 1e-6);

        // '=' zooms timeline in anchored at playhead.
        let z0 = timeline.zoom;
        let _ = run_key_event(
            &ctx,
            Key::Equals,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(timeline.zoom > z0);

        // Ctrl+G toggles Graph Editor window.
        let out_g = run_key_event(
            &ctx,
            Key::G,
            Modifiers::COMMAND,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out_g.toggle_graph_editor);
        assert!(graph.open);
    }
}
