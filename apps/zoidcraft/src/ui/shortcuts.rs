//! Complete ClipMaker 3 keyboard shortcut dispatcher (Spec 01 §6).
//!
//! All shortcuts are strictly suppressed whenever `ctx.wants_keyboard_input()` is `true`.

use std::collections::BTreeSet;

use egui::{Context, Key, Modifiers};
use zc_session::{Command, Session};

use super::graph_editor::{GraphDragState, GraphEditorState};
use super::timeline::TimelineViewState;
use super::viewport::{ViewportDragState, ViewportState};

/// Actions returned by [`dispatch_shortcuts`] that require host-shell coordination
/// (native file dialogs, floating window visibility, status toast reporting).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShortcutOutcome {
    pub handled_any: bool,
    pub handled_keys: Vec<Key>,
    pub new_project_requested: bool,
    pub open_project_requested: bool,
    pub save_project_requested: bool,
    pub save_without_assets: bool,
    pub toggle_graph_editor: bool,
    pub cancel_or_close: bool,
    pub last_error: Option<String>,
}

impl ShortcutOutcome {
    fn mark_handled(&mut self, key: Key) {
        self.handled_any = true;
        if !self.handled_keys.contains(&key) {
            self.handled_keys.push(key);
        }
    }
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

    let pressed = ctx.input(|i| {
        let mut p = PressedSnapshot::default();
        for ev in &i.events {
            if let egui::Event::Key {
                key,
                pressed: true,
                repeat,
                modifiers,
                ..
            } = ev
            {
                let m = *modifiers;
                match key {
                    Key::Space if !*repeat => {
                        p.space.get_or_insert(m);
                    }
                    Key::Num0 => {
                        p.num0.get_or_insert(m);
                    }
                    Key::Home => {
                        p.home.get_or_insert(m);
                    }
                    Key::End => {
                        p.end.get_or_insert(m);
                    }
                    Key::ArrowLeft => {
                        p.arrow_left.get_or_insert(m);
                    }
                    Key::ArrowRight => {
                        p.arrow_right.get_or_insert(m);
                    }
                    Key::J => {
                        p.j.get_or_insert(m);
                    }
                    Key::K => {
                        p.k.get_or_insert(m);
                    }
                    Key::L => {
                        p.l.get_or_insert(m);
                    }
                    Key::C => {
                        p.c.get_or_insert(m);
                    }
                    Key::T => {
                        p.t.get_or_insert(m);
                    }
                    Key::Z => {
                        p.z.get_or_insert(m);
                    }
                    Key::Y => {
                        p.y.get_or_insert(m);
                    }
                    Key::S => {
                        p.s.get_or_insert(m);
                    }
                    Key::O => {
                        p.o.get_or_insert(m);
                    }
                    Key::M => {
                        p.m.get_or_insert(m);
                    }
                    Key::N => {
                        p.n.get_or_insert(m);
                    }
                    Key::G => {
                        p.g.get_or_insert(m);
                    }
                    Key::D => {
                        p.d.get_or_insert(m);
                    }
                    Key::Delete => {
                        p.delete.get_or_insert(m);
                    }
                    Key::Backspace => {
                        p.backspace.get_or_insert(m);
                    }
                    Key::Minus => {
                        p.minus.get_or_insert(m);
                    }
                    Key::Equals => {
                        p.equals.get_or_insert(m);
                    }
                    Key::Plus => {
                        p.plus.get_or_insert(m);
                    }
                    Key::Backslash => {
                        p.backslash.get_or_insert(m);
                    }
                    Key::Escape => {
                        p.escape.get_or_insert(m);
                    }
                    _ => {}
                }
            }
        }
        p
    });

    // --- Escape: cancel active drag / rename or close Graph Editor ---
    // Only claim Escape when something was actually cancelled or closed so popups,
    // color pickers, and numeric scrubbers still receive Escape cleanly.
    if let Some(m) = pressed.escape
        && no_mods(m)
    {
        let cancelled = timeline.cancel_active_interaction(session)
            || graph_editor.cancel_active_interaction(session)
            || viewport.cancel_active_interaction(session);
        let closed_graph = if !cancelled && graph_editor.open {
            graph_editor.open = false;
            true
        } else {
            false
        };
        if cancelled || closed_graph {
            outcome.cancel_or_close = true;
            outcome.mark_handled(Key::Escape);
            return outcome;
        }
    }

    // --- File & History (Ctrl/Cmd shortcuts) ---
    if let Some(m) = pressed.z
        && cmd_shift(m)
    {
        exec(session, Command::Redo, Key::Z, &mut outcome);
        return outcome;
    }
    if let Some(m) = pressed.z
        && cmd_only(m)
    {
        exec(session, Command::Undo, Key::Z, &mut outcome);
        return outcome;
    }
    if let Some(m) = pressed.y
        && cmd_only(m)
    {
        exec(session, Command::Redo, Key::Y, &mut outcome);
        return outcome;
    }
    if let Some(m) = pressed.s
        && cmd_shift(m)
    {
        outcome.save_project_requested = true;
        outcome.save_without_assets = true;
        outcome.mark_handled(Key::S);
        return outcome;
    }
    if let Some(m) = pressed.s
        && cmd_only(m)
    {
        outcome.save_project_requested = true;
        outcome.save_without_assets = false;
        outcome.mark_handled(Key::S);
        return outcome;
    }
    if let Some(m) = pressed.o
        && cmd_only(m)
    {
        outcome.open_project_requested = true;
        outcome.mark_handled(Key::O);
        return outcome;
    }
    if let Some(m) = pressed.m
        && cmd_only(m)
    {
        outcome.new_project_requested = true;
        outcome.mark_handled(Key::M);
        return outcome;
    }
    if let Some(m) = pressed.n
        && cmd_only(m)
    {
        outcome.new_project_requested = true;
        outcome.mark_handled(Key::N);
        return outcome;
    }
    if let Some(m) = pressed.l
        && cmd_only(m)
    {
        exec(session, Command::ToggleLoop, Key::L, &mut outcome);
        return outcome;
    }
    if let Some(m) = pressed.g
        && cmd_only(m)
    {
        if graph_editor.open {
            graph_editor.cancel_active_interaction(session);
            graph_editor.open = false;
        } else {
            graph_editor.open = true;
            graph_editor.window_focused = true;
        }
        outcome.toggle_graph_editor = true;
        outcome.mark_handled(Key::G);
        return outcome;
    }
    if let Some(m) = pressed.d
        && cmd_only(m)
    {
        duplicate_selected(session, &mut outcome);
        return outcome;
    }
    if let Some(m) = pressed.num0
        && cmd_only(m)
    {
        viewport.reset_2d_view();
        // Consume Cmd/Ctrl+0 so egui's global gui_zoom does not rescale the UI chrome.
        ctx.input_mut(|i| {
            let _ = i.consume_shortcut(&egui::KeyboardShortcut::new(Modifiers::COMMAND, Key::Num0));
            let _ = i.consume_shortcut(&egui::KeyboardShortcut::new(Modifiers::CTRL, Key::Num0));
        });
        outcome.mark_handled(Key::Num0);
        return outcome;
    }

    // --- Playback Transport ---
    let space_pan_active = matches!(graph_editor.drag_state, GraphDragState::Panning { .. })
        || matches!(
            viewport.drag_state,
            ViewportDragState::Canvas2DPan | ViewportDragState::CameraPan
        )
        || ctx.input(|i| i.pointer.primary_down());
    if let Some(m) = pressed.space
        && no_mods(m)
        && !space_pan_active
    {
        let was_playing = session.playback.is_playing();
        exec(session, Command::TogglePlay, Key::Space, &mut outcome);
        if was_playing && !session.playback.is_playing() {
            snap_paused_playhead_to_integer(session);
        }
        return outcome;
    }
    if let Some(m) = pressed.home
        && no_mods(m)
    {
        exec(
            session,
            Command::SetFrame { frame: 0.0 },
            Key::Home,
            &mut outcome,
        );
        return outcome;
    }
    if let Some(m) = pressed.end
        && no_mods(m)
    {
        let last_frame = (session.document().sequence_length - 1.0).max(0.0);
        exec(
            session,
            Command::SetFrame { frame: last_frame },
            Key::End,
            &mut outcome,
        );
        return outcome;
    }
    if let Some(m) = pressed.arrow_left
        && (no_mods(m) || shift_only(m))
    {
        let delta = if m.shift { -5.0 } else { -1.0 };
        exec(
            session,
            Command::StepFrame { delta },
            Key::ArrowLeft,
            &mut outcome,
        );
        return outcome;
    }
    if let Some(m) = pressed.arrow_right
        && (no_mods(m) || shift_only(m))
    {
        let delta = if m.shift { 5.0 } else { 1.0 };
        exec(
            session,
            Command::StepFrame { delta },
            Key::ArrowRight,
            &mut outcome,
        );
        return outcome;
    }
    if let Some(m) = pressed.j
        && no_mods(m)
    {
        let next = session.playback.speed - 0.25;
        exec(
            session,
            Command::SetSpeed { speed: next },
            Key::J,
            &mut outcome,
        );
        return outcome;
    }
    if let Some(m) = pressed.k
        && no_mods(m)
    {
        let was_playing = session.playback.is_playing();
        exec(
            session,
            Command::SetSpeed { speed: 0.0 },
            Key::K,
            &mut outcome,
        );
        if was_playing && !session.playback.is_playing() {
            snap_paused_playhead_to_integer(session);
        }
        return outcome;
    }
    if let Some(m) = pressed.l
        && no_mods(m)
    {
        let next = session.playback.speed + 0.25;
        exec(
            session,
            Command::SetSpeed { speed: next },
            Key::L,
            &mut outcome,
        );
        return outcome;
    }

    // --- Viewport Mode Toggles ---
    if let Some(m) = pressed.c
        && no_mods(m)
    {
        session.playback.toggle_editing_camera();
        outcome.mark_handled(Key::C);
        return outcome;
    }
    if let Some(m) = pressed.t
        && no_mods(m)
    {
        session.playback.toggle_transform_controls_2d();
        outcome.mark_handled(Key::T);
        return outcome;
    }

    // --- Timeline Clip Editing & Zoom ---
    if let Some(m) = pressed.s
        && no_mods(m)
    {
        split_selected_clips_at_playhead(session, &mut outcome);
        return outcome;
    }
    let delete_key = if let Some(m) = pressed.delete
        && no_mods(m)
    {
        Some(Key::Delete)
    } else if let Some(m) = pressed.backspace
        && no_mods(m)
    {
        Some(Key::Backspace)
    } else {
        None
    };
    if let Some(key) = delete_key {
        // Prune stale Graph Editor keyframe refs (e.g. after undo, timeline move, or clip deletion)
        // so non-existent keyframes never hijack Delete/Backspace globally.
        graph_editor.selected_keys.retain(|gk| {
            keyframe_exists_in_session(session, &gk.target_id, &gk.property, gk.local_frame())
        });
        if graph_editor.open
            && graph_editor.window_focused
            && !graph_editor.selected_keys.is_empty()
        {
            delete_graph_editor_keyframes(session, graph_editor, key, &mut outcome);
        } else {
            delete_selected(session, key, &mut outcome);
        }
        return outcome;
    }
    if let Some(m) = pressed.equals
        && (no_mods(m) || shift_only(m))
    {
        timeline.zoom_at_playhead(true, session.playback.current_frame, session.document().fps);
        outcome.mark_handled(Key::Equals);
        return outcome;
    }
    if let Some(m) = pressed.plus
        && (no_mods(m) || shift_only(m))
    {
        timeline.zoom_at_playhead(true, session.playback.current_frame, session.document().fps);
        outcome.mark_handled(Key::Plus);
        return outcome;
    }
    if let Some(m) = pressed.minus
        && no_mods(m)
    {
        timeline.zoom_at_playhead(
            false,
            session.playback.current_frame,
            session.document().fps,
        );
        outcome.mark_handled(Key::Minus);
        return outcome;
    }
    if let Some(m) = pressed.backslash
        && no_mods(m)
    {
        timeline.fit_sequence(session.document().sequence_length, session.document().fps);
        outcome.mark_handled(Key::Backslash);
        return outcome;
    }

    outcome
}

// ---------------------------------------------------------------------------
// Compound Selection Command Helpers (Single Undo Transactions)
// ---------------------------------------------------------------------------

fn prune_stale_clip_subselection(session: &mut Session) {
    let Some(clip_id) = session.selection.clip_ids.first().cloned() else {
        session.selection.object_3d_ids.clear();
        session.selection.object_3d_base_type = None;
        session.selection.effect_ids.clear();
        session.selection.effect_base_type = None;
        return;
    };

    let (valid_objects, valid_effects) = {
        let doc = session.document();
        let found_clip = doc
            .video_tracks
            .iter()
            .chain(doc.audio_tracks.iter())
            .flat_map(|t| t.clips.iter())
            .find(|c| c.id == clip_id);
        match found_clip {
            Some(clip) => (
                clip.objects_3d
                    .iter()
                    .map(|o| o.id.clone())
                    .collect::<BTreeSet<_>>(),
                clip.effects
                    .iter()
                    .map(|e| e.id.clone())
                    .collect::<BTreeSet<_>>(),
            ),
            None => (BTreeSet::new(), BTreeSet::new()),
        }
    };

    session
        .selection
        .object_3d_ids
        .retain(|id| valid_objects.contains(id));
    if session.selection.object_3d_ids.is_empty() {
        session.selection.object_3d_base_type = None;
    }

    session
        .selection
        .effect_ids
        .retain(|id| valid_effects.contains(id));
    if session.selection.effect_ids.is_empty() {
        session.selection.effect_base_type = None;
    }
}

fn keyframe_exists_in_session(
    session: &Session,
    target_id: &str,
    property: &str,
    frame: f64,
) -> bool {
    let doc = session.document();
    let props = if target_id == "sequence" {
        Some(&doc.sequence_properties)
    } else {
        doc.video_tracks
            .iter()
            .chain(doc.audio_tracks.iter())
            .flat_map(|t| t.clips.iter())
            .find_map(|clip| {
                if clip.id == target_id {
                    Some(&clip.properties)
                } else if let Some(obj) = clip.objects_3d.iter().find(|o| o.id == target_id) {
                    Some(&obj.properties)
                } else if let Some(fx) = clip.effects.iter().find(|e| e.id == target_id) {
                    Some(&fx.properties)
                } else {
                    None
                }
            })
    };
    props
        .and_then(|map| map.get(property))
        .is_some_and(|p| p.keyframes.iter().any(|kf| (kf.frame - frame).abs() < 1e-6))
}

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
        outcome.mark_handled(Key::S);
    }
}

fn duplicate_selected(session: &mut Session, outcome: &mut ShortcutOutcome) {
    prune_stale_clip_subselection(session);

    if session.selection.clip_ids.len() == 1
        && !session.selection.object_3d_ids.is_empty()
        && let Some(clip_id) = session.selection.clip_ids.first().cloned()
    {
        let ids = session.selection.object_3d_ids.clone();
        if session.begin_transaction("Duplicate 3D Objects").is_ok() {
            let mut changed = false;
            for object_id in ids {
                match session.execute(Command::DuplicateObject3D {
                    clip_id: clip_id.clone(),
                    object_id,
                }) {
                    Ok(_) => changed = true,
                    Err(err) => outcome.last_error = Some(err.to_string()),
                }
            }
            let _ = session.commit_transaction(changed);
            outcome.mark_handled(Key::D);
        }
        return;
    }

    if session.selection.clip_ids.len() == 1
        && !session.selection.effect_ids.is_empty()
        && let Some(clip_id) = session.selection.clip_ids.first().cloned()
    {
        let ids = session.selection.effect_ids.clone();
        if session.begin_transaction("Duplicate Effects").is_ok() {
            let mut changed = false;
            for effect_id in ids {
                match session.execute(Command::DuplicateEffect {
                    clip_id: clip_id.clone(),
                    effect_id,
                }) {
                    Ok(_) => changed = true,
                    Err(err) => outcome.last_error = Some(err.to_string()),
                }
            }
            let _ = session.commit_transaction(changed);
            outcome.mark_handled(Key::D);
        }
        return;
    }

    let clip_ids = session.selection.clip_ids.clone();
    if !clip_ids.is_empty() && session.begin_transaction("Duplicate Clips").is_ok() {
        let mut changed = false;
        for clip_id in clip_ids {
            match session.execute(Command::DuplicateClip { clip_id }) {
                Ok(_) => changed = true,
                Err(err) => outcome.last_error = Some(err.to_string()),
            }
        }
        let _ = session.commit_transaction(changed);
        outcome.mark_handled(Key::D);
    }
}

fn delete_graph_editor_keyframes(
    session: &mut Session,
    graph_editor: &mut GraphEditorState,
    key: Key,
    outcome: &mut ShortcutOutcome,
) {
    // Deduplicate multi-channel vector GraphKeyIds targeting the same (target_id, property, frame).
    let mut seen = BTreeSet::new();
    let mut unique_targets = Vec::new();
    for gk in &graph_editor.selected_keys {
        let dedup_key = (gk.target_id.clone(), gk.property.clone(), gk.frame_bits);
        if seen.insert(dedup_key.clone()) {
            unique_targets.push((dedup_key, gk.local_frame()));
        }
    }

    let mut deleted = BTreeSet::new();
    if session.begin_transaction("Delete Keyframes").is_ok() {
        let mut changed = false;
        for ((target_id, property, frame_bits), frame) in unique_targets {
            match session.execute(Command::DeleteKeyframe {
                target_id: target_id.clone(),
                property: property.clone(),
                frame,
            }) {
                Ok(_) => {
                    changed = true;
                    deleted.insert((target_id, property, frame_bits));
                }
                Err(err) => outcome.last_error = Some(err.to_string()),
            }
        }
        let _ = session.commit_transaction(changed);
        if !deleted.is_empty() {
            graph_editor.selected_keys.retain(|gk| {
                !deleted.contains(&(gk.target_id.clone(), gk.property.clone(), gk.frame_bits))
            });
            graph_editor.handles.retain(|gk, _| {
                !deleted.contains(&(gk.target_id.clone(), gk.property.clone(), gk.frame_bits))
            });
        }
        outcome.mark_handled(key);
    }
}

fn delete_selected(session: &mut Session, key: Key, outcome: &mut ShortcutOutcome) {
    prune_stale_clip_subselection(session);

    // Prune stale keyframe refs so non-existent keyframes never block deleting objects/clips.
    let valid_kfs: Vec<_> = session
        .selection
        .keyframes
        .iter()
        .filter(|kf| keyframe_exists_in_session(session, &kf.target_id, &kf.property, kf.frame))
        .cloned()
        .collect();
    session.selection.keyframes = valid_kfs;

    // Priority order: selected keyframes -> 3D objects -> effects -> clips.
    if !session.selection.keyframes.is_empty() {
        let kfs = session.selection.keyframes.clone();
        if session.begin_transaction("Delete Keyframes").is_ok() {
            let mut changed = false;
            for kf in kfs {
                match session.execute(Command::DeleteKeyframe {
                    target_id: kf.target_id,
                    property: kf.property,
                    frame: kf.frame,
                }) {
                    Ok(_) => changed = true,
                    Err(err) => outcome.last_error = Some(err.to_string()),
                }
            }
            let _ = session.commit_transaction(changed);
            outcome.mark_handled(key);
        }
        return;
    }

    if session.selection.clip_ids.len() == 1
        && !session.selection.object_3d_ids.is_empty()
        && let Some(clip_id) = session.selection.clip_ids.first().cloned()
    {
        let ids = session.selection.object_3d_ids.clone();
        if session.begin_transaction("Delete 3D Objects").is_ok() {
            let mut changed = false;
            for object_id in ids {
                match session.execute(Command::DeleteObject3D {
                    clip_id: clip_id.clone(),
                    object_id,
                }) {
                    Ok(_) => changed = true,
                    Err(err) => outcome.last_error = Some(err.to_string()),
                }
            }
            let _ = session.commit_transaction(changed);
            outcome.mark_handled(key);
        }
        return;
    }

    if session.selection.clip_ids.len() == 1
        && !session.selection.effect_ids.is_empty()
        && let Some(clip_id) = session.selection.clip_ids.first().cloned()
    {
        let ids = session.selection.effect_ids.clone();
        if session.begin_transaction("Delete Effects").is_ok() {
            let mut changed = false;
            for effect_id in ids {
                match session.execute(Command::DeleteEffect {
                    clip_id: clip_id.clone(),
                    effect_id,
                }) {
                    Ok(_) => changed = true,
                    Err(err) => outcome.last_error = Some(err.to_string()),
                }
            }
            let _ = session.commit_transaction(changed);
            outcome.mark_handled(key);
        }
        return;
    }

    let clip_ids = session.selection.clip_ids.clone();
    if !clip_ids.is_empty() && session.begin_transaction("Delete Clips").is_ok() {
        let mut changed = false;
        for clip_id in clip_ids {
            match session.execute(Command::DeleteClip { clip_id }) {
                Ok(_) => changed = true,
                Err(err) => outcome.last_error = Some(err.to_string()),
            }
        }
        let _ = session.commit_transaction(changed);
        outcome.mark_handled(key);
    }
}

fn snap_paused_playhead_to_integer(session: &mut Session) {
    let max_frame = (session.document().sequence_length - 1.0).max(0.0);
    session.playback.current_frame = session.playback.current_frame.round().clamp(0.0, max_frame);
}

fn exec(session: &mut Session, cmd: Command, key: Key, outcome: &mut ShortcutOutcome) {
    if let Err(err) = session.execute(cmd) {
        outcome.last_error = Some(err.to_string());
    }
    outcome.mark_handled(key);
}

#[derive(Debug, Default)]
struct PressedSnapshot {
    space: Option<Modifiers>,
    num0: Option<Modifiers>,
    home: Option<Modifiers>,
    end: Option<Modifiers>,
    arrow_left: Option<Modifiers>,
    arrow_right: Option<Modifiers>,
    j: Option<Modifiers>,
    k: Option<Modifiers>,
    l: Option<Modifiers>,
    c: Option<Modifiers>,
    t: Option<Modifiers>,
    z: Option<Modifiers>,
    y: Option<Modifiers>,
    s: Option<Modifiers>,
    o: Option<Modifiers>,
    m: Option<Modifiers>,
    n: Option<Modifiers>,
    g: Option<Modifiers>,
    d: Option<Modifiers>,
    delete: Option<Modifiers>,
    backspace: Option<Modifiers>,
    minus: Option<Modifiers>,
    equals: Option<Modifiers>,
    plus: Option<Modifiers>,
    backslash: Option<Modifiers>,
    escape: Option<Modifiers>,
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
    use crate::ui::graph_editor::GraphKeyId;

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
        assert_eq!(out.handled_keys, vec![Key::Space]);
        assert!(session.playback.is_playing());

        // Shift+ArrowRight steps +5 frames (and pauses playback).
        let out_step = run_key_event(
            &ctx,
            Key::ArrowRight,
            Modifiers::SHIFT,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert_eq!(out_step.handled_keys, vec![Key::ArrowRight]);
        assert!(!session.playback.is_playing());
        assert!((session.playback.current_frame - 5.0).abs() < 1e-6);

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

        // Escape does not claim handled_any when nothing is active and Graph Editor is closed.
        let out_esc_idle = run_key_event(
            &ctx,
            Key::Escape,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(!out_esc_idle.handled_any);
        assert!(out_esc_idle.handled_keys.is_empty());

        // Ctrl+G toggles Graph Editor window open, then closes while cancelling active interaction.
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

        graph.show_all_easings_popup = true;
        let out_g_close = run_key_event(
            &ctx,
            Key::G,
            Modifiers::COMMAND,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out_g_close.toggle_graph_editor);
        assert!(!graph.open);
        assert!(!graph.show_all_easings_popup);
    }

    #[test]
    fn delete_in_graph_editor_deletes_selected_keyframes_and_preserves_sole() {
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session
            .selection
            .select_clip(clip_id.clone(), Some("scene"), false);
        session
            .execute(Command::SetKeyframe {
                target_id: clip_id.clone(),
                property: "opacity".to_owned(),
                frame: 0.0,
                value: serde_json::json!(1.0),
                tween: Some(1),
            })
            .expect("kf 0");
        session
            .execute(Command::SetKeyframe {
                target_id: clip_id.clone(),
                property: "opacity".to_owned(),
                frame: 10.0,
                value: serde_json::json!(0.5),
                tween: Some(1),
            })
            .expect("kf 10");

        graph.open = true;
        graph.selected_keys = vec![GraphKeyId::new(&clip_id, "opacity", None, 10.0)];

        let out = run_key_event(
            &ctx,
            Key::Delete,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out.handled_any);
        assert_eq!(out.handled_keys, vec![Key::Delete]);
        assert_eq!(session.document().video_tracks[0].clips.len(), 1);
        assert_eq!(
            session.document().video_tracks[0].clips[0].properties["opacity"]
                .keyframes
                .len(),
            1
        );
        assert!(graph.selected_keys.is_empty());

        // Attempting to delete the sole remaining keyframe preserves it and surfaces error.
        graph.selected_keys = vec![GraphKeyId::new(&clip_id, "opacity", None, 0.0)];
        let out_sole = run_key_event(
            &ctx,
            Key::Backspace,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out_sole.handled_any);
        assert!(out_sole.last_error.is_some());
        assert_eq!(
            session.document().video_tracks[0].clips[0].properties["opacity"]
                .keyframes
                .len(),
            1
        );
    }

    #[test]
    fn stale_or_unfocused_graph_editor_keys_never_hijack_delete() {
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session
            .selection
            .select_clip(clip_id.clone(), Some("scene"), false);

        // Stale Graph Editor keyframe (no longer exists in session): must be pruned and fall
        // through to deleting the selected clip rather than failing with NotFound.
        graph.open = true;
        graph.window_focused = true;
        graph.selected_keys = vec![GraphKeyId::new(&clip_id, "opacity", None, 999.0)];

        let out = run_key_event(
            &ctx,
            Key::Delete,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out.handled_any);
        assert!(out.last_error.is_none());
        assert!(graph.selected_keys.is_empty());
        assert!(session.document().video_tracks[0].clips.is_empty());

        // Undo clip delete, then verify unfocused Graph Editor with a valid sole keyframe
        // lets Delete act on the selected clip instead of failing on the sole keyframe.
        session.execute(Command::Undo).expect("undo delete");
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session
            .selection
            .select_clip(clip_id.clone(), Some("scene"), false);
        graph.open = true;
        graph.window_focused = false;
        graph.selected_keys = vec![GraphKeyId::new(&clip_id, "opacity", None, 0.0)];

        let out2 = run_key_event(
            &ctx,
            Key::Delete,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out2.handled_any);
        assert!(out2.last_error.is_none());
        assert!(session.document().video_tracks[0].clips.is_empty());
    }

    #[test]
    fn multi_clip_selection_delete_and_duplicate_ignore_stale_subselection() {
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        let track_id = session.document().video_tracks[0].id.clone();
        let clip_a = session.document().video_tracks[0].clips[0].id.clone();
        let obj_id = session.document().video_tracks[0].clips[0].objects_3d[0]
            .id
            .clone();
        let orig_obj_count = session.document().video_tracks[0].clips[0].objects_3d.len();

        let clip_b = session
            .execute(Command::AddClip {
                track_id,
                kind: "scene".to_owned(),
                name: Some("Clip B".to_owned()),
                start: 400.0,
                length: 60.0,
            })
            .expect("add clip b")["clip_id"]
            .as_str()
            .unwrap()
            .to_owned();

        // Select Clip A + sub-selected 3D object, then shift-select Clip B.
        let _ = session.selection.select_clip(clip_a, Some("scene"), false);
        let _ = session.selection.select_object_3d(obj_id, "text", false);
        let _ = session.selection.select_clip(clip_b, Some("scene"), true);
        assert_eq!(session.selection.clip_ids.len(), 2);

        // Ctrl+D should duplicate both selected clips, NOT Clip A's 3D object.
        let out_dup = run_key_event(
            &ctx,
            Key::D,
            Modifiers::COMMAND,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out_dup.handled_any);
        assert_eq!(session.document().video_tracks[0].clips.len(), 4);
        assert_eq!(
            session.document().video_tracks[0].clips[0].objects_3d.len(),
            orig_obj_count
        );

        // Delete should delete the multi-selected clips, NOT Clip A's 3D object.
        let _ = session.selection.select_clip(
            session.document().video_tracks[0].clips[0].id.clone(),
            Some("scene"),
            false,
        );
        let _ = session.selection.select_object_3d(
            session.document().video_tracks[0].clips[0].objects_3d[0]
                .id
                .clone(),
            "text",
            false,
        );
        let _ = session.selection.select_clip(
            session.document().video_tracks[0].clips[1].id.clone(),
            Some("scene"),
            true,
        );
        let out_del = run_key_event(
            &ctx,
            Key::Delete,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out_del.handled_any);
        assert_eq!(session.document().video_tracks[0].clips.len(), 2);
    }

    #[test]
    fn pausing_on_fractional_frame_snaps_playhead_to_integer() {
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        session.playback.speed = 1.0;
        session.playback.current_frame = 47.382;
        let _ = run_key_event(
            &ctx,
            Key::Space,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(!session.playback.is_playing());
        assert!((session.playback.current_frame - 47.0).abs() < 1e-6);
    }

    #[test]
    fn cmd_zero_resets_2d_viewport_without_scaling_egui_zoom() {
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        viewport.zoom_2d = 2.5;
        viewport.pan_2d = egui::vec2(120.0, -45.0);
        let initial_gui_zoom = ctx.zoom_factor();

        let out = run_key_event(
            &ctx,
            Key::Num0,
            Modifiers::COMMAND,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out.handled_any);
        assert_eq!(out.handled_keys, vec![Key::Num0]);
        assert!((viewport.zoom_2d - 1.0).abs() < 1e-6);
        assert_eq!(viewport.pan_2d, egui::Vec2::ZERO);
        assert!((ctx.zoom_factor() - initial_gui_zoom).abs() < 1e-6);
    }

    #[test]
    fn space_does_not_toggle_play_while_canvas_pan_drag_is_active() {
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        viewport.drag_state = ViewportDragState::Canvas2DPan;
        let out = run_key_event(
            &ctx,
            Key::Space,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(!out.handled_any);
        assert!(!session.playback.is_playing());
    }
}
