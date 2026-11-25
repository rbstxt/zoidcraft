//! Complete ClipMaker 3 keyboard shortcut dispatcher (Spec 01 §6).
//!
//! All shortcuts are strictly suppressed whenever `ctx.wants_keyboard_input()` is `true`.

use std::collections::BTreeSet;

use egui::{Context, Key, Modifiers};
use zc_project::Uuid;
use zc_session::{Command, Session};

use super::graph_editor::{GraphDragState, GraphEditorState};
use super::layout::PanelFocus;
use super::panels::SideTab;
use super::timeline::TimelineViewState;
use super::viewport::{Gizmo3DMode, ViewportDragState, ViewportState};

use super::clipboard::CLIPBOARD_SENTINEL;
#[cfg(not(test))]
use super::clipboard::{matches_last_internal_copy, read_system_clipboard, write_system_clipboard};

/// Actions returned by [`dispatch_shortcuts`] that require host-shell coordination
/// (native file dialogs, floating window visibility, status toast reporting).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShortcutOutcome {
    pub handled_any: bool,
    pub handled_keys: Vec<Key>,
    pub consumed_clipboard_events: bool,
    pub new_project_requested: bool,
    pub open_project_requested: bool,
    pub save_project_requested: bool,
    pub save_without_assets: bool,
    pub toggle_graph_editor: bool,
    pub cancel_or_close: bool,
    pub status_message: Option<String>,
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

/// Returns `true` if the current selection is a single 3D Scene clip.
#[must_use]
pub fn selected_clip_is_scene3d(session: &Session) -> bool {
    if session.selection.clip_ids.len() != 1 {
        return false;
    }
    let Some(sel_id) = session.selection.clip_ids.first() else {
        return false;
    };
    session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .any(|c| &c.id == sel_id && c.kind == "scene")
}

/// Finds the preferred 3D Scene clip to target for Editing Camera:
/// first an active visible `"scene"` clip covering the playhead (top-most track first),
/// then any active `"scene"` clip at the playhead, then the first `"scene"` clip in the sequence.
#[must_use]
pub fn find_preferred_scene3d_clip_id(session: &Session) -> Option<String> {
    let frame = session.playback.current_frame;
    let tracks = &session.document().video_tracks;
    if let Some(clip_id) = tracks.iter().rev().find_map(|t| {
        if !t.enabled {
            return None;
        }
        t.clips
            .iter()
            .find(|c| {
                c.kind == "scene" && frame >= c.start - 1e-6 && frame < c.start + c.length - 1e-6
            })
            .map(|c| c.id.clone())
    }) {
        return Some(clip_id);
    }
    if let Some(clip_id) = tracks.iter().rev().find_map(|t| {
        t.clips
            .iter()
            .find(|c| {
                c.kind == "scene" && frame >= c.start - 1e-6 && frame < c.start + c.length - 1e-6
            })
            .map(|c| c.id.clone())
    }) {
        return Some(clip_id);
    }
    tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.kind == "scene")
        .map(|c| c.id.clone())
}

/// Toggles Editing Camera mode while keeping `session.playback.editing_camera` and
/// 3D Scene clip selection cleanly synchronized:
/// - If Editing Camera is currently active, turns it OFF.
/// - Otherwise, ensures a `"scene"` clip is selected (auto-selecting an active scene clip
///   at the playhead or the first scene clip in the sequence) and turns Editing Camera ON.
/// - If no `"scene"` clip exists in the sequence, keeps `editing_camera = false`.
pub fn toggle_editing_camera_synced(session: &mut Session) -> bool {
    let scene_selected = selected_clip_is_scene3d(session);
    let currently_active = session
        .selection
        .is_editing_camera_active(&session.playback, scene_selected);
    if currently_active {
        session.playback.editing_camera = false;
        return false;
    }
    if !scene_selected {
        if let Some(scene_id) = find_preferred_scene3d_clip_id(session) {
            let _ = session
                .selection
                .select_clip(scene_id, Some("scene"), false);
        } else {
            session.playback.editing_camera = false;
            return false;
        }
    }
    session.playback.editing_camera = true;
    true
}

/// Dispatches all global and editor ClipMaker 3 keyboard shortcuts for the current frame
/// using [`PanelFocus::Timeline`] as the default focus context.
///
/// Strictly returns an empty [`ShortcutOutcome`] without consuming any keys whenever
/// `ctx.egui_wants_keyboard_input()` is `true` (e.g. text inputs, numeric editors, inline rename).
pub fn dispatch_shortcuts(
    ctx: &Context,
    session: &mut Session,
    timeline: &mut TimelineViewState,
    graph_editor: &mut GraphEditorState,
    viewport: &mut ViewportState,
) -> ShortcutOutcome {
    dispatch_shortcuts_with_focus(
        ctx,
        session,
        timeline,
        graph_editor,
        viewport,
        PanelFocus::Timeline,
    )
}

/// Pluggable OS clipboard callbacks used by [`dispatch_shortcuts_with_clipboard`].
pub struct ClipboardAccess<'a> {
    pub write: &'a mut dyn FnMut(&Context, &str) -> bool,
    pub read: &'a mut dyn FnMut() -> Option<String>,
    pub matches_internal: &'a dyn Fn(&str) -> bool,
}

#[cfg(test)]
mod test_clipboard {
    use std::cell::RefCell;

    use egui::Context;

    #[derive(Default)]
    struct TestClipboardState {
        os_text: Option<String>,
        last_internal: Option<String>,
    }

    thread_local! {
        static TEST_CLIPBOARD: RefCell<TestClipboardState> = RefCell::new(TestClipboardState::default());
    }

    pub fn write(ctx: &Context, text: &str) -> bool {
        ctx.copy_text(text.to_owned());
        TEST_CLIPBOARD.with(|cell| {
            let mut st = cell.borrow_mut();
            st.os_text = Some(text.to_owned());
            st.last_internal = Some(text.to_owned());
        });
        true
    }

    pub fn read() -> Option<String> {
        TEST_CLIPBOARD.with(|cell| cell.borrow().os_text.clone())
    }

    pub fn matches_internal(text: &str) -> bool {
        TEST_CLIPBOARD.with(|cell| cell.borrow().last_internal.as_deref() == Some(text))
    }

    pub fn set_external(text: Option<&str>) {
        TEST_CLIPBOARD.with(|cell| {
            cell.borrow_mut().os_text = text.map(str::to_owned);
        });
    }

    pub fn clear_all() {
        TEST_CLIPBOARD.with(|cell| {
            *cell.borrow_mut() = TestClipboardState::default();
        });
    }
}

#[cfg(test)]
pub(crate) fn clear_test_clipboard() {
    test_clipboard::clear_all();
}

/// Dispatches all global and editor ClipMaker 3 keyboard shortcuts for the current frame,
/// scoping destructive `Delete`/`Backspace` clip deletion to [`PanelFocus::Timeline`],
/// [`PanelFocus::Viewport`], and [`PanelFocus::PlaybackToolbar`] so inspector interactions
/// cannot accidentally delete clips while transport/dropdown interactions still allow clip deletion.
pub fn dispatch_shortcuts_with_focus(
    ctx: &Context,
    session: &mut Session,
    timeline: &mut TimelineViewState,
    graph_editor: &mut GraphEditorState,
    viewport: &mut ViewportState,
    focused_panel: PanelFocus,
) -> ShortcutOutcome {
    dispatch_shortcuts_with_context(
        ctx,
        session,
        timeline,
        graph_editor,
        viewport,
        focused_panel,
        SideTab::Media,
    )
}

/// Dispatches all global and editor keyboard shortcuts with both the focused workspace panel
/// and active sidebar tab, enabling context-aware `Cmd/Ctrl+C` and `Cmd/Ctrl+V` routing
/// across Effects, 3D Objects, and Timeline clips.
pub fn dispatch_shortcuts_with_context(
    ctx: &Context,
    session: &mut Session,
    timeline: &mut TimelineViewState,
    graph_editor: &mut GraphEditorState,
    viewport: &mut ViewportState,
    focused_panel: PanelFocus,
    active_tab: SideTab,
) -> ShortcutOutcome {
    #[cfg(not(test))]
    let mut write_fn = |c: &Context, text: &str| write_system_clipboard(c, text);
    #[cfg(not(test))]
    let mut read_fn = read_system_clipboard;
    #[cfg(not(test))]
    let matches_fn = matches_last_internal_copy;

    #[cfg(test)]
    let mut write_fn = test_clipboard::write;
    #[cfg(test)]
    let mut read_fn = test_clipboard::read;
    #[cfg(test)]
    let matches_fn = test_clipboard::matches_internal;

    let mut clipboard = ClipboardAccess {
        write: &mut write_fn,
        read: &mut read_fn,
        matches_internal: &matches_fn,
    };
    dispatch_shortcuts_with_clipboard(
        ctx,
        session,
        timeline,
        graph_editor,
        viewport,
        focused_panel,
        active_tab,
        &mut clipboard,
    )
}

/// Full shortcut dispatcher with injectable clipboard I/O for deterministic unit testing.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_shortcuts_with_clipboard(
    ctx: &Context,
    session: &mut Session,
    timeline: &mut TimelineViewState,
    graph_editor: &mut GraphEditorState,
    viewport: &mut ViewportState,
    focused_panel: PanelFocus,
    active_tab: SideTab,
    clipboard: &mut ClipboardAccess<'_>,
) -> ShortcutOutcome {
    let mut outcome = ShortcutOutcome::default();
    if ctx.egui_wants_keyboard_input() || ctx.text_edit_focused() {
        return outcome;
    }

    let pressed = ctx.input(|i| {
        let mut p = PressedSnapshot::default();
        for ev in &i.events {
            match ev {
                egui::Event::Copy => {
                    p.ev_copy = true;
                }
                egui::Event::Paste(text) => {
                    if p.paste_text.is_none() {
                        p.paste_text = Some(text.clone());
                    }
                }
                egui::Event::Key {
                    key,
                    pressed: true,
                    repeat,
                    modifiers,
                    ..
                } => {
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
                        Key::V => {
                            p.v.get_or_insert(m);
                        }
                        Key::P => {
                            p.p.get_or_insert(m);
                        }
                        Key::Q => {
                            p.q.get_or_insert(m);
                        }
                        Key::W => {
                            p.w.get_or_insert(m);
                        }
                        Key::E => {
                            p.e.get_or_insert(m);
                        }
                        Key::R => {
                            p.r.get_or_insert(m);
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
                _ => {}
            }
        }
        p
    });

    // --- Escape: cancel active drag / rename or close Graph Editor ---
    // Only claim Escape when something was actually cancelled or closed AND no foreground
    // transient popover (color picker, dropdown, easing menu) is open or already consumed
    // Escape during this frame.
    if let Some(m) = pressed.escape
        && no_mods(m)
        && !super::controls::is_transient_popover_open(ctx)
        && !super::controls::transient_popover_consumed_escape(ctx)
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
    if let Some(m) = pressed.c
        && cmd_shift(m)
    {
        precompose_selected_clips(session, &mut outcome);
        return outcome;
    }
    let graph_focused = graph_editor.open && graph_editor.window_focused;
    let copy_triggered = pressed.ev_copy || pressed.c.is_some_and(cmd_only);
    if !graph_focused && copy_triggered {
        smart_copy_selection(ctx, session, timeline, active_tab, clipboard, &mut outcome);
        return outcome;
    }
    let paste_triggered = pressed.paste_text.is_some() || pressed.v.is_some_and(cmd_only);
    if !graph_focused && paste_triggered {
        smart_paste_clipboard(
            session,
            timeline,
            active_tab,
            pressed.paste_text.as_deref(),
            clipboard,
            &mut outcome,
        );
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

    // --- Viewport Mode Toggles & Pro Tools ---
    if let Some(m) = pressed.v
        && no_mods(m)
    {
        viewport.active_draw_tool = None;
        outcome.mark_handled(Key::V);
        return outcome;
    }
    if let Some(m) = pressed.p
        && no_mods(m)
    {
        viewport.active_draw_tool = Some(0);
        outcome.mark_handled(Key::P);
        return outcome;
    }
    if let Some(m) = pressed.q
        && no_mods(m)
    {
        viewport.active_draw_tool = if viewport.active_draw_tool == Some(2) {
            Some(1)
        } else {
            Some(2)
        };
        outcome.mark_handled(Key::Q);
        return outcome;
    }
    if let Some(m) = pressed.w
        && no_mods(m)
    {
        viewport.gizmo_3d_mode = Gizmo3DMode::Translate;
        outcome.mark_handled(Key::W);
        return outcome;
    }
    if let Some(m) = pressed.e
        && no_mods(m)
    {
        viewport.gizmo_3d_mode = Gizmo3DMode::Rotate;
        outcome.mark_handled(Key::E);
        return outcome;
    }
    if let Some(m) = pressed.r
        && no_mods(m)
    {
        viewport.gizmo_3d_mode = Gizmo3DMode::Scale;
        outcome.mark_handled(Key::R);
        return outcome;
    }
    if let Some(m) = pressed.m
        && no_mods(m)
    {
        let frame = session.playback.current_frame.round().max(0.0);
        let name = format!("Marker {}", session.active_sequence().markers.len() + 1);
        exec(
            session,
            Command::AddMarker {
                frame,
                name,
                color: None,
            },
            Key::M,
            &mut outcome,
        );
        return outcome;
    }
    if let Some(m) = pressed.c
        && no_mods(m)
    {
        let _ = toggle_editing_camera_synced(session);
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
            let allow_clip_delete = matches!(
                focused_panel,
                PanelFocus::Timeline | PanelFocus::Viewport | PanelFocus::PlaybackToolbar
            );
            delete_selected(session, key, allow_clip_delete, &mut outcome);
        }
        return outcome;
    }
    if let Some(m) = pressed.equals
        && (no_mods(m) || shift_only(m) || cmd_only(m) || cmd_shift(m))
    {
        timeline.zoom_at_playhead(true, session.playback.current_frame, session.document().fps);
        if m.command || m.ctrl {
            ctx.input_mut(|i| {
                let _ = i.consume_shortcut(&egui::KeyboardShortcut::new(
                    Modifiers::COMMAND,
                    Key::Equals,
                ));
                let _ =
                    i.consume_shortcut(&egui::KeyboardShortcut::new(Modifiers::CTRL, Key::Equals));
            });
        }
        outcome.mark_handled(Key::Equals);
        return outcome;
    }
    if let Some(m) = pressed.plus
        && (no_mods(m) || shift_only(m) || cmd_only(m) || cmd_shift(m))
    {
        timeline.zoom_at_playhead(true, session.playback.current_frame, session.document().fps);
        if m.command || m.ctrl {
            ctx.input_mut(|i| {
                let _ =
                    i.consume_shortcut(&egui::KeyboardShortcut::new(Modifiers::COMMAND, Key::Plus));
                let _ =
                    i.consume_shortcut(&egui::KeyboardShortcut::new(Modifiers::CTRL, Key::Plus));
            });
        }
        outcome.mark_handled(Key::Plus);
        return outcome;
    }
    if let Some(m) = pressed.minus
        && (no_mods(m) || cmd_only(m))
    {
        timeline.zoom_at_playhead(
            false,
            session.playback.current_frame,
            session.document().fps,
        );
        if m.command || m.ctrl {
            ctx.input_mut(|i| {
                let _ = i
                    .consume_shortcut(&egui::KeyboardShortcut::new(Modifiers::COMMAND, Key::Minus));
                let _ =
                    i.consume_shortcut(&egui::KeyboardShortcut::new(Modifiers::CTRL, Key::Minus));
            });
        }
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

fn delete_selected(
    session: &mut Session,
    key: Key,
    allow_clip_delete: bool,
    outcome: &mut ShortcutOutcome,
) {
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

    if !allow_clip_delete {
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

fn looks_like_pz_clipboard_payload(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.starts_with('{') || trimmed.starts_with('[') || trimmed.contains("gl_FragColor")
}

fn selected_group_object_id(
    session: &Session,
    clip_id: Option<&str>,
    object_id: Option<&str>,
) -> Option<String> {
    let clip_id = clip_id?;
    let object_id = object_id?;
    let is_group = session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == clip_id)
        .and_then(|c| c.objects_3d.iter().find(|o| o.id == object_id))
        .is_some_and(|o| o.kind.eq_ignore_ascii_case("group"));
    is_group.then(|| object_id.to_owned())
}

fn format_paste_pz_json_toast(result: &serde_json::Value) -> String {
    let kind = result.get("kind").and_then(|v| v.as_str()).unwrap_or("");
    let count = result.get("count").and_then(|v| v.as_u64()).unwrap_or(1);
    let base = match kind {
        "effects" => format!(
            "Pasted {count} {} from clipboard",
            if count == 1 { "effect" } else { "effects" }
        ),
        "objects3d" => format!(
            "Pasted {count} {} from clipboard",
            if count == 1 {
                "3D object"
            } else {
                "3D objects"
            }
        ),
        "clips" => format!(
            "Pasted {count} {} from clipboard",
            if count == 1 { "clip" } else { "clips" }
        ),
        "material" => "Pasted material from clipboard".to_owned(),
        "keyframes" => format!(
            "Pasted {count} {} from clipboard",
            if count == 1 { "keyframe" } else { "keyframes" }
        ),
        "masks" => format!(
            "Pasted {count} {} from clipboard",
            if count == 1 { "mask" } else { "masks" }
        ),
        "shape_plus" => format!(
            "Pasted {count} {} from clipboard",
            if count == 1 {
                "shape item"
            } else {
                "shape items"
            }
        ),
        _ => "Pasted item from clipboard".to_owned(),
    };

    let enabled_plugins: Vec<&str> = result
        .get("enabled_plugins")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if enabled_plugins.is_empty() {
        base
    } else {
        let label = if enabled_plugins.len() == 1 {
            "enabled plugin"
        } else {
            "enabled plugins"
        };
        format!("{base} ({label}: {})", enabled_plugins.join(", "))
    }
}

fn smart_copy_selection(
    ctx: &Context,
    session: &mut Session,
    timeline: &mut TimelineViewState,
    active_tab: SideTab,
    clipboard: &mut ClipboardAccess<'_>,
    outcome: &mut ShortcutOutcome,
) {
    prune_stale_clip_subselection(session);

    if active_tab == SideTab::Effects
        && session.selection.clip_ids.len() == 1
        && !session.selection.effect_ids.is_empty()
        && let Some(clip_id) = session.selection.clip_ids.first().cloned()
    {
        let effect_ids = session.selection.effect_ids.clone();
        let n = effect_ids.len();
        match session.copy_effects_as_pz_json(&clip_id, &effect_ids) {
            Ok(json) => {
                let _ = (clipboard.write)(ctx, &json);
                outcome.status_message = Some(format!(
                    "Copied {n} {} JSON to clipboard",
                    if n == 1 { "effect" } else { "effects" }
                ));
            }
            Err(err) => {
                outcome.last_error = Some(err.to_string());
            }
        }
        outcome.consumed_clipboard_events = true;
        outcome.mark_handled(Key::C);
        return;
    }

    if active_tab == SideTab::Objects
        && selected_clip_is_scene3d(session)
        && !session.selection.object_3d_ids.is_empty()
        && let Some(clip_id) = session.selection.clip_ids.first().cloned()
    {
        let object_ids = session.selection.object_3d_ids.clone();
        let n = object_ids.len();
        match session.copy_objects_3d_as_pz_json(&clip_id, &object_ids) {
            Ok(json) => {
                let _ = (clipboard.write)(ctx, &json);
                outcome.status_message = Some(format!(
                    "Copied {n} {} JSON to clipboard",
                    if n == 1 { "3D object" } else { "3D objects" }
                ));
            }
            Err(err) => {
                outcome.last_error = Some(err.to_string());
            }
        }
        outcome.consumed_clipboard_events = true;
        outcome.mark_handled(Key::C);
        return;
    }

    let sel_uuids: Vec<Uuid> = session
        .selection
        .clip_ids
        .iter()
        .filter_map(|id| Uuid::parse_str(id).ok())
        .collect();
    if sel_uuids.is_empty() {
        return;
    }
    match session.copy_clips(&sel_uuids) {
        Ok(snaps) => {
            let n = snaps.len();
            timeline.clip_clipboard = snaps;
            if let Ok(json) = session.copy_clips_as_pz_json(&sel_uuids) {
                let _ = (clipboard.write)(ctx, &json);
            }
            outcome.status_message = Some(format!(
                "Copied {n} {} to clipboard",
                if n == 1 { "clip" } else { "clips" }
            ));
            outcome.consumed_clipboard_events = true;
            outcome.mark_handled(Key::C);
        }
        Err(err) => {
            outcome.last_error = Some(err.to_string());
            outcome.consumed_clipboard_events = true;
            outcome.mark_handled(Key::C);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_pz_json_paste(
    session: &mut Session,
    json_text: String,
    hint: &str,
    target_clip_id: Option<String>,
    parent_object_id: Option<String>,
    target_object_id: Option<String>,
    at_frame: f64,
    outcome: &mut ShortcutOutcome,
) {
    let looks_like_payload = looks_like_pz_clipboard_payload(&json_text);
    match session.execute(Command::PastePzJson {
        json: json_text,
        hint: Some(hint.to_owned()),
        target_clip_id,
        parent_object_id,
        target_object_id,
        target_property: None,
        at_frame: Some(at_frame),
    }) {
        Ok(val) => {
            outcome.status_message = Some(format_paste_pz_json_toast(&val));
            outcome.consumed_clipboard_events = true;
            outcome.mark_handled(Key::V);
        }
        Err(err) => {
            if looks_like_payload {
                outcome.last_error = Some(format!("Paste failed: {err}"));
            }
            outcome.consumed_clipboard_events = true;
            outcome.mark_handled(Key::V);
        }
    }
}

fn smart_paste_clipboard(
    session: &mut Session,
    timeline: &TimelineViewState,
    active_tab: SideTab,
    event_paste_text: Option<&str>,
    clipboard: &mut ClipboardAccess<'_>,
    outcome: &mut ShortcutOutcome,
) {
    prune_stale_clip_subselection(session);

    let clipboard_text = (clipboard.read)()
        .filter(|s| !s.trim().is_empty() && s != CLIPBOARD_SENTINEL)
        .or_else(|| {
            event_paste_text
                .filter(|s| !s.trim().is_empty() && *s != CLIPBOARD_SENTINEL)
                .map(str::to_owned)
        });

    let target_clip_id = session.selection.clip_ids.first().cloned();
    let target_object_id = session.selection.object_3d_ids.first().cloned();
    let parent_object_id = selected_group_object_id(
        session,
        target_clip_id.as_deref(),
        target_object_id.as_deref(),
    );
    let playhead = session.playback.current_frame.round().max(0.0);

    match active_tab {
        SideTab::Effects => {
            if let Some(text) = clipboard_text {
                execute_pz_json_paste(
                    session,
                    text,
                    "effects",
                    target_clip_id,
                    parent_object_id,
                    target_object_id,
                    playhead,
                    outcome,
                );
            }
        }
        SideTab::Objects => {
            if let Some(text) = clipboard_text {
                execute_pz_json_paste(
                    session,
                    text,
                    "objects3d",
                    target_clip_id,
                    parent_object_id,
                    target_object_id,
                    playhead,
                    outcome,
                );
            }
        }
        _ => {
            let use_internal_clip_clipboard = !timeline.clip_clipboard.is_empty()
                && match clipboard_text.as_deref() {
                    None => true,
                    Some(text) => (clipboard.matches_internal)(text),
                };
            if use_internal_clip_clipboard {
                let n = timeline.clip_clipboard.len();
                match session.execute(Command::PasteClips {
                    clips: timeline.clip_clipboard.clone(),
                    target_frame: playhead,
                }) {
                    Ok(_) => {
                        outcome.status_message = Some(format!(
                            "Pasted {n} {}",
                            if n == 1 { "clip" } else { "clips" }
                        ));
                    }
                    Err(err) => {
                        outcome.last_error = Some(err.to_string());
                    }
                }
                outcome.consumed_clipboard_events = true;
                outcome.mark_handled(Key::V);
            } else if let Some(text) = clipboard_text {
                execute_pz_json_paste(
                    session,
                    text,
                    "any",
                    target_clip_id,
                    parent_object_id,
                    target_object_id,
                    playhead,
                    outcome,
                );
            }
        }
    }
}

fn precompose_selected_clips(session: &mut Session, outcome: &mut ShortcutOutcome) {
    let sel_uuids: Vec<Uuid> = session
        .selection
        .clip_ids
        .iter()
        .filter_map(|id| Uuid::parse_str(id).ok())
        .collect();
    if sel_uuids.is_empty() {
        return;
    }
    let name = format!("Precomp {}", session.project.compositions.len() + 1);
    exec(
        session,
        Command::PrecomposeClips {
            clip_ids: sel_uuids,
            name,
        },
        Key::C,
        outcome,
    );
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
    ev_copy: bool,
    paste_text: Option<String>,
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
    v: Option<Modifiers>,
    p: Option<Modifiers>,
    q: Option<Modifiers>,
    w: Option<Modifiers>,
    e: Option<Modifiers>,
    r: Option<Modifiers>,
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

    #[test]
    fn pro_shortcuts_dispatch_draw_tools_gizmo_modes_marker_and_clip_copy_paste() {
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        // P -> Pen tool (Some(0))
        let out_p = run_key_event(
            &ctx,
            Key::P,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out_p.handled_any);
        assert_eq!(viewport.active_draw_tool, Some(0));

        // Q -> Freehand/Quick Shape (Some(2)), repeat Q -> Vertex/Path Edit (Some(1))
        let _ = run_key_event(
            &ctx,
            Key::Q,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert_eq!(viewport.active_draw_tool, Some(2));
        let _ = run_key_event(
            &ctx,
            Key::Q,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert_eq!(viewport.active_draw_tool, Some(1));

        // V -> Select/Transform tool (None)
        let _ = run_key_event(
            &ctx,
            Key::V,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert_eq!(viewport.active_draw_tool, None);

        // W / E / R -> 3D gizmo modes Translate / Rotate / Scale
        let _ = run_key_event(
            &ctx,
            Key::E,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert_eq!(viewport.gizmo_3d_mode, Gizmo3DMode::Rotate);
        let _ = run_key_event(
            &ctx,
            Key::R,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert_eq!(viewport.gizmo_3d_mode, Gizmo3DMode::Scale);
        let _ = run_key_event(
            &ctx,
            Key::W,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert_eq!(viewport.gizmo_3d_mode, Gizmo3DMode::Translate);

        // M -> AddMarker at current playhead frame
        session.playback.current_frame = 42.0;
        let markers_before = session.active_sequence().markers.len();
        let out_m = run_key_event(
            &ctx,
            Key::M,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out_m.handled_any);
        assert_eq!(session.active_sequence().markers.len(), markers_before + 1);
        assert!((session.active_sequence().markers.last().unwrap().frame - 42.0).abs() < 1e-6);

        // Cmd+C / Cmd+V -> Copy and Paste clips at playhead (when Graph Editor is NOT focused)
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session.selection.select_clip(clip_id, Some("scene"), false);
        let clips_before = session.document().video_tracks[0].clips.len();

        // While Graph Editor IS open and focused, Cmd+C must NOT copy clips.
        graph.open = true;
        graph.window_focused = true;
        let out_c_blocked = run_key_event(
            &ctx,
            Key::C,
            Modifiers::COMMAND,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(!out_c_blocked.handled_any);
        assert!(timeline.clip_clipboard.is_empty());

        // Unfocus Graph Editor -> Cmd+C copies selected clip, Cmd+V pastes at playhead.
        graph.window_focused = false;
        let out_copy = run_key_event(
            &ctx,
            Key::C,
            Modifiers::COMMAND,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out_copy.handled_any);
        assert_eq!(timeline.clip_clipboard.len(), 1);

        session.playback.current_frame = 400.0;
        let out_paste = run_key_event(
            &ctx,
            Key::V,
            Modifiers::COMMAND,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out_paste.handled_any);
        assert!(out_paste.last_error.is_none());
        let total_clips_after: usize = session
            .document()
            .video_tracks
            .iter()
            .map(|t| t.clips.len())
            .sum();
        assert_eq!(total_clips_after, clips_before + 1);
    }

    #[test]
    fn space_suppressed_when_text_edit_focused_during_playback() {
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();
        let mut buf = "Hello".to_owned();
        let edit_id = egui::Id::unique("test_space_text_edit");

        session.playback.speed = 1.0;
        assert!(session.playback.is_playing());

        // Frame 1: render TextEdit and request focus.
        let mut out1 = ctx.run_ui(egui::RawInput::default(), |ui| {
            let resp = ui.add(egui::TextEdit::singleline(&mut buf).id(edit_id));
            resp.request_focus();
        });
        out1.textures_delta.clear();

        // Frame 2: press Space while TextEdit is focused -> must NOT pause playback.
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::Key {
            key: Key::Space,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Modifiers::NONE,
        });
        let mut outcome = ShortcutOutcome::default();
        let mut out2 = ctx.run_ui(raw, |ui| {
            outcome = dispatch_shortcuts_with_focus(
                ui.ctx(),
                &mut session,
                &mut timeline,
                &mut graph,
                &mut viewport,
                PanelFocus::SideInspector,
            );
            let _ = ui.add(egui::TextEdit::singleline(&mut buf).id(edit_id));
        });
        out2.textures_delta.clear();

        assert!(!outcome.handled_any);
        assert!(
            session.playback.is_playing(),
            "Space inside focused text input must never pause playback"
        );
    }

    #[test]
    fn delete_and_backspace_in_inspector_do_not_delete_scene_clip() {
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session.selection.select_clip(clip_id, Some("scene"), false);

        for key in [Key::Delete, Key::Backspace] {
            let mut raw = egui::RawInput::default();
            raw.events.push(egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::NONE,
            });
            let mut outcome = ShortcutOutcome::default();
            let mut full = ctx.run_ui(raw, |ui| {
                outcome = dispatch_shortcuts_with_focus(
                    ui.ctx(),
                    &mut session,
                    &mut timeline,
                    &mut graph,
                    &mut viewport,
                    PanelFocus::SideInspector,
                );
            });
            full.textures_delta.clear();
            assert!(!outcome.handled_any);
            assert_eq!(
                session.document().video_tracks[0].clips.len(),
                1,
                "pressing {key:?} while SideInspector has focus must NOT delete the Scene clip"
            );
        }

        // Pressing Delete while Timeline has focus DOES delete the selected clip.
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::Key {
            key: Key::Delete,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Modifiers::NONE,
        });
        let mut outcome = ShortcutOutcome::default();
        let mut full = ctx.run_ui(raw, |ui| {
            outcome = dispatch_shortcuts_with_focus(
                ui.ctx(),
                &mut session,
                &mut timeline,
                &mut graph,
                &mut viewport,
                PanelFocus::Timeline,
            );
        });
        full.textures_delta.clear();
        assert!(outcome.handled_any);
        assert!(session.document().video_tracks[0].clips.is_empty());
    }

    #[test]
    fn cmd_equals_and_cmd_minus_zoom_timeline() {
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        let z0 = timeline.zoom;
        let out_in = run_key_event(
            &ctx,
            Key::Equals,
            Modifiers::COMMAND,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out_in.handled_any);
        assert!(timeline.zoom > z0, "Cmd+= must zoom timeline in");

        let z1 = timeline.zoom;
        let out_out = run_key_event(
            &ctx,
            Key::Minus,
            Modifiers::COMMAND,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out_out.handled_any);
        assert!(timeline.zoom < z1, "Cmd+- must zoom timeline out");
    }

    #[test]
    fn editing_camera_toggle_auto_selects_scene_clip_when_unselected() {
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        session.selection.clear_all();
        session.playback.editing_camera = false;

        let out = run_key_event(
            &ctx,
            Key::C,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out.handled_any);
        assert!(session.playback.editing_camera);
        assert!(
            selected_clip_is_scene3d(&session),
            "toggling Editing Camera ON with no clip selected must auto-select an active Scene clip"
        );

        // Second press turns Editing Camera back OFF cleanly.
        let out2 = run_key_event(
            &ctx,
            Key::C,
            Modifiers::NONE,
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
        );
        assert!(out2.handled_any);
        assert!(!session.playback.editing_camera);
    }

    #[allow(clippy::too_many_arguments)]
    fn run_events_with_context(
        ctx: &Context,
        events: Vec<egui::Event>,
        session: &mut Session,
        timeline: &mut TimelineViewState,
        graph: &mut GraphEditorState,
        viewport: &mut ViewportState,
        focused_panel: PanelFocus,
        active_tab: SideTab,
    ) -> ShortcutOutcome {
        let raw = egui::RawInput {
            events,
            ..Default::default()
        };
        let mut outcome = ShortcutOutcome::default();
        let mut full = ctx.run_ui(raw, |ui| {
            outcome = dispatch_shortcuts_with_context(
                ui.ctx(),
                session,
                timeline,
                graph,
                viewport,
                focused_panel,
                active_tab,
            );
        });
        full.textures_delta.clear();
        outcome
    }

    #[test]
    fn smart_clipboard_effects_copy_paste_shader_json_and_deduplicates_same_frame() {
        test_clipboard::clear_all();
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session
            .selection
            .select_clip(clip_id.clone(), Some("scene"), false);
        let initial_fx = session.document().video_tracks[0].clips[0].effects.len();

        // 1. External CM3 Custom Shader JSON pasted via BOTH Event::Paste AND Cmd+V in the same
        //    frame must paste exactly ONCE (zero double-paste).
        let shader_json = r#"{
            "type": "shader",
            "properties": {
                "name": "Prism Test",
                "fragShader": "uniform sampler2D tDiffuse; uniform float Gain; varying vec2 vUv; void main() { gl_FragColor = vec4(vec3(Gain), 1.0); }"
            },
            "customProperties": [
                { "name": "Gain", "type": 0, "animated": false, "keyframes": [{ "frame": 0, "value": 1.25, "tween": 1 }] }
            ]
        }"#;
        let out_paste = run_events_with_context(
            &ctx,
            vec![
                egui::Event::Paste(shader_json.to_owned()),
                egui::Event::Key {
                    key: Key::V,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Modifiers::COMMAND,
                },
            ],
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
            PanelFocus::SideInspector,
            SideTab::Effects,
        );
        assert!(out_paste.handled_any);
        assert!(out_paste.consumed_clipboard_events);
        assert!(out_paste.last_error.is_none());
        assert_eq!(
            out_paste.status_message.as_deref(),
            Some("Pasted 1 effect from clipboard")
        );
        assert_eq!(
            session.document().video_tracks[0].clips[0].effects.len(),
            initial_fx + 1,
            "Event::Paste + Cmd+V in the same frame must deduplicate to a single paste"
        );

        // 2. Smart Cmd+C on Effects tab copies selected effect JSON to clipboard.
        let out_copy = run_events_with_context(
            &ctx,
            vec![egui::Event::Key {
                key: Key::C,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::COMMAND,
            }],
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
            PanelFocus::SideInspector,
            SideTab::Effects,
        );
        assert!(out_copy.handled_any);
        assert!(out_copy.consumed_clipboard_events);
        assert_eq!(
            out_copy.status_message.as_deref(),
            Some("Copied 1 effect JSON to clipboard")
        );
        let copied_json = test_clipboard::read().expect("copied effect json");
        assert!(copied_json.contains("Prism Test"));

        // 3. Cmd+V WITHOUT Event::Paste (macOS eframe fallback path) reads OS clipboard & pastes!
        let out_paste2 = run_events_with_context(
            &ctx,
            vec![egui::Event::Key {
                key: Key::V,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::COMMAND,
            }],
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
            PanelFocus::SideInspector,
            SideTab::Effects,
        );
        assert!(out_paste2.handled_any);
        assert!(out_paste2.last_error.is_none());
        assert_eq!(
            session.document().video_tracks[0].clips[0].effects.len(),
            initial_fx + 2
        );

        // 4. Plain non-JSON clipboard text fails silently (no error toast), while broken JSON
        //    produces a concise error toast.
        test_clipboard::set_external(Some("plain hello text"));
        let out_plain = run_events_with_context(
            &ctx,
            vec![egui::Event::Key {
                key: Key::V,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::COMMAND,
            }],
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
            PanelFocus::SideInspector,
            SideTab::Effects,
        );
        assert!(out_plain.last_error.is_none());

        test_clipboard::set_external(Some("{broken json"));
        let out_bad_json = run_events_with_context(
            &ctx,
            vec![egui::Event::Key {
                key: Key::V,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::COMMAND,
            }],
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
            PanelFocus::SideInspector,
            SideTab::Effects,
        );
        assert!(
            out_bad_json
                .last_error
                .as_deref()
                .is_some_and(|e| e.starts_with("Paste failed:")),
            "malformed JSON paste must surface a concise error toast"
        );
        test_clipboard::clear_all();
    }

    #[test]
    fn smart_clipboard_objects3d_and_timeline_external_override() {
        test_clipboard::clear_all();
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let obj_id = session.document().video_tracks[0].clips[0].objects_3d[0]
            .id
            .clone();
        let initial_objs = session.document().video_tracks[0].clips[0].objects_3d.len();
        let _ = session
            .selection
            .select_clip(clip_id.clone(), Some("scene"), false);
        let _ = session
            .selection
            .select_object_3d(obj_id.clone(), "text", false);

        // 1. Copy 3D object on Objects tab -> writes JSON to clipboard & pastes back via Cmd+V.
        let out_copy_obj = run_events_with_context(
            &ctx,
            vec![egui::Event::Key {
                key: Key::C,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::COMMAND,
            }],
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
            PanelFocus::SideInspector,
            SideTab::Objects,
        );
        assert_eq!(
            out_copy_obj.status_message.as_deref(),
            Some("Copied 1 3D object JSON to clipboard")
        );
        let obj_json = test_clipboard::read().expect("copied 3d object json");
        assert!(obj_json.trim().starts_with('{'));

        let out_paste_obj = run_events_with_context(
            &ctx,
            vec![egui::Event::Key {
                key: Key::V,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::COMMAND,
            }],
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
            PanelFocus::SideInspector,
            SideTab::Objects,
        );
        assert!(out_paste_obj.last_error.is_none());
        assert_eq!(
            out_paste_obj.status_message.as_deref(),
            Some("Pasted 1 3D object from clipboard")
        );
        assert_eq!(
            session.document().video_tracks[0].clips[0].objects_3d.len(),
            initial_objs + 1
        );

        // 2. Copy clip on Timeline (Media tab) populates BOTH internal clip_clipboard AND system JSON.
        let out_copy_clip = run_events_with_context(
            &ctx,
            vec![egui::Event::Key {
                key: Key::C,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::COMMAND,
            }],
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
            PanelFocus::Timeline,
            SideTab::Media,
        );
        assert_eq!(timeline.clip_clipboard.len(), 1);
        assert!(out_copy_clip.handled_any);
        let clip_json = test_clipboard::read().expect("copied clip json");
        assert!(clip_json.contains("\"layer\"") || clip_json.contains("\"start\""));

        // 3. When external clipboard now holds an Effect JSON (not matching our internal copy),
        //    Timeline Cmd+V routes to PastePzJson("any") and attaches the effect instead of pasting the old clip!
        let fx_before = session.document().video_tracks[0].clips[0].effects.len();
        test_clipboard::set_external(Some(
            r#"{"type":"bloom","properties":{"name":"External Bloom"}}"#,
        ));
        let out_ext_paste = run_events_with_context(
            &ctx,
            vec![egui::Event::Key {
                key: Key::V,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::COMMAND,
            }],
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
            PanelFocus::Timeline,
            SideTab::Media,
        );
        assert!(out_ext_paste.last_error.is_none());
        assert_eq!(
            out_ext_paste.status_message.as_deref(),
            Some("Pasted 1 effect from clipboard")
        );
        assert_eq!(
            session.document().video_tracks[0].clips[0].effects.len(),
            fx_before + 1
        );
        test_clipboard::clear_all();
    }

    #[test]
    fn live_os_clipboard_overrides_stale_event_paste_text_and_sentinel() {
        test_clipboard::clear_all();
        let ctx = Context::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();

        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let _ = session.selection.select_clip(clip_id, Some("scene"), false);
        let initial_fx = session.document().video_tracks[0].clips[0].effects.len();

        // Live OS clipboard holds a fresh external Bloom effect while egui's Event::Paste
        // carries a stale rgbshift JSON from an earlier in-app copy.
        test_clipboard::set_external(Some(
            r#"{"type":"bloom","properties":{"name":"Live OS Bloom"}}"#,
        ));
        let out = run_events_with_context(
            &ctx,
            vec![egui::Event::Paste(
                r#"{"type":"rgbshift","properties":{"name":"Stale Egui Effect"}}"#.to_owned(),
            )],
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
            PanelFocus::SideInspector,
            SideTab::Effects,
        );
        assert!(out.handled_any);
        assert!(out.last_error.is_none());
        let effects = &session.document().video_tracks[0].clips[0].effects;
        assert_eq!(effects.len(), initial_fx + 1);
        assert_eq!(
            effects.last().map(|e| e.name.as_str()),
            Some("Live OS Bloom"),
            "live OS clipboard must take priority over stale Event::Paste payload"
        );

        // Fresh launch: Event::Paste(CLIPBOARD_SENTINEL) with live OS clipboard pastes the OS JSON.
        test_clipboard::set_external(Some(
            r#"{"type":"rgbshift","properties":{"name":"Fresh Launch Shift"}}"#,
        ));
        let out2 = run_events_with_context(
            &ctx,
            vec![egui::Event::Paste(CLIPBOARD_SENTINEL.to_owned())],
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
            PanelFocus::SideInspector,
            SideTab::Effects,
        );
        assert!(out2.handled_any);
        assert!(out2.last_error.is_none());
        let effects = &session.document().video_tracks[0].clips[0].effects;
        assert_eq!(effects.len(), initial_fx + 2);
        assert_eq!(
            effects.last().map(|e| e.name.as_str()),
            Some("Fresh Launch Shift")
        );

        // Fresh launch with empty OS clipboard + sentinel does nothing silently.
        test_clipboard::clear_all();
        let out3 = run_events_with_context(
            &ctx,
            vec![egui::Event::Paste(CLIPBOARD_SENTINEL.to_owned())],
            &mut session,
            &mut timeline,
            &mut graph,
            &mut viewport,
            PanelFocus::SideInspector,
            SideTab::Effects,
        );
        assert!(out3.last_error.is_none());
        assert_eq!(
            session.document().video_tracks[0].clips[0].effects.len(),
            initial_fx + 2
        );
    }

    #[test]
    fn delete_and_backspace_work_after_playback_toolbar_interaction() {
        for key in [Key::Delete, Key::Backspace] {
            let ctx = Context::default();
            let mut session = Session::new();
            let mut timeline = TimelineViewState::new();
            let mut graph = GraphEditorState::new();
            let mut viewport = ViewportState::new();

            let clip_id = session.document().video_tracks[0].clips[0].id.clone();
            let _ = session.selection.select_clip(clip_id, Some("scene"), false);
            assert_eq!(session.document().video_tracks[0].clips.len(), 1);

            let out = run_events_with_context(
                &ctx,
                vec![egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Modifiers::NONE,
                }],
                &mut session,
                &mut timeline,
                &mut graph,
                &mut viewport,
                PanelFocus::PlaybackToolbar,
                SideTab::Media,
            );
            assert!(
                out.handled_any,
                "pressing {key:?} with PlaybackToolbar focus must delete selected timeline clip"
            );
            assert!(
                session.document().video_tracks[0].clips.is_empty(),
                "selected clip must be deleted when pressing {key:?} after playback toolbar interaction"
            );
        }
    }

    #[test]
    fn escape_closes_open_color_picker_before_graph_editor() {
        use crate::ui::controls::{
            FloatingWindowManager, FloatingWindowSpec, cm3_floating_window, cm3_hsva_color_picker,
            is_transient_popover_open, transient_popover_consumed_escape,
        };
        use crate::ui::theme::Cm3Theme;
        use egui::{Id, Pos2, Rect, Vec2};
        use zc_color::Rgba;

        let ctx = Context::default();
        let theme = Cm3Theme::default();
        let mut session = Session::new();
        let mut timeline = TimelineViewState::new();
        let mut graph = GraphEditorState::new();
        let mut viewport = ViewportState::new();
        let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));
        let picker_id = Id::unique("test_esc_color_picker");
        let mut color = Rgba::new(1.0, 1.0, 1.0, 1.0);

        // Open Graph Editor and a floating window underneath.
        graph.open = true;
        graph.window_focused = true;
        let mut fw_mgr = FloatingWindowManager::default();
        let fw_spec = FloatingWindowSpec::new("expr_editor", "Expression Editor");
        fw_mgr.open_or_focus(&fw_spec, vp);

        let run_pass = |events: Vec<egui::Event>, body: &mut dyn FnMut(&mut egui::Ui)| {
            let input = egui::RawInput {
                screen_rect: Some(vp),
                events,
                ..Default::default()
            };
            let mut out = ctx.run_ui(input, |ui| {
                crate::ui::controls::reset_transient_popover_escape_flag(ui.ctx());
                egui::CentralPanel::default().show(ui, |ui| body(ui));
            });
            out.textures_delta.clear();
        };

        // Frame 1: Render closed color picker to obtain swatch position.
        let mut swatch_center = Pos2::ZERO;
        run_pass(Vec::new(), &mut |ui| {
            let start = ui.cursor().min;
            let _ = cm3_hsva_color_picker(ui, picker_id, &mut color, false, false, vp, &theme);
            swatch_center = Pos2::new(start.x + 15.0, start.y + 12.0);
        });

        // Frame 2: Click the swatch to open the Color Picker popover.
        let mut open_ev = crate::ui::controls::ColorPickerEvent::default();
        run_pass(
            vec![
                egui::Event::PointerMoved(swatch_center),
                egui::Event::PointerButton {
                    pos: swatch_center,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Modifiers::NONE,
                },
                egui::Event::PointerButton {
                    pos: swatch_center,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Modifiers::NONE,
                },
            ],
            &mut |ui| {
                open_ev =
                    cm3_hsva_color_picker(ui, picker_id, &mut color, false, false, vp, &theme);
            },
        );
        assert!(
            open_ev.started,
            "color picker popover must open on swatch click"
        );
        assert!(
            is_transient_popover_open(&ctx),
            "open color picker must mark transient popover open"
        );

        // Frame 3: Press Escape. In main.rs, dispatch_shortcuts runs BEFORE inspector widgets
        // and floating windows. Escape must dismiss ONLY the Color Picker, leaving BOTH
        // Graph Editor and the floating window open!
        let mut shortcut_out = ShortcutOutcome::default();
        let mut cancel_ev = crate::ui::controls::ColorPickerEvent::default();
        let mut fw_ev = crate::ui::controls::FloatingWindowEvent::default();
        let mut consumed_flag = false;
        run_pass(
            vec![egui::Event::Key {
                key: Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::NONE,
            }],
            &mut |ui| {
                shortcut_out = dispatch_shortcuts(
                    ui.ctx(),
                    &mut session,
                    &mut timeline,
                    &mut graph,
                    &mut viewport,
                );
                cancel_ev =
                    cm3_hsva_color_picker(ui, picker_id, &mut color, false, false, vp, &theme);
                consumed_flag = transient_popover_consumed_escape(ui.ctx());
                fw_ev = cm3_floating_window(ui.ctx(), &mut fw_mgr, &fw_spec, vp, &theme, |_| {});
            },
        );
        assert!(
            !shortcut_out.cancel_or_close,
            "dispatch_shortcuts must yield Escape while Color Picker popover is open"
        );
        assert!(
            cancel_ev.cancelled,
            "Color Picker popover must handle and cancel on Escape"
        );
        assert!(
            consumed_flag,
            "Color Picker popover must mark Escape consumed for the current pass"
        );
        assert!(
            graph.open,
            "Graph Editor must stay OPEN when Escape dismisses the foreground Color Picker"
        );
        assert!(
            !fw_ev.closed && fw_mgr.is_open("expr_editor"),
            "Floating window underneath must stay OPEN when Escape dismisses Color Picker"
        );

        // Frame 4: Press Escape again (no popover open) -> closes Graph Editor cleanly.
        let mut shortcut_out2 = ShortcutOutcome::default();
        run_pass(
            vec![egui::Event::Key {
                key: Key::Escape,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: Modifiers::NONE,
            }],
            &mut |ui| {
                shortcut_out2 = dispatch_shortcuts(
                    ui.ctx(),
                    &mut session,
                    &mut timeline,
                    &mut graph,
                    &mut viewport,
                );
            },
        );
        assert!(
            shortcut_out2.cancel_or_close && !graph.open,
            "Second Escape after Color Picker closed must dismiss Graph Editor"
        );
    }
}
