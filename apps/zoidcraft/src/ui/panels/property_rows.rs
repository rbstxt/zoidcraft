use super::effects_inspector::oversized_text_summary;
use super::shared_widgets::{
    eval_session_expr_f64, eval_session_expr_vec2, eval_session_expr_vec3, eval_session_expr_vec4,
};
use super::state::{
    ExprEditorTarget, StopwatchContextMenuState, begin_inspector_tx, clear_inspector_tx_state,
    close_existing_tx_before_start, commit_inspector_tx_if_open, owns_inspector_tx,
    run_atomic_commit_outside_foreign_tx, touch_inspector_tx_owner,
};
use super::*;

// ---------------------------------------------------------------------------
// Shared Property Binding Helpers (Stopwatch + Keyframe Bar + Session Execute)
// ---------------------------------------------------------------------------

fn find_clip_ref<'a>(session: &'a Session, clip_id: &str) -> Option<&'a SessionClip> {
    for track in session
        .document()
        .video_tracks
        .iter()
        .chain(session.document().audio_tracks.iter())
    {
        if let Some(clip) = track.clips.iter().find(|c| c.id == clip_id) {
            return Some(clip);
        }
    }
    None
}

pub(super) fn find_clip_cached(
    session: &Session,
    state: &mut PanelsState,
    clip_id: &str,
) -> Option<std::sync::Arc<SessionClip>> {
    let rev = session.revision();
    if let Some((cached_rev, cached_id, cached_clip)) = &state.cached_selected_clip
        && *cached_rev == rev
        && cached_id == clip_id
    {
        return Some(std::sync::Arc::clone(cached_clip));
    }
    let clip = std::sync::Arc::new(find_clip_ref(session, clip_id)?.clone());
    state.cached_selected_clip = Some((rev, clip_id.to_owned(), std::sync::Arc::clone(&clip)));
    Some(clip)
}

#[cfg(test)]
pub(super) fn find_clip_clone(session: &Session, clip_id: &str) -> Option<SessionClip> {
    find_clip_ref(session, clip_id).cloned()
}

pub(super) fn lookup_session_prop_ref<'a>(
    session: &'a Session,
    target_id: &str,
    property: &str,
) -> Option<&'a SessionProperty> {
    if target_id == "sequence" {
        return session.document().sequence_properties.get(property);
    }
    for track in session
        .document()
        .video_tracks
        .iter()
        .chain(session.document().audio_tracks.iter())
    {
        for clip in &track.clips {
            if clip.id == target_id {
                return clip.properties.get(property);
            }
            if let Some(obj) = clip.objects_3d.iter().find(|o| o.id == target_id) {
                return obj.properties.get(property);
            }
            if let Some(fx) = clip.effects.iter().find(|e| e.id == target_id) {
                return fx.properties.get(property);
            }
        }
    }
    None
}

#[cfg(test)]
pub(super) fn lookup_session_prop(
    session: &Session,
    target_id: &str,
    property: &str,
) -> Option<SessionProperty> {
    lookup_session_prop_ref(session, target_id, property).cloned()
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

pub(super) fn eval_prop_value(prop: Option<&SessionProperty>, local_frame: f64) -> Option<Value> {
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

pub(super) fn eval_prop_f64(
    prop: Option<&SessionProperty>,
    local_frame: f64,
    fallback: f64,
) -> f64 {
    eval_prop_value(prop, local_frame)
        .and_then(|v| v.as_f64())
        .unwrap_or(fallback)
}

pub(super) fn eval_prop_u32(
    prop: Option<&SessionProperty>,
    local_frame: f64,
    fallback: u32,
) -> u32 {
    eval_prop_value(prop, local_frame)
        .and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_f64().map(|f| f.round().max(0.0) as u64))
                .or_else(|| v.as_bool().map(u64::from))
        })
        .map_or(fallback, |v| v as u32)
}

pub(super) fn eval_prop_vec2(
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

pub(super) fn eval_prop_vec3(
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

pub(super) fn eval_prop_str(
    prop: Option<&SessionProperty>,
    local_frame: f64,
    fallback: &str,
) -> String {
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

pub(super) fn normalized_kf_frame(session: &Session, local_frame: f64) -> f64 {
    if session.playback.speed.abs() > 1e-6 {
        local_frame
    } else {
        local_frame.round()
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn panel_property_row<R>(
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

pub(super) fn resolve_kf_write_frame(
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
pub(super) fn apply_kf_bar_event(
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

#[allow(clippy::too_many_arguments)]
fn apply_kf_bar_event_live(
    session: &mut Session,
    target_id: &str,
    prop_name: &str,
    local_frame: f64,
    clip_start: f64,
    current_val: Value,
    ev: KeyframeBarEvent,
    actions: &mut Vec<PanelAction>,
) {
    if !ev.prev_kf
        && !ev.next_kf
        && !ev.toggle_kf
        && !ev.toggle_interp
        && ev.set_easing.is_none()
        && !ev.open_easing_plus
    {
        return;
    }
    let prop_opt = lookup_session_prop_ref(session, target_id, prop_name).cloned();
    apply_kf_bar_event(
        session,
        target_id,
        prop_name,
        prop_opt.as_ref(),
        local_frame,
        clip_start,
        current_val,
        ev,
        actions,
    );
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
        let frame = resolve_kf_write_frame(
            session,
            lookup_session_prop_ref(session, target_id, prop_name),
            local_frame,
        );
        Command::SetKeyframe {
            target_id: target_id.to_owned(),
            property: prop_name.to_owned(),
            frame,
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
pub(super) fn apply_scrub_to_session(
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
pub(super) fn apply_scrub_to_session_with_owner_key(
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
pub(super) fn apply_color_picker_to_session(
    session: &mut Session,
    target_id: &str,
    prop_name: &str,
    animated: bool,
    local_frame: f64,
    value: Value,
    cev: super::super::controls::ColorPickerEvent,
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
    session: &Session,
    prop: Option<&SessionProperty>,
    theme: &Cm3Theme,
) -> bool {
    let mut open_expr = false;
    if let Some(expr) = prop.and_then(|p| p.expression.as_deref()) {
        let mode = if session.plugins.is_enabled("expressions-plus") {
            zc_expr::ExprMode::ExpressionsPlus
        } else {
            zc_expr::ExprMode::VanillaCm3
        };
        let compile_err = zc_expr::compile(expr.trim(), mode)
            .err()
            .map(|e| e.to_string());
        let is_err = compile_err.is_some();
        let icon_color = if is_err { DANGER_TEXT } else { theme.value() };
        let label_color = if is_err { DANGER_TEXT } else { TEXT_MUTED };
        let tip = match &compile_err {
            Some(err) => format!("Expression error: {err} (click to edit)"),
            None => "Edit expression".to_owned(),
        };
        ui.horizontal(|ui| {
            if cm3_icon_button(ui, Cm3Icon::Code, 17.0, icon_color, &tip, theme) {
                open_expr = true;
            }
            let lbl_resp = ui.add(
                egui::Label::new(
                    egui::RichText::new(expr)
                        .font(theme.font(FONT_COMPACT))
                        .color(label_color),
                )
                .truncate(),
            );
            if let Some(err) = &compile_err {
                lbl_resp.on_hover_text(format!("Expression error: {err}"));
            }
        });
    }
    open_expr
}

fn stopwatch_button_with_right_click(
    ui: &mut Ui,
    prop_key: &str,
    animated: bool,
    has_expr: bool,
    theme: &Cm3Theme,
) -> (super::super::controls::StopwatchEvent, Option<Pos2>) {
    let sw_min = ui.cursor().min;
    let ev = cm3_stopwatch_button(ui, prop_key, animated, has_expr, theme);
    let sw_rect = Rect::from_min_size(sw_min, Vec2::splat(21.0));
    let ctx_pos = ui.input(|i| {
        if i.pointer.secondary_clicked() {
            i.pointer.interact_pos().filter(|p| sw_rect.contains(*p))
        } else {
            None
        }
    });
    (ev, ctx_pos)
}

#[allow(clippy::too_many_arguments)]
fn maybe_open_stopwatch_context_menu(
    state: &mut PanelsState,
    target_id: &str,
    prop_key: &str,
    existing_expr: Option<&str>,
    local_frame: f64,
    current_val: Value,
    ctx_pos: Option<Pos2>,
) {
    if let Some(pos) = ctx_pos {
        state.stopwatch_ctx_menu = Some(StopwatchContextMenuState {
            target_id: target_id.to_owned(),
            property: prop_key.to_owned(),
            existing_expr: existing_expr.map(str::to_owned),
            local_frame,
            current_val,
            pos,
        });
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn show_session_number_prop(
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
    let prop = lookup_session_prop_ref(session, target_id, prop_key);
    let animated = prop.is_some_and(|p| p.animated);
    let existing_expr = prop.and_then(|p| p.expression.clone());
    let has_expr = existing_expr.is_some();
    // Clip `time` initializes as a non-animated 2-keyframe linear ramp [0 -> start, length -> end].
    let base_val = if prop_key == "time"
        && let Some(p) = prop
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
        eval_prop_f64(prop, local_frame, default_val)
    };
    let mut val = if let Some(expr_code) = existing_expr.as_deref() {
        eval_session_expr_f64(session, state, target_id, prop_key, expr_code, base_val)
            .unwrap_or(base_val)
    } else {
        base_val
    };
    let easing_plus = session.plugins.is_enabled("easing-plus");

    let mut sw_ev = Default::default();
    let mut sw_ctx_pos = None;
    let mut kf_ev = KeyframeBarEvent::default();
    let mut open_expr = false;

    let scrub_ev = panel_property_row(
        ui,
        label,
        false,
        state,
        theme,
        |ui| {
            (sw_ev, sw_ctx_pos) =
                stopwatch_button_with_right_click(ui, prop_key, animated, has_expr, theme);
        },
        |ui| {
            if animated && !has_expr {
                let bar_state =
                    build_kf_bar_state(prop, local_frame, true, easing_plus, viewport_rect);
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
            if has_expr && show_expression_inline_chip(ui, session, prop, theme) {
                open_expr = true;
            }
            ev
        },
    );

    maybe_open_stopwatch_context_menu(
        state,
        target_id,
        prop_key,
        existing_expr.as_deref(),
        local_frame,
        json!(base_val),
        sw_ctx_pos,
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
    apply_kf_bar_event_live(
        session,
        target_id,
        prop_key,
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
pub(super) fn show_session_vec2_prop(
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
    let prop = lookup_session_prop_ref(session, target_id, prop_key);
    let animated = prop.is_some_and(|p| p.animated);
    let existing_expr = prop.and_then(|p| p.expression.clone());
    let has_expr = existing_expr.is_some();
    let base_val = eval_prop_vec2(prop, local_frame, default_val);
    let mut val = if let Some(expr_code) = existing_expr.as_deref() {
        eval_session_expr_vec2(session, state, target_id, prop_key, expr_code, base_val)
            .unwrap_or(base_val)
    } else {
        base_val
    };
    let easing_plus = session.plugins.is_enabled("easing-plus");
    let expand_key = (target_id.to_owned(), prop_key.to_owned());
    let mut expanded = state.expanded_vec_props.contains(&expand_key);

    let mut sw_ev = Default::default();
    let mut sw_ctx_pos = None;
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
            (sw_ev, sw_ctx_pos) =
                stopwatch_button_with_right_click(ui, prop_key, animated, has_expr, theme);
        },
        |ui| {
            if animated && !has_expr {
                let bar_state =
                    build_kf_bar_state(prop, local_frame, true, easing_plus, viewport_rect);
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
            if has_expr && show_expression_inline_chip(ui, session, prop, theme) {
                open_expr = true;
            }
            ev
        },
    );

    maybe_open_stopwatch_context_menu(
        state,
        target_id,
        prop_key,
        existing_expr.as_deref(),
        local_frame,
        json!(base_val),
        sw_ctx_pos,
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
    apply_kf_bar_event_live(
        session,
        target_id,
        prop_key,
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
pub(super) fn show_session_vec3_prop(
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
    let prop = lookup_session_prop_ref(session, target_id, prop_key);
    let animated = prop.is_some_and(|p| p.animated);
    let existing_expr = prop.and_then(|p| p.expression.clone());
    let has_expr = existing_expr.is_some();
    let base_val = eval_prop_vec3(prop, local_frame, default_val);
    let mut val = if let Some(expr_code) = existing_expr.as_deref() {
        eval_session_expr_vec3(session, state, target_id, prop_key, expr_code, base_val)
            .unwrap_or(base_val)
    } else {
        base_val
    };
    let easing_plus = session.plugins.is_enabled("easing-plus");
    let expand_key = (target_id.to_owned(), prop_key.to_owned());
    let mut expanded = state.expanded_vec_props.contains(&expand_key);

    let mut sw_ev = Default::default();
    let mut sw_ctx_pos = None;
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
            (sw_ev, sw_ctx_pos) =
                stopwatch_button_with_right_click(ui, prop_key, animated, has_expr, theme);
        },
        |ui| {
            if animated && !has_expr {
                let bar_state =
                    build_kf_bar_state(prop, local_frame, true, easing_plus, viewport_rect);
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
            if has_expr && show_expression_inline_chip(ui, session, prop, theme) {
                open_expr = true;
            }
            ev
        },
    );

    maybe_open_stopwatch_context_menu(
        state,
        target_id,
        prop_key,
        existing_expr.as_deref(),
        local_frame,
        json!(base_val),
        sw_ctx_pos,
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
    apply_kf_bar_event_live(
        session,
        target_id,
        prop_key,
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
pub(super) fn show_session_vec4_prop(
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
pub(super) fn show_session_vec4_prop_ex(
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
    let prop = lookup_session_prop_ref(session, target_id, prop_key);
    let animated = animatable && prop.is_some_and(|p| p.animated);
    let existing_expr = prop.and_then(|p| p.expression.clone());
    let has_expr = animatable && existing_expr.is_some();
    let base_val = eval_prop_vec4(prop, local_frame, default_val);
    let mut val = if has_expr && let Some(expr_code) = existing_expr.as_deref() {
        eval_session_expr_vec4(session, state, target_id, prop_key, expr_code, base_val)
            .unwrap_or(base_val)
    } else {
        base_val
    };
    let easing_plus = session.plugins.is_enabled("easing-plus");

    let mut sw_ev = Default::default();
    let mut sw_ctx_pos = None;
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
                (sw_ev, sw_ctx_pos) =
                    stopwatch_button_with_right_click(ui, prop_key, animated, has_expr, theme);
            }
        },
        |ui| {
            if animated && !has_expr {
                let bar_state =
                    build_kf_bar_state(prop, local_frame, true, easing_plus, viewport_rect);
                kf_ev = cm3_keyframe_bar(
                    ui,
                    Id::unique((target_id, prop_key, "kf")),
                    bar_state,
                    theme,
                );
            }
            let cfg = config.with_read_only(config.read_only || has_expr);
            let total = super::super::controls::cm3_vector4(
                ui,
                Id::unique((target_id, prop_key, "vec4")),
                &mut val,
                cfg,
                theme,
            );
            if has_expr && show_expression_inline_chip(ui, session, prop, theme) {
                open_expr = true;
            }
            total
        },
    );

    if animatable {
        maybe_open_stopwatch_context_menu(
            state,
            target_id,
            prop_key,
            existing_expr.as_deref(),
            local_frame,
            json!(base_val),
            sw_ctx_pos,
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
                existing_expr.unwrap_or_else(|| json!(val).to_string()),
                state,
                viewport_rect,
            );
        }
        apply_kf_bar_event_live(
            session,
            target_id,
            prop_key,
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
pub(super) fn show_session_color_prop(
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
    let prop = lookup_session_prop_ref(session, target_id, prop_key);
    let animated = prop.is_some_and(|p| p.animated);
    let existing_expr = prop.and_then(|p| p.expression.clone());
    let has_expr = existing_expr.is_some();
    let base_rgb = eval_prop_vec3(prop, local_frame, default_rgb);
    let rgb_f64 = if let Some(expr_code) = existing_expr.as_deref() {
        eval_session_expr_vec3(session, state, target_id, prop_key, expr_code, base_rgb)
            .unwrap_or(base_rgb)
    } else {
        base_rgb
    };
    let mut rgba = Rgba::new(rgb_f64[0] as f32, rgb_f64[1] as f32, rgb_f64[2] as f32, 1.0);
    let easing_plus = session.plugins.is_enabled("easing-plus");

    let mut sw_ev = Default::default();
    let mut sw_ctx_pos = None;
    let mut kf_ev = KeyframeBarEvent::default();
    let mut open_expr = false;

    let cev = panel_property_row(
        ui,
        label,
        false,
        state,
        theme,
        |ui| {
            (sw_ev, sw_ctx_pos) =
                stopwatch_button_with_right_click(ui, prop_key, animated, has_expr, theme);
        },
        |ui| {
            if animated && !has_expr {
                let bar_state =
                    build_kf_bar_state(prop, local_frame, true, easing_plus, viewport_rect);
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
            if has_expr && show_expression_inline_chip(ui, session, prop, theme) {
                open_expr = true;
            }
            ev
        },
    );

    maybe_open_stopwatch_context_menu(
        state,
        target_id,
        prop_key,
        existing_expr.as_deref(),
        local_frame,
        json!(base_rgb),
        sw_ctx_pos,
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
    apply_kf_bar_event_live(
        session,
        target_id,
        prop_key,
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
pub(super) fn show_session_color_rgba_prop(
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
    let prop = lookup_session_prop_ref(session, target_id, prop_key);
    let animated = prop.is_some_and(|p| p.animated);
    let existing_expr = prop.and_then(|p| p.expression.clone());
    let has_expr = existing_expr.is_some();
    let base_rgba = eval_prop_vec4(prop, local_frame, default_rgba);
    let rgba_f64 = if let Some(expr_code) = existing_expr.as_deref() {
        eval_session_expr_vec4(session, state, target_id, prop_key, expr_code, base_rgba)
            .unwrap_or(base_rgba)
    } else {
        base_rgba
    };
    let mut rgba = Rgba::new(
        rgba_f64[0] as f32,
        rgba_f64[1] as f32,
        rgba_f64[2] as f32,
        rgba_f64[3] as f32,
    );
    let easing_plus = session.plugins.is_enabled("easing-plus");

    let mut sw_ev = Default::default();
    let mut sw_ctx_pos = None;
    let mut kf_ev = KeyframeBarEvent::default();
    let mut open_expr = false;

    let cev = panel_property_row(
        ui,
        label,
        false,
        state,
        theme,
        |ui| {
            (sw_ev, sw_ctx_pos) =
                stopwatch_button_with_right_click(ui, prop_key, animated, has_expr, theme);
        },
        |ui| {
            if animated && !has_expr {
                let bar_state =
                    build_kf_bar_state(prop, local_frame, true, easing_plus, viewport_rect);
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
            if has_expr && show_expression_inline_chip(ui, session, prop, theme) {
                open_expr = true;
            }
            ev
        },
    );

    maybe_open_stopwatch_context_menu(
        state,
        target_id,
        prop_key,
        existing_expr.as_deref(),
        local_frame,
        json!(base_rgba),
        sw_ctx_pos,
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
    apply_kf_bar_event_live(
        session,
        target_id,
        prop_key,
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
pub(super) fn show_session_option_prop(
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
pub(super) fn show_session_option_prop_ex(
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
    let prop = lookup_session_prop_ref(session, target_id, prop_key);
    let animated = animatable && prop.is_some_and(|p| p.animated);
    let existing_expr = prop.and_then(|p| p.expression.clone());
    let has_expr = animatable && existing_expr.is_some();
    let base_idx = eval_prop_u32(prop, local_frame, default_idx) as usize;
    let mut idx = if has_expr
        && let Some(expr_code) = existing_expr.as_deref()
        && let Some(live_n) = eval_session_expr_f64(
            session,
            state,
            target_id,
            prop_key,
            expr_code,
            base_idx as f64,
        ) {
        let max_opt = options.len().saturating_sub(1);
        (live_n.round().max(0.0) as usize).min(max_opt)
    } else {
        base_idx
    };
    let mut sw_ev = Default::default();
    let mut sw_ctx_pos = None;
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
                (sw_ev, sw_ctx_pos) =
                    stopwatch_button_with_right_click(ui, prop_key, animated, has_expr, theme);
            }
        },
        |ui| {
            if animated && !has_expr {
                let bar_state = build_kf_bar_state(prop, local_frame, false, false, viewport_rect);
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
            if has_expr && show_expression_inline_chip(ui, session, prop, theme) {
                open_expr = true;
            }
            ch
        },
    );

    if animatable {
        maybe_open_stopwatch_context_menu(
            state,
            target_id,
            prop_key,
            existing_expr.as_deref(),
            local_frame,
            json!(base_idx),
            sw_ctx_pos,
        );
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
        apply_kf_bar_event_live(
            session,
            target_id,
            prop_key,
            local_frame,
            clip_start,
            json!(idx),
            kf_ev,
            actions,
        );
    }
    if changed {
        let cmd = if animated {
            let frame = resolve_kf_write_frame(
                session,
                lookup_session_prop_ref(session, target_id, prop_key),
                local_frame,
            );
            Command::SetKeyframe {
                target_id: target_id.to_owned(),
                property: prop_key.to_owned(),
                frame,
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
pub(super) fn show_session_curve_prop(
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

    let prop = lookup_session_prop_ref(session, target_id, prop_key);
    let mut curve_val = eval_prop_value(prop, local_frame)
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
    let ch_aliases: &[&str] = match ch_key {
        "rgb" => &["master", "rgb", "luma"],
        "r" => &["r", "red"],
        "g" => &["g", "green"],
        "b" => &["b", "blue"],
        "a" => &["a", "alpha"],
        _ => &[ch_key],
    };
    let mut pts: Vec<[f64; 2]> = ch_aliases
        .iter()
        .find_map(|k| curve_val.get(*k).and_then(Value::as_array))
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    if let Some(a) = p.as_array() {
                        Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?])
                    } else if let Some(obj) = p.as_object() {
                        Some([obj.get("x")?.as_f64()?, obj.get("y")?.as_f64()?])
                    } else {
                        None
                    }
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
            if ch_key == "rgb" {
                obj.insert("master".to_owned(), json!(pts));
            }
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
    initial: String,
    editing: bool,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn show_session_text_prop(
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
    show_session_text_prop_ex(
        ui,
        session,
        target_id,
        prop_key,
        label,
        default_text,
        false,
        local_frame,
        state,
        theme,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn show_session_text_prop_ex(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    prop_key: &str,
    label: &str,
    default_text: &str,
    multiline: bool,
    local_frame: f64,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    touch_inspector_tx_owner(state, target_id, prop_key);
    let (animated, current) = {
        let prop = lookup_session_prop_ref(session, target_id, prop_key);
        (
            prop.is_some_and(|p| p.animated),
            eval_prop_str(prop, local_frame, default_text),
        )
    };

    // Guard multi-megabyte fuzzer / pasted strings from stalling egui TextEdit layout every frame.
    if let Some(summary) = oversized_text_summary(&current) {
        panel_property_row(
            ui,
            label,
            true,
            state,
            theme,
            |_| {},
            |ui| {
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(&summary)
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_MUTED),
                    )
                    .truncate(),
                )
                .on_hover_text(
                    "Text exceeds 8,192 characters; inline editing disabled to preserve responsiveness.",
                );
            },
        );
        return;
    }

    let state_id = Id::unique((target_id, prop_key, "text_edit_state"));
    let edit_id = Id::unique((target_id, prop_key, "text_edit"));

    let mut edit_st = ui
        .data_mut(|d| d.get_temp::<TextPropEditState>(state_id))
        .unwrap_or_else(|| TextPropEditState {
            buf: current.clone(),
            initial: current.clone(),
            editing: false,
        });
    // Only sync from session when NOT actively editing so focus-loss frame retains typed buffer.
    if !edit_st.editing {
        edit_st.buf = current.clone();
        edit_st.initial = current.clone();
    }

    let (esc_pressed, enter_pressed, cmd_or_ctrl) = ui.input(|i| {
        (
            i.key_pressed(Key::Escape),
            i.key_pressed(Key::Enter),
            i.modifiers.ctrl || i.modifiers.command,
        )
    });
    let commit_on_enter = if multiline {
        cmd_or_ctrl && enter_pressed
    } else {
        enter_pressed
    };
    let mut gained_focus = false;
    let mut text_changed = false;
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
            let field_w = ui.available_width().min(200.0);
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
                    if multiline {
                        ScrollArea::vertical()
                            .id_salt((edit_id, "multiline_scroll"))
                            .max_height(64.0)
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                ui.add(
                                    TextEdit::multiline(&mut edit_st.buf)
                                        .id(edit_id)
                                        .frame(Frame::NONE)
                                        .font(theme.font(FONT_BODY))
                                        .text_color(theme.value())
                                        .desired_rows(2)
                                        .desired_width(field_w),
                                )
                            })
                            .inner
                    } else {
                        ui.add(
                            TextEdit::singleline(&mut edit_st.buf)
                                .id(edit_id)
                                .frame(Frame::NONE)
                                .font(theme.font(FONT_BODY))
                                .text_color(theme.value())
                                .desired_width(field_w),
                        )
                    }
                })
                .inner;

            if resp.gained_focus() {
                gained_focus = true;
                edit_st.editing = true;
                edit_st.initial = current.clone();
            }
            if resp.changed() {
                text_changed = true;
                if !edit_st.editing {
                    edit_st.editing = true;
                    edit_st.initial = current.clone();
                }
            }
            if edit_st.editing {
                if esc_pressed {
                    cancel = true;
                    ui.memory_mut(|m| m.surrender_focus(edit_id));
                } else if resp.lost_focus() || (resp.has_focus() && commit_on_enter) {
                    commit = true;
                    ui.memory_mut(|m| m.surrender_focus(edit_id));
                }
            }
        },
    );

    if gained_focus && !cancel {
        close_existing_tx_before_start(session, state);
        let _ = begin_inspector_tx(
            session,
            state,
            target_id,
            prop_key,
            format!("edit {prop_key}"),
        );
    }

    if cancel {
        if owns_inspector_tx(state, target_id, prop_key) && session.history.is_in_transaction() {
            let _ = session.cancel_transaction();
            clear_inspector_tx_state(state);
        }
        edit_st.buf = edit_st.initial.clone();
        edit_st.editing = false;
    } else {
        if (text_changed || commit) && edit_st.buf != current {
            if !owns_inspector_tx(state, target_id, prop_key) {
                close_existing_tx_before_start(session, state);
                let _ = begin_inspector_tx(
                    session,
                    state,
                    target_id,
                    prop_key,
                    format!("edit {prop_key}"),
                );
            }
            let cmd = if animated {
                let frame = resolve_kf_write_frame(
                    session,
                    lookup_session_prop_ref(session, target_id, prop_key),
                    local_frame,
                );
                Command::SetKeyframe {
                    target_id: target_id.to_owned(),
                    property: prop_key.to_owned(),
                    frame,
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
            if session.execute(cmd).is_ok() && owns_inspector_tx(state, target_id, prop_key) {
                state.inspector_tx_has_edits = true;
            }
        }
        if commit {
            if owns_inspector_tx(state, target_id, prop_key) {
                commit_inspector_tx_if_open(session, state);
            }
            edit_st.editing = false;
        }
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
    ev: super::super::controls::StopwatchEvent,
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

pub(super) const EXPR_EDITOR_DEFAULT_SIZE: Vec2 = Vec2::new(460.0, 340.0);

fn is_bare_constant_literal(code: &str) -> bool {
    let trimmed = code.trim();
    if trimmed.is_empty() {
        return true;
    }
    if matches!(trimmed, "true" | "false" | "null") {
        return true;
    }
    if trimmed.parse::<f64>().is_ok() {
        return true;
    }
    if trimmed.len() >= 2
        && ((trimmed.starts_with('"') && trimmed.ends_with('"'))
            || (trimmed.starts_with('\'') && trimmed.ends_with('\'')))
    {
        return true;
    }
    if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(trimmed)
        && arr.iter().all(Value::is_number)
    {
        return true;
    }
    false
}

pub(super) fn insert_expression_snippet(current_code: &str, snippet: &str) -> String {
    if is_bare_constant_literal(current_code) {
        snippet.to_owned()
    } else {
        format!("{}\n{snippet}", current_code.trim_end())
    }
}

pub(super) fn clear_property_expression(
    session: &mut Session,
    target_id: &str,
    property: &str,
) -> Result<(), zc_session::SessionError> {
    let should_reset_animated =
        lookup_session_prop_ref(session, target_id, property).is_some_and(|p| {
            p.animated
                && (p.keyframes.is_empty()
                    || (p.keyframes.len() == 1 && p.keyframes[0].frame.abs() < 1e-6))
        });
    if should_reset_animated {
        session
            .execute(Command::TogglePropertyAnimated {
                target_id: target_id.to_owned(),
                property: property.to_owned(),
                frame: 0.0,
            })
            .map(|_| ())
    } else {
        session
            .execute(Command::SetExpression {
                target_id: target_id.to_owned(),
                property: property.to_owned(),
                expression: None,
            })
            .map(|_| ())
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
    state.expr_editor_just_opened = true;
    let (_, clamped_size) = compute_clamped_editor_window_rect(
        viewport_rect,
        EXPR_EDITOR_DEFAULT_SIZE.x,
        EXPR_EDITOR_DEFAULT_SIZE.y,
    );
    let mut spec = FloatingWindowSpec::new("expression_editor", "Expression Editor")
        .with_subtitle(prop_key.to_owned())
        .with_default_size(clamped_size);
    spec.min_size = Vec2::new(340.0, 260.0);
    state.floating_windows.open_or_focus(&spec, viewport_rect);
    clamp_floating_window_in_screen(&mut state.floating_windows, &spec, viewport_rect);
}

pub(super) fn show_stopwatch_context_menu(
    ctx: &egui::Context,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
) {
    let Some(menu) = state.stopwatch_ctx_menu.clone() else {
        return;
    };
    let screen = {
        let s = ctx.input(|i| i.viewport_rect());
        if s.width() > 100.0 && s.height() > 100.0 {
            s
        } else {
            viewport_rect
        }
    };
    let menu_w = 156.0_f32;
    let menu_h = if menu.existing_expr.is_some() {
        84.0
    } else {
        58.0
    };
    let pos = Pos2::new(
        menu.pos.x.clamp(
            screen.left() + 4.0,
            (screen.right() - menu_w - 4.0).max(screen.left() + 4.0),
        ),
        menu.pos.y.clamp(
            screen.top() + 4.0,
            (screen.bottom() - menu_h - 4.0).max(screen.top() + 4.0),
        ),
    );
    let mut close_menu = ctx.input(|i| i.key_pressed(Key::Escape));
    let mut action_edit_expr = false;
    let mut action_clear_expr = false;
    let mut action_toggle_kf = false;

    let area_resp = Area::new(Id::unique("stopwatch_ctx_menu"))
        .order(Order::Foreground)
        .fixed_pos(pos)
        .show(ctx, |ui| {
            Frame::NONE
                .fill(BG_PANEL)
                .stroke(Stroke::new(1.0, BORDER_CONTROL))
                .inner_margin(egui::Margin::same(4))
                .show(ui, |ui| {
                    ui.set_min_width(menu_w - 8.0);
                    ui.spacing_mut().item_spacing.y = 2.0;
                    if ui
                        .add_sized(
                            Vec2::new(menu_w - 8.0, 22.0),
                            egui::Button::new(
                                egui::RichText::new("Edit Expression...")
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_PRIMARY),
                            )
                            .fill(BG_BUTTON)
                            .stroke(Stroke::NONE),
                        )
                        .clicked()
                    {
                        action_edit_expr = true;
                        close_menu = true;
                    }
                    if menu.existing_expr.is_some()
                        && ui
                            .add_sized(
                                Vec2::new(menu_w - 8.0, 22.0),
                                egui::Button::new(
                                    egui::RichText::new("Clear Expression")
                                        .font(theme.font(FONT_SMALL))
                                        .color(DANGER_TEXT),
                                )
                                .fill(BG_BUTTON)
                                .stroke(Stroke::NONE),
                            )
                            .clicked()
                    {
                        action_clear_expr = true;
                        close_menu = true;
                    }
                    if ui
                        .add_sized(
                            Vec2::new(menu_w - 8.0, 22.0),
                            egui::Button::new(
                                egui::RichText::new("Toggle Keyframe")
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_PRIMARY),
                            )
                            .fill(BG_BUTTON)
                            .stroke(Stroke::NONE),
                        )
                        .clicked()
                    {
                        action_toggle_kf = true;
                        close_menu = true;
                    }
                });
        });

    if !close_menu
        && ctx.input(|i| {
            i.pointer.primary_pressed()
                && i.pointer
                    .interact_pos()
                    .is_some_and(|p| !area_resp.response.rect.contains(p))
        })
    {
        close_menu = true;
    }

    if action_edit_expr {
        let init_code = menu
            .existing_expr
            .clone()
            .unwrap_or_else(|| menu.current_val.to_string());
        if menu.existing_expr.is_none() {
            let _ = session.execute(Command::SetExpression {
                target_id: menu.target_id.clone(),
                property: menu.property.clone(),
                expression: Some(init_code.clone()),
            });
        }
        open_expression_window(&menu.target_id, &menu.property, init_code, state, screen);
    } else if action_clear_expr {
        let _ = clear_property_expression(session, &menu.target_id, &menu.property);
        if state
            .expr_editor
            .as_ref()
            .is_some_and(|e| e.target_id == menu.target_id && e.property == menu.property)
        {
            state.expr_editor = None;
            state.floating_windows.close("expression_editor");
        }
    } else if action_toggle_kf {
        let _ = session.execute(Command::TogglePropertyAnimated {
            target_id: menu.target_id.clone(),
            property: menu.property.clone(),
            frame: normalized_kf_frame(session, menu.local_frame),
        });
    }

    if close_menu {
        state.stopwatch_ctx_menu = None;
    }
}
