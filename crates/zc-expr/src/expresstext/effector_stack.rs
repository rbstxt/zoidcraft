use super::*;

pub(super) fn sections_at(
    layers: &[EffectorLayer<'_>],
    frame: f64,
) -> BTreeMap<String, Vec<String>> {
    let mut sections: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for layer in layers {
        let kind_str = layer.kind.as_str();
        let letters = read_str(layer.props, layer.kind.prop_keys().letters, frame, "");
        sections
            .entry(kind_str.to_string())
            .or_default()
            .push(letters);
    }
    sections
}

fn evaluate_effectors_with_sections(
    layers: &[EffectorLayer<'_>],
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    aux: &EvalAux,
    sections: &BTreeMap<String, Vec<String>>,
) -> LetterControls {
    let mut out = LetterControls::default();
    for layer in layers {
        match layer.kind {
            EffectorKind::Delay | EffectorKind::Repeater | EffectorKind::Voronoi => {}
            EffectorKind::Shake => {
                evaluate_shake(
                    layer.props,
                    frame,
                    index,
                    list,
                    base,
                    &mut out,
                    sections,
                    aux,
                );
            }
            EffectorKind::Plain => {
                evaluate_plain(layer.props, frame, index, list, base, &mut out, sections);
            }
            EffectorKind::Random => {
                evaluate_random(layer.props, frame, index, list, base, &mut out, sections);
            }
            EffectorKind::Formula => {
                evaluate_formula(
                    layer.props,
                    frame,
                    index,
                    list,
                    base,
                    &mut out,
                    sections,
                    aux,
                );
            }
            EffectorKind::Twist => {
                evaluate_twist(layer.props, frame, index, list, base, &mut out, sections);
            }
            EffectorKind::Wrap => {
                evaluate_wrap(layer.props, frame, index, list, base, &mut out, sections);
            }
            EffectorKind::Orbit => {
                evaluate_orbit(
                    layer.props,
                    frame,
                    index,
                    list,
                    base,
                    &mut out,
                    sections,
                    aux,
                );
            }
            EffectorKind::Spline => {
                evaluate_spline(
                    layer.props,
                    frame,
                    index,
                    list,
                    base,
                    &mut out,
                    sections,
                    aux,
                );
            }
        }
    }
    out
}

/// Evaluates all non-Delay effectors in `layers` for one letter at `frame`.
pub fn evaluate_effectors(
    layers: &[EffectorLayer<'_>],
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    aux: &EvalAux,
) -> LetterControls {
    let sections = sections_at(layers, frame);
    evaluate_effectors_with_sections(layers, frame, index, list, base, aux, &sections)
}

#[derive(Debug, Clone, Copy)]
struct DelayParams {
    strength: f64,
    mode: i32,
    position: bool,
    scale: bool,
    rotation: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct DelayVelocity {
    position: [f64; 3],
    rotation: [f64; 3],
    scale: [f64; 3],
}

fn delay_link_params(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    sections: &BTreeMap<String, Vec<String>>,
) -> Option<DelayParams> {
    if read_num(props, "delayEnable", frame, 1.0) == 0.0 {
        return None;
    }
    let selected = resolve_selection(
        &read_str(props, "delayLetters", frame, ""),
        list.total,
        sections,
    );
    if !is_letter_selected(&selected, index) {
        return None;
    }
    let gated = apply_field_weight(
        strength_weight(read_num(props, "delayStrength", frame, 100.0)),
        props,
        EffectorKind::Delay,
        frame,
        base,
    );
    if !gated.is_finite() || gated <= 0.0 {
        return None;
    }
    let mode_raw = js_round(read_num(props, "delayMode", frame, 0.0)) as i32;
    let mode = match mode_raw {
        1 => 1,
        2 => 2,
        _ => 0,
    };
    Some(DelayParams {
        strength: gated.clamp(0.0, 1.0),
        mode,
        position: read_num(props, "delayPosition", frame, 1.0) != 0.0,
        scale: read_num(props, "delayScale", frame, 1.0) != 0.0,
        rotation: read_num(props, "delayRotation", frame, 1.0) != 0.0,
    })
}

fn chase_axis_group(
    cur: &mut [f64; 3],
    vel: &mut [f64; 3],
    want: [f64; 3],
    travel: [f64; 3],
    chase: Option<DelayParams>,
    enabled: bool,
    step: f64,
) {
    let Some(p) = chase.filter(|p| enabled && p.strength > 0.0) else {
        *cur = want;
        *vel = [0.0; 3];
        return;
    };
    let s = p.strength;
    for axis in 0..3 {
        let from = cur[axis];
        let to = want[axis];
        if p.mode == 2 {
            let soft = s.clamp(0.0, 1.0).sqrt();
            let stiff = 1.0 + (DELAY_SPRING_STIFF - 1.0) * soft;
            let damp = DELAY_SPRING_DAMP_MIN + (DELAY_SPRING_DAMP - DELAY_SPRING_DAMP_MIN) * s;
            vel[axis] += (to - from) * stiff * step;
            vel[axis] *= damp.powf(step);
            cur[axis] = from + vel[axis] * step;
        } else if p.mode == 1 {
            let k = 1.0 + (DELAY_BLEND_CATCH - 1.0) * s;
            cur[axis] = from + (to - from) * (k * step).min(1.0);
        } else {
            let pace = travel[axis];
            let rate = if pace > 0.0 {
                (pace / (s * DELAY_EVEN_SPAN).max(1e-9)) * step
            } else {
                0.0
            };
            let diff = to - from;
            cur[axis] = from + diff.clamp(-rate, rate);
        }
    }
}

fn blend_delay_output(target: LetterControls, chased: LetterControls, s: f64) -> LetterControls {
    if !s.is_finite() || s <= 0.0 {
        return target;
    }
    if s >= 1.0 {
        return chased;
    }
    let lerp3 = |a: [f64; 3], b: [f64; 3]| {
        [
            a[0] + (b[0] - a[0]) * s,
            a[1] + (b[1] - a[1]) * s,
            a[2] + (b[2] - a[2]) * s,
        ]
    };
    LetterControls {
        position: lerp3(target.position, chased.position),
        rotation: lerp3(target.rotation, chased.rotation),
        scale: lerp3(target.scale, chased.scale),
    }
}

fn chase_step(
    state: LetterControls,
    velocity: &mut DelayVelocity,
    own: LetterControls,
    params: Option<DelayParams>,
    travel: LetterControls,
    dt: f64,
) -> (LetterControls, LetterControls) {
    let step = if dt.is_finite() { dt.max(0.0) } else { 1.0 };
    let mut chased = state;
    chase_axis_group(
        &mut chased.position,
        &mut velocity.position,
        own.position,
        travel.position,
        params,
        params.is_some_and(|p| p.position),
        step,
    );
    chase_axis_group(
        &mut chased.rotation,
        &mut velocity.rotation,
        own.rotation,
        travel.rotation,
        params,
        params.is_some_and(|p| p.rotation),
        step,
    );
    chase_axis_group(
        &mut chased.scale,
        &mut velocity.scale,
        own.scale,
        travel.scale,
        params,
        params.is_some_and(|p| p.scale),
        step,
    );
    let output = if params.is_some_and(|p| p.mode == 2) {
        blend_delay_output(own, chased, params.map_or(0.0, |p| p.strength))
    } else {
        chased
    };
    (chased, output)
}

/// Evaluates the full ExpressText+ effector stack (including stacked Delay/Blend/Spring layers)
/// for one letter at `frame` (`expresstext-delay.js:144-165, 281-361, 451-508`).
pub fn evaluate_stack(
    layers: &[EffectorLayer<'_>],
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    aux: &EvalAux,
) -> LetterControls {
    let delays: Vec<&EffectorLayer<'_>> = layers
        .iter()
        .filter(|l| l.kind == EffectorKind::Delay)
        .collect();
    if delays.is_empty() || !frame.is_finite() || frame <= 0.0 {
        return evaluate_effectors(layers, frame, index, list, base, aux);
    }

    let last_step = frame.floor() as usize;
    let first_step = last_step.saturating_sub(DELAY_SIM_WINDOW);
    let step_count = last_step - first_step + 1;
    let frac = frame - (last_step as f64);

    let all_letters_static = layers
        .iter()
        .all(|l| l.props.is_static(l.kind.prop_keys().letters));
    let hoisted_sections = if all_letters_static {
        Some(sections_at(layers, frame))
    } else {
        None
    };
    let dynamic_sections_table: Vec<BTreeMap<String, Vec<String>>> = if all_letters_static {
        Vec::new()
    } else {
        (first_step..=last_step)
            .map(|f| sections_at(layers, f as f64))
            .collect()
    };
    let dynamic_tail_sections = if !all_letters_static && frac > 1e-9 {
        Some(sections_at(layers, frame))
    } else {
        None
    };

    // Stage 0 table: raw effector outputs at integer frames `first_step..=last_step` + fractional end.
    let mut stage_table: Vec<LetterControls> = (first_step..=last_step)
        .enumerate()
        .map(|(k, f)| {
            let sections = hoisted_sections
                .as_ref()
                .unwrap_or_else(|| &dynamic_sections_table[k]);
            evaluate_effectors_with_sections(layers, f as f64, index, list, base, aux, sections)
        })
        .collect();
    let mut stage_tail = if frac > 1e-9 {
        let sections = hoisted_sections
            .as_ref()
            .or(dynamic_tail_sections.as_ref())
            .unwrap();
        evaluate_effectors_with_sections(layers, frame, index, list, base, aux, sections)
    } else {
        *stage_table.last().unwrap()
    };

    for delay in delays {
        // Compute travel across the window for Even pacing.
        let mut lo_pos = [f64::INFINITY; 3];
        let mut hi_pos = [f64::NEG_INFINITY; 3];
        let mut lo_rot = [f64::INFINITY; 3];
        let mut hi_rot = [f64::NEG_INFINITY; 3];
        let mut lo_scl = [f64::INFINITY; 3];
        let mut hi_scl = [f64::NEG_INFINITY; 3];
        for t in &stage_table {
            for a in 0..3 {
                lo_pos[a] = lo_pos[a].min(t.position[a]);
                hi_pos[a] = hi_pos[a].max(t.position[a]);
                lo_rot[a] = lo_rot[a].min(t.rotation[a]);
                hi_rot[a] = hi_rot[a].max(t.rotation[a]);
                lo_scl[a] = lo_scl[a].min(t.scale[a]);
                hi_scl[a] = hi_scl[a].max(t.scale[a]);
            }
        }
        let span3 = |lo: [f64; 3], hi: [f64; 3]| {
            [
                if hi[0] > lo[0] { hi[0] - lo[0] } else { 0.0 },
                if hi[1] > lo[1] { hi[1] - lo[1] } else { 0.0 },
                if hi[2] > lo[2] { hi[2] - lo[2] } else { 0.0 },
            ]
        };
        let travel = LetterControls {
            position: span3(lo_pos, hi_pos),
            rotation: span3(lo_rot, hi_rot),
            scale: span3(lo_scl, hi_scl),
        };

        let mut state = stage_table[0];
        let mut velocity = DelayVelocity::default();
        let mut next_table = Vec::with_capacity(step_count);
        for (k, &own) in stage_table.iter().enumerate() {
            let f = (first_step + k) as f64;
            let sections = hoisted_sections
                .as_ref()
                .unwrap_or_else(|| &dynamic_sections_table[k]);
            let params = delay_link_params(delay.props, f, index, list, base, sections);
            let (next_state, out) = chase_step(state, &mut velocity, own, params, travel, 1.0);
            state = next_state;
            next_table.push(out);
        }
        let next_tail = if frac > 1e-9 {
            let sections = hoisted_sections
                .as_ref()
                .or(dynamic_tail_sections.as_ref())
                .unwrap();
            let params = delay_link_params(delay.props, frame, index, list, base, sections);
            let mut tail_vel = velocity;
            let (_, out) = chase_step(state, &mut tail_vel, stage_tail, params, travel, frac);
            out
        } else {
            *next_table.last().unwrap()
        };
        stage_table = next_table;
        stage_tail = next_tail;
    }

    stage_tail
}

// ---------------------------------------------------------------------------
// Repeater Effector (`expresstext-layout.js` & `expresstext-render.js`)
