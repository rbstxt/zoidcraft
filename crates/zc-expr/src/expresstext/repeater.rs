use super::*;

/// Distribution mode for ExpressText+ Repeater copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RepeaterMode {
    /// Step linearly per copy slot (`offset * slot`).
    #[default]
    Linear = 0,
    /// Scatter deterministically within `[range_min, range_max]` using 3D lattice hash.
    Random = 1,
    /// Trail behind the letter's per-frame velocity (`-velocity * lag * slot`).
    Echo = 2,
}

impl RepeaterMode {
    pub fn from_raw(value: f64) -> Self {
        match if value.is_finite() {
            js_round(value) as i32
        } else {
            0
        } {
            1 => Self::Random,
            2 => Self::Echo,
            _ => Self::Linear,
        }
    }
}

/// Evaluated parameters for one Repeater effector layer at a given frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepeaterSpec {
    pub copies: usize,
    pub mode: RepeaterMode,
    pub offset: [f64; 3],
    pub rotation_step: [f64; 3],
    pub scale_step: [f64; 3],
    pub range_min: [f64; 3],
    pub range_max: [f64; 3],
    pub seed: f64,
    pub lag: f64,
    /// Fade factor in `[0, 1]` (Zoidium `repeaterFade / 100`).
    pub fade: f64,
    /// Starting opacity for slot 0..copies fade (default `1.0`).
    pub opacity_start: f64,
    /// Optional explicit end opacity at `slot == copies`; when `None`, uses Zoidium
    /// `1.0 - fade * (slot / copies)`.
    pub opacity_end: Option<f64>,
}

impl Default for RepeaterSpec {
    fn default() -> Self {
        Self {
            copies: 3,
            mode: RepeaterMode::Linear,
            offset: [10.0, 0.0, 0.0],
            rotation_step: [0.0, 0.0, 0.0],
            scale_step: [1.0, 1.0, 1.0],
            range_min: [-10.0, -10.0, -10.0],
            range_max: [10.0, 10.0, 10.0],
            seed: 1.0,
            lag: 2.0,
            fade: 0.5,
            opacity_start: 1.0,
            opacity_end: None,
        }
    }
}

/// One generated ghost copy instance for a letter produced by a Repeater effector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepeaterInstance {
    /// Index of the Repeater layer inside the evaluated effector stack.
    pub layer_index: usize,
    /// 1-based copy slot (`1..=copies`).
    pub copy_index: usize,
    /// Weighted position delta added on top of the letter's evaluated position.
    pub position_offset: [f64; 3],
    /// Weighted Euler rotation delta (radians) added on top of the letter's rotation.
    pub rotation_offset: [f64; 3],
    /// Weighted scale multiplier applied on top of the letter's scale.
    pub scale_multiplier: [f64; 3],
    /// Full composed `LetterControls` for this ghost instance.
    pub controls: LetterControls,
    /// Per-copy opacity multiplier in `[0, 1]`.
    pub opacity: f64,
    /// Effective effector weight (strength * field falloff) for this letter.
    pub weight: f64,
}

/// Reads a [`RepeaterSpec`] from an ExpressText+ property map at `frame`.
pub fn repeater_spec_from_props(props: &dyn EtPropReader, frame: f64) -> RepeaterSpec {
    let enabled = read_num(props, "repeaterEnable", frame, 1.0) != 0.0;
    let copies = if enabled {
        let raw = read_num(props, "repeaterCopies", frame, 3.0);
        let floored = if raw.is_finite() {
            raw.floor() as isize
        } else {
            3
        };
        floored.clamp(0, REPEATER_MAX_COPIES as isize) as usize
    } else {
        0
    };
    let offset = if props.read("repeaterOffset", frame).is_some() {
        read_vec3(props, "repeaterOffset", frame, [10.0, 0.0, 0.0])
    } else {
        read_vec3(props, "repeaterPosition", frame, [10.0, 0.0, 0.0])
    };
    let rotation_step = read_vec3(props, "repeaterRotation", frame, [0.0, 0.0, 0.0]);
    let scale_step = read_vec3(props, "repeaterScale", frame, [1.0, 1.0, 1.0]);
    let fade = (read_num(props, "repeaterFade", frame, 50.0) / 100.0).clamp(0.0, 1.0);
    let opacity_start = read_num(props, "repeaterOpacityStart", frame, 1.0).clamp(0.0, 1.0);
    let opacity_end = props
        .read("repeaterOpacityEnd", frame)
        .map(|v| v.as_number(1.0 - fade).clamp(0.0, 1.0));
    RepeaterSpec {
        copies,
        mode: RepeaterMode::from_raw(read_num(props, "repeaterMode", frame, 0.0)),
        offset,
        rotation_step,
        scale_step,
        range_min: read_vec3(props, "repeaterRangeMin", frame, [-10.0, -10.0, -10.0]),
        range_max: read_vec3(props, "repeaterRangeMax", frame, [10.0, 10.0, 10.0]),
        seed: read_num(props, "repeaterSeed", frame, 1.0),
        lag: read_num(props, "repeaterLag", frame, 2.0).max(0.0),
        fade,
        opacity_start,
        opacity_end,
    }
}

/// Raw unweighted copy offset for 1-based `slot`, matching Zoidium
/// `expresstext-layout.js:repeaterCopyOffset`.
pub fn repeater_copy_offset(
    spec: &RepeaterSpec,
    letter_index: usize,
    slot: usize,
    velocity: Option<[f64; 3]>,
) -> [f64; 3] {
    let slot_f = slot as f64;
    match spec.mode {
        RepeaterMode::Echo => {
            let v = velocity.unwrap_or([0.0, 0.0, 0.0]);
            [
                -v[0] * spec.lag * slot_f,
                -v[1] * spec.lag * slot_f,
                -v[2] * spec.lag * slot_f,
            ]
        }
        RepeaterMode::Random => {
            let seed_int = js_round(
                (if spec.seed.is_finite() {
                    spec.seed
                } else {
                    1.0
                }) * 1000.0,
            ) as i32;
            let mut out = [0.0; 3];
            for (axis, out_val) in out.iter_mut().enumerate() {
                let u = hash_lattice(
                    letter_index as i32,
                    (slot as i32).wrapping_mul(57).wrapping_add(11),
                    (axis as i32).wrapping_mul(131).wrapping_add(7),
                    seed_int,
                );
                *out_val = spec.range_min[axis] + (spec.range_max[axis] - spec.range_min[axis]) * u;
            }
            out
        }
        RepeaterMode::Linear => [
            spec.offset[0] * slot_f,
            spec.offset[1] * slot_f,
            spec.offset[2] * slot_f,
        ],
    }
}

/// Opacity multiplier in `[0, 1]` for 1-based `slot`, matching Zoidium
/// `expresstext-layout.js:repeaterCopyOpacity` (and supporting explicit start/end fade).
pub fn repeater_copy_opacity(spec: &RepeaterSpec, slot: usize) -> f64 {
    if spec.copies == 0 {
        return 1.0;
    }
    let t = (slot as f64) / (spec.copies as f64);
    if let Some(end_op) = spec.opacity_end {
        (spec.opacity_start + (end_op - spec.opacity_start) * t).clamp(0.0, 1.0)
    } else {
        (spec.opacity_start * (1.0 - spec.fade * t)).clamp(0.0, 1.0)
    }
}

/// Evaluates all Repeater effector layers in `layers` for one letter at `frame`, returning
/// per-copy instance transforms and opacity multipliers.
pub fn evaluate_expresstext_repeater_instances(
    layers: &[EffectorLayer<'_>],
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    aux: &EvalAux,
) -> Vec<RepeaterInstance> {
    let sections = sections_at(layers, frame);
    let mut active_repeaters = Vec::new();
    let mut needs_echo_velocity = false;

    for (layer_idx, layer) in layers.iter().enumerate() {
        if layer.kind != EffectorKind::Repeater {
            continue;
        }
        if read_num(layer.props, "repeaterEnable", frame, 1.0) == 0.0 {
            continue;
        }
        let ctx = letter_context(
            layer.props,
            EffectorKind::Repeater,
            frame,
            index,
            list,
            &sections,
        );
        if !is_letter_selected(&ctx.selected, index) {
            continue;
        }
        let weight = apply_field_weight(
            ctx.strength,
            layer.props,
            EffectorKind::Repeater,
            frame,
            base,
        );
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        let local_frame = frame - ctx.delay;
        let spec = repeater_spec_from_props(layer.props, local_frame);
        if spec.copies == 0 {
            continue;
        }
        if spec.mode == RepeaterMode::Echo {
            needs_echo_velocity = true;
        }
        active_repeaters.push((layer_idx, spec, weight));
    }

    if active_repeaters.is_empty() {
        return Vec::new();
    }

    let base_controls = evaluate_stack(layers, frame, index, list, base, aux);
    let velocity = if needs_echo_velocity {
        let prev = evaluate_stack(layers, frame - 1.0, index, list, base, aux);
        Some([
            base_controls.position[0] - prev.position[0],
            base_controls.position[1] - prev.position[1],
            base_controls.position[2] - prev.position[2],
        ])
    } else {
        None
    };

    let total_instances: usize = active_repeaters.iter().map(|(_, s, _)| s.copies).sum();
    let mut out = Vec::with_capacity(total_instances);

    for (layer_index, spec, weight) in active_repeaters {
        for slot in 1..=spec.copies {
            let raw = repeater_copy_offset(&spec, index, slot, velocity);
            let slot_w = (slot as f64) * weight;
            let pos_off = [raw[0] * weight, raw[1] * weight, raw[2] * weight];
            let rot_off = [
                spec.rotation_step[0] * slot_w,
                spec.rotation_step[1] * slot_w,
                spec.rotation_step[2] * slot_w,
            ];
            let scl_mul = [
                spec.scale_step[0].max(1e-6).powf(slot_w),
                spec.scale_step[1].max(1e-6).powf(slot_w),
                spec.scale_step[2].max(1e-6).powf(slot_w),
            ];
            let controls = LetterControls {
                position: [
                    base_controls.position[0] + pos_off[0],
                    base_controls.position[1] + pos_off[1],
                    base_controls.position[2] + pos_off[2],
                ],
                rotation: [
                    base_controls.rotation[0] + rot_off[0],
                    base_controls.rotation[1] + rot_off[1],
                    base_controls.rotation[2] + rot_off[2],
                ],
                scale: [
                    (base_controls.scale[0] * scl_mul[0]).max(1e-6),
                    (base_controls.scale[1] * scl_mul[1]).max(1e-6),
                    (base_controls.scale[2] * scl_mul[2]).max(1e-6),
                ],
            };
            out.push(RepeaterInstance {
                layer_index,
                copy_index: slot,
                position_offset: pos_off,
                rotation_offset: rot_off,
                scale_multiplier: scl_mul,
                controls,
                opacity: repeater_copy_opacity(&spec, slot),
                weight,
            });
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Voronoi Shatter Effector (`expresstext-voronoi.js` & `expresstext-render.js`)
