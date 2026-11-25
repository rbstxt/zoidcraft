use super::*;

fn add_scaled(target: &mut [f64; 3], v: [f64; 3], amount: f64) {
    target[0] += v[0] * amount;
    target[1] += v[1] * amount;
    target[2] += v[2] * amount;
}

fn mul_scale(target: &mut [f64; 3], v: [f64; 3], amount: f64) {
    target[0] *= 1.0 + (v[0] - 1.0) * amount;
    target[1] *= 1.0 + (v[1] - 1.0) * amount;
    target[2] *= 1.0 + (v[2] - 1.0) * amount;
}

pub fn rotate_axis_x(v: [f64; 3], a: f64) -> [f64; 3] {
    let (s, c) = a.sin_cos();
    [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]]
}

pub fn rotate_axis_y(v: [f64; 3], a: f64) -> [f64; 3] {
    let (s, c) = a.sin_cos();
    [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]]
}

pub fn rotate_axis_z(v: [f64; 3], a: f64) -> [f64; 3] {
    let (s, c) = a.sin_cos();
    [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]
}

/// XYZ-Euler rotation (`R = Rx * Ry * Rz`, matching Three.js and `expresstext-effectors.js:225-239`).
pub fn rotate_euler_xyz(v: [f64; 3], e: [f64; 3]) -> [f64; 3] {
    let (sx, cx) = e[0].sin_cos();
    let (sy, cy) = e[1].sin_cos();
    let (sz, cz) = e[2].sin_cos();
    let x1 = cz * v[0] - sz * v[1];
    let y1 = sz * v[0] + cz * v[1];
    let z1 = v[2];
    let x2 = cy * x1 + sy * z1;
    let y2 = y1;
    let z2 = -sy * x1 + cy * z1;
    [x2, cx * y2 - sx * z2, sx * y2 + cx * z2]
}

pub fn euler_from_basis(x_axis: [f64; 3], y_axis: [f64; 3], z_axis: [f64; 3]) -> [f64; 3] {
    let clamped = z_axis[0].clamp(-1.0, 1.0);
    if clamped.abs() < 0.999_999_9 {
        [
            (-z_axis[1]).atan2(z_axis[2]),
            clamped.asin(),
            (-y_axis[0]).atan2(x_axis[0]),
        ]
    } else {
        [y_axis[2].atan2(y_axis[1]), clamped.asin(), 0.0]
    }
}

pub fn world_base_of(local_base: [f64; 3], world: TextTransform) -> [f64; 3] {
    let scaled = [
        local_base[0] * world.scale[0],
        local_base[1] * world.scale[1],
        local_base[2] * world.scale[2],
    ];
    let spun = rotate_euler_xyz(scaled, world.rotation);
    [
        world.position[0] + spun[0],
        world.position[1] + spun[1],
        world.position[2] + spun[2],
    ]
}

pub fn local_base_of(world_point: [f64; 3], world: TextTransform) -> [f64; 3] {
    let d = [
        world_point[0] - world.position[0],
        world_point[1] - world.position[1],
        world_point[2] - world.position[2],
    ];
    let unx = rotate_axis_x(d, -world.rotation[0]);
    let uny = rotate_axis_y(unx, -world.rotation[1]);
    let unz = rotate_axis_z(uny, -world.rotation[2]);
    [
        if world.scale[0] == 0.0 {
            0.0
        } else {
            unz[0] / world.scale[0]
        },
        if world.scale[1] == 0.0 {
            0.0
        } else {
            unz[1] / world.scale[1]
        },
        if world.scale[2] == 0.0 {
            0.0
        } else {
            unz[2] / world.scale[2]
        },
    ]
}

pub(super) struct LetterCtx {
    pub(super) selected: LetterSelection,
    pub(super) delay: f64,
    pub(super) strength: f64,
}

pub(super) fn letter_context(
    props: &dyn EtPropReader,
    kind: EffectorKind,
    frame: f64,
    index: usize,
    list: &LetterList,
    sections: &BTreeMap<String, Vec<String>>,
) -> LetterCtx {
    let keys = kind.prop_keys();
    let letters_str = read_str(props, keys.letters, frame, "");
    let selected = resolve_selection(&letters_str, list.total, sections);
    let order: Vec<usize> = match &selected {
        None => list.order.clone(),
        Some(set) => list
            .order
            .iter()
            .copied()
            .filter(|i| set.contains(i))
            .collect(),
    };
    let slots = order.len().max(1);
    let pos = order
        .iter()
        .position(|&i| i == index)
        .map_or(-1_isize, |p| p as isize);
    let rank = delay_rank(
        read_num(props, keys.delay_order, frame, 0.0),
        pos,
        slots,
        read_num(props, keys.seed, frame, 1.0),
    );
    let delay = rank * read_num(props, keys.delay, frame, 0.0);
    let strength = strength_weight(read_num(props, keys.strength, frame, 100.0));
    LetterCtx {
        selected,
        delay,
        strength,
    }
}

pub(super) fn apply_field_weight(
    weight: f64,
    props: &dyn EtPropReader,
    kind: EffectorKind,
    frame: f64,
    base: [f64; 3],
) -> f64 {
    if weight <= 0.0 {
        return 0.0;
    }
    let keys = kind.prop_keys();
    let field_kind = js_round(read_num(props, keys.field, frame, 0.0));
    if field_kind == 0.0 {
        return weight;
    }
    weight
        * field_weight(
            field_kind,
            base,
            read_vec3(props, keys.field_position, frame, [0.0, 0.0, 0.0]),
            read_vec3(props, keys.field_scale, frame, [100.0, 100.0, 100.0]),
        )
}

fn eval_rate(aux: &EvalAux) -> f64 {
    if aux.rate.is_finite() && aux.rate > 0.0 {
        aux.rate
    } else {
        30.0
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_shake(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
    aux: &EvalAux,
) {
    if read_num(props, "shakeEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, EffectorKind::Shake, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, EffectorKind::Shake, frame, base);
    if weight <= 0.0 {
        return;
    }
    let local_frame = frame - ctx.delay;
    let frequency = read_num(props, "shakeFrequency", frame, 1.0);
    let offset = read_num(props, "shakeOffset", frame, 0.0);
    let mode = js_round(read_num(props, "shakeMode", frame, 0.0)) as i32;
    let value = if mode == 1 {
        let time = local_frame * frequency / eval_rate(aux) + offset;
        let seed = js_round(read_num(props, "shakeSeed", frame, 1.0) * 1000.0) as i32;
        fractal_noise3(time, 0.0, 0.0, seed, 3, false) * 2.0 - 1.0
    } else {
        shake_value(local_frame, frequency, offset, 0.0, eval_rate(aux))
    };
    let amount = value * weight;
    add_scaled(
        &mut out.position,
        read_vec3(props, "shakePosition", frame, [0.0, 0.0, 0.0]),
        amount,
    );
    add_scaled(
        &mut out.rotation,
        read_vec3(props, "shakeRotation", frame, [0.0, 0.0, 0.0]),
        amount,
    );
    mul_scale(
        &mut out.scale,
        read_vec3(props, "shakeScale", frame, [1.0, 1.0, 1.0]),
        amount,
    );
}

pub(super) fn evaluate_plain(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
) {
    if read_num(props, "plainEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, EffectorKind::Plain, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, EffectorKind::Plain, frame, base);
    if weight <= 0.0 {
        return;
    }
    let local_frame = frame - ctx.delay;
    let mut position = read_vec3(props, "plainPosition", local_frame, [0.0, 0.0, 0.0]);
    let mut rotation = read_vec3(props, "plainRotation", local_frame, [0.0, 0.0, 0.0]);
    let invert_odd = js_round(read_num(props, "plainInvertOdd", frame, 0.0)) as i32 == 1;
    if invert_odd && (index + 1) % 2 == 1 {
        position = [-position[0], -position[1], -position[2]];
        rotation = [-rotation[0], -rotation[1], -rotation[2]];
    }
    add_scaled(&mut out.position, position, weight);
    add_scaled(&mut out.rotation, rotation, weight);
    mul_scale(
        &mut out.scale,
        read_vec3(props, "plainScale", local_frame, [1.0, 1.0, 1.0]),
        weight,
    );
}

struct RandomOpts {
    seed_int: i32,
    sync: bool,
    indexed: bool,
    space: i32,
    anim: f64,
    noise: f64,
    rank: isize,
    count: usize,
}

fn random_unit(
    mode: i32,
    index: usize,
    group: i32,
    axis: i32,
    base: [f64; 3],
    frame: f64,
    opts: &RandomOpts,
) -> f64 {
    let m = match mode {
        1..=4 => mode,
        _ => 0,
    };
    let seed_int = opts.seed_int;
    let group_salt = if opts.sync {
        0
    } else {
        (group + 1).wrapping_mul(100_003)
    };
    if m == 4 {
        let salt = seed_int
            .wrapping_add(group_salt)
            .wrapping_add(axis.wrapping_mul(104_729));
        let count = opts.count.max(1);
        let strata = shuffled_strata(count, salt);
        let rank = opts.rank.clamp(0, (count - 1) as isize) as usize;
        return (strata[rank] as f64 + 0.5) / (count as f64);
    }
    if m == 2 || m == 3 {
        let scale = opts.noise.max(0.0) / 100.0;
        let idx_f = index as f64;
        let at = if opts.space == 1 {
            [idx_f * 3.71, idx_f * 9.13 + 4.7, idx_f * 5.37 + 9.1]
        } else {
            base
        };
        let t = (frame * opts.anim.max(0.0)) / 100.0 / 30.0;
        let freq = scale / 15.0;
        let gx = if opts.sync {
            0.0
        } else {
            f64::from(group) * 13.13
        };
        let ax = if opts.indexed {
            f64::from(axis) * 19.19
        } else {
            0.0
        };
        let x = at[0] * freq + 0.63 * t + ax;
        let y = at[1] * freq + 1.17 * t + gx;
        let z = at[2] * freq + 0.81 * t;
        return if m == 3 {
            fractal_noise3(x, y, z, seed_int, 4, true)
        } else {
            fractal_noise3(x, y, z, seed_int, 2, false)
        };
    }
    if m == 1 {
        let idx_i = index as i32;
        let u = hash_lattice(idx_i, group_salt, axis * 3, seed_int)
            + hash_lattice(idx_i + 1009, group_salt, axis * 3 + 1, seed_int)
            + hash_lattice(idx_i + 2017, group_salt, axis * 3 + 2, seed_int);
        return u / 3.0;
    }
    hash_lattice(index as i32, group_salt, axis, seed_int)
}

pub(super) fn evaluate_random(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
) {
    if read_num(props, "randomEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, EffectorKind::Random, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let raw_strength = read_num(props, "randomStrength", frame, 100.0) / 100.0;
    let field = apply_field_weight(1.0, props, EffectorKind::Random, frame, base);
    let weight = raw_strength * field;
    if weight == 0.0 || weight.is_nan() {
        return;
    }
    let seed = read_num(props, "randomSeed", frame, 1.0);
    let mode = js_round(read_num(props, "randomMode", frame, 0.0)) as i32;
    let minimum = read_num(props, "randomMinimum", frame, 0.0) / 100.0;
    let maximum = read_num(props, "randomMaximum", frame, 100.0) / 100.0;
    let all: Vec<usize> = if list.order.is_empty() {
        (0..list.total).collect()
    } else {
        list.order.clone()
    };
    let sel_array: Vec<usize> = match &ctx.selected {
        None => all,
        Some(set) => all.into_iter().filter(|i| set.contains(i)).collect(),
    };
    let opts = RandomOpts {
        seed_int: js_round(seed * 1000.0) as i32,
        sync: read_num(props, "randomSynchronized", frame, 0.0) != 0.0,
        indexed: read_num(props, "randomIndexed", frame, 1.0) != 0.0,
        space: js_round(read_num(props, "randomSpace", frame, 0.0)) as i32,
        anim: read_num(props, "randomAnimationSpeed", frame, 0.0).max(0.0),
        noise: read_num(props, "randomNoiseScale", frame, 100.0).max(0.0),
        rank: sel_array
            .iter()
            .position(|&i| i == index)
            .map_or(-1, |p| p as isize),
        count: sel_array.len(),
    };
    let local_frame = frame - ctx.delay;
    let pos_min = read_vec3(
        props,
        "randomPositionMin",
        local_frame,
        [-10.0, -10.0, -10.0],
    );
    let pos_max = read_vec3(props, "randomPositionMax", local_frame, [10.0, 10.0, 10.0]);
    let rot_min = read_vec3(props, "randomRotationMin", local_frame, [0.0, 0.0, 0.0]);
    let rot_max = read_vec3(props, "randomRotationMax", local_frame, [0.0, 0.0, 0.0]);
    let scl_min = read_vec3(props, "randomScaleMin", local_frame, [1.0, 1.0, 1.0]);
    let scl_max = read_vec3(props, "randomScaleMax", local_frame, [1.0, 1.0, 1.0]);
    let remap = |u: f64| minimum + u * (maximum - minimum);
    for axis in 0..3 {
        let up = remap(random_unit(
            mode,
            index,
            0,
            axis as i32,
            base,
            local_frame,
            &opts,
        ));
        let ur = remap(random_unit(
            mode,
            index,
            1,
            axis as i32,
            base,
            local_frame,
            &opts,
        ));
        let us = remap(random_unit(
            mode,
            index,
            2,
            axis as i32,
            base,
            local_frame,
            &opts,
        ));
        out.position[axis] += (pos_min[axis] + (pos_max[axis] - pos_min[axis]) * up) * weight;
        out.rotation[axis] += (rot_min[axis] + (rot_max[axis] - rot_min[axis]) * ur) * weight;
        let target = scl_min[axis] + (scl_max[axis] - scl_min[axis]) * us;
        out.scale[axis] *= 1.0 + (target - 1.0) * weight;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_formula(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
    aux: &EvalAux,
) {
    if read_num(props, "formulaEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, EffectorKind::Formula, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let raw_strength = read_num(props, "formulaStrength", frame, 100.0) / 100.0;
    let field = apply_field_weight(1.0, props, EffectorKind::Formula, frame, base);
    let weight = raw_strength * field;
    if weight == 0.0 || weight.is_nan() {
        return;
    }
    let local_frame = frame - ctx.delay;
    let expr = read_str(props, "formulaExpression", frame, "sin(u*360*f+t*90)");
    let order = if list.order.is_empty() {
        vec![index]
    } else {
        list.order.clone()
    };
    let rank = order.iter().position(|&i| i == index).unwrap_or(0);
    let slots = order.len().max(1);
    let scope = FormulaScope {
        px: base[0],
        py: base[1],
        pz: base[2],
        rx: out.rotation[0] * 180.0 / PI,
        ry: out.rotation[1] * 180.0 / PI,
        rz: out.rotation[2] * 180.0 / PI,
        sx: out.scale[0],
        sy: out.scale[1],
        sz: out.scale[2],
        u: if slots > 1 {
            (rank as f64) / ((slots - 1) as f64)
        } else {
            0.0
        },
        v: 0.0,
        w: 0.0,
        id: index as f64,
        count: list.total as f64,
        falloff: field,
        t: (local_frame / eval_rate(aux)) * read_num(props, "formulaTime", frame, 1.0),
        f: read_num(props, "formulaFrequency", frame, 1.0),
    };
    let scalar = eval_formula(&expr, &scope);
    let minimum = read_num(props, "formulaMinimum", frame, 0.0) / 100.0;
    let maximum = read_num(props, "formulaMaximum", frame, 100.0) / 100.0;
    let remapped = minimum + scalar * (maximum - minimum);
    let pos_amp = read_vec3(props, "formulaPosition", local_frame, [0.0, 10.0, 0.0]);
    let rot_amp = read_vec3(props, "formulaRotation", local_frame, [0.0, 0.0, 0.0]);
    let scl_amp = read_vec3(props, "formulaScale", local_frame, [1.0, 1.0, 1.0]);
    add_scaled(&mut out.position, pos_amp, remapped * weight);
    add_scaled(&mut out.rotation, rot_amp, remapped * weight);
    mul_scale(
        &mut out.scale,
        [
            1.0 + (scl_amp[0] - 1.0) * remapped,
            1.0 + (scl_amp[1] - 1.0) * remapped,
            1.0 + (scl_amp[2] - 1.0) * remapped,
        ],
        weight,
    );
}

pub(super) fn evaluate_twist(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
) {
    if read_num(props, "twistEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, EffectorKind::Twist, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, EffectorKind::Twist, frame, base);
    if weight <= 0.0 {
        return;
    }
    let local_frame = frame - ctx.delay;
    let angle = read_num(props, "twistAngle", local_frame, 0.0);
    let axis = js_round(read_num(props, "twistAxis", local_frame, 2.0)) as i32;
    let order = if list.order.is_empty() {
        vec![index]
    } else {
        list.order.clone()
    };
    let slots = order.len().max(1);
    let twirl = match order.iter().position(|&i| i == index) {
        Some(at) if slots > 1 => angle * ((at as f64) / ((slots - 1) as f64) - 0.5),
        _ => 0.0,
    };
    let ax = match axis {
        0 => 0,
        1 => 1,
        _ => 2,
    };
    out.rotation[ax] += twirl * weight;
}

pub(super) fn evaluate_wrap(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
) {
    if read_num(props, "wrapEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, EffectorKind::Wrap, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, EffectorKind::Wrap, frame, base);
    if weight <= 0.0 {
        return;
    }
    let local_frame = frame - ctx.delay;
    let amplitude = read_num(props, "wrapAmplitude", local_frame, 0.0);
    let frequency = read_num(props, "wrapFrequency", local_frame, 1.0).max(0.0);
    let phase = read_num(props, "wrapPhase", local_frame, 0.0);
    let dir = js_round(read_num(props, "wrapDirection", local_frame, 1.0)) as i32;
    let order = if list.order.is_empty() {
        vec![index]
    } else {
        list.order.clone()
    };
    let slots = order.len().max(1);
    let u = match order.iter().position(|&i| i == index) {
        Some(at) if slots > 1 => (at as f64) / ((slots - 1) as f64),
        _ => 0.0,
    };
    let wave = amplitude * (2.0 * PI * frequency * u + phase).sin();
    let ax = match dir {
        0 => 0,
        2 => 2,
        _ => 1,
    };
    out.position[ax] += wave * weight;
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_orbit(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
    aux: &EvalAux,
) {
    if read_num(props, "orbitEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, EffectorKind::Orbit, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let weight = apply_field_weight(ctx.strength, props, EffectorKind::Orbit, frame, base);
    if weight <= 0.0 {
        return;
    }
    let local_frame = frame - ctx.delay;
    let angle = read_vec3(props, "orbitAngle", local_frame, [0.0, 0.0, 0.0]);
    let radius = read_num(props, "orbitRadius", local_frame, 100.0) / 100.0;
    let order = if list.order.is_empty() {
        vec![index]
    } else {
        list.order.clone()
    };
    let mut center = base;
    if let Some(shared) = &aux.bases {
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        let mut n = 0usize;
        for &gi in &order {
            if !is_letter_selected(&ctx.selected, gi) {
                continue;
            }
            if let Some(raw) = shared.get(gi) {
                cx += raw[0];
                cy += raw[1];
                cz += raw[2];
                n += 1;
            }
        }
        if n > 0 {
            let world_center = [cx / (n as f64), cy / (n as f64), cz / (n as f64)];
            center = match aux.self_world {
                Some(sw) => local_base_of(world_center, sw),
                None => world_center,
            };
        }
    }
    let rel = [
        (base[0] - center[0]) * radius,
        (base[1] - center[1]) * radius,
        (base[2] - center[2]) * radius,
    ];
    let spun = rotate_euler_xyz(rel, angle);
    let target = [
        center[0] + spun[0],
        center[1] + spun[1],
        center[2] + spun[2],
    ];
    out.position[0] += (target[0] - base[0] - out.position[0]) * weight;
    out.position[1] += (target[1] - base[1] - out.position[1]) * weight;
    out.position[2] += (target[2] - base[2] - out.position[2]) * weight;
    out.rotation[0] += angle[0] * weight;
    out.rotation[1] += angle[1] * weight;
    out.rotation[2] += angle[2] * weight;
}

struct SplineParams {
    length: f64,
    radius: f64,
    radius_end: f64,
    height: f64,
    turns: f64,
    plane: i32,
    reverse: bool,
    offset: f64,
    start: f64,
    end: f64,
    clamp: bool,
    tilt: [f64; 3],
}

fn sample_spline_path(path: i32, p: &SplineParams, u: f64) -> ([f64; 3], [f64; 3]) {
    let uu = if u.is_finite() { u } else { 0.0 };
    let (point, tangent, normal) = if path == 2 {
        let theta = 2.0 * PI * p.turns * uu;
        let r = p.radius + (p.radius_end - p.radius) * uu;
        let dtheta = 2.0 * PI * p.turns;
        let dr = p.radius_end - p.radius;
        (
            [r * theta.cos(), uu * p.height, r * theta.sin()],
            [
                dr * theta.cos() - r * theta.sin() * dtheta,
                p.height,
                dr * theta.sin() + r * theta.cos() * dtheta,
            ],
            [0.0, 0.0, 1.0],
        )
    } else if path == 1 {
        let theta = 2.0 * PI * uu;
        if p.plane == 1 {
            (
                [p.radius * theta.cos(), 0.0, p.radius * theta.sin()],
                [-theta.sin(), 0.0, theta.cos()],
                [0.0, 1.0, 0.0],
            )
        } else if p.plane == 2 {
            (
                [0.0, p.radius * theta.cos(), p.radius * theta.sin()],
                [0.0, -theta.sin(), theta.cos()],
                [1.0, 0.0, 0.0],
            )
        } else {
            (
                [p.radius * theta.cos(), p.radius * theta.sin(), 0.0],
                [-theta.sin(), theta.cos(), 0.0],
                [0.0, 0.0, 1.0],
            )
        }
    } else {
        (
            [(uu - 0.5) * p.length, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        )
    };
    let pivot = if path == 2 {
        [0.0, p.height / 2.0, 0.0]
    } else {
        [0.0, 0.0, 0.0]
    };
    let rel = [
        point[0] - pivot[0],
        point[1] - pivot[1],
        point[2] - pivot[2],
    ];
    let spun = rotate_euler_xyz(rel, p.tilt);
    let position = [pivot[0] + spun[0], pivot[1] + spun[1], pivot[2] + spun[2]];
    let t = rotate_euler_xyz(tangent, p.tilt);
    let n = rotate_euler_xyz(normal, p.tilt);
    let tl = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    let tl = if tl == 0.0 { 1.0 } else { tl };
    let tn = [t[0] / tl, t[1] / tl, t[2] / tl];
    let yv = [
        n[1] * tn[2] - n[2] * tn[1],
        n[2] * tn[0] - n[0] * tn[2],
        n[0] * tn[1] - n[1] * tn[0],
    ];
    let yl = (yv[0] * yv[0] + yv[1] * yv[1] + yv[2] * yv[2]).sqrt();
    let euler = if yl > 1e-9 {
        euler_from_basis(tn, [yv[0] / yl, yv[1] / yl, yv[2] / yl], n)
    } else {
        [0.0, 0.0, tn[1].atan2(tn[0])]
    };
    (position, euler)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_spline(
    props: &dyn EtPropReader,
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
    out: &mut LetterControls,
    sections: &BTreeMap<String, Vec<String>>,
    aux: &EvalAux,
) {
    if read_num(props, "splineEnable", frame, 1.0) == 0.0 {
        return;
    }
    let ctx = letter_context(props, EffectorKind::Spline, frame, index, list, sections);
    if !is_letter_selected(&ctx.selected, index) {
        return;
    }
    let raw_strength = read_num(props, "splineStrength", frame, 100.0) / 100.0;
    let field = apply_field_weight(1.0, props, EffectorKind::Spline, frame, base);
    let weight = raw_strength * field;
    if weight == 0.0 || weight.is_nan() {
        return;
    }
    let mode = js_round(read_num(props, "splineMode", frame, 0.0)) as i32;
    let transform = js_round(read_num(props, "splineTransform", frame, 0.0)) as i32;
    let local_frame = frame - ctx.delay;
    let path_raw = js_round(read_num(props, "splinePath", frame, 0.0)) as i32;
    let path = match path_raw {
        1 => 1,
        2 => 2,
        _ => 0,
    };
    let radius = read_num(props, "splineRadius", local_frame, 50.0).max(0.0);
    let p = SplineParams {
        length: read_num(props, "splineLength", local_frame, 100.0).max(0.0),
        radius,
        radius_end: read_num(props, "splineRadiusEnd", local_frame, radius).max(0.0),
        height: read_num(props, "splineHeight", local_frame, 100.0).max(0.0),
        turns: read_num(props, "splineTurns", local_frame, 1.0),
        plane: js_round(read_num(props, "splinePlane", local_frame, 0.0)) as i32,
        reverse: read_num(props, "splineReverse", local_frame, 0.0) != 0.0,
        offset: read_num(props, "splineOffset", local_frame, 0.0) / 100.0,
        start: read_num(props, "splineStart", local_frame, 0.0) / 100.0,
        end: read_num(props, "splineEnd", local_frame, 100.0) / 100.0,
        clamp: read_num(props, "splineClamp", local_frame, 1.0) != 0.0,
        tilt: read_vec3(props, "splineTilt", local_frame, [0.0, 0.0, 0.0]),
    };
    let order = if list.order.is_empty() {
        vec![index]
    } else {
        list.order.clone()
    };
    let slots = order.len().max(1);
    let rank = order.iter().position(|&i| i == index);
    let mut base_t = if mode == 1 {
        field
    } else if mode == 2 {
        if let Some((lo, hi)) = aux.extent
            && hi > lo
        {
            ((base[0] - lo) / (hi - lo)).clamp(0.0, 1.0)
        } else if slots > 1 {
            (rank.unwrap_or(0) as f64) / ((slots - 1) as f64)
        } else {
            0.0
        }
    } else if slots > 1 {
        (rank.unwrap_or(0) as f64) / ((slots - 1) as f64)
    } else {
        0.0
    };
    if p.reverse {
        base_t = 1.0 - base_t;
    }
    let pin = |x: f64| {
        if p.clamp {
            let lo = p.start.min(p.end);
            let hi = p.start.max(p.end);
            x.clamp(lo, hi)
        } else {
            x - x.floor()
        }
    };
    let place = |x: f64| pin(p.start + x * (p.end - p.start) + p.offset);
    let pos_toggle = read_num(props, "splinePosition", frame, 1.0) != 0.0;
    let rot_toggle = read_num(props, "splineRotation", frame, 1.0) != 0.0;
    if transform == 1 {
        let (a_pos, _) = sample_spline_path(path, &p, pin(p.start + p.offset));
        let (b_pos, _) = sample_spline_path(path, &p, pin(p.start));
        if pos_toggle {
            out.position[0] += (a_pos[0] - b_pos[0]) * weight;
            out.position[1] += (a_pos[1] - b_pos[1]) * weight;
            out.position[2] += (a_pos[2] - b_pos[2]) * weight;
        }
    } else if pos_toggle {
        let (smp_pos, _) = sample_spline_path(path, &p, place(base_t));
        out.position[0] += (smp_pos[0] - base[0] - out.position[0]) * weight;
        out.position[1] += (smp_pos[1] - base[1] - out.position[1]) * weight;
        out.position[2] += (smp_pos[2] - base[2] - out.position[2]) * weight;
    }
    if rot_toggle {
        let (_, smp_euler) = sample_spline_path(path, &p, place(base_t));
        let heading = read_num(props, "splineHeading", local_frame, 0.0);
        let pitch = read_num(props, "splinePitch", local_frame, 0.0);
        let bank = read_num(props, "splineBank", local_frame, 0.0);
        out.rotation[0] += (pitch + smp_euler[0]) * weight;
        out.rotation[1] += (heading + smp_euler[1]) * weight;
        out.rotation[2] += (bank + smp_euler[2]) * weight;
    }
}
