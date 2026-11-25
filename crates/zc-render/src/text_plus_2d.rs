//! Text+ 2D stacked per-character animator evaluation for native `LayerKind::Text2D`.
//!
//! Mirrors Zoidium's `plugins/text-plus/{text-plus-2d-properties.js,text-plus-2d-draw.js,
//! text-plus-controls.js,text-plus-utils.js}` and `zoidium/shared/random.js`:
//! - Up to 3 stacked character categories (`Character`, `Character 2`, `Character 3`)
//!   stored under `characterProperties` + `extraCharacterProperties[0..2]`.
//! - All 6 animator modes:
//!   1. `character-transform` (`Character Transform`)
//!   2. `gradient-transform` (`Gradient Transform`)
//!   3. `character-shake` (`Character Shake`)
//!   4. `selected-transform` (`Selected Character Transform`)
//!   5. `selected-shake` (`Selected Character Shake`)
//!   6. `random-scatter` (`Random Scatter`)
//! - Full CM3 keyframe evaluation (33 easings, Hold, Bezier handles) plus optional
//!   expression evaluation callback.

use std::collections::BTreeSet;
use std::f64::consts::TAU;

use serde_json::Value;
use zc_anim::{
    BezierSegment, EasingKind, InterpolationMode, eval_cubic_bezier, eval_easing,
    eval_segment_bezier,
};

/// Maximum stacked Text+ 2D character categories (matching Zoidium `MAX_STACK = 3`).
pub const MAX_TEXT_PLUS_2D_STACK: usize = 3;

/// Expression evaluation callback type `(expression_source, frame) -> Option<f64>`.
pub type ExprEvalFn<'a> = &'a dyn Fn(&str, f64) -> Option<f64>;

/// Per-character 2D transform output produced by [`evaluate_text_plus_2d_chars`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Char2DTransform {
    /// Character offset `[dx, dy]` in layer units.
    pub offset: [f32; 2],
    /// Character scale `[sx, sy]` (clamped `>= 0.001` per Zoidium).
    pub scale: [f32; 2],
    /// Combined Z-axis rotation + shake angle in radians.
    pub rotation_rad: f32,
    /// Combined opacity multiplier (`0.0..=1.0`).
    pub opacity: f32,
}

impl Default for Char2DTransform {
    fn default() -> Self {
        Self::neutral()
    }
}

impl Char2DTransform {
    /// Identity per-character transform.
    #[must_use]
    pub const fn neutral() -> Self {
        Self {
            offset: [0.0, 0.0],
            scale: [1.0, 1.0],
            rotation_rad: 0.0,
            opacity: 1.0,
        }
    }

    /// Returns `true` if this transform leaves the character unmodified.
    #[must_use]
    pub fn is_neutral(self) -> bool {
        self.offset[0] == 0.0
            && self.offset[1] == 0.0
            && self.scale[0] == 1.0
            && self.scale[1] == 1.0
            && self.rotation_rad == 0.0
            && (self.opacity - 1.0).abs() <= f32::EPSILON
    }
}

/// Text+ 2D animator mode matching Zoidium `MODES` (`text-plus-2d-properties.js:10-13`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextPlus2DMode {
    None = 0,
    CharacterTransform = 1,
    GradientTransform = 2,
    CharacterShake = 3,
    SelectedTransform = 4,
    SelectedShake = 5,
    RandomScatter = 6,
}

impl TextPlus2DMode {
    fn from_value(v: Option<&Value>, frame: f64, eval_expr: Option<ExprEvalFn<'_>>) -> Self {
        let Some(val) = v else {
            return Self::None;
        };
        if let Some(s) = val.as_str() {
            return Self::from_str_token(s);
        }
        if let Some(s) = val.get("value").and_then(Value::as_str) {
            return Self::from_str_token(s);
        }
        let idx = eval_anim_f64(Some(val), frame, 0.0, eval_expr).round() as i64;
        match idx {
            1 => Self::CharacterTransform,
            2 => Self::GradientTransform,
            3 => Self::CharacterShake,
            4 => Self::SelectedTransform,
            5 => Self::SelectedShake,
            6 => Self::RandomScatter,
            _ => Self::None,
        }
    }

    fn from_str_token(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "1" | "character-transform" | "character transform" => Self::CharacterTransform,
            "2" | "gradient-transform" | "gradient transform" => Self::GradientTransform,
            "3" | "character-shake" | "character shake" => Self::CharacterShake,
            "4" | "selected-transform" | "selected character transform" => Self::SelectedTransform,
            "5" | "selected-shake" | "selected character shake" => Self::SelectedShake,
            "6" | "random-scatter" | "random scatter" => Self::RandomScatter,
            _ => Self::None,
        }
    }
}

/// Delay ordering matching Zoidium `DELAY_ORDERS` (`["start", "end", "middle", "random"]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DelayOrder {
    Start,
    End,
    Middle,
    Random,
}

impl DelayOrder {
    fn from_value(v: Option<&Value>, frame: f64, eval_expr: Option<ExprEvalFn<'_>>) -> Self {
        let Some(val) = v else {
            return Self::Start;
        };
        if let Some(s) = val.as_str() {
            return Self::from_token(s);
        }
        if let Some(s) = val.get("value").and_then(Value::as_str) {
            return Self::from_token(s);
        }
        let idx = eval_anim_f64(Some(val), frame, 0.0, eval_expr).round() as i64;
        match idx {
            1 => Self::End,
            2 => Self::Middle,
            3 => Self::Random,
            _ => Self::Start,
        }
    }

    fn from_token(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "1" | "end" | "reverse" => Self::End,
            "2" | "middle" | "center out" | "center-out" => Self::Middle,
            "3" | "random" => Self::Random,
            _ => Self::Start,
        }
    }
}

/// Evaluates all stacked Text+ 2D character categories (`Character`, `Character 2`, `Character 3`)
/// for `char_count` characters at `frame`.
///
/// Accepts either:
/// - Zoidium's persisted `zoidiumTextPlus2D` object:
///   `{ "schemaVersion": 1, "characterProperties": { ... }, "extraCharacterProperties": [ ... ] }`
/// - A direct category object containing `{ "mode": ..., ... }`
/// - An array of up to 3 category objects.
#[must_use]
pub fn evaluate_text_plus_2d_chars(
    text_plus_json: &Value,
    char_count: usize,
    frame: f64,
    fps: f64,
    eval_expr: Option<ExprEvalFn<'_>>,
) -> Vec<Char2DTransform> {
    if char_count == 0 {
        return Vec::new();
    }
    let _ = fps;

    // Respect Zoidium schemaVersion guard (`text-plus-2d.js:18`): unknown future versions
    // pass through untouched rather than misinterpreting them.
    if let Some(ver) = text_plus_json.get("schemaVersion").and_then(Value::as_i64)
        && ver != 1
    {
        return vec![Char2DTransform::neutral(); char_count];
    }

    let categories = collect_categories(text_plus_json);
    if categories.is_empty() {
        return vec![Char2DTransform::neutral(); char_count];
    }

    let frame = if frame.is_finite() { frame } else { 0.0 };
    let mut results = vec![Char2DTransform::neutral(); char_count];

    for cat in categories.into_iter().take(MAX_TEXT_PLUS_2D_STACK) {
        let mode = TextPlus2DMode::from_value(cat.get("mode"), frame, eval_expr);
        if mode == TextPlus2DMode::None {
            continue;
        }
        let selected_indices = if matches!(
            mode,
            TextPlus2DMode::SelectedTransform | TextPlus2DMode::SelectedShake
        ) {
            let raw = eval_text_value(cat.get("characterNumbers"), frame, "1");
            Some(parse_character_selection(&raw, char_count))
        } else {
            None
        };

        for (char_idx, out) in results.iter_mut().enumerate() {
            let step = evaluate_category_for_char(
                cat,
                mode,
                char_idx,
                char_count,
                selected_indices.as_deref(),
                frame,
                eval_expr,
            );
            if !step.is_neutral() {
                *out = compose_char_transform(*out, step);
            }
        }
    }

    results
}

fn collect_categories(root: &Value) -> Vec<&Value> {
    let mut out = Vec::with_capacity(MAX_TEXT_PLUS_2D_STACK);
    if let Some(arr) = root.as_array() {
        for item in arr.iter().take(MAX_TEXT_PLUS_2D_STACK) {
            if item.is_object() {
                out.push(item);
            }
        }
        return out;
    }
    if let Some(obj) = root.as_object() {
        if let Some(first) = obj.get("characterProperties")
            && first.is_object()
        {
            out.push(first);
            if let Some(extras) = obj
                .get("extraCharacterProperties")
                .and_then(Value::as_array)
            {
                for extra in extras.iter().take(MAX_TEXT_PLUS_2D_STACK - 1) {
                    if extra.is_object() {
                        out.push(extra);
                    }
                }
            }
            return out;
        }
        if let Some(cats) = obj.get("categories").and_then(Value::as_array) {
            for item in cats.iter().take(MAX_TEXT_PLUS_2D_STACK) {
                if item.is_object() {
                    out.push(item);
                }
            }
            return out;
        }
        if obj.contains_key("mode") {
            out.push(root);
        }
    }
    out
}

/// Composes an earlier category transform `inner` with a later category `outer` applied on top.
///
/// Mirrors `text-plus-2d-draw.js:68-76`: the last category is applied outermost so each later
/// category transforms the character as the earlier categories left it.
fn compose_char_transform(inner: Char2DTransform, outer: Char2DTransform) -> Char2DTransform {
    let (sin_r, cos_r) = outer.rotation_rad.sin_cos();
    let sx_in = inner.offset[0] * outer.scale[0];
    let sy_in = inner.offset[1] * outer.scale[1];
    let rot_dx = cos_r * sx_in - sin_r * sy_in;
    let rot_dy = sin_r * sx_in + cos_r * sy_in;
    Char2DTransform {
        offset: [outer.offset[0] + rot_dx, outer.offset[1] + rot_dy],
        scale: [
            (inner.scale[0] * outer.scale[0]).max(0.001),
            (inner.scale[1] * outer.scale[1]).max(0.001),
        ],
        rotation_rad: inner.rotation_rad + outer.rotation_rad,
        opacity: (inner.opacity * outer.opacity).clamp(0.0, 1.0),
    }
}

fn evaluate_category_for_char(
    props: &Value,
    mode: TextPlus2DMode,
    index: usize,
    count: usize,
    selected: Option<&[usize]>,
    frame: f64,
    eval_expr: Option<ExprEvalFn<'_>>,
) -> Char2DTransform {
    match mode {
        TextPlus2DMode::None => Char2DTransform::neutral(),
        TextPlus2DMode::GradientTransform => {
            eval_gradient_transform(props, index, count, frame, eval_expr)
        }
        TextPlus2DMode::RandomScatter => eval_random_scatter(props, index, frame, eval_expr),
        TextPlus2DMode::CharacterTransform => {
            eval_character_transform(props, index, count, frame, eval_expr)
        }
        TextPlus2DMode::CharacterShake => {
            eval_character_shake(props, index, count, frame, eval_expr)
        }
        TextPlus2DMode::SelectedTransform => {
            let Some(sel) = selected else {
                return Char2DTransform::neutral();
            };
            let Some(pos) = sel.iter().position(|&i| i == index) else {
                return Char2DTransform::neutral();
            };
            eval_character_transform(props, pos, sel.len(), frame, eval_expr)
        }
        TextPlus2DMode::SelectedShake => {
            let Some(sel) = selected else {
                return Char2DTransform::neutral();
            };
            let Some(pos) = sel.iter().position(|&i| i == index) else {
                return Char2DTransform::neutral();
            };
            eval_character_shake(props, pos, sel.len(), frame, eval_expr)
        }
    }
}

fn eval_character_transform(
    props: &Value,
    index: usize,
    count: usize,
    frame: f64,
    eval_expr: Option<ExprEvalFn<'_>>,
) -> Char2DTransform {
    let delay = configured_character_offset(
        props,
        frame,
        index,
        count,
        "delayPerCharacter",
        "delayOrder",
        "randomSeed",
        eval_expr,
    );
    let t_frame = (frame - delay).max(0.0);
    let pos = eval_anim_vec2(props.get("charPosition"), t_frame, [0.0, 0.0], eval_expr);
    let scale = eval_anim_vec2(props.get("charScale"), t_frame, [1.0, 1.0], eval_expr);
    let rot = eval_anim_rot_z(props.get("charRotation"), t_frame, 0.0, eval_expr);
    let opacity = eval_anim_f64(
        props.get("charOpacity").or_else(|| props.get("opacity")),
        t_frame,
        1.0,
        eval_expr,
    );

    Char2DTransform {
        offset: [pos[0] as f32, pos[1] as f32],
        scale: [(scale[0] as f32).max(0.001), (scale[1] as f32).max(0.001)],
        rotation_rad: rot as f32,
        opacity: (opacity as f32).clamp(0.0, 1.0),
    }
}

fn eval_gradient_transform(
    props: &Value,
    index: usize,
    count: usize,
    frame: f64,
    eval_expr: Option<ExprEvalFn<'_>>,
) -> Char2DTransform {
    let amount = if count <= 1 {
        0.0
    } else {
        (index.min(count - 1) as f64) / ((count - 1) as f64)
    };
    let p0 = eval_anim_vec2(props.get("firstPosition"), frame, [0.0, 0.0], eval_expr);
    let p1 = eval_anim_vec2(props.get("lastPosition"), frame, [0.0, 0.0], eval_expr);
    let s0 = eval_anim_vec2(props.get("firstScale"), frame, [1.0, 1.0], eval_expr);
    let s1 = eval_anim_vec2(props.get("lastScale"), frame, [1.0, 1.0], eval_expr);
    let r0 = eval_anim_rot_z(props.get("firstRotation"), frame, 0.0, eval_expr);
    let r1 = eval_anim_rot_z(props.get("lastRotation"), frame, 0.0, eval_expr);
    let o0 = eval_anim_f64(props.get("firstOpacity"), frame, 1.0, eval_expr);
    let o1 = eval_anim_f64(props.get("lastOpacity"), frame, 1.0, eval_expr);

    Char2DTransform {
        offset: [
            lerp(p0[0], p1[0], amount) as f32,
            lerp(p0[1], p1[1], amount) as f32,
        ],
        scale: [
            (lerp(s0[0], s1[0], amount) as f32).max(0.001),
            (lerp(s0[1], s1[1], amount) as f32).max(0.001),
        ],
        rotation_rad: lerp(r0, r1, amount) as f32,
        opacity: (lerp(o0, o1, amount) as f32).clamp(0.0, 1.0),
    }
}

fn eval_character_shake(
    props: &Value,
    index: usize,
    count: usize,
    frame: f64,
    eval_expr: Option<ExprEvalFn<'_>>,
) -> Char2DTransform {
    let phase_offset = configured_character_offset(
        props,
        frame,
        index,
        count,
        "shakeDelayPerCharacter",
        "shakeDelayOrder",
        "shakeRandomSeed",
        eval_expr,
    );
    let amplitude = eval_anim_f64(props.get("shakeAmplitude"), frame, 0.0, eval_expr);
    let period = eval_anim_f64(props.get("shakePeriod"), frame, 24.0, eval_expr);
    let phase =
        frame + eval_anim_f64(props.get("shakePhase"), frame, 0.0, eval_expr) - phase_offset;
    let angle = shake_angle(amplitude, period, phase);

    Char2DTransform {
        offset: [0.0, 0.0],
        scale: [1.0, 1.0],
        rotation_rad: angle as f32,
        opacity: 1.0,
    }
}

fn eval_random_scatter(
    props: &Value,
    index: usize,
    frame: f64,
    eval_expr: Option<ExprEvalFn<'_>>,
) -> Char2DTransform {
    let strength = eval_anim_f64(props.get("amount"), frame, 0.0, eval_expr).max(0.0);
    if strength == 0.0 {
        return Char2DTransform::neutral();
    }
    let pos_min = eval_anim_vec2(props.get("positionMin"), frame, [-10.0, -10.0], eval_expr);
    let pos_max = eval_anim_vec2(props.get("positionMax"), frame, [10.0, 10.0], eval_expr);
    let rot_min = eval_anim_rot_z(props.get("rotationMin"), frame, 0.0, eval_expr);
    let rot_max = eval_anim_rot_z(props.get("rotationMax"), frame, 0.0, eval_expr);
    let scale_min = eval_anim_vec2(props.get("scaleMin"), frame, [1.0, 1.0], eval_expr);
    let scale_max = eval_anim_vec2(props.get("scaleMax"), frame, [1.0, 1.0], eval_expr);
    let seed = eval_anim_f64(props.get("seed"), frame, 1.0, eval_expr);

    // Channel layout matches Zoidium `text-plus-utils.js:255-275`:
    // position -> 20, 21, (22); rotation -> (23, 24), 25 (Z); scale -> 26, 27, (28).
    let px = lerp(
        pos_min[0].min(pos_max[0]),
        pos_min[0].max(pos_max[0]),
        text_seeded_random(seed, index as i32, 20),
    ) * strength;
    let py = lerp(
        pos_min[1].min(pos_max[1]),
        pos_min[1].max(pos_max[1]),
        text_seeded_random(seed, index as i32, 21),
    ) * strength;
    let rz = lerp(
        rot_min.min(rot_max),
        rot_min.max(rot_max),
        text_seeded_random(seed, index as i32, 25),
    ) * strength;
    let sx_t = lerp(
        scale_min[0].min(scale_max[0]),
        scale_min[0].max(scale_max[0]),
        text_seeded_random(seed, index as i32, 26),
    );
    let sy_t = lerp(
        scale_min[1].min(scale_max[1]),
        scale_min[1].max(scale_max[1]),
        text_seeded_random(seed, index as i32, 27),
    );

    Char2DTransform {
        offset: [px as f32, py as f32],
        scale: [
            (lerp(1.0, sx_t, strength) as f32).max(0.001),
            (lerp(1.0, sy_t, strength) as f32).max(0.001),
        ],
        rotation_rad: rz as f32,
        opacity: 1.0,
    }
}

#[allow(clippy::too_many_arguments)]
fn configured_character_offset(
    props: &Value,
    frame: f64,
    index: usize,
    count: usize,
    delay_key: &str,
    order_key: &str,
    seed_key: &str,
    eval_expr: Option<ExprEvalFn<'_>>,
) -> f64 {
    let step = eval_anim_f64(props.get(delay_key), frame, 0.0, eval_expr).max(0.0);
    if step == 0.0 {
        return 0.0;
    }
    let order = DelayOrder::from_value(props.get(order_key), frame, eval_expr);
    let seed = eval_anim_f64(props.get(seed_key), frame, 1.0, eval_expr);
    delay_rank(order, index, count, seed) as f64 * step
}

/// Exact Zoidium `getDelayRank` (`text-plus-utils.js:307-328`).
fn delay_rank(order: DelayOrder, index: usize, count: usize, seed: f64) -> usize {
    if count <= 1 {
        return 0;
    }
    let position = index.min(count - 1);
    match order {
        DelayOrder::Start => position,
        DelayOrder::End => count - 1 - position,
        DelayOrder::Middle => {
            let center = (count - 1) / 2;
            if position == center {
                0
            } else if position > center {
                (position - center) * 2 - 1
            } else {
                (center - position) * 2
            }
        }
        DelayOrder::Random => {
            let perm = random_order(count, seed);
            perm.iter().position(|&v| v == position).unwrap_or(position)
        }
    }
}

/// Exact Zoidium `getRandomOrder` (`text-plus-utils.js:286-304`).
fn random_order(count: usize, seed: f64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..count).collect();
    for i in (1..count).rev() {
        let r = text_seeded_random(seed, i as i32, 11);
        let swap = ((r * (i + 1) as f64).floor() as usize).min(i);
        order.swap(i, swap);
    }
    order
}

/// Exact Zoidium `getShakeAngle` (`text-plus-utils.js:352-359`).
#[must_use]
pub fn shake_angle(amplitude_rad: f64, period_frames: f64, phase: f64) -> f64 {
    if !period_frames.is_finite()
        || period_frames <= 0.0
        || amplitude_rad == 0.0
        || !amplitude_rad.is_finite()
    {
        return 0.0;
    }
    amplitude_rad * (TAU * phase / period_frames).sin()
}

/// Parses 1-based character indices matching Zoidium `parseCharacterSelection`
/// (`text-plus-controls.js:183-211`), returning sorted deduplicated 0-based indices.
#[must_use]
pub fn parse_character_selection(raw: &str, count: usize) -> Vec<usize> {
    let mut seen = BTreeSet::new();
    for token in raw.split(|c: char| {
        c.is_whitespace() || matches!(c, ',' | '.' | '\u{3002}' | '\u{FF0C}' | '\u{FF0E}')
    }) {
        if token.is_empty() || !token.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        if let Ok(num) = token.parse::<usize>()
            && num >= 1
            && num <= count
        {
            seen.insert(num - 1);
        }
    }
    seen.into_iter().collect()
}

/// Bit-exact port of Zoidium `ZoidiumPluginApis.random.textSeedHash` (`zoidium/shared/random.js:44-55`).
#[must_use]
pub fn text_seed_hash(seed: f64) -> i32 {
    let numeric_seed = if seed.is_finite() { seed } else { 1.0 };
    if numeric_seed.fract() == 0.0
        && numeric_seed >= f64::from(i32::MIN)
        && numeric_seed <= f64::from(i32::MAX)
    {
        return numeric_seed as i32;
    }
    let bits = numeric_seed.to_le_bytes();
    let lo = u32::from_le_bytes([bits[0], bits[1], bits[2], bits[3]]);
    let hi = u32::from_le_bytes([bits[4], bits[5], bits[6], bits[7]]);
    let mut value = lo ^ hi.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x045d_9f3b);
    value ^= value >> 16;
    value as i32
}

/// Bit-exact port of Zoidium `ZoidiumPluginApis.random.textSeededRandom` (`zoidium/shared/random.js:57-64`).
#[must_use]
pub fn text_seeded_random(seed: f64, index: i32, channel: i32) -> f64 {
    let mut value =
        (text_seed_hash(seed) as u32) ^ (index.wrapping_add(1) as u32).wrapping_mul(0x045d_9f3b);
    value ^= (channel.wrapping_add(1) as u32).wrapping_mul(0x27d4_eb2d);
    value = (value ^ (value >> 16)).wrapping_mul(0x045d_9f3b);
    value = (value ^ (value >> 16)).wrapping_mul(0x045d_9f3b);
    value ^= value >> 16;
    f64::from(value) / 4_294_967_296.0
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Full CM3 Keyframe / Easing / Bezier / Expression Property Evaluation
// ---------------------------------------------------------------------------

fn eval_text_value(val: Option<&Value>, frame: f64, default: &str) -> String {
    let Some(v) = val else {
        return default.to_owned();
    };
    if let Some(s) = v.as_str() {
        return s.to_owned();
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        let mut pts: Vec<(f64, &str)> = kfs
            .iter()
            .filter_map(|k| {
                let f = k.get("frame").and_then(Value::as_f64)?;
                let s = k.get("value").and_then(Value::as_str)?;
                Some((f, s))
            })
            .collect();
        if !pts.is_empty() {
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut prev = pts[0].1;
            for (kf_frame, s) in pts {
                if kf_frame > frame {
                    break;
                }
                prev = s;
            }
            return prev.to_owned();
        }
    }
    if let Some(inner) = v.get("value") {
        return eval_text_value(Some(inner), frame, default);
    }
    default.to_owned()
}

fn eval_anim_f64(
    val: Option<&Value>,
    frame: f64,
    default: f64,
    eval_expr: Option<ExprEvalFn<'_>>,
) -> f64 {
    let Some(v) = val else {
        return default;
    };
    if let Some(n) = v.as_f64() {
        return if n.is_finite() { n } else { default };
    }
    if let Some(b) = v.as_bool() {
        return if b { 1.0 } else { 0.0 };
    }
    if let Some(s) = v.as_str()
        && let Ok(n) = s.trim().parse::<f64>()
        && n.is_finite()
    {
        return n;
    }
    if let Some(expr) = v.get("expression").and_then(Value::as_str)
        && !expr.trim().is_empty()
        && let Some(cb) = eval_expr
        && let Some(out) = cb(expr, frame)
        && out.is_finite()
    {
        return out;
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        return eval_keyframe_scalar(kfs, frame, default);
    }
    if let Some(inner) = v.get("value") {
        return eval_anim_f64(Some(inner), frame, default, eval_expr);
    }
    default
}

fn eval_anim_vec2(
    val: Option<&Value>,
    frame: f64,
    default: [f64; 2],
    eval_expr: Option<ExprEvalFn<'_>>,
) -> [f64; 2] {
    let Some(v) = val else {
        return default;
    };
    if let Some(arr) = v.as_array()
        && arr.len() >= 2
    {
        return [
            arr[0]
                .as_f64()
                .filter(|x| x.is_finite())
                .unwrap_or(default[0]),
            arr[1]
                .as_f64()
                .filter(|x| x.is_finite())
                .unwrap_or(default[1]),
        ];
    }
    if let Some(objs) = v.get("objects").and_then(Value::as_array)
        && objs.len() >= 2
    {
        return [
            eval_anim_f64(Some(&objs[0]), frame, default[0], eval_expr),
            eval_anim_f64(Some(&objs[1]), frame, default[1], eval_expr),
        ];
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        let x = eval_keyframe_component(kfs, frame, 0, default[0]);
        let y = eval_keyframe_component(kfs, frame, 1, default[1]);
        return [x, y];
    }
    if let Some(inner) = v.get("value") {
        return eval_anim_vec2(Some(inner), frame, default, eval_expr);
    }
    default
}

fn eval_anim_rot_z(
    val: Option<&Value>,
    frame: f64,
    default: f64,
    eval_expr: Option<ExprEvalFn<'_>>,
) -> f64 {
    let Some(v) = val else {
        return default;
    };
    if let Some(arr) = v.as_array() {
        let idx = if arr.len() >= 3 { 2 } else { 0 };
        return arr
            .get(idx)
            .and_then(Value::as_f64)
            .filter(|x| x.is_finite())
            .unwrap_or(default);
    }
    if let Some(objs) = v.get("objects").and_then(Value::as_array) {
        let idx = if objs.len() >= 3 { 2 } else { 0 };
        return eval_anim_f64(objs.get(idx), frame, default, eval_expr);
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        if kfs[0].get("value").is_some_and(Value::is_array) {
            let len = kfs[0]
                .get("value")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            let idx = if len >= 3 { 2 } else { 0 };
            return eval_keyframe_component(kfs, frame, idx, default);
        }
        return eval_keyframe_scalar(kfs, frame, default);
    }
    eval_anim_f64(Some(v), frame, default, eval_expr)
}

fn parse_normalized_bezier(kf: &Value) -> Option<[f64; 4]> {
    for key in ["bezier", "controlPoints"] {
        if let Some(v) = kf.get(key) {
            if let Some(arr) = v.as_array()
                && arr.len() == 4
                && arr.iter().all(Value::is_number)
            {
                return Some([
                    arr[0].as_f64()?,
                    arr[1].as_f64()?,
                    arr[2].as_f64()?,
                    arr[3].as_f64()?,
                ]);
            }
            if let Some(obj) = v.as_object()
                && let (Some(x1), Some(y1), Some(x2), Some(y2)) = (
                    obj.get("x1").and_then(Value::as_f64),
                    obj.get("y1").and_then(Value::as_f64),
                    obj.get("x2").and_then(Value::as_f64),
                    obj.get("y2").and_then(Value::as_f64),
                )
            {
                return Some([x1, y1, x2, y2]);
            }
        }
    }
    None
}

fn parse_handle_pair(kf: &Value, is_out: bool) -> Option<[f64; 2]> {
    let direct_key = if is_out { "outTangent" } else { "inTangent" };
    if let Some(arr) = kf.get(direct_key).and_then(Value::as_array)
        && arr.len() >= 2
    {
        return Some([arr[0].as_f64()?, arr[1].as_f64()?]);
    }
    if let Some(cp) = kf.get("controlPoints") {
        if let Some(arr) = cp.as_array()
            && arr.len() == 2
            && arr[0].is_array()
            && arr[1].is_array()
        {
            let h = arr[usize::from(is_out)].as_array()?;
            if h.len() >= 2 {
                return Some([h[0].as_f64()?, h[1].as_f64()?]);
            }
        }
        let sub = if is_out { "out" } else { "in" };
        if let Some(arr) = cp.get(sub).and_then(Value::as_array)
            && arr.len() >= 2
        {
            return Some([arr[0].as_f64()?, arr[1].as_f64()?]);
        }
    }
    None
}

fn eval_segment_value(k0: &Value, v0: f64, k1: &Value, v1: f64, frame: f64) -> f64 {
    let f0 = k0.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
    let f1 = k1.get("frame").and_then(Value::as_f64).unwrap_or(f0);
    let span = f1 - f0;
    if span <= 1e-12 || frame <= f0 {
        return v0;
    }
    if frame >= f1 {
        return v1;
    }
    let tween_code = k0
        .get("tween")
        .and_then(Value::as_u64)
        .map_or(1_u16, |t| t.min(u64::from(u16::MAX)) as u16);
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(tween_code);
    if mode == InterpolationMode::Discrete || easing == EasingKind::Hold {
        return v0;
    }

    let u = ((frame - f0) / span).clamp(0.0, 1.0);
    if let Some([x1, y1, x2, y2]) = parse_normalized_bezier(k0) {
        let e = eval_cubic_bezier(x1, y1, x2, y2, u);
        return lerp(v0, v1, e);
    }
    if mode == InterpolationMode::Bezier
        || parse_handle_pair(k0, true).is_some()
        || parse_handle_pair(k1, false).is_some()
    {
        let out_t = parse_handle_pair(k0, true);
        let in_t = parse_handle_pair(k1, false);
        if out_t.is_some() || in_t.is_some() {
            return eval_segment_bezier(
                BezierSegment {
                    start_frame: f0,
                    start_value: v0,
                    out_tangent: out_t.unwrap_or([span / 3.0, (v1 - v0) / 3.0]),
                    end_frame: f1,
                    end_value: v1,
                    in_tangent: in_t.unwrap_or([-span / 3.0, -(v1 - v0) / 3.0]),
                },
                frame,
            );
        }
    }

    let e = eval_easing(easing, u);
    lerp(v0, v1, e)
}

fn eval_keyframe_scalar(kfs: &[Value], frame: f64, default: f64) -> f64 {
    let mut sorted: Vec<(f64, f64, &Value)> = kfs
        .iter()
        .filter_map(|k| {
            let f = k.get("frame").and_then(Value::as_f64)?;
            let v = k.get("value").and_then(|v| {
                v.as_f64()
                    .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
            })?;
            Some((f, v, k))
        })
        .collect();
    if sorted.is_empty() {
        return default;
    }
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    if frame <= sorted[0].0 {
        return sorted[0].1;
    }
    if frame >= sorted[sorted.len() - 1].0 {
        return sorted[sorted.len() - 1].1;
    }
    for w in sorted.windows(2) {
        if frame <= w[1].0 {
            return eval_segment_value(w[0].2, w[0].1, w[1].2, w[1].1, frame);
        }
    }
    sorted[sorted.len() - 1].1
}

fn eval_keyframe_component(kfs: &[Value], frame: f64, comp: usize, default: f64) -> f64 {
    let mut sorted: Vec<(f64, f64, &Value)> = kfs
        .iter()
        .filter_map(|k| {
            let f = k.get("frame").and_then(Value::as_f64)?;
            let arr = k.get("value").and_then(Value::as_array)?;
            let v = arr.get(comp).and_then(Value::as_f64)?;
            Some((f, v, k))
        })
        .collect();
    if sorted.is_empty() {
        return default;
    }
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    if frame <= sorted[0].0 {
        return sorted[0].1;
    }
    if frame >= sorted[sorted.len() - 1].0 {
        return sorted[sorted.len() - 1].1;
    }
    for w in sorted.windows(2) {
        if frame <= w[1].0 {
            return eval_segment_value(w[0].2, w[0].1, w[1].2, w[1].1, frame);
        }
    }
    sorted[sorted.len() - 1].1
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn character_transform_delay_orders_and_easings() {
        // Keyframed charPosition from [0, 0] at frame 0 to [100, 0] at frame 10 with QuadIn (tween=2)
        // and delayPerCharacter = 5 in Forward (0), Reverse (1), Center-out (2), Random (3).
        let payload = json!({
            "schemaVersion": 1,
            "characterProperties": {
                "mode": 1,
                "charPosition": {
                    "animated": true,
                    "keyframes": [
                        { "frame": 0, "value": [0.0, 0.0], "tween": 2 },
                        { "frame": 10, "value": [100.0, 0.0], "tween": 1 }
                    ]
                },
                "charScale": [1.5, 0.75],
                "charRotation": 0.5,
                "delayPerCharacter": 5.0,
                "delayOrder": 0
            }
        });

        let out = evaluate_text_plus_2d_chars(&payload, 3, 10.0, 30.0, None);
        assert_eq!(out.len(), 3);
        // Char 0: delayed frame 10 -> x=100
        assert!((out[0].offset[0] - 100.0).abs() < 1e-4);
        // Char 1: delayed frame 5 -> QuadIn(0.5) = 0.25 -> x=25
        assert!((out[1].offset[0] - 25.0).abs() < 1e-4);
        // Char 2: delayed frame 0 -> x=0
        assert!(out[2].offset[0].abs() < 1e-4);
        assert!((out[0].scale[0] - 1.5).abs() < 1e-4);
        assert!((out[0].rotation_rad - 0.5).abs() < 1e-4);

        // Reverse order: char 2 moves first (frame 10 -> x=100), char 0 stays at 0
        let mut rev = payload.clone();
        rev["characterProperties"]["delayOrder"] = json!(1);
        let out_rev = evaluate_text_plus_2d_chars(&rev, 3, 10.0, 30.0, None);
        assert!(out_rev[0].offset[0].abs() < 1e-4);
        assert!((out_rev[2].offset[0] - 100.0).abs() < 1e-4);

        // Center-out order: char 1 is center (rank 0 -> x=100)
        let mut mid = payload.clone();
        mid["characterProperties"]["delayOrder"] = json!(2);
        let out_mid = evaluate_text_plus_2d_chars(&mid, 3, 10.0, 30.0, None);
        assert!((out_mid[1].offset[0] - 100.0).abs() < 1e-4);
        assert!((out_mid[2].offset[0] - 25.0).abs() < 1e-4);
        assert!(out_mid[0].offset[0].abs() < 1e-4);
    }

    #[test]
    fn gradient_transform_interpolates_first_to_last() {
        let payload = json!({
            "schemaVersion": 1,
            "characterProperties": {
                "mode": 2,
                "firstPosition": [-40.0, 0.0],
                "lastPosition": [40.0, 20.0],
                "firstScale": [0.5, 0.5],
                "lastScale": [1.5, 1.5],
                "firstRotation": -0.4,
                "lastRotation": 0.4
            }
        });
        let out = evaluate_text_plus_2d_chars(&payload, 5, 0.0, 30.0, None);
        assert_eq!(out.len(), 5);
        assert!((out[0].offset[0] + 40.0).abs() < 1e-4);
        assert!((out[2].offset[0]).abs() < 1e-4);
        assert!((out[2].offset[1] - 10.0).abs() < 1e-4);
        assert!((out[2].scale[0] - 1.0).abs() < 1e-4);
        assert!(out[2].rotation_rad.abs() < 1e-4);
        assert!((out[4].offset[0] - 40.0).abs() < 1e-4);
        assert!((out[4].rotation_rad - 0.4).abs() < 1e-4);
    }

    #[test]
    fn character_shake_and_selected_animators() {
        // Character Shake: period=24, amplitude=1.0, phase=6 -> sin(pi/2) = 1.0 for char 0,
        // phaseOffset=6 -> char 1 has phase=0 -> sin(0) = 0.0.
        let shake = json!({
            "schemaVersion": 1,
            "characterProperties": {
                "mode": 3,
                "shakeAmplitude": 1.0,
                "shakePeriod": 24.0,
                "shakePhase": 6.0,
                "shakeDelayPerCharacter": 6.0
            }
        });
        let out = evaluate_text_plus_2d_chars(&shake, 3, 0.0, 30.0, None);
        assert!((out[0].rotation_rad - 1.0).abs() < 1e-4);
        assert!(out[1].rotation_rad.abs() < 1e-4);
        assert!((out[2].rotation_rad + 1.0).abs() < 1e-4);

        // Selected Character Transform: only characters 2 and 4 (1-based) move.
        let sel_tf = json!({
            "schemaVersion": 1,
            "characterProperties": {
                "mode": 4,
                "characterNumbers": "2, 4",
                "charPosition": [0.0, 30.0]
            }
        });
        let out_sel = evaluate_text_plus_2d_chars(&sel_tf, 5, 0.0, 30.0, None);
        assert!(out_sel[0].is_neutral());
        assert!((out_sel[1].offset[1] - 30.0).abs() < 1e-4);
        assert!(out_sel[2].is_neutral());
        assert!((out_sel[3].offset[1] - 30.0).abs() < 1e-4);
        assert!(out_sel[4].is_neutral());

        // Selected Character Shake: only character 3 (1-based) shakes.
        let sel_shake = json!({
            "schemaVersion": 1,
            "characterProperties": {
                "mode": 5,
                "characterNumbers": "3",
                "shakeAmplitude": 0.5,
                "shakePeriod": 24.0,
                "shakePhase": 6.0
            }
        });
        let out_ss = evaluate_text_plus_2d_chars(&sel_shake, 4, 0.0, 30.0, None);
        assert!(out_ss[0].is_neutral());
        assert!(out_ss[1].is_neutral());
        assert!((out_ss[2].rotation_rad - 0.5).abs() < 1e-4);
        assert!(out_ss[3].is_neutral());
    }

    #[test]
    fn random_scatter_determinism_and_stacked_three_categories_with_expression() {
        let scatter = json!({
            "schemaVersion": 1,
            "characterProperties": {
                "mode": 6,
                "amount": 1.0,
                "positionMin": [-20.0, -10.0],
                "positionMax": [20.0, 10.0],
                "rotationMin": -0.5,
                "rotationMax": 0.5,
                "scaleMin": [0.5, 0.5],
                "scaleMax": [1.5, 1.5],
                "seed": 7.0
            },
            "extraCharacterProperties": [
                {
                    "mode": 1,
                    "charPosition": {
                        "objects": [
                            { "expression": "frame * 2" },
                            { "value": 10.0 }
                        ]
                    },
                    "charScale": [1.0, 1.0],
                    "charRotation": 0.0
                },
                {
                    "mode": 3,
                    "shakeAmplitude": 0.25,
                    "shakePeriod": 24.0,
                    "shakePhase": 6.0
                }
            ]
        });

        let eval_cb = |expr: &str, f: f64| -> Option<f64> {
            if expr == "frame * 2" {
                Some(f * 2.0)
            } else {
                None
            }
        };

        let a = evaluate_text_plus_2d_chars(&scatter, 4, 5.0, 30.0, Some(&eval_cb));
        let b = evaluate_text_plus_2d_chars(&scatter, 4, 5.0, 30.0, Some(&eval_cb));
        assert_eq!(a, b);
        // Scatter produces distinct offsets per character, category 2 adds [10, 10],
        // category 3 rotates outermost by +0.25 rad.
        assert_ne!(a[0].offset, a[1].offset);
        assert_ne!(a[0].rotation_rad, a[1].rotation_rad);

        // Verify stacked category 3 frame-driven shake rotation composes onto 2-category result.
        let mut two_cats = scatter.clone();
        two_cats["extraCharacterProperties"][1]["mode"] = json!(0);
        let two = evaluate_text_plus_2d_chars(&two_cats, 4, 5.0, 30.0, Some(&eval_cb));
        let expected_shake = shake_angle(0.25, 24.0, 5.0 + 6.0) as f32;
        assert!((a[0].rotation_rad - (two[0].rotation_rad + expected_shake)).abs() < 1e-4);
    }

    #[test]
    fn hold_and_bezier_keyframes_in_text_plus_2d() {
        let payload = json!({
            "schemaVersion": 1,
            "characterProperties": {
                "mode": 1,
                "charPosition": {
                    "objects": [
                        {
                            "keyframes": [
                                { "frame": 0, "value": 10.0, "tween": 0 },
                                { "frame": 10, "value": 90.0, "tween": 1 }
                            ]
                        },
                        {
                            "keyframes": [
                                { "frame": 0, "value": 0.0, "tween": 257, "bezier": [0.0, 1.0, 0.0, 1.0] },
                                { "frame": 10, "value": 100.0, "tween": 1 }
                            ]
                        }
                    ]
                }
            }
        });
        let out = evaluate_text_plus_2d_chars(&payload, 1, 5.0, 30.0, None);
        // X is Hold (tween=0): stays at 10.0 until frame 10.
        assert!((out[0].offset[0] - 10.0).abs() < 1e-4);
        // Y is fast-out cubic Bezier (0,1,0,1): at t=0.5, value is significantly > 50.
        assert!(
            out[0].offset[1] > 75.0,
            "expected bezier y > 75, got {}",
            out[0].offset[1]
        );
    }

    #[test]
    fn character_shake_and_selected_shake_oscillate_across_frames() {
        let shake = json!({
            "schemaVersion": 1,
            "characterProperties": {
                "mode": 3,
                "shakeAmplitude": 0.45,
                "shakePeriod": 24.0,
                "shakePhase": 0.0,
                "shakeDelayPerCharacter": 3.0
            }
        });
        let f0 = evaluate_text_plus_2d_chars(&shake, 4, 0.0, 30.0, None);
        let f5 = evaluate_text_plus_2d_chars(&shake, 4, 5.0, 30.0, None);
        let f15 = evaluate_text_plus_2d_chars(&shake, 4, 15.0, 30.0, None);
        let f30 = evaluate_text_plus_2d_chars(&shake, 4, 30.0, 30.0, None);

        // Character 0 must produce distinct angles at frames 0, 5, 15, 30.
        let r = [
            f0[0].rotation_rad,
            f5[0].rotation_rad,
            f15[0].rotation_rad,
            f30[0].rotation_rad,
        ];
        assert!(r[0].abs() < 1e-4, "f0 expected ~0, got {}", r[0]);
        assert!(r[1] > 0.35, "f5 expected positive peak, got {}", r[1]);
        assert!(r[2] < -0.25, "f15 expected negative swing, got {}", r[2]);
        assert!(
            (r[3] - 0.45).abs() < 1e-4,
            "f30 expected +amplitude (sin(2.5*pi)), got {}",
            r[3]
        );
        for i in 0..r.len() {
            for j in (i + 1)..r.len() {
                assert!(
                    (r[i] - r[j]).abs() > 0.01,
                    "frames {i} and {j} must be distinct: {r:?}"
                );
            }
        }
        // Per-character delay must stagger character 1 relative to character 0.
        assert!((f5[0].rotation_rad - f5[1].rotation_rad).abs() > 0.1);

        // Selected Shake (mode 5) must also oscillate across frames on selected chars only.
        let sel_shake = json!({
            "schemaVersion": 1,
            "characterProperties": {
                "mode": 5,
                "characterNumbers": "2",
                "shakeAmplitude": 0.6,
                "shakePeriod": 20.0,
                "shakePhase": 0.0
            }
        });
        let s0 = evaluate_text_plus_2d_chars(&sel_shake, 3, 0.0, 30.0, None);
        let s5 = evaluate_text_plus_2d_chars(&sel_shake, 3, 5.0, 30.0, None);
        let s15 = evaluate_text_plus_2d_chars(&sel_shake, 3, 15.0, 30.0, None);
        assert!(s5[0].is_neutral() && s5[2].is_neutral());
        assert!(s0[1].rotation_rad.abs() < 1e-4);
        assert!((s5[1].rotation_rad - 0.6).abs() < 1e-4);
        assert!((s15[1].rotation_rad + 0.6).abs() < 1e-4);
    }
}
