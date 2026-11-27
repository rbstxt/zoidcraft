use super::*;

/// A single parsed keyframe in an ExpressText+ [`EtPropertyTrack`].
#[derive(Debug, Clone, PartialEq)]
pub struct EtKeyframe {
    pub frame: f64,
    pub value: EtValue,
    /// Raw CM3 tween bitfield (`low byte = easing index`, `high byte & 1 => cubic Bezier`).
    pub tween: u32,
    pub easing_index: u8,
    pub is_bezier: bool,
    /// `[in_handle, out_handle]` relative `(delta_frame, delta_value)`.
    pub control_points: [[f64; 2]; 2],
    pub continuous_tangent: bool,
}

impl EtKeyframe {
    pub fn linear(frame: f64, value: EtValue) -> Self {
        Self {
            frame,
            value,
            tween: 1,
            easing_index: 1,
            is_bezier: false,
            control_points: [[-10.0, 0.0], [10.0, 0.0]],
            continuous_tangent: true,
        }
    }

    pub fn from_cm3_tween(
        frame: f64,
        value: EtValue,
        tween: u32,
        control_points: [[f64; 2]; 2],
        continuous_tangent: bool,
    ) -> Self {
        let easing_index = (tween & 0xFF) as u8;
        let is_bezier = easing_index != 0 && ((tween >> 8) & 1) == 1;
        Self {
            frame,
            value,
            tween,
            easing_index,
            is_bezier,
            control_points,
            continuous_tangent,
        }
    }
}

/// Keyframeable / expression-driven ExpressText+ property track.
#[derive(Debug, Clone, PartialEq)]
pub struct EtPropertyTrack {
    pub animated: bool,
    pub expression: Option<String>,
    pub value: EtValue,
    pub keyframes: Vec<EtKeyframe>,
    /// Per-axis scalar sub-tracks for vector group properties (`objects` / `channels`).
    pub channels: Vec<EtPropertyTrack>,
}

impl EtPropertyTrack {
    pub fn constant(value: EtValue) -> Self {
        Self {
            animated: false,
            expression: None,
            value,
            keyframes: Vec::new(),
            channels: Vec::new(),
        }
    }

    /// Returns `true` when this track produces a constant value across all frames.
    pub fn is_static(&self) -> bool {
        if self.expression.is_some() {
            return false;
        }
        if !self.channels.is_empty() {
            return self.channels.iter().all(Self::is_static);
        }
        !self.animated || self.keyframes.len() <= 1
    }

    /// Evaluates this track (channels, keyframes, easing, Bezier handles, and optional expression)
    /// at `frame`.
    pub fn evaluate(&self, frame: f64) -> EtValue {
        let base = if self.channels.len() >= 3 {
            EtValue::Vec3([
                self.channels[0].evaluate(frame).as_number(0.0),
                self.channels[1].evaluate(frame).as_number(0.0),
                self.channels[2].evaluate(frame).as_number(0.0),
            ])
        } else if self.channels.len() == 1 {
            self.channels[0].evaluate(frame)
        } else if self.animated && !self.keyframes.is_empty() {
            eval_et_keyframes(&self.keyframes, frame, &self.value)
        } else {
            self.value.clone()
        };

        if let Some(expr) = &self.expression {
            let ctx = crate::BasicContext {
                frame,
                fps: 30.0,
                value: et_value_to_expr_value(&base),
                ..crate::BasicContext::default()
            };
            if let Ok(val) = crate::eval(expr, &ctx, &crate::EvalConfig::expressions_plus())
                && let Some(et) = expr_value_to_et_value(&val, &base)
            {
                return et;
            }
        }

        base
    }
}

/// Owned effector configuration parsed from an `expressProperties` /
/// `customProperties` / `objects` JSON block.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpressEffectorConfig {
    pub kind: EffectorKind,
    pub properties: BTreeMap<String, EtValue>,
    pub tracks: BTreeMap<String, EtPropertyTrack>,
}

impl ExpressEffectorConfig {
    pub fn as_layer(&self) -> EffectorLayer<'_> {
        EffectorLayer {
            kind: self.kind,
            props: self,
        }
    }
}

impl EtPropReader for ExpressEffectorConfig {
    fn read(&self, key: &str, frame: f64) -> Option<EtValue> {
        if let Some(track) = self.tracks.get(key) {
            return Some(track.evaluate(frame));
        }
        self.properties.read(key, frame)
    }

    fn is_static(&self, key: &str) -> bool {
        if let Some(track) = self.tracks.get(key) {
            return track.is_static();
        }
        true
    }
}

/// Complete container-level ExpressText+ configuration parsed from JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpressTextConfig {
    pub letters_mode: LettersMode,
    pub rotation_order: String,
    pub effectors: Vec<ExpressEffectorConfig>,
}

impl Default for ExpressTextConfig {
    fn default() -> Self {
        Self {
            letters_mode: LettersMode::PerText,
            rotation_order: "XYZ".to_string(),
            effectors: Vec::new(),
        }
    }
}

fn et_value_to_expr_value(v: &EtValue) -> crate::Value {
    match v {
        EtValue::Number(n) => crate::Value::Number(*n),
        EtValue::Vec3(a) => crate::Value::vec3(a[0], a[1], a[2]),
        EtValue::Text(s) => crate::Value::String(s.clone()),
    }
}

fn expr_value_to_et_value(v: &crate::Value, template: &EtValue) -> Option<EtValue> {
    match template {
        EtValue::Number(_) => {
            let n = v.to_number();
            n.is_finite().then_some(EtValue::Number(n))
        }
        EtValue::Vec3(fallback) => {
            if let Some(arr) = v.as_f64_vec()
                && arr.len() >= 3
                && arr[0].is_finite()
                && arr[1].is_finite()
                && arr[2].is_finite()
            {
                return Some(EtValue::Vec3([arr[0], arr[1], arr[2]]));
            }
            if let Some(n) = v.as_f64()
                && n.is_finite()
            {
                return Some(EtValue::Vec3([n, fallback[1], fallback[2]]));
            }
            None
        }
        EtValue::Text(_) => Some(EtValue::Text(v.to_js_string())),
    }
}

fn eval_cm3_easing(index: u8, t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    match index {
        0 => 0.0,
        1 => t,
        2 => t * t,
        3 => t * (2.0 - t),
        4 => {
            if t < 0.5 {
                2.0 * t * t
            } else {
                -1.0 + (4.0 - 2.0 * t) * t
            }
        }
        5 => t * t * t,
        6 => {
            let k = t - 1.0;
            k * k * k + 1.0
        }
        7 => {
            if t < 0.5 {
                4.0 * t * t * t
            } else {
                let k = 2.0 * t - 2.0;
                0.5 * k * k * k + 1.0
            }
        }
        8 => t.powi(4),
        9 => 1.0 - (1.0 - t).powi(4),
        10 => {
            if t < 0.5 {
                8.0 * t.powi(4)
            } else {
                1.0 - 8.0 * (1.0 - t).powi(4)
            }
        }
        11 => t.powi(5),
        12 => 1.0 - (1.0 - t).powi(5),
        13 => {
            if t < 0.5 {
                16.0 * t.powi(5)
            } else {
                1.0 - 16.0 * (1.0 - t).powi(5)
            }
        }
        14 => 1.0 - (t * std::f64::consts::FRAC_PI_2).cos(),
        15 => (t * std::f64::consts::FRAC_PI_2).sin(),
        16 => 0.5 * (1.0 - (PI * t).cos()),
        17 => {
            if t == 0.0 {
                0.0
            } else {
                2.0_f64.powf(10.0 * (t - 1.0))
            }
        }
        18 => {
            if t == 1.0 {
                1.0
            } else {
                1.0 - 2.0_f64.powf(-10.0 * t)
            }
        }
        19 => {
            if t == 0.0 || t == 1.0 {
                t
            } else if t < 0.5 {
                0.5 * 2.0_f64.powf(20.0 * t - 10.0)
            } else {
                1.0 - 0.5 * 2.0_f64.powf(-20.0 * t + 10.0)
            }
        }
        20 => 1.0 - (1.0 - t * t).max(0.0).sqrt(),
        21 => (1.0 - (t - 1.0).powi(2)).max(0.0).sqrt(),
        22 => {
            if t < 0.5 {
                0.5 * (1.0 - (1.0 - 4.0 * t * t).max(0.0).sqrt())
            } else {
                0.5 * ((1.0 - (2.0 * t - 2.0).powi(2)).max(0.0).sqrt() + 1.0)
            }
        }
        32 => {
            let k = t - 0.5;
            4.0 * k * k * k + 0.5
        }
        _ => t,
    }
}

fn solve_bezier_u(
    frame_a: f64,
    out_h: [f64; 2],
    frame_b: f64,
    in_h: [f64; 2],
    target_f: f64,
) -> (f64, f64) {
    let dt = frame_b - frame_a;
    let out_x = out_h[0].max(0.0);
    let in_x = (-in_h[0]).max(0.0);
    let sum = out_x + in_x;
    let corr = if sum > dt && sum > 1e-9 {
        dt / sum
    } else {
        1.0
    };
    let p0x = frame_a;
    let p1x = frame_a + out_x * corr;
    let p2x = frame_b - in_x * corr;
    let p3x = frame_b;

    let sample_x = |u: f64| {
        let inv = 1.0 - u;
        inv * inv * inv * p0x
            + 3.0 * inv * inv * u * p1x
            + 3.0 * inv * u * u * p2x
            + u * u * u * p3x
    };
    let sample_dx = |u: f64| {
        let inv = 1.0 - u;
        3.0 * inv * inv * (p1x - p0x) + 6.0 * inv * u * (p2x - p1x) + 3.0 * u * u * (p3x - p2x)
    };

    let mut u = ((target_f - frame_a) / dt).clamp(0.0, 1.0);
    for _ in 0..8 {
        let err = sample_x(u) - target_f;
        if err.abs() <= 1e-9 {
            break;
        }
        let dx = sample_dx(u);
        if dx.abs() <= 1e-12 {
            break;
        }
        u = (u - err / dx).clamp(0.0, 1.0);
    }
    (u, corr)
}

fn eval_bezier_scalar(
    frame_a: f64,
    va: f64,
    out_h: [f64; 2],
    frame_b: f64,
    vb: f64,
    in_h: [f64; 2],
    target_f: f64,
) -> f64 {
    let dt = frame_b - frame_a;
    if dt.abs() <= 1e-9 {
        return vb;
    }
    let (u, corr) = solve_bezier_u(frame_a, out_h, frame_b, in_h, target_f);
    let inv = 1.0 - u;
    let p0y = va;
    let p1y = va + out_h[1] * corr;
    let p2y = vb + in_h[1] * corr;
    let p3y = vb;
    inv * inv * inv * p0y + 3.0 * inv * inv * u * p1y + 3.0 * inv * u * u * p2y + u * u * u * p3y
}

fn eval_et_keyframes(keyframes: &[EtKeyframe], frame: f64, fallback: &EtValue) -> EtValue {
    let Some(first) = keyframes.first() else {
        return fallback.clone();
    };
    let last = &keyframes[keyframes.len() - 1];
    if keyframes.len() == 1 || frame <= first.frame {
        return first.value.clone();
    }
    if frame >= last.frame {
        return last.value.clone();
    }
    for pair in keyframes.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if frame >= a.frame && frame <= b.frame {
            let span = b.frame - a.frame;
            if span.abs() <= 1e-9 {
                return b.value.clone();
            }
            if b.easing_index == 0 {
                return a.value.clone();
            }
            return match (&a.value, &b.value) {
                (EtValue::Number(va), EtValue::Number(vb)) => {
                    if b.is_bezier {
                        EtValue::Number(eval_bezier_scalar(
                            a.frame,
                            *va,
                            a.control_points[1],
                            b.frame,
                            *vb,
                            b.control_points[0],
                            frame,
                        ))
                    } else {
                        let u = ((frame - a.frame) / span).clamp(0.0, 1.0);
                        let eased = eval_cm3_easing(b.easing_index, u);
                        EtValue::Number(va + (vb - va) * eased)
                    }
                }
                (EtValue::Vec3(va), EtValue::Vec3(vb)) => {
                    let factor = if b.is_bezier {
                        eval_bezier_scalar(
                            a.frame,
                            0.0,
                            a.control_points[1],
                            b.frame,
                            1.0,
                            b.control_points[0],
                            frame,
                        )
                    } else {
                        let u = ((frame - a.frame) / span).clamp(0.0, 1.0);
                        eval_cm3_easing(b.easing_index, u)
                    };
                    EtValue::Vec3([
                        va[0] + (vb[0] - va[0]) * factor,
                        va[1] + (vb[1] - va[1]) * factor,
                        va[2] + (vb[2] - va[2]) * factor,
                    ])
                }
                _ => a.value.clone(),
            };
        }
    }
    last.value.clone()
}

fn parse_control_points(raw: Option<&serde_json::Value>) -> [[f64; 2]; 2] {
    let mut cp = [[-10.0, 0.0], [10.0, 0.0]];
    let Some(val) = raw else {
        return cp;
    };
    let items = json_raw_array(val);
    if items.len() >= 2 && items[0].is_array() && items[1].is_array() {
        for (slot, item) in cp.iter_mut().zip(items.iter().take(2)) {
            if let Some(pair) = item.as_array()
                && pair.len() >= 2
            {
                slot[0] = pair[0].as_f64().unwrap_or(slot[0]);
                slot[1] = pair[1].as_f64().unwrap_or(slot[1]);
            }
        }
    } else if items.len() >= 4
        && let (Some(x0), Some(y0), Some(x1), Some(y1)) = (
            items[0].as_f64(),
            items[1].as_f64(),
            items[2].as_f64(),
            items[3].as_f64(),
        )
    {
        cp = [[x0, y0], [x1, y1]];
    }
    cp
}

fn json_to_primitive_et_value(raw: &serde_json::Value) -> Option<EtValue> {
    match raw {
        serde_json::Value::Number(n) => n.as_f64().filter(|v| v.is_finite()).map(EtValue::Number),
        serde_json::Value::Bool(b) => Some(EtValue::Number(if *b { 1.0 } else { 0.0 })),
        serde_json::Value::String(s) => Some(EtValue::Text(s.clone())),
        serde_json::Value::Array(arr) => {
            if arr.len() >= 3 {
                let x = arr[0].as_f64()?;
                let y = arr[1].as_f64()?;
                let z = arr[2].as_f64()?;
                if x.is_finite() && y.is_finite() && z.is_finite() {
                    return Some(EtValue::Vec3([x, y, z]));
                }
            }
            None
        }
        _ => None,
    }
}

fn parse_expression_field(map: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    map.get("expression")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
}

/// Parses any ExpressText+ property JSON value (bare primitive, wrapped `{ value }`,
/// multi-keyframe `{ animated, expression?, keyframes: [...] }`, or per-channel vector
/// `{ objects: [...] }` / `{ channels: [...] }`) into an [`EtPropertyTrack`].
pub fn json_to_et_track(raw: &serde_json::Value) -> Option<EtPropertyTrack> {
    if let Some(prim) = json_to_primitive_et_value(raw) {
        return Some(EtPropertyTrack::constant(prim));
    }
    let map = raw.as_object()?;
    let map_expr = parse_expression_field(map);
    let map_animated = map.get("animated").and_then(serde_json::Value::as_bool);

    // 1. Per-axis channel group: `{ objects: [ch0, ch1, ch2] }` or `{ channels: [...] }`.
    if let Some(ch_raw) = map.get("objects").or_else(|| map.get("channels")) {
        let items = json_raw_array(ch_raw);
        let mut channels = Vec::with_capacity(items.len());
        for item in items {
            if let Some(ch) = json_to_et_track(item) {
                channels.push(ch);
            }
        }
        if channels.len() == 1 {
            let mut ch0 = channels.into_iter().next()?;
            ch0.animated = ch0.animated || map_animated.unwrap_or(false);
            if ch0.expression.is_none() {
                ch0.expression = map_expr;
            }
            return Some(ch0);
        }
        if channels.len() >= 3 {
            let x = channels[0].value.as_number(0.0);
            let y = channels[1].value.as_number(0.0);
            let z = channels[2].value.as_number(0.0);
            let animated = map_animated.unwrap_or(false)
                || channels
                    .iter()
                    .any(|c| c.animated || c.expression.is_some());
            return Some(EtPropertyTrack {
                animated,
                expression: map_expr,
                value: EtValue::Vec3([x, y, z]),
                keyframes: Vec::new(),
                channels,
            });
        }
    }

    // 2. Keyframed property: `{ animated?, expression?, keyframes: [...] }` (or compact `"k"`).
    if let Some(kfs_raw) = map.get("keyframes").or_else(|| map.get("k")) {
        let kf_items = json_raw_array(kfs_raw);
        let mut keyframes = Vec::with_capacity(kf_items.len());
        for kf in kf_items {
            let Some(kobj) = kf.as_object() else {
                continue;
            };
            let Some(inner_val) = kobj.get("value").or_else(|| kobj.get("v")) else {
                continue;
            };
            let Some(et_val) = json_to_et_value(inner_val) else {
                continue;
            };
            let frame = kobj
                .get("frame")
                .or_else(|| kobj.get("f"))
                .or_else(|| kobj.get("t"))
                .and_then(serde_json::Value::as_f64)
                .filter(|v| v.is_finite())
                .unwrap_or(0.0);
            let default_tween: u32 = match et_val {
                EtValue::Text(_) => 0,
                EtValue::Number(_) | EtValue::Vec3(_) => 1,
            };
            let tween = kobj
                .get("tween")
                .or_else(|| kobj.get("e"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(u64::from(default_tween)) as u32;
            let cp = parse_control_points(
                kobj.get("controlPoints")
                    .or_else(|| kobj.get("handles"))
                    .or_else(|| kobj.get("cp")),
            );
            let cont = kobj
                .get("continuousTangent")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true);
            keyframes.push(EtKeyframe::from_cm3_tween(frame, et_val, tween, cp, cont));
        }

        if !keyframes.is_empty() {
            keyframes.sort_by(|a, b| a.frame.total_cmp(&b.frame));
            let rep_value = keyframes
                .iter()
                .find(|k| k.frame.abs() < 1e-6)
                .or_else(|| keyframes.first())
                .map(|k| k.value.clone())?;
            let animated = map_animated.unwrap_or(true);
            return Some(EtPropertyTrack {
                animated,
                expression: map_expr,
                value: rep_value,
                keyframes,
                channels: Vec::new(),
            });
        }
    }

    // 3. Wrapped property `{ value: ..., animated?, expression? }`.
    if let Some(inner) = map.get("value")
        && let Some(mut track) = json_to_et_track(inner)
    {
        if let Some(anim) = map_animated {
            track.animated = track.animated || anim;
        }
        if track.expression.is_none() {
            track.expression = map_expr;
        }
        return Some(track);
    }

    None
}

fn json_to_et_value(raw: &serde_json::Value) -> Option<EtValue> {
    json_to_et_track(raw).map(|track| track.value)
}

/// Normalizes either a JSON array `[a, b]` or CM3 numeric-keyed list `{"0": a, "1": b, "length": 2}`.
fn json_raw_array(val: &serde_json::Value) -> Vec<&serde_json::Value> {
    match val {
        serde_json::Value::Array(arr) => arr.iter().collect(),
        serde_json::Value::Object(map) => {
            let len = map
                .get("length")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n.min(256) as usize);
            let mut out = Vec::new();
            if let Some(count) = len {
                for i in 0..count {
                    if let Some(v) = map.get(&i.to_string()) {
                        out.push(v);
                    }
                }
            } else {
                let mut indexed: Vec<(usize, &serde_json::Value)> = map
                    .iter()
                    .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (idx, v)))
                    .collect();
                indexed.sort_by_key(|(idx, _)| *idx);
                out.extend(indexed.into_iter().map(|(_, v)| v));
            }
            out
        }
        _ => Vec::new(),
    }
}

fn parse_flat_express_props_map(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Vec<ExpressEffectorConfig> {
    type GroupedLayerProps = (BTreeMap<String, EtValue>, BTreeMap<String, EtPropertyTrack>);
    let mut grouped: BTreeMap<usize, GroupedLayerProps> = BTreeMap::new();
    for (key, raw_val) in map {
        if let Some(kind) = EffectorKind::from_property_key(key)
            && let Some(track) = json_to_et_track(raw_val)
        {
            let order_idx = EffectorKind::ALL
                .iter()
                .position(|&k| k == kind)
                .unwrap_or(0);
            let entry = grouped.entry(order_idx).or_default();
            entry.0.insert(key.clone(), track.value.clone());
            entry.1.insert(key.clone(), track);
        }
    }
    grouped
        .into_iter()
        .map(|(idx, (properties, tracks))| ExpressEffectorConfig {
            kind: EffectorKind::ALL[idx],
            properties,
            tracks,
        })
        .collect()
}

fn parse_custom_section(item: &serde_json::Value) -> Option<ExpressEffectorConfig> {
    let obj = item.as_object()?;
    let mut props = BTreeMap::new();
    let mut tracks = BTreeMap::new();
    let mut detected_kind: Option<EffectorKind> = obj
        .get("kind")
        .or_else(|| obj.get("name"))
        .and_then(serde_json::Value::as_str)
        .and_then(EffectorKind::from_name)
        .or_else(|| {
            obj.get("type")
                .and_then(|t| {
                    t.as_str()
                        .or_else(|| t.get("name").and_then(serde_json::Value::as_str))
                })
                .and_then(EffectorKind::from_name)
        });

    let def_rows = obj
        .get("type")
        .and_then(|t| t.get("objects"))
        .map(json_raw_array)
        .unwrap_or_default();
    let live_rows = obj.get("objects").map(json_raw_array).unwrap_or_default();

    for (idx, row) in live_rows.iter().enumerate() {
        let key = row
            .get("effectorKey")
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                row.get("definition")
                    .and_then(|d| d.get("effectorKey"))
                    .and_then(serde_json::Value::as_str)
            })
            .or_else(|| {
                def_rows
                    .get(idx)
                    .and_then(|d| d.get("effectorKey"))
                    .and_then(serde_json::Value::as_str)
            });
        if let Some(k) = key {
            if detected_kind.is_none() {
                detected_kind = EffectorKind::from_property_key(k);
            }
            if let Some(track) = json_to_et_track(row) {
                props.insert(k.to_string(), track.value.clone());
                tracks.insert(k.to_string(), track);
            }
        }
    }

    if let Some(serde_json::Value::Object(explicit)) = obj
        .get("properties")
        .or_else(|| obj.get("expressProperties"))
    {
        for (k, v) in explicit {
            if detected_kind.is_none() {
                detected_kind = EffectorKind::from_property_key(k);
            }
            if let Some(track) = json_to_et_track(v) {
                props.insert(k.clone(), track.value.clone());
                tracks.insert(k.clone(), track);
            }
        }
    }

    let kind = detected_kind?;
    Some(ExpressEffectorConfig {
        kind,
        properties: props,
        tracks,
    })
}

/// Resolves the container `LettersMode` from a container or `expressProperties` JSON block.
pub fn resolve_letters_mode(value: &serde_json::Value, has_content: bool) -> LettersMode {
    let ep = value
        .get("expressProperties")
        .filter(|v| v.is_object())
        .unwrap_or(value);
    let Some(map) = ep.as_object() else {
        return if has_content {
            LettersMode::AcrossTexts
        } else {
            LettersMode::PerText
        };
    };
    if let Some(raw_mode) = map.get("lettersMode") {
        if let Some(s) = raw_mode.as_str() {
            return match s.trim().to_ascii_lowercase().as_str() {
                "pertext" | "per_text" | "per text" | "0" => LettersMode::PerText,
                _ => LettersMode::AcrossTexts,
            };
        }
        if let Some(v) = json_to_et_value(raw_mode) {
            return LettersMode::from_raw(v.as_number(1.0));
        }
    }
    let has_version = map.contains_key("lettersVersion");
    let looks_pre_scope = !map.is_empty();
    if !has_version && (looks_pre_scope || has_content) {
        LettersMode::AcrossTexts
    } else {
        LettersMode::PerText
    }
}

/// Parses an ExpressText+ object JSON or `expressProperties` JSON block into
/// container settings (`LettersMode`, `rotation_order`) and ordered `Vec<ExpressEffectorConfig>`.
pub fn parse_express_text_config(value: &serde_json::Value) -> Result<ExpressTextConfig, String> {
    if value.is_null() {
        return Ok(ExpressTextConfig::default());
    }
    if let Some(arr) = value.as_array() {
        let effectors = arr.iter().filter_map(parse_custom_section).collect();
        return Ok(ExpressTextConfig {
            letters_mode: LettersMode::PerText,
            rotation_order: "XYZ".to_string(),
            effectors,
        });
    }
    let Some(root) = value.as_object() else {
        return Err("expressProperties JSON must be an object or array".to_string());
    };

    let has_custom = root
        .get("customProperties")
        .is_some_and(|v| !json_raw_array(v).is_empty());
    let has_objects = root
        .get("objects")
        .is_some_and(|v| !json_raw_array(v).is_empty());
    let letters_mode = resolve_letters_mode(value, has_custom || has_objects);

    let ep_map = root
        .get("expressProperties")
        .and_then(serde_json::Value::as_object)
        .unwrap_or(root);

    let rotation_order = ep_map
        .get("rotationOrder")
        .and_then(json_to_et_value)
        .and_then(|v| v.as_str().map(str::to_string))
        .filter(|s| matches!(s.as_str(), "XYZ" | "XZY" | "YXZ" | "YZX" | "ZXY" | "ZYX"))
        .unwrap_or_else(|| "XYZ".to_string());

    let mut effectors = Vec::new();

    // 1. Explicit `effectors` array if present.
    if let Some(list) = root.get("effectors").or_else(|| ep_map.get("effectors")) {
        for item in json_raw_array(list) {
            if let Some(cfg) = parse_custom_section(item) {
                effectors.push(cfg);
            }
        }
    }

    // 2. Inline `customProperties` effector sections (panel order).
    if let Some(customs) = root.get("customProperties") {
        for item in json_raw_array(customs) {
            if let Some(cfg) = parse_custom_section(item) {
                effectors.push(cfg);
            }
        }
    }

    // 3. Legacy child effector objects in `objects`.
    if let Some(objs) = root.get("objects") {
        for child in json_raw_array(objs) {
            if let Some(kind) = child
                .get("type")
                .and_then(serde_json::Value::as_str)
                .and_then(EffectorKind::from_name)
            {
                let mut props = BTreeMap::new();
                let mut tracks = BTreeMap::new();
                if let Some(child_ep) = child
                    .get("expressProperties")
                    .or_else(|| child.get("properties"))
                    .and_then(serde_json::Value::as_object)
                {
                    for (k, v) in child_ep {
                        if let Some(track) = json_to_et_track(v) {
                            props.insert(k.clone(), track.value.clone());
                            tracks.insert(k.clone(), track);
                        }
                    }
                }
                effectors.push(ExpressEffectorConfig {
                    kind,
                    properties: props,
                    tracks,
                });
            }
        }
    }

    // 4. Flat prefixed keys inside `expressProperties` (or root map).
    effectors.extend(parse_flat_express_props_map(ep_map));

    Ok(ExpressTextConfig {
        letters_mode,
        rotation_order,
        effectors,
    })
}

/// Parses an `expressProperties` (or full ExpressText+ object) JSON into
/// `Vec<ExpressEffectorConfig>`.
pub fn parse_express_properties(
    value: &serde_json::Value,
) -> Result<Vec<ExpressEffectorConfig>, String> {
    parse_express_text_config(value).map(|cfg| cfg.effectors)
}

/// Parses a raw JSON string containing `expressProperties` into
/// `Vec<ExpressEffectorConfig>`.
pub fn parse_express_properties_str(json: &str) -> Result<Vec<ExpressEffectorConfig>, String> {
    let val: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid expressProperties JSON: {e}"))?;
    parse_express_properties(&val)
}

/// Parses a raw JSON string into [`ExpressTextConfig`].
pub fn parse_express_text_config_str(json: &str) -> Result<ExpressTextConfig, String> {
    let val: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid ExpressText+ JSON: {e}"))?;
    parse_express_text_config(&val)
}
