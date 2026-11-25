use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use serde_json::Value;
use zc_project::Uuid;

use crate::{
    SessionKeyframe, SessionProperty, eval_session_property_at, property_track_to_session,
};

/// Mirrors a CM3 16-bit `tween` code for time-reversed playback so that every
/// segment `[k_i, k_{i+1}]` satisfies `E_rev(t) = 1 - E(1 - t)`.
fn mirror_cm3_tween_code(tween: u16) -> u16 {
    let low = (tween & 0xFF) as u8;
    if low == 0 {
        // Hold (`ease_0`) always stays Hold regardless of high bits.
        return 0;
    }
    let mirrored_low: u8 = match low {
        2 => 3,
        3 => 2,
        5 => 6,
        6 => 5,
        8 => 9,
        9 => 8,
        11 => 12,
        12 => 11,
        14 => 15,
        15 => 14,
        17 => 18,
        18 => 17,
        20 => 21,
        21 => 20,
        23 => 24,
        24 => 23,
        26 => 27,
        27 => 26,
        29 => 30,
        30 => 29,
        other => other,
    };
    (tween & 0xFF00) | u16::from(mirrored_low)
}

fn is_hold_tween(tween: u16) -> bool {
    (tween & 0xFF) == 0
}

/// Reverses a sorted keyframe curve across `[0, duration]`, migrating each segment's
/// CM3 right-endpoint tween code to the new right endpoint with mirrored easing and
/// mirrored Bezier handles, and preserving Hold (`tween == 0`) segment interiors.
pub(super) fn reverse_session_keyframes(keyframes: &mut Vec<SessionKeyframe>, duration: f64) {
    if keyframes.is_empty() {
        return;
    }
    keyframes.sort_by(|a, b| a.frame.total_cmp(&b.frame));
    let old = std::mem::take(keyframes);
    let n = old.len();
    let mut reversed = Vec::with_capacity(n);
    for j in 0..n {
        let src = &old[n - 1 - j];
        let new_frame = duration - src.frame;
        let tween = if j == 0 {
            1
        } else {
            // Segment `new[j - 1] -> new[j]` reverses `old[n - 1 - j] -> old[n - j]`,
            // whose CM3 tween metadata lives on its original right endpoint `old[n - j]`.
            let orig_right = &old[n - j];
            let orig_left = src;
            let seg_len = orig_right.frame - orig_left.frame;
            if is_hold_tween(orig_right.tween)
                && orig_left.value != orig_right.value
                && seg_len > 1e-12
            {
                // Forward Hold `[t0: v0, t1: v1 (Hold)]` evaluates to `v0` throughout `[t0, t1)`
                // and jumps to `v1` at `t1`. When reversed to `[d - t1, d - t0]`, keep `v1` at
                // `new_start` and step to `v0` immediately after `new_start` (scaled for short
                // intervals) so the entire reversed segment interior evaluates to `v0`.
                let new_start = duration - orig_right.frame;
                let step_offset = (seg_len * 0.25).min(1e-4);
                let step_frame = new_start + step_offset;
                if step_frame > new_start
                    && step_frame < new_frame
                    && reversed
                        .last()
                        .is_none_or(|prev: &SessionKeyframe| step_frame > prev.frame)
                {
                    let mut step_kf = orig_left.clone();
                    step_kf.frame = step_frame;
                    step_kf.tween = 0;
                    step_kf.continuous_tangent = false;
                    step_kf.control_points = [[0.0, 0.0], [0.0, 0.0]];
                    reversed.push(step_kf);
                }
            }
            mirror_cm3_tween_code(orig_right.tween)
        };
        let old_in = src.control_points[0];
        let old_out = src.control_points[1];
        let mut kf = src.clone();
        kf.frame = new_frame;
        kf.tween = tween;
        kf.control_points = [[-old_out[0], old_out[1]], [-old_in[0], old_in[1]]];
        reversed.push(kf);
    }
    reversed.sort_by(|a, b| a.frame.total_cmp(&b.frame));
    *keyframes = reversed;
}

/// Rewrites standalone `frame` and `time` identifiers inside a cloned sequence control
/// expression so evaluating it in a precomposed child timeline (shifted by `frame_offset`
/// frames at `fps`) yields the exact same result as evaluating in the parent timeline.
fn offset_expr_time_origin(expr: &str, frame_offset: f64, fps: f64) -> String {
    if expr.trim().is_empty() || !frame_offset.is_finite() || frame_offset.abs() <= 1e-9 {
        return expr.to_owned();
    }
    let Ok(tokens) = zc_expr::lexer::Lexer::new(expr).tokenize() else {
        return expr.to_owned();
    };
    let time_offset = frame_offset / fps.max(1.0);
    let frame_repl = if frame_offset >= 0.0 {
        format!("(frame + {frame_offset:?})")
    } else {
        format!("(frame - {:?})", -frame_offset)
    };
    let time_repl = if time_offset >= 0.0 {
        format!("(time + {time_offset:?})")
    } else {
        format!("(time - {:?})", -time_offset)
    };

    // Track lexical scopes so local parameters/bindings (`function f(frame) { return frame; }`,
    // `(time) => time + 1`, `{ let frame = 0; ... }`) shadow `frame`/`time` ONLY inside their
    // own scope without suppressing outer references.
    #[derive(Clone, Copy)]
    struct ScopeFrame {
        shadows_frame: bool,
        shadows_time: bool,
        is_expr_arrow: bool,
        expr_paren_depth: usize,
        expr_bracket_depth: usize,
    }
    let mut scopes = vec![ScopeFrame {
        shadows_frame: false,
        shadows_time: false,
        is_expr_arrow: false,
        expr_paren_depth: 0,
        expr_bracket_depth: 0,
    }];
    let mut pending_block_shadow: Option<(bool, bool)> = None;
    let mut skip_param_until: usize = 0;
    let mut replacements: Vec<(usize, usize, &str)> = Vec::new();

    for (idx, tok) in tokens.iter().enumerate() {
        use zc_expr::lexer::TokenKind;

        // Close any expression-bodied arrow scopes terminated by outer delimiters.
        while let Some(top) = scopes.last_mut() {
            if !top.is_expr_arrow {
                break;
            }
            let closes_arrow = top.expr_paren_depth == 0
                && top.expr_bracket_depth == 0
                && matches!(
                    tok.kind,
                    TokenKind::Semicolon
                        | TokenKind::Comma
                        | TokenKind::RParen
                        | TokenKind::RBracket
                        | TokenKind::RBrace
                );
            if closes_arrow {
                scopes.pop();
            } else {
                break;
            }
        }

        if idx < skip_param_until {
            continue;
        }

        match &tok.kind {
            TokenKind::LBrace => {
                let (sf, st) = pending_block_shadow.take().unwrap_or((false, false));
                scopes.push(ScopeFrame {
                    shadows_frame: sf,
                    shadows_time: st,
                    is_expr_arrow: false,
                    expr_paren_depth: 0,
                    expr_bracket_depth: 0,
                });
                continue;
            }
            TokenKind::RBrace => {
                if scopes.len() > 1 {
                    scopes.pop();
                }
                continue;
            }
            TokenKind::LParen => {
                // Check for arrow parameter list `(a, frame) => ...`
                let mut depth = 0usize;
                let mut k = idx;
                let mut param_sf = false;
                let mut param_st = false;
                let mut valid_params = true;
                while k < tokens.len() {
                    match &tokens[k].kind {
                        TokenKind::LParen => depth += 1,
                        TokenKind::RParen => {
                            depth = depth.saturating_sub(1);
                            if depth == 0 {
                                break;
                            }
                        }
                        TokenKind::Ident(name) if depth == 1 => {
                            if name == "frame" {
                                param_sf = true;
                            } else if name == "time" {
                                param_st = true;
                            }
                        }
                        TokenKind::Comma | TokenKind::Eq if depth == 1 => {}
                        _ if depth == 1 => {
                            // Still allow default expressions after `=`, only bail on obvious statements.
                            if matches!(tokens[k].kind, TokenKind::Semicolon) {
                                valid_params = false;
                                break;
                            }
                        }
                        _ => {}
                    }
                    k += 1;
                }
                if valid_params
                    && k < tokens.len()
                    && matches!(
                        tokens.get(k + 1).map(|t| &t.kind),
                        Some(TokenKind::FatArrow)
                    )
                {
                    if matches!(tokens.get(k + 2).map(|t| &t.kind), Some(TokenKind::LBrace)) {
                        pending_block_shadow = Some((param_sf, param_st));
                    } else {
                        scopes.push(ScopeFrame {
                            shadows_frame: param_sf,
                            shadows_time: param_st,
                            is_expr_arrow: true,
                            expr_paren_depth: 0,
                            expr_bracket_depth: 0,
                        });
                    }
                    skip_param_until = k + 2;
                    continue;
                }
                if let Some(top) = scopes.last_mut()
                    && top.is_expr_arrow
                {
                    top.expr_paren_depth += 1;
                }
                continue;
            }
            TokenKind::RParen => {
                if let Some(top) = scopes.last_mut()
                    && top.is_expr_arrow
                {
                    top.expr_paren_depth = top.expr_paren_depth.saturating_sub(1);
                }
                continue;
            }
            TokenKind::LBracket => {
                if let Some(top) = scopes.last_mut()
                    && top.is_expr_arrow
                {
                    top.expr_bracket_depth += 1;
                }
                continue;
            }
            TokenKind::RBracket => {
                if let Some(top) = scopes.last_mut()
                    && top.is_expr_arrow
                {
                    top.expr_bracket_depth = top.expr_bracket_depth.saturating_sub(1);
                }
                continue;
            }
            TokenKind::Var | TokenKind::Let | TokenKind::Const => {
                if let Some(next_tok) = tokens.get(idx + 1)
                    && let TokenKind::Ident(name) = &next_tok.kind
                    && let Some(top) = scopes.last_mut()
                {
                    if name == "frame" {
                        top.shadows_frame = true;
                    } else if name == "time" {
                        top.shadows_time = true;
                    }
                }
                continue;
            }
            TokenKind::Function => {
                let mut k = idx + 1;
                if matches!(tokens.get(k).map(|t| &t.kind), Some(TokenKind::Ident(_))) {
                    k += 1;
                }
                if matches!(tokens.get(k).map(|t| &t.kind), Some(TokenKind::LParen)) {
                    let mut depth = 0usize;
                    let mut param_sf = false;
                    let mut param_st = false;
                    while k < tokens.len() {
                        match &tokens[k].kind {
                            TokenKind::LParen => depth += 1,
                            TokenKind::RParen => {
                                depth = depth.saturating_sub(1);
                                if depth == 0 {
                                    break;
                                }
                            }
                            TokenKind::Ident(name) if depth == 1 => {
                                if name == "frame" {
                                    param_sf = true;
                                } else if name == "time" {
                                    param_st = true;
                                }
                            }
                            _ => {}
                        }
                        k += 1;
                    }
                    pending_block_shadow = Some((param_sf, param_st));
                    skip_param_until = k + 1;
                }
                continue;
            }
            TokenKind::Ident(name)
                if matches!(
                    tokens.get(idx + 1).map(|t| &t.kind),
                    Some(TokenKind::FatArrow)
                ) =>
            {
                let param_sf = name == "frame";
                let param_st = name == "time";
                if matches!(
                    tokens.get(idx + 2).map(|t| &t.kind),
                    Some(TokenKind::LBrace)
                ) {
                    pending_block_shadow = Some((param_sf, param_st));
                } else {
                    scopes.push(ScopeFrame {
                        shadows_frame: param_sf,
                        shadows_time: param_st,
                        is_expr_arrow: true,
                        expr_paren_depth: 0,
                        expr_bracket_depth: 0,
                    });
                }
                skip_param_until = idx + 2;
                continue;
            }
            _ => {}
        }

        let TokenKind::Ident(name) = &tok.kind else {
            continue;
        };
        let is_frame_shadowed = scopes.iter().any(|s| s.shadows_frame);
        let is_time_shadowed = scopes.iter().any(|s| s.shadows_time);
        let repl = if name == "frame" && !is_frame_shadowed {
            frame_repl.as_str()
        } else if name == "time" && !is_time_shadowed {
            time_repl.as_str()
        } else {
            continue;
        };
        let prev = idx
            .checked_sub(1)
            .and_then(|p| tokens.get(p))
            .map(|t| &t.kind);
        let next = tokens.get(idx + 1).map(|t| &t.kind);

        // Skip member accesses (`obj.frame`, `obj?.time`), declaration bindings,
        // assignment/update targets, and object literal property keys (`{ frame: 1 }`).
        if matches!(
            prev,
            Some(
                TokenKind::Dot
                    | TokenKind::OptionalChain
                    | TokenKind::Var
                    | TokenKind::Let
                    | TokenKind::Const
                    | TokenKind::Function
                    | TokenKind::PlusPlus
                    | TokenKind::MinusMinus
            )
        ) {
            continue;
        }
        if matches!(
            next,
            Some(
                TokenKind::Eq
                    | TokenKind::PlusEq
                    | TokenKind::MinusEq
                    | TokenKind::StarEq
                    | TokenKind::SlashEq
                    | TokenKind::PercentEq
                    | TokenKind::StarStarEq
                    | TokenKind::PlusPlus
                    | TokenKind::MinusMinus
            )
        ) {
            continue;
        }
        if matches!(prev, Some(TokenKind::LBrace | TokenKind::Comma))
            && matches!(next, Some(TokenKind::Colon))
        {
            continue;
        }
        replacements.push((tok.span.start, tok.span.end, repl));
    }

    if replacements.is_empty() {
        return expr.to_owned();
    }

    let mut out = String::with_capacity(expr.len() + replacements.len() * 20);
    let mut cursor = 0usize;
    for (start, end, repl) in replacements {
        if start >= cursor && end <= expr.len() {
            out.push_str(&expr[cursor..start]);
            out.push_str(repl);
            cursor = end;
        }
    }
    out.push_str(&expr[cursor..]);

    if zc_expr::parser::parse_source(&out, zc_expr::ExprMode::ExpressionsPlus).is_ok() {
        out
    } else {
        expr.to_owned()
    }
}

pub(super) fn offset_property_track_expressions_time_origin(
    track: &mut zc_project::PropertyTrack,
    frame_offset: f64,
    fps: f64,
) {
    if let Some(expr) = track.expression.as_deref() {
        let rewritten = offset_expr_time_origin(expr, frame_offset, fps);
        track.expression = Some(rewritten);
    }
    for ch in &mut track.channels {
        offset_property_track_expressions_time_origin(ch, frame_offset, fps);
    }
}

fn json_to_expr_value(val: &Value) -> zc_expr::Value {
    match val {
        Value::Null => zc_expr::Value::Null,
        Value::Bool(b) => zc_expr::Value::Bool(*b),
        Value::Number(n) => zc_expr::Value::Number(n.as_f64().unwrap_or(0.0)),
        Value::String(s) => zc_expr::Value::String(s.clone()),
        Value::Array(items) => {
            zc_expr::Value::Array(items.iter().map(json_to_expr_value).collect())
        }
        Value::Object(map) => zc_expr::Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), json_to_expr_value(v)))
                .collect(),
        ),
    }
}

fn eval_project_track_base(track: &zc_project::PropertyTrack, frame: f64) -> zc_expr::Value {
    if !track.channels.is_empty() {
        let vals = track
            .channels
            .iter()
            .map(|ch| {
                let sp = property_track_to_session(ch);
                json_to_expr_value(&eval_session_property_at(&sp, frame))
            })
            .collect();
        return zc_expr::Value::Array(vals);
    }
    let sp = property_track_to_session(track);
    json_to_expr_value(&eval_session_property_at(&sp, frame))
}

fn sorted_unique_track_key_frames(track: &zc_project::PropertyTrack) -> Vec<f64> {
    let mut frames: Vec<f64> = if !track.channels.is_empty() {
        track
            .channels
            .iter()
            .flat_map(|ch| ch.keyframes.iter().map(|k| k.frame))
            .collect()
    } else {
        track.keyframes.iter().map(|k| k.frame).collect()
    };
    frames.sort_by(f64::total_cmp);
    let mut dedup: Vec<f64> = Vec::with_capacity(frames.len());
    for f in frames {
        if dedup.last().is_none_or(|&prev| (f - prev).abs() > 1e-9) {
            dedup.push(f);
        }
    }
    dedup
}

fn lookup_layer_track<'a>(
    layer: &'a zc_project::Layer,
    name: &str,
) -> Option<&'a zc_project::PropertyTrack> {
    match name.trim().to_ascii_lowercase().as_str() {
        "position" | "pos" => Some(&layer.position),
        "scale" => Some(&layer.scale),
        "rotation" | "rot" => Some(&layer.rotation),
        "anchor" | "anchorpoint" | "anchor point" => Some(&layer.anchor),
        "opacity" => Some(&layer.opacity),
        _ => None,
    }
}

fn lookup_object_track<'a>(
    obj: &'a zc_project::Object3D,
    name: &str,
) -> Option<&'a zc_project::PropertyTrack> {
    let key = name.trim().to_ascii_lowercase();
    match key.as_str() {
        "position" | "pos" => return Some(&obj.position),
        "rotation" | "rot" => return Some(&obj.rotation),
        "scale" => return Some(&obj.scale),
        "enabled" | "visible" => return Some(&obj.enabled),
        _ => {}
    }
    match &obj.kind {
        zc_project::Object3DKind::Camera(cam) => match key.as_str() {
            "fov" => Some(&cam.fov),
            "zoom" => Some(&cam.zoom),
            "shakeamplitude" | "shake amplitude" => Some(&cam.shake_amplitude),
            "shakespeed" | "shake speed" => Some(&cam.shake_speed),
            _ => None,
        },
        zc_project::Object3DKind::Light(light) => match key.as_str() {
            "color" => Some(&light.color),
            "groundcolor" | "ground color" => Some(&light.ground_color),
            "target" => Some(&light.target),
            "intensity" => Some(&light.intensity),
            "angle" => Some(&light.angle),
            _ => None,
        },
        zc_project::Object3DKind::Group {
            custom_properties, ..
        } => custom_properties
            .iter()
            .find(|cp| cp.name == name || cp.name.eq_ignore_ascii_case(name))
            .map(|cp| &cp.track),
        zc_project::Object3DKind::Particles(p) => (key == "time").then_some(&p.time),
        zc_project::Object3DKind::Model(m) => m.model_properties.get(name).or_else(|| {
            m.model_properties
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v)
        }),
        _ => None,
    }
}

fn lookup_effect_track<'a>(
    fx: &'a zc_project::EffectInstance,
    name: &str,
) -> Option<&'a zc_project::PropertyTrack> {
    if name.eq_ignore_ascii_case("enabled") {
        return Some(&fx.enabled);
    }
    if let Some(t) = fx.params.get(name) {
        return Some(t);
    }
    if let Some((_, t)) = fx.params.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)) {
        return Some(t);
    }
    match &fx.kind {
        zc_project::EffectKind::EffectGroup {
            custom_properties, ..
        }
        | zc_project::EffectKind::CustomShader {
            custom_properties, ..
        } => custom_properties
            .iter()
            .find(|cp| cp.name == name || cp.name.eq_ignore_ascii_case(name))
            .map(|cp| &cp.track),
        _ => None,
    }
}

/// Shared lazy evaluation state for `FreezeClipFrame` expressions matching `zc-render`'s
/// `EngineExprContext` semantics (lazy control evaluation, strict ID lookup, cycle guard,
/// and clip/layer/sequence handle resolution).
pub(super) struct FreezeExprState<'a> {
    pub(super) project: &'a zc_project::Project,
    pub(super) sequence: &'a zc_project::Sequence,
    pub(super) abs_frame: f64,
    pub(super) fps: f64,
    pub(super) config: &'a zc_expr::EvalConfig,
    pub(super) control_cache: RefCell<HashMap<String, zc_expr::Value>>,
    pub(super) active_stack: RefCell<HashSet<String>>,
}

impl<'a> FreezeExprState<'a> {
    fn find_clip_by_id(&self, id: Uuid) -> Option<&'a zc_project::VideoClip> {
        self.sequence
            .video_tracks
            .iter()
            .flat_map(|t| t.clips.iter())
            .find(|c| c.id == id)
    }

    fn find_layer_owner(
        &self,
        layer_id: Uuid,
    ) -> Option<(&'a zc_project::VideoClip, &'a zc_project::Layer)> {
        fn search_layer(layer: &zc_project::Layer, target: Uuid) -> Option<&zc_project::Layer> {
            if layer.id == target {
                return Some(layer);
            }
            if let zc_project::LayerKind::Composite { layers } = &layer.kind {
                for child in layers {
                    if let Some(found) = search_layer(child, target) {
                        return Some(found);
                    }
                }
            }
            None
        }
        for track in &self.sequence.video_tracks {
            for clip in &track.clips {
                if let Some(layer) = search_layer(&clip.layer, layer_id) {
                    return Some((clip, layer));
                }
            }
        }
        None
    }

    fn find_object_owner(
        &self,
        obj_id: Uuid,
    ) -> Option<(
        &'a zc_project::VideoClip,
        &'a zc_project::Layer,
        Option<Uuid>,
        &'a zc_project::Object3D,
    )> {
        fn search_objs(
            objs: &[zc_project::Object3D],
            parent: Option<Uuid>,
            target: Uuid,
        ) -> Option<(Option<Uuid>, &zc_project::Object3D)> {
            for o in objs {
                if o.id == target {
                    return Some((parent, o));
                }
                if let zc_project::Object3DKind::Group { children, .. } = &o.kind
                    && let Some(found) = search_objs(children, Some(o.id), target)
                {
                    return Some(found);
                }
            }
            None
        }
        for track in &self.sequence.video_tracks {
            for clip in &track.clips {
                if let zc_project::LayerKind::Scene3D { objects, .. } = &clip.layer.kind
                    && let Some((parent, obj)) = search_objs(objects, None, obj_id)
                {
                    return Some((clip, &clip.layer, parent, obj));
                }
            }
        }
        None
    }

    fn find_effect_owner(
        &self,
        fx_id: Uuid,
    ) -> Option<(
        &'a zc_project::VideoClip,
        &'a zc_project::Layer,
        Option<Uuid>,
        &'a zc_project::EffectInstance,
    )> {
        fn search_fxs(
            fxs: &[zc_project::EffectInstance],
            parent: Option<Uuid>,
            target: Uuid,
        ) -> Option<(Option<Uuid>, &zc_project::EffectInstance)> {
            for fx in fxs {
                if fx.id == target {
                    return Some((parent, fx));
                }
                if let zc_project::EffectKind::EffectGroup { children, .. } = &fx.kind
                    && let Some(found) = search_fxs(children, Some(fx.id), target)
                {
                    return Some(found);
                }
            }
            None
        }
        for track in &self.sequence.video_tracks {
            for clip in &track.clips {
                if let Some((parent, fx)) = search_fxs(&clip.layer.effects, None, fx_id) {
                    return Some((clip, &clip.layer, parent, fx));
                }
            }
        }
        None
    }

    fn eval_track_checked(
        &self,
        track: &'a zc_project::PropertyTrack,
        frame: f64,
        time: f64,
        self_prop_id: zc_expr::HandleId,
        scope: FreezeExprScope,
    ) -> Result<zc_expr::Value, zc_expr::ExprError> {
        let has_expr = track
            .expression
            .as_deref()
            .is_some_and(|s| !s.trim().is_empty());
        let base_value = if !track.channels.is_empty() && !has_expr {
            let mut vals = Vec::with_capacity(track.channels.len());
            for (idx, ch) in track.channels.iter().enumerate() {
                let ch_id = zc_expr::HandleId::new(format!("{}:ch{idx}", self_prop_id.as_str()));
                vals.push(self.eval_track_checked(ch, frame, time, ch_id, scope)?);
            }
            zc_expr::Value::Array(vals)
        } else {
            eval_project_track_base(track, frame)
        };

        if let Some(expr_src) = track
            .expression
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let ctx = FreezeTimeExprContext {
                state: self,
                frame,
                time,
                base_value: base_value.clone(),
                self_prop_id,
                self_track: Some(track),
                time_prop_fallback: None,
                active_clip_id: scope.clip_id,
                active_layer_id: scope.layer_id,
                active_object_id: scope.object_id,
                active_effect_id: scope.effect_id,
            };
            let outcome = zc_expr::eval_with_fallback(expr_src, &ctx, self.config);
            if let Some(err) = outcome.error {
                if matches!(err, zc_expr::ExprError::CycleDetected(_)) {
                    return Err(err);
                }
                return Ok(base_value);
            }
            return Ok(coerce_expr_value_to_track_type(
                &outcome.value,
                &track.value,
                &base_value,
            ));
        }

        Ok(base_value)
    }

    fn eval_control(
        &self,
        ctrl: &'a zc_project::SequenceControl,
        cycle_key: &str,
    ) -> Result<zc_expr::Value, zc_expr::ExprError> {
        let mut expr_val = self.eval_track_checked(
            &ctrl.track,
            self.abs_frame,
            self.abs_frame / self.fps.max(1.0),
            zc_expr::HandleId::new(cycle_key.to_owned()),
            FreezeExprScope::default(),
        )?;
        if ctrl.kind == zc_project::SequenceControlKind::Checkbox {
            expr_val = match expr_val {
                zc_expr::Value::String(s) => {
                    zc_expr::Value::Number(s.trim().parse::<f64>().unwrap_or(0.0))
                }
                zc_expr::Value::Bool(b) => zc_expr::Value::Number(if b { 1.0 } else { 0.0 }),
                other => other,
            };
        }
        Ok(expr_val)
    }
}

/// Coerces an evaluated expression value to match the target `PropertyTrack`'s expected type,
/// mirroring `zc_render::engine::coerce_expr_value` so Freeze evaluation matches preview 1:1.
fn coerce_expr_value_to_track_type(
    expr: &zc_expr::Value,
    template: &zc_project::PropertyValue,
    fallback: &zc_expr::Value,
) -> zc_expr::Value {
    use zc_project::PropertyValue;
    match template {
        PropertyValue::Number(_) => {
            let n = expr.to_number();
            if n.is_finite() {
                zc_expr::Value::Number(n)
            } else {
                fallback.clone()
            }
        }
        PropertyValue::Option(_) => {
            let n = expr.to_number();
            if n.is_finite() && n >= 0.0 {
                zc_expr::Value::Number(n.round())
            } else {
                fallback.clone()
            }
        }
        PropertyValue::Bool(_) => zc_expr::Value::Bool(expr.to_bool()),
        PropertyValue::Vec2(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                zc_expr::Value::Array(vec![
                    zc_expr::Value::Number(
                        arr.first()
                            .copied()
                            .filter(|v| v.is_finite())
                            .unwrap_or(def[0]),
                    ),
                    zc_expr::Value::Number(
                        arr.get(1)
                            .copied()
                            .filter(|v| v.is_finite())
                            .unwrap_or(def[1]),
                    ),
                ])
            } else if let Some(n) = expr.as_f64().filter(|v| v.is_finite()) {
                zc_expr::Value::Array(vec![zc_expr::Value::Number(n), zc_expr::Value::Number(n)])
            } else {
                fallback.clone()
            }
        }
        PropertyValue::Vec3(def) | PropertyValue::Color(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                zc_expr::Value::Array(vec![
                    zc_expr::Value::Number(
                        arr.first()
                            .copied()
                            .filter(|v| v.is_finite())
                            .unwrap_or(def[0]),
                    ),
                    zc_expr::Value::Number(
                        arr.get(1)
                            .copied()
                            .filter(|v| v.is_finite())
                            .unwrap_or(def[1]),
                    ),
                    zc_expr::Value::Number(
                        arr.get(2)
                            .copied()
                            .filter(|v| v.is_finite())
                            .unwrap_or(def[2]),
                    ),
                ])
            } else if let Some(n) = expr.as_f64().filter(|v| v.is_finite()) {
                zc_expr::Value::Array(vec![
                    zc_expr::Value::Number(n),
                    zc_expr::Value::Number(n),
                    zc_expr::Value::Number(n),
                ])
            } else {
                fallback.clone()
            }
        }
        PropertyValue::Vec4(def) | PropertyValue::ColorRgba(def) => {
            if let Some(arr) = expr.as_f64_vec() {
                zc_expr::Value::Array(vec![
                    zc_expr::Value::Number(
                        arr.first()
                            .copied()
                            .filter(|v| v.is_finite())
                            .unwrap_or(def[0]),
                    ),
                    zc_expr::Value::Number(
                        arr.get(1)
                            .copied()
                            .filter(|v| v.is_finite())
                            .unwrap_or(def[1]),
                    ),
                    zc_expr::Value::Number(
                        arr.get(2)
                            .copied()
                            .filter(|v| v.is_finite())
                            .unwrap_or(def[2]),
                    ),
                    zc_expr::Value::Number(
                        arr.get(3)
                            .copied()
                            .filter(|v| v.is_finite())
                            .unwrap_or(def[3]),
                    ),
                ])
            } else if let Some(n) = expr.as_f64().filter(|v| v.is_finite()) {
                zc_expr::Value::Array(vec![
                    zc_expr::Value::Number(n),
                    zc_expr::Value::Number(n),
                    zc_expr::Value::Number(n),
                    zc_expr::Value::Number(n),
                ])
            } else {
                fallback.clone()
            }
        }
        PropertyValue::Text(_) => zc_expr::Value::String(expr.to_js_string()),
        _ => expr.clone(),
    }
}

#[derive(Clone, Copy, Default)]
struct FreezeExprScope {
    clip_id: Option<Uuid>,
    layer_id: Option<Uuid>,
    object_id: Option<Uuid>,
    effect_id: Option<Uuid>,
}

pub(super) struct FreezeTimeExprContext<'a> {
    pub(super) state: &'a FreezeExprState<'a>,
    pub(super) frame: f64,
    pub(super) time: f64,
    pub(super) base_value: zc_expr::Value,
    pub(super) self_prop_id: zc_expr::HandleId,
    pub(super) self_track: Option<&'a zc_project::PropertyTrack>,
    pub(super) time_prop_fallback: Option<&'a SessionProperty>,
    pub(super) active_clip_id: Option<Uuid>,
    pub(super) active_layer_id: Option<Uuid>,
    pub(super) active_object_id: Option<Uuid>,
    pub(super) active_effect_id: Option<Uuid>,
}

type ResolvedPropTrack<'a> = (
    &'a zc_project::PropertyTrack,
    Option<&'a zc_project::VideoClip>,
    Option<Uuid>,
    Option<Uuid>,
    Option<Uuid>,
);

impl<'a> FreezeTimeExprContext<'a> {
    fn resolve_property_track(&self, id: &zc_expr::HandleId) -> Option<ResolvedPropTrack<'a>> {
        let rest = id.as_str().strip_prefix("prop:")?;
        if let Some(after) = rest.strip_prefix("clip:") {
            let (u_str, prop) = after.split_once(':')?;
            let u = Uuid::parse_str(u_str).ok()?;
            let clip = self.state.find_clip_by_id(u)?;
            if prop.eq_ignore_ascii_case("time") {
                return Some((
                    &clip.time_curve,
                    Some(clip),
                    Some(clip.layer.id),
                    None,
                    None,
                ));
            }
        } else if let Some(after) = rest.strip_prefix("layer:") {
            let (u_str, prop) = after.split_once(':')?;
            let u = Uuid::parse_str(u_str).ok()?;
            let (clip, layer) = self.state.find_layer_owner(u)?;
            let track = lookup_layer_track(layer, prop)?;
            return Some((track, Some(clip), Some(layer.id), None, None));
        } else if let Some(after) = rest.strip_prefix("obj:") {
            let (u_str, prop) = after.split_once(':')?;
            let u = Uuid::parse_str(u_str).ok()?;
            let (clip, layer, _, obj) = self.state.find_object_owner(u)?;
            let track = lookup_object_track(obj, prop)?;
            return Some((track, Some(clip), Some(layer.id), Some(obj.id), None));
        } else if let Some(after) = rest.strip_prefix("fx:") {
            let (u_str, prop) = after.split_once(':')?;
            let u = Uuid::parse_str(u_str).ok()?;
            let (clip, layer, _, fx) = self.state.find_effect_owner(u)?;
            let track = lookup_effect_track(fx, prop)?;
            return Some((track, Some(clip), Some(layer.id), None, Some(fx.id)));
        }
        None
    }
}

impl zc_expr::ExprContext for FreezeTimeExprContext<'_> {
    fn frame(&self) -> f64 {
        self.frame
    }

    fn time(&self) -> f64 {
        self.time
    }

    fn fps(&self) -> f64 {
        self.state.fps
    }

    fn value(&self) -> zc_expr::Value {
        self.base_value.clone()
    }

    fn this_property_id(&self) -> Option<zc_expr::HandleId> {
        Some(self.self_prop_id.clone())
    }

    fn this_object_id(&self) -> Option<zc_expr::HandleId> {
        self.active_object_id
            .map(|id| zc_expr::HandleId::new(format!("obj:{id}")))
    }

    fn this_layer_id(&self) -> Option<zc_expr::HandleId> {
        self.active_layer_id
            .map(|id| zc_expr::HandleId::new(format!("layer:{id}")))
    }

    fn this_clip_id(&self) -> Option<zc_expr::HandleId> {
        self.active_clip_id
            .map(|id| zc_expr::HandleId::new(format!("clip:{id}")))
    }

    fn this_sequence_id(&self) -> Option<zc_expr::HandleId> {
        Some(zc_expr::HandleId::new("seq"))
    }

    fn property_local_frame(&self, id: &zc_expr::HandleId) -> Option<f64> {
        if id == &self.self_prop_id {
            return Some(self.frame);
        }
        self.resolve_property_track(id)
            .map(|(_, clip_opt, _, _, _)| {
                clip_opt.map_or(self.state.abs_frame, |c| self.state.abs_frame - c.start)
            })
    }

    fn handle_name(&self, id: &zc_expr::HandleId) -> Option<String> {
        let s = id.as_str();
        if s == "seq" {
            return Some(self.state.sequence.name.clone());
        }
        if let Some(u) = s
            .strip_prefix("clip:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            return self.state.find_clip_by_id(u).map(|c| c.name.clone());
        }
        if let Some(u) = s
            .strip_prefix("layer:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            return self.state.find_layer_owner(u).map(|(_, l)| l.name.clone());
        }
        if let Some(u) = s.strip_prefix("obj:").and_then(|v| Uuid::parse_str(v).ok()) {
            return self
                .state
                .find_object_owner(u)
                .map(|(_, _, _, o)| o.name.clone());
        }
        if let Some(u) = s.strip_prefix("fx:").and_then(|v| Uuid::parse_str(v).ok()) {
            return self
                .state
                .find_effect_owner(u)
                .map(|(_, _, _, f)| f.name.clone());
        }
        if let Some(rest) = s.strip_prefix("prop:")
            && let Some((_, prop_name)) = rest.rsplit_once(':')
        {
            return Some(prop_name.to_owned());
        }
        None
    }

    fn parent_of(
        &self,
        id: &zc_expr::HandleId,
        levels: usize,
    ) -> Result<Option<zc_expr::HandleRef>, zc_expr::ExprError> {
        use zc_expr::{HandleId, HandleRef};
        if levels == 0 {
            return Ok(None);
        }
        let mut current: Option<HandleRef> = if let Some(rest) = id.as_str().strip_prefix("prop:") {
            if let Some(after) = rest.strip_prefix("obj:") {
                after
                    .split_once(':')
                    .map(|(u, _)| HandleRef::Object(HandleId::new(format!("obj:{u}"))))
            } else if let Some(after) = rest.strip_prefix("fx:") {
                after
                    .split_once(':')
                    .map(|(u, _)| HandleRef::Effect(HandleId::new(format!("fx:{u}"))))
            } else if let Some(after) = rest.strip_prefix("layer:") {
                after
                    .split_once(':')
                    .map(|(u, _)| HandleRef::Layer(HandleId::new(format!("layer:{u}"))))
            } else if let Some(after) = rest.strip_prefix("clip:") {
                after
                    .split_once(':')
                    .map(|(u, _)| HandleRef::Clip(HandleId::new(format!("clip:{u}"))))
            } else {
                None
            }
        } else if let Some(u) = id
            .as_str()
            .strip_prefix("obj:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            self.state
                .find_object_owner(u)
                .map(|(_, layer, parent_obj, _)| {
                    if let Some(po) = parent_obj {
                        HandleRef::Object(HandleId::new(format!("obj:{po}")))
                    } else {
                        HandleRef::Layer(HandleId::new(format!("layer:{}", layer.id)))
                    }
                })
        } else if let Some(u) = id
            .as_str()
            .strip_prefix("fx:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            self.state
                .find_effect_owner(u)
                .map(|(_, layer, parent_fx, _)| {
                    if let Some(pf) = parent_fx {
                        HandleRef::Effect(HandleId::new(format!("fx:{pf}")))
                    } else {
                        HandleRef::Layer(HandleId::new(format!("layer:{}", layer.id)))
                    }
                })
        } else if let Some(u) = id
            .as_str()
            .strip_prefix("layer:")
            .and_then(|v| Uuid::parse_str(v).ok())
        {
            self.state
                .find_layer_owner(u)
                .map(|(clip, _)| HandleRef::Clip(HandleId::new(format!("clip:{}", clip.id))))
        } else if id.as_str().starts_with("clip:") {
            Some(HandleRef::Sequence)
        } else {
            None
        };

        for _ in 1..levels {
            let Some(ref cur) = current else {
                return Ok(None);
            };
            current = match cur {
                HandleRef::Object(hid)
                | HandleRef::Effect(hid)
                | HandleRef::Layer(hid)
                | HandleRef::Clip(hid)
                | HandleRef::Property(hid) => self.parent_of(hid, 1)?,
                HandleRef::Sequence | HandleRef::Controls => None,
            };
        }
        Ok(current)
    }

    fn property_value(&self, id: &zc_expr::HandleId) -> Result<zc_expr::Value, zc_expr::ExprError> {
        if id == &self.self_prop_id {
            return Ok(self.value());
        }
        if let Some(seq_prop) = id.as_str().strip_prefix("prop:seq:") {
            return match seq_prop.to_ascii_lowercase().as_str() {
                "resolution" => Ok(zc_expr::Value::vec2(
                    f64::from(self.state.sequence.resolution[0]),
                    f64::from(self.state.sequence.resolution[1]),
                )),
                "width" => Ok(zc_expr::Value::Number(f64::from(
                    self.state.sequence.resolution[0],
                ))),
                "height" => Ok(zc_expr::Value::Number(f64::from(
                    self.state.sequence.resolution[1],
                ))),
                "rate" | "fps" => Ok(zc_expr::Value::Number(self.state.fps)),
                "length" | "duration" => {
                    Ok(zc_expr::Value::Number(self.state.sequence.length_frames))
                }
                _ => Err(zc_expr::ExprError::NotFound(format!(
                    "sequence property '{seq_prop}'"
                ))),
            };
        }
        if let Some(after) = id.as_str().strip_prefix("prop:clip:")
            && let Some((u_str, prop)) = after.split_once(':')
            && let Ok(u) = Uuid::parse_str(u_str)
            && let Some(clip) = self.state.find_clip_by_id(u)
        {
            match prop.to_ascii_lowercase().as_str() {
                "start" => return Ok(zc_expr::Value::Number(clip.start)),
                "length" | "duration" => return Ok(zc_expr::Value::Number(clip.length)),
                _ => {}
            }
        }
        if let Some(after) = id.as_str().strip_prefix("prop:layer:")
            && let Some((u_str, prop)) = after.split_once(':')
            && let Ok(u) = Uuid::parse_str(u_str)
            && let Some((_, layer)) = self.state.find_layer_owner(u)
        {
            match prop.to_ascii_lowercase().as_str() {
                "width" => return Ok(zc_expr::Value::Number(f64::from(layer.resolution[0]))),
                "height" => return Ok(zc_expr::Value::Number(f64::from(layer.resolution[1]))),
                "resolution" => {
                    return Ok(zc_expr::Value::vec2(
                        f64::from(layer.resolution[0]),
                        f64::from(layer.resolution[1]),
                    ));
                }
                _ => {}
            }
        }

        let key = id.as_str().to_owned();
        if !self.state.active_stack.borrow_mut().insert(key.clone()) {
            return Err(zc_expr::ExprError::CycleDetected(key));
        }

        let res = if let Some((track, clip_opt, layer_id, obj_id, fx_id)) =
            self.resolve_property_track(id)
        {
            let target_local_frame =
                clip_opt.map_or(self.state.abs_frame, |c| self.state.abs_frame - c.start);
            let target_time = if let Some(c) = clip_opt {
                eval_project_track_base(&c.time_curve, target_local_frame)
                    .as_f64()
                    .unwrap_or(target_local_frame / self.state.fps.max(1.0))
            } else {
                target_local_frame / self.state.fps.max(1.0)
            };
            self.state.eval_track_checked(
                track,
                target_local_frame,
                target_time,
                id.clone(),
                FreezeExprScope {
                    clip_id: clip_opt.map(|c| c.id),
                    layer_id,
                    object_id: obj_id,
                    effect_id: fx_id,
                },
            )
        } else {
            Err(zc_expr::ExprError::NotFound(format!(
                "property '{}'",
                id.as_str()
            )))
        };

        self.state.active_stack.borrow_mut().remove(&key);
        res
    }

    fn property_value_at_time(
        &self,
        id: &zc_expr::HandleId,
        seconds: f64,
    ) -> Result<zc_expr::Value, zc_expr::ExprError> {
        let sample_frame = seconds * self.state.fps;
        if id == &self.self_prop_id {
            if let Some(track) = self.self_track {
                return Ok(eval_project_track_base(track, sample_frame));
            }
            if let Some(prop) = self.time_prop_fallback {
                let sampled = eval_session_property_at(prop, sample_frame);
                return Ok(json_to_expr_value(&sampled));
            }
            return Ok(self.value());
        }
        if let Some((track, _, _, _, _)) = self.resolve_property_track(id) {
            return Ok(eval_project_track_base(track, sample_frame));
        }
        self.property_value(id)
    }

    fn property_num_keys(&self, id: &zc_expr::HandleId) -> usize {
        if id == &self.self_prop_id {
            if let Some(track) = self.self_track {
                return sorted_unique_track_key_frames(track).len();
            }
            return self.time_prop_fallback.map_or(0, |p| p.keyframes.len());
        }
        self.resolve_property_track(id)
            .map_or(0, |(track, _, _, _, _)| {
                sorted_unique_track_key_frames(track).len()
            })
    }

    fn property_key(
        &self,
        id: &zc_expr::HandleId,
        index_1based: usize,
    ) -> Result<zc_expr::KeyframeRecord, zc_expr::ExprError> {
        if index_1based >= 1 {
            if id == &self.self_prop_id {
                if let Some(track) = self.self_track {
                    let frames = sorted_unique_track_key_frames(track);
                    if let Some(&kf_frame) = frames.get(index_1based - 1) {
                        return Ok(zc_expr::KeyframeRecord {
                            index: index_1based,
                            time: kf_frame / self.state.fps.max(1.0),
                            value: eval_project_track_base(track, kf_frame),
                        });
                    }
                } else if let Some(prop) = self.time_prop_fallback {
                    let mut sorted: Vec<&SessionKeyframe> = prop.keyframes.iter().collect();
                    sorted.sort_by(|a, b| a.frame.total_cmp(&b.frame));
                    if let Some(kf) = sorted.get(index_1based - 1) {
                        return Ok(zc_expr::KeyframeRecord {
                            index: index_1based,
                            time: kf.frame / self.state.fps.max(1.0),
                            value: json_to_expr_value(&kf.value),
                        });
                    }
                }
            } else if let Some((track, _, _, _, _)) = self.resolve_property_track(id) {
                let frames = sorted_unique_track_key_frames(track);
                if let Some(&kf_frame) = frames.get(index_1based - 1) {
                    return Ok(zc_expr::KeyframeRecord {
                        index: index_1based,
                        time: kf_frame / self.state.fps.max(1.0),
                        value: eval_project_track_base(track, kf_frame),
                    });
                }
            }
        }
        Err(zc_expr::ExprError::RangeError(format!(
            "keyframe index {index_1based} out of range on property '{}'",
            id.as_str()
        )))
    }

    fn layer(&self, query: &zc_expr::NameOrIndex) -> Result<zc_expr::HandleId, zc_expr::ExprError> {
        match query {
            zc_expr::NameOrIndex::Index(idx) => {
                if *idx >= 1
                    && let Some(track) = self.state.sequence.video_tracks.get((*idx - 1) as usize)
                    && let Some(clip) = track.clips.iter().find(|c| {
                        c.length > 0.0
                            && self.state.abs_frame >= c.start
                            && self.state.abs_frame < c.start + c.length
                    })
                {
                    return Ok(zc_expr::HandleId::new(format!("layer:{}", clip.layer.id)));
                }
                Err(zc_expr::ExprError::NotFound(format!("layer index {idx}")))
            }
            zc_expr::NameOrIndex::Name(name) => {
                for track in &self.state.sequence.video_tracks {
                    for clip in &track.clips {
                        if clip.length > 0.0
                            && self.state.abs_frame >= clip.start
                            && self.state.abs_frame < clip.start + clip.length
                            && (clip.layer.name == *name
                                || clip.name == *name
                                || track.name == *name)
                        {
                            return Ok(zc_expr::HandleId::new(format!("layer:{}", clip.layer.id)));
                        }
                    }
                }
                for track in &self.state.sequence.video_tracks {
                    for clip in &track.clips {
                        if clip.length > 0.0
                            && self.state.abs_frame >= clip.start
                            && self.state.abs_frame < clip.start + clip.length
                            && (clip.layer.name.eq_ignore_ascii_case(name)
                                || clip.name.eq_ignore_ascii_case(name))
                        {
                            return Ok(zc_expr::HandleId::new(format!("layer:{}", clip.layer.id)));
                        }
                    }
                }
                Err(zc_expr::ExprError::NotFound(format!("layer '{name}'")))
            }
        }
    }

    fn effect(
        &self,
        owner: Option<&zc_expr::HandleId>,
        name: &str,
    ) -> Result<zc_expr::HandleId, zc_expr::ExprError> {
        fn find_fx<'b>(
            effects: &'b [zc_project::EffectInstance],
            name: &str,
        ) -> Option<&'b zc_project::EffectInstance> {
            for fx in effects {
                if fx.name == name {
                    return Some(fx);
                }
                match &fx.kind {
                    zc_project::EffectKind::Standard(id) if id == name => return Some(fx),
                    zc_project::EffectKind::Plugin { effect_id, .. } if effect_id == name => {
                        return Some(fx);
                    }
                    zc_project::EffectKind::EffectGroup { children, .. } => {
                        if let Some(found) = find_fx(children, name) {
                            return Some(found);
                        }
                    }
                    _ => {}
                }
            }
            effects.iter().find(|fx| fx.name.eq_ignore_ascii_case(name))
        }

        let target_layer_id = owner
            .and_then(|h| {
                h.as_str()
                    .strip_prefix("layer:")
                    .and_then(|u| Uuid::parse_str(u).ok())
            })
            .or(self.active_layer_id);

        if let Some(owner_h) = owner
            && let Some(u) = owner_h
                .as_str()
                .strip_prefix("fx:")
                .and_then(|v| Uuid::parse_str(v).ok())
            && let Some((_, _, _, parent_fx)) = self.state.find_effect_owner(u)
            && let zc_project::EffectKind::EffectGroup { children, .. } = &parent_fx.kind
            && let Some(found) = find_fx(children, name)
        {
            return Ok(zc_expr::HandleId::new(format!("fx:{}", found.id)));
        }

        if let Some(lid) = target_layer_id
            && let Some((_, layer)) = self.state.find_layer_owner(lid)
            && let Some(found) = find_fx(&layer.effects, name)
        {
            return Ok(zc_expr::HandleId::new(format!("fx:{}", found.id)));
        }
        Err(zc_expr::ExprError::NotFound(format!("effect '{name}'")))
    }

    fn effect_param(
        &self,
        effect_id: &zc_expr::HandleId,
        param: &str,
    ) -> Result<zc_expr::HandleId, zc_expr::ExprError> {
        if let Some(u) = effect_id
            .as_str()
            .strip_prefix("fx:")
            .and_then(|v| Uuid::parse_str(v).ok())
            && let Some((_, _, _, fx)) = self.state.find_effect_owner(u)
            && lookup_effect_track(fx, param).is_some()
        {
            return Ok(zc_expr::HandleId::new(format!("prop:fx:{u}:{param}")));
        }
        Err(zc_expr::ExprError::NotFound(format!(
            "effect '{}'.param('{param}')",
            effect_id.as_str()
        )))
    }

    fn object(
        &self,
        owner: Option<&zc_expr::HandleId>,
        name: &str,
    ) -> Result<zc_expr::HandleId, zc_expr::ExprError> {
        fn find_obj<'b>(
            objs: &'b [zc_project::Object3D],
            name: &str,
        ) -> Option<&'b zc_project::Object3D> {
            for o in objs {
                if o.name == name {
                    return Some(o);
                }
                if let zc_project::Object3DKind::Group { children, .. } = &o.kind
                    && let Some(found) = find_obj(children, name)
                {
                    return Some(found);
                }
            }
            for o in objs {
                if o.name.eq_ignore_ascii_case(name) {
                    return Some(o);
                }
                if let zc_project::Object3DKind::Group { children, .. } = &o.kind
                    && let Some(found) = find_obj(children, name)
                {
                    return Some(found);
                }
            }
            None
        }

        if let Some(owner_h) = owner {
            if let Some(u) = owner_h
                .as_str()
                .strip_prefix("obj:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, _, _, parent_obj)) = self.state.find_object_owner(u)
                && let zc_project::Object3DKind::Group { children, .. } = &parent_obj.kind
                && let Some(found) = find_obj(children, name)
            {
                return Ok(zc_expr::HandleId::new(format!("obj:{}", found.id)));
            }
            if let Some(u) = owner_h
                .as_str()
                .strip_prefix("layer:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, layer)) = self.state.find_layer_owner(u)
                && let zc_project::LayerKind::Scene3D { objects, .. } = &layer.kind
                && let Some(found) = find_obj(objects, name)
            {
                return Ok(zc_expr::HandleId::new(format!("obj:{}", found.id)));
            }
        }

        if let Some(lid) = self.active_layer_id
            && let Some((_, layer)) = self.state.find_layer_owner(lid)
            && let zc_project::LayerKind::Scene3D { objects, .. } = &layer.kind
            && let Some(found) = find_obj(objects, name)
        {
            return Ok(zc_expr::HandleId::new(format!("obj:{}", found.id)));
        }
        Err(zc_expr::ExprError::NotFound(format!("object '{name}'")))
    }

    fn property(
        &self,
        owner: Option<&zc_expr::HandleId>,
        name: &str,
    ) -> Result<zc_expr::HandleId, zc_expr::ExprError> {
        if let Some(owner_h) = owner {
            let s = owner_h.as_str();
            if s == "seq" {
                return Ok(zc_expr::HandleId::new(format!("prop:seq:{name}")));
            }
            if let Some(u) = s.strip_prefix("obj:").and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, _, _, obj)) = self.state.find_object_owner(u)
                && lookup_object_track(obj, name).is_some()
            {
                return Ok(zc_expr::HandleId::new(format!("prop:obj:{u}:{name}")));
            }
            if let Some(u) = s.strip_prefix("fx:").and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, _, _, fx)) = self.state.find_effect_owner(u)
                && lookup_effect_track(fx, name).is_some()
            {
                return Ok(zc_expr::HandleId::new(format!("prop:fx:{u}:{name}")));
            }
            if let Some(u) = s
                .strip_prefix("layer:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && let Some((_, layer)) = self.state.find_layer_owner(u)
                && (lookup_layer_track(layer, name).is_some()
                    || matches!(
                        name.trim().to_ascii_lowercase().as_str(),
                        "width" | "height" | "resolution"
                    ))
            {
                return Ok(zc_expr::HandleId::new(format!("prop:layer:{u}:{name}")));
            }
            if let Some(u) = s
                .strip_prefix("clip:")
                .and_then(|v| Uuid::parse_str(v).ok())
                && self.state.find_clip_by_id(u).is_some()
                && matches!(
                    name.trim().to_ascii_lowercase().as_str(),
                    "time" | "start" | "length" | "duration"
                )
            {
                return Ok(zc_expr::HandleId::new(format!("prop:clip:{u}:{name}")));
            }
            return Err(zc_expr::ExprError::NotFound(format!(
                "property '{name}' on '{}'",
                owner_h.as_str()
            )));
        }

        if let Some(oid) = self.active_object_id
            && let Some((_, _, _, obj)) = self.state.find_object_owner(oid)
            && lookup_object_track(obj, name).is_some()
        {
            return Ok(zc_expr::HandleId::new(format!("prop:obj:{oid}:{name}")));
        }
        if let Some(fid) = self.active_effect_id
            && let Some((_, _, _, fx)) = self.state.find_effect_owner(fid)
            && lookup_effect_track(fx, name).is_some()
        {
            return Ok(zc_expr::HandleId::new(format!("prop:fx:{fid}:{name}")));
        }
        if let Some(lid) = self.active_layer_id
            && let Some((_, layer)) = self.state.find_layer_owner(lid)
            && (lookup_layer_track(layer, name).is_some()
                || matches!(
                    name.trim().to_ascii_lowercase().as_str(),
                    "width" | "height" | "resolution"
                ))
        {
            return Ok(zc_expr::HandleId::new(format!("prop:layer:{lid}:{name}")));
        }
        if let Some(cid) = self.active_clip_id
            && matches!(
                name.trim().to_ascii_lowercase().as_str(),
                "time" | "start" | "length" | "duration"
            )
        {
            return Ok(zc_expr::HandleId::new(format!("prop:clip:{cid}:{name}")));
        }
        Err(zc_expr::ExprError::NotFound(format!("property '{name}'")))
    }

    fn prop(&self, path: &str) -> Result<zc_expr::HandleId, zc_expr::ExprError> {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            return Err(zc_expr::ExprError::NotFound("empty prop path".to_owned()));
        }
        if let Some(rel) = trimmed.strip_prefix("./") {
            return self.property(None, rel);
        }
        if let Some(rel) = trimmed.strip_prefix("../") {
            let cur_owner = self
                .this_object_id()
                .or_else(|| self.this_layer_id())
                .or_else(|| self.this_clip_id());
            if let Some(cur_h) = cur_owner
                && let Ok(Some(parent_ref)) = self.parent_of(&cur_h, 1)
            {
                let parent_id = match parent_ref {
                    zc_expr::HandleRef::Object(h)
                    | zc_expr::HandleRef::Layer(h)
                    | zc_expr::HandleRef::Clip(h)
                    | zc_expr::HandleRef::Effect(h)
                    | zc_expr::HandleRef::Property(h) => Some(h),
                    zc_expr::HandleRef::Sequence => Some(zc_expr::HandleId::new("seq")),
                    zc_expr::HandleRef::Controls => None,
                };
                return self.property(parent_id.as_ref(), rel);
            }
            return Err(zc_expr::ExprError::NotFound(format!("prop('{path}')")));
        }

        let parts: Vec<&str> = trimmed
            .trim_start_matches('/')
            .split('/')
            .filter(|p| !p.is_empty())
            .collect();
        if parts.is_empty() {
            return Err(zc_expr::ExprError::NotFound(format!("prop('{path}')")));
        }
        if parts.len() == 1 {
            return self.property(None, parts[0]);
        }
        let mut current = self.layer(&zc_expr::NameOrIndex::Name(parts[0].to_owned()))?;
        for seg in &parts[1..parts.len() - 1] {
            if let Ok(obj_h) = self.object(Some(&current), seg) {
                current = obj_h;
            } else if let Ok(fx_h) = self.effect(Some(&current), seg) {
                current = fx_h;
            } else {
                return Err(zc_expr::ExprError::NotFound(format!(
                    "prop('{path}') segment '{seg}'"
                )));
            }
        }
        self.property(Some(&current), parts[parts.len() - 1])
    }

    fn control(
        &self,
        name: &str,
        stable_id: Option<&str>,
    ) -> Result<zc_expr::Value, zc_expr::ExprError> {
        // Strict ID lookup matching `zc-render`: when `stable_id` is given, never fall back to `name`.
        let ctrl_opt = if let Some(sid) = stable_id {
            self.state
                .sequence
                .controls
                .iter()
                .find(|c| c.id == sid)
                .or_else(|| {
                    self.state
                        .project
                        .sequence
                        .controls
                        .iter()
                        .find(|c| c.id == sid)
                })
        } else {
            self.state
                .sequence
                .controls
                .iter()
                .find(|c| c.name == name)
                .or_else(|| {
                    self.state
                        .project
                        .sequence
                        .controls
                        .iter()
                        .find(|c| c.name == name)
                })
        };

        let Some(ctrl) = ctrl_opt else {
            return Err(match stable_id {
                Some(sid) => zc_expr::ExprError::NotFound(format!("control '{name}' (id '{sid}')")),
                None => zc_expr::ExprError::NotFound(format!("control '{name}'")),
            });
        };

        if let Some(cached) = self.state.control_cache.borrow().get(&ctrl.id).cloned() {
            return Ok(cached);
        }
        let cycle_key = format!("ctrl:{}", ctrl.id);
        if !self
            .state
            .active_stack
            .borrow_mut()
            .insert(cycle_key.clone())
        {
            return Err(zc_expr::ExprError::CycleDetected(cycle_key));
        }
        let evaluated = self.state.eval_control(ctrl, &cycle_key);
        self.state.active_stack.borrow_mut().remove(&cycle_key);
        if let Ok(val) = &evaluated {
            self.state
                .control_cache
                .borrow_mut()
                .insert(ctrl.id.clone(), val.clone());
        }
        evaluated
    }
}
