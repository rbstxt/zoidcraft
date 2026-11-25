use super::*;

pub(super) fn truncate_diag(s: &str) -> std::borrow::Cow<'_, str> {
    const MAX_DIAG_NAME_BYTES: usize = 128;
    if s.len() <= MAX_DIAG_NAME_BYTES {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut end = MAX_DIAG_NAME_BYTES;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    std::borrow::Cow::Owned(format!("{}...", &s[..end]))
}

pub(super) fn is_catchable_error(err: &ExprError) -> bool {
    !matches!(
        err,
        ExprError::StepLimitExceeded { .. }
            | ExprError::RecursionLimitExceeded { .. }
            | ExprError::Forbidden(_)
            | ExprError::CycleDetected(_)
    )
}

pub(super) fn normalize_slice_index(idx: f64, len: isize) -> usize {
    if idx.is_nan() {
        return 0;
    }
    if !idx.is_finite() {
        return if idx.is_sign_positive() {
            len.max(0) as usize
        } else {
            0
        };
    }
    let i = idx.trunc() as isize;
    if i < 0 {
        (len + i).max(0) as usize
    } else {
        i.min(len).max(0) as usize
    }
}

pub(super) fn static_math_builtin_name(name: &str) -> Option<&'static str> {
    Some(match name {
        "abs" => "Math.abs",
        "acos" => "Math.acos",
        "acosh" => "Math.acosh",
        "asin" => "Math.asin",
        "asinh" => "Math.asinh",
        "atan" => "Math.atan",
        "atan2" => "Math.atan2",
        "atanh" => "Math.atanh",
        "cbrt" => "Math.cbrt",
        "ceil" => "Math.ceil",
        "clz32" => "Math.clz32",
        "cos" => "Math.cos",
        "cosh" => "Math.cosh",
        "exp" => "Math.exp",
        "expm1" => "Math.expm1",
        "floor" => "Math.floor",
        "fround" => "Math.fround",
        "hypot" => "Math.hypot",
        "imul" => "Math.imul",
        "log" => "Math.log",
        "log1p" => "Math.log1p",
        "log2" => "Math.log2",
        "log10" => "Math.log10",
        "max" => "Math.max",
        "min" => "Math.min",
        "pow" => "Math.pow",
        "round" => "Math.round",
        "sign" => "Math.sign",
        "sin" => "Math.sin",
        "sinh" => "Math.sinh",
        "sqrt" => "Math.sqrt",
        "tan" => "Math.tan",
        "tanh" => "Math.tanh",
        "trunc" => "Math.trunc",
        _ => return None,
    })
}

pub(super) fn static_global_builtin_name(name: &str) -> Option<&'static str> {
    Some(match name {
        "isNaN" => "isNaN",
        "isFinite" => "isFinite",
        "parseInt" => "parseInt",
        "parseFloat" => "parseFloat",
        "add" => "add",
        "sub" => "sub",
        "mul" => "mul",
        "div" => "div",
        "clamp" => "clamp",
        "dot" => "dot",
        "cross" => "cross",
        "normalize" => "normalize",
        "length" => "length",
        "wave" => "wave",
        "lerp" => "lerp",
        "linear" => "linear",
        "catmullRom" => "catmullRom",
        "ease" => "ease",
        "random" => "random",
        "gaussRandom" => "gaussRandom",
        "shake" => "shake",
        "easeIn" => "easeIn",
        "easeOut" => "easeOut",
        "degreesToRadians" => "degreesToRadians",
        "radiansToDegrees" => "radiansToDegrees",
        "lookAt" => "lookAt",
        "seedRandom" => "seedRandom",
        "wiggle" => "wiggle",
        "layer" => "layer",
        "effect" => "effect",
        "object" => "object",
        "property" => "property",
        "prop" => "prop",
        "control" => "control",
        "toWorld" => "toWorld",
        "fromWorld" => "fromWorld",
        "worldPosition" => "worldPosition",
        "worldRotation" => "worldRotation",
        "toScreen" => "toScreen",
        "fromScreen" => "fromScreen",
        _ => return None,
    })
}

pub(super) fn is_methods_member(name: &str, mode: ExprMode) -> bool {
    if matches!(
        name,
        "add"
            | "sub"
            | "mul"
            | "div"
            | "clamp"
            | "dot"
            | "cross"
            | "normalize"
            | "length"
            | "wave"
            | "lerp"
            | "linear"
            | "catmullRom"
            | "ease"
            | "random"
            | "gaussRandom"
            | "shake"
    ) {
        return true;
    }
    mode == ExprMode::ExpressionsPlus
        && matches!(
            name,
            "wiggle" | "easeIn" | "easeOut" | "degreesToRadians" | "radiansToDegrees"
        )
}

pub(super) struct PropertyEnterGuard<'a> {
    pub(super) ctx: &'a dyn ExprContext,
    pub(super) id: &'a HandleId,
}

impl Drop for PropertyEnterGuard<'_> {
    fn drop(&mut self) {
        self.ctx.exit_property(self.id);
    }
}
