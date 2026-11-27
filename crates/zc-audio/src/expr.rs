//! Deterministic scalar expression evaluator for audio clip properties
//! (`time`, `volume`, `pan`) powered by [`zc_expr`].
//!
//! Supports full Expressions+ and ClipMaker 3 JavaScript-subset semantics:
//! arithmetic, comparisons, logical ops, ternary (`?:`), `if`/`else`, `for`/`while`,
//! `switch`, `var`/`let`/`const`, user-defined and arrow functions, arrays/objects,
//! `frame`, `time`, `fps`, `value`, `Math.*` constants/functions, CM3/Expressions+
//! built-ins (`linear`, `ease`, `easeIn`, `easeOut`, `clamp`, `wiggle`, `seedRandom`,
//! `random`, `gaussRandom`, `wave`, `shake`, `degreesToRadians`, `radiansToDegrees`),
//! and `control(name, id?)` / `controls.<Name>` lookups against
//! [`zc_project::SequenceControl`].
//!
//! Any syntax or runtime error gracefully falls back to the caller's base keyframed `value`.

use std::collections::HashMap;

use zc_expr::{CompiledExpr, EvalConfig, ExprContext, ExprError, ExprMode, Value};

/// Maximum VM execution steps per audio expression call to bound pathological loops
/// while permitting genuine JS control flow (`if`/`else`, `for`/`while`, helpers, `wiggle`).
const AUDIO_EXPR_MAX_STEPS: u64 = 16_384;
/// Maximum nested call / recursion depth for audio expressions.
const AUDIO_EXPR_MAX_RECURSION: usize = 64;

/// Runtime variables supplied to an audio scalar expression evaluation.
#[derive(Debug, Clone)]
pub struct AudioExprContext<'a> {
    /// Clip-local frame number (`frame`).
    pub frame: f64,
    /// Clip time in seconds (`time`). For `time` expressions, this is the pre-expression
    /// keyframed time; for `volume`/`pan`, this is the evaluated clip media time.
    pub time: f64,
    /// Sequence frame rate (`fps`).
    pub fps: f64,
    /// Pre-expression keyframed scalar value (`value`).
    pub value: f64,
    /// Resolved sequence controls keyed by `name` (for `control(name)`).
    pub controls_by_name: &'a HashMap<String, f64>,
    /// Resolved sequence controls keyed by stable `id` (for `control(name, id)`).
    pub controls_by_id: &'a HashMap<String, f64>,
}

struct AudioExprBridge<'a> {
    ctx: &'a AudioExprContext<'a>,
}

impl ExprContext for AudioExprBridge<'_> {
    fn frame(&self) -> f64 {
        self.ctx.frame
    }

    fn time(&self) -> f64 {
        self.ctx.time
    }

    fn fps(&self) -> f64 {
        self.ctx.fps
    }

    fn value(&self) -> Value {
        Value::Number(self.ctx.value)
    }

    fn control(&self, name: &str, stable_id: Option<&str>) -> Result<Value, ExprError> {
        if let Some(id) = stable_id {
            return self
                .ctx
                .controls_by_id
                .get(id)
                .copied()
                .map(Value::Number)
                .ok_or_else(|| ExprError::NotFound(format!("unknown control id `{id}`")));
        }
        self.ctx
            .controls_by_name
            .get(name)
            .copied()
            .map(Value::Number)
            .ok_or_else(|| ExprError::NotFound(format!("unknown control `{name}`")))
    }
}

/// Pre-parsed audio scalar expression backed by [`zc_expr::CompiledExpr`] for repeated
/// evaluation without re-lexing or re-parsing.
#[derive(Debug, Clone)]
pub struct CompiledAudioExpr {
    inner: Option<CompiledExpr>,
}

impl CompiledAudioExpr {
    /// Returns `true` if the compiled expression contains no executable statements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_none()
    }
}

/// Tokenizes and parses a CM3 / Expressions+ scalar expression once for repeated evaluation.
pub fn compile_audio_expression(source: &str) -> Result<CompiledAudioExpr, String> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return Ok(CompiledAudioExpr { inner: None });
    }

    let compiled =
        CompiledExpr::compile(trimmed, ExprMode::ExpressionsPlus).map_err(|err| err.to_string())?;
    if compiled.program().body.is_empty() {
        return Ok(CompiledAudioExpr { inner: None });
    }

    // Audio expressions always expect Expressions+ scalar `clamp(x, lo, hi)` and 5-arg
    // `linear`/`ease` remaps alongside 3-arg CM3 forms even when an expression does not
    // reference another extended identifier. Prepending a no-op `undefined;` statement
    // opts the program into extended semantics without altering completion value or scope.
    let compiled = if compiled.is_extended() {
        compiled
    } else {
        let extended_src = format!("undefined;\n{trimmed}");
        CompiledExpr::compile(&extended_src, ExprMode::ExpressionsPlus)
            .map_err(|err| err.to_string())?
    };

    Ok(CompiledAudioExpr {
        inner: Some(compiled),
    })
}

/// Evaluates a pre-compiled [`CompiledAudioExpr`] against `ctx`.
pub fn eval_compiled_audio_expression(
    compiled: &CompiledAudioExpr,
    ctx: &AudioExprContext<'_>,
) -> Result<f64, String> {
    let Some(expr) = compiled.inner.as_ref() else {
        return Ok(ctx.value);
    };
    let bridge = AudioExprBridge { ctx };
    let config = EvalConfig {
        mode: ExprMode::ExpressionsPlus,
        max_steps: AUDIO_EXPR_MAX_STEPS,
        max_recursion: AUDIO_EXPR_MAX_RECURSION,
    };
    let value = expr
        .evaluate(&bridge, &config)
        .map_err(|err| err.to_string())?;
    value_to_f64(&value)
}

/// Evaluates a pre-compiled [`CompiledAudioExpr`], falling back to `ctx.value` on error or
/// non-finite output.
#[must_use]
pub fn eval_compiled_audio_expression_or_fallback(
    compiled: &CompiledAudioExpr,
    ctx: &AudioExprContext<'_>,
) -> f64 {
    match eval_compiled_audio_expression(compiled, ctx) {
        Ok(v) if v.is_finite() => v,
        Ok(_) => ctx.value,
        Err(err) => {
            tracing::debug!("compiled audio expression fallback ({err})");
            ctx.value
        }
    }
}

/// Evaluates a CM3 / Expressions+ scalar expression for an audio clip property,
/// falling back to `ctx.value` if the expression is empty, unsupported, non-finite, or errors.
#[must_use]
pub fn eval_audio_expression_or_fallback(source: &str, ctx: &AudioExprContext<'_>) -> f64 {
    match eval_audio_expression(source, ctx) {
        Ok(v) if v.is_finite() => v,
        Ok(_) => ctx.value,
        Err(err) => {
            tracing::debug!("audio expression fallback ({err}): {source}");
            ctx.value
        }
    }
}

/// Evaluates a CM3 / Expressions+ scalar expression for an audio clip property.
pub fn eval_audio_expression(source: &str, ctx: &AudioExprContext<'_>) -> Result<f64, String> {
    let compiled = compile_audio_expression(source)?;
    eval_compiled_audio_expression(&compiled, ctx)
}

fn value_to_f64(value: &Value) -> Result<f64, String> {
    match value {
        Value::Number(n) => Ok(*n),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("cannot coerce string `{s}` to number")),
        Value::Undefined => Err("audio expression evaluated to undefined".to_owned()),
        Value::Null => Err("audio expression evaluated to null".to_owned()),
        Value::Array(_) => Err("cannot coerce array to scalar audio value".to_owned()),
        Value::Object(_) => Err("cannot coerce object to scalar audio value".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;

    #[test]
    fn evaluates_audio_volume_pan_and_time_expressions() {
        let mut by_name = HashMap::new();
        by_name.insert("Gain".to_owned(), 0.75);
        let mut by_id = HashMap::new();
        by_id.insert("c_gain".to_owned(), 0.5);

        let ctx = AudioExprContext {
            frame: 15.0,
            time: 0.5,
            fps: 30.0,
            value: 1.0,
            controls_by_name: &by_name,
            controls_by_id: &by_id,
        };

        // linear remap 5-arg
        let vol = eval_audio_expression_or_fallback("linear(frame, 0, 30, 0, 1)", &ctx);
        assert!((vol - 0.5).abs() < 1e-9);

        // Quadratic easeIn / easeOut
        let ein = eval_audio_expression_or_fallback("easeIn(0.5, 0, 10)", &ctx);
        assert!((ein - 2.5).abs() < 1e-9);
        let eout = eval_audio_expression_or_fallback("easeOut(0.5, 0, 10)", &ctx);
        assert!((eout - 7.5).abs() < 1e-9);

        // Math.sin pan modulation
        let pan = eval_audio_expression_or_fallback("Math.sin(time * Math.PI)", &ctx);
        assert!((pan - 1.0).abs() < 1e-9);

        // Sequence control by name and by explicit id (no fallback when explicit id missing)
        assert!((eval_audio_expression_or_fallback("control('Gain')", &ctx) - 0.75).abs() < 1e-9);
        assert!(
            (eval_audio_expression_or_fallback("control('Gain', 'c_gain')", &ctx) - 0.5).abs()
                < 1e-9
        );
        // Missing explicit id falls back to base value (1.0) rather than control('Gain')
        assert!(
            (eval_audio_expression_or_fallback("control('Gain', 'missing')", &ctx) - 1.0).abs()
                < 1e-9
        );

        // Multi-statement with var and ternary
        let speed = eval_audio_expression_or_fallback(
            "var t = time * 2; return frame >= 15 ? t + 0.25 : 0;",
            &ctx,
        );
        assert!((speed - 1.25).abs() < 1e-9);

        // Invalid expression gracefully falls back to `ctx.value`
        let fb = eval_audio_expression_or_fallback("thisLayer.toWorld([0,0,0])", &ctx);
        assert!((fb - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cm3_parity_for_random_wave_linear_and_ease() {
        let empty = HashMap::new();
        let ctx = AudioExprContext {
            frame: 15.0,
            time: 0.5,
            fps: 30.0,
            value: 0.0,
            controls_by_name: &empty,
            controls_by_id: &empty,
        };

        // 1. random(seed): fract(10000 * sin(seed)) in [0, 1), including negative sin(seed).
        for seed in [0.0_f64, 1.0, 4.0, 42.0, -1.5, 123.456] {
            let v = 10_000.0 * seed.sin();
            let expected = v - v.floor();
            let got = eval_audio_expression(&format!("random({seed})"), &ctx).unwrap();
            assert!(
                (got - expected).abs() < 1e-12,
                "random({seed}): got {got}, expected {expected}"
            );
            assert!(
                (0.0..1.0).contains(&got),
                "random({seed}) out of [0, 1): {got}"
            );
        }

        // 2. wave(x): CM3 square wave `2 * ((trunc(x) as i32) & 1) - 1`.
        for (expr, expected) in [
            ("wave(0)", -1.0),
            ("wave(1)", 1.0),
            ("wave(2)", -1.0),
            ("wave(3.9)", 1.0),
            ("wave(2.9)", -1.0),
            ("wave(-1.5)", 1.0),
            ("wave(-2.5)", -1.0),
        ] {
            let got = eval_audio_expression(expr, &ctx).unwrap();
            assert!(
                (got - expected).abs() < 1e-12,
                "{expr}: got {got}, expected {expected}"
            );
        }

        // 3. 3-arg linear(a, b, t): CM3 parameter order `(b - a) * t + a` (unclamped), plus 5-arg remap.
        assert!(
            (eval_audio_expression("linear(10, 20, 0.25)", &ctx).unwrap() - 12.5).abs() < 1e-12
        );
        assert!((eval_audio_expression("linear(10, 20, 2.0)", &ctx).unwrap() - 30.0).abs() < 1e-12);
        assert!(
            (eval_audio_expression("linear(0.5, 0, 1, 100, 200)", &ctx).unwrap() - 150.0).abs()
                < 1e-12
        );

        // 4. ease(t, family, dir): CM3 string easing curves (including Cubic OutIn).
        assert!(
            (eval_audio_expression("ease(0.25, \"Cubic\", \"OutIn\")", &ctx).unwrap() - 0.4375)
                .abs()
                < 1e-12
        );
        assert!(
            (eval_audio_expression("ease(0.5, 'Quadratic', 'In')", &ctx).unwrap() - 0.25).abs()
                < 1e-12
        );
        assert!(
            (eval_audio_expression("ease(0.5, 'Quadratic', 'InOut')", &ctx).unwrap() - 0.5).abs()
                < 1e-12
        );
        assert!(
            (eval_audio_expression("ease(0.5, 'Cubic', 'Out')", &ctx).unwrap() - 0.875).abs()
                < 1e-12
        );
        assert!(
            (eval_audio_expression("ease(0.25, 'Linear', 'In')", &ctx).unwrap() - 0.25).abs()
                < 1e-12
        );
        assert!(
            (eval_audio_expression("ease(0.25, 'Sinusoidal', 'InOut')", &ctx).unwrap()
                - 0.5 * (1.0 - (PI * 0.25).cos()))
            .abs()
                < 1e-12
        );
        // 1-arg default Quadratic.InOut and 5-arg smoothstep still work.
        assert!((eval_audio_expression("ease(0.25)", &ctx).unwrap() - 0.125).abs() < 1e-12);
        assert!(
            (eval_audio_expression("ease(0.25, 0, 1, 0, 100)", &ctx).unwrap() - 15.625).abs()
                < 1e-12
        );
        // Unknown curve returns error (and fallback returns ctx.value).
        assert!(eval_audio_expression("ease(0.5, 'NoSuchCurve', 'In')", &ctx).is_err());
    }

    #[test]
    fn compiled_audio_expression_matches_uncompiled_across_frames() {
        let mut by_name = HashMap::new();
        by_name.insert("Gain".to_owned(), 0.65);
        let mut by_id = HashMap::new();
        by_id.insert("c_pan".to_owned(), -0.4);

        let expressions = [
            "linear(frame, 0, 60, 0.2, 1.0) * control('Gain')",
            "var ph = time * Math.PI * 2; return Math.sin(ph) * control('Pan', 'c_pan');",
            "frame >= 30 ? ease(time, 'Cubic', 'InOut') : clamp(value + wave(frame) * 0.1, 0, 1)",
            "random(frame + 1) * 0.5 + 0.25",
            "   // comments only",
        ];

        for src in expressions {
            let compiled = compile_audio_expression(src).expect("valid expression compiles");
            for f in 0..64 {
                let ctx = AudioExprContext {
                    frame: f64::from(f),
                    time: f64::from(f) / 30.0,
                    fps: 30.0,
                    value: 0.8,
                    controls_by_name: &by_name,
                    controls_by_id: &by_id,
                };
                let expected = eval_audio_expression_or_fallback(src, &ctx);
                let got = eval_compiled_audio_expression_or_fallback(&compiled, &ctx);
                assert!(
                    (got - expected).abs() < 1e-12,
                    "mismatch for `{src}` at frame {f}: got {got}, expected {expected}"
                );
            }
        }

        // Syntax error rejected once at compile time.
        assert!(compile_audio_expression("1 + (2 * 3").is_err());
    }

    #[test]
    fn supports_full_js_control_flow_functions_wiggle_and_scalar_clamp() {
        let mut by_name = HashMap::new();
        by_name.insert("Duck".to_owned(), 0.35);
        let empty = HashMap::new();
        let ctx = AudioExprContext {
            frame: 24.0,
            time: 0.8,
            fps: 30.0,
            value: 0.75,
            controls_by_name: &by_name,
            controls_by_id: &empty,
        };

        // 1. if/else + let + for loop + block completion value
        let loop_expr = r#"
            let acc = 0;
            if (frame > 10) {
                for (let i = 1; i <= 4; i++) {
                    acc += i * 0.05;
                }
            } else {
                acc = 0.1;
            }
            clamp(acc, 0, 1)
        "#;
        let loop_val = eval_audio_expression(loop_expr, &ctx).expect("JS control flow evaluates");
        // (1 + 2 + 3 + 4) * 0.05 = 0.5
        assert!((loop_val - 0.5).abs() < 1e-12, "got {loop_val}");

        // 2. Named helper + arrow function + controls.<Name> property syntax
        let fn_expr = r#"
            function envelope(t) {
                const scale = (x) => ease(x, 0, 1, 0.0, 1.0);
                if (t < 0.5) {
                    return scale(t * 2);
                }
                return 1.0 - controls.Duck;
            }
            envelope(time)
        "#;
        let fn_val = eval_audio_expression(fn_expr, &ctx).expect("functions + controls evaluate");
        assert!((fn_val - 0.65).abs() < 1e-12, "got {fn_val}");

        // 3. Deterministic wiggle() around ctx.value
        let w1 = eval_audio_expression("wiggle(2, 0.2)", &ctx).expect("wiggle evaluates");
        let w2 = eval_audio_expression("wiggle(2, 0.2)", &ctx).expect("wiggle is deterministic");
        assert!((w1 - w2).abs() < 1e-12);
        assert!(
            (w1 - ctx.value).abs() <= 0.2 + 1e-9,
            "wiggle must stay within amplitude bounds around base value: got {w1}"
        );

        // 4. Scalar clamp without referencing `value` or `fps`
        assert!((eval_audio_expression("clamp(1.75, 0, 1)", &ctx).unwrap() - 1.0).abs() < 1e-12);
        assert!((eval_audio_expression("clamp(-0.5, 0, 1)", &ctx).unwrap() - 0.0).abs() < 1e-12);

        // 5. Infinite loop hits step budget and falls back cleanly to ctx.value
        let infinite = "while (true) {}";
        assert!(eval_audio_expression(infinite, &ctx).is_err());
        assert!((eval_audio_expression_or_fallback(infinite, &ctx) - ctx.value).abs() < 1e-12);
    }
}
