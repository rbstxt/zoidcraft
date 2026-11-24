#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use zc_expr::{
    BasicContext, EvalConfig, ExprContext, ExprError, ExprMode, HandleId, KeyframeRecord, Value,
    compile, eval,
};

fn eval_plus(src: &str) -> Value {
    eval(
        src,
        &BasicContext::default(),
        &EvalConfig::expressions_plus(),
    )
    .unwrap_or_else(|e| panic!("expected Ok for `{src}`, got {e:?}"))
}

fn eval_plus_err(src: &str) -> ExprError {
    eval(
        src,
        &BasicContext::default(),
        &EvalConfig::expressions_plus(),
    )
    .expect_err(&format!("expected Err for `{src}`"))
}

struct AuditContext {
    frame: f64,
    value: Value,
    properties: BTreeMap<String, Value>,
    cyclic_prop_id: HandleId,
    keyframes: Vec<KeyframeRecord>,
}

impl Default for AuditContext {
    fn default() -> Self {
        Self {
            frame: 0.0,
            value: Value::vec3(1.0, 2.0, 3.0),
            properties: BTreeMap::new(),
            cyclic_prop_id: HandleId::new("cyclic"),
            keyframes: vec![KeyframeRecord {
                index: 1,
                time: 0.0,
                value: Value::Number(10.0),
            }],
        }
    }
}

impl ExprContext for AuditContext {
    fn frame(&self) -> f64 {
        self.frame
    }

    fn value(&self) -> Value {
        self.value.clone()
    }

    fn properties(&self) -> BTreeMap<String, Value> {
        self.properties.clone()
    }

    fn this_layer_id(&self) -> Option<HandleId> {
        Some(HandleId::new("layer1"))
    }

    fn prop(&self, path: &str) -> Result<HandleId, ExprError> {
        if path == "cyclic" {
            Ok(self.cyclic_prop_id.clone())
        } else {
            Err(ExprError::NotFound(path.to_string()))
        }
    }

    fn property_value(&self, id: &HandleId) -> Result<Value, ExprError> {
        if id == &self.cyclic_prop_id {
            Err(ExprError::CycleDetected("cyclic".to_string()))
        } else if self.this_property_id().as_ref() == Some(id) {
            Ok(self.value.clone())
        } else {
            Err(ExprError::NotFound(id.as_str().to_string()))
        }
    }

    fn property_key(
        &self,
        _id: &HandleId,
        index_1_based: usize,
    ) -> Result<KeyframeRecord, ExprError> {
        self.keyframes
            .get(index_1_based - 1)
            .cloned()
            .ok_or_else(|| ExprError::RangeError("no such key".to_string()))
    }
}

#[test]
fn bug01_json_parse_utf8_boundary_surrogates_and_plus_sign() {
    // Must not panic when `\u` crosses a multi-byte UTF-8 char boundary.
    assert!(matches!(
        eval_plus_err(r#"JSON.parse('"\\u00あ"')"#),
        ExprError::SyntaxError { .. }
    ));
    // Must reject leading `+` in `\uXXXX`.
    assert!(matches!(
        eval_plus_err(r#"JSON.parse('"\\u+041"')"#),
        ExprError::SyntaxError { .. }
    ));
    // Valid RFC 8259 UTF-16 surrogate pair decodes cleanly.
    assert_eq!(
        eval_plus(r#"JSON.parse('"\\uD83D\\uDE00"')"#),
        Value::String("😀".to_string())
    );
    // Unpaired surrogates return SyntaxError cleanly.
    assert!(matches!(
        eval_plus_err(r#"JSON.parse('"\\uD83D"')"#),
        ExprError::SyntaxError { .. }
    ));
    assert!(matches!(
        eval_plus_err(r#"JSON.parse('"\\uDE00"')"#),
        ExprError::SyntaxError { .. }
    ));
}

#[test]
fn bug02_arrow_param_lookahead_no_exponential_backtracking() {
    let nested = format!("{}1{}", "(a=".repeat(40), ")".repeat(40));
    let t0 = Instant::now();
    let _ = compile(&nested, ExprMode::ExpressionsPlus);
    let _ = compile(&nested, ExprMode::VanillaCm3);
    assert!(
        t0.elapsed() < Duration::from_secs(1),
        "nested parenthesized assignment took too long: {:?}",
        t0.elapsed()
    );

    // Valid arrow functions with nested defaults and destructuring still parse and run.
    assert_eq!(
        eval_plus("const f = (a, b = (c) => c + 1) => a + b(2); f(10)"),
        Value::Number(13.0)
    );
    assert_eq!(
        eval_plus("const g = ({ x = 5 }) => x * 2; g({})"),
        Value::Number(10.0)
    );
    assert!(matches!(
        compile("(a) => a", ExprMode::VanillaCm3),
        Err(ExprError::ModeError { .. })
    ));
}

#[test]
fn bug03_try_finally_does_not_swallow_fatal_sandbox_errors() {
    // Forbidden error must not be swallowed by finally return.
    assert!(matches!(
        eval_plus_err(
            r#"function f() { try { const k = "ran" + "dom"; Math[k](); } finally { return 1; } } f()"#
        ),
        ExprError::Forbidden(_)
    ));

    // RecursionLimitExceeded must not be swallowed by finally return.
    let rec_cfg = EvalConfig {
        mode: ExprMode::ExpressionsPlus,
        max_steps: EvalConfig::DEFAULT_MAX_STEPS,
        max_recursion: 32,
    };
    let rec_err = eval(
        "function f() { try { return f(); } finally { return 42; } } f()",
        &BasicContext::default(),
        &rec_cfg,
    )
    .expect_err("RecursionLimitExceeded must not be swallowed");
    assert!(matches!(
        rec_err,
        ExprError::RecursionLimitExceeded { limit: 32 }
    ));

    // StepLimitExceeded must not be swallowed by finally return.
    let small_cfg = EvalConfig {
        mode: ExprMode::ExpressionsPlus,
        max_steps: 200,
        max_recursion: 64,
    };
    let step_err = eval(
        "function f() { try { while (true) {} } finally { return 1; } } f()",
        &BasicContext::default(),
        &small_cfg,
    )
    .expect_err("StepLimitExceeded must not be swallowed");
    assert!(matches!(step_err, ExprError::StepLimitExceeded { .. }));

    // CycleDetected must not be swallowed by finally return.
    let ctx = AuditContext::default();
    let cyc_err = eval(
        r#"function f() { try { return prop("cyclic") + 1; } finally { return 0; } } f()"#,
        &ctx,
        &EvalConfig::expressions_plus(),
    )
    .expect_err("CycleDetected must not be swallowed");
    assert!(matches!(cyc_err, ExprError::CycleDetected(_)));
}

#[test]
fn bug04_seed_random_and_lazy_random_do_not_corrupt_wiggle() {
    let cfg = EvalConfig::expressions_plus();
    let ctx_f0 = AuditContext {
        frame: 0.0,
        value: Value::Number(0.0),
        ..Default::default()
    };
    let ctx_f100 = AuditContext {
        frame: 100.0,
        value: Value::Number(0.0),
        ..Default::default()
    };

    // Explicit `t = 5.0` with `seedRandom(42)` (timeless=false) must be identical across frames.
    let w0 = eval(
        "seedRandom(42); wiggle(0.5, 10, 1, 0.5, 5.0)",
        &ctx_f0,
        &cfg,
    )
    .unwrap();
    let w100 = eval(
        "seedRandom(42); wiggle(0.5, 10, 1, 0.5, 5.0)",
        &ctx_f100,
        &cfg,
    )
    .unwrap();
    assert_eq!(w0, w100);

    // Subframe continuity for `seedRandom(42); wiggle(0.1, 10)`.
    let ctx_a = AuditContext {
        frame: 10.0,
        value: Value::Number(0.0),
        ..Default::default()
    };
    let ctx_b = AuditContext {
        frame: 10.01,
        value: Value::Number(0.0),
        ..Default::default()
    };
    let wa = eval("seedRandom(42); wiggle(0.1, 10)", &ctx_a, &cfg)
        .unwrap()
        .as_f64()
        .unwrap();
    let wb = eval("seedRandom(42); wiggle(0.1, 10)", &ctx_b, &cfg)
        .unwrap()
        .as_f64()
        .unwrap();
    assert!(
        (wa - wb).abs() < 0.25,
        "wiggle jumped discontinuously across 0.01 frame: {wa} vs {wb}"
    );

    // Unseeded `random()` before `wiggle()` must not change wiggle's seed offset.
    let w_plain = eval("wiggle(0.5, 10, 1, 0.5, 5.0)", &ctx_f100, &cfg).unwrap();
    let w_after_rand = eval("random(); wiggle(0.5, 10, 1, 0.5, 5.0)", &ctx_f100, &cfg).unwrap();
    assert_eq!(w_plain, w_after_rand);

    // Unseeded `random()` before `random(10)` must keep 1-arg numeric `random(10)` as CM3 hash.
    let cm3_ref = eval("random(10)", &ctx_f100, &cfg).unwrap();
    let cm3_after_zero_arg = eval("random(); random(10)", &ctx_f100, &cfg).unwrap();
    assert_eq!(cm3_ref, cm3_after_zero_arg);
}

#[test]
fn bug05_unary_operator_precedence_with_exponentiation() {
    for bad in ["-2 ** 2", "-x ** 2", "~2 ** 2", "!1 ** 2", "2 ** -3 ** 2"] {
        assert!(
            matches!(
                compile(bad, ExprMode::ExpressionsPlus),
                Err(ExprError::SyntaxError { .. })
            ),
            "expected SyntaxError for `{bad}`"
        );
    }

    assert_eq!(eval_plus("(-2) ** 2"), Value::Number(4.0));
    assert_eq!(eval_plus("-(2 ** 2)"), Value::Number(-4.0));
    assert_eq!(eval_plus("2 ** -3"), Value::Number(0.125));
    assert_eq!(eval_plus("let x = 2; ++x ** 2"), Value::Number(9.0));
    assert_eq!(eval_plus("let x = 2; x++ ** 2"), Value::Number(4.0));
}

#[test]
fn bug06_assign_and_update_evaluate_target_once_and_lhs_before_rhs() {
    assert_eq!(
        eval_plus("let a = [10, 20]; let i = 0; a[i++] += 5; [a[0], a[1], i]"),
        Value::from_f64_slice(&[15.0, 20.0, 1.0])
    );
    assert_eq!(
        eval_plus("let a = [10, 20]; let i = 0; let r = a[i++]++; [r, a[0], a[1], i]"),
        Value::from_f64_slice(&[10.0, 11.0, 20.0, 1.0])
    );
    assert_eq!(
        eval_plus("let x = 10; x += (x = 100, 5); x"),
        Value::Number(15.0)
    );
    assert_eq!(
        eval_plus("let a = [0, 0]; let i = 0; a[i] = (i = 1, 42); [a[0], a[1], i]"),
        Value::from_f64_slice(&[42.0, 0.0, 1.0])
    );
    assert_eq!(
        eval_plus(
            "let c = 0; let o = { x: 10 }; function g() { c++; return o; } g().x += 5; [o.x, c]"
        ),
        Value::from_f64_slice(&[15.0, 1.0])
    );
}

#[test]
fn bug07_optional_chaining_short_circuit_and_missing_methods() {
    // Must not evaluate args when optional call target is nullish.
    assert_eq!(
        eval_plus(
            "let ran = 0; const o = { f: null }; const r = o?.f?.(++ran); [r === undefined, ran]"
        ),
        Value::Array(vec![Value::Bool(true), Value::Number(0.0)])
    );

    // Non-optional member access on nullish receiver must throw TypeError even if followed by `?.()`.
    assert!(matches!(
        eval_plus_err("const o = { a: null }; o?.a.b?.()"),
        ExprError::TypeError(_)
    ));

    // Optional calls of absent methods on built-in types, handles, and namespaces yield undefined.
    let ctx = AuditContext::default();
    let cfg = EvalConfig::expressions_plus();
    for expr in [
        "let ran = 0; const r = [1, 2]?.nope?.(++ran); [r === undefined, ran]",
        r#"let ran = 0; const r = "hi"?.nope?.(++ran); [r === undefined, ran]"#,
        "let ran = 0; const r = (5)?.nope?.(++ran); [r === undefined, ran]",
        "let ran = 0; const r = thisLayer?.nope?.(++ran); [r === undefined, ran]",
        "let ran = 0; const r = Math?.nope?.(++ran); [r === undefined, ran]",
    ] {
        let res = eval(expr, &ctx, &cfg).unwrap_or_else(|e| panic!("failed `{expr}`: {e:?}"));
        assert_eq!(
            res,
            Value::Array(vec![Value::Bool(true), Value::Number(0.0)]),
            "unexpected result for `{expr}`"
        );
    }

    // Non-nullish non-callable property still throws TypeError.
    assert!(matches!(
        eval_plus_err("const o = { f: 123 }; o?.f?.(1)"),
        ExprError::TypeError(_)
    ));
}

#[test]
fn bug08_normalize_slice_index_nan_and_undefined() {
    assert_eq!(
        eval_plus("[1, 2, 3].slice(undefined)"),
        Value::from_f64_slice(&[1.0, 2.0, 3.0])
    );
    assert_eq!(
        eval_plus("[1, 2, 3].slice(NaN)"),
        Value::from_f64_slice(&[1.0, 2.0, 3.0])
    );
    assert_eq!(eval_plus("[1, 2, 3].slice(0, NaN)"), Value::Array(vec![]));
    assert_eq!(
        eval_plus(r#""hello".slice(undefined)"#),
        Value::String("hello".to_string())
    );
    assert_eq!(
        eval_plus(r#""hello".slice(NaN, 3)"#),
        Value::String("hel".to_string())
    );
    assert_eq!(
        eval_plus("[1, 2, 3].fill(9, undefined)"),
        Value::from_f64_slice(&[9.0, 9.0, 9.0])
    );
    assert_eq!(
        eval_plus("[1, 2, 3].fill(9, 0, undefined)"),
        Value::from_f64_slice(&[9.0, 9.0, 9.0])
    );
    assert_eq!(
        eval_plus("[1, 2, 3].fill(9, 0, NaN)"),
        Value::from_f64_slice(&[1.0, 2.0, 3.0])
    );
}

#[test]
fn bug09_intern_literal_string_is_o1_in_loops() {
    let huge = "a".repeat(2 * 1024 * 1024);
    let expr = format!(
        r#"var n = 0; for (var i = 0; i < 50000; i++) {{ var s = "{huge}"; n += s.length; }} n"#
    );
    let t0 = Instant::now();
    let res = eval_plus(&expr);
    assert_eq!(res, Value::Number(50_000.0 * (2 * 1024 * 1024) as f64));
    assert!(
        t0.elapsed() < Duration::from_secs(2),
        "literal string lookup loop took too long: {:?}",
        t0.elapsed()
    );
}

#[test]
fn bug10_array_higher_order_methods_do_not_clone_vec_upfront() {
    // Recursive `find` on a large array stays within memory limits without cloning the full Vec.
    let res = eval_plus(
        "var a = Array(65535); var calls = 0; function f(n) { calls++; if (n > 0) a.find(() => { f(n - 1); return true; }); } f(20); calls",
    );
    assert_eq!(res, Value::Number(21.0));

    // In-place mutation during iteration is visible to subsequent indices.
    assert_eq!(
        eval_plus(
            "let a = [1, 2, 3]; let seen = []; a.forEach((v, i) => { seen.push(v); if (i === 0) a[1] = 20; }); seen"
        ),
        Value::from_f64_slice(&[1.0, 20.0, 3.0])
    );

    // `filter` charges heap budget incrementally as kept elements are pushed.
    let err = eval_plus_err(
        "var a = Array(20000).fill(1); var xs = []; for (var i = 0; i < 120; i++) { xs.push(a.filter(() => true)); } xs.length",
    );
    assert!(matches!(err, ExprError::RangeError(_)));
}

#[test]
fn bug11_json_parse_enforces_container_and_heap_limits_incrementally() {
    // Array exceeding MAX_ARRAY_LEN (65,536) is rejected during JSON.parse before interning.
    let err = eval_plus_err(r#"JSON.parse("[" + "0,".repeat(10000).repeat(10) + "0]")"#);
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for huge JSON array, got {err:?}"
    );
}

#[test]
fn bug12_throw_across_function_boundary_preserves_thrown_value() {
    assert_eq!(
        eval_plus(r#"try { (() => { throw "boom"; })(); } catch (e) { e === "boom" }"#),
        Value::Bool(true)
    );
    assert_eq!(
        eval_plus(
            "function inner() { throw { code: 404, msg: 'missing' }; } function outer() { return inner(); } try { outer(); } catch (e) { e.code }"
        ),
        Value::Number(404.0)
    );
    assert_eq!(
        eval_plus(
            "function f() { throw 123; } let side = 0; try { try { f(); } finally { side = 1; } } catch (e) { [e, side] }"
        ),
        Value::from_f64_slice(&[123.0, 1.0])
    );
    assert_eq!(
        eval_plus_err(r#"(() => { throw "uncaught_boom"; })()"#),
        ExprError::Thrown("uncaught_boom".to_string())
    );
}

#[test]
fn bug13_function_decl_can_be_reassigned_inside_body() {
    assert_eq!(
        eval_plus("function foo() { foo = () => 1; return foo(); } foo()"),
        Value::Number(1.0)
    );
    // Named function expression binding remains immutable inside itself.
    assert!(matches!(
        eval_plus_err("const f = function self() { self = 1; }; f()"),
        ExprError::TypeError(_)
    ));
}

#[test]
fn bug14_value_and_properties_cached_per_evaluation() {
    let mut props = BTreeMap::new();
    for i in 0..50 {
        props.insert(format!("k{i}"), Value::Number(i as f64));
    }
    props.insert("speed".to_string(), Value::Number(2.0));
    let ctx = AuditContext {
        value: Value::vec3(1.0, 2.0, 3.0),
        properties: props,
        ..Default::default()
    };
    let cfg = EvalConfig::expressions_plus();

    assert_eq!(
        eval("value === value && properties === properties", &ctx, &cfg).unwrap(),
        Value::Bool(true)
    );

    let sum = eval(
        "var acc = 0; for (var i = 0; i < 50000; i++) { acc += properties.speed; } acc",
        &ctx,
        &cfg,
    )
    .expect("reading properties in a 50k loop should not exhaust heap budget");
    assert_eq!(sum, Value::Number(100_000.0));
}

#[test]
fn bug15_vanilla_cm3_for_in_undeclared_target_fails_at_compile_time() {
    assert!(matches!(
        compile("for (unknownVar in {}) {}", ExprMode::VanillaCm3),
        Err(ExprError::ReferenceError(name)) if name == "unknownVar"
    ));
    assert!(compile("var x; for (x in {}) {}", ExprMode::VanillaCm3).is_ok());
    assert!(compile("for (var x in {}) {}", ExprMode::VanillaCm3).is_ok());
}

#[test]
fn bug16_number_predicates_propagate_property_handle_errors() {
    let ctx = AuditContext::default();
    let cfg = EvalConfig::expressions_plus();
    for expr in [
        r#"Number.isFinite(prop("cyclic"))"#,
        r#"Number.isNaN(prop("cyclic"))"#,
        r#"Number.isInteger(prop("cyclic"))"#,
    ] {
        let err = eval(expr, &ctx, &cfg).expect_err(&format!("expected error for `{expr}`"));
        assert!(
            matches!(err, ExprError::CycleDetected(_)),
            "expected CycleDetected for `{expr}`, got {err:?}"
        );
    }
}

#[test]
fn bug17_string_replace_budget_when_pattern_not_found() {
    // Must not falsely reject a no-op replace when `s.len() + to.len() > MAX_STRING_BYTES`.
    let res = eval_plus(
        r#"var s = "x".repeat(10000).repeat(400); var rep = "b".repeat(10000).repeat(200); s.replace("z", rep).length"#,
    );
    assert_eq!(res, Value::Number(4_000_000.0));

    // Normal replacement still works accurately.
    assert_eq!(
        eval_plus(r#""hello world world".replace("world", "rust")"#),
        Value::String("hello rust world".to_string())
    );
}

#[test]
fn bug18_namespace_strict_eq_and_array_includes_nan_and_from_index() {
    assert_eq!(
        eval_plus(
            "Math === Math && Array === Array && Object === Object && Number === Number && String === String && Boolean === Boolean && JSON === JSON && Math !== JSON"
        ),
        Value::Bool(true)
    );
    assert_eq!(eval_plus("[NaN].includes(NaN)"), Value::Bool(true));
    assert_eq!(eval_plus("[1, 2, 1].includes(1, 1)"), Value::Bool(true));
    assert_eq!(eval_plus("[1, 2, 3].includes(1, 1)"), Value::Bool(false));
    assert_eq!(eval_plus("[1, 2, 1].indexOf(1, 1)"), Value::Number(2.0));
    assert_eq!(eval_plus("[1, 2, 1].indexOf(1, -1)"), Value::Number(2.0));
}

#[test]
fn bug19_is_cacheable_rejects_aliased_mutations_loop_invalidation_and_meta_ops() {
    let non_cacheable = [
        "let o = { p: properties }; o.p.x = 1;",
        "let a = [value]; a[0].push(1);",
        "let a = [value]; a?.[0]?.push(1);",
        "let a = {}; while (time > 0) { a.x = 1; a = properties; }",
        "let a = {}; do { a.x = 1; a = properties; } while (time > 0);",
        "let a = {}; for (let i = 0; i < 2; i++) { a.x = 1; a = properties; }",
        r#"let o = {}; delete o["__proto__"];"#,
        r#"let o = {}; o["constructor"]++;"#,
    ];
    for sample in non_cacheable {
        let compiled = compile(sample, ExprMode::ExpressionsPlus)
            .unwrap_or_else(|e| panic!("failed to compile `{sample}`: {e:?}"));
        assert!(
            !compiled.is_cacheable(),
            "expected `{sample}` to be non-cacheable"
        );
    }
}

#[test]
fn bug20_keyframe_to_object_charges_heap_budget() {
    let ctx = AuditContext::default();
    let cfg = EvalConfig::expressions_plus();
    let err = eval(
        "var xs = []; for (var i = 0; i < 700000; i++) { thisProperty.key(1); } xs.length",
        &ctx,
        &cfg,
    )
    .expect_err("700k keyframe object allocations must exceed 64 MiB heap budget");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for keyframe object allocation bomb, got {err:?}"
    );
}

#[test]
fn bug21_object_destructuring_rejects_non_ident_shorthand() {
    for bad in [
        r#"const { "a b" } = {};"#,
        "const { 1 } = {};",
        "const { if } = {};",
    ] {
        assert!(
            matches!(
                compile(bad, ExprMode::ExpressionsPlus),
                Err(ExprError::SyntaxError { .. })
            ),
            "expected SyntaxError for `{bad}`"
        );
    }

    assert_eq!(
        eval_plus(r#"const { "a b": x, 1: y, if: z } = { "a b": 1, 1: 2, if: 3 }; [x, y, z]"#),
        Value::from_f64_slice(&[1.0, 2.0, 3.0])
    );
}

#[test]
fn bug22_number_to_fixed_and_to_string_radix() {
    assert_eq!(
        eval_plus("(1.5).toFixed(undefined)"),
        Value::String("2".to_string())
    );
    assert_eq!(
        eval_plus("(Infinity).toFixed(2)"),
        Value::String("Infinity".to_string())
    );
    assert_eq!(
        eval_plus("(-Infinity).toFixed(2)"),
        Value::String("-Infinity".to_string())
    );
    assert_eq!(
        eval_plus("(NaN).toFixed(2)"),
        Value::String("NaN".to_string())
    );

    assert_eq!(
        eval_plus("(255).toString(16)"),
        Value::String("ff".to_string())
    );
    assert_eq!(
        eval_plus("(-255).toString(16)"),
        Value::String("-ff".to_string())
    );
    assert_eq!(
        eval_plus("(10).toString(2)"),
        Value::String("1010".to_string())
    );
    assert_eq!(
        eval_plus("(35).toString(36)"),
        Value::String("z".to_string())
    );
    assert_eq!(
        eval_plus("(0.5).toString(2)"),
        Value::String("0.1".to_string())
    );
    assert_eq!(
        eval_plus("(255).toString(undefined)"),
        Value::String("255".to_string())
    );

    for bad in [
        "(255).toString(1)",
        "(255).toString(37)",
        "(255).toString(NaN)",
    ] {
        assert!(
            matches!(eval_plus_err(bad), ExprError::RangeError(_)),
            "expected RangeError for `{bad}`"
        );
    }
}

#[test]
fn p0_is_cacheable_nested_loops_linear_compile_time() {
    // 50 nested `while (1) { a = {}; ... }` loops must compile in milliseconds
    // without exponential fixed-point re-traversal.
    let depth = 50;
    let mut src = String::from("let a = {}; ");
    for _ in 0..depth {
        src.push_str("while (1) { a = {}; ");
    }
    src.push_str("a = properties; ");
    for _ in 0..depth {
        src.push_str("} ");
    }
    src.push_str("a.x = 1;");

    let t0 = Instant::now();
    let compiled = compile(&src, ExprMode::ExpressionsPlus)
        .expect("50 nested while loops should compile cleanly");
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_millis(500),
        "compile() on {depth} nested loops took too long: {elapsed:?}"
    );
    assert!(
        !compiled.is_cacheable(),
        "loop-reassigned alias mutated after loop must remain non-cacheable"
    );

    // Pure loop mutating a fresh local array that is never reassigned remains cacheable.
    let pure = compile(
        "let out = []; for (let i = 0; i < 4; i++) { out.push(i); } out",
        ExprMode::ExpressionsPlus,
    )
    .unwrap();
    assert!(pure.is_cacheable());

    // Loop invalidation via update, test, and for-of/for-in target remains non-cacheable.
    for sample in [
        "let a = {}; for (let i = 0; i < 2; i++, a = properties) { a.x = 1; }",
        "let a = {}; while ((a = properties)) { a.x = 1; }",
        "let a = {}; do { a.x = 1; } while ((a = properties));",
        "let a = {}; for (a of [properties]) { a.x = 1; }",
        "let a = {}; for (a in properties) { a.x = 1; }",
    ] {
        let c = compile(sample, ExprMode::ExpressionsPlus).unwrap();
        assert!(!c.is_cacheable(), "expected `{sample}` to be non-cacheable");
    }
}

#[test]
fn p1_optional_chain_property_handle_with_nullish_value_allows_handle_members() {
    struct NullishPropContext {
        current_value: Value,
    }

    impl ExprContext for NullishPropContext {
        fn frame(&self) -> f64 {
            0.0
        }

        fn value(&self) -> Value {
            Value::Number(0.0)
        }

        fn this_layer_id(&self) -> Option<HandleId> {
            Some(HandleId::new("layer1"))
        }

        fn property(&self, _scope: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
            if name == "foo" {
                Ok(HandleId::new("prop_foo"))
            } else {
                Err(ExprError::NotFound(name.to_string()))
            }
        }

        fn handle_name(&self, id: &HandleId) -> Option<String> {
            (id.as_str() == "prop_foo").then(|| "foo".to_string())
        }

        fn property_value(&self, id: &HandleId) -> Result<Value, ExprError> {
            if id.as_str() == "prop_foo" {
                Ok(self.current_value.clone())
            } else {
                Err(ExprError::NotFound(id.as_str().to_string()))
            }
        }

        fn property_value_at_time(&self, id: &HandleId, seconds: f64) -> Result<Value, ExprError> {
            if id.as_str() == "prop_foo" {
                Ok(Value::Number(seconds * 10.0))
            } else {
                Err(ExprError::NotFound(id.as_str().to_string()))
            }
        }

        fn property_num_keys(&self, id: &HandleId) -> usize {
            if id.as_str() == "prop_foo" { 3 } else { 0 }
        }
    }

    let cfg = EvalConfig::expressions_plus();
    for nullish in [Value::Null, Value::Undefined] {
        let ctx = NullishPropContext {
            current_value: nullish,
        };
        assert_eq!(
            eval(r#"thisLayer?.property("foo").numKeys"#, &ctx, &cfg).unwrap(),
            Value::Number(3.0)
        );
        assert_eq!(
            eval(r#"thisLayer?.property("foo").valueAtTime(1.0)"#, &ctx, &cfg).unwrap(),
            Value::Number(10.0)
        );
        assert_eq!(
            eval(r#"thisLayer?.property("foo").name"#, &ctx, &cfg).unwrap(),
            Value::String("foo".to_string())
        );
        // Optional chaining (`?.`) on a PropertyHandle whose value is nullish still short-circuits.
        assert_eq!(
            eval(r#"thisLayer?.property("foo")?.numKeys"#, &ctx, &cfg).unwrap(),
            Value::Undefined
        );
    }

    // Standard nullish member checks remain intact.
    assert_eq!(eval_plus("null?.a.b"), Value::Undefined);
    for expr in [
        "(null?.a).b",
        "({a: null}).a.b",
        "({a: null})?.a.b",
        "({a: null})?.a.b?.()",
        "({a: undefined})?.a.b",
    ] {
        assert!(
            matches!(eval_plus_err(expr), ExprError::TypeError(_)),
            "expected TypeError for `{expr}`"
        );
    }
}
