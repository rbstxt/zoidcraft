#![forbid(unsafe_code)]

use zc_expr::{BasicContext, EvalConfig, ExprError, eval};

#[test]
fn rejects_string_repeat_concat_array_and_dag_export_memory_bombs() {
    let ctx = BasicContext::default();
    let cfg = EvalConfig::default();

    // 1. Nested repeat bomb (would request ~1 TB without byte limits).
    let err = eval(
        r#""x".repeat(10000).repeat(10000).repeat(10000)"#,
        &ctx,
        &cfg,
    )
    .expect_err("nested repeat bomb must fail");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for repeat bomb, got {err:?}"
    );

    // 2. Exponential string doubling loop.
    let err = eval(
        r#"var s = "x"; for (var i = 0; i < 60; i++) { s = s + s; } s"#,
        &ctx,
        &cfg,
    )
    .expect_err("string doubling bomb must fail");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for concat bomb, got {err:?}"
    );

    // 3. Shallow shared DAG array expansion bomb (depth 20 -> 2^20 = 1M expanded nodes).
    let err = eval(
        r#"var a = [1, 2]; for (var i = 0; i < 20; i++) { a = [a, a]; } a"#,
        &ctx,
        &cfg,
    )
    .expect_err("exponential shared DAG export must fail");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for DAG export bomb, got {err:?}"
    );

    // 4. Repeated large array allocations exhausting cumulative 64 MiB heap budget (120 * 20k * 32B = 76.8 MiB).
    let err = eval(
        r#"var xs = []; for (var i = 0; i < 120; i++) { xs.push(Array(20000)); } xs.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("array arena bomb must fail");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for array arena bomb, got {err:?}"
    );

    // 5. Pushing the same 4 MiB string 65,536 times must share Rc<str> storage (no 256 GB OOM)
    // and exceeding MAX_ARRAY_LEN must fail cleanly with RangeError.
    let ok = eval(
        r#"var s = "x".repeat(10000).repeat(400); var xs = []; for (var i = 0; i < 2000; i++) { xs.push(s); } xs.length"#,
        &ctx,
        &cfg,
    )
    .expect("sharing immutable Rc<str> across array pushes should succeed within budget");
    assert_eq!(ok, zc_expr::Value::Number(2000.0));

    let err = eval(
        r#"var s = "x".repeat(10000).repeat(400); var xs = []; for (var i = 0; i < 70000; i++) { xs.push(s); } xs.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("pushing past MAX_ARRAY_LEN must fail");
    assert!(matches!(err, ExprError::RangeError(_)));

    // 6. Repeated fresh string transforms (toUpperCase / trim / slice) must charge heap budget.
    let err = eval(
        r#"var s = "a".repeat(10000).repeat(300); var xs = []; for (var i = 0; i < 40; i++) { xs.push(s.toUpperCase()); } xs.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("repeated 3 MiB toUpperCase allocations must exceed 64 MiB heap budget");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for repeated toUpperCase, got {err:?}"
    );

    // 7. Array.from on a large string must enforce MAX_ARRAY_LEN.
    let err = eval(
        r#"Array.from("x".repeat(10000).repeat(10)).length"#,
        &ctx,
        &cfg,
    )
    .expect_err("Array.from on 100k-char string must exceed MAX_ARRAY_LEN");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for Array.from, got {err:?}"
    );

    // 8. Repeated Object.entries on large arrays must charge heap budget (100 * 20k * 96B > 64 MiB).
    let err = eval(
        r#"var a = Array(20000); for (var i = 0; i < 100; i++) { Object.entries(a); } a.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("repeated Object.entries allocations must exceed heap budget");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for Object.entries bomb, got {err:?}"
    );
}
