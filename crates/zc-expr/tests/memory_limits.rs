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

    // 4. Repeated large array allocations exhausting cumulative heap budget.
    let err = eval(
        r#"var xs = []; for (var i = 0; i < 60; i++) { xs.push(Array(20000)); } xs.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("array arena bomb must fail");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for array arena bomb, got {err:?}"
    );
}
