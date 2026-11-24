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

    // 9. Sparse indexed array growth must enforce MAX_ARRAY_LEN and heap budget.
    let err = eval(r#"var a = []; a[100000] = 1; a.length"#, &ctx, &cfg)
        .expect_err("indexed array growth past MAX_ARRAY_LEN must fail");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for sparse array assignment, got {err:?}"
    );
    let err = eval(
        r#"var xs = []; for (var i = 0; i < 120; i++) { var a = []; a[20000] = 1; xs.push(a); } xs.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("repeated indexed array growth must exhaust 64 MiB heap budget");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for repeated indexed growth, got {err:?}"
    );

    // 10. Repeated JSON.parse interning must charge heap budget.
    let err = eval(
        r#"var j = JSON.stringify("x".repeat(10000).repeat(200)); var xs = []; for (var i = 0; i < 50; i++) { xs.push(JSON.parse(j)); } xs.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("repeated 2 MiB JSON.parse allocations must exceed 64 MiB heap budget");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for JSON.parse bomb, got {err:?}"
    );

    // 11. Computed object key flood must charge heap budget and cap key count.
    let err = eval(
        r#"var o = {}; for (var i = 0; i < 70000; i++) { o["k" + i] = 1; } 1"#,
        &ctx,
        &cfg,
    )
    .expect_err("object key flood past MAX_ARRAY_LEN must fail");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for object key flood, got {err:?}"
    );

    // 12. O(N) string operations on a 4 MiB string must NOT hang (ASCII charCodeAt is O(1);
    // linear scans like includes/non-ASCII charge proportional steps and trip StepLimitExceeded quickly).
    let start = std::time::Instant::now();
    let res = eval(
        r#"var s = "x".repeat(10000).repeat(400); var n = 0; for (var i = 0; i < 50000; i++) { n += s.charCodeAt(i & 1023); } n"#,
        &ctx,
        &cfg,
    )
    .expect("O(1) ASCII charCodeAt loop should complete quickly");
    assert_eq!(res, zc_expr::Value::Number(50_000.0 * 120.0));
    assert!(
        start.elapsed() < std::time::Duration::from_secs(2),
        "ASCII charCodeAt loop was unexpectedly slow: {:?}",
        start.elapsed()
    );

    let scan_start = std::time::Instant::now();
    let err = eval(
        r#"var s = "x".repeat(10000).repeat(400); var n = 0; for (var i = 0; i < 1000000; i++) { if (s.includes("z")) n++; } n"#,
        &ctx,
        &cfg,
    )
    .expect_err("1M scans of 4 MiB string must hit StepLimitExceeded immediately");
    assert!(
        matches!(err, ExprError::StepLimitExceeded { .. }),
        "expected StepLimitExceeded for O(N) scan loop, got {err:?}"
    );
    assert!(
        scan_start.elapsed() < std::time::Duration::from_secs(2),
        "O(N) scan loop took too long before tripping step limit: {:?}",
        scan_start.elapsed()
    );
}
