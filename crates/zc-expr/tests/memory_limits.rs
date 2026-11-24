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

    // 13. Computed property lookup, numeric coercion, and non-pointer-equal string comparison
    // on 4 MiB strings must charge proportional work steps and hit StepLimitExceeded promptly.
    for (label, expr) in [
        (
            "computed key lookup",
            r#"var s = "x".repeat(10000).repeat(400); var o = {}; var n = 0; for (var i = 0; i < 1000000; i++) { if (o[s] === undefined) n++; } n"#,
        ),
        (
            "string numeric coercion",
            r#"var t = " ".repeat(10000).repeat(400); var n = 0; for (var i = 0; i < 1000000; i++) { n += +t; } n"#,
        ),
        (
            "distinct 4 MiB string equality",
            r#"var a = "x".repeat(10000).repeat(400); var b = "x".repeat(10000).repeat(400); var n = 0; for (var i = 0; i < 1000000; i++) { if (a === b) n++; } n"#,
        ),
    ] {
        let t0 = std::time::Instant::now();
        let err = eval(expr, &ctx, &cfg)
            .expect_err(&format!("{label} must trip StepLimitExceeded quickly"));
        assert!(
            matches!(err, ExprError::StepLimitExceeded { .. }),
            "expected StepLimitExceeded for {label}, got {err:?}"
        );
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(5),
            "{label} took too long: {:?}",
            t0.elapsed()
        );
    }

    // 14. Object.assign / Object.entries / object rest destructuring copying large computed keys
    // must charge key byte sizes.
    let err = eval(
        r#"var k = "k".repeat(10000).repeat(300); var src = {}; src[k] = 1; var xs = []; for (var i = 0; i < 40; i++) { var dst = {}; Object.assign(dst, src); xs.push(dst); } xs.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("Object.assign copying 3 MiB key 40 times must exceed 64 MiB heap budget");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for Object.assign key copy bomb, got {err:?}"
    );

    let err = eval(
        r#"var k = "k".repeat(10000).repeat(300); var src = {}; src[k] = 1; var xs = []; for (var i = 0; i < 40; i++) { var {...dst} = src; xs.push(dst); } xs.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("object rest destructuring copying 3 MiB key 40 times must exceed 64 MiB heap budget");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for object rest key copy bomb, got {err:?}"
    );

    // 15. Vector math allocation (-a) must charge heap budget and O(N) array reverse/includes
    // loops must charge proportional steps.
    let err = eval(
        r#"var a = Array(20000).fill(1); var xs = []; for (var i = 0; i < 120; i++) { xs.push(-a); } xs.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("repeated 20k vector negation must exceed 64 MiB heap budget");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for vector negation bomb, got {err:?}"
    );

    let arr_t0 = std::time::Instant::now();
    let err = eval(
        r#"var a = Array(20000).fill(1); for (var i = 0; i < 100000; i++) { a.reverse(); } a.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("100k reversals of 20k array must hit StepLimitExceeded promptly");
    assert!(
        matches!(err, ExprError::StepLimitExceeded { .. }),
        "expected StepLimitExceeded for array reverse loop, got {err:?}"
    );
    assert!(
        arr_t0.elapsed() < std::time::Duration::from_secs(2),
        "array reverse loop took too long: {:?}",
        arr_t0.elapsed()
    );

    // 16. Function call spread on 4 MiB strings must preflight MAX_ARRAY_LEN before expanding.
    let spread_t0 = std::time::Instant::now();
    let err = eval(
        r#"var s = "x".repeat(10000).repeat(400); function f(...a) { return a.length; } f(...s, ...s, ...s, ...s)"#,
        &ctx,
        &cfg,
    )
    .expect_err("spreading 4 MiB string into call arguments must fail immediately");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for call spread bomb, got {err:?}"
    );
    assert!(
        spread_t0.elapsed() < std::time::Duration::from_secs(2),
        "call spread rejection took too long: {:?}",
        spread_t0.elapsed()
    );

    // 17. Lazy string array destructuring and for..of must be O(1) for prefix/break and reject huge rest.
    let lazy_t0 = std::time::Instant::now();
    let res = eval(
        r#"var s = "x".repeat(10000).repeat(400); var hits = 0; for (var i = 0; i < 1000; i++) { var [c] = s; if (c === "x") hits++; for (var ch of s) { if (ch === "x") hits++; break; } } hits"#,
        &ctx,
        &cfg,
    )
    .expect("lazy prefix destructuring and break for..of on 4 MiB string should finish fast");
    assert_eq!(res, zc_expr::Value::Number(2000.0));
    assert!(
        lazy_t0.elapsed() < std::time::Duration::from_secs(2),
        "lazy string destructuring/for..of took too long: {:?}",
        lazy_t0.elapsed()
    );

    let err = eval(
        r#"var s = "x".repeat(10000).repeat(400); var [x, ...rest] = s; rest.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("string rest destructuring past MAX_ARRAY_LEN must fail immediately");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for string rest destructuring, got {err:?}"
    );

    // 18. for..in over an object with a 3 MiB key must charge retained snapshot key bytes.
    let err = eval(
        r#"var k = "k".repeat(10000).repeat(300); var o = {}; o[k] = 1; var xs = []; for (var i = 0; i < 40; i++) { for (var p in o) { xs.push(p); } } xs.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("for..in key snapshot bomb must exceed 64 MiB heap budget");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for for..in key bomb, got {err:?}"
    );

    // 19. Catchable JSON.stringify escape expansion and large export loops must charge work and
    // trip StepLimitExceeded quickly without allocating 24 MB escape buffers.
    for (label, expr) in [
        (
            "JSON.stringify control-char escape loop",
            r#"var s = String.fromCharCode(0).repeat(10000).repeat(400); for (var i = 0; i < 1000; i++) { try { JSON.stringify(s); } catch (e) {} } 0"#,
        ),
        (
            "JSON.stringify oversized array export loop",
            r#"var s = "x".repeat(200); var a = Array(50000).fill(s); for (var i = 0; i < 1000; i++) { try { JSON.stringify(a); } catch (e) {} } 0"#,
        ),
    ] {
        let t0 = std::time::Instant::now();
        let err = eval(expr, &ctx, &cfg)
            .expect_err(&format!("{label} must trip StepLimitExceeded quickly"));
        assert!(
            matches!(err, ExprError::StepLimitExceeded { .. }),
            "expected StepLimitExceeded for {label}, got {err:?}"
        );
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(5),
            "{label} took too long: {:?}",
            t0.elapsed()
        );
    }

    // 20. Cyclic object with 12 MiB of keys must reject cycle immediately before cloning 128 levels.
    let cyc_t0 = std::time::Instant::now();
    let err = eval(
        r#"var k = "z".repeat(10000).repeat(400); var o = {}; o.a = o; o[k + "1"] = 1; o[k + "2"] = 2; o[k + "3"] = 3; JSON.stringify(o)"#,
        &ctx,
        &cfg,
    )
    .expect_err("cyclic object export must fail immediately");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for cyclic object export, got {err:?}"
    );
    assert!(
        cyc_t0.elapsed() < std::time::Duration::from_secs(2),
        "cyclic object rejection took too long: {:?}",
        cyc_t0.elapsed()
    );

    // 21. Recursive array call spreads must charge argument buffer allocations against the 64 MiB budget.
    let err = eval(
        r#"var a = Array(65535); function sink() { return arguments_ignored(); } function descend(n) { if (n <= 0) return 0; return sink(...a, descend(n - 1)); } descend(64)"#,
        &ctx,
        &cfg,
    )
    .expect_err("recursive 65,535-element array spread must exceed 64 MiB heap budget");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for recursive array call spread bomb, got {err:?}"
    );

    // 22. Lexical `for (let <long_ident> = ...)` cloning must charge identifier byte lengths.
    let long_ident = "a".repeat(64 * 1024);
    let long_ident_expr = format!("var i = 0; for (let {long_ident} = 0; i < 2000; i++) {{}} i");
    let err = eval(&long_ident_expr, &ctx, &cfg).expect_err(
        "cloning 64 KiB lexical loop identifier 2000 times must exceed 64 MiB heap budget",
    );
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for lexical loop identifier bomb, got {err:?}"
    );

    // 23. Ordinary block `let <64 KiB ident> = 0` inside a loop must charge identifier bytes.
    let block_ident_expr = format!("for (var i = 0; i < 4000; i++) {{ let {long_ident} = 0; }} i");
    let err = eval(&block_ident_expr, &ctx, &cfg).expect_err(
        "binding 64 KiB identifier in 4000 block scopes must exceed 64 MiB heap budget",
    );
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for block lexical identifier bomb, got {err:?}"
    );

    // 24. Creating thousands of closures referencing a 256 KiB literal must share the AST body
    // (zero deep-clone OOM) and finish quickly within budget.
    let payload = "p".repeat(256 * 1024);
    let closure_expr = format!(
        r#"var fns = []; for (var i = 0; i < 4096; i++) {{ fns.push(function () {{ return "{payload}"; }}); }} fns.length"#
    );
    let closure_t0 = std::time::Instant::now();
    let res = eval(&closure_expr, &ctx, &cfg)
        .expect("4096 closures sharing one AST body should succeed without cloning 1 GiB");
    assert_eq!(res, zc_expr::Value::Number(4096.0));
    assert!(
        closure_t0.elapsed() < std::time::Duration::from_secs(5),
        "closure creation loop was unexpectedly slow: {:?}",
        closure_t0.elapsed()
    );

    // 25. Filling a 65,536-element array with Math[large_key] must share Rc<str> (no 4 GiB OOM)
    // and repeated Math[large_key] lookups must charge the heap budget.
    let fill_t0 = std::time::Instant::now();
    let res = eval(
        r#"var k = "x".repeat(256).repeat(256); Array(65536).fill(Math[k]).length"#,
        &ctx,
        &cfg,
    )
    .expect("Array.fill with shared Rc<str> BuiltinFn should succeed without 4 GiB duplication");
    assert_eq!(res, zc_expr::Value::Number(65536.0));
    assert!(
        fill_t0.elapsed() < std::time::Duration::from_secs(5),
        "Array.fill(Math[k]) took too long: {:?}",
        fill_t0.elapsed()
    );

    let err = eval(
        r#"var k = "x".repeat(10000).repeat(300); var xs = []; for (var i = 0; i < 40; i++) { xs.push(Math[k]); } xs.length"#,
        &ctx,
        &cfg,
    )
    .expect_err("repeated 3 MiB BuiltinFn lookups must exceed 64 MiB heap budget");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for repeated BuiltinFn lookup bomb, got {err:?}"
    );

    // 26. Recursive object literal with large static key must not retain uncharged keys across calls.
    let big_key = "k".repeat(4 * 1024 * 1024);
    let rec_obj_expr =
        format!(r#"function f(n) {{ return n ? {{"{big_key}": f(n - 1)}} : 0; }} f(32)"#);
    let err = eval(&rec_obj_expr, &ctx, &cfg)
        .expect_err("recursive 4 MiB static-key object literal must exceed 64 MiB heap budget");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for recursive static-key object literal, got {err:?}"
    );

    // 27. Recursive array literal spread `[...a, descend(n - 1)]` must charge incrementally before recursion.
    let err = eval(
        r#"var a = Array(65535); function descend(n) { return n ? [...a, descend(n - 1)] : 0; } descend(64)"#,
        &ctx,
        &cfg,
    )
    .expect_err("recursive array literal spread must exceed 64 MiB heap budget");
    assert!(
        matches!(err, ExprError::RangeError(_)),
        "expected RangeError for recursive array literal spread, got {err:?}"
    );
}
