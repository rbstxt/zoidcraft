use std::cell::RefCell;
use std::collections::HashSet;

use crate::{EvalConfig, ExprContext, ExprError, ExprMode, HandleId, Value, eval};

struct HookTrackingCtx {
    events: RefCell<Vec<(String, &'static str)>>,
    active: RefCell<HashSet<String>>,
    cyclic_pair: bool,
    cyclic_at_time: Option<(&'static str, &'static str)>,
}

impl HookTrackingCtx {
    pub(super) fn new(cyclic_pair: bool) -> Self {
        Self {
            events: RefCell::new(Vec::new()),
            active: RefCell::new(HashSet::new()),
            cyclic_pair,
            cyclic_at_time: None,
        }
    }

    fn with_cyclic_at_time(a_expr: &'static str, b_expr: &'static str) -> Self {
        Self {
            events: RefCell::new(Vec::new()),
            active: RefCell::new(HashSet::new()),
            cyclic_pair: false,
            cyclic_at_time: Some((a_expr, b_expr)),
        }
    }

    fn assert_balanced_and_empty(&self) {
        assert!(
            self.active.borrow().is_empty(),
            "active property stack must be empty after eval, got {:?}",
            self.active.borrow()
        );
        let events = self.events.borrow();
        let mut stack: Vec<&str> = Vec::new();
        for (id, kind) in events.iter() {
            match *kind {
                "enter" => stack.push(id.as_str()),
                "exit" => {
                    let top = stack.pop();
                    assert_eq!(
                        top,
                        Some(id.as_str()),
                        "unbalanced enter/exit events: {events:?}"
                    );
                }
                other => panic!("unexpected event kind {other}"),
            }
        }
        assert!(stack.is_empty(), "unclosed enter events: {events:?}");
    }
}

impl ExprContext for HookTrackingCtx {
    fn frame(&self) -> f64 {
        10.0
    }

    fn time(&self) -> f64 {
        0.5
    }

    fn fps(&self) -> f64 {
        30.0
    }

    fn value(&self) -> Value {
        Value::Number(1.0)
    }

    fn this_object_id(&self) -> Option<HandleId> {
        Some(HandleId::new("obj:1"))
    }

    fn property(&self, _owner: Option<&HandleId>, name: &str) -> Result<HandleId, ExprError> {
        Ok(HandleId::new(format!("prop:{name}")))
    }

    fn prop(&self, path: &str) -> Result<HandleId, ExprError> {
        Ok(HandleId::new(format!("prop:{path}")))
    }

    fn enter_property(&self, id: &HandleId) -> Result<(), ExprError> {
        let key = id.as_str().to_string();
        if !self.active.borrow_mut().insert(key.clone()) {
            return Err(ExprError::CycleDetected(key));
        }
        self.events.borrow_mut().push((key, "enter"));
        Ok(())
    }

    fn exit_property(&self, id: &HandleId) {
        let key = id.as_str().to_string();
        let removed = self.active.borrow_mut().remove(&key);
        assert!(
            removed,
            "exit_property called without matching enter for {key}"
        );
        self.events.borrow_mut().push((key, "exit"));
    }

    fn property_value(&self, id: &HandleId) -> Result<Value, ExprError> {
        match id.as_str() {
            "prop:a" => {
                if self.cyclic_pair {
                    let cfg = EvalConfig {
                        mode: ExprMode::ExpressionsPlus,
                        ..EvalConfig::default()
                    };
                    eval("property('b').value", self, &cfg)
                } else {
                    Ok(Value::Number(42.0))
                }
            }
            "prop:b" => {
                if self.cyclic_pair {
                    let cfg = EvalConfig {
                        mode: ExprMode::ExpressionsPlus,
                        ..EvalConfig::default()
                    };
                    eval("property('a').value", self, &cfg)
                } else {
                    Ok(Value::Number(7.0))
                }
            }
            "prop:arr" => Ok(Value::Array(vec![
                Value::Number(10.0),
                Value::Number(20.0),
                Value::Number(30.0),
            ])),
            "prop:boom" => Err(ExprError::NotFound("boom".to_string())),
            other => Err(ExprError::NotFound(other.to_string())),
        }
    }

    fn property_value_at_time(&self, id: &HandleId, seconds: f64) -> Result<Value, ExprError> {
        if let Some((a_expr, b_expr)) = self.cyclic_at_time {
            let cfg = EvalConfig {
                mode: ExprMode::ExpressionsPlus,
                ..EvalConfig::default()
            };
            return match id.as_str() {
                "prop:a" => eval(a_expr, self, &cfg),
                "prop:b" => eval(b_expr, self, &cfg),
                other => Err(ExprError::NotFound(other.to_string())),
            };
        }
        match id.as_str() {
            "prop:a" => Ok(Value::Number(seconds * 100.0)),
            "prop:boom" => Err(ExprError::NotFound("boom_at_time".to_string())),
            other => Err(ExprError::NotFound(other.to_string())),
        }
    }
}

#[test]
fn property_enter_and_exit_hooks_are_symmetric_across_all_read_paths() {
    let cfg = EvalConfig {
        mode: ExprMode::ExpressionsPlus,
        ..EvalConfig::default()
    };
    let ctx = HookTrackingCtx::new(false);

    assert_eq!(
        eval("property('a').value", &ctx, &cfg).unwrap(),
        Value::Number(42.0)
    );
    assert_eq!(
        eval("property('a').valueAtTime(1.5)", &ctx, &cfg).unwrap(),
        Value::Number(150.0)
    );
    assert_eq!(eval("prop('a')", &ctx, &cfg).unwrap(), Value::Number(42.0));
    assert_eq!(
        eval("thisObject.param('a')", &ctx, &cfg).unwrap(),
        Value::Number(42.0)
    );
    assert_eq!(
        eval("thisObject('a')", &ctx, &cfg).unwrap(),
        Value::Number(42.0)
    );
    assert_eq!(
        eval("property('arr').length", &ctx, &cfg).unwrap(),
        Value::Number(3.0)
    );
    assert_eq!(
        eval("property('a') + 1", &ctx, &cfg).unwrap(),
        Value::Number(43.0)
    );
    assert_eq!(
        eval("property('a')", &ctx, &cfg).unwrap(),
        Value::Number(42.0)
    );
    assert_eq!(
        eval("property('arr').join(',')", &ctx, &cfg).unwrap(),
        Value::String("10,20,30".to_string())
    );
    assert_eq!(
        eval("const p = property('a'); p.nonexistent?.()", &ctx, &cfg).unwrap(),
        Value::Undefined
    );

    ctx.assert_balanced_and_empty();
    assert_eq!(ctx.events.borrow().len(), 20);
}

#[test]
fn property_exit_hook_runs_on_read_error_and_cycle_propagates() {
    let cfg = EvalConfig {
        mode: ExprMode::ExpressionsPlus,
        ..EvalConfig::default()
    };

    // Error during property_value and property_value_at_time still calls exit_property.
    let err_ctx = HookTrackingCtx::new(false);
    assert!(matches!(
        eval("property('boom').value", &err_ctx, &cfg),
        Err(ExprError::NotFound(_))
    ));
    assert!(matches!(
        eval("property('boom').valueAtTime(2.0)", &err_ctx, &cfg),
        Err(ExprError::NotFound(_))
    ));
    err_ctx.assert_balanced_and_empty();
    assert_eq!(
        *err_ctx.events.borrow(),
        vec![
            ("prop:boom".to_string(), "enter"),
            ("prop:boom".to_string(), "exit"),
            ("prop:boom".to_string(), "enter"),
            ("prop:boom".to_string(), "exit"),
        ]
    );

    // Mutual cycle a -> b -> a returns CycleDetected and cleanly exits b then a.
    let cycle_ctx = HookTrackingCtx::new(true);
    let res = eval("property('a').value", &cycle_ctx, &cfg);
    assert!(
        matches!(res, Err(ExprError::CycleDetected(ref id)) if id == "prop:a"),
        "expected CycleDetected(prop:a), got {res:?}"
    );
    cycle_ctx.assert_balanced_and_empty();
    assert_eq!(
        *cycle_ctx.events.borrow(),
        vec![
            ("prop:a".to_string(), "enter"),
            ("prop:b".to_string(), "enter"),
            ("prop:b".to_string(), "exit"),
            ("prop:a".to_string(), "exit"),
        ]
    );
}

#[test]
fn property_velocity_and_speed_hooks_are_symmetric_on_success_and_error() {
    let cfg = EvalConfig {
        mode: ExprMode::ExpressionsPlus,
        ..EvalConfig::default()
    };

    // Normal return: all 4 derivative read paths enter and exit symmetrically.
    let ctx = HookTrackingCtx::new(false);
    for expr in [
        "property('a').velocity",
        "property('a').speed",
        "property('a').velocityAtTime(1.5)",
        "property('a').speedAtTime(1.5)",
    ] {
        let val = eval(expr, &ctx, &cfg).unwrap_or_else(|e| panic!("{expr} failed: {e:?}"));
        let n = val.as_f64().expect("derivative should evaluate to number");
        assert!(
            (n - 100.0).abs() < 1e-6,
            "{expr} returned {n}, expected ~100.0"
        );
    }
    ctx.assert_balanced_and_empty();
    assert_eq!(
        *ctx.events.borrow(),
        vec![
            ("prop:a".to_string(), "enter"),
            ("prop:a".to_string(), "exit"),
            ("prop:a".to_string(), "enter"),
            ("prop:a".to_string(), "exit"),
            ("prop:a".to_string(), "enter"),
            ("prop:a".to_string(), "exit"),
            ("prop:a".to_string(), "enter"),
            ("prop:a".to_string(), "exit"),
        ]
    );

    // Evaluation errors inside property_velocity / property_speed still run exit_property.
    let err_ctx = HookTrackingCtx::new(false);
    for expr in [
        "property('boom').velocity",
        "property('boom').speed",
        "property('boom').velocityAtTime(1.5)",
        "property('boom').speedAtTime(1.5)",
    ] {
        assert!(
            matches!(eval(expr, &err_ctx, &cfg), Err(ExprError::NotFound(_))),
            "expected NotFound for {expr}"
        );
    }
    err_ctx.assert_balanced_and_empty();
    assert_eq!(
        *err_ctx.events.borrow(),
        vec![
            ("prop:boom".to_string(), "enter"),
            ("prop:boom".to_string(), "exit"),
            ("prop:boom".to_string(), "enter"),
            ("prop:boom".to_string(), "exit"),
            ("prop:boom".to_string(), "enter"),
            ("prop:boom".to_string(), "exit"),
            ("prop:boom".to_string(), "enter"),
            ("prop:boom".to_string(), "exit"),
        ]
    );

    // When enter_property itself returns CycleDetected, exit_property must NOT be called.
    let pre_entered_ctx = HookTrackingCtx::new(false);
    pre_entered_ctx
        .active
        .borrow_mut()
        .insert("prop:a".to_string());
    for expr in [
        "property('a').velocity",
        "property('a').speed",
        "property('a').velocityAtTime(1.5)",
        "property('a').speedAtTime(1.5)",
    ] {
        let res = eval(expr, &pre_entered_ctx, &cfg);
        assert!(
            matches!(res, Err(ExprError::CycleDetected(ref id)) if id == "prop:a"),
            "expected CycleDetected(prop:a) for {expr}, got {res:?}"
        );
    }
    assert!(
        pre_entered_ctx.events.borrow().is_empty(),
        "exit_property must not run when enter_property fails"
    );
    assert!(pre_entered_ctx.active.borrow().contains("prop:a"));
}

#[test]
fn property_velocity_and_speed_cycles_return_cycle_detected_without_leaking_state() {
    let cfg = EvalConfig {
        mode: ExprMode::ExpressionsPlus,
        ..EvalConfig::default()
    };

    let cases = [
        (
            "property('a').velocity",
            "property('b').velocity",
            "property('a').velocity",
        ),
        (
            "property('a').speed",
            "property('b').speed",
            "property('a').speed",
        ),
        (
            "property('a').velocityAtTime(1.0)",
            "property('b').velocityAtTime(1.0)",
            "property('a').velocityAtTime(1.0)",
        ),
        (
            "property('a').speedAtTime(1.0)",
            "property('b').speedAtTime(1.0)",
            "property('a').speedAtTime(1.0)",
        ),
    ];

    for (root_expr, a_expr, b_expr) in cases {
        let cycle_ctx = HookTrackingCtx::with_cyclic_at_time(a_expr, b_expr);
        let res = eval(root_expr, &cycle_ctx, &cfg);
        assert!(
            matches!(res, Err(ExprError::CycleDetected(ref id)) if id == "prop:a"),
            "expected CycleDetected(prop:a) for {root_expr}, got {res:?}"
        );
        cycle_ctx.assert_balanced_and_empty();
        assert_eq!(
            *cycle_ctx.events.borrow(),
            vec![
                ("prop:a".to_string(), "enter"),
                ("prop:b".to_string(), "enter"),
                ("prop:b".to_string(), "exit"),
                ("prop:a".to_string(), "exit"),
            ],
            "unexpected enter/exit sequence for {root_expr}"
        );
    }
}
