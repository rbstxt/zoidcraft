//! Deterministic tree-walking evaluator with arena-allocated environments,
//! strict step and recursion budgets, vector arithmetic, and host scene handles.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use crate::ast::{
    ArrayElem, ArrayPatternElem, AssignOp, AssignTarget, BinaryOp, CallArg, ChainOp, Expr, ForInit,
    FunctionBody, Literal, LogicalOp, ObjectKey, ObjectPatternProp, ObjectProp, Param, Pattern,
    Program, Stmt, TemplatePart, UnaryOp, UpdateOp, VarKind, is_forbidden_ident,
};
use crate::builtins::{
    DetRng, ae_ease, ae_ease_in, ae_ease_out, ae_linear, catmull_rom_value, clamp_f64, cm3_ease,
    cm3_gauss_random, cm3_random, cm3_shake, cm3_wave, eval_math_fn, eval_wiggle, format_js_number,
    js_parse_float, js_parse_int, json_parse, json_stringify, lerp_value, look_at_xyz_euler,
    math_constant, to_i32, to_u32, vec_add, vec_cross, vec_div_componentwise, vec_div_scalar,
    vec_dot, vec_length, vec_mul_componentwise, vec_mul_scalar, vec_normalize, vec_sub,
};
use crate::{
    EvalConfig, ExprContext, ExprError, ExprMode, HandleId, HandleRef, KeyframeRecord, NameOrIndex,
    Value,
};

const MAX_EXPORT_DEPTH: usize = 128;

/// Evaluates a parsed [`Program`] against `ctx` using `config`.
pub fn evaluate_program(
    program: &Program,
    ctx: &dyn ExprContext,
    config: &EvalConfig,
) -> Result<Value, ExprError> {
    let mut evaluator = Evaluator::new(ctx, *config);
    let root_env = evaluator.push_env(None);
    evaluator.hoist_decls(&program.body, root_env)?;

    let mut completion = RuntimeValue::Undefined;
    for stmt in &program.body {
        match evaluator.exec_stmt(stmt, root_env)? {
            Flow::Normal(v) => {
                if !matches!(v, RuntimeValue::Empty) {
                    completion = v;
                }
            }
            Flow::Return(v) => {
                completion = v;
                break;
            }
            Flow::Throw(v) => {
                let msg = evaluator.runtime_to_js_string(&v)?;
                return Err(ExprError::Thrown(msg));
            }
            Flow::Break => {
                return Err(ExprError::SyntaxError {
                    offset: 0,
                    message: "illegal 'break' statement".to_string(),
                });
            }
            Flow::Continue => {
                return Err(ExprError::SyntaxError {
                    offset: 0,
                    message: "illegal 'continue' statement".to_string(),
                });
            }
        }
    }

    evaluator.export_value(&completion, 0)
}

#[derive(Debug, Clone)]
enum RuntimeValue {
    /// Sentinel used by statements that do not update the completion value.
    Empty,
    Undefined,
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(usize),
    Object(usize),
    Closure(usize),
    BuiltinFn(String),
    Namespace(NamespaceKind),
    Handle(HandleRef),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NamespaceKind {
    Math,
    Array,
    Number,
    String,
    Boolean,
    Object,
    Json,
}

enum Flow {
    Normal(RuntimeValue),
    Return(RuntimeValue),
    Break,
    Continue,
    Throw(RuntimeValue),
}

struct Binding {
    value: RuntimeValue,
    mutable: bool,
    is_var: bool,
}

struct EnvFrame {
    parent: Option<usize>,
    is_function_scope: bool,
    bindings: BTreeMap<String, Binding>,
}

struct Closure {
    params: Vec<Param>,
    body: FunctionBody,
    captured_env: usize,
    name: Option<String>,
}

struct Evaluator<'a> {
    ctx: &'a dyn ExprContext,
    config: EvalConfig,
    steps: u64,
    recursion_depth: usize,
    envs: Vec<EnvFrame>,
    arrays: Vec<Vec<RuntimeValue>>,
    objects: Vec<BTreeMap<String, RuntimeValue>>,
    closures: Vec<Closure>,
    rng: Option<DetRng>,
    stringifying_arrays: Vec<usize>,
}

impl<'a> Evaluator<'a> {
    fn new(ctx: &'a dyn ExprContext, config: EvalConfig) -> Self {
        Self {
            ctx,
            config,
            steps: 0,
            recursion_depth: 0,
            envs: Vec::new(),
            arrays: Vec::new(),
            objects: Vec::new(),
            closures: Vec::new(),
            rng: None,
            stringifying_arrays: Vec::new(),
        }
    }

    fn tick(&mut self) -> Result<(), ExprError> {
        self.steps += 1;
        if self.steps > self.config.max_steps {
            return Err(ExprError::StepLimitExceeded {
                limit: self.config.max_steps,
            });
        }
        Ok(())
    }

    fn push_env(&mut self, parent: Option<usize>) -> usize {
        let id = self.envs.len();
        self.envs.push(EnvFrame {
            parent,
            is_function_scope: parent.is_none(),
            bindings: BTreeMap::new(),
        });
        id
    }

    fn push_fn_env(&mut self, parent: usize) -> usize {
        let id = self.envs.len();
        self.envs.push(EnvFrame {
            parent: Some(parent),
            is_function_scope: true,
            bindings: BTreeMap::new(),
        });
        id
    }

    fn function_scope_of(&self, mut env: usize) -> usize {
        loop {
            if self.envs[env].is_function_scope {
                return env;
            }
            match self.envs[env].parent {
                Some(p) => env = p,
                None => return env,
            }
        }
    }

    fn define_binding(&mut self, env: usize, name: String, value: RuntimeValue, kind: VarKind) {
        let target_env = if kind == VarKind::Var {
            self.function_scope_of(env)
        } else {
            env
        };
        self.envs[target_env].bindings.insert(
            name,
            Binding {
                value,
                mutable: kind != VarKind::Const,
                is_var: kind == VarKind::Var,
            },
        );
    }

    fn lookup_local(&self, mut env: usize, name: &str) -> Option<RuntimeValue> {
        loop {
            if let Some(b) = self.envs[env].bindings.get(name) {
                return Some(b.value.clone());
            }
            env = self.envs[env].parent?;
        }
    }

    fn assign_local(
        &mut self,
        mut env: usize,
        name: &str,
        value: RuntimeValue,
    ) -> Result<(), ExprError> {
        loop {
            if let Some(binding) = self.envs[env].bindings.get_mut(name) {
                if !binding.mutable {
                    return Err(ExprError::TypeError(format!(
                        "assignment to constant variable '{name}'"
                    )));
                }
                binding.value = value;
                return Ok(());
            }
            match self.envs[env].parent {
                Some(p) => env = p,
                None => return Err(ExprError::ReferenceError(name.to_string())),
            }
        }
    }

    fn intern_value(&mut self, val: Value) -> RuntimeValue {
        match val {
            Value::Undefined => RuntimeValue::Undefined,
            Value::Null => RuntimeValue::Null,
            Value::Bool(b) => RuntimeValue::Bool(b),
            Value::Number(n) => RuntimeValue::Number(n),
            Value::String(s) => RuntimeValue::String(s),
            Value::Array(items) => {
                let converted: Vec<RuntimeValue> =
                    items.into_iter().map(|v| self.intern_value(v)).collect();
                let id = self.arrays.len();
                self.arrays.push(converted);
                RuntimeValue::Array(id)
            }
            Value::Object(map) => {
                let converted: BTreeMap<String, RuntimeValue> = map
                    .into_iter()
                    .map(|(k, v)| (k, self.intern_value(v)))
                    .collect();
                let id = self.objects.len();
                self.objects.push(converted);
                RuntimeValue::Object(id)
            }
        }
    }

    fn export_value(&mut self, rv: &RuntimeValue, depth: usize) -> Result<Value, ExprError> {
        if depth > MAX_EXPORT_DEPTH {
            return Err(ExprError::RangeError(
                "circular or overly deep value cannot be returned".to_string(),
            ));
        }
        match rv {
            RuntimeValue::Empty | RuntimeValue::Undefined => Ok(Value::Undefined),
            RuntimeValue::Null => Ok(Value::Null),
            RuntimeValue::Bool(b) => Ok(Value::Bool(*b)),
            RuntimeValue::Number(n) => Ok(Value::Number(*n)),
            RuntimeValue::String(s) => Ok(Value::String(s.clone())),
            RuntimeValue::Array(id) => {
                let items = self.arrays[*id].clone();
                let mut out = Vec::with_capacity(items.len());
                for item in &items {
                    out.push(self.export_value(item, depth + 1)?);
                }
                Ok(Value::Array(out))
            }
            RuntimeValue::Object(id) => {
                let entries = self.objects[*id].clone();
                let mut out = BTreeMap::new();
                for (k, v) in &entries {
                    out.insert(k.clone(), self.export_value(v, depth + 1)?);
                }
                Ok(Value::Object(out))
            }
            RuntimeValue::Handle(HandleRef::Property(prop_id)) => self.ctx.property_value(prop_id),
            RuntimeValue::Handle(_)
            | RuntimeValue::Closure(_)
            | RuntimeValue::BuiltinFn(_)
            | RuntimeValue::Namespace(_) => Ok(Value::Undefined),
        }
    }

    /// Unwraps a `Property` handle to its current evaluated value (`valueOf` semantics).
    fn deref_property_handle(&mut self, rv: RuntimeValue) -> Result<RuntimeValue, ExprError> {
        if let RuntimeValue::Handle(HandleRef::Property(prop_id)) = rv {
            let val = self.ctx.property_value(&prop_id)?;
            Ok(self.intern_value(val))
        } else {
            Ok(rv)
        }
    }

    fn runtime_as_f64_vec(&mut self, rv: &RuntimeValue) -> Result<Option<Vec<f64>>, ExprError> {
        let rv = self.deref_property_handle(rv.clone())?;
        let RuntimeValue::Array(id) = rv else {
            return Ok(None);
        };
        let items = &self.arrays[id];
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            match item {
                RuntimeValue::Number(n) => out.push(*n),
                _ => return Ok(None),
            }
        }
        Ok(Some(out))
    }

    fn runtime_to_bool(&mut self, rv: &RuntimeValue) -> Result<bool, ExprError> {
        let rv = self.deref_property_handle(rv.clone())?;
        Ok(match rv {
            RuntimeValue::Empty | RuntimeValue::Undefined | RuntimeValue::Null => false,
            RuntimeValue::Bool(b) => b,
            RuntimeValue::Number(n) => n != 0.0 && !n.is_nan(),
            RuntimeValue::String(s) => !s.is_empty(),
            RuntimeValue::Array(_)
            | RuntimeValue::Object(_)
            | RuntimeValue::Closure(_)
            | RuntimeValue::BuiltinFn(_)
            | RuntimeValue::Namespace(_)
            | RuntimeValue::Handle(_) => true,
        })
    }

    fn runtime_to_number(&mut self, rv: &RuntimeValue) -> Result<f64, ExprError> {
        let rv = self.deref_property_handle(rv.clone())?;
        Ok(match rv {
            RuntimeValue::Empty | RuntimeValue::Undefined => f64::NAN,
            RuntimeValue::Null => 0.0,
            RuntimeValue::Bool(b) => {
                if b {
                    1.0
                } else {
                    0.0
                }
            }
            RuntimeValue::Number(n) => n,
            RuntimeValue::String(s) => {
                let t = s.trim();
                if t.is_empty() {
                    0.0
                } else {
                    t.parse::<f64>().unwrap_or(f64::NAN)
                }
            }
            _ => f64::NAN,
        })
    }

    fn runtime_to_js_string(&mut self, rv: &RuntimeValue) -> Result<String, ExprError> {
        self.tick()?;
        let rv = self.deref_property_handle(rv.clone())?;
        Ok(match rv {
            RuntimeValue::Empty | RuntimeValue::Undefined => "undefined".to_string(),
            RuntimeValue::Null => "null".to_string(),
            RuntimeValue::Bool(b) => b.to_string(),
            RuntimeValue::Number(n) => format_js_number(n),
            RuntimeValue::String(s) => s,
            RuntimeValue::Array(id) => self.join_array_to_string(id, ",")?,
            RuntimeValue::Object(_) | RuntimeValue::Namespace(_) | RuntimeValue::Handle(_) => {
                "[object Object]".to_string()
            }
            RuntimeValue::Closure(_) | RuntimeValue::BuiltinFn(_) => "[Function]".to_string(),
        })
    }

    fn join_array_to_string(&mut self, id: usize, sep: &str) -> Result<String, ExprError> {
        // Per ECMAScript Array.prototype.join, cyclic array references yield "".
        if self.stringifying_arrays.contains(&id) {
            return Ok(String::new());
        }
        if self.stringifying_arrays.len() >= self.config.max_recursion {
            return Err(ExprError::RecursionLimitExceeded {
                limit: self.config.max_recursion,
            });
        }
        self.stringifying_arrays.push(id);
        let items = self.arrays[id].clone();
        let mut parts = Vec::with_capacity(items.len());
        let res = (|| -> Result<String, ExprError> {
            for item in &items {
                self.tick()?;
                match item {
                    RuntimeValue::Null | RuntimeValue::Undefined => parts.push(String::new()),
                    other => parts.push(self.runtime_to_js_string(other)?),
                }
            }
            Ok(parts.join(sep))
        })();
        self.stringifying_arrays.pop();
        res
    }

    fn runtime_to_vec3(&mut self, rv: &RuntimeValue) -> Result<[f64; 3], ExprError> {
        let v = self.runtime_as_f64_vec(rv)?.ok_or_else(|| {
            ExprError::TypeError("expected 3D numeric vector [x, y, z]".to_string())
        })?;
        Ok([
            v.first().copied().unwrap_or(0.0),
            v.get(1).copied().unwrap_or(0.0),
            v.get(2).copied().unwrap_or(0.0),
        ])
    }

    fn alloc_array(&mut self, elems: Vec<RuntimeValue>) -> RuntimeValue {
        let id = self.arrays.len();
        self.arrays.push(elems);
        RuntimeValue::Array(id)
    }

    fn alloc_f64_vec(&mut self, vals: Vec<f64>) -> RuntimeValue {
        let elems = vals.into_iter().map(RuntimeValue::Number).collect();
        self.alloc_array(elems)
    }

    fn alloc_object(&mut self, map: BTreeMap<String, RuntimeValue>) -> RuntimeValue {
        let id = self.objects.len();
        self.objects.push(map);
        RuntimeValue::Object(id)
    }

    fn keyframe_to_object(&mut self, kf: KeyframeRecord) -> RuntimeValue {
        let mut map = BTreeMap::new();
        map.insert("index".to_string(), RuntimeValue::Number(kf.index as f64));
        map.insert("time".to_string(), RuntimeValue::Number(kf.time));
        let val = self.intern_value(kf.value);
        map.insert("value".to_string(), val);
        self.alloc_object(map)
    }

    // -----------------------------------------------------------------------
    // Hoisting & Pattern Binding
    // -----------------------------------------------------------------------

    fn hoist_decls(&mut self, stmts: &[Stmt], env: usize) -> Result<(), ExprError> {
        let fn_scope = self.function_scope_of(env);
        for stmt in stmts {
            match stmt {
                Stmt::VarDecl {
                    kind: VarKind::Var,
                    decls,
                } => {
                    for d in decls {
                        self.hoist_pattern_var(&d.pattern, fn_scope);
                    }
                }
                Stmt::FunctionDecl(f) => {
                    let cid = self.closures.len();
                    self.closures.push(Closure {
                        params: f.params.clone(),
                        body: FunctionBody::Block(f.body.clone()),
                        captured_env: fn_scope,
                        name: Some(f.name.clone()),
                    });
                    self.define_binding(
                        fn_scope,
                        f.name.clone(),
                        RuntimeValue::Closure(cid),
                        VarKind::Var,
                    );
                }
                Stmt::Block(inner) => self.hoist_decls(inner, fn_scope)?,
                Stmt::If {
                    consequent,
                    alternate,
                    ..
                } => {
                    self.hoist_decls(std::slice::from_ref(consequent.as_ref()), fn_scope)?;
                    if let Some(alt) = alternate {
                        self.hoist_decls(std::slice::from_ref(alt.as_ref()), fn_scope)?;
                    }
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                    self.hoist_decls(std::slice::from_ref(body.as_ref()), fn_scope)?;
                }
                Stmt::For(f) => {
                    if let Some(ForInit::VarDecl {
                        kind: VarKind::Var,
                        decls,
                    }) = &f.init
                    {
                        for d in decls {
                            self.hoist_pattern_var(&d.pattern, fn_scope);
                        }
                    }
                    self.hoist_decls(std::slice::from_ref(&f.body), fn_scope)?;
                }
                Stmt::ForOf(f) => {
                    if f.kind == Some(VarKind::Var) {
                        self.hoist_pattern_var(&f.target, fn_scope);
                    }
                    self.hoist_decls(std::slice::from_ref(&f.body), fn_scope)?;
                }
                Stmt::ForIn(f) => {
                    if f.kind == Some(VarKind::Var) {
                        self.hoist_pattern_var(&f.target, fn_scope);
                    }
                    self.hoist_decls(std::slice::from_ref(&f.body), fn_scope)?;
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        self.hoist_decls(&c.consequent, fn_scope)?;
                    }
                }
                Stmt::Try(t) => {
                    self.hoist_decls(&t.block, fn_scope)?;
                    if let Some(cb) = &t.catch_body {
                        self.hoist_decls(cb, fn_scope)?;
                    }
                    if let Some(fb) = &t.finally_body {
                        self.hoist_decls(fb, fn_scope)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn hoist_pattern_var(&mut self, pattern: &Pattern, fn_scope: usize) {
        match pattern {
            Pattern::Ident(name) => {
                self.envs[fn_scope]
                    .bindings
                    .entry(name.clone())
                    .or_insert(Binding {
                        value: RuntimeValue::Undefined,
                        mutable: true,
                        is_var: true,
                    });
            }
            Pattern::Array(elems) => {
                for e in elems {
                    match e {
                        ArrayPatternElem::Hole => {}
                        ArrayPatternElem::Binding { pattern, .. } => {
                            self.hoist_pattern_var(pattern, fn_scope);
                        }
                        ArrayPatternElem::Rest(name) => {
                            self.hoist_pattern_var(&Pattern::Ident(name.clone()), fn_scope);
                        }
                    }
                }
            }
            Pattern::Object(props) => {
                for p in props {
                    match p {
                        ObjectPatternProp::Prop { pattern, .. } => {
                            self.hoist_pattern_var(pattern, fn_scope);
                        }
                        ObjectPatternProp::Rest(name) => {
                            self.hoist_pattern_var(&Pattern::Ident(name.clone()), fn_scope);
                        }
                    }
                }
            }
        }
    }

    fn bind_pattern(
        &mut self,
        pattern: &Pattern,
        val: RuntimeValue,
        env: usize,
        kind: VarKind,
    ) -> Result<(), ExprError> {
        let val = self.deref_property_handle(val)?;
        match pattern {
            Pattern::Ident(name) => {
                self.define_binding(env, name.clone(), val, kind);
                Ok(())
            }
            Pattern::Array(elems) => {
                let items = match val {
                    RuntimeValue::Array(id) => self.arrays[id].clone(),
                    RuntimeValue::String(s) => s
                        .chars()
                        .map(|c| RuntimeValue::String(c.to_string()))
                        .collect(),
                    _ => {
                        return Err(ExprError::TypeError(
                            "cannot destructure non-iterable value as array".to_string(),
                        ));
                    }
                };
                for (i, elem) in elems.iter().enumerate() {
                    match elem {
                        ArrayPatternElem::Hole => {}
                        ArrayPatternElem::Binding { pattern, default } => {
                            let mut item = items.get(i).cloned().unwrap_or(RuntimeValue::Undefined);
                            if matches!(item, RuntimeValue::Undefined)
                                && let Some(def_expr) = default
                            {
                                item = self.eval_expr(def_expr, env)?;
                            }
                            self.bind_pattern(pattern, item, env, kind)?;
                        }
                        ArrayPatternElem::Rest(name) => {
                            let rest_items = if i < items.len() {
                                items[i..].to_vec()
                            } else {
                                Vec::new()
                            };
                            let arr = self.alloc_array(rest_items);
                            self.define_binding(env, name.clone(), arr, kind);
                        }
                    }
                }
                Ok(())
            }
            Pattern::Object(props) => {
                if matches!(val, RuntimeValue::Null | RuntimeValue::Undefined) {
                    return Err(ExprError::TypeError(
                        "cannot destructure null or undefined".to_string(),
                    ));
                }
                let mut used_keys = Vec::new();
                for prop in props {
                    match prop {
                        ObjectPatternProp::Prop {
                            key,
                            pattern,
                            default,
                        } => {
                            used_keys.push(key.clone());
                            let mut pv = self.get_property(&val, key)?;
                            if matches!(pv, RuntimeValue::Undefined)
                                && let Some(def_expr) = default
                            {
                                pv = self.eval_expr(def_expr, env)?;
                            }
                            self.bind_pattern(pattern, pv, env, kind)?;
                        }
                        ObjectPatternProp::Rest(name) => {
                            let mut rest = BTreeMap::new();
                            if let RuntimeValue::Object(id) = val {
                                let entries = self.objects[id].clone();
                                for (k, v) in entries {
                                    if !used_keys.contains(&k) {
                                        rest.insert(k, v);
                                    }
                                }
                            }
                            let obj = self.alloc_object(rest);
                            self.define_binding(env, name.clone(), obj, kind);
                        }
                    }
                }
                Ok(())
            }
        }
    }

    // -----------------------------------------------------------------------
    // Statement Execution
    // -----------------------------------------------------------------------

    fn exec_stmt(&mut self, stmt: &Stmt, env: usize) -> Result<Flow, ExprError> {
        self.tick()?;
        match stmt {
            Stmt::Empty => Ok(Flow::Normal(RuntimeValue::Empty)),
            Stmt::Expr(expr) => Ok(Flow::Normal(self.eval_expr(expr, env)?)),
            Stmt::Block(stmts) => {
                let block_env = self.push_env(Some(env));
                self.exec_stmt_list(stmts, block_env)
            }
            Stmt::VarDecl { kind, decls } => {
                for d in decls {
                    let val = match &d.init {
                        Some(expr) => self.eval_expr(expr, env)?,
                        None => RuntimeValue::Undefined,
                    };
                    self.bind_pattern(&d.pattern, val, env, *kind)?;
                }
                Ok(Flow::Normal(RuntimeValue::Empty))
            }
            Stmt::FunctionDecl(_) => Ok(Flow::Normal(RuntimeValue::Empty)),
            Stmt::If {
                test,
                consequent,
                alternate,
            } => {
                let tv = self.eval_expr(test, env)?;
                if self.runtime_to_bool(&tv)? {
                    self.exec_stmt(consequent, env)
                } else if let Some(alt) = alternate {
                    self.exec_stmt(alt, env)
                } else {
                    Ok(Flow::Normal(RuntimeValue::Empty))
                }
            }
            Stmt::While { test, body } => {
                let mut last = RuntimeValue::Empty;
                loop {
                    self.tick()?;
                    let tv = self.eval_expr(test, env)?;
                    if !self.runtime_to_bool(&tv)? {
                        break;
                    }
                    match self.exec_stmt(body, env)? {
                        Flow::Normal(v) => {
                            if !matches!(v, RuntimeValue::Empty) {
                                last = v;
                            }
                        }
                        Flow::Break => break,
                        Flow::Continue => continue,
                        other @ (Flow::Return(_) | Flow::Throw(_)) => return Ok(other),
                    }
                }
                Ok(Flow::Normal(last))
            }
            Stmt::DoWhile { body, test } => {
                let mut last = RuntimeValue::Empty;
                loop {
                    self.tick()?;
                    match self.exec_stmt(body, env)? {
                        Flow::Normal(v) => {
                            if !matches!(v, RuntimeValue::Empty) {
                                last = v;
                            }
                        }
                        Flow::Break => break,
                        Flow::Continue => {}
                        other @ (Flow::Return(_) | Flow::Throw(_)) => return Ok(other),
                    }
                    let tv = self.eval_expr(test, env)?;
                    if !self.runtime_to_bool(&tv)? {
                        break;
                    }
                }
                Ok(Flow::Normal(last))
            }
            Stmt::For(for_stmt) => self.exec_for_stmt(for_stmt, env),
            Stmt::ForOf(for_of) => self.exec_for_of_stmt(for_of, env),
            Stmt::ForIn(for_in) => self.exec_for_in_stmt(for_in, env),
            Stmt::Switch {
                discriminant,
                cases,
            } => {
                let disc = self.eval_expr(discriminant, env)?;
                let switch_env = self.push_env(Some(env));
                let mut matched_idx = None;
                let mut default_idx = None;
                for (i, case) in cases.iter().enumerate() {
                    if let Some(test_expr) = &case.test {
                        let tv = self.eval_expr(test_expr, switch_env)?;
                        if self.strict_eq(&disc, &tv)? {
                            matched_idx = Some(i);
                            break;
                        }
                    } else if default_idx.is_none() {
                        default_idx = Some(i);
                    }
                }
                let start = matched_idx.or(default_idx);
                let mut last = RuntimeValue::Empty;
                if let Some(start_idx) = start {
                    for case in &cases[start_idx..] {
                        match self.exec_stmt_list(&case.consequent, switch_env)? {
                            Flow::Normal(v) => {
                                if !matches!(v, RuntimeValue::Empty) {
                                    last = v;
                                }
                            }
                            Flow::Break => break,
                            other @ (Flow::Continue | Flow::Return(_) | Flow::Throw(_)) => {
                                return Ok(other);
                            }
                        }
                    }
                }
                Ok(Flow::Normal(last))
            }
            Stmt::Break => Ok(Flow::Break),
            Stmt::Continue => Ok(Flow::Continue),
            Stmt::Return(opt) => {
                let val = match opt {
                    Some(e) => self.eval_expr(e, env)?,
                    None => RuntimeValue::Undefined,
                };
                Ok(Flow::Return(val))
            }
            Stmt::Throw(expr) => {
                let val = self.eval_expr(expr, env)?;
                Ok(Flow::Throw(val))
            }
            Stmt::Try(try_stmt) => self.exec_try_stmt(try_stmt, env),
        }
    }

    fn exec_stmt_list(&mut self, stmts: &[Stmt], env: usize) -> Result<Flow, ExprError> {
        let mut last = RuntimeValue::Empty;
        for stmt in stmts {
            match self.exec_stmt(stmt, env)? {
                Flow::Normal(v) => {
                    if !matches!(v, RuntimeValue::Empty) {
                        last = v;
                    }
                }
                other => return Ok(other),
            }
        }
        Ok(Flow::Normal(last))
    }

    fn exec_for_stmt(
        &mut self,
        for_stmt: &crate::ast::ForStmt,
        env: usize,
    ) -> Result<Flow, ExprError> {
        let mut is_lexical = false;
        let mut loop_env = env;

        if let Some(init) = &for_stmt.init {
            match init {
                ForInit::VarDecl { kind, decls } => {
                    if *kind != VarKind::Var {
                        is_lexical = true;
                        loop_env = self.push_env(Some(env));
                    }
                    for d in decls {
                        let val = match &d.init {
                            Some(expr) => self.eval_expr(expr, loop_env)?,
                            None => RuntimeValue::Undefined,
                        };
                        self.bind_pattern(&d.pattern, val, loop_env, *kind)?;
                    }
                }
                ForInit::Expr(e) => {
                    self.eval_expr(e, env)?;
                }
            }
        }

        let mut last = RuntimeValue::Empty;
        let mut current_env = loop_env;
        loop {
            self.tick()?;
            if let Some(test) = &for_stmt.test {
                let tv = self.eval_expr(test, current_env)?;
                if !self.runtime_to_bool(&tv)? {
                    break;
                }
            }

            // Per-iteration fresh frame for `let`/`const` so closures capture iteration values.
            let body_env = if is_lexical {
                self.clone_lexical_env(current_env, env)
            } else {
                current_env
            };

            match self.exec_stmt(&for_stmt.body, body_env)? {
                Flow::Normal(v) => {
                    if !matches!(v, RuntimeValue::Empty) {
                        last = v;
                    }
                }
                Flow::Break => break,
                Flow::Continue => {}
                other @ (Flow::Return(_) | Flow::Throw(_)) => return Ok(other),
            }

            if is_lexical {
                current_env = self.clone_lexical_env(body_env, env);
            }
            if let Some(update) = &for_stmt.update {
                self.eval_expr(update, current_env)?;
            }
        }
        Ok(Flow::Normal(last))
    }

    fn clone_lexical_env(&mut self, src_env: usize, parent: usize) -> usize {
        let cloned_bindings: BTreeMap<String, Binding> = self.envs[src_env]
            .bindings
            .iter()
            .map(|(k, b)| {
                (
                    k.clone(),
                    Binding {
                        value: b.value.clone(),
                        mutable: b.mutable,
                        is_var: b.is_var,
                    },
                )
            })
            .collect();
        let id = self.envs.len();
        self.envs.push(EnvFrame {
            parent: Some(parent),
            is_function_scope: false,
            bindings: cloned_bindings,
        });
        id
    }

    fn exec_for_of_stmt(
        &mut self,
        for_of: &crate::ast::ForOfStmt,
        env: usize,
    ) -> Result<Flow, ExprError> {
        let iter_val = self.eval_expr(&for_of.iterable, env)?;
        let iter_val = self.deref_property_handle(iter_val)?;
        let items: Vec<RuntimeValue> = match iter_val {
            RuntimeValue::Array(id) => self.arrays[id].clone(),
            RuntimeValue::String(s) => s
                .chars()
                .map(|c| RuntimeValue::String(c.to_string()))
                .collect(),
            _ => {
                return Err(ExprError::TypeError(
                    "value is not iterable in for..of loop".to_string(),
                ));
            }
        };

        let mut last = RuntimeValue::Empty;
        for item in items {
            self.tick()?;
            let iter_env = match for_of.kind {
                Some(VarKind::Let | VarKind::Const) => self.push_env(Some(env)),
                _ => env,
            };
            match for_of.kind {
                Some(kind) => self.bind_pattern(&for_of.target, item, iter_env, kind)?,
                None => {
                    if let Pattern::Ident(name) = &for_of.target {
                        self.assign_local(env, name, item)?;
                    } else {
                        return Err(ExprError::TypeError(
                            "unsupported for..of assignment target".to_string(),
                        ));
                    }
                }
            }
            match self.exec_stmt(&for_of.body, iter_env)? {
                Flow::Normal(v) => {
                    if !matches!(v, RuntimeValue::Empty) {
                        last = v;
                    }
                }
                Flow::Break => break,
                Flow::Continue => continue,
                other @ (Flow::Return(_) | Flow::Throw(_)) => return Ok(other),
            }
        }
        Ok(Flow::Normal(last))
    }

    fn exec_for_in_stmt(
        &mut self,
        for_in: &crate::ast::ForInStmt,
        env: usize,
    ) -> Result<Flow, ExprError> {
        let obj_val = self.eval_expr(&for_in.object, env)?;
        let obj_val = self.deref_property_handle(obj_val)?;
        let keys: Vec<String> = match obj_val {
            RuntimeValue::Object(id) => self.objects[id].keys().cloned().collect(),
            RuntimeValue::Array(id) => (0..self.arrays[id].len()).map(|i| i.to_string()).collect(),
            RuntimeValue::Null | RuntimeValue::Undefined => Vec::new(),
            _ => Vec::new(),
        };

        let mut last = RuntimeValue::Empty;
        for key in keys {
            self.tick()?;
            let iter_env = match for_in.kind {
                Some(VarKind::Let | VarKind::Const) => self.push_env(Some(env)),
                _ => env,
            };
            let kv = RuntimeValue::String(key);
            match for_in.kind {
                Some(kind) => self.bind_pattern(&for_in.target, kv, iter_env, kind)?,
                None => {
                    if let Pattern::Ident(name) = &for_in.target {
                        self.assign_local(env, name, kv)?;
                    }
                }
            }
            match self.exec_stmt(&for_in.body, iter_env)? {
                Flow::Normal(v) => {
                    if !matches!(v, RuntimeValue::Empty) {
                        last = v;
                    }
                }
                Flow::Break => break,
                Flow::Continue => continue,
                other @ (Flow::Return(_) | Flow::Throw(_)) => return Ok(other),
            }
        }
        Ok(Flow::Normal(last))
    }

    fn exec_try_stmt(
        &mut self,
        try_stmt: &crate::ast::TryStmt,
        env: usize,
    ) -> Result<Flow, ExprError> {
        let try_env = self.push_env(Some(env));
        let try_res = self.exec_stmt_list(&try_stmt.block, try_env);

        let mut outcome: Result<Flow, ExprError> = match try_res {
            Ok(Flow::Throw(thrown)) => {
                if let Some(catch_body) = &try_stmt.catch_body {
                    let catch_env = self.push_env(Some(env));
                    if let Some(pat) = &try_stmt.catch_param {
                        self.bind_pattern(pat, thrown, catch_env, VarKind::Let)?;
                    }
                    self.exec_stmt_list(catch_body, catch_env)
                } else {
                    Ok(Flow::Throw(thrown))
                }
            }
            Err(err) if is_catchable_error(&err) => {
                if let Some(catch_body) = &try_stmt.catch_body {
                    let catch_env = self.push_env(Some(env));
                    if let Some(pat) = &try_stmt.catch_param {
                        let mut err_obj = BTreeMap::new();
                        err_obj
                            .insert("message".to_string(), RuntimeValue::String(err.to_string()));
                        let err_val = self.alloc_object(err_obj);
                        self.bind_pattern(pat, err_val, catch_env, VarKind::Let)?;
                    }
                    self.exec_stmt_list(catch_body, catch_env)
                } else {
                    Err(err)
                }
            }
            other => other,
        };

        if let Some(finally_body) = &try_stmt.finally_body {
            let finally_env = self.push_env(Some(env));
            match self.exec_stmt_list(finally_body, finally_env)? {
                Flow::Normal(_) => {}
                abrupt => outcome = Ok(abrupt),
            }
        }

        outcome
    }

    // -----------------------------------------------------------------------
    // Expression Evaluation
    // -----------------------------------------------------------------------

    fn eval_expr(&mut self, expr: &Expr, env: usize) -> Result<RuntimeValue, ExprError> {
        self.tick()?;
        match expr {
            Expr::Literal(lit) => Ok(match lit {
                Literal::Undefined => RuntimeValue::Undefined,
                Literal::Null => RuntimeValue::Null,
                Literal::Bool(b) => RuntimeValue::Bool(*b),
                Literal::Number(n) => RuntimeValue::Number(*n),
                Literal::String(s) => RuntimeValue::String(s.clone()),
            }),
            Expr::Ident(name) => self.resolve_ident(name, env),
            Expr::Template(parts) => {
                let mut out = String::new();
                for part in parts {
                    match part {
                        TemplatePart::Text(t) => out.push_str(t),
                        TemplatePart::Expr(e) => {
                            let v = self.eval_expr(e, env)?;
                            out.push_str(&self.runtime_to_js_string(&v)?);
                        }
                    }
                }
                Ok(RuntimeValue::String(out))
            }
            Expr::Array(elems) => {
                let mut items = Vec::with_capacity(elems.len());
                for elem in elems {
                    match elem {
                        ArrayElem::Expr(e) => items.push(self.eval_expr(e, env)?),
                        ArrayElem::Spread(e) => {
                            let val = self.eval_expr(e, env)?;
                            let val = self.deref_property_handle(val)?;
                            match val {
                                RuntimeValue::Array(id) => {
                                    let spread_items = self.arrays[id].clone();
                                    items.extend(spread_items);
                                }
                                RuntimeValue::String(s) => {
                                    items.extend(
                                        s.chars().map(|c| RuntimeValue::String(c.to_string())),
                                    );
                                }
                                _ => {
                                    return Err(ExprError::TypeError(
                                        "spread operand is not iterable".to_string(),
                                    ));
                                }
                            }
                        }
                    }
                }
                Ok(self.alloc_array(items))
            }
            Expr::Object(props) => {
                let mut map = BTreeMap::new();
                for prop in props {
                    match prop {
                        ObjectProp::KeyValue { key, value } => {
                            let k = match key {
                                ObjectKey::Static(s) => s.clone(),
                                ObjectKey::Computed(e) => {
                                    let kv = self.eval_expr(e, env)?;
                                    self.runtime_to_js_string(&kv)?
                                }
                            };
                            let v = self.eval_expr(value, env)?;
                            map.insert(k, v);
                        }
                        ObjectProp::Spread(e) => {
                            let val = self.eval_expr(e, env)?;
                            let val = self.deref_property_handle(val)?;
                            if let RuntimeValue::Object(id) = val {
                                let entries = self.objects[id].clone();
                                map.extend(entries);
                            }
                        }
                    }
                }
                Ok(self.alloc_object(map))
            }
            Expr::Unary { op, expr } => self.eval_unary(*op, expr, env),
            Expr::Update { op, prefix, target } => self.eval_update(*op, *prefix, target, env),
            Expr::Binary { op, left, right } => {
                let l = self.eval_expr(left, env)?;
                let r = self.eval_expr(right, env)?;
                self.eval_binary(*op, l, r)
            }
            Expr::Logical { op, left, right } => {
                let l = self.eval_expr(left, env)?;
                match op {
                    LogicalOp::And => {
                        if self.runtime_to_bool(&l)? {
                            self.eval_expr(right, env)
                        } else {
                            Ok(l)
                        }
                    }
                    LogicalOp::Or => {
                        if self.runtime_to_bool(&l)? {
                            Ok(l)
                        } else {
                            self.eval_expr(right, env)
                        }
                    }
                    LogicalOp::Nullish => {
                        let unwrapped = self.deref_property_handle(l.clone())?;
                        if matches!(unwrapped, RuntimeValue::Null | RuntimeValue::Undefined) {
                            self.eval_expr(right, env)
                        } else {
                            Ok(l)
                        }
                    }
                }
            }
            Expr::Conditional {
                test,
                consequent,
                alternate,
            } => {
                let tv = self.eval_expr(test, env)?;
                if self.runtime_to_bool(&tv)? {
                    self.eval_expr(consequent, env)
                } else {
                    self.eval_expr(alternate, env)
                }
            }
            Expr::Assign { op, target, value } => self.eval_assign(*op, target, value, env),
            Expr::Member { object, prop } => {
                let obj = self.eval_expr(object, env)?;
                self.get_property(&obj, prop)
            }
            Expr::Index { object, index } => {
                let obj = self.eval_expr(object, env)?;
                let idx = self.eval_expr(index, env)?;
                self.get_computed(&obj, &idx)
            }
            Expr::Call { callee, args } => self.eval_call_expr(callee, args, env),
            Expr::OptionalChain { base, ops } => self.eval_optional_chain(base, ops, env),
            Expr::Function(f) => {
                let cid = self.closures.len();
                self.closures.push(Closure {
                    params: f.params.clone(),
                    body: f.body.clone(),
                    captured_env: env,
                    name: f.name.clone(),
                });
                Ok(RuntimeValue::Closure(cid))
            }
            Expr::Sequence(items) => {
                let mut last = RuntimeValue::Undefined;
                for item in items {
                    last = self.eval_expr(item, env)?;
                }
                Ok(last)
            }
        }
    }

    fn resolve_ident(&mut self, name: &str, env: usize) -> Result<RuntimeValue, ExprError> {
        if is_forbidden_ident(name) {
            return Err(ExprError::Forbidden(name.to_string()));
        }
        if let Some(v) = self.lookup_local(env, name) {
            return Ok(v);
        }

        // Core globals available in both Vanilla CM3 and Expressions+
        match name {
            "Infinity" => return Ok(RuntimeValue::Number(f64::INFINITY)),
            "NaN" => return Ok(RuntimeValue::Number(f64::NAN)),
            "undefined" => return Ok(RuntimeValue::Undefined),
            "frame" => return Ok(RuntimeValue::Number(self.ctx.frame())),
            "time" => return Ok(RuntimeValue::Number(self.ctx.time())),
            "fps" => return Ok(RuntimeValue::Number(self.ctx.fps())),
            "value" => {
                let v = self.ctx.value();
                return Ok(self.intern_value(v));
            }
            "properties" => {
                let props = self.ctx.properties();
                return Ok(self.intern_value(Value::Object(props)));
            }
            "Math" => return Ok(RuntimeValue::Namespace(NamespaceKind::Math)),
            "isNaN" | "isFinite" | "parseInt" | "parseFloat" => {
                return Ok(RuntimeValue::BuiltinFn(name.to_string()));
            }
            // Exact 17 Vanilla CM3 built-in functions
            "add" | "sub" | "mul" | "div" | "clamp" | "dot" | "cross" | "normalize" | "length"
            | "wave" | "lerp" | "linear" | "catmullRom" | "ease" | "random" | "gaussRandom"
            | "shake" => {
                return Ok(RuntimeValue::BuiltinFn(name.to_string()));
            }
            _ => {}
        }

        // Top-level Math constants & functions (CM3 compatibility + Expressions+)
        if let Some(c) = math_constant(name) {
            return Ok(RuntimeValue::Number(c));
        }
        if is_top_level_math_fn(name) {
            return Ok(RuntimeValue::BuiltinFn(format!("Math.{name}")));
        }

        // Expressions+ globals & handles
        if self.config.mode == ExprMode::ExpressionsPlus {
            match name {
                "frameDuration" => return Ok(RuntimeValue::Number(self.ctx.frame_duration())),
                "thisProperty" => {
                    return Ok(match self.ctx.this_property_id() {
                        Some(id) => RuntimeValue::Handle(HandleRef::Property(id)),
                        None => RuntimeValue::Undefined,
                    });
                }
                "thisObject" => {
                    return Ok(match self.ctx.this_object_id() {
                        Some(id) => RuntimeValue::Handle(HandleRef::Object(id)),
                        None => RuntimeValue::Undefined,
                    });
                }
                "thisLayer" => {
                    return Ok(match self.ctx.this_layer_id() {
                        Some(id) => RuntimeValue::Handle(HandleRef::Layer(id)),
                        None => RuntimeValue::Undefined,
                    });
                }
                "thisClip" => {
                    return Ok(match self.ctx.this_clip_id() {
                        Some(id) => RuntimeValue::Handle(HandleRef::Clip(id)),
                        None => RuntimeValue::Undefined,
                    });
                }
                "thisSequence" | "thisComp" => {
                    return Ok(RuntimeValue::Handle(HandleRef::Sequence));
                }
                "controls" => return Ok(RuntimeValue::Handle(HandleRef::Controls)),
                "Array" => return Ok(RuntimeValue::Namespace(NamespaceKind::Array)),
                "Number" => return Ok(RuntimeValue::Namespace(NamespaceKind::Number)),
                "String" => return Ok(RuntimeValue::Namespace(NamespaceKind::String)),
                "Boolean" => return Ok(RuntimeValue::Namespace(NamespaceKind::Boolean)),
                "Object" => return Ok(RuntimeValue::Namespace(NamespaceKind::Object)),
                "JSON" => return Ok(RuntimeValue::Namespace(NamespaceKind::Json)),
                "easeIn" | "easeOut" | "degreesToRadians" | "radiansToDegrees" | "lookAt"
                | "seedRandom" | "wiggle" | "layer" | "effect" | "object" | "property" | "prop"
                | "control" | "toWorld" | "fromWorld" | "worldPosition" | "worldRotation"
                | "toScreen" | "fromScreen" => {
                    return Ok(RuntimeValue::BuiltinFn(name.to_string()));
                }
                _ => {}
            }
        }

        Err(ExprError::ReferenceError(name.to_string()))
    }

    fn eval_unary(
        &mut self,
        op: UnaryOp,
        expr: &Expr,
        env: usize,
    ) -> Result<RuntimeValue, ExprError> {
        match op {
            UnaryOp::TypeOf => {
                // `typeof undeclared` returns "undefined" in JS (except forbidden idents).
                if let Expr::Ident(name) = expr {
                    if is_forbidden_ident(name) {
                        return Err(ExprError::Forbidden(name.clone()));
                    }
                    let val = match self.resolve_ident(name, env) {
                        Ok(v) => v,
                        Err(ExprError::ReferenceError(_)) => RuntimeValue::Undefined,
                        Err(e) => return Err(e),
                    };
                    return Ok(RuntimeValue::String(self.typeof_value(&val)?.to_string()));
                }
                let val = self.eval_expr(expr, env)?;
                Ok(RuntimeValue::String(self.typeof_value(&val)?.to_string()))
            }
            UnaryOp::Delete => match expr {
                Expr::Member { object, prop } => {
                    let obj = self.eval_expr(object, env)?;
                    if let RuntimeValue::Object(id) = obj {
                        self.objects[id].remove(prop);
                    }
                    Ok(RuntimeValue::Bool(true))
                }
                Expr::Index { object, index } => {
                    let obj = self.eval_expr(object, env)?;
                    let idx = self.eval_expr(index, env)?;
                    match obj {
                        RuntimeValue::Object(id) => {
                            let k = self.runtime_to_js_string(&idx)?;
                            self.objects[id].remove(&k);
                        }
                        RuntimeValue::Array(id) => {
                            let i = self.runtime_to_number(&idx)?;
                            if i >= 0.0 && i.fract() == 0.0 {
                                let ui = i as usize;
                                if ui < self.arrays[id].len() {
                                    self.arrays[id][ui] = RuntimeValue::Undefined;
                                }
                            }
                        }
                        _ => {}
                    }
                    Ok(RuntimeValue::Bool(true))
                }
                _ => Ok(RuntimeValue::Bool(false)),
            },
            UnaryOp::Void => {
                let _ = self.eval_expr(expr, env)?;
                Ok(RuntimeValue::Undefined)
            }
            UnaryOp::Not => {
                let v = self.eval_expr(expr, env)?;
                Ok(RuntimeValue::Bool(!self.runtime_to_bool(&v)?))
            }
            UnaryOp::Neg => {
                let v = self.eval_expr(expr, env)?;
                if let Some(vec) = self.runtime_as_f64_vec(&v)? {
                    return Ok(self.alloc_f64_vec(vec.into_iter().map(|x| -x).collect()));
                }
                Ok(RuntimeValue::Number(-self.runtime_to_number(&v)?))
            }
            UnaryOp::Pos => {
                let v = self.eval_expr(expr, env)?;
                if let Some(vec) = self.runtime_as_f64_vec(&v)? {
                    return Ok(self.alloc_f64_vec(vec));
                }
                Ok(RuntimeValue::Number(self.runtime_to_number(&v)?))
            }
            UnaryOp::BitNot => {
                let v = self.eval_expr(expr, env)?;
                let n = self.runtime_to_number(&v)?;
                Ok(RuntimeValue::Number((!to_i32(n)) as f64))
            }
        }
    }

    fn typeof_value(&mut self, rv: &RuntimeValue) -> Result<&'static str, ExprError> {
        let rv = self.deref_property_handle(rv.clone())?;
        Ok(match rv {
            RuntimeValue::Empty | RuntimeValue::Undefined => "undefined",
            RuntimeValue::Null | RuntimeValue::Array(_) | RuntimeValue::Object(_) => "object",
            RuntimeValue::Bool(_) => "boolean",
            RuntimeValue::Number(_) => "number",
            RuntimeValue::String(_) => "string",
            RuntimeValue::Closure(_) | RuntimeValue::BuiltinFn(_) => "function",
            RuntimeValue::Namespace(NamespaceKind::Math | NamespaceKind::Json) => "object",
            RuntimeValue::Namespace(_) => "function",
            RuntimeValue::Handle(_) => "object",
        })
    }

    fn eval_update(
        &mut self,
        op: UpdateOp,
        prefix: bool,
        target: &AssignTarget,
        env: usize,
    ) -> Result<RuntimeValue, ExprError> {
        let delta = match op {
            UpdateOp::Inc => 1.0,
            UpdateOp::Dec => -1.0,
        };
        let old_val = self.read_target(target, env)?;
        let old_num = self.runtime_to_number(&old_val)?;
        let new_num = old_num + delta;
        self.write_target(target, RuntimeValue::Number(new_num), env)?;
        Ok(RuntimeValue::Number(if prefix { new_num } else { old_num }))
    }

    fn eval_assign(
        &mut self,
        op: AssignOp,
        target: &AssignTarget,
        value_expr: &Expr,
        env: usize,
    ) -> Result<RuntimeValue, ExprError> {
        let rhs = self.eval_expr(value_expr, env)?;
        let final_val = match op {
            AssignOp::Assign => rhs,
            AssignOp::AddAssign => {
                let lhs = self.read_target(target, env)?;
                self.eval_binary(BinaryOp::Add, lhs, rhs)?
            }
            AssignOp::SubAssign => {
                let lhs = self.read_target(target, env)?;
                self.eval_binary(BinaryOp::Sub, lhs, rhs)?
            }
            AssignOp::MulAssign => {
                let lhs = self.read_target(target, env)?;
                self.eval_binary(BinaryOp::Mul, lhs, rhs)?
            }
            AssignOp::DivAssign => {
                let lhs = self.read_target(target, env)?;
                self.eval_binary(BinaryOp::Div, lhs, rhs)?
            }
            AssignOp::RemAssign => {
                let lhs = self.read_target(target, env)?;
                self.eval_binary(BinaryOp::Rem, lhs, rhs)?
            }
            AssignOp::PowAssign => {
                let lhs = self.read_target(target, env)?;
                self.eval_binary(BinaryOp::Pow, lhs, rhs)?
            }
        };
        self.write_target(target, final_val.clone(), env)?;
        Ok(final_val)
    }

    fn read_target(
        &mut self,
        target: &AssignTarget,
        env: usize,
    ) -> Result<RuntimeValue, ExprError> {
        match target {
            AssignTarget::Ident(name) => self.resolve_ident(name, env),
            AssignTarget::Member { object, prop } => {
                let obj = self.eval_expr(object, env)?;
                self.get_property(&obj, prop)
            }
            AssignTarget::Index { object, index } => {
                let obj = self.eval_expr(object, env)?;
                let idx = self.eval_expr(index, env)?;
                self.get_computed(&obj, &idx)
            }
        }
    }

    fn write_target(
        &mut self,
        target: &AssignTarget,
        val: RuntimeValue,
        env: usize,
    ) -> Result<(), ExprError> {
        match target {
            AssignTarget::Ident(name) => self.assign_local(env, name, val),
            AssignTarget::Member { object, prop } => {
                let obj = self.eval_expr(object, env)?;
                match obj {
                    RuntimeValue::Object(id) => {
                        self.objects[id].insert(prop.clone(), val);
                        Ok(())
                    }
                    RuntimeValue::Handle(_) | RuntimeValue::Namespace(_) => Err(
                        ExprError::TypeError(format!("cannot mutate read-only property '{prop}'")),
                    ),
                    _ => Err(ExprError::TypeError(format!(
                        "cannot set property '{prop}' on non-object"
                    ))),
                }
            }
            AssignTarget::Index { object, index } => {
                let obj = self.eval_expr(object, env)?;
                let idx = self.eval_expr(index, env)?;
                match obj {
                    RuntimeValue::Array(id) => {
                        let i = self.runtime_to_number(&idx)?;
                        if i < 0.0 || i.fract() != 0.0 || !i.is_finite() {
                            return Err(ExprError::RangeError("invalid array index".to_string()));
                        }
                        let ui = i as usize;
                        if ui > 100_000 {
                            return Err(ExprError::RangeError(
                                "array index exceeds sandbox limit".to_string(),
                            ));
                        }
                        let arr = &mut self.arrays[id];
                        if ui >= arr.len() {
                            arr.resize(ui + 1, RuntimeValue::Undefined);
                        }
                        arr[ui] = val;
                        Ok(())
                    }
                    RuntimeValue::Object(id) => {
                        let key = self.runtime_to_js_string(&idx)?;
                        self.objects[id].insert(key, val);
                        Ok(())
                    }
                    _ => Err(ExprError::TypeError(
                        "cannot assign indexed property on non-object".to_string(),
                    )),
                }
            }
        }
    }

    fn eval_binary(
        &mut self,
        op: BinaryOp,
        left: RuntimeValue,
        right: RuntimeValue,
    ) -> Result<RuntimeValue, ExprError> {
        let left = self.deref_property_handle(left)?;
        let right = self.deref_property_handle(right)?;

        match op {
            BinaryOp::Add => {
                // String concatenation wins if either operand is a string.
                if matches!(left, RuntimeValue::String(_))
                    || matches!(right, RuntimeValue::String(_))
                {
                    let ls = self.runtime_to_js_string(&left)?;
                    let rs = self.runtime_to_js_string(&right)?;
                    return Ok(RuntimeValue::String(format!("{ls}{rs}")));
                }
                // Vector arithmetic (component-wise + scalar broadcast).
                if let Some(res) = self.try_vector_arith(&left, &right, BinaryOp::Add)? {
                    return Ok(res);
                }
                let l = self.runtime_to_number(&left)?;
                let r = self.runtime_to_number(&right)?;
                Ok(RuntimeValue::Number(l + r))
            }
            BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                if let Some(res) = self.try_vector_arith(&left, &right, op)? {
                    return Ok(res);
                }
                let l = self.runtime_to_number(&left)?;
                let r = self.runtime_to_number(&right)?;
                let out = match op {
                    BinaryOp::Sub => l - r,
                    BinaryOp::Mul => l * r,
                    BinaryOp::Div => l / r,
                    _ => unreachable!(),
                };
                Ok(RuntimeValue::Number(out))
            }
            BinaryOp::Rem => {
                let l = self.runtime_to_number(&left)?;
                let r = self.runtime_to_number(&right)?;
                Ok(RuntimeValue::Number(l % r))
            }
            BinaryOp::Pow => {
                let l = self.runtime_to_number(&left)?;
                let r = self.runtime_to_number(&right)?;
                Ok(RuntimeValue::Number(l.powf(r)))
            }
            BinaryOp::StrictEq => Ok(RuntimeValue::Bool(self.strict_eq(&left, &right)?)),
            BinaryOp::StrictNe => Ok(RuntimeValue::Bool(!self.strict_eq(&left, &right)?)),
            BinaryOp::Eq => Ok(RuntimeValue::Bool(self.loose_eq(&left, &right)?)),
            BinaryOp::Ne => Ok(RuntimeValue::Bool(!self.loose_eq(&left, &right)?)),
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                if let (RuntimeValue::String(a), RuntimeValue::String(b)) = (&left, &right) {
                    let ord = match op {
                        BinaryOp::Lt => a < b,
                        BinaryOp::Le => a <= b,
                        BinaryOp::Gt => a > b,
                        BinaryOp::Ge => a >= b,
                        _ => unreachable!(),
                    };
                    return Ok(RuntimeValue::Bool(ord));
                }
                let l = self.runtime_to_number(&left)?;
                let r = self.runtime_to_number(&right)?;
                let res = match op {
                    BinaryOp::Lt => l < r,
                    BinaryOp::Le => l <= r,
                    BinaryOp::Gt => l > r,
                    BinaryOp::Ge => l >= r,
                    _ => unreachable!(),
                };
                Ok(RuntimeValue::Bool(res))
            }
            BinaryOp::BitAnd => {
                let l = to_i32(self.runtime_to_number(&left)?);
                let r = to_i32(self.runtime_to_number(&right)?);
                Ok(RuntimeValue::Number((l & r) as f64))
            }
            BinaryOp::BitOr => {
                let l = to_i32(self.runtime_to_number(&left)?);
                let r = to_i32(self.runtime_to_number(&right)?);
                Ok(RuntimeValue::Number((l | r) as f64))
            }
            BinaryOp::BitXor => {
                let l = to_i32(self.runtime_to_number(&left)?);
                let r = to_i32(self.runtime_to_number(&right)?);
                Ok(RuntimeValue::Number((l ^ r) as f64))
            }
            BinaryOp::Shl => {
                let l = to_i32(self.runtime_to_number(&left)?);
                let r = to_u32(self.runtime_to_number(&right)?) & 0x1f;
                Ok(RuntimeValue::Number(l.wrapping_shl(r) as f64))
            }
            BinaryOp::Shr => {
                let l = to_i32(self.runtime_to_number(&left)?);
                let r = to_u32(self.runtime_to_number(&right)?) & 0x1f;
                Ok(RuntimeValue::Number(l.wrapping_shr(r) as f64))
            }
            BinaryOp::UShr => {
                let l = to_u32(self.runtime_to_number(&left)?);
                let r = to_u32(self.runtime_to_number(&right)?) & 0x1f;
                Ok(RuntimeValue::Number(l.wrapping_shr(r) as f64))
            }
        }
    }

    fn try_vector_arith(
        &mut self,
        left: &RuntimeValue,
        right: &RuntimeValue,
        op: BinaryOp,
    ) -> Result<Option<RuntimeValue>, ExprError> {
        let lv = self.runtime_as_f64_vec(left)?;
        let rv = self.runtime_as_f64_vec(right)?;
        match (lv, rv) {
            (Some(a), Some(b)) => {
                let out = match op {
                    BinaryOp::Add => vec_add(&a, &b),
                    BinaryOp::Sub => vec_sub(&a, &b),
                    BinaryOp::Mul => vec_mul_componentwise(&a, &b),
                    BinaryOp::Div => vec_div_componentwise(&a, &b),
                    _ => unreachable!(),
                };
                Ok(Some(self.alloc_f64_vec(out)))
            }
            (Some(a), None) if matches!(right, RuntimeValue::Number(_)) => {
                let s = self.runtime_to_number(right)?;
                let out = match op {
                    BinaryOp::Add => a.into_iter().map(|x| x + s).collect(),
                    BinaryOp::Sub => a.into_iter().map(|x| x - s).collect(),
                    BinaryOp::Mul => vec_mul_scalar(&a, s),
                    BinaryOp::Div => vec_div_scalar(&a, s),
                    _ => unreachable!(),
                };
                Ok(Some(self.alloc_f64_vec(out)))
            }
            (None, Some(b)) if matches!(left, RuntimeValue::Number(_)) => {
                let s = self.runtime_to_number(left)?;
                let out = match op {
                    BinaryOp::Add => b.into_iter().map(|x| s + x).collect(),
                    BinaryOp::Sub => b.into_iter().map(|x| s - x).collect(),
                    BinaryOp::Mul => vec_mul_scalar(&b, s),
                    BinaryOp::Div => b.into_iter().map(|x| s / x).collect(),
                    _ => unreachable!(),
                };
                Ok(Some(self.alloc_f64_vec(out)))
            }
            _ => Ok(None),
        }
    }

    fn strict_eq(&mut self, a: &RuntimeValue, b: &RuntimeValue) -> Result<bool, ExprError> {
        let a = self.deref_property_handle(a.clone())?;
        let b = self.deref_property_handle(b.clone())?;
        Ok(match (a, b) {
            (RuntimeValue::Undefined, RuntimeValue::Undefined)
            | (RuntimeValue::Null, RuntimeValue::Null) => true,
            (RuntimeValue::Bool(x), RuntimeValue::Bool(y)) => x == y,
            (RuntimeValue::Number(x), RuntimeValue::Number(y)) => x == y,
            (RuntimeValue::String(x), RuntimeValue::String(y)) => x == y,
            (RuntimeValue::Array(x), RuntimeValue::Array(y)) => x == y,
            (RuntimeValue::Object(x), RuntimeValue::Object(y)) => x == y,
            (RuntimeValue::Closure(x), RuntimeValue::Closure(y)) => x == y,
            (RuntimeValue::BuiltinFn(x), RuntimeValue::BuiltinFn(y)) => x == y,
            (RuntimeValue::Handle(x), RuntimeValue::Handle(y)) => x == y,
            _ => false,
        })
    }

    fn loose_eq(&mut self, a: &RuntimeValue, b: &RuntimeValue) -> Result<bool, ExprError> {
        let a = self.deref_property_handle(a.clone())?;
        let b = self.deref_property_handle(b.clone())?;
        match (&a, &b) {
            (RuntimeValue::Null, RuntimeValue::Undefined)
            | (RuntimeValue::Undefined, RuntimeValue::Null) => Ok(true),
            (RuntimeValue::Number(x), RuntimeValue::String(_)) => {
                Ok(*x == self.runtime_to_number(&b)?)
            }
            (RuntimeValue::String(_), RuntimeValue::Number(y)) => {
                Ok(self.runtime_to_number(&a)? == *y)
            }
            (RuntimeValue::Bool(_), _) => {
                let n = RuntimeValue::Number(self.runtime_to_number(&a)?);
                self.loose_eq(&n, &b)
            }
            (_, RuntimeValue::Bool(_)) => {
                let n = RuntimeValue::Number(self.runtime_to_number(&b)?);
                self.loose_eq(&a, &n)
            }
            _ => self.strict_eq(&a, &b),
        }
    }

    // -----------------------------------------------------------------------
    // Property Access & Optional Chaining
    // -----------------------------------------------------------------------

    fn get_property(&mut self, obj: &RuntimeValue, prop: &str) -> Result<RuntimeValue, ExprError> {
        if matches!(prop, "__proto__" | "constructor" | "prototype") {
            return Ok(RuntimeValue::Undefined);
        }

        match obj {
            RuntimeValue::Null | RuntimeValue::Undefined => Err(ExprError::TypeError(format!(
                "cannot read property '{prop}' of null or undefined"
            ))),
            RuntimeValue::Namespace(NamespaceKind::Math) => {
                if prop == "random" {
                    return Err(ExprError::Forbidden("Math.random".to_string()));
                }
                if let Some(c) = math_constant(prop) {
                    return Ok(RuntimeValue::Number(c));
                }
                Ok(RuntimeValue::BuiltinFn(format!("Math.{prop}")))
            }
            RuntimeValue::Namespace(NamespaceKind::Number) => match prop {
                "MAX_SAFE_INTEGER" => Ok(RuntimeValue::Number(9_007_199_254_740_991.0)),
                "MIN_SAFE_INTEGER" => Ok(RuntimeValue::Number(-9_007_199_254_740_991.0)),
                "EPSILON" => Ok(RuntimeValue::Number(f64::EPSILON)),
                "POSITIVE_INFINITY" => Ok(RuntimeValue::Number(f64::INFINITY)),
                "NEGATIVE_INFINITY" => Ok(RuntimeValue::Number(f64::NEG_INFINITY)),
                "NaN" => Ok(RuntimeValue::Number(f64::NAN)),
                _ => Ok(RuntimeValue::BuiltinFn(format!("Number.{prop}"))),
            },
            RuntimeValue::Namespace(NamespaceKind::Array) => {
                Ok(RuntimeValue::BuiltinFn(format!("Array.{prop}")))
            }
            RuntimeValue::Namespace(NamespaceKind::String) => {
                Ok(RuntimeValue::BuiltinFn(format!("String.{prop}")))
            }
            RuntimeValue::Namespace(NamespaceKind::Object) => {
                Ok(RuntimeValue::BuiltinFn(format!("Object.{prop}")))
            }
            RuntimeValue::Namespace(NamespaceKind::Json) => {
                Ok(RuntimeValue::BuiltinFn(format!("JSON.{prop}")))
            }
            RuntimeValue::Namespace(NamespaceKind::Boolean) => Ok(RuntimeValue::Undefined),
            RuntimeValue::Handle(handle) => self.get_handle_property(handle, prop),
            RuntimeValue::Array(id) => {
                if prop == "length" {
                    return Ok(RuntimeValue::Number(self.arrays[*id].len() as f64));
                }
                if let Ok(idx) = prop.parse::<usize>() {
                    return Ok(self.arrays[*id]
                        .get(idx)
                        .cloned()
                        .unwrap_or(RuntimeValue::Undefined));
                }
                Ok(RuntimeValue::Undefined)
            }
            RuntimeValue::String(s) => {
                if prop == "length" {
                    return Ok(RuntimeValue::Number(s.chars().count() as f64));
                }
                if let Ok(idx) = prop.parse::<usize>() {
                    return Ok(s
                        .chars()
                        .nth(idx)
                        .map(|c| RuntimeValue::String(c.to_string()))
                        .unwrap_or(RuntimeValue::Undefined));
                }
                Ok(RuntimeValue::Undefined)
            }
            RuntimeValue::Object(id) => Ok(self.objects[*id]
                .get(prop)
                .cloned()
                .unwrap_or(RuntimeValue::Undefined)),
            _ => Ok(RuntimeValue::Undefined),
        }
    }

    fn get_computed(
        &mut self,
        obj: &RuntimeValue,
        idx: &RuntimeValue,
    ) -> Result<RuntimeValue, ExprError> {
        let idx = self.deref_property_handle(idx.clone())?;
        if matches!(obj, RuntimeValue::Namespace(NamespaceKind::Math)) {
            let key = self.runtime_to_js_string(&idx)?;
            return self.get_property(obj, &key);
        }
        if matches!(obj, RuntimeValue::Handle(HandleRef::Controls)) {
            let name = self.runtime_to_js_string(&idx)?;
            let val = self.ctx.control(&name, None)?;
            return Ok(self.intern_value(val));
        }

        // Auto-deref PropertyHandle when indexed like `thisProperty[0]` or `prop("./Pos")[1]`.
        let obj = self.deref_property_handle(obj.clone())?;
        match obj {
            RuntimeValue::Null | RuntimeValue::Undefined => Err(ExprError::TypeError(
                "cannot read indexed property of null or undefined".to_string(),
            )),
            RuntimeValue::Array(id) => {
                if let RuntimeValue::Number(n) = idx {
                    if n >= 0.0 && n.fract() == 0.0 {
                        return Ok(self.arrays[id]
                            .get(n as usize)
                            .cloned()
                            .unwrap_or(RuntimeValue::Undefined));
                    }
                    return Ok(RuntimeValue::Undefined);
                }
                let key = self.runtime_to_js_string(&idx)?;
                self.get_property(&RuntimeValue::Array(id), &key)
            }
            RuntimeValue::String(s) => {
                if let RuntimeValue::Number(n) = idx {
                    if n >= 0.0 && n.fract() == 0.0 {
                        return Ok(s
                            .chars()
                            .nth(n as usize)
                            .map(|c| RuntimeValue::String(c.to_string()))
                            .unwrap_or(RuntimeValue::Undefined));
                    }
                    return Ok(RuntimeValue::Undefined);
                }
                let key = self.runtime_to_js_string(&idx)?;
                self.get_property(&RuntimeValue::String(s), &key)
            }
            RuntimeValue::Object(id) => {
                let key = self.runtime_to_js_string(&idx)?;
                self.get_property(&RuntimeValue::Object(id), &key)
            }
            other => {
                let key = self.runtime_to_js_string(&idx)?;
                self.get_property(&other, &key)
            }
        }
    }

    fn get_handle_property(
        &mut self,
        handle: &HandleRef,
        prop: &str,
    ) -> Result<RuntimeValue, ExprError> {
        match handle {
            HandleRef::Property(id) => match prop {
                "value" => {
                    let v = self.ctx.property_value(id)?;
                    Ok(self.intern_value(v))
                }
                "velocity" => {
                    let v = self.ctx.property_velocity(id, self.ctx.time())?;
                    Ok(self.intern_value(v))
                }
                "speed" => {
                    let s = self.ctx.property_speed(id, self.ctx.time())?;
                    Ok(RuntimeValue::Number(s))
                }
                "numKeys" => Ok(RuntimeValue::Number(self.ctx.property_num_keys(id) as f64)),
                "name" => Ok(self
                    .ctx
                    .handle_name(id)
                    .map(RuntimeValue::String)
                    .unwrap_or(RuntimeValue::Undefined)),
                "parent" => match self.ctx.parent_of(id, 1)? {
                    Some(p) => Ok(RuntimeValue::Handle(p)),
                    None => Ok(RuntimeValue::Null),
                },
                "length" => {
                    let v = self.ctx.property_value(id)?;
                    let rv = self.intern_value(v);
                    self.get_property(&rv, "length")
                }
                _ => Ok(RuntimeValue::Undefined),
            },
            HandleRef::Object(id)
            | HandleRef::Layer(id)
            | HandleRef::Clip(id)
            | HandleRef::Effect(id) => match prop {
                "name" => Ok(self
                    .ctx
                    .handle_name(id)
                    .map(RuntimeValue::String)
                    .unwrap_or(RuntimeValue::Undefined)),
                "parent" => match self.ctx.parent_of(id, 1)? {
                    Some(p) => Ok(RuntimeValue::Handle(p)),
                    None => Ok(RuntimeValue::Null),
                },
                _ => Ok(RuntimeValue::Undefined),
            },
            HandleRef::Sequence => match prop {
                "frameDuration" => Ok(RuntimeValue::Number(self.ctx.frame_duration())),
                "fps" => Ok(RuntimeValue::Number(self.ctx.fps())),
                _ => Ok(RuntimeValue::Undefined),
            },
            HandleRef::Controls => {
                let val = self.ctx.control(prop, None)?;
                Ok(self.intern_value(val))
            }
        }
    }

    fn eval_optional_chain(
        &mut self,
        base: &Expr,
        ops: &[ChainOp],
        env: usize,
    ) -> Result<RuntimeValue, ExprError> {
        let mut current = self.eval_expr(base, env)?;
        let mut idx = 0;
        while idx < ops.len() {
            let is_nullish = {
                let derefed = self.deref_property_handle(current.clone())?;
                matches!(derefed, RuntimeValue::Null | RuntimeValue::Undefined)
            };
            match &ops[idx] {
                ChainOp::Member { prop, optional } => {
                    if *optional && is_nullish {
                        return Ok(RuntimeValue::Undefined);
                    }
                    // Look ahead for method call `a?.b(args)` or `a?.b?.(args)`.
                    if let Some(ChainOp::Call {
                        args,
                        optional: call_opt,
                    }) = ops.get(idx + 1)
                    {
                        let eval_args = self.eval_call_args(args, env)?;
                        if *call_opt && is_nullish {
                            return Ok(RuntimeValue::Undefined);
                        }
                        current = self.call_method_on(current, prop, eval_args, *call_opt)?;
                        idx += 2;
                        continue;
                    }
                    current = self.get_property(&current, prop)?;
                }
                ChainOp::Index { index, optional } => {
                    if *optional && is_nullish {
                        return Ok(RuntimeValue::Undefined);
                    }
                    let key = self.eval_expr(index, env)?;
                    current = self.get_computed(&current, &key)?;
                }
                ChainOp::Call { args, optional } => {
                    if *optional && is_nullish {
                        return Ok(RuntimeValue::Undefined);
                    }
                    let eval_args = self.eval_call_args(args, env)?;
                    current = self.call_runtime_value(current, eval_args)?;
                }
            }
            idx += 1;
        }
        Ok(current)
    }

    // -----------------------------------------------------------------------
    // Function & Method Calls
    // -----------------------------------------------------------------------

    fn eval_call_args(
        &mut self,
        args: &[CallArg],
        env: usize,
    ) -> Result<Vec<RuntimeValue>, ExprError> {
        let mut out = Vec::with_capacity(args.len());
        for arg in args {
            match arg {
                CallArg::Expr(e) => out.push(self.eval_expr(e, env)?),
                CallArg::Spread(e) => {
                    let v = self.eval_expr(e, env)?;
                    let v = self.deref_property_handle(v)?;
                    match v {
                        RuntimeValue::Array(id) => out.extend(self.arrays[id].clone()),
                        RuntimeValue::String(s) => {
                            out.extend(s.chars().map(|c| RuntimeValue::String(c.to_string())));
                        }
                        _ => {
                            return Err(ExprError::TypeError(
                                "cannot spread non-iterable call argument".to_string(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn eval_call_expr(
        &mut self,
        callee: &Expr,
        args: &[CallArg],
        env: usize,
    ) -> Result<RuntimeValue, ExprError> {
        if let Expr::Member { object, prop } = callee {
            let receiver = self.eval_expr(object, env)?;
            let eval_args = self.eval_call_args(args, env)?;
            return self.call_method_on(receiver, prop, eval_args, false);
        }
        let callee_val = self.eval_expr(callee, env)?;
        let eval_args = self.eval_call_args(args, env)?;
        self.call_runtime_value(callee_val, eval_args)
    }

    fn call_method_on(
        &mut self,
        receiver: RuntimeValue,
        method: &str,
        args: Vec<RuntimeValue>,
        optional_call: bool,
    ) -> Result<RuntimeValue, ExprError> {
        if matches!(receiver, RuntimeValue::Null | RuntimeValue::Undefined) {
            return Err(ExprError::TypeError(format!(
                "cannot call method '{method}' on null or undefined"
            )));
        }

        // Namespaces (`Math.*`, `Array.*`, `Number.*`, `Object.*`, `JSON.*`, `String.*`)
        if let RuntimeValue::Namespace(ns) = receiver {
            return self.call_namespace_method(ns, method, args);
        }

        // Scene / Property / Layer / Object / Effect handles
        if let RuntimeValue::Handle(handle) = receiver {
            return self.call_handle_method(&handle, method, args);
        }

        // User-defined object properties that are functions
        if let RuntimeValue::Object(id) = receiver {
            let val = self.objects[id]
                .get(method)
                .cloned()
                .unwrap_or(RuntimeValue::Undefined);
            if optional_call && matches!(val, RuntimeValue::Null | RuntimeValue::Undefined) {
                return Ok(RuntimeValue::Undefined);
            }
            return self.call_runtime_value(val, args);
        }

        // Built-in prototype methods on Array, String, Number
        match receiver {
            RuntimeValue::Array(id) => self.call_array_method(id, method, args),
            RuntimeValue::String(s) => self.call_string_method(&s, method, args),
            RuntimeValue::Number(n) => self.call_number_method(n, method, args),
            _ => {
                if optional_call {
                    Ok(RuntimeValue::Undefined)
                } else {
                    Err(ExprError::TypeError(format!(
                        "'{method}' is not a function"
                    )))
                }
            }
        }
    }

    fn call_runtime_value(
        &mut self,
        callee: RuntimeValue,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        self.tick()?;
        match callee {
            RuntimeValue::Closure(cid) => self.call_closure(cid, args),
            RuntimeValue::BuiltinFn(name) => self.call_builtin(&name, args),
            RuntimeValue::Handle(HandleRef::Effect(effect_id)) => {
                // Callable effect handle: `effect("Glow")("Radius")`.
                let param_name = match args.first() {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => {
                        return Err(ExprError::TypeError(
                            "effect parameter lookup requires a name".to_string(),
                        ));
                    }
                };
                let prop_id = self.ctx.effect_param(&effect_id, &param_name)?;
                Ok(RuntimeValue::Handle(HandleRef::Property(prop_id)))
            }
            RuntimeValue::Namespace(NamespaceKind::Number) => {
                let n = match args.first() {
                    Some(v) => self.runtime_to_number(v)?,
                    None => 0.0,
                };
                Ok(RuntimeValue::Number(n))
            }
            RuntimeValue::Namespace(NamespaceKind::String) => {
                let s = match args.first() {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => String::new(),
                };
                Ok(RuntimeValue::String(s))
            }
            RuntimeValue::Namespace(NamespaceKind::Boolean) => {
                let b = match args.first() {
                    Some(v) => self.runtime_to_bool(v)?,
                    None => false,
                };
                Ok(RuntimeValue::Bool(b))
            }
            RuntimeValue::Namespace(NamespaceKind::Array) => {
                if args.len() == 1
                    && let RuntimeValue::Number(n) = args[0]
                    && n >= 0.0
                    && n.fract() == 0.0
                    && n <= 100_000.0
                {
                    return Ok(self.alloc_array(vec![RuntimeValue::Undefined; n as usize]));
                }
                Ok(self.alloc_array(args))
            }
            _ => Err(ExprError::TypeError("value is not callable".to_string())),
        }
    }

    fn call_closure(
        &mut self,
        cid: usize,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.config.max_recursion {
            self.recursion_depth -= 1;
            return Err(ExprError::RecursionLimitExceeded {
                limit: self.config.max_recursion,
            });
        }

        let res = self.call_closure_inner(cid, args);
        self.recursion_depth -= 1;
        res
    }

    fn call_closure_inner(
        &mut self,
        cid: usize,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        let params = self.closures[cid].params.clone();
        let body = self.closures[cid].body.clone();
        let captured_env = self.closures[cid].captured_env;
        let fn_name = self.closures[cid].name.clone();

        let call_env = self.push_fn_env(captured_env);
        if let Some(name) = fn_name {
            self.define_binding(call_env, name, RuntimeValue::Closure(cid), VarKind::Const);
        }

        for (i, param) in params.iter().enumerate() {
            if param.is_rest {
                let rest = if i < args.len() {
                    args[i..].to_vec()
                } else {
                    Vec::new()
                };
                let arr = self.alloc_array(rest);
                self.bind_pattern(&param.pattern, arr, call_env, VarKind::Let)?;
            } else {
                let mut val = args.get(i).cloned().unwrap_or(RuntimeValue::Undefined);
                if matches!(val, RuntimeValue::Undefined)
                    && let Some(def_expr) = &param.default
                {
                    val = self.eval_expr(def_expr, call_env)?;
                }
                self.bind_pattern(&param.pattern, val, call_env, VarKind::Let)?;
            }
        }

        match body {
            FunctionBody::Expr(expr) => self.eval_expr(&expr, call_env),
            FunctionBody::Block(stmts) => {
                self.hoist_decls(&stmts, call_env)?;
                match self.exec_stmt_list(&stmts, call_env)? {
                    Flow::Normal(_) => Ok(RuntimeValue::Undefined),
                    Flow::Return(v) => Ok(v),
                    Flow::Throw(v) => {
                        let msg = self.runtime_to_js_string(&v)?;
                        Err(ExprError::Thrown(msg))
                    }
                    Flow::Break | Flow::Continue => Err(ExprError::SyntaxError {
                        offset: 0,
                        message: "illegal jump statement inside function".to_string(),
                    }),
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Namespace & Handle Methods
    // -----------------------------------------------------------------------

    fn call_namespace_method(
        &mut self,
        ns: NamespaceKind,
        method: &str,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        match ns {
            NamespaceKind::Math => {
                if method == "random" {
                    return Err(ExprError::Forbidden("Math.random".to_string()));
                }
                let mut nums = Vec::with_capacity(args.len());
                for a in &args {
                    nums.push(self.runtime_to_number(a)?);
                }
                match eval_math_fn(method, &nums)? {
                    Some(v) => Ok(RuntimeValue::Number(v)),
                    None => Err(ExprError::TypeError(format!(
                        "Math.{method} is not a function"
                    ))),
                }
            }
            NamespaceKind::Array => match method {
                "isArray" => {
                    let is_arr = match args.first() {
                        Some(v) => {
                            let d = self.deref_property_handle(v.clone())?;
                            matches!(d, RuntimeValue::Array(_))
                        }
                        None => false,
                    };
                    Ok(RuntimeValue::Bool(is_arr))
                }
                "from" => {
                    let Some(src) = args.first() else {
                        return Ok(self.alloc_array(Vec::new()));
                    };
                    let src = self.deref_property_handle(src.clone())?;
                    let items = match src {
                        RuntimeValue::Array(id) => self.arrays[id].clone(),
                        RuntimeValue::String(s) => s
                            .chars()
                            .map(|c| RuntimeValue::String(c.to_string()))
                            .collect(),
                        _ => Vec::new(),
                    };
                    Ok(self.alloc_array(items))
                }
                _ => Err(ExprError::TypeError(format!(
                    "Array.{method} is not a function"
                ))),
            },
            NamespaceKind::Number => match method {
                "isFinite" => {
                    let ok = matches!(
                        args.first().cloned().map(|v| self.deref_property_handle(v)),
                        Some(Ok(RuntimeValue::Number(n))) if n.is_finite()
                    );
                    Ok(RuntimeValue::Bool(ok))
                }
                "isNaN" => {
                    let ok = matches!(
                        args.first().cloned().map(|v| self.deref_property_handle(v)),
                        Some(Ok(RuntimeValue::Number(n))) if n.is_nan()
                    );
                    Ok(RuntimeValue::Bool(ok))
                }
                "isInteger" => {
                    let ok = matches!(
                        args.first().cloned().map(|v| self.deref_property_handle(v)),
                        Some(Ok(RuntimeValue::Number(n))) if n.is_finite() && n.fract() == 0.0
                    );
                    Ok(RuntimeValue::Bool(ok))
                }
                "parseFloat" => {
                    let s = match args.first() {
                        Some(v) => self.runtime_to_js_string(v)?,
                        None => String::new(),
                    };
                    Ok(RuntimeValue::Number(js_parse_float(&s)))
                }
                "parseInt" => {
                    let s = match args.first() {
                        Some(v) => self.runtime_to_js_string(v)?,
                        None => String::new(),
                    };
                    let radix = match args.get(1) {
                        Some(v) => Some(to_i32(self.runtime_to_number(v)?)),
                        None => None,
                    };
                    Ok(RuntimeValue::Number(js_parse_int(&s, radix)))
                }
                _ => Err(ExprError::TypeError(format!(
                    "Number.{method} is not a function"
                ))),
            },
            NamespaceKind::String => match method {
                "fromCharCode" => {
                    let mut out = String::new();
                    for a in &args {
                        let code = to_u32(self.runtime_to_number(a)?) & 0xFFFF;
                        if let Some(ch) = char::from_u32(code) {
                            out.push(ch);
                        }
                    }
                    Ok(RuntimeValue::String(out))
                }
                _ => Err(ExprError::TypeError(format!(
                    "String.{method} is not a function"
                ))),
            },
            NamespaceKind::Object => match method {
                "keys" => {
                    let Some(v) = args.first() else {
                        return Ok(self.alloc_array(Vec::new()));
                    };
                    let v = self.deref_property_handle(v.clone())?;
                    let keys: Vec<RuntimeValue> = match v {
                        RuntimeValue::Object(id) => self.objects[id]
                            .keys()
                            .cloned()
                            .map(RuntimeValue::String)
                            .collect(),
                        RuntimeValue::Array(id) => (0..self.arrays[id].len())
                            .map(|i| RuntimeValue::String(i.to_string()))
                            .collect(),
                        _ => Vec::new(),
                    };
                    Ok(self.alloc_array(keys))
                }
                "values" => {
                    let Some(v) = args.first() else {
                        return Ok(self.alloc_array(Vec::new()));
                    };
                    let v = self.deref_property_handle(v.clone())?;
                    let vals: Vec<RuntimeValue> = match v {
                        RuntimeValue::Object(id) => self.objects[id].values().cloned().collect(),
                        RuntimeValue::Array(id) => self.arrays[id].clone(),
                        _ => Vec::new(),
                    };
                    Ok(self.alloc_array(vals))
                }
                "entries" => {
                    let Some(v) = args.first() else {
                        return Ok(self.alloc_array(Vec::new()));
                    };
                    let v = self.deref_property_handle(v.clone())?;
                    let pairs: Vec<(String, RuntimeValue)> = match v {
                        RuntimeValue::Object(id) => self.objects[id]
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                        RuntimeValue::Array(id) => self.arrays[id]
                            .iter()
                            .enumerate()
                            .map(|(i, v)| (i.to_string(), v.clone()))
                            .collect(),
                        _ => Vec::new(),
                    };
                    let mut out = Vec::with_capacity(pairs.len());
                    for (k, val) in pairs {
                        out.push(self.alloc_array(vec![RuntimeValue::String(k), val]));
                    }
                    Ok(self.alloc_array(out))
                }
                "assign" => {
                    let Some(target) = args.first().cloned() else {
                        return Err(ExprError::TypeError(
                            "Object.assign requires a target object".to_string(),
                        ));
                    };
                    let RuntimeValue::Object(tid) = target else {
                        return Err(ExprError::TypeError(
                            "Object.assign target must be an object".to_string(),
                        ));
                    };
                    for src in args.iter().skip(1) {
                        let d = self.deref_property_handle(src.clone())?;
                        if let RuntimeValue::Object(sid) = d {
                            let entries = self.objects[sid].clone();
                            self.objects[tid].extend(entries);
                        }
                    }
                    Ok(RuntimeValue::Object(tid))
                }
                _ => Err(ExprError::TypeError(format!(
                    "Object.{method} is not a function"
                ))),
            },
            NamespaceKind::Json => match method {
                "stringify" => {
                    let val = match args.first() {
                        Some(v) => self.export_value(v, 0)?,
                        None => return Ok(RuntimeValue::Undefined),
                    };
                    if matches!(val, Value::Undefined) {
                        return Ok(RuntimeValue::Undefined);
                    }
                    Ok(RuntimeValue::String(json_stringify(&val)?))
                }
                "parse" => {
                    let s = match args.first() {
                        Some(v) => self.runtime_to_js_string(v)?,
                        None => {
                            return Err(ExprError::SyntaxError {
                                offset: 0,
                                message: "JSON.parse requires a string argument".to_string(),
                            });
                        }
                    };
                    let parsed = json_parse(&s)?;
                    Ok(self.intern_value(parsed))
                }
                _ => Err(ExprError::TypeError(format!(
                    "JSON.{method} is not a function"
                ))),
            },
            NamespaceKind::Boolean => Err(ExprError::TypeError(format!(
                "Boolean.{method} is not a function"
            ))),
        }
    }

    fn call_handle_method(
        &mut self,
        handle: &HandleRef,
        method: &str,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        match handle {
            HandleRef::Property(id) => match method {
                "valueAtTime" => {
                    let t = match args.first() {
                        Some(v) => self.runtime_to_number(v)?,
                        None => self.ctx.time(),
                    };
                    let val = self.ctx.property_value_at_time(id, t)?;
                    Ok(self.intern_value(val))
                }
                "velocityAtTime" => {
                    let t = match args.first() {
                        Some(v) => self.runtime_to_number(v)?,
                        None => self.ctx.time(),
                    };
                    let val = self.ctx.property_velocity(id, t)?;
                    Ok(self.intern_value(val))
                }
                "speedAtTime" => {
                    let t = match args.first() {
                        Some(v) => self.runtime_to_number(v)?,
                        None => self.ctx.time(),
                    };
                    Ok(RuntimeValue::Number(self.ctx.property_speed(id, t)?))
                }
                "key" => {
                    let idx_f = match args.first() {
                        Some(v) => self.runtime_to_number(v)?,
                        None => {
                            return Err(ExprError::RangeError(
                                "key() requires a 1-based index".to_string(),
                            ));
                        }
                    };
                    if idx_f < 1.0 || idx_f.fract() != 0.0 || !idx_f.is_finite() {
                        return Err(ExprError::RangeError(format!(
                            "key index {idx_f} must be a positive 1-based integer"
                        )));
                    }
                    let kf = self.ctx.property_key(id, idx_f as usize)?;
                    Ok(self.keyframe_to_object(kf))
                }
                "nearestKey" => {
                    let t = match args.first() {
                        Some(v) => self.runtime_to_number(v)?,
                        None => self.ctx.time(),
                    };
                    let kf = self.ctx.property_nearest_key(id, t)?;
                    Ok(self.keyframe_to_object(kf))
                }
                "parent" => self.resolve_parent_call(id, &args),
                _ => {
                    // Fall back to methods on the property's underlying value (e.g. array/string).
                    let val = self.ctx.property_value(id)?;
                    let rv = self.intern_value(val);
                    self.call_method_on(rv, method, args, false)
                }
            },
            HandleRef::Effect(id) => match method {
                "param" | "property" => {
                    let name = self.expect_string_arg(&args, 0, method)?;
                    let pid = self.ctx.effect_param(id, &name)?;
                    Ok(RuntimeValue::Handle(HandleRef::Property(pid)))
                }
                "parent" => self.resolve_parent_call(id, &args),
                _ => Err(ExprError::TypeError(format!(
                    "Effect.{method} is not a function"
                ))),
            },
            HandleRef::Object(id) | HandleRef::Layer(id) | HandleRef::Clip(id) => match method {
                "parent" => self.resolve_parent_call(id, &args),
                "effect" => {
                    let name = self.expect_string_arg(&args, 0, "effect")?;
                    let eid = self.ctx.effect(Some(id), &name)?;
                    Ok(RuntimeValue::Handle(HandleRef::Effect(eid)))
                }
                "object" => {
                    let name = self.expect_string_arg(&args, 0, "object")?;
                    let oid = self.ctx.object(Some(id), &name)?;
                    Ok(RuntimeValue::Handle(HandleRef::Object(oid)))
                }
                "property" => {
                    let name = self.expect_string_arg(&args, 0, "property")?;
                    let pid = self.ctx.property(Some(id), &name)?;
                    Ok(RuntimeValue::Handle(HandleRef::Property(pid)))
                }
                "layer" => {
                    let query = self.parse_name_or_index_arg(&args)?;
                    let lid = self.ctx.layer(&query)?;
                    Ok(RuntimeValue::Handle(HandleRef::Layer(lid)))
                }
                "toWorld" => {
                    let pt = match args.first() {
                        Some(v) => self.runtime_to_vec3(v)?,
                        None => [0.0, 0.0, 0.0],
                    };
                    let out = self.ctx.to_world(Some(id), pt)?;
                    Ok(self.alloc_f64_vec(out.to_vec()))
                }
                "fromWorld" => {
                    let pt = match args.first() {
                        Some(v) => self.runtime_to_vec3(v)?,
                        None => [0.0, 0.0, 0.0],
                    };
                    let out = self.ctx.from_world(Some(id), pt)?;
                    Ok(self.alloc_f64_vec(out.to_vec()))
                }
                "worldPosition" => {
                    let out = self.ctx.world_position(Some(id))?;
                    Ok(self.alloc_f64_vec(out.to_vec()))
                }
                "worldRotation" => {
                    let out = self.ctx.world_rotation(Some(id))?;
                    Ok(self.alloc_f64_vec(out.to_vec()))
                }
                "toScreen" => {
                    let pt = match args.first() {
                        Some(v) => self.runtime_to_vec3(v)?,
                        None => [0.0, 0.0, 0.0],
                    };
                    let out = self.ctx.to_screen(Some(id), pt)?;
                    Ok(self.alloc_f64_vec(out.to_vec()))
                }
                "fromScreen" => {
                    let pt = match args.first() {
                        Some(v) => self.runtime_to_vec3(v)?,
                        None => [0.0, 0.0, 0.0],
                    };
                    let out = self.ctx.from_screen(Some(id), pt)?;
                    Ok(self.alloc_f64_vec(out.to_vec()))
                }
                _ => Err(ExprError::TypeError(format!(
                    "handle.{method} is not a function"
                ))),
            },
            HandleRef::Sequence => match method {
                "layer" => {
                    let query = self.parse_name_or_index_arg(&args)?;
                    let lid = self.ctx.layer(&query)?;
                    Ok(RuntimeValue::Handle(HandleRef::Layer(lid)))
                }
                _ => Err(ExprError::TypeError(format!(
                    "Sequence.{method} is not a function"
                ))),
            },
            HandleRef::Controls => Err(ExprError::TypeError(format!(
                "controls.{method} is not a function"
            ))),
        }
    }

    fn resolve_parent_call(
        &mut self,
        id: &HandleId,
        args: &[RuntimeValue],
    ) -> Result<RuntimeValue, ExprError> {
        let levels = match args.first() {
            Some(v) => {
                let n = self.runtime_to_number(v)?;
                if n < 1.0 || !n.is_finite() {
                    return Ok(RuntimeValue::Null);
                }
                n.floor() as usize
            }
            None => 1,
        };
        match self.ctx.parent_of(id, levels)? {
            Some(p) => Ok(RuntimeValue::Handle(p)),
            None => Ok(RuntimeValue::Null),
        }
    }

    fn expect_string_arg(
        &mut self,
        args: &[RuntimeValue],
        index: usize,
        fn_name: &str,
    ) -> Result<String, ExprError> {
        match args.get(index) {
            Some(v) => self.runtime_to_js_string(v),
            None => Err(ExprError::TypeError(format!(
                "{fn_name}() missing required string argument"
            ))),
        }
    }

    fn parse_name_or_index_arg(&mut self, args: &[RuntimeValue]) -> Result<NameOrIndex, ExprError> {
        let Some(first) = args.first() else {
            return Err(ExprError::TypeError(
                "layer() requires a name or 1-based index".to_string(),
            ));
        };
        let d = self.deref_property_handle(first.clone())?;
        match d {
            RuntimeValue::Number(n) => Ok(NameOrIndex::Index(n.trunc() as i64)),
            other => Ok(NameOrIndex::Name(self.runtime_to_js_string(&other)?)),
        }
    }

    // -----------------------------------------------------------------------
    // Array, String, Number Methods
    // -----------------------------------------------------------------------

    fn call_array_method(
        &mut self,
        id: usize,
        method: &str,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        match method {
            "push" => {
                self.arrays[id].extend(args);
                Ok(RuntimeValue::Number(self.arrays[id].len() as f64))
            }
            "pop" => Ok(self.arrays[id].pop().unwrap_or(RuntimeValue::Undefined)),
            "shift" => {
                if self.arrays[id].is_empty() {
                    Ok(RuntimeValue::Undefined)
                } else {
                    Ok(self.arrays[id].remove(0))
                }
            }
            "unshift" => {
                let mut new_vec = args;
                new_vec.append(&mut self.arrays[id]);
                self.arrays[id] = new_vec;
                Ok(RuntimeValue::Number(self.arrays[id].len() as f64))
            }
            "reverse" => {
                self.arrays[id].reverse();
                Ok(RuntimeValue::Array(id))
            }
            "slice" => {
                let len = self.arrays[id].len() as isize;
                let start = match args.first() {
                    Some(v) => normalize_slice_index(self.runtime_to_number(v)?, len),
                    None => 0,
                };
                let end = match args.get(1) {
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        normalize_slice_index(self.runtime_to_number(v)?, len)
                    }
                    _ => len as usize,
                };
                let slice = if start < end {
                    self.arrays[id][start..end].to_vec()
                } else {
                    Vec::new()
                };
                Ok(self.alloc_array(slice))
            }
            "concat" => {
                let mut out = self.arrays[id].clone();
                for a in args {
                    let d = self.deref_property_handle(a)?;
                    match d {
                        RuntimeValue::Array(other_id) => {
                            out.extend(self.arrays[other_id].clone());
                        }
                        other => out.push(other),
                    }
                }
                Ok(self.alloc_array(out))
            }
            "join" => {
                let sep = match args.first() {
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        self.runtime_to_js_string(v)?
                    }
                    _ => ",".to_string(),
                };
                Ok(RuntimeValue::String(self.join_array_to_string(id, &sep)?))
            }
            "indexOf" => {
                let needle = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let items = self.arrays[id].clone();
                for (i, item) in items.iter().enumerate() {
                    if self.strict_eq(item, &needle)? {
                        return Ok(RuntimeValue::Number(i as f64));
                    }
                }
                Ok(RuntimeValue::Number(-1.0))
            }
            "includes" => {
                let needle = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let items = self.arrays[id].clone();
                for item in &items {
                    if self.strict_eq(item, &needle)? {
                        return Ok(RuntimeValue::Bool(true));
                    }
                }
                Ok(RuntimeValue::Bool(false))
            }
            "map" => {
                let cb = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let items = self.arrays[id].clone();
                let mut out = Vec::with_capacity(items.len());
                for (i, item) in items.into_iter().enumerate() {
                    let res = self.call_runtime_value(
                        cb.clone(),
                        vec![
                            item,
                            RuntimeValue::Number(i as f64),
                            RuntimeValue::Array(id),
                        ],
                    )?;
                    out.push(res);
                }
                Ok(self.alloc_array(out))
            }
            "filter" => {
                let cb = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let items = self.arrays[id].clone();
                let mut out = Vec::new();
                for (i, item) in items.into_iter().enumerate() {
                    let keep = self.call_runtime_value(
                        cb.clone(),
                        vec![
                            item.clone(),
                            RuntimeValue::Number(i as f64),
                            RuntimeValue::Array(id),
                        ],
                    )?;
                    if self.runtime_to_bool(&keep)? {
                        out.push(item);
                    }
                }
                Ok(self.alloc_array(out))
            }
            "reduce" => {
                let cb = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let items = self.arrays[id].clone();
                let (mut acc, start_idx) = if let Some(init) = args.get(1) {
                    (init.clone(), 0)
                } else if !items.is_empty() {
                    (items[0].clone(), 1)
                } else {
                    return Err(ExprError::TypeError(
                        "reduce of empty array with no initial value".to_string(),
                    ));
                };
                for (i, item) in items.into_iter().enumerate().skip(start_idx) {
                    acc = self.call_runtime_value(
                        cb.clone(),
                        vec![
                            acc,
                            item,
                            RuntimeValue::Number(i as f64),
                            RuntimeValue::Array(id),
                        ],
                    )?;
                }
                Ok(acc)
            }
            "forEach" => {
                let cb = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let items = self.arrays[id].clone();
                for (i, item) in items.into_iter().enumerate() {
                    self.call_runtime_value(
                        cb.clone(),
                        vec![
                            item,
                            RuntimeValue::Number(i as f64),
                            RuntimeValue::Array(id),
                        ],
                    )?;
                }
                Ok(RuntimeValue::Undefined)
            }
            "find" => {
                let cb = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let items = self.arrays[id].clone();
                for (i, item) in items.into_iter().enumerate() {
                    let hit = self.call_runtime_value(
                        cb.clone(),
                        vec![
                            item.clone(),
                            RuntimeValue::Number(i as f64),
                            RuntimeValue::Array(id),
                        ],
                    )?;
                    if self.runtime_to_bool(&hit)? {
                        return Ok(item);
                    }
                }
                Ok(RuntimeValue::Undefined)
            }
            "some" => {
                let cb = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let items = self.arrays[id].clone();
                for (i, item) in items.into_iter().enumerate() {
                    let hit = self.call_runtime_value(
                        cb.clone(),
                        vec![
                            item,
                            RuntimeValue::Number(i as f64),
                            RuntimeValue::Array(id),
                        ],
                    )?;
                    if self.runtime_to_bool(&hit)? {
                        return Ok(RuntimeValue::Bool(true));
                    }
                }
                Ok(RuntimeValue::Bool(false))
            }
            "every" => {
                let cb = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let items = self.arrays[id].clone();
                for (i, item) in items.into_iter().enumerate() {
                    let hit = self.call_runtime_value(
                        cb.clone(),
                        vec![
                            item,
                            RuntimeValue::Number(i as f64),
                            RuntimeValue::Array(id),
                        ],
                    )?;
                    if !self.runtime_to_bool(&hit)? {
                        return Ok(RuntimeValue::Bool(false));
                    }
                }
                Ok(RuntimeValue::Bool(true))
            }
            "fill" => {
                let val = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let len = self.arrays[id].len() as isize;
                let start = match args.get(1) {
                    Some(v) => normalize_slice_index(self.runtime_to_number(v)?, len),
                    None => 0,
                };
                let end = match args.get(2) {
                    Some(v) => normalize_slice_index(self.runtime_to_number(v)?, len),
                    None => len as usize,
                };
                if start < end {
                    for slot in &mut self.arrays[id][start..end] {
                        *slot = val.clone();
                    }
                }
                Ok(RuntimeValue::Array(id))
            }
            _ => Err(ExprError::TypeError(format!(
                "Array.prototype.{method} is not a function"
            ))),
        }
    }

    fn call_string_method(
        &mut self,
        s: &str,
        method: &str,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        let chars: Vec<char> = s.chars().collect();
        let len = chars.len() as isize;
        match method {
            "toLowerCase" => Ok(RuntimeValue::String(s.to_lowercase())),
            "toUpperCase" => Ok(RuntimeValue::String(s.to_uppercase())),
            "trim" => Ok(RuntimeValue::String(s.trim().to_string())),
            "trimStart" => Ok(RuntimeValue::String(s.trim_start().to_string())),
            "trimEnd" => Ok(RuntimeValue::String(s.trim_end().to_string())),
            "slice" => {
                let start = match args.first() {
                    Some(v) => normalize_slice_index(self.runtime_to_number(v)?, len),
                    None => 0,
                };
                let end = match args.get(1) {
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        normalize_slice_index(self.runtime_to_number(v)?, len)
                    }
                    _ => len as usize,
                };
                let out: String = if start < end {
                    chars[start..end].iter().collect()
                } else {
                    String::new()
                };
                Ok(RuntimeValue::String(out))
            }
            "substring" => {
                let a = match args.first() {
                    Some(v) => clamp_f64(self.runtime_to_number(v)?, 0.0, len as f64) as usize,
                    None => 0,
                };
                let b = match args.get(1) {
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        clamp_f64(self.runtime_to_number(v)?, 0.0, len as f64) as usize
                    }
                    _ => len as usize,
                };
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                Ok(RuntimeValue::String(chars[lo..hi].iter().collect()))
            }
            "indexOf" => {
                let pat = match args.first() {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => "undefined".to_string(),
                };
                let idx = s
                    .find(&pat)
                    .map(|byte_idx| s[..byte_idx].chars().count() as f64)
                    .unwrap_or(-1.0);
                Ok(RuntimeValue::Number(idx))
            }
            "includes" => {
                let pat = match args.first() {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => "undefined".to_string(),
                };
                Ok(RuntimeValue::Bool(s.contains(&pat)))
            }
            "startsWith" => {
                let pat = match args.first() {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => "undefined".to_string(),
                };
                Ok(RuntimeValue::Bool(s.starts_with(&pat)))
            }
            "endsWith" => {
                let pat = match args.first() {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => "undefined".to_string(),
                };
                Ok(RuntimeValue::Bool(s.ends_with(&pat)))
            }
            "split" => {
                let sep = match args.first() {
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        self.runtime_to_js_string(v)?
                    }
                    _ => return Ok(self.alloc_array(vec![RuntimeValue::String(s.to_string())])),
                };
                let parts: Vec<RuntimeValue> = if sep.is_empty() {
                    chars
                        .into_iter()
                        .map(|c| RuntimeValue::String(c.to_string()))
                        .collect()
                } else {
                    s.split(&sep)
                        .map(|part| RuntimeValue::String(part.to_string()))
                        .collect()
                };
                Ok(self.alloc_array(parts))
            }
            "replace" => {
                let from = match args.first() {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => "undefined".to_string(),
                };
                let to = match args.get(1) {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => "undefined".to_string(),
                };
                Ok(RuntimeValue::String(s.replacen(&from, &to, 1)))
            }
            "replaceAll" => {
                let from = match args.first() {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => "undefined".to_string(),
                };
                let to = match args.get(1) {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => "undefined".to_string(),
                };
                Ok(RuntimeValue::String(s.replace(&from, &to)))
            }
            "charAt" => {
                let idx = match args.first() {
                    Some(v) => self.runtime_to_number(v)?,
                    None => 0.0,
                };
                let out = if idx >= 0.0 && (idx as usize) < chars.len() {
                    chars[idx as usize].to_string()
                } else {
                    String::new()
                };
                Ok(RuntimeValue::String(out))
            }
            "charCodeAt" => {
                let idx = match args.first() {
                    Some(v) => self.runtime_to_number(v)?,
                    None => 0.0,
                };
                let out = if idx >= 0.0 && (idx as usize) < chars.len() {
                    (chars[idx as usize] as u32) as f64
                } else {
                    f64::NAN
                };
                Ok(RuntimeValue::Number(out))
            }
            "repeat" => {
                let count = match args.first() {
                    Some(v) => self.runtime_to_number(v)?,
                    None => 0.0,
                };
                if count < 0.0 || !count.is_finite() || count > 10_000.0 {
                    return Err(ExprError::RangeError("invalid repeat count".to_string()));
                }
                Ok(RuntimeValue::String(s.repeat(count as usize)))
            }
            _ => Err(ExprError::TypeError(format!(
                "String.prototype.{method} is not a function"
            ))),
        }
    }

    fn call_number_method(
        &mut self,
        n: f64,
        method: &str,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        match method {
            "toFixed" => {
                let digits = match args.first() {
                    Some(v) => self.runtime_to_number(v)?,
                    None => 0.0,
                };
                if !(0.0..=100.0).contains(&digits) {
                    return Err(ExprError::RangeError(
                        "toFixed() digits argument must be between 0 and 100".to_string(),
                    ));
                }
                let prec = digits as usize;
                Ok(RuntimeValue::String(format!("{n:.prec$}")))
            }
            "toString" => Ok(RuntimeValue::String(format_js_number(n))),
            _ => Err(ExprError::TypeError(format!(
                "Number.prototype.{method} is not a function"
            ))),
        }
    }

    // -----------------------------------------------------------------------
    // Global Built-in Function Dispatch
    // -----------------------------------------------------------------------

    fn call_builtin(
        &mut self,
        name: &str,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        if let Some(math_name) = name.strip_prefix("Math.") {
            return self.call_namespace_method(NamespaceKind::Math, math_name, args);
        }
        if let Some(rest) = name.strip_prefix("Array.") {
            return self.call_namespace_method(NamespaceKind::Array, rest, args);
        }
        if let Some(rest) = name.strip_prefix("Number.") {
            return self.call_namespace_method(NamespaceKind::Number, rest, args);
        }
        if let Some(rest) = name.strip_prefix("String.") {
            return self.call_namespace_method(NamespaceKind::String, rest, args);
        }
        if let Some(rest) = name.strip_prefix("Object.") {
            return self.call_namespace_method(NamespaceKind::Object, rest, args);
        }
        if let Some(rest) = name.strip_prefix("JSON.") {
            return self.call_namespace_method(NamespaceKind::Json, rest, args);
        }

        match name {
            "isNaN" => {
                let n = match args.first() {
                    Some(v) => self.runtime_to_number(v)?,
                    None => f64::NAN,
                };
                Ok(RuntimeValue::Bool(n.is_nan()))
            }
            "isFinite" => {
                let n = match args.first() {
                    Some(v) => self.runtime_to_number(v)?,
                    None => f64::NAN,
                };
                Ok(RuntimeValue::Bool(n.is_finite()))
            }
            "parseInt" => self.call_namespace_method(NamespaceKind::Number, "parseInt", args),
            "parseFloat" => self.call_namespace_method(NamespaceKind::Number, "parseFloat", args),

            // --- Exact 17 Vanilla CM3 Built-ins (Spec 03 §6.4) ---
            "add" => {
                let a = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let b = args.get(1).cloned().unwrap_or(RuntimeValue::Undefined);
                self.eval_binary(BinaryOp::Add, a, b)
            }
            "sub" => {
                let a = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let b = args.get(1).cloned().unwrap_or(RuntimeValue::Undefined);
                self.eval_binary(BinaryOp::Sub, a, b)
            }
            "mul" => {
                let a = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let b = args.get(1).cloned().unwrap_or(RuntimeValue::Undefined);
                self.eval_binary(BinaryOp::Mul, a, b)
            }
            "div" => {
                let a = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let b = args.get(1).cloned().unwrap_or(RuntimeValue::Undefined);
                self.eval_binary(BinaryOp::Div, a, b)
            }
            "clamp" => self.builtin_clamp(&args),
            "dot" => {
                let a = self.require_vec_arg(&args, 0, "dot")?;
                let b = self.require_vec_arg(&args, 1, "dot")?;
                Ok(RuntimeValue::Number(vec_dot(&a, &b)))
            }
            "cross" => {
                let a = self.require_vec_arg(&args, 0, "cross")?;
                let b = self.require_vec_arg(&args, 1, "cross")?;
                Ok(self.alloc_f64_vec(vec_cross(&a, &b)))
            }
            "normalize" => {
                let v = self.require_vec_arg(&args, 0, "normalize")?;
                Ok(self.alloc_f64_vec(vec_normalize(&v)))
            }
            "length" => {
                let Some(first) = args.first() else {
                    return Ok(RuntimeValue::Number(0.0));
                };
                if let Some(second) = args.get(1) {
                    let a = self.runtime_as_f64_vec(first)?;
                    let b = self.runtime_as_f64_vec(second)?;
                    match (a, b) {
                        (Some(va), Some(vb)) => {
                            Ok(RuntimeValue::Number(vec_length(&vec_sub(&va, &vb))))
                        }
                        _ => {
                            let x = self.runtime_to_number(first)?;
                            let y = self.runtime_to_number(second)?;
                            Ok(RuntimeValue::Number((x - y).abs()))
                        }
                    }
                } else if let Some(va) = self.runtime_as_f64_vec(first)? {
                    Ok(RuntimeValue::Number(vec_length(&va)))
                } else {
                    Ok(RuntimeValue::Number(self.runtime_to_number(first)?.abs()))
                }
            }
            "wave" => {
                let n = match args.first() {
                    Some(v) => to_i32(self.runtime_to_number(v)?),
                    None => 0,
                };
                Ok(RuntimeValue::Number(cm3_wave(n)))
            }
            "lerp" => {
                let a = self.export_arg(&args, 0)?;
                let b = self.export_arg(&args, 1)?;
                let t = self.number_arg(&args, 2, 0.0)?;
                let res = lerp_value(&a, &b, t)?;
                Ok(self.intern_value(res))
            }
            "linear" => self.builtin_linear(&args),
            "catmullRom" => {
                let p0 = self.export_arg(&args, 0)?;
                let p1 = self.export_arg(&args, 1)?;
                let p2 = self.export_arg(&args, 2)?;
                let p3 = self.export_arg(&args, 3)?;
                let t = self.number_arg(&args, 4, 0.0)?;
                let res = catmull_rom_value(&p0, &p1, &p2, &p3, t)?;
                Ok(self.intern_value(res))
            }
            "ease" => self.builtin_ease(&args),
            "easeIn" => self.builtin_ae_ease_dir(&args, true),
            "easeOut" => self.builtin_ae_ease_dir(&args, false),
            "random" => self.builtin_random(&args),
            "gaussRandom" => {
                let seed = self.number_arg(&args, 0, 0.0)?;
                let mean = self.number_arg(&args, 1, 0.5)?;
                let std_dev = self.number_arg(&args, 2, 0.25)?;
                Ok(RuntimeValue::Number(cm3_gauss_random(seed, mean, std_dev)))
            }
            "shake" => {
                let f = self.number_arg(&args, 0, self.ctx.frame())?;
                let freq = self.number_arg(&args, 1, 1.0)?;
                let phase = self.number_arg(&args, 2, 0.0)?;
                let wave_amp = self.number_arg(&args, 3, 1.0)?;
                let noise_amp = self.number_arg(&args, 4, 0.0)?;
                let smooth = self.number_arg(&args, 5, 1.0)?;
                Ok(RuntimeValue::Number(cm3_shake(
                    f, freq, phase, wave_amp, noise_amp, smooth,
                )))
            }

            // --- Expressions+ Extended Math & Animation ---
            "degreesToRadians" => {
                let Some(first) = args.first() else {
                    return Ok(RuntimeValue::Number(f64::NAN));
                };
                if let Some(v) = self.runtime_as_f64_vec(first)? {
                    return Ok(self.alloc_f64_vec(v.into_iter().map(|x| x * PI / 180.0).collect()));
                }
                let x = self.runtime_to_number(first)?;
                Ok(RuntimeValue::Number(x * PI / 180.0))
            }
            "radiansToDegrees" => {
                let Some(first) = args.first() else {
                    return Ok(RuntimeValue::Number(f64::NAN));
                };
                if let Some(v) = self.runtime_as_f64_vec(first)? {
                    return Ok(self.alloc_f64_vec(v.into_iter().map(|x| x * 180.0 / PI).collect()));
                }
                let x = self.runtime_to_number(first)?;
                Ok(RuntimeValue::Number(x * 180.0 / PI))
            }
            "lookAt" => {
                let from = match args.first() {
                    Some(v) => self.runtime_to_vec3(v)?,
                    None => [0.0, 0.0, 0.0],
                };
                let to = match args.get(1) {
                    Some(v) => self.runtime_to_vec3(v)?,
                    None => [0.0, 0.0, 1.0],
                };
                let euler = look_at_xyz_euler(from, to);
                Ok(self.alloc_f64_vec(euler.to_vec()))
            }
            "seedRandom" => {
                let seed = self.number_arg(&args, 0, 0.0)?;
                let timeless = match args.get(1) {
                    Some(v) => self.runtime_to_bool(v)?,
                    None => false,
                };
                self.rng = Some(DetRng::from_seed(seed, self.ctx.frame(), timeless));
                Ok(RuntimeValue::Undefined)
            }
            "wiggle" => {
                let freq = self.number_arg(&args, 0, 1.0)?;
                let amp = self.export_arg(&args, 1)?;
                let octaves = self.number_arg(&args, 2, 1.0)?.max(1.0) as usize;
                let amp_mult = self.number_arg(&args, 3, 0.5)?;
                let t = self.number_arg(&args, 4, self.ctx.frame())?;
                let seed_offset = self.rng.as_ref().map_or(0.0, DetRng::seed_offset);
                let base_val = self.ctx.value();
                let res = eval_wiggle(&base_val, freq, &amp, octaves, amp_mult, t, seed_offset);
                Ok(self.intern_value(res))
            }

            // --- Expressions+ Scene & Property Access ---
            "layer" => {
                let query = self.parse_name_or_index_arg(&args)?;
                let lid = self.ctx.layer(&query)?;
                Ok(RuntimeValue::Handle(HandleRef::Layer(lid)))
            }
            "effect" => {
                let name = self.expect_string_arg(&args, 0, "effect")?;
                let eid = self.ctx.effect(None, &name)?;
                Ok(RuntimeValue::Handle(HandleRef::Effect(eid)))
            }
            "object" => {
                let name = self.expect_string_arg(&args, 0, "object")?;
                let oid = self.ctx.object(None, &name)?;
                Ok(RuntimeValue::Handle(HandleRef::Object(oid)))
            }
            "property" => {
                let name = self.expect_string_arg(&args, 0, "property")?;
                let pid = self.ctx.property(None, &name)?;
                Ok(RuntimeValue::Handle(HandleRef::Property(pid)))
            }
            "prop" => {
                let path = self.expect_string_arg(&args, 0, "prop")?;
                let pid = self.ctx.prop(&path)?;
                Ok(RuntimeValue::Handle(HandleRef::Property(pid)))
            }
            "control" => {
                let ctrl_name = self.expect_string_arg(&args, 0, "control")?;
                let stable_id = match args.get(1) {
                    Some(v) if !matches!(v, RuntimeValue::Undefined | RuntimeValue::Null) => {
                        Some(self.runtime_to_js_string(v)?)
                    }
                    _ => None,
                };
                let val = self.ctx.control(&ctrl_name, stable_id.as_deref())?;
                Ok(self.intern_value(val))
            }
            "toWorld" => {
                let pt = match args.first() {
                    Some(v) => self.runtime_to_vec3(v)?,
                    None => [0.0, 0.0, 0.0],
                };
                let out = self.ctx.to_world(None, pt)?;
                Ok(self.alloc_f64_vec(out.to_vec()))
            }
            "fromWorld" => {
                let pt = match args.first() {
                    Some(v) => self.runtime_to_vec3(v)?,
                    None => [0.0, 0.0, 0.0],
                };
                let out = self.ctx.from_world(None, pt)?;
                Ok(self.alloc_f64_vec(out.to_vec()))
            }
            "worldPosition" => {
                let out = self.ctx.world_position(None)?;
                Ok(self.alloc_f64_vec(out.to_vec()))
            }
            "worldRotation" => {
                let out = self.ctx.world_rotation(None)?;
                Ok(self.alloc_f64_vec(out.to_vec()))
            }
            "toScreen" => {
                let pt = match args.first() {
                    Some(v) => self.runtime_to_vec3(v)?,
                    None => [0.0, 0.0, 0.0],
                };
                let out = self.ctx.to_screen(None, pt)?;
                Ok(self.alloc_f64_vec(out.to_vec()))
            }
            "fromScreen" => {
                let pt = match args.first() {
                    Some(v) => self.runtime_to_vec3(v)?,
                    None => [0.0, 0.0, 0.0],
                };
                let out = self.ctx.from_screen(None, pt)?;
                Ok(self.alloc_f64_vec(out.to_vec()))
            }
            _ => Err(ExprError::ReferenceError(name.to_string())),
        }
    }

    fn builtin_clamp(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, ExprError> {
        let Some(first) = args.first() else {
            return Ok(RuntimeValue::Number(f64::NAN));
        };
        let lo_val = args.get(1);
        let hi_val = args.get(2);

        if let Some(v) = self.runtime_as_f64_vec(first)? {
            let lo_vec = match lo_val {
                Some(lv) => self.runtime_as_f64_vec(lv)?,
                None => None,
            };
            let hi_vec = match hi_val {
                Some(hv) => self.runtime_as_f64_vec(hv)?,
                None => None,
            };
            let lo_s = self.number_arg(args, 1, 0.0)?;
            let hi_s = self.number_arg(args, 2, 1.0)?;
            let out = v
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let lo = lo_vec
                        .as_ref()
                        .and_then(|lv| lv.get(i).copied())
                        .unwrap_or(lo_s);
                    let hi = hi_vec
                        .as_ref()
                        .and_then(|hv| hv.get(i).copied())
                        .unwrap_or(hi_s);
                    clamp_f64(x, lo, hi)
                })
                .collect();
            Ok(self.alloc_f64_vec(out))
        } else {
            let x = self.runtime_to_number(first)?;
            let lo = self.number_arg(args, 1, 0.0)?;
            let hi = self.number_arg(args, 2, 1.0)?;
            Ok(RuntimeValue::Number(clamp_f64(x, lo, hi)))
        }
    }

    fn builtin_linear(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, ExprError> {
        if args.len() == 5 {
            if self.config.mode == ExprMode::VanillaCm3 {
                return Err(ExprError::ModeError {
                    mode: self.config.mode,
                    feature: "5-argument linear(t, tMin, tMax, a, b)".to_string(),
                });
            }
            let t = self.number_arg(args, 0, 0.0)?;
            let t_min = self.number_arg(args, 1, 0.0)?;
            let t_max = self.number_arg(args, 2, 1.0)?;
            let a = self.export_arg(args, 3)?;
            let b = self.export_arg(args, 4)?;
            let res = ae_linear(t, t_min, t_max, &a, &b)?;
            return Ok(self.intern_value(res));
        }
        // Vanilla CM3 3-argument form: `linear(a, b, t) = (b - a) * t + a`
        let a = self.export_arg(args, 0)?;
        let b = self.export_arg(args, 1)?;
        let t = self.number_arg(args, 2, 0.0)?;
        let res = lerp_value(&a, &b, t)?;
        Ok(self.intern_value(res))
    }

    fn builtin_ease(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, ExprError> {
        if args.len() == 5 {
            if self.config.mode == ExprMode::VanillaCm3 {
                return Err(ExprError::ModeError {
                    mode: self.config.mode,
                    feature: "5-argument ease(t, tMin, tMax, a, b)".to_string(),
                });
            }
            let t = self.number_arg(args, 0, 0.0)?;
            let t_min = self.number_arg(args, 1, 0.0)?;
            let t_max = self.number_arg(args, 2, 1.0)?;
            let a = self.export_arg(args, 3)?;
            let b = self.export_arg(args, 4)?;
            let res = ae_ease(t, t_min, t_max, &a, &b)?;
            return Ok(self.intern_value(res));
        }
        // Vanilla CM3 form: `ease(t, family = "Quadratic", dir = "InOut")`
        let t = self.number_arg(args, 0, 0.0)?;
        let family = match args.get(1) {
            Some(v) if !matches!(v, RuntimeValue::Undefined) => self.runtime_to_js_string(v)?,
            _ => "Quadratic".to_string(),
        };
        let dir = match args.get(2) {
            Some(v) if !matches!(v, RuntimeValue::Undefined) => self.runtime_to_js_string(v)?,
            _ => "InOut".to_string(),
        };
        Ok(RuntimeValue::Number(cm3_ease(t, &family, &dir)?))
    }

    fn builtin_ae_ease_dir(
        &mut self,
        args: &[RuntimeValue],
        is_in: bool,
    ) -> Result<RuntimeValue, ExprError> {
        if args.len() >= 5 {
            let t = self.number_arg(args, 0, 0.0)?;
            let t_min = self.number_arg(args, 1, 0.0)?;
            let t_max = self.number_arg(args, 2, 1.0)?;
            let a = self.export_arg(args, 3)?;
            let b = self.export_arg(args, 4)?;
            let res = if is_in {
                ae_ease_in(t, t_min, t_max, &a, &b)?
            } else {
                ae_ease_out(t, t_min, t_max, &a, &b)?
            };
            Ok(self.intern_value(res))
        } else {
            let t = clamp_f64(self.number_arg(args, 0, 0.0)?, 0.0, 1.0);
            let s = if is_in {
                t * t * t
            } else {
                let inv = 1.0 - t;
                1.0 - inv * inv * inv
            };
            Ok(RuntimeValue::Number(s))
        }
    }

    fn builtin_random(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, ExprError> {
        // If `seedRandom()` has been invoked in Expressions+, advance the deterministic stream.
        if self.config.mode == ExprMode::ExpressionsPlus && self.rng.is_some() {
            return self.sample_rng_stream(args);
        }

        // In Vanilla CM3 (or before `seedRandom` in Expressions+):
        // - 1-arg numeric `random(seed)` uses exact CM3 deterministic hash.
        // - 0-arg or 2-arg in Expressions+ initializes a default frame-seeded stream.
        if args.len() == 1 {
            if let Some(max_vec) = self.runtime_as_f64_vec(&args[0])?
                && self.config.mode == ExprMode::ExpressionsPlus
            {
                let frame = self.ctx.frame();
                let rng = self
                    .rng
                    .get_or_insert_with(|| DetRng::from_seed(0.0, frame, false));
                let out = max_vec.into_iter().map(|m| rng.next_f64() * m).collect();
                return Ok(self.alloc_f64_vec(out));
            }
            let seed = self.number_arg(args, 0, 0.0)?;
            return Ok(RuntimeValue::Number(cm3_random(seed)));
        }

        if self.config.mode == ExprMode::VanillaCm3 {
            let seed = self.number_arg(args, 0, 0.0)?;
            return Ok(RuntimeValue::Number(cm3_random(seed)));
        }

        let frame = self.ctx.frame();
        self.rng
            .get_or_insert_with(|| DetRng::from_seed(0.0, frame, false));
        self.sample_rng_stream(args)
    }

    fn sample_rng_stream(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, ExprError> {
        match args.len() {
            0 => {
                let val = self.rng.as_mut().map_or(0.0, DetRng::next_f64);
                Ok(RuntimeValue::Number(val))
            }
            1 => {
                if let Some(max_vec) = self.runtime_as_f64_vec(&args[0])? {
                    let rng = self.rng.as_mut().ok_or_else(|| {
                        ExprError::RuntimeError("PRNG stream not initialized".to_string())
                    })?;
                    let out = max_vec.into_iter().map(|m| rng.next_f64() * m).collect();
                    return Ok(self.alloc_f64_vec(out));
                }
                let max_val = self.runtime_to_number(&args[0])?;
                let r = self.rng.as_mut().map_or(0.0, DetRng::next_f64);
                Ok(RuntimeValue::Number(r * max_val))
            }
            _ => {
                let lo_vec = self.runtime_as_f64_vec(&args[0])?;
                let hi_vec = self.runtime_as_f64_vec(&args[1])?;
                if let (Some(lo), Some(hi)) = (lo_vec, hi_vec) {
                    let len = lo.len().max(hi.len());
                    let rng = self.rng.as_mut().ok_or_else(|| {
                        ExprError::RuntimeError("PRNG stream not initialized".to_string())
                    })?;
                    let mut out = Vec::with_capacity(len);
                    for i in 0..len {
                        let a = lo.get(i).copied().unwrap_or(0.0);
                        let b = hi.get(i).copied().unwrap_or(0.0);
                        out.push((b - a) * rng.next_f64() + a);
                    }
                    return Ok(self.alloc_f64_vec(out));
                }
                let lo = self.runtime_to_number(&args[0])?;
                let hi = self.runtime_to_number(&args[1])?;
                let r = self.rng.as_mut().map_or(0.0, DetRng::next_f64);
                Ok(RuntimeValue::Number((hi - lo) * r + lo))
            }
        }
    }

    fn require_vec_arg(
        &mut self,
        args: &[RuntimeValue],
        index: usize,
        fn_name: &str,
    ) -> Result<Vec<f64>, ExprError> {
        let Some(arg) = args.get(index) else {
            return Err(ExprError::TypeError(format!(
                "{fn_name}() missing numeric vector argument at index {index}"
            )));
        };
        self.runtime_as_f64_vec(arg)?.ok_or_else(|| {
            ExprError::TypeError(format!(
                "{fn_name}() expected numeric vector argument at index {index}"
            ))
        })
    }

    fn number_arg(
        &mut self,
        args: &[RuntimeValue],
        index: usize,
        default: f64,
    ) -> Result<f64, ExprError> {
        match args.get(index) {
            Some(v) if !matches!(v, RuntimeValue::Undefined) => self.runtime_to_number(v),
            _ => Ok(default),
        }
    }

    fn export_arg(&mut self, args: &[RuntimeValue], index: usize) -> Result<Value, ExprError> {
        match args.get(index) {
            Some(v) => self.export_value(v, 0),
            None => Ok(Value::Number(0.0)),
        }
    }
}

fn is_catchable_error(err: &ExprError) -> bool {
    !matches!(
        err,
        ExprError::StepLimitExceeded { .. }
            | ExprError::RecursionLimitExceeded { .. }
            | ExprError::Forbidden(_)
            | ExprError::CycleDetected(_)
    )
}

fn normalize_slice_index(idx: f64, len: isize) -> usize {
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

fn is_top_level_math_fn(name: &str) -> bool {
    matches!(
        name,
        "abs"
            | "acos"
            | "acosh"
            | "asin"
            | "asinh"
            | "atan"
            | "atan2"
            | "atanh"
            | "cbrt"
            | "ceil"
            | "clz32"
            | "cos"
            | "cosh"
            | "exp"
            | "expm1"
            | "floor"
            | "fround"
            | "hypot"
            | "imul"
            | "log"
            | "log1p"
            | "log2"
            | "log10"
            | "max"
            | "min"
            | "pow"
            | "round"
            | "sign"
            | "sin"
            | "sinh"
            | "sqrt"
            | "tan"
            | "tanh"
            | "trunc"
    )
}
