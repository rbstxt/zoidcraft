use super::*;

impl<'a, 'ast> Evaluator<'a, 'ast> {
    pub(super) fn intern_value(&mut self, val: Value) -> Result<RuntimeValue, ExprError> {
        match val {
            Value::Undefined => Ok(RuntimeValue::Undefined),
            Value::Null => Ok(RuntimeValue::Null),
            Value::Bool(b) => Ok(RuntimeValue::Bool(b)),
            Value::Number(n) => Ok(RuntimeValue::Number(n)),
            Value::String(s) => self.alloc_string(s),
            Value::Array(items) => {
                self.check_array_alloc(items.len())?;
                let mut converted = Vec::with_capacity(items.len());
                for v in items {
                    converted.push(self.intern_value(v)?);
                }
                let id = self.arrays.len();
                self.arrays.push(converted);
                Ok(RuntimeValue::Array(id))
            }
            Value::Object(map) => {
                self.check_array_alloc(map.len())?;
                let key_bytes: usize = map.keys().map(String::len).sum();
                self.charge_alloc_bytes(key_bytes)?;
                let mut converted = BTreeMap::new();
                for (k, v) in map {
                    converted.insert(k, self.intern_value(v)?);
                }
                let id = self.objects.len();
                self.objects.push(converted);
                Ok(RuntimeValue::Object(id))
            }
        }
    }

    pub(super) fn export_root_value(&mut self, rv: &RuntimeValue) -> Result<Value, ExprError> {
        let mut nodes = 0_usize;
        let mut bytes = 0_usize;
        let mut active_arrays = Vec::new();
        let mut active_objects = Vec::new();
        self.export_value(
            rv,
            0,
            &mut nodes,
            &mut bytes,
            &mut active_arrays,
            &mut active_objects,
        )
    }

    fn export_value(
        &mut self,
        rv: &RuntimeValue,
        depth: usize,
        nodes: &mut usize,
        bytes: &mut usize,
        active_arrays: &mut Vec<usize>,
        active_objects: &mut Vec<usize>,
    ) -> Result<Value, ExprError> {
        self.tick()?;
        if depth > MAX_EXPORT_DEPTH {
            return Err(ExprError::RangeError(
                "circular or overly deep value cannot be returned".to_string(),
            ));
        }
        *nodes = nodes.saturating_add(1);
        if *nodes > MAX_EXPORT_NODES {
            return Err(ExprError::RangeError(format!(
                "returned value exceeds {MAX_EXPORT_NODES} element budget"
            )));
        }
        *bytes = bytes.saturating_add(ARRAY_ELEM_BUDGET_BYTES);
        if *bytes > MAX_EXPORT_BYTES {
            return Err(ExprError::RangeError(format!(
                "returned value exceeds {MAX_EXPORT_BYTES} byte budget"
            )));
        }
        match rv {
            RuntimeValue::Empty | RuntimeValue::Undefined => Ok(Value::Undefined),
            RuntimeValue::Null => Ok(Value::Null),
            RuntimeValue::Bool(b) => Ok(Value::Bool(*b)),
            RuntimeValue::Number(n) => Ok(Value::Number(*n)),
            RuntimeValue::String(s) => {
                self.charge_byte_work(s.len())?;
                *bytes = bytes.saturating_add(s.len());
                if *bytes > MAX_EXPORT_BYTES {
                    return Err(ExprError::RangeError(format!(
                        "returned value exceeds {MAX_EXPORT_BYTES} byte budget"
                    )));
                }
                Ok(Value::String(s.to_string()))
            }
            RuntimeValue::Array(id) => {
                if active_arrays.contains(id) {
                    return Err(ExprError::RangeError(
                        "circular array cannot be exported".to_string(),
                    ));
                }
                let len = self.arrays[*id].len();
                if nodes.saturating_add(len) > MAX_EXPORT_NODES {
                    return Err(ExprError::RangeError(format!(
                        "returned value exceeds {MAX_EXPORT_NODES} element budget"
                    )));
                }
                if bytes.saturating_add(len.saturating_mul(ARRAY_ELEM_BUDGET_BYTES))
                    > MAX_EXPORT_BYTES
                {
                    return Err(ExprError::RangeError(format!(
                        "returned value exceeds {MAX_EXPORT_BYTES} byte budget"
                    )));
                }
                self.charge_work(len)?;
                active_arrays.push(*id);
                let res = (|| -> Result<Value, ExprError> {
                    let mut out = Vec::with_capacity(len);
                    for i in 0..len {
                        let item = self.arrays[*id][i].clone();
                        out.push(self.export_value(
                            &item,
                            depth + 1,
                            nodes,
                            bytes,
                            active_arrays,
                            active_objects,
                        )?);
                    }
                    Ok(Value::Array(out))
                })();
                active_arrays.pop();
                res
            }
            RuntimeValue::Object(id) => {
                if active_objects.contains(id) {
                    return Err(ExprError::RangeError(
                        "circular object cannot be exported".to_string(),
                    ));
                }
                let len = self.objects[*id].len();
                if nodes.saturating_add(len) > MAX_EXPORT_NODES {
                    return Err(ExprError::RangeError(format!(
                        "returned value exceeds {MAX_EXPORT_NODES} element budget"
                    )));
                }
                let key_bytes: usize = self.objects[*id].keys().map(String::len).sum();
                let container_bytes =
                    key_bytes.saturating_add(len.saturating_mul(ARRAY_ELEM_BUDGET_BYTES));
                if bytes.saturating_add(container_bytes) > MAX_EXPORT_BYTES {
                    return Err(ExprError::RangeError(format!(
                        "returned value exceeds {MAX_EXPORT_BYTES} byte budget"
                    )));
                }
                self.charge_work(len)?;
                self.charge_byte_work(key_bytes)?;
                *bytes = bytes.saturating_add(key_bytes);
                active_objects.push(*id);
                let entries: Vec<(String, RuntimeValue)> = self.objects[*id]
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                let res = (|| -> Result<Value, ExprError> {
                    let mut out = BTreeMap::new();
                    for (k, v) in entries {
                        let exported = self.export_value(
                            &v,
                            depth + 1,
                            nodes,
                            bytes,
                            active_arrays,
                            active_objects,
                        )?;
                        out.insert(k, exported);
                    }
                    Ok(Value::Object(out))
                })();
                active_objects.pop();
                res
            }
            RuntimeValue::Handle(HandleRef::Property(prop_id)) => self.read_property_value(prop_id),
            RuntimeValue::Handle(_)
            | RuntimeValue::Closure(_)
            | RuntimeValue::BuiltinFn(_)
            | RuntimeValue::Namespace(_)
            | RuntimeValue::Intrinsic(_) => Ok(Value::Undefined),
        }
    }

    pub(super) fn read_property_value(&self, id: &HandleId) -> Result<Value, ExprError> {
        self.ctx.enter_property(id)?;
        let _guard = PropertyEnterGuard { ctx: self.ctx, id };
        self.ctx.property_value(id)
    }

    pub(super) fn read_property_value_at_time(
        &self,
        id: &HandleId,
        seconds: f64,
    ) -> Result<Value, ExprError> {
        self.ctx.enter_property(id)?;
        let _guard = PropertyEnterGuard { ctx: self.ctx, id };
        self.ctx.property_value_at_time(id, seconds)
    }

    pub(super) fn read_property_velocity(
        &self,
        id: &HandleId,
        at_time: f64,
    ) -> Result<Value, ExprError> {
        self.ctx.enter_property(id)?;
        let _guard = PropertyEnterGuard { ctx: self.ctx, id };
        self.ctx.property_velocity(id, at_time)
    }

    pub(super) fn read_property_speed(
        &self,
        id: &HandleId,
        at_time: f64,
    ) -> Result<f64, ExprError> {
        self.ctx.enter_property(id)?;
        let _guard = PropertyEnterGuard { ctx: self.ctx, id };
        self.ctx.property_speed(id, at_time)
    }

    /// Unwraps a `Property` handle to its current evaluated value (`valueOf` semantics).
    pub(super) fn deref_property_handle(
        &mut self,
        rv: RuntimeValue,
    ) -> Result<RuntimeValue, ExprError> {
        if let RuntimeValue::Handle(HandleRef::Property(prop_id)) = rv {
            let val = self.read_property_value(&prop_id)?;
            self.intern_value(val)
        } else {
            Ok(rv)
        }
    }

    pub(super) fn runtime_as_f64_vec(
        &mut self,
        rv: &RuntimeValue,
    ) -> Result<Option<Vec<f64>>, ExprError> {
        let rv = self.deref_property_handle(rv.clone())?;
        let RuntimeValue::Array(id) = rv else {
            return Ok(None);
        };
        let len = self.arrays[id].len();
        self.charge_work(len)?;
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

    pub(super) fn runtime_to_bool(&mut self, rv: &RuntimeValue) -> Result<bool, ExprError> {
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
            | RuntimeValue::Intrinsic(_)
            | RuntimeValue::Handle(_) => true,
        })
    }

    pub(super) fn runtime_to_number(&mut self, rv: &RuntimeValue) -> Result<f64, ExprError> {
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
                self.charge_byte_work(s.len())?;
                let t = if s.is_ascii {
                    let bytes = s.as_bytes();
                    let is_js_ascii_ws =
                        |b: u8| matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C);
                    let start = bytes
                        .iter()
                        .position(|&b| !is_js_ascii_ws(b))
                        .unwrap_or(bytes.len());
                    let end = bytes[start..]
                        .iter()
                        .rposition(|&b| !is_js_ascii_ws(b))
                        .map_or(start, |i| start + i + 1);
                    &s.as_str()[start..end]
                } else {
                    s.trim()
                };
                if t.is_empty() {
                    0.0
                } else {
                    t.parse::<f64>().unwrap_or(f64::NAN)
                }
            }
            _ => f64::NAN,
        })
    }

    pub(super) fn runtime_to_js_string(&mut self, rv: &RuntimeValue) -> Result<String, ExprError> {
        self.tick()?;
        let rv = self.deref_property_handle(rv.clone())?;
        Ok(match rv {
            RuntimeValue::Empty | RuntimeValue::Undefined => "undefined".to_string(),
            RuntimeValue::Null => "null".to_string(),
            RuntimeValue::Bool(b) => b.to_string(),
            RuntimeValue::Number(n) => format_js_number(n),
            RuntimeValue::String(s) => {
                self.charge_byte_work(s.len())?;
                s.to_string()
            }
            RuntimeValue::Array(id) => self.join_array_to_string(id, ",")?,
            RuntimeValue::Object(_) | RuntimeValue::Namespace(_) | RuntimeValue::Handle(_) => {
                "[object Object]".to_string()
            }
            RuntimeValue::Intrinsic(IntrinsicSentinel::FunctionCtor) => "[Function]".to_string(),
            RuntimeValue::Intrinsic(_) => "[object Object]".to_string(),
            RuntimeValue::Closure(_) | RuntimeValue::BuiltinFn(_) => "[Function]".to_string(),
        })
    }

    pub(super) fn join_array_to_string(
        &mut self,
        id: usize,
        sep: &str,
    ) -> Result<String, ExprError> {
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
            let mut total_bytes = 0_usize;
            for (idx, item) in items.iter().enumerate() {
                self.tick()?;
                let piece = match item {
                    RuntimeValue::Null | RuntimeValue::Undefined => String::new(),
                    other => self.runtime_to_js_string(other)?,
                };
                let add = piece
                    .len()
                    .checked_add(if idx > 0 { sep.len() } else { 0 })
                    .ok_or_else(|| ExprError::RangeError("joined string overflow".to_string()))?;
                total_bytes = total_bytes
                    .checked_add(add)
                    .ok_or_else(|| ExprError::RangeError("joined string overflow".to_string()))?;
                if total_bytes > MAX_STRING_BYTES {
                    return Err(ExprError::RangeError(format!(
                        "joined string length exceeds {MAX_STRING_BYTES} byte limit"
                    )));
                }
                parts.push(piece);
            }
            self.check_string_alloc(total_bytes)?;
            Ok(parts.join(sep))
        })();
        self.stringifying_arrays.pop();
        res
    }

    pub(super) fn runtime_to_vec3(&mut self, rv: &RuntimeValue) -> Result<[f64; 3], ExprError> {
        let v = self.runtime_as_f64_vec(rv)?.ok_or_else(|| {
            ExprError::TypeError("expected 3D numeric vector [x, y, z]".to_string())
        })?;
        Ok([
            v.first().copied().unwrap_or(0.0),
            v.get(1).copied().unwrap_or(0.0),
            v.get(2).copied().unwrap_or(0.0),
        ])
    }

    pub(super) fn alloc_array(&mut self, elems: Vec<RuntimeValue>) -> RuntimeValue {
        let id = self.arrays.len();
        self.arrays.push(elems);
        RuntimeValue::Array(id)
    }

    pub(super) fn alloc_f64_vec(&mut self, vals: Vec<f64>) -> Result<RuntimeValue, ExprError> {
        self.check_array_alloc(vals.len())?;
        let elems = vals.into_iter().map(RuntimeValue::Number).collect();
        Ok(self.alloc_array(elems))
    }

    pub(super) fn alloc_object(&mut self, map: BTreeMap<String, RuntimeValue>) -> RuntimeValue {
        let id = self.objects.len();
        self.objects.push(map);
        RuntimeValue::Object(id)
    }

    pub(super) fn keyframe_to_object(
        &mut self,
        kf: KeyframeRecord,
    ) -> Result<RuntimeValue, ExprError> {
        self.check_array_alloc(3)?;
        self.charge_alloc_bytes("index".len() + "time".len() + "value".len())?;
        let mut map = BTreeMap::new();
        map.insert("index".to_string(), RuntimeValue::Number(kf.index as f64));
        map.insert("time".to_string(), RuntimeValue::Number(kf.time));
        let val = self.intern_value(kf.value)?;
        map.insert("value".to_string(), val);
        Ok(self.alloc_object(map))
    }

    // -----------------------------------------------------------------------
    // Hoisting & Pattern Binding
    // -----------------------------------------------------------------------

    pub(super) fn hoist_decls(&mut self, stmts: &'ast [Stmt], env: usize) -> Result<(), ExprError> {
        let fn_scope = self.function_scope_of(env);
        for stmt in stmts {
            match stmt {
                Stmt::VarDecl {
                    kind: VarKind::Var,
                    decls,
                } => {
                    for d in decls {
                        self.hoist_pattern_var(&d.pattern, fn_scope)?;
                    }
                }
                Stmt::FunctionDecl(f) => {
                    let cid = self.alloc_closure(Closure {
                        params: &f.params,
                        body: ClosureBody::Block(&f.body),
                        captured_env: fn_scope,
                        name: None,
                    })?;
                    self.define_binding(
                        fn_scope,
                        f.name.clone(),
                        RuntimeValue::Closure(cid),
                        VarKind::Var,
                    )?;
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
                            self.hoist_pattern_var(&d.pattern, fn_scope)?;
                        }
                    }
                    self.hoist_decls(std::slice::from_ref(&f.body), fn_scope)?;
                }
                Stmt::ForOf(f) => {
                    if f.kind == Some(VarKind::Var) {
                        self.hoist_pattern_var(&f.target, fn_scope)?;
                    }
                    self.hoist_decls(std::slice::from_ref(&f.body), fn_scope)?;
                }
                Stmt::ForIn(f) => {
                    if f.kind == Some(VarKind::Var) {
                        self.hoist_pattern_var(&f.target, fn_scope)?;
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

    fn hoist_pattern_var(&mut self, pattern: &Pattern, fn_scope: usize) -> Result<(), ExprError> {
        match pattern {
            Pattern::Ident(name) => {
                if !self.envs[fn_scope].bindings.contains_key(name) {
                    self.charge_alloc_bytes(ARRAY_ELEM_BUDGET_BYTES.saturating_add(name.len()))?;
                    self.envs[fn_scope].bindings.insert(
                        name.clone(),
                        Binding {
                            value: RuntimeValue::Undefined,
                            mutable: true,
                            is_var: true,
                        },
                    );
                }
            }
            Pattern::Array(elems) => {
                for e in elems {
                    match e {
                        ArrayPatternElem::Hole => {}
                        ArrayPatternElem::Binding { pattern, .. } => {
                            self.hoist_pattern_var(pattern, fn_scope)?;
                        }
                        ArrayPatternElem::Rest(name) => {
                            self.hoist_pattern_var(&Pattern::Ident(name.clone()), fn_scope)?;
                        }
                    }
                }
            }
            Pattern::Object(props) => {
                for p in props {
                    match p {
                        ObjectPatternProp::Prop { pattern, .. } => {
                            self.hoist_pattern_var(pattern, fn_scope)?;
                        }
                        ObjectPatternProp::Rest(name) => {
                            self.hoist_pattern_var(&Pattern::Ident(name.clone()), fn_scope)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
