use super::*;

impl<'a, 'ast> Evaluator<'a, 'ast> {
    pub(super) fn eval_expr(
        &mut self,
        expr: &'ast Expr,
        env: usize,
    ) -> Result<RuntimeValue, ExprError> {
        self.tick()?;
        match expr {
            Expr::Literal(lit) => Ok(match lit {
                Literal::Undefined => RuntimeValue::Undefined,
                Literal::Null => RuntimeValue::Null,
                Literal::Bool(b) => RuntimeValue::Bool(*b),
                Literal::Number(n) => RuntimeValue::Number(*n),
                Literal::String(s) => self.intern_literal_string(s)?,
            }),
            Expr::Ident(name) => self.resolve_ident(name, env),
            Expr::Template(parts) => {
                let mut out = String::new();
                for part in parts {
                    let piece: std::borrow::Cow<'_, str> = match part {
                        TemplatePart::Text(t) => std::borrow::Cow::Borrowed(t.as_str()),
                        TemplatePart::Expr(e) => {
                            let v = self.eval_expr(e, env)?;
                            std::borrow::Cow::Owned(self.runtime_to_js_string(&v)?)
                        }
                    };
                    let new_len = out.len().checked_add(piece.len()).ok_or_else(|| {
                        ExprError::RangeError("template string overflow".to_string())
                    })?;
                    if new_len > MAX_STRING_BYTES {
                        return Err(ExprError::RangeError(format!(
                            "template string length exceeds {MAX_STRING_BYTES} byte limit"
                        )));
                    }
                    self.charge_alloc_bytes(piece.len())?;
                    out.push_str(&piece);
                }
                Ok(RuntimeValue::String(JsStr::from(out)))
            }
            Expr::Array(elems) => {
                let mut items = Vec::with_capacity(elems.len());
                for elem in elems {
                    match elem {
                        ArrayElem::Expr(e) => {
                            let v = self.eval_expr(e, env)?;
                            if items.len() >= MAX_ARRAY_LEN {
                                return Err(ExprError::RangeError(format!(
                                    "array length exceeds {MAX_ARRAY_LEN} limit"
                                )));
                            }
                            self.charge_alloc_bytes(ARRAY_ELEM_BUDGET_BYTES)?;
                            items.push(v);
                        }
                        ArrayElem::Spread(e) => {
                            let val = self.eval_expr(e, env)?;
                            let val = self.deref_property_handle(val)?;
                            match val {
                                RuntimeValue::Array(id) => {
                                    let add = self.arrays[id].len();
                                    let next_len =
                                        items.len().checked_add(add).ok_or_else(|| {
                                            ExprError::RangeError(
                                                "array length overflow".to_string(),
                                            )
                                        })?;
                                    if next_len > MAX_ARRAY_LEN {
                                        return Err(ExprError::RangeError(format!(
                                            "array length {next_len} exceeds {MAX_ARRAY_LEN} limit"
                                        )));
                                    }
                                    self.check_array_alloc(add)?;
                                    self.charge_work(add)?;
                                    items.extend_from_slice(&self.arrays[id]);
                                }
                                RuntimeValue::String(s) => {
                                    let add = self.string_char_len(&s)?;
                                    let next_len =
                                        items.len().checked_add(add).ok_or_else(|| {
                                            ExprError::RangeError(
                                                "array length overflow".to_string(),
                                            )
                                        })?;
                                    if next_len > MAX_ARRAY_LEN {
                                        return Err(ExprError::RangeError(format!(
                                            "array length {next_len} exceeds {MAX_ARRAY_LEN} limit"
                                        )));
                                    }
                                    self.check_array_alloc(add)?;
                                    self.check_string_alloc(s.len())?;
                                    self.charge_byte_work(s.len())?;
                                    items.extend(
                                        s.chars().map(|c| {
                                            RuntimeValue::String(JsStr::from(c.to_string()))
                                        }),
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
                enum KeyRef<'k> {
                    Static(&'k str),
                    Owned(String),
                }
                impl KeyRef<'_> {
                    fn as_str(&self) -> &str {
                        match self {
                            Self::Static(s) => s,
                            Self::Owned(s) => s.as_str(),
                        }
                    }
                }
                let mut map = BTreeMap::new();
                for prop in props {
                    match prop {
                        ObjectProp::KeyValue { key, value } => {
                            let k = match key {
                                ObjectKey::Static(s) => KeyRef::Static(s.as_str()),
                                ObjectKey::Computed(e) => {
                                    let kv = self.eval_expr(e, env)?;
                                    let owned = self.runtime_to_js_string(&kv)?;
                                    self.check_string_alloc(owned.len())?;
                                    KeyRef::Owned(owned)
                                }
                            };
                            let v = self.eval_expr(value, env)?;
                            if !map.contains_key(k.as_str()) {
                                if map.len() >= MAX_ARRAY_LEN {
                                    return Err(ExprError::RangeError(format!(
                                        "object property count exceeds {MAX_ARRAY_LEN} limit"
                                    )));
                                }
                                let extra_bytes = match &k {
                                    KeyRef::Static(s) => {
                                        s.len().saturating_add(ARRAY_ELEM_BUDGET_BYTES)
                                    }
                                    KeyRef::Owned(_) => ARRAY_ELEM_BUDGET_BYTES,
                                };
                                self.charge_alloc_bytes(extra_bytes)?;
                            }
                            let owned_key = match k {
                                KeyRef::Static(s) => s.to_owned(),
                                KeyRef::Owned(s) => s,
                            };
                            map.insert(owned_key, v);
                        }
                        ObjectProp::Spread(e) => {
                            let val = self.eval_expr(e, env)?;
                            let val = self.deref_property_handle(val)?;
                            if let RuntimeValue::Object(id) = val {
                                let add_len = self.objects[id].len();
                                let next_len = map.len().saturating_add(add_len);
                                if next_len > MAX_ARRAY_LEN {
                                    return Err(ExprError::RangeError(format!(
                                        "object property count {next_len} exceeds {MAX_ARRAY_LEN} limit"
                                    )));
                                }
                                let key_bytes: usize =
                                    self.objects[id].keys().map(String::len).sum();
                                self.check_array_alloc(add_len)?;
                                self.charge_alloc_bytes(key_bytes)?;
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
                let body = match &f.body {
                    FunctionBody::Expr(expr) => ClosureBody::Expr(expr.as_ref()),
                    FunctionBody::Block(stmts) => ClosureBody::Block(stmts.as_slice()),
                };
                let cid = self.alloc_closure(Closure {
                    params: &f.params,
                    body,
                    captured_env: env,
                    name: f.name.as_deref(),
                })?;
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
            "methods" => return Ok(RuntimeValue::Namespace(NamespaceKind::Methods)),
            "properties" => {
                if let Some(p) = &self.cached_properties {
                    return Ok(p.clone());
                }
                let props = self.ctx.properties();
                let interned = self.intern_value(Value::Object(props))?;
                self.cached_properties = Some(interned.clone());
                return Ok(interned);
            }
            "Math" => return Ok(RuntimeValue::Namespace(NamespaceKind::Math)),
            "isNaN" | "isFinite" | "parseInt" | "parseFloat" => {
                if let Some(s) = static_global_builtin_name(name) {
                    return Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(s)));
                }
                return self.alloc_builtin_fn(name.to_string());
            }
            // Exact 17 Vanilla CM3 built-in functions
            "add" | "sub" | "mul" | "div" | "clamp" | "dot" | "cross" | "normalize" | "length"
            | "wave" | "lerp" | "linear" | "catmullRom" | "ease" | "random" | "gaussRandom"
            | "shake" => {
                if let Some(s) = static_global_builtin_name(name) {
                    return Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(s)));
                }
                return self.alloc_builtin_fn(name.to_string());
            }
            _ => {}
        }

        // Top-level Math constants & functions (CM3 compatibility + Expressions+)
        if let Some(c) = math_constant(name) {
            return Ok(RuntimeValue::Number(c));
        }
        if let Some(math_fn) = static_math_builtin_name(name) {
            return Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(math_fn)));
        }

        // Expressions+ globals & handles
        if self.config.mode == ExprMode::ExpressionsPlus {
            match name {
                "fps" => return Ok(RuntimeValue::Number(self.ctx.fps())),
                "value" => {
                    if let Some(v) = &self.cached_value {
                        return Ok(v.clone());
                    }
                    let v = self.ctx.value();
                    let interned = self.intern_value(v)?;
                    self.cached_value = Some(interned.clone());
                    return Ok(interned);
                }
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
                    if let Some(s) = static_global_builtin_name(name) {
                        return Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(s)));
                    }
                    return self.alloc_builtin_fn(name.to_string());
                }
                _ => {}
            }
        }

        Err(ExprError::ReferenceError(name.to_string()))
    }

    fn eval_unary(
        &mut self,
        op: UnaryOp,
        expr: &'ast Expr,
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
                    return Ok(RuntimeValue::String(JsStr::from(self.typeof_value(&val)?)));
                }
                let val = self.eval_expr(expr, env)?;
                Ok(RuntimeValue::String(JsStr::from(self.typeof_value(&val)?)))
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
                    return self.alloc_f64_vec(vec.into_iter().map(|x| -x).collect());
                }
                Ok(RuntimeValue::Number(-self.runtime_to_number(&v)?))
            }
            UnaryOp::Pos => {
                let v = self.eval_expr(expr, env)?;
                if let Some(vec) = self.runtime_as_f64_vec(&v)? {
                    return self.alloc_f64_vec(vec);
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
            RuntimeValue::Namespace(
                NamespaceKind::Math | NamespaceKind::Json | NamespaceKind::Methods,
            ) => "object",
            RuntimeValue::Namespace(_) => "function",
            RuntimeValue::Intrinsic(IntrinsicSentinel::FunctionCtor) => "function",
            RuntimeValue::Intrinsic(_) => "object",
            RuntimeValue::Handle(_) => "object",
        })
    }

    fn eval_update(
        &mut self,
        op: UpdateOp,
        prefix: bool,
        target: &'ast AssignTarget,
        env: usize,
    ) -> Result<RuntimeValue, ExprError> {
        let delta = match op {
            UpdateOp::Inc => 1.0,
            UpdateOp::Dec => -1.0,
        };
        let resolved = self.resolve_target(target, env)?;
        let old_val = self.read_resolved_target(&resolved, env)?;
        let old_num = self.runtime_to_number(&old_val)?;
        let new_num = old_num + delta;
        self.write_resolved_target(&resolved, RuntimeValue::Number(new_num), env)?;
        Ok(RuntimeValue::Number(if prefix { new_num } else { old_num }))
    }

    fn eval_assign(
        &mut self,
        op: AssignOp,
        target: &'ast AssignTarget,
        value_expr: &'ast Expr,
        env: usize,
    ) -> Result<RuntimeValue, ExprError> {
        let resolved = self.resolve_target(target, env)?;
        let final_val = match op {
            AssignOp::Assign => self.eval_expr(value_expr, env)?,
            _ => {
                let lhs = self.read_resolved_target(&resolved, env)?;
                let rhs = self.eval_expr(value_expr, env)?;
                let bin_op = match op {
                    AssignOp::AddAssign => BinaryOp::Add,
                    AssignOp::SubAssign => BinaryOp::Sub,
                    AssignOp::MulAssign => BinaryOp::Mul,
                    AssignOp::DivAssign => BinaryOp::Div,
                    AssignOp::RemAssign => BinaryOp::Rem,
                    AssignOp::PowAssign => BinaryOp::Pow,
                    AssignOp::Assign => unreachable!(),
                };
                self.eval_binary(bin_op, lhs, rhs)?
            }
        };
        self.write_resolved_target(&resolved, final_val.clone(), env)?;
        Ok(final_val)
    }

    fn resolve_target(
        &mut self,
        target: &'ast AssignTarget,
        env: usize,
    ) -> Result<ResolvedTarget<'ast>, ExprError> {
        match target {
            AssignTarget::Ident(name) => Ok(ResolvedTarget::Ident(name.as_str())),
            AssignTarget::Member { object, prop } => {
                let object = self.eval_expr(object, env)?;
                Ok(ResolvedTarget::Member {
                    object,
                    prop: prop.as_str(),
                })
            }
            AssignTarget::Index { object, index } => {
                let object = self.eval_expr(object, env)?;
                let index = self.eval_expr(index, env)?;
                Ok(ResolvedTarget::Index { object, index })
            }
        }
    }

    fn read_resolved_target(
        &mut self,
        target: &ResolvedTarget<'ast>,
        env: usize,
    ) -> Result<RuntimeValue, ExprError> {
        match target {
            ResolvedTarget::Ident(name) => self.resolve_ident(name, env),
            ResolvedTarget::Member { object, prop } => self.get_property(object, prop),
            ResolvedTarget::Index { object, index } => self.get_computed(object, index),
        }
    }

    fn write_resolved_target(
        &mut self,
        target: &ResolvedTarget<'ast>,
        val: RuntimeValue,
        env: usize,
    ) -> Result<(), ExprError> {
        match target {
            ResolvedTarget::Ident(name) => self.assign_local(env, name, val),
            ResolvedTarget::Member { object, prop } => match object {
                RuntimeValue::Object(id) => {
                    let id = *id;
                    if !self.objects[id].contains_key(*prop) {
                        if self.objects[id].len() >= MAX_ARRAY_LEN {
                            return Err(ExprError::RangeError(format!(
                                "object property count exceeds {MAX_ARRAY_LEN} limit"
                            )));
                        }
                        self.charge_alloc_bytes(
                            prop.len().saturating_add(ARRAY_ELEM_BUDGET_BYTES),
                        )?;
                    }
                    self.objects[id].insert((*prop).to_string(), val);
                    Ok(())
                }
                RuntimeValue::Handle(_)
                | RuntimeValue::Namespace(_)
                | RuntimeValue::Intrinsic(_) => Err(ExprError::TypeError(format!(
                    "cannot mutate read-only property '{prop}'"
                ))),
                _ => Err(ExprError::TypeError(format!(
                    "cannot set property '{prop}' on non-object"
                ))),
            },
            ResolvedTarget::Index { object, index } => match object {
                RuntimeValue::Array(id) => {
                    let id = *id;
                    let i = self.runtime_to_number(index)?;
                    if i < 0.0 || i.fract() != 0.0 || !i.is_finite() {
                        return Err(ExprError::RangeError("invalid array index".to_string()));
                    }
                    let ui = i as usize;
                    if ui >= MAX_ARRAY_LEN {
                        return Err(ExprError::RangeError(format!(
                            "array index {ui} exceeds {MAX_ARRAY_LEN} limit"
                        )));
                    }
                    let cur_len = self.arrays[id].len();
                    if ui >= cur_len {
                        let added = ui + 1 - cur_len;
                        self.check_array_alloc(added)?;
                        self.arrays[id].resize(ui + 1, RuntimeValue::Undefined);
                    }
                    self.arrays[id][ui] = val;
                    Ok(())
                }
                RuntimeValue::Object(id) => {
                    let id = *id;
                    let key = self.runtime_to_js_string(index)?;
                    if !self.objects[id].contains_key(&key) {
                        if self.objects[id].len() >= MAX_ARRAY_LEN {
                            return Err(ExprError::RangeError(format!(
                                "object property count exceeds {MAX_ARRAY_LEN} limit"
                            )));
                        }
                        self.charge_alloc_bytes(key.len().saturating_add(ARRAY_ELEM_BUDGET_BYTES))?;
                    }
                    self.objects[id].insert(key, val);
                    Ok(())
                }
                _ => Err(ExprError::TypeError(
                    "cannot assign indexed property on non-object".to_string(),
                )),
            },
        }
    }
}
