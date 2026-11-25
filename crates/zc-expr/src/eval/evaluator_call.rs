use super::*;

impl<'a, 'ast> Evaluator<'a, 'ast> {
    pub(super) fn eval_call_args(
        &mut self,
        args: &'ast [CallArg],
        env: usize,
    ) -> Result<Vec<RuntimeValue>, ExprError> {
        let mut out = Vec::with_capacity(args.len());
        for arg in args {
            match arg {
                CallArg::Expr(e) => {
                    if out.len() >= MAX_ARRAY_LEN {
                        return Err(ExprError::RangeError(format!(
                            "call argument count exceeds {MAX_ARRAY_LEN} limit"
                        )));
                    }
                    out.push(self.eval_expr(e, env)?);
                }
                CallArg::Spread(e) => {
                    let v = self.eval_expr(e, env)?;
                    let v = self.deref_property_handle(v)?;
                    match v {
                        RuntimeValue::Array(id) => {
                            let add = self.arrays[id].len();
                            let next_len = out.len().checked_add(add).ok_or_else(|| {
                                ExprError::RangeError("call argument count overflow".to_string())
                            })?;
                            if next_len > MAX_ARRAY_LEN {
                                return Err(ExprError::RangeError(format!(
                                    "call argument count {next_len} exceeds {MAX_ARRAY_LEN} limit"
                                )));
                            }
                            self.check_array_alloc(add)?;
                            self.charge_work(add)?;
                            out.extend_from_slice(&self.arrays[id]);
                        }
                        RuntimeValue::String(s) => {
                            let add = self.string_char_len(&s)?;
                            let next_len = out.len().checked_add(add).ok_or_else(|| {
                                ExprError::RangeError("call argument count overflow".to_string())
                            })?;
                            if next_len > MAX_ARRAY_LEN {
                                return Err(ExprError::RangeError(format!(
                                    "call argument count {next_len} exceeds {MAX_ARRAY_LEN} limit"
                                )));
                            }
                            self.check_array_alloc(add)?;
                            self.check_string_alloc(s.len())?;
                            self.charge_byte_work(s.len())?;
                            out.extend(
                                s.chars()
                                    .map(|c| RuntimeValue::String(JsStr::from(c.to_string()))),
                            );
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

    pub(super) fn eval_call_expr(
        &mut self,
        callee: &'ast Expr,
        args: &'ast [CallArg],
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

    pub(super) fn call_method_on(
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

        // Guard Function constructor invocations (`sin.constructor(...)`, `F.call(...)`, etc.).
        if method == "constructor" {
            let ctor = Self::resolve_meta_property(&receiver, "constructor");
            return self.call_runtime_value(ctor, args);
        }
        if matches!(
            receiver,
            RuntimeValue::Intrinsic(IntrinsicSentinel::FunctionCtor)
        ) && matches!(method, "call" | "apply" | "bind")
        {
            return Err(ExprError::Forbidden(
                "Expression cannot call Function constructors or eval".to_string(),
            ));
        }

        // Namespaces (`Math.*`, `methods.*`, `Array.*`, `Number.*`, `Object.*`, `JSON.*`, `String.*`)
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

    pub(super) fn call_runtime_value(
        &mut self,
        callee: RuntimeValue,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        self.tick()?;
        match callee {
            RuntimeValue::Closure(cid) => self.call_closure(cid, args),
            RuntimeValue::BuiltinFn(name) => self.call_builtin(name.as_str(), args),
            RuntimeValue::Intrinsic(IntrinsicSentinel::FunctionCtor) => Err(ExprError::Forbidden(
                "Expression cannot call Function constructors or eval".to_string(),
            )),
            RuntimeValue::Handle(
                handle @ (HandleRef::Effect(_)
                | HandleRef::Object(_)
                | HandleRef::Layer(_)
                | HandleRef::Clip(_)
                | HandleRef::Sequence),
            ) => self.call_object_handle(&handle, &args),
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
                self.alloc_string(s)
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
                {
                    if n < 0.0 || !n.is_finite() || n.fract() != 0.0 || n > (MAX_ARRAY_LEN as f64) {
                        return Err(ExprError::RangeError("invalid array length".to_string()));
                    }
                    let count = n as usize;
                    self.check_array_alloc(count)?;
                    return Ok(self.alloc_array(vec![RuntimeValue::Undefined; count]));
                }
                self.check_array_alloc(args.len())?;
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
        let closure = self.closures[cid];

        let call_env = self.push_fn_env(closure.captured_env)?;
        if let Some(name) = closure.name {
            self.define_binding(
                call_env,
                name.to_string(),
                RuntimeValue::Closure(cid),
                VarKind::Const,
            )?;
        }

        for (i, param) in closure.params.iter().enumerate() {
            if param.is_rest {
                let rest_len = args.len().saturating_sub(i);
                self.check_array_alloc(rest_len)?;
                self.charge_work(rest_len)?;
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

        match closure.body {
            ClosureBody::Expr(expr) => self.eval_expr(expr, call_env),
            ClosureBody::Block(stmts) => {
                self.hoist_decls(stmts, call_env)?;
                match self.exec_stmt_list(stmts, call_env)? {
                    Flow::Normal(_) => Ok(RuntimeValue::Undefined),
                    Flow::Return(v) => Ok(v),
                    Flow::Throw(v) => {
                        self.pending_throw = Some(v);
                        Err(ExprError::Thrown(String::new()))
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

    pub(super) fn call_namespace_method(
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
                let result = if args.len() <= 4 {
                    let mut stack_nums = [0.0_f64; 4];
                    for (slot, a) in stack_nums.iter_mut().zip(args.iter()) {
                        *slot = self.runtime_to_number(a)?;
                    }
                    eval_math_fn(method, &stack_nums[..args.len()])?
                } else {
                    let mut nums = Vec::with_capacity(args.len());
                    for a in &args {
                        nums.push(self.runtime_to_number(a)?);
                    }
                    eval_math_fn(method, &nums)?
                };
                match result {
                    Some(v) => Ok(RuntimeValue::Number(v)),
                    None => Err(ExprError::TypeError(format!(
                        "Math.{} is not a function",
                        truncate_diag(method)
                    ))),
                }
            }
            NamespaceKind::Methods => {
                if is_methods_member(method, self.config.mode) {
                    self.call_builtin(method, args)
                } else {
                    Err(ExprError::TypeError(format!(
                        "methods.{} is not a function",
                        truncate_diag(method)
                    )))
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
                        RuntimeValue::Array(id) => {
                            let len = self.arrays[id].len();
                            self.check_array_alloc(len)?;
                            self.charge_work(len)?;
                            self.arrays[id].clone()
                        }
                        RuntimeValue::String(s) => {
                            let count = self.string_char_len(&s)?;
                            self.check_array_alloc(count)?;
                            self.check_string_alloc(s.len())?;
                            self.charge_byte_work(s.len())?;
                            s.chars()
                                .map(|c| RuntimeValue::String(JsStr::from(c.to_string())))
                                .collect()
                        }
                        _ => Vec::new(),
                    };
                    Ok(self.alloc_array(items))
                }
                _ => Err(ExprError::TypeError(format!(
                    "Array.{} is not a function",
                    truncate_diag(method)
                ))),
            },
            NamespaceKind::Number => match method {
                "isFinite" => {
                    let Some(v) = args.first() else {
                        return Ok(RuntimeValue::Bool(false));
                    };
                    let d = self.deref_property_handle(v.clone())?;
                    Ok(RuntimeValue::Bool(
                        matches!(d, RuntimeValue::Number(n) if n.is_finite()),
                    ))
                }
                "isNaN" => {
                    let Some(v) = args.first() else {
                        return Ok(RuntimeValue::Bool(false));
                    };
                    let d = self.deref_property_handle(v.clone())?;
                    Ok(RuntimeValue::Bool(
                        matches!(d, RuntimeValue::Number(n) if n.is_nan()),
                    ))
                }
                "isInteger" => {
                    let Some(v) = args.first() else {
                        return Ok(RuntimeValue::Bool(false));
                    };
                    let d = self.deref_property_handle(v.clone())?;
                    Ok(RuntimeValue::Bool(
                        matches!(d, RuntimeValue::Number(n) if n.is_finite() && n.fract() == 0.0),
                    ))
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
                    "Number.{} is not a function",
                    truncate_diag(method)
                ))),
            },
            NamespaceKind::String => match method {
                "fromCharCode" => {
                    self.check_string_alloc(args.len().saturating_mul(4))?;
                    let mut out = String::new();
                    for a in &args {
                        let code = to_u32(self.runtime_to_number(a)?) & 0xFFFF;
                        if let Some(ch) = char::from_u32(code) {
                            out.push(ch);
                        }
                    }
                    Ok(RuntimeValue::String(JsStr::from(out)))
                }
                _ => Err(ExprError::TypeError(format!(
                    "String.{} is not a function",
                    truncate_diag(method)
                ))),
            },
            NamespaceKind::Object => match method {
                "keys" => {
                    let Some(v) = args.first() else {
                        return Ok(self.alloc_array(Vec::new()));
                    };
                    let v = self.deref_property_handle(v.clone())?;
                    let keys: Vec<RuntimeValue> = match v {
                        RuntimeValue::Object(id) => {
                            self.check_array_alloc(self.objects[id].len())?;
                            let key_bytes: usize = self.objects[id].keys().map(String::len).sum();
                            self.charge_alloc_bytes(key_bytes)?;
                            self.objects[id]
                                .keys()
                                .map(|k| RuntimeValue::String(JsStr::from(k.as_str())))
                                .collect()
                        }
                        RuntimeValue::Array(id) => {
                            let n = self.arrays[id].len();
                            self.check_array_alloc(n)?;
                            self.charge_alloc_bytes(n.saturating_mul(8))?;
                            (0..n)
                                .map(|i| RuntimeValue::String(JsStr::from(i.to_string())))
                                .collect()
                        }
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
                        RuntimeValue::Object(id) => {
                            self.check_array_alloc(self.objects[id].len())?;
                            self.objects[id].values().cloned().collect()
                        }
                        RuntimeValue::Array(id) => {
                            self.check_array_alloc(self.arrays[id].len())?;
                            self.arrays[id].clone()
                        }
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
                        RuntimeValue::Object(id) => {
                            let n = self.objects[id].len();
                            self.check_array_alloc(n)?;
                            let key_bytes: usize = self.objects[id].keys().map(String::len).sum();
                            self.charge_alloc_bytes(
                                n.saturating_mul(2 * ARRAY_ELEM_BUDGET_BYTES)
                                    .saturating_add(key_bytes),
                            )?;
                            self.objects[id]
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect()
                        }
                        RuntimeValue::Array(id) => {
                            let n = self.arrays[id].len();
                            self.check_array_alloc(n)?;
                            self.charge_alloc_bytes(
                                n.saturating_mul(2 * ARRAY_ELEM_BUDGET_BYTES + 8),
                            )?;
                            self.arrays[id]
                                .iter()
                                .enumerate()
                                .map(|(i, v)| (i.to_string(), v.clone()))
                                .collect()
                        }
                        _ => Vec::new(),
                    };
                    let mut out = Vec::with_capacity(pairs.len());
                    for (k, val) in pairs {
                        out.push(self.alloc_array(vec![RuntimeValue::String(JsStr::from(k)), val]));
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
                            let add_len = self.objects[sid].len();
                            let next_len = self.objects[tid].len().saturating_add(add_len);
                            if next_len > MAX_ARRAY_LEN {
                                return Err(ExprError::RangeError(format!(
                                    "object property count {next_len} exceeds {MAX_ARRAY_LEN} limit"
                                )));
                            }
                            let key_bytes: usize = self.objects[sid].keys().map(String::len).sum();
                            self.check_array_alloc(add_len)?;
                            self.charge_alloc_bytes(key_bytes)?;
                            let entries = self.objects[sid].clone();
                            self.objects[tid].extend(entries);
                        }
                    }
                    Ok(RuntimeValue::Object(tid))
                }
                _ => Err(ExprError::TypeError(format!(
                    "Object.{} is not a function",
                    truncate_diag(method)
                ))),
            },
            NamespaceKind::Json => match method {
                "stringify" => {
                    let val = match args.first() {
                        Some(v) => self.export_root_value(v)?,
                        None => return Ok(RuntimeValue::Undefined),
                    };
                    if matches!(val, Value::Undefined) {
                        return Ok(RuntimeValue::Undefined);
                    }
                    let json = json_stringify_bounded(&val, MAX_STRING_BYTES, &mut |bytes| {
                        self.charge_byte_work(bytes)
                    })?;
                    self.check_string_alloc(json.len())?;
                    Ok(RuntimeValue::String(JsStr::from(json)))
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
                    let parsed = json_parse_bounded(&s, MAX_ARRAY_LEN, &mut |bytes| {
                        self.charge_alloc_bytes(bytes)
                    })?;
                    self.intern_value(parsed)
                }
                _ => Err(ExprError::TypeError(format!(
                    "JSON.{} is not a function",
                    truncate_diag(method)
                ))),
            },
            NamespaceKind::Boolean => Err(ExprError::TypeError(format!(
                "Boolean.{} is not a function",
                truncate_diag(method)
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
                    let val = self.read_property_value_at_time(id, t)?;
                    self.intern_value(val)
                }
                "velocityAtTime" => {
                    let t = match args.first() {
                        Some(v) => self.runtime_to_number(v)?,
                        None => self.ctx.time(),
                    };
                    let val = self.read_property_velocity(id, t)?;
                    self.intern_value(val)
                }
                "speedAtTime" => {
                    let t = match args.first() {
                        Some(v) => self.runtime_to_number(v)?,
                        None => self.ctx.time(),
                    };
                    Ok(RuntimeValue::Number(self.read_property_speed(id, t)?))
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
                    self.keyframe_to_object(kf)
                }
                "nearestKey" => {
                    let t = match args.first() {
                        Some(v) => self.runtime_to_number(v)?,
                        None => self.ctx.time(),
                    };
                    let kf = self.ctx.property_nearest_key(id, t)?;
                    self.keyframe_to_object(kf)
                }
                "parent" => self.resolve_parent_call(id, &args),
                _ => {
                    // Fall back to methods on the property's underlying value (e.g. array/string).
                    let val = self.read_property_value(id)?;
                    let rv = self.intern_value(val)?;
                    self.call_method_on(rv, method, args, false)
                }
            },
            HandleRef::Effect(_)
            | HandleRef::Object(_)
            | HandleRef::Layer(_)
            | HandleRef::Clip(_)
            | HandleRef::Sequence => self.call_object_handle_method(handle, method, args),
            HandleRef::Controls => Err(ExprError::TypeError(format!(
                "controls.{method} is not a function"
            ))),
        }
    }

    pub(super) fn resolve_object_handle_property(
        &self,
        handle: &HandleRef,
        name: &str,
    ) -> Result<HandleId, ExprError> {
        match handle {
            HandleRef::Effect(id) => self
                .ctx
                .effect_param(id, name)
                .or_else(|_| self.ctx.property(Some(id), name)),
            HandleRef::Object(id) | HandleRef::Layer(id) | HandleRef::Clip(id) => {
                self.ctx.property(Some(id), name)
            }
            HandleRef::Sequence => self.ctx.sequence_property(name),
            HandleRef::Property(id) => Ok(id.clone()),
            HandleRef::Controls => Err(ExprError::NotFound(format!("property '{name}'"))),
        }
    }

    /// Unified callable object handle matching Zoidium `plugins/expressions-plus/api.js:147-150`:
    /// - `handle(n)` where `n` is a number climbs `n - 1` parents (`handle(1) === handle`,
    ///   `thisObject.parent(2)` climbs 1 parent from `thisObject.parent`).
    /// - `handle("Prop")` reads and returns a copied value of property `"Prop"`.
    fn call_object_handle(
        &mut self,
        handle: &HandleRef,
        args: &[RuntimeValue],
    ) -> Result<RuntimeValue, ExprError> {
        let Some(first) = args.first() else {
            return Err(ExprError::TypeError(
                "object handle call requires a property name or ancestor index".to_string(),
            ));
        };
        let d = self.deref_property_handle(first.clone())?;
        if let RuntimeValue::Number(n) = d {
            if !n.is_finite() {
                return Ok(RuntimeValue::Null);
            }
            let steps = if n <= 1.0 { 0 } else { (n.ceil() as usize) - 1 };
            if steps == 0 {
                return Ok(RuntimeValue::Handle(handle.clone()));
            }
            let owner_id = match handle {
                HandleRef::Effect(id)
                | HandleRef::Object(id)
                | HandleRef::Layer(id)
                | HandleRef::Clip(id) => Some(id.clone()),
                HandleRef::Sequence => self.ctx.this_sequence_id(),
                _ => None,
            };
            let Some(id) = owner_id else {
                return Ok(RuntimeValue::Null);
            };
            return match self.ctx.parent_of(&id, steps)? {
                Some(p) => Ok(RuntimeValue::Handle(p)),
                None => Ok(RuntimeValue::Null),
            };
        }
        let name = self.runtime_to_js_string(&d)?;
        let pid = self.resolve_object_handle_property(handle, &name)?;
        let val = self.read_property_value(&pid)?;
        self.intern_value(val)
    }
}
