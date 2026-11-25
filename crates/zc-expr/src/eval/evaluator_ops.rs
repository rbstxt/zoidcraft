use super::*;

impl<'a, 'ast> Evaluator<'a, 'ast> {
    pub(super) fn eval_binary(
        &mut self,
        op: BinaryOp,
        left: RuntimeValue,
        right: RuntimeValue,
    ) -> Result<RuntimeValue, ExprError> {
        let left = self.deref_property_handle(left)?;
        let right = self.deref_property_handle(right)?;

        if let (RuntimeValue::Number(l), RuntimeValue::Number(r)) = (&left, &right) {
            let (l, r) = (*l, *r);
            return Ok(match op {
                BinaryOp::Add => RuntimeValue::Number(l + r),
                BinaryOp::Sub => RuntimeValue::Number(l - r),
                BinaryOp::Mul => RuntimeValue::Number(l * r),
                BinaryOp::Div => RuntimeValue::Number(l / r),
                BinaryOp::Rem => RuntimeValue::Number(l % r),
                BinaryOp::Pow => RuntimeValue::Number(l.powf(r)),
                BinaryOp::StrictEq | BinaryOp::Eq => RuntimeValue::Bool(l == r),
                BinaryOp::StrictNe | BinaryOp::Ne => RuntimeValue::Bool(l != r),
                BinaryOp::Lt => RuntimeValue::Bool(l < r),
                BinaryOp::Le => RuntimeValue::Bool(l <= r),
                BinaryOp::Gt => RuntimeValue::Bool(l > r),
                BinaryOp::Ge => RuntimeValue::Bool(l >= r),
                BinaryOp::BitAnd => RuntimeValue::Number((to_i32(l) & to_i32(r)) as f64),
                BinaryOp::BitOr => RuntimeValue::Number((to_i32(l) | to_i32(r)) as f64),
                BinaryOp::BitXor => RuntimeValue::Number((to_i32(l) ^ to_i32(r)) as f64),
                BinaryOp::Shl => {
                    RuntimeValue::Number(to_i32(l).wrapping_shl(to_u32(r) & 0x1f) as f64)
                }
                BinaryOp::Shr => {
                    RuntimeValue::Number(to_i32(l).wrapping_shr(to_u32(r) & 0x1f) as f64)
                }
                BinaryOp::UShr => {
                    RuntimeValue::Number(to_u32(l).wrapping_shr(to_u32(r) & 0x1f) as f64)
                }
            });
        }

        match op {
            BinaryOp::Add => {
                // String concatenation wins if either operand is a string.
                if matches!(left, RuntimeValue::String(_))
                    || matches!(right, RuntimeValue::String(_))
                {
                    let ls = self.runtime_to_js_string(&left)?;
                    let rs = self.runtime_to_js_string(&right)?;
                    let total = ls.len().checked_add(rs.len()).ok_or_else(|| {
                        ExprError::RangeError("string concatenation overflow".to_string())
                    })?;
                    self.check_string_alloc(total)?;
                    let mut combined = String::with_capacity(total);
                    combined.push_str(&ls);
                    combined.push_str(&rs);
                    return Ok(RuntimeValue::String(JsStr::from(combined)));
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
                    if !Rc::ptr_eq(&a.text, &b.text) {
                        self.charge_byte_work(a.len().min(b.len()))?;
                    }
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
                Ok(Some(self.alloc_f64_vec(out)?))
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
                Ok(Some(self.alloc_f64_vec(out)?))
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
                Ok(Some(self.alloc_f64_vec(out)?))
            }
            _ => Ok(None),
        }
    }

    pub(super) fn strict_eq(
        &mut self,
        a: &RuntimeValue,
        b: &RuntimeValue,
    ) -> Result<bool, ExprError> {
        let a = self.deref_property_handle(a.clone())?;
        let b = self.deref_property_handle(b.clone())?;
        Ok(match (a, b) {
            (RuntimeValue::Undefined, RuntimeValue::Undefined)
            | (RuntimeValue::Null, RuntimeValue::Null) => true,
            (RuntimeValue::Bool(x), RuntimeValue::Bool(y)) => x == y,
            (RuntimeValue::Number(x), RuntimeValue::Number(y)) => x == y,
            (RuntimeValue::String(x), RuntimeValue::String(y)) => {
                if Rc::ptr_eq(&x.text, &y.text) {
                    true
                } else {
                    if x.len() == y.len() {
                        self.charge_byte_work(x.len())?;
                    }
                    x == y
                }
            }
            (RuntimeValue::Array(x), RuntimeValue::Array(y)) => x == y,
            (RuntimeValue::Object(x), RuntimeValue::Object(y)) => x == y,
            (RuntimeValue::Closure(x), RuntimeValue::Closure(y)) => x == y,
            (RuntimeValue::BuiltinFn(x), RuntimeValue::BuiltinFn(y)) => match (x, y) {
                (BuiltinRef::Static(a), BuiltinRef::Static(b)) => a == b,
                (BuiltinRef::Dynamic(a), BuiltinRef::Dynamic(b)) => {
                    if Rc::ptr_eq(&a, &b) {
                        true
                    } else if a.len() != b.len() {
                        false
                    } else {
                        self.charge_byte_work(a.len())?;
                        a == b
                    }
                }
                (a, b) => a.as_str() == b.as_str(),
            },
            (RuntimeValue::Handle(x), RuntimeValue::Handle(y)) => x == y,
            (RuntimeValue::Namespace(x), RuntimeValue::Namespace(y)) => x == y,
            (RuntimeValue::Intrinsic(x), RuntimeValue::Intrinsic(y)) => x == y,
            _ => false,
        })
    }

    pub(super) fn same_value_zero(
        &mut self,
        a: &RuntimeValue,
        b: &RuntimeValue,
    ) -> Result<bool, ExprError> {
        let a = self.deref_property_handle(a.clone())?;
        let b = self.deref_property_handle(b.clone())?;
        if let (RuntimeValue::Number(x), RuntimeValue::Number(y)) = (&a, &b)
            && x.is_nan()
            && y.is_nan()
        {
            return Ok(true);
        }
        self.strict_eq(&a, &b)
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

    pub(super) fn resolve_meta_property(obj: &RuntimeValue, prop: &str) -> RuntimeValue {
        match prop {
            "constructor" => match obj {
                RuntimeValue::Array(_) | RuntimeValue::Intrinsic(IntrinsicSentinel::ArrayProto) => {
                    RuntimeValue::Namespace(NamespaceKind::Array)
                }
                RuntimeValue::Object(_)
                | RuntimeValue::Namespace(
                    NamespaceKind::Math | NamespaceKind::Json | NamespaceKind::Methods,
                )
                | RuntimeValue::Intrinsic(IntrinsicSentinel::ObjectProto) => {
                    RuntimeValue::Namespace(NamespaceKind::Object)
                }
                RuntimeValue::Number(_)
                | RuntimeValue::Intrinsic(IntrinsicSentinel::NumberProto) => {
                    RuntimeValue::Namespace(NamespaceKind::Number)
                }
                RuntimeValue::String(_)
                | RuntimeValue::Intrinsic(IntrinsicSentinel::StringProto) => {
                    RuntimeValue::Namespace(NamespaceKind::String)
                }
                RuntimeValue::Bool(_)
                | RuntimeValue::Intrinsic(IntrinsicSentinel::BooleanProto) => {
                    RuntimeValue::Namespace(NamespaceKind::Boolean)
                }
                RuntimeValue::Closure(_)
                | RuntimeValue::BuiltinFn(_)
                | RuntimeValue::Namespace(
                    NamespaceKind::Array
                    | NamespaceKind::Number
                    | NamespaceKind::String
                    | NamespaceKind::Boolean
                    | NamespaceKind::Object,
                )
                | RuntimeValue::Intrinsic(
                    IntrinsicSentinel::FunctionCtor | IntrinsicSentinel::FunctionProto,
                ) => RuntimeValue::Intrinsic(IntrinsicSentinel::FunctionCtor),
                // Null-prototype handles (`Object.create(null)`) and nullish have no constructor.
                _ => RuntimeValue::Undefined,
            },
            "__proto__" => match obj {
                RuntimeValue::Array(_) => RuntimeValue::Intrinsic(IntrinsicSentinel::ArrayProto),
                RuntimeValue::Object(_)
                | RuntimeValue::Namespace(
                    NamespaceKind::Math | NamespaceKind::Json | NamespaceKind::Methods,
                ) => RuntimeValue::Intrinsic(IntrinsicSentinel::ObjectProto),
                RuntimeValue::Number(_) => RuntimeValue::Intrinsic(IntrinsicSentinel::NumberProto),
                RuntimeValue::String(_) => RuntimeValue::Intrinsic(IntrinsicSentinel::StringProto),
                RuntimeValue::Bool(_) => RuntimeValue::Intrinsic(IntrinsicSentinel::BooleanProto),
                RuntimeValue::Closure(_)
                | RuntimeValue::BuiltinFn(_)
                | RuntimeValue::Namespace(
                    NamespaceKind::Array
                    | NamespaceKind::Number
                    | NamespaceKind::String
                    | NamespaceKind::Boolean
                    | NamespaceKind::Object,
                )
                | RuntimeValue::Intrinsic(IntrinsicSentinel::FunctionCtor) => {
                    RuntimeValue::Intrinsic(IntrinsicSentinel::FunctionProto)
                }
                RuntimeValue::Intrinsic(
                    IntrinsicSentinel::ArrayProto
                    | IntrinsicSentinel::NumberProto
                    | IntrinsicSentinel::StringProto
                    | IntrinsicSentinel::BooleanProto
                    | IntrinsicSentinel::FunctionProto,
                ) => RuntimeValue::Intrinsic(IntrinsicSentinel::ObjectProto),
                RuntimeValue::Intrinsic(IntrinsicSentinel::ObjectProto) => RuntimeValue::Null,
                _ => RuntimeValue::Undefined,
            },
            "prototype" => match obj {
                RuntimeValue::Namespace(NamespaceKind::Array) => {
                    RuntimeValue::Intrinsic(IntrinsicSentinel::ArrayProto)
                }
                RuntimeValue::Namespace(NamespaceKind::Object) => {
                    RuntimeValue::Intrinsic(IntrinsicSentinel::ObjectProto)
                }
                RuntimeValue::Namespace(NamespaceKind::Number) => {
                    RuntimeValue::Intrinsic(IntrinsicSentinel::NumberProto)
                }
                RuntimeValue::Namespace(NamespaceKind::String) => {
                    RuntimeValue::Intrinsic(IntrinsicSentinel::StringProto)
                }
                RuntimeValue::Namespace(NamespaceKind::Boolean) => {
                    RuntimeValue::Intrinsic(IntrinsicSentinel::BooleanProto)
                }
                RuntimeValue::Intrinsic(IntrinsicSentinel::FunctionCtor) => {
                    RuntimeValue::Intrinsic(IntrinsicSentinel::FunctionProto)
                }
                _ => RuntimeValue::Undefined,
            },
            _ => RuntimeValue::Undefined,
        }
    }

    pub(super) fn get_property(
        &mut self,
        obj: &RuntimeValue,
        prop: &str,
    ) -> Result<RuntimeValue, ExprError> {
        if matches!(obj, RuntimeValue::Null | RuntimeValue::Undefined) {
            return Err(ExprError::TypeError(format!(
                "cannot read property '{}' of null or undefined",
                truncate_diag(prop)
            )));
        }
        if matches!(prop, "__proto__" | "constructor" | "prototype") {
            return Ok(Self::resolve_meta_property(obj, prop));
        }

        match obj {
            RuntimeValue::Null | RuntimeValue::Undefined => unreachable!(),
            RuntimeValue::Namespace(NamespaceKind::Math) => {
                if prop == "random" {
                    return Err(ExprError::Forbidden("Math.random".to_string()));
                }
                if let Some(c) = math_constant(prop) {
                    return Ok(RuntimeValue::Number(c));
                }
                if let Some(s) = static_math_builtin_name(prop) {
                    return Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(s)));
                }
                self.alloc_builtin_fn(format!("Math.{prop}"))
            }
            RuntimeValue::Namespace(NamespaceKind::Methods) => {
                if is_methods_member(prop, self.config.mode) {
                    if let Some(s) = static_global_builtin_name(prop) {
                        Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(s)))
                    } else {
                        self.alloc_builtin_fn(prop.to_string())
                    }
                } else {
                    Ok(RuntimeValue::Undefined)
                }
            }
            RuntimeValue::Namespace(NamespaceKind::Number) => match prop {
                "MAX_SAFE_INTEGER" => Ok(RuntimeValue::Number(9_007_199_254_740_991.0)),
                "MIN_SAFE_INTEGER" => Ok(RuntimeValue::Number(-9_007_199_254_740_991.0)),
                "EPSILON" => Ok(RuntimeValue::Number(f64::EPSILON)),
                "POSITIVE_INFINITY" => Ok(RuntimeValue::Number(f64::INFINITY)),
                "NEGATIVE_INFINITY" => Ok(RuntimeValue::Number(f64::NEG_INFINITY)),
                "NaN" => Ok(RuntimeValue::Number(f64::NAN)),
                "isNaN" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static("Number.isNaN"))),
                "isFinite" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(
                    "Number.isFinite",
                ))),
                "isInteger" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(
                    "Number.isInteger",
                ))),
                "isSafeInteger" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(
                    "Number.isSafeInteger",
                ))),
                "parseInt" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(
                    "Number.parseInt",
                ))),
                "parseFloat" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(
                    "Number.parseFloat",
                ))),
                _ => self.alloc_builtin_fn(format!("Number.{prop}")),
            },
            RuntimeValue::Namespace(NamespaceKind::Array) => match prop {
                "isArray" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static("Array.isArray"))),
                "from" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static("Array.from"))),
                "of" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static("Array.of"))),
                _ => self.alloc_builtin_fn(format!("Array.{prop}")),
            },
            RuntimeValue::Namespace(NamespaceKind::String) => match prop {
                "fromCharCode" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(
                    "String.fromCharCode",
                ))),
                "fromCodePoint" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(
                    "String.fromCodePoint",
                ))),
                _ => self.alloc_builtin_fn(format!("String.{prop}")),
            },
            RuntimeValue::Namespace(NamespaceKind::Object) => match prop {
                "keys" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static("Object.keys"))),
                "values" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static("Object.values"))),
                "entries" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(
                    "Object.entries",
                ))),
                "assign" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static("Object.assign"))),
                "fromEntries" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(
                    "Object.fromEntries",
                ))),
                "hasOwn" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static("Object.hasOwn"))),
                "is" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static("Object.is"))),
                "freeze" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static("Object.freeze"))),
                "create" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static("Object.create"))),
                _ => self.alloc_builtin_fn(format!("Object.{prop}")),
            },
            RuntimeValue::Namespace(NamespaceKind::Json) => match prop {
                "parse" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static("JSON.parse"))),
                "stringify" => Ok(RuntimeValue::BuiltinFn(BuiltinRef::Static(
                    "JSON.stringify",
                ))),
                _ => self.alloc_builtin_fn(format!("JSON.{prop}")),
            },
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
                    let len = self.string_char_len(s)?;
                    return Ok(RuntimeValue::Number(len as f64));
                }
                if let Ok(idx) = prop.parse::<usize>() {
                    return Ok(self
                        .string_char_at(s, idx)?
                        .map(|c| RuntimeValue::String(JsStr::from(c.to_string())))
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

    pub(super) fn get_computed(
        &mut self,
        obj: &RuntimeValue,
        idx: &RuntimeValue,
    ) -> Result<RuntimeValue, ExprError> {
        let idx = self.deref_property_handle(idx.clone())?;
        if matches!(
            obj,
            RuntimeValue::Namespace(NamespaceKind::Math | NamespaceKind::Methods)
        ) {
            let key = self.runtime_to_js_string(&idx)?;
            return self.get_property(obj, &key);
        }
        if matches!(obj, RuntimeValue::Handle(HandleRef::Controls)) {
            let name = self.runtime_to_js_string(&idx)?;
            let val = self.ctx.control(&name, None)?;
            return self.intern_value(val);
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
                    if n >= 0.0 && n.fract() == 0.0 && n.is_finite() {
                        return Ok(self
                            .string_char_at(&s, n as usize)?
                            .map(|c| RuntimeValue::String(JsStr::from(c.to_string())))
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

    fn property_derivative_time(&self, id: &HandleId) -> f64 {
        let fps = self.ctx.fps();
        if fps > 0.0 {
            let local_frame = self
                .ctx
                .property_local_frame(id)
                .unwrap_or_else(|| self.ctx.frame());
            local_frame / fps
        } else {
            self.ctx.time()
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
                    let v = self.read_property_value(id)?;
                    self.intern_value(v)
                }
                "velocity" => {
                    let at_time = self.property_derivative_time(id);
                    let v = self.read_property_velocity(id, at_time)?;
                    self.intern_value(v)
                }
                "speed" => {
                    let at_time = self.property_derivative_time(id);
                    let s = self.read_property_speed(id, at_time)?;
                    Ok(RuntimeValue::Number(s))
                }
                "numKeys" => Ok(RuntimeValue::Number(self.ctx.property_num_keys(id) as f64)),
                "name" => Ok(self
                    .ctx
                    .handle_name(id)
                    .map(|s| RuntimeValue::String(JsStr::from(s)))
                    .unwrap_or(RuntimeValue::Undefined)),
                "parent" => match self.ctx.parent_of(id, 1)? {
                    Some(p) => Ok(RuntimeValue::Handle(p)),
                    None => Ok(RuntimeValue::Null),
                },
                "length" => {
                    let v = self.read_property_value(id)?;
                    let rv = self.intern_value(v)?;
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
                    .map(|s| RuntimeValue::String(JsStr::from(s)))
                    .unwrap_or(RuntimeValue::Undefined)),
                "parent" => match self.ctx.parent_of(id, 1)? {
                    Some(p) => Ok(RuntimeValue::Handle(p)),
                    None => Ok(RuntimeValue::Null),
                },
                "frameDuration" => Ok(RuntimeValue::Number(self.ctx.frame_duration())),
                _ => Ok(RuntimeValue::Undefined),
            },
            HandleRef::Sequence => match prop {
                "frameDuration" => Ok(RuntimeValue::Number(self.ctx.frame_duration())),
                "fps" => Ok(RuntimeValue::Number(self.ctx.fps())),
                "name" => Ok(self
                    .ctx
                    .this_sequence_id()
                    .and_then(|id| self.ctx.handle_name(&id))
                    .map(|s| RuntimeValue::String(JsStr::from(s)))
                    .unwrap_or(RuntimeValue::Undefined)),
                "parent" => {
                    if let Some(id) = self.ctx.this_sequence_id() {
                        match self.ctx.parent_of(&id, 1)? {
                            Some(p) => Ok(RuntimeValue::Handle(p)),
                            None => Ok(RuntimeValue::Null),
                        }
                    } else {
                        Ok(RuntimeValue::Null)
                    }
                }
                _ => Ok(RuntimeValue::Undefined),
            },
            HandleRef::Controls => {
                let val = self.ctx.control(prop, None)?;
                self.intern_value(val)
            }
        }
    }

    pub(super) fn eval_optional_chain(
        &mut self,
        base: &'ast Expr,
        ops: &'ast [ChainOp],
        env: usize,
    ) -> Result<RuntimeValue, ExprError> {
        let mut current = self.eval_expr(base, env)?;
        let mut idx = 0;
        while idx < ops.len() {
            match &ops[idx] {
                ChainOp::Member { prop, optional } => {
                    if *optional {
                        let derefed = self.deref_property_handle(current.clone())?;
                        if matches!(derefed, RuntimeValue::Null | RuntimeValue::Undefined) {
                            return Ok(RuntimeValue::Undefined);
                        }
                    } else if matches!(current, RuntimeValue::Null | RuntimeValue::Undefined) {
                        return Err(ExprError::TypeError(format!(
                            "cannot read property '{}' of null or undefined",
                            truncate_diag(prop)
                        )));
                    }
                    // Look ahead for method call `a?.b(args)` or `a?.b?.(args)`.
                    if let Some(ChainOp::Call {
                        args,
                        optional: call_opt,
                    }) = ops.get(idx + 1)
                    {
                        if *call_opt && self.member_is_nullish(&current, prop)? {
                            return Ok(RuntimeValue::Undefined);
                        }
                        let eval_args = self.eval_call_args(args, env)?;
                        current = self.call_method_on(current, prop, eval_args, *call_opt)?;
                        idx += 2;
                        continue;
                    }
                    current = self.get_property(&current, prop)?;
                }
                ChainOp::Index { index, optional } => {
                    if *optional {
                        let derefed = self.deref_property_handle(current.clone())?;
                        if matches!(derefed, RuntimeValue::Null | RuntimeValue::Undefined) {
                            return Ok(RuntimeValue::Undefined);
                        }
                    }
                    let key = self.eval_expr(index, env)?;
                    current = self.get_computed(&current, &key)?;
                }
                ChainOp::Call { args, optional } => {
                    if *optional {
                        let derefed = self.deref_property_handle(current.clone())?;
                        if matches!(derefed, RuntimeValue::Null | RuntimeValue::Undefined) {
                            return Ok(RuntimeValue::Undefined);
                        }
                    }
                    let eval_args = self.eval_call_args(args, env)?;
                    current = self.call_runtime_value(current, eval_args)?;
                }
            }
            idx += 1;
        }
        Ok(current)
    }

    /// Returns `true` when accessing `receiver.prop` yields `null` or `undefined`
    /// (including absent built-in prototype/namespace/handle methods), used to
    /// short-circuit optional calls `receiver.prop?.(args)` before evaluating `args`.
    fn member_is_nullish(
        &mut self,
        receiver: &RuntimeValue,
        prop: &str,
    ) -> Result<bool, ExprError> {
        if matches!(prop, "__proto__" | "constructor" | "prototype") {
            let meta = Self::resolve_meta_property(receiver, prop);
            return Ok(matches!(meta, RuntimeValue::Null | RuntimeValue::Undefined));
        }
        match receiver {
            RuntimeValue::Object(id) => {
                let Some(val) = self.objects[*id].get(prop).cloned() else {
                    return Ok(true);
                };
                let derefed = self.deref_property_handle(val)?;
                Ok(matches!(
                    derefed,
                    RuntimeValue::Null | RuntimeValue::Undefined
                ))
            }
            RuntimeValue::Namespace(ns) => match ns {
                NamespaceKind::Math => {
                    if prop == "random" {
                        return Err(ExprError::Forbidden("Math.random".to_string()));
                    }
                    Ok(static_math_builtin_name(prop).is_none() && math_constant(prop).is_none())
                }
                NamespaceKind::Methods => Ok(!is_methods_member(prop, self.config.mode)),
                NamespaceKind::Array => Ok(!matches!(prop, "isArray" | "from")),
                NamespaceKind::Number => Ok(!matches!(
                    prop,
                    "isFinite"
                        | "isNaN"
                        | "isInteger"
                        | "parseFloat"
                        | "parseInt"
                        | "MAX_SAFE_INTEGER"
                        | "MIN_SAFE_INTEGER"
                        | "EPSILON"
                        | "POSITIVE_INFINITY"
                        | "NEGATIVE_INFINITY"
                        | "NaN"
                )),
                NamespaceKind::String => Ok(prop != "fromCharCode"),
                NamespaceKind::Object => {
                    Ok(!matches!(prop, "keys" | "values" | "entries" | "assign"))
                }
                NamespaceKind::Json => Ok(!matches!(prop, "stringify" | "parse")),
                NamespaceKind::Boolean => Ok(true),
            },
            RuntimeValue::Array(id) => {
                if matches!(
                    prop,
                    "push"
                        | "pop"
                        | "shift"
                        | "unshift"
                        | "reverse"
                        | "slice"
                        | "concat"
                        | "join"
                        | "indexOf"
                        | "includes"
                        | "map"
                        | "filter"
                        | "reduce"
                        | "forEach"
                        | "find"
                        | "some"
                        | "every"
                        | "fill"
                        | "length"
                ) {
                    return Ok(false);
                }
                if let Ok(idx) = prop.parse::<usize>()
                    && let Some(elem) = self.arrays[*id].get(idx).cloned()
                {
                    let derefed = self.deref_property_handle(elem)?;
                    return Ok(matches!(
                        derefed,
                        RuntimeValue::Null | RuntimeValue::Undefined
                    ));
                }
                Ok(true)
            }
            RuntimeValue::String(s) => {
                if matches!(
                    prop,
                    "toLowerCase"
                        | "toUpperCase"
                        | "trim"
                        | "trimStart"
                        | "trimEnd"
                        | "slice"
                        | "substring"
                        | "indexOf"
                        | "includes"
                        | "startsWith"
                        | "endsWith"
                        | "split"
                        | "replace"
                        | "replaceAll"
                        | "charAt"
                        | "charCodeAt"
                        | "repeat"
                        | "length"
                ) {
                    return Ok(false);
                }
                if let Ok(idx) = prop.parse::<usize>() {
                    return Ok(self.string_char_at(s, idx)?.is_none());
                }
                Ok(true)
            }
            RuntimeValue::Number(_) => Ok(!matches!(prop, "toFixed" | "toString")),
            RuntimeValue::Handle(handle) => match handle {
                HandleRef::Property(id) => match prop {
                    "valueAtTime" | "velocityAtTime" | "speedAtTime" | "key" | "nearestKey"
                    | "parent" | "value" | "velocity" | "speed" | "numKeys" | "length" => Ok(false),
                    "name" => Ok(self.ctx.handle_name(id).is_none()),
                    _ => {
                        let val = self.read_property_value(id)?;
                        let rv = self.intern_value(val)?;
                        if matches!(rv, RuntimeValue::Null | RuntimeValue::Undefined) {
                            Ok(true)
                        } else {
                            self.member_is_nullish(&rv, prop)
                        }
                    }
                },
                HandleRef::Effect(id)
                | HandleRef::Object(id)
                | HandleRef::Layer(id)
                | HandleRef::Clip(id) => match prop {
                    "param" | "property" | "parent" | "effect" | "object" | "content" | "layer"
                    | "frameDuration" | "toWorld" | "fromWorld" | "worldPosition"
                    | "worldRotation" | "toScreen" | "fromScreen" => Ok(false),
                    "name" => Ok(self.ctx.handle_name(id).is_none()),
                    _ => Ok(true),
                },
                HandleRef::Sequence => match prop {
                    "param" | "property" | "parent" | "effect" | "object" | "content" | "layer"
                    | "frameDuration" | "fps" | "toWorld" | "fromWorld" | "worldPosition"
                    | "worldRotation" | "toScreen" | "fromScreen" => Ok(false),
                    "name" => Ok(self
                        .ctx
                        .this_sequence_id()
                        .and_then(|id| self.ctx.handle_name(&id))
                        .is_none()),
                    _ => Ok(true),
                },
                HandleRef::Controls => Ok(true),
            },
            _ => Ok(true),
        }
    }

    // -----------------------------------------------------------------------
    // Function & Method Calls
    // -----------------------------------------------------------------------
}
