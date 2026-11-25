use super::*;

impl<'a, 'ast> Evaluator<'a, 'ast> {
    pub(super) fn call_builtin(
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
                self.alloc_f64_vec(vec_cross(&a, &b))
            }
            "normalize" => {
                let v = self.require_vec_arg(&args, 0, "normalize")?;
                self.alloc_f64_vec(vec_normalize(&v))
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
                self.intern_value(res)
            }
            "linear" => self.builtin_linear(&args),
            "catmullRom" => {
                let p0 = self.export_arg(&args, 0)?;
                let p1 = self.export_arg(&args, 1)?;
                let p2 = self.export_arg(&args, 2)?;
                let p3 = self.export_arg(&args, 3)?;
                let t = self.number_arg(&args, 4, 0.0)?;
                let res = catmull_rom_value(&p0, &p1, &p2, &p3, t)?;
                self.intern_value(res)
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
                    return self.alloc_f64_vec(v.into_iter().map(|x| x * PI / 180.0).collect());
                }
                let x = self.runtime_to_number(first)?;
                Ok(RuntimeValue::Number(x * PI / 180.0))
            }
            "radiansToDegrees" => {
                let Some(first) = args.first() else {
                    return Ok(RuntimeValue::Number(f64::NAN));
                };
                if let Some(v) = self.runtime_as_f64_vec(first)? {
                    return self.alloc_f64_vec(v.into_iter().map(|x| x * 180.0 / PI).collect());
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
                self.alloc_f64_vec(euler.to_vec())
            }
            "seedRandom" => {
                let seed = self.number_arg(&args, 0, 0.0)?;
                let timeless = match args.get(1) {
                    Some(v) => self.runtime_to_bool(v)?,
                    None => false,
                };
                self.rng_seed = seed;
                self.rng_timeless = timeless;
                self.rng_index = 0;
                self.rng_seeded = true;
                Ok(RuntimeValue::Undefined)
            }
            "wiggle" => {
                let freq = self.number_arg(&args, 0, 1.0)?;
                let amp = match args.get(1) {
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        self.export_root_value(v)?
                    }
                    _ => Value::Number(1.0),
                };
                let octaves_f = self.number_arg(&args, 2, 2.0)?;
                let octaves = if octaves_f.is_finite() {
                    js_round(octaves_f).clamp(1.0, 1024.0) as usize
                } else {
                    1
                };
                let amp_mult = self.number_arg(&args, 3, 0.45)?;
                let t = self.number_arg(&args, 4, self.ctx.frame())?;
                let addr = self.ctx.property_address();
                let seed = property_seed(addr.as_deref());
                let base_val = self.ctx.value();
                let res = eval_wiggle(&base_val, freq, &amp, octaves, amp_mult, t, seed);
                self.intern_value(res)
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
                let val = self.read_property_value(&pid)?;
                self.intern_value(val)
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
                self.intern_value(val)
            }
            "toWorld" => {
                let pt = match args.first() {
                    Some(v) => self.runtime_to_vec3(v)?,
                    None => [0.0, 0.0, 0.0],
                };
                let out = self.ctx.to_world(None, pt)?;
                self.alloc_f64_vec(out.to_vec())
            }
            "fromWorld" => {
                let pt = match args.first() {
                    Some(v) => self.runtime_to_vec3(v)?,
                    None => [0.0, 0.0, 0.0],
                };
                let out = self.ctx.from_world(None, pt)?;
                self.alloc_f64_vec(out.to_vec())
            }
            "worldPosition" => {
                let out = self.ctx.world_position(None)?;
                self.alloc_f64_vec(out.to_vec())
            }
            "worldRotation" => {
                let out = self.ctx.world_rotation(None)?;
                self.alloc_f64_vec(out.to_vec())
            }
            "toScreen" => {
                let pt = match args.first() {
                    Some(v) => self.runtime_to_vec3(v)?,
                    None => [0.0, 0.0, 0.0],
                };
                let out = self.ctx.to_screen(None, pt)?;
                self.alloc_f64_vec(out.to_vec())
            }
            "fromScreen" => {
                let pt = match args.first() {
                    Some(v) => self.runtime_to_vec3(v)?,
                    None => [0.0, 0.0, 0.0],
                };
                let out = self.ctx.from_screen(None, pt)?;
                self.alloc_f64_vec(out.to_vec())
            }
            _ => Err(ExprError::ReferenceError(truncate_diag(name).into_owned())),
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
            self.alloc_f64_vec(out)
        } else if self.config.mode == ExprMode::ExpressionsPlus && self.is_extended {
            let x = self.runtime_to_number(first)?;
            let lo = self.number_arg(args, 1, 0.0)?;
            let hi = self.number_arg(args, 2, 1.0)?;
            Ok(RuntimeValue::Number(clamp_f64(x, lo, hi)))
        } else {
            // Pure CM3 `clamp` only operates on arrays (`new Array(v.length)` -> `[undefined]`).
            self.check_array_alloc(1)?;
            Ok(self.alloc_array(vec![RuntimeValue::Undefined]))
        }
    }

    fn builtin_linear(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, ExprError> {
        if args.len() == 5 && self.config.mode == ExprMode::ExpressionsPlus && self.is_extended {
            let t = self.number_arg(args, 0, 0.0)?;
            let t_min = self.number_arg(args, 1, 0.0)?;
            let t_max = self.number_arg(args, 2, 1.0)?;
            let a = self.export_arg(args, 3)?;
            let b = self.export_arg(args, 4)?;
            let res = ae_linear(t, t_min, t_max, &a, &b)?;
            return self.intern_value(res);
        }
        // Vanilla CM3 3-argument form: `linear(a, b, t) = (b - a) * t + a` (ignoring extra args).
        let a = self.export_arg(args, 0)?;
        let b = self.export_arg(args, 1)?;
        let t = self.number_arg(args, 2, 0.0)?;
        let res = lerp_value(&a, &b, t)?;
        self.intern_value(res)
    }

    fn builtin_ease(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, ExprError> {
        if args.len() == 5 && self.config.mode == ExprMode::ExpressionsPlus && self.is_extended {
            let t = self.number_arg(args, 0, 0.0)?;
            let t_min = self.number_arg(args, 1, 0.0)?;
            let t_max = self.number_arg(args, 2, 1.0)?;
            let a = self.export_arg(args, 3)?;
            let b = self.export_arg(args, 4)?;
            let res = ae_ease(t, t_min, t_max, &a, &b)?;
            return self.intern_value(res);
        }
        // Vanilla CM3 form: `ease(t, family = "Quadratic", dir = "InOut")` (ignoring extra args).
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
        if args.len() == 5 {
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
            self.intern_value(res)
        } else if args.len() >= 3 {
            // 3-arg Zoidium `interpolate([t, a, b], curve)` (`api.js:7`).
            let t = clamp_f64(self.number_arg(args, 0, 0.0)?, 0.0, 1.0);
            let s = if is_in { t * t } else { t * (2.0 - t) };
            let a = self.export_arg(args, 1)?;
            let b = self.export_arg(args, 2)?;
            let res = lerp_value(&a, &b, s)?;
            self.intern_value(res)
        } else {
            let t = clamp_f64(self.number_arg(args, 0, 0.0)?, 0.0, 1.0);
            let s = if is_in { t * t } else { t * (2.0 - t) };
            Ok(RuntimeValue::Number(s))
        }
    }

    fn builtin_random(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, ExprError> {
        // Matches Zoidium `plugins/expressions-plus/api.js:249-253`:
        // - 0-arg after `seedRandom(seed, timeless)`:
        //   `cm3_random(seed + (timeless ? 0 : frame) * 17.173 + randomIndex++ * 71.137)`.
        // - 0-arg without `seedRandom`: `cm3_random(NaN)` -> `NaN`.
        // - 1+ args: ALWAYS keeps CM3's 1-arg `cm3_random(args[0])` hash.
        if self.config.mode == ExprMode::ExpressionsPlus && args.is_empty() && self.rng_seeded {
            let frame = if self.rng_timeless {
                0.0
            } else {
                self.ctx.frame()
            };
            let n = self.rng_seed + frame * 17.173 + (self.rng_index as f64) * 71.137;
            self.rng_index = self.rng_index.saturating_add(1);
            return Ok(RuntimeValue::Number(cm3_random(n)));
        }

        let seed = match args.first() {
            Some(v) => self.runtime_to_number(v)?,
            None => f64::NAN,
        };
        Ok(RuntimeValue::Number(cm3_random(seed)))
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
            Some(v) => self.export_root_value(v),
            None => Ok(Value::Number(0.0)),
        }
    }
}
