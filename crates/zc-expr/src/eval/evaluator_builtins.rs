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
            "gaussRandom" => self.builtin_gauss_random(&args),
            "shake" => {
                let f = self.number_arg(&args, 0, self.effective_frame())?;
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
                let pid = self.ctx.this_property_id();
                self.eval_wiggle_common(pid.as_ref(), &args, false)
            }
            "noise" => self.builtin_noise(&args),
            "posterizeTime" => self.builtin_posterize_time(&args),
            "loopIn" => {
                if let Some(pid) = self.ctx.this_property_id() {
                    self.eval_property_loop(&pid, &args, true)
                } else {
                    let v = self.ctx.value();
                    self.intern_value(v)
                }
            }
            "loopOut" => {
                if let Some(pid) = self.ctx.this_property_id() {
                    self.eval_property_loop(&pid, &args, false)
                } else {
                    let v = self.ctx.value();
                    self.intern_value(v)
                }
            }
            "smooth" => {
                if let Some(pid) = self.ctx.this_property_id() {
                    self.eval_property_smooth(&pid, &args)
                } else {
                    let v = self.ctx.value();
                    self.intern_value(v)
                }
            }
            "focusDistanceTo" => self.builtin_focus_distance_to(&args),
            "audio" => self.builtin_audio(&args),
            "audioLevel" => self.builtin_audio_level(&args),

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
                self.effective_frame()
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

    pub(super) fn extended_builtin_static_name(name: &str) -> Option<&'static str> {
        Some(match name {
            "noise" => "noise",
            "posterizeTime" => "posterizeTime",
            "loopIn" => "loopIn",
            "loopOut" => "loopOut",
            "smooth" => "smooth",
            "focusDistanceTo" => "focusDistanceTo",
            "audio" => "audio",
            "audioLevel" => "audioLevel",
            _ => return None,
        })
    }

    pub(super) fn posterized_time(&self) -> Option<f64> {
        if let Some(root) = self.envs.first()
            && let Some(b) = root.bindings.get("__zc_posterized_time__")
            && let RuntimeValue::Number(t) = b.value
        {
            return Some(t);
        }
        None
    }

    pub(super) fn effective_time(&self) -> f64 {
        self.posterized_time().unwrap_or_else(|| self.ctx.time())
    }

    pub(super) fn effective_frame(&self) -> f64 {
        if let Some(root) = self.envs.first()
            && let Some(b) = root.bindings.get("__zc_posterized_frame__")
            && let RuntimeValue::Number(f) = b.value
        {
            return f;
        }
        self.ctx.frame()
    }

    fn builtin_gauss_random(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, ExprError> {
        if self.config.mode == ExprMode::ExpressionsPlus && self.rng_seeded {
            let frame = if self.rng_timeless {
                0.0
            } else {
                self.effective_frame()
            };
            let n = self.rng_seed + frame * 17.173 + (self.rng_index as f64) * 71.137;
            self.rng_index = self.rng_index.saturating_add(1);
            if args.is_empty() {
                let mut sum = 0.0;
                for k in 0..5 {
                    sum += cm3_random(n + (k as f64) * 13.371 + 1.0);
                }
                return Ok(RuntimeValue::Number(0.25 * (sum - 2.5) / 2.5 + 0.5));
            }
            let seed = self.number_arg(args, 0, 0.0)?;
            let mean = self.number_arg(args, 1, 0.5)?;
            let std_dev = self.number_arg(args, 2, 0.25)?;
            return Ok(RuntimeValue::Number(cm3_gauss_random(
                seed + n + 1.0,
                mean,
                std_dev,
            )));
        }
        let seed = self.number_arg(args, 0, 0.0)?;
        let mean = self.number_arg(args, 1, 0.5)?;
        let std_dev = self.number_arg(args, 2, 0.25)?;
        Ok(RuntimeValue::Number(cm3_gauss_random(seed, mean, std_dev)))
    }

    pub(super) fn eval_property_wiggle(
        &mut self,
        id: &HandleId,
        args: &[RuntimeValue],
    ) -> Result<RuntimeValue, ExprError> {
        self.eval_wiggle_common(Some(id), args, true)
    }

    fn eval_wiggle_common(
        &mut self,
        prop_id: Option<&HandleId>,
        args: &[RuntimeValue],
        anchor_at_t: bool,
    ) -> Result<RuntimeValue, ExprError> {
        let freq = self.number_arg(args, 0, 1.0)?;
        let amp = match args.get(1) {
            Some(v) if !matches!(v, RuntimeValue::Undefined) => self.export_root_value(v)?,
            _ => Value::Number(1.0),
        };
        let octaves_f = self.number_arg(args, 2, 2.0)?;
        let octaves = if octaves_f.is_finite() {
            js_round(octaves_f).clamp(1.0, 1024.0) as usize
        } else {
            1
        };
        let amp_mult = self.number_arg(args, 3, 0.45)?;
        let has_explicit_t = args
            .get(4)
            .is_some_and(|v| !matches!(v, RuntimeValue::Undefined));
        let eff_frame = self.effective_frame();
        let t = self.number_arg(args, 4, eff_frame)?;

        let base_val = if anchor_at_t && let Some(id) = prop_id {
            if has_explicit_t {
                self.read_property_value_at_time(id, t)?
            } else if let Some(pt) = self.posterized_time() {
                self.read_property_value_at_time(id, pt)?
            } else {
                self.read_property_value(id)?
            }
        } else if let Some(pt) = self.posterized_time()
            && let Some(id) = prop_id
        {
            self.read_property_value_at_time(id, pt)?
        } else {
            self.ctx.value()
        };

        let base_seed = if prop_id.is_none() || self.ctx.this_property_id().as_ref() == prop_id {
            let addr = self.ctx.property_address();
            property_seed(addr.as_deref())
        } else {
            property_seed(prop_id.map(HandleId::as_str))
        };
        let seed = if self.rng_seeded {
            let rng =
                crate::builtins::DetRng::from_seed(self.rng_seed, eff_frame, self.rng_timeless);
            base_seed + rng.seed_offset()
        } else {
            base_seed
        };

        let res = eval_wiggle(&base_val, freq, &amp, octaves, amp_mult, t, seed);
        self.intern_value(res)
    }

    fn builtin_noise(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, ExprError> {
        let Some(first) = args.first() else {
            return Ok(RuntimeValue::Number(0.0));
        };
        if let Some(vec) = self.runtime_as_f64_vec(first)? {
            let out = match vec.as_slice() {
                [] => 0.0,
                [x] => perlin_noise_1d(*x),
                [x, y] => perlin_noise_2d(*x, *y),
                [x, y, z, ..] => perlin_noise_3d(*x, *y, *z),
            };
            return Ok(RuntimeValue::Number(out));
        }
        let x = self.runtime_to_number(first)?;
        Ok(RuntimeValue::Number(perlin_noise_1d(x)))
    }

    fn builtin_posterize_time(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, ExprError> {
        let p_fps = self.number_arg(args, 0, 0.0)?;
        if !p_fps.is_finite() {
            return Ok(RuntimeValue::Undefined);
        }
        let (q_time, q_frame) = if p_fps <= 0.0 {
            (0.0, 0.0)
        } else {
            let raw_time = self.ctx.time();
            let q_t = (raw_time * p_fps).floor() / p_fps;
            let ctx_fps = self.ctx.fps();
            let q_f = if ctx_fps > 0.0 {
                q_t * ctx_fps
            } else {
                (self.ctx.frame() * p_fps).floor() / p_fps
            };
            (q_t, q_f)
        };
        if let Some(root) = self.envs.first_mut() {
            root.bindings.insert(
                "__zc_posterized_time__".to_string(),
                Binding {
                    value: RuntimeValue::Number(q_time),
                    mutable: true,
                    is_var: false,
                },
            );
            root.bindings.insert(
                "__zc_posterized_frame__".to_string(),
                Binding {
                    value: RuntimeValue::Number(q_frame),
                    mutable: true,
                    is_var: false,
                },
            );
            root.bindings.insert(
                "time".to_string(),
                Binding {
                    value: RuntimeValue::Number(q_time),
                    mutable: true,
                    is_var: false,
                },
            );
            root.bindings.insert(
                "frame".to_string(),
                Binding {
                    value: RuntimeValue::Number(q_frame),
                    mutable: true,
                    is_var: false,
                },
            );
        }
        if let Some(pid) = self.ctx.this_property_id()
            && let Ok(val) = self.read_property_value_at_time(&pid, q_time)
            && let Ok(interned) = self.intern_value(val)
        {
            self.cached_value = Some(interned);
        }
        Ok(RuntimeValue::Undefined)
    }

    pub(super) fn eval_property_loop(
        &mut self,
        id: &HandleId,
        args: &[RuntimeValue],
        is_in: bool,
    ) -> Result<RuntimeValue, ExprError> {
        let loop_type = match args.first() {
            Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                self.runtime_to_js_string(v)?.to_ascii_lowercase()
            }
            _ => "cycle".to_string(),
        };
        let num_kf_f = self.number_arg(args, 1, 0.0)?;
        let num_kf = if num_kf_f.is_finite() && num_kf_f > 0.0 {
            num_kf_f.trunc() as usize
        } else {
            0
        };

        let t = self.effective_time();
        let count = self.ctx.property_num_keys(id);
        if count < 2 {
            let val = self.read_property_value_at_time(id, t)?;
            return self.intern_value(val);
        }

        let (k_start, k_end) = if is_in {
            let first = self.ctx.property_key(id, 1)?;
            if t >= first.time {
                let val = self.read_property_value_at_time(id, t)?;
                return self.intern_value(val);
            }
            let end_idx = if num_kf == 0 {
                count
            } else {
                (1 + num_kf).min(count)
            };
            let end = self.ctx.property_key(id, end_idx)?;
            (first, end)
        } else {
            let last = self.ctx.property_key(id, count)?;
            if t <= last.time {
                let val = self.read_property_value_at_time(id, t)?;
                return self.intern_value(val);
            }
            let start_idx = if num_kf == 0 {
                1
            } else {
                count.saturating_sub(num_kf).max(1)
            };
            let start = self.ctx.property_key(id, start_idx)?;
            (start, last)
        };

        let duration = k_end.time - k_start.time;
        if duration <= 1e-12 {
            let anchor_t = if is_in { k_start.time } else { k_end.time };
            let val = self.read_property_value_at_time(id, anchor_t)?;
            return self.intern_value(val);
        }

        let result_val = match loop_type.as_str() {
            "cycle" => {
                let rem = (t - k_start.time).rem_euclid(duration);
                self.read_property_value_at_time(id, k_start.time + rem)?
            }
            "pingpong" => {
                let sample_t = if is_in {
                    let before = k_start.time - t;
                    let cycle = (before / duration).floor() as i64;
                    let rem = before - (cycle as f64) * duration;
                    if (cycle & 1) == 0 {
                        k_start.time + rem
                    } else {
                        k_end.time - rem
                    }
                } else {
                    let elapsed = t - k_end.time;
                    let cycle = (elapsed / duration).floor() as i64;
                    let rem = elapsed - (cycle as f64) * duration;
                    if (cycle & 1) == 0 {
                        k_end.time - rem
                    } else {
                        k_start.time + rem
                    }
                };
                self.read_property_value_at_time(id, sample_t)?
            }
            "offset" => {
                let offset_t = (t - k_start.time).rem_euclid(duration);
                let cycles = ((t - k_start.time) / duration).floor();
                let base = self.read_property_value_at_time(id, k_start.time + offset_t)?;
                let delta = value_sub(&k_end.value, &k_start.value);
                value_add_scaled(&base, &delta, cycles)
            }
            "continue" => {
                let dt = (duration * 0.001).clamp(1e-6, 0.001);
                if is_in {
                    let v0 = self.read_property_value_at_time(id, k_start.time)?;
                    let v1 = self.read_property_value_at_time(id, k_start.time + dt)?;
                    let slope = value_scale(&value_sub(&v1, &v0), 1.0 / dt);
                    value_add_scaled(&v0, &slope, t - k_start.time)
                } else {
                    let v_end = self.read_property_value_at_time(id, k_end.time)?;
                    let v_prev = self.read_property_value_at_time(id, k_end.time - dt)?;
                    let slope = value_scale(&value_sub(&v_end, &v_prev), 1.0 / dt);
                    value_add_scaled(&v_end, &slope, t - k_end.time)
                }
            }
            other => {
                return Err(ExprError::RangeError(format!(
                    "unknown loop type '{}'",
                    truncate_diag(other)
                )));
            }
        };

        self.intern_value(result_val)
    }

    pub(super) fn eval_property_smooth(
        &mut self,
        id: &HandleId,
        args: &[RuntimeValue],
    ) -> Result<RuntimeValue, ExprError> {
        let width = self.number_arg(args, 0, 0.2)?;
        let samples_f = self.number_arg(args, 1, 5.0)?;
        let t = self.number_arg(args, 2, self.effective_time())?;
        let samples = if samples_f.is_finite() {
            (samples_f.floor() as isize).clamp(1, 256) as usize
        } else {
            1
        };
        if samples <= 1 || !width.is_finite() || width <= 0.0 {
            let val = self.read_property_value_at_time(id, t)?;
            return self.intern_value(val);
        }
        self.charge_work(samples)?;
        let half = width * 0.5;
        let denom = (samples - 1) as f64;
        let mut vals = Vec::with_capacity(samples);
        for i in 0..samples {
            let sample_t = (t - half) + width * (i as f64) / denom;
            vals.push(self.read_property_value_at_time(id, sample_t)?);
        }
        let avg = value_average(&vals);
        self.intern_value(avg)
    }

    fn builtin_focus_distance_to(
        &mut self,
        args: &[RuntimeValue],
    ) -> Result<RuntimeValue, ExprError> {
        let Some(first) = args.first().cloned() else {
            return Ok(RuntimeValue::Number(0.0));
        };
        let target = self.deref_property_handle(first)?;
        if matches!(target, RuntimeValue::Null | RuntimeValue::Undefined) {
            return Ok(RuntimeValue::Number(0.0));
        }
        let origin = self.ctx.world_position(None).unwrap_or([0.0, 0.0, 0.0]);
        let target_pos = if let Some(vec) = self.runtime_as_f64_vec(&target)? {
            [
                vec.first().copied().unwrap_or(0.0),
                vec.get(1).copied().unwrap_or(0.0),
                vec.get(2).copied().unwrap_or(0.0),
            ]
        } else if let RuntimeValue::Handle(
            HandleRef::Object(oid) | HandleRef::Layer(oid) | HandleRef::Clip(oid),
        ) = target
        {
            match self.ctx.world_position(Some(&oid)) {
                Ok(pos) => pos,
                Err(_) => return Ok(RuntimeValue::Number(0.0)),
            }
        } else {
            let name = self.runtime_to_js_string(&target)?;
            let target_id = self
                .ctx
                .object(None, &name)
                .or_else(|_| self.ctx.layer(&NameOrIndex::Name(name)));
            match target_id {
                Ok(oid) => match self.ctx.world_position(Some(&oid)) {
                    Ok(pos) => pos,
                    Err(_) => return Ok(RuntimeValue::Number(0.0)),
                },
                Err(_) => return Ok(RuntimeValue::Number(0.0)),
            }
        };
        let dx = target_pos[0] - origin[0];
        let dy = target_pos[1] - origin[1];
        let dz = target_pos[2] - origin[2];
        Ok(RuntimeValue::Number((dx * dx + dy * dy + dz * dz).sqrt()))
    }

    fn builtin_audio(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, ExprError> {
        let band = match args.first() {
            Some(v) if !matches!(v, RuntimeValue::Undefined) => self.export_root_value(v)?,
            _ => Value::String("all".to_string()),
        };
        let t = self.number_arg(args, 1, self.effective_time())?;
        let energy = self.ctx.audio_band(&band, t);
        Ok(RuntimeValue::Number(if energy.is_finite() {
            energy
        } else {
            0.0
        }))
    }

    fn builtin_audio_level(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, ExprError> {
        let t = self.number_arg(args, 0, self.effective_time())?;
        let level = self.ctx.audio_level(t);
        Ok(RuntimeValue::Number(if level.is_finite() {
            level
        } else {
            0.0
        }))
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

// ---------------------------------------------------------------------------
// Deterministic Perlin Gradient Noise & Value Vector Helpers
// ---------------------------------------------------------------------------

const PERLIN_PERM: [u8; 256] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219,
    203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230,
    220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76,
    132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
    186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
    59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
    70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
    241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180,
];

#[inline]
fn perlin_fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn perlin_lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}

#[inline]
fn perlin_hash(idx: usize) -> usize {
    usize::from(PERLIN_PERM[idx & 255])
}

#[inline]
fn perlin_grad3(hash: usize, x: f64, y: f64, z: f64) -> f64 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    let a = if (h & 1) == 0 { u } else { -u };
    let b = if (h & 2) == 0 { v } else { -v };
    a + b
}

fn perlin_noise_3d_raw(x: f64, y: f64, z: f64) -> f64 {
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return 0.0;
    }
    let xf_floor = x.floor();
    let yf_floor = y.floor();
    let zf_floor = z.floor();
    let xi = (xf_floor as i64).rem_euclid(256) as usize;
    let yi = (yf_floor as i64).rem_euclid(256) as usize;
    let zi = (zf_floor as i64).rem_euclid(256) as usize;

    let xf = x - xf_floor;
    let yf = y - yf_floor;
    let zf = z - zf_floor;

    let u = perlin_fade(xf);
    let v = perlin_fade(yf);
    let w = perlin_fade(zf);

    let a = perlin_hash(xi) + yi;
    let aa = perlin_hash(a) + zi;
    let ab = perlin_hash(a + 1) + zi;
    let b = perlin_hash(xi + 1) + yi;
    let ba = perlin_hash(b) + zi;
    let bb = perlin_hash(b + 1) + zi;

    let x1 = perlin_lerp(
        u,
        perlin_grad3(perlin_hash(aa), xf, yf, zf),
        perlin_grad3(perlin_hash(ba), xf - 1.0, yf, zf),
    );
    let x2 = perlin_lerp(
        u,
        perlin_grad3(perlin_hash(ab), xf, yf - 1.0, zf),
        perlin_grad3(perlin_hash(bb), xf - 1.0, yf - 1.0, zf),
    );
    let y1 = perlin_lerp(v, x1, x2);

    let x3 = perlin_lerp(
        u,
        perlin_grad3(perlin_hash(aa + 1), xf, yf, zf - 1.0),
        perlin_grad3(perlin_hash(ba + 1), xf - 1.0, yf, zf - 1.0),
    );
    let x4 = perlin_lerp(
        u,
        perlin_grad3(perlin_hash(ab + 1), xf, yf - 1.0, zf - 1.0),
        perlin_grad3(perlin_hash(bb + 1), xf - 1.0, yf - 1.0, zf - 1.0),
    );
    let y2 = perlin_lerp(v, x3, x4);

    perlin_lerp(w, y1, y2).clamp(-1.0, 1.0)
}

fn perlin_noise_1d(x: f64) -> f64 {
    perlin_noise_3d_raw(x + 0.137, 0.293, 0.419)
}

fn perlin_noise_2d(x: f64, y: f64) -> f64 {
    perlin_noise_3d_raw(x + 0.137, y + 0.293, 0.419)
}

fn perlin_noise_3d(x: f64, y: f64, z: f64) -> f64 {
    perlin_noise_3d_raw(x + 0.137, y + 0.293, z + 0.419)
}

fn value_sub(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => Value::Number(x - y),
        (Value::Array(va), Value::Array(vb)) => {
            let len = va.len().max(vb.len());
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let x = va.get(i).and_then(Value::as_f64).unwrap_or(0.0);
                let y = vb.get(i).and_then(Value::as_f64).unwrap_or(0.0);
                out.push(Value::Number(x - y));
            }
            Value::Array(out)
        }
        _ => Value::Number(a.to_number() - b.to_number()),
    }
}

fn value_scale(v: &Value, s: f64) -> Value {
    match v {
        Value::Number(x) => Value::Number(x * s),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| Value::Number(item.to_number() * s))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn value_add_scaled(base: &Value, delta: &Value, scale: f64) -> Value {
    match (base, delta) {
        (Value::Number(x), Value::Number(d)) => Value::Number(x + d * scale),
        (Value::Array(vb), Value::Array(vd)) => {
            let len = vb.len().max(vd.len());
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let x = vb.get(i).and_then(Value::as_f64).unwrap_or(0.0);
                let d = vd.get(i).and_then(Value::as_f64).unwrap_or(0.0);
                out.push(Value::Number(x + d * scale));
            }
            Value::Array(out)
        }
        (Value::Array(vb), Value::Number(d)) => Value::Array(
            vb.iter()
                .map(|item| Value::Number(item.to_number() + d * scale))
                .collect(),
        ),
        _ => base.clone(),
    }
}

fn value_average(samples: &[Value]) -> Value {
    let Some(first) = samples.first() else {
        return Value::Number(0.0);
    };
    let inv_n = 1.0 / (samples.len() as f64);
    match first {
        Value::Number(_) => {
            let sum: f64 = samples.iter().map(Value::to_number).sum();
            Value::Number(sum * inv_n)
        }
        Value::Array(first_items) => {
            let len = first_items.len();
            let mut sums = vec![0.0_f64; len];
            for sample in samples {
                if let Value::Array(items) = sample {
                    for (acc, item) in sums.iter_mut().zip(items.iter()) {
                        *acc += item.to_number();
                    }
                }
            }
            Value::Array(sums.into_iter().map(|s| Value::Number(s * inv_n)).collect())
        }
        _ => samples[samples.len() / 2].clone(),
    }
}
