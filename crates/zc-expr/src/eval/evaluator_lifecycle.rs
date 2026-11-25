use super::*;

impl<'a, 'ast> Evaluator<'a, 'ast> {
    pub(super) fn new(ctx: &'a dyn ExprContext, config: EvalConfig, is_extended: bool) -> Self {
        Self {
            ctx,
            config,
            is_extended,
            steps: 0,
            recursion_depth: 0,
            allocated_bytes: 0,
            envs: Vec::new(),
            arrays: Vec::new(),
            objects: Vec::new(),
            closures: Vec::new(),
            literal_strings: HashMap::new(),
            rng_seed: 0.0,
            rng_timeless: false,
            rng_index: 0,
            rng_seeded: false,
            cached_value: None,
            cached_properties: None,
            pending_throw: None,
            stringifying_arrays: Vec::new(),
        }
    }

    pub(super) fn charge_work(&mut self, units: usize) -> Result<(), ExprError> {
        let delta = (units as u64).max(1);
        self.steps = self.steps.saturating_add(delta);
        if self.steps > self.config.max_steps {
            return Err(ExprError::StepLimitExceeded {
                limit: self.config.max_steps,
            });
        }
        Ok(())
    }

    /// Charge execution steps proportional to linear byte scans (`1 step per 8 bytes`, min 1).
    /// With default `max_steps = 10_000_000`, this caps total linear byte work at ~80 MB while
    /// keeping the 64 MiB heap budget (`MAX_HEAP_BYTES`) reachable for allocation bombs.
    pub(super) fn charge_byte_work(&mut self, bytes: usize) -> Result<(), ExprError> {
        self.charge_work((bytes / 8).max(1))
    }

    pub(super) fn intern_literal_string(
        &mut self,
        s: &'ast str,
    ) -> Result<RuntimeValue, ExprError> {
        let key = (s.as_ptr() as usize, s.len());
        if let Some(js) = self.literal_strings.get(&key) {
            return Ok(RuntimeValue::String(js.clone()));
        }
        self.check_string_alloc(s.len())?;
        let js = JsStr::from(s);
        self.literal_strings.insert(key, js.clone());
        Ok(RuntimeValue::String(js))
    }

    pub(super) fn string_char_len(&mut self, s: &JsStr) -> Result<usize, ExprError> {
        if s.is_ascii {
            Ok(s.len())
        } else {
            self.charge_byte_work(s.len())?;
            Ok(s.chars().count())
        }
    }

    pub(super) fn string_char_at(
        &mut self,
        s: &JsStr,
        idx: usize,
    ) -> Result<Option<char>, ExprError> {
        if s.is_ascii {
            Ok(s.as_bytes().get(idx).map(|&b| char::from(b)))
        } else {
            self.charge_byte_work(idx.min(s.len()))?;
            Ok(s.chars().nth(idx))
        }
    }

    pub(super) fn string_byte_offset_for_char_index(s: &JsStr, char_idx: usize) -> usize {
        if s.is_ascii {
            char_idx.min(s.len())
        } else {
            s.char_indices()
                .nth(char_idx)
                .map_or(s.len(), |(byte_idx, _)| byte_idx)
        }
    }

    pub(super) fn charge_alloc_bytes(&mut self, bytes: usize) -> Result<(), ExprError> {
        let next = self.allocated_bytes.saturating_add(bytes);
        if next > MAX_HEAP_BYTES {
            return Err(ExprError::RangeError(format!(
                "expression memory budget exceeded ({MAX_HEAP_BYTES} bytes)"
            )));
        }
        self.allocated_bytes = next;
        Ok(())
    }

    pub(super) fn check_string_alloc(&mut self, byte_len: usize) -> Result<(), ExprError> {
        if byte_len > MAX_STRING_BYTES {
            return Err(ExprError::RangeError(format!(
                "string length {byte_len} exceeds {MAX_STRING_BYTES} byte limit"
            )));
        }
        self.charge_alloc_bytes(byte_len)
    }

    pub(super) fn alloc_string(&mut self, s: String) -> Result<RuntimeValue, ExprError> {
        self.check_string_alloc(s.len())?;
        Ok(RuntimeValue::String(JsStr::from(s)))
    }

    pub(super) fn alloc_builtin_fn(&mut self, name: String) -> Result<RuntimeValue, ExprError> {
        self.check_string_alloc(name.len())?;
        Ok(RuntimeValue::BuiltinFn(BuiltinRef::Dynamic(Rc::from(name))))
    }

    pub(super) fn check_array_alloc(&mut self, elem_count: usize) -> Result<(), ExprError> {
        if elem_count > MAX_ARRAY_LEN {
            return Err(ExprError::RangeError(format!(
                "array length {elem_count} exceeds {MAX_ARRAY_LEN} limit"
            )));
        }
        let bytes = elem_count
            .checked_mul(ARRAY_ELEM_BUDGET_BYTES)
            .ok_or_else(|| ExprError::RangeError("array allocation overflow".to_string()))?;
        self.charge_alloc_bytes(bytes)
    }

    pub(super) fn tick(&mut self) -> Result<(), ExprError> {
        self.steps += 1;
        if self.steps > self.config.max_steps {
            return Err(ExprError::StepLimitExceeded {
                limit: self.config.max_steps,
            });
        }
        Ok(())
    }

    pub(super) fn push_env(&mut self, parent: Option<usize>) -> Result<usize, ExprError> {
        self.charge_alloc_bytes(64)?;
        let id = self.envs.len();
        self.envs.push(EnvFrame {
            parent,
            is_function_scope: parent.is_none(),
            bindings: BTreeMap::new(),
        });
        Ok(id)
    }

    pub(super) fn push_fn_env(&mut self, parent: usize) -> Result<usize, ExprError> {
        self.charge_alloc_bytes(64)?;
        let id = self.envs.len();
        self.envs.push(EnvFrame {
            parent: Some(parent),
            is_function_scope: true,
            bindings: BTreeMap::new(),
        });
        Ok(id)
    }

    pub(super) fn function_scope_of(&self, mut env: usize) -> usize {
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

    pub(super) fn define_binding(
        &mut self,
        env: usize,
        name: String,
        value: RuntimeValue,
        kind: VarKind,
    ) -> Result<(), ExprError> {
        let target_env = if kind == VarKind::Var {
            self.function_scope_of(env)
        } else {
            env
        };
        if !self.envs[target_env].bindings.contains_key(&name) {
            self.charge_alloc_bytes(ARRAY_ELEM_BUDGET_BYTES.saturating_add(name.len()))?;
        }
        self.envs[target_env].bindings.insert(
            name,
            Binding {
                value,
                mutable: kind != VarKind::Const,
                is_var: kind == VarKind::Var,
            },
        );
        Ok(())
    }

    pub(super) fn alloc_closure(&mut self, closure: Closure<'ast>) -> Result<usize, ExprError> {
        self.charge_alloc_bytes(64)?;
        let cid = self.closures.len();
        self.closures.push(closure);
        Ok(cid)
    }

    pub(super) fn lookup_local(&self, mut env: usize, name: &str) -> Option<RuntimeValue> {
        loop {
            if let Some(b) = self.envs[env].bindings.get(name) {
                return Some(b.value.clone());
            }
            env = self.envs[env].parent?;
        }
    }

    pub(super) fn assign_local(
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
}
