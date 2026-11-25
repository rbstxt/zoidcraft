use super::*;

impl<'a, 'ast> Evaluator<'a, 'ast> {
    pub(super) fn call_object_handle_method(
        &mut self,
        handle: &HandleRef,
        method: &str,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        let owner_id: Option<HandleId> = match handle {
            HandleRef::Effect(id)
            | HandleRef::Object(id)
            | HandleRef::Layer(id)
            | HandleRef::Clip(id) => Some(id.clone()),
            HandleRef::Sequence => self.ctx.this_sequence_id(),
            _ => None,
        };
        match method {
            "param" => {
                let name = self.expect_string_arg(&args, 0, "param")?;
                let pid = self.resolve_object_handle_property(handle, &name)?;
                let val = self.read_property_value(&pid)?;
                self.intern_value(val)
            }
            "property" => {
                let name = self.expect_string_arg(&args, 0, "property")?;
                let pid = self.resolve_object_handle_property(handle, &name)?;
                Ok(RuntimeValue::Handle(HandleRef::Property(pid)))
            }
            "parent" => match owner_id {
                Some(ref id) => self.resolve_parent_call(id, &args),
                None => Ok(RuntimeValue::Null),
            },
            "effect" => {
                let name = self.expect_string_arg(&args, 0, "effect")?;
                let eid = self.ctx.effect(owner_id.as_ref(), &name)?;
                Ok(RuntimeValue::Handle(HandleRef::Effect(eid)))
            }
            "object" => {
                let name = self.expect_string_arg(&args, 0, "object")?;
                let oid = self.ctx.object(owner_id.as_ref(), &name)?;
                Ok(RuntimeValue::Handle(HandleRef::Object(oid)))
            }
            "content" => {
                let name = self.expect_string_arg(&args, 0, "content")?;
                let cid = self.ctx.content(owner_id.as_ref(), &name)?;
                Ok(RuntimeValue::Handle(HandleRef::Object(cid)))
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
                let out = self.ctx.to_world(owner_id.as_ref(), pt)?;
                self.alloc_f64_vec(out.to_vec())
            }
            "fromWorld" => {
                let pt = match args.first() {
                    Some(v) => self.runtime_to_vec3(v)?,
                    None => [0.0, 0.0, 0.0],
                };
                let out = self.ctx.from_world(owner_id.as_ref(), pt)?;
                self.alloc_f64_vec(out.to_vec())
            }
            "worldPosition" => {
                let out = self.ctx.world_position(owner_id.as_ref())?;
                self.alloc_f64_vec(out.to_vec())
            }
            "worldRotation" => {
                let out = self.ctx.world_rotation(owner_id.as_ref())?;
                self.alloc_f64_vec(out.to_vec())
            }
            "toScreen" => {
                let pt = match args.first() {
                    Some(v) => self.runtime_to_vec3(v)?,
                    None => [0.0, 0.0, 0.0],
                };
                let out = self.ctx.to_screen(owner_id.as_ref(), pt)?;
                self.alloc_f64_vec(out.to_vec())
            }
            "fromScreen" => {
                let pt = match args.first() {
                    Some(v) => self.runtime_to_vec3(v)?,
                    None => [0.0, 0.0, 0.0],
                };
                let out = self.ctx.from_screen(owner_id.as_ref(), pt)?;
                self.alloc_f64_vec(out.to_vec())
            }
            _ => Err(ExprError::TypeError(format!(
                "handle.{method} is not a function"
            ))),
        }
    }

    pub(super) fn resolve_parent_call(
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

    pub(super) fn expect_string_arg(
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

    pub(super) fn parse_name_or_index_arg(
        &mut self,
        args: &[RuntimeValue],
    ) -> Result<NameOrIndex, ExprError> {
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

    pub(super) fn call_array_method(
        &mut self,
        id: usize,
        method: &str,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        match method {
            "push" => {
                let next_len = self.arrays[id]
                    .len()
                    .checked_add(args.len())
                    .ok_or_else(|| ExprError::RangeError("array length overflow".to_string()))?;
                if next_len > MAX_ARRAY_LEN {
                    return Err(ExprError::RangeError(format!(
                        "array length {next_len} exceeds {MAX_ARRAY_LEN} limit"
                    )));
                }
                self.check_array_alloc(args.len())?;
                self.arrays[id].extend(args);
                Ok(RuntimeValue::Number(self.arrays[id].len() as f64))
            }
            "pop" => Ok(self.arrays[id].pop().unwrap_or(RuntimeValue::Undefined)),
            "shift" => {
                if self.arrays[id].is_empty() {
                    Ok(RuntimeValue::Undefined)
                } else {
                    self.charge_work(self.arrays[id].len())?;
                    Ok(self.arrays[id].remove(0))
                }
            }
            "unshift" => {
                let next_len = self.arrays[id]
                    .len()
                    .checked_add(args.len())
                    .ok_or_else(|| ExprError::RangeError("array length overflow".to_string()))?;
                if next_len > MAX_ARRAY_LEN {
                    return Err(ExprError::RangeError(format!(
                        "array length {next_len} exceeds {MAX_ARRAY_LEN} limit"
                    )));
                }
                self.check_array_alloc(args.len())?;
                self.charge_work(next_len)?;
                let mut new_vec = args;
                new_vec.append(&mut self.arrays[id]);
                self.arrays[id] = new_vec;
                Ok(RuntimeValue::Number(self.arrays[id].len() as f64))
            }
            "reverse" => {
                self.charge_work(self.arrays[id].len())?;
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
                let slice_len = end.saturating_sub(start);
                self.check_array_alloc(slice_len)?;
                self.charge_work(slice_len)?;
                let slice = if start < end {
                    self.arrays[id][start..end].to_vec()
                } else {
                    Vec::new()
                };
                Ok(self.alloc_array(slice))
            }
            "concat" => {
                let mut total_len = self.arrays[id].len();
                let mut resolved = Vec::with_capacity(args.len());
                for a in args {
                    let d = self.deref_property_handle(a)?;
                    let add = match &d {
                        RuntimeValue::Array(other_id) => self.arrays[*other_id].len(),
                        _ => 1,
                    };
                    total_len = total_len.checked_add(add).ok_or_else(|| {
                        ExprError::RangeError("array length overflow".to_string())
                    })?;
                    if total_len > MAX_ARRAY_LEN {
                        return Err(ExprError::RangeError(format!(
                            "array length {total_len} exceeds {MAX_ARRAY_LEN} limit"
                        )));
                    }
                    resolved.push(d);
                }
                self.check_array_alloc(total_len)?;
                self.charge_work(total_len)?;
                let mut out = Vec::with_capacity(total_len);
                out.extend(self.arrays[id].iter().cloned());
                for d in resolved {
                    match d {
                        RuntimeValue::Array(other_id) => {
                            out.extend(self.arrays[other_id].iter().cloned());
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
                Ok(RuntimeValue::String(JsStr::from(
                    self.join_array_to_string(id, &sep)?,
                )))
            }
            "indexOf" => {
                let needle = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let len = self.arrays[id].len();
                let start = match args.get(1) {
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        normalize_slice_index(self.runtime_to_number(v)?, len as isize)
                    }
                    _ => 0,
                };
                self.charge_work(len.saturating_sub(start))?;
                for i in start..len {
                    let Some(item) = self.arrays[id].get(i).cloned() else {
                        break;
                    };
                    if self.strict_eq(&item, &needle)? {
                        return Ok(RuntimeValue::Number(i as f64));
                    }
                }
                Ok(RuntimeValue::Number(-1.0))
            }
            "includes" => {
                let needle = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let len = self.arrays[id].len();
                let start = match args.get(1) {
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        normalize_slice_index(self.runtime_to_number(v)?, len as isize)
                    }
                    _ => 0,
                };
                self.charge_work(len.saturating_sub(start))?;
                for i in start..len {
                    let Some(item) = self.arrays[id].get(i).cloned() else {
                        break;
                    };
                    if self.same_value_zero(&item, &needle)? {
                        return Ok(RuntimeValue::Bool(true));
                    }
                }
                Ok(RuntimeValue::Bool(false))
            }
            "map" => {
                let cb = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let len = self.arrays[id].len();
                self.check_array_alloc(len)?;
                self.charge_work(len)?;
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let Some(item) = self.arrays[id].get(i).cloned() else {
                        break;
                    };
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
                let len = self.arrays[id].len();
                self.charge_work(len)?;
                let mut out = Vec::new();
                for i in 0..len {
                    let Some(item) = self.arrays[id].get(i).cloned() else {
                        break;
                    };
                    let keep = self.call_runtime_value(
                        cb.clone(),
                        vec![
                            item.clone(),
                            RuntimeValue::Number(i as f64),
                            RuntimeValue::Array(id),
                        ],
                    )?;
                    if self.runtime_to_bool(&keep)? {
                        self.charge_alloc_bytes(ARRAY_ELEM_BUDGET_BYTES)?;
                        out.push(item);
                    }
                }
                Ok(self.alloc_array(out))
            }
            "reduce" => {
                let cb = args.first().cloned().unwrap_or(RuntimeValue::Undefined);
                let len = self.arrays[id].len();
                self.charge_work(len)?;
                let (mut acc, start_idx) = if let Some(init) = args.get(1) {
                    (init.clone(), 0)
                } else if let Some(first) = self.arrays[id].first().cloned() {
                    (first, 1)
                } else {
                    return Err(ExprError::TypeError(
                        "reduce of empty array with no initial value".to_string(),
                    ));
                };
                for i in start_idx..len {
                    let Some(item) = self.arrays[id].get(i).cloned() else {
                        break;
                    };
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
                let len = self.arrays[id].len();
                self.charge_work(len)?;
                for i in 0..len {
                    let Some(item) = self.arrays[id].get(i).cloned() else {
                        break;
                    };
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
                let len = self.arrays[id].len();
                self.charge_work(len)?;
                for i in 0..len {
                    let Some(item) = self.arrays[id].get(i).cloned() else {
                        break;
                    };
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
                let len = self.arrays[id].len();
                self.charge_work(len)?;
                for i in 0..len {
                    let Some(item) = self.arrays[id].get(i).cloned() else {
                        break;
                    };
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
                let len = self.arrays[id].len();
                self.charge_work(len)?;
                for i in 0..len {
                    let Some(item) = self.arrays[id].get(i).cloned() else {
                        break;
                    };
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
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        normalize_slice_index(self.runtime_to_number(v)?, len)
                    }
                    _ => len as usize,
                };
                if start < end {
                    self.charge_work(end - start)?;
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

    pub(super) fn call_string_method(
        &mut self,
        s: &JsStr,
        method: &str,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        match method {
            "toLowerCase" => {
                self.check_string_alloc(s.len())?;
                self.charge_byte_work(s.len())?;
                let out = s.to_lowercase();
                if out.len() > s.len() {
                    if out.len() > MAX_STRING_BYTES {
                        return Err(ExprError::RangeError(format!(
                            "string length {} exceeds {MAX_STRING_BYTES} byte limit",
                            out.len()
                        )));
                    }
                    self.charge_alloc_bytes(out.len() - s.len())?;
                }
                Ok(RuntimeValue::String(JsStr::from(out)))
            }
            "toUpperCase" => {
                self.check_string_alloc(s.len())?;
                self.charge_byte_work(s.len())?;
                let out = s.to_uppercase();
                if out.len() > s.len() {
                    if out.len() > MAX_STRING_BYTES {
                        return Err(ExprError::RangeError(format!(
                            "string length {} exceeds {MAX_STRING_BYTES} byte limit",
                            out.len()
                        )));
                    }
                    self.charge_alloc_bytes(out.len() - s.len())?;
                }
                Ok(RuntimeValue::String(JsStr::from(out)))
            }
            "trim" => {
                self.charge_byte_work(s.len())?;
                let t = s.trim();
                self.check_string_alloc(t.len())?;
                Ok(RuntimeValue::String(JsStr::from(t)))
            }
            "trimStart" => {
                self.charge_byte_work(s.len())?;
                let t = s.trim_start();
                self.check_string_alloc(t.len())?;
                Ok(RuntimeValue::String(JsStr::from(t)))
            }
            "trimEnd" => {
                self.charge_byte_work(s.len())?;
                let t = s.trim_end();
                self.check_string_alloc(t.len())?;
                Ok(RuntimeValue::String(JsStr::from(t)))
            }
            "slice" => {
                let char_len = self.string_char_len(s)? as isize;
                let start = match args.first() {
                    Some(v) => normalize_slice_index(self.runtime_to_number(v)?, char_len),
                    None => 0,
                };
                let end = match args.get(1) {
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        normalize_slice_index(self.runtime_to_number(v)?, char_len)
                    }
                    _ => char_len as usize,
                };
                if start < end {
                    let b_lo = Self::string_byte_offset_for_char_index(s, start);
                    let b_hi = Self::string_byte_offset_for_char_index(s, end);
                    let slice = &s.as_str()[b_lo..b_hi];
                    self.check_string_alloc(slice.len())?;
                    self.charge_byte_work(slice.len())?;
                    Ok(RuntimeValue::String(JsStr::from(slice)))
                } else {
                    Ok(RuntimeValue::String(JsStr::from("")))
                }
            }
            "substring" => {
                let char_len = self.string_char_len(s)?;
                let a = match args.first() {
                    Some(v) => clamp_f64(self.runtime_to_number(v)?, 0.0, char_len as f64) as usize,
                    None => 0,
                };
                let b = match args.get(1) {
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        clamp_f64(self.runtime_to_number(v)?, 0.0, char_len as f64) as usize
                    }
                    _ => char_len,
                };
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                let b_lo = Self::string_byte_offset_for_char_index(s, lo);
                let b_hi = Self::string_byte_offset_for_char_index(s, hi);
                let slice = &s.as_str()[b_lo..b_hi];
                self.check_string_alloc(slice.len())?;
                self.charge_byte_work(slice.len())?;
                Ok(RuntimeValue::String(JsStr::from(slice)))
            }
            "indexOf" => {
                let pat = match args.first() {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => "undefined".to_string(),
                };
                self.charge_byte_work(s.len().saturating_add(pat.len()))?;
                let idx = s
                    .find(&pat)
                    .map(|byte_idx| {
                        if s.is_ascii {
                            byte_idx as f64
                        } else {
                            s.as_str()[..byte_idx].chars().count() as f64
                        }
                    })
                    .unwrap_or(-1.0);
                Ok(RuntimeValue::Number(idx))
            }
            "includes" => {
                let pat = match args.first() {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => "undefined".to_string(),
                };
                self.charge_byte_work(s.len().saturating_add(pat.len()))?;
                Ok(RuntimeValue::Bool(s.contains(&pat)))
            }
            "startsWith" => {
                let pat = match args.first() {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => "undefined".to_string(),
                };
                self.charge_byte_work(pat.len())?;
                Ok(RuntimeValue::Bool(s.starts_with(&pat)))
            }
            "endsWith" => {
                let pat = match args.first() {
                    Some(v) => self.runtime_to_js_string(v)?,
                    None => "undefined".to_string(),
                };
                self.charge_byte_work(pat.len())?;
                Ok(RuntimeValue::Bool(s.ends_with(&pat)))
            }
            "split" => {
                let sep = match args.first() {
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        self.runtime_to_js_string(v)?
                    }
                    _ => {
                        self.check_string_alloc(s.len())?;
                        return Ok(self.alloc_array(vec![RuntimeValue::String(s.clone())]));
                    }
                };
                if sep.is_empty() {
                    let part_count = self.string_char_len(s)?;
                    self.check_array_alloc(part_count)?;
                    self.check_string_alloc(s.len())?;
                    self.charge_byte_work(s.len())?;
                    let parts: Vec<RuntimeValue> = s
                        .chars()
                        .map(|c| RuntimeValue::String(JsStr::from(c.to_string())))
                        .collect();
                    return Ok(self.alloc_array(parts));
                }
                self.charge_byte_work(s.len())?;
                let part_count = s.matches(&sep).count().saturating_add(1);
                self.check_array_alloc(part_count)?;
                self.check_string_alloc(s.len())?;
                let parts: Vec<RuntimeValue> = s
                    .split(&sep)
                    .map(|part| RuntimeValue::String(JsStr::from(part)))
                    .collect();
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
                self.charge_byte_work(s.len())?;
                let Some(pos) = s.find(&from) else {
                    return Ok(RuntimeValue::String(s.clone()));
                };
                let out_len = (s.len() - from.len())
                    .checked_add(to.len())
                    .ok_or_else(|| ExprError::RangeError("replace string overflow".to_string()))?;
                self.check_string_alloc(out_len)?;
                let mut out = String::with_capacity(out_len);
                out.push_str(&s[..pos]);
                out.push_str(&to);
                out.push_str(&s[pos + from.len()..]);
                Ok(RuntimeValue::String(JsStr::from(out)))
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
                self.charge_byte_work(s.len())?;
                let matches = if from.is_empty() {
                    self.string_char_len(s)?.saturating_add(1)
                } else {
                    s.matches(&from).count()
                };
                let added = matches.checked_mul(to.len()).ok_or_else(|| {
                    ExprError::RangeError("replaceAll string overflow".to_string())
                })?;
                let removed = matches.saturating_mul(from.len());
                let est = s
                    .len()
                    .saturating_sub(removed)
                    .checked_add(added)
                    .ok_or_else(|| {
                        ExprError::RangeError("replaceAll string overflow".to_string())
                    })?;
                self.check_string_alloc(est)?;
                Ok(RuntimeValue::String(JsStr::from(s.replace(&from, &to))))
            }
            "charAt" => {
                let idx = match args.first() {
                    Some(v) => self.runtime_to_number(v)?,
                    None => 0.0,
                };
                let out = if idx >= 0.0 && idx.is_finite() {
                    self.string_char_at(s, idx as usize)?
                        .map(|c| c.to_string())
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                Ok(RuntimeValue::String(JsStr::from(out)))
            }
            "charCodeAt" => {
                let idx = match args.first() {
                    Some(v) => self.runtime_to_number(v)?,
                    None => 0.0,
                };
                let out = if idx >= 0.0 && idx.is_finite() {
                    self.string_char_at(s, idx as usize)?
                        .map_or(f64::NAN, |c| (c as u32) as f64)
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
                let count_usize = count as usize;
                let total_bytes = s.len().checked_mul(count_usize).ok_or_else(|| {
                    ExprError::RangeError("string repeat length overflow".to_string())
                })?;
                self.check_string_alloc(total_bytes)?;
                self.charge_byte_work(total_bytes)?;
                Ok(RuntimeValue::String(JsStr::from(s.repeat(count_usize))))
            }
            _ => Err(ExprError::TypeError(format!(
                "String.prototype.{method} is not a function"
            ))),
        }
    }

    pub(super) fn call_number_method(
        &mut self,
        n: f64,
        method: &str,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, ExprError> {
        match method {
            "toFixed" => {
                let digits = match args.first() {
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        self.runtime_to_number(v)?
                    }
                    _ => 0.0,
                };
                let digits_int = if digits.is_nan() { 0.0 } else { digits.trunc() };
                if !(0.0..=100.0).contains(&digits_int) {
                    return Err(ExprError::RangeError(
                        "toFixed() digits argument must be between 0 and 100".to_string(),
                    ));
                }
                if !n.is_finite() {
                    return self.alloc_string(format_js_number(n));
                }
                let prec = digits_int as usize;
                self.alloc_string(format!("{n:.prec$}"))
            }
            "toString" => {
                let radix = match args.first() {
                    Some(v) if !matches!(v, RuntimeValue::Undefined) => {
                        let r = self.runtime_to_number(v)?;
                        if r.is_nan() {
                            return Err(ExprError::RangeError(
                                "toString() radix must be between 2 and 36".to_string(),
                            ));
                        }
                        let ri = r.trunc();
                        if !(2.0..=36.0).contains(&ri) {
                            return Err(ExprError::RangeError(
                                "toString() radix must be between 2 and 36".to_string(),
                            ));
                        }
                        ri as u32
                    }
                    _ => 10,
                };
                let formatted = if radix == 10 {
                    format_js_number(n)
                } else {
                    format_js_number_radix(n, radix)
                };
                self.alloc_string(formatted)
            }
            _ => Err(ExprError::TypeError(format!(
                "Number.prototype.{method} is not a function"
            ))),
        }
    }

    // -----------------------------------------------------------------------
    // Global Built-in Function Dispatch
    // -----------------------------------------------------------------------
}
