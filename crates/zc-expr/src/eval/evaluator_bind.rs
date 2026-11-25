use super::*;

impl<'a, 'ast> Evaluator<'a, 'ast> {
    pub(super) fn bind_pattern(
        &mut self,
        pattern: &'ast Pattern,
        val: RuntimeValue,
        env: usize,
        kind: VarKind,
    ) -> Result<(), ExprError> {
        let val = self.deref_property_handle(val)?;
        match pattern {
            Pattern::Ident(name) => {
                self.define_binding(env, name.clone(), val, kind)?;
                Ok(())
            }
            Pattern::Array(elems) => {
                match val {
                    RuntimeValue::Array(id) => {
                        for (i, elem) in elems.iter().enumerate() {
                            match elem {
                                ArrayPatternElem::Hole => {}
                                ArrayPatternElem::Binding { pattern, default } => {
                                    let mut item = self.arrays[id]
                                        .get(i)
                                        .cloned()
                                        .unwrap_or(RuntimeValue::Undefined);
                                    if matches!(item, RuntimeValue::Undefined)
                                        && let Some(def_expr) = default
                                    {
                                        item = self.eval_expr(def_expr, env)?;
                                    }
                                    self.bind_pattern(pattern, item, env, kind)?;
                                }
                                ArrayPatternElem::Rest(name) => {
                                    let rest_len = self.arrays[id].len().saturating_sub(i);
                                    self.check_array_alloc(rest_len)?;
                                    self.charge_work(rest_len)?;
                                    let rest_items = if i < self.arrays[id].len() {
                                        self.arrays[id][i..].to_vec()
                                    } else {
                                        Vec::new()
                                    };
                                    let arr = self.alloc_array(rest_items);
                                    self.define_binding(env, name.clone(), arr, kind)?;
                                }
                            }
                        }
                    }
                    RuntimeValue::String(s) => {
                        // Iterate string characters lazily by UTF-8 byte cursor so destructuring
                        // a prefix like `const [c] = s` on a 4 MiB string is O(1).
                        let mut byte_pos = 0_usize;
                        for elem in elems {
                            match elem {
                                ArrayPatternElem::Hole => {
                                    if let Some(ch) = s.as_str()[byte_pos..].chars().next() {
                                        byte_pos += ch.len_utf8();
                                    }
                                }
                                ArrayPatternElem::Binding { pattern, default } => {
                                    let mut item =
                                        if let Some(ch) = s.as_str()[byte_pos..].chars().next() {
                                            byte_pos += ch.len_utf8();
                                            RuntimeValue::String(JsStr::from(ch.to_string()))
                                        } else {
                                            RuntimeValue::Undefined
                                        };
                                    if matches!(item, RuntimeValue::Undefined)
                                        && let Some(def_expr) = default
                                    {
                                        item = self.eval_expr(def_expr, env)?;
                                    }
                                    self.bind_pattern(pattern, item, env, kind)?;
                                }
                                ArrayPatternElem::Rest(name) => {
                                    let rem = &s.as_str()[byte_pos..];
                                    let rest_len = if s.is_ascii {
                                        rem.len()
                                    } else {
                                        self.charge_byte_work(rem.len())?;
                                        rem.chars().count()
                                    };
                                    self.check_array_alloc(rest_len)?;
                                    self.check_string_alloc(rem.len())?;
                                    self.charge_byte_work(rem.len())?;
                                    let rest_items: Vec<RuntimeValue> = rem
                                        .chars()
                                        .map(|c| RuntimeValue::String(JsStr::from(c.to_string())))
                                        .collect();
                                    byte_pos = s.len();
                                    let arr = self.alloc_array(rest_items);
                                    self.define_binding(env, name.clone(), arr, kind)?;
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(ExprError::TypeError(
                            "cannot destructure non-iterable value as array".to_string(),
                        ));
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
                                let n = self.objects[id].len();
                                let key_bytes: usize = self.objects[id]
                                    .keys()
                                    .filter(|k| !used_keys.contains(k))
                                    .map(String::len)
                                    .sum();
                                self.check_array_alloc(n)?;
                                self.charge_alloc_bytes(key_bytes)?;
                                self.charge_work(n)?;
                                let entries = self.objects[id].clone();
                                for (k, v) in entries {
                                    if !used_keys.contains(&k) {
                                        rest.insert(k, v);
                                    }
                                }
                            }
                            let obj = self.alloc_object(rest);
                            self.define_binding(env, name.clone(), obj, kind)?;
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

    pub(super) fn exec_stmt(&mut self, stmt: &'ast Stmt, env: usize) -> Result<Flow, ExprError> {
        self.tick()?;
        match stmt {
            Stmt::Empty => Ok(Flow::Normal(RuntimeValue::Empty)),
            Stmt::Expr(expr) => Ok(Flow::Normal(self.eval_expr(expr, env)?)),
            Stmt::Block(stmts) => {
                let block_env = self.push_env(Some(env))?;
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
                let switch_env = self.push_env(Some(env))?;
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

    pub(super) fn exec_stmt_list(
        &mut self,
        stmts: &'ast [Stmt],
        env: usize,
    ) -> Result<Flow, ExprError> {
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
        for_stmt: &'ast crate::ast::ForStmt,
        env: usize,
    ) -> Result<Flow, ExprError> {
        let mut is_lexical = false;
        let mut loop_env = env;

        if let Some(init) = &for_stmt.init {
            match init {
                ForInit::VarDecl { kind, decls } => {
                    if *kind != VarKind::Var {
                        is_lexical = true;
                        loop_env = self.push_env(Some(env))?;
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
                self.clone_lexical_env(current_env, env)?
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
                current_env = self.clone_lexical_env(body_env, env)?;
            }
            if let Some(update) = &for_stmt.update {
                self.eval_expr(update, current_env)?;
            }
        }
        Ok(Flow::Normal(last))
    }

    fn clone_lexical_env(&mut self, src_env: usize, parent: usize) -> Result<usize, ExprError> {
        let count = self.envs[src_env].bindings.len();
        let name_bytes: usize = self.envs[src_env].bindings.keys().map(String::len).sum();
        self.charge_alloc_bytes(
            64_usize
                .saturating_add(count.saturating_mul(64))
                .saturating_add(name_bytes),
        )?;
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
        Ok(id)
    }

    fn exec_for_of_stmt(
        &mut self,
        for_of: &'ast crate::ast::ForOfStmt,
        env: usize,
    ) -> Result<Flow, ExprError> {
        let iter_val = self.eval_expr(&for_of.iterable, env)?;
        let iter_val = self.deref_property_handle(iter_val)?;
        if !matches!(iter_val, RuntimeValue::Array(_) | RuntimeValue::String(_)) {
            return Err(ExprError::TypeError(
                "value is not iterable in for..of loop".to_string(),
            ));
        }

        let mut last = RuntimeValue::Empty;
        let mut idx = 0_usize;
        let mut byte_pos = 0_usize;
        loop {
            let item = match &iter_val {
                RuntimeValue::Array(id) => {
                    let Some(elem) = self.arrays[*id].get(idx).cloned() else {
                        break;
                    };
                    idx += 1;
                    elem
                }
                RuntimeValue::String(s) => {
                    let Some(ch) = s.as_str()[byte_pos..].chars().next() else {
                        break;
                    };
                    byte_pos += ch.len_utf8();
                    RuntimeValue::String(JsStr::from(ch.to_string()))
                }
                _ => unreachable!(),
            };
            self.tick()?;
            let iter_env = match for_of.kind {
                Some(VarKind::Let | VarKind::Const) => self.push_env(Some(env))?,
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
        for_in: &'ast crate::ast::ForInStmt,
        env: usize,
    ) -> Result<Flow, ExprError> {
        let obj_val = self.eval_expr(&for_in.object, env)?;
        let obj_val = self.deref_property_handle(obj_val)?;

        enum ForInKeys {
            Object(Vec<JsStr>),
            ArrayLen(usize),
            Empty,
        }

        let source = match obj_val {
            RuntimeValue::Object(id) => {
                let key_count = self.objects[id].len();
                let key_bytes: usize = self.objects[id].keys().map(String::len).sum();
                self.check_array_alloc(key_count)?;
                self.charge_alloc_bytes(key_bytes)?;
                self.charge_work(key_count)?;
                self.charge_byte_work(key_bytes)?;
                let snapshot: Vec<JsStr> = self.objects[id]
                    .keys()
                    .map(|k| JsStr::from(k.as_str()))
                    .collect();
                ForInKeys::Object(snapshot)
            }
            RuntimeValue::Array(id) => ForInKeys::ArrayLen(self.arrays[id].len()),
            RuntimeValue::Null | RuntimeValue::Undefined => ForInKeys::Empty,
            _ => ForInKeys::Empty,
        };

        let mut last = RuntimeValue::Empty;
        let len = match &source {
            ForInKeys::Object(keys) => keys.len(),
            ForInKeys::ArrayLen(n) => *n,
            ForInKeys::Empty => 0,
        };
        for idx in 0..len {
            self.tick()?;
            let iter_env = match for_in.kind {
                Some(VarKind::Let | VarKind::Const) => self.push_env(Some(env))?,
                _ => env,
            };
            let kv = match &source {
                ForInKeys::Object(keys) => RuntimeValue::String(keys[idx].clone()),
                ForInKeys::ArrayLen(_) => RuntimeValue::String(JsStr::from(idx.to_string())),
                ForInKeys::Empty => unreachable!(),
            };
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
        try_stmt: &'ast crate::ast::TryStmt,
        env: usize,
    ) -> Result<Flow, ExprError> {
        let try_env = self.push_env(Some(env))?;
        let try_res = match self.exec_stmt_list(&try_stmt.block, try_env) {
            Err(ExprError::Thrown(_)) if self.pending_throw.is_some() => {
                let thrown = self.pending_throw.take().unwrap_or(RuntimeValue::Undefined);
                Ok(Flow::Throw(thrown))
            }
            other => other,
        };

        let mut outcome: Result<Flow, ExprError> = match try_res {
            Ok(Flow::Throw(thrown)) => {
                if let Some(catch_body) = &try_stmt.catch_body {
                    let catch_env = self.push_env(Some(env))?;
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
                    let catch_env = self.push_env(Some(env))?;
                    if let Some(pat) = &try_stmt.catch_param {
                        let msg = err.to_string();
                        self.check_string_alloc(msg.len())?;
                        self.charge_alloc_bytes(ARRAY_ELEM_BUDGET_BYTES)?;
                        let mut err_obj = BTreeMap::new();
                        err_obj.insert(
                            "message".to_string(),
                            RuntimeValue::String(JsStr::from(msg)),
                        );
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

        if matches!(&outcome, Err(ExprError::Thrown(_)))
            && let Some(thrown) = self.pending_throw.take()
        {
            outcome = Ok(Flow::Throw(thrown));
        }

        if matches!(&outcome, Err(err) if !is_catchable_error(err)) {
            return outcome;
        }

        if let Some(finally_body) = &try_stmt.finally_body {
            let finally_env = self.push_env(Some(env))?;
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
}
