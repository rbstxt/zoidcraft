use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn rewrite_body(
    tokens: &[Token],
    uniforms: &HashMap<&str, &UniformDecl>,
    struct_uniforms: &HashMap<String, StructUniformDecl>,
    struct_defs: &HashMap<String, Vec<StructFieldDecl>>,
    global_const_ints: &HashMap<String, u32>,
    samplers: &HashSet<&str>,
    cube_samplers: &HashSet<String>,
    sampler_fns: &SamplerFnMap,
    user_fns: &HashSet<String>,
    frag_coord_res_y: FragCoordResY,
    state: &mut RewriteState,
    budget: &mut RewriteBudget,
) -> Result<Vec<Token>, ShaderError> {
    let ctx = RewriteCtx {
        uniforms,
        struct_uniforms,
        struct_defs,
        global_const_ints,
        samplers,
        cube_samplers,
        sampler_fns,
        user_fns,
        frag_coord_res_y,
    };
    let mut scopes: Vec<ScopeFrame> = vec![ScopeFrame::new(ScopeKind::Root)];
    let mut user_types: HashSet<String> = struct_defs.keys().cloned().collect();
    rewrite_slice(tokens, &ctx, &mut scopes, &mut user_types, state, 0, budget)
}

fn rewrite_slice(
    tokens: &[Token],
    ctx: &RewriteCtx<'_>,
    scopes: &mut Vec<ScopeFrame>,
    user_types: &mut HashSet<String>,
    state: &mut RewriteState,
    depth: usize,
    budget: &mut RewriteBudget,
) -> Result<Vec<Token>, ShaderError> {
    if depth > MAX_REWRITE_DEPTH {
        return Err(ShaderError::GlslParse(format!(
            "expression rewrite nesting depth exceeds {MAX_REWRITE_DEPTH} limit"
        )));
    }
    let mut out = Vec::with_capacity(tokens.len());
    let mut pending_scope_vars: Vec<String> = Vec::new();
    let mut pending_scope_samplers: Vec<String> = Vec::new();
    let mut pending_scope_cube_samplers: Vec<String> = Vec::new();
    let mut pending_decl_var: Option<String> = None;
    let mut decl_in_progress = false;
    let mut decl_expecting_name = false;
    let mut decl_is_const = false;
    let mut decl_type_is_int = false;
    let mut decl_paren_depth: usize = 0;
    let mut in_param_list = false;
    let mut param_paren_depth: usize = 0;
    let mut i = 0;

    while i < tokens.len() {
        budget.charge(1)?;
        let prev_is_dot = out
            .iter()
            .rev()
            .find(|t| !matches!(t, Token::Newline))
            .is_some_and(|t| matches!(t, Token::Punct(p) if p == "."));

        // Transition a `for (...)` waiting for its body into a single-statement body scope
        // when the body does not begin with `{`, `;`, or a nested `for`.
        if !matches!(&tokens[i], Token::Newline | Token::Preprocessor(_))
            && !matches!(&tokens[i], Token::Punct(p) if p == "{" || p == ";")
            && !matches!(&tokens[i], Token::Ident(kw) if kw == "for")
            && let Some(top) = scopes.last_mut()
        {
            match &mut top.kind {
                ScopeKind::ForBodyStart => {
                    let is_if = matches!(&tokens[i], Token::Ident(kw) if kw == "if");
                    top.kind = ScopeKind::ForSingleStmt {
                        if_depth: usize::from(is_if),
                    };
                }
                ScopeKind::ForSingleStmt { if_depth } => {
                    if matches!(&tokens[i], Token::Ident(kw) if kw == "if") {
                        *if_depth += 1;
                    }
                }
                _ => {}
            }
        }

        match &tokens[i] {
            Token::Preprocessor(line) => {
                out.push(Token::Preprocessor(rewrite_preprocessor_line(
                    line,
                    ctx,
                    state,
                    depth + 1,
                    budget,
                )?));
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "{" => {
                flush_pending_decl(&mut pending_decl_var, scopes);
                let mut new_scope = ScopeFrame::new(ScopeKind::Brace);
                for var in pending_scope_vars.drain(..) {
                    new_scope.vars.insert(var);
                }
                for samp in pending_scope_samplers.drain(..) {
                    new_scope.samplers.insert(samp);
                }
                for cube in pending_scope_cube_samplers.drain(..) {
                    new_scope.cube_samplers.insert(cube);
                }
                scopes.push(new_scope);
                decl_in_progress = false;
                decl_expecting_name = false;
                decl_is_const = false;
                decl_type_is_int = false;
                in_param_list = false;
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "}" => {
                flush_pending_decl(&mut pending_decl_var, scopes);
                if scopes.len() > 1 {
                    scopes.pop();
                }
                let next_i = next_non_newline(tokens, i + 1);
                let next_is_else =
                    matches!(tokens.get(next_i), Some(Token::Ident(kw)) if kw == "else");
                pop_completed_for_scopes(scopes, next_is_else);
                decl_in_progress = false;
                decl_expecting_name = false;
                decl_is_const = false;
                decl_type_is_int = false;
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == ";" => {
                flush_pending_decl(&mut pending_decl_var, scopes);
                decl_in_progress = false;
                decl_expecting_name = false;
                decl_is_const = false;
                decl_type_is_int = false;
                if !in_param_list {
                    pending_scope_vars.clear();
                    pending_scope_samplers.clear();
                    pending_scope_cube_samplers.clear();
                }
                let next_i = next_non_newline(tokens, i + 1);
                let next_is_else =
                    matches!(tokens.get(next_i), Some(Token::Ident(kw)) if kw == "else");
                pop_completed_for_scopes(scopes, next_is_else);
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "(" => {
                if in_param_list {
                    param_paren_depth += 1;
                } else {
                    if let Some(top) = scopes.last_mut()
                        && let ScopeKind::ForHeader { paren_depth } = &mut top.kind
                    {
                        *paren_depth += 1;
                    }
                    if decl_in_progress {
                        decl_paren_depth += 1;
                    }
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == ")" => {
                if in_param_list {
                    param_paren_depth = param_paren_depth.saturating_sub(1);
                    if param_paren_depth == 0 {
                        in_param_list = false;
                    }
                } else {
                    if decl_in_progress {
                        decl_paren_depth = decl_paren_depth.saturating_sub(1);
                    }
                    if let Some(top) = scopes.last_mut()
                        && let ScopeKind::ForHeader { paren_depth } = &mut top.kind
                    {
                        *paren_depth = paren_depth.saturating_sub(1);
                        if *paren_depth == 0 {
                            if let Some(var) = pending_decl_var.take() {
                                top.vars.insert(var);
                            }
                            decl_in_progress = false;
                            decl_expecting_name = false;
                            decl_is_const = false;
                            decl_type_is_int = false;
                            top.kind = ScopeKind::ForBodyStart;
                        }
                    }
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "," && decl_in_progress && decl_paren_depth == 0 => {
                flush_pending_decl(&mut pending_decl_var, scopes);
                decl_expecting_name = true;
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p)
                if p == "="
                    && decl_is_const
                    && decl_type_is_int
                    && decl_paren_depth == 0
                    && pending_decl_var.is_some() =>
            {
                // Peek initializer expression for local `const int/uint` so `[N]` array sizes
                // can be constant-folded for Naga 30.0.2.
                if let Some(var_name) = pending_decl_var.as_ref() {
                    let expr_start = next_non_newline(tokens, i + 1);
                    let mut scan = expr_start;
                    let mut pd: usize = 0;
                    let mut bd: usize = 0;
                    while scan < tokens.len() {
                        match &tokens[scan] {
                            Token::Punct(sp) if sp == "(" => pd += 1,
                            Token::Punct(sp) if sp == ")" => pd = pd.saturating_sub(1),
                            Token::Punct(sp) if sp == "[" => bd += 1,
                            Token::Punct(sp) if sp == "]" => bd = bd.saturating_sub(1),
                            Token::Punct(sp) if pd == 0 && bd == 0 && (sp == "," || sp == ";") => {
                                break;
                            }
                            _ => {}
                        }
                        scan += 1;
                    }
                    if let Some((val, _)) = eval_const_int_with_scopes(
                        &tokens[expr_start..scan],
                        scopes,
                        ctx.global_const_ints,
                    ) && let Some(current_scope) = scopes.last_mut()
                    {
                        current_scope.const_ints.insert(var_name.clone(), val);
                    }
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "[" && scopes.len() > 1 => {
                if let Some((inner, after_bracket)) = parse_bracket_expr(tokens, i)
                    && let Some((val, uses_local)) =
                        eval_const_int_with_scopes(&inner, scopes, ctx.global_const_ints)
                    && uses_local
                    && val >= 0
                {
                    out.push(Token::Punct("[".to_string()));
                    out.push(Token::Number(val.to_string()));
                    out.push(Token::Punct("]".to_string()));
                    budget.charge(2)?;
                    i = after_bracket;
                    continue;
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Ident(name) if !prev_is_dot && name == "const" => {
                decl_is_const = true;
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Ident(name) if name == "struct" => {
                out.push(tokens[i].clone());
                let next_i = next_non_newline(tokens, i + 1);
                if let Some(Token::Ident(struct_name)) = tokens.get(next_i) {
                    user_types.insert(struct_name.clone());
                }
                i += 1;
                continue;
            }
            Token::Ident(name) if name == "for" => {
                out.push(tokens[i].clone());
                let next_i = next_non_newline(tokens, i + 1);
                if matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "(") {
                    scopes.push(ScopeFrame::new(ScopeKind::ForHeader { paren_depth: 0 }));
                }
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && (name == "sampler2D" || name == "samplerCube")
                    && in_param_list
                    && param_paren_depth == 1 =>
            {
                let is_cube = name == "samplerCube";
                let glsl450_tex_ty = if is_cube { "textureCube" } else { "texture2D" };
                let next_i = next_non_newline(tokens, i + 1);
                if let Some(Token::Ident(var_name)) = tokens.get(next_i) {
                    pending_scope_vars.push(var_name.clone());
                    pending_scope_samplers.push(var_name.clone());
                    if is_cube {
                        pending_scope_cube_samplers.push(var_name.clone());
                    }
                    let esc_var = escape_glsl450_ident(var_name);
                    out.push(Token::Ident(glsl450_tex_ty.to_string()));
                    out.push(Token::Ident(esc_var.clone()));
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("sampler".to_string()));
                    out.push(Token::Ident(format!("{esc_var}_sampler")));
                    budget.charge(4)?;
                    i = next_i + 1;
                    continue;
                }
                out.push(Token::Ident(glsl450_tex_ty.to_string()));
                out.push(Token::Punct(",".to_string()));
                out.push(Token::Ident("sampler".to_string()));
                budget.charge(2)?;
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot && (is_glsl_builtin_type(name) || user_types.contains(name)) =>
            {
                let emitted_type = if user_types.contains(name) {
                    Token::Ident(escape_glsl450_ident(name))
                } else {
                    tokens[i].clone()
                };
                if decl_is_const && (name == "int" || name == "uint") {
                    decl_type_is_int = true;
                }
                let next_i = next_non_newline(tokens, i + 1);
                if let Some(Token::Ident(var_name)) = tokens.get(next_i) {
                    let after_name = next_non_newline(tokens, next_i + 1);
                    if matches!(tokens.get(after_name), Some(Token::Punct(p)) if p == "(") {
                        // Function declaration / definition: enter parameter list tracking.
                        in_param_list = true;
                        param_paren_depth = 0;
                        decl_is_const = false;
                        decl_type_is_int = false;
                        out.push(emitted_type);
                        i += 1;
                        continue;
                    }
                    // Local variable, struct field, or function parameter declaration.
                    if in_param_list {
                        pending_scope_vars.push(var_name.clone());
                    } else {
                        flush_pending_decl(&mut pending_decl_var, scopes);
                        pending_decl_var = Some(var_name.clone());
                    }
                    out.push(emitted_type);
                    out.push(Token::Ident(escape_glsl450_ident(var_name)));
                    decl_in_progress = !in_param_list;
                    decl_expecting_name = false;
                    decl_paren_depth = 0;
                    i = next_i + 1;
                    continue;
                }
                out.push(emitted_type);
                i += 1;
                continue;
            }
            Token::Ident(name) if decl_expecting_name => {
                flush_pending_decl(&mut pending_decl_var, scopes);
                pending_decl_var = Some(name.clone());
                decl_expecting_name = false;
                out.push(Token::Ident(escape_glsl450_ident(name)));
                i += 1;
                continue;
            }
            Token::Ident(name) if name == "gl_FragColor" && !prev_is_dot => {
                out.push(Token::Ident("_zc_frag_color".to_string()));
                i += 1;
                continue;
            }
            Token::Ident(name) if name == "gl_FragCoord" && !prev_is_dot => {
                let before_len = out.len();
                emit_gl_frag_coord(&mut out, ctx.frag_coord_res_y);
                budget.charge(out.len().saturating_sub(before_len))?;
                i += 1;
                continue;
            }
            Token::Ident(name) if name == "gl_PointCoord" && !prev_is_dot => {
                out.push(Token::Ident("vec2".to_string()));
                out.push(Token::Punct("(".to_string()));
                out.push(Token::Number("0.5".to_string()));
                out.push(Token::Punct(")".to_string()));
                budget.charge(3)?;
                i += 1;
                continue;
            }
            Token::Ident(name)
                if name == "bgCoord"
                    && !prev_is_dot
                    && !is_shadowed("bgCoord", scopes)
                    && !ctx.uniforms.contains_key("bgCoord")
                    && !ctx.struct_uniforms.contains_key("bgCoord")
                    && !ctx.samplers.contains("bgCoord") =>
            {
                out.push(Token::Ident("vUv".to_string()));
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && scopes.len() > 1
                    && !decl_is_const
                    && (name == "normalize" || name == "pow")
                    && !is_shadowed(name, scopes)
                    && !ctx.user_fns.contains(name.as_str())
                    && matches!(
                        tokens.get(next_non_newline(tokens, i + 1)),
                        Some(Token::Punct(p)) if p == "("
                    ) =>
            {
                let next_i = next_non_newline(tokens, i + 1);
                let expected_args = if name == "normalize" { 1 } else { 2 };
                if let Some((arg_ranges, after_call)) = parse_call_arg_ranges(tokens, next_i)?
                    && arg_ranges.len() == expected_args
                {
                    let helper_name = if name == "normalize" {
                        state.uses_zc_normalize = true;
                        "_zc_normalize"
                    } else {
                        state.uses_zc_pow = true;
                        "_zc_pow"
                    };
                    let before_len = out.len();
                    out.push(Token::Ident(helper_name.to_string()));
                    out.push(Token::Punct("(".to_string()));
                    for (arg_idx, &(lo, hi)) in arg_ranges.iter().enumerate() {
                        if arg_idx > 0 {
                            out.push(Token::Punct(",".to_string()));
                        }
                        let rewritten_arg = rewrite_slice(
                            &tokens[lo..hi],
                            ctx,
                            scopes,
                            user_types,
                            state,
                            depth + 1,
                            budget,
                        )?;
                        out.extend(rewritten_arg);
                    }
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && name == "dFdy"
                    && !is_shadowed(name, scopes)
                    && !ctx.user_fns.contains("dFdy")
                    && matches!(
                        tokens.get(next_non_newline(tokens, i + 1)),
                        Some(Token::Punct(p)) if p == "("
                    ) =>
            {
                let next_i = next_non_newline(tokens, i + 1);
                if let Some((arg_ranges, after_call)) = parse_call_arg_ranges(tokens, next_i)?
                    && arg_ranges.len() == 1
                {
                    let (lo, hi) = arg_ranges[0];
                    let rewritten_arg = rewrite_slice(
                        &tokens[lo..hi],
                        ctx,
                        scopes,
                        user_types,
                        state,
                        depth + 1,
                        budget,
                    )?;
                    let before_len = out.len();
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Punct("-".to_string()));
                    out.push(Token::Ident("dFdy".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(rewritten_arg);
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && !ctx.uniforms.contains_key(name.as_str())
                    && matches!(
                        tokens.get(next_non_newline(tokens, i + 1)),
                        Some(Token::Punct(p)) if p == "("
                    )
                    && (matches!(
                        name.as_str(),
                        "texture2D"
                            | "texture2DProj"
                            | "texture2DLodEXT"
                            | "texture2DGradEXT"
                            | "texture2DProjLodEXT"
                            | "texture2DProjGradEXT"
                            | "textureCube"
                            | "textureCubeLodEXT"
                            | "textureCubeGradEXT"
                    ) || (name == "texture"
                        && !ctx.user_fns.contains("texture")
                        && !is_shadowed("texture", scopes))) =>
            {
                let next_i = next_non_newline(tokens, i + 1);
                if let Some((arg_ranges, after_call)) = parse_call_arg_ranges(tokens, next_i)?
                    && arg_ranges.len() >= 2
                    && let Some(tex_ident) =
                        first_single_ident(&tokens[arg_ranges[0].0..arg_ranges[0].1])
                    && is_sampler_in_scope(tex_ident, ctx.samplers, scopes)
                {
                    let is_cube =
                        matches!(
                            name.as_str(),
                            "textureCube" | "textureCubeLodEXT" | "textureCubeGradEXT"
                        ) || is_cube_sampler_in_scope(tex_ident, ctx.cube_samplers, scopes);
                    let is_proj = matches!(
                        name.as_str(),
                        "texture2DProj" | "texture2DProjLodEXT" | "texture2DProjGradEXT"
                    );
                    let is_grad_2d =
                        matches!(name.as_str(), "texture2DGradEXT" | "texture2DProjGradEXT");
                    let fn_name = match name.as_str() {
                        "texture2DProj" => "textureProj",
                        "texture2DLodEXT" | "textureCubeLodEXT" => "textureLod",
                        "texture2DGradEXT" | "textureCubeGradEXT" => "textureGrad",
                        "texture2DProjLodEXT" => "textureProjLod",
                        "texture2DProjGradEXT" => "textureProjGrad",
                        _ => "texture",
                    };
                    let sampler_ctor = if is_cube { "samplerCube" } else { "sampler2D" };
                    let esc_tex = escape_glsl450_ident(tex_ident);
                    let before_len = out.len();
                    out.push(Token::Ident(fn_name.to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Ident(sampler_ctor.to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Ident(esc_tex.clone()));
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident(format!("{esc_tex}_sampler")));
                    out.push(Token::Punct(")".to_string()));
                    for (arg_idx, &(lo, hi)) in arg_ranges[1..].iter().enumerate() {
                        out.push(Token::Punct(",".to_string()));
                        let rewritten_arg = rewrite_slice(
                            &tokens[lo..hi],
                            ctx,
                            scopes,
                            user_types,
                            state,
                            depth + 1,
                            budget,
                        )?;
                        if arg_idx == 0 && !is_cube {
                            // Convert logical CM3 bottom-left UV to WebGPU top-left texture UV at the sampling boundary.
                            let helper = if is_proj {
                                "_zc_flip_uv_proj"
                            } else {
                                "_zc_flip_uv"
                            };
                            out.push(Token::Ident(helper.to_string()));
                            out.push(Token::Punct("(".to_string()));
                            out.extend(rewritten_arg);
                            out.push(Token::Punct(")".to_string()));
                        } else if is_grad_2d && (arg_idx == 1 || arg_idx == 2) {
                            // Flip Y derivatives to match V-flipped texture sampling.
                            state.uses_flip_grad = true;
                            out.push(Token::Ident("_zc_flip_grad".to_string()));
                            out.push(Token::Punct("(".to_string()));
                            out.extend(rewritten_arg);
                            out.push(Token::Punct(")".to_string()));
                        } else {
                            out.extend(rewritten_arg);
                        }
                    }
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }
                out.push(Token::Ident(escape_glsl450_ident(name)));
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && !(in_param_list && param_paren_depth == 0)
                    && ctx.sampler_fns.contains_key(name.as_str())
                    && matches!(
                        tokens.get(next_non_newline(tokens, i + 1)),
                        Some(Token::Punct(p)) if p == "("
                    ) =>
            {
                let next_i = next_non_newline(tokens, i + 1);
                if let Some((arg_ranges, after_call)) = parse_call_arg_ranges(tokens, next_i)?
                    && let Some(sampler_arg_indices) = ctx
                        .sampler_fns
                        .get(name.as_str())
                        .and_then(|arities| arities.get(&arg_ranges.len()))
                {
                    let before_len = out.len();
                    out.push(Token::Ident(escape_glsl450_ident(name)));
                    out.push(Token::Punct("(".to_string()));
                    for (arg_idx, &(lo, hi)) in arg_ranges.iter().enumerate() {
                        if arg_idx > 0 {
                            out.push(Token::Punct(",".to_string()));
                        }
                        if sampler_arg_indices.contains(&arg_idx)
                            && let Some(tex_ident) = first_single_ident(&tokens[lo..hi])
                            && is_sampler_in_scope(tex_ident, ctx.samplers, scopes)
                        {
                            let esc_tex = escape_glsl450_ident(tex_ident);
                            out.push(Token::Ident(esc_tex.clone()));
                            out.push(Token::Punct(",".to_string()));
                            out.push(Token::Ident(format!("{esc_tex}_sampler")));
                        } else {
                            let rewritten_arg = rewrite_slice(
                                &tokens[lo..hi],
                                ctx,
                                scopes,
                                user_types,
                                state,
                                depth + 1,
                                budget,
                            )?;
                            out.extend(rewritten_arg);
                        }
                    }
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }
                out.push(Token::Ident(escape_glsl450_ident(name)));
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && !is_shadowed(name, scopes)
                    && ctx.struct_uniforms.contains_key(name.as_str()) =>
            {
                let sdecl = &ctx.struct_uniforms[name.as_str()];
                let before_len = out.len();
                if let Some(len) = sdecl.array_len {
                    let next_i = next_non_newline(tokens, i + 1);
                    if matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "[")
                        && let Some((idx_tokens, after_bracket)) =
                            parse_bracket_expr(tokens, next_i)
                    {
                        if let Some((val, _)) =
                            eval_const_int_with_scopes(&idx_tokens, scopes, ctx.global_const_ints)
                            && val >= 0
                            && (val as u32) < len
                        {
                            let elem_prefix = format!("{}[{val}]", sdecl.name);
                            emit_struct_uniform_expr(
                                &mut out,
                                &elem_prefix,
                                &sdecl.struct_type,
                                ctx,
                                state,
                                0,
                            )?;
                        } else {
                            let rewritten_idx = rewrite_slice(
                                &idx_tokens,
                                ctx,
                                scopes,
                                user_types,
                                state,
                                depth + 1,
                                budget,
                            )?;
                            out.push(Token::Ident(escape_glsl450_ident(&sdecl.struct_type)));
                            out.push(Token::Punct("[".to_string()));
                            out.push(Token::Number(len.to_string()));
                            out.push(Token::Punct("]".to_string()));
                            out.push(Token::Punct("(".to_string()));
                            for k in 0..len {
                                if k > 0 {
                                    out.push(Token::Punct(",".to_string()));
                                }
                                let elem_prefix = format!("{}[{k}]", sdecl.name);
                                emit_struct_uniform_expr(
                                    &mut out,
                                    &elem_prefix,
                                    &sdecl.struct_type,
                                    ctx,
                                    state,
                                    0,
                                )?;
                            }
                            out.push(Token::Punct(")".to_string()));
                            out.push(Token::Punct("[".to_string()));
                            out.extend(rewritten_idx);
                            out.push(Token::Punct("]".to_string()));
                        }
                        budget.charge(out.len().saturating_sub(before_len))?;
                        i = after_bracket;
                        continue;
                    }
                    out.push(Token::Ident(escape_glsl450_ident(&sdecl.struct_type)));
                    out.push(Token::Punct("[".to_string()));
                    out.push(Token::Number(len.to_string()));
                    out.push(Token::Punct("]".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    for k in 0..len {
                        if k > 0 {
                            out.push(Token::Punct(",".to_string()));
                        }
                        let elem_prefix = format!("{}[{k}]", sdecl.name);
                        emit_struct_uniform_expr(
                            &mut out,
                            &elem_prefix,
                            &sdecl.struct_type,
                            ctx,
                            state,
                            0,
                        )?;
                    }
                    out.push(Token::Punct(")".to_string()));
                } else {
                    emit_struct_uniform_expr(
                        &mut out,
                        &sdecl.name,
                        &sdecl.struct_type,
                        ctx,
                        state,
                        0,
                    )?;
                }
                budget.charge(out.len().saturating_sub(before_len))?;
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && !is_shadowed(name, scopes)
                    && ctx.uniforms.contains_key(name.as_str()) =>
            {
                let decl = ctx.uniforms[name.as_str()];
                let field_id = ubo_field_ident(&decl.name);
                if decl.array_len.is_some() {
                    let next_i = next_non_newline(tokens, i + 1);
                    if matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "[")
                        && let Some((idx_tokens, after_bracket)) =
                            parse_bracket_expr(tokens, next_i)
                    {
                        let rewritten_idx = rewrite_slice(
                            &idx_tokens,
                            ctx,
                            scopes,
                            user_types,
                            state,
                            depth + 1,
                            budget,
                        )?;
                        let before_len = out.len();
                        if decl.ty == UniformType::Mat2 {
                            state.mat2_array_loaders.insert(decl.name.clone());
                            out.push(Token::Ident(format!("_zc_load_{field_id}")));
                            out.push(Token::Punct("(".to_string()));
                            out.push(Token::Ident("int".to_string()));
                            out.push(Token::Punct("(".to_string()));
                            out.extend(rewritten_idx);
                            out.push(Token::Punct(")".to_string()));
                            out.push(Token::Punct(")".to_string()));
                        } else if matches!(
                            decl.ty,
                            UniformType::Bvec2 | UniformType::Bvec3 | UniformType::Bvec4
                        ) {
                            let zero_ty = match decl.ty {
                                UniformType::Bvec2 => "uvec2",
                                UniformType::Bvec3 => "uvec3",
                                _ => "uvec4",
                            };
                            out.push(Token::Ident("notEqual".to_string()));
                            out.push(Token::Punct("(".to_string()));
                            out.push(Token::Ident("_zc_u".to_string()));
                            out.push(Token::Punct(".".to_string()));
                            out.push(Token::Ident(field_id));
                            out.push(Token::Punct("[".to_string()));
                            out.extend(rewritten_idx);
                            out.push(Token::Punct("]".to_string()));
                            if let Some(swizzle) = decl.ty.array_swizzle() {
                                out.push(Token::Punct(".".to_string()));
                                out.push(Token::Ident(swizzle.trim_start_matches('.').to_string()));
                            }
                            out.push(Token::Punct(",".to_string()));
                            out.push(Token::Ident(zero_ty.to_string()));
                            out.push(Token::Punct("(".to_string()));
                            out.push(Token::Number("0u".to_string()));
                            out.push(Token::Punct(")".to_string()));
                            out.push(Token::Punct(")".to_string()));
                        } else {
                            if decl.ty == UniformType::Bool {
                                out.push(Token::Punct("(".to_string()));
                            }
                            out.push(Token::Ident("_zc_u".to_string()));
                            out.push(Token::Punct(".".to_string()));
                            out.push(Token::Ident(field_id));
                            out.push(Token::Punct("[".to_string()));
                            out.extend(rewritten_idx);
                            out.push(Token::Punct("]".to_string()));
                            if let Some(swizzle) = decl.ty.array_swizzle() {
                                out.push(Token::Punct(".".to_string()));
                                out.push(Token::Ident(swizzle.trim_start_matches('.').to_string()));
                            }
                            if decl.ty == UniformType::Bool {
                                out.push(Token::Punct("!=".to_string()));
                                out.push(Token::Number("0u".to_string()));
                                out.push(Token::Punct(")".to_string()));
                            }
                        }
                        budget.charge(out.len().saturating_sub(before_len))?;
                        i = after_bracket;
                        continue;
                    }
                    let before_len = out.len();
                    emit_bare_array_uniform_read(&mut out, decl, state);
                    budget.charge(out.len().saturating_sub(before_len))?;
                } else {
                    let before_len = out.len();
                    emit_non_array_uniform_read(&mut out, decl);
                    budget.charge(out.len().saturating_sub(before_len))?;
                }
                i += 1;
                continue;
            }
            Token::Ident(name) => {
                out.push(Token::Ident(escape_glsl450_ident(name)));
                i += 1;
            }
            _ => {
                out.push(tokens[i].clone());
                i += 1;
            }
        }
    }

    flush_pending_decl(&mut pending_decl_var, scopes);
    Ok(out)
}

/// Removes unreachable statements following an unconditional block-level `return ...;` up to the
/// enclosing `}` (or `case`/`default` label).
///
/// Naga 30.0.2's GLSL frontend rejects dead code containing function calls after an unconditional
/// `return;` (such as `mosaic.glsl`'s `#ifdef GL_EXT_shader_texture_lod ... return; #endif`
/// fallback loop) with `Expression not visited by the appropriate statement`.
pub(super) fn strip_unreachable_after_return(tokens: Vec<Token>) -> Vec<Token> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut brace_depth: usize = 0;
    let mut paren_depth: usize = 0;
    let mut prev_is_block_stmt_boundary = false;
    let mut iter = tokens.into_iter().peekable();

    while let Some(tok) = iter.next() {
        if brace_depth > 0
            && paren_depth == 0
            && prev_is_block_stmt_boundary
            && matches!(&tok, Token::Ident(kw) if kw == "return")
        {
            // Emit `return ...;` up through its terminating `;`.
            out.push(tok);
            let mut ret_pd: usize = 0;
            for t in iter.by_ref() {
                let done = match &t {
                    Token::Punct(p) if p == "(" => {
                        ret_pd += 1;
                        false
                    }
                    Token::Punct(p) if p == ")" => {
                        ret_pd = ret_pd.saturating_sub(1);
                        false
                    }
                    Token::Punct(p) if p == ";" && ret_pd == 0 => true,
                    _ => false,
                };
                out.push(t);
                if done {
                    break;
                }
            }
            // Skip unreachable tokens in the same block until matching `}` or `case`/`default`.
            let mut nested_braces: usize = 0;
            while let Some(next_tok) = iter.peek() {
                match next_tok {
                    Token::Punct(p) if p == "{" => {
                        nested_braces += 1;
                        iter.next();
                    }
                    Token::Punct(p) if p == "}" => {
                        if nested_braces == 0 {
                            break;
                        }
                        nested_braces -= 1;
                        iter.next();
                    }
                    Token::Ident(kw) if nested_braces == 0 && (kw == "case" || kw == "default") => {
                        break;
                    }
                    _ => {
                        iter.next();
                    }
                }
            }
            prev_is_block_stmt_boundary = true;
            continue;
        }

        match &tok {
            Token::Punct(p) if p == "{" => {
                brace_depth += 1;
                prev_is_block_stmt_boundary = true;
            }
            Token::Punct(p) if p == "}" => {
                brace_depth = brace_depth.saturating_sub(1);
                prev_is_block_stmt_boundary = true;
            }
            Token::Punct(p) if p == "(" => {
                paren_depth += 1;
                prev_is_block_stmt_boundary = false;
            }
            Token::Punct(p) if p == ")" => {
                paren_depth = paren_depth.saturating_sub(1);
                prev_is_block_stmt_boundary = false;
            }
            Token::Punct(p) if p == ";" || p == ":" => {
                prev_is_block_stmt_boundary = true;
            }
            Token::Newline => {}
            _ => {
                prev_is_block_stmt_boundary = false;
            }
        }
        out.push(tok);
    }

    out
}

fn rewrite_preprocessor_line(
    line: &str,
    ctx: &RewriteCtx<'_>,
    state: &mut RewriteState,
    depth: usize,
    budget: &mut RewriteBudget,
) -> Result<String, ShaderError> {
    let trimmed = line.trim();
    let Some(after_hash) = trimmed.strip_prefix('#') else {
        return Ok(line.to_string());
    };
    let directive = after_hash.trim_start();
    let Some(after_define) = directive.strip_prefix("define") else {
        return Ok(line.to_string());
    };
    if !after_define.starts_with(char::is_whitespace) {
        return Ok(line.to_string());
    }

    let rest = after_define.trim_start();
    let mut name_end = 0;
    for (idx, ch) in rest.char_indices() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            name_end = idx + ch.len_utf8();
        } else {
            break;
        }
    }
    if name_end == 0 {
        return Ok(line.to_string());
    }

    let macro_name = &rest[..name_end];
    let after_name = &rest[name_end..];
    let mut macro_params = HashSet::new();
    let (header_suffix, body) = if let Some(after_open) = after_name.strip_prefix('(') {
        if let Some(close_idx) = after_open.find(')') {
            let params_str = &after_open[..close_idx];
            for p in params_str.split(',') {
                let pt = p.trim();
                if !pt.is_empty() {
                    macro_params.insert(pt.to_string());
                }
            }
            (&after_name[..=close_idx + 1], &after_open[close_idx + 1..])
        } else {
            return Ok(line.to_string());
        }
    } else {
        ("", after_name)
    };

    if body.trim().is_empty() {
        return Ok(line.to_string());
    }

    let body_tokens = tokenize(body);
    let mut root_scope = ScopeFrame::new(ScopeKind::Root);
    root_scope.vars = macro_params;
    let mut scopes = vec![root_scope];
    let mut user_types: HashSet<String> = ctx.struct_defs.keys().cloned().collect();
    let rewritten = rewrite_slice(
        &body_tokens,
        ctx,
        &mut scopes,
        &mut user_types,
        state,
        depth,
        budget,
    )?;
    let rendered = render_tokens(&rewritten).trim().to_string();
    Ok(format!("#define {macro_name}{header_suffix} {rendered}"))
}
