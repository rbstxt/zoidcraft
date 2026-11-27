use super::*;

#[derive(Debug, Clone)]
pub(super) struct UniformDecl {
    pub(super) name: String,
    pub(super) ty: UniformType,
    pub(super) array_len: Option<u32>,
    pub(super) default_value: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct StructFieldDecl {
    pub(super) ty_name: String,
    pub(super) name: String,
    pub(super) array_len: Option<u32>,
}

#[derive(Debug, Clone)]
pub(super) struct StructUniformDecl {
    pub(super) name: String,
    pub(super) struct_type: String,
    pub(super) array_len: Option<u32>,
}

#[derive(Debug, Clone)]
pub(super) struct VaryingDecl {
    pub(super) ty: String,
    pub(super) name: String,
    pub(super) array_len: Option<u32>,
}

pub(super) struct ExtractedDecls {
    pub(super) uniforms: Vec<UniformDecl>,
    pub(super) struct_uniforms: HashMap<String, StructUniformDecl>,
    pub(super) struct_defs: HashMap<String, Vec<StructFieldDecl>>,
    pub(super) const_ints: HashMap<String, u32>,
    pub(super) samplers: Vec<String>,
    pub(super) cube_samplers: HashSet<String>,
    pub(super) extra_varyings: Vec<VaryingDecl>,
    pub(super) body_tokens: Vec<Token>,
}

pub(super) fn is_precision_qual(s: &str) -> bool {
    matches!(s, "highp" | "mediump" | "lowp")
}

pub(super) fn next_non_newline(tokens: &[Token], mut idx: usize) -> usize {
    while idx < tokens.len() && matches!(tokens[idx], Token::Newline) {
        idx += 1;
    }
    idx
}

fn parse_array_len_bracket(
    tokens: &[Token],
    open_bracket_idx: usize,
    int_defines: &HashMap<String, u32>,
    decl_name: &str,
) -> Result<(u32, usize), ShaderError> {
    let Some((inner, after_bracket)) = parse_bracket_expr(tokens, open_bracket_idx) else {
        return Err(ShaderError::InvalidUniformDeclaration(format!(
            "expected ']' in array declaration `{decl_name}`"
        )));
    };
    let len_val = eval_const_int_tokens(&inner, int_defines)
        .and_then(|v| u32::try_from(v).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| {
            ShaderError::InvalidUniformDeclaration(format!("invalid array length on `{decl_name}`"))
        })?;
    Ok((len_val, after_bracket))
}

pub(super) fn extract_declarations(
    tokens: &[Token],
    int_defines: &HashMap<String, u32>,
) -> Result<ExtractedDecls, ShaderError> {
    let mut uniforms: Vec<UniformDecl> = Vec::new();
    let mut uniform_types: HashMap<String, (UniformType, Option<u32>)> = HashMap::new();
    let mut struct_uniforms: HashMap<String, StructUniformDecl> = HashMap::new();
    let mut struct_defs: HashMap<String, Vec<StructFieldDecl>> = HashMap::new();
    let mut const_ints: HashMap<String, u32> = int_defines.clone();
    let mut samplers: Vec<String> = Vec::new();
    let mut sampler_seen: HashSet<String> = HashSet::new();
    let mut cube_samplers: HashSet<String> = HashSet::new();
    let mut extra_varyings: Vec<VaryingDecl> = Vec::new();
    let mut varying_seen: HashSet<String> = HashSet::new();
    let mut body_tokens: Vec<Token> = Vec::with_capacity(tokens.len());

    let mut brace_depth: usize = 0;
    let mut i = 0;

    while i < tokens.len() {
        match &tokens[i] {
            Token::Punct(p) if p == "{" => {
                brace_depth += 1;
                body_tokens.push(tokens[i].clone());
                i += 1;
            }
            Token::Punct(p) if p == "}" => {
                brace_depth = brace_depth.saturating_sub(1);
                body_tokens.push(tokens[i].clone());
                i += 1;
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "struct" => {
                if let Some((struct_name, fields)) =
                    parse_top_level_struct_def(tokens, i, &const_ints)
                {
                    struct_defs.insert(struct_name, fields);
                }
                body_tokens.push(tokens[i].clone());
                i += 1;
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "const" => {
                collect_top_level_const_ints(tokens, i, &mut const_ints);
                body_tokens.push(tokens[i].clone());
                i += 1;
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "precision" => {
                // Strip `precision <qual> <type>;`
                while i < tokens.len() && !matches!(&tokens[i], Token::Punct(p) if p == ";") {
                    i += 1;
                }
                if i < tokens.len() {
                    i += 1;
                }
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "invariant" => {
                let next_i = next_non_newline(tokens, i + 1);
                if matches!(tokens.get(next_i), Some(Token::Ident(next_kw)) if next_kw == "varying")
                {
                    i = parse_varying_decl(
                        tokens,
                        next_i + 1,
                        &const_ints,
                        &mut extra_varyings,
                        &mut varying_seen,
                    )?;
                } else {
                    body_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "varying" => {
                i = parse_varying_decl(
                    tokens,
                    i + 1,
                    &const_ints,
                    &mut extra_varyings,
                    &mut varying_seen,
                )?;
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "uniform" => {
                i = parse_uniform_decl(
                    tokens,
                    i + 1,
                    &const_ints,
                    &struct_defs,
                    &mut uniforms,
                    &mut uniform_types,
                    &mut struct_uniforms,
                    &mut samplers,
                    &mut sampler_seen,
                    &mut cube_samplers,
                )?;
            }
            _ => {
                body_tokens.push(tokens[i].clone());
                i += 1;
            }
        }
    }

    Ok(ExtractedDecls {
        uniforms,
        struct_uniforms,
        struct_defs,
        const_ints,
        samplers,
        cube_samplers,
        extra_varyings,
        body_tokens,
    })
}

fn collect_top_level_const_ints(
    tokens: &[Token],
    const_idx: usize,
    const_ints: &mut HashMap<String, u32>,
) {
    let mut i = next_non_newline(tokens, const_idx + 1);
    if let Some(Token::Ident(q)) = tokens.get(i)
        && is_precision_qual(q)
    {
        i = next_non_newline(tokens, i + 1);
    }
    if !matches!(tokens.get(i), Some(Token::Ident(ty)) if ty == "int" || ty == "uint") {
        return;
    }
    i = next_non_newline(tokens, i + 1);

    while i < tokens.len() {
        let Some(Token::Ident(name)) = tokens.get(i) else {
            return;
        };
        let var_name = name.clone();
        i = next_non_newline(tokens, i + 1);
        if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "=") {
            i = next_non_newline(tokens, i + 1);
            let expr_start = i;
            let mut paren_depth: usize = 0;
            let mut bracket_depth: usize = 0;
            while i < tokens.len() {
                match &tokens[i] {
                    Token::Punct(p) if p == "(" => paren_depth += 1,
                    Token::Punct(p) if p == ")" => paren_depth = paren_depth.saturating_sub(1),
                    Token::Punct(p) if p == "[" => bracket_depth += 1,
                    Token::Punct(p) if p == "]" => bracket_depth = bracket_depth.saturating_sub(1),
                    Token::Punct(p)
                        if paren_depth == 0 && bracket_depth == 0 && (p == "," || p == ";") =>
                    {
                        break;
                    }
                    _ => {}
                }
                i += 1;
            }
            if let Some(val) = eval_const_int_tokens(&tokens[expr_start..i], const_ints)
                && let Ok(u) = u32::try_from(val)
            {
                const_ints.insert(var_name, u);
            }
        }
        match tokens.get(i) {
            Some(Token::Punct(p)) if p == "," => {
                i = next_non_newline(tokens, i + 1);
            }
            _ => return,
        }
    }
}

fn parse_top_level_struct_def(
    tokens: &[Token],
    struct_idx: usize,
    const_ints: &HashMap<String, u32>,
) -> Option<(String, Vec<StructFieldDecl>)> {
    let name_idx = next_non_newline(tokens, struct_idx + 1);
    let Token::Ident(struct_name) = tokens.get(name_idx)? else {
        return None;
    };
    let open_brace = next_non_newline(tokens, name_idx + 1);
    if !matches!(tokens.get(open_brace), Some(Token::Punct(p)) if p == "{") {
        return None;
    }

    let mut fields = Vec::new();
    let mut i = next_non_newline(tokens, open_brace + 1);
    while i < tokens.len() {
        if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "}") {
            return Some((struct_name.clone(), fields));
        }
        if let Some(Token::Ident(q)) = tokens.get(i)
            && is_precision_qual(q)
        {
            i = next_non_newline(tokens, i + 1);
        }
        let Token::Ident(ty_name) = tokens.get(i)? else {
            return None;
        };
        let field_ty = ty_name.clone();
        i = next_non_newline(tokens, i + 1);
        while i < tokens.len() {
            let Token::Ident(field_name) = tokens.get(i)? else {
                return None;
            };
            i = next_non_newline(tokens, i + 1);
            let mut array_len = None;
            if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "[") {
                let (len_val, after_bracket) =
                    parse_array_len_bracket(tokens, i, const_ints, field_name).ok()?;
                array_len = Some(len_val);
                i = next_non_newline(tokens, after_bracket);
            }
            fields.push(StructFieldDecl {
                ty_name: field_ty.clone(),
                name: field_name.clone(),
                array_len,
            });
            match tokens.get(i) {
                Some(Token::Punct(p)) if p == "," => {
                    i = next_non_newline(tokens, i + 1);
                }
                Some(Token::Punct(p)) if p == ";" => {
                    i = next_non_newline(tokens, i + 1);
                    break;
                }
                _ => return None,
            }
        }
    }
    None
}

fn flatten_struct_uniform_fields(
    prefix: &str,
    struct_name: &str,
    struct_defs: &HashMap<String, Vec<StructFieldDecl>>,
    uniforms: &mut Vec<UniformDecl>,
    uniform_types: &mut HashMap<String, (UniformType, Option<u32>)>,
    sampler_seen: &HashSet<String>,
    depth: usize,
) -> Result<(), ShaderError> {
    if depth > 16 {
        return Err(ShaderError::InvalidUniformDeclaration(format!(
            "struct uniform `{prefix}` exceeds maximum nesting depth"
        )));
    }
    let Some(fields) = struct_defs.get(struct_name) else {
        return Err(ShaderError::InvalidUniformDeclaration(format!(
            "unsupported uniform struct type `{struct_name}`"
        )));
    };
    for field in fields {
        let member_prefix = format!("{prefix}.{}", field.name);
        if let Some(ty) = UniformType::from_glsl(&field.ty_name) {
            if sampler_seen.contains(&member_prefix) {
                return Err(ShaderError::DuplicateUniform {
                    name: member_prefix,
                });
            }
            if let Some(&(prev_ty, prev_len)) = uniform_types.get(&member_prefix) {
                if prev_ty != ty || prev_len != field.array_len {
                    return Err(ShaderError::DuplicateUniform {
                        name: member_prefix,
                    });
                }
            } else {
                uniform_types.insert(member_prefix.clone(), (ty, field.array_len));
                uniforms.push(UniformDecl {
                    name: member_prefix,
                    ty,
                    array_len: field.array_len,
                    default_value: None,
                });
            }
        } else if struct_defs.contains_key(&field.ty_name) {
            if let Some(len) = field.array_len {
                if len == 0 || len > MAX_UNIFORM_ARRAY_LEN {
                    return Err(ShaderError::InvalidUniformDeclaration(format!(
                        "struct uniform array `{member_prefix}` length {len} exceeds allowed range"
                    )));
                }
                for idx in 0..len {
                    let elem_prefix = format!("{member_prefix}[{idx}]");
                    flatten_struct_uniform_fields(
                        &elem_prefix,
                        &field.ty_name,
                        struct_defs,
                        uniforms,
                        uniform_types,
                        sampler_seen,
                        depth + 1,
                    )?;
                }
            } else {
                flatten_struct_uniform_fields(
                    &member_prefix,
                    &field.ty_name,
                    struct_defs,
                    uniforms,
                    uniform_types,
                    sampler_seen,
                    depth + 1,
                )?;
            }
        } else {
            return Err(ShaderError::InvalidUniformDeclaration(format!(
                "unsupported uniform struct field type `{}` in `{struct_name}`",
                field.ty_name
            )));
        }
    }
    Ok(())
}

fn parse_varying_decl(
    tokens: &[Token],
    mut i: usize,
    int_defines: &HashMap<String, u32>,
    extra_varyings: &mut Vec<VaryingDecl>,
    varying_seen: &mut HashSet<String>,
) -> Result<usize, ShaderError> {
    i = next_non_newline(tokens, i);
    if let Some(Token::Ident(q)) = tokens.get(i)
        && is_precision_qual(q)
    {
        i = next_non_newline(tokens, i + 1);
    }

    let ty = match tokens.get(i) {
        Some(Token::Ident(t)) => t.clone(),
        _ => {
            return Err(ShaderError::InvalidUniformDeclaration(
                "expected type after varying".to_string(),
            ));
        }
    };
    i = next_non_newline(tokens, i + 1);

    while i < tokens.len() {
        let name = match tokens.get(i) {
            Some(Token::Ident(n)) => n.clone(),
            _ => {
                return Err(ShaderError::InvalidUniformDeclaration(
                    "expected identifier in varying declaration".to_string(),
                ));
            }
        };
        i = next_non_newline(tokens, i + 1);

        let mut array_len = None;
        if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "[") {
            let (len_val, after_bracket) = parse_array_len_bracket(tokens, i, int_defines, &name)?;
            if len_val > MAX_UNIFORM_ARRAY_LEN {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "varying array `{name}` length {len_val} exceeds allowed range 1..={MAX_UNIFORM_ARRAY_LEN}"
                )));
            }
            array_len = Some(len_val);
            i = next_non_newline(tokens, after_bracket);
        }

        if name != "vUv"
            && name != "vUvScaled"
            && name != "bgCoord"
            && varying_seen.insert(name.clone())
        {
            extra_varyings.push(VaryingDecl {
                ty: ty.clone(),
                name,
                array_len,
            });
        }
        match tokens.get(i) {
            Some(Token::Punct(p)) if p == "," => {
                i = next_non_newline(tokens, i + 1);
            }
            Some(Token::Punct(p)) if p == ";" => {
                return Ok(i + 1);
            }
            _ => {
                return Err(ShaderError::InvalidUniformDeclaration(
                    "expected ',' or ';' in varying declaration".to_string(),
                ));
            }
        }
    }

    Err(ShaderError::InvalidUniformDeclaration(
        "unterminated varying declaration".to_string(),
    ))
}

fn format_initializer_tokens(tokens: &[Token]) -> String {
    let mut out = String::new();
    let mut prev: Option<&Token> = None;
    for tok in tokens {
        match tok {
            Token::Newline | Token::Preprocessor(_) => {}
            Token::Ident(s) | Token::Number(s) => {
                if let Some(p) = prev {
                    match p {
                        Token::Ident(_) | Token::Number(_) => out.push(' '),
                        Token::Punct(ps)
                            if !matches!(ps.as_str(), "." | "(" | "[" | "!" | "~" | "-" | "+") =>
                        {
                            out.push(' ');
                        }
                        _ => {}
                    }
                }
                out.push_str(s);
                prev = Some(tok);
            }
            Token::Punct(s) => {
                if let Some(p) = prev {
                    let is_unary_sign = (s == "-" || s == "+")
                        && matches!(p, Token::Punct(ps) if matches!(ps.as_str(), "(" | "[" | ","));
                    let need_space = if is_unary_sign {
                        matches!(p, Token::Punct(ps) if ps == ",")
                    } else {
                        match (p, s.as_str()) {
                            (_, "." | ";" | "," | ")" | "]" | "(" | "[") => false,
                            (Token::Punct(ps), _) if matches!(ps.as_str(), "." | "(" | "[") => {
                                false
                            }
                            _ => true,
                        }
                    };
                    if need_space {
                        out.push(' ');
                    }
                }
                out.push_str(s);
                prev = Some(tok);
            }
        }
    }
    out.trim().to_string()
}

#[allow(clippy::too_many_arguments)]
fn parse_uniform_decl(
    tokens: &[Token],
    mut i: usize,
    int_defines: &HashMap<String, u32>,
    struct_defs: &HashMap<String, Vec<StructFieldDecl>>,
    uniforms: &mut Vec<UniformDecl>,
    uniform_types: &mut HashMap<String, (UniformType, Option<u32>)>,
    struct_uniforms: &mut HashMap<String, StructUniformDecl>,
    samplers: &mut Vec<String>,
    sampler_seen: &mut HashSet<String>,
    cube_samplers: &mut HashSet<String>,
) -> Result<usize, ShaderError> {
    i = next_non_newline(tokens, i);
    if let Some(Token::Ident(q)) = tokens.get(i)
        && is_precision_qual(q)
    {
        i = next_non_newline(tokens, i + 1);
    }

    let ty_ident = match tokens.get(i) {
        Some(Token::Ident(t)) => t.as_str(),
        _ => {
            return Err(ShaderError::InvalidUniformDeclaration(
                "expected type after uniform".to_string(),
            ));
        }
    };
    i = next_non_newline(tokens, i + 1);

    let is_sampler2d = ty_ident == "sampler2D";
    let is_sampler_cube = ty_ident == "samplerCube";
    let is_struct = !is_sampler2d && !is_sampler_cube && struct_defs.contains_key(ty_ident);
    let uniform_ty = if is_sampler2d || is_sampler_cube || is_struct {
        None
    } else {
        Some(UniformType::from_glsl(ty_ident).ok_or_else(|| {
            ShaderError::InvalidUniformDeclaration(format!("unsupported uniform type `{ty_ident}`"))
        })?)
    };

    while i < tokens.len() {
        let name = match tokens.get(i) {
            Some(Token::Ident(n)) => n.clone(),
            _ => {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "expected uniform name after `{ty_ident}`"
                )));
            }
        };
        i = next_non_newline(tokens, i + 1);

        // Optional array suffix `[N]` (supports decimal, hex, uint suffix, and constant expressions)
        let mut array_len = None;
        if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "[") {
            let (len_val, after_bracket) = parse_array_len_bracket(tokens, i, int_defines, &name)?;
            array_len = Some(len_val);
            i = next_non_newline(tokens, after_bracket);
        }

        // Parse optional non-standard initializer `= ...` up to `,` or `;`
        let mut default_value: Option<String> = None;
        if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "=") {
            i = next_non_newline(tokens, i + 1);
            let expr_start = i;
            let mut paren_depth: usize = 0;
            let mut bracket_depth: usize = 0;
            while i < tokens.len() {
                match &tokens[i] {
                    Token::Punct(p) if p == "(" => paren_depth += 1,
                    Token::Punct(p) if p == ")" => paren_depth = paren_depth.saturating_sub(1),
                    Token::Punct(p) if p == "[" => bracket_depth += 1,
                    Token::Punct(p) if p == "]" => bracket_depth = bracket_depth.saturating_sub(1),
                    Token::Punct(p)
                        if paren_depth == 0 && bracket_depth == 0 && (p == "," || p == ";") =>
                    {
                        break;
                    }
                    _ => {}
                }
                i += 1;
            }
            let rendered = format_initializer_tokens(&tokens[expr_start..i]);
            if !rendered.is_empty() {
                default_value = Some(rendered);
            }
        }

        if is_sampler2d || is_sampler_cube {
            if array_len.is_some() {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "{ty_ident} arrays are not supported (`{name}`)"
                )));
            }
            if uniform_types.contains_key(&name) || struct_uniforms.contains_key(&name) {
                return Err(ShaderError::DuplicateUniform { name });
            }
            if sampler_seen.contains(&name) {
                if cube_samplers.contains(&name) != is_sampler_cube {
                    return Err(ShaderError::DuplicateUniform { name });
                }
            } else {
                sampler_seen.insert(name.clone());
                if is_sampler_cube {
                    cube_samplers.insert(name.clone());
                }
                samplers.push(name);
            }
        } else if is_struct {
            if sampler_seen.contains(&name) || uniform_types.contains_key(&name) {
                return Err(ShaderError::DuplicateUniform { name });
            }
            if let Some(prev) = struct_uniforms.get(&name) {
                if prev.struct_type != ty_ident || prev.array_len != array_len {
                    return Err(ShaderError::DuplicateUniform { name });
                }
            } else {
                if let Some(len) = array_len {
                    if len == 0 || len > MAX_UNIFORM_ARRAY_LEN {
                        return Err(ShaderError::InvalidUniformDeclaration(format!(
                            "uniform struct array `{name}` length {len} exceeds allowed range 1..={MAX_UNIFORM_ARRAY_LEN}"
                        )));
                    }
                    for idx in 0..len {
                        let elem_prefix = format!("{name}[{idx}]");
                        flatten_struct_uniform_fields(
                            &elem_prefix,
                            ty_ident,
                            struct_defs,
                            uniforms,
                            uniform_types,
                            sampler_seen,
                            0,
                        )?;
                    }
                } else {
                    flatten_struct_uniform_fields(
                        &name,
                        ty_ident,
                        struct_defs,
                        uniforms,
                        uniform_types,
                        sampler_seen,
                        0,
                    )?;
                }
                struct_uniforms.insert(
                    name.clone(),
                    StructUniformDecl {
                        name,
                        struct_type: ty_ident.to_string(),
                        array_len,
                    },
                );
            }
        } else if let Some(ty) = uniform_ty {
            if sampler_seen.contains(&name) || struct_uniforms.contains_key(&name) {
                return Err(ShaderError::DuplicateUniform { name });
            }
            if let Some(&(prev_ty, prev_len)) = uniform_types.get(&name) {
                if prev_ty != ty || prev_len != array_len {
                    return Err(ShaderError::DuplicateUniform { name });
                }
            } else {
                uniform_types.insert(name.clone(), (ty, array_len));
                uniforms.push(UniformDecl {
                    name,
                    ty,
                    array_len,
                    default_value,
                });
            }
        }

        i = next_non_newline(tokens, i);
        match tokens.get(i) {
            Some(Token::Punct(p)) if p == "," => {
                i = next_non_newline(tokens, i + 1);
            }
            Some(Token::Punct(p)) if p == ";" => {
                return Ok(i + 1);
            }
            _ => {
                return Err(ShaderError::InvalidUniformDeclaration(
                    "expected ',' or ';' after uniform declaration".to_string(),
                ));
            }
        }
    }

    Err(ShaderError::InvalidUniformDeclaration(
        "unterminated uniform declaration".to_string(),
    ))
}

fn checked_align_up(value: u32, alignment: u32) -> Option<u32> {
    if alignment <= 1 {
        Some(value)
    } else {
        value.div_ceil(alignment).checked_mul(alignment)
    }
}

pub(super) fn build_uniform_layout(
    decls: &[UniformDecl],
) -> Result<UniformBlockLayout, ShaderError> {
    if decls.is_empty() {
        return Ok(UniformBlockLayout {
            set: 0,
            binding: 0,
            fields: Vec::new(),
            byte_size: 0,
        });
    }

    let mut fields = Vec::with_capacity(decls.len());
    let mut cursor: u32 = 0;

    for decl in decls {
        if let Some(len) = decl.array_len {
            if len == 0 || len > MAX_UNIFORM_ARRAY_LEN {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "uniform array `{}` length {len} exceeds allowed range 1..={MAX_UNIFORM_ARRAY_LEN}",
                    decl.name
                )));
            }
            let stride = decl.ty.array_stride();
            let offset = checked_align_up(cursor, 16).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform layout offset overflow at `{}`",
                    decl.name
                ))
            })?;
            let size = stride.checked_mul(len).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform array `{}` byte size overflow",
                    decl.name
                ))
            })?;
            cursor = offset.checked_add(size).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform layout size overflow at `{}`",
                    decl.name
                ))
            })?;
            if cursor > MAX_UNIFORM_BUFFER_BYTES {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "uniform block size exceeds {MAX_UNIFORM_BUFFER_BYTES} bytes at `{}`",
                    decl.name
                )));
            }
            fields.push(UniformField {
                name: decl.name.clone(),
                ty: decl.ty,
                array_len: Some(len),
                offset,
                size,
                array_stride: stride,
                default_value: decl.default_value.clone(),
            });
        } else {
            let align = decl.ty.base_alignment();
            let size = decl.ty.byte_size();
            let offset = checked_align_up(cursor, align).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform layout offset overflow at `{}`",
                    decl.name
                ))
            })?;
            cursor = offset.checked_add(size).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform layout size overflow at `{}`",
                    decl.name
                ))
            })?;
            if cursor > MAX_UNIFORM_BUFFER_BYTES {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "uniform block size exceeds {MAX_UNIFORM_BUFFER_BYTES} bytes at `{}`",
                    decl.name
                )));
            }
            fields.push(UniformField {
                name: decl.name.clone(),
                ty: decl.ty,
                array_len: None,
                offset,
                size,
                array_stride: 0,
                default_value: decl.default_value.clone(),
            });
        }
    }

    let byte_size = checked_align_up(cursor, 16).ok_or_else(|| {
        ShaderError::InvalidUniformDeclaration("uniform block aligned size overflow".to_string())
    })?;
    if byte_size > MAX_UNIFORM_BUFFER_BYTES {
        return Err(ShaderError::InvalidUniformDeclaration(format!(
            "uniform block size {byte_size} exceeds {MAX_UNIFORM_BUFFER_BYTES} bytes"
        )));
    }

    Ok(UniformBlockLayout {
        set: 0,
        binding: 0,
        fields,
        byte_size,
    })
}

pub(super) fn build_texture_bindings(
    samplers: &[String],
    cube_samplers: &HashSet<String>,
) -> Vec<TextureBinding> {
    let mut ordered: Vec<&String> = Vec::with_capacity(samplers.len());
    if let Some(diffuse) = samplers.iter().find(|s| s.as_str() == "tDiffuse") {
        ordered.push(diffuse);
    }
    for s in samplers {
        if s != "tDiffuse" {
            ordered.push(s);
        }
    }

    // Binding 0 in set 0 is reserved for `ZcCustomUniforms`.
    let mut next_binding: u32 = 1;
    ordered
        .into_iter()
        .map(|name| {
            let texture_binding = next_binding;
            let sampler_binding = next_binding + 1;
            next_binding += 2;
            let is_cube = cube_samplers.contains(name);
            TextureBinding {
                name: name.clone(),
                group: 0,
                texture_binding,
                sampler_binding,
                is_diffuse: name == "tDiffuse",
                kind: if is_cube {
                    TextureKind::TexCube
                } else {
                    TextureKind::Tex2D
                },
                is_cube,
            }
        })
        .collect()
}

#[cfg(test)]
mod decls_tests {
    use super::*;

    #[test]
    fn parses_uniform_initializer_defaults_and_texture_kinds() {
        let src = r#"
            precision highp float;
            uniform float Gain = 2.0;
            uniform float Neg = -0.75, Plain;
            uniform vec3 Tint = vec3(1.0, 0.5, 0.25);
            uniform vec2 Broadcast = vec2(0.5);
            uniform int Steps = -4;
            uniform bool Enabled = true;
            uniform sampler2D tDiffuse;
            uniform samplerCube EnvMap;
            varying vec2 vUv;
            void main() {
                vec4 a = texture2D(tDiffuse, vUv);
                vec4 b = textureCube(EnvMap, vec3(vUv, 1.0));
                float s = Enabled ? Gain + Neg + Plain + float(Steps) : 0.0;
                gl_FragColor = vec4((a.rgb + b.rgb) * Tint + vec3(Broadcast, s * 0.01), 1.0);
            }
        "#;
        let compiled = translate_glsl_es100(src).expect("uniform defaults must compile");

        let gain = compiled.uniform_layout.field("Gain").unwrap();
        assert_eq!(gain.default_value.as_deref(), Some("2.0"));
        assert_eq!(gain.parsed_default_value(), Some(UniformValue::Float(2.0)));

        let neg = compiled.uniform_layout.field("Neg").unwrap();
        assert_eq!(neg.default_value.as_deref(), Some("-0.75"));
        assert_eq!(neg.parsed_default_value(), Some(UniformValue::Float(-0.75)));

        let plain = compiled.uniform_layout.field("Plain").unwrap();
        assert_eq!(plain.default_value, None);
        assert_eq!(plain.parsed_default_value(), None);

        let tint = compiled.uniform_layout.field("Tint").unwrap();
        assert_eq!(
            tint.parsed_default_value(),
            Some(UniformValue::Vec3([1.0, 0.5, 0.25]))
        );

        let broadcast = compiled.uniform_layout.field("Broadcast").unwrap();
        assert_eq!(
            broadcast.parsed_default_value(),
            Some(UniformValue::Vec2([0.5, 0.5]))
        );

        let steps = compiled.uniform_layout.field("Steps").unwrap();
        assert_eq!(steps.parsed_default_value(), Some(UniformValue::Int(-4)));

        let enabled = compiled.uniform_layout.field("Enabled").unwrap();
        assert_eq!(
            enabled.parsed_default_value(),
            Some(UniformValue::Bool(true))
        );

        assert_eq!(compiled.texture_bindings.len(), 2);
        let td = &compiled.texture_bindings[0];
        assert_eq!(td.name, "tDiffuse");
        assert!(td.is_diffuse);
        assert!(!td.is_cube);
        assert_eq!(td.kind, TextureKind::Tex2D);

        let env = &compiled.texture_bindings[1];
        assert_eq!(env.name, "EnvMap");
        assert!(!env.is_diffuse);
        assert!(env.is_cube);
        assert_eq!(env.kind, TextureKind::TexCube);
    }
}

// ---------------------------------------------------------------------------
// Stage 5: Rewrite Include Sampling Helpers, texture2D, gl_FragColor & Uniforms
