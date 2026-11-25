use super::*;

type CallArgRanges = (Vec<(usize, usize)>, usize);

pub(super) fn parse_call_arg_ranges(
    tokens: &[Token],
    open_paren_idx: usize,
) -> Result<Option<CallArgRanges>, ShaderError> {
    let mut args = Vec::new();
    let mut arg_start = open_paren_idx + 1;
    let mut depth: usize = 1;
    let mut i = open_paren_idx + 1;

    while i < tokens.len() {
        match &tokens[i] {
            Token::Punct(p) if p == "(" => {
                depth += 1;
                if depth > MAX_REWRITE_DEPTH {
                    return Err(ShaderError::GlslParse(format!(
                        "parenthesis nesting depth exceeds {MAX_REWRITE_DEPTH} limit"
                    )));
                }
            }
            Token::Punct(p) if p == ")" => {
                depth -= 1;
                if depth == 0 {
                    if arg_start < i || !args.is_empty() {
                        args.push((arg_start, i));
                    }
                    return Ok(Some((args, i + 1)));
                }
            }
            Token::Punct(p) if p == "," && depth == 1 => {
                args.push((arg_start, i));
                arg_start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    Ok(None)
}

pub(super) fn expand_builtin_include_helpers(
    tokens: &[Token],
    included: &HashSet<String>,
    depth: usize,
    budget: &mut RewriteBudget,
) -> Result<Vec<Token>, ShaderError> {
    if depth > MAX_REWRITE_DEPTH {
        return Err(ShaderError::GlslParse(format!(
            "helper call nesting depth exceeds {MAX_REWRITE_DEPTH} limit"
        )));
    }
    let has_grad = included.contains("zoidium/gradient");
    let has_curve = included.contains("zoidium/curve");
    let has_text = included.contains("zoidium/text");
    if !has_grad && !has_curve && !has_text {
        return Ok(tokens.to_vec());
    }

    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;

    while i < tokens.len() {
        if let Token::Ident(name) = &tokens[i] {
            let is_helper = (has_grad && name == "zSampleGradient")
                || (has_curve && (name == "zSampleCurve" || name == "zSampleCurveValue"))
                || (has_text && name == "zSampleText");
            let next_i = next_non_newline(tokens, i + 1);
            if is_helper
                && matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "(")
                && let Some((arg_ranges, after_call)) = parse_call_arg_ranges(tokens, next_i)?
            {
                let mut args = Vec::with_capacity(arg_ranges.len());
                for (lo, hi) in arg_ranges {
                    args.push(expand_builtin_include_helpers(
                        &tokens[lo..hi],
                        included,
                        depth + 1,
                        budget,
                    )?);
                }

                if has_grad && name == "zSampleGradient" && (args.len() == 2 || args.len() == 3) {
                    let before_len = out.len();
                    out.push(Token::Ident("texture2D".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[0].iter().cloned());
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("vec2".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    if args.len() == 3 {
                        out.push(Token::Ident("zGradientUv".to_string()));
                        out.push(Token::Punct("(".to_string()));
                        out.extend(args[1].iter().cloned());
                        out.push(Token::Punct(",".to_string()));
                        out.extend(args[2].iter().cloned());
                        out.push(Token::Punct(")".to_string()));
                    } else {
                        out.push(Token::Ident("zGradientUvClamp".to_string()));
                        out.push(Token::Punct("(".to_string()));
                        out.extend(args[1].iter().cloned());
                        out.push(Token::Punct(")".to_string()));
                    }
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Number("0.5".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }

                if has_curve
                    && (name == "zSampleCurve" || name == "zSampleCurveValue")
                    && args.len() == 2
                {
                    let before_len = out.len();
                    out.push(Token::Ident("texture2D".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[0].iter().cloned());
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("vec2".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Ident("zCurveUv".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[1].iter().cloned());
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Number("0.5".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    if name == "zSampleCurveValue" {
                        out.push(Token::Punct(".".to_string()));
                        out.push(Token::Ident("r".to_string()));
                    }
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }

                if has_text && name == "zSampleText" && args.len() == 2 {
                    let before_len = out.len();
                    out.push(Token::Ident("texture2D".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[0].iter().cloned());
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("clamp".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[1].iter().cloned());
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("vec2".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Number("0.0".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("vec2".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Number("1.0".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }
            }
        }

        budget.charge(1)?;
        out.push(tokens[i].clone());
        i += 1;
    }

    Ok(out)
}

pub(super) fn is_glsl_builtin_type(ident: &str) -> bool {
    matches!(
        ident,
        "void"
            | "float"
            | "int"
            | "uint"
            | "bool"
            | "vec2"
            | "vec3"
            | "vec4"
            | "ivec2"
            | "ivec3"
            | "ivec4"
            | "uvec2"
            | "uvec3"
            | "uvec4"
            | "bvec2"
            | "bvec3"
            | "bvec4"
            | "mat2"
            | "mat3"
            | "mat4"
            | "sampler2D"
            | "samplerCube"
    )
}

/// Identifiers that are valid user names in GLSL ES 1.00 but collide with GLSL 450 / Vulkan
/// keywords or the GLSL 450 `texture()`/`textureProj()`/`textureLod()`/`textureGrad()` built-ins
/// emitted by this shim.
fn is_glsl450_escaped_ident(name: &str) -> bool {
    matches!(
        name,
        "sample"
            | "smooth"
            | "flat"
            | "noperspective"
            | "centroid"
            | "patch"
            | "subroutine"
            | "buffer"
            | "shared"
            | "coherent"
            | "volatile"
            | "restrict"
            | "readonly"
            | "writeonly"
            | "layout"
            | "precise"
            | "double"
            | "dvec2"
            | "dvec3"
            | "dvec4"
            | "dmat2"
            | "dmat3"
            | "dmat4"
            | "dmat2x2"
            | "dmat2x3"
            | "dmat2x4"
            | "dmat3x2"
            | "dmat3x3"
            | "dmat3x4"
            | "dmat4x2"
            | "dmat4x3"
            | "dmat4x4"
            | "atomic_uint"
            | "sampler"
            | "texture"
            | "textureProj"
            | "textureLod"
            | "textureGrad"
            | "textureProjLod"
            | "textureProjGrad"
    )
}

pub(super) fn escape_glsl450_ident(name: &str) -> String {
    if is_glsl450_escaped_ident(name) {
        format!("_zc_id_{name}")
    } else {
        name.to_string()
    }
}

/// Converts a `UniformDecl::name` (which may be a flattened struct member like `uLight.color`
/// or `uLights[0].color`) into a valid GLSL 450 identifier for `ZcCustomUniforms`.
pub(super) fn ubo_field_ident(name: &str) -> String {
    if !name.contains(['.', '[', ']']) {
        return escape_glsl450_ident(name);
    }
    let mut out = String::from("_zc_sf_");
    for ch in name.chars() {
        match ch {
            '.' => out.push_str("__"),
            '[' => out.push_str("_i"),
            ']' => out.push('_'),
            _ => out.push(ch),
        }
    }
    out
}

pub(super) fn first_single_ident(arg: &[Token]) -> Option<&str> {
    let mut slice = arg;
    loop {
        while matches!(slice.first(), Some(Token::Newline)) {
            slice = &slice[1..];
        }
        while matches!(slice.last(), Some(Token::Newline)) {
            slice = &slice[..slice.len() - 1];
        }
        if slice.len() >= 2
            && matches!(slice.first(), Some(Token::Punct(p)) if p == "(")
            && matches!(slice.last(), Some(Token::Punct(p)) if p == ")")
        {
            let mut depth = 0_usize;
            let mut wraps_all = true;
            for (idx, tok) in slice.iter().enumerate() {
                match tok {
                    Token::Punct(p) if p == "(" => depth += 1,
                    Token::Punct(p) if p == ")" => {
                        depth -= 1;
                        if depth == 0 && idx + 1 < slice.len() {
                            wraps_all = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if wraps_all && depth == 0 {
                slice = &slice[1..slice.len() - 1];
                continue;
            }
        }
        break;
    }
    if let [Token::Ident(name)] = slice {
        Some(name.as_str())
    } else {
        None
    }
}

pub(super) fn parse_bracket_expr(
    tokens: &[Token],
    open_bracket_idx: usize,
) -> Option<(Vec<Token>, usize)> {
    let mut inner = Vec::new();
    let mut depth: usize = 1;
    let mut i = open_bracket_idx + 1;

    while i < tokens.len() {
        match &tokens[i] {
            Token::Punct(p) if p == "[" => {
                depth += 1;
                inner.push(tokens[i].clone());
            }
            Token::Punct(p) if p == "]" => {
                depth -= 1;
                if depth == 0 {
                    return Some((inner, i + 1));
                }
                inner.push(tokens[i].clone());
            }
            _ => inner.push(tokens[i].clone()),
        }
        i += 1;
    }
    None
}

pub(super) type SamplerFnMap = HashMap<String, HashMap<usize, HashSet<usize>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FragCoordResY {
    None,
    Float,
    CastFloat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScopeKind {
    Root,
    Brace,
    ForHeader { paren_depth: usize },
    ForBodyStart,
    ForSingleStmt { if_depth: usize },
}

#[derive(Debug, Clone)]
pub(super) struct ScopeFrame {
    pub(super) vars: HashSet<String>,
    pub(super) samplers: HashSet<String>,
    pub(super) cube_samplers: HashSet<String>,
    pub(super) const_ints: HashMap<String, i64>,
    pub(super) kind: ScopeKind,
}

impl ScopeFrame {
    pub(super) fn new(kind: ScopeKind) -> Self {
        Self {
            vars: HashSet::new(),
            samplers: HashSet::new(),
            cube_samplers: HashSet::new(),
            const_ints: HashMap::new(),
            kind,
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct RewriteState {
    /// Names of `mat2[N]` uniforms indexed dynamically (`_zc_load_<ubo_field>(int)`).
    pub(super) mat2_array_loaders: HashSet<String>,
    /// Names of widened uniform arrays passed bare (`_zc_unpack_<ubo_field>()`).
    pub(super) array_unpackers: HashSet<String>,
    /// Whether any non-const `normalize(...)` call was rewritten to `_zc_normalize(...)`.
    pub(super) uses_zc_normalize: bool,
    /// Whether any non-const `pow(...)` call was rewritten to `_zc_pow(...)`.
    pub(super) uses_zc_pow: bool,
    /// Whether `_zc_flip_grad` is referenced.
    pub(super) uses_flip_grad: bool,
}

pub(super) struct RewriteCtx<'a> {
    pub(super) uniforms: &'a HashMap<&'a str, &'a UniformDecl>,
    pub(super) struct_uniforms: &'a HashMap<String, StructUniformDecl>,
    pub(super) struct_defs: &'a HashMap<String, Vec<StructFieldDecl>>,
    pub(super) global_const_ints: &'a HashMap<String, u32>,
    pub(super) samplers: &'a HashSet<&'a str>,
    pub(super) cube_samplers: &'a HashSet<String>,
    pub(super) sampler_fns: &'a SamplerFnMap,
    pub(super) user_fns: &'a HashSet<String>,
    pub(super) frag_coord_res_y: FragCoordResY,
}

pub(super) fn tokens_reference_ident(tokens: &[Token], target: &str) -> bool {
    for tok in tokens {
        match tok {
            Token::Ident(name) if name == target => return true,
            Token::Preprocessor(line) if line.contains(target) => {
                let body = line.trim_start().strip_prefix('#').unwrap_or(line);
                if tokenize(body)
                    .iter()
                    .any(|t| matches!(t, Token::Ident(name) if name == target))
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

pub(super) fn frag_coord_resolution_mode(uniforms: &HashMap<&str, &UniformDecl>) -> FragCoordResY {
    let Some(decl) = uniforms.get("resolution") else {
        return FragCoordResY::None;
    };
    if decl.array_len.is_some() {
        return FragCoordResY::None;
    }
    match decl.ty {
        UniformType::Vec2 | UniformType::Vec3 | UniformType::Vec4 => FragCoordResY::Float,
        UniformType::Ivec2
        | UniformType::Ivec3
        | UniformType::Ivec4
        | UniformType::Uvec2
        | UniformType::Uvec3
        | UniformType::Uvec4 => FragCoordResY::CastFloat,
        _ => FragCoordResY::None,
    }
}

pub(super) fn emit_gl_frag_coord(out: &mut Vec<Token>, res_y: FragCoordResY) {
    match res_y {
        FragCoordResY::None => {
            out.push(Token::Ident("gl_FragCoord".to_string()));
        }
        FragCoordResY::Float | FragCoordResY::CastFloat => {
            out.push(Token::Punct("(".to_string()));
            out.push(Token::Ident("vec4".to_string()));
            out.push(Token::Punct("(".to_string()));
            out.push(Token::Ident("gl_FragCoord".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("x".to_string()));
            out.push(Token::Punct(",".to_string()));
            if res_y == FragCoordResY::CastFloat {
                out.push(Token::Ident("float".to_string()));
                out.push(Token::Punct("(".to_string()));
            }
            out.push(Token::Ident("_zc_u".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("resolution".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("y".to_string()));
            if res_y == FragCoordResY::CastFloat {
                out.push(Token::Punct(")".to_string()));
            }
            out.push(Token::Punct("-".to_string()));
            out.push(Token::Ident("gl_FragCoord".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("y".to_string()));
            out.push(Token::Punct(",".to_string()));
            out.push(Token::Ident("gl_FragCoord".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("z".to_string()));
            out.push(Token::Punct(",".to_string()));
            out.push(Token::Ident("gl_FragCoord".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("w".to_string()));
            out.push(Token::Punct(")".to_string()));
            out.push(Token::Punct(")".to_string()));
        }
    }
}

pub(super) fn collect_function_info(
    tokens: &[Token],
) -> Result<(SamplerFnMap, HashSet<String>), ShaderError> {
    let mut map: SamplerFnMap = HashMap::new();
    let mut user_fns: HashSet<String> = HashSet::new();
    let mut brace_depth: usize = 0;
    let mut prev_non_newline: Option<&Token> = None;
    let mut i = 0;

    while i < tokens.len() {
        match &tokens[i] {
            Token::Newline => {
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "{" => {
                brace_depth += 1;
            }
            Token::Punct(p) if p == "}" => {
                brace_depth = brace_depth.saturating_sub(1);
            }
            Token::Ident(fn_name)
                if brace_depth == 0
                    && matches!(prev_non_newline, Some(Token::Ident(_)))
                    && !matches!(
                        fn_name.as_str(),
                        "if" | "for"
                            | "while"
                            | "switch"
                            | "return"
                            | "else"
                            | "do"
                            | "struct"
                            | "uniform"
                            | "varying"
                            | "precision"
                    ) =>
            {
                let next_i = next_non_newline(tokens, i + 1);
                if matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "(")
                    && let Some((param_ranges, after_paren)) =
                        parse_call_arg_ranges(tokens, next_i)?
                {
                    let after_tok = next_non_newline(tokens, after_paren);
                    if matches!(
                        tokens.get(after_tok),
                        Some(Token::Punct(p)) if p == "{" || p == ";"
                    ) {
                        user_fns.insert(fn_name.clone());
                        let mut sampler_indices = HashSet::new();
                        for (idx, &(lo, hi)) in param_ranges.iter().enumerate() {
                            if tokens[lo..hi].iter().any(|t| {
                                matches!(t, Token::Ident(s) if s == "sampler2D" || s == "samplerCube")
                            }) {
                                sampler_indices.insert(idx);
                            }
                        }
                        if !sampler_indices.is_empty() {
                            map.entry(fn_name.clone())
                                .or_default()
                                .entry(param_ranges.len())
                                .or_default()
                                .extend(sampler_indices);
                        }
                    }
                }
            }
            _ => {}
        }
        prev_non_newline = Some(&tokens[i]);
        i += 1;
    }

    Ok((map, user_fns))
}

pub(super) fn is_shadowed(name: &str, scopes: &[ScopeFrame]) -> bool {
    scopes.iter().rev().any(|scope| scope.vars.contains(name))
}

pub(super) fn is_sampler_in_scope(
    name: &str,
    samplers: &HashSet<&str>,
    scopes: &[ScopeFrame],
) -> bool {
    for scope in scopes.iter().rev() {
        if scope.samplers.contains(name) {
            return true;
        }
        if scope.vars.contains(name) {
            return false;
        }
    }
    samplers.contains(name)
}

pub(super) fn flush_pending_decl(pending_decl_var: &mut Option<String>, scopes: &mut [ScopeFrame]) {
    if let Some(var) = pending_decl_var.take()
        && let Some(current_scope) = scopes.last_mut()
    {
        current_scope.vars.insert(var);
    }
}

pub(super) fn pop_completed_for_scopes(scopes: &mut Vec<ScopeFrame>, next_is_else: bool) {
    while let Some(top) = scopes.last_mut() {
        match &mut top.kind {
            ScopeKind::ForBodyStart => {
                scopes.pop();
            }
            ScopeKind::ForSingleStmt { if_depth } => {
                if next_is_else && *if_depth > 0 {
                    *if_depth -= 1;
                    break;
                }
                scopes.pop();
            }
            _ => break,
        }
    }
}

pub(super) fn is_cube_sampler_in_scope(
    name: &str,
    cube_samplers: &HashSet<String>,
    scopes: &[ScopeFrame],
) -> bool {
    for scope in scopes.iter().rev() {
        if scope.cube_samplers.contains(name) {
            return true;
        }
        if scope.samplers.contains(name) || scope.vars.contains(name) {
            return false;
        }
    }
    cube_samplers.contains(name)
}

pub(super) fn eval_const_int_with_scopes(
    tokens: &[Token],
    scopes: &[ScopeFrame],
    global_const_ints: &HashMap<String, u32>,
) -> Option<(i64, bool)> {
    // Build effective map respecting shadowing from inner to outer scopes.
    let mut effective: HashMap<String, u32> = global_const_ints.clone();
    let mut local_names: HashSet<String> = HashSet::new();
    for scope in scopes.iter().skip(1) {
        for var in &scope.vars {
            if !scope.const_ints.contains_key(var) {
                effective.remove(var);
                local_names.remove(var);
            }
        }
        for (k, &v) in &scope.const_ints {
            if let Ok(u) = u32::try_from(v) {
                effective.insert(k.clone(), u);
                local_names.insert(k.clone());
            } else {
                effective.remove(k);
                local_names.remove(k);
            }
        }
    }
    let uses_local = tokens
        .iter()
        .any(|t| matches!(t, Token::Ident(name) if local_names.contains(name)));
    let val = eval_const_int_tokens(tokens, &effective)?;
    Some((val, uses_local))
}

pub(super) fn emit_non_array_uniform_read(out: &mut Vec<Token>, decl: &UniformDecl) {
    let field_id = ubo_field_ident(&decl.name);
    match decl.ty {
        UniformType::Bool => {
            out.push(Token::Punct("(".to_string()));
            out.push(Token::Ident("_zc_u".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident(field_id));
            out.push(Token::Punct("!=".to_string()));
            out.push(Token::Number("0u".to_string()));
            out.push(Token::Punct(")".to_string()));
        }
        UniformType::Bvec2 | UniformType::Bvec3 | UniformType::Bvec4 => {
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
            out.push(Token::Punct(",".to_string()));
            out.push(Token::Ident(zero_ty.to_string()));
            out.push(Token::Punct("(".to_string()));
            out.push(Token::Number("0u".to_string()));
            out.push(Token::Punct(")".to_string()));
            out.push(Token::Punct(")".to_string()));
        }
        UniformType::Mat2 => {
            out.push(Token::Punct("(".to_string()));
            out.push(Token::Ident("mat2".to_string()));
            out.push(Token::Punct("(".to_string()));
            out.push(Token::Ident("_zc_u".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident(field_id.clone()));
            out.push(Token::Punct("[".to_string()));
            out.push(Token::Number("0".to_string()));
            out.push(Token::Punct("]".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("xy".to_string()));
            out.push(Token::Punct(",".to_string()));
            out.push(Token::Ident("_zc_u".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident(field_id));
            out.push(Token::Punct("[".to_string()));
            out.push(Token::Number("1".to_string()));
            out.push(Token::Punct("]".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident("xy".to_string()));
            out.push(Token::Punct(")".to_string()));
            out.push(Token::Punct(")".to_string()));
        }
        _ => {
            out.push(Token::Ident("_zc_u".to_string()));
            out.push(Token::Punct(".".to_string()));
            out.push(Token::Ident(field_id));
        }
    }
}

pub(super) fn emit_bare_array_uniform_read(
    out: &mut Vec<Token>,
    decl: &UniformDecl,
    state: &mut RewriteState,
) {
    let field_id = ubo_field_ident(&decl.name);
    if decl.ty.needs_array_unpack() {
        state.array_unpackers.insert(decl.name.clone());
        out.push(Token::Ident(format!("_zc_unpack_{field_id}")));
        out.push(Token::Punct("(".to_string()));
        out.push(Token::Punct(")".to_string()));
    } else {
        out.push(Token::Ident("_zc_u".to_string()));
        out.push(Token::Punct(".".to_string()));
        out.push(Token::Ident(field_id));
    }
}

pub(super) fn emit_struct_uniform_expr(
    out: &mut Vec<Token>,
    prefix: &str,
    struct_name: &str,
    ctx: &RewriteCtx<'_>,
    state: &mut RewriteState,
    depth: usize,
) -> Result<(), ShaderError> {
    if depth > 16 {
        return Err(ShaderError::GlslParse(format!(
            "struct uniform `{prefix}` exceeds maximum nesting depth"
        )));
    }
    let Some(fields) = ctx.struct_defs.get(struct_name) else {
        return Err(ShaderError::GlslParse(format!(
            "unknown struct type `{struct_name}`"
        )));
    };
    out.push(Token::Ident(escape_glsl450_ident(struct_name)));
    out.push(Token::Punct("(".to_string()));
    for (idx, field) in fields.iter().enumerate() {
        if idx > 0 {
            out.push(Token::Punct(",".to_string()));
        }
        let member_key = format!("{prefix}.{}", field.name);
        if let Some(decl) = ctx.uniforms.get(member_key.as_str()) {
            if decl.array_len.is_some() {
                emit_bare_array_uniform_read(out, decl, state);
            } else {
                emit_non_array_uniform_read(out, decl);
            }
        } else if ctx.struct_defs.contains_key(&field.ty_name) {
            if let Some(len) = field.array_len {
                out.push(Token::Ident(escape_glsl450_ident(&field.ty_name)));
                out.push(Token::Punct("[".to_string()));
                out.push(Token::Number(len.to_string()));
                out.push(Token::Punct("]".to_string()));
                out.push(Token::Punct("(".to_string()));
                for k in 0..len {
                    if k > 0 {
                        out.push(Token::Punct(",".to_string()));
                    }
                    let elem_prefix = format!("{member_key}[{k}]");
                    emit_struct_uniform_expr(
                        out,
                        &elem_prefix,
                        &field.ty_name,
                        ctx,
                        state,
                        depth + 1,
                    )?;
                }
                out.push(Token::Punct(")".to_string()));
            } else {
                emit_struct_uniform_expr(out, &member_key, &field.ty_name, ctx, state, depth + 1)?;
            }
        }
    }
    out.push(Token::Punct(")".to_string()));
    Ok(())
}
