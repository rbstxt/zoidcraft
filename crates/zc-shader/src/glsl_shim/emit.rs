use super::*;

pub(super) fn render_tokens(tokens: &[Token]) -> String {
    let mut out = String::new();
    let mut prev: Option<&Token> = None;

    for tok in tokens {
        match tok {
            Token::Newline => {
                if !out.ends_with('\n') {
                    out.push('\n');
                }
                prev = Some(tok);
            }
            Token::Preprocessor(line) => {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str(line);
                out.push('\n');
                prev = Some(&Token::Newline);
            }
            Token::Ident(s) | Token::Number(s) => {
                if let Some(p) = prev {
                    match p {
                        Token::Ident(_) | Token::Number(_) => out.push(' '),
                        Token::Punct(ps) if !matches!(ps.as_str(), "." | "(" | "[" | "!" | "~") => {
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
                    let need_space = match (p, s.as_str()) {
                        (Token::Newline, _) => false,
                        (_, "." | ";" | "," | ")" | "]" | "(" | "[") => false,
                        (Token::Punct(ps), _) if matches!(ps.as_str(), "." | "(" | "[") => false,
                        (Token::Punct(ps), cur)
                            if (ps == "-" && cur == "-")
                                || (ps == "+" && cur == "+")
                                || (ps == "/" && cur == "*") =>
                        {
                            true
                        }
                        (Token::Ident(_) | Token::Number(_), "++" | "--") => false,
                        _ => true,
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

    out
}

// ---------------------------------------------------------------------------
// Stage 6: Assemble Vulkan GLSL 450 Module

pub(super) fn assemble_glsl450(
    extra_varyings: &[VaryingDecl],
    uniforms: &[UniformDecl],
    texture_bindings: &[TextureBinding],
    cube_samplers: &HashSet<String>,
    rewrite_state: &RewriteState,
    body_glsl: &str,
) -> String {
    let mut src = String::with_capacity(body_glsl.len() + 512);
    src.push_str("#version 450\n");
    src.push_str("#define GL_ES 1\n");
    src.push_str("#define GL_FRAGMENT_PRECISION_HIGH 1\n");
    src.push_str("#define GL_OES_standard_derivatives 1\n");
    src.push_str("#define GL_EXT_shader_texture_lod 1\n");

    src.push_str("\nlayout(location = 0) in vec2 vUv;\n");
    src.push_str("layout(location = 1) in vec2 vUvScaled;\n");
    let mut next_loc: u32 = 2;
    for var in extra_varyings {
        let esc_name = escape_glsl450_ident(&var.name);
        let cols: u32 = match var.ty.as_str() {
            "mat2" => 2,
            "mat3" => 3,
            "mat4" => 4,
            _ => 1,
        };
        if let Some(len) = var.array_len {
            src.push_str(&format!(
                "layout(location = {next_loc}) in {} {esc_name}[{len}];\n",
                var.ty
            ));
            next_loc += cols.saturating_mul(len);
        } else {
            src.push_str(&format!(
                "layout(location = {next_loc}) in {} {esc_name};\n",
                var.ty
            ));
            next_loc += cols;
        }
    }
    src.push_str("layout(location = 0) out vec4 _zc_frag_color;\n");

    if !uniforms.is_empty() {
        src.push_str("\nlayout(set = 0, binding = 0, std140) uniform ZcCustomUniforms {\n");
        for u in uniforms {
            let glsl_ty = u.ty.ubo_glsl_type(u.array_len.is_some());
            let field_id = ubo_field_ident(&u.name);
            if let Some(len) = u.array_len {
                if u.ty == UniformType::Mat2 {
                    src.push_str(&format!(
                        "    vec4 {field_id}[{}];\n",
                        len.saturating_mul(2)
                    ));
                } else {
                    src.push_str(&format!("    {glsl_ty} {field_id}[{len}];\n"));
                }
            } else if u.ty == UniformType::Mat2 {
                src.push_str(&format!("    vec4 {field_id}[2];\n"));
            } else {
                src.push_str(&format!("    {glsl_ty} {field_id};\n"));
            }
        }
        src.push_str("} _zc_u;\n");

        // Emit dynamic loaders for `uniform mat2 A[N]` indexed reads.
        for u in uniforms {
            if u.ty == UniformType::Mat2
                && u.array_len.is_some()
                && rewrite_state.mat2_array_loaders.contains(&u.name)
            {
                let field_id = ubo_field_ident(&u.name);
                src.push_str(&format!(
                    "mat2 _zc_load_{field_id}(int _zc_i) {{ return mat2(_zc_u.{field_id}[_zc_i * 2].xy, _zc_u.{field_id}[_zc_i * 2 + 1].xy); }}\n"
                ));
            }
        }

        // Emit unpack helpers for widened uniform arrays passed bare to functions.
        for u in uniforms {
            if let Some(len) = u.array_len
                && u.ty.needs_array_unpack()
                && rewrite_state.array_unpackers.contains(&u.name)
            {
                let field_id = ubo_field_ident(&u.name);
                let glsl_ty = u.ty.glsl_name();
                src.push_str(&format!(
                    "{glsl_ty}[{len}] _zc_unpack_{field_id}() {{ return {glsl_ty}[{len}]("
                ));
                for idx in 0..len {
                    if idx > 0 {
                        src.push_str(", ");
                    }
                    match u.ty {
                        UniformType::Bool => {
                            src.push_str(&format!("(_zc_u.{field_id}[{idx}].x != 0u)"));
                        }
                        UniformType::Bvec2 => {
                            src.push_str(&format!(
                                "notEqual(_zc_u.{field_id}[{idx}].xy, uvec2(0u))"
                            ));
                        }
                        UniformType::Bvec3 => {
                            src.push_str(&format!("notEqual(_zc_u.{field_id}[{idx}], uvec3(0u))"));
                        }
                        UniformType::Bvec4 => {
                            src.push_str(&format!("notEqual(_zc_u.{field_id}[{idx}], uvec4(0u))"));
                        }
                        UniformType::Mat2 => {
                            let c0 = idx * 2;
                            let c1 = idx * 2 + 1;
                            src.push_str(&format!(
                                "mat2(_zc_u.{field_id}[{c0}].xy, _zc_u.{field_id}[{c1}].xy)"
                            ));
                        }
                        _ => {
                            let swizzle = u.ty.array_swizzle().unwrap_or("");
                            src.push_str(&format!("_zc_u.{field_id}[{idx}]{swizzle}"));
                        }
                    }
                }
                src.push_str("); }\n");
            }
        }
    }

    let emit_uv_helpers = !texture_bindings.is_empty() || body_glsl.contains("_zc_flip_uv");
    if !texture_bindings.is_empty() || emit_uv_helpers {
        src.push('\n');
        for tb in texture_bindings {
            let esc_name = escape_glsl450_ident(&tb.name);
            let tex_ty = if cube_samplers.contains(&tb.name) {
                "textureCube"
            } else {
                "texture2D"
            };
            src.push_str(&format!(
                "layout(set = {}, binding = {}) uniform {tex_ty} {esc_name};\n",
                tb.group, tb.texture_binding
            ));
            src.push_str(&format!(
                "layout(set = {}, binding = {}) uniform sampler {esc_name}_sampler;\n",
                tb.group, tb.sampler_binding
            ));
        }
        if emit_uv_helpers {
            src.push_str("vec2 _zc_flip_uv(vec2 uv) { return vec2(uv.x, 1.0 - uv.y); }\n");
            src.push_str(
                "vec3 _zc_flip_uv_proj(vec3 uvw) { return vec3(uvw.x, uvw.z - uvw.y, uvw.z); }\n",
            );
            src.push_str(
                "vec4 _zc_flip_uv_proj(vec4 uvw) { return vec4(uvw.x, uvw.w - uvw.y, uvw.z, uvw.w); }\n",
            );
        }
    }
    if rewrite_state.uses_flip_grad || body_glsl.contains("_zc_flip_grad") {
        src.push_str("vec2 _zc_flip_grad(vec2 g) { return vec2(g.x, -g.y); }\n");
    }
    if rewrite_state.uses_zc_normalize {
        src.push_str("float _zc_normalize(float v) { return v / abs(v); }\n");
        src.push_str("vec2 _zc_normalize(vec2 v) { return normalize(v); }\n");
        src.push_str("vec3 _zc_normalize(vec3 v) { return normalize(v); }\n");
        src.push_str("vec4 _zc_normalize(vec4 v) { return normalize(v); }\n");
    }
    if rewrite_state.uses_zc_pow {
        src.push_str("float _zc_pow(float x, float y) { return pow(x, y); }\n");
        src.push_str("vec2 _zc_pow(vec2 x, vec2 y) { return pow(x, y); }\n");
        src.push_str("vec3 _zc_pow(vec3 x, vec3 y) { return pow(x, y); }\n");
        src.push_str("vec4 _zc_pow(vec4 x, vec4 y) { return pow(x, y); }\n");
    }

    src.push('\n');
    src.push_str(body_glsl.trim());
    src.push('\n');
    src
}
