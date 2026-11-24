#![forbid(unsafe_code)]

use zc_shader::{MAX_GLSL_SOURCE_BYTES, ShaderError, translate_glsl_es100};

#[test]
fn rejects_deeply_nested_texture2d_and_oversized_shader_sources() {
    // 1. Deeply nested texture2D calls (depth 80 > MAX_REWRITE_DEPTH 64).
    let mut expr = String::from("vUv");
    for _ in 0..80 {
        expr = format!("texture2D(tDiffuse, ({expr}).xy)");
    }
    let nested_src = format!(
        "uniform sampler2D tDiffuse;\nvarying vec2 vUv;\nvoid main() {{ gl_FragColor = {expr}; }}\n"
    );
    let err = translate_glsl_es100(&nested_src)
        .expect_err("deeply nested texture2D calls must be rejected before rewriting");
    assert!(
        matches!(err, ShaderError::GlslParse(_)),
        "expected GlslParse error, got {err:?}"
    );

    // 2. Oversized GLSL source (> MAX_GLSL_SOURCE_BYTES).
    let mut huge = String::with_capacity(MAX_GLSL_SOURCE_BYTES + 128);
    huge.push_str("void main() { gl_FragColor = vec4(1.0); }\n// ");
    huge.push_str(&"a".repeat(MAX_GLSL_SOURCE_BYTES + 64));
    let err = translate_glsl_es100(&huge).expect_err("oversized shader source must be rejected");
    assert!(
        matches!(err, ShaderError::GlslParse(_)),
        "expected GlslParse error, got {err:?}"
    );
}
