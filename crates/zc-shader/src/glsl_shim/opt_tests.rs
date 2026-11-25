use super::*;

#[test]
fn bind_helper_arguments_forward_copy_and_zero_alloc_uniform_scan() {
    let src = r#"precision highp float;
// uniform vec2 CurveA_range;
uniform highp vec2 CurveB_range;
uniform vec2 other, CurveC_range;
void main() {
float a = zCurve(CurveA, zCurve(CurveB, 0.5));
float c = zCurve(CurveC, a);
gl_FragColor = vec4(c);
}"#;
    let out = bind_curve_ranges(src, ["CurveA", "CurveB", "CurveC"]);
    assert!(out.contains("uniform vec2 CurveA_range;"));
    assert_eq!(out.matches("CurveB_range").count(), 2); // decl + call arg only
    assert_eq!(out.matches("CurveC_range").count(), 2); // decl + call arg only
    assert!(out.contains("zCurve(CurveA, zCurve(CurveB, 0.5, CurveB_range), CurveA_range)"));
    assert!(out.contains("zCurve(CurveC, a, CurveC_range)"));
}

#[test]
fn translation_cache_hits_respects_options_and_stays_bounded() {
    let src =
        "precision mediump float; uniform float Gain; void main() { gl_FragColor = vec4(Gain); }";
    let mut opts_a = CompileOptions::default();
    opts_a.defines.insert("MODE".to_string(), "1".to_string());
    let mut opts_b = CompileOptions::default();
    opts_b.defines.insert("MODE".to_string(), "2".to_string());

    let r1 = translate_glsl_es100_with_options(src, &opts_a).expect("compile a1");
    let r2 = translate_glsl_es100_with_options(src, &opts_a).expect("compile a2 cached");
    assert_eq!(r1, r2);

    let src_if = "precision mediump float; void main() {\n#if MODE == 1\n gl_FragColor = vec4(1.0);\n#else\n gl_FragColor = vec4(0.0);\n#endif\n}";
    let ra = translate_glsl_es100_with_options(src_if, &opts_a).expect("mode 1");
    let rb = translate_glsl_es100_with_options(src_if, &opts_b).expect("mode 2");
    assert_ne!(ra.wgsl, rb.wgsl);

    // Verify cache capacity cap is strictly enforced.
    let mut local_cache = TranslationCache::default();
    for i in 0..(TRANSLATION_CACHE_CAP + 32) {
        local_cache.insert(&format!("s{i}"), &opts_a, r1.clone());
    }
    assert_eq!(local_cache.len, TRANSLATION_CACHE_CAP);
    assert!(local_cache.get("s0", &opts_a).is_none());
    assert!(
        local_cache
            .get(&format!("s{}", TRANSLATION_CACHE_CAP + 31), &opts_a)
            .is_some()
    );
}

#[test]
fn pack_uniforms_lazy_normalized_lookup_handles_exact_and_spaced_keys() {
    let src = "precision mediump float; uniform float My_Param; uniform float Plain; void main() { gl_FragColor = vec4(My_Param + Plain); }";
    let compiled = translate_glsl_es100_with_options(src, &CompileOptions::default()).unwrap();

    let mut exact = BTreeMap::new();
    exact.insert("My_Param".to_string(), UniformValue::Float(2.5));
    exact.insert("Plain".to_string(), UniformValue::Float(4.0));
    let buf_exact = pack_uniforms(&compiled.uniform_layout, &exact);

    let mut spaced = BTreeMap::new();
    spaced.insert("My Param".to_string(), UniformValue::Float(2.5));
    spaced.insert("Plain".to_string(), UniformValue::Float(4.0));
    let buf_spaced = pack_uniforms(&compiled.uniform_layout, &spaced);
    assert_eq!(buf_exact, buf_spaced);
}
