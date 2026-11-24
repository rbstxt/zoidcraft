//! CI-safe golden corpus & Naga WGSL validation test suite for `zc-shader`,
//! anchored directly to Zoidium's:
//! - `plugins/shader-plus/code-templates.js` (all 9 templates)
//! - `plugins/shader-plus/glsl-library.js` + `plugins/expressions-plus/glsl-library.js`
//!   (all 8 `#include <zoidium/*>` libraries)
//! - `test/fixtures/node-effect-glsl.json` (`vectorMath-4-*` const-eval regressions & complex nodes)
//! - `plugins/alipfx/shaders/mosaic.glsl` (`GL_EXT_shader_texture_lod` + dead-code-after-return)
//! - `mat2`, `struct`, `bvec*`, and `samplerCube` edge cases
//!
//! Every translated WGSL module is parsed with `naga::front::wgsl::parse_str` and validated
//! with `naga::valid::Validator::new(ValidationFlags::all(), Capabilities::empty())`.
//!
//! All golden shaders are embedded directly so this suite ALWAYS runs on Linux/Windows/macOS CI.
//! When a sibling `/Users/dia/Projects/GitHub/Zoidium` checkout is present locally, an
//! additional corpus sweep validates standalone Zoidium plugin fragment shaders as well.

use std::path::{Path, PathBuf};

use naga::valid::{Capabilities, ValidationFlags, Validator};
use zc_shader::{
    CompileOptions, CompiledCustomShader, bind_curve_ranges, bind_gradient_cycles,
    translate_glsl_es100, translate_glsl_es100_with_options,
};

/// Translates GLSL ES 1.00 -> GLSL 450 -> WGSL and strictly validates the resulting WGSL
/// module with Naga (`ValidationFlags::all()`, `Capabilities::empty()`).
fn compile_and_validate_wgsl(
    name: &str,
    glsl_src: &str,
    options: &CompileOptions,
) -> CompiledCustomShader {
    let compiled = translate_glsl_es100_with_options(glsl_src, options).unwrap_or_else(|err| {
        panic!("[{name}] GLSL ES 1.00 -> WGSL translation failed: {err}");
    });
    assert!(
        compiled.wgsl.contains("@fragment"),
        "[{name}] generated WGSL missing @fragment entry point"
    );

    let module = naga::front::wgsl::parse_str(&compiled.wgsl).unwrap_or_else(|err| {
        panic!(
            "[{name}] naga WGSL parse failed:\n{}\n--- WGSL ---\n{}",
            err.emit_to_string(&compiled.wgsl),
            compiled.wgsl
        );
    });
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
    validator.validate(&module).unwrap_or_else(|err| {
        panic!(
            "[{name}] naga WGSL validation failed:\n{}\n--- WGSL ---\n{}",
            err.emit_to_string(&compiled.wgsl),
            compiled.wgsl
        );
    });
    compiled
}

// ===========================================================================
// 1. All 9 Shader+ `code-templates.js` Templates
// ===========================================================================

const SHADER_PLUS_TEMPLATES: [(&str, &str); 9] = [
    (
        "passthrough",
        "precision highp float;\nuniform sampler2D tDiffuse;\nvarying vec2 vUvScaled;\nvoid main() {\n  gl_FragColor = texture2D(tDiffuse, vUvScaled);\n}",
    ),
    (
        "color",
        "precision highp float;\nuniform sampler2D tDiffuse;\nuniform float Amount;\nvarying vec2 vUvScaled;\nvoid main() {\n  vec4 color = texture2D(tDiffuse, vUvScaled);\n  gl_FragColor = vec4(color.rgb * (0.5 + Amount), color.a);\n}",
    ),
    (
        "gradient",
        "precision highp float;\n// #include <zoidium/gradient>\nuniform sampler2D tDiffuse;\nuniform sampler2D Ramp;\nvarying vec2 vUvScaled;\nvoid main() {\n  vec4 color = texture2D(tDiffuse, vUvScaled);\n  float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n  gl_FragColor = vec4(zGradient(Ramp, luma).rgb, color.a);\n}",
    ),
    (
        "curve",
        "precision highp float;\n// #include <zoidium/curve>\nuniform sampler2D tDiffuse;\nuniform sampler2D Remap;\nvarying vec2 vUvScaled;\nvoid main() {\n  vec4 color = texture2D(tDiffuse, vUvScaled);\n  gl_FragColor = vec4(zCurves(Remap, color.rgb), color.a);\n}",
    ),
    (
        "noise",
        "precision highp float;\n// #include <zoidium/noise>\nuniform float zTime;\nvarying vec2 vUv;\nvoid main() {\n  float n = zNoise(vec3(vUv * 6.0, zTime));\n  gl_FragColor = vec4(vec3(n), 1.0);\n}",
    ),
    (
        "text",
        "precision highp float;\n// #include <zoidium/text>\nuniform sampler2D tDiffuse;\nuniform sampler2D Title;\nvarying vec2 vUvScaled;\nvarying vec2 vUv;\nvoid main() {\n  vec4 color = texture2D(tDiffuse, vUvScaled);\n  float coverage = zText(Title, vUv);\n  gl_FragColor = vec4(mix(color.rgb, vec3(1.0), coverage), max(color.a, coverage));\n}",
    ),
    (
        "velocity",
        "precision highp float;\nuniform sampler2D tDiffuse;\nuniform vec2 Velocity;\nvarying vec2 vUvScaled;\nvoid main() {\n  gl_FragColor = texture2D(tDiffuse, vUvScaled + Velocity * 0.01);\n}",
    ),
    (
        "time-motion",
        "precision highp float;\n// #include <zoidium/temporal>\nuniform sampler2D Before;\nuniform sampler2D After;\nuniform vec2 resolution;\nuniform vec2 uvScale;\nvarying vec2 vUvScaled;\nvoid main() {\n  vec2 motion = zTimeMotion(Before, After, vUvScaled, uvScale / resolution);\n  gl_FragColor = vec4(vec2(0.5) + motion * 20.0, 0.5, 1.0);\n}",
    ),
    (
        "time-jitter",
        "precision highp float;\nuniform sampler2D Jitter;\nvarying vec2 vUvScaled;\nvoid main() {\n  gl_FragColor = texture2D(Jitter, vUvScaled);\n}",
    ),
];

#[test]
fn all_nine_shader_plus_templates_translate_and_validate_in_naga() {
    let opts = CompileOptions::default();
    for (id, source) in SHADER_PLUS_TEMPLATES {
        let compiled = compile_and_validate_wgsl(&format!("shader-plus/{id}"), source, &opts);
        if id == "time-motion" {
            assert_eq!(compiled.texture_bindings.len(), 2);
        }
    }

    // Also verify `bind_curve_ranges` on the `curve` template and `bind_gradient_cycles` on a
    // `zGradientCycle` shader match Zoidium `glsl-library.js` argument injection.
    let bound_curve = bind_curve_ranges(SHADER_PLUS_TEMPLATES[3].1, ["Remap"]);
    assert!(bound_curve.contains("uniform vec2 Remap_range;"));
    assert!(bound_curve.contains("zCurves(Remap, color.rgb, Remap_range)"));
    let compiled_curve = compile_and_validate_wgsl("shader-plus/curve-bound", &bound_curve, &opts);
    assert!(compiled_curve.uniform_layout.field("Remap_range").is_some());

    let grad_cycle_src = "precision highp float;\n// #include <zoidium/gradient>\nuniform sampler2D Ramp;\nvarying vec2 vUv;\nvoid main() {\n  gl_FragColor = zGradientCycle(Ramp, vUv.x);\n}";
    let bound_grad = bind_gradient_cycles(grad_cycle_src, ["Ramp"]);
    assert!(bound_grad.contains("uniform float Ramp_repetitions;"));
    assert!(bound_grad.contains("zGradientCycle(Ramp, vUv.x, Ramp_repetitions)"));
    let compiled_grad =
        compile_and_validate_wgsl("shader-plus/grad-cycle-bound", &bound_grad, &opts);
    assert!(
        compiled_grad
            .uniform_layout
            .field("Ramp_repetitions")
            .is_some()
    );
}

// ===========================================================================
// 2. All 8 `#include <zoidium/*>` Libraries (Individual + Combined Deduplication)
// ===========================================================================

#[test]
fn all_eight_zoidium_include_libraries_validate_individually_and_combined() {
    let opts = CompileOptions::default();

    let single_library_cases: [(&str, &str); 8] = [
        (
            "temporal",
            "precision highp float;\n// #include <zoidium/temporal>\nuniform sampler2D A;\nuniform sampler2D B;\nvarying vec2 vUv;\nvoid main() {\n  vec2 m = zTimeMotion(A, B, vUv, vec2(0.001));\n  gl_FragColor = vec4(m, 0.0, 1.0);\n}",
        ),
        (
            "gradient",
            "precision highp float;\n// #include <zoidium/gradient>\nuniform sampler2D Ramp;\nvarying vec2 vUv;\nvoid main() {\n  vec4 g = zGradientCycle(Ramp, vUv.x, 2.0);\n  gl_FragColor = vec4(zLinearToSrgb(zSrgbToLinear(g.rgb)), g.a);\n}",
        ),
        (
            "curve",
            "precision highp float;\n// #include <zoidium/curve>\nuniform sampler2D Curve;\nvarying vec2 vUv;\nvoid main() {\n  float c = zCurve(Curve, vUv.x);\n  vec4 cv = zCurves(Curve, vec4(vUv, c, 1.0), vec2(-1.0, 1.0));\n  gl_FragColor = cv;\n}",
        ),
        (
            "text",
            "precision highp float;\n// #include <zoidium/text>\nuniform sampler2D Atlas;\nvarying vec2 vUv;\nvoid main() {\n  float a = zText(Atlas, vUv) * 0.5 + zGlyph(Atlas, vec2(4.0, 4.0), 3.0, vUv) * 0.5;\n  gl_FragColor = vec4(vec3(a), 1.0);\n}",
        ),
        (
            "hash",
            "precision highp float;\n// #include <zoidium/hash>\nvarying vec2 vUv;\nvoid main() {\n  float h = zHash(vUv.x) + zHash(vUv) + zHash(vec3(vUv, 1.0));\n  gl_FragColor = vec4(vec3(h / 3.0), 1.0);\n}",
        ),
        (
            "noise",
            "precision highp float;\n// #include <zoidium/noise>\nvarying vec2 vUv;\nvoid main() {\n  float n = 0.5 * (zNoise(vUv * 4.0) + zNoise(vec3(vUv * 4.0, 0.5)));\n  gl_FragColor = vec4(vec3(n), 1.0);\n}",
        ),
        (
            "color",
            "precision highp float;\n// #include <zoidium/color>\nvarying vec2 vUv;\nvoid main() {\n  vec3 hsv = zRgbToHsv(vec3(vUv, 0.5));\n  vec3 rgb = zLinearToSrgb(zSrgbToLinear(zHsvToRgb(hsv)));\n  gl_FragColor = vec4(rgb, 1.0);\n}",
        ),
        (
            "sdf",
            "precision highp float;\n// #include <zoidium/sdf>\nvarying vec2 vUv;\nvoid main() {\n  vec2 p = vUv - 0.5;\n  float d = zSmoothUnion(zSdCircle(p, 0.25), zSdBox(p, vec2(0.2)), 0.05);\n  float d3 = zSmoothUnion(zSdSphere(vec3(p, 0.0), 0.2), zSdBox(vec3(p, 0.0), vec3(0.15)), 0.05);\n  gl_FragColor = vec4(vec3(step(0.0, d + d3)), 1.0);\n}",
        ),
    ];

    for (name, src) in single_library_cases {
        compile_and_validate_wgsl(&format!("include/{name}"), src, &opts);
    }

    // Combined shader including ALL 8 libraries (and duplicate includes to prove deduplication).
    let combined = r#"
        precision highp float;
        // #include <zoidium/temporal>
        // #include <zoidium/gradient>
        // #include <zoidium/curve>
        // #include <zoidium/text>
        // #include <zoidium/hash>
        // #include <zoidium/noise>
        // #include <zoidium/color>
        // #include <zoidium/sdf>
        // #include <zoidium/hash>
        // #include <zoidium/color>
        uniform sampler2D tDiffuse;
        uniform sampler2D PrevTex;
        uniform sampler2D RampTex;
        uniform sampler2D CurveTex;
        uniform sampler2D TextTex;
        uniform float zTime;
        varying vec2 vUv;
        varying vec2 vUvScaled;
        void main() {
            vec2 m = zTimeMotion(PrevTex, tDiffuse, vUvScaled, vec2(0.001));
            float n = zNoise(vec3(vUv * 4.0 + m, zTime)) + zHash(vUv);
            vec4 g = zGradient(RampTex, fract(n));
            vec3 c = zCurves(CurveTex, g.rgb);
            vec3 rgb = zHsvToRgb(zRgbToHsv(zLinearToSrgb(zSrgbToLinear(c))));
            float cov = zText(TextTex, vUv);
            float sdf = zSmoothUnion(zSdCircle(vUv - 0.5, 0.3), zSdBox(vUv - 0.5, vec2(0.2)), 0.05);
            gl_FragColor = vec4(mix(rgb, vec3(cov), step(0.0, -sdf)), 1.0);
        }
    "#;
    let compiled = compile_and_validate_wgsl("include/all-8-combined", combined, &opts);
    assert_eq!(compiled.texture_bindings.len(), 5);
}

// ===========================================================================
// 3. Node Effect `vectorMath-4-*` Const-Eval Regressions & Complex Node Shaders
// ===========================================================================

/// Exact Node Effect GLSL runtime header from Zoidium `plugins/node-effect/glsl.js`.
const NODE_EFFECT_HEADER: &str = r#"
precision highp float;
precision highp int;
uniform sampler2D tDiffuse;
uniform vec2 uvScale;
uniform vec2 resolution;
varying vec2 vUv;
vec4 ne_sample(vec2 uv) {
 vec4 p=texture2D(tDiffuse,uv*uvScale);
 return vec4(p.a>0.000001?p.rgb/p.a:vec3(0.0),p.a);
}
float ne_safe(float x) { return x<=1.0e30 && x>=-1.0e30 ? x : 0.0; }
float ne_div(float a,float b) { return abs(b)<1.0e-12 ? 0.0 : ne_safe(a/b); }
vec3 ne_rgb(vec3 c) {
 vec3 p=abs(fract(c.xxx+vec3(0.0,2.0/3.0,1.0/3.0))*6.0-3.0);
 return c.z*mix(vec3(1.0),clamp(p-1.0,0.0,1.0),c.y);
}
vec3 ne_hsv(vec3 c) {
 float hi=max(c.r,max(c.g,c.b)),lo=min(c.r,min(c.g,c.b)),d=hi-lo,h=0.0;
 if(d>0.0) { if(hi==c.r) h=mod((c.g-c.b)/d,6.0); else if(hi==c.g) h=(c.b-c.r)/d+2.0; else h=(c.r-c.g)/d+4.0; }
 return vec3(fract(h/6.0),hi==0.0?0.0:d/hi,hi);
}
float ne_math(float a,float b,float c,int mode) {
 float r=0.0;
 if(mode==0) r=a+b;
 else if(mode==1) r=a-b;
 else if(mode==2) r=a*b;
 else if(mode==3) r=ne_div(a,b);
 else if(mode==4) r=min(a,b);
 else if(mode==5) r=max(a,b);
 else if(mode==6) r=a<0.0?(fract(b)==0.0?pow(-a,b)*(mod(abs(b),2.0)==0.0?1.0:-1.0):0.0):pow(a,b);
 else if(mode==7) r=abs(b)<1.0e-12?0.0:a-b*(a/b<0.0?ceil(a/b):floor(a/b));
 else if(mode==8) r=abs(a);
 else if(mode==9) r=1.0-a;
 else if(mode==10) r=clamp(a,0.0,1.0);
 else if(mode==11) r=abs(a-b)<=0.01?1.0:0.0;
 else if(mode==12) r=sin(a);
 else if(mode==13) r=floor(a+0.5);
 else if(mode==14) r=sqrt(a);
 else if(mode==15) r=a>0.0 && b>0.0 && abs(b-1.0)>1.0e-12 ? log(a)/log(b) : 0.0;
 else if(mode==16) r=exp(a);
 else if(mode==17) r=floor(a);
 else if(mode==18) r=ceil(a);
 else if(mode==19) r=fract(a);
 else if(mode==20) r=sign(a);
 else if(mode==21 || mode==22) { float k=max(c,0.0),h=k==0.0?0.0:max(k-abs(a-b),0.0)/k; r=mode==21?min(a,b)-h*h*k*0.25:max(a,b)+h*h*k*0.25; }
 else if(mode==23) r=abs(b)<1.0e-12?0.0:floor(a/b)*b;
 else if(mode==24) r=b==c?c:a-(b-c)*floor((a-c)/(b-c));
 else if(mode==25) r=abs(b)<1.0e-12?0.0:abs(b)-abs(mod(a,2.0*abs(b))-abs(b));
 else if(mode==26) r=cos(a);
 else if(mode==27) r=tan(a);
 else if(mode==28) r=a==0.0 && b==0.0?0.0:atan(a,b);
 else if(mode==29) r=a>b?1.0:0.0;
 else if(mode==30) r=a<b?1.0:0.0;
 return ne_safe(r);
}
float ne_blend(float a,float b,int m) {
 if(m==1)return a+b; if(m==2)return a-b; if(m==3)return a*b;
 if(m==4)return 1.0-(1.0-a)*(1.0-b);
 if(m==5)return a<0.5?2.0*a*b:1.0-2.0*(1.0-a)*(1.0-b);
 if(m==6)return min(a,b); if(m==7)return max(a,b); if(m==8)return abs(a-b);
 if(m==9)return b<=0.5?a-(1.0-2.0*b)*a*(1.0-a):a+(2.0*b-1.0)*((a<=0.25?((16.0*a-12.0)*a+4.0)*a:sqrt(a))-a);
 if(m==10)return b>=1.0?1.0:min(1.0,a/(1.0-b));
 if(m==11)return b<=0.0?0.0:1.0-min(1.0,(1.0-a)/b);
 if(m==12)return a+2.0*b-1.0; if(m==13)return ne_div(a,b); if(m==14)return a+b-2.0*a*b;
 return b;
}
vec3 ne_mix(vec3 a,vec3 b,float f,int m) {
 vec3 r;
 if(m>=15) {vec3 x=ne_hsv(a),y=ne_hsv(b);r=ne_rgb(vec3(m==15||m==17?y.x:x.x,m==16||m==17?y.y:x.y,m==18?y.z:x.z));}
 else r=vec3(ne_blend(a.r,b.r,m),ne_blend(a.g,b.g,m),ne_blend(a.b,b.b,m));
 return mix(a,r,clamp(f,0.0,1.0));
}
float ne_hash(vec2 p) {return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float ne_noise(vec2 p) {
 vec2 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);
 return mix(mix(ne_hash(i),ne_hash(i+vec2(1,0)),f.x),mix(ne_hash(i+vec2(0,1)),ne_hash(i+vec2(1,1)),f.x),f.y);
}
vec3 ne_voronoi(vec2 p,float t) {
 vec2 cell=floor(p),f=fract(p),nearest=vec2(0.0),best=vec2(0.0);float f1=1.0e10,f2=f1;
 for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
  vec2 g=vec2(float(x),float(y)),c=cell+g;
  vec2 point=0.5+0.5*sin(t+6.2831853*vec2(ne_hash(c),ne_hash(c+19.1)));
  vec2 r=g+point-f;float d=dot(r,r);
  if(d<f1){f2=f1;f1=d;nearest=r;best=g;}else if(d<f2)f2=d;
 }
 float edge=1.0e10;
 for(int y=-2;y<=2;y++)for(int x=-2;x<=2;x++) {
  vec2 g=best+vec2(float(x),float(y)),c=cell+g;
  vec2 r=g+0.5+0.5*sin(t+6.2831853*vec2(ne_hash(c),ne_hash(c+19.1)))-f;
  vec2 d=r-nearest;if(dot(d,d)>0.000001)edge=min(edge,dot(0.5*(nearest+r),normalize(d)));
 }
 return vec3(sqrt(f1),sqrt(f2),edge);
}
"#;

#[test]
fn node_effect_vector_math_4_const_eval_and_complex_nodes_validate_in_naga() {
    let opts = CompileOptions::default();

    // Exact compiled bodies of `vectorMath-4-out` and `vectorMath-4-value` from Zoidium's
    // `node-effect/compiler.js` (where `normalize((vec4(0.0,0.0,0.0,1.0)).rgb)` sits inside a
    // constant ternary guard `length(vec3(0.0)) > 0.0 ? normalize(vec3(0.0)) : vec3(0.0)`).
    let bodies: [(&str, &str); 6] = [
        (
            "vectorMath-4-out",
            r#"
void main(){
 vec4 n0=vec4(length((vec4(0.0,0.0,0.0,1.0)).rgb)>0.0?normalize((vec4(0.0,0.0,0.0,1.0)).rgb):vec3(0.0),1.0);
 vec4 n1=vec4(vec3(n0.x),1.0);
 float ne_alpha=clamp((vec4(vec3(1.0),1.0)).x,0.0,1.0);
 gl_FragColor=vec4((n0).rgb*ne_alpha,ne_alpha);
}
"#,
        ),
        (
            "vectorMath-4-value",
            r#"
void main(){
 vec4 n0=vec4(length((vec4(0.0,0.0,0.0,1.0)).rgb)>0.0?normalize((vec4(0.0,0.0,0.0,1.0)).rgb):vec3(0.0),1.0);
 vec4 n1=vec4(vec3(n0.x),1.0);
 float ne_alpha=clamp((vec4(vec3(1.0),1.0)).x,0.0,1.0);
 gl_FragColor=vec4((n1).rgb*ne_alpha,ne_alpha);
}
"#,
        ),
        (
            "voronoi-2-out",
            r#"
void main(){
 vec4 n0=vec4(vec3(ne_voronoi((vec4(vUv,0.0,1.0)).xy*(vec4(vec3(5.0),1.0)).x,(vec4(vec3(0.0),1.0)).x)[2]),1.0);
 float ne_alpha=clamp((vec4(vec3(1.0),1.0)).x,0.0,1.0);
 gl_FragColor=vec4((n0).rgb*ne_alpha,ne_alpha);
}
"#,
        ),
        (
            "math-24-out",
            r#"
void main(){
 vec4 n0=vec4(vec3(ne_math((vec4(vec3(0.5),1.0)).x,(vec4(vec3(1.0),1.0)).x,(vec4(vec3(0.0),1.0)).x,24)),1.0);
 float ne_alpha=clamp((vec4(vec3(1.0),1.0)).x,0.0,1.0);
 gl_FragColor=vec4((n0).rgb*ne_alpha,ne_alpha);
}
"#,
        ),
        (
            "mix-18-color",
            r#"
void main(){
 vec4 n0=ne_sample(vUv);
 vec4 n1=vec4(ne_mix((n0).rgb,(vec4(vec3(0.5,0.25,0.75),1.0)).rgb,(vec4(vec3(0.5),1.0)).x,18),1.0);
 float ne_alpha=clamp((n0).a,0.0,1.0);
 gl_FragColor=vec4((n1).rgb*ne_alpha,ne_alpha);
}
"#,
        ),
        (
            "colorRamp-0-color",
            r#"
vec4 ne_ramp0(float t){
 t=clamp(t,0.0,1.0);
 if(t<0.0)return vec4(0.0,0.0,0.0,1.0);
 if(t<1.0){float u=clamp((t-0.0)/1.0,0.0,1.0);
 return mix(vec4(0.0,0.0,0.0,1.0),vec4(1.0,1.0,1.0,1.0),u);}
 return vec4(1.0,1.0,1.0,1.0);}
void main(){
 vec4 n0=ne_ramp0((vec4(vUv,0.0,1.0)).x);
 float ne_alpha=clamp((n0).a,0.0,1.0);
 gl_FragColor=vec4((n0).rgb*ne_alpha,ne_alpha);
}
"#,
        ),
    ];

    for (name, body) in bodies {
        let full = format!("{NODE_EFFECT_HEADER}\n{body}");
        compile_and_validate_wgsl(&format!("node-effect/{name}"), &full, &opts);
    }
}

// ===========================================================================
// 4. `GL_EXT_shader_texture_lod` + Dead-Code-After-Return (`mosaic.glsl` pattern)
// ===========================================================================

/// Verbatim `plugins/alipfx/shaders/mosaic.glsl` from Zoidium: exercises
/// `#ifdef GL_EXT_shader_texture_lod`, `#extension GL_EXT_shader_texture_lod : enable`,
/// `texture2DGradEXT`, nested `#if USE_LINEAR_LIGHT`, and dead code after unconditional `return;`.
const ALIPFX_MOSAIC_GLSL: &str = r#"
precision highp float;
precision highp int;

#ifdef GL_EXT_shader_texture_lod
#extension GL_EXT_shader_texture_lod : enable
#endif

uniform sampler2D tDiffuse;
uniform vec2 uvScale;
uniform vec2 resolution;
varying vec2 vUvScaled;

uniform float Horizontal_Blocks;
uniform float Vertical_Blocks;
uniform float Sharp_Colors;

#define AVG_GRID 4
#define USE_LINEAR_LIGHT 0

vec3 toLinear(vec3 c)  { return pow(c, vec3(2.2)); }
vec3 toGamma(vec3 c)   { return pow(c, vec3(1.0 / 2.2)); }

vec4 sampleBuffer(vec2 layerUV)
{
  return texture2D(tDiffuse, layerUV * uvScale);
}

vec2 snapToTexelCenter(vec2 bufferUV)
{
  vec2 px = bufferUV * resolution;
  px = floor(px) + 0.5;
  return px / resolution;
}

void main()
{
  vec2 uv = vUvScaled / uvScale;
  uv = clamp(uv, 0.0, 0.999999);

  vec2 blocks = vec2(max(1.0, Horizontal_Blocks), max(1.0, Vertical_Blocks));
  vec2 cellSize = 1.0 / blocks;
  vec2 cell = floor(uv * blocks);
  vec2 centerUV = (cell + 0.5) * cellSize;
  centerUV = clamp(centerUV, 0.0, 0.999999);

  if (Sharp_Colors >= 0.5)
  {
    vec2 centerBuf = snapToTexelCenter(centerUV * uvScale);
    gl_FragColor = texture2D(tDiffuse, centerBuf);
    return;
  }

  #ifdef GL_EXT_shader_texture_lod
    vec2 gradX = vec2(cellSize.x, 0.0) * uvScale;
    vec2 gradY = vec2(0.0, cellSize.y) * uvScale;

    vec2 centerBufUV = centerUV * uvScale;
    vec4 c = texture2DGradEXT(tDiffuse, centerBufUV, gradX, gradY);

    #if USE_LINEAR_LIGHT
      c.rgb = toGamma(toLinear(c.rgb));
    #endif

    gl_FragColor = c;
    return;
  #endif

  vec2 baseUV = cell * cellSize;
  vec3 accRGB = vec3(0.0);
  float accA = 0.0;

  for (int y = 0; y < AVG_GRID; y++)
  {
    for (int x = 0; x < AVG_GRID; x++)
    {
      vec2 f = (vec2(float(x), float(y)) + 0.5) / float(AVG_GRID);
      vec2 suv = baseUV + f * cellSize;
      suv = clamp(suv, 0.0, 0.999999);

      vec2 bufUV = snapToTexelCenter(suv * uvScale);
      vec4 s = texture2D(tDiffuse, bufUV);

      #if USE_LINEAR_LIGHT
        accRGB += toLinear(s.rgb);
      #else
        accRGB += s.rgb;
      #endif

      accA += s.a;
    }
  }

  float invN = 1.0 / float(AVG_GRID * AVG_GRID);

  #if USE_LINEAR_LIGHT
    vec3 outRGB = toGamma(accRGB * invN);
  #else
    vec3 outRGB = accRGB * invN;
  #endif

  gl_FragColor = vec4(outRGB, accA * invN);
}
"#;

#[test]
fn mosaic_gl_ext_shader_texture_lod_and_dead_code_after_return_validate_in_naga() {
    // 1. Default options (GL_EXT_shader_texture_lod defined by translator or fallback path)
    let compiled_default = compile_and_validate_wgsl(
        "alipfx/mosaic-default",
        ALIPFX_MOSAIC_GLSL,
        &CompileOptions::default(),
    );
    assert_eq!(compiled_default.texture_bindings.len(), 1);

    // 2. Explicit GL_EXT_shader_texture_lod = 1 (forces `texture2DGradEXT` + dead code after `return;`)
    let mut lod_opts = CompileOptions::default();
    lod_opts
        .defines
        .insert("GL_EXT_shader_texture_lod".to_string(), "1".to_string());
    let compiled_lod =
        compile_and_validate_wgsl("alipfx/mosaic-ext-lod", ALIPFX_MOSAIC_GLSL, &lod_opts);
    assert!(
        compiled_lod.wgsl.contains("textureSampleGrad"),
        "expected textureSampleGrad in WGSL for texture2DGradEXT"
    );
}

// ===========================================================================
// 5. `mat2` / `struct` / `bvec*` / `samplerCube` Edge Cases
// ===========================================================================

#[test]
fn mat2_struct_bvec_and_sampler_cube_shaders_validate_in_naga() {
    let opts = CompileOptions::default();

    let mat2_struct_bvec = r#"
        precision highp float;
        varying vec2 vUvScaled;
        uniform sampler2D tDiffuse;
        uniform mat2 Transform2D;
        uniform vec2 Threshold;
        uniform float Gain;

        struct SampleInfo {
            vec2 uv;
            vec4 color;
            bvec2 flags;
        };

        SampleInfo evalSample(vec2 uv) {
            SampleInfo info;
            mat2 rot = mat2(0.0, -1.0, 1.0, 0.0);
            vec2 warped = clamp((rot * (Transform2D * (uv - 0.5))) + 0.5, vec2(0.0), vec2(1.0));
            info.uv = warped;
            info.color = texture2D(tDiffuse, warped);
            info.flags = greaterThan(warped, Threshold);
            return info;
        }

        void main() {
            SampleInfo s = evalSample(vUvScaled);
            bvec3 mask = lessThanEqual(s.color.rgb, vec3(0.5));
            bvec4 allCheck = equal(s.color, vec4(0.0));
            float boost = (any(s.flags) && !all(allCheck)) ? Gain : 1.0;
            vec3 outRgb = mix(s.color.rgb, s.color.rgb * boost, vec3(not(mask)));
            gl_FragColor = vec4(outRgb, s.color.a);
        }
    "#;
    let compiled_msb = compile_and_validate_wgsl("edge/mat2-struct-bvec", mat2_struct_bvec, &opts);
    assert!(compiled_msb.uniform_layout.field("Transform2D").is_some());

    let sampler_cube = r#"
        precision highp float;
        varying vec2 vUv;
        uniform samplerCube EnvMap;
        uniform float Exposure;

        void main() {
            vec2 ndc = vUv * 2.0 - 1.0;
            vec3 dir = normalize(vec3(ndc, 1.0));
            vec4 env = textureCube(EnvMap, dir);
            gl_FragColor = vec4(env.rgb * Exposure, env.a);
        }
    "#;
    let compiled_cube = compile_and_validate_wgsl("edge/sampler-cube", sampler_cube, &opts);
    assert_eq!(compiled_cube.texture_bindings.len(), 1);
    assert_eq!(compiled_cube.texture_bindings[0].name, "EnvMap");
    assert!(
        compiled_cube.wgsl.contains("texture_cube<f32>"),
        "expected texture_cube<f32> in WGSL"
    );

    // Quick sanity check on `translate_glsl_es100` convenience wrapper as well.
    let simple = translate_glsl_es100(SHADER_PLUS_TEMPLATES[0].1).expect("passthrough translates");
    let module = naga::front::wgsl::parse_str(&simple.wgsl).expect("wgsl parses");
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .expect("wgsl validates");
}

// ===========================================================================
// 6. Optional Local External Zoidium Corpus Sweep (Skipped Gracefully on CI)
// ===========================================================================

fn collect_glsl_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|d| d.path())).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            collect_glsl_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("glsl") {
            out.push(path);
        }
    }
}

#[test]
fn optional_external_zoidium_plugin_glsl_corpus_validates_in_naga() {
    let zoidium_plugins = Path::new("/Users/dia/Projects/GitHub/Zoidium/plugins");
    if !zoidium_plugins.is_dir() {
        // Gracefully skip on CI where the sibling Zoidium repository is not checked out.
        return;
    }

    // Standalone fragment shader directories across Zoidium's plugin packs.
    let subdirs = [
        "alipfx/shaders",
        "afterclip/shaders",
        "ccfx/shaders",
        "native-fx/shaders",
        "openzoid-legacy/shaders",
        "sglow-hq-v2-preview/shaders",
        "node-effect",
    ];

    let mut files = Vec::new();
    for sub in subdirs {
        collect_glsl_files(&zoidium_plugins.join(sub), &mut files);
    }

    let opts = CompileOptions::default();
    let mut tested = 0_usize;
    let mut failures = Vec::new();

    for path in files {
        let Ok(src) = std::fs::read_to_string(&path) else {
            continue;
        };
        // Skip helper/include chunks without `main` and vertex shaders (`gl_Position`).
        if !src.contains("main") || !src.contains("gl_FragColor") || src.contains("gl_Position") {
            continue;
        }
        tested += 1;
        let rel = path
            .strip_prefix(zoidium_plugins)
            .unwrap_or(&path)
            .display()
            .to_string();
        match translate_glsl_es100_with_options(&src, &opts) {
            Ok(compiled) => match naga::front::wgsl::parse_str(&compiled.wgsl) {
                Ok(module) => {
                    let mut validator =
                        Validator::new(ValidationFlags::all(), Capabilities::empty());
                    if let Err(err) = validator.validate(&module) {
                        failures.push(format!("{rel}: naga validate error: {err}"));
                    }
                }
                Err(err) => {
                    failures.push(format!("{rel}: naga parse error: {err}"));
                }
            },
            Err(err) => {
                failures.push(format!("{rel}: translate error: {err}"));
            }
        }
    }

    assert!(
        tested > 200,
        "expected > 200 standalone Zoidium fragment shaders when repo is present, tested {tested}"
    );
    assert!(
        failures.is_empty(),
        "external Zoidium corpus had {}/{tested} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
