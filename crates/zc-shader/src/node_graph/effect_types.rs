use super::*;

/// Exposed uniform parameter declared by a `parameter` node in a Node Effect graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeEffectParameter {
    pub id: String,
    pub uniform_name: String,
    pub label: String,
    /// `0` = float, `1` = color (`vec3`), `2` = point (`vec2`).
    pub kind: u8,
    pub default_value: Vec<f32>,
    pub min: f32,
    pub max: f32,
}

/// Output of [`compile_node_effect_to_glsl`].
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledNodeEffect {
    /// Complete GLSL ES 1.00 fragment shader ready for [`crate::translate_glsl_es100`].
    pub glsl_source: String,
    /// Exposed `parameter` node uniforms.
    pub parameters: Vec<NodeEffectParameter>,
    /// True when the graph reads `ne_time`.
    pub uses_time: bool,
    /// Estimated per-pixel `tDiffuse` sample count (capped at 256).
    pub sample_count: usize,
    pub node_count: usize,
    pub link_count: usize,
}

impl CompiledNodeEffect {
    /// Translates the compiled GLSL ES 1.00 shader directly to WGSL via `zc-shader`.
    pub fn to_wgsl(&self) -> Result<crate::CompiledCustomShader, crate::ShaderError> {
        crate::translate_glsl_es100(&self.glsl_source)
    }
}

pub(super) const NODE_EFFECT_PASSTHROUGH_GLSL: &str = "\
precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 uvScale;
uniform vec2 resolution;
varying vec2 vUv;
void main() {
    gl_FragColor = texture2D(tDiffuse, vUv);
}
";

pub(super) const NODE_EFFECT_HEADER_GLSL: &str = r#"precision highp float;
precision highp int;
uniform sampler2D tDiffuse;
uniform vec2 uvScale;
uniform vec2 resolution;
varying vec2 vUv;
vec4 ne_sample(vec2 uv) {
 vec4 p = texture2D(tDiffuse, uv * uvScale);
 return vec4(p.a > 0.000001 ? p.rgb / p.a : vec3(0.0), p.a);
}
float ne_safe(float x) { return x <= 1.0e30 && x >= -1.0e30 ? x : 0.0; }
float ne_div(float a, float b) { return abs(b) < 1.0e-12 ? 0.0 : ne_safe(a / b); }
vec3 ne_rgb(vec3 c) {
 vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
 return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}
vec3 ne_hsv(vec3 c) {
 float hi = max(c.r, max(c.g, c.b)), lo = min(c.r, min(c.g, c.b)), d = hi - lo, h = 0.0;
 if (d > 0.0) {
  if (hi == c.r) h = mod((c.g - c.b) / d, 6.0);
  else if (hi == c.g) h = (c.b - c.r) / d + 2.0;
  else h = (c.r - c.g) / d + 4.0;
 }
 return vec3(fract(h / 6.0), hi == 0.0 ? 0.0 : d / hi, hi);
}
float ne_math(float a, float b, float c, int mode) {
 float r = 0.0;
 if (mode == 0) r = a + b;
 else if (mode == 1) r = a - b;
 else if (mode == 2) r = a * b;
 else if (mode == 3) r = ne_div(a, b);
 else if (mode == 4) r = min(a, b);
 else if (mode == 5) r = max(a, b);
 else if (mode == 6) r = a < 0.0 ? (fract(b) == 0.0 ? pow(-a, b) * (mod(abs(b), 2.0) == 0.0 ? 1.0 : -1.0) : 0.0) : pow(a, b);
 else if (mode == 7) r = abs(b) < 1.0e-12 ? 0.0 : a - b * (a / b < 0.0 ? ceil(a / b) : floor(a / b));
 else if (mode == 8) r = abs(a);
 else if (mode == 9) r = 1.0 - a;
 else if (mode == 10) r = clamp(a, 0.0, 1.0);
 else if (mode == 11) r = abs(a - b) <= 0.01 ? 1.0 : 0.0;
 else if (mode == 12) r = sin(a);
 else if (mode == 13) r = floor(a + 0.5);
 else if (mode == 14) r = sqrt(max(a, 0.0));
 else if (mode == 15) r = a > 0.0 && b > 0.0 && abs(b - 1.0) > 1.0e-12 ? log(a) / log(b) : 0.0;
 else if (mode == 16) r = exp(clamp(a, -60.0, 60.0));
 else if (mode == 17) r = floor(a);
 else if (mode == 18) r = ceil(a);
 else if (mode == 19) r = fract(a);
 else if (mode == 20) r = sign(a);
 else if (mode == 21 || mode == 22) {
  float k = max(c, 0.0);
  float h = k == 0.0 ? 0.0 : max(k - abs(a - b), 0.0) / k;
  r = mode == 21 ? min(a, b) - h * h * k * 0.25 : max(a, b) + h * h * k * 0.25;
 }
 else if (mode == 23) r = abs(b) < 1.0e-12 ? 0.0 : floor(a / b) * b;
 else if (mode == 24) r = b == c ? c : a - (b - c) * floor((a - c) / (b - c));
 else if (mode == 25) r = abs(b) < 1.0e-12 ? 0.0 : abs(b) - abs(mod(a, 2.0 * abs(b)) - abs(b));
 else if (mode == 26) r = cos(a);
 else if (mode == 27) r = tan(a);
 else if (mode == 28) r = a == 0.0 && b == 0.0 ? 0.0 : atan(a, b);
 else if (mode == 29) r = a > b ? 1.0 : 0.0;
 else if (mode == 30) r = a < b ? 1.0 : 0.0;
 return ne_safe(r);
}
float ne_blend(float a, float b, int m) {
 if (m == 1) return a + b;
 if (m == 2) return a - b;
 if (m == 3) return a * b;
 if (m == 4) return 1.0 - (1.0 - a) * (1.0 - b);
 if (m == 5) return a < 0.5 ? 2.0 * a * b : 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
 if (m == 6) return min(a, b);
 if (m == 7) return max(a, b);
 if (m == 8) return abs(a - b);
 if (m == 9) return b <= 0.5 ? a - (1.0 - 2.0 * b) * a * (1.0 - a) : a + (2.0 * b - 1.0) * ((a <= 0.25 ? ((16.0 * a - 12.0) * a + 4.0) * a : sqrt(max(a, 0.0))) - a);
 if (m == 10) return b >= 1.0 ? 1.0 : min(1.0, a / max(1.0 - b, 1.0e-6));
 if (m == 11) return b <= 0.0 ? 0.0 : 1.0 - min(1.0, (1.0 - a) / max(b, 1.0e-6));
 if (m == 12) return a + 2.0 * b - 1.0;
 if (m == 13) return ne_div(a, b);
 if (m == 14) return a + b - 2.0 * a * b;
 return b;
}
vec3 ne_mix(vec3 a, vec3 b, float f, int m) {
 vec3 r;
 if (m >= 15) {
  vec3 x = ne_hsv(a), y = ne_hsv(b);
  r = ne_rgb(vec3(m == 15 || m == 17 ? y.x : x.x, m == 16 || m == 17 ? y.y : x.y, m == 18 ? y.z : x.z));
 } else {
  r = vec3(ne_blend(a.r, b.r, m), ne_blend(a.g, b.g, m), ne_blend(a.b, b.b, m));
 }
 return mix(a, r, clamp(f, 0.0, 1.0));
}
float ne_hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float ne_noise(vec2 p) {
 vec2 i = floor(p), f = fract(p);
 f = f * f * (3.0 - 2.0 * f);
 return mix(mix(ne_hash(i), ne_hash(i + vec2(1.0, 0.0)), f.x), mix(ne_hash(i + vec2(0.0, 1.0)), ne_hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
float ne_fbm(vec2 p, float lacunarity, float gain) {
 float sum = 0.0, amp = 0.5, norm = 0.0;
 vec2 q = p;
 for (int i = 0; i < 4; i++) {
  sum += ne_noise(q) * amp;
  norm += amp;
  q *= lacunarity;
  amp *= gain;
 }
 return norm > 0.0 ? sum / norm : 0.0;
}
vec3 ne_voronoi(vec2 p, float t) {
 vec2 cell = floor(p), f = fract(p), nearest = vec2(0.0), best = vec2(0.0);
 float f1 = 1.0e10, f2 = f1;
 for (int y = -1; y <= 1; y++) {
  for (int x = -1; x <= 1; x++) {
   vec2 g = vec2(float(x), float(y)), c = cell + g;
   vec2 point = 0.5 + 0.5 * sin(t + 6.2831853 * vec2(ne_hash(c), ne_hash(c + 19.1)));
   vec2 r = g + point - f;
   float d = dot(r, r);
   if (d < f1) { f2 = f1; f1 = d; nearest = r; best = g; }
   else if (d < f2) { f2 = d; }
  }
 }
 float edge = 1.0e10;
 for (int y = -2; y <= 2; y++) {
  for (int x = -2; x <= 2; x++) {
   vec2 g = best + vec2(float(x), float(y)), c = cell + g;
   vec2 r = g + 0.5 + 0.5 * sin(t + 6.2831853 * vec2(ne_hash(c), ne_hash(c + 19.1))) - f;
   vec2 d = r - nearest;
   if (dot(d, d) > 0.000001) {
    edge = min(edge, dot(0.5 * (nearest + r), normalize(d)));
   }
  }
 }
 return vec3(sqrt(f1), sqrt(f2), edge);
}
"#;

pub(super) fn glsl_float(n: f64) -> String {
    let clamped = if n.is_finite() {
        n.clamp(-1.0e8, 1.0e8)
    } else {
        0.0
    };
    let s = format!("{clamped:.6}");
    let trimmed = s.trim_end_matches('0');
    if trimmed.ends_with('.') {
        format!("{trimmed}0")
    } else if trimmed == "-0.0" {
        "0.0".to_string()
    } else {
        trimmed.to_string()
    }
}

pub(super) fn glsl_scalar(expr: &str) -> String {
    format!("vec4(vec3({expr}), 1.0)")
}

pub(super) fn glsl_rgb(expr: &str) -> String {
    format!("vec4({expr}, 1.0)")
}

pub(super) fn glsl_literal_value(val: Option<&Value>, fallback_num: f64) -> String {
    if let Some(v) = val {
        if let Some(n) = v.as_f64()
            && n.is_finite()
        {
            return glsl_scalar(&glsl_float(n));
        }
        if let Some(b) = v.as_bool() {
            return glsl_scalar(if b { "1.0" } else { "0.0" });
        }
        if let Some(s) = v.as_str()
            && let Some([r, g, b]) = parse_hex_color(s)
        {
            return glsl_rgb(&format!(
                "vec3({}, {}, {})",
                glsl_float(r),
                glsl_float(g),
                glsl_float(b)
            ));
        }
        if let Some(arr) = v.as_array() {
            if arr.len() >= 3 {
                let r = arr[0].as_f64().unwrap_or(0.0);
                let g = arr[1].as_f64().unwrap_or(0.0);
                let b = arr[2].as_f64().unwrap_or(0.0);
                return glsl_rgb(&format!(
                    "vec3({}, {}, {})",
                    glsl_float(r),
                    glsl_float(g),
                    glsl_float(b)
                ));
            }
            if arr.len() == 2 {
                let x = arr[0].as_f64().unwrap_or(0.0);
                let y = arr[1].as_f64().unwrap_or(0.0);
                return format!("vec4({}, {}, 0.0, 1.0)", glsl_float(x), glsl_float(y));
            }
        }
    }
    glsl_scalar(&glsl_float(fallback_num))
}

pub(super) fn encode_param_uniform(node_id: &str) -> String {
    let mut out = String::from("ne_");
    for byte in node_id.bytes() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

pub(super) fn normalize_effect_node_type(raw: &str) -> String {
    let lower = raw.trim().to_ascii_lowercase();
    match lower.as_str() {
        "layer" | "source" | "image" | "input" => "layer".to_string(),
        "uv" | "texcoord" => "uv".to_string(),
        "pixel" | "pixelcoord" | "position" => "pixel".to_string(),
        "resolution" => "resolution".to_string(),
        "time" => "time".to_string(),
        "parameter" | "uniform" => "parameter".to_string(),
        "brightness" | "brightnesscontrast" | "brightness_contrast" => "brightness".to_string(),
        "hue" | "huesaturation" | "hue_saturation" | "hsv" => "hue".to_string(),
        "colorbalance" | "color_balance" => "colorBalance".to_string(),
        "invert" => "invert".to_string(),
        "grayscale" | "greyscale" | "desaturate" => "grayscale".to_string(),
        "levels" => "levels".to_string(),
        "colorramp" | "color_ramp" | "ramp" => "colorRamp".to_string(),
        "mix" | "blend" => "mix".to_string(),
        "separatergb" | "separate_rgb" | "channelsplit" | "channel_split" => {
            "separateRGB".to_string()
        }
        "separatehsv" | "separate_hsv" => "separateHSV".to_string(),
        "combinergb" | "combine_rgb" | "combine" | "channelcombine" => "combineRGB".to_string(),
        "combinehsv" | "combine_hsv" => "combineHSV".to_string(),
        "math" => "math".to_string(),
        "vectormath" | "vector_math" => "vectorMath".to_string(),
        "maprange" | "map_range" | "remap" => "mapRange".to_string(),
        "clamp" => "clamp".to_string(),
        "smoothstep" => "smoothstep".to_string(),
        "trig" => "trig".to_string(),
        "wave" => "wave".to_string(),
        "noise" => "noise".to_string(),
        "fbm" => "fbm".to_string(),
        "voronoi" | "worley" => "voronoi".to_string(),
        "checker" | "checkerboard" => "checker".to_string(),
        "gradient" => "gradient".to_string(),
        "solid" | "color" => "solid".to_string(),
        "vignette" => "vignette".to_string(),
        "grain" => "grain".to_string(),
        "uvtransform" | "uv_transform" => "uvTransform".to_string(),
        "displace" | "displacement" => "displace".to_string(),
        "swirl" | "twirl" => "swirl".to_string(),
        "kaleidoscope" => "kaleidoscope".to_string(),
        "pixelate" | "mosaic" => "pixelate".to_string(),
        "mirror" => "mirror".to_string(),
        "sample" => "sample".to_string(),
        "blur" | "boxblur" | "box_blur" => "blur".to_string(),
        "sharpen" => "sharpen".to_string(),
        "edge" | "edgedetect" | "edge_detect" | "sobel" => "edge".to_string(),
        "chromaticaberration" | "chromatic_aberration" => "chromaticAberration".to_string(),
        "glow" | "bloom" => "glow".to_string(),
        "output" | "final" => "output".to_string(),
        other => other.to_string(),
    }
}

pub(super) fn is_color_output(
    node_type: &str,
    port: &str,
    params: &BTreeMap<String, Value>,
) -> bool {
    let p = port.to_ascii_lowercase();
    if p == "color" || p == "rgba" || p == "rgb" {
        return true;
    }
    if node_type == "parameter" {
        let kind = param_f64(params, &["kind"], 0.0).round() as i32;
        return kind == 1;
    }
    matches!(
        node_type,
        "layer"
            | "mix"
            | "colorRamp"
            | "hue"
            | "brightness"
            | "colorBalance"
            | "levels"
            | "invert"
            | "grayscale"
            | "combineRGB"
            | "combineHSV"
            | "solid"
            | "vignette"
            | "grain"
            | "sample"
            | "blur"
            | "sharpen"
            | "chromaticAberration"
            | "glow"
    ) && (p == "out" || p == "color")
}
