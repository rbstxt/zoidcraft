//! Node Graph Compilers for Zoidium `node-effect` (GLSL ES 1.00 fragment shader codegen)
//! and `material-plus` `NodeMaterial` (deterministic CPU RGBA8 PBR texture baker).
//!
//! - [`compile_node_effect_to_glsl`]: compiles Zoidium `node-effect` graphs into Naga-valid
//!   GLSL ES 1.00 fragment shaders feeding directly into [`crate::translate_glsl_es100`].
//! - [`bake_node_material_maps`]: evaluates Zoidium `material-plus` procedural graphs into
//!   deterministic CPU RGBA8 texture maps (`color`, `emissive`, `roughness`, `metalness`,
//!   `normal`, `bump`, `opacity`) plus scalar PBR fallbacks.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::f64::consts::{PI, SQRT_2, TAU};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::shader_plus::{ZoidiumCurvePoint, evaluate_zoidium_curve_points};

// ===========================================================================
// Shared Graph Parsing, Topological Sort & Color/Value Helpers
// ===========================================================================

const MAX_EFFECT_NODES: usize = 128;
const MAX_EFFECT_LINKS: usize = 256;
const MAX_EFFECT_SAMPLES: usize = 256;
const MAX_MATERIAL_NODES: usize = 200;
const MAX_MATERIAL_LINKS: usize = 400;
const MAX_BAKE_MEMORY_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone)]
struct RawNode {
    id: String,
    kind: String,
    params: BTreeMap<String, Value>,
}

#[derive(Debug, Clone)]
struct RawLink {
    from_node: String,
    from_port: String,
    to_node: String,
    to_port: String,
}

fn parse_raw_graph(
    graph: &Value,
    max_nodes: usize,
    max_links: usize,
) -> Result<(Vec<RawNode>, Vec<RawLink>), String> {
    if graph.is_null() {
        return Ok((Vec::new(), Vec::new()));
    }
    let Some(obj) = graph.as_object() else {
        return Err("node graph must be a JSON object".to_string());
    };

    let raw_nodes = obj
        .get("nodes")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if raw_nodes.len() > max_nodes {
        return Err(format!(
            "node graph exceeds maximum of {max_nodes} nodes (got {})",
            raw_nodes.len()
        ));
    }

    let raw_links = obj
        .get("links")
        .or_else(|| obj.get("edges"))
        .or_else(|| obj.get("connections"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if raw_links.len() > max_links {
        return Err(format!(
            "node graph exceeds maximum of {max_links} links (got {})",
            raw_links.len()
        ));
    }

    let mut seen_ids = BTreeSet::new();
    let mut nodes = Vec::with_capacity(raw_nodes.len());
    for (idx, item) in raw_nodes.iter().enumerate() {
        let Some(nobj) = item.as_object() else {
            return Err(format!("node at index {idx} is not an object"));
        };
        let id = nobj
            .get("id")
            .and_then(|v| {
                v.as_str()
                    .map(str::to_string)
                    .or_else(|| v.as_i64().map(|n| n.to_string()))
            })
            .filter(|s| !s.is_empty() && s.len() <= 128)
            .ok_or_else(|| format!("node at index {idx} has missing or invalid id"))?;
        if !seen_ids.insert(id.clone()) {
            return Err(format!("duplicate node id `{id}`"));
        }
        let kind = nobj
            .get("type")
            .or_else(|| nobj.get("kind"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("node `{id}` is missing a valid `type`"))?
            .to_string();

        let mut params = BTreeMap::new();
        // Merge lower-precedence `values` / `inputs` maps first, then `params` on top.
        for key in ["values", "inputs", "params"] {
            if let Some(Value::Object(map)) = nobj.get(key) {
                for (k, v) in map {
                    params.insert(k.clone(), v.clone());
                }
            }
        }
        nodes.push(RawNode { id, kind, params });
    }

    let mut links = Vec::with_capacity(raw_links.len());
    for (idx, item) in raw_links.iter().enumerate() {
        let Some(lobj) = item.as_object() else {
            return Err(format!("link at index {idx} is not an object"));
        };
        let (from_node, from_port) = extract_endpoint(
            lobj,
            "from",
            "fromNode",
            "source",
            "fromPort",
            "sourcePort",
            "fromSocket",
            "out",
        )
        .ok_or_else(|| format!("link at index {idx} has invalid source endpoint"))?;
        let (to_node, to_port) = extract_endpoint(
            lobj,
            "to",
            "toNode",
            "target",
            "toPort",
            "targetPort",
            "toSocket",
            "in",
        )
        .ok_or_else(|| format!("link at index {idx} has invalid target endpoint"))?;

        if !seen_ids.contains(&from_node) || !seen_ids.contains(&to_node) {
            return Err(format!(
                "link at index {idx} references unknown node (`{from_node}` -> `{to_node}`)"
            ));
        }
        links.push(RawLink {
            from_node,
            from_port,
            to_node,
            to_port,
        });
    }

    Ok((nodes, links))
}

#[allow(clippy::too_many_arguments)]
fn extract_endpoint(
    obj: &serde_json::Map<String, Value>,
    nested_key: &str,
    node_key_a: &str,
    node_key_b: &str,
    port_key_a: &str,
    port_key_b: &str,
    port_key_c: &str,
    default_port: &str,
) -> Option<(String, String)> {
    if let Some(Value::Object(ep)) = obj.get(nested_key) {
        let node = ep.get("node").or_else(|| ep.get("id")).and_then(|v| {
            v.as_str()
                .map(str::to_string)
                .or_else(|| v.as_i64().map(|n| n.to_string()))
        })?;
        let port = ep
            .get("port")
            .or_else(|| ep.get("socket"))
            .and_then(Value::as_str)
            .unwrap_or(default_port)
            .to_string();
        return Some((node, port));
    }
    let node = obj
        .get(nested_key)
        .or_else(|| obj.get(node_key_a))
        .or_else(|| obj.get(node_key_b))
        .and_then(|v| {
            v.as_str()
                .map(str::to_string)
                .or_else(|| v.as_i64().map(|n| n.to_string()))
        })?;
    let port = obj
        .get(port_key_a)
        .or_else(|| obj.get(port_key_b))
        .or_else(|| obj.get(port_key_c))
        .and_then(Value::as_str)
        .unwrap_or(default_port)
        .to_string();
    Some((node, port))
}

/// Topological sort (Kahn's algorithm) returning upstream-first node IDs, or `Err` on cycle.
fn topological_sort(nodes: &[RawNode], links: &[RawLink]) -> Result<Vec<String>, String> {
    let mut in_degree: BTreeMap<&str, usize> = BTreeMap::new();
    let mut dependents: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for node in nodes {
        in_degree.insert(node.id.as_str(), 0);
        dependents.insert(node.id.as_str(), Vec::new());
    }
    for link in links {
        if link.from_node == link.to_node {
            return Err(format!("self-cycle detected on node `{}`", link.from_node));
        }
        if let Some(deg) = in_degree.get_mut(link.to_node.as_str()) {
            *deg += 1;
        }
        if let Some(list) = dependents.get_mut(link.from_node.as_str()) {
            list.push(link.to_node.as_str());
        }
    }

    let mut queue: VecDeque<&str> = nodes
        .iter()
        .filter(|n| in_degree.get(n.id.as_str()).copied().unwrap_or(0) == 0)
        .map(|n| n.id.as_str())
        .collect();
    let mut order = Vec::with_capacity(nodes.len());

    while let Some(id) = queue.pop_front() {
        order.push(id.to_string());
        if let Some(nexts) = dependents.get(id) {
            for &next in nexts {
                if let Some(deg) = in_degree.get_mut(next) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(next);
                    }
                }
            }
        }
    }

    if order.len() != nodes.len() {
        return Err("node graph connections cannot form a cycle".to_string());
    }
    Ok(order)
}

fn param_f64(params: &BTreeMap<String, Value>, keys: &[&str], default: f64) -> f64 {
    for &key in keys {
        if let Some(v) = params.get(key) {
            if let Some(n) = v.as_f64()
                && n.is_finite()
            {
                return n;
            }
            if let Some(b) = v.as_bool() {
                return if b { 1.0 } else { 0.0 };
            }
            if let Some(s) = v.as_str()
                && let Ok(n) = s.trim().parse::<f64>()
                && n.is_finite()
            {
                return n;
            }
        }
    }
    default
}

fn param_str<'a>(params: &'a BTreeMap<String, Value>, keys: &[&str]) -> Option<&'a str> {
    for &key in keys {
        if let Some(s) = params.get(key).and_then(Value::as_str) {
            return Some(s);
        }
    }
    None
}

fn parse_hex_color(hex: &str) -> Option<[f64; 3]> {
    let s = hex.trim().strip_prefix('#')?;
    if s.len() != 6 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let r = u8::from_str_radix(&s[0..2], 16).ok()?;
    let g = u8::from_str_radix(&s[2..4], 16).ok()?;
    let b = u8::from_str_radix(&s[4..6], 16).ok()?;
    Some([
        f64::from(r) / 255.0,
        f64::from(g) / 255.0,
        f64::from(b) / 255.0,
    ])
}

fn param_color3(params: &BTreeMap<String, Value>, keys: &[&str], default: [f64; 3]) -> [f64; 3] {
    for &key in keys {
        if let Some(v) = params.get(key) {
            if let Some(s) = v.as_str()
                && let Some(rgb) = parse_hex_color(s)
            {
                return rgb;
            }
            if let Some(arr) = v.as_array()
                && arr.len() >= 3
            {
                let r = arr[0].as_f64().unwrap_or(default[0]);
                let g = arr[1].as_f64().unwrap_or(default[1]);
                let b = arr[2].as_f64().unwrap_or(default[2]);
                if r.is_finite() && g.is_finite() && b.is_finite() {
                    let scale = if r > 1.0 || g > 1.0 || b > 1.0 {
                        1.0 / 255.0
                    } else {
                        1.0
                    };
                    return [
                        (r * scale).clamp(0.0, 1.0),
                        (g * scale).clamp(0.0, 1.0),
                        (b * scale).clamp(0.0, 1.0),
                    ];
                }
            }
        }
    }
    default
}

fn parse_ramp_stops(params: &BTreeMap<String, Value>) -> Vec<[f64; 5]> {
    let mut stops = Vec::new();
    if let Some(Value::Array(arr)) = params.get("stops") {
        for entry in arr.iter().take(32) {
            if let Some(s) = entry.as_array()
                && s.len() >= 5
            {
                let pos = s[0].as_f64().unwrap_or(0.0).clamp(0.0, 1.0);
                let r = s[1].as_f64().unwrap_or(0.0).clamp(0.0, 1.0);
                let g = s[2].as_f64().unwrap_or(0.0).clamp(0.0, 1.0);
                let b = s[3].as_f64().unwrap_or(0.0).clamp(0.0, 1.0);
                let a = s[4].as_f64().unwrap_or(1.0).clamp(0.0, 1.0);
                stops.push([pos, r, g, b, a]);
            } else if let Some(obj) = entry.as_object() {
                let pos = obj
                    .get("position")
                    .or_else(|| obj.get("pos"))
                    .or_else(|| obj.get("t"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0);
                let mut col = [0.0, 0.0, 0.0];
                let mut alpha = 1.0;
                if let Some(cv) = obj.get("color") {
                    if let Some(s) = cv.as_str()
                        && let Some(rgb) = parse_hex_color(s)
                    {
                        col = rgb;
                    } else if let Some(ca) = cv.as_array()
                        && ca.len() >= 3
                    {
                        col = [
                            ca[0].as_f64().unwrap_or(0.0).clamp(0.0, 1.0),
                            ca[1].as_f64().unwrap_or(0.0).clamp(0.0, 1.0),
                            ca[2].as_f64().unwrap_or(0.0).clamp(0.0, 1.0),
                        ];
                        if let Some(a) = ca.get(3).and_then(Value::as_f64) {
                            alpha = a.clamp(0.0, 1.0);
                        }
                    }
                }
                if let Some(a) = obj.get("alpha").and_then(Value::as_f64) {
                    alpha = a.clamp(0.0, 1.0);
                }
                stops.push([pos, col[0], col[1], col[2], alpha]);
            }
        }
    }
    if stops.len() < 2 {
        return vec![[0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]];
    }
    stops.sort_by(|a, b| a[0].total_cmp(&b[0]));
    stops
}

fn resolve_blend_mode(params: &BTreeMap<String, Value>, keys: &[&str]) -> i32 {
    if let Some(name) = param_str(params, keys) {
        return match name.trim().to_ascii_lowercase().as_str() {
            "normal" | "mix" | "lerp" => 0,
            "add" => 1,
            "subtract" | "sub" => 2,
            "multiply" | "mul" => 3,
            "screen" => 4,
            "overlay" => 5,
            "darken" | "min" => 6,
            "lighten" | "max" => 7,
            "difference" | "diff" => 8,
            "soft light" | "softlight" | "soft_light" => 9,
            "dodge" | "color dodge" => 10,
            "burn" | "color burn" => 11,
            "linear light" | "linearlight" | "linear_light" => 12,
            "divide" | "div" => 13,
            "exclusion" => 14,
            "hue" => 15,
            "saturation" => 16,
            "color" => 17,
            "value" | "luminosity" => 18,
            _ => 0,
        };
    }
    (param_f64(params, keys, 0.0).round() as i32).clamp(0, 18)
}

fn resolve_math_op(params: &BTreeMap<String, Value>, keys: &[&str], default: i32) -> i32 {
    if let Some(name) = param_str(params, keys) {
        return match name.trim().to_ascii_lowercase().as_str() {
            "add" | "+" => 0,
            "subtract" | "sub" | "-" => 1,
            "multiply" | "mul" | "*" => 2,
            "divide" | "div" | "/" => 3,
            "minimum" | "min" => 4,
            "maximum" | "max" => 5,
            "power" | "pow" => 6,
            "modulo" | "mod" => 7,
            "absolute" | "abs" => 8,
            "invert" => 9,
            "clamp" => 10,
            "compare" => 11,
            "sine" | "sin" => 12,
            "round" => 13,
            "square root" | "sqrt" => 14,
            "logarithm" | "log" => 15,
            "exponent" | "exp" => 16,
            "floor" => 17,
            "ceiling" | "ceil" => 18,
            "fraction" | "fract" => 19,
            "sign" => 20,
            "smooth minimum" | "smoothmin" | "smin" => 21,
            "smooth maximum" | "smoothmax" | "smax" => 22,
            "snap" => 23,
            "wrap" => 24,
            "ping-pong" | "pingpong" => 25,
            "cosine" | "cos" => 26,
            "tangent" | "tan" => 27,
            "arctan2" | "atan2" => 28,
            "greater than" | "gt" | ">" => 29,
            "less than" | "lt" | "<" => 30,
            _ => default,
        };
    }
    (param_f64(params, keys, f64::from(default)).round() as i32).clamp(0, 30)
}

fn resolve_vector_math_op(params: &BTreeMap<String, Value>) -> i32 {
    if let Some(name) = param_str(params, &["mode", "op", "operation"]) {
        return match name.trim().to_ascii_lowercase().as_str() {
            "add" => 0,
            "subtract" | "sub" => 1,
            "multiply" | "mul" => 2,
            "divide" | "div" => 3,
            "normalize" | "norm" => 4,
            "dot product" | "dot" => 5,
            "length" | "len" => 6,
            "distance" | "dist" => 7,
            "cross product" | "cross" => 8,
            _ => 0,
        };
    }
    (param_f64(params, &["mode", "op", "operation"], 0.0).round() as i32).clamp(0, 8)
}

// ===========================================================================
// A. Node Effect Compiler -> GLSL ES 1.00 (`compile_node_effect_to_glsl`)
// ===========================================================================

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

const NODE_EFFECT_PASSTHROUGH_GLSL: &str = "\
precision highp float;
uniform sampler2D tDiffuse;
uniform vec2 uvScale;
uniform vec2 resolution;
varying vec2 vUv;
void main() {
    gl_FragColor = texture2D(tDiffuse, vUv);
}
";

const NODE_EFFECT_HEADER_GLSL: &str = r#"precision highp float;
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

fn glsl_float(n: f64) -> String {
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

fn glsl_scalar(expr: &str) -> String {
    format!("vec4(vec3({expr}), 1.0)")
}

fn glsl_rgb(expr: &str) -> String {
    format!("vec4({expr}, 1.0)")
}

fn glsl_literal_value(val: Option<&Value>, fallback_num: f64) -> String {
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

fn encode_param_uniform(node_id: &str) -> String {
    let mut out = String::from("ne_");
    for byte in node_id.bytes() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn normalize_effect_node_type(raw: &str) -> String {
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

fn is_color_output(node_type: &str, port: &str, params: &BTreeMap<String, Value>) -> bool {
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

struct EffectCompiler<'a> {
    by_id: BTreeMap<&'a str, &'a RawNode>,
    incoming: BTreeMap<(String, String), (String, String)>,
    emitted: BTreeMap<String, BTreeMap<String, String>>,
    lines: Vec<String>,
    functions: Vec<String>,
    parameters: Vec<NodeEffectParameter>,
    counter: usize,
    samples: usize,
    uses_time: bool,
}

impl<'a> EffectCompiler<'a> {
    fn cost(&mut self, n: usize) -> Result<(), String> {
        self.samples += n;
        if self.samples > MAX_EFFECT_SAMPLES {
            return Err(format!(
                "Node Effect exceeds the {MAX_EFFECT_SAMPLES} source samples per pixel limit"
            ));
        }
        Ok(())
    }

    fn alloc_var(&mut self, prefix: &str) -> String {
        let id = self.counter;
        self.counter += 1;
        format!("{prefix}{id}")
    }

    fn push_out(
        &mut self,
        outs: &mut BTreeMap<String, String>,
        port: &str,
        expr: String,
    ) -> String {
        let name = self.alloc_var("n");
        self.lines.push(format!(" vec4 {name} = {expr};"));
        outs.insert(port.to_string(), name.clone());
        name
    }

    fn resolve_link(&self, node_id: &str, port_aliases: &[&str]) -> Option<(String, String)> {
        for &alias in port_aliases {
            if let Some(src) = self.incoming.get(&(node_id.to_string(), alias.to_string())) {
                return Some(src.clone());
            }
        }
        None
    }

    fn read_port_expr(
        &self,
        node: &RawNode,
        port_aliases: &[&str],
        fallback: Option<String>,
        default_num: f64,
    ) -> String {
        if let Some((src_node, src_port)) = self.resolve_link(&node.id, port_aliases)
            && let Some(src_outs) = self.emitted.get(&src_node)
        {
            if let Some(var) = src_outs.get(&src_port) {
                return var.clone();
            }
            // Fallback to primary output when port names differ across aliases (`out` vs `color`).
            if let Some(var) = src_outs
                .get("color")
                .or_else(|| src_outs.get("out"))
                .or_else(|| src_outs.values().next())
            {
                return var.clone();
            }
        }
        for &alias in port_aliases {
            if let Some(val) = node.params.get(alias) {
                return glsl_literal_value(Some(val), default_num);
            }
        }
        fallback.unwrap_or_else(|| glsl_scalar(&glsl_float(default_num)))
    }

    fn read_num(&self, node: &RawNode, port_aliases: &[&str], default_num: f64) -> String {
        if let Some((src_node, src_port)) = self.resolve_link(&node.id, port_aliases)
            && let Some(src_raw) = self.by_id.get(src_node.as_str())
        {
            let src_norm = normalize_effect_node_type(&src_raw.kind);
            let expr = self.read_port_expr(node, port_aliases, None, default_num);
            if is_color_output(&src_norm, &src_port, &src_raw.params) {
                return format!("dot(({expr}).rgb, vec3(0.3, 0.59, 0.11))");
            }
            return format!("({expr}).x");
        }
        let expr = self.read_port_expr(node, port_aliases, None, default_num);
        format!("({expr}).x")
    }

    fn read_col(&self, node: &RawNode, port_aliases: &[&str], default_rgb: [f64; 3]) -> String {
        let fb = glsl_rgb(&format!(
            "vec3({}, {}, {})",
            glsl_float(default_rgb[0]),
            glsl_float(default_rgb[1]),
            glsl_float(default_rgb[2])
        ));
        let expr = self.read_port_expr(node, port_aliases, Some(fb), 0.0);
        format!("({expr}).rgb")
    }

    fn read_uv(&self, node: &RawNode) -> String {
        let expr = self.read_port_expr(
            node,
            &["uv", "coords", "position"],
            Some("vec4(vUv, 0.0, 1.0)".to_string()),
            0.0,
        );
        format!("({expr}).xy")
    }

    fn emit_node(&mut self, node: &RawNode) -> Result<(), String> {
        let norm = normalize_effect_node_type(&node.kind);
        let mut outs = BTreeMap::new();

        match norm.as_str() {
            "layer" => {
                self.cost(1)?;
                let col = self.push_out(&mut outs, "color", "ne_sample(vUv)".to_string());
                outs.insert("out".to_string(), col.clone());
                self.push_out(&mut outs, "alpha", glsl_scalar(&format!("({col}).a")));
            }
            "uv" => {
                let v = self.push_out(&mut outs, "out", "vec4(vUv, 0.0, 1.0)".to_string());
                outs.insert("uv".to_string(), v);
            }
            "pixel" => {
                self.push_out(
                    &mut outs,
                    "out",
                    "vec4(vUv * resolution, 0.0, 1.0)".to_string(),
                );
            }
            "resolution" => {
                self.push_out(&mut outs, "out", "vec4(resolution, 0.0, 1.0)".to_string());
            }
            "time" => {
                self.uses_time = true;
                self.push_out(&mut outs, "out", glsl_scalar("ne_time"));
            }
            "parameter" => {
                let kind =
                    (param_f64(&node.params, &["kind"], 0.0).round() as i32).clamp(0, 2) as u8;
                let uniform_name = encode_param_uniform(&node.id);
                let label = param_str(&node.params, &["name", "label"])
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .unwrap_or("Parameter")
                    .to_string();
                let min_v = param_f64(&node.params, &["min"], 0.0) as f32;
                let max_v = param_f64(&node.params, &["max"], 1.0) as f32;
                let default_value = match kind {
                    1 => {
                        let c = param_color3(&node.params, &["color", "value"], [1.0, 1.0, 1.0]);
                        vec![c[0] as f32, c[1] as f32, c[2] as f32]
                    }
                    2 => vec![
                        param_f64(&node.params, &["x"], 0.5) as f32,
                        param_f64(&node.params, &["y"], 0.5) as f32,
                    ],
                    _ => vec![param_f64(&node.params, &["value"], 0.5) as f32],
                };
                let expr = match kind {
                    1 => glsl_rgb(&uniform_name),
                    2 => format!("vec4({uniform_name}, 0.0, 1.0)"),
                    _ => glsl_scalar(&uniform_name),
                };
                self.parameters.push(NodeEffectParameter {
                    id: node.id.clone(),
                    uniform_name,
                    label,
                    kind,
                    default_value,
                    min: min_v.min(max_v),
                    max: min_v.max(max_v),
                });
                let v = self.push_out(&mut outs, "out", expr);
                outs.insert("color".to_string(), v);
            }
            "noise" => {
                let uv = self.read_uv(node);
                let scale = self.read_num(node, &["scale"], 8.0);
                let time = self.read_num(node, &["time"], 0.0);
                let seed = self.read_num(node, &["seed"], 0.0);
                let q = format!("({uv} * {scale} + vec2({time}, {seed}))");
                self.push_out(&mut outs, "out", glsl_scalar(&format!("ne_noise({q})")));
                self.push_out(
                    &mut outs,
                    "color",
                    glsl_rgb(&format!(
                        "vec3(ne_noise({q}), ne_noise({q} + 17.3), ne_noise({q} + 41.7))"
                    )),
                );
            }
            "fbm" => {
                let uv = self.read_uv(node);
                let scale = self.read_num(node, &["scale"], 6.0);
                let time = self.read_num(node, &["time"], 0.0);
                let seed = self.read_num(node, &["seed"], 0.0);
                let lac = self.read_num(node, &["lacunarity"], 2.0);
                let gain = self.read_num(node, &["gain", "persistence"], 0.5);
                let q = format!("({uv} * {scale} + vec2({time}, {seed}))");
                let v = self.push_out(
                    &mut outs,
                    "out",
                    glsl_scalar(&format!("ne_fbm({q}, {lac}, {gain})")),
                );
                outs.insert("color".to_string(), v);
            }
            "voronoi" => {
                let uv = self.read_uv(node);
                let scale = self.read_num(node, &["scale"], 8.0);
                let time = self.read_num(node, &["time"], 0.0);
                let mode = if let Some(m) = param_str(&node.params, &["mode", "feature"]) {
                    match m.trim().to_ascii_lowercase().as_str() {
                        "f2" => 1,
                        "edge" | "distance to edge" => 2,
                        _ => 0,
                    }
                } else {
                    (param_f64(&node.params, &["mode", "feature"], 0.0).round() as i32).clamp(0, 2)
                };
                let v = self.push_out(
                    &mut outs,
                    "out",
                    glsl_scalar(&format!("ne_voronoi({uv} * {scale}, {time})[{mode}]")),
                );
                outs.insert("color".to_string(), v);
            }
            "gradient" => {
                let uv = self.read_uv(node);
                let angle = self.read_num(node, &["angle"], 0.0);
                let mode = if let Some(m) = param_str(&node.params, &["mode", "type"]) {
                    match m.trim().to_ascii_lowercase().as_str() {
                        "radial" | "circular" => 1,
                        "angular" => 2,
                        _ => 0,
                    }
                } else {
                    (param_f64(&node.params, &["mode", "type"], 0.0).round() as i32).clamp(0, 2)
                };
                let expr = match mode {
                    1 => format!("length({uv} - 0.5) * 2.0"),
                    2 => format!(
                        "(({uv}).x == 0.5 && ({uv}).y == 0.5 ? 0.5 : atan(({uv}).y - 0.5, ({uv}).x - 0.5) / 6.2831853 + 0.5)"
                    ),
                    _ => format!("dot({uv} - 0.5, vec2(cos({angle}), sin({angle}))) + 0.5"),
                };
                let v = self.push_out(&mut outs, "out", glsl_scalar(&expr));
                outs.insert("color".to_string(), v);
            }
            "checker" => {
                let uv = self.read_uv(node);
                let scale = self.read_num(node, &["scale"], 8.0);
                let a = self.read_col(node, &["a", "color1"], [0.0, 0.0, 0.0]);
                let b = self.read_col(node, &["b", "color2"], [1.0, 1.0, 1.0]);
                let fac = self.push_out(
                    &mut outs,
                    "out",
                    glsl_scalar(&format!(
                        "mod(floor(({uv}).x * {scale}) + floor(({uv}).y * {scale}), 2.0)"
                    )),
                );
                self.push_out(
                    &mut outs,
                    "color",
                    glsl_rgb(&format!("mix({a}, {b}, {fac}.x)")),
                );
            }
            "wave" => {
                let uv = self.read_uv(node);
                let scale = self.read_num(node, &["scale"], 8.0);
                let time = self.read_num(node, &["time"], 0.0);
                let angle = self.read_num(node, &["angle"], 0.0);
                let v = self.push_out(
                    &mut outs,
                    "out",
                    glsl_scalar(&format!(
                        "0.5 + 0.5 * sin(dot({uv}, vec2(cos({angle}), sin({angle}))) * {scale} * 6.2831853 + {time})"
                    )),
                );
                outs.insert("color".to_string(), v);
            }
            "solid" => {
                let col = self.read_col(node, &["color", "a"], [1.0, 1.0, 1.0]);
                let alpha = self.read_num(node, &["alpha", "opacity"], 1.0);
                let v = self.push_out(
                    &mut outs,
                    "color",
                    format!("vec4({col}, clamp({alpha}, 0.0, 1.0))"),
                );
                outs.insert("out".to_string(), v);
            }
            "vignette" => {
                let uv = self.read_uv(node);
                let radius = self.read_num(node, &["radius"], 0.75);
                let softness = self.read_num(node, &["softness"], 0.45);
                let strength = self.read_num(node, &["strength", "amount"], 1.0);
                let mask_expr = format!(
                    "mix(1.0, 1.0 - smoothstep({radius} - max({softness}, 0.0001), {radius}, length({uv} - 0.5) * 1.41421356), clamp({strength}, 0.0, 1.0))"
                );
                if self.resolve_link(&node.id, &["color", "in"]).is_some() {
                    let src = self.read_port_expr(node, &["color", "in"], None, 1.0);
                    let m = self.push_out(&mut outs, "out", glsl_scalar(&mask_expr));
                    self.push_out(
                        &mut outs,
                        "color",
                        format!("vec4(({src}).rgb * ({m}).x, ({src}).a)"),
                    );
                } else {
                    let m = self.push_out(&mut outs, "out", glsl_scalar(&mask_expr));
                    outs.insert("color".to_string(), m);
                }
            }
            "grain" => {
                let uv = self.read_uv(node);
                let amount = self.read_num(node, &["amount", "intensity", "strength"], 0.12);
                let scale = self.read_num(node, &["scale"], 1.0);
                let seed = self.read_num(node, &["seed", "time"], 0.0);
                let g_expr = format!(
                    "(ne_hash(floor({uv} * resolution * max({scale}, 0.001)) + vec2({seed} * 37.0, {seed} * 17.0)) * 2.0 - 1.0) * {amount}"
                );
                if self.resolve_link(&node.id, &["color", "in"]).is_some() {
                    let src = self.read_port_expr(node, &["color", "in"], None, 0.5);
                    let v = self.push_out(
                        &mut outs,
                        "color",
                        format!("vec4(clamp(({src}).rgb + vec3({g_expr}), 0.0, 1.0), ({src}).a)"),
                    );
                    outs.insert("out".to_string(), v);
                } else {
                    let v = self.push_out(
                        &mut outs,
                        "out",
                        glsl_scalar(&format!("clamp(0.5 + {g_expr}, 0.0, 1.0)")),
                    );
                    outs.insert("color".to_string(), v);
                }
            }
            "math" => {
                let mode = resolve_math_op(&node.params, &["mode", "op", "operation"], 0);
                let a = self.read_num(node, &["a"], 0.5);
                let b = self.read_num(node, &["b"], 0.5);
                let c = self.read_num(node, &["c"], 0.5);
                self.push_out(
                    &mut outs,
                    "out",
                    glsl_scalar(&format!("ne_math({a}, {b}, {c}, {mode})")),
                );
            }
            "vectorMath" => {
                let mode = resolve_vector_math_op(&node.params);
                let a = self.read_col(node, &["a"], [0.0, 0.0, 0.0]);
                let b = self.read_col(node, &["b"], [0.0, 0.0, 0.0]);
                let expr = match mode {
                    0 => format!("{a} + {b}"),
                    1 => format!("{a} - {b}"),
                    2 => format!("{a} * {b}"),
                    3 => format!(
                        "vec3(ne_div(({a}).x, ({b}).x), ne_div(({a}).y, ({b}).y), ne_div(({a}).z, ({b}).z))"
                    ),
                    4 => format!("length({a}) > 0.0 ? normalize({a}) : vec3(0.0)"),
                    5 => format!("vec3(dot({a}, {b}))"),
                    6 => format!("vec3(length({a}))"),
                    7 => format!("vec3(distance({a}, {b}))"),
                    _ => format!("cross({a}, {b})"),
                };
                let vec_out = self.push_out(&mut outs, "out", glsl_rgb(&expr));
                self.push_out(&mut outs, "value", glsl_scalar(&format!("{vec_out}.x")));
            }
            "mapRange" => {
                let val = self.read_num(node, &["value", "in"], 0.5);
                let from_min = self.read_num(node, &["fromMin", "inMin"], 0.0);
                let from_max = self.read_num(node, &["fromMax", "inMax"], 1.0);
                let to_min = self.read_num(node, &["toMin", "outMin"], 0.0);
                let to_max = self.read_num(node, &["toMax", "outMax"], 1.0);
                let do_clamp = param_f64(&node.params, &["clamp"], 0.0) != 0.0;
                let mut t = format!("ne_div({val} - {from_min}, {from_max} - {from_min})");
                if do_clamp {
                    t = format!("clamp({t}, 0.0, 1.0)");
                }
                self.push_out(
                    &mut outs,
                    "out",
                    glsl_scalar(&format!("mix({to_min}, {to_max}, {t})")),
                );
            }
            "clamp" => {
                let val = self.read_num(node, &["value", "in"], 0.5);
                let lo = self.read_num(node, &["min"], 0.0);
                let hi = self.read_num(node, &["max"], 1.0);
                self.push_out(
                    &mut outs,
                    "out",
                    glsl_scalar(&format!("clamp({val}, min({lo}, {hi}), max({lo}, {hi}))")),
                );
            }
            "smoothstep" => {
                let e0 = self.read_num(node, &["edge0", "min"], 0.0);
                let e1 = self.read_num(node, &["edge1", "max"], 1.0);
                let x = self.read_num(node, &["x", "value", "in"], 0.5);
                self.push_out(
                    &mut outs,
                    "out",
                    glsl_scalar(&format!(
                        "smoothstep({e0}, abs({e1} - {e0}) < 1.0e-6 ? {e0} + 1.0e-6 : {e1}, {x})"
                    )),
                );
            }
            "trig" => {
                let x = self.read_num(node, &["x", "value", "a"], 0.0);
                let op = param_str(&node.params, &["op", "mode", "function"])
                    .unwrap_or("sin")
                    .to_ascii_lowercase();
                let expr = match op.as_str() {
                    "cos" | "cosine" | "1" => format!("cos({x})"),
                    "tan" | "tangent" | "2" => format!("tan({x})"),
                    "asin" | "3" => format!("asin(clamp({x}, -1.0, 1.0))"),
                    "acos" | "4" => format!("acos(clamp({x}, -1.0, 1.0))"),
                    "atan" | "5" => format!("atan({x})"),
                    _ => format!("sin({x})"),
                };
                self.push_out(&mut outs, "out", glsl_scalar(&expr));
            }
            "mix" => {
                let mode = resolve_blend_mode(&node.params, &["mode", "blendMode"]);
                let a = self.read_col(node, &["a", "base"], [0.0, 0.0, 0.0]);
                let b = self.read_col(node, &["b", "top"], [1.0, 1.0, 1.0]);
                let fac = self.read_num(node, &["factor", "opacity", "mix"], 0.5);
                let v = self.push_out(
                    &mut outs,
                    "color",
                    glsl_rgb(&format!("ne_mix({a}, {b}, {fac}, {mode})")),
                );
                outs.insert("out".to_string(), v);
            }
            "colorRamp" => {
                let stops = parse_ramp_stops(&node.params);
                let interp =
                    (param_f64(&node.params, &["interpolation"], 1.0).round() as i32).clamp(0, 3);
                let fn_name = self.alloc_var("ne_ramp");
                let vec4_stop = |s: [f64; 5]| {
                    format!(
                        "vec4({}, {}, {}, {})",
                        glsl_float(s[1]),
                        glsl_float(s[2]),
                        glsl_float(s[3]),
                        glsl_float(s[4])
                    )
                };
                let mut body = vec![
                    format!("vec4 {fn_name}(float t) {{"),
                    " t = clamp(t, 0.0, 1.0);".to_string(),
                    format!(
                        " if (t < {}) return {};",
                        glsl_float(stops[0][0]),
                        vec4_stop(stops[0])
                    ),
                ];
                for i in 0..stops.len() - 1 {
                    let a = stops[i];
                    let b = stops[i + 1];
                    let prev = stops[i.saturating_sub(1)];
                    let after = stops[(i + 2).min(stops.len() - 1)];
                    let span = if (b[0] - a[0]).abs() > 1e-12 {
                        b[0] - a[0]
                    } else {
                        1.0
                    };
                    body.push(format!(
                        " if (t < {}) {{ float u = clamp((t - {}) / {}, 0.0, 1.0);",
                        glsl_float(b[0]),
                        glsl_float(a[0]),
                        glsl_float(span)
                    ));
                    if interp == 0 {
                        body.push(format!("  return {}; }}", vec4_stop(a)));
                    } else if interp == 3 {
                        let va = vec4_stop(a);
                        let vb = vec4_stop(b);
                        let vp = vec4_stop(prev);
                        let vn = vec4_stop(after);
                        body.push(format!(
                            "  return clamp(0.5 * (2.0 * {va} + (-{vp} + {vb}) * u + (2.0 * {vp} - 5.0 * {va} + 4.0 * {vb} - {vn}) * u * u + (-{vp} + 3.0 * {va} - 3.0 * {vb} + {vn}) * u * u * u), 0.0, 1.0); }}"
                        ));
                    } else {
                        let ease = if interp == 2 {
                            "u * u * (3.0 - 2.0 * u)"
                        } else {
                            "u"
                        };
                        body.push(format!(
                            "  return mix({}, {}, {ease}); }}",
                            vec4_stop(a),
                            vec4_stop(b)
                        ));
                    }
                }
                body.push(format!(
                    " return {};\n}}",
                    vec4_stop(*stops.last().unwrap())
                ));
                self.functions.push(body.join("\n"));

                let factor = self.read_num(node, &["factor", "t", "in"], 0.5);
                let c = self.push_out(&mut outs, "color", format!("{fn_name}({factor})"));
                outs.insert("out".to_string(), c.clone());
                self.push_out(&mut outs, "alpha", glsl_scalar(&format!("{c}.a")));
            }
            "separateRGB" => {
                let src = self.read_port_expr(node, &["color", "in"], None, 0.0);
                self.push_out(&mut outs, "r", glsl_scalar(&format!("({src}).r")));
                self.push_out(&mut outs, "g", glsl_scalar(&format!("({src}).g")));
                self.push_out(&mut outs, "b", glsl_scalar(&format!("({src}).b")));
                self.push_out(&mut outs, "a", glsl_scalar(&format!("({src}).a")));
            }
            "separateHSV" => {
                let c = self.read_col(node, &["color", "in"], [0.0, 0.0, 0.0]);
                let hsv = self.alloc_var("nhsv");
                self.lines.push(format!(" vec3 {hsv} = ne_hsv({c});"));
                self.push_out(&mut outs, "h", glsl_scalar(&format!("{hsv}.x")));
                self.push_out(&mut outs, "s", glsl_scalar(&format!("{hsv}.y")));
                self.push_out(&mut outs, "v", glsl_scalar(&format!("{hsv}.z")));
            }
            "combineRGB" => {
                let r = self.read_num(node, &["r", "red"], 0.0);
                let g = self.read_num(node, &["g", "green"], 0.0);
                let b = self.read_num(node, &["b", "blue"], 0.0);
                let a = self.read_num(node, &["a", "alpha"], 1.0);
                let v = self.push_out(
                    &mut outs,
                    "color",
                    format!("vec4({r}, {g}, {b}, clamp({a}, 0.0, 1.0))"),
                );
                outs.insert("out".to_string(), v);
            }
            "combineHSV" => {
                let h = self.read_num(node, &["h", "hue"], 0.0);
                let s = self.read_num(node, &["s", "saturation"], 0.0);
                let v = self.read_num(node, &["v", "value"], 0.0);
                let out = self.push_out(
                    &mut outs,
                    "color",
                    glsl_rgb(&format!("ne_rgb(vec3({h}, {s}, {v}))")),
                );
                outs.insert("out".to_string(), out);
            }
            "hue" => {
                let c = self.read_col(node, &["color", "in"], [1.0, 1.0, 1.0]);
                let h = self.read_num(node, &["hue", "h"], 0.0);
                let s = self.read_num(node, &["saturation", "s"], 1.0);
                let v = self.read_num(node, &["value", "v"], 1.0);
                let out = self.push_out(
                    &mut outs,
                    "color",
                    glsl_rgb(&format!(
                        "ne_rgb(ne_hsv({c}) * vec3(1.0, {s}, {v}) + vec3({h}, 0.0, 0.0))"
                    )),
                );
                outs.insert("out".to_string(), out);
            }
            "brightness" => {
                let c = self.read_col(node, &["color", "in"], [0.5, 0.5, 0.5]);
                let br = self.read_num(node, &["brightness"], 0.0);
                let ct = self.read_num(node, &["contrast"], 0.0);
                let out = self.push_out(
                    &mut outs,
                    "color",
                    glsl_rgb(&format!("({c} - 0.5) * (1.0 + {ct}) + 0.5 + {br}")),
                );
                outs.insert("out".to_string(), out);
            }
            "colorBalance" => {
                let c = self.read_col(node, &["color", "in"], [0.5, 0.5, 0.5]);
                let lift = self.read_col(node, &["lift", "shadows"], [0.0, 0.0, 0.0]);
                let gamma = self.read_col(node, &["gamma", "midtones"], [1.0, 1.0, 1.0]);
                let gain = self.read_col(node, &["gain", "highlights"], [1.0, 1.0, 1.0]);
                let out = self.push_out(
                    &mut outs,
                    "color",
                    glsl_rgb(&format!(
                        "pow(clamp(({c} + {lift}) * {gain}, 0.0, 1.0), 1.0 / max({gamma}, vec3(0.01)))"
                    )),
                );
                outs.insert("out".to_string(), out);
            }
            "levels" => {
                let c = self.read_col(node, &["color", "in"], [0.5, 0.5, 0.5]);
                let black = self.read_num(node, &["black", "inBlack"], 0.0);
                let white = self.read_num(node, &["white", "inWhite"], 1.0);
                let gamma = self.read_num(node, &["gamma"], 1.0);
                let out_black = self.read_num(node, &["outBlack"], 0.0);
                let out_white = self.read_num(node, &["outWhite"], 1.0);
                let out = self.push_out(
                    &mut outs,
                    "color",
                    glsl_rgb(&format!(
                        "mix(vec3({out_black}), vec3({out_white}), pow(clamp(({c} - {black}) / max({white} - {black}, 1.0e-12), 0.0, 1.0), vec3(1.0 / max({gamma}, 1.0e-6))))"
                    )),
                );
                outs.insert("out".to_string(), out);
            }
            "invert" => {
                let c = self.read_col(node, &["color", "in"], [0.5, 0.5, 0.5]);
                let fac = self.read_num(node, &["factor", "strength"], 1.0);
                let out = self.push_out(
                    &mut outs,
                    "color",
                    glsl_rgb(&format!("mix({c}, 1.0 - {c}, clamp({fac}, 0.0, 1.0))")),
                );
                outs.insert("out".to_string(), out);
            }
            "grayscale" => {
                let c = self.read_col(node, &["color", "in"], [0.5, 0.5, 0.5]);
                let fac = self.read_num(node, &["factor", "amount"], 1.0);
                let out = self.push_out(
                    &mut outs,
                    "color",
                    glsl_rgb(&format!(
                        "mix({c}, vec3(dot({c}, vec3(0.2126, 0.7152, 0.0722))), clamp({fac}, 0.0, 1.0))"
                    )),
                );
                outs.insert("out".to_string(), out);
            }
            "uvTransform" => {
                let uv = self.read_uv(node);
                let ox = self.read_num(node, &["offsetX", "translateX"], 0.0);
                let oy = self.read_num(node, &["offsetY", "translateY"], 0.0);
                let sx = self.read_num(node, &["scaleX", "scale"], 1.0);
                let sy = self.read_num(node, &["scaleY", "scale"], 1.0);
                let ang = self.read_num(node, &["angle", "rotation"], 0.0);
                let p = self.alloc_var("nuv");
                let ca = self.alloc_var("nca");
                let sa = self.alloc_var("nsa");
                self.lines.push(format!(
                    " vec2 {p} = ({uv} - 0.5 - vec2({ox}, {oy})) / max(abs(vec2({sx}, {sy})), vec2(0.0001));"
                ));
                self.lines
                    .push(format!(" float {ca} = cos({ang}), {sa} = sin({ang});"));
                let v = self.push_out(
                    &mut outs,
                    "out",
                    format!(
                        "vec4(vec2({p}.x * {ca} - {p}.y * {sa}, {p}.x * {sa} + {p}.y * {ca}) + 0.5, 0.0, 1.0)"
                    ),
                );
                outs.insert("uv".to_string(), v);
            }
            "displace" => {
                let uv = self.read_uv(node);
                let map = self.read_col(node, &["map", "displacement", "offset"], [0.5, 0.5, 0.0]);
                let strength = self.read_num(node, &["strength", "amount"], 0.05);
                let uvv = self.push_out(
                    &mut outs,
                    "uv",
                    format!(
                        "vec4(clamp({uv} + ({map}.xy - 0.5) * {strength}, 0.0, 1.0), 0.0, 1.0)"
                    ),
                );
                outs.insert("out".to_string(), uvv.clone());
                self.cost(1)?;
                self.push_out(&mut outs, "color", format!("ne_sample({uvv}.xy)"));
            }
            "swirl" => {
                let uv = self.read_uv(node);
                let radius = self.read_num(node, &["radius"], 0.5);
                let angle = self.read_num(node, &["angle", "strength"], 1.5);
                let cx = self.read_num(node, &["cx", "centerX"], 0.5);
                let cy = self.read_num(node, &["cy", "centerY"], 0.5);
                let fn_name = self.alloc_var("ne_swirl");
                self.functions.push(format!(
                    "vec2 {fn_name}(vec2 uv, vec2 c, float r, float a) {{\n vec2 d = uv - c;\n float dist = length(d);\n float f = clamp(1.0 - dist / max(r, 0.0001), 0.0, 1.0);\n float theta = f * f * a;\n float cs = cos(theta), sn = sin(theta);\n return c + vec2(d.x * cs - d.y * sn, d.x * sn + d.y * cs);\n}}"
                ));
                let uvv = self.push_out(
                    &mut outs,
                    "uv",
                    format!("vec4({fn_name}({uv}, vec2({cx}, {cy}), {radius}, {angle}), 0.0, 1.0)"),
                );
                outs.insert("out".to_string(), uvv.clone());
                self.cost(1)?;
                self.push_out(&mut outs, "color", format!("ne_sample({uvv}.xy)"));
            }
            "kaleidoscope" => {
                let uv = self.read_uv(node);
                let segs = self.read_num(node, &["segments", "sides"], 6.0);
                let angle = self.read_num(node, &["angle", "rotation"], 0.0);
                let fn_name = self.alloc_var("ne_kaleid");
                self.functions.push(format!(
                    "vec2 {fn_name}(vec2 uv, float segs, float rot) {{\n vec2 p = uv - 0.5;\n float r = length(p);\n float a = (p.x == 0.0 && p.y == 0.0) ? 0.0 : atan(p.y, p.x) + rot;\n float sector = 6.2831853 / max(segs, 1.0);\n a = mod(a, sector);\n a = abs(a - 0.5 * sector);\n return clamp(vec2(cos(a), sin(a)) * r + 0.5, 0.0, 1.0);\n}}"
                ));
                let uvv = self.push_out(
                    &mut outs,
                    "uv",
                    format!("vec4({fn_name}({uv}, {segs}, {angle}), 0.0, 1.0)"),
                );
                outs.insert("out".to_string(), uvv.clone());
                self.cost(1)?;
                self.push_out(&mut outs, "color", format!("ne_sample({uvv}.xy)"));
            }
            "pixelate" => {
                let uv = self.read_uv(node);
                let blocks = self.read_num(node, &["blocks", "scale", "size"], 32.0);
                let uvv = self.push_out(
                    &mut outs,
                    "uv",
                    format!(
                        "vec4((floor({uv} * max({blocks}, 1.0)) + 0.5) / max({blocks}, 1.0), 0.0, 1.0)"
                    ),
                );
                outs.insert("out".to_string(), uvv.clone());
                self.cost(1)?;
                self.push_out(&mut outs, "color", format!("ne_sample({uvv}.xy)"));
            }
            "mirror" => {
                let uv = self.read_uv(node);
                let mode =
                    (param_f64(&node.params, &["mode", "axis"], 0.0).round() as i32).clamp(0, 2);
                let expr = match mode {
                    1 => format!("vec2(({uv}).x, 1.0 - abs(({uv}).y * 2.0 - 1.0))"),
                    2 => format!(
                        "vec2(1.0 - abs(({uv}).x * 2.0 - 1.0), 1.0 - abs(({uv}).y * 2.0 - 1.0))"
                    ),
                    _ => format!("vec2(1.0 - abs(({uv}).x * 2.0 - 1.0), ({uv}).y)"),
                };
                let uvv = self.push_out(&mut outs, "uv", format!("vec4({expr}, 0.0, 1.0)"));
                outs.insert("out".to_string(), uvv.clone());
                self.cost(1)?;
                self.push_out(&mut outs, "color", format!("ne_sample({uvv}.xy)"));
            }
            "sample" => {
                let uv = self.read_uv(node);
                self.cost(1)?;
                let c = self.push_out(&mut outs, "color", format!("ne_sample({uv})"));
                outs.insert("out".to_string(), c.clone());
                self.push_out(&mut outs, "alpha", glsl_scalar(&format!("{c}.a")));
            }
            "blur" => {
                let uv = self.read_uv(node);
                let mode = (param_f64(&node.params, &["mode"], 0.0).round() as i32).clamp(0, 3);
                let taps = (param_f64(&node.params, &["taps", "samples"], 9.0).round() as usize)
                    .clamp(1, 32);
                let radius = self.read_num(node, &["radius"], 8.0);
                let angle = self.read_num(node, &["angle"], 0.0);
                let cx = self.read_num(node, &["cx"], 0.5);
                let cy = self.read_num(node, &["cy"], 0.5);
                let fn_name = self.alloc_var("ne_blur");
                let count = if mode < 2 {
                    ((taps as f64).sqrt().floor() as usize).clamp(1, 5)
                } else {
                    taps
                };
                self.cost(if mode < 2 { count * count } else { count })?;
                let mut fn_lines = vec![format!(
                    "vec4 {fn_name}(vec2 uv, float radius, float angle, vec2 center) {{\n vec4 sum = vec4(0.0);\n float total = 0.0;"
                )];
                if mode < 2 {
                    for y in 0..count {
                        for x in 0..count {
                            let dx = if count == 1 {
                                0.0
                            } else {
                                (x as f64) / ((count - 1) as f64) * 2.0 - 1.0
                            };
                            let dy = if count == 1 {
                                0.0
                            } else {
                                (y as f64) / ((count - 1) as f64) * 2.0 - 1.0
                            };
                            let weight = if mode == 1 {
                                (-2.0 * (dx * dx + dy * dy)).exp()
                            } else {
                                1.0
                            };
                            fn_lines.push(format!(
                                " {{ vec4 s = ne_sample(uv + vec2({dx:.6}, {dy:.6}) * radius / resolution); float w = {weight:.6}; sum += vec4(s.rgb * s.a, s.a) * w; total += w; }}"
                            ));
                        }
                    }
                } else {
                    for i in 0..count {
                        let t = if count == 1 {
                            0.0
                        } else {
                            (i as f64) / ((count - 1) as f64) * 2.0 - 1.0
                        };
                        let off = if mode == 2 {
                            format!("vec2(cos(angle), sin(angle)) * radius * ({t:.6}) / resolution")
                        } else {
                            format!(
                                "(uv - center) * radius * ({t:.6}) / max(resolution.x, resolution.y)"
                            )
                        };
                        fn_lines.push(format!(
                            " {{ vec4 s = ne_sample(uv + {off}); sum += vec4(s.rgb * s.a, s.a); total += 1.0; }}"
                        ));
                    }
                }
                fn_lines.push(
                    " return vec4(sum.a > 0.000001 ? sum.rgb / sum.a : vec3(0.0), sum.a / max(total, 1.0));\n}"
                        .to_string(),
                );
                self.functions.push(fn_lines.join("\n"));
                let c = self.push_out(
                    &mut outs,
                    "color",
                    format!("{fn_name}({uv}, {radius}, {angle}, vec2({cx}, {cy}))"),
                );
                outs.insert("out".to_string(), c.clone());
                self.push_out(&mut outs, "alpha", glsl_scalar(&format!("{c}.a")));
            }
            "sharpen" => {
                let uv = self.read_uv(node);
                let radius = self.read_num(node, &["radius"], 1.0);
                let amount = self.read_num(node, &["amount", "strength"], 1.0);
                self.cost(5)?;
                let fn_name = self.alloc_var("ne_sharpen");
                self.functions.push(format!(
                    "vec4 {fn_name}(vec2 uv, float radius, float amount) {{\n vec2 px = radius / max(resolution, vec2(1.0));\n vec4 c = ne_sample(uv);\n vec3 nb = 0.25 * (ne_sample(uv + vec2(px.x, 0.0)).rgb + ne_sample(uv - vec2(px.x, 0.0)).rgb + ne_sample(uv + vec2(0.0, px.y)).rgb + ne_sample(uv - vec2(0.0, px.y)).rgb);\n return vec4(clamp(c.rgb + (c.rgb - nb) * amount, 0.0, 1.0), c.a);\n}}"
                ));
                let c = self.push_out(
                    &mut outs,
                    "color",
                    format!("{fn_name}({uv}, {radius}, {amount})"),
                );
                outs.insert("out".to_string(), c.clone());
                self.push_out(&mut outs, "alpha", glsl_scalar(&format!("{c}.a")));
            }
            "edge" => {
                let uv = self.read_uv(node);
                let radius = self.read_num(node, &["radius"], 1.0);
                let strength = self.read_num(node, &["strength"], 1.0);
                self.cost(9)?;
                let fn_name = self.alloc_var("ne_edge");
                let sx = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
                let sy = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
                let mut fn_lines = vec![format!(
                    "vec4 {fn_name}(vec2 uv, float radius, float strength) {{\n float gx = 0.0, gy = 0.0, alpha = 0.0;"
                )];
                for y in 0..3 {
                    for x in 0..3 {
                        let ox = (x as f64) - 1.0;
                        let oy = (y as f64) - 1.0;
                        let alpha_set = if x == 1 && y == 1 {
                            " alpha = s.a;"
                        } else {
                            ""
                        };
                        fn_lines.push(format!(
                            " {{ vec4 s = ne_sample(uv + vec2({ox:.1}, {oy:.1}) * radius / resolution); float l = dot(s.rgb * s.a, vec3(0.3, 0.59, 0.11)); gx += l * {:.1}; gy += l * {:.1};{alpha_set} }}",
                            sx[y][x], sy[y][x]
                        ));
                    }
                }
                fn_lines.push(
                    " return vec4(vec3(length(vec2(gx, gy)) * strength), alpha);\n}".to_string(),
                );
                self.functions.push(fn_lines.join("\n"));
                let c = self.push_out(
                    &mut outs,
                    "color",
                    format!("{fn_name}({uv}, {radius}, {strength})"),
                );
                self.push_out(&mut outs, "out", glsl_scalar(&format!("{c}.x")));
                self.push_out(&mut outs, "alpha", glsl_scalar(&format!("{c}.a")));
            }
            "chromaticAberration" => {
                let uv = self.read_uv(node);
                let amount = self.read_num(node, &["amount", "offset", "strength"], 0.008);
                let angle = self.read_num(node, &["angle"], 0.0);
                self.cost(3)?;
                let fn_name = self.alloc_var("ne_ca");
                self.functions.push(format!(
                    "vec4 {fn_name}(vec2 uv, float amount, float angle) {{\n vec2 dir = vec2(cos(angle), sin(angle)) * amount;\n vec4 r = ne_sample(clamp(uv + dir, 0.0, 1.0));\n vec4 g = ne_sample(clamp(uv, 0.0, 1.0));\n vec4 b = ne_sample(clamp(uv - dir, 0.0, 1.0));\n return vec4(r.r, g.g, b.b, g.a);\n}}"
                ));
                let c = self.push_out(
                    &mut outs,
                    "color",
                    format!("{fn_name}({uv}, {amount}, {angle})"),
                );
                outs.insert("out".to_string(), c.clone());
                self.push_out(&mut outs, "alpha", glsl_scalar(&format!("{c}.a")));
            }
            "glow" => {
                let uv = self.read_uv(node);
                let threshold = self.read_num(node, &["threshold"], 0.6);
                let radius = self.read_num(node, &["radius"], 6.0);
                let intensity = self.read_num(node, &["intensity", "strength", "amount"], 0.8);
                self.cost(9)?;
                let fn_name = self.alloc_var("ne_glow");
                self.functions.push(format!(
                    "vec4 {fn_name}(vec2 uv, float threshold, float radius, float intensity) {{\n vec4 base = ne_sample(uv);\n vec3 acc = vec3(0.0);\n float total = 0.0;\n for (int y = -1; y <= 1; y++) {{\n  for (int x = -1; x <= 1; x++) {{\n   vec4 s = ne_sample(clamp(uv + vec2(float(x), float(y)) * radius / max(resolution, vec2(1.0)), 0.0, 1.0));\n   float l = dot(s.rgb, vec3(0.2126, 0.7152, 0.0722));\n   float hi = max(l - threshold, 0.0) / max(1.0 - threshold, 0.001);\n   acc += s.rgb * hi;\n   total += 1.0;\n  }}\n }}\n return vec4(clamp(base.rgb + (acc / total) * intensity, 0.0, 1.0), base.a);\n}}"
                ));
                let c = self.push_out(
                    &mut outs,
                    "color",
                    format!("{fn_name}({uv}, {threshold}, {radius}, {intensity})"),
                );
                outs.insert("out".to_string(), c.clone());
                self.push_out(&mut outs, "alpha", glsl_scalar(&format!("{c}.a")));
            }
            "output" => {
                if self.resolve_link(&node.id, &["color", "in"]).is_none() {
                    self.cost(1)?;
                }
                let color = self.read_port_expr(
                    node,
                    &["color", "in"],
                    Some("ne_sample(vUv)".to_string()),
                    0.0,
                );
                let alpha = if self.resolve_link(&node.id, &["alpha"]).is_some()
                    || node.params.contains_key("alpha")
                {
                    self.read_num(node, &["alpha"], 1.0)
                } else {
                    "1.0".to_string()
                };
                self.lines
                    .push(format!(" float ne_alpha = clamp({alpha}, 0.0, 1.0);"));
                self.lines.push(format!(
                    " gl_FragColor = vec4(({color}).rgb * ne_alpha, ne_alpha);"
                ));
            }
            other => {
                return Err(format!(
                    "node `{}`: unsupported node type `{other}`",
                    node.id
                ));
            }
        }

        self.emitted.insert(node.id.clone(), outs);
        Ok(())
    }
}

/// Compiles a Zoidium `node-effect` JSON graph into a Naga-compatible GLSL ES 1.00
/// fragment shader.
///
/// Empty graphs or graphs with no `output` node compile cleanly to a pass-through shader
/// (`gl_FragColor = texture2D(tDiffuse, vUv);`). Cyclic graphs return `Err`.
pub fn compile_node_effect_to_glsl(graph: &Value) -> Result<CompiledNodeEffect, String> {
    let (nodes, links) = parse_raw_graph(graph, MAX_EFFECT_NODES, MAX_EFFECT_LINKS)?;
    if nodes.is_empty() {
        return Ok(CompiledNodeEffect {
            glsl_source: NODE_EFFECT_PASSTHROUGH_GLSL.to_string(),
            parameters: Vec::new(),
            uses_time: false,
            sample_count: 1,
            node_count: 0,
            link_count: 0,
        });
    }

    // Always validate acyclicity across the entire graph.
    let topo = topological_sort(&nodes, &links)?;

    let output_nodes: Vec<&RawNode> = nodes
        .iter()
        .filter(|n| normalize_effect_node_type(&n.kind) == "output")
        .collect();
    if output_nodes.is_empty() {
        return Ok(CompiledNodeEffect {
            glsl_source: NODE_EFFECT_PASSTHROUGH_GLSL.to_string(),
            parameters: Vec::new(),
            uses_time: false,
            sample_count: 1,
            node_count: nodes.len(),
            link_count: links.len(),
        });
    }
    if output_nodes.len() > 1 {
        return Err("Node Effect graph cannot contain multiple Output nodes".to_string());
    }
    let output_id = output_nodes[0].id.as_str();

    // Find only the nodes reachable upstream from the Output node so dead branches are skipped.
    let mut incoming: BTreeMap<(String, String), (String, String)> = BTreeMap::new();
    let mut parents: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for link in &links {
        let key = (link.to_node.clone(), link.to_port.clone());
        if incoming.contains_key(&key) {
            return Err(format!(
                "duplicate incoming link to `{}:{}`",
                link.to_node, link.to_port
            ));
        }
        incoming.insert(key, (link.from_node.clone(), link.from_port.clone()));
        parents
            .entry(link.to_node.as_str())
            .or_default()
            .push(link.from_node.as_str());
    }

    let mut reachable = BTreeSet::new();
    let mut stack = vec![output_id];
    while let Some(id) = stack.pop() {
        if reachable.insert(id)
            && let Some(ps) = parents.get(id)
        {
            for &p in ps {
                stack.push(p);
            }
        }
    }

    let by_id: BTreeMap<&str, &RawNode> = nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let mut compiler = EffectCompiler {
        by_id,
        incoming,
        emitted: BTreeMap::new(),
        lines: Vec::new(),
        functions: Vec::new(),
        parameters: Vec::new(),
        counter: 0,
        samples: 0,
        uses_time: false,
    };

    for id in &topo {
        if reachable.contains(id.as_str())
            && let Some(&node) = compiler.by_id.get(id.as_str())
        {
            compiler.emit_node(node)?;
        }
    }

    let mut uniform_decls = Vec::new();
    for param in &compiler.parameters {
        let ty = match param.kind {
            1 => "vec3",
            2 => "vec2",
            _ => "float",
        };
        uniform_decls.push(format!("uniform {ty} {};", param.uniform_name));
    }
    if compiler.uses_time {
        uniform_decls.push("uniform float ne_time;".to_string());
    }

    let mut glsl_source = String::from(NODE_EFFECT_HEADER_GLSL);
    if !uniform_decls.is_empty() {
        glsl_source.push_str(&uniform_decls.join("\n"));
        glsl_source.push('\n');
    }
    if !compiler.functions.is_empty() {
        glsl_source.push_str(&compiler.functions.join("\n"));
        glsl_source.push('\n');
    }
    glsl_source.push_str("void main() {\n");
    glsl_source.push_str(&compiler.lines.join("\n"));
    glsl_source.push_str("\n}\n");

    Ok(CompiledNodeEffect {
        glsl_source,
        parameters: compiler.parameters,
        uses_time: compiler.uses_time,
        sample_count: compiler.samples,
        node_count: nodes.len(),
        link_count: links.len(),
    })
}

// ===========================================================================
// B. NodeMaterial Procedural Texture Baker (`bake_node_material_maps`)
// ===========================================================================

/// One baked square RGBA8 texture map (`width * height * 4` bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BakedMap {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// Complete PBR texture set and scalar fallbacks baked from a Zoidium `material-plus`
/// `NodeMaterial` procedural graph.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedNodeMaterialMaps {
    pub resolution: u32,
    pub color: Option<BakedMap>,
    pub emissive: Option<BakedMap>,
    pub roughness: Option<BakedMap>,
    pub metalness: Option<BakedMap>,
    pub normal: Option<BakedMap>,
    pub bump: Option<BakedMap>,
    pub opacity: Option<BakedMap>,
    pub base_color_factor: [f32; 4],
    pub emissive_factor: [f32; 3],
    pub roughness_factor: f32,
    pub metalness_factor: f32,
    pub opacity_factor: f32,
    pub normal_strength: f32,
}

impl Default for BakedNodeMaterialMaps {
    fn default() -> Self {
        Self {
            resolution: 128,
            color: None,
            emissive: None,
            roughness: None,
            metalness: None,
            normal: None,
            bump: None,
            opacity: None,
            base_color_factor: [0.8, 0.8, 0.8, 1.0],
            emissive_factor: [0.0, 0.0, 0.0],
            roughness_factor: 0.5,
            metalness_factor: 0.0,
            opacity_factor: 1.0,
            normal_strength: 1.0,
        }
    }
}

/// Tracks live and peak intermediate buffer allocations during `bake_node_material_maps`,
/// enforcing a strict pre-allocation memory budget.
#[derive(Debug)]
struct BakeMemoryTracker {
    live_bytes: AtomicUsize,
    peak_bytes: AtomicUsize,
    budget_bytes: usize,
}

impl BakeMemoryTracker {
    fn new(budget_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            live_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            budget_bytes,
        })
    }

    fn try_reserve(self: &Arc<Self>, bytes: usize) -> Result<MemoryReservation, String> {
        let prev = self.live_bytes.load(Ordering::Relaxed);
        let next = prev.checked_add(bytes).ok_or_else(|| {
            format!(
                "NodeMaterial bake memory budget exceeded ({prev} + {bytes} bytes > {} bytes)",
                self.budget_bytes
            )
        })?;
        if next > self.budget_bytes {
            return Err(format!(
                "NodeMaterial bake memory budget exceeded: requested {bytes} bytes with {prev} bytes live (budget {} bytes)",
                self.budget_bytes
            ));
        }
        self.live_bytes.store(next, Ordering::Relaxed);
        let peak = self.peak_bytes.load(Ordering::Relaxed);
        if next > peak {
            self.peak_bytes.store(next, Ordering::Relaxed);
        }
        Ok(MemoryReservation {
            bytes,
            tracker: Arc::clone(self),
        })
    }

    #[cfg(test)]
    fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn live_bytes(&self) -> usize {
        self.live_bytes.load(Ordering::Relaxed)
    }
}

#[derive(Debug)]
struct MemoryReservation {
    bytes: usize,
    tracker: Arc<BakeMemoryTracker>,
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.tracker
            .live_bytes
            .fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

#[derive(Debug)]
struct TrackedImageBuf {
    data: Vec<f32>,
    _reservation: MemoryReservation,
}

impl TrackedImageBuf {
    fn alloc_rgba(size: usize, tracker: &Arc<BakeMemoryTracker>) -> Result<Self, String> {
        let floats = size
            .checked_mul(size)
            .and_then(|px| px.checked_mul(4))
            .ok_or_else(|| format!("NodeMaterial buffer size overflow for {size}x{size}"))?;
        let bytes = floats
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("NodeMaterial byte size overflow for {size}x{size}"))?;
        let reservation = tracker.try_reserve(bytes)?;
        Ok(Self {
            data: vec![0.0f32; floats],
            _reservation: reservation,
        })
    }
}

/// Internal per-pixel RGBA float buffer in `[0, 1]` (or unbounded for math/UV).
#[derive(Debug, Clone)]
enum MatBuf {
    Scalar(f64),
    Uniform([f64; 4]),
    Image(Arc<TrackedImageBuf>), // len == size * size * 4, RGBA in [0, 1] (or raw float)
}

impl MatBuf {
    fn from_image(buf: TrackedImageBuf) -> Self {
        Self::Image(Arc::new(buf))
    }

    fn is_spatial(&self) -> bool {
        matches!(self, Self::Image(_))
    }

    fn sample_rgba(&self, px_idx: usize) -> [f64; 4] {
        match self {
            Self::Scalar(v) => [*v, *v, *v, 1.0],
            Self::Uniform(c) => *c,
            Self::Image(buf) => {
                let i = px_idx * 4;
                [
                    f64::from(buf.data[i]),
                    f64::from(buf.data[i + 1]),
                    f64::from(buf.data[i + 2]),
                    f64::from(buf.data[i + 3]),
                ]
            }
        }
    }

    fn sample_luma(&self, px_idx: usize) -> f64 {
        match self {
            Self::Scalar(v) => *v,
            Self::Uniform(c) => {
                if (c[0] - c[1]).abs() < 1e-9 && (c[1] - c[2]).abs() < 1e-9 {
                    c[0]
                } else {
                    c[0] * 0.3 + c[1] * 0.59 + c[2] * 0.11
                }
            }
            Self::Image(buf) => {
                let i = px_idx * 4;
                let r = f64::from(buf.data[i]);
                let g = f64::from(buf.data[i + 1]);
                let b = f64::from(buf.data[i + 2]);
                if (r - g).abs() < 1e-7 && (g - b).abs() < 1e-7 {
                    r
                } else {
                    r * 0.3 + g * 0.59 + b * 0.11
                }
            }
        }
    }

    fn mean_rgba(&self) -> [f32; 4] {
        match self {
            Self::Scalar(v) => {
                let c = (*v as f32).clamp(0.0, 1.0);
                [c, c, c, 1.0]
            }
            Self::Uniform(c) => [
                (c[0] as f32).clamp(0.0, 1.0),
                (c[1] as f32).clamp(0.0, 1.0),
                (c[2] as f32).clamp(0.0, 1.0),
                (c[3] as f32).clamp(0.0, 1.0),
            ],
            Self::Image(buf) => {
                let n = (buf.data.len() / 4).max(1) as f64;
                let mut sum = [0.0f64; 4];
                for px in buf.data.as_chunks::<4>().0 {
                    sum[0] += f64::from(px[0]);
                    sum[1] += f64::from(px[1]);
                    sum[2] += f64::from(px[2]);
                    sum[3] += f64::from(px[3]);
                }
                [
                    (sum[0] / n).clamp(0.0, 1.0) as f32,
                    (sum[1] / n).clamp(0.0, 1.0) as f32,
                    (sum[2] / n).clamp(0.0, 1.0) as f32,
                    (sum[3] / n).clamp(0.0, 1.0) as f32,
                ]
            }
        }
    }

    fn to_rgba8(&self, size: usize) -> BakedMap {
        let total = size * size;
        let mut rgba8 = vec![0u8; total * 4];
        match self {
            Self::Scalar(v) => {
                let b = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                for px in rgba8.as_chunks_mut::<4>().0 {
                    px[0] = b;
                    px[1] = b;
                    px[2] = b;
                    px[3] = 255;
                }
            }
            Self::Uniform(c) => {
                let r = (c[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (c[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (c[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                let a = (c[3].clamp(0.0, 1.0) * 255.0).round() as u8;
                for px in rgba8.as_chunks_mut::<4>().0 {
                    px[0] = r;
                    px[1] = g;
                    px[2] = b;
                    px[3] = a;
                }
            }
            Self::Image(buf) => {
                for (dst, src) in rgba8
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(buf.data.as_chunks::<4>().0.iter())
                {
                    dst[0] = (f64::from(src[0]).clamp(0.0, 1.0) * 255.0).round() as u8;
                    dst[1] = (f64::from(src[1]).clamp(0.0, 1.0) * 255.0).round() as u8;
                    dst[2] = (f64::from(src[2]).clamp(0.0, 1.0) * 255.0).round() as u8;
                    dst[3] = (f64::from(src[3]).clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        BakedMap {
            width: size as u32,
            height: size as u32,
            rgba8,
        }
    }
}

/// Mulberry32 PRNG matching Zoidium `graph-buffers.js:random`.
struct Mulberry32(u32);

impl Mulberry32 {
    fn new(seed: i32) -> Self {
        Self(seed as u32)
    }

    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b_79f5);
        let mut t = (self.0 ^ (self.0 >> 15)).wrapping_mul(1 | self.0);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        f64::from(t ^ (t >> 14)) / 4_294_967_296.0
    }
}

/// Deterministic integer hash in `[0, 1)` matching Zoidium `graph-procedural.js:hash`.
fn proc_hash(x: i32, y: i32, seed: i32) -> f64 {
    let mut h = x.wrapping_mul(374_761_393_i32)
        ^ y.wrapping_mul(668_265_263_i32)
        ^ seed.wrapping_mul(1_442_695_041_i32);
    h = (h ^ (((h as u32) >> 13) as i32)).wrapping_mul(1_274_126_177_i32);
    let u = (h as u32) ^ ((h as u32) >> 16);
    f64::from(u) / 4_294_967_296.0
}

fn rgb_to_hsv(r: f64, g: f64, b: f64) -> [f64; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d > 0.0 {
        if (max - r).abs() < 1e-12 {
            (g - b) / d + if g < b { 6.0 } else { 0.0 }
        } else if (max - g).abs() < 1e-12 {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        }
    } else {
        0.0
    };
    [h / 6.0, if max > 0.0 { d / max } else { 0.0 }, max]
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [f64; 3] {
    let hh = (h - h.floor()) * 6.0;
    let sector = hh.floor() as i32;
    let f = hh - f64::from(sector);
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match sector {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

fn mat_blend_channel(base: f64, top: f64, mode: i32) -> f64 {
    match mode {
        1 => base + top,
        2 => base - top,
        3 => base * top,
        4 => 1.0 - (1.0 - base) * (1.0 - top),
        5 => {
            if base < 0.5 {
                2.0 * base * top
            } else {
                1.0 - 2.0 * (1.0 - base) * (1.0 - top)
            }
        }
        6 => base.min(top),
        7 => base.max(top),
        8 => (base - top).abs(),
        9 => {
            if top <= 0.5 {
                base - (1.0 - 2.0 * top) * base * (1.0 - base)
            } else {
                let d = if base <= 0.25 {
                    ((16.0 * base - 12.0) * base + 4.0) * base
                } else {
                    base.max(0.0).sqrt()
                };
                base + (2.0 * top - 1.0) * (d - base)
            }
        }
        10 => {
            if top >= 1.0 {
                1.0
            } else {
                (base / (1.0 - top)).min(1.0)
            }
        }
        11 => {
            if top <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - base) / top).min(1.0)
            }
        }
        12 => base + 2.0 * top - 1.0,
        13 => {
            if top.abs() < 1e-12 {
                0.0
            } else {
                base / top
            }
        }
        14 => base + top - 2.0 * base * top,
        _ => top,
    }
}

fn mat_blend_rgb(a: [f64; 3], b: [f64; 3], opacity: f64, mode: i32) -> [f64; 3] {
    let op = opacity.clamp(0.0, 1.0);
    let mixed = if mode >= 15 {
        let ha = rgb_to_hsv(a[0], a[1], a[2]);
        let hb = rgb_to_hsv(b[0], b[1], b[2]);
        hsv_to_rgb(
            if mode == 15 || mode == 17 {
                hb[0]
            } else {
                ha[0]
            },
            if mode == 16 || mode == 17 {
                hb[1]
            } else {
                ha[1]
            },
            if mode == 18 { hb[2] } else { ha[2] },
        )
    } else {
        [
            mat_blend_channel(a[0], b[0], mode),
            mat_blend_channel(a[1], b[1], mode),
            mat_blend_channel(a[2], b[2], mode),
        ]
    };
    [
        a[0] + (mixed[0] - a[0]) * op,
        a[1] + (mixed[1] - a[1]) * op,
        a[2] + (mixed[2] - a[2]) * op,
    ]
}

fn mat_math_op(x: f64, y: f64, z: f64, op: i32) -> f64 {
    let v = match op {
        0 => x + y,
        1 => x - y,
        2 => x * y,
        3 => {
            if y.abs() < 1e-12 {
                0.0
            } else {
                x / y
            }
        }
        4 => x.min(y),
        5 => x.max(y),
        6 => x.powf(y),
        7 => {
            if y.abs() < 1e-12 {
                0.0
            } else {
                x % y
            }
        }
        8 => x.abs(),
        9 => 1.0 - x,
        10 => x.clamp(0.0, 1.0),
        11 => {
            if (x - y).abs() <= 0.01 {
                1.0
            } else {
                0.0
            }
        }
        12 => x.sin(),
        13 => x.round(),
        14 => x.sqrt(),
        15 => {
            if x > 0.0 && y > 0.0 && (y - 1.0).abs() > 1e-12 {
                x.ln() / y.ln()
            } else {
                0.0
            }
        }
        16 => x.exp(),
        17 => x.floor(),
        18 => x.ceil(),
        19 => x - x.floor(),
        20 => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        21 | 22 => {
            let k = z.max(0.0);
            let a = if op == 22 { -x } else { x };
            let b = if op == 22 { -y } else { y };
            let h = if k > 0.0 {
                (k - (a - b).abs()).max(0.0) / k
            } else {
                0.0
            };
            (a.min(b) - h * h * k * 0.25) * if op == 22 { -1.0 } else { 1.0 }
        }
        23 => {
            if y.abs() < 1e-12 {
                0.0
            } else {
                (x / y).floor() * y
            }
        }
        24 => {
            let range = y - z;
            if range.abs() < 1e-12 {
                z
            } else {
                x - range * ((x - z) / range).floor()
            }
        }
        25 => {
            let scale = y.abs();
            if scale < 1e-12 {
                0.0
            } else {
                let m = ((x % (2.0 * scale)) + 2.0 * scale) % (2.0 * scale);
                scale - (m - scale).abs()
            }
        }
        26 => x.cos(),
        27 => x.tan(),
        28 => x.atan2(y),
        29 => {
            if x > y {
                1.0
            } else {
                0.0
            }
        }
        30 => {
            if x < y {
                1.0
            } else {
                0.0
            }
        }
        _ => x,
    };
    if v.is_finite() {
        v.clamp(-1.0e30, 1.0e30)
    } else {
        0.0
    }
}

fn wavelength_color(wavelength: f64) -> [f64; 3] {
    let (mut r, mut g, mut b) = (0.0, 0.0, 0.0);
    if wavelength < 440.0 {
        r = -(wavelength - 440.0) / 60.0;
        b = 1.0;
    } else if wavelength < 490.0 {
        g = (wavelength - 440.0) / 50.0;
        b = 1.0;
    } else if wavelength < 510.0 {
        g = 1.0;
        b = -(wavelength - 510.0) / 20.0;
    } else if wavelength < 580.0 {
        r = (wavelength - 510.0) / 70.0;
        g = 1.0;
    } else if wavelength < 645.0 {
        r = 1.0;
        g = -(wavelength - 645.0) / 65.0;
    } else {
        r = 1.0;
    }
    let factor = if wavelength < 420.0 {
        0.3 + 0.7 * (wavelength - 380.0) / 40.0
    } else if wavelength <= 700.0 {
        1.0
    } else {
        0.3 + 0.7 * (780.0 - wavelength) / 80.0
    }
    .clamp(0.0, 1.0);
    [
        (r * factor).clamp(0.0, 1.0),
        (g * factor).clamp(0.0, 1.0),
        (b * factor).clamp(0.0, 1.0),
    ]
}

fn blackbody_color(kelvin: f64) -> [f64; 3] {
    let temp = kelvin.clamp(1000.0, 40000.0) / 100.0;
    let red = if temp <= 66.0 {
        255.0
    } else {
        329.698_727_446 * (temp - 60.0).powf(-0.133_204_759_2)
    };
    let green = if temp <= 66.0 {
        99.470_802_586_1 * temp.ln() - 161.119_568_166_1
    } else {
        288.122_169_528_3 * (temp - 60.0).powf(-0.075_514_849_2)
    };
    let blue = if temp >= 66.0 {
        255.0
    } else if temp <= 19.0 {
        0.0
    } else {
        138.517_731_223_1 * (temp - 10.0).ln() - 305.044_792_730_7
    };
    [
        (red / 255.0).clamp(0.0, 1.0),
        (green / 255.0).clamp(0.0, 1.0),
        (blue / 255.0).clamp(0.0, 1.0),
    ]
}

fn gaussian_spectrum_color(center: f64, width: f64) -> [f64; 3] {
    let w = width.max(1.0);
    let mut sum = [0.0; 3];
    let mut total = 0.0;
    let mut wl = 380.0;
    while wl <= 780.0 {
        let weight = (-0.5 * ((wl - center) / w).powi(2)).exp();
        let c = wavelength_color(wl);
        sum[0] += c[0] * weight;
        sum[1] += c[1] * weight;
        sum[2] += c[2] * weight;
        total += weight;
        wl += 5.0;
    }
    if total <= 0.0 {
        return [0.0; 3];
    }
    [
        (sum[0] / total).clamp(0.0, 1.0),
        (sum[1] / total).clamp(0.0, 1.0),
        (sum[2] / total).clamp(0.0, 1.0),
    ]
}

fn eval_ramp_cpu(stops: &[[f64; 5]], value: f64, mode: i32) -> [f64; 4] {
    let t = value.clamp(0.0, 1.0);
    let mut idx = 0usize;
    while idx + 1 < stops.len() && stops[idx + 1][0] <= t {
        idx += 1;
    }
    let a = stops[idx];
    let b = stops[(idx + 1).min(stops.len() - 1)];
    let span = b[0] - a[0];
    let mut f = if span.abs() > 1e-12 {
        ((t - a[0]) / span).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if mode == 0 {
        f = 0.0;
    } else if mode == 2 {
        f = f * f * (3.0 - 2.0 * f);
    }
    let mut out = [0.0; 4];
    for c in 0..4 {
        let x = a[c + 1];
        let y = b[c + 1];
        let v = if mode == 3 {
            let before = stops[idx.saturating_sub(1)][c + 1];
            let after = stops[(idx + 2).min(stops.len() - 1)][c + 1];
            0.5 * (2.0 * x
                + (-before + y) * f
                + (2.0 * before - 5.0 * x + 4.0 * y - after) * f * f
                + (-before + 3.0 * x - 3.0 * y + after) * f * f * f)
        } else {
            x + (y - x) * f
        };
        out[c] = v.clamp(0.0, 1.0);
    }
    out
}

fn coords_at(uv: Option<&MatBuf>, px_idx: usize, size: usize, x: usize, y: usize) -> (f64, f64) {
    if let Some(buf) = uv {
        let c = buf.sample_rgba(px_idx);
        (c[0], c[1])
    } else {
        ((x as f64) / (size as f64), (y as f64) / (size as f64))
    }
}

struct FractalParams {
    scale: f64,
    ratio: f64,
    octaves: usize,
    seed: i32,
    brightness: f64,
    contrast: f64,
    transform: u8,
}

/// Periodic multi-octave fractal lattice noise matching Zoidium `graph-buffers.js:fractalBuffer`.
fn eval_fractal_buffer(
    size: usize,
    fp: FractalParams,
    uv: Option<&MatBuf>,
    tracker: &Arc<BakeMemoryTracker>,
) -> Result<TrackedImageBuf, String> {
    let octs = fp.octaves.clamp(1, 8);
    let mut freq_x = fp.scale.round().max(1.0) as usize;
    let mut freq_y = (fp.scale * fp.ratio).round().max(1.0) as usize;
    let mut rng = Mulberry32::new(if fp.seed == 0 { 1 } else { fp.seed });
    let values_len = size
        .checked_mul(size)
        .ok_or_else(|| format!("NodeMaterial fractal size overflow for {size}x{size}"))?;
    let values_bytes = values_len
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| format!("NodeMaterial fractal byte overflow for {size}x{size}"))?;
    let _values_reservation = tracker.try_reserve(values_bytes)?;
    let mut values = vec![0.0f64; values_len];
    let mut amp = 1.0;
    let mut total = 0.0;

    for _ in 0..octs {
        let cells_x = freq_x.clamp(1, 1024);
        let cells_y = freq_y.clamp(1, 1024);
        let mut lattice = vec![0.0f64; cells_x * cells_y];
        for v in &mut lattice {
            *v = rng.next_f64();
        }
        for y in 0..size {
            for x in 0..size {
                let px = x + y * size;
                let (u, v) = coords_at(uv, px, size, x, y);
                let gx = u * (cells_x as f64);
                let gy = v * (cells_y as f64);
                let x0 = gx.floor() as i64;
                let y0 = gy.floor() as i64;
                let fx = gx - (x0 as f64);
                let fy = gy - (y0 as f64);
                let sx = fx * fx * (3.0 - 2.0 * fx);
                let sy = fy * fy * (3.0 - 2.0 * fy);
                let ix0 = x0.rem_euclid(cells_x as i64) as usize;
                let ix1 = (ix0 + 1) % cells_x;
                let iy0 = y0.rem_euclid(cells_y as i64) as usize;
                let iy1 = (iy0 + 1) % cells_y;
                let v00 = lattice[ix0 + iy0 * cells_x];
                let v10 = lattice[ix1 + iy0 * cells_x];
                let v01 = lattice[ix0 + iy1 * cells_x];
                let v11 = lattice[ix1 + iy1 * cells_x];
                let top = v00 + (v10 - v00) * sx;
                let bot = v01 + (v11 - v01) * sx;
                let mut val = top + (bot - top) * sy;
                if fp.transform == 1 {
                    val = 1.0 - (2.0 * val - 1.0).abs();
                } else if fp.transform == 2 {
                    val = (2.0 * val - 1.0).abs();
                }
                values[px] += val * amp;
            }
        }
        total += amp;
        amp *= 0.5;
        freq_x = freq_x.saturating_mul(2);
        freq_y = freq_y.saturating_mul(2);
    }

    let mut out = TrackedImageBuf::alloc_rgba(size, tracker)?;
    for (p, &val) in values.iter().enumerate() {
        let mut n = val / total;
        n = (n - 0.5) * fp.contrast + 0.5;
        n = (n * fp.brightness).clamp(0.0, 1.0);
        let f = n as f32;
        let i = p * 4;
        out.data[i] = f;
        out.data[i + 1] = f;
        out.data[i + 2] = f;
        out.data[i + 3] = 1.0;
    }
    Ok(out)
}

fn voronoi_distance(dx: f64, dy: f64, metric: i32, exponent: f64) -> f64 {
    let x = dx.abs();
    let y = dy.abs();
    match metric {
        1 => x + y,
        2 => x.max(y),
        3 if (exponent - 1.0).abs() < 1e-9 => x + y,
        3 if (exponent - 2.0).abs() > 1e-9 => {
            (x.powf(exponent) + y.powf(exponent)).powf(1.0 / exponent)
        }
        _ => (x * x + y * y).sqrt(),
    }
}

/// Evaluates Voronoi F1/F2/Smooth-F1/Distance-to-Edge matching Zoidium `graph-procedural.js`.
fn eval_voronoi_buffers(
    size: usize,
    params: &BTreeMap<String, Value>,
    uv: Option<&MatBuf>,
    tracker: &Arc<BakeMemoryTracker>,
) -> Result<(MatBuf, MatBuf, MatBuf), String> {
    let scale = param_f64(params, &["scale"], 5.0).clamp(0.01, 64.0);
    let randomness = param_f64(params, &["randomness"], 1.0).clamp(0.0, 1.0);
    let seed = param_f64(params, &["seed"], 1.0).round() as i32;
    let feature = if let Some(s) = param_str(params, &["feature", "mode"]) {
        match s.trim().to_ascii_lowercase().as_str() {
            "f2" => 1,
            "smooth f1" | "smoothf1" => 2,
            "distance to edge" | "edge" => 3,
            _ => 0,
        }
    } else {
        (param_f64(params, &["feature", "mode"], 0.0).round() as i32).clamp(0, 3)
    };
    let metric = (param_f64(params, &["metric", "distance"], 0.0).round() as i32).clamp(0, 3);
    let exponent = param_f64(params, &["exponent"], 2.0).max(0.1);
    let smoothness = param_f64(params, &["smoothness"], 0.5).max(0.0001);
    let squared = feature == 3
        || (feature != 2 && (metric == 0 || (metric == 3 && (exponent - 2.0).abs() < 1e-9)));

    let mut dist_out = TrackedImageBuf::alloc_rgba(size, tracker)?;
    let mut color_out = TrackedImageBuf::alloc_rgba(size, tracker)?;
    let mut pos_out = TrackedImageBuf::alloc_rgba(size, tracker)?;

    for y in 0..size {
        for x in 0..size {
            let p = x + y * size;
            let (uu, vv) = coords_at(uv, p, size, x, y);
            let u = uu * scale;
            let v = vv * scale;
            let cx = u.floor() as i32;
            let cy = v.floor() as i32;

            let mut first = f64::INFINITY;
            let mut second = f64::INFINITY;
            let mut smooth = f64::INFINITY;
            let mut near_gx = cx;
            let mut near_gy = cy;
            let mut near_px = u;
            let mut near_py = v;

            for dy in -2..=2 {
                for dx in -2..=2 {
                    let gx = cx + dx;
                    let gy = cy + dy;
                    let px = f64::from(gx) + 0.5 + (proc_hash(gx, gy, seed) - 0.5) * randomness;
                    let py = f64::from(gy)
                        + 0.5
                        + (proc_hash(gx, gy, seed.wrapping_add(173)) - 0.5) * randomness;
                    let ax = px - u;
                    let ay = py - v;
                    let d = if squared {
                        ax * ax + ay * ay
                    } else {
                        voronoi_distance(ax, ay, metric, exponent)
                    };
                    if d < first {
                        second = first;
                        first = d;
                        near_gx = gx;
                        near_gy = gy;
                        near_px = px;
                        near_py = py;
                    } else if d < second {
                        second = d;
                    }
                    if feature == 2 {
                        if !smooth.is_finite() {
                            smooth = d;
                        } else {
                            let h = (smoothness - (smooth - d).abs()).max(0.0) / smoothness;
                            smooth = smooth.min(d) - h * h * smoothness * 0.25;
                        }
                    }
                }
            }

            let mut value = match feature {
                1 => second,
                2 => smooth.max(0.0),
                _ => first,
            };
            if squared {
                value = value.max(0.0).sqrt();
            }
            if feature == 3 {
                let mut edge = f64::INFINITY;
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let gx = cx + dx;
                        let gy = cy + dy;
                        if gx == near_gx && gy == near_gy {
                            continue;
                        }
                        let px = f64::from(gx) + 0.5 + (proc_hash(gx, gy, seed) - 0.5) * randomness;
                        let py = f64::from(gy)
                            + 0.5
                            + (proc_hash(gx, gy, seed.wrapping_add(173)) - 0.5) * randomness;
                        let ax = px - near_px;
                        let ay = py - near_py;
                        let len = (ax * ax + ay * ay).sqrt();
                        if len > 1e-12 {
                            let d = ((px + near_px) * 0.5 - u) * ax / len
                                + ((py + near_py) * 0.5 - v) * ay / len;
                            edge = edge.min(d);
                        }
                    }
                }
                value = edge.max(0.0);
            }

            let idx = p * 4;
            let vf = value.clamp(0.0, 1.0) as f32;
            dist_out.data[idx] = vf;
            dist_out.data[idx + 1] = vf;
            dist_out.data[idx + 2] = vf;
            dist_out.data[idx + 3] = 1.0;

            color_out.data[idx] = proc_hash(near_gx, near_gy, seed.wrapping_add(331)) as f32;
            color_out.data[idx + 1] = proc_hash(near_gx, near_gy, seed.wrapping_add(468)) as f32;
            color_out.data[idx + 2] = proc_hash(near_gx, near_gy, seed.wrapping_add(605)) as f32;
            color_out.data[idx + 3] = 1.0;

            pos_out.data[idx] = (near_px / scale).clamp(0.0, 1.0) as f32;
            pos_out.data[idx + 1] = (near_py / scale).clamp(0.0, 1.0) as f32;
            pos_out.data[idx + 2] = 0.0;
            pos_out.data[idx + 3] = 1.0;
        }
    }

    Ok((
        MatBuf::from_image(dist_out),
        MatBuf::from_image(color_out),
        MatBuf::from_image(pos_out),
    ))
}

/// Converts a height map to a tangent-space normal map using a 3x3 Sobel filter
/// (wrapping seamlessly at tile boundaries).
fn height_to_normal_sobel(
    source: &MatBuf,
    size: usize,
    strength: f64,
    invert: bool,
    tracker: &Arc<BakeMemoryTracker>,
) -> Result<MatBuf, String> {
    let mut out = TrackedImageBuf::alloc_rgba(size, tracker)?;
    let s = strength.max(0.0) * if invert { -1.0 } else { 1.0 };
    let h_at = |x: isize, y: isize| -> f64 {
        let wx = x.rem_euclid(size as isize) as usize;
        let wy = y.rem_euclid(size as isize) as usize;
        source.sample_luma(wx + wy * size)
    };
    let scale = (size as f64) * 0.125 * s;
    for y in 0..size {
        for x in 0..size {
            let xi = x as isize;
            let yi = y as isize;
            let tl = h_at(xi - 1, yi - 1);
            let t = h_at(xi, yi - 1);
            let tr = h_at(xi + 1, yi - 1);
            let l = h_at(xi - 1, yi);
            let r = h_at(xi + 1, yi);
            let bl = h_at(xi - 1, yi + 1);
            let b = h_at(xi, yi + 1);
            let br = h_at(xi + 1, yi + 1);
            // Sobel horizontal & vertical gradients (tangent V runs upward while image row Y runs downward).
            let dx = (tl + 2.0 * l + bl) - (tr + 2.0 * r + br);
            let dy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
            let nx = dx * scale;
            let ny = dy * scale;
            let nz = 1.0;
            let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-12);
            let i = (x + y * size) * 4;
            out.data[i] = ((nx / len) * 0.5 + 0.5) as f32;
            out.data[i + 1] = ((ny / len) * 0.5 + 0.5) as f32;
            out.data[i + 2] = ((nz / len) * 0.5 + 0.5) as f32;
            out.data[i + 3] = 1.0;
        }
    }
    Ok(MatBuf::from_image(out))
}

fn normalize_tangent_normal(
    source: &MatBuf,
    size: usize,
    strength: f64,
    tracker: &Arc<BakeMemoryTracker>,
) -> Result<MatBuf, String> {
    let mut out = TrackedImageBuf::alloc_rgba(size, tracker)?;
    for p in 0..(size * size) {
        let c = source.sample_rgba(p);
        let nx = (c[0] * 2.0 - 1.0) * strength;
        let ny = (c[1] * 2.0 - 1.0) * strength;
        let nz = c[2] * 2.0 - 1.0;
        let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-12);
        let i = p * 4;
        out.data[i] = ((nx / len) * 0.5 + 0.5).clamp(0.0, 1.0) as f32;
        out.data[i + 1] = ((ny / len) * 0.5 + 0.5).clamp(0.0, 1.0) as f32;
        out.data[i + 2] = ((nz / len) * 0.5 + 0.5).clamp(0.0, 1.0) as f32;
        out.data[i + 3] = 1.0;
    }
    Ok(MatBuf::from_image(out))
}

fn parse_curve_points_channel(val: Option<&Value>) -> Vec<ZoidiumCurvePoint> {
    let mut pts = Vec::new();
    if let Some(Value::Array(arr)) = val {
        for item in arr {
            if let Some(a) = item.as_array()
                && a.len() >= 2
            {
                let x = a[0].as_f64().unwrap_or(0.0) as f32;
                let y = a[1].as_f64().unwrap_or(0.0) as f32;
                pts.push(ZoidiumCurvePoint::new(x, y));
            } else if let Some(o) = item.as_object() {
                let x = o.get("x").and_then(Value::as_f64).unwrap_or(0.0) as f32;
                let y = o.get("y").and_then(Value::as_f64).unwrap_or(0.0) as f32;
                pts.push(ZoidiumCurvePoint::new(x, y));
            }
        }
    }
    if pts.len() < 2 {
        vec![
            ZoidiumCurvePoint::new(0.0, 0.0),
            ZoidiumCurvePoint::new(1.0, 1.0),
        ]
    } else {
        pts
    }
}

fn normalize_material_node_type(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "output" | "pbroutput" | "pbr_output" | "materialoutput" => "output".to_string(),
        "noise" => "noise".to_string(),
        "voronoi" | "worley" => "voronoi".to_string(),
        "gradient" => "gradient".to_string(),
        "checker" | "checkerboard" => "checker".to_string(),
        "wave" => "wave".to_string(),
        "sinewave" | "sine_wave" => "sineWave".to_string(),
        "brick" => "brick".to_string(),
        "marble" => "marble".to_string(),
        "magic" => "magic".to_string(),
        "ridgedfractal" | "ridged_fractal" => "ridgedFractal".to_string(),
        "turbulence" => "turbulence".to_string(),
        "dirt" => "dirt".to_string(),
        "whitenoise" | "white_noise" => "whiteNoise".to_string(),
        "colorramp" | "color_ramp" | "ramp" => "colorRamp".to_string(),
        "colorizer" => "colorizer".to_string(),
        "maprange" | "map_range" | "remap" => "mapRange".to_string(),
        "rgbcurves" | "rgb_curves" | "curves" => "rgbCurves".to_string(),
        "separatergb" | "separate_rgb" => "separateRGB".to_string(),
        "combinergb" | "combine_rgb" => "combineRGB".to_string(),
        "separatehsv" | "separate_hsv" => "separateHSV".to_string(),
        "combinehsv" | "combine_hsv" => "combineHSV".to_string(),
        "colorcorrection" | "color_correction" => "colorCorrection".to_string(),
        "invert" => "invert".to_string(),
        "clamp" | "clamptexture" | "clamp_texture" => "clampTexture".to_string(),
        "mix" | "blend" => "mix".to_string(),
        "cosinemix" | "cosine_mix" => "cosineMix".to_string(),
        "multiply" => "multiply".to_string(),
        "add" => "add".to_string(),
        "subtract" => "subtract".to_string(),
        "compare" => "compare".to_string(),
        "math" => "math".to_string(),
        "legacymath" => "legacyMath".to_string(),
        "uv" | "projection" | "texcoord" => "projection".to_string(),
        "uvw" | "uvtransform" | "uv_transform" => "uvw".to_string(),
        "rgbspectrum" | "rgb_spectrum" => "rgbSpectrum".to_string(),
        "gaussianspectrum" | "gaussian_spectrum" => "gaussianSpectrum".to_string(),
        "blackbody" => "blackbody".to_string(),
        "color" | "solid" => "color".to_string(),
        "float" | "value" | "scalar" => "float".to_string(),
        "randomcolor" | "random_color" => "randomColor".to_string(),
        "bump" | "heighttonormal" | "height_to_normal" => "bump".to_string(),
        "fresnel" => "fresnel".to_string(),
        other => other.to_string(),
    }
}

struct MaterialEvaluator<'a> {
    size: usize,
    incoming: BTreeMap<(String, String), (String, String)>,
    outputs: BTreeMap<String, BTreeMap<String, MatBuf>>,
    tracker: Arc<BakeMemoryTracker>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> MaterialEvaluator<'a> {
    fn alloc_image(&self) -> Result<TrackedImageBuf, String> {
        TrackedImageBuf::alloc_rgba(self.size, &self.tracker)
    }

    fn get_input(&self, node_id: &str, aliases: &[&str]) -> Option<&MatBuf> {
        for &alias in aliases {
            if let Some((src_id, src_port)) =
                self.incoming.get(&(node_id.to_string(), alias.to_string()))
                && let Some(src_map) = self.outputs.get(src_id)
            {
                if let Some(buf) = src_map.get(src_port) {
                    return Some(buf);
                }
                if let Some(buf) = src_map
                    .get("out")
                    .or_else(|| src_map.get("color"))
                    .or_else(|| src_map.values().next())
                {
                    return Some(buf);
                }
            }
        }
        None
    }

    fn eval_node(&mut self, node: &RawNode) -> Result<(), String> {
        let size = self.size;
        let kind = normalize_material_node_type(&node.kind);
        let p = &node.params;
        let mut node_outs = BTreeMap::new();

        match kind.as_str() {
            "output" => {}
            "float" => {
                let v = if let Some(inp) = self.get_input(&node.id, &["value", "in"]) {
                    inp.clone()
                } else {
                    MatBuf::Scalar(param_f64(p, &["value"], 1.0))
                };
                node_outs.insert("out".to_string(), v);
            }
            "color" => {
                let buf = if let Some(inp) = self.get_input(&node.id, &["color", "in"]) {
                    inp.clone()
                } else {
                    let c = param_color3(p, &["color"], [0.8, 0.8, 0.8]);
                    MatBuf::Uniform([c[0], c[1], c[2], 1.0])
                };
                node_outs.insert("out".to_string(), buf.clone());
                node_outs.insert("color".to_string(), buf);
            }
            "randomColor" => {
                let seed = param_f64(p, &["seed"], 1.0).round() as i32;
                let boost = param_f64(p, &["brightness"], 1.0);
                let mut rng = Mulberry32::new(if seed == 0 { 1 } else { seed });
                let c = [
                    (rng.next_f64() * boost).clamp(0.0, 1.0),
                    (rng.next_f64() * boost).clamp(0.0, 1.0),
                    (rng.next_f64() * boost).clamp(0.0, 1.0),
                    1.0,
                ];
                node_outs.insert("out".to_string(), MatBuf::Uniform(c));
            }
            "noise" | "ridgedFractal" | "turbulence" | "dirt" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let default_scale = if kind == "noise" || kind == "dirt" {
                    4.0
                } else {
                    6.0
                };
                let scale = param_f64(p, &["scale"], default_scale);
                let ratio = param_f64(p, &["ratio"], 1.0);
                let octaves = param_f64(p, &["octaves"], 4.0).round().max(1.0) as usize;
                let seed = param_f64(p, &["seed"], 1.0).round() as i32;
                let brightness = param_f64(p, &["brightness"], 1.0);
                let contrast = if kind == "dirt" {
                    2.0
                } else {
                    param_f64(p, &["contrast"], 1.0)
                };
                let transform = match kind.as_str() {
                    "ridgedFractal" => 1,
                    "turbulence" | "dirt" => 2,
                    _ => 0,
                };
                let mut main_buf = eval_fractal_buffer(
                    size,
                    FractalParams {
                        scale,
                        ratio,
                        octaves,
                        seed,
                        brightness,
                        contrast,
                        transform,
                    },
                    uv.as_ref(),
                    &self.tracker,
                )?;
                if kind == "dirt" {
                    let radius = param_f64(p, &["radius"], 1.4);
                    for px in main_buf.data.as_chunks_mut::<4>().0 {
                        let v = (((1.0 - f64::from(px[0])) - 0.5) * radius + 0.5).clamp(0.0, 1.0)
                            as f32;
                        px[0] = v;
                        px[1] = v;
                        px[2] = v;
                    }
                }
                if kind == "noise" {
                    let mut col = self.alloc_image()?;
                    for ch in 0..3 {
                        let cbuf = eval_fractal_buffer(
                            size,
                            FractalParams {
                                scale,
                                ratio,
                                octaves,
                                seed: seed.wrapping_add(173 * (ch as i32) + 331),
                                brightness,
                                contrast,
                                transform: 0,
                            },
                            uv.as_ref(),
                            &self.tracker,
                        )?;
                        for i in 0..(size * size) {
                            col.data[i * 4 + ch] = cbuf.data[i * 4];
                            col.data[i * 4 + 3] = 1.0;
                        }
                    }
                    node_outs.insert("color".to_string(), MatBuf::from_image(col));
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(main_buf));
            }
            "voronoi" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let (dist, col, pos) = eval_voronoi_buffers(size, p, uv.as_ref(), &self.tracker)?;
                node_outs.insert("out".to_string(), dist.clone());
                node_outs.insert("distance".to_string(), dist);
                node_outs.insert("color".to_string(), col);
                node_outs.insert("position".to_string(), pos);
            }
            "gradient" => {
                let pos_in = self.get_input(&node.id, &["t", "factor", "in"]).cloned();
                let base_in = self.get_input(&node.id, &["a", "base"]).cloned();
                let grad_type = (param_f64(p, &["type", "mode"], 0.0).round() as i32).clamp(0, 3);
                let ox = param_f64(p, &["offsetX"], 0.0);
                let oy = param_f64(p, &["offsetY"], 0.0);
                let sx = param_f64(p, &["scaleX"], 1.0).max(0.001);
                let sy = param_f64(p, &["scaleY"], 1.0).max(0.001);
                let rot = -param_f64(p, &["rotation"], 0.0) * PI / 180.0;
                let (sin_a, cos_a) = rot.sin_cos();
                let midpoint = param_f64(p, &["midpoint"], 0.5).clamp(0.001, 0.999);
                let opacity = param_f64(p, &["opacity"], 1.0).clamp(0.0, 1.0);
                let blend_mode = resolve_blend_mode(p, &["blendMode"]);
                let c1 = param_color3(p, &["color1"], [0.1, 0.1, 0.12]);
                let c2 = param_color3(p, &["color2"], [1.0, 1.0, 1.0]);
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let t = if let Some(ref pi) = pos_in {
                            pi.sample_luma(px).clamp(0.0, 1.0)
                        } else {
                            let cu = ((x as f64) / (size as f64) - 0.5 - ox) / sx;
                            let cv = ((y as f64) / (size as f64) - 0.5 - oy) / sy;
                            let ru = cu * cos_a - cv * sin_a;
                            let rv = cu * sin_a + cv * cos_a;
                            match grad_type {
                                1 => ((ru * ru + rv * rv).sqrt() * 2.0).min(1.0),
                                2 => rv.atan2(ru) / TAU + 0.5,
                                3 => ((ru.abs() + rv.abs()) * 2.0).min(1.0),
                                _ => ru + 0.5,
                            }
                            .clamp(0.0, 1.0)
                        };
                        let ramp = if t < midpoint {
                            0.5 * (t / midpoint)
                        } else {
                            0.5 + 0.5 * ((t - midpoint) / (1.0 - midpoint))
                        };
                        let cr = [
                            c1[0] + (c2[0] - c1[0]) * ramp,
                            c1[1] + (c2[1] - c1[1]) * ramp,
                            c1[2] + (c2[2] - c1[2]) * ramp,
                        ];
                        let rgb = if let Some(ref bi) = base_in {
                            let b = bi.sample_rgba(px);
                            mat_blend_rgb([b[0], b[1], b[2]], cr, opacity, blend_mode)
                        } else {
                            cr
                        };
                        let i = px * 4;
                        out.data[i] = rgb[0].clamp(0.0, 1.0) as f32;
                        out.data[i + 1] = rgb[1].clamp(0.0, 1.0) as f32;
                        out.data[i + 2] = rgb[2].clamp(0.0, 1.0) as f32;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "checker" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let scale = param_f64(p, &["scale"], 8.0).round().max(1.0);
                let c1 = param_color3(p, &["color1", "a"], [0.0, 0.0, 0.0]);
                let c2 = param_color3(p, &["color2", "b"], [1.0, 1.0, 1.0]);
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (u, v) = coords_at(uv.as_ref(), px, size, x, y);
                        let parity =
                            ((u * scale).floor() as i64 + (v * scale).floor() as i64).rem_euclid(2);
                        let c = if parity == 0 { c1 } else { c2 };
                        let i = px * 4;
                        out.data[i] = c[0] as f32;
                        out.data[i + 1] = c[1] as f32;
                        out.data[i + 2] = c[2] as f32;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "wave" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let wave_type = (param_f64(p, &["type"], 0.0).round() as i32).clamp(0, 1);
                let direction = (param_f64(p, &["direction"], 0.0).round() as i32).clamp(0, 2);
                let profile = (param_f64(p, &["profile"], 0.0).round() as i32).clamp(0, 2);
                let scale = param_f64(p, &["scale"], 5.0).max(0.01);
                let distortion = param_f64(p, &["distortion"], 0.0);
                let detail = param_f64(p, &["detail"], 2.0).round().max(1.0) as usize;
                let seed = param_f64(p, &["seed"], 1.0).round() as i32;
                let dist_buf = if distortion.abs() > 1e-9 {
                    Some(eval_fractal_buffer(
                        size,
                        FractalParams {
                            scale,
                            ratio: 1.0,
                            octaves: detail,
                            seed,
                            brightness: 1.0,
                            contrast: 1.0,
                            transform: 0,
                        },
                        uv.as_ref(),
                        &self.tracker,
                    )?)
                } else {
                    None
                };
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (uu, vv) = coords_at(uv.as_ref(), px, size, x, y);
                        let u = uu * scale;
                        let v = vv * scale;
                        let mut phase = if wave_type == 1 {
                            (u - scale * 0.5).hypot(v - scale * 0.5)
                        } else if direction == 1 {
                            v
                        } else if direction == 2 {
                            (u + v) / SQRT_2
                        } else {
                            u
                        };
                        if let Some(ref db) = dist_buf {
                            phase += (f64::from(db.data[px * 4]) - 0.5) * distortion;
                        }
                        let t = phase - phase.floor();
                        let val = match profile {
                            1 => t,
                            2 => 1.0 - (2.0 * t - 1.0).abs(),
                            _ => 0.5 - 0.5 * (TAU * phase).cos(),
                        }
                        .clamp(0.0, 1.0) as f32;
                        let i = px * 4;
                        out.data[i] = val;
                        out.data[i + 1] = val;
                        out.data[i + 2] = val;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "sineWave" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let scale = param_f64(p, &["scale"], 6.0);
                let phase = param_f64(p, &["phase"], 0.0);
                let axis = (param_f64(p, &["axis"], 0.0).round() as i32).clamp(0, 1);
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (u, v) = coords_at(uv.as_ref(), px, size, x, y);
                        let coord = if axis == 1 { v } else { u };
                        let val = (0.5 + 0.5 * ((coord * scale + phase) * TAU).sin())
                            .clamp(0.0, 1.0) as f32;
                        let i = px * 4;
                        out.data[i] = val;
                        out.data[i + 1] = val;
                        out.data[i + 2] = val;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "brick" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let scale = param_f64(p, &["scale"], 5.0).max(0.01);
                let width = param_f64(p, &["width"], 1.0).max(0.01);
                let height = param_f64(p, &["height"], 0.5).max(0.01);
                let mortar_size = param_f64(p, &["mortar"], 0.04).clamp(0.0, 0.5);
                let offset = param_f64(p, &["offset"], 0.5);
                let seed = param_f64(p, &["seed"], 1.0).round() as i32;
                let c1 = param_color3(p, &["color1"], [0.6, 0.17, 0.08]);
                let c2 = param_color3(p, &["color2"], [0.3, 0.06, 0.03]);
                let mc = param_color3(p, &["mortarColor"], [0.08, 0.08, 0.08]);
                let mut col_out = self.alloc_image()?;
                let mut mortar_out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (uu, vv) = coords_at(uv.as_ref(), px, size, x, y);
                        let u = uu * scale;
                        let v = vv * scale;
                        let row = (v / height).floor() as i32;
                        let bx = u / width + if row.rem_euclid(2) != 0 { offset } else { 0.0 };
                        let col = bx.floor() as i32;
                        let fx = (bx - bx.floor()) * width;
                        let fy = ((v / height) - (v / height).floor()) * height;
                        let is_mortar = fx.min(width - fx).min(fy).min(height - fy) < mortar_size;
                        let t = proc_hash(col, row, seed);
                        let i = px * 4;
                        let mv = if is_mortar { 1.0f32 } else { 0.0f32 };
                        mortar_out.data[i] = mv;
                        mortar_out.data[i + 1] = mv;
                        mortar_out.data[i + 2] = mv;
                        mortar_out.data[i + 3] = 1.0;
                        for ch in 0..3 {
                            col_out.data[i + ch] = if is_mortar {
                                mc[ch] as f32
                            } else {
                                (c1[ch] + (c2[ch] - c1[ch]) * t) as f32
                            };
                        }
                        col_out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(col_out));
                node_outs.insert("mortar".to_string(), MatBuf::from_image(mortar_out));
            }
            "marble" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let scale = param_f64(p, &["scale"], 5.0);
                let veins = param_f64(p, &["veins"], 3.0);
                let fscale = param_f64(p, &["fractalScale"], 6.0);
                let octaves = param_f64(p, &["octaves"], 4.0).round().max(1.0) as usize;
                let seed = param_f64(p, &["seed"], 1.0).round() as i32;
                let fractal = eval_fractal_buffer(
                    size,
                    FractalParams {
                        scale: fscale,
                        ratio: 1.0,
                        octaves,
                        seed,
                        brightness: 1.0,
                        contrast: 1.0,
                        transform: 0,
                    },
                    uv.as_ref(),
                    &self.tracker,
                )?;
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (_, v) = coords_at(uv.as_ref(), px, size, x, y);
                        let coord = v * scale + f64::from(fractal.data[px * 4]) * veins;
                        let val = (0.5 + 0.5 * (coord * TAU).sin()).clamp(0.0, 1.0) as f32;
                        let i = px * 4;
                        out.data[i] = val;
                        out.data[i + 1] = val;
                        out.data[i + 2] = val;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "magic" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let scale = param_f64(p, &["scale"], 5.0).max(0.01);
                let depth = (param_f64(p, &["depth"], 2.0).round() as usize).clamp(0, 10);
                let dist = param_f64(p, &["distortion"], 1.0);
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (uu, vv) = coords_at(uv.as_ref(), px, size, x, y);
                        let u = uu * scale;
                        let v = vv * scale;
                        let mut a = (u + v).sin();
                        let mut c = (u - v).cos();
                        let mut d = (-u + v).sin();
                        for _ in 0..depth {
                            a = ((a + c + d) * dist).sin();
                            c = ((-a + c - d) * dist).cos();
                            d = ((-a - c + d) * dist).sin();
                        }
                        let i = px * 4;
                        out.data[i] = (a * 0.5 + 0.5) as f32;
                        out.data[i + 1] = (c * 0.5 + 0.5) as f32;
                        out.data[i + 2] = (d * 0.5 + 0.5) as f32;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "whiteNoise" => {
                let uv = self.get_input(&node.id, &["uv"]).cloned();
                let scale = param_f64(p, &["scale"], 256.0).max(1.0);
                let seed = param_f64(p, &["seed"], 1.0).round() as i32;
                let mut val_out = self.alloc_image()?;
                let mut col_out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let px = x + y * size;
                        let (uu, vv) = coords_at(uv.as_ref(), px, size, x, y);
                        let ix = (uu * scale).floor() as i32;
                        let iy = (vv * scale).floor() as i32;
                        let v0 = proc_hash(ix, iy, seed) as f32;
                        let i = px * 4;
                        val_out.data[i] = v0;
                        val_out.data[i + 1] = v0;
                        val_out.data[i + 2] = v0;
                        val_out.data[i + 3] = 1.0;
                        col_out.data[i] = v0;
                        col_out.data[i + 1] = proc_hash(ix, iy, seed.wrapping_add(137)) as f32;
                        col_out.data[i + 2] = proc_hash(ix, iy, seed.wrapping_add(274)) as f32;
                        col_out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(val_out));
                node_outs.insert("color".to_string(), MatBuf::from_image(col_out));
            }
            "colorRamp" => {
                let fac_in = self.get_input(&node.id, &["factor", "t", "in"]).cloned();
                let stops = parse_ramp_stops(p);
                let interp = (param_f64(p, &["interpolation"], 1.0).round() as i32).clamp(0, 3);
                let default_fac = param_f64(p, &["factor"], 0.5);
                let mut col_out = self.alloc_image()?;
                let mut alp_out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let f = fac_in.as_ref().map_or(default_fac, |b| b.sample_luma(px));
                    let rgba = eval_ramp_cpu(&stops, f, interp);
                    let i = px * 4;
                    col_out.data[i] = rgba[0] as f32;
                    col_out.data[i + 1] = rgba[1] as f32;
                    col_out.data[i + 2] = rgba[2] as f32;
                    col_out.data[i + 3] = rgba[3] as f32;
                    let af = rgba[3] as f32;
                    alp_out.data[i] = af;
                    alp_out.data[i + 1] = af;
                    alp_out.data[i + 2] = af;
                    alp_out.data[i + 3] = 1.0;
                }
                let col_buf = MatBuf::from_image(col_out);
                node_outs.insert("out".to_string(), col_buf.clone());
                node_outs.insert("color".to_string(), col_buf);
                node_outs.insert("alpha".to_string(), MatBuf::from_image(alp_out));
            }
            "colorizer" => {
                let src = self.get_input(&node.id, &["a", "in"]).cloned();
                let cols = [
                    param_color3(p, &["color1"], [0.0, 0.0, 0.0]),
                    param_color3(p, &["color2"], [0.25, 0.25, 0.3]),
                    param_color3(p, &["color3"], [0.7, 0.7, 0.75]),
                    param_color3(p, &["color4"], [1.0, 1.0, 1.0]),
                    param_color3(p, &["color5"], [1.0, 1.0, 1.0]),
                ];
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let t = src
                        .as_ref()
                        .map_or(0.5, |b| b.sample_luma(px))
                        .clamp(0.0, 1.0);
                    let scaled = t * 4.0;
                    let seg = (scaled.floor() as usize).min(3);
                    let local = scaled - (seg as f64);
                    let c0 = cols[seg];
                    let c1 = cols[seg + 1];
                    let i = px * 4;
                    out.data[i] = (c0[0] + (c1[0] - c0[0]) * local) as f32;
                    out.data[i + 1] = (c0[1] + (c1[1] - c0[1]) * local) as f32;
                    out.data[i + 2] = (c0[2] + (c1[2] - c0[2]) * local) as f32;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "mapRange" => {
                let val_in = self.get_input(&node.id, &["value", "in"]).cloned();
                let def_val = param_f64(p, &["value"], 0.5);
                let from_min = param_f64(p, &["fromMin"], 0.0);
                let from_max = param_f64(p, &["fromMax"], 1.0);
                let to_min = param_f64(p, &["toMin"], 0.0);
                let to_max = param_f64(p, &["toMax"], 1.0);
                let interp = (param_f64(p, &["interpolation"], 0.0).round() as i32).clamp(0, 3);
                let steps = param_f64(p, &["steps"], 4.0).floor().max(1.0);
                let do_clamp = param_f64(p, &["clamp"], 1.0) != 0.0;
                let denom = if (from_max - from_min).abs() > 1e-12 {
                    from_max - from_min
                } else {
                    1e-12
                };
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let v = val_in.as_ref().map_or(def_val, |b| b.sample_luma(px));
                    let mut t = (v - from_min) / denom;
                    if do_clamp || interp >= 2 {
                        t = t.clamp(0.0, 1.0);
                    }
                    if interp == 1 {
                        t = (t * (steps + 1.0)).floor() / steps;
                        if do_clamp {
                            t = t.min(1.0);
                        }
                    } else if interp == 2 {
                        t = t * t * (3.0 - 2.0 * t);
                    } else if interp == 3 {
                        t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
                    }
                    let mapped = (to_min + t * (to_max - to_min)) as f32;
                    let i = px * 4;
                    out.data[i] = mapped;
                    out.data[i + 1] = mapped;
                    out.data[i + 2] = mapped;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "rgbCurves" => {
                let src = self.get_input(&node.id, &["color", "a", "in"]).cloned();
                let fac = param_f64(p, &["factor"], 1.0).clamp(0.0, 1.0);
                let curves_val = p.get("curves");
                let master = parse_curve_points_channel(curves_val.and_then(|v| {
                    v.get("c")
                        .or_else(|| v.get("master"))
                        .or_else(|| v.get("rgb"))
                }));
                let red = parse_curve_points_channel(curves_val.and_then(|v| v.get("r")));
                let green = parse_curve_points_channel(curves_val.and_then(|v| v.get("g")));
                let blue = parse_curve_points_channel(curves_val.and_then(|v| v.get("b")));
                let mut lut_r = [0.0f32; 256];
                let mut lut_g = [0.0f32; 256];
                let mut lut_b = [0.0f32; 256];
                for i in 0..256 {
                    let x = (i as f32) / 255.0;
                    let m = evaluate_zoidium_curve_points(&master, x);
                    lut_r[i] = evaluate_zoidium_curve_points(&red, m);
                    lut_g[i] = evaluate_zoidium_curve_points(&green, m);
                    lut_b[i] = evaluate_zoidium_curve_points(&blue, m);
                }
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let c = src
                        .as_ref()
                        .map_or([0.0, 0.0, 0.0, 1.0], |b| b.sample_rgba(px));
                    let ir = (c[0].clamp(0.0, 1.0) * 255.0).round() as usize;
                    let ig = (c[1].clamp(0.0, 1.0) * 255.0).round() as usize;
                    let ib = (c[2].clamp(0.0, 1.0) * 255.0).round() as usize;
                    let i = px * 4;
                    out.data[i] = (c[0] + (f64::from(lut_r[ir]) - c[0]) * fac) as f32;
                    out.data[i + 1] = (c[1] + (f64::from(lut_g[ig]) - c[1]) * fac) as f32;
                    out.data[i + 2] = (c[2] + (f64::from(lut_b[ib]) - c[2]) * fac) as f32;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "separateRGB" | "separateHSV" => {
                let src = self.get_input(&node.id, &["color", "in"]).cloned();
                let is_hsv = kind == "separateHSV";
                let keys = if is_hsv {
                    ["h", "s", "v"]
                } else {
                    ["r", "g", "b"]
                };
                let mut chans = [
                    self.alloc_image()?,
                    self.alloc_image()?,
                    self.alloc_image()?,
                ];
                for px in 0..(size * size) {
                    let rgba = src
                        .as_ref()
                        .map_or([0.0, 0.0, 0.0, 1.0], |b| b.sample_rgba(px));
                    let vals = if is_hsv {
                        rgb_to_hsv(rgba[0], rgba[1], rgba[2])
                    } else {
                        [rgba[0], rgba[1], rgba[2]]
                    };
                    let i = px * 4;
                    for c in 0..3 {
                        let f = vals[c] as f32;
                        chans[c].data[i] = f;
                        chans[c].data[i + 1] = f;
                        chans[c].data[i + 2] = f;
                        chans[c].data[i + 3] = 1.0;
                    }
                }
                for (k, buf) in keys.into_iter().zip(chans) {
                    node_outs.insert(k.to_string(), MatBuf::from_image(buf));
                }
                if let Some(first) = node_outs.get(keys[0]).cloned() {
                    node_outs.insert("out".to_string(), first);
                }
            }
            "combineRGB" | "combineHSV" => {
                let is_hsv = kind == "combineHSV";
                let (k0, k1, k2) = if is_hsv {
                    ("h", "s", "v")
                } else {
                    ("r", "g", "b")
                };
                let b0 = self.get_input(&node.id, &[k0]).cloned();
                let b1 = self.get_input(&node.id, &[k1]).cloned();
                let b2 = self.get_input(&node.id, &[k2]).cloned();
                let d0 = param_f64(p, &[k0], 0.0);
                let d1 = param_f64(p, &[k1], 0.0);
                let d2 = param_f64(p, &[k2], 0.0);
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let v0 = b0.as_ref().map_or(d0, |b| b.sample_luma(px));
                    let v1 = b1.as_ref().map_or(d1, |b| b.sample_luma(px));
                    let v2 = b2.as_ref().map_or(d2, |b| b.sample_luma(px));
                    let rgb = if is_hsv {
                        hsv_to_rgb(v0, v1, v2)
                    } else {
                        [v0, v1, v2]
                    };
                    let i = px * 4;
                    out.data[i] = rgb[0].clamp(0.0, 1.0) as f32;
                    out.data[i + 1] = rgb[1].clamp(0.0, 1.0) as f32;
                    out.data[i + 2] = rgb[2].clamp(0.0, 1.0) as f32;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "colorCorrection" => {
                let src = self.get_input(&node.id, &["a", "color", "in"]).cloned();
                let brightness = param_f64(p, &["brightness"], 1.0);
                let contrast = param_f64(p, &["contrast"], 1.0);
                let gamma = param_f64(p, &["gamma"], 1.0).max(0.001);
                let hue = param_f64(p, &["hue"], 0.0) / 360.0;
                let saturation = param_f64(p, &["saturation"], 1.0);
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let c = src
                        .as_ref()
                        .map_or([1.0, 1.0, 1.0, 1.0], |b| b.sample_rgba(px));
                    let mut rgb = [0.0; 3];
                    for ch in 0..3 {
                        let mut v = ((c[ch] - 0.5) * contrast + 0.5).clamp(0.0, 1.0);
                        if (gamma - 1.0).abs() > 1e-9 {
                            v = v.powf(1.0 / gamma);
                        }
                        rgb[ch] = (v * brightness).clamp(0.0, 1.0);
                    }
                    if hue.abs() > 1e-9 || (saturation - 1.0).abs() > 1e-9 {
                        let hsv = rgb_to_hsv(rgb[0], rgb[1], rgb[2]);
                        let mut sh = (hsv[0] + hue) % 1.0;
                        if sh < 0.0 {
                            sh += 1.0;
                        }
                        rgb = hsv_to_rgb(sh, (hsv[1] * saturation).clamp(0.0, 1.0), hsv[2]);
                    }
                    let i = px * 4;
                    out.data[i] = rgb[0] as f32;
                    out.data[i + 1] = rgb[1] as f32;
                    out.data[i + 2] = rgb[2] as f32;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "invert" => {
                let src = self.get_input(&node.id, &["a", "color", "in"]).cloned();
                let strength = param_f64(p, &["strength", "factor"], 1.0).clamp(0.0, 1.0);
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let c = src
                        .as_ref()
                        .map_or([0.5, 0.5, 0.5, 1.0], |b| b.sample_rgba(px));
                    let i = px * 4;
                    for (ch, &val) in c.iter().enumerate().take(3) {
                        out.data[i + ch] =
                            (val + (1.0 - 2.0 * val) * strength).clamp(0.0, 1.0) as f32;
                    }
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "clampTexture" => {
                let src = self.get_input(&node.id, &["a", "in", "value"]).cloned();
                let lo = param_f64(p, &["min"], 0.0);
                let hi = param_f64(p, &["max"], 1.0);
                let (min_v, max_v) = (lo.min(hi), lo.max(hi));
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let c = src
                        .as_ref()
                        .map_or([0.5, 0.5, 0.5, 1.0], |b| b.sample_rgba(px));
                    let i = px * 4;
                    out.data[i] = c[0].clamp(min_v, max_v) as f32;
                    out.data[i + 1] = c[1].clamp(min_v, max_v) as f32;
                    out.data[i + 2] = c[2].clamp(min_v, max_v) as f32;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "mix" => {
                let a = self.get_input(&node.id, &["a", "base"]).cloned();
                let b = self.get_input(&node.id, &["b", "top"]).cloned();
                let op_in = self.get_input(&node.id, &["opacity", "factor"]).cloned();
                let mode = resolve_blend_mode(p, &["mode", "blendMode"]);
                let def_op = param_f64(p, &["opacity", "factor"], 1.0).clamp(0.0, 1.0);
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let ca = a
                        .as_ref()
                        .map_or([0.0, 0.0, 0.0, 1.0], |v| v.sample_rgba(px));
                    let cb = b.as_ref().map_or(ca, |v| v.sample_rgba(px));
                    let op = op_in.as_ref().map_or(def_op, |v| v.sample_luma(px));
                    let rgb = mat_blend_rgb([ca[0], ca[1], ca[2]], [cb[0], cb[1], cb[2]], op, mode);
                    let i = px * 4;
                    out.data[i] = rgb[0].clamp(0.0, 1.0) as f32;
                    out.data[i + 1] = rgb[1].clamp(0.0, 1.0) as f32;
                    out.data[i + 2] = rgb[2].clamp(0.0, 1.0) as f32;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "cosineMix" => {
                let a = self.get_input(&node.id, &["a"]).cloned();
                let b = self.get_input(&node.id, &["b"]).cloned();
                let m = param_f64(p, &["mix", "factor"], 0.5).clamp(0.0, 1.0);
                let t = 0.5 - 0.5 * (PI * m).cos();
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let ca = a
                        .as_ref()
                        .map_or([0.0, 0.0, 0.0, 1.0], |v| v.sample_rgba(px));
                    let cb = b.as_ref().map_or(ca, |v| v.sample_rgba(px));
                    let i = px * 4;
                    for ch in 0..3 {
                        out.data[i + ch] = (ca[ch] + (cb[ch] - ca[ch]) * t) as f32;
                    }
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "multiply" | "add" | "subtract" => {
                let a = self.get_input(&node.id, &["a"]).cloned();
                let b = self.get_input(&node.id, &["b"]).cloned();
                let def_b = param_f64(
                    p,
                    &["value", "b"],
                    if kind == "multiply" { 1.0 } else { 0.0 },
                );
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let va = a.as_ref().map_or(0.0, |v| v.sample_luma(px));
                    let vb = b.as_ref().map_or(def_b, |v| v.sample_luma(px));
                    let res = match kind.as_str() {
                        "add" => va + vb,
                        "subtract" => va - vb,
                        _ => va * vb,
                    }
                    .clamp(0.0, 1.0) as f32;
                    let i = px * 4;
                    out.data[i] = res;
                    out.data[i + 1] = res;
                    out.data[i + 2] = res;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "compare" => {
                let a = self.get_input(&node.id, &["a"]).cloned();
                let b = self.get_input(&node.id, &["b"]).cloned();
                let op = (param_f64(p, &["op"], 0.0).round() as i32).clamp(0, 2);
                let def_b = param_f64(p, &["value"], 0.5);
                let eps = param_f64(p, &["epsilon"], 0.01);
                let mut out = self.alloc_image()?;
                for px in 0..(size * size) {
                    let va = a.as_ref().map_or(0.0, |v| v.sample_luma(px));
                    let vb = b.as_ref().map_or(def_b, |v| v.sample_luma(px));
                    let hit = match op {
                        1 => va < vb,
                        2 => (va - vb).abs() <= eps,
                        _ => va > vb,
                    };
                    let f = if hit { 1.0f32 } else { 0.0f32 };
                    let i = px * 4;
                    out.data[i] = f;
                    out.data[i + 1] = f;
                    out.data[i + 2] = f;
                    out.data[i + 3] = 1.0;
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "math" | "legacyMath" => {
                let a = self.get_input(&node.id, &["a"]).cloned();
                let b = self.get_input(&node.id, &["b"]).cloned();
                let c = self.get_input(&node.id, &["c"]).cloned();
                let op = resolve_math_op(
                    p,
                    &["op", "mode", "operation"],
                    if kind == "legacyMath" { 2 } else { 0 },
                );
                let def_a = param_f64(p, &["a"], 0.0);
                let def_b = param_f64(p, &["b", "value"], 1.0);
                let def_c = param_f64(p, &["c"], 0.5);
                if a.as_ref().is_none_or(|v| !v.is_spatial())
                    && b.as_ref().is_none_or(|v| !v.is_spatial())
                    && c.as_ref().is_none_or(|v| !v.is_spatial())
                {
                    let va = a.as_ref().map_or(def_a, |v| v.sample_luma(0));
                    let vb = b.as_ref().map_or(def_b, |v| v.sample_luma(0));
                    let vc = c.as_ref().map_or(def_c, |v| v.sample_luma(0));
                    node_outs.insert(
                        "out".to_string(),
                        MatBuf::Scalar(mat_math_op(va, vb, vc, op)),
                    );
                } else {
                    let mut out = self.alloc_image()?;
                    for px in 0..(size * size) {
                        let va = a.as_ref().map_or(def_a, |v| v.sample_luma(px));
                        let vb = b.as_ref().map_or(def_b, |v| v.sample_luma(px));
                        let vc = c.as_ref().map_or(def_c, |v| v.sample_luma(px));
                        let res = mat_math_op(va, vb, vc, op) as f32;
                        let i = px * 4;
                        out.data[i] = res;
                        out.data[i + 1] = res;
                        out.data[i + 2] = res;
                        out.data[i + 3] = 1.0;
                    }
                    node_outs.insert("out".to_string(), MatBuf::from_image(out));
                }
            }
            "projection" => {
                let mode = (param_f64(p, &["mode", "projection"], 0.0).round() as i32).clamp(0, 3);
                let ox = param_f64(p, &["offsetX"], 0.0);
                let oy = param_f64(p, &["offsetY"], 0.0);
                let sx = param_f64(p, &["scaleX"], 1.0).max(0.001);
                let sy = param_f64(p, &["scaleY"], 1.0).max(0.001);
                let rot = -param_f64(p, &["rotation"], 0.0) * PI / 180.0;
                let (sin_a, cos_a) = rot.sin_cos();
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let u = ((x as f64) / (size as f64) - 0.5 - ox) / sx;
                        let v = ((y as f64) / (size as f64) - 0.5 - oy) / sy;
                        let ru = u * cos_a - v * sin_a;
                        let rv = u * sin_a + v * cos_a;
                        let mut ou = if mode >= 1 {
                            rv.atan2(ru) / TAU + 0.5
                        } else {
                            ru + 0.5
                        };
                        let mut ov = match mode {
                            1 => ((ru * ru + rv * rv).sqrt() * 2.0).min(1.0),
                            3 => {
                                ((ru * ru + rv * rv).sqrt() * 2.0).clamp(0.0, 1.0).asin()
                                    / (PI * 0.5)
                            }
                            _ => rv + 0.5,
                        };
                        ou -= ou.floor();
                        ov -= ov.floor();
                        let i = (x + y * size) * 4;
                        out.data[i] = ou as f32;
                        out.data[i + 1] = ov as f32;
                        out.data[i + 2] = 0.0;
                        out.data[i + 3] = 1.0;
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "uvw" => {
                let src = self.get_input(&node.id, &["a", "uv", "in"]).cloned();
                let ox = param_f64(p, &["offsetX"], 0.0);
                let oy = param_f64(p, &["offsetY"], 0.0);
                let sx = param_f64(p, &["scaleX"], 1.0).max(0.001);
                let sy = param_f64(p, &["scaleY"], 1.0).max(0.001);
                let rot = -param_f64(p, &["rotation"], 0.0) * PI / 180.0;
                let (sin_a, cos_a) = rot.sin_cos();
                let mut out = self.alloc_image()?;
                for y in 0..size {
                    for x in 0..size {
                        let u = ((x as f64) / (size as f64) - 0.5 - ox) / sx;
                        let v = ((y as f64) / (size as f64) - 0.5 - oy) / sy;
                        let mut ru = u * cos_a - v * sin_a + 0.5;
                        let mut rv = u * sin_a + v * cos_a + 0.5;
                        ru -= ru.floor();
                        rv -= rv.floor();
                        let i = (x + y * size) * 4;
                        if let Some(ref s) = src {
                            let sx_px = ((ru * (size as f64)).floor() as usize).min(size - 1);
                            let sy_px = ((rv * (size as f64)).floor() as usize).min(size - 1);
                            let c = s.sample_rgba(sx_px + sy_px * size);
                            out.data[i] = c[0] as f32;
                            out.data[i + 1] = c[1] as f32;
                            out.data[i + 2] = c[2] as f32;
                            out.data[i + 3] = 1.0;
                        } else {
                            out.data[i] = ru as f32;
                            out.data[i + 1] = rv as f32;
                            out.data[i + 2] = 0.0;
                            out.data[i + 3] = 1.0;
                        }
                    }
                }
                node_outs.insert("out".to_string(), MatBuf::from_image(out));
            }
            "rgbSpectrum" => {
                let inp = self.get_input(&node.id, &["t", "in"]).cloned();
                let wl = param_f64(p, &["wavelength"], 550.0);
                if let Some(ref src) = inp {
                    let mut out = self.alloc_image()?;
                    for px in 0..(size * size) {
                        let w =
                            (380.0 + src.sample_luma(px) * 400.0 + wl - 550.0).clamp(380.0, 780.0);
                        let c = wavelength_color(w);
                        let i = px * 4;
                        out.data[i] = c[0] as f32;
                        out.data[i + 1] = c[1] as f32;
                        out.data[i + 2] = c[2] as f32;
                        out.data[i + 3] = 1.0;
                    }
                    node_outs.insert("out".to_string(), MatBuf::from_image(out));
                } else {
                    let c = wavelength_color(wl.clamp(380.0, 780.0));
                    node_outs.insert("out".to_string(), MatBuf::Uniform([c[0], c[1], c[2], 1.0]));
                }
            }
            "gaussianSpectrum" => {
                let inp = self.get_input(&node.id, &["t", "in"]).cloned();
                let center = param_f64(p, &["center"], 550.0);
                let width = param_f64(p, &["width"], 60.0);
                if let Some(ref src) = inp {
                    let mut out = self.alloc_image()?;
                    for px in 0..(size * size) {
                        let ct = (380.0 + src.sample_luma(px) * 400.0 + center - 550.0)
                            .clamp(380.0, 780.0);
                        let c = gaussian_spectrum_color(ct, width);
                        let i = px * 4;
                        out.data[i] = c[0] as f32;
                        out.data[i + 1] = c[1] as f32;
                        out.data[i + 2] = c[2] as f32;
                        out.data[i + 3] = 1.0;
                    }
                    node_outs.insert("out".to_string(), MatBuf::from_image(out));
                } else {
                    let c = gaussian_spectrum_color(center.clamp(380.0, 780.0), width);
                    node_outs.insert("out".to_string(), MatBuf::Uniform([c[0], c[1], c[2], 1.0]));
                }
            }
            "blackbody" => {
                let inp = self.get_input(&node.id, &["t", "in"]).cloned();
                let temp = param_f64(p, &["temperature", "kelvin"], 4000.0);
                if let Some(ref src) = inp {
                    let mut out = self.alloc_image()?;
                    for px in 0..(size * size) {
                        let k = 1000.0 + src.sample_luma(px) * 39000.0 + temp - 4000.0;
                        let c = blackbody_color(k);
                        let i = px * 4;
                        out.data[i] = c[0] as f32;
                        out.data[i + 1] = c[1] as f32;
                        out.data[i + 2] = c[2] as f32;
                        out.data[i + 3] = 1.0;
                    }
                    node_outs.insert("out".to_string(), MatBuf::from_image(out));
                } else {
                    let c = blackbody_color(temp);
                    node_outs.insert("out".to_string(), MatBuf::Uniform([c[0], c[1], c[2], 1.0]));
                }
            }
            "bump" => {
                let height = self
                    .get_input(&node.id, &["height", "in", "a", "bump"])
                    .cloned()
                    .unwrap_or(MatBuf::Scalar(0.5));
                let strength = param_f64(p, &["strength", "distance"], 1.0);
                let invert = param_f64(p, &["invert"], 0.0) != 0.0;
                let normal =
                    height_to_normal_sobel(&height, size, strength, invert, &self.tracker)?;
                node_outs.insert("out".to_string(), normal.clone());
                node_outs.insert("normal".to_string(), normal);
            }
            "fresnel" => {
                let c = param_color3(p, &["color"], [1.0, 1.0, 1.0]);
                node_outs.insert("out".to_string(), MatBuf::Uniform([c[0], c[1], c[2], 1.0]));
            }
            other => {
                return Err(format!(
                    "node `{}`: unsupported NodeMaterial node type `{other}`",
                    node.id
                ));
            }
        }

        self.outputs.insert(node.id.clone(), node_outs);
        Ok(())
    }
}

/// Evaluates a Zoidium `material-plus` `NodeMaterial` procedural graph into deterministic
/// CPU RGBA8 PBR texture maps at `resolution.clamp(32, 1024)` plus scalar PBR fallbacks.
pub fn bake_node_material_maps(
    graph: &Value,
    resolution: u32,
) -> Result<BakedNodeMaterialMaps, String> {
    let (baked, _tracker) =
        bake_node_material_maps_with_budget(graph, resolution, MAX_BAKE_MEMORY_BYTES)?;
    Ok(baked)
}

fn bake_node_material_maps_with_budget(
    graph: &Value,
    resolution: u32,
    memory_budget_bytes: usize,
) -> Result<(BakedNodeMaterialMaps, Arc<BakeMemoryTracker>), String> {
    let res = resolution.clamp(32, 1024);
    let size = res as usize;
    let mut baked = BakedNodeMaterialMaps {
        resolution: res,
        ..Default::default()
    };
    let tracker = BakeMemoryTracker::new(memory_budget_bytes);

    let (nodes, links) = parse_raw_graph(graph, MAX_MATERIAL_NODES, MAX_MATERIAL_LINKS)?;
    if nodes.is_empty() {
        return Ok((baked, tracker));
    }

    let topo = topological_sort(&nodes, &links)?;

    let mut incoming: BTreeMap<(String, String), (String, String)> = BTreeMap::new();
    for link in &links {
        incoming.insert(
            (link.to_node.clone(), link.to_port.clone()),
            (link.from_node.clone(), link.from_port.clone()),
        );
    }
    let by_id: BTreeMap<&str, &RawNode> = nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    let Some(output_node) = nodes
        .iter()
        .find(|n| normalize_material_node_type(&n.kind) == "output")
    else {
        return Ok((baked, tracker));
    };

    // Map each consumer node ID to the deduplicated set of producer node IDs feeding it.
    let mut producers_by_consumer: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for ((to_node, _), (from_node, _)) in &incoming {
        producers_by_consumer
            .entry(to_node.clone())
            .or_default()
            .insert(from_node.clone());
    }

    // Backward reachability walk from the terminal Material Output node so disconnected nodes
    // are never evaluated and allocate zero bytes.
    let mut reachable: BTreeSet<String> = BTreeSet::new();
    let mut queue = VecDeque::new();
    reachable.insert(output_node.id.clone());
    queue.push_back(output_node.id.clone());
    while let Some(curr) = queue.pop_front() {
        if let Some(producers) = producers_by_consumer.get(&curr) {
            for prod in producers {
                if reachable.insert(prod.clone()) {
                    queue.push_back(prod.clone());
                }
            }
        }
    }

    // Remaining reachable consumer counts (refcount) per producer node so intermediate buffers
    // can be dropped immediately after their last consumer finishes reading them.
    let mut remaining_consumers: BTreeMap<String, usize> = BTreeMap::new();
    for consumer_id in &reachable {
        if let Some(producers) = producers_by_consumer.get(consumer_id) {
            for prod in producers {
                *remaining_consumers.entry(prod.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut eval = MaterialEvaluator {
        size,
        incoming,
        outputs: BTreeMap::new(),
        tracker: Arc::clone(&tracker),
        _marker: std::marker::PhantomData,
    };

    for id in &topo {
        if !reachable.contains(id) || id == &output_node.id {
            continue;
        }
        if let Some(&node) = by_id.get(id.as_str()) {
            eval.eval_node(node)?;
            if let Some(producers) = producers_by_consumer.get(id) {
                for prod in producers {
                    if let Some(rc) = remaining_consumers.get_mut(prod) {
                        *rc = rc.saturating_sub(1);
                        if *rc == 0 {
                            eval.outputs.remove(prod);
                        }
                    }
                }
            }
        }
    }

    let out_id = output_node.id.as_str();
    let normal_mode =
        (param_f64(&output_node.params, &["normalMode"], 0.0).round() as i32).clamp(0, 1);
    let normal_strength = param_f64(&output_node.params, &["normalStrength"], 1.0).clamp(0.0, 8.0);
    baked.normal_strength = normal_strength as f32;

    if let Some(buf) = eval.get_input(out_id, &["color", "baseColor", "albedo"]) {
        baked.base_color_factor = buf.mean_rgba();
        baked.color = Some(buf.to_rgba8(size));
    }
    if let Some(buf) = eval.get_input(out_id, &["emissive", "emission", "luminance"]) {
        let m = buf.mean_rgba();
        baked.emissive_factor = [m[0], m[1], m[2]];
        baked.emissive = Some(buf.to_rgba8(size));
    }
    if let Some(buf) = eval.get_input(out_id, &["roughness"]) {
        let m = buf.mean_rgba();
        baked.roughness_factor = m[0];
        baked.roughness = Some(buf.to_rgba8(size));
    }
    if let Some(buf) = eval.get_input(out_id, &["metalness", "metallic"]) {
        let m = buf.mean_rgba();
        baked.metalness_factor = m[0];
        baked.metalness = Some(buf.to_rgba8(size));
    }
    if let Some(buf) = eval.get_input(out_id, &["opacity", "alpha"]) {
        let m = buf.mean_rgba();
        baked.opacity_factor = m[0];
        baked.opacity = Some(buf.to_rgba8(size));
    }
    if let Some(buf) = eval.get_input(out_id, &["bump"]) {
        baked.bump = Some(buf.to_rgba8(size));
    }

    if let Some(buf) = eval.get_input(out_id, &["normal"]) {
        let normal_buf = if normal_mode == 1 {
            height_to_normal_sobel(buf, size, normal_strength, false, &eval.tracker)?
        } else {
            normalize_tangent_normal(buf, size, normal_strength, &eval.tracker)?
        };
        baked.normal = Some(normal_buf.to_rgba8(size));
    } else if let Some(bump_buf) = eval.get_input(out_id, &["bump"]) {
        // Derive tangent-space normal map via Sobel height-to-normal whenever `bump` is wired.
        let normal_buf =
            height_to_normal_sobel(bump_buf, size, normal_strength, false, &eval.tracker)?;
        baked.normal = Some(normal_buf.to_rgba8(size));
    }

    drop(eval);
    Ok((baked, tracker))
}

#[cfg(test)]
mod tests {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    use serde_json::json;

    use super::*;
    use crate::translate_glsl_es100;

    fn assert_valid_wgsl(glsl: &str, label: &str) {
        let compiled = translate_glsl_es100(glsl)
            .unwrap_or_else(|e| panic!("[{label}] GLSL ES 1.00 -> WGSL failed: {e}\n{glsl}"));
        assert!(compiled.wgsl.contains("@fragment"));
        let module = naga::front::wgsl::parse_str(&compiled.wgsl).unwrap_or_else(|e| {
            panic!(
                "[{label}] WGSL parse failed:\n{}",
                e.emit_to_string(&compiled.wgsl)
            )
        });
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&module)
            .unwrap_or_else(|e| {
                panic!(
                    "[{label}] WGSL validate failed:\n{}",
                    e.emit_to_string(&compiled.wgsl)
                )
            });
    }

    fn is_non_constant(map: &BakedMap) -> bool {
        let first = &map.rgba8[0..4];
        map.rgba8.as_chunks::<4>().0.iter().any(|px| px != first)
    }

    #[test]
    fn empty_and_outputless_node_effect_compile_to_passthrough_wgsl() {
        for graph in [
            json!({}),
            json!({ "nodes": [], "links": [] }),
            json!({ "nodes": [{ "id": "n1", "type": "noise" }], "links": [] }),
        ] {
            let res = compile_node_effect_to_glsl(&graph).expect("passthrough must compile");
            assert!(
                res.glsl_source
                    .contains("gl_FragColor = texture2D(tDiffuse, vUv);")
            );
            assert_valid_wgsl(&res.glsl_source, "passthrough");
        }
    }

    #[test]
    fn node_effect_detects_cycles() {
        let cyclic = json!({
            "nodes": [
                { "id": "a", "type": "math", "params": { "mode": 0 } },
                { "id": "b", "type": "math", "params": { "mode": 2 } },
                { "id": "out", "type": "output" }
            ],
            "links": [
                { "from": { "node": "a", "port": "out" }, "to": { "node": "b", "port": "a" } },
                { "from": { "node": "b", "port": "out" }, "to": { "node": "a", "port": "a" } },
                { "from": { "node": "b", "port": "out" }, "to": { "node": "out", "port": "color" } }
            ]
        });
        let err = compile_node_effect_to_glsl(&cyclic).unwrap_err();
        assert!(err.contains("cycle"), "expected cycle error, got: {err}");
    }

    #[test]
    fn compiles_multi_node_effect_graphs_through_naga_wgsl() {
        // Graph 1: UV -> swirl -> sample -> brightnessContrast + voronoi -> colorRamp -> mix -> vignette -> output
        let rich_graph = json!({
            "version": 1,
            "nodes": [
                { "id": "uv", "type": "uv" },
                { "id": "t", "type": "time" },
                { "id": "p_gain", "type": "parameter", "params": { "name": "Gain", "kind": 0, "value": 0.65, "min": 0.0, "max": 1.0 } },
                { "id": "sw", "type": "swirl", "params": { "radius": 0.5, "angle": 1.2 } },
                { "id": "samp", "type": "sample" },
                { "id": "bc", "type": "brightnessContrast", "params": { "brightness": 0.1, "contrast": 0.2 } },
                { "id": "vor", "type": "voronoi", "params": { "mode": 2, "scale": 6.0 } },
                { "id": "ramp", "type": "colorRamp", "params": {
                    "interpolation": 2,
                    "stops": [[0.0, 0.1, 0.0, 0.3, 1.0], [0.5, 0.9, 0.4, 0.1, 1.0], [1.0, 1.0, 1.0, 0.8, 1.0]]
                }},
                { "id": "blend", "type": "mix", "params": { "mode": "overlay" } },
                { "id": "vig", "type": "vignette", "params": { "radius": 0.8, "softness": 0.35, "strength": 0.7 } },
                { "id": "out", "type": "output" }
            ],
            "links": [
                { "from": { "node": "uv", "port": "out" }, "to": { "node": "sw", "port": "uv" } },
                { "from": { "node": "sw", "port": "uv" }, "to": { "node": "samp", "port": "uv" } },
                { "from": { "node": "samp", "port": "color" }, "to": { "node": "bc", "port": "color" } },
                { "from": { "node": "t", "port": "out" }, "to": { "node": "vor", "port": "time" } },
                { "from": { "node": "vor", "port": "out" }, "to": { "node": "ramp", "port": "factor" } },
                { "from": { "node": "bc", "port": "color" }, "to": { "node": "blend", "port": "a" } },
                { "from": { "node": "ramp", "port": "color" }, "to": { "node": "blend", "port": "b" } },
                { "from": { "node": "p_gain", "port": "out" }, "to": { "node": "blend", "port": "factor" } },
                { "from": { "node": "blend", "port": "color" }, "to": { "node": "vig", "port": "color" } },
                { "from": { "node": "vig", "port": "color" }, "to": { "node": "out", "port": "color" } },
                { "from": { "node": "samp", "port": "alpha" }, "to": { "node": "out", "port": "alpha" } }
            ]
        });
        let compiled = compile_node_effect_to_glsl(&rich_graph).expect("rich graph must compile");
        assert!(compiled.uses_time);
        assert_eq!(compiled.parameters.len(), 1);
        assert_eq!(compiled.parameters[0].label, "Gain");
        assert_valid_wgsl(&compiled.glsl_source, "rich-node-effect");

        // Graph 2: blur + edgeDetect + chromaticAberration + glow + fbm + kaleidoscope + math/vectorMath
        let filter_graph = json!({
            "nodes": [
                { "id": "kal", "type": "kaleidoscope", "params": { "segments": 6 } },
                { "id": "blur", "type": "boxBlur", "params": { "mode": 1, "taps": 9, "radius": 4.0 } },
                { "id": "edge", "type": "edgeDetect", "params": { "radius": 1.0, "strength": 1.5 } },
                { "id": "ca", "type": "chromaticAberration", "params": { "amount": 0.01 } },
                { "id": "glw", "type": "glow", "params": { "threshold": 0.5, "radius": 4.0, "intensity": 0.6 } },
                { "id": "fbm", "type": "fbm", "params": { "scale": 5.0 } },
                { "id": "vm", "type": "vectorMath", "params": { "mode": "normalize" } },
                { "id": "m1", "type": "mix", "params": { "mode": 1, "factor": 0.4 } },
                { "id": "m2", "type": "mix", "params": { "mode": 4, "factor": 0.3 } },
                { "id": "out", "type": "output" }
            ],
            "edges": [
                { "from": "kal", "fromPort": "uv", "to": "blur", "toPort": "uv" },
                { "from": "blur", "fromPort": "color", "to": "m1", "toPort": "a" },
                { "from": "edge", "fromPort": "color", "to": "m1", "toPort": "b" },
                { "from": "ca", "fromPort": "color", "to": "vm", "toPort": "a" },
                { "from": "m1", "fromPort": "color", "to": "m2", "toPort": "a" },
                { "from": "glw", "fromPort": "color", "to": "m2", "toPort": "b" },
                { "from": "fbm", "fromPort": "out", "to": "m2", "toPort": "factor" },
                { "from": "m2", "fromPort": "color", "to": "out", "toPort": "color" }
            ]
        });
        let compiled_filters =
            compile_node_effect_to_glsl(&filter_graph).expect("filter graph must compile");
        assert_valid_wgsl(&compiled_filters.glsl_source, "filter-node-effect");
    }

    #[test]
    fn bakes_multi_node_material_maps_with_color_ramp_and_sobel_normal() {
        // Multi-node graph:
        // - noise -> colorRamp -> output.color
        // - voronoi(Edge) -> bump -> output.normal
        // - brick -> separateRGB.r -> mapRange -> output.roughness
        // - wave -> math(multiply) -> output.metalness
        // - blackbody -> output.emissive
        let graph = json!({
            "nodes": [
                { "id": "n", "type": "noise", "params": { "scale": 4, "octaves": 4, "seed": 7, "contrast": 1.6 } },
                { "id": "ramp", "type": "colorRamp", "params": {
                    "interpolation": 1,
                    "stops": [[0.0, 0.05, 0.1, 0.2, 1.0], [0.5, 0.8, 0.3, 0.1, 1.0], [1.0, 1.0, 0.95, 0.8, 1.0]]
                }},
                { "id": "vor", "type": "voronoi", "params": { "feature": 3, "scale": 6, "seed": 3 } },
                { "id": "bmp", "type": "bump", "params": { "strength": 2.0 } },
                { "id": "brk", "type": "brick", "params": { "scale": 4, "seed": 2 } },
                { "id": "sep", "type": "separateRGB" },
                { "id": "mr", "type": "mapRange", "params": { "fromMin": 0.0, "fromMax": 1.0, "toMin": 0.2, "toMax": 0.9 } },
                { "id": "wv", "type": "wave", "params": { "scale": 4, "distortion": 1.5 } },
                { "id": "bb", "type": "blackbody", "params": { "temperature": 3200 } },
                { "id": "out", "type": "pbrOutput", "params": { "normalStrength": 1.5 } }
            ],
            "links": [
                { "from": "n", "fromPort": "out", "to": "ramp", "toPort": "factor" },
                { "from": "ramp", "fromPort": "out", "to": "out", "toPort": "color" },
                { "from": "vor", "fromPort": "out", "to": "bmp", "toPort": "height" },
                { "from": "bmp", "fromPort": "normal", "to": "out", "toPort": "normal" },
                { "from": "brk", "fromPort": "out", "to": "sep", "toPort": "color" },
                { "from": "sep", "fromPort": "r", "to": "mr", "toPort": "value" },
                { "from": "mr", "fromPort": "out", "to": "out", "toPort": "roughness" },
                { "from": "wv", "fromPort": "out", "to": "out", "toPort": "metalness" },
                { "from": "bb", "fromPort": "out", "to": "out", "toPort": "emissive" }
            ]
        });

        // Clamps low resolution up to 32.
        let maps = bake_node_material_maps(&graph, 16).expect("material bake must succeed");
        assert_eq!(maps.resolution, 32);

        let color = maps.color.as_ref().expect("color map wired");
        assert_eq!(color.rgba8.len(), 32 * 32 * 4);
        assert!(is_non_constant(color), "color map must vary spatially");

        let normal = maps.normal.as_ref().expect("normal map wired");
        assert_eq!(normal.rgba8.len(), 32 * 32 * 4);
        assert!(
            is_non_constant(normal),
            "Sobel normal map must vary spatially"
        );
        // Tangent-space normal Z channel must be positive (> 127) everywhere.
        assert!(
            normal
                .rgba8
                .as_chunks::<4>()
                .0
                .iter()
                .all(|px| px[2] >= 128)
        );

        let roughness = maps.roughness.as_ref().expect("roughness map wired");
        assert!(
            is_non_constant(roughness),
            "roughness map must vary spatially"
        );
        assert!(maps.roughness_factor >= 0.15 && maps.roughness_factor <= 0.95);

        let metalness = maps.metalness.as_ref().expect("metalness map wired");
        assert!(
            is_non_constant(metalness),
            "metalness map must vary spatially"
        );

        let emissive = maps.emissive.as_ref().expect("emissive map wired");
        // 3200K blackbody is warm orange (R > G > B).
        assert!(emissive.rgba8[0] > emissive.rgba8[1] && emissive.rgba8[1] > emissive.rgba8[2]);

        // Unwired opacity stays None with default scalar fallback 1.0.
        assert!(maps.opacity.is_none());
        assert!((maps.opacity_factor - 1.0).abs() < 1e-6);

        // Determinism check: identical input yields byte-identical outputs.
        let maps_again = bake_node_material_maps(&graph, 16).expect("deterministic bake");
        assert_eq!(maps, maps_again);
    }

    #[test]
    fn bakes_default_bump_to_sobel_normal_and_scalar_fallbacks() {
        let graph = json!({
            "nodes": [
                { "id": "m", "type": "marble", "params": { "scale": 4, "veins": 3, "seed": 5 } },
                { "id": "r", "type": "ridgedFractal", "params": { "scale": 5, "seed": 2 } },
                { "id": "t", "type": "turbulence", "params": { "scale": 5, "seed": 3 } },
                { "id": "mx", "type": "mix", "params": { "mode": "screen", "opacity": 0.5 } },
                { "id": "cc", "type": "colorCorrection", "params": { "contrast": 1.3, "hue": 30, "saturation": 1.2 } },
                { "id": "inv", "type": "invert", "params": { "strength": 1.0 } },
                { "id": "out", "type": "output", "params": { "normalStrength": 2.0 } }
            ],
            "links": [
                { "from": "r", "fromPort": "out", "to": "mx", "toPort": "a" },
                { "from": "t", "fromPort": "out", "to": "mx", "toPort": "b" },
                { "from": "mx", "fromPort": "out", "to": "cc", "toPort": "a" },
                { "from": "cc", "fromPort": "out", "to": "inv", "toPort": "a" },
                { "from": "inv", "fromPort": "out", "to": "out", "toPort": "color" },
                { "from": "m", "fromPort": "out", "to": "out", "toPort": "bump" }
            ]
        });

        let maps = bake_node_material_maps(&graph, 64).expect("bump->normal bake must succeed");
        assert_eq!(maps.resolution, 64);
        assert!(is_non_constant(maps.color.as_ref().unwrap()));
        assert!(is_non_constant(maps.bump.as_ref().unwrap()));
        // Auto-derived Sobel normal from `output.bump`.
        let normal = maps.normal.as_ref().expect("normal derived from bump");
        assert!(is_non_constant(normal));
        // Unwired roughness / metalness return default scalar PBR fallbacks.
        assert!(maps.roughness.is_none());
        assert!((maps.roughness_factor - 0.5).abs() < 1e-6);
        assert!(maps.metalness.is_none());
        assert!((maps.metalness_factor - 0.0).abs() < 1e-6);
    }

    #[test]
    fn node_material_bake_skips_disconnected_nodes() {
        // Disconnected subgraph includes heavy spatial nodes and an unsupported node type.
        // Only the constant color node wired to `out.color` is reachable and should allocate 0 image bytes.
        let graph = json!({
            "nodes": [
                { "id": "col", "type": "color", "params": { "color": [0.2, 0.4, 0.8] } },
                { "id": "dead_vor", "type": "voronoi", "params": { "scale": 8 } },
                { "id": "dead_noise", "type": "noise", "params": { "scale": 8, "octaves": 6 } },
                { "id": "dead_bad", "type": "nonExistentNodeType" },
                { "id": "out", "type": "output" }
            ],
            "links": [
                { "from": "col", "fromPort": "out", "to": "out", "toPort": "color" },
                { "from": "dead_vor", "fromPort": "out", "to": "dead_bad", "toPort": "in" }
            ]
        });

        let (maps, tracker) = bake_node_material_maps_with_budget(&graph, 64, 1024)
            .expect("disconnected nodes skipped");
        assert_eq!(
            tracker.peak_bytes(),
            0,
            "disconnected nodes must not allocate any intermediate image bytes"
        );
        assert_eq!(tracker.live_bytes(), 0);
        assert!(maps.color.is_some());
    }

    #[test]
    fn node_material_bake_releases_chain_intermediates_eagerly() {
        // 32-node linear chain at 64x64: each RGBA f32 buffer is 64*64*16 = 65,536 bytes.
        // Without eager refcount eviction, peak memory would be 32 * 64 KiB = 2 MiB.
        // With eager eviction, only producer + consumer (2 buffers = 128 KiB) are live at once.
        let chain_len = 32usize;
        let res = 64u32;
        let one_buf_bytes = (res as usize) * (res as usize) * 4 * std::mem::size_of::<f32>();

        let mut nodes = vec![json!({ "id": "n0", "type": "checker", "params": { "scale": 4 } })];
        let mut links = Vec::new();
        for i in 1..chain_len {
            nodes.push(json!({
                "id": format!("n{i}"),
                "type": "invert",
                "params": { "strength": 1.0 }
            }));
            links.push(json!({
                "from": format!("n{}", i - 1),
                "fromPort": "out",
                "to": format!("n{i}"),
                "toPort": "a"
            }));
        }
        nodes.push(json!({ "id": "out", "type": "output" }));
        links.push(json!({
            "from": format!("n{}", chain_len - 1),
            "fromPort": "out",
            "to": "out",
            "toPort": "color"
        }));

        let graph = json!({ "nodes": nodes, "links": links });
        // Budget of 3 buffers would be impossible without eager release (chain has 32 spatial nodes).
        let (maps, tracker) = bake_node_material_maps_with_budget(&graph, res, one_buf_bytes * 3)
            .expect("long chain must succeed within tight budget via eager release");
        assert!(maps.color.is_some());
        assert_eq!(
            tracker.live_bytes(),
            0,
            "all intermediate buffers must be freed after bake"
        );
        assert!(
            tracker.peak_bytes() <= one_buf_bytes * 2,
            "expected peak <= 2 buffers ({} bytes), got {} bytes",
            one_buf_bytes * 2,
            tracker.peak_bytes()
        );
    }

    #[test]
    fn node_material_bake_enforces_memory_budget_cleanly() {
        // Voronoi allocates 3 RGBA f32 buffers (dist, color, position) simultaneously.
        // Setting budget below 3 buffers (and testing a wide fan-in exceeding MAX_BAKE_MEMORY_BYTES)
        // must fail cleanly before exceeding the budget.
        let res = 64u32;
        let one_buf_bytes = (res as usize) * (res as usize) * 4 * std::mem::size_of::<f32>();
        let graph = json!({
            "nodes": [
                { "id": "v", "type": "voronoi", "params": { "scale": 4 } },
                { "id": "out", "type": "output" }
            ],
            "links": [
                { "from": "v", "fromPort": "out", "to": "out", "toPort": "color" }
            ]
        });

        let err = bake_node_material_maps_with_budget(&graph, res, one_buf_bytes * 2)
            .expect_err("exceeding bake memory budget must return Err");
        assert!(
            err.contains("memory budget exceeded"),
            "expected memory budget error, got: {err}"
        );

        // Also verify the default 256 MiB public API budget rejects a wide fan-in at 1024x1024
        // (6 independent `separateRGB` leaves = 18 * 16 MiB = 288 MiB > 256 MiB when all leaves
        // are evaluated first by Kahn's topological sort).
        let mut wide_nodes = Vec::new();
        let mut wide_links = Vec::new();
        for i in 0..6 {
            wide_nodes.push(json!({ "id": format!("s{i}"), "type": "separateRGB" }));
        }
        wide_nodes.push(json!({ "id": "m0", "type": "mix" }));
        wide_links.push(json!({ "from": "s0", "fromPort": "r", "to": "m0", "toPort": "a" }));
        wide_links.push(json!({ "from": "s1", "fromPort": "r", "to": "m0", "toPort": "b" }));
        for i in 2..6 {
            wide_nodes.push(json!({ "id": format!("m{}", i - 1), "type": "mix" }));
            wide_links.push(json!({ "from": format!("m{}", i - 2), "fromPort": "out", "to": format!("m{}", i - 1), "toPort": "a" }));
            wide_links.push(json!({ "from": format!("s{i}"), "fromPort": "r", "to": format!("m{}", i - 1), "toPort": "b" }));
        }
        wide_nodes.push(json!({ "id": "out", "type": "output" }));
        wide_links.push(json!({ "from": "m4", "fromPort": "out", "to": "out", "toPort": "color" }));
        let wide_graph = json!({ "nodes": wide_nodes, "links": wide_links });
        let public_err = bake_node_material_maps(&wide_graph, 1024)
            .expect_err("wide 1024x1024 graph exceeding 256 MiB budget must return Err");
        assert!(
            public_err.contains("memory budget exceeded"),
            "expected public API memory budget error, got: {public_err}"
        );
    }
}
