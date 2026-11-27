use super::*;

pub(super) const MAX_EFFECT_NODES: usize = 128;

pub(super) const MAX_EFFECT_LINKS: usize = 256;

pub(super) const MAX_EFFECT_SAMPLES: usize = 256;

pub(super) const MAX_MATERIAL_NODES: usize = 200;

pub(super) const MAX_MATERIAL_LINKS: usize = 400;

pub(super) const MAX_BAKE_MEMORY_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone)]
pub(super) struct RawNode {
    pub(super) id: String,
    pub(super) kind: String,
    pub(super) params: BTreeMap<String, Value>,
}

#[derive(Debug, Clone)]
pub(super) struct RawLink {
    pub(super) from_node: String,
    pub(super) from_port: String,
    pub(super) to_node: String,
    pub(super) to_port: String,
}

pub(super) fn parse_raw_graph(
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
        // Pull top-level image/asset reference keys first (lowest precedence).
        for asset_key in [
            "assetId", "asset_id", "asset", "image", "texture", "src", "media", "file", "url",
        ] {
            if let Some(v) = nobj.get(asset_key) {
                params.insert(asset_key.to_string(), v.clone());
            }
        }
        // Merge lower-precedence `values` / `inputs` maps next, then `params` on top.
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
pub(super) fn topological_sort(
    nodes: &[RawNode],
    links: &[RawLink],
) -> Result<Vec<String>, String> {
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

pub(super) fn param_f64(params: &BTreeMap<String, Value>, keys: &[&str], default: f64) -> f64 {
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

pub(super) fn param_str<'a>(params: &'a BTreeMap<String, Value>, keys: &[&str]) -> Option<&'a str> {
    for &key in keys {
        if let Some(s) = params.get(key).and_then(Value::as_str) {
            return Some(s);
        }
    }
    None
}

pub(super) fn parse_hex_color(hex: &str) -> Option<[f64; 3]> {
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

pub(super) fn param_color3(
    params: &BTreeMap<String, Value>,
    keys: &[&str],
    default: [f64; 3],
) -> [f64; 3] {
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

pub(super) fn parse_ramp_stops(params: &BTreeMap<String, Value>) -> Vec<[f64; 5]> {
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

pub(super) fn resolve_blend_mode(params: &BTreeMap<String, Value>, keys: &[&str]) -> i32 {
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
            "hard light" | "hardlight" | "hard_light" => 19,
            _ => 0,
        };
    }
    (param_f64(params, keys, 0.0).round() as i32).clamp(0, 19)
}

pub(super) fn resolve_math_op(
    params: &BTreeMap<String, Value>,
    keys: &[&str],
    default: i32,
) -> i32 {
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
            "smooth min" | "smooth minimum" | "smoothmin" | "smin" => 21,
            "smooth max" | "smooth maximum" | "smoothmax" | "smax" => 22,
            "snap" => 23,
            "wrap" => 24,
            "ping-pong" | "pingpong" => 25,
            "cosine" | "cos" => 26,
            "tangent" | "tan" => 27,
            "arctan2" | "atan2" => 28,
            "greater than" | "gt" | ">" => 29,
            "less than" | "lt" | "<" => 30,
            "step" => 31,
            "negate" | "neg" => 32,
            _ => default,
        };
    }
    (param_f64(params, keys, f64::from(default)).round() as i32).clamp(0, 32)
}

pub(super) fn resolve_vector_math_op(params: &BTreeMap<String, Value>) -> i32 {
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
            "scale" => 9,
            _ => 0,
        };
    }
    (param_f64(params, &["mode", "op", "operation"], 0.0).round() as i32).clamp(0, 9)
}

pub(super) fn resolve_voronoi_mode(params: &BTreeMap<String, Value>) -> i32 {
    if let Some(name) = param_str(params, &["mode", "feature"]) {
        return match name.trim().to_ascii_lowercase().as_str() {
            "f1" | "f1 distance" => 0,
            "f2" | "f2 distance" => 1,
            "edge" | "distance to edge" | "f2 - f1 edge" => 2,
            "cell id" | "cellid" | "cell_id" | "cell" => 3,
            _ => 0,
        };
    }
    (param_f64(params, &["mode", "feature"], 0.0).round() as i32).clamp(0, 3)
}

pub(super) fn resolve_ramp_interp(params: &BTreeMap<String, Value>) -> i32 {
    if let Some(name) = param_str(params, &["interpolation", "interp", "mode"]) {
        return match name.trim().to_ascii_lowercase().as_str() {
            "constant" | "step" | "none" => 0,
            "linear" => 1,
            "smooth" | "ease" | "smoothstep" => 2,
            "cubic" | "b-spline" | "bspline" | "catmull-rom" | "cardinal" => 3,
            _ => 1,
        };
    }
    (param_f64(params, &["interpolation", "interp", "mode"], 1.0).round() as i32).clamp(0, 3)
}

pub(super) fn resolve_normal_mode(params: &BTreeMap<String, Value>) -> i32 {
    if let Some(name) = param_str(params, &["normalMode", "mode"]) {
        return match name.trim().to_ascii_lowercase().as_str() {
            "tangent" | "tangent normal" | "normal" => 0,
            "bump" | "bump height" | "height" => 1,
            _ => 0,
        };
    }
    (param_f64(params, &["normalMode", "mode"], 0.0).round() as i32).clamp(0, 1)
}

pub(super) fn resolve_trig_op(params: &BTreeMap<String, Value>) -> i32 {
    if let Some(name) = param_str(params, &["op", "mode", "function"]) {
        return match name.trim().to_ascii_lowercase().as_str() {
            "sin" | "sine" | "0" => 0,
            "cos" | "cosine" | "1" => 1,
            "tan" | "tangent" | "2" => 2,
            "asin" | "arcsin" | "3" => 3,
            "acos" | "arccos" | "4" => 4,
            "atan" | "arctan" | "5" => 5,
            _ => 0,
        };
    }
    (param_f64(params, &["op", "mode", "function"], 0.0).round() as i32).clamp(0, 5)
}

pub(super) fn resolve_blur_mode(params: &BTreeMap<String, Value>) -> i32 {
    if let Some(name) = param_str(params, &["mode", "type"]) {
        return match name.trim().to_ascii_lowercase().as_str() {
            "box" => 0,
            "gaussian" | "gauss" => 1,
            "directional" | "motion" => 2,
            "radial" | "zoom" => 3,
            _ => 0,
        };
    }
    (param_f64(params, &["mode", "type"], 0.0).round() as i32).clamp(0, 3)
}

#[cfg(test)]
mod graph_parse_tests {
    use super::*;
    use serde_json::json;

    fn map1(k: &str, v: Value) -> BTreeMap<String, Value> {
        let mut m = BTreeMap::new();
        m.insert(k.to_string(), v);
        m
    }

    #[test]
    fn blend_mode_indices_and_names_match_1_to_1() {
        let names = [
            "Mix",
            "Add",
            "Subtract",
            "Multiply",
            "Screen",
            "Overlay",
            "Darken",
            "Lighten",
            "Difference",
            "Soft Light",
            "Color Dodge",
            "Color Burn",
            "Linear Light",
            "Divide",
            "Exclusion",
            "Hue",
            "Saturation",
            "Color",
            "Luminosity",
            "Hard Light",
        ];
        for (idx, name) in names.iter().enumerate() {
            assert_eq!(
                resolve_blend_mode(&map1("mode", json!(name)), &["mode"]),
                idx as i32,
                "blend name {name}"
            );
            assert_eq!(
                resolve_blend_mode(&map1("mode", json!(idx)), &["mode"]),
                idx as i32,
                "blend index {idx}"
            );
        }
    }

    #[test]
    fn math_op_indices_and_names_match_1_to_1() {
        let names = [
            "Add",
            "Subtract",
            "Multiply",
            "Divide",
            "Min",
            "Max",
            "Power",
            "Modulo",
            "Abs",
            "Invert",
            "Clamp",
            "Compare",
            "Sin",
            "Round",
            "Sqrt",
            "Log",
            "Exp",
            "Floor",
            "Ceil",
            "Fract",
            "Sign",
            "Smooth Min",
            "Smooth Max",
            "Snap",
            "Wrap",
            "Ping-Pong",
            "Cos",
            "Tan",
            "Atan2",
            "Greater Than",
            "Less Than",
            "Step",
            "Negate",
        ];
        for (idx, name) in names.iter().enumerate() {
            assert_eq!(
                resolve_math_op(&map1("mode", json!(name)), &["mode"], 0),
                idx as i32,
                "math name {name}"
            );
            assert_eq!(
                resolve_math_op(&map1("mode", json!(idx)), &["mode"], 0),
                idx as i32,
                "math index {idx}"
            );
        }
    }

    #[test]
    fn vector_math_voronoi_ramp_normal_and_trig_resolvers_match_indices() {
        let vec_names = [
            "Add",
            "Subtract",
            "Multiply",
            "Divide",
            "Normalize",
            "Dot",
            "Length",
            "Distance",
            "Cross",
            "Scale",
        ];
        for (idx, name) in vec_names.iter().enumerate() {
            assert_eq!(
                resolve_vector_math_op(&map1("mode", json!(name))),
                idx as i32
            );
            assert_eq!(
                resolve_vector_math_op(&map1("mode", json!(idx))),
                idx as i32
            );
        }

        let vor_names = ["F1 Distance", "F2 Distance", "Edge", "Cell ID"];
        for (idx, name) in vor_names.iter().enumerate() {
            assert_eq!(resolve_voronoi_mode(&map1("mode", json!(name))), idx as i32);
            assert_eq!(resolve_voronoi_mode(&map1("mode", json!(idx))), idx as i32);
        }

        let ramp_names = ["Constant", "Linear", "Smooth", "Cubic"];
        for (idx, name) in ramp_names.iter().enumerate() {
            assert_eq!(
                resolve_ramp_interp(&map1("interpolation", json!(name))),
                idx as i32
            );
            assert_eq!(
                resolve_ramp_interp(&map1("interpolation", json!(idx))),
                idx as i32
            );
        }

        assert_eq!(
            resolve_normal_mode(&map1("normalMode", json!("Tangent Normal"))),
            0
        );
        assert_eq!(resolve_normal_mode(&map1("normalMode", json!(0))), 0);
        assert_eq!(
            resolve_normal_mode(&map1("normalMode", json!("Bump Height"))),
            1
        );
        assert_eq!(resolve_normal_mode(&map1("normalMode", json!(1))), 1);

        let trig_names = ["Sin", "Cos", "Tan", "Asin", "Acos", "Atan"];
        for (idx, name) in trig_names.iter().enumerate() {
            assert_eq!(resolve_trig_op(&map1("op", json!(name))), idx as i32);
            assert_eq!(resolve_trig_op(&map1("op", json!(idx))), idx as i32);
        }
    }
}

// ===========================================================================
// A. Node Effect Compiler -> GLSL ES 1.00 (`compile_node_effect_to_glsl`)
