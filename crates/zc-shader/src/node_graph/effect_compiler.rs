use super::*;

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
                    _ => vec![param_f64(&node.params, &["value", "default"], 0.5) as f32],
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
                let mode = resolve_voronoi_mode(&node.params);
                if mode == 3 {
                    let fn_name = self.alloc_var("ne_voronoi_cell");
                    self.functions.push(format!(
                        "vec3 {fn_name}(vec2 p, float t) {{\n vec2 cell = floor(p), f = fract(p), best_c = cell;\n float f1 = 1.0e10;\n for (int y = -1; y <= 1; y++) {{\n  for (int x = -1; x <= 1; x++) {{\n   vec2 g = vec2(float(x), float(y)), c = cell + g;\n   vec2 point = 0.5 + 0.5 * sin(t + 6.2831853 * vec2(ne_hash(c), ne_hash(c + 19.1)));\n   vec2 r = g + point - f;\n   float d = dot(r, r);\n   if (d < f1) {{ f1 = d; best_c = c; }}\n  }}\n }}\n return vec3(ne_hash(best_c), ne_hash(best_c + 17.3), ne_hash(best_c + 41.7));\n}}"
                    ));
                    let cell_var = self.alloc_var("nvc");
                    self.lines.push(format!(
                        " vec3 {cell_var} = {fn_name}({uv} * {scale}, {time});"
                    ));
                    self.push_out(&mut outs, "out", glsl_scalar(&format!("{cell_var}.x")));
                    self.push_out(&mut outs, "color", glsl_rgb(&cell_var));
                } else {
                    let v = self.push_out(
                        &mut outs,
                        "out",
                        glsl_scalar(&format!("ne_voronoi({uv} * {scale}, {time})[{mode}]")),
                    );
                    outs.insert("color".to_string(), v);
                }
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
                let expr = match mode {
                    31 => format!("step({b}, {a})"),
                    32 => format!("ne_safe(-({a}))"),
                    _ => format!("ne_math({a}, {b}, {c}, {mode})"),
                };
                self.push_out(&mut outs, "out", glsl_scalar(&expr));
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
                    8 => format!("cross({a}, {b})"),
                    _ => {
                        let scale = self.read_num(node, &["scale", "b"], 1.0);
                        format!("{a} * ({scale})")
                    }
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
                let op = resolve_trig_op(&node.params);
                let expr = match op {
                    1 => format!("cos({x})"),
                    2 => format!("tan({x})"),
                    3 => format!("asin(clamp({x}, -1.0, 1.0))"),
                    4 => format!("acos(clamp({x}, -1.0, 1.0))"),
                    5 => format!("atan({x})"),
                    _ => format!("sin({x})"),
                };
                self.push_out(&mut outs, "out", glsl_scalar(&expr));
            }
            "mix" => {
                let mode = resolve_blend_mode(&node.params, &["mode", "blendMode"]);
                let a_v4 = self.read_port_expr(
                    node,
                    &["a", "base"],
                    Some(glsl_rgb("vec3(0.0, 0.0, 0.0)")),
                    0.0,
                );
                let b_v4 = self.read_port_expr(
                    node,
                    &["b", "top"],
                    Some(glsl_rgb("vec3(1.0, 1.0, 1.0)")),
                    1.0,
                );
                let fac = self.read_num(node, &["factor", "opacity", "mix"], 0.5);
                let a_var = self.alloc_var("nma");
                let b_var = self.alloc_var("nmb");
                let f_var = self.alloc_var("nmf");
                self.lines.push(format!(" vec4 {a_var} = {a_v4};"));
                self.lines.push(format!(" vec4 {b_var} = {b_v4};"));
                self.lines
                    .push(format!(" float {f_var} = clamp({fac}, 0.0, 1.0);"));
                let rgb_expr = if mode == 19 {
                    let fn_name = self.alloc_var("ne_hard_light");
                    self.functions.push(format!(
                        "vec3 {fn_name}(vec3 a, vec3 b, float f) {{\n vec3 r = vec3(\n  b.r < 0.5 ? 2.0 * a.r * b.r : 1.0 - 2.0 * (1.0 - a.r) * (1.0 - b.r),\n  b.g < 0.5 ? 2.0 * a.g * b.g : 1.0 - 2.0 * (1.0 - a.g) * (1.0 - b.g),\n  b.b < 0.5 ? 2.0 * a.b * b.b : 1.0 - 2.0 * (1.0 - a.b) * (1.0 - b.b)\n );\n return mix(a, r, clamp(f, 0.0, 1.0));\n}}"
                    ));
                    format!("{fn_name}({a_var}.rgb, {b_var}.rgb, {f_var})")
                } else {
                    format!("ne_mix({a_var}.rgb, {b_var}.rgb, {f_var}, {mode})")
                };
                let v = self.push_out(
                    &mut outs,
                    "color",
                    format!("vec4({rgb_expr}, mix({a_var}.a, {b_var}.a, {f_var}))"),
                );
                outs.insert("out".to_string(), v);
            }
            "colorRamp" => {
                let stops = parse_ramp_stops(&node.params);
                let interp = resolve_ramp_interp(&node.params);
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
                let src = self.read_port_expr(
                    node,
                    &["color", "in"],
                    Some(glsl_rgb("vec3(1.0, 1.0, 1.0)")),
                    1.0,
                );
                let s_var = self.alloc_var("nsrc");
                self.lines.push(format!(" vec4 {s_var} = {src};"));
                let h = self.read_num(node, &["hue", "h"], 0.0);
                let s = self.read_num(node, &["saturation", "s"], 1.0);
                let v = self.read_num(node, &["value", "v"], 1.0);
                let out = self.push_out(
                    &mut outs,
                    "color",
                    format!(
                        "vec4(ne_rgb(ne_hsv({s_var}.rgb) * vec3(1.0, {s}, {v}) + vec3({h}, 0.0, 0.0)), {s_var}.a)"
                    ),
                );
                outs.insert("out".to_string(), out);
            }
            "brightness" => {
                let src = self.read_port_expr(
                    node,
                    &["color", "in"],
                    Some(glsl_rgb("vec3(0.5, 0.5, 0.5)")),
                    0.5,
                );
                let s_var = self.alloc_var("nsrc");
                self.lines.push(format!(" vec4 {s_var} = {src};"));
                let br = self.read_num(node, &["brightness"], 0.0);
                let ct = self.read_num(node, &["contrast"], 0.0);
                let out = self.push_out(
                    &mut outs,
                    "color",
                    format!("vec4(({s_var}.rgb - 0.5) * (1.0 + {ct}) + 0.5 + {br}, {s_var}.a)"),
                );
                outs.insert("out".to_string(), out);
            }
            "colorBalance" => {
                let src = self.read_port_expr(
                    node,
                    &["color", "in"],
                    Some(glsl_rgb("vec3(0.5, 0.5, 0.5)")),
                    0.5,
                );
                let s_var = self.alloc_var("nsrc");
                self.lines.push(format!(" vec4 {s_var} = {src};"));
                let lift = self.read_col(node, &["lift", "shadows"], [0.0, 0.0, 0.0]);
                let gamma = self.read_col(node, &["gamma", "midtones"], [1.0, 1.0, 1.0]);
                let gain = self.read_col(node, &["gain", "highlights"], [1.0, 1.0, 1.0]);
                let out = self.push_out(
                    &mut outs,
                    "color",
                    format!(
                        "vec4(pow(clamp(({s_var}.rgb + {lift}) * {gain}, 0.0, 1.0), 1.0 / max({gamma}, vec3(0.01))), {s_var}.a)"
                    ),
                );
                outs.insert("out".to_string(), out);
            }
            "levels" => {
                let src = self.read_port_expr(
                    node,
                    &["color", "in"],
                    Some(glsl_rgb("vec3(0.5, 0.5, 0.5)")),
                    0.5,
                );
                let s_var = self.alloc_var("nsrc");
                self.lines.push(format!(" vec4 {s_var} = {src};"));
                let black = self.read_num(node, &["black", "inBlack"], 0.0);
                let white = self.read_num(node, &["white", "inWhite"], 1.0);
                let gamma = self.read_num(node, &["gamma"], 1.0);
                let out_black = self.read_num(node, &["outBlack"], 0.0);
                let out_white = self.read_num(node, &["outWhite"], 1.0);
                let out = self.push_out(
                    &mut outs,
                    "color",
                    format!(
                        "vec4(mix(vec3({out_black}), vec3({out_white}), pow(clamp(({s_var}.rgb - {black}) / max({white} - {black}, 1.0e-12), 0.0, 1.0), vec3(1.0 / max({gamma}, 1.0e-6)))), {s_var}.a)"
                    ),
                );
                outs.insert("out".to_string(), out);
            }
            "invert" => {
                let src = self.read_port_expr(
                    node,
                    &["color", "in"],
                    Some(glsl_rgb("vec3(0.5, 0.5, 0.5)")),
                    0.5,
                );
                let s_var = self.alloc_var("nsrc");
                self.lines.push(format!(" vec4 {s_var} = {src};"));
                let fac = self.read_num(node, &["factor", "strength"], 1.0);
                let out = self.push_out(
                    &mut outs,
                    "color",
                    format!(
                        "vec4(mix({s_var}.rgb, 1.0 - {s_var}.rgb, clamp({fac}, 0.0, 1.0)), {s_var}.a)"
                    ),
                );
                outs.insert("out".to_string(), out);
            }
            "grayscale" => {
                let src = self.read_port_expr(
                    node,
                    &["color", "in"],
                    Some(glsl_rgb("vec3(0.5, 0.5, 0.5)")),
                    0.5,
                );
                let s_var = self.alloc_var("nsrc");
                self.lines.push(format!(" vec4 {s_var} = {src};"));
                let fac = self.read_num(node, &["factor", "amount"], 1.0);
                let out = self.push_out(
                    &mut outs,
                    "color",
                    format!(
                        "vec4(mix({s_var}.rgb, vec3(dot({s_var}.rgb, vec3(0.2126, 0.7152, 0.0722))), clamp({fac}, 0.0, 1.0)), {s_var}.a)"
                    ),
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
                let mode = resolve_blur_mode(&node.params);
                let taps = (param_f64(&node.params, &["taps", "samples"], 9.0).round() as usize)
                    .clamp(1, 32);
                let radius = self.read_num(node, &["radius"], 8.0);
                let angle = self.read_num(node, &["angle"], 0.0);
                let cx = self.read_num(node, &["cx"], 0.5);
                let cy = self.read_num(node, &["cy"], 0.5);
                let fn_name = self.alloc_var("ne_blur");
                let mut fn_lines = vec![format!(
                    "vec4 {fn_name}(vec2 uv, float radius, float angle, vec2 center) {{\n vec2 px = radius / max(resolution, vec2(1.0));\n vec4 sum = vec4(0.0);\n float total = 0.0;"
                )];
                if mode == 0 {
                    // Box blur: always use an odd symmetric grid (3x3 minimum for taps >= 2)
                    // so taps=3 never collapses to a 1x1 no-op.
                    let side: usize = if taps <= 1 {
                        1
                    } else if taps <= 9 {
                        3
                    } else if taps <= 25 {
                        5
                    } else {
                        7
                    };
                    self.cost(side * side)?;
                    for y in 0..side {
                        for x in 0..side {
                            let dx = if side == 1 {
                                0.0
                            } else {
                                (x as f64) / ((side - 1) as f64) * 2.0 - 1.0
                            };
                            let dy = if side == 1 {
                                0.0
                            } else {
                                (y as f64) / ((side - 1) as f64) * 2.0 - 1.0
                            };
                            fn_lines.push(format!(
                                " {{ vec4 s = ne_sample(clamp(uv + vec2({dx:.6}, {dy:.6}) * px, 0.0, 1.0)); sum += vec4(s.rgb * s.a, s.a); total += 1.0; }}"
                            ));
                        }
                    }
                } else if mode == 1 {
                    // Radial Gaussian blur: center sample + staggered concentric rings so even
                    // low tap counts (e.g. taps=3) produce a real multi-tap isotropic Gaussian blur.
                    let rings: &[(f64, usize, f64)] = if taps <= 1 {
                        &[]
                    } else if taps <= 9 {
                        &[(0.5, 4, 0.0), (1.0, 8, PI / 8.0)]
                    } else if taps <= 20 {
                        &[(0.5, 8, 0.0), (1.0, 16, PI / 16.0)]
                    } else {
                        &[(0.35, 8, 0.0), (0.68, 12, PI / 12.0), (1.0, 16, PI / 16.0)]
                    };
                    let total_samples = 1 + rings.iter().map(|&(_, n, _)| n).sum::<usize>();
                    self.cost(total_samples)?;
                    fn_lines.push(
                        " { vec4 s = ne_sample(clamp(uv, 0.0, 1.0)); sum += vec4(s.rgb * s.a, s.a); total += 1.0; }"
                            .to_string(),
                    );
                    for &(r_norm, ring_n, phase) in rings {
                        let weight = (-2.0 * r_norm * r_norm).exp();
                        for k in 0..ring_n {
                            let theta = phase + TAU * (k as f64) / (ring_n as f64);
                            let dx = r_norm * theta.cos();
                            let dy = r_norm * theta.sin();
                            fn_lines.push(format!(
                                " {{ vec4 s = ne_sample(clamp(uv + vec2({dx:.6}, {dy:.6}) * px, 0.0, 1.0)); float w = {weight:.6}; sum += vec4(s.rgb * s.a, s.a) * w; total += w; }}"
                            ));
                        }
                    }
                } else {
                    let count = taps;
                    self.cost(count)?;
                    for i in 0..count {
                        let t = if count == 1 {
                            0.0
                        } else {
                            (i as f64) / ((count - 1) as f64) * 2.0 - 1.0
                        };
                        let weight = (-2.0 * t * t).exp();
                        let off = if mode == 2 {
                            format!("vec2(cos(angle), sin(angle)) * px * ({t:.6})")
                        } else {
                            format!(
                                "(uv - center) * radius * ({t:.6}) / max(max(resolution.x, resolution.y), 1.0)"
                            )
                        };
                        fn_lines.push(format!(
                            " {{ vec4 s = ne_sample(clamp(uv + {off}, 0.0, 1.0)); float w = {weight:.6}; sum += vec4(s.rgb * s.a, s.a) * w; total += w; }}"
                        ));
                    }
                }
                fn_lines.push(
                    " return vec4(sum.a > 0.000001 ? sum.rgb / sum.a : vec3(0.0), sum.a / max(total, 0.000001));\n}"
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
                // Smooth 3-ring radial glow: 1 center + 6 inner + 8 mid + 10 outer = 25 weighted samples.
                let rings: &[(f64, usize, f64)] =
                    &[(0.35, 6, 0.0), (0.68, 8, PI / 8.0), (1.0, 10, PI / 10.0)];
                let total_samples = 1 + rings.iter().map(|&(_, n, _)| n).sum::<usize>();
                self.cost(total_samples)?;
                let fn_name = self.alloc_var("ne_glow");
                let mut fn_lines = vec![format!(
                    "vec4 {fn_name}(vec2 uv, float threshold, float radius, float intensity) {{\n vec2 px = radius / max(resolution, vec2(1.0));\n vec4 base = ne_sample(clamp(uv, 0.0, 1.0));\n vec3 acc = vec3(0.0);\n float total = 0.0;\n float inv_span = 1.0 / max(1.0 - threshold, 0.001);\n {{\n  float l = dot(base.rgb, vec3(0.2126, 0.7152, 0.0722));\n  float hi = max(l - threshold, 0.0) * inv_span;\n  acc += base.rgb * hi;\n  total += 1.0;\n }}"
                )];
                for &(r_norm, ring_n, phase) in rings {
                    let weight = (-2.0 * r_norm * r_norm).exp();
                    for k in 0..ring_n {
                        let theta = phase + TAU * (k as f64) / (ring_n as f64);
                        let dx = r_norm * theta.cos();
                        let dy = r_norm * theta.sin();
                        fn_lines.push(format!(
                            " {{\n  vec4 s = ne_sample(clamp(uv + vec2({dx:.6}, {dy:.6}) * px, 0.0, 1.0));\n  float l = dot(s.rgb, vec3(0.2126, 0.7152, 0.0722));\n  float hi = max(l - threshold, 0.0) * inv_span;\n  float w = {weight:.6};\n  acc += s.rgb * hi * w;\n  total += w;\n }}"
                        ));
                    }
                }
                fn_lines.push(
                    " return vec4(clamp(base.rgb + (acc / max(total, 0.0001)) * intensity, 0.0, 1.0), base.a);\n}"
                        .to_string(),
                );
                self.functions.push(fn_lines.join("\n"));
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
                let color_expr = self.read_port_expr(
                    node,
                    &["color", "in"],
                    Some("ne_sample(vUv)".to_string()),
                    0.0,
                );
                let out_col = self.alloc_var("ne_out");
                self.lines.push(format!(" vec4 {out_col} = {color_expr};"));
                let alpha = if self.resolve_link(&node.id, &["alpha"]).is_some()
                    || node.params.contains_key("alpha")
                {
                    self.read_num(node, &["alpha"], 1.0)
                } else {
                    format!("{out_col}.a")
                };
                self.lines
                    .push(format!(" float ne_alpha = clamp({alpha}, 0.0, 1.0);"));
                self.lines.push(format!(
                    " gl_FragColor = vec4({out_col}.rgb * ne_alpha, ne_alpha);"
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

#[cfg(test)]
mod effect_compiler_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn trig_node_compiles_all_six_modes_for_integers_and_strings() {
        let cases: &[(i32, &str, &str)] = &[
            (0, "sin", "sin("),
            (1, "cos", "cos("),
            (2, "tan", "tan("),
            (3, "asin", "asin(clamp("),
            (4, "acos", "acos(clamp("),
            (5, "atan", "atan("),
        ];
        for &(idx, name, expected) in cases {
            for op_val in [json!(idx), json!(name)] {
                let graph = json!({
                    "nodes": [
                        { "id": "t", "type": "trig", "params": { "op": op_val, "x": 0.5 } },
                        { "id": "out", "type": "output" }
                    ],
                    "links": [
                        { "from": "t", "fromPort": "out", "to": "out", "toPort": "color" }
                    ]
                });
                let compiled = compile_node_effect_to_glsl(&graph).expect("trig compile");
                assert!(
                    compiled.glsl_source.contains(expected),
                    "trig op {op_val} expected `{expected}` in:\n{}",
                    compiled.glsl_source
                );
                compiled.to_wgsl().expect("trig WGSL must validate");
            }
        }
    }

    #[test]
    fn new_enum_modes_hard_light_step_negate_scale_voronoi_cell_id_compile_to_wgsl() {
        let graph = json!({
            "nodes": [
                { "id": "vor", "type": "voronoi", "params": { "mode": 3, "scale": 5.0 } },
                { "id": "st", "type": "math", "params": { "mode": 31, "a": 0.7, "b": 0.5 } },
                { "id": "ng", "type": "math", "params": { "mode": 32, "a": 0.4 } },
                { "id": "vm", "type": "vectorMath", "params": { "mode": 9, "scale": 2.0 } },
                { "id": "mx", "type": "mix", "params": { "mode": 19, "factor": 0.6 } },
                { "id": "out", "type": "output" }
            ],
            "links": [
                { "from": "vor", "fromPort": "color", "to": "vm", "toPort": "a" },
                { "from": "st", "fromPort": "out", "to": "vm", "toPort": "b" },
                { "from": "vm", "fromPort": "out", "to": "mx", "toPort": "a" },
                { "from": "ng", "fromPort": "out", "to": "mx", "toPort": "b" },
                { "from": "mx", "fromPort": "color", "to": "out", "toPort": "color" }
            ]
        });
        let compiled = compile_node_effect_to_glsl(&graph).expect("new enum modes compile");
        assert!(compiled.glsl_source.contains("ne_voronoi_cell"));
        assert!(compiled.glsl_source.contains("step("));
        assert!(compiled.glsl_source.contains("ne_safe(-("));
        assert!(compiled.glsl_source.contains("ne_hard_light"));
        compiled.to_wgsl().expect("new enum modes WGSL validate");
    }

    #[test]
    fn blur_taps_3_does_not_collapse_glow_is_multi_ring_and_output_preserves_alpha() {
        // 1. Box blur & Gaussian blur with taps=3 must never collapse to 1x1.
        for mode in [0, 1] {
            let graph = json!({
                "nodes": [
                    { "id": "b", "type": "blur", "params": { "mode": mode, "taps": 3, "radius": 6.0 } },
                    { "id": "out", "type": "output" }
                ],
                "links": [
                    { "from": "b", "fromPort": "color", "to": "out", "toPort": "color" }
                ]
            });
            let compiled = compile_node_effect_to_glsl(&graph).expect("blur taps=3 compile");
            assert!(
                compiled.sample_count >= 9,
                "blur mode={mode} taps=3 collapsed to {} samples",
                compiled.sample_count
            );
            // Output alpha must preserve upstream `.a`, never hardcoded `clamp(1.0, 0.0, 1.0)`.
            assert!(
                !compiled.glsl_source.contains("clamp(1.0, 0.0, 1.0)"),
                "unconnected output alpha must preserve input alpha:\n{}",
                compiled.glsl_source
            );
            assert!(compiled.glsl_source.contains(".a, 0.0, 1.0"));
            compiled.to_wgsl().expect("blur taps=3 WGSL validate");
        }

        // 2. Multi-ring radial glow uses 25 weighted samples and validates in Naga.
        let glow_graph = json!({
            "nodes": [
                { "id": "g", "type": "glow", "params": { "threshold": 0.5, "radius": 8.0, "intensity": 0.9 } },
                { "id": "out", "type": "output" }
            ],
            "links": [
                { "from": "g", "fromPort": "color", "to": "out", "toPort": "color" }
            ]
        });
        let glow_compiled = compile_node_effect_to_glsl(&glow_graph).expect("glow compile");
        assert_eq!(glow_compiled.sample_count, 25);
        glow_compiled.to_wgsl().expect("glow WGSL validate");
    }
}

// ===========================================================================
// B. NodeMaterial Procedural Texture Baker (`bake_node_material_maps`)
