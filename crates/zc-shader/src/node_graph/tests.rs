use super::*;

use naga::valid::{Capabilities, ValidationFlags, Validator};
use serde_json::json;

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

    let out_rgba8_bytes = 64 * 64 * 4;
    let (maps, tracker) = bake_node_material_maps_with_budget(&graph, 64, out_rgba8_bytes)
        .expect("disconnected nodes skipped");
    assert_eq!(
        tracker.peak_bytes(),
        out_rgba8_bytes,
        "disconnected nodes must not allocate any intermediate image bytes (only the single output RGBA8 map)"
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

#[test]
fn node_material_bake_tracks_fractal_lattice_and_output_rgba8_maps() {
    // 1. Fractal lattice temporary buffer reservation:
    // At 32x32 with scale=256 and octaves=1:
    //   values buffer  = 32 * 32 * 8   =   8,192 bytes
    //   lattice buffer = 256 * 256 * 8 = 524,288 bytes (512 KiB)
    //   f32 RGBA out   = 32 * 32 * 16  =  16,384 bytes
    //   RGBA8 output   = 32 * 32 * 4   =   4,096 bytes
    // Peak live bytes during eval_fractal_buffer is values + lattice = 532,480 bytes.
    let res = 32usize;
    let values_bytes = res * res * std::mem::size_of::<f64>();
    let lattice_bytes = 256 * 256 * std::mem::size_of::<f64>();
    let fractal_peak = values_bytes + lattice_bytes;
    let fractal_graph = json!({
        "nodes": [
            { "id": "rf", "type": "ridgedFractal", "params": { "scale": 256, "octaves": 1, "seed": 1 } },
            { "id": "out", "type": "output" }
        ],
        "links": [
            { "from": "rf", "fromPort": "out", "to": "out", "toPort": "color" }
        ]
    });

    let err = bake_node_material_maps_with_budget(&fractal_graph, res as u32, fractal_peak - 1)
        .expect_err("fractal lattice exceeding tight budget must fail cleanly");
    assert!(
        err.contains("memory budget exceeded"),
        "expected fractal lattice budget error, got: {err}"
    );

    let (maps, tracker) =
        bake_node_material_maps_with_budget(&fractal_graph, res as u32, fractal_peak)
            .expect("fractal bake must succeed at exact values + lattice budget");
    assert!(maps.color.is_some());
    assert_eq!(tracker.peak_bytes(), fractal_peak);
    assert_eq!(tracker.live_bytes(), 0);

    // 2. Final output RGBA8 maps reservation (combined with live producer buffer):
    // A spatial `checker` node wired to 4 output slots at 64x64:
    //   producer f32 RGBA = 64 * 64 * 16 = 65,536 bytes
    //   4x RGBA8 outputs  = 4 * (64 * 64 * 4) = 65,536 bytes
    //   Combined peak     = 131,072 bytes.
    let out_res = 64usize;
    let producer_bytes = out_res * out_res * 4 * std::mem::size_of::<f32>();
    let one_rgba8_bytes = out_res * out_res * 4;
    let combined_peak = producer_bytes + one_rgba8_bytes * 4;
    let multi_out_graph = json!({
        "nodes": [
            { "id": "chk", "type": "checker", "params": { "scale": 4 } },
            { "id": "out", "type": "output" }
        ],
        "links": [
            { "from": "chk", "fromPort": "out", "to": "out", "toPort": "color" },
            { "from": "chk", "fromPort": "out", "to": "out", "toPort": "emissive" },
            { "from": "chk", "fromPort": "out", "to": "out", "toPort": "roughness" },
            { "from": "chk", "fromPort": "out", "to": "out", "toPort": "metalness" }
        ]
    });

    let err_out =
        bake_node_material_maps_with_budget(&multi_out_graph, out_res as u32, combined_peak - 1)
            .expect_err("output RGBA8 maps exceeding tight budget must fail cleanly");
    assert!(
        err_out.contains("memory budget exceeded"),
        "expected output RGBA8 budget error, got: {err_out}"
    );

    let (maps_out, tracker_out) =
        bake_node_material_maps_with_budget(&multi_out_graph, out_res as u32, combined_peak)
            .expect("multi-output bake must succeed at exact producer + 4x RGBA8 budget");
    assert!(maps_out.color.is_some());
    assert!(maps_out.emissive.is_some());
    assert!(maps_out.roughness.is_some());
    assert!(maps_out.metalness.is_some());
    assert_eq!(tracker_out.peak_bytes(), combined_peak);
    assert_eq!(tracker_out.live_bytes(), 0);
}
