use std::collections::HashMap;

use super::*;

/// Evaluates a Zoidium `material-plus` `NodeMaterial` procedural graph into deterministic
/// CPU RGBA8 PBR texture maps at `resolution.clamp(MIN_NODE_MATERIAL_RES, MAX_NODE_MATERIAL_RES)`
/// plus scalar PBR fallbacks.
pub fn bake_node_material_maps(
    graph: &Value,
    resolution: u32,
) -> Result<BakedNodeMaterialMaps, String> {
    let empty_images = HashMap::new();
    bake_node_material_maps_with_images(graph, resolution, &empty_images)
}

/// Evaluates a Zoidium `material-plus` `NodeMaterial` procedural graph with real per-pixel
/// bilinear RGBA image sampling for any `Image` nodes whose asset key is present in `images`.
pub fn bake_node_material_maps_with_images(
    graph: &Value,
    resolution: u32,
    images: &HashMap<String, (u32, u32, Vec<u8>)>,
) -> Result<BakedNodeMaterialMaps, String> {
    let (baked, _tracker) = bake_node_material_maps_with_images_and_budget(
        graph,
        resolution,
        images,
        MAX_BAKE_MEMORY_BYTES,
    )?;
    Ok(baked)
}

/// Alias for [`bake_node_material_maps_with_images`].
pub fn bake_node_material_with_images(
    graph: &Value,
    resolution: u32,
    images: &HashMap<String, (u32, u32, Vec<u8>)>,
) -> Result<BakedNodeMaterialMaps, String> {
    bake_node_material_maps_with_images(graph, resolution, images)
}

#[allow(dead_code)]
pub(super) fn bake_node_material_maps_with_budget(
    graph: &Value,
    resolution: u32,
    memory_budget_bytes: usize,
) -> Result<(BakedNodeMaterialMaps, Arc<BakeMemoryTracker>), String> {
    let empty_images = HashMap::new();
    bake_node_material_maps_with_images_and_budget(
        graph,
        resolution,
        &empty_images,
        memory_budget_bytes,
    )
}

pub(super) fn bake_node_material_maps_with_images_and_budget(
    graph: &Value,
    resolution: u32,
    images: &HashMap<String, (u32, u32, Vec<u8>)>,
    memory_budget_bytes: usize,
) -> Result<(BakedNodeMaterialMaps, Arc<BakeMemoryTracker>), String> {
    let res = resolution.clamp(MIN_NODE_MATERIAL_RES, MAX_NODE_MATERIAL_RES);
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
        images,
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
    let normal_mode = resolve_normal_mode(&output_node.params);
    let normal_strength = param_f64(&output_node.params, &["normalStrength"], 1.0).clamp(0.0, 8.0);
    baked.normal_strength = normal_strength as f32;

    let mut output_reservations: Vec<MemoryReservation> = Vec::new();

    if let Some(buf) = eval.get_input(out_id, &["color", "baseColor", "albedo"]) {
        baked.base_color_factor = buf.mean_rgba();
        let (map, res) = buf.to_rgba8(size, &eval.tracker)?;
        baked.color = Some(map);
        output_reservations.push(res);
    }
    if let Some(buf) = eval.get_input(out_id, &["emissive", "emission", "luminance"]) {
        let m = buf.mean_rgba();
        baked.emissive_factor = [m[0], m[1], m[2]];
        let (map, res) = buf.to_rgba8(size, &eval.tracker)?;
        baked.emissive = Some(map);
        output_reservations.push(res);
    }
    if let Some(buf) = eval.get_input(out_id, &["roughness"]) {
        let m = buf.mean_rgba();
        baked.roughness_factor = m[0];
        let (map, res) = buf.to_rgba8(size, &eval.tracker)?;
        baked.roughness = Some(map);
        output_reservations.push(res);
    }
    if let Some(buf) = eval.get_input(out_id, &["metalness", "metallic"]) {
        let m = buf.mean_rgba();
        baked.metalness_factor = m[0];
        let (map, res) = buf.to_rgba8(size, &eval.tracker)?;
        baked.metalness = Some(map);
        output_reservations.push(res);
    }
    if let Some(buf) = eval.get_input(out_id, &["opacity", "alpha"]) {
        let m = buf.mean_rgba();
        baked.opacity_factor = m[0];
        let (map, res) = buf.to_rgba8(size, &eval.tracker)?;
        baked.opacity = Some(map);
        output_reservations.push(res);
    }
    if let Some(buf) = eval.get_input(out_id, &["fresnel", "specular", "reflectance"]) {
        let m = buf.mean_rgba();
        baked.fresnel_factor = [m[0], m[1], m[2]];
        let (map, res) = buf.to_rgba8(size, &eval.tracker)?;
        baked.fresnel = Some(map);
        output_reservations.push(res);
    }
    if let Some(buf) = eval.get_input(out_id, &["bump"]) {
        let (map, res) = buf.to_rgba8(size, &eval.tracker)?;
        baked.bump = Some(map);
        output_reservations.push(res);
    }

    if let Some(buf) = eval.get_input(out_id, &["normal"]) {
        let normal_buf = if normal_mode == 1 {
            height_to_normal_sobel(buf, size, normal_strength, false, &eval.tracker)?
        } else {
            normalize_tangent_normal(buf, size, normal_strength, &eval.tracker)?
        };
        let (map, res) = normal_buf.to_rgba8(size, &eval.tracker)?;
        baked.normal = Some(map);
        output_reservations.push(res);
    } else if let Some(bump_buf) = eval.get_input(out_id, &["bump"]) {
        // Derive tangent-space normal map via Sobel height-to-normal whenever `bump` is wired.
        let normal_buf =
            height_to_normal_sobel(bump_buf, size, normal_strength, false, &eval.tracker)?;
        let (map, res) = normal_buf.to_rgba8(size, &eval.tracker)?;
        baked.normal = Some(map);
        output_reservations.push(res);
    }

    drop(eval);
    drop(output_reservations);
    Ok((baked, tracker))
}

#[cfg(test)]
mod material_bake_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn max_resolution_cap_is_2048() {
        assert_eq!(MAX_NODE_MATERIAL_RES, 2048);
        let graph = json!({
            "nodes": [
                { "id": "c", "type": "color", "params": { "color": [0.25, 0.5, 0.75] } },
                { "id": "out", "type": "output" }
            ],
            "links": [
                { "from": "c", "fromPort": "out", "to": "out", "toPort": "color" }
            ]
        });
        let maps = bake_node_material_maps(&graph, 4096).expect("2048 bake must succeed");
        assert_eq!(maps.resolution, 2048);
        assert_eq!(maps.color.as_ref().unwrap().rgba8.len(), 2048 * 2048 * 4);
    }

    #[test]
    fn image_node_samples_real_bilinear_rgba_pixels() {
        // 2x2 RGBA8 image:
        // (0,0)=Red(255,0,0,255)   (1,0)=Green(0,255,0,128)
        // (0,1)=Blue(0,0,255,64)   (1,1)=White(255,255,255,255)
        let mut images = HashMap::new();
        images.insert(
            "tex_checker".to_string(),
            (
                2u32,
                2u32,
                vec![
                    255, 0, 0, 255, 0, 255, 0, 128, // row 0
                    0, 0, 255, 64, 255, 255, 255, 255, // row 1
                ],
            ),
        );

        let graph = json!({
            "nodes": [
                { "id": "img", "type": "image", "params": { "assetId": "tex_checker" } },
                { "id": "out", "type": "output" }
            ],
            "links": [
                { "from": "img", "fromPort": "color", "to": "out", "toPort": "color" },
                { "from": "img", "fromPort": "alpha", "to": "out", "toPort": "opacity" }
            ]
        });

        let maps =
            bake_node_material_with_images(&graph, 32, &images).expect("image bake must succeed");
        let color = maps.color.as_ref().expect("color map");
        let opacity = maps.opacity.as_ref().expect("opacity map");

        // Sample near each quadrant center (8,8), (24,8), (8,24), (24,24) in 32x32 output:
        let px = |m: &BakedMap, x: usize, y: usize| -> [u8; 4] {
            let i = (x + y * 32) * 4;
            [m.rgba8[i], m.rgba8[i + 1], m.rgba8[i + 2], m.rgba8[i + 3]]
        };
        let tl = px(color, 8, 8);
        let tr = px(color, 24, 8);
        let bl = px(color, 8, 24);
        let br = px(color, 24, 24);
        assert!(tl[0] > 220 && tl[1] < 35 && tl[2] < 35, "TL red: {tl:?}");
        assert!(tr[1] > 220 && tr[0] < 35 && tr[2] < 35, "TR green: {tr:?}");
        assert!(bl[2] > 220 && bl[0] < 35 && bl[1] < 35, "BL blue: {bl:?}");
        assert!(
            br[0] > 220 && br[1] > 220 && br[2] > 220,
            "BR white: {br:?}"
        );
        // Opacity port carries the alpha channel spatially (not collapsed to 255).
        assert!(px(opacity, 8, 8)[0] > 220);
        assert!(px(opacity, 8, 24)[0] < 100);
    }

    #[test]
    fn fresnel_node_uses_3d_normals_and_populates_output_fresnel_map() {
        // 1. Unwired normal: smooth hemisphere Schlick Fresnel (center == F0, edges brighter, corners NOT flat-clamped equal to mid-edge).
        let graph_hemi = json!({
            "nodes": [
                { "id": "fr", "type": "fresnel", "params": { "ior": 1.5, "power": 5.0, "mix": 1.0 } },
                { "id": "out", "type": "output" }
            ],
            "links": [
                { "from": "fr", "fromPort": "out", "to": "out", "toPort": "fresnel" }
            ]
        });
        let maps = bake_node_material_maps(&graph_hemi, 32).expect("fresnel bake");
        let fmap = maps.fresnel.as_ref().expect("fresnel output map populated");
        let center_r = fmap.rgba8[(16 + 16 * 32) * 4];
        let edge_r = fmap.rgba8[(16 * 32) * 4];
        assert!(
            edge_r > center_r + 100,
            "grazing edge ({edge_r}) must be much brighter than normal-incidence center ({center_r})"
        );
        assert!(maps.fresnel_factor[0] > 0.0);

        // 2. Wired normal map: a flat normal vs a tilted normal produce distinct Schlick Fresnel values.
        let make_norm_graph = |norm_rgb: [f64; 3]| {
            json!({
                "nodes": [
                    { "id": "n", "type": "color", "params": { "color": norm_rgb } },
                    { "id": "fr", "type": "fresnel", "params": { "ior": 1.5, "power": 5.0, "mix": 1.0 } },
                    { "id": "out", "type": "output" }
                ],
                "links": [
                    { "from": "n", "fromPort": "color", "to": "fr", "toPort": "normal" },
                    { "from": "fr", "fromPort": "out", "to": "out", "toPort": "fresnel" }
                ]
            })
        };
        let flat = bake_node_material_maps(&make_norm_graph([0.5, 0.5, 1.0]), 32).unwrap();
        let tilted = bake_node_material_maps(&make_norm_graph([0.95, 0.5, 0.6]), 32).unwrap();
        assert!(
            tilted.fresnel_factor[0] > flat.fresnel_factor[0] + 0.1,
            "tilted normal Fresnel ({}) must exceed flat normal Fresnel ({})",
            tilted.fresnel_factor[0],
            flat.fresnel_factor[0]
        );
    }

    #[test]
    fn normal_mode_and_new_blend_math_ramp_modes_evaluate_accurately() {
        // Hard Light (mode 19): base=0.5, top=0.25 -> 2*0.5*0.25 = 0.25
        let hl_graph = json!({
            "nodes": [
                { "id": "a", "type": "color", "params": { "color": [0.5, 0.5, 0.5] } },
                { "id": "b", "type": "color", "params": { "color": [0.25, 0.75, 0.5] } },
                { "id": "m", "type": "mix", "params": { "mode": 19, "opacity": 1.0 } },
                { "id": "out", "type": "output" }
            ],
            "links": [
                { "from": "a", "fromPort": "out", "to": "m", "toPort": "a" },
                { "from": "b", "fromPort": "out", "to": "m", "toPort": "b" },
                { "from": "m", "fromPort": "out", "to": "out", "toPort": "color" }
            ]
        });
        let hl = bake_node_material_maps(&hl_graph, 32).unwrap();
        assert!((hl.base_color_factor[0] - 0.25).abs() < 0.02);
        assert!((hl.base_color_factor[1] - 0.75).abs() < 0.02);

        // Math Step (31) and Negate (32)
        let step_graph = json!({
            "nodes": [
                { "id": "m", "type": "math", "params": { "mode": 31, "a": 0.8, "b": 0.5 } },
                { "id": "out", "type": "output" }
            ],
            "links": [
                { "from": "m", "fromPort": "out", "to": "out", "toPort": "roughness" }
            ]
        });
        let step_res = bake_node_material_maps(&step_graph, 32).unwrap();
        assert!((step_res.roughness_factor - 1.0).abs() < 1e-4);

        // Normal mode 0 (Tangent Normal) vs 1 (Bump Height) on a horizontal gradient
        let make_norm_mode = |mode: i32| {
            json!({
                "nodes": [
                    { "id": "g", "type": "gradient", "params": { "mode": 0 } },
                    { "id": "out", "type": "output", "params": { "normalMode": mode, "normalStrength": 2.0 } }
                ],
                "links": [
                    { "from": "g", "fromPort": "out", "to": "out", "toPort": "normal" }
                ]
            })
        };
        let as_tangent = bake_node_material_maps(&make_norm_mode(0), 32).unwrap();
        let as_bump = bake_node_material_maps(&make_norm_mode(1), 32).unwrap();
        assert_ne!(
            as_tangent.normal.as_ref().unwrap().rgba8,
            as_bump.normal.as_ref().unwrap().rgba8
        );
    }
}
