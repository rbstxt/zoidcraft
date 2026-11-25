use super::*;

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

pub(super) fn bake_node_material_maps_with_budget(
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
