use super::*;
use zc_scene::{DeformerFrame, voronoi_piece_color};

// ---------------------------------------------------------------------------
// Effector+, Repeater & ExpressText+ Scene Runtime
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SceneEffectorKind {
    Repeater(RepeaterMode),
    RadialRepeater,
    GridRepeater,
    Twist,
    Warp,
    Shear,
    Plain,
    Delay,
    VoronoiFracture,
}

pub(super) fn detect_scene_effector_kind(obj: &Object3D) -> Option<SceneEffectorKind> {
    let raw = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            obj.preserved
                .get("effectorType")
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| {
            obj.preserved
                .get("type")
                .and_then(serde_json::Value::as_str)
        })
        .or(match &obj.kind {
            Object3DKind::Dummy { original_type, .. } => original_type.as_deref(),
            _ => None,
        })?;
    let mut normalized = raw.trim().to_ascii_lowercase().replace('_', "-");
    if let Some(rest) = normalized.strip_prefix("zoidcraft:") {
        normalized = format!("zoidium:{rest}");
    }
    if let Some(rest) = normalized.strip_prefix("zoidium:effector-plus/") {
        normalized = format!("zoidium:repeater/{rest}");
    } else if let Some(rest) = normalized.strip_prefix("effector-plus/") {
        normalized = format!("zoidium:repeater/{rest}");
    }
    match normalized.as_str() {
        "zoidium:repeater/repeater" | "repeater" | "step-repeater" => {
            Some(SceneEffectorKind::Repeater(RepeaterMode::Step))
        }
        "zoidium:repeater/linear-repeater" | "linear-repeater" => {
            Some(SceneEffectorKind::Repeater(RepeaterMode::Linear))
        }
        "zoidium:repeater/random-repeater" | "random-repeater" => {
            Some(SceneEffectorKind::Repeater(RepeaterMode::Random))
        }
        "zoidium:repeater/echo-repeater" | "echo-repeater" => {
            Some(SceneEffectorKind::Repeater(RepeaterMode::Echo))
        }
        "zoidium:repeater/radial"
        | "zoidium:repeater/radial-repeater"
        | "radial-repeater"
        | "radial" => Some(SceneEffectorKind::RadialRepeater),
        "zoidium:repeater/grid" | "zoidium:repeater/grid-repeater" | "grid-repeater" | "grid" => {
            Some(SceneEffectorKind::GridRepeater)
        }
        "zoidium:repeater/twist" | "twist" | "7" => Some(SceneEffectorKind::Twist),
        "zoidium:repeater/warp" | "zoidium:repeater/bend" | "warp" | "bend" | "8" => {
            Some(SceneEffectorKind::Warp)
        }
        "zoidium:repeater/shear" | "shear" => Some(SceneEffectorKind::Shear),
        "zoidium:repeater/voronoi-fracture" | "voronoi-fracture" | "fracture" | "9" => {
            Some(SceneEffectorKind::VoronoiFracture)
        }
        "zoidium:repeater/plain" | "plain" | "plain-effector" => Some(SceneEffectorKind::Plain),
        "zoidium:repeater/delay" | "delay" | "delay-effector" => Some(SceneEffectorKind::Delay),
        _ => None,
    }
}

pub(super) fn collect_node_effector_chain(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    node_idx: usize,
    effector_kinds: &[Option<SceneEffectorKind>],
    has_mesh_descendant: &[bool],
) -> Vec<usize> {
    let mut chain = Vec::new();
    let push_unique = |chain: &mut Vec<usize>, idx: usize| {
        if !chain.contains(&idx) {
            chain.push(idx);
        }
    };
    let collect_sibling_deformers =
        |chain: &mut Vec<usize>, target_idx: usize, shared_parent: Option<usize>| {
            for (s_idx, s_node) in eval_graph.nodes.iter().enumerate() {
                if s_idx != target_idx
                    && s_node.parent == shared_parent
                    && s_node.render_enabled
                    && effector_kinds[s_idx].is_some()
                    && !has_mesh_descendant[s_idx]
                {
                    push_unique(chain, s_idx);
                }
            }
        };

    let parent = eval_graph.nodes[node_idx].parent;
    // 1. Sibling childless deformers sharing the same parent (C4D / sibling convention).
    collect_sibling_deformers(&mut chain, node_idx, parent);

    // 2. Walk the FULL ancestor chain innermost -> outermost, skipping through plain `Group`
    //    containers so `Effector -> Group -> Text/Shape/Model` AND
    //    `Parent -> [Effector, Group -> Text/Shape/Model]` both apply identically to direct children.
    let mut cur = parent;
    let mut guard = 0usize;
    while let Some(anc) = cur {
        if guard >= 256 {
            break;
        }
        let anc_node = &eval_graph.nodes[anc];
        if anc_node.render_enabled {
            if effector_kinds[anc].is_some() {
                push_unique(&mut chain, anc);
            } else if matches!(anc_node.obj.kind, Object3DKind::Group { .. }) {
                collect_sibling_deformers(&mut chain, anc, anc_node.parent);
            }
        }
        cur = anc_node.parent;
        guard += 1;
    }
    chain
}

pub(super) struct DeformedInstance {
    pub(super) mesh: Arc<MeshData>,
    pub(super) world: Mat4,
}

/// Precomputed shared AABB in an effector's local space across all affected descendant meshes.
#[derive(Debug, Clone, Copy)]
pub(super) struct SharedEffectorBounds {
    pub(super) bounds: Aabb3,
    /// Number of distinct scene object nodes (`Object3D`) sharing this effector.
    pub(super) distinct_nodes: usize,
    /// Total number of mesh entries (including split `Text3D` characters) sharing this effector.
    pub(super) mesh_count: usize,
}

/// Character-level Voronoi fracture behavior when a `Text3D` object is inside a Voronoi Fracture effector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VoronoiCharacterMode {
    /// Whole-volume fracture (or Zoidium `characters` / `invertCharacters` range filtering).
    Volume,
    /// Each character glyph moves as one intact piece with character-aligned piece index,
    /// exploding from the shared text center with staggered displacement/rotation and shared field.
    WholeCharacter,
    /// Each character glyph is fractured into `cells` shards with a character-aligned seed
    /// (`seed + char_index`) while sharing the whole-text field bounds.
    PerCharacterSubdivide,
}

/// Resolved per-effector character configuration for a `VoronoiFracture` node.
#[derive(Debug, Clone)]
pub(super) struct VoronoiCharacterConfig {
    pub(super) effector_idx: usize,
    pub(super) characters_spec: String,
    pub(super) invert_characters: bool,
}

/// Character identity metadata attached when evaluating a split `Text3D` glyph mesh.
#[derive(Debug, Clone, Copy)]
pub(super) struct CharacterFractureContext {
    /// 0-based local character index (`ch.index`: spaces count, newlines do not).
    pub(super) char_index: usize,
}

pub(super) fn parse_voronoi_character_mode(
    obj: &Object3D,
    env: &EvalEnv<'_>,
) -> VoronoiCharacterMode {
    if let Some(s) = eval_obj_str(
        obj,
        &[
            "fractureMode",
            "characterMode",
            "fracture_mode",
            "character_mode",
        ],
        env,
    ) {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "character" | "characters" | "whole-character" | "by-character" | "piece"
            | "pieces" | "1" => return VoronoiCharacterMode::WholeCharacter,
            "per-character" | "split-characters" | "subdivide-characters" | "2" => {
                return VoronoiCharacterMode::PerCharacterSubdivide;
            }
            "volume" | "mesh" | "whole" | "0" => return VoronoiCharacterMode::Volume,
            _ => {}
        }
    } else if let Some(track) = lookup_obj_prop_value(
        obj,
        &[
            "fractureMode",
            "characterMode",
            "fracture_mode",
            "character_mode",
        ],
    ) && let Some(n) = eval_property_track(&track, env).as_f64()
    {
        return match n.round() as i32 {
            1 => VoronoiCharacterMode::WholeCharacter,
            2 => VoronoiCharacterMode::PerCharacterSubdivide,
            _ => VoronoiCharacterMode::Volume,
        };
    }

    if eval_obj_bool(obj, &["byCharacter", "by_character"], false, env) {
        return VoronoiCharacterMode::WholeCharacter;
    }
    if eval_obj_bool(
        obj,
        &[
            "perCharacter",
            "per_character",
            "splitCharacters",
            "split_characters",
        ],
        false,
        env,
    ) {
        return VoronoiCharacterMode::PerCharacterSubdivide;
    }
    VoronoiCharacterMode::Volume
}

pub(super) fn parse_voronoi_character_config(
    obj: &Object3D,
    effector_idx: usize,
    env: &EvalEnv<'_>,
) -> Option<VoronoiCharacterConfig> {
    let characters_spec = eval_obj_str(
        obj,
        &["characters", "characterRanges", "character_ranges"],
        env,
    )
    .unwrap_or_default();
    let invert_characters =
        eval_obj_bool(obj, &["invertCharacters", "invert_characters"], false, env);
    let mode = parse_voronoi_character_mode(obj, env);
    // Matches Zoidium `effector-fracture-characters.js`: empty `characters` uses the whole-text
    // path even if `invertCharacters` is enabled, unless an explicit character fracture mode is set.
    if !characters_spec.trim().is_empty() || mode != VoronoiCharacterMode::Volume {
        Some(VoronoiCharacterConfig {
            effector_idx,
            characters_spec,
            invert_characters,
        })
    } else {
        None
    }
}

pub(super) fn union_aabb(a: Option<Aabb3>, b: Option<Aabb3>) -> Option<Aabb3> {
    match (a, b) {
        (Some(x), Some(y)) => Some(Aabb3 {
            min: x.min.min(y.min),
            max: x.max.max(y.max),
        }),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

pub(super) fn accumulate_effector_bounds(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    node_idx: usize,
    mesh: &MeshData,
    mesh_world: Mat4,
    effector_chain: &[usize],
    effector_kinds: &[Option<SceneEffectorKind>],
    acc: &mut HashMap<Uuid, (Aabb3, HashSet<usize>, usize)>,
) {
    if mesh.vertices.is_empty() {
        return;
    }
    for &e_idx in effector_chain {
        let Some(kind) = effector_kinds[e_idx] else {
            continue;
        };
        if !matches!(
            kind,
            SceneEffectorKind::Twist
                | SceneEffectorKind::Warp
                | SceneEffectorKind::Shear
                | SceneEffectorKind::Plain
                | SceneEffectorKind::VoronoiFracture
        ) {
            continue;
        }
        let e_node = &eval_graph.nodes[e_idx];
        let Some(df) = compute_deformer_frame(e_node.world_mat, mesh_world) else {
            continue;
        };
        let local_aabb = if df.identity {
            mesh.bounds
        } else {
            let mut min = Vec3::splat(f32::INFINITY);
            let mut max = Vec3::splat(f32::NEG_INFINITY);
            let mut any = false;
            for v in &mesh.vertices {
                let p = df.forward.transform_point3(Vec3::from_array(v.position));
                if p.is_finite() {
                    min = min.min(p);
                    max = max.max(p);
                    any = true;
                }
            }
            if !any {
                continue;
            }
            Aabb3 { min, max }
        };
        acc.entry(e_node.obj.id)
            .and_modify(|(b, nodes, count)| {
                b.min = b.min.min(local_aabb.min);
                b.max = b.max.max(local_aabb.max);
                nodes.insert(node_idx);
                *count += 1;
            })
            .or_insert_with(|| {
                let mut nodes = HashSet::new();
                nodes.insert(node_idx);
                (local_aabb, nodes, 1)
            });
    }
}

pub(super) fn transform_mesh_by_mat4(mesh: &MeshData, pos_mat: Mat4, normal_mat: Mat4) -> MeshData {
    let mut vertices = Vec::with_capacity(mesh.vertices.len());
    for v in &mesh.vertices {
        let pos = pos_mat
            .transform_point3(Vec3::from_array(v.position))
            .to_array();
        let n_in = Vec3::from_array(v.normal);
        let n_tx = normal_mat.transform_vector3(n_in);
        let normal = if n_tx.length_squared() > 1e-12 {
            n_tx.normalize().to_array()
        } else {
            v.normal
        };
        vertices.push(Vertex3D::new(pos, normal, v.uv, v.material_slot));
    }
    MeshData::with_colors(vertices, mesh.indices.clone(), mesh.colors.clone())
}

/// Builds a single-piece [`VoronoiFractureStage`] representing one intact character glyph
/// indexed by `char_index`, anchored around `shared_center` so `apply_voronoi_fragment_motion`
/// displaces, scatters, spins, scales, and colors each character with full Zoidium math.
pub(super) fn build_whole_character_voronoi_stage(
    mesh: &MeshData,
    char_index: usize,
    shared_center: Vec3,
    seed: i32,
    closed: bool,
) -> VoronoiFractureStage {
    let piece_count = char_index + 1;
    let mut sum = [0.0_f64; 3];
    for v in &mesh.vertices {
        sum[0] += f64::from(v.position[0]);
        sum[1] += f64::from(v.position[1]);
        sum[2] += f64::from(v.position[2]);
    }
    let centroid = if mesh.vertices.is_empty() {
        0.5 * (mesh.bounds.min + mesh.bounds.max)
    } else {
        let denom = mesh.vertices.len() as f64;
        Vec3::new(
            (sum[0] / denom) as f32,
            (sum[1] / denom) as f32,
            (sum[2] / denom) as f32,
        )
    };
    let mut piece_centroids = vec![Vec3::ZERO; piece_count];
    piece_centroids[char_index] = centroid;
    let c = voronoi_piece_color(seed, char_index as i32);
    let random_vertex_colors = vec![[c[0], c[1], c[2], 1.0]; mesh.vertices.len()];
    VoronoiFractureStage {
        rest_mesh: mesh.clone(),
        piece_ids: vec![char_index as u32; mesh.vertices.len()],
        piece_centroids,
        center: shared_center,
        seed,
        cells: piece_count as u32,
        closed,
        random_vertex_colors,
    }
}

fn get_or_build_voronoi_stage(
    mesh: &MeshData,
    params: &VoronoiFractureParams,
    voronoi_cache: &mut HashMap<u64, CachedVoronoiStage>,
    cache_tick: &mut u64,
) -> Arc<VoronoiFractureStage> {
    let topo_key = hash_voronoi_topology_key(mesh, params);
    *cache_tick = cache_tick.wrapping_add(1);
    let tick = *cache_tick;
    if let Some(entry) = voronoi_cache.get_mut(&topo_key) {
        entry.last_used = tick;
        return Arc::clone(&entry.stage);
    }
    let built = Arc::new(build_voronoi_fracture_stage(mesh, params));
    if voronoi_cache.len() >= 64
        && let Some((&oldest_key, _)) = voronoi_cache.iter().min_by_key(|(_, v)| v.last_used)
    {
        voronoi_cache.remove(&oldest_key);
    }
    voronoi_cache.insert(
        topo_key,
        CachedVoronoiStage {
            stage: Arc::clone(&built),
            last_used: tick,
        },
    );
    built
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_voronoi_to_instance(
    mesh: &MeshData,
    params: &VoronoiFractureParams,
    char_mode: VoronoiCharacterMode,
    char_ctx: Option<CharacterFractureContext>,
    field_shared_bounds: Option<Aabb3>,
    use_unified_grid: bool,
    df: Option<&DeformerFrame>,
    voronoi_cache: &mut HashMap<u64, CachedVoronoiStage>,
    cache_tick: &mut u64,
) -> MeshData {
    let Some(frame) = df else {
        return mesh.clone();
    };

    // Mode 1: Whole-character fracture on split Text3D glyphs.
    if char_mode == VoronoiCharacterMode::WholeCharacter
        && let Some(ctx) = char_ctx
    {
        let eff_mesh = if frame.identity {
            mesh.clone()
        } else {
            transform_mesh_by_mat4(mesh, frame.forward, frame.inverse.transpose())
        };
        let eff_bounds = field_shared_bounds.unwrap_or(eff_mesh.bounds);
        let shared_center = 0.5 * (eff_bounds.min + eff_bounds.max);
        let stage = build_whole_character_voronoi_stage(
            &eff_mesh,
            ctx.char_index,
            shared_center,
            params.seed,
            params.closed,
        );
        let deformed_eff = apply_voronoi_fragment_motion(&stage, params, Some(eff_bounds), None);
        return if frame.identity {
            deformed_eff
        } else {
            transform_mesh_by_mat4(&deformed_eff, frame.inverse, frame.forward.transpose())
        };
    }

    // Mode 2: Per-character Voronoi subdivision with character-aligned seed and shared field bounds.
    if char_mode == VoronoiCharacterMode::PerCharacterSubdivide
        && let Some(ctx) = char_ctx
    {
        let mut char_params = params.clone();
        char_params.seed = params.seed.wrapping_add(ctx.char_index as i32);
        let stage = get_or_build_voronoi_stage(mesh, &char_params, voronoi_cache, cache_tick);
        return apply_voronoi_fragment_motion(&stage, &char_params, field_shared_bounds, df);
    }

    // Mode 3: Multi-child unified Voronoi volume (e.g. `Voronoi -> Group -> [Obj1, Obj2]` or
    // repeated instances): transform into effector space and override `bounds = shared_bounds` so
    // sibling meshes share the exact same Voronoi cell grid, bisector planes, and explosion center.
    if use_unified_grid && let Some(unified_bounds) = field_shared_bounds {
        let mut eff_mesh = if frame.identity {
            mesh.clone()
        } else {
            transform_mesh_by_mat4(mesh, frame.forward, frame.inverse.transpose())
        };
        eff_mesh.bounds = unified_bounds;
        let stage = get_or_build_voronoi_stage(&eff_mesh, params, voronoi_cache, cache_tick);
        let deformed_eff =
            apply_voronoi_fragment_motion(&stage, params, Some(unified_bounds), None);
        return if frame.identity {
            deformed_eff
        } else {
            transform_mesh_by_mat4(&deformed_eff, frame.inverse, frame.forward.transpose())
        };
    }

    // Mode 4: Single-mesh volume fracture (or single-Text3D `characters` selection with shared field bounds).
    let stage = get_or_build_voronoi_stage(mesh, params, voronoi_cache, cache_tick);
    apply_voronoi_fragment_motion(&stage, params, field_shared_bounds, df)
}

pub(super) fn env_at_frame<'a>(env: &EvalEnv<'a>, target_frame: f64) -> EvalEnv<'a> {
    let clamped = target_frame.max(0.0);
    let delta = clamped - env.frame;
    EvalEnv {
        frame: clamped,
        abs_frame: (env.abs_frame + delta).max(0.0),
        time: clamped / env.fps.max(1.0),
        ..env.clone()
    }
}

fn hash_voronoi_topology_key(mesh: &MeshData, params: &VoronoiFractureParams) -> u64 {
    let mut h = fnv1a_hash64(b"voronoi_stage_v2");
    h = hash_u64(h, mesh.vertices.len() as u64);
    h = hash_u64(h, mesh.indices.len() as u64);
    h = hash_u64(h, u64::from(params.cells));
    h = hash_u64(h, params.seed as u64);
    h = hash_u64(h, u64::from(params.closed));
    h = hash_u64(h, u64::from(params.distribution as u32));
    h = hash_u64(h, u64::from(params.cell_scale.x.to_bits()));
    h = hash_u64(h, u64::from(params.cell_scale.y.to_bits()));
    h = hash_u64(h, u64::from(params.cell_scale.z.to_bits()));
    h = hash_u64(h, u64::from(mesh.bounds.min.x.to_bits()));
    h = hash_u64(h, u64::from(mesh.bounds.min.y.to_bits()));
    h = hash_u64(h, u64::from(mesh.bounds.min.z.to_bits()));
    h = hash_u64(h, u64::from(mesh.bounds.max.x.to_bits()));
    h = hash_u64(h, u64::from(mesh.bounds.max.y.to_bits()));
    h = hash_u64(h, u64::from(mesh.bounds.max.z.to_bits()));
    let stride = (mesh.vertices.len() / 64).max(1);
    for v in mesh.vertices.iter().step_by(stride) {
        h = hash_u64(h, u64::from(v.position[0].to_bits()));
        h = hash_u64(h, u64::from(v.position[1].to_bits()));
        h = hash_u64(h, u64::from(v.position[2].to_bits()));
        h = hash_u64(h, u64::from(v.material_slot));
    }
    h
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_scene_effector_chain(
    objects: &[Object3D],
    parent_world: Mat4,
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    node_idx: usize,
    rest_mesh: Arc<MeshData>,
    initial_world: Mat4,
    effector_chain: &[usize],
    effector_kinds: &[Option<SceneEffectorKind>],
    shared_effector_bounds: &HashMap<Uuid, SharedEffectorBounds>,
    char_ctx: Option<CharacterFractureContext>,
    env: &EvalEnv<'_>,
    voronoi_cache: &mut HashMap<u64, CachedVoronoiStage>,
    cache_tick: &mut u64,
) -> Vec<DeformedInstance> {
    let mut instances = vec![DeformedInstance {
        mesh: Arc::clone(&rest_mesh),
        world: initial_world,
    }];
    let mut upstream_deformers: Vec<usize> = Vec::new();
    let mut curved_upstream = false;
    let node_world = eval_graph.nodes[node_idx].world_mat;
    let initial_local_offset = if node_world.determinant().abs() > 1e-9 {
        node_world.inverse() * initial_world
    } else {
        Mat4::IDENTITY
    };

    for (step_idx, &e_idx) in effector_chain.iter().enumerate() {
        let Some(kind) = effector_kinds[e_idx] else {
            continue;
        };
        let e_node = &eval_graph.nodes[e_idx];
        let e_obj_env = env.with_object(e_node.obj.id);
        let e_env = e_obj_env.with_parent_props(&e_node.parent_props);
        let e_world = e_node.world_mat;

        match kind {
            SceneEffectorKind::Repeater(mode) => {
                let rep = parse_repeater_params(e_node.obj, mode, &e_env);
                if rep.mode == RepeaterMode::Echo {
                    let frames = repeater_echo_frames(env.frame, rep.count, rep.delay_frames);
                    let mut next = Vec::with_capacity(frames.len() * instances.len());
                    let mut echo_frame_cache: HashMap<u64, Vec<DeformedInstance>> = HashMap::new();
                    let target_uuid = eval_graph.nodes[node_idx].obj.id;
                    for ef in frames {
                        if (ef - env.frame).abs() < 1e-6 {
                            for inst in &instances {
                                next.push(DeformedInstance {
                                    mesh: Arc::clone(&inst.mesh),
                                    world: inst.world,
                                });
                            }
                            continue;
                        }
                        let frame_key = ef.to_bits();
                        if let Some(cached) = echo_frame_cache.get(&frame_key) {
                            for inst in cached {
                                next.push(DeformedInstance {
                                    mesh: Arc::clone(&inst.mesh),
                                    world: inst.world,
                                });
                            }
                            continue;
                        }
                        let echo_env = env_at_frame(env, ef);
                        let echo_graph = evaluate_scene_graph_3d(objects, parent_world, &echo_env);
                        let echo_e_world = echo_graph
                            .by_uuid
                            .get(&e_node.obj.id)
                            .map_or(e_world, |&ei| echo_graph.nodes[ei].world_mat);
                        let inv_echo_e = if echo_e_world.determinant().abs() > 1e-9 {
                            echo_e_world.inverse()
                        } else {
                            Mat4::IDENTITY
                        };
                        // Anchor historical child transforms into the Echo Repeater's current frame
                        // (matching Zoidium `scene-repeaters.js:rebuildEchoInstances`).
                        let anchor_delta = e_world * inv_echo_e;
                        let echo_child_world = echo_graph
                            .by_uuid
                            .get(&target_uuid)
                            .map_or(initial_world, |&i| {
                                echo_graph.nodes[i].world_mat * initial_local_offset
                            });
                        let evaluated = if step_idx == 0 {
                            vec![DeformedInstance {
                                mesh: Arc::clone(&rest_mesh),
                                world: anchor_delta * echo_child_world,
                            }]
                        } else {
                            let echo_upstream: Vec<usize> = effector_chain[..step_idx]
                                .iter()
                                .filter_map(|&u_idx| {
                                    echo_graph
                                        .by_uuid
                                        .get(&eval_graph.nodes[u_idx].obj.id)
                                        .copied()
                                })
                                .collect();
                            let echo_kinds: Vec<Option<SceneEffectorKind>> = echo_graph
                                .nodes
                                .iter()
                                .map(|n| detect_scene_effector_kind(n.obj))
                                .collect();
                            let echo_node_idx =
                                echo_graph.by_uuid.get(&target_uuid).copied().unwrap_or(0);
                            let mut echo_insts = apply_scene_effector_chain(
                                objects,
                                parent_world,
                                &echo_graph,
                                echo_node_idx,
                                Arc::clone(&rest_mesh),
                                echo_child_world,
                                &echo_upstream,
                                &echo_kinds,
                                shared_effector_bounds,
                                char_ctx,
                                &echo_env,
                                voronoi_cache,
                                cache_tick,
                            );
                            for inst in &mut echo_insts {
                                inst.world = anchor_delta * inst.world;
                            }
                            echo_insts
                        };
                        for inst in &evaluated {
                            next.push(DeformedInstance {
                                mesh: Arc::clone(&inst.mesh),
                                world: inst.world,
                            });
                        }
                        echo_frame_cache.insert(frame_key, evaluated);
                    }
                    instances = next;
                } else {
                    let count = rep.count.clamp(1, MAX_REPEATS);
                    let inv_r = if e_world.determinant().abs() > 1e-9 {
                        e_world.inverse()
                    } else {
                        Mat4::IDENTITY
                    };
                    let mut next = Vec::with_capacity(instances.len() * count as usize);
                    for inst in &instances {
                        let rel = inv_r * inst.world;
                        for copy_i in 0..count {
                            let tr = repeater_instance_transform(&rep, copy_i);
                            let copy_local = Transform3D {
                                position: tr.position,
                                rotation: tr.rotation_rad,
                                scale: tr.scale,
                                euler_order: rep.euler_order,
                            }
                            .to_matrix();
                            next.push(DeformedInstance {
                                mesh: Arc::clone(&inst.mesh),
                                world: e_world * copy_local * rel,
                            });
                        }
                    }
                    instances = next;
                }
            }
            SceneEffectorKind::RadialRepeater => {
                let count = eval_obj_u32(e_node.obj, &["count", "copies", "repeats"], 8, &e_env)
                    .clamp(1, MAX_REPEATS);
                let has_sweep = lookup_obj_prop_value(e_node.obj, &["sweep"]).is_some();
                let has_end_angle =
                    lookup_obj_prop_value(e_node.obj, &["endAngle", "end_angle"]).is_some();
                let zoidium_radial = has_sweep && !has_end_angle;
                let radius = eval_obj_f32(
                    e_node.obj,
                    &["radius"],
                    if zoidium_radial { 100.0 } else { 20.0 },
                    &e_env,
                );
                let radians_mode = e_node
                    .obj
                    .preserved
                    .get("zoidiumRadialRadians")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(zoidium_radial);
                let raw_start =
                    eval_obj_f32(e_node.obj, &["startAngle", "start_angle"], 0.0, &e_env);
                let start_angle_rad = if radians_mode {
                    raw_start
                } else {
                    raw_start.to_radians()
                };
                let span_rad = if zoidium_radial {
                    let raw_sweep =
                        eval_obj_f32(e_node.obj, &["sweep"], std::f32::consts::TAU, &e_env);
                    if radians_mode {
                        raw_sweep
                    } else {
                        raw_sweep.to_radians()
                    }
                } else {
                    let raw_end =
                        eval_obj_f32(e_node.obj, &["endAngle", "end_angle"], 360.0, &e_env);
                    let end_rad = if radians_mode {
                        raw_end
                    } else {
                        raw_end.to_radians()
                    };
                    end_rad - start_angle_rad
                };
                let default_align = u32::from(!zoidium_radial);
                let align_to_circle = if let Some(s) = eval_obj_str(
                    e_node.obj,
                    &["alignToCircle", "align_to_circle", "align"],
                    &e_env,
                ) {
                    !matches!(
                        s.trim().to_ascii_lowercase().as_str(),
                        "0" | "false" | "no" | "off"
                    )
                } else {
                    eval_obj_u32(
                        e_node.obj,
                        &["alignToCircle", "align_to_circle", "align"],
                        default_align,
                        &e_env,
                    ) != 0
                };
                // Axis: 0 = X (YZ plane), 1 = Y (XZ plane), 2 = Z (XY plane).
                // Zoidium `plane`: 0 = XY (axis 2), 1 = XZ (axis 1, default), 2 = YZ (axis 0).
                let axis = if lookup_obj_prop_value(e_node.obj, &["axis"]).is_some() {
                    if let Some(s) = eval_obj_str(e_node.obj, &["axis"], &e_env) {
                        match s.trim().to_ascii_lowercase().as_str() {
                            "x" | "0" => 0u32,
                            "y" | "1" => 1u32,
                            _ => 2u32,
                        }
                    } else {
                        eval_obj_u32(e_node.obj, &["axis"], 2, &e_env).min(2)
                    }
                } else if lookup_obj_prop_value(e_node.obj, &["plane"]).is_some() {
                    match eval_obj_u32(e_node.obj, &["plane"], 1, &e_env) {
                        0 => 2u32, // XY -> around Z
                        2 => 0u32, // YZ -> around X
                        _ => 1u32, // XZ -> around Y
                    }
                } else if zoidium_radial {
                    1u32
                } else {
                    2u32
                };
                let rot_step = eval_obj_vec3(
                    e_node.obj,
                    &["rotationStep", "rotation_step"],
                    Vec3::ZERO,
                    &e_env,
                );
                let scale_end =
                    eval_obj_vec3(e_node.obj, &["scaleEnd", "scale_end"], Vec3::ONE, &e_env);
                let full_circle = (span_rad.abs() - std::f32::consts::TAU).abs() < 1e-3;
                let inv_r = if e_world.determinant().abs() > 1e-9 {
                    e_world.inverse()
                } else {
                    Mat4::IDENTITY
                };
                let euler_order = map_euler_order(e_node.obj.euler_order);
                let mut next = Vec::with_capacity(instances.len() * count as usize);
                for inst in &instances {
                    let rel = inv_r * inst.world;
                    for copy_i in 0..count {
                        let angle_t = if count <= 1 {
                            0.0
                        } else if full_circle {
                            (copy_i as f32) / (count as f32)
                        } else {
                            (copy_i as f32) / ((count - 1) as f32)
                        };
                        let scale_t = if count <= 1 {
                            0.0
                        } else {
                            (copy_i as f32) / ((count - 1) as f32)
                        };
                        let a = start_angle_rad + span_rad * angle_t;
                        let (s, c) = a.sin_cos();
                        let (pos, base_rot) = match axis {
                            0 => (
                                Vec3::new(0.0, c * radius, s * radius),
                                if align_to_circle {
                                    Vec3::new(a, 0.0, 0.0)
                                } else {
                                    Vec3::ZERO
                                },
                            ),
                            1 => (
                                Vec3::new(c * radius, 0.0, s * radius),
                                if align_to_circle {
                                    Vec3::new(0.0, -a, 0.0)
                                } else {
                                    Vec3::ZERO
                                },
                            ),
                            _ => (
                                Vec3::new(c * radius, s * radius, 0.0),
                                if align_to_circle {
                                    Vec3::new(0.0, 0.0, a)
                                } else {
                                    Vec3::ZERO
                                },
                            ),
                        };
                        let rot = base_rot + rot_step * (copy_i as f32);
                        let scale = Vec3::ONE.lerp(scale_end, scale_t);
                        let copy_local = Transform3D {
                            position: pos,
                            rotation: rot,
                            scale,
                            euler_order,
                        }
                        .to_matrix();
                        next.push(DeformedInstance {
                            mesh: Arc::clone(&inst.mesh),
                            world: e_world * copy_local * rel,
                        });
                    }
                }
                instances = next;
            }
            SceneEffectorKind::GridRepeater => {
                let nx = eval_obj_u32(e_node.obj, &["countX", "count_x", "xCount"], 3, &e_env)
                    .clamp(1, 32);
                let ny = eval_obj_u32(e_node.obj, &["countY", "count_y", "yCount"], 3, &e_env)
                    .clamp(1, 32);
                let nz = eval_obj_u32(e_node.obj, &["countZ", "count_z", "zCount"], 1, &e_env)
                    .clamp(1, 32);
                let default_sp = eval_obj_f32(e_node.obj, &["spacing"], 12.0, &e_env);
                let spacing_v = eval_obj_vec3(
                    e_node.obj,
                    &["spacing", "gridSpacing", "step"],
                    Vec3::splat(default_sp),
                    &e_env,
                );
                let sp_x =
                    eval_obj_f32(e_node.obj, &["spacingX", "spacing_x"], spacing_v.x, &e_env);
                let sp_y =
                    eval_obj_f32(e_node.obj, &["spacingY", "spacing_y"], spacing_v.y, &e_env);
                let sp_z =
                    eval_obj_f32(e_node.obj, &["spacingZ", "spacing_z"], spacing_v.z, &e_env);
                let centered = eval_obj_bool(e_node.obj, &["centered"], true, &e_env);
                let inv_r = if e_world.determinant().abs() > 1e-9 {
                    e_world.inverse()
                } else {
                    Mat4::IDENTITY
                };
                let total_cap = (nx * ny * nz).min(MAX_REPEATS) as usize;
                let mut next = Vec::with_capacity(instances.len() * total_cap);
                for inst in &instances {
                    let rel = inv_r * inst.world;
                    let mut emitted = 0usize;
                    let center_x = if centered {
                        (nx as f32 - 1.0) * 0.5
                    } else {
                        0.0
                    };
                    let center_y = if centered {
                        (ny as f32 - 1.0) * 0.5
                    } else {
                        0.0
                    };
                    let center_z = if centered {
                        (nz as f32 - 1.0) * 0.5
                    } else {
                        0.0
                    };
                    'grid: for iz in 0..nz {
                        let oz = (iz as f32 - center_z) * sp_z;
                        for iy in 0..ny {
                            let oy = (iy as f32 - center_y) * sp_y;
                            for ix in 0..nx {
                                if emitted >= total_cap {
                                    break 'grid;
                                }
                                let ox = (ix as f32 - center_x) * sp_x;
                                let copy_local = Mat4::from_translation(Vec3::new(ox, oy, oz));
                                next.push(DeformedInstance {
                                    mesh: Arc::clone(&inst.mesh),
                                    world: e_world * copy_local * rel,
                                });
                                emitted += 1;
                            }
                        }
                    }
                }
                instances = next;
            }
            SceneEffectorKind::Twist => {
                let params = parse_twist_params(e_node.obj, &e_env);
                let group_bounds = shared_effector_bounds
                    .get(&e_node.obj.id)
                    .filter(|entry| entry.mesh_count >= 2)
                    .map(|entry| entry.bounds);
                let shared_bounds = union_aabb(
                    group_bounds,
                    compute_instances_shared_bounds(&instances, e_world),
                );
                for inst in &mut instances {
                    let Some(df) = compute_deformer_frame(e_world, inst.world) else {
                        continue;
                    };
                    let deformed =
                        apply_twist_with_params(&inst.mesh, &params, shared_bounds, Some(&df));
                    inst.mesh = Arc::new(deformed);
                }
                upstream_deformers.push(e_idx);
                curved_upstream = true;
            }
            SceneEffectorKind::Warp => {
                let params = parse_warp_params(e_node.obj, &e_env);
                let group_bounds = shared_effector_bounds
                    .get(&e_node.obj.id)
                    .filter(|entry| entry.mesh_count >= 2)
                    .map(|entry| entry.bounds);
                let shared_bounds = union_aabb(
                    group_bounds,
                    compute_instances_shared_bounds(&instances, e_world),
                );
                for inst in &mut instances {
                    let Some(df) = compute_deformer_frame(e_world, inst.world) else {
                        continue;
                    };
                    let deformed =
                        apply_warp_with_params(&inst.mesh, &params, shared_bounds, Some(&df));
                    inst.mesh = Arc::new(deformed);
                }
                upstream_deformers.push(e_idx);
                curved_upstream = true;
            }
            SceneEffectorKind::Shear => {
                let params = parse_shear_params(e_node.obj, &e_env);
                let group_bounds = shared_effector_bounds
                    .get(&e_node.obj.id)
                    .filter(|entry| entry.mesh_count >= 2)
                    .map(|entry| entry.bounds);
                let shared_bounds = union_aabb(
                    group_bounds,
                    compute_instances_shared_bounds(&instances, e_world),
                );
                for inst in &mut instances {
                    let Some(df) = compute_deformer_frame(e_world, inst.world) else {
                        continue;
                    };
                    let deformed =
                        apply_shear_with_params(&inst.mesh, &params, shared_bounds, Some(&df));
                    inst.mesh = Arc::new(deformed);
                }
                upstream_deformers.push(e_idx);
                curved_upstream = true;
            }
            SceneEffectorKind::Plain => {
                let params = parse_plain_params(e_node.obj, &e_env);
                let group_bounds = shared_effector_bounds
                    .get(&e_node.obj.id)
                    .filter(|entry| entry.mesh_count >= 2)
                    .map(|entry| entry.bounds);
                let shared_bounds = union_aabb(
                    group_bounds,
                    compute_instances_shared_bounds(&instances, e_world),
                );
                for inst in &mut instances {
                    let Some(df) = compute_deformer_frame(e_world, inst.world) else {
                        continue;
                    };
                    let deformed = apply_plain(&inst.mesh, &params, shared_bounds, Some(&df));
                    inst.mesh = Arc::new(deformed);
                }
                upstream_deformers.push(e_idx);
            }
            SceneEffectorKind::VoronoiFracture => {
                let params = parse_voronoi_params(e_node.obj, &e_env);
                let char_mode = parse_voronoi_character_mode(e_node.obj, &e_env);
                let group_entry = shared_effector_bounds.get(&e_node.obj.id).copied();
                let group_bounds = group_entry
                    .filter(|entry| entry.mesh_count >= 2)
                    .map(|entry| entry.bounds);
                let inst_bounds = compute_instances_shared_bounds(&instances, e_world);
                let field_shared_bounds = union_aabb(group_bounds, inst_bounds);
                let use_unified_grid =
                    group_entry.is_some_and(|e| e.distinct_nodes >= 2) || instances.len() > 1;
                for inst in &mut instances {
                    let df = compute_deformer_frame(e_world, inst.world);
                    let deformed = apply_voronoi_to_instance(
                        &inst.mesh,
                        &params,
                        char_mode,
                        char_ctx,
                        field_shared_bounds,
                        use_unified_grid,
                        df.as_ref(),
                        voronoi_cache,
                        cache_tick,
                    );
                    inst.mesh = Arc::new(deformed);
                }
                upstream_deformers.push(e_idx);
            }
            SceneEffectorKind::Delay => {
                let params = parse_delay_params(e_node.obj, &e_env);
                let window = params.window.clamp(1, 120);
                if params.strength > 0.0 && env.frame > 0.0 && !upstream_deformers.is_empty() {
                    let target_uuid = eval_graph.nodes[node_idx].obj.id;
                    let mut history_by_inst: Vec<Vec<MeshData>> =
                        vec![Vec::with_capacity(window as usize); instances.len()];
                    for step in 1..=window {
                        let hf = (env.frame - f64::from(step)).max(0.0);
                        let h_env = env_at_frame(env, hf);
                        let h_graph = evaluate_scene_graph_3d(objects, parent_world, &h_env);
                        let h_world = h_graph
                            .by_uuid
                            .get(&target_uuid)
                            .map_or(initial_world, |&i| {
                                h_graph.nodes[i].world_mat * initial_local_offset
                            });
                        let h_upstream: Vec<usize> = upstream_deformers
                            .iter()
                            .filter_map(|&u_idx| {
                                h_graph
                                    .by_uuid
                                    .get(&eval_graph.nodes[u_idx].obj.id)
                                    .copied()
                            })
                            .collect();
                        let h_kinds: Vec<Option<SceneEffectorKind>> = h_graph
                            .nodes
                            .iter()
                            .map(|n| detect_scene_effector_kind(n.obj))
                            .collect();
                        let h_node_idx = h_graph.by_uuid.get(&target_uuid).copied().unwrap_or(0);
                        let h_insts = apply_scene_effector_chain(
                            objects,
                            parent_world,
                            &h_graph,
                            h_node_idx,
                            Arc::clone(&rest_mesh),
                            h_world,
                            &h_upstream,
                            &h_kinds,
                            shared_effector_bounds,
                            char_ctx,
                            &h_env,
                            voronoi_cache,
                            cache_tick,
                        );
                        for (slot, hi) in history_by_inst.iter_mut().zip(h_insts) {
                            slot.push((*hi.mesh).clone());
                        }
                    }
                    for (inst, hist) in instances.iter_mut().zip(history_by_inst.iter()) {
                        let hist_refs: Vec<&MeshData> = hist.iter().collect();
                        let ref_rest = if rest_mesh.vertices.len() == inst.mesh.vertices.len() {
                            rest_mesh.as_ref()
                        } else {
                            inst.mesh.as_ref()
                        };
                        let delayed = apply_delay(
                            ref_rest,
                            inst.mesh.as_ref(),
                            &hist_refs,
                            &params,
                            curved_upstream,
                        );
                        inst.mesh = Arc::new(delayed);
                    }
                }
            }
        }
    }

    instances
}

pub(super) fn compute_instances_shared_bounds(
    instances: &[DeformedInstance],
    deformer_world: Mat4,
) -> Option<Aabb3> {
    if instances.len() <= 1 {
        return None;
    }
    let pairs: Vec<(&MeshData, Mat4)> = instances
        .iter()
        .map(|i| (i.mesh.as_ref(), i.world))
        .collect();
    compute_shared_deformer_bounds(&pairs, deformer_world)
}

pub(super) fn json_to_property_track(
    val: &serde_json::Value,
    default: PropertyValue,
) -> PropertyTrack {
    match val {
        serde_json::Value::Number(n) => PropertyTrack::number(
            n.as_f64()
                .unwrap_or_else(|| default.as_f64().unwrap_or(0.0)),
        ),
        serde_json::Value::Bool(b) => PropertyTrack::constant(PropertyValue::Bool(*b)),
        serde_json::Value::String(s) => PropertyTrack::text(s.clone()),
        serde_json::Value::Array(arr) => {
            if arr.len() >= 3 {
                PropertyTrack::vec3([
                    arr[0].as_f64().unwrap_or(0.0),
                    arr[1].as_f64().unwrap_or(0.0),
                    arr[2].as_f64().unwrap_or(0.0),
                ])
            } else if arr.len() == 2 {
                PropertyTrack::vec2([
                    arr[0].as_f64().unwrap_or(0.0),
                    arr[1].as_f64().unwrap_or(0.0),
                ])
            } else if let Some(v) = arr.first().and_then(serde_json::Value::as_f64) {
                PropertyTrack::number(v)
            } else {
                PropertyTrack::constant(default)
            }
        }
        serde_json::Value::Object(map) => {
            if let Ok(track) = serde_json::from_value::<PropertyTrack>(val.clone()) {
                return track;
            }
            if let (Some(x), Some(y), Some(z)) = (
                map.get("x").and_then(serde_json::Value::as_f64),
                map.get("y").and_then(serde_json::Value::as_f64),
                map.get("z").and_then(serde_json::Value::as_f64),
            ) {
                return PropertyTrack::vec3([x, y, z]);
            }
            // CM3 Vector3 serialized as `{ "objects": [x_track, y_track, z_track] }` (preserving per-axis keyframes!).
            if let Some(objs_val) = map.get("objects") {
                let items: Vec<&serde_json::Value> = match objs_val {
                    serde_json::Value::Array(arr) => arr.iter().collect(),
                    serde_json::Value::Object(om) => {
                        let mut indexed: Vec<(usize, &serde_json::Value)> = om
                            .iter()
                            .filter_map(|(k, v)| k.parse::<usize>().ok().map(|i| (i, v)))
                            .collect();
                        indexed.sort_by_key(|(i, _)| *i);
                        indexed.into_iter().map(|(_, v)| v).collect()
                    }
                    _ => Vec::new(),
                };
                if items.len() >= 3 {
                    let tx = json_to_property_track(items[0], PropertyValue::Number(0.0));
                    let ty = json_to_property_track(items[1], PropertyValue::Number(0.0));
                    let tz = json_to_property_track(items[2], PropertyValue::Number(0.0));
                    let base_v = [
                        tx.value.as_f64().unwrap_or(0.0),
                        ty.value.as_f64().unwrap_or(0.0),
                        tz.value.as_f64().unwrap_or(0.0),
                    ];
                    let any_anim = tx.animated
                        || ty.animated
                        || tz.animated
                        || !tx.keyframes.is_empty()
                        || !ty.keyframes.is_empty()
                        || !tz.keyframes.is_empty();
                    if !any_anim {
                        return PropertyTrack::vec3(base_v);
                    }
                    let mut times: Vec<f64> = Vec::new();
                    for tr in [&tx, &ty, &tz] {
                        for kf in &tr.keyframes {
                            if kf.frame.is_finite()
                                && !times.iter().any(|&t| (t - kf.frame).abs() < 1e-6)
                            {
                                times.push(kf.frame);
                            }
                        }
                    }
                    times.sort_by(f64::total_cmp);
                    let mut keyframes = Vec::with_capacity(times.len());
                    for frame in times {
                        let vx = eval_track_keyframes_only(&tx, frame)
                            .as_f64()
                            .unwrap_or(base_v[0]);
                        let vy = eval_track_keyframes_only(&ty, frame)
                            .as_f64()
                            .unwrap_or(base_v[1]);
                        let vz = eval_track_keyframes_only(&tz, frame)
                            .as_f64()
                            .unwrap_or(base_v[2]);
                        keyframes.push(PropertyKeyframe::from_cm3_tween(
                            frame,
                            PropertyValue::Vec3([vx, vy, vz]),
                            1,
                            [[-10.0, 0.0], [10.0, 0.0]],
                            true,
                        ));
                    }
                    return PropertyTrack {
                        value: PropertyValue::Vec3(base_v),
                        animated: true,
                        keyframes,
                        channels: Vec::new(),
                        expression: None,
                        static_behavior: ProjectStaticBehavior::default(),
                    };
                }
            }
            let explicit_base = map
                .get("value")
                .or_else(|| map.get("v"))
                .map(|v| json_to_property_value(v, &default));
            let animated = map
                .get("animated")
                .or_else(|| map.get("a"))
                .is_some_and(|v| parse_preserved_bool_flag(Some(v), false));
            let expression = map
                .get("expression")
                .or_else(|| map.get("e"))
                .and_then(serde_json::Value::as_str)
                .filter(|s| !s.trim().is_empty())
                .map(ToOwned::to_owned);
            let mut keyframes = Vec::new();
            if let Some(kf_arr) = map
                .get("keyframes")
                .or_else(|| map.get("k"))
                .and_then(serde_json::Value::as_array)
            {
                for kf in kf_arr {
                    if let Some(kobj) = kf.as_object()
                        && let Some(frame) = kobj
                            .get("frame")
                            .or_else(|| kobj.get("t"))
                            .or_else(|| kobj.get("f"))
                            .and_then(serde_json::Value::as_f64)
                    {
                        let kv = kobj.get("value").or_else(|| kobj.get("v")).map_or_else(
                            || explicit_base.clone().unwrap_or_else(|| default.clone()),
                            |v| json_to_property_value(v, &default),
                        );
                        let tween = kobj
                            .get("tween")
                            .or_else(|| kobj.get("e"))
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(1) as u32;
                        let mut cp = [[-10.0, 0.0], [10.0, 0.0]];
                        if let Some(cparr) = kobj
                            .get("controlPoints")
                            .or_else(|| kobj.get("cp"))
                            .or_else(|| kobj.get("handles"))
                            .and_then(serde_json::Value::as_array)
                            && cparr.len() >= 2
                        {
                            for i in 0..2 {
                                if let Some(pair) = cparr[i].as_array()
                                    && pair.len() >= 2
                                {
                                    cp[i][0] = pair[0].as_f64().unwrap_or(cp[i][0]);
                                    cp[i][1] = pair[1].as_f64().unwrap_or(cp[i][1]);
                                }
                            }
                        }
                        keyframes
                            .push(PropertyKeyframe::from_cm3_tween(frame, kv, tween, cp, true));
                    }
                }
            }
            let mut channels = Vec::new();
            if let Some(ch_arr) = map
                .get("channels")
                .or_else(|| map.get("objects"))
                .or_else(|| map.get("p"))
                .and_then(serde_json::Value::as_array)
            {
                for ch in ch_arr {
                    channels.push(json_to_property_track(ch, PropertyValue::Number(0.0)));
                }
            }
            let base_val = if let Some(v) = explicit_base {
                v
            } else if !channels.is_empty() {
                let vals: Vec<f64> = channels
                    .iter()
                    .map(|c| c.value.as_f64().unwrap_or(0.0))
                    .collect();
                match vals.as_slice() {
                    [x, y, z, w, ..] => PropertyValue::Vec4([*x, *y, *z, *w]),
                    [x, y, z] => PropertyValue::Vec3([*x, *y, *z]),
                    [x, y] => PropertyValue::Vec2([*x, *y]),
                    [x] => PropertyValue::Number(*x),
                    _ => default,
                }
            } else if let Some(first_kf) = keyframes
                .iter()
                .find(|k| k.frame.abs() < 1e-6)
                .or_else(|| keyframes.first())
            {
                first_kf.value.clone()
            } else {
                default
            };
            let any_channel_animated = channels
                .iter()
                .any(|c| c.animated || c.expression.is_some() || !c.keyframes.is_empty());
            PropertyTrack {
                animated: animated
                    || expression.is_some()
                    || !keyframes.is_empty()
                    || any_channel_animated,
                expression,
                static_behavior: ProjectStaticBehavior::Constant,
                value: base_val,
                keyframes,
                channels,
            }
        }
        serde_json::Value::Null => PropertyTrack::constant(default),
    }
}

pub(super) fn json_to_property_value(
    val: &serde_json::Value,
    default: &PropertyValue,
) -> PropertyValue {
    match val {
        serde_json::Value::Number(n) => PropertyValue::Number(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::Bool(b) => PropertyValue::Bool(*b),
        serde_json::Value::String(s) => PropertyValue::Text(s.clone()),
        serde_json::Value::Array(arr) if arr.len() >= 3 => PropertyValue::Vec3([
            arr[0].as_f64().unwrap_or(0.0),
            arr[1].as_f64().unwrap_or(0.0),
            arr[2].as_f64().unwrap_or(0.0),
        ]),
        serde_json::Value::Array(arr) if arr.len() == 2 => PropertyValue::Vec2([
            arr[0].as_f64().unwrap_or(0.0),
            arr[1].as_f64().unwrap_or(0.0),
        ]),
        _ => default.clone(),
    }
}

pub(super) fn lookup_obj_prop_value(obj: &Object3D, keys: &[&str]) -> Option<PropertyTrack> {
    if let Object3DKind::Group {
        custom_properties, ..
    } = &obj.kind
    {
        for &key in keys {
            if let Some(cp) = custom_properties
                .iter()
                .find(|cp| cp.name.eq_ignore_ascii_case(key))
            {
                return Some(cp.track.clone());
            }
        }
    }
    for &key in keys {
        let prop_key = format!("property.{key}");
        if let Some(v) = obj.preserved.get(&prop_key) {
            return Some(json_to_property_track(v, PropertyValue::Number(0.0)));
        }
        for container in [
            "characterProperties",
            "repeaterProperties",
            "effectorProperties",
            "fieldProperties",
            "properties",
        ] {
            if let Some(m) = obj
                .preserved
                .get(container)
                .and_then(serde_json::Value::as_object)
                && let Some(v) = m.get(key).or_else(|| {
                    m.iter()
                        .find(|(k, _)| k.eq_ignore_ascii_case(key))
                        .map(|(_, val)| val)
                })
            {
                return Some(json_to_property_track(v, PropertyValue::Number(0.0)));
            }
        }
        if let Some(v) = obj.preserved.get(key) {
            return Some(json_to_property_track(v, PropertyValue::Number(0.0)));
        }
        if let Some((_, v)) = obj.preserved.iter().find(|(k, _)| {
            k.strip_prefix("property.")
                .is_some_and(|stripped| stripped.eq_ignore_ascii_case(key))
        }) {
            return Some(json_to_property_track(v, PropertyValue::Number(0.0)));
        }
    }
    None
}

pub(super) fn eval_obj_f32(obj: &Object3D, keys: &[&str], default: f32, env: &EvalEnv<'_>) -> f32 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_f64())
        .map_or(default, |v| v as f32)
}

pub(super) fn eval_obj_f64(obj: &Object3D, keys: &[&str], default: f64, env: &EvalEnv<'_>) -> f64 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_f64())
        .unwrap_or(default)
}

pub(super) fn eval_obj_u32(obj: &Object3D, keys: &[&str], default: u32, env: &EvalEnv<'_>) -> u32 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_f64())
        .map_or(default, |v| v.round().max(0.0) as u32)
}

pub(super) fn eval_obj_i32(obj: &Object3D, keys: &[&str], default: i32, env: &EvalEnv<'_>) -> i32 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_f64())
        .map_or(default, |v| v.round() as i32)
}

pub(super) fn eval_obj_bool(
    obj: &Object3D,
    keys: &[&str],
    default: bool,
    env: &EvalEnv<'_>,
) -> bool {
    lookup_obj_prop_value(obj, keys).map_or(default, |t| match eval_property_track(&t, env) {
        PropertyValue::Bool(b) => b,
        PropertyValue::Option(o) => o != 0,
        PropertyValue::Number(n) => n.abs() > 1e-6,
        PropertyValue::Text(s) | PropertyValue::List(s) => {
            match s.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "on" | "yes" => true,
                "0" | "false" | "off" | "no" => false,
                _ => default,
            }
        }
        _ => default,
    })
}

pub(super) fn eval_obj_vec3(
    obj: &Object3D,
    keys: &[&str],
    default: Vec3,
    env: &EvalEnv<'_>,
) -> Vec3 {
    lookup_obj_prop_value(obj, keys)
        .and_then(|t| eval_property_track(&t, env).as_vec3())
        .map_or(default, |v| {
            Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32)
        })
}

pub(super) fn eval_obj_str(obj: &Object3D, keys: &[&str], env: &EvalEnv<'_>) -> Option<String> {
    lookup_obj_prop_value(obj, keys).and_then(|t| match eval_property_track(&t, env) {
        PropertyValue::Text(s) | PropertyValue::List(s) => Some(s),
        _ => None,
    })
}

pub(super) fn parse_deform_axis_from_obj(
    obj: &Object3D,
    keys: &[&str],
    default: DeformAxis,
    env: &EvalEnv<'_>,
) -> DeformAxis {
    if let Some(s) = eval_obj_str(obj, keys, env) {
        return match s.trim().to_ascii_lowercase().as_str() {
            "x" | "0" => DeformAxis::X,
            "y" | "1" => DeformAxis::Y,
            "z" | "2" => DeformAxis::Z,
            _ => default,
        };
    }
    if let Some(t) = lookup_obj_prop_value(obj, keys)
        && let Some(n) = eval_property_track(&t, env).as_f64()
    {
        return match n.round() as i32 {
            0 => DeformAxis::X,
            1 => DeformAxis::Y,
            2 => DeformAxis::Z,
            _ => default,
        };
    }
    default
}

pub(super) fn parse_warp_axis_from_obj(
    obj: &Object3D,
    keys: &[&str],
    env: &EvalEnv<'_>,
) -> WarpAxis {
    if let Some(s) = eval_obj_str(obj, keys, env) {
        return match s.trim().to_ascii_lowercase().as_str() {
            "z" | "1" | "into_z" | "intoz" => WarpAxis::IntoZ,
            _ => WarpAxis::IntoY,
        };
    }
    WarpAxis::from_u32(eval_obj_u32(obj, keys, 0, env))
}

pub(super) fn parse_subdivision_from_obj(obj: &Object3D, env: &EvalEnv<'_>) -> u32 {
    if obj
        .preserved
        .get("zoidiumSubdivisionMode")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| s.eq_ignore_ascii_case("legacy"))
        && let Some(lvl) = obj
            .preserved
            .get("zoidiumLegacySubdivisionLevel")
            .and_then(serde_json::Value::as_u64)
    {
        return 4_u32.pow((lvl as u32).min(4));
    }
    if let Some(track) = lookup_obj_prop_value(obj, &["subdivision", "subdivide", "polygons"]) {
        match eval_property_track(&track, env) {
            PropertyValue::Option(idx) => {
                // Zoidium option indices 0..=4 map to 4^idx polygons per triangle (1, 4, 16, 64, 256).
                return 4_u32.pow(idx.min(4));
            }
            PropertyValue::Text(s) | PropertyValue::List(s) => {
                if let Ok(n) = s.trim().parse::<u32>() {
                    return n.clamp(1, 256);
                }
            }
            PropertyValue::Number(n) => {
                return (n.round().max(1.0) as u32).clamp(1, 256);
            }
            _ => {}
        }
    }
    // Legacy Zoidium fallback (`curveQuality` + `polygonCount`, `scene-deform.js:53`).
    if lookup_obj_prop_value(obj, &["curveQuality", "polygonCount"]).is_some() {
        let quality_on = eval_obj_bool(obj, &["curveQuality"], true, env);
        if quality_on {
            let poly = eval_obj_u32(obj, &["polygonCount"], 16, env);
            return zc_scene::subdivision_polygons(poly);
        }
    }
    1
}

pub(super) fn parse_effector_field_from_obj(
    obj: &Object3D,
    env: &EvalEnv<'_>,
) -> Option<EffectorField> {
    let field_type = if let Some(s) = eval_obj_str(obj, &["field", "fieldType", "field_type"], env)
    {
        match s.trim().to_ascii_lowercase().as_str() {
            "linear" | "1" => EffectorFieldType::Linear,
            "box" | "2" => EffectorFieldType::Box,
            "sphere" | "3" => EffectorFieldType::Sphere,
            "cylinder" | "4" => EffectorFieldType::Cylinder,
            "noise" | "random" | "5" => EffectorFieldType::Noise,
            _ => EffectorFieldType::Infinite,
        }
    } else {
        EffectorFieldType::from_u32(eval_obj_u32(
            obj,
            &["field", "fieldType", "field_type"],
            0,
            env,
        ))
    };
    let invert = eval_obj_bool(obj, &["fieldInvert", "field_invert"], false, env);
    if field_type == EffectorFieldType::Infinite && !invert {
        return None;
    }
    // When a Noise field explicitly specifies zero noise strength, treat as uniform (infinite) weight.
    if field_type == EffectorFieldType::Noise
        && lookup_obj_prop_value(obj, &["fieldNoiseStrength", "field_noise_strength"]).is_some()
        && eval_obj_f32(
            obj,
            &["fieldNoiseStrength", "field_noise_strength"],
            100.0,
            env,
        )
        .abs()
            <= 1e-6
    {
        return if invert {
            Some(EffectorField {
                field_type: EffectorFieldType::Infinite,
                invert: true,
                ..EffectorField::default()
            })
        } else {
            None
        };
    }
    let curve = if let Some(s) = eval_obj_str(
        obj,
        &[
            "fieldCurve",
            "field_curve",
            "fieldFalloffCurve",
            "fieldRemap",
            "field_remap",
        ],
        env,
    ) {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "smooth" | "smoothstep" | "1" => FieldFalloffCurve::Smooth,
            "easein" | "ease-in" | "in" | "2" => FieldFalloffCurve::EaseIn,
            "easeout" | "ease-out" | "out" | "3" => FieldFalloffCurve::EaseOut,
            _ => FieldFalloffCurve::Linear,
        }
    } else {
        FieldFalloffCurve::from_u32(eval_obj_u32(
            obj,
            &[
                "fieldCurve",
                "field_curve",
                "fieldFalloffCurve",
                "fieldRemap",
                "field_remap",
            ],
            0,
            env,
        ))
    };
    let base_evolution = eval_obj_f32(
        obj,
        &["fieldNoiseEvolution", "fieldEvolution", "field_evolution"],
        0.0,
        env,
    );
    let noise_speed = eval_obj_f32(obj, &["fieldNoiseSpeed", "field_noise_speed"], 0.0, env);
    let evolution = base_evolution + noise_speed * (env.time as f32);
    Some(EffectorField {
        field_type,
        position: eval_obj_vec3(
            obj,
            &[
                "fieldPosition",
                "field_position",
                "fieldOffset",
                "field_offset",
            ],
            Vec3::ZERO,
            env,
        ),
        rotation_deg: eval_obj_vec3(obj, &["fieldRotation", "field_rotation"], Vec3::ZERO, env),
        scale: eval_obj_vec3(obj, &["fieldScale", "field_scale"], Vec3::splat(100.0), env),
        falloff: eval_obj_f32(obj, &["fieldFalloff", "field_falloff"], 100.0, env),
        curve,
        invert,
        sweep: eval_obj_f32(obj, &["fieldSweep", "field_sweep"], 100.0, env),
        noise_scale: eval_obj_f32(obj, &["fieldNoiseScale", "field_noise_scale"], 100.0, env),
        evolution,
        // Matches Zoidium `scene-fields.js:310`: Noise field falls back to object's `seed` property.
        seed: eval_obj_i32(obj, &["fieldSeed", "field_seed", "seed"], 1, env),
        legacy: eval_obj_bool(obj, &["fieldLegacy", "legacyField"], false, env),
    })
}

pub(super) fn parse_twist_params(obj: &Object3D, env: &EvalEnv<'_>) -> TwistParams {
    // Support explicit radian property (`angleRad`/`angle_rad`); otherwise ZoidCraft UI stores
    // Twist `angle` in degrees (default 0.0).
    let strength = eval_obj_f32(obj, &["strength", "twistStrength"], 1.0, env);
    let angle = if lookup_obj_prop_value(obj, &["angleRad", "angle_rad"]).is_some() {
        eval_obj_f32(obj, &["angleRad", "angle_rad"], 0.0, env) * strength
    } else {
        let angle_deg = eval_obj_f32(obj, &["angle", "twistAngle"], 0.0, env);
        (angle_deg * strength).to_radians()
    };
    TwistParams {
        angle,
        axis: parse_deform_axis_from_obj(obj, &["axis", "twistAxis"], DeformAxis::Y, env),
        offset: eval_obj_f32(obj, &["offset", "twistOffset"], 0.0, env),
        subdivision: parse_subdivision_from_obj(obj, env),
        field: parse_effector_field_from_obj(obj, env),
    }
}

pub(super) fn parse_warp_params(obj: &Object3D, env: &EvalEnv<'_>) -> WarpParams {
    let strength = eval_obj_f32(
        obj,
        &["amount", "strength", "bend", "angle", "warpStrength"],
        0.0,
        env,
    );
    let is_legacy = obj
        .preserved
        .get("zoidiumSubdivisionMode")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| s.eq_ignore_ascii_case("legacy"));
    let mut subdivision = parse_subdivision_from_obj(obj, env);
    if !is_legacy && strength.abs() > 1e-4 && subdivision <= 1 {
        subdivision = 16;
    }
    WarpParams {
        strength,
        axis: parse_warp_axis_from_obj(obj, &["axis", "warpAxis"], env),
        offset: eval_obj_f32(obj, &["offset", "warpOffset"], 0.0, env),
        subdivision,
        field: parse_effector_field_from_obj(obj, env),
    }
}

pub(super) fn parse_shear_params(obj: &Object3D, env: &EvalEnv<'_>) -> ShearParams {
    ShearParams {
        amount: eval_obj_f32(obj, &["amount", "strength", "shearAmount"], 0.0, env),
        axis: parse_deform_axis_from_obj(obj, &["axis", "shearAxis"], DeformAxis::Y, env),
        direction: parse_deform_axis_from_obj(
            obj,
            &["direction", "shearDirection"],
            DeformAxis::X,
            env,
        ),
        pivot: eval_obj_vec3(obj, &["pivot", "shearPivot"], Vec3::ZERO, env),
        subdivision: parse_subdivision_from_obj(obj, env),
        field: parse_effector_field_from_obj(obj, env),
    }
}

pub(super) fn parse_plain_params(obj: &Object3D, env: &EvalEnv<'_>) -> PlainParams {
    PlainParams {
        position_offset: eval_obj_vec3(
            obj,
            &[
                "positionOffset",
                "offsetPosition",
                "position_offset",
                "offset",
                "plainPosition",
            ],
            Vec3::ZERO,
            env,
        ),
        rotation_offset_deg: eval_obj_vec3(
            obj,
            &[
                "rotationOffset",
                "offsetRotation",
                "rotation_offset",
                "rotationOffsetDeg",
                "plainRotation",
            ],
            Vec3::ZERO,
            env,
        ),
        scale_offset: eval_obj_vec3(
            obj,
            &["scaleOffset", "offsetScale", "scale_offset", "plainScale"],
            Vec3::ZERO,
            env,
        ),
        uniform_scale: eval_obj_f32(obj, &["uniformScale", "uniform_scale"], 0.0, env),
        field: parse_effector_field_from_obj(obj, env),
    }
}

pub(super) fn parse_delay_params(obj: &Object3D, env: &EvalEnv<'_>) -> DelayParams {
    let mode = if let Some(s) = eval_obj_str(obj, &["mode", "delayMode"], env) {
        match s.trim().to_ascii_lowercase().as_str() {
            "average" | "1" => DelayMode::Average,
            "spring" | "2" => DelayMode::Spring,
            _ => DelayMode::Blend,
        }
    } else {
        DelayMode::from_u32(eval_obj_u32(obj, &["mode", "delayMode"], 0, env))
    };
    let raw_strength = eval_obj_f32(obj, &["strength", "delayStrength"], 75.0, env);
    let strength = if raw_strength > 1.0 {
        (raw_strength / 100.0).clamp(0.0, 1.0)
    } else {
        raw_strength.clamp(0.0, 1.0)
    };
    DelayParams {
        mode,
        strength,
        window: eval_obj_u32(obj, &["window", "delayWindow", "frames"], 12, env).clamp(1, 120),
        spring_damping: eval_obj_f32(
            obj,
            &["springDamping", "spring_damping", "damping"],
            SPRING_DAMPING,
            env,
        ),
        spring_cycles: eval_obj_f32(
            obj,
            &["springCycles", "spring_cycles", "cycles"],
            SPRING_CYCLES,
            env,
        ),
    }
}

pub(super) fn parse_voronoi_params(obj: &Object3D, env: &EvalEnv<'_>) -> VoronoiFractureParams {
    let distribution =
        if let Some(s) = eval_obj_str(obj, &["distribution", "siteDistribution"], env) {
            match s.trim().to_ascii_lowercase().as_str() {
                "surface" | "1" => FractureDistribution::Surface,
                "center" | "2" => FractureDistribution::Center,
                "edges" | "3" => FractureDistribution::Edges,
                _ => FractureDistribution::Uniform,
            }
        } else {
            FractureDistribution::from_u32(eval_obj_u32(
                obj,
                &["distribution", "siteDistribution"],
                0,
                env,
            ))
        };
    VoronoiFractureParams {
        cells: eval_obj_u32(
            obj,
            &["cells", "cellCount", "cell_count", "pieces"],
            24,
            env,
        )
        .clamp(1, 1000),
        seed: eval_obj_i32(obj, &["seed"], 1, env),
        closed: eval_obj_bool(obj, &["closed", "cap", "caps"], true, env),
        distribution,
        cell_scale: eval_obj_vec3(obj, &["cellScale", "cell_scale"], Vec3::splat(100.0), env),
        distance: eval_obj_f32(obj, &["distance", "explode", "explosion"], 0.0, env),
        scatter: eval_obj_f32(obj, &["scatter"], 0.0, env),
        offset: eval_obj_f32(obj, &["offset", "shrink"], 0.0, env),
        spin_rad: if lookup_obj_prop_value(obj, &["spin", "spinRad", "spin_rad"]).is_some() {
            eval_obj_f32(obj, &["spin", "spinRad", "spin_rad"], 0.0, env)
        } else {
            eval_obj_f32(obj, &["spinDeg", "spin_deg"], 0.0, env).to_radians()
        },
        direction: eval_obj_vec3(
            obj,
            &["fragmentDirection", "direction", "fragment_direction"],
            Vec3::ZERO,
            env,
        ),
        rotation_deg: eval_obj_vec3(
            obj,
            &["fragmentRotation", "rotationDeg", "rotation_deg"],
            Vec3::ZERO,
            env,
        ),
        fragment_scale: eval_obj_f32(obj, &["fragmentScale", "fragment_scale"], 100.0, env),
        gravity: eval_obj_f32(obj, &["gravity"], 0.0, env),
        randomness: eval_obj_f32(obj, &["randomness"], 0.0, env),
        random_colors: eval_obj_u32(obj, &["colors", "randomColors"], 1, env) != 0,
        field: parse_effector_field_from_obj(obj, env),
    }
}

pub(super) fn parse_repeater_params(
    obj: &Object3D,
    default_mode: RepeaterMode,
    env: &EvalEnv<'_>,
) -> RepeaterParams {
    let mode = if let Some(s) = eval_obj_str(obj, &["mode", "repeaterMode"], env) {
        match s.trim().to_ascii_lowercase().as_str() {
            "linear" | "1" => RepeaterMode::Linear,
            "random" | "2" => RepeaterMode::Random,
            "echo" | "3" => RepeaterMode::Echo,
            "step" | "0" => RepeaterMode::Step,
            _ => default_mode,
        }
    } else {
        default_mode
    };
    RepeaterParams {
        mode,
        count: eval_obj_u32(obj, &["count", "copies", "repeats"], 5, env).clamp(1, MAX_REPEATS),
        euler_order: map_euler_order(obj.euler_order),
        position_step: eval_obj_vec3(
            obj,
            &["positionStep", "position_step", "offset", "positionOffset"],
            Vec3::new(10.0, 0.0, 0.0),
            env,
        ),
        rotation_step_rad: eval_obj_vec3(
            obj,
            &["rotationStep", "rotation_step", "rotationOffset"],
            Vec3::ZERO,
            env,
        ),
        scale_step: eval_obj_vec3(
            obj,
            &["scaleStep", "scale_step", "scaleOffset"],
            Vec3::ONE,
            env,
        ),
        position_end: eval_obj_vec3(
            obj,
            &["positionEnd", "position_end", "endPosition", "offset"],
            Vec3::new(40.0, 0.0, 0.0),
            env,
        ),
        rotation_end_rad: eval_obj_vec3(
            obj,
            &["rotationEnd", "rotation_end", "endRotation"],
            Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2),
            env,
        ),
        scale_end: eval_obj_vec3(obj, &["scaleEnd", "scale_end", "endScale"], Vec3::ONE, env),
        position_min: eval_obj_vec3(
            obj,
            &["positionMin", "position_min"],
            Vec3::splat(-10.0),
            env,
        ),
        position_max: eval_obj_vec3(
            obj,
            &["positionMax", "position_max"],
            Vec3::splat(10.0),
            env,
        ),
        rotation_min_rad: eval_obj_vec3(obj, &["rotationMin", "rotation_min"], Vec3::ZERO, env),
        rotation_max_rad: eval_obj_vec3(obj, &["rotationMax", "rotation_max"], Vec3::ZERO, env),
        scale_min: eval_obj_vec3(obj, &["scaleMin", "scale_min"], Vec3::ONE, env),
        scale_max: eval_obj_vec3(obj, &["scaleMax", "scale_max"], Vec3::ONE, env),
        seed: eval_obj_i32(obj, &["seed"], 1, env),
        delay_frames: eval_obj_f64(
            obj,
            &[
                "delay",
                "delayFrames",
                "delay_frames",
                "frameDelay",
                "echoDelay",
            ],
            5.0,
            env,
        ),
    }
}

// ---------------------------------------------------------------------------
// ExpressText+ Runtime Adapter (`zc-expr::expresstext`)
// ---------------------------------------------------------------------------

pub(super) struct OwnedExpressTextLayer {
    pub(super) kind: ExpressTextEffectorKind,
    pub(super) props: BTreeMap<String, PropertyTrack>,
}

pub(super) struct SceneEtPropReader<'a> {
    pub(super) base_env: &'a EvalEnv<'a>,
    pub(super) prefix: &'static str,
    pub(super) props: &'a BTreeMap<String, PropertyTrack>,
}

impl SceneEtPropReader<'_> {
    fn eval_track_at(&self, track: &PropertyTrack, frame: f64) -> Option<EtValue> {
        let env_f = env_at_frame(self.base_env, frame);
        let val = eval_property_track(track, &env_f);
        match val {
            PropertyValue::Number(n) => Some(EtValue::Number(n)),
            PropertyValue::Option(o) => Some(EtValue::Number(f64::from(o))),
            PropertyValue::Bool(b) => Some(EtValue::Number(if b { 1.0 } else { 0.0 })),
            PropertyValue::Vec3(v) | PropertyValue::Color(v) => Some(EtValue::Vec3(v)),
            PropertyValue::Text(s) | PropertyValue::List(s) => Some(EtValue::Text(s)),
            _ => None,
        }
    }

    fn lookup_direct(&self, key: &str) -> Option<&PropertyTrack> {
        self.props.get(key).or_else(|| {
            let stripped = key.strip_prefix(self.prefix)?;
            let mut chars = stripped.chars();
            let first = chars.next()?;
            let lower = format!("{}{}", first.to_ascii_lowercase(), chars.as_str());
            self.props.get(&lower).or_else(|| self.props.get(stripped))
        })
    }
}

impl EtPropReader for SceneEtPropReader<'_> {
    fn read(&self, key: &str, frame: f64) -> Option<EtValue> {
        if let Some(track) = self.lookup_direct(key) {
            return self.eval_track_at(track, frame);
        }

        // Wire the 6 ExpressText+ compatibility / default alias keys:
        // `delayFrames`, `splineAmplitude`, `randomPosition`, `randomRotation`, `repeaterCount`.
        match key {
            "delayStrength" => {
                if let Some(tr) = self
                    .props
                    .get("delayFrames")
                    .or_else(|| self.props.get("frames"))
                {
                    let frames = self.eval_track_at(tr, frame)?.as_number(0.0).max(0.0);
                    let pct = if frames <= 1e-6 {
                        0.0
                    } else {
                        ((frames / (frames + 4.0)) * 100.0).clamp(0.0, 99.0)
                    };
                    return Some(EtValue::Number(pct));
                }
            }
            "splineRadius" | "splineHeight" => {
                if let Some(tr) = self
                    .props
                    .get("splineAmplitude")
                    .or_else(|| self.props.get("amplitude"))
                {
                    return self.eval_track_at(tr, frame);
                }
            }
            "splinePath" => {
                if self.props.contains_key("splineAmplitude")
                    || self.props.contains_key("amplitude")
                {
                    return Some(EtValue::Number(1.0));
                }
            }
            "randomPositionMin" | "randomPositionMax" => {
                if let Some(tr) = self
                    .props
                    .get("randomPosition")
                    .or_else(|| self.props.get("position"))
                {
                    let v = match self.eval_track_at(tr, frame)? {
                        EtValue::Vec3(a) => a,
                        EtValue::Number(n) => [n, n, n],
                        EtValue::Text(_) => return None,
                    };
                    return Some(if key == "randomPositionMin" {
                        EtValue::Vec3([-v[0].abs(), -v[1].abs(), -v[2].abs()])
                    } else {
                        EtValue::Vec3([v[0].abs(), v[1].abs(), v[2].abs()])
                    });
                }
            }
            "randomRotationMin" | "randomRotationMax" => {
                if let Some(tr) = self
                    .props
                    .get("randomRotation")
                    .or_else(|| self.props.get("rotation"))
                {
                    let v = match self.eval_track_at(tr, frame)? {
                        EtValue::Vec3(a) => a,
                        EtValue::Number(n) => [n, n, n],
                        EtValue::Text(_) => return None,
                    };
                    return Some(if key == "randomRotationMin" {
                        EtValue::Vec3([-v[0].abs(), -v[1].abs(), -v[2].abs()])
                    } else {
                        EtValue::Vec3([v[0].abs(), v[1].abs(), v[2].abs()])
                    });
                }
            }
            "repeaterCopies" => {
                if let Some(tr) = self
                    .props
                    .get("repeaterCount")
                    .or_else(|| self.props.get("count"))
                    .or_else(|| self.props.get("copies"))
                {
                    return self.eval_track_at(tr, frame);
                }
            }
            _ => {
                if key.ends_with("Delay") {
                    let frames_key = format!("{key}Frames");
                    if let Some(tr) = self
                        .props
                        .get(&frames_key)
                        .or_else(|| self.props.get("delayFrames"))
                    {
                        return self.eval_track_at(tr, frame);
                    }
                }
            }
        }
        None
    }
}

pub(super) fn parse_geom_euler_order(raw: &str) -> GeomEulerOrder {
    match raw.trim().to_ascii_uppercase().as_str() {
        "XZY" | "1" => GeomEulerOrder::XZY,
        "YXZ" | "2" => GeomEulerOrder::YXZ,
        "YZX" | "3" => GeomEulerOrder::YZX,
        "ZXY" | "4" => GeomEulerOrder::ZXY,
        "ZYX" | "5" => GeomEulerOrder::ZYX,
        _ => GeomEulerOrder::XYZ,
    }
}

fn json_raw_items(val: &serde_json::Value) -> Vec<&serde_json::Value> {
    match val {
        serde_json::Value::Array(arr) => arr.iter().collect(),
        serde_json::Value::Object(map) => {
            if let Some(len) = map
                .get("length")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n.min(256) as usize)
            {
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    if let Some(v) = map.get(&i.to_string()) {
                        out.push(v);
                    }
                }
                out
            } else {
                let mut indexed: Vec<(usize, &serde_json::Value)> = map
                    .iter()
                    .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (idx, v)))
                    .collect();
                indexed.sort_by_key(|(idx, _)| *idx);
                indexed.into_iter().map(|(_, v)| v).collect()
            }
        }
        _ => Vec::new(),
    }
}

/// Collects live animated `PropertyTrack`s from a raw `expressProperties` JSON block so
/// keyframed/expression-driven ExpressText+ properties evaluate dynamically at `env.frame`
/// instead of freezing at keyframe 0.
fn collect_live_express_tracks_from_json(
    ep_val: &serde_json::Value,
    out_tracks: &mut BTreeMap<String, PropertyTrack>,
) {
    let mut record_map = |map: &serde_json::Map<String, serde_json::Value>| {
        for (k, v) in map {
            if matches!(
                k.as_str(),
                "effectors"
                    | "customProperties"
                    | "objects"
                    | "expressProperties"
                    | "type"
                    | "kind"
            ) {
                continue;
            }
            out_tracks.insert(
                k.clone(),
                json_to_property_track(v, PropertyValue::Number(0.0)),
            );
        }
    };

    let collect_section = |item: &serde_json::Value, dst: &mut BTreeMap<String, PropertyTrack>| {
        let Some(obj) = item.as_object() else {
            return;
        };
        let def_rows = obj
            .get("type")
            .and_then(|t| t.get("objects"))
            .map(json_raw_items)
            .unwrap_or_default();
        let live_rows = obj.get("objects").map(json_raw_items).unwrap_or_default();
        for (idx, row) in live_rows.iter().enumerate() {
            let key = row
                .get("effectorKey")
                .and_then(serde_json::Value::as_str)
                .or_else(|| {
                    row.get("definition")
                        .and_then(|d| d.get("effectorKey"))
                        .and_then(serde_json::Value::as_str)
                })
                .or_else(|| {
                    def_rows
                        .get(idx)
                        .and_then(|d| d.get("effectorKey"))
                        .and_then(serde_json::Value::as_str)
                });
            if let Some(k) = key {
                dst.insert(
                    k.to_owned(),
                    json_to_property_track(row, PropertyValue::Number(0.0)),
                );
            }
        }
        if let Some(explicit) = obj
            .get("properties")
            .or_else(|| obj.get("expressProperties"))
            .and_then(serde_json::Value::as_object)
        {
            for (k, v) in explicit {
                dst.insert(
                    k.clone(),
                    json_to_property_track(v, PropertyValue::Number(0.0)),
                );
            }
        }
    };

    if let Some(arr) = ep_val.as_array() {
        for item in arr {
            collect_section(item, out_tracks);
        }
        return;
    }
    let Some(root) = ep_val.as_object() else {
        return;
    };
    let ep_map = root
        .get("expressProperties")
        .and_then(serde_json::Value::as_object)
        .unwrap_or(root);
    record_map(ep_map);
    if let Some(list) = root.get("effectors").or_else(|| ep_map.get("effectors")) {
        for item in json_raw_items(list) {
            collect_section(item, out_tracks);
        }
    }
    if let Some(customs) = root.get("customProperties") {
        for item in json_raw_items(customs) {
            collect_section(item, out_tracks);
        }
    }
    if let Some(objs) = root.get("objects") {
        for child in json_raw_items(objs) {
            if let Some(child_ep) = child
                .get("expressProperties")
                .or_else(|| child.get("properties"))
                .and_then(serde_json::Value::as_object)
            {
                for (k, v) in child_ep {
                    out_tracks.insert(
                        k.clone(),
                        json_to_property_track(v, PropertyValue::Number(0.0)),
                    );
                }
            }
        }
    }
}

fn parse_letters_mode_value(val: &serde_json::Value) -> Option<ExpressTextLettersMode> {
    if let Some(n) = val.as_f64() {
        return Some(ExpressTextLettersMode::from_raw(n));
    }
    if let Some(s) = val.as_str() {
        let low = s.trim().to_ascii_lowercase();
        if low.contains("across") || low == "1" {
            return Some(ExpressTextLettersMode::AcrossTexts);
        }
        if low.contains("per") || low == "0" {
            return Some(ExpressTextLettersMode::PerText);
        }
    }
    if let Some(inner) = val.get("value").or_else(|| val.get("v")) {
        return parse_letters_mode_value(inner);
    }
    None
}

fn parse_rotation_order_value(val: &serde_json::Value) -> Option<GeomEulerOrder> {
    if let Some(s) = val.as_str() {
        let t = s.trim();
        if !t.is_empty() {
            return Some(parse_geom_euler_order(t));
        }
    }
    if let Some(n) = val.as_u64() {
        return Some(parse_geom_euler_order(&n.to_string()));
    }
    if let Some(f) = val.as_f64() {
        return Some(parse_geom_euler_order(&(f.round() as i64).to_string()));
    }
    if let Some(inner) = val.get("value").or_else(|| val.get("v")) {
        return parse_rotation_order_value(inner);
    }
    None
}

pub(super) fn parse_expresstext_kind(token: &str) -> Option<ExpressTextEffectorKind> {
    ExpressTextEffectorKind::from_name(token).or_else(|| {
        match token.trim().to_ascii_lowercase().as_str() {
            "shake" => Some(ExpressTextEffectorKind::Shake),
            "plain" => Some(ExpressTextEffectorKind::Plain),
            "random" => Some(ExpressTextEffectorKind::Random),
            "formula" => Some(ExpressTextEffectorKind::Formula),
            "twist" => Some(ExpressTextEffectorKind::Twist),
            "wrap" => Some(ExpressTextEffectorKind::Wrap),
            "orbit" => Some(ExpressTextEffectorKind::Orbit),
            "spline" => Some(ExpressTextEffectorKind::Spline),
            "delay" => Some(ExpressTextEffectorKind::Delay),
            "repeater" => Some(ExpressTextEffectorKind::Repeater),
            "voronoi" | "shatter" => Some(ExpressTextEffectorKind::Voronoi),
            _ => None,
        }
    })
}

pub(super) fn extract_expresstext_layers_from_obj(
    obj: &Object3D,
    env: &EvalEnv<'_>,
) -> (
    Vec<OwnedExpressTextLayer>,
    Option<ExpressTextLettersMode>,
    Option<GeomEulerOrder>,
) {
    let mut out = Vec::new();
    let mut letters_mode: Option<ExpressTextLettersMode> = None;
    let mut rotation_order: Option<GeomEulerOrder> = None;

    // Collect live animated tracks from `Group.custom_properties` and `preserved["property.*"]`
    // so animated ExpressText+ tracks evaluate dynamically at `env.frame`.
    let mut live_obj_tracks: BTreeMap<String, PropertyTrack> = BTreeMap::new();
    if let Object3DKind::Group {
        custom_properties, ..
    } = &obj.kind
    {
        for cp in custom_properties {
            live_obj_tracks.insert(cp.name.clone(), cp.track.clone());
        }
    }
    for (k, v) in &obj.preserved {
        if let Some(stripped) = k.strip_prefix("property.") {
            live_obj_tracks
                .entry(stripped.to_owned())
                .or_insert_with(|| json_to_property_track(v, PropertyValue::Number(0.0)));
        }
    }

    // 0. Canonical Zoidium `preserved["expressProperties"]` via `parse_express_text_config`,
    //    upgraded with live keyframed `PropertyTrack`s from raw JSON + `live_obj_tracks`.
    if let Some(ep_val) = obj.preserved.get("expressProperties")
        && let Ok(cfg) = parse_express_text_config(ep_val)
    {
        letters_mode = Some(cfg.letters_mode);
        let parsed_order = parse_geom_euler_order(&cfg.rotation_order);
        if cfg.rotation_order != "XYZ" || ep_val.get("rotationOrder").is_some() {
            rotation_order = Some(parsed_order);
        }
        let mut live_ep_tracks = BTreeMap::new();
        collect_live_express_tracks_from_json(ep_val, &mut live_ep_tracks);

        for eff in cfg.effectors {
            let mut props = BTreeMap::new();
            for (k, v) in &eff.properties {
                if let Some(live_tr) = live_obj_tracks.get(k).or_else(|| live_ep_tracks.get(k)) {
                    props.insert(k.clone(), live_tr.clone());
                } else {
                    let pv = match v {
                        EtValue::Number(n) => PropertyValue::Number(*n),
                        EtValue::Vec3(a) => PropertyValue::Vec3(*a),
                        EtValue::Text(s) => PropertyValue::Text(s.clone()),
                    };
                    props.insert(k.clone(), PropertyTrack::constant(pv));
                }
            }
            // Merge any alias/extra live keys matching this effector's prefix (e.g. `delayFrames`,
            // `splineAmplitude`, `randomPosition`, `randomRotation`, `repeaterCount`).
            let prefix = eff.kind.as_str();
            for (k, tr) in live_ep_tracks.iter().chain(live_obj_tracks.iter()) {
                if k.starts_with(prefix) && k.len() > prefix.len() {
                    props.entry(k.clone()).or_insert_with(|| tr.clone());
                }
            }
            out.push(OwnedExpressTextLayer {
                kind: eff.kind,
                props,
            });
        }
        // Also extract any effector layers present only via alias keys (e.g. `delayFrames` or
        // `repeaterCount` without standard keys in `parse_express_text_config`).
        let mut extra_from_live = Vec::new();
        let mut combined_live = live_ep_tracks;
        for (k, v) in &live_obj_tracks {
            combined_live.insert(k.clone(), v.clone());
        }
        extract_flat_expresstext_tracks(&combined_live, &mut extra_from_live);
        for extra in extra_from_live {
            if !out.iter().any(|existing| existing.kind == extra.kind) {
                out.push(extra);
            }
        }
    }

    // Check explicit `lettersMode` / `property.lettersMode` / `letters` in preserved or custom properties.
    if letters_mode.is_none()
        && let Some(tr) = live_obj_tracks
            .get("lettersMode")
            .or_else(|| live_obj_tracks.get("letters"))
    {
        match eval_property_track(tr, env) {
            PropertyValue::Number(n) => {
                letters_mode = Some(ExpressTextLettersMode::from_raw(n));
            }
            PropertyValue::Option(o) => {
                letters_mode = Some(ExpressTextLettersMode::from_raw(f64::from(o)));
            }
            PropertyValue::Text(s) => {
                let low = s.trim().to_ascii_lowercase();
                if low.contains("across") || low == "1" {
                    letters_mode = Some(ExpressTextLettersMode::AcrossTexts);
                } else if low.contains("per") || low == "0" {
                    letters_mode = Some(ExpressTextLettersMode::PerText);
                }
            }
            _ => {}
        }
    }
    if letters_mode.is_none()
        && let Some(lm_val) = obj
            .preserved
            .get("lettersMode")
            .or_else(|| obj.preserved.get("property.lettersMode"))
            .or_else(|| obj.preserved.get("letters"))
            .or_else(|| obj.preserved.get("property.letters"))
    {
        letters_mode = parse_letters_mode_value(lm_val);
    }

    // Check explicit `rotationOrder` / `property.rotationOrder` in preserved or custom properties.
    if rotation_order.is_none()
        && let Some(tr) = live_obj_tracks.get("rotationOrder")
    {
        match eval_property_track(tr, env) {
            PropertyValue::Text(s) if !s.trim().is_empty() => {
                rotation_order = Some(parse_geom_euler_order(&s));
            }
            PropertyValue::Option(o) => {
                rotation_order = Some(parse_geom_euler_order(&o.to_string()));
            }
            PropertyValue::Number(n) => {
                rotation_order = Some(parse_geom_euler_order(&(n.round() as i64).to_string()));
            }
            _ => {}
        }
    }
    if rotation_order.is_none()
        && let Some(ro_val) = obj
            .preserved
            .get("rotationOrder")
            .or_else(|| obj.preserved.get("property.rotationOrder"))
    {
        rotation_order = parse_rotation_order_value(ro_val);
    }

    // 1. Explicit array in preserved["expressTextEffectors" | "expresstextEffectors" | "expressText" | "expresstext"]
    for key in [
        "expressTextEffectors",
        "expresstextEffectors",
        "expressText",
        "expresstext",
    ] {
        let Some(val) = obj.preserved.get(key) else {
            continue;
        };
        let arr_opt = val.as_array().or_else(|| {
            val.as_object()
                .and_then(|m| m.get("effectors").or_else(|| m.get("layers")))
                .and_then(serde_json::Value::as_array)
        });
        if let Some(arr) = arr_opt {
            for item in arr {
                let Some(map) = item.as_object() else {
                    continue;
                };
                let Some(kind_str) = map
                    .get("kind")
                    .or_else(|| map.get("type"))
                    .and_then(serde_json::Value::as_str)
                else {
                    continue;
                };
                let Some(kind) = parse_expresstext_kind(kind_str) else {
                    continue;
                };
                let mut props = BTreeMap::new();
                if let Some(sub) = map.get("props").and_then(serde_json::Value::as_object) {
                    for (pk, pv) in sub {
                        props.insert(
                            pk.clone(),
                            json_to_property_track(pv, PropertyValue::Number(0.0)),
                        );
                    }
                }
                for (pk, pv) in map {
                    if !matches!(pk.as_str(), "kind" | "type" | "props") {
                        props.insert(
                            pk.clone(),
                            json_to_property_track(pv, PropertyValue::Number(0.0)),
                        );
                    }
                }
                out.push(OwnedExpressTextLayer { kind, props });
            }
        } else if let Some(map) = val.as_object() {
            extract_flat_expresstext_props_from_map(map, &mut out);
        }
    }

    // 2. Zoidium ExpressText+ container/sub-effector Group properties (`custom_properties` + `preserved["property.*"]`).
    let et_plugin_str = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)
        .map(|s| s.trim().to_ascii_lowercase());
    let is_et_plugin = et_plugin_str.as_deref().is_some_and(|low| {
        low.starts_with("zoidium:expresstext")
            || low.starts_with("zoidcraft:expresstext")
            || low.starts_with("expresstext")
    });
    if is_et_plugin && out.is_empty() {
        extract_flat_expresstext_tracks(&live_obj_tracks, &mut out);
        // Dedicated sub-effector class (`zoidium:expresstext-plus/shake`, etc.) with unprefixed keys.
        if out.is_empty()
            && let Some(sub) = et_plugin_str
                .as_deref()
                .and_then(|s| s.rsplit_once('/').map(|(_, r)| r))
            && let Some(kind) = parse_expresstext_kind(sub)
        {
            out.push(OwnedExpressTextLayer {
                kind,
                props: live_obj_tracks,
            });
        }
    }

    (out, letters_mode, rotation_order)
}

pub(super) fn extract_flat_expresstext_props_from_map(
    map: &serde_json::Map<String, serde_json::Value>,
    out: &mut Vec<OwnedExpressTextLayer>,
) {
    let mut flat = BTreeMap::new();
    for (k, v) in map {
        flat.insert(
            k.clone(),
            json_to_property_track(v, PropertyValue::Number(0.0)),
        );
    }
    extract_flat_expresstext_tracks(&flat, out);
}

pub(super) fn extract_flat_expresstext_tracks(
    flat: &BTreeMap<String, PropertyTrack>,
    out: &mut Vec<OwnedExpressTextLayer>,
) {
    const KINDS: [ExpressTextEffectorKind; 11] = [
        ExpressTextEffectorKind::Shake,
        ExpressTextEffectorKind::Plain,
        ExpressTextEffectorKind::Random,
        ExpressTextEffectorKind::Formula,
        ExpressTextEffectorKind::Twist,
        ExpressTextEffectorKind::Wrap,
        ExpressTextEffectorKind::Orbit,
        ExpressTextEffectorKind::Spline,
        ExpressTextEffectorKind::Delay,
        ExpressTextEffectorKind::Repeater,
        ExpressTextEffectorKind::Voronoi,
    ];
    for kind in KINDS {
        let prefix = kind.as_str();
        let mut props = BTreeMap::new();
        for (k, v) in flat {
            if k.starts_with(prefix) && k.len() > prefix.len() {
                props.insert(k.clone(), v.clone());
            }
        }
        if !props.is_empty() {
            out.push(OwnedExpressTextLayer { kind, props });
        }
    }
}

pub(super) fn collect_expresstext_layers_for_node(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    node_idx: usize,
    env: &EvalEnv<'_>,
) -> (
    Vec<OwnedExpressTextLayer>,
    ExpressTextLettersMode,
    Option<usize>,
    GeomEulerOrder,
) {
    let self_env = env.with_object(eval_graph.nodes[node_idx].obj.id);
    let (mut layers, mut mode, mut rot_order) =
        extract_expresstext_layers_from_obj(eval_graph.nodes[node_idx].obj, &self_env);
    let mut container_idx: Option<usize> = None;
    let mut cur = eval_graph.nodes[node_idx].parent;
    let mut guard = 0usize;
    while let Some(anc) = cur {
        if guard >= 256 {
            break;
        }
        if eval_graph.nodes[anc].render_enabled {
            let anc_env = env.with_object(eval_graph.nodes[anc].obj.id);
            let (anc_layers, anc_mode, anc_rot_order) =
                extract_expresstext_layers_from_obj(eval_graph.nodes[anc].obj, &anc_env);
            if !anc_layers.is_empty() {
                if container_idx.is_none() {
                    container_idx = Some(anc);
                }
                layers.extend(anc_layers);
            }
            if mode.is_none() && anc_mode.is_some() {
                mode = anc_mode;
                if container_idx.is_none() {
                    container_idx = Some(anc);
                }
            }
            if rot_order.is_none() && anc_rot_order.is_some() {
                rot_order = anc_rot_order;
            }
        }
        cur = eval_graph.nodes[anc].parent;
        guard += 1;
    }
    (
        layers,
        mode.unwrap_or(ExpressTextLettersMode::PerText),
        container_idx,
        rot_order.unwrap_or(GeomEulerOrder::XYZ),
    )
}

pub(super) fn compute_across_texts_range(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    container_idx: usize,
    target_node_idx: usize,
    target_glyph_count: usize,
) -> (usize, usize) {
    let mut offset = 0usize;
    let mut total = 0usize;
    let mut found = false;
    let mut stack: Vec<usize> = eval_graph.nodes[container_idx]
        .children
        .iter()
        .rev()
        .copied()
        .collect();
    let mut guard = 0usize;
    while let Some(idx) = stack.pop() {
        if guard >= 4096 {
            break;
        }
        guard += 1;
        let node = &eval_graph.nodes[idx];
        if !node.render_enabled {
            continue;
        }
        if let Object3DKind::Text3D(td) = &node.obj.kind {
            let count = if idx == target_node_idx {
                target_glyph_count
            } else {
                td.text.chars().filter(|c| !c.is_whitespace()).count()
            };
            if idx == target_node_idx {
                offset = total;
                found = true;
            }
            total += count;
        }
        for &c in node.children.iter().rev() {
            stack.push(c);
        }
    }
    if !found {
        (0, target_glyph_count)
    } else {
        (offset, total.max(target_glyph_count))
    }
}

pub(super) fn shatter_mesh_data_expresstext(
    mesh: &MeshData,
    params: &zc_expr::expresstext::VoronoiGlyphParams,
) -> MeshData {
    let mut flat_pos = Vec::with_capacity(mesh.indices.len() * 3);
    let mut flat_norm = Vec::with_capacity(mesh.indices.len() * 3);
    let mut flat_uv = Vec::with_capacity(mesh.indices.len() * 2);
    let mut uniform_slot: Option<u32> = None;
    let mut multi_slot = false;
    let mut orig_tri_slots: Vec<(Vec3, u32)> = Vec::with_capacity(mesh.indices.len() / 3);
    type QuantizedTriKey = ([i32; 3], [i32; 3], [i32; 3]);
    let mut exact_tri_slots: HashMap<QuantizedTriKey, u32> =
        HashMap::with_capacity(mesh.indices.len() / 3);
    let quantize_pos = |p: [f32; 3]| -> [i32; 3] {
        [
            (p[0] * 1000.0).round() as i32,
            (p[1] * 1000.0).round() as i32,
            (p[2] * 1000.0).round() as i32,
        ]
    };

    for tri in mesh.indices.as_chunks::<3>().0 {
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];
        for v in [v0, v1, v2] {
            flat_pos.extend_from_slice(&v.position);
            flat_norm.extend_from_slice(&v.normal);
            flat_uv.extend_from_slice(&v.uv);
        }
        let slot = v0.material_slot;
        match uniform_slot {
            None => uniform_slot = Some(slot),
            Some(prev) if prev != slot => multi_slot = true,
            _ => {}
        }
        let c = (Vec3::from_array(v0.position)
            + Vec3::from_array(v1.position)
            + Vec3::from_array(v2.position))
            * (1.0 / 3.0);
        orig_tri_slots.push((c, slot));
        exact_tri_slots.insert(
            (
                quantize_pos(v0.position),
                quantize_pos(v1.position),
                quantize_pos(v2.position),
            ),
            slot,
        );
    }

    let chunks = shatter_glyph_triangles(&flat_pos, &flat_norm, &flat_uv, None, params);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let default_slot = uniform_slot.unwrap_or(0);

    for chunk in &chunks {
        let posed = apply_chunk_pose_to_positions(chunk);
        let rot_q = euler_to_quat(
            Vec3::new(
                chunk.rotation[0] as f32,
                chunk.rotation[1] as f32,
                chunk.rotation[2] as f32,
            ),
            GeomEulerOrder::XYZ,
        );
        let tri_count = posed.len() / 9;
        for t in 0..tri_count {
            let b = t * 9;
            let p0 = Vec3::new(posed[b], posed[b + 1], posed[b + 2]);
            let p1 = Vec3::new(posed[b + 3], posed[b + 4], posed[b + 5]);
            let p2 = Vec3::new(posed[b + 6], posed[b + 7], posed[b + 8]);
            let face_n = (p1 - p0).cross(p2 - p0).normalize_or_zero().to_array();

            // Preserve real interpolated UVs from `chunk.uvs` instead of overwriting with defaults.
            let ub = t * 6;
            let (uv0, uv1, uv2) = if ub + 5 < chunk.uvs.len() {
                (
                    [chunk.uvs[ub], chunk.uvs[ub + 1]],
                    [chunk.uvs[ub + 2], chunk.uvs[ub + 3]],
                    [chunk.uvs[ub + 4], chunk.uvs[ub + 5]],
                )
            } else {
                ([0.0, 0.0], [1.0, 0.0], [0.5, 1.0])
            };

            // Preserve rotated smooth vertex normals from `chunk.normals` when valid.
            let (n0, n1, n2) = if b + 8 < chunk.normals.len() {
                let rn = |off: usize| -> [f32; 3] {
                    let raw = Vec3::new(
                        chunk.normals[off],
                        chunk.normals[off + 1],
                        chunk.normals[off + 2],
                    );
                    let rotated = (rot_q * raw).normalize_or_zero();
                    if rotated.length_squared() > 1e-4 {
                        rotated.to_array()
                    } else {
                        face_n
                    }
                };
                (rn(b), rn(b + 3), rn(b + 6))
            } else {
                (face_n, face_n, face_n)
            };

            // Preserve per-triangle `material_slot` (front vs side/bevel) from original mesh.
            let slot = if !multi_slot {
                default_slot
            } else if b + 8 < chunk.positions.len() {
                let q0 = [
                    chunk.positions[b],
                    chunk.positions[b + 1],
                    chunk.positions[b + 2],
                ];
                let q1 = [
                    chunk.positions[b + 3],
                    chunk.positions[b + 4],
                    chunk.positions[b + 5],
                ];
                let q2 = [
                    chunk.positions[b + 6],
                    chunk.positions[b + 7],
                    chunk.positions[b + 8],
                ];
                exact_tri_slots
                    .get(&(quantize_pos(q0), quantize_pos(q1), quantize_pos(q2)))
                    .copied()
                    .unwrap_or_else(|| {
                        let qc =
                            (Vec3::from_array(q0) + Vec3::from_array(q1) + Vec3::from_array(q2))
                                * (1.0 / 3.0);
                        orig_tri_slots
                            .iter()
                            .min_by(|(ca, _), (cb, _)| {
                                ca.distance_squared(qc).total_cmp(&cb.distance_squared(qc))
                            })
                            .map_or(default_slot, |&(_, s)| s)
                    })
            } else {
                default_slot
            };

            let base = vertices.len() as u32;
            vertices.push(Vertex3D::new(p0.to_array(), n0, uv0, slot));
            vertices.push(Vertex3D::new(p1.to_array(), n1, uv1, slot));
            vertices.push(Vertex3D::new(p2.to_array(), n2, uv2, slot));
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
    }
    MeshData::new(vertices, indices)
}

// ---------------------------------------------------------------------------
// Text+ 3D Modifier Groups (all 7 `zoidium:text-plus/*` groups)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TextPlus3DGroupKind {
    TextPlus,
    GradientTransform,
    CharacterShake,
    SelectedTransform,
    SelectedShake,
    RandomScatter,
    CharacterMaterial,
}

pub(super) fn detect_text_plus_3d_group_kind(obj: &Object3D) -> Option<TextPlus3DGroupKind> {
    let raw = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            obj.preserved
                .get("effectorType")
                .and_then(serde_json::Value::as_str)
        })
        .or_else(|| {
            obj.preserved
                .get("type")
                .and_then(serde_json::Value::as_str)
        })
        .or(match &obj.kind {
            Object3DKind::Dummy { original_type, .. } => original_type.as_deref(),
            _ => None,
        })?;
    let mut low = raw.trim().to_ascii_lowercase().replace('_', "-");
    if let Some(rest) = low.strip_prefix("zoidcraft:") {
        low = format!("zoidium:{rest}");
    }
    match low.as_str() {
        "zoidium:text-plus/text-plus" | "zoidium:text-plus/container" => {
            Some(TextPlus3DGroupKind::TextPlus)
        }
        "zoidium:text-plus/gradient-transform" => Some(TextPlus3DGroupKind::GradientTransform),
        "zoidium:text-plus/character-shake" => Some(TextPlus3DGroupKind::CharacterShake),
        "zoidium:text-plus/selected-transform" => Some(TextPlus3DGroupKind::SelectedTransform),
        "zoidium:text-plus/selected-shake" => Some(TextPlus3DGroupKind::SelectedShake),
        "zoidium:text-plus/random-scatter" => Some(TextPlus3DGroupKind::RandomScatter),
        "zoidium:text-plus/character-material" => Some(TextPlus3DGroupKind::CharacterMaterial),
        _ => None,
    }
}

fn text_plus_delay_rank(order: &str, char_idx: usize, total_chars: usize, seed: f64) -> f64 {
    if total_chars <= 1 {
        return 0.0;
    }
    let pos = char_idx.min(total_chars - 1);
    match order.trim().to_ascii_lowercase().as_str() {
        "end" | "1" => (total_chars - 1 - pos) as f64,
        "middle" | "2" => {
            let center = (total_chars - 1) / 2;
            if pos == center {
                0.0
            } else if pos > center {
                ((pos - center) * 2 - 1) as f64
            } else {
                ((center - pos) * 2) as f64
            }
        }
        "random" | "3" => {
            let mut perm: Vec<usize> = (0..total_chars).collect();
            for i in (1..total_chars).rev() {
                let r = text_seeded_random(seed, i as i32, 11);
                let swap = ((r * (i + 1) as f64).floor() as usize).min(i);
                perm.swap(i, swap);
            }
            perm.iter().position(|&v| v == pos).unwrap_or(pos) as f64
        }
        _ => pos as f64,
    }
}

pub(super) fn collect_text_plus_3d_groups(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    node_idx: usize,
) -> Vec<(usize, TextPlus3DGroupKind)> {
    let mut out = Vec::new();
    let mut cur = eval_graph.nodes[node_idx].parent;
    let mut guard = 0usize;
    while let Some(anc) = cur {
        if guard >= 256 {
            break;
        }
        if eval_graph.nodes[anc].render_enabled
            && let Some(kind) = detect_text_plus_3d_group_kind(eval_graph.nodes[anc].obj)
        {
            out.push((anc, kind));
        }
        cur = eval_graph.nodes[anc].parent;
        guard += 1;
    }
    out.reverse();
    out
}

pub(super) fn evaluate_text_plus_3d_for_char(
    eval_graph: &EvaluatedSceneGraph3D<'_>,
    groups: &[(usize, TextPlus3DGroupKind)],
    char_idx: usize,
    total_chars: usize,
    env: &EvalEnv<'_>,
) -> (Vec3, Vec3, Vec3) {
    let mut pos = Vec3::ZERO;
    let mut rot = Vec3::ZERO;
    let mut scale = Vec3::ONE;
    let t_norm = if total_chars <= 1 {
        0.0
    } else {
        char_idx as f32 / (total_chars - 1) as f32
    };

    for &(g_idx, kind) in groups {
        let g_obj = eval_graph.nodes[g_idx].obj;
        let g_env = env.with_object(g_obj.id);
        let strength = eval_obj_f32(g_obj, &["strength", "amount"], 1.0, &g_env);
        match kind {
            TextPlus3DGroupKind::TextPlus => {
                let delay = eval_obj_f64(g_obj, &["delayPerCharacter", "delay"], 0.0, &g_env);
                let delay_order = eval_obj_str(g_obj, &["delayOrder"], &g_env)
                    .or_else(|| {
                        Some(
                            eval_obj_f64(g_obj, &["delayOrder"], 0.0, &g_env)
                                .round()
                                .to_string(),
                        )
                    })
                    .unwrap_or_default();
                let seed = eval_obj_f64(g_obj, &["randomSeed", "seed"], 1.0, &g_env);
                let char_env = if delay.abs() > 1e-6 && total_chars > 1 {
                    let order_rank =
                        text_plus_delay_rank(&delay_order, char_idx, total_chars, seed);
                    g_env.with_frame(g_env.frame - order_rank * delay)
                } else {
                    g_env.clone()
                };
                let p_off = eval_obj_vec3(
                    g_obj,
                    &["charPosition", "positionOffset"],
                    Vec3::ZERO,
                    &char_env,
                );
                let r_off = eval_obj_vec3(
                    g_obj,
                    &["charRotation", "rotationOffset"],
                    Vec3::ZERO,
                    &char_env,
                );
                let s_off =
                    eval_obj_vec3(g_obj, &["charScale", "scaleOffset"], Vec3::ONE, &char_env);
                pos += p_off * strength;
                rot += r_off * strength;
                scale *= Vec3::ONE.lerp(s_off, strength.clamp(0.0, 1.0));
            }
            TextPlus3DGroupKind::GradientTransform => {
                let reverse = eval_obj_bool(g_obj, &["reverse"], false, &g_env);
                let u = if reverse { 1.0 - t_norm } else { t_norm } * strength;
                let p_start = eval_obj_vec3(
                    g_obj,
                    &["firstPosition", "startPosition", "posStart"],
                    Vec3::ZERO,
                    &g_env,
                );
                let p_end = eval_obj_vec3(
                    g_obj,
                    &["lastPosition", "endPosition", "posEnd", "position"],
                    Vec3::ZERO,
                    &g_env,
                );
                let r_start = eval_obj_vec3(
                    g_obj,
                    &["firstRotation", "startRotation", "rotStart"],
                    Vec3::ZERO,
                    &g_env,
                );
                let r_end = eval_obj_vec3(
                    g_obj,
                    &["lastRotation", "endRotation", "rotEnd", "rotation"],
                    Vec3::ZERO,
                    &g_env,
                );
                let s_start = eval_obj_vec3(
                    g_obj,
                    &["firstScale", "startScale", "scaleStart"],
                    Vec3::ONE,
                    &g_env,
                );
                let s_end = eval_obj_vec3(
                    g_obj,
                    &["lastScale", "endScale", "scaleEnd", "scale"],
                    Vec3::ONE,
                    &g_env,
                );
                pos += p_start.lerp(p_end, u);
                rot += r_start.lerp(r_end, u);
                scale *= s_start.lerp(s_end, u);
            }
            TextPlus3DGroupKind::CharacterShake | TextPlus3DGroupKind::SelectedShake => {
                if kind == TextPlus3DGroupKind::SelectedShake {
                    let sel_raw = eval_obj_str(g_obj, &["characterNumbers", "characters"], &g_env)
                        .unwrap_or_else(|| "1".to_owned());
                    let sel = parse_character_selection(&sel_raw, total_chars);
                    if !sel.contains(&char_idx) {
                        continue;
                    }
                }
                let has_zoidium_shake = lookup_obj_prop_value(
                    g_obj,
                    &[
                        "shakeAmplitude",
                        "shakePeriod",
                        "shakePhase",
                        "shakeAxis",
                        "shakeDelayPerCharacter",
                        "shakeDelayOrder",
                        "shakeRandomSeed",
                    ],
                )
                .is_some();
                let p_amp = eval_obj_vec3(
                    g_obj,
                    &["positionAmplitude", "amplitude"],
                    Vec3::ZERO,
                    &g_env,
                );
                let r_amp = eval_obj_vec3(g_obj, &["rotationAmplitude"], Vec3::ZERO, &g_env);
                let s_amp = eval_obj_vec3(g_obj, &["scaleAmplitude"], Vec3::ZERO, &g_env);
                let has_legacy_vec_amp = p_amp.length_squared() > 1e-12
                    || r_amp.length_squared() > 1e-12
                    || s_amp.length_squared() > 1e-12;

                if has_zoidium_shake || !has_legacy_vec_amp {
                    let amp = eval_obj_f64(g_obj, &["shakeAmplitude"], 0.25, &g_env);
                    let period = eval_obj_f64(g_obj, &["shakePeriod"], 24.0, &g_env).max(0.1);
                    let phase = eval_obj_f64(g_obj, &["shakePhase"], 0.0, &g_env);
                    let delay = eval_obj_f64(
                        g_obj,
                        &["shakeDelayPerCharacter", "delayPerCharacter", "delay"],
                        0.0,
                        &g_env,
                    );
                    let delay_order =
                        eval_obj_str(g_obj, &["shakeDelayOrder", "delayOrder"], &g_env)
                            .or_else(|| {
                                Some(
                                    eval_obj_f64(
                                        g_obj,
                                        &["shakeDelayOrder", "delayOrder"],
                                        0.0,
                                        &g_env,
                                    )
                                    .round()
                                    .to_string(),
                                )
                            })
                            .unwrap_or_default();
                    let seed = eval_obj_f64(
                        g_obj,
                        &["shakeRandomSeed", "randomSeed", "seed"],
                        1.0,
                        &g_env,
                    );
                    let rank = text_plus_delay_rank(&delay_order, char_idx, total_chars, seed);
                    let ang = shake_angle(amp, period, env.frame - rank * delay + phase) as f32
                        * strength;
                    let axis_vec =
                        lookup_obj_prop_value(g_obj, &["shakeAxis"]).map_or(Vec3::Z, |track| {
                            match eval_property_track(&track, &g_env) {
                                PropertyValue::Vec3(v) | PropertyValue::Color(v) => {
                                    let raw = Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32);
                                    if raw.length_squared() > 1e-12 {
                                        raw.normalize()
                                    } else {
                                        Vec3::Z
                                    }
                                }
                                PropertyValue::Text(s) => {
                                    match s.trim().to_ascii_lowercase().as_str() {
                                        "x" | "0" => Vec3::X,
                                        "y" | "1" => Vec3::Y,
                                        _ => Vec3::Z,
                                    }
                                }
                                other => match other.as_f64().map(|v| v.round() as i32) {
                                    Some(0) => Vec3::X,
                                    Some(1) => Vec3::Y,
                                    _ => Vec3::Z,
                                },
                            }
                        });
                    rot += axis_vec * ang;
                }
                if has_legacy_vec_amp {
                    let freq = eval_obj_f64(g_obj, &["frequency", "speed"], 4.0, &g_env);
                    let stagger = eval_obj_f64(g_obj, &["stagger", "phaseOffset"], 0.35, &g_env);
                    let period = (env.fps / freq.max(0.01)).max(1.0);
                    let ph = env.frame + char_idx as f64 * stagger * period;
                    let ax = shake_angle(1.0, period, ph);
                    let ay = shake_angle(1.0, period, ph + period * 0.27);
                    let az = shake_angle(1.0, period, ph + period * 0.53);
                    pos += Vec3::new(
                        p_amp.x * ax as f32,
                        p_amp.y * ay as f32,
                        p_amp.z * az as f32,
                    ) * strength;
                    rot += Vec3::new(
                        r_amp.x * ax as f32,
                        r_amp.y * ay as f32,
                        r_amp.z * az as f32,
                    ) * strength;
                    scale *= Vec3::ONE
                        + Vec3::new(
                            s_amp.x * ax as f32,
                            s_amp.y * ay as f32,
                            s_amp.z * az as f32,
                        ) * strength;
                }
            }
            TextPlus3DGroupKind::SelectedTransform => {
                let sel_raw = eval_obj_str(g_obj, &["characterNumbers", "characters"], &g_env)
                    .unwrap_or_else(|| "1".to_owned());
                let sel = parse_character_selection(&sel_raw, total_chars);
                if !sel.contains(&char_idx) {
                    continue;
                }
                let p_off = eval_obj_vec3(
                    g_obj,
                    &["charPosition", "positionOffset", "position"],
                    Vec3::ZERO,
                    &g_env,
                );
                let r_off = eval_obj_vec3(
                    g_obj,
                    &["charRotation", "rotationOffset", "rotation"],
                    Vec3::ZERO,
                    &g_env,
                );
                let s_off = eval_obj_vec3(
                    g_obj,
                    &["charScale", "scaleOffset", "scale"],
                    Vec3::ONE,
                    &g_env,
                );
                pos += p_off * strength;
                rot += r_off * strength;
                scale *= Vec3::ONE.lerp(s_off, strength.clamp(0.0, 1.0));
            }
            TextPlus3DGroupKind::RandomScatter => {
                let seed = eval_obj_f64(g_obj, &["seed", "randomSeed"], 1.0, &g_env);
                let u01 = |ch: i32| text_seeded_random(seed, char_idx as i32, ch) as f32;
                let has_minmax = lookup_obj_prop_value(
                    g_obj,
                    &[
                        "positionMin",
                        "positionMax",
                        "rotationMin",
                        "rotationMax",
                        "scaleMin",
                        "scaleMax",
                    ],
                )
                .is_some();
                if has_minmax {
                    let pmin = eval_obj_vec3(g_obj, &["positionMin"], Vec3::ZERO, &g_env);
                    let pmax = eval_obj_vec3(g_obj, &["positionMax"], Vec3::ZERO, &g_env);
                    let rmin = eval_obj_vec3(g_obj, &["rotationMin"], Vec3::ZERO, &g_env);
                    let rmax = eval_obj_vec3(g_obj, &["rotationMax"], Vec3::ZERO, &g_env);
                    let smin = eval_obj_vec3(g_obj, &["scaleMin"], Vec3::ONE, &g_env);
                    let smax = eval_obj_vec3(g_obj, &["scaleMax"], Vec3::ONE, &g_env);
                    pos += Vec3::new(
                        pmin.x + (pmax.x - pmin.x) * u01(0),
                        pmin.y + (pmax.y - pmin.y) * u01(1),
                        pmin.z + (pmax.z - pmin.z) * u01(2),
                    ) * strength;
                    rot += Vec3::new(
                        rmin.x + (rmax.x - rmin.x) * u01(3),
                        rmin.y + (rmax.y - rmin.y) * u01(4),
                        rmin.z + (rmax.z - rmin.z) * u01(5),
                    ) * strength;
                    let s_rand = Vec3::new(
                        smin.x + (smax.x - smin.x) * u01(6),
                        smin.y + (smax.y - smin.y) * u01(7),
                        smin.z + (smax.z - smin.z) * u01(8),
                    );
                    scale *= Vec3::ONE.lerp(s_rand, strength.clamp(0.0, 1.0));
                } else {
                    let p_amp = eval_obj_vec3(
                        g_obj,
                        &["positionRange", "position", "scatter"],
                        Vec3::ZERO,
                        &g_env,
                    );
                    let r_amp =
                        eval_obj_vec3(g_obj, &["rotationRange", "rotation"], Vec3::ZERO, &g_env);
                    let s_amp = eval_obj_vec3(g_obj, &["scaleRange", "scale"], Vec3::ZERO, &g_env);
                    let r = |ch: i32| u01(ch) * 2.0 - 1.0;
                    pos += Vec3::new(p_amp.x * r(0), p_amp.y * r(1), p_amp.z * r(2)) * strength;
                    rot += Vec3::new(r_amp.x * r(3), r_amp.y * r(4), r_amp.z * r(5)) * strength;
                    scale *= Vec3::ONE
                        + Vec3::new(s_amp.x * r(6), s_amp.y * r(7), s_amp.z * r(8)) * strength;
                }
            }
            TextPlus3DGroupKind::CharacterMaterial => {}
        }
    }
    (pos, rot, scale)
}

pub(super) fn parse_inline_field(item: &serde_json::Value) -> Option<EffectorField> {
    let field_val = item.get("field").or_else(|| item.get("fieldType"));
    let field_type = match field_val {
        Some(serde_json::Value::Number(n)) => {
            EffectorFieldType::from_u32(n.as_u64().unwrap_or(0) as u32)
        }
        Some(serde_json::Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "linear" | "1" => EffectorFieldType::Linear,
            "box" | "2" => EffectorFieldType::Box,
            "sphere" | "3" => EffectorFieldType::Sphere,
            "cylinder" | "4" => EffectorFieldType::Cylinder,
            "noise" | "random" | "5" => EffectorFieldType::Noise,
            _ => EffectorFieldType::Infinite,
        },
        _ => EffectorFieldType::Infinite,
    };
    let invert = parse_preserved_bool_flag(item.get("fieldInvert"), false);
    if field_type == EffectorFieldType::Infinite && !invert {
        return None;
    }
    let curve = match item
        .get("fieldCurve")
        .or_else(|| item.get("fieldFalloffCurve"))
        .or_else(|| item.get("fieldRemap"))
    {
        Some(serde_json::Value::String(s)) => {
            match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
                "smooth" | "smoothstep" | "1" => FieldFalloffCurve::Smooth,
                "easein" | "ease-in" | "in" | "2" => FieldFalloffCurve::EaseIn,
                "easeout" | "ease-out" | "out" | "3" => FieldFalloffCurve::EaseOut,
                _ => FieldFalloffCurve::Linear,
            }
        }
        Some(serde_json::Value::Number(n)) => {
            FieldFalloffCurve::from_u32(n.as_u64().unwrap_or(0) as u32)
        }
        _ => FieldFalloffCurve::Linear,
    };
    Some(EffectorField {
        field_type,
        position: json_vec3(
            item.get("fieldPosition")
                .or_else(|| item.get("fieldOffset")),
            Vec3::ZERO,
        ),
        rotation_deg: json_vec3(item.get("fieldRotation"), Vec3::ZERO),
        scale: json_vec3(item.get("fieldScale"), Vec3::splat(100.0)),
        falloff: item
            .get("fieldFalloff")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(100.0) as f32,
        curve,
        invert,
        sweep: item
            .get("fieldSweep")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(100.0) as f32,
        noise_scale: item
            .get("fieldNoiseScale")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(100.0) as f32,
        evolution: item
            .get("fieldNoiseEvolution")
            .or_else(|| item.get("fieldEvolution"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32,
        seed: item
            .get("fieldSeed")
            .or_else(|| item.get("seed"))
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(1) as i32,
        legacy: parse_preserved_bool_flag(
            item.get("fieldLegacy").or_else(|| item.get("legacyField")),
            false,
        ),
    })
}

pub(super) fn apply_object_effectors(
    mut mesh: MeshData,
    obj: &Object3D,
    env: &EvalEnv<'_>,
) -> MeshData {
    if !(env.plugins.is_enabled("repeater") || env.plugins.is_enabled("effector-plus")) {
        return mesh;
    }
    let Some(list) = obj
        .preserved
        .get("effectors")
        .and_then(serde_json::Value::as_array)
    else {
        return mesh;
    };

    for item in list {
        let kind = item
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let axis = match item
            .get("axis")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("y")
            .to_ascii_lowercase()
            .as_str()
        {
            "x" => DeformAxis::X,
            "z" => DeformAxis::Z,
            _ => DeformAxis::Y,
        };
        let field = parse_inline_field(item);
        match kind.trim().to_ascii_lowercase().as_str() {
            "repeater" | "linear_repeater" | "linear-repeater" | "random_repeater"
            | "random-repeater" => {
                let mode = match kind.trim().to_ascii_lowercase().as_str() {
                    "linear_repeater" | "linear-repeater" => RepeaterMode::Linear,
                    "random_repeater" | "random-repeater" => RepeaterMode::Random,
                    _ => match item
                        .get("mode")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("step")
                        .to_ascii_lowercase()
                        .as_str()
                    {
                        "linear" => RepeaterMode::Linear,
                        "random" => RepeaterMode::Random,
                        _ => RepeaterMode::Step,
                    },
                };
                let count = item
                    .get("count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(1) as u32;
                let offset = json_vec3(
                    item.get("positionStep").or_else(|| item.get("offset")),
                    Vec3::ZERO,
                );
                let rotation = json_vec3(
                    item.get("rotationStep").or_else(|| item.get("rotation")),
                    Vec3::ZERO,
                );
                let scale = json_vec3(
                    item.get("scaleStep").or_else(|| item.get("scale")),
                    Vec3::ONE,
                );
                let params = RepeaterParams {
                    mode,
                    count: count.clamp(1, MAX_REPEATS),
                    euler_order: map_euler_order(obj.euler_order),
                    position_step: offset,
                    rotation_step_rad: rotation,
                    scale_step: scale,
                    position_end: json_vec3(
                        item.get("positionEnd").or_else(|| item.get("offset")),
                        Vec3::new(40.0, 0.0, 0.0),
                    ),
                    rotation_end_rad: json_vec3(
                        item.get("rotationEnd").or_else(|| item.get("rotation")),
                        Vec3::ZERO,
                    ),
                    scale_end: json_vec3(
                        item.get("scaleEnd").or_else(|| item.get("scale")),
                        Vec3::ONE,
                    ),
                    position_min: json_vec3(item.get("positionMin"), Vec3::splat(-10.0)),
                    position_max: json_vec3(item.get("positionMax"), Vec3::splat(10.0)),
                    rotation_min_rad: json_vec3(item.get("rotationMin"), Vec3::ZERO),
                    rotation_max_rad: json_vec3(item.get("rotationMax"), Vec3::ZERO),
                    scale_min: json_vec3(item.get("scaleMin"), Vec3::ONE),
                    scale_max: json_vec3(item.get("scaleMax"), Vec3::ONE),
                    seed: item
                        .get("seed")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(1) as i32,
                    delay_frames: 5.0,
                };
                mesh = apply_repeater_with_params(&mesh, &params);
            }
            "twist" => {
                let params = TwistParams {
                    angle: item
                        .get("angle")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    axis,
                    offset: item
                        .get("offset")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    subdivision: item
                        .get("subdivision")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(1) as u32,
                    field,
                };
                let bounds = mesh.bounds;
                mesh = apply_twist_with_params(&mesh, &params, Some(bounds), None);
            }
            "warp" => {
                let warp_axis = match item.get("axis") {
                    Some(serde_json::Value::Number(n)) => {
                        WarpAxis::from_u32(n.as_u64().unwrap_or(0) as u32)
                    }
                    _ => match axis {
                        DeformAxis::Z => WarpAxis::IntoZ,
                        DeformAxis::X | DeformAxis::Y => WarpAxis::IntoY,
                    },
                };
                let params = WarpParams {
                    strength: item
                        .get("strength")
                        .or_else(|| item.get("bend"))
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    axis: warp_axis,
                    offset: item
                        .get("offset")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    subdivision: item
                        .get("subdivision")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(1) as u32,
                    field,
                };
                mesh = apply_warp_with_params(&mesh, &params, None, None);
            }
            "shear" => {
                let direction = match item
                    .get("direction")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("x")
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "y" | "1" => DeformAxis::Y,
                    "z" | "2" => DeformAxis::Z,
                    _ => DeformAxis::X,
                };
                let params = ShearParams {
                    amount: item
                        .get("amount")
                        .or_else(|| item.get("strength"))
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    axis,
                    direction,
                    pivot: json_vec3(item.get("pivot"), Vec3::ZERO),
                    subdivision: item
                        .get("subdivision")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(1) as u32,
                    field,
                };
                mesh = apply_shear_with_params(&mesh, &params, None, None);
            }
            "plain" => {
                let params = PlainParams {
                    position_offset: json_vec3(
                        item.get("positionOffset").or_else(|| item.get("offset")),
                        Vec3::ZERO,
                    ),
                    rotation_offset_deg: json_vec3(
                        item.get("rotationOffset").or_else(|| item.get("rotation")),
                        Vec3::ZERO,
                    ),
                    scale_offset: json_vec3(
                        item.get("scaleOffset").or_else(|| item.get("scale")),
                        Vec3::ZERO,
                    ),
                    uniform_scale: item
                        .get("uniformScale")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    field,
                };
                mesh = apply_plain(&mesh, &params, None, None);
            }
            "voronoi_fracture" | "voronoi-fracture" | "fracture" => {
                let params = VoronoiFractureParams {
                    cells: item
                        .get("cell_count")
                        .or_else(|| item.get("cells"))
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(8) as u32,
                    seed: item
                        .get("seed")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(1) as i32,
                    closed: parse_preserved_bool_flag(item.get("closed"), true),
                    distance: item
                        .get("distance")
                        .or_else(|| item.get("explode"))
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    scatter: item
                        .get("scatter")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    offset: item
                        .get("offset")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    spin_rad: item
                        .get("spin")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    direction: json_vec3(item.get("direction"), Vec3::ZERO),
                    rotation_deg: json_vec3(item.get("rotationDeg"), Vec3::ZERO),
                    fragment_scale: item
                        .get("fragmentScale")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(100.0) as f32,
                    gravity: item
                        .get("gravity")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    random_colors: parse_preserved_bool_flag(item.get("colors"), false),
                    randomness: item
                        .get("randomness")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0) as f32,
                    field,
                    ..VoronoiFractureParams::default()
                };
                mesh = apply_voronoi_fracture_with_params(&mesh, &params, None, None);
            }
            _ => {}
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use zc_scene::{evaluate_field, prepare_field_for_bounds};

    fn make_unit_eval_env<'a>(
        project: &'a Project,
        plugins: &'a PluginRegistry,
        controls: &'a RefCell<BTreeMap<String, ExprValue>>,
        parent_props: &'a BTreeMap<String, ExprValue>,
        expr_stack: &'a RefCell<HashSet<String>>,
        frame: f64,
        fps: f64,
    ) -> EvalEnv<'a> {
        EvalEnv {
            frame,
            abs_frame: frame,
            time: frame / fps.max(1.0),
            fps,
            sequence_resolution: project.sequence.resolution,
            project,
            sequence: &project.sequence,
            plugins,
            controls,
            parent_props,
            active_clip_id: None,
            active_layer_id: None,
            active_object_id: None,
            active_effect_id: None,
            active_property_id: None,
            expr_stack,
        }
    }

    fn make_empty_group(name: &str) -> Object3D {
        Object3D::new(
            name,
            Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: Vec::new(),
            },
        )
    }

    fn mean_world_x(mesh: &MeshData, world: Mat4) -> f32 {
        assert!(!mesh.vertices.is_empty());
        let sum: f64 = mesh
            .vertices
            .iter()
            .map(|v| f64::from(world.transform_point3(Vec3::from_array(v.position)).x))
            .sum();
        (sum / mesh.vertices.len() as f64) as f32
    }

    #[test]
    fn shared_bounds_voronoi_fracture_aligns_multi_child_seams_and_explosion_center() {
        let proj = Project::new_cm3_default();
        let mut plugins = PluginRegistry::new();
        plugins.enable("repeater").expect("enable repeater");
        let controls = RefCell::new(BTreeMap::new());
        let parent_props = BTreeMap::new();
        let expr_stack = RefCell::new(HashSet::new());
        let env = make_unit_eval_env(
            &proj,
            &plugins,
            &controls,
            &parent_props,
            &expr_stack,
            0.0,
            30.0,
        );

        let mut left_box = Object3D::new_shape_3d(
            "LeftBox",
            Shape3DKind::Box {
                size: [10.0, 10.0, 10.0],
            },
        );
        left_box.position = PropertyTrack::vec3([-5.0, 0.0, 0.0]);

        let mut right_box = Object3D::new_shape_3d(
            "RightBox",
            Shape3DKind::Box {
                size: [10.0, 10.0, 10.0],
            },
        );
        right_box.position = PropertyTrack::vec3([5.0, 0.0, 0.0]);

        let group = Object3D::new(
            "PairGroup",
            Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: vec![left_box.clone(), right_box.clone()],
            },
        );

        let mut voronoi = Object3D::new(
            "Voronoi",
            Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: vec![group],
            },
        );
        voronoi.preserved.insert(
            "zoidiumPluginObjectType".to_owned(),
            serde_json::json!("zoidium:repeater/voronoi-fracture"),
        );
        voronoi
            .preserved
            .insert("property.cells".to_owned(), serde_json::json!(12));
        voronoi
            .preserved
            .insert("property.seed".to_owned(), serde_json::json!(7));
        voronoi
            .preserved
            .insert("property.distance".to_owned(), serde_json::json!(6.0));

        let objects = vec![voronoi.clone()];
        let eval_graph = evaluate_scene_graph_3d(&objects, Mat4::IDENTITY, &env);
        let effector_kinds: Vec<Option<SceneEffectorKind>> = eval_graph
            .nodes
            .iter()
            .map(|n| detect_scene_effector_kind(n.obj))
            .collect();
        let has_mesh_desc = vec![true; eval_graph.nodes.len()];

        let left_idx = eval_graph.by_uuid[&left_box.id];
        let right_idx = eval_graph.by_uuid[&right_box.id];
        let left_chain =
            collect_node_effector_chain(&eval_graph, left_idx, &effector_kinds, &has_mesh_desc);
        let right_chain =
            collect_node_effector_chain(&eval_graph, right_idx, &effector_kinds, &has_mesh_desc);
        assert_eq!(left_chain.len(), 1);
        assert_eq!(right_chain, left_chain);

        let base_box = Arc::new(build_box(10.0, 10.0, 10.0));
        let mut bounds_acc: HashMap<Uuid, (Aabb3, HashSet<usize>, usize)> = HashMap::new();
        accumulate_effector_bounds(
            &eval_graph,
            left_idx,
            &base_box,
            eval_graph.nodes[left_idx].world_mat,
            &left_chain,
            &effector_kinds,
            &mut bounds_acc,
        );
        accumulate_effector_bounds(
            &eval_graph,
            right_idx,
            &base_box,
            eval_graph.nodes[right_idx].world_mat,
            &right_chain,
            &effector_kinds,
            &mut bounds_acc,
        );
        let shared_bounds: HashMap<Uuid, SharedEffectorBounds> = bounds_acc
            .into_iter()
            .map(|(id, (bounds, nodes, mesh_count))| {
                (
                    id,
                    SharedEffectorBounds {
                        bounds,
                        distinct_nodes: nodes.len(),
                        mesh_count,
                    },
                )
            })
            .collect();

        let sb = shared_bounds[&voronoi.id];
        assert_eq!(sb.distinct_nodes, 2);
        assert_eq!(sb.mesh_count, 2);
        assert!((sb.bounds.min.x - (-10.0)).abs() < 1e-4);
        assert!((sb.bounds.max.x - 10.0).abs() < 1e-4);

        let mut vcache = HashMap::new();
        let mut tick = 0u64;
        let left_insts = apply_scene_effector_chain(
            &objects,
            Mat4::IDENTITY,
            &eval_graph,
            left_idx,
            Arc::clone(&base_box),
            eval_graph.nodes[left_idx].world_mat,
            &left_chain,
            &effector_kinds,
            &shared_bounds,
            None,
            &env,
            &mut vcache,
            &mut tick,
        );
        let right_insts = apply_scene_effector_chain(
            &objects,
            Mat4::IDENTITY,
            &eval_graph,
            right_idx,
            Arc::clone(&base_box),
            eval_graph.nodes[right_idx].world_mat,
            &right_chain,
            &effector_kinds,
            &shared_bounds,
            None,
            &env,
            &mut vcache,
            &mut tick,
        );

        let left_mean_x = mean_world_x(&left_insts[0].mesh, left_insts[0].world);
        let right_mean_x = mean_world_x(&right_insts[0].mesh, right_insts[0].world);
        assert!(
            left_mean_x < -6.0,
            "left sibling must explode outward in -X from shared center (mean_x={left_mean_x})"
        );
        assert!(
            right_mean_x > 6.0,
            "right sibling must explode outward in +X from shared center (mean_x={right_mean_x})"
        );
    }

    #[test]
    fn shared_bounds_warp_keeps_touching_sibling_seam_continuous() {
        let proj = Project::new_cm3_default();
        let mut plugins = PluginRegistry::new();
        plugins.enable("repeater").expect("enable repeater");
        let controls = RefCell::new(BTreeMap::new());
        let parent_props = BTreeMap::new();
        let expr_stack = RefCell::new(HashSet::new());
        let env = make_unit_eval_env(
            &proj,
            &plugins,
            &controls,
            &parent_props,
            &expr_stack,
            0.0,
            30.0,
        );

        let mut left_box = Object3D::new_shape_3d(
            "Left",
            Shape3DKind::Box {
                size: [10.0, 4.0, 4.0],
            },
        );
        left_box.position = PropertyTrack::vec3([-5.0, 0.0, 0.0]);
        let mut right_box = Object3D::new_shape_3d(
            "Right",
            Shape3DKind::Box {
                size: [10.0, 4.0, 4.0],
            },
        );
        right_box.position = PropertyTrack::vec3([5.0, 0.0, 0.0]);

        let mut warp = Object3D::new(
            "Warp",
            Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: vec![left_box.clone(), right_box.clone()],
            },
        );
        warp.preserved.insert(
            "zoidiumPluginObjectType".to_owned(),
            serde_json::json!("zoidium:repeater/warp"),
        );
        warp.preserved
            .insert("property.amount".to_owned(), serde_json::json!(1.2));
        warp.preserved
            .insert("property.subdivision".to_owned(), serde_json::json!(4));

        let objects = vec![warp.clone()];
        let eval_graph = evaluate_scene_graph_3d(&objects, Mat4::IDENTITY, &env);
        let effector_kinds: Vec<Option<SceneEffectorKind>> = eval_graph
            .nodes
            .iter()
            .map(|n| detect_scene_effector_kind(n.obj))
            .collect();
        let has_mesh_desc = vec![true; eval_graph.nodes.len()];
        let left_idx = eval_graph.by_uuid[&left_box.id];
        let right_idx = eval_graph.by_uuid[&right_box.id];
        let chain =
            collect_node_effector_chain(&eval_graph, left_idx, &effector_kinds, &has_mesh_desc);

        let base_mesh = Arc::new(build_box(10.0, 4.0, 4.0));
        let mut bounds_acc = HashMap::new();
        accumulate_effector_bounds(
            &eval_graph,
            left_idx,
            &base_mesh,
            eval_graph.nodes[left_idx].world_mat,
            &chain,
            &effector_kinds,
            &mut bounds_acc,
        );
        accumulate_effector_bounds(
            &eval_graph,
            right_idx,
            &base_mesh,
            eval_graph.nodes[right_idx].world_mat,
            &chain,
            &effector_kinds,
            &mut bounds_acc,
        );
        let shared_bounds: HashMap<Uuid, SharedEffectorBounds> = bounds_acc
            .into_iter()
            .map(|(id, (bounds, nodes, mesh_count))| {
                (
                    id,
                    SharedEffectorBounds {
                        bounds,
                        distinct_nodes: nodes.len(),
                        mesh_count,
                    },
                )
            })
            .collect();

        let mut vcache = HashMap::new();
        let mut tick = 0u64;
        let left_out = apply_scene_effector_chain(
            &objects,
            Mat4::IDENTITY,
            &eval_graph,
            left_idx,
            Arc::clone(&base_mesh),
            eval_graph.nodes[left_idx].world_mat,
            &chain,
            &effector_kinds,
            &shared_bounds,
            None,
            &env,
            &mut vcache,
            &mut tick,
        );
        let right_out = apply_scene_effector_chain(
            &objects,
            Mat4::IDENTITY,
            &eval_graph,
            right_idx,
            Arc::clone(&base_mesh),
            eval_graph.nodes[right_idx].world_mat,
            &chain,
            &effector_kinds,
            &shared_bounds,
            None,
            &env,
            &mut vcache,
            &mut tick,
        );

        // Vertices on the shared touching seam `x = 0` (the unified Warp pivot) must remain
        // at unbent Y = +-2.0 on BOTH siblings, proving unified bounds continuity.
        let seam_max_y = |inst: &DeformedInstance| -> f32 {
            inst.mesh
                .vertices
                .iter()
                .map(|v| inst.world.transform_point3(Vec3::from_array(v.position)))
                .filter(|p| p.x.abs() < 1e-2)
                .map(|p| p.y)
                .fold(f32::NEG_INFINITY, f32::max)
        };
        let ly = seam_max_y(&left_out[0]);
        let ry = seam_max_y(&right_out[0]);
        assert!(
            (ly - 2.0).abs() < 1e-3 && (ry - 2.0).abs() < 1e-3,
            "shared Warp pivot at x=0 must keep touching seam aligned (ly={ly}, ry={ry})"
        );
    }

    #[test]
    fn per_character_voronoi_fracture_modes_and_selection_rules() {
        let proj = Project::new_cm3_default();
        let plugins = PluginRegistry::new();
        let controls = RefCell::new(BTreeMap::new());
        let parent_props = BTreeMap::new();
        let expr_stack = RefCell::new(HashSet::new());
        let env = make_unit_eval_env(
            &proj,
            &plugins,
            &controls,
            &parent_props,
            &expr_stack,
            0.0,
            30.0,
        );

        // Empty `characters` with `invertCharacters = true` keeps whole-text path (Zoidium parity).
        let mut whole_obj = make_empty_group("VoronoiWhole");
        whole_obj
            .preserved
            .insert("property.characters".to_owned(), serde_json::json!(""));
        whole_obj
            .preserved
            .insert("property.invertCharacters".to_owned(), serde_json::json!(1));
        assert!(parse_voronoi_character_config(&whole_obj, 0, &env).is_none());
        assert_eq!(
            parse_voronoi_character_mode(&whole_obj, &env),
            VoronoiCharacterMode::Volume
        );

        // Explicit `byCharacter` / `fractureMode = "character"` -> WholeCharacter.
        let mut by_char = make_empty_group("VoronoiByChar");
        by_char
            .preserved
            .insert("property.byCharacter".to_owned(), serde_json::json!(true));
        assert_eq!(
            parse_voronoi_character_mode(&by_char, &env),
            VoronoiCharacterMode::WholeCharacter
        );
        assert!(parse_voronoi_character_config(&by_char, 0, &env).is_some());

        // Explicit `perCharacter` / `splitCharacters` -> PerCharacterSubdivide.
        let mut per_char = make_empty_group("VoronoiPerChar");
        per_char
            .preserved
            .insert("property.perCharacter".to_owned(), serde_json::json!("on"));
        assert_eq!(
            parse_voronoi_character_mode(&per_char, &env),
            VoronoiCharacterMode::PerCharacterSubdivide
        );
        assert!(parse_voronoi_character_config(&per_char, 0, &env).is_some());

        // WholeCharacter execution keeps each glyph intact while exploding outward from shared center.
        let glyph = build_box(4.0, 6.0, 2.0);
        let shared_text_bounds = Aabb3 {
            min: Vec3::new(-12.0, -3.0, -1.0),
            max: Vec3::new(12.0, 3.0, 1.0),
        };
        let params = VoronoiFractureParams {
            cells: 12,
            seed: 5,
            distance: 5.0,
            spin_rad: 0.6,
            random_colors: true,
            ..VoronoiFractureParams::default()
        };
        let mut vcache = HashMap::new();
        let mut tick = 0u64;
        let df_left = compute_deformer_frame(
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(-10.0, 0.0, 0.0)),
        );
        let df_right = compute_deformer_frame(
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        );
        let out_left = apply_voronoi_to_instance(
            &glyph,
            &params,
            VoronoiCharacterMode::WholeCharacter,
            Some(CharacterFractureContext { char_index: 0 }),
            Some(shared_text_bounds),
            false,
            df_left.as_ref(),
            &mut vcache,
            &mut tick,
        );
        let out_right = apply_voronoi_to_instance(
            &glyph,
            &params,
            VoronoiCharacterMode::WholeCharacter,
            Some(CharacterFractureContext { char_index: 2 }),
            Some(shared_text_bounds),
            false,
            df_right.as_ref(),
            &mut vcache,
            &mut tick,
        );
        assert_eq!(out_left.vertices.len(), glyph.vertices.len());
        assert_eq!(out_right.vertices.len(), glyph.vertices.len());
        let left_world = Mat4::from_translation(Vec3::new(-10.0, 0.0, 0.0));
        let right_world = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        assert!(mean_world_x(&out_left, left_world) < -14.0);
        assert!(mean_world_x(&out_right, right_world) > 14.0);
        assert_ne!(out_left.colors[0], out_right.colors[0]);

        // PerCharacterSubdivide fractures repeated glyphs with distinct character-aligned seeds.
        let sub_0 = apply_voronoi_to_instance(
            &glyph,
            &params,
            VoronoiCharacterMode::PerCharacterSubdivide,
            Some(CharacterFractureContext { char_index: 0 }),
            Some(shared_text_bounds),
            false,
            df_left.as_ref(),
            &mut vcache,
            &mut tick,
        );
        let sub_1 = apply_voronoi_to_instance(
            &glyph,
            &params,
            VoronoiCharacterMode::PerCharacterSubdivide,
            Some(CharacterFractureContext { char_index: 1 }),
            Some(shared_text_bounds),
            false,
            df_left.as_ref(),
            &mut vcache,
            &mut tick,
        );
        assert!(sub_0.vertices.len() > glyph.vertices.len());
        assert_ne!(sub_0.vertices[0].position, sub_1.vertices[0].position);
    }

    #[test]
    fn field_falloff_invert_clamp_remap_and_noise_math() {
        let proj = Project::new_cm3_default();
        let plugins = PluginRegistry::new();
        let controls = RefCell::new(BTreeMap::new());
        let parent_props = BTreeMap::new();
        let expr_stack = RefCell::new(HashSet::new());
        let env_t2 = make_unit_eval_env(
            &proj,
            &plugins,
            &controls,
            &parent_props,
            &expr_stack,
            60.0,
            30.0,
        );

        // Verify field alias parsing: `random` -> Noise, `fieldOffset`, `fieldRemap`, `fieldInvert`,
        // `fieldNoiseSpeed`, and `seed` fallback (`scene-fields.js:310`).
        let mut obj = make_empty_group("FieldObj");
        obj.preserved
            .insert("property.field".to_owned(), serde_json::json!("random"));
        obj.preserved.insert(
            "property.fieldOffset".to_owned(),
            serde_json::json!([2.0, -3.0, 4.0]),
        );
        obj.preserved.insert(
            "property.fieldRotation".to_owned(),
            serde_json::json!([0.0, 45.0, 0.0]),
        );
        obj.preserved
            .insert("property.fieldFalloff".to_owned(), serde_json::json!(50.0));
        obj.preserved.insert(
            "property.fieldRemap".to_owned(),
            serde_json::json!("ease-in"),
        );
        obj.preserved
            .insert("property.fieldInvert".to_owned(), serde_json::json!("on"));
        obj.preserved.insert(
            "property.fieldNoiseScale".to_owned(),
            serde_json::json!(40.0),
        );
        obj.preserved.insert(
            "property.fieldNoiseSpeed".to_owned(),
            serde_json::json!(1.5),
        );
        obj.preserved
            .insert("property.seed".to_owned(), serde_json::json!(42));

        let parsed = parse_effector_field_from_obj(&obj, &env_t2).expect("parsed noise field");
        assert_eq!(parsed.field_type, EffectorFieldType::Noise);
        assert_eq!(parsed.position, Vec3::new(2.0, -3.0, 4.0));
        assert_eq!(parsed.rotation_deg, Vec3::new(0.0, 45.0, 0.0));
        assert!((parsed.falloff - 50.0).abs() < 1e-5);
        assert_eq!(parsed.curve, FieldFalloffCurve::EaseIn);
        assert!(parsed.invert);
        assert!((parsed.noise_scale - 40.0).abs() < 1e-5);
        assert!((parsed.evolution - 3.0).abs() < 1e-5);
        assert_eq!(parsed.seed, 42);

        // Verify Linear sweep, Box, Sphere falloff, invert, clamp [0, 1], and remap curves.
        let bounds = Aabb3 {
            min: Vec3::new(-10.0, -10.0, -10.0),
            max: Vec3::new(10.0, 10.0, 10.0),
        };
        let linear_field = EffectorField {
            field_type: EffectorFieldType::Linear,
            sweep: 50.0,
            falloff: 50.0,
            curve: FieldFalloffCurve::Linear,
            invert: false,
            ..EffectorField::default()
        };
        let prep_lin = prepare_field_for_bounds(&linear_field, bounds);
        let w_left = evaluate_field(Vec3::new(-10.0, 0.0, 0.0), &prep_lin);
        let w_mid = evaluate_field(Vec3::new(0.0, 0.0, 0.0), &prep_lin);
        let w_right = evaluate_field(Vec3::new(10.0, 0.0, 0.0), &prep_lin);
        assert!((w_left - 1.0).abs() < 1e-6);
        assert!((w_mid - 0.5).abs() < 1e-6);
        assert!(w_right.abs() < 1e-6);

        let linear_inv = EffectorField {
            invert: true,
            ..linear_field.clone()
        };
        let prep_inv = prepare_field_for_bounds(&linear_inv, bounds);
        assert!(evaluate_field(Vec3::new(-10.0, 0.0, 0.0), &prep_inv).abs() < 1e-6);
        assert!((evaluate_field(Vec3::new(10.0, 0.0, 0.0), &prep_inv) - 1.0).abs() < 1e-6);

        let ease_in_field = EffectorField {
            curve: FieldFalloffCurve::EaseIn,
            ..linear_field.clone()
        };
        let ease_out_field = EffectorField {
            curve: FieldFalloffCurve::EaseOut,
            ..linear_field
        };
        let prep_in = prepare_field_for_bounds(&ease_in_field, bounds);
        let prep_out = prepare_field_for_bounds(&ease_out_field, bounds);
        assert!((evaluate_field(Vec3::ZERO, &prep_in) - 0.25).abs() < 1e-6);
        assert!((evaluate_field(Vec3::ZERO, &prep_out) - 0.75).abs() < 1e-6);

        // Sphere field clamped strictly to [0, 1] across extreme coordinates.
        let sphere_field = EffectorField {
            field_type: EffectorFieldType::Sphere,
            falloff: 100.0,
            ..EffectorField::default()
        };
        let prep_sph = prepare_field_for_bounds(&sphere_field, bounds);
        assert!((evaluate_field(Vec3::ZERO, &prep_sph) - 1.0).abs() < 1e-6);
        assert!(evaluate_field(Vec3::new(100.0, 0.0, 0.0), &prep_sph).abs() < 1e-6);
        assert!((evaluate_field(Vec3::new(5.0, 0.0, 0.0), &prep_sph) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn echo_repeater_evaluates_historical_frames_anchored_to_current_repeater_pose() {
        let proj = Project::new_cm3_default();
        let mut plugins = PluginRegistry::new();
        plugins.enable("repeater").expect("enable repeater");
        let controls = RefCell::new(BTreeMap::new());
        let parent_props = BTreeMap::new();
        let expr_stack = RefCell::new(HashSet::new());
        let env = make_unit_eval_env(
            &proj,
            &plugins,
            &controls,
            &parent_props,
            &expr_stack,
            15.0,
            30.0,
        );

        let linear_vec3_track = |x0: f64, x1: f64, end_frame: f64| PropertyTrack {
            animated: true,
            expression: None,
            static_behavior: ProjectStaticBehavior::Constant,
            value: PropertyValue::Vec3([x0, 0.0, 0.0]),
            keyframes: vec![
                PropertyKeyframe::from_cm3_tween(
                    0.0,
                    PropertyValue::Vec3([x0, 0.0, 0.0]),
                    1,
                    [[-10.0, 0.0], [10.0, 0.0]],
                    true,
                ),
                PropertyKeyframe::from_cm3_tween(
                    end_frame,
                    PropertyValue::Vec3([x1, 0.0, 0.0]),
                    1,
                    [[-10.0, 0.0], [10.0, 0.0]],
                    true,
                ),
            ],
            channels: Vec::new(),
        };

        let mut child = Object3D::new_shape_3d(
            "MovingBox",
            Shape3DKind::Box {
                size: [4.0, 4.0, 4.0],
            },
        );
        // Child moves 1 unit/frame from x=0 at frame 0 to x=30 at frame 30.
        child.position = linear_vec3_track(0.0, 30.0, 30.0);

        let mut echo = Object3D::new(
            "Echo",
            Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: Vec::new(),
                children: vec![child.clone()],
            },
        );
        echo.preserved.insert(
            "zoidiumPluginObjectType".to_owned(),
            serde_json::json!("zoidium:repeater/echo-repeater"),
        );
        echo.preserved.insert(
            "repeaterProperties".to_owned(),
            serde_json::json!({ "count": 3, "delay": 5.0 }),
        );
        // Echo Repeater itself moves to x=100 at frame 15; copies must stay anchored in its current frame.
        echo.position = linear_vec3_track(0.0, 100.0, 15.0);

        let objects = vec![echo];
        let eval_graph = evaluate_scene_graph_3d(&objects, Mat4::IDENTITY, &env);
        let effector_kinds: Vec<Option<SceneEffectorKind>> = eval_graph
            .nodes
            .iter()
            .map(|n| detect_scene_effector_kind(n.obj))
            .collect();
        let has_mesh_desc = vec![true; eval_graph.nodes.len()];
        let child_idx = eval_graph.by_uuid[&child.id];
        let chain =
            collect_node_effector_chain(&eval_graph, child_idx, &effector_kinds, &has_mesh_desc);

        let mut vcache = HashMap::new();
        let mut tick = 0u64;
        let insts = apply_scene_effector_chain(
            &objects,
            Mat4::IDENTITY,
            &eval_graph,
            child_idx,
            Arc::new(build_box(4.0, 4.0, 4.0)),
            eval_graph.nodes[child_idx].world_mat,
            &chain,
            &effector_kinds,
            &HashMap::new(),
            None,
            &env,
            &mut vcache,
            &mut tick,
        );
        assert_eq!(insts.len(), 3);
        let xs: Vec<f32> = insts.iter().map(|i| i.world.w_axis.x).collect();
        assert!(
            (xs[0] - 115.0).abs() < 1e-3
                && (xs[1] - 110.0).abs() < 1e-3
                && (xs[2] - 105.0).abs() < 1e-3,
            "Echo Repeater must evaluate child at frames [15, 10, 5] anchored to current repeater x=100, got {xs:?}"
        );
    }
}
