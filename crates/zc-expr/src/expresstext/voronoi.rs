use super::*;

/// Halton low-discrepancy sequence value in `[0, 1)` (`expresstext-voronoi.js:10-21`).
pub fn halton_sequence(index: f64, base: f64) -> f64 {
    let mut f = 1.0;
    let mut r = 0.0;
    let mut i = if index.is_finite() {
        (index.floor() as i64).max(1)
    } else {
        1
    };
    let b = if base.is_finite() {
        (base.floor() as i64).max(2)
    } else {
        2
    };
    while i > 0 {
        f /= b as f64;
        r += f * ((i % b) as f64);
        i /= b;
    }
    r
}

/// One raw Voronoi cell partitioned from a glyph's triangles.
#[derive(Debug, Clone, PartialEq)]
pub struct VoronoiCell {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub uvs: Vec<f32>,
    pub center: [f64; 3],
    pub triangle_indices: Vec<usize>,
}

/// Offset and Euler rotation (radians) of one Voronoi chunk (`expresstext-voronoi.js:111-126`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoronoiChunkPose {
    pub offset: [f64; 3],
    pub rotation: [f64; 3],
}

/// Deterministic Voronoi shatter parameters for a single glyph mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoronoiGlyphParams {
    /// Target Voronoi cell count (`1..=64`, Zoidium `voronoiChunks`, default `8`).
    pub cells: usize,
    /// Deterministic site & sign seed (Zoidium `voronoiSeed`, default `1.0`).
    pub seed: f64,
    /// Extra deterministic per-chunk random scatter displacement (default `0.0`).
    pub scatter: f64,
    /// Radial outward explosion distance (Zoidium `voronoiExplode`, default `0.0`).
    pub distance: f64,
    /// Tumble Euler rotation amplitude in radians (Zoidium `voronoiSpin`, default `[0, 0, 0]`).
    pub spin: [f64; 3],
    /// Downward gravity displacement along `-Y` (default `0.0`).
    pub gravity: f64,
    /// Site jitter blend in `[0, 1]` (`1.0` = full Halton distribution matching Zoidium).
    pub randomness: f64,
    /// Uniform scale multiplier per shattered chunk (default `1.0`).
    pub scale: f64,
    /// Effective effector weight (strength * field falloff, default `1.0`).
    pub weight: f64,
    /// Letter index used to salt deterministic per-chunk spin signs.
    pub letter_index: usize,
}

impl Default for VoronoiGlyphParams {
    fn default() -> Self {
        Self {
            cells: 8,
            seed: 1.0,
            scatter: 0.0,
            distance: 0.0,
            spin: [0.0, 0.0, 0.0],
            gravity: 0.0,
            randomness: 1.0,
            scale: 1.0,
            weight: 1.0,
            letter_index: 0,
        }
    }
}

/// One shattered glyph chunk ready for rendering in `zc-render`.
#[derive(Debug, Clone, PartialEq)]
pub struct ShatteredVoronoiChunk {
    pub cell_index: usize,
    /// Original cell triangle vertex positions (9 floats per triangle).
    pub positions: Vec<f32>,
    /// Original cell triangle vertex normals (9 floats per triangle, if provided).
    pub normals: Vec<f32>,
    /// Original cell triangle vertex UVs (6 floats per triangle, if provided).
    pub uvs: Vec<f32>,
    /// Centroid of the cell in glyph-local space.
    pub center: [f64; 3],
    /// World/glyph-local displacement vector for this chunk.
    pub offset: [f64; 3],
    /// Signed tumble Euler rotation (radians) about `center`.
    pub rotation: [f64; 3],
    /// Chunk scale multiplier about `center`.
    pub scale: [f64; 3],
    /// Deterministic per-axis spin signs (`-1.0` or `1.0`).
    pub signs: [f64; 3],
}

/// Partitions flat triangle arrays (`9` floats per triangle in `positions` / `normals`,
/// `6` floats per triangle in `uvs`) into Halton-seeded Voronoi cells matching Zoidium
/// `expresstext-voronoi.js:splitVoronoiCells`.
pub fn split_voronoi_cells(
    positions: &[f32],
    normals: &[f32],
    uvs: &[f32],
    count: usize,
    seed: f64,
) -> Vec<VoronoiCell> {
    split_voronoi_cells_with_randomness(positions, normals, uvs, count, seed, 1.0)
}

fn split_voronoi_cells_with_randomness(
    positions: &[f32],
    normals: &[f32],
    uvs: &[f32],
    count: usize,
    seed: f64,
    randomness: f64,
) -> Vec<VoronoiCell> {
    let tri_count = positions.len() / 9;
    let want = count.clamp(1, VORONOI_MAX_CHUNKS);
    if tri_count == 0 {
        return Vec::new();
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    let usable = tri_count * 9;
    for chunk in positions[..usable].as_chunks::<3>().0 {
        let x = f64::from(chunk[0]);
        let y = f64::from(chunk[1]);
        let z = f64::from(chunk[2]);
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if y > max_y {
            max_y = y;
        }
        if z < min_z {
            min_z = z;
        }
        if z > max_z {
            max_z = z;
        }
    }

    let seed_num = if seed.is_finite() { seed } else { 1.0 };
    let seed_frac = seed_num - seed_num.floor() + if seed_num < 0.0 { 1.0 } else { 0.0 };
    let span_x = if max_x > min_x { max_x - min_x } else { 1.0 };
    let span_y = if max_y > min_y { max_y - min_y } else { 1.0 };
    let span_z = if max_z > min_z { max_z - min_z } else { 1.0 };
    let rand_blend = if randomness.is_finite() {
        randomness.clamp(0.0, 1.0)
    } else {
        1.0
    };

    let site_count = want.min(tri_count);
    let mut sites = Vec::with_capacity(site_count);
    for s in 0..site_count {
        let hx = (halton_sequence((s + 1) as f64, 2.0) + seed_frac * 0.618_033_988_7) % 1.0;
        let hy = (halton_sequence((s + 1) as f64, 3.0) + seed_frac * 0.381_966_011_3) % 1.0;
        let hz = (halton_sequence((s + 1) as f64, 5.0) + seed_frac * 0.5) % 1.0;
        let ux = (s as f64 + 0.5) / (site_count as f64);
        let jx = ux + (hx - ux) * rand_blend;
        let jy = 0.5 + (hy - 0.5) * rand_blend;
        let jz = 0.5 + (hz - 0.5) * rand_blend;
        sites.push([
            min_x + span_x * jx,
            min_y + span_y * jy,
            min_z + span_z * jz,
        ]);
    }

    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); site_count];
    for t in 0..tri_count {
        let base = t * 9;
        let cx = (f64::from(positions[base])
            + f64::from(positions[base + 3])
            + f64::from(positions[base + 6]))
            / 3.0;
        let cy = (f64::from(positions[base + 1])
            + f64::from(positions[base + 4])
            + f64::from(positions[base + 7]))
            / 3.0;
        let cz = (f64::from(positions[base + 2])
            + f64::from(positions[base + 5])
            + f64::from(positions[base + 8]))
            / 3.0;
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for (s, site) in sites.iter().enumerate() {
            let dx = cx - site[0];
            let dy = cy - site[1];
            let dz = cz - site[2];
            let d = dx * dx + dy * dy + dz * dz;
            if d < best_d {
                best_d = d;
                best = s;
            }
        }
        buckets[best].push(t);
    }

    let mut out = Vec::new();
    for tris in buckets {
        if tris.is_empty() {
            continue;
        }
        let mut pos_out = Vec::with_capacity(tris.len() * 9);
        let mut nor_out = Vec::with_capacity(if normals.is_empty() {
            0
        } else {
            tris.len() * 9
        });
        let mut uv_out = Vec::with_capacity(if uvs.is_empty() { 0 } else { tris.len() * 6 });
        let mut ccx = 0.0;
        let mut ccy = 0.0;
        let mut ccz = 0.0;
        for &t in &tris {
            let p_base = t * 9;
            pos_out.extend_from_slice(&positions[p_base..p_base + 9]);
            if normals.len() >= p_base + 9 {
                nor_out.extend_from_slice(&normals[p_base..p_base + 9]);
            }
            let u_base = t * 6;
            if uvs.len() >= u_base + 6 {
                uv_out.extend_from_slice(&uvs[u_base..u_base + 6]);
            }
            ccx += (f64::from(positions[p_base])
                + f64::from(positions[p_base + 3])
                + f64::from(positions[p_base + 6]))
                / 3.0;
            ccy += (f64::from(positions[p_base + 1])
                + f64::from(positions[p_base + 4])
                + f64::from(positions[p_base + 7]))
                / 3.0;
            ccz += (f64::from(positions[p_base + 2])
                + f64::from(positions[p_base + 5])
                + f64::from(positions[p_base + 8]))
                / 3.0;
        }
        let n = tris.len() as f64;
        out.push(VoronoiCell {
            positions: pos_out,
            normals: nor_out,
            uvs: uv_out,
            center: [ccx / n, ccy / n, ccz / n],
            triangle_indices: tris,
        });
    }
    out
}

/// Computes radial explosion offset and signed spin for one Voronoi chunk, matching Zoidium
/// `expresstext-voronoi.js:voronoiChunkPose`.
pub fn voronoi_chunk_pose(
    chunk_center: [f64; 3],
    letter_center: [f64; 3],
    explode: f64,
    spin: [f64; 3],
    signs: [f64; 3],
) -> VoronoiChunkPose {
    let dx = chunk_center[0] - letter_center[0];
    let dy = chunk_center[1] - letter_center[1];
    let dz = chunk_center[2] - letter_center[2];
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    let dir = if len > 1e-9 {
        [dx / len, dy / len, dz / len]
    } else {
        [0.0, 1.0, 0.0]
    };
    let ex = if explode.is_finite() { explode } else { 0.0 };
    VoronoiChunkPose {
        offset: [dir[0] * ex, dir[1] * ex, dir[2] * ex],
        rotation: [spin[0] * signs[0], spin[1] * signs[1], spin[2] * signs[2]],
    }
}

/// Deterministically shatters a glyph's triangle mesh into Halton-seeded Voronoi chunks with
/// evaluated explosion, scatter, gravity, tumble spin, and per-chunk scale.
pub fn shatter_glyph_triangles(
    positions: &[f32],
    normals: &[f32],
    uvs: &[f32],
    letter_center: Option<[f64; 3]>,
    params: &VoronoiGlyphParams,
) -> Vec<ShatteredVoronoiChunk> {
    let cells = split_voronoi_cells_with_randomness(
        positions,
        normals,
        uvs,
        params.cells,
        params.seed,
        params.randomness,
    );
    if cells.is_empty() {
        return Vec::new();
    }

    let center = letter_center.unwrap_or_else(|| {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        let usable = (positions.len() / 9) * 9;
        for v in positions[..usable].as_chunks::<3>().0 {
            for axis in 0..3 {
                let c = f64::from(v[axis]);
                if c < min[axis] {
                    min[axis] = c;
                }
                if c > max[axis] {
                    max[axis] = c;
                }
            }
        }
        [
            0.5 * (min[0] + max[0]),
            0.5 * (min[1] + max[1]),
            0.5 * (min[2] + max[2]),
        ]
    });

    let seed_num = if params.seed.is_finite() {
        params.seed
    } else {
        1.0
    };
    let seed_int = js_round(seed_num * 1000.0) as i32;
    let w = if params.weight.is_finite() {
        params.weight.max(0.0)
    } else {
        0.0
    };
    let explode = if params.distance.is_finite() {
        params.distance * w
    } else {
        0.0
    };
    let spin = [params.spin[0] * w, params.spin[1] * w, params.spin[2] * w];
    let scatter = if params.scatter.is_finite() {
        params.scatter * w
    } else {
        0.0
    };
    let gravity = if params.gravity.is_finite() {
        params.gravity * w
    } else {
        0.0
    };
    let chunk_scale = if params.scale.is_finite() {
        (1.0 + (params.scale - 1.0) * w).max(1e-6)
    } else {
        1.0
    };

    let mut out = Vec::with_capacity(cells.len());
    for (ci, cell) in cells.into_iter().enumerate() {
        let mut signs = [1.0; 3];
        for (ax, sign) in signs.iter_mut().enumerate() {
            let iy = (params.letter_index as i32)
                .wrapping_mul(131)
                .wrapping_add((ax as i32).wrapping_mul(17));
            *sign = if hash_lattice(ci as i32, iy, seed_int, 0) > 0.5 {
                1.0
            } else {
                -1.0
            };
        }

        let mut pose = voronoi_chunk_pose(cell.center, center, explode, spin, signs);

        if scatter != 0.0 {
            for (ax, off) in pose.offset.iter_mut().enumerate() {
                let jitter = hash_lattice(
                    ci as i32,
                    (params.letter_index as i32).wrapping_mul(73) + 19,
                    (ax as i32).wrapping_mul(29) + 5,
                    seed_int,
                ) * 2.0
                    - 1.0;
                *off += jitter * scatter;
            }
        }
        if gravity != 0.0 {
            let g_var = 0.5
                + 0.5
                    * hash_lattice(
                        ci as i32,
                        (params.letter_index as i32).wrapping_mul(97) + 31,
                        41,
                        seed_int,
                    );
            pose.offset[1] -= gravity * g_var;
        }

        out.push(ShatteredVoronoiChunk {
            cell_index: ci,
            positions: cell.positions,
            normals: cell.normals,
            uvs: cell.uvs,
            center: cell.center,
            offset: pose.offset,
            rotation: pose.rotation,
            scale: [chunk_scale; 3],
            signs,
        });
    }
    out
}

/// Applies a [`ShatteredVoronoiChunk`]'s pose (`scale`, `rotation`, `offset`) around its
/// centroid to produce baked triangle vertex positions.
pub fn apply_chunk_pose_to_positions(chunk: &ShatteredVoronoiChunk) -> Vec<f32> {
    let (sx, cx) = chunk.rotation[0].sin_cos();
    let (sy, cy) = chunk.rotation[1].sin_cos();
    let (sz, cz) = chunk.rotation[2].sin_cos();

    // Precompute R = Rx * Ry * Rz once per chunk (matches `rotate_euler_xyz`).
    let m00 = cy * cz;
    let m01 = -cy * sz;
    let m02 = sy;
    let m10 = cx * sz + sx * sy * cz;
    let m11 = cx * cz - sx * sy * sz;
    let m12 = -sx * cy;
    let m20 = sx * sz - cx * sy * cz;
    let m21 = sx * cz + cx * sy * sz;
    let m22 = cx * cy;

    let [c0, c1, c2] = chunk.center;
    let [s0, s1, s2] = chunk.scale;
    let tx = c0 + chunk.offset[0];
    let ty = c1 + chunk.offset[1];
    let tz = c2 + chunk.offset[2];

    let mut baked = Vec::with_capacity(chunk.positions.len());
    for v in chunk.positions.as_chunks::<3>().0 {
        let rx = (f64::from(v[0]) - c0) * s0;
        let ry = (f64::from(v[1]) - c1) * s1;
        let rz = (f64::from(v[2]) - c2) * s2;
        baked.push((m00 * rx + m01 * ry + m02 * rz + tx) as f32);
        baked.push((m10 * rx + m11 * ry + m12 * rz + ty) as f32);
        baked.push((m20 * rx + m21 * ry + m22 * rz + tz) as f32);
    }
    baked
}

/// Evaluates all enabled Voronoi effector layers in `layers` for one letter at `frame`
/// (partition cell count & seed from the first enabled Voronoi layer; explode, spin,
/// scatter, and gravity summed across all active layers, matching `expresstext-render.js:712-724,864-882`).
pub fn evaluate_expresstext_voronoi_params(
    layers: &[EffectorLayer<'_>],
    frame: f64,
    index: usize,
    list: &LetterList,
    base: [f64; 3],
) -> Option<VoronoiGlyphParams> {
    let sections = sections_at(layers, frame);
    let mut first_partition: Option<(usize, f64, f64)> = None;
    let mut explode = 0.0;
    let mut spin = [0.0; 3];
    let mut scatter = 0.0;
    let mut gravity = 0.0;
    let mut scale_acc = 1.0;
    let mut total_weight = 0.0;

    for layer in layers {
        if layer.kind != EffectorKind::Voronoi {
            continue;
        }
        if read_num(layer.props, "voronoiEnable", frame, 1.0) == 0.0 {
            continue;
        }
        if first_partition.is_none() {
            let raw_chunks = read_num(
                layer.props,
                if layer.props.read("voronoiChunks", frame).is_some() {
                    "voronoiChunks"
                } else {
                    "voronoiCells"
                },
                frame,
                8.0,
            );
            let chunks = (if raw_chunks.is_finite() {
                raw_chunks.floor() as isize
            } else {
                8
            })
            .clamp(0, VORONOI_MAX_CHUNKS as isize) as usize;
            if chunks > 0 {
                let seed = read_num(layer.props, "voronoiSeed", frame, 1.0);
                let randomness =
                    read_num(layer.props, "voronoiRandomness", frame, 1.0).clamp(0.0, 1.0);
                first_partition = Some((chunks, seed, randomness));
            }
        }
        let ctx = letter_context(
            layer.props,
            EffectorKind::Voronoi,
            frame,
            index,
            list,
            &sections,
        );
        if !is_letter_selected(&ctx.selected, index) {
            continue;
        }
        let w = apply_field_weight(
            ctx.strength,
            layer.props,
            EffectorKind::Voronoi,
            frame,
            base,
        );
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let local = frame - ctx.delay;
        let dist_key = if layer.props.read("voronoiExplode", local).is_some() {
            "voronoiExplode"
        } else {
            "voronoiDistance"
        };
        explode += read_num(layer.props, dist_key, local, 0.0) * w;
        let sv = read_vec3(layer.props, "voronoiSpin", local, [0.0, 0.0, 0.0]);
        spin[0] += sv[0] * w;
        spin[1] += sv[1] * w;
        spin[2] += sv[2] * w;
        scatter += read_num(layer.props, "voronoiScatter", local, 0.0) * w;
        gravity += read_num(layer.props, "voronoiGravity", local, 0.0) * w;
        let s = read_num(layer.props, "voronoiScale", local, 1.0);
        scale_acc *= (1.0 + (s - 1.0) * w).max(1e-6);
        total_weight += w;
    }

    let (cells, seed, randomness) = first_partition?;
    if total_weight <= 0.0 {
        return None;
    }
    Some(VoronoiGlyphParams {
        cells,
        seed,
        scatter,
        distance: explode,
        spin,
        gravity,
        randomness,
        scale: scale_acc,
        // Distance/spin/scatter/gravity/scale are already weighted per layer above.
        weight: 1.0,
        letter_index: index,
    })
}

// ---------------------------------------------------------------------------
// Zoidium `expressProperties` JSON Parser (`Vec<ExpressEffectorConfig>`)
