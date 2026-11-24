use std::collections::HashMap;

use thiserror::Error;
use zc_geom::{Vec2, Vec3};

use crate::mesh::{MeshData, Vertex3D};

/// Errors returned while parsing a Wavefront `.obj` mesh.
#[derive(Debug, Error)]
pub enum ObjLoadError {
    #[error("invalid OBJ syntax at line {line}: {message}")]
    InvalidSyntax { line: usize, message: String },
    #[error("OBJ index out of bounds at line {line}: {index}")]
    IndexOutOfBounds { line: usize, index: i64 },
    #[error("OBJ contains no valid geometry")]
    EmptyMesh,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum SmoothKey {
    /// Vertex already specifies an explicit `vn` index.
    Explicit,
    /// Missing `vn` with active non-zero smoothing group ID.
    Group(u32),
    /// Missing `vn` with `s off` / `s 0` — isolated per face for hard flat normals.
    Flat(u32),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct VertexKey {
    pos: usize,
    uv: Option<usize>,
    norm: Option<usize>,
    material_slot: u32,
    smooth: SmoothKey,
}

fn resolve_obj_index(raw: i64, len: usize, line: usize) -> Result<usize, ObjLoadError> {
    if raw == 0 {
        return Err(ObjLoadError::IndexOutOfBounds { line, index: raw });
    }
    let resolved = if raw > 0 { raw - 1 } else { len as i64 + raw };
    if resolved < 0 || (resolved as usize) >= len {
        return Err(ObjLoadError::IndexOutOfBounds { line, index: raw });
    }
    Ok(resolved as usize)
}

fn parse_f32_token(token: &str, line: usize) -> Result<f32, ObjLoadError> {
    let val: f32 = token.parse().map_err(|_| ObjLoadError::InvalidSyntax {
        line,
        message: format!("invalid float '{token}'"),
    })?;
    if !val.is_finite() {
        return Err(ObjLoadError::InvalidSyntax {
            line,
            message: format!("non-finite float '{token}'"),
        });
    }
    Ok(val)
}

fn parse_i64_token(token: &str, line: usize) -> Result<i64, ObjLoadError> {
    token.parse().map_err(|_| ObjLoadError::InvalidSyntax {
        line,
        message: format!("invalid index '{token}'"),
    })
}

fn collect_logical_lines(source: &str) -> Vec<(usize, String)> {
    let source = source.strip_prefix('\u{FEFF}').unwrap_or(source);
    let mut logical_lines = Vec::new();
    let mut acc = String::new();
    let mut start_line = 0_usize;

    for (zero_line, raw_line) in source.lines().enumerate() {
        let line_no = zero_line + 1;
        let without_comment = raw_line.split('#').next().unwrap_or("");
        let has_comment = raw_line.contains('#');
        let trimmed_end = without_comment.trim_end_matches([' ', '\t']);

        if !has_comment && let Some(continued) = trimmed_end.strip_suffix('\\') {
            if acc.is_empty() {
                start_line = line_no;
            }
            acc.push_str(continued);
            acc.push(' ');
            continue;
        }

        if acc.is_empty() {
            let trimmed = without_comment.trim();
            if !trimmed.is_empty() {
                logical_lines.push((line_no, trimmed.to_owned()));
            }
        } else {
            acc.push_str(without_comment);
            let trimmed = acc.trim();
            if !trimmed.is_empty() {
                logical_lines.push((start_line, trimmed.to_owned()));
            }
            acc.clear();
        }
    }

    if !acc.is_empty() {
        let trimmed = acc.trim();
        if !trimmed.is_empty() {
            logical_lines.push((start_line, trimmed.to_owned()));
        }
    }

    logical_lines
}

fn cross_2d(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn point_in_tri_2d(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let c1 = cross_2d(a, b, p);
    let c2 = cross_2d(b, c, p);
    let c3 = cross_2d(c, a, p);
    c1 >= -1e-6 && c2 >= -1e-6 && c3 >= -1e-6
}

fn ear_clip_2d(poly: &[Vec2]) -> Vec<[usize; 3]> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }
    let mut active: Vec<usize> = (0..n).collect();
    let mut tris = Vec::with_capacity(n - 2);
    let mut guard = n * n;

    while active.len() > 3 && guard > 0 {
        guard -= 1;
        let m = active.len();
        let mut clipped = false;

        for min_cross in [1e-8_f32, -1e-8_f32] {
            for k in 0..m {
                let prev = active[(k + m - 1) % m];
                let curr = active[k];
                let next = active[(k + 1) % m];
                let a = poly[prev];
                let b = poly[curr];
                let c = poly[next];

                if cross_2d(a, b, c) <= min_cross {
                    continue;
                }

                let mut contains_other = false;
                for &idx in &active {
                    if idx == prev || idx == curr || idx == next {
                        continue;
                    }
                    let p = poly[idx];
                    if p.distance_squared(a) <= 1e-10
                        || p.distance_squared(b) <= 1e-10
                        || p.distance_squared(c) <= 1e-10
                    {
                        continue;
                    }
                    if point_in_tri_2d(p, a, b, c) {
                        contains_other = true;
                        break;
                    }
                }
                if !contains_other {
                    tris.push([prev, curr, next]);
                    active.remove(k);
                    clipped = true;
                    break;
                }
            }
            if clipped {
                break;
            }
        }

        if !clipped {
            let root = active[0];
            for pair in active[1..].windows(2) {
                tris.push([root, pair[0], pair[1]]);
            }
            return tris;
        }
    }

    if active.len() == 3 {
        tris.push([active[0], active[1], active[2]]);
    }
    tris
}

fn triangulate_face(face_verts: &[u32], vertices: &[Vertex3D], indices: &mut Vec<u32>) {
    if face_verts.len() == 3 {
        indices.extend_from_slice(&[face_verts[0], face_verts[1], face_verts[2]]);
        return;
    }

    let m = face_verts.len();
    let mut newell = Vec3::ZERO;
    for i in 0..m {
        let a = Vec3::from_array(vertices[face_verts[i] as usize].position);
        let b = Vec3::from_array(vertices[face_verts[(i + 1) % m] as usize].position);
        newell.x += (a.y - b.y) * (a.z + b.z);
        newell.y += (a.z - b.z) * (a.x + b.x);
        newell.z += (a.x - b.x) * (a.y + b.y);
    }

    let face_n = newell.normalize_or_zero();
    if face_n.length_squared() <= 1e-10 {
        let root = face_verts[0];
        for pair in face_verts[1..].windows(2) {
            indices.extend_from_slice(&[root, pair[0], pair[1]]);
        }
        return;
    }

    let ref_axis = if face_n.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent_u = face_n.cross(ref_axis).normalize_or_zero();
    let tangent_v = face_n.cross(tangent_u).normalize_or_zero();
    let origin = Vec3::from_array(vertices[face_verts[0] as usize].position);

    let projected: Vec<Vec2> = face_verts
        .iter()
        .map(|&vi| {
            let d = Vec3::from_array(vertices[vi as usize].position) - origin;
            Vec2::new(d.dot(tangent_u), d.dot(tangent_v))
        })
        .collect();

    for [i0, i1, i2] in ear_clip_2d(&projected) {
        indices.extend_from_slice(&[face_verts[i0], face_verts[i1], face_verts[i2]]);
    }
}

/// Clean-room Wavefront `.obj` parser supporting `v`, `vt`, `vn`, triangles/quads/n-gons,
/// negative relative indices, `usemtl` material slots, and automatic normal generation.
pub fn parse_obj(source: &str) -> Result<MeshData, ObjLoadError> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut texcoords: Vec<[f32; 2]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();

    let mut material_map: HashMap<String, u32> = HashMap::new();
    let mut current_slot: u32 = 0;
    let mut has_seen_usemtl = false;
    let mut has_unassigned_faces = false;

    // Default smoothing group 1 keeps smooth normal averaging when no `s` directive appears.
    // `s off` or `s 0` sets `current_smoothing = 0` (flat per-face normals).
    let mut current_smoothing: u32 = 1;
    let mut named_smoothing: HashMap<String, u32> = HashMap::new();
    let mut next_named_smoothing: u32 = 0x8000_0000;
    let mut face_counter: u32 = 0;

    let mut vertices: Vec<Vertex3D> = Vec::new();
    let mut missing_normal: Vec<bool> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut dedup: HashMap<VertexKey, u32> = HashMap::new();
    let mut missing_any_normal = false;

    for (line_no, line) in collect_logical_lines(source) {
        let mut parts = line.split_whitespace();
        let Some(keyword) = parts.next() else {
            continue;
        };

        match keyword {
            "v" => {
                let x = parse_f32_token(
                    parts.next().ok_or_else(|| ObjLoadError::InvalidSyntax {
                        line: line_no,
                        message: String::from("vertex missing X"),
                    })?,
                    line_no,
                )?;
                let y = parse_f32_token(
                    parts.next().ok_or_else(|| ObjLoadError::InvalidSyntax {
                        line: line_no,
                        message: String::from("vertex missing Y"),
                    })?,
                    line_no,
                )?;
                let z = parse_f32_token(
                    parts.next().ok_or_else(|| ObjLoadError::InvalidSyntax {
                        line: line_no,
                        message: String::from("vertex missing Z"),
                    })?,
                    line_no,
                )?;
                positions.push([x, y, z]);
            }
            "vt" => {
                let u = parse_f32_token(
                    parts.next().ok_or_else(|| ObjLoadError::InvalidSyntax {
                        line: line_no,
                        message: String::from("texcoord missing U"),
                    })?,
                    line_no,
                )?;
                let v = match parts.next() {
                    Some(tok) => parse_f32_token(tok, line_no)?,
                    None => 0.0,
                };
                texcoords.push([u, v]);
            }
            "vn" => {
                let nx = parse_f32_token(
                    parts.next().ok_or_else(|| ObjLoadError::InvalidSyntax {
                        line: line_no,
                        message: String::from("normal missing X"),
                    })?,
                    line_no,
                )?;
                let ny = parse_f32_token(
                    parts.next().ok_or_else(|| ObjLoadError::InvalidSyntax {
                        line: line_no,
                        message: String::from("normal missing Y"),
                    })?,
                    line_no,
                )?;
                let nz = parse_f32_token(
                    parts.next().ok_or_else(|| ObjLoadError::InvalidSyntax {
                        line: line_no,
                        message: String::from("normal missing Z"),
                    })?,
                    line_no,
                )?;
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                let unit = if len > 1e-8 {
                    [nx / len, ny / len, nz / len]
                } else {
                    [0.0, 0.0, 1.0]
                };
                normals.push(unit);
            }
            "usemtl" => {
                let name = line.strip_prefix(keyword).unwrap_or("").trim();
                if !name.is_empty() {
                    has_seen_usemtl = true;
                    let offset = u32::from(has_unassigned_faces);
                    let next_id = offset + material_map.len() as u32;
                    current_slot = *material_map.entry(name.to_owned()).or_insert(next_id);
                }
            }
            "s" => {
                let raw = parts.next().unwrap_or("off");
                if raw.eq_ignore_ascii_case("off") || raw == "0" {
                    current_smoothing = 0;
                } else if let Ok(group_id) = raw.parse::<u32>() {
                    current_smoothing = group_id;
                } else {
                    current_smoothing =
                        *named_smoothing.entry(raw.to_owned()).or_insert_with(|| {
                            let id = next_named_smoothing;
                            next_named_smoothing = next_named_smoothing.saturating_add(1);
                            id
                        });
                }
            }
            "f" => {
                if !has_seen_usemtl {
                    has_unassigned_faces = true;
                }
                let face_id = face_counter;
                face_counter = face_counter.wrapping_add(1);
                let mut face_verts: Vec<u32> = Vec::new();
                for token in parts {
                    let mut comps = token.split('/');
                    let v_str = comps.next().unwrap_or("");
                    let vt_str = comps.next().unwrap_or("");
                    let vn_str = comps.next().unwrap_or("");

                    let pos_idx = resolve_obj_index(
                        parse_i64_token(v_str, line_no)?,
                        positions.len(),
                        line_no,
                    )?;
                    let uv_idx = if !vt_str.is_empty() {
                        Some(resolve_obj_index(
                            parse_i64_token(vt_str, line_no)?,
                            texcoords.len(),
                            line_no,
                        )?)
                    } else {
                        None
                    };
                    let norm_idx = if !vn_str.is_empty() {
                        Some(resolve_obj_index(
                            parse_i64_token(vn_str, line_no)?,
                            normals.len(),
                            line_no,
                        )?)
                    } else {
                        missing_any_normal = true;
                        None
                    };

                    let smooth = if norm_idx.is_some() {
                        SmoothKey::Explicit
                    } else if current_smoothing == 0 {
                        SmoothKey::Flat(face_id)
                    } else {
                        SmoothKey::Group(current_smoothing)
                    };

                    let key = VertexKey {
                        pos: pos_idx,
                        uv: uv_idx,
                        norm: norm_idx,
                        material_slot: current_slot,
                        smooth,
                    };

                    let idx = if let Some(&existing) = dedup.get(&key) {
                        existing
                    } else {
                        let new_idx = vertices.len() as u32;
                        let uv = uv_idx.map_or([0.0, 0.0], |i| texcoords[i]);
                        let norm = norm_idx.map_or([0.0, 0.0, 1.0], |i| normals[i]);
                        vertices.push(Vertex3D::new(positions[pos_idx], norm, uv, current_slot));
                        missing_normal.push(norm_idx.is_none());
                        dedup.insert(key, new_idx);
                        new_idx
                    };
                    face_verts.push(idx);
                }

                if face_verts.len() < 3 {
                    return Err(ObjLoadError::InvalidSyntax {
                        line: line_no,
                        message: String::from("face has fewer than 3 vertices"),
                    });
                }

                triangulate_face(&face_verts, &vertices, &mut indices);
            }
            _ => {
                // Ignore unsupported/metadata statements (o, g, mtllib, vp, l, etc.)
            }
        }
    }

    if vertices.is_empty() || indices.is_empty() {
        return Err(ObjLoadError::EmptyMesh);
    }

    let mut mesh = MeshData::new(vertices, indices);
    if missing_any_normal {
        let mut accum = vec![Vec3::ZERO; mesh.vertices.len()];
        for tri in mesh.indices.as_chunks::<3>().0 {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
                continue;
            }
            if !missing_normal[i0] && !missing_normal[i1] && !missing_normal[i2] {
                continue;
            }
            let p0 = Vec3::from_array(mesh.vertices[i0].position);
            let p1 = Vec3::from_array(mesh.vertices[i1].position);
            let p2 = Vec3::from_array(mesh.vertices[i2].position);
            let weighted_n = (p1 - p0).cross(p2 - p0);
            if missing_normal[i0] {
                accum[i0] += weighted_n;
            }
            if missing_normal[i1] {
                accum[i1] += weighted_n;
            }
            if missing_normal[i2] {
                accum[i2] += weighted_n;
            }
        }
        for ((v, &needs_norm), n) in mesh.vertices.iter_mut().zip(&missing_normal).zip(accum) {
            if needs_norm {
                let unit = if n.length_squared() > 1e-12 {
                    n.normalize()
                } else {
                    Vec3::Z
                };
                v.normal = unit.to_array();
            }
        }
    }
    Ok(mesh)
}
