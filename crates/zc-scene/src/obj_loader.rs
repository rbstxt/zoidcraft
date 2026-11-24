use std::collections::HashMap;

use thiserror::Error;

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
struct VertexKey {
    pos: usize,
    uv: Option<usize>,
    norm: Option<usize>,
    material_slot: u32,
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

/// Clean-room Wavefront `.obj` parser supporting `v`, `vt`, `vn`, triangles/quads/n-gons,
/// negative relative indices, `usemtl` material slots, and automatic normal generation.
pub fn parse_obj(source: &str) -> Result<MeshData, ObjLoadError> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut texcoords: Vec<[f32; 2]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();

    let mut material_map: HashMap<String, u32> = HashMap::new();
    let mut current_slot: u32 = 0;

    let mut vertices: Vec<Vertex3D> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut dedup: HashMap<VertexKey, u32> = HashMap::new();
    let mut missing_any_normal = false;

    for (zero_line, raw_line) in source.lines().enumerate() {
        let line_no = zero_line + 1;
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

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
                if let Some(name) = parts.next() {
                    let next_id = material_map.len() as u32;
                    current_slot = *material_map.entry(name.to_owned()).or_insert(next_id);
                }
            }
            "f" => {
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

                    let key = VertexKey {
                        pos: pos_idx,
                        uv: uv_idx,
                        norm: norm_idx,
                        material_slot: current_slot,
                    };

                    let idx = if let Some(&existing) = dedup.get(&key) {
                        existing
                    } else {
                        let new_idx = vertices.len() as u32;
                        let uv = uv_idx.map_or([0.0, 0.0], |i| texcoords[i]);
                        let norm = norm_idx.map_or([0.0, 0.0, 1.0], |i| normals[i]);
                        vertices.push(Vertex3D::new(positions[pos_idx], norm, uv, current_slot));
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

                // Triangle fan for triangles, quads, and convex n-gons
                let root = face_verts[0];
                for pair in face_verts[1..].windows(2) {
                    indices.push(root);
                    indices.push(pair[0]);
                    indices.push(pair[1]);
                }
            }
            _ => {
                // Ignore unsupported/metadata statements (o, g, s, mtllib, vp, l, etc.)
            }
        }
    }

    if vertices.is_empty() || indices.is_empty() {
        return Err(ObjLoadError::EmptyMesh);
    }

    let mut mesh = MeshData::new(vertices, indices);
    if missing_any_normal {
        mesh.compute_smooth_normals();
    }
    Ok(mesh)
}
