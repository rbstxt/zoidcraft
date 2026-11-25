use super::*;

/// Packs live uniform values into a `std140`-aligned byte buffer matching `layout`.
///
/// Missing uniforms remain zero-initialized. Numeric values coerce cleanly between `Float`,
/// `Int`, `Uint`, and `Bool` to match ClipMaker 3 / Shader+ dynamic property storage.
/// Keys containing whitespace are matched via [`fix_property_name`] when no exact key matches.
pub fn pack_uniforms(
    layout: &UniformBlockLayout,
    values: &BTreeMap<String, UniformValue>,
) -> Vec<u8> {
    let mut buf = vec![0u8; layout.byte_size as usize];
    let mut normalized_lookup: Option<Option<HashMap<String, &UniformValue>>> = None;

    for field in &layout.fields {
        let value = values.get(&field.name).or_else(|| {
            let lookup = normalized_lookup.get_or_insert_with(|| {
                let mut map = HashMap::new();
                for (k, v) in values {
                    if k.chars().any(char::is_whitespace) {
                        map.entry(fix_property_name(k)).or_insert(v);
                    }
                }
                if map.is_empty() { None } else { Some(map) }
            });
            lookup
                .as_ref()
                .and_then(|map| map.get(&field.name).copied())
        });
        let Some(value) = value else {
            continue;
        };

        if let Some(array_len) = field.array_len {
            if let UniformValue::Array(items) = value {
                let count = items.len().min(array_len as usize);
                for (index, item) in items.iter().take(count).enumerate() {
                    let elem_offset = field.offset as usize + index * field.array_stride as usize;
                    write_uniform_element(&mut buf, elem_offset, field.ty, item);
                }
            } else if array_len > 0 {
                write_uniform_element(&mut buf, field.offset as usize, field.ty, value);
            }
        } else {
            write_uniform_element(&mut buf, field.offset as usize, field.ty, value);
        }
    }

    buf
}

fn write_uniform_element(buf: &mut [u8], offset: usize, ty: UniformType, value: &UniformValue) {
    match ty {
        UniformType::Float => write_f32(buf, offset, coerce_f32(value)),
        UniformType::Int => write_i32(buf, offset, coerce_i32(value)),
        UniformType::Uint => write_u32(buf, offset, coerce_u32(value)),
        UniformType::Bool => write_u32(buf, offset, u32::from(coerce_bool(value))),
        UniformType::Vec2 => {
            let v = match *value {
                UniformValue::Vec2(v) => v,
                UniformValue::Vec3(v) => [v[0], v[1]],
                UniformValue::Vec4(v) => [v[0], v[1]],
                UniformValue::Float(s) => [s, s],
                _ => [0.0; 2],
            };
            write_f32_slice(buf, offset, &v);
        }
        UniformType::Vec3 => {
            let v = match *value {
                UniformValue::Vec3(v) => v,
                UniformValue::Vec4(v) => [v[0], v[1], v[2]],
                UniformValue::Float(s) => [s, s, s],
                _ => [0.0; 3],
            };
            write_f32_slice(buf, offset, &v);
        }
        UniformType::Vec4 => {
            let v = match *value {
                UniformValue::Vec4(v) => v,
                UniformValue::Vec3(v) => [v[0], v[1], v[2], 1.0],
                UniformValue::Float(s) => [s, s, s, s],
                _ => [0.0; 4],
            };
            write_f32_slice(buf, offset, &v);
        }
        UniformType::Ivec2 => {
            let v = match *value {
                UniformValue::Ivec2(v) => v,
                UniformValue::Vec2(v) => [v[0].round() as i32, v[1].round() as i32],
                _ => [0; 2],
            };
            write_i32_slice(buf, offset, &v);
        }
        UniformType::Ivec3 => {
            let v = match *value {
                UniformValue::Ivec3(v) => v,
                UniformValue::Vec3(v) => [
                    v[0].round() as i32,
                    v[1].round() as i32,
                    v[2].round() as i32,
                ],
                _ => [0; 3],
            };
            write_i32_slice(buf, offset, &v);
        }
        UniformType::Ivec4 => {
            let v = match *value {
                UniformValue::Ivec4(v) => v,
                UniformValue::Vec4(v) => [
                    v[0].round() as i32,
                    v[1].round() as i32,
                    v[2].round() as i32,
                    v[3].round() as i32,
                ],
                _ => [0; 4],
            };
            write_i32_slice(buf, offset, &v);
        }
        UniformType::Uvec2 => {
            let v = match *value {
                UniformValue::Uvec2(v) => v,
                UniformValue::Ivec2(v) => [v[0].max(0) as u32, v[1].max(0) as u32],
                _ => [0; 2],
            };
            write_u32_slice(buf, offset, &v);
        }
        UniformType::Uvec3 => {
            let v = match *value {
                UniformValue::Uvec3(v) => v,
                UniformValue::Ivec3(v) => {
                    [v[0].max(0) as u32, v[1].max(0) as u32, v[2].max(0) as u32]
                }
                _ => [0; 3],
            };
            write_u32_slice(buf, offset, &v);
        }
        UniformType::Uvec4 => {
            let v = match *value {
                UniformValue::Uvec4(v) => v,
                UniformValue::Ivec4(v) => [
                    v[0].max(0) as u32,
                    v[1].max(0) as u32,
                    v[2].max(0) as u32,
                    v[3].max(0) as u32,
                ],
                _ => [0; 4],
            };
            write_u32_slice(buf, offset, &v);
        }
        UniformType::Bvec2 => {
            let v = match *value {
                UniformValue::Bvec2(v) => [u32::from(v[0]), u32::from(v[1])],
                UniformValue::Uvec2(v) => [u32::from(v[0] != 0), u32::from(v[1] != 0)],
                UniformValue::Ivec2(v) => [u32::from(v[0] != 0), u32::from(v[1] != 0)],
                UniformValue::Vec2(v) => [u32::from(v[0] != 0.0), u32::from(v[1] != 0.0)],
                UniformValue::Bool(b) => [u32::from(b); 2],
                _ => [0; 2],
            };
            write_u32_slice(buf, offset, &v);
        }
        UniformType::Bvec3 => {
            let v = match *value {
                UniformValue::Bvec3(v) => [u32::from(v[0]), u32::from(v[1]), u32::from(v[2])],
                UniformValue::Uvec3(v) => [
                    u32::from(v[0] != 0),
                    u32::from(v[1] != 0),
                    u32::from(v[2] != 0),
                ],
                UniformValue::Ivec3(v) => [
                    u32::from(v[0] != 0),
                    u32::from(v[1] != 0),
                    u32::from(v[2] != 0),
                ],
                UniformValue::Vec3(v) => [
                    u32::from(v[0] != 0.0),
                    u32::from(v[1] != 0.0),
                    u32::from(v[2] != 0.0),
                ],
                UniformValue::Bool(b) => [u32::from(b); 3],
                _ => [0; 3],
            };
            write_u32_slice(buf, offset, &v);
        }
        UniformType::Bvec4 => {
            let v = match *value {
                UniformValue::Bvec4(v) => [
                    u32::from(v[0]),
                    u32::from(v[1]),
                    u32::from(v[2]),
                    u32::from(v[3]),
                ],
                UniformValue::Uvec4(v) => [
                    u32::from(v[0] != 0),
                    u32::from(v[1] != 0),
                    u32::from(v[2] != 0),
                    u32::from(v[3] != 0),
                ],
                UniformValue::Ivec4(v) => [
                    u32::from(v[0] != 0),
                    u32::from(v[1] != 0),
                    u32::from(v[2] != 0),
                    u32::from(v[3] != 0),
                ],
                UniformValue::Vec4(v) => [
                    u32::from(v[0] != 0.0),
                    u32::from(v[1] != 0.0),
                    u32::from(v[2] != 0.0),
                    u32::from(v[3] != 0.0),
                ],
                UniformValue::Bool(b) => [u32::from(b); 4],
                _ => [0; 4],
            };
            write_u32_slice(buf, offset, &v);
        }
        UniformType::Mat2 => {
            if let UniformValue::Mat2(m) = *value {
                write_f32_slice(buf, offset, &m[0..2]);
                write_f32_slice(buf, offset + 16, &m[2..4]);
            }
        }
        UniformType::Mat3 => {
            if let UniformValue::Mat3(m) = *value {
                write_f32_slice(buf, offset, &m[0..3]);
                write_f32_slice(buf, offset + 16, &m[3..6]);
                write_f32_slice(buf, offset + 32, &m[6..9]);
            }
        }
        UniformType::Mat4 => {
            if let UniformValue::Mat4(m) = *value {
                write_f32_slice(buf, offset, &m);
            }
        }
    }
}

fn coerce_f32(value: &UniformValue) -> f32 {
    match *value {
        UniformValue::Float(v) => v,
        UniformValue::Int(v) => v as f32,
        UniformValue::Uint(v) => v as f32,
        UniformValue::Bool(true) => 1.0,
        UniformValue::Bool(false) => 0.0,
        _ => 0.0,
    }
}

fn coerce_i32(value: &UniformValue) -> i32 {
    match *value {
        UniformValue::Int(v) => v,
        UniformValue::Float(v) => v.round() as i32,
        UniformValue::Uint(v) => v as i32,
        UniformValue::Bool(b) => i32::from(b),
        _ => 0,
    }
}

fn coerce_u32(value: &UniformValue) -> u32 {
    match *value {
        UniformValue::Uint(v) => v,
        UniformValue::Int(v) => v.max(0) as u32,
        UniformValue::Float(v) => v.max(0.0).round() as u32,
        UniformValue::Bool(b) => u32::from(b),
        _ => 0,
    }
}

fn coerce_bool(value: &UniformValue) -> bool {
    match *value {
        UniformValue::Bool(b) => b,
        UniformValue::Float(v) => v != 0.0,
        UniformValue::Int(v) => v != 0,
        UniformValue::Uint(v) => v != 0,
        _ => false,
    }
}

fn write_f32(buf: &mut [u8], offset: usize, value: f32) {
    if offset + 4 <= buf.len() {
        buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn write_i32(buf: &mut [u8], offset: usize, value: i32) {
    if offset + 4 <= buf.len() {
        buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn write_u32(buf: &mut [u8], offset: usize, value: u32) {
    if offset + 4 <= buf.len() {
        buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn write_f32_slice(buf: &mut [u8], offset: usize, values: &[f32]) {
    for (idx, &v) in values.iter().enumerate() {
        write_f32(buf, offset + idx * 4, v);
    }
}

fn write_i32_slice(buf: &mut [u8], offset: usize, values: &[i32]) {
    for (idx, &v) in values.iter().enumerate() {
        write_i32(buf, offset + idx * 4, v);
    }
}

fn write_u32_slice(buf: &mut [u8], offset: usize, values: &[u32]) {
    for (idx, &v) in values.iter().enumerate() {
        write_u32(buf, offset + idx * 4, v);
    }
}

// ---------------------------------------------------------------------------
// Stage 1: GLSL ES 1.00 Preprocessor (comments, conditionals, macros, #include, #extension)
