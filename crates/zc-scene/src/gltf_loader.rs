//! Self-contained, pure safe Rust GLB / glTF 2.0 mesh loader (zero external JSON/base64 crates).
//!
//! Supports:
//! - Binary `.glb` 2.0 containers (JSON + BIN chunks) and text `.gltf` 2.0 JSON + `data:` base64
//!   buffer URIs (plus optional external buffer slices).
//! - `POSITION` (`VEC3` `f32`), `NORMAL` (`VEC3` `f32`, auto-generated if absent),
//!   `TEXCOORD_0` (`VEC2` `f32` / normalized `u8` / `u16`), `INDICES` (`u8` / `u16` / `u32`).
//! - Primitive topologies: `TRIANGLES` (`4`), `TRIANGLE_STRIP` (`5`), `TRIANGLE_FAN` (`6`).
//! - Node hierarchy traversal (`matrix` and `translation` / `rotation` / `scale` TRS composition).
//! - Multi-primitive `material_slot` preservation.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use thiserror::Error;
use zc_geom::{Mat4, Quat, Vec3};

use crate::mesh::{MeshData, Vertex3D};

const GLB_MAGIC: u32 = 0x4654_6C67; // "glTF"
const GLB_VERSION_2: u32 = 2;
const CHUNK_TYPE_JSON: u32 = 0x4E4F_534A; // "JSON"
const CHUNK_TYPE_BIN: u32 = 0x004E_4942; // "BIN\0"

const COMP_UNSIGNED_BYTE: u32 = 5121;
const COMP_UNSIGNED_SHORT: u32 = 5123;
const COMP_UNSIGNED_INT: u32 = 5125;
const COMP_FLOAT: u32 = 5126;

const MODE_TRIANGLES: u32 = 4;
const MODE_TRIANGLE_STRIP: u32 = 5;
const MODE_TRIANGLE_FAN: u32 = 6;

const MAX_JSON_DEPTH: usize = 128;
const MAX_NODE_DEPTH: usize = 128;
const MAX_SCENE_NODE_VISITS: usize = 4_096;
const MAX_TOTAL_VERTICES: usize = 2_000_000;
const MAX_TOTAL_INDICES: usize = 6_000_000;

/// Errors returned while parsing a `.glb` or `.gltf` 2.0 mesh.
#[derive(Debug, Error)]
pub enum GltfLoadError {
    #[error("invalid GLB header: {0}")]
    InvalidHeader(String),
    #[error("invalid glTF JSON: {0}")]
    InvalidJson(String),
    #[error("invalid glTF data URI: {0}")]
    InvalidDataUri(String),
    #[error("invalid glTF buffer or accessor: {0}")]
    InvalidBuffer(String),
    #[error("unsupported glTF feature: {0}")]
    UnsupportedFeature(String),
    #[error("glTF contains no valid geometry")]
    EmptyMesh,
}

/* ----------------------------------------------------- Minimal JSON parser */

#[derive(Debug, Clone)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(HashMap<String, JsonValue>),
}

impl JsonValue {
    fn as_object(&self) -> Option<&HashMap<String, JsonValue>> {
        match self {
            Self::Object(map) => Some(map),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(arr) => Some(arr),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) if n.is_finite() => Some(*n),
            _ => None,
        }
    }

    fn as_usize(&self) -> Option<usize> {
        let n = self.as_f64()?;
        if n < 0.0 || n.fract() != 0.0 || n > usize::MAX as f64 {
            return None;
        }
        Some(n as usize)
    }

    fn as_u32(&self) -> Option<u32> {
        let n = self.as_f64()?;
        if n < 0.0 || n.fract() != 0.0 || n > u32::MAX as f64 {
            return None;
        }
        Some(n as u32)
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()?.get(key)
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        let trimmed = input.strip_prefix('\u{FEFF}').unwrap_or(input);
        Self {
            bytes: trimmed.as_bytes(),
            pos: 0,
            depth: 0,
        }
    }

    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_root(&mut self) -> Result<JsonValue, GltfLoadError> {
        self.skip_ws();
        let val = self.parse_value()?;
        self.skip_ws();
        if self.pos < self.bytes.len() {
            return Err(GltfLoadError::InvalidJson(format!(
                "trailing bytes at offset {}",
                self.pos
            )));
        }
        Ok(val)
    }

    fn parse_value(&mut self) -> Result<JsonValue, GltfLoadError> {
        self.skip_ws();
        let Some(b) = self.peek() else {
            return Err(GltfLoadError::InvalidJson(String::from("unexpected EOF")));
        };
        match b {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => self.parse_string().map(JsonValue::String),
            b't' => self.expect_literal(b"true", JsonValue::Bool(true)),
            b'f' => self.expect_literal(b"false", JsonValue::Bool(false)),
            b'n' => self.expect_literal(b"null", JsonValue::Null),
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => Err(GltfLoadError::InvalidJson(format!(
                "unexpected byte 0x{other:02X} at offset {}",
                self.pos
            ))),
        }
    }

    fn expect_literal(&mut self, lit: &[u8], val: JsonValue) -> Result<JsonValue, GltfLoadError> {
        if self.bytes.get(self.pos..self.pos + lit.len()) == Some(lit) {
            self.pos += lit.len();
            Ok(val)
        } else {
            Err(GltfLoadError::InvalidJson(format!(
                "invalid literal at offset {}",
                self.pos
            )))
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, GltfLoadError> {
        if self.depth >= MAX_JSON_DEPTH {
            return Err(GltfLoadError::InvalidJson(String::from(
                "nesting depth exceeded",
            )));
        }
        self.depth += 1;
        self.pos += 1; // '{'
        let mut map = HashMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(GltfLoadError::InvalidJson(format!(
                    "expected object key at offset {}",
                    self.pos
                )));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(GltfLoadError::InvalidJson(format!(
                    "expected ':' at offset {}",
                    self.pos
                )));
            }
            self.pos += 1;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(JsonValue::Object(map));
                }
                _ => {
                    return Err(GltfLoadError::InvalidJson(format!(
                        "expected ',' or '}}' at offset {}",
                        self.pos
                    )));
                }
            }
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue, GltfLoadError> {
        if self.depth >= MAX_JSON_DEPTH {
            return Err(GltfLoadError::InvalidJson(String::from(
                "nesting depth exceeded",
            )));
        }
        self.depth += 1;
        self.pos += 1; // '['
        let mut arr = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(JsonValue::Array(arr));
        }
        loop {
            let val = self.parse_value()?;
            arr.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(JsonValue::Array(arr));
                }
                _ => {
                    return Err(GltfLoadError::InvalidJson(format!(
                        "expected ',' or ']' at offset {}",
                        self.pos
                    )));
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, GltfLoadError> {
        let Some(slice) = self.bytes.get(self.pos..self.pos + 4) else {
            return Err(GltfLoadError::InvalidJson(String::from(
                "truncated \\u escape",
            )));
        };
        let mut val = 0_u16;
        for &b in slice {
            let digit = match b {
                b'0'..=b'9' => u16::from(b - b'0'),
                b'a'..=b'f' => u16::from(b - b'a' + 10),
                b'A'..=b'F' => u16::from(b - b'A' + 10),
                _ => {
                    return Err(GltfLoadError::InvalidJson(String::from(
                        "invalid hex digit in \\u escape",
                    )));
                }
            };
            val = (val << 4) | digit;
        }
        self.pos += 4;
        Ok(val)
    }

    fn parse_string(&mut self) -> Result<String, GltfLoadError> {
        self.pos += 1; // opening '"'
        let mut out = String::new();
        while let Some(&b) = self.bytes.get(self.pos) {
            match b {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let Some(&esc) = self.bytes.get(self.pos) else {
                        return Err(GltfLoadError::InvalidJson(String::from(
                            "unterminated escape",
                        )));
                    };
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let code = self.parse_hex4()?;
                            if (0xD800..=0xDBFF).contains(&code) {
                                if self.bytes.get(self.pos..self.pos + 2) != Some(b"\\u") {
                                    return Err(GltfLoadError::InvalidJson(String::from(
                                        "missing low surrogate",
                                    )));
                                }
                                self.pos += 2;
                                let low = self.parse_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err(GltfLoadError::InvalidJson(String::from(
                                        "invalid low surrogate",
                                    )));
                                }
                                let cp = 0x10000
                                    + ((u32::from(code) - 0xD800) << 10)
                                    + (u32::from(low) - 0xDC00);
                                let ch = char::from_u32(cp).ok_or_else(|| {
                                    GltfLoadError::InvalidJson(String::from(
                                        "invalid unicode scalar",
                                    ))
                                })?;
                                out.push(ch);
                            } else {
                                let ch = char::from_u32(u32::from(code)).ok_or_else(|| {
                                    GltfLoadError::InvalidJson(String::from(
                                        "invalid unicode scalar",
                                    ))
                                })?;
                                out.push(ch);
                            }
                        }
                        _ => {
                            return Err(GltfLoadError::InvalidJson(format!(
                                "invalid escape byte 0x{esc:02X}"
                            )));
                        }
                    }
                }
                0x00..=0x1F => {
                    return Err(GltfLoadError::InvalidJson(String::from(
                        "unescaped control character in string",
                    )));
                }
                _ => {
                    let start = self.pos;
                    while let Some(&c) = self.bytes.get(self.pos) {
                        if c == b'"' || c == b'\\' || c < 0x20 {
                            break;
                        }
                        self.pos += 1;
                    }
                    let chunk =
                        std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| {
                            GltfLoadError::InvalidJson(String::from("invalid UTF-8 in string"))
                        })?;
                    out.push_str(chunk);
                }
            }
        }
        Err(GltfLoadError::InvalidJson(String::from(
            "unterminated string",
        )))
    }

    fn parse_number(&mut self) -> Result<JsonValue, GltfLoadError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        let mut has_digits = false;
        while let Some(b'0'..=b'9') = self.peek() {
            has_digits = true;
            self.pos += 1;
        }
        if !has_digits {
            return Err(GltfLoadError::InvalidJson(format!(
                "invalid number at offset {start}"
            )));
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            let mut frac_digits = false;
            while let Some(b'0'..=b'9') = self.peek() {
                frac_digits = true;
                self.pos += 1;
            }
            if !frac_digits {
                return Err(GltfLoadError::InvalidJson(format!(
                    "invalid fractional number at offset {start}"
                )));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            let mut exp_digits = false;
            while let Some(b'0'..=b'9') = self.peek() {
                exp_digits = true;
                self.pos += 1;
            }
            if !exp_digits {
                return Err(GltfLoadError::InvalidJson(format!(
                    "invalid exponent at offset {start}"
                )));
            }
        }
        let slice = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| GltfLoadError::InvalidJson(String::from("invalid UTF-8 in number")))?;
        let val: f64 = slice
            .parse()
            .map_err(|_| GltfLoadError::InvalidJson(format!("failed to parse number '{slice}'")))?;
        if !val.is_finite() {
            return Err(GltfLoadError::InvalidJson(format!(
                "non-finite number '{slice}'"
            )));
        }
        Ok(JsonValue::Number(val))
    }
}

/* ----------------------------------------------------- Base64 URI decoder */

fn decode_base64(input: &str) -> Result<Vec<u8>, GltfLoadError> {
    let mut out = Vec::with_capacity((input.len() / 4) * 3);
    let mut buf = [0_u8; 4];
    let mut count = 0_usize;
    let mut pad = 0_usize;

    for b in input.bytes() {
        if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
            continue;
        }
        if b == b'=' {
            buf[count] = 0;
            count += 1;
            pad += 1;
        } else {
            if pad > 0 {
                return Err(GltfLoadError::InvalidDataUri(String::from(
                    "unexpected byte after base64 padding",
                )));
            }
            let val = match b {
                b'A'..=b'Z' => b - b'A',
                b'a'..=b'z' => b - b'a' + 26,
                b'0'..=b'9' => b - b'0' + 52,
                b'+' | b'-' => 62,
                b'/' | b'_' => 63,
                _ => {
                    return Err(GltfLoadError::InvalidDataUri(format!(
                        "invalid base64 byte 0x{b:02X}"
                    )));
                }
            };
            buf[count] = val;
            count += 1;
        }
        if count == 4 {
            let n = (u32::from(buf[0]) << 18)
                | (u32::from(buf[1]) << 12)
                | (u32::from(buf[2]) << 6)
                | u32::from(buf[3]);
            out.push(((n >> 16) & 0xFF) as u8);
            if pad < 2 {
                out.push(((n >> 8) & 0xFF) as u8);
            }
            if pad == 0 {
                out.push((n & 0xFF) as u8);
            }
            count = 0;
            pad = 0;
        }
    }
    if count != 0 {
        return Err(GltfLoadError::InvalidDataUri(String::from(
            "unaligned base64 length",
        )));
    }
    Ok(out)
}

fn decode_data_uri(uri: &str) -> Result<Vec<u8>, GltfLoadError> {
    let rest = uri.strip_prefix("data:").ok_or_else(|| {
        GltfLoadError::InvalidDataUri(String::from("URI does not start with data:"))
    })?;
    let (meta, payload) = rest.split_once(',').ok_or_else(|| {
        GltfLoadError::InvalidDataUri(String::from("missing comma separator in data URI"))
    })?;
    if !meta.to_ascii_lowercase().contains(";base64") {
        return Err(GltfLoadError::UnsupportedFeature(String::from(
            "only base64 data URIs are supported",
        )));
    }
    decode_base64(payload)
}

/* ----------------------------------------------------- Buffer / Accessors */

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32, GltfLoadError> {
    let slice = bytes.get(offset..offset + 4).ok_or_else(|| {
        GltfLoadError::InvalidBuffer(format!("read u32 out of bounds at {offset}"))
    })?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u16_le(bytes: &[u8], offset: usize) -> Result<u16, GltfLoadError> {
    let slice = bytes.get(offset..offset + 2).ok_or_else(|| {
        GltfLoadError::InvalidBuffer(format!("read u16 out of bounds at {offset}"))
    })?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_f32_le(bytes: &[u8], offset: usize) -> Result<f32, GltfLoadError> {
    let slice = bytes.get(offset..offset + 4).ok_or_else(|| {
        GltfLoadError::InvalidBuffer(format!("read f32 out of bounds at {offset}"))
    })?;
    let v = f32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]);
    if !v.is_finite() {
        return Err(GltfLoadError::InvalidBuffer(format!(
            "non-finite f32 at offset {offset}"
        )));
    }
    Ok(v)
}

struct BufferViewInfo {
    buffer: usize,
    byte_offset: usize,
    byte_length: usize,
    byte_stride: Option<usize>,
}

struct AccessorInfo {
    buffer_view: usize,
    byte_offset: usize,
    component_type: u32,
    count: usize,
    accessor_type: String,
    normalized: bool,
}

fn component_size(comp_type: u32) -> Result<usize, GltfLoadError> {
    match comp_type {
        5120 | COMP_UNSIGNED_BYTE => Ok(1),
        5122 | COMP_UNSIGNED_SHORT => Ok(2),
        COMP_UNSIGNED_INT | COMP_FLOAT => Ok(4),
        other => Err(GltfLoadError::InvalidBuffer(format!(
            "unknown componentType {other}"
        ))),
    }
}

fn type_components(accessor_type: &str) -> Result<usize, GltfLoadError> {
    match accessor_type {
        "SCALAR" => Ok(1),
        "VEC2" => Ok(2),
        "VEC3" => Ok(3),
        "VEC4" => Ok(4),
        "MAT2" => Ok(4),
        "MAT3" => Ok(9),
        "MAT4" => Ok(16),
        other => Err(GltfLoadError::InvalidBuffer(format!(
            "unknown accessor type '{other}'"
        ))),
    }
}

struct AccessorSlice<'a> {
    bytes: &'a [u8],
    base: usize,
    stride: usize,
    count: usize,
    component_type: u32,
    normalized: bool,
}

struct GltfDocument<'a> {
    root: JsonValue,
    buffers: Vec<Cow<'a, [u8]>>,
    external_buffers: &'a [&'a [u8]],
    buffer_views: Vec<BufferViewInfo>,
    accessors: Vec<AccessorInfo>,
}

impl<'a> GltfDocument<'a> {
    fn buffer_bytes(&self, index: usize) -> Result<&[u8], GltfLoadError> {
        if let Some(buf) = self.buffers.get(index)
            && !buf.is_empty()
        {
            return Ok(buf.as_ref());
        }
        if let Some(ext) = self.external_buffers.get(index) {
            return Ok(ext);
        }
        Err(GltfLoadError::InvalidBuffer(format!(
            "missing buffer {index}"
        )))
    }

    fn accessor_slice_params(
        &self,
        accessor_idx: usize,
        expected_type: &str,
        max_count: usize,
    ) -> Result<AccessorSlice<'_>, GltfLoadError> {
        let acc = self.accessors.get(accessor_idx).ok_or_else(|| {
            GltfLoadError::InvalidBuffer(format!("accessor {accessor_idx} out of bounds"))
        })?;
        if acc.count > max_count {
            return Err(GltfLoadError::InvalidJson(format!(
                "accessor {accessor_idx} count {} exceeds maximum budget of {max_count}",
                acc.count
            )));
        }
        if acc.accessor_type != expected_type {
            return Err(GltfLoadError::InvalidBuffer(format!(
                "accessor {accessor_idx}: expected {expected_type}, got {}",
                acc.accessor_type
            )));
        }
        let bv = self.buffer_views.get(acc.buffer_view).ok_or_else(|| {
            GltfLoadError::InvalidBuffer(format!("bufferView {} out of bounds", acc.buffer_view))
        })?;
        let raw = self.buffer_bytes(bv.buffer)?;
        let elem_size = component_size(acc.component_type)? * type_components(&acc.accessor_type)?;
        let stride = bv.byte_stride.unwrap_or(elem_size);
        if stride < elem_size {
            return Err(GltfLoadError::InvalidBuffer(format!(
                "bufferView {} byteStride {stride} < element size {elem_size}",
                acc.buffer_view
            )));
        }
        let bv_end = bv
            .byte_offset
            .checked_add(bv.byte_length)
            .ok_or_else(|| GltfLoadError::InvalidBuffer(String::from("bufferView overflow")))?;
        if bv_end > raw.len() {
            return Err(GltfLoadError::InvalidBuffer(format!(
                "bufferView {} exceeds buffer length",
                acc.buffer_view
            )));
        }
        let view_slice = &raw[bv.byte_offset..bv_end];
        if acc.count > 0 {
            let last_end = acc
                .byte_offset
                .checked_add(
                    (acc.count - 1)
                        .checked_mul(stride)
                        .and_then(|v| v.checked_add(elem_size))
                        .ok_or_else(|| {
                            GltfLoadError::InvalidBuffer(String::from("accessor span overflow"))
                        })?,
                )
                .ok_or_else(|| GltfLoadError::InvalidBuffer(String::from("accessor overflow")))?;
            if last_end > view_slice.len() {
                return Err(GltfLoadError::InvalidBuffer(format!(
                    "accessor {accessor_idx} exceeds bufferView bounds"
                )));
            }
        }
        Ok(AccessorSlice {
            bytes: view_slice,
            base: acc.byte_offset,
            stride,
            count: acc.count,
            component_type: acc.component_type,
            normalized: acc.normalized,
        })
    }

    fn read_vec3_f32(&self, accessor_idx: usize) -> Result<Vec<[f32; 3]>, GltfLoadError> {
        let s = self.accessor_slice_params(accessor_idx, "VEC3", MAX_TOTAL_VERTICES)?;
        if s.component_type != COMP_FLOAT {
            return Err(GltfLoadError::UnsupportedFeature(format!(
                "VEC3 accessor {accessor_idx} requires f32 (5126), got {}",
                s.component_type
            )));
        }
        let mut out = Vec::with_capacity(s.count);
        for i in 0..s.count {
            let off = s.base + i * s.stride;
            out.push([
                read_f32_le(s.bytes, off)?,
                read_f32_le(s.bytes, off + 4)?,
                read_f32_le(s.bytes, off + 8)?,
            ]);
        }
        Ok(out)
    }

    fn read_vec2_f32(&self, accessor_idx: usize) -> Result<Vec<[f32; 2]>, GltfLoadError> {
        let s = self.accessor_slice_params(accessor_idx, "VEC2", MAX_TOTAL_VERTICES)?;
        let mut out = Vec::with_capacity(s.count);
        for i in 0..s.count {
            let off = s.base + i * s.stride;
            let uv = match s.component_type {
                COMP_FLOAT => [read_f32_le(s.bytes, off)?, read_f32_le(s.bytes, off + 4)?],
                COMP_UNSIGNED_BYTE if s.normalized => [
                    f32::from(s.bytes[off]) / 255.0,
                    f32::from(s.bytes[off + 1]) / 255.0,
                ],
                COMP_UNSIGNED_SHORT if s.normalized => [
                    f32::from(read_u16_le(s.bytes, off)?) / 65535.0,
                    f32::from(read_u16_le(s.bytes, off + 2)?) / 65535.0,
                ],
                other => {
                    return Err(GltfLoadError::UnsupportedFeature(format!(
                        "unsupported TEXCOORD_0 componentType {other} (normalized={})",
                        s.normalized
                    )));
                }
            };
            out.push(uv);
        }
        Ok(out)
    }

    fn read_color0(&self, accessor_idx: usize) -> Result<Vec<[f32; 4]>, GltfLoadError> {
        let acc = self.accessors.get(accessor_idx).ok_or_else(|| {
            GltfLoadError::InvalidBuffer(format!("accessor {accessor_idx} out of bounds"))
        })?;
        let is_vec4 = match acc.accessor_type.as_str() {
            "VEC3" => false,
            "VEC4" => true,
            other => {
                return Err(GltfLoadError::UnsupportedFeature(format!(
                    "unsupported COLOR_0 accessor type '{other}' (expected VEC3 or VEC4)"
                )));
            }
        };
        let expected_type = if is_vec4 { "VEC4" } else { "VEC3" };
        let s = self.accessor_slice_params(accessor_idx, expected_type, MAX_TOTAL_VERTICES)?;
        let mut out = Vec::with_capacity(s.count);
        for i in 0..s.count {
            let off = s.base + i * s.stride;
            let rgba = match s.component_type {
                COMP_FLOAT => [
                    read_f32_le(s.bytes, off)?,
                    read_f32_le(s.bytes, off + 4)?,
                    read_f32_le(s.bytes, off + 8)?,
                    if is_vec4 {
                        read_f32_le(s.bytes, off + 12)?
                    } else {
                        1.0
                    },
                ],
                COMP_UNSIGNED_BYTE => [
                    f32::from(s.bytes[off]) / 255.0,
                    f32::from(s.bytes[off + 1]) / 255.0,
                    f32::from(s.bytes[off + 2]) / 255.0,
                    if is_vec4 {
                        f32::from(s.bytes[off + 3]) / 255.0
                    } else {
                        1.0
                    },
                ],
                COMP_UNSIGNED_SHORT => [
                    f32::from(read_u16_le(s.bytes, off)?) / 65535.0,
                    f32::from(read_u16_le(s.bytes, off + 2)?) / 65535.0,
                    f32::from(read_u16_le(s.bytes, off + 4)?) / 65535.0,
                    if is_vec4 {
                        f32::from(read_u16_le(s.bytes, off + 6)?) / 65535.0
                    } else {
                        1.0
                    },
                ],
                other => {
                    return Err(GltfLoadError::UnsupportedFeature(format!(
                        "unsupported COLOR_0 componentType {other}"
                    )));
                }
            };
            out.push(rgba);
        }
        Ok(out)
    }

    fn material_base_color_factor(&self, material_idx: usize) -> Option<[f32; 4]> {
        let mat = self
            .root
            .get("materials")
            .and_then(JsonValue::as_array)?
            .get(material_idx)?;
        let factor_arr = mat
            .get("pbrMetallicRoughness")
            .and_then(|pbr| pbr.get("baseColorFactor"))
            .or_else(|| {
                mat.get("extensions")
                    .and_then(|ext| ext.get("KHR_materials_pbrSpecularGlossiness"))
                    .and_then(|sg| sg.get("diffuseFactor"))
            })
            .and_then(JsonValue::as_array)?;
        if factor_arr.len() < 3 {
            return None;
        }
        let r = factor_arr[0].as_f64()? as f32;
        let g = factor_arr[1].as_f64()? as f32;
        let b = factor_arr[2].as_f64()? as f32;
        let a = factor_arr
            .get(3)
            .and_then(JsonValue::as_f64)
            .map_or(1.0_f32, |v| v as f32);
        Some([r, g, b, a])
    }

    fn read_indices(&self, accessor_idx: usize) -> Result<Vec<u32>, GltfLoadError> {
        let s = self.accessor_slice_params(accessor_idx, "SCALAR", MAX_TOTAL_INDICES)?;
        let mut out = Vec::with_capacity(s.count);
        for i in 0..s.count {
            let off = s.base + i * s.stride;
            let idx = match s.component_type {
                COMP_UNSIGNED_BYTE => u32::from(s.bytes[off]),
                COMP_UNSIGNED_SHORT => u32::from(read_u16_le(s.bytes, off)?),
                COMP_UNSIGNED_INT => read_u32_le(s.bytes, off)?,
                other => {
                    return Err(GltfLoadError::InvalidBuffer(format!(
                        "invalid index componentType {other}"
                    )));
                }
            };
            out.push(idx);
        }
        Ok(out)
    }

    fn build_mesh(
        &self,
        mesh_idx: usize,
        existing_vertices: usize,
        existing_indices: usize,
    ) -> Result<MeshData, GltfLoadError> {
        let meshes = self
            .root
            .get("meshes")
            .and_then(JsonValue::as_array)
            .ok_or(GltfLoadError::EmptyMesh)?;
        let mesh_obj = meshes.get(mesh_idx).ok_or_else(|| {
            GltfLoadError::InvalidJson(format!("mesh index {mesh_idx} out of bounds"))
        })?;
        let primitives = mesh_obj
            .get("primitives")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                GltfLoadError::InvalidJson(format!("mesh {mesh_idx} missing primitives"))
            })?;

        let mut combined = MeshData::empty();

        for prim in primitives {
            let mode = prim
                .get("mode")
                .and_then(JsonValue::as_u32)
                .unwrap_or(MODE_TRIANGLES);
            if !matches!(
                mode,
                MODE_TRIANGLES | MODE_TRIANGLE_STRIP | MODE_TRIANGLE_FAN
            ) {
                continue;
            }
            let attrs = prim.get("attributes").ok_or_else(|| {
                GltfLoadError::InvalidJson(String::from("primitive missing attributes"))
            })?;
            let Some(pos_acc) = attrs.get("POSITION").and_then(JsonValue::as_usize) else {
                continue;
            };
            let pos_count = self
                .accessors
                .get(pos_acc)
                .ok_or_else(|| {
                    GltfLoadError::InvalidBuffer(format!("accessor {pos_acc} out of bounds"))
                })?
                .count;
            let projected_verts = existing_vertices
                .checked_add(combined.vertices.len())
                .and_then(|v| v.checked_add(pos_count))
                .ok_or_else(|| {
                    GltfLoadError::InvalidJson(format!(
                        "glTF scene exceeds maximum vertex budget of {MAX_TOTAL_VERTICES}"
                    ))
                })?;
            if projected_verts > MAX_TOTAL_VERTICES {
                return Err(GltfLoadError::InvalidJson(format!(
                    "glTF scene exceeds maximum vertex budget of {MAX_TOTAL_VERTICES}"
                )));
            }

            let raw_index_count =
                if let Some(i_acc) = prim.get("indices").and_then(JsonValue::as_usize) {
                    let count = self
                        .accessors
                        .get(i_acc)
                        .ok_or_else(|| {
                            GltfLoadError::InvalidBuffer(format!("accessor {i_acc} out of bounds"))
                        })?
                        .count;
                    if count > MAX_TOTAL_INDICES {
                        return Err(GltfLoadError::InvalidJson(format!(
                            "glTF scene exceeds maximum index budget of {MAX_TOTAL_INDICES}"
                        )));
                    }
                    count
                } else {
                    pos_count
                };
            let est_tri_indices = match mode {
                MODE_TRIANGLES => (raw_index_count / 3).saturating_mul(3),
                MODE_TRIANGLE_STRIP | MODE_TRIANGLE_FAN => {
                    raw_index_count.saturating_sub(2).saturating_mul(3)
                }
                _ => 0,
            };
            let projected_indices = existing_indices
                .checked_add(combined.indices.len())
                .and_then(|v| v.checked_add(est_tri_indices))
                .ok_or_else(|| {
                    GltfLoadError::InvalidJson(format!(
                        "glTF scene exceeds maximum index budget of {MAX_TOTAL_INDICES}"
                    ))
                })?;
            if projected_indices > MAX_TOTAL_INDICES {
                return Err(GltfLoadError::InvalidJson(format!(
                    "glTF scene exceeds maximum index budget of {MAX_TOTAL_INDICES}"
                )));
            }

            let positions = self.read_vec3_f32(pos_acc)?;
            if positions.is_empty() {
                continue;
            }

            let normals = if let Some(n_acc) = attrs.get("NORMAL").and_then(JsonValue::as_usize) {
                let n_count = self
                    .accessors
                    .get(n_acc)
                    .ok_or_else(|| {
                        GltfLoadError::InvalidBuffer(format!("accessor {n_acc} out of bounds"))
                    })?
                    .count;
                if n_count > MAX_TOTAL_VERTICES {
                    return Err(GltfLoadError::InvalidJson(format!(
                        "NORMAL accessor {n_acc} count {n_count} exceeds maximum vertex budget of {MAX_TOTAL_VERTICES}"
                    )));
                }
                if n_count != positions.len() {
                    return Err(GltfLoadError::InvalidBuffer(String::from(
                        "NORMAL count does not match POSITION count",
                    )));
                }
                Some(self.read_vec3_f32(n_acc)?)
            } else {
                None
            };

            let uvs = if let Some(uv_acc) = attrs.get("TEXCOORD_0").and_then(JsonValue::as_usize) {
                let uv_count = self
                    .accessors
                    .get(uv_acc)
                    .ok_or_else(|| {
                        GltfLoadError::InvalidBuffer(format!("accessor {uv_acc} out of bounds"))
                    })?
                    .count;
                if uv_count > MAX_TOTAL_VERTICES {
                    return Err(GltfLoadError::InvalidJson(format!(
                        "TEXCOORD_0 accessor {uv_acc} count {uv_count} exceeds maximum vertex budget of {MAX_TOTAL_VERTICES}"
                    )));
                }
                if uv_count != positions.len() {
                    return Err(GltfLoadError::InvalidBuffer(String::from(
                        "TEXCOORD_0 count does not match POSITION count",
                    )));
                }
                Some(self.read_vec2_f32(uv_acc)?)
            } else {
                None
            };

            let color0 = if let Some(c_acc) = attrs.get("COLOR_0").and_then(JsonValue::as_usize) {
                let c_count = self
                    .accessors
                    .get(c_acc)
                    .ok_or_else(|| {
                        GltfLoadError::InvalidBuffer(format!("accessor {c_acc} out of bounds"))
                    })?
                    .count;
                if c_count > MAX_TOTAL_VERTICES {
                    return Err(GltfLoadError::InvalidJson(format!(
                        "COLOR_0 accessor {c_acc} count {c_count} exceeds maximum vertex budget of {MAX_TOTAL_VERTICES}"
                    )));
                }
                if c_count != positions.len() {
                    return Err(GltfLoadError::InvalidBuffer(String::from(
                        "COLOR_0 count does not match POSITION count",
                    )));
                }
                Some(self.read_color0(c_acc)?)
            } else {
                None
            };

            let mut raw_indices: Vec<u32> =
                if let Some(i_acc) = prim.get("indices").and_then(JsonValue::as_usize) {
                    self.read_indices(i_acc)?
                } else {
                    (0..positions.len() as u32).collect()
                };

            let vcount = positions.len() as u32;
            for &idx in &raw_indices {
                if idx >= vcount {
                    return Err(GltfLoadError::InvalidBuffer(format!(
                        "index {idx} exceeds vertex count {vcount}"
                    )));
                }
            }

            let tri_indices = match mode {
                MODE_TRIANGLES => {
                    let usable = (raw_indices.len() / 3) * 3;
                    raw_indices.truncate(usable);
                    raw_indices
                }
                MODE_TRIANGLE_STRIP => {
                    let tri_cap = raw_indices.len().saturating_sub(2).saturating_mul(3);
                    let mut out_idx = Vec::with_capacity(tri_cap);
                    for k in 0..raw_indices.len().saturating_sub(2) {
                        if k % 2 == 0 {
                            out_idx.extend_from_slice(&[
                                raw_indices[k],
                                raw_indices[k + 1],
                                raw_indices[k + 2],
                            ]);
                        } else {
                            out_idx.extend_from_slice(&[
                                raw_indices[k],
                                raw_indices[k + 2],
                                raw_indices[k + 1],
                            ]);
                        }
                    }
                    out_idx
                }
                MODE_TRIANGLE_FAN if raw_indices.len() >= 3 => {
                    let tri_cap = raw_indices.len().saturating_sub(2).saturating_mul(3);
                    let mut out_idx = Vec::with_capacity(tri_cap);
                    let root_i = raw_indices[0];
                    for k in 1..(raw_indices.len() - 1) {
                        out_idx.extend_from_slice(&[root_i, raw_indices[k], raw_indices[k + 1]]);
                    }
                    out_idx
                }
                _ => Vec::new(),
            };

            if tri_indices.len() < 3 {
                continue;
            }

            let material_idx = prim.get("material").and_then(JsonValue::as_usize);
            let material_slot = material_idx.map_or(0, |idx| idx as u32);
            let base_color_factor =
                material_idx.and_then(|idx| self.material_base_color_factor(idx));

            let mut vertices = Vec::with_capacity(positions.len());
            for (vi, &pos) in positions.iter().enumerate() {
                let n = if let Some(ref ns) = normals {
                    let nv = Vec3::from_array(ns[vi]);
                    if nv.length_squared() > 1e-12 {
                        nv.normalize().to_array()
                    } else {
                        [0.0, 0.0, 1.0]
                    }
                } else {
                    [0.0, 0.0, 1.0]
                };
                let uv = uvs.as_ref().map_or([0.0, 0.0], |ts| ts[vi]);
                vertices.push(Vertex3D::new(pos, n, uv, material_slot));
            }

            let mut prim_mesh = match (color0, base_color_factor) {
                (Some(mut cols), Some(factor)) => {
                    for c in &mut cols {
                        c[0] *= factor[0];
                        c[1] *= factor[1];
                        c[2] *= factor[2];
                        c[3] *= factor[3];
                    }
                    MeshData::with_colors(vertices, tri_indices, cols)
                }
                (Some(cols), None) => MeshData::with_colors(vertices, tri_indices, cols),
                (None, Some(factor)) => {
                    let cols = vec![factor; vertices.len()];
                    MeshData::with_colors(vertices, tri_indices, cols)
                }
                (None, None) => MeshData::new(vertices, tri_indices),
            };
            if normals.is_none() {
                prim_mesh.compute_smooth_normals();
            }
            if combined.vertices.is_empty() {
                combined = prim_mesh;
            } else {
                combined.merge(&prim_mesh);
            }
        }

        Ok(combined)
    }

    fn node_local_matrix(node: &JsonValue) -> Result<Mat4, GltfLoadError> {
        if let Some(mat_arr) = node.get("matrix").and_then(JsonValue::as_array) {
            if mat_arr.len() != 16 {
                return Err(GltfLoadError::InvalidJson(String::from(
                    "node matrix must have 16 elements",
                )));
            }
            let mut cols = [0.0_f32; 16];
            for (i, item) in mat_arr.iter().enumerate() {
                cols[i] = item.as_f64().ok_or_else(|| {
                    GltfLoadError::InvalidJson(String::from("non-numeric matrix element"))
                })? as f32;
            }
            return Ok(Mat4::from_cols_array(&cols));
        }

        let translation = if let Some(arr) = node.get("translation").and_then(JsonValue::as_array) {
            if arr.len() != 3 {
                return Err(GltfLoadError::InvalidJson(String::from(
                    "node translation must have 3 elements",
                )));
            }
            Vec3::new(
                arr[0].as_f64().unwrap_or(0.0) as f32,
                arr[1].as_f64().unwrap_or(0.0) as f32,
                arr[2].as_f64().unwrap_or(0.0) as f32,
            )
        } else {
            Vec3::ZERO
        };

        let rotation = if let Some(arr) = node.get("rotation").and_then(JsonValue::as_array) {
            if arr.len() != 4 {
                return Err(GltfLoadError::InvalidJson(String::from(
                    "node rotation must have 4 elements",
                )));
            }
            let q = Quat::from_xyzw(
                arr[0].as_f64().unwrap_or(0.0) as f32,
                arr[1].as_f64().unwrap_or(0.0) as f32,
                arr[2].as_f64().unwrap_or(0.0) as f32,
                arr[3].as_f64().unwrap_or(1.0) as f32,
            );
            if q.length_squared() > 1e-12 {
                q.normalize()
            } else {
                Quat::IDENTITY
            }
        } else {
            Quat::IDENTITY
        };

        let scale = if let Some(arr) = node.get("scale").and_then(JsonValue::as_array) {
            if arr.len() != 3 {
                return Err(GltfLoadError::InvalidJson(String::from(
                    "node scale must have 3 elements",
                )));
            }
            Vec3::new(
                arr[0].as_f64().unwrap_or(1.0) as f32,
                arr[1].as_f64().unwrap_or(1.0) as f32,
                arr[2].as_f64().unwrap_or(1.0) as f32,
            )
        } else {
            Vec3::ONE
        };

        Ok(Mat4::from_scale_rotation_translation(
            scale,
            rotation,
            translation,
        ))
    }

    fn assemble_scene(&self) -> Result<MeshData, GltfLoadError> {
        enum TraversalStep {
            Enter {
                node_idx: usize,
                parent_world: Mat4,
                depth: usize,
            },
            Exit {
                node_idx: usize,
            },
        }

        let mut out = MeshData::empty();
        let mut total_vertices = 0usize;
        let mut total_indices = 0usize;

        if let Some(nodes) = self.root.get("nodes").and_then(JsonValue::as_array)
            && !nodes.is_empty()
        {
            let root_indices: Vec<usize> = if let Some(scenes) =
                self.root.get("scenes").and_then(JsonValue::as_array)
                && !scenes.is_empty()
            {
                let scene_idx = self
                    .root
                    .get("scene")
                    .and_then(JsonValue::as_usize)
                    .unwrap_or(0)
                    .min(scenes.len() - 1);
                if let Some(arr) = scenes[scene_idx].get("nodes").and_then(JsonValue::as_array) {
                    if arr.len() > MAX_SCENE_NODE_VISITS {
                        return Err(GltfLoadError::InvalidJson(format!(
                            "glTF scene traversal exceeds maximum node visit budget of {MAX_SCENE_NODE_VISITS}"
                        )));
                    }
                    arr.iter().filter_map(JsonValue::as_usize).collect()
                } else {
                    Vec::new()
                }
            } else {
                let mut child_set = HashSet::new();
                for node in nodes {
                    if let Some(children) = node.get("children").and_then(JsonValue::as_array) {
                        for c in children {
                            if let Some(ci) = c.as_usize() {
                                child_set.insert(ci);
                            }
                        }
                    }
                }
                let mut roots = Vec::new();
                for i in 0..nodes.len() {
                    if !child_set.contains(&i) {
                        if roots.len() >= MAX_SCENE_NODE_VISITS {
                            return Err(GltfLoadError::InvalidJson(format!(
                                "glTF scene traversal exceeds maximum node visit budget of {MAX_SCENE_NODE_VISITS}"
                            )));
                        }
                        roots.push(i);
                    }
                }
                if roots.is_empty() {
                    return Err(GltfLoadError::InvalidJson(String::from(
                        "cyclic glTF node graph has no root nodes",
                    )));
                }
                roots
            };

            let mut scheduled_visits = root_indices.len();
            if scheduled_visits > MAX_SCENE_NODE_VISITS {
                return Err(GltfLoadError::InvalidJson(format!(
                    "glTF scene traversal exceeds maximum node visit budget of {MAX_SCENE_NODE_VISITS}"
                )));
            }

            let mut active_ancestors = HashSet::new();
            let mut visits = 0usize;
            let mut stack = Vec::with_capacity(root_indices.len());
            for &root_idx in root_indices.iter().rev() {
                stack.push(TraversalStep::Enter {
                    node_idx: root_idx,
                    parent_world: Mat4::IDENTITY,
                    depth: 0,
                });
            }

            while let Some(step) = stack.pop() {
                match step {
                    TraversalStep::Exit { node_idx } => {
                        active_ancestors.remove(&node_idx);
                    }
                    TraversalStep::Enter {
                        node_idx,
                        parent_world,
                        depth,
                    } => {
                        if depth >= MAX_NODE_DEPTH {
                            return Err(GltfLoadError::InvalidJson(format!(
                                "glTF node hierarchy exceeds maximum depth of {MAX_NODE_DEPTH}"
                            )));
                        }
                        if visits >= MAX_SCENE_NODE_VISITS {
                            return Err(GltfLoadError::InvalidJson(format!(
                                "glTF scene traversal exceeds maximum node visit budget of {MAX_SCENE_NODE_VISITS}"
                            )));
                        }
                        if !active_ancestors.insert(node_idx) {
                            return Err(GltfLoadError::InvalidJson(format!(
                                "cyclic glTF node reference detected at node {node_idx}"
                            )));
                        }
                        visits += 1;

                        let node = nodes.get(node_idx).ok_or_else(|| {
                            GltfLoadError::InvalidJson(format!(
                                "node index {node_idx} out of bounds"
                            ))
                        })?;
                        let local = Self::node_local_matrix(node)?;
                        let world = parent_world * local;

                        if let Some(mesh_idx) = node.get("mesh").and_then(JsonValue::as_usize) {
                            let mut m = self.build_mesh(mesh_idx, total_vertices, total_indices)?;
                            if !m.vertices.is_empty() {
                                let next_verts = total_vertices
                                    .checked_add(m.vertices.len())
                                    .filter(|&v| v <= MAX_TOTAL_VERTICES)
                                    .ok_or_else(|| {
                                        GltfLoadError::InvalidJson(format!(
                                            "glTF scene exceeds maximum vertex budget of {MAX_TOTAL_VERTICES}"
                                        ))
                                    })?;
                                let next_idxs = total_indices
                                    .checked_add(m.indices.len())
                                    .filter(|&i| i <= MAX_TOTAL_INDICES)
                                    .ok_or_else(|| {
                                        GltfLoadError::InvalidJson(format!(
                                            "glTF scene exceeds maximum index budget of {MAX_TOTAL_INDICES}"
                                        ))
                                    })?;
                                total_vertices = next_verts;
                                total_indices = next_idxs;
                                if world != Mat4::IDENTITY {
                                    m.apply_matrix(world);
                                }
                                if out.vertices.is_empty() {
                                    out = m;
                                } else {
                                    out.merge(&m);
                                }
                            }
                        }

                        stack.push(TraversalStep::Exit { node_idx });

                        if let Some(children) = node.get("children").and_then(JsonValue::as_array) {
                            scheduled_visits = scheduled_visits
                                .checked_add(children.len())
                                .filter(|&total| total <= MAX_SCENE_NODE_VISITS)
                                .ok_or_else(|| {
                                    GltfLoadError::InvalidJson(format!(
                                        "glTF scene traversal exceeds maximum node visit budget of {MAX_SCENE_NODE_VISITS}"
                                    ))
                                })?;
                            for child in children.iter().rev() {
                                if let Some(child_idx) = child.as_usize() {
                                    stack.push(TraversalStep::Enter {
                                        node_idx: child_idx,
                                        parent_world: world,
                                        depth: depth + 1,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        } else if let Some(meshes) = self.root.get("meshes").and_then(JsonValue::as_array) {
            for mesh_idx in 0..meshes.len() {
                let m = self.build_mesh(mesh_idx, total_vertices, total_indices)?;
                let next_verts = total_vertices
                    .checked_add(m.vertices.len())
                    .filter(|&v| v <= MAX_TOTAL_VERTICES)
                    .ok_or_else(|| {
                        GltfLoadError::InvalidJson(format!(
                            "glTF scene exceeds maximum vertex budget of {MAX_TOTAL_VERTICES}"
                        ))
                    })?;
                let next_idxs = total_indices
                    .checked_add(m.indices.len())
                    .filter(|&i| i <= MAX_TOTAL_INDICES)
                    .ok_or_else(|| {
                        GltfLoadError::InvalidJson(format!(
                            "glTF scene exceeds maximum index budget of {MAX_TOTAL_INDICES}"
                        ))
                    })?;
                total_vertices = next_verts;
                total_indices = next_idxs;
                if out.vertices.is_empty() {
                    out = m;
                } else {
                    out.merge(&m);
                }
            }
        }

        if out.vertices.is_empty() || out.indices.is_empty() {
            return Err(GltfLoadError::EmptyMesh);
        }
        Ok(out)
    }
}

fn parse_document(
    json_str: &str,
    glb_bin_chunk: Option<&[u8]>,
    external_buffers: &[&[u8]],
) -> Result<MeshData, GltfLoadError> {
    parse_document_with_resolver::<fn(&str) -> Option<Vec<u8>>>(
        json_str,
        glb_bin_chunk,
        external_buffers,
        None,
    )
}

fn parse_document_with_resolver<F>(
    json_str: &str,
    glb_bin_chunk: Option<&[u8]>,
    external_buffers: &[&[u8]],
    mut resolver: Option<F>,
) -> Result<MeshData, GltfLoadError>
where
    F: FnMut(&str) -> Option<Vec<u8>>,
{
    let root = JsonParser::new(json_str).parse_root()?;

    let mut buffers: Vec<Cow<'_, [u8]>> = Vec::new();
    if let Some(buf_arr) = root.get("buffers").and_then(JsonValue::as_array) {
        buffers.reserve(buf_arr.len());
        for (i, buf_val) in buf_arr.iter().enumerate() {
            if let Some(uri) = buf_val.get("uri").and_then(JsonValue::as_str) {
                if uri.starts_with("data:") {
                    buffers.push(Cow::Owned(decode_data_uri(uri)?));
                } else if let Some(resolved) = resolver.as_mut().and_then(|f| f(uri)) {
                    buffers.push(Cow::Owned(resolved));
                } else {
                    buffers.push(Cow::Borrowed(&[])); // Resolved via external_buffers[i]
                }
            } else if i == 0
                && let Some(bin) = glb_bin_chunk
            {
                buffers.push(Cow::Borrowed(bin));
            } else {
                buffers.push(Cow::Borrowed(&[]));
            }
        }
    }

    let mut buffer_views = Vec::new();
    if let Some(bv_arr) = root.get("bufferViews").and_then(JsonValue::as_array) {
        buffer_views.reserve(bv_arr.len());
        for bv in bv_arr {
            let buffer = bv
                .get("buffer")
                .and_then(JsonValue::as_usize)
                .ok_or_else(|| {
                    GltfLoadError::InvalidJson(String::from("bufferView missing buffer"))
                })?;
            let byte_length = bv
                .get("byteLength")
                .and_then(JsonValue::as_usize)
                .ok_or_else(|| {
                    GltfLoadError::InvalidJson(String::from("bufferView missing byteLength"))
                })?;
            let byte_offset = bv
                .get("byteOffset")
                .and_then(JsonValue::as_usize)
                .unwrap_or(0);
            let byte_stride = bv.get("byteStride").and_then(JsonValue::as_usize);
            buffer_views.push(BufferViewInfo {
                buffer,
                byte_offset,
                byte_length,
                byte_stride,
            });
        }
    }

    let mut accessors = Vec::new();
    if let Some(acc_arr) = root.get("accessors").and_then(JsonValue::as_array) {
        accessors.reserve(acc_arr.len());
        for acc in acc_arr {
            let buffer_view = acc
                .get("bufferView")
                .and_then(JsonValue::as_usize)
                .ok_or_else(|| {
                    GltfLoadError::UnsupportedFeature(String::from(
                        "sparse/zero-initialized accessor without bufferView",
                    ))
                })?;
            let byte_offset = acc
                .get("byteOffset")
                .and_then(JsonValue::as_usize)
                .unwrap_or(0);
            let component_type = acc
                .get("componentType")
                .and_then(JsonValue::as_u32)
                .ok_or_else(|| {
                    GltfLoadError::InvalidJson(String::from("accessor missing componentType"))
                })?;
            let count = acc
                .get("count")
                .and_then(JsonValue::as_usize)
                .ok_or_else(|| {
                    GltfLoadError::InvalidJson(String::from("accessor missing count"))
                })?;
            let accessor_type = acc
                .get("type")
                .and_then(JsonValue::as_str)
                .ok_or_else(|| GltfLoadError::InvalidJson(String::from("accessor missing type")))?
                .to_owned();
            let normalized = acc
                .get("normalized")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false);
            accessors.push(AccessorInfo {
                buffer_view,
                byte_offset,
                component_type,
                count,
                accessor_type,
                normalized,
            });
        }
    }

    let doc = GltfDocument {
        root,
        buffers,
        external_buffers,
        buffer_views,
        accessors,
    };
    doc.assemble_scene()
}

fn extract_glb_chunks(bytes: &[u8]) -> Result<(&str, Option<&[u8]>), GltfLoadError> {
    if bytes.len() < 12 {
        return Err(GltfLoadError::InvalidHeader(String::from(
            "buffer shorter than 12-byte GLB header",
        )));
    }
    let magic = read_u32_le(bytes, 0)?;
    if magic != GLB_MAGIC {
        return Err(GltfLoadError::InvalidHeader(format!(
            "bad magic 0x{magic:08X} (expected 0x{GLB_MAGIC:08X})"
        )));
    }
    let version = read_u32_le(bytes, 4)?;
    if version != GLB_VERSION_2 {
        return Err(GltfLoadError::InvalidHeader(format!(
            "unsupported GLB version {version} (expected 2)"
        )));
    }
    let declared_len = read_u32_le(bytes, 8)? as usize;
    if declared_len < 12 || declared_len > bytes.len() {
        return Err(GltfLoadError::InvalidHeader(format!(
            "declared length {declared_len} invalid for {} input bytes",
            bytes.len()
        )));
    }

    let mut offset = 12_usize;
    let mut json_chunk: Option<&[u8]> = None;
    let mut bin_chunk: Option<&[u8]> = None;

    while offset + 8 <= declared_len {
        let chunk_len = read_u32_le(bytes, offset)? as usize;
        let chunk_type = read_u32_le(bytes, offset + 4)?;
        let data_start = offset + 8;
        let data_end = data_start.checked_add(chunk_len).ok_or_else(|| {
            GltfLoadError::InvalidHeader(String::from("GLB chunk length overflow"))
        })?;
        if data_end > declared_len {
            return Err(GltfLoadError::InvalidHeader(format!(
                "GLB chunk at {offset} exceeds declared length {declared_len}"
            )));
        }
        let payload = &bytes[data_start..data_end];
        match chunk_type {
            CHUNK_TYPE_JSON if json_chunk.is_none() => json_chunk = Some(payload),
            CHUNK_TYPE_BIN if bin_chunk.is_none() => bin_chunk = Some(payload),
            _ => {}
        }
        offset = data_end;
    }

    let json_bytes = json_chunk
        .ok_or_else(|| GltfLoadError::InvalidHeader(String::from("GLB missing JSON chunk")))?;
    let json_str = std::str::from_utf8(json_bytes)
        .map_err(|_| GltfLoadError::InvalidJson(String::from("JSON chunk is not valid UTF-8")))?;
    Ok((json_str, bin_chunk))
}

/// Parses a binary glTF 2.0 (`.glb`) container into a [`MeshData`].
pub fn parse_glb(bytes: &[u8]) -> Result<MeshData, GltfLoadError> {
    let (json_str, bin_chunk) = extract_glb_chunks(bytes)?;
    parse_document(json_str, bin_chunk, &[])
}

/// Parses a text glTF 2.0 (`.gltf`) JSON document with optional external buffer slices.
pub fn parse_gltf(json_str: &str, external_buffers: &[&[u8]]) -> Result<MeshData, GltfLoadError> {
    parse_document(json_str, None, external_buffers)
}

/// Auto-detects binary `.glb` (`b"glTF"` magic) or UTF-8 `.gltf` JSON and parses it into a
/// [`MeshData`].
pub fn parse_gltf_or_glb(bytes: &[u8]) -> Result<MeshData, GltfLoadError> {
    if bytes.len() >= 4 && &bytes[..4] == b"glTF" {
        parse_glb(bytes)
    } else {
        let text = std::str::from_utf8(bytes)
            .map_err(|_| GltfLoadError::InvalidJson(String::from("input is not valid UTF-8")))?;
        parse_gltf(text, &[])
    }
}

/// Loads a `.glb` or `.gltf` byte slice into a [`MeshData`].
pub fn load_gltf(bytes: &[u8]) -> Result<MeshData, GltfLoadError> {
    parse_gltf_or_glb(bytes)
}

/// Loads a `.glb` or `.gltf` byte slice, resolving external non-`data:` buffer URIs (such as
/// `"mesh.bin"`) via `resolver`.
pub fn load_gltf_with_resolver<F>(bytes: &[u8], resolver: F) -> Result<MeshData, GltfLoadError>
where
    F: FnMut(&str) -> Option<Vec<u8>>,
{
    if bytes.len() >= 4 && &bytes[..4] == b"glTF" {
        let (json_str, bin_chunk) = extract_glb_chunks(bytes)?;
        parse_document_with_resolver(json_str, bin_chunk, &[], Some(resolver))
    } else {
        let text = std::str::from_utf8(bytes)
            .map_err(|_| GltfLoadError::InvalidJson(String::from("input is not valid UTF-8")))?;
        parse_document_with_resolver(text, None, &[], Some(resolver))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_triangle_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        for p in [[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for c in p {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        for idx in [0_u16, 1, 2] {
            bin.extend_from_slice(&idx.to_le_bytes());
        }
        bin
    }

    fn wrap_gltf_with_nodes(nodes_json: &str, bin_len: usize) -> String {
        format!(
            r#"{{
              "asset": {{"version": "2.0"}},
              "scene": 0,
              "scenes": [{{"nodes": [0]}}],
              "nodes": [{nodes_json}],
              "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}, "indices": 1}}]}}],
              "buffers": [{{"byteLength": {bin_len}}}],
              "bufferViews": [
                {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
                {{"buffer": 0, "byteOffset": 36, "byteLength": 6}}
              ],
              "accessors": [
                {{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"}},
                {{"bufferView": 1, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR"}}
              ]
            }}"#
        )
    }

    #[test]
    fn deep_node_chain_exceeding_max_depth_returns_err_without_stack_overflow() {
        let bin = single_triangle_bin();
        // Build a 256-node linear chain (> MAX_NODE_DEPTH = 128).
        let chain_len = MAX_NODE_DEPTH + 128;
        let mut nodes = Vec::with_capacity(chain_len);
        for i in 0..chain_len {
            if i + 1 < chain_len {
                nodes.push(format!(r#"{{"children": [{}]}}"#, i + 1));
            } else {
                nodes.push(String::from(r#"{"mesh": 0}"#));
            }
        }
        let gltf = wrap_gltf_with_nodes(&nodes.join(","), bin.len());
        let err = parse_gltf(&gltf, &[&bin]).expect_err("deep node chain must fail cleanly");
        assert!(
            err.to_string().contains("depth"),
            "expected depth budget error, got: {err}"
        );
    }

    #[test]
    fn diamond_dag_exceeding_max_scene_node_visits_returns_err_cleanly() {
        let bin = single_triangle_bin();
        // 16-level diamond DAG where each node points twice to the next level -> 2^16 = 65,536 visits
        // (> MAX_SCENE_NODE_VISITS = 4,096) while depth is only 16 (< MAX_NODE_DEPTH).
        let levels = 16;
        let mut nodes = Vec::with_capacity(levels + 1);
        for i in 0..levels {
            nodes.push(format!(r#"{{"children": [{n}, {n}]}}"#, n = i + 1));
        }
        nodes.push(String::from(r#"{"mesh": 0}"#));
        let gltf = wrap_gltf_with_nodes(&nodes.join(","), bin.len());
        let err =
            parse_gltf(&gltf, &[&bin]).expect_err("exponential diamond DAG must fail cleanly");
        assert!(
            err.to_string().contains("visit budget"),
            "expected node visit budget error, got: {err}"
        );
    }

    #[test]
    fn cyclic_node_graph_returns_err_cleanly() {
        let bin = single_triangle_bin();
        let nodes = r#"{"children": [1], "mesh": 0}, {"children": [0]}"#;
        let gltf = wrap_gltf_with_nodes(nodes, bin.len());
        let err = parse_gltf(&gltf, &[&bin]).expect_err("cyclic node graph must fail cleanly");
        assert!(
            err.to_string().contains("cyclic"),
            "expected cyclic node error, got: {err}"
        );
    }

    #[test]
    fn oversized_normal_and_texcoord_accessor_counts_rejected_before_decode() {
        let bin = single_triangle_bin();
        let huge_count = 50_000_000usize;

        // NORMAL claims 50,000,000 elements over a tiny 42-byte buffer -> must be rejected
        // by pre-decode count validation before touching bufferView bounds or allocating.
        let gltf_normal = format!(
            r#"{{
              "asset": {{"version": "2.0"}},
              "scene": 0,
              "scenes": [{{"nodes": [0]}}],
              "nodes": [{{"mesh": 0}}],
              "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0, "NORMAL": 2}}, "indices": 1}}]}}],
              "buffers": [{{"byteLength": {bin_len}}}],
              "bufferViews": [
                {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
                {{"buffer": 0, "byteOffset": 36, "byteLength": 6}}
              ],
              "accessors": [
                {{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"}},
                {{"bufferView": 1, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR"}},
                {{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": {huge_count}, "type": "VEC3"}}
              ]
            }}"#,
            bin_len = bin.len()
        );
        let err = parse_gltf(&gltf_normal, &[&bin])
            .expect_err("oversized NORMAL count must fail before decoding");
        assert!(
            matches!(err, GltfLoadError::InvalidJson(_))
                && err.to_string().contains("NORMAL")
                && err.to_string().contains("vertex budget"),
            "expected pre-decode NORMAL vertex budget error, got: {err}"
        );

        // TEXCOORD_0 claims 50,000,000 elements over a tiny buffer -> must also be rejected
        // by pre-decode count validation.
        let gltf_uv = format!(
            r#"{{
              "asset": {{"version": "2.0"}},
              "scene": 0,
              "scenes": [{{"nodes": [0]}}],
              "nodes": [{{"mesh": 0}}],
              "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0, "TEXCOORD_0": 2}}, "indices": 1}}]}}],
              "buffers": [{{"byteLength": {bin_len}}}],
              "bufferViews": [
                {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
                {{"buffer": 0, "byteOffset": 36, "byteLength": 6}}
              ],
              "accessors": [
                {{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"}},
                {{"bufferView": 1, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR"}},
                {{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": {huge_count}, "type": "VEC2"}}
              ]
            }}"#,
            bin_len = bin.len()
        );
        let err_uv = parse_gltf(&gltf_uv, &[&bin])
            .expect_err("oversized TEXCOORD_0 count must fail before decoding");
        assert!(
            matches!(err_uv, GltfLoadError::InvalidJson(_))
                && err_uv.to_string().contains("TEXCOORD_0")
                && err_uv.to_string().contains("vertex budget"),
            "expected pre-decode TEXCOORD_0 vertex budget error, got: {err_uv}"
        );
    }

    #[test]
    fn node_listing_excessive_children_rejected_before_pushing_stack() {
        let bin = single_triangle_bin();
        // Node 0 lists MAX_SCENE_NODE_VISITS + 1 children pointing to leaf node 1.
        let many_children = vec!["1"; MAX_SCENE_NODE_VISITS + 1].join(",");
        let nodes = format!(r#"{{"children": [{many_children}]}}, {{"mesh": 0}}"#);
        let gltf = wrap_gltf_with_nodes(&nodes, bin.len());
        let err = parse_gltf(&gltf, &[&bin])
            .expect_err("node listing > MAX_SCENE_NODE_VISITS children must fail cleanly");
        assert!(
            matches!(err, GltfLoadError::InvalidJson(_))
                && err.to_string().contains("visit budget"),
            "expected node visit budget error, got: {err}"
        );
    }

    #[test]
    fn color0_vertex_colors_base_color_factor_and_external_bin_resolver() {
        // Build binary buffer containing:
        // [0..36]:   3 VEC3 f32 positions
        // [36..42]:  3 u16 indices
        // [42..44]:  2-byte pad
        // [44..56]:  3 VEC4 u8 COLOR_0 vertices
        // [56..74]:  3 VEC3 u16 COLOR_0 vertices
        let mut bin = single_triangle_bin();
        bin.extend_from_slice(&[0_u8, 0]);
        let u8_colors: [[u8; 4]; 3] = [[255, 0, 0, 255], [0, 255, 0, 128], [0, 0, 255, 255]];
        for c in u8_colors {
            bin.extend_from_slice(&c);
        }
        let u16_colors: [[u16; 3]; 3] = [[65535, 32768, 0], [0, 65535, 0], [0, 0, 65535]];
        for c in u16_colors {
            for ch in c {
                bin.extend_from_slice(&ch.to_le_bytes());
            }
        }

        // 1. COLOR_0 VEC4 UNSIGNED_BYTE multiplied by material baseColorFactor + external .bin resolver
        let gltf_combined = format!(
            r#"{{
              "asset": {{"version": "2.0"}},
              "scene": 0,
              "scenes": [{{"nodes": [0]}}],
              "nodes": [{{"mesh": 0}}],
              "materials": [
                {{"pbrMetallicRoughness": {{"baseColorFactor": [0.5, 0.5, 1.0, 0.5]}}}}
              ],
              "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0, "COLOR_0": 2}}, "indices": 1, "material": 0}}]}}],
              "buffers": [{{"uri": "external_mesh.bin", "byteLength": {len}}}],
              "bufferViews": [
                {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
                {{"buffer": 0, "byteOffset": 36, "byteLength": 6}},
                {{"buffer": 0, "byteOffset": 44, "byteLength": 12}}
              ],
              "accessors": [
                {{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"}},
                {{"bufferView": 1, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR"}},
                {{"bufferView": 2, "byteOffset": 0, "componentType": 5121, "normalized": true, "count": 3, "type": "VEC4"}}
              ]
            }}"#,
            len = bin.len()
        );
        let mesh = load_gltf_with_resolver(gltf_combined.as_bytes(), |uri| {
            (uri == "external_mesh.bin").then(|| bin.clone())
        })
        .expect("glTF with COLOR_0 + baseColorFactor + external resolver");
        assert_eq!(mesh.colors.len(), 3);
        assert!((mesh.colors[0][0] - 0.5).abs() < 1e-3);
        assert!((mesh.colors[0][3] - 0.5).abs() < 1e-3);
        assert!((mesh.colors[1][1] - 0.5).abs() < 1e-3);
        assert!((mesh.colors[2][2] - 1.0).abs() < 1e-3);

        // 2. COLOR_0 VEC3 UNSIGNED_SHORT (alpha defaults to 1.0)
        let gltf_u16 = format!(
            r#"{{
              "asset": {{"version": "2.0"}},
              "scene": 0,
              "scenes": [{{"nodes": [0]}}],
              "nodes": [{{"mesh": 0}}],
              "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0, "COLOR_0": 2}}, "indices": 1}}]}}],
              "buffers": [{{"byteLength": {len}}}],
              "bufferViews": [
                {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
                {{"buffer": 0, "byteOffset": 36, "byteLength": 6}},
                {{"buffer": 0, "byteOffset": 56, "byteLength": 18}}
              ],
              "accessors": [
                {{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"}},
                {{"bufferView": 1, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR"}},
                {{"bufferView": 2, "byteOffset": 0, "componentType": 5123, "normalized": true, "count": 3, "type": "VEC3"}}
              ]
            }}"#,
            len = bin.len()
        );
        let mesh_u16 = parse_gltf(&gltf_u16, &[&bin]).expect("glTF with VEC3 u16 COLOR_0");
        assert_eq!(mesh_u16.colors.len(), 3);
        assert!((mesh_u16.colors[0][0] - 1.0).abs() < 1e-3);
        assert!((mesh_u16.colors[0][1] - 0.5).abs() < 1e-3);
        assert!((mesh_u16.colors[0][3] - 1.0).abs() < 1e-5);

        // 3. Material baseColorFactor only (no COLOR_0 attribute)
        let gltf_mat_only = format!(
            r#"{{
              "asset": {{"version": "2.0"}},
              "scene": 0,
              "scenes": [{{"nodes": [0]}}],
              "nodes": [{{"mesh": 0}}],
              "materials": [
                {{"pbrMetallicRoughness": {{"baseColorFactor": [0.2, 0.6, 0.8, 0.9]}}}}
              ],
              "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}, "indices": 1, "material": 0}}]}}],
              "buffers": [{{"byteLength": {len}}}],
              "bufferViews": [
                {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
                {{"buffer": 0, "byteOffset": 36, "byteLength": 6}}
              ],
              "accessors": [
                {{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"}},
                {{"bufferView": 1, "byteOffset": 0, "componentType": 5123, "count": 3, "type": "SCALAR"}}
              ]
            }}"#,
            len = bin.len()
        );
        let mesh_mat = parse_gltf(&gltf_mat_only, &[&bin]).expect("glTF with baseColorFactor");
        assert_eq!(mesh_mat.colors.len(), 3);
        for c in &mesh_mat.colors {
            assert!((c[0] - 0.2).abs() < 1e-5);
            assert!((c[1] - 0.6).abs() < 1e-5);
            assert!((c[2] - 0.8).abs() < 1e-5);
            assert!((c[3] - 0.9).abs() < 1e-5);
        }
    }
}
