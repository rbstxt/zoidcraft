//! Clean-room GLSL ES 1.00 -> GLSL 450 -> `naga` -> WGSL translator for Panzoid Custom Shaders.
//!
//! Translates WebGL 1 / ClipMaker 3 custom fragment shaders (Spec 04 §3) into validated
//! WGSL modules along with deterministic `std140` uniform buffer layouts and texture
//! binding tables for `zc-render`.

use std::collections::{BTreeMap, HashMap, HashSet};

use naga::back::wgsl::{self, WriterFlags};
use naga::front::glsl::{Frontend, Options};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MAX_INCLUDE_DEPTH: usize = 8;
/// Maximum uniform array element count allowed in a custom shader UBO.
const MAX_UNIFORM_ARRAY_LEN: u32 = 1024;
/// Maximum custom shader uniform block byte size (matches WebGPU minimum 64 KiB UBO binding limit).
const MAX_UNIFORM_BUFFER_BYTES: u32 = 65_536;

const INCLUDE_GRADIENT: &str = r#"
float zGradientUv(float t, float repetitions) {
    float reps = max(repetitions, 0.0001);
    float u = fract(t * reps);
    return clamp(u, 0.5 / 256.0, 255.5 / 256.0);
}
float zGradientUvClamp(float t) {
    return clamp(t, 0.5 / 256.0, 255.5 / 256.0);
}
float zSrgbToLinearChannel(float c) {
    return c <= 0.04045 ? c / 12.92 : pow(max((c + 0.055) / 1.055, 0.0), 2.4);
}
vec3 zSrgbToLinear(vec3 c) {
    return vec3(
        zSrgbToLinearChannel(c.r),
        zSrgbToLinearChannel(c.g),
        zSrgbToLinearChannel(c.b)
    );
}
float zLinearToSrgbChannel(float c) {
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055;
}
vec3 zLinearToSrgb(vec3 c) {
    return vec3(
        zLinearToSrgbChannel(c.r),
        zLinearToSrgbChannel(c.g),
        zLinearToSrgbChannel(c.b)
    );
}
"#;

const INCLUDE_CURVE: &str = r#"
float zCurveUv(float t) {
    return clamp(t, 0.5 / 1024.0, 1023.5 / 1024.0);
}
float zCurveMapRange(float v, vec2 range) {
    return mix(range.x, range.y, v);
}
"#;

const INCLUDE_TEXT: &str = r#"
float zTextSdfAlpha(float dist, float edge, float softness) {
    float s = max(softness, 0.0001);
    return smoothstep(edge - s, edge + s, dist);
}
float zTextSdfAlphaAA(float dist, float edge) {
    float w = max(fwidth(dist), 0.001);
    return smoothstep(edge - w, edge + w, dist);
}
vec2 zTextAtlasUv(vec2 localUv, float glyphIndex, vec2 grid) {
    vec2 safeGrid = max(grid, vec2(1.0));
    float col = mod(glyphIndex, safeGrid.x);
    float row = floor(glyphIndex / safeGrid.x);
    return (vec2(col, row) + clamp(localUv, vec2(0.0), vec2(1.0))) / safeGrid;
}
"#;

/// Errors emitted while preprocessing, translating, validating, or lowering a custom shader.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ShaderError {
    #[error("shader source is empty")]
    EmptySource,
    #[error("unknown shader include: <{0}>")]
    UnknownInclude(String),
    #[error("shader include depth limit exceeded while expanding <{0}>")]
    IncludeDepthExceeded(String),
    #[error("unsupported WebGL 1 extension: {0}")]
    UnsupportedExtension(String),
    #[error("unsupported legacy WebGL 1 feature: {0}")]
    UnsupportedFeature(String),
    #[error("invalid uniform declaration: {0}")]
    InvalidUniformDeclaration(String),
    #[error("duplicate uniform declaration with conflicting type: {name}")]
    DuplicateUniform { name: String },
    #[error("GLSL 450 parse error: {0}")]
    GlslParse(String),
    #[error("Naga shader validation error: {0}")]
    Validation(String),
    #[error("WGSL emission error: {0}")]
    WgslEmit(String),
}

/// Optional compilation inputs such as static dropdown `#define` values and extra includes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompileOptions {
    /// Preprocessor macros injected right after `#version 450` (e.g., Shader+ `StaticDropdown`s).
    pub defines: BTreeMap<String, String>,
    /// Additional virtual `#include <name>` snippets beyond the built-in `zoidium/*` library.
    pub extra_includes: BTreeMap<String, String>,
}

/// Scalar, vector, or matrix type stored inside the `std140` custom uniform buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UniformType {
    Float,
    Int,
    Uint,
    Bool,
    Vec2,
    Vec3,
    Vec4,
    Ivec2,
    Ivec3,
    Ivec4,
    Uvec2,
    Uvec3,
    Uvec4,
    Mat2,
    Mat3,
    Mat4,
}

impl UniformType {
    /// Parses a GLSL ES 1.00 / 450 non-opaque uniform type keyword.
    pub fn from_glsl(name: &str) -> Option<Self> {
        match name {
            "float" => Some(Self::Float),
            "int" => Some(Self::Int),
            "uint" => Some(Self::Uint),
            "bool" => Some(Self::Bool),
            "vec2" => Some(Self::Vec2),
            "vec3" => Some(Self::Vec3),
            "vec4" => Some(Self::Vec4),
            "ivec2" => Some(Self::Ivec2),
            "ivec3" => Some(Self::Ivec3),
            "ivec4" => Some(Self::Ivec4),
            "uvec2" => Some(Self::Uvec2),
            "uvec3" => Some(Self::Uvec3),
            "uvec4" => Some(Self::Uvec4),
            "mat2" => Some(Self::Mat2),
            "mat3" => Some(Self::Mat3),
            "mat4" => Some(Self::Mat4),
            _ => None,
        }
    }

    /// Base alignment in bytes under `std140` rules.
    pub const fn base_alignment(self) -> u32 {
        match self {
            Self::Float | Self::Int | Self::Uint | Self::Bool => 4,
            Self::Vec2 | Self::Ivec2 | Self::Uvec2 => 8,
            Self::Vec3
            | Self::Vec4
            | Self::Ivec3
            | Self::Ivec4
            | Self::Uvec3
            | Self::Uvec4
            | Self::Mat2
            | Self::Mat3
            | Self::Mat4 => 16,
        }
    }

    /// Size in bytes of a single non-array element under `std140` rules.
    pub const fn byte_size(self) -> u32 {
        match self {
            Self::Float | Self::Int | Self::Uint | Self::Bool => 4,
            Self::Vec2 | Self::Ivec2 | Self::Uvec2 => 8,
            Self::Vec3 | Self::Ivec3 | Self::Uvec3 => 12,
            Self::Vec4 | Self::Ivec4 | Self::Uvec4 => 16,
            Self::Mat2 => 32,
            Self::Mat3 => 48,
            Self::Mat4 => 64,
        }
    }

    /// Stride in bytes between consecutive array elements under `std140` rules.
    pub const fn array_stride(self) -> u32 {
        align_up(self.byte_size(), 16)
    }

    /// GLSL 450 storage type used inside `ZcCustomUniforms`.
    ///
    /// - `bool` is stored as `uint` for WGSL uniform host-shareability.
    /// - Scalar and `vec2` arrays are stored as 4-component vectors so WGSL uniform arrays
    ///   keep their natural 16-byte stride without requiring unsupported `@stride` attributes.
    const fn ubo_glsl_type(self, is_array: bool) -> &'static str {
        if is_array {
            match self {
                Self::Float | Self::Vec2 | Self::Vec4 => "vec4",
                Self::Int | Self::Ivec2 | Self::Ivec4 => "ivec4",
                Self::Uint | Self::Bool | Self::Uvec2 | Self::Uvec4 => "uvec4",
                Self::Vec3 => "vec3",
                Self::Ivec3 => "ivec3",
                Self::Uvec3 => "uvec3",
                Self::Mat2 => "mat2",
                Self::Mat3 => "mat3",
                Self::Mat4 => "mat4",
            }
        } else {
            match self {
                Self::Float => "float",
                Self::Int => "int",
                Self::Uint | Self::Bool => "uint",
                Self::Vec2 => "vec2",
                Self::Vec3 => "vec3",
                Self::Vec4 => "vec4",
                Self::Ivec2 => "ivec2",
                Self::Ivec3 => "ivec3",
                Self::Ivec4 => "ivec4",
                Self::Uvec2 => "uvec2",
                Self::Uvec3 => "uvec3",
                Self::Uvec4 => "uvec4",
                Self::Mat2 => "mat2",
                Self::Mat3 => "mat3",
                Self::Mat4 => "mat4",
            }
        }
    }

    const fn array_swizzle(self) -> Option<&'static str> {
        match self {
            Self::Float | Self::Int | Self::Uint | Self::Bool => Some(".x"),
            Self::Vec2 | Self::Ivec2 | Self::Uvec2 => Some(".xy"),
            Self::Vec3
            | Self::Vec4
            | Self::Ivec3
            | Self::Ivec4
            | Self::Uvec3
            | Self::Uvec4
            | Self::Mat2
            | Self::Mat3
            | Self::Mat4 => None,
        }
    }
}

/// One uniform member inside the packed `std140` custom uniform buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniformField {
    pub name: String,
    pub ty: UniformType,
    /// `None` for non-array uniforms, `Some(len)` for fixed-size arrays like `vec4 Colors[16]`.
    pub array_len: Option<u32>,
    /// Byte offset from the start of the `std140` uniform buffer.
    pub offset: u32,
    /// Total byte size of this member in the `std140` uniform buffer.
    pub size: u32,
    /// Array element stride in bytes (`0` for non-array uniforms).
    pub array_stride: u32,
}

/// Complete `std140` byte layout for `ZcCustomUniforms`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniformBlockLayout {
    pub set: u32,
    pub binding: u32,
    pub fields: Vec<UniformField>,
    /// Total buffer size in bytes (`0` when `fields` is empty, otherwise rounded to a multiple of 16).
    pub byte_size: u32,
}

impl UniformBlockLayout {
    /// Looks up a field by uniform name.
    pub fn field(&self, name: &str) -> Option<&UniformField> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// Binding assignment for a `sampler2D` split into separate WGSL `texture_2d<f32>` + `sampler`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextureBinding {
    pub name: String,
    pub group: u32,
    pub texture_binding: u32,
    pub sampler_binding: u32,
    pub is_diffuse: bool,
}

/// Result of translating and validating a Panzoid GLSL ES 1.00 custom fragment shader.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledCustomShader {
    /// Validated WGSL source emitted by `naga::back::wgsl`.
    pub wgsl: String,
    /// Exact `std140` layout for packing live scalar/vector/array uniforms into a GPU buffer.
    pub uniform_layout: UniformBlockLayout,
    /// Separate `texture2D` + `sampler` bindings (`tDiffuse` is always first when declared).
    pub texture_bindings: Vec<TextureBinding>,
    /// Intermediate Vulkan-flavored GLSL 450 source fed to `naga::front::glsl`.
    pub glsl450_source: String,
}

/// Runtime value packed into `ZcCustomUniforms` via [`pack_uniforms`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UniformValue {
    Float(f32),
    Int(i32),
    Uint(u32),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Ivec2([i32; 2]),
    Ivec3([i32; 3]),
    Ivec4([i32; 4]),
    Uvec2([u32; 2]),
    Uvec3([u32; 3]),
    Uvec4([u32; 4]),
    /// Column-major 2x2 matrix `[c0r0, c0r1, c1r0, c1r1]`.
    Mat2([f32; 4]),
    /// Column-major 3x3 matrix.
    Mat3([f32; 9]),
    /// Column-major 4x4 matrix.
    Mat4([f32; 16]),
    /// Elements for fixed-size uniform arrays such as `vec4 Colors[16]`.
    Array(Vec<UniformValue>),
}

/// Rounds `value` up to the next multiple of `alignment`.
pub const fn align_up(value: u32, alignment: u32) -> u32 {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

/// Translates a Panzoid GLSL ES 1.00 fragment shader into validated WGSL using default options.
pub fn translate_glsl_es100(source: &str) -> Result<CompiledCustomShader, ShaderError> {
    translate_glsl_es100_with_options(source, &CompileOptions::default())
}

/// Maximum allowed GLSL source size before or after preprocessing (512 KiB).
pub const MAX_GLSL_SOURCE_BYTES: usize = 512 * 1024;
/// Maximum allowed call/rewrite recursion depth in the GLSL ES 1.00 shim.
pub const MAX_REWRITE_DEPTH: usize = 64;
/// Maximum allowed tokens emitted during GLSL rewriting.
pub const MAX_REWRITE_TOKENS: usize = 200_000;

struct RewriteBudget {
    tokens_emitted: usize,
}

impl RewriteBudget {
    const fn new() -> Self {
        Self { tokens_emitted: 0 }
    }

    fn charge(&mut self, count: usize) -> Result<(), ShaderError> {
        self.tokens_emitted = self.tokens_emitted.saturating_add(count);
        if self.tokens_emitted > MAX_REWRITE_TOKENS {
            return Err(ShaderError::GlslParse(format!(
                "shader rewrite exceeded {MAX_REWRITE_TOKENS} token budget"
            )));
        }
        Ok(())
    }
}

/// Translates a Panzoid GLSL ES 1.00 fragment shader into validated WGSL with custom defines/includes.
pub fn translate_glsl_es100_with_options(
    source: &str,
    options: &CompileOptions,
) -> Result<CompiledCustomShader, ShaderError> {
    if source.trim().is_empty() {
        return Err(ShaderError::EmptySource);
    }
    if source.len() > MAX_GLSL_SOURCE_BYTES {
        return Err(ShaderError::GlslParse(format!(
            "shader source ({} bytes) exceeds {} byte limit",
            source.len(),
            MAX_GLSL_SOURCE_BYTES
        )));
    }

    let mut included = HashSet::new();
    let mut int_defines = harvest_option_int_defines(&options.defines);
    let preprocessed = preprocess_lines(source, options, &mut included, &mut int_defines, 0)?;
    if preprocessed.len() > MAX_GLSL_SOURCE_BYTES {
        return Err(ShaderError::GlslParse(format!(
            "preprocessed shader ({} bytes) exceeds {} byte limit",
            preprocessed.len(),
            MAX_GLSL_SOURCE_BYTES
        )));
    }

    let tokens = tokenize(&preprocessed);
    if tokens.len() > MAX_REWRITE_TOKENS {
        return Err(ShaderError::GlslParse(format!(
            "shader token count ({}) exceeds {} limit",
            tokens.len(),
            MAX_REWRITE_TOKENS
        )));
    }
    reject_unsupported_features(&tokens)?;

    let extracted = extract_declarations(&tokens, &int_defines)?;
    let uniform_layout = build_uniform_layout(&extracted.uniforms)?;
    let texture_bindings = build_texture_bindings(&extracted.samplers);

    let uniform_map: HashMap<&str, &UniformDecl> = extracted
        .uniforms
        .iter()
        .map(|u| (u.name.as_str(), u))
        .collect();
    let sampler_set: HashSet<&str> = extracted.samplers.iter().map(String::as_str).collect();

    let mut budget = RewriteBudget::new();
    let body_tokens =
        expand_builtin_include_helpers(&extracted.body_tokens, &included, 0, &mut budget)?;
    let rewritten_tokens = rewrite_body(&body_tokens, &uniform_map, &sampler_set, &mut budget)?;
    let body_glsl = render_tokens(&rewritten_tokens);

    let glsl450_source = assemble_glsl450(
        options,
        &extracted.extra_varyings,
        &extracted.uniforms,
        &texture_bindings,
        &body_glsl,
    );

    let mut frontend = Frontend::default();
    let glsl_options = Options::from(naga::ShaderStage::Fragment);
    let module = frontend
        .parse(&glsl_options, &glsl450_source)
        .map_err(|err| ShaderError::GlslParse(err.to_string()))?;

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
    let module_info = validator
        .validate(&module)
        .map_err(|err| ShaderError::Validation(err.to_string()))?;

    let wgsl = wgsl::write_string(&module, &module_info, WriterFlags::empty())
        .map_err(|err| ShaderError::WgslEmit(err.to_string()))?;

    Ok(CompiledCustomShader {
        wgsl,
        uniform_layout,
        texture_bindings,
        glsl450_source,
    })
}

/// Packs live uniform values into a `std140`-aligned byte buffer matching `layout`.
///
/// Missing uniforms remain zero-initialized. Numeric values coerce cleanly between `Float`,
/// `Int`, `Uint`, and `Bool` to match ClipMaker 3 / Shader+ dynamic property storage.
pub fn pack_uniforms(
    layout: &UniformBlockLayout,
    values: &BTreeMap<String, UniformValue>,
) -> Vec<u8> {
    let mut buf = vec![0u8; layout.byte_size as usize];

    for field in &layout.fields {
        let Some(value) = values.get(&field.name) else {
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
// Stage 1: Line Preprocessor (#include, #version, #extension, integer #define)
// ---------------------------------------------------------------------------

fn harvest_option_int_defines(defines: &BTreeMap<String, String>) -> HashMap<String, u32> {
    let mut map = HashMap::new();
    for (k, v) in defines {
        if let Ok(n) = v.trim().parse::<u32>() {
            map.insert(k.clone(), n);
        }
    }
    map
}

fn lookup_include<'a>(name: &str, options: &'a CompileOptions) -> Option<&'a str> {
    if let Some(custom) = options.extra_includes.get(name) {
        return Some(custom.as_str());
    }
    match name {
        "zoidium/gradient" => Some(INCLUDE_GRADIENT),
        "zoidium/curve" => Some(INCLUDE_CURVE),
        "zoidium/text" => Some(INCLUDE_TEXT),
        _ => None,
    }
}

fn parse_include_directive(trimmed: &str) -> Option<&str> {
    let rest = if let Some(after_comment) = trimmed.strip_prefix("//") {
        after_comment.trim_start().strip_prefix("#include")?
    } else {
        trimmed.strip_prefix("#include")?
    };

    let arg = rest.trim();
    if let Some(inner) = arg.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
        Some(inner.trim())
    } else if let Some(inner) = arg.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        Some(inner.trim())
    } else {
        None
    }
}

fn preprocess_lines(
    source: &str,
    options: &CompileOptions,
    included: &mut HashSet<String>,
    int_defines: &mut HashMap<String, u32>,
    depth: usize,
) -> Result<String, ShaderError> {
    let mut out = String::with_capacity(source.len());
    let mut in_block_comment = false;

    for raw_line in source.lines() {
        if in_block_comment {
            if let Some(end_idx) = raw_line.find("*/") {
                in_block_comment = false;
                let after = &raw_line[end_idx + 2..];
                if !after.trim().is_empty() {
                    out.push_str(after);
                    out.push('\n');
                }
            }
            continue;
        }

        let trimmed = raw_line.trim();
        if let Some(inc_name) = parse_include_directive(trimmed) {
            if depth >= MAX_INCLUDE_DEPTH {
                return Err(ShaderError::IncludeDepthExceeded(inc_name.to_string()));
            }
            let snippet = lookup_include(inc_name, options)
                .ok_or_else(|| ShaderError::UnknownInclude(inc_name.to_string()))?;
            if included.insert(inc_name.to_string()) {
                let expanded =
                    preprocess_lines(snippet, options, included, int_defines, depth + 1)?;
                out.push_str(&expanded);
                out.push('\n');
            }
            continue;
        }

        if let Some(after_hash) = trimmed.strip_prefix('#') {
            let directive = after_hash.trim_start();
            if directive.starts_with("version")
                && directive[7..]
                    .chars()
                    .next()
                    .is_none_or(char::is_whitespace)
            {
                continue;
            }
            if let Some(ext_rest) = directive.strip_prefix("extension")
                && ext_rest.chars().next().is_none_or(char::is_whitespace)
            {
                let ext_name = ext_rest.split(':').next().unwrap_or("").trim();
                if ext_name == "GL_OES_standard_derivatives" {
                    continue;
                }
                return Err(ShaderError::UnsupportedExtension(ext_name.to_string()));
            }
            if let Some(def_rest) = directive.strip_prefix("define")
                && def_rest.chars().next().is_some_and(char::is_whitespace)
            {
                let mut parts = def_rest.split_whitespace();
                if let (Some(name), Some(val)) = (parts.next(), parts.next())
                    && !name.contains('(')
                    && let Ok(n) = val.parse::<u32>()
                {
                    int_defines.insert(name.to_string(), n);
                }
            }
        }

        // Update block-comment state for subsequent lines while keeping line content intact.
        let bytes = raw_line.as_bytes();
        let mut i = 0;
        while i + 1 < bytes.len() {
            if !in_block_comment && bytes[i] == b'/' && bytes[i + 1] == b'/' {
                break;
            }
            if !in_block_comment && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                in_block_comment = true;
                i += 2;
                continue;
            }
            if in_block_comment && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                in_block_comment = false;
                i += 2;
                continue;
            }
            i += 1;
        }

        out.push_str(raw_line);
        out.push('\n');
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Stage 2: GLSL Lexer (comments stripped cleanly)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Ident(String),
    Number(String),
    Punct(String),
    Preprocessor(String),
    Newline,
}

fn tokenize(src: &str) -> Vec<Token> {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut tokens = Vec::new();
    let mut i = 0;
    let mut at_line_start = true;

    while i < len {
        let b = bytes[i];

        if b == b'\r' {
            i += 1;
            continue;
        }
        if b == b'\n' {
            tokens.push(Token::Newline);
            i += 1;
            at_line_start = true;
            continue;
        }
        if b == b' ' || b == b'\t' {
            i += 1;
            continue;
        }

        // Line comment
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Block comment
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < len && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                if bytes[i] == b'\n' {
                    tokens.push(Token::Newline);
                    at_line_start = true;
                }
                i += 1;
            }
            if i + 1 < len {
                i += 2;
            }
            continue;
        }

        // Preprocessor line (#define, #ifdef, #if, #elif, #else, #endif, #undef)
        if b == b'#' && at_line_start {
            let start = i;
            while i < len && bytes[i] != b'\n' {
                if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
                    break;
                }
                i += 1;
            }
            let directive = src[start..i].trim_end().to_string();
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            tokens.push(Token::Preprocessor(directive));
            at_line_start = false;
            continue;
        }

        at_line_start = false;

        // Identifier
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            i += 1;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            tokens.push(Token::Ident(src[start..i].to_string()));
            continue;
        }

        // Number (decimal, hex, float starting with digit or `.digit`)
        if b.is_ascii_digit() || (b == b'.' && i + 1 < len && bytes[i + 1].is_ascii_digit()) {
            let start = i;
            if b == b'0' && i + 1 < len && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
                i += 2;
                while i < len && bytes[i].is_ascii_hexdigit() {
                    i += 1;
                }
                if i < len && (bytes[i] == b'u' || bytes[i] == b'U') {
                    i += 1;
                }
            } else {
                while i < len && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i < len && bytes[i] == b'.' {
                    i += 1;
                    while i < len && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < len && (bytes[i] == b'e' || bytes[i] == b'E') {
                    i += 1;
                    if i < len && (bytes[i] == b'+' || bytes[i] == b'-') {
                        i += 1;
                    }
                    while i < len && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < len
                    && (bytes[i] == b'u'
                        || bytes[i] == b'U'
                        || bytes[i] == b'f'
                        || bytes[i] == b'F')
                {
                    i += 1;
                }
            }
            tokens.push(Token::Number(src[start..i].to_string()));
            continue;
        }

        // Multi-character operators
        if i + 2 < len {
            let tri = &src[i..i + 3];
            if tri == "<<=" || tri == ">>=" {
                tokens.push(Token::Punct(tri.to_string()));
                i += 3;
                continue;
            }
        }
        if i + 1 < len {
            let pair = &src[i..i + 2];
            if matches!(
                pair,
                "<=" | ">="
                    | "=="
                    | "!="
                    | "&&"
                    | "||"
                    | "^^"
                    | "+="
                    | "-="
                    | "*="
                    | "/="
                    | "%="
                    | "++"
                    | "--"
                    | "<<"
                    | ">>"
                    | "&="
                    | "|="
                    | "^="
            ) {
                tokens.push(Token::Punct(pair.to_string()));
                i += 2;
                continue;
            }
        }

        // Single-character punctuation
        let ch = src[i..].chars().next().unwrap_or(' ');
        tokens.push(Token::Punct(ch.to_string()));
        i += ch.len_utf8();
    }

    tokens
}

// ---------------------------------------------------------------------------
// Stage 3: Reject Unsupported Legacy WebGL 1 Built-ins
// ---------------------------------------------------------------------------

fn reject_unsupported_features(tokens: &[Token]) -> Result<(), ShaderError> {
    const FORBIDDEN_IDENTS: &[&str] = &[
        "texture2DLodEXT",
        "texture2DGradEXT",
        "texture2DProjLodEXT",
        "texture2DProjGradEXT",
        "textureCubeLodEXT",
        "textureCubeGradEXT",
        "gl_FragDepthEXT",
        "gl_FragData",
    ];

    for tok in tokens {
        if let Token::Ident(name) = tok
            && FORBIDDEN_IDENTS.contains(&name.as_str())
        {
            return Err(ShaderError::UnsupportedFeature(name.clone()));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 4: Extract Top-Level Uniforms, Varyings & Strip Precision Statements
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct UniformDecl {
    name: String,
    ty: UniformType,
    array_len: Option<u32>,
}

#[derive(Debug, Clone)]
struct VaryingDecl {
    ty: String,
    name: String,
}

struct ExtractedDecls {
    uniforms: Vec<UniformDecl>,
    samplers: Vec<String>,
    extra_varyings: Vec<VaryingDecl>,
    body_tokens: Vec<Token>,
}

fn is_precision_qual(s: &str) -> bool {
    matches!(s, "highp" | "mediump" | "lowp")
}

fn next_non_newline(tokens: &[Token], mut idx: usize) -> usize {
    while idx < tokens.len() && matches!(tokens[idx], Token::Newline) {
        idx += 1;
    }
    idx
}

fn extract_declarations(
    tokens: &[Token],
    int_defines: &HashMap<String, u32>,
) -> Result<ExtractedDecls, ShaderError> {
    let mut uniforms: Vec<UniformDecl> = Vec::new();
    let mut uniform_types: HashMap<String, (UniformType, Option<u32>)> = HashMap::new();
    let mut samplers: Vec<String> = Vec::new();
    let mut sampler_seen: HashSet<String> = HashSet::new();
    let mut extra_varyings: Vec<VaryingDecl> = Vec::new();
    let mut varying_seen: HashSet<String> = HashSet::new();
    let mut body_tokens: Vec<Token> = Vec::with_capacity(tokens.len());

    let mut brace_depth: usize = 0;
    let mut i = 0;

    while i < tokens.len() {
        match &tokens[i] {
            Token::Punct(p) if p == "{" => {
                brace_depth += 1;
                body_tokens.push(tokens[i].clone());
                i += 1;
            }
            Token::Punct(p) if p == "}" => {
                brace_depth = brace_depth.saturating_sub(1);
                body_tokens.push(tokens[i].clone());
                i += 1;
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "precision" => {
                // Strip `precision <qual> <type>;`
                while i < tokens.len() && !matches!(&tokens[i], Token::Punct(p) if p == ";") {
                    i += 1;
                }
                if i < tokens.len() {
                    i += 1;
                }
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "varying" => {
                i = parse_varying_decl(tokens, i + 1, &mut extra_varyings, &mut varying_seen)?;
            }
            Token::Ident(kw) if brace_depth == 0 && kw == "uniform" => {
                i = parse_uniform_decl(
                    tokens,
                    i + 1,
                    int_defines,
                    &mut uniforms,
                    &mut uniform_types,
                    &mut samplers,
                    &mut sampler_seen,
                )?;
            }
            _ => {
                body_tokens.push(tokens[i].clone());
                i += 1;
            }
        }
    }

    Ok(ExtractedDecls {
        uniforms,
        samplers,
        extra_varyings,
        body_tokens,
    })
}

fn parse_varying_decl(
    tokens: &[Token],
    mut i: usize,
    extra_varyings: &mut Vec<VaryingDecl>,
    varying_seen: &mut HashSet<String>,
) -> Result<usize, ShaderError> {
    i = next_non_newline(tokens, i);
    if let Some(Token::Ident(q)) = tokens.get(i)
        && is_precision_qual(q)
    {
        i = next_non_newline(tokens, i + 1);
    }

    let ty = match tokens.get(i) {
        Some(Token::Ident(t)) => t.clone(),
        _ => {
            return Err(ShaderError::InvalidUniformDeclaration(
                "expected type after varying".to_string(),
            ));
        }
    };
    i = next_non_newline(tokens, i + 1);

    while i < tokens.len() {
        let name = match tokens.get(i) {
            Some(Token::Ident(n)) => n.clone(),
            _ => {
                return Err(ShaderError::InvalidUniformDeclaration(
                    "expected identifier in varying declaration".to_string(),
                ));
            }
        };
        if name != "vUv" && name != "vUvScaled" && varying_seen.insert(name.clone()) {
            extra_varyings.push(VaryingDecl {
                ty: ty.clone(),
                name,
            });
        }
        i = next_non_newline(tokens, i + 1);
        match tokens.get(i) {
            Some(Token::Punct(p)) if p == "," => {
                i = next_non_newline(tokens, i + 1);
            }
            Some(Token::Punct(p)) if p == ";" => {
                return Ok(i + 1);
            }
            _ => {
                return Err(ShaderError::InvalidUniformDeclaration(
                    "expected ',' or ';' in varying declaration".to_string(),
                ));
            }
        }
    }

    Err(ShaderError::InvalidUniformDeclaration(
        "unterminated varying declaration".to_string(),
    ))
}

fn parse_uniform_decl(
    tokens: &[Token],
    mut i: usize,
    int_defines: &HashMap<String, u32>,
    uniforms: &mut Vec<UniformDecl>,
    uniform_types: &mut HashMap<String, (UniformType, Option<u32>)>,
    samplers: &mut Vec<String>,
    sampler_seen: &mut HashSet<String>,
) -> Result<usize, ShaderError> {
    i = next_non_newline(tokens, i);
    if let Some(Token::Ident(q)) = tokens.get(i)
        && is_precision_qual(q)
    {
        i = next_non_newline(tokens, i + 1);
    }

    let ty_ident = match tokens.get(i) {
        Some(Token::Ident(t)) => t.as_str(),
        _ => {
            return Err(ShaderError::InvalidUniformDeclaration(
                "expected type after uniform".to_string(),
            ));
        }
    };
    i = next_non_newline(tokens, i + 1);

    let is_sampler2d = ty_ident == "sampler2D";
    let uniform_ty = if is_sampler2d {
        None
    } else {
        Some(UniformType::from_glsl(ty_ident).ok_or_else(|| {
            ShaderError::InvalidUniformDeclaration(format!("unsupported uniform type `{ty_ident}`"))
        })?)
    };

    while i < tokens.len() {
        let name = match tokens.get(i) {
            Some(Token::Ident(n)) => n.clone(),
            _ => {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "expected uniform name after `{ty_ident}`"
                )));
            }
        };
        i = next_non_newline(tokens, i + 1);

        // Optional array suffix `[N]`
        let mut array_len = None;
        if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "[") {
            i = next_non_newline(tokens, i + 1);
            let len_val = match tokens.get(i) {
                Some(Token::Number(n)) => n.trim_end_matches(['u', 'U']).parse::<u32>().ok(),
                Some(Token::Ident(macro_name)) => int_defines.get(macro_name).copied(),
                _ => None,
            }
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "invalid array length on uniform `{name}`"
                ))
            })?;
            array_len = Some(len_val);
            i = next_non_newline(tokens, i + 1);
            if !matches!(tokens.get(i), Some(Token::Punct(p)) if p == "]") {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "expected ']' in array uniform `{name}`"
                )));
            }
            i = next_non_newline(tokens, i + 1);
        }

        // Skip optional non-standard initializer `= ...` up to `,` or `;`
        if matches!(tokens.get(i), Some(Token::Punct(p)) if p == "=") {
            i += 1;
            let mut paren_depth: usize = 0;
            while i < tokens.len() {
                match &tokens[i] {
                    Token::Punct(p) if p == "(" => paren_depth += 1,
                    Token::Punct(p) if p == ")" => paren_depth = paren_depth.saturating_sub(1),
                    Token::Punct(p) if paren_depth == 0 && (p == "," || p == ";") => break,
                    _ => {}
                }
                i += 1;
            }
        }

        if is_sampler2d {
            if array_len.is_some() {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "sampler2D arrays are not supported (`{name}`)"
                )));
            }
            if uniform_types.contains_key(&name) {
                return Err(ShaderError::DuplicateUniform { name });
            }
            if sampler_seen.insert(name.clone()) {
                samplers.push(name);
            }
        } else if let Some(ty) = uniform_ty {
            if sampler_seen.contains(&name) {
                return Err(ShaderError::DuplicateUniform { name });
            }
            if let Some(&(prev_ty, prev_len)) = uniform_types.get(&name) {
                if prev_ty != ty || prev_len != array_len {
                    return Err(ShaderError::DuplicateUniform { name });
                }
            } else {
                uniform_types.insert(name.clone(), (ty, array_len));
                uniforms.push(UniformDecl {
                    name,
                    ty,
                    array_len,
                });
            }
        }

        i = next_non_newline(tokens, i);
        match tokens.get(i) {
            Some(Token::Punct(p)) if p == "," => {
                i = next_non_newline(tokens, i + 1);
            }
            Some(Token::Punct(p)) if p == ";" => {
                return Ok(i + 1);
            }
            _ => {
                return Err(ShaderError::InvalidUniformDeclaration(
                    "expected ',' or ';' after uniform declaration".to_string(),
                ));
            }
        }
    }

    Err(ShaderError::InvalidUniformDeclaration(
        "unterminated uniform declaration".to_string(),
    ))
}

fn checked_align_up(value: u32, alignment: u32) -> Option<u32> {
    if alignment <= 1 {
        Some(value)
    } else {
        value.div_ceil(alignment).checked_mul(alignment)
    }
}

fn build_uniform_layout(decls: &[UniformDecl]) -> Result<UniformBlockLayout, ShaderError> {
    if decls.is_empty() {
        return Ok(UniformBlockLayout {
            set: 0,
            binding: 0,
            fields: Vec::new(),
            byte_size: 0,
        });
    }

    let mut fields = Vec::with_capacity(decls.len());
    let mut cursor: u32 = 0;

    for decl in decls {
        if let Some(len) = decl.array_len {
            if len == 0 || len > MAX_UNIFORM_ARRAY_LEN {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "uniform array `{}` length {len} exceeds allowed range 1..={MAX_UNIFORM_ARRAY_LEN}",
                    decl.name
                )));
            }
            let stride = decl.ty.array_stride();
            let offset = checked_align_up(cursor, 16).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform layout offset overflow at `{}`",
                    decl.name
                ))
            })?;
            let size = stride.checked_mul(len).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform array `{}` byte size overflow",
                    decl.name
                ))
            })?;
            cursor = offset.checked_add(size).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform layout size overflow at `{}`",
                    decl.name
                ))
            })?;
            if cursor > MAX_UNIFORM_BUFFER_BYTES {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "uniform block size exceeds {MAX_UNIFORM_BUFFER_BYTES} bytes at `{}`",
                    decl.name
                )));
            }
            fields.push(UniformField {
                name: decl.name.clone(),
                ty: decl.ty,
                array_len: Some(len),
                offset,
                size,
                array_stride: stride,
            });
        } else {
            let align = decl.ty.base_alignment();
            let size = decl.ty.byte_size();
            let offset = checked_align_up(cursor, align).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform layout offset overflow at `{}`",
                    decl.name
                ))
            })?;
            cursor = offset.checked_add(size).ok_or_else(|| {
                ShaderError::InvalidUniformDeclaration(format!(
                    "uniform layout size overflow at `{}`",
                    decl.name
                ))
            })?;
            if cursor > MAX_UNIFORM_BUFFER_BYTES {
                return Err(ShaderError::InvalidUniformDeclaration(format!(
                    "uniform block size exceeds {MAX_UNIFORM_BUFFER_BYTES} bytes at `{}`",
                    decl.name
                )));
            }
            fields.push(UniformField {
                name: decl.name.clone(),
                ty: decl.ty,
                array_len: None,
                offset,
                size,
                array_stride: 0,
            });
        }
    }

    let byte_size = checked_align_up(cursor, 16).ok_or_else(|| {
        ShaderError::InvalidUniformDeclaration("uniform block aligned size overflow".to_string())
    })?;
    if byte_size > MAX_UNIFORM_BUFFER_BYTES {
        return Err(ShaderError::InvalidUniformDeclaration(format!(
            "uniform block size {byte_size} exceeds {MAX_UNIFORM_BUFFER_BYTES} bytes"
        )));
    }

    Ok(UniformBlockLayout {
        set: 0,
        binding: 0,
        fields,
        byte_size,
    })
}

fn build_texture_bindings(samplers: &[String]) -> Vec<TextureBinding> {
    let mut ordered: Vec<&String> = Vec::with_capacity(samplers.len());
    if let Some(diffuse) = samplers.iter().find(|s| s.as_str() == "tDiffuse") {
        ordered.push(diffuse);
    }
    for s in samplers {
        if s != "tDiffuse" {
            ordered.push(s);
        }
    }

    // Binding 0 in set 0 is reserved for `ZcCustomUniforms`.
    let mut next_binding: u32 = 1;
    ordered
        .into_iter()
        .map(|name| {
            let texture_binding = next_binding;
            let sampler_binding = next_binding + 1;
            next_binding += 2;
            TextureBinding {
                name: name.clone(),
                group: 0,
                texture_binding,
                sampler_binding,
                is_diffuse: name == "tDiffuse",
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Stage 5: Rewrite Include Sampling Helpers, texture2D, gl_FragColor & Uniforms
// ---------------------------------------------------------------------------

type CallArgRanges = (Vec<(usize, usize)>, usize);

fn parse_call_arg_ranges(
    tokens: &[Token],
    open_paren_idx: usize,
) -> Result<Option<CallArgRanges>, ShaderError> {
    let mut args = Vec::new();
    let mut arg_start = open_paren_idx + 1;
    let mut depth: usize = 1;
    let mut i = open_paren_idx + 1;

    while i < tokens.len() {
        match &tokens[i] {
            Token::Punct(p) if p == "(" => {
                depth += 1;
                if depth > MAX_REWRITE_DEPTH {
                    return Err(ShaderError::GlslParse(format!(
                        "parenthesis nesting depth exceeds {MAX_REWRITE_DEPTH} limit"
                    )));
                }
            }
            Token::Punct(p) if p == ")" => {
                depth -= 1;
                if depth == 0 {
                    if arg_start < i || !args.is_empty() {
                        args.push((arg_start, i));
                    }
                    return Ok(Some((args, i + 1)));
                }
            }
            Token::Punct(p) if p == "," && depth == 1 => {
                args.push((arg_start, i));
                arg_start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    Ok(None)
}

fn expand_builtin_include_helpers(
    tokens: &[Token],
    included: &HashSet<String>,
    depth: usize,
    budget: &mut RewriteBudget,
) -> Result<Vec<Token>, ShaderError> {
    if depth > MAX_REWRITE_DEPTH {
        return Err(ShaderError::GlslParse(format!(
            "helper call nesting depth exceeds {MAX_REWRITE_DEPTH} limit"
        )));
    }
    let has_grad = included.contains("zoidium/gradient");
    let has_curve = included.contains("zoidium/curve");
    let has_text = included.contains("zoidium/text");
    if !has_grad && !has_curve && !has_text {
        return Ok(tokens.to_vec());
    }

    let mut out = Vec::with_capacity(tokens.len());
    let mut i = 0;

    while i < tokens.len() {
        if let Token::Ident(name) = &tokens[i] {
            let is_helper = (has_grad && name == "zSampleGradient")
                || (has_curve && (name == "zSampleCurve" || name == "zSampleCurveValue"))
                || (has_text && name == "zSampleText");
            let next_i = next_non_newline(tokens, i + 1);
            if is_helper
                && matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "(")
                && let Some((arg_ranges, after_call)) = parse_call_arg_ranges(tokens, next_i)?
            {
                let mut args = Vec::with_capacity(arg_ranges.len());
                for (lo, hi) in arg_ranges {
                    args.push(expand_builtin_include_helpers(
                        &tokens[lo..hi],
                        included,
                        depth + 1,
                        budget,
                    )?);
                }

                if has_grad && name == "zSampleGradient" && (args.len() == 2 || args.len() == 3) {
                    let before_len = out.len();
                    out.push(Token::Ident("texture2D".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[0].iter().cloned());
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("vec2".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    if args.len() == 3 {
                        out.push(Token::Ident("zGradientUv".to_string()));
                        out.push(Token::Punct("(".to_string()));
                        out.extend(args[1].iter().cloned());
                        out.push(Token::Punct(",".to_string()));
                        out.extend(args[2].iter().cloned());
                        out.push(Token::Punct(")".to_string()));
                    } else {
                        out.push(Token::Ident("zGradientUvClamp".to_string()));
                        out.push(Token::Punct("(".to_string()));
                        out.extend(args[1].iter().cloned());
                        out.push(Token::Punct(")".to_string()));
                    }
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Number("0.5".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }

                if has_curve
                    && (name == "zSampleCurve" || name == "zSampleCurveValue")
                    && args.len() == 2
                {
                    let before_len = out.len();
                    out.push(Token::Ident("texture2D".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[0].iter().cloned());
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("vec2".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Ident("zCurveUv".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[1].iter().cloned());
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Number("0.5".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    if name == "zSampleCurveValue" {
                        out.push(Token::Punct(".".to_string()));
                        out.push(Token::Ident("r".to_string()));
                    }
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }

                if has_text && name == "zSampleText" && args.len() == 2 {
                    let before_len = out.len();
                    out.push(Token::Ident("texture2D".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[0].iter().cloned());
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("clamp".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.extend(args[1].iter().cloned());
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("vec2".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Number("0.0".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident("vec2".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Number("1.0".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }
            }
        }

        budget.charge(1)?;
        out.push(tokens[i].clone());
        i += 1;
    }

    Ok(out)
}

fn is_glsl_builtin_type(ident: &str) -> bool {
    matches!(
        ident,
        "void"
            | "float"
            | "int"
            | "uint"
            | "bool"
            | "vec2"
            | "vec3"
            | "vec4"
            | "ivec2"
            | "ivec3"
            | "ivec4"
            | "uvec2"
            | "uvec3"
            | "uvec4"
            | "bvec2"
            | "bvec3"
            | "bvec4"
            | "mat2"
            | "mat3"
            | "mat4"
            | "sampler2D"
            | "sampler"
    )
}

fn first_single_ident(arg: &[Token]) -> Option<&str> {
    let mut found = None;
    for tok in arg {
        match tok {
            Token::Newline => {}
            Token::Ident(name) if found.is_none() => found = Some(name.as_str()),
            _ => return None,
        }
    }
    found
}

fn parse_bracket_expr(tokens: &[Token], open_bracket_idx: usize) -> Option<(Vec<Token>, usize)> {
    let mut inner = Vec::new();
    let mut depth: usize = 1;
    let mut i = open_bracket_idx + 1;

    while i < tokens.len() {
        match &tokens[i] {
            Token::Punct(p) if p == "[" => {
                depth += 1;
                inner.push(tokens[i].clone());
            }
            Token::Punct(p) if p == "]" => {
                depth -= 1;
                if depth == 0 {
                    return Some((inner, i + 1));
                }
                inner.push(tokens[i].clone());
            }
            _ => inner.push(tokens[i].clone()),
        }
        i += 1;
    }
    None
}

fn is_shadowed(name: &str, scopes: &[HashSet<String>]) -> bool {
    scopes.iter().rev().any(|scope| scope.contains(name))
}

fn rewrite_body(
    tokens: &[Token],
    uniforms: &HashMap<&str, &UniformDecl>,
    samplers: &HashSet<&str>,
    budget: &mut RewriteBudget,
) -> Result<Vec<Token>, ShaderError> {
    let mut scopes: Vec<HashSet<String>> = vec![HashSet::new()];
    let mut user_types: HashSet<String> = HashSet::new();
    rewrite_slice(
        tokens,
        uniforms,
        samplers,
        &mut scopes,
        &mut user_types,
        0,
        budget,
    )
}

fn rewrite_slice(
    tokens: &[Token],
    uniforms: &HashMap<&str, &UniformDecl>,
    samplers: &HashSet<&str>,
    scopes: &mut Vec<HashSet<String>>,
    user_types: &mut HashSet<String>,
    depth: usize,
    budget: &mut RewriteBudget,
) -> Result<Vec<Token>, ShaderError> {
    if depth > MAX_REWRITE_DEPTH {
        return Err(ShaderError::GlslParse(format!(
            "expression rewrite nesting depth exceeds {MAX_REWRITE_DEPTH} limit"
        )));
    }
    let mut out = Vec::with_capacity(tokens.len());
    let mut pending_scope_vars: Vec<String> = Vec::new();
    let mut decl_in_progress = false;
    let mut decl_expecting_name = false;
    let mut decl_paren_depth: usize = 0;
    let mut in_param_list = false;
    let mut param_paren_depth: usize = 0;
    let mut i = 0;

    while i < tokens.len() {
        budget.charge(1)?;
        let prev_is_dot = out
            .iter()
            .rev()
            .find(|t| !matches!(t, Token::Newline))
            .is_some_and(|t| matches!(t, Token::Punct(p) if p == "."));

        match &tokens[i] {
            Token::Preprocessor(line) => {
                out.push(Token::Preprocessor(rewrite_preprocessor_line(
                    line,
                    uniforms,
                    samplers,
                    depth + 1,
                    budget,
                )?));
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "{" => {
                let mut new_scope = HashSet::new();
                for var in pending_scope_vars.drain(..) {
                    new_scope.insert(var);
                }
                scopes.push(new_scope);
                decl_in_progress = false;
                decl_expecting_name = false;
                in_param_list = false;
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "}" => {
                if scopes.len() > 1 {
                    scopes.pop();
                }
                decl_in_progress = false;
                decl_expecting_name = false;
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == ";" => {
                decl_in_progress = false;
                decl_expecting_name = false;
                if !in_param_list {
                    pending_scope_vars.clear();
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "(" => {
                if in_param_list {
                    param_paren_depth += 1;
                } else if decl_in_progress {
                    decl_paren_depth += 1;
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == ")" => {
                if in_param_list {
                    param_paren_depth = param_paren_depth.saturating_sub(1);
                    if param_paren_depth == 0 {
                        in_param_list = false;
                    }
                } else if decl_in_progress {
                    decl_paren_depth = decl_paren_depth.saturating_sub(1);
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Punct(p) if p == "," && decl_in_progress && decl_paren_depth == 0 => {
                decl_expecting_name = true;
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Ident(name) if name == "struct" => {
                out.push(tokens[i].clone());
                let next_i = next_non_newline(tokens, i + 1);
                if let Some(Token::Ident(struct_name)) = tokens.get(next_i) {
                    user_types.insert(struct_name.clone());
                }
                i += 1;
                continue;
            }
            Token::Ident(name) if name == "for" => {
                out.push(tokens[i].clone());
                let next_i = next_non_newline(tokens, i + 1);
                if matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "(") {
                    in_param_list = true;
                    param_paren_depth = 0;
                }
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot && (is_glsl_builtin_type(name) || user_types.contains(name)) =>
            {
                let next_i = next_non_newline(tokens, i + 1);
                if let Some(Token::Ident(var_name)) = tokens.get(next_i) {
                    let after_name = next_non_newline(tokens, next_i + 1);
                    if matches!(tokens.get(after_name), Some(Token::Punct(p)) if p == "(") {
                        // Function declaration / definition: enter parameter list tracking.
                        in_param_list = true;
                        param_paren_depth = 0;
                        out.push(tokens[i].clone());
                        i += 1;
                        continue;
                    }
                    // Local variable, struct field, or function parameter declaration.
                    if in_param_list {
                        pending_scope_vars.push(var_name.clone());
                    } else if let Some(current_scope) = scopes.last_mut() {
                        current_scope.insert(var_name.clone());
                    }
                    out.push(tokens[i].clone());
                    out.push(Token::Ident(var_name.clone()));
                    decl_in_progress = !in_param_list;
                    decl_expecting_name = false;
                    decl_paren_depth = 0;
                    i = next_i + 1;
                    continue;
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Ident(name) if decl_expecting_name => {
                if let Some(current_scope) = scopes.last_mut() {
                    current_scope.insert(name.clone());
                }
                decl_expecting_name = false;
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Ident(name) if name == "gl_FragColor" && !prev_is_dot => {
                out.push(Token::Ident("_zc_frag_color".to_string()));
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && matches!(name.as_str(), "texture2D" | "texture2DProj" | "texture") =>
            {
                let next_i = next_non_newline(tokens, i + 1);
                if matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "(")
                    && let Some((arg_ranges, after_call)) = parse_call_arg_ranges(tokens, next_i)?
                    && arg_ranges.len() >= 2
                    && let Some(tex_ident) =
                        first_single_ident(&tokens[arg_ranges[0].0..arg_ranges[0].1])
                    && samplers.contains(tex_ident)
                {
                    let is_proj = name == "texture2DProj";
                    let fn_name = if is_proj { "textureProj" } else { "texture" };
                    let before_len = out.len();
                    out.push(Token::Ident(fn_name.to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Ident("sampler2D".to_string()));
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Ident(tex_ident.to_string()));
                    out.push(Token::Punct(",".to_string()));
                    out.push(Token::Ident(format!("{tex_ident}_sampler")));
                    out.push(Token::Punct(")".to_string()));
                    for (arg_idx, &(lo, hi)) in arg_ranges[1..].iter().enumerate() {
                        out.push(Token::Punct(",".to_string()));
                        let rewritten_arg = rewrite_slice(
                            &tokens[lo..hi],
                            uniforms,
                            samplers,
                            scopes,
                            user_types,
                            depth + 1,
                            budget,
                        )?;
                        if arg_idx == 0 {
                            // Convert logical CM3 bottom-left UV to WebGPU top-left texture UV at the sampling boundary.
                            let helper = if is_proj {
                                "_zc_flip_uv_proj"
                            } else {
                                "_zc_flip_uv"
                            };
                            out.push(Token::Ident(helper.to_string()));
                            out.push(Token::Punct("(".to_string()));
                            out.extend(rewritten_arg);
                            out.push(Token::Punct(")".to_string()));
                        } else {
                            out.extend(rewritten_arg);
                        }
                    }
                    out.push(Token::Punct(")".to_string()));
                    budget.charge(out.len().saturating_sub(before_len))?;
                    i = after_call;
                    continue;
                }
                out.push(tokens[i].clone());
                i += 1;
                continue;
            }
            Token::Ident(name)
                if !prev_is_dot
                    && !is_shadowed(name, scopes)
                    && uniforms.contains_key(name.as_str()) =>
            {
                let decl = uniforms[name.as_str()];
                if decl.array_len.is_some() {
                    let next_i = next_non_newline(tokens, i + 1);
                    if let Some(swizzle) = decl.ty.array_swizzle()
                        && matches!(tokens.get(next_i), Some(Token::Punct(p)) if p == "[")
                        && let Some((idx_tokens, after_bracket)) =
                            parse_bracket_expr(tokens, next_i)
                    {
                        let rewritten_idx = rewrite_slice(
                            &idx_tokens,
                            uniforms,
                            samplers,
                            scopes,
                            user_types,
                            depth + 1,
                            budget,
                        )?;
                        if decl.ty == UniformType::Bool {
                            out.push(Token::Punct("(".to_string()));
                        }
                        out.push(Token::Ident("_zc_u".to_string()));
                        out.push(Token::Punct(".".to_string()));
                        out.push(Token::Ident(decl.name.clone()));
                        out.push(Token::Punct("[".to_string()));
                        out.extend(rewritten_idx);
                        out.push(Token::Punct("]".to_string()));
                        out.push(Token::Punct(".".to_string()));
                        out.push(Token::Ident(swizzle.trim_start_matches('.').to_string()));
                        if decl.ty == UniformType::Bool {
                            out.push(Token::Punct("!=".to_string()));
                            out.push(Token::Number("0u".to_string()));
                            out.push(Token::Punct(")".to_string()));
                        }
                        i = after_bracket;
                        continue;
                    }
                    out.push(Token::Ident("_zc_u".to_string()));
                    out.push(Token::Punct(".".to_string()));
                    out.push(Token::Ident(decl.name.clone()));
                } else if decl.ty == UniformType::Bool {
                    out.push(Token::Punct("(".to_string()));
                    out.push(Token::Ident("_zc_u".to_string()));
                    out.push(Token::Punct(".".to_string()));
                    out.push(Token::Ident(decl.name.clone()));
                    out.push(Token::Punct("!=".to_string()));
                    out.push(Token::Number("0u".to_string()));
                    out.push(Token::Punct(")".to_string()));
                } else {
                    out.push(Token::Ident("_zc_u".to_string()));
                    out.push(Token::Punct(".".to_string()));
                    out.push(Token::Ident(decl.name.clone()));
                }
                i += 1;
                continue;
            }
            _ => {
                out.push(tokens[i].clone());
                i += 1;
            }
        }
    }

    Ok(out)
}

fn rewrite_preprocessor_line(
    line: &str,
    uniforms: &HashMap<&str, &UniformDecl>,
    samplers: &HashSet<&str>,
    depth: usize,
    budget: &mut RewriteBudget,
) -> Result<String, ShaderError> {
    let trimmed = line.trim();
    let Some(after_hash) = trimmed.strip_prefix('#') else {
        return Ok(line.to_string());
    };
    let directive = after_hash.trim_start();
    let Some(after_define) = directive.strip_prefix("define") else {
        return Ok(line.to_string());
    };
    if !after_define.starts_with(char::is_whitespace) {
        return Ok(line.to_string());
    }

    let rest = after_define.trim_start();
    let mut name_end = 0;
    for (idx, ch) in rest.char_indices() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            name_end = idx + ch.len_utf8();
        } else {
            break;
        }
    }
    if name_end == 0 {
        return Ok(line.to_string());
    }

    let macro_name = &rest[..name_end];
    let after_name = &rest[name_end..];
    let mut macro_params = HashSet::new();
    let (header_suffix, body) = if let Some(after_open) = after_name.strip_prefix('(') {
        if let Some(close_idx) = after_open.find(')') {
            let params_str = &after_open[..close_idx];
            for p in params_str.split(',') {
                let pt = p.trim();
                if !pt.is_empty() {
                    macro_params.insert(pt.to_string());
                }
            }
            (&after_name[..=close_idx + 1], &after_open[close_idx + 1..])
        } else {
            return Ok(line.to_string());
        }
    } else {
        ("", after_name)
    };

    if body.trim().is_empty() {
        return Ok(line.to_string());
    }

    let body_tokens = tokenize(body);
    let mut scopes = vec![macro_params];
    let mut user_types = HashSet::new();
    let rewritten = rewrite_slice(
        &body_tokens,
        uniforms,
        samplers,
        &mut scopes,
        &mut user_types,
        depth,
        budget,
    )?;
    let rendered = render_tokens(&rewritten).trim().to_string();
    Ok(format!("#define {macro_name}{header_suffix} {rendered}"))
}

fn render_tokens(tokens: &[Token]) -> String {
    let mut out = String::new();
    let mut prev: Option<&Token> = None;

    for tok in tokens {
        match tok {
            Token::Newline => {
                if !out.ends_with('\n') {
                    out.push('\n');
                }
                prev = Some(tok);
            }
            Token::Preprocessor(line) => {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str(line);
                out.push('\n');
                prev = Some(&Token::Newline);
            }
            Token::Ident(s) | Token::Number(s) => {
                if let Some(p) = prev {
                    match p {
                        Token::Ident(_) | Token::Number(_) => out.push(' '),
                        Token::Punct(ps) if !matches!(ps.as_str(), "." | "(" | "[" | "!" | "~") => {
                            out.push(' ');
                        }
                        _ => {}
                    }
                }
                out.push_str(s);
                prev = Some(tok);
            }
            Token::Punct(s) => {
                if let Some(p) = prev {
                    let need_space = match (p, s.as_str()) {
                        (Token::Newline, _) => false,
                        (_, "." | ";" | "," | ")" | "]" | "(" | "[") => false,
                        (Token::Punct(ps), _) if matches!(ps.as_str(), "." | "(" | "[") => false,
                        (Token::Punct(ps), cur)
                            if (ps == "-" && cur == "-")
                                || (ps == "+" && cur == "+")
                                || (ps == "/" && cur == "*") =>
                        {
                            true
                        }
                        (Token::Ident(_) | Token::Number(_), "++" | "--") => false,
                        _ => true,
                    };
                    if need_space {
                        out.push(' ');
                    }
                }
                out.push_str(s);
                prev = Some(tok);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Stage 6: Assemble Vulkan GLSL 450 Module
// ---------------------------------------------------------------------------

fn assemble_glsl450(
    options: &CompileOptions,
    extra_varyings: &[VaryingDecl],
    uniforms: &[UniformDecl],
    texture_bindings: &[TextureBinding],
    body_glsl: &str,
) -> String {
    let mut src = String::with_capacity(body_glsl.len() + 512);
    src.push_str("#version 450\n");
    src.push_str("#define GL_ES 1\n");
    src.push_str("#define GL_FRAGMENT_PRECISION_HIGH 1\n");

    for (name, value) in &options.defines {
        if value.is_empty() {
            src.push_str(&format!("#define {name}\n"));
        } else {
            src.push_str(&format!("#define {name} {value}\n"));
        }
    }

    src.push_str("\nlayout(location = 0) in vec2 vUv;\n");
    src.push_str("layout(location = 1) in vec2 vUvScaled;\n");
    for (idx, var) in extra_varyings.iter().enumerate() {
        let loc = idx + 2;
        src.push_str(&format!(
            "layout(location = {loc}) in {} {};\n",
            var.ty, var.name
        ));
    }
    src.push_str("layout(location = 0) out vec4 _zc_frag_color;\n");

    if !uniforms.is_empty() {
        src.push_str("\nlayout(set = 0, binding = 0, std140) uniform ZcCustomUniforms {\n");
        for u in uniforms {
            let glsl_ty = u.ty.ubo_glsl_type(u.array_len.is_some());
            if let Some(len) = u.array_len {
                src.push_str(&format!("    {glsl_ty} {}[{len}];\n", u.name));
            } else {
                src.push_str(&format!("    {glsl_ty} {};\n", u.name));
            }
        }
        src.push_str("} _zc_u;\n");
    }

    if !texture_bindings.is_empty() {
        src.push('\n');
        for tb in texture_bindings {
            src.push_str(&format!(
                "layout(set = {}, binding = {}) uniform texture2D {};\n",
                tb.group, tb.texture_binding, tb.name
            ));
            src.push_str(&format!(
                "layout(set = {}, binding = {}) uniform sampler {}_sampler;\n",
                tb.group, tb.sampler_binding, tb.name
            ));
        }
        src.push_str("vec2 _zc_flip_uv(vec2 uv) { return vec2(uv.x, 1.0 - uv.y); }\n");
        src.push_str(
            "vec3 _zc_flip_uv_proj(vec3 uvw) { return vec3(uvw.x, uvw.z - uvw.y, uvw.z); }\n",
        );
        src.push_str(
            "vec4 _zc_flip_uv_proj(vec4 uvw) { return vec4(uvw.x, uvw.w - uvw.y, uvw.z, uvw.w); }\n",
        );
    }

    src.push('\n');
    src.push_str(body_glsl.trim());
    src.push('\n');
    src
}
