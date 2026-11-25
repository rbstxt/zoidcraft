use super::*;

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
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    Bvec2,
    Bvec3,
    Bvec4,
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
            "bvec2" => Some(Self::Bvec2),
            "bvec3" => Some(Self::Bvec3),
            "bvec4" => Some(Self::Bvec4),
            "mat2" => Some(Self::Mat2),
            "mat3" => Some(Self::Mat3),
            "mat4" => Some(Self::Mat4),
            _ => None,
        }
    }

    pub(super) const fn glsl_name(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Int => "int",
            Self::Uint => "uint",
            Self::Bool => "bool",
            Self::Vec2 => "vec2",
            Self::Vec3 => "vec3",
            Self::Vec4 => "vec4",
            Self::Ivec2 => "ivec2",
            Self::Ivec3 => "ivec3",
            Self::Ivec4 => "ivec4",
            Self::Uvec2 => "uvec2",
            Self::Uvec3 => "uvec3",
            Self::Uvec4 => "uvec4",
            Self::Bvec2 => "bvec2",
            Self::Bvec3 => "bvec3",
            Self::Bvec4 => "bvec4",
            Self::Mat2 => "mat2",
            Self::Mat3 => "mat3",
            Self::Mat4 => "mat4",
        }
    }

    /// Base alignment in bytes under `std140` rules.
    pub const fn base_alignment(self) -> u32 {
        match self {
            Self::Float | Self::Int | Self::Uint | Self::Bool => 4,
            Self::Vec2 | Self::Ivec2 | Self::Uvec2 | Self::Bvec2 => 8,
            Self::Vec3
            | Self::Vec4
            | Self::Ivec3
            | Self::Ivec4
            | Self::Uvec3
            | Self::Uvec4
            | Self::Bvec3
            | Self::Bvec4
            | Self::Mat2
            | Self::Mat3
            | Self::Mat4 => 16,
        }
    }

    /// Size in bytes of a single non-array element under `std140` rules.
    pub const fn byte_size(self) -> u32 {
        match self {
            Self::Float | Self::Int | Self::Uint | Self::Bool => 4,
            Self::Vec2 | Self::Ivec2 | Self::Uvec2 | Self::Bvec2 => 8,
            Self::Vec3 | Self::Ivec3 | Self::Uvec3 | Self::Bvec3 => 12,
            Self::Vec4 | Self::Ivec4 | Self::Uvec4 | Self::Bvec4 => 16,
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
    /// - `bool`/`bvec*` are stored as `uint`/`uvec*` for WGSL uniform host-shareability.
    /// - `mat2` is stored as `vec4[2]` (or `vec4[2*N]` for arrays) because WGSL uniform
    ///   `mat2x2<f32>` cannot express `std140`'s 16-byte column stride.
    /// - Scalar and `vec2` arrays are stored as 4-component vectors so WGSL uniform arrays
    ///   keep their natural 16-byte stride without requiring unsupported `@stride` attributes.
    pub(super) const fn ubo_glsl_type(self, is_array: bool) -> &'static str {
        if is_array {
            match self {
                Self::Float | Self::Vec2 | Self::Vec4 | Self::Mat2 => "vec4",
                Self::Int | Self::Ivec2 | Self::Ivec4 => "ivec4",
                Self::Uint | Self::Bool | Self::Uvec2 | Self::Uvec4 | Self::Bvec2 | Self::Bvec4 => {
                    "uvec4"
                }
                Self::Vec3 => "vec3",
                Self::Ivec3 => "ivec3",
                Self::Uvec3 | Self::Bvec3 => "uvec3",
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
                Self::Vec4 | Self::Mat2 => "vec4",
                Self::Ivec2 => "ivec2",
                Self::Ivec3 => "ivec3",
                Self::Ivec4 => "ivec4",
                Self::Uvec2 | Self::Bvec2 => "uvec2",
                Self::Uvec3 | Self::Bvec3 => "uvec3",
                Self::Uvec4 | Self::Bvec4 => "uvec4",
                Self::Mat3 => "mat3",
                Self::Mat4 => "mat4",
            }
        }
    }

    pub(super) const fn array_swizzle(self) -> Option<&'static str> {
        match self {
            Self::Float | Self::Int | Self::Uint | Self::Bool => Some(".x"),
            Self::Vec2 | Self::Ivec2 | Self::Uvec2 | Self::Bvec2 => Some(".xy"),
            Self::Vec3
            | Self::Vec4
            | Self::Ivec3
            | Self::Ivec4
            | Self::Uvec3
            | Self::Uvec4
            | Self::Bvec3
            | Self::Bvec4
            | Self::Mat2
            | Self::Mat3
            | Self::Mat4 => None,
        }
    }

    pub(super) const fn needs_array_unpack(self) -> bool {
        matches!(
            self,
            Self::Float
                | Self::Int
                | Self::Uint
                | Self::Bool
                | Self::Vec2
                | Self::Ivec2
                | Self::Uvec2
                | Self::Bvec2
                | Self::Bvec3
                | Self::Bvec4
                | Self::Mat2
        )
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
    Bvec2([bool; 2]),
    Bvec3([bool; 3]),
    Bvec4([bool; 4]),
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

pub(super) struct RewriteBudget {
    tokens_emitted: usize,
}

impl RewriteBudget {
    const fn new() -> Self {
        Self { tokens_emitted: 0 }
    }

    pub(super) fn charge(&mut self, count: usize) -> Result<(), ShaderError> {
        self.tokens_emitted = self.tokens_emitted.saturating_add(count);
        if self.tokens_emitted > MAX_REWRITE_TOKENS {
            return Err(ShaderError::GlslParse(format!(
                "shader rewrite exceeded {MAX_REWRITE_TOKENS} token budget"
            )));
        }
        Ok(())
    }
}

pub(super) const TRANSLATION_CACHE_CAP: usize = 128;

struct TranslationCacheEntry {
    source: String,
    options: CompileOptions,
    compiled: CompiledCustomShader,
}

#[derive(Default)]
pub(super) struct TranslationCache {
    buckets: HashMap<u64, Vec<TranslationCacheEntry>>,
    order: VecDeque<u64>,
    pub(super) len: usize,
}

impl TranslationCache {
    fn key_hash(source: &str, options: &CompileOptions) -> u64 {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        options.hash(&mut hasher);
        hasher.finish()
    }

    pub(super) fn get(
        &self,
        source: &str,
        options: &CompileOptions,
    ) -> Option<CompiledCustomShader> {
        let hash = Self::key_hash(source, options);
        let bucket = self.buckets.get(&hash)?;
        bucket
            .iter()
            .find(|entry| entry.source == source && entry.options == *options)
            .map(|entry| entry.compiled.clone())
    }

    pub(super) fn insert(
        &mut self,
        source: &str,
        options: &CompileOptions,
        compiled: CompiledCustomShader,
    ) {
        let hash = Self::key_hash(source, options);
        if let Some(bucket) = self.buckets.get(&hash)
            && bucket
                .iter()
                .any(|entry| entry.source == source && entry.options == *options)
        {
            return;
        }
        if self.len >= TRANSLATION_CACHE_CAP
            && let Some(old_hash) = self.order.pop_front()
            && let Some(bucket) = self.buckets.get_mut(&old_hash)
            && !bucket.is_empty()
        {
            bucket.remove(0);
            if bucket.is_empty() {
                self.buckets.remove(&old_hash);
            }
            self.len -= 1;
        }
        self.buckets
            .entry(hash)
            .or_default()
            .push(TranslationCacheEntry {
                source: source.to_owned(),
                options: options.clone(),
                compiled,
            });
        self.order.push_back(hash);
        self.len += 1;
    }
}

static TRANSLATION_CACHE: OnceLock<Mutex<TranslationCache>> = OnceLock::new();

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

    let cache = TRANSLATION_CACHE.get_or_init(|| Mutex::new(TranslationCache::default()));
    {
        let guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(hit) = guard.get(source, options) {
            return Ok(hit);
        }
    }

    let compiled = translate_glsl_es100_with_options_uncached(source, options)?;
    {
        let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
        guard.insert(source, options, compiled.clone());
    }
    Ok(compiled)
}

fn translate_glsl_es100_with_options_uncached(
    source: &str,
    options: &CompileOptions,
) -> Result<CompiledCustomShader, ShaderError> {
    let preprocessed = preprocess_source(source, options)?;
    if preprocessed.source.len() > MAX_GLSL_SOURCE_BYTES {
        return Err(ShaderError::GlslParse(format!(
            "preprocessed shader ({} bytes) exceeds {} byte limit",
            preprocessed.source.len(),
            MAX_GLSL_SOURCE_BYTES
        )));
    }

    let tokens = tokenize(&preprocessed.source);
    if tokens.len() > MAX_REWRITE_TOKENS {
        return Err(ShaderError::GlslParse(format!(
            "shader token count ({}) exceeds {} limit",
            tokens.len(),
            MAX_REWRITE_TOKENS
        )));
    }
    reject_unsupported_features(&tokens)?;

    let mut extracted = extract_declarations(&tokens, &preprocessed.int_defines)?;

    let mut budget = RewriteBudget::new();
    let body_tokens = expand_builtin_include_helpers(
        &extracted.body_tokens,
        &preprocessed.included,
        0,
        &mut budget,
    )?;

    let uses_frag_coord = tokens_reference_ident(&body_tokens, "gl_FragCoord");
    if uses_frag_coord
        && !extracted.uniforms.iter().any(|u| u.name == "resolution")
        && !extracted.samplers.iter().any(|s| s == "resolution")
    {
        extracted.uniforms.push(UniformDecl {
            name: "resolution".to_string(),
            ty: UniformType::Vec2,
            array_len: None,
        });
    }

    let uniform_layout = build_uniform_layout(&extracted.uniforms)?;
    let texture_bindings = build_texture_bindings(&extracted.samplers);

    let uniform_map: HashMap<&str, &UniformDecl> = extracted
        .uniforms
        .iter()
        .map(|u| (u.name.as_str(), u))
        .collect();
    let sampler_set: HashSet<&str> = extracted.samplers.iter().map(String::as_str).collect();
    let (sampler_fns, user_fns) = collect_function_info(&body_tokens)?;
    let frag_coord_res_y = if uses_frag_coord {
        frag_coord_resolution_mode(&uniform_map)
    } else {
        FragCoordResY::None
    };

    let mut rewrite_state = RewriteState::default();
    let rewritten_tokens = strip_unreachable_after_return(rewrite_body(
        &body_tokens,
        &uniform_map,
        &extracted.struct_uniforms,
        &extracted.struct_defs,
        &extracted.const_ints,
        &sampler_set,
        &extracted.cube_samplers,
        &sampler_fns,
        &user_fns,
        frag_coord_res_y,
        &mut rewrite_state,
        &mut budget,
    )?);
    let body_glsl = render_tokens(&rewritten_tokens);

    let glsl450_source = assemble_glsl450(
        &extracted.extra_varyings,
        &extracted.uniforms,
        &texture_bindings,
        &extracted.cube_samplers,
        &rewrite_state,
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
        .map_err(|err| ShaderError::Validation(err.emit_to_string(&glsl450_source)))?;

    let raw_wgsl = wgsl::write_string(&module, &module_info, WriterFlags::empty())
        .map_err(|err| ShaderError::WgslEmit(err.to_string()))?;
    let wgsl = maybe_prepend_derivative_uniformity_diagnostic(raw_wgsl);

    Ok(CompiledCustomShader {
        wgsl,
        uniform_layout,
        texture_bindings,
        glsl450_source,
    })
}

fn wgsl_uses_implicit_derivatives(wgsl: &str) -> bool {
    const NEEDLES: &[&str] = &[
        "textureSample(",
        "textureSampleBias(",
        "dpdx(",
        "dpdxCoarse(",
        "dpdxFine(",
        "dpdy(",
        "dpdyCoarse(",
        "dpdyFine(",
        "fwidth(",
        "fwidthCoarse(",
        "fwidthFine(",
    ];
    NEEDLES.iter().any(|needle| wgsl.contains(needle))
}

fn maybe_prepend_derivative_uniformity_diagnostic(wgsl: String) -> String {
    if !wgsl_uses_implicit_derivatives(&wgsl)
        || wgsl.contains("diagnostic(off, derivative_uniformity)")
    {
        return wgsl;
    }
    // Insert right after any leading `enable ...;` directives so global directives stay valid.
    let mut insert_pos = 0;
    for line in wgsl.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with("enable ") || trimmed.is_empty() {
            insert_pos += line.len();
        } else {
            break;
        }
    }
    let mut out = String::with_capacity(wgsl.len() + 48);
    out.push_str(&wgsl[..insert_pos]);
    out.push_str("diagnostic(off, derivative_uniformity);\n");
    out.push_str(&wgsl[insert_pos..]);
    out
}
