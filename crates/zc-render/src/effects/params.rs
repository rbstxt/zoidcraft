//! Typed effect parameter values, GPU uniforms, and uniform packing/transform math.

use super::*;

/// Typed effect parameter value.
#[derive(Debug, Clone, PartialEq)]
pub enum EffectParamValue {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Int(i32),
    Bool(bool),
    String(String),
    Lut1D(Vec<[u8; 4]>),
    Lut3D { size: u32, data: Vec<u8> },
}

/// Evaluated parameter snapshot for a single post-processing effect invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectParams {
    pub values: BTreeMap<String, EffectParamValue>,
    pub frame: f32,
    pub time: f32,
    pub resolution: [f32; 2],
}

impl EffectParams {
    /// Creates an empty parameter set for a frame at `resolution`, `time` (seconds), and `frame`.
    #[must_use]
    pub fn new(resolution: [f32; 2], time: f32, frame: f32) -> Self {
        Self {
            values: BTreeMap::new(),
            frame,
            time,
            resolution,
        }
    }

    /// Builder-style parameter insertion.
    #[must_use]
    pub fn with_param(mut self, key: impl Into<String>, value: EffectParamValue) -> Self {
        self.values.insert(key.into(), value);
        self
    }

    /// Inserts or updates a parameter value.
    pub fn set(&mut self, key: impl Into<String>, value: EffectParamValue) {
        self.values.insert(key.into(), value);
    }

    /// Returns whether the effect is enabled (defaults to `true` when omitted).
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.bool_or("enabled", true)
    }

    /// Reads a scalar float parameter with fallback default.
    #[must_use]
    pub fn float_or(&self, key: &str, default: f32) -> f32 {
        match self.values.get(key) {
            Some(EffectParamValue::Float(v)) => *v,
            Some(EffectParamValue::Int(v)) => *v as f32,
            Some(EffectParamValue::Bool(v)) => {
                if *v {
                    1.0
                } else {
                    0.0
                }
            }
            _ => default,
        }
    }

    /// Reads an integer parameter with fallback default.
    #[must_use]
    pub fn int_or(&self, key: &str, default: i32) -> i32 {
        match self.values.get(key) {
            Some(EffectParamValue::Int(v)) => *v,
            Some(EffectParamValue::Float(v)) => v.round() as i32,
            Some(EffectParamValue::Bool(v)) => i32::from(*v),
            _ => default,
        }
    }

    /// Reads a boolean parameter with fallback default.
    #[must_use]
    pub fn bool_or(&self, key: &str, default: bool) -> bool {
        match self.values.get(key) {
            Some(EffectParamValue::Bool(v)) => *v,
            Some(EffectParamValue::Int(v)) => *v != 0,
            Some(EffectParamValue::Float(v)) => v.abs() > BYPASS_EPS,
            _ => default,
        }
    }

    /// Reads a `Vec2` parameter with fallback default.
    #[must_use]
    pub fn vec2_or(&self, key: &str, default: [f32; 2]) -> [f32; 2] {
        match self.values.get(key) {
            Some(EffectParamValue::Vec2(v)) => *v,
            Some(EffectParamValue::Float(v)) => [*v, *v],
            _ => default,
        }
    }

    /// Reads a `Vec3` parameter with fallback default.
    #[must_use]
    pub fn vec3_or(&self, key: &str, default: [f32; 3]) -> [f32; 3] {
        match self.values.get(key) {
            Some(EffectParamValue::Vec3(v)) => *v,
            Some(EffectParamValue::Vec4(v)) => [v[0], v[1], v[2]],
            Some(EffectParamValue::Float(v)) => [*v, *v, *v],
            _ => default,
        }
    }

    /// Reads a `Vec4` parameter with fallback default.
    #[must_use]
    pub fn vec4_or(&self, key: &str, default: [f32; 4]) -> [f32; 4] {
        match self.values.get(key) {
            Some(EffectParamValue::Vec4(v)) => *v,
            Some(EffectParamValue::Vec3(v)) => [v[0], v[1], v[2], 1.0],
            _ => default,
        }
    }

    /// Reads a string parameter if present.
    #[must_use]
    pub fn string(&self, key: &str) -> Option<&str> {
        match self.values.get(key) {
            Some(EffectParamValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Shared 256-byte uniform block matching WGSL `EffectUniforms` byte-for-byte.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Pod, Zeroable)]
pub struct EffectUniforms {
    pub resolution: [f32; 2],
    pub time: f32,
    pub frame: f32,
    pub mode: u32,
    pub flags: u32,
    pub pass: u32,
    pub extra: u32,
    pub p0: [f32; 4],
    pub p1: [f32; 4],
    pub p2: [f32; 4],
    pub p3: [f32; 4],
    pub p4: [f32; 4],
    pub p5: [f32; 4],
    pub p6: [f32; 4],
    pub p7: [f32; 4],
    pub c0: [f32; 4],
    pub c1: [f32; 4],
    pub c2: [f32; 4],
    pub c3: [f32; 4],
    pub m0: [f32; 4],
    pub m1: [f32; 4],
}

const _: () = assert!(size_of::<EffectUniforms>() == 256);

impl EffectUniforms {
    /// Initializes a zeroed uniform block populated with `resolution`, `time`, and `frame`.
    #[must_use]
    pub fn from_params(params: &EffectParams) -> Self {
        let mut u = Self::zeroed();
        u.resolution = [params.resolution[0].max(1.0), params.resolution[1].max(1.0)];
        u.time = params.time;
        u.frame = params.frame;
        u
    }
}

/// Input, output, and optional auxiliary texture views for an effect pass.
pub struct EffectTargets<'a> {
    pub input: &'a wgpu::TextureView,
    pub output: &'a wgpu::TextureView,
    pub aux: Option<&'a wgpu::TextureView>,
    pub aux2: Option<&'a wgpu::TextureView>,
}

/// Errors produced by effect lookup, validation, or execution.
#[derive(Debug, Error)]
pub enum EffectError {
    #[error("unknown effect id: `{0}`")]
    UnknownEffect(String),
    #[error("invalid render target dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    #[error("custom WGSL shader rejected: {0}")]
    CustomShaderInvalid(String),
}

/// Maps any of the 43 CM3 `PZ.layer.blendModes` names or indices to its WGSL blend mode code (`0..=42`).
#[must_use]
pub fn parse_blend_mode(value: Option<&EffectParamValue>) -> u32 {
    match value {
        Some(EffectParamValue::Int(i)) => (*i).clamp(0, 42) as u32,
        Some(EffectParamValue::Float(f)) => (f.round() as i32).clamp(0, 42) as u32,
        Some(EffectParamValue::String(s)) => match s.trim().to_ascii_uppercase().as_str() {
            "BLEND_SRC_OVER" | "SRC_OVER" | "NORMAL" => 0,
            "BLEND_MULTIPLY" | "MULTIPLY" => 1,
            "BLEND_SCREEN" | "SCREEN" => 2,
            "BLEND_OVERLAY" | "OVERLAY" => 3,
            "BLEND_DARKEN" | "DARKEN" => 4,
            "BLEND_LIGHTEN" | "LIGHTEN" => 5,
            "BLEND_COLORDODGE" | "BLEND_COLOR_DODGE" | "COLOR_DODGE" | "COLORDODGE" => 6,
            "BLEND_COLORBURN" | "BLEND_COLOR_BURN" | "COLOR_BURN" | "COLORBURN" => 7,
            "BLEND_HARDLIGHT" | "BLEND_HARD_LIGHT" | "HARD_LIGHT" | "HARDLIGHT" => 8,
            "BLEND_SOFTLIGHT" | "BLEND_SOFT_LIGHT" | "SOFT_LIGHT" | "SOFTLIGHT" => 9,
            "BLEND_DIFFERENCE" | "DIFFERENCE" => 10,
            "BLEND_EXCLUSION" | "EXCLUSION" => 11,
            "BLEND_HUE" | "HUE" => 12,
            "BLEND_SATURATION" | "SATURATION" => 13,
            "BLEND_COLOR" | "COLOR" => 14,
            "BLEND_LUMINOSITY" | "LUMINOSITY" => 15,
            "BLEND_PLUS" | "BLEND_ADD" | "ADD" | "PLUS" => 16,
            "BLEND_MINUS" | "BLEND_SUBTRACT" | "SUBTRACT" | "MINUS" => 17,
            "BLEND_LINEARDODGE" | "BLEND_LINEAR_DODGE" | "LINEARDODGE" | "LINEAR_DODGE" => 18,
            "BLEND_LINEARBURN" | "BLEND_LINEAR_BURN" | "LINEARBURN" | "LINEAR_BURN" => 19,
            "BLEND_VIVIDLIGHT" | "BLEND_VIVID_LIGHT" | "VIVIDLIGHT" | "VIVID_LIGHT" => 20,
            "BLEND_LINEARLIGHT" | "BLEND_LINEAR_LIGHT" | "LINEARLIGHT" | "LINEAR_LIGHT" => 21,
            "BLEND_PINLIGHT" | "BLEND_PIN_LIGHT" | "PINLIGHT" | "PIN_LIGHT" => 22,
            "BLEND_HARDMIX" | "BLEND_HARD_MIX" | "HARDMIX" | "HARD_MIX" => 23,
            "BLEND_INVERT" | "INVERT" => 24,
            "BLEND_INVERT_RGB" | "INVERT_RGB" => 25,
            "BLEND_PLUS_DARKER" | "PLUS_DARKER" => 26,
            "BLEND_CONTRAST" | "CONTRAST" => 27,
            "BLEND_INVERT_OVG" | "INVERT_OVG" => 28,
            "BLEND_RED" | "RED" => 29,
            "BLEND_GREEN" | "GREEN" => 30,
            "BLEND_BLUE" | "BLUE" => 31,
            "BLEND_ZERO" | "ZERO" => 32,
            "BLEND_SRC" | "SRC" => 33,
            "BLEND_DST" | "DST" => 34,
            "BLEND_DST_OVER" | "DST_OVER" => 35,
            "BLEND_SRC_IN" | "SRC_IN" => 36,
            "BLEND_DST_IN" | "DST_IN" => 37,
            "BLEND_SRC_OUT" | "SRC_OUT" => 38,
            "BLEND_DST_OUT" | "DST_OUT" => 39,
            "BLEND_SRC_ATOP" | "SRC_ATOP" => 40,
            "BLEND_DST_ATOP" | "DST_ATOP" => 41,
            "BLEND_XOR" | "XOR" => 42,
            _ => 0,
        },
        _ => 0,
    }
}

/// Evaluates Spec 04 §2.1 neutral auto-bypass rules.
///
/// Returns `true` when `enabled == false` OR the effect's documented neutral
/// parameter condition holds, allowing the compositor to skip the pass with
/// zero GPU copy.
#[must_use]
pub fn should_bypass(effect_id: &str, params: &EffectParams) -> bool {
    if !params.enabled() {
        return true;
    }
    match canonical_effect_id(effect_id) {
        "bloom" => params.float_or("amount", 1.0).abs() <= BYPASS_EPS,
        "boxblur" | "directionalblur" | "gaussianblur" => {
            params.float_or("delta", 1.0).abs() <= BYPASS_EPS
        }
        "colorize" | "edgedetection" => params.float_or("amount", 1.0).abs() <= BYPASS_EPS,
        "cubelut" => {
            !matches!(
                params.values.get("lut"),
                Some(EffectParamValue::Lut1D(entries)) if !entries.is_empty()
            ) && !matches!(
                params.values.get("lut"),
                Some(EffectParamValue::Lut3D { size, data }) if *size >= 2 && !data.is_empty()
            )
        }
        "fisheye" => params.float_or("amount", 25.0).abs() <= BYPASS_EPS,
        "glitch" => params.float_or("amount", 0.4).abs() <= BYPASS_EPS,
        "grayscale" => (params.float_or("saturation", 1.0) - 1.0).abs() <= BYPASS_EPS,
        "hueshift" => params.float_or("amount", 0.5).abs() <= BYPASS_EPS,
        "pixelated" => {
            let sz = params.vec2_or("size", [16.0, 9.0]);
            (sz[0] - 1.0).abs() <= BYPASS_EPS && (sz[1] - 1.0).abs() <= BYPASS_EPS
        }
        "pulse" => {
            let t = params.float_or("time", 0.0);
            let frac = t.rem_euclid(1.0);
            frac <= BYPASS_EPS || (1.0 - frac) <= BYPASS_EPS
        }
        "rgbshift" => params.float_or("amount", 10.0).abs() <= BYPASS_EPS,
        "scanlines" | "shutter" => params.float_or("opacity", 1.0).abs() <= BYPASS_EPS,
        "static" => params.float_or("amount", 0.5).abs() <= BYPASS_EPS,
        "swirl" => params.float_or("angle", 0.2).abs() <= BYPASS_EPS,
        "wavy" => params.float_or("amount", 3.0).abs() <= BYPASS_EPS,
        // Plugin neutral bypasses (radialblurspin does NOT bypass on density == 0 in Zoidium)
        "smartmotionblur" => {
            let amt = if params.values.contains_key("amount") {
                params.float_or("amount", 0.5)
            } else {
                params.float_or("blurAmount", 0.5)
            };
            let sens = if params.values.contains_key("sensitivity") {
                params.float_or("sensitivity", 70.0)
            } else {
                params.float_or("motionSensitivity", 70.0)
            };
            amt.abs() <= BYPASS_EPS || sens.abs() <= BYPASS_EPS
        }
        _ => false,
    }
}

/// Packs `EffectUniforms` for any of the 44 standard effects or 12 plugin effects.
#[must_use]
pub fn pack_effect_uniforms(
    effect_id: &str,
    params: &EffectParams,
    has_aux: bool,
    has_aux2: bool,
) -> EffectUniforms {
    let mut u = EffectUniforms::from_params(params);
    let res_scale =
        if let Some(EffectParamValue::Vec2(seq)) = params.values.get("_sequenceResolution") {
            [
                (params.resolution[0].max(1.0) / seq[0].max(1.0)).max(1e-3),
                (params.resolution[1].max(1.0) / seq[1].max(1.0)).max(1e-3),
            ]
        } else {
            [1.0, 1.0]
        };
    match canonical_effect_id(effect_id) {
        "brightnesscontrast" => {
            u.p0 = [
                params.float_or("brightness", 0.0),
                params.float_or("contrast", 1.0),
                0.0,
                0.0,
            ];
        }
        "chromakey" => {
            let bg = params.vec3_or("backgroundColor", [0.157, 0.776, 0.129]);
            let w = params.vec3_or("weights", [4.0, 1.0, 2.0]);
            u.c0 = [bg[0], bg[1], bg[2], 1.0];
            u.p0 = [w[0], w[1], w[2], params.float_or("soften", 0.5)];
            let inv = u32::from(params.bool_or("invert", false));
            let msk = u32::from(params.bool_or("mask", false));
            u.flags = inv | (msk << 1);
        }
        "colorcorrection" => {
            let pw = params.vec3_or("powRGB", [1.0, 1.0, 1.0]);
            let ml = params.vec3_or("mulRGB", [0.8, 0.8, 0.8]);
            let ad = params.vec3_or("addRGB", [0.08, 0.08, 0.08]);
            u.p0 = [pw[0], pw[1], pw[2], 0.0];
            u.p1 = [ml[0], ml[1], ml[2], 0.0];
            u.p2 = [ad[0], ad[1], ad[2], 0.0];
        }
        "colorgrading" => {
            let sh = params.vec3_or("shadows", [1.0, 1.0, 1.0]);
            let md = params.vec3_or("midtones", [1.0, 1.0, 1.0]);
            let hi = params.vec3_or("highlights", [1.0, 1.0, 1.0]);
            u.c0 = [sh[0], sh[1], sh[2], 1.0];
            u.c1 = [md[0], md[1], md[2], 1.0];
            u.c2 = [hi[0], hi[1], hi[2], 1.0];
        }
        "colorize" => {
            let col = params.vec3_or("color", [1.0, 0.0, 0.0]);
            u.c0 = [col[0], col[1], col[2], 1.0];
            u.p0 = [params.float_or("amount", 1.0), 0.0, 0.0, 0.0];
        }
        "coloroverlay" => {
            let col = params.vec3_or("color", [1.0, 1.0, 1.0]);
            u.c0 = [col[0], col[1], col[2], 1.0];
            u.p0 = [params.float_or("opacity", 1.0), 0.0, 0.0, 0.0];
            u.mode = parse_blend_mode(params.values.get("blending"));
        }
        "cubelut" => {
            let dmin = params.vec3_or("domainMin", [0.0, 0.0, 0.0]);
            let dmax = params.vec3_or("domainMax", [1.0, 1.0, 1.0]);
            u.p0 = [dmin[0], dmin[1], dmin[2], 0.0];
            u.p1 = [dmax[0], dmax[1], dmax[2], 0.0];
            match params.values.get("lut") {
                Some(EffectParamValue::Lut1D(entries)) => {
                    u.mode = 0;
                    u.p2 = [entries.len().max(2) as f32, 0.0, 0.0, 0.0];
                }
                Some(EffectParamValue::Lut3D { size, .. }) => {
                    u.mode = 1;
                    u.p2 = [(*size).max(2) as f32, 0.0, 0.0, 0.0];
                }
                _ => {
                    u.mode = 0;
                    u.p2 = [2.0, 0.0, 0.0, 0.0];
                }
            }
        }
        "exposure" => {
            u.p0 = [
                params.float_or("exposure", 0.0),
                params.float_or("gamma", 0.0),
                0.0,
                0.0,
            ];
        }
        "gradientoverlay" => {
            let off = params.vec2_or("offset", [0.0, 0.0]);
            let sc = params.vec2_or("scale", [1.0, 1.0]);
            let rot = params.float_or("rotation", 0.0);
            let (m0, m1) = compute_gradient_uv_transform(off, sc, rot, u.resolution);
            u.mode = parse_blend_mode(params.values.get("blending"));
            u.extra = params.int_or("gradientType", 0).clamp(0, 4) as u32;
            u.p0 = [off[0], off[1], sc[0], sc[1]];
            u.p1 = [rot, params.float_or("opacity", 1.0), 0.0, 0.0];
            u.m0 = m0;
            u.m1 = m1;
        }
        "grayscale" => {
            u.p0 = [params.float_or("saturation", 1.0), 0.0, 0.0, 0.0];
        }
        "hueshift" => {
            u.p0 = [params.float_or("amount", 0.5), 0.0, 0.0, 0.0];
        }
        "lumakey" => {
            u.p0 = [
                params.float_or("threshold", 0.5),
                params.float_or("soften", 0.1),
                0.0,
                0.0,
            ];
            let inv = u32::from(params.bool_or("invert", false));
            let msk = u32::from(params.bool_or("mask", false));
            u.flags = inv | (msk << 1);
        }
        "mask" => {
            let pscale = params.vec2_or("_previewScale", [1.0, 1.0]);
            let pos = params.vec2_or("position", [0.0, 0.0]);
            let sz = params.vec2_or("size", [250.0, 250.0]);
            u.mode = params.int_or("mode", 0).clamp(0, 1) as u32;
            u.flags = u32::from(params.bool_or("invert", false));
            u.p0 = [
                pos[0] * pscale[0],
                pos[1] * pscale[1],
                sz[0] * pscale[0],
                sz[1] * pscale[1],
            ];
            u.p1 = [
                params.float_or("rotation", 0.0),
                params.float_or("feather", 0.1),
                params.float_or("opacity", 1.0),
                0.0,
            ];
        }
        "negative" | "technicolor" | "fxaa" => {}
        "overlay" => {
            let off = params.vec2_or("offset", [0.0, 0.0]);
            let sc = params.vec2_or("scale", [1.0, 1.0]);
            let rot = params.float_or("rotation", 0.0);
            let (m0, m1) = compute_overlay_uv_transform(off, sc, rot, u.resolution);
            u.mode = parse_blend_mode(params.values.get("blending"));
            let rep = u32::from(params.bool_or("repeat", true));
            u.flags = rep | (u32::from(has_aux) << 1);
            u.p0 = [off[0], off[1], sc[0], sc[1]];
            u.p1 = [rot, params.float_or("opacity", 1.0), 0.0, 0.0];
            u.m0 = m0;
            u.m1 = m1;
        }
        "posterize" => {
            u.p0 = [
                params.float_or("gamma", 0.6),
                params.float_or("regions", 8.0),
                0.0,
                0.0,
            ];
        }
        "directionalblur" => {
            u.p0 = [
                params.float_or("delta", 1.0),
                params.float_or("direction", 0.0),
                0.0,
                0.0,
            ];
        }
        "edgedetection" => {
            u.p0 = [params.float_or("amount", 1.0), 0.0, 0.0, 0.0];
        }
        "glitch" => {
            u.p0 = [
                params.float_or("time", params.time),
                params.float_or("amount", 0.4),
                params.float_or("offset", 0.1),
                params.float_or("speed", 4.0),
            ];
        }
        "radialblur" => {
            let center = params.vec2_or("center", [0.0, 0.0]);
            u.p0 = [
                params.float_or("density", 0.5),
                params.float_or("dither", 0.5),
                params.float_or("decay", 0.9),
                params.float_or("weight", 0.2),
            ];
            u.p1 = [center[0], center[1], 0.0, 0.0];
            u.flags = u32::from(params.bool_or("overbright", false));
        }
        "rgbshift" => {
            u.p0 = [
                params.float_or("amount", 10.0),
                params.float_or("angle", 0.0),
                0.0,
                0.0,
            ];
        }
        "scanlines" => {
            let col = params.vec3_or("color", [0.0, 0.0, 0.0]);
            u.c0 = [col[0], col[1], col[2], 1.0];
            u.p0 = [
                params.float_or("opacity", 1.0),
                params.float_or("thickness", 1.0),
                params.float_or("offset", 0.0),
                params.float_or("feather", 0.25),
            ];
            u.p1 = [params.float_or("angle", FRAC_PI_2), 0.0, 0.0, 0.0];
        }
        "sharpen" => {
            u.p0 = [params.float_or("delta", 1.0), 0.0, 0.0, 0.0];
        }
        "shutter" => {
            let col = params.vec3_or("color", [0.0, 0.0, 0.0]);
            u.mode = params.int_or("mode", 1).clamp(0, 2) as u32;
            u.c0 = [col[0], col[1], col[2], 1.0];
            u.p0 = [
                params.float_or("opacity", 1.0),
                params.float_or("covered", 0.25),
                params.float_or("angle", FRAC_PI_2),
                params.float_or("fade", 0.001),
            ];
        }
        "static" => {
            u.mode = params.int_or("blending", 1).clamp(0, 3) as u32;
            u.p0 = [
                params.float_or("size", 1.0),
                params.float_or("amount", 0.5),
                0.0,
                0.0,
            ];
        }
        "vignette" => {
            let col = params.vec3_or("color", [0.0, 0.0, 0.0]);
            let center = params.vec2_or("center", [0.5, 0.5]);
            u.c0 = [col[0], col[1], col[2], 1.0];
            let has_cm3_params = (params.values.contains_key("offset")
                || params.values.contains_key("darkness"))
                && !params.values.contains_key("radius")
                && !params.values.contains_key("strength")
                && !params.values.contains_key("amount");
            if has_cm3_params {
                u.mode = 1;
                u.p0 = [
                    center[0],
                    center[1],
                    params.float_or("offset", 1.0).max(0.0),
                    params.float_or("darkness", 1.0).clamp(0.0, 4.0),
                ];
            } else {
                let radius = params
                    .float_or("radius", params.float_or("size", 0.75))
                    .clamp(0.0, 2.0);
                let softness = params
                    .float_or(
                        "softness",
                        params.float_or("feather", params.float_or("falloff", 0.45)),
                    )
                    .max(1e-3);
                let strength = params
                    .float_or(
                        "strength",
                        params.float_or("amount", params.float_or("opacity", 0.65)),
                    )
                    .clamp(0.0, 1.0);
                let aspect_mul = params
                    .float_or("roundness", params.float_or("aspect", 1.0))
                    .clamp(0.05, 4.0);
                u.mode = 0;
                u.p0 = [center[0], center[1], radius, softness];
                u.p1 = [strength, aspect_mul, 0.0, 0.0];
            }
        }
        "crop" => {
            u.p0 = [
                params.float_or("left", 0.0),
                params.float_or("top", 0.0),
                params.float_or("right", 0.0),
                params.float_or("bottom", 0.0),
            ];
        }
        "displacementmap" => {
            let doff = params.vec2_or("displacementOffset", [0.0, 0.0]);
            let off = params.vec2_or("offset", [0.0, 0.0]);
            let sc = params.vec2_or("scale", [1.0, 1.0]);
            let rot = params.float_or("rotation", 0.0);
            let ud = params.vec3_or("uDisplacement", [1.0, 0.0, 0.0]);
            let vd = params.vec3_or("vDisplacement", [1.0, 0.0, 0.0]);
            let (m0, m1) = compute_overlay_uv_transform(off, sc, rot, u.resolution);
            u.mode = params.int_or("wrap", 2).clamp(0, 2) as u32;
            u.flags = u32::from(has_aux);
            u.p0 = [doff[0], doff[1], params.float_or("amount", 0.1), rot];
            u.p1 = [off[0], off[1], sc[0], sc[1]];
            u.p2 = [ud[0], ud[1], ud[2], 0.0];
            u.p3 = [vd[0], vd[1], vd[2], 0.0];
            u.m0 = m0;
            u.m1 = m1;
        }
        "duplicate" => {
            let off = params.vec2_or("offset", [0.0, 0.0]);
            let mult = params.vec2_or("multiplier", [3.0, 3.0]);
            let rot = params.float_or("rotation", 0.0);
            let (m0, m1) = compute_duplicate_uv_transform(off, mult, rot, u.resolution);
            u.mode = (params.int_or("repeat", 0).clamp(0, 1) as u32) + 1;
            u.p0 = [off[0], off[1], mult[0], mult[1]];
            u.p1 = [rot, 0.0, 0.0, 0.0];
            u.m0 = m0;
            u.m1 = m1;
        }
        "fisheye" => {
            u.p0 = [params.float_or("amount", 25.0), 0.0, 0.0, 0.0];
        }
        "flip" => {
            let h = u32::from(params.bool_or("horizontal", true));
            let v = u32::from(params.bool_or("vertical", false));
            u.flags = h | (v << 1);
        }
        "mirror" => {
            u.mode = params.int_or("mode", 0).clamp(0, 7) as u32;
        }
        "pixelated" => {
            let sz = params.vec2_or("size", [16.0, 9.0]);
            u.p0 = [
                (sz[0] * res_scale[0]).max(1.0),
                (sz[1] * res_scale[1]).max(1.0),
                0.0,
                0.0,
            ];
        }
        "pixelsort" => {
            let dir = u32::from(params.int_or("direction", 0) != 0);
            let rev = u32::from(params.bool_or("reverse", false));
            let aff = u32::from(params.int_or("affect", 0) != 0);
            u.flags = dir | (rev << 1) | (aff << 2);
            u.p0 = [params.float_or("threshold", 1.0), 0.0, 0.0, 0.0];
        }
        "pulse" => {
            u.p0 = [
                params.float_or("time", 0.0),
                params.float_or("amplitude", 300.0),
                params.float_or("size", 0.5),
                0.0,
            ];
        }
        "shake" => {
            u.mode = params.int_or("wrap", 2).clamp(0, 2) as u32;
            if let Some((h_inv, depth_plane)) = compute_shake_inverse_homography(params) {
                u.flags = 1;
                u.p0 = [h_inv[0][0], h_inv[0][1], h_inv[0][2], 0.0];
                u.p1 = [h_inv[1][0], h_inv[1][1], h_inv[1][2], 0.0];
                u.p2 = [h_inv[2][0], h_inv[2][1], h_inv[2][2], 0.0];
                u.p4 = depth_plane;
            } else {
                u.flags = 0;
            }
        }
        "swirl" => {
            u.p0 = [params.float_or("angle", 0.2), 0.0, 0.0, 0.0];
        }
        "transform" => {
            let uv_scale = params.vec2_or("uvScale", [1.0, 1.0]);
            // CM3 transform.js: 0 = tile -> wrap_uv 1u, 1 = reflect -> wrap_uv 2u
            let wrap_idx = match params.values.get("wrap") {
                Some(EffectParamValue::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
                    "reflect" | "mirror" | "mirrored" | "1" => 1u32,
                    _ => 0u32,
                },
                _ => params.int_or("wrap", 0).clamp(0, 1) as u32,
            };
            u.mode = wrap_idx + 1;
            u.p3 = [uv_scale[0], uv_scale[1], 0.0, 0.0];
            if let Some((h_inv, depth_plane)) = compute_transform_inverse_homography(params) {
                u.flags = 1;
                u.p0 = [h_inv[0][0], h_inv[0][1], h_inv[0][2], 0.0];
                u.p1 = [h_inv[1][0], h_inv[1][1], h_inv[1][2], 0.0];
                u.p2 = [h_inv[2][0], h_inv[2][1], h_inv[2][2], 0.0];
                u.p4 = depth_plane;
            } else {
                u.flags = 0;
            }
        }
        "wavy" => {
            u.p0 = [
                params.float_or("amount", 3.0),
                params.float_or("size", 25.0),
                params.float_or("time", 0.0),
                params.float_or("angle", 0.0),
            ];
        }
        // Zoidium plugin GPU effects
        "colorcurves" => {
            let (has_lut, lut_size) = match params.values.get("lut") {
                Some(EffectParamValue::Lut1D(entries)) if !entries.is_empty() => {
                    (true, entries.len().max(2) as f32)
                }
                _ => (
                    has_aux || params.string("curves").is_some(),
                    params.float_or("lutSize", 256.0),
                ),
            };
            let affect_alpha = u32::from(params.bool_or("affectAlpha", false));
            u.flags = affect_alpha | (u32::from(has_lut) << 1);
            u.p0 = [lut_size, params.float_or("mix", 1.0), 0.0, 0.0];
        }
        "radialblurspin" => {
            let center = params.vec2_or("center", [0.5, 0.5]);
            let density = if params.values.contains_key("density") {
                params.float_or("density", 0.5)
            } else if params.values.contains_key("amount") {
                params.float_or("amount", 10.0) * 0.05
            } else if params.values.contains_key("angle") {
                params.float_or("angle", 10.0) * 0.05
            } else if params.values.contains_key("strength") {
                params.float_or("strength", 0.5)
            } else {
                0.5
            };
            let dither = params.float_or("dither", 0.5);
            let decay = params.float_or("decay", 0.9);
            let weight = params.float_or("weight", 0.2);
            u.flags = u32::from(params.bool_or("overbright", false));
            u.p0 = [center[0], center[1], density, dither];
            u.p1 = [decay, weight, 0.0, 0.0];
        }
        "vhs" => {
            // Supports ALL 29 Zoidium vhs-shaders.js parameters + legacy aliases
            let amount = params.float_or("amount", 1.0);
            let persistence = params.float_or("persistence", 0.35);
            let signal_str = if params.values.contains_key("signalStrength") {
                params.float_or("signalStrength", 0.0)
            } else {
                params.float_or("chromaBleed", 0.0)
            };
            let color_stripes = params.float_or("colorStripes", 0.0);
            let chroma_crawl = params.float_or("chromaCrawl", 0.0);
            let chroma_loss = if params.values.contains_key("chromaLoss") {
                params.float_or("chromaLoss", 0.0)
            } else {
                params.float_or("chromaBleed", 0.0) * 0.5
            };
            let flicker = params.float_or("flicker", 0.0);
            let vhs_sharpen = params.float_or("vhsSharpen", 0.0);
            let interlace = params.float_or("interlace", 0.3);
            let static_warp = if params.values.contains_key("staticWarp") {
                params.float_or("staticWarp", 0.0)
            } else {
                params.float_or("noise", 0.15)
            };
            let fast_forward = params.float_or("fastForward", 0.0);
            let tracking = if params.values.contains_key("tracking") {
                params.float_or("tracking", 0.0)
            } else if params.values.contains_key("headSwitch") {
                params.float_or("headSwitch", 0.25) * 0.5
            } else {
                0.0
            };
            let aniso_warp = params.float_or("anisotropyWarp", 0.0);
            let rgb_split = params.float_or("rgbSplit", 0.25);
            let tear = params.float_or("tear", 0.0);
            let hum_bar = params.float_or("humBar", 0.0);
            let blue_screen = params.float_or("blueScreen", 0.0);
            let skew = params.float_or("skew", 0.0);
            let crease = params.float_or("crease", 0.0);
            let ghost = params.float_or("ghost", 0.0);
            let osd = if params.values.contains_key("osd") {
                params.float_or("osd", 0.0)
            } else if params.bool_or("showTimestamp", false) {
                1.0
            } else {
                0.0
            };
            let noise_evo = params.float_or("noiseEvo", 0.35);
            let noise_op = params.float_or("noiseOpacity", 0.75);
            let blinds = if params.values.contains_key("blinds") {
                params.float_or("blinds", 0.0)
            } else {
                params.float_or("scanlines", 0.0)
            };
            let snow_dots = params.float_or("snowDots", 0.0);
            let speed_dots = params.float_or("speedDots", 0.0);
            let aniso_snow = params.float_or("anisoSnow", 0.0);
            let jitter_x = if params.values.contains_key("jitterX") {
                params.float_or("jitterX", 0.0)
            } else if params.values.contains_key("trackingJitter") {
                params.float_or("trackingJitter", 0.2)
            } else {
                params.float_or("jitter", 0.0)
            };
            let jitter_y = params.float_or("jitterY", 0.0);
            let roll = params.float_or("roll", 0.0);
            let roll_speed = params.float_or("rollSpeed", 0.5);
            let glitch_seed = if params.values.contains_key("glitchSeed") {
                params.float_or("glitchSeed", 0.0)
            } else {
                params.float_or("seed", 0.0)
            };

            u.flags = u32::from(has_aux);
            u.p0 = [amount, persistence, signal_str, color_stripes];
            u.p1 = [chroma_crawl, chroma_loss, flicker, vhs_sharpen];
            u.p2 = [interlace, static_warp, fast_forward, tracking];
            u.p3 = [aniso_warp, rgb_split, tear, hum_bar];
            u.p4 = [blue_screen, skew, crease, ghost];
            u.p5 = [osd, noise_evo, noise_op, blinds];
            u.p6 = [snow_dots, speed_dots, aniso_snow, jitter_x];
            u.p7 = [jitter_y, roll, roll_speed, glitch_seed];
        }
        "jpegdamage" => {
            let has_sapphire = params.values.contains_key("resFactor")
                || params.values.contains_key("scaleLights")
                || params.values.contains_key("errRate")
                || params.values.contains_key("allFreq");
            let q_default = if has_sapphire { 0.1 } else { 25.0 };
            let q = params.float_or("quality", q_default);
            let q_norm = if q > 1.0 { q / 100.0 } else { q }.clamp(0.01, 1.0);
            let res_factor = params.float_or("resFactor", 1.0).clamp(0.05, 1.0);
            let res_rel_x = params.float_or("resRelX", 1.0).clamp(0.1, 4.0);
            let low_w = (u.resolution[0] * res_factor * res_rel_x)
                .round()
                .clamp(8.0, u.resolution[0].max(8.0));
            let low_h = (u.resolution[1] * res_factor)
                .round()
                .clamp(8.0, u.resolution[1].max(8.0));
            let eff_time = if params.values.contains_key("time") {
                params.float_or("time", params.time)
            } else {
                params.time
            };
            u.p0 = [
                q_norm,
                params.float_or("amount", 1.0),
                res_factor,
                res_rel_x,
            ];
            u.p1 = [
                params.float_or("allFreq", 1.0),
                params.float_or("xFreq", 1.0),
                params.float_or("yFreq", 1.0),
                params.float_or("lowFreq", 1.0),
            ];
            u.p2 = [
                params.float_or("midFreq", 1.0),
                params.float_or("highFreq", 1.0),
                params.float_or("affectLuma", 1.0),
                params.float_or("affectChroma", 1.0),
            ];
            u.p3 = [
                params.float_or("errRate", 0.0),
                params.float_or("errDensity", 0.5),
                params.float_or("errAmp", 1.0),
                params.float_or("errCoherence", 0.5),
            ];
            u.p4 = [
                params.float_or("jitterFrames", 0.0),
                params.float_or("randSeed", 0.0),
                params.float_or("scaleLights", 1.0),
                params.float_or("offsetDarks", 0.0),
            ];
            u.p5 = [params.float_or("saturation", 1.0), low_w, low_h, eff_time];
        }
        "datamosh" => {
            let motion_mode = if params.values.contains_key("motion") {
                params.int_or("motion", 0)
            } else if params.values.contains_key("motionFamily") {
                params.int_or("motionFamily", 0)
            } else {
                params.int_or("mode", 0)
            };
            let algorithm = if params.values.contains_key("algorithm") {
                params.float_or("algorithm", 0.0)
            } else {
                params.float_or("lookMode", 0.0)
            };
            let bs = params.float_or("blockSize", 16.0).clamp(2.0, 128.0);
            let spacing = params.float_or("spacing", 4.0).clamp(1.0, 32.0);
            let interval = params.float_or("interval", 1.0).max(0.05);
            let eff_time = if params.values.contains_key("time") {
                params.float_or("time", params.time)
            } else {
                params.time
            };
            let segment = (eff_time / interval).floor();
            u.mode = motion_mode.clamp(0, 8) as u32;
            u.flags = u32::from(has_aux) | (u32::from(has_aux2) << 1);
            u.p0 = [
                params.float_or("amount", 0.8),
                params.float_or("intensity", 1.0),
                bs,
                params.float_or("seed", 1.0),
            ];
            u.p1 = [
                algorithm,
                params.float_or("hold", 0.0),
                params.float_or("speed", 1.0),
                params.float_or("acceleration", 0.25),
            ];
            u.p2 = [
                params.float_or("blend", 0.85),
                params.float_or("threshold", 0.08),
                interval,
                params.float_or("samples", 4.0),
            ];
            u.p3 = [bs, spacing, segment, eff_time];
        }
        "tracery" => {
            let key = params.vec3_or("keyColor", [1.0, 1.0, 1.0]);
            let box_col = if params.values.contains_key("boxColor") {
                params.vec3_or("boxColor", [0.2, 0.9, 1.0])
            } else {
                params.vec3_or("accentColor", [0.2, 0.9, 1.0])
            };
            let marker_col = params.vec3_or("markerColor", box_col);
            let line_col = params.vec3_or("lineColor", box_col);
            let soften = if params.values.contains_key("blur") {
                params.float_or("blur", 0.08)
            } else {
                params.float_or("soften", 0.08)
            };
            let raw_thresh = params.float_or("threshold", 0.5);
            let norm_thresh = if raw_thresh > 1.0 {
                (raw_thresh / 100.0).clamp(0.0, 1.0)
            } else {
                raw_thresh.clamp(0.0, 1.0)
            };
            let grid_divs = if params.values.contains_key("gridDivisions") {
                params.float_or("gridDivisions", 12.0)
            } else {
                params.float_or("gridSize", 12.0)
            };
            let box_op = params.float_or("boxOpacity", 1.0);
            let marker_op = params.float_or("markerOpacity", 1.0);
            let line_op = params.float_or("lineOpacity", 0.9);
            u.c0 = [key[0], key[1], key[2], 1.0];
            u.c1 = [box_col[0], box_col[1], box_col[2], box_op];
            u.c2 = [marker_col[0], marker_col[1], marker_col[2], marker_op];
            u.c3 = [line_col[0], line_col[1], line_col[2], line_op];
            u.p0 = [
                norm_thresh,
                soften,
                grid_divs,
                params.float_or("opacity", 1.0),
            ];
            u.p1 = [
                params.float_or("boxLineWidth", 2.0),
                params.float_or("markerSize", 10.0),
                params.float_or("markerRotation", 0.0).to_radians(),
                params.float_or("markerSides", 4.0),
            ];
            u.p2 = [
                params.float_or("lineThickness", 2.0),
                params.float_or("arrowSize", 10.0),
                params.float_or("arrowPosition", 0.5),
                params.int_or("boxShape", 0).clamp(0, 3) as f32,
            ];
            let p1_xy = params.vec2_or("point1Pos", [0.25, 0.35]);
            let p2_xy = params.vec2_or("point2Pos", [0.75, 0.35]);
            let p3_xy = params.vec2_or("point3Pos", [0.5, 0.7]);
            u.p3 = [
                params.float_or("point1X", p1_xy[0]),
                params.float_or("point1Y", p1_xy[1]),
                params.float_or("point1Size", 28.0),
                params.float_or("point2X", p2_xy[0]),
            ];
            u.p4 = [
                params.float_or("point2Y", p2_xy[1]),
                params.float_or("point2Size", 28.0),
                params.float_or("point3X", p3_xy[0]),
                params.float_or("point3Y", p3_xy[1]),
            ];
            u.p5 = [
                params.float_or("point3Size", 28.0),
                params.float_or("dashLength", 8.0),
                params.float_or("dashGap", 6.0),
                params.int_or("lineType", 0).clamp(0, 3) as f32,
            ];
            u.p6 = [
                params.int_or("markerShape", 0).clamp(0, 3) as f32,
                params.int_or("boxFill", 0).clamp(0, 3) as f32,
                params.float_or("labelOffsetX", 45.0),
                params.float_or("labelOffsetY", 35.0),
            ];
            let show_mask = u32::from(params.bool_or("showMask", false));
            let grid_on =
                u32::from(params.bool_or("gridEnable", params.bool_or("showGrid", false)));
            let box_on = u32::from(params.bool_or("boxEnable", true));
            let lines_on = u32::from(params.bool_or("linesEnable", true));
            let arrow_on = u32::from(params.bool_or("arrowEnable", false));
            let labels_on = u32::from(params.bool_or("labelsEnable", false));
            let marker_filled = u32::from(params.bool_or("markerFilled", true));
            let detect_on = u32::from(params.bool_or("detectEnable", true));
            let dash_on = u32::from(params.bool_or("dashEnable", false));
            let pt1_on = u32::from(params.bool_or("point1Enable", false));
            let pt2_on = u32::from(params.bool_or("point2Enable", false));
            let pt3_on = u32::from(params.bool_or("point3Enable", false));
            u.flags = show_mask
                | (grid_on << 1)
                | (box_on << 2)
                | (lines_on << 3)
                | (arrow_on << 4)
                | (labels_on << 5)
                | (marker_filled << 6)
                | (detect_on << 7)
                | (dash_on << 8)
                | (pt1_on << 9)
                | (pt2_on << 10)
                | (pt3_on << 11);
        }
        "ascii" => {
            let cell = if params.values.contains_key("blockSize") {
                let b = params.float_or("blockSize", 23.0).max(4.0);
                let c = params.float_or("charSize", 18.0).max(4.0);
                [c, b]
            } else {
                params.vec2_or("cellSize", [8.0, 12.0])
            };
            let default_mode = if params.values.contains_key("blockSize") {
                0
            } else {
                1
            };
            let color_mode = params.int_or("colorMode", default_mode).clamp(0, 4) as u32;
            let fg = match color_mode {
                2 => params.vec3_or("duotoneLight", params.vec3_or("fgColor", [1.0, 1.0, 1.0])),
                3 => params.vec3_or("monoColor", params.vec3_or("fgColor", [0.2, 1.0, 0.4])),
                _ => params.vec3_or("whiteColor", params.vec3_or("fgColor", [1.0, 1.0, 1.0])),
            };
            let bg = match color_mode {
                2 => params.vec3_or("duotoneDark", params.vec3_or("bgColor", [0.05, 0.05, 0.1])),
                _ => {
                    if params.values.contains_key("backgroundColor") {
                        params.vec3_or("backgroundColor", [0.03, 0.03, 0.04])
                    } else {
                        params.vec3_or("bgColor", [0.0, 0.0, 0.0])
                    }
                }
            };
            let black_col = params.vec3_or("blackColor", [0.15, 0.15, 0.18]);
            let grey_col = params.vec3_or("greyColor", [0.6, 0.6, 0.65]);
            let charset = match params.values.get("charset") {
                Some(EffectParamValue::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
                    "dense" | "detailed" => 1.0,
                    "blocks" | "shade" => 2.0,
                    "binary" => 3.0,
                    "katakana" | "symbols" => 4.0,
                    _ => 0.0,
                },
                _ => params.int_or("charset", 0).clamp(0, 4) as f32,
            };
            let font_family = params.int_or("fontFamily", 0).clamp(0, 4) as f32;
            u.mode = color_mode;
            u.p0 = [
                (cell[0] * res_scale[0]).max(2.0),
                (cell[1] * res_scale[1]).max(2.0),
                params.float_or("brightness", 0.0),
                params.float_or("contrast", 1.0),
            ];
            u.p1 = [
                charset,
                font_family,
                params.float_or("randomScale", 0.0),
                params.float_or("noiseIntensity", 0.0),
            ];
            u.p2 = [
                params.float_or("sineSpeed", 1.0),
                params.float_or("sineFreq", 1.5),
                params.float_or("sineSize", 16.0) * res_scale[1],
                params.float_or("sineIntensity", 1.0),
            ];
            u.p3 = [
                params.float_or("noiseScale", 1.0),
                params.float_or("noiseSpeed", 1.0),
                params.float_or("noiseOctaves", 3.0),
                params.float_or("noisePersistence", 0.5),
            ];
            u.p4 = [
                params.float_or("scrollWaves", 2.0),
                params.float_or("scrollSpeed", 1.0),
                params.float_or("glitchAmount", 12.0) * res_scale[0],
                params.float_or("glitchSpeed", 1.0),
            ];
            u.p5 = [
                params.float_or("blackOpacity", 1.0),
                params.float_or("whiteOpacity", 1.0),
                params.float_or("greyOpacity", 1.0),
                params.float_or("alphaOpacity", 0.0),
            ];
            u.c0 = [fg[0], fg[1], fg[2], 1.0];
            u.c1 = [bg[0], bg[1], bg[2], 1.0];
            u.c2 = [black_col[0], black_col[1], black_col[2], 1.0];
            u.c3 = [grey_col[0], grey_col[1], grey_col[2], 1.0];
            let rand_chars = u32::from(params.bool_or("randomChars", false));
            let sine_on = u32::from(params.bool_or("sineEnable", false));
            let sine_rad = u32::from(params.bool_or("sineRadial", false));
            let noise_on = u32::from(params.bool_or("noiseEnable", false));
            let noise_turb = u32::from(params.bool_or("noiseTurbulent", false));
            let scroll_on = u32::from(params.bool_or("scrollEnable", false));
            let glitch_on = u32::from(params.bool_or("glitchEnable", false));
            u.flags = rand_chars
                | (sine_on << 1)
                | (sine_rad << 2)
                | (noise_on << 3)
                | (noise_turb << 4)
                | (scroll_on << 5)
                | (glitch_on << 6);
        }
        "magiclooks" => {
            let sh = params.vec3_or("shadowTint", [1.0, 1.0, 1.0]);
            let hi = params.vec3_or("highlightTint", [1.0, 1.0, 1.0]);
            u.p0 = [
                params.float_or("exposure", 0.0),
                params.float_or("contrast", 1.0),
                params.float_or("saturation", 1.0),
                params.float_or("diffusion", 0.0),
            ];
            u.p1 = [
                params.float_or("temperature", 0.0),
                params.float_or("tint", 0.0),
                params.float_or("splitBalance", 0.5),
                params.float_or("strength", 1.0),
            ];
            u.p2 = [
                params.float_or("vignetteStrength", 0.0),
                params.float_or("vignetteRadius", 0.75),
                params.float_or("chromaticAberration", 0.0),
                params.float_or("grain", 0.0),
            ];
            u.c0 = [sh[0], sh[1], sh[2], 1.0];
            u.c1 = [hi[0], hi[1], hi[2], 1.0];
        }
        "smartmotionblur" => {
            let amt = if params.values.contains_key("amount") {
                params.float_or("amount", 0.5)
            } else {
                params.float_or("blurAmount", 0.5)
            };
            let sens = if params.values.contains_key("sensitivity") {
                params.float_or("sensitivity", 70.0)
            } else {
                params.float_or("motionSensitivity", 70.0)
            };
            let is_lite_sens = effect_id == "smart-motion-blur-lite"
                || params.values.contains_key("qualityPresetVersion")
                || params.values.contains_key("trackFrame");
            let sens01 = if is_lite_sens && sens <= 1.0 {
                sens.clamp(0.0, 1.0)
            } else {
                (sens / 100.0).clamp(0.0, 1.0)
            };
            let smooth = if params.values.contains_key("smoothness") {
                params.float_or("smoothness", 50.0)
            } else {
                params.float_or("vectorSmoothness", 50.0)
            };
            let quality = params.int_or("quality", 1).clamp(0, 2);
            let taps = match quality {
                0 => 8u32,
                2 => 24u32,
                _ => 16u32,
            };
            let shutter_angle = params.float_or("shutterAngle", 180.0).clamp(0.0, 720.0);
            let shutter_phase = params.float_or("shutterPhase", -90.0).clamp(-360.0, 360.0);
            let shutter_span = (shutter_angle / 180.0).max(0.0);
            let shutter_start = shutter_phase / 180.0;

            u.mode = params.int_or("view", 0).clamp(0, 2) as u32;
            u.extra = taps;
            u.flags = u32::from(has_aux) | (u32::from(has_aux2) << 1);
            u.p0 = [
                amt,
                sens01,
                params.float_or("maxDisplacement", 256.0).clamp(1.0, 256.0),
                (smooth / 100.0).clamp(0.0, 1.0),
            ];
            u.p1 = [shutter_start, shutter_span, 0.0, 0.0];
        }
        "echo" => {
            let is_legacy = effect_id == "echo-legacy"
                || params.bool_or("legacyMode", false)
                || (params.values.contains_key("strength")
                    && !params.values.contains_key("composite")
                    && !params.values.contains_key("operator"));
            if is_legacy {
                let weight = params.float_or("weight", params.float_or("startingIntensity", 1.0));
                let decay = params.float_or("decay", 0.5);
                let strength = params.float_or("strength", 1.0);
                let threshold = params.float_or("threshold", 0.0);
                let wsum = params.float_or("wsum", 1.0);
                let finalize = if params.bool_or("finalize", false) {
                    1.0
                } else {
                    0.0
                };
                u.mode = params.int_or("mode", 0).clamp(0, 3) as u32;
                u.flags = u32::from(has_aux) | (1u32 << 1);
                u.p0 = [weight, decay, strength, threshold];
                u.p1 = [wsum, finalize, 0.0, 0.0];
            } else {
                let weight = if params.values.contains_key("startingIntensity") {
                    params.float_or("startingIntensity", 1.0)
                } else if params.values.contains_key("opacity") {
                    params.float_or("opacity", 0.75)
                } else {
                    params.float_or("weight", 1.0)
                };
                let mode = if params.values.contains_key("composite") {
                    // Zoidium native-fx echo.js: 0=over, 1=behind, 2=add, 3=screen, 4=maximum
                    match params.int_or("composite", 0) {
                        0 => 4,
                        1 => 5,
                        2 => 0,
                        3 => 3,
                        _ => 1,
                    }
                } else {
                    params.int_or("operator", 0).clamp(0, 6) as u32
                };
                let newest_on_top = u32::from(
                    params.bool_or("newestOnTop", params.bool_or("compositeInFront", false)),
                );
                u.mode = mode;
                u.flags = u32::from(has_aux) | (newest_on_top << 2);
                u.p0 = [weight, params.float_or("decay", 0.5), 1.0, 0.0];
            }
        }
        // 16 Zoidium nativeEffects + helpers
        "timedisplacement" => {
            let max_disp = params.float_or("maxDisplacement", 1.0);
            let time_res = params.int_or("timeResolution", 30).clamp(2, 120) as f32;
            let stretch = u32::from(params.bool_or("stretchMap", true)) as f32;
            let map_scale = params.vec2_or("mapScale", [1.0, 1.0]);
            let map_off = params.vec2_or("mapOffset", [0.0, 0.0]);
            let has_slice_idx = params.values.contains_key("sliceIndex");
            u.mode = if has_slice_idx { 0 } else { 1 };
            u.flags = u32::from(has_aux) | (u32::from(has_aux2) << 1);
            u.p0 = [
                params.float_or("sliceIndex", 0.0),
                params.float_or("numSlices", time_res),
                map_scale[0],
                map_scale[1],
            ];
            u.p1 = [map_off[0], map_off[1], max_disp, stretch];
        }
        "ccwidetime" => {
            let fwd = params.int_or("forwardSteps", 2).clamp(0, 64) as f32;
            let bwd = params.int_or("backwardSteps", 2).clamp(0, 64) as f32;
            let default_w = 1.0 / (1.0 + fwd + bwd).max(1.0);
            let wt = params.float_or("weight", default_w);
            let is_init = u32::from(params.bool_or("init", false));
            let is_accum = u32::from(params.bool_or("nativeMotionBlur", false));
            u.flags = u32::from(has_aux) | (is_init << 1) | (is_accum << 2);
            u.p0 = [wt, fwd, bwd, 0.0];
        }
        "timedifference" => {
            let contrast = params.float_or("contrast", 100.0);
            let gain = (1.0 + contrast / 25.0).max(0.0);
            let offset = params.float_or("offset", 0.04);
            let abs_diff = u32::from(params.bool_or("absoluteDifference", false));
            u.mode = params.int_or("alphaChannel", 0).clamp(0, 8) as u32;
            u.flags = u32::from(has_aux) | (abs_diff << 1);
            u.p0 = [gain, offset, contrast, 0.0];
        }
        "colordiffkey" => {
            let key_raw = params.vec3_or("keyColor", [0.0, 1.0, 0.0]);
            let accurate = params.int_or("matchingMethod", 0) != 0;
            let kc = if accurate {
                [
                    srgb_to_linear_cpu(key_raw[0]),
                    srgb_to_linear_cpu(key_raw[1]),
                    srgb_to_linear_cpu(key_raw[2]),
                ]
            } else {
                key_raw
            };
            let (pri, o1, o2) = dominant_channel_triple(kc);
            let diff1 = (kc[pri] - kc[o1]).max(0.05);
            let diff2 = (kc[pri] - kc[o2]).max(0.05);
            u.mode = params.int_or("view", 0).clamp(0, 5) as u32;
            u.c0 = [key_raw[0], key_raw[1], key_raw[2], 1.0];
            u.p0 = [
                pri as f32,
                o1 as f32,
                o2 as f32,
                f32::from(u8::from(accurate)),
            ];
            u.p1 = [
                diff1,
                diff2,
                params.float_or("partialAGamma", 1.0),
                params.float_or("partialBGamma", 1.0),
            ];
            u.p2 = [
                params.float_or("partialAInBlack", 0.0),
                params.float_or("partialAInWhite", 255.0),
                params.float_or("partialAOutBlack", 0.0),
                params.float_or("partialAOutWhite", 255.0),
            ];
            u.p3 = [
                params.float_or("partialBInBlack", 0.0),
                params.float_or("partialBInWhite", 255.0),
                params.float_or("partialBOutBlack", 0.0),
                params.float_or("partialBOutWhite", 255.0),
            ];
            u.p4 = [
                params.float_or("matteInBlack", 0.0),
                params.float_or("matteInWhite", 255.0),
                params.float_or("matteGamma", 1.0),
                params.float_or("matchingSoftness", 0.0) / 100.0,
            ];
        }
        "advspill" => {
            let key = params.vec3_or("keyColor", [0.0, 1.0, 0.0]);
            let (pri, o1, o2) = dominant_channel_triple(key);
            let key_hue = rgb_hue_cpu(key);
            let method = params.int_or("method", 0);
            let tolerance = if method == 0 {
                1.0
            } else {
                (params.float_or("tolerance", 50.0) / 100.0).clamp(0.0, 1.0)
            };
            let spill_range = (params.float_or("spillRange", 50.0) / 100.0).clamp(0.0, 1.0);
            let suppression = (params.float_or("suppression", 100.0) / 100.0).clamp(0.0, 2.0);
            let luma_corr = if params.values.contains_key("preserveLuminance") {
                if params.bool_or("preserveLuminance", true) {
                    1.0
                } else {
                    0.0
                }
            } else {
                (params.float_or("lumaCorrection", 100.0) / 100.0).clamp(0.0, 2.0)
            };
            let desat = (params.float_or("desaturate", 0.0) / 100.0).clamp(0.0, 1.0);
            let neutral = (params.float_or("neutral", 0.0) / 100.0).clamp(0.0, 1.0);
            u.p0 = [pri as f32, o1 as f32, o2 as f32, key_hue];
            u.p1 = [suppression, spill_range, luma_corr, tolerance];
            u.p2 = [desat, neutral, 0.0, 0.0];
        }
        "keycleaner" => {
            let rad = params
                .float_or("additionalEdgeRadius", 10.0)
                .clamp(0.0, 50.0);
            let chatter = f32::from(u8::from(params.bool_or("reduceChatter", false)));
            let alpha_con_raw = if params.values.contains_key("alphaContrast") {
                params.float_or("alphaContrast", 100.0)
            } else {
                params.float_or("alphaContrastStrength", 0.0)
            };
            let alpha_con = (alpha_con_raw / 100.0).clamp(0.0, 1.0);
            let contrast = 1.0 + alpha_con * 4.0;
            let recover_raw = if params.values.contains_key("strength") {
                params.float_or("strength", 100.0)
            } else {
                params.float_or("recoverDetails", 50.0)
            };
            let recover = (recover_raw / 100.0).clamp(0.0, 1.0);
            let est_sigma = (rad * 0.5).max(0.0);
            let est_step = (rad / 4.0).max(1.0);
            u.p0 = [chatter, contrast, recover, est_sigma];
            u.p1 = [est_step, 0.0, 0.0, 0.0];
        }
        "linearcolorkey" => {
            let key = params.vec3_or("keyColor", [0.0, 0.0, 1.0]);
            let tol = (params.float_or("matchingTolerance", 10.0) / 100.0).clamp(0.0, 1.0);
            let soft = (params.float_or("matchingSoftness", 10.0) / 100.0).clamp(0.0, 1.0);
            let keep = f32::from(u8::from(params.int_or("keyOperation", 0) != 0));
            let match_mode = params.int_or("matchingColors", 0).clamp(0, 2) as f32;
            u.mode = params.int_or("view", 0).clamp(0, 2) as u32;
            u.c0 = [key[0], key[1], key[2], 1.0];
            u.p0 = [match_mode, tol, soft, keep];
        }
        "radiowaves" => {
            let prod = params.vec2_or("producerPoint", [0.5, 0.5]);
            let col = params.vec4_or("color", [1.0, 1.0, 1.0, 1.0]);
            let sides_raw = if params.values.contains_key("polygonSides") {
                params.int_or("polygonSides", 0)
            } else {
                params.int_or("sides", 0)
            };
            let is_star = params.bool_or("star", false);
            // Default `sides = 64` without star represents a smooth circle (`0.0` in shader).
            let sides = if !is_star && sides_raw >= 64 {
                0
            } else {
                sides_raw.clamp(0, 64)
            };
            let sides_f = if sides >= 3 { sides as f32 } else { 0.0 };
            let star_k = if is_star && sides >= 3 {
                if params.values.contains_key("starInnerRadius") {
                    params.float_or("starInnerRadius", 0.5).clamp(0.05, 0.99)
                } else {
                    (1.0 + params.float_or("starDepth", -0.3)).clamp(0.05, 0.99)
                }
            } else {
                0.0
            };
            let t_sec = if params.values.contains_key("time") {
                params.float_or("time", params.time)
            } else {
                params.time
            };
            let comp_orig = u32::from(if params.values.contains_key("composite") {
                params.bool_or("composite", true)
            } else {
                params.bool_or("compositeOriginal", true)
            });
            let do_reflect = u32::from(params.bool_or("reflection", false));
            let op_raw = params.float_or("opacity", 1.0);
            let op01 = if op_raw > 1.0 {
                (op_raw / 100.0).clamp(0.0, 1.0)
            } else {
                op_raw.clamp(0.0, 1.0)
            };
            u.flags = comp_orig | (do_reflect << 1);
            u.c0 = col;
            u.p0 = [
                prod[0],
                prod[1],
                params.float_or("frequency", 1.0).max(0.01),
                params.float_or("expansion", 100.0),
            ];
            u.p1 = [
                params.float_or("orientation", 0.0).to_radians(),
                params.float_or("direction", 0.0).to_radians(),
                params.float_or("velocity", 0.0),
                params.float_or("spin", 0.0).to_radians(),
            ];
            u.p2 = [
                params.float_or("lifespan", 3.0).max(0.01),
                params.float_or("startWidth", 6.0),
                params.float_or("endWidth", 2.0),
                params.float_or("fadeIn", 0.2),
            ];
            u.p3 = [params.float_or("fadeOut", 0.5), op01, sides_f, star_k];
            u.p4 = [
                t_sec,
                params.int_or("profile", 0).clamp(0, 6) as f32,
                0.0,
                0.0,
            ];
        }
        "beam" => {
            let sp = params.vec2_or("startPoint", [0.25, 0.5]);
            let ep = params.vec2_or("endPoint", [0.75, 0.5]);
            let len_raw = params.float_or("length", 50.0);
            let len01 = if len_raw > 1.0 {
                (len_raw / 100.0).clamp(0.0, 1.0)
            } else {
                len_raw.clamp(0.0, 1.0)
            };
            let time_raw = params.float_or("time", 50.0);
            let time_is_secs = params.bool_or("_timeIsSeconds", false);
            let time01 = if time_is_secs {
                // Default `expr = "time"` supplies playback seconds: loop smoothly across visible travel!
                (time_raw * 0.35 + 0.15).rem_euclid(1.0)
            } else if time_raw > 1.0 {
                (time_raw / 100.0).clamp(0.0, 1.0)
            } else {
                time_raw.clamp(0.0, 1.0)
            };
            let soft_raw = params.float_or("softness", 50.0);
            let soft01 = if soft_raw > 1.0 {
                (soft_raw / 100.0).clamp(0.0, 1.0)
            } else {
                soft_raw.clamp(0.0, 1.0)
            };
            let t0 = (time01 * (1.0 + len01) - len01).clamp(0.0, 1.0);
            let t1 = (time01 * (1.0 + len01)).clamp(0.0, 1.0);
            let in_col = params.vec3_or("insideColor", [1.0, 1.0, 1.0]);
            let out_col = params.vec3_or("outsideColor", [1.0, 0.15, 0.15]);
            let comp_orig = u32::from(if params.values.contains_key("composite") {
                params.bool_or("composite", true)
            } else {
                params.bool_or("compositeOriginal", true)
            });
            let persp = u32::from(params.bool_or("perspective", true));
            u.flags = comp_orig | (persp << 1);
            u.c0 = [in_col[0], in_col[1], in_col[2], 1.0];
            u.c1 = [out_col[0], out_col[1], out_col[2], 1.0];
            u.p0 = [sp[0], sp[1], ep[0], ep[1]];
            u.p1 = [t0, t1, len01, soft01];
            u.p2 = [
                params.float_or("startThickness", 10.0),
                params.float_or("endThickness", 10.0),
                0.0,
                0.0,
            ];
        }
        "ccpageturn" => {
            let fp = params.vec2_or("foldPosition", [0.75, 0.75]);
            let paper = params.vec3_or("paperColor", [0.85, 0.85, 0.85]);
            let back_op_raw = params.float_or("backOpacity", 70.0);
            let back_op = if back_op_raw > 1.0 {
                (back_op_raw / 100.0).clamp(0.0, 1.0)
            } else {
                back_op_raw.clamp(0.0, 1.0)
            };
            u.mode = params.int_or("render", 0).clamp(0, 2) as u32;
            u.c0 = [paper[0], paper[1], paper[2], 1.0];
            u.p0 = [
                fp[0],
                fp[1],
                params.float_or("foldDirection", 135.0),
                params.float_or("foldRadius", 60.0),
            ];
            u.p1 = [
                params.float_or("lightDirection", 45.0),
                back_op,
                params.int_or("backPage", 0).clamp(0, 1) as f32,
                0.0,
            ];
        }
        "cardwipe" => {
            let comp_raw = if params.values.contains_key("completion") {
                params.float_or("completion", 0.0)
            } else if params.values.contains_key("transitionCompletion") {
                params.float_or("transitionCompletion", 0.0)
            } else {
                0.0
            };
            let comp01 = (comp_raw / 100.0).clamp(0.0, 1.0);
            let tw_raw = params.float_or("transitionWidth", 50.0);
            let tw01 = (tw_raw / 100.0).clamp(0.01, 1.0);
            let back_col = params.vec3_or("backColor", [0.2, 0.2, 0.25]);
            let seed = if params.values.contains_key("randomSeed") {
                params.int_or("randomSeed", 1)
            } else {
                params.int_or("seed", 1)
            }
            .unsigned_abs()
            .max(1);
            u.mode = params.int_or("backLayer", 0).clamp(0, 1) as u32;
            u.extra = seed;
            u.c0 = [back_col[0], back_col[1], back_col[2], 1.0];
            u.p0 = [
                params.int_or("rows", 8).clamp(1, 200) as f32,
                params.int_or("columns", 8).clamp(1, 200) as f32,
                params.float_or("cardScale", 1.0).clamp(0.05, 4.0),
                comp01,
            ];
            u.p1 = [
                tw01,
                params.int_or("flipAxis", 0).clamp(0, 2) as f32,
                params.int_or("flipDirection", 0).clamp(0, 2) as f32,
                params.int_or("flipOrder", 0).clamp(0, 8) as f32,
            ];
            u.p2 = [
                (params.float_or("timingRandomness", 0.0) / 100.0).clamp(0.0, 1.0),
                params.float_or("cameraDistance", 2.5).max(1.0),
                params.float_or("lightIntensity", 1.0).max(0.0),
                params.float_or("ambientLight", 0.3).clamp(0.0, 1.0),
            ];
        }
        "ccpixelpolly" => {
            let t = if params.values.contains_key("time") {
                params.float_or("time", 0.0) - params.float_or("startTime", 0.0)
            } else {
                params.time - params.float_or("startTime", 0.0)
            };
            let fc = params.vec2_or("forceCenter", [0.5, 0.5]);
            let obj_mode = params.int_or("object", 1).clamp(0, 3);
            let flat_mode = if params.values.contains_key("object") {
                obj_mode == 0 || obj_mode == 2
            } else {
                params.int_or("texture", 0) != 0
            };
            let seed = if params.values.contains_key("randomSeed") {
                params.float_or("randomSeed", 1.0)
            } else {
                params.float_or("seed", 1.0)
            };
            u.flags = u32::from(flat_mode);
            u.p0 = [
                t.max(0.0),
                params.float_or("gridSpacing", 10.0).clamp(1.0, 500.0),
                params.float_or("force", 100.0),
                params.float_or("gravity", 1.0),
            ];
            u.p1 = [fc[0], fc[1], params.float_or("spinning", 0.0), seed];
        }
        "radialshadow" => {
            let ls = params.vec2_or("lightSource", [0.3, 0.3]);
            let col = params.vec3_or("shadowColor", [0.0, 0.0, 0.0]);
            let op_raw = params.float_or("opacity", 75.0);
            let op01 = if op_raw > 1.0 {
                (op_raw / 100.0).clamp(0.0, 1.0)
            } else {
                op_raw.clamp(0.0, 1.0)
            };
            let ci_raw = params.float_or("colorInfluence", 0.0);
            let ci01 = if ci_raw > 1.0 {
                (ci_raw / 100.0).clamp(0.0, 1.0)
            } else {
                ci_raw.clamp(0.0, 1.0)
            };
            u.mode = params.int_or("render", 0).clamp(0, 1) as u32;
            u.flags = u32::from(params.bool_or("shadowOnly", false));
            u.c0 = [col[0], col[1], col[2], 1.0];
            u.p0 = [
                ls[0],
                ls[1],
                params.float_or("projectionDistance", 50.0),
                params.float_or("softness", 12.0).max(0.0),
            ];
            u.p1 = [op01, ci01, 0.0, 0.0];
        }
        "layerinputdisplacement" => {
            let doff = params.vec2_or("displacementOffset", [0.0, 0.0]);
            let off = params.vec2_or("offset", [0.0, 0.0]);
            let sc = params.vec2_or("scale", [1.0, 1.0]);
            let rot = params.float_or("rotation", 0.0);
            let ud = params.vec3_or("uDisplacement", [1.0, 0.0, 0.0]);
            let vd = params.vec3_or("vDisplacement", [0.0, 1.0, 0.0]);
            let (m0, m1) = compute_overlay_uv_transform(off, sc, rot, u.resolution);
            u.mode = params.int_or("wrap", 2).clamp(0, 2) as u32;
            u.flags = u32::from(has_aux);
            u.p0 = [doff[0], doff[1], params.float_or("amount", 0.1), rot];
            u.p1 = [off[0], off[1], sc[0], sc[1]];
            u.p2 = [ud[0], ud[1], ud[2], 0.0];
            u.p3 = [vd[0], vd[1], vd[2], 0.0];
            u.m0 = m0;
            u.m1 = m1;
        }
        "framemix" => {
            let frac = if params.values.contains_key("frac") {
                params.float_or("frac", 0.5)
            } else {
                params.float_or("mix", 0.5)
            };
            u.mode = params
                .int_or("blendMode", params.int_or("mode", 0))
                .clamp(0, 2) as u32;
            u.flags = u32::from(has_aux);
            u.p0 = [frac.clamp(0.0, 1.0), 0.0, 0.0, 0.0];
        }
        "shutteraccum" => {
            let wt = params.float_or("weight", 1.0);
            let inv_total = params.float_or("invTotalWeight", 1.0);
            let is_init = u32::from(params.bool_or("init", false));
            let is_final = u32::from(params.bool_or("finalize", false));
            u.flags = u32::from(has_aux) | (is_init << 1) | (is_final << 2);
            u.p0 = [wt, inv_total, 0.0, 0.0];
        }
        _ => {}
    }
    u
}

/// Computes the exact CM3 `gradientoverlay.js` `uvTransform` matrix rows `(m0, m1)`.
pub(crate) fn compute_gradient_uv_transform(
    offset: [f32; 2],
    scale: [f32; 2],
    rotation: f32,
    resolution: [f32; 2],
) -> ([f32; 4], [f32; 4]) {
    let w = resolution[0].max(1.0);
    let h = resolution[1].max(1.0);
    let max_dim = w.max(h);
    let aspect_w = max_dim / w;
    let aspect_h = max_dim / h;
    let i = if scale[0].abs() < 1e-6 {
        1e-6
    } else {
        scale[0]
    };
    let r = if scale[1].abs() < 1e-6 {
        1e-6
    } else {
        scale[1]
    };
    let a0 = 1.0 / (i * aspect_w);
    let a1 = 1.0 / (r * aspect_h);
    let (n, p) = rotation.sin_cos();
    let m0 = [
        a0 * p,
        a1 * n * r / i,
        -a0 * p * 0.5 - a1 * n * r / i * 0.5 - offset[0] * p / i - offset[1] * n / i + 0.5,
        0.0,
    ];
    let m1 = [
        -a0 * n * i / r,
        a1 * p,
        a0 * n * i / r * 0.5 - a1 * p * 0.5 + offset[0] * n / r - offset[1] * p / r + 0.5,
        0.0,
    ];
    (m0, m1)
}

/// Computes the exact CM3 `overlay.js` / `displacementmap.js` `uvTransform` matrix rows `(m0, m1)`.
pub(crate) fn compute_overlay_uv_transform(
    offset: [f32; 2],
    scale: [f32; 2],
    rotation: f32,
    resolution: [f32; 2],
) -> ([f32; 4], [f32; 4]) {
    let w = resolution[0].max(1.0);
    let h = resolution[1].max(1.0);
    let aspect = w / h;
    let sx = if scale[0].abs() < 1e-6 {
        1e-6
    } else {
        scale[0]
    };
    let sy = if scale[1].abs() < 1e-6 {
        1e-6
    } else {
        scale[1]
    };
    let r = aspect * sx;
    let i = sy;
    let s0 = 1.0 / sx;
    let s1 = 1.0 / sy;
    let (n, p) = rotation.sin_cos();
    let m0 = [
        s0 * p,
        s1 * n * i / r,
        -s0 * p * 0.5 - s1 * n * i / r * 0.5 - offset[0] * p / r - offset[1] * n / r + 0.5,
        0.0,
    ];
    let m1 = [
        -s0 * n * r / i,
        s1 * p,
        s0 * n * r / i * 0.5 - s1 * p * 0.5 + offset[0] * n / i - offset[1] * p / i + 0.5,
        0.0,
    ];
    (m0, m1)
}

/// Computes the exact CM3 `duplicate.js` `uvTransform` matrix rows `(m0, m1)`.
pub(crate) fn compute_duplicate_uv_transform(
    offset: [f32; 2],
    multiplier: [f32; 2],
    rotation: f32,
    resolution: [f32; 2],
) -> ([f32; 4], [f32; 4]) {
    let w = resolution[0].max(1.0);
    let h = resolution[1].max(1.0);
    let aspect = w / h;
    let s0 = if multiplier[0].abs() < 1e-6 {
        1e-6
    } else {
        multiplier[0]
    };
    let s1 = if multiplier[1].abs() < 1e-6 {
        1e-6
    } else {
        multiplier[1]
    };
    let r = aspect * s1;
    let i = s0;
    let (sin_a, cos_a) = rotation.sin_cos();
    let m0 = [
        s0 * cos_a,
        s1 * sin_a * i / r,
        -s0 * cos_a * 0.5 - s1 * sin_a * i / r * 0.5 + offset[0] + 0.5,
        0.0,
    ];
    let m1 = [
        -s0 * sin_a * r / i,
        s1 * cos_a,
        s0 * sin_a * r / i * 0.5 - s1 * cos_a * 0.5 + offset[1] + 0.5,
        0.0,
    ];
    (m0, m1)
}

/// Computes the 3x3 inverse homography (row-major) and camera-space depth plane
/// `[u_cam.z, v_cam.z, o_cam.z, far]` for `shake.js` using exact `PZ.shake` 3D camera transform.
pub(crate) fn compute_shake_inverse_homography(
    params: &EffectParams,
) -> Option<([[f32; 3]; 3], [f32; 4])> {
    let w = params.resolution[0].max(1.0);
    let h = params.resolution[1].max(1.0);
    let d = w.max(h);
    let amp = params.float_or("amplitude", 0.01);
    let speed = params.float_or("speed", 15.0);
    // CM3 shake.js uses exact `-zoom` without an arbitrary `d * 0.85` clamp.
    // Only when `"zoom"` is omitted on tiny synthetic test targets (`d <= 101`, e.g. 64x64)
    // scale the omitted default so the quad isn't behind the near plane.
    let default_zoom = if d <= 101.0 {
        (d * 0.1).min(100.0)
    } else {
        100.0
    };
    let zoom = params.float_or("zoom", default_zoom);
    let shake_mode = params.int_or("shake.mode", 1);
    let chan_amp = if params.values.contains_key("shake.amplitude") {
        params.vec4_or("shake.amplitude", [1000.0, 1000.0, 1000.0, 10.0])
    } else {
        params.vec4_or("shake.amp", [1000.0, 1000.0, 1000.0, 10.0])
    };
    let chan_freq = params.vec4_or("shake.frequency", [0.6, 0.8, 0.7, 0.9]);
    let chan_phase = params.vec4_or("shake.phase", [0.0, 11.0, 20.0, 31.0]);
    let chan_noise = params.vec4_or("shake.noise", [0.9, 0.9, 0.9, 0.55]);
    let chan_smooth = params.vec4_or("shake.smooth", [1.0, 1.0, 1.0, 1.0]);
    let shake_params = zc_scene::CameraShakeParams {
        mode: if shake_mode == 0 {
            zc_scene::CameraShakeMode::Rotate
        } else {
            zc_scene::CameraShakeMode::Move
        },
        amplitude: chan_amp,
        frequency: chan_freq,
        phase: chan_phase,
        noise: chan_noise,
        smooth: [
            chan_smooth[0] >= 0.5,
            chan_smooth[1] >= 0.5,
            chan_smooth[2] >= 0.5,
            chan_smooth[3] >= 0.5,
        ],
    };

    // shake.js: camera starts at (0, 0, -zoom), then PZ.shake.shake(r, amp, camera)
    // right-multiplies M_shake (with * 0.1 rad scale on Rotate/Roll).
    // Guard only against singular division (`d <= 1e-4`), matching CM3 shake.js verbatim.
    if !d.is_finite() || d <= 1e-4 {
        return None;
    }
    let m_shake = zc_scene::evaluate_camera_shake(
        f64::from(params.frame),
        f64::from(amp),
        f64::from(speed),
        &shake_params,
    );
    let m_cam = zc_geom::Mat4::from_translation(zc_geom::Vec3::new(0.0, 0.0, -zoom)) * m_shake;
    let m_cam_inv = m_cam.inverse();

    let u_cam = m_cam_inv.transform_vector3(zc_geom::Vec3::new(0.5 * w, 0.0, 0.0));
    let v_cam = m_cam_inv.transform_vector3(zc_geom::Vec3::new(0.0, 0.5 * h, 0.0));
    let o_cam = m_cam_inv.transform_point3(zc_geom::Vec3::new(0.0, 0.0, -d));

    let fx = 2.0 * d / w;
    let fy = 2.0 * d / h;
    let m = [
        [fx * u_cam.x, fx * v_cam.x, fx * o_cam.x],
        [fy * u_cam.y, fy * v_cam.y, fy * o_cam.y],
        [-u_cam.z, -v_cam.z, -o_cam.z],
    ];
    let h_inv = invert_mat3(m)?;
    Some((h_inv, [u_cam.z, v_cam.z, o_cam.z, 2.0 * d]))
}

/// Computes the 3x3 inverse homography (row-major) and camera-space depth plane
/// `[u_cam.z, v_cam.z, o_cam.z, far]` mapping screen NDC `(nx, ny, 1)`
/// to quad local coordinates `(qx, qy, qw)` for the `transform` effect.
pub(crate) fn compute_transform_inverse_homography(
    params: &EffectParams,
) -> Option<([[f32; 3]; 3], [f32; 4])> {
    let w = params.resolution[0].max(1.0);
    let h = params.resolution[1].max(1.0);
    let d = w.max(h);

    let ortho = params.int_or("cameraType", 0) == 1;
    let cam_pos = params.vec3_or("cameraPosition", [0.0, 0.0, 0.0]);
    let cam_rot = params.vec3_or("cameraRotation", [0.0, 0.0, 0.0]);
    let img_pos = params.vec3_or("imagePosition", [0.0, 0.0, 0.0]);
    let img_rot = params.vec3_or("imageRotation", [0.0, 0.0, 0.0]);
    let img_scale = params.vec2_or("imageScale", [1.0, 1.0]);

    let r_img = euler_xyz_matrix(img_rot);
    let r_cam = euler_xyz_matrix(cam_rot);

    let u_world = mat3_mul_vec(r_img, [0.5 * w * img_scale[0], 0.0, 0.0]);
    let v_world = mat3_mul_vec(r_img, [0.0, 0.5 * h * img_scale[1], 0.0]);
    let o_world = [
        img_pos[0] - cam_pos[0],
        img_pos[1] - cam_pos[1],
        (img_pos[2] - d) - cam_pos[2],
    ];

    let r_cam_t = mat3_transpose(r_cam);
    let u_cam = mat3_mul_vec(r_cam_t, u_world);
    let v_cam = mat3_mul_vec(r_cam_t, v_world);
    let o_cam = mat3_mul_vec(r_cam_t, o_world);

    let cols = if ortho {
        [
            [(2.0 / w) * u_cam[0], (2.0 / h) * u_cam[1], 0.0],
            [(2.0 / w) * v_cam[0], (2.0 / h) * v_cam[1], 0.0],
            [(2.0 / w) * o_cam[0], (2.0 / h) * o_cam[1], 1.0],
        ]
    } else {
        let fx = 2.0 * d / w;
        let fy = 2.0 * d / h;
        [
            [fx * u_cam[0], fy * u_cam[1], -u_cam[2]],
            [fx * v_cam[0], fy * v_cam[1], -v_cam[2]],
            [fx * o_cam[0], fy * o_cam[1], -o_cam[2]],
        ]
    };

    // Convert column-major `cols` into row-major 3x3 matrix and invert.
    let m = [
        [cols[0][0], cols[1][0], cols[2][0]],
        [cols[0][1], cols[1][1], cols[2][1]],
        [cols[0][2], cols[1][2], cols[2][2]],
    ];
    let h_inv = invert_mat3(m)?;
    Some((h_inv, [u_cam[2], v_cam[2], o_cam[2], 2.0 * d]))
}

pub(crate) fn euler_xyz_matrix(rot: [f32; 3]) -> [[f32; 3]; 3] {
    let (sx, cx) = rot[0].sin_cos();
    let (sy, cy) = rot[1].sin_cos();
    let (sz, cz) = rot[2].sin_cos();
    // Intrinsic XYZ = R_x * R_y * R_z (row-major).
    [
        [cy * cz, -cy * sz, sy],
        [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
        [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy],
    ]
}

pub(crate) fn mat3_transpose(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

pub(crate) fn mat3_mul_vec(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub(crate) fn invert_mat3(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    if det.abs() < 1e-10 {
        return None;
    }
    let inv_det = 1.0 / det;
    let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);
    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    Some([
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ])
}

/// FNV-1a 64-bit hash for deterministic custom WGSL shader pipeline caching.
#[must_use]
pub fn fnv1a_hash64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

pub(crate) const ADDITIVE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};
