pub(crate) use std::borrow::Cow;
pub(crate) use std::collections::{BTreeMap, HashMap};
pub(crate) use std::f32::consts::PI;

pub(crate) use bytemuck::{Pod, Zeroable};
pub(crate) use zc_shader::{
    CompiledCustomShader, TextureBinding, UniformBlockLayout, UniformValue, pack_uniforms,
};

pub(crate) use crate::context::{GpuContext, RenderError, RenderTarget};

mod particles_overlays;
mod pass;
mod shaders;
mod types;

#[cfg(test)]
mod tests;

pub use self::particles_overlays::*;
pub use self::pass::*;
pub(crate) use self::shaders::*;
pub use self::types::*;

/// Maximum dynamic lights evaluated per 3D scene pass.
pub const MAX_SCENE_LIGHTS: usize = 8;

/// Default directional/spot shadow map resolution (per Spec 03 §3.6).
pub const SHADOW_MAP_SIZE: u32 = 1024;

/// Maximum simultaneous 2D (Spot / Directional) shadow-casting lights per pass.
pub const MAX_SHADOW_MAPS_2D: usize = 4;

/// Maximum simultaneous omni-directional PointLight shadow casters per pass.
pub const MAX_POINT_SHADOW_LIGHTS: usize = 2;

/// Per-face resolution of PointLight cube-face shadow maps.
pub const POINT_SHADOW_MAP_SIZE: u32 = 512;

/// Default CM3 shadow camera near plane (`core-1.0.102.js`: `shadow.camera.near = 5`).
pub const SHADOW_CAMERA_NEAR: f32 = 5.0;

/// Default CM3 shadow camera far plane (`core-1.0.102.js`: `shadow.camera.far = 1500`).
pub const SHADOW_CAMERA_FAR: f32 = 1500.0;

/// Tile edge size in pixels for Camera+ 3-pass bokeh DOF (`scene-dof.js`: `TILE = 16`).
pub(crate) const DOF_TILE_SIZE: u32 = 16;
