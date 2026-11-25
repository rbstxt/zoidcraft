//! Node Graph Compilers for Zoidium `node-effect` (GLSL ES 1.00 fragment shader codegen)
//! and `material-plus` `NodeMaterial` (deterministic CPU RGBA8 PBR texture baker).
//!
//! - [`compile_node_effect_to_glsl`]: compiles Zoidium `node-effect` graphs into Naga-valid
//!   GLSL ES 1.00 fragment shaders feeding directly into [`crate::translate_glsl_es100`].
//! - [`bake_node_material_maps`]: evaluates Zoidium `material-plus` procedural graphs into
//!   deterministic CPU RGBA8 texture maps (`color`, `emissive`, `roughness`, `metalness`,
//!   `normal`, `bump`, `opacity`) plus scalar PBR fallbacks.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::f64::consts::{PI, SQRT_2, TAU};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::shader_plus::{PreparedCurve, ZoidiumCurvePoint};

mod effect_compiler;
mod effect_types;
mod graph_parse;
mod material_bake;
mod material_eval;
mod material_ops;
mod material_types;
#[cfg(test)]
mod tests;

pub use effect_compiler::*;
pub use effect_types::*;
use graph_parse::*;
pub use material_bake::*;
use material_eval::*;
use material_ops::*;
pub use material_types::*;
