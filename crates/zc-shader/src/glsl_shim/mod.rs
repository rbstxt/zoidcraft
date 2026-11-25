//! Clean-room GLSL ES 1.00 -> GLSL 450 -> `naga` -> WGSL translator for Panzoid Custom Shaders.
//!
//! Translates WebGL 1 / ClipMaker 3 custom fragment shaders (Spec 04 §3) into validated
//! WGSL modules along with deterministic `std140` uniform buffer layouts and texture
//! binding tables for `zc-render`.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

use naga::back::wgsl::{self, WriterFlags};
use naga::front::glsl::{Frontend, Options};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod decls;
mod emit;
mod includes;
mod lexer;
#[cfg(test)]
mod opt_tests;
mod preprocess;
mod rewrite;
mod rewrite_helpers;
mod types;
mod uniforms;

use decls::*;
use emit::*;
pub use includes::*;
use lexer::*;
use preprocess::*;
use rewrite::*;
use rewrite_helpers::*;
pub use types::*;
pub use uniforms::*;
