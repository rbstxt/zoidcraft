//! Clean-room Rust port of Zoidium ExpressText+ (`plugins/expresstext-plus/`):
//! - C4D-style Formula micro-language (`expresstext-formula.js`)
//! - Seeded PRNG, 3D integer-lattice / fractal value noise, and stagger ranks (`expresstext-random.js`)
//! - Letter selection & linear/box/sphere field falloff (`expresstext-select.js`)
//! - All 11 per-letter effectors: Shake, Plain, Random, Formula, Delay, Spline, Orbit,
//!   Repeater, Twist, Wrap, Voronoi (`expresstext-effectors.js`, `expresstext-layout.js`,
//!   `expresstext-voronoi.js`, `expresstext-delay.js`)
//! - `LettersMode` (`PerText` vs `AcrossTexts`) and `expressProperties` JSON parser

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::f64::consts::{E, PI};
use std::rc::Rc;

use serde::{Deserialize, Serialize};

use crate::builtins::js_round;

mod config;
mod effector_math;
mod effector_stack;
mod effector_types;
mod formula;
mod noise;
mod repeater;
mod selection;
#[cfg(test)]
mod tests;
mod value;
mod voronoi;

pub use config::*;
pub use effector_math::*;
pub use effector_stack::*;
pub use effector_types::*;
pub use formula::*;
pub use noise::*;
pub use repeater::*;
pub use selection::*;
pub use value::*;
pub use voronoi::*;
