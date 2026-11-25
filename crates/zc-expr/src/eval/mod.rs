//! Deterministic tree-walking evaluator with arena-allocated environments,
//! strict step and recursion budgets, vector arithmetic, and host scene handles.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;
use std::rc::Rc;

use crate::ast::{
    ArrayElem, ArrayPatternElem, AssignOp, AssignTarget, BinaryOp, CallArg, ChainOp, Expr, ForInit,
    FunctionBody, Literal, LogicalOp, ObjectKey, ObjectPatternProp, ObjectProp, Param, Pattern,
    Program, Stmt, TemplatePart, UnaryOp, UpdateOp, VarKind, is_extended, is_forbidden_ident,
};
use crate::builtins::{
    ae_ease, ae_ease_in, ae_ease_out, ae_linear, catmull_rom_value, clamp_f64, cm3_ease,
    cm3_gauss_random, cm3_random, cm3_shake, cm3_wave, eval_math_fn, eval_wiggle, format_js_number,
    format_js_number_radix, js_parse_float, js_parse_int, js_round, json_parse_bounded,
    json_stringify_bounded, lerp_value, look_at_xyz_euler, math_constant, property_seed, to_i32,
    to_u32, vec_add, vec_cross, vec_div_componentwise, vec_div_scalar, vec_dot, vec_length,
    vec_mul_componentwise, vec_mul_scalar, vec_normalize, vec_sub,
};
use crate::{
    EvalConfig, ExprContext, ExprError, ExprMode, HandleId, HandleRef, KeyframeRecord, NameOrIndex,
    Value,
};

mod api;
mod evaluator_bind;
mod evaluator_builtins;
mod evaluator_call;
mod evaluator_convert;
mod evaluator_expr;
mod evaluator_lifecycle;
mod evaluator_methods;
mod evaluator_ops;
mod helpers;
#[cfg(test)]
mod tests;
mod types;

pub use api::*;
use helpers::*;
use types::*;
