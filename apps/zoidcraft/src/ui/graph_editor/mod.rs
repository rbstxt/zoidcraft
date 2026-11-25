//! Dual-mode Vanilla CM3 Graph Editor & Zoidium Graph Editor 2 (`graph-plus`) + Easing+ (`easing-plus`)
//! widget (Spec 02 §4).

use egui::Color32;

mod curves;
mod easing_plus;
mod edits;
mod interactions;
mod state;
#[cfg(test)]
mod tests;

pub use curves::*;
pub use easing_plus::*;
pub use edits::*;
pub use interactions::*;
pub use state::*;

pub const COLOR_BG: Color32 = Color32::from_rgb(0x22, 0x22, 0x22);
pub const COLOR_CANVAS_BG: Color32 = Color32::from_rgb(0x1b, 0x1b, 0x1b);
pub const COLOR_TRACK_LIST_BG: Color32 = Color32::from_rgb(0x26, 0x26, 0x26);
pub const COLOR_RULER_BG: Color32 = Color32::from_rgb(0x24, 0x24, 0x24);
pub const COLOR_GRID_MINOR: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);
pub const COLOR_GRID_MAJOR: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
pub const COLOR_GRID_ZERO: Color32 = Color32::from_rgb(0x6d, 0x6d, 0x6d);
pub const COLOR_GRID_VANILLA_MAJOR: Color32 = Color32::from_rgb(0x1e, 0x1e, 0x1e);
pub const COLOR_GRID_VANILLA_MINOR: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
pub const COLOR_LABEL_MUTED: Color32 = Color32::from_rgb(0x77, 0x77, 0x77);
pub const COLOR_TEXT: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
pub const COLOR_TEXT_BRIGHT: Color32 = Color32::from_rgb(0xe4, 0xe4, 0xe4);
pub const COLOR_TITLE_TEXT: Color32 = Color32::from_rgb(0xda, 0xda, 0xda);
pub const COLOR_BORDER: Color32 = Color32::from_rgb(0x1b, 0x1b, 0x1b);
pub const COLOR_WINDOW_BORDER: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
pub const COLOR_PLAYHEAD: Color32 = Color32::from_rgb(0xb7, 0x1e, 0x1e);
pub const COLOR_PLAYHEAD_VANILLA: Color32 =
    Color32::from_rgba_premultiplied(0xaa, 0x00, 0x00, 0xaa);
pub const COLOR_SNAP_GUIDE: Color32 = Color32::from_rgba_premultiplied(114, 114, 114, 175);
pub const COLOR_HANDLE: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
pub const COLOR_TILE_BG: Color32 = Color32::from_rgb(0x30, 0x30, 0x30);
pub const COLOR_TILE_HOVER: Color32 = Color32::from_rgb(0x3f, 0x3f, 0x3f);

/// Multi-channel curve colors: X/R, Y/G, Z/B, W/A (Spec 02 §4.2).
pub const COLOR_AXIS_X: Color32 = Color32::from_rgb(0xed, 0x6a, 0x68);
pub const COLOR_AXIS_Y: Color32 = Color32::from_rgb(0x81, 0xc7, 0x84);
pub const COLOR_AXIS_Z: Color32 = Color32::from_rgb(0x64, 0xb5, 0xf6);
pub const COLOR_AXIS_W: Color32 = Color32::from_rgb(0xc7, 0x92, 0xea);

pub const TRACK_LIST_WIDTH: f32 = 188.0;
pub const TOP_RULER_HEIGHT: f32 = 30.0;
pub const STATUS_BAR_HEIGHT: f32 = 24.0;
pub const SNAP_PX: f32 = 8.0;
pub const MIN_HANDLE_SCREEN_PX: f32 = 24.0;
