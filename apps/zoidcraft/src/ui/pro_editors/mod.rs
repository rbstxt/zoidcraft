//! Interactive Pro Editor floating windows for ZoidCraft:
//! - **Node Graph Editor** (`NodeGraphEditorState`, [`show_node_graph_editor_window`]):
//!   Visual node-based editor supporting both 2D `nodeeffect` graphs (GLSL ES 1.00 codegen via
//!   `zc_shader::compile_node_effect_to_glsl`) and 3D `NodeMaterial` procedural PBR graphs
//!   (texture map baking via `zc_shader::bake_node_material_maps`).
//! - **Color Looks Setup** (`LooksEditorState`, [`show_looks_editor_window`]):
//!   5-stage color grading pipeline editor (**Subject / Matte / Lens / Camera / Post**) covering
//!   all 29 Color Looks tools, presets, reordering, and live parameter editing.
//! - **Retro & Glitch FX Setup** (`EffectSetupState`, [`show_effect_setup_window`]):
//!   Dedicated floating setup windows for **VHS Setup**, **Datamosh Setup**, **ASCII Setup**,
//!   and **Tracery Setup** with presets, deterministic seeded bursts, and atomic Undo.
//! - **3D Scene Designers** (`LensFlareOptionsState`, `ParticleDesignerState`):
//!   Floating **Lens Flares Options** browser/inspector and 3-column **Particle & Light Designer**
//!   for Particular / Form / Plexus.

use crate::ui::controls::FloatingWindowManager;
use crate::ui::theme::Cm3Theme;
use egui::Rect;
use zc_session::Session;

mod effect_setup;
mod floating;
mod looks;
mod looks_catalog;
mod node_graph;
mod node_graph_catalog;
mod node_material_catalog;
mod scene_designers;
#[cfg(test)]
mod tests;

pub use effect_setup::*;
pub use floating::*;
pub use looks::*;
pub use looks_catalog::*;
pub use node_graph::*;
pub use node_graph_catalog::*;
pub use node_material_catalog::*;
pub use scene_designers::*;

// ===========================================================================
// Shared Entry Point
// ===========================================================================

/// Renders all open Pro Editor floating windows.
#[allow(clippy::too_many_arguments)]
pub fn show_pro_editor_windows(
    ctx: &egui::Context,
    viewport_rect: Rect,
    floating_windows: &mut FloatingWindowManager,
    node_editor: &mut NodeGraphEditorState,
    looks_editor: &mut LooksEditorState,
    effect_setup: &mut EffectSetupState,
    lens_flare_options: &mut LensFlareOptionsState,
    particle_designer: &mut ParticleDesignerState,
    session: &mut Session,
    theme: &Cm3Theme,
) {
    show_node_graph_editor_window(
        ctx,
        viewport_rect,
        floating_windows,
        node_editor,
        session,
        theme,
    );
    show_looks_editor_window(
        ctx,
        viewport_rect,
        floating_windows,
        looks_editor,
        session,
        theme,
    );
    show_effect_setup_window(
        ctx,
        viewport_rect,
        floating_windows,
        effect_setup,
        session,
        theme,
    );
    show_lens_flare_options_window(
        ctx,
        viewport_rect,
        floating_windows,
        lens_flare_options,
        session,
        theme,
    );
    show_particle_designer_window(
        ctx,
        viewport_rect,
        floating_windows,
        particle_designer,
        session,
        theme,
    );
}
