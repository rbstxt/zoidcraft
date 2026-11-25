//! Complete CM3 & Zoidium left-rail inspector panels (Media, Sequence, Edit, Objects,
//! Effects, Export, Restore, Plugins, Assets, About, Settings) wired directly to
//! `zc_session::Session`.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::PathBuf;

use egui::{
    Align2, Area, Color32, CornerRadius, CursorIcon, Frame, Id, Key, Mesh, Order, Painter, Pos2,
    Rect, ScrollArea, Sense, Shape, Stroke, StrokeKind, TextEdit, Ui, Vec2,
    scroll_area::ScrollBarVisibility,
};
use serde_json::{Value, json};
use zc_anim::{BezierSegment, InterpolationMode, eval_easing, eval_segment_bezier};
use zc_color::{BlendMode, Hsv, Rgb, Rgba};
use zc_plugin::{PluginCategory, PluginDescriptor};
use zc_project::{
    AssetKind, CustomProperty, EffectCategory, EffectInstance, EffectKind, GradientStop, LayerKind,
    LayerMask, Marker, MaskMode, MaterialSpec, Object3D, Object3DKind, PresetShapeKind,
    PropertyTrack, PropertyValue, SequenceControlKind, Uuid, create_effect_by_id,
    standard_effect_catalog, zoidium_effect_catalog,
};
use zc_render::shape_plus::{
    ALL_SHAPE_PLUS_KINDS, add_shape_plus_item, default_shape_plus_payload,
    duplicate_shape_plus_item, import_svg_to_shape_plus, parse_shape_plus_document,
    remove_shape_plus_item, reorder_shape_plus_item, shape_plus_kind_title,
};
use zc_scene::all_particle_sprite_names;
use zc_session::{Command, Session, SessionClip, SessionObject3D, SessionProperty};
use zc_shader::{CompileOptions, fix_property_name, translate_glsl_es100_with_options};

use super::clipboard::{read_system_clipboard, write_system_clipboard};
use super::pro_editors::{
    LooksEditorState, NodeGraphEditorState, NodeGraphTargetKind, clamp_floating_window_in_screen,
    compute_clamped_editor_window_rect, consume_escape_for_floating_editor,
    mark_floating_editor_escape_consumed, show_pro_editor_windows,
};

use super::controls::{
    ButtonVariant, FloatingWindowManager, FloatingWindowSpec, FuzzyItem, FuzzyPickerEvent,
    FuzzyPickerState, KeyframeBarEvent, KeyframeBarState, PropertyColumnState, ScrubConfig,
    ScrubEvent, cm3_collapsible_section, cm3_collapsible_section_with_action,
    cm3_column_adjust_splitter, cm3_dropdown, cm3_floating_window, cm3_gradient_bar,
    cm3_hsva_color_picker, cm3_icon_button, cm3_keyframe_bar, cm3_prop_button, cm3_property_row,
    cm3_property_row_aligned, cm3_scrub_number, cm3_stopwatch_button, cm3_vector2,
    cm3_vector2_with_sep, cm3_vector3, fuzzy_score, paint_cm3_tree_box,
};
use super::layout::{SplitOrientation, SplitPanelState, begin_split_panel, show_split_pane};
use super::theme::{
    BG_BUTTON, BG_DROPDOWN_HOVER, BG_HOVER, BG_INPUT, BG_OPTION_ACTIVE, BG_OPTION_HOVER, BG_PANEL,
    BG_SUNKEN, BG_TITLE, BORDER_CONTROL, BORDER_FILTER, BORDER_OPTION_ACTIVE, BORDER_PANEL,
    BORDER_ROW_SOFT, Cm3Icon, Cm3Theme, DANGER_TEXT, EditorFontChoice, FONT_BADGE, FONT_BANNER,
    FONT_BODY, FONT_COMPACT, FONT_SMALL, FONT_TITLE, TEXT_BRIGHT, TEXT_DISABLED, TEXT_EMPTY,
    TEXT_FAINT, TEXT_MUTED, TEXT_OPTION_DESC, TEXT_PRIMARY, TEXT_STRONG, TOOLBAR_ICON_INACTIVE,
    ThemeProfileChoice, WARNING_BAR, WARNING_SURFACE, color32_with_alpha, paint_badge,
    paint_cm3_icon, rgb_to_color32,
};

mod edit_tab;
mod effects_inspector;
mod effects_tab;
mod export_tab;
mod media_tab;
mod objects_2d;
mod objects_extended;
mod objects_inspector;
mod objects_materials;
mod objects_particles;
mod objects_tab;
mod plugins_settings;
mod property_rows;
mod sequence_tab;
mod shared_widgets;
mod state;
#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub(crate) use export_tab::{
    allowed_audio_codecs, allowed_video_codecs, available_export_containers,
    sanitize_export_settings, supports_transparent_alpha,
};
pub use state::{
    AppSettings, ExportAudioCodec, ExportCm3Preset, ExportContainer, ExportJobStatus, ExportMode,
    ExportQueueJob, ExportRateControl, ExportSettings, ExportStatus, ExportVideoCodec, PanelAction,
    PanelsState, ProjectAction, RecoverySnapshotInfo, SideTab,
};

use self::edit_tab::show_edit_tab;
use self::effects_tab::show_effects_tab;
use self::export_tab::show_export_tab;
use self::media_tab::{show_assets_tab, show_media_tab};
use self::objects_tab::show_objects_tab;
use self::plugins_settings::{
    show_about_tab, show_plugins_tab, show_restore_tab, show_settings_tab,
};
use self::property_rows::show_stopwatch_context_menu;
use self::sequence_tab::show_sequence_tab;
use self::shared_widgets::show_inspector_floating_windows;
use self::state::{clear_inspector_tx_state, commit_inspector_tx_if_open};

/// Renders the currently active CM3 / Zoidium side panel inside `panel_rect`.
#[allow(clippy::too_many_arguments)]
pub fn show_side_panel(
    ui: &mut Ui,
    tab: SideTab,
    panel_rect: Rect,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &mut Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    state.inspector_tx_owner_seen = false;
    state.inspector_root_indent = false;
    if !session.history.is_in_transaction() {
        clear_inspector_tx_state(state);
    }

    // Track history revisions so the Restore / History tab displays a recent activity list.
    let cur_rev = session.history.revision();
    if cur_rev != state.last_history_rev {
        state.last_history_rev = cur_rev;
        if let Some(lbl) = session.history.peek_undo_label() {
            state.history_log.push_front((cur_rev, lbl.to_owned()));
            while state.history_log.len() > 32 {
                state.history_log.pop_back();
            }
        }
    }

    match tab {
        SideTab::Media => show_media_tab(ui, session, theme, state, actions),
        SideTab::Sequence => {
            show_sequence_tab(
                ui,
                panel_rect,
                viewport_rect,
                session,
                theme,
                state,
                actions,
            );
        }
        SideTab::Edit => {
            show_edit_tab(
                ui,
                panel_rect,
                viewport_rect,
                session,
                theme,
                state,
                actions,
            );
        }
        SideTab::Objects => {
            show_objects_tab(
                ui,
                panel_rect,
                viewport_rect,
                session,
                theme,
                state,
                actions,
            );
        }
        SideTab::Effects => {
            show_effects_tab(
                ui,
                panel_rect,
                viewport_rect,
                session,
                theme,
                state,
                actions,
            );
        }
        SideTab::Export => {
            show_export_tab(ui, viewport_rect, session, theme, state, actions);
        }
        SideTab::Restore => show_restore_tab(ui, session, theme, state, actions),
        SideTab::Plugins => show_plugins_tab(ui, session, theme, state),
        SideTab::Assets => show_assets_tab(ui, session, theme, state, actions),
        SideTab::About => show_about_tab(ui, session, theme),
        SideTab::Settings => show_settings_tab(ui, viewport_rect, session, theme, state),
    }

    // If an inspector widget owning an open undo transaction was unmounted mid-edit
    // (e.g. collapsed section, tab switch, selection change, or read-only flip), close cleanly.
    if state.inspector_tx_owner.is_some() && !state.inspector_tx_owner_seen {
        commit_inspector_tx_if_open(session, state);
    }

    // Floating editors spawned from inspectors (Custom Shader GLSL editor, Expression editor)
    show_inspector_floating_windows(ui.ctx(), viewport_rect, session, theme, state);

    // Right-click stopwatch context menu (Edit Expression / Clear Expression / Toggle Keyframe)
    show_stopwatch_context_menu(ui.ctx(), viewport_rect, session, theme, state);
}
