use crate::ui::controls::{paint_checkerboard, parse_css_rgba};
use crate::ui::theme::rgba_to_color32;
use zc_shader::shader_plus::{
    ConditionClause, MAX_PALETTE_COLORS, RenameRule, VisibilityCondition, ZoidiumRenameRule,
    ZoidiumVisibilityWhen,
};

use super::effects_tab::{
    DEFAULT_CUSTOM_SHADER_GLSL, collect_effect_group_child_rows, delete_effect_and_select_adjacent,
    find_effect_group_child, find_effect_group_custom_properties,
    render_effect_group_custom_properties, show_nested_effect_group_child_inspector,
};
use super::media_tab::clip_kind_for_asset;
use super::property_rows::{
    eval_prop_str, eval_prop_value, lookup_session_prop_ref, panel_property_row,
    resolve_kf_write_frame, show_session_color_prop, show_session_color_rgba_prop,
    show_session_curve_prop, show_session_number_prop, show_session_option_prop,
    show_session_option_prop_ex, show_session_text_prop, show_session_vec2_prop,
    show_session_vec3_prop, show_session_vec4_prop, show_session_vec4_prop_ex,
};
use super::shared_widgets::{
    empty_panel_message, humanize_prop_label, property_header_with_actions,
};
use super::state::{
    ShaderEditorTarget, begin_inspector_tx, clear_inspector_tx_state,
    close_existing_tx_before_start, commit_inspector_tx_if_open, copy_pz_json_to_clipboard,
    owns_inspector_tx, paste_pz_json_from_clipboard, run_atomic_commit_outside_foreign_tx,
    touch_inspector_tx_owner,
};
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn show_effect_inspector(
    ui: &mut Ui,
    inspector_rect: Rect,
    viewport_rect: Rect,
    clip: &SessionClip,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let Some(fx_id) = session.selection.effect_ids.first().cloned() else {
        empty_panel_message(ui, "select an effect", theme);
        return;
    };
    let Some(fx) = clip.effects.iter().find(|e| e.id == fx_id).cloned() else {
        if let Some(child_fx) = find_effect_group_child(session, &clip.id, &fx_id) {
            show_nested_effect_group_child_inspector(
                ui,
                inspector_rect,
                viewport_rect,
                clip,
                &child_fx,
                session,
                theme,
                state,
                actions,
            );
            return;
        }
        empty_panel_message(ui, "select an effect", theme);
        return;
    };

    cm3_column_adjust_splitter(
        ui,
        Id::unique("fx_inspector_col_splitter"),
        inspector_rect,
        &mut state.prop_columns,
    );

    let local_frame = session.playback.current_frame - clip.start;
    let clip_start = clip.start;

    ScrollArea::vertical()
        .id_salt(("fx_inspector_scroll", &fx.id))
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            let (root_open, hdr_act) = property_header_with_actions(
                ui,
                Id::unique(("fx_root_hdr", &fx.id)),
                &fx.name,
                state.prop_columns.label_ratio,
                theme,
            );
            if hdr_act.copy_json {
                let res = session.copy_effects_as_pz_json(&clip.id, std::slice::from_ref(&fx.id));
                copy_pz_json_to_clipboard(
                    ui.ctx(),
                    res,
                    "Copied effect JSON to clipboard",
                    actions,
                );
            }
            if hdr_act.paste_json {
                paste_pz_json_from_clipboard(
                    session,
                    state,
                    actions,
                    "effects",
                    Some(clip.id.clone()),
                    None,
                    None,
                    None,
                    "effect JSON or GLSL",
                );
            }
            if hdr_act.reset {
                let _ = session.execute(Command::ResetPropertyList {
                    target_id: fx.id.clone(),
                });
            }
            if hdr_act.duplicate {
                let _ = session.execute(Command::DuplicateEffect {
                    clip_id: clip.id.clone(),
                    effect_id: fx.id.clone(),
                });
            }
            if hdr_act.remove {
                commit_inspector_tx_if_open(session, state);
                delete_effect_and_select_adjacent(session, &clip.id, &fx.id);
                return;
            }
            if !root_open {
                return;
            }
            state.inspector_root_indent = true;

            // Always show Enabled toggle
            show_session_option_prop(
                ui,
                session,
                &fx.id,
                "enabled",
                "Enabled",
                &["off", "on"],
                1,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );

            // Effect Group dedicated inspector
            if fx.kind == "group" {
                let child_count =
                    collect_effect_group_child_rows(session, &clip.id, &fx.id, &BTreeSet::new())
                        .len();
                panel_property_row(
                    ui,
                    "Group type",
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "Effect Group ({child_count} child{})",
                                if child_count == 1 { "" } else { "ren" }
                            ))
                            .font(theme.font(FONT_BODY))
                            .color(TEXT_MUTED),
                        );
                    },
                );
                let group_custom_props =
                    find_effect_group_custom_properties(session, &clip.id, &fx.id);
                let rendered_cp_names: BTreeSet<String> =
                    group_custom_props.iter().map(|c| c.name.clone()).collect();
                if !group_custom_props.is_empty() {
                    render_effect_group_custom_properties(
                        ui,
                        session,
                        &clip.id,
                        &fx.id,
                        &group_custom_props,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                let extra_keys: Vec<(String, Value)> = fx
                    .properties
                    .iter()
                    .filter(|(k, _)| {
                        k.as_str() != "enabled" && !rendered_cp_names.contains(k.as_str())
                    })
                    .filter_map(|(k, p)| {
                        let val = p
                            .keyframes
                            .first()
                            .map_or_else(|| p.default_value.clone(), |kf| kf.value.clone());
                        if val.is_null() {
                            None
                        } else {
                            Some((k.clone(), val))
                        }
                    })
                    .collect();
                for (key, val) in extra_keys {
                    render_dynamic_json_param(
                        ui,
                        session,
                        &fx.id,
                        &key,
                        &key,
                        &val,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                return;
            }

            // Custom Shader specialized inspector + Shader+ controls
            if fx.kind == "customshader" {
                let cur_code = eval_prop_str(
                    fx.properties.get("fragShader"),
                    local_frame,
                    DEFAULT_CUSTOM_SHADER_GLSL,
                );
                let open_shader = panel_property_row(
                    ui,
                    "Fragment shader",
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        let mut clicked = false;
                        ui.horizontal(|ui| {
                            if cm3_icon_button(
                                ui,
                                Cm3Icon::Code,
                                17.0,
                                theme.value(),
                                "Open shader code editor",
                                theme,
                            ) || cm3_prop_button(
                                ui,
                                "Edit shader",
                                ButtonVariant::Normal,
                                false,
                                theme,
                            )
                            .clicked()
                            {
                                clicked = true;
                            }
                        });
                        clicked
                    },
                );
                if open_shader {
                    state.shader_editor = Some(ShaderEditorTarget {
                        effect_id: fx.id.clone(),
                        property: "fragShader".to_owned(),
                        effect_name: fx.name.clone(),
                        code: cur_code.clone(),
                        font_px: 14.0,
                        error_message: None,
                    });
                    state.shader_editor_just_opened = true;
                    let spec = FloatingWindowSpec::new("shader_editor", "Shader Editor")
                        .with_subtitle(fx.name.clone())
                        .with_default_size(Vec2::new(560.0, 440.0));
                    state.floating_windows.open_or_focus(&spec, viewport_rect);
                }

                // Shader+ Custom Property builder & live uniform list
                let shader_plus = session.plugins.is_enabled("shader-plus");
                let sec_title = if shader_plus {
                    "Custom Properties (Shader+)"
                } else {
                    "Custom Properties"
                };
                state.inspector_root_indent = false;
                cm3_collapsible_section(
                    ui,
                    Id::unique("shader_custom_props"),
                    sec_title,
                    true,
                    theme,
                    |ui| {
                        let prop_kinds: &[&str] = if shader_plus {
                            &[
                                "Slider",
                                "Vector2",
                                "Vector3",
                                "Color (RGB)",
                                "Color + Alpha",
                                "Toggle",
                                "Angle (deg)",
                                "Seed",
                                "Dropdown",
                                "StaticDropdown",
                                "Gradient",
                                "Curve",
                                "Palette",
                                "Text",
                                "Path",
                                "TimeSample",
                                "LayerSource",
                            ]
                        } else {
                            &["Slider", "Vector2", "Vector3", "Color (RGB)"]
                        };
                        if state.new_shader_prop_kind >= prop_kinds.len() {
                            state.new_shader_prop_kind = 0;
                        }
                        Frame::NONE
                            .inner_margin(egui::Margin::symmetric(6, 4))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 4.0;
                                    let name_w = (ui.available_width() - 148.0).clamp(56.0, 110.0);
                                    Frame::NONE
                                        .fill(BG_INPUT)
                                        .stroke(Stroke::new(1.0, BORDER_CONTROL))
                                        .inner_margin(egui::Margin::symmetric(4, 2))
                                        .show(ui, |ui| {
                                            ui.add(
                                                TextEdit::singleline(
                                                    &mut state.new_shader_prop_name,
                                                )
                                                .frame(Frame::NONE)
                                                .font(theme.font(FONT_SMALL))
                                                .text_color(theme.value())
                                                .desired_width(name_w),
                                            );
                                        });
                                    cm3_dropdown(
                                        ui,
                                        Id::unique("shader_new_prop_kind"),
                                        &mut state.new_shader_prop_kind,
                                        prop_kinds,
                                        false,
                                        viewport_rect,
                                        theme,
                                    );
                                    if cm3_icon_button(
                                        ui,
                                        Cm3Icon::Add,
                                        17.0,
                                        TEXT_PRIMARY,
                                        "add uniform property",
                                        theme,
                                    ) {
                                        let raw_name = state.new_shader_prop_name.trim();
                                        if !raw_name.is_empty() {
                                            let prop_key = fix_property_name(raw_name);
                                            if !prop_key.is_empty()
                                                && prop_key != "enabled"
                                                && prop_key != "fragShader"
                                            {
                                                let kind_idx = state.new_shader_prop_kind;
                                                let default_val =
                                                    shader_plus_default_value(kind_idx);
                                                let meta_json =
                                                    shader_plus_default_metadata(kind_idx);
                                                state.shader_prop_kinds.insert(
                                                    (fx.id.clone(), prop_key.clone()),
                                                    kind_idx,
                                                );
                                                let started_tx =
                                                    !session.history.is_in_transaction()
                                                        && session
                                                            .begin_transaction("add custom uniform")
                                                            .is_ok();
                                                let _ =
                                                    session.execute(Command::SetPropertyValue {
                                                        target_id: fx.id.clone(),
                                                        property: prop_key.clone(),
                                                        value: default_val,
                                                    });
                                                let _ =
                                                    session.execute(Command::SetPropertyValue {
                                                        target_id: fx.id.clone(),
                                                        property: format!("__meta.{prop_key}"),
                                                        value: meta_json,
                                                    });
                                                if started_tx && session.history.is_in_transaction()
                                                {
                                                    let _ = session.commit_transaction(true);
                                                }
                                            }
                                        }
                                    }
                                });
                            });

                        let (num_values, str_values) = collect_effect_eval_values(&fx, local_frame);
                        let custom_keys: Vec<(String, Value)> = fx
                            .properties
                            .iter()
                            .filter(|(k, _)| {
                                !matches!(
                                    k.as_str(),
                                    "enabled" | "fragShader" | "vertShader" | "name"
                                )
                            })
                            .filter_map(|(k, p)| {
                                let val = p
                                    .keyframes
                                    .first()
                                    .map_or_else(|| p.default_value.clone(), |kf| kf.value.clone());
                                if val.is_null() {
                                    None
                                } else {
                                    Some((k.clone(), val))
                                }
                            })
                            .collect();

                        for (key, val) in custom_keys {
                            let parsed_meta = session
                                .document()
                                .property_metadata(&fx.id, &key)
                                .map(parse_shader_plus_metadata)
                                .unwrap_or_default();
                            if let Some(meta_kind) = parsed_meta.kind_idx {
                                state
                                    .shader_prop_kinds
                                    .entry((fx.id.clone(), key.clone()))
                                    .or_insert(meta_kind);
                            }
                            if let Some(ref vis_cond) = parsed_meta.visible_when
                                && !evaluate_shader_visibility(vis_cond, &num_values, &str_values)
                            {
                                continue;
                            }
                            let base_label = parsed_meta
                                .label
                                .as_deref()
                                .unwrap_or_else(|| key.strip_prefix("u_").unwrap_or(&key));
                            let resolved_label = resolve_shader_label(
                                base_label,
                                &parsed_meta.rename_when,
                                &num_values,
                                &str_values,
                            );
                            let kind_hint = parsed_meta.kind_idx.or_else(|| {
                                state
                                    .shader_prop_kinds
                                    .get(&(fx.id.clone(), key.clone()))
                                    .copied()
                            });
                            render_custom_shader_uniform(
                                ui,
                                session,
                                &fx.id,
                                &key,
                                resolved_label,
                                &val,
                                kind_hint,
                                &parsed_meta,
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );
                        }
                    },
                );
                return;
            }

            // Standard CM3 & Zoidium plugin effects: look up canonical schema via `create_effect_by_id`
            let base_id = fx.kind.rsplit('/').next().unwrap_or(fx.kind.as_str());
            if let Some((setup_kind, btn_label)) = match base_id {
                "vhs" => Some((EffectSetupKind::Vhs, "Open VHS Setup")),
                "datamosh" => Some((EffectSetupKind::Datamosh, "Open Datamosh Setup")),
                "ascii" => Some((EffectSetupKind::Ascii, "Open ASCII Setup")),
                "tracery" => Some((EffectSetupKind::Tracery, "Open Tracery Setup")),
                _ => None,
            } {
                let open_clicked = panel_property_row(
                    ui,
                    "Setup",
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_prop_button(ui, btn_label, ButtonVariant::Primary, true, theme)
                            .clicked()
                    },
                );
                if open_clicked {
                    state.effect_setup_editor.open_for_effect(
                        setup_kind,
                        fx.id.clone(),
                        fx.name.clone(),
                    );
                    let pref = setup_kind.default_size();
                    let (_, clamped_sz) =
                        compute_clamped_editor_window_rect(viewport_rect, pref.x, pref.y);
                    let spec = FloatingWindowSpec::new(setup_kind.window_id(), setup_kind.title())
                        .with_subtitle(setup_kind.subtitle())
                        .with_default_size(clamped_sz);
                    state.floating_windows.open_or_focus(&spec, viewport_rect);
                }
            }
            if let Some(template) = create_effect_by_id(base_id) {
                for (param_key, track) in &template.params {
                    render_template_param(
                        ui,
                        session,
                        &fx.id,
                        base_id,
                        param_key,
                        &track.value,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
            } else {
                // Fallback: render any dynamic properties present on the SessionEffect
                let keys: Vec<(String, Value)> = fx
                    .properties
                    .iter()
                    .filter(|(k, _)| k.as_str() != "enabled")
                    .filter_map(|(k, p)| {
                        let val = p
                            .keyframes
                            .first()
                            .map_or_else(|| p.default_value.clone(), |kf| kf.value.clone());
                        if val.is_null() {
                            None
                        } else {
                            Some((k.clone(), val))
                        }
                    })
                    .collect();
                if keys.is_empty() {
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("no additional parameters")
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_FAINT),
                    );
                } else {
                    for (key, val) in keys {
                        render_dynamic_json_param(
                            ui,
                            session,
                            &fx.id,
                            &key,
                            &key,
                            &val,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                }
            }
        });
}

pub(super) fn template_option_labels(effect_kind: &str, key: &str) -> &'static [&'static str] {
    match (effect_kind, key) {
        ("mirror", "mode") => &[
            "left",
            "right",
            "top",
            "bottom",
            "left + top",
            "right + top",
            "left + bottom",
            "right + bottom",
        ],
        ("static", "blending") => &["none", "add", "subtract", "multiply"],
        ("sglow", "combine") => &["multiply", "add", "screen", "difference", "overlay"],
        ("sglow", "quality") => &["fast", "balanced", "high", "ultra"],
        ("sglow", "show") => &["result", "threshold"],
        ("sglow", "edgeMode") => &["transparent", "reflect"],
        ("sglow", "opacity") => &["all opaque", "normal", "as premult"],
        ("sglow", "blurEngine" | "engine") => &["performance", "ae match 3-box"],
        ("sglow", "atmosphereModel") => &["ae style", "legacy clouds", "fine dust"],
        ("sglow", "timeMode") => &["sequence", "manual"],
        ("echo", "operator") => &[
            "add",
            "maximum",
            "minimum",
            "screen",
            "composite in front",
            "composite in back",
            "blend",
        ],
        ("echo", "composite") => &["over", "behind", "add", "screen", "maximum"],
        ("echo-legacy", "mode") => &["smear", "maximum", "additive", "screen"],
        ("ascii", "colorMode") => &["band", "footage", "duotone", "mono", "invert"],
        ("ascii", "fontFamily") => &["monospace", "sans", "serif", "condensed", "pixel"],
        ("ascii", "charset") => &["standard", "blocks", "detailed", "minimal", "custom"],
        ("ascii", "noiseType") => &["perlin", "simplex", "fractal", "turbulence"],
        ("datamosh", "algorithm") => DATAMOSH_ALGORITHM_ITEMS,
        ("datamosh", "motionFamily" | "motion") => DATAMOSH_MOTION_ITEMS,
        ("datamosh", "lookMode") => &["classic", "smear", "echo"],
        ("smartmotionblur", "view") => &["result", "motion vectors", "confidence"],
        ("smartmotionblur", "reconstruction") => &["automatic", "optical flow", "shutter samples"],
        ("smartmotionblur", "analysisFrames" | "analysis") => &["prev & next", "prev", "next"],
        ("smartmotionblur", "quality") => &["low", "medium", "high"],
        ("smart-motion-blur-lite", "quality") => &["full", "1/2", "1/4", "1/8"],
        ("smart-motion-blur-lite", "trackFrame") => &["previous", "next"],
        ("cardwipe", "flipAxis") => &["X", "Y", "Random"],
        ("cardwipe", "flipDirection") => &["Positive", "Negative", "Random"],
        ("cardwipe", "flipOrder") => &[
            "Left to Right",
            "Right to Left",
            "Top to Bottom",
            "Bottom to Top",
            "Top-Left to Bottom-Right",
            "Top-Right to Bottom-Left",
            "Bottom-Left to Top-Right",
            "Bottom-Right to Top-Left",
            "Gradient",
        ],
        ("cardwipe", "backLayer") => &["Color", "Transparent"],
        ("timecode", "displayFormat") => &[
            "SMPTE",
            "Frames",
            "Feet + Frames 35mm",
            "Feet + Frames 16mm",
        ],
        ("timecode", "timeSource") => &["Layer Source", "Composition", "Custom"],
        ("ccpixelpolly", "object") => &[
            "Triangles (flat)",
            "Triangles (textured)",
            "Squares (flat)",
            "Squares (textured)",
        ],
        ("ccpageturn", "render") => &["front & back", "back page", "front page"],
        ("ccpageturn", "backSource") => &["mirrored", "paper color"],
        ("radialshadow", "render") => &["regular", "glass"],
        ("timedifference", "alphaChannel") => &[
            "source only",
            "target only",
            "average",
            "max",
            "full on",
            "lightness",
            "brightest",
            "difference",
            "difference (white)",
        ],
        ("colordiffkey", "view") => &[
            "final output",
            "source only",
            "corrected matte",
            "uncorrected matte",
            "corrected partial A",
            "corrected partial B",
        ],
        ("colordiffkey", "colorMatchingAccuracy") => &["faster", "more accurate"],
        ("advspill", "method") => &["standard", "ultra"],
        ("linearcolorkey", "view") => &["final", "source", "matte"],
        ("linearcolorkey", "matchColors") => &["RGB", "Hue", "Chroma"],
        ("linearcolorkey", "keyOperation") => &["key out", "key in"],
        ("radiowaves", "waveType") => &["polygon", "circle"],
        ("radiowaves", "profile") => &[
            "square",
            "taper front",
            "taper back",
            "taper both",
            "sine",
            "sawtooth out",
            "sawtooth in",
        ],
        ("numbers", "format") => &[
            "number",
            "number (leading zeroes)",
            "timecode 30",
            "timecode 25",
            "timecode 24",
            "time (HH:MM:SS)",
            "numerical date",
            "short date",
            "long date",
            "hexadecimal",
        ],
        ("numbers", "displayOptions") => {
            &["fill", "stroke", "fill over stroke", "stroke over fill"]
        }
        ("tracery", "detectionQuality") => &["low", "medium", "high", "extreme"],
        ("tracery", "boxShape") => &["rectangle", "square", "circle", "ellipse"],
        ("tracery", "boxFill") => &["none", "solid", "diagonal hatch", "invert"],
        ("tracery", "markerShape") => &["dot", "plus", "cross", "polygon"],
        ("tracery", "gridMode") => &["cartesian", "edge"],
        ("tracery", "lineType") => &["spline", "pcb traces", "smooth bend", "step bend"],
        ("tracery", "labelMode") => &["coordinates", "dimensions", "area", "node id", "custom"],
        ("shutter", "mode") => &["linear", "double sided", "radial"],
        ("mask", "mode") => &["rectangle", "ellipse"],
        ("shake", "shake.mode") | ("shake", "mode") => &["rotate", "move"],
        ("overlay", "repeat") => &["off", "on"],
        ("duplicate", "repeat") => &["tile", "reflect"],
        ("timeoffset", "units") => &["frames", "seconds"],
        ("timeremap", "mode") => &["speed", "remap"],
        ("timeremap", "frameBlending" | "blend") => &["off", "mix", "optical flow"],
        ("transform", "wrap") => &["tile", "reflect"],
        (_, "wrap") => &["clamp", "tile", "reflect"],
        (_, "mask") => &["result", "mask"],
        (_, "gradientType") => &["linear", "radial", "angular", "reflected", "diamond"],
        (_, "direction") => &["vertical", "horizontal"],
        (_, "affect") => &["shadows", "highlights"],
        (_, "cameraType") => &["perspective", "orthographic"],
        _ => &["off", "on"],
    }
}

pub(super) fn template_number_scrub_config_for_default(
    effect_kind: &str,
    key: &str,
    default_val: f64,
) -> ScrubConfig {
    let mut cfg = template_number_scrub_config(effect_kind, key);
    if default_val.is_finite() && cfg.max < default_val {
        let hi = if default_val <= 100.0 {
            100.0
        } else {
            default_val * 2.0
        };
        cfg = ScrubConfig::new(cfg.min.min(0.0), hi, 1.0, 1);
    }
    cfg
}

fn template_number_scrub_config(effect_kind: &str, key: &str) -> ScrubConfig {
    // Drop Shadow stores `angle` directly in degrees (0..360, default 135).
    if effect_kind == "dropshadow" && key == "angle" {
        return ScrubConfig::new(-3600.0, 3600.0, 1.0, 1);
    }
    // Angle parameters stored in radians, displayed & scrubbed in degrees
    if matches!(key, "rotation" | "angle")
        || (effect_kind == "directionalblur" && key == "direction")
    {
        return ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 1)
            .with_scale_factor(std::f64::consts::PI / 180.0);
    }

    match (effect_kind, key) {
        ("radialblurspin", "density") => ScrubConfig::new(-2.0, 2.0, 0.02, 3),
        // Zoidium plugin effects whose opacity/threshold/percentage params are authored in 0..100
        ("radiowaves" | "timecode" | "radialshadow", "opacity")
        | ("ccpageturn", "backPageOpacity")
        | ("tracery", "threshold") => ScrubConfig::new(0.0, 100.0, 1.0, 1),
        ("keycleaner", "alphaContrast") => ScrubConfig::new(100.0, 500.0, 1.0, 1),
        ("keycleaner", "additionalEdgeRadius") => ScrubConfig::new(0.0, 100.0, 0.5, 1),
        (
            _,
            "opacity"
            | "sourceOpacity"
            | "affectAlpha"
            | "glowFromAlpha"
            | "glowUnderSource"
            | "lightBackground"
            | "threshold"
            | "blendThreshold"
            | "soften"
            | "blendSoften"
            | "feather"
            | "covered"
            | "fade"
            | "decay"
            | "dither"
            | "density"
            | "splitBalance"
            | "vignetteStrength"
            | "vignetteRadius"
            | "grain"
            | "diffusion"
            | "chromaticAberration"
            | "trackingJitter"
            | "chromaBleed"
            | "noise"
            | "headSwitch"
            | "scanlines"
            | "atmosphereDetail"
            | "blurAmount",
        ) => ScrubConfig::new(0.0, 1.0, 0.01, 3),
        ("crop", "left" | "top" | "right" | "bottom") => ScrubConfig::new(0.0, 1.0, 0.01, 3),
        ("posterize", "regions") => ScrubConfig::new(2.0, 256.0, 1.0, 0),
        ("echo", "numberOfEchoes") | ("radialblurspin", "samples") => {
            ScrubConfig::new(1.0, 64.0, 1.0, 0)
        }
        ("posterizetime", "frameRate") => ScrubConfig::new(1.0, 240.0, 1.0, 1),
        ("jpegdamage", "quality") => ScrubConfig::new(0.01, 1.0, 0.01, 2),
        ("jpegdamage", "resFactor") => ScrubConfig::new(1.0, 8.0, 0.1, 2),
        ("jpegdamage", "blockSize")
        | ("datamosh", "blockSize")
        | ("tracery", "gridSize")
        | ("ascii", "cellSize") => ScrubConfig::new(2.0, 256.0, 1.0, 0),
        ("smart-motion-blur-lite", "sensitivity") => ScrubConfig::new(0.0, 1.0, 0.01, 2),
        ("smartmotionblur", "motionSensitivity" | "vectorSmoothness") => {
            ScrubConfig::new(0.0, 100.0, 1.0, 1)
        }
        ("smartmotionblur", "maxDisplacement") => ScrubConfig::new(1.0, 2048.0, 1.0, 0),
        ("sglow", "glowWidth") => ScrubConfig::new(0.0, 1000.0, 1.0, 1),
        ("fisheye", "amount") | ("pulse", "amplitude") | ("shake", "zoom") => {
            ScrubConfig::new(-1000.0, 1000.0, 1.0, 1)
        }
        ("brightnesscontrast", "brightness")
        | ("exposure", "exposure")
        | ("magiclooks", "exposure" | "temperature" | "tint") => {
            ScrubConfig::new(-10.0, 10.0, 0.05, 2)
        }
        ("brightnesscontrast", "contrast")
        | ("grayscale", "saturation")
        | ("magiclooks", "contrast" | "saturation" | "strength")
        | ("bloom", "amount" | "strength" | "radius")
        | ("anamorphicflare", "amount" | "strength" | "radius") => {
            ScrubConfig::new(0.0, 10.0, 0.05, 2)
        }
        (_, "seed" | "atmosphereSeed") => ScrubConfig::new(0.0, 100_000.0, 1.0, 2),
        (_, "delta" | "blur" | "softness") => ScrubConfig::new(0.0, 500.0, 0.1, 2),
        (_, "time") => ScrubConfig::new(-100_000.0, 100_000.0, 0.05, 3),
        _ => ScrubConfig::new(-10_000.0, 10_000.0, 0.05, 3),
    }
}

#[allow(clippy::too_many_arguments)]
fn render_template_param(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    effect_kind: &str,
    key: &str,
    default_val: &PropertyValue,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let display_label = humanize_prop_label(key);
    let lbl = display_label.as_str();
    match default_val {
        PropertyValue::Number(def) => {
            let cfg = template_number_scrub_config_for_default(effect_kind, key, *def);
            show_session_number_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                *def,
                cfg,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::Vec2(def) => {
            show_session_vec2_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                *def,
                ScrubConfig::new(-10_000.0, 10_000.0, 0.05, 3),
                key.contains("scale") || key.contains("size") || key.contains("multiplier"),
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::Vec3(def) => {
            let cfg = if key == "cameraRotation" || key == "imageRotation" {
                ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 1)
                    .with_scale_factor(std::f64::consts::PI / 180.0)
            } else {
                ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 3)
            };
            show_session_vec3_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                *def,
                cfg,
                false,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::Vec4(def) => {
            let (cfg, animatable) = match key {
                "shake.amplitude" | "shake.phase" => {
                    (ScrubConfig::new(-10_000.0, 10_000.0, 0.5, 1), false)
                }
                "shake.frequency" | "shake.noise" => (ScrubConfig::new(0.0, 100.0, 0.05, 2), false),
                "shake.smooth" => (ScrubConfig::new(0.0, 10.0, 1.0, 0), false),
                _ => (ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 2), true),
            };
            show_session_vec4_prop_ex(
                ui,
                session,
                target_id,
                key,
                lbl,
                *def,
                cfg,
                animatable,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::Color(def) => {
            show_session_color_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                *def,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::ColorRgba(def) => {
            show_session_color_rgba_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                *def,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::Option(def) => {
            let opts = template_option_labels(effect_kind, key);
            let static_option = matches!(
                key,
                "gradientType"
                    | "wrap"
                    | "cameraType"
                    | "quality"
                    | "blurEngine"
                    | "engine"
                    | "atmosphereModel"
                    | "edgeMode"
                    | "shake.mode"
                    | "mode"
                    | "combine"
                    | "composite"
                    | "show"
                    | "timeMode"
                    | "operator"
                    | "colorMode"
                    | "fontFamily"
                    | "charset"
                    | "noiseType"
                    | "algorithm"
                    | "motionFamily"
                    | "motion"
                    | "lookMode"
                    | "view"
                    | "reconstruction"
                    | "analysisFrames"
                    | "analysis"
                    | "trackFrame"
                    | "units"
                    | "frameBlending"
                    | "blend"
                    | "direction"
                    | "affect"
                    | "flipAxis"
                    | "flipDirection"
                    | "flipOrder"
                    | "backLayer"
                    | "displayFormat"
                    | "timeSource"
                    | "object"
                    | "render"
                    | "backSource"
                    | "alphaChannel"
                    | "colorMatchingAccuracy"
                    | "method"
                    | "matchColors"
                    | "keyOperation"
                    | "waveType"
                    | "profile"
                    | "format"
                    | "displayOptions"
                    | "detectionQuality"
                    | "boxShape"
                    | "boxFill"
                    | "markerShape"
                    | "gridMode"
                    | "lineType"
                    | "labelMode"
            );
            show_session_option_prop_ex(
                ui,
                session,
                target_id,
                key,
                lbl,
                opts,
                *def,
                !static_option,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::List(def) if key == "blending" => {
            let cur = eval_prop_str(
                lookup_session_prop_ref(session, target_id, key),
                local_frame,
                def,
            );
            let mut idx = BlendMode::ALL
                .iter()
                .position(|b| b.token() == cur)
                .unwrap_or(3);
            let labels: Vec<&str> = BlendMode::ALL.iter().map(|b| b.display_name()).collect();
            let changed = panel_property_row(
                ui,
                lbl,
                true,
                state,
                theme,
                |_| {},
                |ui| {
                    cm3_dropdown(
                        ui,
                        Id::unique((target_id, key)),
                        &mut idx,
                        &labels,
                        false,
                        viewport_rect,
                        theme,
                    )
                },
            );
            if changed {
                let _ = session.execute(Command::SetPropertyValue {
                    target_id: target_id.to_owned(),
                    property: key.to_owned(),
                    value: json!(BlendMode::ALL[idx].token()),
                });
            }
        }
        PropertyValue::Gradient(def) => {
            touch_inspector_tx_owner(state, target_id, key);
            let (animated, mut stops) = {
                let prop_opt = lookup_session_prop_ref(session, target_id, key);
                (
                    prop_opt.is_some_and(|p| p.animated),
                    eval_prop_value(prop_opt, local_frame)
                        .and_then(|v| serde_json::from_value::<Vec<GradientStop>>(v).ok())
                        .unwrap_or_else(|| def.clone()),
                )
            };
            let ev = panel_property_row(
                ui,
                lbl,
                true,
                state,
                theme,
                |_| {},
                |ui| {
                    cm3_gradient_bar(
                        ui,
                        Id::unique((target_id, key)),
                        &mut stops,
                        viewport_rect,
                        theme,
                    )
                },
            );
            let owns_grad_tx = owns_inspector_tx(state, target_id, key);
            if ev.cancelled {
                if owns_grad_tx {
                    if session.history.is_in_transaction() {
                        let _ = session.cancel_transaction();
                    }
                    clear_inspector_tx_state(state);
                }
            } else {
                if ev.changed && !ev.committed && !owns_grad_tx {
                    close_existing_tx_before_start(session, state);
                    begin_inspector_tx(session, state, target_id, key, format!("edit {key}"));
                }
                let owns_grad_tx = owns_inspector_tx(state, target_id, key);
                if ev.changed
                    && let Ok(v) = serde_json::to_value(&stops)
                {
                    let exec_grad = |session: &mut Session| {
                        let cmd = if animated {
                            Command::SetKeyframe {
                                target_id: target_id.to_owned(),
                                property: key.to_owned(),
                                frame: resolve_kf_write_frame(
                                    session,
                                    lookup_session_prop_ref(session, target_id, key),
                                    local_frame,
                                ),
                                value: v,
                                tween: None,
                            }
                        } else {
                            Command::SetPropertyValue {
                                target_id: target_id.to_owned(),
                                property: key.to_owned(),
                                value: v,
                            }
                        };
                        let _ = session.execute(cmd);
                    };
                    if ev.committed && !owns_grad_tx {
                        run_atomic_commit_outside_foreign_tx(session, state, exec_grad);
                    } else {
                        exec_grad(session);
                        if owns_grad_tx {
                            state.inspector_tx_has_edits = true;
                        }
                    }
                }
                if ev.committed && owns_grad_tx {
                    if session.history.is_in_transaction() {
                        let _ = session.commit_transaction(true);
                    }
                    clear_inspector_tx_state(state);
                }
            }
        }
        PropertyValue::Curve(def) => {
            show_session_curve_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                def,
                local_frame,
                state,
                theme,
            );
        }
        PropertyValue::Text(def) | PropertyValue::List(def) | PropertyValue::Shader(def) => {
            show_session_text_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                def,
                local_frame,
                state,
                theme,
            );
        }
        PropertyValue::Asset(def) => {
            let is_lut = key == "lut";
            let cur_id = eval_prop_str(
                lookup_session_prop_ref(session, target_id, key),
                local_frame,
                def.as_deref().unwrap_or(""),
            );

            // Build sorted project asset list for dropdown selection
            let mut assets: Vec<(String, String)> = session
                .project
                .assets
                .values()
                .map(|a| (a.id.clone(), a.filename.clone()))
                .collect();
            assets.sort_by(|a, b| {
                if is_lut {
                    let a_cube = a.1.to_ascii_lowercase().ends_with(".cube");
                    let b_cube = b.1.to_ascii_lowercase().ends_with(".cube");
                    b_cube.cmp(&a_cube).then_with(|| a.1.cmp(&b.1))
                } else {
                    a.1.cmp(&b.1)
                }
            });

            let mut option_labels: Vec<String> = Vec::with_capacity(assets.len() + 1);
            option_labels.push("(none)".to_owned());
            for (_, fname) in &assets {
                option_labels.push(fname.clone());
            }
            let mut selected_idx = if cur_id.is_empty() {
                0
            } else {
                assets
                    .iter()
                    .position(|(id, _)| id == &cur_id)
                    .map_or(0, |i| i + 1)
            };
            let label_refs: Vec<&str> = option_labels.iter().map(String::as_str).collect();

            let mut dd_changed = false;
            let mut clear_clicked = false;
            panel_property_row(
                ui,
                lbl,
                true,
                state,
                theme,
                |_| {},
                |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        dd_changed = cm3_dropdown(
                            ui,
                            Id::unique((target_id, key, "asset_dd")),
                            &mut selected_idx,
                            &label_refs,
                            false,
                            viewport_rect,
                            theme,
                        );
                        if cm3_icon_button(
                            ui,
                            Cm3Icon::Load,
                            17.0,
                            TEXT_PRIMARY,
                            if is_lut {
                                "Load .cube LUT file..."
                            } else {
                                "Choose asset file..."
                            },
                            theme,
                        ) {
                            actions.push(PanelAction::PickAssetForProperty {
                                target_id: target_id.to_owned(),
                                property: key.to_owned(),
                                is_lut,
                            });
                        }
                        if !cur_id.is_empty()
                            && cm3_icon_button(
                                ui,
                                Cm3Icon::Remove,
                                17.0,
                                TEXT_PRIMARY,
                                "Clear asset",
                                theme,
                            )
                        {
                            clear_clicked = true;
                        }
                    });
                },
            );

            if clear_clicked {
                let _ = session.execute(Command::SetPropertyValue {
                    target_id: target_id.to_owned(),
                    property: key.to_owned(),
                    value: Value::Null,
                });
            } else if dd_changed {
                let new_val = if selected_idx == 0 {
                    Value::Null
                } else {
                    json!(assets[selected_idx - 1].0)
                };
                let _ = session.execute(Command::SetPropertyValue {
                    target_id: target_id.to_owned(),
                    property: key.to_owned(),
                    value: new_val,
                });
            }
        }
        PropertyValue::Bool(def) => {
            show_session_option_prop(
                ui,
                session,
                target_id,
                key,
                lbl,
                &["off", "on"],
                u32::from(*def),
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        PropertyValue::Json(_) => {
            let btn_label = match effect_kind {
                "nodeeffect" => "Edit Node Graph",
                "magiclooks" => "Edit Color Looks",
                _ => "Edit JSON",
            };
            let clicked = panel_property_row(
                ui,
                lbl,
                true,
                state,
                theme,
                |_| {},
                |ui| cm3_prop_button(ui, btn_label, ButtonVariant::Primary, true, theme).clicked(),
            );
            if clicked {
                let cur_val = eval_prop_value(
                    lookup_session_prop_ref(session, target_id, key),
                    local_frame,
                );
                match effect_kind {
                    "nodeeffect" => {
                        state.node_graph_editor.open_for_target(
                            NodeGraphTargetKind::NodeEffect,
                            target_id,
                            key,
                            "Node Effect",
                            cur_val.as_ref(),
                        );
                        let (_, clamped_sz) =
                            compute_clamped_editor_window_rect(viewport_rect, 860.0, 560.0);
                        let spec =
                            FloatingWindowSpec::new("pro_node_graph_editor", "Node Graph Editor")
                                .with_default_size(clamped_sz);
                        state.floating_windows.open_or_focus(&spec, viewport_rect);
                    }
                    "magiclooks" => {
                        state.looks_editor.open_for_effect(
                            target_id,
                            "Color Looks",
                            cur_val.as_ref(),
                        );
                        let (_, clamped_sz) =
                            compute_clamped_editor_window_rect(viewport_rect, 860.0, 560.0);
                        let spec =
                            FloatingWindowSpec::new("pro_color_looks_editor", "Color Looks Setup")
                                .with_default_size(clamped_sz);
                        state.floating_windows.open_or_focus(&spec, viewport_rect);
                    }
                    _ => {
                        actions.push(PanelAction::StatusMessage(format!(
                            "{btn_label}: inspect via effect properties"
                        )));
                    }
                }
            }
        }
    }
}

/// Maximum inline text length rendered in a live `egui::TextEdit` property widget.
/// Values exceeding this threshold render a compact read-only summary label so multi-megabyte
/// strings never stall layout or drop frame rate.
pub(super) const MAX_INLINE_TEXT_CHARS: usize = 8_192;

/// Returns a compact `"<N KB text>"` summary when `text` exceeds [`MAX_INLINE_TEXT_CHARS`].
pub(super) fn oversized_text_summary(text: &str) -> Option<String> {
    if text.len() > MAX_INLINE_TEXT_CHARS && text.chars().nth(MAX_INLINE_TEXT_CHARS).is_some() {
        let kb = text.len().div_ceil(1024).max(1);
        Some(format!("<{kb} KB text>"))
    } else {
        None
    }
}

/// Detects the single-element default `GradientStop` placeholder produced when a CM3 `.pz`
/// shader custom property with `type: 5` (`bool` in Shader+) is imported without gradient stops.
pub(super) fn is_type5_bool_placeholder_value(val: &Value) -> bool {
    let Some(arr) = val.as_array() else {
        return false;
    };
    if arr.len() != 1 {
        return false;
    }
    let Some(obj) = arr[0].as_object() else {
        return false;
    };
    let pos_zero = obj
        .get("position")
        .or_else(|| obj.get("pos"))
        .and_then(Value::as_f64)
        .is_some_and(|p| p.abs() < 1e-6);
    let white_default = obj
        .get("color")
        .and_then(Value::as_str)
        .is_some_and(|s| s.eq_ignore_ascii_case("rgba(255,255,255,1.0)"));
    pos_zero && white_default
}

/// Looks up the imported CM3 `CustomProperty::prop_type` for a custom shader effect or
/// 3D custom shader material uniform.
fn lookup_custom_shader_prop_type(session: &Session, target_id: &str, key: &str) -> Option<u32> {
    fn find_in_objects(objects: &[Object3D], target_uuid: Uuid, key: &str) -> Option<u32> {
        let stripped = key
            .strip_prefix("material.")
            .or_else(|| key.strip_prefix("frontMaterial."))
            .or_else(|| key.strip_prefix("sideMaterial."))
            .unwrap_or(key);
        for obj in objects {
            if obj.id == target_uuid
                && let Some(MaterialSpec::CustomShader {
                    custom_properties, ..
                }) = obj.material.as_ref()
                && let Some(cp) = custom_properties
                    .iter()
                    .find(|c| c.name == key || c.name == stripped)
            {
                return cp.prop_type;
            }
            if let Object3DKind::Group { children, .. } = &obj.kind
                && let Some(found) = find_in_objects(children, target_uuid, key)
            {
                return Some(found);
            }
        }
        None
    }

    let target_uuid = Uuid::parse_str(target_id).ok()?;
    for track in &session.active_sequence().video_tracks {
        for clip in &track.clips {
            for eff in &clip.layer.effects {
                if eff.id == target_uuid
                    && let EffectKind::CustomShader {
                        custom_properties, ..
                    } = &eff.kind
                    && let Some(cp) = custom_properties.iter().find(|c| c.name == key)
                {
                    return cp.prop_type;
                }
            }
            if let LayerKind::Scene3D { objects, .. } = &clip.layer.kind
                && let Some(found) = find_in_objects(objects, target_uuid, key)
            {
                return Some(found);
            }
        }
    }
    None
}

/// Builds `(display_labels, asset_ids, selected_idx)` for a `sampler2D` (`prop_type == 8`)
/// shader uniform dropdown: `"None"` followed by all project image assets sorted by filename.
pub(super) fn collect_shader_image_asset_choices(
    session: &Session,
    current_asset_id: &str,
) -> (Vec<String>, Vec<String>, usize) {
    let mut images: Vec<(String, String)> = session
        .project
        .assets
        .values()
        .filter(|a| {
            a.kind == AssetKind::Image || clip_kind_for_asset(a.kind, &a.filename) == "image"
        })
        .map(|a| (a.filename.clone(), a.id.clone()))
        .collect();
    images.sort_by_key(|a| a.0.to_ascii_lowercase());

    let mut labels = Vec::with_capacity(images.len() + 2);
    let mut ids = Vec::with_capacity(images.len() + 2);
    labels.push("None".to_owned());
    ids.push(String::new());
    for (name, id) in images {
        labels.push(name);
        ids.push(id);
    }

    let trimmed = current_asset_id.trim();
    let selected_idx = if trimmed.is_empty() {
        0
    } else if let Some(pos) = ids.iter().position(|id| id == trimmed) {
        pos
    } else {
        let short: String = trimmed.chars().take(12).collect();
        labels.push(format!("{short}… (unresolved)"));
        ids.push(trimmed.to_owned());
        ids.len() - 1
    };

    (labels, ids, selected_idx)
}

/// Parsed Shader+ custom property metadata hydrated from `session.document().property_metadata`.
#[derive(Debug, Clone, Default)]
pub(super) struct ShaderPlusPropMeta {
    pub(super) kind_idx: Option<usize>,
    pub(super) label: Option<String>,
    pub(super) items: Vec<String>,
    pub(super) min: Option<f64>,
    pub(super) max: Option<f64>,
    pub(super) step: Option<f64>,
    pub(super) decimals: Option<u8>,
    pub(super) visible_when: Option<VisibilityCondition>,
    pub(super) rename_when: Vec<RenameRule>,
    pub(super) is_layer_source: bool,
}

/// Returns the default JSON value for one of the 17 Shader+ custom property kinds.
pub(super) fn shader_plus_default_value(kind_idx: usize) -> Value {
    match kind_idx {
        1 => json!([0.0, 0.0]),
        2 => json!([0.0, 0.0, 0.0]),
        3 => json!([1.0, 1.0, 1.0]),
        4 => json!([1.0, 1.0, 1.0, 1.0]),
        5 => json!(true),
        6 => json!(0.0),
        7 => json!(1),
        8 | 9 => json!(0),
        10 => json!([
            {"pos": 0.0, "color": [0.0, 0.0, 0.0], "opacity": 1.0},
            {"pos": 1.0, "color": [1.0, 1.0, 1.0], "opacity": 1.0}
        ]),
        11 => json!({
            "master": [[0.0, 0.0], [1.0, 1.0]],
            "rgb": [[0.0, 0.0], [1.0, 1.0]],
            "common": [[0.0, 0.0], [1.0, 1.0]]
        }),
        12 => json!([
            [1.0, 0.3, 0.3, 1.0],
            [0.3, 0.9, 0.5, 1.0],
            [0.2, 0.5, 1.0, 1.0]
        ]),
        13 => json!("Shader Text"),
        14 => json!({
            "vertices": [
                {"point": [0.15, 0.5], "in": [0.0, 0.0], "out": [0.0, 0.0]},
                {"point": [0.85, 0.5], "in": [0.0, 0.0], "out": [0.0, 0.0]}
            ],
            "closed": false
        }),
        15 => json!(0.0),
        16 => json!(""),
        _ => json!(0.5),
    }
}

/// Builds the persisted `__meta.<propName>` JSON object for one of the 17 Shader+ custom property
/// kinds so property kinds survive reload and are visible to `zc-render`.
pub(super) fn shader_plus_default_metadata(kind_idx: usize) -> Value {
    match kind_idx {
        1 => json!({ "type": { "dynamic": true, "zoidiumShader": { "kind": "vec2" } } }),
        2 => json!({ "type": { "dynamic": true, "zoidiumShader": { "kind": "vec3" } } }),
        3 => json!({ "type": { "dynamic": true, "zoidiumShader": { "kind": "color" } } }),
        4 => json!({ "type": { "dynamic": true, "zoidiumShader": { "kind": "colorAlpha" } } }),
        5 => json!({ "type": { "dynamic": true, "zoidiumShader": { "kind": "toggle" } } }),
        6 => json!({ "type": { "dynamic": true, "zoidiumShader": { "kind": "angle" } } }),
        7 => json!({
            "type": {
                "dynamic": true,
                "zoidiumShader": { "kind": "seed", "min": 0.0, "max": 100_000.0 }
            }
        }),
        8 => json!({
            "type": {
                "dynamic": true,
                "zoidiumShader": {
                    "kind": "dropdown",
                    "items": ["Option 1", "Option 2", "Option 3"]
                }
            }
        }),
        9 => json!({
            "type": {
                "dynamic": false,
                "zoidiumShader": {
                    "kind": "staticDropdown",
                    "items": ["Mode 1", "Mode 2", "Mode 3"]
                }
            }
        }),
        10 => json!({
            "type": {
                "dynamic": true,
                "zoidiumShader": {
                    "kind": "gradient",
                    "cyclic": false,
                    "repetitions": 1.0
                }
            }
        }),
        11 => json!({ "type": { "dynamic": true, "zoidiumShader": { "kind": "curve" } } }),
        12 => json!({ "type": { "dynamic": true, "zoidiumShader": { "kind": "palette" } } }),
        13 => json!({ "type": { "dynamic": true, "zoidiumShader": { "kind": "text" } } }),
        14 => json!({ "type": { "dynamic": true, "zoidiumShader": { "kind": "path" } } }),
        15 => json!({
            "type": {
                "dynamic": true,
                "zoidiumShader": { "kind": "timeSample", "unit": "frames" }
            }
        }),
        16 => json!({
            "type": {
                "dynamic": true,
                "_zoidiumShaderLayerSource": true,
                "zoidiumShader": { "kind": "layerSource" }
            }
        }),
        _ => json!({ "type": { "dynamic": true, "zoidiumShader": { "kind": "slider" } } }),
    }
}

/// Maps a persisted `zoidiumShader.kind` string into the 17-entry Shader+ kind index.
pub(super) fn shader_plus_kind_from_str(kind: &str) -> Option<usize> {
    match kind.trim().to_ascii_lowercase().as_str() {
        "slider" | "float" | "number" | "int" | "integer" => Some(0),
        "vec2" | "vector2" => Some(1),
        "vec3" | "vector3" => Some(2),
        "color" | "rgb" | "colorrgb" => Some(3),
        "coloralpha" | "color_alpha" | "rgba" | "vec4" | "vector4" => Some(4),
        "toggle" | "bool" | "boolean" => Some(5),
        "angle" => Some(6),
        "seed" => Some(7),
        "dropdown" | "select" | "enum" | "option" => Some(8),
        "staticdropdown" | "static_dropdown" => Some(9),
        "gradient" => Some(10),
        "curve" => Some(11),
        "palette" => Some(12),
        "text" => Some(13),
        "path" => Some(14),
        "timesample" | "time_sample" => Some(15),
        "layersource" | "layer_source" | "layerinput" | "layer_input" => Some(16),
        _ => None,
    }
}

fn parse_shorthand_visibility_condition(val: &Value) -> Option<VisibilityCondition> {
    let obj = val.as_object()?;
    let inner = obj.get("when").and_then(Value::as_object).unwrap_or(obj);
    let prop = inner
        .get("prop")
        .or_else(|| inner.get("property"))
        .or_else(|| inner.get("target_property"))
        .and_then(Value::as_str)?
        .trim();
    if prop.is_empty() {
        return None;
    }
    if let Some(expr) = inner
        .get("expr")
        .or_else(|| inner.get("condition"))
        .and_then(Value::as_str)
    {
        return VisibilityCondition::parse(prop, expr).ok();
    }
    if let Some(op) = inner.get("op").and_then(Value::as_str) {
        let clause = match op.trim() {
            "=" | "==" => {
                if let Some(n) = inner.get("value").and_then(Value::as_f64) {
                    ConditionClause::Exact(n)
                } else {
                    ConditionClause::ExactString(
                        inner.get("value").and_then(Value::as_str)?.to_owned(),
                    )
                }
            }
            "!=" => ConditionClause::NotEqual(inner.get("value").and_then(Value::as_f64)?),
            ">" => ConditionClause::Greater(inner.get("value").and_then(Value::as_f64)?),
            ">=" => ConditionClause::GreaterOrEqual(inner.get("value").and_then(Value::as_f64)?),
            "<" => ConditionClause::Less(inner.get("value").and_then(Value::as_f64)?),
            "<=" => ConditionClause::LessOrEqual(inner.get("value").and_then(Value::as_f64)?),
            "range" => {
                let min = inner.get("min").and_then(Value::as_f64)?;
                let max = inner.get("max").and_then(Value::as_f64)?;
                ConditionClause::Range {
                    min: min.min(max),
                    max: min.max(max),
                }
            }
            _ => return None,
        };
        return Some(VisibilityCondition {
            target_property: prop.to_owned(),
            clauses: vec![clause],
        });
    }
    if let Some(n) = inner.get("value").and_then(Value::as_f64) {
        return Some(VisibilityCondition {
            target_property: prop.to_owned(),
            clauses: vec![ConditionClause::Exact(n)],
        });
    }
    if let Some(s) = inner.get("value").and_then(Value::as_str) {
        return Some(VisibilityCondition {
            target_property: prop.to_owned(),
            clauses: vec![ConditionClause::ExactString(s.to_owned())],
        });
    }
    None
}

fn parse_visibility_condition_value(val: &Value) -> Option<VisibilityCondition> {
    if let Ok(z_when) = serde_json::from_value::<ZoidiumVisibilityWhen>(val.clone())
        && let Ok(cond) = VisibilityCondition::from_zoidium(&z_when)
    {
        return Some(cond);
    }
    if let Ok(direct) = serde_json::from_value::<VisibilityCondition>(val.clone())
        && !direct.target_property.trim().is_empty()
        && !direct.clauses.is_empty()
    {
        return Some(direct);
    }
    parse_shorthand_visibility_condition(val)
}

fn parse_rename_rule_value(val: &Value) -> Option<RenameRule> {
    if let Ok(z_rule) = serde_json::from_value::<ZoidiumRenameRule>(val.clone())
        && let Ok(rule) = RenameRule::from_zoidium(&z_rule)
    {
        return Some(rule);
    }
    if let Ok(direct) = serde_json::from_value::<RenameRule>(val.clone())
        && !direct.label.trim().is_empty()
        && !direct.condition.clauses.is_empty()
    {
        return Some(direct);
    }
    let obj = val.as_object()?;
    let label = obj
        .get("label")
        .or_else(|| obj.get("name"))
        .and_then(Value::as_str)?
        .trim();
    if label.is_empty() {
        return None;
    }
    let condition = parse_visibility_condition_value(val)?;
    Some(RenameRule {
        condition,
        label: label.to_owned(),
    })
}

/// Hydrates [`ShaderPlusPropMeta`] from a `SessionDocument::property_metadata` map.
pub(super) fn parse_shader_plus_metadata(meta: &BTreeMap<String, Value>) -> ShaderPlusPropMeta {
    let mut out = ShaderPlusPropMeta::default();
    let type_obj = meta.get("type").and_then(Value::as_object);
    let z_shader = type_obj
        .and_then(|t| t.get("zoidiumShader"))
        .and_then(Value::as_object)
        .or_else(|| meta.get("zoidiumShader").and_then(Value::as_object));

    let kind_str = z_shader
        .and_then(|s| s.get("kind"))
        .and_then(Value::as_str)
        .or_else(|| type_obj.and_then(|t| t.get("kind")).and_then(Value::as_str))
        .or_else(|| meta.get("kind").and_then(Value::as_str));
    out.kind_idx = kind_str.and_then(shader_plus_kind_from_str);

    let layer_flag = [
        "_zoidiumShaderLayerSource",
        "_zoidiumLayerSource",
        "isLayerInput",
        "is_layer_input",
    ]
    .iter()
    .any(|k| {
        type_obj
            .and_then(|t| t.get(*k))
            .and_then(Value::as_bool)
            .or_else(|| meta.get(*k).and_then(Value::as_bool))
            == Some(true)
    });
    out.is_layer_source = out.kind_idx == Some(16) || layer_flag;
    if out.is_layer_source && out.kind_idx.is_none() {
        out.kind_idx = Some(16);
    }

    out.label = z_shader
        .and_then(|s| s.get("label"))
        .or_else(|| type_obj.and_then(|t| t.get("label")))
        .or_else(|| meta.get("label"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);

    let items_val = z_shader
        .and_then(|s| s.get("items"))
        .or_else(|| type_obj.and_then(|t| t.get("items")))
        .or_else(|| meta.get("items"));
    if let Some(v) = items_val {
        if let Some(s) = v.as_str() {
            out.items = s
                .split(';')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(str::to_owned)
                .collect();
        } else if let Some(arr) = v.as_array() {
            for item in arr {
                if let Some(s) = item.as_str() {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        out.items.push(trimmed.to_owned());
                    }
                } else if let Some(s) = item
                    .get("label")
                    .or_else(|| item.get("name"))
                    .or_else(|| item.get("value"))
                    .and_then(Value::as_str)
                {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        out.items.push(trimmed.to_owned());
                    }
                }
            }
        }
    }

    let lookup_f64 = |field: &str| -> Option<f64> {
        z_shader
            .and_then(|s| s.get(field))
            .or_else(|| type_obj.and_then(|t| t.get(field)))
            .or_else(|| meta.get(field))
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
    };
    out.min = lookup_f64("min");
    out.max = lookup_f64("max");
    out.step = lookup_f64("step");
    out.decimals = z_shader
        .and_then(|s| s.get("decimals"))
        .or_else(|| type_obj.and_then(|t| t.get("decimals")))
        .or_else(|| meta.get("decimals"))
        .and_then(Value::as_u64)
        .map(|d| d.min(6) as u8);

    if let Some(vis_val) = z_shader
        .and_then(|s| s.get("visibleWhen").or_else(|| s.get("visibility")))
        .or_else(|| type_obj.and_then(|t| t.get("visibleWhen").or_else(|| t.get("visibility"))))
        .or_else(|| meta.get("visibleWhen").or_else(|| meta.get("visibility")))
    {
        out.visible_when = parse_visibility_condition_value(vis_val);
    }

    if let Some(ren_arr) = z_shader
        .and_then(|s| s.get("renameWhen").or_else(|| s.get("rename")))
        .or_else(|| type_obj.and_then(|t| t.get("renameWhen").or_else(|| t.get("rename"))))
        .or_else(|| meta.get("renameWhen").or_else(|| meta.get("rename")))
        .and_then(Value::as_array)
    {
        for entry in ren_arr {
            if let Some(rule) = parse_rename_rule_value(entry) {
                out.rename_when.push(rule);
            }
        }
    }

    out
}

/// Evaluates all properties on `fx` at `local_frame` into numeric and string lookup maps for
/// Shader+ `VisibilityCondition` and `RenameRule` evaluation.
pub(super) fn collect_effect_eval_values(
    fx: &zc_session::SessionEffect,
    local_frame: f64,
) -> (BTreeMap<String, f64>, BTreeMap<String, String>) {
    let mut nums = BTreeMap::new();
    let mut strs = BTreeMap::new();
    for (k, prop) in &fx.properties {
        let Some(val) = eval_prop_value(Some(prop), local_frame) else {
            continue;
        };
        let norm = fix_property_name(k);
        match val {
            Value::Number(n) => {
                if let Some(f) = n.as_f64()
                    && f.is_finite()
                {
                    nums.insert(k.clone(), f);
                    nums.insert(norm, f);
                }
            }
            Value::Bool(b) => {
                let f = if b { 1.0 } else { 0.0 };
                nums.insert(k.clone(), f);
                nums.insert(norm, f);
            }
            Value::String(s) => {
                if let Ok(f) = s.trim().parse::<f64>()
                    && f.is_finite()
                {
                    nums.insert(k.clone(), f);
                    nums.insert(norm.clone(), f);
                }
                strs.insert(k.clone(), s.clone());
                strs.insert(norm, s);
            }
            _ => {}
        }
    }
    (nums, strs)
}

/// Evaluates a Shader+ [`VisibilityCondition`] against evaluated numeric and string property maps.
pub(super) fn evaluate_shader_visibility(
    cond: &VisibilityCondition,
    num_values: &BTreeMap<String, f64>,
    str_values: &BTreeMap<String, String>,
) -> bool {
    let norm = fix_property_name(&cond.target_property);
    if let Some(s) = str_values
        .get(&cond.target_property)
        .or_else(|| str_values.get(&norm))
        && cond.matches_static_dropdown(s)
    {
        return true;
    }
    if let Some(&v) = num_values
        .get(&cond.target_property)
        .or_else(|| num_values.get(&norm))
    {
        return cond.matches_value(v);
    }
    false
}

/// Resolves the dynamic display label using Shader+ [`RenameRule`]s, falling back to `default_label`.
pub(super) fn resolve_shader_label<'a>(
    default_label: &'a str,
    rules: &'a [RenameRule],
    num_values: &BTreeMap<String, f64>,
    str_values: &BTreeMap<String, String>,
) -> &'a str {
    for rule in rules {
        if evaluate_shader_visibility(&rule.condition, num_values, str_values) {
            return rule.label.as_str();
        }
    }
    default_label
}

/// Builds `(display_labels, layer_refs, selected_idx)` for a Shader+ `LayerSource` dropdown:
/// `"None"` followed by all video tracks (with their clip names) writing `"track:id:<track_uuid>"`.
pub(super) fn collect_shader_layer_source_choices(
    session: &Session,
    current_ref: &str,
) -> (Vec<String>, Vec<String>, usize) {
    let tracks = &session.document().video_tracks;
    let mut labels = Vec::with_capacity(tracks.len() + 2);
    let mut refs = Vec::with_capacity(tracks.len() + 2);
    labels.push("None".to_owned());
    refs.push(String::new());

    for track in tracks {
        let ref_token = format!("track:id:{}", track.id);
        let label = match track.clips.len() {
            0 => format!("{} (empty)", track.name),
            1 => format!("{} — {}", track.name, track.clips[0].name),
            n => {
                let first = &track.clips[0].name;
                format!("{} ({n} clips: {first}…)", track.name)
            }
        };
        labels.push(label);
        refs.push(ref_token);
    }

    let trimmed = current_ref.trim();
    let selected_idx = if trimmed.is_empty() {
        0
    } else if let Some(pos) = refs.iter().position(|r| r == trimmed) {
        pos
    } else {
        let short: String = trimmed.chars().take(20).collect();
        labels.push(format!("{short} (unresolved)"));
        refs.push(trimmed.to_owned());
        refs.len() - 1
    };

    (labels, refs, selected_idx)
}

/// Returns the `ScrubConfig` for a numeric custom shader uniform, honoring persisted Shader+
/// `min`, `max`, `step`, and `decimals` metadata bounds when present.
pub(super) fn custom_shader_number_scrub_config(
    label: &str,
    kind_hint: Option<usize>,
) -> ScrubConfig {
    let lower = label.to_ascii_lowercase();
    if kind_hint == Some(6) {
        ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 1)
            .with_scale_factor(std::f64::consts::PI / 180.0)
    } else if kind_hint == Some(7) || lower.contains("seed") {
        ScrubConfig::new(0.0, 100_000.0, 1.0, 0)
    } else if kind_hint == Some(15) {
        ScrubConfig::new(-120.0, 120.0, 1.0, 1)
    } else {
        ScrubConfig::new(-10_000.0, 10_000.0, 0.05, 3)
    }
}

fn custom_shader_scrub_config_with_meta(
    label: &str,
    kind_hint: Option<usize>,
    meta: &ShaderPlusPropMeta,
) -> ScrubConfig {
    let mut cfg = custom_shader_number_scrub_config(label, kind_hint);
    if let Some(min) = meta.min {
        cfg.min = min;
    }
    if let Some(max) = meta.max {
        cfg.max = max.max(cfg.min);
    }
    if let Some(step) = meta.step
        && step > 0.0
    {
        cfg.step = step;
    }
    if let Some(dec) = meta.decimals {
        cfg.decimals = usize::from(dec);
    }
    cfg
}

/// Formats a compact human-readable label for complex non-primitive JSON values (gradients,
/// curves, palettes, objects) so custom/dynamic property rows never render blank.
pub(super) fn summarize_complex_json_value(val: &Value) -> String {
    match val {
        Value::Array(arr) => {
            if arr.iter().all(|v| {
                v.as_object().is_some_and(|o| {
                    o.contains_key("color") && (o.contains_key("pos") || o.contains_key("position"))
                })
            }) && !arr.is_empty()
            {
                format!("Gradient ({} stops)", arr.len())
            } else if arr.iter().all(|v| v.as_array().is_some()) && !arr.is_empty() {
                format!("List ({} items)", arr.len())
            } else {
                format!("Array ({})", arr.len())
            }
        }
        Value::Object(map) => {
            if map.contains_key("master") || map.contains_key("rgb") {
                "Curve".to_owned()
            } else {
                format!("Object ({} keys)", map.len())
            }
        }
        Value::Null => "none".to_owned(),
        _ => val.to_string(),
    }
}

fn parse_json_color_rgba(v: &Value) -> Option<[f32; 4]> {
    if let Some(arr) = v.as_array()
        && arr.len() >= 3
        && arr.iter().take(4).all(Value::is_number)
    {
        let r = arr[0].as_f64().unwrap_or(1.0).clamp(0.0, 1.0) as f32;
        let g = arr[1].as_f64().unwrap_or(1.0).clamp(0.0, 1.0) as f32;
        let b = arr[2].as_f64().unwrap_or(1.0).clamp(0.0, 1.0) as f32;
        let a = arr
            .get(3)
            .and_then(Value::as_f64)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0) as f32;
        return Some([r, g, b, a]);
    }
    if let Some(s) = v.as_str() {
        let c = parse_css_rgba(s);
        return Some(c.0);
    }
    None
}

fn parse_shader_gradient_stops(val: &Value) -> Vec<(f64, [f32; 4])> {
    let stops_arr = val
        .get("stops")
        .and_then(Value::as_array)
        .or_else(|| val.as_array());
    let mut out = Vec::new();
    if let Some(arr) = stops_arr {
        for item in arr {
            if let Some(obj) = item.as_object() {
                let pos = obj
                    .get("pos")
                    .or_else(|| obj.get("position"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0);
                let mut rgba = obj
                    .get("color")
                    .and_then(parse_json_color_rgba)
                    .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                if let Some(op) = obj
                    .get("opacity")
                    .or_else(|| obj.get("alpha"))
                    .and_then(Value::as_f64)
                {
                    rgba[3] = op.clamp(0.0, 1.0) as f32;
                }
                out.push((pos, rgba));
            }
        }
    }
    if out.is_empty() {
        out.push((0.0, [0.0, 0.0, 0.0, 1.0]));
        out.push((1.0, [1.0, 1.0, 1.0, 1.0]));
    }
    out
}

fn serialize_shader_gradient_stops(stops: &[(f64, [f32; 4])]) -> Value {
    let arr: Vec<Value> = stops
        .iter()
        .map(|(pos, rgba)| {
            json!({
                "pos": pos.clamp(0.0, 1.0),
                "color": [f64::from(rgba[0]), f64::from(rgba[1]), f64::from(rgba[2])],
                "opacity": f64::from(rgba[3].clamp(0.0, 1.0))
            })
        })
        .collect();
    Value::Array(arr)
}

fn sample_shader_gradient_stops(stops: &[(f64, [f32; 4])], t: f64) -> Rgba {
    if stops.is_empty() {
        return Rgba::WHITE;
    }
    let mut sorted = stops.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    if t <= sorted[0].0 {
        let c = sorted[0].1;
        return Rgba::new(c[0], c[1], c[2], c[3]);
    }
    if let Some(last) = sorted.last()
        && t >= last.0
    {
        let c = last.1;
        return Rgba::new(c[0], c[1], c[2], c[3]);
    }
    for pair in sorted.windows(2) {
        let (p0, c0) = pair[0];
        let (p1, c1) = pair[1];
        if t >= p0 && t <= p1 {
            let span = (p1 - p0).max(1e-6);
            let u = ((t - p0) / span) as f32;
            let inv = 1.0 - u;
            return Rgba::new(
                c0[0] * inv + c1[0] * u,
                c0[1] * inv + c1[1] * u,
                c0[2] * inv + c1[2] * u,
                c0[3] * inv + c1[3] * u,
            );
        }
    }
    let c = sorted[0].1;
    Rgba::new(c[0], c[1], c[2], c[3])
}

#[allow(clippy::too_many_arguments)]
fn render_shader_gradient_editor(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    key: &str,
    label: &str,
    val: &Value,
    local_frame: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let live_val = eval_prop_value(
        lookup_session_prop_ref(session, target_id, key),
        local_frame,
    )
    .unwrap_or_else(|| val.clone());
    let mut stops = parse_shader_gradient_stops(&live_val);
    let mut dirty = false;
    let mut eyedropper_idx: Option<usize> = None;

    panel_property_row(
        ui,
        label,
        true,
        state,
        theme,
        |_| {},
        |ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 3.0;
                // Live smooth gradient preview bar
                let bar_w = ui.available_width().clamp(96.0, 220.0);
                let (bar_rect, _) = ui.allocate_exact_size(Vec2::new(bar_w, 20.0), Sense::hover());
                paint_checkerboard(ui.painter(), bar_rect, 5.0);
                const SEGS: usize = 24;
                let mut mesh = Mesh::default();
                for i in 0..SEGS {
                    let t0 = i as f64 / SEGS as f64;
                    let t1 = (i + 1) as f64 / SEGS as f64;
                    let c0 = rgba_to_color32(sample_shader_gradient_stops(&stops, t0));
                    let c1 = rgba_to_color32(sample_shader_gradient_stops(&stops, t1));
                    let x0 = bar_rect.left() + bar_rect.width() * (t0 as f32);
                    let x1 = bar_rect.left() + bar_rect.width() * (t1 as f32);
                    let tl = mesh.vertices.len() as u32;
                    mesh.colored_vertex(Pos2::new(x0, bar_rect.top()), c0);
                    let tr = mesh.vertices.len() as u32;
                    mesh.colored_vertex(Pos2::new(x1, bar_rect.top()), c1);
                    let br = mesh.vertices.len() as u32;
                    mesh.colored_vertex(Pos2::new(x1, bar_rect.bottom()), c1);
                    let bl = mesh.vertices.len() as u32;
                    mesh.colored_vertex(Pos2::new(x0, bar_rect.bottom()), c0);
                    mesh.add_triangle(tl, tr, br);
                    mesh.add_triangle(tl, br, bl);
                }
                ui.painter().add(Shape::mesh(mesh));
                ui.painter().rect_stroke(
                    bar_rect,
                    CornerRadius::ZERO,
                    Stroke::new(1.0, BORDER_CONTROL),
                    StrokeKind::Inside,
                );

                // Per-stop offset (0..=1), color picker + eyedropper, and remove button
                let mut remove_idx: Option<usize> = None;
                let stop_count = stops.len();
                for (idx, (pos, rgba_arr)) in stops.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 3.0;
                        ui.allocate_ui(Vec2::new(48.0, 20.0), |ui| {
                            let ev = cm3_scrub_number(
                                ui,
                                Id::unique((target_id, key, "grad_pos", idx)),
                                pos,
                                ScrubConfig::new(0.0, 1.0, 0.01, 2),
                                theme,
                            );
                            if ev.changed {
                                *pos = pos.clamp(0.0, 1.0);
                                dirty = true;
                            }
                        });
                        let mut rgba =
                            Rgba::new(rgba_arr[0], rgba_arr[1], rgba_arr[2], rgba_arr[3]);
                        let cev = cm3_hsva_color_picker(
                            ui,
                            Id::unique((target_id, key, "grad_col", idx)),
                            &mut rgba,
                            true,
                            false,
                            viewport_rect,
                            theme,
                        );
                        if cev.changed {
                            *rgba_arr = rgba.0;
                            dirty = true;
                        }
                        if cev.eyedropper_clicked {
                            eyedropper_idx = Some(idx);
                        }
                        if stop_count > 1
                            && cm3_icon_button(
                                ui,
                                Cm3Icon::Remove,
                                13.0,
                                TEXT_MUTED,
                                "Remove stop",
                                theme,
                            )
                        {
                            remove_idx = Some(idx);
                        }
                    });
                }
                if let Some(rm) = remove_idx
                    && stops.len() > 1
                {
                    stops.remove(rm);
                    dirty = true;
                }

                if stops.len() < 16
                    && cm3_prop_button(ui, "+ Add Stop", ButtonVariant::Normal, false, theme)
                        .clicked()
                {
                    let new_pos = if stops.len() >= 2 {
                        let a = stops[stops.len() - 2].0;
                        let b = stops[stops.len() - 1].0;
                        ((a + b) * 0.5).clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                    let sampled = sample_shader_gradient_stops(&stops, new_pos);
                    stops.push((new_pos, sampled.0));
                    stops.sort_by(|a, b| a.0.total_cmp(&b.0));
                    dirty = true;
                }
            });
        },
    );

    if let Some(idx) = eyedropper_idx {
        actions.push(PanelAction::EyedropperActivate {
            target_id: target_id.to_owned(),
            property: format!("{key}[{idx}]"),
        });
    }
    if dirty {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: target_id.to_owned(),
            property: key.to_owned(),
            value: serialize_shader_gradient_stops(&stops),
        });
    }
}

fn parse_shader_palette_colors(val: &Value) -> Vec<[f32; 4]> {
    let arr_opt = val
        .as_array()
        .or_else(|| val.get("colors").and_then(Value::as_array));
    let mut out = Vec::new();
    if let Some(arr) = arr_opt {
        for item in arr.iter().take(MAX_PALETTE_COLORS) {
            if let Some(rgba) = parse_json_color_rgba(item) {
                out.push(rgba);
            }
        }
    }
    if out.is_empty() {
        out.push([1.0, 0.3, 0.3, 1.0]);
        out.push([0.3, 0.9, 0.5, 1.0]);
        out.push([0.2, 0.5, 1.0, 1.0]);
    }
    out
}

fn serialize_shader_palette_colors(colors: &[[f32; 4]]) -> Value {
    let arr: Vec<Value> = colors
        .iter()
        .take(MAX_PALETTE_COLORS)
        .map(|c| {
            json!([
                f64::from(c[0]),
                f64::from(c[1]),
                f64::from(c[2]),
                f64::from(c[3])
            ])
        })
        .collect();
    Value::Array(arr)
}

#[allow(clippy::too_many_arguments)]
fn render_shader_palette_editor(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    key: &str,
    label: &str,
    val: &Value,
    local_frame: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let live_val = eval_prop_value(
        lookup_session_prop_ref(session, target_id, key),
        local_frame,
    )
    .unwrap_or_else(|| val.clone());
    let mut colors = parse_shader_palette_colors(&live_val);
    let mut dirty = false;
    let mut eyedropper_idx: Option<usize> = None;

    panel_property_row(
        ui,
        label,
        true,
        state,
        theme,
        |_| {},
        |ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 3.0;
                // Live palette swatch strip preview
                let bar_w = ui.available_width().clamp(96.0, 220.0);
                let (bar_rect, _) = ui.allocate_exact_size(Vec2::new(bar_w, 16.0), Sense::hover());
                paint_checkerboard(ui.painter(), bar_rect, 4.0);
                let n = colors.len().max(1) as f32;
                let seg_w = bar_rect.width() / n;
                for (i, c) in colors.iter().enumerate() {
                    let r = Rect::from_min_max(
                        Pos2::new(bar_rect.left() + i as f32 * seg_w, bar_rect.top()),
                        Pos2::new(bar_rect.left() + (i + 1) as f32 * seg_w, bar_rect.bottom()),
                    );
                    ui.painter().rect_filled(
                        r,
                        CornerRadius::ZERO,
                        rgba_to_color32(Rgba::new(c[0], c[1], c[2], c[3])),
                    );
                }
                ui.painter().rect_stroke(
                    bar_rect,
                    CornerRadius::ZERO,
                    Stroke::new(1.0, BORDER_CONTROL),
                    StrokeKind::Inside,
                );

                // Per-swatch color picker + eyedropper + remove
                let mut remove_idx: Option<usize> = None;
                let count = colors.len();
                for (idx, c) in colors.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        ui.label(
                            egui::RichText::new(format!("#{}", idx + 1))
                                .font(theme.font(FONT_COMPACT))
                                .color(TEXT_MUTED),
                        );
                        let mut rgba = Rgba::new(c[0], c[1], c[2], c[3]);
                        let cev = cm3_hsva_color_picker(
                            ui,
                            Id::unique((target_id, key, "pal_col", idx)),
                            &mut rgba,
                            true,
                            false,
                            viewport_rect,
                            theme,
                        );
                        if cev.changed {
                            *c = rgba.0;
                            dirty = true;
                        }
                        if cev.eyedropper_clicked {
                            eyedropper_idx = Some(idx);
                        }
                        if count > 1
                            && cm3_icon_button(
                                ui,
                                Cm3Icon::Remove,
                                13.0,
                                TEXT_MUTED,
                                "Remove color",
                                theme,
                            )
                        {
                            remove_idx = Some(idx);
                        }
                    });
                }
                if let Some(rm) = remove_idx
                    && colors.len() > 1
                {
                    colors.remove(rm);
                    dirty = true;
                }
                if colors.len() < MAX_PALETTE_COLORS
                    && cm3_prop_button(ui, "+ Add Stop", ButtonVariant::Normal, false, theme)
                        .clicked()
                {
                    let next = colors.last().copied().unwrap_or([1.0, 1.0, 1.0, 1.0]);
                    colors.push(next);
                    dirty = true;
                }
            });
        },
    );

    if let Some(idx) = eyedropper_idx {
        actions.push(PanelAction::EyedropperActivate {
            target_id: target_id.to_owned(),
            property: format!("{key}[{idx}]"),
        });
    }
    if dirty {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: target_id.to_owned(),
            property: key.to_owned(),
            value: serialize_shader_palette_colors(&colors),
        });
    }
}

fn parse_shader_curve_points(val: &Value) -> Vec<[f64; 2]> {
    let arr_opt = val
        .get("common")
        .or_else(|| val.get("master"))
        .or_else(|| val.get("rgb"))
        .or_else(|| val.get("points"))
        .and_then(Value::as_array)
        .or_else(|| val.as_array());
    let mut pts = Vec::new();
    if let Some(arr) = arr_opt {
        for item in arr.iter().take(16) {
            if let Some(pair) = item.as_array()
                && pair.len() >= 2
            {
                let x = pair[0].as_f64().unwrap_or(0.0).clamp(0.0, 1.0);
                let y = pair[1].as_f64().unwrap_or(0.0).clamp(0.0, 1.0);
                pts.push([x, y]);
            } else if let Some(obj) = item.as_object() {
                let x = obj
                    .get("x")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0);
                let y = obj
                    .get("y")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0);
                pts.push([x, y]);
            }
        }
    }
    if pts.len() < 2 {
        pts = vec![[0.0, 0.0], [1.0, 1.0]];
    }
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
    pts
}

#[allow(clippy::too_many_arguments)]
fn render_shader_curve_editor(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    key: &str,
    label: &str,
    val: &Value,
    local_frame: f64,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    let live_val = eval_prop_value(
        lookup_session_prop_ref(session, target_id, key),
        local_frame,
    )
    .unwrap_or_else(|| val.clone());
    let mut pts = parse_shader_curve_points(&live_val);
    let drag_id = Id::unique((target_id, key, "shader_curve_drag"));
    let mut dragging_pt: Option<usize> = ui.data_mut(|d| d.get_temp(drag_id)).unwrap_or(None);
    let mut dirty = false;

    panel_property_row(
        ui,
        label,
        true,
        state,
        theme,
        |_| {},
        |ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 3.0;
                // Interactive preview canvas
                let canvas_w = ui.available_width().clamp(110.0, 210.0);
                let (rect, resp) =
                    ui.allocate_exact_size(Vec2::new(canvas_w, 84.0), Sense::click_and_drag());
                ui.painter().rect_filled(rect, CornerRadius::ZERO, BG_INPUT);
                ui.painter().rect_stroke(
                    rect,
                    CornerRadius::ZERO,
                    Stroke::new(1.0, BORDER_CONTROL),
                    StrokeKind::Inside,
                );
                let grid_stroke = Stroke::new(1.0, Color32::from_rgb(0x2c, 0x2c, 0x2c));
                for i in 1..4 {
                    let frac = i as f32 / 4.0;
                    let gx = rect.left() + rect.width() * frac;
                    let gy = rect.top() + rect.height() * frac;
                    ui.painter().line_segment(
                        [Pos2::new(gx, rect.top()), Pos2::new(gx, rect.bottom())],
                        grid_stroke,
                    );
                    ui.painter().line_segment(
                        [Pos2::new(rect.left(), gy), Pos2::new(rect.right(), gy)],
                        grid_stroke,
                    );
                }
                ui.painter().line_segment(
                    [rect.left_bottom(), rect.right_top()],
                    Stroke::new(1.0, Color32::from_rgb(0x3a, 0x3a, 0x3a)),
                );

                let to_screen = |p: [f64; 2]| -> Pos2 {
                    Pos2::new(
                        rect.left() + (p[0].clamp(0.0, 1.0) as f32) * rect.width(),
                        rect.bottom() - (p[1].clamp(0.0, 1.0) as f32) * rect.height(),
                    )
                };
                let from_screen = |pos: Pos2| -> [f64; 2] {
                    [
                        f64::from(((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0)),
                        f64::from(((rect.bottom() - pos.y) / rect.height()).clamp(0.0, 1.0)),
                    ]
                };

                if resp.drag_started()
                    && let Some(pos) = ui.input(|i| i.pointer.interact_pos())
                    && let Some((best_i, _)) = pts
                        .iter()
                        .enumerate()
                        .map(|(i, p)| (i, to_screen(*p).distance(pos)))
                        .filter(|(_, d)| *d <= 10.0)
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                {
                    dragging_pt = Some(best_i);
                }
                if resp.dragged()
                    && let Some(pt_idx) = dragging_pt
                    && let Some(pos) = ui.input(|i| i.pointer.interact_pos())
                    && pt_idx < pts.len()
                {
                    let mut np = from_screen(pos);
                    if pt_idx == 0 {
                        np[0] = 0.0;
                    } else if pt_idx + 1 == pts.len() {
                        np[0] = 1.0;
                    } else {
                        let min_x = (pts[pt_idx - 1][0] + 0.01).min(0.99);
                        let max_x = (pts[pt_idx + 1][0] - 0.01).max(min_x);
                        np[0] = np[0].clamp(min_x, max_x);
                    }
                    if (pts[pt_idx][0] - np[0]).abs() > 1e-4
                        || (pts[pt_idx][1] - np[1]).abs() > 1e-4
                    {
                        pts[pt_idx] = np;
                        dirty = true;
                    }
                }
                if resp.drag_stopped() {
                    dragging_pt = None;
                }
                if resp.clicked()
                    && dragging_pt.is_none()
                    && let Some(pos) = resp.interact_pointer_pos()
                {
                    let near_existing = pts.iter().any(|p| to_screen(*p).distance(pos) <= 10.0);
                    if !near_existing && pts.len() < 16 {
                        pts.push(from_screen(pos));
                        pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
                        dirty = true;
                    }
                }
                if resp.secondary_clicked()
                    && let Some(pos) = resp.interact_pointer_pos()
                    && pts.len() > 2
                    && let Some((best_i, _)) = pts
                        .iter()
                        .enumerate()
                        .skip(1)
                        .take(pts.len() - 2)
                        .map(|(i, p)| (i, to_screen(*p).distance(pos)))
                        .filter(|(_, d)| *d <= 10.0)
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                {
                    pts.remove(best_i);
                    dirty = true;
                }

                let line_pts: Vec<Pos2> = pts.iter().map(|p| to_screen(*p)).collect();
                ui.painter().add(Shape::line(
                    line_pts.clone(),
                    Stroke::new(1.5, theme.accent()),
                ));
                for (i, sp) in line_pts.iter().enumerate() {
                    let h = Rect::from_center_size(*sp, Vec2::splat(7.0));
                    ui.painter().rect_filled(
                        h,
                        CornerRadius::ZERO,
                        if dragging_pt == Some(i) {
                            TEXT_BRIGHT
                        } else {
                            theme.accent()
                        },
                    );
                    ui.painter().rect_stroke(
                        h,
                        CornerRadius::ZERO,
                        Stroke::new(1.0, BORDER_CONTROL),
                        StrokeKind::Inside,
                    );
                }

                // Editable [x, y] control points list
                let mut remove_idx: Option<usize> = None;
                let pt_count = pts.len();
                for (idx, pt) in pts.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 3.0;
                        let mut cur = *pt;
                        let ev = cm3_vector2(
                            ui,
                            Id::unique((target_id, key, "curve_pt", idx)),
                            &mut cur,
                            ScrubConfig::new(0.0, 1.0, 0.01, 2),
                            false,
                            theme,
                        );
                        if ev.changed {
                            *pt = [cur[0].clamp(0.0, 1.0), cur[1].clamp(0.0, 1.0)];
                            dirty = true;
                        }
                        if pt_count > 2
                            && cm3_icon_button(
                                ui,
                                Cm3Icon::Remove,
                                13.0,
                                TEXT_MUTED,
                                "Remove point",
                                theme,
                            )
                        {
                            remove_idx = Some(idx);
                        }
                    });
                }
                if let Some(rm) = remove_idx
                    && pts.len() > 2
                {
                    pts.remove(rm);
                    dirty = true;
                }

                // + Add Point & Reset buttons
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    if pts.len() < 16
                        && cm3_prop_button(ui, "+ Add Point", ButtonVariant::Normal, false, theme)
                            .clicked()
                    {
                        let mid = if pts.len() >= 2 {
                            let a = pts[pts.len() - 2];
                            let b = pts[pts.len() - 1];
                            [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
                        } else {
                            [0.5, 0.5]
                        };
                        pts.push(mid);
                        pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
                        dirty = true;
                    }
                    if cm3_prop_button(ui, "Reset", ButtonVariant::Normal, false, theme).clicked() {
                        pts = vec![[0.0, 0.0], [1.0, 1.0]];
                        dirty = true;
                    }
                });
            });
        },
    );

    ui.data_mut(|d| d.insert_temp(drag_id, dragging_pt));

    if dirty {
        pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let mut next = if live_val.is_object() {
            live_val
        } else {
            json!({})
        };
        if let Some(obj) = next.as_object_mut() {
            obj.insert("master".to_owned(), json!(pts));
            obj.insert("rgb".to_owned(), json!(pts));
            obj.insert("common".to_owned(), json!(pts));
        }
        let _ = session.execute(Command::SetPropertyValue {
            target_id: target_id.to_owned(),
            property: key.to_owned(),
            value: next,
        });
    }
}

fn parse_shader_path_value(val: &Value) -> (Vec<[f64; 2]>, bool) {
    let closed = val.get("closed").and_then(Value::as_bool).unwrap_or(false);
    let mut verts = Vec::new();
    let arr_opt = val
        .get("vertices")
        .or_else(|| val.get("points"))
        .and_then(Value::as_array)
        .or_else(|| val.as_array());
    if let Some(arr) = arr_opt {
        for item in arr {
            let pt_val = item.get("point").unwrap_or(item);
            if let Some(pair) = pt_val.as_array()
                && pair.len() >= 2
                && let (Some(x), Some(y)) = (pair[0].as_f64(), pair[1].as_f64())
            {
                verts.push([x, y]);
            }
        }
    }
    if verts.len() < 2 {
        verts = vec![[0.15, 0.5], [0.85, 0.5]];
    }
    (verts, closed)
}

fn serialize_shader_path_value(verts: &[[f64; 2]], closed: bool) -> Value {
    let vertices: Vec<Value> = verts
        .iter()
        .map(|p| {
            json!({
                "point": [p[0], p[1]],
                "in": [0.0, 0.0],
                "out": [0.0, 0.0]
            })
        })
        .collect();
    json!({
        "vertices": vertices,
        "closed": closed
    })
}

#[allow(clippy::too_many_arguments)]
fn render_shader_path_editor(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    key: &str,
    label: &str,
    val: &Value,
    local_frame: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    let live_val = eval_prop_value(
        lookup_session_prop_ref(session, target_id, key),
        local_frame,
    )
    .unwrap_or_else(|| val.clone());
    let (mut verts, mut closed) = parse_shader_path_value(&live_val);
    let mut dirty = false;

    panel_property_row(
        ui,
        label,
        true,
        state,
        theme,
        |_| {},
        |ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 3.0;
                // Mini 2D path preview canvas
                let preview_w = ui.available_width().clamp(110.0, 210.0);
                let (rect, _) = ui.allocate_exact_size(Vec2::new(preview_w, 56.0), Sense::hover());
                ui.painter().rect_filled(rect, CornerRadius::ZERO, BG_INPUT);
                ui.painter().rect_stroke(
                    rect,
                    CornerRadius::ZERO,
                    Stroke::new(1.0, BORDER_CONTROL),
                    StrokeKind::Inside,
                );
                let mut screen_pts: Vec<Pos2> = verts
                    .iter()
                    .map(|p| {
                        Pos2::new(
                            rect.left() + (p[0].clamp(0.0, 1.0) as f32) * rect.width(),
                            rect.bottom() - (p[1].clamp(0.0, 1.0) as f32) * rect.height(),
                        )
                    })
                    .collect();
                if closed && let Some(&first) = screen_pts.first() {
                    screen_pts.push(first);
                }
                if screen_pts.len() >= 2 {
                    ui.painter().add(Shape::line(
                        screen_pts.clone(),
                        Stroke::new(1.5, theme.accent()),
                    ));
                }
                for sp in screen_pts.iter().take(verts.len()) {
                    ui.painter().rect_filled(
                        Rect::from_center_size(*sp, Vec2::splat(5.0)),
                        CornerRadius::ZERO,
                        TEXT_BRIGHT,
                    );
                }

                // Closed toggle + Add Vertex
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    let mut closed_idx = usize::from(closed);
                    if cm3_dropdown(
                        ui,
                        Id::unique((target_id, key, "path_closed")),
                        &mut closed_idx,
                        &["Open", "Closed"],
                        false,
                        viewport_rect,
                        theme,
                    ) {
                        closed = closed_idx != 0;
                        dirty = true;
                    }
                    if verts.len() < 32
                        && cm3_prop_button(ui, "+ Add Vertex", ButtonVariant::Normal, false, theme)
                            .clicked()
                    {
                        let next = verts
                            .last()
                            .map_or([0.5, 0.5], |p| [(p[0] + 0.1).min(1.0), p[1]]);
                        verts.push(next);
                        dirty = true;
                    }
                });

                // Per-vertex [x, y] editors
                let mut remove_idx: Option<usize> = None;
                let vert_count = verts.len();
                for (idx, v) in verts.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 3.0;
                        ui.label(
                            egui::RichText::new(format!("#{idx}"))
                                .font(theme.font(FONT_COMPACT))
                                .color(TEXT_MUTED),
                        );
                        let mut cur = *v;
                        let ev = cm3_vector2(
                            ui,
                            Id::unique((target_id, key, "path_vert", idx)),
                            &mut cur,
                            ScrubConfig::new(-10.0, 10.0, 0.01, 3),
                            false,
                            theme,
                        );
                        if ev.changed {
                            *v = cur;
                            dirty = true;
                        }
                        if vert_count > 2
                            && cm3_icon_button(
                                ui,
                                Cm3Icon::Remove,
                                13.0,
                                TEXT_MUTED,
                                "Remove vertex",
                                theme,
                            )
                        {
                            remove_idx = Some(idx);
                        }
                    });
                }
                if let Some(rm) = remove_idx
                    && verts.len() > 2
                {
                    verts.remove(rm);
                    dirty = true;
                }
            });
        },
    );

    if dirty {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: target_id.to_owned(),
            property: key.to_owned(),
            value: serialize_shader_path_value(&verts, closed),
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn render_shader_dropdown_editor(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    key: &str,
    label: &str,
    val: &Value,
    is_static: bool,
    meta: &ShaderPlusPropMeta,
    local_frame: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    let mut items: Vec<String> = if !meta.items.is_empty() {
        meta.items.clone()
    } else if is_static {
        vec![
            "Mode 1".to_owned(),
            "Mode 2".to_owned(),
            "Mode 3".to_owned(),
        ]
    } else {
        vec![
            "Option 1".to_owned(),
            "Option 2".to_owned(),
            "Option 3".to_owned(),
        ]
    };

    let default_idx = val
        .as_u64()
        .or_else(|| val.as_f64().map(|f| f.max(0.0).round() as u64))
        .unwrap_or(0) as usize;
    let cur_idx = eval_prop_value(
        lookup_session_prop_ref(session, target_id, key),
        local_frame,
    )
    .and_then(|v| {
        v.as_u64()
            .or_else(|| v.as_f64().map(|f| f.max(0.0).round() as u64))
            .or_else(|| v.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
    })
    .map_or(default_idx, |n| n as usize)
    .min(items.len().saturating_sub(1));

    let edit_open_id = Id::unique((target_id, key, "shader_dd_edit_open"));
    let mut edit_open: bool = ui.data_mut(|d| d.get_temp(edit_open_id)).unwrap_or(false);
    let mut sel_idx = cur_idx;
    let mut value_changed = false;
    let mut items_changed = false;

    panel_property_row(
        ui,
        label,
        true,
        state,
        theme,
        |_| {},
        |ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 3.0;
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    let item_refs: Vec<&str> = items.iter().map(String::as_str).collect();
                    if cm3_dropdown(
                        ui,
                        Id::unique((target_id, key, "shader_dd_select")),
                        &mut sel_idx,
                        &item_refs,
                        false,
                        viewport_rect,
                        theme,
                    ) {
                        value_changed = true;
                    }
                    let toggle_lbl = if edit_open { "Done" } else { "Edit options" };
                    if cm3_prop_button(ui, toggle_lbl, ButtonVariant::Normal, false, theme)
                        .clicked()
                    {
                        edit_open = !edit_open;
                    }
                });

                if edit_open {
                    let mut remove_idx: Option<usize> = None;
                    let count = items.len();
                    for (idx, opt) in items.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 3.0;
                            let edit_w = (ui.available_width() - 26.0).max(60.0);
                            let resp = Frame::NONE
                                .fill(BG_INPUT)
                                .stroke(Stroke::new(1.0, BORDER_CONTROL))
                                .inner_margin(egui::Margin::symmetric(4, 1))
                                .show(ui, |ui| {
                                    ui.add(
                                        TextEdit::singleline(opt)
                                            .id(Id::unique((target_id, key, "dd_opt", idx)))
                                            .frame(Frame::NONE)
                                            .font(theme.font(FONT_SMALL))
                                            .text_color(theme.value())
                                            .desired_width(edit_w),
                                    )
                                })
                                .inner;
                            if resp.changed() {
                                items_changed = true;
                            }
                            if count > 1
                                && cm3_icon_button(
                                    ui,
                                    Cm3Icon::Remove,
                                    13.0,
                                    TEXT_MUTED,
                                    "Remove option",
                                    theme,
                                )
                            {
                                remove_idx = Some(idx);
                            }
                        });
                    }
                    if let Some(rm) = remove_idx
                        && items.len() > 1
                    {
                        items.remove(rm);
                        if sel_idx >= items.len() {
                            sel_idx = items.len() - 1;
                            value_changed = true;
                        }
                        items_changed = true;
                    }
                    if cm3_prop_button(ui, "+ Add Option", ButtonVariant::Normal, false, theme)
                        .clicked()
                    {
                        items.push(format!("Option {}", items.len() + 1));
                        items_changed = true;
                    }
                }
            });
        },
    );

    ui.data_mut(|d| d.insert_temp(edit_open_id, edit_open));

    if value_changed {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: target_id.to_owned(),
            property: key.to_owned(),
            value: json!(sel_idx as u32),
        });
    }
    if items_changed {
        let mut updated_meta: serde_json::Map<String, Value> = session
            .document()
            .property_metadata(target_id, key)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        let type_entry = updated_meta
            .entry("type".to_owned())
            .or_insert_with(|| json!({}));
        if !type_entry.is_object() {
            *type_entry = json!({});
        }
        if let Some(type_obj) = type_entry.as_object_mut() {
            type_obj.insert("dynamic".to_owned(), json!(!is_static));
            let zs_entry = type_obj
                .entry("zoidiumShader".to_owned())
                .or_insert_with(|| json!({}));
            if !zs_entry.is_object() {
                *zs_entry = json!({});
            }
            if let Some(zs_obj) = zs_entry.as_object_mut() {
                zs_obj.insert(
                    "kind".to_owned(),
                    json!(if is_static {
                        "staticDropdown"
                    } else {
                        "dropdown"
                    }),
                );
                zs_obj.insert("items".to_owned(), json!(items));
            }
        }
        let _ = session.execute(Command::SetPropertyValue {
            target_id: target_id.to_owned(),
            property: format!("__meta.{key}"),
            value: Value::Object(updated_meta),
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn render_shader_layer_source_editor(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    key: &str,
    label: &str,
    val: &Value,
    local_frame: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
) {
    let default_ref = val.as_str().unwrap_or("");
    let cur_ref = eval_prop_str(
        lookup_session_prop_ref(session, target_id, key),
        local_frame,
        default_ref,
    );
    let (layer_labels, layer_refs, mut sel_idx) =
        collect_shader_layer_source_choices(session, &cur_ref);
    let label_refs: Vec<&str> = layer_labels.iter().map(String::as_str).collect();
    let changed = panel_property_row(
        ui,
        label,
        true,
        state,
        theme,
        |_| {},
        |ui| {
            cm3_dropdown(
                ui,
                Id::unique((target_id, key, "shader_layer_source")),
                &mut sel_idx,
                &label_refs,
                false,
                viewport_rect,
                theme,
            )
        },
    );
    if changed && let Some(chosen_ref) = layer_refs.get(sel_idx) {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: target_id.to_owned(),
            property: key.to_owned(),
            value: json!(chosen_ref),
        });
    }
}

fn is_gradient_json_value(val: &Value) -> bool {
    let arr = val
        .get("stops")
        .and_then(Value::as_array)
        .or_else(|| val.as_array());
    arr.is_some_and(|a| {
        !a.is_empty()
            && a.iter().all(|v| {
                v.as_object().is_some_and(|o| {
                    o.contains_key("color") && (o.contains_key("pos") || o.contains_key("position"))
                })
            })
    })
}

fn is_palette_json_value(val: &Value) -> bool {
    let arr = val
        .get("colors")
        .and_then(Value::as_array)
        .or_else(|| val.as_array());
    arr.is_some_and(|a| {
        !a.is_empty()
            && a.iter().all(|v| {
                v.as_array().is_some_and(|c| {
                    (c.len() == 3 || c.len() == 4) && c.iter().all(Value::is_number)
                })
            })
    })
}

fn is_curve_json_value(val: &Value) -> bool {
    val.as_object().is_some_and(|m| {
        m.contains_key("master") || m.contains_key("rgb") || m.contains_key("common")
    })
}

fn is_path_json_value(val: &Value) -> bool {
    if let Some(obj) = val.as_object() {
        return obj.contains_key("vertices") && obj.get("vertices").is_some_and(Value::is_array);
    }
    val.as_array().is_some_and(|a| {
        a.len() >= 2
            && a.iter().all(|v| {
                v.as_array()
                    .is_some_and(|p| p.len() == 2 && p.iter().all(Value::is_number))
            })
    })
}

#[allow(clippy::too_many_arguments)]
fn render_custom_shader_uniform(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    key: &str,
    label: &str,
    val: &Value,
    kind_hint: Option<usize>,
    meta: &ShaderPlusPropMeta,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let lower = label.to_ascii_lowercase();
    let mut delete_uniform = false;
    let imported_prop_type = lookup_custom_shader_prop_type(session, target_id, key);
    let is_layer_source = kind_hint == Some(16) || meta.is_layer_source;
    let is_sampler2d = !is_layer_source && imported_prop_type == Some(8);
    let is_dropdown = matches!(kind_hint, Some(8 | 9));
    let is_gradient = kind_hint == Some(10) || is_gradient_json_value(val);
    let is_curve = kind_hint == Some(11) || is_curve_json_value(val);
    let is_palette = kind_hint == Some(12) || (kind_hint.is_none() && is_palette_json_value(val));
    let is_path = kind_hint == Some(14) || (kind_hint.is_none() && is_path_json_value(val));
    let is_bool_toggle = matches!(val, Value::Bool(_))
        || kind_hint == Some(5)
        || (imported_prop_type == Some(5) && is_type5_bool_placeholder_value(val));

    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        ui.vertical(|ui| {
            ui.set_max_width((ui.available_width() - 24.0).max(80.0));
            if is_layer_source {
                render_shader_layer_source_editor(
                    ui,
                    session,
                    target_id,
                    key,
                    label,
                    val,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                );
            } else if is_sampler2d {
                let default_asset = val.as_str().unwrap_or("");
                let cur_asset = eval_prop_str(
                    lookup_session_prop_ref(session, target_id, key),
                    local_frame,
                    default_asset,
                );
                let (asset_labels, asset_ids, mut sel_idx) =
                    collect_shader_image_asset_choices(session, &cur_asset);
                let label_refs: Vec<&str> = asset_labels.iter().map(String::as_str).collect();
                let changed = panel_property_row(
                    ui,
                    label,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_dropdown(
                            ui,
                            Id::unique((target_id, key, "shader_sampler2d_asset")),
                            &mut sel_idx,
                            &label_refs,
                            false,
                            viewport_rect,
                            theme,
                        )
                    },
                );
                if changed && let Some(chosen_id) = asset_ids.get(sel_idx) {
                    let _ = session.execute(Command::SetPropertyValue {
                        target_id: target_id.to_owned(),
                        property: key.to_owned(),
                        value: json!(chosen_id),
                    });
                }
            } else if is_dropdown {
                render_shader_dropdown_editor(
                    ui,
                    session,
                    target_id,
                    key,
                    label,
                    val,
                    kind_hint == Some(9),
                    meta,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                );
            } else if is_gradient {
                render_shader_gradient_editor(
                    ui,
                    session,
                    target_id,
                    key,
                    label,
                    val,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            } else if is_palette {
                render_shader_palette_editor(
                    ui,
                    session,
                    target_id,
                    key,
                    label,
                    val,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            } else if is_curve {
                render_shader_curve_editor(
                    ui,
                    session,
                    target_id,
                    key,
                    label,
                    val,
                    local_frame,
                    state,
                    theme,
                );
            } else if is_path {
                render_shader_path_editor(
                    ui,
                    session,
                    target_id,
                    key,
                    label,
                    val,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                );
            } else if is_bool_toggle {
                let default_bool = match val {
                    Value::Bool(b) => *b,
                    Value::Number(n) => n.as_f64().unwrap_or(0.0) >= 0.5,
                    _ => false,
                };
                let cur_bool = eval_prop_value(
                    lookup_session_prop_ref(session, target_id, key),
                    local_frame,
                )
                .map_or(default_bool, |v| {
                    v.as_bool()
                        .or_else(|| v.as_f64().map(|f| f >= 0.5))
                        .unwrap_or(default_bool)
                });
                let mut idx = usize::from(cur_bool);
                let changed = panel_property_row(
                    ui,
                    label,
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_dropdown(
                            ui,
                            Id::unique((target_id, key, "shader_bool_toggle")),
                            &mut idx,
                            &["off", "on"],
                            false,
                            viewport_rect,
                            theme,
                        )
                    },
                );
                if changed {
                    state
                        .shader_prop_kinds
                        .insert((target_id.to_owned(), key.to_owned()), 5);
                    let _ = session.execute(Command::SetPropertyValue {
                        target_id: target_id.to_owned(),
                        property: key.to_owned(),
                        value: json!(idx != 0),
                    });
                }
            } else {
                match val {
                    Value::Number(n) => {
                        let scrub_cfg =
                            custom_shader_scrub_config_with_meta(label, kind_hint, meta);
                        let default_val = if kind_hint == Some(7) || lower.contains("seed") {
                            n.as_f64().unwrap_or(1.0)
                        } else {
                            n.as_f64().unwrap_or(0.0)
                        };
                        show_session_number_prop(
                            ui,
                            session,
                            target_id,
                            key,
                            label,
                            default_val,
                            scrub_cfg,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                    Value::Array(arr) if arr.len() == 4 && arr.iter().all(Value::is_number) => {
                        let def = [
                            arr[0].as_f64().unwrap_or(1.0),
                            arr[1].as_f64().unwrap_or(1.0),
                            arr[2].as_f64().unwrap_or(1.0),
                            arr[3].as_f64().unwrap_or(1.0),
                        ];
                        show_session_color_rgba_prop(
                            ui,
                            session,
                            target_id,
                            key,
                            label,
                            def,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                    Value::Array(arr)
                        if arr.len() == 3
                            && arr.iter().all(Value::is_number)
                            && (kind_hint == Some(3) || lower.contains("color")) =>
                    {
                        let def = [
                            arr[0].as_f64().unwrap_or(1.0),
                            arr[1].as_f64().unwrap_or(1.0),
                            arr[2].as_f64().unwrap_or(1.0),
                        ];
                        show_session_color_prop(
                            ui,
                            session,
                            target_id,
                            key,
                            label,
                            def,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                    _ => {
                        render_dynamic_json_param(
                            ui,
                            session,
                            target_id,
                            key,
                            label,
                            val,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                }
            }
        });

        if cm3_icon_button(
            ui,
            Cm3Icon::Remove,
            15.0,
            TEXT_MUTED,
            "remove custom uniform",
            theme,
        ) {
            delete_uniform = true;
        }
    });

    if delete_uniform {
        state
            .shader_prop_kinds
            .remove(&(target_id.to_owned(), key.to_owned()));
        let started_tx = !session.history.is_in_transaction()
            && session.begin_transaction("remove custom uniform").is_ok();
        let _ = session.execute(Command::SetPropertyValue {
            target_id: target_id.to_owned(),
            property: format!("__meta.{key}"),
            value: Value::Null,
        });
        let _ = session.execute(Command::RemoveProperty {
            target_id: target_id.to_owned(),
            property: key.to_owned(),
        });
        if started_tx && session.history.is_in_transaction() {
            let _ = session.commit_transaction(true);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_dynamic_json_param(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    key: &str,
    label: &str,
    val: &Value,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    if is_gradient_json_value(val) {
        render_shader_gradient_editor(
            ui,
            session,
            target_id,
            key,
            label,
            val,
            local_frame,
            viewport_rect,
            state,
            theme,
            actions,
        );
        return;
    }
    if is_curve_json_value(val) {
        render_shader_curve_editor(
            ui,
            session,
            target_id,
            key,
            label,
            val,
            local_frame,
            state,
            theme,
        );
        return;
    }
    if is_path_json_value(val) {
        render_shader_path_editor(
            ui,
            session,
            target_id,
            key,
            label,
            val,
            local_frame,
            viewport_rect,
            state,
            theme,
        );
        return;
    }
    match val {
        Value::Bool(b) => {
            show_session_option_prop(
                ui,
                session,
                target_id,
                key,
                label,
                &["off", "on"],
                u32::from(*b),
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        Value::Number(n) => {
            show_session_number_prop(
                ui,
                session,
                target_id,
                key,
                label,
                n.as_f64().unwrap_or(0.0),
                ScrubConfig::new(-10_000.0, 10_000.0, 0.05, 3),
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        Value::Array(arr) if arr.len() == 2 && arr.iter().all(Value::is_number) => {
            let def = [
                arr[0].as_f64().unwrap_or(0.0),
                arr[1].as_f64().unwrap_or(0.0),
            ];
            show_session_vec2_prop(
                ui,
                session,
                target_id,
                key,
                label,
                def,
                ScrubConfig::new(-10_000.0, 10_000.0, 0.05, 3),
                false,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        }
        Value::Array(arr) if arr.len() == 3 && arr.iter().all(Value::is_number) => {
            let def = [
                arr[0].as_f64().unwrap_or(0.0),
                arr[1].as_f64().unwrap_or(0.0),
                arr[2].as_f64().unwrap_or(0.0),
            ];
            if label.to_ascii_lowercase().contains("color") {
                show_session_color_prop(
                    ui,
                    session,
                    target_id,
                    key,
                    label,
                    def,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            } else {
                show_session_vec3_prop(
                    ui,
                    session,
                    target_id,
                    key,
                    label,
                    def,
                    ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 3),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            }
        }
        Value::Array(arr) if arr.len() == 4 && arr.iter().all(Value::is_number) => {
            let def = [
                arr[0].as_f64().unwrap_or(0.0),
                arr[1].as_f64().unwrap_or(0.0),
                arr[2].as_f64().unwrap_or(0.0),
                arr[3].as_f64().unwrap_or(1.0),
            ];
            if label.to_ascii_lowercase().contains("color") {
                show_session_color_rgba_prop(
                    ui,
                    session,
                    target_id,
                    key,
                    label,
                    def,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            } else {
                show_session_vec4_prop(
                    ui,
                    session,
                    target_id,
                    key,
                    label,
                    def,
                    ScrubConfig::new(-10_000.0, 10_000.0, 0.1, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            }
        }
        Value::String(s) => {
            show_session_text_prop(
                ui,
                session,
                target_id,
                key,
                label,
                s,
                local_frame,
                state,
                theme,
            );
        }
        other => {
            let summary = summarize_complex_json_value(other);
            panel_property_row(
                ui,
                label,
                true,
                state,
                theme,
                |_| {},
                |ui| {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(summary)
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_MUTED),
                        )
                        .truncate(),
                    );
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_plus_metadata_persists_and_hydrates_all_17_kinds() {
        let mut session = Session::new();
        let clip_id = session.document().video_tracks[0].clips[0].id.clone();
        let added = session
            .execute(Command::AddEffect {
                clip_id,
                kind: "customshader".to_owned(),
                name: Some("Shader+ Test".to_owned()),
            })
            .expect("add customshader");
        let fx_id = added["effect_id"].as_str().expect("effect id").to_owned();

        for kind_idx in 0..17 {
            let prop_name = format!("u_prop_{kind_idx}");
            session
                .execute(Command::SetPropertyValue {
                    target_id: fx_id.clone(),
                    property: prop_name.clone(),
                    value: shader_plus_default_value(kind_idx),
                })
                .expect("set value");
            session
                .execute(Command::SetPropertyValue {
                    target_id: fx_id.clone(),
                    property: format!("__meta.{prop_name}"),
                    value: shader_plus_default_metadata(kind_idx),
                })
                .expect("set meta");
        }

        // Round-trip through Project -> Session reload and verify hydration.
        let reloaded =
            Session::from_project(session.project.clone(), zc_plugin::PluginRegistry::new());
        for kind_idx in 0..17 {
            let prop_name = format!("u_prop_{kind_idx}");
            let raw_meta = reloaded
                .document()
                .property_metadata(&fx_id, &prop_name)
                .expect("metadata must survive reload");
            let parsed = parse_shader_plus_metadata(raw_meta);
            assert_eq!(
                parsed.kind_idx,
                Some(kind_idx),
                "kind_idx {kind_idx} must hydrate accurately"
            );
            if kind_idx == 8 {
                assert_eq!(parsed.items, vec!["Option 1", "Option 2", "Option 3"]);
            } else if kind_idx == 9 {
                assert_eq!(parsed.items, vec!["Mode 1", "Mode 2", "Mode 3"]);
            } else if kind_idx == 16 {
                assert!(parsed.is_layer_source);
            }
        }
    }

    #[test]
    fn shader_plus_visibility_and_rename_rules_evaluate_against_effect_values() {
        let mut meta = BTreeMap::new();
        meta.insert(
            "type".to_owned(),
            json!({
                "zoidiumShader": {
                    "kind": "dropdown",
                    "items": ["Normal", "Add", "Screen"],
                    "visibility": { "when": { "prop": "Mix", "op": ">", "value": 0.0 } },
                    "rename": [
                        { "when": { "prop": "Mix", "op": ">", "value": 0.5 }, "label": "Strong Blend" },
                        { "when": { "prop": "Mode", "op": "=", "value": 2.0 }, "label": "Screen Mode" }
                    ]
                }
            }),
        );
        let parsed = parse_shader_plus_metadata(&meta);
        assert_eq!(parsed.kind_idx, Some(8));
        assert_eq!(parsed.items, vec!["Normal", "Add", "Screen"]);
        let vis = parsed.visible_when.as_ref().expect("visibility condition");

        let mut nums = BTreeMap::new();
        let strs = BTreeMap::new();
        nums.insert("Mix".to_owned(), 0.0);
        nums.insert("Mode".to_owned(), 1.0);
        assert!(
            !evaluate_shader_visibility(vis, &nums, &strs),
            "Mix == 0.0 should hide property"
        );
        assert_eq!(
            resolve_shader_label("Blend", &parsed.rename_when, &nums, &strs),
            "Blend"
        );

        nums.insert("Mix".to_owned(), 0.75);
        assert!(
            evaluate_shader_visibility(vis, &nums, &strs),
            "Mix == 0.75 should show property"
        );
        assert_eq!(
            resolve_shader_label("Blend", &parsed.rename_when, &nums, &strs),
            "Strong Blend"
        );

        // Also verify Zoidium saved JSON schema `{ property, values, conditions }`.
        let mut z_meta = BTreeMap::new();
        z_meta.insert(
            "type".to_owned(),
            json!({
                "zoidiumShader": {
                    "kind": "slider",
                    "visibleWhen": { "property": "Mode", "values": [1, 2], "conditions": [] },
                    "renameWhen": [{ "property": "Mode", "name": "Additive Gain", "values": [1] }]
                }
            }),
        );
        let z_parsed = parse_shader_plus_metadata(&z_meta);
        assert!(evaluate_shader_visibility(
            z_parsed.visible_when.as_ref().expect("z_vis"),
            &nums,
            &strs
        ));
        assert_eq!(
            resolve_shader_label("Gain", &z_parsed.rename_when, &nums, &strs),
            "Additive Gain"
        );
    }

    #[test]
    fn layer_source_choices_list_none_and_video_tracks_with_track_id_tokens() {
        let session = Session::new();
        let first_track_id = session.document().video_tracks[0].id.clone();
        let token = format!("track:id:{first_track_id}");

        let (labels, refs, sel_none) = collect_shader_layer_source_choices(&session, "");
        assert_eq!(labels[0], "None");
        assert_eq!(refs[0], "");
        assert_eq!(sel_none, 0);
        assert!(refs.contains(&token));

        let (_, _, sel_track) = collect_shader_layer_source_choices(&session, &token);
        assert_eq!(refs[sel_track], token);
    }
}
