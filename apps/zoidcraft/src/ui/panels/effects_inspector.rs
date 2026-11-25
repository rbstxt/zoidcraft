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
                let child_count = collect_effect_group_child_rows(
                    session,
                    &clip.id,
                    &fx.id,
                    &BTreeSet::new(),
                )
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
                                                let default_val = match kind_idx {
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
                                                        "rgb": [[0.0, 0.0], [1.0, 1.0]]
                                                    }),
                                                    12 => json!([[1.0, 0.3, 0.3], [0.3, 0.9, 0.5], [0.2, 0.5, 1.0]]),
                                                    13 => json!("Shader Text"),
                                                    14 => json!([[0.0, 0.0], [1.0, 1.0]]),
                                                    15 => json!(0.0),
                                                    16 => json!(""),
                                                    _ => json!(0.5),
                                                };
                                                state.shader_prop_kinds.insert(
                                                    (fx.id.clone(), prop_key.clone()),
                                                    kind_idx,
                                                );
                                                let _ = session.execute(Command::SetPropertyValue {
                                                    target_id: fx.id.clone(),
                                                    property: prop_key,
                                                    value: default_val,
                                                });
                                            }
                                        }
                                    }
                                });
                            });

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
                            let label = key.strip_prefix("u_").unwrap_or(&key);
                            let kind_hint = state
                                .shader_prop_kinds
                                .get(&(fx.id.clone(), key.clone()))
                                .copied();
                            render_custom_shader_uniform(
                                ui,
                                session,
                                &fx.id,
                                &key,
                                label,
                                &val,
                                kind_hint,
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
        ("sglow", "quality") => &["low", "medium", "high"],
        ("sglow", "show") => &["result", "glow only", "threshold"],
        ("sglow", "edgeMode") => &["transparent", "clamp", "reflect"],
        ("sglow", "opacity") => &["normal", "affect alpha"],
        ("sglow", "blurEngine" | "engine") => &["gaussian", "box", "dual kawase"],
        ("sglow", "atmosphereModel") => &["perlin", "simplex", "fbm"],
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
        ("echo-legacy", "mode") => &["blend", "add", "screen", "maximum"],
        ("ascii", "colorMode") => &["band", "footage", "duotone", "mono", "invert"],
        ("ascii", "fontFamily") => &["monospace", "sans", "serif", "condensed", "pixel"],
        ("ascii", "charset") => &["standard", "dense", "blocks", "binary", "symbols"],
        ("ascii", "noiseType") => &["value", "perlin", "simplex"],
        ("datamosh", "algorithm") => &[
            "classic block",
            "horizontal tear",
            "vertical tear",
            "zoom smear",
            "spin drift",
            "wave warp",
            "block swap",
            "scatter",
            "bloom push",
            "melt",
            "echo drift",
            "glitch slice",
            "auto motion",
        ],
        ("datamosh", "motionFamily" | "motion") => &[
            "block",
            "horizontal",
            "vertical",
            "zoom",
            "rotate",
            "wave",
            "swap",
            "random",
        ],
        ("datamosh", "lookMode") => &["classic", "smear", "echo"],
        ("smartmotionblur", "view") => &["result", "motion vectors", "confidence"],
        ("smartmotionblur", "reconstruction") => &["optical flow", "shutter samples"],
        ("smartmotionblur", "analysisFrames" | "analysis") => &["prev & next", "prev", "next"],
        ("smartmotionblur" | "smart-motion-blur-lite", "quality") => &["low", "medium", "high"],
        ("smart-motion-blur-lite", "trackFrame") => &["both", "prev", "next"],
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
        ("timecode", "timeSource") => &["Sequence", "Clip", "Custom"],
        ("ccpixelpolly", "object") => &[
            "Triangles (flat)",
            "Triangles (textured)",
            "Squares (flat)",
            "Squares (textured)",
        ],
        ("ccpageturn", "render") => &["front & back", "front only", "back only"],
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
            "final",
            "matte",
            "source",
            "partial A",
            "partial B",
            "matte corrected",
        ],
        ("colordiffkey", "colorMatchingAccuracy") => &["faster", "more accurate"],
        ("advspill", "method") => &["standard", "ultra"],
        ("linearcolorkey", "view") => &["final", "matte", "source"],
        ("linearcolorkey", "matchColors") => &["RGB", "Hue", "Chroma"],
        ("linearcolorkey", "keyOperation") => &["key out", "key in"],
        ("radiowaves", "waveType") => &["polygon", "image contour", "mask"],
        ("radiowaves", "profile") => &[
            "flat",
            "sawtooth down",
            "sawtooth up",
            "triangle",
            "sine",
            "half-in",
            "half-out",
        ],
        ("numbers", "format") => &[
            "number",
            "number (leading zeroes)",
            "timecode 30",
            "timecode 24",
            "timecode 25",
            "time (HH:MM:SS)",
            "short date",
            "hexadecimal",
        ],
        ("numbers", "displayOptions") => {
            &["fill", "stroke", "fill over stroke", "stroke over fill"]
        }
        ("tracery", "detectionQuality") => &["low", "medium", "high"],
        ("tracery", "boxShape") => &["rectangle", "corners", "circle", "diamond"],
        ("tracery", "boxFill") => &["none", "solid", "tint", "invert"],
        ("tracery", "markerShape") => &["cross", "circle", "square", "diamond"],
        ("tracery", "gridMode") => &["square", "crosshair", "dots"],
        ("tracery", "lineType") => &["straight", "curved", "step", "bezier"],
        ("tracery", "labelMode") => &["index", "coordinates", "both"],
        ("shutter", "mode") => &["linear", "double sided", "radial"],
        ("mask", "mode") => &["rectangle", "ellipse"],
        ("shake", "shake.mode") | ("shake", "mode") => &["rotate", "move"],
        ("overlay", "repeat") => &["off", "on"],
        ("duplicate", "repeat") => &["tile", "reflect"],
        ("timeoffset", "units") => &["frames", "seconds"],
        ("timeremap", "mode") => &["source time", "speed"],
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
    // Angle parameters stored in radians, displayed & scrubbed in degrees
    if matches!(key, "rotation" | "angle")
        || (effect_kind == "directionalblur" && key == "direction")
    {
        return ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 1)
            .with_scale_factor(std::f64::consts::PI / 180.0);
    }

    match (effect_kind, key) {
        // Zoidium plugin effects whose opacity/threshold/percentage params are authored in 0..100
        ("radiowaves" | "timecode" | "radialshadow", "opacity")
        | ("ccpageturn", "backPageOpacity")
        | ("tracery", "threshold") => ScrubConfig::new(0.0, 100.0, 1.0, 1),
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
        ("jpegdamage", "quality") => ScrubConfig::new(1.0, 100.0, 1.0, 0),
        ("jpegdamage", "blockSize")
        | ("datamosh", "blockSize")
        | ("tracery", "gridSize")
        | ("ascii", "cellSize") => ScrubConfig::new(2.0, 256.0, 1.0, 0),
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

/// Returns the `ScrubConfig` for a numeric custom shader uniform.
/// Only explicit angle properties (`kind_hint == Some(6)`) convert radians to degrees;
/// raw PanZoid scalar uniforms (`type: 0`, e.g. `Rotation: 25.0`) stay at `scale_factor == 1.0`.
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
    } else {
        ScrubConfig::new(-10_000.0, 10_000.0, 0.05, 3)
    }
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

#[allow(clippy::too_many_arguments)]
fn render_custom_shader_uniform(
    ui: &mut Ui,
    session: &mut Session,
    target_id: &str,
    key: &str,
    label: &str,
    val: &Value,
    kind_hint: Option<usize>,
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
    let is_sampler2d = imported_prop_type == Some(8);
    let is_bool_toggle = matches!(val, Value::Bool(_))
        || kind_hint == Some(5)
        || (imported_prop_type == Some(5) && is_type5_bool_placeholder_value(val));

    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        ui.vertical(|ui| {
            ui.set_max_width((ui.available_width() - 24.0).max(80.0));
            if is_sampler2d {
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
                        let scrub_cfg = custom_shader_number_scrub_config(label, kind_hint);
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
        let _ = session.execute(Command::RemoveProperty {
            target_id: target_id.to_owned(),
            property: key.to_owned(),
        });
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
