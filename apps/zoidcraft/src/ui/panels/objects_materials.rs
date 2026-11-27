use super::effects_tab::DEFAULT_CUSTOM_SHADER_GLSL;
use super::property_rows::{
    eval_prop_str, eval_prop_u32, eval_prop_value, lookup_session_prop_ref, panel_property_row,
    show_session_color_prop, show_session_number_prop, show_session_option_prop_ex,
    show_session_vec2_prop,
};
use super::shared_widgets::show_session_asset_prop;
use super::state::{
    ShaderEditorTarget, copy_pz_json_to_clipboard, header_compact_action_button,
    paste_pz_json_from_clipboard,
};
use super::*;

// ---------------------------------------------------------------------------
// 9. Complete Reusable Material Slot Inspector (All 9 Modes + 7 PBR+ Surface Sections + NodeMaterial)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) fn show_material_slot_inspector(
    ui: &mut Ui,
    session: &mut Session,
    clip_id: &str,
    obj_id: &str,
    obj_name: &str,
    prefix: &str,
    section_title: &str,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let k = |sub: &str| format!("{prefix}.{sub}");
    let sec_id = Id::unique(("obj_mat_slot", obj_id, prefix));
    let mut open = ui
        .data_mut(|d| d.get_temp::<bool>(sec_id))
        .unwrap_or(prefix == "material");

    ui.spacing_mut().item_spacing.y = 0.0;
    let indent_px = 15.0_f32;
    let (hdr_rect, hdr_resp) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), Sense::click());
    let hdr_resp = hdr_resp.on_hover_cursor(CursorIcon::PointingHand);
    ui.painter()
        .rect_filled(hdr_rect, CornerRadius::ZERO, BG_PANEL);
    let div_y = (hdr_rect.bottom() - 1.0).round();
    ui.painter().rect_filled(
        Rect::from_min_max(
            Pos2::new(hdr_rect.left(), div_y),
            Pos2::new(hdr_rect.right(), div_y + 1.0),
        ),
        CornerRadius::ZERO,
        BORDER_ROW_SOFT,
    );

    let mut action_clicked = false;
    let mut action_hovered = false;
    let mut title_right = hdr_rect.right() - 4.0;
    if prefix == "material" {
        let btn_y = (hdr_rect.center().y - 9.0).round();
        let copy_w = (ui
            .painter()
            .layout_no_wrap("Copy".to_owned(), theme.font(FONT_COMPACT), TEXT_PRIMARY)
            .size()
            .x
            .ceil()
            + 10.0)
            .max(36.0);
        let paste_w = (ui
            .painter()
            .layout_no_wrap("Paste".to_owned(), theme.font(FONT_COMPACT), TEXT_PRIMARY)
            .size()
            .x
            .ceil()
            + 10.0)
            .max(40.0);
        let paste_rect = Rect::from_min_size(
            Pos2::new((hdr_rect.right() - paste_w - 6.0).round(), btn_y),
            Vec2::new(paste_w, 18.0),
        );
        let copy_rect = Rect::from_min_size(
            Pos2::new((paste_rect.left() - copy_w - 4.0).round(), btn_y),
            Vec2::new(copy_w, 18.0),
        );
        let (copy_clicked, copy_hov) = header_compact_action_button(
            ui,
            sec_id.with("copy_mat"),
            copy_rect,
            "Copy",
            "Copy Material JSON (Cmd+C)",
            theme,
        );
        let (paste_clicked, paste_hov) = header_compact_action_button(
            ui,
            sec_id.with("paste_mat"),
            paste_rect,
            "Paste",
            "Paste Material or Node Material JSON (Cmd+V)",
            theme,
        );
        action_hovered = copy_hov || paste_hov;
        title_right = copy_rect.left() - 4.0;
        if copy_clicked {
            action_clicked = true;
            let res = session.copy_material_as_pz_json(clip_id, obj_id);
            copy_pz_json_to_clipboard(ui.ctx(), res, "Copied Material JSON to clipboard", actions);
        } else if paste_clicked {
            action_clicked = true;
            paste_pz_json_from_clipboard(
                session,
                state,
                actions,
                "material",
                Some(clip_id.to_owned()),
                None,
                Some(obj_id.to_owned()),
                None,
                "Material JSON",
            );
        }
    }

    paint_cm3_tree_box(
        ui.painter(),
        Pos2::new(
            hdr_rect.left() + 10.5 + indent_px,
            hdr_rect.center().y + 0.5,
        ),
        open,
        if hdr_resp.hovered() && !action_hovered {
            TEXT_BRIGHT
        } else {
            TEXT_MUTED
        },
    );
    let title_x = hdr_rect.left() + 21.0 + indent_px;
    let title_clip = Rect::from_min_max(
        Pos2::new(title_x - 1.0, hdr_rect.top()),
        Pos2::new(title_right.max(title_x), hdr_rect.bottom()),
    );
    ui.painter().with_clip_rect(title_clip).text(
        Pos2::new(title_x, hdr_rect.center().y),
        Align2::LEFT_CENTER,
        section_title,
        theme.font(FONT_BODY),
        TEXT_PRIMARY,
    );

    if hdr_resp.clicked() && !action_clicked {
        open = !open;
    }
    ui.data_mut(|d| d.insert_temp(sec_id, open));

    if !open {
        return;
    }

    let prev_indent = state.inspector_root_indent;
    state.inspector_root_indent = true;
    {
        let mat_options: Vec<&str> = if session.plugins.is_enabled("material-plus") {
            vec![
                "single color",
                "texture",
                "video",
                "custom PBR",
                "custom shader",
                "PBR+ (Material+)",
                "Matcap (Material+)",
                "Image+ (Material+)",
                "Node Material",
            ]
        } else {
            vec![
                "single color",
                "texture",
                "video",
                "custom PBR",
                "custom shader",
            ]
        };
        let type_key = k("type");
        let prev_mat_type = eval_prop_u32(
            lookup_session_prop_ref(session, obj_id, &type_key),
            local_frame,
            0,
        );
        show_session_option_prop_ex(
            ui,
            session,
            obj_id,
            &type_key,
            "Material",
            &mat_options,
            0,
            false,
            local_frame,
            clip_start,
            viewport_rect,
            state,
            theme,
            actions,
        );
        let mat_type = eval_prop_u32(
            lookup_session_prop_ref(session, obj_id, &type_key),
            local_frame,
            0,
        );
        if mat_type == 8 {
            let needs_graph_init = !node_material_has_populated_graph(session, obj_id, prefix);
            if prev_mat_type != 8 || needs_graph_init {
                initialize_node_material_defaults(session, obj_id, prefix, prev_mat_type != 8);
            }
        }
        match mat_type {
            0 => {
                show_session_color_prop(
                    ui,
                    session,
                    obj_id,
                    &k("color"),
                    "Color",
                    [1.0, 1.0, 1.0],
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_option_prop_ex(
                    ui,
                    session,
                    obj_id,
                    &k("side"),
                    "Render side",
                    &["front", "back", "double"],
                    0,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            }
            1 => {
                show_session_asset_prop(
                    ui,
                    session,
                    obj_id,
                    &k("texture"),
                    "Image",
                    false,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_option_prop_ex(
                    ui,
                    session,
                    obj_id,
                    &k("side"),
                    "Render side",
                    &["front", "back", "double"],
                    0,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            }
            2 => {
                show_session_asset_prop(
                    ui,
                    session,
                    obj_id,
                    &k("texture"),
                    "Video",
                    false,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_color_prop(
                    ui,
                    session,
                    obj_id,
                    &k("color"),
                    "Color",
                    [1.0, 1.0, 1.0],
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("opacity"),
                    "Opacity",
                    1.0,
                    ScrubConfig::new(0.0, 1.0, 0.05, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            }
            3 => {
                // Custom PBR + UV Normalized / Wrap / Repeat / Offset / NormalMap / Reflection
                show_session_color_prop(
                    ui,
                    session,
                    obj_id,
                    &k("color"),
                    "Color",
                    [1.0, 1.0, 1.0],
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_color_prop(
                    ui,
                    session,
                    obj_id,
                    &k("emissive"),
                    "Emissive",
                    [0.0, 0.0, 0.0],
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("roughness"),
                    "Roughness",
                    0.5,
                    ScrubConfig::new(0.0, 1.0, 0.02, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("metalness"),
                    "Metalness",
                    0.5,
                    ScrubConfig::new(0.0, 1.0, 0.02, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_asset_prop(
                    ui,
                    session,
                    obj_id,
                    &k("texture"),
                    "Albedo map",
                    false,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_asset_prop(
                    ui,
                    session,
                    obj_id,
                    &k("normalMap"),
                    "Normal map",
                    false,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec2_prop(
                    ui,
                    session,
                    obj_id,
                    &k("normalScale"),
                    "Normal scale",
                    [1.0, 1.0],
                    ScrubConfig::new(-10.0, 10.0, 0.05, 2),
                    true,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_option_prop_ex(
                    ui,
                    session,
                    obj_id,
                    &k("uvNormalized"),
                    "UV normalized",
                    &["off", "on"],
                    1,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_option_prop_ex(
                    ui,
                    session,
                    obj_id,
                    &k("wrap"),
                    "UV wrap",
                    &["Clamp", "Repeat", "Mirrored"],
                    1,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec2_prop(
                    ui,
                    session,
                    obj_id,
                    &k("repeat"),
                    "UV repeat",
                    [1.0, 1.0],
                    ScrubConfig::new(-100.0, 100.0, 0.05, 2),
                    true,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec2_prop(
                    ui,
                    session,
                    obj_id,
                    &k("offset"),
                    "UV offset",
                    [0.0, 0.0],
                    ScrubConfig::new(-100.0, 100.0, 0.02, 2),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("rotation"),
                    "UV rotation",
                    0.0,
                    ScrubConfig::new(-360.0, 360.0, 1.0, 1),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("reflection"),
                    "Reflection",
                    0.0,
                    ScrubConfig::new(0.0, 2.0, 0.05, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("opacity"),
                    "Opacity",
                    1.0,
                    ScrubConfig::new(0.0, 1.0, 0.02, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_option_prop_ex(
                    ui,
                    session,
                    obj_id,
                    &k("side"),
                    "Render side",
                    &["front", "back", "double"],
                    0,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            }
            4 => {
                let frag_key = k("fragShader");
                let cur_code = eval_prop_str(
                    lookup_session_prop_ref(session, obj_id, &frag_key),
                    local_frame,
                    DEFAULT_CUSTOM_SHADER_GLSL,
                );
                cm3_property_row_aligned(
                    ui,
                    "Fragment shader",
                    true,
                    state.prop_columns,
                    theme,
                    |_| {},
                    |ui| {
                        if cm3_prop_button(ui, "Edit shader", ButtonVariant::Normal, false, theme)
                            .clicked()
                        {
                            state.shader_editor = Some(ShaderEditorTarget {
                                effect_id: obj_id.to_owned(),
                                property: frag_key.clone(),
                                effect_name: obj_name.to_owned(),
                                code: cur_code.clone(),
                                font_px: 14.0,
                                error_message: None,
                            });
                            state.shader_editor_just_opened = true;
                            let spec = FloatingWindowSpec::new("shader_editor", "Shader Editor")
                                .with_subtitle(obj_name.to_owned())
                                .with_default_size(Vec2::new(560.0, 440.0));
                            state.floating_windows.open_or_focus(&spec, viewport_rect);
                        }
                    },
                );
            }
            5 => {
                // PBR+ (Material+): All Maps + AO + All 7 Surface Sections
                show_session_color_prop(
                    ui,
                    session,
                    obj_id,
                    &k("color"),
                    "Base color",
                    [1.0, 1.0, 1.0],
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_asset_prop(
                    ui,
                    session,
                    obj_id,
                    &k("texture"),
                    "Base color map",
                    false,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_color_prop(
                    ui,
                    session,
                    obj_id,
                    &k("emissive"),
                    "Emissive",
                    [0.0, 0.0, 0.0],
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("emissiveIntensity"),
                    "Emissive int",
                    1.0,
                    ScrubConfig::new(0.0, 20.0, 0.05, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_asset_prop(
                    ui,
                    session,
                    obj_id,
                    &k("emissiveMap"),
                    "Emissive map",
                    false,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("roughness"),
                    "Roughness",
                    0.5,
                    ScrubConfig::new(0.0, 1.0, 0.02, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_asset_prop(
                    ui,
                    session,
                    obj_id,
                    &k("roughnessMap"),
                    "Roughness map",
                    false,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("metalness"),
                    "Metalness",
                    0.0,
                    ScrubConfig::new(0.0, 1.0, 0.02, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_asset_prop(
                    ui,
                    session,
                    obj_id,
                    &k("metalnessMap"),
                    "Metalness map",
                    false,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_asset_prop(
                    ui,
                    session,
                    obj_id,
                    &k("normalMap"),
                    "Normal map",
                    false,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_asset_prop(
                    ui,
                    session,
                    obj_id,
                    &k("alphaMap"),
                    "Alpha map",
                    false,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("alphaTest"),
                    "Alpha cutoff",
                    0.0,
                    ScrubConfig::new(0.0, 1.0, 0.01, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("opacity"),
                    "Opacity",
                    1.0,
                    ScrubConfig::new(0.0, 1.0, 0.02, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_option_prop_ex(
                    ui,
                    session,
                    obj_id,
                    &k("side"),
                    "Render side",
                    &["front", "back", "double"],
                    0,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );

                // 7 PBR+ Surface Sections
                cm3_collapsible_section(
                    ui,
                    Id::unique((obj_id, prefix, "surf_ao")),
                    "Surface: Ambient Occlusion",
                    false,
                    theme,
                    |ui| {
                        show_session_option_prop_ex(
                            ui,
                            session,
                            obj_id,
                            &k("aoEnabled"),
                            "Enable AO",
                            &["off", "on"],
                            0,
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_asset_prop(
                            ui,
                            session,
                            obj_id,
                            &k("aoTexture"),
                            "AO map",
                            false,
                            local_frame,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            obj_id,
                            &k("aoIntensity"),
                            "AO intensity",
                            1.0,
                            ScrubConfig::new(0.0, 4.0, 0.05, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_color_prop(
                            ui,
                            session,
                            obj_id,
                            &k("aoColor"),
                            "AO color",
                            [0.0, 0.0, 0.0],
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            obj_id,
                            &k("aoDistance"),
                            "AO distance",
                            1.0,
                            ScrubConfig::new(0.01, 100.0, 0.05, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    },
                );
                cm3_collapsible_section(
                    ui,
                    Id::unique((obj_id, prefix, "surf_cel")),
                    "Surface: Cel Shading",
                    false,
                    theme,
                    |ui| {
                        show_session_option_prop_ex(
                            ui,
                            session,
                            obj_id,
                            &k("celEnabled"),
                            "Enable cel",
                            &["off", "on"],
                            0,
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            obj_id,
                            &k("celSteps"),
                            "Steps",
                            4.0,
                            ScrubConfig::new(1.0, 32.0, 1.0, 0),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            obj_id,
                            &k("celSmoothness"),
                            "Smoothness",
                            0.05,
                            ScrubConfig::new(0.0, 1.0, 0.01, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            obj_id,
                            &k("celSpecularSteps"),
                            "Spec steps",
                            2.0,
                            ScrubConfig::new(1.0, 16.0, 1.0, 0),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    },
                );
                cm3_collapsible_section(
                    ui,
                    Id::unique((obj_id, prefix, "surf_fresnel")),
                    "Surface: Fresnel Rim",
                    false,
                    theme,
                    |ui| {
                        show_session_option_prop_ex(
                            ui,
                            session,
                            obj_id,
                            &k("fresnelEnabled"),
                            "Enable fresnel",
                            &["off", "on"],
                            0,
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_color_prop(
                            ui,
                            session,
                            obj_id,
                            &k("fresnelColor"),
                            "Rim color",
                            [1.0, 1.0, 1.0],
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            obj_id,
                            &k("fresnelPower"),
                            "Power",
                            3.0,
                            ScrubConfig::new(0.2, 16.0, 0.1, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            obj_id,
                            &k("fresnelMixStrength"),
                            "Mix strength",
                            1.0,
                            ScrubConfig::new(0.0, 1.0, 0.02, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    },
                );
                cm3_collapsible_section(
                    ui,
                    Id::unique((obj_id, prefix, "surf_ramp")),
                    "Surface: Color Ramp",
                    false,
                    theme,
                    |ui| {
                        show_session_option_prop_ex(
                            ui,
                            session,
                            obj_id,
                            &k("rampEnabled"),
                            "Enable ramp",
                            &["off", "on"],
                            0,
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_option_prop_ex(
                            ui,
                            session,
                            obj_id,
                            &k("rampSource"),
                            "Source",
                            &["Luminance", "Normal Y", "Fresnel", "UV X", "UV Y"],
                            0,
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_color_prop(
                            ui,
                            session,
                            obj_id,
                            &k("rampColorA"),
                            "Low color",
                            [0.1, 0.15, 0.25],
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_color_prop(
                            ui,
                            session,
                            obj_id,
                            &k("rampColorB"),
                            "High color",
                            [1.0, 0.9, 0.7],
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            obj_id,
                            &k("rampFactor"),
                            "Factor",
                            0.5,
                            ScrubConfig::new(0.0, 1.0, 0.02, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    },
                );
                cm3_collapsible_section(
                    ui,
                    Id::unique((obj_id, prefix, "surf_bump")),
                    "Surface: 3D Bump Noise",
                    false,
                    theme,
                    |ui| {
                        show_session_option_prop_ex(
                            ui,
                            session,
                            obj_id,
                            &k("bumpNoise"),
                            "Enable bump",
                            &["off", "on"],
                            0,
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            obj_id,
                            &k("bumpNoiseScale"),
                            "Scale",
                            4.0,
                            ScrubConfig::new(0.1, 100.0, 0.2, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            obj_id,
                            &k("bumpStrength"),
                            "Strength",
                            0.3,
                            ScrubConfig::new(0.0, 4.0, 0.05, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    },
                );
                cm3_collapsible_section(
                    ui,
                    Id::unique((obj_id, prefix, "surf_uv")),
                    "Surface: UV Projection",
                    false,
                    theme,
                    |ui| {
                        show_session_option_prop_ex(
                            ui,
                            session,
                            obj_id,
                            &k("projection"),
                            "Projection",
                            &["UV", "Spherical", "Cylindrical", "Cubic"],
                            0,
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_vec2_prop(
                            ui,
                            session,
                            obj_id,
                            &k("projScale"),
                            "Proj scale",
                            [1.0, 1.0],
                            ScrubConfig::new(-100.0, 100.0, 0.05, 2),
                            true,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_vec2_prop(
                            ui,
                            session,
                            obj_id,
                            &k("projTranslate"),
                            "Proj offset",
                            [0.0, 0.0],
                            ScrubConfig::new(-100.0, 100.0, 0.02, 2),
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            obj_id,
                            &k("projRotation"),
                            "Proj rotation",
                            0.0,
                            ScrubConfig::new(-360.0, 360.0, 1.0, 1),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_vec2_prop(
                            ui,
                            session,
                            obj_id,
                            &k("repeat"),
                            "UV repeat",
                            [1.0, 1.0],
                            ScrubConfig::new(-100.0, 100.0, 0.05, 2),
                            true,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_vec2_prop(
                            ui,
                            session,
                            obj_id,
                            &k("offset"),
                            "UV offset",
                            [0.0, 0.0],
                            ScrubConfig::new(-100.0, 100.0, 0.02, 2),
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    },
                );
                cm3_collapsible_section(
                    ui,
                    Id::unique((obj_id, prefix, "surf_env")),
                    "Surface: Env Reflection",
                    false,
                    theme,
                    |ui| {
                        show_session_option_prop_ex(
                            ui,
                            session,
                            obj_id,
                            &k("reflection"),
                            "Enable env",
                            &["off", "on"],
                            0,
                            false,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            obj_id,
                            &k("reflectionIntensity"),
                            "Reflectivity",
                            1.0,
                            ScrubConfig::new(0.0, 4.0, 0.05, 2),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_color_prop(
                            ui,
                            session,
                            obj_id,
                            &k("reflectionEnvColorTop"),
                            "Sky color",
                            [0.7, 0.85, 1.0],
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_color_prop(
                            ui,
                            session,
                            obj_id,
                            &k("reflectionEnvColorBottom"),
                            "Ground color",
                            [0.2, 0.2, 0.25],
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_session_number_prop(
                            ui,
                            session,
                            obj_id,
                            &k("reflectionEnvRotation"),
                            "Env rotation",
                            0.0,
                            ScrubConfig::new(-360.0, 360.0, 1.0, 1),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    },
                );
            }
            6 => {
                // Matcap
                show_session_asset_prop(
                    ui,
                    session,
                    obj_id,
                    &k("texture"),
                    "Matcap image",
                    false,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_color_prop(
                    ui,
                    session,
                    obj_id,
                    &k("color"),
                    "Tint color",
                    [1.0, 1.0, 1.0],
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("brightness"),
                    "Brightness",
                    1.0,
                    ScrubConfig::new(0.0, 4.0, 0.05, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("opacity"),
                    "Opacity",
                    1.0,
                    ScrubConfig::new(0.0, 1.0, 0.02, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_option_prop_ex(
                    ui,
                    session,
                    obj_id,
                    &k("side"),
                    "Render side",
                    &["front", "back", "double"],
                    0,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            }
            7 => {
                // Image+
                show_session_asset_prop(
                    ui,
                    session,
                    obj_id,
                    &k("texture"),
                    "Image",
                    false,
                    local_frame,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_option_prop_ex(
                    ui,
                    session,
                    obj_id,
                    &k("wrap"),
                    "UV wrap",
                    &["Clamp", "Repeat", "Mirrored"],
                    0,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec2_prop(
                    ui,
                    session,
                    obj_id,
                    &k("repeat"),
                    "Repeat",
                    [1.0, 1.0],
                    ScrubConfig::new(-100.0, 100.0, 0.05, 2),
                    true,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec2_prop(
                    ui,
                    session,
                    obj_id,
                    &k("offset"),
                    "Offset",
                    [0.0, 0.0],
                    ScrubConfig::new(-100.0, 100.0, 0.02, 2),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("rotation"),
                    "Rotation",
                    0.0,
                    ScrubConfig::new(-360.0, 360.0, 1.0, 1),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_number_prop(
                    ui,
                    session,
                    obj_id,
                    &k("opacity"),
                    "Opacity",
                    1.0,
                    ScrubConfig::new(0.0, 1.0, 0.02, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_option_prop_ex(
                    ui,
                    session,
                    obj_id,
                    &k("side"),
                    "Render side",
                    &["front", "back", "double"],
                    0,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
            }
            8 => {
                // NodeMaterial -> Opens real interactive Node Graph Editor!
                let res_key = k("bakeResolution");
                let cur_res_raw = eval_prop_u32(
                    lookup_session_prop_ref(session, obj_id, &res_key),
                    local_frame,
                    512,
                );
                let cur_res = match cur_res_raw {
                    0 => 128,
                    1 => 256,
                    2 => 512,
                    3 => 1024,
                    4 => 2048,
                    other => other,
                };
                let res_opts = [256u32, 512, 1024, 2048];
                let mut res_idx = res_opts.iter().position(|&r| r == cur_res).unwrap_or(1);
                if panel_property_row(
                    ui,
                    "Bake resolution",
                    true,
                    state,
                    theme,
                    |_| {},
                    |ui| {
                        cm3_dropdown(
                            ui,
                            Id::unique((obj_id, prefix, "node_mat_res")),
                            &mut res_idx,
                            &["256×256", "512×512", "1024×1024", "2048×2048"],
                            false,
                            viewport_rect,
                            theme,
                        )
                    },
                ) {
                    let _ = session.execute(Command::SetPropertyValue {
                        target_id: obj_id.to_owned(),
                        property: res_key,
                        value: json!(res_opts[res_idx]),
                    });
                }
                let graph_key = k("graph");
                let open_node_mat = Frame::NONE
                    .inner_margin(egui::Margin {
                        left: 21,
                        right: 12,
                        top: 5,
                        bottom: 5,
                    })
                    .show(ui, |ui| {
                        cm3_prop_button(
                            ui,
                            "Edit Node Material",
                            ButtonVariant::Primary,
                            true,
                            theme,
                        )
                        .clicked()
                    })
                    .inner;
                if open_node_mat {
                    let cur_graph = eval_prop_value(
                        lookup_session_prop_ref(session, obj_id, &graph_key),
                        local_frame,
                    );
                    state.node_graph_editor.open_for_target(
                        NodeGraphTargetKind::NodeMaterial,
                        obj_id,
                        &graph_key,
                        obj_name,
                        cur_graph.as_ref(),
                    );
                    let (_, clamped_sz) =
                        compute_clamped_editor_window_rect(viewport_rect, 860.0, 560.0);
                    let spec =
                        FloatingWindowSpec::new("pro_node_graph_editor", "Node Graph Editor")
                            .with_default_size(clamped_sz);
                    state.floating_windows.open_or_focus(&spec, viewport_rect);
                }
            }
            _ => {}
        }
    }
    state.inspector_root_indent = prev_indent;
}

/// Formats a canonical material property key (`"material.graph"`, `"frontMaterial.bakeResolution"`),
/// normalizing callers that pass either `"material"` or `"material."`.
fn node_material_prop_key(prefix: &str, suffix: &str) -> String {
    let base = prefix.trim_end_matches('.');
    format!("{base}.{suffix}")
}

/// Returns `true` when the `NodeMaterial` at `(obj_id, prefix)` already has a non-empty `nodes` graph.
pub(super) fn node_material_has_populated_graph(
    session: &Session,
    obj_id: &str,
    prefix: &str,
) -> bool {
    let graph_key = node_material_prop_key(prefix, "graph");
    eval_prop_value(lookup_session_prop_ref(session, obj_id, &graph_key), 0.0)
        .and_then(|v| {
            if let Some(s) = v.as_str() {
                serde_json::from_str::<Value>(s).ok()
            } else {
                Some(v)
            }
        })
        .and_then(|v| {
            v.get("nodes")
                .and_then(Value::as_array)
                .map(|a| !a.is_empty())
        })
        .unwrap_or(false)
}

/// Initializes a 3D object's `Node Material` with the default procedural graph
/// (`Perlin Noise -> Color Ramp -> Material Output`) and `resolution: 512` (`bakeResolution: 2`)
/// so it renders and exports/copies cleanly immediately on material switch.
pub(super) fn initialize_node_material_defaults(
    session: &mut Session,
    obj_id: &str,
    prefix: &str,
    just_switched: bool,
) {
    let res_key = node_material_prop_key(prefix, "bakeResolution");
    let graph_key = node_material_prop_key(prefix, "graph");
    let needs_res = just_switched || lookup_session_prop_ref(session, obj_id, &res_key).is_none();
    let needs_graph = !node_material_has_populated_graph(session, obj_id, prefix);
    if !needs_res && !needs_graph {
        return;
    }
    let started_tx = if !session.history.is_in_transaction() {
        session
            .begin_transaction("initialize node material")
            .is_ok()
    } else {
        false
    };
    if needs_res {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: obj_id.to_owned(),
            property: res_key,
            value: json!(512),
        });
    }
    if needs_graph {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: obj_id.to_owned(),
            property: graph_key,
            value: NodeGraphEditorState::default_node_material_graph_json(),
        });
    }
    if started_tx && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    }
}
