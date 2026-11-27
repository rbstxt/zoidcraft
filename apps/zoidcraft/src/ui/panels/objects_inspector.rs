use super::effects_inspector::summarize_complex_json_value;
use super::objects_extended::show_extended_3d_object_props;
use super::objects_materials::show_material_slot_inspector;
use super::objects_particles::show_particles_3d_inspector;
use super::objects_tab::is_container_object;
use super::property_rows::{
    apply_scrub_to_session, eval_prop_f64, eval_prop_str, eval_prop_u32, eval_prop_vec3,
    lookup_session_prop_ref, open_expression_window, panel_property_row, show_session_color_prop,
    show_session_color_rgba_prop, show_session_number_prop, show_session_option_prop,
    show_session_option_prop_ex, show_session_text_prop, show_session_text_prop_ex,
    show_session_vec2_prop, show_session_vec3_prop, show_session_vec4_prop,
    show_session_vec4_prop_ex,
};
use super::shared_widgets::{
    apply_paired_scalar_scrub, empty_panel_message, eval_session_expr_f64,
    property_header_with_actions, show_font_selector_row, show_session_asset_prop,
};
use super::state::{
    commit_inspector_tx_if_open, copy_pz_json_to_clipboard, paste_pz_json_from_clipboard,
    touch_inspector_tx_owner,
};
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn show_object_inspector(
    ui: &mut Ui,
    inspector_rect: Rect,
    viewport_rect: Rect,
    clip: &SessionClip,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let Some(obj_id) = session.selection.object_3d_ids.first().cloned() else {
        empty_panel_message(ui, "select an object", theme);
        return;
    };
    let Some(obj) = clip.objects_3d.iter().find(|o| o.id == obj_id).cloned() else {
        empty_panel_message(ui, "select an object", theme);
        return;
    };

    cm3_column_adjust_splitter(
        ui,
        Id::unique("obj_inspector_col_splitter"),
        inspector_rect,
        &mut state.prop_columns,
    );

    let local_frame = session.playback.current_frame - clip.start;
    let clip_start = clip.start;

    ScrollArea::vertical()
        .id_salt(("obj_inspector_scroll", &obj.id))
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            let (root_open, hdr_act) = property_header_with_actions(
                ui,
                Id::unique(("obj_root_hdr", &obj.id)),
                &obj.name,
                state.prop_columns.label_ratio,
                theme,
            );
            if hdr_act.copy_json {
                let res =
                    session.copy_objects_3d_as_pz_json(&clip.id, std::slice::from_ref(&obj.id));
                copy_pz_json_to_clipboard(
                    ui.ctx(),
                    res,
                    "Copied 3D object JSON to clipboard",
                    actions,
                );
            }
            if hdr_act.paste_json {
                let parent_group_id = if obj.kind.eq_ignore_ascii_case("group") {
                    Some(obj.id.clone())
                } else {
                    None
                };
                paste_pz_json_from_clipboard(
                    session,
                    state,
                    actions,
                    "objects3d",
                    Some(clip.id.clone()),
                    parent_group_id,
                    None,
                    None,
                    "3D object JSON",
                );
            }
            if hdr_act.reset {
                let _ = session.execute(Command::ResetPropertyList {
                    target_id: obj.id.clone(),
                });
            }
            if hdr_act.duplicate {
                let _ = session.execute(Command::DuplicateObject3D {
                    clip_id: clip.id.clone(),
                    object_id: obj.id.clone(),
                });
            }
            if hdr_act.remove {
                commit_inspector_tx_if_open(session, state);
                let _ = session.execute(Command::DeleteObject3D {
                    clip_id: clip.id.clone(),
                    object_id: obj.id.clone(),
                });
                return;
            }
            if !root_open {
                return;
            }
            state.inspector_root_indent = true;

            let is_camera = obj.kind.contains("camera");
            let is_light = obj.kind.contains("light");
            let is_text = obj.kind == "text" || obj.kind == "text_3d";
            let is_particles = obj.kind == "particles";

            // Build reusable target-object dropdown list for Track to / Follow constraints
            let constraint_targets: Vec<(String, String)> = clip
                .objects_3d
                .iter()
                .filter(|other| other.id != obj.id)
                .map(|other| (other.id.clone(), other.name.clone()))
                .collect();

            if is_camera {
                // Camera inspector (exact Zoidium 11_objects_camera.png flat layout)
                show_session_vec3_prop(
                    ui,
                    session,
                    &obj.id,
                    "position",
                    "Position",
                    [0.0, 0.0, 80.0],
                    ScrubConfig::new(-100_000.0, 100_000.0, 1.0, 2),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec3_prop(
                    ui,
                    session,
                    &obj.id,
                    "rotation",
                    "Rotation",
                    [0.0, 0.0, 0.0],
                    ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 2)
                        .with_scale_factor(std::f64::consts::PI / 180.0),
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
                    &obj.id,
                    "eulerOrder",
                    "Rotation order",
                    &["XYZ", "YZX", "ZXY", "XZY", "YXZ", "ZYX"],
                    0,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );

                if session.plugins.is_enabled("constraints") {
                    show_constraint_target_rows(
                        ui,
                        session,
                        &obj.id,
                        &constraint_targets,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }

                // All 8 procedural Camera Shake properties (11_objects_camera.png)
                show_session_number_prop(
                    ui,
                    session,
                    &obj.id,
                    "shakeAmplitude",
                    "Shake amp",
                    0.0,
                    ScrubConfig::new(0.0, 1.0, 0.01, 3),
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
                    &obj.id,
                    "shakeSpeed",
                    "Shake freq",
                    30.0,
                    ScrubConfig::new(0.0, 1000.0, 0.5, 2),
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
                    &obj.id,
                    "shake.mode",
                    "Shake mode",
                    &["rotate", "move"],
                    0,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec4_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "shake.amplitude",
                    "Amplitude",
                    [12.0, 12.0, 40.0, 20.0],
                    ScrubConfig::new(0.0, 1000.0, 0.5, 1),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec4_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "shake.frequency",
                    "Frequency",
                    [0.60, 0.80, 0.70, 0.90],
                    ScrubConfig::new(0.0, 100.0, 0.05, 2),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec4_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "shake.phase",
                    "Phase",
                    [0.0, 11.0, 20.0, 31.0],
                    ScrubConfig::new(-1000.0, 1000.0, 0.5, 1),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec4_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "shake.noise",
                    "Noise",
                    [0.90, 0.90, 0.90, 0.55],
                    ScrubConfig::new(0.0, 10.0, 0.05, 2),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec4_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "shake.smooth",
                    "Smooth",
                    [1.0, 1.0, 1.0, 1.0],
                    ScrubConfig::new(0.0, 10.0, 1.0, 0),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );

                // Camera projection & Camera+ optics below shake rows
                show_session_option_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "camera.kind",
                    "Camera type",
                    &["Perspective", "Orthographic"],
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
                    &obj.id,
                    "fov",
                    "FOV",
                    60.0,
                    ScrubConfig::new(1.0, 179.0, 0.5, 1),
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
                    &obj.id,
                    "near",
                    "Near",
                    0.1,
                    ScrubConfig::new(0.01, 1000.0, 0.1, 2),
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
                    &obj.id,
                    "far",
                    "Far",
                    5000.0,
                    ScrubConfig::new(1.0, 100_000.0, 10.0, 0),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                if session.plugins.is_enabled("camera-plus") {
                    show_camera_plus_controls(
                        ui,
                        session,
                        &obj.id,
                        clip,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
            } else if is_light {
                // Light inspector: flat, no Rotation/Scale; conditional by light.kind (including Ambient)
                show_session_option_prop(
                    ui,
                    session,
                    &obj.id,
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
                let default_lkind = match obj.kind.as_str() {
                    "point_light" => 1,
                    "directional_light" => 2,
                    "hemisphere_light" => 3,
                    "ambient_light" => 4,
                    _ => 0,
                };
                show_session_option_prop_ex(
                    ui,
                    session,
                    &obj.id,
                    "light.kind",
                    "Light type",
                    &["Spot", "Point", "Directional", "Hemisphere", "Ambient"],
                    default_lkind,
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                let lkind = eval_prop_u32(
                    lookup_session_prop_ref(session, &obj.id, "light.kind"),
                    local_frame,
                    default_lkind,
                );
                if lkind != 3 && lkind != 4 {
                    show_session_vec3_prop(
                        ui,
                        session,
                        &obj.id,
                        "position",
                        "Position",
                        [0.0, 10.0, 0.0],
                        ScrubConfig::new(-100_000.0, 100_000.0, 1.0, 2),
                        false,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                if lkind == 0 || lkind == 2 {
                    show_session_vec3_prop(
                        ui,
                        session,
                        &obj.id,
                        "target",
                        "Target",
                        [0.0, 0.0, 0.0],
                        ScrubConfig::new(-100_000.0, 100_000.0, 1.0, 2),
                        false,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                show_session_color_prop(
                    ui,
                    session,
                    &obj.id,
                    "color",
                    if lkind == 3 { "Sky color" } else { "Color" },
                    [1.0, 1.0, 1.0],
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                if lkind == 3 {
                    show_session_color_prop(
                        ui,
                        session,
                        &obj.id,
                        "groundColor",
                        "Ground color",
                        [1.0, 1.0, 1.0],
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                show_session_number_prop(
                    ui,
                    session,
                    &obj.id,
                    "intensity",
                    "Intensity",
                    1.0,
                    ScrubConfig::new(0.0, 100.0, 0.1, 2),
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                if lkind == 0 {
                    show_session_number_prop(
                        ui,
                        session,
                        &obj.id,
                        "angle",
                        "Angle",
                        60.0,
                        ScrubConfig::new(0.0, 90.0, 0.5, 1),
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
                        &obj.id,
                        "penumbra",
                        "Penumbra",
                        0.0,
                        ScrubConfig::new(0.0, 1.0, 0.05, 2),
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                if lkind == 0 || lkind == 1 {
                    show_session_number_prop(
                        ui,
                        session,
                        &obj.id,
                        "distance",
                        "Range",
                        0.0,
                        ScrubConfig::new(0.0, 10_000.0, 1.0, 1),
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
                        &obj.id,
                        "decay",
                        "Decay",
                        1.0,
                        ScrubConfig::new(0.0, 10.0, 0.1, 2),
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                if obj.kind.contains("area-light") || obj.kind.contains("area_light") {
                    show_session_number_prop(
                        ui,
                        session,
                        &obj.id,
                        "width",
                        "Area width",
                        20.0,
                        ScrubConfig::new(0.1, 5000.0, 0.5, 2),
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
                        &obj.id,
                        "height",
                        "Area height",
                        12.0,
                        ScrubConfig::new(0.1, 5000.0, 0.5, 2),
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                } else if obj.kind.contains("ies-light") || obj.kind.contains("ies_light") {
                    show_session_asset_prop(
                        ui,
                        session,
                        &obj.id,
                        "iesProfile",
                        "IES profile",
                        false,
                        local_frame,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                } else if obj.kind.contains("sun-light") || obj.kind.contains("sun_light") {
                    show_session_number_prop(
                        ui,
                        session,
                        &obj.id,
                        "sunElevation",
                        "Elevation (°)",
                        45.0,
                        ScrubConfig::new(-90.0, 90.0, 0.5, 1),
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
                        &obj.id,
                        "sunAzimuth",
                        "Azimuth (°)",
                        135.0,
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
                        &obj.id,
                        "temperature",
                        "Temp (K)",
                        5600.0,
                        ScrubConfig::new(1000.0, 15_000.0, 50.0, 0),
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
                if lkind != 3 && lkind != 4 {
                    show_session_option_prop_ex(
                        ui,
                        session,
                        &obj.id,
                        "castShadow",
                        "Cast shadow",
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
                    let casts = eval_prop_u32(
                        lookup_session_prop_ref(session, &obj.id, "castShadow"),
                        local_frame,
                        0,
                    ) != 0;
                    if casts {
                        let cur_sz = eval_prop_u32(
                            lookup_session_prop_ref(session, &obj.id, "shadowMapSize"),
                            local_frame,
                            1024,
                        );
                        let map_sizes = [512u32, 1024, 2048, 4096];
                        let mut sz_idx = map_sizes.iter().position(|&s| s == cur_sz).unwrap_or(1);
                        let sz_changed = panel_property_row(
                            ui,
                            "Shadow map size",
                            true,
                            state,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_dropdown(
                                    ui,
                                    Id::unique((&obj.id, "shadow_map_size")),
                                    &mut sz_idx,
                                    &["512", "1024", "2048", "4096"],
                                    false,
                                    viewport_rect,
                                    theme,
                                )
                            },
                        );
                        if sz_changed {
                            let _ = session.execute(Command::SetPropertyValue {
                                target_id: obj.id.clone(),
                                property: "shadowMapSize".to_owned(),
                                value: json!(map_sizes[sz_idx]),
                            });
                        }
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            "shadowBias",
                            "Shadow bias",
                            -0.0005,
                            ScrubConfig::new(-0.05, 0.05, 0.0001, 5),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                }
                if session.plugins.is_enabled("constraints") {
                    show_constraint_target_rows(
                        ui,
                        session,
                        &obj.id,
                        &constraint_targets,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }
            } else {
                // Standard flat object properties (Text3D, Shapes, Model, Group, Particles)
                show_session_option_prop(
                    ui,
                    session,
                    &obj.id,
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
                show_session_vec3_prop(
                    ui,
                    session,
                    &obj.id,
                    "position",
                    "Position",
                    [0.0, 0.0, 0.0],
                    ScrubConfig::new(-100_000.0, 100_000.0, 1.0, 2),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec3_prop(
                    ui,
                    session,
                    &obj.id,
                    "rotation",
                    "Rotation",
                    [0.0, 0.0, 0.0],
                    ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 2)
                        .with_scale_factor(std::f64::consts::PI / 180.0),
                    false,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );
                show_session_vec3_prop(
                    ui,
                    session,
                    &obj.id,
                    "scale",
                    "Scale",
                    [1.0, 1.0, 1.0],
                    ScrubConfig::new(-1000.0, 1000.0, 0.05, 3),
                    true,
                    local_frame,
                    clip_start,
                    viewport_rect,
                    state,
                    theme,
                    actions,
                );

                if is_text {
                    // Non-animated 3D Text geometry properties matching CM3 Spec 03 §3.4
                    show_session_text_prop_ex(
                        ui,
                        session,
                        &obj.id,
                        "text",
                        "Text",
                        "text",
                        true,
                        local_frame,
                        state,
                        theme,
                    );
                    // Font selector (bundled + project font assets + OS system fonts + file picker)
                    show_font_selector_row(
                        ui,
                        session,
                        &obj.id,
                        "text3d_font",
                        local_frame,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );

                    // Combined non-animated Size ([height, thickness], default [20, 3], 0 decimals)
                    touch_inspector_tx_owner(state, &obj.id, "text3d_size");
                    let h_val = eval_prop_f64(
                        lookup_session_prop_ref(session, &obj.id, "height"),
                        local_frame,
                        20.0,
                    );
                    let t_val = eval_prop_f64(
                        lookup_session_prop_ref(session, &obj.id, "thickness"),
                        local_frame,
                        3.0,
                    );
                    let mut size_vec = [h_val, t_val];
                    let size_ev = panel_property_row(
                        ui,
                        "Size",
                        true,
                        state,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_vector2(
                                ui,
                                Id::unique((&obj.id, "text3d_size")),
                                &mut size_vec,
                                ScrubConfig::new(0.1, 500.0, 1.0, 0),
                                false,
                                theme,
                            )
                        },
                    );
                    apply_paired_scalar_scrub(
                        session,
                        &obj.id,
                        "text3d_size",
                        ("height", size_vec[0]),
                        ("thickness", size_vec[1].max(0.0)),
                        size_ev,
                        state,
                    );

                    // Detail (non-animated scalar, default 5)
                    touch_inspector_tx_owner(state, &obj.id, "detail");
                    let mut detail_val = eval_prop_f64(
                        lookup_session_prop_ref(session, &obj.id, "detail"),
                        local_frame,
                        5.0,
                    );
                    let detail_ev = panel_property_row(
                        ui,
                        "Detail",
                        true,
                        state,
                        theme,
                        |_| {},
                        |ui| {
                            cm3_scrub_number(
                                ui,
                                Id::unique((&obj.id, "detail")),
                                &mut detail_val,
                                ScrubConfig::new(1.0, 200.0, 1.0, 0),
                                theme,
                            )
                        },
                    );
                    apply_scrub_to_session(
                        session,
                        &obj.id,
                        "detail",
                        false,
                        local_frame,
                        json!(detail_val.round().clamp(1.0, 200.0)),
                        detail_ev,
                        state,
                    );

                    // Bevel group (non-animated off/on, default 0). When enabled, all bevel
                    // sub-controls are grouped consecutively right below `Bevel`; when off, they hide.
                    show_session_option_prop_ex(
                        ui,
                        session,
                        &obj.id,
                        "bevelEnabled",
                        "Bevel",
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
                    let bevel_on = eval_prop_u32(
                        lookup_session_prop_ref(session, &obj.id, "bevelEnabled"),
                        local_frame,
                        0,
                    ) != 0;
                    if bevel_on {
                        touch_inspector_tx_owner(state, &obj.id, "text3d_bevel_size");
                        let bt = eval_prop_f64(
                            lookup_session_prop_ref(session, &obj.id, "bevelThickness"),
                            local_frame,
                            0.1,
                        );
                        let bs = eval_prop_f64(
                            lookup_session_prop_ref(session, &obj.id, "bevelSize"),
                            local_frame,
                            0.5,
                        );
                        let mut bevel_vec = [bt, bs];
                        let bevel_ev = panel_property_row(
                            ui,
                            "Size",
                            true,
                            state,
                            theme,
                            |_| {},
                            |ui| {
                                cm3_vector2(
                                    ui,
                                    Id::unique((&obj.id, "text3d_bevel_size")),
                                    &mut bevel_vec,
                                    ScrubConfig::new(0.0, 100.0, 0.05, 2),
                                    false,
                                    theme,
                                )
                            },
                        );
                        apply_paired_scalar_scrub(
                            session,
                            &obj.id,
                            "text3d_bevel_size",
                            ("bevelThickness", bevel_vec[0]),
                            ("bevelSize", bevel_vec[1]),
                            bevel_ev,
                            state,
                        );

                        if session.plugins.is_enabled("text-plus") {
                            show_session_option_prop_ex(
                                ui,
                                session,
                                &obj.id,
                                "bevelSide",
                                "Side",
                                &["Both", "Front", "Back"],
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
                                &obj.id,
                                "bevelDetail",
                                "Detail",
                                3.0,
                                ScrubConfig::new(1.0, 32.0, 1.0, 0),
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
                                &obj.id,
                                "bevelProfile",
                                "Profile",
                                &["Linear", "Convex", "Concave", "Step"],
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
                                &obj.id,
                                "bevelTension",
                                "Tension",
                                0.5,
                                ScrubConfig::new(0.0, 1.0, 0.05, 2),
                                local_frame,
                                clip_start,
                                viewport_rect,
                                state,
                                theme,
                                actions,
                            );
                        }
                    }

                    // Center point (positionMode)
                    show_session_option_prop_ex(
                        ui,
                        session,
                        &obj.id,
                        "positionMode",
                        "Center point",
                        &["center", "left", "right", "top", "bottom", "front", "back"],
                        0,
                        false,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );

                    if session.plugins.is_enabled("text-plus") {
                        show_text3d_plus_controls(
                            ui,
                            session,
                            &clip.id,
                            &obj.id,
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                } else if is_particles {
                    show_particles_3d_inspector(
                        ui,
                        session,
                        &obj.id,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                } else {
                    // Shape / Geometry+ / Repeater & Effector+ / Text+ / ExpressText+ / Flares / Particle Suite
                    show_extended_3d_object_props(
                        ui,
                        session,
                        &obj,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }

                // Constraints for non-camera/non-light objects
                if session.plugins.is_enabled("constraints") {
                    show_constraint_target_rows(
                        ui,
                        session,
                        &obj.id,
                        &constraint_targets,
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                }

                // Material section for all mesh/geometry objects (Text3D, Shapes, Models)
                let is_mesh_object = !is_light
                    && !is_camera
                    && !is_particles
                    && !is_container_object(&obj.kind)
                    && !obj.kind.contains("optical-flares")
                    && !obj.kind.contains("trapcode-suite");
                if is_mesh_object {
                    state.inspector_root_indent = false;
                    show_material_slot_inspector(
                        ui,
                        session,
                        &clip.id,
                        &obj.id,
                        &obj.name,
                        "material",
                        "Material",
                        local_frame,
                        clip_start,
                        viewport_rect,
                        state,
                        theme,
                        actions,
                    );
                    if is_text && session.plugins.is_enabled("text-plus") {
                        show_material_slot_inspector(
                            ui,
                            session,
                            &clip.id,
                            &obj.id,
                            &format!("{} (Front)", obj.name),
                            "frontMaterial",
                            "Front Material",
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                        show_material_slot_inspector(
                            ui,
                            session,
                            &clip.id,
                            &obj.id,
                            &format!("{} (Side)", obj.name),
                            "sideMaterial",
                            "Side Material",
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

            // Render any extra/custom properties on the 3D object (such as imported legacy
            // expression custom properties) so they remain visible and editable.
            show_object_custom_properties(
                ui,
                session,
                &obj,
                local_frame,
                clip_start,
                viewport_rect,
                state,
                theme,
                actions,
            );
        });
}

#[allow(clippy::too_many_arguments)]
fn show_constraint_target_rows(
    ui: &mut Ui,
    session: &mut Session,
    obj_id: &str,
    targets: &[(String, String)],
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let mut labels: Vec<String> = Vec::with_capacity(targets.len() + 1);
    labels.push("None".to_owned());
    for (_, name) in targets {
        labels.push(name.clone());
    }
    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();

    // Track to target dropdown
    let track_enabled = eval_prop_u32(
        lookup_session_prop_ref(session, obj_id, "constraint.trackTo.enabled"),
        local_frame,
        0,
    ) != 0;
    let track_target = eval_prop_str(
        lookup_session_prop_ref(session, obj_id, "constraint.trackTo.targetId"),
        local_frame,
        "",
    );
    let mut track_idx = if !track_enabled || track_target.is_empty() {
        0
    } else {
        targets
            .iter()
            .position(|(id, _)| id == &track_target)
            .map_or(0, |i| i + 1)
    };
    let track_changed = panel_property_row(
        ui,
        "Track to",
        true,
        state,
        theme,
        |_| {},
        |ui| {
            cm3_dropdown(
                ui,
                Id::unique((obj_id, "constraint_track_to_target")),
                &mut track_idx,
                &label_refs,
                false,
                viewport_rect,
                theme,
            )
        },
    );
    if track_changed {
        let started_tx = session.begin_transaction("set track to").is_ok();
        if track_idx == 0 {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.trackTo.enabled".to_owned(),
                value: json!(0),
            });
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.trackTo.targetId".to_owned(),
                value: json!(""),
            });
        } else {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.trackTo.enabled".to_owned(),
                value: json!(1),
            });
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.trackTo.targetId".to_owned(),
                value: json!(targets[track_idx - 1].0),
            });
        }
        if started_tx && session.history.is_in_transaction() {
            let _ = session.commit_transaction(true);
        }
    }
    if track_idx > 0 {
        show_session_option_prop_ex(
            ui,
            session,
            obj_id,
            "constraint.trackTo.trackAxis",
            "Track axis",
            &["+X", "+Y", "+Z", "-X", "-Y", "-Z"],
            2,
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
            "constraint.trackTo.upAxis",
            "Up axis",
            &["X", "Y", "Z"],
            1,
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
            "constraint.trackTo.influence",
            "Track weight %",
            100.0,
            ScrubConfig::new(0.0, 100.0, 1.0, 1),
            local_frame,
            clip_start,
            viewport_rect,
            state,
            theme,
            actions,
        );
    }

    // Follow target dropdown
    let follow_enabled = eval_prop_u32(
        lookup_session_prop_ref(session, obj_id, "constraint.follow.enabled"),
        local_frame,
        0,
    ) != 0;
    let follow_target = eval_prop_str(
        lookup_session_prop_ref(session, obj_id, "constraint.follow.targetId"),
        local_frame,
        "",
    );
    let mut follow_idx = if !follow_enabled || follow_target.is_empty() {
        0
    } else {
        targets
            .iter()
            .position(|(id, _)| id == &follow_target)
            .map_or(0, |i| i + 1)
    };
    let follow_changed = panel_property_row(
        ui,
        "Follow",
        true,
        state,
        theme,
        |_| {},
        |ui| {
            cm3_dropdown(
                ui,
                Id::unique((obj_id, "constraint_follow_target")),
                &mut follow_idx,
                &label_refs,
                false,
                viewport_rect,
                theme,
            )
        },
    );
    if follow_changed {
        let started_tx = session.begin_transaction("set follow").is_ok();
        if follow_idx == 0 {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.follow.enabled".to_owned(),
                value: json!(0),
            });
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.follow.targetId".to_owned(),
                value: json!(""),
            });
        } else {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.follow.enabled".to_owned(),
                value: json!(1),
            });
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "constraint.follow.targetId".to_owned(),
                value: json!(targets[follow_idx - 1].0),
            });
        }
        if started_tx && session.history.is_in_transaction() {
            let _ = session.commit_transaction(true);
        }
    }
    if follow_idx > 0 {
        show_session_option_prop_ex(
            ui,
            session,
            obj_id,
            "constraint.follow.space",
            "Follow space",
            &["World", "Local"],
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
            "constraint.follow.followPosition",
            "Follow pos",
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
            "constraint.follow.followRotation",
            "Follow rot",
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
            "constraint.follow.followScale",
            "Follow scale",
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
        show_session_vec3_prop(
            ui,
            session,
            obj_id,
            "constraint.follow.positionOffset",
            "Pos offset",
            [0.0, 0.0, 0.0],
            ScrubConfig::new(-100_000.0, 100_000.0, 1.0, 2),
            false,
            local_frame,
            clip_start,
            viewport_rect,
            state,
            theme,
            actions,
        );
        show_session_vec3_prop(
            ui,
            session,
            obj_id,
            "constraint.follow.rotationOffset",
            "Rot offset",
            [0.0, 0.0, 0.0],
            ScrubConfig::new(-36_000.0, 36_000.0, 1.0, 1)
                .with_scale_factor(std::f64::consts::PI / 180.0),
            false,
            local_frame,
            clip_start,
            viewport_rect,
            state,
            theme,
            actions,
        );
        show_session_vec3_prop(
            ui,
            session,
            obj_id,
            "constraint.follow.scaleOffset",
            "Scale offset",
            [1.0, 1.0, 1.0],
            ScrubConfig::new(-1000.0, 1000.0, 0.05, 3),
            true,
            local_frame,
            clip_start,
            viewport_rect,
            state,
            theme,
            actions,
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Camera+ Optics, DOF Focus Linker, Motion Blur & 11-Key Vibrate
// ---------------------------------------------------------------------------

const CAMERA_FILM_GATE_PRESETS: &[(&str, f64)] = &[
    ("36mm Full Frame", 36.0),
    ("24.89mm Super 35", 24.89),
    ("22.3mm APS-C (Canon)", 22.3),
    ("23.6mm APS-C (Super)", 23.6),
    ("17.3mm Micro 4/3", 17.3),
    ("35mm Academy Cinema", 21.95),
    ("70mm Large Format", 70.0),
    ("12.52mm Super 16", 12.52),
    ("10.26mm 16mm", 10.26),
    ("13.2mm 1-inch", 13.2),
    ("8.8mm 2/3-inch", 8.8),
    ("6.4mm 1/2-inch", 6.4),
    ("4.8mm 1/3-inch", 4.8),
];

#[allow(clippy::too_many_arguments)]
fn show_camera_plus_controls(
    ui: &mut Ui,
    session: &mut Session,
    obj_id: &str,
    clip: &SessionClip,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    show_session_option_prop_ex(
        ui,
        session,
        obj_id,
        "active",
        "Active camera",
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
    show_session_number_prop(
        ui,
        session,
        obj_id,
        "zoom",
        "Zoom",
        1.0,
        ScrubConfig::new(0.05, 50.0, 0.05, 2),
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
        "focalLength",
        "Focal length (mm)",
        35.0,
        ScrubConfig::new(4.0, 600.0, 0.5, 1),
        local_frame,
        clip_start,
        viewport_rect,
        state,
        theme,
        actions,
    );

    // Bidirectional Horizontal & Vertical FOV synced with focalLength + filmGate + aspect ratio.
    {
        let aspect = (f64::from(session.document().resolution[0].max(1))
            / f64::from(session.document().resolution[1].max(1)))
        .max(0.1);
        let fl = eval_prop_f64(
            lookup_session_prop_ref(session, obj_id, "focalLength"),
            local_frame,
            35.0,
        )
        .max(1.0);
        let gate_w = eval_prop_f64(
            lookup_session_prop_ref(session, obj_id, "filmGate"),
            local_frame,
            36.0,
        )
        .max(1.0);
        let gate_h = gate_w / aspect;
        let mut fov_h = (2.0 * (gate_w / (2.0 * fl)).atan()).to_degrees();
        let mut fov_v = (2.0 * (gate_h / (2.0 * fl)).atan()).to_degrees();
        let mut new_fl: Option<f64> = None;
        let mut new_fov_v: Option<f64> = None;
        panel_property_row(
            ui,
            "FOV (H °)",
            true,
            state,
            theme,
            |_| {},
            |ui| {
                let r = ui.add(
                    egui::DragValue::new(&mut fov_h)
                        .range(1.0..=175.0)
                        .speed(0.5)
                        .fixed_decimals(1),
                );
                if r.changed() {
                    let rad = fov_h.clamp(1.0, 175.0).to_radians();
                    let fl_next = (gate_w / (2.0 * (rad * 0.5).tan())).clamp(4.0, 600.0);
                    let fv_next = (2.0 * (gate_h / (2.0 * fl_next)).atan()).to_degrees();
                    new_fl = Some(fl_next);
                    new_fov_v = Some(fv_next);
                }
            },
        );
        panel_property_row(
            ui,
            "FOV (V °)",
            true,
            state,
            theme,
            |_| {},
            |ui| {
                let r = ui.add(
                    egui::DragValue::new(&mut fov_v)
                        .range(1.0..=175.0)
                        .speed(0.5)
                        .fixed_decimals(1),
                );
                if r.changed() {
                    let rad = fov_v.clamp(1.0, 175.0).to_radians();
                    let fl_next = (gate_h / (2.0 * (rad * 0.5).tan())).clamp(4.0, 600.0);
                    new_fl = Some(fl_next);
                    new_fov_v = Some(fov_v.clamp(1.0, 175.0));
                }
            },
        );
        if let (Some(fl_next), Some(fv_next)) = (new_fl, new_fov_v) {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "focalLength".to_owned(),
                value: json!(fl_next),
            });
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "fov".to_owned(),
                value: json!(fv_next),
            });
        }
    }

    // 12 Film Gate presets dropdown + numeric scrub
    let cur_gate = eval_prop_f64(
        lookup_session_prop_ref(session, obj_id, "filmGate"),
        local_frame,
        36.0,
    );
    let gate_labels: Vec<&str> = CAMERA_FILM_GATE_PRESETS.iter().map(|(l, _)| *l).collect();
    let mut gate_idx = CAMERA_FILM_GATE_PRESETS
        .iter()
        .enumerate()
        .min_by(|a, b| {
            (a.1.1 - cur_gate)
                .abs()
                .total_cmp(&(b.1.1 - cur_gate).abs())
        })
        .map_or(0, |(i, _)| i);
    let gate_changed = panel_property_row(
        ui,
        "Gate preset",
        true,
        state,
        theme,
        |_| {},
        |ui| {
            cm3_dropdown(
                ui,
                Id::unique((obj_id, "cam_film_gate_preset")),
                &mut gate_idx,
                &gate_labels,
                false,
                viewport_rect,
                theme,
            )
        },
    );
    if gate_changed {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: obj_id.to_owned(),
            property: "filmGate".to_owned(),
            value: json!(CAMERA_FILM_GATE_PRESETS[gate_idx].1),
        });
    }
    show_session_number_prop(
        ui,
        session,
        obj_id,
        "filmGate",
        "Film gate",
        36.0,
        ScrubConfig::new(2.0, 120.0, 0.1, 2),
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
        "filmOffsetX",
        "Offset X",
        0.0,
        ScrubConfig::new(-100.0, 100.0, 0.1, 2),
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
        "filmOffsetY",
        "Offset Y",
        0.0,
        ScrubConfig::new(-100.0, 100.0, 0.1, 2),
        local_frame,
        clip_start,
        viewport_rect,
        state,
        theme,
        actions,
    );

    // Per-camera Motion Blur
    show_session_option_prop_ex(
        ui,
        session,
        obj_id,
        "motionBlur.enabled",
        "Camera MB",
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
        "motionBlur.samples",
        "MB samples",
        8.0,
        ScrubConfig::new(1.0, 64.0, 1.0, 0),
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
        "motionBlur.shutter",
        "MB shutter",
        0.5,
        ScrubConfig::new(0.0, 2.0, 0.05, 2),
        local_frame,
        clip_start,
        viewport_rect,
        state,
        theme,
        actions,
    );

    // Depth of Field + Focus Distance Linker
    show_session_option_prop_ex(
        ui,
        session,
        obj_id,
        "dofEnabled",
        "Depth of field",
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
        "dofFocusDistance",
        "Focus dist",
        80.0,
        ScrubConfig::new(0.1, 10_000.0, 1.0, 1),
        local_frame,
        clip_start,
        viewport_rect,
        state,
        theme,
        actions,
    );

    // Focus Distance Linker: live `focusDistanceTo("<target>")` expression link + Unlink bake
    let focus_targets: Vec<(String, String, [f64; 3])> = clip
        .objects_3d
        .iter()
        .filter(|o| o.id != obj_id)
        .map(|o| {
            let p = eval_prop_vec3(o.properties.get("position"), local_frame, [0.0, 0.0, 0.0]);
            (o.id.clone(), o.name.clone(), p)
        })
        .collect();
    let cur_focus_expr = lookup_session_prop_ref(session, obj_id, "dofFocusDistance")
        .and_then(|p| p.expression.clone());
    let linked_target_name = cur_focus_expr
        .as_deref()
        .and_then(parse_focus_linker_target);

    if !focus_targets.is_empty() {
        let mut labels: Vec<String> = vec!["None (Unlinked)".to_owned()];
        let mut pick_idx = 0usize;
        for (idx, (_, n, _)) in focus_targets.iter().enumerate() {
            labels.push(n.clone());
            if linked_target_name.as_deref() == Some(n.as_str()) {
                pick_idx = idx + 1;
            }
        }
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let picked = panel_property_row(
            ui,
            "Focus linker",
            true,
            state,
            theme,
            |_| {},
            |ui| {
                cm3_dropdown(
                    ui,
                    Id::unique((obj_id, "dof_focus_linker")),
                    &mut pick_idx,
                    &label_refs,
                    false,
                    viewport_rect,
                    theme,
                )
            },
        );
        if picked {
            let cam_pos = eval_prop_vec3(
                lookup_session_prop_ref(session, obj_id, "position"),
                local_frame,
                [0.0, 0.0, 80.0],
            );
            if pick_idx > 0 {
                let (_, tgt_name, tgt_pos) = &focus_targets[pick_idx - 1];
                let dx = cam_pos[0] - tgt_pos[0];
                let dy = cam_pos[1] - tgt_pos[1];
                let dz = cam_pos[2] - tgt_pos[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                let _ = link_focus_distance(session, obj_id, tgt_name, dist);
            } else if linked_target_name.is_some() {
                let base_dist = eval_prop_f64(
                    lookup_session_prop_ref(session, obj_id, "dofFocusDistance"),
                    local_frame,
                    80.0,
                );
                let baked = cur_focus_expr
                    .as_deref()
                    .and_then(|expr| {
                        eval_session_expr_f64(
                            session,
                            state,
                            obj_id,
                            "dofFocusDistance",
                            expr,
                            base_dist,
                        )
                    })
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .unwrap_or(base_dist);
                let _ = unlink_focus_distance(session, obj_id, local_frame, baked);
            }
        }
    }

    if let Some(ref linked_name) = linked_target_name {
        let mut do_unlink = false;
        panel_property_row(
            ui,
            "Focus link",
            true,
            state,
            theme,
            |_| {},
            |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(format!("Live: {linked_name}"))
                                .font(theme.font(FONT_COMPACT))
                                .color(theme.value()),
                        )
                        .truncate(),
                    );
                    if cm3_prop_button(ui, "Unlink", ButtonVariant::Normal, false, theme)
                        .on_hover_text(
                            "Clear live focusDistanceTo() expression and bake current distance",
                        )
                        .clicked()
                    {
                        do_unlink = true;
                    }
                });
            },
        );
        if do_unlink {
            let base_dist = eval_prop_f64(
                lookup_session_prop_ref(session, obj_id, "dofFocusDistance"),
                local_frame,
                80.0,
            );
            let euclidean_fallback =
                focus_targets
                    .iter()
                    .find(|(_, n, _)| n == linked_name)
                    .map(|(_, _, tgt_pos)| {
                        let cam_pos = eval_prop_vec3(
                            lookup_session_prop_ref(session, obj_id, "position"),
                            local_frame,
                            [0.0, 0.0, 80.0],
                        );
                        let dx = cam_pos[0] - tgt_pos[0];
                        let dy = cam_pos[1] - tgt_pos[1];
                        let dz = cam_pos[2] - tgt_pos[2];
                        (dx * dx + dy * dy + dz * dz).sqrt().max(0.1)
                    });
            let baked = cur_focus_expr
                .as_deref()
                .and_then(|expr| {
                    eval_session_expr_f64(
                        session,
                        state,
                        obj_id,
                        "dofFocusDistance",
                        expr,
                        base_dist,
                    )
                })
                .filter(|v| v.is_finite() && *v > 0.0)
                .or(euclidean_fallback)
                .unwrap_or(base_dist);
            let _ = unlink_focus_distance(session, obj_id, local_frame, baked);
        }
    }

    show_session_number_prop(
        ui,
        session,
        obj_id,
        "dofAperture",
        "Aperture",
        2.8,
        ScrubConfig::new(0.0, 40.0, 0.1, 2),
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
        "dofFocusAreaWidth",
        "Focus area",
        10.0,
        ScrubConfig::new(0.0, 2000.0, 0.5, 1),
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
        "dofNearBlur",
        "Near blur",
        100.0,
        ScrubConfig::new(0.0, 400.0, 1.0, 1),
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
        "dofFarBlur",
        "Far blur",
        100.0,
        ScrubConfig::new(0.0, 400.0, 1.0, 1),
        local_frame,
        clip_start,
        viewport_rect,
        state,
        theme,
        actions,
    );

    // Collapsible Camera Vibrate (all 11 vibrate.* keys)
    state.inspector_root_indent = false;
    cm3_collapsible_section(
        ui,
        Id::unique((obj_id, "camera_vibrate_section")),
        "Camera Vibrate",
        false,
        theme,
        |ui| {
            show_session_option_prop_ex(
                ui,
                session,
                obj_id,
                "vibrate.enabled",
                "Enabled",
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
                "vibrate.seed",
                "Seed",
                1.0,
                ScrubConfig::new(0.0, 100_000.0, 1.0, 0),
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
                "vibrate.enablePosition",
                "Enable position",
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
            show_session_vec3_prop(
                ui,
                session,
                obj_id,
                "vibrate.positionAmplitude",
                "Pos amplitude",
                [1.0, 1.0, 1.0],
                ScrubConfig::new(0.0, 1000.0, 0.1, 2),
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
                "vibrate.enableRotation",
                "Enable rotation",
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
            show_session_vec3_prop(
                ui,
                session,
                obj_id,
                "vibrate.rotationAmplitude",
                "Rot amplitude",
                [2.0, 2.0, 2.0],
                ScrubConfig::new(0.0, 360.0, 0.1, 2),
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
                "vibrate.frequency",
                "Master frequency",
                2.0,
                ScrubConfig::new(0.0, 60.0, 0.1, 2),
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
                "vibrate.positionFrequency",
                "Pos frequency",
                2.0,
                ScrubConfig::new(0.0, 60.0, 0.1, 2),
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
                "vibrate.rotationFrequency",
                "Rot frequency",
                2.0,
                ScrubConfig::new(0.0, 60.0, 0.1, 2),
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
                "vibrate.regularPulse",
                "Regular pulse",
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
                "vibrate.relative",
                "Relative",
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
        },
    );
}

// ---------------------------------------------------------------------------
// 6. Text3D Text+ 9 Properties & Split Into Letters
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn show_text3d_plus_controls(
    ui: &mut Ui,
    session: &mut Session,
    clip_id: &str,
    obj_id: &str,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    // Non-bevel Text+ controls (Bevel sub-controls are grouped consecutively right under `Bevel`)
    show_session_number_prop(
        ui,
        session,
        obj_id,
        "spacing",
        "Letter spacing",
        0.0,
        ScrubConfig::new(-200.0, 200.0, 0.25, 2),
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
        "kerning",
        "Kerning",
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
        "textScaleMode",
        "Scale mode",
        &["Current", "Legacy"],
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
        "characterIndexOffset",
        "Char offset",
        0.0,
        ScrubConfig::new(-1000.0, 1000.0, 1.0, 0),
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
        "characterCountOverride",
        "Char count",
        0.0,
        ScrubConfig::new(0.0, 1000.0, 1.0, 0),
        local_frame,
        clip_start,
        viewport_rect,
        state,
        theme,
        actions,
    );

    panel_property_row(
        ui,
        "Split letters",
        true,
        state,
        theme,
        |_| {},
        |ui| {
            if cm3_prop_button(ui, "Split Letters", ButtonVariant::Normal, true, theme).clicked()
                && let (Ok(cid), Ok(oid)) = (Uuid::parse_str(clip_id), Uuid::parse_str(obj_id))
            {
                let _ = session.execute(Command::SplitText3DIntoLetters {
                    clip_id: cid,
                    object_id: oid,
                });
            }
        },
    );
}

// ---------------------------------------------------------------------------
// 7. Custom Properties (Imported Legacy Expression & Dynamic Properties)
// ---------------------------------------------------------------------------

/// Returns `true` when `key` is already handled by the standard inspector sections for `obj_kind`.
pub(super) fn is_standard_3d_object_property(obj_kind: &str, key: &str) -> bool {
    if matches!(
        key,
        "name" | "enabled" | "position" | "rotation" | "scale" | "eulerOrder"
    ) || key.starts_with("material.")
        || key.starts_with("frontMaterial.")
        || key.starts_with("sideMaterial.")
        || key.starts_with("constraint.")
    {
        return true;
    }

    if obj_kind.contains("camera") {
        return key.starts_with("camera.")
            || key.starts_with("shake.")
            || key.starts_with("motionBlur.")
            || key.starts_with("vibrate.")
            || matches!(
                key,
                "shakeAmplitude"
                    | "shakeSpeed"
                    | "fov"
                    | "near"
                    | "far"
                    | "active"
                    | "zoom"
                    | "focalLength"
                    | "filmGate"
                    | "filmOffsetX"
                    | "filmOffsetY"
                    | "dofEnabled"
                    | "dofFocusDistance"
                    | "dofAperture"
                    | "dofFocusAreaWidth"
                    | "dofNearBlur"
                    | "dofFarBlur"
            );
    }

    if obj_kind.contains("light") {
        return key.starts_with("light.")
            || matches!(
                key,
                "target"
                    | "color"
                    | "groundColor"
                    | "intensity"
                    | "angle"
                    | "penumbra"
                    | "distance"
                    | "decay"
                    | "width"
                    | "height"
                    | "iesProfile"
                    | "sunElevation"
                    | "sunAzimuth"
                    | "temperature"
                    | "castShadow"
                    | "shadowMapSize"
                    | "shadowBias"
            );
    }

    if obj_kind == "text" || obj_kind == "text_3d" {
        return matches!(
            key,
            "text"
                | "font"
                | "height"
                | "thickness"
                | "detail"
                | "bevelEnabled"
                | "bevelThickness"
                | "bevelSize"
                | "bevelSide"
                | "bevelDetail"
                | "bevelProfile"
                | "bevelTension"
                | "positionMode"
                | "spacing"
                | "kerning"
                | "textScaleMode"
                | "characterIndexOffset"
                | "characterCountOverride"
        );
    }

    if obj_kind == "particles" {
        return key.starts_with("particles.");
    }

    let kind_norm = obj_kind.replace('_', "-");
    match kind_norm.as_str() {
        "box" => return key == "box_size",
        "cylinder" => {
            return matches!(
                key,
                "cylinder_size" | "cylinder_detail" | "cylinder_openEnded"
            );
        }
        "rectangle" => return key == "rect_size",
        "circle" => return matches!(key, "circle_size" | "circle_detail"),
        "sphere" => return matches!(key, "sphere_size" | "sphere_detail"),
        "donut" => return matches!(key, "donut_size" | "donut_detail"),
        "wire" => {
            return matches!(
                key,
                "path_size" | "path_thickness" | "path_closed" | "path_detail"
            );
        }
        "model" => return matches!(key, "customGeometry" | "subdivision"),
        "group" => return key == "reflectionVisibility",
        _ => {}
    }

    if kind_norm.contains("geometry-plus") {
        return key.starts_with("roundedBox.")
            || key.starts_with("polyhedron.")
            || key.starts_with("cone.")
            || key.starts_with("capsule.")
            || key.starts_with("tube.")
            || key.starts_with("gear.")
            || key.starts_with("helix.")
            || key.starts_with("spline.")
            || key.starts_with("torusKnot.")
            || key.starts_with("lathe.")
            || key.starts_with("superquadric.");
    }

    if kind_norm.contains("optical-flares") {
        return matches!(
            key,
            "brightness"
                | "anamorphic"
                | "flareColor"
                | "ghosts"
                | "starburstRays"
                | "ringIntensity"
                | "occlusion"
                | "Enable"
                | "Strength"
                | "Delay"
        );
    }

    if kind_norm.contains("trapcode-suite") {
        return matches!(
            key,
            "rate"
                | "lifetime"
                | "velocity"
                | "gravity"
                | "emitterSize"
                | "particleSize"
                | "particleColor"
                | "particlesX"
                | "particlesY"
                | "particlesZ"
                | "sizeX"
                | "sizeY"
                | "sizeZ"
                | "displace"
                | "count"
                | "maxDistance"
                | "lineColor"
                | "beamThickness"
                | "size"
                | "radius"
                | "seed"
                | "randomness"
                | "opacity"
                | "colors"
        );
    }

    let is_field_key = matches!(
        key,
        "field"
            | "fieldPosition"
            | "fieldRotation"
            | "fieldScale"
            | "fieldFalloff"
            | "fieldCurve"
            | "fieldInvert"
            | "fieldSweep"
            | "fieldNoiseScale"
            | "fieldNoiseEvolution"
            | "fieldNoiseSpeed"
            | "fieldEvolution"
            | "fieldSeed"
    );

    if kind_norm.contains("repeater") {
        return is_field_key
            || matches!(
                key,
                "count"
                    | "countX"
                    | "countY"
                    | "countZ"
                    | "positionOffset"
                    | "rotationOffset"
                    | "scaleOffset"
                    | "positionStep"
                    | "rotationStep"
                    | "scaleStep"
                    | "positionMin"
                    | "positionMax"
                    | "rotationMin"
                    | "rotationMax"
                    | "scaleMin"
                    | "scaleMax"
                    | "positionEnd"
                    | "rotationEnd"
                    | "scaleEnd"
                    | "firstPosition"
                    | "firstRotation"
                    | "firstScale"
                    | "lastPosition"
                    | "lastRotation"
                    | "lastScale"
                    | "radius"
                    | "startAngle"
                    | "endAngle"
                    | "alignToCircle"
                    | "axis"
                    | "angle"
                    | "amount"
                    | "mode"
                    | "direction"
                    | "distribution"
                    | "distance"
                    | "delay"
                    | "window"
                    | "offset"
                    | "spacing"
                    | "seed"
                    | "cells"
                    | "cellScale"
                    | "scatter"
                    | "spin"
                    | "strength"
                    | "springCycles"
                    | "springDamping"
                    | "pivot"
                    | "uniformScale"
                    | "closed"
                    | "fractureMode"
                    | "Delay"
                    | "Enable"
                    | "Strength"
            );
    }

    if kind_norm.contains("text-plus") {
        return is_field_key
            || matches!(
                key,
                "firstPosition"
                    | "firstRotation"
                    | "firstScale"
                    | "lastPosition"
                    | "lastRotation"
                    | "lastScale"
                    | "positionOffset"
                    | "rotationOffset"
                    | "scaleOffset"
                    | "positionMin"
                    | "positionMax"
                    | "rotationMin"
                    | "rotationMax"
                    | "scaleMin"
                    | "scaleMax"
                    | "seed"
                    | "randomness"
                    | "scatter"
                    | "spin"
                    | "strength"
                    | "amount"
                    | "mode"
                    | "direction"
                    | "delay"
                    | "offset"
                    | "characters"
                    | "characterNumbers"
                    | "invertCharacters"
                    | "charPosition"
                    | "charRotation"
                    | "charScale"
                    | "shakeAmplitude"
                    | "shakeFrequency"
                    | "shakePhase"
                    | "shakeSeed"
                    | "shakeMode"
                    | "shakeAxis"
                    | "shakePosition"
                    | "shakeRotation"
                    | "shakeScale"
                    | "shakePeriod"
                    | "shakeOffset"
                    | "shakeDelayPerCharacter"
            );
    }

    if kind_norm.contains("expresstext") {
        return is_field_key
            || matches!(
                key,
                "lettersMode"
                    | "delay"
                    | "delayMode"
                    | "delayOrder"
                    | "delayPerCharacter"
                    | "delayPosition"
                    | "delayRotation"
                    | "delayScale"
                    | "plainInvertOdd"
                    | "plainPosition"
                    | "plainRotation"
                    | "plainScale"
                    | "randomAnimationSpeed"
                    | "randomMode"
                    | "randomNoiseScale"
                    | "randomPositionMax"
                    | "randomPositionMin"
                    | "randomRotationMax"
                    | "randomRotationMin"
                    | "randomSeed"
                    | "formulaExpression"
                    | "formulaFrequency"
                    | "formulaPosition"
                    | "formulaRotation"
                    | "formulaScale"
                    | "formulaTime"
                    | "splineHeight"
                    | "splineLength"
                    | "splineOffset"
                    | "splinePath"
                    | "splineRadius"
                    | "splineTurns"
                    | "orbitAngle"
                    | "orbitRadius"
                    | "repeaterCopies"
                    | "repeaterFade"
                    | "repeaterLag"
                    | "repeaterMode"
                    | "repeaterOffset"
                    | "repeaterRotation"
                    | "repeaterScale"
                    | "twistAngle"
                    | "twistAxis"
                    | "wrapAmplitude"
                    | "wrapDirection"
                    | "wrapFrequency"
                    | "wrapPhase"
                    | "voronoiChunks"
                    | "voronoiDistance"
                    | "voronoiGravity"
                    | "voronoiScatter"
                    | "voronoiSeed"
                    | "voronoiSpin"
                    | "fragmentDirection"
                    | "fragmentRotation"
                    | "fragmentScale"
                    | "shakeAmplitude"
                    | "shakeFrequency"
                    | "shakePhase"
                    | "shakeSeed"
                    | "shakeMode"
                    | "shakeAxis"
                    | "shakePosition"
                    | "shakeRotation"
                    | "shakeScale"
                    | "shakePeriod"
                    | "shakeOffset"
                    | "shakeDelayPerCharacter"
            );
    }

    false
}

#[allow(clippy::too_many_arguments)]
fn show_object_custom_properties(
    ui: &mut Ui,
    session: &mut Session,
    obj: &SessionObject3D,
    local_frame: f64,
    clip_start: f64,
    viewport_rect: Rect,
    state: &mut PanelsState,
    theme: &Cm3Theme,
    actions: &mut Vec<PanelAction>,
) {
    let custom_props: Vec<(
        String,
        Value,
        Option<zc_session::SessionValueKind>,
        Option<String>,
    )> = obj
        .properties
        .iter()
        .filter(|(k, _)| !is_standard_3d_object_property(&obj.kind, k))
        .filter_map(|(k, prop)| {
            let val = prop
                .keyframes
                .first()
                .map_or_else(|| prop.default_value.clone(), |kf| kf.value.clone());
            if val.is_null() && prop.expression.is_none() {
                None
            } else {
                let value_kind = prop.keyframes.first().and_then(|kf| kf.value_kind);
                Some((k.clone(), val, value_kind, prop.expression.clone()))
            }
        })
        .collect();

    if custom_props.is_empty() {
        return;
    }

    state.inspector_root_indent = false;
    cm3_collapsible_section(
        ui,
        Id::unique((&obj.id, "obj_custom_properties")),
        "Custom Properties",
        true,
        theme,
        |ui| {
            for (key, val, value_kind, expr) in custom_props {
                let lower = key.to_ascii_lowercase();
                let is_color = matches!(value_kind, Some(zc_session::SessionValueKind::Color))
                    || lower.contains("color");

                match &val {
                    Value::Null => {
                        show_session_number_prop(
                            ui,
                            session,
                            &obj.id,
                            &key,
                            &key,
                            0.0,
                            ScrubConfig::new(-100_000.0, 100_000.0, 0.1, 3),
                            local_frame,
                            clip_start,
                            viewport_rect,
                            state,
                            theme,
                            actions,
                        );
                    }
                    Value::Bool(b) => {
                        show_session_option_prop(
                            ui,
                            session,
                            &obj.id,
                            &key,
                            &key,
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
                            &obj.id,
                            &key,
                            &key,
                            n.as_f64().unwrap_or(0.0),
                            ScrubConfig::new(-100_000.0, 100_000.0, 0.1, 3),
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
                            &obj.id,
                            &key,
                            &key,
                            def,
                            ScrubConfig::new(-100_000.0, 100_000.0, 0.1, 3),
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
                        if is_color {
                            show_session_color_prop(
                                ui,
                                session,
                                &obj.id,
                                &key,
                                &key,
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
                                &obj.id,
                                &key,
                                &key,
                                def,
                                ScrubConfig::new(-100_000.0, 100_000.0, 0.1, 3),
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
                        if is_color {
                            show_session_color_rgba_prop(
                                ui,
                                session,
                                &obj.id,
                                &key,
                                &key,
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
                                &obj.id,
                                &key,
                                &key,
                                def,
                                ScrubConfig::new(-100_000.0, 100_000.0, 0.1, 3),
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
                            &obj.id,
                            &key,
                            &key,
                            s,
                            local_frame,
                            state,
                            theme,
                        );
                        if let Some(expr_code) = expr {
                            let open_fx = panel_property_row(
                                ui,
                                "",
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
                                            "Edit expression",
                                            theme,
                                        ) {
                                            clicked = true;
                                        }
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(&expr_code)
                                                    .font(theme.font(FONT_COMPACT))
                                                    .color(TEXT_MUTED),
                                            )
                                            .truncate(),
                                        );
                                    });
                                    clicked
                                },
                            );
                            if open_fx {
                                open_expression_window(
                                    &obj.id,
                                    &key,
                                    expr_code,
                                    state,
                                    viewport_rect,
                                );
                            }
                        }
                    }
                    other => {
                        let summary = summarize_complex_json_value(other);
                        let mut open_fx = false;
                        panel_property_row(
                            ui,
                            &key,
                            true,
                            state,
                            theme,
                            |_| {},
                            |ui| {
                                ui.horizontal(|ui| {
                                    if expr.is_some()
                                        && cm3_icon_button(
                                            ui,
                                            Cm3Icon::Code,
                                            17.0,
                                            theme.value(),
                                            "Edit expression",
                                            theme,
                                        )
                                    {
                                        open_fx = true;
                                    }
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(summary)
                                                .font(theme.font(FONT_SMALL))
                                                .color(TEXT_MUTED),
                                        )
                                        .truncate(),
                                    );
                                });
                            },
                        );
                        if open_fx && let Some(expr_code) = expr {
                            open_expression_window(&obj.id, &key, expr_code, state, viewport_rect);
                        }
                    }
                }
            }
        },
    );
}

/// Formats a live `focusDistanceTo("<target_name>")` expression for Camera+ DoF linking.
pub(super) fn build_focus_linker_expression(target_name: &str) -> String {
    let quoted = serde_json::to_string(target_name)
        .unwrap_or_else(|_| format!("\"{}\"", target_name.replace('"', "\\\"")));
    format!("focusDistanceTo({quoted})")
}

/// Extracts the linked target name from a `focusDistanceTo("<name>")` expression, if present.
pub(super) fn parse_focus_linker_target(expr: &str) -> Option<String> {
    let trimmed = expr.trim();
    let inner = trimmed
        .strip_prefix("focusDistanceTo(")?
        .strip_suffix(')')?
        .trim();
    if let Ok(parsed) = serde_json::from_str::<String>(inner) {
        return (!parsed.is_empty()).then_some(parsed);
    }
    if let Some(single) = inner.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')) {
        let unescaped = single.replace("\\'", "'").replace("\\\\", "\\");
        return (!unescaped.is_empty()).then_some(unescaped);
    }
    None
}

/// Links a camera's `dofFocusDistance` to `target_name` via a live `focusDistanceTo()` expression
/// while storing the current distance as the fallback static value in a single undo transaction.
pub(super) fn link_focus_distance(
    session: &mut Session,
    obj_id: &str,
    target_name: &str,
    current_distance: f64,
) -> Result<(), zc_session::SessionError> {
    let rounded = if current_distance.is_finite() {
        (current_distance.max(0.1) * 10.0).round() / 10.0
    } else {
        80.0
    };
    let expr = build_focus_linker_expression(target_name);
    let started_tx = if !session.history.is_in_transaction() {
        session.begin_transaction("Link focus distance").is_ok()
    } else {
        false
    };
    let res = session
        .execute(Command::SetPropertyValue {
            target_id: obj_id.to_owned(),
            property: "dofFocusDistance".to_owned(),
            value: json!(rounded),
        })
        .and_then(|_| {
            session.execute(Command::SetExpression {
                target_id: obj_id.to_owned(),
                property: "dofFocusDistance".to_owned(),
                expression: Some(expr),
            })
        })
        .map(|_| ());
    if started_tx && session.history.is_in_transaction() {
        if res.is_ok() {
            let _ = session.commit_transaction(true);
        } else {
            let _ = session.cancel_transaction();
        }
    }
    res
}

/// Clears the live `focusDistanceTo()` expression on a camera's `dofFocusDistance` and bakes the
/// currently evaluated distance into the property in a single undo transaction.
pub(super) fn unlink_focus_distance(
    session: &mut Session,
    obj_id: &str,
    local_frame: f64,
    baked_distance: f64,
) -> Result<(), zc_session::SessionError> {
    let rounded = if baked_distance.is_finite() {
        (baked_distance.max(0.1) * 10.0).round() / 10.0
    } else {
        80.0
    };
    let is_multi_kf = lookup_session_prop_ref(session, obj_id, "dofFocusDistance")
        .is_some_and(|p| p.animated && p.keyframes.len() > 1);
    let started_tx = if !session.history.is_in_transaction() {
        session.begin_transaction("Unlink focus distance").is_ok()
    } else {
        false
    };
    let res = session
        .execute(Command::SetExpression {
            target_id: obj_id.to_owned(),
            property: "dofFocusDistance".to_owned(),
            expression: None,
        })
        .and_then(|_| {
            if is_multi_kf {
                session.execute(Command::SetKeyframe {
                    target_id: obj_id.to_owned(),
                    property: "dofFocusDistance".to_owned(),
                    frame: local_frame.round(),
                    value: json!(rounded),
                    tween: None,
                })
            } else {
                session.execute(Command::SetPropertyValue {
                    target_id: obj_id.to_owned(),
                    property: "dofFocusDistance".to_owned(),
                    value: json!(rounded),
                })
            }
        })
        .map(|_| ());
    if started_tx && session.history.is_in_transaction() {
        if res.is_ok() {
            let _ = session.commit_transaction(true);
        } else {
            let _ = session.cancel_transaction();
        }
    }
    res
}

#[cfg(test)]
mod custom_props_tests {
    use super::{
        build_focus_linker_expression, is_standard_3d_object_property, link_focus_distance,
        lookup_session_prop_ref, parse_focus_linker_target, unlink_focus_distance,
    };
    use serde_json::json;
    use zc_session::{Command, Session};

    #[test]
    fn custom_3d_properties_are_not_classified_as_standard() {
        assert!(is_standard_3d_object_property("box", "position"));
        assert!(is_standard_3d_object_property("box", "box_size"));
        assert!(is_standard_3d_object_property("box", "material.color"));
        assert!(is_standard_3d_object_property("camera", "fov"));
        assert!(is_standard_3d_object_property("spot_light", "intensity"));
        assert!(is_standard_3d_object_property("text", "bevelEnabled"));
        assert!(is_standard_3d_object_property(
            "particles",
            "particles.rate"
        ));
        assert!(is_standard_3d_object_property(
            "zoidium:geometry-plus/lathe",
            "lathe.segments"
        ));
        assert!(is_standard_3d_object_property(
            "zoidium:geometry-plus/superquadric",
            "superquadric.eastWest"
        ));
        assert!(is_standard_3d_object_property(
            "zoidium:repeater/voronoi-fracture",
            "fractureMode"
        ));
        assert!(is_standard_3d_object_property(
            "zoidium:repeater/voronoi-fracture",
            "fieldNoiseSpeed"
        ));

        for kind in ["box", "text", "camera", "spot_light", "group", "particles"] {
            assert!(!is_standard_3d_object_property(kind, "customSpeed"));
            assert!(!is_standard_3d_object_property(kind, "wiggleAmp"));
            assert!(!is_standard_3d_object_property(kind, "expr_offset"));
            assert!(!is_standard_3d_object_property(kind, "customColor"));
        }
    }

    #[test]
    fn focus_linker_expression_generation_parsing_and_unlink_roundtrip() {
        assert_eq!(
            build_focus_linker_expression("Hero Cube"),
            "focusDistanceTo(\"Hero Cube\")"
        );
        assert_eq!(
            parse_focus_linker_target("focusDistanceTo(\"Hero Cube\")"),
            Some("Hero Cube".to_owned())
        );
        assert_eq!(
            parse_focus_linker_target("  focusDistanceTo('Cube 2')  "),
            Some("Cube 2".to_owned())
        );
        assert_eq!(
            build_focus_linker_expression("Quote \"A\""),
            "focusDistanceTo(\"Quote \\\"A\\\"\")"
        );
        assert_eq!(
            parse_focus_linker_target(&build_focus_linker_expression("Quote \"A\"")),
            Some("Quote \"A\"".to_owned())
        );
        assert_eq!(parse_focus_linker_target("wiggle(2, 10)"), None);

        let mut session = Session::new();
        let track_id = session.document().video_tracks[0].id.clone();
        let scene_clip_id = session
            .execute(Command::AddClip {
                track_id,
                kind: "scene".to_owned(),
                name: Some("Focus Scene".to_owned()),
                start: 0.0,
                length: 120.0,
            })
            .expect("add scene clip")["clip_id"]
            .as_str()
            .expect("clip_id")
            .to_owned();
        let cam_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_clip_id.clone(),
                parent_id: None,
                kind: "camera".to_owned(),
                name: Some("Main Camera".to_owned()),
            })
            .expect("add camera")["object_id"]
            .as_str()
            .expect("camera id")
            .to_owned();

        link_focus_distance(&mut session, &cam_id, "Hero Cube", 142.34)
            .expect("link focus distance");
        let linked_prop = lookup_session_prop_ref(&session, &cam_id, "dofFocusDistance")
            .expect("dofFocusDistance prop");
        assert_eq!(
            linked_prop.expression.as_deref(),
            Some("focusDistanceTo(\"Hero Cube\")")
        );
        assert_eq!(
            linked_prop.keyframes.first().map(|k| &k.value),
            Some(&json!(142.3))
        );

        unlink_focus_distance(&mut session, &cam_id, 0.0, 215.67).expect("unlink focus distance");
        let unlinked_prop = lookup_session_prop_ref(&session, &cam_id, "dofFocusDistance")
            .expect("dofFocusDistance prop after unlink");
        assert_eq!(unlinked_prop.expression, None);
        assert_eq!(
            unlinked_prop.keyframes.first().map(|k| &k.value),
            Some(&json!(215.7))
        );
    }
}
