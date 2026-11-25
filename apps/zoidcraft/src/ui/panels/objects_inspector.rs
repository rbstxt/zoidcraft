use super::objects_extended::show_extended_3d_object_props;
use super::objects_materials::show_material_slot_inspector;
use super::objects_particles::show_particles_3d_inspector;
use super::objects_tab::is_container_object;
use super::property_rows::{
    apply_scrub_to_session, eval_prop_f64, eval_prop_str, eval_prop_u32, eval_prop_vec3,
    lookup_session_prop_ref, panel_property_row, show_session_color_prop, show_session_number_prop,
    show_session_option_prop, show_session_option_prop_ex, show_session_text_prop_ex,
    show_session_vec3_prop, show_session_vec4_prop_ex,
};
use super::shared_widgets::{
    apply_paired_scalar_scrub, empty_panel_message, property_header_with_actions,
    show_font_selector_row, show_session_asset_prop,
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
                return;
            }

            if is_light {
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
                return;
            }

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
            &["-Z", "+Z", "+Y", "-Y", "+X", "-X"],
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
            "constraint.trackTo.upAxis",
            "Up axis",
            &["+Y", "+Z", "+X", "-Y", "-Z", "-X"],
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
            "constraint.trackTo.influence",
            "Track weight %",
            1.0,
            ScrubConfig::new(0.0, 100.0, 1.0, 1).with_scale_factor(0.01),
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
            0,
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

    // Focus Distance Linker: compute Euclidean distance to any 3D object in the scene
    let focus_targets: Vec<(String, String, [f64; 3])> = clip
        .objects_3d
        .iter()
        .filter(|o| o.id != obj_id)
        .map(|o| {
            let p = eval_prop_vec3(o.properties.get("position"), local_frame, [0.0, 0.0, 0.0]);
            (o.id.clone(), o.name.clone(), p)
        })
        .collect();
    if !focus_targets.is_empty() {
        let mut labels: Vec<String> = vec!["Pick target to focus...".to_owned()];
        for (_, n, _) in &focus_targets {
            labels.push(n.clone());
        }
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let mut pick_idx = 0usize;
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
        if picked && pick_idx > 0 {
            let cam_pos = eval_prop_vec3(
                lookup_session_prop_ref(session, obj_id, "position"),
                local_frame,
                [0.0, 0.0, 80.0],
            );
            let tgt_pos = focus_targets[pick_idx - 1].2;
            let dx = cam_pos[0] - tgt_pos[0];
            let dy = cam_pos[1] - tgt_pos[1];
            let dz = cam_pos[2] - tgt_pos[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
            let _ = session.execute(Command::SetPropertyValue {
                target_id: obj_id.to_owned(),
                property: "dofFocusDistance".to_owned(),
                value: json!((dist * 10.0).round() / 10.0),
            });
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
                1,
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
        &["Height", "Width", "Uniform"],
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
