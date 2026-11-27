use super::edit_tab::default_layer_mask;
use super::effects_inspector::{
    MAX_INLINE_TEXT_CHARS, collect_shader_image_asset_choices, custom_shader_number_scrub_config,
    is_type5_bool_placeholder_value, oversized_text_summary,
    template_number_scrub_config_for_default, template_option_labels,
};
use super::effects_tab::{
    build_effect_picker_items, delete_effect_and_select_adjacent,
    find_effect_group_custom_properties, update_effect_group_custom_property,
};
use super::export_tab::container_label;
use super::media_tab::add_preset_clip;
use super::objects_2d::{
    add_shape_plus_item_smart, default_text_plus_2d_root, shape_plus_prop_sort_key,
    shape_plus_string_enum_options, text_plus_2d_slot_mut, tp2d_read_f64, tp2d_read_vec2,
    tp2d_write_f64, tp2d_write_vec2,
};
use super::objects_materials::{
    initialize_node_material_defaults, node_material_has_populated_graph,
};
use super::objects_tab::{apply_object_drop, build_object_picker_items, is_container_object};
use super::plugins_settings::{
    build_debug_info_report, humanize_history_label, rank_filtered_plugin_categories,
    show_settings_tab,
};
use super::property_rows::{
    EXPR_EDITOR_DEFAULT_SIZE, apply_color_picker_to_session, apply_kf_bar_event,
    apply_scrub_to_session, apply_scrub_to_session_with_owner_key, clear_property_expression,
    eval_prop_f64, eval_prop_str, eval_prop_vec2, eval_prop_vec3, find_clip_cached,
    find_clip_clone, insert_expression_snippet, lookup_session_prop, show_session_text_prop,
};
use super::sequence_tab::{
    collect_mirrored_sequence_control_keys, sanitize_control_identifier,
    sequence_header_label_ratio, sequence_motion_blur_labels,
    set_sequence_background_color_undoable,
};
use super::shared_widgets::{
    SHADER_PLUS_STARTER_TEMPLATES, apply_paired_scalar_scrub, compile_expr_cached,
    compute_asset_usage_counts_cached, compute_shader_editor_window_rect,
    discover_system_fonts_in_dirs, evaluate_expression_live_preview, format_shader_compile_error,
    is_supported_system_font_ext,
};
use super::state::{
    TreeDropMode, WidgetTxTracker, apply_widget_tx_command, owns_inspector_tx, paste_pz_json_text,
};
use super::*;
use zc_session::SessionKeyframe;

#[test]
fn eval_prop_value_uses_incoming_b_tween_semantics() {
    let prop = SessionProperty {
        animated: true,
        default_value: json!(1.0),
        expression: None,
        keyframes: vec![
            SessionKeyframe {
                frame: 0.0,
                value: json!(1.0),
                tween: 1, // Linear on k0
                control_points: [[-10.0, 0.0], [10.0, 0.0]],
                continuous_tangent: true,
                value_kind: None,
            },
            SessionKeyframe {
                frame: 20.0,
                value: json!(0.0),
                tween: 0, // Hold (Discrete) on k1 governs segment [0, 20]
                control_points: [[-10.0, 0.0], [10.0, 0.0]],
                continuous_tangent: true,
                value_kind: None,
            },
        ],
    };
    let v = eval_prop_f64(Some(&prop), 10.0, -1.0);
    assert!(
        (v - 1.0).abs() < 1e-6,
        "expected Hold value 1.0 governed by b.tween, got {v}"
    );
}

#[test]
fn apply_scrub_to_session_handles_cancel_atomic_commit_and_stale_tx() {
    let mut session = Session::new();
    let mut state = PanelsState::default();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();

    // 1. Atomic TextEdit commit (changed && committed, no prior tx) leaves no open tx
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "opacity",
        false,
        0.0,
        json!(0.5),
        ScrubEvent {
            started: false,
            changed: true,
            committed: true,
            cancelled: false,
        },
        &mut state,
    );
    assert!(!session.history.is_in_transaction());
    let prop = lookup_session_prop(&session, &clip_id, "opacity");
    assert!((eval_prop_f64(prop.as_ref(), 0.0, 1.0) - 0.5).abs() < 1e-6);

    // 2. Start + change + cancel rolls back cleanly with no leaked transaction
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "opacity",
        false,
        0.0,
        json!(0.2),
        ScrubEvent {
            started: true,
            changed: true,
            committed: false,
            cancelled: false,
        },
        &mut state,
    );
    assert!(session.history.is_in_transaction());
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "opacity",
        false,
        0.0,
        json!(0.2),
        ScrubEvent {
            started: false,
            changed: false,
            committed: false,
            cancelled: true,
        },
        &mut state,
    );
    assert!(!session.history.is_in_transaction());
    let prop = lookup_session_prop(&session, &clip_id, "opacity");
    assert!((eval_prop_f64(prop.as_ref(), 0.0, 1.0) - 0.5).abs() < 1e-6);

    // 3. Stale open transaction is closed cleanly when a new started event arrives
    session.begin_transaction("stale").unwrap();
    assert!(session.history.is_in_transaction());
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "opacity",
        false,
        0.0,
        json!(0.8),
        ScrubEvent {
            started: true,
            changed: true,
            committed: false,
            cancelled: false,
        },
        &mut state,
    );
    assert!(session.history.is_in_transaction());
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "opacity",
        false,
        0.0,
        json!(0.8),
        ScrubEvent {
            started: false,
            changed: false,
            committed: true,
            cancelled: false,
        },
        &mut state,
    );
    assert!(!session.history.is_in_transaction());
    let prop = lookup_session_prop(&session, &clip_id, "opacity");
    assert!((eval_prop_f64(prop.as_ref(), 0.0, 1.0) - 0.8).abs() < 1e-6);
}

#[test]
fn inspector_tx_ownership_guard_same_frame_vector_and_lazy_color_picker() {
    let mut session = Session::new();
    let mut state = PanelsState::default();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();

    // 1. Ownership guard: Widget A starts a tx; Widget B's cancel/commit must NOT touch it.
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "opacity",
        false,
        0.0,
        json!(0.4),
        ScrubEvent {
            started: true,
            changed: true,
            committed: false,
            cancelled: false,
        },
        &mut state,
    );
    assert!(session.history.is_in_transaction());
    assert!(owns_inspector_tx(&state, &clip_id, "opacity"));

    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "rotation",
        false,
        0.0,
        json!(10.0),
        ScrubEvent {
            started: false,
            changed: false,
            committed: false,
            cancelled: true,
        },
        &mut state,
    );
    assert!(session.history.is_in_transaction());
    assert!(owns_inspector_tx(&state, &clip_id, "opacity"));

    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "rotation",
        false,
        0.0,
        json!(10.0),
        ScrubEvent {
            started: false,
            changed: false,
            committed: true,
            cancelled: false,
        },
        &mut state,
    );
    assert!(session.history.is_in_transaction());
    assert!(owns_inspector_tx(&state, &clip_id, "opacity"));

    // Commit Widget A cleanly.
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "opacity",
        false,
        0.0,
        json!(0.4),
        ScrubEvent {
            started: false,
            changed: false,
            committed: true,
            cancelled: false,
        },
        &mut state,
    );
    assert!(!session.history.is_in_transaction());
    assert!(state.inspector_tx_owner.is_none());

    // 2. Same-frame started && changed && committed on multi-axis vector commits axis 0
    // atomically AND opens a fresh owned tx for axis 1.
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "position",
        false,
        0.0,
        json!([12.0, 0.0]),
        ScrubEvent {
            started: true,
            changed: true,
            committed: true,
            cancelled: false,
        },
        &mut state,
    );
    assert!(session.history.is_in_transaction());
    assert!(owns_inspector_tx(&state, &clip_id, "position"));
    // Cancelling axis 1's fresh drag must preserve axis 0's committed [12.0, 0.0]!
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "position",
        false,
        0.0,
        json!([12.0, 99.0]),
        ScrubEvent {
            started: false,
            changed: true,
            committed: false,
            cancelled: false,
        },
        &mut state,
    );
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "position",
        false,
        0.0,
        json!([12.0, 99.0]),
        ScrubEvent {
            started: false,
            changed: false,
            committed: false,
            cancelled: true,
        },
        &mut state,
    );
    assert!(!session.history.is_in_transaction());
    let pos = eval_prop_vec2(
        lookup_session_prop(&session, &clip_id, "position").as_ref(),
        0.0,
        [0.0, 0.0],
    );
    assert!((pos[0] - 12.0).abs() < 1e-6 && pos[1].abs() < 1e-6);

    // 3. Lazy color picker: opening unchanged popup does NOT hold an open tx, so clicking
    // another control + cancelling the unchanged picker never rolls back that control.
    apply_color_picker_to_session(
        &mut session,
        &clip_id,
        "backgroundColor",
        false,
        0.0,
        json!([0.0, 0.0, 0.0]),
        super::super::controls::ColorPickerEvent {
            started: true,
            changed: false,
            committed: false,
            cancelled: false,
            eyedropper_clicked: false,
        },
        &mut state,
    );
    assert!(!session.history.is_in_transaction());
    let _ = session.execute(Command::SetPropertyValue {
        target_id: clip_id.clone(),
        property: "opacity".to_owned(),
        value: json!(0.75),
    });
    apply_color_picker_to_session(
        &mut session,
        &clip_id,
        "backgroundColor",
        false,
        0.0,
        json!([0.0, 0.0, 0.0]),
        super::super::controls::ColorPickerEvent {
            started: false,
            changed: false,
            committed: false,
            cancelled: true,
            eyedropper_clicked: false,
        },
        &mut state,
    );
    let op = eval_prop_f64(
        lookup_session_prop(&session, &clip_id, "opacity").as_ref(),
        0.0,
        1.0,
    );
    assert!((op - 0.75).abs() < 1e-6);
}

#[test]
fn kf_bar_hold_to_bezier_toggle_and_epsilon_delete() {
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    session
        .execute(Command::TogglePropertyAnimated {
            target_id: clip_id.clone(),
            property: "opacity".to_owned(),
            frame: 0.0,
        })
        .unwrap();
    session
        .execute(Command::SetKeyframe {
            target_id: clip_id.clone(),
            property: "opacity".to_owned(),
            frame: 10.0,
            value: json!(0.25),
            tween: Some(0), // Hold
        })
        .unwrap();

    let prop = lookup_session_prop(&session, &clip_id, "opacity");
    let mut actions = Vec::new();
    // Query slightly offset (5e-5, within 1e-4 but > 1e-6) and toggle Hold -> Bezier
    apply_kf_bar_event(
        &mut session,
        &clip_id,
        "opacity",
        prop.as_ref(),
        10.00005,
        0.0,
        json!(0.25),
        KeyframeBarEvent {
            toggle_interp: true,
            ..Default::default()
        },
        &mut actions,
    );
    let prop_after = lookup_session_prop(&session, &clip_id, "opacity").unwrap();
    assert_eq!(prop_after.keyframes.len(), 2, "must not duplicate keyframe");
    let kf = prop_after
        .keyframes
        .iter()
        .find(|k| (k.frame - 10.0).abs() < 1e-6)
        .unwrap();
    let (mode, _) = InterpolationMode::from_cm3_tween_code(kf.tween);
    assert_eq!(mode, InterpolationMode::Bezier);

    // Delete using slightly offset local_frame must still delete exact kf.frame
    apply_kf_bar_event(
        &mut session,
        &clip_id,
        "opacity",
        Some(&prop_after),
        10.00005,
        0.0,
        json!(0.25),
        KeyframeBarEvent {
            toggle_kf: true,
            ..Default::default()
        },
        &mut actions,
    );
    let prop_deleted = lookup_session_prop(&session, &clip_id, "opacity").unwrap();
    assert_eq!(prop_deleted.keyframes.len(), 1);
}

#[test]
fn transform_wrap_has_all_three_modes_and_objects_picker_matches_zoidium() {
    assert_eq!(
        template_option_labels("transform", "wrap"),
        &["tile", "reflect"]
    );
    assert_eq!(
        template_option_labels("shake", "wrap"),
        &["clamp", "tile", "reflect"]
    );

    let session = Session::new();
    let items = build_object_picker_items(&session);
    let core: Vec<_> = items.iter().filter(|i| i.category == "Objects").collect();
    assert_eq!(core.len(), 10);
    assert_eq!(core[0].name, "Shape");
    assert_eq!(core[1].name, "Text");
    assert_eq!(core[2].name, "Light");
    assert!(
        items
            .iter()
            .all(|i| i.id != "zoidium:geometry-plus/superquadric")
    );
}

#[test]
fn wave4b_pickers_containers_and_live_expr_diagnostics() {
    let mut session = Session::new();
    for pid in [
        "geometry-plus",
        "repeater",
        "camera-plus",
        "text-plus",
        "expresstext-plus",
        "optical-flares",
        "trapcode-suite",
        "smart-motion-blur-lite",
        "layer-input",
    ] {
        let _ = session.execute(Command::EnablePlugin {
            plugin_id: pid.to_owned(),
        });
    }

    // Container check covers group, repeaters, text-plus, expresstext-plus
    assert!(is_container_object("group"));
    assert!(is_container_object("zoidium:repeater/linear-repeater"));
    assert!(is_container_object("zoidium:repeater/plain"));
    assert!(is_container_object("zoidium:text-plus/text-plus"));
    assert!(is_container_object("zoidium:expresstext-plus/expresstext"));

    // Object picker uses canonical hyphenated IDs and zero third-party brands
    let obj_items = build_object_picker_items(&session);
    for req_id in [
        "ambient_light",
        "zoidium:geometry-plus/rounded-box",
        "zoidium:geometry-plus/gear",
        "zoidium:geometry-plus/helix",
        "zoidium:geometry-plus/spline",
        "zoidium:repeater/plain",
        "zoidium:repeater/delay",
        "zoidium:repeater/voronoi-fracture",
        "zoidium:camera-plus/camera",
        "zoidium:text-plus/text-plus",
        "zoidium:expresstext-plus/expresstext",
        "zoidium:trapcode-suite/area-light",
    ] {
        assert!(
            obj_items.iter().any(|i| i.id == req_id),
            "missing object picker item `{req_id}`"
        );
    }
    let forbidden = [
        "trapcode",
        "magic bullet",
        "red giant",
        "maxon",
        "videocopilot",
    ];
    for item in &obj_items {
        let txt = format!("{} {} {}", item.category, item.name, item.desc).to_ascii_lowercase();
        for bad in forbidden {
            assert!(!txt.contains(bad), "forbidden brand `{bad}` in `{txt}`");
        }
    }

    // Effect picker has no duplicate smart-motion-blur-lite/zsmb or layer-input/layer_input
    let fx_items = build_effect_picker_items(&session);
    assert!(
        fx_items
            .iter()
            .all(|i| i.id != "zoidium:smart-motion-blur-lite/zsmb")
    );
    assert!(
        fx_items
            .iter()
            .all(|i| i.id != "zoidium:layer-input/layer_input")
    );

    // Live expression diagnostics
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let ok = evaluate_expression_live_preview(&session, &clip_id, "opacity", "1 + 2 * 3");
    assert_eq!(ok.as_deref(), Ok("7"));
    let err = evaluate_expression_live_preview(&session, &clip_id, "opacity", "1 + *");
    assert!(err.is_err());

    // All 4 Shader+ starter templates compile via translate_glsl_es100_with_options
    for (name, src) in SHADER_PLUS_STARTER_TEMPLATES {
        assert!(
            translate_glsl_es100_with_options(src, &CompileOptions::default()).is_ok(),
            "starter template `{name}` failed GLSL translation"
        );
    }
}

#[test]
fn alt_drag_clone_is_single_undo_transaction() {
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    session
        .execute(Command::AddObject3D {
            clip_id: clip_id.clone(),
            parent_id: None,
            kind: "box".to_owned(),
            name: Some("Box".to_owned()),
        })
        .unwrap();
    let clip = session.document().video_tracks[0].clips[0].clone();
    assert!(clip.objects_3d.len() >= 2);
    let src_id = clip.objects_3d[0].id.clone();
    let dst_id = clip.objects_3d[1].id.clone();
    let undo_before = session.history.undo_len();
    let count_before = clip.objects_3d.len();

    apply_object_drop(
        &mut session,
        &clip,
        &src_id,
        &dst_id,
        TreeDropMode::After,
        true,
    );
    assert_eq!(session.history.undo_len(), undo_before + 1);
    assert_eq!(
        session.document().video_tracks[0].clips[0].objects_3d.len(),
        count_before + 1
    );

    session.execute(Command::Undo).unwrap();
    assert_eq!(
        session.document().video_tracks[0].clips[0].objects_3d.len(),
        count_before
    );
}

#[test]
fn expanded_sub_axis_text_edit_commit_preserves_parent_pending_drag_tx() {
    let mut session = Session::new();
    let mut state = PanelsState::default();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();

    // Parent inline vector row starts PendingDrag on pointer-down (0 edits so far).
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "position",
        false,
        0.0,
        json!([0.0, 0.0]),
        ScrubEvent {
            started: true,
            changed: false,
            committed: false,
            cancelled: false,
        },
        &mut state,
    );
    assert!(session.history.is_in_transaction());
    assert!(owns_inspector_tx(&state, &clip_id, "position"));
    assert!(!state.inspector_tx_has_edits);

    // Same frame: expanded X sub-axis blurs its TextEdit with a committed value.
    // Because its tx_owner_key ("position::axis0") is distinct from parent ("position"),
    // it must NOT prematurely close parent's PendingDrag tx AND must commit outside it.
    apply_scrub_to_session_with_owner_key(
        &mut session,
        &clip_id,
        "position",
        "position::axis0",
        false,
        0.0,
        json!([42.0, 0.0]),
        ScrubEvent {
            started: false,
            changed: true,
            committed: true,
            cancelled: false,
        },
        &mut state,
    );
    assert!(session.history.is_in_transaction());
    assert!(owns_inspector_tx(&state, &clip_id, "position"));
    assert!(!state.inspector_tx_has_edits);

    // Same frame: an unchanged sub-axis Y blur must also leave parent's PendingDrag tx open.
    apply_scrub_to_session_with_owner_key(
        &mut session,
        &clip_id,
        "position",
        "position::axis1",
        false,
        0.0,
        json!([42.0, 0.0]),
        ScrubEvent {
            started: false,
            changed: false,
            committed: true,
            cancelled: false,
        },
        &mut state,
    );
    assert!(session.history.is_in_transaction());
    assert!(owns_inspector_tx(&state, &clip_id, "position"));

    // Next frame: parent transitions PendingDrag -> TextEdit and cancels its tx.
    // Sub-axis X's committed value [42.0, 0.0] must be preserved!
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "position",
        false,
        0.0,
        json!([42.0, 0.0]),
        ScrubEvent {
            started: false,
            changed: false,
            committed: false,
            cancelled: true,
        },
        &mut state,
    );
    assert!(!session.history.is_in_transaction());
    assert!(state.inspector_tx_owner.is_none());
    let pos = eval_prop_vec2(
        lookup_session_prop(&session, &clip_id, "position").as_ref(),
        0.0,
        [0.0, 0.0],
    );
    assert!((pos[0] - 42.0).abs() < 1e-6 && pos[1].abs() < 1e-6);
}

#[test]
fn upward_click_from_text_edit_to_pending_drag_preserves_committed_edit() {
    let mut session = Session::new();
    let mut state = PanelsState::default();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();

    // 1. Scrub TextEdit (lower widget "rotation") -> upward click onto upper widget ("opacity").
    let undo_before = session.history.undo_len();
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "opacity",
        false,
        0.0,
        json!(1.0),
        ScrubEvent {
            started: true,
            changed: false,
            committed: false,
            cancelled: false,
        },
        &mut state,
    );
    assert!(session.history.is_in_transaction());
    assert!(owns_inspector_tx(&state, &clip_id, "opacity"));

    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "rotation",
        false,
        0.0,
        json!(45.0),
        ScrubEvent {
            started: false,
            changed: true,
            committed: true,
            cancelled: false,
        },
        &mut state,
    );
    assert!(session.history.is_in_transaction());
    assert!(owns_inspector_tx(&state, &clip_id, "opacity"));
    assert!(!state.inspector_tx_has_edits);

    // Pointer-up on upper widget: PendingDrag -> TextEdit cancels opacity's tx.
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "opacity",
        false,
        0.0,
        json!(1.0),
        ScrubEvent {
            started: false,
            changed: false,
            committed: false,
            cancelled: true,
        },
        &mut state,
    );
    assert!(!session.history.is_in_transaction());
    assert_eq!(session.history.undo_len(), undo_before + 1);
    let rot = eval_prop_f64(
        lookup_session_prop(&session, &clip_id, "rotation").as_ref(),
        0.0,
        0.0,
    );
    assert!((rot - 45.0).abs() < 1e-6);

    // 2. Atomic color-picker commit (hex/swatch) while upper widget holds unedited PendingDrag.
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "opacity",
        false,
        0.0,
        json!(1.0),
        ScrubEvent {
            started: true,
            changed: false,
            committed: false,
            cancelled: false,
        },
        &mut state,
    );
    apply_color_picker_to_session(
        &mut session,
        &clip_id,
        "backgroundColor",
        false,
        0.0,
        json!([0.25, 0.5, 0.75]),
        super::super::controls::ColorPickerEvent {
            started: false,
            changed: true,
            committed: true,
            cancelled: false,
            eyedropper_clicked: false,
        },
        &mut state,
    );
    assert!(session.history.is_in_transaction());
    assert!(owns_inspector_tx(&state, &clip_id, "opacity"));
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "opacity",
        false,
        0.0,
        json!(1.0),
        ScrubEvent {
            started: false,
            changed: false,
            committed: false,
            cancelled: true,
        },
        &mut state,
    );
    assert!(!session.history.is_in_transaction());
    let bg = eval_prop_vec3(
        lookup_session_prop(&session, &clip_id, "backgroundColor").as_ref(),
        0.0,
        [0.0, 0.0, 0.0],
    );
    assert!(
        (bg[0] - 0.25).abs() < 1e-6 && (bg[1] - 0.5).abs() < 1e-6 && (bg[2] - 0.75).abs() < 1e-6
    );

    // 3. Paired scalar atomic commit while upper widget holds unedited PendingDrag.
    let undo_before_paired = session.history.undo_len();
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "opacity",
        false,
        0.0,
        json!(1.0),
        ScrubEvent {
            started: true,
            changed: false,
            committed: false,
            cancelled: false,
        },
        &mut state,
    );
    apply_paired_scalar_scrub(
        &mut session,
        &clip_id,
        "paired_test",
        ("propA", 3.0),
        ("propB", 7.0),
        ScrubEvent {
            started: false,
            changed: true,
            committed: true,
            cancelled: false,
        },
        &mut state,
    );
    assert!(session.history.is_in_transaction());
    assert!(owns_inspector_tx(&state, &clip_id, "opacity"));
    apply_scrub_to_session(
        &mut session,
        &clip_id,
        "opacity",
        false,
        0.0,
        json!(1.0),
        ScrubEvent {
            started: false,
            changed: false,
            committed: false,
            cancelled: true,
        },
        &mut state,
    );
    assert!(!session.history.is_in_transaction());
    assert_eq!(session.history.undo_len(), undo_before_paired + 1);
    let a = eval_prop_f64(
        lookup_session_prop(&session, &clip_id, "propA").as_ref(),
        0.0,
        0.0,
    );
    let b = eval_prop_f64(
        lookup_session_prop(&session, &clip_id, "propB").as_ref(),
        0.0,
        0.0,
    );
    assert!((a - 3.0).abs() < 1e-6 && (b - 7.0).abs() < 1e-6);
}

#[test]
fn export_containers_labels_codec_rules_and_transparent_alpha_cover_gif_wav_m4a() {
    // Container ALL and LABELS remain 1-to-1 and include GIF, WAV, and M4A.
    assert_eq!(ExportContainer::ALL.len(), ExportContainer::LABELS.len());
    for (c, expected) in ExportContainer::ALL
        .iter()
        .zip(ExportContainer::LABELS.iter())
    {
        assert_eq!(container_label(*c), *expected);
    }
    assert_eq!(container_label(ExportContainer::Gif), "GIF (.gif)");
    assert_eq!(container_label(ExportContainer::Wav), "WAV Audio (.wav)");
    assert_eq!(container_label(ExportContainer::M4a), "M4A Audio (.m4a)");

    // MP4 gating leaves GIF, WAV, and M4A available.
    assert_eq!(available_export_containers(true), &ExportContainer::ALL);
    let no_mp4 = available_export_containers(false);
    assert!(!no_mp4.contains(&ExportContainer::Mp4));
    assert!(no_mp4.contains(&ExportContainer::Gif));
    assert!(no_mp4.contains(&ExportContainer::Wav));
    assert!(no_mp4.contains(&ExportContainer::M4a));

    // Capabilities & codec compatibility matrix.
    assert!(ExportContainer::Wav.is_audio_only());
    assert!(ExportContainer::M4a.is_audio_only());
    assert!(!ExportContainer::Gif.is_audio_only());

    assert!(!ExportContainer::Gif.uses_video_codec());
    assert!(!ExportContainer::Wav.uses_video_codec());
    assert!(!ExportContainer::M4a.uses_video_codec());
    assert!(ExportContainer::Mp4.uses_video_codec());
    assert!(ExportContainer::WebM.uses_video_codec());
    assert!(ExportContainer::Mkv.uses_video_codec());

    assert!(!ExportContainer::PngSequence.supports_audio());
    assert!(!ExportContainer::Gif.supports_audio());
    assert!(ExportContainer::Wav.supports_audio());
    assert!(ExportContainer::M4a.supports_audio());

    assert!(!ExportContainer::PngSequence.requires_ffmpeg());
    assert!(!ExportContainer::Wav.requires_ffmpeg());
    assert!(ExportContainer::Gif.requires_ffmpeg());
    assert!(ExportContainer::M4a.requires_ffmpeg());

    assert!(allowed_video_codecs(ExportContainer::Gif).is_empty());
    assert!(allowed_video_codecs(ExportContainer::Wav).is_empty());
    assert!(allowed_video_codecs(ExportContainer::M4a).is_empty());

    assert_eq!(
        allowed_audio_codecs(ExportContainer::Gif),
        &[ExportAudioCodec::None]
    );
    assert_eq!(
        allowed_audio_codecs(ExportContainer::Wav),
        &[ExportAudioCodec::PcmS16Le]
    );
    assert_eq!(
        allowed_audio_codecs(ExportContainer::M4a),
        &[ExportAudioCodec::AacLc]
    );

    // Transparent alpha compatibility.
    assert!(supports_transparent_alpha(
        ExportContainer::PngSequence,
        ExportVideoCodec::Png
    ));
    assert!(supports_transparent_alpha(
        ExportContainer::Gif,
        ExportVideoCodec::H264
    ));
    assert!(supports_transparent_alpha(
        ExportContainer::WebM,
        ExportVideoCodec::Vp9
    ));
    assert!(supports_transparent_alpha(
        ExportContainer::Mkv,
        ExportVideoCodec::Vp9
    ));
    assert!(!supports_transparent_alpha(
        ExportContainer::Mp4,
        ExportVideoCodec::Vp9
    ));
    assert!(!supports_transparent_alpha(
        ExportContainer::WebM,
        ExportVideoCodec::Av1
    ));
    assert!(!supports_transparent_alpha(
        ExportContainer::Wav,
        ExportVideoCodec::Vp9
    ));
    assert!(!supports_transparent_alpha(
        ExportContainer::M4a,
        ExportVideoCodec::Vp9
    ));

    // Sanitization transitions across GIF, WAV, M4A, and VP9.
    let mut cfg = ExportSettings {
        container: ExportContainer::Gif,
        audio_codec: ExportAudioCodec::AacLc,
        transparent_alpha: true,
        ..ExportSettings::default()
    };
    sanitize_export_settings(&mut cfg, true);
    assert_eq!(cfg.audio_codec, ExportAudioCodec::None);
    assert!(cfg.transparent_alpha);

    cfg.container = ExportContainer::Wav;
    sanitize_export_settings(&mut cfg, true);
    assert_eq!(cfg.audio_codec, ExportAudioCodec::PcmS16Le);
    assert!(!cfg.transparent_alpha);

    cfg.container = ExportContainer::M4a;
    cfg.transparent_alpha = true;
    sanitize_export_settings(&mut cfg, true);
    assert_eq!(cfg.audio_codec, ExportAudioCodec::AacLc);
    assert!(!cfg.transparent_alpha);

    cfg.container = ExportContainer::Mp4;
    sanitize_export_settings(&mut cfg, false);
    assert_eq!(cfg.container, ExportContainer::WebM);
    assert_eq!(cfg.video_codec, ExportVideoCodec::Vp9);
    assert_eq!(cfg.audio_codec, ExportAudioCodec::Opus);
}

#[test]
fn panels_state_queues_and_drains_export_pz_project_action() {
    let mut state = PanelsState::default();
    assert!(state.take_project_actions().is_empty());

    state.pending_project_actions.push(ProjectAction::ExportPz);
    assert_eq!(state.take_project_actions(), vec![ProjectAction::ExportPz]);
    assert!(state.take_project_actions().is_empty());
}

#[test]
fn add_preset_clip_places_at_playhead_and_creates_new_track_on_overlap() {
    let mut session = Session::new();
    session.playback.current_frame = 24.0;
    let initial_tracks = session.document().video_tracks.len();
    add_preset_clip(&mut session, "text_2d", "Text");

    let selected_id = session
        .selection
        .clip_ids
        .first()
        .expect("newly added clip must be immediately selected")
        .clone();
    let added = find_clip_clone(&session, &selected_id).expect("find newly added clip");
    assert!(
        (added.start - 24.0).abs() < 1e-6,
        "new clip must start at playhead (24), not off-screen; got {}",
        added.start
    );
    // Default V1 has a clip covering [0, 180), so adding at frame 24 creates a new track above
    assert!(
        session.document().video_tracks.len() > initial_tracks,
        "overlapping playhead must create a new video track instead of pushing clip to frame 180+"
    );
}

#[test]
fn default_layer_mask_is_centered_in_sequence_pixels() {
    let mask = default_layer_mask(1, 1920, 1080);
    let PropertyValue::Json(Value::Array(arr)) = &mask.path.value else {
        panic!("expected JSON array path");
    };
    assert_eq!(arr.len(), 24);
    let xs = [
        arr[0].as_f64().unwrap(),
        arr[6].as_f64().unwrap(),
        arr[12].as_f64().unwrap(),
        arr[18].as_f64().unwrap(),
    ];
    let ys = [
        arr[1].as_f64().unwrap(),
        arr[7].as_f64().unwrap(),
        arr[13].as_f64().unwrap(),
        arr[19].as_f64().unwrap(),
    ];
    assert_eq!(xs, [480.0, 1440.0, 1440.0, 480.0]);
    assert_eq!(ys, [270.0, 270.0, 810.0, 810.0]);
}

#[test]
fn format_shader_compile_error_strips_naga_debug_noise() {
    let raw = r#"GLSL 450 parse error: [Error { kind: InvalidToken(Word("bogus")), meta: Span { start: 18, end: 23 } }]"#;
    let src = "void main() {\n    bogus;\n}\n";
    let formatted = format_shader_compile_error(raw, src);
    assert!(
        !formatted.contains("Error {"),
        "raw Debug leaked: {formatted}"
    );
    assert!(
        !formatted.contains("Span {"),
        "Span Debug leaked: {formatted}"
    );
    assert!(
        formatted.contains("Unexpected token `bogus`"),
        "missing readable message: {formatted}"
    );
    assert!(
        formatted.contains("line 2"),
        "missing line hint: {formatted}"
    );
}

#[test]
fn shader_editor_window_clears_left_inspector_and_stays_in_screen() {
    let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 752.0));
    let (pos, size) = compute_shader_editor_window_rect(screen);
    assert!(
        pos.x >= 390.0,
        "shader editor must clear left inspector (x >= 390): got {}",
        pos.x
    );
    assert!(pos.x + size.x <= screen.right() - 8.0);
    assert!(pos.y + size.y <= screen.bottom() - 8.0);
}

#[test]
fn shape_plus_smart_add_and_property_ordering() {
    assert!(shape_plus_prop_sort_key("name") < shape_plus_prop_sort_key("position"));
    assert!(shape_plus_prop_sort_key("position") < shape_plus_prop_sort_key("scale"));
    assert_eq!(
        shape_plus_string_enum_options("repeater", "composite"),
        Some(&["below", "above"][..])
    );
    assert_eq!(
        shape_plus_string_enum_options("fill", "fillRule"),
        Some(&["nonzero", "even-odd"][..])
    );
    assert_eq!(
        shape_plus_string_enum_options("polystar", "mode"),
        Some(&["star", "polygon"][..])
    );
    assert_eq!(
        shape_plus_string_enum_options("trim", "mode"),
        Some(&["simultaneous", "individual"][..])
    );
    assert_eq!(
        shape_plus_string_enum_options("mergePaths", "mode"),
        Some(&["merge", "add", "subtract", "intersect", "exclude"][..])
    );

    // Adding a shape at root wraps it in a Group with a default Fill so it is visible
    let mut payload = json!({ "version": 1, "contents": [] });
    let rect_id = add_shape_plus_item_smart(&mut payload, None, "rectangle")
        .expect("smart add rectangle at root");
    let root_contents = payload["contents"].as_array().expect("root contents");
    assert_eq!(root_contents.len(), 1);
    assert_eq!(root_contents[0]["kind"], "group");
    let group_children = root_contents[0]["contents"]
        .as_array()
        .expect("group children");
    assert!(group_children.iter().any(|c| c["id"] == rect_id));
    assert!(group_children.iter().any(|c| c["kind"] == "fill"));

    // Selecting a child inside a group inserts the next shape inside that same enclosing group
    let ell_id = add_shape_plus_item_smart(&mut payload, Some(&rect_id), "ellipse")
        .expect("smart add ellipse inside enclosing group");
    let group_children2 = payload["contents"][0]["contents"]
        .as_array()
        .expect("group children");
    assert!(group_children2.iter().any(|c| c["id"] == ell_id));
    assert_eq!(payload["contents"].as_array().unwrap().len(), 1);
}

#[test]
fn format_shader_compile_error_formats_unknown_variable_cleanly_with_balanced_parens() {
    let raw = r#"GLSL 450 parse error: [Error { kind: UnknownVariable("gl_FragColo"), meta: Span { start: 18, end: 29 } }]"#;
    let src = "void main() {\n    gl_FragColo = vec4(1.0);\n}\n";
    let formatted = format_shader_compile_error(raw, src);
    assert!(
        formatted.contains("Unknown variable `gl_FragColo`"),
        "unexpected formatting: {formatted}"
    );
    let opens = formatted.chars().filter(|&c| c == '(').count();
    let closes = formatted.chars().filter(|&c| c == ')').count();
    assert_eq!(
        opens, closes,
        "unbalanced parentheses in formatted error: {formatted}"
    );
}

#[test]
fn pz_clipboard_paste_helper_dispatches_effects_objects_material_masks_and_toasts() {
    let mut session = Session::new();
    let mut state = PanelsState::default();
    let mut actions = Vec::new();

    // Create a 3D scene clip with a Cube object
    let track_id = session.document().video_tracks[0].id.clone();
    let clip_res = session
        .execute(Command::AddClip {
            track_id,
            kind: "scene".to_owned(),
            name: Some("Scene".to_owned()),
            start: 0.0,
            length: 120.0,
        })
        .expect("add scene clip");
    let clip_id = clip_res["clip_id"].as_str().unwrap().to_owned();
    let obj_res = session
        .execute(Command::AddObject3D {
            clip_id: clip_id.clone(),
            kind: "shape".to_owned(),
            name: Some("Cube".to_owned()),
            parent_id: None,
        })
        .expect("add cube");
    let obj_id = obj_res["object_id"].as_str().unwrap().to_owned();

    // 1. Paste Effect JSON
    let ok = paste_pz_json_text(
        &mut session,
        &mut state,
        &mut actions,
        r#"{"type":"bloom","properties":{"name":"Glow"}}"#.to_owned(),
        "effects",
        Some(clip_id.clone()),
        None,
        None,
        None,
    );
    assert!(ok);
    assert!(matches!(
        actions.last(),
        Some(PanelAction::StatusMessage(m)) if m.contains("Pasted 1 effect")
    ));

    // 2. Copy & paste 3D Object JSON
    let obj_json = session
        .copy_objects_3d_as_pz_json(&clip_id, std::slice::from_ref(&obj_id))
        .expect("copy 3D object");
    let ok_obj = paste_pz_json_text(
        &mut session,
        &mut state,
        &mut actions,
        obj_json,
        "objects3d",
        Some(clip_id.clone()),
        None,
        None,
        None,
    );
    assert!(ok_obj);
    assert!(matches!(
        actions.last(),
        Some(PanelAction::StatusMessage(m)) if m.contains("Pasted 1 3D object")
    ));

    // 3. Copy & paste Material JSON
    let mat_json = session
        .copy_material_as_pz_json(&clip_id, &obj_id)
        .expect("copy material");
    let ok_mat = paste_pz_json_text(
        &mut session,
        &mut state,
        &mut actions,
        mat_json,
        "material",
        Some(clip_id.clone()),
        None,
        Some(obj_id.clone()),
        None,
    );
    assert!(ok_mat);
    assert!(matches!(
        actions.last(),
        Some(PanelAction::StatusMessage(m)) if m.contains("Pasted 3D material")
    ));

    // 4. Add a mask, copy & paste masks JSON
    let cid_uuid = Uuid::parse_str(&clip_id).unwrap();
    let mask = default_layer_mask(1, 1920, 1080);
    session
        .execute(Command::AddLayerMask {
            clip_id: cid_uuid,
            mask,
        })
        .expect("add mask");
    let masks_json = session.copy_masks_as_pz_json(&clip_id).expect("copy masks");
    let ok_masks = paste_pz_json_text(
        &mut session,
        &mut state,
        &mut actions,
        masks_json.clone(),
        "masks",
        Some(clip_id.clone()),
        None,
        None,
        None,
    );
    assert!(
        ok_masks,
        "masks paste failed: {:?}\njson={masks_json}",
        actions.last()
    );
    assert!(matches!(
        actions.last(),
        Some(PanelAction::StatusMessage(m)) if m.contains("Pasted 1 mask")
    ));

    // 5. Invalid JSON surfaces a clear Paste failed toast without panicking
    let ok_bad = paste_pz_json_text(
        &mut session,
        &mut state,
        &mut actions,
        "not valid json".to_owned(),
        "objects3d",
        Some(clip_id),
        None,
        None,
        None,
    );
    assert!(!ok_bad);
    assert!(matches!(
        actions.last(),
        Some(PanelAction::StatusMessage(m)) if m.starts_with("Paste failed:")
    ));
}

#[test]
fn live_2d_text_edit_commits_per_keystroke_and_coalesces_undo_transaction() {
    let mut session = Session::new();
    let track_id = session.document().video_tracks[0].id.clone();
    let clip_res = session
        .execute(Command::AddClip {
            track_id,
            kind: "text".to_owned(),
            name: Some("Text".to_owned()),
            start: 0.0,
            length: 120.0,
        })
        .expect("add 2d text clip");
    let clip_id = clip_res["clip_id"].as_str().unwrap().to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: clip_id.clone(),
            property: "text".to_owned(),
            value: json!("text"),
        })
        .expect("seed initial text");
    let undo_before = session.history.undo_len();

    let ctx = egui::Context::default();
    let theme = Cm3Theme::default();
    let mut state = PanelsState::default();
    let edit_id = Id::unique((clip_id.as_str(), "text", "text_edit"));

    // Frame 1: render once & focus the text input
    let mut out = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(320.0, 200.0))),
            ..Default::default()
        },
        |ui| {
            ui.memory_mut(|m| m.request_focus(edit_id));
            show_session_text_prop(
                ui,
                &mut session,
                &clip_id,
                "text",
                "Text",
                "text",
                0.0,
                &mut state,
                &theme,
            );
        },
    );
    out.textures_delta.clear();

    // Frame 2: type "!" while still focused -> session must update IMMEDIATELY (live preview)
    let mut out = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(320.0, 200.0))),
            events: vec![egui::Event::Text("!".to_owned())],
            ..Default::default()
        },
        |ui| {
            show_session_text_prop(
                ui,
                &mut session,
                &clip_id,
                "text",
                "Text",
                "text",
                0.0,
                &mut state,
                &theme,
            );
        },
    );
    out.textures_delta.clear();
    let live_val = eval_prop_str(
        lookup_session_prop(&session, &clip_id, "text").as_ref(),
        0.0,
        "",
    );
    assert!(
        live_val.contains('!'),
        "2D text must update live on keystroke while focused, got `{live_val}`"
    );
    assert!(session.history.is_in_transaction());

    // Frame 3: type "?" then press Enter -> commits into ONE coalesced undo entry
    let mut out = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(320.0, 200.0))),
            events: vec![
                egui::Event::Text("?".to_owned()),
                egui::Event::Key {
                    key: Key::Enter,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ui| {
            show_session_text_prop(
                ui,
                &mut session,
                &clip_id,
                "text",
                "Text",
                "text",
                0.0,
                &mut state,
                &theme,
            );
        },
    );
    out.textures_delta.clear();
    assert!(!session.history.is_in_transaction());
    assert_eq!(
        session.history.undo_len(),
        undo_before + 1,
        "multiple live keystrokes must coalesce into a single undo step"
    );
}

#[test]
fn default_fps_edit_commits_on_enter_and_when_leaving_settings() {
    let ctx = egui::Context::default();
    let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(360.0, 640.0));
    let mut state = PanelsState::default();
    let mut theme = Cm3Theme::default();
    let mut session = Session::new();
    session.execute(Command::SetFps { fps: 24.0 }).unwrap();

    let mut draw = |events, visible, focus_input| {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(rect),
                events,
                ..Default::default()
            },
            |ui| {
                if focus_input {
                    ctx.memory_mut(|m| m.request_focus(PanelsState::default_fps_input_id()));
                }
                if visible {
                    let mut actions = Vec::new();
                    show_settings_tab(ui, rect, &mut session, &mut theme, &mut state, &mut actions);
                }
                state.commit_default_fps_on_blur(&ctx, &mut session, visible);
            },
        );
        output.textures_delta.clear();
        (state.app_settings.default_fps, session.document().fps)
    };
    let key_event = |key, modifiers| egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    };
    let replace_text = |text: &str| {
        vec![
            key_event(Key::A, egui::Modifiers::COMMAND),
            egui::Event::Text(text.to_owned()),
        ]
    };

    assert_eq!(draw(vec![], true, false), (30.0, 24.0));
    assert_eq!(draw(replace_text("60"), true, true), (30.0, 24.0));
    assert_eq!(
        draw(
            vec![key_event(Key::Enter, egui::Modifiers::NONE)],
            true,
            false
        ),
        (60.0, 60.0),
    );
    assert_eq!(draw(replace_text("300"), true, true), (60.0, 60.0));
    assert_eq!(draw(vec![], false, false), (240.0, 240.0));
    assert_eq!(draw(vec![], true, false), (240.0, 240.0));
}

#[test]
fn settings_defaults_debug_report_and_sequence_control_dedup() {
    assert!(AppSettings::default().show_fps_overlay);

    let mut session = Session::new();
    let report = build_debug_info_report(&session, &AppSettings::default());
    assert!(report.contains("ZoidCraft"));
    assert!(report.contains("OS:"));
    assert!(!report.to_ascii_lowercase().contains("browser"));
    assert!(!report.to_ascii_lowercase().contains("reload"));

    session
        .execute(Command::AddSequenceControl {
            name: "ctl_value".to_owned(),
            kind: SequenceControlKind::Slider,
            value: PropertyValue::Number(0.0),
            min: Some(0.0),
            max: Some(100.0),
            step: Some(0.1),
        })
        .expect("add sequence control");
    let mirrored = collect_mirrored_sequence_control_keys(&session);
    assert!(
        mirrored.contains("ctl_value"),
        "mirrored ctl_value must be deduplicated from legacy ctl_* rows"
    );
}

#[test]
fn text_plus_2d_canonical_keys_evaluate_in_renderer_with_legacy_fallback() {
    // 1. Canonical Character Transform & Selected Transform serialize exact zc-render keys
    let mut root = default_text_plus_2d_root();
    let slot0 = text_plus_2d_slot_mut(&mut root, 0).expect("slot 0");
    slot0.insert("mode".to_owned(), json!(1));
    tp2d_write_vec2(slot0, "charPosition", [24.0, -12.0]);
    tp2d_write_vec2(slot0, "charScale", [1.5, 0.8]);
    tp2d_write_f64(slot0, "charRotation", 0.35);
    tp2d_write_f64(slot0, "charOpacity", 0.75);
    tp2d_write_f64(slot0, "delayPerCharacter", 2.0);

    let slot1 = text_plus_2d_slot_mut(&mut root, 1).expect("slot 1");
    slot1.insert("mode".to_owned(), json!(4));
    slot1.insert("characterNumbers".to_owned(), json!("2"));
    tp2d_write_vec2(slot1, "charPosition", [0.0, 50.0]);

    let evaluated =
        zc_render::text_plus_2d::evaluate_text_plus_2d_chars(&root, 3, 10.0, 30.0, None);
    assert_eq!(evaluated.len(), 3);
    assert!(
        !evaluated[0].is_neutral(),
        "Character Transform must produce non-neutral output in zc-render"
    );
    assert!((evaluated[0].offset[0] - 24.0).abs() < 1e-4);
    assert!((evaluated[0].scale[0] - 1.5).abs() < 1e-4);
    assert!((evaluated[0].rotation_rad - 0.35).abs() < 1e-4);
    assert!((evaluated[0].opacity - 0.75).abs() < 1e-4);
    // Second character also receives Selected Transform Y offset
    assert!(
        (evaluated[1].offset[1] - 38.0).abs() < 1e-3,
        "Selected Transform must apply to character 2, got {:?}",
        evaluated[1].offset
    );

    // 2. Selected Shake serializes exact shakeAmplitude / shakePeriod / shakePhase / characterNumbers
    let mut shake_root = default_text_plus_2d_root();
    let s0 = text_plus_2d_slot_mut(&mut shake_root, 0).expect("shake slot");
    s0.insert("mode".to_owned(), json!(5));
    s0.insert("characterNumbers".to_owned(), json!("1, 3"));
    tp2d_write_f64(s0, "shakeAmplitude", 0.5);
    tp2d_write_f64(s0, "shakePeriod", 24.0);
    tp2d_write_f64(s0, "shakePhase", 6.0);
    let shake_out =
        zc_render::text_plus_2d::evaluate_text_plus_2d_chars(&shake_root, 3, 0.0, 30.0, None);
    assert!((shake_out[0].rotation_rad - 0.5).abs() < 1e-4);
    assert!(shake_out[1].is_neutral());
    assert!((shake_out[2].rotation_rad - 0.5).abs() < 1e-4);

    // 3. Backward-compatible legacy read fallback
    let legacy = json!({
        "mode": 1,
        "offsetX": 15.0,
        "offsetY": -5.0,
        "scaleX": 2.0,
        "scaleY": 0.5,
        "rotation": 0.25,
        "opacity": 0.6,
        "delay": 3.0,
    });
    let legacy_map = legacy.as_object().unwrap();
    assert_eq!(
        tp2d_read_vec2(
            legacy_map,
            "charPosition",
            ("offsetX", "offsetY"),
            [0.0, 0.0]
        ),
        [15.0, -5.0]
    );
    assert_eq!(
        tp2d_read_vec2(legacy_map, "charScale", ("scaleX", "scaleY"), [1.0, 1.0]),
        [2.0, 0.5]
    );
    assert!((tp2d_read_f64(legacy_map, "charRotation", &["rotation"], 0.0) - 0.25).abs() < 1e-9);
    assert!((tp2d_read_f64(legacy_map, "charOpacity", &["opacity"], 1.0) - 0.6).abs() < 1e-9);
    assert!((tp2d_read_f64(legacy_map, "delayPerCharacter", &["delay"], 0.0) - 3.0).abs() < 1e-9);
}

#[test]
fn expression_editor_snippets_default_size_and_clear_resets_animated() {
    // Default window size must be at least 460x340 so Live/Diagnostic + buttons never clip
    assert_eq!(EXPR_EDITOR_DEFAULT_SIZE, Vec2::new(460.0, 340.0));

    // Bare constant literals and vector placeholders are cleanly REPLACED (never concatenated)
    assert_eq!(
        insert_expression_snippet("", "wiggle(2, 15)"),
        "wiggle(2, 15)"
    );
    assert_eq!(
        insert_expression_snippet("0.0", "Math.sin(time * 2.0) * 0.5 + 0.5"),
        "Math.sin(time * 2.0) * 0.5 + 0.5"
    );
    assert_eq!(
        insert_expression_snippet("[0.0,0.0,0.0]", "Math.sin(time * 2.0) * 0.5 + 0.5"),
        "Math.sin(time * 2.0) * 0.5 + 0.5"
    );
    assert_eq!(
        insert_expression_snippet("[1.0, 1.0, 1.0]", "wiggle(2, 15)"),
        "wiggle(2, 15)"
    );
    // Existing non-trivial expressions append on a fresh line instead of smashing together
    assert_eq!(
        insert_expression_snippet("const a = 1;", "a + time"),
        "const a = 1;\na + time"
    );

    // Clear Expression on an unanimated property resets `animated = false`
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    session
        .execute(Command::SetExpression {
            target_id: clip_id.clone(),
            property: "opacity".to_owned(),
            expression: Some("0.5".to_owned()),
        })
        .expect("set expression");
    let prop_before = lookup_session_prop(&session, &clip_id, "opacity").unwrap();
    assert!(prop_before.animated);
    assert!(prop_before.expression.is_some());

    clear_property_expression(&mut session, &clip_id, "opacity").expect("clear expression");
    let prop_after = lookup_session_prop(&session, &clip_id, "opacity").unwrap();
    assert!(prop_after.expression.is_none());
    assert!(
        !prop_after.animated,
        "clearing expression with no real keyframes must clear animated = false"
    );
}

#[test]
fn radial_and_grid_repeaters_appear_in_3d_object_picker() {
    let mut session = Session::new();
    session
        .execute(Command::EnablePlugin {
            plugin_id: "repeater".to_owned(),
        })
        .expect("enable repeater plugin");
    let items = build_object_picker_items(&session);
    assert!(
        items
            .iter()
            .any(|i| i.id == "zoidium:repeater/radial" && i.name == "Radial Repeater"),
        "Radial Repeater must be listed in 3D Objects picker"
    );
    assert!(
        items
            .iter()
            .any(|i| i.id == "zoidium:repeater/grid" && i.name == "Grid Repeater"),
        "Grid Repeater must be listed in 3D Objects picker"
    );
}

#[test]
fn sequence_control_identifier_sanitizes_spaces_and_leading_digits() {
    assert_eq!(
        sanitize_control_identifier("Slider 1", "ctl_slider_1"),
        "slider_1"
    );
    assert_eq!(
        sanitize_control_identifier("   ", "ctl_slider_1"),
        "ctl_slider_1"
    );
    assert_eq!(
        sanitize_control_identifier("123 boost!", "ctl_slider_1"),
        "ctl_123_boost"
    );
}

#[test]
fn system_font_discovery_includes_ttc_and_deduplicates_cleanly() {
    assert!(is_supported_system_font_ext("ttf"));
    assert!(is_supported_system_font_ext("OTF"));
    assert!(is_supported_system_font_ext("ttc"));
    assert!(is_supported_system_font_ext("TTC"));
    assert!(!is_supported_system_font_ext("woff2"));

    let tmp = std::env::temp_dir().join(format!("zc_ttc_test_{}", Uuid::new_v4()));
    std::fs::create_dir_all(&tmp).expect("create temp font dir");
    std::fs::write(tmp.join("Helvetica.ttc"), b"ttc").expect("write Helvetica.ttc");
    std::fs::write(tmp.join("Hiragino Sans GB.ttc"), b"ttc").expect("write Hiragino.ttc");
    std::fs::write(tmp.join("Menlo.ttc"), b"ttc").expect("write Menlo.ttc");
    std::fs::write(tmp.join("Avenir.ttf"), b"ttf").expect("write Avenir.ttf");
    std::fs::write(tmp.join("avenir.ttc"), b"ttc").expect("write avenir.ttc");
    std::fs::write(tmp.join(".SFNS.ttc"), b"hidden").expect("write hidden font");

    let found = discover_system_fonts_in_dirs(vec![tmp.clone()]);
    let _ = std::fs::remove_dir_all(&tmp);

    let names: Vec<&str> = found.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"Helvetica"), "missing Helvetica: {names:?}");
    assert!(
        names.contains(&"Hiragino Sans GB"),
        "missing Hiragino Sans GB: {names:?}"
    );
    assert!(names.contains(&"Menlo"), "missing Menlo: {names:?}");
    assert!(!names.iter().any(|n| n.starts_with('.')));
    // Case-insensitive dedup should keep only one Avenir entry (preferring .ttf over .ttc)
    let avenir_entries: Vec<_> = found
        .iter()
        .filter(|(n, _)| n.eq_ignore_ascii_case("avenir"))
        .collect();
    assert_eq!(avenir_entries.len(), 1);
    assert_eq!(
        avenir_entries[0]
            .1
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase),
        Some("ttf".to_owned())
    );
}

#[test]
fn zoidium_plugin_option_labels_and_opacity_0_100_range() {
    assert_eq!(
        template_option_labels("cardwipe", "flipAxis"),
        &["X", "Y", "Random"]
    );
    assert_eq!(
        template_option_labels("cardwipe", "flipDirection"),
        &["Positive", "Negative", "Random"]
    );
    assert_eq!(template_option_labels("cardwipe", "flipOrder").len(), 9);
    assert_eq!(
        template_option_labels("cardwipe", "backLayer"),
        &["Color", "Transparent"]
    );
    assert_eq!(
        template_option_labels("timecode", "displayFormat"),
        &[
            "SMPTE",
            "Frames",
            "Feet + Frames 35mm",
            "Feet + Frames 16mm",
        ]
    );
    assert_eq!(
        template_option_labels("timecode", "timeSource"),
        &["Layer Source", "Composition", "Custom"]
    );
    assert_eq!(
        template_option_labels("ccpixelpolly", "object"),
        &[
            "Triangles (flat)",
            "Triangles (textured)",
            "Squares (flat)",
            "Squares (textured)",
        ]
    );
    assert_eq!(
        template_option_labels("echo", "composite"),
        &["over", "behind", "add", "screen", "maximum"]
    );
    assert_eq!(template_option_labels("datamosh", "algorithm").len(), 81);
    assert_eq!(template_option_labels("datamosh", "motion").len(), 9);

    // 0..100 opacity / percentage params must NOT clamp to 1.0
    for (fx, key, def) in [
        ("radiowaves", "opacity", 100.0),
        ("timecode", "opacity", 100.0),
        ("radialshadow", "opacity", 50.0),
        ("ccpageturn", "backPageOpacity", 75.0),
        ("tracery", "threshold", 50.0),
    ] {
        let cfg = template_number_scrub_config_for_default(fx, key, def);
        assert!(
            cfg.max >= 100.0,
            "{fx}.{key} (default {def}) must allow 0..100 range, got max={}",
            cfg.max
        );
    }
    // Standard 0..1 opacity stays 0..1
    let std_op = template_number_scrub_config_for_default("dropshadow", "opacity", 0.5);
    assert!((std_op.max - 1.0).abs() < 1e-6);
}

#[test]
fn node_material_default_initialization_sets_512_and_procedural_graph() {
    let mut session = Session::new();
    session
        .execute(Command::EnablePlugin {
            plugin_id: "material-plus".to_owned(),
        })
        .expect("enable material-plus");
    let track_id = session.document().video_tracks[0].id.clone();
    let clip_res = session
        .execute(Command::AddClip {
            track_id,
            kind: "scene".to_owned(),
            name: Some("Scene".to_owned()),
            start: 0.0,
            length: 60.0,
        })
        .expect("add scene clip");
    let clip_id = clip_res["clip_id"].as_str().unwrap().to_owned();
    let obj_res = session
        .execute(Command::AddObject3D {
            clip_id,
            parent_id: None,
            kind: "box".to_owned(),
            name: Some("Cube".to_owned()),
        })
        .expect("add box");
    let obj_id = obj_res["object_id"].as_str().unwrap().to_owned();

    session
        .execute(Command::SetPropertyValue {
            target_id: obj_id.clone(),
            property: "material.type".to_owned(),
            value: json!(8),
        })
        .expect("switch to Node Material");
    initialize_node_material_defaults(&mut session, &obj_id, "material.", true);

    let proj = session.to_project();
    let cube = proj.sequence.video_tracks[0]
        .clips
        .iter()
        .flat_map(|c| match &c.layer.kind {
            zc_project::LayerKind::Scene3D { objects, .. } => objects.iter().collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .find(|o| o.id.to_string() == obj_id)
        .expect("cube in synced project");
    match cube.material.as_ref().expect("material") {
        zc_project::MaterialSpec::NodeMaterial {
            resolution, graph, ..
        } => {
            assert_eq!(
                *resolution, 512,
                "expected resolution 512 (bakeResolution: 2)"
            );
            let nodes = graph["nodes"].as_array().expect("default graph nodes");
            assert_eq!(nodes.len(), 3);
            assert_eq!(nodes[0]["type"], "noise");
            assert_eq!(nodes[1]["type"], "colorRamp");
            assert_eq!(nodes[2]["type"], "output");
        }
        other => panic!("expected MaterialSpec::NodeMaterial, got {other:?}"),
    }
}

#[test]
fn sequence_transparent_background_is_undoable_and_media_clips_clamp_to_sequence_length() {
    let mut session = Session::new();
    assert_eq!(
        session.active_sequence().background_color,
        [0.0, 0.0, 0.0, 1.0]
    );
    set_sequence_background_color_undoable(
        &mut session,
        [0.2, 0.4, 0.6, 0.0],
        "enable transparent background",
    );
    assert_eq!(
        session.active_sequence().background_color,
        [0.2, 0.4, 0.6, 0.0]
    );
    session.execute(Command::Undo).expect("undo background");
    assert_eq!(
        session.active_sequence().background_color,
        [0.0, 0.0, 0.0, 1.0]
    );

    // Short sequence: adding Preset shape / Composite must clamp to sequence length
    session
        .execute(Command::SetSequenceLength { frames: 60.0 })
        .expect("set short sequence length");
    add_preset_clip(&mut session, "preset_shape", "Preset shape");
    let added = session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.kind == "preset_shape")
        .expect("preset_shape clip");
    assert!((added.length - 60.0).abs() < 1e-6);
}

#[test]
fn oversized_text_guard_shader_type5_placeholder_and_history_humanizer() {
    assert_eq!(oversized_text_summary("hello"), None);
    let big = "a".repeat(MAX_INLINE_TEXT_CHARS + 2048);
    assert_eq!(
        oversized_text_summary(&big),
        Some("<10 KB text>".to_owned())
    );

    let placeholder = json!([{"position": 0.0, "color": "rgba(255,255,255,1.0)"}]);
    assert!(is_type5_bool_placeholder_value(&placeholder));
    let real_grad = json!([
        {"position": 0.0, "color": "rgba(0,0,0,1.0)"},
        {"position": 1.0, "color": "rgba(255,255,255,1.0)"}
    ]);
    assert!(!is_type5_bool_placeholder_value(&real_grad));

    assert_eq!(
        humanize_history_label("timeline.trim_clip_left"),
        "Trim clip left"
    );
    assert_eq!(
        humanize_history_label("scene.add_object_3d"),
        "Add 3D object"
    );
}

#[test]
fn custom_shader_rotation_scalar_uniform_uses_unit_scale_factor() {
    let rot_imported = custom_shader_number_scrub_config("Rotation", None);
    assert!(
        (rot_imported.scale_factor - 1.0).abs() < 1e-12,
        "PanZoid scalar Rotation uniform (type: 0) must NOT be multiplied by 180/pi, got {}",
        rot_imported.scale_factor
    );
    let rot_scalar = custom_shader_number_scrub_config("Rotation", Some(0));
    assert!((rot_scalar.scale_factor - 1.0).abs() < 1e-12);

    let angle_scalar = custom_shader_number_scrub_config("Angle", None);
    assert!((angle_scalar.scale_factor - 1.0).abs() < 1e-12);

    let explicit_angle = custom_shader_number_scrub_config("Angle", Some(6));
    assert!(
        (explicit_angle.scale_factor - std::f64::consts::PI / 180.0).abs() < 1e-12,
        "explicit angle kind (6) must still convert radians <-> degrees"
    );
}

#[test]
fn switching_sphere_to_node_material_with_real_prefix_seeds_and_copies_full_pz_json() {
    let mut session = Session::new();
    let track_id = session.document().video_tracks[0].id.clone();
    let clip_res = session
        .execute(Command::AddClip {
            track_id,
            kind: "scene".to_owned(),
            name: Some("Scene".to_owned()),
            start: 0.0,
            length: 60.0,
        })
        .expect("add scene clip");
    let clip_id = clip_res["clip_id"].as_str().unwrap().to_owned();
    let obj_res = session
        .execute(Command::AddObject3D {
            clip_id: clip_id.clone(),
            parent_id: None,
            kind: "sphere".to_owned(),
            name: Some("Sphere".to_owned()),
        })
        .expect("add sphere");
    let obj_id = obj_res["object_id"].as_str().unwrap().to_owned();

    // Simulate selecting "Node Material" (index 8) in `show_material_slot_inspector`,
    // which passes `prefix = "material"` (without a trailing dot).
    session
        .execute(Command::SetPropertyValue {
            target_id: obj_id.clone(),
            property: "material.type".to_owned(),
            value: json!(8),
        })
        .expect("switch to Node Material");
    assert!(!node_material_has_populated_graph(
        &session, &obj_id, "material"
    ));
    initialize_node_material_defaults(&mut session, &obj_id, "material", true);
    assert!(node_material_has_populated_graph(
        &session, &obj_id, "material"
    ));

    let copied = session
        .copy_material_as_pz_json(&clip_id, &obj_id)
        .expect("copy_material_as_pz_json");
    let parsed: Value = serde_json::from_str(&copied).expect("valid pz material json");
    assert_eq!(parsed["type"], "nodes");
    assert_eq!(parsed["properties"]["bakeResolution"], 2);
    let graph_obj: Value = if parsed.get("nodes").is_some() {
        parsed.clone()
    } else if let Some(raw_str) = parsed["properties"]["graph"].as_str() {
        serde_json::from_str(raw_str).expect("properties.graph JSON string")
    } else {
        parsed["properties"]["graph"].clone()
    };
    let nodes = graph_obj["nodes"]
        .as_array()
        .expect("copied NodeMaterial JSON must include nodes array");
    let links = graph_obj["links"]
        .as_array()
        .expect("copied NodeMaterial JSON must include links array");
    assert_eq!(nodes.len(), 3);
    assert_eq!(links.len(), 2);
}

#[test]
fn sampler2d_shader_uniform_lists_project_image_assets() {
    let mut session = Session::new();
    session.project.assets.insert(
        "hash_img_a".to_owned(),
        zc_project::AssetRecord {
            id: "hash_img_a".to_owned(),
            kind: AssetKind::Image,
            source: zc_project::AssetSource::default(),
            filename: "nebula.png".to_owned(),
            url: None,
            external: false,
            bytes: None,
            missing: false,
            sha256: String::new(),
            size: 1024,
            preserved: BTreeMap::new(),
        },
    );
    session.project.assets.insert(
        "hash_audio_b".to_owned(),
        zc_project::AssetRecord {
            id: "hash_audio_b".to_owned(),
            kind: AssetKind::Av,
            source: zc_project::AssetSource::default(),
            filename: "beat.wav".to_owned(),
            url: None,
            external: false,
            bytes: None,
            missing: false,
            sha256: String::new(),
            size: 2048,
            preserved: BTreeMap::new(),
        },
    );
    let (labels, ids, sel) = collect_shader_image_asset_choices(&session, "hash_img_a");
    assert_eq!(labels, vec!["None".to_owned(), "nebula.png".to_owned()]);
    assert_eq!(ids, vec![String::new(), "hash_img_a".to_owned()]);
    assert_eq!(sel, 1);

    let (_, _, none_sel) = collect_shader_image_asset_choices(&session, "");
    assert_eq!(none_sel, 0);
}

#[test]
fn delete_selected_effect_auto_selects_adjacent_remaining_effect() {
    let mut session = Session::new();
    let track_id = session.document().video_tracks[0].id.clone();
    let clip_res = session
        .execute(Command::AddClip {
            track_id,
            kind: "text".to_owned(),
            name: Some("Title".to_owned()),
            start: 0.0,
            length: 60.0,
        })
        .expect("add clip");
    let clip_id = clip_res["clip_id"].as_str().unwrap().to_owned();

    let fx1 = session
        .execute(Command::AddEffect {
            clip_id: clip_id.clone(),
            kind: "boxblur".to_owned(),
            name: Some("Blur".to_owned()),
        })
        .unwrap()["effect_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let fx2 = session
        .execute(Command::AddEffect {
            clip_id: clip_id.clone(),
            kind: "bloom".to_owned(),
            name: Some("Bloom".to_owned()),
        })
        .unwrap()["effect_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let fx3 = session
        .execute(Command::AddEffect {
            clip_id: clip_id.clone(),
            kind: "vignette".to_owned(),
            name: Some("Vignette".to_owned()),
        })
        .unwrap()["effect_id"]
        .as_str()
        .unwrap()
        .to_owned();

    // Select middle effect (fx2) and delete it -> selects next remaining effect (fx3).
    session
        .selection
        .select_effect(&fx2, "bloom", false)
        .unwrap();
    delete_effect_and_select_adjacent(&mut session, &clip_id, &fx2);
    assert_eq!(session.selection.effect_ids, vec![fx3.clone()]);

    // Delete last remaining selected effect (fx3) -> falls back to previous effect (fx1).
    delete_effect_and_select_adjacent(&mut session, &clip_id, &fx3);
    assert_eq!(session.selection.effect_ids, vec![fx1.clone()]);

    // Delete the sole remaining effect (fx1) -> selection becomes empty cleanly.
    delete_effect_and_select_adjacent(&mut session, &clip_id, &fx1);
    assert!(session.selection.effect_ids.is_empty());
}

#[test]
fn plugins_search_ranks_title_matches_before_tagline_matches() {
    let session = Session::new();
    let ranked = rank_filtered_plugin_categories(&session, "native");
    let ordered_names: Vec<String> = ranked
        .into_iter()
        .flat_map(|(_, _, descs)| descs.into_iter().map(|d| d.name))
        .collect();
    assert!(
        !ordered_names.is_empty(),
        "expected at least one match for 'native'"
    );
    assert_eq!(
        ordered_names[0], "Native FX",
        "expected title match 'Native FX' first, got: {ordered_names:?}"
    );
}

#[test]
fn sequence_narrow_layout_labels_and_header_ratio_stay_distinct() {
    let (s_narrow, sh_narrow) = sequence_motion_blur_labels(195.0);
    assert_ne!(s_narrow, sh_narrow);
    assert_eq!((s_narrow, sh_narrow), ("Samp.", "Shutt."));
    assert_eq!(sequence_motion_blur_labels(280.0), ("Samples", "Shutter"));

    let ratio = sequence_header_label_ratio(195.0, 0.40);
    assert!(
        ratio * 195.0 >= 110.0,
        "expected enough header room for 'Sequence' before reset icon, got {}px",
        ratio * 195.0
    );
}

#[test]
fn effect_group_custom_properties_can_be_found_and_updated() {
    let mut session = Session::new();
    let track_id = session.document().video_tracks[0].id.clone();
    let clip_res = session
        .execute(Command::AddClip {
            track_id,
            kind: "text".to_owned(),
            name: Some("Title".to_owned()),
            start: 0.0,
            length: 60.0,
        })
        .expect("add clip");
    let clip_id = clip_res["clip_id"].as_str().unwrap().to_owned();
    let clip_uuid = Uuid::parse_str(&clip_id).unwrap();
    let group_uuid = Uuid::new_v4();

    for track in &mut session.active_sequence_mut().video_tracks {
        for vc in &mut track.clips {
            if vc.id == clip_uuid {
                vc.layer.effects.push(EffectInstance {
                    id: group_uuid,
                    name: "Macro Group".to_owned(),
                    enabled: PropertyTrack::constant(PropertyValue::Option(1)),
                    kind: EffectKind::EffectGroup {
                        custom_properties: vec![CustomProperty {
                            name: "u_intensity".to_owned(),
                            prop_type: Some(0),
                            track: PropertyTrack::constant(PropertyValue::Number(0.5)),
                            metadata: BTreeMap::new(),
                        }],
                        children: Vec::new(),
                    },
                    params: BTreeMap::new(),
                    preserved: BTreeMap::new(),
                });
            }
        }
    }

    let group_id = group_uuid.to_string();
    let props = find_effect_group_custom_properties(&session, &clip_id, &group_id);
    assert_eq!(props.len(), 1);
    assert_eq!(props[0].name, "u_intensity");
    assert_eq!(props[0].track.value, PropertyValue::Number(0.5));

    assert!(update_effect_group_custom_property(
        &mut session,
        &clip_id,
        &group_id,
        "u_intensity",
        PropertyValue::Number(2.25),
    ));
    let updated = find_effect_group_custom_properties(&session, &clip_id, &group_id);
    assert_eq!(updated[0].track.value, PropertyValue::Number(2.25));
}

#[test]
fn font_dropdown_popup_mouse_wheel_scrolls_list_without_scrolling_parent() {
    let ctx = egui::Context::default();
    let popup_id = Id::unique("test_font_popup_scroll");
    let mut parent_offset_y = 0.0_f32;
    let mut popup_offset_y = 0.0_f32;
    let mut popup_center = Pos2::new(150.0, 180.0);

    let run_frame = |input: egui::RawInput,
                     parent_out: &mut f32,
                     popup_out: &mut f32,
                     center_out: &mut Pos2| {
        let mut full_out = ctx.run_ui(input, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.set_max_width(280.0);
                ui.set_max_height(260.0);
                let parent_scroll = ScrollArea::vertical()
                    .id_salt("test_parent_scroll")
                    .max_height(240.0)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for i in 0..40 {
                            ui.label(format!("Inspector row {i}"));
                        }
                        let area_resp = Area::new(popup_id)
                            .order(Order::Foreground)
                            .fixed_pos(Pos2::new(20.0, 60.0))
                            .interactable(true)
                            .movable(false)
                            .show(ui.ctx(), |ui| {
                                Frame::NONE
                                    .inner_margin(egui::Margin::same(6))
                                    .show(ui, |ui| {
                                        ui.set_min_width(240.0);
                                        ui.set_max_width(240.0);
                                        let list_scroll = ScrollArea::vertical()
                                            .id_salt((popup_id, "list"))
                                            .min_scrolled_height(180.0)
                                            .max_height(180.0)
                                            .auto_shrink([false, false])
                                            .show(ui, |ui| {
                                                for idx in 0..40 {
                                                    let _ = ui.allocate_exact_size(
                                                        Vec2::new(220.0, 24.0),
                                                        Sense::click(),
                                                    );
                                                    let _ = idx;
                                                }
                                            });
                                        *popup_out = list_scroll.state.clamped_offset().y;
                                    });
                            });
                        *center_out = area_resp.response.rect.center();
                        if ui.input(|i| {
                            i.pointer
                                .hover_pos()
                                .is_some_and(|p| area_resp.response.rect.contains(p))
                        }) {
                            ui.input_mut(|i| {
                                i.smooth_scroll_delta = Vec2::ZERO;
                            });
                        }
                    });
                *parent_out = parent_scroll.state.clamped_offset().y;
            });
        });
        full_out.textures_delta.clear();
    };

    // Frame 1: initial sizing pass; Frame 2: visible pass registering interactable layer.
    run_frame(
        egui::RawInput {
            time: Some(0.0),
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
            ..Default::default()
        },
        &mut parent_offset_y,
        &mut popup_offset_y,
        &mut popup_center,
    );
    run_frame(
        egui::RawInput {
            time: Some(0.016),
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
            ..Default::default()
        },
        &mut parent_offset_y,
        &mut popup_offset_y,
        &mut popup_center,
    );

    // Frame 3: hover inside the font popup and wheel-scroll down.
    run_frame(
        egui::RawInput {
            time: Some(0.048),
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
            events: vec![
                egui::Event::PointerMoved(popup_center),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: Vec2::new(0.0, -120.0),
                    modifiers: egui::Modifiers::NONE,
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Wheel,
                },
            ],
            ..Default::default()
        },
        &mut parent_offset_y,
        &mut popup_offset_y,
        &mut popup_center,
    );

    assert!(
        popup_offset_y > 10.0,
        "expected font dropdown list to scroll on mouse wheel, got offset={popup_offset_y}, parent={parent_offset_y}, popup_center={popup_center:?}"
    );
    assert!(
        parent_offset_y.abs() < 1e-3,
        "expected parent inspector ScrollArea NOT to scroll while wheeling over font popup, got {parent_offset_y}"
    );
}

#[test]
fn caches_reuse_across_frames_and_invalidate_on_revision() {
    let mut session = Session::new();
    let track_id = session.document().video_tracks[0].id.clone();
    let res = session
        .execute(Command::AddClip {
            track_id,
            kind: "scene".to_owned(),
            start: 0.0,
            length: 60.0,
            name: Some("Scene".to_owned()),
        })
        .unwrap();
    let clip_id = res["clip_id"].as_str().unwrap().to_owned();
    let mut state = PanelsState::default();

    // 1. Asset usage cache
    let u1 = compute_asset_usage_counts_cached(&session, &mut state);
    let u2 = compute_asset_usage_counts_cached(&session, &mut state);
    assert!(std::sync::Arc::ptr_eq(&u1, &u2));

    // 2. Selected clip cache
    let c1 = find_clip_cached(&session, &mut state, &clip_id).unwrap();
    let c2 = find_clip_cached(&session, &mut state, &clip_id).unwrap();
    assert!(std::sync::Arc::ptr_eq(&c1, &c2));

    // 3. Compiled expression cache
    let e1 = compile_expr_cached(
        &mut state,
        "Math.sin(time) * 10",
        zc_expr::ExprMode::ExpressionsPlus,
    )
    .unwrap();
    let e2 = compile_expr_cached(
        &mut state,
        "Math.sin(time) * 10",
        zc_expr::ExprMode::ExpressionsPlus,
    )
    .unwrap();
    assert!(std::sync::Arc::ptr_eq(&e1, &e2));

    // Mutating session bumps revision and invalidates revision-keyed caches.
    session
        .execute(Command::SetPropertyValue {
            target_id: clip_id.clone(),
            property: "opacity".to_owned(),
            value: json!(0.5),
        })
        .unwrap();
    let c3 = find_clip_cached(&session, &mut state, &clip_id).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&c1, &c3));
    let u3 = compute_asset_usage_counts_cached(&session, &mut state);
    assert!(!std::sync::Arc::ptr_eq(&u1, &u3));
}

#[test]
fn widget_tx_command_groups_continuous_drag_into_single_undo_entry() {
    let ctx = egui::Context::default();
    let mut session = Session::new();
    let track_id = session.document().video_tracks[0].id.clone();
    let res = session
        .execute(Command::AddClip {
            track_id,
            kind: "shape".to_owned(),
            start: 0.0,
            length: 60.0,
            name: Some("Box".to_owned()),
        })
        .unwrap();
    let clip_id = res["clip_id"].as_str().unwrap().to_owned();
    let mut state = PanelsState::default();
    let initial_undo_len = session.history.undo_len();

    let run_step = |primary_down: bool,
                    tracker: WidgetTxTracker,
                    val: f64,
                    session: &mut Session,
                    state: &mut PanelsState| {
        let mut events = Vec::new();
        if primary_down {
            events.push(egui::Event::PointerButton {
                pos: Pos2::new(20.0, 20.0),
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            });
        } else {
            events.push(egui::Event::PointerButton {
                pos: Pos2::new(20.0, 20.0),
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            });
        }
        let raw = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(300.0, 120.0))),
            events,
            ..Default::default()
        };
        let mut full = ctx.run_ui(raw, |ui| {
            apply_widget_tx_command(
                ui,
                session,
                state,
                &clip_id,
                "opacity_slider",
                "edit opacity",
                tracker,
                |session| {
                    let _ = session.execute(Command::SetPropertyValue {
                        target_id: clip_id.clone(),
                        property: "opacity".to_owned(),
                        value: json!(val),
                    });
                },
            );
        });
        full.textures_delta.clear();
    };

    // Start drag + 3 mid-drag frames + release frame -> must produce EXACTLY 1 undo entry.
    run_step(
        true,
        WidgetTxTracker {
            drag_started: true,
            drag_stopped: false,
            changed: true,
        },
        0.8,
        &mut session,
        &mut state,
    );
    for v in [0.7, 0.6, 0.5] {
        run_step(
            true,
            WidgetTxTracker {
                drag_started: false,
                drag_stopped: false,
                changed: true,
            },
            v,
            &mut session,
            &mut state,
        );
    }
    assert!(session.history.is_in_transaction());
    run_step(
        false,
        WidgetTxTracker {
            drag_started: false,
            drag_stopped: true,
            changed: false,
        },
        0.5,
        &mut session,
        &mut state,
    );
    assert!(!session.history.is_in_transaction());
    assert_eq!(
        session.history.undo_len(),
        initial_undo_len + 1,
        "continuous slider drag must coalesce into a single undo entry"
    );
}
