//! Unit tests for the graph editor.

use super::*;
use egui::{Color32, Rect, Shape, pos2, vec2};
use std::collections::BTreeMap;
use zc_color::ThemePalette;
use zc_session::{Command, Session};

#[test]
fn graph_coordinate_roundtrip() {
    let state = GraphEditorState::new();
    let rect = Rect::from_min_size(pos2(100.0, 50.0), vec2(600.0, 400.0));
    let x = state.frame_to_screen_x(120.0, rect);
    let f_back = state.screen_x_to_frame(x, rect);
    assert!((f_back - 120.0).abs() < 1e-4);

    let y = state.value_to_screen_y(42.5, rect);
    let v_back = state.screen_y_to_value(y, rect);
    assert!((v_back - 42.5).abs() < 1e-4);
}

#[test]
fn adaptive_grid_uses_1_2_5_10_steps() {
    let (major, minor) = adaptive_grid_step(0.35);
    assert!(major > 0.0);
    assert!((minor * 5.0 - major).abs() < 1e-9);
}

#[test]
fn channel_value_write_preserves_other_vector_axes() {
    let v = serde_json::json!([1.0, 2.0, 3.0]);
    let updated = write_channel_value(&v, Some(1), 9.5);
    assert_eq!(updated, serde_json::json!([1.0, 9.5, 3.0]));
}

#[test]
fn parse_cubic_bezier_accepts_css_and_array_formats() {
    assert_eq!(
        parse_cubic_bezier_str("cubic-bezier(0.25, 0.1, 0.25, 1.4)"),
        Some([0.25, 0.1, 0.25, 1.4])
    );
    assert_eq!(
        parse_cubic_bezier_str("CUBIC-BEZIER( 0.42 , -0.2 , 0.58 , 1.0 );"),
        Some([0.42, -0.2, 0.58, 1.0])
    );
    assert_eq!(
        parse_cubic_bezier_str("[0.42, -0.2, 0.58, 1.0]"),
        Some([0.42, -0.2, 0.58, 1.0])
    );
    assert_eq!(
        parse_cubic_bezier_str("1.5, 0.0, -0.5, 1.0"),
        Some([1.0, 0.0, 0.0, 1.0])
    );
    assert_eq!(parse_cubic_bezier_str("not-a-curve"), None);
}

#[test]
fn speed_graph_numerical_derivative_and_handle_conversion() {
    // Linear ramp from 0 to 120 over 30 frames at 30 fps => exact velocity = 120 u/s.
    let track = CurveTrack {
        track_key: "clip1:opacity".to_string(),
        target_id: "clip1".to_string(),
        property: "opacity".to_string(),
        channel_index: None,
        display_name: "opacity".to_string(),
        color: COLOR_AXIS_X,
        frame_offset: 0.0,
        is_expression: false,
        expression: None,
        keyframes: vec![
            CurveKeyframe {
                local_frame: 0.0,
                value: 0.0,
                full_value: serde_json::json!(0.0),
                tween: 1,
                control_points: [[-10.0, 0.0], [10.0, 0.0]],
                continuous_tangent: true,
            },
            CurveKeyframe {
                local_frame: 30.0,
                value: 120.0,
                full_value: serde_json::json!(120.0),
                tween: 1,
                control_points: [[-10.0, 0.0], [10.0, 0.0]],
                continuous_tangent: true,
            },
        ],
    };
    let handles = BTreeMap::new();
    let v_mid = eval_curve_velocity(&track, 15.0, 30.0, &handles);
    assert!(
        (v_mid - 120.0).abs() < 1e-3,
        "expected 120 u/s for linear ramp, got {v_mid}"
    );
    assert_eq!(eval_curve_velocity(&track, -5.0, 30.0, &handles), 0.0);
    assert_eq!(eval_curve_velocity(&track, 40.0, 30.0, &handles), 0.0);

    // Speed handle conversion: outgoing handle dx=10 frames at speed 150 u/s @ 30 fps => dy = 50.
    let out_t = speed_handle_to_tangent(10.0, 150.0, 30.0, true, 30.0);
    assert!((out_t[0] - 10.0).abs() < 1e-9);
    assert!((out_t[1] - 50.0).abs() < 1e-9);
    // Incoming handle dx=-15 frames at speed 150 u/s @ 30 fps => dy = -75.
    let in_t = speed_handle_to_tangent(-15.0, 150.0, 30.0, false, 30.0);
    assert!((in_t[0] - (-15.0)).abs() < 1e-9);
    assert!((in_t[1] - (-75.0)).abs() < 1e-9);
}

#[test]
fn expression_curve_sampling_evaluates_live_waveform() {
    let session = Session::new();
    let track = CurveTrack {
        track_key: "clip1:x".to_string(),
        target_id: "clip1".to_string(),
        property: "x".to_string(),
        channel_index: None,
        display_name: "x".to_string(),
        color: COLOR_AXIS_X,
        frame_offset: 0.0,
        is_expression: true,
        expression: Some("Math.sin(time * Math.PI) * 100".to_string()),
        keyframes: vec![CurveKeyframe {
            local_frame: 0.0,
            value: 0.0,
            full_value: serde_json::json!(0.0),
            tween: 1,
            control_points: [[-10.0, 0.0], [10.0, 0.0]],
            continuous_tangent: true,
        }],
    };
    let handles = BTreeMap::new();
    let samples = sample_expression_curve(&track, session.document(), &handles, 0.0, 30.0, 30);
    assert_eq!(samples.len(), 31);
    // At frame 0 (time = 0s): sin(0) * 100 = 0
    assert!(samples[0].1.abs() < 1e-3);
    // At frame 15 (time = 0.5s at 30 fps): sin(pi/2) * 100 = 100
    assert!(
        (samples[15].1 - 100.0).abs() < 1e-2,
        "expected ~100 at half-second peak, got {}",
        samples[15].1
    );
    // Velocity at frame 0 (t=0) of 100*sin(pi*t) is 100*pi ~= 314.159 u/s.
    let vel_samples =
        sample_expression_velocity_curve(&track, session.document(), &handles, 0.0, 30.0, 30);
    assert!(
        (vel_samples[0].1 - 100.0 * std::f64::consts::PI).abs() < 2.0,
        "expected ~314.16 u/s at t=0, got {}",
        vel_samples[0].1
    );
}

#[test]
fn headless_graph_editor_pinch_cmd_scroll_trackpad_pan_and_ruler_x_only() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let mut state = GraphEditorState::new();
    state.open = true;
    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(900.0, 600.0));
    // Curve canvas center is comfortably inside the right-hand canvas in both Vanilla and Graph+ modes.
    let canvas_pos = pos2(560.0, 320.0);

    let step = |session: &mut Session,
                state: &mut GraphEditorState,
                time: f64,
                modifiers: egui::Modifiers,
                mut events: Vec<egui::Event>| {
        events.insert(0, egui::Event::ModifiersChanged(modifiers));
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(time),
            events,
            focused: true,
            ..Default::default()
        };
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    let _ = show_graph_editor_panel(ui, session, state, &palette);
                });
        });
        full.textures_delta.clear();
    };

    // Prime layout.
    step(
        &mut session,
        &mut state,
        0.0,
        egui::Modifiers::NONE,
        vec![egui::Event::PointerMoved(canvas_pos)],
    );

    // 1. Trackpad pinch on curve canvas zooms BOTH X and Y in (shrinks units/px!).
    let (zx0, zy0) = (state.zoom_x, state.zoom_y);
    step(
        &mut session,
        &mut state,
        0.016,
        egui::Modifiers::NONE,
        vec![
            egui::Event::PointerMoved(canvas_pos),
            egui::Event::Zoom(1.25),
        ],
    );
    assert!(
        state.zoom_x < zx0,
        "pinch zoom-in must shrink zoom_x (units/px)"
    );
    assert!(
        state.zoom_y < zy0,
        "pinch zoom-in must shrink zoom_y (units/px)"
    );

    // 2. Alt + Trackpad pinch zooms Y only.
    let (zx1, zy1) = (state.zoom_x, state.zoom_y);
    step(
        &mut session,
        &mut state,
        0.032,
        egui::Modifiers::ALT,
        vec![
            egui::Event::PointerMoved(canvas_pos),
            egui::Event::Zoom(1.25),
        ],
    );
    assert!(
        (state.zoom_x - zx1).abs() < 1e-9,
        "Alt+Pinch must not alter zoom_x"
    );
    assert!(state.zoom_y < zy1, "Alt+Pinch must zoom Y");

    // 3. Cmd+Scroll on curve canvas zooms BOTH X and Y (regression test for egui 0.37).
    let (zx2, zy2) = (state.zoom_x, state.zoom_y);
    step(
        &mut session,
        &mut state,
        0.064,
        egui::Modifiers::COMMAND,
        vec![
            egui::Event::PointerMoved(canvas_pos),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: vec2(0.0, 1.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Wheel,
                modifiers: egui::Modifiers::COMMAND,
            },
        ],
    );
    assert!(state.zoom_x < zx2, "Cmd+Scroll must zoom X in");
    assert!(state.zoom_y < zy2, "Cmd+Scroll must zoom Y in");

    // 4. Trackpad two-finger scroll pans BOTH X and Y without zooming either axis.
    let (cx_before, cy_before) = (state.center_x, state.center_y);
    let (zx3, zy3) = (state.zoom_x, state.zoom_y);
    step(
        &mut session,
        &mut state,
        0.35,
        egui::Modifiers::NONE,
        vec![
            egui::Event::PointerMoved(canvas_pos),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: vec2(12.0, -8.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Trackpad,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(
        (state.zoom_x - zx3).abs() < 1e-9,
        "trackpad pan must not zoom X"
    );
    assert!(
        (state.zoom_y - zy3).abs() < 1e-9,
        "trackpad pan must not zoom Y"
    );
    assert!(
        state.center_x < cx_before,
        "positive trackpad dx pans center_x left"
    );
    assert!(
        state.center_y < cy_before,
        "negative trackpad dy pans center_y down"
    );

    // 5. Ruler gestures always affect X only (never zoom_y or center_y).
    let ruler_ctx = egui::Context::default();
    let ruler_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(600.0, TOP_RULER_HEIGHT));
    let ruler_hover = pos2(300.0, 15.0);
    let step_ruler = |session: &mut Session,
                      state: &mut GraphEditorState,
                      time: f64,
                      modifiers: egui::Modifiers,
                      mut events: Vec<egui::Event>| {
        events.insert(0, egui::Event::ModifiersChanged(modifiers));
        let raw = egui::RawInput {
            screen_rect: Some(ruler_rect),
            time: Some(time),
            events,
            focused: true,
            ..Default::default()
        };
        let mut full = ruler_ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    let mut act = GraphEditorAction::default();
                    show_graph_ruler(ui, session, state, ruler_rect, true, &mut act);
                });
        });
        full.textures_delta.clear();
    };

    step_ruler(
        &mut session,
        &mut state,
        0.0,
        egui::Modifiers::NONE,
        vec![egui::Event::PointerMoved(ruler_hover)],
    );
    let (zx_r, zy_r) = (state.zoom_x, state.zoom_y);
    step_ruler(
        &mut session,
        &mut state,
        0.016,
        egui::Modifiers::NONE,
        vec![
            egui::Event::PointerMoved(ruler_hover),
            egui::Event::Zoom(1.25),
        ],
    );
    assert!(state.zoom_x < zx_r, "ruler pinch must zoom X");
    assert!(
        (state.zoom_y - zy_r).abs() < 1e-9,
        "ruler pinch must leave zoom_y unchanged"
    );

    let (cx_r, cy_r) = (state.center_x, state.center_y);
    step_ruler(
        &mut session,
        &mut state,
        0.032,
        egui::Modifiers::NONE,
        vec![
            egui::Event::PointerMoved(ruler_hover),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: vec2(10.0, -15.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Trackpad,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(
        state.center_x < cx_r,
        "ruler trackpad pan must move center_x"
    );
    assert!(
        (state.center_y - cy_r).abs() < 1e-9,
        "ruler trackpad pan must leave center_y unchanged"
    );
}

#[test]
fn elide_channel_name_preserves_axis_suffixes() {
    let short = elide_channel_name("Hero.pos.X", 20);
    assert_eq!(short, "Hero.pos.X");

    let elided_x = elide_channel_name("HeroTitleText.position.X", 12);
    assert!(
        elided_x.ends_with(".X"),
        "elided channel name must preserve .X suffix, got '{elided_x}'"
    );
    assert!(elided_x.chars().count() <= 12);

    let elided_z = elide_channel_name("VeryLongObjectName.rotation.Z", 10);
    assert!(
        elided_z.ends_with(".Z"),
        "elided channel name must preserve .Z suffix, got '{elided_z}'"
    );
    assert!(elided_z.chars().count() <= 10);
}

#[test]
fn hold_and_linear_segments_omit_bezier_handles() {
    let state = GraphEditorState::new();
    let make_track = |tween0: u16, tween1: u16| CurveTrack {
        track_key: "t:p".to_owned(),
        target_id: "t".to_owned(),
        property: "p".to_owned(),
        channel_index: None,
        display_name: "p".to_owned(),
        color: Color32::WHITE,
        frame_offset: 0.0,
        is_expression: false,
        expression: None,
        keyframes: vec![
            CurveKeyframe {
                local_frame: 0.0,
                value: 0.0,
                full_value: serde_json::json!(0.0),
                tween: tween0,
                control_points: [[0.25, 0.25], [0.75, 0.75]],
                continuous_tangent: true,
            },
            CurveKeyframe {
                local_frame: 30.0,
                value: 100.0,
                full_value: serde_json::json!(100.0),
                tween: tween1,
                control_points: [[0.25, 0.25], [0.75, 0.75]],
                continuous_tangent: true,
            },
        ],
    };
    let hp = BezierHandlePair {
        in_tangent: [-5.0, 0.0],
        out_tangent: [5.0, 0.0],
        continuous: true,
    };

    // Hold (0) and Linear (1) must NOT produce tangent handles.
    let hold_track = make_track(1, 0);
    assert!(effective_in_tangent(&state, &hold_track, 1, hp).is_none());
    assert!(effective_out_tangent(&state, &hold_track, 0, hp).is_none());

    let linear_track = make_track(1, 1);
    assert!(effective_in_tangent(&state, &linear_track, 1, hp).is_none());
    assert!(effective_out_tangent(&state, &linear_track, 0, hp).is_none());

    // Ease (7) and Bezier must produce tangent handles.
    let ease_track = make_track(1, 7);
    assert!(effective_in_tangent(&state, &ease_track, 1, hp).is_some());
    assert!(effective_out_tangent(&state, &ease_track, 0, hp).is_some());
}

#[test]
fn scalar_tracks_render_solo_and_zero_baseline_label_always_visible() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session
        .selection
        .select_clip(clip_id.clone(), Some("scene"), false);
    let fx_id = session
        .execute(Command::AddEffect {
            clip_id: clip_id.clone(),
            kind: "bloom".to_owned(),
            name: Some("Bloom".to_owned()),
        })
        .unwrap()["effect_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let _ = session.execute(Command::SetKeyframe {
        target_id: fx_id.clone(),
        property: "amount".to_owned(),
        frame: 0.0,
        value: serde_json::json!(0.0),
        tween: Some(7),
    });
    let _ = session.execute(Command::SetKeyframe {
        target_id: fx_id.clone(),
        property: "amount".to_owned(),
        frame: 30.0,
        value: serde_json::json!(-10.0),
        tween: Some(7),
    });

    let mut state = GraphEditorState::new();
    state.open = true;
    // Position 0.0 very close to the top of the Y viewport so naive LEFT_BOTTOM would clip it.
    state.center_y = -5.0;
    state.zoom_y = 0.04;

    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(1280.0, 800.0));
    let mut raw = egui::RawInput {
        screen_rect: Some(screen_rect),
        time: Some(0.0),
        focused: true,
        ..Default::default()
    };
    // Frame 0 runs egui::Window's sizing pass; Frame 1 paints the visible canvas & Y-axis labels.
    let mut out0 = ctx.run_ui(raw.clone(), |ui| {
        let _ = show_graph_editor_window(ui.ctx(), &mut session, &mut state, &palette);
    });
    out0.textures_delta.clear();
    raw.time = Some(0.02);
    let mut out = ctx.run_ui(raw.clone(), |ui| {
        let _ = show_graph_editor_window(ui.ctx(), &mut session, &mut state, &palette);
    });
    out.textures_delta.clear();
    let has_zero_label = out
        .shapes
        .iter()
        .any(|cs| matches!(&cs.shape, Shape::Text(ts) if ts.galley.text() == "0"));
    assert!(
        has_zero_label,
        "Y-axis must render visible '0' baseline label whenever 0.0 is in visible range"
    );

    // Verify scalar track (`Bloom.amount`) has a Solo button rendered (no channel_index).
    let tracks = collect_curve_tracks(&session, &state, false);
    let scalar = tracks
        .iter()
        .find(|t| t.property == "amount" && t.channel_index.is_none())
        .expect("scalar Bloom.amount track");
    assert!(state.is_track_visible(&scalar.track_key));
    state.solo_track = Some(scalar.track_key.clone());
    raw.time = Some(0.05);
    let mut out2 = ctx.run_ui(raw, |ui| {
        let _ = show_graph_editor_window(ui.ctx(), &mut session, &mut state, &palette);
    });
    out2.textures_delta.clear();
    assert_eq!(state.solo_track.as_deref(), Some(scalar.track_key.as_str()));
}

#[test]
fn elide_channel_name_distinguishes_camera_position_rotation_scale() {
    // Previously both Camera.position.X and Camera.rotation.X collapsed to `Came..ion.X`.
    let pos_11 = elide_channel_name("Camera.position.X", 11);
    let rot_11 = elide_channel_name("Camera.rotation.X", 11);
    let scl_11 = elide_channel_name("Camera.scale.X", 11);
    assert_eq!(pos_11, "Cam.pos.X");
    assert_eq!(rot_11, "Cam.rot.X");
    assert_eq!(scl_11, "Cam.scl.X");
    assert_ne!(pos_11, rot_11);
    assert_ne!(pos_11, scl_11);
    assert_ne!(rot_11, scl_11);

    // Even at ultra-narrow widths (7..=16), position vs rotation must stay distinct.
    for width in 7..=16 {
        let p = elide_channel_name("Camera.position.X", width);
        let r = elide_channel_name("Camera.rotation.X", width);
        assert!(p.chars().count() <= width, "width={width}, got '{p}'");
        assert!(r.chars().count() <= width, "width={width}, got '{r}'");
        assert!(p.ends_with(".X"), "width={width}, got '{p}'");
        assert!(r.ends_with(".X"), "width={width}, got '{r}'");
        assert_ne!(
            p, r,
            "Camera.position.X and Camera.rotation.X collided as '{p}' at width={width}"
        );
    }

    // Common scalar property abbreviations.
    assert_eq!(elide_channel_name("Camera.opacity", 10), "Camera.op");
    assert_eq!(elide_channel_name("Bloom.amount", 10), "Bloom.amt");
}

#[test]
fn graph_editor_pastes_property_track_and_keyframe_array_json_and_css_easing() {
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session.selection.select_clip(clip_id.clone(), None, false);
    session
        .execute(Command::SetKeyframe {
            target_id: clip_id.clone(),
            property: "opacity".to_owned(),
            frame: 0.0,
            value: serde_json::json!(1.0),
            tween: Some(1),
        })
        .expect("seed opacity keyframe");

    let mut state = GraphEditorState::new();
    state.focused_property = Some((clip_id.clone(), "opacity".to_owned()));
    state.follow_selection = false;
    let tracks = collect_curve_tracks(&session, &state, false);
    assert!(!tracks.is_empty());

    // 1. Paste full property-track JSON with expression + keyframes.
    let prop_track_json = r#"{
            "animated": true,
            "expression": "Math.abs(Math.sin(time))",
            "keyframes": [
                { "frame": 0, "value": 0.2, "tween": 1 },
                { "frame": 24, "value": 0.9, "tween": 1 }
            ]
        }"#;
    let mut action = GraphEditorAction::default();
    assert!(handle_graph_paste_text(
        &mut session,
        &mut state,
        &tracks,
        Some(prop_track_json),
        &mut action,
    ));
    assert!(action.curves_modified);
    let prop = &session.document().video_tracks[0].clips[0].properties["opacity"];
    assert_eq!(prop.keyframes.len(), 2);
    assert_eq!(prop.expression.as_deref(), Some("Math.abs(Math.sin(time))"));
    assert!((prop.keyframes[1].frame - 24.0).abs() < 1e-6);

    // 2. Paste bare keyframe array JSON shifted to current playhead (frame 10).
    session.playback.current_frame = 10.0;
    let kf_array_json = r#"[
            { "frame": 0, "value": 0.1, "tween": 1 },
            { "frame": 15, "value": 0.8, "tween": 1 }
        ]"#;
    let mut action2 = GraphEditorAction::default();
    assert!(handle_graph_paste_text(
        &mut session,
        &mut state,
        &tracks,
        Some(kf_array_json),
        &mut action2,
    ));
    assert!(action2.curves_modified);
    let prop2 = &session.document().video_tracks[0].clips[0].properties["opacity"];
    assert_eq!(prop2.keyframes.len(), 2);
    assert!((prop2.keyframes[0].frame - 10.0).abs() < 1e-6);
    assert!((prop2.keyframes[1].frame - 25.0).abs() < 1e-6);

    // 3. Round-trip copy_property_as_pz_json succeeds on the pasted track.
    let exported = session
        .copy_property_as_pz_json(&clip_id, "opacity")
        .expect("export property track json");
    assert!(exported.contains("\"keyframes\""));

    // 4. CSS cubic-bezier(...) still parses and updates easing_plus_curve cleanly.
    let tracks2 = collect_curve_tracks(&session, &state, false);
    let mut action3 = GraphEditorAction::default();
    assert!(handle_graph_paste_text(
        &mut session,
        &mut state,
        &tracks2,
        Some("cubic-bezier(0.20, 0.80, 0.40, 1.00)"),
        &mut action3,
    ));
    assert!((state.easing_plus_curve[0] - 0.20).abs() < 1e-4);
    assert!((state.easing_plus_curve[1] - 0.80).abs() < 1e-4);
}

#[test]
fn graph_editor_scalar_curve_json_paste_onto_vec3_preserves_other_channels_and_roundtrips_copy() {
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session.selection.select_clip(clip_id.clone(), None, false);

    // Seed a Vec3 animated property (`position`) with distinct X/Y/Z trajectories.
    session
        .execute(Command::SetKeyframe {
            target_id: clip_id.clone(),
            property: "position".to_owned(),
            frame: 0.0,
            value: serde_json::json!([10.0, 20.0, 30.0]),
            tween: Some(1),
        })
        .expect("seed position kf0");
    session
        .execute(Command::SetKeyframe {
            target_id: clip_id.clone(),
            property: "position".to_owned(),
            frame: 60.0,
            value: serde_json::json!([10.0, 80.0, 90.0]),
            tween: Some(1),
        })
        .expect("seed position kf1");

    let mut state = GraphEditorState::new();
    state.focused_property = Some((clip_id.clone(), "position".to_owned()));
    state.follow_selection = false;
    session.playback.current_frame = 0.0;

    let tracks = collect_curve_tracks(&session, &state, true);
    assert_eq!(tracks.len(), 3, "Vec3 position must expand to .X/.Y/.Z");

    // 1. Paste shorthand scalar PanZoid keyframe array onto default active channel (.X = 0).
    let scalar_json = r#"[{"t":0,"v":0},{"t":30,"v":100}]"#;
    let mut action = GraphEditorAction::default();
    assert!(try_paste_pz_keyframes_json(
        &mut session,
        &mut state,
        &tracks,
        scalar_json,
        &mut action,
    ));
    assert!(action.curves_modified);

    let prop = &session.document().video_tracks[0].clips[0].properties["position"];
    assert_eq!(prop.keyframes.len(), 2);
    let kf0 = prop.keyframes[0]
        .value
        .as_array()
        .expect("kf0 must remain Vec3 array, not collapse to scalar");
    let kf1 = prop.keyframes[1]
        .value
        .as_array()
        .expect("kf1 must remain Vec3 array, not collapse to scalar");
    assert_eq!(kf0.len(), 3);
    assert_eq!(kf1.len(), 3);
    // X updated to 0 -> 100; Y interpolated 20 -> 50; Z interpolated 30 -> 60.
    assert!((kf0[0].as_f64().unwrap() - 0.0).abs() < 1e-6);
    assert!((kf0[1].as_f64().unwrap() - 20.0).abs() < 1e-6);
    assert!((kf0[2].as_f64().unwrap() - 30.0).abs() < 1e-6);
    assert!((kf1[0].as_f64().unwrap() - 100.0).abs() < 1e-6);
    assert!((kf1[1].as_f64().unwrap() - 50.0).abs() < 1e-6);
    assert!((kf1[2].as_f64().unwrap() - 60.0).abs() < 1e-6);

    // .X/.Y/.Z tracks must still exist intact AND Copy Curve JSON must succeed!
    let tracks_after = collect_curve_tracks(&session, &state, true);
    assert_eq!(
        tracks_after.len(),
        3,
        "Vec3 position must still expose all 3 (.X/.Y/.Z) channels after scalar paste"
    );
    let copied = session
        .copy_property_as_pz_json(&clip_id, "position")
        .expect("Copy Curve JSON must succeed seamlessly after scalar paste onto Vec3");
    assert!(copied.contains("\"keyframes\""));

    // 2. Select a key on .Y (channel_index = Some(1)) and paste another scalar curve:
    //    only .Y updates while .X and .Z remain intact!
    state.selected_keys = vec![GraphKeyId::new(&clip_id, "position", Some(1), 0.0)];
    let scalar_y_json = r#"[{"t":0,"v":5},{"t":30,"v":55}]"#;
    let mut action2 = GraphEditorAction::default();
    assert!(try_paste_pz_keyframes_json(
        &mut session,
        &mut state,
        &tracks_after,
        scalar_y_json,
        &mut action2,
    ));
    let prop2 = &session.document().video_tracks[0].clips[0].properties["position"];
    let y0 = prop2.keyframes[0].value.as_array().expect("Vec3");
    let y1 = prop2.keyframes[1].value.as_array().expect("Vec3");
    assert!((y0[0].as_f64().unwrap() - 0.0).abs() < 1e-6);
    assert!((y0[1].as_f64().unwrap() - 5.0).abs() < 1e-6);
    assert!((y0[2].as_f64().unwrap() - 30.0).abs() < 1e-6);
    assert!((y1[0].as_f64().unwrap() - 100.0).abs() < 1e-6);
    assert!((y1[1].as_f64().unwrap() - 55.0).abs() < 1e-6);
    assert!((y1[2].as_f64().unwrap() - 60.0).abs() < 1e-6);
}
