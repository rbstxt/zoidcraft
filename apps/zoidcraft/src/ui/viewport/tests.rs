//! Viewport unit tests.

use super::*;
use zc_session::SessionKeyframe;

#[test]
pub(crate) fn aspect_fit_canvas_rect_centers_16_9_in_wide_and_tall_viewports() {
    let wide = Rect::from_min_size(pos2(0.0, 0.0), vec2(1200.0, 540.0));
    let c = compute_canvas_rect(wide, 1920, 1080);
    assert!((c.width() - 960.0).abs() < 1.0);
    assert!((c.height() - 540.0).abs() < 1.0);
    assert!((c.left() - 120.0).abs() < 1.0);

    let tall = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 800.0));
    let c2 = compute_canvas_rect(tall, 1920, 1080);
    assert!((c2.width() - 640.0).abs() < 1.0);
    assert!((c2.height() - 360.0).abs() < 1.0);
    assert!((c2.top() - 220.0).abs() < 1.0);
}

#[test]
pub(crate) fn editor_camera_orbit_pan_dolly_and_projection() {
    let mut cam = EditorCameraState::new();
    let initial_dist = cam.distance();
    cam.orbit(40.0, -20.0, 1.0);
    assert!((cam.distance() - initial_dist).abs() < 1e-3);

    cam.dolly(-100.0, 1.0);
    assert!(cam.distance() < initial_dist);

    cam.pan(25.0, 15.0, 1.0);
    assert!(cam.target.length() > 0.0);

    cam.reset();
    let canvas = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
    let origin_screen = cam
        .project_world_point(Vec3::ZERO, canvas)
        .expect("origin projects in front of default camera");
    assert!((origin_screen.x - 480.0).abs() < 1.0);
    assert!((origin_screen.y - 270.0).abs() < 1.0);
}

#[test]
pub(crate) fn exponential_dolly_clamps_and_pan_is_world_scale() {
    let mut cam = EditorCameraState::new();
    let d0 = cam.distance();
    // Extreme scroll spike clamps factor to 0.5x rather than collapsing to near plane.
    cam.dolly(-10_000.0, 1.0);
    assert!((cam.distance() - d0 * 0.5).abs() < 1e-3);

    cam.reset();
    let canvas = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
    let before = cam
        .project_world_point(Vec3::ZERO, canvas)
        .expect("origin projects");
    cam.pan_viewport(40.0, -25.0, canvas.height(), 1.0);
    let after = cam
        .project_world_point(Vec3::ZERO, canvas)
        .expect("origin still projects");
    assert!((after.x - before.x - 40.0).abs() < 1.0);
    assert!((after.y - before.y - (-25.0)).abs() < 1.0);
}

#[test]
pub(crate) fn unwrapped_rotation_crosses_negative_x_without_360_jump() {
    let v_above = vec2(-100.0, -1.0);
    let v_below = vec2(-100.0, 1.0);
    let delta_ccw = unwrapped_rotation_delta(v_above, v_below).expect("non-degenerate");
    assert!(delta_ccw > 0.0 && delta_ccw < 0.05);

    let delta_cw = unwrapped_rotation_delta(v_below, v_above).expect("non-degenerate");
    assert!(delta_cw < 0.0 && delta_cw > -0.05);
}

#[test]
pub(crate) fn negative_scale_and_shift_vertical_proportional_scale_work() {
    // Negative X scale: dragging Right handle right makes |scale.x| larger (more negative),
    // without shrinking or flipping inside-out.
    let s = compute_scaled_2d(
        Transform2DHandle::Right,
        vec2(48.0, 0.0),
        GeomVec2::new(-1.0, 1.0),
        0.0,
        GeomVec2::ZERO,
        1920.0,
        1080.0,
        0.5,
        false,
    );
    assert!((s.x - (-1.1)).abs() < 1e-3);
    assert!((s.y - 1.0).abs() < 1e-3);

    // Shift + vertical drag on BottomRight corner scales both axes proportionally.
    let p = compute_scaled_2d(
        Transform2DHandle::BottomRight,
        vec2(0.0, 135.0),
        GeomVec2::new(1.0, 1.0),
        0.0,
        GeomVec2::ZERO,
        1920.0,
        1080.0,
        0.5,
        true,
    );
    assert!((p.x - 1.5).abs() < 1e-3);
    assert!((p.y - 1.5).abs() < 1e-3);
}

#[test]
pub(crate) fn eval_prop_scalar_interpolates_bezier_segments() {
    let bezier_tween = (1u16 << 8) | 1u16; // InterpolationMode::Bezier
    let prop = SessionProperty {
        animated: true,
        expression: None,
        default_value: json!(0.0),
        keyframes: vec![
            SessionKeyframe {
                frame: 0.0,
                value: json!(0.0),
                tween: 1,
                control_points: [[-10.0, 0.0], [10.0, 80.0]],
                continuous_tangent: true,
                value_kind: None,
            },
            SessionKeyframe {
                frame: 30.0,
                value: json!(100.0),
                tween: bezier_tween,
                control_points: [[-10.0, 0.0], [10.0, 0.0]],
                continuous_tangent: true,
                value_kind: None,
            },
        ],
    };
    let mid = eval_prop_scalar(&prop, 15.0, 0.0);
    // Steep positive outgoing tangent pushes midpoint significantly above linear 50.0.
    assert!(
        mid > 65.0 && mid < 95.0,
        "expected bezier curve > 65, got {mid}"
    );
}

#[test]
pub(crate) fn eval_prop_helpers_use_default_value_when_keyframes_empty() {
    let scalar_prop = SessionProperty {
        animated: false,
        expression: None,
        default_value: json!(42.5),
        keyframes: Vec::new(),
    };
    assert!((eval_prop_scalar(&scalar_prop, 10.0, 0.0) - 42.5).abs() < 1e-6);

    let vec2_prop = SessionProperty {
        animated: false,
        expression: None,
        default_value: json!([120.0, -45.0]),
        keyframes: Vec::new(),
    };
    let v2 = eval_prop_vec2(&vec2_prop, 10.0, GeomVec2::ZERO);
    assert!((v2.x - 120.0).abs() < 1e-4 && (v2.y - (-45.0)).abs() < 1e-4);

    let vec3_prop = SessionProperty {
        animated: false,
        expression: None,
        default_value: json!([10.0, 20.0, 30.0]),
        keyframes: Vec::new(),
    };
    let v3 = eval_prop_vec3(&vec3_prop, 10.0, Vec3::ZERO);
    assert!((v3.x - 10.0).abs() < 1e-4 && (v3.y - 20.0).abs() < 1e-4 && (v3.z - 30.0).abs() < 1e-4);
}

#[test]
pub(crate) fn double_click_resets_editing_camera_even_with_transform_2d_enabled() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session.selection.select_clip(clip_id, Some("scene"), false);
    session.playback.editing_camera = true;
    session.playback.transform_controls_2d = true;

    let mut state = ViewportState::new();
    state.camera.orbit(60.0, -25.0, 1.0);
    assert_ne!(state.camera, EditorCameraState::new());

    let click_pos = pos2(320.0, 180.0);
    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
    let mut step = |time: f64, events: Vec<egui::Event>| -> ViewportAction {
        let mut raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(time),
            events,
            ..Default::default()
        };
        raw.focused = true;
        let mut action = ViewportAction::default();
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    action = show_viewport(ui, &mut session, &mut state, None, &palette);
                });
        });
        full.textures_delta.clear();
        action
    };

    // Frame 0: initial layout so layer hit-test knows viewport layer.
    step(0.0, vec![egui::Event::PointerMoved(click_pos)]);
    // Click 1 press + release.
    step(
        0.05,
        vec![
            egui::Event::PointerMoved(click_pos),
            egui::Event::PointerButton {
                pos: click_pos,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    step(
        0.10,
        vec![
            egui::Event::PointerMoved(click_pos),
            egui::Event::PointerButton {
                pos: click_pos,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    // Click 2 press + release (double-click).
    step(
        0.15,
        vec![
            egui::Event::PointerMoved(click_pos),
            egui::Event::PointerButton {
                pos: click_pos,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let release2 = step(
        0.20,
        vec![
            egui::Event::PointerMoved(click_pos),
            egui::Event::PointerButton {
                pos: click_pos,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );

    assert!(
        release2.camera_changed,
        "double-click release must reset editing camera even when 2D transform overlay is active"
    );
    assert_eq!(state.camera, EditorCameraState::new());
    assert!(!session.history.is_in_transaction());
}

#[test]
pub(crate) fn exponential_dolly_has_zero_roundtrip_drift() {
    let mut cam = EditorCameraState::new();
    let d0 = cam.distance();
    for _ in 0..40 {
        cam.dolly(12.0, 1.0);
        cam.dolly(-12.0, 1.0);
    }
    assert!(
        (cam.distance() - d0).abs() < 1e-3,
        "back-and-forth exponential dolly must not drift: before={d0}, after={}",
        cam.distance()
    );
}

#[test]
pub(crate) fn viewport_2d_zoom_at_preserves_anchor_and_reset_restores_fit() {
    let panel = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 450.0));
    let fit = compute_canvas_rect(panel, 1920, 1080);
    let anchor = pos2(600.0, 150.0);

    let mut state = ViewportState::new();
    let norm_before = vec2(
        (anchor.x - fit.center().x) / fit.width(),
        (anchor.y - fit.center().y) / fit.height(),
    );
    state.zoom_2d_at(panel, anchor, 2.0);
    assert!((state.zoom_2d - 2.0).abs() < 1e-4);
    assert!(state.is_2d_view_transformed());

    let zoomed_rect =
        Rect::from_center_size(fit.center() + state.pan_2d, fit.size() * state.zoom_2d);
    let norm_after = vec2(
        (anchor.x - zoomed_rect.center().x) / zoomed_rect.width(),
        (anchor.y - zoomed_rect.center().y) / zoomed_rect.height(),
    );
    assert!(
        (norm_before - norm_after).length() < 1e-3,
        "anchor normalized canvas coords must stay fixed across 2D zoom: before={norm_before:?}, after={norm_after:?}"
    );

    state.reset_2d_view();
    assert!((state.zoom_2d - 1.0).abs() < 1e-6);
    assert_eq!(state.pan_2d, Vec2::ZERO);
    assert!(!state.is_2d_view_transformed());
}

#[test]
pub(crate) fn headless_viewport_2d_pinch_cmd_scroll_trackpad_pan_and_cmd0_reset() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    session.playback.editing_camera = false;
    let mut state = ViewportState::new();
    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
    let hover = pos2(440.0, 120.0);

    let step = |session: &mut Session,
                state: &mut ViewportState,
                time: f64,
                modifiers: egui::Modifiers,
                mut events: Vec<egui::Event>|
     -> ViewportAction {
        events.insert(0, egui::Event::ModifiersChanged(modifiers));
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(time),
            events,
            focused: true,
            ..Default::default()
        };
        let mut action = ViewportAction::default();
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    action = show_viewport(ui, session, state, None, &palette);
                });
        });
        full.textures_delta.clear();
        action
    };

    // Frame 0: prime layer hit-test.
    let a0 = step(
        &mut session,
        &mut state,
        0.0,
        egui::Modifiers::NONE,
        vec![egui::Event::PointerMoved(hover)],
    );
    let fit = a0.canvas_rect.expect("fit rect");

    // 1. Trackpad pinch zooms 2D canvas around hover anchor.
    let _ = step(
        &mut session,
        &mut state,
        0.016,
        egui::Modifiers::NONE,
        vec![egui::Event::PointerMoved(hover), egui::Event::Zoom(1.5)],
    );
    assert!((state.zoom_2d - 1.5).abs() < 1e-3);
    assert!(
        state.pan_2d.length() > 1.0,
        "off-center pinch must shift pan_2d to preserve anchor"
    );

    // 2. Cmd+Scroll zooms 2D canvas further (egui 0.37 folds into zoom_delta).
    let z_before_cmd = state.zoom_2d;
    let _ = step(
        &mut session,
        &mut state,
        0.032,
        egui::Modifiers::COMMAND,
        vec![
            egui::Event::PointerMoved(hover),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: vec2(0.0, 1.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Wheel,
                modifiers: egui::Modifiers::COMMAND,
            },
        ],
    );
    assert!(
        state.zoom_2d > z_before_cmd,
        "Cmd+Scroll must zoom 2D canvas in"
    );

    // 3. Two-finger trackpad scroll pans 2D canvas smoothly in X and Y.
    let pan_before = state.pan_2d;
    let z_before_pan = state.zoom_2d;
    let _ = step(
        &mut session,
        &mut state,
        0.050,
        egui::Modifiers::NONE,
        vec![
            egui::Event::PointerMoved(hover),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: vec2(18.0, -12.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Trackpad,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(
        (state.zoom_2d - z_before_pan).abs() < 1e-4,
        "trackpad pan must not zoom"
    );
    assert!((state.pan_2d.x - pan_before.x - 18.0).abs() < 1.0);
    assert!((state.pan_2d.y - pan_before.y - (-12.0)).abs() < 1.0);

    // 4. Cmd+0 resets 2D view back to fit.
    let a_reset = step(
        &mut session,
        &mut state,
        0.070,
        egui::Modifiers::COMMAND,
        vec![
            egui::Event::PointerMoved(hover),
            egui::Event::Key {
                key: egui::Key::Num0,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::COMMAND,
            },
        ],
    );
    assert!((state.zoom_2d - 1.0).abs() < 1e-5);
    assert_eq!(state.pan_2d, Vec2::ZERO);
    assert_eq!(a_reset.canvas_rect, Some(fit));
}

#[test]
pub(crate) fn headless_viewport_3d_editing_camera_pinch_modifier_wheel_and_trackpad_gestures() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session.selection.select_clip(clip_id, Some("scene"), false);
    session.playback.editing_camera = true;

    let mut state = ViewportState::new();
    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
    let hover = pos2(320.0, 180.0);

    let step = |session: &mut Session,
                state: &mut ViewportState,
                time: f64,
                modifiers: egui::Modifiers,
                mut events: Vec<egui::Event>|
     -> ViewportAction {
        events.insert(0, egui::Event::ModifiersChanged(modifiers));
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(time),
            events,
            focused: true,
            ..Default::default()
        };
        let mut action = ViewportAction::default();
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    action = show_viewport(ui, session, state, None, &palette);
                });
        });
        full.textures_delta.clear();
        action
    };

    // Prime layout.
    let _ = step(
        &mut session,
        &mut state,
        0.0,
        egui::Modifiers::NONE,
        vec![egui::Event::PointerMoved(hover)],
    );

    // 1. Trackpad pinch dollies camera closer.
    let d0 = state.camera.distance();
    let a_pinch = step(
        &mut session,
        &mut state,
        0.016,
        egui::Modifiers::NONE,
        vec![egui::Event::PointerMoved(hover), egui::Event::Zoom(1.25)],
    );
    assert!(a_pinch.camera_changed);
    assert!(state.camera.distance() < d0);

    // 2. Cmd+Wheel fine dolly (was dead code in egui 0.37).
    let d1 = state.camera.distance();
    let a_cmd = step(
        &mut session,
        &mut state,
        0.032,
        egui::Modifiers::COMMAND,
        vec![
            egui::Event::PointerMoved(hover),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: vec2(0.0, 1.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Wheel,
                modifiers: egui::Modifiers::COMMAND,
            },
        ],
    );
    assert!(a_cmd.camera_changed);
    assert!(state.camera.distance() < d1);

    // 3. Shift+Wheel fast dolly (was dead code in egui 0.37 because Shift shifts Y -> X).
    let d2 = state.camera.distance();
    let a_shift = step(
        &mut session,
        &mut state,
        0.35,
        egui::Modifiers::SHIFT,
        vec![
            egui::Event::PointerMoved(hover),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: vec2(0.0, 1.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Wheel,
                modifiers: egui::Modifiers::SHIFT,
            },
        ],
    );
    assert!(a_shift.camera_changed);
    assert!(state.camera.distance() < d2);

    // 4. Two-finger trackpad swipe orbits camera while preserving distance.
    let pos_before_orbit = state.camera.position;
    let dist_before_orbit = state.camera.distance();
    let a_orbit = step(
        &mut session,
        &mut state,
        0.65,
        egui::Modifiers::NONE,
        vec![
            egui::Event::PointerMoved(hover),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: vec2(20.0, -12.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Trackpad,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(a_orbit.camera_changed);
    assert!((state.camera.distance() - dist_before_orbit).abs() < 1e-3);
    assert!((state.camera.position - pos_before_orbit).length() > 0.1);

    // 5. Shift + two-finger trackpad pans both X and Y (un-remapped raw wheel points).
    let target_before_pan = state.camera.target;
    let a_pan = step(
        &mut session,
        &mut state,
        0.68,
        egui::Modifiers::SHIFT,
        vec![
            egui::Event::PointerMoved(hover),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: vec2(15.0, 15.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Trackpad,
                modifiers: egui::Modifiers::SHIFT,
            },
        ],
    );
    assert!(a_pan.camera_changed);
    let target_delta = state.camera.target - target_before_pan;
    assert!(target_delta.length() > 0.1);
    assert!(
        target_delta.y.abs() > 1e-3,
        "Shift+Trackpad pan must preserve vertical Y delta (not folded onto X)"
    );
}

pub(crate) fn test_static_prop(v: Value) -> SessionProperty {
    SessionProperty {
        animated: false,
        expression: None,
        default_value: v,
        keyframes: Vec::new(),
    }
}

#[test]
pub(crate) fn resolved_camera_3d_handles_camera_plus_focal_length_film_gate_and_offset() {
    let mut cam_props = BTreeMap::new();
    cam_props.insert("enabled".to_owned(), test_static_prop(json!(1)));
    cam_props.insert("active".to_owned(), test_static_prop(json!(1)));
    cam_props.insert(
        "position".to_owned(),
        test_static_prop(json!([0.0, 0.0, 100.0])),
    );
    cam_props.insert(
        "rotation".to_owned(),
        test_static_prop(json!([0.0, 0.0, 0.0])),
    );
    cam_props.insert("scale".to_owned(), test_static_prop(json!([1.0, 1.0, 1.0])));
    cam_props.insert("focalLength".to_owned(), test_static_prop(json!(50.0)));
    cam_props.insert("filmGate".to_owned(), test_static_prop(json!(36.0)));
    cam_props.insert("filmOffsetX".to_owned(), test_static_prop(json!(25.0)));
    cam_props.insert("filmOffsetY".to_owned(), test_static_prop(json!(10.0)));
    cam_props.insert("zoom".to_owned(), test_static_prop(json!(1.0)));

    let cam_obj = SessionObject3D {
        id: "cam-plus-1".to_owned(),
        parent_id: None,
        kind: "zoidium:camera-plus/camera".to_owned(),
        name: "Camera+".to_owned(),
        source_id: None,
        properties: cam_props.clone(),
    };

    // Also include a legacy snake_case camera to verify override & kind detection.
    let mut cam2_props = cam_props;
    cam2_props.insert("filmOffsetX".to_owned(), test_static_prop(json!(0.0)));
    cam2_props.insert("filmOffsetY".to_owned(), test_static_prop(json!(0.0)));
    cam2_props.insert("active".to_owned(), test_static_prop(json!(0)));
    let cam_obj2 = SessionObject3D {
        id: "cam-legacy-2".to_owned(),
        parent_id: None,
        kind: "zoidium:camera_plus/camera".to_owned(),
        name: "Cam 2".to_owned(),
        source_id: None,
        properties: cam2_props,
    };

    let clip = SessionClip {
        id: "clip-scene-1".to_owned(),
        kind: "scene".to_owned(),
        name: "Scene".to_owned(),
        start: 0.0,
        length: 120.0,
        time_start: 0.0,
        time_end: 120.0,
        time_animated: false,
        media_length: None,
        source_clip_id: None,
        properties: BTreeMap::new(),
        objects_3d: vec![cam_obj, cam_obj2],
        effects: Vec::new(),
    };

    let canvas = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
    let cam_offset = ResolvedCamera3D::from_scene_clip(&clip, 0.0, 1920.0, 1080.0);
    let cam_centered = ResolvedCamera3D::from_scene_clip_with_override(
        &clip,
        0.0,
        1920.0,
        1080.0,
        Some("cam-legacy-2"),
    );

    assert_eq!(cam_offset.camera_object_id.as_deref(), Some("cam-plus-1"));
    assert_eq!(
        cam_centered.camera_object_id.as_deref(),
        Some("cam-legacy-2")
    );
    // 50mm lens on 36mm horizontal gate at 16:9 has ~22.9 deg vertical FOV (not default 60 deg).
    assert!(
        (cam_offset.fov_deg - 22.9).abs() < 1.0,
        "expected ~22.9 deg vertical FOV for 50mm/36mm 16:9, got {}",
        cam_offset.fov_deg
    );

    let p_centered = cam_centered
        .project_world_point(Vec3::ZERO, canvas)
        .expect("origin projects");
    let p_shifted = cam_offset
        .project_world_point(Vec3::ZERO, canvas)
        .expect("origin projects with film offset");
    assert!((p_centered.x - 480.0).abs() < 1.0);
    assert!((p_centered.y - 270.0).abs() < 1.0);
    assert!(
        (p_shifted.x - p_centered.x).abs() > 20.0,
        "filmOffsetX must shift projected origin horizontally"
    );
    assert!(
        (p_shifted.y - p_centered.y).abs() > 10.0,
        "filmOffsetY must shift projected origin vertically"
    );
}

#[test]
pub(crate) fn headless_gizmo_3d_translate_rotate_scale_and_raycast_pick() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session
        .selection
        .select_clip(clip_id.clone(), Some("scene"), false);
    session.playback.editing_camera = true;

    // Add a 3D box at the origin.
    let add_res = session
        .execute(Command::AddObject3D {
            clip_id: clip_id.clone(),
            parent_id: None,
            kind: "box".to_owned(),
            name: Some("GizmoBox".to_owned()),
        })
        .expect("add 3d box");
    let obj_id = add_res
        .get("object_id")
        .and_then(Value::as_str)
        .expect("object_id")
        .to_owned();
    let _ = session.execute(Command::SetPropertyValue {
        target_id: obj_id.clone(),
        property: "position".to_owned(),
        value: json!([0.0, 0.0, 0.0]),
    });
    let _ = session.execute(Command::SetPropertyValue {
        target_id: obj_id.clone(),
        property: "scale".to_owned(),
        value: json!([1.0, 1.0, 1.0]),
    });

    let mut state = ViewportState::new();
    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
    let step = |session: &mut Session,
                state: &mut ViewportState,
                time: f64,
                events: Vec<egui::Event>|
     -> ViewportAction {
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(time),
            events,
            focused: true,
            ..Default::default()
        };
        let mut action = ViewportAction::default();
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    action = show_viewport(ui, session, state, None, &palette);
                });
        });
        full.textures_delta.clear();
        action
    };

    // 1. Test 3D Raycast Picking at origin projection.
    let cam_res = ResolvedCamera3D::from_editor_camera(&state.camera, 640.0 / 360.0);
    let origin_screen = cam_res
        .project_world_point(Vec3::ZERO, screen_rect)
        .expect("origin projects");
    let picked =
        raycast_pick_3d_object(&session, &state, origin_screen, screen_rect, 1920.0, 1080.0);
    assert!(
        picked.is_some(),
        "raycast at box screen center must hit a 3D object"
    );

    // 2. Select the box and drag the +X Translate gizmo shaft rightwards.
    let _ = session
        .selection
        .select_object_3d(obj_id.clone(), "box", false);
    state.gizmo_3d_mode = Gizmo3DMode::Translate;
    state.gizmo_3d_space = Gizmo3DSpace::World;

    let world_len =
        (GIZMO_SCREEN_LEN_PX * cam_res.world_units_per_px(Vec3::ZERO, screen_rect)).max(0.05);
    let x_mid = cam_res
        .project_world_point(Vec3::X * (world_len * 0.65), screen_rect)
        .expect("X shaft projects");
    let x_drag_end = x_mid + vec2(40.0, 0.0);

    // Prime layout.
    step(
        &mut session,
        &mut state,
        0.0,
        vec![egui::Event::PointerMoved(x_mid)],
    );
    // Press on X axis shaft.
    step(
        &mut session,
        &mut state,
        0.016,
        vec![
            egui::Event::PointerMoved(x_mid),
            egui::Event::PointerButton {
                pos: x_mid,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    // Drag right.
    let drag_act = step(
        &mut session,
        &mut state,
        0.032,
        vec![egui::Event::PointerMoved(x_drag_end)],
    );
    // Release.
    step(
        &mut session,
        &mut state,
        0.048,
        vec![
            egui::Event::PointerMoved(x_drag_end),
            egui::Event::PointerButton {
                pos: x_drag_end,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(drag_act.transform_modified);
    assert!(!session.history.is_in_transaction());

    let moved_pos = session.document().video_tracks[0].clips[0]
        .objects_3d
        .iter()
        .find(|o| o.id == obj_id)
        .and_then(|o| o.properties.get("position"))
        .map(|p| eval_prop_vec3(p, 0.0, Vec3::ZERO))
        .expect("box position");
    assert!(
        moved_pos.x > 5.0 && moved_pos.y.abs() < 1e-3 && moved_pos.z.abs() < 1e-3,
        "X translate gizmo must move box along +X only, got {moved_pos:?}"
    );

    // 3. Verify Rotate & Scale gizmo pure math helpers.
    let rot_after = compute_gizmo_3d_rotate(
        Gizmo3DHandle::AxisY,
        Gizmo3DSpace::Local,
        Vec3::NEG_Z,
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Mat4::IDENTITY,
        Vec3::ZERO,
        GeomEulerOrder::YXZ,
        0.5,
        false,
    );
    assert!(
        rot_after.y.abs() > 0.4,
        "Y rotation gizmo must rotate around Y, got {rot_after:?}"
    );

    let scale_after = compute_gizmo_3d_scale(
        &cam_res,
        screen_rect,
        Gizmo3DHandle::AxisX,
        Vec3::ZERO,
        origin_screen,
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::ONE,
        x_mid,
        x_mid + vec2(36.0, 0.0),
        false,
    );
    assert!(
        scale_after.x > 1.3 && (scale_after.y - 1.0).abs() < 1e-4,
        "X scale gizmo must scale X only, got {scale_after:?}"
    );
}

#[test]
pub(crate) fn headless_shape_plus_pen_tool_click_drag_close_commits_valid_payload() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let track_id = session.document().video_tracks[0].id.clone();

    // Add a Shape+ clip.
    let res = session
        .execute(Command::AddClip {
            track_id,
            kind: "zoidium:shape-plus/shape-layer".to_owned(),
            name: Some("Vector Layer".to_owned()),
            start: 0.0,
            length: 120.0,
        })
        .expect("add shape-plus clip");
    let clip_id = res
        .get("clip_id")
        .and_then(Value::as_str)
        .expect("clip_id")
        .to_owned();
    let _ = session.selection.select_clip(
        clip_id.clone(),
        Some("zoidium:shape-plus/shape-layer"),
        false,
    );
    session.playback.editing_camera = false;

    let mut state = ViewportState::new();
    state.active_draw_tool = Some(0); // Pen tool

    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
    let step = |session: &mut Session,
                state: &mut ViewportState,
                time: f64,
                events: Vec<egui::Event>|
     -> ViewportAction {
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(time),
            events,
            focused: true,
            ..Default::default()
        };
        let mut action = ViewportAction::default();
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    action = show_viewport(ui, session, state, None, &palette);
                });
        });
        full.textures_delta.clear();
        action
    };

    let p0 = pos2(220.0, 140.0);
    let p1 = pos2(420.0, 140.0);
    let p1_handle = pos2(450.0, 170.0);
    let p2 = pos2(320.0, 260.0);

    // Prime layout.
    step(
        &mut session,
        &mut state,
        0.0,
        vec![egui::Event::PointerMoved(p0)],
    );

    // Vertex 0: sharp click.
    step(
        &mut session,
        &mut state,
        0.02,
        vec![
            egui::Event::PointerMoved(p0),
            egui::Event::PointerButton {
                pos: p0,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert_eq!(
        state.pen_draft.as_ref().map(|d| d.vertices.len()),
        Some(1),
        "after p0 press"
    );
    step(
        &mut session,
        &mut state,
        0.04,
        vec![
            egui::Event::PointerMoved(p0),
            egui::Event::PointerButton {
                pos: p0,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );

    // Vertex 1: click + drag to pull symmetric Bezier handles.
    step(
        &mut session,
        &mut state,
        0.06,
        vec![
            egui::Event::PointerMoved(p1),
            egui::Event::PointerButton {
                pos: p1,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    step(
        &mut session,
        &mut state,
        0.08,
        vec![egui::Event::PointerMoved(p1_handle)],
    );
    step(
        &mut session,
        &mut state,
        0.10,
        vec![
            egui::Event::PointerMoved(p1_handle),
            egui::Event::PointerButton {
                pos: p1_handle,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert_eq!(
        state.pen_draft.as_ref().map(|d| d.vertices.len()),
        Some(2),
        "after p1 drag"
    );

    // Vertex 2: sharp click.
    step(
        &mut session,
        &mut state,
        0.12,
        vec![
            egui::Event::PointerMoved(p2),
            egui::Event::PointerButton {
                pos: p2,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    step(
        &mut session,
        &mut state,
        0.14,
        vec![
            egui::Event::PointerMoved(p2),
            egui::Event::PointerButton {
                pos: p2,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert_eq!(
        state.pen_draft.as_ref().map(|d| d.vertices.len()),
        Some(3),
        "after p2 click"
    );

    // Close contour by clicking p0 again.
    let close_act = step(
        &mut session,
        &mut state,
        0.16,
        vec![
            egui::Event::PointerMoved(p0),
            egui::Event::PointerButton {
                pos: p0,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(
        close_act.transform_modified,
        "expected close_act.transform_modified=true; pen_draft={:?}, is_sp={}, clip_tf_exists={}, vc_kind={:?}",
        state.pen_draft,
        is_clip_shape_plus(&session, &clip_id),
        active_single_video_clip_transform(&session).is_some(),
        find_project_video_clip(&session, &clip_id).map(|v| &v.layer.kind)
    );
    assert!(state.pen_draft.is_none());

    // Inspect committed Shape+ payload and validate against canonical Shape+ validator!
    let vc = find_project_video_clip(&session, &clip_id).expect("shape plus video clip");
    let LayerKind::Plugin { payload, .. } = &vc.layer.kind else {
        panic!("expected Plugin layer kind");
    };
    assert!(
        zc_render::shape_plus::validate_shape_plus_payload(payload).is_ok(),
        "committed Shape+ payload must pass canonical validator: {payload}"
    );
    let contours = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
    assert!(
        contours.iter().any(|c| c.closed
            && c.vertices.len() == 3
            && c.vertices[1].out_h[0].hypot(c.vertices[1].out_h[1]) > 5.0),
        "expected closed 3-vertex Shape+ path with curved vertex 1 handles"
    );
}

#[test]
pub(crate) fn layer_mask_vertex_drag_and_de_casteljau_split_emit_update_layer_mask() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let clip_uuid = Uuid::parse_str(&clip_id).expect("clip uuid");
    let _ = session
        .selection
        .select_clip(clip_id.clone(), Some("scene"), false);
    session.playback.editing_camera = false;

    // Add a rectangular LayerMask in [0..1920, 0..1080] mask space.
    let mask = LayerMask {
        id: "mask-test-1".to_owned(),
        name: "Mask 1".to_owned(),
        enabled: true,
        mode: MaskMode::Add,
        invert: false,
        feather: 0.0,
        expansion: 0.0,
        opacity: 1.0,
        closed: true,
        locked: false,
        outline_color: "#58a6ff".to_owned(),
        vertex_ids: vec![
            "v1".to_owned(),
            "v2".to_owned(),
            "v3".to_owned(),
            "v4".to_owned(),
        ],
        vertex_modes: BTreeMap::new(),
        path: PropertyTrack::constant(PropertyValue::Json(json!([
            480.0, 270.0, 0.0, 0.0, 0.0, 0.0, 1440.0, 270.0, 0.0, 0.0, 0.0, 0.0, 1440.0, 810.0,
            0.0, 0.0, 0.0, 0.0, 480.0, 810.0, 0.0, 0.0, 0.0, 0.0
        ]))),
        preserved: BTreeMap::new(),
    };
    session
        .execute(Command::AddLayerMask {
            clip_id: clip_uuid,
            mask,
        })
        .expect("add layer mask");

    let mut state = ViewportState::new();
    state.active_draw_tool = Some(1); // Bezier/Vertex Edit tool

    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
    let step = |session: &mut Session,
                state: &mut ViewportState,
                time: f64,
                events: Vec<egui::Event>|
     -> ViewportAction {
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(time),
            events,
            focused: true,
            ..Default::default()
        };
        let mut action = ViewportAction::default();
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    action = show_viewport(ui, session, state, None, &palette);
                });
        });
        full.textures_delta.clear();
        action
    };

    // Screen mapping: 640x360 = 1/3 scale of 1920x1080.
    // v1 = (480, 270) -> screen (160, 90); top segment midpoint -> screen (320, 90).
    let v1_s = pos2(160.0, 90.0);
    let v1_drag_to = pos2(190.0, 105.0);
    let top_mid_s = pos2(335.0, 97.5);

    step(
        &mut session,
        &mut state,
        0.0,
        vec![egui::Event::PointerMoved(v1_s)],
    );
    // 1. Drag vertex 0.
    step(
        &mut session,
        &mut state,
        0.02,
        vec![
            egui::Event::PointerMoved(v1_s),
            egui::Event::PointerButton {
                pos: v1_s,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let drag_act = step(
        &mut session,
        &mut state,
        0.04,
        vec![egui::Event::PointerMoved(v1_drag_to)],
    );
    step(
        &mut session,
        &mut state,
        0.06,
        vec![
            egui::Event::PointerMoved(v1_drag_to),
            egui::Event::PointerButton {
                pos: v1_drag_to,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(drag_act.transform_modified);
    let after_drag = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
    assert!((after_drag[0].vertices[0].point[0] - 570.0).abs() < 3.0);
    assert!((after_drag[0].vertices[0].point[1] - 315.0).abs() < 3.0);

    // 2. Click on the top Bezier segment to split it via De Casteljau subdivision.
    let split_act = step(
        &mut session,
        &mut state,
        0.10,
        vec![
            egui::Event::PointerMoved(top_mid_s),
            egui::Event::PointerButton {
                pos: top_mid_s,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    step(
        &mut session,
        &mut state,
        0.12,
        vec![
            egui::Event::PointerMoved(top_mid_s),
            egui::Event::PointerButton {
                pos: top_mid_s,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(split_act.transform_modified);
    let after_split = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
    assert_eq!(
        after_split[0].vertices.len(),
        5,
        "De Casteljau split on segment must insert 1 vertex (4 -> 5)"
    );

    // Verify exact De Casteljau midpoint preservation on a curved cubic.
    let a = BezierVertex2D {
        point: [0.0, 0.0],
        in_h: [0.0, 0.0],
        out_h: [40.0, 120.0],
    };
    let b = BezierVertex2D {
        point: [200.0, 0.0],
        in_h: [-40.0, 120.0],
        out_h: [0.0, 0.0],
    };
    let orig_mid = eval_cubic_point_2d(&a, &b, 0.5);
    let (a2, mid, b2) = subdivide_cubic_bezier(&a, &b, 0.5);
    assert!((mid.point[0] - orig_mid[0]).abs() < 1e-6);
    assert!((mid.point[1] - orig_mid[1]).abs() < 1e-6);
    let orig_quarter = eval_cubic_point_2d(&a, &b, 0.25);
    let left_half = eval_cubic_point_2d(&a2, &mid, 0.5);
    assert!((left_half[0] - orig_quarter[0]).abs() < 1e-6);
    assert!((left_half[1] - orig_quarter[1]).abs() < 1e-6);
    let orig_three_quarter = eval_cubic_point_2d(&a, &b, 0.75);
    let right_half = eval_cubic_point_2d(&mid, &b2, 0.5);
    assert!((right_half[0] - orig_three_quarter[0]).abs() < 1e-6);
    assert!((right_half[1] - orig_three_quarter[1]).abs() < 1e-6);
}

#[test]
pub(crate) fn anchor_drag_2d_and_canvas_snapping_math() {
    // 1. Anchor drag with position compensation keeps any layer point visually stationary.
    let init_anchor = GeomVec2::new(10.0, -20.0);
    let init_pos = GeomVec2::new(50.0, 30.0);
    let scale = GeomVec2::new(2.0, 0.5);
    let rot = 0.4_f32;
    let world_delta = GeomVec2::new(40.0, -15.0);
    let (new_anchor, new_pos) =
        compute_anchor_drag_2d(init_anchor, init_pos, scale, rot, world_delta, true);

    let eval_world = |q: GeomVec2, pos: GeomVec2, anchor: GeomVec2| -> GeomVec2 {
        let ds = GeomVec2::new((q.x - anchor.x) * scale.x, (q.y - anchor.y) * scale.y);
        let (sr, cr) = rot.sin_cos();
        GeomVec2::new(pos.x + ds.x * cr - ds.y * sr, pos.y + ds.x * sr + ds.y * cr)
    };
    let probe = GeomVec2::new(123.0, -45.0);
    let before = eval_world(probe, init_pos, init_anchor);
    let after = eval_world(probe, new_pos, new_anchor);
    assert!(
        (before.x - after.x).abs() < 1e-3 && (before.y - after.y).abs() < 1e-3,
        "compensated anchor drag must keep layer pixels stationary: before={before:?}, after={after:?}"
    );

    // 2. Canvas snapping to center (0, 0) and Rule of Thirds (seq_w/6 = 320, seq_h/6 = 180).
    let snap_center = snap_2d_position(
        GeomVec2::new(4.0, -3.5),
        GeomVec2::new(0.5, 0.5),
        GeomVec2::ZERO,
        0.0,
        1920.0,
        1080.0,
        10.0,
    );
    assert!((snap_center.snapped_pos.x - 0.0).abs() < 1e-4);
    assert!((snap_center.snapped_pos.y - 0.0).abs() < 1e-4);
    assert_eq!(snap_center.guide_x, Some(0.0));
    assert_eq!(snap_center.guide_y, Some(0.0));

    let snap_thirds = snap_2d_position(
        GeomVec2::new(324.0, -176.0),
        GeomVec2::new(0.25, 0.25),
        GeomVec2::ZERO,
        0.0,
        1920.0,
        1080.0,
        10.0,
    );
    assert!((snap_thirds.snapped_pos.x - 320.0).abs() < 1e-4);
    assert!((snap_thirds.snapped_pos.y - (-180.0)).abs() < 1e-4);
    assert_eq!(snap_thirds.guide_x, Some(320.0));
    assert_eq!(snap_thirds.guide_y, Some(-180.0));
}

#[test]
pub(crate) fn safe_guides_geometry_computes_exact_action_title_thirds_and_center() {
    let canvas = Rect::from_min_size(pos2(100.0, 50.0), vec2(960.0, 540.0));
    let g = compute_safe_guides(canvas);
    assert!((g.action_safe.width() - 960.0 * 0.90).abs() < 1e-3);
    assert!((g.action_safe.height() - 540.0 * 0.90).abs() < 1e-3);
    assert!((g.title_safe.width() - 960.0 * 0.80).abs() < 1e-3);
    assert!((g.title_safe.height() - 540.0 * 0.80).abs() < 1e-3);
    assert!((g.center.x - 580.0).abs() < 1e-3);
    assert!((g.center.y - 320.0).abs() < 1e-3);
    assert!((g.thirds_x[0] - (100.0 + 320.0)).abs() < 1e-3);
    assert!((g.thirds_x[1] - (100.0 + 640.0)).abs() < 1e-3);
    assert!((g.thirds_y[0] - (50.0 + 180.0)).abs() < 1e-3);
    assert!((g.thirds_y[1] - (50.0 + 360.0)).abs() < 1e-3);
}

#[test]
pub(crate) fn gizmo_3d_and_wireframes_hidden_when_editing_camera_is_off() {
    let mut session = Session::new();
    let vtrack = session.document().video_tracks[0].id.clone();
    let clip_id = session
        .execute(Command::AddClip {
            track_id: vtrack,
            kind: "scene".to_owned(),
            name: Some("Scene".to_owned()),
            start: 0.0,
            length: 120.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let text_id = session
        .execute(Command::AddObject3D {
            clip_id: clip_id.clone(),
            kind: "Text3D".to_owned(),
            name: Some("Text".to_owned()),
            parent_id: None,
        })
        .unwrap()["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let _ = session.selection.select_clip(clip_id, Some("scene"), false);
    let _ = session
        .selection
        .select_object_3d(text_id.clone(), "Text3D", false);

    // Editing Camera OFF (clean final preview mode): 3D gizmos must NOT appear or respond to drag.
    session.playback.editing_camera = false;
    let ctx = egui::Context::default();
    let mut vp = ViewportState::new();
    let palette = ThemePalette::default();
    let panel_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 450.0));
    let mut raw = egui::RawInput {
        screen_rect: Some(panel_rect),
        ..Default::default()
    };
    let mut out_off = ctx.run_ui(raw.clone(), |ui| {
        let _ = show_viewport(ui, &mut session, &mut vp, None, &palette);
    });
    out_off.textures_delta.clear();
    let has_gizmo_axis_label_off = out_off.shapes.iter().any(|cs| {
        matches!(&cs.shape, Shape::Text(ts) if matches!(ts.galley.text(), "X" | "Y" | "Z" | "W" | "World"))
    });
    assert!(
        !has_gizmo_axis_label_off,
        "3D gizmo arrows and W/World HUD must NOT render when Editing Camera is OFF"
    );

    // Editing Camera ON: 3D gizmo arrows and W/World HUD MUST render.
    session.playback.editing_camera = true;
    raw.time = Some(0.1);
    let mut out_on = ctx.run_ui(raw, |ui| {
        let _ = show_viewport(ui, &mut session, &mut vp, None, &palette);
    });
    out_on.textures_delta.clear();
    let has_gizmo_axis_label_on = out_on.shapes.iter().any(|cs| {
        matches!(&cs.shape, Shape::Text(ts) if matches!(ts.galley.text(), "X" | "Y" | "Z" | "W"))
    });
    assert!(
        has_gizmo_axis_label_on,
        "3D gizmo arrows and W HUD must render when Editing Camera is ON"
    );
}

#[test]
pub(crate) fn editor_cam_dropdown_opens_on_primary_click_without_double_toggle() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session.selection.select_clip(clip_id, Some("scene"), false);
    session.playback.editing_camera = true;

    let mut state = ViewportState::new();
    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 450.0));
    // Camera Switcher button sits at x = 8 + 72 + 1 + 46 + 1 = 128..242, y = 8..30.
    let cam_btn = pos2(180.0, 19.0);

    let run =
        |session: &mut Session, state: &mut ViewportState, time: f64, events: Vec<egui::Event>| {
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
                        let _ = show_viewport(ui, session, state, None, &palette);
                    });
            });
            full.textures_delta.clear();
            full.shapes
        };

    let _ = run(
        &mut session,
        &mut state,
        0.0,
        vec![egui::Event::PointerMoved(cam_btn)],
    );
    let _ = run(
        &mut session,
        &mut state,
        0.02,
        vec![
            egui::Event::PointerMoved(cam_btn),
            egui::Event::PointerButton {
                pos: cam_btn,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let _ = run(
        &mut session,
        &mut state,
        0.04,
        vec![
            egui::Event::PointerMoved(cam_btn),
            egui::Event::PointerButton {
                pos: cam_btn,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    // egui::Popup runs an invisible sizing pass on the opening frame and paints visibly on the next frame.
    let shapes = run(
        &mut session,
        &mut state,
        0.06,
        vec![egui::Event::PointerMoved(cam_btn)],
    );
    let has_menu_item = shapes.iter().any(|cs| {
        matches!(&cs.shape, Shape::Text(ts) if ts.galley.text().contains("Active Scene Camera"))
    });
    assert!(
        has_menu_item,
        "Clicking Editor Cam dropdown must open Camera Switcher menu"
    );
}

#[test]
pub(crate) fn force_refresh_and_fps_overlay_and_safe_guides_individual_toggles() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session.selection.select_clip(clip_id, Some("scene"), false);
    session.playback.editing_camera = true;

    let mut state = ViewportState::new();
    assert!(state.show_fps_overlay, "show_fps_overlay defaults true");

    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 450.0));
    let run = |session: &mut Session, state: &mut ViewportState, time: f64| {
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(time),
            focused: true,
            ..Default::default()
        };
        let mut action = ViewportAction::default();
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    action = show_viewport(ui, session, state, None, &palette);
                });
        });
        full.textures_delta.clear();
        (action, full.shapes)
    };

    let (_, shapes_fps_on) = run(&mut session, &mut state, 0.0);
    assert!(
        shapes_fps_on
            .iter()
            .any(|cs| matches!(&cs.shape, Shape::Text(ts) if ts.galley.text().contains("fps"))),
        "FPS badge renders when show_fps_overlay is true"
    );

    // Toggle Editing Camera OFF and hide FPS overlay.
    session.playback.editing_camera = false;
    state.show_fps_overlay = false;
    let (act_toggle, shapes_fps_off) = run(&mut session, &mut state, 0.05);
    assert!(
        act_toggle.force_refresh,
        "Toggling Editing Camera must set action.force_refresh = true"
    );
    assert!(
        !shapes_fps_off
            .iter()
            .any(|cs| matches!(&cs.shape, Shape::Text(ts) if ts.galley.text().contains("fps"))),
        "FPS badge must hide when show_fps_overlay is false"
    );

    // Individual Action Safe (90%) guide renders without Title Safe (80%) and label sits at bottom-left.
    state.show_safe_guides = true;
    state.safe_guides = SafeGuideOptions {
        action_safe: true,
        title_safe: false,
        rule_of_thirds: false,
        center_crosshair: false,
    };
    let (_, shapes_guides) = run(&mut session, &mut state, 0.10);
    let mut found_90 = false;
    let mut found_80 = false;
    for cs in &shapes_guides {
        if let Shape::Text(ts) = &cs.shape {
            if ts.galley.text() == "90%" {
                found_90 = true;
                assert!(
                    ts.pos.y > 100.0,
                    "90% Action Safe label must sit clear of the top-left HUD bar (y={})",
                    ts.pos.y
                );
            }
            if ts.galley.text() == "80%" {
                found_80 = true;
            }
        }
    }
    assert!(
        found_90,
        "Action Safe 90% guide label must render when enabled"
    );
    assert!(
        !found_80,
        "Title Safe 80% guide label must stay hidden when individually disabled"
    );
}

#[test]
pub(crate) fn freehand_mask_auto_closes_and_coalesced_vertex_drag_updates_mask() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session
        .selection
        .select_clip(clip_id.clone(), Some("scene"), false);
    session.playback.editing_camera = false;

    let mut state = ViewportState::new();
    state.active_draw_tool = Some(2); // Freehand / Quick Shape tool
    state.quick_shape_kind = QuickShapeKind::Freehand;

    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
    let step = |session: &mut Session,
                state: &mut ViewportState,
                time: f64,
                events: Vec<egui::Event>|
     -> ViewportAction {
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(time),
            events,
            focused: true,
            ..Default::default()
        };
        let mut action = ViewportAction::default();
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    action = show_viewport(ui, session, state, None, &palette);
                });
        });
        full.textures_delta.clear();
        action
    };

    // Draw an OPEN freehand stroke on a non-Shape+ layer (endpoints far apart).
    let p0 = pos2(150.0, 120.0);
    let p1 = pos2(260.0, 120.0);
    let p2 = pos2(260.0, 220.0);
    step(
        &mut session,
        &mut state,
        0.0,
        vec![egui::Event::PointerMoved(p0)],
    );
    step(
        &mut session,
        &mut state,
        0.02,
        vec![
            egui::Event::PointerMoved(p0),
            egui::Event::PointerButton {
                pos: p0,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    step(
        &mut session,
        &mut state,
        0.04,
        vec![egui::Event::PointerMoved(p1)],
    );
    step(
        &mut session,
        &mut state,
        0.06,
        vec![egui::Event::PointerMoved(p2)],
    );
    let commit_act = step(
        &mut session,
        &mut state,
        0.08,
        vec![
            egui::Event::PointerMoved(p2),
            egui::Event::PointerButton {
                pos: p2,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(commit_act.transform_modified);
    let contours = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
    assert_eq!(contours.len(), 1);
    assert!(
        contours[0].closed,
        "Freehand Layer Mask must auto-close on commit so it forms a valid filled mask"
    );

    // Switch to Bezier vertex edit tool and perform a coalesced press+move drag on vertex 0.
    state.active_draw_tool = Some(1);
    let v0_before = contours[0].vertices[0].point;
    let v0_screen = pos2((v0_before[0] / 3.0) as f32, (v0_before[1] / 3.0) as f32);
    let drag_to = pos2(v0_screen.x + 30.0, v0_screen.y + 18.0);
    step(
        &mut session,
        &mut state,
        0.10,
        vec![egui::Event::PointerMoved(v0_screen)],
    );
    let coalesced_act = step(
        &mut session,
        &mut state,
        0.12,
        vec![
            egui::Event::PointerButton {
                pos: v0_screen,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerMoved(drag_to),
        ],
    );
    step(
        &mut session,
        &mut state,
        0.14,
        vec![egui::Event::PointerButton {
            pos: drag_to,
            button: PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
    assert!(coalesced_act.transform_modified);
    let after = collect_editable_contours_for_clip(&session, &clip_id, 0.0);
    let v0_after = after[0].vertices[0].point;
    assert!(
        (v0_after[0] - v0_before[0] - 90.0).abs() < 4.0
            && (v0_after[1] - v0_before[1] - 54.0).abs() < 4.0,
        "Layer mask vertex must move cleanly on coalesced press+drag (before={v0_before:?}, after={v0_after:?})"
    );
}

#[test]
pub(crate) fn discrete_wheel_2d_zoom_step_is_gentle() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    session.playback.editing_camera = false;
    let mut state = ViewportState::new();
    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 450.0));
    let hover = pos2(400.0, 225.0);

    let step =
        |session: &mut Session, state: &mut ViewportState, time: f64, events: Vec<egui::Event>| {
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
                        let _ = show_viewport(ui, session, state, None, &palette);
                    });
            });
            full.textures_delta.clear();
        };

    step(
        &mut session,
        &mut state,
        0.0,
        vec![egui::Event::PointerMoved(hover)],
    );
    step(
        &mut session,
        &mut state,
        0.02,
        vec![
            egui::Event::PointerMoved(hover),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: vec2(0.0, 1.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Wheel,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(
        state.zoom_2d > 1.02 && state.zoom_2d <= 1.25,
        "Single discrete wheel tick must zoom gently (~1.05x-1.22x), got {:.3}x",
        state.zoom_2d
    );
}

#[test]
pub(crate) fn freehand_mask_commit_preserves_single_active_tool_without_dual_highlight() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session
        .selection
        .select_clip(clip_id.clone(), Some("scene"), false);
    session.playback.editing_camera = false;
    // Start with 2D transform controls enabled to ensure entering/committing Freehand
    // never leaves both 2D Transform and Freehand highlighted simultaneously.
    session.playback.transform_controls_2d = true;

    let mut state = ViewportState::new();
    state.active_draw_tool = Some(2);
    state.quick_shape_kind = QuickShapeKind::Freehand;

    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(640.0, 360.0));
    let step = |session: &mut Session,
                state: &mut ViewportState,
                time: f64,
                events: Vec<egui::Event>|
     -> ViewportAction {
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(time),
            events,
            focused: true,
            ..Default::default()
        };
        let mut action = ViewportAction::default();
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    action = show_viewport(ui, session, state, None, &palette);
                });
        });
        full.textures_delta.clear();
        action
    };

    let p0 = pos2(160.0, 120.0);
    let p1 = pos2(260.0, 140.0);
    let p2 = pos2(220.0, 220.0);
    step(
        &mut session,
        &mut state,
        0.0,
        vec![egui::Event::PointerMoved(p0)],
    );
    step(
        &mut session,
        &mut state,
        0.02,
        vec![
            egui::Event::PointerMoved(p0),
            egui::Event::PointerButton {
                pos: p0,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    step(
        &mut session,
        &mut state,
        0.04,
        vec![egui::Event::PointerMoved(p1)],
    );
    let act = step(
        &mut session,
        &mut state,
        0.06,
        vec![
            egui::Event::PointerMoved(p2),
            egui::Event::PointerButton {
                pos: p2,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(act.transform_modified);
    assert_eq!(
        state.active_draw_tool,
        Some(2),
        "Freehand commit must stay cleanly on Freehand tool (`Some(2)`)"
    );
    assert!(
        !session.playback.transform_controls_2d,
        "2D transform controls must not stay dual-highlighted alongside active Freehand tool"
    );
    assert!(
        state.selected_path_vertex.is_none(),
        "Freehand commit must not leave Bezier-edit vertex handles active while in Freehand mode"
    );

    // Re-enabling 2D transform controls must cleanly exit the draw tool (mutually exclusive).
    session.playback.transform_controls_2d = true;
    step(&mut session, &mut state, 0.08, vec![]);
    assert_eq!(state.active_draw_tool, None);
    assert!(session.playback.transform_controls_2d);
}

#[test]
pub(crate) fn fast_flick_on_2d_corner_handle_uses_press_origin_and_scales_instead_of_body_move() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let track_id = session.document().video_tracks[0].id.clone();
    session
        .execute(Command::AddClip {
            track_id,
            kind: "shape_layer".to_string(),
            name: Some("FlickRect".to_string()),
            start: 0.0,
            length: 120.0,
        })
        .expect("add 2D video clip");
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session.selection.select_clip(&clip_id, None, false);
    session.playback.transform_controls_2d = true;
    session.playback.editing_camera = false;

    let panel = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
    let mut state = ViewportState::new();
    let step = |session: &mut Session,
                state: &mut ViewportState,
                time: f64,
                events: Vec<egui::Event>|
     -> ViewportAction {
        let mut action = ViewportAction::default();
        let raw = egui::RawInput {
            screen_rect: Some(panel),
            time: Some(time),
            focused: true,
            events,
            ..Default::default()
        };
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    action = show_viewport(ui, session, state, None, &palette);
                });
        });
        full.textures_delta.clear();
        action
    };

    // Scale the clip to 0.5x so its TopLeft corner handle sits strictly inside the canvas.
    let _ = session.execute(Command::SetPropertyValue {
        target_id: clip_id.clone(),
        property: "scale".to_string(),
        value: json!([0.5, 0.5]),
    });
    let _ = step(&mut session, &mut state, 0.0, vec![]);

    let seq_w = session.document().resolution[0].max(1) as f32;
    let seq_h = session.document().resolution[1].max(1) as f32;
    let canvas = compute_canvas_rect(panel, seq_w as u32, seq_h as u32);
    let ratio = canvas.width() / seq_w;
    // TopLeft corner of a 0.5x scale centered layer is at center - (0.25 * seq_w, 0.25 * seq_h) * ratio.
    let tl_corner = pos2(
        canvas.center().x - 0.25 * seq_w * ratio,
        canvas.center().y - 0.25 * seq_h * ratio,
    );
    // Simulate a fast click-drag flick: press_origin is right on TopLeft handle,
    // and pointer moves 22 px diagonally inward (> 8.5 px handle hit radius) in the SAME press frame!
    let flick_end = pos2(tl_corner.x + 22.0, tl_corner.y + 22.0);
    let act = step(
        &mut session,
        &mut state,
        0.02,
        vec![
            egui::Event::PointerMoved(tl_corner),
            egui::Event::PointerButton {
                pos: tl_corner,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerMoved(flick_end),
        ],
    );

    assert!(
        matches!(
            &state.drag_state,
            ViewportDragState::Transform2D(d) if d.handle == Transform2DHandle::TopLeft
        ),
        "fast flick starting on TopLeft handle must grab Transform2DHandle::TopLeft (not Body), got {:?}",
        state.drag_state
    );
    assert!(act.transform_modified);

    // Position must remain unchanged (no Body fall-through), while scale shrinks from 0.5.
    let tf = active_single_video_clip_transform(&session).expect("active transform");
    assert!(
        tf.pos_2d.x.abs() < 1e-3 && tf.pos_2d.y.abs() < 1e-3,
        "corner handle flick must not move position: {:?}",
        tf.pos_2d
    );
    assert!(
        tf.scale_2d.x < 0.48 && tf.scale_2d.y < 0.48,
        "inward TopLeft flick must scale layer down, got {:?}",
        tf.scale_2d
    );
}

#[test]
pub(crate) fn fast_flick_on_3d_gizmo_axis_uses_press_origin() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    session
        .execute(Command::AddObject3D {
            clip_id: clip_id.clone(),
            kind: "box".to_string(),
            name: Some("Box".to_string()),
            parent_id: None,
        })
        .expect("add 3D box");
    let obj_id = session.document().video_tracks[0].clips[0]
        .objects_3d
        .last()
        .expect("box object")
        .id
        .clone();
    let _ = session
        .selection
        .select_clip(&clip_id, Some("scene"), false);
    let _ = session.selection.select_object_3d(&obj_id, "box", false);
    session.playback.editing_camera = true;

    let panel = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
    let mut state = ViewportState::new();
    state.gizmo_3d_mode = Gizmo3DMode::Translate;

    let step = |session: &mut Session,
                state: &mut ViewportState,
                time: f64,
                events: Vec<egui::Event>|
     -> ViewportAction {
        let mut action = ViewportAction::default();
        let raw = egui::RawInput {
            screen_rect: Some(panel),
            time: Some(time),
            focused: true,
            events,
            ..Default::default()
        };
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    action = show_viewport(ui, session, state, None, &palette);
                });
        });
        full.textures_delta.clear();
        action
    };
    let _ = step(&mut session, &mut state, 0.0, vec![]);

    let canvas = compute_canvas_rect(panel, 1920, 1080);
    let origin_s = state
        .camera
        .project_world_point(Vec3::ZERO, canvas)
        .expect("origin projects");
    let aspect = canvas.width() / canvas.height().max(1.0);
    let world_len = (GIZMO_SCREEN_LEN_PX
        * ResolvedCamera3D::from_editor_camera(&state.camera, aspect)
            .world_units_per_px(Vec3::ZERO, canvas))
    .max(0.05);
    let tip_x = state
        .camera
        .project_world_point(Vec3::X * world_len, canvas)
        .expect("tip_x projects");
    let press_pt = pos2(
        origin_s.x + (tip_x.x - origin_s.x) * 0.75,
        origin_s.y + (tip_x.y - origin_s.y) * 0.75,
    );
    // Coalesced fast flick 24 px off the X axis shaft in the press frame.
    let flick_pt = pos2(press_pt.x + 24.0, press_pt.y - 24.0);
    let act = step(
        &mut session,
        &mut state,
        0.02,
        vec![
            egui::Event::PointerMoved(press_pt),
            egui::Event::PointerButton {
                pos: press_pt,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerMoved(flick_pt),
        ],
    );

    assert!(
        matches!(
            &state.drag_state,
            ViewportDragState::Gizmo3D(d) if d.handle == Gizmo3DHandle::AxisX
        ),
        "fast flick starting on 3D AxisX handle must grab Gizmo3DHandle::AxisX, got {:?}",
        state.drag_state
    );
    assert!(act.transform_modified);
}

#[test]
pub(crate) fn editing_camera_shift_left_drag_pans_and_discrete_wheel_dollies_gently() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session
        .selection
        .select_clip(&clip_id, Some("scene"), false);
    session.playback.editing_camera = true;

    let panel = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
    let mut state = ViewportState::new();
    let step = |session: &mut Session,
                state: &mut ViewportState,
                time: f64,
                modifiers: egui::Modifiers,
                mut events: Vec<egui::Event>|
     -> ViewportAction {
        let mut action = ViewportAction::default();
        events.insert(0, egui::Event::ModifiersChanged(modifiers));
        let raw = egui::RawInput {
            screen_rect: Some(panel),
            time: Some(time),
            focused: true,
            events,
            ..Default::default()
        };
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    action = show_viewport(ui, session, state, None, &palette);
                });
        });
        full.textures_delta.clear();
        action
    };

    let p0 = pos2(480.0, 270.0);
    let p1 = pos2(540.0, 240.0);
    let _ = step(
        &mut session,
        &mut state,
        0.0,
        egui::Modifiers::NONE,
        vec![egui::Event::PointerMoved(p0)],
    );

    // 1. Shift+Left-Drag must pan (shifting target & position together while preserving distance).
    let target_before = state.camera.target;
    let dist_before = state.camera.distance();
    let _ = step(
        &mut session,
        &mut state,
        0.02,
        egui::Modifiers::SHIFT,
        vec![
            egui::Event::PointerMoved(p0),
            egui::Event::PointerButton {
                pos: p0,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::SHIFT,
            },
        ],
    );
    let act_pan = step(
        &mut session,
        &mut state,
        0.04,
        egui::Modifiers::SHIFT,
        vec![egui::Event::PointerMoved(p1)],
    );
    assert!(act_pan.camera_changed);
    assert!(
        matches!(state.drag_state, ViewportDragState::CameraPan),
        "Shift+Left-Drag must enter CameraPan, got {:?}",
        state.drag_state
    );
    assert!(
        (state.camera.target - target_before).length() > 1e-2,
        "Shift+Left-Drag must translate camera target"
    );
    assert!(
        (state.camera.distance() - dist_before).abs() < 1e-3,
        "Shift+Left-Drag pan must preserve camera distance"
    );
    let _ = step(
        &mut session,
        &mut state,
        0.06,
        egui::Modifiers::NONE,
        vec![egui::Event::PointerButton {
            pos: p1,
            button: PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
    assert!(matches!(state.drag_state, ViewportDragState::None));

    // 2. 15 discrete mouse wheel zoom-out ticks (3 lines each) must dolly smoothly & controllably
    //    (e.g. ~2.5x–3.0x distance rather than shrinking the entire 3D grid to a 15x+ speck).
    state.camera.reset();
    let d0 = state.camera.distance();
    let mut t = 0.2;
    for _ in 0..15 {
        t += 0.05;
        let _ = step(
            &mut session,
            &mut state,
            t,
            egui::Modifiers::NONE,
            vec![
                egui::Event::PointerMoved(p0),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Line,
                    delta: vec2(0.0, -3.0),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Wheel,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
    }
    // Flush any egui wheel smoothing tail.
    for _ in 0..8 {
        t += 0.05;
        let _ = step(
            &mut session,
            &mut state,
            t,
            egui::Modifiers::NONE,
            vec![egui::Event::PointerMoved(p0)],
        );
    }
    let ratio = state.camera.distance() / d0;
    assert!(
        ratio > 1.3 && ratio < 4.0,
        "15 discrete wheel ticks must dolly smoothly within (1.3x, 4.0x), got {ratio:.2}x"
    );
}

#[test]
pub(crate) fn gizmo_3d_rotate_normalizes_tiny_or_negative_zero_euler_components() {
    let out = compute_gizmo_3d_rotate(
        Gizmo3DHandle::AxisY,
        Gizmo3DSpace::World,
        -Vec3::Z,
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Mat4::IDENTITY,
        Vec3::new(-0.0, 0.0, -0.0),
        GeomEulerOrder::XYZ,
        0.5,
        false,
    );
    assert!(out.y.abs() > 0.4);
    assert_eq!(out.x, 0.0);
    assert_eq!(out.z, 0.0);
    assert!(
        out.x.is_sign_positive() && out.z.is_sign_positive(),
        "3D rotation gizmo Euler output must never emit -0.0: {out:?}"
    );

    // Rotating by a tiny sub-epsilon angle must also normalize cleanly to +0.0.
    let tiny = compute_gizmo_3d_rotate(
        Gizmo3DHandle::AxisX,
        Gizmo3DSpace::Local,
        -Vec3::Z,
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Mat4::IDENTITY,
        Vec3::ZERO,
        GeomEulerOrder::XYZ,
        -1e-6,
        false,
    );
    assert_eq!(tiny, Vec3::ZERO);
    assert!(tiny.x.is_sign_positive() && tiny.y.is_sign_positive() && tiny.z.is_sign_positive());
}

#[test]
pub(crate) fn transparency_grid_paints_single_cached_2x2_textured_quad() {
    let ctx = egui::Context::default();
    let canvas = Rect::from_min_size(pos2(0.0, 0.0), vec2(1280.0, 720.0));
    let raw = egui::RawInput {
        screen_rect: Some(canvas),
        ..Default::default()
    };
    let mut full1 = ctx.run_ui(raw.clone(), |ui| {
        paint_transparency_grid(ui.painter(), canvas, canvas);
    });
    assert_eq!(
        full1.shapes.len(),
        1,
        "transparency grid must emit a single textured quad instead of per-cell meshes"
    );
    if let Shape::Mesh(mesh) = &full1.shapes[0].shape {
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.indices.len(), 6);
    } else {
        panic!("expected single textured Mesh quad for transparency grid");
    }
    let first_set_count = full1.textures_delta.set.len();
    assert!(
        first_set_count >= 1,
        "first frame should upload 2x2 checker texture"
    );
    full1.textures_delta.clear();

    // Second frame must reuse the cached TextureHandle (zero new texture uploads).
    let mut full2 = ctx.run_ui(raw, |ui| {
        paint_transparency_grid(ui.painter(), canvas, canvas);
    });
    assert_eq!(full2.shapes.len(), 1);
    assert_eq!(
        full2.textures_delta.set.len(),
        0,
        "subsequent frames must reuse the cached 2x2 checker texture without re-uploading"
    );
    full2.textures_delta.clear();
}
