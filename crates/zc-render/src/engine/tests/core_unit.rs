use super::*;

#[test]
fn property_track_evaluates_keyframes_bezier_and_expression_modes() {
    let project = Project::new_cm3_default();
    let controls = RefCell::new(BTreeMap::new());
    let props = BTreeMap::new();
    let stack = RefCell::new(HashSet::new());

    // Linear keyframe track: 0 -> 100 across frames 0..=10
    let mut linear = PropertyTrack::number(0.0);
    linear.animated = true;
    linear.keyframes = vec![
        PropertyKeyframe::linear(0.0, PropertyValue::Number(0.0)),
        PropertyKeyframe::linear(10.0, PropertyValue::Number(100.0)),
    ];

    let plus_plugins = PluginRegistry::new();
    let env_mid = test_eval_env(
        &project,
        &plus_plugins,
        &controls,
        &props,
        &stack,
        5.0,
        5.0 / 30.0,
        30.0,
    );
    assert!((eval_f64(&linear, &env_mid) - 50.0).abs() < 1e-6);

    // Expression using `const` succeeds with Expressions+ enabled, falls back to base
    // value when Expressions+ is disabled (VanillaCm3 mode).
    let expr_track = PropertyTrack::number_with_expr(42.0, "const k = 2; value * k + frame");
    let plus_val = eval_f64(&expr_track, &env_mid);
    assert!((plus_val - 89.0).abs() < 1e-6);

    let mut vanilla_plugins = PluginRegistry::new();
    vanilla_plugins.disable_all_optional();
    let env_vanilla = test_eval_env(
        &project,
        &vanilla_plugins,
        &controls,
        &props,
        &stack,
        5.0,
        5.0 / 30.0,
        30.0,
    );
    let vanilla_val = eval_f64(&expr_track, &env_vanilla);
    assert!((vanilla_val - 42.0).abs() < 1e-6);
}

#[test]
fn headless_project_engine_renders_default_scene_and_selftest_effects() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(msg)) if !crate::require_gpu_in_tests() => {
            eprintln!("skipping project engine GPU test (no adapter): {msg}");
            return;
        }
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let report = engine
        .selftest_all_effects(64, 64)
        .expect("selftest_all_effects must succeed");
    assert_eq!(report.standard_total, 44);
    assert_eq!(report.plugin_total, 12);
    assert!(
        report.all_passed(),
        "selftest failed: std={}/{}, plugin={}/{}, scene_non_black={}",
        report.standard_passed,
        report.standard_total,
        report.plugin_passed,
        report.plugin_total,
        report.default_scene_non_black_pixels,
    );
}

#[test]
fn malformed_cube_lut_and_unpremultiply_are_safe() {
    assert!(parse_cube_lut_bytes(b"LUT_3D_SIZE 4096\n0 0 0\n", 8192).is_none());
    assert!(parse_cube_lut_bytes(b"LUT_3D_SIZE 4294967295\n0 0 0\n", 8192).is_none());
    assert!(parse_cube_lut_bytes(b"LUT_3D_SIZE 1\n0 0 0\n", 8192).is_none());
    // 128^3 unwrapped is 16384x128 which exceeds an 8192 max_texture_dimension_2d limit.
    assert!(parse_cube_lut_bytes(b"LUT_3D_SIZE 128\n0 0 0\n", 8192).is_none());
    // 1D LUT exceeding max_texture_dimension_2d is rejected.
    assert!(parse_cube_lut_bytes(b"LUT_1D_SIZE 16384\n0 0 0\n", 8192).is_none());

    // Straight alpha recovery on translucent premultiplied pixels.
    let straight = crate::unpremultiply_rgba8(&[128, 64, 0, 128, 0, 0, 0, 0, 200, 100, 50, 255]);
    assert_eq!(
        straight,
        vec![255, 128, 0, 128, 0, 0, 0, 0, 200, 100, 50, 255]
    );
}

#[test]
fn gpu_safety_dimensions_custom_shader_uv_and_deep_nesting() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let plugins = PluginRegistry::new();

    // 1. Oversized dimensions return InvalidDimensions before any GPU/CPU allocation.
    let too_big = engine
        .ctx
        .device
        .limits()
        .max_texture_dimension_2d
        .saturating_add(1);
    assert!(matches!(
        engine.selftest_all_effects(too_big, 64),
        Err(RenderError::InvalidDimensions { .. })
    ));
    assert!(matches!(
        engine.render_frame_rgba8(&Project::new_cm3_default(), &plugins, 0.0, too_big, 64),
        Err(RenderError::InvalidDimensions { .. })
    ));

    // 2. Hostile custom shaders (huge uniform array, missing vertex varying) return Err(Shader), never panic.
    let mut bad_shader_proj = Project::new_cm3_default();
    bad_shader_proj.sequence.video_tracks[0].clips[0]
        .layer
        .effects
        .push(EffectInstance {
            id: Uuid::new_v4(),
            name: "HugeUniformArray".to_owned(),
            enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
            kind: EffectKind::CustomShader {
                frag_shader: "precision mediump float; uniform vec4 Big[1000000]; void main() { gl_FragColor = Big[0]; }".to_owned(),
                custom_properties: Vec::new(),
            },
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        });
    assert!(matches!(
        engine.render_frame_rgba8(&bad_shader_proj, &plugins, 0.0, 32, 32),
        Err(RenderError::Shader(_))
    ));

    let mut extra_varying_proj = Project::new_cm3_default();
    extra_varying_proj.sequence.video_tracks[0].clips[0]
        .layer
        .effects
        .push(EffectInstance {
            id: Uuid::new_v4(),
            name: "ExtraVarying".to_owned(),
            enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
            kind: EffectKind::CustomShader {
                frag_shader: "precision mediump float; varying vec2 vMissing; void main() { gl_FragColor = vec4(vMissing, 0.0, 1.0); }".to_owned(),
                custom_properties: Vec::new(),
            },
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        });
    assert!(matches!(
        engine.render_frame_rgba8(&extra_varying_proj, &plugins, 0.0, 32, 32),
        Err(RenderError::Shader(_))
    ));

    // 3. CM3 bottom-left vUv convention: top row must have vUv.y ~ 1.0, bottom row ~ 0.0,
    // and a second pass sampling texture2D(tDiffuse, vUvScaled) must preserve upright orientation.
    let mut uv_proj = Project::new_cm3_default();
    uv_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Generator;
    uv_proj.sequence.video_tracks[0].clips[0]
        .layer
        .effects
        .push(EffectInstance {
            id: Uuid::new_v4(),
            name: "PaintUv".to_owned(),
            enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
            kind: EffectKind::CustomShader {
                frag_shader: "precision mediump float; varying vec2 vUv; void main() { gl_FragColor = vec4(vUv.x, vUv.y, 0.0, 1.0); }".to_owned(),
                custom_properties: Vec::new(),
            },
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        });
    uv_proj.sequence.video_tracks[0].clips[0]
        .layer
        .effects
        .push(EffectInstance {
            id: Uuid::new_v4(),
            name: "CopyDiffuse".to_owned(),
            enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
            kind: EffectKind::CustomShader {
                frag_shader: "precision mediump float; varying vec2 vUvScaled; uniform sampler2D tDiffuse; void main() { gl_FragColor = texture2D(tDiffuse, vUvScaled); }".to_owned(),
                custom_properties: Vec::new(),
            },
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        });
    let uv_pixels = engine
        .render_frame_rgba8(&uv_proj, &plugins, 0.0, 16, 16)
        .expect("custom shader UV test must render");
    // Row 0 in top-down readback is the TOP of the image -> CM3 vUv.y near 1.0 (green > 220).
    let top_g = uv_pixels[(8 * 4) + 1];
    // Row 15 in top-down readback is the BOTTOM of the image -> CM3 vUv.y near 0.0 (green < 35).
    let bottom_g = uv_pixels[((15 * 16 + 8) * 4) + 1];
    assert!(
        top_g > 220 && bottom_g < 35,
        "expected CM3 bottom-left UV orientation preserved upright, got top_g={top_g}, bottom_g={bottom_g}"
    );

    // 4. Deep Composite and EffectGroup nesting completes cleanly without stack overflow.
    let mut deep_layer = Layer {
        id: Uuid::new_v4(),
        name: "Leaf".to_owned(),
        kind: LayerKind::Generator,
        position: PropertyTrack::vec2([0.0, 0.0]),
        scale: PropertyTrack::vec2([1.0, 1.0]),
        rotation: PropertyTrack::number(0.0),
        anchor: PropertyTrack::vec2([0.0, 0.0]),
        opacity: PropertyTrack::number(1.0),
        blending: "BLEND_SRC_OVER".to_owned(),
        effects: Vec::new(),
        resolution: [16, 16],
        preserved: BTreeMap::new(),
    };
    for _ in 0..24 {
        deep_layer = Layer {
            id: Uuid::new_v4(),
            name: "NestedComposite".to_owned(),
            kind: LayerKind::Composite {
                layers: vec![deep_layer],
            },
            position: PropertyTrack::vec2([0.0, 0.0]),
            scale: PropertyTrack::vec2([1.0, 1.0]),
            rotation: PropertyTrack::number(0.0),
            anchor: PropertyTrack::vec2([0.0, 0.0]),
            opacity: PropertyTrack::number(1.0),
            blending: "BLEND_SRC_OVER".to_owned(),
            effects: Vec::new(),
            resolution: [16, 16],
            preserved: BTreeMap::new(),
        };
    }
    let mut deep_fx = EffectInstance {
        id: Uuid::new_v4(),
        name: "LeafGroup".to_owned(),
        enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
        kind: EffectKind::EffectGroup {
            custom_properties: Vec::new(),
            children: Vec::new(),
        },
        params: BTreeMap::new(),
        preserved: BTreeMap::new(),
    };
    for _ in 0..24 {
        deep_fx = EffectInstance {
            id: Uuid::new_v4(),
            name: "NestedGroup".to_owned(),
            enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
            kind: EffectKind::EffectGroup {
                custom_properties: Vec::new(),
                children: vec![deep_fx],
            },
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        };
    }
    deep_layer.effects.push(deep_fx);
    let mut deep_proj = Project::new_cm3_default();
    deep_proj.sequence.video_tracks[0].clips[0].layer = deep_layer;
    let _ = engine
        .render_frame_rgba8(&deep_proj, &plugins, 0.0, 16, 16)
        .expect("deep Composite + EffectGroup must respect recursion caps");
}

#[test]
fn bezier_keyframe_segment_evaluates_persisted_control_points() {
    let project = Project::new_cm3_default();
    let plugins = PluginRegistry::new();
    let controls = RefCell::new(BTreeMap::new());
    let parent_props = BTreeMap::new();
    let stack = RefCell::new(HashSet::new());
    let bezier_tween = (1u32 << 8) | 1;
    let track = PropertyTrack {
        animated: true,
        expression: None,
        static_behavior: ProjectStaticBehavior::Constant,
        value: PropertyValue::Number(0.0),
        keyframes: vec![
            PropertyKeyframe::from_cm3_tween(
                0.0,
                PropertyValue::Number(0.0),
                1,
                [[-5.0, 0.0], [1.0, 1.0]],
                false,
            ),
            PropertyKeyframe::from_cm3_tween(
                30.0,
                PropertyValue::Number(1.0),
                bezier_tween,
                [[-1.0, 0.0], [5.0, 0.0]],
                false,
            ),
        ],
        channels: Vec::new(),
    };
    let env = test_eval_env(
        &project,
        &plugins,
        &controls,
        &parent_props,
        &stack,
        15.0,
        0.5,
        30.0,
    );
    let val = eval_f64(&track, &env);
    // With aggressive ease-out handles (out=[1,1], in=[-1,0]), midpoint value is far above linear 0.5.
    assert!(
        val > 0.75,
        "expected Bezier segment to evaluate persisted control points (>0.75), got {val}"
    );
}

#[test]
fn cube_lut_parser_parses_domain_bounds_without_corrupting_entries() {
    let cube = b"\
TITLE \"Test LUT 1 2 3\"
LUT_3D_SIZE 2
DOMAIN_MIN 0.1 0.2 0.3
DOMAIN_MAX 0.9 0.8 0.7
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
";
    let (lut, dmin, dmax) =
        parse_cube_lut_bytes(cube, 8192).expect("valid 3D .cube with DOMAIN_MIN/MAX");
    assert!((dmin[0] - 0.1).abs() < 1e-5);
    assert!((dmin[1] - 0.2).abs() < 1e-5);
    assert!((dmin[2] - 0.3).abs() < 1e-5);
    assert!((dmax[0] - 0.9).abs() < 1e-5);
    assert!((dmax[1] - 0.8).abs() < 1e-5);
    assert!((dmax[2] - 0.7).abs() < 1e-5);
    match lut {
        RegParamValue::Lut3D { size, data } => {
            assert_eq!(size, 2);
            // Entry (r=0,g=0,b=0) -> dst_idx 0 must be [0, 0, 0, 255].
            assert_eq!(&data[0..4], &[0, 0, 0, 255]);
            // Entry (r=1,g=0,b=0) -> dst_idx 4 must be [255, 0, 0, 255] (NOT shifted by DOMAIN_MAX).
            assert_eq!(&data[4..8], &[255, 0, 0, 255]);
        }
        other => panic!("expected Lut3D, got {other:?}"),
    }
}

#[test]
fn coerce_expr_value_sanitizes_nan_and_inf_in_vectors() {
    let nan_arr = ExprValue::Array(vec![
        ExprValue::Number(f64::NAN),
        ExprValue::Number(f64::INFINITY),
        ExprValue::Number(0.5),
        ExprValue::Number(f64::NEG_INFINITY),
    ]);
    let v3 = coerce_expr_value(&nan_arr, &PropertyValue::Vec3([1.0, 2.0, 3.0]));
    assert_eq!(v3, PropertyValue::Vec3([1.0, 2.0, 0.5]));

    let v4 = coerce_expr_value(&nan_arr, &PropertyValue::ColorRgba([0.1, 0.2, 0.3, 1.0]));
    assert_eq!(v4, PropertyValue::ColorRgba([0.1, 0.2, 0.5, 1.0]));

    let nan_scalar = ExprValue::Number(f64::NAN);
    let v2 = coerce_expr_value(&nan_scalar, &PropertyValue::Vec2([4.0, 5.0]));
    assert_eq!(v2, PropertyValue::Vec2([4.0, 5.0]));
}

#[test]
fn ortho_camera_respects_viewport_aspect_and_sequence_resolution() {
    let mut project = Project::new_cm3_default();
    project.sequence.resolution = [500, 500];
    let plugins = PluginRegistry::new();
    let controls = RefCell::new(BTreeMap::new());
    let parent_props = BTreeMap::new();
    let stack = RefCell::new(HashSet::new());
    let env = test_eval_env(
        &project,
        &plugins,
        &controls,
        &parent_props,
        &stack,
        0.0,
        0.0,
        30.0,
    );
    let cam = zc_project::CameraData {
        kind: CameraKind::Orthographic,
        ..Default::default()
    };
    let params = build_camera_params_without_vibrate(&cam, Mat4::IDENTITY, &env, [200.0, 100.0]);
    // half_h = 0.05 * 500 = 25.0 -> proj[1][1] = 1 / 25.0
    // aspect = 2.0 -> half_w = 50.0 -> proj[0][0] = 1 / 50.0
    assert!((params.proj[1][1] - (1.0 / 25.0)).abs() < 1e-5);
    assert!((params.proj[0][0] - (1.0 / 50.0)).abs() < 1e-5);
}

#[test]
fn hold_tween_beats_stale_easing_and_bezier_flags() {
    let mut hold_kf = PropertyKeyframe::linear(10.0, PropertyValue::Number(100.0));
    hold_kf.tween = 0; // Hold! Even though easing_index == 1 and we set is_bezier = true.
    hold_kf.easing_index = 1;
    hold_kf.is_bezier = true;

    let track = PropertyTrack {
        animated: true,
        expression: None,
        static_behavior: ProjectStaticBehavior::Constant,
        value: PropertyValue::Number(0.0),
        keyframes: vec![
            PropertyKeyframe::linear(0.0, PropertyValue::Number(10.0)),
            hold_kf,
        ],
        channels: Vec::new(),
    };
    let mid = eval_track_keyframes_only(&track, 5.0).as_f64().unwrap();
    assert!(
        (mid - 10.0).abs() < 1e-6,
        "Hold (tween == 0) must hold start value 10.0, got {mid}"
    );

    // Strict bit-8 Bezier check: tween = (3 << 8) | 1 = 769 is NOT Bezier, uses linear ease 1.
    let non_bezier = PropertyKeyframe::from_cm3_tween(
        10.0,
        PropertyValue::Number(100.0),
        (3 << 8) | 1,
        [[-1.0, 0.0], [1.0, 1.0]],
        false,
    );
    let track2 = PropertyTrack {
        animated: false, // Multi-keyframe track still evaluates even when animated == false!
        expression: None,
        static_behavior: ProjectStaticBehavior::Constant,
        value: PropertyValue::Number(0.0),
        keyframes: vec![
            PropertyKeyframe::linear(0.0, PropertyValue::Number(0.0)),
            non_bezier,
        ],
        channels: Vec::new(),
    };
    let mid2 = eval_track_keyframes_only(&track2, 5.0).as_f64().unwrap();
    assert!(
        (mid2 - 50.0).abs() < 1e-6,
        "tween=(3<<8)|1 must evaluate linear (50.0), got {mid2}"
    );
}

#[test]
fn constraints_tarjan_scc_follow_track_to_light_world_target_and_vibrate() {
    let mut proj = Project::new_cm3_default();
    let mut plugins = PluginRegistry::new();
    plugins.enable("constraints").expect("enable constraints");
    plugins.enable("camera-plus").expect("enable camera-plus");
    let controls = RefCell::new(BTreeMap::new());
    let parent_props = BTreeMap::new();
    let stack = RefCell::new(HashSet::new());

    // Target cube at (10, 0, 0)
    let mut target_obj = Object3D::new_shape_3d("Target", Shape3DKind::Box { size: [1.0; 3] });
    target_obj.position = PropertyTrack::vec3([10.0, 0.0, 0.0]);
    target_obj.constraints.zoidium_id = Some("target_1".to_owned());

    // Follower follows Target with +5 Y offset
    let mut follower = Object3D::new_shape_3d("Follower", Shape3DKind::Box { size: [1.0; 3] });
    follower.constraints.zoidium_id = Some("follower_1".to_owned());
    follower.constraints.follow = Some(zc_project::FollowConstraint {
        target_id: "target_1".to_owned(),
        space: 0,
        follow_position: true,
        follow_rotation: false,
        follow_scale: false,
        position_offset: PropertyTrack::vec3([0.0, 5.0, 0.0]),
        rotation_offset: PropertyTrack::vec3([0.0, 0.0, 0.0]),
        scale_offset: PropertyTrack::vec3([1.0, 1.0, 1.0]),
        fallback_pose: None,
    });

    // Tracker at origin tracks Follower (chained dependency: Follower -> Target, Tracker -> Follower)
    let mut tracker = Object3D::new_shape_3d("Tracker", Shape3DKind::Box { size: [1.0; 3] });
    tracker.position = PropertyTrack::vec3([0.0, 5.0, 0.0]);
    tracker.constraints.zoidium_id = Some("tracker_1".to_owned());
    tracker.constraints.track_to = Some(zc_project::TrackToConstraint {
        target_id: "follower_1".to_owned(),
        track_axis: 2, // +Z points toward follower at (10, 5, 0) => +X direction
        up_axis: 1,    // +Y up
        influence: PropertyTrack::number(100.0),
    });

    // Spot light inside a translated Group: target MUST stay in world space (P5 P1)
    let mut spot = Object3D::new_spot_light();
    spot.position = PropertyTrack::vec3([0.0, 10.0, 0.0]);
    if let Object3DKind::Light(ref mut l) = spot.kind {
        l.target = PropertyTrack::vec3([2.0, 3.0, 4.0]);
    }
    let mut grp = Object3D::new(
        "OffsetGroup",
        Object3DKind::Group {
            reflection_visibility: 0,
            custom_properties: Vec::new(),
            children: vec![spot],
        },
    );
    grp.position = PropertyTrack::vec3([100.0, 200.0, 300.0]);

    // Cyclic pair A <-> B must be skipped cleanly without infinite loop or panic
    let mut cyc_a = Object3D::new_shape_3d("CycA", Shape3DKind::Box { size: [1.0; 3] });
    cyc_a.constraints.zoidium_id = Some("cyc_a".to_owned());
    cyc_a.constraints.track_to = Some(zc_project::TrackToConstraint {
        target_id: "cyc_b".to_owned(),
        track_axis: 2,
        up_axis: 1,
        influence: PropertyTrack::number(100.0),
    });
    let mut cyc_b = Object3D::new_shape_3d("CycB", Shape3DKind::Box { size: [1.0; 3] });
    cyc_b.constraints.zoidium_id = Some("cyc_b".to_owned());
    cyc_b.constraints.track_to = Some(zc_project::TrackToConstraint {
        target_id: "cyc_a".to_owned(),
        track_axis: 2,
        up_axis: 1,
        influence: PropertyTrack::number(100.0),
    });

    // Put tracker BEFORE follower in authored order to verify topological sort!
    let objs = vec![tracker, follower, target_obj, grp, cyc_a, cyc_b];
    proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Scene3D {
        objects: objs.clone(),
        shadow_mode: ProjectCompatMode::Current,
    };
    let env = test_eval_env(
        &proj,
        &plugins,
        &controls,
        &parent_props,
        &stack,
        0.0,
        0.0,
        30.0,
    );
    let graph = evaluate_scene_graph_3d(&objs, Mat4::IDENTITY, &env);

    let f_idx = graph.by_uuid[&objs[1].id];
    let f_pos = graph.nodes[f_idx].world_mat.w_axis.truncate();
    assert!(
        (f_pos - Vec3::new(10.0, 5.0, 0.0)).length() < 1e-4,
        "Follow constraint must place follower at (10, 5, 0), got {f_pos:?}"
    );

    let tr_idx = graph.by_uuid[&objs[0].id];
    let tr_forward = graph.nodes[tr_idx].world_mat.z_axis.truncate().normalize();
    assert!(
        (tr_forward - Vec3::X).length() < 1e-4,
        "TrackTo +Z must aim along +X toward follower at (10, 5, 0), got {tr_forward:?}"
    );

    // Spot light inside group at (100, 200, 300): light position moves, target stays world (2,3,4).
    let spot_node = graph
        .nodes
        .iter()
        .find(|n| matches!(n.obj.kind, Object3DKind::Light(_)))
        .expect("spot node");
    assert!(
        (spot_node.world_mat.w_axis.truncate() - Vec3::new(100.0, 210.0, 300.0)).length() < 1e-4
    );
    assert!(
        (spot_node.light_target_world.unwrap() - Vec3::new(2.0, 3.0, 4.0)).length() < 1e-4,
        "light.target must stay in world space, got {:?}",
        spot_node.light_target_world
    );

    // Verify Camera+ Vibrate exact sine-hash noise & degree->radian additive Euler.
    let cam = zc_project::CameraData {
        vibrate: Some(zc_project::CameraVibrateParams {
            enabled: true,
            seed: 42,
            position_amplitude: PropertyTrack::vec3([10.0, 0.0, 0.0]),
            rotation_amplitude: PropertyTrack::vec3([0.0, 15.0, 0.0]), // 15 degrees
            frequency: PropertyTrack::number(0.5),
            regular_pulse: false,
            relative: false,
            enable_position: true,
            enable_rotation: true,
            position_frequency: PropertyTrack::number(0.5),
            rotation_frequency: PropertyTrack::number(0.5),
        }),
        ..Default::default()
    };
    let env_f3 = test_eval_env(
        &proj,
        &plugins,
        &controls,
        &parent_props,
        &stack,
        3.0,
        0.1,
        30.0,
    );
    let cam_obj = Object3D::new("Cam+", Object3DKind::Camera(Box::new(cam.clone())));
    let cam_p = build_camera_params_from_node(
        &cam,
        &cam_obj,
        Mat4::IDENTITY,
        Vec3::new(0.0, 0.0, 80.0),
        Quat::IDENTITY,
        Vec3::ONE,
        GeomEulerOrder::XYZ,
        &env_f3,
        [1920.0, 1080.0],
    );
    let expected_dx = (vibrate_sample_noise(3.0 * 0.5, 42.0, false) * 10.0) as f32;
    assert!(
        (cam_p.eye_position[0] - expected_dx).abs() < 1e-4,
        "Camera+ Vibrate X offset must match Zoidium sine hash ({expected_dx}), got {}",
        cam_p.eye_position[0]
    );
}

#[test]
fn expr_context_cross_property_cycle_detection_controls_and_world_coords() {
    let mut proj = Project::new_cm3_default();
    let mut plugins = PluginRegistry::new();
    plugins.enable("expressions-plus").expect("enable expr+");

    // Add Sequence Control
    proj.sequence.controls.push(zc_project::SequenceControl {
        id: "ctl_gain_id".to_owned(),
        name: "Gain".to_owned(),
        kind: SequenceControlKind::Slider,
        track: PropertyTrack::number(3.0),
        preserved: BTreeMap::new(),
    });

    let mut box_a = Object3D::new_shape_3d("BoxA", Shape3DKind::Box { size: [1.0; 3] });
    box_a.position = PropertyTrack::vec3([12.0, 4.0, 0.0]);
    let mut box_b = Object3D::new_shape_3d("BoxB", Shape3DKind::Box { size: [1.0; 3] });
    // Cross-object read + control lookup
    box_b.position = PropertyTrack {
        animated: true,
        expression: Some(
            "var p = thisLayer.object('BoxA').property('position').value; [p[0] * control('Gain', 'ctl_gain_id'), p[1], 0]"
                .to_owned(),
        ),
        static_behavior: ProjectStaticBehavior::Constant,
        value: PropertyValue::Vec3([0.0, 0.0, 0.0]),
        keyframes: Vec::new(),
        channels: Vec::new(),
    };
    let box_a_id = box_a.id;
    let box_b_id = box_b.id;

    let cam = Object3D::new_perspective_camera();
    proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Scene3D {
        objects: vec![cam, box_a, box_b],
        shadow_mode: ProjectCompatMode::Current,
    };

    let controls = RefCell::new(BTreeMap::new());
    let parent_props = BTreeMap::new();
    let stack = RefCell::new(HashSet::new());
    let env = test_eval_env(
        &proj,
        &plugins,
        &controls,
        &parent_props,
        &stack,
        0.0,
        0.0,
        30.0,
    );

    let LayerKind::Scene3D { objects, .. } = &proj.sequence.video_tracks[0].clips[0].layer.kind
    else {
        unreachable!();
    };
    let pos_b = eval_vec3(
        &objects[2].position,
        &env.with_object(box_b_id)
            .with_property(format!("prop:obj:{box_b_id}:position")),
    );
    assert!(
        (pos_b[0] - 36.0).abs() < 1e-4 && (pos_b[1] - 4.0).abs() < 1e-4,
        "cross-property expression + control() must evaluate to [36, 4, 0], got {pos_b:?}"
    );

    // Strict control(name, wrong_id) must fail (never fall back to name!)
    let strict_track = PropertyTrack::number_with_expr(99.0, "control('Gain', 'wrong_id')");
    assert!((eval_f64(&strict_track, &env) - 99.0).abs() < 1e-6);

    // worldPosition() on BoxA returns [12, 4, 0]
    let wp_track = PropertyTrack::number_with_expr(0.0, "thisObject.worldPosition()[0]");
    let wp_x = eval_f64(&wp_track, &env.with_object(box_a_id));
    assert!(
        (wp_x - 12.0).abs() < 1e-4,
        "thisObject.worldPosition()[0] must be 12.0, got {wp_x}"
    );
}

#[test]
fn cjk_fallback_font_rejects_latin_only_and_validates_glyph_coverage() {
    assert!(
        !font_supports_cjk(DEFAULT_TTF_FONT_BYTES),
        "Latin-only Geist-Regular.ttf must not pass CJK glyph validation"
    );
    assert!(
        !font_supports_cjk(b"not a valid font"),
        "invalid font bytes must not pass CJK glyph validation"
    );
    if let Some(bytes) = resolve_system_cjk_fallback_font() {
        assert!(
            font_supports_cjk(&bytes),
            "resolved OS CJK fallback font must actually contain CJK glyphs"
        );
    }
}

#[test]
fn shape_plus_is_time_dependent_gates_static_vs_animated_payloads() {
    let static_payload = serde_json::json!({
        "version": 1,
        "contents": [{
            "type": "rect",
            "size": { "value": [200.0, 100.0], "keyframes": [] },
            "expression": ""
        }]
    });
    assert!(
        !shape_plus_is_time_dependent(&static_payload),
        "static Shape+ payload with empty keyframes & expression must not be time-dependent"
    );

    let kf_payload = serde_json::json!({
        "contents": [{
            "type": "rect",
            "size": { "keyframes": [{ "frame": 0.0, "value": [100.0, 100.0] }] }
        }]
    });
    assert!(shape_plus_is_time_dependent(&kf_payload));

    let expr_payload = serde_json::json!({
        "contents": [{
            "type": "rect",
            "rotation": { "expression": "time * 90" }
        }]
    });
    assert!(shape_plus_is_time_dependent(&expr_payload));

    for op in ["wigglePaths", "wiggleTransform", "textAnimator"] {
        let as_type = serde_json::json!({ "contents": [{ "type": op }] });
        assert!(
            shape_plus_is_time_dependent(&as_type),
            "Shape+ item with type={op} must be time-dependent"
        );
        let as_key = serde_json::json!({ op: {} });
        assert!(
            shape_plus_is_time_dependent(&as_key),
            "Shape+ payload with key={op} must be time-dependent"
        );
    }
}

#[test]
fn stroked_text2d_caches_rasterized_edt_texture_across_frames() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(e) => e,
        Err(err) => {
            assert!(
                !crate::require_gpu_in_tests(),
                "ZOIDCRAFT_REQUIRE_GPU=1 set but headless GPU unavailable: {err}"
            );
            return;
        }
    };
    let plugins = PluginRegistry::new();
    let mut p2d = Project::new_cm3_default();
    p2d.sequence.resolution = [160, 90];
    p2d.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
    p2d.sequence.video_tracks[0].clips.clear();

    let node = Shape2DNode {
        id: Uuid::new_v4(),
        name: "Text".to_owned(),
        kind: Shape2DKind::Group {
            children: Vec::new(),
        },
        position: PropertyTrack::vec2([0.0, 0.0]),
        rotation: PropertyTrack::number(0.0),
        scale: PropertyTrack::vec2([1.0, 1.0]),
        operations: vec![
            DrawOp::Stroke {
                color: PropertyTrack::color([0.1, 0.8, 1.0]),
                opacity: PropertyTrack::number(1.0),
                size: PropertyTrack::number(6.0),
            },
            DrawOp::Fill {
                color: PropertyTrack::constant(PropertyValue::ColorRgba([1.0, 1.0, 1.0, 1.0])),
            },
        ],
        preserved: BTreeMap::new(),
    };
    p2d.sequence.video_tracks[0]
        .clips
        .push(zc_project::VideoClip {
            id: Uuid::new_v4(),
            name: "StrokedText2D".to_owned(),
            start: 0.0,
            length: 60.0,
            link: None,
            media: None,
            time_curve: PropertyTrack::linear_time_curve(60.0, 30),
            layer: Layer::new(
                "StrokedText2D",
                LayerKind::Text2D {
                    text: "HI".to_owned(),
                    font: Some("Geist-Regular.ttf".to_owned()),
                    size: 40,
                    shapes: vec![node],
                },
                [160, 90],
            ),
            preserved: BTreeMap::new(),
        });

    assert_eq!(engine.cached_texture_count(), 0);
    let px_f0 = engine
        .render_frame_rgba8(&p2d, &plugins, 0.0, 160, 90)
        .expect("stroked text2d frame 0");
    assert!(
        count_non_black_rgb_pixels(&px_f0) > 50,
        "stroked Text2D must render visible pixels"
    );
    let after_f0 = engine.cached_texture_count();
    assert_eq!(
        after_f0, 1,
        "stroked Text2D must insert its finished texture into texture_cache"
    );

    let px_f10 = engine
        .render_frame_rgba8(&p2d, &plugins, 10.0, 160, 90)
        .expect("stroked text2d frame 10");
    assert_eq!(
        engine.cached_texture_count(),
        after_f0,
        "unchanged stroked Text2D on frame 10 must hit texture_cache without creating a new entry"
    );
    assert_eq!(
        px_f0, px_f10,
        "cached stroked Text2D must match frame 0 bit-for-bit"
    );
}

#[test]
fn eval_track_keyframes_only_sorted_fast_path_matches_unsorted() {
    let mut sorted_track = PropertyTrack::number(0.0);
    sorted_track.animated = true;
    sorted_track.keyframes = vec![
        PropertyKeyframe::linear(0.0, PropertyValue::Number(10.0)),
        PropertyKeyframe::linear(10.0, PropertyValue::Number(30.0)),
        PropertyKeyframe::linear(20.0, PropertyValue::Number(90.0)),
        PropertyKeyframe::linear(30.0, PropertyValue::Number(100.0)),
    ];
    let mut unsorted_track = sorted_track.clone();
    unsorted_track.keyframes.swap(1, 2);

    for frame in [-5.0, 0.0, 2.5, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0] {
        let a = eval_track_keyframes_only(&sorted_track, frame)
            .as_f64()
            .unwrap();
        let b = eval_track_keyframes_only(&unsorted_track, frame)
            .as_f64()
            .unwrap();
        assert!(
            (a - b).abs() < 1e-9,
            "sorted fast-path ({a}) must match unsorted fallback ({b}) at frame {frame}"
        );
    }
}
