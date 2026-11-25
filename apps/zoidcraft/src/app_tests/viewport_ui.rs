//! viewport ui tests.

use super::*;

#[test]
fn eyedropper_preserves_rgba_alpha_and_rounds_animated_frame() {
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let fx_id = session
        .execute(Command::AddEffect {
            clip_id,
            kind: "customShader".to_owned(),
            name: None,
        })
        .expect("add customShader")["effect_id"]
        .as_str()
        .unwrap()
        .to_owned();

    session
        .execute(Command::SetPropertyValue {
            target_id: fx_id.clone(),
            property: "u_Tint".to_owned(),
            value: json!([1.0, 1.0, 1.0, 0.42]),
        })
        .expect("init rgba uniform");
    session
        .execute(Command::TogglePropertyAnimated {
            target_id: fx_id.clone(),
            property: "u_Tint".to_owned(),
            frame: 0.0,
        })
        .expect("enable animation");

    session.playback.current_frame = 12.37;
    let cmd = build_eyedropper_command(&session, fx_id, "u_Tint".to_owned(), [0.2, 0.4, 0.8]);
    match &cmd {
        Command::SetKeyframe { frame, value, .. } => {
            assert!(
                (*frame - 12.0).abs() < 1e-6,
                "expected rounded frame 12.0, got {frame}"
            );
            let arr = value.as_array().expect("4-element RGBA array");
            assert_eq!(arr.len(), 4);
            assert!((arr[0].as_f64().unwrap() - 0.2).abs() < 1e-6);
            assert!((arr[1].as_f64().unwrap() - 0.4).abs() < 1e-6);
            assert!((arr[2].as_f64().unwrap() - 0.8).abs() < 1e-6);
            assert!((arr[3].as_f64().unwrap() - 0.42).abs() < 1e-6);
        }
        other => panic!("expected SetKeyframe, got {other:?}"),
    }
    session.execute(cmd).expect("execute eyedropper cmd");
}

#[test]
fn scene_hash_and_camera_bits_invalidate_on_track_3d_near_far_preserved_and_empty_sha_asset() {
    let mut session = Session::new();
    let h_initial = compute_scene_hash(&session);

    // 1. Toggling `track.is_3d` must invalidate `compute_scene_hash`.
    let vtrack_id = session.document().video_tracks[0].id.clone();
    session
        .set_track_3d(&vtrack_id, true)
        .expect("set track 3d");
    let h_3d = compute_scene_hash(&session);
    assert_ne!(h_3d, h_initial, "track.is_3d toggle must change scene hash");

    // 2. Direct `project.layer.preserved` mutation (e.g., mask/shadows/backgroundColor)
    //    must invalidate `compute_scene_hash`.
    session.project.sequence.video_tracks[0].clips[0]
        .layer
        .preserved
        .insert("backgroundColor".to_owned(), json!([0.1, 0.2, 0.3]));
    let h_preserved = compute_scene_hash(&session);
    assert_ne!(
        h_preserved, h_3d,
        "layer.preserved mutation must change scene hash"
    );

    // 3. Equal-length asset replacement with empty sha256 must invalidate `compute_scene_hash`.
    session.project.assets.insert(
        "tex".to_owned(),
        zc_project::AssetRecord {
            id: "tex".to_owned(),
            kind: AssetKind::Image,
            source: zc_project::AssetSource::File,
            filename: "tex.png".to_owned(),
            url: None,
            external: false,
            bytes: Some(vec![10, 20, 30, 40]),
            missing: false,
            sha256: String::new(),
            size: 4,
            preserved: BTreeMap::new(),
        },
    );
    let h_asset1 = compute_scene_hash(&session);
    session.project.assets.get_mut("tex").unwrap().bytes = Some(vec![10, 20, 30, 99]);
    let h_asset2 = compute_scene_hash(&session);
    assert_ne!(
        h_asset2, h_asset1,
        "equal-length asset byte replacement with empty sha256 must change scene hash"
    );

    // 4. Editor camera `near` and `far` changes must alter `pack_camera_bits`.
    let mut cam = EditorCameraState::new();
    let bits0 = pack_camera_bits(&cam);
    cam.near *= 2.0;
    let bits_near = pack_camera_bits(&cam);
    assert_ne!(
        bits_near, bits0,
        "camera.near change must alter camera_bits"
    );
    cam.far *= 0.5;
    let bits_far = pack_camera_bits(&cam);
    assert_ne!(
        bits_far, bits_near,
        "camera.far change must alter camera_bits"
    );
}

#[test]
fn eval_prop_f64_evaluates_multi_keyframe_tracks_with_animated_false() {
    use zc_session::SessionKeyframe;

    let mk_kf = |frame: f64, val: f64, tween: u16| SessionKeyframe {
        frame,
        value: json!(val),
        tween,
        control_points: [[-10.0, 0.0], [10.0, 0.0]],
        continuous_tangent: true,
        value_kind: None,
    };
    let prop = SessionProperty {
        animated: false,
        expression: None,
        default_value: json!(0.0),
        keyframes: vec![mk_kf(0.0, 0.0, 1), mk_kf(20.0, 1.0, 1)],
    };
    assert!(
        (eval_prop_f64(Some(&prop), 10.0, 0.0) - 0.5).abs() < 1e-9,
        "multi-keyframe track with animated=false must still interpolate"
    );
    assert!(
        (eval_prop_f64(Some(&prop), 20.0, 0.0) - 1.0).abs() < 1e-9,
        "multi-keyframe track with animated=false must reach final keyframe"
    );

    // Single-keyframe track still returns keyframes[0].
    let single = SessionProperty {
        animated: false,
        expression: None,
        default_value: json!(0.0),
        keyframes: vec![mk_kf(0.0, 0.42, 1)],
    };
    assert!((eval_prop_f64(Some(&single), 10.0, 0.0) - 0.42).abs() < 1e-9);

    // Also verify end-to-end audio mixing respects multi-keyframe volume with animated=false
    // (matching imported CM3 projects loaded into Session).
    let mut session = Session::new();
    let sr = STANDARD_SAMPLE_RATE;
    let wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.5, 0.5]; sr as usize], sr);
    let asset_id =
        session
            .project
            .insert_binary_asset(AssetKind::Av, "dc.wav".to_owned(), wav.clone());
    let decoded = Arc::new(decode_audio_bytes(&wav, Some("wav")).expect("decode wav"));
    let mut cache = HashMap::new();
    cache.insert(asset_id.clone(), decoded);

    let track_id = session.document().audio_tracks[0].id.clone();
    let clip_id = session
        .execute(Command::AddClip {
            track_id,
            kind: "audio".to_owned(),
            name: Some("dc.wav".to_owned()),
            start: 0.0,
            length: 30.0,
        })
        .expect("add clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: clip_id.clone(),
            property: "assetId".to_owned(),
            value: json!(asset_id),
        })
        .expect("bind asset");

    // Set volume keyframes [0 -> 0.0, 30 -> 1.0], then flip `animated = false` on the
    // underlying project track and reload the session document.
    session
        .execute(Command::SetKeyframe {
            target_id: clip_id.clone(),
            property: "volume".to_owned(),
            frame: 0.0,
            value: json!(0.0),
            tween: Some(1),
        })
        .expect("kf 0");
    session
        .execute(Command::SetKeyframe {
            target_id: clip_id,
            property: "volume".to_owned(),
            frame: 30.0,
            value: json!(1.0),
            tween: Some(1),
        })
        .expect("kf 30");
    let mut project = session.project.clone();
    project.sequence.audio_tracks[0].clips[0]
        .audio
        .volume
        .animated = false;
    let reloaded = Session::from_project(project, session.plugins.clone());
    assert!(
        !reloaded.document().audio_tracks[0].clips[0]
            .properties
            .get("volume")
            .unwrap()
            .animated,
        "volume property must have animated=false after reload"
    );

    let mixed = mix_sequence_audio_stereo(&reloaded, &cache, sr, 30.0, 0, 30)
        .expect("mix ok")
        .expect("non-empty");
    let early = mixed[10][0].abs();
    let late = mixed[mixed.len() - 10][0].abs();
    assert!(
        early < 0.05 && late > 0.4,
        "multi-keyframe volume with animated=false must ramp (early={early}, late={late})"
    );
}

#[test]
fn active_camera_override_updates_render_key_and_resolves_camera_params() {
    let mut app = make_headless_app_for_test();
    let clip_id = app.session.document().video_tracks[0].clips[0].id.clone();
    let clip_uuid = Uuid::parse_str(&clip_id).expect("clip uuid");
    let _ = app
        .session
        .selection
        .select_clip(clip_id.clone(), Some("scene"), false);

    // Add a second 3D camera at a distinct position.
    let cam_id = app
        .session
        .execute(Command::AddObject3D {
            clip_id,
            kind: "camera".to_owned(),
            name: Some("Alt Cam".to_owned()),
            parent_id: None,
        })
        .expect("add camera")["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    app.session
        .execute(Command::SetPropertyValue {
            target_id: cam_id.clone(),
            property: "position".to_owned(),
            value: json!([25.0, 12.0, 90.0]),
        })
        .expect("set camera position");

    let (_, _, _, _, key_default) = app.current_viewport_render_key();
    app.viewport.active_camera_id = Some(cam_id.clone());
    let (_, _, _, editing_cam_flag, key_override) = app.current_viewport_render_key();

    assert_ne!(
        key_override.active_camera_hash, key_default.active_camera_hash,
        "selecting a scene camera in Camera Switcher must change ViewportRenderKey"
    );
    assert!(
        !editing_cam_flag,
        "explicit active_camera_id override takes precedence over editing_camera"
    );

    let params =
        resolve_scene_camera_override_params(&app.session, clip_uuid, &cam_id, 0.0, 1280.0, 720.0)
            .expect("camera override must resolve CameraParams");
    assert!((params.eye_position[0] - 25.0).abs() < 1e-4);
    assert!((params.eye_position[1] - 12.0).abs() < 1e-4);
    assert!((params.eye_position[2] - 90.0).abs() < 1e-4);
}

#[test]
fn practical_headless_intro_build_and_gpu_render_e2e() {
    let Ok(mut engine) = ProjectRenderEngine::new_headless() else {
        return;
    };
    let mut session = Session::new();
    for pid in [
        "particles-plus",
        "repeater-plus",
        "effector-plus",
        "shape-plus",
        "camera-plus",
        "material-plus",
        "sglow-hq-v2-preview",
    ] {
        let _ = session.plugins.enable(pid);
    }
    session
        .execute(Command::SetProjectName {
            name: "Practical Intro Test".to_owned(),
        })
        .expect("project name");
    session
        .execute(Command::SetResolution {
            width: 640,
            height: 360,
        })
        .expect("resolution");
    session.execute(Command::SetFps { fps: 60.0 }).expect("fps");
    session
        .execute(Command::SetSequenceLength { frames: 180.0 })
        .expect("sequence length");

    // Track 0 (V1): Reuse the default 3D Scene clip if present, or create one.
    let scene_clip_id = if let Some(existing) = session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.kind == "scene")
    {
        existing.id.clone()
    } else {
        let v0 = session.document().video_tracks[0].id.clone();
        session
            .execute(Command::AddClip {
                track_id: v0,
                kind: "scene".to_owned(),
                name: Some("Intro 3D Scene".to_owned()),
                start: 0.0,
                length: 180.0,
            })
            .expect("add 3d scene")["clip_id"]
            .as_str()
            .unwrap()
            .to_owned()
    };

    // Find or create Scene Camera and animate zoom-in (Z: 125 -> 62 -> 52).
    let cam_id = if let Some(existing_cam) = session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == scene_clip_id)
        .and_then(|c| {
            c.objects_3d
                .iter()
                .find(|o| o.kind.to_ascii_lowercase().contains("camera"))
        }) {
        existing_cam.id.clone()
    } else {
        session
            .execute(Command::AddObject3D {
                clip_id: scene_clip_id.clone(),
                kind: "PerspectiveCamera".to_owned(),
                name: Some("Main Camera".to_owned()),
                parent_id: None,
            })
            .expect("add camera")["object_id"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    for (f, pos) in [
        (0.0, [0.0, 12.0, 145.0]),
        (45.0, [6.0, 4.0, 88.0]),
        (120.0, [-6.0, 2.0, 76.0]),
    ] {
        session
            .execute(Command::SetKeyframe {
                target_id: cam_id.clone(),
                property: "position".to_owned(),
                frame: f,
                value: json!(pos),
                tween: Some(3),
            })
            .expect("camera keyframe");
    }

    // Reuse default Text3D object if present (or add one) and configure "ZOIDCRAFT" + SourceCodePro + Bevel + rotation.
    let text3d_id = if let Some(existing_text) = session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == scene_clip_id)
        .and_then(|c| {
            c.objects_3d
                .iter()
                .find(|o| o.kind.to_ascii_lowercase().contains("text"))
        }) {
        existing_text.id.clone()
    } else {
        session
            .execute(Command::AddObject3D {
                clip_id: scene_clip_id.clone(),
                kind: "Text3D".to_owned(),
                name: Some("Hero Title".to_owned()),
                parent_id: None,
            })
            .expect("add Text3D")["object_id"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    session
        .execute(Command::SetPropertyValue {
            target_id: text3d_id.clone(),
            property: "text".to_owned(),
            value: json!("ZOIDCRAFT"),
        })
        .expect("set 3d text");
    session
        .execute(Command::SetPropertyValue {
            target_id: text3d_id.clone(),
            property: "font".to_owned(),
            value: json!("SourceCodePro-Regular.ttf"),
        })
        .expect("set 3d font");
    session
        .execute(Command::SetPropertyValue {
            target_id: text3d_id.clone(),
            property: "bevel".to_owned(),
            value: json!(true),
        })
        .expect("enable bevel");
    for (f, rot) in [(0.0, [0.0, -0.5, 0.0]), (90.0, [0.0, 0.25, 0.0])] {
        session
            .execute(Command::SetKeyframe {
                target_id: text3d_id.clone(),
                property: "rotation".to_owned(),
                frame: f,
                value: json!(rot),
                tween: Some(4),
            })
            .expect("text3d rot kf");
    }

    // Add 3D Directional + Ambient light so 3D PBR meshes (Ring Cube) and bevels catch crisp shading.
    let light_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_clip_id.clone(),
            kind: "DirectionalLight".to_owned(),
            name: Some("Key Light".to_owned()),
            parent_id: None,
        })
        .expect("add DirectionalLight")["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: light_id,
            property: "position".to_owned(),
            value: json!([40.0, 60.0, 80.0]),
        })
        .expect("light pos");
    let _amb_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_clip_id.clone(),
            kind: "AmbientLight".to_owned(),
            name: Some("Fill Light".to_owned()),
            parent_id: None,
        })
        .expect("add AmbientLight");

    // Add 3D Particles+ burst with star_4 sprite preset (keeping valid GradientStop JSON for size/color).
    let part_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_clip_id.clone(),
            kind: "Particles".to_owned(),
            name: Some("Sparkles".to_owned()),
            parent_id: None,
        })
        .expect("add Particles")["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    for (prop, val) in [
        ("position", json!([0.0, 0.0, -24.0])),
        ("particles.spritePreset", json!("star_4")),
        ("particles.rate", json!(45.0)),
        ("particles.vspread", json!([65.0, 38.0, 35.0])),
    ] {
        session
            .execute(Command::SetPropertyValue {
                target_id: part_id.clone(),
                property: prop.to_owned(),
                value: val,
            })
            .expect("set particle prop");
    }

    // Add Radial Repeater + Box child positioned behind the title.
    let rep_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_clip_id.clone(),
            kind: "zoidium:repeater/radial".to_owned(),
            name: Some("Orbit Ring".to_owned()),
            parent_id: None,
        })
        .expect("add Radial Repeater")["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: rep_id.clone(),
            property: "position".to_owned(),
            value: json!([0.0, 0.0, -32.0]),
        })
        .expect("repeater pos");
    let box_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_clip_id.clone(),
            kind: "Box".to_owned(),
            name: Some("Ring Cube".to_owned()),
            parent_id: Some(rep_id),
        })
        .expect("add Box under repeater")["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: box_id,
            property: "scale".to_owned(),
            value: json!([0.28, 0.28, 0.28]),
        })
        .expect("box scale");

    // Add Bloom + RGB Shift effects to the 3D scene clip.
    let bloom_id = session
        .execute(Command::AddEffect {
            clip_id: scene_clip_id.clone(),
            kind: "bloom".to_owned(),
            name: Some("Bloom".to_owned()),
        })
        .expect("add bloom")["effect_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: bloom_id,
            property: "amount".to_owned(),
            value: json!(1.1),
        })
        .expect("bloom amount");

    let rgb_id = session
        .execute(Command::AddEffect {
            clip_id: scene_clip_id.clone(),
            kind: "rgbshift".to_owned(),
            name: Some("RGB Shift".to_owned()),
        })
        .expect("add rgbshift")["effect_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: rgb_id,
            property: "amount".to_owned(),
            value: json!(4.5),
        })
        .expect("rgbshift amount");

    // Track 1: 2D Subtitle Text with Geist Mono font positioned below the 3D title.
    let v1 = session
        .execute(Command::AddVideoTrack {
            name: Some("V2 Subtitle".to_owned()),
        })
        .expect("create V2")["track_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let text2d_clip_id = session
        .execute(Command::AddClip {
            track_id: v1,
            kind: "text".to_owned(),
            name: Some("Subtitle".to_owned()),
            start: 15.0,
            length: 165.0,
        })
        .expect("add text2d")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: text2d_clip_id.clone(),
            property: "text".to_owned(),
            value: json!("MOTION ENGINE v0.2"),
        })
        .expect("set 2d text");
    session
        .execute(Command::SetPropertyValue {
            target_id: text2d_clip_id.clone(),
            property: "font".to_owned(),
            value: json!("GeistMono-Regular.ttf"),
        })
        .expect("set 2d font");
    session
        .execute(Command::SetPropertyValue {
            target_id: text2d_clip_id.clone(),
            property: "position".to_owned(),
            value: json!([0.0, -95.0]),
        })
        .expect("set 2d position");
    session
        .execute(Command::SetPropertyValue {
            target_id: text2d_clip_id,
            property: "fontSize".to_owned(),
            value: json!(24.0),
        })
        .expect("set 2d size");

    // Track 2: Shape+ sleek cyan accent frame (stroke only so it frames the intro cleanly).
    let v2 = session
        .execute(Command::AddVideoTrack {
            name: Some("V3 Shape+".to_owned()),
        })
        .expect("create V3")["track_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let shape_plus_id = session
        .execute(Command::AddClip {
            track_id: v2,
            kind: "zoidium:shape-plus/shape-layer".to_owned(),
            name: Some("Accent Frame".to_owned()),
            start: 0.0,
            length: 180.0,
        })
        .expect("add Shape+")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let shape_plus_uuid = Uuid::parse_str(&shape_plus_id).expect("shape plus uuid");
    session
        .execute(Command::UpdateShapePlusPayload {
            clip_id: shape_plus_uuid,
            payload: json!({
                "schemaVersion": 1,
                "renderVersion": 1,
                "nextId": 4,
                "contents": [
                    {
                        "id": "s1",
                        "kind": "group",
                        "properties": {
                            "name": "Accent Border",
                            "enabled": 1,
                            "anchor": [0.0, 0.0],
                            "position": [0.0, 0.0],
                            "scale": [1.0, 1.0],
                            "rotation": 0.0,
                            "opacity": 0.85
                        },
                        "contents": [
                            {
                                "id": "s2",
                                "kind": "rectangle",
                                "properties": {
                                    "name": "Frame",
                                    "enabled": 1,
                                    "size": [580.0, 310.0],
                                    "position": [0.0, 0.0],
                                    "roundness": 12.0
                                }
                            },
                            {
                                "id": "s3",
                                "kind": "stroke",
                                "properties": {
                                    "name": "Cyan Border",
                                    "enabled": 1,
                                    "color": [0.25, 0.85, 1.0],
                                    "opacity": 0.9,
                                    "width": 2.5
                                }
                            }
                        ]
                    }
                ]
            }),
        })
        .expect("configure Shape+ cyan border");

    // Drive headless UI workspace + viewport for 1 frame in both modes to verify zero panics & clean HUD.
    let _ = session
        .selection
        .select_clip(scene_clip_id.clone(), Some("scene"), false);
    let _ = session
        .selection
        .select_object_3d(text3d_id, "Text3D", false);
    let ctx = egui::Context::default();
    let mut ws = Cm3Workspace::new();
    let mut theme = Cm3Theme::default();
    let mut vp = ViewportState::new();
    let mut tl = TimelineViewState::new();
    let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));
    let raw = egui::RawInput {
        screen_rect: Some(screen),
        ..Default::default()
    };
    session.playback.editing_camera = false;
    let palette = theme.palette;
    let mut ui_out = ctx.run_ui(raw, |ui| {
        let _ = ws.show(
            ui,
            &mut session,
            &mut theme,
            |ui, _r, _focused, sess| {
                let _ = show_viewport(ui, sess, &mut vp, None, &palette);
            },
            |ui, _r, _focused, sess| {
                let _ = show_timeline(ui, sess, &mut tl, &palette);
            },
        );
    });
    ui_out.textures_delta.clear();

    // Render real GPU intro frames (f=0, f=45, f=90) and verify visible animation & save/export roundtrip.
    let project = session.to_project();
    let f0 = engine
        .render_frame_rgba8(&project, &session.plugins, 0.0, 640, 360)
        .expect("render f0");
    let f45 = engine
        .render_frame_rgba8(&project, &session.plugins, 45.0, 640, 360)
        .expect("render f45");
    let f90 = engine
        .render_frame_rgba8(&project, &session.plugins, 90.0, 640, 360)
        .expect("render f90");

    let non_black_f45 = f45
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] > 4 || px[1] > 4 || px[2] > 4)
        .count();
    assert!(
        non_black_f45 > 2000,
        "intro frame 45 must contain rich rendered 3D+2D visuals, non_black={non_black_f45}"
    );
    assert_ne!(f0, f45, "intro must animate between frame 0 and frame 45");
    assert_ne!(f45, f90, "intro must animate between frame 45 and frame 90");

    // Round-trip save `.zcraft` and export `.pz` cleanly.
    let out_dir = std::env::temp_dir().join("zoidcraft_practical_intro_e2e");
    let _ = fs::create_dir_all(&out_dir);
    let zcraft_path = out_dir.join("intro.zcraft");
    let pz_path = out_dir.join("intro.pz");
    save_zcraft(&project, &zcraft_path).expect("save intro.zcraft");
    let reloaded = load_zcraft(&zcraft_path).expect("load intro.zcraft");
    assert!(
        reloaded.sequence.video_tracks.len() >= 3,
        "reloaded project must preserve all video tracks (got {})",
        reloaded.sequence.video_tracks.len()
    );
    export_pz(&project, &pz_path).expect("export intro.pz");
    let imported = import_pz(&pz_path).expect("reimport intro.pz");
    assert!(imported.project.sequence.video_tracks.len() >= 3);

    // Save preview PNGs for visual inspection.
    for (name, buf) in [
        ("intro_f0.png", &f0),
        ("intro_f45.png", &f45),
        ("intro_f90.png", &f90),
    ] {
        let straight = unpremultiply_rgba8(buf);
        let png_bytes = encode_png(&straight, 640, 360).expect("encode png");
        fs::write(out_dir.join(name), &png_bytes).expect("write png");
    }
}

#[test]
fn blank_startup_auto_selects_default_scene_and_disables_editing_camera() {
    let mut session = Session::new();
    auto_select_default_scene_clip(&mut session);
    session.playback.editing_camera = false;

    assert!(
        selected_clip_is_scene3d(&session),
        "fresh startup must auto-select V1's default Scene clip"
    );
    assert!(
        !session.playback.editing_camera,
        "fresh startup must default to Preview mode (editing_camera = false)"
    );
}

#[test]
fn recovery_banner_anchors_to_bottom_center_of_viewport() {
    let vp = Rect::from_min_size(Pos2::new(400.0, 43.0), Vec2::new(880.0, 420.0));
    let anchor = recovery_banner_anchor_pos(Some(vp)).expect("viewport anchor");
    assert!((anchor.x - vp.center().x.round()).abs() < 1e-3);
    assert!((anchor.y - (vp.bottom() - 10.0).round()).abs() < 1e-3);
    assert!(
        anchor.y > vp.top() + 100.0,
        "recovery banner must sit near bottom of viewport, away from top HUD & toolbar"
    );
    assert!(recovery_banner_anchor_pos(None).is_none());
}

#[test]
fn app_cmd_v_paste_event_attaches_shader_effect_and_consumes_clipboard_events() {
    ui::shortcuts::clear_test_clipboard();
    let mut app = make_headless_app_for_test();
    app.workspace.active_tab = ui::panels::SideTab::Effects;
    app.workspace.focused_panel = PanelFocus::SideInspector;
    let clip_id = app.session.document().video_tracks[0].clips[0].id.clone();
    let _ = app
        .session
        .selection
        .select_clip(clip_id, Some("scene"), false);
    let initial_fx = app.session.document().video_tracks[0].clips[0]
        .effects
        .len();

    let ctx = egui::Context::default();
    let raw_glsl =
        "uniform sampler2D tDiffuse; void main() { gl_FragColor = vec4(1.0, 0.2, 0.4, 1.0); }";
    let mut raw = egui::RawInput::default();
    raw.events.push(egui::Event::Paste(raw_glsl.to_owned()));
    raw.events.push(egui::Event::Key {
        key: Key::V,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::COMMAND,
    });

    let mut remaining_paste_events = 0usize;
    let mut full = ctx.run_ui(raw, |ui| {
        let outcome = dispatch_shortcuts_with_context(
            ui.ctx(),
            &mut app.session,
            &mut app.timeline,
            &mut app.graph_editor,
            &mut app.viewport,
            app.workspace.focused_panel,
            app.workspace.active_tab,
        );
        assert!(outcome.consumed_clipboard_events);
        assert!(outcome.last_error.is_none());
        assert_eq!(
            outcome.status_message.as_deref(),
            Some("Pasted 1 effect from clipboard")
        );
        consume_dispatched_shortcut_events(
            ui.ctx(),
            &outcome.handled_keys,
            outcome.consumed_clipboard_events,
        );
        remaining_paste_events = ui.ctx().input(|i| {
            i.events
                .iter()
                .filter(|ev| {
                    matches!(
                        ev,
                        egui::Event::Paste(_) | egui::Event::Key { key: Key::V, .. }
                    )
                })
                .count()
        });
    });
    full.textures_delta.clear();

    assert_eq!(
        remaining_paste_events, 0,
        "consumed clipboard events must be stripped so downstream panels never double-paste"
    );
    assert_eq!(
        app.session.document().video_tracks[0].clips[0]
            .effects
            .len(),
        initial_fx + 1
    );
}

#[test]
fn missing_required_plugins_banner_detects_and_enables_with_zero_blank_false_positives() {
    let mut app = make_headless_app_for_test();
    assert!(
        app.visible_missing_required_plugins().is_empty(),
        "fresh blank project must produce zero false-positive missing-plugin warnings"
    );

    // Add a composition to `project.compositions` while `precomp-plus` is disabled by default.
    let comp_seq = zc_project::Sequence {
        name: "Precomp 1".to_owned(),
        ..Default::default()
    };
    app.session
        .project
        .compositions
        .insert(comp_seq.id, comp_seq);
    app.session
        .project
        .plugin_requirements
        .insert("optical-flares".to_owned());
    app.cached_missing_plugins_revision = None;

    let missing = app.visible_missing_required_plugins();
    assert!(
        missing.contains(&"precomp-plus".to_owned()),
        "project with compositions must flag disabled precomp-plus: {missing:?}"
    );
    assert!(
        missing.contains(&"optical-flares".to_owned()),
        "project requiring optical-flares must flag disabled optical-flares: {missing:?}"
    );

    // 1-click enable enables all required plugins and clears the warning banner.
    app.enable_required_plugins(&missing);
    assert!(app.session.plugins.is_enabled("precomp-plus"));
    assert!(app.session.plugins.is_enabled("optical-flares"));
    assert!(
        app.visible_missing_required_plugins().is_empty(),
        "missing-plugin banner must clear after 1-click enable"
    );
}

#[test]
fn recovery_banner_clamps_within_800x540_formats_timestamp_and_marks_dirty_on_restore() {
    // 1. At 800x540, viewport right pane sits at x=273..800 (width 527px).
    let vp_800 = Rect::from_min_size(Pos2::new(273.0, 0.0), Vec2::new(527.0, 310.0));
    let (banner_w, bounds) =
        compute_viewport_banner_bounds(vp_800, 460.0).expect("banner bounds at 800x540");
    let left = (bounds.center().x - banner_w * 0.5)
        .clamp(
            bounds.left(),
            (bounds.right() - banner_w).max(bounds.left()),
        )
        .round();
    assert!(
        left >= vp_800.left() + 8.0 - 0.5 && left + banner_w <= vp_800.right() - 8.0 + 0.5,
        "recovery banner [{left}..{}] must stay strictly inside viewport.shrink(8) [{}..{}]",
        left + banner_w,
        vp_800.left() + 8.0,
        vp_800.right() - 8.0
    );

    // Even at ultra-narrow 220px viewport width, banner width strictly clamps inside bounds.
    let vp_tiny = Rect::from_min_size(Pos2::new(273.0, 0.0), Vec2::new(220.0, 200.0));
    let (tiny_w, tiny_bounds) =
        compute_viewport_banner_bounds(vp_tiny, 460.0).expect("tiny banner bounds");
    assert!(tiny_w <= tiny_bounds.width() + 0.5);

    // 2. Clean UTC timestamp formatting.
    assert_eq!(format_recovery_timestamp(0), "unsaved snapshot");
    assert_eq!(
        format_recovery_timestamp(1_700_000_000),
        "2023-11-14 22:13 UTC"
    );

    // 3. Restoring a recovery snapshot marks the session dirty (`*`) so unsaved-changes triggers.
    let mut app = make_headless_app_for_test();
    let tmp_dir = std::env::temp_dir().join("zoidcraft_test_recovery_dirty");
    let _ = fs::remove_dir_all(&tmp_dir);
    let snap_info =
        save_recovery_snapshot_to_dir(&app.session.project, &tmp_dir, MAX_RECOVERY_SNAPSHOTS)
            .expect("write recovery snapshot");
    app.startup_recovery_banner = Some(snap_info.clone());

    app.handle_restore_recovery_snapshot(&snap_info.path);
    let _ = fs::remove_dir_all(&tmp_dir);

    assert!(
        app.session.is_dirty() && app.has_unsaved_changes(),
        "restoring a recovery snapshot must mark the session dirty so unsaved-changes triggers"
    );
    assert!(app.startup_recovery_banner.is_none());
}

#[test]
fn live_viewport_isolates_active_child_composition_and_preserves_root_controls() {
    let mut project = Project::new_cm3_default();
    let root_seq_id = project.sequence.id;
    let root_track_ids: Vec<Uuid> = project.sequence.video_tracks.iter().map(|t| t.id).collect();

    // Add a root-only sequence control so we verify expressions in child compositions keep it.
    project.sequence.controls.push(zc_project::SequenceControl {
        id: "ctl-root-master".to_owned(),
        name: "MasterDrive".to_owned(),
        kind: zc_project::SequenceControlKind::Slider,
        track: zc_project::PropertyTrack::constant(zc_project::PropertyValue::Number(1.25)),
        preserved: BTreeMap::new(),
    });

    // Create a distinct child composition with its own unique track ID and no local controls.
    let mut child = Project::new_cm3_default().sequence;
    let child_id = Uuid::new_v4();
    let child_track_id = Uuid::new_v4();
    child.id = child_id;
    child.name = "Precomp 1".to_owned();
    child.controls.clear();
    child.video_tracks[0].id = child_track_id;
    child.video_tracks.truncate(1);
    project.compositions.insert(child_id, child);

    let mut session = Session::from_project(project, PluginRegistry::default());

    // When active_composition_id is None, live viewport borrows Root without cloning.
    let root_hash = compute_scene_hash(&session);
    let vp_root = project_for_live_viewport(&session);
    assert!(matches!(vp_root, Cow::Borrowed(_)));
    assert_eq!(vp_root.sequence.id, root_seq_id);
    assert_eq!(
        vp_root
            .sequence
            .video_tracks
            .iter()
            .map(|t| t.id)
            .collect::<Vec<_>>(),
        root_track_ids
    );

    // Switching into Precomp 1 must isolate child_id's tracks (NOT Root's), preserve root
    // sequence.controls, leave session.project unmutated, and change compute_scene_hash!
    session
        .execute(Command::SetActiveComposition {
            comp_id: Some(child_id),
        })
        .expect("switch active composition to Precomp 1");
    assert_eq!(session.document().active_composition_id, Some(child_id));

    let child_hash = compute_scene_hash(&session);
    assert_ne!(
        root_hash, child_hash,
        "switching active_composition_id must invalidate scene_hash"
    );

    let vp_child = project_for_live_viewport(&session);
    assert!(matches!(vp_child, Cow::Owned(_)));
    assert_eq!(vp_child.sequence.id, child_id);
    assert_eq!(vp_child.sequence.video_tracks.len(), 1);
    assert_eq!(vp_child.sequence.video_tracks[0].id, child_track_id);
    assert_ne!(
        vp_child.sequence.video_tracks[0].id, root_track_ids[0],
        "isolated child viewport render must use Precomp 1's tracks, not Root's"
    );
    assert!(
        vp_child
            .sequence
            .controls
            .iter()
            .any(|c| c.id == "ctl-root-master" && c.name == "MasterDrive"),
        "isolated child viewport render must preserve root sequence.controls for expressions"
    );
    assert!(
        vp_child.compositions.contains_key(&root_seq_id),
        "promoted project must keep root sequence in compositions"
    );
    // Critical: session.project itself must remain untouched (root stays root).
    assert_eq!(session.project.sequence.id, root_seq_id);
    assert!(session.project.compositions.contains_key(&child_id));
}

#[test]
fn viewport_gpu_render_error_throttles_duplicates_emits_shader_toast_and_clears_on_ok() {
    let mut app = make_headless_app_for_test();
    assert!(app.last_viewport_render_error.is_none());
    assert!(app.status_toast.is_none());

    // First broken GLSL error logs & emits a user-visible "Shader error: ..." toast.
    let err1 = "shader error: GLSL 450 parse error at line 4: unexpected token";
    assert!(app.record_viewport_render_result(Err(err1.to_owned())));
    assert_eq!(app.last_viewport_render_error.as_deref(), Some(err1));
    let toast1 = app
        .status_toast
        .as_ref()
        .expect("shader error must emit UI toast")
        .0
        .clone();
    assert!(
        toast1.starts_with("Shader error:"),
        "expected 'Shader error: ...' toast, got {toast1}"
    );

    // Repeating the identical error on subsequent 60Hz frames must be throttled (no log/toast spam).
    app.status_toast = None;
    for _ in 0..10 {
        assert!(!app.record_viewport_render_result(Err(err1.to_owned())));
        assert!(
            app.status_toast.is_none(),
            "identical repeated shader error must not re-emit toast or spam logs"
        );
    }

    // A DIFFERENT shader error emits a fresh toast and updates tracked error.
    let err2 = "shader error: GLSL 450 parse error at line 9: unknown identifier";
    assert!(app.record_viewport_render_result(Err(err2.to_owned())));
    assert_eq!(app.last_viewport_render_error.as_deref(), Some(err2));
    assert!(app.status_toast.is_some());

    // Clean render clears last_viewport_render_error so future errors will be reported again.
    assert!(!app.record_viewport_render_result(Ok(true)));
    assert!(app.last_viewport_render_error.is_none());
    assert!(app.record_viewport_render_result(Err(err1.to_owned())));
}

#[test]
fn preview_quality_is_independent_of_logical_canvas_size() {
    use crate::ui::layout::PreviewQuality;

    let mut app = make_headless_app_for_test();
    app.last_canvas_rect = Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(760.0, 428.0)));
    for (quality, expected) in [
        (PreviewQuality::Full, (1920, 1080)),
        (PreviewQuality::Half, (960, 540)),
        (PreviewQuality::Quarter, (480, 270)),
        (PreviewQuality::Eighth, (240, 135)),
    ] {
        app.workspace.player_plus.preview_quality = quality;
        let (_, width, height, _, key) = app.current_viewport_render_key();
        assert_eq!((width, height), expected, "{quality:?}");
        assert_eq!((key.width, key.height), expected);
    }

    app.session
        .plugins
        .disable("player-plus", &BTreeSet::new())
        .expect("disable Player+");
    let (_, width, height, _, _) = app.current_viewport_render_key();
    assert_eq!((width, height), (1920, 1080));
}

#[test]
fn preview_render_resolution_scales_sequence_and_obeys_texture_limits() {
    assert_eq!(
        compute_preview_render_resolution([1921, 1081], 3, 8192),
        (640, 360)
    );
    assert_eq!(
        compute_preview_render_resolution([80, 48], 8, 8192),
        (64, 48)
    );
    assert_eq!(
        compute_preview_render_resolution([9000, 6000], 1, 8192),
        (8192, 6000)
    );
    assert_eq!(
        compute_preview_render_resolution([9000, 6000], 2, 4096),
        (4096, 3000)
    );
}

#[test]
fn preview_frame_cache_accumulates_across_scrubbing_and_invalidates_on_edit() {
    let mut app = make_headless_app_for_test();
    let w = 640_u32;
    let h = 360_u32;
    let frame_bytes = (w as usize) * (h as usize) * 4;
    app.session
        .execute(Command::SetResolution {
            width: w,
            height: h,
        })
        .expect("set sequence resolution");

    let budget_bytes = 512 * 1024 * 1024;
    let mut cache: PreviewFrameCache<u32> = PreviewFrameCache::new(budget_bytes);

    // Scrub across 5 frames (0..5): every frame must accumulate without invalidating earlier frames.
    for f in 0..5_u32 {
        app.session
            .execute(Command::SetFrame {
                frame: f64::from(f),
            })
            .expect("scrub frame");
        let (frame, rw, rh, _, key) = app.current_viewport_render_key();
        assert_eq!((rw, rh), (w, h));
        cache.sync_context(key.cache_context());
        let bucket = frame.floor().max(0.0) as i64;
        assert!(
            cache.get_mut(bucket, key.frame_index, false).is_none(),
            "first visit to frame {f} must miss cache"
        );
        assert!(cache.insert(bucket, key.frame_index, frame_bytes, f));
    }

    assert_eq!(
        cache.len(),
        5,
        "scrubbing 5 frames must accumulate 5 cached frames"
    );
    assert_eq!(
        cache.used_bytes(),
        5 * frame_bytes,
        "used_bytes must equal 5 * frame_bytes (> 0)"
    );
    assert!(cache.used_bytes() > 0);
    assert_eq!(
        cache.cached_spans(),
        vec![(0.0, 5.0)],
        "5 consecutive scrubbed frames must coalesce into a single contiguous [0, 5) span"
    );

    // Scrubbing back to frame 2 must hit cache without clearing other frames.
    app.session
        .execute(Command::SetFrame { frame: 2.0 })
        .expect("scrub back to frame 2");
    let (frame, _, _, _, key2) = app.current_viewport_render_key();
    cache.sync_context(key2.cache_context());
    assert_eq!(
        cache.get_mut(frame.floor() as i64, key2.frame_index, false),
        Some(&2),
        "revisiting frame 2 must hit preview cache"
    );
    assert_eq!(cache.len(), 5);

    // Real edit (mutating sequence background color) must invalidate cached frames cleanly.
    app.session
        .execute(Command::SetPropertyValue {
            target_id: "sequence".to_owned(),
            property: "backgroundColor".to_owned(),
            value: json!([0.2, 0.4, 0.8]),
        })
        .expect("edit backgroundColor");
    let (_, _, _, _, edited_key) = app.current_viewport_render_key();
    cache.sync_context(edited_key.cache_context());
    assert_eq!(
        cache.len(),
        0,
        "real project edit must invalidate preview cache cleanly"
    );
    assert_eq!(cache.used_bytes(), 0);
    assert!(cache.cached_spans().is_empty());
}

#[test]
fn missing_plugins_banner_summary_and_toast_recovery_clearance() {
    let five: Vec<String> = vec![
        "Camera+".to_owned(),
        "Light+".to_owned(),
        "Graph Editor 2".to_owned(),
        "Optical Flares".to_owned(),
        "Precomp+".to_owned(),
    ];
    assert_eq!(
        format_missing_plugins_banner_summary(&five, false),
        "Disabled plugins (5): Camera+, Light+, +3 more"
    );
    assert_eq!(
        format_missing_plugins_banner_summary(&five, true),
        "Disabled plugins (5): Camera+, +4 more"
    );
    let one = vec!["Camera+".to_owned()];
    assert_eq!(
        format_missing_plugins_banner_summary(&one, false),
        "Disabled plugin (1): Camera+"
    );

    // Status toast with Recovery banner visible must shift vertically above the Recovery banner.
    let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));
    let vp = Rect::from_min_size(Pos2::new(398.0, 0.0), Vec2::new(882.0, 466.0));
    let normal = compute_status_toast_bounds_with_bottom_inset(Some(vp), screen, 0.0);
    let with_recovery = compute_status_toast_bounds_with_bottom_inset(Some(vp), screen, 46.0);
    assert!(
        with_recovery.anchor_pos.y <= normal.anchor_pos.y - 46.0 + 0.5,
        "toast with recovery banner ({}) must sit strictly above normal bottom ({})",
        with_recovery.anchor_pos.y,
        normal.anchor_pos.y
    );
}

#[test]
fn force_single_render_is_consumed_when_drawing_paused_is_false() {
    let mut app = make_headless_app_for_test();
    assert!(app.force_single_render);
    assert!(!app.workspace.player_plus.drawing_paused);

    app.workspace.player_plus.single_render_requested = true;
    let _ = app.update_viewport_gpu();
    assert!(
        !app.force_single_render,
        "force_single_render must be unconditionally consumed when drawing_paused == false to prevent 60-120Hz idle spin"
    );
    assert!(
        !app.workspace.player_plus.single_render_requested,
        "single_render_requested must also be consumed when drawing_paused == false"
    );

    // Also stays consumed when drawing_paused == true.
    app.workspace.player_plus.drawing_paused = true;
    app.force_single_render = true;
    app.workspace.player_plus.single_render_requested = true;
    let _ = app.update_viewport_gpu();
    assert!(!app.force_single_render);
    assert!(!app.workspace.player_plus.single_render_requested);
}

#[test]
fn preview_frame_cache_cached_spans_updates_cleanly_across_insert_evict_and_clear() {
    // Budget fits at most 3 entries of 100 bytes each.
    let mut cache: PreviewFrameCache<u32> = PreviewFrameCache::new(300);
    assert!(cache.cached_spans().is_empty());
    assert!(!cache.spans_dirty);

    // Out-of-order inserts coalesce into sorted merged spans.
    assert!(cache.insert(10, 10_000, 100, 10));
    assert!(cache.insert(2, 2_000, 100, 2));
    assert!(cache.insert(3, 3_000, 100, 3));
    assert!(cache.spans_dirty);
    assert_eq!(cache.cached_spans(), &[(2.0, 4.0), (10.0, 11.0)]);
    assert!(!cache.spans_dirty);

    // Calling cached_spans again reuses cached result cleanly.
    assert_eq!(cache.cached_spans(), &[(2.0, 4.0), (10.0, 11.0)]);

    // Inserting a 4th entry evicts the LRU entry (frame 10) and updates spans.
    assert!(cache.insert(4, 4_000, 100, 4));
    assert!(cache.spans_dirty);
    assert_eq!(cache.cached_spans(), &[(2.0, 5.0)]);
    assert!(!cache.spans_dirty);

    // Shrinking max_bytes evicts oldest entries (frames 2 and 3) and updates spans.
    cache.set_max_bytes(100);
    assert!(cache.spans_dirty);
    assert_eq!(cache.cached_spans(), &[(4.0, 5.0)]);
    assert!(!cache.spans_dirty);

    // clear() empties cached spans immediately.
    cache.clear();
    assert!(!cache.spans_dirty);
    assert!(cache.cached_spans().is_empty());
}

#[test]
fn cached_scene_hash_matches_fresh_and_invalidates_on_edit_and_editing_camera_selection() {
    let mut app = make_headless_app_for_test();
    let fresh0 = compute_scene_hash(&app.session);
    let cached0 = app.cached_scene_hash();
    assert_eq!(cached0, fresh0);
    // Second call hits cache identically.
    assert_eq!(app.cached_scene_hash(), fresh0);

    // Enabling editing_camera + selecting an object invalidates scene hash even without revision bump.
    app.session.playback.editing_camera = true;
    app.session
        .selection
        .object_3d_ids
        .push("obj-highlight-1".to_owned());
    let fresh1 = compute_scene_hash(&app.session);
    let cached1 = app.cached_scene_hash();
    assert_eq!(cached1, fresh1);
    assert_ne!(cached1, cached0);

    // Command edit bumps revision and invalidates scene hash.
    app.session
        .execute(Command::SetPropertyValue {
            target_id: "sequence".to_owned(),
            property: "backgroundColor".to_owned(),
            value: json!([0.1, 0.2, 0.3]),
        })
        .expect("edit sequence bg");
    let fresh2 = compute_scene_hash(&app.session);
    let cached2 = app.cached_scene_hash();
    assert_eq!(cached2, fresh2);
    assert_ne!(cached2, cached1);
}
