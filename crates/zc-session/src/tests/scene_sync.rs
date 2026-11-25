use crate::*;

#[test]
fn multi_straddle_move_locked_track_trim_rate_selection_playback_and_misc_fixes() {
    // 1. Multi-straddle move preserves Image payload on 3rd fragment.
    let mut proj = zc_project::Project::new_cm3_default();
    let img_id = zc_project::Uuid::new_v4();
    let img_clip = zc_project::VideoClip {
        id: img_id,
        name: "Img".to_string(),
        start: 0.0,
        length: 100.0,
        link: None,
        media: Some("hero".to_string()),
        time_curve: zc_project::PropertyTrack::linear_time_curve(100.0, 30),
        layer: zc_project::Layer::new(
            "Img",
            zc_project::LayerKind::Image {
                image: Some("hero".to_string()),
            },
            [1920, 1080],
        ),
        preserved: BTreeMap::new(),
    };
    proj.sequence.video_tracks[0].clips = vec![img_clip];
    let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
    let v1 = session.document().video_tracks[0].id.clone();
    let v2 = session
        .execute(Command::AddVideoTrack { name: None })
        .expect("v2")["track_id"]
        .as_str()
        .unwrap()
        .to_string();
    let m1 = session
        .execute(Command::AddClip {
            track_id: v2.clone(),
            kind: "video".to_string(),
            name: None,
            start: 20.0,
            length: 10.0,
        })
        .expect("m1")["clip_id"]
        .as_str()
        .unwrap()
        .to_string();
    let m2 = session
        .execute(Command::AddClip {
            track_id: v2.clone(),
            kind: "video".to_string(),
            name: None,
            start: 50.0,
            length: 10.0,
        })
        .expect("m2")["clip_id"]
        .as_str()
        .unwrap()
        .to_string();
    session
        .move_clips_atomic(&[(m1, v1.clone(), 20.0), (m2, v1.clone(), 50.0)])
        .expect("multi-straddle move");
    let last_frag = session.project.sequence.video_tracks[0]
        .clips
        .last()
        .expect("3rd fragment");
    assert!(matches!(
        &last_frag.layer.kind,
        zc_project::LayerKind::Image { image: Some(id) } if id == "hero"
    ));

    // 2. RemoveTrack rejects locked tracks.
    session
        .execute(Command::SetTrackLocked {
            track_id: v2.clone(),
            locked: true,
        })
        .expect("lock v2");
    assert_eq!(
        session
            .execute(Command::RemoveTrack { track_id: v2 })
            .expect_err("locked track cannot be removed")
            .code(),
        "track_locked"
    );

    // 3. Clearing assetId sets Image.image and vc.media to None.
    session
        .execute(Command::SetPropertyValue {
            target_id: img_id.to_string(),
            property: "assetId".to_string(),
            value: Value::Null,
        })
        .expect("clear assetId");
    assert!(matches!(
        session.project.sequence.video_tracks[0].clips[0].layer.kind,
        zc_project::LayerKind::Image { image: None }
    ));

    // 4. TrimClipLeft rate-aware clamping (2x speed).
    let mut s_trim = Session::new();
    let c_trim = s_trim.document().video_tracks[0].clips[0].id.clone();
    s_trim.document.video_tracks[0].clips[0].start = 30.0;
    s_trim.document.video_tracks[0].clips[0].length = 30.0;
    s_trim.document.video_tracks[0].clips[0].time_start = 1.0;
    s_trim.document.video_tracks[0].clips[0].time_end = 3.0; // 2x speed at 30fps
    s_trim
        .execute(Command::TrimClipLeft {
            clip_id: c_trim,
            delta_frames: -30.0,
            rate_stretch: false,
        })
        .expect("trim left 2x");
    let trimmed = &s_trim.document().video_tracks[0].clips[0];
    assert!((trimmed.start - 15.0).abs() < 1e-6);
    assert!((trimmed.length - 45.0).abs() < 1e-6);
    assert!(trimmed.time_start.abs() < 1e-6);

    // 5. SelectionState cleanup, mixed-type guard, NaN keyframe rejection, and trim keyframe selection sync.
    let mut sel = SelectionState::default();
    assert_eq!(sel.select_clip("a", None, false), Ok(true));
    assert!(sel.select_clip("b", Some("scene"), true).is_err());
    assert_eq!(sel.select_object_3d("o1", "box", false), Ok(true));
    assert!(!sel.select_keyframe(KeyframeRef::new("a", "opacity", None, f64::NAN), false));
    assert!(sel.keyframes.is_empty());
    assert_eq!(sel.select_clip("a", None, true), Ok(false));
    assert!(sel.object_3d_ids.is_empty());

    let mut s_kfsel = Session::new();
    let ck_id = s_kfsel.document().video_tracks[0].clips[0].id.clone();
    s_kfsel
        .execute(Command::SetKeyframe {
            target_id: ck_id.clone(),
            property: "opacity".to_string(),
            frame: 20.0,
            value: json!(0.5),
            tween: Some(1),
        })
        .expect("kf20");
    s_kfsel
        .selection
        .select_keyframe(KeyframeRef::new(&ck_id, "opacity", None, 20.0), false);
    s_kfsel
        .execute(Command::TrimClipLeft {
            clip_id: ck_id.clone(),
            delta_frames: 5.0,
            rate_stretch: false,
        })
        .expect("trim left shifts selected kf");
    assert!((s_kfsel.selection.keyframes[0].frame - 15.0).abs() < 1e-6);
    s_kfsel
        .execute(Command::DeleteKeyframe {
            target_id: ck_id,
            property: "opacity".to_string(),
            frame: s_kfsel.selection.keyframes[0].frame,
        })
        .expect("delete shifted selected kf must succeed");

    // 5b. Straddle-split discards < 1e-6 slivers instead of creating degenerate clips.
    let mut s_sliver = Session::new();
    let tv = s_sliver.document().video_tracks[0].id.clone();
    let base_c = s_sliver.document().video_tracks[0].clips[0].id.clone();
    s_sliver
        .execute(Command::DeleteClip { clip_id: base_c })
        .expect("del default");
    s_sliver
        .execute(Command::AddClip {
            track_id: tv.clone(),
            kind: "video".to_string(),
            name: None,
            start: 10.0,
            length: 20.0,
        })
        .expect("add base");
    s_sliver
        .execute(Command::AddClip {
            track_id: tv,
            kind: "video".to_string(),
            name: None,
            start: 10.0 + 1e-8,
            length: 10.0,
        })
        .expect("add overlapping with tiny left sliver");
    assert!(
        s_sliver.document().video_tracks[0]
            .clips
            .iter()
            .all(|c| c.length >= 1e-6),
        "straddle-split must discard < 1e-6 slivers"
    );

    // 6. PlaybackState::advance guards infinite fps and tiny negative wrap.
    let mut pb = PlaybackState {
        current_frame: 0.0,
        speed: -1.0,
        looping: true,
        ..PlaybackState::default()
    };
    pb.advance(1.0, f64::INFINITY, 100.0);
    assert!(pb.current_frame.is_finite());
    pb.advance(1e-22, 30.0, 100.0);
    assert!(pb.current_frame >= 0.0 && pb.current_frame < 100.0);

    // 7. Geometry+ kind round-trip, boolean prop sync, and SetFps rounding.
    let mut s_misc = Session::new();
    let sc_id = s_misc.document().video_tracks[0].clips[0].id.clone();
    s_misc
        .execute(Command::AddObject3D {
            clip_id: sc_id,
            parent_id: None,
            kind: "zoidium:geometry-plus/cone".to_string(),
            name: Some("Cone".to_string()),
        })
        .expect("add cone");
    let reloaded_cone =
        Session::from_project(s_misc.project.clone(), zc_plugin::PluginRegistry::new());
    assert!(
        reloaded_cone.document().video_tracks[0].clips[0]
            .objects_3d
            .iter()
            .any(|o| o.kind == "zoidium:geometry-plus/cone")
    );

    s_misc
        .execute(Command::SetPropertyValue {
            target_id: "sequence".to_string(),
            property: "motionBlur.enabled".to_string(),
            value: json!(true),
        })
        .expect("bool motionBlur");
    assert!(s_misc.project.sequence.motion_blur.enabled);

    s_misc
        .execute(Command::SetFps { fps: 29.97 })
        .expect("set fps");
    assert_eq!(s_misc.document().fps, 30.0);
    assert_eq!(s_misc.project.sequence.fps, 30);
}

#[test]
fn repeater_container_kinds_sync_as_group_with_children_and_preserved_type() {
    let mut session = Session::new();
    let scene_id = session.document().video_tracks[0].clips[0].id.clone();

    let group_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "group".to_string(),
            name: Some("Plain Group".to_string()),
        })
        .expect("add plain group")["object_id"]
        .as_str()
        .unwrap()
        .to_string();

    for (repeater_kind, canonical_kind, child_name) in [
        (
            "zoidcraft:repeater/linear_repeater",
            "zoidium:repeater/linear-repeater",
            "ZC Child",
        ),
        (
            "zoidium:repeater/echo_repeater",
            "zoidium:repeater/echo-repeater",
            "Zoidium Child",
        ),
    ] {
        let rep_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: None,
                kind: repeater_kind.to_string(),
                name: Some(repeater_kind.to_string()),
            })
            .expect("add repeater")["object_id"]
            .as_str()
            .unwrap()
            .to_string();

        let child_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: Some(rep_id.clone()),
                kind: "sphere".to_string(),
                name: Some(child_name.to_string()),
            })
            .expect("add child under repeater")["object_id"]
            .as_str()
            .unwrap()
            .to_string();

        let zc_project::LayerKind::Scene3D { objects, .. } =
            &session.project.sequence.video_tracks[0].clips[0].layer.kind
        else {
            panic!("expected Scene3D");
        };
        let proj_rep = objects
            .iter()
            .find(|o| o.id.to_string() == rep_id)
            .expect("repeater in project");
        assert_eq!(
            proj_rep.preserved.get("zoidiumPluginObjectType"),
            Some(&json!(canonical_kind)),
            "repeater preserved[\"zoidiumPluginObjectType\"] must canonicalize {repeater_kind} -> {canonical_kind}"
        );
        assert!(
            proj_rep.material.is_none(),
            "repeater must never attach bogus PBR material"
        );
        assert!(
            !proj_rep.preserved.contains_key("type"),
            "repeater should not store stale preserved[\"type\"]"
        );
        let zc_project::Object3DKind::Group { children, .. } = &proj_rep.kind else {
            panic!(
                "expected {repeater_kind} to sync as Group, got {:?}",
                proj_rep.kind
            );
        };
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id.to_string(), child_id);
        assert_eq!(children[0].name, child_name);
    }

    let zc_project::LayerKind::Scene3D { objects, .. } =
        &session.project.sequence.video_tracks[0].clips[0].layer.kind
    else {
        panic!("expected Scene3D");
    };
    let proj_group = objects
        .iter()
        .find(|o| o.id.to_string() == group_id)
        .expect("plain group in project");
    assert!(
        !proj_group.preserved.contains_key("zoidiumPluginObjectType"),
        "plain group should not gain preserved[\"zoidiumPluginObjectType\"]"
    );
    assert!(
        !proj_group.preserved.contains_key("type"),
        "plain group should not gain preserved[\"type\"]"
    );

    let reloaded = Session::from_project(session.project.clone(), zc_plugin::PluginRegistry::new());
    let reloaded_objs = &reloaded.document().video_tracks[0].clips[0].objects_3d;
    for canonical_kind in [
        "zoidium:repeater/linear-repeater",
        "zoidium:repeater/echo-repeater",
    ] {
        let rep = reloaded_objs
            .iter()
            .find(|o| o.kind == canonical_kind)
            .unwrap_or_else(|| panic!("expected {canonical_kind} after reload"));
        assert!(
            reloaded_objs
                .iter()
                .any(|o| o.parent_id.as_deref() == Some(rep.id.as_str())),
            "child of {canonical_kind} must survive reload"
        );
    }
}

#[test]
fn remove_time_property_on_audio_clip_rebuilds_linear_time_curve() {
    let mut session = Session::new();
    let audio_track_id = session
        .execute(Command::AddAudioTrack {
            name: Some("Audio 1".to_string()),
        })
        .expect("add audio track")["track_id"]
        .as_str()
        .unwrap()
        .to_string();
    let clip_id = session
        .execute(Command::AddClip {
            track_id: audio_track_id,
            kind: "audio".to_string(),
            name: Some("Voice".to_string()),
            start: 0.0,
            length: 60.0,
        })
        .expect("add audio clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Trim left by 15 frames at 30fps so time_start = 0.5, time_end = 2.0, length = 45.0.
    session
        .execute(Command::TrimClipLeft {
            clip_id: clip_id.clone(),
            delta_frames: 15.0,
            rate_stretch: false,
        })
        .expect("trim audio clip left");

    // Animate "time" with a custom keyframe and expression.
    session
        .execute(Command::SetKeyframe {
            target_id: clip_id.clone(),
            property: "time".to_string(),
            frame: 20.0,
            value: json!(9.25),
            tween: Some(1),
        })
        .expect("set audio time keyframe");
    session
        .execute(Command::SetExpression {
            target_id: clip_id.clone(),
            property: "time".to_string(),
            expression: Some("time * 2".to_string()),
        })
        .expect("set audio time expression");

    let proj_before = session
        .project
        .sequence
        .audio_tracks
        .iter()
        .flat_map(|t| &t.clips)
        .find(|c| c.id.to_string() == clip_id)
        .expect("audio clip in project before remove");
    assert!(proj_before.time_curve.animated);
    assert_eq!(
        proj_before.time_curve.expression.as_deref(),
        Some("time * 2")
    );

    // Removing "time" must clear time_animated and rebuild linear ac.time_curve.
    session
        .execute(Command::RemoveProperty {
            target_id: clip_id.clone(),
            property: "time".to_string(),
        })
        .expect("remove audio time property");

    let sclip = session
        .document()
        .audio_tracks
        .iter()
        .flat_map(|t| &t.clips)
        .find(|c| c.id == clip_id)
        .expect("audio clip in document");
    assert!(!sclip.time_animated);
    assert!(!sclip.properties.contains_key("time"));

    let proj_after = session
        .project
        .sequence
        .audio_tracks
        .iter()
        .flat_map(|t| &t.clips)
        .find(|c| c.id.to_string() == clip_id)
        .expect("audio clip in project after remove");
    let tc = &proj_after.time_curve;
    assert!(!tc.animated);
    assert!(tc.expression.is_none());
    assert_eq!(tc.static_behavior, zc_project::StaticBehavior::Linear);
    assert_eq!(tc.value, zc_project::PropertyValue::Number(0.5));
    assert_eq!(tc.keyframes.len(), 2);
    assert!((tc.keyframes[0].frame - 0.0).abs() < 1e-9);
    assert_eq!(
        tc.keyframes[0].value,
        zc_project::PropertyValue::Number(0.5)
    );
    assert!((tc.keyframes[1].frame - 45.0).abs() < 1e-9);
    assert_eq!(
        tc.keyframes[1].value,
        zc_project::PropertyValue::Number(2.0)
    );
}

#[test]
fn stroke_only_and_grouped_2d_shapes_and_text_never_gain_fake_white_fill() {
    let mut proj = zc_project::Project::new_cm3_default();
    let stroke_only_shape = zc_project::VideoClip {
        id: zc_project::Uuid::new_v4(),
        name: "StrokeOnlyShape".to_string(),
        start: 0.0,
        length: 60.0,
        link: None,
        media: None,
        time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
        layer: zc_project::Layer::new(
            "StrokeOnlyShape",
            zc_project::LayerKind::Shape2D {
                shapes: vec![zc_project::Shape2DNode {
                    id: zc_project::Uuid::new_v4(),
                    name: "Path".to_string(),
                    kind: zc_project::Shape2DKind::Path {
                        path: json!({"preset": "rectangle"}),
                    },
                    position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                    rotation: zc_project::PropertyTrack::number(0.0),
                    scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                    operations: vec![zc_project::DrawOp::Stroke {
                        color: zc_project::PropertyTrack::color([0.1, 0.9, 0.3]),
                        size: zc_project::PropertyTrack::number(6.0),
                        opacity: zc_project::PropertyTrack::number(1.0),
                    }],
                    preserved: BTreeMap::new(),
                }],
            },
            [1920, 1080],
        ),
        preserved: BTreeMap::new(),
    };
    let stroke_only_preset = zc_project::VideoClip {
        id: zc_project::Uuid::new_v4(),
        name: "StrokeOnlyPreset".to_string(),
        start: 60.0,
        length: 60.0,
        link: None,
        media: None,
        time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
        layer: zc_project::Layer::new(
            "StrokeOnlyPreset",
            zc_project::LayerKind::PresetShape2D {
                preset: zc_project::PresetShapeKind::Ellipse,
                shapes: vec![zc_project::Shape2DNode {
                    id: zc_project::Uuid::new_v4(),
                    name: "Ring".to_string(),
                    kind: zc_project::Shape2DKind::Group {
                        children: Vec::new(),
                    },
                    position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                    rotation: zc_project::PropertyTrack::number(0.0),
                    scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                    operations: vec![zc_project::DrawOp::Stroke {
                        color: zc_project::PropertyTrack::color([1.0, 0.5, 0.0]),
                        size: zc_project::PropertyTrack::number(4.0),
                        opacity: zc_project::PropertyTrack::number(0.8),
                    }],
                    preserved: BTreeMap::new(),
                }],
            },
            [1920, 1080],
        ),
        preserved: BTreeMap::new(),
    };
    let outline_only_text = zc_project::VideoClip {
        id: zc_project::Uuid::new_v4(),
        name: "OutlineText".to_string(),
        start: 120.0,
        length: 60.0,
        link: None,
        media: None,
        time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
        layer: zc_project::Layer::new(
            "OutlineText",
            zc_project::LayerKind::Text2D {
                text: "OUTLINE".to_string(),
                font: None,
                size: 96,
                shapes: vec![zc_project::Shape2DNode {
                    id: zc_project::Uuid::new_v4(),
                    name: "TextStroke".to_string(),
                    kind: zc_project::Shape2DKind::Group {
                        children: Vec::new(),
                    },
                    position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                    rotation: zc_project::PropertyTrack::number(0.0),
                    scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                    operations: vec![zc_project::DrawOp::Stroke {
                        color: zc_project::PropertyTrack::color([0.0, 0.5, 1.0]),
                        size: zc_project::PropertyTrack::number(3.0),
                        opacity: zc_project::PropertyTrack::number(1.0),
                    }],
                    preserved: BTreeMap::new(),
                }],
            },
            [1920, 1080],
        ),
        preserved: BTreeMap::new(),
    };
    let grouped_shape_id = zc_project::Uuid::new_v4();
    let grouped_shape = zc_project::VideoClip {
        id: grouped_shape_id,
        name: "GroupedShape".to_string(),
        start: 180.0,
        length: 60.0,
        link: None,
        media: None,
        time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
        layer: zc_project::Layer::new(
            "GroupedShape",
            zc_project::LayerKind::Shape2D {
                shapes: vec![zc_project::Shape2DNode {
                    id: zc_project::Uuid::new_v4(),
                    name: "OuterGroup".to_string(),
                    kind: zc_project::Shape2DKind::Group {
                        children: vec![zc_project::Shape2DNode {
                            id: zc_project::Uuid::new_v4(),
                            name: "InnerPath".to_string(),
                            kind: zc_project::Shape2DKind::Path {
                                path: json!({"preset": "ellipse"}),
                            },
                            position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                            rotation: zc_project::PropertyTrack::number(0.0),
                            scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                            operations: vec![zc_project::DrawOp::Fill {
                                color: zc_project::PropertyTrack::color([0.2, 0.8, 0.4]),
                            }],
                            preserved: BTreeMap::new(),
                        }],
                    },
                    position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                    rotation: zc_project::PropertyTrack::number(0.0),
                    scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                    operations: Vec::new(),
                    preserved: BTreeMap::new(),
                }],
            },
            [1920, 1080],
        ),
        preserved: BTreeMap::new(),
    };
    let empty_shape = zc_project::VideoClip {
        id: zc_project::Uuid::new_v4(),
        name: "EmptyShape".to_string(),
        start: 240.0,
        length: 30.0,
        link: None,
        media: None,
        time_curve: zc_project::PropertyTrack::linear_time_curve(30.0, 30),
        layer: zc_project::Layer::new(
            "EmptyShape",
            zc_project::LayerKind::Shape2D { shapes: Vec::new() },
            [1920, 1080],
        ),
        preserved: BTreeMap::new(),
    };

    let orig_kinds = [
        stroke_only_shape.layer.kind.clone(),
        stroke_only_preset.layer.kind.clone(),
        outline_only_text.layer.kind.clone(),
        grouped_shape.layer.kind.clone(),
        empty_shape.layer.kind.clone(),
    ];
    let stroke_only_shape_id = stroke_only_shape.id.to_string();
    proj.sequence.video_tracks[0].clips = vec![
        stroke_only_shape,
        stroke_only_preset,
        outline_only_text,
        grouped_shape,
        empty_shape,
    ];

    let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
    // Perform an unrelated edit to trigger sync_document_to_project().
    session
        .execute(Command::SetProjectName {
            name: "Unrelated Edit".to_string(),
        })
        .expect("rename");

    for (idx, expected) in orig_kinds.iter().enumerate() {
        assert_eq!(
            &session.project.sequence.video_tracks[0].clips[idx]
                .layer
                .kind,
            expected,
            "clip {idx} LayerKind must remain lossless after unrelated edit"
        );
    }

    // Editing fillColor on the grouped shape updates the nested DrawOp::Fill in place
    // without injecting a spurious top-level DrawOp::Fill on OuterGroup.
    session
        .execute(Command::SetPropertyValue {
            target_id: grouped_shape_id.to_string(),
            property: "fillColor".to_string(),
            value: json!([0.9, 0.1, 0.2]),
        })
        .expect("edit nested fillColor");
    if let zc_project::LayerKind::Shape2D { shapes } =
        &session.project.sequence.video_tracks[0].clips[3].layer.kind
    {
        assert!(
            shapes[0].operations.is_empty(),
            "OuterGroup must not receive a duplicate top-level Fill"
        );
        let zc_project::Shape2DKind::Group { children } = &shapes[0].kind else {
            panic!("expected Group");
        };
        let zc_project::DrawOp::Fill { color } = &children[0].operations[0] else {
            panic!("expected nested Fill");
        };
        assert_eq!(
            color.value,
            zc_project::PropertyValue::Color([0.9, 0.1, 0.2])
        );
    } else {
        panic!("expected Shape2D");
    }

    // Explicitly setting fillColor on the stroke-only shape adds a Fill, and Undo removes it.
    session
        .execute(Command::SetPropertyValue {
            target_id: stroke_only_shape_id,
            property: "fillColor".to_string(),
            value: json!([0.2, 0.3, 0.4]),
        })
        .expect("explicitly add fill");
    assert!(matches!(
        &session.project.sequence.video_tracks[0].clips[0].layer.kind,
        zc_project::LayerKind::Shape2D { shapes }
            if shapes[0].operations.iter().any(|op| matches!(op, zc_project::DrawOp::Fill { .. }))
    ));
    session.execute(Command::Undo).expect("undo add fill");
    assert_eq!(
        session.project.sequence.video_tracks[0].clips[0].layer.kind,
        orig_kinds[0]
    );
}

#[test]
fn custom_shader_uniforms_preserve_order_bezier_channels_and_never_duplicate_into_params() {
    let mut proj = zc_project::Project::new_cm3_default();
    let make_bezier_channel =
        |v0: f64, v1: f64, cp: [[f64; 2]; 2], expr: Option<&str>| zc_project::PropertyTrack {
            animated: true,
            expression: expr.map(str::to_owned),
            static_behavior: zc_project::StaticBehavior::Constant,
            value: zc_project::PropertyValue::Number(v0),
            keyframes: vec![
                zc_project::PropertyKeyframe::from_cm3_tween(
                    0.0,
                    zc_project::PropertyValue::Number(v0),
                    0x101,
                    cp,
                    false,
                ),
                zc_project::PropertyKeyframe::from_cm3_tween(
                    30.0,
                    zc_project::PropertyValue::Number(v1),
                    0x101,
                    [[-6.0, -0.2], [6.0, 0.2]],
                    false,
                ),
            ],
            channels: Vec::new(),
        };
    let color_track = zc_project::PropertyTrack {
        animated: true,
        expression: None,
        static_behavior: zc_project::StaticBehavior::Constant,
        value: zc_project::PropertyValue::Color([1.0, 0.2, 0.0]),
        keyframes: Vec::new(),
        channels: vec![
            make_bezier_channel(1.0, 0.0, [[-4.0, 0.1], [5.0, -0.1]], None),
            make_bezier_channel(0.2, 0.8, [[-2.0, 0.3], [7.0, 0.4]], None),
            make_bezier_channel(0.0, 1.0, [[-8.0, 0.0], [3.0, 0.5]], Some("sin(time)")),
        ],
    };
    let mut color_meta = BTreeMap::new();
    color_meta.insert("min".to_string(), json!(0.0));
    color_meta.insert("max".to_string(), json!(1.0));
    let fx_id = zc_project::Uuid::new_v4();
    let mut stale_params = BTreeMap::new();
    // Simulate a stale flattened copy from an old save; load must prefer custom_properties.
    stale_params.insert(
        "Color".to_string(),
        zc_project::PropertyTrack::color([0.0, 0.0, 0.0]),
    );
    let fx = zc_project::EffectInstance {
        id: fx_id,
        name: "CustomFX".to_string(),
        kind: zc_project::EffectKind::CustomShader {
            frag_shader: "void main() { gl_FragColor = vec4(1.0); }".to_string(),
            custom_properties: vec![
                zc_project::CustomProperty {
                    name: "Zeta".to_string(),
                    prop_type: Some(0),
                    track: zc_project::PropertyTrack::number(1.5),
                    metadata: BTreeMap::new(),
                },
                zc_project::CustomProperty {
                    name: "Alpha".to_string(),
                    prop_type: Some(0),
                    track: zc_project::PropertyTrack::number(0.75),
                    metadata: BTreeMap::new(),
                },
                zc_project::CustomProperty {
                    name: "Color".to_string(),
                    prop_type: Some(5),
                    track: color_track.clone(),
                    metadata: color_meta.clone(),
                },
            ],
        },
        enabled: zc_project::PropertyTrack::option(1),
        params: stale_params,
        preserved: BTreeMap::new(),
    };
    proj.sequence.video_tracks[0].clips[0]
        .layer
        .effects
        .push(fx);

    let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
    // Unrelated edit + editing only Zeta must preserve order, Color's Bezier channels, and empty fx.params.
    session
        .execute(Command::SetProjectName {
            name: "After Unrelated".to_string(),
        })
        .expect("rename");
    session
        .execute(Command::SetPropertyValue {
            target_id: fx_id.to_string(),
            property: "Zeta".to_string(),
            value: json!(2.5),
        })
        .expect("edit Zeta");

    let assert_shader_intact = |p: &zc_project::Project| {
        let synced_fx = &p.sequence.video_tracks[0].clips[0].layer.effects[0];
        assert!(
            synced_fx.params.is_empty(),
            "CustomShader uniforms must never duplicate into fx.params: {:?}",
            synced_fx.params
        );
        let zc_project::EffectKind::CustomShader {
            custom_properties, ..
        } = &synced_fx.kind
        else {
            panic!("expected CustomShader");
        };
        let names: Vec<&str> = custom_properties.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["Zeta", "Alpha", "Color"]);
        assert_eq!(
            custom_properties[0].track.value,
            zc_project::PropertyValue::Number(2.5)
        );
        assert_eq!(custom_properties[2].prop_type, Some(5));
        assert_eq!(custom_properties[2].metadata, color_meta);
        assert_eq!(custom_properties[2].track, color_track);
    };
    assert_shader_intact(&session.project);

    // Reload from project and do another unrelated edit -> still lossless!
    let mut reloaded =
        Session::from_project(session.project.clone(), zc_plugin::PluginRegistry::new());
    reloaded
        .execute(Command::SetProjectName {
            name: "After Reload".to_string(),
        })
        .expect("rename after reload");
    assert_shader_intact(&reloaded.project);
}

#[test]
fn scene3d_preserved_shadows_and_complex_background_color_survive_sync() {
    let mut proj = zc_project::Project::new_cm3_default();
    let complex_bg = json!({
        "animated": true,
        "value": [0.1, 0.2, 0.3, 0.75],
        "keyframes": [{"frame": 0, "value": [0.1, 0.2, 0.3, 0.75]}]
    });
    proj.sequence.video_tracks[0].clips[0]
        .layer
        .preserved
        .insert("property.shadows".to_string(), json!(0));
    proj.sequence.video_tracks[0].clips[0]
        .layer
        .preserved
        .insert("property.backgroundColor".to_string(), complex_bg.clone());

    let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
    let props = &session.document().video_tracks[0].clips[0].properties;
    assert_eq!(prop_f64_static(props, "shadows"), Some(0.0));
    assert_eq!(prop_f64_static(props, "sceneBackground"), Some(1.0));
    assert_eq!(prop_f64_static(props, "backdropOpacity"), Some(0.75));

    session
        .execute(Command::SetProjectName {
            name: "Keep Preserved".to_string(),
        })
        .expect("rename");
    let preserved = &session.project.sequence.video_tracks[0].clips[0]
        .layer
        .preserved;
    assert_eq!(preserved.get("property.shadows"), Some(&json!(0)));
    assert_eq!(preserved.get("property.backgroundColor"), Some(&complex_bg));

    // Explicit edit to shadows overrides cleanly.
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    session
        .execute(Command::SetPropertyValue {
            target_id: clip_id,
            property: "shadows".to_string(),
            value: json!(1),
        })
        .expect("enable shadows");
    assert_eq!(
        session.project.sequence.video_tracks[0].clips[0]
            .layer
            .preserved
            .get("shadows"),
        Some(&json!(1))
    );
}

#[test]
fn complex_3d_objects_materials_constraints_round_trip_and_sync_edits() {
    let mut proj = zc_project::Project::new_cm3_default();
    let cam_id = zc_project::Uuid::new_v4();
    let mut cam = zc_project::Object3D::new(
        "MainCam",
        zc_project::Object3DKind::Camera(Box::new(zc_project::CameraData {
            kind: zc_project::CameraKind::Perspective,
            fov: zc_project::PropertyTrack::number(52.0),
            near: 0.25,
            far: 2500.0,
            zoom: zc_project::PropertyTrack::number(1.2),
            shake_amplitude: zc_project::PropertyTrack::number(0.45),
            shake_speed: zc_project::PropertyTrack::number(18.0),
            shake: zc_project::CameraShakeParams {
                mode: 1,
                amplitude: [5.0, 6.0, 7.0, 8.0],
                frequency: [0.2, 0.3, 0.4, 0.5],
                phase: [1.0, 2.0, 3.0, 4.0],
                noise: [0.1, 0.2, 0.3, 0.4],
                smooth: [2.0, 2.0, 2.0, 2.0],
            },
            dof: Some(zc_project::CameraDofParams {
                enabled: false,
                focus_distance: zc_project::PropertyTrack::number(120.0),
                aperture: zc_project::PropertyTrack::number(4.0),
                focus_area_width: zc_project::PropertyTrack::number(15.0),
                near_blur: zc_project::PropertyTrack::number(0.6),
                far_blur: zc_project::PropertyTrack::number(0.9),
                ..Default::default()
            }),
            vibrate: Some(zc_project::CameraVibrateParams {
                enabled: true,
                seed: 42,
                position_amplitude: zc_project::PropertyTrack::vec3([1.0, 2.0, 3.0]),
                rotation_amplitude: zc_project::PropertyTrack::vec3([0.1, 0.2, 0.3]),
                frequency: zc_project::PropertyTrack::number(3.5),
                ..Default::default()
            }),
            ..Default::default()
        })),
    );
    cam.id = cam_id;
    cam.euler_order = zc_project::EulerOrder::Yxz;
    cam.constraints.track_to = Some(zc_project::TrackToConstraint {
        target_id: "target-1".to_string(),
        track_axis: 5,
        up_axis: 2,
        influence: zc_project::PropertyTrack::number(0.8),
    });
    cam.constraints.follow = Some(zc_project::FollowConstraint {
        target_id: "target-2".to_string(),
        space: 1,
        follow_position: true,
        follow_rotation: false,
        follow_scale: false,
        position_offset: zc_project::PropertyTrack::vec3([1.0, 2.0, 3.0]),
        rotation_offset: zc_project::PropertyTrack::vec3([0.1, 0.2, 0.3]),
        scale_offset: zc_project::PropertyTrack::vec3([1.5, 1.5, 1.5]),
        fallback_pose: Some(json!({"p":[0,0,0]})),
    });

    let text_id = zc_project::Uuid::new_v4();
    let mut text = zc_project::Object3D::new(
        "HeroText",
        zc_project::Object3DKind::Text3D(Box::new(zc_project::Text3DData {
            text: "HELLO".to_string(),
            font: Some("custom_font.ttf".to_string()),
            size: [30.0, 6.0],
            detail: 120,
            bevel: true,
            bevel_thickness: 0.5,
            bevel_size: 0.8,
            position_mode: zc_project::Text3DAlign::Right,
            text_scale_mode: zc_project::CompatMode::Current,
            bevel_tension: 0.75,
            ..Default::default()
        })),
    );
    text.id = text_id;
    text.material = Some(zc_project::MaterialSpec::Unknown {
        material_type: "vendor:mystery".to_string(),
        payload: json!({"custom": 99}),
    });

    let box_id = zc_project::Uuid::new_v4();
    let mut bx = zc_project::Object3D::new(
        "Cube",
        zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Box {
            size: [12.0, 24.0, 36.0],
        }),
    );
    bx.id = box_id;
    let pbr = zc_project::CustomPbrMaterial {
        opacity: zc_project::PropertyTrack::number(0.5),
        transparent: false, // Must NOT be silently flipped to true!
        side: 2,
        normal_map: Some("norm_asset".to_string()),
        normal_scale: 2.2,
        reflection: true,
        ..Default::default()
    };
    bx.material = Some(zc_project::MaterialSpec::CustomPbr(Box::new(pbr)));

    let cyl_id = zc_project::Uuid::new_v4();
    let mut cyl = zc_project::Object3D::new(
        "OpenCyl",
        zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Cylinder {
            top_radius: 3.0,
            bottom_radius: 7.0,
            height: 20.0,
            detail: 32,
            open_ended: true,
        }),
    );
    cyl.id = cyl_id;
    let pbr_plus = zc_project::PbrPlusMaterial {
        opacity: zc_project::PropertyTrack::number(0.4),
        transparent: false,
        side: 1,
        roughness_map: Some("rough_map".to_string()),
        alpha_map: Some("alpha_map".to_string()),
        reflection_intensity: 0.7,
        ..Default::default()
    };
    cyl.material = Some(zc_project::MaterialSpec::PbrPlus(Box::new(pbr_plus)));

    let grp_id = zc_project::Uuid::new_v4();
    let mut grp = zc_project::Object3D::new(
        "ReflGroup",
        zc_project::Object3DKind::Group {
            reflection_visibility: 2,
            custom_properties: vec![zc_project::CustomProperty {
                name: "speed".to_string(),
                prop_type: Some(0),
                track: zc_project::PropertyTrack::number(9.0),
                metadata: BTreeMap::new(),
            }],
            children: Vec::new(),
        },
    );
    grp.id = grp_id;

    let mdl_id = zc_project::Uuid::new_v4();
    let mut mdl = zc_project::Object3D::new(
        "Ship",
        zc_project::Object3DKind::Model(Box::new(zc_project::ModelData {
            preset_type: None,
            asset_id: Some("ship.glb".to_string()),
            model_properties: BTreeMap::new(),
        })),
    );
    mdl.id = mdl_id;
    let mc = zc_project::MatcapMaterial {
        opacity: zc_project::PropertyTrack::number(0.6),
        transparent: false,
        side: 2,
        ..Default::default()
    };
    mdl.material = Some(zc_project::MaterialSpec::Matcap(Box::new(mc)));

    let orig_objs = vec![cam, text, bx, cyl, grp, mdl];
    proj.sequence.video_tracks[0].clips[0].layer.kind = zc_project::LayerKind::Scene3D {
        objects: orig_objs.clone(),
        shadow_mode: zc_project::CompatMode::Current,
    };

    let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
    // Verify Shape3D kind strings match inspector ("box", "cylinder").
    let s_objs = &session.document().video_tracks[0].clips[0].objects_3d;
    assert_eq!(s_objs[2].kind, "box");
    assert_eq!(s_objs[3].kind, "cylinder");

    // Unrelated edit must preserve EVERY field of EVERY 3D object losslessly.
    session
        .execute(Command::SetProjectName {
            name: "Lossless 3D".to_string(),
        })
        .expect("rename");
    let zc_project::LayerKind::Scene3D { objects, .. } =
        &session.project.sequence.video_tracks[0].clips[0].layer.kind
    else {
        panic!("expected Scene3D");
    };
    assert_eq!(objects, &orig_objs);

    // Now verify edits to previously-dropped fields actually sync to session.project!
    session
        .execute(Command::SetPropertyValue {
            target_id: cam_id.to_string(),
            property: "eulerOrder".to_string(),
            value: json!(5),
        })
        .expect("set eulerOrder");
    session
        .execute(Command::SetPropertyValue {
            target_id: cam_id.to_string(),
            property: "shake.mode".to_string(),
            value: json!(0),
        })
        .expect("set shake.mode");
    session
        .execute(Command::SetPropertyValue {
            target_id: text_id.to_string(),
            property: "positionMode".to_string(),
            value: json!(4),
        })
        .expect("set positionMode");
    session
        .execute(Command::SetPropertyValue {
            target_id: text_id.to_string(),
            property: "font".to_string(),
            value: json!("other.otf"),
        })
        .expect("set font");
    session
        .execute(Command::SetPropertyValue {
            target_id: text_id.to_string(),
            property: "material.type".to_string(),
            value: json!(3),
        })
        .expect("replace Unknown material with CustomPbr");
    session
        .execute(Command::SetPropertyValue {
            target_id: box_id.to_string(),
            property: "box_size".to_string(),
            value: json!([50.0, 60.0, 70.0]),
        })
        .expect("set box_size");
    session
        .execute(Command::SetPropertyValue {
            target_id: cyl_id.to_string(),
            property: "cylinder_openEnded".to_string(),
            value: json!(1),
        })
        .expect("close cylinder");
    session
        .execute(Command::SetPropertyValue {
            target_id: grp_id.to_string(),
            property: "reflectionVisibility".to_string(),
            value: json!(1),
        })
        .expect("set reflectionVisibility");
    session
        .execute(Command::SetPropertyValue {
            target_id: mdl_id.to_string(),
            property: "customGeometry".to_string(),
            value: json!("fighter.glb"),
        })
        .expect("set model customGeometry");
    session
        .execute(Command::SetPropertyValue {
            target_id: mdl_id.to_string(),
            property: "material.side".to_string(),
            value: json!(0),
        })
        .expect("set matcap side");

    let zc_project::LayerKind::Scene3D { objects, .. } =
        &session.project.sequence.video_tracks[0].clips[0].layer.kind
    else {
        panic!("expected Scene3D");
    };
    assert_eq!(objects[0].euler_order, zc_project::EulerOrder::Zyx);
    if let zc_project::Object3DKind::Camera(c) = &objects[0].kind {
        assert_eq!(c.shake.mode, 0);
    } else {
        panic!("expected Camera");
    }
    if let zc_project::Object3DKind::Text3D(t) = &objects[1].kind {
        assert_eq!(t.position_mode, zc_project::Text3DAlign::Bottom);
        assert_eq!(t.font.as_deref(), Some("other.otf"));
        assert_eq!(t.detail, 120);
    } else {
        panic!("expected Text3D");
    }
    assert!(matches!(
        objects[1].material,
        Some(zc_project::MaterialSpec::CustomPbr(_))
    ));
    assert!(matches!(
        objects[2].kind,
        zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Box { size })
            if size == [50.0, 60.0, 70.0]
    ));
    assert!(matches!(
        objects[3].kind,
        zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Cylinder {
            open_ended: false,
            ..
        })
    ));
    assert!(matches!(
        objects[4].kind,
        zc_project::Object3DKind::Group {
            reflection_visibility: 1,
            ..
        }
    ));
    assert!(matches!(
        &objects[5].kind,
        zc_project::Object3DKind::Model(m) if m.asset_id.as_deref() == Some("fighter.glb")
    ));
    assert!(matches!(
        &objects[5].material,
        Some(zc_project::MaterialSpec::Matcap(mc)) if mc.side == 0 && !mc.transparent
    ));
}

#[test]
fn freeze_frame_time_controls_order_audio_mute_cloned_uuids_and_selection_domains() {
    let mut proj = zc_project::Project::new_cm3_default();
    // 1. Single-keyframe freeze-frame time_curve on video & audio clips.
    let freeze_curve = zc_project::PropertyTrack {
        animated: false,
        expression: None,
        static_behavior: zc_project::StaticBehavior::Linear,
        value: zc_project::PropertyValue::Number(5.0),
        keyframes: vec![zc_project::PropertyKeyframe::linear(
            0.0,
            zc_project::PropertyValue::Number(5.0),
        )],
        channels: Vec::new(),
    };
    proj.sequence.video_tracks[0].clips[0].time_curve = freeze_curve.clone();

    // 2. Duplicate-named & non-alphabetical Sequence.controls with non-scalar kinds.
    proj.sequence.controls = vec![
        zc_project::SequenceControl {
            id: "ctl-z".to_string(),
            name: "Zebra".to_string(),
            kind: zc_project::SequenceControlKind::Color,
            track: zc_project::PropertyTrack::color([0.1, 0.2, 0.3]),
            preserved: BTreeMap::new(),
        },
        zc_project::SequenceControl {
            id: "ctl-g1".to_string(),
            name: "Gain".to_string(),
            kind: zc_project::SequenceControlKind::Slider,
            track: zc_project::PropertyTrack::number(1.0),
            preserved: BTreeMap::new(),
        },
        zc_project::SequenceControl {
            id: "ctl-g2".to_string(),
            name: "Gain".to_string(),
            kind: zc_project::SequenceControlKind::Point3D,
            track: zc_project::PropertyTrack::vec3([1.0, 2.0, 3.0]),
            preserved: BTreeMap::new(),
        },
    ];

    // 3. AudioTrack with (enabled: false, muted: false) and (enabled: true, muted: true) + media_length.
    let mut media_pres = BTreeMap::new();
    media_pres.insert("durationSeconds".to_string(), json!(4.0));
    proj.media.push(zc_project::MediaItem {
        id: zc_project::Uuid::new_v4(),
        name: "beat.wav".to_string(),
        asset_id: Some("beat_asset".to_string()),
        kind: zc_project::AssetKind::Av,
        preserved: media_pres,
    });
    proj.sequence.audio_tracks = vec![
        zc_project::AudioTrack {
            id: zc_project::Uuid::new_v4(),
            name: "A1".to_string(),
            enabled: false,
            muted: false,
            solo: false,
            locked: false,
            clips: vec![zc_project::AudioClip {
                id: zc_project::Uuid::new_v4(),
                name: "Beat".to_string(),
                start: 0.0,
                length: 60.0,
                link: None,
                media: Some("beat_asset".to_string()),
                time_curve: freeze_curve.clone(),
                audio: zc_project::AudioClipData::default(),
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        },
        zc_project::AudioTrack {
            id: zc_project::Uuid::new_v4(),
            name: "A2".to_string(),
            enabled: true,
            muted: true,
            solo: false,
            locked: false,
            clips: Vec::new(),
            preserved: BTreeMap::new(),
        },
    ];

    // 4. Composite clip with nested Scene3D + EffectGroup children for duplicate UUID check.
    let comp_clip_id = zc_project::Uuid::new_v4();
    let nested_child_obj_id = zc_project::Uuid::new_v4();
    let nested_layer_id = zc_project::Uuid::new_v4();
    let fx_child_id = zc_project::Uuid::new_v4();
    let mut comp_layer = zc_project::Layer::new(
        "Comp",
        zc_project::LayerKind::Composite {
            layers: vec![zc_project::Layer {
                id: nested_layer_id,
                name: "NestedScene".to_string(),
                kind: zc_project::LayerKind::Scene3D {
                    objects: vec![zc_project::Object3D {
                        id: nested_child_obj_id,
                        ..zc_project::Object3D::new_perspective_camera()
                    }],
                    shadow_mode: zc_project::CompatMode::Current,
                },
                resolution: [1920, 1080],
                position: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                scale: zc_project::PropertyTrack::vec2([1.0, 1.0]),
                rotation: zc_project::PropertyTrack::number(0.0),
                anchor: zc_project::PropertyTrack::vec2([0.0, 0.0]),
                opacity: zc_project::PropertyTrack::number(1.0),
                blending: "BLEND_SRC_OVER".to_string(),
                effects: Vec::new(),
                preserved: BTreeMap::new(),
            }],
        },
        [1920, 1080],
    );
    comp_layer.effects.push(zc_project::EffectInstance {
        id: zc_project::Uuid::new_v4(),
        name: "GroupFX".to_string(),
        kind: zc_project::EffectKind::EffectGroup {
            custom_properties: Vec::new(),
            children: vec![zc_project::EffectInstance {
                id: fx_child_id,
                name: "InnerBlur".to_string(),
                kind: zc_project::EffectKind::Standard("boxblur".to_string()),
                enabled: zc_project::PropertyTrack::option(1),
                params: BTreeMap::new(),
                preserved: BTreeMap::new(),
            }],
        },
        enabled: zc_project::PropertyTrack::option(1),
        params: BTreeMap::new(),
        preserved: BTreeMap::new(),
    });
    proj.sequence.video_tracks[0]
        .clips
        .push(zc_project::VideoClip {
            id: comp_clip_id,
            name: "CompClip".to_string(),
            start: 200.0,
            length: 60.0,
            link: None,
            media: None,
            time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
            layer: comp_layer,
            preserved: BTreeMap::new(),
        });

    let mut session = Session::from_project(proj, zc_plugin::PluginRegistry::new());
    assert_eq!(
        session.document().audio_tracks[0].clips[0].media_length,
        Some(120.0)
    );
    let rev0 = session.revision();

    // Unrelated edit.
    session
        .execute(Command::SetProjectName {
            name: "Synced".to_string(),
        })
        .expect("rename");
    assert!(session.revision() > rev0);

    // Freeze-frame time curves preserved verbatim.
    assert_eq!(
        session.project.sequence.video_tracks[0].clips[0].time_curve,
        freeze_curve
    );
    assert_eq!(
        session.project.sequence.audio_tracks[0].clips[0].time_curve,
        freeze_curve
    );

    // Controls order, duplicate names, and kinds preserved verbatim.
    let ctl_summary: Vec<(&str, &str, zc_project::SequenceControlKind)> = session
        .project
        .sequence
        .controls
        .iter()
        .map(|c| (c.id.as_str(), c.name.as_str(), c.kind))
        .collect();
    assert_eq!(
        ctl_summary,
        vec![
            ("ctl-z", "Zebra", zc_project::SequenceControlKind::Color),
            ("ctl-g1", "Gain", zc_project::SequenceControlKind::Slider),
            ("ctl-g2", "Gain", zc_project::SequenceControlKind::Point3D),
        ]
    );

    // AudioTrack (enabled, muted) preserved independently.
    assert!(!session.project.sequence.audio_tracks[0].enabled);
    assert!(!session.project.sequence.audio_tracks[0].muted);
    assert!(session.project.sequence.audio_tracks[1].enabled);
    assert!(session.project.sequence.audio_tracks[1].muted);

    // Duplicating Composite clip regenerates nested layer, nested 3D object, and EffectGroup child UUIDs.
    let dup_res = session
        .execute(Command::DuplicateClip {
            clip_id: comp_clip_id.to_string(),
        })
        .expect("duplicate composite clip");
    let dup_id = dup_res["clip_id"].as_str().unwrap().to_string();
    let dup_clip = session.project.sequence.video_tracks[0]
        .clips
        .iter()
        .find(|c| c.id.to_string() == dup_id)
        .expect("duplicated composite clip");
    let orig_clip = session.project.sequence.video_tracks[0]
        .clips
        .iter()
        .find(|c| c.id == comp_clip_id)
        .expect("original composite clip");
    assert_ne!(dup_clip.layer.id, orig_clip.layer.id);
    if let zc_project::LayerKind::Composite { layers } = &dup_clip.layer.kind {
        assert_ne!(layers[0].id, nested_layer_id);
        if let zc_project::LayerKind::Scene3D { objects, .. } = &layers[0].kind {
            assert_ne!(objects[0].id, nested_child_obj_id);
        } else {
            panic!("expected nested Scene3D");
        }
    } else {
        panic!("expected Composite");
    }
    if let zc_project::EffectKind::EffectGroup { children, .. } = &dup_clip.layer.effects[0].kind {
        assert_ne!(children[0].id, fx_child_id);
    } else {
        panic!("expected EffectGroup");
    }

    // SelectionState clears conflicting sub-selections across domains.
    let mut sel = SelectionState::default();
    sel.select_object_3d("obj-1", "box", false).expect("obj");
    sel.select_keyframe(KeyframeRef::new("obj-1", "position", None, 0.0), false);
    sel.select_effect("fx-1", "bloom", false).expect("fx");
    assert!(sel.object_3d_ids.is_empty());
    assert!(sel.keyframes.is_empty());
    assert_eq!(sel.effect_ids, vec!["fx-1"]);
    sel.select_keyframe(KeyframeRef::new("fx-1", "amount", None, 5.0), false);
    sel.select_object_3d("obj-2", "sphere", false)
        .expect("obj2");
    assert!(sel.effect_ids.is_empty());
    assert!(sel.keyframes.is_empty());
    assert_eq!(sel.object_3d_ids, vec!["obj-2"]);
}

#[test]
fn repeater_zoidium_plugin_object_type_roundtrips_through_pz_and_migrates_legacy_type() {
    let mut session = Session::new();
    let scene_id = session.document().video_tracks[0].clips[0].id.clone();

    let rep_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "zoidium:repeater/echo-repeater".to_string(),
            name: Some("Echo Repeater".to_string()),
        })
        .expect("add echo repeater")["object_id"]
        .as_str()
        .unwrap()
        .to_string();

    session
        .execute(Command::AddObject3D {
            clip_id: scene_id,
            parent_id: Some(rep_id),
            kind: "sphere".to_string(),
            name: Some("Echo Child".to_string()),
        })
        .expect("add child under echo repeater");

    // Export to .pz and re-import: must emit the real repeater plugin type, not plain CM3 group 5.
    let pz_bytes = zc_project::export_pz_to_bytes(&session.project).expect("export pz");
    let imported = zc_project::import_pz_from_bytes(&pz_bytes).expect("import pz");
    let zc_project::LayerKind::Scene3D { objects, .. } = &imported.project.sequence.video_tracks[0]
        .clips[0]
        .layer
        .kind
    else {
        panic!("expected Scene3D after pz import");
    };
    let imported_rep = objects
        .iter()
        .find(|o| o.name == "Echo Repeater")
        .expect("re-imported echo repeater");
    assert_eq!(
        imported_rep.preserved.get("zoidiumPluginObjectType"),
        Some(&json!("zoidium:repeater/echo-repeater")),
        ".pz round-trip must preserve zoidiumPluginObjectType"
    );
    assert!(
        !imported_rep.preserved.contains_key("type"),
        ".pz round-trip must not leak stale preserved[\"type\"]"
    );
    let reloaded = Session::from_project(imported.project, zc_plugin::PluginRegistry::new());
    let reloaded_objs = &reloaded.document().video_tracks[0].clips[0].objects_3d;
    let doc_rep = reloaded_objs
        .iter()
        .find(|o| o.name == "Echo Repeater")
        .expect("echo repeater in SessionDocument");
    assert_eq!(doc_rep.kind, "zoidium:repeater/echo-repeater");
    assert!(
        reloaded_objs
            .iter()
            .any(|o| o.name == "Echo Child" && o.parent_id.as_deref() == Some(doc_rep.id.as_str())),
        "child of echo repeater must survive .pz round-trip"
    );

    // Legacy fallback: an existing Group with preserved["type"] is recognized on load and
    // migrated to preserved["zoidiumPluginObjectType"] without stale "type" on sync.
    let mut legacy_project = zc_project::Project::new_cm3_default();
    let mut legacy_rep = zc_project::Object3D::new(
        "Legacy Linear Repeater",
        zc_project::Object3DKind::Group {
            reflection_visibility: 0,
            custom_properties: Vec::new(),
            children: Vec::new(),
        },
    );
    legacy_rep.preserved.insert(
        "type".to_string(),
        json!("zoidium:repeater/linear-repeater"),
    );
    if let zc_project::LayerKind::Scene3D { objects, .. } =
        &mut legacy_project.sequence.video_tracks[0].clips[0].layer.kind
    {
        objects.push(legacy_rep);
    }
    let mut legacy_session =
        Session::from_project(legacy_project, zc_plugin::PluginRegistry::new());
    assert!(
        legacy_session.document().video_tracks[0].clips[0]
            .objects_3d
            .iter()
            .any(|o| o.name == "Legacy Linear Repeater"
                && o.kind == "zoidium:repeater/linear-repeater"),
        "SessionDocument::from_project must fall back to preserved[\"type\"]"
    );
    legacy_session
        .execute(Command::SetProjectName {
            name: "Migrated".to_string(),
        })
        .expect("trigger sync");
    let zc_project::LayerKind::Scene3D { objects, .. } =
        &legacy_session.project.sequence.video_tracks[0].clips[0]
            .layer
            .kind
    else {
        panic!("expected Scene3D");
    };
    let synced_legacy = objects
        .iter()
        .find(|o| o.name == "Legacy Linear Repeater")
        .expect("synced legacy repeater");
    assert_eq!(
        synced_legacy.preserved.get("zoidiumPluginObjectType"),
        Some(&json!("zoidium:repeater/linear-repeater")),
        "sync must migrate legacy repeater to preserved[\"zoidiumPluginObjectType\"]"
    );
    assert!(
        !synced_legacy.preserved.contains_key("type"),
        "sync must remove stale preserved[\"type\"] on repeater"
    );
}

#[test]
fn set_fps_clears_stale_exact_fps_and_undo_redo_restores_it() {
    let mut project = zc_project::Project::new_cm3_default();
    project.sequence.fps = 30;
    project.sequence.exact_fps = Some(29.97);
    let mut session = Session::from_project(project, zc_plugin::PluginRegistry::new());

    // Unrelated edits must preserve fractional exact_fps.
    session
        .execute(Command::SetProjectName {
            name: "NTSC Project".to_string(),
        })
        .expect("rename");
    assert_eq!(session.project.sequence.fps, 30);
    assert_eq!(session.project.sequence.exact_fps, Some(29.97));

    // SetFps to a new integer FPS clears stale exact_fps so .pz export uses 60, not 29.97.
    session
        .execute(Command::SetFps { fps: 60.0 })
        .expect("set fps 60");
    assert_eq!(session.document().fps, 60.0);
    assert_eq!(session.project.sequence.fps, 60);
    assert_eq!(session.project.sequence.exact_fps, None);

    let pz_bytes = zc_project::export_pz_to_bytes(&session.project).expect("export pz");
    let reimported = zc_project::import_pz_from_bytes(&pz_bytes).expect("import pz");
    assert_eq!(reimported.project.sequence.fps, 60);
    assert!(
        reimported
            .project
            .sequence
            .exact_fps
            .is_none_or(|r| (r - 60.0).abs() < 1e-9),
        "exported .pz rate must be 60, not stale 29.97 (got {:?})",
        reimported.project.sequence.exact_fps
    );

    // Undo restores exact_fps; Redo clears it again.
    session.execute(Command::Undo).expect("undo SetFps");
    assert_eq!(session.document().fps, 30.0);
    assert_eq!(session.project.sequence.fps, 30);
    assert_eq!(session.project.sequence.exact_fps, Some(29.97));

    session.execute(Command::Redo).expect("redo SetFps");
    assert_eq!(session.document().fps, 60.0);
    assert_eq!(session.project.sequence.fps, 60);
    assert_eq!(session.project.sequence.exact_fps, None);

    // Explicitly setting integer 30 on a 29.97 project also clears exact_fps, and Undo restores it.
    session.execute(Command::Undo).expect("undo back to 29.97");
    assert_eq!(session.project.sequence.exact_fps, Some(29.97));
    session
        .execute(Command::SetFps { fps: 30.0 })
        .expect("set fps 30");
    assert_eq!(session.project.sequence.fps, 30);
    assert_eq!(session.project.sequence.exact_fps, None);
    session.execute(Command::Undo).expect("undo snap to 30");
    assert_eq!(session.project.sequence.exact_fps, Some(29.97));
}
