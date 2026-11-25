use crate::property_conv::{prop_vec2_static, property_track_matches_session};
use crate::sync_from_project::default_clip_properties;
use crate::sync_objects_to_project::rebuild_object_tree;
use crate::*;

#[test]
fn qa_lights_particles_repeaters_split_letters_locks_playback_and_validation() {
    let mut session = Session::new();
    let vtrack_id = session.document().video_tracks[0].id.clone();
    let scene_id = session.document().video_tracks[0].clips[0].id.clone();

    // 1. Lights: camelCase normalization + useful default position + shadow_bias 0.002.
    for (raw_kind, expected_kind, expects_offset, expects_target) in [
        ("light", zc_project::LightKind::Spot, true, true),
        ("spotLight", zc_project::LightKind::Spot, true, true),
        ("SpotLight", zc_project::LightKind::Spot, true, true),
        (
            "directionalLight",
            zc_project::LightKind::Directional,
            true,
            true,
        ),
        (
            "DirectionalLight",
            zc_project::LightKind::Directional,
            true,
            true,
        ),
        ("pointLight", zc_project::LightKind::Point, true, false),
        ("PointLight", zc_project::LightKind::Point, true, false),
        ("ambientLight", zc_project::LightKind::Ambient, false, false),
        ("AmbientLight", zc_project::LightKind::Ambient, false, false),
        (
            "hemisphereLight",
            zc_project::LightKind::Hemisphere,
            false,
            false,
        ),
        (
            "HemisphereLight",
            zc_project::LightKind::Hemisphere,
            false,
            false,
        ),
    ] {
        let oid = session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: None,
                kind: raw_kind.to_owned(),
                name: None,
            })
            .unwrap_or_else(|e| panic!("add {raw_kind}: {e}"))["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let proj = session.to_project();
        let zc_project::LayerKind::Scene3D { ref objects, .. } =
            proj.sequence.video_tracks[0].clips[0].layer.kind
        else {
            panic!("expected Scene3D");
        };
        let obj = objects
            .iter()
            .find(|o| o.id.to_string() == oid)
            .expect("light object");
        let zc_project::Object3DKind::Light(ref light) = obj.kind else {
            panic!("{raw_kind} must create Light, got {:?}", obj.kind);
        };
        assert_eq!(light.kind, expected_kind, "{raw_kind} light kind");
        assert!(
            (light.shadow_bias - 0.002).abs() < 1e-9,
            "{raw_kind} shadow_bias"
        );
        if expects_offset {
            assert_eq!(
                obj.position.value,
                zc_project::PropertyValue::Vec3([25.0, 35.0, 55.0]),
                "{raw_kind} default position"
            );
        }
        if expects_target {
            assert_eq!(
                light.target.value,
                zc_project::PropertyValue::Vec3([0.0, 0.0, 0.0]),
                "{raw_kind} default target"
            );
        }
    }

    // 2. Particles: crisp non-255 size gradient & normal blending.
    let pid = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "particles".to_owned(),
            name: None,
        })
        .expect("add particles")["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let proj = session.to_project();
    let zc_project::LayerKind::Scene3D { ref objects, .. } =
        proj.sequence.video_tracks[0].clips[0].layer.kind
    else {
        panic!("expected Scene3D");
    };
    let pobj = objects
        .iter()
        .find(|o| o.id.to_string() == pid)
        .expect("particles");
    let zc_project::Object3DKind::Particles(ref pdata) = pobj.kind else {
        panic!("expected Particles");
    };
    assert_eq!(pdata.size.len(), 2);
    assert_eq!(pdata.size[0].color, "rgba(14,14,14,1.0)");
    assert_eq!(pdata.size[1].color, "rgba(4,4,4,1.0)");
    assert_eq!(pdata.blending, 0);

    // 3. Radial & Grid Repeaters + aliases + zero-warning .pz round-trip.
    for (raw_kind, canonical_kind, expected_key) in [
        (
            "zoidium:repeater/radial",
            "zoidium:repeater/radial",
            "radius",
        ),
        (
            "radial-repeater",
            "zoidium:repeater/radial",
            "alignToCircle",
        ),
        ("zoidium:repeater/grid", "zoidium:repeater/grid", "countX"),
        ("grid-repeater", "zoidium:repeater/grid", "spacing"),
    ] {
        let rid = session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: None,
                kind: raw_kind.to_owned(),
                name: None,
            })
            .expect("add repeater")["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let doc_obj = session.document().video_tracks[0].clips[0]
            .objects_3d
            .iter()
            .find(|o| o.id == rid)
            .expect("doc repeater");
        assert_eq!(doc_obj.kind, canonical_kind);
        assert!(
            doc_obj.properties.contains_key(expected_key),
            "{raw_kind} missing {expected_key}"
        );
        session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: Some(rid),
                kind: "box".to_owned(),
                name: Some("Child".to_owned()),
            })
            .expect("parent child under radial/grid repeater");
    }
    let pz_bytes = zc_project::export_pz_to_bytes(&session.to_project()).expect("export pz");
    let reimported = zc_project::import_pz_from_bytes(&pz_bytes).expect("import pz");
    assert!(
        reimported.warnings.is_empty(),
        "radial/grid repeater .pz round-trip must emit zero warnings, got {:?}",
        reimported.warnings
    );

    // 4. Split into Letters: proportional advance ("WILL"), multiline "\n", auto-select group.
    let txt_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "text_3d".to_owned(),
            name: Some("Multiline".to_owned()),
        })
        .expect("add text")["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: txt_id.clone(),
            property: "text".to_owned(),
            value: json!("WILL\nHI"),
        })
        .expect("set multiline text");
    let split_res = session
        .execute(Command::SplitText3DIntoLetters {
            clip_id: zc_project::Uuid::parse_str(&scene_id).unwrap(),
            object_id: zc_project::Uuid::parse_str(&txt_id).unwrap(),
        })
        .expect("split multiline");
    let group_id = split_res["group_id"].as_str().unwrap().to_owned();
    assert_eq!(
        session.selection.object_3d_ids,
        vec![group_id.clone()],
        "splitting into letters must auto-select the created Text group"
    );
    let proj = session.to_project();
    let zc_project::LayerKind::Scene3D { ref objects, .. } =
        proj.sequence.video_tracks[0].clips[0].layer.kind
    else {
        panic!("expected Scene3D");
    };
    let grp = objects
        .iter()
        .find(|o| o.id.to_string() == group_id)
        .expect("split group");
    let zc_project::Object3DKind::Group { ref children, .. } = grp.kind else {
        panic!("expected Group");
    };
    assert_eq!(children.len(), 6); // W, I, L, L, H, I
    let pos_of = |idx: usize| match children[idx].position.value {
        zc_project::PropertyValue::Vec3(v) => v,
        ref other => panic!("expected Vec3, got {other:?}"),
    };
    let [wx, wy, _] = pos_of(0);
    let [ix, iy, _] = pos_of(1);
    let [lx1, _, _] = pos_of(2);
    let [lx2, _, _] = pos_of(3);
    let [_, hy, _] = pos_of(4);
    assert!((wy - iy).abs() < 1e-6, "same line Y");
    assert!(
        (wy - hy - 24.0).abs() < 1e-6,
        "second line must drop by height * line_height (20 * 1.2 = 24), got {}",
        wy - hy
    );
    let gap_w_i = ix - wx;
    let gap_i_l = lx1 - ix;
    let gap_l_l = lx2 - lx1;
    assert!(
        gap_w_i > gap_i_l + 2.0,
        "W->I center distance ({gap_w_i}) must exceed I->L ({gap_i_l}) under proportional advance"
    );
    assert!(
        (gap_w_i - gap_l_l).abs() > 0.1,
        "proportional widths must differ from fixed monospace"
    );

    // 5. Validation: SetPropertyValue rejects wrapped object on string/text properties.
    let bad_text_err = session
        .execute(Command::SetPropertyValue {
            target_id: children[0].id.to_string(),
            property: "text".to_owned(),
            value: json!({ "kind": "text", "value": "NOPE" }),
        })
        .expect_err("wrapped object on text must be rejected");
    assert_eq!(bad_text_err.code(), "invalid_argument");
    let letter_after = session.document().video_tracks[0].clips[0]
        .objects_3d
        .iter()
        .find(|o| o.id == children[0].id.to_string())
        .expect("letter W");
    assert_eq!(
        prop_str_static(&letter_after.properties, "text").as_deref(),
        Some("W"),
        "rejected SetPropertyValue must not clear existing text"
    );

    // 6. Validation: AddEffect rejects unknown bogus effect IDs, permits valid & custom/plugin IDs.
    let fake_fx_err = session
        .execute(Command::AddEffect {
            clip_id: scene_id.clone(),
            kind: "totally_fake_effect_xyz".to_owned(),
            name: None,
        })
        .expect_err("bogus effect kind must fail");
    assert_eq!(fake_fx_err.code(), "invalid_operation");
    session
        .execute(Command::AddEffect {
            clip_id: scene_id.clone(),
            kind: "customShader".to_owned(),
            name: None,
        })
        .expect("customShader allowed");
    session
        .execute(Command::AddEffect {
            clip_id: scene_id.clone(),
            kind: "zoidium:custom-pack/my-filter".to_owned(),
            name: None,
        })
        .expect("namespaced custom plugin effect allowed");

    // 7. Playback: toggle_play remembers last non-zero speed across pause/resume.
    session
        .execute(Command::SetSpeed { speed: 2.0 })
        .expect("set 2x");
    assert!((session.playback.speed - 2.0).abs() < 1e-9);
    session.execute(Command::TogglePlay).expect("pause");
    assert!((session.playback.speed - 0.0).abs() < 1e-9);
    session.execute(Command::TogglePlay).expect("resume 2x");
    assert!((session.playback.speed - 2.0).abs() < 1e-9);
    session
        .execute(Command::SetSpeed { speed: -1.5 })
        .expect("set -1.5x");
    session.execute(Command::TogglePlay).expect("pause reverse");
    session.execute(Command::TogglePlay).expect("resume -1.5x");
    assert!((session.playback.speed - (-1.5)).abs() < 1e-9);

    // 8. Locked-track error displays human-readable track name ('V1'), not raw UUID.
    session
        .execute(Command::SetTrackLocked {
            track_id: vtrack_id.clone(),
            locked: true,
        })
        .expect("lock V1");
    let lock_err = session
        .execute(Command::AddObject3D {
            clip_id: scene_id,
            parent_id: None,
            kind: "box".to_owned(),
            name: None,
        })
        .expect_err("locked track must reject edit");
    assert_eq!(lock_err.code(), "track_locked");
    assert_eq!(lock_err.to_string(), "track 'V1' is locked");
    assert!(!lock_err.to_string().contains(&vtrack_id));
}

#[test]
fn qa_c2_session_defaults_particular_radial_unique_names_and_sphere_radius() {
    let mut session = Session::new();
    let vtrack_id = session.document.video_tracks[0].id.clone();
    let scene_id = session
        .execute(Command::AddClip {
            track_id: vtrack_id,
            kind: "scene".to_owned(),
            start: 0.0,
            length: 180.0,
            name: Some("Defaults Scene".to_owned()),
        })
        .expect("add scene clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();

    // 1. Particular default particleColor is vivid sky cyan-blue [0.40, 0.85, 1.0].
    let part_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "zoidium:trapcode-suite/particular".to_owned(),
            name: Some("Particular".to_owned()),
        })
        .expect("add particular")["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip = &session.document.video_tracks[0].clips[0];
    let part_sobj = clip
        .objects_3d
        .iter()
        .find(|o| o.id == part_id)
        .expect("particular session obj");
    assert_eq!(
        part_sobj.properties["particleColor"].default_value,
        json!([0.40, 0.85, 1.0])
    );
    let zc_project::LayerKind::Scene3D { objects, .. } =
        &session.project.sequence.video_tracks[0].clips[0].layer.kind
    else {
        panic!("expected Scene3D");
    };
    let part_proj = objects
        .iter()
        .find(|o| o.id.to_string() == part_id)
        .expect("particular project obj");
    assert_eq!(
        part_proj.preserved.get("property.particleColor"),
        Some(&json!([0.40, 0.85, 1.0]))
    );

    // 2. Radial Repeater default axis is 2 (Z-axis / XY-plane facing front camera).
    let radial_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "zoidium:repeater/radial".to_owned(),
            name: Some("Radial Repeater".to_owned()),
        })
        .expect("add radial repeater")["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip = &session.document.video_tracks[0].clips[0];
    let radial_sobj = clip
        .objects_3d
        .iter()
        .find(|o| o.id == radial_id)
        .expect("radial session obj");
    assert_eq!(radial_sobj.properties["axis"].default_value, json!(2));
    let zc_project::LayerKind::Scene3D { objects, .. } =
        &session.project.sequence.video_tracks[0].clips[0].layer.kind
    else {
        panic!("expected Scene3D");
    };
    let radial_proj = objects
        .iter()
        .find(|o| o.id.to_string() == radial_id)
        .expect("radial project obj");
    assert_eq!(radial_proj.preserved.get("property.axis"), Some(&json!(2)));

    // 3. Unique auto-numbered 3D object names on collision (both explicit and default None).
    let cam1 = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "camera".to_owned(),
            name: Some("Camera".to_owned()),
        })
        .expect("add Camera 1");
    let cam2 = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "camera".to_owned(),
            name: Some("Camera".to_owned()),
        })
        .expect("add Camera 2");
    let cam3 = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "camera".to_owned(),
            name: Some("Camera".to_owned()),
        })
        .expect("add Camera 3");
    assert_eq!(cam1["name"], json!("Camera"));
    assert_eq!(cam2["name"], json!("Camera 2"));
    assert_eq!(cam3["name"], json!("Camera 3"));

    // Deleting "Camera 2" and adding another "Camera" reuses the first free slot ("Camera 2").
    let cam2_id = cam2["object_id"].as_str().unwrap().to_owned();
    session
        .execute(Command::DeleteObject3D {
            clip_id: scene_id.clone(),
            object_id: cam2_id,
        })
        .expect("delete Camera 2");
    let cam2_again = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "camera".to_owned(),
            name: Some("Camera".to_owned()),
        })
        .expect("re-add Camera 2");
    assert_eq!(cam2_again["name"], json!("Camera 2"));

    // Also works when name is None (defaults to canonical kind).
    let anon1 = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "group".to_owned(),
            name: None,
        })
        .expect("add anon group 1");
    let anon2 = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "group".to_owned(),
            name: None,
        })
        .expect("add anon group 2");
    assert_eq!(anon1["name"], json!("group"));
    assert_eq!(anon2["name"], json!("group 2"));

    // 4. Sphere default radius & detail in session properties match synced Shape3DKind::Sphere.
    let sphere_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_id,
            parent_id: None,
            kind: "sphere".to_owned(),
            name: Some("Sphere".to_owned()),
        })
        .expect("add sphere")["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip = &session.document.video_tracks[0].clips[0];
    let sphere_sobj = clip
        .objects_3d
        .iter()
        .find(|o| o.id == sphere_id)
        .expect("sphere session obj");
    assert_eq!(
        prop_f64_static(&sphere_sobj.properties, "sphere_size"),
        Some(5.0),
        "sphere session property sphere_size must match rendered radius 5.0"
    );
    assert_eq!(
        prop_vec2_static(&sphere_sobj.properties, "sphere_detail"),
        Some([24.0, 16.0]),
        "sphere session property sphere_detail must match rendered [24, 16]"
    );
    let zc_project::LayerKind::Scene3D { objects, .. } =
        &session.project.sequence.video_tracks[0].clips[0].layer.kind
    else {
        panic!("expected Scene3D");
    };
    let sphere_proj = objects
        .iter()
        .find(|o| o.id.to_string() == sphere_id)
        .expect("sphere project obj");
    assert!(
        matches!(
            sphere_proj.kind,
            zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Sphere {
                radius,
                width_segments: 24,
                height_segments: 16,
            }) if (radius - 5.0).abs() < 1e-9
        ),
        "synced Sphere must match session properties (radius=5.0, 24x16), got {:?}",
        sphere_proj.kind
    );
}

#[test]
fn precompose_and_create_composition_auto_enable_precomp_plus_and_undo_restores_plugins() {
    let mut session = Session::new();
    // Start with precomp-plus explicitly disabled (Vanilla CM3 mode).
    let _ = session.plugins.disable("precomp-plus", &BTreeSet::new());
    assert!(!session.plugins.is_enabled("precomp-plus"));

    // 1. PrecomposeClips auto-enables `precomp-plus` and Undo restores disabled state.
    let orig_clip_uuid = parse_uuid_or_new(&session.document.video_tracks[0].clips[0].id);
    let precomp_res = session
        .execute(Command::PrecomposeClips {
            clip_ids: vec![orig_clip_uuid],
            name: "Precomp 1".to_owned(),
        })
        .expect("precompose clips");
    let comp_id_str = precomp_res["comp_id"].as_str().unwrap().to_owned();
    assert!(
        session.plugins.is_enabled("precomp-plus"),
        "PrecomposeClips must auto-enable precomp-plus"
    );
    assert!(session.project.plugin_requirements.contains("precomp-plus"));

    // inspect_state must expose `compositions` and `active_composition_id`.
    let state = session.inspect_state();
    assert!(state["active_composition_id"].is_null());
    let comps = state["compositions"]
        .as_array()
        .expect("compositions array in inspect_state");
    assert_eq!(comps.len(), 1);
    assert_eq!(comps[0]["id"], comp_id_str);
    assert_eq!(comps[0]["name"], "Precomp 1");
    assert!(comps[0]["width"].as_u64().unwrap() > 0);
    assert!(comps[0]["height"].as_u64().unwrap() > 0);
    assert!(comps[0]["fps"].as_u64().unwrap() > 0);

    // Entering the child composition updates `active_composition_id` in inspect_state.
    let comp_uuid = parse_uuid_or_new(&comp_id_str);
    session
        .execute(Command::SetActiveComposition {
            comp_id: Some(comp_uuid),
        })
        .expect("enter child comp");
    assert_eq!(
        session.inspect_state()["active_composition_id"].as_str(),
        Some(comp_id_str.as_str())
    );

    // Pure navigation SetActiveComposition must NOT push an undo snapshot when `precomp-plus` is already enabled.
    // A single Undo must undo PrecomposeClips directly -> precomp-plus must be disabled again!
    session.execute(Command::Undo).expect("undo precompose");
    assert!(
        !session.plugins.is_enabled("precomp-plus"),
        "Undo of PrecomposeClips must restore prior disabled state of precomp-plus"
    );

    // Redo PrecomposeClips -> precomp-plus must be enabled again!
    session.execute(Command::Redo).expect("redo precompose");
    assert!(
        session.plugins.is_enabled("precomp-plus"),
        "Redo of PrecomposeClips must re-enable precomp-plus"
    );

    // 2. CreateComposition also auto-enables `precomp-plus` and Undo restores prior state.
    session.execute(Command::Undo).expect("undo again");
    assert!(!session.plugins.is_enabled("precomp-plus"));
    session
        .execute(Command::CreateComposition {
            name: "SubComp".to_owned(),
            resolution: [640, 360],
            fps: 30,
            length_frames: 60,
        })
        .expect("create composition");
    assert!(session.plugins.is_enabled("precomp-plus"));
    session
        .execute(Command::Undo)
        .expect("undo create composition");
    assert!(!session.plugins.is_enabled("precomp-plus"));
}

#[test]
fn qa4_set_active_composition_pure_navigation_does_not_dirty_and_friendly_split_error() {
    let mut session = Session::new();
    let clip_id = session.document.video_tracks[0].clips[0].id.clone();

    // Friendly SplitClip error when playhead/at_frame is outside clip range
    let split_err = session
        .execute(Command::SplitClip {
            clip_id: clip_id.clone(),
            at_frame: 9999.0,
        })
        .expect_err("split outside clip range must fail");
    assert_eq!(
        split_err,
        SessionError::InvalidOperation(
            "Move playhead inside the selected clip to split".to_owned()
        )
    );

    // Create a composition (enables `precomp-plus`), then mark project clean
    let created = session
        .execute(Command::CreateComposition {
            name: "NavComp".to_owned(),
            resolution: [1280, 720],
            fps: 30,
            length_frames: 60,
        })
        .expect("create composition");
    let comp_uuid = parse_uuid_or_new(created["comp_id"].as_str().unwrap());
    session.mark_saved();
    assert!(!session.is_dirty());
    let undo_before = session.history.undo_len();

    // Pure navigation into child comp and back to main sequence must NOT dirty project or push undo
    session
        .execute(Command::SetActiveComposition {
            comp_id: Some(comp_uuid),
        })
        .expect("enter child comp");
    assert!(
        !session.is_dirty(),
        "entering composition with precomp-plus already enabled must not dirty project"
    );
    assert_eq!(session.history.undo_len(), undo_before);

    session
        .execute(Command::SetActiveComposition { comp_id: None })
        .expect("return to main sequence");
    assert!(
        !session.is_dirty(),
        "returning to main sequence must not dirty project"
    );
    assert_eq!(session.history.undo_len(), undo_before);
}

#[test]
fn duplicate_clip_avoids_overlapping_existing_clips_and_selects_new_duplicate() {
    let mut session = Session::new();
    let v1_id = session.document.video_tracks[0].id.clone();
    // Start with Clip A [0, 60) and Clip B [60, 120) on V1.
    let clip_a_id = session.document.video_tracks[0].clips[0].id.clone();
    let orig_len = session.document.video_tracks[0].clips[0].length;
    session
        .execute(Command::TrimClipRight {
            clip_id: clip_a_id.clone(),
            delta_frames: 60.0 - orig_len,
            rate_stretch: false,
        })
        .expect("trim clip A to 60 frames");
    let clip_b_id = session
        .execute(Command::AddClip {
            track_id: v1_id.clone(),
            kind: "text".to_owned(),
            name: Some("Clip B".to_owned()),
            start: 60.0,
            length: 60.0,
        })
        .expect("add clip B")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();

    // Select Clip A and duplicate it: target range [60, 120) overlaps Clip B on V1,
    // so DuplicateClip must append V2 and place the duplicate there without harming Clip B!
    session
        .selection
        .select_clip(clip_a_id.clone(), Some("scene"), false)
        .unwrap();
    let dup_res = session
        .execute(Command::DuplicateClip {
            clip_id: clip_a_id.clone(),
        })
        .expect("duplicate clip A");
    let dup_id = dup_res["clip_id"].as_str().unwrap().to_owned();

    // Clip B must remain completely intact on V1 at [60, 120).
    let v1_clip_b = session.document.video_tracks[0]
        .clips
        .iter()
        .find(|c| c.id == clip_b_id)
        .expect("Clip B must not be overwritten");
    assert!((v1_clip_b.start - 60.0).abs() < 1e-6);
    assert!((v1_clip_b.length - 60.0).abs() < 1e-6);

    // Duplicate must live on newly appended V2 at [60, 120) and be selected.
    assert_eq!(session.document.video_tracks.len(), 2);
    let v2_dup = session.document.video_tracks[1]
        .clips
        .iter()
        .find(|c| c.id == dup_id)
        .expect("duplicate clip must be placed on non-overlapping V2");
    assert!((v2_dup.start - 60.0).abs() < 1e-6);
    assert_eq!(session.selection.clip_ids, vec![dup_id]);

    // Undo must restore original selection pointing back at Clip A and remove V2.
    session.execute(Command::Undo).expect("undo duplicate");
    assert_eq!(session.selection.clip_ids, vec![clip_a_id]);
    assert_eq!(session.document.video_tracks.len(), 1);
}

#[test]
fn qa6_expresstext_plus_all_11_modifier_defaults() {
    let mut session = Session::new();
    let scene_clip_id = session.document.video_tracks[0].clips[0].id.clone();

    let modifiers: &[(&str, &str)] = &[
        ("shake", "shakePosition"),
        ("plain", "plainPosition"),
        ("random", "randomPosition"),
        ("formula", "formulaPosition"),
        ("delay", "delayFrames"),
        ("spline", "splineAmplitude"),
        ("orbit", "orbitAngle"),
        ("repeater", "repeaterOffset"),
        ("twist", "twistAngle"),
        ("wrap", "wrapAmplitude"),
        ("voronoi", "voronoiDistance"),
    ];

    for &(sub, key_param) in modifiers {
        let kind = format!("zoidium:expresstext-plus/{sub}");
        let added = session
            .execute(Command::AddObject3D {
                clip_id: scene_clip_id.clone(),
                kind,
                name: Some(format!("ET-{sub}")),
                parent_id: None,
            })
            .expect("add ExpressText+ modifier");
        let obj_id = added["object_id"].as_str().unwrap();
        let obj = session.document.video_tracks[0].clips[0]
            .objects_3d
            .iter()
            .find(|o| o.id == obj_id)
            .expect("added modifier object");

        assert_eq!(
            obj.properties["lettersMode"].default_value,
            json!("per_letter"),
            "{sub}: lettersMode must be per_letter"
        );
        let enable_key = format!("{sub}Enable");
        assert!(
            obj.properties.contains_key(&enable_key),
            "{sub}: missing {enable_key} (must not be {sub}Enabled)"
        );
        assert!(
            !obj.properties.contains_key(&format!("{sub}Enabled")),
            "{sub}: stale {sub}Enabled key must not exist"
        );
        assert_eq!(obj.properties[&enable_key].default_value, json!(1));
        assert_eq!(
            obj.properties[&format!("{sub}Strength")].default_value,
            json!(100.0),
            "{sub}: strength must be 100.0 (0..100 scale)"
        );
        assert!(
            obj.properties.contains_key(key_param),
            "{sub}: missing visible default param {key_param}"
        );
    }
}

#[test]
fn qa6_session_invariants_plugin_auto_enable_set_active_paste_clips_and_validation() {
    let mut session = Session::new();
    let v1_id = session.document.video_tracks[0].id.clone();
    let clip_id = session.document.video_tracks[0].clips[0].id.clone();

    // 1. `effects.add_effect` auto-enables plugin atomically & Undo reverts `enabled`
    assert!(!session.plugins.is_enabled("sglow-hq-v2-preview"));
    session
        .execute(Command::AddEffect {
            clip_id: clip_id.clone(),
            kind: "sglow-hq-v2-preview".to_owned(),
            name: None,
        })
        .expect("add sglow plugin effect");
    assert!(session.plugins.is_enabled("sglow-hq-v2-preview"));

    // `plugins.disable_all_optional` must NEVER report `vanilla_cm3_mode: true` when deps are active
    let dis = session
        .execute(Command::DisableAllOptionalPlugins)
        .expect("disable optional plugins");
    assert_eq!(dis["vanilla_cm3_mode"], json!(false));

    // Undo reverts `sglow-hq-v2-preview` enabled back to false
    session.execute(Command::Undo).expect("undo add sglow");
    assert!(!session.plugins.is_enabled("sglow-hq-v2-preview"));

    // 2. `timeline.add_clip` auto-enables plugin (`shape_plus`) & Undo reverts `enabled`
    let dis_clean = session
        .execute(Command::DisableAllOptionalPlugins)
        .expect("disable all optional plugins when no deps active");
    assert_eq!(dis_clean["vanilla_cm3_mode"], json!(true));
    assert!(!session.plugins.is_enabled("shape-plus"));
    session
        .execute(Command::AddClip {
            track_id: v1_id.clone(),
            kind: "shape_plus".to_owned(),
            name: Some("SP".to_owned()),
            start: 180.0,
            length: 30.0,
        })
        .expect("add shape_plus clip");
    assert!(session.plugins.is_enabled("shape-plus"));
    session.execute(Command::Undo).expect("undo add shape_plus");
    assert!(!session.plugins.is_enabled("shape-plus"));

    // 3. `timeline.add_clip` rejects unknown/bogus kind with InvalidArgument
    let bad_kind = session
        .execute(Command::AddClip {
            track_id: v1_id.clone(),
            kind: "totally_bogus_xyz".to_owned(),
            name: None,
            start: 180.0,
            length: 30.0,
        })
        .expect_err("bogus clip kind must fail");
    assert!(matches!(bad_kind, SessionError::InvalidArgument(_)));

    // 4. `project.set_sequence_length` rejects <= 0.0, non-finite, and > 1_000_000.0
    for bad_len in [0.0, -10.0, f64::NAN, f64::INFINITY, 1_000_001.0] {
        let err = session
            .execute(Command::SetSequenceLength { frames: bad_len })
            .expect_err("invalid sequence length must fail");
        assert!(
            matches!(err, SessionError::InvalidArgument(_)),
            "len {bad_len} expected InvalidArgument, got {err:?}"
        );
    }

    // 5. `comp.set_active` is non-undoable and unrelated Undo/Redo does NOT warp active_composition_id
    let cmd = Command::SetActiveComposition { comp_id: None };
    assert!(!cmd.is_undoable());
    assert!(!session.registry.get("comp.set_active").unwrap().undoable);

    let created = session
        .execute(Command::CreateComposition {
            name: "SubA".to_owned(),
            resolution: [1280, 720],
            fps: 30,
            length_frames: 60,
        })
        .expect("create comp");
    let comp_uuid = parse_uuid_or_new(created["comp_id"].as_str().unwrap());
    session
        .execute(Command::SetActiveComposition {
            comp_id: Some(comp_uuid),
        })
        .expect("enter SubA");
    session
        .execute(Command::SetProjectName {
            name: "Renamed".to_owned(),
        })
        .expect("rename while in SubA");
    session
        .execute(Command::SetActiveComposition { comp_id: None })
        .expect("return to Main");
    assert_eq!(session.document.active_composition_id, None);
    session.execute(Command::Undo).expect("undo rename");
    assert_eq!(
        session.document.active_composition_id, None,
        "Undo of unrelated edit must not warp active_composition_id back into SubA"
    );
    session.execute(Command::Redo).expect("redo rename");
    assert_eq!(
        session.document.active_composition_id, None,
        "Redo of unrelated edit must keep active_composition_id in Main"
    );

    // 6. `timeline.paste_clips` attaches `layer_masks` & `precomp_offset` even when `video_clip` is None
    let dummy_clip = SessionClip {
        id: "temp-clip".to_owned(),
        kind: "scene".to_owned(),
        name: "PastedMasked".to_owned(),
        start: 0.0,
        length: 45.0,
        time_start: 0.0,
        time_end: 1.5,
        time_animated: false,
        media_length: None,
        source_clip_id: None,
        properties: default_clip_properties(TrackKind::Video, "scene", [0.0, 0.0, 0.0, 1.0]),
        objects_3d: Vec::new(),
        effects: Vec::new(),
    };
    let mask = zc_project::LayerMask {
        id: "mask-qa6".to_owned(),
        name: "Mask 1".to_owned(),
        enabled: true,
        mode: zc_project::MaskMode::Add,
        invert: false,
        feather: 0.0,
        expansion: 0.0,
        opacity: 1.0,
        closed: true,
        locked: false,
        outline_color: String::new(),
        vertex_ids: Vec::new(),
        vertex_modes: BTreeMap::new(),
        path: zc_project::PropertyTrack::constant(zc_project::PropertyValue::Json(json!([
            0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0
        ]))),
        preserved: BTreeMap::new(),
    };
    let snap = ClipSnapshot {
        track_kind: TrackKind::Video,
        track_id: Some(v1_id),
        clip: dummy_clip,
        video_clip: None,
        audio_clip: None,
        layer_masks: vec![mask],
        precomp_offset: Some(12.5),
    };
    let pasted = session
        .execute(Command::PasteClips {
            clips: vec![snap],
            target_frame: 200.0,
        })
        .expect("paste clip without video_clip");
    let pasted_id = parse_uuid_or_new(pasted["clip_ids"][0].as_str().unwrap());
    let pasted_vc = session
        .project
        .sequence
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == pasted_id)
        .expect("pasted video clip");
    let layer_id = pasted_vc.layer.id;
    assert_eq!(
        session
            .project
            .sequence
            .layer_masks
            .get(&layer_id)
            .map(Vec::len),
        Some(1),
        "layer_masks must attach even when snap.video_clip is None"
    );
    assert_eq!(
        session.project.sequence.precomp_offsets.get(&layer_id),
        Some(&12.5),
        "precomp_offset must attach even when snap.video_clip is None"
    );
}

#[test]
fn large_video_asset_edits_skip_sha256_rehash_and_restore_on_undo_redo() {
    const VIDEO_BYTES: usize = 8 * 1024 * 1024; // 8 MiB
    let mut video_payload = vec![0x5Au8; VIDEO_BYTES];
    video_payload[0] = 0x11;
    video_payload[VIDEO_BYTES / 2] = 0x22;
    video_payload[VIDEO_BYTES - 1] = 0x33;

    let mut project = zc_project::Project::new_cm3_default();
    let prehashed_digest = "0123456789abcdef".repeat(4);
    let video_key = project.insert_binary_asset_prehashed(
        zc_project::AssetKind::Av,
        "large_clip.mp4",
        Some("/videos/large_clip.mp4".to_owned()),
        video_payload.clone(),
        prehashed_digest.clone(),
    );
    // Also add a small unhashed asset so we verify fallback hashing populates `sha256` & `size`.
    let unhashed_bytes = b"small-unhashed-asset".to_vec();
    project.assets.insert(
        "unhashed-key".to_owned(),
        zc_project::AssetRecord {
            id: "unhashed-key".to_owned(),
            kind: zc_project::AssetKind::Image,
            source: zc_project::AssetSource::File,
            filename: "thumb.png".to_owned(),
            url: None,
            external: false,
            bytes: Some(unhashed_bytes.clone()),
            missing: false,
            sha256: String::new(),
            size: 0,
            preserved: BTreeMap::new(),
        },
    );

    let mut session = Session::from_project(project, zc_plugin::PluginRegistry::new());

    let started = std::time::Instant::now();
    for i in 0..30_u32 {
        session
            .execute(Command::SetProjectName {
                name: format!("Edit {i}"),
            })
            .expect("sequential edit");
    }
    let elapsed = started.elapsed();
    assert!(
        elapsed < std::time::Duration::from_millis(100),
        "30 sequential edits with an 8 MiB video asset must avoid per-edit SHA-256 re-hashing (took {elapsed:?})"
    );

    // Prehashed digest was reused in O(1) without recomputing SHA-256 on the 8 MiB video.
    assert_eq!(session.project.assets[&video_key].sha256, prehashed_digest);
    assert_eq!(session.project.assets[&video_key].size, VIDEO_BYTES as u64);
    assert!(session.asset_blobs.contains_key(&prehashed_digest));
    // Fallback path populated sha256 & size on the unhashed asset so future edits are O(1).
    let expected_small_sha = zc_project::sha256_hex(&unhashed_bytes);
    assert_eq!(
        session.project.assets["unhashed-key"].sha256,
        expected_small_sha
    );
    assert_eq!(
        session.project.assets["unhashed-key"].size,
        unhashed_bytes.len() as u64
    );

    // Undo and Redo restore exact asset bytes, sha256, size, and disk url.
    for _ in 0..5 {
        session.execute(Command::Undo).expect("undo");
    }
    assert_eq!(session.document().project_name, "Edit 24");
    let restored_undo = &session.project.assets[&video_key];
    assert_eq!(
        restored_undo.bytes.as_deref(),
        Some(video_payload.as_slice())
    );
    assert_eq!(restored_undo.sha256, prehashed_digest);
    assert_eq!(restored_undo.size, VIDEO_BYTES as u64);
    assert_eq!(restored_undo.url.as_deref(), Some("/videos/large_clip.mp4"));

    for _ in 0..5 {
        session.execute(Command::Redo).expect("redo");
    }
    assert_eq!(session.document().project_name, "Edit 29");
    let restored_redo = &session.project.assets[&video_key];
    assert_eq!(
        restored_redo.bytes.as_deref(),
        Some(video_payload.as_slice())
    );
    assert_eq!(restored_redo.sha256, prehashed_digest);
    assert_eq!(restored_redo.size, VIDEO_BYTES as u64);
    assert_eq!(
        session.project.assets["unhashed-key"].bytes.as_deref(),
        Some(unhashed_bytes.as_slice())
    );
}

#[test]
fn structural_estimated_bytes_tracks_assets_and_document_growth() {
    let mut session = Session::new();
    let base_bytes = session.snapshot().estimated_bytes();
    assert!(base_bytes > 0, "baseline snapshot size must be non-zero");

    let payload = vec![0xABu8; 256 * 1024];
    session
        .project
        .insert_binary_asset(zc_project::AssetKind::Image, "big.png", payload.clone());
    let with_asset_bytes = session.snapshot().estimated_bytes();
    assert!(
        with_asset_bytes >= base_bytes + payload.len(),
        "structural estimate must include embedded asset bytes ({with_asset_bytes} vs {} + {})",
        base_bytes,
        payload.len()
    );

    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    for i in 0..25 {
        session
            .execute(Command::AddObject3D {
                clip_id: clip_id.clone(),
                parent_id: None,
                kind: "box".to_owned(),
                name: Some(format!("Cube {i}")),
            })
            .expect("add box");
    }
    let with_objects_bytes = session.snapshot().estimated_bytes();
    assert!(
        with_objects_bytes > with_asset_bytes,
        "adding 25 3D objects must grow structural estimate ({with_objects_bytes} > {with_asset_bytes})"
    );
}

#[test]
fn rebuild_object_tree_handles_large_hierarchy_in_linear_time_with_exact_order() {
    // Build a 1,000-node hierarchy: 10 root groups, each with 99 children (some nested).
    let mut flat = Vec::with_capacity(1000);
    let mut counter = 1u64;
    let mut next_uuid = || {
        let id = format!("00000000-0000-4000-8000-{counter:012x}");
        counter += 1;
        id
    };

    let mut root_ids = Vec::new();
    for g in 0..10 {
        let gid = next_uuid();
        root_ids.push(gid.clone());
        flat.push(SessionObject3D {
            id: gid.clone(),
            kind: "group".to_owned(),
            name: format!("Group {g}"),
            parent_id: None,
            source_id: None,
            properties: BTreeMap::new(),
        });
        let sub_gid = next_uuid();
        flat.push(SessionObject3D {
            id: sub_gid.clone(),
            kind: "group".to_owned(),
            name: format!("SubGroup {g}"),
            parent_id: Some(gid.clone()),
            source_id: None,
            properties: BTreeMap::new(),
        });
        for c in 0..98 {
            let parent = if c % 2 == 0 { &gid } else { &sub_gid };
            flat.push(SessionObject3D {
                id: next_uuid(),
                kind: "box".to_owned(),
                name: format!("Box {g}_{c}"),
                parent_id: Some(parent.clone()),
                source_id: None,
                properties: BTreeMap::new(),
            });
        }
    }
    assert_eq!(flat.len(), 1000);

    let mut existing = BTreeMap::new();
    let started = std::time::Instant::now();
    let tree = rebuild_object_tree(&flat, &mut existing, None);
    let elapsed = started.elapsed();

    assert_eq!(tree.len(), 10);
    for (g, root) in tree.iter().enumerate() {
        assert_eq!(root.name, format!("Group {g}"));
        let zc_project::Object3DKind::Group { ref children, .. } = root.kind else {
            panic!("expected root group");
        };
        // 1 SubGroup + 49 even boxes
        assert_eq!(children.len(), 50);
        assert_eq!(children[0].name, format!("SubGroup {g}"));
        let zc_project::Object3DKind::Group {
            children: ref sub_children,
            ..
        } = children[0].kind
        else {
            panic!("expected nested subgroup");
        };
        // 49 odd boxes
        assert_eq!(sub_children.len(), 49);
        assert_eq!(sub_children[0].name, format!("Box {g}_1"));
        assert_eq!(sub_children[48].name, format!("Box {g}_97"));
    }
    assert!(
        elapsed < std::time::Duration::from_millis(500),
        "O(N) rebuild_object_tree on 1,000 objects should be fast (took {elapsed:?})"
    );
}

#[test]
fn property_track_matches_session_agrees_with_property_track_to_session() {
    let mut animated_track = zc_project::PropertyTrack::vec3([1.0, 2.0, 3.0]);
    animated_track.animated = true;
    animated_track.expression = Some("time * 2.0".to_owned());
    animated_track.keyframes = vec![
        zc_project::PropertyKeyframe::linear(0.0, zc_project::PropertyValue::Vec3([1.0, 2.0, 3.0])),
        zc_project::PropertyKeyframe::linear(
            30.0,
            zc_project::PropertyValue::Vec3([4.0, 5.0, 6.0]),
        ),
    ];

    let mut channel_track = zc_project::PropertyTrack::vec3([0.0, 0.0, 0.0]);
    channel_track.channels = vec![
        zc_project::PropertyTrack::number(1.5),
        zc_project::PropertyTrack::number(2.5),
        zc_project::PropertyTrack::number(3.5),
    ];

    let mut animated_channel_track = channel_track.clone();
    animated_channel_track.channels[0].animated = true;
    animated_channel_track.channels[0].keyframes = vec![
        zc_project::PropertyKeyframe::linear(0.0, zc_project::PropertyValue::Number(0.0)),
        zc_project::PropertyKeyframe::linear(10.0, zc_project::PropertyValue::Number(10.0)),
    ];

    let tracks = [
        zc_project::PropertyTrack::number(0.75),
        zc_project::PropertyTrack::option(2),
        zc_project::PropertyTrack::color([0.1, 0.2, 0.3]),
        zc_project::PropertyTrack::asset(Some("asset-1".to_owned())),
        zc_project::PropertyTrack::asset(None),
        animated_track,
        channel_track,
        animated_channel_track,
    ];

    for track in &tracks {
        let prop = property_track_to_session(track);
        assert!(property_track_matches_session(track, &prop));

        let mut mutated_anim = prop.clone();
        mutated_anim.animated = !mutated_anim.animated;
        assert!(!property_track_matches_session(track, &mutated_anim));
        assert_ne!(property_track_to_session(track), mutated_anim);

        let mut mutated_kf = prop.clone();
        mutated_kf.keyframes[0].frame += 1.0;
        assert!(!property_track_matches_session(track, &mutated_kf));
        assert_ne!(property_track_to_session(track), mutated_kf);
    }
}
