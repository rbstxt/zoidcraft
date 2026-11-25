use std::io::Cursor;

use crate::*;

#[test]
fn bounded_undo_stack_never_exceeds_max_entries_or_max_bytes() {
    // 1. Entry count cap: 1,000 edits -> capped strictly at 200.
    let mut entry_capped = UndoStack::<String>::with_limits(200, DEFAULT_MAX_BYTES);
    let mut current = String::from("state-0");
    for i in 1..=1_000_usize {
        entry_capped.push(format!("edit-{i}"), current);
        current = format!("state-{i}");
        assert!(entry_capped.undo_len() <= 200);
        assert!(entry_capped.total_bytes() <= DEFAULT_MAX_BYTES);
    }
    assert_eq!(entry_capped.undo_len(), 200);

    // 2. Byte budget cap: 1,000 edits of 8 KiB each -> capped strictly at 64 KiB.
    let budget = 64 * 1024;
    let mut byte_capped = UndoStack::<Vec<u8>>::with_limits(500, budget);
    for i in 0..1_000_usize {
        let blob = vec![(i % 251) as u8; 8 * 1024];
        byte_capped.push(format!("blob-{i}"), blob);
        assert!(byte_capped.undo_len() <= 500);
        assert!(byte_capped.total_bytes() <= budget);
    }
    assert_eq!(byte_capped.undo_len(), 8);
    assert_eq!(byte_capped.total_bytes(), budget);

    // 3. Session-level cap across 1,000 live edits.
    let mut session = Session::with_history_limits(50, 256 * 1024);
    for i in 0..1_000_usize {
        session
            .execute(Command::SetProjectName {
                name: format!("Project {i}"),
            })
            .expect("rename succeeds");
        assert!(session.history.undo_len() <= 50);
        assert!(session.history.total_bytes() <= 256 * 1024);
    }
    assert!(session.history.undo_len() <= 50);
    assert!(session.history.total_bytes() <= 256 * 1024);
}

#[test]
fn live_transaction_rollback_and_single_step_commit() {
    let mut session = Session::new();
    session.mark_saved();
    assert!(!session.is_dirty());
    let initial_res = session.document().resolution;

    // Live drag scrubbed 25 times then cancelled (Escape) -> exact rollback, 0 undo entries.
    session
        .begin_transaction("Scrub Resolution")
        .expect("begin tx");
    for step in 1..=25_u32 {
        session
            .execute(Command::SetResolution {
                width: 1000 + step * 10,
                height: 720,
            })
            .expect("live edit");
    }
    assert_eq!(session.history.undo_len(), 0);
    assert!(session.is_dirty());

    session.cancel_transaction().expect("cancel tx");
    assert_eq!(session.document().resolution, initial_res);
    assert!(!session.history.can_undo());
    assert!(!session.is_dirty());

    // Live drag scrubbed 40 times then committed (mouse-up) -> EXACTLY ONE undo step.
    session
        .begin_transaction("Scrub Resolution Commit")
        .expect("begin tx 2");
    for step in 1..=40_u32 {
        session
            .execute(Command::SetResolution {
                width: 1280 + step * 16,
                height: 720 + step * 9,
            })
            .expect("live edit 2");
    }
    assert_eq!(session.history.undo_len(), 0);

    let committed = session.commit_transaction(true).expect("commit tx");
    assert!(committed);
    assert_eq!(session.history.undo_len(), 1);
    assert_eq!(
        session.document().resolution,
        [1280 + 40 * 16, 720 + 40 * 9]
    );
    assert!(session.is_dirty());

    // Single undo restores the pre-drag resolution cleanly.
    session.execute(Command::Undo).expect("undo");
    assert_eq!(session.document().resolution, initial_res);
    assert!(!session.is_dirty());

    // Single redo reapplies the final post-drag resolution cleanly.
    session.execute(Command::Redo).expect("redo");
    assert_eq!(
        session.document().resolution,
        [1280 + 40 * 16, 720 + 40 * 9]
    );
    assert!(session.is_dirty());
}

#[test]
fn json_lines_dispatch_round_trip() {
    let mut session = Session::new();

    let resp_name = session.dispatch_json_line(
        r#"{"id":"req-1","command":"project.set_name","params":{"name":"Demo Intro"}}"#,
    );
    assert!(resp_name.ok);
    assert_eq!(resp_name.id, json!("req-1"));
    assert_eq!(
        resp_name.result.as_ref().and_then(|v| v.get("name")),
        Some(&json!("Demo Intro"))
    );

    let resp_track = session.dispatch_json_line(
        r#"{"id":2,"command":"timeline.add_video_track","params":{"name":"V1"}}"#,
    );
    assert!(resp_track.ok);
    let track_id = resp_track
        .result
        .as_ref()
        .and_then(|v| v.get("track_id"))
        .and_then(Value::as_str)
        .expect("track_id present")
        .to_string();

    let add_clip_line = json!({
        "id": 3,
        "command": "timeline.add_clip",
        "params": {
            "track_id": track_id,
            "kind": "scene",
            "name": "3D Scene",
            "start": 0.0,
            "length": 120.0
        }
    })
    .to_string();
    let resp_clip = session.dispatch_json_line(&add_clip_line);
    assert!(resp_clip.ok);
    let clip_id = resp_clip
        .result
        .as_ref()
        .and_then(|v| v.get("clip_id"))
        .and_then(Value::as_str)
        .expect("clip_id present")
        .to_string();

    // Stream multiple JSONL requests through BufRead/Write.
    let batch = format!(
        "{}\n\n{}\n{}\n{}\n",
        json!({"id": "b1", "command": "playback.set_frame", "params": {"frame": 42.0}}),
        json!({"id": "b2", "command": "scene.add_object_3d", "params": {"clip_id": clip_id, "kind": "shape.box", "name": "Cube"}}),
        json!({"id": "b3", "command": "control.inspect_state", "params": {}}),
        json!({"id": "b4", "command": "control.list_commands", "params": {}}),
    );
    let mut output = Vec::new();
    let count = dispatch_json_lines(&mut session, Cursor::new(batch.as_bytes()), &mut output)
        .expect("stream dispatch");
    assert_eq!(count, 4);

    let lines: Vec<ControlResponse> = String::from_utf8(output)
        .expect("valid utf8")
        .lines()
        .map(|l| serde_json::from_str(l).expect("valid response json"))
        .collect();
    assert_eq!(lines.len(), 4);
    assert!(lines.iter().all(|r| r.ok));
    assert_eq!(lines[0].id, json!("b1"));
    assert!((session.playback.current_frame - 42.0).abs() < 1e-9);
    assert_eq!(
        lines[3].result.as_ref().and_then(|v| v.get("count")),
        Some(&json!(76))
    );

    // Malformed JSON and unknown commands return structured errors without panicking.
    let bad_json = session.dispatch_json_line("{not-valid-json");
    assert!(!bad_json.ok);
    assert_eq!(
        bad_json.error.as_ref().map(|e| e.code.as_str()),
        Some("parse_error")
    );

    let unknown =
        session.dispatch_json_line(r#"{"id":"bad","command":"does.not.exist","params":{}}"#);
    assert!(!unknown.ok);
    assert_eq!(unknown.id, json!("bad"));
    assert_eq!(
        unknown.error.as_ref().map(|e| e.code.as_str()),
        Some("invalid_command")
    );
}

#[test]
fn same_type_multi_selection_guard_enforced() {
    let mut sel = SelectionState::default();

    // Plain click selects a 3D shape.
    assert_eq!(
        sel.select_object_3d("obj-1", "object3d.shape", false),
        Ok(true)
    );
    // Shift+Click on another 3D shape succeeds.
    assert_eq!(
        sel.select_object_3d("obj-2", "object3d.shape", true),
        Ok(true)
    );
    assert_eq!(sel.object_3d_ids, vec!["obj-1", "obj-2"]);

    // Shift+Click on a different baseTypeString ("object3d.light") MUST be rejected
    // without altering the existing selection (Spec 01 §4.7).
    let mixed = sel.select_object_3d("light-1", "object3d.light", true);
    assert_eq!(
        mixed,
        Err(SelectionError::MixedType {
            domain: SelectionDomain::Object3D,
            expected: "object3d.shape".to_string(),
            actual: "object3d.light".to_string(),
        })
    );
    assert_eq!(sel.object_3d_ids, vec!["obj-1", "obj-2"]);

    // Shift+Click on already-selected item toggles it off.
    assert_eq!(
        sel.select_object_3d("obj-1", "object3d.shape", true),
        Ok(false)
    );
    assert_eq!(sel.object_3d_ids, vec!["obj-2"]);

    // Plain click replaces selection with the new type cleanly.
    assert_eq!(
        sel.select_object_3d("light-1", "object3d.light", false),
        Ok(true)
    );
    assert_eq!(sel.object_3d_ids, vec!["light-1"]);
    assert_eq!(sel.object_3d_base_type.as_deref(), Some("object3d.light"));

    // Same guard applies to Effects.
    assert_eq!(sel.select_effect("fx-1", "bloom", false), Ok(true));
    assert_eq!(sel.select_effect("fx-2", "bloom", true), Ok(true));
    assert!(matches!(
        sel.select_effect("fx-3", "boxblur", true),
        Err(SelectionError::MixedType {
            domain: SelectionDomain::Effect,
            ..
        })
    ));
    assert_eq!(sel.effect_ids, vec!["fx-1", "fx-2"]);
}

#[test]
fn clip_split_shifts_right_keyframes_and_keyframe_delete_preserves_last() {
    let mut session = Session::new();
    let track_id = session
        .execute(Command::AddVideoTrack { name: None })
        .expect("track")
        .get("track_id")
        .and_then(Value::as_str)
        .expect("id")
        .to_string();
    let clip_id = session
        .execute(Command::AddClip {
            track_id,
            kind: "scene".to_string(),
            name: Some("Scene".to_string()),
            start: 10.0,
            length: 100.0,
        })
        .expect("clip")
        .get("clip_id")
        .and_then(Value::as_str)
        .expect("id")
        .to_string();

    session
        .execute(Command::SetKeyframe {
            target_id: clip_id.clone(),
            property: "opacity".to_string(),
            frame: 60.0,
            value: json!(0.5),
            tween: Some(1),
        })
        .expect("set kf 60");

    // Delete the initial frame-0 keyframe first (leaving 60.0 as the sole keyframe),
    // then verify deleting the sole remaining keyframe is rejected (Spec 01 §4.2).
    session
        .execute(Command::DeleteKeyframe {
            target_id: clip_id.clone(),
            property: "opacity".to_string(),
            frame: 0.0,
        })
        .expect("delete initial kf 0");

    let err = session
        .execute(Command::DeleteKeyframe {
            target_id: clip_id.clone(),
            property: "opacity".to_string(),
            frame: 60.0,
        })
        .expect_err("sole keyframe cannot be deleted");
    assert_eq!(err.code(), "invalid_operation");

    // Split clip at sequence frame 50 (local offset 40) -> right-half keyframe moves to 20.
    let split_res = session
        .execute(Command::SplitClip {
            clip_id: clip_id.clone(),
            at_frame: 50.0,
        })
        .expect("split clip");
    let right_id = split_res
        .get("right_clip_id")
        .and_then(Value::as_str)
        .expect("right id");
    let track = session
        .document()
        .video_tracks
        .iter()
        .find(|t| t.clips.iter().any(|c| c.id == right_id))
        .expect("track containing right clip");
    let left_clip = track
        .clips
        .iter()
        .find(|c| c.id == clip_id)
        .expect("left clip exists");
    let right_clip = track
        .clips
        .iter()
        .find(|c| c.id == right_id)
        .expect("right clip exists");
    assert!((left_clip.length - 40.0).abs() < 1e-9);
    assert!((left_clip.time_end - (40.0 / 30.0)).abs() < 1e-9);
    assert!((right_clip.start - 50.0).abs() < 1e-9);
    assert!((right_clip.length - 60.0).abs() < 1e-9);
    assert!((right_clip.time_start - (40.0 / 30.0)).abs() < 1e-9);
    assert!((right_clip.time_end - (100.0 / 30.0)).abs() < 1e-9);
    let shifted_frame = right_clip.properties["opacity"].keyframes[0].frame;
    assert!((shifted_frame - 20.0).abs() < 1e-9);
}

#[test]
fn session_edits_persist_to_project_and_preserve_embedded_assets_and_undo_budget() {
    let mut project = zc_project::Project::new_cm3_default();
    let asset_bytes = vec![0xABu8; 1024 * 1024]; // 1 MiB embedded media
    let asset_key = project.insert_binary_asset(
        zc_project::AssetKind::Image,
        "hero.png",
        asset_bytes.clone(),
    );
    project
        .plugin_requirements
        .insert("expressions-plus".to_string());

    let mut session = Session::from_project(project, zc_plugin::PluginRegistry::new());
    // Default project has 1 video track with 1 Scene clip and length 180.
    assert_eq!(session.document().video_tracks.len(), 1);
    assert_eq!(session.document().video_tracks[0].clips.len(), 1);
    assert!((session.document().sequence_length - 180.0).abs() < 1e-9);

    // Plugin lock must honor project.plugin_requirements ("expressions-plus").
    assert!(
        session
            .active_plugin_dependencies()
            .contains("expressions-plus")
    );
    let disable_err = session.execute(Command::DisablePlugin {
        plugin_id: "expressions-plus".to_string(),
    });
    assert!(disable_err.is_err(), "in-use plugin cannot be disabled");

    // Perform edits and verify they write back to session.project without dropping asset bytes.
    session
        .execute(Command::SetProjectName {
            name: "Persisted Edit".to_string(),
        })
        .expect("rename");
    session
        .execute(Command::SetFps { fps: 60.0 })
        .expect("set fps");
    assert_eq!(session.project.name, "Persisted Edit");
    assert_eq!(session.project.sequence.fps, 60);
    assert_eq!(
        session.project.assets[&asset_key].bytes.as_deref(),
        Some(asset_bytes.as_slice())
    );

    // History snapshots deduplicate asset blobs instead of cloning 1 MiB per edit,
    // while a direct snapshot carrying raw bytes accurately reports >= 1 MiB.
    assert!(session.history.total_bytes() < 256 * 1024);
    assert!(session.snapshot().estimated_bytes() >= asset_bytes.len());

    // Round-trip save/load preserves edits AND embedded asset bytes AND undo restores cleanly.
    let archive = zc_project::save_zcraft_to_bytes(&session.project).expect("save zcraft");
    let reloaded = zc_project::load_zcraft_from_bytes(&archive).expect("load zcraft");
    assert_eq!(reloaded.name, "Persisted Edit");
    assert_eq!(reloaded.sequence.fps, 60);
    assert_eq!(
        reloaded.assets[&asset_key].bytes.as_deref(),
        Some(asset_bytes.as_slice())
    );

    // Undo restores FPS and rehydrates embedded asset bytes cleanly.
    session.execute(Command::Undo).expect("undo fps");
    assert_eq!(session.project.sequence.fps, 30);
    assert_eq!(
        session.project.assets[&asset_key].bytes.as_deref(),
        Some(asset_bytes.as_slice())
    );
}

#[test]
fn timeline_enforces_non_overlap_and_updates_media_time_endpoints() {
    let mut session = Session::new();
    let track_id = session
        .execute(Command::AddVideoTrack {
            name: Some("V2".to_string()),
        })
        .expect("add track")
        .get("track_id")
        .and_then(Value::as_str)
        .expect("id")
        .to_string();

    // Place clip A at [0, 10) and clip B at [5, 15) -> A is right-trimmed to [0, 5), no overlap.
    let clip_a = session
        .execute(Command::AddClip {
            track_id: track_id.clone(),
            kind: "video".to_string(),
            name: Some("A".to_string()),
            start: 0.0,
            length: 10.0,
        })
        .expect("add A")
        .get("clip_id")
        .and_then(Value::as_str)
        .expect("id")
        .to_string();
    let clip_b = session
        .execute(Command::AddClip {
            track_id: track_id.clone(),
            kind: "video".to_string(),
            name: Some("B".to_string()),
            start: 5.0,
            length: 10.0,
        })
        .expect("add B")
        .get("clip_id")
        .and_then(Value::as_str)
        .expect("id")
        .to_string();

    let track = session
        .document()
        .video_tracks
        .iter()
        .find(|t| t.id == track_id)
        .expect("track");
    assert_eq!(track.clips.len(), 2);
    assert!((track.clips[0].start - 0.0).abs() < 1e-9);
    assert!((track.clips[0].length - 5.0).abs() < 1e-9);
    assert!((track.clips[0].time_end - (5.0 / 30.0)).abs() < 1e-9);
    assert!((track.clips[1].start - 5.0).abs() < 1e-9);
    assert!((track.clips[1].length - 10.0).abs() < 1e-9);
    assert!(track.clips[0].start + track.clips[0].length <= track.clips[1].start + 1e-9);

    // Trim B left by +3 frames -> start=8, length=7, time_start advances by 3/30s.
    session
        .execute(Command::TrimClipLeft {
            clip_id: clip_b.clone(),
            delta_frames: 3.0,
            rate_stretch: false,
        })
        .expect("trim left");
    let b_after_left = session
        .document()
        .video_tracks
        .iter()
        .find(|t| t.id == track_id)
        .unwrap()
        .clips
        .iter()
        .find(|c| c.id == clip_b)
        .unwrap();
    assert!((b_after_left.start - 8.0).abs() < 1e-9);
    assert!((b_after_left.length - 7.0).abs() < 1e-9);
    assert!((b_after_left.time_start - (3.0 / 30.0)).abs() < 1e-9);
    assert!((b_after_left.time_end - (10.0 / 30.0)).abs() < 1e-9);

    // Trim A right by +5 frames into [0, 10) -> overwrites B's left edge cleanly with zero overlap.
    session
        .execute(Command::TrimClipRight {
            clip_id: clip_a,
            delta_frames: 5.0,
            rate_stretch: false,
        })
        .expect("trim right");
    let clips = &session
        .document()
        .video_tracks
        .iter()
        .find(|t| t.id == track_id)
        .unwrap()
        .clips;
    assert_eq!(clips.len(), 2);
    assert!((clips[0].length - 10.0).abs() < 1e-9);
    assert!((clips[1].start - 10.0).abs() < 1e-9);
    assert!((clips[1].length - 5.0).abs() < 1e-9);
    assert!(clips[0].start + clips[0].length <= clips[1].start + 1e-9);
}

#[test]
fn round2_typed_properties_cloned_clip_payloads_asset_versions_and_clip_links() {
    let mut project = zc_project::Project::new_cm3_default();
    // Add an Image clip with an effect carrying Color and Asset properties + a linked Audio clip.
    let img_clip_id = zc_project::Uuid::new_v4();
    let mut img_layer = zc_project::Layer::new(
        "Logo",
        zc_project::LayerKind::Image {
            image: Some("logo-sha".to_string()),
        },
        [1920, 1080],
    );
    let mut fx = zc_project::create_effect_by_id("coloroverlay").expect("coloroverlay");
    fx.params.insert(
        "asset_ref".to_string(),
        zc_project::PropertyTrack {
            animated: false,
            expression: None,
            static_behavior: zc_project::StaticBehavior::Constant,
            value: zc_project::PropertyValue::Asset(Some("lut-sha".to_string())),
            keyframes: Vec::new(),
            channels: Vec::new(),
        },
    );
    img_layer.effects.push(fx);
    let img_clip = zc_project::VideoClip {
        id: img_clip_id,
        name: "Logo".to_string(),
        start: 0.0,
        length: 60.0,
        link: Some(1),
        media: Some("media-v".to_string()),
        time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
        layer: img_layer,
        preserved: BTreeMap::new(),
    };
    project.sequence.video_tracks[0].clips = vec![img_clip];

    let audio_clip_id = zc_project::Uuid::new_v4();
    project.sequence.audio_tracks = vec![zc_project::AudioTrack {
        id: zc_project::Uuid::new_v4(),
        name: "A1".to_string(),
        enabled: true,
        muted: false,
        solo: false,
        locked: false,
        clips: vec![zc_project::AudioClip {
            id: audio_clip_id,
            name: "Music".to_string(),
            start: 0.0,
            length: 60.0,
            link: Some(1),
            media: Some("song-asset".to_string()),
            time_curve: zc_project::PropertyTrack::linear_time_curve(60.0, 30),
            audio: zc_project::AudioClipData::default(),
            preserved: BTreeMap::new(),
        }],
        preserved: BTreeMap::new(),
    }];
    project
        .sequence
        .clip_links
        .insert(1, vec![img_clip_id, audio_clip_id]);

    // Add one asset with bytes and one missing asset with bytes == None.
    let k_present =
        project.insert_binary_asset(zc_project::AssetKind::Image, "v1.png", vec![1u8, 2, 3, 4]);
    project.assets.insert(
        "missing-key".to_string(),
        zc_project::AssetRecord {
            id: "missing-key".to_string(),
            kind: zc_project::AssetKind::Image,
            source: zc_project::AssetSource::File,
            filename: "gone.png".to_string(),
            url: None,
            external: false,
            bytes: None,
            missing: true,
            sha256: String::new(),
            size: 0,
            preserved: BTreeMap::new(),
        },
    );

    let mut session = Session::from_project(project, zc_plugin::PluginRegistry::new());

    // 1. Unrelated edit must NOT corrupt Option/Color/Asset PropertyValue variants.
    session
        .execute(Command::SetProjectName {
            name: "Typed".to_string(),
        })
        .expect("rename");
    let fx_after = &session.project.sequence.video_tracks[0].clips[0]
        .layer
        .effects[0];
    assert!(matches!(
        fx_after.enabled.value,
        zc_project::PropertyValue::Option(_)
    ));
    assert!(matches!(
        fx_after.params["color"].value,
        zc_project::PropertyValue::Color(_)
    ));
    assert_eq!(
        fx_after.params["asset_ref"].value,
        zc_project::PropertyValue::Asset(Some("lut-sha".to_string()))
    );

    // 2. Split image clip -> right half MUST retain LayerKind::Image { image: Some("logo-sha") }.
    session
        .execute(Command::SplitClip {
            clip_id: img_clip_id.to_string(),
            at_frame: 30.0,
        })
        .expect("split image clip");
    assert_eq!(session.project.sequence.video_tracks[0].clips.len(), 2);
    assert!(matches!(
        &session.project.sequence.video_tracks[0].clips[1].layer.kind,
        zc_project::LayerKind::Image { image: Some(id) } if id == "logo-sha"
    ));

    // 3. Duplicate audio clip -> duplicate MUST retain `media: Some("song-asset")`.
    session
        .execute(Command::DuplicateClip {
            clip_id: audio_clip_id.to_string(),
        })
        .expect("duplicate audio clip");
    assert_eq!(
        session.project.sequence.audio_tracks[0].clips[1]
            .media
            .as_deref(),
        Some("song-asset")
    );

    // 4. Engulf original image clip via AddClip -> stale UUID is pruned from sequence.clip_links.
    let vtrack_id = session.document().video_tracks[0].id.clone();
    session
        .execute(Command::AddClip {
            track_id: vtrack_id,
            kind: "video".to_string(),
            name: Some("Cover".to_string()),
            start: 0.0,
            length: 30.0,
        })
        .expect("engulf left half");
    assert!(
        !session.project.sequence.clip_links[&1].contains(&img_clip_id),
        "engulfed clip UUID must be pruned from clip_links"
    );

    // 5. Mutating bytes under an existing asset key + Undo restores exact original bytes
    //    while leaving originally-None assets as None.
    session
        .execute(Command::SetProjectName {
            name: "BeforeBlobReplace".to_string(),
        })
        .expect("edit 1");
    session.project.assets.get_mut(&k_present).unwrap().bytes = Some(vec![9u8, 9, 9]);
    session
        .execute(Command::SetProjectName {
            name: "AfterBlobReplace".to_string(),
        })
        .expect("edit 2");
    session.execute(Command::Undo).expect("undo edit 2");
    assert_eq!(
        session.project.assets[&k_present].bytes.as_deref(),
        Some(&[9u8, 9, 9][..])
    );
    session.execute(Command::Undo).expect("undo edit 1");
    assert_eq!(
        session.project.assets[&k_present].bytes.as_deref(),
        Some(&[1u8, 2, 3, 4][..])
    );
    assert!(session.project.assets["missing-key"].bytes.is_none());
}

#[test]
fn round3_cloned_child_payloads_and_shifted_keyframe_types_and_bezier_handles_survive() {
    let mut project = zc_project::Project::new_cm3_default();
    // Attach a custom GLSL shader effect AND a 3D object with custom material + preserved
    // + a shifted Color keyframe with custom Bezier handles on a Vec3-base track.
    let clip_id = project.sequence.video_tracks[0].clips[0].id;
    let custom_handles = [[-4.5, 1.25], [7.25, -2.5]];
    let mut custom_fx =
        zc_project::create_custom_shader("MyShader", "void main() { gl_FragColor = vec4(1.0); }");
    custom_fx.params.insert(
        "tint".to_string(),
        zc_project::PropertyTrack {
            animated: true,
            expression: None,
            static_behavior: zc_project::StaticBehavior::Constant,
            value: zc_project::PropertyValue::Vec3([0.0, 0.0, 0.0]),
            keyframes: vec![zc_project::PropertyKeyframe::from_cm3_tween(
                40.0,
                zc_project::PropertyValue::Color([0.2, 0.4, 0.8]),
                257,
                custom_handles,
                false,
            )],
            channels: Vec::new(),
        },
    );
    project.sequence.video_tracks[0].clips[0]
        .layer
        .effects
        .push(custom_fx);

    if let zc_project::LayerKind::Scene3D {
        ref mut objects, ..
    } = project.sequence.video_tracks[0].clips[0].layer.kind
    {
        objects[0]
            .preserved
            .insert("customTag".to_string(), json!("keep-on-clone"));
    }

    let mut session = Session::from_project(project, zc_plugin::PluginRegistry::new());
    // Split at frame 30 -> right half clones 3D objects & custom shader effect with new IDs
    // AND shifts the keyframe from frame 40 to frame 10.
    session
        .execute(Command::SplitClip {
            clip_id: clip_id.to_string(),
            at_frame: 30.0,
        })
        .expect("split scene clip");

    let right_clip = &session.project.sequence.video_tracks[0].clips[1];
    // Cloned 3D object keeps preserved metadata & material.
    if let zc_project::LayerKind::Scene3D { ref objects, .. } = right_clip.layer.kind {
        assert_eq!(
            objects[0].preserved.get("customTag"),
            Some(&json!("keep-on-clone"))
        );
    } else {
        panic!("expected Scene3D on right half");
    }

    // Cloned custom shader effect keeps frag_shader AND shifted keyframe keeps Color type + Bezier handles.
    let right_fx = &right_clip.layer.effects[0];
    assert!(
        !right_fx.params.contains_key("tint"),
        "CustomShader uniforms must not duplicate into fx.params"
    );
    let zc_project::EffectKind::CustomShader {
        frag_shader,
        custom_properties,
    } = &right_fx.kind
    else {
        panic!("expected CustomShader");
    };
    assert!(frag_shader.contains("gl_FragColor"));
    let tint_cp = custom_properties
        .iter()
        .find(|cp| cp.name == "tint")
        .expect("tint custom property");
    let shifted_kf = &tint_cp.track.keyframes[0];
    assert!((shifted_kf.frame - 10.0).abs() < 1e-9);
    assert_eq!(
        shifted_kf.value,
        zc_project::PropertyValue::Color([0.2, 0.4, 0.8])
    );
    assert_eq!(shifted_kf.control_points, custom_handles);
    assert!(!shifted_kf.continuous_tangent);
}

#[test]
fn move_clips_atomic_preserves_adjacent_clips_without_trimming() {
    let mut session = Session::new();
    let track_id = session.document().video_tracks[0].id.clone();
    // Replace default clip with two adjacent clips: [0, 10) and [10, 20).
    let default_clip_id = session.document().video_tracks[0].clips[0].id.clone();
    session
        .execute(Command::DeleteClip {
            clip_id: default_clip_id,
        })
        .expect("delete default clip");
    let c1 = session
        .execute(Command::AddClip {
            track_id: track_id.clone(),
            kind: "scene".to_string(),
            name: Some("Clip A".to_string()),
            start: 0.0,
            length: 10.0,
        })
        .expect("add clip A")["clip_id"]
        .as_str()
        .unwrap()
        .to_string();
    let c2 = session
        .execute(Command::AddClip {
            track_id: track_id.clone(),
            kind: "scene".to_string(),
            name: Some("Clip B".to_string()),
            start: 10.0,
            length: 10.0,
        })
        .expect("add clip B")["clip_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Moving both right by +5 frames must produce exact [5, 15) and [15, 25) with zero trimming.
    session
        .move_clips_atomic(&[
            (c1.clone(), track_id.clone(), 5.0),
            (c2.clone(), track_id.clone(), 15.0),
        ])
        .expect("atomic group move");
    let clips = &session.document().video_tracks[0].clips;
    assert_eq!(clips.len(), 2);
    assert!((clips[0].start - 5.0).abs() < 1e-9);
    assert!((clips[0].length - 10.0).abs() < 1e-9);
    assert!((clips[1].start - 15.0).abs() < 1e-9);
    assert!((clips[1].length - 10.0).abs() < 1e-9);

    // Overlapping targets inside the selected group must be rejected atomically without mutating state.
    let err = session.move_clips_atomic(&[
        (c1.clone(), track_id.clone(), 10.0),
        (c2.clone(), track_id.clone(), 15.0),
    ]);
    assert!(err.is_err());
    let clips_after_err = &session.document().video_tracks[0].clips;
    assert!((clips_after_err[0].start - 5.0).abs() < 1e-9);
    assert!((clips_after_err[1].start - 15.0).abs() < 1e-9);

    // Single Undo restores [0, 10) and [10, 20).
    session.execute(Command::Undo).expect("undo group move");
    let clips_undone = &session.document().video_tracks[0].clips;
    assert!((clips_undone[0].start - 0.0).abs() < 1e-9);
    assert!((clips_undone[1].start - 10.0).abs() < 1e-9);
}

#[test]
fn set_keyframe_bezier_handles_persists_to_project_and_undoes_cleanly() {
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    session
        .execute(Command::SetKeyframe {
            target_id: clip_id.clone(),
            property: "opacity".to_string(),
            frame: 0.0,
            value: json!(0.0),
            tween: Some(1),
        })
        .expect("kf0");
    session
        .execute(Command::SetKeyframe {
            target_id: clip_id.clone(),
            property: "opacity".to_string(),
            frame: 30.0,
            value: json!(1.0),
            tween: Some(1),
        })
        .expect("kf1");

    let handles_0 = [[-5.0, 0.0], [2.0, 0.9]];
    let handles_1 = [[-2.0, -0.1], [5.0, 0.0]];
    let bezier_tween = (1u16 << 8) | 1;
    session
        .set_keyframe_bezier_handles(&clip_id, "opacity", 0.0, handles_0, false, None)
        .expect("set kf0 handles");
    session
        .set_keyframe_bezier_handles(
            &clip_id,
            "opacity",
            30.0,
            handles_1,
            false,
            Some(bezier_tween),
        )
        .expect("set kf1 handles");

    let proj_track = &session.project.sequence.video_tracks[0].clips[0]
        .layer
        .opacity;
    assert_eq!(proj_track.keyframes[0].control_points, handles_0);
    assert!(!proj_track.keyframes[0].continuous_tangent);
    assert_eq!(proj_track.keyframes[1].control_points, handles_1);
    assert!(proj_track.keyframes[1].is_bezier);

    session.execute(Command::Undo).expect("undo kf1 handles");
    let proj_track_after = &session.project.sequence.video_tracks[0].clips[0]
        .layer
        .opacity;
    assert!(!proj_track_after.keyframes[1].is_bezier);
}

#[test]
fn scene3d_transparent_background_and_backdrop_opacity_round_trip() {
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();

    // Default Scene3D clip starts Transparent (sceneBackground = 0, no preserved backgroundColor).
    assert_eq!(
        prop_f64_static(
            &session.document().video_tracks[0].clips[0].properties,
            "sceneBackground"
        ),
        Some(0.0)
    );
    assert!(
        !session.project.sequence.video_tracks[0].clips[0]
            .layer
            .preserved
            .contains_key("backgroundColor")
    );

    // Unrelated edit MUST NOT inject opaque black into vc.layer.preserved["backgroundColor"].
    session
        .execute(Command::SetProjectName {
            name: "Still Transparent".to_string(),
        })
        .expect("rename");
    assert!(
        !session.project.sequence.video_tracks[0].clips[0]
            .layer
            .preserved
            .contains_key("backgroundColor"),
        "transparent Scene3D clip must stay transparent after unrelated edits"
    );

    // Switch to Color mode with custom RGB & opacity -> writes 4-element RGBA.
    session
        .execute(Command::SetPropertyValue {
            target_id: clip_id.clone(),
            property: "sceneBackground".to_string(),
            value: json!(1),
        })
        .expect("set sceneBackground");
    session
        .execute(Command::SetPropertyValue {
            target_id: clip_id.clone(),
            property: "backgroundColor".to_string(),
            value: json!([0.2, 0.4, 0.6]),
        })
        .expect("set backgroundColor");
    session
        .execute(Command::SetPropertyValue {
            target_id: clip_id.clone(),
            property: "backdropOpacity".to_string(),
            value: json!(0.5),
        })
        .expect("set backdropOpacity");
    assert_eq!(
        session.project.sequence.video_tracks[0].clips[0]
            .layer
            .preserved
            .get("backgroundColor"),
        Some(&json!([0.2, 0.4, 0.6, 0.5]))
    );

    // Reloading via Session::from_project preserves sceneBackground=1, RGB, and backdropOpacity=0.5.
    let reloaded = Session::from_project(session.project.clone(), zc_plugin::PluginRegistry::new());
    let props = &reloaded.document().video_tracks[0].clips[0].properties;
    assert_eq!(prop_f64_static(props, "sceneBackground"), Some(1.0));
    assert_eq!(prop_f64_static(props, "backdropOpacity"), Some(0.5));
    assert_eq!(
        props["backgroundColor"].keyframes[0].value,
        json!([0.2, 0.4, 0.6])
    );

    // Switching back to Transparent removes backgroundColor from preserved.
    session
        .execute(Command::SetPropertyValue {
            target_id: clip_id,
            property: "sceneBackground".to_string(),
            value: json!(0),
        })
        .expect("back to transparent");
    assert!(
        !session.project.sequence.video_tracks[0].clips[0]
            .layer
            .preserved
            .contains_key("backgroundColor")
    );
}

#[test]
fn remove_property_deletes_custom_shader_uniform_and_undoes_cleanly() {
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let fx_id = session
        .execute(Command::AddEffect {
            clip_id,
            kind: "customshader".to_string(),
            name: Some("Shader".to_string()),
        })
        .expect("add customshader")["effect_id"]
        .as_str()
        .unwrap()
        .to_string();

    session
        .execute(Command::SetKeyframe {
            target_id: fx_id.clone(),
            property: "u_gain".to_string(),
            frame: 15.0,
            value: json!(2.5),
            tween: Some(3),
        })
        .expect("set uniform kf");
    session
        .execute(Command::SetExpression {
            target_id: fx_id.clone(),
            property: "u_gain".to_string(),
            expression: Some("time * 2".to_string()),
        })
        .expect("set uniform expr");
    session
        .selection
        .select_keyframe(KeyframeRef::new(&fx_id, "u_gain", None, 15.0), false);

    let prev_prop =
        session.document().video_tracks[0].clips[0].effects[0].properties["u_gain"].clone();

    // RemoveProperty deletes from document, project custom_properties, project fx.params, and selection.
    session
        .execute(Command::RemoveProperty {
            target_id: fx_id.clone(),
            property: "u_gain".to_string(),
        })
        .expect("remove property");
    assert!(
        !session.document().video_tracks[0].clips[0].effects[0]
            .properties
            .contains_key("u_gain")
    );
    assert!(session.selection.keyframes.is_empty());
    let proj_fx = &session.project.sequence.video_tracks[0].clips[0]
        .layer
        .effects[0];
    assert!(!proj_fx.params.contains_key("u_gain"));
    if let zc_project::EffectKind::CustomShader {
        custom_properties, ..
    } = &proj_fx.kind
    {
        assert!(custom_properties.iter().all(|cp| cp.name != "u_gain"));
    } else {
        panic!("expected CustomShader");
    }

    // Removing again returns NotFound.
    assert_eq!(
        session
            .execute(Command::RemoveProperty {
                target_id: fx_id.clone(),
                property: "u_gain".to_string(),
            })
            .expect_err("missing property")
            .code(),
        "not_found"
    );

    // Undo restores exact previous SessionProperty (keyframes + expression) and project state.
    session.execute(Command::Undo).expect("undo remove");
    assert_eq!(
        session.document().video_tracks[0].clips[0].effects[0].properties["u_gain"],
        prev_prop
    );
    let restored_fx = &session.project.sequence.video_tracks[0].clips[0]
        .layer
        .effects[0];
    assert!(
        !restored_fx.params.contains_key("u_gain"),
        "CustomShader uniforms must not duplicate into fx.params"
    );
    if let zc_project::EffectKind::CustomShader {
        custom_properties, ..
    } = &restored_fx.kind
    {
        assert!(custom_properties.iter().any(|cp| cp.name == "u_gain"));
    } else {
        panic!("expected CustomShader");
    }

    // Redo removes it again.
    session.execute(Command::Redo).expect("redo remove");
    assert!(
        !session.document().video_tracks[0].clips[0].effects[0]
            .properties
            .contains_key("u_gain")
    );
    assert!(
        !session.project.sequence.video_tracks[0].clips[0]
            .layer
            .effects[0]
            .params
            .contains_key("u_gain")
    );
}

#[test]
fn non_group_3d_parenting_and_invalid_clip_targets_are_rejected() {
    let mut session = Session::new();
    let vtrack_id = session.document().video_tracks[0].id.clone();
    let scene_clip_id = session.document().video_tracks[0].clips[0].id.clone();

    let box_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_clip_id.clone(),
            parent_id: None,
            kind: "shape.box".to_string(),
            name: Some("Cube".to_string()),
        })
        .expect("add box")["object_id"]
        .as_str()
        .unwrap()
        .to_string();
    let sphere_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_clip_id.clone(),
            parent_id: None,
            kind: "shape.sphere".to_string(),
            name: Some("Sphere".to_string()),
        })
        .expect("add sphere")["object_id"]
        .as_str()
        .unwrap()
        .to_string();
    let group_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_clip_id.clone(),
            parent_id: None,
            kind: "group".to_string(),
            name: Some("Group".to_string()),
        })
        .expect("add group")["object_id"]
        .as_str()
        .unwrap()
        .to_string();

    // Parenting under a non-Group box fails in both AddObject3D and ReorderObject3D.
    assert_eq!(
        session
            .execute(Command::AddObject3D {
                clip_id: scene_clip_id.clone(),
                parent_id: Some(box_id.clone()),
                kind: "shape.circle".to_string(),
                name: None,
            })
            .expect_err("cannot parent under box")
            .code(),
        "invalid_operation"
    );
    assert_eq!(
        session
            .execute(Command::ReorderObject3D {
                clip_id: scene_clip_id.clone(),
                object_id: sphere_id.clone(),
                parent_id: Some(box_id),
                index: 0,
            })
            .expect_err("cannot reparent under box")
            .code(),
        "invalid_operation"
    );
    // Parenting under a Group succeeds and survives sync into project.
    session
        .execute(Command::ReorderObject3D {
            clip_id: scene_clip_id,
            object_id: sphere_id,
            parent_id: Some(group_id),
            index: 0,
        })
        .expect("reparent under group");

    // Mismatched track/clip kinds, AddObject3D on video clip, AddEffect on audio clip are rejected.
    let atrack_id = session.document().audio_tracks[0].id.clone();
    assert_eq!(
        session
            .execute(Command::AddClip {
                track_id: vtrack_id.clone(),
                kind: "audio".to_string(),
                name: None,
                start: 200.0,
                length: 30.0,
            })
            .expect_err("audio clip on video track")
            .code(),
        "invalid_operation"
    );
    assert_eq!(
        session
            .execute(Command::AddClip {
                track_id: atrack_id.clone(),
                kind: "scene".to_string(),
                name: None,
                start: 0.0,
                length: 30.0,
            })
            .expect_err("scene clip on audio track")
            .code(),
        "invalid_operation"
    );

    let video_clip_id = session
        .execute(Command::AddClip {
            track_id: vtrack_id,
            kind: "video".to_string(),
            name: Some("Vid".to_string()),
            start: 200.0,
            length: 30.0,
        })
        .expect("add video clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(
        session
            .execute(Command::AddObject3D {
                clip_id: video_clip_id,
                parent_id: None,
                kind: "shape.box".to_string(),
                name: None,
            })
            .expect_err("3d object on video clip")
            .code(),
        "invalid_operation"
    );

    let audio_clip_id = session
        .execute(Command::AddClip {
            track_id: atrack_id,
            kind: "audio".to_string(),
            name: Some("Aud".to_string()),
            start: 0.0,
            length: 30.0,
        })
        .expect("add audio clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(
        session
            .execute(Command::AddEffect {
                clip_id: audio_clip_id,
                kind: "bloom".to_string(),
                name: None,
            })
            .expect_err("effect on audio clip")
            .code(),
        "invalid_operation"
    );
}

#[test]
fn bezier_time_curve_survives_load_and_time_edits_sync_time_animated() {
    let mut project = zc_project::Project::new_cm3_default();
    let bezier_handles = [[-5.0, -0.2], [5.0, 0.2]];
    project.sequence.video_tracks[0].clips[0].time_curve = zc_project::PropertyTrack {
        animated: true,
        expression: None,
        static_behavior: zc_project::StaticBehavior::Linear,
        value: zc_project::PropertyValue::Number(0.0),
        keyframes: vec![
            zc_project::PropertyKeyframe::linear(0.0, zc_project::PropertyValue::Number(0.0)),
            zc_project::PropertyKeyframe::from_cm3_tween(
                180.0,
                zc_project::PropertyValue::Number(6.0),
                (1 << 8) | 1,
                bezier_handles,
                false,
            ),
        ],
        channels: Vec::new(),
    };

    let mut session = Session::from_project(project, zc_plugin::PluginRegistry::new());
    assert!(session.document().video_tracks[0].clips[0].time_animated);
    session
        .execute(Command::SetProjectName {
            name: "Keep Bezier Time".to_string(),
        })
        .expect("rename");
    let tc = &session.project.sequence.video_tracks[0].clips[0].time_curve;
    assert!(tc.keyframes[1].is_bezier);
    assert_eq!(tc.keyframes[1].control_points, bezier_handles);

    // Editing "time" on a linear clip marks time_animated = true so trimming doesn't wipe keyframes.
    let mut s2 = Session::new();
    let clip_id = s2.document().video_tracks[0].clips[0].id.clone();
    s2.execute(Command::SetKeyframe {
        target_id: clip_id.clone(),
        property: "time".to_string(),
        frame: 90.0,
        value: json!(4.5),
        tween: Some(3),
    })
    .expect("set time kf");
    assert!(s2.document().video_tracks[0].clips[0].time_animated);
    s2.execute(Command::TrimClipRight {
        clip_id: clip_id.clone(),
        delta_frames: -10.0,
        rate_stretch: false,
    })
    .expect("trim right");
    assert_eq!(
        s2.document().video_tracks[0].clips[0].properties["time"]
            .keyframes
            .len(),
        3
    );

    // ResetPropertyList clears expression on "time".
    s2.execute(Command::SetExpression {
        target_id: clip_id.clone(),
        property: "time".to_string(),
        expression: Some("time * 0.5".to_string()),
    })
    .expect("set time expr");
    s2.execute(Command::ResetPropertyList {
        target_id: clip_id.clone(),
    })
    .expect("reset props");
    assert!(
        s2.document().video_tracks[0].clips[0].properties["time"]
            .expression
            .is_none()
    );

    // SetPropertyValue on linear clip "time" updates clip.time_start so trim keeps it.
    s2.execute(Command::SetPropertyValue {
        target_id: clip_id.clone(),
        property: "time".to_string(),
        value: json!(1.5),
    })
    .expect("set time static");
    assert!((s2.document().video_tracks[0].clips[0].time_start - 1.5).abs() < 1e-9);
}

#[test]
fn asset_blob_gc_evicts_unreferenced_digests() {
    let mut project = zc_project::Project::new_cm3_default();
    let k = project.insert_binary_asset(zc_project::AssetKind::Image, "img.png", vec![1u8; 4096]);
    let mut session = Session::from_project(project, zc_plugin::PluginRegistry::new());
    session.history.set_limits(2, DEFAULT_MAX_BYTES);

    for i in 0..5_u8 {
        session.project.assets.get_mut(&k).unwrap().bytes = Some(vec![i; 4096]);
        session
            .execute(Command::SetProjectName {
                name: format!("Edit {i}"),
            })
            .expect("edit");
    }
    assert!(
        session.asset_blobs.len() <= 2,
        "dead asset blobs must be garbage-collected on history eviction, found {}",
        session.asset_blobs.len()
    );
}

#[test]
fn plugin_deps_restore_undo_external_locks_bare_zoidium_and_custom_plugins() {
    // 1. Stuck lock regression after Undo.
    let mut s = Session::new();
    let clip_id = s.document().video_tracks[0].clips[0].id.clone();
    s.execute(Command::SetExpression {
        target_id: clip_id.clone(),
        property: "opacity".to_string(),
        expression: Some("0.5".to_string()),
    })
    .expect("add expr");
    s.execute(Command::SetExpression {
        target_id: clip_id.clone(),
        property: "opacity".to_string(),
        expression: None,
    })
    .expect("clear expr");
    s.execute(Command::Undo).expect("undo clear expr");
    s.execute(Command::SetExpression {
        target_id: clip_id.clone(),
        property: "opacity".to_string(),
        expression: None,
    })
    .expect("clear expr again");
    s.execute(Command::DisablePlugin {
        plugin_id: "expressions-plus".to_string(),
    })
    .expect("plugin lock must release after undo + re-remove");

    // 2. External plugin requirement survives add -> undo -> unrelated edit.
    let mut proj = zc_project::Project::new_cm3_default();
    proj.plugin_requirements
        .insert("expressions-plus".to_string());
    let mut s2 = Session::from_project(proj, zc_plugin::PluginRegistry::new());
    let clip_id2 = s2.document().video_tracks[0].clips[0].id.clone();
    s2.execute(Command::SetExpression {
        target_id: clip_id2,
        property: "opacity".to_string(),
        expression: Some("0.5".to_string()),
    })
    .expect("add doc expr");
    s2.execute(Command::Undo).expect("undo doc expr");
    s2.execute(Command::SetProjectName {
        name: "Unrelated".to_string(),
    })
    .expect("unrelated edit");
    assert!(
        s2.project.plugin_requirements.contains("expressions-plus"),
        "external plugin requirement must not be dropped"
    );

    // 3. Bare Zoidium effect normalizes kind & locks its plugin.
    let mut s3 = Session::new();
    let clip_id3 = s3.document().video_tracks[0].clips[0].id.clone();
    s3.execute(Command::EnablePlugin {
        plugin_id: "openzoid-legacy".to_string(),
    })
    .expect("enable openzoid-legacy");
    let fx_res = s3
        .execute(Command::AddEffect {
            clip_id: clip_id3,
            kind: "vhs".to_string(),
            name: None,
        })
        .expect("add bare vhs");
    assert_eq!(fx_res["kind"], json!("zoidium:openzoid-legacy/vhs"));
    assert!(s3.active_plugin_dependencies().contains("openzoid-legacy"));
    assert!(
        s3.execute(Command::DisablePlugin {
            plugin_id: "openzoid-legacy".to_string(),
        })
        .is_err()
    );

    // 3b. Structural project plugin dependencies are detected without JSON prefix strings.
    let mut struct_proj = zc_project::Project::new_cm3_default();
    struct_proj.sequence.video_tracks[0].is_3d = true;
    struct_proj
        .sequence
        .controls
        .push(zc_project::SequenceControl {
            id: "ctl_gain".to_string(),
            name: "gain".to_string(),
            kind: zc_project::SequenceControlKind::Slider,
            track: zc_project::PropertyTrack::number(1.0),
            preserved: BTreeMap::new(),
        });
    let mut cone = zc_project::Object3D::new(
        "Cone",
        zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Cone {
            radius: 5.0,
            top_radius: 0.0,
            height: 10.0,
            detail: 16,
            open_ended: false,
            shading: 0,
        }),
    );
    cone.material = Some(zc_project::MaterialSpec::PbrPlus(Box::default()));
    cone.constraints.track_to = Some(zc_project::TrackToConstraint {
        target_id: "cam".to_string(),
        track_axis: 5,
        up_axis: 1,
        influence: zc_project::PropertyTrack::number(1.0),
    });
    if let zc_project::LayerKind::Scene3D {
        ref mut objects, ..
    } = struct_proj.sequence.video_tracks[0].clips[0].layer.kind
    {
        objects.push(cone);
    }
    struct_proj.sequence.video_tracks[0]
        .clips
        .push(zc_project::VideoClip {
            id: zc_project::Uuid::new_v4(),
            name: "Precomp".to_string(),
            start: 200.0,
            length: 30.0,
            link: None,
            media: None,
            time_curve: zc_project::PropertyTrack::linear_time_curve(30.0, 30),
            layer: zc_project::Layer::new(
                "Precomp",
                zc_project::LayerKind::PrecompRef {
                    composition_id: zc_project::Uuid::new_v4(),
                },
                [1920, 1080],
            ),
            preserved: BTreeMap::new(),
        });
    let s_struct = Session::from_project(struct_proj, zc_plugin::PluginRegistry::new());
    let deps = s_struct.active_plugin_dependencies();
    for expected in [
        "camera-plus",
        "expressions-plus",
        "geometry-plus",
        "material-plus",
        "constraints",
        "precomp-plus",
    ] {
        assert!(deps.contains(expected), "missing structural dep {expected}");
    }

    // 4. Custom registered plugin appears in ListPlugins & DisableAllOptionalPlugins.
    let mut s4 = Session::new();
    s4.plugins
        .register(zc_plugin::PluginDescriptor {
            id: "my-custom-pack".to_string(),
            name: "My Custom Pack".to_string(),
            author: "Test".to_string(),
            version: "1.0.0".to_string(),
            category: zc_plugin::PluginCategory::Adjustment,
            kind: zc_plugin::PluginKind::ShaderPack,
            tagline: "Custom".to_string(),
            description: "Custom".to_string(),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [255, 255, 255],
            features: Vec::new(),
        })
        .expect("register custom plugin");
    let listed = s4.execute(Command::ListPlugins).expect("list plugins");
    assert!(
        listed["plugins"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["id"] == "my-custom-pack")
    );
    s4.execute(Command::DisableAllOptionalPlugins)
        .expect("disable all optional");
    assert!(s4.plugins.is_vanilla_cm3_mode());
    assert!(!s4.plugins.is_enabled("my-custom-pack"));
}

#[test]
fn add_clip_defaults_and_property_sync_and_toggle_tween_semantics() {
    let mut session = Session::new();
    let vtrack_id = session.document().video_tracks[0].id.clone();

    // AddClip populates default properties so TogglePropertyAnimated preserves opacity=1 & scale=[1,1].
    let shape_id = session
        .execute(Command::AddClip {
            track_id: vtrack_id,
            kind: "shape".to_string(),
            name: Some("Shape".to_string()),
            start: 200.0,
            length: 60.0,
        })
        .expect("add shape")["clip_id"]
        .as_str()
        .unwrap()
        .to_string();
    session
        .execute(Command::TogglePropertyAnimated {
            target_id: shape_id.clone(),
            property: "opacity".to_string(),
            frame: 10.0,
        })
        .expect("toggle opacity");
    let shape_clip = session.document().video_tracks[0]
        .clips
        .iter()
        .find(|c| c.id == shape_id)
        .unwrap();
    assert_eq!(
        shape_clip.properties["opacity"].keyframes[0].value,
        json!(1.0)
    );
    assert_eq!(
        shape_clip.properties["scale"].keyframes[0].value,
        json!([1.0, 1.0])
    );

    // SetExpression(None) on missing property must NOT insert Value::Null.
    session
        .execute(Command::SetExpression {
            target_id: shape_id.clone(),
            property: "nonexistent".to_string(),
            expression: None,
        })
        .expect("clear missing expr");
    let shape_clip = session.document().video_tracks[0]
        .clips
        .iter()
        .find(|c| c.id == shape_id)
        .unwrap();
    assert!(!shape_clip.properties.contains_key("nonexistent"));

    // Animated strokeWidth starting at 0.0 creates DrawOp::Stroke in project.
    session
        .execute(Command::SetKeyframe {
            target_id: shape_id.clone(),
            property: "strokeWidth".to_string(),
            frame: 0.0,
            value: json!(0.0),
            tween: Some(1),
        })
        .expect("stroke kf0");
    session
        .execute(Command::SetKeyframe {
            target_id: shape_id.clone(),
            property: "strokeWidth".to_string(),
            frame: 30.0,
            value: json!(8.0),
            tween: Some(1),
        })
        .expect("stroke kf1");
    let proj_shape = session.project.sequence.video_tracks[0]
        .clips
        .iter()
        .find(|c| c.id.to_string() == shape_id)
        .unwrap();
    if let zc_project::LayerKind::Shape2D { shapes } = &proj_shape.layer.kind {
        assert!(
            shapes.iter().any(|s| s
                .operations
                .iter()
                .any(|op| matches!(op, zc_project::DrawOp::Stroke { .. }))),
            "animated strokeWidth starting at 0 must create DrawOp::Stroke"
        );
    } else {
        panic!("expected Shape2D");
    }

    // Per-layer resolution survives edits.
    session
        .execute(Command::SetPropertyValue {
            target_id: shape_id.clone(),
            property: "resolution".to_string(),
            value: json!([640.0, 360.0]),
        })
        .expect("set custom layer res");
    session
        .execute(Command::SetProjectName {
            name: "Keep Res".to_string(),
        })
        .expect("rename");
    let proj_shape = session.project.sequence.video_tracks[0]
        .clips
        .iter()
        .find(|c| c.id.to_string() == shape_id)
        .unwrap();
    assert_eq!(proj_shape.layer.resolution, [640, 360]);

    // TogglePropertyAnimated uses right keyframe's tween (b.tween == 0 holds, a.tween == 0 with b.tween == 1 interpolates).
    let clip0 = session.document().video_tracks[0].clips[0].id.clone();
    session
        .execute(Command::SetKeyframe {
            target_id: clip0.clone(),
            property: "opacity".to_string(),
            frame: 0.0,
            value: json!(0.0),
            tween: Some(0),
        })
        .expect("kf0 hold");
    session
        .execute(Command::SetKeyframe {
            target_id: clip0.clone(),
            property: "opacity".to_string(),
            frame: 10.0,
            value: json!(1.0),
            tween: Some(1),
        })
        .expect("kf1 linear");
    session
        .execute(Command::TogglePropertyAnimated {
            target_id: clip0.clone(),
            property: "opacity".to_string(),
            frame: 5.0,
        })
        .expect("collapse linear segment");
    let v = session.document().video_tracks[0].clips[0].properties["opacity"].keyframes[0]
        .value
        .as_f64()
        .unwrap();
    assert!((v - 0.5).abs() < 1e-6, "expected 0.5, got {v}");

    session
        .execute(Command::SetKeyframe {
            target_id: clip0.clone(),
            property: "rotation".to_string(),
            frame: 0.0,
            value: json!(10.0),
            tween: Some(1),
        })
        .expect("rot kf0");
    session
        .execute(Command::SetKeyframe {
            target_id: clip0.clone(),
            property: "rotation".to_string(),
            frame: 10.0,
            value: json!(90.0),
            tween: Some(0),
        })
        .expect("rot kf1 hold");
    session
        .execute(Command::TogglePropertyAnimated {
            target_id: clip0,
            property: "rotation".to_string(),
            frame: 5.0,
        })
        .expect("collapse hold segment");
    let r = session.document().video_tracks[0].clips[0].properties["rotation"].keyframes[0]
        .value
        .as_f64()
        .unwrap();
    assert!((r - 10.0).abs() < 1e-6, "expected held 10.0, got {r}");
}
