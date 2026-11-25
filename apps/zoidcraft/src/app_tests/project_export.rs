//! project export tests.

use super::*;

#[test]
fn disconnected_export_worker_reports_error_and_unblocks_session() {
    let (tx, rx) = mpsc::channel::<ExportWorkerEvent>();
    let cancel = Arc::new(AtomicBool::new(false));
    let handle = thread::spawn(move || {
        let _tx = tx;
        // Simulate abrupt worker exit/panic without sending ExportWorkerEvent::Finished.
    });
    let _ = handle.join();

    let job = ActiveExportJob {
        cancel,
        rx,
        handle: None,
        started_at: Instant::now(),
        queue_job_id: None,
    };
    let mut status = ExportStatus {
        exporting: true,
        ..ExportStatus::default()
    };
    let res = poll_export_job(&job, &mut status);
    assert!(
        matches!(res, Some(Err(ref msg)) if msg.contains("terminated unexpectedly")),
        "expected disconnected worker error, got {res:?}"
    );
}

#[test]
fn save_without_assets_rejects_overwriting_current_project_file() {
    let dir = std::env::temp_dir();
    let current = dir.join("zoidcraft_current_test.zcraft");
    let separate = dir.join("zoidcraft_current_test-no-assets.zcraft");

    assert!(
        validate_save_without_assets_target(Some(&current), &current).is_err(),
        "saving without assets over the active project file must be refused"
    );
    assert!(
        validate_save_without_assets_target(Some(&current), &separate).is_ok(),
        "saving without assets to a distinct path must be allowed"
    );
    assert!(
        validate_save_without_assets_target(None, &current).is_ok(),
        "saving without assets when no current file exists must be allowed"
    );
}

#[test]
fn media_import_transaction_marks_session_dirty_and_undoes_cleanly() {
    let mut session = Session::new();
    session.mark_saved();
    assert!(!session.is_dirty());

    session
        .begin_transaction("Import logo.png")
        .expect("begin import tx");
    let asset_id =
        session
            .project
            .insert_binary_asset(AssetKind::Image, "logo.png", vec![1, 2, 3, 4]);
    session.project.media.push(MediaItem {
        id: Uuid::new_v4(),
        name: "logo.png".to_owned(),
        asset_id: Some(asset_id.clone()),
        kind: AssetKind::Image,
        preserved: BTreeMap::new(),
    });
    assert!(session.commit_transaction(true).expect("commit import tx"));

    assert!(
        session.is_dirty(),
        "media-only import must mark session dirty so unsaved-changes prompt triggers"
    );
    assert!(session.project.assets.contains_key(&asset_id));

    session.execute(Command::Undo).expect("undo media import");
    assert!(!session.is_dirty());
    assert!(!session.project.assets.contains_key(&asset_id));
}

#[test]
fn active_export_bounded_cancel_never_hangs_on_stalled_worker() {
    let (_tx, rx) = mpsc::channel::<ExportWorkerEvent>();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_worker = Arc::clone(&cancel);
    let handle = thread::spawn(move || {
        // Simulate a stalled worker that ignores cancel for 300ms.
        let _ = cancel_worker;
        thread::sleep(Duration::from_millis(300));
    });
    let mut job = ActiveExportJob {
        cancel: Arc::clone(&cancel),
        rx,
        handle: Some(handle),
        started_at: Instant::now(),
        queue_job_id: None,
    };
    let started = Instant::now();
    let joined = job.cancel_and_join_bounded(Duration::from_millis(25));
    assert!(cancel.load(Ordering::Acquire), "cancel flag must be set");
    assert!(!joined, "stalled worker should detach after bounded wait");
    assert!(
        started.elapsed() < Duration::from_millis(200),
        "bounded teardown must not block UI thread for the full stall duration"
    );
}

#[test]
fn classify_dropped_path_recognizes_projects_and_media() {
    assert_eq!(
        classify_dropped_path(Path::new("/tmp/demo.pz")),
        DroppedFileKind::Project
    );
    assert_eq!(
        classify_dropped_path(Path::new("/tmp/Demo.ZCRAFT")),
        DroppedFileKind::Project
    );
    for media in [
        "a.png", "b.jpg", "c.mp4", "d.wav", "e.mp3", "f.ogg", "g.ttf", "h.otf", "i.obj", "j.glb",
        "k.cube",
    ] {
        assert_eq!(
            classify_dropped_path(Path::new(media)),
            DroppedFileKind::Media,
            "{media} should classify as Media"
        );
    }
    assert_eq!(
        classify_dropped_path(Path::new("/tmp/readme.txt")),
        DroppedFileKind::Unsupported
    );
}

#[test]
fn convert_export_settings_maps_gif_wav_m4a_and_transparent_alpha() {
    // GIF: maps to Gif, disables audio, preserves transparent_alpha, ignores stale video_codec
    let gif_ui = UiExportSettings {
        container: UiExportContainer::Gif,
        video_codec: ExportVideoCodec::Png,
        audio_codec: ExportAudioCodec::AacLc,
        transparent_alpha: true,
        width: 320,
        height: 180,
        fps: 15.0,
        ..UiExportSettings::default()
    };
    let gif_core = convert_export_settings(&gif_ui, 30.0, 0, 60, PathBuf::from("anim.gif"))
        .expect("gif settings convert");
    assert_eq!(gif_core.container, CoreExportContainer::Gif);
    assert_eq!(gif_core.audio_codec, AudioCodec::None);
    assert!(gif_core.transparent_alpha);
    assert_eq!(gif_core.end_frame, 30);
    gif_core.validate().expect("gif settings validate");

    // WAV: audio-only fast path, forces PCM, clears transparent_alpha, keeps exact sequence duration
    let wav_ui = UiExportSettings {
        container: UiExportContainer::Wav,
        video_codec: ExportVideoCodec::Png,
        audio_codec: ExportAudioCodec::None,
        transparent_alpha: true,
        fps: 60.0,
        sample_rate_hz: 44_100,
        ..UiExportSettings::default()
    };
    let wav_core = convert_export_settings(&wav_ui, 30.0, 10, 40, PathBuf::from("out.wav"))
        .expect("wav settings convert");
    assert_eq!(wav_core.container, CoreExportContainer::Wav);
    assert!(wav_core.container.is_audio_only());
    assert_eq!(wav_core.audio_codec, AudioCodec::Pcm);
    assert!(!wav_core.transparent_alpha);
    assert_eq!(wav_core.fps, 30.0);
    assert_eq!(wav_core.end_frame, 30);
    assert_eq!(wav_core.sample_rate, 44_100);
    wav_core.validate().expect("wav settings validate");

    // M4A: audio-only fast path, forces AAC, clears transparent_alpha, preserves bitrate & sample rate
    let m4a_ui = UiExportSettings {
        container: UiExportContainer::M4a,
        video_codec: ExportVideoCodec::Png,
        audio_codec: ExportAudioCodec::None,
        audio_bitrate_kbps: 256,
        sample_rate_hz: 48_000,
        transparent_alpha: true,
        ..UiExportSettings::default()
    };
    let m4a_core = convert_export_settings(&m4a_ui, 24.0, 0, 48, PathBuf::from("out.m4a"))
        .expect("m4a settings convert");
    assert_eq!(m4a_core.container, CoreExportContainer::M4a);
    assert!(m4a_core.container.is_audio_only());
    assert_eq!(m4a_core.audio_codec, AudioCodec::Aac);
    assert!(!m4a_core.transparent_alpha);
    assert_eq!(m4a_core.audio_bitrate_kbps, 256);
    assert_eq!(m4a_core.fps, 24.0);
    assert_eq!(m4a_core.end_frame, 48);
    m4a_core.validate().expect("m4a settings validate");

    // WebM/MKV VP9 transparent alpha propagates and validates cleanly
    for (ui_c, core_c, ac) in [
        (
            UiExportContainer::WebM,
            CoreExportContainer::Webm,
            ExportAudioCodec::Opus,
        ),
        (
            UiExportContainer::Mkv,
            CoreExportContainer::Mkv,
            ExportAudioCodec::PcmS16Le,
        ),
    ] {
        let ui = UiExportSettings {
            container: ui_c,
            video_codec: ExportVideoCodec::Vp9,
            audio_codec: ac,
            transparent_alpha: true,
            ..UiExportSettings::default()
        };
        let core = convert_export_settings(&ui, 30.0, 0, 30, PathBuf::from("alpha.out"))
            .expect("vp9 alpha convert");
        assert_eq!(core.container, core_c);
        assert_eq!(core.video_codec, VideoCodec::Vp9);
        assert!(core.transparent_alpha);
        core.validate().expect("vp9 alpha validates");
    }

    // MP4 rejects PNG video codec and rejects transparent_alpha
    let mp4_png = UiExportSettings {
        container: UiExportContainer::Mp4,
        video_codec: ExportVideoCodec::Png,
        ..UiExportSettings::default()
    };
    assert!(convert_export_settings(&mp4_png, 30.0, 0, 30, PathBuf::from("bad.mp4")).is_err());

    let mp4_alpha = UiExportSettings {
        container: UiExportContainer::Mp4,
        video_codec: ExportVideoCodec::Vp9,
        transparent_alpha: true,
        ..UiExportSettings::default()
    };
    let mp4_core = convert_export_settings(&mp4_alpha, 30.0, 0, 30, PathBuf::from("bad.mp4"))
        .expect("convert mp4");
    assert!(mp4_core.validate().is_err());

    // File-save dialog filters for GIF / WAV / M4A
    assert_eq!(
        export_file_dialog_filter(UiExportContainer::Gif),
        ("GIF Animation", "gif")
    );
    assert_eq!(
        export_file_dialog_filter(UiExportContainer::Wav),
        ("WAV Audio", "wav")
    );
    assert_eq!(
        export_file_dialog_filter(UiExportContainer::M4a),
        ("M4A Audio", "m4a")
    );
}

#[test]
fn render_export_frame_propagates_transparent_alpha_background() {
    let Ok(mut engine) = ProjectRenderEngine::new_headless() else {
        // Skip GPU readback assertion on headless environments without an adapter.
        return;
    };
    let mut session = Session::new();
    // Clear video clips so the canvas background is directly visible.
    session.project.sequence.video_tracks.clear();

    let mut transparent_rgba = Vec::new();
    render_export_frame_into(
        &mut engine,
        &session,
        0.0,
        16,
        16,
        true,
        &mut transparent_rgba,
    )
    .expect("transparent frame render");
    assert_eq!(transparent_rgba.len(), 16 * 16 * 4);
    assert!(
        transparent_rgba
            .as_chunks::<4>()
            .0
            .iter()
            .all(|&px| px == [0, 0, 0, 0]),
        "transparent export background must be [0, 0, 0, 0], got first px {:?}",
        &transparent_rgba[..4]
    );

    let mut opaque_rgba = Vec::new();
    render_export_frame_into(&mut engine, &session, 0.0, 16, 16, false, &mut opaque_rgba)
        .expect("opaque frame render");
    assert_eq!(opaque_rgba.len(), 16 * 16 * 4);
    assert!(
        opaque_rgba.as_chunks::<4>().0.iter().all(|px| px[3] == 255),
        "opaque export background must have alpha 255, got first px {:?}",
        &opaque_rgba[..4]
    );
}

#[test]
fn prepare_project_for_pz_export_round_trips_via_export_and_import_pz() {
    let mut session = Session::new();
    session
        .execute(Command::SetProjectName {
            name: "Export Demo".to_owned(),
        })
        .expect("rename project");

    let prepared = prepare_project_for_pz_export(&session);
    let tmp_dir = std::env::temp_dir().join(format!(
        "zoidcraft_gui_export_pz_test_{}",
        std::process::id()
    ));
    let _ = fs::create_dir_all(&tmp_dir);
    let out_path = tmp_dir.join(format!("{}.pz", sanitize_filename(&prepared.name)));

    export_pz(&prepared, &out_path).expect("export_pz must succeed");
    let reimported = import_pz(&out_path).expect("import_pz must read exported archive");
    let _ = fs::remove_dir_all(&tmp_dir);

    assert_eq!(reimported.project.name, "Export Demo");
    assert_eq!(
        reimported.project.sequence.resolution,
        prepared.sequence.resolution
    );
    assert_eq!(reimported.project.sequence.fps, prepared.sequence.fps);
}

#[test]
fn autosave_snapshot_save_list_and_prune_in_temp_dir() {
    let dir = std::env::temp_dir().join(format!(
        "zoidcraft_autosave_test_{}_{}",
        std::process::id(),
        Uuid::new_v4()
    ));
    let _ = fs::remove_dir_all(&dir);

    let mut project = Project::new_cm3_default();
    project.name = "Recovery Test Project".to_owned();

    // Save 5 snapshots with max_keep = 3; older ones must be pruned automatically.
    for idx in 0..5 {
        project.name = format!("Recovery Snapshot {idx}");
        let saved = save_recovery_snapshot_to_dir(&project, &dir, 3)
            .expect("save recovery snapshot must succeed");
        assert!(saved.path.is_file());
        assert!(saved.size_bytes > 0);
        assert!(saved.filename.starts_with("autosave-"));
        assert!(saved.filename.ends_with(".zcraft"));
    }

    let listed = list_recovery_snapshots_in_dir(&dir);
    assert_eq!(
        listed.len(),
        3,
        "pruning must retain only the newest 3 recovery snapshots"
    );
    // Latest snapshot must be readable via load_zcraft and preserve project data.
    let restored = load_zcraft(&listed[0].path).expect("restored snapshot must load cleanly");
    assert!(restored.name.starts_with("Recovery Snapshot"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn export_queue_add_reorder_status_progression_and_eta_formatting() {
    let mut queue = Vec::new();
    let cfg_a = UiExportSettings {
        container: UiExportContainer::Mp4,
        width: 1920,
        height: 1080,
        fps: 60.0,
        ..UiExportSettings::default()
    };
    let cfg_b = UiExportSettings {
        container: UiExportContainer::WebM,
        width: 1280,
        height: 720,
        fps: 30.0,
        ..UiExportSettings::default()
    };

    queue_add_job(
        &mut queue,
        1,
        format_export_job_label("Intro", &cfg_a),
        PathBuf::from("/tmp/intro.mp4"),
        cfg_a,
    );
    queue_add_job(
        &mut queue,
        2,
        format_export_job_label("Teaser", &cfg_b),
        PathBuf::from("/tmp/teaser.webm"),
        cfg_b,
    );
    assert_eq!(queue.len(), 2);
    assert_eq!(queue_next_pending_index(&queue), Some(0));

    // Reorder job 2 above job 1.
    assert!(queue_move_job(&mut queue, 1, -1));
    assert_eq!(queue[0].id, 2);
    assert_eq!(queue[1].id, 1);

    // Running job cannot be reordered; status progression selects next Queued item.
    queue[0].status = ExportJobStatus::Running;
    assert!(!queue_move_job(&mut queue, 0, 1));
    assert_eq!(queue_next_pending_index(&queue), Some(1));
    queue[0].status = ExportJobStatus::Completed;
    queue[1].status = ExportJobStatus::Completed;
    assert_eq!(queue_next_pending_index(&queue), None);

    // Live FPS & ETA calculations + formatting.
    let (fps, eta) = compute_export_fps_and_eta(50, 300, 2.0);
    assert!((fps.unwrap() - 25.0).abs() < 1e-6);
    assert!((eta.unwrap() - 10.0).abs() < 1e-6);
    assert_eq!(format_eta_mm_ss(10.0), "00:10");
    assert_eq!(format_eta_mm_ss(125.0), "02:05");
    assert_eq!(format_eta_mm_ss(3665.0), "1:01:05");
    assert_eq!(compute_export_fps_and_eta(0, 300, 2.0), (None, None));
}

#[test]
fn queued_export_snapshot_preserves_original_project_across_project_switch_and_cleanup() {
    let mut app = make_headless_app_for_test();

    // Configure Project A with a distinct name, fps, and active composition.
    app.session
        .execute(Command::SetProjectName {
            name: "Project Alpha".to_owned(),
        })
        .expect("rename project A");
    app.session
        .execute(Command::SetFps { fps: 60.0 })
        .expect("set project A fps");
    let comp_res = app
        .session
        .execute(Command::CreateComposition {
            name: "Alpha Subcomp".to_owned(),
            resolution: [1280, 720],
            fps: 48,
            length_frames: 120,
        })
        .expect("create comp");
    let comp_id_str = comp_res["comp_id"].as_str().unwrap().to_owned();
    let comp_uuid = Uuid::parse_str(&comp_id_str).unwrap();
    app.session
        .execute(Command::SetActiveComposition {
            comp_id: Some(comp_uuid),
        })
        .expect("activate comp");

    let cfg_a = UiExportSettings {
        container: UiExportContainer::Mp4,
        width: 1280,
        height: 720,
        fps: 48.0,
        ..UiExportSettings::default()
    };
    let job_a = app.enqueue_export_job(cfg_a, PathBuf::from("/tmp/alpha_out.mp4"));

    // Switch the active session to a completely different Project B.
    app.perform_new_project();
    app.session
        .execute(Command::SetProjectName {
            name: "Project Beta".to_owned(),
        })
        .expect("rename project B");
    app.session
        .execute(Command::SetFps { fps: 24.0 })
        .expect("set project B fps");

    // Resolving Job A's snapshot MUST return Project A (not Project B!).
    let snap_a = app
        .export_queue_snapshots
        .get(&job_a)
        .expect("job A snapshot must survive project switch");
    assert_eq!(snap_a.project.name, "Project Alpha");
    assert_eq!(
        snap_a.active_composition_id.as_deref(),
        Some(comp_id_str.as_str())
    );
    assert!((snap_a.effective_fps() - 48.0).abs() < 1e-6);
    assert_eq!(app.session.document().project_name, "Project Beta");

    // Enqueue jobs for Project B and verify Remove & Clear Finished clean up snapshots.
    let job_b = app.enqueue_export_job(
        UiExportSettings::default(),
        PathBuf::from("/tmp/beta_remove.mp4"),
    );
    let job_c = app.enqueue_export_job(
        UiExportSettings::default(),
        PathBuf::from("/tmp/beta_finished.mp4"),
    );
    assert_eq!(app.export_queue_snapshots.len(), 3);

    app.remove_export_queue_job(job_b);
    assert!(!app.export_queue_snapshots.contains_key(&job_b));
    assert_eq!(app.export_queue_snapshots.len(), 2);

    if let Some(job) = app
        .workspace
        .panels
        .export_queue
        .iter_mut()
        .find(|j| j.id == job_c)
    {
        job.status = ExportJobStatus::Completed;
    }
    app.clear_finished_export_queue();
    assert!(!app.export_queue_snapshots.contains_key(&job_c));
    assert!(app.export_queue_snapshots.contains_key(&job_a));
    assert_eq!(app.export_queue_snapshots.len(), 1);
}

#[test]
fn queued_export_from_child_composition_promotes_and_mixes_child_not_root() {
    use zc_project::{
        AudioClip, AudioClipData, AudioTrack, Layer, PropertyTrack, Sequence, VideoClip, VideoTrack,
    };

    let sr = STANDARD_SAMPLE_RATE;
    let mut project = Project::new_cm3_default();
    project.name = "Child Comp Export Test".to_owned();
    project.sequence.name = "Root Comp".to_owned();
    project.sequence.fps = 30;
    project.sequence.exact_fps = None;
    project.sequence.length_frames = 60.0;
    let root_id = project.sequence.id;

    let root_wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.80, 0.80]; sr as usize * 2], sr);
    let child_wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.25, 0.25]; sr as usize * 2], sr);
    let root_audio_id =
        project.insert_binary_asset(AssetKind::Av, "root.wav".to_owned(), root_wav.clone());
    let child_audio_id =
        project.insert_binary_asset(AssetKind::Av, "child.wav".to_owned(), child_wav.clone());
    let root_decoded =
        Arc::new(decode_audio_bytes(&root_wav, Some("wav")).expect("decode root wav"));
    let child_decoded =
        Arc::new(decode_audio_bytes(&child_wav, Some("wav")).expect("decode child wav"));

    // Root sequence has distinct video clip ("Root Video") and loud audio (0.80).
    project.sequence.video_tracks = vec![VideoTrack {
        id: Uuid::new_v4(),
        name: "V1".to_owned(),
        enabled: true,
        solo: false,
        locked: false,
        is_3d: false,
        clips: vec![VideoClip {
            id: Uuid::new_v4(),
            name: "Root Video".to_owned(),
            start: 0.0,
            length: 60.0,
            link: None,
            media: None,
            time_curve: PropertyTrack::linear_time_curve(60.0, 30),
            layer: Layer::new(
                "Root Video",
                LayerKind::Shape2D { shapes: Vec::new() },
                [1920, 1080],
            ),
            preserved: BTreeMap::new(),
        }],
        preserved: BTreeMap::new(),
    }];
    project.sequence.audio_tracks = vec![AudioTrack {
        id: Uuid::new_v4(),
        name: "A1".to_owned(),
        enabled: true,
        muted: false,
        solo: false,
        locked: false,
        clips: vec![AudioClip {
            id: Uuid::new_v4(),
            name: "Root Audio".to_owned(),
            start: 0.0,
            length: 60.0,
            link: None,
            media: Some(root_audio_id.clone()),
            time_curve: PropertyTrack::linear_time_curve(60.0, 30),
            audio: AudioClipData::default(),
            preserved: BTreeMap::new(),
        }],
        preserved: BTreeMap::new(),
    }];

    // Child composition has different FPS (48), distinct video ("Child Video"), and quieter audio (0.25).
    let child_comp_id = Uuid::new_v4();
    let child_seq = Sequence {
        id: child_comp_id,
        name: "Child Comp".to_owned(),
        resolution: [1280, 720],
        fps: 48,
        exact_fps: Some(48.0),
        length_frames: 48.0,
        video_tracks: vec![VideoTrack {
            id: Uuid::new_v4(),
            name: "V1".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![VideoClip {
                id: Uuid::new_v4(),
                name: "Child Video".to_owned(),
                start: 0.0,
                length: 48.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(48.0, 48),
                layer: Layer::new(
                    "Child Video",
                    LayerKind::Shape2D { shapes: Vec::new() },
                    [1280, 720],
                ),
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        }],
        audio_tracks: vec![AudioTrack {
            id: Uuid::new_v4(),
            name: "A1".to_owned(),
            enabled: true,
            muted: false,
            solo: false,
            locked: false,
            clips: vec![AudioClip {
                id: Uuid::new_v4(),
                name: "Child Audio".to_owned(),
                start: 0.0,
                length: 48.0,
                link: None,
                media: Some(child_audio_id.clone()),
                time_curve: PropertyTrack::linear_time_curve(48.0, 48),
                audio: AudioClipData::default(),
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        }],
        ..Sequence::default()
    };
    project.compositions.insert(child_comp_id, child_seq);

    let mut app = make_headless_app_for_test();
    app.session = Session::from_project(project, PluginRegistry::new());
    app.decoded_audio
        .insert(root_audio_id.clone(), root_decoded);
    app.decoded_audio
        .insert(child_audio_id.clone(), child_decoded);

    // Activate child composition and enqueue export job.
    app.session
        .execute(Command::SetActiveComposition {
            comp_id: Some(child_comp_id),
        })
        .expect("activate child composition");
    let ui_cfg = UiExportSettings {
        container: UiExportContainer::Wav,
        sample_rate_hz: sr,
        fps: 48.0,
        start_frame: 0.0,
        end_frame: 24.0,
        ..UiExportSettings::default()
    };
    let job_id = app.enqueue_export_job(ui_cfg.clone(), PathBuf::from("/tmp/child_export.wav"));
    let snap = app
        .export_queue_snapshots
        .remove(&job_id)
        .expect("captured snapshot");

    let seq_fps = snap.effective_fps();
    assert!((seq_fps - 48.0).abs() < 1e-6);

    let mut export_project = snap.project;
    promote_active_composition_for_export(
        &mut export_project,
        snap.active_composition_id.as_deref(),
    );

    // Promoted export project must render child composition (not root), while retaining root in compositions.
    assert_eq!(export_project.sequence.id, child_comp_id);
    assert_eq!(export_project.sequence.name, "Child Comp");
    assert_eq!(export_project.sequence.fps, 48);
    assert_eq!(
        export_project.sequence.video_tracks[0].clips[0].name,
        "Child Video"
    );
    assert!(
        export_project.compositions.contains_key(&root_id),
        "previous root sequence must be preserved in compositions map"
    );

    let core_settings = convert_export_settings(
        &ui_cfg,
        seq_fps,
        0,
        24,
        PathBuf::from("/tmp/child_export.wav"),
    )
    .expect("core export settings");

    // Audio arc collection must pick only child_audio (not root_audio).
    let export_audio_buffers = collect_in_range_audio_arcs_for_project(
        &export_project,
        snap.plugins.is_enabled("precomp-plus"),
        &snap.decoded_audio,
        &core_settings,
        0,
        24,
    );
    assert!(export_audio_buffers.contains_key(&child_audio_id));
    assert!(!export_audio_buffers.contains_key(&root_audio_id));

    // Offline audio mix must come from child composition (~0.25) at 48 fps (24/48 = 0.5s = 24_000 samples).
    let export_session = Session::from_project(export_project, snap.plugins);
    let mixed = build_offline_export_audio(
        &export_session,
        &export_audio_buffers,
        &core_settings,
        seq_fps,
        0,
        24,
    )
    .expect("offline audio mix succeeds")
    .expect("offline audio produces stereo samples");
    assert_eq!(
        mixed.len(),
        (sr / 2) as usize,
        "24 frames at 48 fps must produce 0.5s of audio samples"
    );
    let mid = mixed[mixed.len() / 2];
    assert!(
        (mid[0] - 0.25).abs() < 0.02 && (mid[1] - 0.25).abs() < 0.02,
        "exported audio must mix child composition (~0.25), not root (~0.80), got {mid:?}"
    );
}

#[test]
fn promote_active_composition_preserves_root_only_controls_as_fallback() {
    let mut project = Project::new_cm3_default();
    let root_ctrl = zc_project::SequenceControl {
        id: "ctl-global-gain".to_owned(),
        name: "GlobalGain".to_owned(),
        kind: zc_project::SequenceControlKind::Slider,
        track: zc_project::PropertyTrack::constant(zc_project::PropertyValue::Number(0.75)),
        preserved: BTreeMap::new(),
    };
    project.sequence.controls.push(root_ctrl);

    let mut child = Project::new_cm3_default().sequence;
    let child_id = Uuid::new_v4();
    child.id = child_id;
    child.name = "Child".to_owned();
    child.controls.clear();
    project.compositions.insert(child_id, child);

    promote_active_composition_for_export(&mut project, Some(&child_id.to_string()));
    assert_eq!(project.sequence.id, child_id);
    assert!(
        project
            .sequence
            .controls
            .iter()
            .any(|c| c.id == "ctl-global-gain" && c.name == "GlobalGain"),
        "promoted child export must retain root-only sequence controls as fallbacks"
    );
}

#[test]
fn video_import_creates_new_track_above_at_playhead_when_v1_overlaps() {
    let mut app = make_headless_app_for_test();
    app.session.playback.current_frame = 12.0;

    // V1 already has the default Scene clip spanning [0, 300), so no existing video track is free at frame 12.
    assert!(choose_free_video_track_at(&app.session, 12.0, 150.0).is_none());

    // Write a tiny valid 2x2 PNG to temp dir and import it via `import_media_from_path`.
    let png_bytes = encode_png(&[255u8; 2 * 2 * 4], 2, 2).expect("encode tiny png");
    let tmp_png = std::env::temp_dir().join("zoidcraft_qa_import_overlap_test.png");
    fs::write(&tmp_png, &png_bytes).expect("write tiny png");

    let msg = app
        .import_media_from_path(&tmp_png)
        .expect("import image at overlapping playhead");
    let _ = fs::remove_file(&tmp_png);
    assert!(msg.contains("Imported"), "unexpected toast: {msg}");

    // Must have created V2 above V1 and placed the new image clip right at frame 12!
    assert_eq!(
        app.session.document().video_tracks.len(),
        2,
        "overlapping import must create a new video track above V1"
    );
    let v2_clips = &app.session.document().video_tracks[1].clips;
    assert_eq!(v2_clips.len(), 1);
    assert_eq!(v2_clips[0].kind, "image");
    assert!(
        (v2_clips[0].start - 12.0).abs() < 1e-6,
        "newly imported image clip must start at current_frame=12, got {}",
        v2_clips[0].start
    );
    assert_eq!(
        app.session.selection.clip_ids.first().map(String::as_str),
        Some(v2_clips[0].id.as_str())
    );
}

#[test]
fn video_import_renders_visible_animated_frames_on_gpu() {
    let ffmpeg_bin = zc_render::video::resolve_ffmpeg_binary_path();
    if !ProcessCommand::new(&ffmpeg_bin)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
    {
        return;
    }
    let require_gpu = std::env::var("ZOIDCRAFT_REQUIRE_GPU").as_deref() == Ok("1");
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(e) => e,
        Err(RenderError::NoAdapter(_)) if !require_gpu => return,
        Err(err) => panic!("failed to init headless engine: {err}"),
    };

    // Generate a 1.5s MP4 that transitions from bright red (t=0) to bright green (t=0.8s).
    let tmp_mp4 = std::env::temp_dir().join(format!(
        "zoidcraft_e2e_video_render_{}.mp4",
        std::process::id()
    ));
    let mut encoded = false;
    for codec in ["libx264", "mpeg4"] {
        let mut cmd = ProcessCommand::new(&ffmpeg_bin);
        cmd.args([
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x180:r=30:d=0.6",
            "-f",
            "lavfi",
            "-i",
            "color=c=lime:s=320x180:r=30:d=0.9",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
        ]);
        if codec == "libx264" {
            cmd.args(["-preset", "ultrafast"]);
        }
        cmd.arg(&tmp_mp4)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if cmd.status().is_ok_and(|s| s.success()) && tmp_mp4.is_file() {
            encoded = true;
            break;
        }
    }
    if !encoded {
        let _ = fs::remove_file(&tmp_mp4);
        return;
    }

    let mut app = make_headless_app_for_test();
    app.session.playback.current_frame = 0.0;
    let msg = app
        .import_media_from_path(&tmp_mp4)
        .expect("import mp4 video");
    assert!(
        msg.contains("320x180"),
        "expected probed dims in toast: {msg}"
    );
    assert_eq!(app.session.project.sequence.video_tracks.len(), 2);

    let f0 = engine
        .render_frame_rgba8(&app.session.project, &app.session.plugins, 0.0, 160, 90)
        .expect("render imported video frame 0");
    let f30 = engine
        .render_frame_rgba8(&app.session.project, &app.session.plugins, 30.0, 160, 90)
        .expect("render imported video frame 30");
    let _ = fs::remove_file(&tmp_mp4);

    let center_idx = ((45 * 160) + 80) * 4;
    assert!(
        f0[center_idx] > 180 && f0[center_idx + 1] < 80 && f0[center_idx + 3] > 240,
        "frame 0 must render bright red video pixels over V1 scene, got RGBA={:?}",
        &f0[center_idx..center_idx + 4]
    );
    assert!(
        f30[center_idx + 1] > 180 && f30[center_idx] < 80 && f30[center_idx + 3] > 240,
        "frame 30 must advance to bright green video pixels, got RGBA={:?}",
        &f30[center_idx..center_idx + 4]
    );
}

#[test]
fn enabled_required_plugins_persist_across_zcraft_save_and_reload_without_polluting_pz() {
    let mut app = make_headless_app_for_test();
    app.session
        .project
        .plugin_requirements
        .insert("camera-plus".to_owned());
    app.session
        .project
        .plugin_requirements
        .insert("light-plus".to_owned());
    app.session
        .project
        .plugin_requirements
        .insert("graph-plus".to_owned());
    app.cached_missing_plugins_revision = None;

    let missing = app.visible_missing_required_plugins();
    assert_eq!(missing.len(), 3);

    // 1. Enable required plugins (simulating "Enable Required Plugins" banner click).
    app.enable_required_plugins(&missing);
    assert!(app.visible_missing_required_plugins().is_empty());
    assert!(app.session.plugins.is_enabled("camera-plus"));
    assert!(app.session.plugins.is_enabled("light-plus"));
    assert!(app.session.plugins.is_enabled("graph-plus"));

    // 2. Save `.zcraft` via real GUI save helper.
    let tmp_dir = std::env::temp_dir().join(format!("zoidcraft_plugin_persist_{}", Uuid::new_v4()));
    fs::create_dir_all(&tmp_dir).expect("create temp dir");
    let zcraft_path = tmp_dir.join("persisted_plugins.zcraft");
    assert!(
        app.save_project_to_path(&zcraft_path, true),
        "saving .zcraft with enabled plugins must succeed"
    );

    // Vanilla `.pz` export preparation must NOT leak `zoidcraftEnabledPlugins`.
    let pz_proj = prepare_project_for_pz_export(&app.session);
    assert!(
        !pz_proj
            .sequence
            .preserved
            .contains_key(ZOIDCRAFT_ENABLED_PLUGINS_KEY),
        ".pz export must not be polluted with ZoidCraft-only persisted plugin state"
    );

    // 3. Reopen the saved `.zcraft` in a completely fresh ZoidCraftApp (vanilla startup state).
    let mut fresh_app = make_headless_app_for_test();
    assert!(!fresh_app.session.plugins.is_enabled("camera-plus"));
    assert!(!fresh_app.session.plugins.is_enabled("light-plus"));
    assert!(!fresh_app.session.plugins.is_enabled("graph-plus"));

    fresh_app.open_project_from_path(&zcraft_path);
    assert!(
        fresh_app.session.plugins.is_enabled("camera-plus"),
        "camera-plus must stay enabled after reload"
    );
    assert!(
        fresh_app.session.plugins.is_enabled("light-plus"),
        "light-plus must stay enabled after reload"
    );
    assert!(
        fresh_app.session.plugins.is_enabled("graph-plus"),
        "graph-plus must stay enabled after reload"
    );
    assert!(
        fresh_app.visible_missing_required_plugins().is_empty(),
        "reopening saved .zcraft must produce zero missing-plugin banner warnings"
    );
    assert!(
        fresh_app.workspace.panels.plugin_notice.is_none(),
        "plugin_notice must stay None when saved enabled plugins are restored"
    );

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn video_extensions_and_drop_classification_support_all_common_containers() {
    for ext in [
        "mp4", "mov", "mkv", "webm", "m4v", "avi", "MP4", "MOV", "MKV",
    ] {
        assert!(
            is_video_extension(ext),
            "expected '{ext}' to be recognized as a video extension"
        );
        let path = PathBuf::from(format!("clip.{ext}"));
        assert_eq!(
            classify_dropped_path(&path),
            DroppedFileKind::Media,
            "expected '{}' to classify as Media",
            path.display()
        );
        assert!(is_video_path(&path));
    }
    for non_video in ["png", "jpg", "wav", "mp3", "zcraft", "pz", "txt"] {
        assert!(
            !is_video_extension(non_video),
            "expected '{non_video}' not to be treated as video"
        );
    }
    let id1 = compute_external_asset_id("/tmp/clip.mov", 1_000_000_000, 42, b"header-a");
    let id2 = compute_external_asset_id("/tmp/clip.mov", 1_000_000_000, 42, b"header-a");
    let id3 = compute_external_asset_id("/tmp/clip.mov", 1_000_000_000, 43, b"header-a");
    assert_eq!(id1.len(), 64);
    assert_eq!(id1, id2);
    assert_ne!(id1, id3);
}

#[test]
fn with_live_viewport_project_is_zero_clone_and_restores_on_ok_and_err() {
    let mut project = Project::new_cm3_default();
    let root_seq_id = project.sequence.id;
    let asset_id =
        project.insert_binary_asset(AssetKind::Av, "big.mp4".to_owned(), vec![0x5a; 4096]);
    let original_bytes_ptr = project.assets[&asset_id]
        .bytes
        .as_ref()
        .expect("asset bytes")
        .as_ptr();

    project.sequence.controls.push(zc_project::SequenceControl {
        id: "ctl-root-gain".to_owned(),
        name: "RootGain".to_owned(),
        kind: zc_project::SequenceControlKind::Slider,
        track: zc_project::PropertyTrack::constant(zc_project::PropertyValue::Number(2.0)),
        preserved: BTreeMap::new(),
    });

    let mut child = Project::new_cm3_default().sequence;
    let child_id = Uuid::new_v4();
    let child_track_id = Uuid::new_v4();
    child.id = child_id;
    child.name = "ChildComp".to_owned();
    child.controls.clear();
    child.video_tracks[0].id = child_track_id;
    child.video_tracks.truncate(1);
    project.compositions.insert(child_id, child);

    let observed_ok = with_live_viewport_project(&mut project, Some(child_id), |vp| {
        assert_eq!(vp.sequence.id, child_id);
        assert_eq!(vp.sequence.video_tracks.len(), 1);
        assert_eq!(vp.sequence.video_tracks[0].id, child_track_id);
        assert!(
            vp.sequence
                .controls
                .iter()
                .any(|c| c.id == "ctl-root-gain" && c.name == "RootGain"),
            "child render must see root controls as fallbacks"
        );
        assert!(vp.compositions.contains_key(&root_seq_id));
        // Zero-clone guarantee: asset byte buffer address is identical!
        let inner_ptr = vp.assets[&asset_id]
            .bytes
            .as_ref()
            .expect("asset bytes")
            .as_ptr();
        assert_eq!(inner_ptr, original_bytes_ptr);
        Ok::<usize, &'static str>(42)
    });
    assert_eq!(observed_ok, Ok(42));
    assert_eq!(project.sequence.id, root_seq_id);
    assert!(project.compositions.contains_key(&child_id));
    assert!(
        project.compositions[&child_id].controls.is_empty(),
        "child controls must be restored without leaked root fallback controls"
    );

    // Must also restore cleanly on Err!
    let observed_err = with_live_viewport_project(&mut project, Some(child_id), |vp| {
        assert_eq!(vp.sequence.id, child_id);
        Err::<(), &'static str>("simulated render error")
    });
    assert_eq!(observed_err, Err("simulated render error"));
    assert_eq!(project.sequence.id, root_seq_id);
    assert!(project.compositions.contains_key(&child_id));
    assert!(project.compositions[&child_id].controls.is_empty());
    assert_eq!(
        project.assets[&asset_id]
            .bytes
            .as_ref()
            .expect("asset bytes")
            .as_ptr(),
        original_bytes_ptr
    );
}

#[test]
fn autosave_recovery_snapshot_externalizes_large_local_assets_without_mutating_session() {
    let mut session = Session::new();

    // 1. Large (> 8 MiB) asset WITH local URL -> externalized in recovery copy only.
    let large_bytes = vec![0xAB; AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES + 1024];
    let large_local_id =
        session
            .project
            .insert_binary_asset(AssetKind::Av, "large.mov".to_owned(), large_bytes);
    session
        .project
        .assets
        .get_mut(&large_local_id)
        .expect("large asset")
        .url = Some("/tmp/large.mov".to_owned());

    // 2. Small (<= 8 MiB) asset WITH local URL -> stays embedded in recovery copy.
    let small_local_id = session.project.insert_binary_asset(
        AssetKind::Image,
        "thumb.png".to_owned(),
        vec![1, 2, 3, 4],
    );
    session
        .project
        .assets
        .get_mut(&small_local_id)
        .expect("small asset")
        .url = Some("/tmp/thumb.png".to_owned());

    // 3. Large (> 8 MiB) asset WITHOUT local URL -> stays embedded (cannot externalize).
    let large_mem_id = session.project.insert_binary_asset(
        AssetKind::Av,
        "in_memory.mp4".to_owned(),
        vec![0xCD; AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES + 512],
    );

    let recovery = prepare_project_for_autosave(&session);

    let rec_large_local = recovery
        .assets
        .get(&large_local_id)
        .expect("recovery large");
    assert!(rec_large_local.external);
    assert!(rec_large_local.bytes.is_none());
    assert_eq!(rec_large_local.url.as_deref(), Some("/tmp/large.mov"));

    let rec_small_local = recovery
        .assets
        .get(&small_local_id)
        .expect("recovery small");
    assert!(!rec_small_local.external);
    assert_eq!(rec_small_local.bytes.as_deref(), Some(&[1, 2, 3, 4][..]));

    let rec_large_mem = recovery.assets.get(&large_mem_id).expect("recovery mem");
    assert!(!rec_large_mem.external);
    assert!(rec_large_mem.bytes.is_some());

    // Original session.project must remain completely untouched!
    let orig_large_local = session
        .project
        .assets
        .get(&large_local_id)
        .expect("orig large");
    assert!(!orig_large_local.external);
    assert_eq!(
        orig_large_local.bytes.as_ref().map(Vec::len),
        Some(AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES + 1024)
    );
}

#[test]
fn cached_project_render_hash_matches_fresh_hash_and_invalidates_on_revision() {
    let mut app = make_headless_app_for_test();
    let rev0 = app.session.revision();
    let fresh0 = zc_render::cache::project_render_hash(&app.session.project, &app.session.plugins);
    let cached0 = app.cached_project_render_hash();
    assert_eq!(cached0, fresh0);
    assert_eq!(app.cached_project_hash_rev, Some(rev0));

    // Second call at same revision returns identical cached hash.
    assert_eq!(app.cached_project_render_hash(), fresh0);

    // Mutating the project via session command advances revision and invalidates cached hash.
    app.session
        .execute(Command::SetResolution {
            width: 1280,
            height: 720,
        })
        .expect("set resolution");
    let rev1 = app.session.revision();
    assert_ne!(rev1, rev0);
    let fresh1 = zc_render::cache::project_render_hash(&app.session.project, &app.session.plugins);
    let cached1 = app.cached_project_render_hash();
    assert_eq!(cached1, fresh1);
    assert_ne!(cached1, cached0);
    assert_eq!(app.cached_project_hash_rev, Some(rev1));

    // Toggling a plugin also advances session.revision() and updates cached hash.
    let plugin_cmd = if app.session.plugins.is_enabled("camera-plus") {
        Command::DisablePlugin {
            plugin_id: "camera-plus".to_owned(),
        }
    } else {
        Command::EnablePlugin {
            plugin_id: "camera-plus".to_owned(),
        }
    };
    app.session.execute(plugin_cmd).expect("toggle camera-plus");
    let rev2 = app.session.revision();
    assert_ne!(rev2, rev1);
    let fresh2 = zc_render::cache::project_render_hash(&app.session.project, &app.session.plugins);
    let cached2 = app.cached_project_render_hash();
    assert_eq!(cached2, fresh2);
    assert_ne!(cached2, cached1);
    assert_eq!(app.cached_project_hash_rev, Some(rev2));
}
