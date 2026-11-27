//! Timeline unit tests.

use super::*;

#[test]
pub(crate) fn frame_pixel_roundtrip_exact() {
    let fps = 30.0;
    let zoom = 0.3;
    let px = frame_to_px(90.0, fps, zoom);
    assert!((px - 360.0).abs() < 1e-9);
    let back = px_to_frame(px, fps, zoom);
    assert!((back - 90.0).abs() < 1e-9);
}

#[test]
pub(crate) fn ruler_interval_is_positive_and_monotonic() {
    let (step_a, px_a) = ruler_interval(0.3);
    let (step_b, px_b) = ruler_interval(1.0);
    assert!(step_a > 0.0 && px_a >= 4.0);
    assert!(step_b > 0.0 && px_b >= 4.0);
    assert!(step_b <= step_a);
}

#[test]
pub(crate) fn anchored_zoom_preserves_cursor_frame() {
    let fps = 60.0;
    let old_zoom = 0.5;
    let new_zoom = old_zoom * ZOOM_STEP_FACTOR;
    let old_scroll = 240.0;
    let anchor_x = 180.0;
    let frame_before = px_to_frame(old_scroll + anchor_x, fps, old_zoom);
    let new_scroll = zoom_anchored_scroll(old_scroll, anchor_x, old_zoom, new_zoom);
    let frame_after = px_to_frame(new_scroll + anchor_x, fps, new_zoom);
    assert!((frame_before - frame_after).abs() < 1e-9);
}

#[test]
pub(crate) fn non_overlapping_resolution_clamps_to_neighbor_edge() {
    let track = SessionTrack {
        id: "v1".to_string(),
        kind: TrackKind::Video,
        name: "V1".to_string(),
        enabled: true,
        solo: false,
        locked: false,
        is_3d: false,
        clips: vec![SessionClip {
            id: "c1".to_string(),
            kind: "scene".to_string(),
            name: "Clip 1".to_string(),
            start: 50.0,
            length: 40.0,
            time_start: 0.0,
            time_end: 0.0,
            time_animated: false,
            media_length: None,
            source_clip_id: None,
            properties: BTreeMap::new(),
            objects_3d: Vec::new(),
            effects: Vec::new(),
        }],
    };
    let exclude = BTreeSet::new();
    let resolved = resolve_non_overlapping_start(&track, 70.0, 20.0, &exclude);
    assert_eq!(resolved, Some(90.0));
}

#[test]
pub(crate) fn audio_meter_peak_decays_nonlinearly() {
    let p1 = decay_audio_meter_peak(0.8, 0.0);
    assert!(p1 < 0.8 && p1 > 0.79);
    let p2 = decay_audio_meter_peak(0.2, 0.5);
    assert!((p2 - 0.5).abs() < 1e-6);
}

#[test]
pub(crate) fn marker_hit_testing_and_snapping_include_marker_frames() {
    let mut session = Session::new();
    session
        .execute(Command::AddMarker {
            frame: 45.0,
            name: "Drop".to_string(),
            color: Some("#ed6a68".to_string()),
        })
        .expect("add marker");
    session
        .execute(Command::AddMarker {
            frame: 120.0,
            name: "Outro".to_string(),
            color: None,
        })
        .expect("add second marker");

    // Snap targets include marker frames so clips & keyframes snap cleanly to markers.
    let targets = collect_clip_snap_targets(&session, &BTreeSet::new());
    assert!(targets.iter().any(|&f| (f - 45.0).abs() < 1e-9));
    assert!(targets.iter().any(|&f| (f - 120.0).abs() < 1e-9));

    let fps = 30.0;
    let zoom = 0.3;
    let (snapped, hit_frame) = snap_frame(44.6, &targets, fps, zoom, true);
    assert_eq!(snapped, 45.0);
    assert_eq!(hit_frame, Some(45.0));

    // Marker hit testing on ruler.
    let ruler_rect = Rect::from_min_size(pos2(200.0, 0.0), vec2(800.0, RULER_HEIGHT));
    let m0_x = ruler_rect.left() + frame_to_px(45.0, fps, zoom) as f32;
    assert_eq!(
        hit_test_marker(
            &session.active_sequence().markers,
            ruler_rect,
            0.0,
            fps,
            zoom,
            pos2(m0_x + 2.0, ruler_rect.bottom() - 8.0),
        ),
        Some(0)
    );
    assert_eq!(
        hit_test_marker(
            &session.active_sequence().markers,
            ruler_rect,
            0.0,
            fps,
            zoom,
            pos2(m0_x + 40.0, ruler_rect.bottom() - 8.0),
        ),
        None
    );
    assert_eq!(
        parse_marker_hex_color("#ed6a68"),
        Some(Color32::from_rgb(0xed, 0x6a, 0x68))
    );
}

#[test]
pub(crate) fn active_composition_breadcrumb_reflects_nested_precomp() {
    let mut session = Session::new();
    assert_eq!(active_composition_breadcrumb(&session), None);

    let created = session
        .execute(Command::CreateComposition {
            name: "Intro Precomp".to_string(),
            resolution: [1920, 1080],
            fps: 30,
            length_frames: 150,
        })
        .expect("create comp");
    let comp_id = zc_project::Uuid::parse_str(
        created
            .get("comp_id")
            .and_then(Value::as_str)
            .expect("comp_id"),
    )
    .expect("uuid");

    session
        .execute(Command::SetActiveComposition {
            comp_id: Some(comp_id),
        })
        .expect("enter comp");
    let crumb = active_composition_breadcrumb(&session).expect("breadcrumb active");
    assert_eq!(crumb.comp_id, comp_id);
    assert_eq!(crumb.comp_name, "Intro Precomp");

    session
        .execute(Command::SetActiveComposition { comp_id: None })
        .expect("return to main");
    assert_eq!(active_composition_breadcrumb(&session), None);
}

#[test]
pub(crate) fn keyframe_dope_sheet_includes_animated_layer_mask_and_shape_plus_rows() {
    let mut session = Session::new();
    let v_track_id = session.document().video_tracks[0].id.clone();
    let added = session
        .execute(Command::AddClip {
            track_id: v_track_id,
            kind: "zoidium:shape-plus/layer".to_string(),
            name: Some("Vector Layer".to_string()),
            start: 10.0,
            length: 60.0,
        })
        .expect("add shape-plus clip");
    let clip_id_str = added
        .get("clip_id")
        .and_then(Value::as_str)
        .expect("clip_id")
        .to_string();
    let clip_uuid = zc_project::Uuid::parse_str(&clip_id_str).expect("uuid");
    let _ = session.selection.select_clip(&clip_id_str, None, false);

    // Attach an animated Shape+ payload with keyframed roundness.
    let payload = serde_json::json!({
        "schemaVersion": 1,
        "renderVersion": 1,
        "contents": [
            {
                "id": "s1",
                "kind": "rectangle",
                "properties": {
                    "name": "Rect 1",
                    "roundness": {
                        "animated": true,
                        "keyframes": [
                            { "frame": 0.0, "value": 0.0, "tween": 1 },
                            { "frame": 30.0, "value": 24.0, "tween": 1 }
                        ]
                    }
                }
            }
        ]
    });
    session
        .execute(Command::UpdateShapePlusPayload {
            clip_id: clip_uuid,
            payload,
        })
        .expect("update shape-plus");

    // Attach an animated LayerMask path.
    let mut path_track =
        zc_project::PropertyTrack::constant(zc_project::PropertyValue::Json(serde_json::json!([])));
    path_track.animated = true;
    path_track.keyframes = vec![
        zc_project::PropertyKeyframe::from_cm3_tween(
            0.0,
            zc_project::PropertyValue::Json(serde_json::json!([])),
            1,
            [[-10.0, 0.0], [10.0, 0.0]],
            true,
        ),
        zc_project::PropertyKeyframe::from_cm3_tween(
            20.0,
            zc_project::PropertyValue::Json(serde_json::json!([])),
            1,
            [[-10.0, 0.0], [10.0, 0.0]],
            true,
        ),
    ];
    let mask = zc_project::LayerMask {
        id: "mask-1".to_string(),
        name: "Reveal Mask".to_string(),
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
        path: path_track,
        preserved: BTreeMap::new(),
    };
    session
        .execute(Command::AddLayerMask {
            clip_id: clip_uuid,
            mask,
        })
        .expect("add mask");

    let rows = collect_keyframe_rows(&session);
    assert!(
        rows.iter()
            .any(|r| r.display_label == "Mask:Reveal Mask.path" && r.keyframes.len() == 2),
        "expected animated LayerMask row in dope sheet, got {:?}",
        rows.iter().map(|r| &r.display_label).collect::<Vec<_>>()
    );
    assert!(
        rows.iter()
            .any(|r| r.display_label == "Shape+:Rect 1.roundness" && r.keyframes.len() == 2),
        "expected animated Shape+ row in dope sheet, got {:?}",
        rows.iter().map(|r| &r.display_label).collect::<Vec<_>>()
    );
}

#[test]
pub(crate) fn split_audio_clip_waveform_reads_media_offset_not_asset_start() {
    // 1.0s asset at 100 peaks/sec: [0..0.5s) is silence (0), [0.5..1.0s) is loud tone (+/-120).
    let mut min_buf = vec![0_i8; 100];
    let mut max_buf = vec![0_i8; 100];
    for i in 50..100 {
        min_buf[i] = -120;
        max_buf[i] = 120;
    }
    let peaks = AudioWaveformPeaks {
        points_per_sec: 100.0,
        min_buf,
        max_buf,
    };
    let right_split_clip = SessionClip {
        id: "right".to_string(),
        kind: "audio".to_string(),
        name: "Split Right".to_string(),
        start: 15.0,
        length: 15.0,
        time_start: 0.5,
        time_end: 1.0,
        time_animated: false,
        media_length: Some(1.0),
        source_clip_id: None,
        properties: BTreeMap::new(),
        objects_3d: Vec::new(),
        effects: Vec::new(),
    };

    let (idx, count) =
        waveform_peak_slice_for_column(&right_split_clip, 30.0, peaks.points_per_sec, 0, 50);
    assert_eq!(
        idx, 50,
        "right split clip column 0 must index from time_start=0.5s"
    );
    assert!(count >= 1);
    assert_eq!(peaks.max_buf[idx], 120);
    assert_eq!(peaks.min_buf[idx], -120);

    // Reversed clip playing from 1.0s down to 0.5s must index descending within [0.5s, 1.0s],
    // never past the 1.0s asset end.
    let mut reverse_clip = right_split_clip;
    reverse_clip.time_start = 1.0;
    reverse_clip.time_end = 0.5;
    let (rev_first_idx, rev_first_count) =
        waveform_peak_slice_for_column(&reverse_clip, 30.0, peaks.points_per_sec, 0, 50);
    let (rev_last_idx, _) =
        waveform_peak_slice_for_column(&reverse_clip, 30.0, peaks.points_per_sec, 49, 50);
    assert_eq!(
        rev_first_idx, 99,
        "reverse clip column 0 must sample [0.99s, 1.0s]"
    );
    assert_eq!(rev_first_count, 1);
    assert_eq!(peaks.max_buf[rev_first_idx], 120);
    assert_eq!(
        rev_last_idx, 50,
        "reverse clip final column must sample [0.50s, 0.51s]"
    );
    assert_eq!(peaks.max_buf[rev_last_idx], 120);
}

#[test]
pub(crate) fn video_tracks_stack_bottom_up_and_audio_tracks_top_down() {
    let video_rect = Rect::from_min_max(pos2(587.0, 664.0), pos2(1232.0, 712.0));
    let (v0_y0, v0_y1) = track_row_y_bounds(TrackKind::Video, 0, 30.0, video_rect, 0.0);
    assert_eq!((v0_y0, v0_y1), (682.0, 712.0));
    assert_eq!(
        clip_y_bounds(TrackKind::Video, v0_y0, v0_y1),
        (683.0, 712.0)
    );
    assert_eq!(
        y_to_track_index(TrackKind::Video, 695.0, 30.0, video_rect, 0.0, 2),
        0
    );
    assert_eq!(
        y_to_track_index(TrackKind::Video, 670.0, 30.0, video_rect, 0.0, 2),
        1
    );

    let audio_rect = Rect::from_min_max(pos2(587.0, 716.0), pos2(1232.0, 799.0));
    let (a0_y0, a0_y1) = track_row_y_bounds(TrackKind::Audio, 0, 50.0, audio_rect, 0.0);
    assert_eq!((a0_y0, a0_y1), (716.0, 766.0));
    assert_eq!(
        clip_y_bounds(TrackKind::Audio, a0_y0, a0_y1),
        (716.0, 765.0)
    );
    assert_eq!(
        y_to_track_index(TrackKind::Audio, 740.0, 50.0, audio_rect, 0.0, 2),
        0
    );
    assert_eq!(
        y_to_track_index(TrackKind::Audio, 780.0, 50.0, audio_rect, 0.0, 2),
        1
    );
}

#[test]
pub(crate) fn default_video_audio_split_matches_zoidium_at_720p_and_800p() {
    // 1280x720 -> tracks_height = 107 px -> split at 38 px.
    assert_eq!(
        default_video_audio_split_y(
            107.0,
            DEFAULT_VIDEO_TRACK_HEIGHT,
            DEFAULT_AUDIO_TRACK_HEIGHT
        ),
        38.0
    );
    // 1280x800 -> tracks_height = 135 px -> split at 48 px (V1 682..712, divider 712..716, A1 716..766).
    let split_800 = default_video_audio_split_y(
        135.0,
        DEFAULT_VIDEO_TRACK_HEIGHT,
        DEFAULT_AUDIO_TRACK_HEIGHT,
    );
    assert_eq!(split_800, 48.0);
    let tracks_top = 664.0;
    let divider_top = tracks_top + split_800;
    assert_eq!(divider_top, 712.0);
    assert_eq!(divider_top + DIVIDER_HEIGHT, 716.0);
    assert!(!TimelineViewState::new().video_audio_split_custom);
}

#[test]
pub(crate) fn waveform_skips_silence_renders_symmetric_peaks_and_handles_freeze_and_time_curve() {
    // Silence yields None (no spurious 1 px bar).
    assert_eq!(compute_waveform_column_y(0, 0, 10.0, 40.0), None);
    // Symmetric peak spans symmetrically around center (10 + 20 = 30).
    let (y_top, y_bot) = compute_waveform_column_y(127, -128, 10.0, 40.0).expect("loud");
    assert!((y_top - 10.0).abs() < 1e-3);
    assert!((y_bot - 50.0).abs() < 1e-3);
    // Positive-only signal rises above center without extending below center (after min 1px).
    let (pt, pb) = compute_waveform_column_y(64, 0, 10.0, 40.0).expect("pos");
    assert!(pt < 30.0);
    assert!((pb - 30.0).abs() < 1e-3);

    // Freeze-frame clip (`time_start == time_end > 0`) samples the frozen instant for every column.
    let freeze_clip = SessionClip {
        id: "freeze".to_string(),
        kind: "audio".to_string(),
        name: "Freeze".to_string(),
        start: 0.0,
        length: 60.0,
        time_start: 0.75,
        time_end: 0.75,
        time_animated: false,
        media_length: Some(2.0),
        source_clip_id: None,
        properties: BTreeMap::new(),
        objects_3d: Vec::new(),
        effects: Vec::new(),
    };
    let (f_idx0, _) = waveform_peak_slice_for_column(&freeze_clip, 30.0, 100.0, 0, 50);
    let (f_idx_end, _) = waveform_peak_slice_for_column(&freeze_clip, 30.0, 100.0, 49, 50);
    assert_eq!(f_idx0, 75);
    assert_eq!(f_idx_end, 75);

    // Animated non-linear `time` curve samples interior keyframes instead of linear endpoints.
    let mut props = BTreeMap::new();
    props.insert(
        "time".to_string(),
        SessionProperty {
            animated: true,
            expression: None,
            default_value: serde_json::json!(0.0),
            keyframes: vec![
                zc_session::SessionKeyframe {
                    frame: 0.0,
                    value: serde_json::json!(0.0),
                    tween: 1,
                    control_points: [[-10.0, 0.0], [10.0, 0.0]],
                    continuous_tangent: true,
                    value_kind: None,
                },
                zc_session::SessionKeyframe {
                    frame: 30.0,
                    value: serde_json::json!(1.5),
                    tween: 1,
                    control_points: [[-10.0, 0.0], [10.0, 0.0]],
                    continuous_tangent: true,
                    value_kind: None,
                },
                zc_session::SessionKeyframe {
                    frame: 60.0,
                    value: serde_json::json!(0.5),
                    tween: 1,
                    control_points: [[-10.0, 0.0], [10.0, 0.0]],
                    continuous_tangent: true,
                    value_kind: None,
                },
            ],
        },
    );
    let anim_clip = SessionClip {
        id: "anim".to_string(),
        kind: "audio".to_string(),
        name: "Anim".to_string(),
        start: 0.0,
        length: 60.0,
        time_start: 0.0,
        time_end: 0.5,
        time_animated: true,
        media_length: Some(2.0),
        source_clip_id: None,
        properties: props,
        objects_3d: Vec::new(),
        effects: Vec::new(),
    };
    // Midpoint column (x = 25 of 50 -> local frame 30) should sample around 1.5 s (idx ~150).
    let (mid_idx, _) = waveform_peak_slice_for_column(&anim_clip, 30.0, 100.0, 25, 50);
    assert!(
        (145..=155).contains(&mid_idx),
        "expected animated time curve midpoint near 1.5s (idx 150), got {mid_idx}"
    );
}

#[test]
pub(crate) fn headless_timeline_pinch_cmd_scroll_trackpad_pan_and_occlusion() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    for _ in 0..3 {
        let _ = session.execute(Command::AddAudioTrack { name: None });
    }
    let mut state = TimelineViewState::new();
    state.scroll_x = 120.0;
    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(1000.0, 260.0));
    let lane_pos = pos2(480.0, 160.0);
    let video_header_pos = pos2(100.0, 120.0);

    let step = |session: &mut Session,
                state: &mut TimelineViewState,
                time: f64,
                modifiers: egui::Modifiers,
                mut events: Vec<egui::Event>,
                occlude_with_window: bool| {
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
                    let _ = show_timeline(ui, session, state, &palette);
                });
            if occlude_with_window {
                egui::Window::new("Floating Overlay")
                    .fixed_pos(pos2(300.0, 40.0))
                    .fixed_size(vec2(360.0, 200.0))
                    .show(ui.ctx(), |ui| {
                        ui.label("Overlay");
                        ui.allocate_space(vec2(360.0, 200.0));
                    });
            }
        });
        full.textures_delta.clear();
    };

    // Prime layer hit-testing.
    step(
        &mut session,
        &mut state,
        0.0,
        egui::Modifiers::NONE,
        vec![egui::Event::PointerMoved(lane_pos)],
        false,
    );

    // 1. Trackpad pinch zooms timeline anchored at cursor X.
    let z0 = state.zoom;
    let anchor_x = f64::from(lane_pos.x - HEADER_WIDTH);
    let frame_before = px_to_frame(state.scroll_x + anchor_x, session.document().fps, z0);
    step(
        &mut session,
        &mut state,
        0.016,
        egui::Modifiers::NONE,
        vec![egui::Event::PointerMoved(lane_pos), egui::Event::Zoom(1.25)],
        false,
    );
    assert!(state.zoom > z0 * 1.2);
    let frame_after = px_to_frame(
        state.scroll_x + anchor_x,
        session.document().fps,
        state.zoom,
    );
    assert!(
        (frame_before - frame_after).abs() < 1e-3,
        "pinch zoom must preserve frame under cursor"
    );

    // 2. Cmd+Scroll zooms timeline (regression test for egui 0.37 zeroing smooth_scroll_delta).
    let z1 = state.zoom;
    step(
        &mut session,
        &mut state,
        0.032,
        egui::Modifiers::COMMAND,
        vec![
            egui::Event::PointerMoved(lane_pos),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: vec2(0.0, 1.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Wheel,
                modifiers: egui::Modifiers::COMMAND,
            },
        ],
        false,
    );
    assert!(state.zoom > z1, "Cmd+Scroll must zoom timeline in");

    // 3. Cmd+Scroll over video track header adjusts track height smoothly.
    let h0 = state.video_track_height;
    let dynamic_video_header_pos = pos2(
        video_header_pos.x,
        RULER_HEIGHT + state.video_audio_split_y * 0.5,
    );
    step(
        &mut session,
        &mut state,
        0.050,
        egui::Modifiers::COMMAND,
        vec![
            egui::Event::PointerMoved(dynamic_video_header_pos),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: vec2(0.0, 1.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Wheel,
                modifiers: egui::Modifiers::COMMAND,
            },
        ],
        false,
    );
    assert!(
        state.video_track_height > h0,
        "Cmd+Scroll over header must grow track height"
    );

    // 4. Two-finger diagonal trackpad pan scrolls X and Y simultaneously without axis-fight.
    let scroll_x_before = state.scroll_x;
    let scroll_y_before = state.audio_scroll_y;
    let audio_pos = pos2(480.0, 210.0);
    step(
        &mut session,
        &mut state,
        0.070,
        egui::Modifiers::NONE,
        vec![
            egui::Event::PointerMoved(audio_pos),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: vec2(-16.0, -14.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Trackpad,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        false,
    );
    assert!(
        state.scroll_x > scroll_x_before,
        "diagonal trackpad pan must move scroll_x"
    );
    assert!(
        state.audio_scroll_y > scroll_y_before,
        "diagonal trackpad pan must move audio_scroll_y"
    );

    // 5. Occlusion safety: floating window above timeline blocks pinch and scroll completely.
    // Prime window layer across two frames (frame 1: egui sizing pass with interactable=false,
    // frame 2: first visible pass with interactable=true) so layer_id_at(lane_pos) sees it.
    step(
        &mut session,
        &mut state,
        0.10,
        egui::Modifiers::NONE,
        vec![egui::Event::PointerMoved(lane_pos)],
        true,
    );
    step(
        &mut session,
        &mut state,
        0.11,
        egui::Modifiers::NONE,
        vec![egui::Event::PointerMoved(lane_pos)],
        true,
    );
    let z_occ = state.zoom;
    let sx_occ = state.scroll_x;
    let vy_occ = state.video_scroll_y;
    step(
        &mut session,
        &mut state,
        0.12,
        egui::Modifiers::NONE,
        vec![
            egui::Event::PointerMoved(lane_pos),
            egui::Event::Zoom(1.4),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: vec2(-20.0, -20.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Trackpad,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        true,
    );
    assert!(
        (state.zoom - z_occ).abs() < 1e-9,
        "occluded timeline must ignore pinch"
    );
    assert!(
        (state.scroll_x - sx_occ).abs() < 1e-9,
        "occluded timeline must ignore scroll X"
    );
    assert!(
        (state.video_scroll_y - vy_occ).abs() < 1e-6,
        "occluded timeline must ignore scroll Y"
    );
}

#[test]
pub(crate) fn empty_keyframe_area_collapses_and_effect_selection_filters_keyframe_lanes() {
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session
        .selection
        .select_clip(clip_id.clone(), Some("scene"), false);

    // No keyframes on default scene -> keyframe rows empty.
    assert!(collect_keyframe_rows(&session).is_empty());

    // Add a 3D object keyframe AND an effect keyframe on the same clip.
    let obj_id = session
        .execute(Command::AddObject3D {
            clip_id: clip_id.clone(),
            kind: "Cube".to_owned(),
            name: Some("Box".to_owned()),
            parent_id: None,
        })
        .unwrap()["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let _ = session.execute(Command::SetKeyframe {
        target_id: obj_id.clone(),
        property: "position".to_owned(),
        frame: 0.0,
        value: serde_json::json!([0.0, 0.0, 0.0]),
        tween: None,
    });
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
        frame: 10.0,
        value: serde_json::json!(0.8),
        tween: None,
    });

    // Clip-only selection shows both 3D object & effect keyframes.
    let rows_all = collect_keyframe_rows(&session);
    assert!(rows_all.iter().any(|r| r.target_id == obj_id));
    assert!(rows_all.iter().any(|r| r.target_id == fx_id));

    // Selecting the Effect filters keyframe lanes exclusively to that Effect.
    let _ = session
        .selection
        .select_effect(fx_id.clone(), "bloom", false);
    let rows_fx = collect_keyframe_rows(&session);
    assert!(rows_fx.iter().all(|r| r.target_id == fx_id));
    assert_eq!(rows_fx.len(), 1);
}

#[test]
pub(crate) fn right_click_unselected_clip_and_empty_video_track_open_menus_immediately() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    // Start with clip UNSELECTED.
    session.selection.clear_clips();

    let mut state = TimelineViewState::new();
    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(900.0, 220.0));
    let run = |session: &mut Session,
               state: &mut TimelineViewState,
               time: f64,
               events: Vec<egui::Event>| {
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
                    let _ = show_timeline(ui, session, state, &palette);
                });
        });
        full.textures_delta.clear();
    };

    // Prime layout: kf_height=0 -> tracks_top=35, tracks_height=185, video_split_y=97 -> video_rect.bottom=132, V1 sits at y=102..132.
    run(&mut session, &mut state, 0.0, vec![]);
    let clip_pos = pos2(240.0, 118.0);
    // Right-click unselected clip on V1: must select clip AND keep Clip context menu open on the same click.
    run(
        &mut session,
        &mut state,
        0.02,
        vec![
            egui::Event::PointerMoved(clip_pos),
            egui::Event::PointerButton {
                pos: clip_pos,
                button: PointerButton::Secondary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    run(
        &mut session,
        &mut state,
        0.04,
        vec![
            egui::Event::PointerMoved(clip_pos),
            egui::Event::PointerButton {
                pos: clip_pos,
                button: PointerButton::Secondary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(
        session.selection.clip_ids.contains(&clip_id),
        "Right-clicking unselected clip must immediately select it"
    );
    assert!(
        matches!(state.context_menu, Some(TimelineContextMenu::Clip { .. })),
        "Right-clicking unselected clip must open Clip context menu on first right-click"
    );

    // Close menu, then right-click empty video space above V1 (y = 65.0).
    state.context_menu = None;
    let empty_v_pos = pos2(650.0, 65.0);
    run(
        &mut session,
        &mut state,
        0.06,
        vec![
            egui::Event::PointerMoved(empty_v_pos),
            egui::Event::PointerButton {
                pos: empty_v_pos,
                button: PointerButton::Secondary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    run(
        &mut session,
        &mut state,
        0.08,
        vec![
            egui::Event::PointerMoved(empty_v_pos),
            egui::Event::PointerButton {
                pos: empty_v_pos,
                button: PointerButton::Secondary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(
        matches!(state.context_menu, Some(TimelineContextMenu::Lane { .. })),
        "Right-clicking empty video track space must open Lane context menu"
    );
}

#[test]
pub(crate) fn playhead_jump_auto_scrolls_and_ruler_wheel_zooms() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let mut state = TimelineViewState::new();
    state.zoom = 1.5; // wide enough that end of sequence is off-screen
    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(700.0, 220.0));

    let run = |session: &mut Session,
               state: &mut TimelineViewState,
               time: f64,
               events: Vec<egui::Event>| {
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
                    let _ = show_timeline(ui, session, state, &palette);
                });
        });
        full.textures_delta.clear();
    };

    run(&mut session, &mut state, 0.0, vec![]);
    assert_eq!(state.scroll_x, 0.0);

    // Jump playhead near sequence end -> scroll_x must auto-scroll to keep playhead visible.
    let end_f = (session.document().sequence_length - 1.0).max(0.0);
    let _ = session.execute(Command::SetFrame { frame: end_f });
    run(&mut session, &mut state, 0.02, vec![]);
    assert!(
        state.scroll_x > 0.0,
        "Jumping playhead off-screen must auto-scroll scroll_x"
    );

    // Jump back to frame 0 -> scroll_x returns to 0.
    let _ = session.execute(Command::SetFrame { frame: 0.0 });
    run(&mut session, &mut state, 0.04, vec![]);
    assert!(
        state.scroll_x < 1.0,
        "Jumping playhead back to frame 0 must scroll back to start"
    );

    // Wheel over ruler zooms timeline.
    let z0 = state.zoom;
    let ruler_pos = pos2(380.0, 16.0);
    run(
        &mut session,
        &mut state,
        0.06,
        vec![
            egui::Event::PointerMoved(ruler_pos),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: vec2(0.0, 1.0),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Wheel,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert!(state.zoom > z0, "Mouse wheel over ruler must zoom timeline");
}

#[test]
pub(crate) fn video_vertical_scrollbar_is_draggable_without_grabbing_clips_and_stays_pinned_when_scrolled_x()
 {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    // Add enough video tracks to overflow the video lane viewport.
    for _ in 0..5 {
        session
            .execute(Command::AddVideoTrack { name: None })
            .expect("add video track");
    }
    // Extend the V1 clip across the entire right edge so a clip bar sits directly underneath
    // the scrollbar gutter at `lanes_rect.right()`.
    let v1_clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let v1_orig_start = session.document().video_tracks[0].clips[0].start;
    session
        .execute(Command::TrimClipRight {
            clip_id: v1_clip_id.clone(),
            delta_frames: 240.0,
            rate_stretch: false,
        })
        .expect("extend v1 clip");

    let mut state = TimelineViewState::new();
    // Also scroll horizontally (`scroll_x > 0`) to prove the vertical scrollbar remains pinned
    // to `video_lanes_rect.right()` regardless of `scroll_x`.
    state.scroll_x = 120.0;
    state.video_scroll_y = 0.0;
    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(760.0, 220.0));

    let run = |session: &mut Session,
               state: &mut TimelineViewState,
               time: f64,
               events: Vec<egui::Event>|
     -> egui::FullOutput {
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
                    let _ = show_timeline(ui, session, state, &palette);
                });
        });
        full.textures_delta.clear();
        full
    };

    // Frame 0 establishes layout & paints the scrollbar at `scroll_x = 120.0`.
    let out0 = run(&mut session, &mut state, 0.0, vec![]);
    assert!(
        state.scroll_x > 100.0,
        "horizontal scroll_x must remain > 0 during test"
    );
    // Verify scrollbar track/thumb geometry is pinned near `screen_rect.right()` (x = 760).
    let scrollbar_painted_at_right = out0.shapes.iter().any(|cs| match &cs.shape {
        egui::Shape::Rect(rs) => {
            rs.rect.right() >= screen_rect.right() - 3.5
                && rs.rect.right() <= screen_rect.right() - 1.0
                && rs.rect.width() >= 4.0
                && rs.rect.width() <= 8.0
                && rs.rect.top() >= RULER_HEIGHT
                && rs.rect.bottom() <= RULER_HEIGHT + state.video_audio_split_y + 1.0
        }
        _ => false,
    });
    assert!(
        scrollbar_painted_at_right,
        "Video vertical scrollbar must stay painted and pinned to lanes_rect.right() when scroll_x > 0"
    );

    // Press on the bottom of the vertical scrollbar (where V1 thumb sits at `video_scroll_y = 0`)
    // and drag UPWARD to scroll to higher tracks (V2..V6).
    let bar_x = screen_rect.right() - 5.0;
    let thumb_start_y = RULER_HEIGHT + state.video_audio_split_y - 12.0;
    let drag_target_y = RULER_HEIGHT + 12.0;
    let p_start = pos2(bar_x, thumb_start_y);
    let p_end = pos2(bar_x, drag_target_y);

    run(
        &mut session,
        &mut state,
        0.02,
        vec![egui::Event::PointerMoved(p_start)],
    );
    run(
        &mut session,
        &mut state,
        0.04,
        vec![
            egui::Event::PointerMoved(p_start),
            egui::Event::PointerButton {
                pos: p_start,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    run(
        &mut session,
        &mut state,
        0.06,
        vec![egui::Event::PointerMoved(p_end)],
    );
    run(
        &mut session,
        &mut state,
        0.08,
        vec![
            egui::Event::PointerMoved(p_end),
            egui::Event::PointerButton {
                pos: p_end,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );

    assert!(
        state.video_scroll_y > 20.0,
        "Dragging video vertical scrollbar upward must increase video_scroll_y, got {}",
        state.video_scroll_y
    );
    assert!(
        matches!(state.drag_state, TimelineDragState::None),
        "Scrollbar drag must never initiate a clip drag or marquee select"
    );
    assert!(
        (session.document().video_tracks[0].clips[0].start - v1_orig_start).abs() < 1e-6,
        "Dragging vertical scrollbar must NOT move the clip underneath"
    );

    // Now V1 is scrolled off-screen; clicking the pinned `+` button in `video_header_rect`
    // must STILL work and add a new video track!
    let tracks_before = session.document().video_tracks.len();
    let video_header_rect = Rect::from_min_max(
        pos2(0.0, RULER_HEIGHT),
        pos2(HEADER_WIDTH, RULER_HEIGHT + state.video_audio_split_y),
    );
    let add_pos = pinned_add_track_button_rect(
        TrackKind::Video,
        video_header_rect,
        session.plugins.is_enabled("camera-plus"),
    )
    .center();
    run(
        &mut session,
        &mut state,
        0.10,
        vec![egui::Event::PointerMoved(add_pos)],
    );
    run(
        &mut session,
        &mut state,
        0.12,
        vec![
            egui::Event::PointerMoved(add_pos),
            egui::Event::PointerButton {
                pos: add_pos,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    run(
        &mut session,
        &mut state,
        0.14,
        vec![
            egui::Event::PointerMoved(add_pos),
            egui::Event::PointerButton {
                pos: add_pos,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    assert_eq!(
        session.document().video_tracks.len(),
        tracks_before + 1,
        "Pinned `+` add-video-track button must remain clickable even when V1 is scrolled off-screen"
    );
}

#[test]
pub(crate) fn timeline_context_menu_paste_supports_external_pz_clips_and_effect_fallback() {
    let mut session = Session::new();
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let clip_uuid = zc_project::Uuid::parse_str(&clip_id).expect("clip uuid");
    let _ = session.selection.select_clip(clip_id.clone(), None, false);

    // 1. Export a clip as PanZoid CM3 JSON and paste it at frame 45 via external clipboard text.
    let clip_json = session
        .copy_clips_as_pz_json(&[clip_uuid])
        .expect("serialize clip as pz json");
    let state = TimelineViewState::new();
    assert!(paste_timeline_at_playhead_with_text(
        &mut session,
        &state,
        Some(&clip_json),
        45.0,
    ));
    let total_video_clips: usize = session
        .document()
        .video_tracks
        .iter()
        .map(|t| t.clips.len())
        .sum();
    assert_eq!(
        total_video_clips, 2,
        "Pasted PanZoid clip JSON must add a second clip"
    );

    // 2. Fallback to `hint: "any"` when external clipboard contains an effect JSON!
    let _ = session.selection.select_clip(clip_id.clone(), None, false);
    let fx_before = session.document().video_tracks[0].clips[0].effects.len();
    let bloom_json = r#"{"type":"bloom","properties":{"name":"Bloom","amount":0.75}}"#;
    assert!(paste_timeline_at_playhead_with_text(
        &mut session,
        &state,
        Some(bloom_json),
        0.0,
    ));
    assert_eq!(
        session.document().video_tracks[0].clips[0].effects.len(),
        fx_before + 1,
        "Effect JSON on system clipboard must fall back to hint='any' and paste onto selected clip"
    );
}

#[test]
pub(crate) fn vscrollbar_geometry_never_panics_across_extreme_heights() {
    let heights = [0.0_f32, 4.0, 12.0, 15.9, 16.0, 17.0, 18.0, 24.0, 100.0];
    for bottom_up in [false, true] {
        for h in heights {
            let rect = Rect::from_min_size(pos2(200.0, 50.0), vec2(400.0, h));
            let total_h = 300.0_f32;
            let max_scroll = (total_h - h).max(0.0);
            let geom = vscrollbar_geometry(rect, max_scroll * 0.5, max_scroll, total_h, bottom_up);
            if h < 16.0 {
                assert!(
                    geom.is_none(),
                    "expected None for height={h}, bottom_up={bottom_up}"
                );
            } else {
                let (hit_rect, track_rect, thumb_rect) = geom.unwrap_or_else(|| {
                    panic!("expected Some for height={h}, bottom_up={bottom_up}")
                });
                assert!(hit_rect.is_positive());
                assert!(track_rect.height() > 2.0);
                assert!(thumb_rect.height() > 0.0);
                assert!(thumb_rect.height() <= track_rect.height() + 1e-3);
                assert!(thumb_rect.top() >= track_rect.top() - 1e-3);
                assert!(thumb_rect.bottom() <= track_rect.bottom() + 1e-3);
            }
        }
    }
}

#[test]
pub(crate) fn video_audio_split_balances_track_counts_and_add_buttons_never_overlap_rows() {
    let tracks_height = 175.0_f32; // Typical 1280x752 timeline track area height.
    let avail = tracks_height - DIVIDER_HEIGHT;

    // 1. (1V, 2A): Video does not waste 60px above V1; Audio gets room for A1+A2.
    let split_1v_2a = compute_video_audio_split_y(
        tracks_height,
        1,
        2,
        DEFAULT_VIDEO_TRACK_HEIGHT,
        DEFAULT_AUDIO_TRACK_HEIGHT,
    );
    let audio_1v_2a = avail - split_1v_2a;
    assert!(
        split_1v_2a <= TRACK_PANE_BAR_HEIGHT + DEFAULT_VIDEO_TRACK_HEIGHT + 16.0,
        "1V/2A video pane must not leave huge empty gap above V1, got {split_1v_2a}"
    );
    assert!(
        audio_1v_2a >= TRACK_PANE_BAR_HEIGHT + 2.0 * DEFAULT_AUDIO_TRACK_HEIGHT - 1.0,
        "1V/2A audio pane must fit A1+A2 without clipping, got {audio_1v_2a}"
    );

    // 2. (4V, 1A): Video gets the lion's share while Audio still fits A1 cleanly.
    let split_4v_1a = compute_video_audio_split_y(
        tracks_height,
        4,
        1,
        DEFAULT_VIDEO_TRACK_HEIGHT,
        DEFAULT_AUDIO_TRACK_HEIGHT,
    );
    let audio_4v_1a = avail - split_4v_1a;
    assert!(
        split_4v_1a > audio_4v_1a,
        "4V/1A must allocate more height to Video ({split_4v_1a}) than Audio ({audio_4v_1a})"
    );
    assert!(
        audio_4v_1a >= TRACK_PANE_BAR_HEIGHT + DEFAULT_AUDIO_TRACK_HEIGHT,
        "4V/1A must still fit A1 completely, got {audio_4v_1a}"
    );

    // 3. (4V, 3A): Both overflow; split allocates proportionally without starving either side.
    let split_4v_3a = compute_video_audio_split_y(
        tracks_height,
        4,
        3,
        DEFAULT_VIDEO_TRACK_HEIGHT,
        DEFAULT_AUDIO_TRACK_HEIGHT,
    );
    let audio_4v_3a = avail - split_4v_3a;
    assert!(
        split_4v_3a >= TRACK_PANE_BAR_HEIGHT + DEFAULT_VIDEO_TRACK_HEIGHT
            && audio_4v_3a >= TRACK_PANE_BAR_HEIGHT + DEFAULT_AUDIO_TRACK_HEIGHT,
        "4V/3A must preserve minimums for both Video ({split_4v_3a}) and Audio ({audio_4v_3a})"
    );
    assert!(
        audio_4v_3a > split_4v_3a,
        "4V (120px) vs 3A (150px) must allocate proportionally more room to Audio ({audio_4v_3a}) than Video ({split_4v_3a})"
    );

    // 4. Verify `+` button rects sit strictly inside each pane's dedicated top strip and
    //    NEVER overlap ANY visible track row (with or without Camera+, keyframes, or breadcrumb).
    for top_y in [RULER_HEIGHT, RULER_HEIGHT + BREADCRUMB_BAR_HEIGHT + 64.0] {
        let v_header = Rect::from_min_size(pos2(0.0, top_y), vec2(HEADER_WIDTH, split_4v_3a));
        let v_lanes = Rect::from_min_size(pos2(HEADER_WIDTH, top_y), vec2(800.0, split_4v_3a));
        let (v_strip, v_rows_header, v_rows_lanes) = split_pane_header_strip(v_header, v_lanes);
        assert!(v_strip.is_some());

        let a_top = top_y + split_4v_3a + DIVIDER_HEIGHT;
        let a_header = Rect::from_min_size(pos2(0.0, a_top), vec2(HEADER_WIDTH, audio_4v_3a));
        let a_lanes = Rect::from_min_size(pos2(HEADER_WIDTH, a_top), vec2(800.0, audio_4v_3a));
        let (a_strip, a_rows_header, a_rows_lanes) = split_pane_header_strip(a_header, a_lanes);
        assert!(a_strip.is_some());

        for cam_plus in [false, true] {
            let v_add = pinned_add_track_button_rect(TrackKind::Video, v_header, cam_plus);
            let a_add = pinned_add_track_button_rect(TrackKind::Audio, a_header, cam_plus);
            assert!(
                !v_add.intersects(v_rows_header),
                "Video `+` button ({v_add:?}) must never intersect video track rows ({v_rows_header:?})"
            );
            assert!(
                !a_add.intersects(a_rows_header),
                "Audio `+` button ({a_add:?}) must never intersect audio track rows ({a_rows_header:?})"
            );
            for idx in 0..4 {
                let (y0, y1) = track_row_y_bounds(
                    TrackKind::Video,
                    idx,
                    DEFAULT_VIDEO_TRACK_HEIGHT,
                    v_rows_lanes,
                    0.0,
                );
                let row_rect = Rect::from_min_max(
                    pos2(v_rows_header.left(), y0),
                    pos2(v_rows_header.right(), y1),
                )
                .intersect(v_rows_header);
                if row_rect.is_positive() {
                    assert!(!v_add.intersects(row_rect));
                }
            }
            for idx in 0..3 {
                let (y0, y1) = track_row_y_bounds(
                    TrackKind::Audio,
                    idx,
                    DEFAULT_AUDIO_TRACK_HEIGHT,
                    a_rows_lanes,
                    0.0,
                );
                let row_rect = Rect::from_min_max(
                    pos2(a_rows_header.left(), y0),
                    pos2(a_rows_header.right(), y1),
                )
                .intersect(a_rows_header);
                if row_rect.is_positive() {
                    assert!(!a_add.intersects(row_rect));
                }
            }
        }
    }
}

#[test]
pub(crate) fn precomp_target_composition_id_extracts_from_precomp_clip_and_enters_comp() {
    let mut session = Session::new();
    let orig_clip = session.document().video_tracks[0].clips[0].clone();
    assert_eq!(
        precomp_target_composition_id(&session, &orig_clip),
        None,
        "Regular video clip must not report a precomp target composition ID"
    );

    let orig_uuid = zc_project::Uuid::parse_str(&orig_clip.id).expect("clip uuid");
    session
        .execute(Command::PrecomposeClips {
            clip_ids: vec![orig_uuid],
            name: "Nested FX".to_string(),
        })
        .expect("precompose clip");

    let precomp_clip = session.document().video_tracks[0].clips[0].clone();
    let target_comp_id = precomp_target_composition_id(&session, &precomp_clip)
        .expect("Precomp clip must expose target composition UUID");
    assert!(session.project.compositions.contains_key(&target_comp_id));

    session
        .execute(Command::SetActiveComposition {
            comp_id: Some(target_comp_id),
        })
        .expect("enter precomp composition");
    assert_eq!(
        session.document().active_composition_id,
        Some(target_comp_id)
    );
}

#[test]
pub(crate) fn vertical_clip_drag_onto_occupied_track_snaps_nearby_edge_or_rejects_overlap() {
    let track = SessionTrack {
        id: "v1".to_string(),
        kind: TrackKind::Video,
        name: "V1".to_string(),
        enabled: true,
        solo: false,
        locked: false,
        is_3d: false,
        clips: vec![SessionClip {
            id: "c1".to_string(),
            kind: "scene_3d".to_string(),
            name: "Scene 1".to_string(),
            start: 0.0,
            length: 300.0,
            time_start: 0.0,
            time_end: 0.0,
            time_animated: false,
            media_length: None,
            source_clip_id: None,
            properties: BTreeMap::new(),
            objects_3d: Vec::new(),
            effects: Vec::new(),
        }],
    };
    let exclude = BTreeSet::new();

    // 1. Open slot at [320, 380]: accepted immediately.
    assert_eq!(
        resolve_non_overlapping_start_within_range(&track, 320.0, 60.0, &exclude, Some(12.0)),
        Some(320.0)
    );

    // 2. Minor edge overlap at [292, 352] (8 frames into [0, 300]): snaps cleanly flush to 300.0.
    assert_eq!(
        resolve_non_overlapping_start_within_range(&track, 292.0, 60.0, &exclude, Some(12.0)),
        Some(300.0)
    );

    // 3. Broadly occupied drop ([0, 300] directly onto [0, 300]): rejected (`None`) instead of teleporting off-screen or overlapping.
    assert_eq!(
        resolve_non_overlapping_start_within_range(&track, 0.0, 300.0, &exclude, Some(12.0)),
        None
    );
    assert_eq!(
        resolve_non_overlapping_start_within_range(&track, 80.0, 60.0, &exclude, Some(12.0)),
        None
    );
}

#[test]
pub(crate) fn video_scroll_clamps_cleanly_across_window_resizes_with_three_video_tracks() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    session
        .execute(Command::AddVideoTrack { name: None })
        .expect("add V2");
    session
        .execute(Command::AddVideoTrack { name: None })
        .expect("add V3");
    assert_eq!(session.document().video_tracks.len(), 3);

    let mut state = TimelineViewState::new();
    let mut t = 0.0_f64;
    let mut run_at = |session: &mut Session, state: &mut TimelineViewState, w: f32, h: f32| {
        // Timeline panel receives ~35% of window height (after 40 px top chrome).
        let tl_h = ((h - 40.0) * 0.35).max(120.0);
        let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(w, tl_h));
        t += 0.02;
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(t),
            focused: true,
            ..Default::default()
        };
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    let _ = show_timeline(ui, session, state, &palette);
                });
        });
        full.textures_delta.clear();
        tl_h
    };

    // 1. Start compact (`800x540`), scroll up to V3 (or inject large stranded scroll).
    let tl_h_800 = run_at(&mut session, &mut state, 800.0, 540.0);
    state.video_scroll_y = 200.0;
    let _ = run_at(&mut session, &mut state, 800.0, 540.0);
    let strip_h_800 = if state.video_audio_split_y >= TRACK_PANE_BAR_HEIGHT + 18.0 {
        TRACK_PANE_BAR_HEIGHT
    } else {
        0.0
    };
    let max_v_800 =
        (3.0 * state.video_track_height - (state.video_audio_split_y - strip_h_800)).max(0.0);
    assert!(
        state.video_scroll_y <= max_v_800 + 1e-3 && state.video_scroll_y >= 0.0,
        "800x540 (tl_h={tl_h_800}) must clamp video_scroll_y ({}) within [0, {max_v_800}]",
        state.video_scroll_y
    );

    // 2. Resize taller (`1440x860`): all 3 video tracks fit -> video_scroll_y resets to 0.0!
    let _ = run_at(&mut session, &mut state, 1440.0, 860.0);
    assert!(
        state.video_scroll_y.abs() < 1e-4,
        "Resizing 800x540 -> 1440x860 must clear stranded video_scroll_y to 0.0, got {}",
        state.video_scroll_y
    );

    // 3. Resize to `900x600`, inject stranded scroll, then resize to `1280x752`:
    //    all 3 video tracks fit cleanly at 1280x752 with zero stranded scroll!
    let _ = run_at(&mut session, &mut state, 900.0, 600.0);
    state.video_scroll_y = 95.0;
    let _ = run_at(&mut session, &mut state, 1280.0, 752.0);
    assert!(
        state.video_scroll_y.abs() < 1e-4,
        "Resizing 900x600 -> 1280x752 with 3V/1A must clear video_scroll_y to 0.0, got {}",
        state.video_scroll_y
    );
}

#[test]
pub(crate) fn video_pane_prioritizes_three_video_tracks_over_single_empty_audio_and_shows_scrollbar_on_overflow()
 {
    // 1. Default 1280x752 timeline track area (~175 px) with (3V, 1A empty):
    //    V1..V3 (90 px + 20 px header = 110 px) fit cleanly without clipping.
    let tracks_h_default = 175.0_f32;
    let split_default = compute_video_audio_split_y_for_activity(
        tracks_h_default,
        3,
        1,
        DEFAULT_VIDEO_TRACK_HEIGHT,
        DEFAULT_AUDIO_TRACK_HEIGHT,
        false,
    );
    let avail_default = tracks_h_default - DIVIDER_HEIGHT;
    let audio_default = avail_default - split_default;
    assert!(
        split_default >= TRACK_PANE_BAR_HEIGHT + 3.0 * DEFAULT_VIDEO_TRACK_HEIGHT,
        "3V/1A(empty) at 175px must fit V1..V3 cleanly (>=110px), got video={split_default}, audio={audio_default}"
    );
    assert!(
        split_default > audio_default * 1.5,
        "Video pane ({split_default}) must clearly take priority over empty A1 ({audio_default})"
    );

    // 2. With 1 keyframe lane (~27 px) or Precomp breadcrumb (~24 px) reducing tracks_height to 148 px:
    //    Video still keeps >65% priority (empty A1 must NOT steal half the timeline height!),
    //    and any vertical overflow (> 0.5 px) produces valid visible scrollbar geometry.
    let tracks_h_reduced = 148.0_f32;
    let split_reduced = compute_video_audio_split_y_for_activity(
        tracks_h_reduced,
        3,
        1,
        DEFAULT_VIDEO_TRACK_HEIGHT,
        DEFAULT_AUDIO_TRACK_HEIGHT,
        false,
    );
    let avail_reduced = tracks_h_reduced - DIVIDER_HEIGHT;
    let audio_reduced = avail_reduced - split_reduced;
    assert!(
        split_reduced >= avail_reduced * 0.64,
        "3V/1A(empty) with keyframe/breadcrumb bar must give Video >=64% priority, got video={split_reduced} / {avail_reduced} (audio={audio_reduced})"
    );
    assert!(
        audio_reduced < avail_reduced * 0.40,
        "Empty A1 must stay compact (<40% of timeline), got {audio_reduced} / {avail_reduced}"
    );

    // 3. Even a tiny 2 px vertical overflow (`max_video_scroll > 0.5`) yields valid persistent
    //    scrollbar geometry with visible thumb travel gap.
    let total_v_h = 3.0 * DEFAULT_VIDEO_TRACK_HEIGHT; // 90 px
    let tight_lanes_rect = Rect::from_min_size(pos2(200.0, 55.0), vec2(800.0, total_v_h - 2.0));
    let max_scroll = total_v_h - tight_lanes_rect.height();
    assert!((max_scroll - 2.0).abs() < 1e-4);
    let (hit_rect, track_rect, thumb_rect) =
        vscrollbar_geometry(tight_lanes_rect, 0.0, max_scroll, total_v_h, true)
            .expect("2px overflow must still produce valid persistent vertical scrollbar geometry");
    assert!(hit_rect.is_positive());
    assert!(track_rect.height() > thumb_rect.height() + 4.0);
    assert!(
        (thumb_rect.bottom() - track_rect.bottom()).abs() < 1e-3,
        "bottom-up video scrollbar thumb must anchor at bottom when scroll_y == 0"
    );
}

#[test]
pub(crate) fn clip_label_elision_appends_ellipsis_cleanly_when_narrower_than_full_label() {
    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(400.0, 100.0))),
        ..Default::default()
    };
    let mut full = ctx.run_ui(raw, |ui| {
        let painter = ui.painter();
        let font = FontId::monospace(14.0);
        let color = Color32::WHITE;
        let label = "Adjustment Layer";

        let full_galley = painter.layout_no_wrap(label.to_owned(), font.clone(), color);
        let full_w = full_galley.size().x;

        // 1. Wide clip: full label fits intact without ellipsis.
        let wide = layout_ellipsized_clip_label(painter, label, font.clone(), color, full_w + 8.0)
            .expect("wide label galley");
        assert_eq!(wide.text(), label);

        // 2. Narrow trimmed clip (~60% width): truncates cleanly with trailing `…` inside max width.
        let narrow_w = full_w * 0.60;
        let narrow = layout_ellipsized_clip_label(painter, label, font.clone(), color, narrow_w)
            .expect("narrow ellipsized galley");
        assert!(
            narrow.text().ends_with('…'),
            "narrow clip label must end with ellipsis, got '{}'",
            narrow.text()
        );
        assert!(
            narrow.text().chars().count() < label.chars().count(),
            "narrow clip label must be shorter than full label, got '{}'",
            narrow.text()
        );
        assert!(
            narrow.size().x <= narrow_w + 0.5,
            "ellipsized label width ({}) must fit inside max_width ({narrow_w})",
            narrow.size().x
        );

        // 3. Extremely narrow clip (< 4 px): returns None rather than painting a clipped sliver.
        assert!(
            layout_ellipsized_clip_label(painter, label, font, color, 3.0).is_none(),
            "tiny clip width must suppress label cleanly"
        );
    });
    full.textures_delta.clear();
}

#[test]
pub(crate) fn clicking_already_selected_clip_surrenders_text_edit_focus_and_claims_timeline_focus()
{
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    let track_id = session.document().video_tracks[0].id.clone();
    session
        .execute(Command::AddClip {
            track_id,
            kind: "text".to_string(),
            name: Some("Title".to_string()),
            start: 0.0,
            length: 120.0,
        })
        .expect("add text clip");
    let clip_id = session.document().video_tracks[0].clips[0].id.clone();
    let _ = session.selection.select_clip(&clip_id, None, false);

    let mut state = TimelineViewState::new();
    state.focused = false;
    let mut font_search = String::from("Inter");
    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(900.0, 260.0));

    let run = |session: &mut Session,
               state: &mut TimelineViewState,
               font_search: &mut String,
               time: f64,
               focus_text_edit: bool,
               events: Vec<egui::Event>|
     -> (TimelineAction, Pos2) {
        let mut action = TimelineAction::default();
        let mut click_pos = pos2(0.0, 0.0);
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(time),
            focused: true,
            events,
            ..Default::default()
        };
        let mut full = ctx.run_ui(raw, |ui| {
            egui::Panel::top("mock_font_search")
                .exact_size(40.0)
                .show(ui, |ui| {
                    let resp = ui.text_edit_singleline(font_search);
                    if focus_text_edit {
                        resp.request_focus();
                    }
                });
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    let avail = ui.available_rect_before_wrap();
                    action = show_timeline(ui, session, state, &palette);
                    // Compute V1 clip midpoint for subsequent click.
                    let tracks_top = avail.top() + RULER_HEIGHT;
                    let video_bottom = tracks_top + state.video_audio_split_y;
                    click_pos = pos2(avail.left() + HEADER_WIDTH + 60.0, video_bottom - 15.0);
                });
        });
        full.textures_delta.clear();
        (action, click_pos)
    };

    // 1. Focus the mock searchable Font dropdown TextEdit while clip is already selected.
    let (_, clip_click_pos) = run(
        &mut session,
        &mut state,
        &mut font_search,
        0.0,
        true,
        vec![],
    );
    assert!(
        ctx.egui_wants_keyboard_input(),
        "setup: Font search TextEdit must hold keyboard focus before clicking timeline clip"
    );
    assert!(!state.focused);

    // 2. Click the ALREADY-SELECTED clip on the timeline (press + release).
    let (a_press, _) = run(
        &mut session,
        &mut state,
        &mut font_search,
        0.02,
        false,
        vec![
            egui::Event::PointerMoved(clip_click_pos),
            egui::Event::PointerButton {
                pos: clip_click_pos,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let (a_release, _) = run(
        &mut session,
        &mut state,
        &mut font_search,
        0.04,
        false,
        vec![
            egui::Event::PointerMoved(clip_click_pos),
            egui::Event::PointerButton {
                pos: clip_click_pos,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );

    assert!(
        a_press.clicked_timeline || a_release.clicked_timeline,
        "clicking an already-selected clip must report action.clicked_timeline = true"
    );
    assert!(state.focused, "timeline must regain focus on clip click");
    assert!(
        !ctx.egui_wants_keyboard_input(),
        "clicking an already-selected clip must surrender stale TextEdit focus so Cmd+C works immediately"
    );
    assert_eq!(session.selection.clip_ids, vec![clip_id]);
}

#[test]
pub(crate) fn expanding_keyframe_lanes_scrolls_selected_video_track_into_view() {
    let ctx = egui::Context::default();
    let palette = ThemePalette::default();
    let mut session = Session::new();
    session
        .execute(Command::AddVideoTrack { name: None })
        .expect("add V2");
    session
        .execute(Command::AddVideoTrack { name: None })
        .expect("add V3");
    let v3_id = session.document().video_tracks[2].id.clone();
    session
        .execute(Command::AddClip {
            track_id: v3_id,
            kind: "shape_layer".to_string(),
            name: Some("TopClip".to_string()),
            start: 0.0,
            length: 120.0,
        })
        .expect("add clip on V3");
    let clip_id = session.document().video_tracks[2].clips[0].id.clone();
    let _ = session.selection.select_clip(&clip_id, None, false);

    let mut state = TimelineViewState::new();
    let screen_rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 210.0));
    let mut t = 0.0;
    let mut run = |session: &mut Session, state: &mut TimelineViewState| {
        t += 0.02;
        let raw = egui::RawInput {
            screen_rect: Some(screen_rect),
            time: Some(t),
            focused: true,
            ..Default::default()
        };
        let mut full = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    let _ = show_timeline(ui, session, state, &palette);
                });
        });
        full.textures_delta.clear();
    };

    // 1. With no keyframes, all 3 video tracks fit -> video_scroll_y == 0.0.
    run(&mut session, &mut state);
    assert!(state.video_scroll_y.abs() < 1e-3);

    // 2. Animate 3 properties on the selected V3 clip so keyframe lanes expand (~64 px),
    //    shrinking the video pane height so 3 tracks overflow.
    for prop in ["position", "scale", "opacity"] {
        let val = if prop == "opacity" {
            serde_json::json!(1.0)
        } else {
            serde_json::json!([0.0, 0.0])
        };
        let _ = session.execute(Command::SetKeyframe {
            target_id: clip_id.clone(),
            property: prop.to_string(),
            frame: 0.0,
            value: val,
            tween: None,
        });
    }
    run(&mut session, &mut state);

    // Verify V3 (track index 2) is automatically scrolled into full view inside video_tracks_lanes_rect.
    let kf_rows = collect_keyframe_rows(&session);
    assert!(kf_rows.len() >= 3);
    let kf_h =
        ((kf_rows.len() as f32 * KEYFRAME_ROW_HEIGHT) + 4.0).min(KEYFRAME_ROW_HEIGHT * 4.0 + 4.0);
    let tracks_top = RULER_HEIGHT + kf_h + KEYFRAME_SPLITTER_HEIGHT;
    let v_rect = Rect::from_min_size(
        pos2(HEADER_WIDTH, tracks_top),
        vec2(
            screen_rect.width() - HEADER_WIDTH,
            state.video_audio_split_y,
        ),
    );
    let (_, _, v_lanes_rect) = split_pane_header_strip(
        Rect::from_min_size(
            pos2(0.0, tracks_top),
            vec2(HEADER_WIDTH, state.video_audio_split_y),
        ),
        v_rect,
    );
    let (v3_y0, v3_y1) = track_row_y_bounds(
        TrackKind::Video,
        2,
        state.video_track_height,
        v_lanes_rect,
        state.video_scroll_y,
    );
    assert!(
        state.video_scroll_y > 0.5,
        "expanding keyframe lanes must auto-scroll video_scroll_y > 0 to keep V3 visible, got {}",
        state.video_scroll_y
    );
    assert!(
        v3_y0 >= v_lanes_rect.top() - 0.5 && v3_y1 <= v_lanes_rect.bottom() + 0.5,
        "selected V3 track [{v3_y0}, {v3_y1}] must remain fully visible inside [{}, {}]",
        v_lanes_rect.top(),
        v_lanes_rect.bottom()
    );
}

#[test]
pub(crate) fn track_header_visible_layout_prevents_mid_glyph_slicing_and_keeps_scrollbar_on_overflow()
 {
    let clip_rect = Rect::from_min_size(pos2(0.0, 55.0), vec2(HEADER_WIDTH, 70.0));

    // 1. 12 px sliver peeking underneath `VIDEO +`: suppressed (`None`) so glyphs/icons are never sliced in half.
    let sliver_row = Rect::from_min_size(pos2(0.0, 37.0), vec2(HEADER_WIDTH, 30.0));
    assert_eq!(sliver_row.intersect(clip_rect).height(), 12.0);
    assert!(
        track_header_visible_layout(TrackKind::Video, sliver_row, clip_rect).is_none(),
        "12 px partially-scrolled sliver under VIDEO + must not paint sliced text or icons"
    );

    // 2. 18 px partial row (e.g. V2 at 800x540): centers text & icons strictly inside the visible band.
    let partial_row = Rect::from_min_size(pos2(0.0, 43.0), vec2(HEADER_WIDTH, 30.0));
    let vis_partial = partial_row.intersect(clip_rect);
    assert_eq!(vis_partial.height(), 18.0);
    let (cy, icon_side, font_px) =
        track_header_visible_layout(TrackKind::Video, partial_row, clip_rect)
            .expect("18 px visible row fits compact unclipped layout");
    assert!(
        cy - icon_side * 0.5 >= vis_partial.top() && cy + icon_side * 0.5 <= vis_partial.bottom(),
        "icons centered at {cy} (size {icon_side}) must stay strictly inside [{}, {}]",
        vis_partial.top(),
        vis_partial.bottom()
    );
    assert!(
        cy - font_px * 0.5 >= vis_partial.top() && cy + font_px * 0.5 <= vis_partial.bottom(),
        "label centered at {cy} (font {font_px}) must stay strictly inside [{}, {}]",
        vis_partial.top(),
        vis_partial.bottom()
    );

    // 3. Full 30 px row: preserves exact standard 18 px icons & 15 px font.
    let full_row = Rect::from_min_size(pos2(0.0, 95.0), vec2(HEADER_WIDTH, 30.0));
    let (_, full_icon, full_font) =
        track_header_visible_layout(TrackKind::Video, full_row, clip_rect)
            .expect("full row layout");
    assert_eq!(full_icon, 18.0);
    assert_eq!(full_font, 15.0);

    // 4. Overflowing 3-4 video tracks at 800x540 always yields valid persistent vertical scrollbar geometry.
    let lanes_800 = Rect::from_min_size(pos2(HEADER_WIDTH, 55.0), vec2(600.0, 48.0));
    for n_tracks in [2usize, 3, 4] {
        let total_h = n_tracks as f32 * DEFAULT_VIDEO_TRACK_HEIGHT;
        let max_scroll = total_h - lanes_800.height();
        assert!(max_scroll > 0.5);
        let geom = vscrollbar_geometry(lanes_800, 0.0, max_scroll, total_h, true);
        assert!(
            geom.is_some(),
            "{n_tracks} overflowing video tracks at 800x540 must always show persistent vertical scrollbar"
        );
    }
}

#[test]
pub(crate) fn keyframe_rows_cache_reuses_arc_until_revision_or_selection_changes() {
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
    let _ = session.selection.select_clip(&clip_id, None, false);

    let mut state = TimelineViewState::new();
    let first = state.keyframe_rows_cached(&session);
    let second = state.keyframe_rows_cached(&session);
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "keyframe rows cache must reuse the same Arc when revision and selection are unchanged"
    );

    session
        .execute(Command::TogglePropertyAnimated {
            target_id: clip_id.clone(),
            property: "opacity".to_owned(),
            frame: 0.0,
        })
        .unwrap();
    let after_rev = state.keyframe_rows_cached(&session);
    assert!(
        !std::sync::Arc::ptr_eq(&first, &after_rev),
        "keyframe rows cache must invalidate when session revision changes"
    );
    assert!(!after_rev.is_empty());

    session.selection.clear_clips();
    let after_sel = state.keyframe_rows_cached(&session);
    assert!(
        !std::sync::Arc::ptr_eq(&after_rev, &after_sel),
        "keyframe rows cache must invalidate when selection changes"
    );
}

#[test]
pub(crate) fn audio_waveform_batches_all_columns_into_single_mesh() {
    let ctx = egui::Context::default();
    let clip = ClipPaintInfo {
        id: "aud_clip_1".to_owned(),
        display_name: "Audio".to_owned(),
        start: 0.0,
        length: 60.0,
        time_start: 0.0,
        time_end: 2.0,
        time_animated: false,
        has_time_keyframes: false,
        is_precomp: false,
    };
    let peaks = AudioWaveformPeaks {
        points_per_sec: 100.0,
        max_buf: vec![96_i8; 256],
        min_buf: vec![-96_i8; 256],
    };
    let clip_rect = Rect::from_min_size(pos2(10.0, 10.0), vec2(102.0, 48.0));
    let raw = egui::RawInput {
        screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(300.0, 120.0))),
        ..Default::default()
    };
    let mut full = ctx.run_ui(raw, |ui| {
        paint_audio_clip_waveform(
            ui.painter(),
            clip_rect,
            clip_rect,
            &clip,
            None,
            30.0,
            Some(&peaks),
        );
    });
    let mut mesh_count = 0usize;
    let mut total_verts = 0usize;
    for clipped in &full.shapes {
        if let Shape::Mesh(mesh) = &clipped.shape {
            mesh_count += 1;
            total_verts += mesh.vertices.len();
        }
    }
    full.textures_delta.clear();
    assert_eq!(
        mesh_count, 1,
        "all 100 waveform columns must batch into a single egui::Mesh"
    );
    assert_eq!(total_verts, 100 * 4);
}

#[test]
pub(crate) fn marker_hit_test_leaves_top_scrub_band_free_and_hits_lower_pin_and_label() {
    let mut session = Session::new();
    session
        .execute(Command::AddMarker {
            frame: 45.0,
            name: "BeatDrop".to_string(),
            color: Some("#ed6a68".to_string()),
        })
        .expect("add marker");

    let fps = 30.0;
    let zoom = 0.3;
    let ruler_rect = Rect::from_min_size(pos2(200.0, 0.0), vec2(800.0, RULER_HEIGHT));
    let m0_x = (ruler_rect.left() + frame_to_px(45.0, fps, zoom) as f32).round();
    let markers = &session.active_sequence().markers;

    // Top ~57% scrubbing band (including directly above the marker diamond and label)
    // must NEVER hit a marker so playhead scrubbing is unobstructed.
    for y_offset in [2.0_f32, 8.0, 12.0, 16.0, 19.0] {
        assert_eq!(
            hit_test_marker(markers, ruler_rect, 0.0, fps, zoom, pos2(m0_x, y_offset)),
            None,
            "top scrubbing band y={y_offset} above marker pin must never hit marker"
        );
        assert_eq!(
            hit_test_marker(
                markers,
                ruler_rect,
                0.0,
                fps,
                zoom,
                pos2(m0_x + 14.0, y_offset)
            ),
            None,
            "top scrubbing band y={y_offset} above marker label must never hit marker"
        );
    }

    // Lower marker band hits the diamond pin and the visible label badge.
    let cy = marker_pin_center_y(ruler_rect);
    assert_eq!(
        hit_test_marker(markers, ruler_rect, 0.0, fps, zoom, pos2(m0_x, cy)),
        Some(0)
    );
    assert_eq!(
        hit_test_marker(markers, ruler_rect, 0.0, fps, zoom, pos2(m0_x + 14.0, cy)),
        Some(0)
    );
    // Far right of the label misses cleanly.
    assert_eq!(
        hit_test_marker(markers, ruler_rect, 0.0, fps, zoom, pos2(m0_x + 80.0, cy)),
        None
    );
}

#[test]
pub(crate) fn timeline_shift_click_range_and_ctrl_toggle_multi_selection() {
    let mut session = Session::new();
    let mut state = TimelineViewState::new();
    let track_id = session.document().video_tracks[0].id.clone();
    let c0 = session.document().video_tracks[0].clips[0].id.clone();

    let add = |session: &mut Session, name: &str, start: f64| -> String {
        session
            .execute(Command::AddClip {
                track_id: track_id.clone(),
                kind: "image".to_owned(),
                name: Some(name.to_owned()),
                start,
                length: 30.0,
            })
            .expect("add clip")["clip_id"]
            .as_str()
            .expect("clip_id")
            .to_owned()
    };
    let c1 = add(&mut session, "C1", 310.0);
    let c2 = add(&mut session, "C2", 350.0);
    let c3 = add(&mut session, "C3", 390.0);

    // Simulate initial single-type clip selection (e.g. from Media tab / default scene)
    // followed by Shift-click range selection across mixed clip kinds.
    let _ = session
        .selection
        .select_clip(c0.clone(), Some("scene"), false);
    apply_timeline_clip_click(&mut session, &mut state, &c2, true, false);
    assert_eq!(
        session.selection.clip_ids,
        vec![c0.clone(), c1.clone(), c2.clone()]
    );

    // Shift-clicking a further clip extends from the original anchor (`c0`).
    apply_timeline_clip_click(&mut session, &mut state, &c3, true, false);
    assert_eq!(
        session.selection.clip_ids,
        vec![c0.clone(), c1.clone(), c2.clone(), c3.clone()]
    );

    // Shift-clicking a closer clip shrinks back to `c0..=c1`.
    apply_timeline_clip_click(&mut session, &mut state, &c1, true, false);
    assert_eq!(session.selection.clip_ids, vec![c0.clone(), c1.clone()]);

    // Ctrl/Cmd-clicking toggles individual clips without clearing existing selection.
    apply_timeline_clip_click(&mut session, &mut state, &c3, false, true);
    assert_eq!(
        session.selection.clip_ids,
        vec![c0.clone(), c1.clone(), c3.clone()]
    );
    apply_timeline_clip_click(&mut session, &mut state, &c1, false, true);
    assert_eq!(session.selection.clip_ids, vec![c0.clone(), c3.clone()]);

    // Reverse Shift-click range from `c3` back to `c1`.
    apply_timeline_clip_click(&mut session, &mut state, &c3, false, false);
    apply_timeline_clip_click(&mut session, &mut state, &c1, true, false);
    assert_eq!(
        session.selection.clip_ids,
        vec![c1.clone(), c2.clone(), c3.clone()]
    );
}
