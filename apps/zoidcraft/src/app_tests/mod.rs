//! Unit-test suites split by subsystem.

use super::*;

mod audio;
mod project_export;
mod viewport_ui;

fn make_headless_app_for_test() -> ZoidCraftApp {
    let session = Session::new();
    let fps = session.document().fps.max(1.0);
    let seq_len = session.document().sequence_length.max(1.0);
    let init_epoch = session.playback.epoch_id;
    let init_loop = session.playback.looping;
    ZoidCraftApp {
        session,
        theme: Cm3Theme::default(),
        applied_theme: None,
        workspace: Cm3Workspace::new(),
        timeline: TimelineViewState::new(),
        graph_editor: GraphEditorState::new(),
        viewport: ViewportState::new(),
        renderer: None,
        current_file: None,
        decoded_audio: HashMap::new(),
        asset_waveform_peaks: HashMap::new(),
        audio_clock: MasterPlaybackClock::new(fps, seq_len),
        audio_output: None,
        active_audio_window: None,
        audio_mix_generation: 0,
        pending_audio_mix: None,
        pending_audio_mix_target: None,
        cached_audio_mix_hash: None,
        audio_mix_hash: 0,
        audio_window_start: 0,
        audio_window_end: 0,
        audio_stream_attempted_hash: 0,
        audio_failure_latch: AudioOutputFailureLatch::default(),
        last_audio_epoch: init_epoch,
        last_audio_playing: false,
        last_audio_speed: 0.0,
        last_audio_frame: 0.0,
        last_audio_loop: init_loop,
        last_audio_fps: fps,
        last_audio_seq_len: seq_len,
        ffmpeg_probe_rx: None,
        pending_media_imports: Vec::new(),
        active_export: None,
        retired_exports: Vec::new(),
        next_export_job_id: 1,
        export_queue_snapshots: HashMap::new(),
        recovery_dir: std::env::temp_dir().join("zoidcraft-test-recovery"),
        last_autosave_at: Instant::now(),
        last_autosave_revision: 0,
        pending_autosave: None,
        startup_recovery_banner: None,
        clipboard_bridge_seeded: false,
        cached_project_hash_rev: None,
        cached_project_hash: 0,
        cached_scene_hash: None,
        last_audio_peaks_sync_rev: None,
        last_audio_peaks_asset_count: 0,
        cached_missing_plugins_revision: None,
        cached_missing_plugins: Vec::new(),
        dismissed_missing_plugins: BTreeSet::new(),
        pending_eyedropper: None,
        eyedropper_swallow_pointer: false,
        pending_unsaved_dialog: None,
        allow_close_now: false,
        last_canvas_rect: None,
        last_viewport_rect: None,
        force_single_render: true,
        last_tick: Instant::now(),
        last_title: String::new(),
        status_toast: None,
        last_viewport_render_error: None,
    }
}
