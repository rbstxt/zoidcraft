use super::*;
use crate::ui::layout::PreviewQuality;
use crate::ui::panels::AppSettings;
use crate::ui::theme::EditorFontChoice;
use zc_color::Rgb;

struct TestPreferencesDir(PathBuf);

impl TestPreferencesDir {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!("zoidcraft-preferences-test-{}", Uuid::new_v4())))
    }

    fn path(&self) -> PathBuf {
        self.0.join("preferences.json")
    }
}

impl Drop for TestPreferencesDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn preferences_round_trip_and_flush_pending_changes_on_drop() {
    let dir = TestPreferencesDir::new();
    let path = dir.path();
    let mut app = make_headless_app_for_test();
    app.preferences_path = Some(path.clone());
    app.workspace.panels.app_settings = AppSettings {
        default_fps: 60.0,
        show_fps_overlay: false,
        preview_cache_mib: 768,
        undo_max_entries: 123,
        undo_max_mib: 42,
        ffmpeg_path: Some("/opt/custom/bin/ffmpeg".to_owned()),
        ffprobe_path: Some("/opt/custom/bin/ffprobe".to_owned()),
        ffmpeg_setup_dismissed: true,
        check_updates_on_startup: false,
        skipped_update_version: Some("0.2.7".to_owned()),
    };
    app.workspace.player_plus.preview_quality = PreviewQuality::Quarter;
    app.theme.set_custom_hue_sat(40.0, 125.0);
    app.theme.set_custom_main(Rgb::from_u8(23, 128, 95));
    app.theme.font_scale = 1.25;
    app.theme.editor_font = EditorFontChoice::GeistMono;
    let plugins = &mut app.session.plugins;
    assert!(plugins.is_enabled("player-plus"));
    assert!(!plugins.is_enabled("easing-plus"));
    plugins.disable("player-plus", &BTreeSet::new()).unwrap();
    plugins.enable("easing-plus").unwrap();
    let disabled_config = json!({"previewScale": 0.25, "options": [true, null]});
    plugins
        .set_plugin_config("player-plus", disabled_config.clone())
        .unwrap();
    plugins
        .set_plugin_config("easing-plus", json!({"preset": "overshoot"}))
        .unwrap();
    let enabled_ids = plugins.enabled_ids().clone();
    let plugin_configs = plugins.export_settings().plugin_configs;

    app.save_user_preferences(false);
    let saved = UserPreferences::load(&path).expect("load saved preferences");
    assert_eq!(saved.app_settings, app.workspace.panels.app_settings);
    assert_eq!(saved.preview_quality, PreviewQuality::Quarter);
    assert_eq!(saved.theme, app.theme);
    assert_eq!(
        saved.enabled_plugins,
        enabled_ids.iter().cloned().collect::<Vec<_>>()
    );
    assert_eq!(saved.plugin_configs, plugin_configs);

    let mut restored_plugins = PluginRegistry::new();
    saved.restore_plugins(&mut restored_plugins);
    assert_eq!(restored_plugins.enabled_ids(), &enabled_ids);
    assert_eq!(
        restored_plugins.plugin_config("player-plus"),
        Some(&disabled_config)
    );
    assert_eq!(
        restored_plugins.export_settings().plugin_configs,
        plugin_configs
    );

    let edited_config = json!({"previewScale": 0.5});
    app.session
        .plugins
        .set_plugin_config("player-plus", edited_config.clone())
        .unwrap();
    app.save_user_preferences(false);
    assert_eq!(
        UserPreferences::load(&path).unwrap().plugin_configs["player-plus"],
        edited_config
    );

    app.workspace.player_plus.preview_quality = PreviewQuality::Half;
    drop(app);
    assert_eq!(
        UserPreferences::load(&path).unwrap().preview_quality,
        PreviewQuality::Half
    );
    assert_eq!(fs::read_dir(&dir.0).unwrap().count(), 1);
}

#[test]
fn preferences_ignore_corrupt_files_and_default_missing_fields() {
    let dir = TestPreferencesDir::new();
    let path = dir.path();
    assert!(UserPreferences::load(&path).is_none());
    write_atomic(&path, b"{").unwrap();
    assert!(UserPreferences::load(&path).is_none());
    write_atomic(
        &path,
        br#"{"app_settings":{"default_fps":360},"enabled_plugins":[]}"#,
    )
    .unwrap();

    let prefs = UserPreferences::load(&path).expect("load partial preferences");
    assert_eq!(
        prefs.app_settings,
        AppSettings {
            default_fps: 240.0,
            ..AppSettings::default()
        }
    );
    assert_eq!(prefs.preview_quality, PreviewQuality::Full);
    assert_eq!(prefs.theme, Cm3Theme::default());
    let mut plugins = PluginRegistry::new();
    prefs.restore_plugins(&mut plugins);
    assert!(plugins.is_vanilla_cm3_mode());
    assert!(plugins.is_enabled("core"));
}

#[test]
fn new_project_applies_default_fps_and_preserves_plugins_and_configs() {
    let mut app = make_headless_app_for_test();
    app.workspace.panels.app_settings.default_fps = 60.0;
    app.workspace.player_plus.preview_quality = PreviewQuality::Half;
    app.session.execute(Command::SetFps { fps: 24.0 }).unwrap();
    app.session.mark_saved();
    let plugins = &mut app.session.plugins;
    plugins.disable("player-plus", &BTreeSet::new()).unwrap();
    plugins.enable("easing-plus").unwrap();
    plugins
        .set_plugin_config("player-plus", json!({"previewScale": 0.5}))
        .unwrap();
    plugins
        .set_plugin_config("easing-plus", json!({"preset": "overshoot"}))
        .unwrap();
    let enabled_ids = plugins.enabled_ids().clone();
    let plugin_configs = plugins.export_settings().plugin_configs;

    app.handle_new_project();

    assert!(app.pending_unsaved_dialog.is_none());
    assert_eq!(app.session.document().fps, 60.0);
    assert_eq!(app.session.project.sequence.fps, 60);
    assert_eq!(app.workspace.panels.app_settings.default_fps, 60.0);
    assert_eq!(
        app.workspace.player_plus.preview_quality,
        PreviewQuality::Half
    );
    assert!(!app.session.is_dirty());
    assert_eq!(app.session.history.undo_len(), 0);
    assert_eq!(app.session.plugins.enabled_ids(), &enabled_ids);
    assert_eq!(
        app.session.plugins.export_settings().plugin_configs,
        plugin_configs
    );
}

#[test]
fn update_settings_round_trip_trim_and_affect_fingerprint() {
    let dir = TestPreferencesDir::new();
    let path = dir.path();
    let mut app = make_headless_app_for_test();
    app.preferences_path = Some(path.clone());

    let fp_initial = preferences_fingerprint(&app.workspace, &app.session.plugins, &app.theme);
    app.workspace.panels.app_settings.check_updates_on_startup = false;
    let fp_toggled = preferences_fingerprint(&app.workspace, &app.session.plugins, &app.theme);
    assert_ne!(fp_initial, fp_toggled);

    app.workspace.panels.app_settings.skipped_update_version = Some("  0.2.8  ".to_owned());
    let fp_skipped = preferences_fingerprint(&app.workspace, &app.session.plugins, &app.theme);
    assert_ne!(fp_toggled, fp_skipped);

    app.save_user_preferences(false);
    let loaded = UserPreferences::load(&path).expect("load saved update preferences");
    assert!(!loaded.app_settings.check_updates_on_startup);
    assert_eq!(
        loaded.app_settings.skipped_update_version.as_deref(),
        Some("0.2.8")
    );

    // Blank / whitespace-only skipped_update_version normalizes to None on load.
    app.workspace.panels.app_settings.skipped_update_version = Some("   ".to_owned());
    let fp_blank = preferences_fingerprint(&app.workspace, &app.session.plugins, &app.theme);
    assert_ne!(fp_skipped, fp_blank);
    app.save_user_preferences(false);
    let loaded_blank = UserPreferences::load(&path).expect("load blank skipped version");
    assert_eq!(loaded_blank.app_settings.skipped_update_version, None);
}
