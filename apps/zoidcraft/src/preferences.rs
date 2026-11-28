use std::collections::{BTreeMap, BTreeSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::{env, fs};

use serde_json::{Value, json};
use zc_plugin::{PluginRegistry, PluginSettingsState};
use zc_project::ProjectError;
#[cfg(not(target_arch = "wasm32"))]
use zc_project::write_atomic;
use zc_session::{Command, Session};

use crate::ZoidCraftApp;
use crate::ui::layout::{Cm3Workspace, PreviewQuality};
use crate::ui::panels::AppSettings;
use crate::ui::theme::{Cm3Theme, EditorFontChoice, ThemeProfileChoice};

#[cfg(target_arch = "wasm32")]
const STORAGE_KEY: &str = "zoidcraft.preferences.v1";

#[cfg(target_arch = "wasm32")]
pub(crate) fn default_preferences_path() -> PathBuf {
    PathBuf::from(STORAGE_KEY)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn default_preferences_path() -> PathBuf {
    let env_path = |key| {
        env::var_os(key)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };

    #[cfg(target_os = "macos")]
    let config_dir =
        env_path("HOME").map(|home| home.join("Library/Application Support/ZoidCraft"));
    #[cfg(target_os = "windows")]
    let config_dir = env_path("APPDATA").map(|appdata| appdata.join("ZoidCraft"));
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let config_dir = env_path("XDG_CONFIG_HOME")
        .filter(|path| path.is_absolute())
        .or_else(|| env_path("HOME").map(|home| home.join(".config")))
        .map(|config| config.join("zoidcraft"));

    config_dir.map_or_else(
        || env::temp_dir().join("zoidcraft-preferences.json"),
        |dir| dir.join("preferences.json"),
    )
}

#[derive(Debug, PartialEq)]
pub(crate) struct UserPreferences {
    pub(crate) app_settings: AppSettings,
    pub(crate) enabled_plugins: Vec<String>,
    pub(crate) plugin_configs: BTreeMap<String, Value>,
    pub(crate) preview_quality: PreviewQuality,
    pub(crate) theme: Cm3Theme,
}

impl UserPreferences {
    pub(crate) fn load(path: &Path) -> Option<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        let mut json: Value = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
        #[cfg(target_arch = "wasm32")]
        let mut json: Value = serde_json::from_str(
            &web_sys::window()?
                .local_storage()
                .ok()??
                .get_item(path.to_str()?)
                .ok()??,
        )
        .ok()?;
        if !json.is_object() {
            return None;
        }

        let settings = &json["app_settings"];
        let defaults = AppSettings::default();
        let app_settings = AppSettings {
            default_fps: settings["default_fps"]
                .as_f64()
                .unwrap_or(defaults.default_fps)
                .clamp(1.0, 240.0),
            preview_cache_mib: serde_json::from_value(settings["preview_cache_mib"].clone())
                .unwrap_or(defaults.preview_cache_mib),
            undo_max_entries: serde_json::from_value(settings["undo_max_entries"].clone())
                .unwrap_or(defaults.undo_max_entries),
            undo_max_mib: serde_json::from_value(settings["undo_max_mib"].clone())
                .unwrap_or(defaults.undo_max_mib),
            show_fps_overlay: settings["show_fps_overlay"]
                .as_bool()
                .unwrap_or(defaults.show_fps_overlay),
            ffmpeg_path: settings["ffmpeg_path"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
            ffprobe_path: settings["ffprobe_path"]
                .as_str()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
            ffmpeg_setup_dismissed: settings["ffmpeg_setup_dismissed"]
                .as_bool()
                .unwrap_or(defaults.ffmpeg_setup_dismissed),
            check_updates_on_startup: settings["check_updates_on_startup"]
                .as_bool()
                .unwrap_or(defaults.check_updates_on_startup),
            skipped_update_version: settings.get("skipped_update_version").map_or(
                defaults.skipped_update_version,
                |value| {
                    value
                        .as_str()
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned)
                },
            ),
        };
        let preview_quality = PreviewQuality::ALL
            .into_iter()
            .find(|quality| json["preview_quality"].as_u64() == Some(u64::from(quality.divisor())))
            .unwrap_or_default();

        let saved_theme = &json["theme"];
        let defaults = Cm3Theme::default();
        let mut theme = Cm3Theme {
            profile: ThemeProfileChoice::ALL
                .into_iter()
                .find(|profile| saved_theme["profile"].as_str() == Some(profile.label()))
                .unwrap_or_default(),
            custom_hue_deg: serde_json::from_value::<f32>(saved_theme["custom_hue_deg"].clone())
                .unwrap_or(defaults.custom_hue_deg)
                .clamp(-180.0, 180.0),
            custom_sat_percent: serde_json::from_value::<f32>(
                saved_theme["custom_sat_percent"].clone(),
            )
            .unwrap_or(defaults.custom_sat_percent)
            .clamp(0.0, 200.0),
            custom_main_rgb: serde_json::from_value(saved_theme["custom_main_rgb"].clone())
                .unwrap_or(defaults.custom_main_rgb),
            font_scale: serde_json::from_value::<f32>(saved_theme["font_scale"].clone())
                .unwrap_or(defaults.font_scale)
                .clamp(0.75, 1.5),
            editor_font: EditorFontChoice::ALL
                .into_iter()
                .find(|font| saved_theme["editor_font"].as_str() == Some(font.label()))
                .unwrap_or_default(),
            ..defaults
        };
        theme.recompute_palette();

        Some(Self {
            app_settings,
            enabled_plugins: serde_json::from_value(json["enabled_plugins"].take()).unwrap_or_else(
                |_| {
                    PluginRegistry::new()
                        .enabled_ids()
                        .iter()
                        .cloned()
                        .collect()
                },
            ),
            plugin_configs: serde_json::from_value(json["plugin_configs"].take())
                .unwrap_or_default(),
            preview_quality,
            theme,
        })
    }

    pub(crate) fn restore_plugins(&self, plugins: &mut PluginRegistry) {
        plugins.apply_settings(&PluginSettingsState {
            enabled_plugins: self
                .enabled_plugins
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>(),
            plugin_configs: self.plugin_configs.clone(),
            ..PluginSettingsState::default()
        });
    }

    pub(crate) fn save(&self, path: &Path) -> Result<(), ProjectError> {
        let settings = &self.app_settings;
        let theme = &self.theme;
        let json = json!({
            "app_settings": {
                "default_fps": settings.default_fps,
                "preview_cache_mib": settings.preview_cache_mib,
                "undo_max_entries": settings.undo_max_entries,
                "undo_max_mib": settings.undo_max_mib,
                "show_fps_overlay": settings.show_fps_overlay,
                "ffmpeg_path": settings.ffmpeg_path,
                "ffprobe_path": settings.ffprobe_path,
                "ffmpeg_setup_dismissed": settings.ffmpeg_setup_dismissed,
                "check_updates_on_startup": settings.check_updates_on_startup,
                "skipped_update_version": settings.skipped_update_version,
            },
            "enabled_plugins": self.enabled_plugins,
            "plugin_configs": self.plugin_configs,
            "preview_quality": self.preview_quality.divisor(),
            "theme": {
                "profile": theme.profile.label(),
                "custom_hue_deg": theme.custom_hue_deg,
                "custom_sat_percent": theme.custom_sat_percent,
                "custom_main_rgb": theme.custom_main_rgb,
                "font_scale": theme.font_scale,
                "editor_font": theme.editor_font.label(),
            },
        });
        #[cfg(not(target_arch = "wasm32"))]
        {
            write_atomic(path, &serde_json::to_vec_pretty(&json)?)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let error = || std::io::Error::other("Browser local storage is unavailable");
            let storage = web_sys::window()
                .and_then(|window| window.local_storage().ok().flatten())
                .ok_or_else(error)?;
            storage
                .set_item(
                    path.to_str().ok_or_else(error)?,
                    &serde_json::to_string(&json)?,
                )
                .map_err(|_| error().into())
        }
    }
}

pub(crate) fn preferences_fingerprint(
    workspace: &Cm3Workspace,
    plugins: &PluginRegistry,
    theme: &Cm3Theme,
) -> u64 {
    let settings = &workspace.panels.app_settings;
    let mut hasher = DefaultHasher::new();
    (
        settings.default_fps.to_bits(),
        settings.preview_cache_mib,
        settings.undo_max_entries,
        settings.undo_max_mib,
        settings.show_fps_overlay,
        settings.ffmpeg_path.as_deref(),
        settings.ffprobe_path.as_deref(),
        settings.ffmpeg_setup_dismissed,
        settings.check_updates_on_startup,
        settings.skipped_update_version.as_deref(),
        workspace.player_plus.preview_quality.divisor(),
    )
        .hash(&mut hasher);
    (
        theme.profile,
        theme.custom_hue_deg.to_bits(),
        theme.custom_sat_percent.to_bits(),
        theme.custom_main_rgb.0.map(f32::to_bits),
        theme.font_scale.to_bits(),
        theme.editor_font,
    )
        .hash(&mut hasher);
    plugins.enabled_ids().hash(&mut hasher);
    for plugin in plugins.all() {
        if let Some(config) = plugins.plugin_config(&plugin.id) {
            (&plugin.id, config).hash(&mut hasher);
        }
    }
    hasher.finish()
}

pub(crate) fn apply_default_project_fps(session: &mut Session, fps: f64) {
    if session.document().fps != fps.round() && session.execute(Command::SetFps { fps }).is_ok() {
        // The initial FPS is part of the starter project, so it has no undo step.
        session.history.clear();
    }
    session.mark_saved();
}

impl ZoidCraftApp {
    pub(crate) fn save_user_preferences(&mut self, retry_failed_write: bool) {
        let Some(path) = &self.preferences_path else {
            return;
        };
        let plugins = &self.session.plugins;
        let fingerprint = preferences_fingerprint(&self.workspace, plugins, &self.theme);
        if let Some((previous, saved)) = self.last_preferences_write
            && previous == fingerprint
            && (saved || !retry_failed_write)
        {
            return;
        }

        let preferences = UserPreferences {
            app_settings: self.workspace.panels.app_settings.clone(),
            enabled_plugins: plugins.enabled_ids().iter().cloned().collect(),
            plugin_configs: plugins.export_settings().plugin_configs,
            preview_quality: self.workspace.player_plus.preview_quality,
            theme: self.theme.clone(),
        };
        let result = preferences.save(path);
        self.last_preferences_write = Some((fingerprint, result.is_ok()));
        if let Err(err) = result {
            tracing::warn!(path = %path.display(), "failed to save preferences: {err}");
        }
    }
}
