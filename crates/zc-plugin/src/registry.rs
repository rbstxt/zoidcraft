use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use thiserror::Error;

use crate::catalog::builtin_plugins;
use crate::descriptor::{PluginCategory, PluginDescriptor};
use crate::persistence::PluginSettingsState;

/// Errors returned by [`PluginRegistry`] operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PluginError {
    #[error("plugin '{id}' is not registered")]
    NotFound { id: String },

    #[error("plugin '{id}' is always enabled and cannot be disabled")]
    AlwaysEnabled { id: String },

    #[error("plugin '{id}' is currently in use by the active project and cannot be disabled")]
    InUse { id: String },

    #[error("plugin '{id}' is already registered")]
    AlreadyRegistered { id: String },
}

/// Maps every built-in plugin ID to a unique single-bit mask in `u64` for O(1) `is_enabled`.
const fn builtin_plugin_bit(id: &str) -> Option<u64> {
    let shift: u32 = match id.as_bytes() {
        b"core" => 0,
        b"easing-plus" => 1,
        b"expressions-plus" => 2,
        b"shader-plus" => 3,
        b"player-plus" => 4,
        b"asset-manager" => 5,
        b"precomp-plus" => 6,
        b"gif-support" => 7,
        b"graph-plus" => 8,
        b"motion-path" => 9,
        b"mp4-export" => 10,
        b"light-plus" => 11,
        b"geometry-plus" => 12,
        b"repeater" => 13,
        b"effector-plus" => 14,
        b"material-plus" => 15,
        b"text-plus" => 16,
        b"expresstext-plus" => 17,
        b"particles-plus" => 18,
        b"trapcode-suite" => 19,
        b"optical-flares" => 20,
        b"camera-plus" => 21,
        b"shape-plus" => 22,
        b"constraints" => 23,
        b"native-fx" => 24,
        b"openzoid-legacy" => 25,
        b"magic-looks" => 26,
        b"node-effect" => 27,
        b"smart-motion-blur-lite" => 28,
        b"sglow-hq-v2-preview" => 29,
        b"afterclip" => 30,
        b"alipfx-shader-pack-4" => 31,
        b"ccfx-shader-pack" => 32,
        b"layer-input" => 33,
        _ => return None,
    };
    Some(1u64 << shift)
}

/// Runtime registry managing built-in and user-installed plugins, live enable/disable
/// toggling, in-use project dependency locks, and Vanilla CM3 mode detection.
#[derive(Debug, Clone)]
pub struct PluginRegistry {
    order: Vec<String>,
    descriptors: BTreeMap<String, PluginDescriptor>,
    enabled: BTreeSet<String>,
    enabled_builtins: u64,
    optional_enabled_count: usize,
    configs: BTreeMap<String, Value>,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    /// Creates a registry pre-populated with all built-in ZoidCraft plugins from Spec 04 §5,
    /// enabling plugins according to `default_enabled || always_enabled`.
    pub fn new() -> Self {
        let mut reg = Self::empty();
        for desc in builtin_plugins() {
            let _ = reg.register(desc);
        }
        reg
    }

    /// Creates an empty registry with no plugins registered.
    pub fn empty() -> Self {
        Self {
            order: Vec::new(),
            descriptors: BTreeMap::new(),
            enabled: BTreeSet::new(),
            enabled_builtins: 0,
            optional_enabled_count: 0,
            configs: BTreeMap::new(),
        }
    }

    fn insert_enabled(&mut self, id: String, always_enabled: bool) {
        let bit = builtin_plugin_bit(&id);
        if self.enabled.insert(id) {
            if let Some(mask) = bit {
                self.enabled_builtins |= mask;
            }
            if !always_enabled {
                self.optional_enabled_count += 1;
            }
        }
    }

    fn remove_enabled(&mut self, id: &str, always_enabled: bool) {
        if self.enabled.remove(id) {
            if let Some(mask) = builtin_plugin_bit(id) {
                self.enabled_builtins &= !mask;
            }
            if !always_enabled {
                self.optional_enabled_count = self.optional_enabled_count.saturating_sub(1);
            }
        }
    }

    /// Registers a new plugin descriptor (e.g., a user-installed declarative ShaderPack).
    pub fn register(&mut self, descriptor: PluginDescriptor) -> Result<(), PluginError> {
        if self.descriptors.contains_key(&descriptor.id) {
            return Err(PluginError::AlreadyRegistered { id: descriptor.id });
        }
        let id = descriptor.id.clone();
        let should_enable = descriptor.default_enabled || descriptor.always_enabled;
        let always_enabled = descriptor.always_enabled;
        self.order.push(id.clone());
        self.descriptors.insert(id.clone(), descriptor);
        if should_enable {
            self.insert_enabled(id, always_enabled);
        }
        Ok(())
    }

    /// Returns total number of registered plugins (including hidden `core`).
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Returns `true` if no plugins are registered.
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// Looks up a registered plugin descriptor by ID.
    pub fn get(&self, id: &str) -> Option<&PluginDescriptor> {
        self.descriptors.get(id)
    }

    /// Iterates over all registered plugins in registration order.
    pub fn all(&self) -> impl Iterator<Item = &PluginDescriptor> {
        self.order.iter().filter_map(|id| self.descriptors.get(id))
    }

    /// Returns all visible plugins belonging to `category` in registration order.
    pub fn visible_in_category(&self, category: PluginCategory) -> Vec<&PluginDescriptor> {
        self.all()
            .filter(|d| d.visible && d.category == category)
            .collect()
    }

    /// Returns `true` if the plugin is registered and currently enabled.
    #[inline]
    pub fn is_enabled(&self, id: &str) -> bool {
        if let Some(mask) = builtin_plugin_bit(id) {
            return (self.enabled_builtins & mask) != 0;
        }
        self.enabled.contains(id)
    }

    /// Returns the sorted set of all currently enabled plugin IDs.
    pub fn enabled_ids(&self) -> &BTreeSet<String> {
        &self.enabled
    }

    /// Enables a registered plugin live without restart.
    pub fn enable(&mut self, id: &str) -> Result<(), PluginError> {
        let Some(desc) = self.descriptors.get(id) else {
            return Err(PluginError::NotFound { id: id.to_owned() });
        };
        let always_enabled = desc.always_enabled;
        self.insert_enabled(id.to_owned(), always_enabled);
        Ok(())
    }

    /// Disables a registered plugin live, enforcing:
    /// 1. `always_enabled` plugins (`core`) cannot be disabled (`PluginError::AlwaysEnabled`),
    /// 2. Plugins referenced by `active_project_deps` cannot be disabled (`PluginError::InUse`).
    pub fn disable(
        &mut self,
        id: &str,
        active_project_deps: &BTreeSet<String>,
    ) -> Result<(), PluginError> {
        let desc = self
            .descriptors
            .get(id)
            .ok_or_else(|| PluginError::NotFound { id: id.to_owned() })?;
        if desc.always_enabled {
            return Err(PluginError::AlwaysEnabled { id: id.to_owned() });
        }
        if active_project_deps.contains(id) {
            return Err(PluginError::InUse { id: id.to_owned() });
        }
        self.remove_enabled(id, false);
        Ok(())
    }

    /// Disables every optional plugin (leaving only `always_enabled` plugins such as `core`),
    /// putting ZoidCraft into pure Vanilla CM3 mode.
    pub fn disable_all_optional(&mut self) {
        let mut next_builtins = 0u64;
        self.enabled.retain(|id| {
            let keep = self
                .descriptors
                .get(id)
                .is_some_and(|desc| desc.always_enabled);
            if keep && let Some(mask) = builtin_plugin_bit(id) {
                next_builtins |= mask;
            }
            keep
        });
        self.enabled_builtins = next_builtins;
        self.optional_enabled_count = 0;
    }

    /// Disables every optional plugin only if none of them are referenced by `active_project_deps`.
    pub fn disable_all_optional_checked(
        &mut self,
        active_project_deps: &BTreeSet<String>,
    ) -> Result<(), PluginError> {
        for id in &self.enabled {
            if let Some(desc) = self.descriptors.get(id)
                && !desc.always_enabled
                && active_project_deps.contains(id)
            {
                return Err(PluginError::InUse { id: id.clone() });
            }
        }
        self.disable_all_optional();
        Ok(())
    }

    /// Restores all plugins to their `default_enabled || always_enabled` state.
    pub fn reset_to_defaults(&mut self) {
        self.enabled.clear();
        self.enabled_builtins = 0;
        self.optional_enabled_count = 0;
        let to_enable: Vec<(String, bool)> = self
            .descriptors
            .values()
            .filter(|desc| desc.default_enabled || desc.always_enabled)
            .map(|desc| (desc.id.clone(), desc.always_enabled))
            .collect();
        for (id, always_enabled) in to_enable {
            self.insert_enabled(id, always_enabled);
        }
    }

    /// Returns `true` when zero optional plugins are enabled (only `always_enabled` plugins
    /// such as hidden `core` remain active), guaranteeing 100% Vanilla CM3 behavior.
    #[inline]
    pub fn is_vanilla_cm3_mode(&self) -> bool {
        self.optional_enabled_count == 0
    }

    /// Lists plugin IDs required by a loaded project that are not currently enabled.
    pub fn missing_required_plugins(&self, project_deps: &BTreeSet<String>) -> Vec<String> {
        project_deps
            .iter()
            .filter(|id| !self.is_enabled(id))
            .cloned()
            .collect()
    }

    /// Stores a JSON configuration blob for a registered plugin.
    pub fn set_plugin_config(&mut self, id: &str, config: Value) -> Result<(), PluginError> {
        if !self.descriptors.contains_key(id) {
            return Err(PluginError::NotFound { id: id.to_owned() });
        }
        self.configs.insert(id.to_owned(), config);
        Ok(())
    }

    /// Reads the stored JSON configuration blob for a plugin, if any.
    pub fn plugin_config(&self, id: &str) -> Option<&Value> {
        self.configs.get(id)
    }

    /// Exports current enabled optional plugin IDs and per-plugin configs for JSON persistence.
    /// Hidden `always_enabled` plugins (`core`) are excluded from persisted metadata per Spec 04 §5.1.
    pub fn export_settings(&self) -> PluginSettingsState {
        let enabled_plugins = self
            .enabled
            .iter()
            .filter(|id| {
                self.descriptors
                    .get(*id)
                    .is_some_and(|desc| !desc.always_enabled)
            })
            .cloned()
            .collect();
        PluginSettingsState {
            schema_version: PluginSettingsState::CURRENT_SCHEMA_VERSION,
            enabled_plugins,
            plugin_configs: self.configs.clone(),
        }
    }

    /// Applies persisted [`PluginSettingsState`] while preserving `always_enabled` plugins
    /// and ignoring unknown plugin IDs gracefully.
    pub fn apply_settings(&mut self, settings: &PluginSettingsState) {
        self.enabled.clear();
        self.enabled_builtins = 0;
        self.optional_enabled_count = 0;
        let to_enable: Vec<(String, bool)> = self
            .descriptors
            .values()
            .filter(|desc| desc.always_enabled || settings.enabled_plugins.contains(&desc.id))
            .map(|desc| (desc.id.clone(), desc.always_enabled))
            .collect();
        for (id, always_enabled) in to_enable {
            self.insert_enabled(id, always_enabled);
        }
        self.configs = settings
            .plugin_configs
            .iter()
            .filter(|(id, _)| self.descriptors.contains_key(*id))
            .map(|(id, val)| (id.clone(), val.clone()))
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::descriptor::PluginKind;
    use crate::persistence::PreservedPluginData;

    #[test]
    fn builtin_catalog_completeness_and_categories() {
        let reg = PluginRegistry::new();

        let required_ids = [
            "core",
            "easing-plus",
            "expressions-plus",
            "shader-plus",
            "player-plus",
            "asset-manager",
            "precomp-plus",
            "gif-support",
            "graph-plus",
            "motion-path",
            "mp4-export",
            "light-plus",
            "geometry-plus",
            "repeater",
            "effector-plus",
            "material-plus",
            "text-plus",
            "expresstext-plus",
            "particles-plus",
            "trapcode-suite",
            "optical-flares",
            "camera-plus",
            "shape-plus",
            "constraints",
            "native-fx",
            "openzoid-legacy",
            "magic-looks",
            "node-effect",
            "layer-input",
            "smart-motion-blur-lite",
            "sglow-hq-v2-preview",
        ];

        for id in required_ids {
            let desc = reg
                .get(id)
                .unwrap_or_else(|| panic!("missing built-in plugin '{id}'"));
            assert_eq!(desc.id, id);
            assert_eq!(desc.native_namespace(), format!("zoidcraft:{id}"));
            assert_eq!(desc.compat_namespace(), format!("zoidium:{id}"));
        }

        // Core is hidden, always enabled, and Kind::Core.
        let core = reg.get("core").expect("core plugin must exist");
        assert!(core.always_enabled);
        assert!(!core.visible);
        assert_eq!(core.kind, PluginKind::Core);

        // Experimental category contains layer-input with warning banner.
        let experimental = reg.visible_in_category(PluginCategory::Experimental);
        assert_eq!(experimental.len(), 1);
        assert_eq!(experimental[0].id, "layer-input");
        assert!(experimental[0].warning.is_some());
        assert!(PluginCategory::Experimental.collapsed_by_default());

        // Visible utility list does not expose hidden core.
        let utilities = reg.visible_in_category(PluginCategory::Utilities);
        assert!(utilities.iter().all(|d| d.id != "core"));
        assert_eq!(utilities.len(), 10);
    }

    #[test]
    fn vanilla_cm3_mode_and_core_lock() {
        let mut reg = PluginRegistry::new();
        assert!(!reg.is_vanilla_cm3_mode());
        assert!(reg.is_enabled("core"));
        assert!(reg.is_enabled("expressions-plus"));

        let no_deps = BTreeSet::new();

        // Attempting to disable `core` must fail with AlwaysEnabled.
        assert_eq!(
            reg.disable("core", &no_deps),
            Err(PluginError::AlwaysEnabled {
                id: "core".to_owned(),
            })
        );
        assert!(reg.is_enabled("core"));

        // Disabling all optional plugins switches to Vanilla CM3 mode while keeping core active.
        reg.disable_all_optional();
        assert!(reg.is_vanilla_cm3_mode());
        assert!(reg.is_enabled("core"));
        assert!(!reg.is_enabled("expressions-plus"));
        assert!(!reg.is_enabled("shader-plus"));

        // Re-enabling any optional plugin exits Vanilla CM3 mode.
        reg.enable("graph-plus").expect("graph-plus should enable");
        assert!(!reg.is_vanilla_cm3_mode());
        assert!(reg.is_enabled("graph-plus"));

        // Disabling it returns to Vanilla CM3 mode.
        reg.disable("graph-plus", &no_deps)
            .expect("unused plugin should disable");
        assert!(reg.is_vanilla_cm3_mode());
    }

    #[test]
    fn in_use_lock_prevents_disabling_active_project_dependencies() {
        let mut reg = PluginRegistry::new();
        reg.enable("camera-plus").expect("enable camera-plus");

        let mut active_deps = BTreeSet::new();
        active_deps.insert("camera-plus".to_owned());
        active_deps.insert("expressions-plus".to_owned());

        assert_eq!(
            reg.disable("camera-plus", &active_deps),
            Err(PluginError::InUse {
                id: "camera-plus".to_owned(),
            })
        );
        assert!(reg.is_enabled("camera-plus"));

        // Checked bulk disable must also reject without mutating state.
        assert!(matches!(
            reg.disable_all_optional_checked(&active_deps),
            Err(PluginError::InUse { .. })
        ));
        assert!(reg.is_enabled("expressions-plus"));

        // Unreferenced plugin can still be disabled freely.
        reg.disable("player-plus", &active_deps)
            .expect("unreferenced plugin should disable");
        assert!(!reg.is_enabled("player-plus"));

        // Unknown plugin ID returns NotFound.
        assert_eq!(
            reg.disable("nonexistent-plugin", &active_deps),
            Err(PluginError::NotFound {
                id: "nonexistent-plugin".to_owned(),
            })
        );
    }

    #[test]
    fn settings_persistence_and_preserved_plugin_data_round_trip() {
        let mut reg = PluginRegistry::new();
        reg.disable_all_optional();
        reg.enable("easing-plus").expect("enable easing-plus");
        reg.enable("native-fx").expect("enable native-fx");
        reg.set_plugin_config("native-fx", json!({ "maxTaps": 64, "quality": "high" }))
            .expect("set config");

        let exported = reg.export_settings();
        // `core` must never be written to persisted enabled_plugins.
        assert!(!exported.enabled_plugins.contains("core"));
        assert!(exported.enabled_plugins.contains("easing-plus"));
        assert!(exported.enabled_plugins.contains("native-fx"));

        let json_str = exported.to_json_pretty().expect("serialize settings");
        let parsed = PluginSettingsState::from_json_str(&json_str).expect("deserialize settings");
        assert_eq!(exported, parsed);

        let mut restored = PluginRegistry::new();
        restored.apply_settings(&parsed);
        assert!(restored.is_enabled("core"));
        assert!(restored.is_enabled("easing-plus"));
        assert!(restored.is_enabled("native-fx"));
        assert!(!restored.is_enabled("shader-plus"));
        assert_eq!(
            restored.plugin_config("native-fx"),
            Some(&json!({ "maxTaps": 64, "quality": "high" }))
        );

        // PreservedPluginData lossless round-trip for disabled/unknown plugin payloads.
        let mut preserved = PreservedPluginData::new();
        preserved.insert(
            "zoidium:magic-looks",
            json!({
                "looksInstances": [
                    { "tool": "exposure", "amount": 0.42, "enabled": true },
                    { "tool": "halation", "radius": 18.5 }
                ],
                "unknownExtra": [1, "two", null, { "nested": true }]
            }),
        );
        preserved.insert(
            "thirdparty:custom-shader-pack",
            json!({ "version": 3, "rawBlob": "preserve-verbatim" }),
        );
        assert_eq!(preserved.len(), 2);
        assert!(!preserved.is_empty());

        let raw_json = serde_json::to_string(&preserved).expect("serialize preserved data");
        let round_tripped: PreservedPluginData =
            serde_json::from_str(&raw_json).expect("deserialize preserved data");
        assert_eq!(round_tripped, preserved);
        assert_eq!(
            round_tripped.get("thirdparty:custom-shader-pack"),
            Some(&json!({ "version": 3, "rawBlob": "preserve-verbatim" }))
        );
    }

    #[test]
    fn bitmask_and_optional_count_stay_in_sync_with_enabled_set() {
        let builtins = builtin_plugins();
        let mut seen_bits = 0u64;
        for desc in &builtins {
            let bit = builtin_plugin_bit(&desc.id)
                .unwrap_or_else(|| panic!("builtin plugin '{}' missing from bitmask", desc.id));
            assert_eq!(bit.count_ones(), 1);
            assert_eq!(
                seen_bits & bit,
                0,
                "duplicate bitmask entry for '{}'",
                desc.id
            );
            seen_bits |= bit;
        }

        let assert_invariants = |reg: &PluginRegistry| {
            for desc in reg.all() {
                assert_eq!(
                    reg.is_enabled(&desc.id),
                    reg.enabled_ids().contains(&desc.id),
                    "is_enabled mismatch for '{}'",
                    desc.id
                );
            }
            let expected_vanilla = reg
                .enabled_ids()
                .iter()
                .all(|id| reg.get(id).is_some_and(|desc| desc.always_enabled));
            assert_eq!(reg.is_vanilla_cm3_mode(), expected_vanilla);
        };

        let mut reg = PluginRegistry::new();
        assert_invariants(&reg);

        // Register a non-builtin user plugin enabled by default.
        let custom = PluginDescriptor {
            id: "custom-user-pack".to_owned(),
            name: "Custom User Pack".to_owned(),
            author: "User".to_owned(),
            version: "1.0.0".to_owned(),
            category: PluginCategory::Adjustment,
            kind: PluginKind::ShaderPack,
            tagline: "Custom".to_owned(),
            description: "Custom".to_owned(),
            default_enabled: true,
            always_enabled: false,
            visible: true,
            warning: None,
            accent_color: [0, 0, 0],
            features: Vec::new(),
        };
        reg.register(custom).expect("register custom");
        assert!(reg.is_enabled("custom-user-pack"));
        assert_invariants(&reg);

        // Idempotent enable & disable must not drift optional_enabled_count.
        let no_deps = BTreeSet::new();
        reg.enable("expressions-plus").unwrap();
        reg.enable("expressions-plus").unwrap();
        assert_invariants(&reg);
        reg.disable("expressions-plus", &no_deps).unwrap();
        reg.disable("expressions-plus", &no_deps).unwrap();
        assert_invariants(&reg);

        reg.disable_all_optional();
        assert!(reg.is_vanilla_cm3_mode());
        assert!(!reg.is_enabled("custom-user-pack"));
        assert_invariants(&reg);

        reg.enable("custom-user-pack").unwrap();
        assert!(!reg.is_vanilla_cm3_mode());
        assert_invariants(&reg);

        reg.disable("custom-user-pack", &no_deps).unwrap();
        assert!(reg.is_vanilla_cm3_mode());
        assert_invariants(&reg);

        reg.reset_to_defaults();
        assert!(!reg.is_vanilla_cm3_mode());
        assert_invariants(&reg);
    }
}
