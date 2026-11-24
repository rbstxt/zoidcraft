use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Serializable user settings storing enabled plugin IDs and per-plugin configuration blobs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginSettingsState {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub enabled_plugins: BTreeSet<String>,
    #[serde(default)]
    pub plugin_configs: BTreeMap<String, Value>,
}

const fn default_schema_version() -> u32 {
    PluginSettingsState::CURRENT_SCHEMA_VERSION
}

impl Default for PluginSettingsState {
    fn default() -> Self {
        Self {
            schema_version: Self::CURRENT_SCHEMA_VERSION,
            enabled_plugins: BTreeSet::new(),
            plugin_configs: BTreeMap::new(),
        }
    }
}

impl PluginSettingsState {
    pub const CURRENT_SCHEMA_VERSION: u32 = 1;

    /// Serializes settings to pretty-printed JSON.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parses settings from a JSON string.
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Lossless JSON payload map preserving unknown or disabled plugin data across
/// project load, edit, copy/paste, and re-save without mutating payloads (Spec 04 §5.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(transparent)]
pub struct PreservedPluginData {
    entries: BTreeMap<String, Value>,
}

impl PreservedPluginData {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn from_map(entries: BTreeMap<String, Value>) -> Self {
        Self { entries }
    }

    pub fn into_inner(self) -> BTreeMap<String, Value> {
        self.entries
    }

    pub fn as_map(&self) -> &BTreeMap<String, Value> {
        &self.entries
    }

    pub fn insert(&mut self, key: impl Into<String>, value: Value) -> Option<Value> {
        self.entries.insert(key.into(), value)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.get(key)
    }

    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.entries.remove(key)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.entries.iter()
    }

    /// Inserts entries from `other` whose keys are not already present in `self`.
    pub fn merge_missing(&mut self, other: &Self) {
        for (k, v) in &other.entries {
            self.entries.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
}
