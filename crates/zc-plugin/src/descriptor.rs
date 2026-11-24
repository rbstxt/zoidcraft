use serde::{Deserialize, Serialize};

/// UI categories shown in the Plugins panel (Spec 04 §5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PluginCategory {
    Utilities,
    Scene,
    Adjustment,
    Experimental,
}

impl PluginCategory {
    /// Display title for the Plugins panel section header.
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Utilities => "Utilities",
            Self::Scene => "Scene Extensions",
            Self::Adjustment => "Adjustment Extensions",
            Self::Experimental => "Experimental",
        }
    }

    /// Whether the category section starts collapsed by default in the Plugins UI.
    pub const fn collapsed_by_default(self) -> bool {
        matches!(self, Self::Experimental)
    }
}

/// Functional kind of a registered native or declarative plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PluginKind {
    ShaderPack,
    NativeFx,
    Object,
    MaterialPack,
    Extension,
    Core,
}

/// Metadata descriptor for a registered ZoidCraft native plugin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginDescriptor {
    /// Stable plugin identifier (matches Zoidium/CM3 dependency manifests for seamless interop).
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Author / maintainer attribution.
    pub author: String,
    /// Semantic version string.
    pub version: String,
    /// Plugins panel category.
    pub category: PluginCategory,
    /// Functional plugin classification.
    pub kind: PluginKind,
    /// Short one-line summary shown in the Plugins list.
    pub tagline: String,
    /// Detailed behavioral description.
    pub description: String,
    /// Enabled state on a fresh installation.
    pub default_enabled: bool,
    /// Always enabled and cannot be disabled (used by `core`).
    pub always_enabled: bool,
    /// Shown in the Plugins panel UI (`false` for hidden `core`).
    pub visible: bool,
    /// Optional warning banner displayed in the Plugins panel (e.g. Experimental plugins).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
    /// RGB accent badge color `[r, g, b]`.
    pub accent_color: [u8; 3],
    /// Declared capability/feature tags provided by this plugin.
    #[serde(default)]
    pub features: Vec<String>,
}

impl PluginDescriptor {
    /// Clean-room native namespace prefix (`zoidcraft:<id>`).
    pub fn native_namespace(&self) -> String {
        format!("zoidcraft:{}", self.id)
    }

    /// Compatibility namespace prefix (`zoidium:<id>`) for `.pz` interop.
    pub fn compat_namespace(&self) -> String {
        format!("zoidium:{}", self.id)
    }
}
