#![forbid(unsafe_code)]

//! Clean-room native plugin registry, live toggling, project dependency locks,
//! and lossless unknown-plugin payload preservation for ZoidCraft.

pub mod catalog;
pub mod descriptor;
pub mod persistence;
pub mod registry;

pub use catalog::{EXPERIMENTAL_WARNING, builtin_plugins};
pub use descriptor::{PluginCategory, PluginDescriptor, PluginKind};
pub use persistence::{PluginSettingsState, PreservedPluginData};
pub use registry::{PluginError, PluginRegistry};
