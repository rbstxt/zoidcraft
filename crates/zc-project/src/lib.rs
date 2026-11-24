#![forbid(unsafe_code)]
#![recursion_limit = "512"]

//! Clean-room project data model, effect catalog, crash-safe atomic file I/O,
//! native `.zcraft` archive codec, and best-effort `.pz` (ClipMaker 3) importer
//! for ZoidCraft.

pub mod atomic;
pub mod effects;
pub mod model;
pub mod pz_import;
pub mod sha256;
pub mod zcraft;

use thiserror::Error;

pub use atomic::{write_atomic, write_atomic_with};
pub use effects::{
    EffectCategory, EffectDescriptor, STANDARD_EFFECT_IDS, ZOIDIUM_PLUGIN_EFFECT_IDS,
    create_custom_shader, create_effect_by_id, create_effect_group, create_standard_effect,
    create_zoidium_effect, find_effect_descriptor, standard_effect_catalog, zoidium_effect_catalog,
};
pub use model::{
    AssetKind, AssetRecord, AssetSource, AudioClip, AudioClipData, AudioTrack, CameraData,
    CameraDofParams, CameraKind, CameraShakeParams, CameraVibrateParams, CompatMode,
    CustomPbrMaterial, CustomProperty, DrawOp, EffectInstance, EffectKind, EulerOrder,
    FollowConstraint, GradientStop, ImagePlusMaterial, Layer, LayerKind, LightData, LightKind,
    Marker, MatcapMaterial, MaterialSpec, MediaItem, ModelData, MotionBlurSettings, Object3D,
    Object3DKind, ObjectConstraints, ParticlesData, PbrPlusMaterial, PresetShapeKind, Project,
    PropertyKeyframe, PropertyTrack, PropertyValue, Sequence, SequenceControl, SequenceControlKind,
    Shape2DKind, Shape2DNode, Shape3DKind, StaticBehavior, Text3DAlign, Text3DData,
    TextureMaterial, TrackToConstraint, VideoClip, VideoMaterial, VideoTrack,
};
pub use pz_import::{PzImportResult, import_pz, import_pz_from_bytes};
pub use sha256::{cm3_asset_key, sha256_bytes, sha256_hex};
pub use uuid::Uuid;
pub use zcraft::{
    IntegrityEntry, MAX_ARCHIVE_ENTRIES, MAX_JSON_ENTRY_BYTES, MAX_JSON_NESTING_DEPTH,
    MAX_UNCOMPRESSED_ARCHIVE_BYTES, ManifestPluginRequirement, ZcraftMeta, load_zcraft,
    load_zcraft_from_bytes, save_zcraft, save_zcraft_to_bytes,
};

/// Errors produced by project serialization, atomic I/O, archive validation, and `.pz` import.
#[derive(Debug, Error)]
pub enum ProjectError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("invalid or unsafe archive path: {0:?}")]
    InvalidArchivePath(String),

    #[error("disallowed archive entry type {kind} for path {path:?}")]
    DisallowedEntryType { path: String, kind: String },

    #[error("uncompressed archive exceeds limit of {limit_bytes} bytes")]
    ArchiveTooLarge { limit_bytes: u64 },

    #[error("JSON entry {entry:?} exceeds limit of {limit_bytes} bytes")]
    JsonEntryTooLarge { entry: String, limit_bytes: u64 },

    #[error("archive exceeds maximum entry count of {limit}")]
    TooManyEntries { limit: usize },

    #[error("JSON nesting depth exceeds maximum of {max_depth}")]
    JsonNestingTooDeep { max_depth: usize },

    #[error("missing required archive entry: {0}")]
    MissingEntry(&'static str),

    #[error("integrity mismatch for {entry:?}: expected {expected}, got {actual}")]
    IntegrityMismatch {
        entry: String,
        expected: String,
        actual: String,
    },

    #[error("unsupported or malformed project format: {0}")]
    UnsupportedFormat(String),
}
