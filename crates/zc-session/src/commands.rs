use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use zc_project::Uuid;

use crate::{SessionClip, TrackKind};

/// Serializable clipboard snapshot of a timeline clip (plus lossless project layer,
/// `LayerMask` list, and `Precomp+` frame offset) for [`Command::PasteClips`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipSnapshot {
    pub track_kind: TrackKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_id: Option<String>,
    pub clip: SessionClip,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_clip: Option<zc_project::VideoClip>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_clip: Option<zc_project::AudioClip>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layer_masks: Vec<zc_project::LayerMask>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precomp_offset: Option<f64>,
}

/// Unified serializable command enum covering every core ZoidCraft editor operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "command", content = "params")]
pub enum Command {
    // --- Project ---
    #[serde(rename = "project.new")]
    NewProject,
    #[serde(rename = "project.set_name")]
    SetProjectName { name: String },
    #[serde(rename = "project.set_resolution")]
    SetResolution { width: u32, height: u32 },
    #[serde(rename = "project.set_fps")]
    SetFps { fps: f64 },
    #[serde(rename = "project.set_sequence_length")]
    SetSequenceLength { frames: f64 },
    #[serde(rename = "project.undo")]
    Undo,
    #[serde(rename = "project.redo")]
    Redo,

    // --- Playback & Timeline ---
    #[serde(rename = "playback.set_frame")]
    SetFrame { frame: f64 },
    #[serde(rename = "playback.step_frame")]
    StepFrame { delta: f64 },
    #[serde(rename = "playback.toggle_play")]
    TogglePlay,
    #[serde(rename = "playback.set_speed")]
    SetSpeed { speed: f64 },
    #[serde(rename = "playback.toggle_loop")]
    ToggleLoop,
    #[serde(rename = "timeline.add_video_track")]
    AddVideoTrack {
        #[serde(default)]
        name: Option<String>,
    },
    #[serde(rename = "timeline.add_audio_track")]
    AddAudioTrack {
        #[serde(default)]
        name: Option<String>,
    },
    #[serde(rename = "timeline.remove_track")]
    RemoveTrack { track_id: String },
    #[serde(rename = "timeline.set_track_enabled")]
    SetTrackEnabled { track_id: String, enabled: bool },
    #[serde(rename = "timeline.set_track_solo")]
    SetTrackSolo { track_id: String, solo: bool },
    #[serde(rename = "timeline.set_track_locked")]
    SetTrackLocked { track_id: String, locked: bool },
    #[serde(rename = "timeline.add_clip")]
    AddClip {
        track_id: String,
        kind: String,
        #[serde(default)]
        name: Option<String>,
        start: f64,
        length: f64,
    },
    #[serde(rename = "timeline.move_clip")]
    MoveClip {
        clip_id: String,
        #[serde(default)]
        target_track_id: Option<String>,
        start: f64,
    },
    #[serde(rename = "timeline.trim_clip_left")]
    TrimClipLeft {
        clip_id: String,
        delta_frames: f64,
        #[serde(default)]
        rate_stretch: bool,
    },
    #[serde(rename = "timeline.trim_clip_right")]
    TrimClipRight {
        clip_id: String,
        delta_frames: f64,
        #[serde(default)]
        rate_stretch: bool,
    },
    #[serde(rename = "timeline.split_clip")]
    SplitClip { clip_id: String, at_frame: f64 },
    #[serde(rename = "timeline.duplicate_clip")]
    DuplicateClip { clip_id: String },
    #[serde(rename = "timeline.delete_clip")]
    DeleteClip { clip_id: String },

    // --- Scene & Effects ---
    #[serde(rename = "scene.add_object_3d")]
    AddObject3D {
        clip_id: String,
        #[serde(default)]
        parent_id: Option<String>,
        kind: String,
        #[serde(default)]
        name: Option<String>,
    },
    #[serde(rename = "scene.delete_object_3d")]
    DeleteObject3D { clip_id: String, object_id: String },
    #[serde(rename = "scene.duplicate_object_3d")]
    DuplicateObject3D { clip_id: String, object_id: String },
    #[serde(rename = "scene.reorder_object_3d")]
    ReorderObject3D {
        clip_id: String,
        object_id: String,
        #[serde(default)]
        parent_id: Option<String>,
        index: usize,
    },
    #[serde(rename = "effects.add_effect")]
    AddEffect {
        clip_id: String,
        kind: String,
        #[serde(default)]
        name: Option<String>,
    },
    #[serde(rename = "effects.delete_effect")]
    DeleteEffect { clip_id: String, effect_id: String },
    #[serde(rename = "effects.duplicate_effect")]
    DuplicateEffect { clip_id: String, effect_id: String },
    #[serde(rename = "effects.reorder_effect")]
    ReorderEffect {
        clip_id: String,
        effect_id: String,
        index: usize,
    },
    #[serde(rename = "property.set_value")]
    SetPropertyValue {
        target_id: String,
        property: String,
        value: Value,
    },
    #[serde(rename = "property.toggle_animated")]
    TogglePropertyAnimated {
        target_id: String,
        property: String,
        frame: f64,
    },
    #[serde(rename = "property.set_keyframe")]
    SetKeyframe {
        target_id: String,
        property: String,
        frame: f64,
        value: Value,
        #[serde(default)]
        tween: Option<u16>,
    },
    #[serde(rename = "property.delete_keyframe")]
    DeleteKeyframe {
        target_id: String,
        property: String,
        frame: f64,
    },
    #[serde(rename = "property.set_expression")]
    SetExpression {
        target_id: String,
        property: String,
        #[serde(default)]
        expression: Option<String>,
    },
    #[serde(rename = "property.remove")]
    RemoveProperty { target_id: String, property: String },
    #[serde(rename = "property.reset_list")]
    ResetPropertyList { target_id: String },

    // --- Plugins ---
    #[serde(rename = "plugins.enable")]
    EnablePlugin { plugin_id: String },
    #[serde(rename = "plugins.disable")]
    DisablePlugin { plugin_id: String },
    #[serde(rename = "plugins.disable_all_optional")]
    DisableAllOptionalPlugins,

    // --- Control / Query ---
    #[serde(rename = "control.inspect_state")]
    InspectState,
    #[serde(rename = "control.list_commands")]
    ListCommands,
    #[serde(rename = "control.list_plugins")]
    ListPlugins,

    // --- Shape+ ---
    #[serde(rename = "shape_plus.update_payload")]
    UpdateShapePlusPayload { clip_id: Uuid, payload: Value },

    // --- Layer Masks (zoidiumMasks) ---
    #[serde(rename = "mask.add")]
    AddLayerMask {
        clip_id: Uuid,
        mask: zc_project::LayerMask,
    },
    #[serde(rename = "mask.update")]
    UpdateLayerMask {
        clip_id: Uuid,
        mask_id: String,
        mask: zc_project::LayerMask,
    },
    #[serde(rename = "mask.remove")]
    RemoveLayerMask { clip_id: Uuid, mask_id: String },
    #[serde(rename = "mask.reorder")]
    ReorderLayerMask {
        clip_id: Uuid,
        mask_id: String,
        to_index: usize,
    },
    #[serde(rename = "mask.duplicate")]
    DuplicateLayerMask { clip_id: Uuid, mask_id: String },

    // --- Precomp+ Compositions ---
    #[serde(rename = "comp.create")]
    CreateComposition {
        name: String,
        resolution: [u32; 2],
        fps: u32,
        length_frames: u32,
    },
    #[serde(rename = "comp.rename")]
    RenameComposition { comp_id: Uuid, name: String },
    #[serde(rename = "comp.duplicate")]
    DuplicateComposition { comp_id: Uuid },
    #[serde(rename = "comp.delete")]
    DeleteComposition { comp_id: Uuid },
    #[serde(rename = "comp.precompose_clips")]
    PrecomposeClips { clip_ids: Vec<Uuid>, name: String },
    #[serde(rename = "comp.set_active")]
    SetActiveComposition {
        #[serde(default)]
        comp_id: Option<Uuid>,
    },
    #[serde(rename = "comp.set_clip_target")]
    SetPrecompClipTarget { clip_id: Uuid, composition_id: Uuid },

    // --- Layer Preserved / PresetShape2D / Split Letters ---
    #[serde(rename = "layer.update_preserved")]
    UpdateLayerPreserved {
        clip_id: Uuid,
        key: String,
        #[serde(default)]
        value: Option<Value>,
    },
    #[serde(rename = "layer.set_preset_shape_kind")]
    SetPresetShapeKind {
        clip_id: Uuid,
        shape_kind: zc_project::PresetShapeKind,
    },
    #[serde(rename = "scene.split_text_3d_into_letters")]
    SplitText3DIntoLetters { clip_id: Uuid, object_id: Uuid },

    // --- Pro Timeline / Markers / Sequence Controls / Assets ---
    #[serde(rename = "timeline.add_marker")]
    AddMarker {
        frame: f64,
        name: String,
        #[serde(default)]
        color: Option<String>,
    },
    #[serde(rename = "timeline.update_marker")]
    UpdateMarker {
        index: usize,
        frame: f64,
        name: String,
        #[serde(default)]
        color: Option<String>,
    },
    #[serde(rename = "timeline.delete_marker")]
    DeleteMarker { index: usize },
    #[serde(rename = "sequence.add_control")]
    AddSequenceControl {
        name: String,
        kind: zc_project::SequenceControlKind,
        value: zc_project::PropertyValue,
        #[serde(default)]
        min: Option<f64>,
        #[serde(default)]
        max: Option<f64>,
        #[serde(default)]
        step: Option<f64>,
    },
    #[serde(rename = "sequence.update_control")]
    UpdateSequenceControl {
        control_id: String,
        name: String,
        value: zc_project::PropertyValue,
        #[serde(default)]
        min: Option<f64>,
        #[serde(default)]
        max: Option<f64>,
        #[serde(default)]
        step: Option<f64>,
    },
    #[serde(rename = "sequence.delete_control")]
    DeleteSequenceControl { control_id: String },
    #[serde(rename = "sequence.reorder_control")]
    ReorderSequenceControl { control_id: String, to_index: usize },
    #[serde(rename = "timeline.paste_clips")]
    PasteClips {
        clips: Vec<ClipSnapshot>,
        target_frame: f64,
    },
    #[serde(rename = "timeline.freeze_clip_frame")]
    FreezeClipFrame { clip_id: Uuid, frame: f64 },
    #[serde(rename = "timeline.reverse_clip")]
    ReverseClip { clip_id: Uuid },
    #[serde(rename = "assets.relink")]
    RelinkAsset {
        asset_id: String,
        filename: String,
        bytes: Vec<u8>,
    },
    #[serde(rename = "assets.purge_unused")]
    PurgeUnusedAssets,

    // --- PanZoid (CM3 / Zoidium) Bi-Directional Clipboard ---
    #[serde(rename = "clipboard.paste_pz_json")]
    PastePzJson {
        json: String,
        #[serde(default)]
        hint: Option<String>,
        #[serde(default)]
        target_clip_id: Option<String>,
        #[serde(default)]
        parent_object_id: Option<String>,
        #[serde(default)]
        target_object_id: Option<String>,
        #[serde(default)]
        target_property: Option<String>,
        #[serde(default)]
        at_frame: Option<f64>,
    },
    #[serde(rename = "clipboard.copy_pz_json")]
    CopyPzJson {
        kind: String,
        #[serde(default)]
        clip_id: Option<String>,
        #[serde(default)]
        target_ids: Vec<String>,
        #[serde(default)]
        property: Option<String>,
    },
}

impl Command {
    /// Canonical dot-separated command identifier matching [`CommandMeta::id`].
    #[must_use]
    pub const fn id(&self) -> &'static str {
        match self {
            Self::NewProject => "project.new",
            Self::SetProjectName { .. } => "project.set_name",
            Self::SetResolution { .. } => "project.set_resolution",
            Self::SetFps { .. } => "project.set_fps",
            Self::SetSequenceLength { .. } => "project.set_sequence_length",
            Self::Undo => "project.undo",
            Self::Redo => "project.redo",
            Self::SetFrame { .. } => "playback.set_frame",
            Self::StepFrame { .. } => "playback.step_frame",
            Self::TogglePlay => "playback.toggle_play",
            Self::SetSpeed { .. } => "playback.set_speed",
            Self::ToggleLoop => "playback.toggle_loop",
            Self::AddVideoTrack { .. } => "timeline.add_video_track",
            Self::AddAudioTrack { .. } => "timeline.add_audio_track",
            Self::RemoveTrack { .. } => "timeline.remove_track",
            Self::SetTrackEnabled { .. } => "timeline.set_track_enabled",
            Self::SetTrackSolo { .. } => "timeline.set_track_solo",
            Self::SetTrackLocked { .. } => "timeline.set_track_locked",
            Self::AddClip { .. } => "timeline.add_clip",
            Self::MoveClip { .. } => "timeline.move_clip",
            Self::TrimClipLeft { .. } => "timeline.trim_clip_left",
            Self::TrimClipRight { .. } => "timeline.trim_clip_right",
            Self::SplitClip { .. } => "timeline.split_clip",
            Self::DuplicateClip { .. } => "timeline.duplicate_clip",
            Self::DeleteClip { .. } => "timeline.delete_clip",
            Self::AddObject3D { .. } => "scene.add_object_3d",
            Self::DeleteObject3D { .. } => "scene.delete_object_3d",
            Self::DuplicateObject3D { .. } => "scene.duplicate_object_3d",
            Self::ReorderObject3D { .. } => "scene.reorder_object_3d",
            Self::AddEffect { .. } => "effects.add_effect",
            Self::DeleteEffect { .. } => "effects.delete_effect",
            Self::DuplicateEffect { .. } => "effects.duplicate_effect",
            Self::ReorderEffect { .. } => "effects.reorder_effect",
            Self::SetPropertyValue { .. } => "property.set_value",
            Self::TogglePropertyAnimated { .. } => "property.toggle_animated",
            Self::SetKeyframe { .. } => "property.set_keyframe",
            Self::DeleteKeyframe { .. } => "property.delete_keyframe",
            Self::SetExpression { .. } => "property.set_expression",
            Self::RemoveProperty { .. } => "property.remove",
            Self::ResetPropertyList { .. } => "property.reset_list",
            Self::EnablePlugin { .. } => "plugins.enable",
            Self::DisablePlugin { .. } => "plugins.disable",
            Self::DisableAllOptionalPlugins => "plugins.disable_all_optional",
            Self::InspectState => "control.inspect_state",
            Self::ListCommands => "control.list_commands",
            Self::ListPlugins => "control.list_plugins",
            Self::UpdateShapePlusPayload { .. } => "shape_plus.update_payload",
            Self::AddLayerMask { .. } => "mask.add",
            Self::UpdateLayerMask { .. } => "mask.update",
            Self::RemoveLayerMask { .. } => "mask.remove",
            Self::ReorderLayerMask { .. } => "mask.reorder",
            Self::DuplicateLayerMask { .. } => "mask.duplicate",
            Self::CreateComposition { .. } => "comp.create",
            Self::RenameComposition { .. } => "comp.rename",
            Self::DuplicateComposition { .. } => "comp.duplicate",
            Self::DeleteComposition { .. } => "comp.delete",
            Self::PrecomposeClips { .. } => "comp.precompose_clips",
            Self::SetActiveComposition { .. } => "comp.set_active",
            Self::SetPrecompClipTarget { .. } => "comp.set_clip_target",
            Self::UpdateLayerPreserved { .. } => "layer.update_preserved",
            Self::SetPresetShapeKind { .. } => "layer.set_preset_shape_kind",
            Self::SplitText3DIntoLetters { .. } => "scene.split_text_3d_into_letters",
            Self::AddMarker { .. } => "timeline.add_marker",
            Self::UpdateMarker { .. } => "timeline.update_marker",
            Self::DeleteMarker { .. } => "timeline.delete_marker",
            Self::AddSequenceControl { .. } => "sequence.add_control",
            Self::UpdateSequenceControl { .. } => "sequence.update_control",
            Self::DeleteSequenceControl { .. } => "sequence.delete_control",
            Self::ReorderSequenceControl { .. } => "sequence.reorder_control",
            Self::PasteClips { .. } => "timeline.paste_clips",
            Self::FreezeClipFrame { .. } => "timeline.freeze_clip_frame",
            Self::ReverseClip { .. } => "timeline.reverse_clip",
            Self::RelinkAsset { .. } => "assets.relink",
            Self::PurgeUnusedAssets => "assets.purge_unused",
            Self::PastePzJson { .. } => "clipboard.paste_pz_json",
            Self::CopyPzJson { .. } => "clipboard.copy_pz_json",
        }
    }

    /// Returns `true` if this command mutates project state and should push an undo step.
    #[must_use]
    pub const fn is_undoable(&self) -> bool {
        matches!(
            self,
            Self::SetProjectName { .. }
                | Self::SetResolution { .. }
                | Self::SetFps { .. }
                | Self::SetSequenceLength { .. }
                | Self::AddVideoTrack { .. }
                | Self::AddAudioTrack { .. }
                | Self::RemoveTrack { .. }
                | Self::SetTrackEnabled { .. }
                | Self::SetTrackSolo { .. }
                | Self::SetTrackLocked { .. }
                | Self::AddClip { .. }
                | Self::MoveClip { .. }
                | Self::TrimClipLeft { .. }
                | Self::TrimClipRight { .. }
                | Self::SplitClip { .. }
                | Self::DuplicateClip { .. }
                | Self::DeleteClip { .. }
                | Self::AddObject3D { .. }
                | Self::DeleteObject3D { .. }
                | Self::DuplicateObject3D { .. }
                | Self::ReorderObject3D { .. }
                | Self::AddEffect { .. }
                | Self::DeleteEffect { .. }
                | Self::DuplicateEffect { .. }
                | Self::ReorderEffect { .. }
                | Self::SetPropertyValue { .. }
                | Self::TogglePropertyAnimated { .. }
                | Self::SetKeyframe { .. }
                | Self::DeleteKeyframe { .. }
                | Self::SetExpression { .. }
                | Self::RemoveProperty { .. }
                | Self::ResetPropertyList { .. }
                | Self::UpdateShapePlusPayload { .. }
                | Self::AddLayerMask { .. }
                | Self::UpdateLayerMask { .. }
                | Self::RemoveLayerMask { .. }
                | Self::ReorderLayerMask { .. }
                | Self::DuplicateLayerMask { .. }
                | Self::CreateComposition { .. }
                | Self::RenameComposition { .. }
                | Self::DuplicateComposition { .. }
                | Self::DeleteComposition { .. }
                | Self::PrecomposeClips { .. }
                | Self::SetPrecompClipTarget { .. }
                | Self::UpdateLayerPreserved { .. }
                | Self::SetPresetShapeKind { .. }
                | Self::SplitText3DIntoLetters { .. }
                | Self::AddMarker { .. }
                | Self::UpdateMarker { .. }
                | Self::DeleteMarker { .. }
                | Self::AddSequenceControl { .. }
                | Self::UpdateSequenceControl { .. }
                | Self::DeleteSequenceControl { .. }
                | Self::ReorderSequenceControl { .. }
                | Self::PasteClips { .. }
                | Self::FreezeClipFrame { .. }
                | Self::ReverseClip { .. }
                | Self::RelinkAsset { .. }
                | Self::PurgeUnusedAssets
                | Self::PastePzJson { .. }
        )
    }

    /// Resolves a canonical command ID or variant name + JSON parameter object into a [`Command`].
    pub fn from_name_and_params(name: &str, params: Value) -> Result<Self, String> {
        let canonical = resolve_canonical_command_id(name)
            .ok_or_else(|| format!("unknown command '{name}'"))?;

        if is_unit_command(canonical) {
            match &params {
                Value::Null => {}
                Value::Object(map) if map.is_empty() => {}
                _ => {
                    return Err(format!("command '{canonical}' does not accept parameters"));
                }
            }
            let tagged = json!({ "command": canonical });
            return serde_json::from_value(tagged)
                .map_err(|err| format!("failed to construct command '{canonical}': {err}"));
        }

        let normalized_params = match params {
            Value::Null => json!({}),
            other => other,
        };
        let tagged = json!({
            "command": canonical,
            "params": normalized_params,
        });
        serde_json::from_value(tagged)
            .map_err(|err| format!("invalid parameters for '{canonical}': {err}"))
    }
}

fn is_unit_command(canonical: &str) -> bool {
    matches!(
        canonical,
        "project.new"
            | "project.undo"
            | "project.redo"
            | "playback.toggle_play"
            | "playback.toggle_loop"
            | "plugins.disable_all_optional"
            | "control.inspect_state"
            | "control.list_commands"
            | "control.list_plugins"
            | "assets.purge_unused"
    )
}

fn resolve_canonical_command_id(name: &str) -> Option<&'static str> {
    match name {
        "project.new" | "NewProject" => Some("project.new"),
        "project.set_name" | "SetProjectName" => Some("project.set_name"),
        "project.set_resolution" | "SetResolution" => Some("project.set_resolution"),
        "project.set_fps" | "SetFps" => Some("project.set_fps"),
        "project.set_sequence_length" | "SetSequenceLength" => Some("project.set_sequence_length"),
        "project.undo" | "Undo" => Some("project.undo"),
        "project.redo" | "Redo" => Some("project.redo"),
        "playback.set_frame" | "SetFrame" => Some("playback.set_frame"),
        "playback.step_frame" | "StepFrame" => Some("playback.step_frame"),
        "playback.toggle_play" | "TogglePlay" => Some("playback.toggle_play"),
        "playback.set_speed" | "SetSpeed" => Some("playback.set_speed"),
        "playback.toggle_loop" | "ToggleLoop" => Some("playback.toggle_loop"),
        "timeline.add_video_track" | "AddVideoTrack" => Some("timeline.add_video_track"),
        "timeline.add_audio_track" | "AddAudioTrack" => Some("timeline.add_audio_track"),
        "timeline.remove_track" | "RemoveTrack" => Some("timeline.remove_track"),
        "timeline.set_track_enabled" | "SetTrackEnabled" => Some("timeline.set_track_enabled"),
        "timeline.set_track_solo" | "SetTrackSolo" => Some("timeline.set_track_solo"),
        "timeline.set_track_locked" | "SetTrackLocked" => Some("timeline.set_track_locked"),
        "timeline.add_clip" | "AddClip" => Some("timeline.add_clip"),
        "timeline.move_clip" | "MoveClip" => Some("timeline.move_clip"),
        "timeline.trim_clip_left" | "TrimClipLeft" => Some("timeline.trim_clip_left"),
        "timeline.trim_clip_right" | "TrimClipRight" => Some("timeline.trim_clip_right"),
        "timeline.split_clip" | "SplitClip" => Some("timeline.split_clip"),
        "timeline.duplicate_clip" | "DuplicateClip" => Some("timeline.duplicate_clip"),
        "timeline.delete_clip" | "DeleteClip" => Some("timeline.delete_clip"),
        "scene.add_object_3d" | "AddObject3D" => Some("scene.add_object_3d"),
        "scene.delete_object_3d" | "DeleteObject3D" => Some("scene.delete_object_3d"),
        "scene.duplicate_object_3d" | "DuplicateObject3D" => Some("scene.duplicate_object_3d"),
        "scene.reorder_object_3d" | "ReorderObject3D" => Some("scene.reorder_object_3d"),
        "effects.add_effect" | "AddEffect" => Some("effects.add_effect"),
        "effects.delete_effect" | "DeleteEffect" => Some("effects.delete_effect"),
        "effects.duplicate_effect" | "DuplicateEffect" => Some("effects.duplicate_effect"),
        "effects.reorder_effect" | "ReorderEffect" => Some("effects.reorder_effect"),
        "property.set_value" | "SetPropertyValue" => Some("property.set_value"),
        "property.toggle_animated" | "TogglePropertyAnimated" => Some("property.toggle_animated"),
        "property.set_keyframe" | "SetKeyframe" => Some("property.set_keyframe"),
        "property.delete_keyframe" | "DeleteKeyframe" => Some("property.delete_keyframe"),
        "property.set_expression" | "SetExpression" => Some("property.set_expression"),
        "property.remove" | "RemoveProperty" => Some("property.remove"),
        "property.reset_list" | "ResetPropertyList" => Some("property.reset_list"),
        "plugins.enable" | "EnablePlugin" => Some("plugins.enable"),
        "plugins.disable" | "DisablePlugin" => Some("plugins.disable"),
        "plugins.disable_all_optional" | "DisableAllOptionalPlugins" => {
            Some("plugins.disable_all_optional")
        }
        "control.inspect_state" | "InspectState" => Some("control.inspect_state"),
        "control.list_commands" | "ListCommands" => Some("control.list_commands"),
        "control.list_plugins" | "ListPlugins" => Some("control.list_plugins"),
        "shape_plus.update_payload" | "UpdateShapePlusPayload" => Some("shape_plus.update_payload"),
        "mask.add" | "AddLayerMask" => Some("mask.add"),
        "mask.update" | "UpdateLayerMask" => Some("mask.update"),
        "mask.remove" | "RemoveLayerMask" => Some("mask.remove"),
        "mask.reorder" | "ReorderLayerMask" => Some("mask.reorder"),
        "mask.duplicate" | "DuplicateLayerMask" => Some("mask.duplicate"),
        "comp.create" | "CreateComposition" => Some("comp.create"),
        "comp.rename" | "RenameComposition" => Some("comp.rename"),
        "comp.duplicate" | "DuplicateComposition" => Some("comp.duplicate"),
        "comp.delete" | "DeleteComposition" => Some("comp.delete"),
        "comp.precompose_clips" | "PrecomposeClips" => Some("comp.precompose_clips"),
        "comp.set_active" | "SetActiveComposition" => Some("comp.set_active"),
        "comp.set_clip_target" | "SetPrecompClipTarget" => Some("comp.set_clip_target"),
        "layer.update_preserved" | "UpdateLayerPreserved" => Some("layer.update_preserved"),
        "layer.set_preset_shape_kind" | "SetPresetShapeKind" => Some("layer.set_preset_shape_kind"),
        "scene.split_text_3d_into_letters" | "SplitText3DIntoLetters" => {
            Some("scene.split_text_3d_into_letters")
        }
        "timeline.add_marker" | "AddMarker" => Some("timeline.add_marker"),
        "timeline.update_marker" | "UpdateMarker" => Some("timeline.update_marker"),
        "timeline.delete_marker" | "DeleteMarker" => Some("timeline.delete_marker"),
        "sequence.add_control" | "AddSequenceControl" => Some("sequence.add_control"),
        "sequence.update_control" | "UpdateSequenceControl" => Some("sequence.update_control"),
        "sequence.delete_control" | "DeleteSequenceControl" => Some("sequence.delete_control"),
        "sequence.reorder_control" | "ReorderSequenceControl" => Some("sequence.reorder_control"),
        "timeline.paste_clips" | "PasteClips" => Some("timeline.paste_clips"),
        "timeline.freeze_clip_frame" | "FreezeClipFrame" => Some("timeline.freeze_clip_frame"),
        "timeline.reverse_clip" | "ReverseClip" => Some("timeline.reverse_clip"),
        "assets.relink" | "RelinkAsset" => Some("assets.relink"),
        "assets.purge_unused" | "PurgeUnusedAssets" => Some("assets.purge_unused"),
        "clipboard.paste_pz_json" | "PastePzJson" => Some("clipboard.paste_pz_json"),
        "clipboard.copy_pz_json" | "CopyPzJson" => Some("clipboard.copy_pz_json"),
        _ => None,
    }
}

/// Metadata describing a registered command for menus, command palettes, CLI help, and MCP schemas.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandMeta {
    pub id: String,
    pub menu_path: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shortcut: Option<String>,
    pub undoable: bool,
    pub params_schema: Value,
}

impl CommandMeta {
    fn new(
        id: &str,
        menu_path: &str,
        description: &str,
        shortcut: Option<&str>,
        undoable: bool,
        params_schema: Value,
    ) -> Self {
        Self {
            id: id.to_string(),
            menu_path: menu_path.to_string(),
            description: description.to_string(),
            shortcut: shortcut.map(str::to_string),
            undoable,
            params_schema,
        }
    }
}

/// ArtCraft-style command registry indexing every built-in and plugin command with
/// menu paths, shortcuts, and JSON parameter schemas.
#[derive(Debug, Clone)]
pub struct CommandRegistry {
    commands: BTreeMap<String, CommandMeta>,
}

impl Default for CommandRegistry {
    fn default() -> Self {
        Self::new_default()
    }
}

impl CommandRegistry {
    /// Creates a registry populated with all 46 built-in ZoidCraft commands.
    #[must_use]
    pub fn new_default() -> Self {
        let mut reg = Self {
            commands: BTreeMap::new(),
        };
        for meta in builtin_command_metadata() {
            reg.register(meta);
        }
        reg
    }

    /// Registers or updates a command's metadata entry.
    pub fn register(&mut self, meta: CommandMeta) {
        self.commands.insert(meta.id.clone(), meta);
    }

    /// Looks up metadata by canonical command ID (or variant alias).
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&CommandMeta> {
        if let Some(meta) = self.commands.get(id) {
            return Some(meta);
        }
        let canonical = resolve_canonical_command_id(id)?;
        self.commands.get(canonical)
    }

    /// Lists all registered commands in deterministic ID order.
    #[must_use]
    pub fn list(&self) -> Vec<&CommandMeta> {
        self.commands.values().collect()
    }

    /// Case-insensitive search across command ID, menu path, and description.
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<&CommandMeta> {
        let q = query.trim().to_ascii_lowercase();
        if q.is_empty() {
            return self.list();
        }
        self.commands
            .values()
            .filter(|m| {
                m.id.to_ascii_lowercase().contains(&q)
                    || m.menu_path.to_ascii_lowercase().contains(&q)
                    || m.description.to_ascii_lowercase().contains(&q)
            })
            .collect()
    }

    /// Number of registered commands.
    #[must_use]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Returns `true` if no commands are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

fn empty_schema() -> Value {
    json!({
        "type": "object",
        "properties": {},
        "additionalProperties": false
    })
}

fn object_schema(properties: Value, required: &[&str]) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
}

fn builtin_command_metadata() -> Vec<CommandMeta> {
    vec![
        // --- Project ---
        CommandMeta::new(
            "project.new",
            "File/New Project",
            "Create a fresh default ClipMaker 3 project and reset session history",
            Some("Ctrl+N"),
            false,
            empty_schema(),
        ),
        CommandMeta::new(
            "project.set_name",
            "File/Rename Project",
            "Set the active project display name",
            None,
            true,
            object_schema(json!({ "name": { "type": "string" } }), &["name"]),
        ),
        CommandMeta::new(
            "project.set_resolution",
            "Sequence/Set Resolution",
            "Set the sequence output width and height in pixels",
            None,
            true,
            object_schema(
                json!({
                    "width": { "type": "integer", "minimum": 1, "maximum": 7680 },
                    "height": { "type": "integer", "minimum": 1, "maximum": 7680 }
                }),
                &["width", "height"],
            ),
        ),
        CommandMeta::new(
            "project.set_fps",
            "Sequence/Set Frame Rate",
            "Set the sequence frame rate in frames per second",
            None,
            true,
            object_schema(
                json!({ "fps": { "type": "number", "exclusiveMinimum": 0.0, "maximum": 240.0 } }),
                &["fps"],
            ),
        ),
        CommandMeta::new(
            "project.set_sequence_length",
            "Sequence/Set Sequence Length",
            "Set the sequence total duration in frames",
            None,
            true,
            object_schema(
                json!({ "frames": { "type": "number", "minimum": 0.0 } }),
                &["frames"],
            ),
        ),
        CommandMeta::new(
            "project.undo",
            "Edit/Undo",
            "Undo the most recent history operation",
            Some("Ctrl+Z"),
            false,
            empty_schema(),
        ),
        CommandMeta::new(
            "project.redo",
            "Edit/Redo",
            "Redo the most recently undone history operation",
            Some("Ctrl+Y"),
            false,
            empty_schema(),
        ),
        // --- Playback & Timeline ---
        CommandMeta::new(
            "playback.set_frame",
            "Playback/Seek Frame",
            "Pause playback and move the playhead to the specified frame",
            None,
            false,
            object_schema(json!({ "frame": { "type": "number" } }), &["frame"]),
        ),
        CommandMeta::new(
            "playback.step_frame",
            "Playback/Step Frame",
            "Pause playback and step the playhead by a relative frame delta",
            Some("ArrowLeft / ArrowRight"),
            false,
            object_schema(json!({ "delta": { "type": "number" } }), &["delta"]),
        ),
        CommandMeta::new(
            "playback.toggle_play",
            "Playback/Play / Pause",
            "Toggle playback speed between paused (0) and forward (1)",
            Some("Space"),
            false,
            empty_schema(),
        ),
        CommandMeta::new(
            "playback.set_speed",
            "Playback/Set Speed",
            "Set playback shuttle speed multiplier (0 = paused, negative = reverse)",
            Some("J / K / L"),
            false,
            object_schema(json!({ "speed": { "type": "number" } }), &["speed"]),
        ),
        CommandMeta::new(
            "playback.toggle_loop",
            "Playback/Toggle Loop",
            "Toggle playback looping at sequence boundaries",
            Some("Ctrl+L"),
            false,
            empty_schema(),
        ),
        CommandMeta::new(
            "timeline.add_video_track",
            "Timeline/Tracks/Add Video Track",
            "Append a new video track at the top of the compositing stack",
            None,
            true,
            object_schema(json!({ "name": { "type": ["string", "null"] } }), &[]),
        ),
        CommandMeta::new(
            "timeline.add_audio_track",
            "Timeline/Tracks/Add Audio Track",
            "Append a new audio track below existing audio tracks",
            None,
            true,
            object_schema(json!({ "name": { "type": ["string", "null"] } }), &[]),
        ),
        CommandMeta::new(
            "timeline.remove_track",
            "Timeline/Tracks/Remove Track",
            "Remove a video or audio track and all clips on it",
            None,
            true,
            object_schema(json!({ "track_id": { "type": "string" } }), &["track_id"]),
        ),
        CommandMeta::new(
            "timeline.set_track_enabled",
            "Timeline/Tracks/Toggle Track Enabled",
            "Enable or disable rendering/mixing for a track",
            None,
            true,
            object_schema(
                json!({
                    "track_id": { "type": "string" },
                    "enabled": { "type": "boolean" }
                }),
                &["track_id", "enabled"],
            ),
        ),
        CommandMeta::new(
            "timeline.set_track_solo",
            "Timeline/Tracks/Toggle Track Solo",
            "Toggle solo state on a track",
            None,
            true,
            object_schema(
                json!({
                    "track_id": { "type": "string" },
                    "solo": { "type": "boolean" }
                }),
                &["track_id", "solo"],
            ),
        ),
        CommandMeta::new(
            "timeline.set_track_locked",
            "Timeline/Tracks/Toggle Track Locked",
            "Lock or unlock editing for clips on a track",
            None,
            true,
            object_schema(
                json!({
                    "track_id": { "type": "string" },
                    "locked": { "type": "boolean" }
                }),
                &["track_id", "locked"],
            ),
        ),
        CommandMeta::new(
            "timeline.add_clip",
            "Timeline/Clips/Add Clip",
            "Create a clip of the requested layer or audio kind on a track",
            None,
            true,
            object_schema(
                json!({
                    "track_id": { "type": "string" },
                    "kind": { "type": "string" },
                    "name": { "type": ["string", "null"] },
                    "start": { "type": "number", "minimum": 0.0 },
                    "length": { "type": "number", "minimum": 1.0 }
                }),
                &["track_id", "kind", "start", "length"],
            ),
        ),
        CommandMeta::new(
            "timeline.move_clip",
            "Timeline/Clips/Move Clip",
            "Move a clip to a new start frame and optional destination track of matching media type",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "target_track_id": { "type": ["string", "null"] },
                    "start": { "type": "number", "minimum": 0.0 }
                }),
                &["clip_id", "start"],
            ),
        ),
        CommandMeta::new(
            "timeline.trim_clip_left",
            "Timeline/Clips/Trim Left",
            "Trim or rate-stretch a clip from its left edge",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "delta_frames": { "type": "number" },
                    "rate_stretch": { "type": "boolean", "default": false }
                }),
                &["clip_id", "delta_frames"],
            ),
        ),
        CommandMeta::new(
            "timeline.trim_clip_right",
            "Timeline/Clips/Trim Right",
            "Trim or rate-stretch a clip from its right edge",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "delta_frames": { "type": "number" },
                    "rate_stretch": { "type": "boolean", "default": false }
                }),
                &["clip_id", "delta_frames"],
            ),
        ),
        CommandMeta::new(
            "timeline.split_clip",
            "Timeline/Clips/Split Clip",
            "Split a clip at an interior timeline frame, shifting right-half keyframes",
            Some("C"),
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "at_frame": { "type": "number" }
                }),
                &["clip_id", "at_frame"],
            ),
        ),
        CommandMeta::new(
            "timeline.duplicate_clip",
            "Timeline/Clips/Duplicate Clip",
            "Duplicate a clip on its track immediately after the original",
            None,
            true,
            object_schema(json!({ "clip_id": { "type": "string" } }), &["clip_id"]),
        ),
        CommandMeta::new(
            "timeline.delete_clip",
            "Timeline/Clips/Delete Clip",
            "Delete a clip from its track",
            Some("Delete"),
            true,
            object_schema(json!({ "clip_id": { "type": "string" } }), &["clip_id"]),
        ),
        // --- Scene & Effects ---
        CommandMeta::new(
            "scene.add_object_3d",
            "Objects/Add 3D Object",
            "Add a 3D object to a Scene clip (optionally nested under a parent group)",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "parent_id": { "type": ["string", "null"] },
                    "kind": { "type": "string" },
                    "name": { "type": ["string", "null"] }
                }),
                &["clip_id", "kind"],
            ),
        ),
        CommandMeta::new(
            "scene.delete_object_3d",
            "Objects/Delete 3D Object",
            "Delete a 3D object and its descendants from a Scene clip",
            Some("Delete"),
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "object_id": { "type": "string" }
                }),
                &["clip_id", "object_id"],
            ),
        ),
        CommandMeta::new(
            "scene.duplicate_object_3d",
            "Objects/Duplicate 3D Object",
            "Duplicate a 3D object within its Scene clip",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "object_id": { "type": "string" }
                }),
                &["clip_id", "object_id"],
            ),
        ),
        CommandMeta::new(
            "scene.reorder_object_3d",
            "Objects/Reorder 3D Object",
            "Reorder or reparent a 3D object within a Scene clip (rejecting cycles)",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "object_id": { "type": "string" },
                    "parent_id": { "type": ["string", "null"] },
                    "index": { "type": "integer", "minimum": 0 }
                }),
                &["clip_id", "object_id", "index"],
            ),
        ),
        CommandMeta::new(
            "effects.add_effect",
            "Effects/Add Effect",
            "Append an effect to a clip's effect stack",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "kind": { "type": "string" },
                    "name": { "type": ["string", "null"] }
                }),
                &["clip_id", "kind"],
            ),
        ),
        CommandMeta::new(
            "effects.delete_effect",
            "Effects/Delete Effect",
            "Delete an effect from a clip's effect stack",
            Some("Delete"),
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "effect_id": { "type": "string" }
                }),
                &["clip_id", "effect_id"],
            ),
        ),
        CommandMeta::new(
            "effects.duplicate_effect",
            "Effects/Duplicate Effect",
            "Duplicate an effect immediately after the original in the clip stack",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "effect_id": { "type": "string" }
                }),
                &["clip_id", "effect_id"],
            ),
        ),
        CommandMeta::new(
            "effects.reorder_effect",
            "Effects/Reorder Effect",
            "Move an effect to a new position in the clip's effect stack",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "effect_id": { "type": "string" },
                    "index": { "type": "integer", "minimum": 0 }
                }),
                &["clip_id", "effect_id", "index"],
            ),
        ),
        CommandMeta::new(
            "property.set_value",
            "Properties/Set Value",
            "Set a property's static value (or keyframe at frame 0 when static)",
            None,
            true,
            object_schema(
                json!({
                    "target_id": { "type": "string" },
                    "property": { "type": "string" },
                    "value": {}
                }),
                &["target_id", "property", "value"],
            ),
        ),
        CommandMeta::new(
            "property.toggle_animated",
            "Properties/Toggle Animation",
            "Toggle animation stopwatch on a property at the given clip-local frame",
            None,
            true,
            object_schema(
                json!({
                    "target_id": { "type": "string" },
                    "property": { "type": "string" },
                    "frame": { "type": "number" }
                }),
                &["target_id", "property", "frame"],
            ),
        ),
        CommandMeta::new(
            "property.set_keyframe",
            "Properties/Set Keyframe",
            "Insert or update a keyframe on a property at the given frame",
            None,
            true,
            object_schema(
                json!({
                    "target_id": { "type": "string" },
                    "property": { "type": "string" },
                    "frame": { "type": "number" },
                    "value": {},
                    "tween": { "type": ["integer", "null"], "minimum": 0, "maximum": 65535 }
                }),
                &["target_id", "property", "frame", "value"],
            ),
        ),
        CommandMeta::new(
            "property.delete_keyframe",
            "Properties/Delete Keyframe",
            "Delete a keyframe at the given frame (preserving the sole remaining keyframe)",
            Some("Delete"),
            true,
            object_schema(
                json!({
                    "target_id": { "type": "string" },
                    "property": { "type": "string" },
                    "frame": { "type": "number" }
                }),
                &["target_id", "property", "frame"],
            ),
        ),
        CommandMeta::new(
            "property.set_expression",
            "Properties/Set Expression",
            "Attach or clear a deterministic expression string on a property",
            None,
            true,
            object_schema(
                json!({
                    "target_id": { "type": "string" },
                    "property": { "type": "string" },
                    "expression": { "type": ["string", "null"] }
                }),
                &["target_id", "property"],
            ),
        ),
        CommandMeta::new(
            "property.remove",
            "Properties/Remove Property",
            "Remove a dynamic or custom property (and its keyframes/expressions) from a target",
            None,
            true,
            object_schema(
                json!({
                    "target_id": { "type": "string" },
                    "property": { "type": "string" }
                }),
                &["target_id", "property"],
            ),
        ),
        CommandMeta::new(
            "property.reset_list",
            "Properties/Reset Property List",
            "Reset all properties on a target to static defaults at frame 0 in one undo step",
            None,
            true,
            object_schema(json!({ "target_id": { "type": "string" } }), &["target_id"]),
        ),
        // --- Plugins ---
        CommandMeta::new(
            "plugins.enable",
            "Plugins/Enable Plugin",
            "Enable an optional native Zoidium plugin by ID",
            None,
            false,
            object_schema(json!({ "plugin_id": { "type": "string" } }), &["plugin_id"]),
        ),
        CommandMeta::new(
            "plugins.disable",
            "Plugins/Disable Plugin",
            "Disable an optional native Zoidium plugin (blocked if in use by active project)",
            None,
            false,
            object_schema(json!({ "plugin_id": { "type": "string" } }), &["plugin_id"]),
        ),
        CommandMeta::new(
            "plugins.disable_all_optional",
            "Plugins/Disable All Optional Plugins",
            "Disable all unlocked optional plugins to enter Vanilla CM3 parity mode",
            None,
            false,
            empty_schema(),
        ),
        // --- Control / Query ---
        CommandMeta::new(
            "control.inspect_state",
            "Control/Inspect State",
            "Return a deterministic JSON snapshot of the active session state",
            None,
            false,
            empty_schema(),
        ),
        CommandMeta::new(
            "control.list_commands",
            "Control/List Commands",
            "List all registered commands with menu paths, shortcuts, and JSON schemas",
            None,
            false,
            empty_schema(),
        ),
        CommandMeta::new(
            "control.list_plugins",
            "Control/List Plugins",
            "List all native Zoidium plugins with enabled state, category, and in-use lock status",
            None,
            false,
            empty_schema(),
        ),
        // --- Shape+ ---
        CommandMeta::new(
            "shape_plus.update_payload",
            "Shape+/Update Payload",
            "Replace a Shape+ layer's JSON vector document payload",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "payload": { "type": "object" }
                }),
                &["clip_id", "payload"],
            ),
        ),
        // --- Layer Masks ---
        CommandMeta::new(
            "mask.add",
            "Masks/Add Layer Mask",
            "Attach a new Shape+ LayerMask to a video clip's layer",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "mask": { "type": "object" }
                }),
                &["clip_id", "mask"],
            ),
        ),
        CommandMeta::new(
            "mask.update",
            "Masks/Update Layer Mask",
            "Update an existing LayerMask on a video clip's layer",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "mask_id": { "type": "string" },
                    "mask": { "type": "object" }
                }),
                &["clip_id", "mask_id", "mask"],
            ),
        ),
        CommandMeta::new(
            "mask.remove",
            "Masks/Remove Layer Mask",
            "Remove a LayerMask from a video clip's layer",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "mask_id": { "type": "string" }
                }),
                &["clip_id", "mask_id"],
            ),
        ),
        CommandMeta::new(
            "mask.reorder",
            "Masks/Reorder Layer Mask",
            "Move a LayerMask to a new index in the layer's mask stack",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "mask_id": { "type": "string" },
                    "to_index": { "type": "integer", "minimum": 0 }
                }),
                &["clip_id", "mask_id", "to_index"],
            ),
        ),
        CommandMeta::new(
            "mask.duplicate",
            "Masks/Duplicate Layer Mask",
            "Duplicate a LayerMask immediately after the original",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "mask_id": { "type": "string" }
                }),
                &["clip_id", "mask_id"],
            ),
        ),
        // --- Precomp+ Compositions ---
        CommandMeta::new(
            "comp.create",
            "Composition/New Composition",
            "Create a new nested Precomp+ composition",
            None,
            true,
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "resolution": { "type": "array", "items": { "type": "integer" }, "minItems": 2, "maxItems": 2 },
                    "fps": { "type": "integer", "minimum": 1, "maximum": 240 },
                    "length_frames": { "type": "integer", "minimum": 1 }
                }),
                &["name", "resolution", "fps", "length_frames"],
            ),
        ),
        CommandMeta::new(
            "comp.rename",
            "Composition/Rename Composition",
            "Rename an existing nested Precomp+ composition",
            None,
            true,
            object_schema(
                json!({
                    "comp_id": { "type": "string" },
                    "name": { "type": "string" }
                }),
                &["comp_id", "name"],
            ),
        ),
        CommandMeta::new(
            "comp.duplicate",
            "Composition/Duplicate Composition",
            "Duplicate an existing nested Precomp+ composition with fresh IDs",
            None,
            true,
            object_schema(json!({ "comp_id": { "type": "string" } }), &["comp_id"]),
        ),
        CommandMeta::new(
            "comp.delete",
            "Composition/Delete Composition",
            "Delete a nested Precomp+ composition if not referenced by any PrecompRef clip",
            None,
            true,
            object_schema(json!({ "comp_id": { "type": "string" } }), &["comp_id"]),
        ),
        CommandMeta::new(
            "comp.precompose_clips",
            "Composition/Precompose Clips",
            "Move selected clips into a new nested composition and replace with a PrecompRef clip",
            None,
            true,
            object_schema(
                json!({
                    "clip_ids": { "type": "array", "items": { "type": "string" } },
                    "name": { "type": "string" }
                }),
                &["clip_ids", "name"],
            ),
        ),
        CommandMeta::new(
            "comp.set_active",
            "Composition/Switch Active Composition",
            "Switch active timeline editing between root sequence (null) and a nested composition",
            None,
            false,
            object_schema(json!({ "comp_id": { "type": ["string", "null"] } }), &[]),
        ),
        CommandMeta::new(
            "comp.set_clip_target",
            "Composition/Set Precomp Clip Target",
            "Retarget a PrecompRef clip to reference a different nested composition",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "composition_id": { "type": "string" }
                }),
                &["clip_id", "composition_id"],
            ),
        ),
        // --- Layer Preserved / PresetShape2D / Split Letters ---
        CommandMeta::new(
            "layer.update_preserved",
            "Layer/Update Preserved Field",
            "Insert, update, or remove a key in a clip layer's preserved metadata map",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "key": { "type": "string" },
                    "value": {}
                }),
                &["clip_id", "key"],
            ),
        ),
        CommandMeta::new(
            "layer.set_preset_shape_kind",
            "Layer/Set Preset Shape Kind",
            "Switch a PresetShape2D clip between Ellipse and Rectangle",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "shape_kind": { "type": "string", "enum": ["ellipse", "rectangle"] }
                }),
                &["clip_id", "shape_kind"],
            ),
        ),
        CommandMeta::new(
            "scene.split_text_3d_into_letters",
            "Objects/Split 3D Text Into Letters",
            "Split a Text3D object into a Group containing one Text3D per non-whitespace character",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "object_id": { "type": "string" }
                }),
                &["clip_id", "object_id"],
            ),
        ),
        // --- Pro Timeline / Markers / Sequence Controls / Assets ---
        CommandMeta::new(
            "timeline.add_marker",
            "Timeline/Markers/Add Marker",
            "Add a named timeline marker at a frame",
            Some("M"),
            true,
            object_schema(
                json!({
                    "frame": { "type": "number", "minimum": 0.0 },
                    "name": { "type": "string" },
                    "color": { "type": ["string", "null"] }
                }),
                &["frame", "name"],
            ),
        ),
        CommandMeta::new(
            "timeline.update_marker",
            "Timeline/Markers/Update Marker",
            "Update an existing timeline marker's frame, name, or color",
            None,
            true,
            object_schema(
                json!({
                    "index": { "type": "integer", "minimum": 0 },
                    "frame": { "type": "number", "minimum": 0.0 },
                    "name": { "type": "string" },
                    "color": { "type": ["string", "null"] }
                }),
                &["index", "frame", "name"],
            ),
        ),
        CommandMeta::new(
            "timeline.delete_marker",
            "Timeline/Markers/Delete Marker",
            "Delete a timeline marker by index",
            None,
            true,
            object_schema(
                json!({ "index": { "type": "integer", "minimum": 0 } }),
                &["index"],
            ),
        ),
        CommandMeta::new(
            "sequence.add_control",
            "Sequence/Controls/Add Sequence Control",
            "Add an Expressions+ sequence control (Slider, Angle, Checkbox, Color, Point2D, Point3D)",
            None,
            true,
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "kind": { "type": "string" },
                    "value": { "type": "object" },
                    "min": { "type": ["number", "null"] },
                    "max": { "type": ["number", "null"] },
                    "step": { "type": ["number", "null"] }
                }),
                &["name", "kind", "value"],
            ),
        ),
        CommandMeta::new(
            "sequence.update_control",
            "Sequence/Controls/Update Sequence Control",
            "Update an Expressions+ sequence control's name, value, or range bounds",
            None,
            true,
            object_schema(
                json!({
                    "control_id": { "type": "string" },
                    "name": { "type": "string" },
                    "value": { "type": "object" },
                    "min": { "type": ["number", "null"] },
                    "max": { "type": ["number", "null"] },
                    "step": { "type": ["number", "null"] }
                }),
                &["control_id", "name", "value"],
            ),
        ),
        CommandMeta::new(
            "sequence.delete_control",
            "Sequence/Controls/Delete Sequence Control",
            "Delete an Expressions+ sequence control by ID",
            None,
            true,
            object_schema(
                json!({ "control_id": { "type": "string" } }),
                &["control_id"],
            ),
        ),
        CommandMeta::new(
            "sequence.reorder_control",
            "Sequence/Controls/Reorder Sequence Control",
            "Move an Expressions+ sequence control to a new index",
            None,
            true,
            object_schema(
                json!({
                    "control_id": { "type": "string" },
                    "to_index": { "type": "integer", "minimum": 0 }
                }),
                &["control_id", "to_index"],
            ),
        ),
        CommandMeta::new(
            "timeline.paste_clips",
            "Edit/Paste Clips",
            "Paste one or more copied ClipSnapshots starting at target_frame",
            Some("Ctrl+V"),
            true,
            object_schema(
                json!({
                    "clips": { "type": "array", "items": { "type": "object" } },
                    "target_frame": { "type": "number", "minimum": 0.0 }
                }),
                &["clips", "target_frame"],
            ),
        ),
        CommandMeta::new(
            "timeline.freeze_clip_frame",
            "Timeline/Clips/Freeze Frame",
            "Freeze a clip's media time curve at the specified timeline frame",
            None,
            true,
            object_schema(
                json!({
                    "clip_id": { "type": "string" },
                    "frame": { "type": "number" }
                }),
                &["clip_id", "frame"],
            ),
        ),
        CommandMeta::new(
            "timeline.reverse_clip",
            "Timeline/Clips/Reverse Clip",
            "Reverse a clip's media playback time curve",
            None,
            true,
            object_schema(json!({ "clip_id": { "type": "string" } }), &["clip_id"]),
        ),
        CommandMeta::new(
            "assets.relink",
            "Assets/Relink Asset",
            "Replace an asset record's filename and binary payload, clearing missing status",
            None,
            true,
            object_schema(
                json!({
                    "asset_id": { "type": "string" },
                    "filename": { "type": "string" },
                    "bytes": { "type": "array", "items": { "type": "integer" } }
                }),
                &["asset_id", "filename", "bytes"],
            ),
        ),
        CommandMeta::new(
            "assets.purge_unused",
            "Assets/Purge Unused Assets",
            "Remove all assets not referenced anywhere in the project or its compositions",
            None,
            true,
            empty_schema(),
        ),
        CommandMeta::new(
            "clipboard.paste_pz_json",
            "Edit/Paste PanZoid JSON",
            "Parse and paste PanZoid (CM3 / Zoidium) JSON or raw GLSL shader code into the active sequence",
            None,
            true,
            object_schema(
                json!({
                    "json": { "type": "string" },
                    "hint": { "type": "string" },
                    "target_clip_id": { "type": "string" },
                    "parent_object_id": { "type": "string" },
                    "target_object_id": { "type": "string" },
                    "target_property": { "type": "string" },
                    "at_frame": { "type": "number" }
                }),
                &["json"],
            ),
        ),
        CommandMeta::new(
            "clipboard.copy_pz_json",
            "Edit/Copy PanZoid JSON",
            "Serialize selected effects, 3D objects, clips, materials, properties, masks, or Shape+ contents as PanZoid CM3 JSON",
            None,
            false,
            object_schema(
                json!({
                    "kind": { "type": "string" },
                    "clip_id": { "type": "string" },
                    "target_ids": { "type": "array", "items": { "type": "string" } },
                    "property": { "type": "string" }
                }),
                &["kind"],
            ),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_contains_all_builtin_commands() {
        let reg = CommandRegistry::new_default();
        assert_eq!(reg.len(), 76);
        assert!(reg.get("project.new").is_some());
        assert!(reg.get("NewProject").is_some());
        assert!(reg.get("property.remove").is_some());
        assert!(reg.get("RemoveProperty").is_some());
        assert!(reg.get("control.inspect_state").is_some());
        assert!(reg.get("shape_plus.update_payload").is_some());
        assert!(reg.get("mask.add").is_some());
        assert!(reg.get("comp.precompose_clips").is_some());
        assert!(reg.get("assets.purge_unused").is_some());
        assert!(reg.get("clipboard.paste_pz_json").is_some());
        assert!(reg.get("clipboard.copy_pz_json").is_some());
    }

    #[test]
    fn command_round_trip_serialization() {
        let cmd = Command::SetResolution {
            width: 1920,
            height: 1080,
        };
        let json = serde_json::to_string(&cmd).expect("serialize");
        let parsed: Command = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, cmd);
        assert_eq!(parsed.id(), "project.set_resolution");
        assert!(parsed.is_undoable());
    }
}
