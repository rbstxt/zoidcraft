use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Timecode display format `0`: `HH:MM:SS.mmm` (Spec 02 §1.3).
pub const TIME_FORMAT_TIMECODE_MS: u8 = 0;

/// Timecode display format `1`: `HH:MM:SS:FF` (Spec 02 §1.3).
pub const TIME_FORMAT_TIMECODE_FRAMES: u8 = 1;

/// Timecode display format `2`: integer frame number (CM3 default, Spec 01 §1.7).
pub const TIME_FORMAT_FRAME_NUMBER: u8 = 2;

/// Selection domain within the CM3 editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionDomain {
    Clip,
    Object3D,
    Effect,
    Keyframe,
}

/// Errors produced by selection guards.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SelectionError {
    #[error(
        "cannot multi-select mixed types in {domain:?}: active type is '{expected}', clicked '{actual}'"
    )]
    MixedType {
        domain: SelectionDomain,
        expected: String,
        actual: String,
    },
}

/// Reference to an individual keyframe selected in the timeline or graph editor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyframeRef {
    pub target_id: String,
    pub property: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
    pub frame: f64,
}

impl KeyframeRef {
    /// Creates a new keyframe reference.
    #[must_use]
    pub fn new(
        target_id: impl Into<String>,
        property: impl Into<String>,
        channel: Option<String>,
        frame: f64,
    ) -> Self {
        Self {
            target_id: target_id.into(),
            property: property.into(),
            channel,
            frame,
        }
    }

    /// Returns `true` if `other_frame` matches this keyframe's frame within micro-frame tolerance.
    #[must_use]
    pub fn matches_frame(&self, other_frame: f64) -> bool {
        (self.frame - other_frame).abs() < 1e-6
    }

    fn canonical_frame_bits(&self) -> u64 {
        (self.frame + 0.0).to_bits()
    }
}

impl PartialEq for KeyframeRef {
    fn eq(&self, other: &Self) -> bool {
        self.target_id == other.target_id
            && self.property == other.property
            && self.channel == other.channel
            && self.canonical_frame_bits() == other.canonical_frame_bits()
    }
}

impl Eq for KeyframeRef {}

/// Active CM3 editor selection state with Spec 01 §4.7 same-type multi-selection guard.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectionState {
    pub clip_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_base_type: Option<String>,
    pub object_3d_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_3d_base_type: Option<String>,
    pub effect_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect_base_type: Option<String>,
    pub keyframes: Vec<KeyframeRef>,
}

impl SelectionState {
    /// Selects or toggles a timeline clip.
    ///
    /// When `shift` is `false`, clears prior clip selection and selects `clip_id`.
    /// When `shift` is `true`, toggles membership while enforcing the same-type guard
    /// if `base_type` is supplied.
    ///
    /// Returns `Ok(true)` if the item is now selected, `Ok(false)` if toggled off.
    pub fn select_clip(
        &mut self,
        clip_id: impl Into<String>,
        base_type: Option<&str>,
        shift: bool,
    ) -> Result<bool, SelectionError> {
        let clip_id = clip_id.into();
        if !shift {
            self.clip_ids.clear();
            self.clip_ids.push(clip_id);
            self.clip_base_type = base_type.map(str::to_string);
            return Ok(true);
        }

        if let Some(pos) = self.clip_ids.iter().position(|id| id == &clip_id) {
            self.clip_ids.remove(pos);
            if self.clip_ids.is_empty() {
                self.clip_base_type = None;
            }
            return Ok(false);
        }

        if let (Some(expected), Some(actual)) = (self.clip_base_type.as_deref(), base_type)
            && expected != actual
        {
            return Err(SelectionError::MixedType {
                domain: SelectionDomain::Clip,
                expected: expected.to_string(),
                actual: actual.to_string(),
            });
        }

        if self.clip_base_type.is_none() {
            self.clip_base_type = base_type.map(str::to_string);
        }
        self.clip_ids.push(clip_id);
        Ok(true)
    }

    /// Selects or toggles a 3D object with strict Spec 01 §4.7 `baseTypeString` multi-select guard.
    ///
    /// Plain click (`shift == false`) replaces the selection cleanly with any type.
    /// `Shift+Click` (`shift == true`) rejects mixing different `base_type` values without
    /// altering the existing selection.
    pub fn select_object_3d(
        &mut self,
        object_id: impl Into<String>,
        base_type: impl Into<String>,
        shift: bool,
    ) -> Result<bool, SelectionError> {
        let object_id = object_id.into();
        let base_type = base_type.into();

        if !shift {
            self.object_3d_ids.clear();
            self.object_3d_ids.push(object_id);
            self.object_3d_base_type = Some(base_type);
            return Ok(true);
        }

        if let Some(pos) = self.object_3d_ids.iter().position(|id| id == &object_id) {
            self.object_3d_ids.remove(pos);
            if self.object_3d_ids.is_empty() {
                self.object_3d_base_type = None;
            }
            return Ok(false);
        }

        if let Some(expected) = self.object_3d_base_type.as_deref()
            && expected != base_type
        {
            return Err(SelectionError::MixedType {
                domain: SelectionDomain::Object3D,
                expected: expected.to_string(),
                actual: base_type,
            });
        }

        self.object_3d_base_type = Some(base_type);
        self.object_3d_ids.push(object_id);
        Ok(true)
    }

    /// Selects or toggles an effect with strict Spec 01 §4.7 `baseTypeString` multi-select guard.
    pub fn select_effect(
        &mut self,
        effect_id: impl Into<String>,
        base_type: impl Into<String>,
        shift: bool,
    ) -> Result<bool, SelectionError> {
        let effect_id = effect_id.into();
        let base_type = base_type.into();

        if !shift {
            self.effect_ids.clear();
            self.effect_ids.push(effect_id);
            self.effect_base_type = Some(base_type);
            return Ok(true);
        }

        if let Some(pos) = self.effect_ids.iter().position(|id| id == &effect_id) {
            self.effect_ids.remove(pos);
            if self.effect_ids.is_empty() {
                self.effect_base_type = None;
            }
            return Ok(false);
        }

        if let Some(expected) = self.effect_base_type.as_deref()
            && expected != base_type
        {
            return Err(SelectionError::MixedType {
                domain: SelectionDomain::Effect,
                expected: expected.to_string(),
                actual: base_type,
            });
        }

        self.effect_base_type = Some(base_type);
        self.effect_ids.push(effect_id);
        Ok(true)
    }

    /// Selects or toggles a keyframe reference.
    pub fn select_keyframe(&mut self, kf: KeyframeRef, shift: bool) -> bool {
        if !shift {
            self.keyframes.clear();
            self.keyframes.push(kf);
            return true;
        }

        if let Some(pos) = self.keyframes.iter().position(|existing| {
            existing.target_id == kf.target_id
                && existing.property == kf.property
                && existing.channel == kf.channel
                && existing.matches_frame(kf.frame)
        }) {
            self.keyframes.remove(pos);
            return false;
        }

        self.keyframes.push(kf);
        true
    }

    /// Clears selected clips.
    pub fn clear_clips(&mut self) {
        self.clip_ids.clear();
        self.clip_base_type = None;
    }

    /// Clears selected 3D objects.
    pub fn clear_objects_3d(&mut self) {
        self.object_3d_ids.clear();
        self.object_3d_base_type = None;
    }

    /// Clears selected effects.
    pub fn clear_effects(&mut self) {
        self.effect_ids.clear();
        self.effect_base_type = None;
    }

    /// Clears selected keyframes.
    pub fn clear_keyframes(&mut self) {
        self.keyframes.clear();
    }

    /// Clears all selections across every domain.
    pub fn clear_all(&mut self) {
        self.clear_clips();
        self.clear_objects_3d();
        self.clear_effects();
        self.clear_keyframes();
    }

    /// Removes a deleted clip ID from selection.
    pub fn prune_clip(&mut self, clip_id: &str) {
        self.clip_ids.retain(|id| id != clip_id);
        if self.clip_ids.is_empty() {
            self.clip_base_type = None;
        }
        self.prune_keyframes_for_target(clip_id);
    }

    /// Removes a deleted 3D object ID from selection.
    pub fn prune_object_3d(&mut self, object_id: &str) {
        self.object_3d_ids.retain(|id| id != object_id);
        if self.object_3d_ids.is_empty() {
            self.object_3d_base_type = None;
        }
        self.prune_keyframes_for_target(object_id);
    }

    /// Removes a deleted effect ID from selection.
    pub fn prune_effect(&mut self, effect_id: &str) {
        self.effect_ids.retain(|id| id != effect_id);
        if self.effect_ids.is_empty() {
            self.effect_base_type = None;
        }
        self.prune_keyframes_for_target(effect_id);
    }

    /// Removes all selected keyframes belonging to `target_id`.
    pub fn prune_keyframes_for_target(&mut self, target_id: &str) {
        self.keyframes.retain(|kf| kf.target_id != target_id);
    }

    /// Returns `true` when CM3 Editing Camera mode should engage (Spec 01 §5.1):
    /// `editing_camera == true` and exactly one 3D Scene clip is selected.
    #[must_use]
    pub fn is_editing_camera_active(
        &self,
        playback: &PlaybackState,
        selected_clip_is_scene: bool,
    ) -> bool {
        playback.editing_camera && self.clip_ids.len() == 1 && selected_clip_is_scene
    }
}

/// CM3 playback transport and viewport toggle state (Spec 01 §1.6 & Spec 02 §5.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlaybackState {
    pub current_frame: f64,
    pub speed: f64,
    pub looping: bool,
    pub time_format: u8,
    pub editing_camera: bool,
    pub transform_controls_2d: bool,
    /// Monotonic audio/video synchronization epoch (Spec 02 §5.4).
    #[serde(default)]
    pub epoch_id: u64,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            current_frame: 0.0,
            speed: 0.0,
            looping: true,
            time_format: TIME_FORMAT_FRAME_NUMBER,
            editing_camera: true,
            transform_controls_2d: false,
            epoch_id: 0,
        }
    }
}

impl PlaybackState {
    /// Returns `true` when playback is actively advancing (`speed != 0`).
    #[must_use]
    pub fn is_playing(&self) -> bool {
        self.speed.abs() > f64::EPSILON
    }

    /// Integer timeline frame passed to sequence evaluation and compositor (`round(exactFrame)`).
    #[must_use]
    pub fn current_frame_index(&self) -> i64 {
        self.current_frame.round() as i64
    }

    /// Increments the playback synchronization epoch so audio/video decoders resync.
    pub fn bump_epoch(&mut self) {
        self.epoch_id = self.epoch_id.saturating_add(1);
    }

    /// Scrubs directly to `frame`, pausing playback (`speed = 0`) and clamping to
    /// `[0, max(total_frames - 1, 0)]` per Spec 02 §5.1.
    pub fn set_frame(&mut self, frame: f64, total_frames: f64) {
        let max_frame = (total_frames - 1.0).max(0.0);
        self.speed = 0.0;
        self.current_frame = if frame.is_finite() {
            frame.clamp(0.0, max_frame)
        } else {
            0.0
        };
        self.bump_epoch();
    }

    /// Pauses playback and steps `current_frame` by `delta` frames.
    pub fn step_frame(&mut self, delta: f64, total_frames: f64) {
        let target = self.current_frame + delta;
        self.set_frame(target, total_frames);
    }

    /// Toggles playback speed between `0.0` (paused) and `1.0` (normal forward).
    pub fn toggle_play(&mut self) {
        if self.is_playing() {
            self.speed = 0.0;
        } else {
            self.speed = 1.0;
        }
        self.bump_epoch();
    }

    /// Sets playback speed multiplier (`0` = paused, `1` = 1x forward, `-1` = 1x reverse).
    pub fn set_speed(&mut self, speed: f64) {
        if speed.is_finite() {
            self.speed = speed;
            self.bump_epoch();
        }
    }

    /// Toggles looping mode.
    pub fn toggle_loop(&mut self) {
        self.looping = !self.looping;
    }

    /// Cycles timecode display format modulo 3 (`HH:MM:SS.mmm` -> `HH:MM:SS:FF` -> `Frame`).
    pub fn cycle_time_format(&mut self) {
        self.time_format = (self.time_format + 1) % 3;
    }

    /// Sets explicit timecode format clamped modulo 3.
    pub fn set_time_format(&mut self, format: u8) {
        self.time_format = format % 3;
    }

    /// Toggles 3D Editing Camera mode (`C`).
    pub fn toggle_editing_camera(&mut self) {
        self.editing_camera = !self.editing_camera;
    }

    /// Toggles 2D Layer Transform Controls overlay (`T`).
    pub fn toggle_transform_controls_2d(&mut self) {
        self.transform_controls_2d = !self.transform_controls_2d;
    }

    /// Advances playback by `dt_seconds` using exact CM3 boundary and loop semantics (Spec 02 §5.1).
    pub fn advance(&mut self, dt_seconds: f64, fps: f64, total_frames: f64) {
        if !self.is_playing() || !dt_seconds.is_finite() || dt_seconds <= 0.0 || fps <= 0.0 {
            return;
        }

        let next = self.current_frame + dt_seconds * fps * self.speed;
        if next >= total_frames || next < 0.0 {
            if self.looping && total_frames > 0.0 {
                self.current_frame = next.rem_euclid(total_frames);
            } else {
                let max_frame = (total_frames - 1.0).max(0.0);
                self.current_frame = next.clamp(0.0, max_frame);
                self.speed = 0.0;
            }
            self.bump_epoch();
        } else {
            self.current_frame = next;
        }
    }

    /// Formats `current_frame` according to `self.time_format` and `fps` (Spec 02 §1.3).
    #[must_use]
    pub fn format_timecode(&self, fps: f64) -> String {
        let safe_fps = if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            30.0
        };
        let frame = self.current_frame.max(0.0);

        match self.time_format % 3 {
            TIME_FORMAT_TIMECODE_MS => {
                let t = frame / safe_fps;
                let hours = (t / 3600.0).floor() as u64;
                let minutes = ((t / 60.0) % 60.0).floor() as u64;
                let seconds = t % 60.0;
                format!("{hours:02}:{minutes:02}:{seconds:06.3}")
            }
            TIME_FORMAT_TIMECODE_FRAMES => {
                let fps_int = safe_fps.round().max(1.0) as u64;
                let total_frames = frame.round() as u64;
                let ff = total_frames % fps_int;
                let total_secs = total_frames / fps_int;
                let ss = total_secs % 60;
                let mm = (total_secs / 60) % 60;
                let hh = total_secs / 3600;
                format!("{hh:02}:{mm:02}:{ss:02}:{ff:02}")
            }
            _ => format!("{}", frame.round() as i64),
        }
    }
}
