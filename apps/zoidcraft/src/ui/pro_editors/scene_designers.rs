//! Floating 3D Scene Designer & Options windows matching Zoidium:
//! - **Lens Flares Options** (`LensFlareOptionsState`, [`show_lens_flare_options_window`]):
//!   Browser (Flare / Elements / Presets with Cinematic, Rainbow & Gold categories + favorites),
//!   procedural vector thumbnails, live optical parameter editor, and atomic Reset All.
//! - **Particle & Light Designer** (`ParticleDesignerState`, [`show_particle_designer_window`]):
//!   3-column designer for 3D Particle Emitter (Particular), Particle Lattice (Form), and
//!   Point Cloud Network (Plexus) with Presets + Color Palettes, Transport + Block Tiles, and
//!   selected Block Parameter inspector.

use std::collections::BTreeSet;

use crate::ui::controls::{
    ButtonVariant, FloatingWindowManager, FloatingWindowSpec, ScrubConfig, cm3_floating_window,
    cm3_hsva_color_picker, cm3_prop_button, cm3_scrub_number, request_eyedropper_sampler,
};
use crate::ui::theme::{
    BG_BUTTON, BG_HOVER, BG_INPUT, BG_PANEL, BG_SUNKEN, BG_TITLE, BORDER_CONTROL, BORDER_PANEL,
    Cm3Theme, FONT_BODY, FONT_COMPACT, FONT_SMALL, FONT_TITLE, TEXT_BRIGHT, TEXT_FAINT, TEXT_MUTED,
    TEXT_PRIMARY, TEXT_STRONG, color32_with_alpha,
};
use egui::{Color32, CornerRadius, Id, Key, Pos2, Rect, ScrollArea, Sense, Stroke, Ui, Vec2};
use serde_json::{Map, Value, json};
use zc_color::{Hsv, Rgb, Rgba};
use zc_session::{Command, Session, SessionObject3D, SessionProperty};

use super::*;

// ===========================================================================
// Shared Object3D Lookup & Property Read/Write Helpers
// ===========================================================================

fn find_object3d_in_session<'a>(
    session: &'a Session,
    object_id: &str,
) -> Option<(&'a SessionObject3D, f64)> {
    for track in &session.document().video_tracks {
        for clip in &track.clips {
            let local_frame = session.playback.current_frame - clip.start;
            for obj in &clip.objects_3d {
                if obj.id == object_id {
                    return Some((obj, local_frame));
                }
            }
        }
    }
    None
}

fn read_obj_prop(prop: Option<&SessionProperty>, local_frame: f64) -> Option<Value> {
    let p = prop?;
    if p.keyframes.is_empty() {
        return Some(p.default_value.clone());
    }
    if !p.animated {
        return Some(p.keyframes[0].value.clone());
    }
    let mut best = &p.keyframes[0];
    for kf in &p.keyframes {
        if kf.frame <= local_frame + 1e-6 {
            best = kf;
        } else {
            break;
        }
    }
    Some(best.value.clone())
}

fn read_obj_f64(obj: &SessionObject3D, key: &str, default: f64, local_frame: f64) -> f64 {
    read_obj_prop(obj.properties.get(key), local_frame)
        .and_then(|v| {
            v.as_f64()
                .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
        })
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

fn read_obj_color(
    obj: &SessionObject3D,
    key: &str,
    default: [f64; 3],
    local_frame: f64,
) -> [f64; 3] {
    read_obj_prop(obj.properties.get(key), local_frame)
        .and_then(|v| {
            let arr = v.as_array()?;
            if arr.len() >= 3 {
                Some([
                    arr[0].as_f64().unwrap_or(default[0]),
                    arr[1].as_f64().unwrap_or(default[1]),
                    arr[2].as_f64().unwrap_or(default[2]),
                ])
            } else {
                None
            }
        })
        .unwrap_or(default)
}

fn write_obj_prop(
    session: &mut Session,
    object_id: &str,
    key: &str,
    value: Value,
    local_frame: f64,
) {
    let animated = find_object3d_in_session(session, object_id)
        .and_then(|(obj, _)| obj.properties.get(key))
        .is_some_and(|p| p.animated);
    if animated {
        let _ = session.execute(Command::SetKeyframe {
            target_id: object_id.to_owned(),
            property: key.to_owned(),
            frame: local_frame,
            value,
            tween: None,
        });
    } else {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: object_id.to_owned(),
            property: key.to_owned(),
            value,
        });
    }
}

pub fn apply_obj_batch_tx(
    session: &mut Session,
    object_id: &str,
    label: &str,
    assignments: &[(&str, Value)],
    local_frame: f64,
) {
    if assignments.is_empty() {
        return;
    }
    let opened = session.begin_transaction(label.to_owned()).is_ok();
    for (key, val) in assignments {
        write_obj_prop(session, object_id, key, val.clone(), local_frame);
    }
    if opened && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    }
}

// ===========================================================================
// 1. 3D Lens Flares Options Window (`plugins/optical-flares/optflares-window.js`)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LensFlareBrowserTab {
    Flare,
    Elements,
    Presets,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LensFlarePresetCategory {
    Cinematic,
    Rainbow,
    Gold,
}

impl LensFlarePresetCategory {
    pub const ALL: [Self; 3] = [Self::Cinematic, Self::Rainbow, Self::Gold];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cinematic => "Cinematic",
            Self::Rainbow => "Rainbow (8)",
            Self::Gold => "Gold (8)",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LensFlareOptionsState {
    pub open: bool,
    pub object_id: String,
    pub object_label: String,
    pub browser_tab: LensFlareBrowserTab,
    pub preset_category: LensFlarePresetCategory,
    pub fav_only: bool,
    pub favorites: BTreeSet<String>,
    pub selected_element: Option<usize>,
    pub just_opened: bool,
}

impl Default for LensFlareOptionsState {
    fn default() -> Self {
        Self {
            open: false,
            object_id: String::new(),
            object_label: String::new(),
            browser_tab: LensFlareBrowserTab::Flare,
            preset_category: LensFlarePresetCategory::Cinematic,
            fav_only: false,
            favorites: BTreeSet::new(),
            selected_element: Some(0),
            just_opened: false,
        }
    }
}

impl LensFlareOptionsState {
    pub fn open_for_object(
        &mut self,
        object_id: impl Into<String>,
        object_label: impl Into<String>,
    ) {
        self.object_id = object_id.into();
        self.object_label = object_label.into();
        self.selected_element = Some(0);
        self.open = true;
        self.just_opened = true;
    }
}

#[derive(Clone, Copy)]
pub struct LensFlarePreset {
    pub id: &'static str,
    pub name: &'static str,
    pub category: LensFlarePresetCategory,
    pub thumb_kind: u8,
    pub color: [f64; 3],
    pub brightness: f64,
    pub size: f64,
    pub anamorphic: f64,
    pub ghosts: u32,
    pub starburst_rays: u32,
    pub ring_intensity: f64,
    pub angle: f64,
}

pub const LENS_FLARE_BASE_PRESETS: &[LensFlarePreset] = &[
    LensFlarePreset {
        id: "default",
        name: "Default Flare",
        category: LensFlarePresetCategory::Cinematic,
        thumb_kind: 0,
        color: [1.0, 0.92, 0.8],
        brightness: 1.0,
        size: 1.0,
        anamorphic: 0.5,
        ghosts: 4,
        starburst_rays: 8,
        ring_intensity: 0.4,
        angle: 0.0,
    },
    LensFlarePreset {
        id: "anamorphic",
        name: "Anamorphic Blue",
        category: LensFlarePresetCategory::Cinematic,
        thumb_kind: 1,
        color: [0.35, 0.72, 1.0],
        brightness: 1.4,
        size: 1.25,
        anamorphic: 1.65,
        ghosts: 6,
        starburst_rays: 4,
        ring_intensity: 0.3,
        angle: 0.0,
    },
    LensFlarePreset {
        id: "scifi",
        name: "Sci-Fi Core",
        category: LensFlarePresetCategory::Cinematic,
        thumb_kind: 2,
        color: [0.25, 0.95, 0.9],
        brightness: 1.55,
        size: 1.1,
        anamorphic: 1.2,
        ghosts: 8,
        starburst_rays: 12,
        ring_intensity: 0.9,
        angle: 15.0,
    },
    LensFlarePreset {
        id: "sparkle",
        name: "Sparkle Star",
        category: LensFlarePresetCategory::Cinematic,
        thumb_kind: 3,
        color: [1.0, 0.96, 0.86],
        brightness: 1.35,
        size: 0.95,
        anamorphic: 0.2,
        ghosts: 3,
        starburst_rays: 16,
        ring_intensity: 0.25,
        angle: 22.5,
    },
    LensFlarePreset {
        id: "sunset",
        name: "Golden Hour",
        category: LensFlarePresetCategory::Gold,
        thumb_kind: 0,
        color: [1.0, 0.65, 0.22],
        brightness: 1.6,
        size: 1.45,
        anamorphic: 0.75,
        ghosts: 6,
        starburst_rays: 6,
        ring_intensity: 0.65,
        angle: -8.0,
    },
    LensFlarePreset {
        id: "prism",
        name: "Prismatic Halo",
        category: LensFlarePresetCategory::Rainbow,
        thumb_kind: 2,
        color: [0.88, 0.55, 1.0],
        brightness: 1.3,
        size: 1.2,
        anamorphic: 0.9,
        ghosts: 8,
        starburst_rays: 10,
        ring_intensity: 1.15,
        angle: 12.0,
    },
];

/// Metadata for each of the 12 Optical Flare element types matching `zc-render`'s
/// `summarize_optical_flare_stack` element type IDs (`0..=11`).
#[derive(Debug, Clone, Copy)]
pub struct FlareElementTypeDef {
    pub type_id: u32,
    pub name: &'static str,
    pub thumb_kind: u8,
    pub default_color: [f64; 3],
    pub default_scale: f64,
    pub default_position: f64,
    pub default_aspect: f64,
}

pub const FLARE_ELEMENT_TYPES: &[FlareElementTypeDef] = &[
    FlareElementTypeDef {
        type_id: 0,
        name: "Glow",
        thumb_kind: 0,
        default_color: [1.0, 0.94, 0.84],
        default_scale: 110.0,
        default_position: 0.0,
        default_aspect: 1.0,
    },
    FlareElementTypeDef {
        type_id: 2,
        name: "Iris",
        thumb_kind: 2,
        default_color: [0.55, 0.85, 1.0],
        default_scale: 85.0,
        default_position: 0.45,
        default_aspect: 1.0,
    },
    FlareElementTypeDef {
        type_id: 1,
        name: "Multi Iris",
        thumb_kind: 2,
        default_color: [0.45, 0.9, 0.85],
        default_scale: 95.0,
        default_position: 0.65,
        default_aspect: 1.0,
    },
    FlareElementTypeDef {
        type_id: 3,
        name: "Streak",
        thumb_kind: 1,
        default_color: [0.35, 0.75, 1.0],
        default_scale: 120.0,
        default_position: 0.0,
        default_aspect: 10.0,
    },
    FlareElementTypeDef {
        type_id: 4,
        name: "Shimmer",
        thumb_kind: 3,
        default_color: [1.0, 0.92, 0.75],
        default_scale: 90.0,
        default_position: 0.0,
        default_aspect: 1.0,
    },
    FlareElementTypeDef {
        type_id: 5,
        name: "Glint",
        thumb_kind: 3,
        default_color: [0.6, 0.92, 1.0],
        default_scale: 80.0,
        default_position: 0.0,
        default_aspect: 1.0,
    },
    FlareElementTypeDef {
        type_id: 6,
        name: "Spike Ball",
        thumb_kind: 3,
        default_color: [1.0, 0.72, 0.35],
        default_scale: 95.0,
        default_position: 0.0,
        default_aspect: 1.0,
    },
    FlareElementTypeDef {
        type_id: 7,
        name: "Sparkle",
        thumb_kind: 3,
        default_color: [1.0, 0.96, 0.88],
        default_scale: 85.0,
        default_position: 0.15,
        default_aspect: 1.0,
    },
    FlareElementTypeDef {
        type_id: 8,
        name: "Ring",
        thumb_kind: 2,
        default_color: [1.0, 0.78, 0.42],
        default_scale: 105.0,
        default_position: 0.0,
        default_aspect: 1.0,
    },
    FlareElementTypeDef {
        type_id: 9,
        name: "Hoop",
        thumb_kind: 2,
        default_color: [0.78, 0.58, 1.0],
        default_scale: 115.0,
        default_position: -0.25,
        default_aspect: 1.0,
    },
    FlareElementTypeDef {
        type_id: 11,
        name: "Lens Orbs",
        thumb_kind: 2,
        default_color: [0.75, 0.88, 1.0],
        default_scale: 75.0,
        default_position: -0.55,
        default_aspect: 1.0,
    },
    FlareElementTypeDef {
        type_id: 10,
        name: "Caustic",
        thumb_kind: 2,
        default_color: [0.4, 0.95, 0.9],
        default_scale: 90.0,
        default_position: 0.85,
        default_aspect: 1.0,
    },
];

#[must_use]
pub fn flare_element_type_def(type_id: u32) -> &'static FlareElementTypeDef {
    FLARE_ELEMENT_TYPES
        .iter()
        .find(|d| d.type_id == type_id)
        .unwrap_or(&FLARE_ELEMENT_TYPES[0])
}

/// One editable optical element inside `flareSetup.stack`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpticalFlareElement {
    pub id: String,
    pub name: String,
    pub element_type: u32,
    pub enabled: bool,
    pub solo: bool,
    /// Brightness / opacity in percent (`0.0..=200.0`, default `100.0`).
    pub brightness: f64,
    /// Element scale in percent (`5.0..=400.0`, default `100.0`).
    pub scale: f64,
    /// Position along the flare optical axis (`-2.0..=2.0`, `0.0` = light source).
    pub position: f64,
    /// Aspect ratio (used by anamorphic streaks, default `1.0`).
    pub aspect_ratio: f64,
    /// Element tint RGB in `0.0..=1.0`.
    pub color: [f64; 3],
}

impl OpticalFlareElement {
    #[must_use]
    pub fn from_type(type_id: u32, serial: usize) -> Self {
        let def = flare_element_type_def(type_id);
        Self {
            id: format!("el_{serial}"),
            name: def.name.to_owned(),
            element_type: def.type_id,
            enabled: true,
            solo: false,
            brightness: 100.0,
            scale: def.default_scale,
            position: def.default_position,
            aspect_ratio: def.default_aspect,
            color: def.default_color,
        }
    }

    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "name": self.name,
            "type": self.element_type,
            "properties": {
                "element": {
                    "enabled": if self.enabled { 1 } else { 0 },
                    "solo": if self.solo { 1 } else { 0 },
                    "opacity": self.brightness,
                    "elementType": self.element_type,
                },
                "globalParams": {
                    "scale": self.scale,
                    "distance": self.position,
                    "position": self.position,
                    "aspectRatio": self.aspect_ratio,
                    "color": self.color,
                }
            }
        })
    }

    #[must_use]
    pub fn from_json(val: &Value, idx: usize) -> Option<Self> {
        let obj = val.as_object()?;
        let el_props = obj.get("properties").and_then(|p| p.get("element"));
        let gp_props = obj.get("properties").and_then(|p| p.get("globalParams"));

        let read_num = |group: Option<&Value>, key: &str, def: f64| -> f64 {
            group
                .and_then(|g| g.get(key))
                .and_then(|v| {
                    v.as_f64()
                        .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                        .or_else(|| v.get("value").and_then(Value::as_f64))
                })
                .filter(|n| n.is_finite())
                .unwrap_or(def)
        };

        let element_type = obj
            .get("type")
            .and_then(Value::as_u64)
            .map(|v| v as u32)
            .unwrap_or_else(|| read_num(el_props, "elementType", 0.0).round().max(0.0) as u32)
            .min(11);
        let def = flare_element_type_def(element_type);

        let id = obj
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("el_{}", idx + 1));
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| def.name.to_owned());
        let enabled = read_num(el_props, "enabled", 1.0) >= 0.5;
        let solo = read_num(el_props, "solo", 0.0) >= 0.5;
        let brightness = read_num(el_props, "opacity", 100.0).clamp(0.0, 200.0);
        let scale = read_num(gp_props, "scale", def.default_scale).clamp(5.0, 400.0);
        let position = gp_props
            .and_then(|g| g.get("position").or_else(|| g.get("distance")))
            .and_then(Value::as_f64)
            .filter(|n| n.is_finite())
            .unwrap_or(def.default_position)
            .clamp(-2.0, 2.0);
        let aspect_ratio = read_num(gp_props, "aspectRatio", def.default_aspect).clamp(0.1, 20.0);
        let color = gp_props
            .and_then(|g| g.get("color"))
            .and_then(Value::as_array)
            .filter(|a| a.len() >= 3)
            .map(|a| {
                [
                    a[0].as_f64()
                        .unwrap_or(def.default_color[0])
                        .clamp(0.0, 1.0),
                    a[1].as_f64()
                        .unwrap_or(def.default_color[1])
                        .clamp(0.0, 1.0),
                    a[2].as_f64()
                        .unwrap_or(def.default_color[2])
                        .clamp(0.0, 1.0),
                ]
            })
            .unwrap_or(def.default_color);

        Some(Self {
            id,
            name,
            element_type,
            enabled,
            solo,
            brightness,
            scale,
            position,
            aspect_ratio,
            color,
        })
    }
}

/// Summary optical uniforms derived from an [`OpticalFlareElement`] stack so the GPU flare
/// renderer reflects every stack edit immediately.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlareStackSummary {
    pub brightness: f64,
    pub scale: f64,
    pub anamorphic: f64,
    pub ghosts: u32,
    pub starburst_rays: u32,
    pub ring_intensity: f64,
    pub color: [f64; 3],
}

#[must_use]
pub fn summarize_flare_stack_uniforms(stack: &[OpticalFlareElement]) -> FlareStackSummary {
    let any_solo = stack.iter().any(|e| e.solo);
    let mut streak_acc = 0.0_f64;
    let mut ring_acc = 0.0_f64;
    let mut ghost_acc = 0_u32;
    let mut ray_acc = 0_u32;
    let mut col_sum = [0.0_f64; 3];
    let mut col_weight = 0.0_f64;
    let mut bright_acc = 0.0_f64;
    let mut scale_sum = 0.0_f64;
    let mut active_count = 0_u32;

    for el in stack {
        if !el.enabled || (any_solo && !el.solo) {
            continue;
        }
        let op = (el.brightness / 100.0).clamp(0.0, 2.0);
        let sc = (el.scale / 100.0).clamp(0.05, 4.0);
        if op <= 1e-3 {
            continue;
        }
        active_count = active_count.saturating_add(1);
        let axis_attenuation = 1.0 / (1.0 + el.position.abs() * 0.2);
        let w = op * sc.clamp(0.25, 2.0) * axis_attenuation;
        col_sum[0] += el.color[0] * w;
        col_sum[1] += el.color[1] * w;
        col_sum[2] += el.color[2] * w;
        col_weight += w;
        bright_acc += op * (if el.element_type == 0 { 0.75 } else { 0.25 });
        scale_sum += sc;

        match el.element_type {
            1 => ghost_acc = ghost_acc.saturating_add(2),
            2 | 10 | 11 => ghost_acc = ghost_acc.saturating_add(1),
            3 => streak_acc += op * (el.aspect_ratio / 6.0).clamp(0.35, 2.2) * sc.clamp(0.4, 1.8),
            4 => ray_acc = ray_acc.saturating_add(6),
            5 => ray_acc = ray_acc.saturating_add(4),
            6 => ray_acc = ray_acc.saturating_add(8),
            7 => ray_acc = ray_acc.saturating_add(14),
            8 => ring_acc += op * 0.55 * sc.clamp(0.4, 1.6),
            9 => ring_acc += op * 0.75 * sc.clamp(0.4, 1.6),
            _ => {}
        }
    }

    let color = if col_weight > 1e-4 {
        [
            (col_sum[0] / col_weight).clamp(0.0, 1.0),
            (col_sum[1] / col_weight).clamp(0.0, 1.0),
            (col_sum[2] / col_weight).clamp(0.0, 1.0),
        ]
    } else {
        [1.0, 0.92, 0.8]
    };

    let brightness = if active_count == 0 {
        0.0
    } else {
        bright_acc.clamp(0.1, 10.0)
    };
    let scale = if active_count == 0 {
        1.0
    } else {
        (scale_sum / f64::from(active_count)).clamp(0.05, 8.0)
    };

    FlareStackSummary {
        brightness,
        scale,
        anamorphic: streak_acc.clamp(0.0, 4.0),
        ghosts: ghost_acc.min(8),
        starburst_rays: ray_acc.min(24),
        ring_intensity: ring_acc.clamp(0.0, 2.0),
        color,
    }
}

#[must_use]
pub fn default_flare_stack_from_preset(preset: &LensFlarePreset) -> Vec<OpticalFlareElement> {
    let mut stack = Vec::new();
    let mut glow = OpticalFlareElement::from_type(0, 1);
    glow.color = preset.color;
    glow.brightness = (preset.brightness * 90.0).clamp(20.0, 200.0);
    glow.scale = (preset.size * 100.0).clamp(20.0, 300.0);
    stack.push(glow);

    if preset.anamorphic > 0.05 {
        let mut streak = OpticalFlareElement::from_type(3, 2);
        streak.color = preset.color;
        streak.brightness = (preset.anamorphic * 70.0).clamp(15.0, 200.0);
        streak.aspect_ratio = (preset.anamorphic * 6.5).clamp(3.0, 18.0);
        stack.push(streak);
    }
    if preset.ghosts > 0 {
        let mut iris = OpticalFlareElement::from_type(1, 3);
        iris.color = preset.color;
        iris.position = 0.55;
        stack.push(iris);
    }
    if preset.starburst_rays > 0 {
        let mut shimmer = OpticalFlareElement::from_type(4, 4);
        shimmer.color = preset.color;
        stack.push(shimmer);
    }
    if preset.ring_intensity > 0.05 {
        let mut ring = OpticalFlareElement::from_type(8, 5);
        ring.color = preset.color;
        ring.brightness = (preset.ring_intensity * 100.0).clamp(20.0, 200.0);
        stack.push(ring);
    }
    stack
}

#[must_use]
pub fn read_optical_flare_stack(
    obj: &SessionObject3D,
    local_frame: f64,
) -> Vec<OpticalFlareElement> {
    if let Some(setup) = read_obj_prop(obj.properties.get("flareSetup"), local_frame) {
        let arr_opt = setup
            .get("stack")
            .and_then(Value::as_array)
            .or_else(|| setup.as_array());
        if let Some(arr) = arr_opt {
            let parsed: Vec<OpticalFlareElement> = arr
                .iter()
                .enumerate()
                .filter_map(|(i, v)| OpticalFlareElement::from_json(v, i))
                .collect();
            if !parsed.is_empty() {
                return parsed;
            }
        }
    }
    // Synthesize a clean starter stack from the object's live optical properties.
    let color = read_obj_color(obj, "flareColor", [1.0, 0.92, 0.8], local_frame);
    let brightness = read_obj_f64(obj, "brightness", 1.0, local_frame);
    let size = read_obj_f64(obj, "size", 1.0, local_frame);
    let anamorphic = read_obj_f64(obj, "anamorphic", 0.5, local_frame);
    let ghosts = read_obj_f64(obj, "ghosts", 4.0, local_frame)
        .round()
        .max(0.0) as u32;
    let starburst_rays = read_obj_f64(obj, "starburstRays", 8.0, local_frame)
        .round()
        .max(0.0) as u32;
    let ring_intensity = read_obj_f64(obj, "ringIntensity", 0.4, local_frame);
    default_flare_stack_from_preset(&LensFlarePreset {
        id: "current",
        name: "Current",
        category: LensFlarePresetCategory::Cinematic,
        thumb_kind: 0,
        color,
        brightness,
        size,
        anamorphic,
        ghosts,
        starburst_rays,
        ring_intensity,
        angle: 0.0,
    })
}

/// Persists `flareSetup.stack` AND automatically re-derives and syncs all summary optical
/// uniform properties (`brightness`, `scale`, `size`, `anamorphic`, `ghosts`, `starburstRays`,
/// `ringIntensity`, `color`, `flareColor`) in a single atomic Undo transaction.
pub fn commit_optical_flare_stack(
    session: &mut Session,
    obj: &SessionObject3D,
    stack: &[OpticalFlareElement],
    label: &str,
    local_frame: f64,
) {
    let mut setup_map = read_obj_prop(obj.properties.get("flareSetup"), local_frame)
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default();
    setup_map.insert(
        "stack".to_owned(),
        Value::Array(stack.iter().map(OpticalFlareElement::to_json).collect()),
    );
    let summary = summarize_flare_stack_uniforms(stack);
    let batch = [
        ("flareSetup", Value::Object(setup_map)),
        ("brightness", json!(summary.brightness)),
        ("scale", json!(summary.scale)),
        ("size", json!(summary.scale)),
        ("anamorphic", json!(summary.anamorphic)),
        ("ghosts", json!(summary.ghosts)),
        ("starburstRays", json!(summary.starburst_rays)),
        ("ringIntensity", json!(summary.ring_intensity)),
        ("color", json!(summary.color)),
        ("flareColor", json!(summary.color)),
    ];
    apply_obj_batch_tx(session, &obj.id, label, &batch, local_frame);
}

pub type LensFlareElementSpec = (&'static str, u8, [f64; 3], f64, u32, u32, f64);

pub const LENS_FLARE_ELEMENTS: &[LensFlareElementSpec] = &[
    ("Glow Core", 0, [1.0, 0.95, 0.88], 0.15, 2, 0, 0.1),
    ("Anamorphic Streak", 1, [0.35, 0.75, 1.0], 1.85, 3, 2, 0.2),
    ("Chromatic Hoop", 2, [0.75, 0.55, 1.0], 0.3, 4, 0, 1.35),
    ("Starburst Shimmer", 3, [1.0, 0.92, 0.75], 0.25, 2, 16, 0.2),
    ("Multi-Iris Ghosts", 2, [0.45, 0.9, 0.85], 0.4, 8, 4, 0.55),
    ("Warm Halo Ring", 2, [1.0, 0.72, 0.3], 0.45, 5, 6, 0.95),
    ("Cyan Lens Glint", 1, [0.2, 0.88, 1.0], 1.4, 5, 8, 0.45),
    ("Amber Spike", 3, [1.0, 0.58, 0.18], 0.65, 4, 12, 0.35),
];

fn lens_flare_category_presets(cat: LensFlarePresetCategory) -> Vec<LensFlarePreset> {
    match cat {
        LensFlarePresetCategory::Cinematic => vec![
            LENS_FLARE_BASE_PRESETS[0],
            LENS_FLARE_BASE_PRESETS[1],
            LENS_FLARE_BASE_PRESETS[2],
            LENS_FLARE_BASE_PRESETS[3],
            LensFlarePreset {
                id: "cin_noir",
                name: "Silver Noir",
                category: cat,
                thumb_kind: 1,
                color: [0.85, 0.9, 1.0],
                brightness: 1.15,
                size: 1.1,
                anamorphic: 1.4,
                ghosts: 5,
                starburst_rays: 6,
                ring_intensity: 0.35,
                angle: 0.0,
            },
            LensFlarePreset {
                id: "cin_amber",
                name: "Tungsten Key",
                category: cat,
                thumb_kind: 0,
                color: [1.0, 0.78, 0.48],
                brightness: 1.3,
                size: 1.15,
                anamorphic: 0.85,
                ghosts: 5,
                starburst_rays: 8,
                ring_intensity: 0.45,
                angle: 5.0,
            },
        ],
        LensFlarePresetCategory::Rainbow => (0..8)
            .map(|i| {
                let hue = i as f32 / 8.0;
                let rgb = Rgb::from_hsv(Hsv {
                    h: hue,
                    s: 0.68,
                    v: 1.0,
                });
                const NAMES: [&str; 8] = [
                    "Spectrum Red",
                    "Solar Orange",
                    "Neon Lime",
                    "Emerald Prism",
                    "Aqua Beam",
                    "Cobalt Drift",
                    "Violet Ring",
                    "Magenta Nova",
                ];
                LensFlarePreset {
                    id: NAMES[i],
                    name: NAMES[i],
                    category: cat,
                    thumb_kind: (i % 4) as u8,
                    color: [f64::from(rgb.r()), f64::from(rgb.g()), f64::from(rgb.b())],
                    brightness: 1.2 + (i % 3) as f64 * 0.2,
                    size: 0.95 + (i % 4) as f64 * 0.15,
                    anamorphic: 0.5 + (i % 3) as f64 * 0.45,
                    ghosts: 4 + (i as u32 % 5),
                    starburst_rays: 6 + (i as u32 % 4) * 3,
                    ring_intensity: 0.5 + (i % 3) as f64 * 0.3,
                    angle: (i as f64 - 3.5) * 6.0,
                }
            })
            .collect(),
        LensFlarePresetCategory::Gold => (0..8)
            .map(|i| {
                const NAMES: [&str; 8] = [
                    "Sunbeam",
                    "Champagne",
                    "Brass Flare",
                    "Amber Streak",
                    "Honey Ring",
                    "Gilded Star",
                    "Desert Glare",
                    "Copper Core",
                ];
                let t = i as f64 / 7.0;
                LensFlarePreset {
                    id: NAMES[i],
                    name: NAMES[i],
                    category: cat,
                    thumb_kind: (i % 4) as u8,
                    color: [1.0, 0.82 - t * 0.28, 0.45 - t * 0.25],
                    brightness: 1.15 + (i % 3) as f64 * 0.25,
                    size: 1.0 + (i % 4) as f64 * 0.15,
                    anamorphic: 0.35 + (i % 4) as f64 * 0.35,
                    ghosts: 3 + (i as u32 % 5),
                    starburst_rays: 6 + (i as u32 % 3) * 4,
                    ring_intensity: 0.35 + (i % 3) as f64 * 0.25,
                    angle: (i as f64 - 3.5) * 4.0,
                }
            })
            .collect(),
    }
}

fn paint_lens_flare_thumb(painter: &egui::Painter, rect: Rect, kind: u8, rgb: [f64; 3]) {
    painter.rect_filled(rect, CornerRadius::same(2), Color32::from_rgb(10, 12, 16));
    let c = rect.center();
    let r = rect.height().min(rect.width()) * 0.36;
    let col = Color32::from_rgb(
        (rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
        (rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
        (rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
    );
    painter.circle_filled(c, r * 1.15, color32_with_alpha(col, 48));
    painter.circle_filled(c, r * 0.55, color32_with_alpha(col, 140));
    painter.circle_filled(c, r * 0.22, Color32::WHITE);
    match kind {
        1 => {
            // Anamorphic horizontal streak
            let streak = Rect::from_center_size(c, Vec2::new(rect.width() * 0.86, 2.5));
            painter.rect_filled(streak, CornerRadius::same(1), color32_with_alpha(col, 210));
        }
        2 => {
            // Halo ring + ghost circles
            painter.circle_stroke(c, r * 0.92, Stroke::new(1.2, color32_with_alpha(col, 180)));
            let g1 = Pos2::new(c.x - rect.width() * 0.24, c.y + rect.height() * 0.12);
            let g2 = Pos2::new(c.x + rect.width() * 0.26, c.y - rect.height() * 0.14);
            painter.circle_filled(g1, r * 0.26, color32_with_alpha(col, 95));
            painter.circle_filled(g2, r * 0.18, color32_with_alpha(col, 115));
        }
        3 => {
            // Starburst rays
            for k in 0..8 {
                let a = (k as f32) * std::f32::consts::FRAC_PI_4;
                let len = if k % 2 == 0 { r * 1.35 } else { r * 0.85 };
                let p = Pos2::new(c.x + a.cos() * len, c.y + a.sin() * len);
                painter.line_segment([c, p], Stroke::new(1.0, color32_with_alpha(col, 195)));
            }
        }
        _ => {}
    }
    painter.rect_stroke(
        rect,
        CornerRadius::same(2),
        Stroke::new(1.0, BORDER_PANEL),
        egui::StrokeKind::Inside,
    );
}

fn apply_lens_flare_preset(
    session: &mut Session,
    object_id: &str,
    preset: &LensFlarePreset,
    local_frame: f64,
) {
    let stack = default_flare_stack_from_preset(preset);
    let flare_setup = json!({
        "stack": stack.iter().map(OpticalFlareElement::to_json).collect::<Vec<_>>(),
    });
    let batch = [
        ("flareSetup", flare_setup),
        ("color", json!(preset.color)),
        ("flareColor", json!(preset.color)),
        ("brightness", json!(preset.brightness)),
        ("scale", json!(preset.size)),
        ("size", json!(preset.size)),
        ("anamorphic", json!(preset.anamorphic)),
        ("ghosts", json!(preset.ghosts)),
        ("starburstRays", json!(preset.starburst_rays)),
        ("ringIntensity", json!(preset.ring_intensity)),
        ("angle", json!(preset.angle)),
    ];
    apply_obj_batch_tx(
        session,
        object_id,
        &format!("Lens Flare preset: {}", preset.name),
        &batch,
        local_frame,
    );
}

/// Renders the floating 3D Lens Flares Options window when `state.open` is true.
pub fn show_lens_flare_options_window(
    ctx: &egui::Context,
    viewport_rect: Rect,
    floating_windows: &mut FloatingWindowManager,
    state: &mut LensFlareOptionsState,
    session: &mut Session,
    theme: &Cm3Theme,
) {
    if !state.open {
        return;
    }
    let Some((obj, local_frame)) =
        find_object3d_in_session(session, &state.object_id).map(|(o, lf)| (o.clone(), lf))
    else {
        state.open = false;
        floating_windows.close("pro_lens_flare_options");
        return;
    };

    let ctx_screen = ctx.input(|i| i.viewport_rect());
    let screen = if ctx_screen.width() > 100.0 && ctx_screen.height() > 100.0 {
        ctx_screen
    } else {
        viewport_rect
    };
    let (_, default_size) = compute_clamped_editor_window_rect(screen, 720.0, 520.0);
    let spec = FloatingWindowSpec::new("pro_lens_flare_options", "Lens Flares Options")
        .with_subtitle(if state.object_label.is_empty() {
            "3D Lens Flare".to_owned()
        } else {
            state.object_label.clone()
        })
        .with_default_size(default_size);

    ensure_pro_window_in_screen(floating_windows, &spec, screen, &mut state.just_opened);

    let mut close_requested = false;
    let ev = cm3_floating_window(ctx, floating_windows, &spec, screen, theme, |ui| {
        let body_h = (ui.available_height() - 34.0).max(160.0);
        let right_w = 240.0_f32;
        let left_w = (ui.available_width() - right_w - 8.0).max(200.0);

        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;

            // Left: Browser (Flare / Elements / Presets)
            ui.allocate_ui_with_layout(
                Vec2::new(left_w, body_h),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    ui.horizontal(|ui| {
                        for (tab, label) in [
                            (LensFlareBrowserTab::Flare, "Flare"),
                            (LensFlareBrowserTab::Elements, "Elements"),
                            (LensFlareBrowserTab::Presets, "Presets"),
                        ] {
                            let variant = if state.browser_tab == tab {
                                ButtonVariant::Primary
                            } else {
                                ButtonVariant::Normal
                            };
                            if cm3_prop_button(ui, label, variant, false, theme).clicked() {
                                state.browser_tab = tab;
                            }
                        }
                    });
                    ui.add_space(4.0);

                    egui::Frame::NONE
                        .fill(BG_SUNKEN)
                        .stroke(Stroke::new(1.0, BORDER_PANEL))
                        .corner_radius(CornerRadius::same(3))
                        .inner_margin(egui::Margin::same(6))
                        .show(ui, |ui| {
                            let inner_h = (body_h - 32.0).max(100.0);
                            ui.set_min_height(inner_h);
                            ui.set_max_height(inner_h);
                            ScrollArea::vertical()
                                .id_salt(("lf_browser_scroll", &obj.id))
                                .auto_shrink([false, false])
                                .show(ui, |ui| match state.browser_tab {
                                    LensFlareBrowserTab::Flare => {
                                        ui.horizontal_wrapped(|ui| {
                                            ui.spacing_mut().item_spacing = Vec2::new(6.0, 6.0);
                                            for preset in LENS_FLARE_BASE_PRESETS {
                                                let tile_w = 132.0;
                                                let (rect, resp) = ui.allocate_exact_size(
                                                    Vec2::new(tile_w, 66.0),
                                                    Sense::click(),
                                                );
                                                let thumb_rect = Rect::from_min_size(
                                                    rect.min,
                                                    Vec2::new(tile_w, 46.0),
                                                );
                                                paint_lens_flare_thumb(
                                                    ui.painter(),
                                                    thumb_rect,
                                                    preset.thumb_kind,
                                                    preset.color,
                                                );
                                                ui.painter().text(
                                                    Pos2::new(
                                                        rect.left() + 4.0,
                                                        rect.bottom() - 9.0,
                                                    ),
                                                    egui::Align2::LEFT_CENTER,
                                                    preset.name,
                                                    theme.font(FONT_COMPACT),
                                                    if resp.hovered() {
                                                        TEXT_BRIGHT
                                                    } else {
                                                        TEXT_PRIMARY
                                                    },
                                                );
                                                if resp.clicked() {
                                                    apply_lens_flare_preset(
                                                        session,
                                                        &obj.id,
                                                        preset,
                                                        local_frame,
                                                    );
                                                }
                                            }
                                        });
                                    }
                                    LensFlareBrowserTab::Elements => {
                                        let mut stack = read_optical_flare_stack(&obj, local_frame);
                                        let mut stack_dirty_label: Option<String> = None;

                                        // Top bar: + Add Element dropdown (all 12 flare element types) + count
                                        ui.horizontal(|ui| {
                                            egui::ComboBox::from_id_salt((
                                                "lf_add_element",
                                                &obj.id,
                                            ))
                                            .selected_text("+ Add Element")
                                            .width(142.0)
                                            .show_ui(
                                                ui,
                                                |ui| {
                                                    for def in FLARE_ELEMENT_TYPES {
                                                        if ui
                                                            .selectable_label(false, def.name)
                                                            .clicked()
                                                        {
                                                            let next_serial = stack.len() + 1;
                                                            stack.push(
                                                                OpticalFlareElement::from_type(
                                                                    def.type_id,
                                                                    next_serial,
                                                                ),
                                                            );
                                                            state.selected_element =
                                                                Some(stack.len().saturating_sub(1));
                                                            stack_dirty_label = Some(format!(
                                                                "Add Lens Flare element: {}",
                                                                def.name
                                                            ));
                                                        }
                                                    }
                                                },
                                            );
                                            ui.label(
                                                egui::RichText::new(format!(
                                                    "{} element{}",
                                                    stack.len(),
                                                    if stack.len() == 1 { "" } else { "s" }
                                                ))
                                                .font(theme.font(FONT_COMPACT))
                                                .color(TEXT_MUTED),
                                            );
                                        });
                                        ui.add_space(4.0);

                                        // Mini optical-axis strip showing active element positions (-1..+1)
                                        let (axis_rect, _) = ui.allocate_exact_size(
                                            Vec2::new(ui.available_width().max(120.0), 22.0),
                                            Sense::hover(),
                                        );
                                        ui.painter().rect_filled(
                                            axis_rect,
                                            CornerRadius::same(2),
                                            Color32::from_rgb(10, 12, 16),
                                        );
                                        ui.painter().line_segment(
                                            [
                                                Pos2::new(
                                                    axis_rect.left() + 8.0,
                                                    axis_rect.center().y,
                                                ),
                                                Pos2::new(
                                                    axis_rect.right() - 8.0,
                                                    axis_rect.center().y,
                                                ),
                                            ],
                                            Stroke::new(1.0, BORDER_CONTROL),
                                        );
                                        let any_solo = stack.iter().any(|e| e.solo);
                                        for (ei, el) in stack.iter().enumerate() {
                                            if !el.enabled || (any_solo && !el.solo) {
                                                continue;
                                            }
                                            let t = ((el.position as f32) * 0.25 + 0.5)
                                                .clamp(0.05, 0.95);
                                            let px = axis_rect.left()
                                                + 8.0
                                                + (axis_rect.width() - 16.0) * t;
                                            let col = Color32::from_rgb(
                                                (el.color[0].clamp(0.0, 1.0) * 255.0) as u8,
                                                (el.color[1].clamp(0.0, 1.0) * 255.0) as u8,
                                                (el.color[2].clamp(0.0, 1.0) * 255.0) as u8,
                                            );
                                            let r = if state.selected_element == Some(ei) {
                                                5.0
                                            } else {
                                                3.5
                                            };
                                            ui.painter().circle_filled(
                                                Pos2::new(px, axis_rect.center().y),
                                                r,
                                                col,
                                            );
                                        }
                                        ui.add_space(6.0);

                                        // Editable Element Stack rows
                                        let mut row_action: Option<(usize, &'static str)> = None;
                                        let stack_len = stack.len();
                                        #[allow(clippy::needless_range_loop)]
                                        for idx in 0..stack_len {
                                            let is_sel = state.selected_element == Some(idx);
                                            let def =
                                                flare_element_type_def(stack[idx].element_type);
                                            let enabled_now = stack[idx].enabled;
                                            let solo_now = stack[idx].solo;
                                            egui::Frame::NONE
                                                .fill(if is_sel { BG_HOVER } else { BG_INPUT })
                                                .stroke(Stroke::new(
                                                    1.0,
                                                    if is_sel {
                                                        theme.accent()
                                                    } else {
                                                        BORDER_CONTROL
                                                    },
                                                ))
                                                .corner_radius(CornerRadius::same(3))
                                                .inner_margin(egui::Margin::symmetric(5, 4))
                                                .show(ui, |ui| {
                                                    ui.horizontal(|ui| {
                                                        ui.spacing_mut().item_spacing.x = 4.0;
                                                        let (thumb_r, _) = ui.allocate_exact_size(
                                                            Vec2::new(26.0, 18.0),
                                                            Sense::hover(),
                                                        );
                                                        paint_lens_flare_thumb(
                                                            ui.painter(),
                                                            thumb_r,
                                                            def.thumb_kind,
                                                            stack[idx].color,
                                                        );
                                                        if ui
                                                            .checkbox(&mut stack[idx].enabled, "")
                                                            .on_hover_text("Enable / Disable")
                                                            .changed()
                                                        {
                                                            stack_dirty_label = Some(format!(
                                                                "Toggle Lens Flare element: {}",
                                                                stack[idx].name
                                                            ));
                                                        }
                                                        let solo_btn = egui::Button::new(
                                                            egui::RichText::new("S")
                                                                .font(theme.font(FONT_COMPACT))
                                                                .color(if solo_now {
                                                                    TEXT_BRIGHT
                                                                } else {
                                                                    TEXT_MUTED
                                                                }),
                                                        )
                                                        .fill(if solo_now {
                                                            theme.accent()
                                                        } else {
                                                            BG_BUTTON
                                                        });
                                                        if ui
                                                            .add(solo_btn)
                                                            .on_hover_text("Solo element")
                                                            .clicked()
                                                        {
                                                            stack[idx].solo = !stack[idx].solo;
                                                            stack_dirty_label = Some(format!(
                                                                "Solo Lens Flare element: {}",
                                                                stack[idx].name
                                                            ));
                                                        }
                                                        if ui
                                                            .selectable_label(
                                                                is_sel,
                                                                egui::RichText::new(format!(
                                                                    "{} ({})",
                                                                    stack[idx].name, def.name
                                                                ))
                                                                .font(theme.font(FONT_SMALL))
                                                                .color(if enabled_now {
                                                                    TEXT_PRIMARY
                                                                } else {
                                                                    TEXT_FAINT
                                                                }),
                                                            )
                                                            .clicked()
                                                        {
                                                            state.selected_element = Some(idx);
                                                        }
                                                        ui.with_layout(
                                                            egui::Layout::right_to_left(
                                                                egui::Align::Center,
                                                            ),
                                                            |ui| {
                                                                if ui
                                                                    .small_button("×")
                                                                    .on_hover_text("Delete element")
                                                                    .clicked()
                                                                {
                                                                    row_action = Some((idx, "del"));
                                                                }
                                                                if ui
                                                                    .small_button("D")
                                                                    .on_hover_text(
                                                                        "Duplicate element",
                                                                    )
                                                                    .clicked()
                                                                {
                                                                    row_action = Some((idx, "dup"));
                                                                }
                                                                if idx + 1 < stack_len
                                                                    && ui
                                                                        .small_button("↓")
                                                                        .on_hover_text("Move down")
                                                                        .clicked()
                                                                {
                                                                    row_action =
                                                                        Some((idx, "down"));
                                                                }
                                                                if idx > 0
                                                                    && ui
                                                                        .small_button("↑")
                                                                        .on_hover_text("Move up")
                                                                        .clicked()
                                                                {
                                                                    row_action = Some((idx, "up"));
                                                                }
                                                            },
                                                        );
                                                    });

                                                    // Per-element controls (Brightness, Scale, Position, Tint)
                                                    if is_sel {
                                                        ui.add_space(4.0);
                                                        ui.horizontal_wrapped(|ui| {
                                                            ui.spacing_mut().item_spacing =
                                                                Vec2::new(6.0, 4.0);
                                                            ui.label(
                                                                egui::RichText::new("Bright%")
                                                                    .font(theme.font(FONT_COMPACT))
                                                                    .color(TEXT_MUTED),
                                                            );
                                                            if cm3_scrub_number(
                                                                ui,
                                                                Id::unique((
                                                                    "lf_el_br", &obj.id, idx,
                                                                )),
                                                                &mut stack[idx].brightness,
                                                                ScrubConfig::new(
                                                                    0.0, 200.0, 1.0, 0,
                                                                ),
                                                                theme,
                                                            )
                                                            .changed
                                                            {
                                                                stack_dirty_label = Some(
                                                                    "Edit Lens Flare brightness"
                                                                        .to_owned(),
                                                                );
                                                            }
                                                            ui.label(
                                                                egui::RichText::new("Scale%")
                                                                    .font(theme.font(FONT_COMPACT))
                                                                    .color(TEXT_MUTED),
                                                            );
                                                            if cm3_scrub_number(
                                                                ui,
                                                                Id::unique((
                                                                    "lf_el_sc", &obj.id, idx,
                                                                )),
                                                                &mut stack[idx].scale,
                                                                ScrubConfig::new(
                                                                    5.0, 400.0, 2.0, 0,
                                                                ),
                                                                theme,
                                                            )
                                                            .changed
                                                            {
                                                                stack_dirty_label = Some(
                                                                    "Edit Lens Flare scale"
                                                                        .to_owned(),
                                                                );
                                                            }
                                                            ui.label(
                                                                egui::RichText::new("Axis Pos")
                                                                    .font(theme.font(FONT_COMPACT))
                                                                    .color(TEXT_MUTED),
                                                            );
                                                            if cm3_scrub_number(
                                                                ui,
                                                                Id::unique((
                                                                    "lf_el_pos",
                                                                    &obj.id,
                                                                    idx,
                                                                )),
                                                                &mut stack[idx].position,
                                                                ScrubConfig::new(
                                                                    -2.0, 2.0, 0.02, 2,
                                                                ),
                                                                theme,
                                                            )
                                                            .changed
                                                            {
                                                                stack_dirty_label = Some(
                                                                    "Edit Lens Flare position"
                                                                        .to_owned(),
                                                                );
                                                            }
                                                            ui.label(
                                                                egui::RichText::new("Tint")
                                                                    .font(theme.font(FONT_COMPACT))
                                                                    .color(TEXT_MUTED),
                                                            );
                                                            let mut tint = Rgba::new(
                                                                stack[idx].color[0] as f32,
                                                                stack[idx].color[1] as f32,
                                                                stack[idx].color[2] as f32,
                                                                1.0,
                                                            );
                                                            let el_picker_id = Id::unique((
                                                                "lf_el_tint",
                                                                &obj.id,
                                                                idx,
                                                            ));
                                                            let cev = cm3_hsva_color_picker(
                                                                ui,
                                                                el_picker_id,
                                                                &mut tint,
                                                                false,
                                                                false,
                                                                screen,
                                                                theme,
                                                            );
                                                            if cev.eyedropper_clicked {
                                                                request_eyedropper_sampler(
                                                                    ui.ctx(),
                                                                    el_picker_id,
                                                                );
                                                            }
                                                            if cev.changed {
                                                                stack[idx].color = [
                                                                    f64::from(tint.0[0]),
                                                                    f64::from(tint.0[1]),
                                                                    f64::from(tint.0[2]),
                                                                ];
                                                                stack_dirty_label = Some(
                                                                    "Edit Lens Flare tint"
                                                                        .to_owned(),
                                                                );
                                                            }
                                                        });
                                                    }
                                                });
                                            ui.add_space(3.0);
                                        }

                                        if let Some((idx, cmd)) = row_action {
                                            match cmd {
                                                "up" if idx > 0 => {
                                                    stack.swap(idx, idx - 1);
                                                    state.selected_element = Some(idx - 1);
                                                    stack_dirty_label =
                                                        Some("Move Lens Flare element up".into());
                                                }
                                                "down" if idx + 1 < stack.len() => {
                                                    stack.swap(idx, idx + 1);
                                                    state.selected_element = Some(idx + 1);
                                                    stack_dirty_label =
                                                        Some("Move Lens Flare element down".into());
                                                }
                                                "dup" if idx < stack.len() => {
                                                    let mut copy = stack[idx].clone();
                                                    copy.id = format!("el_{}", stack.len() + 1);
                                                    copy.name = format!("{} Copy", copy.name);
                                                    copy.solo = false;
                                                    stack.insert(idx + 1, copy);
                                                    state.selected_element = Some(idx + 1);
                                                    stack_dirty_label =
                                                        Some("Duplicate Lens Flare element".into());
                                                }
                                                "del" if idx < stack.len() => {
                                                    stack.remove(idx);
                                                    state.selected_element = if stack.is_empty() {
                                                        None
                                                    } else {
                                                        Some(idx.min(stack.len() - 1))
                                                    };
                                                    stack_dirty_label =
                                                        Some("Delete Lens Flare element".into());
                                                }
                                                _ => {}
                                            }
                                        }

                                        if let Some(label) = stack_dirty_label {
                                            commit_optical_flare_stack(
                                                session,
                                                &obj,
                                                &stack,
                                                &label,
                                                local_frame,
                                            );
                                        }
                                    }
                                    LensFlareBrowserTab::Presets => {
                                        ui.horizontal(|ui| {
                                            egui::ComboBox::from_id_salt("lf_preset_cat")
                                                .selected_text(state.preset_category.label())
                                                .width(130.0)
                                                .show_ui(ui, |ui| {
                                                    for cat in LensFlarePresetCategory::ALL {
                                                        ui.selectable_value(
                                                            &mut state.preset_category,
                                                            cat,
                                                            cat.label(),
                                                        );
                                                    }
                                                });
                                            ui.checkbox(&mut state.fav_only, "Favorites only");
                                        });
                                        ui.add_space(6.0);
                                        let presets =
                                            lens_flare_category_presets(state.preset_category);
                                        let mut shown = 0usize;
                                        ui.horizontal_wrapped(|ui| {
                                            ui.spacing_mut().item_spacing = Vec2::new(6.0, 6.0);
                                            for preset in &presets {
                                                let fav_key = format!(
                                                    "{:?}:{}",
                                                    state.preset_category, preset.name
                                                );
                                                let is_fav = state.favorites.contains(&fav_key);
                                                if state.fav_only && !is_fav {
                                                    continue;
                                                }
                                                shown += 1;
                                                ui.vertical(|ui| {
                                                    let (rect, resp) = ui.allocate_exact_size(
                                                        Vec2::new(132.0, 46.0),
                                                        Sense::click(),
                                                    );
                                                    paint_lens_flare_thumb(
                                                        ui.painter(),
                                                        rect,
                                                        preset.thumb_kind,
                                                        preset.color,
                                                    );
                                                    if resp.clicked() {
                                                        apply_lens_flare_preset(
                                                            session,
                                                            &obj.id,
                                                            preset,
                                                            local_frame,
                                                        );
                                                    }
                                                    ui.horizontal(|ui| {
                                                        ui.label(
                                                            egui::RichText::new(preset.name)
                                                                .font(theme.font(FONT_COMPACT))
                                                                .color(TEXT_PRIMARY),
                                                        );
                                                        if ui
                                                            .small_button(if is_fav {
                                                                "★"
                                                            } else {
                                                                "☆"
                                                            })
                                                            .clicked()
                                                        {
                                                            if is_fav {
                                                                state.favorites.remove(&fav_key);
                                                            } else {
                                                                state.favorites.insert(fav_key);
                                                            }
                                                        }
                                                    });
                                                });
                                            }
                                        });
                                        if shown == 0 {
                                            ui.label(
                                                egui::RichText::new(
                                                    "No favorite presets in this category.",
                                                )
                                                .font(theme.font(FONT_SMALL))
                                                .color(TEXT_MUTED),
                                            );
                                        }
                                    }
                                });
                        });
                },
            );

            // Right: Optical Parameters
            ui.allocate_ui_with_layout(
                Vec2::new(right_w, body_h),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    egui::Frame::NONE
                        .fill(BG_PANEL)
                        .stroke(Stroke::new(1.0, BORDER_PANEL))
                        .corner_radius(CornerRadius::same(3))
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            ui.set_width((right_w - 18.0).max(160.0));
                            ui.set_min_height((body_h - 18.0).max(120.0));
                            ui.label(
                                egui::RichText::new("Global & Optical")
                                    .font(theme.font(FONT_TITLE))
                                    .color(TEXT_STRONG),
                            );
                            ui.add_space(6.0);

                            // Flare color
                            let rgb =
                                read_obj_color(&obj, "flareColor", [1.0, 0.92, 0.8], local_frame);
                            let mut color =
                                Rgba::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, 1.0);
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new("Flare color")
                                        .font(theme.font(FONT_SMALL))
                                        .color(TEXT_PRIMARY),
                                );
                                let picker_id = Id::unique(("lf_opt_col", &obj.id));
                                let cev = cm3_hsva_color_picker(
                                    ui, picker_id, &mut color, false, false, screen, theme,
                                );
                                if cev.eyedropper_clicked {
                                    request_eyedropper_sampler(ui.ctx(), picker_id);
                                }
                                if cev.changed {
                                    write_obj_prop(
                                        session,
                                        &obj.id,
                                        "flareColor",
                                        json!([
                                            f64::from(color.0[0]),
                                            f64::from(color.0[1]),
                                            f64::from(color.0[2])
                                        ]),
                                        local_frame,
                                    );
                                }
                            });

                            for &(key, label, def, cfg) in &[
                                (
                                    "brightness",
                                    "Brightness",
                                    1.0,
                                    ScrubConfig::new(0.0, 10.0, 0.05, 2),
                                ),
                                ("size", "Size", 1.0, ScrubConfig::new(0.05, 10.0, 0.05, 2)),
                                (
                                    "anamorphic",
                                    "Anamorphic",
                                    0.5,
                                    ScrubConfig::new(0.0, 4.0, 0.02, 2),
                                ),
                                ("ghosts", "Ghosts", 4.0, ScrubConfig::new(0.0, 24.0, 1.0, 0)),
                                (
                                    "starburstRays",
                                    "Starburst rays",
                                    8.0,
                                    ScrubConfig::new(0.0, 32.0, 1.0, 0),
                                ),
                                (
                                    "ringIntensity",
                                    "Ring intensity",
                                    0.4,
                                    ScrubConfig::new(0.0, 2.0, 0.02, 2),
                                ),
                                (
                                    "angle",
                                    "Rotation",
                                    0.0,
                                    ScrubConfig::new(-360.0, 360.0, 1.0, 1),
                                ),
                            ] {
                                let mut v = read_obj_f64(&obj, key, def, local_frame);
                                ui.horizontal(|ui| {
                                    ui.add_sized(
                                        Vec2::new(105.0, 20.0),
                                        egui::Label::new(
                                            egui::RichText::new(label)
                                                .font(theme.font(FONT_SMALL))
                                                .color(TEXT_PRIMARY),
                                        )
                                        .truncate(),
                                    );
                                    let ev = cm3_scrub_number(
                                        ui,
                                        Id::unique(("lf_opt_num", &obj.id, key)),
                                        &mut v,
                                        cfg,
                                        theme,
                                    );
                                    if ev.changed {
                                        write_obj_prop(
                                            session,
                                            &obj.id,
                                            key,
                                            json!(v),
                                            local_frame,
                                        );
                                    }
                                });
                            }

                            let mut occ = read_obj_f64(&obj, "occlusion", 1.0, local_frame) >= 0.5;
                            if ui.checkbox(&mut occ, "Depth occlusion").changed() {
                                write_obj_prop(
                                    session,
                                    &obj.id,
                                    "occlusion",
                                    json!(if occ { 1 } else { 0 }),
                                    local_frame,
                                );
                            }

                            ui.add_space(10.0);
                            if cm3_prop_button(ui, "Reset All", ButtonVariant::Normal, true, theme)
                                .clicked()
                            {
                                apply_lens_flare_preset(
                                    session,
                                    &obj.id,
                                    &LENS_FLARE_BASE_PRESETS[0],
                                    local_frame,
                                );
                            }
                        });
                },
            );
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Click any flare tile to apply • Live 3D viewport preview")
                    .font(theme.font(FONT_COMPACT))
                    .color(TEXT_FAINT),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if cm3_prop_button(ui, "Done", ButtonVariant::Primary, false, theme).clicked() {
                    close_requested = true;
                }
            });
        });
    });

    clamp_floating_window_in_screen(floating_windows, &spec, screen);

    if ev.closed && ctx.input(|i| i.key_pressed(Key::Escape)) {
        mark_floating_editor_escape_consumed(ctx);
    } else if !ev.closed
        && consume_escape_for_floating_editor(ctx, floating_windows, "pro_lens_flare_options")
    {
        close_requested = true;
    }

    if close_requested || ev.closed {
        state.open = false;
        floating_windows.close("pro_lens_flare_options");
    }
}

// ===========================================================================
// 2. Particle & Light Suite Designer (`plugins/trapcode-suite/designer.js`)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleDesignerKind {
    Particular,
    Form,
    Plexus,
}

impl ParticleDesignerKind {
    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Self::Particular => "3D Particle Emitter Designer",
            Self::Form => "Particle Lattice Designer",
            Self::Plexus => "Point Cloud Network Designer",
        }
    }

    #[must_use]
    pub const fn blocks(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Particular => &[
                ("emitter", "Emitter"),
                ("particle", "Particle"),
                ("physics", "Physics"),
                ("turbulence", "Turbulence"),
                ("render", "Render"),
            ],
            Self::Form => &[
                ("grid", "Base Form"),
                ("displace", "Displace"),
                ("turbulence", "Turbulence"),
                ("particle", "Particle"),
                ("render", "Render"),
            ],
            Self::Plexus => &[
                ("points", "Point Cloud"),
                ("lines", "Connections"),
                ("turbulence", "Turbulence"),
                ("render", "Render"),
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParticleDesignerState {
    pub open: bool,
    pub kind: ParticleDesignerKind,
    pub object_id: String,
    pub object_label: String,
    pub show_presets: bool,
    pub show_params: bool,
    pub selected_block: usize,
    pub just_opened: bool,
}

impl Default for ParticleDesignerState {
    fn default() -> Self {
        Self {
            open: false,
            kind: ParticleDesignerKind::Particular,
            object_id: String::new(),
            object_label: String::new(),
            show_presets: true,
            show_params: true,
            selected_block: 0,
            just_opened: false,
        }
    }
}

impl ParticleDesignerState {
    pub fn open_for_object(
        &mut self,
        kind: ParticleDesignerKind,
        object_id: impl Into<String>,
        object_label: impl Into<String>,
    ) {
        self.kind = kind;
        self.object_id = object_id.into();
        self.object_label = object_label.into();
        self.selected_block = 0;
        self.open = true;
        self.just_opened = true;
    }
}

const DESIGNER_PALETTES: &[(&str, [f64; 3], [Color32; 3])] = &[
    (
        "Cyan Pulse",
        [0.25, 0.85, 1.0],
        [
            Color32::from_rgb(20, 90, 180),
            Color32::from_rgb(64, 215, 255),
            Color32::from_rgb(210, 248, 255),
        ],
    ),
    (
        "Solar Ember",
        [1.0, 0.55, 0.18],
        [
            Color32::from_rgb(160, 35, 15),
            Color32::from_rgb(255, 140, 45),
            Color32::from_rgb(255, 235, 150),
        ],
    ),
    (
        "Synthwave",
        [0.95, 0.28, 0.78],
        [
            Color32::from_rgb(110, 25, 195),
            Color32::from_rgb(245, 70, 200),
            Color32::from_rgb(70, 225, 255),
        ],
    ),
    (
        "Aurora Mint",
        [0.32, 0.95, 0.68],
        [
            Color32::from_rgb(20, 120, 105),
            Color32::from_rgb(80, 242, 175),
            Color32::from_rgb(205, 255, 230),
        ],
    ),
    (
        "Ghost White",
        [0.92, 0.95, 1.0],
        [
            Color32::from_rgb(110, 125, 145),
            Color32::from_rgb(215, 225, 240),
            Color32::from_rgb(255, 255, 255),
        ],
    ),
];

/// Builds a 3-stop `particles.colorOverLife` JSON array (`t = 0.0, 0.5, 1.0`) from a 3-color
/// designer palette.
#[must_use]
pub fn build_palette_color_over_life(stops: [Color32; 3]) -> Value {
    let to_rgb = |c: Color32| {
        [
            f64::from(c.r()) / 255.0,
            f64::from(c.g()) / 255.0,
            f64::from(c.b()) / 255.0,
        ]
    };
    json!([
        { "t": 0.0, "color": to_rgb(stops[0]) },
        { "t": 0.5, "color": to_rgb(stops[1]) },
        { "t": 1.0, "color": to_rgb(stops[2]) },
    ])
}

/// Applies a 3-color designer palette atomically, writing BOTH the primary color property
/// (`particleColor` / `lineColor`) AND the 3-stop `particles.colorOverLife` + `colorOverLife`
/// gradient JSON (and updating active render blocks if present).
pub fn apply_designer_palette(
    session: &mut Session,
    obj: &SessionObject3D,
    kind: ParticleDesignerKind,
    palette_name: &str,
    rgb: [f64; 3],
    stops: [Color32; 3],
    local_frame: f64,
) {
    let color_key = match kind {
        ParticleDesignerKind::Plexus => "lineColor",
        _ => "particleColor",
    };
    let gradient = build_palette_color_over_life(stops);
    let mut batch: Vec<(&str, Value)> = vec![
        (color_key, json!(rgb)),
        ("particles.colorOverLife", gradient.clone()),
        ("colorOverLife", gradient),
    ];
    if obj.properties.contains_key("blocks") {
        let mut blocks = read_particle_blocks(obj, kind, local_frame);
        for blk in &mut blocks {
            if blk.enabled && matches!(blk.kind.as_str(), "render" | "appearance") {
                blk.params.insert(color_key.to_owned(), json!(rgb));
            }
        }
        batch.push((
            "blocks",
            Value::Array(blocks.iter().map(ParticleDesignerBlock::to_json).collect()),
        ));
    }
    apply_obj_batch_tx(
        session,
        &obj.id,
        &format!("Designer palette: {palette_name}"),
        &batch,
        local_frame,
    );
}

/// One editable block inside the Particle & Light Designer (`blocks` JSON property).
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleDesignerBlock {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub enabled: bool,
    pub params: Map<String, Value>,
}

fn canonical_block_kind(raw: &str) -> String {
    if raw.eq_ignore_ascii_case("appearance") {
        "render".to_owned()
    } else {
        raw.to_ascii_lowercase()
    }
}

fn block_default_label(kind: ParticleDesignerKind, block_kind: &str) -> &'static str {
    let canon = if block_kind.eq_ignore_ascii_case("appearance") {
        "render"
    } else {
        block_kind
    };
    kind.blocks()
        .iter()
        .find(|&&(k, _)| k == canon)
        .map_or("Block", |&(_, l)| l)
}

impl ParticleDesignerBlock {
    #[must_use]
    pub fn new_default(kind: ParticleDesignerKind, block_kind: &str, serial: usize) -> Self {
        let bk = canonical_block_kind(block_kind);
        let label = block_default_label(kind, &bk).to_owned();
        let mut params = Map::new();
        match (kind, bk.as_str()) {
            (ParticleDesignerKind::Particular, "emitter") => {
                params.insert("count".into(), json!(64.0));
                params.insert("rate".into(), json!(30.0));
                params.insert("emitterSize".into(), json!([6.0, 6.0, 6.0]));
                params.insert("seed".into(), json!(1.0));
            }
            (ParticleDesignerKind::Particular, "particle") => {
                params.insert("lifetime".into(), json!(3.0));
                params.insert("particleSize".into(), json!(3.0));
            }
            (ParticleDesignerKind::Particular, "physics") => {
                params.insert("velocity".into(), json!(12.0));
                params.insert("gravity".into(), json!(-4.0));
            }
            (_, "turbulence") => {
                params.insert("turbulence".into(), json!(1.5));
                params.insert("turbulenceScale".into(), json!(2.0));
                params.insert("turbulenceSpeed".into(), json!(1.0));
            }
            (ParticleDesignerKind::Particular, "render") => {
                params.insert("particleColor".into(), json!([0.4, 0.85, 1.0]));
                params.insert("opacity".into(), json!(1.0));
            }
            (ParticleDesignerKind::Form, "grid") => {
                params.insert("particlesX".into(), json!(6.0));
                params.insert("particlesY".into(), json!(6.0));
                params.insert("particlesZ".into(), json!(4.0));
                params.insert("sizeX".into(), json!(24.0));
                params.insert("sizeY".into(), json!(24.0));
                params.insert("sizeZ".into(), json!(16.0));
            }
            (ParticleDesignerKind::Form, "displace") => {
                params.insert("displace".into(), json!(2.5));
                params.insert("seed".into(), json!(1.0));
            }
            (ParticleDesignerKind::Form, "particle") => {
                params.insert("particleSize".into(), json!(0.85));
            }
            (ParticleDesignerKind::Form, "render") => {
                params.insert("particleColor".into(), json!([0.85, 0.92, 1.0]));
                params.insert("opacity".into(), json!(1.0));
            }
            (ParticleDesignerKind::Plexus, "points") => {
                params.insert("count".into(), json!(24.0));
                params.insert("radius".into(), json!(18.0));
                params.insert("seed".into(), json!(1.0));
            }
            (ParticleDesignerKind::Plexus, "lines") => {
                params.insert("maxDistance".into(), json!(15.3));
                params.insert("beamThickness".into(), json!(0.25));
            }
            (ParticleDesignerKind::Plexus, "render") => {
                params.insert("lineColor".into(), json!([0.85, 0.92, 1.0]));
                params.insert("opacity".into(), json!(1.0));
            }
            _ => {}
        }
        Self {
            id: format!("blk_{serial}"),
            kind: bk,
            label,
            enabled: true,
            params,
        }
    }

    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "enabled": self.enabled,
            "params": Value::Object(self.params.clone()),
        })
    }

    #[must_use]
    pub fn from_json(val: &Value, kind: ParticleDesignerKind, idx: usize) -> Option<Self> {
        let obj = val.as_object()?;
        let raw_kind = obj
            .get("kind")
            .or_else(|| obj.get("type"))
            .and_then(Value::as_str)?;
        let bk = canonical_block_kind(raw_kind);
        let id = obj
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("blk_{}", idx + 1));
        let label = obj
            .get("label")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| block_default_label(kind, &bk).to_owned());
        let enabled = obj.get("enabled").and_then(Value::as_bool).unwrap_or(true);
        let params = obj
            .get("params")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        Some(Self {
            id,
            kind: bk,
            label,
            enabled,
            params,
        })
    }
}

#[must_use]
pub fn read_particle_blocks(
    obj: &SessionObject3D,
    kind: ParticleDesignerKind,
    local_frame: f64,
) -> Vec<ParticleDesignerBlock> {
    if let Some(val) = read_obj_prop(obj.properties.get("blocks"), local_frame)
        && let Some(arr) = val.as_array()
    {
        let parsed: Vec<ParticleDesignerBlock> = arr
            .iter()
            .enumerate()
            .filter_map(|(i, v)| ParticleDesignerBlock::from_json(v, kind, i))
            .collect();
        if !parsed.is_empty() {
            return parsed;
        }
    }

    // Build initial starter blocks populated from the object's live properties.
    let mut blocks = Vec::new();
    for (i, &(bkey, _)) in kind.blocks().iter().enumerate() {
        if bkey == "turbulence" && !obj.properties.contains_key("turbulence") {
            continue;
        }
        let mut blk = ParticleDesignerBlock::new_default(kind, bkey, i + 1);
        for key in blk.params.keys().cloned().collect::<Vec<_>>() {
            if let Some(live_val) = read_obj_prop(obj.properties.get(&key), local_frame) {
                blk.params.insert(key, live_val);
            }
        }
        blocks.push(blk);
    }
    blocks
}

fn blk_f64(blk: &ParticleDesignerBlock, key: &str, default: f64) -> f64 {
    blk.params
        .get(key)
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

fn blk_vec3(blk: &ParticleDesignerBlock, key: &str, default: [f64; 3]) -> [f64; 3] {
    blk.params
        .get(key)
        .and_then(Value::as_array)
        .filter(|a| a.len() >= 3)
        .map(|a| {
            [
                a[0].as_f64().unwrap_or(default[0]),
                a[1].as_f64().unwrap_or(default[1]),
                a[2].as_f64().unwrap_or(default[2]),
            ]
        })
        .unwrap_or(default)
}

/// Aggregates all enabled blocks into the live particle object properties read by `zc-render`.
#[must_use]
pub fn sync_particle_blocks_to_live_props(
    kind: ParticleDesignerKind,
    blocks: &[ParticleDesignerBlock],
) -> Vec<(&'static str, Value)> {
    let mut out: Vec<(&'static str, Value)> = Vec::new();
    match kind {
        ParticleDesignerKind::Particular => {
            let mut has_emitter = false;
            let mut count_sum = 0.0_f64;
            let mut rate_sum = 0.0_f64;
            let mut emitter_size = [6.0, 6.0, 6.0];
            let mut seed = 1.0_f64;
            let mut has_particle = false;
            let mut lifetime = 3.0_f64;
            let mut particle_size = 3.0_f64;
            let mut has_physics = false;
            let mut velocity_sum = 0.0_f64;
            let mut gravity_sum = 0.0_f64;
            let mut has_turb = false;
            let mut turb_sum = 0.0_f64;
            let mut turb_scale = 2.0_f64;
            let mut turb_speed = 1.0_f64;
            let mut has_render = false;
            let mut color = [0.4, 0.85, 1.0];
            let mut opacity = 1.0_f64;

            for blk in blocks.iter().filter(|b| b.enabled) {
                match blk.kind.as_str() {
                    "emitter" => {
                        has_emitter = true;
                        count_sum += blk_f64(blk, "count", 64.0).max(0.0);
                        rate_sum += blk_f64(blk, "rate", 30.0).max(0.0);
                        emitter_size = blk_vec3(blk, "emitterSize", emitter_size);
                        seed = blk_f64(blk, "seed", seed);
                    }
                    "particle" => {
                        has_particle = true;
                        lifetime = blk_f64(blk, "lifetime", lifetime);
                        particle_size = blk_f64(blk, "particleSize", particle_size);
                    }
                    "physics" => {
                        has_physics = true;
                        velocity_sum += blk_f64(blk, "velocity", 12.0);
                        gravity_sum += blk_f64(blk, "gravity", -4.0);
                    }
                    "turbulence" => {
                        has_turb = true;
                        turb_sum += blk_f64(blk, "turbulence", 1.5).max(0.0);
                        turb_scale = blk_f64(blk, "turbulenceScale", turb_scale);
                        turb_speed = blk_f64(blk, "turbulenceSpeed", turb_speed);
                    }
                    "render" | "appearance" => {
                        has_render = true;
                        color = blk_vec3(blk, "particleColor", color);
                        opacity = blk_f64(blk, "opacity", opacity);
                    }
                    _ => {}
                }
            }
            out.push((
                "count",
                json!(if has_emitter {
                    count_sum.clamp(1.0, 4096.0)
                } else {
                    0.0
                }),
            ));
            out.push((
                "rate",
                json!(if has_emitter {
                    rate_sum.clamp(0.0, 2000.0)
                } else {
                    0.0
                }),
            ));
            if has_emitter {
                out.push(("emitterSize", json!(emitter_size)));
                out.push(("seed", json!(seed)));
            }
            if has_particle {
                out.push(("lifetime", json!(lifetime)));
                out.push(("particleSize", json!(particle_size)));
            }
            out.push((
                "velocity",
                json!(if has_physics { velocity_sum } else { 0.0 }),
            ));
            out.push((
                "gravity",
                json!(if has_physics { gravity_sum } else { 0.0 }),
            ));
            out.push(("turbulence", json!(if has_turb { turb_sum } else { 0.0 })));
            if has_turb {
                out.push(("turbulenceScale", json!(turb_scale)));
                out.push(("turbulenceSpeed", json!(turb_speed)));
            }
            if has_render {
                out.push(("particleColor", json!(color)));
                out.push(("opacity", json!(opacity)));
            } else {
                out.push(("opacity", json!(0.0)));
            }
        }
        ParticleDesignerKind::Form => {
            let mut has_grid = false;
            let mut px = 6.0_f64;
            let mut py = 6.0_f64;
            let mut pz = 4.0_f64;
            let mut sx = 24.0_f64;
            let mut sy = 24.0_f64;
            let mut sz = 16.0_f64;
            let mut disp_sum = 0.0_f64;
            let mut seed = 1.0_f64;
            let mut has_disp = false;
            let mut turb_sum = 0.0_f64;
            let mut has_particle = false;
            let mut psize = 0.85_f64;
            let mut has_render = false;
            let mut color = [0.85, 0.92, 1.0];
            let mut opacity = 1.0_f64;

            for blk in blocks.iter().filter(|b| b.enabled) {
                match blk.kind.as_str() {
                    "grid" => {
                        has_grid = true;
                        px = blk_f64(blk, "particlesX", px);
                        py = blk_f64(blk, "particlesY", py);
                        pz = blk_f64(blk, "particlesZ", pz);
                        sx = blk_f64(blk, "sizeX", sx);
                        sy = blk_f64(blk, "sizeY", sy);
                        sz = blk_f64(blk, "sizeZ", sz);
                    }
                    "displace" => {
                        has_disp = true;
                        disp_sum += blk_f64(blk, "displace", 2.5);
                        seed = blk_f64(blk, "seed", seed);
                    }
                    "turbulence" => {
                        turb_sum += blk_f64(blk, "turbulence", 1.5).max(0.0);
                    }
                    "particle" => {
                        has_particle = true;
                        psize = blk_f64(blk, "particleSize", psize);
                    }
                    "render" | "appearance" => {
                        has_render = true;
                        color = blk_vec3(blk, "particleColor", color);
                        opacity = blk_f64(blk, "opacity", opacity);
                    }
                    _ => {}
                }
            }
            if has_grid {
                out.push(("particlesX", json!(px)));
                out.push(("particlesY", json!(py)));
                out.push(("particlesZ", json!(pz)));
                out.push(("sizeX", json!(sx)));
                out.push(("sizeY", json!(sy)));
                out.push(("sizeZ", json!(sz)));
            }
            out.push(("displace", json!(if has_disp { disp_sum } else { 0.0 })));
            if has_disp {
                out.push(("seed", json!(seed)));
            }
            out.push(("turbulence", json!(turb_sum)));
            if has_particle {
                out.push(("particleSize", json!(psize)));
            }
            if has_render {
                out.push(("particleColor", json!(color)));
                out.push(("opacity", json!(opacity)));
            } else {
                out.push(("opacity", json!(0.0)));
            }
        }
        ParticleDesignerKind::Plexus => {
            let mut has_points = false;
            let mut count_sum = 0.0_f64;
            let mut radius = 18.0_f64;
            let mut seed = 1.0_f64;
            let mut has_lines = false;
            let mut max_dist = 15.3_f64;
            let mut beam_thick = 0.25_f64;
            let mut turb_sum = 0.0_f64;
            let mut has_render = false;
            let mut color = [0.85, 0.92, 1.0];
            let mut opacity = 1.0_f64;

            for blk in blocks.iter().filter(|b| b.enabled) {
                match blk.kind.as_str() {
                    "points" => {
                        has_points = true;
                        count_sum += blk_f64(blk, "count", 24.0).max(0.0);
                        radius = blk_f64(blk, "radius", radius);
                        seed = blk_f64(blk, "seed", seed);
                    }
                    "lines" => {
                        has_lines = true;
                        max_dist = blk_f64(blk, "maxDistance", max_dist);
                        beam_thick = blk_f64(blk, "beamThickness", beam_thick);
                    }
                    "turbulence" => {
                        turb_sum += blk_f64(blk, "turbulence", 1.5).max(0.0);
                    }
                    "render" | "appearance" => {
                        has_render = true;
                        color = blk_vec3(blk, "lineColor", color);
                        opacity = blk_f64(blk, "opacity", opacity);
                    }
                    _ => {}
                }
            }
            out.push((
                "count",
                json!(if has_points {
                    count_sum.clamp(2.0, 512.0)
                } else {
                    0.0
                }),
            ));
            if has_points {
                out.push(("radius", json!(radius)));
                out.push(("seed", json!(seed)));
            }
            out.push(("maxDistance", json!(if has_lines { max_dist } else { 0.0 })));
            if has_lines {
                out.push(("beamThickness", json!(beam_thick)));
            }
            out.push(("turbulence", json!(turb_sum)));
            if has_render {
                out.push(("lineColor", json!(color)));
                out.push(("opacity", json!(opacity)));
            } else {
                out.push(("opacity", json!(0.0)));
            }
        }
    }
    out
}

/// Persists `blocks` JSON AND syncs active enabled block parameters into live particle
/// properties in a single atomic Undo transaction.
pub fn commit_particle_blocks(
    session: &mut Session,
    object_id: &str,
    kind: ParticleDesignerKind,
    blocks: &[ParticleDesignerBlock],
    label: &str,
    local_frame: f64,
) {
    let mut batch = sync_particle_blocks_to_live_props(kind, blocks);
    batch.insert(
        0,
        (
            "blocks",
            Value::Array(blocks.iter().map(ParticleDesignerBlock::to_json).collect()),
        ),
    );
    apply_obj_batch_tx(session, object_id, label, &batch, local_frame);
}

fn designer_presets_for(
    kind: ParticleDesignerKind,
) -> Vec<(&'static str, Vec<(&'static str, Value)>)> {
    match kind {
        ParticleDesignerKind::Particular => vec![
            (
                "Spark Fountain",
                vec![
                    ("count", json!(180)),
                    ("rate", json!(60.0)),
                    ("lifetime", json!(2.4)),
                    ("particleSize", json!(2.2)),
                    ("velocity", json!(20.0)),
                    ("gravity", json!(-9.8)),
                    ("emitterSize", json!([3.0, 2.0, 3.0])),
                    ("particleColor", json!([1.0, 0.65, 0.2])),
                    ("opacity", json!(1.0)),
                ],
            ),
            (
                "Floating Dust",
                vec![
                    ("count", json!(120)),
                    ("rate", json!(20.0)),
                    ("lifetime", json!(5.0)),
                    ("particleSize", json!(1.5)),
                    ("velocity", json!(2.5)),
                    ("gravity", json!(0.2)),
                    ("emitterSize", json!([28.0, 18.0, 28.0])),
                    ("particleColor", json!([0.88, 0.94, 1.0])),
                    ("opacity", json!(0.75)),
                ],
            ),
            (
                "Neon Burst",
                vec![
                    ("count", json!(240)),
                    ("rate", json!(90.0)),
                    ("lifetime", json!(1.8)),
                    ("particleSize", json!(3.5)),
                    ("velocity", json!(28.0)),
                    ("gravity", json!(0.0)),
                    ("emitterSize", json!([4.0, 4.0, 4.0])),
                    ("particleColor", json!([0.95, 0.28, 0.8])),
                    ("opacity", json!(0.95)),
                ],
            ),
            (
                "Snowfall",
                vec![
                    ("count", json!(200)),
                    ("rate", json!(35.0)),
                    ("lifetime", json!(6.0)),
                    ("particleSize", json!(2.0)),
                    ("velocity", json!(3.0)),
                    ("gravity", json!(-2.2)),
                    ("emitterSize", json!([40.0, 6.0, 30.0])),
                    ("particleColor", json!([0.96, 0.98, 1.0])),
                    ("opacity", json!(0.9)),
                ],
            ),
        ],
        ParticleDesignerKind::Form => vec![
            (
                "Crystal Cube",
                vec![
                    ("particlesX", json!(7)),
                    ("particlesY", json!(7)),
                    ("particlesZ", json!(5)),
                    ("sizeX", json!(24.0)),
                    ("sizeY", json!(24.0)),
                    ("sizeZ", json!(18.0)),
                    ("displace", json!(1.5)),
                    ("particleSize", json!(0.9)),
                    ("particleColor", json!([0.45, 0.88, 1.0])),
                ],
            ),
            (
                "Ocean Wave Grid",
                vec![
                    ("particlesX", json!(12)),
                    ("particlesY", json!(2)),
                    ("particlesZ", json!(12)),
                    ("sizeX", json!(42.0)),
                    ("sizeY", json!(4.0)),
                    ("sizeZ", json!(42.0)),
                    ("displace", json!(5.5)),
                    ("particleSize", json!(0.75)),
                    ("particleColor", json!([0.2, 0.72, 1.0])),
                ],
            ),
            (
                "Audio Ribbon",
                vec![
                    ("particlesX", json!(14)),
                    ("particlesY", json!(4)),
                    ("particlesZ", json!(2)),
                    ("sizeX", json!(48.0)),
                    ("sizeY", json!(10.0)),
                    ("sizeZ", json!(4.0)),
                    ("displace", json!(6.5)),
                    ("particleSize", json!(1.1)),
                    ("particleColor", json!([0.95, 0.32, 0.72])),
                ],
            ),
        ],
        ParticleDesignerKind::Plexus => vec![
            (
                "Neural Mesh",
                vec![
                    ("count", json!(32)),
                    ("radius", json!(20.0)),
                    ("maxDistance", json!(16.0)),
                    ("beamThickness", json!(0.22)),
                    ("lineColor", json!([0.45, 0.88, 1.0])),
                    ("opacity", json!(0.9)),
                ],
            ),
            (
                "Dense Constellation",
                vec![
                    ("count", json!(48)),
                    ("radius", json!(26.0)),
                    ("maxDistance", json!(19.0)),
                    ("beamThickness", json!(0.18)),
                    ("lineColor", json!([0.92, 0.95, 1.0])),
                    ("opacity", json!(0.85)),
                ],
            ),
            (
                "Laser Web",
                vec![
                    ("count", json!(20)),
                    ("radius", json!(16.0)),
                    ("maxDistance", json!(22.0)),
                    ("beamThickness", json!(0.4)),
                    ("lineColor", json!([1.0, 0.35, 0.75])),
                    ("opacity", json!(1.0)),
                ],
            ),
        ],
    }
}

fn paint_block_tile_graphic(painter: &egui::Painter, rect: Rect, block_key: &str, accent: Color32) {
    painter.rect_filled(rect, CornerRadius::same(2), BG_INPUT);
    let c = rect.center();
    let r = rect.width().min(rect.height()) * 0.32;
    match block_key {
        "emitter" | "points" => {
            painter.circle_stroke(c, r, Stroke::new(1.2, accent));
            for i in 0..6 {
                let a = i as f32 * std::f32::consts::TAU / 6.0;
                painter.circle_filled(
                    Pos2::new(c.x + a.cos() * r * 0.65, c.y + a.sin() * r * 0.65),
                    2.2,
                    TEXT_BRIGHT,
                );
            }
        }
        "grid" => {
            for gx in -1..=1 {
                for gy in -1..=1 {
                    painter.circle_filled(
                        Pos2::new(c.x + gx as f32 * r * 0.6, c.y + gy as f32 * r * 0.6),
                        2.2,
                        accent,
                    );
                }
            }
        }
        "lines" => {
            let pts = [
                Pos2::new(c.x - r, c.y - r * 0.5),
                Pos2::new(c.x + r * 0.8, c.y - r * 0.6),
                Pos2::new(c.x - r * 0.2, c.y + r * 0.8),
                Pos2::new(c.x + r, c.y + r * 0.4),
            ];
            for i in 0..pts.len() {
                for j in (i + 1)..pts.len() {
                    painter.line_segment(
                        [pts[i], pts[j]],
                        Stroke::new(1.0, color32_with_alpha(accent, 180)),
                    );
                }
                painter.circle_filled(pts[i], 2.5, TEXT_BRIGHT);
            }
        }
        "physics" | "displace" | "turbulence" => {
            let mut pts = Vec::with_capacity(9);
            for i in 0..=8 {
                let t = i as f32 / 8.0;
                let x = rect.left() + 8.0 + (rect.width() - 16.0) * t;
                let y = c.y + (t * std::f32::consts::TAU).sin() * r * 0.65;
                pts.push(Pos2::new(x, y));
            }
            painter.add(egui::Shape::line(pts, Stroke::new(1.6, accent)));
        }
        _ => {
            painter.circle_filled(c, r * 0.7, color32_with_alpha(accent, 90));
            painter.circle_filled(c, r * 0.35, accent);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn block_num_row(
    ui: &mut Ui,
    obj_id: &str,
    blk: &mut ParticleDesignerBlock,
    key: &str,
    label: &str,
    default: f64,
    cfg: ScrubConfig,
    theme: &Cm3Theme,
) -> bool {
    let mut v = blk_f64(blk, key, default);
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.add_sized(
            Vec2::new(112.0, 20.0),
            egui::Label::new(
                egui::RichText::new(label)
                    .font(theme.font(FONT_SMALL))
                    .color(TEXT_PRIMARY),
            )
            .truncate(),
        );
        let ev = cm3_scrub_number(
            ui,
            Id::unique(("tc_blk_num", obj_id, &blk.id, key)),
            &mut v,
            cfg,
            theme,
        );
        if ev.changed {
            blk.params.insert(key.to_owned(), json!(v));
            changed = true;
        }
    });
    changed
}

#[allow(clippy::too_many_arguments)]
fn block_vec3_row(
    ui: &mut Ui,
    obj_id: &str,
    blk: &mut ParticleDesignerBlock,
    key: &str,
    label: &str,
    default: [f64; 3],
    cfg: ScrubConfig,
    theme: &Cm3Theme,
) -> bool {
    let mut v = blk_vec3(blk, key, default);
    let mut changed = false;
    ui.label(
        egui::RichText::new(label)
            .font(theme.font(FONT_SMALL))
            .color(TEXT_MUTED),
    );
    ui.horizontal(|ui| {
        for (axis, axis_lbl) in ["X", "Y", "Z"].iter().enumerate() {
            ui.label(
                egui::RichText::new(*axis_lbl)
                    .font(theme.font(FONT_COMPACT))
                    .color(TEXT_FAINT),
            );
            let ev = cm3_scrub_number(
                ui,
                Id::unique(("tc_blk_v3", obj_id, &blk.id, key, axis)),
                &mut v[axis],
                cfg,
                theme,
            );
            if ev.changed {
                changed = true;
            }
        }
        if changed {
            blk.params.insert(key.to_owned(), json!(v));
        }
    });
    changed
}

#[allow(clippy::too_many_arguments)]
fn block_color_row(
    ui: &mut Ui,
    obj_id: &str,
    blk: &mut ParticleDesignerBlock,
    key: &str,
    label: &str,
    default: [f64; 3],
    screen: Rect,
    theme: &Cm3Theme,
) -> bool {
    let rgb = blk_vec3(blk, key, default);
    let mut color = Rgba::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, 1.0);
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.add_sized(
            Vec2::new(112.0, 20.0),
            egui::Label::new(
                egui::RichText::new(label)
                    .font(theme.font(FONT_SMALL))
                    .color(TEXT_PRIMARY),
            )
            .truncate(),
        );
        let picker_id = Id::unique(("tc_blk_col", obj_id, &blk.id, key));
        let ev = cm3_hsva_color_picker(ui, picker_id, &mut color, false, false, screen, theme);
        if ev.eyedropper_clicked {
            request_eyedropper_sampler(ui.ctx(), picker_id);
        }
        if ev.changed {
            blk.params.insert(
                key.to_owned(),
                json!([
                    f64::from(color.0[0]),
                    f64::from(color.0[1]),
                    f64::from(color.0[2])
                ]),
            );
            changed = true;
        }
    });
    changed
}

/// Renders the floating 3-column Particle & Light Suite Designer window when `state.open` is true.
pub fn show_particle_designer_window(
    ctx: &egui::Context,
    viewport_rect: Rect,
    floating_windows: &mut FloatingWindowManager,
    state: &mut ParticleDesignerState,
    session: &mut Session,
    theme: &Cm3Theme,
) {
    if !state.open {
        return;
    }
    let Some((obj, local_frame)) =
        find_object3d_in_session(session, &state.object_id).map(|(o, lf)| (o.clone(), lf))
    else {
        state.open = false;
        floating_windows.close("pro_particle_designer");
        return;
    };

    let ctx_screen = ctx.input(|i| i.viewport_rect());
    let screen = if ctx_screen.width() > 100.0 && ctx_screen.height() > 100.0 {
        ctx_screen
    } else {
        viewport_rect
    };
    let (_, default_size) = compute_clamped_editor_window_rect(screen, 860.0, 520.0);
    let spec = FloatingWindowSpec::new("pro_particle_designer", state.kind.title())
        .with_subtitle(if state.object_label.is_empty() {
            "Generative Suite".to_owned()
        } else {
            state.object_label.clone()
        })
        .with_default_size(default_size);

    ensure_pro_window_in_screen(floating_windows, &spec, screen, &mut state.just_opened);

    let mut close_requested = false;
    let mut blocks = read_particle_blocks(&obj, state.kind, local_frame);
    if !blocks.is_empty() && state.selected_block >= blocks.len() {
        state.selected_block = blocks.len() - 1;
    }
    let mut blocks_dirty_label: Option<String> = None;

    let ev = cm3_floating_window(ctx, floating_windows, &spec, screen, theme, |ui| {
        // Top toolbar (Presets & Blocks toggles matching designer.js)
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            if cm3_prop_button(
                ui,
                "Presets",
                if state.show_presets {
                    ButtonVariant::Primary
                } else {
                    ButtonVariant::Normal
                },
                false,
                theme,
            )
            .clicked()
            {
                state.show_presets = !state.show_presets;
            }
            if cm3_prop_button(
                ui,
                "Blocks",
                if state.show_params {
                    ButtonVariant::Primary
                } else {
                    ButtonVariant::Normal
                },
                false,
                theme,
            )
            .clicked()
            {
                state.show_params = !state.show_params;
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if cm3_prop_button(ui, "Done", ButtonVariant::Primary, false, theme).clicked() {
                    close_requested = true;
                }
            });
        });

        ui.add_space(6.0);
        let body_h = (ui.available_height() - 4.0).max(160.0);
        let presets_w = if state.show_presets { 190.0 } else { 0.0 };
        let params_w = if state.show_params { 270.0 } else { 0.0 };
        let gaps = (if state.show_presets { 8.0 } else { 0.0 })
            + (if state.show_params { 8.0 } else { 0.0 });
        let center_w = (ui.available_width() - presets_w - params_w - gaps).max(180.0);

        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 8.0;

            // Left: Presets + Color Palettes
            if state.show_presets {
                ui.allocate_ui_with_layout(
                    Vec2::new(presets_w, body_h),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        egui::Frame::NONE
                            .fill(BG_SUNKEN)
                            .stroke(Stroke::new(1.0, BORDER_PANEL))
                            .corner_radius(CornerRadius::same(3))
                            .inner_margin(egui::Margin::same(6))
                            .show(ui, |ui| {
                                ui.set_width((presets_w - 14.0).max(120.0));
                                ui.set_min_height((body_h - 14.0).max(100.0));
                                ui.set_max_height((body_h - 14.0).max(100.0));
                                ScrollArea::vertical()
                                    .id_salt(("tc_des_presets", &obj.id))
                                    .auto_shrink([false, false])
                                    .show(ui, |ui| {
                                        ui.label(
                                            egui::RichText::new("Presets")
                                                .font(theme.font(FONT_SMALL))
                                                .color(TEXT_MUTED),
                                        );
                                        ui.add_space(3.0);
                                        for (name, mut batch) in designer_presets_for(state.kind) {
                                            if cm3_prop_button(
                                                ui,
                                                name,
                                                ButtonVariant::Normal,
                                                true,
                                                theme,
                                            )
                                            .clicked()
                                            {
                                                batch.push(("blocks", Value::Null));
                                                apply_obj_batch_tx(
                                                    session,
                                                    &obj.id,
                                                    &format!("Designer preset: {name}"),
                                                    &batch,
                                                    local_frame,
                                                );
                                            }
                                            ui.add_space(2.0);
                                        }
                                        ui.add_space(8.0);
                                        ui.label(
                                            egui::RichText::new("Palette")
                                                .font(theme.font(FONT_SMALL))
                                                .color(TEXT_MUTED),
                                        );
                                        ui.add_space(3.0);
                                        for &(name, rgb, stops) in DESIGNER_PALETTES {
                                            let (row_rect, resp) = ui.allocate_exact_size(
                                                Vec2::new(ui.available_width(), 22.0),
                                                Sense::click(),
                                            );
                                            if resp.hovered() {
                                                ui.painter().rect_filled(
                                                    row_rect,
                                                    CornerRadius::same(2),
                                                    BG_HOVER,
                                                );
                                            }
                                            let sw_w = 54.0;
                                            let sw_rect = Rect::from_min_size(
                                                Pos2::new(
                                                    row_rect.left() + 2.0,
                                                    row_rect.top() + 4.0,
                                                ),
                                                Vec2::new(sw_w, 14.0),
                                            );
                                            for (si, &c) in stops.iter().enumerate() {
                                                let seg = Rect::from_min_size(
                                                    Pos2::new(
                                                        sw_rect.left() + (sw_w / 3.0) * si as f32,
                                                        sw_rect.top(),
                                                    ),
                                                    Vec2::new(sw_w / 3.0, 14.0),
                                                );
                                                ui.painter().rect_filled(
                                                    seg,
                                                    CornerRadius::ZERO,
                                                    c,
                                                );
                                            }
                                            ui.painter().rect_stroke(
                                                sw_rect,
                                                CornerRadius::same(1),
                                                Stroke::new(1.0, BORDER_CONTROL),
                                                egui::StrokeKind::Inside,
                                            );
                                            ui.painter().text(
                                                Pos2::new(
                                                    sw_rect.right() + 6.0,
                                                    row_rect.center().y,
                                                ),
                                                egui::Align2::LEFT_CENTER,
                                                name,
                                                theme.font(FONT_COMPACT),
                                                TEXT_PRIMARY,
                                            );
                                            if resp.clicked() {
                                                apply_designer_palette(
                                                    session,
                                                    &obj,
                                                    state.kind,
                                                    name,
                                                    rgb,
                                                    stops,
                                                    local_frame,
                                                );
                                            }
                                        }
                                    });
                            });
                    },
                );
            }

            // Center: Transport Bar + Block Tiles Grid + Active System Strip
            ui.allocate_ui_with_layout(
                Vec2::new(center_w, body_h),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    egui::Frame::NONE
                        .fill(BG_PANEL)
                        .stroke(Stroke::new(1.0, BORDER_PANEL))
                        .corner_radius(CornerRadius::same(3))
                        .show(ui, |ui| {
                            ui.set_width((center_w - 2.0).max(160.0));
                            ui.set_min_height((body_h - 2.0).max(120.0));

                            // Transport strip
                            egui::Frame::NONE
                                .fill(BG_TITLE)
                                .inner_margin(egui::Margin::symmetric(8, 6))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        let total_f =
                                            session.active_sequence().length_frames.max(1.0);
                                        if cm3_prop_button(
                                            ui,
                                            "◀",
                                            ButtonVariant::Normal,
                                            false,
                                            theme,
                                        )
                                        .clicked()
                                        {
                                            session.playback.step_frame(-1.0, total_f);
                                        }
                                        if cm3_prop_button(
                                            ui,
                                            if session.playback.is_playing() {
                                                "Pause"
                                            } else {
                                                "Play"
                                            },
                                            ButtonVariant::Normal,
                                            false,
                                            theme,
                                        )
                                        .clicked()
                                        {
                                            let _ = session.execute(Command::TogglePlay);
                                        }
                                        if cm3_prop_button(
                                            ui,
                                            "▶",
                                            ButtonVariant::Normal,
                                            false,
                                            theme,
                                        )
                                        .clicked()
                                        {
                                            session.playback.step_frame(1.0, total_f);
                                        }
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                let cur =
                                                    session.playback.current_frame.round().max(0.0)
                                                        as u64;
                                                ui.label(
                                                    egui::RichText::new(format!("{cur:04}"))
                                                        .font(theme.font(FONT_BODY))
                                                        .color(theme.value()),
                                                );
                                            },
                                        );
                                    });
                                });

                            ui.add_space(6.0);
                            // Editable Blocks Toolbar (+ Add Block) & Stack Cards
                            egui::Frame::NONE
                                .inner_margin(egui::Margin::symmetric(8, 4))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        egui::ComboBox::from_id_salt((
                                            "tc_add_block",
                                            &obj.id,
                                        ))
                                        .selected_text("+ Add Block")
                                        .width(128.0)
                                        .show_ui(ui, |ui| {
                                            for &(bkey, bname) in state.kind.blocks() {
                                                if ui.selectable_label(false, bname).clicked() {
                                                    let serial = blocks.len() + 1;
                                                    blocks.push(
                                                        ParticleDesignerBlock::new_default(
                                                            state.kind, bkey, serial,
                                                        ),
                                                    );
                                                    state.selected_block =
                                                        blocks.len().saturating_sub(1);
                                                    blocks_dirty_label = Some(format!(
                                                        "Add Designer block: {bname}"
                                                    ));
                                                }
                                            }
                                        });
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "{} block{}",
                                                blocks.len(),
                                                if blocks.len() == 1 { "" } else { "s" }
                                            ))
                                            .font(theme.font(FONT_COMPACT))
                                            .color(TEXT_MUTED),
                                        );
                                    });
                                    ui.add_space(6.0);

                                    let mut block_action: Option<(usize, &'static str)> = None;
                                    let block_count = blocks.len();
                                    ScrollArea::vertical()
                                        .id_salt(("tc_des_blocks_scroll", &obj.id))
                                        .max_height((body_h - 86.0).max(80.0))
                                        .auto_shrink([false, false])
                                        .show(ui, |ui| {
                                            if blocks.is_empty() {
                                                ui.label(
                                                    egui::RichText::new(
                                                        "No blocks active. Click '+ Add Block' to add one.",
                                                    )
                                                    .font(theme.font(FONT_SMALL))
                                                    .color(TEXT_FAINT),
                                                );
                                            }
                                            #[allow(clippy::needless_range_loop)]
                                            for idx in 0..block_count {
                                                let selected = idx == state.selected_block;
                                                let enabled = blocks[idx].enabled;
                                                let bkey = blocks[idx].kind.clone();
                                                let bname = blocks[idx].label.clone();
                                                egui::Frame::NONE
                                                    .fill(if selected {
                                                        BG_BUTTON
                                                    } else {
                                                        BG_TITLE
                                                    })
                                                    .stroke(Stroke::new(
                                                        if selected { 1.5 } else { 1.0 },
                                                        if selected {
                                                            theme.accent()
                                                        } else {
                                                            BORDER_PANEL
                                                        },
                                                    ))
                                                    .corner_radius(CornerRadius::same(3))
                                                    .inner_margin(egui::Margin::symmetric(6, 5))
                                                    .show(ui, |ui| {
                                                        ui.horizontal(|ui| {
                                                            ui.spacing_mut().item_spacing.x = 5.0;
                                                            if ui
                                                                .checkbox(
                                                                    &mut blocks[idx].enabled,
                                                                    "",
                                                                )
                                                                .on_hover_text("Enable / Disable block")
                                                                .changed()
                                                            {
                                                                blocks_dirty_label = Some(format!(
                                                                    "Toggle Designer block: {bname}"
                                                                ));
                                                            }
                                                            let (g_rect, g_resp) =
                                                                ui.allocate_exact_size(
                                                                    Vec2::new(34.0, 24.0),
                                                                    Sense::click(),
                                                                );
                                                            paint_block_tile_graphic(
                                                                ui.painter(),
                                                                g_rect,
                                                                &bkey,
                                                                if enabled {
                                                                    theme.accent()
                                                                } else {
                                                                    TEXT_FAINT
                                                                },
                                                            );
                                                            if g_resp.clicked() {
                                                                state.selected_block = idx;
                                                            }
                                                            if ui
                                                                .selectable_label(
                                                                    selected,
                                                                    egui::RichText::new(
                                                                        bname.as_str(),
                                                                    )
                                                                    .font(theme.font(FONT_SMALL))
                                                                    .color(if !enabled {
                                                                        TEXT_FAINT
                                                                    } else if selected {
                                                                        theme.value()
                                                                    } else {
                                                                        TEXT_PRIMARY
                                                                    }),
                                                                )
                                                                .clicked()
                                                            {
                                                                state.selected_block = idx;
                                                            }
                                                            ui.with_layout(
                                                                egui::Layout::right_to_left(
                                                                    egui::Align::Center,
                                                                ),
                                                                |ui| {
                                                                    if ui
                                                                        .small_button("×")
                                                                        .on_hover_text("Delete block")
                                                                        .clicked()
                                                                    {
                                                                        block_action =
                                                                            Some((idx, "del"));
                                                                    }
                                                                    if ui
                                                                        .small_button("D")
                                                                        .on_hover_text(
                                                                            "Duplicate block",
                                                                        )
                                                                        .clicked()
                                                                    {
                                                                        block_action =
                                                                            Some((idx, "dup"));
                                                                    }
                                                                    if idx + 1 < block_count
                                                                        && ui
                                                                            .small_button("↓")
                                                                            .on_hover_text(
                                                                                "Move down",
                                                                            )
                                                                            .clicked()
                                                                    {
                                                                        block_action =
                                                                            Some((idx, "down"));
                                                                    }
                                                                    if idx > 0
                                                                        && ui
                                                                            .small_button("↑")
                                                                            .on_hover_text(
                                                                                "Move up",
                                                                            )
                                                                            .clicked()
                                                                    {
                                                                        block_action =
                                                                            Some((idx, "up"));
                                                                    }
                                                                },
                                                            );
                                                        });
                                                    });
                                                ui.add_space(4.0);
                                            }
                                        });

                                    if let Some((idx, cmd)) = block_action {
                                        match cmd {
                                            "up" if idx > 0 => {
                                                blocks.swap(idx, idx - 1);
                                                state.selected_block = idx - 1;
                                                blocks_dirty_label =
                                                    Some("Move Designer block up".into());
                                            }
                                            "down" if idx + 1 < blocks.len() => {
                                                blocks.swap(idx, idx + 1);
                                                state.selected_block = idx + 1;
                                                blocks_dirty_label =
                                                    Some("Move Designer block down".into());
                                            }
                                            "dup" if idx < blocks.len() => {
                                                let mut copy = blocks[idx].clone();
                                                copy.id = format!("blk_{}", blocks.len() + 1);
                                                copy.label = format!("{} Copy", copy.label);
                                                blocks.insert(idx + 1, copy);
                                                state.selected_block = idx + 1;
                                                blocks_dirty_label =
                                                    Some("Duplicate Designer block".into());
                                            }
                                            "del" if idx < blocks.len() => {
                                                blocks.remove(idx);
                                                state.selected_block = if blocks.is_empty() {
                                                    0
                                                } else {
                                                    idx.min(blocks.len() - 1)
                                                };
                                                blocks_dirty_label =
                                                    Some("Delete Designer block".into());
                                            }
                                            _ => {}
                                        }
                                    }
                                });
                        });
                },
            );

            // Right: Selected Block Parameters
            if state.show_params {
                ui.allocate_ui_with_layout(
                    Vec2::new(params_w, body_h),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        egui::Frame::NONE
                            .fill(BG_PANEL)
                            .stroke(Stroke::new(1.0, BORDER_PANEL))
                            .corner_radius(CornerRadius::same(3))
                            .inner_margin(egui::Margin::same(8))
                            .show(ui, |ui| {
                                ui.set_width((params_w - 18.0).max(160.0));
                                ui.set_min_height((body_h - 18.0).max(100.0));
                                ui.set_max_height((body_h - 18.0).max(100.0));
                                if let Some(blk) = blocks.get_mut(state.selected_block) {
                                    let blk_id = blk.id.clone();
                                    let bkey = blk.kind.clone();
                                    let bname = blk.label.clone();
                                    ui.label(
                                        egui::RichText::new(bname.as_str())
                                            .font(theme.font(FONT_TITLE))
                                            .color(TEXT_STRONG),
                                    );
                                    ui.add_space(6.0);
                                    let mut param_changed = false;
                                    ScrollArea::vertical()
                                        .id_salt(("tc_des_params", &obj.id, &blk_id))
                                        .auto_shrink([false, false])
                                        .show(ui, |ui| {
                                            ui.spacing_mut().item_spacing.y = 4.0;
                                            match (state.kind, bkey.as_str()) {
                                                (ParticleDesignerKind::Particular, "emitter") => {
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "count",
                                                        "Particle count",
                                                        64.0,
                                                        ScrubConfig::new(1.0, 2048.0, 2.0, 0),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "rate",
                                                        "Particles / sec",
                                                        30.0,
                                                        ScrubConfig::new(1.0, 1000.0, 1.0, 1),
                                                        theme,
                                                    );
                                                    param_changed |= block_vec3_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "emitterSize",
                                                        "Emitter size",
                                                        [6.0, 6.0, 6.0],
                                                        ScrubConfig::new(0.0, 500.0, 0.25, 2),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "seed",
                                                        "Seed",
                                                        1.0,
                                                        ScrubConfig::new(0.0, 100_000.0, 1.0, 0),
                                                        theme,
                                                    );
                                                }
                                                (ParticleDesignerKind::Particular, "particle") => {
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "lifetime",
                                                        "Life (sec)",
                                                        3.0,
                                                        ScrubConfig::new(0.1, 60.0, 0.1, 2),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "particleSize",
                                                        "Size",
                                                        3.0,
                                                        ScrubConfig::new(0.1, 100.0, 0.1, 2),
                                                        theme,
                                                    );
                                                }
                                                (ParticleDesignerKind::Particular, "physics") => {
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "velocity",
                                                        "Velocity",
                                                        12.0,
                                                        ScrubConfig::new(-200.0, 200.0, 0.5, 2),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "gravity",
                                                        "Gravity",
                                                        -4.0,
                                                        ScrubConfig::new(-200.0, 200.0, 0.25, 2),
                                                        theme,
                                                    );
                                                }
                                                (_, "turbulence") => {
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "turbulence",
                                                        "Turbulence",
                                                        1.5,
                                                        ScrubConfig::new(0.0, 50.0, 0.1, 2),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "turbulenceScale",
                                                        "Noise scale",
                                                        2.0,
                                                        ScrubConfig::new(0.1, 20.0, 0.1, 2),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "turbulenceSpeed",
                                                        "Evolution speed",
                                                        1.0,
                                                        ScrubConfig::new(0.0, 10.0, 0.05, 2),
                                                        theme,
                                                    );
                                                }
                                                (
                                                    ParticleDesignerKind::Particular,
                                                    "render" | "appearance",
                                                ) => {
                                                    param_changed |= block_color_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "particleColor",
                                                        "Color",
                                                        [0.4, 0.85, 1.0],
                                                        screen,
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "opacity",
                                                        "Opacity",
                                                        1.0,
                                                        ScrubConfig::new(0.0, 1.0, 0.02, 2),
                                                        theme,
                                                    );
                                                }
                                                (ParticleDesignerKind::Form, "grid") => {
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "particlesX",
                                                        "Particles in X",
                                                        6.0,
                                                        ScrubConfig::new(1.0, 64.0, 1.0, 0),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "particlesY",
                                                        "Particles in Y",
                                                        6.0,
                                                        ScrubConfig::new(1.0, 64.0, 1.0, 0),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "particlesZ",
                                                        "Particles in Z",
                                                        4.0,
                                                        ScrubConfig::new(1.0, 64.0, 1.0, 0),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "sizeX",
                                                        "Size X",
                                                        24.0,
                                                        ScrubConfig::new(0.5, 500.0, 0.5, 1),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "sizeY",
                                                        "Size Y",
                                                        24.0,
                                                        ScrubConfig::new(0.5, 500.0, 0.5, 1),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "sizeZ",
                                                        "Size Z",
                                                        16.0,
                                                        ScrubConfig::new(0.5, 500.0, 0.5, 1),
                                                        theme,
                                                    );
                                                }
                                                (ParticleDesignerKind::Form, "displace") => {
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "displace",
                                                        "Displace",
                                                        2.5,
                                                        ScrubConfig::new(-100.0, 100.0, 0.1, 2),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "seed",
                                                        "Seed",
                                                        1.0,
                                                        ScrubConfig::new(0.0, 100_000.0, 1.0, 0),
                                                        theme,
                                                    );
                                                }
                                                (ParticleDesignerKind::Form, "particle") => {
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "particleSize",
                                                        "Particle size",
                                                        0.85,
                                                        ScrubConfig::new(0.05, 50.0, 0.05, 2),
                                                        theme,
                                                    );
                                                }
                                                (
                                                    ParticleDesignerKind::Form,
                                                    "render" | "appearance",
                                                ) => {
                                                    param_changed |= block_color_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "particleColor",
                                                        "Color",
                                                        [0.85, 0.92, 1.0],
                                                        screen,
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "opacity",
                                                        "Opacity",
                                                        1.0,
                                                        ScrubConfig::new(0.0, 1.0, 0.02, 2),
                                                        theme,
                                                    );
                                                }
                                                (ParticleDesignerKind::Plexus, "points") => {
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "count",
                                                        "Point count",
                                                        24.0,
                                                        ScrubConfig::new(2.0, 256.0, 1.0, 0),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "radius",
                                                        "Cloud radius",
                                                        18.0,
                                                        ScrubConfig::new(0.5, 500.0, 0.5, 1),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "seed",
                                                        "Seed",
                                                        1.0,
                                                        ScrubConfig::new(0.0, 100_000.0, 1.0, 0),
                                                        theme,
                                                    );
                                                }
                                                (ParticleDesignerKind::Plexus, "lines") => {
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "maxDistance",
                                                        "Link distance",
                                                        15.3,
                                                        ScrubConfig::new(0.1, 500.0, 0.5, 1),
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "beamThickness",
                                                        "Beam thickness",
                                                        0.25,
                                                        ScrubConfig::new(0.02, 10.0, 0.02, 2),
                                                        theme,
                                                    );
                                                }
                                                (
                                                    ParticleDesignerKind::Plexus,
                                                    "render" | "appearance",
                                                ) => {
                                                    param_changed |= block_color_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "lineColor",
                                                        "Line color",
                                                        [0.85, 0.92, 1.0],
                                                        screen,
                                                        theme,
                                                    );
                                                    param_changed |= block_num_row(
                                                        ui,
                                                        &obj.id,
                                                        blk,
                                                        "opacity",
                                                        "Opacity",
                                                        1.0,
                                                        ScrubConfig::new(0.0, 1.0, 0.02, 2),
                                                        theme,
                                                    );
                                                }
                                                _ => {}
                                            }
                                        });
                                    if param_changed {
                                        blocks_dirty_label = Some(format!("Edit {bname} block"));
                                    }
                                } else {
                                    ui.label(
                                        egui::RichText::new("Select or add a block to edit.")
                                            .font(theme.font(FONT_SMALL))
                                            .color(TEXT_MUTED),
                                    );
                                }
                            });
                    },
                );
            }
        });
    });

    if let Some(label) = blocks_dirty_label {
        commit_particle_blocks(session, &obj.id, state.kind, &blocks, &label, local_frame);
    }

    clamp_floating_window_in_screen(floating_windows, &spec, screen);

    if ev.closed && ctx.input(|i| i.key_pressed(Key::Escape)) {
        mark_floating_editor_escape_consumed(ctx);
    } else if !ev.closed
        && consume_escape_for_floating_editor(ctx, floating_windows, "pro_particle_designer")
    {
        close_requested = true;
    }

    if close_requested || ev.closed {
        state.open = false;
        floating_windows.close("pro_particle_designer");
    }
}

#[cfg(test)]
mod scene_designers_tests {
    use super::*;

    fn make_session_with_object3d(kind: &str) -> (Session, String) {
        let mut session = Session::new();
        let track_id = session.document().video_tracks[0].id.clone();
        session
            .execute(Command::AddClip {
                track_id,
                kind: "scene".into(),
                start: 0.0,
                length: 120.0,
                name: Some("Scene".into()),
            })
            .expect("add clip");
        let clip_id = session.document().video_tracks[0]
            .clips
            .last()
            .expect("clip")
            .id
            .clone();
        session
            .execute(Command::AddObject3D {
                clip_id,
                kind: kind.into(),
                name: Some("TestObj".into()),
                parent_id: None,
            })
            .expect("add object3d");
        let obj_id = session.document().video_tracks[0]
            .clips
            .last()
            .expect("clip")
            .objects_3d
            .last()
            .expect("obj")
            .id
            .clone();
        (session, obj_id)
    }

    #[test]
    fn optical_flare_stack_supports_all_12_types_serializes_and_syncs_uniforms() {
        assert_eq!(FLARE_ELEMENT_TYPES.len(), 12);
        let (mut session, obj_id) = make_session_with_object3d("zoidium:optical-flares/flare");
        let obj = find_object3d_in_session(&session, &obj_id)
            .expect("flare obj")
            .0
            .clone();

        let mut stack = vec![
            OpticalFlareElement::from_type(0, 1), // Glow
            OpticalFlareElement::from_type(3, 2), // Streak
            OpticalFlareElement::from_type(1, 3), // Multi Iris
            OpticalFlareElement::from_type(7, 4), // Sparkle
            OpticalFlareElement::from_type(9, 5), // Hoop
        ];
        stack[1].color = [0.2, 0.7, 1.0];
        stack[1].position = 0.25;
        commit_optical_flare_stack(&mut session, &obj, &stack, "Commit stack", 0.0);

        let updated = find_object3d_in_session(&session, &obj_id)
            .expect("updated flare")
            .0
            .clone();
        let roundtripped = read_optical_flare_stack(&updated, 0.0);
        assert_eq!(roundtripped.len(), 5);
        assert_eq!(roundtripped[1].element_type, 3);
        assert!((roundtripped[1].position - 0.25).abs() < 1e-6);

        assert!(read_obj_f64(&updated, "anamorphic", 0.0, 0.0) > 0.5);
        assert_eq!(read_obj_f64(&updated, "ghosts", 0.0, 0.0).round() as u32, 2);
        assert_eq!(
            read_obj_f64(&updated, "starburstRays", 0.0, 0.0).round() as u32,
            14
        );
        assert!(read_obj_f64(&updated, "ringIntensity", 0.0, 0.0) > 0.5);

        // Soloing only the Streak element must zero ghosts/rays/rings while keeping anamorphic!
        stack[1].solo = true;
        commit_optical_flare_stack(&mut session, &updated, &stack, "Solo streak", 0.0);
        let soloed = find_object3d_in_session(&session, &obj_id)
            .expect("soloed flare")
            .0
            .clone();
        assert!(read_obj_f64(&soloed, "anamorphic", 0.0, 0.0) > 0.5);
        assert_eq!(read_obj_f64(&soloed, "ghosts", 0.0, 0.0).round() as u32, 0);
        assert_eq!(
            read_obj_f64(&soloed, "starburstRays", 0.0, 0.0).round() as u32,
            0
        );
        assert!((read_obj_f64(&soloed, "ringIntensity", 1.0, 0.0)).abs() < 1e-6);
    }

    #[test]
    fn particle_designer_palette_writes_primary_and_three_stop_color_over_life() {
        let (mut session, obj_id) = make_session_with_object3d("zoidium:trapcode-suite/particular");
        let obj = find_object3d_in_session(&session, &obj_id)
            .expect("particular obj")
            .0
            .clone();

        let (name, rgb, stops) = DESIGNER_PALETTES[1]; // Solar Ember
        apply_designer_palette(
            &mut session,
            &obj,
            ParticleDesignerKind::Particular,
            name,
            rgb,
            stops,
            0.0,
        );

        let updated = find_object3d_in_session(&session, &obj_id)
            .expect("updated particular")
            .0
            .clone();
        let primary = read_obj_color(&updated, "particleColor", [0.0; 3], 0.0);
        assert!((primary[0] - rgb[0]).abs() < 1e-5);
        assert!((primary[1] - rgb[1]).abs() < 1e-5);
        assert!((primary[2] - rgb[2]).abs() < 1e-5);

        for key in ["particles.colorOverLife", "colorOverLife"] {
            let val = read_obj_prop(updated.properties.get(key), 0.0)
                .unwrap_or_else(|| panic!("expected `{key}` property"));
            let arr = val.as_array().expect("gradient stop array");
            assert_eq!(arr.len(), 3);
            assert!((arr[0]["t"].as_f64().unwrap() - 0.0).abs() < 1e-6);
            assert!((arr[1]["t"].as_f64().unwrap() - 0.5).abs() < 1e-6);
            assert!((arr[2]["t"].as_f64().unwrap() - 1.0).abs() < 1e-6);
            assert_eq!(arr[0]["color"].as_array().unwrap().len(), 3);
        }
    }

    #[test]
    fn particle_designer_blocks_add_duplicate_toggle_reorder_delete_and_sync_live_props() {
        let (mut session, obj_id) = make_session_with_object3d("zoidium:trapcode-suite/particular");
        let obj = find_object3d_in_session(&session, &obj_id)
            .expect("particular obj")
            .0
            .clone();

        let mut blocks = read_particle_blocks(&obj, ParticleDesignerKind::Particular, 0.0);
        let init_len = blocks.len();
        assert!(init_len >= 4);

        // Add a Turbulence block
        let mut turb =
            ParticleDesignerBlock::new_default(ParticleDesignerKind::Particular, "turbulence", 10);
        turb.params.insert("turbulence".into(), json!(3.5));
        blocks.push(turb);

        // Duplicate Emitter block to double emitted particle count
        let mut dup_emitter = blocks[0].clone();
        dup_emitter.id = "blk_dup_emitter".into();
        let base_count = blk_f64(&blocks[0], "count", 64.0);
        blocks.insert(1, dup_emitter);

        // Disable Physics block -> live gravity must sync to 0.0
        for blk in &mut blocks {
            if blk.kind == "physics" {
                blk.enabled = false;
            }
        }

        // Reorder first two blocks cleanly
        blocks.swap(0, 1);

        commit_particle_blocks(
            &mut session,
            &obj_id,
            ParticleDesignerKind::Particular,
            &blocks,
            "Update blocks",
            0.0,
        );

        let updated = find_object3d_in_session(&session, &obj_id)
            .expect("updated particular")
            .0
            .clone();
        let reloaded = read_particle_blocks(&updated, ParticleDesignerKind::Particular, 0.0);
        assert_eq!(reloaded.len(), init_len + 2);
        assert!((read_obj_f64(&updated, "count", 0.0, 0.0) - base_count * 2.0).abs() < 1e-5);
        assert!((read_obj_f64(&updated, "turbulence", 0.0, 0.0) - 3.5).abs() < 1e-5);
        assert!((read_obj_f64(&updated, "gravity", -4.0, 0.0)).abs() < 1e-5);

        // Delete Turbulence block -> live turbulence resets to 0.0
        let mut after_del = reloaded;
        after_del.retain(|b| b.kind != "turbulence");
        commit_particle_blocks(
            &mut session,
            &obj_id,
            ParticleDesignerKind::Particular,
            &after_del,
            "Delete turbulence block",
            0.0,
        );
        let updated2 = find_object3d_in_session(&session, &obj_id)
            .expect("updated2 particular")
            .0
            .clone();
        assert!((read_obj_f64(&updated2, "turbulence", 1.0, 0.0)).abs() < 1e-5);
    }
}
