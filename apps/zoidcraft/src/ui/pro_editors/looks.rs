//! Color Looks Setup editor state, UI and presets.

use crate::ui::controls::{
    ButtonVariant, FloatingWindowManager, FloatingWindowSpec, cm3_floating_window, cm3_prop_button,
};
use crate::ui::theme::{
    BG_BUTTON, BG_HOVER, BG_INPUT, BG_PANEL, BG_SUNKEN, BORDER_CONTROL, BORDER_PANEL, Cm3Theme,
    DANGER_TEXT, FONT_BODY, FONT_COMPACT, FONT_SMALL, FONT_TITLE, TEXT_FAINT, TEXT_MUTED,
    TEXT_PRIMARY, TEXT_STRONG, color32_with_alpha,
};
use egui::{Color32, CornerRadius, Id, Key, Rect, ScrollArea, Sense, Stroke, TextEdit, Ui, Vec2};
use serde_json::{Map, Value, json};
use zc_session::{Command, Session};

use super::*;

/// 5-stage Color Looks pipeline execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LooksStage {
    Subject,
    Matte,
    Lens,
    Camera,
    Post,
}

impl LooksStage {
    pub const ALL: [Self; 5] = [
        Self::Subject,
        Self::Matte,
        Self::Lens,
        Self::Camera,
        Self::Post,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Subject => "Subject",
            Self::Matte => "Matte",
            Self::Lens => "Lens",
            Self::Camera => "Camera",
            Self::Post => "Post",
        }
    }

    #[must_use]
    pub const fn badge_color(self) -> Color32 {
        match self {
            Self::Subject => Color32::from_rgb(76, 152, 224),
            Self::Matte => Color32::from_rgb(138, 108, 212),
            Self::Lens => Color32::from_rgb(228, 156, 68),
            Self::Camera => Color32::from_rgb(74, 186, 134),
            Self::Post => Color32::from_rgb(218, 92, 118),
        }
    }
}

/// Parameter descriptor for a Color Looks tool.
#[derive(Debug, Clone, Copy)]
pub enum LooksParamKind {
    Float { min: f64, max: f64, step: f64 },
    Bool,
    Choice(&'static [&'static str]),
    ColorRgb,
}

#[derive(Debug, Clone, Copy)]
pub struct LooksParamSpec {
    pub key: &'static str,
    pub label: &'static str,
    pub kind: LooksParamKind,
    pub default_f64: f64,
}

/// Static catalog entry for one of the 29 Color Looks tools.
#[derive(Debug, Clone, Copy)]
pub struct LooksToolSpec {
    pub tool_id: &'static str,
    pub label: &'static str,
    pub stage: LooksStage,
    pub params: &'static [LooksParamSpec],
}

/// One active tool instance in a `magiclooks` effect's `looksInstances` chain.
#[derive(Debug, Clone, PartialEq)]
pub struct LooksToolInstance {
    pub tool_id: String,
    pub enabled: bool,
    pub params: Map<String, Value>,
}

/// Interactive state for the Color Looks Setup window.
#[derive(Debug, Clone)]
pub struct LooksEditorState {
    pub open: bool,
    pub effect_id: String,
    pub effect_label: String,
    pub chain: Vec<LooksToolInstance>,
    pub selected_index: Option<usize>,
    pub picker_open: bool,
    pub picker_search: String,
    pub picker_stage_filter: Option<LooksStage>,
    pub auto_apply: bool,
    pub dirty: bool,
    pub just_opened: bool,
}

impl Default for LooksEditorState {
    fn default() -> Self {
        Self {
            open: false,
            effect_id: String::new(),
            effect_label: String::new(),
            chain: Vec::new(),
            selected_index: None,
            picker_open: false,
            picker_search: String::new(),
            picker_stage_filter: None,
            auto_apply: true,
            dirty: false,
            just_opened: false,
        }
    }
}

impl LooksEditorState {
    /// Opens the Color Looks Setup window for the given `magiclooks` effect ID and `looksInstances` value.
    pub fn open_for_effect(
        &mut self,
        effect_id: impl Into<String>,
        effect_label: impl Into<String>,
        looks_instances_json: Option<&Value>,
    ) {
        self.open = true;
        self.just_opened = true;
        self.effect_id = effect_id.into();
        self.effect_label = effect_label.into();
        self.picker_open = false;
        self.picker_search.clear();
        self.picker_stage_filter = None;
        self.load_from_json(looks_instances_json);
        self.dirty = false;
    }

    /// Parses a `looksInstances` JSON array (or JSON-encoded string).
    pub fn load_from_json(&mut self, value: Option<&Value>) {
        self.chain.clear();
        let owned;
        let arr = match value {
            Some(Value::Array(a)) => Some(a.as_slice()),
            Some(Value::String(s)) => {
                owned = serde_json::from_str::<Value>(s).ok();
                owned.as_ref().and_then(Value::as_array).map(Vec::as_slice)
            }
            _ => None,
        };
        if let Some(items) = arr {
            for item in items {
                let Some(obj) = item.as_object() else {
                    continue;
                };
                let Some(tool_id) = obj
                    .get("toolId")
                    .or_else(|| obj.get("tool_id"))
                    .or_else(|| obj.get("id"))
                    .and_then(Value::as_str)
                else {
                    continue;
                };
                let enabled = obj.get("enabled").and_then(Value::as_bool).unwrap_or(true);
                let mut params = Map::new();
                for (k, v) in obj {
                    if !matches!(
                        k.as_str(),
                        "toolId" | "tool_id" | "id" | "enabled" | "params"
                    ) {
                        params.insert(k.clone(), v.clone());
                    }
                }
                if let Some(Value::Object(nested)) = obj.get("params") {
                    for (k, v) in nested {
                        params.insert(k.clone(), v.clone());
                    }
                }
                self.chain.push(LooksToolInstance {
                    tool_id: tool_id.to_string(),
                    enabled,
                    params,
                });
            }
        }
        self.selected_index = if self.chain.is_empty() { None } else { Some(0) };
    }

    /// Serializes the active tool chain to the canonical `looksInstances` JSON array.
    #[must_use]
    pub fn to_json(&self) -> Value {
        Value::Array(
            self.chain
                .iter()
                .map(|inst| {
                    json!({
                        "toolId": inst.tool_id,
                        "enabled": inst.enabled,
                        "params": Value::Object(inst.params.clone()),
                    })
                })
                .collect(),
        )
    }

    /// Appends a tool by canonical `tool_id` with default parameters and selects it.
    pub fn add_tool(&mut self, tool_id: &str) -> usize {
        let mut params = Map::new();
        if let Some(spec) = find_looks_tool(tool_id) {
            for p in spec.params {
                match p.kind {
                    LooksParamKind::Float { .. } => {
                        params.insert(p.key.into(), json!(p.default_f64));
                    }
                    LooksParamKind::Bool => {
                        params.insert(p.key.into(), json!(p.default_f64 != 0.0));
                    }
                    LooksParamKind::Choice(_) => {
                        params.insert(p.key.into(), json!(p.default_f64.round() as i64));
                    }
                    LooksParamKind::ColorRgb => {
                        params.insert(
                            p.key.into(),
                            json!([p.default_f64, p.default_f64, p.default_f64]),
                        );
                    }
                }
            }
        }
        self.chain.push(LooksToolInstance {
            tool_id: tool_id.to_string(),
            enabled: true,
            params,
        });
        let idx = self.chain.len() - 1;
        self.selected_index = Some(idx);
        self.dirty = true;
        idx
    }

    /// Duplicates the tool at `index`.
    pub fn duplicate_tool(&mut self, index: usize) -> Option<usize> {
        let copy = self.chain.get(index)?.clone();
        let insert_at = index + 1;
        self.chain.insert(insert_at, copy);
        self.selected_index = Some(insert_at);
        self.dirty = true;
        Some(insert_at)
    }

    /// Removes the tool at `index`.
    pub fn remove_tool(&mut self, index: usize) -> bool {
        if index >= self.chain.len() {
            return false;
        }
        self.chain.remove(index);
        if self.chain.is_empty() {
            self.selected_index = None;
        } else {
            self.selected_index = Some(index.min(self.chain.len() - 1));
        }
        self.dirty = true;
        true
    }

    /// Moves a tool one position earlier (`delta = -1`) or later (`delta = 1`).
    pub fn move_tool(&mut self, index: usize, delta: isize) -> bool {
        let target = index as isize + delta;
        if target < 0 || target as usize >= self.chain.len() {
            return false;
        }
        let target = target as usize;
        self.chain.swap(index, target);
        self.selected_index = Some(target);
        self.dirty = true;
        true
    }

    /// Applies one of the 6 built-in curated Look Presets.
    pub fn apply_preset(&mut self, preset_name: &str) {
        self.chain.clear();
        match preset_name {
            "Cinema Warm" => {
                self.push_preset_tool(
                    "warm-cool",
                    json!({"warmCool": 0.28, "tint": 0.04, "exposure": 0.08}),
                );
                self.push_preset_tool("lift-gamma-gain", json!({"strength": 0.65, "lift": [0.94, 0.96, 1.04], "gamma": [1.04, 1.0, 0.96], "gain": [1.08, 1.02, 0.92]}));
                self.push_preset_tool(
                    "diffusion",
                    json!({"size": 0.35, "grade": 0.32, "glow": 0.25, "highlightsOnly": true}),
                );
                self.push_preset_tool(
                    "vignette",
                    json!({"radius": 0.72, "spread": 0.45, "strength": 0.38}),
                );
                self.push_preset_tool("auto-shoulder", json!({"strength": 0.45}));
            }
            "Teal & Orange" => {
                self.push_preset_tool("mojo", json!({"mojo": 0.78, "punch": 0.35, "blueSqueeze": 0.45, "skinSqueeze": 0.55, "strength": 0.85}));
                self.push_preset_tool(
                    "color-contrast",
                    json!({"pivot": 0.48, "exposure": 0.12, "color": [1.0, 0.68, 0.42]}),
                );
                self.push_preset_tool("anamorphic-flare", json!({"size": 0.42, "boost": 0.35, "threshold": 0.78, "color": [0.35, 0.78, 1.0]}));
                self.push_preset_tool(
                    "vignette",
                    json!({"radius": 0.68, "spread": 0.4, "strength": 0.42}),
                );
            }
            "Neon Cyber" => {
                self.push_preset_tool(
                    "hsl-colors",
                    json!({"cyanSat": 0.45, "blueSat": 0.35, "magentaSat": 0.5, "purpleSat": 0.4}),
                );
                self.push_preset_tool(
                    "contrast",
                    json!({"contrast": 0.38, "pivot": 0.45, "exposure": 0.05}),
                );
                self.push_preset_tool(
                    "chromatic-aberration",
                    json!({"redCyan": 0.25, "greenMagenta": -0.1, "blueYellow": 0.18}),
                );
                self.push_preset_tool("star-filter", json!({"size": 0.4, "boost": 0.45, "threshold": 0.72, "angle": 30.0, "color": [0.65, 0.9, 1.0]}));
                self.push_preset_tool(
                    "diffusion",
                    json!({"size": 0.45, "glow": 0.42, "highlightsOnly": true}),
                );
            }
            "Film Noir" => {
                self.push_preset_tool("ranged-saturation", json!({"satHighlight": -1.0, "satMidtone": -1.0, "satShadow": -1.0, "exposure": 0.05}));
                self.push_preset_tool(
                    "contrast",
                    json!({"contrast": 0.55, "pivot": 0.46, "exposure": 0.0}),
                );
                self.push_preset_tool("crush", json!({"gamma": 1.25, "exposure": -0.12}));
                self.push_preset_tool(
                    "vignette",
                    json!({"radius": 0.58, "spread": 0.42, "strength": 0.62}),
                );
                self.push_preset_tool(
                    "edge-softness",
                    json!({"blurSize": 0.22, "radius": 0.72, "spread": 0.35}),
                );
            }
            "Pastel Dream" => {
                self.push_preset_tool("lift-gamma-gain", json!({"strength": 0.55, "lift": [1.06, 1.02, 1.08], "gamma": [1.05, 1.03, 1.06], "gain": [0.98, 0.96, 1.02]}));
                self.push_preset_tool(
                    "lightflex",
                    json!({"boost": 0.32, "exposure": 0.1, "color": [1.0, 0.88, 0.94]}),
                );
                self.push_preset_tool(
                    "diffusion",
                    json!({"size": 0.55, "grade": 0.4, "glow": 0.38}),
                );
                self.push_preset_tool(
                    "ranged-saturation",
                    json!({"satHighlight": -0.25, "satMidtone": -0.15, "satShadow": -0.2}),
                );
            }
            "Vintage Chrome" => {
                self.push_preset_tool("three-strip", json!({"strength": 0.68, "exposure": 0.06}));
                self.push_preset_tool(
                    "color-reversal",
                    json!({"strength": 0.35, "exposure": 0.04}),
                );
                self.push_preset_tool("haze-flare", json!({"spillage": 0.3, "softness": 0.55, "reach": 0.4, "exposure": 0.2, "color": [1.0, 0.86, 0.68]}));
                self.push_preset_tool(
                    "lens-distortion",
                    json!({"distortion": -0.12, "flatness": 0.5}),
                );
                self.push_preset_tool(
                    "vignette",
                    json!({"radius": 0.64, "spread": 0.48, "strength": 0.48}),
                );
            }
            _ => {}
        }
        self.selected_index = if self.chain.is_empty() { None } else { Some(0) };
        self.dirty = true;
    }

    pub fn push_preset_tool(&mut self, tool_id: &str, overrides: Value) {
        let idx = self.add_tool(tool_id);
        if let (Some(inst), Some(map)) = (self.chain.get_mut(idx), overrides.as_object()) {
            for (k, v) in map {
                inst.params.insert(k.clone(), v.clone());
            }
        }
    }

    /// Commits `looksInstances` into `Session` via `Command::SetPropertyValue`.
    pub fn commit_to_session(&mut self, session: &mut Session) -> bool {
        if self.effect_id.is_empty() {
            return false;
        }
        let ok = session
            .execute(Command::SetPropertyValue {
                target_id: self.effect_id.clone(),
                property: "looksInstances".to_string(),
                value: self.to_json(),
            })
            .is_ok();
        if ok {
            self.dirty = false;
        }
        ok
    }
}

pub const LOOKS_PRESET_NAMES: &[&str] = &[
    "Cinema Warm",
    "Teal & Orange",
    "Neon Cyber",
    "Film Noir",
    "Pastel Dream",
    "Vintage Chrome",
];

/// Renders the floating Color Looks Setup window when `state.open` is true.
pub fn show_looks_editor_window(
    ctx: &egui::Context,
    viewport_rect: Rect,
    floating_windows: &mut FloatingWindowManager,
    state: &mut LooksEditorState,
    session: &mut Session,
    theme: &Cm3Theme,
) {
    if !state.open {
        return;
    }

    let ctx_screen = ctx.input(|i| i.viewport_rect());
    let screen = if ctx_screen.width() > 100.0 && ctx_screen.height() > 100.0 {
        ctx_screen
    } else {
        viewport_rect
    };
    let (_, default_size) = compute_clamped_editor_window_rect(screen, 860.0, 560.0);

    let spec = FloatingWindowSpec::new("pro_color_looks_editor", "Color Looks Setup")
        .with_subtitle(state.effect_label.clone())
        .with_default_size(default_size);

    ensure_pro_window_in_screen(floating_windows, &spec, screen, &mut state.just_opened);

    let mut close_requested = false;
    let mut should_commit = false;

    let ev = cm3_floating_window(ctx, floating_windows, &spec, screen, theme, |ui| {
        // --- Top Bar: Presets + Add Tool + Actions ---
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            if cm3_prop_button(
                ui,
                if state.picker_open {
                    "Hide Tool Browser"
                } else {
                    "+ Add Tool (29)"
                },
                if state.picker_open {
                    ButtonVariant::Primary
                } else {
                    ButtonVariant::Normal
                },
                false,
                theme,
            )
            .clicked()
            {
                state.picker_open = !state.picker_open;
                state.picker_stage_filter = None;
            }

            egui::ComboBox::from_id_salt(Id::unique("looks_preset_combo"))
                .selected_text("Look Presets")
                .width(132.0)
                .show_ui(ui, |ui| {
                    for preset in LOOKS_PRESET_NAMES {
                        if ui.selectable_label(false, *preset).clicked() {
                            state.apply_preset(preset);
                            if state.auto_apply {
                                should_commit = true;
                            }
                        }
                    }
                });

            if !state.chain.is_empty()
                && cm3_prop_button(ui, "Clear All", ButtonVariant::Danger, false, theme).clicked()
            {
                state.chain.clear();
                state.selected_index = None;
                state.dirty = true;
                if state.auto_apply {
                    should_commit = true;
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if cm3_prop_button(ui, "Close", ButtonVariant::Normal, false, theme).clicked() {
                    close_requested = true;
                }
                let apply_label = if cfg!(target_os = "macos") {
                    if state.dirty {
                        "Apply * (Cmd+Enter)"
                    } else {
                        "Apply (Cmd+Enter)"
                    }
                } else if state.dirty {
                    "Apply * (Ctrl+Enter)"
                } else {
                    "Apply (Ctrl+Enter)"
                };
                if cm3_prop_button(ui, apply_label, ButtonVariant::Primary, false, theme).clicked()
                {
                    should_commit = true;
                }
                ui.checkbox(&mut state.auto_apply, "Auto-Apply");
            });
        });

        if ui.input(|i| (i.modifiers.ctrl || i.modifiers.command) && i.key_pressed(Key::Enter)) {
            should_commit = true;
        }

        ui.add_space(6.0);

        // --- 5-Stage Pipeline Strip (Subject -> Matte -> Lens -> Camera -> Post) ---
        egui::Frame::NONE
            .fill(BG_SUNKEN)
            .stroke(Stroke::new(1.0, BORDER_PANEL))
            .corner_radius(CornerRadius::same(4))
            .inner_margin(egui::Margin::same(6))
            .show(ui, |ui| {
                let stage_w = ((ui.available_width() - 24.0) / 5.0).max(96.0);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    for stage in LooksStage::ALL {
                        ui.vertical(|ui| {
                            ui.set_width(stage_w);
                            ui.horizontal(|ui| {
                                let (dot_r, _) =
                                    ui.allocate_exact_size(Vec2::splat(7.0), Sense::hover());
                                ui.painter().circle_filled(
                                    dot_r.center(),
                                    3.5,
                                    stage.badge_color(),
                                );
                                ui.label(
                                    egui::RichText::new(stage.label())
                                        .font(theme.font(FONT_SMALL))
                                        .color(TEXT_STRONG),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui.small_button("+").clicked() {
                                            state.picker_open = true;
                                            state.picker_stage_filter = Some(stage);
                                        }
                                    },
                                );
                            });
                            ui.add_space(2.0);
                            let mut stage_count = 0usize;
                            for (idx, inst) in state.chain.iter().enumerate() {
                                let Some(spec) = find_looks_tool(&inst.tool_id) else {
                                    continue;
                                };
                                if spec.stage != stage {
                                    continue;
                                }
                                stage_count += 1;
                                let is_sel = state.selected_index == Some(idx);
                                let fill = if is_sel {
                                    color32_with_alpha(stage.badge_color(), 68)
                                } else if inst.enabled {
                                    BG_BUTTON
                                } else {
                                    BG_PANEL
                                };
                                let text_col = if inst.enabled {
                                    TEXT_PRIMARY
                                } else {
                                    TEXT_FAINT
                                };
                                let resp = ui.add_sized(
                                    Vec2::new(stage_w, 20.0),
                                    egui::Button::new(
                                        egui::RichText::new(spec.label)
                                            .font(theme.font(FONT_COMPACT))
                                            .color(text_col),
                                    )
                                    .fill(fill)
                                    .stroke(Stroke::new(
                                        1.0,
                                        if is_sel {
                                            stage.badge_color()
                                        } else {
                                            BORDER_CONTROL
                                        },
                                    )),
                                );
                                if resp.clicked() {
                                    state.selected_index = Some(idx);
                                }
                            }
                            if stage_count == 0 {
                                ui.label(
                                    egui::RichText::new("Empty")
                                        .font(theme.font(FONT_COMPACT))
                                        .color(TEXT_FAINT),
                                );
                            }
                        });
                    }
                });
            });

        ui.add_space(6.0);

        // --- Main Split: Optional Tool Picker | Active Chain List | Tool Parameters ---
        let body_h = (ui.available_height() - 4.0).max(180.0);
        let picker_w = if state.picker_open { 220.0 } else { 0.0 };
        let chain_w = 240.0;
        let params_w = (ui.available_width()
            - picker_w
            - chain_w
            - if state.picker_open { 12.0 } else { 6.0 })
        .max(220.0);

        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;

            // Optional Tool Browser
            if state.picker_open {
                ui.allocate_ui_with_layout(
                    Vec2::new(picker_w, body_h),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        egui::Frame::NONE
                            .fill(BG_SUNKEN)
                            .stroke(Stroke::new(1.0, BORDER_PANEL))
                            .corner_radius(CornerRadius::same(4))
                            .inner_margin(egui::Margin::same(6))
                            .show(ui, |ui| {
                                ui.set_width((picker_w - 14.0).max(100.0));
                                ui.set_min_height((body_h - 14.0).max(100.0));
                                ui.set_max_height((body_h - 14.0).max(100.0));
                                if show_looks_picker_ui(ui, state, theme) && state.auto_apply {
                                    should_commit = true;
                                }
                            });
                    },
                );
            }

            // Active Chain Order Column
            ui.allocate_ui_with_layout(
                Vec2::new(chain_w, body_h),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    egui::Frame::NONE
                        .fill(BG_PANEL)
                        .stroke(Stroke::new(1.0, BORDER_PANEL))
                        .corner_radius(CornerRadius::same(4))
                        .inner_margin(egui::Margin::same(6))
                        .show(ui, |ui| {
                            ui.set_width((chain_w - 14.0).max(100.0));
                            ui.set_min_height((body_h - 14.0).max(100.0));
                            ui.set_max_height((body_h - 14.0).max(100.0));
                            if show_looks_chain_list_ui(ui, state, theme) && state.auto_apply {
                                should_commit = true;
                            }
                        });
                },
            );

            // Selected Tool Parameter Inspector
            ui.allocate_ui_with_layout(
                Vec2::new(params_w, body_h),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    egui::Frame::NONE
                        .fill(BG_PANEL)
                        .stroke(Stroke::new(1.0, BORDER_PANEL))
                        .corner_radius(CornerRadius::same(4))
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            ui.set_width((params_w - 18.0).max(120.0));
                            ui.set_min_height((body_h - 18.0).max(100.0));
                            ui.set_max_height((body_h - 18.0).max(100.0));
                            if show_looks_tool_params_ui(ui, state, theme) && state.auto_apply {
                                should_commit = true;
                            }
                        });
                },
            );
        });
    });

    clamp_floating_window_in_screen(floating_windows, &spec, screen);

    if ev.closed && ctx.input(|i| i.key_pressed(Key::Escape)) {
        mark_floating_editor_escape_consumed(ctx);
    } else if !ev.closed
        && consume_escape_for_floating_editor(ctx, floating_windows, "pro_color_looks_editor")
    {
        close_requested = true;
    }

    if should_commit {
        state.commit_to_session(session);
    }
    if close_requested || ev.closed {
        state.open = false;
        floating_windows.close("pro_color_looks_editor");
    }
}

pub fn show_looks_picker_ui(ui: &mut Ui, state: &mut LooksEditorState, theme: &Cm3Theme) -> bool {
    let mut changed = false;
    ui.label(
        egui::RichText::new("Tools (29)")
            .font(theme.font(FONT_TITLE))
            .color(TEXT_STRONG),
    );
    ui.add_space(3.0);
    ui.add(
        TextEdit::singleline(&mut state.picker_search)
            .hint_text("Filter tools...")
            .desired_width(ui.available_width()),
    );
    ui.add_space(3.0);
    ui.horizontal_wrapped(|ui| {
        if ui
            .selectable_label(state.picker_stage_filter.is_none(), "All")
            .clicked()
        {
            state.picker_stage_filter = None;
        }
        for stage in LooksStage::ALL {
            if ui
                .selectable_label(state.picker_stage_filter == Some(stage), stage.label())
                .clicked()
            {
                state.picker_stage_filter = Some(stage);
            }
        }
    });
    ui.add_space(4.0);

    let q = state.picker_search.trim().to_ascii_lowercase();
    ScrollArea::vertical()
        .id_salt("looks_tool_picker_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for stage in LooksStage::ALL {
                if let Some(filter) = state.picker_stage_filter
                    && filter != stage
                {
                    continue;
                }
                let mut header_shown = false;
                for spec in LOOKS_TOOL_CATALOG.iter().filter(|s| s.stage == stage) {
                    if !q.is_empty()
                        && !spec.label.to_ascii_lowercase().contains(&q)
                        && !spec.tool_id.to_ascii_lowercase().contains(&q)
                    {
                        continue;
                    }
                    if !header_shown {
                        ui.add_space(3.0);
                        ui.label(
                            egui::RichText::new(stage.label().to_ascii_uppercase())
                                .font(theme.font(FONT_COMPACT))
                                .color(stage.badge_color()),
                        );
                        header_shown = true;
                    }
                    if ui
                        .add_sized(
                            Vec2::new(ui.available_width(), 22.0),
                            egui::Button::new(
                                egui::RichText::new(spec.label)
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_PRIMARY),
                            )
                            .fill(BG_BUTTON)
                            .stroke(Stroke::new(1.0, BORDER_CONTROL)),
                        )
                        .clicked()
                    {
                        state.add_tool(spec.tool_id);
                        changed = true;
                    }
                }
            }
        });
    changed
}

pub fn show_looks_chain_list_ui(
    ui: &mut Ui,
    state: &mut LooksEditorState,
    theme: &Cm3Theme,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("Active Chain ({})", state.chain.len()))
                .font(theme.font(FONT_TITLE))
                .color(TEXT_STRONG),
        );
    });
    ui.separator();

    if state.chain.is_empty() {
        ui.label(
            egui::RichText::new("No tools active.\nPick a preset or click '+ Add Tool'.")
                .font(theme.font(FONT_SMALL))
                .color(TEXT_FAINT),
        );
        return false;
    }

    let mut action: Option<(usize, &'static str)> = None;
    ScrollArea::vertical()
        .id_salt("looks_chain_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let count = state.chain.len();
            for idx in 0..count {
                let selected = state.selected_index == Some(idx);
                let tool_id = state.chain[idx].tool_id.clone();
                let spec = find_looks_tool(&tool_id);
                let label = spec.map(|s| s.label).unwrap_or(tool_id.as_str());
                let stage_col = spec
                    .map(|s| s.stage.badge_color())
                    .unwrap_or_else(|| theme.accent());

                egui::Frame::NONE
                    .fill(if selected { BG_HOVER } else { BG_INPUT })
                    .stroke(Stroke::new(
                        1.0,
                        if selected { stage_col } else { BORDER_CONTROL },
                    ))
                    .corner_radius(CornerRadius::same(3))
                    .inner_margin(egui::Margin::symmetric(5, 3))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut state.chain[idx].enabled, "").changed() {
                                state.dirty = true;
                                changed = true;
                            }
                            let label_resp = ui.selectable_label(
                                selected,
                                egui::RichText::new(label)
                                    .font(theme.font(FONT_SMALL))
                                    .color(if state.chain[idx].enabled {
                                        TEXT_PRIMARY
                                    } else {
                                        TEXT_FAINT
                                    }),
                            );
                            if label_resp.clicked() {
                                state.selected_index = Some(idx);
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.small_button("×").on_hover_text("Delete").clicked() {
                                        action = Some((idx, "del"));
                                    }
                                    if ui.small_button("D").on_hover_text("Duplicate").clicked() {
                                        action = Some((idx, "dup"));
                                    }
                                    if idx + 1 < count
                                        && ui.small_button("↓").on_hover_text("Move Down").clicked()
                                    {
                                        action = Some((idx, "down"));
                                    }
                                    if idx > 0
                                        && ui.small_button("↑").on_hover_text("Move Up").clicked()
                                    {
                                        action = Some((idx, "up"));
                                    }
                                },
                            );
                        });
                    });
                ui.add_space(3.0);
            }
        });

    if let Some((idx, cmd)) = action {
        match cmd {
            "del" => changed |= state.remove_tool(idx),
            "dup" => changed |= state.duplicate_tool(idx).is_some(),
            "up" => changed |= state.move_tool(idx, -1),
            "down" => changed |= state.move_tool(idx, 1),
            _ => {}
        }
    }
    changed
}

pub fn show_looks_tool_params_ui(
    ui: &mut Ui,
    state: &mut LooksEditorState,
    theme: &Cm3Theme,
) -> bool {
    let mut changed = false;
    let Some(idx) = state.selected_index.filter(|&i| i < state.chain.len()) else {
        ui.label(
            egui::RichText::new("Select a tool in the chain to edit its parameters.")
                .font(theme.font(FONT_SMALL))
                .color(TEXT_MUTED),
        );
        return false;
    };

    let tool_id = state.chain[idx].tool_id.clone();
    let Some(spec) = find_looks_tool(&tool_id) else {
        ui.label(
            egui::RichText::new(format!("Unknown tool: {tool_id}"))
                .font(theme.font(FONT_SMALL))
                .color(DANGER_TEXT),
        );
        return false;
    };

    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(spec.label)
                .font(theme.font(FONT_TITLE))
                .color(TEXT_STRONG),
        );
        ui.label(
            egui::RichText::new(format!("[{}]", spec.stage.label()))
                .font(theme.font(FONT_SMALL))
                .color(spec.stage.badge_color()),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if cm3_prop_button(ui, "Reset Tool", ButtonVariant::Normal, false, theme).clicked() {
                state.chain[idx].params.clear();
                for p in spec.params {
                    match p.kind {
                        LooksParamKind::Float { .. } => {
                            state.chain[idx]
                                .params
                                .insert(p.key.into(), json!(p.default_f64));
                        }
                        LooksParamKind::Bool => {
                            state.chain[idx]
                                .params
                                .insert(p.key.into(), json!(p.default_f64 != 0.0));
                        }
                        LooksParamKind::Choice(_) => {
                            state.chain[idx]
                                .params
                                .insert(p.key.into(), json!(p.default_f64.round() as i64));
                        }
                        LooksParamKind::ColorRgb => {
                            state.chain[idx].params.insert(
                                p.key.into(),
                                json!([p.default_f64, p.default_f64, p.default_f64]),
                            );
                        }
                    }
                }
                state.dirty = true;
                changed = true;
            }
        });
    });
    ui.separator();

    ScrollArea::vertical()
        .id_salt("looks_tool_params_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let inst = &mut state.chain[idx];
            for param in spec.params {
                ui.add_space(3.0);
                match param.kind {
                    LooksParamKind::Float { min, max, step } => {
                        let mut val = inst
                            .params
                            .get(param.key)
                            .and_then(Value::as_f64)
                            .unwrap_or(param.default_f64);
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(param.label)
                                    .font(theme.font(FONT_BODY))
                                    .color(TEXT_PRIMARY),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .add(
                                            egui::Slider::new(&mut val, min..=max)
                                                .step_by(step)
                                                .max_decimals(3),
                                        )
                                        .changed()
                                    {
                                        inst.params.insert(param.key.into(), json!(val));
                                        changed = true;
                                    }
                                },
                            );
                        });
                    }
                    LooksParamKind::Bool => {
                        let mut val = inst
                            .params
                            .get(param.key)
                            .map(|v| {
                                v.as_bool()
                                    .unwrap_or_else(|| v.as_f64().unwrap_or(0.0) != 0.0)
                            })
                            .unwrap_or(param.default_f64 != 0.0);
                        if ui.checkbox(&mut val, param.label).changed() {
                            inst.params.insert(param.key.into(), json!(val));
                            changed = true;
                        }
                    }
                    LooksParamKind::Choice(options) => {
                        let mut opt_idx = inst
                            .params
                            .get(param.key)
                            .and_then(Value::as_u64)
                            .map(|v| (v as usize).min(options.len().saturating_sub(1)))
                            .unwrap_or(param.default_f64.round().max(0.0) as usize);
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(param.label)
                                    .font(theme.font(FONT_BODY))
                                    .color(TEXT_PRIMARY),
                            );
                            egui::ComboBox::from_id_salt(Id::unique((
                                "looks_choice",
                                idx,
                                param.key,
                            )))
                            .selected_text(options.get(opt_idx).copied().unwrap_or("—"))
                            .show_ui(ui, |ui| {
                                for (oi, opt) in options.iter().enumerate() {
                                    if ui.selectable_value(&mut opt_idx, oi, *opt).changed() {
                                        inst.params.insert(param.key.into(), json!(opt_idx));
                                        changed = true;
                                    }
                                }
                            });
                        });
                    }
                    LooksParamKind::ColorRgb => {
                        let mut rgb = extract_rgb_f32(
                            inst.params.get(param.key),
                            [param.default_f64 as f32; 3],
                        );
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(param.label)
                                    .font(theme.font(FONT_BODY))
                                    .color(TEXT_PRIMARY),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.color_edit_button_rgb(&mut rgb).changed() {
                                        inst.params.insert(
                                            param.key.into(),
                                            json!([rgb[0] as f64, rgb[1] as f64, rgb[2] as f64]),
                                        );
                                        changed = true;
                                    }
                                },
                            );
                        });
                    }
                }
            }
        });

    if changed {
        state.dirty = true;
    }
    changed
}

#[must_use]
pub fn find_looks_tool(tool_id: &str) -> Option<&'static LooksToolSpec> {
    let lower = tool_id.trim().to_ascii_lowercase();
    LOOKS_TOOL_CATALOG
        .iter()
        .find(|s| s.tool_id.eq_ignore_ascii_case(&lower))
}
