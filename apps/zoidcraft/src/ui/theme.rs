//! Pixel-exact ClipMaker 3 dark chrome palette, Zoidium HSL theme derivation,
//! exact typography scale, and procedural vector icon painter.

use std::f32::consts::TAU;

use egui::{
    Align2, Color32, CornerRadius, FontFamily, FontId, Margin, Painter, Pos2, Rect, Shape, Stroke,
    StrokeKind, TextStyle, Vec2,
};
use zc_anim::{EasingKind, eval_easing};
use zc_color::{Hsl, Rgb, Rgba, ThemePalette, ThemeProfile};

// ---------------------------------------------------------------------------
// Fixed Dark Neutral Chrome & Structural Surfaces (Spec 01 §2.1)
// ---------------------------------------------------------------------------

pub const VOID_BLACK: Color32 = Color32::from_rgb(0x00, 0x00, 0x00);
pub const BG_SUNKEN: Color32 = Color32::from_rgb(0x1d, 0x1d, 0x1d);
pub const BG_INPUT: Color32 = Color32::from_rgb(0x20, 0x20, 0x20);
pub const BG_TITLE: Color32 = Color32::from_rgb(0x22, 0x22, 0x22);
pub const BG_TOOLBAR: Color32 = Color32::from_rgb(0x24, 0x24, 0x24);
pub const BG_PANEL: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x2a);
pub const BG_DROPDOWN_HOVER: Color32 = Color32::from_rgb(0x2c, 0x2c, 0x2c);
pub const BG_OPTION_HOVER: Color32 = Color32::from_rgb(0x2f, 0x2f, 0x2f);
pub const BG_HOVER: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
pub const KF_ROW_EVEN: Color32 = Color32::from_rgb(0x34, 0x34, 0x34);
pub const KF_ROW_ODD: Color32 = Color32::from_rgb(0x3f, 0x3f, 0x3f);
pub const KF_LABEL_EVEN: Color32 = Color32::from_rgb(0x2e, 0x2e, 0x2e);
pub const KF_LABEL_ODD: Color32 = Color32::from_rgb(0x3b, 0x3b, 0x3b);
pub const BG_OPTION_ACTIVE: Color32 = Color32::from_rgb(0x3f, 0x3f, 0x3f);
pub const BG_BUTTON: Color32 = Color32::from_rgb(0x43, 0x43, 0x43);
pub const BG_BUTTON_PRESSED: Color32 = Color32::from_rgb(0x44, 0x44, 0x44);
pub const BG_BACK_BUTTON: Color32 = Color32::from_rgb(0x47, 0x47, 0x47);
pub const SCROLLBAR_THUMB: Color32 = Color32::from_rgb(0x55, 0x55, 0x55);
pub const SCROLLBAR_THUMB_ZOIDIUM: Color32 = Color32::from_rgb(0x6d, 0x6d, 0x6d);

// ---------------------------------------------------------------------------
// Borders & Dividers (Spec 01 §2.2)
// ---------------------------------------------------------------------------

pub const BORDER_CONTROL: Color32 = Color32::from_rgb(0x1b, 0x1b, 0x1b);
pub const BORDER_TRACK_SPLIT: Color32 = Color32::from_rgb(0x21, 0x21, 0x21);
pub const BORDER_CLIP_RIGHT: Color32 = Color32::from_rgb(0x23, 0x23, 0x23);
pub const BORDER_CLIP_LEFT: Color32 = Color32::from_rgb(0x83, 0x83, 0x83);
pub const BORDER_ROW_SOFT: Color32 = Color32::from_rgb(0x24, 0x24, 0x24);
pub const BORDER_PANEL: Color32 = Color32::from_rgb(0x33, 0x33, 0x33);
pub const BORDER_FILTER: Color32 = Color32::from_rgb(0x36, 0x36, 0x36);
pub const BORDER_RULER_MAJOR: Color32 = Color32::from_rgb(0x55, 0x55, 0x55);
pub const BORDER_RULER_MINOR: Color32 = Color32::from_rgb(0x3a, 0x3a, 0x3a);
pub const BORDER_BUTTON_HOVER: Color32 = Color32::from_rgb(0x54, 0x54, 0x54);
pub const BORDER_OPTION_ACTIVE: Color32 = Color32::from_rgb(0x67, 0x67, 0x67);
pub const BORDER_RAIL_FOCUS: Color32 = Color32::from_rgb(0x71, 0x71, 0x71);

// ---------------------------------------------------------------------------
// Text Tiers (Spec 01 §2.3)
// ---------------------------------------------------------------------------

pub const TEXT_BRIGHT: Color32 = Color32::from_rgb(0xe4, 0xe4, 0xe4);
pub const TEXT_STRONG: Color32 = Color32::from_rgb(0xda, 0xda, 0xda);
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xcc, 0xcc, 0xcc);
pub const TEXT_BUTTON: Color32 = Color32::from_rgb(0xcb, 0xcb, 0xcb);
pub const TEXT_TRACK_LABEL: Color32 = Color32::from_rgb(0xbd, 0xbd, 0xbd);
pub const TEXT_DISABLED: Color32 = Color32::from_rgb(0x99, 0x99, 0x99);
pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x9a, 0x9a, 0x9a);
pub const TEXT_OPTION_DESC: Color32 = Color32::from_rgb(0x96, 0x96, 0x96);
pub const TEXT_FAINT: Color32 = Color32::from_rgb(0x6d, 0x6d, 0x6d);
pub const TEXT_EMPTY: Color32 = Color32::from_rgb(0x66, 0x66, 0x66);
pub const TEXT_RULER: Color32 = Color32::from_rgb(0x65, 0x65, 0x65);
pub const TEXT_INACTIVE_TAB: Color32 = Color32::from_rgb(0x30, 0x30, 0x30);

// ---------------------------------------------------------------------------
// Fixed Semantic & Status Colors (Unthemed, Spec 01 §2.4)
// ---------------------------------------------------------------------------

pub const TOGGLE_ACTIVE: Color32 = Color32::from_rgb(0x8a, 0x28, 0x28);
pub const TOOLBAR_ICON_INACTIVE: Color32 = Color32::from_rgb(0xac, 0xac, 0xac);
pub const ICON_ACTIVE_ALERT: Color32 = Color32::from_rgb(0xa9, 0x24, 0x24);
pub const PLAYHEAD_RED: Color32 = Color32::from_rgb(0xb7, 0x1e, 0x1e);
pub const TRIM_HANDLE_RED: Color32 = Color32::from_rgb(0x98, 0x2b, 0x2b);
pub const DANGER_TEXT: Color32 = Color32::from_rgb(0xdf, 0x9b, 0x9b);
pub const WARNING_BAR: Color32 = Color32::from_rgb(0xa9, 0x4b, 0x4b);
pub const WARNING_SURFACE: Color32 = Color32::from_rgba_premultiplied(22, 6, 6, 46);
pub const AUDIO_PEAK_YELLOW: Color32 = Color32::from_rgb(0xce, 0xce, 0x62);
pub const MARKER_GREEN: Color32 = Color32::from_rgb(0x70, 0xc2, 0x9d);
pub const WAVEFORM_FILL: Color32 = Color32::from_rgb(0x24, 0x2d, 0x42);
pub const KF_SELECTED_FILL: Color32 = Color32::from_rgb(0x29, 0x29, 0x29);

// Canonical Classic 5-color accent constants (Spec 01 §2.5)
pub const CLASSIC_ACCENT: Color32 = Color32::from_rgb(0x38, 0x46, 0x68);
pub const CLASSIC_VALUE: Color32 = Color32::from_rgb(0x7e, 0x8f, 0xb9);
pub const CLASSIC_CLIP: Color32 = Color32::from_rgb(0x52, 0x61, 0x83);
pub const CLASSIC_CLIP_SELECTED: Color32 = Color32::from_rgb(0x7f, 0x94, 0xc7);
pub const CLASSIC_CLIP_SELECTION: Color32 = Color32::from_rgb(0x27, 0x34, 0x65);

// ---------------------------------------------------------------------------
// Exact CM3 Typography Scale (Spec 01 §3)
// ---------------------------------------------------------------------------

/// Sidebar elevator tab labels (`22 px`).
pub const FONT_TAB: f32 = 22.0;
/// Window titles & section headers (`18 px`).
pub const FONT_TITLE: f32 = 18.0;
/// Standard UI body, property labels, numeric values, inputs, buttons (`16 px`).
pub const FONT_BODY: f32 = 16.0;
/// Add-picker category banner (`15 px`).
pub const FONT_BANNER: f32 = 15.0;
/// Secondary text, ruler labels, subtitles, descriptions (`14 px`).
pub const FONT_SMALL: f32 = 14.0;
/// Context & anchored menu items (`13 px`).
pub const FONT_MENU: f32 = 13.0;
/// Collapse carets, menu section headings, compact headers (`12 px`).
pub const FONT_COMPACT: f32 = 12.0;
/// Easing button label text, fallback font label (`11 px`).
pub const FONT_EASING: f32 = 11.0;
/// Uppercase dropdown separators & error pill badges (`10 px`).
pub const FONT_BADGE: f32 = 10.0;

/// Returns a CM3 monospace [`FontId`] at the requested pixel size.
#[must_use]
pub fn font_mono(size: f32) -> FontId {
    FontId::new(size, FontFamily::Monospace)
}

// ---------------------------------------------------------------------------
// Color Conversion & Deterministic CM3 Hash Helpers
// ---------------------------------------------------------------------------

/// Converts [`zc_color::Rgb`] to an opaque [`egui::Color32`].
#[must_use]
pub fn rgb_to_color32(rgb: Rgb) -> Color32 {
    let [r, g, b] = rgb.to_u8();
    Color32::from_rgb(r, g, b)
}

/// Converts [`egui::Color32`] to [`zc_color::Rgb`].
#[must_use]
pub fn color32_to_rgb(c: Color32) -> Rgb {
    Rgb::from_u8(c.r(), c.g(), c.b())
}

/// Converts straight [`zc_color::Rgba`] to [`egui::Color32`].
#[must_use]
pub fn rgba_to_color32(rgba: Rgba) -> Color32 {
    let [r, g, b, a] = rgba.to_u8();
    Color32::from_rgba_unmultiplied(r, g, b, a)
}

/// Returns `color` with its alpha replaced by `alpha` (`0..=255`).
#[must_use]
pub fn color32_with_alpha(color: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha)
}

/// Deterministic 32-bit string hash matching CM3's stopwatch & keyframe hue mapping.
#[must_use]
pub fn cm3_string_hash(name: &str) -> u32 {
    let mut hash: i32 = 0;
    for byte in name.bytes() {
        hash = hash
            .wrapping_shl(5)
            .wrapping_sub(hash)
            .wrapping_add(i32::from(byte));
    }
    hash.unsigned_abs()
}

/// Animated stopwatch color: `hsl((40 * stringHash(name)) % 360, 60%, 45%)` (Spec 01 §2.4).
#[must_use]
pub fn stopwatch_active_color(name: &str) -> Color32 {
    let deg = ((cm3_string_hash(name) % 360) * 40) % 360;
    let rgb = Rgb::from_hsl(Hsl {
        h: deg as f32 / 360.0,
        s: 0.60,
        l: 0.45,
    });
    rgb_to_color32(rgb)
}

/// Keyframe diamond `(fill, border)` colors: `hsl(H, 40%, 45%)` and `hsl(H, 40%, 60%)` (Spec 02 §1.5).
#[must_use]
pub fn keyframe_track_color(name: &str) -> (Color32, Color32) {
    let deg = ((cm3_string_hash(name) % 360) * 40) % 360;
    let h = deg as f32 / 360.0;
    let fill = rgb_to_color32(Rgb::from_hsl(Hsl {
        h,
        s: 0.40,
        l: 0.45,
    }));
    let border = rgb_to_color32(Rgb::from_hsl(Hsl {
        h,
        s: 0.40,
        l: 0.60,
    }));
    (fill, border)
}

// ---------------------------------------------------------------------------
// Zoidium Theme Profile Switcher & Live Palette State
// ---------------------------------------------------------------------------

/// Selectable Zoidium theme profile (Spec 01 §2.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ThemeProfileChoice {
    #[default]
    Panzoid,
    Signal,
    Ember,
    Moss,
    Rose,
    Gray,
    CustomHueSat,
    CustomMainColor,
}

impl ThemeProfileChoice {
    /// All selectable profile choices in UI order.
    pub const ALL: [Self; 8] = [
        Self::Panzoid,
        Self::Signal,
        Self::Ember,
        Self::Moss,
        Self::Rose,
        Self::Gray,
        Self::CustomHueSat,
        Self::CustomMainColor,
    ];

    /// Display label for Settings dropdowns.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Panzoid => "Panzoid Classic",
            Self::Signal => "Signal Blue",
            Self::Ember => "Ember Console",
            Self::Moss => "Moss Terminal",
            Self::Rose => "Rose Signal",
            Self::Gray => "Gray",
            Self::CustomHueSat => "Custom (Hue / Sat)",
            Self::CustomMainColor => "Custom (Main Color)",
        }
    }

    /// Maps built-in presets to [`zc_color::ThemeProfile`].
    #[must_use]
    pub const fn builtin_profile(self) -> Option<ThemeProfile> {
        match self {
            Self::Panzoid => Some(ThemeProfile::Panzoid),
            Self::Signal => Some(ThemeProfile::Signal),
            Self::Ember => Some(ThemeProfile::Ember),
            Self::Moss => Some(ThemeProfile::Moss),
            Self::Rose => Some(ThemeProfile::Rose),
            Self::Gray => Some(ThemeProfile::Gray),
            Self::CustomHueSat | Self::CustomMainColor => None,
        }
    }
}

/// Active CM3 / Zoidium UI theme holding the derived 5-color accent palette.
#[derive(Debug, Clone, PartialEq)]
pub struct Cm3Theme {
    pub profile: ThemeProfileChoice,
    /// User hue shift in degrees `[-180, 180]` for `CustomHueSat`.
    pub custom_hue_deg: f32,
    /// User saturation percentage `[0, 200]` for `CustomHueSat` (`100` = neutral).
    pub custom_sat_percent: f32,
    /// User main focus color for `CustomMainColor` (Spec 01 §2.6 A).
    pub custom_main_rgb: Rgb,
    /// Derived 5-color accent palette.
    pub palette: ThemePalette,
    /// Global UI font scale multiplier (`1.0` = exact CM3 px sizes).
    pub font_scale: f32,
}

impl Default for Cm3Theme {
    fn default() -> Self {
        Self {
            profile: ThemeProfileChoice::Panzoid,
            custom_hue_deg: 0.0,
            custom_sat_percent: 100.0,
            custom_main_rgb: ThemePalette::CLASSIC.focus,
            palette: ThemePalette::CLASSIC,
            font_scale: 1.0,
        }
    }
}

impl Cm3Theme {
    /// Creates a theme initialized to a built-in profile.
    #[must_use]
    pub fn from_profile(profile: ThemeProfileChoice) -> Self {
        let mut theme = Self {
            profile,
            ..Self::default()
        };
        theme.recompute_palette();
        theme
    }

    /// Selects a theme profile and immediately recomputes the 5-color palette.
    pub fn set_profile(&mut self, profile: ThemeProfileChoice) {
        self.profile = profile;
        self.recompute_palette();
    }

    /// Sets custom main color (`CustomMainColor`) and derives all 5 CM3 colors via HSL offsets.
    pub fn set_custom_main(&mut self, main: Rgb) {
        self.profile = ThemeProfileChoice::CustomMainColor;
        self.custom_main_rgb = main;
        self.recompute_palette();
    }

    /// Sets custom hue/saturation shift (`CustomHueSat`) and derives all 5 CM3 colors.
    pub fn set_custom_hue_sat(&mut self, hue_deg: f32, sat_percent: f32) {
        self.profile = ThemeProfileChoice::CustomHueSat;
        self.custom_hue_deg = hue_deg.clamp(-180.0, 180.0);
        self.custom_sat_percent = sat_percent.clamp(0.0, 200.0);
        self.recompute_palette();
    }

    /// Recomputes `self.palette` from the active profile & custom parameters.
    pub fn recompute_palette(&mut self) {
        self.palette = match self.profile {
            ThemeProfileChoice::Panzoid => ThemeProfile::Panzoid.palette(),
            ThemeProfileChoice::Signal => ThemeProfile::Signal.palette(),
            ThemeProfileChoice::Ember => ThemeProfile::Ember.palette(),
            ThemeProfileChoice::Moss => ThemeProfile::Moss.palette(),
            ThemeProfileChoice::Rose => ThemeProfile::Rose.palette(),
            ThemeProfileChoice::Gray => ThemeProfile::Gray.palette(),
            ThemeProfileChoice::CustomHueSat => {
                let sat_scale = (self.custom_sat_percent / 100.0).clamp(0.0, 2.0);
                ThemePalette::derive_profile(self.custom_hue_deg, sat_scale)
            }
            ThemeProfileChoice::CustomMainColor => {
                ThemePalette::derive_from_rgb(self.custom_main_rgb)
            }
        };
    }

    /// Primary focus & accent (`--zui-accent`, `#384668` in Classic).
    #[must_use]
    pub fn accent(&self) -> Color32 {
        rgb_to_color32(self.palette.focus)
    }

    /// Numeric/input value & link accent (`--zui-value`, `#7e8fb9` in Classic).
    #[must_use]
    pub fn value(&self) -> Color32 {
        rgb_to_color32(self.palette.link)
    }

    /// Unselected timeline clip fill (`#526183` in Classic).
    #[must_use]
    pub fn clip(&self) -> Color32 {
        rgb_to_color32(self.palette.clip)
    }

    /// Selected timeline clip fill (`#7f94c7` in Classic).
    #[must_use]
    pub fn clip_selected(&self) -> Color32 {
        rgb_to_color32(self.palette.clip_selected)
    }

    /// Active clip selection highlight (`#273465` in Classic).
    #[must_use]
    pub fn clip_selection(&self) -> Color32 {
        rgb_to_color32(self.palette.clip_selection)
    }

    /// `--zui-accent-strong`: 70% accent + 30% `#ffffff`.
    #[must_use]
    pub fn accent_strong(&self) -> Color32 {
        rgb_to_color32(self.palette.accent_strong())
    }

    /// `--zui-value-strong`: 60% value + 40% `#ffffff`.
    #[must_use]
    pub fn value_strong(&self) -> Color32 {
        rgb_to_color32(self.palette.value_strong())
    }

    /// `--zui-bg-title-focused`: 22% accent + 78% `#222222`.
    #[must_use]
    pub fn bg_title_focused(&self) -> Color32 {
        rgb_to_color32(self.palette.bg_title_focused())
    }

    /// `--zui-menu-active`: 55% accent + 45% `#2a2a2a`.
    #[must_use]
    pub fn menu_active(&self) -> Color32 {
        rgb_to_color32(self.palette.menu_active())
    }

    /// Whether custom focus lightness exceeds 60% (light extension chrome hint).
    #[must_use]
    pub fn is_light_chrome(&self) -> bool {
        self.profile == ThemeProfileChoice::CustomMainColor && self.palette.is_light_chrome()
    }

    /// Returns scaled monospace font size clamped to legible bounds.
    #[must_use]
    pub fn font(&self, base_px: f32) -> FontId {
        let scale = self.font_scale.clamp(0.75, 1.5);
        font_mono((base_px * scale).round())
    }
}

/// Embedded open-source Geist Mono Regular font bytes (SIL OFL 1.1).
const GEIST_MONO_TTF: &[u8] = include_bytes!("../../assets/fonts/GeistMono-Regular.ttf");

/// Registers the bundled SIL OFL 1.1 monospace font (`Geist Mono`) as the primary
/// typeface for both `Monospace` and `Proportional` families while preserving
/// egui's built-in fallback fonts for extended Unicode symbols.
pub fn install_cm3_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "GeistMono".to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(GEIST_MONO_TTF)),
    );
    if let Some(mono) = fonts.families.get_mut(&FontFamily::Monospace) {
        mono.insert(0, "GeistMono".to_owned());
    }
    if let Some(prop) = fonts.families.get_mut(&FontFamily::Proportional) {
        prop.insert(0, "GeistMono".to_owned());
    }
    ctx.set_fonts(fonts);
}

/// Applies pixel-exact CM3 visuals, spacing, borders, and monospace typography to `ctx`.
pub fn apply_cm3_style(ctx: &egui::Context, theme: &Cm3Theme) {
    let mut style = (*ctx.global_style()).clone();

    let accent = theme.accent();
    let value = theme.value();
    let scale = theme.font_scale.clamp(0.75, 1.5);

    style
        .text_styles
        .insert(TextStyle::Heading, font_mono((FONT_TITLE * scale).round()));
    style
        .text_styles
        .insert(TextStyle::Body, font_mono((FONT_BODY * scale).round()));
    style
        .text_styles
        .insert(TextStyle::Monospace, font_mono((FONT_BODY * scale).round()));
    style
        .text_styles
        .insert(TextStyle::Button, font_mono((FONT_BODY * scale).round()));
    style
        .text_styles
        .insert(TextStyle::Small, font_mono((FONT_SMALL * scale).round()));

    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = BG_PANEL;
    visuals.window_fill = BG_PANEL;
    visuals.extreme_bg_color = BG_INPUT;
    visuals.faint_bg_color = BG_TITLE;
    visuals.code_bg_color = BG_SUNKEN;
    visuals.override_text_color = Some(TEXT_PRIMARY);
    visuals.hyperlink_color = value;
    visuals.warn_fg_color = AUDIO_PEAK_YELLOW;
    visuals.error_fg_color = DANGER_TEXT;
    visuals.window_corner_radius = CornerRadius::ZERO;
    visuals.menu_corner_radius = CornerRadius::ZERO;
    visuals.window_stroke = Stroke::new(1.0, BORDER_PANEL);
    visuals.selection.bg_fill = accent;
    visuals.selection.stroke = Stroke::new(1.0, accent);

    // Noninteractive widget visuals
    visuals.widgets.noninteractive.bg_fill = BG_PANEL;
    visuals.widgets.noninteractive.weak_bg_fill = BG_TITLE;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER_ROW_SOFT);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::ZERO;

    // Inactive widget visuals
    visuals.widgets.inactive.bg_fill = BG_BUTTON;
    visuals.widgets.inactive.weak_bg_fill = BG_INPUT;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER_CONTROL);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_BUTTON);
    visuals.widgets.inactive.corner_radius = CornerRadius::ZERO;

    // Hovered widget visuals
    visuals.widgets.hovered.bg_fill = BG_BUTTON;
    visuals.widgets.hovered.weak_bg_fill = BG_OPTION_HOVER;
    visuals.widgets.hovered.bg_stroke = Stroke::new(2.0, BORDER_BUTTON_HOVER);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, TEXT_BRIGHT);
    visuals.widgets.hovered.corner_radius = CornerRadius::ZERO;
    visuals.widgets.hovered.expansion = 0.0;

    // Active / pressed widget visuals
    visuals.widgets.active.bg_fill = BG_BUTTON_PRESSED;
    visuals.widgets.active.weak_bg_fill = accent;
    visuals.widgets.active.bg_stroke = Stroke::new(2.0, accent);
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, TEXT_BRIGHT);
    visuals.widgets.active.corner_radius = CornerRadius::ZERO;
    visuals.widgets.active.expansion = 0.0;

    // Open popup / combo visuals
    visuals.widgets.open.bg_fill = BG_INPUT;
    visuals.widgets.open.weak_bg_fill = BG_INPUT;
    visuals.widgets.open.bg_stroke = Stroke::new(1.0, accent);
    visuals.widgets.open.fg_stroke = Stroke::new(1.0, value);
    visuals.widgets.open.corner_radius = CornerRadius::ZERO;

    style.visuals = visuals;
    style.spacing.item_spacing = Vec2::new(5.0, 3.0);
    style.spacing.window_margin = Margin::same(0);
    style.spacing.button_padding = Vec2::new(7.0, 5.0);
    style.spacing.scroll.bar_width = 12.0;

    ctx.set_global_style(style);
}

// ---------------------------------------------------------------------------
// Procedural Vector Icon Painter (All CM3 Toolbar, Rail & Action Icons)
// ---------------------------------------------------------------------------

/// Complete set of CM3 and Zoidium vector icons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cm3Icon {
    // File & History Toolbar
    New,
    Load,
    Save,
    Undo,
    Redo,
    // Elevator Rail Tabs
    Menu,
    Project,
    Sequence,
    Edit2,
    Objects,
    Fx,
    Export,
    About,
    Plugins,
    Settings,
    // Playback & Viewport Toolbar
    Camera,
    Transform,
    Start,
    PrevFrame,
    Play,
    Pause,
    NextFrame,
    End,
    Loop,
    Graph,
    // Property & Tree Controls
    Stopwatch,
    KeyframeHollow,
    KeyframeSolid,
    PrevKf,
    NextKf,
    Add,
    Duplicate,
    Remove,
    Reset,
    Code,
    Eyedropper,
    Error,
    // Floating Window & Misc
    Close,
    Maximize,
    Collapse,
    PopOut,
    Eye,
    Lock,
    InterpStandard,
    InterpBezier,
    ChevronDown,
    ChevronRight,
}

/// Maps normalized `[0, 1] x [0, 1]` coordinates into `rect`.
#[inline]
fn pt(rect: Rect, nx: f32, ny: f32) -> Pos2 {
    Pos2::new(
        rect.left() + rect.width() * nx,
        rect.top() + rect.height() * ny,
    )
}

/// Paints a crisp procedural CM3 vector icon inside `rect` using `color`.
pub fn paint_cm3_icon(painter: &Painter, rect: Rect, icon: Cm3Icon, color: Color32) {
    let side = rect.width().min(rect.height()).max(1.0);
    let r = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let sw = (side * 0.085).clamp(1.2, 2.4);
    let stroke = Stroke::new(sw, color);
    let thin = Stroke::new((side * 0.065).clamp(1.0, 1.8), color);

    match icon {
        Cm3Icon::New => {
            // Page outline with folded top-right corner
            let pts = vec![
                pt(r, 0.22, 0.12),
                pt(r, 0.62, 0.12),
                pt(r, 0.78, 0.28),
                pt(r, 0.78, 0.88),
                pt(r, 0.22, 0.88),
            ];
            painter.add(Shape::closed_line(pts, stroke));
            painter.line_segment([pt(r, 0.62, 0.12), pt(r, 0.62, 0.28)], thin);
            painter.line_segment([pt(r, 0.62, 0.28), pt(r, 0.78, 0.28)], thin);
            // Center plus
            painter.line_segment([pt(r, 0.50, 0.44), pt(r, 0.50, 0.72)], stroke);
            painter.line_segment([pt(r, 0.36, 0.58), pt(r, 0.64, 0.58)], stroke);
        }
        Cm3Icon::Load => {
            // Open folder silhouette
            let pts = vec![
                pt(r, 0.12, 0.24),
                pt(r, 0.40, 0.24),
                pt(r, 0.48, 0.34),
                pt(r, 0.86, 0.34),
                pt(r, 0.86, 0.48),
                pt(r, 0.20, 0.48),
                pt(r, 0.12, 0.82),
            ];
            painter.add(Shape::closed_line(pts, thin));
            let front = vec![
                pt(r, 0.22, 0.48),
                pt(r, 0.90, 0.48),
                pt(r, 0.80, 0.82),
                pt(r, 0.12, 0.82),
            ];
            painter.add(Shape::convex_polygon(front, color, Stroke::NONE));
        }
        Cm3Icon::Save => {
            // Floppy disk
            let body = Rect::from_min_max(pt(r, 0.15, 0.15), pt(r, 0.85, 0.85));
            painter.rect_stroke(body, CornerRadius::ZERO, stroke, StrokeKind::Inside);
            let shutter = Rect::from_min_max(pt(r, 0.30, 0.15), pt(r, 0.70, 0.42));
            painter.rect_filled(shutter, CornerRadius::ZERO, color);
            let label = Rect::from_min_max(pt(r, 0.28, 0.56), pt(r, 0.72, 0.85));
            painter.rect_stroke(label, CornerRadius::ZERO, thin, StrokeKind::Inside);
        }
        Cm3Icon::Undo => {
            // Counter-clockwise curved arrow pointing left
            let head = vec![pt(r, 0.14, 0.36), pt(r, 0.38, 0.18), pt(r, 0.38, 0.52)];
            painter.add(Shape::convex_polygon(head, color, Stroke::NONE));
            let arc = vec![
                pt(r, 0.36, 0.36),
                pt(r, 0.66, 0.36),
                pt(r, 0.82, 0.46),
                pt(r, 0.84, 0.64),
                pt(r, 0.74, 0.78),
                pt(r, 0.46, 0.80),
            ];
            painter.add(Shape::line(arc, stroke));
        }
        Cm3Icon::Redo => {
            // Clockwise curved arrow pointing right
            let head = vec![pt(r, 0.86, 0.36), pt(r, 0.62, 0.18), pt(r, 0.62, 0.52)];
            painter.add(Shape::convex_polygon(head, color, Stroke::NONE));
            let arc = vec![
                pt(r, 0.64, 0.36),
                pt(r, 0.34, 0.36),
                pt(r, 0.18, 0.46),
                pt(r, 0.16, 0.64),
                pt(r, 0.26, 0.78),
                pt(r, 0.54, 0.80),
            ];
            painter.add(Shape::line(arc, stroke));
        }
        Cm3Icon::Menu => {
            // 3 crisp hamburger bars
            for ny in [0.26_f32, 0.50, 0.74] {
                let bar = Rect::from_min_max(pt(r, 0.14, ny - 0.055), pt(r, 0.86, ny + 0.055));
                painter.rect_filled(bar, CornerRadius::ZERO, color);
            }
        }
        Cm3Icon::Project => {
            // Media library 2x2 tiles
            let tiles = [
                (0.14, 0.14, 0.46, 0.46),
                (0.54, 0.14, 0.86, 0.46),
                (0.14, 0.54, 0.46, 0.86),
                (0.54, 0.54, 0.86, 0.86),
            ];
            for (x0, y0, x1, y1) in tiles {
                let tile = Rect::from_min_max(pt(r, x0, y0), pt(r, x1, y1));
                painter.rect_filled(tile, CornerRadius::ZERO, color);
            }
        }
        Cm3Icon::Sequence => {
            // Filmstrip / sequence frame
            let outer = Rect::from_min_max(pt(r, 0.12, 0.18), pt(r, 0.88, 0.82));
            painter.rect_stroke(outer, CornerRadius::ZERO, stroke, StrokeKind::Inside);
            painter.line_segment([pt(r, 0.28, 0.18), pt(r, 0.28, 0.82)], thin);
            painter.line_segment([pt(r, 0.72, 0.18), pt(r, 0.72, 0.82)], thin);
            for ny in [0.34_f32, 0.50, 0.66] {
                painter.line_segment([pt(r, 0.12, ny), pt(r, 0.28, ny)], thin);
                painter.line_segment([pt(r, 0.72, ny), pt(r, 0.88, ny)], thin);
            }
            painter.line_segment([pt(r, 0.28, 0.50), pt(r, 0.72, 0.50)], thin);
        }
        Cm3Icon::Edit2 => {
            // Horizontal mixer sliders (3 tracks with knobs)
            for (ny, kx) in [(0.26_f32, 0.36_f32), (0.50, 0.68), (0.74, 0.44)] {
                painter.line_segment([pt(r, 0.14, ny), pt(r, 0.86, ny)], thin);
                let knob =
                    Rect::from_center_size(pt(r, kx, ny), Vec2::new(side * 0.16, side * 0.18));
                painter.rect_filled(knob, CornerRadius::ZERO, color);
            }
        }
        Cm3Icon::Objects => {
            // Isometric 3D cube wireframe
            let top = pt(r, 0.50, 0.12);
            let tr = pt(r, 0.84, 0.30);
            let br = pt(r, 0.84, 0.70);
            let bot = pt(r, 0.50, 0.88);
            let bl = pt(r, 0.16, 0.70);
            let tl = pt(r, 0.16, 0.30);
            let mid = pt(r, 0.50, 0.48);
            painter.add(Shape::closed_line(vec![top, tr, br, bot, bl, tl], stroke));
            painter.line_segment([tl, mid], thin);
            painter.line_segment([tr, mid], thin);
            painter.line_segment([mid, bot], thin);
        }
        Cm3Icon::Fx => {
            // Crisp geometric "FX" / sparkle star
            let c = r.center();
            let outer = side * 0.38;
            let inner = side * 0.14;
            let mut pts = Vec::with_capacity(8);
            for i in 0..8 {
                let ang = (i as f32) * (TAU / 8.0) - std::f32::consts::FRAC_PI_2;
                let rad = if i % 2 == 0 { outer } else { inner };
                pts.push(Pos2::new(c.x + ang.cos() * rad, c.y + ang.sin() * rad));
            }
            painter.add(Shape::closed_line(pts, stroke));
            painter.circle_filled(c, side * 0.08, color);
        }
        Cm3Icon::Export => {
            // Upward export arrow out of tray
            painter.line_segment([pt(r, 0.50, 0.18), pt(r, 0.50, 0.66)], stroke);
            let head = vec![pt(r, 0.50, 0.12), pt(r, 0.70, 0.36), pt(r, 0.30, 0.36)];
            painter.add(Shape::convex_polygon(head, color, Stroke::NONE));
            let tray = vec![
                pt(r, 0.16, 0.60),
                pt(r, 0.16, 0.84),
                pt(r, 0.84, 0.84),
                pt(r, 0.84, 0.60),
            ];
            painter.add(Shape::line(tray, stroke));
        }
        Cm3Icon::About => {
            // Circled "i"
            painter.circle_stroke(r.center(), side * 0.38, stroke);
            painter.circle_filled(pt(r, 0.50, 0.32), side * 0.06, color);
            painter.line_segment([pt(r, 0.50, 0.46), pt(r, 0.50, 0.74)], stroke);
        }
        Cm3Icon::Plugins => {
            // Modular plugin puzzle/plug icon
            let body = Rect::from_min_max(pt(r, 0.22, 0.34), pt(r, 0.78, 0.84));
            painter.rect_stroke(body, CornerRadius::ZERO, stroke, StrokeKind::Inside);
            painter.line_segment([pt(r, 0.36, 0.14), pt(r, 0.36, 0.34)], stroke);
            painter.line_segment([pt(r, 0.64, 0.14), pt(r, 0.64, 0.34)], stroke);
            painter.circle_filled(pt(r, 0.50, 0.59), side * 0.10, color);
        }
        Cm3Icon::Settings => {
            // 8-tooth gear
            let c = r.center();
            let r_in = side * 0.16;
            let r_mid = side * 0.30;
            let r_out = side * 0.40;
            painter.circle_stroke(c, r_mid, stroke);
            painter.circle_stroke(c, r_in, thin);
            for i in 0..8 {
                let ang = (i as f32) * (TAU / 8.0);
                let p0 = Pos2::new(c.x + ang.cos() * r_mid, c.y + ang.sin() * r_mid);
                let p1 = Pos2::new(c.x + ang.cos() * r_out, c.y + ang.sin() * r_out);
                painter.line_segment([p0, p1], stroke);
            }
        }
        Cm3Icon::Camera => {
            // Cinema camera body + right lens cone
            let body = Rect::from_min_max(pt(r, 0.12, 0.28), pt(r, 0.66, 0.74));
            painter.rect_filled(body, CornerRadius::ZERO, color);
            let lens = vec![
                pt(r, 0.66, 0.42),
                pt(r, 0.88, 0.26),
                pt(r, 0.88, 0.74),
                pt(r, 0.66, 0.58),
            ];
            painter.add(Shape::convex_polygon(lens, color, Stroke::NONE));
        }
        Cm3Icon::Transform => {
            // 2D bounding box with 4 corner handles
            let box_rect = Rect::from_min_max(pt(r, 0.22, 0.22), pt(r, 0.78, 0.78));
            painter.rect_stroke(box_rect, CornerRadius::ZERO, thin, StrokeKind::Middle);
            let hs = Vec2::splat(side * 0.16);
            for corner in [
                box_rect.left_top(),
                box_rect.right_top(),
                box_rect.left_bottom(),
                box_rect.right_bottom(),
            ] {
                painter.rect_filled(
                    Rect::from_center_size(corner, hs),
                    CornerRadius::ZERO,
                    color,
                );
            }
        }
        Cm3Icon::Start => {
            // Vertical bar + left-pointing triangle
            let bar = Rect::from_min_max(pt(r, 0.16, 0.20), pt(r, 0.28, 0.80));
            painter.rect_filled(bar, CornerRadius::ZERO, color);
            let tri = vec![pt(r, 0.32, 0.50), pt(r, 0.82, 0.20), pt(r, 0.82, 0.80)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
        Cm3Icon::PrevFrame => {
            // Step back: vertical bar + compact left triangle
            let bar = Rect::from_min_max(pt(r, 0.24, 0.22), pt(r, 0.34, 0.78));
            painter.rect_filled(bar, CornerRadius::ZERO, color);
            let tri = vec![pt(r, 0.40, 0.50), pt(r, 0.78, 0.24), pt(r, 0.78, 0.76)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
        Cm3Icon::Play => {
            let tri = vec![pt(r, 0.24, 0.16), pt(r, 0.84, 0.50), pt(r, 0.24, 0.84)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
        Cm3Icon::Pause => {
            let left = Rect::from_min_max(pt(r, 0.22, 0.18), pt(r, 0.43, 0.82));
            let right = Rect::from_min_max(pt(r, 0.57, 0.18), pt(r, 0.78, 0.82));
            painter.rect_filled(left, CornerRadius::ZERO, color);
            painter.rect_filled(right, CornerRadius::ZERO, color);
        }
        Cm3Icon::NextFrame => {
            let tri = vec![pt(r, 0.22, 0.24), pt(r, 0.60, 0.50), pt(r, 0.22, 0.76)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
            let bar = Rect::from_min_max(pt(r, 0.66, 0.22), pt(r, 0.76, 0.78));
            painter.rect_filled(bar, CornerRadius::ZERO, color);
        }
        Cm3Icon::End => {
            let tri = vec![pt(r, 0.18, 0.20), pt(r, 0.68, 0.50), pt(r, 0.18, 0.80)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
            let bar = Rect::from_min_max(pt(r, 0.72, 0.20), pt(r, 0.84, 0.80));
            painter.rect_filled(bar, CornerRadius::ZERO, color);
        }
        Cm3Icon::Loop => {
            // Two opposing rounded arrows forming a repeat cycle
            let top = vec![pt(r, 0.18, 0.46), pt(r, 0.18, 0.28), pt(r, 0.72, 0.28)];
            painter.add(Shape::line(top, stroke));
            let top_head = vec![pt(r, 0.86, 0.28), pt(r, 0.68, 0.15), pt(r, 0.68, 0.41)];
            painter.add(Shape::convex_polygon(top_head, color, Stroke::NONE));

            let bot = vec![pt(r, 0.82, 0.54), pt(r, 0.82, 0.72), pt(r, 0.28, 0.72)];
            painter.add(Shape::line(bot, stroke));
            let bot_head = vec![pt(r, 0.14, 0.72), pt(r, 0.32, 0.59), pt(r, 0.32, 0.85)];
            painter.add(Shape::convex_polygon(bot_head, color, Stroke::NONE));
        }
        Cm3Icon::Graph | Cm3Icon::InterpBezier => {
            // S-curve with two tangent handles (CM3 interp_1 icon)
            let mut curve = Vec::with_capacity(13);
            for i in 0..=12 {
                let u = i as f32 / 12.0;
                let y = 3.0 * u * u - 2.0 * u * u * u;
                curve.push(pt(r, 0.14 + u * 0.72, 0.82 - y * 0.64));
            }
            painter.add(Shape::line(curve, stroke));
            painter.line_segment([pt(r, 0.14, 0.82), pt(r, 0.46, 0.82)], thin);
            painter.line_segment([pt(r, 0.86, 0.18), pt(r, 0.54, 0.18)], thin);
            painter.circle_filled(pt(r, 0.46, 0.82), side * 0.06, color);
            painter.circle_filled(pt(r, 0.54, 0.18), side * 0.06, color);
        }
        Cm3Icon::InterpStandard => {
            // Standard easing curve (CM3 interp_0 icon)
            let mut curve = Vec::with_capacity(13);
            for i in 0..=12 {
                let u = i as f32 / 12.0;
                let y = u * u;
                curve.push(pt(r, 0.14 + u * 0.72, 0.82 - y * 0.64));
            }
            painter.add(Shape::line(curve, stroke));
            painter.circle_filled(pt(r, 0.14, 0.82), side * 0.06, color);
            painter.circle_filled(pt(r, 0.86, 0.18), side * 0.06, color);
        }
        Cm3Icon::Stopwatch => {
            // Circular clock body + top button + hand
            let c = pt(r, 0.50, 0.56);
            painter.circle_stroke(c, side * 0.32, stroke);
            painter.line_segment([pt(r, 0.36, 0.14), pt(r, 0.64, 0.14)], stroke);
            painter.line_segment([pt(r, 0.50, 0.14), pt(r, 0.50, 0.24)], stroke);
            painter.line_segment([c, pt(r, 0.64, 0.42)], thin);
            painter.circle_filled(c, side * 0.06, color);
        }
        Cm3Icon::KeyframeHollow => {
            let diamond = vec![
                pt(r, 0.50, 0.14),
                pt(r, 0.86, 0.50),
                pt(r, 0.50, 0.86),
                pt(r, 0.14, 0.50),
            ];
            painter.add(Shape::closed_line(diamond, stroke));
        }
        Cm3Icon::KeyframeSolid => {
            let diamond = vec![
                pt(r, 0.50, 0.14),
                pt(r, 0.86, 0.50),
                pt(r, 0.50, 0.86),
                pt(r, 0.14, 0.50),
            ];
            painter.add(Shape::convex_polygon(diamond, color, Stroke::NONE));
        }
        Cm3Icon::PrevKf => {
            let tri = vec![pt(r, 0.22, 0.50), pt(r, 0.74, 0.20), pt(r, 0.74, 0.80)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
        Cm3Icon::NextKf => {
            let tri = vec![pt(r, 0.78, 0.50), pt(r, 0.26, 0.20), pt(r, 0.26, 0.80)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
        Cm3Icon::Add => {
            painter.line_segment([pt(r, 0.50, 0.16), pt(r, 0.50, 0.84)], stroke);
            painter.line_segment([pt(r, 0.16, 0.50), pt(r, 0.84, 0.50)], stroke);
        }
        Cm3Icon::Duplicate => {
            let back = Rect::from_min_max(pt(r, 0.32, 0.14), pt(r, 0.84, 0.66));
            painter.rect_stroke(back, CornerRadius::ZERO, thin, StrokeKind::Inside);
            let front = Rect::from_min_max(pt(r, 0.16, 0.34), pt(r, 0.68, 0.86));
            painter.rect_filled(front, CornerRadius::ZERO, color);
        }
        Cm3Icon::Remove => {
            // Trash can
            painter.line_segment([pt(r, 0.18, 0.26), pt(r, 0.82, 0.26)], stroke);
            painter.line_segment([pt(r, 0.38, 0.16), pt(r, 0.62, 0.16)], stroke);
            let bin = Rect::from_min_max(pt(r, 0.24, 0.28), pt(r, 0.76, 0.86));
            painter.rect_stroke(bin, CornerRadius::ZERO, thin, StrokeKind::Inside);
            painter.line_segment([pt(r, 0.42, 0.40), pt(r, 0.42, 0.74)], thin);
            painter.line_segment([pt(r, 0.58, 0.40), pt(r, 0.58, 0.74)], thin);
        }
        Cm3Icon::Reset => {
            // Circular reset arrow
            let c = r.center();
            let rad = side * 0.32;
            let mut pts = Vec::with_capacity(14);
            for i in 0..=12 {
                let ang = -0.3 + (i as f32 / 12.0) * (TAU * 0.78);
                pts.push(Pos2::new(c.x + ang.cos() * rad, c.y + ang.sin() * rad));
            }
            painter.add(Shape::line(pts, stroke));
            let head = vec![pt(r, 0.84, 0.18), pt(r, 0.86, 0.46), pt(r, 0.60, 0.34)];
            painter.add(Shape::convex_polygon(head, color, Stroke::NONE));
        }
        Cm3Icon::Code => {
            // Angle brackets < / >
            painter.add(Shape::line(
                vec![pt(r, 0.36, 0.26), pt(r, 0.14, 0.50), pt(r, 0.36, 0.74)],
                stroke,
            ));
            painter.add(Shape::line(
                vec![pt(r, 0.64, 0.26), pt(r, 0.86, 0.50), pt(r, 0.64, 0.74)],
                stroke,
            ));
            painter.line_segment([pt(r, 0.56, 0.22), pt(r, 0.44, 0.78)], thin);
        }
        Cm3Icon::Eyedropper => {
            // Diagonal pipette
            painter.line_segment([pt(r, 0.18, 0.82), pt(r, 0.68, 0.32)], stroke);
            painter.line_segment([pt(r, 0.52, 0.24), pt(r, 0.76, 0.48)], stroke);
            painter.circle_filled(pt(r, 0.76, 0.24), side * 0.13, color);
        }
        Cm3Icon::Error => {
            // Warning triangle with exclamation
            let tri = vec![pt(r, 0.50, 0.14), pt(r, 0.88, 0.84), pt(r, 0.12, 0.84)];
            painter.add(Shape::closed_line(tri, stroke));
            painter.line_segment([pt(r, 0.50, 0.38), pt(r, 0.50, 0.62)], stroke);
            painter.circle_filled(pt(r, 0.50, 0.73), side * 0.05, color);
        }
        Cm3Icon::Close => {
            painter.line_segment([pt(r, 0.20, 0.20), pt(r, 0.80, 0.80)], stroke);
            painter.line_segment([pt(r, 0.80, 0.20), pt(r, 0.20, 0.80)], stroke);
        }
        Cm3Icon::Maximize => {
            let box_rect = Rect::from_min_max(pt(r, 0.18, 0.18), pt(r, 0.82, 0.82));
            painter.rect_stroke(box_rect, CornerRadius::ZERO, stroke, StrokeKind::Inside);
            painter.line_segment([pt(r, 0.18, 0.32), pt(r, 0.82, 0.32)], stroke);
        }
        Cm3Icon::Collapse => {
            painter.line_segment([pt(r, 0.18, 0.54), pt(r, 0.82, 0.54)], stroke);
        }
        Cm3Icon::PopOut => {
            let box_rect = Rect::from_min_max(pt(r, 0.16, 0.32), pt(r, 0.68, 0.84));
            painter.rect_stroke(box_rect, CornerRadius::ZERO, thin, StrokeKind::Inside);
            painter.line_segment([pt(r, 0.46, 0.54), pt(r, 0.84, 0.16)], stroke);
            painter.line_segment([pt(r, 0.56, 0.16), pt(r, 0.84, 0.16)], stroke);
            painter.line_segment([pt(r, 0.84, 0.16), pt(r, 0.84, 0.44)], stroke);
        }
        Cm3Icon::Eye => {
            let top = vec![
                pt(r, 0.10, 0.50),
                pt(r, 0.32, 0.28),
                pt(r, 0.68, 0.28),
                pt(r, 0.90, 0.50),
            ];
            let bot = vec![
                pt(r, 0.10, 0.50),
                pt(r, 0.32, 0.72),
                pt(r, 0.68, 0.72),
                pt(r, 0.90, 0.50),
            ];
            painter.add(Shape::line(top, thin));
            painter.add(Shape::line(bot, thin));
            painter.circle_filled(r.center(), side * 0.13, color);
        }
        Cm3Icon::Lock => {
            let body = Rect::from_min_max(pt(r, 0.22, 0.46), pt(r, 0.78, 0.86));
            painter.rect_filled(body, CornerRadius::ZERO, color);
            let shackle = vec![
                pt(r, 0.34, 0.46),
                pt(r, 0.34, 0.26),
                pt(r, 0.50, 0.16),
                pt(r, 0.66, 0.26),
                pt(r, 0.66, 0.46),
            ];
            painter.add(Shape::line(shackle, stroke));
        }
        Cm3Icon::ChevronDown => {
            let tri = vec![pt(r, 0.22, 0.32), pt(r, 0.78, 0.32), pt(r, 0.50, 0.72)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
        Cm3Icon::ChevronRight => {
            let tri = vec![pt(r, 0.32, 0.22), pt(r, 0.72, 0.50), pt(r, 0.32, 0.78)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
    }
}

/// Plots a miniature easing curve inside `rect` (used in `35x23` easing buttons & 33-curve dropdown).
pub fn paint_easing_preview(
    painter: &Painter,
    rect: Rect,
    easing: EasingKind,
    stroke_color: Color32,
) {
    let inner = rect.shrink2(Vec2::new(4.0, 4.0));
    if inner.width() <= 2.0 || inner.height() <= 2.0 {
        return;
    }

    // Subtle baseline & ceiling guides
    let guide = Stroke::new(1.0, Color32::from_rgb(0x30, 0x30, 0x30));
    painter.line_segment([inner.left_bottom(), inner.right_bottom()], guide);
    painter.line_segment([inner.left_top(), inner.right_top()], guide);

    let stroke = Stroke::new(1.5, stroke_color);
    if matches!(easing, EasingKind::Hold) {
        let mid_right = Pos2::new(inner.right(), inner.bottom());
        painter.line_segment([inner.left_bottom(), mid_right], stroke);
        painter.line_segment([mid_right, inner.right_top()], stroke);
        return;
    }

    const SAMPLES: usize = 20;
    let mut pts = Vec::with_capacity(SAMPLES + 1);
    for i in 0..=SAMPLES {
        let u = i as f64 / SAMPLES as f64;
        let v = eval_easing(easing, u).clamp(-0.25, 1.25) as f32;
        let x = inner.left() + inner.width() * (u as f32);
        let y = inner.bottom() - inner.height() * v;
        pts.push(Pos2::new(x, y));
    }
    painter.add(Shape::line(pts, stroke));
}

/// Paints a small pill badge (used for `"In Use"`, `"Beta"`, error counts).
pub fn paint_badge(painter: &Painter, rect: Rect, text: &str, bg: Color32, fg: Color32) {
    painter.rect_filled(rect, CornerRadius::same(8), bg);
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        text,
        font_mono(FONT_BADGE),
        fg,
    );
}
