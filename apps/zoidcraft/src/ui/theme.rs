//! Pixel-exact ClipMaker 3 dark chrome palette, Zoidium HSL theme derivation,
//! exact typography scale, and procedural vector icon painter.

use std::f32::consts::{FRAC_PI_4, TAU};

use egui::{
    Align2, Color32, CornerRadius, FontFamily, FontId, Margin, Mesh, Painter, Pos2, Rect, Shape,
    Stroke, StrokeKind, TextStyle, Vec2,
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
            Self::Panzoid => "Classic Dark",
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

/// Selectable editor UI monospace font (OFL).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EditorFontChoice {
    #[default]
    SourceCodePro,
    GeistMono,
}

impl EditorFontChoice {
    pub const ALL: [Self; 2] = [Self::SourceCodePro, Self::GeistMono];

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::SourceCodePro => "Source Code Pro",
            Self::GeistMono => "Geist Mono",
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
    /// Optional per-slot color overrides `[focus, link, clip, clip_selected, clip_selection]`.
    pub slot_overrides: [Option<Rgb>; 5],
    /// Derived 5-color accent palette.
    pub palette: ThemePalette,
    /// Global UI font scale multiplier (`1.0` = exact CM3 px sizes).
    pub font_scale: f32,
    /// Selected editor monospace typeface.
    pub editor_font: EditorFontChoice,
}

impl Default for Cm3Theme {
    fn default() -> Self {
        Self {
            profile: ThemeProfileChoice::Panzoid,
            custom_hue_deg: 0.0,
            custom_sat_percent: 100.0,
            custom_main_rgb: ThemePalette::CLASSIC.focus,
            slot_overrides: [None; 5],
            palette: ThemePalette::CLASSIC,
            font_scale: 1.0,
            editor_font: EditorFontChoice::SourceCodePro,
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

    /// Recomputes `self.palette` from the active profile, custom parameters, and per-slot overrides.
    pub fn recompute_palette(&mut self) {
        let mut pal = match self.profile {
            // Exact canonical Zoidium / CM3 Classic colors (#384668 accent, #526183 clip).
            ThemeProfileChoice::Panzoid => ThemePalette::CLASSIC,
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
        if let Some(c) = self.slot_overrides[0] {
            pal.focus = c;
        }
        if let Some(c) = self.slot_overrides[1] {
            pal.link = c;
        }
        if let Some(c) = self.slot_overrides[2] {
            pal.clip = c;
        }
        if let Some(c) = self.slot_overrides[3] {
            pal.clip_selected = c;
        }
        if let Some(c) = self.slot_overrides[4] {
            pal.clip_selection = c;
        }
        self.palette = pal;
    }

    /// Sets an individual palette slot override (`0=focus, 1=link, 2=clip, 3=clipSelected, 4=clipSelection`).
    pub fn set_slot_override(&mut self, slot: usize, rgb: Rgb) {
        if slot < 5 {
            self.slot_overrides[slot] = Some(rgb);
            self.recompute_palette();
        }
    }

    /// Clears all per-slot color overrides and restores derived profile colors.
    pub fn clear_slot_overrides(&mut self) {
        self.slot_overrides = [None; 5];
        self.recompute_palette();
    }

    /// Exports the active theme palette as a Zoidium-compatible JSON object string.
    #[must_use]
    pub fn export_theme_json(&self) -> String {
        let hex = |rgb: Rgb| {
            let [r, g, b] = rgb.to_u8();
            format!("#{r:02x}{g:02x}{b:02x}")
        };
        serde_json::json!({
            "profile": self.profile.label(),
            "focus": hex(self.palette.focus),
            "link": hex(self.palette.link),
            "clip": hex(self.palette.clip),
            "clipSelected": hex(self.palette.clip_selected),
            "clipSelection": hex(self.palette.clip_selection),
        })
        .to_string()
    }

    /// Imports a Zoidium-compatible theme JSON object (`focus`, `link`, `clip`, `clipSelected`, `clipSelection`).
    pub fn import_theme_json(&mut self, text: &str) -> bool {
        let Ok(val) = serde_json::from_str::<serde_json::Value>(text.trim()) else {
            return false;
        };
        let Some(obj) = val.as_object() else {
            return false;
        };
        let keys = ["focus", "link", "clip", "clipSelected", "clipSelection"];
        let mut any = false;
        for (slot, k) in keys.iter().enumerate() {
            if let Some(s) = obj.get(*k).and_then(serde_json::Value::as_str)
                && let Ok(rgba) = Rgba::from_hex(s.trim())
            {
                self.slot_overrides[slot] = Some(Rgb::new(rgba.0[0], rgba.0[1], rgba.0[2]));
                any = true;
            }
        }
        if any {
            self.recompute_palette();
        }
        any
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

    /// Returns a high-contrast foreground color (`TEXT_BRIGHT` or near-black `#141414`)
    /// selected by WCAG 2.1 relative-luminance contrast ratio against `bg`.
    #[must_use]
    pub fn contrast_fg(&self, bg: Color32) -> Color32 {
        fn srgb_to_linear(channel: u8) -> f32 {
            let c = f32::from(channel) / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        }
        fn rel_lum(c: Color32) -> f32 {
            0.2126 * srgb_to_linear(c.r())
                + 0.7152 * srgb_to_linear(c.g())
                + 0.0722 * srgb_to_linear(c.b())
        }
        fn contrast_ratio(l1: f32, l2: f32) -> f32 {
            let (hi, lo) = if l1 >= l2 { (l1, l2) } else { (l2, l1) };
            (hi + 0.05) / (lo + 0.05)
        }

        const DARK_FG: Color32 = Color32::from_rgb(0x14, 0x14, 0x14);
        let bg_lum = rel_lum(bg);
        let light_ratio = contrast_ratio(rel_lum(TEXT_BRIGHT), bg_lum);
        let dark_ratio = contrast_ratio(rel_lum(DARK_FG), bg_lum);
        if dark_ratio > light_ratio {
            DARK_FG
        } else {
            TEXT_BRIGHT
        }
    }
}

/// Embedded Adobe Source Code Pro Regular font bytes (SIL OFL 1.1).
const SOURCE_CODE_PRO_TTF: &[u8] = include_bytes!("../../assets/fonts/SourceCodePro-Regular.ttf");
/// Embedded open-source Geist Mono Regular font bytes (SIL OFL 1.1).
const GEIST_MONO_TTF: &[u8] = include_bytes!("../../assets/fonts/GeistMono-Regular.ttf");

/// Standard OS CJK fallback font candidates probed once at startup via safe `std::fs::read`
/// so Japanese/Chinese/Korean text renders crisply in egui controls without tofu boxes.
#[cfg(not(target_arch = "wasm32"))]
const SYSTEM_CJK_FALLBACK_PATHS: &[&str] = &[
    // macOS
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    // Windows
    r"C:\Windows\Fonts\YuGothR.ttc",
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    // Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
];

#[cfg(not(target_arch = "wasm32"))]
fn load_system_cjk_fallback_font() -> Option<std::sync::Arc<egui::FontData>> {
    static CACHED: std::sync::OnceLock<Option<std::sync::Arc<egui::FontData>>> =
        std::sync::OnceLock::new();
    CACHED
        .get_or_init(|| {
            for path in SYSTEM_CJK_FALLBACK_PATHS {
                if let Ok(bytes) = std::fs::read(path)
                    && !bytes.is_empty()
                {
                    let mut font_data = egui::FontData::from_owned(bytes);
                    font_data.index = 0;
                    return Some(std::sync::Arc::new(font_data));
                }
            }
            None
        })
        .clone()
}

#[cfg(target_arch = "wasm32")]
fn load_system_cjk_fallback_font() -> Option<std::sync::Arc<egui::FontData>> {
    static LOGGED_ONCE: std::sync::Once = std::sync::Once::new();
    LOGGED_ONCE.call_once(|| {
        tracing::debug!(
            "OS CJK system font probing is unavailable on wasm32; using bundled monospace/proportional and egui fallback fonts"
        );
    });
    None
}

/// Rebuilds `FontDefinitions` with both bundled SIL OFL 1.1 fonts (`"source_code_pro"` and
/// `"geist_mono"`) registered and the selected `font` placed first in both `Monospace` and
/// `Proportional` families while preserving egui's built-in fallback fonts and an optional
/// OS CJK fallback font appended after primary UI fonts.
pub fn apply_editor_font(ctx: &egui::Context, font: EditorFontChoice) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "source_code_pro".to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(SOURCE_CODE_PRO_TTF)),
    );
    fonts.font_data.insert(
        "geist_mono".to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(GEIST_MONO_TTF)),
    );
    let has_cjk_fallback = if let Some(cjk_font) = load_system_cjk_fallback_font() {
        fonts
            .font_data
            .insert("system_cjk_fallback".to_owned(), cjk_font);
        true
    } else {
        false
    };
    let (primary, secondary) = match font {
        EditorFontChoice::SourceCodePro => ("source_code_pro", "geist_mono"),
        EditorFontChoice::GeistMono => ("geist_mono", "source_code_pro"),
    };
    for family in [FontFamily::Monospace, FontFamily::Proportional] {
        let list = fonts.families.entry(family).or_default();
        list.retain(|name| name != primary && name != secondary && name != "system_cjk_fallback");
        list.insert(0, secondary.to_owned());
        list.insert(0, primary.to_owned());
        if has_cjk_fallback {
            list.push("system_cjk_fallback".to_owned());
        }
    }
    ctx.set_fonts(fonts);
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::NULL.with("cm3_active_editor_font"), font);
    });
}

/// Registers both bundled SIL OFL 1.1 monospace fonts (`Source Code Pro` and `Geist Mono`),
/// with `Source Code Pro` set as the default primary typeface for both `Monospace` and
/// `Proportional` families.
pub fn install_cm3_fonts(ctx: &egui::Context) {
    apply_editor_font(ctx, EditorFontChoice::SourceCodePro);
}

/// Applies pixel-exact CM3 visuals, spacing, borders, and monospace typography to `ctx`.
pub fn apply_cm3_style(ctx: &egui::Context, theme: &Cm3Theme) {
    let active_font =
        ctx.data(|d| d.get_temp::<EditorFontChoice>(egui::Id::NULL.with("cm3_active_editor_font")));
    if active_font != Some(theme.editor_font) {
        apply_editor_font(ctx, theme.editor_font);
    }

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
    let accent_fg = theme.contrast_fg(accent);
    visuals.selection.bg_fill = accent;
    visuals.selection.stroke = Stroke::new(1.0, accent_fg);

    // Noninteractive widget visuals
    visuals.widgets.noninteractive.bg_fill = BG_PANEL;
    visuals.widgets.noninteractive.weak_bg_fill = BG_TITLE;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER_ROW_SOFT);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::ZERO;

    // Inactive widget visuals
    visuals.widgets.inactive.bg_fill = BG_BUTTON;
    visuals.widgets.inactive.weak_bg_fill = BG_INPUT;
    visuals.widgets.inactive.bg_stroke = Stroke::NONE;
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
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, accent_fg);
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
    ctx.data_mut(|d| {
        d.insert_temp(
            egui::Id::NULL.with("cm3_badge_font"),
            theme.font(FONT_BADGE),
        );
    });
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
    TabRestore,
    About,
    Plugins,
    TabAssets,
    Settings,
    // Playback & Viewport Toolbar
    Camera,
    Transform,
    TransparencyGrid,
    Start,
    PrevFrame,
    Play,
    Pause,
    NextFrame,
    End,
    Loop,
    Graph,
    SoloCircle,
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

/// Appends an unfeathered solid quad into `mesh` (zero epaint anti-alias feather bleed).
#[inline]
fn push_crisp_quad(mesh: &mut Mesh, rect: Rect, color: Color32) {
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return;
    }
    let idx = mesh.vertices.len() as u32;
    mesh.colored_vertex(rect.left_top(), color);
    mesh.colored_vertex(rect.right_top(), color);
    mesh.colored_vertex(rect.right_bottom(), color);
    mesh.colored_vertex(rect.left_bottom(), color);
    mesh.add_triangle(idx, idx + 1, idx + 2);
    mesh.add_triangle(idx, idx + 2, idx + 3);
}

/// Paints a solid axis-rotated rectangle centered at `center`.
fn paint_rotated_rect(
    painter: &Painter,
    center: Pos2,
    angle: f32,
    half_len: f32,
    half_thick: f32,
    color: Color32,
) {
    let (sin, cos) = angle.sin_cos();
    let dir = Vec2::new(cos, sin);
    let perp = Vec2::new(-sin, cos);
    let pts = vec![
        center - dir * half_len - perp * half_thick,
        center + dir * half_len - perp * half_thick,
        center + dir * half_len + perp * half_thick,
        center - dir * half_len + perp * half_thick,
    ];
    painter.add(Shape::convex_polygon(pts, color, Stroke::NONE));
}

/// Paints a solid closed folder silhouette inside normalized bounds `[left..right] x [top..bottom]`.
fn paint_solid_folder(
    painter: &Painter,
    r: Rect,
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
    color: Color32,
) {
    let w = right - left;
    let h = bottom - top;
    let tab_bottom = top + h * 0.18;
    let tab = vec![
        pt(r, left, top),
        pt(r, left + w * 0.38, top),
        pt(r, left + w * 0.46, tab_bottom),
        pt(r, left, tab_bottom),
    ];
    painter.add(Shape::convex_polygon(tab, color, Stroke::NONE));
    let body = Rect::from_min_max(pt(r, left, tab_bottom - h * 0.02), pt(r, right, bottom));
    painter.rect_filled(body, CornerRadius::same(1), color);
}

/// Paints a solid tapered curved swooping arrow (left-pointing for Undo, mirrored for Redo).
fn paint_swoop_arrow(painter: &Painter, r: Rect, mirror_x: bool, color: Color32) {
    let mx = |nx: f32| if mirror_x { 1.0 - nx } else { nx };
    let qbez = |p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), t: f32| {
        let mt = 1.0 - t;
        let x = mt * mt * p0.0 + 2.0 * mt * t * p1.0 + t * t * p2.0;
        let y = mt * mt * p0.1 + 2.0 * mt * t * p1.1 + t * t * p2.1;
        pt(r, mx(x), y)
    };
    let head = vec![
        pt(r, mx(0.18), 0.34),
        pt(r, mx(0.47), 0.13),
        pt(r, mx(0.47), 0.54),
    ];
    painter.add(Shape::convex_polygon(head, color, Stroke::NONE));

    let o0 = (0.44, 0.25);
    let o1 = (0.98, 0.28);
    let o2 = (0.66, 0.89);
    let i0 = (0.44, 0.43);
    let i1 = (0.80, 0.46);
    let i2 = (0.54, 0.83);
    const STEPS: usize = 12;
    for s in 0..STEPS {
        let t0 = s as f32 / STEPS as f32;
        let t1 = ((s + 1) as f32 / STEPS as f32 + if s + 1 < STEPS { 0.015 } else { 0.0 }).min(1.0);
        let quad = vec![
            qbez(o0, o1, o2, t0),
            qbez(o0, o1, o2, t1),
            qbez(i0, i1, i2, t1),
            qbez(i0, i1, i2, t0),
        ];
        painter.add(Shape::convex_polygon(quad, color, Stroke::NONE));
    }
    let tip = vec![
        qbez(o0, o1, o2, 1.0),
        pt(r, mx(0.58), 0.90),
        qbez(i0, i1, i2, 1.0),
    ];
    painter.add(Shape::convex_polygon(tip, color, Stroke::NONE));
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
            // Crisp thin 8-spoked asterisk / starburst (21x21 at 25px)
            let c = r.center();
            let half_len = side * 0.39;
            let half_thick = side * 0.052;
            for i in 0..4 {
                paint_rotated_rect(
                    painter,
                    c,
                    (i as f32) * FRAC_PI_4,
                    half_len,
                    half_thick,
                    color,
                );
            }
        }
        Cm3Icon::Load => {
            // Solid filled flat closed folder (21x19 at 25px)
            paint_solid_folder(painter, r, 0.10, 0.90, 0.14, 0.86, color);
        }
        Cm3Icon::Save => {
            // Solid downward arrow over a single flat horizontal bottom bar (head y=21..26 at 25px)
            let shaft = Rect::from_min_max(pt(r, 0.44, 0.10), pt(r, 0.56, 0.58));
            painter.rect_filled(shaft, CornerRadius::ZERO, color);
            let head = vec![pt(r, 0.50, 0.78), pt(r, 0.24, 0.54), pt(r, 0.76, 0.54)];
            painter.add(Shape::convex_polygon(head, color, Stroke::NONE));
            let tray_bottom = Rect::from_min_max(pt(r, 0.14, 0.78), pt(r, 0.86, 0.90));
            painter.rect_filled(tray_bottom, CornerRadius::ZERO, color);
        }
        Cm3Icon::Undo => {
            // Thick solid curved swooping arrow pointing left
            paint_swoop_arrow(painter, r, false, color);
        }
        Cm3Icon::Redo => {
            // Thick solid curved swooping arrow pointing right
            paint_swoop_arrow(painter, r, true, color);
        }
        Cm3Icon::Menu => {
            // 3 unfeathered crisp 2px full-width horizontal bars at y = +6, +14.5, +24 (at 30px)
            let bar_h = (side * (2.0 / 30.0)).round().max(1.0);
            let x0 = r.left().round();
            let x1 = r.right().round().max(x0 + 1.0);
            let mut mesh = Mesh::default();
            for (frac, snap_y) in [
                (6.0_f32 / 30.0, true),
                (14.5 / 30.0, false),
                (24.0 / 30.0, true),
            ] {
                let raw_y = r.top() + side * frac;
                let y0 = if snap_y {
                    raw_y.round()
                } else {
                    (raw_y * 2.0).round() * 0.5
                };
                push_crisp_quad(
                    &mut mesh,
                    Rect::from_min_max(Pos2::new(x0, y0), Pos2::new(x1, y0 + bar_h)),
                    color,
                );
            }
            painter.add(Shape::mesh(mesh));
        }
        Cm3Icon::Project => {
            // Solid filled closed folder (30x24 at 30px, tab top-left)
            paint_solid_folder(painter, r, 0.01, 0.99, 0.10, 0.90, color);
        }
        Cm3Icon::Sequence => {
            // Two crisp 2px vertical end-bars framing 3 staggered 3px horizontal clip bars
            let mut mesh = Mesh::default();
            let bar_w = (side * (2.0 / 30.0)).round().max(1.0);
            let clip_h = (side * (3.0 / 30.0)).round().max(1.0);
            let y_top = (r.top() + side * (7.0 / 30.0)).round();
            let y_bot = (r.top() + side * (23.0 / 30.0)).round();
            let lx = r.left().round();
            let rx = (r.right() - bar_w).round();
            push_crisp_quad(
                &mut mesh,
                Rect::from_min_max(Pos2::new(lx, y_top), Pos2::new(lx + bar_w, y_bot)),
                color,
            );
            push_crisp_quad(
                &mut mesh,
                Rect::from_min_max(Pos2::new(rx, y_top), Pos2::new(rx + bar_w, y_bot)),
                color,
            );
            for (x0, x1, y_frac) in [
                (0.10_f32, 0.40_f32, 9.0_f32 / 30.0),
                (0.17, 0.63, 13.5 / 30.0),
                (0.47, 0.83, 18.0 / 30.0),
            ] {
                let cx0 = (r.left() + side * x0).round();
                let cx1 = (r.left() + side * x1).round().max(cx0 + 1.0);
                let cy0 = (r.top() + side * y_frac).round();
                push_crisp_quad(
                    &mut mesh,
                    Rect::from_min_max(Pos2::new(cx0, cy0), Pos2::new(cx1, cy0 + clip_h)),
                    color,
                );
            }
            painter.add(Shape::mesh(mesh));
        }
        Cm3Icon::Edit2 => {
            // Solid filled diagonal pencil tilted 45° (22x22 at 30px)
            let dp = |s: f32, w: f32| pt(r, 0.15 + s + w, 0.83 - s + w);
            let hw = 0.095;
            let tip = vec![dp(0.0, 0.0), dp(0.18, -hw), dp(0.18, hw)];
            painter.add(Shape::convex_polygon(tip, color, Stroke::NONE));
            let shaft = vec![dp(0.17, -hw), dp(0.64, -hw), dp(0.64, hw), dp(0.17, hw)];
            painter.add(Shape::convex_polygon(shaft, color, Stroke::NONE));
        }
        Cm3Icon::Objects => {
            // Oblique wireframe 3D cube (28x26 at 30px; thicker edges at Media card size)
            let stroke_w = if side <= 34.0 {
                (side * 0.058).clamp(1.2, 1.8)
            } else {
                (side * 0.068).clamp(2.2, 3.6)
            };
            let cube_stroke = Stroke::new(stroke_w, color);
            let back = Rect::from_min_max(pt(r, 0.08, 0.10), pt(r, 0.65, 0.65));
            let front = Rect::from_min_max(pt(r, 0.35, 0.36), pt(r, 0.92, 0.90));
            painter.rect_stroke(back, CornerRadius::ZERO, cube_stroke, StrokeKind::Middle);
            painter.rect_stroke(front, CornerRadius::ZERO, cube_stroke, StrokeKind::Middle);
            painter.line_segment([back.left_top(), front.left_top()], cube_stroke);
            painter.line_segment([back.right_top(), front.right_top()], cube_stroke);
            painter.line_segment([back.left_bottom(), front.left_bottom()], cube_stroke);
            painter.line_segment([back.right_bottom(), front.right_bottom()], cube_stroke);
        }
        Cm3Icon::Fx => {
            // Solid filled 9-pointed irregular starburst polygon with narrower spikes & compact core
            let c = r.center();
            const SPIKES: usize = 9;
            // (angle_deg, outer_radius_frac, inner_radius_frac)
            let spec: [(f32, f32, f32); SPIKES] = [
                (-96.0, 0.47, 0.22),
                (-66.0, 0.43, 0.21),
                (-12.0, 0.48, 0.23),
                (25.0, 0.48, 0.22),
                (65.0, 0.43, 0.21),
                (96.0, 0.48, 0.22),
                (142.0, 0.44, 0.21),
                (176.0, 0.49, 0.23),
                (212.0, 0.47, 0.22),
            ];
            let polar = |deg: f32, rad_frac: f32| {
                let a = deg.to_radians();
                Pos2::new(
                    c.x + a.cos() * side * rad_frac,
                    c.y + a.sin() * side * rad_frac,
                )
            };
            painter.circle_filled(c, side * 0.22, color);
            for i in 0..SPIKES {
                let (deg_cur, r_out, r_in_cur) = spec[i];
                let (deg_prev, _, r_in_prev) = spec[(i + SPIKES - 1) % SPIKES];
                let (deg_next, _, _) = spec[(i + 1) % SPIKES];
                let mid_prev = {
                    let mut d = deg_cur - deg_prev;
                    if d < 0.0 {
                        d += 360.0;
                    }
                    deg_cur - d * 0.5 - 1.0
                };
                let mid_next = {
                    let mut d = deg_next - deg_cur;
                    if d < 0.0 {
                        d += 360.0;
                    }
                    deg_cur + d * 0.5 + 1.0
                };
                let kite = vec![
                    c,
                    polar(mid_prev, r_in_prev),
                    polar(deg_cur, r_out),
                    polar(mid_next, r_in_cur),
                ];
                painter.add(Shape::convex_polygon(kite, color, Stroke::NONE));
            }
        }
        Cm3Icon::Export => {
            // Wide tall U-tray (4px arms & bottom at 30px) + narrow upward arrow above it
            let shaft = Rect::from_min_max(pt(r, 0.44, 0.28), pt(r, 0.56, 0.63));
            painter.rect_filled(shaft, CornerRadius::ZERO, color);
            let head = vec![pt(r, 0.50, 0.08), pt(r, 0.67, 0.30), pt(r, 0.33, 0.30)];
            painter.add(Shape::convex_polygon(head, color, Stroke::NONE));
            let tray_bottom = Rect::from_min_max(pt(r, 0.07, 0.85), pt(r, 0.93, 0.96));
            let tray_left = Rect::from_min_max(pt(r, 0.07, 0.38), pt(r, 0.18, 0.96));
            let tray_right = Rect::from_min_max(pt(r, 0.82, 0.38), pt(r, 0.93, 0.96));
            painter.rect_filled(tray_bottom, CornerRadius::ZERO, color);
            painter.rect_filled(tray_left, CornerRadius::ZERO, color);
            painter.rect_filled(tray_right, CornerRadius::ZERO, color);
        }
        Cm3Icon::TabRestore => {
            // Clockwise circular restore arrow with top right-pointing triangular head (25x29 at 30px)
            let c = pt(r, 0.51, 0.52);
            let rad = side * 0.35;
            let arc_stroke = Stroke::new((side * 0.135).clamp(1.8, 4.0), color);
            let start_deg = -22.0_f32;
            let end_deg = 268.0_f32;
            const SEGS: usize = 24;
            let mut pts = Vec::with_capacity(SEGS + 1);
            for i in 0..=SEGS {
                let u = i as f32 / SEGS as f32;
                let ang = (start_deg + (end_deg - start_deg) * u).to_radians();
                pts.push(Pos2::new(c.x + ang.cos() * rad, c.y + ang.sin() * rad));
            }
            painter.add(Shape::line(pts, arc_stroke));
            let head = vec![pt(r, 0.73, 0.18), pt(r, 0.50, 0.01), pt(r, 0.50, 0.34)];
            painter.add(Shape::convex_polygon(head, color, Stroke::NONE));
        }
        Cm3Icon::About => {
            // Uncircled bold serif 'i' with detached round dot (12x29 at 30px)
            painter.circle_filled(pt(r, 0.50, 0.13), side * 0.09, color);
            let mut mesh = Mesh::default();
            let snap_box = |x0: f32, y0: f32, x1: f32, y1: f32| {
                Rect::from_min_max(
                    Pos2::new(
                        (r.left() + side * x0).round(),
                        (r.top() + side * y0).round(),
                    ),
                    Pos2::new(
                        (r.left() + side * x1).round(),
                        (r.top() + side * y1).round(),
                    ),
                )
            };
            push_crisp_quad(&mut mesh, snap_box(0.41, 0.30, 0.59, 0.93), color);
            push_crisp_quad(&mut mesh, snap_box(0.31, 0.30, 0.59, 0.40), color);
            push_crisp_quad(&mut mesh, snap_box(0.31, 0.86, 0.69, 0.96), color);
            painter.add(Shape::mesh(mesh));
        }
        Cm3Icon::Plugins => {
            // Solid tilted +45° diamond jigsaw piece (28x28 at 30px):
            // corners at N/S/E/W, protruding round knobs on NW (-u) & SE (+u), notches on NE (-v) & SW (+v)
            let c = r.center();
            let (sin, cos) = FRAC_PI_4.sin_cos();
            let u_vec = Vec2::new(cos, sin);
            let v_vec = Vec2::new(-sin, cos);
            let lp = |u: f32, v: f32| c + u_vec * (u * side) + v_vec * (v * side);
            let local_rect = |u0: f32, v0: f32, u1: f32, v1: f32| {
                painter.add(Shape::convex_polygon(
                    vec![lp(u0, v0), lp(u1, v0), lp(u1, v1), lp(u0, v1)],
                    color,
                    Stroke::NONE,
                ));
            };
            let h = 0.33_f32;
            let n = 0.08_f32;
            let d = 0.12_f32;
            let ov = 0.01_f32;
            // -u wing, center bridge between NE/SW notches, and +u wing
            local_rect(-h, -h, -n, h);
            local_rect(-n - ov, -h + d, n + ov, h - d);
            local_rect(n, -h, h, h);
            // Protruding round knobs on NW (-u) and SE (+u) edges
            local_rect(-h - 0.07, -0.055, -h + ov, 0.055);
            painter.circle_filled(lp(-h - 0.10, 0.0), side * 0.105, color);
            local_rect(h - ov, -0.055, h + 0.07, 0.055);
            painter.circle_filled(lp(h + 0.10, 0.0), side * 0.105, color);
        }
        Cm3Icon::TabAssets => {
            // Crisp unfeathered integer-snapped 4x4 grid-library icon (23..24px with uniform 5px tiles & 1px gaps at 30px)
            const N: usize = 4;
            let step = (side * 0.20).round().max(2.0);
            let gap = (side / 30.0).round().max(1.0);
            let tile_px = (step - gap).max(1.0);
            let grid_span = (N as f32) * step - gap;
            let ox = (r.left() + (side - grid_span) * 0.5).floor();
            let oy = (r.top() + (side - grid_span) * 0.5).floor();
            let mut mesh = Mesh::default();
            for row in 0..N {
                for col in 0..N {
                    let x0 = ox + (col as f32) * step;
                    let y0 = oy + (row as f32) * step;
                    push_crisp_quad(
                        &mut mesh,
                        Rect::from_min_size(Pos2::new(x0, y0), Vec2::splat(tile_px)),
                        color,
                    );
                }
            }
            painter.add(Shape::mesh(mesh));
        }
        Cm3Icon::Settings => {
            // Solid 6-tooth gear (24x24 at 30px): flat top & bottom teeth at 30°+60°k clearly
            // protruding beyond the gear ring. Note: epaint's `circle_stroke` strokes OUTSIDE
            // `radius`, so `radius = 0.135*side` (inner hole ~8px dia) + `width = 0.185*side`
            // produces an outer ring radius of `0.32*side` (~19px dia), leaving the `0.405*side`
            // (~24px dia) teeth boldly protruding with crisp notches between them.
            let c = Pos2::new(
                r.center().x - side * (0.5 / 30.0),
                r.center().y - side * (0.5 / 30.0),
            );
            painter.circle_stroke(c, side * 0.135, Stroke::new(side * 0.185, color));
            let r0 = side * 0.27;
            let r1 = side * 0.405;
            let hw_base = side * 0.14;
            let hw_tip = side * 0.098;
            for i in 0..6 {
                let ang = (i as f32) * (TAU / 6.0) + TAU / 12.0;
                let (sin, cos) = ang.sin_cos();
                let dir = Vec2::new(cos, sin);
                let perp = Vec2::new(-sin, cos);
                let tooth = vec![
                    c + dir * r0 - perp * hw_base,
                    c + dir * r1 - perp * hw_tip,
                    c + dir * r1 + perp * hw_tip,
                    c + dir * r0 + perp * hw_base,
                ];
                painter.add(Shape::convex_polygon(tooth, color, Stroke::NONE));
            }
        }
        Cm3Icon::Camera => {
            // Iconic 2-reel movie camera silhouette (19x17 at 25px): larger left reel + smaller right reel
            let body = Rect::from_min_max(pt(r, 0.14, 0.44), pt(r, 0.68, 0.84));
            painter.rect_filled(body, CornerRadius::ZERO, color);
            painter.circle_filled(pt(r, 0.26, 0.26), side * 0.135, color);
            painter.circle_filled(pt(r, 0.51, 0.31), side * 0.105, color);
            let lens = vec![
                pt(r, 0.66, 0.56),
                pt(r, 0.88, 0.44),
                pt(r, 0.88, 0.84),
                pt(r, 0.66, 0.72),
            ];
            painter.add(Shape::convex_polygon(lens, color, Stroke::NONE));
        }
        Cm3Icon::Transform => {
            // 2D bounding box with crisp 1px unfeathered connecting edges & 4x4 corner handles
            let x0 = (r.left() + side * 0.22).round();
            let x1 = (r.left() + side * 0.78).round();
            let y0 = (r.top() + side * 0.22).round();
            let y1 = (r.top() + side * 0.78).round();
            let mut mesh = Mesh::default();
            push_crisp_quad(
                &mut mesh,
                Rect::from_min_max(Pos2::new(x0, y0), Pos2::new(x1 + 1.0, y0 + 1.0)),
                color,
            );
            push_crisp_quad(
                &mut mesh,
                Rect::from_min_max(Pos2::new(x0, y1), Pos2::new(x1 + 1.0, y1 + 1.0)),
                color,
            );
            push_crisp_quad(
                &mut mesh,
                Rect::from_min_max(Pos2::new(x0, y0), Pos2::new(x0 + 1.0, y1 + 1.0)),
                color,
            );
            push_crisp_quad(
                &mut mesh,
                Rect::from_min_max(Pos2::new(x1, y0), Pos2::new(x1 + 1.0, y1 + 1.0)),
                color,
            );
            let hs = (side * 0.16).round().max(3.0);
            let half = (hs * 0.5).floor();
            for (cx, cy) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
                push_crisp_quad(
                    &mut mesh,
                    Rect::from_min_size(Pos2::new(cx - half, cy - half), Vec2::splat(hs)),
                    color,
                );
            }
            painter.add(Shape::mesh(mesh));
        }
        Cm3Icon::TransparencyGrid => {
            // Crisp integer-snapped 4x4 checkerboard square (16x16 at 25px, shifted +3px Y to match Zoidium)
            const N: usize = 4;
            let cell = (side * 0.16).round().max(1.0);
            let grid_size = cell * (N as f32);
            let gx = (r.center().x - grid_size * 0.5).round();
            let gy = (r.center().y - grid_size * 0.5 + side * 0.10).round();
            let mut mesh = Mesh::default();
            for row in 0..N {
                for col in 0..N {
                    if (row + col) % 2 == 0 {
                        let x0 = gx + (col as f32) * cell;
                        let y0 = gy + (row as f32) * cell;
                        push_crisp_quad(
                            &mut mesh,
                            Rect::from_min_size(Pos2::new(x0, y0), Vec2::splat(cell)),
                            color,
                        );
                    }
                }
            }
            // Crisp 1px outer frame around the 16x16 grid
            let gx1 = gx + grid_size;
            let gy1 = gy + grid_size;
            push_crisp_quad(
                &mut mesh,
                Rect::from_min_max(Pos2::new(gx, gy), Pos2::new(gx1, gy + 1.0)),
                color,
            );
            push_crisp_quad(
                &mut mesh,
                Rect::from_min_max(Pos2::new(gx, gy1 - 1.0), Pos2::new(gx1, gy1)),
                color,
            );
            push_crisp_quad(
                &mut mesh,
                Rect::from_min_max(Pos2::new(gx, gy + 1.0), Pos2::new(gx + 1.0, gy1 - 1.0)),
                color,
            );
            push_crisp_quad(
                &mut mesh,
                Rect::from_min_max(Pos2::new(gx1 - 1.0, gy + 1.0), Pos2::new(gx1, gy1 - 1.0)),
                color,
            );
            painter.add(Shape::mesh(mesh));
        }
        Cm3Icon::Start => {
            // Jump to start (|<): 17x17 inside 25px box
            let bar = Rect::from_min_max(pt(r, 0.18, 0.19), pt(r, 0.32, 0.81));
            painter.rect_filled(bar, CornerRadius::ZERO, color);
            let tri = vec![pt(r, 0.32, 0.50), pt(r, 0.82, 0.19), pt(r, 0.82, 0.81)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
        Cm3Icon::PrevFrame => {
            // Step back one frame (<|): 17x17 inside 25px box
            let tri = vec![pt(r, 0.18, 0.50), pt(r, 0.64, 0.19), pt(r, 0.64, 0.81)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
            let bar = Rect::from_min_max(pt(r, 0.70, 0.19), pt(r, 0.84, 0.81));
            painter.rect_filled(bar, CornerRadius::ZERO, color);
        }
        Cm3Icon::Play => {
            let tri = vec![pt(r, 0.14, 0.14), pt(r, 0.86, 0.50), pt(r, 0.14, 0.86)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
        Cm3Icon::Pause => {
            // Two crisp 5px bars with a 5px inner gap (15x17 at 25px)
            let left = Rect::from_min_max(pt(r, 0.20, 0.18), pt(r, 0.40, 0.82));
            let right = Rect::from_min_max(pt(r, 0.60, 0.18), pt(r, 0.80, 0.82));
            painter.rect_filled(left, CornerRadius::ZERO, color);
            painter.rect_filled(right, CornerRadius::ZERO, color);
        }
        Cm3Icon::NextFrame => {
            // Step forward one frame (|>): 17x17 inside 25px box
            let bar = Rect::from_min_max(pt(r, 0.16, 0.19), pt(r, 0.30, 0.81));
            painter.rect_filled(bar, CornerRadius::ZERO, color);
            let tri = vec![pt(r, 0.36, 0.19), pt(r, 0.82, 0.50), pt(r, 0.36, 0.81)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
        Cm3Icon::End => {
            // Jump to end (>|): 17x17 inside 25px box
            let tri = vec![pt(r, 0.18, 0.19), pt(r, 0.68, 0.50), pt(r, 0.18, 0.81)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
            let bar = Rect::from_min_max(pt(r, 0.68, 0.19), pt(r, 0.82, 0.81));
            painter.rect_filled(bar, CornerRadius::ZERO, color);
        }
        Cm3Icon::Loop => {
            // Two opposing thicker rounded repeat arrows
            let loop_stroke = Stroke::new((side * 0.105).clamp(1.8, 2.8), color);
            let top = vec![
                pt(r, 0.10, 0.72),
                pt(r, 0.10, 0.38),
                pt(r, 0.13, 0.31),
                pt(r, 0.20, 0.28),
                pt(r, 0.60, 0.28),
            ];
            painter.add(Shape::line(top, loop_stroke));
            let top_head = vec![pt(r, 0.76, 0.28), pt(r, 0.56, 0.14), pt(r, 0.56, 0.42)];
            painter.add(Shape::convex_polygon(top_head, color, Stroke::NONE));

            let bot = vec![
                pt(r, 0.90, 0.28),
                pt(r, 0.90, 0.62),
                pt(r, 0.87, 0.69),
                pt(r, 0.80, 0.72),
                pt(r, 0.40, 0.72),
            ];
            painter.add(Shape::line(bot, loop_stroke));
            let bot_head = vec![pt(r, 0.24, 0.72), pt(r, 0.44, 0.58), pt(r, 0.44, 0.86)];
            painter.add(Shape::convex_polygon(bot_head, color, Stroke::NONE));
        }
        Cm3Icon::Graph => {
            // Smooth upward parabolic bell arch + compact baseline endpoints (y=534..548 at 25px)
            const SAMPLES: usize = 16;
            let mut curve = Vec::with_capacity(SAMPLES + 1);
            for i in 0..=SAMPLES {
                let u = i as f32 / SAMPLES as f32;
                let arch = 4.0 * u * (1.0 - u);
                curve.push(pt(r, 0.14 + u * 0.72, 0.74 - arch * 0.52));
            }
            painter.add(Shape::line(curve, thin));
            painter.circle_filled(pt(r, 0.14, 0.74), side * 0.045, color);
            painter.circle_filled(pt(r, 0.86, 0.74), side * 0.045, color);
        }
        Cm3Icon::SoloCircle => {
            // Hollow circle matching Zoidium's timeline track solo button
            painter.circle_stroke(r.center(), side * 0.34, Stroke::new(1.8, color));
        }
        Cm3Icon::InterpBezier => {
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
            // Solid top crown button + angled pusher at ~1:30 + thick hollow ring
            // (epaint circle_stroke strokes OUTSIDE `radius`, so inner_rad = outer - watch_w)
            let c = pt(r, 0.50, 0.61);
            let watch_w = (side * 0.11).clamp(1.6, 2.2);
            let inner_rad = (side * 0.35 - watch_w).max(side * 0.18);
            let watch_stroke = Stroke::new(watch_w, color);
            painter.circle_stroke(c, inner_rad, watch_stroke);
            let crown = Rect::from_min_max(pt(r, 0.38, 0.06), pt(r, 0.62, 0.24));
            painter.rect_filled(crown, CornerRadius::ZERO, color);
            let pusher_ang = (-42.0_f32).to_radians();
            let dir = Vec2::new(pusher_ang.cos(), pusher_ang.sin());
            painter.line_segment(
                [
                    c + dir * (inner_rad + watch_w * 0.5),
                    c + dir * (inner_rad + watch_w + side * 0.11),
                ],
                watch_stroke,
            );
        }
        Cm3Icon::KeyframeHollow => {
            let kf_stroke = Stroke::new((side * 0.11).clamp(1.6, 2.2), color);
            let diamond = vec![
                pt(r, 0.50, 0.12),
                pt(r, 0.88, 0.50),
                pt(r, 0.50, 0.88),
                pt(r, 0.12, 0.50),
            ];
            painter.add(Shape::closed_line(diamond, kf_stroke));
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
            let tri = vec![pt(r, 0.24, 0.50), pt(r, 0.72, 0.12), pt(r, 0.72, 0.88)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
        Cm3Icon::NextKf => {
            let tri = vec![pt(r, 0.76, 0.50), pt(r, 0.28, 0.12), pt(r, 0.28, 0.88)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
        Cm3Icon::Add => {
            // Crisp unfeathered solid + glyph (15x15 with 3px arms at 20px, 2px arms at 15px)
            let thick = if side >= 18.0 { 3.0 } else { 2.0 };
            let span = if side >= 18.0 {
                (side * 0.75).round()
            } else {
                (side * 0.86).round()
            }
            .max(thick);
            let cx = r.center().x.round();
            let cy = r.center().y.round();
            let half_span = (span * 0.5).floor();
            let half_t = (thick * 0.5).floor();
            let x0 = cx - half_span;
            let y0 = cy - half_span;
            let mut mesh = Mesh::default();
            push_crisp_quad(
                &mut mesh,
                Rect::from_min_size(Pos2::new(cx - half_t, y0), Vec2::new(thick, span)),
                color,
            );
            push_crisp_quad(
                &mut mesh,
                Rect::from_min_size(Pos2::new(x0, cy - half_t), Vec2::new(span, thick)),
                color,
            );
            painter.add(Shape::mesh(mesh));
        }
        Cm3Icon::Duplicate => {
            let back = Rect::from_min_max(pt(r, 0.32, 0.14), pt(r, 0.84, 0.66));
            painter.rect_stroke(back, CornerRadius::ZERO, thin, StrokeKind::Inside);
            let front = Rect::from_min_max(pt(r, 0.16, 0.34), pt(r, 0.68, 0.86));
            painter.rect_filled(front, CornerRadius::ZERO, color);
        }
        Cm3Icon::Remove => {
            // Crisp X cross matching Zoidium's delete glyph
            let x_stroke = Stroke::new((side * 0.13).clamp(1.6, 2.2), color);
            painter.line_segment([pt(r, 0.18, 0.18), pt(r, 0.82, 0.82)], x_stroke);
            painter.line_segment([pt(r, 0.82, 0.18), pt(r, 0.18, 0.82)], x_stroke);
        }
        Cm3Icon::Reset => {
            // Tall clockwise circular reload arrow with top right-pointing triangular head
            // (matches Zoidium L1_reset 19x23 at 25px and Sequence/Edit Reset 15x17 at 20px)
            let c = pt(r, 0.50, 0.57);
            let rad = side * 0.31;
            let arc_stroke = Stroke::new((side * 0.12).clamp(1.8, 3.0), color);
            let start_deg = -22.0_f32;
            let end_deg = 268.0_f32;
            const SEGS: usize = 20;
            let mut pts = Vec::with_capacity(SEGS + 1);
            for i in 0..=SEGS {
                let u = i as f32 / SEGS as f32;
                let ang = (start_deg + (end_deg - start_deg) * u).to_radians();
                pts.push(Pos2::new(c.x + ang.cos() * rad, c.y + ang.sin() * rad));
            }
            painter.add(Shape::line(pts, arc_stroke));
            let head = vec![pt(r, 0.72, 0.24), pt(r, 0.49, 0.08), pt(r, 0.49, 0.40)];
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
            // Thick diagonal pipette matching Zoidium swatch eyedropper
            let barrel_stroke = Stroke::new((side * 0.15).clamp(1.8, 2.6), color);
            painter.line_segment([pt(r, 0.20, 0.80), pt(r, 0.66, 0.34)], barrel_stroke);
            painter.line_segment([pt(r, 0.48, 0.22), pt(r, 0.76, 0.50)], barrel_stroke);
            painter.circle_filled(pt(r, 0.74, 0.26), side * 0.15, color);
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
            painter.add(Shape::line(
                vec![pt(r, 0.22, 0.36), pt(r, 0.50, 0.66), pt(r, 0.78, 0.36)],
                Stroke::new(1.6, color),
            ));
        }
        Cm3Icon::ChevronRight => {
            painter.add(Shape::line(
                vec![pt(r, 0.36, 0.22), pt(r, 0.66, 0.50), pt(r, 0.36, 0.78)],
                Stroke::new(1.6, color),
            ));
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

/// Paints a small pill badge (used for `"In Use"`, `"Beta"`, error counts),
/// scaling its monospace font via the active `Cm3Theme`'s `theme.font(FONT_BADGE)`.
pub fn paint_badge(painter: &Painter, rect: Rect, text: &str, bg: Color32, fg: Color32) {
    let font = painter
        .ctx()
        .data(|d| d.get_temp::<FontId>(egui::Id::NULL.with("cm3_badge_font")))
        .unwrap_or_else(|| {
            let scale = painter
                .ctx()
                .global_style()
                .text_styles
                .get(&TextStyle::Body)
                .map_or(1.0, |f| (f.size / FONT_BODY).clamp(0.75, 1.5));
            font_mono((FONT_BADGE * scale).round())
        });
    painter.rect_filled(rect, CornerRadius::same(8), bg);
    painter.text(rect.center(), Align2::CENTER_CENTER, text, font, fg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_cm3_fonts_runs_cleanly_and_layouts_cjk_text() {
        let ctx = egui::Context::default();
        install_cm3_fonts(&ctx);

        let mut mono_size = Vec2::ZERO;
        let mut prop_size = Vec2::ZERO;
        let mut cjk_glyph_distinct = false;
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
            let mono_galley = ui.painter().layout_no_wrap(
                "日本語テスト Zoid 漢字".to_owned(),
                FontId::new(FONT_BODY, FontFamily::Monospace),
                TEXT_PRIMARY,
            );
            let prop_galley = ui.painter().layout_no_wrap(
                "日本語テスト Zoid 漢字".to_owned(),
                FontId::new(FONT_BODY, FontFamily::Proportional),
                TEXT_PRIMARY,
            );
            mono_size = mono_galley.size();
            prop_size = prop_galley.size();

            // Verify CJK characters resolve distinct glyph geometry when a system CJK font exists
            // (instead of falling back to identical replacement/tofu boxes).
            let g_a = ui.painter().layout_no_wrap(
                "日".to_owned(),
                FontId::new(FONT_BODY, FontFamily::Monospace),
                TEXT_PRIMARY,
            );
            let g_b = ui.painter().layout_no_wrap(
                "漢".to_owned(),
                FontId::new(FONT_BODY, FontFamily::Monospace),
                TEXT_PRIMARY,
            );
            if let (Some(r0), Some(r1)) = (g_a.rows.first(), g_b.rows.first())
                && let (Some(gl0), Some(gl1)) = (r0.glyphs.first(), r1.glyphs.first())
            {
                cjk_glyph_distinct = gl0.uv_rect != gl1.uv_rect;
            }
        });
        out.textures_delta.clear();

        assert!(mono_size.x > 0.0 && mono_size.y > 0.0);
        assert!(prop_size.x > 0.0 && prop_size.y > 0.0);
        if load_system_cjk_fallback_font().is_some() {
            assert!(
                cjk_glyph_distinct,
                "with OS CJK fallback loaded, distinct kanji ('日' vs '漢') must map to distinct glyph UVs, not identical tofu boxes"
            );
        }

        // Switching editor font must also preserve clean CJK layout without panicking.
        apply_editor_font(&ctx, EditorFontChoice::GeistMono);
        let mut out2 = ctx.run_ui(egui::RawInput::default(), |ui| {
            let g = ui.painter().layout_no_wrap(
                "日本語テスト".to_owned(),
                FontId::new(FONT_BODY, FontFamily::Monospace),
                TEXT_PRIMARY,
            );
            assert!(g.size().x > 0.0);
        });
        out2.textures_delta.clear();
    }

    #[test]
    fn inactive_interp_bezier_icon_paints_monochrome_without_red_dots() {
        let ctx = egui::Context::default();
        let rect = Rect::from_min_size(Pos2::new(10.0, 10.0), Vec2::splat(25.0));
        let mut out = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::splat(100.0))),
                ..Default::default()
            },
            |ui| {
                paint_cm3_icon(
                    ui.painter(),
                    rect,
                    Cm3Icon::InterpBezier,
                    TOOLBAR_ICON_INACTIVE,
                );
            },
        );
        out.textures_delta.clear();

        let mut inspected = 0usize;
        for clipped in &out.shapes {
            match &clipped.shape {
                Shape::Circle(c) => {
                    inspected += 1;
                    assert_eq!(
                        c.fill, TOOLBAR_ICON_INACTIVE,
                        "inactive Bezier icon dots must be monochrome neutral gray, not red"
                    );
                }
                Shape::Path(p) => {
                    inspected += 1;
                    assert_eq!(
                        p.stroke.color,
                        egui::epaint::ColorMode::Solid(TOOLBAR_ICON_INACTIVE)
                    );
                }
                Shape::LineSegment { stroke, .. } => {
                    inspected += 1;
                    assert_eq!(stroke.color, TOOLBAR_ICON_INACTIVE);
                }
                _ => {}
            }
        }
        assert!(
            inspected >= 3,
            "expected curve, tangent segments, and monochrome handle dots to be painted"
        );
    }
}
