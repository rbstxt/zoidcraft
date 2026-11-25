//! Color primitives, color-space conversions (sRGB, linear RGB, HSL, HSV, OKLab),
//! WCAG 2.x contrast calculations, CM3 theme derivation, and all 43 CM3 premultiplied blend modes.

#![forbid(unsafe_code)]

use std::fmt;

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Safe minimum alpha divisor used when un-premultiplying alpha (`rgb / a.max(1e-6)`).
pub const UNPREMUL_EPS: f32 = 1e-6;

/// Dark clip label color chosen by WCAG auto-contrast (`#2c323f`).
pub const CLIP_LABEL_DARK: Rgb = Rgb::from_u8(0x2c, 0x32, 0x3f);

/// Light clip label color chosen by WCAG auto-contrast (`#f4f4f4`).
pub const CLIP_LABEL_LIGHT: Rgb = Rgb::from_u8(0xf4, 0xf4, 0xf4);

/// Error returned when parsing hex colors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColorError {
    /// Input hex string did not match `#RRGGBB` or `#RRGGBBAA`.
    InvalidHex(String),
}

impl fmt::Display for ColorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHex(s) => write!(f, "invalid hex color '{s}'"),
        }
    }
}

impl std::error::Error for ColorError {}

/// Normalized 3-channel RGB color (`[r, g, b]` in `[0.0, 1.0]`, sRGB by default).
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
#[serde(transparent)]
#[repr(transparent)]
pub struct Rgb(pub [f32; 3]);

impl Rgb {
    /// Opaque black `(0, 0, 0)`.
    pub const BLACK: Self = Self([0.0, 0.0, 0.0]);

    /// Opaque white `(1, 1, 1)`.
    pub const WHITE: Self = Self([1.0, 1.0, 1.0]);

    /// Creates an `Rgb` from normalized floats.
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self([r, g, b])
    }

    /// Creates an `Rgb` from 8-bit sRGB channel values.
    pub const fn from_u8(r: u8, g: u8, b: u8) -> Self {
        Self([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0])
    }

    /// Quantizes to 8-bit `[r, g, b]` using nearest-integer rounding.
    pub fn to_u8(self) -> [u8; 3] {
        [
            (self.0[0].clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.0[1].clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.0[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        ]
    }

    /// Red channel.
    pub const fn r(self) -> f32 {
        self.0[0]
    }

    /// Green channel.
    pub const fn g(self) -> f32 {
        self.0[1]
    }

    /// Blue channel.
    pub const fn b(self) -> f32 {
        self.0[2]
    }

    /// Promotes to straight [`Rgba`] with the given alpha.
    pub const fn with_alpha(self, alpha: f32) -> Rgba {
        Rgba([self.0[0], self.0[1], self.0[2], alpha])
    }

    /// Linearly mixes `self` toward `other` in sRGB (`(1 - t) * self + t * other`).
    pub fn mix(self, other: Self, t: f32) -> Self {
        let u = t.clamp(0.0, 1.0);
        let inv = 1.0 - u;
        Self([
            self.0[0] * inv + other.0[0] * u,
            self.0[1] * inv + other.0[1] * u,
            self.0[2] * inv + other.0[2] * u,
        ])
    }

    /// Parses `#RRGGBB` (or `RRGGBB`) hex string into an `Rgb`.
    pub fn from_hex(hex: &str) -> Result<Self, ColorError> {
        let s = hex.trim().strip_prefix('#').unwrap_or(hex.trim());
        if s.len() != 6 || !s.is_ascii() {
            return Err(ColorError::InvalidHex(hex.to_string()));
        }
        let r = parse_hex_byte(&s[0..2], hex)?;
        let g = parse_hex_byte(&s[2..4], hex)?;
        let b = parse_hex_byte(&s[4..6], hex)?;
        Ok(Self::from_u8(r, g, b))
    }

    /// Formats as lowercase `#rrggbb`.
    pub fn to_hex(self) -> String {
        let [r, g, b] = self.to_u8();
        format!("#{r:02x}{g:02x}{b:02x}")
    }

    /// Converts sRGB-encoded primaries to linear light RGB.
    pub fn to_linear(self) -> Self {
        Self([
            srgb_to_linear_channel(self.0[0]),
            srgb_to_linear_channel(self.0[1]),
            srgb_to_linear_channel(self.0[2]),
        ])
    }

    /// Converts linear light RGB to sRGB-encoded primaries.
    pub fn from_linear(linear: Self) -> Self {
        Self([
            linear_to_srgb_channel(linear.0[0]),
            linear_to_srgb_channel(linear.0[1]),
            linear_to_srgb_channel(linear.0[2]),
        ])
    }

    /// Converts sRGB to [`Hsl`] (`h in [0, 1)`, `s in [0, 1]`, `l in [0, 1]`).
    pub fn to_hsl(self) -> Hsl {
        let (h, s, l) = rgb_to_hsl_f64(
            f64::from(self.0[0]),
            f64::from(self.0[1]),
            f64::from(self.0[2]),
        );
        Hsl {
            h: h as f32,
            s: s as f32,
            l: l as f32,
        }
    }

    /// Converts [`Hsl`] to sRGB.
    pub fn from_hsl(hsl: Hsl) -> Self {
        let (r, g, b) = hsl_to_rgb_f64(f64::from(hsl.h), f64::from(hsl.s), f64::from(hsl.l));
        Self([r as f32, g as f32, b as f32])
    }

    /// Converts sRGB to [`Hsv`] (`h in [0, 1)`, `s in [0, 1]`, `v in [0, 1]`).
    pub fn to_hsv(self) -> Hsv {
        let r = if self.0[0].is_finite() {
            self.0[0].clamp(0.0, 1.0)
        } else {
            0.0
        };
        let g = if self.0[1].is_finite() {
            self.0[1].clamp(0.0, 1.0)
        } else {
            0.0
        };
        let b = if self.0[2].is_finite() {
            self.0[2].clamp(0.0, 1.0)
        } else {
            0.0
        };
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let d = max - min;
        let v = max;
        let s = if max == 0.0 { 0.0 } else { d / max };
        let h = if d == 0.0 {
            0.0
        } else {
            let raw = if max == r {
                (g - b) / d + if g < b { 6.0 } else { 0.0 }
            } else if max == g {
                (b - r) / d + 2.0
            } else {
                (r - g) / d + 4.0
            };
            wrap_unit_f32(raw / 6.0)
        };
        Hsv { h, s, v }
    }

    /// Converts [`Hsv`] to sRGB.
    pub fn from_hsv(hsv: Hsv) -> Self {
        let h6 = wrap_unit_f32(hsv.h) * 6.0;
        let sector = (h6.floor() as i32).rem_euclid(6);
        let f = h6 - h6.floor();
        let v = hsv.v;
        let s = hsv.s;
        let p = v * (1.0 - s);
        let q = v * (1.0 - f * s);
        let t = v * (1.0 - (1.0 - f) * s);
        let (r, g, b) = match sector {
            0 => (v, t, p),
            1 => (q, v, p),
            2 => (p, v, t),
            3 => (p, q, v),
            4 => (t, p, v),
            _ => (v, p, q),
        };
        Self([r, g, b])
    }

    /// Converts sRGB to perceptual [`Oklab`] via linear sRGB.
    pub fn to_oklab(self) -> Oklab {
        let lin = self.to_linear();
        let r = f64::from(lin.0[0]);
        let g = f64::from(lin.0[1]);
        let b = f64::from(lin.0[2]);

        let l = (0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b).cbrt();
        let m = (0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b).cbrt();
        let s = (0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b).cbrt();

        Oklab {
            l: (0.210_454_255_3 * l + 0.793_617_785_0 * m - 0.004_072_046_8 * s) as f32,
            a: (1.977_998_495_1 * l - 2.428_592_205_0 * m + 0.450_593_709_9 * s) as f32,
            b: (0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766_0 * s) as f32,
        }
    }

    /// Converts perceptual [`Oklab`] to sRGB (clamped to the sRGB gamut).
    pub fn from_oklab(ok: Oklab) -> Self {
        let ol = f64::from(ok.l);
        let oa = f64::from(ok.a);
        let ob = f64::from(ok.b);

        let l_ = ol + 0.396_337_777_4 * oa + 0.215_803_757_3 * ob;
        let m_ = ol - 0.105_561_345_8 * oa - 0.063_854_172_8 * ob;
        let s_ = ol - 0.089_484_177_5 * oa - 1.291_485_548_0 * ob;

        let l = l_ * l_ * l_;
        let m = m_ * m_ * m_;
        let s = s_ * s_ * s_;

        let r = (4.076_741_662_1 * l - 3.307_711_591_3 * m + 0.230_969_929_2 * s).clamp(0.0, 1.0);
        let g = (-1.268_438_004_6 * l + 2.609_757_401_1 * m - 0.341_319_396_5 * s).clamp(0.0, 1.0);
        let b = (-0.004_196_086_3 * l - 0.703_418_614_7 * m + 1.707_614_701_0 * s).clamp(0.0, 1.0);

        Self::from_linear(Self([r as f32, g as f32, b as f32]))
    }
}

/// Straight-alpha RGBA color (`[r, g, b, a]` in `[0.0, 1.0]`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
#[serde(transparent)]
#[repr(transparent)]
pub struct Rgba(pub [f32; 4]);

impl Rgba {
    /// Fully transparent black `(0, 0, 0, 0)`.
    pub const TRANSPARENT: Self = Self([0.0, 0.0, 0.0, 0.0]);

    /// Opaque black `(0, 0, 0, 1)`.
    pub const BLACK: Self = Self([0.0, 0.0, 0.0, 1.0]);

    /// Opaque white `(1, 1, 1, 1)`.
    pub const WHITE: Self = Self([1.0, 1.0, 1.0, 1.0]);

    /// Creates a straight `Rgba` from normalized floats.
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self([r, g, b, a])
    }

    /// Creates a straight `Rgba` from 8-bit channel values.
    pub const fn from_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self([
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        ])
    }

    /// Quantizes to 8-bit `[r, g, b, a]` using nearest-integer rounding.
    pub fn to_u8(self) -> [u8; 4] {
        [
            (self.0[0].clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.0[1].clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.0[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.0[3].clamp(0.0, 1.0) * 255.0).round() as u8,
        ]
    }

    /// Extracts the RGB channels.
    pub const fn rgb(self) -> Rgb {
        Rgb([self.0[0], self.0[1], self.0[2]])
    }

    /// Alpha channel.
    pub const fn a(self) -> f32 {
        self.0[3]
    }

    /// Converts straight alpha to premultiplied [`PremulRgba`]: `(r*a, g*a, b*a, a)`.
    pub fn to_premul(self) -> PremulRgba {
        let a = self.0[3];
        PremulRgba([self.0[0] * a, self.0[1] * a, self.0[2] * a, a])
    }

    /// Converts sRGB color channels to linear light while preserving alpha.
    pub fn to_linear(self) -> Self {
        let lin = self.rgb().to_linear();
        Self([lin.0[0], lin.0[1], lin.0[2], self.0[3]])
    }

    /// Converts linear light color channels to sRGB while preserving alpha.
    pub fn from_linear(linear: Self) -> Self {
        let srgb = Rgb::from_linear(linear.rgb());
        Self([srgb.0[0], srgb.0[1], srgb.0[2], linear.0[3]])
    }

    /// Parses `#RRGGBB` (implicit alpha `1.0`) or `#RRGGBBAA` hex strings.
    pub fn from_hex(hex: &str) -> Result<Self, ColorError> {
        let s = hex.trim().strip_prefix('#').unwrap_or(hex.trim());
        if !s.is_ascii() {
            return Err(ColorError::InvalidHex(hex.to_string()));
        }
        match s.len() {
            6 => {
                let r = parse_hex_byte(&s[0..2], hex)?;
                let g = parse_hex_byte(&s[2..4], hex)?;
                let b = parse_hex_byte(&s[4..6], hex)?;
                Ok(Self::from_u8(r, g, b, 255))
            }
            8 => {
                let r = parse_hex_byte(&s[0..2], hex)?;
                let g = parse_hex_byte(&s[2..4], hex)?;
                let b = parse_hex_byte(&s[4..6], hex)?;
                let a = parse_hex_byte(&s[6..8], hex)?;
                Ok(Self::from_u8(r, g, b, a))
            }
            _ => Err(ColorError::InvalidHex(hex.to_string())),
        }
    }

    /// Formats as lowercase `#rrggbb` (dropping alpha).
    pub fn to_hex_rgb(self) -> String {
        self.rgb().to_hex()
    }

    /// Formats as lowercase `#rrggbbaa`.
    pub fn to_hex_rgba(self) -> String {
        let [r, g, b, a] = self.to_u8();
        format!("#{r:02x}{g:02x}{b:02x}{a:02x}")
    }
}

/// Premultiplied-alpha RGBA color (`[r*a, g*a, b*a, a]`) used throughout ZoidCraft's compositor.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
#[serde(transparent)]
#[repr(transparent)]
pub struct PremulRgba(pub [f32; 4]);

impl PremulRgba {
    /// Transparent black `(0, 0, 0, 0)`.
    pub const TRANSPARENT: Self = Self([0.0, 0.0, 0.0, 0.0]);

    /// Opaque black `(0, 0, 0, 1)`.
    pub const BLACK: Self = Self([0.0, 0.0, 0.0, 1.0]);

    /// Opaque white `(1, 1, 1, 1)`.
    pub const WHITE: Self = Self([1.0, 1.0, 1.0, 1.0]);

    /// Creates a `PremulRgba` directly from already-premultiplied components.
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self([r, g, b, a])
    }

    /// Creates a `PremulRgba` by premultiplying straight [`Rgba`].
    pub fn from_straight(straight: Rgba) -> Self {
        straight.to_premul()
    }

    /// Safely un-premultiplies to straight [`Rgba`] using `rgb / a.max(1e-6)`.
    ///
    /// Returns [`Rgba::TRANSPARENT`] when `alpha <= 0.0`.
    pub fn to_straight(self) -> Rgba {
        let a = self.0[3];
        if a <= 0.0 {
            return Rgba::TRANSPARENT;
        }
        let denom = a.max(UNPREMUL_EPS);
        Rgba([self.0[0] / denom, self.0[1] / denom, self.0[2] / denom, a])
    }

    /// Scales all four premultiplied channels by layer `opacity` (Spec 03 §2.5).
    pub fn scale_opacity(self, opacity: f32) -> Self {
        let o = opacity.clamp(0.0, 1.0);
        Self([self.0[0] * o, self.0[1] * o, self.0[2] * o, self.0[3] * o])
    }

    /// Alpha channel.
    pub const fn a(self) -> f32 {
        self.0[3]
    }
}

/// HSL color representation with normalized hue `h in [0, 1)`, saturation `s in [0, 1]`,
/// and lightness `l in [0, 1]`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Hsl {
    pub h: f32,
    pub s: f32,
    pub l: f32,
}

/// HSV color representation with normalized hue `h in [0, 1)`, saturation `s in [0, 1]`,
/// and value `v in [0, 1]`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Hsv {
    pub h: f32,
    pub s: f32,
    pub v: f32,
}

/// Perceptual OKLab color representation (`l in [0, 1]`, opponent axes `a`, `b`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Oklab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

/// Converts a single sRGB channel in `[0, 1]` to linear light using IEC 61966-2-1 / WCAG 2.x.
pub fn srgb_to_linear_channel(c: f32) -> f32 {
    if c <= 0.0 {
        0.0
    } else if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Converts a single linear light channel in `[0, 1]` to sRGB using IEC 61966-2-1.
pub fn linear_to_srgb_channel(c: f32) -> f32 {
    if c <= 0.0 {
        0.0
    } else if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Computes WCAG 2.x relative luminance `Y = 0.2126*R_lin + 0.7152*G_lin + 0.0722*B_lin`.
pub fn relative_luminance(rgb: Rgb) -> f32 {
    let lin = rgb.to_linear();
    0.2126 * lin.0[0] + 0.7152 * lin.0[1] + 0.0722 * lin.0[2]
}

/// Computes WCAG 2.x contrast ratio in `[1.0, 21.0]` between two sRGB colors.
pub fn contrast_ratio(a: Rgb, b: Rgb) -> f32 {
    let la = relative_luminance(a);
    let lb = relative_luminance(b);
    let (hi, lo) = if la >= lb { (la, lb) } else { (lb, la) };
    (hi + 0.05) / (lo + 0.05)
}

/// Chooses the higher-contrast CM3 clip label color (`#2c323f` vs `#f4f4f4`) per Spec 01 §2.6 C.
pub fn choose_clip_label_color(background: Rgb) -> Rgb {
    let y = relative_luminance(background);
    let y_dark = relative_luminance(CLIP_LABEL_DARK);
    let y_light = relative_luminance(CLIP_LABEL_LIGHT);
    if (y + 0.05) / (y_dark + 0.05) >= (y_light + 0.05) / (y + 0.05) {
        CLIP_LABEL_DARK
    } else {
        CLIP_LABEL_LIGHT
    }
}

/// Computes the selected state of a custom-colored timeline clip by shifting its HSL lightness
/// 35% toward white: `L_selected = L + (1 - L) * 0.35` (Spec 01 §2.6 C).
pub fn selected_custom_clip_color(base: Rgb) -> Rgb {
    let (h, s, l) = rgb_to_hsl_f64(
        f64::from(base.0[0]),
        f64::from(base.0[1]),
        f64::from(base.0[2]),
    );
    let l_sel = (l + (1.0 - l) * 0.35).clamp(0.0, 1.0);
    let (r, g, b) = hsl_to_rgb_f64(h, s, l_sel);
    rgb_from_f64_rounded(r, g, b)
}

/// Built-in CM3 / Zoidium theme profile presets (Spec 01 §2.6 B).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThemeProfile {
    /// Panzoid Classic (`0 deg`, `1.0`).
    #[default]
    Panzoid,
    /// Signal Blue (`-18 deg`, `1.0`).
    Signal,
    /// Ember Console (`+154 deg`, `1.0`).
    Ember,
    /// Moss Terminal (`+78 deg`, `1.0`).
    Moss,
    /// Rose Signal (`-72 deg`, `1.0`).
    Rose,
    /// Gray (`0 deg`, `0.0`).
    Gray,
}

impl ThemeProfile {
    /// Returns `(hue_shift_deg, saturation_scale)` for this built-in profile.
    pub const fn parameters(self) -> (f32, f32) {
        match self {
            Self::Panzoid => (0.0, 1.0),
            Self::Signal => (-18.0, 1.0),
            Self::Ember => (154.0, 1.0),
            Self::Moss => (78.0, 1.0),
            Self::Rose => (-72.0, 1.0),
            Self::Gray => (0.0, 0.0),
        }
    }

    /// Derives the 5-color [`ThemePalette`] for this built-in profile.
    pub fn palette(self) -> ThemePalette {
        let (hue_deg, sat_scale) = self.parameters();
        ThemePalette::derive_profile(hue_deg, sat_scale)
    }
}

/// 5-color CM3 accent theme palette (Spec 01 §2.5 & §2.6).
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThemePalette {
    /// Primary focus & accent color (`#384668` in Classic).
    pub focus: Rgb,
    /// Value / link / input text accent (`#7e8fb9` in Classic).
    pub link: Rgb,
    /// Unselected timeline clip fill (`#526183` in Classic).
    pub clip: Rgb,
    /// Selected timeline clip fill (`#7f94c7` in Classic).
    pub clip_selected: Rgb,
    /// Box-selection / active clip highlight (`#273465` in Classic).
    pub clip_selection: Rgb,
}

impl ThemePalette {
    /// Canonical Panzoid Classic 5-color accent palette.
    pub const CLASSIC: Self = Self {
        focus: Rgb::from_u8(56, 70, 104),           // #384668
        link: Rgb::from_u8(126, 143, 185),          // #7e8fb9
        clip: Rgb::from_u8(82, 97, 131),            // #526183
        clip_selected: Rgb::from_u8(127, 148, 199), // #7f94c7
        clip_selection: Rgb::from_u8(39, 52, 101),  // #273465
    };

    /// Derives a complete 5-color palette from a main hex color `#RRGGBB` using the exact
    /// CM3 HSL relative-offset formula (Spec 01 §2.6 A).
    ///
    /// Passing `"#384668"` reproduces all 5 Classic palette colors byte-for-byte.
    pub fn derive_from_main(main_hex: &str) -> Result<Self, ColorError> {
        let main = Rgb::from_hex(main_hex)?;
        Ok(Self::derive_from_rgb(main))
    }

    /// Derives a complete 5-color palette from a main [`Rgb`] color (Spec 01 §2.6 A).
    pub fn derive_from_rgb(main: Rgb) -> Self {
        let (hm, sm, lm) = rgb_to_hsl_f64(
            f64::from(main.0[0]),
            f64::from(main.0[1]),
            f64::from(main.0[2]),
        );
        let (hf, sf, lf) = rgb_to_hsl_f64(
            f64::from(Self::CLASSIC.focus.0[0]),
            f64::from(Self::CLASSIC.focus.0[1]),
            f64::from(Self::CLASSIC.focus.0[2]),
        );

        let derive_entry = |reference: Rgb| -> Rgb {
            let (hp, sp, lp) = rgb_to_hsl_f64(
                f64::from(reference.0[0]),
                f64::from(reference.0[1]),
                f64::from(reference.0[2]),
            );
            let h = wrap_unit_f64(hm + hp - hf);
            let s = if sf > 0.0 {
                (sm * (sp / sf)).clamp(0.0, 1.0)
            } else {
                sm.clamp(0.0, 1.0)
            };
            let l = (lm + lp - lf).clamp(0.0, 1.0);
            let (r, g, b) = hsl_to_rgb_f64(h, s, l);
            rgb_from_f64_rounded(r, g, b)
        };

        Self {
            focus: main,
            link: derive_entry(Self::CLASSIC.link),
            clip: derive_entry(Self::CLASSIC.clip),
            clip_selected: derive_entry(Self::CLASSIC.clip_selected),
            clip_selection: derive_entry(Self::CLASSIC.clip_selection),
        }
    }

    /// Derives a 5-color palette by shifting every Classic entry independently in HSL
    /// by `hue_deg` degrees and multiplying saturation by `sat_scale` (Spec 01 §2.6 B).
    pub fn derive_profile(hue_deg: f32, sat_scale: f32) -> Self {
        let hue_shift = f64::from(hue_deg) / 360.0;
        let scale = f64::from(sat_scale);
        let transform = |reference: Rgb| -> Rgb {
            let (hp, sp, lp) = rgb_to_hsl_f64(
                f64::from(reference.0[0]),
                f64::from(reference.0[1]),
                f64::from(reference.0[2]),
            );
            let h = wrap_unit_f64(hp + hue_shift);
            let s = (sp * scale).clamp(0.0, 1.0);
            let (r, g, b) = hsl_to_rgb_f64(h, s, lp);
            rgb_from_f64_rounded(r, g, b)
        };

        Self {
            focus: transform(Self::CLASSIC.focus),
            link: transform(Self::CLASSIC.link),
            clip: transform(Self::CLASSIC.clip),
            clip_selected: transform(Self::CLASSIC.clip_selected),
            clip_selection: transform(Self::CLASSIC.clip_selection),
        }
    }

    /// Derived `--zui-accent-strong`: 70% `focus` + 30% `#ffffff`.
    pub fn accent_strong(&self) -> Rgb {
        self.focus.mix(Rgb::WHITE, 0.30)
    }

    /// Derived `--zui-value-strong`: 60% `link` + 40% `#ffffff`.
    pub fn value_strong(&self) -> Rgb {
        self.link.mix(Rgb::WHITE, 0.40)
    }

    /// Derived `--zui-bg-title-focused`: 22% `focus` + 78% `#222222`.
    pub fn bg_title_focused(&self) -> Rgb {
        Rgb::from_u8(0x22, 0x22, 0x22).mix(self.focus, 0.22)
    }

    /// Derived `--zui-menu-active`: 55% `focus` + 45% `#2a2a2a`.
    pub fn menu_active(&self) -> Rgb {
        Rgb::from_u8(0x2a, 0x2a, 0x2a).mix(self.focus, 0.55)
    }

    /// Returns `true` when the focus color's HSL lightness exceeds `60%` (Spec 01 §2.6 C).
    pub fn is_light_chrome(&self) -> bool {
        self.focus.to_hsl().l > 0.60
    }
}

impl Default for ThemePalette {
    fn default() -> Self {
        Self::CLASSIC
    }
}

/// Complete set of all 43 Panzoid ClipMaker 3 layer blend modes (Spec 03 §2.6).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendMode {
    #[serde(rename = "BLEND_ZERO")]
    Zero,
    #[serde(rename = "BLEND_SRC")]
    Src,
    #[serde(rename = "BLEND_DST")]
    Dst,
    #[default]
    #[serde(rename = "BLEND_SRC_OVER")]
    SrcOver,
    #[serde(rename = "BLEND_DST_OVER")]
    DstOver,
    #[serde(rename = "BLEND_SRC_IN")]
    SrcIn,
    #[serde(rename = "BLEND_DST_IN")]
    DstIn,
    #[serde(rename = "BLEND_SRC_OUT")]
    SrcOut,
    #[serde(rename = "BLEND_DST_OUT")]
    DstOut,
    #[serde(rename = "BLEND_SRC_ATOP")]
    SrcAtop,
    #[serde(rename = "BLEND_DST_ATOP")]
    DstAtop,
    #[serde(rename = "BLEND_XOR")]
    Xor,
    #[serde(rename = "BLEND_MULTIPLY")]
    Multiply,
    #[serde(rename = "BLEND_SCREEN")]
    Screen,
    #[serde(rename = "BLEND_OVERLAY")]
    Overlay,
    #[serde(rename = "BLEND_DARKEN")]
    Darken,
    #[serde(rename = "BLEND_LIGHTEN")]
    Lighten,
    #[serde(rename = "BLEND_COLORDODGE")]
    ColorDodge,
    #[serde(rename = "BLEND_COLORBURN")]
    ColorBurn,
    #[serde(rename = "BLEND_HARDLIGHT")]
    HardLight,
    #[serde(rename = "BLEND_SOFTLIGHT")]
    SoftLight,
    #[serde(rename = "BLEND_DIFFERENCE")]
    Difference,
    #[serde(rename = "BLEND_EXCLUSION")]
    Exclusion,
    #[serde(rename = "BLEND_INVERT")]
    Invert,
    #[serde(rename = "BLEND_INVERT_RGB")]
    InvertRgb,
    #[serde(rename = "BLEND_LINEARDODGE")]
    LinearDodge,
    #[serde(rename = "BLEND_LINEARBURN")]
    LinearBurn,
    #[serde(rename = "BLEND_VIVIDLIGHT")]
    VividLight,
    #[serde(rename = "BLEND_LINEARLIGHT")]
    LinearLight,
    #[serde(rename = "BLEND_PINLIGHT")]
    PinLight,
    #[serde(rename = "BLEND_HARDMIX")]
    HardMix,
    #[serde(rename = "BLEND_HUE")]
    Hue,
    #[serde(rename = "BLEND_SATURATION")]
    Saturation,
    #[serde(rename = "BLEND_COLOR")]
    Color,
    #[serde(rename = "BLEND_LUMINOSITY")]
    Luminosity,
    #[serde(rename = "BLEND_PLUS")]
    Plus,
    #[serde(rename = "BLEND_PLUS_DARKER")]
    PlusDarker,
    #[serde(rename = "BLEND_MINUS")]
    Minus,
    #[serde(rename = "BLEND_CONTRAST")]
    Contrast,
    #[serde(rename = "BLEND_INVERT_OVG")]
    InvertOvg,
    #[serde(rename = "BLEND_RED")]
    Red,
    #[serde(rename = "BLEND_GREEN")]
    Green,
    #[serde(rename = "BLEND_BLUE")]
    Blue,
}

impl BlendMode {
    /// All 43 CM3 blend modes in canonical Spec 03 §2.6 order.
    pub const ALL: [Self; 43] = [
        Self::Zero,
        Self::Src,
        Self::Dst,
        Self::SrcOver,
        Self::DstOver,
        Self::SrcIn,
        Self::DstIn,
        Self::SrcOut,
        Self::DstOut,
        Self::SrcAtop,
        Self::DstAtop,
        Self::Xor,
        Self::Multiply,
        Self::Screen,
        Self::Overlay,
        Self::Darken,
        Self::Lighten,
        Self::ColorDodge,
        Self::ColorBurn,
        Self::HardLight,
        Self::SoftLight,
        Self::Difference,
        Self::Exclusion,
        Self::Invert,
        Self::InvertRgb,
        Self::LinearDodge,
        Self::LinearBurn,
        Self::VividLight,
        Self::LinearLight,
        Self::PinLight,
        Self::HardMix,
        Self::Hue,
        Self::Saturation,
        Self::Color,
        Self::Luminosity,
        Self::Plus,
        Self::PlusDarker,
        Self::Minus,
        Self::Contrast,
        Self::InvertOvg,
        Self::Red,
        Self::Green,
        Self::Blue,
    ];

    /// Returns the exact CM3 `.pz` / serde string token (`"BLEND_*"`).
    pub const fn token(self) -> &'static str {
        match self {
            Self::Zero => "BLEND_ZERO",
            Self::Src => "BLEND_SRC",
            Self::Dst => "BLEND_DST",
            Self::SrcOver => "BLEND_SRC_OVER",
            Self::DstOver => "BLEND_DST_OVER",
            Self::SrcIn => "BLEND_SRC_IN",
            Self::DstIn => "BLEND_DST_IN",
            Self::SrcOut => "BLEND_SRC_OUT",
            Self::DstOut => "BLEND_DST_OUT",
            Self::SrcAtop => "BLEND_SRC_ATOP",
            Self::DstAtop => "BLEND_DST_ATOP",
            Self::Xor => "BLEND_XOR",
            Self::Multiply => "BLEND_MULTIPLY",
            Self::Screen => "BLEND_SCREEN",
            Self::Overlay => "BLEND_OVERLAY",
            Self::Darken => "BLEND_DARKEN",
            Self::Lighten => "BLEND_LIGHTEN",
            Self::ColorDodge => "BLEND_COLORDODGE",
            Self::ColorBurn => "BLEND_COLORBURN",
            Self::HardLight => "BLEND_HARDLIGHT",
            Self::SoftLight => "BLEND_SOFTLIGHT",
            Self::Difference => "BLEND_DIFFERENCE",
            Self::Exclusion => "BLEND_EXCLUSION",
            Self::Invert => "BLEND_INVERT",
            Self::InvertRgb => "BLEND_INVERT_RGB",
            Self::LinearDodge => "BLEND_LINEARDODGE",
            Self::LinearBurn => "BLEND_LINEARBURN",
            Self::VividLight => "BLEND_VIVIDLIGHT",
            Self::LinearLight => "BLEND_LINEARLIGHT",
            Self::PinLight => "BLEND_PINLIGHT",
            Self::HardMix => "BLEND_HARDMIX",
            Self::Hue => "BLEND_HUE",
            Self::Saturation => "BLEND_SATURATION",
            Self::Color => "BLEND_COLOR",
            Self::Luminosity => "BLEND_LUMINOSITY",
            Self::Plus => "BLEND_PLUS",
            Self::PlusDarker => "BLEND_PLUS_DARKER",
            Self::Minus => "BLEND_MINUS",
            Self::Contrast => "BLEND_CONTRAST",
            Self::InvertOvg => "BLEND_INVERT_OVG",
            Self::Red => "BLEND_RED",
            Self::Green => "BLEND_GREEN",
            Self::Blue => "BLEND_BLUE",
        }
    }

    /// Parses an exact CM3 `"BLEND_*"` token.
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "BLEND_ZERO" => Some(Self::Zero),
            "BLEND_SRC" => Some(Self::Src),
            "BLEND_DST" => Some(Self::Dst),
            "BLEND_SRC_OVER" => Some(Self::SrcOver),
            "BLEND_DST_OVER" => Some(Self::DstOver),
            "BLEND_SRC_IN" => Some(Self::SrcIn),
            "BLEND_DST_IN" => Some(Self::DstIn),
            "BLEND_SRC_OUT" => Some(Self::SrcOut),
            "BLEND_DST_OUT" => Some(Self::DstOut),
            "BLEND_SRC_ATOP" => Some(Self::SrcAtop),
            "BLEND_DST_ATOP" => Some(Self::DstAtop),
            "BLEND_XOR" => Some(Self::Xor),
            "BLEND_MULTIPLY" => Some(Self::Multiply),
            "BLEND_SCREEN" => Some(Self::Screen),
            "BLEND_OVERLAY" => Some(Self::Overlay),
            "BLEND_DARKEN" => Some(Self::Darken),
            "BLEND_LIGHTEN" => Some(Self::Lighten),
            "BLEND_COLORDODGE" => Some(Self::ColorDodge),
            "BLEND_COLORBURN" => Some(Self::ColorBurn),
            "BLEND_HARDLIGHT" => Some(Self::HardLight),
            "BLEND_SOFTLIGHT" => Some(Self::SoftLight),
            "BLEND_DIFFERENCE" => Some(Self::Difference),
            "BLEND_EXCLUSION" => Some(Self::Exclusion),
            "BLEND_INVERT" => Some(Self::Invert),
            "BLEND_INVERT_RGB" => Some(Self::InvertRgb),
            "BLEND_LINEARDODGE" => Some(Self::LinearDodge),
            "BLEND_LINEARBURN" => Some(Self::LinearBurn),
            "BLEND_VIVIDLIGHT" => Some(Self::VividLight),
            "BLEND_LINEARLIGHT" => Some(Self::LinearLight),
            "BLEND_PINLIGHT" => Some(Self::PinLight),
            "BLEND_HARDMIX" => Some(Self::HardMix),
            "BLEND_HUE" => Some(Self::Hue),
            "BLEND_SATURATION" => Some(Self::Saturation),
            "BLEND_COLOR" => Some(Self::Color),
            "BLEND_LUMINOSITY" => Some(Self::Luminosity),
            "BLEND_PLUS" => Some(Self::Plus),
            "BLEND_PLUS_DARKER" => Some(Self::PlusDarker),
            "BLEND_MINUS" => Some(Self::Minus),
            "BLEND_CONTRAST" => Some(Self::Contrast),
            "BLEND_INVERT_OVG" => Some(Self::InvertOvg),
            "BLEND_RED" => Some(Self::Red),
            "BLEND_GREEN" => Some(Self::Green),
            "BLEND_BLUE" => Some(Self::Blue),
            _ => None,
        }
    }

    /// Returns the CM3 UI display label for this blend mode.
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Src => "src",
            Self::Dst => "dest",
            Self::SrcOver => "src over",
            Self::DstOver => "dest over",
            Self::SrcIn => "src in",
            Self::DstIn => "dest in",
            Self::SrcOut => "src out",
            Self::DstOut => "dest out",
            Self::SrcAtop => "src atop",
            Self::DstAtop => "dest atop",
            Self::Xor => "xor",
            Self::Multiply => "multiply",
            Self::Screen => "screen",
            Self::Overlay => "overlay",
            Self::Darken => "darken",
            Self::Lighten => "lighten",
            Self::ColorDodge => "color dodge",
            Self::ColorBurn => "color burn",
            Self::HardLight => "hard light",
            Self::SoftLight => "soft light",
            Self::Difference => "difference",
            Self::Exclusion => "exclusion",
            Self::Invert => "invert",
            Self::InvertRgb => "invert rgb",
            Self::LinearDodge => "linear dodge",
            Self::LinearBurn => "linear burn",
            Self::VividLight => "vivid light",
            Self::LinearLight => "linear light",
            Self::PinLight => "pin light",
            Self::HardMix => "hard mix",
            Self::Hue => "hue",
            Self::Saturation => "saturation",
            Self::Color => "color",
            Self::Luminosity => "luminosity",
            Self::Plus => "add",
            Self::PlusDarker => "add darker",
            Self::Minus => "subtract",
            Self::Contrast => "contrast",
            Self::InvertOvg => "invert ovg",
            Self::Red => "red",
            Self::Green => "green",
            Self::Blue => "blue",
        }
    }
}

impl fmt::Display for BlendMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.display_name())
    }
}

/// CPU reference implementation of all 43 CM3 premultiplied-alpha blend modes (Spec 03 §2.6).
///
/// Both `dst` (backdrop) and `src` (foreground, already multiplied by layer opacity) are
/// premultiplied RGBA values. Returns the premultiplied composited result.
pub fn blend_premul(dst: PremulRgba, src: PremulRgba, mode: BlendMode) -> PremulRgba {
    let [dr, dg, db, da] = dst.0;
    let [sr, sg, sb, sa] = src.0;
    let inv_sa = 1.0 - sa;
    let inv_da = 1.0 - da;

    match mode {
        // Porter-Duff modes
        BlendMode::Zero => PremulRgba::TRANSPARENT,
        BlendMode::Src => src,
        BlendMode::Dst => dst,
        BlendMode::SrcOver => PremulRgba([
            sr + dr * inv_sa,
            sg + dg * inv_sa,
            sb + db * inv_sa,
            sa + da * inv_sa,
        ]),
        BlendMode::DstOver => PremulRgba([
            dr + sr * inv_da,
            dg + sg * inv_da,
            db + sb * inv_da,
            da + sa * inv_da,
        ]),
        BlendMode::SrcIn => PremulRgba([sr * da, sg * da, sb * da, sa * da]),
        BlendMode::DstIn => PremulRgba([dr * sa, dg * sa, db * sa, da * sa]),
        BlendMode::SrcOut => PremulRgba([sr * inv_da, sg * inv_da, sb * inv_da, sa * inv_da]),
        BlendMode::DstOut => PremulRgba([dr * inv_sa, dg * inv_sa, db * inv_sa, da * inv_sa]),
        BlendMode::SrcAtop => PremulRgba([
            sr * da + dr * inv_sa,
            sg * da + dg * inv_sa,
            sb * da + db * inv_sa,
            da,
        ]),
        BlendMode::DstAtop => PremulRgba([
            dr * sa + sr * inv_da,
            dg * sa + sg * inv_da,
            db * sa + sb * inv_da,
            sa,
        ]),
        BlendMode::Xor => PremulRgba([
            sr * inv_da + dr * inv_sa,
            sg * inv_da + dg * inv_sa,
            sb * inv_da + db * inv_sa,
            sa + da - 2.0 * sa * da,
        ]),

        // Separable blend modes
        BlendMode::Multiply => compose_separable(dst, src, |d, s| d * s),
        BlendMode::Screen => compose_separable(dst, src, |d, s| d + s - d * s),
        BlendMode::Overlay => compose_separable(dst, src, |d, s| hard_light(s, d)),
        BlendMode::Darken => compose_separable(dst, src, f32::min),
        BlendMode::Lighten => compose_separable(dst, src, f32::max),
        BlendMode::ColorDodge => compose_separable(dst, src, color_dodge),
        BlendMode::ColorBurn => compose_separable(dst, src, color_burn),
        BlendMode::HardLight => compose_separable(dst, src, hard_light),
        BlendMode::SoftLight => compose_separable(dst, src, soft_light),
        BlendMode::Difference => compose_separable(dst, src, |d, s| (d - s).abs()),
        BlendMode::Exclusion => compose_separable(dst, src, |d, s| d + s - 2.0 * d * s),
        BlendMode::InvertRgb => compose_separable(dst, src, |d, s| s * (1.0 - d) + (1.0 - s) * d),
        BlendMode::LinearDodge => compose_separable(dst, src, |d, s| (d + s).min(1.0)),
        BlendMode::LinearBurn => compose_separable(dst, src, |d, s| (d + s - 1.0).max(0.0)),
        BlendMode::VividLight => compose_separable(dst, src, |d, s| {
            if s <= 0.5 {
                color_burn(d, 2.0 * s)
            } else {
                color_dodge(d, 2.0 * s - 1.0)
            }
        }),
        BlendMode::LinearLight => {
            compose_separable(dst, src, |d, s| (d + 2.0 * s - 1.0).clamp(0.0, 1.0))
        }
        BlendMode::PinLight => compose_separable(dst, src, |d, s| {
            if s <= 0.5 {
                d.min(2.0 * s)
            } else {
                d.max(2.0 * s - 1.0)
            }
        }),
        BlendMode::HardMix => {
            compose_separable(dst, src, |d, s| if d + s >= 1.0 { 1.0 } else { 0.0 })
        }
        BlendMode::Contrast => {
            compose_separable(dst, src, |d, s| (0.5 + 2.0 * (d - 0.5) * s).clamp(0.0, 1.0))
        }

        // Non-separable HSL blend modes (W3C Compositing Level 1)
        BlendMode::Hue => {
            compose_non_separable(dst, src, |cd, cs| set_lum(set_sat(cs, sat(cd)), lum(cd)))
        }
        BlendMode::Saturation => {
            compose_non_separable(dst, src, |cd, cs| set_lum(set_sat(cd, sat(cs)), lum(cd)))
        }
        BlendMode::Color => compose_non_separable(dst, src, |cd, cs| set_lum(cs, lum(cd))),
        BlendMode::Luminosity => compose_non_separable(dst, src, |cd, cs| set_lum(cd, lum(cs))),

        // Extended invert modes
        BlendMode::Invert => PremulRgba([
            sa * (da - dr) + dr * inv_sa,
            sa * (da - dg) + dg * inv_sa,
            sa * (da - db) + db * inv_sa,
            da,
        ]),
        BlendMode::InvertOvg => PremulRgba([
            sa * (da - dr) + dr * inv_sa,
            sa * (da - dg) + dg * inv_sa,
            sa * (da - db) + db * inv_sa,
            sa + da * inv_sa,
        ]),

        // Arithmetic modes
        BlendMode::Plus => PremulRgba([
            (sr + dr).min(1.0),
            (sg + dg).min(1.0),
            (sb + db).min(1.0),
            (sa + da).min(1.0),
        ]),
        BlendMode::PlusDarker => PremulRgba([
            (sr + dr - 1.0).max(0.0),
            (sg + dg - 1.0).max(0.0),
            (sb + db - 1.0).max(0.0),
            (sa + da - 1.0).max(0.0),
        ]),
        BlendMode::Minus => PremulRgba([
            (dr - sr).max(0.0),
            (dg - sg).max(0.0),
            (db - sb).max(0.0),
            da,
        ]),

        // Channel replacement modes
        BlendMode::Red => PremulRgba([sr + dr * inv_sa, dg, db, sa + da * inv_sa]),
        BlendMode::Green => PremulRgba([dr, sg + dg * inv_sa, db, sa + da * inv_sa]),
        BlendMode::Blue => PremulRgba([dr, dg, sb + db * inv_sa, sa + da * inv_sa]),
    }
}

fn compose_separable<F>(dst: PremulRgba, src: PremulRgba, blend: F) -> PremulRgba
where
    F: Fn(f32, f32) -> f32,
{
    let [dr, dg, db, da] = dst.0;
    let [sr, sg, sb, sa] = src.0;
    let cd = dst.to_straight().0;
    let cs = src.to_straight().0;
    let inv_da = 1.0 - da;
    let inv_sa = 1.0 - sa;
    let overlap = sa * da;
    PremulRgba([
        inv_da * sr + inv_sa * dr + overlap * blend(cd[0], cs[0]),
        inv_da * sg + inv_sa * dg + overlap * blend(cd[1], cs[1]),
        inv_da * sb + inv_sa * db + overlap * blend(cd[2], cs[2]),
        sa + da * inv_sa,
    ])
}

fn compose_non_separable<F>(dst: PremulRgba, src: PremulRgba, blend: F) -> PremulRgba
where
    F: Fn([f32; 3], [f32; 3]) -> [f32; 3],
{
    let [dr, dg, db, da] = dst.0;
    let [sr, sg, sb, sa] = src.0;
    let cd = dst.to_straight().rgb().0;
    let cs = src.to_straight().rgb().0;
    let b = blend(cd, cs);
    let inv_da = 1.0 - da;
    let inv_sa = 1.0 - sa;
    let overlap = sa * da;
    PremulRgba([
        inv_da * sr + inv_sa * dr + overlap * b[0],
        inv_da * sg + inv_sa * dg + overlap * b[1],
        inv_da * sb + inv_sa * db + overlap * b[2],
        sa + da * inv_sa,
    ])
}

fn hard_light(d: f32, s: f32) -> f32 {
    if s <= 0.5 {
        2.0 * d * s
    } else {
        1.0 - 2.0 * (1.0 - d) * (1.0 - s)
    }
}

fn color_dodge(d: f32, s: f32) -> f32 {
    if d <= 0.0 {
        0.0
    } else if s >= 1.0 {
        1.0
    } else {
        (d / (1.0 - s)).min(1.0)
    }
}

fn color_burn(d: f32, s: f32) -> f32 {
    if d >= 1.0 {
        1.0
    } else if s <= 0.0 {
        0.0
    } else {
        1.0 - ((1.0 - d) / s).min(1.0)
    }
}

fn soft_light(d: f32, s: f32) -> f32 {
    if s <= 0.5 {
        d - (1.0 - 2.0 * s) * d * (1.0 - d)
    } else {
        let g = if d <= 0.25 {
            ((16.0 * d - 12.0) * d + 4.0) * d
        } else {
            d.sqrt()
        };
        d + (2.0 * s - 1.0) * (g - d)
    }
}

fn lum(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

fn clip_color(mut c: [f32; 3]) -> [f32; 3] {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    if n < 0.0 {
        let denom = (l - n).max(UNPREMUL_EPS);
        for ch in &mut c {
            *ch = l + ((*ch - l) * l) / denom;
        }
    }
    if x > 1.0 {
        let denom = (x - l).max(UNPREMUL_EPS);
        for ch in &mut c {
            *ch = l + ((*ch - l) * (1.0 - l)) / denom;
        }
    }
    c
}

fn set_lum(c: [f32; 3], target_lum: f32) -> [f32; 3] {
    let d = target_lum - lum(c);
    clip_color([c[0] + d, c[1] + d, c[2] + d])
}

fn sat(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
    let mut idx = [0usize, 1, 2];
    if c[idx[0]] > c[idx[1]] {
        idx.swap(0, 1);
    }
    if c[idx[1]] > c[idx[2]] {
        idx.swap(1, 2);
    }
    if c[idx[0]] > c[idx[1]] {
        idx.swap(0, 1);
    }
    let (i_min, i_mid, i_max) = (idx[0], idx[1], idx[2]);
    let mut out = [0.0f32; 3];
    if c[i_max] > c[i_min] {
        out[i_mid] = ((c[i_mid] - c[i_min]) * s) / (c[i_max] - c[i_min]);
        out[i_max] = s;
    }
    out[i_min] = 0.0;
    out
}

fn parse_hex_byte(pair: &str, full: &str) -> Result<u8, ColorError> {
    if pair.len() != 2 || !pair.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(ColorError::InvalidHex(full.to_string()));
    }
    u8::from_str_radix(pair, 16).map_err(|_| ColorError::InvalidHex(full.to_string()))
}

fn wrap_unit_f32(x: f32) -> f32 {
    let w = x.rem_euclid(1.0);
    if w >= 1.0 { 0.0 } else { w }
}

fn wrap_unit_f64(x: f64) -> f64 {
    let w = x.rem_euclid(1.0);
    if w >= 1.0 { 0.0 } else { w }
}

fn rgb_to_hsl_f64(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let r = if r.is_finite() {
        r.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let g = if g.is_finite() {
        g.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let b = if b.is_finite() {
        b.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = 0.5 * (max + min);
    let d = max - min;
    if d == 0.0 {
        return (0.0, 0.0, l);
    }
    let denom = if l > 0.5 { 2.0 - max - min } else { max + min };
    let s = if denom > 0.0 {
        (d / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let h_raw = if max == r {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (wrap_unit_f64(h_raw / 6.0), s, l)
}

fn hsl_to_rgb_f64(h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    let h = wrap_unit_f64(h);
    if s <= 0.0 {
        return (l, l, l);
    }
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    (
        hue_to_rgb_f64(p, q, h + 1.0 / 3.0),
        hue_to_rgb_f64(p, q, h),
        hue_to_rgb_f64(p, q, h - 1.0 / 3.0),
    )
}

fn hue_to_rgb_f64(p: f64, q: f64, t: f64) -> f64 {
    let t = wrap_unit_f64(t);
    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 0.5 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}

fn rgb_from_f64_rounded(r: f64, g: f64, b: f64) -> Rgb {
    Rgb::from_u8(
        (r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (b.clamp(0.0, 1.0) * 255.0).round() as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_rgb_close(a: Rgb, b: Rgb, eps: f32) {
        for i in 0..3 {
            assert!(
                (a.0[i] - b.0[i]).abs() <= eps,
                "channel {i}: expected {} close to {} (eps {eps})",
                a.0[i],
                b.0[i]
            );
        }
    }

    fn assert_premul_close(a: PremulRgba, b: PremulRgba, eps: f32) {
        for i in 0..4 {
            assert!(
                (a.0[i] - b.0[i]).abs() <= eps,
                "channel {i}: expected {} close to {} (eps {eps})",
                a.0[i],
                b.0[i]
            );
        }
    }

    #[test]
    fn classic_theme_derivation_is_byte_for_byte_exact() {
        let derived = ThemePalette::derive_from_main("#384668").unwrap();
        assert_eq!(derived.focus.to_hex(), "#384668");
        assert_eq!(derived.link.to_hex(), "#7e8fb9");
        assert_eq!(derived.clip.to_hex(), "#526183");
        assert_eq!(derived.clip_selected.to_hex(), "#7f94c7");
        assert_eq!(derived.clip_selection.to_hex(), "#273465");

        // Built-in Panzoid Classic profile (0 deg, 1.0 sat) also matches byte-for-byte.
        let classic_prof = ThemePalette::derive_profile(0.0, 1.0);
        assert_eq!(classic_prof.focus.to_hex(), "#384668");
        assert_eq!(classic_prof.link.to_hex(), "#7e8fb9");
        assert_eq!(classic_prof.clip.to_hex(), "#526183");
        assert_eq!(classic_prof.clip_selected.to_hex(), "#7f94c7");
        assert_eq!(classic_prof.clip_selection.to_hex(), "#273465");

        // Gray profile sets saturation to 0 while preserving lightness.
        let gray = ThemeProfile::Gray.palette();
        assert_eq!(gray.focus.to_u8()[0], gray.focus.to_u8()[1]);
        assert_eq!(gray.focus.to_u8()[1], gray.focus.to_u8()[2]);
        assert_eq!(gray.focus.to_hex(), "#505050");

        // Selected custom clip shifts lightness 35% toward white.
        let base = Rgb::from_hex("#384668").unwrap();
        let selected = selected_custom_clip_color(base);
        let l0 = base.to_hsl().l;
        let l1 = selected.to_hsl().l;
        assert!((l1 - (l0 + (1.0 - l0) * 0.35)).abs() < 0.01);
    }

    #[test]
    fn wcag_luminance_contrast_and_clip_label_choice() {
        assert!((relative_luminance(Rgb::BLACK) - 0.0).abs() < 1e-6);
        assert!((relative_luminance(Rgb::WHITE) - 1.0).abs() < 1e-6);
        assert!((contrast_ratio(Rgb::BLACK, Rgb::WHITE) - 21.0).abs() < 1e-4);
        assert!((contrast_ratio(Rgb::WHITE, Rgb::WHITE) - 1.0).abs() < 1e-6);

        assert_eq!(
            choose_clip_label_color(Rgb::from_hex("#384668").unwrap()),
            CLIP_LABEL_LIGHT
        );
        assert_eq!(choose_clip_label_color(Rgb::BLACK), CLIP_LABEL_LIGHT);
        assert_eq!(choose_clip_label_color(Rgb::WHITE), CLIP_LABEL_DARK);
        assert_eq!(
            choose_clip_label_color(Rgb::from_hex("#e0e0e0").unwrap()),
            CLIP_LABEL_DARK
        );
        assert_eq!(CLIP_LABEL_DARK.to_hex(), "#2c323f");
        assert_eq!(CLIP_LABEL_LIGHT.to_hex(), "#f4f4f4");
    }

    #[test]
    fn color_space_round_trips_srgb_hsl_hsv_oklab() {
        let samples = [
            Rgb::BLACK,
            Rgb::WHITE,
            Rgb::new(1.0, 0.0, 0.0),
            Rgb::new(0.0, 1.0, 0.0),
            Rgb::new(0.0, 0.0, 1.0),
            Rgb::new(0.5, 0.5, 0.5),
            Rgb::from_hex("#384668").unwrap(),
            Rgb::from_hex("#e87a3c").unwrap(),
            Rgb::from_hex("#2bb89f").unwrap(),
        ];

        for rgb in samples {
            assert_rgb_close(Rgb::from_linear(rgb.to_linear()), rgb, 1e-5);
            assert_rgb_close(Rgb::from_hsl(rgb.to_hsl()), rgb, 1e-5);
            assert_rgb_close(Rgb::from_hsv(rgb.to_hsv()), rgb, 1e-5);
            assert_rgb_close(Rgb::from_oklab(rgb.to_oklab()), rgb, 1e-4);
        }
    }

    #[test]
    fn hex_parsing_formatting_and_premul_safety() {
        let c = Rgb::from_hex("#384668").unwrap();
        assert_eq!(c.to_hex(), "#384668");
        assert_eq!(Rgb::from_hex("384668").unwrap(), c);
        assert!(Rgb::from_hex("#12345").is_err());
        assert!(Rgb::from_hex("#gggggg").is_err());

        let ca = Rgba::from_hex("#38466880").unwrap();
        assert_eq!(ca.to_hex_rgba(), "#38466880");
        assert_eq!(ca.to_hex_rgb(), "#384668");
        assert_eq!(
            Rgba::from_hex("#384668").unwrap().to_hex_rgba(),
            "#384668ff"
        );

        let p = ca.to_premul();
        let back = p.to_straight();
        for i in 0..4 {
            assert!((back.0[i] - ca.0[i]).abs() < 1e-5);
        }

        // Safe unpremultiply at alpha = 0 and tiny alpha (no NaN / Inf).
        assert_eq!(PremulRgba::TRANSPARENT.to_straight(), Rgba::TRANSPARENT);
        let tiny = PremulRgba::new(1e-8, 1e-8, 1e-8, 1e-8).to_straight();
        assert!(tiny.0.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn all_43_blend_modes_tokens_boundaries_and_invariants() {
        assert_eq!(BlendMode::ALL.len(), 43);
        for mode in BlendMode::ALL {
            let tok = mode.token();
            assert!(tok.starts_with("BLEND_"));
            assert_eq!(BlendMode::from_token(tok), Some(mode));
            assert!(!mode.display_name().is_empty());
        }

        let dst_opaque = Rgba::new(0.7, 0.4, 0.2, 1.0).to_premul();
        let src_opaque = Rgba::new(0.3, 0.8, 0.6, 1.0).to_premul();
        let dst_semi = Rgba::new(0.7, 0.4, 0.2, 0.6).to_premul();
        let src_semi = Rgba::new(0.3, 0.8, 0.6, 0.4).to_premul();

        let pairs = [
            (PremulRgba::TRANSPARENT, PremulRgba::TRANSPARENT),
            (dst_opaque, PremulRgba::TRANSPARENT),
            (PremulRgba::TRANSPARENT, src_opaque),
            (dst_opaque, src_opaque),
            (dst_semi, src_semi),
        ];

        for mode in BlendMode::ALL {
            for (d, s) in pairs {
                let out = blend_premul(d, s, mode);
                for ch in out.0 {
                    assert!(ch.is_finite(), "non-finite output for {mode:?}: {out:?}");
                    assert!(
                        (-1e-5..=1.0 + 1e-5).contains(&ch),
                        "out-of-range channel for {mode:?}: {out:?}"
                    );
                }
                let a = out.0[3];
                for c in &out.0[0..3] {
                    assert!(
                        *c <= a + 1e-5,
                        "premultiplied invariant violated for {mode:?}: {out:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn blend_mode_reference_spot_checks() {
        let d = Rgba::new(0.8, 0.4, 0.2, 1.0).to_premul();
        let s = Rgba::new(0.5, 0.6, 0.9, 1.0).to_premul();

        assert_premul_close(
            blend_premul(d, s, BlendMode::Zero),
            PremulRgba::TRANSPARENT,
            1e-6,
        );
        assert_premul_close(blend_premul(d, s, BlendMode::Src), s, 1e-6);
        assert_premul_close(blend_premul(d, s, BlendMode::Dst), d, 1e-6);
        assert_premul_close(blend_premul(d, s, BlendMode::SrcOver), s, 1e-6);
        assert_premul_close(blend_premul(d, s, BlendMode::DstOver), d, 1e-6);
        assert_premul_close(
            blend_premul(d, s, BlendMode::Multiply),
            PremulRgba::new(0.4, 0.24, 0.18, 1.0),
            1e-5,
        );
        assert_premul_close(
            blend_premul(d, s, BlendMode::Screen),
            PremulRgba::new(0.9, 0.76, 0.92, 1.0),
            1e-5,
        );
        assert_premul_close(
            blend_premul(d, s, BlendMode::Darken),
            PremulRgba::new(0.5, 0.4, 0.2, 1.0),
            1e-5,
        );
        assert_premul_close(
            blend_premul(d, s, BlendMode::Lighten),
            PremulRgba::new(0.8, 0.6, 0.9, 1.0),
            1e-5,
        );
        assert_premul_close(
            blend_premul(d, s, BlendMode::Difference),
            PremulRgba::new(0.3, 0.2, 0.7, 1.0),
            1e-5,
        );
        assert_premul_close(
            blend_premul(d, s, BlendMode::Plus),
            PremulRgba::new(1.0, 1.0, 1.0, 1.0),
            1e-5,
        );
        assert_premul_close(
            blend_premul(d, s, BlendMode::Minus),
            PremulRgba::new(0.3, 0.0, 0.0, 1.0),
            1e-5,
        );
        assert_premul_close(
            blend_premul(d, s, BlendMode::Invert),
            PremulRgba::new(0.2, 0.6, 0.8, 1.0),
            1e-5,
        );
        assert_premul_close(
            blend_premul(d, s, BlendMode::HardMix),
            PremulRgba::new(1.0, 1.0, 1.0, 1.0),
            1e-5,
        );
        assert_premul_close(
            blend_premul(d, s, BlendMode::Red),
            PremulRgba::new(0.5, 0.4, 0.2, 1.0),
            1e-5,
        );
        assert_premul_close(
            blend_premul(d, s, BlendMode::Green),
            PremulRgba::new(0.8, 0.6, 0.2, 1.0),
            1e-5,
        );
        assert_premul_close(
            blend_premul(d, s, BlendMode::Blue),
            PremulRgba::new(0.8, 0.4, 0.9, 1.0),
            1e-5,
        );

        // Non-separable Luminosity preserves destination chroma and adopts source luminance.
        let lum_out = blend_premul(d, s, BlendMode::Luminosity);
        assert!((lum(lum_out.to_straight().rgb().0) - lum(s.to_straight().rgb().0)).abs() < 1e-4);

        // Semi-transparent SrcOver verification.
        let half_red = Rgba::new(1.0, 0.0, 0.0, 0.5).to_premul();
        let opaque_blue = Rgba::new(0.0, 0.0, 1.0, 1.0).to_premul();
        assert_premul_close(
            blend_premul(opaque_blue, half_red, BlendMode::SrcOver),
            PremulRgba::new(0.5, 0.0, 0.5, 1.0),
            1e-6,
        );
    }

    #[test]
    fn contrast_blend_hex_validation_and_out_of_gamut_hsl_hsv() {
        // Bug 8: Contrast formula matches compositor: clamp(0.5 + 2*(d - 0.5)*s, 0, 1).
        let dst = Rgba::new(0.8, 0.4, 0.2, 1.0).to_premul();
        let zero = Rgba::new(0.0, 0.0, 0.0, 1.0).to_premul();
        let mid = Rgba::new(0.5, 0.5, 0.5, 1.0).to_premul();
        let one = Rgba::new(1.0, 1.0, 1.0, 1.0).to_premul();

        assert_premul_close(
            blend_premul(dst, zero, BlendMode::Contrast),
            PremulRgba::new(0.5, 0.5, 0.5, 1.0),
            1e-6,
        );
        assert_premul_close(blend_premul(dst, mid, BlendMode::Contrast), dst, 1e-6);
        assert_premul_close(
            blend_premul(dst, one, BlendMode::Contrast),
            PremulRgba::new(1.0, 0.3, 0.0, 1.0),
            1e-5,
        );

        // Bug 9: Reject '+'-prefixed hex byte pairs.
        assert!(Rgb::from_hex("#+f+f+f").is_err());
        assert!(Rgb::from_hex("#+1+2+3").is_err());
        assert!(Rgba::from_hex("#+1+2+3").is_err());
        assert!(Rgba::from_hex("#+1+2+3+4").is_err());

        // Bug 10: Out-of-gamut and non-finite RGB stay finite and in [0, 1] for HSL/HSV.
        for rgb in [
            Rgb::new(1.5, 0.5, 0.5),
            Rgb::new(0.5, -0.5, -0.5),
            Rgb::new(-0.2, -0.5, -0.5),
            Rgb::new(f32::NAN, 0.5, f32::INFINITY),
        ] {
            let hsl = rgb.to_hsl();
            assert!(hsl.h.is_finite() && (0.0..1.0).contains(&hsl.h));
            assert!(hsl.s.is_finite() && (0.0..=1.0).contains(&hsl.s));
            assert!(hsl.l.is_finite() && (0.0..=1.0).contains(&hsl.l));
            let back = Rgb::from_hsl(hsl);
            assert!(back.0.iter().all(|c| c.is_finite()));

            let hsv = rgb.to_hsv();
            assert!(hsv.h.is_finite() && (0.0..1.0).contains(&hsv.h));
            assert!(hsv.s.is_finite() && (0.0..=1.0).contains(&hsv.s));
            assert!(hsv.v.is_finite() && (0.0..=1.0).contains(&hsv.v));
        }
    }
}
