//! Floating Setup windows for Retro & Glitch FX (`VHS Setup`, `Datamosh Setup`, `ASCII Setup`,
//! `Tracery Setup`), reproducing Zoidium's `vhs-setup.js`, `datamosh-setup.js`, and
//! `effect-windows.js` with deterministic seeded randomizers, presets, and atomic Undo steps.

use crate::ui::controls::{
    ButtonVariant, FloatingWindowManager, FloatingWindowSpec, ScrubConfig, cm3_collapsible_section,
    cm3_floating_window, cm3_hsva_color_picker, cm3_prop_button, cm3_scrub_number,
    request_eyedropper_sampler,
};
use crate::ui::theme::{
    BG_INPUT, BG_PANEL, BG_SUNKEN, BORDER_CONTROL, BORDER_PANEL, Cm3Theme, FONT_BODY, FONT_COMPACT,
    FONT_SMALL, TEXT_FAINT, TEXT_MUTED, TEXT_PRIMARY, TEXT_STRONG, WARNING_BAR, WARNING_SURFACE,
};
use egui::{CornerRadius, Id, Key, Rect, ScrollArea, Stroke, TextEdit, Ui, Vec2};
use serde_json::{Value, json};
use zc_color::Rgba;
use zc_session::{Command, Session, SessionEffect, SessionProperty};

use super::*;

// ===========================================================================
// Shared Deterministic Seeded PRNG (`zoidium/shared/random.js:16-27`)
// ===========================================================================

/// Exact Zoidium `legacyHash01(seed, index)` shared by VHS Setup and Datamosh Setup.
#[must_use]
pub fn legacy_hash01(seed: i32, index: i32) -> f64 {
    let mut h = seed
        .wrapping_mul(374_761_393)
        .wrapping_add(index.wrapping_mul(668_265_263)) as u32;
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    f64::from(h) / 4_294_967_296.0
}

/// Exact Zoidium `nextSeed(seed)` (`((Math.round(Number(seed)) || 0) + 7919) % 10000`).
#[must_use]
pub fn next_seed(seed: f64) -> f64 {
    let base = if seed.is_finite() {
        seed.round() as i64
    } else {
        0
    };
    (((base + 7919) % 10_000 + 10_000) % 10_000) as f64
}

fn same_value(a: f64, b: f64) -> bool {
    (a - b).abs() < 0.015
}

// ===========================================================================
// Public State & Window Entry Point
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectSetupKind {
    Vhs,
    Datamosh,
    Ascii,
    Tracery,
}

impl EffectSetupKind {
    #[must_use]
    pub const fn window_id(self) -> &'static str {
        match self {
            Self::Vhs => "pro_vhs_setup",
            Self::Datamosh => "pro_datamosh_setup",
            Self::Ascii => "pro_ascii_setup",
            Self::Tracery => "pro_tracery_setup",
        }
    }

    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Self::Vhs => "VHS Setup",
            Self::Datamosh => "Datamosh Setup",
            Self::Ascii => "ASCII Setup",
            Self::Tracery => "Tracery Setup",
        }
    }

    #[must_use]
    pub const fn subtitle(self) -> &'static str {
        match self {
            Self::Vhs => "Signal chain",
            Self::Datamosh => "Segment motion",
            Self::Ascii => "Character matrix",
            Self::Tracery => "Callout tracker",
        }
    }

    #[must_use]
    pub const fn default_size(self) -> Vec2 {
        match self {
            Self::Vhs => Vec2::new(430.0, 640.0),
            Self::Datamosh => Vec2::new(480.0, 640.0),
            Self::Ascii => Vec2::new(430.0, 580.0),
            Self::Tracery => Vec2::new(440.0, 620.0),
        }
    }
}

/// Interactive state for the floating Retro & Glitch FX Setup window.
#[derive(Debug, Clone)]
pub struct EffectSetupState {
    pub open: bool,
    pub kind: EffectSetupKind,
    pub effect_id: String,
    pub effect_label: String,
    pub just_opened: bool,
    pub tracery_point_tab: usize,
    pub(crate) active_tx_key: Option<String>,
    pub(crate) tx_seen_this_frame: bool,
}

impl Default for EffectSetupState {
    fn default() -> Self {
        Self {
            open: false,
            kind: EffectSetupKind::Vhs,
            effect_id: String::new(),
            effect_label: String::new(),
            just_opened: false,
            tracery_point_tab: 0,
            active_tx_key: None,
            tx_seen_this_frame: false,
        }
    }
}

impl EffectSetupState {
    pub fn open_for_effect(
        &mut self,
        kind: EffectSetupKind,
        effect_id: impl Into<String>,
        effect_label: impl Into<String>,
    ) {
        self.kind = kind;
        self.effect_id = effect_id.into();
        self.effect_label = effect_label.into();
        self.open = true;
        self.just_opened = true;
    }
}

// ===========================================================================
// VHS Presets & Sections (`plugins/openzoid-legacy/vhs-setup.js`)
// ===========================================================================

pub struct ScalarPreset {
    pub id: &'static str,
    pub label: &'static str,
    pub values: &'static [(&'static str, f64)],
}

pub const VHS_PRESETS: &[ScalarPreset] = &[
    ScalarPreset {
        id: "clean",
        label: "Clean VHS",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.12),
            ("colorStripes", 0.1),
            ("chromaCrawl", 0.08),
            ("chromaLoss", 0.0),
            ("flicker", 0.03),
            ("vhsSharpen", 0.3),
            ("interlace", 0.35),
            ("staticWarp", 0.05),
            ("fastForward", 0.0),
            ("tracking", 0.08),
            ("anisotropyWarp", 0.0),
            ("jitterX", 0.2),
            ("jitterY", 0.2),
            ("roll", 0.0),
            ("rollSpeed", 1.5),
            ("persistence", 0.2),
            ("rgbSplit", 0.08),
            ("tear", 0.05),
            ("humBar", 0.05),
            ("blueScreen", 0.0),
            ("skew", 0.2),
            ("crease", 0.0),
            ("ghost", 0.05),
            ("osd", 0.0),
            ("noiseEvo", 0.4),
            ("noiseOpacity", 0.3),
            ("blinds", 0.0),
            ("snowDots", 0.0),
            ("speedDots", 0.0),
            ("anisoSnow", 0.0),
        ],
    },
    ScalarPreset {
        id: "broadcast",
        label: "Broadcast",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.25),
            ("colorStripes", 0.3),
            ("chromaCrawl", 0.22),
            ("chromaLoss", 0.02),
            ("flicker", 0.06),
            ("vhsSharpen", 0.35),
            ("interlace", 0.45),
            ("staticWarp", 0.12),
            ("fastForward", 0.0),
            ("tracking", 0.18),
            ("anisotropyWarp", 0.0),
            ("jitterX", 0.3),
            ("jitterY", 0.3),
            ("roll", 0.0),
            ("rollSpeed", 1.5),
            ("persistence", 0.25),
            ("rgbSplit", 0.12),
            ("tear", 0.12),
            ("humBar", 0.1),
            ("blueScreen", 0.0),
            ("skew", 0.25),
            ("crease", 0.0),
            ("ghost", 0.08),
            ("osd", 0.0),
            ("noiseEvo", 0.5),
            ("noiseOpacity", 0.4),
            ("blinds", 0.0),
            ("snowDots", 0.0),
            ("speedDots", 0.0),
            ("anisoSnow", 0.0),
        ],
    },
    ScalarPreset {
        id: "damaged",
        label: "Damaged Tape",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.7),
            ("colorStripes", 0.55),
            ("chromaCrawl", 0.5),
            ("chromaLoss", 0.35),
            ("flicker", 0.3),
            ("vhsSharpen", 0.5),
            ("interlace", 0.6),
            ("staticWarp", 0.65),
            ("fastForward", 0.0),
            ("tracking", 0.6),
            ("anisotropyWarp", 0.4),
            ("jitterX", 0.6),
            ("jitterY", 0.6),
            ("roll", 0.25),
            ("rollSpeed", 1.5),
            ("persistence", 0.35),
            ("rgbSplit", 0.4),
            ("tear", 0.6),
            ("humBar", 0.4),
            ("blueScreen", 0.0),
            ("skew", 0.6),
            ("crease", 0.5),
            ("ghost", 0.4),
            ("osd", 0.0),
            ("noiseEvo", 0.7),
            ("noiseOpacity", 0.6),
            ("blinds", 0.0),
            ("snowDots", 0.5),
            ("speedDots", 0.4),
            ("anisoSnow", 0.45),
        ],
    },
    ScalarPreset {
        id: "fastfwd",
        label: "Fast Forward",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.35),
            ("colorStripes", 0.4),
            ("chromaCrawl", 0.3),
            ("chromaLoss", 0.15),
            ("flicker", 0.15),
            ("vhsSharpen", 0.4),
            ("interlace", 0.55),
            ("staticWarp", 0.25),
            ("fastForward", 0.8),
            ("tracking", 0.35),
            ("anisotropyWarp", 0.15),
            ("jitterX", 0.4),
            ("jitterY", 0.5),
            ("roll", 0.35),
            ("rollSpeed", 2.5),
            ("persistence", 0.4),
            ("rgbSplit", 0.3),
            ("tear", 0.35),
            ("humBar", 0.25),
            ("blueScreen", 0.0),
            ("skew", 0.4),
            ("crease", 0.2),
            ("ghost", 0.25),
            ("osd", 0.0),
            ("noiseEvo", 0.8),
            ("noiseOpacity", 0.45),
            ("blinds", 0.0),
            ("snowDots", 0.35),
            ("speedDots", 0.7),
            ("anisoSnow", 0.3),
        ],
    },
    ScalarPreset {
        id: "trackingErr",
        label: "Tracking Error",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.5),
            ("colorStripes", 0.35),
            ("chromaCrawl", 0.35),
            ("chromaLoss", 0.2),
            ("flicker", 0.2),
            ("vhsSharpen", 0.4),
            ("interlace", 0.5),
            ("staticWarp", 0.4),
            ("fastForward", 0.0),
            ("tracking", 0.9),
            ("anisotropyWarp", 0.3),
            ("jitterX", 0.5),
            ("jitterY", 0.5),
            ("roll", 0.15),
            ("rollSpeed", 1.2),
            ("persistence", 0.3),
            ("rgbSplit", 0.35),
            ("tear", 0.5),
            ("humBar", 0.3),
            ("blueScreen", 0.0),
            ("skew", 0.5),
            ("crease", 0.35),
            ("ghost", 0.3),
            ("osd", 0.0),
            ("noiseEvo", 0.7),
            ("noiseOpacity", 0.6),
            ("blinds", 0.0),
            ("snowDots", 0.35),
            ("speedDots", 0.3),
            ("anisoSnow", 0.3),
        ],
    },
    ScalarPreset {
        id: "shot01",
        label: "Shot 01 · MESECAM tears",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.8),
            ("colorStripes", 0.15),
            ("chromaCrawl", 0.3),
            ("chromaLoss", 0.05),
            ("flicker", 0.05),
            ("vhsSharpen", 0.55),
            ("interlace", 0.4),
            ("staticWarp", 0.25),
            ("fastForward", 0.0),
            ("tracking", 0.15),
            ("anisotropyWarp", 0.0),
            ("jitterX", 0.45),
            ("jitterY", 0.3),
            ("roll", 0.0),
            ("rollSpeed", 1.5),
            ("persistence", 0.25),
            ("rgbSplit", 0.25),
            ("tear", 0.15),
            ("humBar", 0.08),
            ("blueScreen", 0.0),
            ("skew", 0.25),
            ("crease", 0.0),
            ("ghost", 0.08),
            ("osd", 0.0),
            ("noiseEvo", 0.5),
            ("noiseOpacity", 0.4),
            ("blinds", 0.0),
            ("snowDots", 0.1),
            ("speedDots", 0.0),
            ("anisoSnow", 0.15),
        ],
    },
    ScalarPreset {
        id: "shot02",
        label: "Shot 02 · WTTW stripes",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.3),
            ("colorStripes", 0.85),
            ("chromaCrawl", 0.3),
            ("chromaLoss", 0.05),
            ("flicker", 0.08),
            ("vhsSharpen", 0.35),
            ("interlace", 0.45),
            ("staticWarp", 0.15),
            ("fastForward", 0.0),
            ("tracking", 0.25),
            ("anisotropyWarp", 0.0),
            ("jitterX", 0.35),
            ("jitterY", 0.3),
            ("roll", 0.0),
            ("rollSpeed", 1.5),
            ("persistence", 0.3),
            ("rgbSplit", 0.3),
            ("tear", 0.15),
            ("humBar", 0.12),
            ("blueScreen", 0.0),
            ("skew", 0.3),
            ("crease", 0.05),
            ("ghost", 0.12),
            ("osd", 1.0),
            ("noiseEvo", 0.5),
            ("noiseOpacity", 0.4),
            ("blinds", 0.0),
            ("snowDots", 0.0),
            ("speedDots", 0.0),
            ("anisoSnow", 0.0),
        ],
    },
    ScalarPreset {
        id: "shot03",
        label: "Shot 03 · WTTW clean",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.15),
            ("colorStripes", 0.15),
            ("chromaCrawl", 0.15),
            ("chromaLoss", 0.02),
            ("flicker", 0.04),
            ("vhsSharpen", 0.5),
            ("interlace", 0.4),
            ("staticWarp", 0.08),
            ("fastForward", 0.0),
            ("tracking", 0.12),
            ("anisotropyWarp", 0.0),
            ("jitterX", 0.25),
            ("jitterY", 0.25),
            ("roll", 0.0),
            ("rollSpeed", 1.5),
            ("persistence", 0.25),
            ("rgbSplit", 0.1),
            ("tear", 0.06),
            ("humBar", 0.06),
            ("blueScreen", 0.0),
            ("skew", 0.2),
            ("crease", 0.0),
            ("ghost", 0.05),
            ("osd", 1.0),
            ("noiseEvo", 0.4),
            ("noiseOpacity", 0.3),
            ("blinds", 0.0),
            ("snowDots", 0.0),
            ("speedDots", 0.0),
            ("anisoSnow", 0.0),
        ],
    },
    ScalarPreset {
        id: "shot04",
        label: "Shot 04 · Pattern blur",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.3),
            ("colorStripes", 0.4),
            ("chromaCrawl", 0.7),
            ("chromaLoss", 0.08),
            ("flicker", 0.08),
            ("vhsSharpen", 0.4),
            ("interlace", 0.7),
            ("staticWarp", 0.15),
            ("fastForward", 0.0),
            ("tracking", 0.2),
            ("anisotropyWarp", 0.0),
            ("jitterX", 0.3),
            ("jitterY", 0.3),
            ("roll", 0.0),
            ("rollSpeed", 1.5),
            ("persistence", 0.45),
            ("rgbSplit", 0.35),
            ("tear", 0.2),
            ("humBar", 0.1),
            ("blueScreen", 0.0),
            ("skew", 0.3),
            ("crease", 0.1),
            ("ghost", 0.15),
            ("osd", 0.0),
            ("noiseEvo", 0.5),
            ("noiseOpacity", 0.4),
            ("blinds", 0.0),
            ("snowDots", 0.1),
            ("speedDots", 0.15),
            ("anisoSnow", 0.1),
        ],
    },
    ScalarPreset {
        id: "shot05",
        label: "Shot 05 · Streak strip",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.5),
            ("colorStripes", 0.4),
            ("chromaCrawl", 0.4),
            ("chromaLoss", 0.2),
            ("flicker", 0.2),
            ("vhsSharpen", 0.3),
            ("interlace", 0.5),
            ("staticWarp", 0.9),
            ("fastForward", 0.0),
            ("tracking", 0.7),
            ("anisotropyWarp", 0.2),
            ("jitterX", 0.7),
            ("jitterY", 0.5),
            ("roll", 0.0),
            ("rollSpeed", 1.5),
            ("persistence", 0.3),
            ("rgbSplit", 0.35),
            ("tear", 0.7),
            ("humBar", 0.3),
            ("blueScreen", 0.0),
            ("skew", 0.5),
            ("crease", 0.5),
            ("ghost", 0.3),
            ("osd", 0.0),
            ("noiseEvo", 0.7),
            ("noiseOpacity", 0.6),
            ("blinds", 0.0),
            ("snowDots", 0.6),
            ("speedDots", 0.4),
            ("anisoSnow", 0.5),
        ],
    },
    ScalarPreset {
        id: "shot06",
        label: "Shot 06 · Desat mono",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.2),
            ("colorStripes", 0.2),
            ("chromaCrawl", 0.1),
            ("chromaLoss", 0.95),
            ("flicker", 0.06),
            ("vhsSharpen", 0.35),
            ("interlace", 0.45),
            ("staticWarp", 0.12),
            ("fastForward", 0.0),
            ("tracking", 0.15),
            ("anisotropyWarp", 0.0),
            ("jitterX", 0.3),
            ("jitterY", 0.3),
            ("roll", 0.0),
            ("rollSpeed", 1.5),
            ("persistence", 0.25),
            ("rgbSplit", 0.1),
            ("tear", 0.08),
            ("humBar", 0.06),
            ("blueScreen", 0.0),
            ("skew", 0.2),
            ("crease", 0.0),
            ("ghost", 0.05),
            ("osd", 0.0),
            ("noiseEvo", 0.4),
            ("noiseOpacity", 0.3),
            ("blinds", 0.0),
            ("snowDots", 0.0),
            ("speedDots", 0.0),
            ("anisoSnow", 0.0),
        ],
    },
    ScalarPreset {
        id: "shot07",
        label: "Shot 07 · Fast forward",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.35),
            ("colorStripes", 0.35),
            ("chromaCrawl", 0.3),
            ("chromaLoss", 0.55),
            ("flicker", 0.12),
            ("vhsSharpen", 0.35),
            ("interlace", 0.5),
            ("staticWarp", 0.35),
            ("fastForward", 0.7),
            ("tracking", 0.3),
            ("anisotropyWarp", 0.1),
            ("jitterX", 0.4),
            ("jitterY", 0.45),
            ("roll", 0.2),
            ("rollSpeed", 2.2),
            ("persistence", 0.35),
            ("rgbSplit", 0.25),
            ("tear", 0.4),
            ("humBar", 0.2),
            ("blueScreen", 0.0),
            ("skew", 0.45),
            ("crease", 0.3),
            ("ghost", 0.2),
            ("osd", 0.0),
            ("noiseEvo", 0.8),
            ("noiseOpacity", 0.45),
            ("blinds", 0.0),
            ("snowDots", 0.4),
            ("speedDots", 0.55),
            ("anisoSnow", 0.35),
        ],
    },
    ScalarPreset {
        id: "shot08",
        label: "Shot 08 · Blue blowout",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.4),
            ("colorStripes", 0.45),
            ("chromaCrawl", 0.4),
            ("chromaLoss", 0.1),
            ("flicker", 0.85),
            ("vhsSharpen", 0.35),
            ("interlace", 0.5),
            ("staticWarp", 0.3),
            ("fastForward", 0.0),
            ("tracking", 0.5),
            ("anisotropyWarp", 0.1),
            ("jitterX", 0.5),
            ("jitterY", 0.4),
            ("roll", 0.0),
            ("rollSpeed", 1.5),
            ("persistence", 0.3),
            ("rgbSplit", 0.3),
            ("tear", 0.3),
            ("humBar", 0.35),
            ("blueScreen", 0.15),
            ("skew", 0.35),
            ("crease", 0.1),
            ("ghost", 0.15),
            ("osd", 0.0),
            ("noiseEvo", 0.5),
            ("noiseOpacity", 0.45),
            ("blinds", 0.0),
            ("snowDots", 0.15),
            ("speedDots", 0.2),
            ("anisoSnow", 0.2),
        ],
    },
    ScalarPreset {
        id: "shot09",
        label: "Shot 09 · Tracking bars",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.4),
            ("colorStripes", 0.35),
            ("chromaCrawl", 0.35),
            ("chromaLoss", 0.25),
            ("flicker", 0.3),
            ("vhsSharpen", 0.35),
            ("interlace", 0.5),
            ("staticWarp", 0.35),
            ("fastForward", 0.0),
            ("tracking", 0.95),
            ("anisotropyWarp", 0.2),
            ("jitterX", 0.55),
            ("jitterY", 0.45),
            ("roll", 0.1),
            ("rollSpeed", 1.2),
            ("persistence", 0.3),
            ("rgbSplit", 0.4),
            ("tear", 0.55),
            ("humBar", 0.3),
            ("blueScreen", 0.1),
            ("skew", 0.55),
            ("crease", 0.4),
            ("ghost", 0.3),
            ("osd", 0.0),
            ("noiseEvo", 0.7),
            ("noiseOpacity", 0.6),
            ("blinds", 0.0),
            ("snowDots", 0.35),
            ("speedDots", 0.3),
            ("anisoSnow", 0.3),
        ],
    },
    ScalarPreset {
        id: "shot10",
        label: "Shot 10 · Aniso split",
        values: &[
            ("amount", 1.0),
            ("signalStrength", 0.45),
            ("colorStripes", 0.4),
            ("chromaCrawl", 0.4),
            ("chromaLoss", 0.45),
            ("flicker", 0.2),
            ("vhsSharpen", 0.35),
            ("interlace", 0.55),
            ("staticWarp", 0.25),
            ("fastForward", 0.0),
            ("tracking", 0.4),
            ("anisotropyWarp", 0.9),
            ("jitterX", 0.5),
            ("jitterY", 0.45),
            ("roll", 0.0),
            ("rollSpeed", 1.5),
            ("persistence", 0.35),
            ("rgbSplit", 0.35),
            ("tear", 0.4),
            ("humBar", 0.25),
            ("blueScreen", 0.1),
            ("skew", 0.4),
            ("crease", 0.6),
            ("ghost", 0.25),
            ("osd", 0.0),
            ("noiseEvo", 0.6),
            ("noiseOpacity", 0.5),
            ("blinds", 0.0),
            ("snowDots", 0.4),
            ("speedDots", 0.3),
            ("anisoSnow", 0.55),
        ],
    },
];

/// Computes deterministic VHS "Glitch!" burst values matching `vhs-setup.js:170-207`.
#[must_use]
pub fn compute_vhs_glitch_burst(current_seed: f64) -> (f64, Vec<(&'static str, f64)>) {
    let seed = next_seed(current_seed);
    let s = seed.round() as i32;
    let range = |lo: f64, hi: f64, at: i32| lo + legacy_hash01(s, at) * (hi - lo);
    let gate = |prob: f64, lo: f64, hi: f64, at: i32, gate_at: i32| {
        if legacy_hash01(s, gate_at) < prob {
            range(lo, hi, at)
        } else {
            0.0
        }
    };
    let values = vec![
        ("glitchSeed", seed),
        ("amount", 1.0),
        ("signalStrength", range(0.4, 1.0, 0)),
        ("colorStripes", range(0.3, 0.9, 1)),
        ("chromaCrawl", range(0.3, 0.9, 2)),
        ("chromaLoss", range(0.1, 0.6, 3)),
        ("flicker", range(0.1, 0.5, 4)),
        ("vhsSharpen", range(0.2, 0.7, 5)),
        ("interlace", range(0.3, 0.8, 6)),
        ("staticWarp", range(0.4, 1.0, 7)),
        ("fastForward", gate(0.35, 0.4, 1.0, 8, 27)),
        ("tracking", range(0.4, 1.0, 9)),
        ("anisotropyWarp", gate(0.5, 0.2, 0.8, 10, 28)),
        ("jitterX", range(0.3, 0.9, 11)),
        ("jitterY", range(0.3, 0.9, 12)),
        ("roll", gate(0.4, 0.2, 0.8, 13, 29)),
        ("rgbSplit", range(0.1, 0.6, 14)),
        ("tear", range(0.2, 0.8, 15)),
        ("humBar", range(0.1, 0.5, 16)),
        ("blueScreen", gate(0.12, 0.4, 1.0, 17, 30)),
        ("skew", range(0.2, 0.8, 18)),
        ("crease", gate(0.4, 0.2, 0.7, 19, 31)),
        ("ghost", range(0.1, 0.5, 20)),
        ("noiseEvo", range(0.3, 1.0, 21)),
        ("noiseOpacity", range(0.3, 0.8, 22)),
        ("blinds", gate(0.3, 0.2, 0.6, 23, 32)),
        ("snowDots", range(0.2, 0.7, 24)),
        ("speedDots", range(0.2, 0.7, 25)),
        ("anisoSnow", range(0.2, 0.6, 26)),
    ];
    (seed, values)
}

const VHS_FIELD_GUIDE: &[(&str, &str)] = &[
    (
        "Quality — Color",
        "Signal Strength (chroma dots/tears); Color Stripes (moving color bars); Chroma Crawl (edge shimmer); Chroma Loss (desat mono); Flicker (B/W exposure diffusion); RGB Split (red/blue split); Blue Screen (signal-loss field).",
    ),
    (
        "Quality — Image",
        "VHS Sharpen (directional blur + left sharpen); Interlace (offset + slats); Static Warp; Fast Forward (sine warp + white dots); Tracking (displaced clip + black bars); Anisotropy Warp; Tear; Hum Bar; Crease; Ghost; Blinds; Snow Dots; Speed Dots; Anisotropy Snow.",
    ),
    (
        "Motion",
        "Jitter X/Y (micro offset); Roll & Roll Speed (vertical roll/rewind); Persistence (interlace frame decay); Skew (top-edge flagwave bend).",
    ),
    ("Overlay", "OSD Timecode — camcorder PLAY + SP timecode."),
    (
        "Noise",
        "Noise Evolution (band noise scroll/boil speed); Noise Opacity (mix amount).",
    ),
];

// ===========================================================================
// Datamosh Presets, 81 Looks & Burst (`plugins/openzoid-legacy/datamosh-setup.js`)
// ===========================================================================

pub const DATAMOSH_MOTION_ITEMS: &[&str] = &[
    "Identity",
    "Scale",
    "Vertical",
    "Horizontal",
    "Zoom",
    "Shear",
    "Wave",
    "Swap",
    "Random",
];

pub const DATAMOSH_ALGORITHM_ITEMS: &[&str] = &[
    "Motion matching",
    "Default",
    "Multiply",
    "Add to x and y",
    "Move vertical",
    "Move horizontal",
    "Zoom 1",
    "Zoom 2",
    "Zoom sin",
    "Shear",
    "Experimental",
    "Swap",
    "Random",
    "Random Blocks 1",
    "Random Blocks 2",
    "Random Blocks 3",
    "Random Blocks 4",
    "Spatial",
    "Multiply by X",
    "Multiply by X no Y",
    "Multiply by Y",
    "Multiply by Y no X",
    "Multiply by inverse X",
    "Multiply by inverse X no Y",
    "Multiply by inverse Y",
    "Multiply by inverse Y no X",
    "Multiply by X&Y",
    "Multiply by inverse X&Y",
    "Multiply by X & inverse Y",
    "Multiply by Y & inverse X",
    "Average",
    "Average X and Y",
    "Average previous 3",
    "Average previous 5",
    "Average previous 10",
    "Average previous 15",
    "Average 1 neighbors",
    "Average 2 neighbors",
    "Average 3 neighbors",
    "Average 4 neighbors",
    "Add previous 3",
    "Add previous 5",
    "Add previous 10",
    "Mirror X",
    "Mirror Y",
    "Mirror left",
    "Mirror right",
    "Mirror top",
    "Mirror bottom",
    "Sweep",
    "Sweep horizontal",
    "Sweep horizontal opposite",
    "Sweep vertical",
    "Sweep vertical opposite",
    "Sin and cos",
    "Middle sin X",
    "Middle sin X no Y",
    "Middle sin Y",
    "Middle sin Y no X",
    "Outside sin X",
    "Outside sin X no Y",
    "Outside sin Y",
    "Outside sin Y no X",
    "Outside half sin X",
    "Outside half sin X no Y",
    "Outside half sin Y",
    "Outside half sin Y no X",
    "Oscillate sin X",
    "Oscillate sin X no Y",
    "Oscillate sin Y",
    "Oscillate sin Y no X",
    "Oscillate sin X&Y",
    "Right horizontal sin tapered V1",
    "Left horizontal sin tapered V1",
    "Horizontal sin tapered V1",
    "Right horizontal sin tapered V2",
    "Left horizontal sin tapered V2",
    "Horizontal sin tapered V2",
    "Horizontal sin tapered V3",
    "Cos X & sin Y",
    "Sin X & cos Y",
];

pub const DATAMOSH_PRESETS: &[ScalarPreset] = &[
    ScalarPreset {
        id: "clean",
        label: "Clean Pass",
        values: &[
            ("algorithm", 1.0),
            ("hold", 0.0),
            ("speed", 1.0),
            ("amount", 1.0),
            ("intensity", 0.25),
            ("acceleration", 0.2),
            ("blend", 0.5),
            ("threshold", 0.25),
            ("blockSize", 12.0),
            ("motion", 0.0),
            ("interval", 30.0),
            ("samples", 6.0),
        ],
    },
    ScalarPreset {
        id: "logo",
        label: "Logo Mosh · Blocky",
        values: &[
            ("algorithm", 13.0),
            ("hold", 0.2),
            ("speed", 1.0),
            ("amount", 1.0),
            ("intensity", 0.85),
            ("acceleration", 0.45),
            ("blend", 0.85),
            ("threshold", 0.08),
            ("blockSize", 14.0),
            ("motion", 8.0),
            ("interval", 30.0),
            ("samples", 6.0),
        ],
    },
    ScalarPreset {
        id: "classic",
        label: "Classic Mosh",
        values: &[
            ("algorithm", 12.0),
            ("hold", 0.15),
            ("speed", 1.0),
            ("amount", 1.0),
            ("intensity", 0.7),
            ("acceleration", 0.35),
            ("blend", 0.8),
            ("threshold", 0.12),
            ("blockSize", 12.0),
            ("motion", 8.0),
            ("interval", 30.0),
            ("samples", 6.0),
        ],
    },
    ScalarPreset {
        id: "iframe",
        label: "I-Frame Kill · Soupy",
        values: &[
            ("algorithm", 12.0),
            ("hold", 0.55),
            ("speed", 0.8),
            ("amount", 1.0),
            ("intensity", 1.2),
            ("acceleration", 0.7),
            ("blend", 0.95),
            ("threshold", 0.05),
            ("blockSize", 10.0),
            ("motion", 8.0),
            ("interval", 30.0),
            ("samples", 6.0),
        ],
    },
    ScalarPreset {
        id: "swap",
        label: "Swap Motion",
        values: &[
            ("algorithm", 11.0),
            ("hold", 0.2),
            ("speed", 1.2),
            ("amount", 1.0),
            ("intensity", 0.9),
            ("acceleration", 0.4),
            ("blend", 0.85),
            ("threshold", 0.1),
            ("blockSize", 12.0),
            ("motion", 7.0),
            ("interval", 30.0),
            ("samples", 6.0),
        ],
    },
    ScalarPreset {
        id: "sweep",
        label: "Sweep Horizontal",
        values: &[
            ("algorithm", 50.0),
            ("hold", 0.15),
            ("speed", 1.5),
            ("amount", 1.0),
            ("intensity", 1.0),
            ("acceleration", 0.3),
            ("blend", 0.85),
            ("threshold", 0.08),
            ("blockSize", 8.0),
            ("motion", 3.0),
            ("interval", 30.0),
            ("samples", 6.0),
        ],
    },
    ScalarPreset {
        id: "sinmelt",
        label: "Sin Melt",
        values: &[
            ("algorithm", 54.0),
            ("hold", 0.25),
            ("speed", 1.2),
            ("amount", 1.0),
            ("intensity", 1.1),
            ("acceleration", 0.5),
            ("blend", 0.9),
            ("threshold", 0.06),
            ("blockSize", 10.0),
            ("motion", 6.0),
            ("interval", 30.0),
            ("samples", 6.0),
        ],
    },
    ScalarPreset {
        id: "mirror",
        label: "Mirror Glitch",
        values: &[
            ("algorithm", 43.0),
            ("hold", 0.15),
            ("speed", 1.0),
            ("amount", 1.0),
            ("intensity", 0.8),
            ("acceleration", 0.3),
            ("blend", 0.8),
            ("threshold", 0.1),
            ("blockSize", 16.0),
            ("motion", 3.0),
            ("interval", 30.0),
            ("samples", 6.0),
        ],
    },
    ScalarPreset {
        id: "zoom",
        label: "Zoom Smear",
        values: &[
            ("algorithm", 6.0),
            ("hold", 0.2),
            ("speed", 1.0),
            ("amount", 1.0),
            ("intensity", 1.0),
            ("acceleration", 0.55),
            ("blend", 0.9),
            ("threshold", 0.08),
            ("blockSize", 12.0),
            ("motion", 4.0),
            ("interval", 30.0),
            ("samples", 6.0),
        ],
    },
    ScalarPreset {
        id: "trail",
        label: "Average Trail x10",
        values: &[
            ("algorithm", 34.0),
            ("hold", 0.1),
            ("speed", 1.0),
            ("amount", 1.0),
            ("intensity", 0.6),
            ("acceleration", 0.6),
            ("blend", 0.9),
            ("threshold", 0.1),
            ("blockSize", 12.0),
            ("motion", 0.0),
            ("interval", 30.0),
            ("samples", 6.0),
        ],
    },
];

/// Computes deterministic Datamosh "Mosh!" burst values matching `datamosh-setup.js:239-266`.
#[must_use]
pub fn compute_datamosh_burst(current_seed: f64) -> (f64, Vec<(&'static str, f64)>) {
    let seed = next_seed(current_seed);
    let s = seed.round() as i32;
    const BLOCK_PICKS: &[f64] = &[6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0];
    const LOOK_PICKS: &[f64] = &[
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 6.0, 9.0, 50.0, 54.0, 67.0, 71.0, 43.0, 34.0,
    ];
    let pick = |list: &[f64], at: i32| -> f64 {
        let idx = ((legacy_hash01(s, at) * list.len() as f64).floor() as usize) % list.len();
        list[idx]
    };
    let range = |lo: f64, hi: f64, at: i32| lo + legacy_hash01(s, at) * (hi - lo);
    let hold = if legacy_hash01(s, 7) < 0.5 {
        range(0.2, 0.6, 8)
    } else {
        range(0.0, 0.2, 9)
    };
    let motion = ((legacy_hash01(s, 6) * 9.0).floor() as usize % 9) as f64;
    let values = vec![
        ("seed", seed),
        ("amount", 1.0),
        ("intensity", range(0.6, 1.6, 0)),
        ("acceleration", range(0.2, 0.8, 1)),
        ("blend", range(0.6, 1.0, 2)),
        ("threshold", range(0.03, 0.2, 3)),
        ("blockSize", pick(BLOCK_PICKS, 4)),
        ("algorithm", pick(LOOK_PICKS, 5)),
        ("motion", motion),
        ("speed", range(0.6, 1.8, 10)),
        ("hold", hold),
    ];
    (seed, values)
}

const DATAMOSH_INFOS: &[(&str, &str)] = &[
    (
        "Remove Frames",
        "Hold = I-frame deletion. Frozen P-chain, sticky pixels.",
    ),
    (
        "Hijack Motion",
        "Look + Intensity + Block Size = your vectors.",
    ),
    (
        "Swap Motion",
        "Look 'Swap' steals axes. Use moving footage.",
    ),
    (
        "Mosh Maps",
        "Stack a Mask effect above Datamosh as the intensity map.",
    ),
    (
        "True Mosh Render",
        "Export > Device render > True datamosh: full video or at clip cuts.",
    ),
];

// ===========================================================================
// ASCII & Tracery Presets (`plugins/openzoid-legacy/effect-windows.js`)
// ===========================================================================

struct AsciiPreset {
    label: &'static str,
    numbers: &'static [(&'static str, f64)],
    background: [f64; 3],
    bands: [(&'static str, &'static str, [f64; 3]); 4],
}

const ASCII_PRESETS: &[AsciiPreset] = &[
    AsciiPreset {
        label: "Red Matrix",
        numbers: &[
            ("blockSize", 23.0),
            ("contrast", 1.0),
            ("brightness", 0.0),
            ("fontFamily", 0.0),
            ("charSize", 18.0),
            ("charset", 0.0),
            ("colorMode", 0.0),
            ("randomCharacters", 0.0),
            ("randomScale", 0.0),
            ("noiseIntensity", 0.0),
            ("sineEnable", 0.0),
            ("noiseEnable", 0.0),
            ("scrollEnable", 0.0),
            ("glitchEnable", 0.0),
            ("glitchAmount", 60.0),
            ("glitchSpeed", 3.0),
        ],
        background: [0.03, 0.03, 0.04],
        bands: [
            ("black", "@", [1.0, 0.16, 0.16]),
            ("white", "#", [1.0, 0.35, 0.35]),
            ("grey", "+", [0.75, 0.2, 0.2]),
            ("alpha", " ", [1.0, 1.0, 1.0]),
        ],
    },
    AsciiPreset {
        label: "Mono Paper",
        numbers: &[
            ("blockSize", 18.0),
            ("contrast", 1.1),
            ("brightness", 0.05),
            ("fontFamily", 1.0),
            ("charSize", 16.0),
            ("charset", 0.0),
            ("colorMode", 0.0),
            ("randomCharacters", 0.0),
            ("randomScale", 0.0),
            ("noiseIntensity", 0.0),
            ("sineEnable", 0.0),
            ("noiseEnable", 0.0),
            ("scrollEnable", 0.0),
            ("glitchEnable", 0.0),
            ("glitchAmount", 60.0),
            ("glitchSpeed", 3.0),
        ],
        background: [0.93, 0.93, 0.92],
        bands: [
            ("black", "@", [0.1, 0.1, 0.1]),
            ("white", ".", [0.4, 0.4, 0.4]),
            ("grey", "+", [0.25, 0.25, 0.25]),
            ("alpha", " ", [0.0, 0.0, 0.0]),
        ],
    },
    AsciiPreset {
        label: "Blueprint",
        numbers: &[
            ("blockSize", 26.0),
            ("contrast", 1.0),
            ("brightness", 0.0),
            ("fontFamily", 0.0),
            ("charSize", 22.0),
            ("charset", 0.0),
            ("colorMode", 0.0),
            ("randomCharacters", 0.0),
            ("randomScale", 40.0),
            ("noiseIntensity", 0.15),
            ("sineEnable", 1.0),
            ("sineRadial", 0.0),
            ("sineSpeed", 1.0),
            ("sineFrequency", 1.0),
            ("sineSize", 50.0),
            ("sineIntensity", 50.0),
            ("noiseEnable", 0.0),
            ("scrollEnable", 0.0),
            ("glitchEnable", 0.0),
            ("glitchAmount", 60.0),
            ("glitchSpeed", 3.0),
        ],
        background: [0.04, 0.08, 0.16],
        bands: [
            ("black", "@", [0.4, 0.7, 1.0]),
            ("white", "#", [0.8, 0.92, 1.0]),
            ("grey", "+", [0.55, 0.75, 1.0]),
            ("alpha", " ", [1.0, 1.0, 1.0]),
        ],
    },
    AsciiPreset {
        label: "Neon Night",
        numbers: &[
            ("blockSize", 20.0),
            ("contrast", 1.2),
            ("brightness", -0.05),
            ("fontFamily", 3.0),
            ("charSize", 18.0),
            ("charset", 0.0),
            ("colorMode", 0.0),
            ("randomCharacters", 1.0),
            ("randomScale", 120.0),
            ("noiseIntensity", 0.2),
            ("sineEnable", 1.0),
            ("sineRadial", 1.0),
            ("sineSpeed", 1.5),
            ("sineFrequency", 2.0),
            ("sineSize", 60.0),
            ("sineIntensity", 70.0),
            ("noiseEnable", 1.0),
            ("noiseType", 2.0),
            ("noiseScale", 1.5),
            ("noiseSpeed", 1.0),
            ("noiseOctaves", 3.0),
            ("noisePersistence", 0.9),
            ("scrollEnable", 1.0),
            ("scrollWaves", 2.0),
            ("scrollSpeed", 3.0),
            ("glitchEnable", 0.0),
            ("glitchAmount", 60.0),
            ("glitchSpeed", 3.0),
        ],
        background: [0.02, 0.02, 0.05],
        bands: [
            ("black", "@", [1.0, 0.2, 0.8]),
            ("white", "#", [0.0, 0.9, 1.0]),
            ("grey", "*", [0.7, 0.4, 1.0]),
            ("alpha", " ", [1.0, 1.0, 1.0]),
        ],
    },
    AsciiPreset {
        label: "Glitch Storm",
        numbers: &[
            ("blockSize", 18.0),
            ("contrast", 1.3),
            ("brightness", 0.0),
            ("fontFamily", 0.0),
            ("charSize", 16.0),
            ("charset", 2.0),
            ("colorMode", 1.0),
            ("randomCharacters", 0.0),
            ("randomScale", 60.0),
            ("noiseIntensity", 0.25),
            ("sineEnable", 0.0),
            ("noiseEnable", 1.0),
            ("noiseType", 3.0),
            ("noiseScale", 2.0),
            ("noiseSpeed", 2.0),
            ("noiseOctaves", 4.0),
            ("noisePersistence", 0.8),
            ("scrollEnable", 0.0),
            ("glitchEnable", 1.0),
            ("glitchAmount", 120.0),
            ("glitchSpeed", 4.0),
        ],
        background: [0.01, 0.01, 0.02],
        bands: [
            ("black", "@", [1.0, 1.0, 1.0]),
            ("white", "#", [1.0, 1.0, 1.0]),
            ("grey", "+", [1.0, 1.0, 1.0]),
            ("alpha", " ", [1.0, 1.0, 1.0]),
        ],
    },
];

struct TraceryPresetPoint {
    x: f64,
    y: f64,
    size: f64,
    dx: f64,
    dy: f64,
}

struct TraceryPreset {
    label: &'static str,
    clear_points: bool,
    numbers: &'static [(&'static str, f64)],
    colors: &'static [(&'static str, [f64; 3])],
    points: &'static [TraceryPresetPoint],
}

const TRACERY_PRESETS: &[TraceryPreset] = &[
    TraceryPreset {
        label: "Surveillance",
        clear_points: false,
        numbers: &[
            ("showBoxes", 1.0),
            ("boxShape", 0.0),
            ("cornerStyle", 0.0),
            ("boxWidth", 2.0),
            ("boxFill", 0.0),
            ("fillOpacity", 0.25),
            ("minSize", 24.0),
            ("markerShape", 0.0),
            ("markerSize", 14.0),
            ("markerFilled", 1.0),
            ("gridEnable", 0.0),
            ("showLines", 1.0),
            ("lineType", 0.0),
            ("lineWidth", 3.6),
            ("maxConnections", 2.0),
            ("splineTension", 0.0),
            ("dashEnable", 0.0),
            ("arrowEnable", 0.0),
            ("labelsEnable", 1.0),
        ],
        colors: &[
            ("lineColor", [1.0, 1.0, 1.0]),
            ("markerColor", [1.0, 0.15, 0.15]),
            ("boxColor", [1.0, 1.0, 1.0]),
        ],
        points: &[
            TraceryPresetPoint {
                x: 30.0,
                y: 35.0,
                size: 130.0,
                dx: 150.0,
                dy: -80.0,
            },
            TraceryPresetPoint {
                x: 70.0,
                y: 60.0,
                size: 110.0,
                dx: -260.0,
                dy: 70.0,
            },
        ],
    },
    TraceryPreset {
        label: "Blueprint",
        clear_points: false,
        numbers: &[
            ("showBoxes", 1.0),
            ("boxShape", 2.0),
            ("cornerStyle", 2.0),
            ("boxWidth", 1.5),
            ("boxFill", 0.0),
            ("fillOpacity", 0.2),
            ("minSize", 20.0),
            ("markerShape", 3.0),
            ("markerSize", 10.0),
            ("markerFilled", 0.0),
            ("gridEnable", 1.0),
            ("gridMode", 0.0),
            ("gridDivisions", 8.0),
            ("gridOpacity", 0.3),
            ("showLines", 1.0),
            ("lineType", 1.0),
            ("lineWidth", 2.0),
            ("maxConnections", 3.0),
            ("dashEnable", 0.0),
            ("arrowEnable", 0.0),
            ("labelsEnable", 1.0),
        ],
        colors: &[
            ("lineColor", [0.4, 0.7, 1.0]),
            ("markerColor", [0.4, 0.7, 1.0]),
            ("boxColor", [0.75, 0.85, 1.0]),
        ],
        points: &[
            TraceryPresetPoint {
                x: 35.0,
                y: 40.0,
                size: 150.0,
                dx: 170.0,
                dy: -90.0,
            },
            TraceryPresetPoint {
                x: 65.0,
                y: 55.0,
                size: 120.0,
                dx: -280.0,
                dy: 80.0,
            },
        ],
    },
    TraceryPreset {
        label: "Neon Callout",
        clear_points: false,
        numbers: &[
            ("showBoxes", 1.0),
            ("boxShape", 0.0),
            ("cornerStyle", 1.0),
            ("boxWidth", 3.0),
            ("boxFill", 1.0),
            ("fillOpacity", 0.2),
            ("minSize", 24.0),
            ("markerShape", 0.0),
            ("markerSize", 18.0),
            ("markerFilled", 1.0),
            ("gridEnable", 0.0),
            ("showLines", 1.0),
            ("lineType", 0.0),
            ("lineWidth", 4.0),
            ("maxConnections", 2.0),
            ("splineTension", 0.3),
            ("dashEnable", 1.0),
            ("dashLength", 12.0),
            ("dashGap", 6.0),
            ("arrowEnable", 1.0),
            ("arrowSize", 23.0),
            ("arrowPosition", 0.7),
            ("labelsEnable", 1.0),
        ],
        colors: &[
            ("lineColor", [1.0, 0.2, 0.8]),
            ("markerColor", [1.0, 0.2, 0.8]),
            ("boxColor", [1.0, 0.4, 0.9]),
        ],
        points: &[TraceryPresetPoint {
            x: 40.0,
            y: 45.0,
            size: 140.0,
            dx: 180.0,
            dy: -100.0,
        }],
    },
    TraceryPreset {
        label: "Minimal",
        clear_points: false,
        numbers: &[
            ("showBoxes", 1.0),
            ("boxShape", 0.0),
            ("cornerStyle", 1.0),
            ("boxWidth", 1.5),
            ("boxFill", 0.0),
            ("fillOpacity", 0.15),
            ("minSize", 16.0),
            ("markerShape", 0.0),
            ("markerSize", 8.0),
            ("markerFilled", 1.0),
            ("gridEnable", 0.0),
            ("showLines", 0.0),
            ("dashEnable", 0.0),
            ("arrowEnable", 0.0),
            ("labelsEnable", 1.0),
        ],
        colors: &[
            ("lineColor", [1.0, 1.0, 1.0]),
            ("markerColor", [1.0, 1.0, 1.0]),
            ("boxColor", [1.0, 1.0, 1.0]),
        ],
        points: &[TraceryPresetPoint {
            x: 50.0,
            y: 50.0,
            size: 120.0,
            dx: 160.0,
            dy: -80.0,
        }],
    },
    TraceryPreset {
        label: "Key Track",
        clear_points: true,
        numbers: &[
            ("detectEnable", 1.0),
            ("threshold", 50.0),
            ("showMask", 0.0),
            ("blurStrength", 5.0),
            ("minSize", 20.0),
            ("showBoxes", 1.0),
            ("boxShape", 0.0),
            ("cornerStyle", 0.0),
            ("boxWidth", 2.0),
            ("boxFill", 0.0),
            ("fillOpacity", 0.25),
            ("markerShape", 0.0),
            ("markerSize", 12.0),
            ("markerFilled", 1.0),
            ("gridEnable", 0.0),
            ("showLines", 1.0),
            ("lineType", 0.0),
            ("lineWidth", 2.5),
            ("maxConnections", 2.0),
            ("splineTension", 0.0),
            ("dashEnable", 1.0),
            ("dashLength", 8.0),
            ("dashGap", 5.0),
            ("arrowEnable", 0.0),
            ("labelsEnable", 1.0),
        ],
        colors: &[
            ("lineColor", [1.0, 1.0, 1.0]),
            ("markerColor", [1.0, 0.15, 0.15]),
            ("boxColor", [1.0, 1.0, 1.0]),
        ],
        points: &[],
    },
];

// ===========================================================================
// Session Property Helpers
// ===========================================================================

fn find_effect_in_session<'a>(
    session: &'a Session,
    effect_id: &str,
) -> Option<(&'a SessionEffect, f64, bool)> {
    for track in session
        .document()
        .video_tracks
        .iter()
        .chain(session.document().audio_tracks.iter())
    {
        for clip in &track.clips {
            let local_frame = session.playback.current_frame - clip.start;
            let is_adjustment = clip.kind == "adjustment";
            for fx in &clip.effects {
                if fx.id == effect_id {
                    return Some((fx, local_frame, is_adjustment));
                }
            }
        }
    }
    None
}

fn read_prop_value(prop: Option<&SessionProperty>, local_frame: f64) -> Option<Value> {
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

fn read_f64(fx: &SessionEffect, key: &str, default: f64, local_frame: f64) -> f64 {
    read_prop_value(fx.properties.get(key), local_frame)
        .and_then(|v| {
            v.as_f64()
                .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
        })
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

fn read_color(fx: &SessionEffect, key: &str, default: [f64; 3], local_frame: f64) -> [f64; 3] {
    read_prop_value(fx.properties.get(key), local_frame)
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

fn read_str(fx: &SessionEffect, key: &str, default: &str, local_frame: f64) -> String {
    read_prop_value(fx.properties.get(key), local_frame)
        .and_then(|v| v.as_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| default.to_owned())
}

fn write_effect_prop(
    session: &mut Session,
    effect_id: &str,
    key: &str,
    value: Value,
    local_frame: f64,
) {
    let animated = find_effect_in_session(session, effect_id)
        .and_then(|(fx, _, _)| fx.properties.get(key))
        .is_some_and(|p| p.animated);
    if animated {
        let _ = session.execute(Command::SetKeyframe {
            target_id: effect_id.to_owned(),
            property: key.to_owned(),
            frame: local_frame,
            value,
            tween: None,
        });
    } else {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: effect_id.to_owned(),
            property: key.to_owned(),
            value,
        });
    }
}

/// Applies a batch of `(key, Value)` pairs to `effect_id` inside one atomic undo transaction.
pub fn apply_effect_batch_tx(
    session: &mut Session,
    effect_id: &str,
    label: &str,
    assignments: &[(String, Value)],
    local_frame: f64,
) {
    if assignments.is_empty() {
        return;
    }
    let opened = session.begin_transaction(label.to_owned()).is_ok();
    for (key, val) in assignments {
        write_effect_prop(session, effect_id, key, val.clone(), local_frame);
    }
    if opened && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    }
}

/// Returns the matching preset ID for `presets` (tolerance `0.015`), or `"custom"`.
#[must_use]
pub fn match_scalar_preset(
    fx: &SessionEffect,
    presets: &[ScalarPreset],
    local_frame: f64,
) -> &'static str {
    for preset in presets {
        let matched = preset.values.iter().all(|&(key, expected)| {
            fx.properties.contains_key(key)
                && same_value(read_f64(fx, key, expected, local_frame), expected)
        });
        if matched {
            return preset.id;
        }
    }
    "custom"
}

// ===========================================================================
// Reusable Compact Setup Row Widgets
// ===========================================================================

fn setup_note_banner(ui: &mut Ui, text: &str, warning: bool, theme: &Cm3Theme) {
    let (bg, border, fg) = if warning {
        (WARNING_SURFACE, WARNING_BAR, TEXT_PRIMARY)
    } else {
        (BG_SUNKEN, BORDER_PANEL, TEXT_MUTED)
    };
    egui::Frame::NONE
        .fill(bg)
        .stroke(Stroke::new(1.0, border))
        .corner_radius(CornerRadius::same(3))
        .inner_margin(egui::Margin::symmetric(8, 5))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(
                egui::RichText::new(text)
                    .font(theme.font(FONT_SMALL))
                    .color(fg),
            );
        });
}

#[allow(clippy::too_many_arguments)]
fn setup_scalar_row(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut EffectSetupState,
    fx: &SessionEffect,
    key: &str,
    label: &str,
    default: f64,
    cfg: ScrubConfig,
    hint: Option<&str>,
    local_frame: f64,
    theme: &Cm3Theme,
) {
    let mut val = read_f64(fx, key, default, local_frame);
    let tx_key = format!("setup:{}:{key}", fx.id);
    if state.active_tx_key.as_deref() == Some(tx_key.as_str()) {
        state.tx_seen_this_frame = true;
    }

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let lbl_w = (ui.available_width() * 0.48).clamp(110.0, 210.0);
        let lbl_resp = ui.add_sized(
            Vec2::new(lbl_w, 20.0),
            egui::Label::new(
                egui::RichText::new(label)
                    .font(theme.font(FONT_SMALL))
                    .color(TEXT_PRIMARY),
            )
            .truncate(),
        );
        if let Some(h) = hint {
            lbl_resp.on_hover_text(h);
        }
        let ev = cm3_scrub_number(
            ui,
            Id::unique(("fx_setup_num", &fx.id, key, label)),
            &mut val,
            cfg,
            theme,
        );
        if ev.cancelled {
            if state.active_tx_key.as_deref() == Some(tx_key.as_str()) {
                if session.history.is_in_transaction() {
                    let _ = session.cancel_transaction();
                }
                state.active_tx_key = None;
            }
            return;
        }
        if ev.started && !ev.committed {
            if session.history.is_in_transaction() {
                let _ = session.commit_transaction(true);
            }
            if session.begin_transaction(format!("edit {label}")).is_ok() {
                state.active_tx_key = Some(tx_key.clone());
                state.tx_seen_this_frame = true;
            }
        }
        if ev.changed {
            write_effect_prop(session, &fx.id, key, json!(val), local_frame);
        }
        if ev.committed && state.active_tx_key.as_deref() == Some(tx_key.as_str()) {
            if session.history.is_in_transaction() {
                let _ = session.commit_transaction(true);
            }
            state.active_tx_key = None;
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn setup_option_row(
    ui: &mut Ui,
    session: &mut Session,
    fx: &SessionEffect,
    key: &str,
    label: &str,
    options: &[&str],
    default_idx: usize,
    local_frame: f64,
    theme: &Cm3Theme,
) {
    let cur = (read_f64(fx, key, default_idx as f64, local_frame).round() as usize)
        .min(options.len().saturating_sub(1));
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let lbl_w = (ui.available_width() * 0.48).clamp(110.0, 210.0);
        ui.add_sized(
            Vec2::new(lbl_w, 20.0),
            egui::Label::new(
                egui::RichText::new(label)
                    .font(theme.font(FONT_SMALL))
                    .color(TEXT_PRIMARY),
            )
            .truncate(),
        );
        if options.len() == 2 && options[0].eq_ignore_ascii_case("off") {
            let mut checked = cur != 0;
            if ui.checkbox(&mut checked, "").changed() {
                write_effect_prop(
                    session,
                    &fx.id,
                    key,
                    json!(if checked { 1 } else { 0 }),
                    local_frame,
                );
            }
        } else {
            let combo_w = ui.available_width().max(110.0);
            egui::ComboBox::from_id_salt(Id::unique(("fx_setup_opt", &fx.id, key)))
                .selected_text(options.get(cur).copied().unwrap_or("—"))
                .width(combo_w)
                .show_ui(ui, |ui| {
                    for (idx, opt) in options.iter().enumerate() {
                        if ui.selectable_label(idx == cur, *opt).clicked() && idx != cur {
                            write_effect_prop(session, &fx.id, key, json!(idx), local_frame);
                        }
                    }
                });
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn setup_color_row(
    ui: &mut Ui,
    session: &mut Session,
    fx: &SessionEffect,
    key: &str,
    label: &str,
    default: [f64; 3],
    local_frame: f64,
    viewport_rect: Rect,
    theme: &Cm3Theme,
) {
    let rgb = read_color(fx, key, default, local_frame);
    let mut color = Rgba::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, 1.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let lbl_w = (ui.available_width() * 0.48).clamp(110.0, 210.0);
        ui.add_sized(
            Vec2::new(lbl_w, 20.0),
            egui::Label::new(
                egui::RichText::new(label)
                    .font(theme.font(FONT_SMALL))
                    .color(TEXT_PRIMARY),
            )
            .truncate(),
        );
        let picker_id = Id::unique(("fx_setup_col", &fx.id, key, label));
        let ev = cm3_hsva_color_picker(
            ui,
            picker_id,
            &mut color,
            false,
            false,
            viewport_rect,
            theme,
        );
        if ev.eyedropper_clicked {
            request_eyedropper_sampler(ui.ctx(), picker_id);
        }
        if ev.changed {
            write_effect_prop(
                session,
                &fx.id,
                key,
                json!([
                    f64::from(color.0[0]),
                    f64::from(color.0[1]),
                    f64::from(color.0[2])
                ]),
                local_frame,
            );
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn setup_text_row(
    ui: &mut Ui,
    session: &mut Session,
    fx: &SessionEffect,
    key: &str,
    label: &str,
    default: &str,
    local_frame: f64,
    theme: &Cm3Theme,
) {
    let mut text = read_str(fx, key, default, local_frame);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let lbl_w = (ui.available_width() * 0.48).clamp(110.0, 210.0);
        ui.add_sized(
            Vec2::new(lbl_w, 20.0),
            egui::Label::new(
                egui::RichText::new(label)
                    .font(theme.font(FONT_SMALL))
                    .color(TEXT_PRIMARY),
            )
            .truncate(),
        );
        egui::Frame::NONE
            .fill(BG_INPUT)
            .stroke(Stroke::new(1.0, BORDER_CONTROL))
            .inner_margin(egui::Margin::symmetric(4, 2))
            .show(ui, |ui| {
                let resp = ui.add(
                    TextEdit::singleline(&mut text)
                        .frame(egui::Frame::NONE)
                        .font(theme.font(FONT_SMALL))
                        .text_color(theme.value())
                        .desired_width(ui.available_width().max(80.0)),
                );
                if resp.changed() {
                    write_effect_prop(session, &fx.id, key, json!(text), local_frame);
                }
            });
    });
}

// ===========================================================================
// Per-Effect Setup Bodies
// ===========================================================================

#[allow(clippy::too_many_arguments)]
fn show_vhs_setup_body(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut EffectSetupState,
    fx: &SessionEffect,
    local_frame: f64,
    is_adjustment: bool,
    theme: &Cm3Theme,
) {
    if !is_adjustment {
        setup_note_banner(
            ui,
            "Persistence samples earlier clip frames, or the composite below an Adjustment layer.",
            true,
            theme,
        );
        ui.add_space(6.0);
    }

    // Preset selector
    let active_preset = match_scalar_preset(fx, VHS_PRESETS, local_frame);
    let active_label = VHS_PRESETS
        .iter()
        .find(|p| p.id == active_preset)
        .map_or("Custom", |p| p.label);
    let mut chosen_preset: Option<&ScalarPreset> = None;
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("Preset")
                .font(theme.font(FONT_BODY))
                .color(TEXT_STRONG),
        );
        egui::ComboBox::from_id_salt(Id::unique(("vhs_setup_preset", &fx.id)))
            .selected_text(active_label)
            .width((ui.available_width() - 8.0).max(150.0))
            .show_ui(ui, |ui| {
                let _ = ui.selectable_label(active_preset == "custom", "Custom");
                for preset in VHS_PRESETS {
                    if ui
                        .selectable_label(active_preset == preset.id, preset.label)
                        .clicked()
                    {
                        chosen_preset = Some(preset);
                    }
                }
            });
    });
    if let Some(preset) = chosen_preset {
        let batch: Vec<(String, Value)> = preset
            .values
            .iter()
            .map(|&(k, v)| (k.to_owned(), json!(v)))
            .collect();
        apply_effect_batch_tx(
            session,
            &fx.id,
            &format!("VHS preset: {}", preset.label),
            &batch,
            local_frame,
        );
    }

    ui.add_space(6.0);
    let unit_cfg = ScrubConfig::new(0.0, 1.0, 0.01, 2);

    // Master
    cm3_collapsible_section(
        ui,
        Id::unique(("vhs_sec_master", &fx.id)),
        "Master",
        true,
        theme,
        |ui| {
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "amount",
                "Amount",
                1.0,
                unit_cfg,
                None,
                local_frame,
                theme,
            );
        },
    );

    // Color
    cm3_collapsible_section(
        ui,
        Id::unique(("vhs_sec_color", &fx.id)),
        "Color",
        true,
        theme,
        |ui| {
            for &(key, label, def) in &[
                ("signalStrength", "Signal Strength", 0.25),
                ("colorStripes", "Color Stripes", 0.25),
                ("chromaCrawl", "Chroma Crawl", 0.2),
                ("chromaLoss", "Chroma Loss", 0.05),
                ("flicker", "Flicker", 0.08),
                ("rgbSplit", "RGB Split", 0.12),
                ("blueScreen", "Blue Screen", 0.0),
            ] {
                setup_scalar_row(
                    ui,
                    session,
                    state,
                    fx,
                    key,
                    label,
                    def,
                    unit_cfg,
                    None,
                    local_frame,
                    theme,
                );
            }
        },
    );

    // Image
    cm3_collapsible_section(
        ui,
        Id::unique(("vhs_sec_image", &fx.id)),
        "Image",
        true,
        theme,
        |ui| {
            for &(key, label, def) in &[
                ("vhsSharpen", "VHS Sharpen", 0.35),
                ("interlace", "Interlace", 0.4),
                ("staticWarp", "Static Warp", 0.2),
                ("fastForward", "Fast Forward", 0.0),
                ("tracking", "Tracking", 0.25),
                ("anisotropyWarp", "Anisotropy Warp", 0.0),
                ("tear", "Tear", 0.12),
                ("humBar", "Hum Bar", 0.1),
                ("crease", "Crease", 0.0),
                ("ghost", "Ghost", 0.08),
                ("blinds", "Blinds", 0.0),
                ("snowDots", "Snow Dots", 0.0),
                ("speedDots", "Speed Dots", 0.0),
                ("anisoSnow", "Anisotropy Snow", 0.0),
            ] {
                setup_scalar_row(
                    ui,
                    session,
                    state,
                    fx,
                    key,
                    label,
                    def,
                    unit_cfg,
                    None,
                    local_frame,
                    theme,
                );
            }
        },
    );

    // Motion
    cm3_collapsible_section(
        ui,
        Id::unique(("vhs_sec_motion", &fx.id)),
        "Motion",
        true,
        theme,
        |ui| {
            for &(key, label, def, cfg, hint) in &[
                ("jitterX", "Jitter X", 0.3, unit_cfg, None),
                ("jitterY", "Jitter Y", 0.3, unit_cfg, None),
                ("roll", "Roll", 0.0, unit_cfg, None),
                (
                    "rollSpeed",
                    "Roll Speed",
                    1.5,
                    ScrubConfig::new(0.0, 10.0, 0.05, 2),
                    None,
                ),
                (
                    "persistence",
                    "Persistence",
                    0.25,
                    unit_cfg,
                    Some("Averages earlier source frames. Needs an Adjustment layer."),
                ),
                ("skew", "Skew", 0.25, unit_cfg, None),
            ] {
                setup_scalar_row(
                    ui,
                    session,
                    state,
                    fx,
                    key,
                    label,
                    def,
                    cfg,
                    hint,
                    local_frame,
                    theme,
                );
            }
        },
    );

    // Noise (collapsed by default)
    cm3_collapsible_section(
        ui,
        Id::unique(("vhs_sec_noise", &fx.id)),
        "Noise",
        false,
        theme,
        |ui| {
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "noiseEvo",
                "Noise Evolution",
                0.5,
                ScrubConfig::new(0.0, 5.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "noiseOpacity",
                "Noise Opacity",
                0.4,
                unit_cfg,
                None,
                local_frame,
                theme,
            );
        },
    );

    // Overlay (collapsed by default)
    cm3_collapsible_section(
        ui,
        Id::unique(("vhs_sec_overlay", &fx.id)),
        "Overlay",
        false,
        theme,
        |ui| {
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "osd",
                "OSD Timecode",
                0.0,
                unit_cfg,
                None,
                local_frame,
                theme,
            );
        },
    );

    ui.add_space(6.0);
    // Clean VHS & Glitch! action row
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        if cm3_prop_button(ui, "Clean VHS", ButtonVariant::Normal, false, theme)
            .on_hover_text("Reset to the clean preset")
            .clicked()
        {
            let batch: Vec<(String, Value)> = VHS_PRESETS[0]
                .values
                .iter()
                .map(|&(k, v)| (k.to_owned(), json!(v)))
                .collect();
            apply_effect_batch_tx(session, &fx.id, "VHS: Clean preset", &batch, local_frame);
        }
        if cm3_prop_button(ui, "Glitch!", ButtonVariant::Primary, false, theme)
            .on_hover_text("Seeded damaged-tape burst (stores a new seed)")
            .clicked()
        {
            let cur_seed = read_f64(fx, "glitchSeed", 0.0, local_frame);
            let (_, burst) = compute_vhs_glitch_burst(cur_seed);
            let batch: Vec<(String, Value)> = burst
                .into_iter()
                .map(|(k, v)| (k.to_owned(), json!(v)))
                .collect();
            apply_effect_batch_tx(session, &fx.id, "VHS: Glitch burst", &batch, local_frame);
        }
    });

    ui.add_space(4.0);
    // Field Guide (collapsed by default)
    cm3_collapsible_section(
        ui,
        Id::unique(("vhs_sec_guide", &fx.id)),
        "Field guide",
        false,
        theme,
        |ui| {
            for &(title, desc) in VHS_FIELD_GUIDE {
                ui.label(
                    egui::RichText::new(format!("{title} — {desc}"))
                        .font(theme.font(FONT_COMPACT))
                        .color(TEXT_MUTED),
                );
                ui.add_space(3.0);
            }
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn show_datamosh_setup_body(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut EffectSetupState,
    fx: &SessionEffect,
    local_frame: f64,
    is_adjustment: bool,
    theme: &Cm3Theme,
) {
    if !is_adjustment {
        setup_note_banner(
            ui,
            "Datamosh reads earlier clip frames. On an Adjustment layer it samples the composite below it.",
            true,
            theme,
        );
        ui.add_space(6.0);
    }

    let active_preset = match_scalar_preset(fx, DATAMOSH_PRESETS, local_frame);
    let active_label = DATAMOSH_PRESETS
        .iter()
        .find(|p| p.id == active_preset)
        .map_or("Custom", |p| p.label);
    let mut chosen_preset: Option<&ScalarPreset> = None;
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("Preset")
                .font(theme.font(FONT_BODY))
                .color(TEXT_STRONG),
        );
        egui::ComboBox::from_id_salt(Id::unique(("dm_setup_preset", &fx.id)))
            .selected_text(active_label)
            .width((ui.available_width() - 8.0).max(160.0))
            .show_ui(ui, |ui| {
                let _ = ui.selectable_label(active_preset == "custom", "Custom");
                for preset in DATAMOSH_PRESETS {
                    if ui
                        .selectable_label(active_preset == preset.id, preset.label)
                        .clicked()
                    {
                        chosen_preset = Some(preset);
                    }
                }
            });
    });
    if let Some(preset) = chosen_preset {
        let batch: Vec<(String, Value)> = preset
            .values
            .iter()
            .map(|&(k, v)| (k.to_owned(), json!(v)))
            .collect();
        apply_effect_batch_tx(
            session,
            &fx.id,
            &format!("Datamosh preset: {}", preset.label),
            &batch,
            local_frame,
        );
    }

    ui.add_space(6.0);
    // Master + action buttons
    cm3_collapsible_section(
        ui,
        Id::unique(("dm_sec_master", &fx.id)),
        "Master",
        true,
        theme,
        |ui| {
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "amount",
                "Amount",
                1.0,
                ScrubConfig::new(0.0, 1.0, 0.01, 2),
                None,
                local_frame,
                theme,
            );
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                if cm3_prop_button(ui, "Clean Pass", ButtonVariant::Normal, false, theme)
                    .on_hover_text("Reset to the gentle preset")
                    .clicked()
                {
                    let batch: Vec<(String, Value)> = DATAMOSH_PRESETS[0]
                        .values
                        .iter()
                        .map(|&(k, v)| (k.to_owned(), json!(v)))
                        .collect();
                    apply_effect_batch_tx(
                        session,
                        &fx.id,
                        "Datamosh: Clean Pass",
                        &batch,
                        local_frame,
                    );
                }
                if cm3_prop_button(ui, "Mosh!", ButtonVariant::Primary, false, theme)
                    .on_hover_text("Seeded soupy mosh burst (stores a new seed)")
                    .clicked()
                {
                    let cur_seed = read_f64(fx, "seed", 0.0, local_frame);
                    let (_, burst) = compute_datamosh_burst(cur_seed);
                    let batch: Vec<(String, Value)> = burst
                        .into_iter()
                        .map(|(k, v)| (k.to_owned(), json!(v)))
                        .collect();
                    apply_effect_batch_tx(
                        session,
                        &fx.id,
                        "Datamosh: Mosh burst",
                        &batch,
                        local_frame,
                    );
                }
                if cm3_prop_button(ui, "Remove Frame", ButtonVariant::Normal, false, theme)
                    .on_hover_text("Seeded I-frame kill: hold to full")
                    .clicked()
                {
                    let cur_seed = read_f64(fx, "seed", 0.0, local_frame);
                    let new_seed = next_seed(cur_seed);
                    let cur_intensity = read_f64(fx, "intensity", 0.6, local_frame);
                    let batch = vec![
                        ("seed".to_owned(), json!(new_seed)),
                        ("hold".to_owned(), json!(1.0)),
                        ("intensity".to_owned(), json!(cur_intensity.max(1.0))),
                    ];
                    apply_effect_batch_tx(
                        session,
                        &fx.id,
                        "Datamosh: Remove Frame",
                        &batch,
                        local_frame,
                    );
                }
            });
        },
    );

    // Cadence
    cm3_collapsible_section(
        ui,
        Id::unique(("dm_sec_cadence", &fx.id)),
        "Cadence",
        true,
        theme,
        |ui| {
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "interval",
                "Keyframe Interval",
                30.0,
                ScrubConfig::new(1.0, 300.0, 1.0, 0),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "samples",
                "Motion Samples",
                6.0,
                ScrubConfig::new(1.0, 15.0, 1.0, 0),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "seed",
                "Seed",
                0.0,
                ScrubConfig::new(0.0, 10_000.0, 1.0, 0),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "hold",
                "Remove Frames (Hold)",
                0.15,
                ScrubConfig::new(0.0, 1.0, 0.01, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "speed",
                "Speed",
                1.0,
                ScrubConfig::new(0.0, 10.0, 0.05, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "time",
                "Time Offset",
                0.0,
                ScrubConfig::new(-1000.0, 1000.0, 0.05, 2),
                None,
                local_frame,
                theme,
            );
            ui.add_space(4.0);
            if cm3_prop_button(ui, "New Seed", ButtonVariant::Normal, false, theme)
                .on_hover_text("Store a new seed for the block pattern")
                .clicked()
            {
                let cur_seed = read_f64(fx, "seed", 0.0, local_frame);
                let next = next_seed(cur_seed);
                write_effect_prop(session, &fx.id, "seed", json!(next), local_frame);
            }
        },
    );

    // Motion
    cm3_collapsible_section(
        ui,
        Id::unique(("dm_sec_motion", &fx.id)),
        "Motion",
        true,
        theme,
        |ui| {
            setup_option_row(
                ui,
                session,
                fx,
                "algorithm",
                "Look Algorithm",
                DATAMOSH_ALGORITHM_ITEMS,
                12,
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "motion",
                "Motion Family",
                DATAMOSH_MOTION_ITEMS,
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "intensity",
                "Intensity",
                0.6,
                ScrubConfig::new(0.0, 5.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "acceleration",
                "Acceleration",
                0.35,
                ScrubConfig::new(0.0, 2.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "blockSize",
                "Block Size",
                12.0,
                ScrubConfig::new(2.0, 128.0, 1.0, 0),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "threshold",
                "Threshold",
                0.12,
                ScrubConfig::new(0.0, 1.0, 0.01, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "blend",
                "Blend",
                0.8,
                ScrubConfig::new(0.0, 1.0, 0.01, 2),
                None,
                local_frame,
                theme,
            );
        },
    );

    // How to mosh guide
    cm3_collapsible_section(
        ui,
        Id::unique(("dm_sec_guide", &fx.id)),
        "How to mosh",
        false,
        theme,
        |ui| {
            for &(name, hint) in DATAMOSH_INFOS {
                ui.label(
                    egui::RichText::new(format!("{name} — {hint}"))
                        .font(theme.font(FONT_COMPACT))
                        .color(TEXT_MUTED),
                );
                ui.add_space(3.0);
            }
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn show_ascii_setup_body(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut EffectSetupState,
    fx: &SessionEffect,
    local_frame: f64,
    viewport_rect: Rect,
    theme: &Cm3Theme,
) {
    // Presets list
    cm3_collapsible_section(
        ui,
        Id::unique(("ascii_sec_presets", &fx.id)),
        "Presets",
        true,
        theme,
        |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(6.0, 4.0);
                for preset in ASCII_PRESETS {
                    if cm3_prop_button(ui, preset.label, ButtonVariant::Normal, false, theme)
                        .clicked()
                    {
                        let mut batch: Vec<(String, Value)> = preset
                            .numbers
                            .iter()
                            .map(|&(k, v)| (k.to_owned(), json!(v)))
                            .collect();
                        batch.push(("backgroundColor".to_owned(), json!(preset.background)));
                        for &(band, fill, col) in &preset.bands {
                            batch.push((format!("{band}Fill"), json!(fill)));
                            batch.push((format!("{band}Color"), json!(col)));
                        }
                        apply_effect_batch_tx(
                            session,
                            &fx.id,
                            &format!("ASCII preset: {}", preset.label),
                            &batch,
                            local_frame,
                        );
                    }
                }
            });
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("ascii_sec_grid", &fx.id)),
        "Grid",
        true,
        theme,
        |ui| {
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "blockSize",
                "Block Size",
                23.0,
                ScrubConfig::new(4.0, 128.0, 1.0, 0),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "charSize",
                "Char Size",
                18.0,
                ScrubConfig::new(4.0, 128.0, 1.0, 0),
                None,
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "fontFamily",
                "Font Family",
                &["Monospace", "Sans", "Serif", "Condensed", "Pixel"],
                0,
                local_frame,
                theme,
            );
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("ascii_sec_image", &fx.id)),
        "Image",
        true,
        theme,
        |ui| {
            setup_color_row(
                ui,
                session,
                fx,
                "backgroundColor",
                "Background Color",
                [0.03, 0.03, 0.04],
                local_frame,
                viewport_rect,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "contrast",
                "Contrast",
                1.0,
                ScrubConfig::new(0.0, 4.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "brightness",
                "Brightness",
                0.0,
                ScrubConfig::new(-1.0, 1.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("ascii_sec_chars", &fx.id)),
        "Characters",
        true,
        theme,
        |ui| {
            setup_option_row(
                ui,
                session,
                fx,
                "charset",
                "Charset",
                &["Standard", "Dense", "Blocks", "Binary", "Symbols"],
                0,
                local_frame,
                theme,
            );
            setup_text_row(
                ui,
                session,
                fx,
                "customChars",
                "Custom Chars",
                "@%#*+=-:. ",
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "randomCharacters",
                "Random Characters",
                &["off", "on"],
                0,
                local_frame,
                theme,
            );
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("ascii_sec_bands", &fx.id)),
        "Bands",
        false,
        theme,
        |ui| {
            for &(band, title, def_fill, def_col) in &[
                ("black", "Black", "@", [1.0, 0.16, 0.16]),
                ("white", "White", "#", [1.0, 0.35, 0.35]),
                ("grey", "Grey", "+", [0.75, 0.2, 0.2]),
                ("alpha", "Alpha", " ", [1.0, 1.0, 1.0]),
            ] {
                setup_text_row(
                    ui,
                    session,
                    fx,
                    &format!("{band}Fill"),
                    &format!("{title} Fill"),
                    def_fill,
                    local_frame,
                    theme,
                );
                setup_text_row(
                    ui,
                    session,
                    fx,
                    &format!("{band}Stroke"),
                    &format!("{title} Stroke"),
                    "",
                    local_frame,
                    theme,
                );
                setup_color_row(
                    ui,
                    session,
                    fx,
                    &format!("{band}Color"),
                    &format!("{title} Color"),
                    def_col,
                    local_frame,
                    viewport_rect,
                    theme,
                );
                setup_scalar_row(
                    ui,
                    session,
                    state,
                    fx,
                    &format!("{band}Opacity"),
                    &format!("{title} Opacity"),
                    1.0,
                    ScrubConfig::new(0.0, 1.0, 0.02, 2),
                    None,
                    local_frame,
                    theme,
                );
            }
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("ascii_sec_colormode", &fx.id)),
        "Color mode",
        true,
        theme,
        |ui| {
            setup_option_row(
                ui,
                session,
                fx,
                "colorMode",
                "Color Mode",
                &["Band", "Footage", "Duotone", "Mono", "Invert"],
                0,
                local_frame,
                theme,
            );
            setup_color_row(
                ui,
                session,
                fx,
                "duotoneDark",
                "Duotone Dark",
                [0.05, 0.05, 0.08],
                local_frame,
                viewport_rect,
                theme,
            );
            setup_color_row(
                ui,
                session,
                fx,
                "duotoneLight",
                "Duotone Light",
                [1.0, 0.2, 0.3],
                local_frame,
                viewport_rect,
                theme,
            );
            setup_color_row(
                ui,
                session,
                fx,
                "monoColor",
                "Mono Color",
                [1.0, 1.0, 1.0],
                local_frame,
                viewport_rect,
                theme,
            );
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("ascii_sec_random", &fx.id)),
        "Random",
        false,
        theme,
        |ui| {
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "randomScale",
                "Random Scale",
                0.0,
                ScrubConfig::new(0.0, 500.0, 1.0, 1),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "noiseIntensity",
                "Noise Intensity",
                0.0,
                ScrubConfig::new(0.0, 1.0, 0.01, 2),
                None,
                local_frame,
                theme,
            );
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("ascii_sec_sine", &fx.id)),
        "Sine waves",
        false,
        theme,
        |ui| {
            setup_option_row(
                ui,
                session,
                fx,
                "sineEnable",
                "Enable",
                &["off", "on"],
                0,
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "sineRadial",
                "Radial",
                &["off", "on"],
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "sineSpeed",
                "Speed",
                1.0,
                ScrubConfig::new(0.0, 20.0, 0.1, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "sineFrequency",
                "Frequency",
                1.0,
                ScrubConfig::new(0.0, 20.0, 0.1, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "sineSize",
                "Size",
                50.0,
                ScrubConfig::new(0.0, 200.0, 1.0, 1),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "sineIntensity",
                "Intensity",
                50.0,
                ScrubConfig::new(0.0, 200.0, 1.0, 1),
                None,
                local_frame,
                theme,
            );
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("ascii_sec_advnoise", &fx.id)),
        "Advanced noise",
        false,
        theme,
        |ui| {
            setup_option_row(
                ui,
                session,
                fx,
                "noiseEnable",
                "Enable",
                &["off", "on"],
                0,
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "noiseType",
                "Noise Type",
                &["Value", "Perlin", "Simplex", "Cellular"],
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "noiseScale",
                "Scale",
                1.0,
                ScrubConfig::new(0.1, 20.0, 0.1, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "noiseSpeed",
                "Speed",
                1.0,
                ScrubConfig::new(0.0, 20.0, 0.1, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "noiseOctaves",
                "Octaves",
                3.0,
                ScrubConfig::new(1.0, 8.0, 1.0, 0),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "noisePersistence",
                "Persistence",
                0.9,
                ScrubConfig::new(0.0, 1.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("ascii_sec_scroll", &fx.id)),
        "Scroll",
        false,
        theme,
        |ui| {
            setup_option_row(
                ui,
                session,
                fx,
                "scrollEnable",
                "Enable",
                &["off", "on"],
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "scrollWaves",
                "Waves",
                1.0,
                ScrubConfig::new(0.0, 20.0, 0.1, 1),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "scrollSpeed",
                "Speed",
                3.0,
                ScrubConfig::new(-20.0, 20.0, 0.1, 1),
                None,
                local_frame,
                theme,
            );
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("ascii_sec_glitch", &fx.id)),
        "Glitch",
        false,
        theme,
        |ui| {
            setup_option_row(
                ui,
                session,
                fx,
                "glitchEnable",
                "Enable",
                &["off", "on"],
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "glitchAmount",
                "Amount",
                60.0,
                ScrubConfig::new(0.0, 300.0, 1.0, 1),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "glitchSpeed",
                "Speed",
                3.0,
                ScrubConfig::new(0.0, 30.0, 0.1, 1),
                None,
                local_frame,
                theme,
            );
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn show_tracery_setup_body(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut EffectSetupState,
    fx: &SessionEffect,
    local_frame: f64,
    viewport_rect: Rect,
    theme: &Cm3Theme,
) {
    cm3_collapsible_section(
        ui,
        Id::unique(("tracery_sec_presets", &fx.id)),
        "Presets",
        true,
        theme,
        |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(6.0, 4.0);
                for preset in TRACERY_PRESETS {
                    if cm3_prop_button(ui, preset.label, ButtonVariant::Normal, false, theme)
                        .clicked()
                    {
                        let mut batch: Vec<(String, Value)> = preset
                            .numbers
                            .iter()
                            .map(|&(k, v)| (k.to_owned(), json!(v)))
                            .collect();
                        for &(ck, rgb) in preset.colors {
                            batch.push((ck.to_owned(), json!(rgb)));
                        }
                        if preset.clear_points {
                            for n in 1..=6 {
                                batch.push((format!("point{n}Enable"), json!(0)));
                            }
                        }
                        for (idx, pt) in preset.points.iter().enumerate() {
                            let n = idx + 1;
                            batch.push((format!("point{n}Enable"), json!(1)));
                            batch.push((format!("point{n}X"), json!(pt.x)));
                            batch.push((format!("point{n}Y"), json!(pt.y)));
                            batch.push((format!("point{n}Size"), json!(pt.size)));
                            batch.push((format!("point{n}LabelDX"), json!(pt.dx)));
                            batch.push((format!("point{n}LabelDY"), json!(pt.dy)));
                        }
                        apply_effect_batch_tx(
                            session,
                            &fx.id,
                            &format!("Tracery preset: {}", preset.label),
                            &batch,
                            local_frame,
                        );
                    }
                }
            });
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("tracery_sec_detect", &fx.id)),
        "Detection",
        true,
        theme,
        |ui| {
            setup_option_row(
                ui,
                session,
                fx,
                "detectEnable",
                "Detect Enable",
                &["off", "on"],
                1,
                local_frame,
                theme,
            );
            setup_color_row(
                ui,
                session,
                fx,
                "keyColor",
                "Key Color",
                [1.0, 1.0, 1.0],
                local_frame,
                viewport_rect,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "threshold",
                "Threshold",
                50.0,
                ScrubConfig::new(0.0, 100.0, 1.0, 1),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "soften",
                "Mask Soften",
                0.08,
                ScrubConfig::new(0.0, 1.0, 0.01, 2),
                Some("Key mask edge feathering"),
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "blurStrength",
                "Blur Strength",
                5.0,
                ScrubConfig::new(0.0, 64.0, 0.5, 1),
                Some("Pre-detection separable Gaussian blur radius (px)"),
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "minSize",
                "Min Box Size",
                16.0,
                ScrubConfig::new(4.0, 512.0, 1.0, 0),
                Some("Minimum tracked feature bounding box size (px)"),
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "showMask",
                "Show Mask",
                &["off", "on"],
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "opacity",
                "Master Opacity",
                1.0,
                ScrubConfig::new(0.0, 1.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("tracery_sec_boxes", &fx.id)),
        "Boxes",
        true,
        theme,
        |ui| {
            setup_option_row(
                ui,
                session,
                fx,
                "showBoxes",
                "Show Boxes",
                &["off", "on"],
                1,
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "boxShape",
                "Box Shape",
                &["Rectangle", "Square", "Ellipse", "Circle"],
                0,
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "cornerStyle",
                "Corner Style",
                &["Bracketed", "Corners Only", "Solid Frame", "Rounded"],
                0,
                local_frame,
                theme,
            );
            setup_color_row(
                ui,
                session,
                fx,
                "boxColor",
                "Box Color",
                [0.2, 0.9, 1.0],
                local_frame,
                viewport_rect,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "boxOpacity",
                "Box Opacity",
                1.0,
                ScrubConfig::new(0.0, 1.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "boxWidth",
                "Box Width",
                2.0,
                ScrubConfig::new(0.5, 24.0, 0.25, 2),
                None,
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "boxFill",
                "Box Fill",
                &["None", "Solid", "Diagonal Hatch", "Invert"],
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "fillOpacity",
                "Fill Opacity",
                0.25,
                ScrubConfig::new(0.0, 1.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("tracery_sec_markers", &fx.id)),
        "Markers",
        true,
        theme,
        |ui| {
            setup_option_row(
                ui,
                session,
                fx,
                "markerShape",
                "Marker Shape",
                &["Dot", "Plus", "Cross", "Polygon"],
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "markerSize",
                "Marker Size",
                10.0,
                ScrubConfig::new(1.0, 128.0, 0.5, 1),
                None,
                local_frame,
                theme,
            );
            setup_color_row(
                ui,
                session,
                fx,
                "markerColor",
                "Marker Color",
                [1.0, 0.15, 0.15],
                local_frame,
                viewport_rect,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "markerOpacity",
                "Marker Opacity",
                1.0,
                ScrubConfig::new(0.0, 1.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "markerRotation",
                "Marker Rotation",
                0.0,
                ScrubConfig::new(-360.0, 360.0, 1.0, 1),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "markerSides",
                "Marker Sides",
                4.0,
                ScrubConfig::new(3.0, 12.0, 1.0, 0),
                None,
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "markerFilled",
                "Marker Filled",
                &["off", "on"],
                1,
                local_frame,
                theme,
            );
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("tracery_sec_grid", &fx.id)),
        "Grid",
        false,
        theme,
        |ui| {
            setup_option_row(
                ui,
                session,
                fx,
                "gridEnable",
                "Grid Enable",
                &["off", "on"],
                0,
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "gridMode",
                "Grid Mode",
                &["Cartesian", "Edge"],
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "gridDivisions",
                "Divisions",
                12.0,
                ScrubConfig::new(2.0, 64.0, 1.0, 0),
                None,
                local_frame,
                theme,
            );
            setup_color_row(
                ui,
                session,
                fx,
                "boxColor",
                "Grid / Box Color",
                [0.2, 0.9, 1.0],
                local_frame,
                viewport_rect,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "gridOpacity",
                "Grid Opacity",
                0.35,
                ScrubConfig::new(0.0, 1.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("tracery_sec_conn", &fx.id)),
        "Connections",
        false,
        theme,
        |ui| {
            setup_option_row(
                ui,
                session,
                fx,
                "showLines",
                "Show Lines",
                &["off", "on"],
                1,
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "lineType",
                "Line Type",
                &["Spline", "PCB Traces", "Step Bend", "Straight"],
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "maxConnections",
                "Max Connections",
                2.0,
                ScrubConfig::new(1.0, 4.0, 1.0, 0),
                Some("Maximum nearest-neighbor connectors per node (1-4)"),
                local_frame,
                theme,
            );
            setup_color_row(
                ui,
                session,
                fx,
                "lineColor",
                "Line Color",
                [0.2, 0.9, 1.0],
                local_frame,
                viewport_rect,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "lineOpacity",
                "Line Opacity",
                0.9,
                ScrubConfig::new(0.0, 1.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "lineWidth",
                "Line Width",
                2.0,
                ScrubConfig::new(0.5, 24.0, 0.2, 2),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "splineTension",
                "Spline Tension",
                0.0,
                ScrubConfig::new(-1.0, 1.0, 0.05, 2),
                None,
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "dashEnable",
                "Dash Enable",
                &["off", "on"],
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "dashLength",
                "Dash Length",
                8.0,
                ScrubConfig::new(1.0, 100.0, 1.0, 1),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "dashGap",
                "Dash Gap",
                6.0,
                ScrubConfig::new(1.0, 100.0, 1.0, 1),
                None,
                local_frame,
                theme,
            );
            setup_option_row(
                ui,
                session,
                fx,
                "arrowEnable",
                "Arrow Enable",
                &["off", "on"],
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "arrowSize",
                "Arrow Size",
                10.0,
                ScrubConfig::new(2.0, 100.0, 1.0, 1),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "arrowPosition",
                "Arrow Position",
                0.5,
                ScrubConfig::new(0.0, 1.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
        },
    );

    cm3_collapsible_section(
        ui,
        Id::unique(("tracery_sec_labels", &fx.id)),
        "Labels",
        false,
        theme,
        |ui| {
            setup_option_row(
                ui,
                session,
                fx,
                "labelsEnable",
                "Labels Enable",
                &["off", "on"],
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "labelOffsetX",
                "Label Offset X",
                45.0,
                ScrubConfig::new(-500.0, 500.0, 2.0, 1),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "labelOffsetY",
                "Label Offset Y",
                35.0,
                ScrubConfig::new(-500.0, 500.0, 2.0, 1),
                None,
                local_frame,
                theme,
            );
            setup_color_row(
                ui,
                session,
                fx,
                "lineColor",
                "Callout / Line Color",
                [0.2, 0.9, 1.0],
                local_frame,
                viewport_rect,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                "lineOpacity",
                "Callout Opacity",
                0.9,
                ScrubConfig::new(0.0, 1.0, 0.02, 2),
                None,
                local_frame,
                theme,
            );
        },
    );

    // Points P1..P6 with quick enable toggles + tabbed detail editor
    cm3_collapsible_section(
        ui,
        Id::unique(("tracery_sec_points", &fx.id)),
        "Points (P1–P6)",
        true,
        theme,
        |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = Vec2::new(4.0, 4.0);
                for n in 1..=6usize {
                    let is_active = state.tracery_point_tab + 1 == n;
                    let enabled = read_f64(fx, &format!("point{n}Enable"), 0.0, local_frame) >= 0.5;
                    let title = if enabled {
                        format!("P{n} ●")
                    } else {
                        format!("P{n}")
                    };
                    let variant = if is_active {
                        ButtonVariant::Primary
                    } else {
                        ButtonVariant::Normal
                    };
                    if cm3_prop_button(ui, &title, variant, false, theme).clicked() {
                        state.tracery_point_tab = n - 1;
                    }
                }
            });
            ui.add_space(4.0);
            let n = state.tracery_point_tab.min(5) + 1;
            setup_option_row(
                ui,
                session,
                fx,
                &format!("point{n}Enable"),
                &format!("Point {n} Enable"),
                &["off", "on"],
                0,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                &format!("point{n}X"),
                "X (%)",
                50.0,
                ScrubConfig::new(0.0, 100.0, 0.5, 1),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                &format!("point{n}Y"),
                "Y (%)",
                50.0,
                ScrubConfig::new(0.0, 100.0, 0.5, 1),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                &format!("point{n}Size"),
                "Box Size",
                120.0,
                ScrubConfig::new(4.0, 1000.0, 2.0, 1),
                None,
                local_frame,
                theme,
            );
            setup_text_row(
                ui,
                session,
                fx,
                &format!("point{n}Label"),
                "Label",
                &format!("P{n}"),
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                &format!("point{n}LabelDX"),
                "Label Offset X",
                140.0,
                ScrubConfig::new(-1000.0, 1000.0, 2.0, 1),
                None,
                local_frame,
                theme,
            );
            setup_scalar_row(
                ui,
                session,
                state,
                fx,
                &format!("point{n}LabelDY"),
                "Label Offset Y",
                -80.0,
                ScrubConfig::new(-1000.0, 1000.0, 2.0, 1),
                None,
                local_frame,
                theme,
            );
        },
    );
}

// ===========================================================================
// Main Floating Window Renderer
// ===========================================================================

/// Renders the active Effect Setup floating window (`VHS`, `Datamosh`, `ASCII`, or `Tracery`).
pub fn show_effect_setup_window(
    ctx: &egui::Context,
    viewport_rect: Rect,
    floating_windows: &mut FloatingWindowManager,
    state: &mut EffectSetupState,
    session: &mut Session,
    theme: &Cm3Theme,
) {
    if !state.open {
        return;
    }

    let Some((fx_snapshot, local_frame, is_adjustment)) =
        find_effect_in_session(session, &state.effect_id)
            .map(|(fx, lf, adj)| (fx.clone(), lf, adj))
    else {
        state.open = false;
        floating_windows.close(state.kind.window_id());
        return;
    };

    let ctx_screen = ctx.input(|i| i.viewport_rect());
    let screen = if ctx_screen.width() > 100.0 && ctx_screen.height() > 100.0 {
        ctx_screen
    } else {
        viewport_rect
    };
    let pref = state.kind.default_size();
    let (_, default_size) = compute_clamped_editor_window_rect(screen, pref.x, pref.y);
    let win_id = state.kind.window_id();

    let subtitle = if state.effect_label.is_empty() {
        state.kind.subtitle().to_owned()
    } else {
        format!("{} — {}", state.kind.subtitle(), state.effect_label)
    };
    let spec = FloatingWindowSpec::new(win_id, state.kind.title())
        .with_subtitle(subtitle)
        .with_default_size(default_size);

    ensure_pro_window_in_screen(floating_windows, &spec, screen, &mut state.just_opened);

    state.tx_seen_this_frame = false;
    let mut close_requested = false;

    let ev = cm3_floating_window(ctx, floating_windows, &spec, screen, theme, |ui| {
        let body_h = (ui.available_height() - 34.0).max(120.0);
        egui::Frame::NONE
            .fill(BG_PANEL)
            .stroke(Stroke::new(1.0, BORDER_PANEL))
            .corner_radius(CornerRadius::same(3))
            .inner_margin(egui::Margin::same(6))
            .show(ui, |ui| {
                ui.set_min_height(body_h);
                ui.set_max_height(body_h);
                ScrollArea::vertical()
                    .id_salt(("fx_setup_scroll", win_id, &fx_snapshot.id))
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 3.0;
                        match state.kind {
                            EffectSetupKind::Vhs => show_vhs_setup_body(
                                ui,
                                session,
                                state,
                                &fx_snapshot,
                                local_frame,
                                is_adjustment,
                                theme,
                            ),
                            EffectSetupKind::Datamosh => show_datamosh_setup_body(
                                ui,
                                session,
                                state,
                                &fx_snapshot,
                                local_frame,
                                is_adjustment,
                                theme,
                            ),
                            EffectSetupKind::Ascii => show_ascii_setup_body(
                                ui,
                                session,
                                state,
                                &fx_snapshot,
                                local_frame,
                                screen,
                                theme,
                            ),
                            EffectSetupKind::Tracery => show_tracery_setup_body(
                                ui,
                                session,
                                state,
                                &fx_snapshot,
                                local_frame,
                                screen,
                                theme,
                            ),
                        }
                    });
            });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Live viewport preview • Atomic undo")
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

    if state.active_tx_key.is_some() && !state.tx_seen_this_frame {
        if session.history.is_in_transaction() {
            let _ = session.commit_transaction(true);
        }
        state.active_tx_key = None;
    }

    clamp_floating_window_in_screen(floating_windows, &spec, screen);

    if ev.closed && ctx.input(|i| i.key_pressed(Key::Escape)) {
        mark_floating_editor_escape_consumed(ctx);
    } else if !ev.closed && consume_escape_for_floating_editor(ctx, floating_windows, win_id) {
        close_requested = true;
    }

    if close_requested || ev.closed {
        if state.active_tx_key.take().is_some() && session.history.is_in_transaction() {
            let _ = session.commit_transaction(true);
        }
        state.open = false;
        floating_windows.close(win_id);
    }
}
