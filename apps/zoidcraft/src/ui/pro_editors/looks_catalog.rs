//! Catalog of all 29 Color Looks tools, plus multi-point curve and `.cube` LUT helpers.

use serde_json::{Map, Value, json};
use zc_session::Session;

use super::*;

// ---------------------------------------------------------------------------
// All 29 Color Looks Tool Definitions (Matching zc-render/src/effects/mod.rs)
// ---------------------------------------------------------------------------

pub const LUT_MODES: &[&str] = &["None", "Hot", "Cold", "Noir", "Custom .cube"];
pub const LUT_GAMMA_MODES: &[&str] = &["Same", "Input (2.2)", "Output (1/2.2)"];

/// Complete 29-tool Color Looks catalog across all 5 execution stages.
pub const LOOKS_TOOL_CATALOG: &[LooksToolSpec] = &[
    // ==================== SUBJECT (6) ====================
    LooksToolSpec {
        tool_id: "color-contrast",
        label: "Color Contrast",
        stage: LooksStage::Subject,
        params: &[
            LooksParamSpec {
                key: "pivot",
                label: "Pivot",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "color",
                label: "Tint Color",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "hsl-colors",
        label: "HSL Colors",
        stage: LooksStage::Subject,
        params: &[
            LooksParamSpec {
                key: "redSat",
                label: "Red Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "orangeSat",
                label: "Orange Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "yellowSat",
                label: "Yellow Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "greenSat",
                label: "Green Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "cyanSat",
                label: "Cyan Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "blueSat",
                label: "Blue Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "purpleSat",
                label: "Purple Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "magentaSat",
                label: "Magenta Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "redLight",
                label: "Red Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "orangeLight",
                label: "Orange Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "yellowLight",
                label: "Yellow Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "greenLight",
                label: "Green Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "cyanLight",
                label: "Cyan Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "blueLight",
                label: "Blue Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "purpleLight",
                label: "Purple Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "magentaLight",
                label: "Magenta Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "color-ranges",
        label: "Color Ranges",
        stage: LooksStage::Subject,
        params: &[
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "highlight",
                label: "Highlight Thresh",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.7,
            },
            LooksParamSpec {
                key: "midtone",
                label: "Midtone Center",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "shadow",
                label: "Shadow Thresh",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "showThreshold",
                label: "Show Threshold",
                kind: LooksParamKind::Bool,
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "highlightColor",
                label: "Highlight Tint",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "midtoneColor",
                label: "Midtone Tint",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "shadowColor",
                label: "Shadow Tint",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "crush",
        label: "Shadow Crush",
        stage: LooksStage::Subject,
        params: &[
            LooksParamSpec {
                key: "gamma",
                label: "Gamma",
                kind: LooksParamKind::Float {
                    min: 0.2,
                    max: 3.0,
                    step: 0.02,
                },
                default_f64: 1.15,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "color",
                label: "Crush Color",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "contrast",
        label: "Contrast",
        stage: LooksStage::Subject,
        params: &[
            LooksParamSpec {
                key: "contrast",
                label: "Contrast",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 2.0,
                    step: 0.01,
                },
                default_f64: 0.2,
            },
            LooksParamSpec {
                key: "pivot",
                label: "Pivot",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "pop",
        label: "Micro Contrast",
        stage: LooksStage::Subject,
        params: &[
            LooksParamSpec {
                key: "pop",
                label: "Amount",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "size",
                label: "Radius",
                kind: LooksParamKind::Float {
                    min: 0.05,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "preserveDetail",
                label: "Preserve Detail",
                kind: LooksParamKind::Bool,
                default_f64: 1.0,
            },
        ],
    },
    // ==================== MATTE (2) ====================
    LooksToolSpec {
        tool_id: "vignette",
        label: "Vignette",
        stage: LooksStage::Matte,
        params: &[
            LooksParamSpec {
                key: "centerX",
                label: "Center X",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "centerY",
                label: "Center Y",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "radius",
                label: "Radius",
                kind: LooksParamKind::Float {
                    min: 0.05,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.7,
            },
            LooksParamSpec {
                key: "aspect",
                label: "Aspect",
                kind: LooksParamKind::Float {
                    min: 0.25,
                    max: 3.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "spread",
                label: "Spread",
                kind: LooksParamKind::Float {
                    min: 0.01,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.4,
            },
            LooksParamSpec {
                key: "falloff",
                label: "Falloff",
                kind: LooksParamKind::Float {
                    min: 0.2,
                    max: 4.0,
                    step: 0.05,
                },
                default_f64: 1.5,
            },
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.4,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -3.0,
                    max: 1.0,
                    step: 0.02,
                },
                default_f64: -0.6,
            },
            LooksParamSpec {
                key: "color",
                label: "Color",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "edge-softness",
        label: "Edge Softness",
        stage: LooksStage::Matte,
        params: &[
            LooksParamSpec {
                key: "blurSize",
                label: "Blur Size",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.25,
            },
            LooksParamSpec {
                key: "quality",
                label: "Quality",
                kind: LooksParamKind::Float {
                    min: 1.0,
                    max: 4.0,
                    step: 1.0,
                },
                default_f64: 2.0,
            },
            LooksParamSpec {
                key: "centerX",
                label: "Center X",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "centerY",
                label: "Center Y",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "radius",
                label: "Radius",
                kind: LooksParamKind::Float {
                    min: 0.1,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.65,
            },
            LooksParamSpec {
                key: "aspect",
                label: "Aspect",
                kind: LooksParamKind::Float {
                    min: 0.25,
                    max: 3.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "spread",
                label: "Spread",
                kind: LooksParamKind::Float {
                    min: 0.01,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
        ],
    },
    // ==================== LENS (8) ====================
    LooksToolSpec {
        tool_id: "diffusion",
        label: "Lens Diffusion",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "size",
                label: "Size",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
            LooksParamSpec {
                key: "grade",
                label: "Grade",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "glow",
                label: "Glow",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 2.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "highlightsOnly",
                label: "Highlights Only",
                kind: LooksParamKind::Bool,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "highlightBias",
                label: "Highlight Bias",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "color",
                label: "Tint",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "star-filter",
        label: "Star Filter",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "size",
                label: "Size",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
            LooksParamSpec {
                key: "boost",
                label: "Boost",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.4,
            },
            LooksParamSpec {
                key: "threshold",
                label: "Threshold",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.75,
            },
            LooksParamSpec {
                key: "showThreshold",
                label: "Show Threshold",
                kind: LooksParamKind::Bool,
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "angle",
                label: "Angle",
                kind: LooksParamKind::Float {
                    min: -180.0,
                    max: 180.0,
                    step: 1.0,
                },
                default_f64: 15.0,
            },
            LooksParamSpec {
                key: "color",
                label: "Tint",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "anamorphic-flare",
        label: "Anamorphic Flare",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "size",
                label: "Streak Size",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.45,
            },
            LooksParamSpec {
                key: "boost",
                label: "Boost",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.35,
            },
            LooksParamSpec {
                key: "threshold",
                label: "Threshold",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.75,
            },
            LooksParamSpec {
                key: "showThreshold",
                label: "Show Threshold",
                kind: LooksParamKind::Bool,
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "reflection",
                label: "Reflection",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.2,
            },
            LooksParamSpec {
                key: "reflBoost",
                label: "Refl Boost",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.25,
            },
            LooksParamSpec {
                key: "color",
                label: "Streak Color",
                kind: LooksParamKind::ColorRgb,
                default_f64: 0.85,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "haze-flare",
        label: "Haze Flare",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "spillage",
                label: "Spillage",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "softness",
                label: "Softness",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "reach",
                label: "Reach",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.4,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.2,
            },
            LooksParamSpec {
                key: "reflExposure",
                label: "Refl Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "reflection",
                label: "Reflection",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.2,
            },
            LooksParamSpec {
                key: "matteSize",
                label: "Matte Size",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "matteShade",
                label: "Matte Shade",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.2,
            },
            LooksParamSpec {
                key: "color",
                label: "Haze Color",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "lens-distortion",
        label: "Lens Distortion",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "distortion",
                label: "Distortion",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: -0.1,
            },
            LooksParamSpec {
                key: "flatness",
                label: "Flatness",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "chromatic-aberration",
        label: "Chromatic Aberration",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "redCyan",
                label: "Red / Cyan",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.2,
            },
            LooksParamSpec {
                key: "greenMagenta",
                label: "Green / Magenta",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "blueYellow",
                label: "Blue / Yellow",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: -0.15,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "shutter-streak",
        label: "Shutter Streak",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "size",
                label: "Length",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "boost",
                label: "Boost",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.4,
            },
            LooksParamSpec {
                key: "falloff",
                label: "Falloff",
                kind: LooksParamKind::Float {
                    min: 0.1,
                    max: 4.0,
                    step: 0.05,
                },
                default_f64: 1.5,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "telecine-net",
        label: "Soft Diffusion Net",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "size",
                label: "Mesh Size",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
        ],
    },
    // ==================== CAMERA (6) ====================
    LooksToolSpec {
        tool_id: "warm-cool",
        label: "Warm / Cool",
        stage: LooksStage::Camera,
        params: &[
            LooksParamSpec {
                key: "warmCool",
                label: "Temperature",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.15,
            },
            LooksParamSpec {
                key: "tint",
                label: "Tint",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "lift-gamma-gain",
        label: "Lift / Gamma / Gain",
        stage: LooksStage::Camera,
        params: &[
            LooksParamSpec {
                key: "gammaSpace",
                label: "Gamma Space",
                kind: LooksParamKind::Float {
                    min: 1.0,
                    max: 3.0,
                    step: 0.05,
                },
                default_f64: 2.2,
            },
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "lift",
                label: "Lift (Shadows)",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "gamma",
                label: "Gamma (Mids)",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "gain",
                label: "Gain (Highs)",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "four-way",
        label: "4-Way Color",
        stage: LooksStage::Camera,
        params: &[
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "contrast",
                label: "Contrast",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "preview",
                label: "Preview Ranges",
                kind: LooksParamKind::Bool,
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "shadows",
                label: "Shadows",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "midtones",
                label: "Midtones",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "highlights",
                label: "Highlights",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "global",
                label: "Global",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "lightflex",
        label: "Shadow Lift Fog",
        stage: LooksStage::Camera,
        params: &[
            LooksParamSpec {
                key: "boost",
                label: "Boost",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.25,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "color",
                label: "Veil Color",
                kind: LooksParamKind::ColorRgb,
                default_f64: 0.9,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "deflare",
        label: "Deflare",
        stage: LooksStage::Camera,
        params: &[
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "auto-shoulder",
        label: "Highlight Shoulder",
        stage: LooksStage::Camera,
        params: &[LooksParamSpec {
            key: "strength",
            label: "Strength",
            kind: LooksParamKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default_f64: 0.5,
        }],
    },
    // ==================== POST (7) ====================
    LooksToolSpec {
        tool_id: "mojo",
        label: "Cinema Balance",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "mojo",
                label: "Balance",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "punch",
                label: "Punch",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.25,
            },
            LooksParamSpec {
                key: "bleach",
                label: "Bleach",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "fade",
                label: "Fade",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "blueSqueeze",
                label: "Blue Squeeze",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "skinSqueeze",
                label: "Warm Protect",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.4,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "coolWarm",
                label: "Cool / Warm",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "greenMagenta",
                label: "Green / Magenta",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "skinYP",
                label: "Skin Hue",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "color-reversal",
        label: "Color Reversal",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "three-strip",
        label: "3-Strip Film",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.6,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "ranged-saturation",
        label: "Ranged Saturation",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "satHighlight",
                label: "Highlight Sat",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 2.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "satMidtone",
                label: "Midtone Sat",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 2.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "satShadow",
                label: "Shadow Sat",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 2.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "thHighlight",
                label: "High Thresh",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.7,
            },
            LooksParamSpec {
                key: "thMidtone",
                label: "Mid Center",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "thShadow",
                label: "Shadow Thresh",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "showThreshold",
                label: "Show Threshold",
                kind: LooksParamKind::Bool,
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "balance",
                label: "Balance",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "curves",
        label: "Tone Curves",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "curvCurvesJson",
                label: "Tone Curves",
                kind: LooksParamKind::ToneCurves {
                    multi_channel: true,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "shadowLift",
                label: "Shadow Lift",
                kind: LooksParamKind::Float {
                    min: -0.5,
                    max: 0.5,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "midGamma",
                label: "Mid Gamma",
                kind: LooksParamKind::Float {
                    min: 0.3,
                    max: 2.5,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "highlightGain",
                label: "Highlight Gain",
                kind: LooksParamKind::Float {
                    min: 0.5,
                    max: 1.8,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "s-curve",
        label: "S-Curve Contrast",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "scvScurveJson",
                label: "S-Curve",
                kind: LooksParamKind::ToneCurves {
                    multi_channel: false,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "strength",
                label: "S-Curve Strength",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
            LooksParamSpec {
                key: "pivot",
                label: "Pivot",
                kind: LooksParamKind::Float {
                    min: 0.1,
                    max: 0.9,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "lut",
        label: "Look LUT",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "mode",
                label: "LUT Profile",
                kind: LooksParamKind::Choice(LUT_MODES),
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "lutAsset",
                label: "Custom .cube LUT",
                kind: LooksParamKind::CubeLutAsset,
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.75,
            },
            LooksParamSpec {
                key: "gamma",
                label: "Gamma",
                kind: LooksParamKind::Choice(LUT_GAMMA_MODES),
                default_f64: 0.0,
            },
        ],
    },
];

// ---------------------------------------------------------------------------
// Multi-Point Tone Curves & Custom `.cube` LUT Helpers
// ---------------------------------------------------------------------------

pub const CURVE_TONE_LABELS: [&str; 5] = ["Shadow", "Low-Mid", "Midtone", "High-Mid", "Highlight"];
pub const CURVE_TONE_DEFAULT_X: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

/// Returns the default 5-point identity curve (Shadow / Low-Mid / Midtone / High-Mid / Highlight).
#[must_use]
pub fn default_5pt_curve() -> Vec<[f64; 2]> {
    CURVE_TONE_DEFAULT_X.iter().map(|&x| [x, x]).collect()
}

/// Evaluates the legacy parametric curves sliders (`shadowLift`, `midGamma`, `highlightGain`)
/// into 5 editable control points for seamless backwards compatibility.
#[must_use]
pub fn synthesize_legacy_curves_5pt(
    shadow_lift: f64,
    mid_gamma: f64,
    highlight_gain: f64,
) -> Vec<[f64; 2]> {
    let inv_gamma = 1.0 / mid_gamma.max(0.05);
    let gain = highlight_gain.max(0.0);
    CURVE_TONE_DEFAULT_X
        .iter()
        .map(|&x| {
            let lifted = (x + shadow_lift * (1.0 - x)).clamp(0.0, 1.0);
            let y = (lifted.powf(inv_gamma) * gain).clamp(0.0, 1.0);
            [x, y]
        })
        .collect()
}

/// Evaluates the parametric S-Curve (`strength`, `pivot`) into 5 editable control points.
#[must_use]
pub fn synthesize_scurve_5pt(strength: f64, pivot: f64) -> Vec<[f64; 2]> {
    let st = strength.clamp(-1.0, 1.0);
    let pv = pivot.clamp(0.05, 0.95);
    CURVE_TONE_DEFAULT_X
        .iter()
        .map(|&x| {
            let y = (x + st
                * 0.35
                * ((x - pv) * std::f64::consts::PI * 2.0).sin()
                * (x * (1.0 - x) * 4.0))
                .clamp(0.0, 1.0);
            [x, y]
        })
        .collect()
}

fn parse_curve_point_array(val: &Value) -> Vec<[f64; 2]> {
    let mut pts = Vec::new();
    if let Some(arr) = val.as_array() {
        for item in arr {
            if let Some(pair) = item.as_array()
                && pair.len() >= 2
                && let (Some(x), Some(y)) = (pair[0].as_f64(), pair[1].as_f64())
                && x.is_finite()
                && y.is_finite()
            {
                pts.push([x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)]);
            } else if let Some(obj) = item.as_object()
                && let (Some(x), Some(y)) = (
                    obj.get("x").and_then(Value::as_f64),
                    obj.get("y").and_then(Value::as_f64),
                )
                && x.is_finite()
                && y.is_finite()
            {
                pts.push([x.clamp(0.0, 1.0), y.clamp(0.0, 1.0)]);
            }
        }
    }
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
    pts
}

fn decode_curves_map(params: &Map<String, Value>) -> Option<Map<String, Value>> {
    for key in ["curvCurvesJson", "curvesJson", "curves"] {
        if let Some(val) = params.get(key) {
            match val {
                Value::Object(map) => return Some(map.clone()),
                Value::String(s) => {
                    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(s) {
                        return Some(map);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Parses the control points for a channel (`"master"`, `"r"`, `"g"`, `"b"`) of the Looks
/// `curves` tool, falling back to legacy `shadowLift`/`midGamma`/`highlightGain` if needed.
#[must_use]
pub fn looks_parse_curve_channel(params: &Map<String, Value>, channel: &str) -> Vec<[f64; 2]> {
    if let Some(map) = decode_curves_map(params) {
        let keys: &[&str] = match channel {
            "r" | "red" => &["r", "red"],
            "g" | "green" => &["g", "green"],
            "b" | "blue" => &["b", "blue"],
            _ => &["master", "c", "rgb", "luma"],
        };
        for &k in keys {
            if let Some(v) = map.get(k) {
                let pts = parse_curve_point_array(v);
                if pts.len() >= 2 {
                    return pts;
                }
            }
        }
    }
    if matches!(channel, "master" | "c" | "rgb" | "luma") {
        let lift = params
            .get("shadowLift")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let gamma = params
            .get("midGamma")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        let gain = params
            .get("highlightGain")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        if lift.abs() > 1e-4 || (gamma - 1.0).abs() > 1e-4 || (gain - 1.0).abs() > 1e-4 {
            return synthesize_legacy_curves_5pt(lift, gamma, gain);
        }
    }
    default_5pt_curve()
}

/// Writes multi-point control points for `channel` (`"master"`, `"r"`, `"g"`, `"b"`) into the
/// canonical `curvCurvesJson` string read by `zc-render`, while syncing legacy fallback sliders
/// when updating `"master"`.
pub fn looks_set_curve_channel(
    params: &mut Map<String, Value>,
    channel: &str,
    mut pts: Vec<[f64; 2]>,
) {
    for pt in &mut pts {
        pt[0] = pt[0].clamp(0.0, 1.0);
        pt[1] = pt[1].clamp(0.0, 1.0);
    }
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
    if pts.len() < 2 {
        pts = default_5pt_curve();
    }

    let mut map = decode_curves_map(params).unwrap_or_else(|| {
        let mut m = Map::new();
        let def = Value::Array(default_5pt_curve().into_iter().map(|p| json!(p)).collect());
        m.insert("master".to_owned(), def.clone());
        m.insert("r".to_owned(), def.clone());
        m.insert("g".to_owned(), def.clone());
        m.insert("b".to_owned(), def);
        m
    });

    let canonical_ch = match channel {
        "r" | "red" => "r",
        "g" | "green" => "g",
        "b" | "blue" => "b",
        _ => "master",
    };
    let serialized: Vec<Value> = pts.iter().map(|&[x, y]| json!([x, y])).collect();
    map.insert(canonical_ch.to_owned(), Value::Array(serialized));

    if let Ok(encoded) = serde_json::to_string(&Value::Object(map)) {
        params.insert("curvCurvesJson".to_owned(), Value::String(encoded));
    }

    if canonical_ch == "master" {
        let y0 = looks_eval_curve_preview(&pts, 0.0);
        let y_mid = looks_eval_curve_preview(&pts, 0.5);
        let y1 = looks_eval_curve_preview(&pts, 1.0);
        params.insert("shadowLift".to_owned(), json!(y0.clamp(-0.5, 0.5)));
        let safe_mid = y_mid.clamp(0.02, 0.98);
        let approx_gamma = (0.5_f64.ln() / safe_mid.ln()).clamp(0.3, 2.5);
        params.insert("midGamma".to_owned(), json!(approx_gamma));
        params.insert("highlightGain".to_owned(), json!(y1.clamp(0.5, 1.8)));
    }
}

/// Parses control points for the Looks `s-curve` tool, falling back to `strength` and `pivot`.
#[must_use]
pub fn looks_parse_scurve_points(params: &Map<String, Value>) -> Vec<[f64; 2]> {
    for key in ["scvScurveJson", "scurveJson", "scurve", "points"] {
        if let Some(val) = params.get(key) {
            let parsed_val;
            let target = match val {
                Value::String(s) => {
                    parsed_val = serde_json::from_str::<Value>(s).ok();
                    parsed_val.as_ref()
                }
                other => Some(other),
            };
            if let Some(v) = target {
                let pts = if v.is_array() {
                    parse_curve_point_array(v)
                } else if let Some(obj) = v.as_object() {
                    obj.get("points")
                        .or_else(|| obj.get("master"))
                        .or_else(|| obj.get("curve"))
                        .map(parse_curve_point_array)
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                if pts.len() >= 2 {
                    return pts;
                }
            }
        }
    }
    let strength = params
        .get("strength")
        .and_then(Value::as_f64)
        .unwrap_or(0.35);
    let pivot = params.get("pivot").and_then(Value::as_f64).unwrap_or(0.5);
    synthesize_scurve_5pt(strength, pivot)
}

/// Writes control points into `scvScurveJson` (read by `zc-render` for the `s-curve` tool).
pub fn looks_set_scurve_points(params: &mut Map<String, Value>, mut pts: Vec<[f64; 2]>) {
    for pt in &mut pts {
        pt[0] = pt[0].clamp(0.0, 1.0);
        pt[1] = pt[1].clamp(0.0, 1.0);
    }
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
    if pts.len() < 2 {
        pts = default_5pt_curve();
    }
    let serialized: Vec<Value> = pts.iter().map(|&[x, y]| json!([x, y])).collect();
    if let Ok(encoded) = serde_json::to_string(&Value::Array(serialized)) {
        params.insert("scvScurveJson".to_owned(), Value::String(encoded));
    }
}

/// Smooth Fritsch-Carlson monotone cubic curve evaluation matching `zc-render`'s 1D curve baker.
#[must_use]
pub fn looks_eval_curve_preview(pts: &[[f64; 2]], x: f64) -> f64 {
    let n = pts.len();
    if n == 0 {
        return x.clamp(0.0, 1.0);
    }
    if n == 1 || x <= pts[0][0] {
        return pts[0][1].clamp(0.0, 1.0);
    }
    if x >= pts[n - 1][0] {
        return pts[n - 1][1].clamp(0.0, 1.0);
    }
    let mut dx = vec![0.0_f64; n - 1];
    let mut dy = vec![0.0_f64; n - 1];
    let mut slope = vec![0.0_f64; n - 1];
    for i in 0..(n - 1) {
        dx[i] = (pts[i + 1][0] - pts[i][0]).max(1e-6);
        dy[i] = pts[i + 1][1] - pts[i][1];
        slope[i] = dy[i] / dx[i];
    }
    let mut tang = vec![0.0_f64; n];
    tang[0] = slope[0];
    tang[n - 1] = slope[n - 2];
    for i in 1..(n - 1) {
        if slope[i - 1] * slope[i] <= 0.0 {
            tang[i] = 0.0;
        } else {
            tang[i] = (slope[i - 1] + slope[i]) * 0.5;
        }
    }
    for i in 0..(n - 1) {
        if slope[i].abs() < 1e-6 {
            tang[i] = 0.0;
            tang[i + 1] = 0.0;
        } else {
            let a = tang[i] / slope[i];
            let b = tang[i + 1] / slope[i];
            let mag = a * a + b * b;
            if mag > 9.0 {
                let s = 3.0 / mag.sqrt();
                tang[i] = s * a * slope[i];
                tang[i + 1] = s * b * slope[i];
            }
        }
    }
    let mut seg = 0usize;
    for i in 0..(n - 1) {
        if x >= pts[i][0] && x <= pts[i + 1][0] {
            seg = i;
            break;
        }
    }
    let h = dx[seg];
    let t = ((x - pts[seg][0]) / h).clamp(0.0, 1.0);
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    (h00 * pts[seg][1] + h10 * h * tang[seg] + h01 * pts[seg + 1][1] + h11 * h * tang[seg + 1])
        .clamp(0.0, 1.0)
}

/// Lists all `.cube` / `AssetKind::Misc` LUT assets available in `session.project.assets`,
/// returning `(asset_id, display_name)` sorted alphabetically by display name.
#[must_use]
pub fn list_cube_lut_assets(session: &Session) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = session
        .project
        .assets
        .iter()
        .filter(|(id, rec)| {
            rec.kind == zc_project::AssetKind::Misc
                || rec.filename.to_ascii_lowercase().ends_with(".cube")
                || id.to_ascii_lowercase().ends_with(".cube")
        })
        .map(|(id, rec)| {
            let display = if rec.filename.is_empty() {
                id.clone()
            } else {
                rec.filename.clone()
            };
            (id.clone(), display)
        })
        .collect();
    out.sort_by_key(|a| a.1.to_ascii_lowercase());
    out
}

/// Imports raw `.cube` LUT file bytes into `session.project.assets` as `AssetKind::Misc`
/// and returns its content-addressed asset ID.
pub fn import_cube_lut_bytes(session: &mut Session, filename: &str, bytes: Vec<u8>) -> String {
    session
        .project
        .insert_binary_asset(zc_project::AssetKind::Misc, filename.to_owned(), bytes)
}

/// Selects (or clears) a custom `.cube` LUT asset on a Looks `lut` tool instance.
/// Selecting an asset automatically switches `mode` to `4` (`Custom .cube`).
pub fn looks_select_cube_lut(params: &mut Map<String, Value>, asset_id: Option<&str>) {
    match asset_id.filter(|s| !s.trim().is_empty()) {
        Some(id) => {
            params.insert("lutAsset".to_owned(), Value::String(id.to_owned()));
            params.insert("mode".to_owned(), json!(4));
        }
        None => {
            params.remove("lutAsset");
        }
    }
}

#[cfg(test)]
mod looks_curves_and_lut_tests {
    use super::*;

    #[test]
    fn multi_point_curves_edit_roundtrips_channels_and_syncs_legacy_params() {
        let mut params = Map::new();
        let edited_master = vec![
            [0.0, 0.05],
            [0.25, 0.20],
            [0.5, 0.62],
            [0.75, 0.85],
            [1.0, 0.98],
        ];
        looks_set_curve_channel(&mut params, "master", edited_master.clone());
        let edited_red = vec![
            [0.0, 0.0],
            [0.25, 0.30],
            [0.5, 0.58],
            [0.75, 0.82],
            [1.0, 1.0],
        ];
        looks_set_curve_channel(&mut params, "r", edited_red.clone());

        assert!(
            params
                .get("curvCurvesJson")
                .and_then(Value::as_str)
                .is_some()
        );
        assert_eq!(looks_parse_curve_channel(&params, "master"), edited_master);
        assert_eq!(looks_parse_curve_channel(&params, "r"), edited_red);
        assert_eq!(looks_parse_curve_channel(&params, "g"), default_5pt_curve());

        // Legacy fallback params remain synced for backwards compatibility.
        let lift = params["shadowLift"].as_f64().expect("shadowLift");
        assert!((lift - 0.05).abs() < 1e-4);
        assert!(params["midGamma"].as_f64().expect("midGamma") > 1.0);

        // Smooth preview stays bounded and interpolates exact control points.
        assert!((looks_eval_curve_preview(&edited_master, 0.5) - 0.62).abs() < 1e-6);
        let mid_hi = looks_eval_curve_preview(&edited_master, 0.625);
        assert!(mid_hi > 0.62 && mid_hi < 0.85);
    }

    #[test]
    fn legacy_curves_params_upgrade_to_5pt_curve_and_scurve_roundtrips() {
        let mut legacy = Map::new();
        legacy.insert("shadowLift".to_owned(), json!(0.1));
        legacy.insert("midGamma".to_owned(), json!(1.2));
        legacy.insert("highlightGain".to_owned(), json!(0.95));
        let pts = looks_parse_curve_channel(&legacy, "master");
        assert_eq!(pts.len(), 5);
        assert!((pts[0][1] - 0.1_f64.powf(1.0 / 1.2) * 0.95).abs() < 1e-5);

        let mut scurve = Map::new();
        let custom_s = vec![
            [0.0, 0.0],
            [0.25, 0.18],
            [0.5, 0.5],
            [0.75, 0.84],
            [1.0, 1.0],
        ];
        looks_set_scurve_points(&mut scurve, custom_s.clone());
        assert_eq!(looks_parse_scurve_points(&scurve), custom_s);
    }

    #[test]
    fn cube_lut_import_list_and_selection_sets_custom_mode() {
        let mut session = Session::new();
        let cube_bytes = b"TITLE \"Warm\"\nLUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n".to_vec();
        let id = import_cube_lut_bytes(&mut session, "warm_film.cube", cube_bytes);
        let assets = list_cube_lut_assets(&session);
        assert!(
            assets
                .iter()
                .any(|(aid, name)| aid == &id && name == "warm_film.cube"),
            "expected imported .cube asset in list: {assets:?}"
        );

        let mut params = Map::new();
        looks_select_cube_lut(&mut params, Some(&id));
        assert_eq!(
            params.get("lutAsset").and_then(Value::as_str),
            Some(id.as_str())
        );
        assert_eq!(params.get("mode").and_then(Value::as_i64), Some(4));

        looks_select_cube_lut(&mut params, None);
        assert!(params.get("lutAsset").is_none());
    }
}
