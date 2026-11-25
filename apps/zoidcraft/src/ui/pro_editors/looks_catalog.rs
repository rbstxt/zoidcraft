//! Catalog of all 29 Color Looks tools.

use super::*;

// ---------------------------------------------------------------------------
// All 29 Color Looks Tool Definitions (Matching zc-render/src/effects/mod.rs)
// ---------------------------------------------------------------------------

pub const GAMMA_SPACES: &[&str] = &["Linear", "sRGB", "Log"];
pub const LUT_MODES: &[&str] = &[
    "Neutral",
    "Warm Film",
    "Cool Cinema",
    "High Contrast",
    "Faded Vintage",
];

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
                default_f64: 0.0,
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
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "centerY",
                label: "Center Y",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
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
                default_f64: 0.0,
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
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "centerY",
                label: "Center Y",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
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
                label: "Space",
                kind: LooksParamKind::Choice(GAMMA_SPACES),
                default_f64: 1.0,
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
                    min: 0.0,
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
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "satMidtone",
                label: "Midtone Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "satShadow",
                label: "Shadow Sat",
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
        tool_id: "curves",
        label: "Tone Curves",
        stage: LooksStage::Post,
        params: &[
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
                kind: LooksParamKind::Float {
                    min: 0.5,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
        ],
    },
];
