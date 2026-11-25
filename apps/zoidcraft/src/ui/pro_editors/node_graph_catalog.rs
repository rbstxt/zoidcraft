//! Catalog of 2D `nodeeffect` graph node types.

use super::*;

/// Complete catalog of 2D `nodeeffect` nodes matching `zc_shader::node_graph`.
pub const EFFECT_NODE_CATALOG: &[NodeTypeSpec] = &[
    // --- Inputs ---
    NodeTypeSpec {
        kind: "layer",
        title: "Source Layer",
        category: "Inputs",
        inputs: &[],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[],
    },
    NodeTypeSpec {
        kind: "uv",
        title: "UV Coords",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        params: &[],
    },
    NodeTypeSpec {
        kind: "pixel",
        title: "Pixel Coord",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "out",
            label: "Pixel",
            data_type: PortDataType::Vec2,
        }],
        params: &[],
    },
    NodeTypeSpec {
        kind: "resolution",
        title: "Resolution",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "out",
            label: "Size",
            data_type: PortDataType::Vec2,
        }],
        params: &[],
    },
    NodeTypeSpec {
        kind: "time",
        title: "Time",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "out",
            label: "Seconds",
            data_type: PortDataType::Float,
        }],
        params: &[],
    },
    NodeTypeSpec {
        kind: "parameter",
        title: "Parameter",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "out",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "label",
                label: "Label",
                kind: NodeParamKind::Text,
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "kind",
                label: "Kind",
                kind: NodeParamKind::Choice(PARAM_KINDS),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "default",
                label: "Default",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.5,
            },
            NodeParamSpec {
                key: "min",
                label: "Min",
                kind: NodeParamKind::Float {
                    min: -1000.0,
                    max: 1000.0,
                    step: 0.1,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "max",
                label: "Max",
                kind: NodeParamKind::Float {
                    min: -1000.0,
                    max: 1000.0,
                    step: 0.1,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "solid",
        title: "Solid Color",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "color",
                label: "Color",
                kind: NodeParamKind::ColorRgb,
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "alpha",
                label: "Alpha",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
        ],
    },
    // --- Generators ---
    NodeTypeSpec {
        kind: "noise",
        title: "Value Noise",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "scale",
                label: "Scale",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "time",
                label: "Time",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "out",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.1,
                    max: 100.0,
                    step: 0.2,
                },
                default_f64: 6.0,
            },
            NodeParamSpec {
                key: "time",
                label: "Evolution",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "fbm",
        title: "FBM Noise",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "scale",
                label: "Scale",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "time",
                label: "Time",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "out",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.1,
                    max: 100.0,
                    step: 0.2,
                },
                default_f64: 4.0,
            },
            NodeParamSpec {
                key: "lacunarity",
                label: "Lacunarity",
                kind: NodeParamKind::Float {
                    min: 1.0,
                    max: 4.0,
                    step: 0.05,
                },
                default_f64: 2.0,
            },
            NodeParamSpec {
                key: "gain",
                label: "Gain",
                kind: NodeParamKind::Float {
                    min: 0.1,
                    max: 0.9,
                    step: 0.02,
                },
                default_f64: 0.5,
            },
        ],
    },
    NodeTypeSpec {
        kind: "voronoi",
        title: "Voronoi",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "scale",
                label: "Scale",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "time",
                label: "Time",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "out",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.5,
                    max: 80.0,
                    step: 0.2,
                },
                default_f64: 5.0,
            },
            NodeParamSpec {
                key: "mode",
                label: "Mode",
                kind: NodeParamKind::Choice(VORONOI_MODES),
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "checker",
        title: "Checkerboard",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "scale",
                label: "Scale",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "a",
                label: "Color A",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "b",
                label: "Color B",
                data_type: PortDataType::Color,
            },
        ],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "out",
                label: "Mask",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 1.0,
                    max: 128.0,
                    step: 0.5,
                },
                default_f64: 8.0,
            },
            NodeParamSpec {
                key: "a",
                label: "Color A",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.1,
            },
            NodeParamSpec {
                key: "b",
                label: "Color B",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.9,
            },
        ],
    },
    NodeTypeSpec {
        kind: "gradient",
        title: "Gradient",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "angle",
                label: "Angle",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "out",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "mode",
                label: "Mode",
                kind: NodeParamKind::Choice(GRADIENT_MODES),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "wave",
        title: "Wave Pattern",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "scale",
                label: "Scale",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "time",
                label: "Time",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "out",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.5,
                    max: 80.0,
                    step: 0.2,
                },
                default_f64: 6.0,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "vignette",
        title: "Vignette",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
        ],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "out",
                label: "Mask",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "radius",
                label: "Radius",
                kind: NodeParamKind::Float {
                    min: 0.05,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.65,
            },
            NodeParamSpec {
                key: "softness",
                label: "Softness",
                kind: NodeParamKind::Float {
                    min: 0.01,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
            NodeParamSpec {
                key: "strength",
                label: "Strength",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
        ],
    },
    NodeTypeSpec {
        kind: "grain",
        title: "Film Grain",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "amount",
                label: "Amount",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "out",
                label: "Noise",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "amount",
                label: "Amount",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.1,
            },
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.25,
                    max: 8.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "seed",
                label: "Seed",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1000.0,
                    step: 1.0,
                },
                default_f64: 1.0,
            },
        ],
    },
    // --- Color ---
    NodeTypeSpec {
        kind: "brightness",
        title: "Brightness / Contrast",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "brightness",
                label: "Brightness",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "contrast",
                label: "Contrast",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "brightness",
                label: "Brightness",
                kind: NodeParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "contrast",
                label: "Contrast",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 3.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "hue",
        title: "HSV Adjust",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "hue",
                label: "Hue",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "saturation",
                label: "Saturation",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "value",
                label: "Value",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "hue",
                label: "Hue Shift",
                kind: NodeParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "saturation",
                label: "Saturation",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 3.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "value",
                label: "Value",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 3.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "colorBalance",
        title: "Color Balance",
        category: "Color",
        inputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "lift",
                label: "Lift",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "gamma",
                label: "Gamma",
                kind: NodeParamKind::ColorRgb,
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "gain",
                label: "Gain",
                kind: NodeParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "levels",
        title: "Levels",
        category: "Color",
        inputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "black",
                label: "In Black",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 0.99,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "white",
                label: "In White",
                kind: NodeParamKind::Float {
                    min: 0.01,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "gamma",
                label: "Gamma",
                kind: NodeParamKind::Float {
                    min: 0.1,
                    max: 4.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "outBlack",
                label: "Out Black",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "outWhite",
                label: "Out White",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "invert",
        title: "Invert",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "factor",
                label: "Factor",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[NodeParamSpec {
            key: "factor",
            label: "Factor",
            kind: NodeParamKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default_f64: 1.0,
        }],
    },
    NodeTypeSpec {
        kind: "grayscale",
        title: "Grayscale",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "factor",
                label: "Factor",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[NodeParamSpec {
            key: "factor",
            label: "Factor",
            kind: NodeParamKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default_f64: 1.0,
        }],
    },
    NodeTypeSpec {
        kind: "colorRamp",
        title: "Color Ramp",
        category: "Color",
        inputs: &[PortSpec {
            id: "factor",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "interpolation",
                label: "Interp",
                kind: NodeParamKind::Choice(RAMP_INTERP),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "stops",
                label: "Stops",
                kind: NodeParamKind::ColorRamp,
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "mix",
        title: "Mix Color",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "a",
                label: "Color A",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "b",
                label: "Color B",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "factor",
                label: "Factor",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "mode",
                label: "Blend Mode",
                kind: NodeParamKind::Choice(MIX_MODES),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "factor",
                label: "Factor",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            NodeParamSpec {
                key: "a",
                label: "Color A",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "b",
                label: "Color B",
                kind: NodeParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "separateRGB",
        title: "Separate RGBA",
        category: "Color",
        inputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        outputs: &[
            PortSpec {
                id: "r",
                label: "R",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "g",
                label: "G",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "b",
                label: "B",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "a",
                label: "A",
                data_type: PortDataType::Float,
            },
        ],
        params: &[],
    },
    NodeTypeSpec {
        kind: "combineRGB",
        title: "Combine RGBA",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "r",
                label: "R",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "g",
                label: "G",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "b",
                label: "B",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "a",
                label: "A",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[NodeParamSpec {
            key: "a",
            label: "Alpha",
            kind: NodeParamKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default_f64: 1.0,
        }],
    },
    NodeTypeSpec {
        kind: "separateHSV",
        title: "Separate HSV",
        category: "Color",
        inputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        outputs: &[
            PortSpec {
                id: "h",
                label: "H",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "s",
                label: "S",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "v",
                label: "V",
                data_type: PortDataType::Float,
            },
        ],
        params: &[],
    },
    NodeTypeSpec {
        kind: "combineHSV",
        title: "Combine HSV",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "h",
                label: "H",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "s",
                label: "S",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "v",
                label: "V",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[],
    },
    // --- Math ---
    NodeTypeSpec {
        kind: "math",
        title: "Scalar Math",
        category: "Math",
        inputs: &[
            PortSpec {
                id: "a",
                label: "A",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "b",
                label: "B",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "out",
            label: "Result",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "mode",
                label: "Operation",
                kind: NodeParamKind::Choice(MATH_MODES),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "a",
                label: "A",
                kind: NodeParamKind::Float {
                    min: -1000.0,
                    max: 1000.0,
                    step: 0.05,
                },
                default_f64: 0.5,
            },
            NodeParamSpec {
                key: "b",
                label: "B",
                kind: NodeParamKind::Float {
                    min: -1000.0,
                    max: 1000.0,
                    step: 0.05,
                },
                default_f64: 0.5,
            },
        ],
    },
    NodeTypeSpec {
        kind: "vectorMath",
        title: "Vector Math",
        category: "Math",
        inputs: &[
            PortSpec {
                id: "a",
                label: "A",
                data_type: PortDataType::Vec2,
            },
            PortSpec {
                id: "b",
                label: "B",
                data_type: PortDataType::Vec2,
            },
        ],
        outputs: &[
            PortSpec {
                id: "out",
                label: "Vector",
                data_type: PortDataType::Vec2,
            },
            PortSpec {
                id: "value",
                label: "Value",
                data_type: PortDataType::Float,
            },
        ],
        params: &[NodeParamSpec {
            key: "mode",
            label: "Operation",
            kind: NodeParamKind::Choice(VEC_MATH_MODES),
            default_f64: 0.0,
        }],
    },
    NodeTypeSpec {
        kind: "mapRange",
        title: "Map Range",
        category: "Math",
        inputs: &[PortSpec {
            id: "value",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Result",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "fromMin",
                label: "From Min",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "fromMax",
                label: "From Max",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "toMin",
                label: "To Min",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "toMax",
                label: "To Max",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "clamp",
                label: "Clamp",
                kind: NodeParamKind::Bool,
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "clamp",
        title: "Clamp",
        category: "Math",
        inputs: &[PortSpec {
            id: "value",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Result",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "min",
                label: "Min",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "max",
                label: "Max",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "smoothstep",
        title: "Smoothstep / Threshold",
        category: "Math",
        inputs: &[PortSpec {
            id: "x",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Result",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "edge0",
                label: "Edge 0",
                kind: NodeParamKind::Float {
                    min: -10.0,
                    max: 10.0,
                    step: 0.01,
                },
                default_f64: 0.25,
            },
            NodeParamSpec {
                key: "edge1",
                label: "Edge 1",
                kind: NodeParamKind::Float {
                    min: -10.0,
                    max: 10.0,
                    step: 0.01,
                },
                default_f64: 0.75,
            },
        ],
    },
    NodeTypeSpec {
        kind: "trig",
        title: "Trigonometry",
        category: "Math",
        inputs: &[PortSpec {
            id: "x",
            label: "Angle",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Result",
            data_type: PortDataType::Float,
        }],
        params: &[NodeParamSpec {
            key: "op",
            label: "Function",
            kind: NodeParamKind::Choice(TRIG_MODES),
            default_f64: 0.0,
        }],
    },
    // --- UV & Filters ---
    NodeTypeSpec {
        kind: "uvTransform",
        title: "UV Transform",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        params: &[
            NodeParamSpec {
                key: "offsetX",
                label: "Offset X",
                kind: NodeParamKind::Float {
                    min: -5.0,
                    max: 5.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "offsetY",
                label: "Offset Y",
                kind: NodeParamKind::Float {
                    min: -5.0,
                    max: 5.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "scaleX",
                label: "Scale X",
                kind: NodeParamKind::Float {
                    min: 0.05,
                    max: 20.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "scaleY",
                label: "Scale Y",
                kind: NodeParamKind::Float {
                    min: 0.05,
                    max: 20.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "displace",
        title: "UV Displace",
        category: "UV & Filters",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "map",
                label: "Map",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "strength",
                label: "Strength",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "color",
                label: "Sampled",
                data_type: PortDataType::Color,
            },
        ],
        params: &[NodeParamSpec {
            key: "strength",
            label: "Strength",
            kind: NodeParamKind::Float {
                min: -0.5,
                max: 0.5,
                step: 0.005,
            },
            default_f64: 0.04,
        }],
    },
    NodeTypeSpec {
        kind: "swirl",
        title: "Swirl",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "color",
                label: "Sampled",
                data_type: PortDataType::Color,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "radius",
                label: "Radius",
                kind: NodeParamKind::Float {
                    min: 0.05,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.5,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -720.0,
                    max: 720.0,
                    step: 2.0,
                },
                default_f64: 90.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "kaleidoscope",
        title: "Kaleidoscope",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "color",
                label: "Sampled",
                data_type: PortDataType::Color,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "segments",
                label: "Segments",
                kind: NodeParamKind::Float {
                    min: 2.0,
                    max: 32.0,
                    step: 1.0,
                },
                default_f64: 6.0,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "pixelate",
        title: "Pixelate",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "color",
                label: "Sampled",
                data_type: PortDataType::Color,
            },
        ],
        params: &[NodeParamSpec {
            key: "blocks",
            label: "Blocks",
            kind: NodeParamKind::Float {
                min: 4.0,
                max: 512.0,
                step: 2.0,
            },
            default_f64: 64.0,
        }],
    },
    NodeTypeSpec {
        kind: "mirror",
        title: "Mirror",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "color",
                label: "Sampled",
                data_type: PortDataType::Color,
            },
        ],
        params: &[NodeParamSpec {
            key: "mode",
            label: "Axis",
            kind: NodeParamKind::Choice(MIRROR_MODES),
            default_f64: 0.0,
        }],
    },
    NodeTypeSpec {
        kind: "sample",
        title: "Sample Layer",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[],
    },
    NodeTypeSpec {
        kind: "blur",
        title: "Blur",
        category: "UV & Filters",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "radius",
                label: "Radius",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "mode",
                label: "Type",
                kind: NodeParamKind::Choice(BLUR_MODES),
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "radius",
                label: "Radius",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 0.1,
                    step: 0.001,
                },
                default_f64: 0.01,
            },
            NodeParamSpec {
                key: "taps",
                label: "Samples",
                kind: NodeParamKind::Int { min: 3, max: 16 },
                default_f64: 9.0,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "sharpen",
        title: "Sharpen",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "radius",
                label: "Radius",
                kind: NodeParamKind::Float {
                    min: 0.0005,
                    max: 0.02,
                    step: 0.0005,
                },
                default_f64: 0.002,
            },
            NodeParamSpec {
                key: "amount",
                label: "Amount",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 4.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "edge",
        title: "Edge Detect",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "out",
                label: "Edge",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "radius",
                label: "Radius",
                kind: NodeParamKind::Float {
                    min: 0.0005,
                    max: 0.02,
                    step: 0.0005,
                },
                default_f64: 0.002,
            },
            NodeParamSpec {
                key: "strength",
                label: "Strength",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 5.0,
                    step: 0.05,
                },
                default_f64: 1.5,
            },
        ],
    },
    NodeTypeSpec {
        kind: "chromaticAberration",
        title: "Chromatic Aberration",
        category: "UV & Filters",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "amount",
                label: "Amount",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "amount",
                label: "Amount",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 0.05,
                    step: 0.001,
                },
                default_f64: 0.006,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "glow",
        title: "Glow",
        category: "UV & Filters",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "intensity",
                label: "Intensity",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "threshold",
                label: "Threshold",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.7,
            },
            NodeParamSpec {
                key: "radius",
                label: "Radius",
                kind: NodeParamKind::Float {
                    min: 0.002,
                    max: 0.08,
                    step: 0.002,
                },
                default_f64: 0.015,
            },
            NodeParamSpec {
                key: "intensity",
                label: "Intensity",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 4.0,
                    step: 0.05,
                },
                default_f64: 0.8,
            },
        ],
    },
    // --- Output ---
    NodeTypeSpec {
        kind: "output",
        title: "Effect Output",
        category: "Output",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[],
        params: &[],
    },
];
