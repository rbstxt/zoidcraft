//! Catalog of 3D `NodeMaterial` procedural PBR node types.

use super::*;

/// Complete catalog of 3D `NodeMaterial` procedural PBR nodes matching `zc_shader::node_graph`.
pub const MATERIAL_NODE_CATALOG: &[NodeTypeSpec] = &[
    // --- Inputs ---
    NodeTypeSpec {
        kind: "float",
        title: "Float Constant",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "out",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        params: &[NodeParamSpec {
            key: "value",
            label: "Value",
            kind: NodeParamKind::Float {
                min: -100.0,
                max: 100.0,
                step: 0.02,
            },
            default_f64: 0.5,
        }],
    },
    NodeTypeSpec {
        kind: "color",
        title: "Color Constant",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[NodeParamSpec {
            key: "color",
            label: "Color",
            kind: NodeParamKind::ColorRgb,
            default_f64: 0.8,
        }],
    },
    NodeTypeSpec {
        kind: "randomColor",
        title: "Random Color",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[NodeParamSpec {
            key: "seed",
            label: "Seed",
            kind: NodeParamKind::Float {
                min: 0.0,
                max: 1000.0,
                step: 1.0,
            },
            default_f64: 1.0,
        }],
    },
    NodeTypeSpec {
        kind: "projection",
        title: "UV Projection",
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
        kind: "uvw",
        title: "UV Transform",
        category: "Inputs",
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
                key: "scaleX",
                label: "Scale X",
                kind: NodeParamKind::Float {
                    min: 0.05,
                    max: 32.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "scaleY",
                label: "Scale Y",
                kind: NodeParamKind::Float {
                    min: 0.05,
                    max: 32.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "offsetX",
                label: "Offset X",
                kind: NodeParamKind::Float {
                    min: -4.0,
                    max: 4.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "offsetY",
                label: "Offset Y",
                kind: NodeParamKind::Float {
                    min: -4.0,
                    max: 4.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "rotation",
                label: "Rotation",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    // --- Procedural ---
    NodeTypeSpec {
        kind: "noise",
        title: "Perlin Noise",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "out",
                label: "Factor",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.1,
                    max: 64.0,
                    step: 0.2,
                },
                default_f64: 4.0,
            },
            NodeParamSpec {
                key: "octaves",
                label: "Octaves",
                kind: NodeParamKind::Int { min: 1, max: 8 },
                default_f64: 4.0,
            },
            NodeParamSpec {
                key: "roughness",
                label: "Detail",
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
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "distance",
                label: "Distance",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "color",
                label: "Cell Color",
                data_type: PortDataType::Color,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.5,
                    max: 64.0,
                    step: 0.2,
                },
                default_f64: 6.0,
            },
            NodeParamSpec {
                key: "jitter",
                label: "Jitter",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "checker",
        title: "Checker",
        category: "Procedural",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
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
                    max: 64.0,
                    step: 0.5,
                },
                default_f64: 8.0,
            },
            NodeParamSpec {
                key: "a",
                label: "Color A",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.15,
            },
            NodeParamSpec {
                key: "b",
                label: "Color B",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.85,
            },
        ],
    },
    NodeTypeSpec {
        kind: "gradient",
        title: "Gradient",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        params: &[NodeParamSpec {
            key: "mode",
            label: "Mode",
            kind: NodeParamKind::Choice(GRADIENT_MODES),
            default_f64: 0.0,
        }],
    },
    NodeTypeSpec {
        kind: "wave",
        title: "Wave",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
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
                    max: 64.0,
                    step: 0.2,
                },
                default_f64: 5.0,
            },
            NodeParamSpec {
                key: "distortion",
                label: "Distortion",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 10.0,
                    step: 0.1,
                },
                default_f64: 1.5,
            },
        ],
    },
    NodeTypeSpec {
        kind: "brick",
        title: "Brick",
        category: "Procedural",
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
                label: "Mortar",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 1.0,
                    max: 32.0,
                    step: 0.5,
                },
                default_f64: 4.0,
            },
            NodeParamSpec {
                key: "mortarSize",
                label: "Mortar",
                kind: NodeParamKind::Float {
                    min: 0.005,
                    max: 0.2,
                    step: 0.005,
                },
                default_f64: 0.03,
            },
        ],
    },
    NodeTypeSpec {
        kind: "marble",
        title: "Marble",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.5,
                    max: 32.0,
                    step: 0.2,
                },
                default_f64: 4.0,
            },
            NodeParamSpec {
                key: "turbulence",
                label: "Veins",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 12.0,
                    step: 0.2,
                },
                default_f64: 5.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "ridgedFractal",
        title: "Ridged Fractal",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.5,
                    max: 32.0,
                    step: 0.2,
                },
                default_f64: 4.0,
            },
            NodeParamSpec {
                key: "octaves",
                label: "Octaves",
                kind: NodeParamKind::Int { min: 1, max: 8 },
                default_f64: 4.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "turbulence",
        title: "Turbulence",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        params: &[NodeParamSpec {
            key: "scale",
            label: "Scale",
            kind: NodeParamKind::Float {
                min: 0.5,
                max: 32.0,
                step: 0.2,
            },
            default_f64: 5.0,
        }],
    },
    NodeTypeSpec {
        kind: "whiteNoise",
        title: "White Noise",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "out",
                label: "Value",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
        ],
        params: &[],
    },
    NodeTypeSpec {
        kind: "blackbody",
        title: "Blackbody",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "temperature",
            label: "Kelvin",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[NodeParamSpec {
            key: "temperature",
            label: "Kelvin",
            kind: NodeParamKind::Float {
                min: 1000.0,
                max: 15000.0,
                step: 100.0,
            },
            default_f64: 5500.0,
        }],
    },
    // --- Color & Math ---
    NodeTypeSpec {
        kind: "colorRamp",
        title: "Color Ramp",
        category: "Color & Math",
        inputs: &[PortSpec {
            id: "factor",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
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
        title: "Mix",
        category: "Color & Math",
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
                default_f64: 0.2,
            },
            NodeParamSpec {
                key: "b",
                label: "Color B",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.8,
            },
        ],
    },
    NodeTypeSpec {
        kind: "colorCorrection",
        title: "Color Correction",
        category: "Color & Math",
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
                key: "gamma",
                label: "Gamma",
                kind: NodeParamKind::Float {
                    min: 0.1,
                    max: 4.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "invert",
        title: "Invert",
        category: "Color & Math",
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
        kind: "mapRange",
        title: "Map Range",
        category: "Color & Math",
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
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "fromMax",
                label: "From Max",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "toMin",
                label: "To Min",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "toMax",
                label: "To Max",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "math",
        title: "Scalar Math",
        category: "Color & Math",
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
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.5,
            },
            NodeParamSpec {
                key: "b",
                label: "B",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.5,
            },
        ],
    },
    // --- Surface ---
    NodeTypeSpec {
        kind: "bump",
        title: "Bump / Normal Map",
        category: "Surface",
        inputs: &[PortSpec {
            id: "height",
            label: "Height",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "normal",
            label: "Normal",
            data_type: PortDataType::Vec3,
        }],
        params: &[NodeParamSpec {
            key: "strength",
            label: "Strength",
            kind: NodeParamKind::Float {
                min: 0.0,
                max: 5.0,
                step: 0.05,
            },
            default_f64: 1.0,
        }],
    },
    NodeTypeSpec {
        kind: "fresnel",
        title: "Fresnel",
        category: "Surface",
        inputs: &[],
        outputs: &[PortSpec {
            id: "out",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        params: &[NodeParamSpec {
            key: "ior",
            label: "IOR",
            kind: NodeParamKind::Float {
                min: 1.0,
                max: 4.0,
                step: 0.05,
            },
            default_f64: 1.45,
        }],
    },
    // --- Output ---
    NodeTypeSpec {
        kind: "output",
        title: "Material Output",
        category: "Output",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Base Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "emissive",
                label: "Emissive",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "roughness",
                label: "Roughness",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "metalness",
                label: "Metalness",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "normal",
                label: "Normal",
                data_type: PortDataType::Vec3,
            },
            PortSpec {
                id: "bump",
                label: "Bump",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "opacity",
                label: "Opacity",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[],
        params: &[
            NodeParamSpec {
                key: "normalMode",
                label: "Normal Mode",
                kind: NodeParamKind::Choice(NORMAL_MODES),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "normalStrength",
                label: "Normal Str",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 5.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
        ],
    },
];
