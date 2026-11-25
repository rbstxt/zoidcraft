use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::{SessionKeyframe, SessionProperty, SessionValueKind, property_track_to_session};

/// Canonical 200x200 centered rectangle path commands for newly created `Shape2D` layers
/// (matching CM3/OpenType `{ commands: [...] }` consumed by `zc_render::engine`).
#[must_use]
pub(crate) fn default_shape2d_rectangle_path() -> Value {
    json!({
        "commands": [
            { "type": "M", "x": -100.0, "y": -100.0 },
            { "type": "L", "x": 100.0, "y": -100.0 },
            { "type": "L", "x": 100.0, "y": 100.0 },
            { "type": "L", "x": -100.0, "y": 100.0 },
            { "type": "Z" }
        ]
    })
}

/// Returns `true` for all 9 Repeater & Effector+ classes (hyphenated or underscored).
#[must_use]
pub(crate) fn is_3d_repeater_kind(kind: &str) -> bool {
    let low = kind.trim().to_ascii_lowercase();
    low.starts_with("zoidium:repeater/") || low.starts_with("zoidcraft:repeater/")
}

/// Returns `true` for all 7 Text+ 3D modifier group classes (hyphenated or underscored).
#[must_use]
pub(crate) fn is_3d_text_plus_kind(kind: &str) -> bool {
    let low = kind.trim().to_ascii_lowercase();
    low.starts_with("zoidium:text-plus/")
        || low.starts_with("zoidcraft:text-plus/")
        || low.starts_with("zoidium:text_plus/")
}

/// Returns `true` for all 12 ExpressText+ 3D container/effector classes (hyphenated or underscored).
#[must_use]
pub(crate) fn is_3d_expresstext_plus_kind(kind: &str) -> bool {
    let low = kind.trim().to_ascii_lowercase();
    low.starts_with("zoidium:expresstext-plus/")
        || low.starts_with("zoidcraft:expresstext-plus/")
        || low.starts_with("zoidium:expresstext_plus/")
}

/// Returns `true` for Optical Flares 3D object kinds (canonical or alias).
#[must_use]
pub(crate) fn is_3d_optical_flare_kind(kind: &str) -> bool {
    let low = kind.trim().to_ascii_lowercase().replace('_', "-");
    low.starts_with("zoidium:optical-flares/") || low.starts_with("zoidcraft:optical-flares/")
}

/// Returns `true` for Trapcode Suite / Trapcode 3D object kinds.
#[must_use]
pub(crate) fn is_3d_trapcode_kind(kind: &str) -> bool {
    let low = kind.trim().to_ascii_lowercase().replace('_', "-");
    low.starts_with("zoidium:trapcode-suite/")
        || low.starts_with("zoidium:trapcode/")
        || low.starts_with("zoidcraft:trapcode-suite/")
        || low.starts_with("zoidcraft:trapcode/")
}

/// Returns `true` for Light+ 3D light object kinds.
#[must_use]
pub(crate) fn is_3d_light_plus_kind(kind: &str) -> bool {
    let low = kind.trim().to_ascii_lowercase().replace('_', "-");
    low.starts_with("zoidium:light-plus/") || low.starts_with("zoidcraft:light-plus/")
}

/// Returns `true` for Camera+ 3D camera object kinds.
#[must_use]
pub(crate) fn is_3d_camera_plus_kind(kind: &str) -> bool {
    let low = kind.trim().to_ascii_lowercase().replace('_', "-");
    matches!(
        low.as_str(),
        "zoidium:camera-plus/camera"
            | "zoidium:camera-plus/camera-plus"
            | "zoidcraft:camera-plus/camera"
            | "zoidcraft:camera-plus/camera-plus"
    )
}

/// Returns `true` for any 3D object kind that acts as a parent container (`Object3DKind::Group`).
#[must_use]
pub(crate) fn is_3d_container_kind(kind: &str) -> bool {
    kind == "group"
        || is_3d_repeater_kind(kind)
        || is_3d_text_plus_kind(kind)
        || is_3d_expresstext_plus_kind(kind)
}

/// Returns `true` for plugin 3D objects whose custom parameters live in `Object3D.preserved["property.*"]`.
#[must_use]
pub(crate) fn uses_preserved_plugin_properties(kind: &str) -> bool {
    is_3d_repeater_kind(kind)
        || is_3d_text_plus_kind(kind)
        || is_3d_expresstext_plus_kind(kind)
        || is_3d_optical_flare_kind(kind)
        || is_3d_trapcode_kind(kind)
}

/// Normalizes a 3D plugin object kind to its canonical hyphenated Zoidium ID while
/// accepting underscore and legacy aliases gracefully.
///
/// For Geometry+ polyhedron sub-aliases (`tetrahedron`, `octahedron`, `icosahedron`, `dodecahedron`),
/// returns `(canonical_kind, Some(shape_index))` where `0=tetra, 1=octa, 2=icosa, 3=dodeca`.
#[must_use]
pub(crate) fn canonicalize_object_3d_kind(raw: &str) -> (String, Option<u32>) {
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();

    if let Some(rest) = lower
        .strip_prefix("zoidium:")
        .or_else(|| lower.strip_prefix("zoidcraft:"))
        && let Some((ns_raw, sub_raw)) = rest.split_once('/')
    {
        let ns = ns_raw.replace('_', "-");
        let sub = sub_raw.replace('_', "-");
        match ns.as_str() {
            "repeater" => {
                let canonical_sub = match sub.as_str() {
                    "repeater" | "step-repeater" => "repeater",
                    "linear-repeater" | "linear" => "linear-repeater",
                    "random-repeater" | "random" => "random-repeater",
                    "echo-repeater" | "echo" => "echo-repeater",
                    "radial" | "radial-repeater" => "radial",
                    "grid" | "grid-repeater" => "grid",
                    "twist" => "twist",
                    "warp" => "warp",
                    "voronoi-fracture" | "fracture" => "voronoi-fracture",
                    "plain" | "plain-effector" => "plain",
                    "delay" | "delay-effector" => "delay",
                    other => return (format!("zoidium:repeater/{other}"), None),
                };
                return (format!("zoidium:repeater/{canonical_sub}"), None);
            }
            "geometry-plus" => match sub.as_str() {
                "rounded-box" => return ("zoidium:geometry-plus/rounded-box".to_owned(), None),
                "polyhedron" => return ("zoidium:geometry-plus/polyhedron".to_owned(), None),
                "tetrahedron" => return ("zoidium:geometry-plus/polyhedron".to_owned(), Some(0)),
                "octahedron" => return ("zoidium:geometry-plus/polyhedron".to_owned(), Some(1)),
                "icosahedron" => return ("zoidium:geometry-plus/polyhedron".to_owned(), Some(2)),
                "dodecahedron" => return ("zoidium:geometry-plus/polyhedron".to_owned(), Some(3)),
                "cone" => return ("zoidium:geometry-plus/cone".to_owned(), None),
                "capsule" => return ("zoidium:geometry-plus/capsule".to_owned(), None),
                "tube" => return ("zoidium:geometry-plus/tube".to_owned(), None),
                "gear" => return ("zoidium:geometry-plus/gear".to_owned(), None),
                "helix" => return ("zoidium:geometry-plus/helix".to_owned(), None),
                "spline" => return ("zoidium:geometry-plus/spline".to_owned(), None),
                "torus-knot" => return ("zoidium:geometry-plus/torus-knot".to_owned(), None),
                other => return (format!("zoidium:geometry-plus/{other}"), None),
            },
            "camera-plus" => match sub.as_str() {
                "camera" | "camera-plus" => {
                    return ("zoidium:camera-plus/camera".to_owned(), None);
                }
                other => return (format!("zoidium:camera-plus/{other}"), None),
            },
            "text-plus" => {
                let canonical_sub = match sub.as_str() {
                    "text-plus" | "container" => "text-plus",
                    "gradient-transform" => "gradient-transform",
                    "character-shake" => "character-shake",
                    "selected-transform" => "selected-transform",
                    "selected-shake" => "selected-shake",
                    "random-scatter" => "random-scatter",
                    "character-material" => "character-material",
                    other => return (format!("zoidium:text-plus/{other}"), None),
                };
                return (format!("zoidium:text-plus/{canonical_sub}"), None);
            }
            "expresstext-plus" => {
                let canonical_sub = match sub.as_str() {
                    "expresstext" | "expresstext-plus" => "expresstext",
                    "shake" => "shake",
                    "plain" => "plain",
                    "random" => "random",
                    "formula" => "formula",
                    "delay" => "delay",
                    "spline" => "spline",
                    "orbit" => "orbit",
                    "repeater" => "repeater",
                    "twist" => "twist",
                    "wrap" | "warp" => "wrap",
                    "voronoi" | "voronoi-fracture" => "voronoi",
                    other => return (format!("zoidium:expresstext-plus/{other}"), None),
                };
                return (format!("zoidium:expresstext-plus/{canonical_sub}"), None);
            }
            "optical-flares" => {
                return ("zoidium:optical-flares/flare".to_owned(), None);
            }
            "trapcode" | "trapcode-suite" => {
                let canonical_ns = if ns == "trapcode" {
                    "trapcode-suite"
                } else {
                    ns.as_str()
                };
                return (format!("zoidium:{canonical_ns}/{sub}"), None);
            }
            "light-plus" => {
                return (format!("zoidium:light-plus/{sub}"), None);
            }
            _ => return (format!("zoidium:{ns}/{sub}"), None),
        }
    }

    if lower == "text3d" {
        return ("text_3d".to_owned(), None);
    }

    match lower.as_str() {
        "radial-repeater" | "radial_repeater" => {
            return ("zoidium:repeater/radial".to_owned(), None);
        }
        "grid-repeater" | "grid_repeater" => {
            return ("zoidium:repeater/grid".to_owned(), None);
        }
        "light" => return ("light".to_owned(), None),
        "spotlight" | "spot-light" | "spot_light" => return ("spot_light".to_owned(), None),
        "pointlight" | "point-light" | "point_light" => return ("point_light".to_owned(), None),
        "directionallight" | "directional-light" | "directional_light" => {
            return ("directional_light".to_owned(), None);
        }
        "hemispherelight" | "hemisphere-light" | "hemisphere_light" => {
            return ("hemisphere_light".to_owned(), None);
        }
        "ambientlight" | "ambient-light" | "ambient_light" => {
            return ("ambient_light".to_owned(), None);
        }
        _ => {}
    }

    (trimmed.to_owned(), None)
}

/// Resolves the `LightKind` for a Trapcode or Light+ light subtype, or `None` if not a light.
#[must_use]
pub(crate) fn plugin_light_kind(kind: &str) -> Option<zc_project::LightKind> {
    let low = kind.trim().to_ascii_lowercase().replace('_', "-");
    if let Some(sub) = low
        .strip_prefix("zoidium:trapcode-suite/")
        .or_else(|| low.strip_prefix("zoidium:trapcode/"))
    {
        return match sub {
            "spot-light" => Some(zc_project::LightKind::Spot),
            "point-light" | "area-light" | "ies-light" | "portal-light" => {
                Some(zc_project::LightKind::Point)
            }
            "infinite-light" | "sun-light" | "directional-light" => {
                Some(zc_project::LightKind::Directional)
            }
            "hemisphere-light" => Some(zc_project::LightKind::Hemisphere),
            "ambient-light" => Some(zc_project::LightKind::Ambient),
            _ => None,
        };
    }
    if let Some(sub) = low.strip_prefix("zoidium:light-plus/") {
        return Some(match sub {
            "point-light" | "point" | "area-light" | "ies-light" | "portal-light" => {
                zc_project::LightKind::Point
            }
            "directional-light" | "directional" | "sun-light" | "infinite-light" => {
                zc_project::LightKind::Directional
            }
            "hemisphere-light" | "hemisphere" => zc_project::LightKind::Hemisphere,
            "ambient-light" | "ambient" => zc_project::LightKind::Ambient,
            _ => zc_project::LightKind::Spot,
        });
    }
    None
}

/// Default starter `ParticlesData` with crisp, non-whiteout particle sizing and emission.
#[must_use]
pub(crate) fn default_particles_data() -> zc_project::ParticlesData {
    zc_project::ParticlesData {
        time: zc_project::PropertyTrack::number_with_expr(0.0, "time"),
        color: vec![
            zc_project::GradientStop {
                position: 0.0,
                color: "rgba(255,255,255,1.0)".to_owned(),
            },
            zc_project::GradientStop {
                position: 1.0,
                color: "rgba(255,255,255,0.0)".to_owned(),
            },
        ],
        size: vec![
            zc_project::GradientStop {
                position: 0.0,
                color: "rgba(14,14,14,1.0)".to_owned(),
            },
            zc_project::GradientStop {
                position: 1.0,
                color: "rgba(4,4,4,1.0)".to_owned(),
            },
        ],
        number: 400,
        rate: 200.0,
        lifetime: 3.0,
        pdist: 0,
        ipos: [0.0, 0.0, 0.0],
        pspread: [20.0, 20.0, 20.0],
        iradius: 0.0,
        rspread: 0.0,
        vdist: 0,
        ivel: [0.0, 0.0, 0.0],
        vspread: [100.0, 100.0, 100.0],
        irvel: 0.0,
        rvspread: 0.0,
        accel: [0.0, 0.0, 0.0],
        iang: 0.0,
        aspread: 0.0,
        angvel: 0.0,
        avspread: 0.0,
        blending: 0,
        texture: None,
        sprite_preset: Some("circle_soft".to_owned()),
    }
}

/// Inserts all 10 standard Effector+ field defaults into a session object's property map.
pub(crate) fn insert_default_field_properties(props: &mut BTreeMap<String, SessionProperty>) {
    props.insert(
        "field".to_owned(),
        SessionProperty::static_value_with_kind(json!(0), SessionValueKind::Option),
    );
    props.insert(
        "fieldPosition".to_owned(),
        SessionProperty::static_value_with_kind(json!([0.0, 0.0, 0.0]), SessionValueKind::Vec3),
    );
    props.insert(
        "fieldRotation".to_owned(),
        SessionProperty::static_value_with_kind(json!([0.0, 0.0, 0.0]), SessionValueKind::Vec3),
    );
    props.insert(
        "fieldScale".to_owned(),
        SessionProperty::static_value_with_kind(
            json!([100.0, 100.0, 100.0]),
            SessionValueKind::Vec3,
        ),
    );
    props.insert(
        "fieldFalloff".to_owned(),
        SessionProperty::static_value(json!(100.0)),
    );
    props.insert(
        "fieldCurve".to_owned(),
        SessionProperty::static_value_with_kind(json!(0), SessionValueKind::Option),
    );
    props.insert(
        "fieldInvert".to_owned(),
        SessionProperty::static_value_with_kind(json!(0), SessionValueKind::Option),
    );
    props.insert(
        "fieldSweep".to_owned(),
        SessionProperty::static_value(json!(100.0)),
    );
    props.insert(
        "fieldNoiseScale".to_owned(),
        SessionProperty::static_value(json!(100.0)),
    );
    props.insert(
        "fieldNoiseEvolution".to_owned(),
        SessionProperty::static_value(json!(0.0)),
    );
}

/// Inserts default properties for all 9 Repeater & Effector+ classes (never adding PBR material).
pub(crate) fn insert_default_repeater_properties(
    canonical_kind: &str,
    props: &mut BTreeMap<String, SessionProperty>,
) {
    match canonical_kind {
        "zoidium:repeater/repeater" => {
            props.insert("count".to_owned(), SessionProperty::static_value(json!(5)));
            props.insert(
                "positionStep".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([10.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "rotationStep".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "scaleStep".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([1.0, 1.0, 1.0]),
                    SessionValueKind::Vec3,
                ),
            );
        }
        "zoidium:repeater/linear-repeater" => {
            props.insert("count".to_owned(), SessionProperty::static_value(json!(5)));
            props.insert(
                "positionEnd".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([40.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "rotationEnd".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "scaleEnd".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([1.0, 1.0, 1.0]),
                    SessionValueKind::Vec3,
                ),
            );
        }
        "zoidium:repeater/random-repeater" => {
            props.insert("count".to_owned(), SessionProperty::static_value(json!(5)));
            props.insert("seed".to_owned(), SessionProperty::static_value(json!(1)));
            props.insert(
                "positionMin".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([-10.0, -10.0, -10.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "positionMax".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([10.0, 10.0, 10.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "rotationMin".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "rotationMax".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "scaleMin".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([1.0, 1.0, 1.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "scaleMax".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([1.0, 1.0, 1.0]),
                    SessionValueKind::Vec3,
                ),
            );
        }
        "zoidium:repeater/echo-repeater" => {
            props.insert("count".to_owned(), SessionProperty::static_value(json!(5)));
            props.insert(
                "delay".to_owned(),
                SessionProperty::static_value(json!(5.0)),
            );
        }
        "zoidium:repeater/radial" => {
            props.insert("count".to_owned(), SessionProperty::static_value(json!(8)));
            props.insert(
                "radius".to_owned(),
                SessionProperty::static_value(json!(28.0)),
            );
            props.insert(
                "startAngle".to_owned(),
                SessionProperty::static_value(json!(0.0)),
            );
            props.insert(
                "endAngle".to_owned(),
                SessionProperty::static_value(json!(360.0)),
            );
            props.insert(
                "alignToCircle".to_owned(),
                SessionProperty::static_value_with_kind(json!(1), SessionValueKind::Option),
            );
            props.insert(
                "axis".to_owned(),
                SessionProperty::static_value_with_kind(json!(2), SessionValueKind::Option),
            );
        }
        "zoidium:repeater/grid" => {
            props.insert("countX".to_owned(), SessionProperty::static_value(json!(3)));
            props.insert("countY".to_owned(), SessionProperty::static_value(json!(3)));
            props.insert("countZ".to_owned(), SessionProperty::static_value(json!(1)));
            props.insert(
                "spacing".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([16.0, 16.0, 16.0]),
                    SessionValueKind::Vec3,
                ),
            );
        }
        "zoidium:repeater/twist" => {
            props.insert(
                "angle".to_owned(),
                SessionProperty::static_value(json!(0.0)),
            );
            props.insert(
                "axis".to_owned(),
                SessionProperty::static_value_with_kind(json!(1), SessionValueKind::Option),
            );
            props.insert(
                "offset".to_owned(),
                SessionProperty::static_value(json!(0.0)),
            );
            props.insert(
                "subdivision".to_owned(),
                SessionProperty::static_value(json!(1)),
            );
            insert_default_field_properties(props);
        }
        "zoidium:repeater/warp" => {
            props.insert(
                "strength".to_owned(),
                SessionProperty::static_value(json!(0.0)),
            );
            props.insert(
                "axis".to_owned(),
                SessionProperty::static_value_with_kind(json!(0), SessionValueKind::Option),
            );
            props.insert(
                "offset".to_owned(),
                SessionProperty::static_value(json!(0.0)),
            );
            props.insert(
                "subdivision".to_owned(),
                SessionProperty::static_value(json!(1)),
            );
            insert_default_field_properties(props);
        }
        "zoidium:repeater/voronoi-fracture" => {
            props.insert("cells".to_owned(), SessionProperty::static_value(json!(24)));
            props.insert("seed".to_owned(), SessionProperty::static_value(json!(1)));
            props.insert(
                "closed".to_owned(),
                SessionProperty::static_value_with_kind(json!(1), SessionValueKind::Option),
            );
            props.insert(
                "distribution".to_owned(),
                SessionProperty::static_value_with_kind(json!(0), SessionValueKind::Option),
            );
            props.insert(
                "cellScale".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([100.0, 100.0, 100.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "distance".to_owned(),
                SessionProperty::static_value(json!(0.0)),
            );
            props.insert(
                "scatter".to_owned(),
                SessionProperty::static_value(json!(0.0)),
            );
            props.insert(
                "offset".to_owned(),
                SessionProperty::static_value(json!(0.0)),
            );
            props.insert("spin".to_owned(), SessionProperty::static_value(json!(0.0)));
            props.insert(
                "direction".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "fragmentRotation".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "fragmentScale".to_owned(),
                SessionProperty::static_value(json!(100.0)),
            );
            props.insert(
                "gravity".to_owned(),
                SessionProperty::static_value(json!(0.0)),
            );
            props.insert(
                "randomness".to_owned(),
                SessionProperty::static_value(json!(0.0)),
            );
            props.insert(
                "colors".to_owned(),
                SessionProperty::static_value_with_kind(json!(0), SessionValueKind::Option),
            );
            insert_default_field_properties(props);
        }
        "zoidium:repeater/plain" => {
            props.insert(
                "positionOffset".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "rotationOffset".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "scaleOffset".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "uniformScale".to_owned(),
                SessionProperty::static_value(json!(0.0)),
            );
            insert_default_field_properties(props);
        }
        "zoidium:repeater/delay" => {
            props.insert(
                "mode".to_owned(),
                SessionProperty::static_value_with_kind(json!(0), SessionValueKind::Option),
            );
            props.insert(
                "strength".to_owned(),
                SessionProperty::static_value(json!(75.0)),
            );
            props.insert(
                "window".to_owned(),
                SessionProperty::static_value(json!(12)),
            );
            props.insert(
                "springDamping".to_owned(),
                SessionProperty::static_value(json!(0.25)),
            );
            props.insert(
                "springCycles".to_owned(),
                SessionProperty::static_value(json!(2.0)),
            );
        }
        _ => {}
    }
}

/// Inserts default properties for all 7 Text+ 3D modifier groups (never adding PBR material).
pub(crate) fn insert_default_text_plus_properties(
    canonical_kind: &str,
    props: &mut BTreeMap<String, SessionProperty>,
) {
    match canonical_kind {
        "zoidium:text-plus/text-plus" => {
            props.insert(
                "strength".to_owned(),
                SessionProperty::static_value(json!(1.0)),
            );
            props.insert(
                "positionOffset".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "rotationOffset".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "scaleOffset".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([1.0, 1.0, 1.0]),
                    SessionValueKind::Vec3,
                ),
            );
        }
        "zoidium:text-plus/gradient-transform" => {
            props.insert(
                "strength".to_owned(),
                SessionProperty::static_value(json!(1.0)),
            );
            props.insert(
                "reverse".to_owned(),
                SessionProperty::static_value_with_kind(json!(0), SessionValueKind::Option),
            );
            props.insert(
                "startPosition".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "endPosition".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "startRotation".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "endRotation".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "startScale".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([1.0, 1.0, 1.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "endScale".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([1.0, 1.0, 1.0]),
                    SessionValueKind::Vec3,
                ),
            );
        }
        "zoidium:text-plus/character-shake" => {
            props.insert(
                "strength".to_owned(),
                SessionProperty::static_value(json!(1.0)),
            );
            props.insert(
                "frequency".to_owned(),
                SessionProperty::static_value(json!(4.0)),
            );
            props.insert(
                "stagger".to_owned(),
                SessionProperty::static_value(json!(0.35)),
            );
            props.insert(
                "positionAmplitude".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "rotationAmplitude".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "scaleAmplitude".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
        }
        "zoidium:text-plus/selected-transform" => {
            props.insert(
                "strength".to_owned(),
                SessionProperty::static_value(json!(1.0)),
            );
            props.insert(
                "characterNumbers".to_owned(),
                SessionProperty::static_value_with_kind(json!("1"), SessionValueKind::Text),
            );
            props.insert(
                "positionOffset".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "rotationOffset".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "scaleOffset".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([1.0, 1.0, 1.0]),
                    SessionValueKind::Vec3,
                ),
            );
        }
        "zoidium:text-plus/selected-shake" => {
            props.insert(
                "strength".to_owned(),
                SessionProperty::static_value(json!(1.0)),
            );
            props.insert(
                "characterNumbers".to_owned(),
                SessionProperty::static_value_with_kind(json!("1"), SessionValueKind::Text),
            );
            props.insert(
                "frequency".to_owned(),
                SessionProperty::static_value(json!(4.0)),
            );
            props.insert(
                "stagger".to_owned(),
                SessionProperty::static_value(json!(0.35)),
            );
            props.insert(
                "positionAmplitude".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "rotationAmplitude".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "scaleAmplitude".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
        }
        "zoidium:text-plus/random-scatter" => {
            props.insert(
                "strength".to_owned(),
                SessionProperty::static_value(json!(1.0)),
            );
            props.insert("seed".to_owned(), SessionProperty::static_value(json!(1)));
            props.insert(
                "positionRange".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "rotationRange".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "scaleRange".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.0, 0.0, 0.0]),
                    SessionValueKind::Vec3,
                ),
            );
        }
        "zoidium:text-plus/character-material" => {
            props.insert(
                "characterNumbers".to_owned(),
                SessionProperty::static_value_with_kind(json!("1"), SessionValueKind::Text),
            );
        }
        _ => {}
    }
}

/// Inserts default properties for Optical Flares 3D object.
pub(crate) fn insert_default_optical_flare_properties(
    props: &mut BTreeMap<String, SessionProperty>,
) {
    props.insert(
        "flareColor".to_owned(),
        SessionProperty::static_value_with_kind(json!([1.0, 0.92, 0.8]), SessionValueKind::Color),
    );
    props.insert(
        "brightness".to_owned(),
        SessionProperty::static_value(json!(1.0)),
    );
    props.insert("size".to_owned(), SessionProperty::static_value(json!(1.0)));
    props.insert(
        "anamorphic".to_owned(),
        SessionProperty::static_value(json!(0.5)),
    );
    props.insert("ghosts".to_owned(), SessionProperty::static_value(json!(4)));
    props.insert(
        "starburstRays".to_owned(),
        SessionProperty::static_value(json!(8)),
    );
    props.insert(
        "ringIntensity".to_owned(),
        SessionProperty::static_value(json!(0.4)),
    );
    props.insert(
        "occlusion".to_owned(),
        SessionProperty::static_value_with_kind(json!(1), SessionValueKind::Option),
    );
    props.insert(
        "angle".to_owned(),
        SessionProperty::static_value(json!(0.0)),
    );
}

/// Inserts default properties for Trapcode Suite Particular / Form / Plexus objects.
pub(crate) fn insert_default_trapcode_properties(
    canonical_kind: &str,
    props: &mut BTreeMap<String, SessionProperty>,
) {
    let sub = canonical_kind
        .strip_prefix("zoidium:trapcode-suite/")
        .or_else(|| canonical_kind.strip_prefix("zoidium:trapcode/"))
        .unwrap_or("");
    match sub {
        "particular" => {
            props.insert("count".to_owned(), SessionProperty::static_value(json!(64)));
            props.insert(
                "rate".to_owned(),
                SessionProperty::static_value(json!(30.0)),
            );
            props.insert(
                "lifetime".to_owned(),
                SessionProperty::static_value(json!(3.0)),
            );
            props.insert(
                "particleSize".to_owned(),
                SessionProperty::static_value(json!(3.0)),
            );
            props.insert(
                "velocity".to_owned(),
                SessionProperty::static_value(json!(12.0)),
            );
            props.insert(
                "gravity".to_owned(),
                SessionProperty::static_value(json!(-4.0)),
            );
            props.insert(
                "emitterSize".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([6.0, 6.0, 6.0]),
                    SessionValueKind::Vec3,
                ),
            );
            props.insert(
                "particleColor".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.40, 0.85, 1.0]),
                    SessionValueKind::Color,
                ),
            );
            props.insert(
                "opacity".to_owned(),
                SessionProperty::static_value(json!(1.0)),
            );
            props.insert("seed".to_owned(), SessionProperty::static_value(json!(1)));
        }
        "form" => {
            props.insert(
                "particlesX".to_owned(),
                SessionProperty::static_value(json!(6)),
            );
            props.insert(
                "particlesY".to_owned(),
                SessionProperty::static_value(json!(6)),
            );
            props.insert(
                "particlesZ".to_owned(),
                SessionProperty::static_value(json!(4)),
            );
            props.insert(
                "sizeX".to_owned(),
                SessionProperty::static_value(json!(24.0)),
            );
            props.insert(
                "sizeY".to_owned(),
                SessionProperty::static_value(json!(24.0)),
            );
            props.insert(
                "sizeZ".to_owned(),
                SessionProperty::static_value(json!(16.0)),
            );
            props.insert(
                "displace".to_owned(),
                SessionProperty::static_value(json!(2.5)),
            );
            props.insert(
                "particleSize".to_owned(),
                SessionProperty::static_value(json!(0.85)),
            );
            props.insert(
                "particleColor".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.85, 0.92, 1.0]),
                    SessionValueKind::Color,
                ),
            );
            props.insert(
                "opacity".to_owned(),
                SessionProperty::static_value(json!(1.0)),
            );
            props.insert("seed".to_owned(), SessionProperty::static_value(json!(1)));
        }
        "plexus" => {
            props.insert("count".to_owned(), SessionProperty::static_value(json!(24)));
            props.insert(
                "radius".to_owned(),
                SessionProperty::static_value(json!(18.0)),
            );
            props.insert(
                "maxDistance".to_owned(),
                SessionProperty::static_value(json!(15.3)),
            );
            props.insert(
                "beamThickness".to_owned(),
                SessionProperty::static_value(json!(0.25)),
            );
            props.insert(
                "lineColor".to_owned(),
                SessionProperty::static_value_with_kind(
                    json!([0.85, 0.92, 1.0]),
                    SessionValueKind::Color,
                ),
            );
            props.insert(
                "opacity".to_owned(),
                SessionProperty::static_value(json!(1.0)),
            );
            props.insert("seed".to_owned(), SessionProperty::static_value(json!(1)));
        }
        _ => {}
    }
}

/// Inserts default properties for ExpressText+ container/effector classes.
pub(crate) fn insert_default_expresstext_properties(
    canonical_kind: &str,
    props: &mut BTreeMap<String, SessionProperty>,
) {
    props.insert(
        "lettersMode".to_owned(),
        SessionProperty::static_value_with_kind(json!("per_letter"), SessionValueKind::Text),
    );
    let sub = canonical_kind
        .strip_prefix("zoidium:expresstext-plus/")
        .unwrap_or("");
    if sub != "expresstext" && !sub.is_empty() {
        props.insert(
            format!("{sub}Enable"),
            SessionProperty::static_value_with_kind(json!(1), SessionValueKind::Option),
        );
        props.insert(
            format!("{sub}Strength"),
            SessionProperty::static_value(json!(100.0)),
        );
        match sub {
            "shake" => {
                props.insert(
                    "shakePosition".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([2.0, 4.0, 0.0]),
                        SessionValueKind::Vec3,
                    ),
                );
                props.insert(
                    "shakeRotation".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([0.0, 0.0, 0.15]),
                        SessionValueKind::Vec3,
                    ),
                );
                props.insert(
                    "shakeFrequency".to_owned(),
                    SessionProperty::static_value(json!(2.0)),
                );
            }
            "plain" => {
                props.insert(
                    "plainPosition".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([0.0, 6.0, 0.0]),
                        SessionValueKind::Vec3,
                    ),
                );
                props.insert(
                    "plainRotation".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([0.0, 0.25, 0.0]),
                        SessionValueKind::Vec3,
                    ),
                );
                props.insert(
                    "plainScale".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([1.15, 1.15, 1.15]),
                        SessionValueKind::Vec3,
                    ),
                );
            }
            "random" => {
                props.insert(
                    "randomMode".to_owned(),
                    SessionProperty::static_value_with_kind(json!(2), SessionValueKind::Option),
                );
                props.insert(
                    "randomPosition".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([6.0, 6.0, 3.0]),
                        SessionValueKind::Vec3,
                    ),
                );
                props.insert(
                    "randomPositionMin".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([-6.0, -6.0, -3.0]),
                        SessionValueKind::Vec3,
                    ),
                );
                props.insert(
                    "randomPositionMax".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([6.0, 6.0, 3.0]),
                        SessionValueKind::Vec3,
                    ),
                );
                props.insert(
                    "randomRotation".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([0.25, 0.25, 0.25]),
                        SessionValueKind::Vec3,
                    ),
                );
                props.insert(
                    "randomRotationMin".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([-0.25, -0.25, -0.25]),
                        SessionValueKind::Vec3,
                    ),
                );
                props.insert(
                    "randomRotationMax".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([0.25, 0.25, 0.25]),
                        SessionValueKind::Vec3,
                    ),
                );
                props.insert(
                    "randomAnimationSpeed".to_owned(),
                    SessionProperty::static_value(json!(100.0)),
                );
            }
            "formula" => {
                props.insert(
                    "formulaPosition".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([0.0, 10.0, 0.0]),
                        SessionValueKind::Vec3,
                    ),
                );
                props.insert(
                    "formulaExpression".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!("sin(u*360*f+t*90)"),
                        SessionValueKind::Text,
                    ),
                );
                props.insert(
                    "formulaFrequency".to_owned(),
                    SessionProperty::static_value(json!(1.0)),
                );
            }
            "delay" => {
                props.insert(
                    "delayMode".to_owned(),
                    SessionProperty::static_value_with_kind(json!(2), SessionValueKind::Option),
                );
                props.insert(
                    "delayFrames".to_owned(),
                    SessionProperty::static_value(json!(6.0)),
                );
            }
            "spline" => {
                props.insert(
                    "splinePath".to_owned(),
                    SessionProperty::static_value_with_kind(json!(1), SessionValueKind::Option),
                );
                props.insert(
                    "splineRadius".to_owned(),
                    SessionProperty::static_value(json!(50.0)),
                );
                props.insert(
                    "splineAmplitude".to_owned(),
                    SessionProperty::static_value(json!(25.0)),
                );
                props.insert(
                    "splineStart".to_owned(),
                    SessionProperty::static_value(json!(0.0)),
                );
                props.insert(
                    "splineEnd".to_owned(),
                    SessionProperty::static_value(json!(100.0)),
                );
            }
            "orbit" => {
                props.insert(
                    "orbitAngle".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([0.0, std::f64::consts::FRAC_PI_4, 0.0]),
                        SessionValueKind::Vec3,
                    ),
                );
                props.insert(
                    "orbitRadius".to_owned(),
                    SessionProperty::static_value(json!(120.0)),
                );
            }
            "repeater" => {
                props.insert(
                    "repeaterCopies".to_owned(),
                    SessionProperty::static_value(json!(3)),
                );
                props.insert(
                    "repeaterCount".to_owned(),
                    SessionProperty::static_value(json!(3)),
                );
                props.insert(
                    "repeaterOffset".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([10.0, 0.0, -5.0]),
                        SessionValueKind::Vec3,
                    ),
                );
                props.insert(
                    "repeaterFade".to_owned(),
                    SessionProperty::static_value(json!(50.0)),
                );
            }
            "twist" => {
                props.insert(
                    "twistAngle".to_owned(),
                    SessionProperty::static_value(json!(std::f64::consts::FRAC_PI_3)),
                );
                props.insert(
                    "twistAxis".to_owned(),
                    SessionProperty::static_value_with_kind(json!(2), SessionValueKind::Option),
                );
            }
            "wrap" => {
                props.insert(
                    "wrapAmplitude".to_owned(),
                    SessionProperty::static_value(json!(8.0)),
                );
                props.insert(
                    "wrapFrequency".to_owned(),
                    SessionProperty::static_value(json!(1.0)),
                );
                props.insert(
                    "wrapDirection".to_owned(),
                    SessionProperty::static_value_with_kind(json!(1), SessionValueKind::Option),
                );
            }
            "voronoi" => {
                props.insert(
                    "voronoiChunks".to_owned(),
                    SessionProperty::static_value(json!(8)),
                );
                props.insert(
                    "voronoiDistance".to_owned(),
                    SessionProperty::static_value(json!(6.0)),
                );
                props.insert(
                    "voronoiScatter".to_owned(),
                    SessionProperty::static_value(json!(2.5)),
                );
                props.insert(
                    "voronoiSpin".to_owned(),
                    SessionProperty::static_value_with_kind(
                        json!([0.5, 0.5, 0.5]),
                        SessionValueKind::Vec3,
                    ),
                );
            }
            _ => {}
        }
    }
}

/// Known Material+ `PbrPlusMaterial.preserved` surface shader keys.
pub(crate) const PBR_PLUS_SURFACE_KEYS: &[&str] = &[
    "aoEnabled",
    "aoIntensity",
    "aoColor",
    "aoDistance",
    "celEnabled",
    "celSteps",
    "celSmoothness",
    "celSpecularSteps",
    "fresnelEnabled",
    "fresnelColor",
    "fresnelPower",
    "fresnelMixStrength",
    "rampEnabled",
    "rampFactor",
    "rampSource",
    "ramp",
    "bumpNoise",
    "bumpStrength",
    "bumpNoiseScale",
    "projection",
    "projScale",
    "projTranslate",
    "projRotation",
    "envMap",
    "reflectionEnvColorTop",
    "reflectionEnvColorBottom",
    "reflectionEnvRotation",
];

/// Converts a JSON value stored under `Object3D.preserved["property.<key>"]` into a
/// `SessionProperty` (handling both plain JSON primitives/arrays and serialized `PropertyTrack`).
#[must_use]
pub(crate) fn preserved_value_to_session_property(val: &Value) -> SessionProperty {
    if val.is_object()
        && let Ok(track) = serde_json::from_value::<zc_project::PropertyTrack>(val.clone())
    {
        return property_track_to_session(&track);
    }
    SessionProperty {
        animated: false,
        expression: None,
        default_value: val.clone(),
        keyframes: vec![SessionKeyframe::new(0.0, val.clone(), 1)],
    }
}

/// Serializes a `SessionProperty` into a `Value` suitable for `Object3D.preserved["property.<key>"]`.
/// Static unanimated properties serialize cleanly as plain JSON values; animated or
/// expression-driven properties serialize as full `PropertyTrack` JSON objects understood
/// by both `zc-render` and `zc-project`.
#[must_use]
pub(crate) fn session_property_to_preserved_value(
    prop: &SessionProperty,
    to_track: impl FnOnce(&SessionProperty) -> zc_project::PropertyTrack,
) -> Value {
    if prop.animated || prop.expression.is_some() || prop.keyframes.len() > 1 {
        let track = to_track(prop);
        serde_json::to_value(track).unwrap_or_else(|_| prop.default_value.clone())
    } else {
        prop.keyframes
            .first()
            .map_or_else(|| prop.default_value.clone(), |kf| kf.value.clone())
    }
}

/// Returns `true` if a `SessionObject3D.properties` key is a built-in core/material/constraint
/// key rather than a plugin `preserved["property.*"]` key.
#[must_use]
pub(crate) fn is_core_object_3d_property(key: &str) -> bool {
    matches!(
        key,
        "enabled" | "position" | "rotation" | "scale" | "eulerOrder" | "reflectionVisibility"
    ) || key.starts_with("material.")
        || key.starts_with("frontMaterial.")
        || key.starts_with("sideMaterial.")
        || key.starts_with("constraint.")
}
