use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};
use uuid::Uuid;

use crate::model::{
    CameraData, CameraDofParams, CameraKind, CameraMotionBlurParams, CameraShakeParams,
    CameraVibrateParams, CompatMode, EulerOrder, FollowConstraint, GradientStop, LightData,
    LightKind, MaterialSpec, ModelData, Object3D, Object3DKind, ObjectConstraints, ParticlesData,
    PropertyTrack, PropertyValue, Shape3DKind, Text3DAlign, Text3DData, TrackToConstraint,
};

use super::effects::parse_custom_properties;
use super::materials::parse_material;
use super::tracks::{
    infer_default_value, parse_gradient_stops, parse_particle_color_stops,
    parse_particle_size_stops, parse_property_track, prop_f64, prop_str, prop_vec2, prop_vec3,
    prop_vec4, value_as_bool, value_as_f64, value_as_f64_array, value_as_u32,
};

pub(crate) fn parse_object3d(
    val: &Value,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> Object3D {
    let Some(obj) = val.as_object() else {
        return Object3D::new(
            "3D Object",
            Object3DKind::Dummy {
                original_type: None,
                payload: val.clone(),
            },
        );
    };

    let props = obj.get("properties").and_then(Value::as_object);
    let obj_type_val = obj.get("type");
    let sub_type = obj.get("objectType").and_then(value_as_u32).unwrap_or(1);

    let mut plugin_object_type: Option<String> = None;
    let mut parsed_material_slots: BTreeMap<String, MaterialSpec> = BTreeMap::new();
    let mut extra_preserved: BTreeMap<String, Value> = BTreeMap::new();

    let (default_name, default_pos, kind) =
        if let Some(type_str) = obj_type_val.and_then(Value::as_str) {
            parse_string_typed_object3d(
                type_str,
                obj,
                props,
                val,
                &mut plugin_object_type,
                &mut parsed_material_slots,
                &mut extra_preserved,
                plugins,
                warnings,
            )
        } else {
            let explicit_type = obj_type_val.and_then(value_as_u32);
            let explicit_object_type = obj.get("objectType").and_then(value_as_u32);
            let has_group_children = obj.get("objects").and_then(Value::as_array).is_some()
                || obj.get("children").and_then(Value::as_array).is_some();
            // Native CM3 3D Group objects may arrive as `{ "objectType": 0, "objects": [...] }`
            // or `{ "type": 0, "objectType": 0, "objects": [...] }`; bare CM3 shapes may omit `"type"`
            // and carry `"objectType": 1..=6`.
            let resolved_numeric_type = match (explicit_type, explicit_object_type) {
                (None, Some(0)) => Some(5),
                (None, Some(1..=6)) => Some(0),
                (Some(0), Some(0)) if has_group_children => Some(5),
                (t, _) => t,
            };
            match resolved_numeric_type {
                Some(0) => parse_shape3d_object(props, sub_type, val, warnings),
                Some(1) => (
                    "Text",
                    [0.0, 0.0, 0.0],
                    Object3DKind::Text3D(Box::new(parse_text3d_data(obj, props, plugins))),
                ),
                Some(2) => {
                    let mut model_props = BTreeMap::new();
                    if let Some(mp) = props
                        .and_then(|p| p.get("modelProperties"))
                        .and_then(Value::as_object)
                    {
                        for (k, v) in mp {
                            let default_val = infer_default_value(v);
                            model_props.insert(k.clone(), parse_property_track(v, default_val));
                        }
                    }
                    (
                        "Model",
                        [0.0, 0.0, 0.0],
                        Object3DKind::Model(Box::new(ModelData {
                            preset_type: Some(sub_type),
                            asset_id: None,
                            model_properties: model_props,
                        })),
                    )
                }
                Some(3) => {
                    let light = parse_light_data(props, sub_type);
                    let name = match light.kind {
                        LightKind::Spot => "Light",
                        LightKind::Point => "Point Light",
                        LightKind::Directional => "Directional Light",
                        LightKind::Hemisphere => "Hemisphere Light",
                        LightKind::Ambient => "Ambient Light",
                    };
                    (name, [0.0, 10.0, 0.0], Object3DKind::Light(Box::new(light)))
                }
                Some(4) => {
                    let sprite_preset = obj
                        .get("_zoidiumParticlesPlusSprite")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(ToOwned::to_owned);
                    let has_particles_plus_props = props
                        .is_some_and(|p| p.contains_key("turbulence") || p.contains_key("drag"))
                        || obj.contains_key("turbulence")
                        || obj.contains_key("drag");
                    if sprite_preset.is_some() || has_particles_plus_props {
                        plugins.insert("particles-plus".to_owned());
                    }
                    let mut particles = parse_particles_data(props);
                    particles.sprite_preset = sprite_preset;
                    (
                        "Particles",
                        [0.0, 0.0, 0.0],
                        Object3DKind::Particles(Box::new(particles)),
                    )
                }
                Some(5) => {
                    let reflection_visibility = props
                        .and_then(|p| p.get("reflectionVisibility"))
                        .and_then(value_as_u32)
                        .unwrap_or(0);
                    let custom_properties = parse_custom_properties(
                        obj.get("customProperties")
                            .or_else(|| props.and_then(|p| p.get("customProperties"))),
                    );
                    let mut children = Vec::new();
                    if let Some(arr) = obj
                        .get("objects")
                        .and_then(Value::as_array)
                        .or_else(|| obj.get("children").and_then(Value::as_array))
                    {
                        for child in arr {
                            children.push(parse_object3d(child, plugins, warnings));
                        }
                    }
                    (
                        "Group",
                        [0.0, 0.0, 0.0],
                        Object3DKind::Group {
                            reflection_visibility,
                            custom_properties,
                            children,
                        },
                    )
                }
                Some(6) => {
                    let cam = parse_camera_data(props, sub_type, plugins);
                    let name = match cam.kind {
                        CameraKind::Perspective => "Perspective Camera",
                        CameraKind::Orthographic => "Orthographic Camera",
                    };
                    (name, [0.0, 0.0, 80.0], Object3DKind::Camera(Box::new(cam)))
                }
                // Legacy Effector+ numeric types: 7 = twist, 8 = warp, 9 = voronoi-fracture.
                Some(7) => parse_string_typed_object3d(
                    "zoidium:repeater/twist",
                    obj,
                    props,
                    val,
                    &mut plugin_object_type,
                    &mut parsed_material_slots,
                    &mut extra_preserved,
                    plugins,
                    warnings,
                ),
                Some(8) => parse_string_typed_object3d(
                    "zoidium:repeater/warp",
                    obj,
                    props,
                    val,
                    &mut plugin_object_type,
                    &mut parsed_material_slots,
                    &mut extra_preserved,
                    plugins,
                    warnings,
                ),
                Some(9) => parse_string_typed_object3d(
                    "zoidium:repeater/voronoi-fracture",
                    obj,
                    props,
                    val,
                    &mut plugin_object_type,
                    &mut parsed_material_slots,
                    &mut extra_preserved,
                    plugins,
                    warnings,
                ),
                Some(13) => {
                    plugins.insert("optical-flares".to_owned());
                    plugin_object_type = Some("zoidium:optical-flares/flare".to_owned());
                    (
                        "Optical Flares",
                        [0.0, 0.0, 0.0],
                        Object3DKind::Dummy {
                            original_type: Some("13".to_owned()),
                            payload: val.clone(),
                        },
                    )
                }
                Some(99) => (
                    "3D Object",
                    [0.0, 0.0, 0.0],
                    Object3DKind::Dummy {
                        original_type: Some("99".to_owned()),
                        payload: val.clone(),
                    },
                ),
                _ => {
                    let raw_t = obj_type_val.map(Value::to_string);
                    if let Some(ref s) = raw_t {
                        warnings.push(format!("preserved unknown 3D object type {s} as Dummy"));
                    }
                    (
                        "3D Object",
                        [0.0, 0.0, 0.0],
                        Object3DKind::Dummy {
                            original_type: raw_t,
                            payload: val.clone(),
                        },
                    )
                }
            }
        };

    let is_camera_or_light = matches!(kind, Object3DKind::Camera(_) | Object3DKind::Light(_));
    let mut out = Object3D::new(default_name, kind);
    out.position = PropertyTrack::vec3(default_pos);
    out.zoidium_uv_mode = CompatMode::Legacy;
    out.zoidium_shadow_mode = CompatMode::Legacy;
    out.material_slots = parsed_material_slots;
    out.preserved.extend(extra_preserved);
    if let Some(pt) = plugin_object_type {
        out.preserved
            .insert("zoidiumPluginObjectType".to_owned(), Value::String(pt));
    }

    // Check top-level compatibility flags on the object before falling back to `properties`.
    if let Some(uv) = obj.get("zoidiumUvMode").and_then(Value::as_str) {
        out.zoidium_uv_mode = CompatMode::from_token(uv);
    }
    if let Some(sh) = obj.get("zoidiumShadowMode").and_then(Value::as_str) {
        out.zoidium_shadow_mode = CompatMode::from_token(sh);
    }

    if let Some(props) = props {
        if let Some(n) = prop_str(props, "name") {
            out.name = n;
        }
        if let Some(v) = props.get("enabled") {
            out.enabled = parse_property_track(v, PropertyValue::Option(1));
        }
        if let Some(v) = props.get("position") {
            out.position = parse_property_track(v, PropertyValue::Vec3(default_pos));
        }
        if let Some(v) = props.get("rotation") {
            out.rotation = parse_property_track(v, PropertyValue::Vec3([0.0, 0.0, 0.0]));
        }
        if let Some(v) = props.get("scale") {
            out.scale = parse_property_track(v, PropertyValue::Vec3([1.0, 1.0, 1.0]));
        }
        if let Some(order_str) = prop_str(props, "eulerOrder")
            && let Some(order) = EulerOrder::from_token(&order_str)
        {
            out.euler_order = order;
        }
        if !obj.contains_key("zoidiumUvMode")
            && let Some(uv) = prop_str(props, "zoidiumUvMode")
        {
            out.zoidium_uv_mode = CompatMode::from_token(&uv);
        }
        if !obj.contains_key("zoidiumShadowMode")
            && let Some(sh) = prop_str(props, "zoidiumShadowMode")
        {
            out.zoidium_shadow_mode = CompatMode::from_token(&sh);
        }
        out.constraints = parse_constraints(props, is_camera_or_light, plugins);

        for (k, v) in props {
            if !is_known_object3d_prop(k) {
                out.preserved.insert(format!("property.{k}"), v.clone());
            }
        }
    }

    if let Some(mat_val) = obj.get("material")
        && !mat_val.is_null()
    {
        let (mat, mat_uv_mode) = parse_material(mat_val, plugins);
        if let Some(mode) = mat_uv_mode {
            out.zoidium_uv_mode = mode;
        }
        out.material = Some(mat);
    } else if matches!(out.kind, Object3DKind::Shape3D(_) | Object3DKind::Text3D(_)) {
        out.material = Some(MaterialSpec::single_color_default());
    }

    // Parse direct `frontMaterial` / `sideMaterial` slots on Text3D and other 3D objects.
    for (slot_key, json_key) in [("front", "frontMaterial"), ("side", "sideMaterial")] {
        if !out.material_slots.contains_key(slot_key)
            && let Some(mat_val) = obj
                .get(json_key)
                .or_else(|| props.and_then(|p| p.get(json_key)))
            && !mat_val.is_null()
        {
            let (mat, mat_uv_mode) = parse_material(mat_val, plugins);
            if let Some(mode) = mat_uv_mode {
                out.zoidium_uv_mode = mode;
            }
            out.material_slots.insert(slot_key.to_owned(), mat);
        }
    }

    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type"
                | "objectType"
                | "properties"
                | "material"
                | "objects"
                | "customProperties"
                | "zoidiumTextScaleMode"
                | "zoidiumUvMode"
                | "zoidiumShadowMode"
                | "frontMaterial"
                | "sideMaterial"
                | "_zoidiumParticlesPlusSprite"
        ) {
            out.preserved.insert(k.clone(), v.clone());
        }
    }

    out
}

#[allow(clippy::too_many_arguments)]
pub(super) fn parse_string_typed_object3d(
    type_str: &str,
    obj: &Map<String, Value>,
    props: Option<&Map<String, Value>>,
    raw_val: &Value,
    plugin_object_type: &mut Option<String>,
    material_slots: &mut BTreeMap<String, MaterialSpec>,
    extra_preserved: &mut BTreeMap<String, Value>,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> (&'static str, [f64; 3], Object3DKind) {
    match type_str {
        "zoidium:camera-plus/camera" | "zoidium:camera-plus/camera_plus" => {
            plugins.insert("camera-plus".to_owned());
            *plugin_object_type = Some("zoidium:camera-plus/camera".to_owned());
            let cam = parse_camera_data(props, 1, plugins);
            (
                "Camera+",
                [0.0, 0.0, 80.0],
                Object3DKind::Camera(Box::new(cam)),
            )
        }
        // --- Geometry+ (8 primitives) ---
        "zoidium:geometry-plus/rounded-box" | "zoidium:geometry-plus/rounded_box" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some("zoidium:geometry-plus/rounded-box".to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let size = props
                .and_then(|p| prop_track_vec3(p, "size"))
                .unwrap_or([40.0, 24.0, 6.0]);
            let radius = props
                .and_then(|p| prop_track_f64(p, "cornerRadius"))
                .unwrap_or(3.0);
            let segments = props
                .and_then(|p| prop_track_f64(p, "cornerSegments"))
                .map_or(6, |v| v.round().max(1.0) as u32);
            let bevel_size = props
                .and_then(|p| prop_track_f64(p, "bevelSize"))
                .or(Some(0.6));
            let bevel_segments = props
                .and_then(|p| prop_track_f64(p, "bevelSegments"))
                .map(|v| v.round().max(0.0) as u32)
                .or(Some(2));
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Rounded Box",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::RoundedBox {
                    size,
                    radius,
                    segments,
                    corner_radius: Some(radius),
                    corner_segments: Some(segments),
                    bevel_size,
                    bevel_segments,
                    shading,
                }),
            )
        }
        "zoidium:geometry-plus/polyhedron" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let shape_idx = props
                .and_then(|p| p.get("shape"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            let radius = props
                .and_then(|p| prop_track_f64(p, "radius"))
                .unwrap_or(12.0);
            let detail = props
                .and_then(|p| prop_track_f64(p, "detail"))
                .map_or(0, |v| v.round().max(0.0) as u32);
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            // Zoidium order: 0=tetrahedron, 1=octahedron, 2=icosahedron, 3=dodecahedron.
            let shape = match shape_idx {
                1 => Shape3DKind::Octahedron {
                    radius,
                    detail,
                    shading,
                },
                2 => Shape3DKind::Icosahedron {
                    radius,
                    detail,
                    shading,
                },
                3 => Shape3DKind::Dodecahedron {
                    radius,
                    detail,
                    shading,
                },
                _ => Shape3DKind::Tetrahedron {
                    radius,
                    detail,
                    shading,
                },
            };
            ("Polyhedron", [0.0, 0.0, 0.0], Object3DKind::Shape3D(shape))
        }
        "zoidium:geometry-plus/cone" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let top_radius = props
                .and_then(|p| prop_track_f64(p, "topRadius"))
                .unwrap_or(0.0);
            let radius = props
                .and_then(|p| prop_track_f64(p, "bottomRadius"))
                .unwrap_or(16.0);
            let height = props
                .and_then(|p| prop_track_f64(p, "height"))
                .unwrap_or(32.0);
            let detail = props
                .and_then(|p| prop_track_f64(p, "segments"))
                .map_or(24, |v| v.round().max(3.0) as u32);
            // Zoidium Cone `endCaps`: 0 = closed, 1 = open (opposite of CM3 Cylinder!).
            let end_caps = props
                .and_then(|p| p.get("endCaps"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Cone",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::Cone {
                    radius,
                    height,
                    detail,
                    open_ended: end_caps != 0,
                    top_radius,
                    shading,
                }),
            )
        }
        "zoidium:geometry-plus/capsule" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let radius = props
                .and_then(|p| prop_track_f64(p, "radius"))
                .unwrap_or(8.0);
            let total_height = props
                .and_then(|p| prop_track_f64(p, "height"))
                .unwrap_or(32.0);
            let length = (total_height - 2.0 * radius).max(0.0);
            let cap_segments = props
                .and_then(|p| prop_track_f64(p, "segments"))
                .map_or(8, |v| v.round().max(1.0) as u32);
            let radial_segments = props
                .and_then(|p| prop_track_f64(p, "radialSegments"))
                .map_or(16, |v| v.round().max(3.0) as u32);
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Capsule",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::Capsule {
                    radius,
                    length,
                    cap_segments,
                    radial_segments,
                    shading,
                }),
            )
        }
        "zoidium:geometry-plus/tube" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let outer_radius = props
                .and_then(|p| prop_track_f64(p, "outerRadius"))
                .unwrap_or(12.0);
            let inner_radius = props
                .and_then(|p| prop_track_f64(p, "innerRadius"))
                .unwrap_or(8.0);
            let height = props
                .and_then(|p| prop_track_f64(p, "height"))
                .unwrap_or(24.0);
            let radial_segments = props
                .and_then(|p| prop_track_f64(p, "radialSegments"))
                .map_or(24, |v| v.round().max(3.0) as u32);
            let end_caps = props
                .and_then(|p| p.get("endCaps"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Tube",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::Tube {
                    outer_radius,
                    inner_radius,
                    height,
                    radial_segments,
                    open_ended: end_caps != 0,
                    shading,
                }),
            )
        }
        "zoidium:geometry-plus/gear" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let teeth = props
                .and_then(|p| prop_track_f64(p, "teeth"))
                .map_or(12, |v| v.round().clamp(3.0, 64.0) as u32);
            let inner_radius = props
                .and_then(|p| prop_track_f64(p, "innerRadius"))
                .unwrap_or(10.0);
            let outer_radius = props
                .and_then(|p| prop_track_f64(p, "outerRadius"))
                .unwrap_or(16.0);
            let thickness = props
                .and_then(|p| prop_track_f64(p, "thickness"))
                .unwrap_or(6.0);
            let tooth_width = props
                .and_then(|p| prop_track_f64(p, "toothWidth"))
                .unwrap_or(0.55);
            let bevel_size = props
                .and_then(|p| prop_track_f64(p, "bevelSize"))
                .unwrap_or(0.3);
            let bevel_segments = props
                .and_then(|p| prop_track_f64(p, "bevelSegments"))
                .map_or(2, |v| v.round().max(0.0) as u32);
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Gear",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::Gear {
                    teeth,
                    inner_radius,
                    outer_radius,
                    thickness,
                    tooth_width,
                    bevel_size,
                    bevel_segments,
                    shading,
                }),
            )
        }
        "zoidium:geometry-plus/helix" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let radius = props
                .and_then(|p| prop_track_f64(p, "radius"))
                .unwrap_or(14.0);
            let height = props
                .and_then(|p| prop_track_f64(p, "height"))
                .unwrap_or(32.0);
            let turns = props
                .and_then(|p| prop_track_f64(p, "turns"))
                .unwrap_or(3.0);
            let tube_radius = props
                .and_then(|p| prop_track_f64(p, "tubeRadius"))
                .unwrap_or(2.0);
            let path_segments = props
                .and_then(|p| prop_track_f64(p, "pathSegments"))
                .map_or(64, |v| v.round().max(8.0) as u32);
            let radial_segments = props
                .and_then(|p| prop_track_f64(p, "radialSegments"))
                .map_or(8, |v| v.round().max(3.0) as u32);
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Helix",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::Helix {
                    radius,
                    height,
                    turns,
                    tube_radius,
                    path_segments,
                    radial_segments,
                    shading,
                }),
            )
        }
        "zoidium:geometry-plus/spline" => {
            plugins.insert("geometry-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            preserve_animated_geometry_props(props, extra_preserved);
            let default_pts: [[f64; 3]; 3] =
                [[-20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [20.0, 0.0, 0.0]];
            let point_count = props
                .and_then(|p| prop_track_f64(p, "points"))
                .map_or(3, |v| v.round().clamp(2.0, 32.0) as usize);
            let mut points = Vec::with_capacity(point_count);
            for idx in 1..=point_count {
                let fallback = default_pts.get(idx - 1).copied().unwrap_or([0.0, 0.0, 0.0]);
                let pt = props
                    .and_then(|p| prop_track_vec3(p, &format!("point{idx}")))
                    .unwrap_or(fallback);
                points.push(pt);
            }
            let interpolation = props
                .and_then(|p| prop_str(p, "interpolation"))
                .unwrap_or_else(|| "Cubic".to_owned());
            let closed = props
                .and_then(|p| p.get("closed"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0)
                != 0;
            let start = props
                .and_then(|p| prop_track_f64(p, "start"))
                .unwrap_or(0.0);
            let end = props.and_then(|p| prop_track_f64(p, "end")).unwrap_or(1.0);
            let thickness = props
                .and_then(|p| prop_track_f64(p, "thickness"))
                .unwrap_or(2.0);
            let thickness_map = props
                .and_then(|p| p.get("thicknessMap"))
                .map(parse_gradient_stops)
                .filter(|v| !v.is_empty())
                .unwrap_or_else(GradientStop::white_default);
            let gradient_space = props
                .and_then(|p| prop_str(p, "gradientSpace"))
                .unwrap_or_else(|| "Curve".to_owned());
            let tension = props
                .and_then(|p| prop_track_f64(p, "tension"))
                .unwrap_or(1.0);
            let path_segments = props
                .and_then(|p| prop_track_f64(p, "pathSegments"))
                .map_or(16, |v| v.round().max(1.0) as u32);
            let radial_segments = props
                .and_then(|p| prop_track_f64(p, "radialSegments"))
                .map_or(8, |v| v.round().max(3.0) as u32);
            let caps = props
                .and_then(|p| prop_str(p, "caps"))
                .unwrap_or_else(|| "Both".to_owned());
            let cap_style = props
                .and_then(|p| prop_str(p, "capStyle"))
                .unwrap_or_else(|| "Flat".to_owned());
            let shading = props
                .and_then(|p| p.get("shading"))
                .and_then(prop_track_u32_val)
                .unwrap_or(0);
            (
                "Spline",
                [0.0, 0.0, 0.0],
                Object3DKind::Shape3D(Shape3DKind::Spline {
                    points,
                    interpolation,
                    closed,
                    start,
                    end,
                    thickness,
                    thickness_map,
                    gradient_space,
                    tension,
                    path_segments,
                    radial_segments,
                    caps,
                    cap_style,
                    shading,
                }),
            )
        }
        // --- ExpressText+ (12 container/effector classes) ---
        "zoidium:expresstext-plus/expresstext"
        | "zoidium:expresstext-plus/shake"
        | "zoidium:expresstext-plus/plain"
        | "zoidium:expresstext-plus/random"
        | "zoidium:expresstext-plus/formula"
        | "zoidium:expresstext-plus/delay"
        | "zoidium:expresstext-plus/spline"
        | "zoidium:expresstext-plus/orbit"
        | "zoidium:expresstext-plus/repeater"
        | "zoidium:expresstext-plus/twist"
        | "zoidium:expresstext-plus/wrap"
        | "zoidium:expresstext-plus/voronoi" => {
            plugins.insert("expresstext-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            let custom_properties = parse_custom_properties(obj.get("customProperties"));
            let mut children = Vec::new();
            if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                for child in arr {
                    children.push(parse_object3d(child, plugins, warnings));
                }
            }
            if let Some(ep) = obj.get("expressProperties") {
                extra_preserved.insert("expressProperties".to_owned(), ep.clone());
            }
            if let Some(lm) = obj.get("lettersMode") {
                extra_preserved.insert("lettersMode".to_owned(), lm.clone());
            }
            let label = match type_str {
                "zoidium:expresstext-plus/expresstext" => "ExpressText+",
                "zoidium:expresstext-plus/shake" => "Shake",
                "zoidium:expresstext-plus/plain" => "Plain",
                "zoidium:expresstext-plus/random" => "Random",
                "zoidium:expresstext-plus/formula" => "Formula",
                "zoidium:expresstext-plus/delay" => "Delay",
                "zoidium:expresstext-plus/spline" => "Spline",
                "zoidium:expresstext-plus/orbit" => "Orbit",
                "zoidium:expresstext-plus/repeater" => "Repeater",
                "zoidium:expresstext-plus/twist" => "Twist",
                "zoidium:expresstext-plus/wrap" => "Wrap",
                "zoidium:expresstext-plus/voronoi" => "Voronoi",
                _ => "ExpressText+",
            };
            (
                label,
                [0.0, 0.0, 0.0],
                Object3DKind::Group {
                    reflection_visibility: 0,
                    custom_properties,
                    children,
                },
            )
        }
        // --- Optical Flares ---
        "zoidium:optical-flares/flare"
        | "zoidium:optical-flares/optical-flare"
        | "zoidium:optical-flares/optflares" => {
            plugins.insert("optical-flares".to_owned());
            *plugin_object_type = Some("zoidium:optical-flares/flare".to_owned());
            (
                "Optical Flares",
                [0.0, 0.0, 0.0],
                Object3DKind::Dummy {
                    original_type: Some(type_str.to_owned()),
                    payload: raw_val.clone(),
                },
            )
        }
        // --- Text+ (7 parent group classes) ---
        "zoidium:text-plus/text-plus"
        | "zoidium:text-plus/gradient-transform"
        | "zoidium:text-plus/character-shake"
        | "zoidium:text-plus/selected-transform"
        | "zoidium:text-plus/selected-shake"
        | "zoidium:text-plus/random-scatter"
        | "zoidium:text-plus/character-material" => {
            plugins.insert("text-plus".to_owned());
            *plugin_object_type = Some(type_str.to_owned());
            let custom_properties = parse_custom_properties(obj.get("customProperties"));
            let mut children = Vec::new();
            if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                for child in arr {
                    children.push(parse_object3d(child, plugins, warnings));
                }
            }
            if let Some(fm) = obj.get("frontMaterial").filter(|v| !v.is_null()) {
                let (mat, _) = parse_material(fm, plugins);
                material_slots.insert("front".to_owned(), mat);
            }
            if let Some(sm) = obj.get("sideMaterial").filter(|v| !v.is_null()) {
                let (mat, _) = parse_material(sm, plugins);
                material_slots.insert("side".to_owned(), mat);
            }
            (
                "Text+",
                [0.0, 0.0, 0.0],
                Object3DKind::Group {
                    reflection_visibility: 0,
                    custom_properties,
                    children,
                },
            )
        }
        // --- Repeater & Effector+ (11 parent group classes) ---
        "zoidium:repeater/repeater"
        | "zoidium:repeater/linear-repeater"
        | "zoidium:repeater/random-repeater"
        | "zoidium:repeater/echo-repeater"
        | "zoidium:repeater/radial"
        | "zoidium:repeater/radial-repeater"
        | "zoidium:repeater/grid"
        | "zoidium:repeater/grid-repeater"
        | "zoidium:repeater/twist"
        | "zoidium:repeater/warp"
        | "zoidium:repeater/voronoi-fracture"
        | "zoidium:repeater/plain"
        | "zoidium:repeater/delay" => {
            plugins.insert("repeater".to_owned());
            let canonical_type = match type_str {
                "zoidium:repeater/radial-repeater" => "zoidium:repeater/radial",
                "zoidium:repeater/grid-repeater" => "zoidium:repeater/grid",
                other => other,
            };
            *plugin_object_type = Some(canonical_type.to_owned());
            // Zoidium project-compatibility.js:52-55: twist & warp default zoidiumSubdivisionMode to "legacy" when omitted.
            if matches!(type_str, "zoidium:repeater/twist" | "zoidium:repeater/warp")
                && !obj.contains_key("zoidiumSubdivisionMode")
            {
                extra_preserved.insert(
                    "zoidiumSubdivisionMode".to_owned(),
                    Value::String("legacy".to_owned()),
                );
            }
            let custom_properties = parse_custom_properties(obj.get("customProperties"));
            let mut children = Vec::new();
            if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                for child in arr {
                    children.push(parse_object3d(child, plugins, warnings));
                }
            }
            let label = match canonical_type {
                "zoidium:repeater/radial" => "Radial Repeater",
                "zoidium:repeater/grid" => "Grid Repeater",
                "zoidium:repeater/twist" => "Twist",
                "zoidium:repeater/warp" => "Warp",
                "zoidium:repeater/voronoi-fracture" => "Voronoi Fracture",
                "zoidium:repeater/plain" => "Plain Effector",
                "zoidium:repeater/delay" => "Delay Effector",
                _ => "Repeater",
            };
            (
                label,
                [0.0, 0.0, 0.0],
                Object3DKind::Group {
                    reflection_visibility: 0,
                    custom_properties,
                    children,
                },
            )
        }
        // --- Trapcode Suite (8 lights + 3 other 3D objects, accepting both `trapcode-suite/` and `trapcode/`) ---
        s if s.starts_with("zoidium:trapcode-suite/") || s.starts_with("zoidium:trapcode/") => {
            plugins.insert("trapcode-suite".to_owned());
            *plugin_object_type = Some(s.to_owned());
            let sub = s
                .strip_prefix("zoidium:trapcode-suite/")
                .or_else(|| s.strip_prefix("zoidium:trapcode/"))
                .unwrap_or(s);
            let light_sub_type = match sub {
                "spot-light" => Some((1u32, "Spot Light")),
                "point-light" => Some((2, "Point Light")),
                "infinite-light" => Some((3, "Infinite Light")),
                "hemisphere-light" => Some((4, "Hemisphere Light")),
                "sun-light" => Some((3, "Sun Light")),
                "area-light" => Some((2, "Area Light")),
                "ies-light" => Some((2, "IES Light")),
                "portal-light" => Some((2, "Portal Light")),
                _ => None,
            };
            if let Some((lt, name)) = light_sub_type {
                let mut light = parse_light_data(props, lt);
                if let Some(p) = props {
                    if let Some(sky) = p.get("skyColor") {
                        light.color =
                            parse_property_track(sky, PropertyValue::Color([1.0, 1.0, 1.0]));
                    }
                    if let Some(d) = prop_track_f64(p, "distance") {
                        light.distance = d;
                    }
                    if let Some(d) = prop_track_f64(p, "decay") {
                        light.decay = d;
                    }
                    if let Some(pen) = prop_track_f64(p, "penumbra") {
                        light.penumbra = pen;
                    }
                }
                (name, [0.0, 10.0, 0.0], Object3DKind::Light(Box::new(light)))
            } else if obj.get("objects").and_then(Value::as_array).is_some() {
                let custom_properties = parse_custom_properties(obj.get("customProperties"));
                let mut children = Vec::new();
                if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                    for child in arr {
                        children.push(parse_object3d(child, plugins, warnings));
                    }
                }
                let label = match sub {
                    "particular" => "Particular",
                    "form" => "Form",
                    "plexus" => "Plexus",
                    _ => "Plugin Group",
                };
                (
                    label,
                    [0.0, 0.0, 0.0],
                    Object3DKind::Group {
                        reflection_visibility: 0,
                        custom_properties,
                        children,
                    },
                )
            } else {
                let label = match sub {
                    "particular" => "Particular",
                    "form" => "Form",
                    "plexus" => "Plexus",
                    _ => "3D Object",
                };
                (
                    label,
                    [0.0, 0.0, 0.0],
                    Object3DKind::Dummy {
                        original_type: Some(s.to_owned()),
                        payload: raw_val.clone(),
                    },
                )
            }
        }
        s if s.starts_with("zoidium:particles-plus/") || s.eq_ignore_ascii_case("particles") => {
            if s.starts_with("zoidium:particles-plus/")
                || props.is_some_and(|p| p.contains_key("turbulence") || p.contains_key("drag"))
                || obj.contains_key("turbulence")
                || obj.contains_key("drag")
            {
                plugins.insert("particles-plus".to_owned());
            }
            if s.starts_with("zoidium:particles-plus/") {
                *plugin_object_type = Some(s.to_owned());
            }
            let mut particles = parse_particles_data(props);
            particles.sprite_preset = obj
                .get("_zoidiumParticlesPlusSprite")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(ToOwned::to_owned);
            (
                "Particles",
                [0.0, 0.0, 0.0],
                Object3DKind::Particles(Box::new(particles)),
            )
        }
        other => {
            if let Some((plugin_id, _)) = other
                .strip_prefix("zoidium:")
                .and_then(|r| r.split_once('/'))
                .filter(|(p, b)| !p.is_empty() && !b.is_empty())
            {
                plugins.insert(plugin_id.to_owned());
            }
            if let Some(arr) = obj.get("objects").and_then(Value::as_array) {
                *plugin_object_type = Some(other.to_owned());
                let custom_properties = parse_custom_properties(obj.get("customProperties"));
                let mut children = Vec::new();
                for child in arr {
                    children.push(parse_object3d(child, plugins, warnings));
                }
                warnings.push(format!(
                    "preserved unknown container 3D object type {other:?} as Group"
                ));
                (
                    "Plugin Group",
                    [0.0, 0.0, 0.0],
                    Object3DKind::Group {
                        reflection_visibility: 0,
                        custom_properties,
                        children,
                    },
                )
            } else {
                warnings.push(format!(
                    "preserved unknown 3D object type {other:?} as Dummy"
                ));
                (
                    "3D Object",
                    [0.0, 0.0, 0.0],
                    Object3DKind::Dummy {
                        original_type: Some(other.to_owned()),
                        payload: raw_val.clone(),
                    },
                )
            }
        }
    }
}

pub(super) fn is_animated_pz_track_value(val: &Value) -> bool {
    let Some(obj) = val.as_object() else {
        return false;
    };
    if obj.get("animated").and_then(Value::as_bool) == Some(true) {
        return true;
    }
    if obj
        .get("expression")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.trim().is_empty())
    {
        return true;
    }
    if obj
        .get("keyframes")
        .and_then(Value::as_array)
        .is_some_and(|kfs| kfs.len() > 1)
    {
        return true;
    }
    if let Some(channels) = obj.get("objects").and_then(Value::as_array) {
        return channels.iter().any(is_animated_pz_track_value);
    }
    false
}

pub(super) fn preserve_animated_geometry_props(
    props: Option<&Map<String, Value>>,
    extra_preserved: &mut BTreeMap<String, Value>,
) {
    let Some(p) = props else {
        return;
    };
    for (k, v) in p {
        if matches!(
            k.as_str(),
            "name" | "enabled" | "position" | "rotation" | "scale" | "eulerOrder"
        ) {
            continue;
        }
        if is_animated_pz_track_value(v) {
            extra_preserved.insert(format!("property.{k}"), v.clone());
        }
    }
}

pub(super) fn prop_track_f64(props: &Map<String, Value>, key: &str) -> Option<f64> {
    let val = props.get(key)?;
    if let Some(n) = value_as_f64(val) {
        return Some(n);
    }
    let track = parse_property_track(val, PropertyValue::Number(0.0));
    track.value.as_f64()
}

pub(super) fn prop_track_vec3(props: &Map<String, Value>, key: &str) -> Option<[f64; 3]> {
    let val = props.get(key)?;
    if let Some(v) = value_as_f64_array::<3>(val) {
        return Some(v);
    }
    let track = parse_property_track(val, PropertyValue::Vec3([0.0, 0.0, 0.0]));
    track.value.as_vec3()
}

pub(super) fn prop_track_u32_val(val: &Value) -> Option<u32> {
    if let Some(n) = value_as_u32(val) {
        return Some(n);
    }
    let track = parse_property_track(val, PropertyValue::Option(0));
    track.value.as_option()
}

pub(super) fn is_known_object3d_prop(key: &str) -> bool {
    matches!(
        key,
        "name"
            | "enabled"
            | "position"
            | "rotation"
            | "scale"
            | "eulerOrder"
            | "geometryProperties"
            | "modelProperties"
            | "shake"
            | "shakeAmplitude"
            | "shakeSpeed"
            | "active"
            | "projection"
            | "focalLength"
            | "filmGate"
            | "zoom"
            | "filmOffsetX"
            | "filmOffsetY"
            | "fov"
            | "near"
            | "far"
            | "depthOfField"
            | "vibrate"
            | "motionBlur"
            | "dof"
            | "dofFocusDistance"
            | "dofAperture"
            | "dofFocusAreaWidth"
            | "dofNearBlurLevel"
            | "dofFarBlurLevel"
            | "reflectionVisibility"
            | "text"
            | "font"
            | "size"
            | "detail"
            | "bevel"
            | "bevelSize"
            | "positionMode"
            | "spacing"
            | "characterIndexOffset"
            | "characterCountOverride"
            | "bevelSide"
            | "bevelDetail"
            | "bevelProfile"
            | "bevelTension"
            | "zoidiumTextScaleMode"
            | "zoidiumUvMode"
            | "zoidiumShadowMode"
            | "zoidiumCastShadows"
            | "zoidiumId"
            | "zoidiumTrackTo"
            | "zoidiumTrackAxis"
            | "zoidiumUpAxis"
            | "zoidiumInfluence"
            | "zoidiumFollow"
            | "zoidiumFollowSpace"
            | "zoidiumFollowPosition"
            | "zoidiumFollowRotation"
            | "zoidiumFollowScale"
            | "zoidiumFollowPositionOffset"
            | "zoidiumFollowRotationOffset"
            | "zoidiumFollowScaleOffset"
            | "zoidiumFollowFallback"
            | "color"
            | "groundColor"
            | "target"
            | "intensity"
            | "angle"
            | "time"
            | "number"
            | "rate"
            | "lifetime"
            | "pdist"
            | "ipos"
            | "pspread"
            | "iradius"
            | "rspread"
            | "vdist"
            | "ivel"
            | "vspread"
            | "irvel"
            | "rvspread"
            | "accel"
            | "iang"
            | "aspread"
            | "angvel"
            | "avspread"
            | "blending"
            | "texture"
    )
}

pub(super) fn parse_shape3d_object(
    props: Option<&Map<String, Value>>,
    sub_type: u32,
    raw_obj: &Value,
    warnings: &mut Vec<String>,
) -> (&'static str, [f64; 3], Object3DKind) {
    let gp = props
        .and_then(|p| p.get("geometryProperties"))
        .and_then(Value::as_object);

    let (name, shape) = match sub_type {
        1 => {
            let size = gp
                .and_then(|g| prop_vec3(g, "box_size"))
                .or_else(|| {
                    props.and_then(|p| prop_vec3(p, "box_size").or_else(|| prop_vec3(p, "size")))
                })
                .filter(|s| s.iter().all(|v| v.is_finite() && *v > 0.0))
                .unwrap_or([10.0, 10.0, 10.0]);
            ("Box", Shape3DKind::Box { size })
        }
        2 => {
            let size = gp
                .and_then(|g| prop_vec3(g, "cylinder_size"))
                .or_else(|| {
                    props.and_then(|p| {
                        prop_vec3(p, "cylinder_size").or_else(|| prop_vec3(p, "size"))
                    })
                })
                .filter(|s| s.iter().all(|v| v.is_finite() && *v > 0.0))
                .unwrap_or([10.0, 10.0, 10.0]);
            let detail = gp
                .and_then(|g| prop_f64(g, "cylinder_detail"))
                .map_or(10, |v| v.round().max(0.0) as u32);
            // CRITICAL CM3 INVERSION: `cylinder_openEnded == 0` means open (`open_ended = true`),
            // `1` means closed (`open_ended = false`).
            let raw_open = gp
                .and_then(|g| g.get("cylinder_openEnded"))
                .and_then(value_as_u32)
                .unwrap_or(0);
            (
                "Cylinder",
                Shape3DKind::Cylinder {
                    top_radius: size[0],
                    bottom_radius: size[1],
                    height: size[2],
                    detail,
                    open_ended: raw_open == 0,
                },
            )
        }
        3 => {
            let size = gp
                .and_then(|g| prop_vec2(g, "rect_size"))
                .or_else(|| {
                    props.and_then(|p| prop_vec2(p, "rect_size").or_else(|| prop_vec2(p, "size")))
                })
                .filter(|s| s.iter().all(|v| v.is_finite() && *v > 0.0))
                .unwrap_or([10.0, 10.0]);
            ("Rectangle", Shape3DKind::Rectangle { size })
        }
        4 => {
            let radius = gp
                .and_then(|g| prop_f64(g, "circle_size"))
                .or_else(|| {
                    props.and_then(|p| prop_f64(p, "circle_size").or_else(|| prop_f64(p, "radius")))
                })
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(10.0);
            let detail = gp
                .and_then(|g| prop_f64(g, "circle_detail"))
                .map_or(10, |v| v.round().max(1.0) as u32);
            ("Circle", Shape3DKind::Circle { radius, detail })
        }
        5 => {
            let radius = gp
                .and_then(|g| prop_f64(g, "sphere_size"))
                .or_else(|| {
                    props.and_then(|p| prop_f64(p, "sphere_size").or_else(|| prop_f64(p, "radius")))
                })
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(10.0);
            let det = gp
                .and_then(|g| prop_vec2(g, "sphere_detail"))
                .unwrap_or([10.0, 10.0]);
            (
                "Sphere",
                Shape3DKind::Sphere {
                    radius,
                    width_segments: det[0].round().max(1.0) as u32,
                    height_segments: det[1].round().max(1.0) as u32,
                },
            )
        }
        6 => {
            let sz = gp
                .and_then(|g| prop_vec2(g, "donut_size"))
                .unwrap_or([10.0, 10.0]);
            let det = gp
                .and_then(|g| prop_vec2(g, "donut_detail"))
                .unwrap_or([10.0, 10.0]);
            (
                "Donut",
                Shape3DKind::Donut {
                    radius: sz[0],
                    tube: sz[1],
                    radial_segments: det[0].round().max(1.0) as u32,
                    tubular_segments: det[1].round().max(1.0) as u32,
                },
            )
        }
        7 => {
            let size = gp
                .and_then(|g| prop_vec3(g, "path_size"))
                .unwrap_or([10.0, 10.0, 10.0]);
            let thickness = gp
                .and_then(|g| prop_f64(g, "path_thickness"))
                .unwrap_or(2.0);
            let loops = gp
                .and_then(|g| prop_f64(g, "path_loops"))
                .map_or(5, |v| v.round().max(1.0) as u32);
            let det = gp
                .and_then(|g| prop_vec2(g, "path_detail"))
                .unwrap_or([10.0, 10.0]);
            (
                "Wire",
                Shape3DKind::Wire {
                    size,
                    thickness,
                    loops,
                    inner_segments: det[0].round().max(1.0) as u32,
                    outer_segments: det[1].round().max(1.0) as u32,
                },
            )
        }
        99 => {
            let asset_id = gp
                .and_then(|g| g.get("customGeometry"))
                .or_else(|| props.and_then(|p| p.get("customGeometry")))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            ("Geometry", Shape3DKind::CustomGeometry { asset_id })
        }
        other => {
            warnings.push(format!(
                "preserved unknown 3D shape objectType {other} as Dummy"
            ));
            return (
                "Shape",
                [0.0, 0.0, 0.0],
                Object3DKind::Dummy {
                    original_type: Some(format!("shape:{other}")),
                    payload: raw_obj.clone(),
                },
            );
        }
    };

    (name, [0.0, 0.0, 0.0], Object3DKind::Shape3D(shape))
}

pub(super) fn parse_text3d_data(
    obj: &Map<String, Value>,
    props: Option<&Map<String, Value>>,
    plugins: &mut BTreeSet<String>,
) -> Text3DData {
    let mut data = Text3DData {
        text_scale_mode: CompatMode::Legacy,
        ..Text3DData::default()
    };

    // Zoidium project-compatibility.js:49-51 stores `zoidiumTextScaleMode` at the top level of the Text3D object.
    if let Some(mode) = obj
        .get("zoidiumTextScaleMode")
        .and_then(Value::as_str)
        .or_else(|| {
            props
                .and_then(|p| p.get("zoidiumTextScaleMode"))
                .and_then(Value::as_str)
        })
    {
        data.text_scale_mode = CompatMode::from_token(mode);
    }

    let Some(props) = props else {
        return data;
    };

    if let Some(t) = prop_str(props, "text") {
        data.text = t;
    }
    if let Some(f) = props.get("font") {
        data.font = f.as_str().map(ToOwned::to_owned);
    }
    if let Some(sz) = prop_vec2(props, "size") {
        data.size = sz;
    } else if let Some(s) = prop_f64(props, "size") {
        data.size[0] = s;
    }
    if let Some(th) = prop_f64(props, "thickness")
        .or_else(|| prop_f64(props, "depth"))
        .or_else(|| prop_f64(props, "height"))
    {
        data.size[1] = th;
    }
    if let Some(d) = prop_f64(props, "detail").or_else(|| prop_f64(props, "curveSegments")) {
        data.detail = d.round().max(1.0) as u32;
    }
    if let Some(b) = props
        .get("bevel")
        .or_else(|| props.get("bevelEnabled"))
        .and_then(value_as_u32)
    {
        data.bevel = b != 0;
    }
    // CRITICAL CM3 MAPPING QUIRK: `bevelSize[0]` => `bevel_thickness`, `bevelSize[1]` => `bevel_size`.
    // Also accept community scalar `bevelSize` + `bevelThickness`.
    if let Some(bs) = prop_vec2(props, "bevelSize") {
        data.bevel_thickness = bs[0];
        data.bevel_size = bs[1];
    } else if let Some(bs) = prop_f64(props, "bevelSize") {
        data.bevel_size = bs;
    }
    if let Some(bt) = prop_f64(props, "bevelThickness") {
        data.bevel_thickness = bt;
    }
    if let Some(pm) = props.get("positionMode").and_then(value_as_u32) {
        data.position_mode = Text3DAlign::from_cm3_id(pm);
    }

    // 7 Text+ properties injected onto native type 1 Text3D (plugins/text-plus/text-plus.js:360-448).
    let mut has_text_plus = false;
    if let Some(v) = prop_f64(props, "spacing") {
        data.spacing = v;
        has_text_plus = true;
    }
    if let Some(v) = props.get("characterIndexOffset") {
        data.character_index_offset = parse_property_track(v, PropertyValue::Number(0.0));
        has_text_plus = true;
    }
    if let Some(v) = props.get("characterCountOverride") {
        data.character_count_override = parse_property_track(v, PropertyValue::Number(0.0));
        has_text_plus = true;
    }
    if let Some(v) = props.get("bevelSide").and_then(value_as_u32) {
        data.bevel_side = v;
        has_text_plus = true;
    }
    if let Some(v) = prop_f64(props, "bevelDetail") {
        data.bevel_detail = v.round().max(1.0) as u32;
        has_text_plus = true;
    }
    if let Some(v) = props.get("bevelProfile").and_then(value_as_u32) {
        data.bevel_profile = v;
        has_text_plus = true;
    }
    if let Some(v) = prop_f64(props, "bevelTension") {
        data.bevel_tension = v;
        has_text_plus = true;
    }
    if has_text_plus {
        plugins.insert("text-plus".to_owned());
    }

    data
}

pub(super) fn parse_light_data(props: Option<&Map<String, Value>>, sub_type: u32) -> LightData {
    let kind = match sub_type {
        2 => LightKind::Point,
        3 => LightKind::Directional,
        4 => LightKind::Hemisphere,
        5 => LightKind::Ambient,
        _ => LightKind::Spot,
    };
    let mut light = LightData::for_kind(kind);
    let Some(props) = props else {
        return light;
    };
    if let Some(v) = props.get("color") {
        light.color = parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0]));
    }
    if let Some(v) = props.get("groundColor") {
        light.ground_color = parse_property_track(v, PropertyValue::Color([1.0, 1.0, 1.0]));
    }
    if let Some(v) = props.get("target") {
        light.target = parse_property_track(v, PropertyValue::Vec3([0.0, 0.0, 0.0]));
    }
    if let Some(v) = props.get("intensity") {
        light.intensity = parse_property_track(v, PropertyValue::Number(1.0));
    }
    if let Some(v) = props.get("angle") {
        light.angle = parse_property_track(v, PropertyValue::Number(60.0));
    }
    if let Some(v) = props.get("zoidiumCastShadows").and_then(value_as_u32) {
        light.cast_shadow = v != 0;
    }
    if let Some(v) = prop_track_f64(props, "penumbra") {
        light.penumbra = v;
    }
    if let Some(v) = prop_track_f64(props, "decay") {
        light.decay = v;
    }
    if let Some(v) = prop_track_f64(props, "distance") {
        light.distance = v;
    }
    light
}

pub(super) fn parse_camera_data(
    props: Option<&Map<String, Value>>,
    sub_type: u32,
    plugins: &mut BTreeSet<String>,
) -> CameraData {
    let mut cam = CameraData {
        kind: if sub_type == 2 {
            CameraKind::Orthographic
        } else {
            CameraKind::Perspective
        },
        ..CameraData::default()
    };
    let Some(props) = props else {
        return cam;
    };

    if let Some(v) = props.get("active").and_then(value_as_bool) {
        cam.active = v;
    }
    if let Some(proj) = prop_str(props, "projection") {
        plugins.insert("camera-plus".to_owned());
        cam.kind = if proj.eq_ignore_ascii_case("orthographic") {
            CameraKind::Orthographic
        } else {
            CameraKind::Perspective
        };
    }
    if let Some(v) = props.get("focalLength") {
        plugins.insert("camera-plus".to_owned());
        cam.focal_length = parse_property_track(v, PropertyValue::Number(35.0));
    }
    if let Some(fg) = prop_f64(props, "filmGate") {
        plugins.insert("camera-plus".to_owned());
        cam.film_gate = fg;
    }
    if let Some(v) = props.get("fov") {
        cam.fov = parse_property_track(v, PropertyValue::Number(60.0));
    } else if props.contains_key("focalLength") || props.contains_key("filmGate") {
        // Derive vertical FOV from horizontal filmGate & focalLength at 16:9 default aspect.
        let fl = cam.focal_length.value.as_f64().unwrap_or(35.0).max(1.0);
        let gate_h = cam.film_gate / (16.0 / 9.0);
        let v_fov_deg = 2.0 * (gate_h / (2.0 * fl)).atan().to_degrees();
        if v_fov_deg.is_finite() && v_fov_deg > 0.0 {
            cam.fov = PropertyTrack::number(v_fov_deg);
        }
    }
    if let Some(near) = prop_f64(props, "near") {
        cam.near = near;
    }
    if let Some(far) = prop_f64(props, "far") {
        cam.far = far;
    }
    if let Some(v) = props.get("zoom") {
        plugins.insert("camera-plus".to_owned());
        cam.zoom = parse_property_track(v, PropertyValue::Number(1.0));
    }
    if let Some(v) = props.get("filmOffsetX") {
        plugins.insert("camera-plus".to_owned());
        cam.film_offset_x = parse_property_track(v, PropertyValue::Number(0.0));
    }
    if let Some(v) = props.get("filmOffsetY") {
        plugins.insert("camera-plus".to_owned());
        cam.film_offset_y = parse_property_track(v, PropertyValue::Number(0.0));
    }

    if let Some(v) = props.get("shakeAmplitude") {
        cam.shake_amplitude = parse_property_track(v, PropertyValue::Number(0.0));
    }
    if let Some(v) = props.get("shakeSpeed") {
        cam.shake_speed = parse_property_track(v, PropertyValue::Number(30.0));
    }
    if let Some(shake_obj) = props.get("shake").and_then(Value::as_object) {
        let mut s = CameraShakeParams::default();
        if let Some(m) = shake_obj.get("mode").and_then(value_as_u32) {
            s.mode = m;
        }
        if let Some(v) = prop_vec4(shake_obj, "amplitude") {
            s.amplitude = v;
        }
        if let Some(v) = prop_vec4(shake_obj, "frequency") {
            s.frequency = v;
        }
        if let Some(v) = prop_vec4(shake_obj, "phase") {
            s.phase = v;
        }
        if let Some(v) = prop_vec4(shake_obj, "noise") {
            s.noise = v;
        }
        if let Some(v) = prop_vec4(shake_obj, "smooth") {
            s.smooth = v;
        }
        cam.shake = s;
    }

    // Camera+ nested `depthOfField` or legacy flat `dof*` keys (camera-runtime.js:456-465).
    let has_legacy_dof = props.contains_key("dof")
        || props.contains_key("dofFocusDistance")
        || props.contains_key("dofAperture")
        || props.contains_key("dofFocusAreaWidth")
        || props.contains_key("dofNearBlurLevel")
        || props.contains_key("dofFarBlurLevel");
    if let Some(dof_obj) = props.get("depthOfField").and_then(Value::as_object) {
        plugins.insert("camera-plus".to_owned());
        cam.dof = Some(parse_camera_dof_map(dof_obj, props));
    } else if has_legacy_dof {
        plugins.insert("camera-plus".to_owned());
        let empty = Map::new();
        cam.dof = Some(parse_camera_dof_map(&empty, props));
    }

    // Camera+ nested `vibrate` object (plugins/camera-plus/vibrate.js:21-35).
    if let Some(vib_obj) = props.get("vibrate").and_then(Value::as_object) {
        plugins.insert("camera-plus".to_owned());
        let mut vib = CameraVibrateParams::default();
        if let Some(v) = vib_obj.get("enabled").and_then(value_as_bool) {
            vib.enabled = v;
        }
        if let Some(v) = vib_obj.get("regularPulse").and_then(value_as_bool) {
            vib.regular_pulse = v;
        }
        if let Some(v) = vib_obj.get("relative").and_then(value_as_bool) {
            vib.relative = v;
        }
        if let Some(v) = vib_obj.get("seed").and_then(value_as_u32) {
            vib.seed = v;
        }
        if let Some(v) = vib_obj.get("enablePosition").and_then(value_as_bool) {
            vib.enable_position = v;
        }
        if let Some(v) = vib_obj.get("positionAmplitude") {
            vib.position_amplitude = parse_property_track(v, PropertyValue::Vec3([0.0, 0.0, 0.0]));
        }
        if let Some(v) = vib_obj.get("positionFrequency") {
            vib.position_frequency = parse_property_track(v, PropertyValue::Number(2.0));
            vib.frequency = vib.position_frequency.clone();
        }
        if let Some(v) = vib_obj.get("enableRotation").and_then(value_as_bool) {
            vib.enable_rotation = v;
        }
        if let Some(v) = vib_obj.get("rotationAmplitude") {
            vib.rotation_amplitude = parse_property_track(v, PropertyValue::Vec3([0.0, 0.0, 0.0]));
        }
        if let Some(v) = vib_obj.get("rotationFrequency") {
            vib.rotation_frequency = parse_property_track(v, PropertyValue::Number(2.0));
        }
        if let Some(v) = vib_obj.get("frequency")
            && !vib_obj.contains_key("positionFrequency")
        {
            vib.frequency = parse_property_track(v, PropertyValue::Number(2.0));
            vib.position_frequency = vib.frequency.clone();
            if !vib_obj.contains_key("rotationFrequency") {
                vib.rotation_frequency = vib.frequency.clone();
            }
        }
        cam.vibrate = Some(vib);
    }

    // Camera+ nested `motionBlur` object (plugins/camera-plus/camera-class.js:103-112).
    if let Some(mb_obj) = props.get("motionBlur").and_then(Value::as_object) {
        plugins.insert("camera-plus".to_owned());
        let mut mb = CameraMotionBlurParams::default();
        if let Some(v) = mb_obj.get("enabled").and_then(value_as_bool) {
            mb.enabled = v;
        }
        if let Some(v) = mb_obj.get("samples").and_then(value_as_u32) {
            mb.samples = v.clamp(1, 8);
        }
        if let Some(v) = mb_obj.get("shutter") {
            mb.shutter = parse_property_track(v, PropertyValue::Number(0.5));
        }
        cam.motion_blur = Some(mb);
    }

    cam
}

pub(super) fn parse_camera_dof_map(
    dof_obj: &Map<String, Value>,
    legacy_props: &Map<String, Value>,
) -> CameraDofParams {
    let mut dof = CameraDofParams::default();
    if let Some(v) = dof_obj.get("enabled").or_else(|| legacy_props.get("dof")) {
        let track = parse_property_track(v, PropertyValue::Option(0));
        dof.enabled = track.value.as_option().unwrap_or(0) != 0;
        if track.animated || track.expression.is_some() || track.keyframes.len() > 1 {
            dof.enabled_track = Some(track);
        }
    }
    if let Some(v) = dof_obj
        .get("focusDistance")
        .or_else(|| legacy_props.get("dofFocusDistance"))
    {
        dof.focus_distance = parse_property_track(v, PropertyValue::Number(80.0));
    }
    if let Some(v) = dof_obj
        .get("aperture")
        .or_else(|| legacy_props.get("dofAperture"))
    {
        dof.aperture = parse_property_track(v, PropertyValue::Number(3.0));
    }
    if let Some(v) = dof_obj
        .get("focusAreaWidth")
        .or_else(|| legacy_props.get("dofFocusAreaWidth"))
    {
        dof.focus_area_width = parse_property_track(v, PropertyValue::Number(0.0));
    }
    if let Some(v) = dof_obj
        .get("nearBlurLevel")
        .or_else(|| legacy_props.get("dofNearBlurLevel"))
    {
        dof.near_blur = parse_property_track(v, PropertyValue::Number(100.0));
    }
    if let Some(v) = dof_obj
        .get("farBlurLevel")
        .or_else(|| legacy_props.get("dofFarBlurLevel"))
    {
        dof.far_blur = parse_property_track(v, PropertyValue::Number(100.0));
    }
    dof
}

pub(super) fn parse_particles_data(props: Option<&Map<String, Value>>) -> ParticlesData {
    // Fall back to sensible non-zero CM3 starter defaults when a `.pz` Particles object omits
    // `number`, `rate`, or `lifetime`. Explicit values in `props` still override these.
    let mut data = ParticlesData {
        number: 100,
        rate: 25.0,
        lifetime: 2.0,
        ..ParticlesData::default()
    };
    let Some(p) = props else {
        return data;
    };
    if let Some(v) = p.get("time") {
        data.time = parse_property_track(v, PropertyValue::Number(0.0));
    }
    if let Some(v) = p.get("color") {
        data.color = parse_particle_color_stops(v);
    }
    if let Some(v) = p.get("size") {
        data.size = parse_particle_size_stops(v);
    }
    if let Some(v) = prop_f64(p, "number")
        .or_else(|| prop_f64(p, "count"))
        .or_else(|| prop_f64(p, "maxParticles"))
    {
        data.number = v.round().max(0.0) as u32;
    }
    if let Some(v) = prop_f64(p, "rate") {
        data.rate = v;
    }
    if let Some(v) = prop_f64(p, "lifetime") {
        data.lifetime = v;
    }
    if let Some(v) = p.get("pdist").and_then(value_as_u32) {
        data.pdist = v;
    }
    if let Some(v) = prop_vec3(p, "ipos") {
        data.ipos = v;
    }
    if let Some(v) = prop_vec3(p, "pspread") {
        data.pspread = v;
    }
    if let Some(v) = prop_f64(p, "iradius") {
        data.iradius = v;
    }
    if let Some(v) = prop_f64(p, "rspread") {
        data.rspread = v;
    }
    if let Some(v) = p.get("vdist").and_then(value_as_u32) {
        data.vdist = v;
    }
    if let Some(v) = prop_vec3(p, "ivel").or_else(|| prop_vec3(p, "velocity")) {
        data.ivel = v;
    }
    if let Some(v) = prop_vec3(p, "vspread")
        .or_else(|| prop_vec3(p, "velocitySpread"))
        .or_else(|| prop_f64(p, "velocitySpread").map(|s| [s, s, s]))
    {
        data.vspread = v;
    }
    if let Some(v) = prop_f64(p, "irvel") {
        data.irvel = v;
    }
    if let Some(v) = prop_f64(p, "rvspread") {
        data.rvspread = v;
    }
    if let Some(v) = prop_vec3(p, "accel")
        .or_else(|| prop_vec3(p, "gravity"))
        .or_else(|| prop_f64(p, "gravity").map(|g| [0.0, g, 0.0]))
    {
        data.accel = v;
    }
    if let Some(v) = prop_f64(p, "iang") {
        data.iang = v;
    }
    if let Some(v) = prop_f64(p, "aspread") {
        data.aspread = v;
    }
    if let Some(v) = prop_f64(p, "angvel") {
        data.angvel = v;
    }
    if let Some(v) = prop_f64(p, "avspread") {
        data.avspread = v;
    }
    if let Some(v) = p.get("blending").and_then(value_as_u32) {
        data.blending = v;
    }
    if let Some(v) = p.get("texture") {
        data.texture = v.as_str().map(ToOwned::to_owned);
    }
    data
}

pub(super) fn parse_constraints(
    props: &Map<String, Value>,
    is_camera_or_light: bool,
    plugins: &mut BTreeSet<String>,
) -> ObjectConstraints {
    let mut constraints = ObjectConstraints {
        zoidium_id: prop_str(props, "zoidiumId").filter(|s| !s.is_empty()),
        track_to: None,
        follow: None,
    };

    if let Some(target_id) = prop_str(props, "zoidiumTrackTo").filter(|s| !s.is_empty()) {
        plugins.insert("constraints".to_owned());
        // Zoidium constraint-properties.js:99: default trackAxis is 5 (-Z) for cameras & lights, 2 (+Z) otherwise.
        let default_track_axis = if is_camera_or_light { 5 } else { 2 };
        constraints.track_to = Some(TrackToConstraint {
            target_id,
            track_axis: props
                .get("zoidiumTrackAxis")
                .and_then(value_as_u32)
                .unwrap_or(default_track_axis),
            up_axis: props
                .get("zoidiumUpAxis")
                .and_then(value_as_u32)
                .unwrap_or(1),
            // Zoidium constraint-properties.js:128: zoidiumInfluence is stored in percent [0, 100], default 100.
            influence: props.get("zoidiumInfluence").map_or_else(
                || PropertyTrack::number(100.0),
                |v| parse_property_track(v, PropertyValue::Number(100.0)),
            ),
        });
    }

    if let Some(target_id) = prop_str(props, "zoidiumFollow").filter(|s| !s.is_empty()) {
        plugins.insert("constraints".to_owned());
        constraints.follow = Some(FollowConstraint {
            target_id,
            space: props
                .get("zoidiumFollowSpace")
                .and_then(value_as_u32)
                .unwrap_or(0),
            follow_position: props
                .get("zoidiumFollowPosition")
                .and_then(value_as_u32)
                .unwrap_or(1)
                != 0,
            follow_rotation: props
                .get("zoidiumFollowRotation")
                .and_then(value_as_u32)
                .unwrap_or(1)
                != 0,
            // Zoidium follow-properties.js:28: zoidiumFollowScale defaults to 1 (On).
            follow_scale: props
                .get("zoidiumFollowScale")
                .and_then(value_as_u32)
                .unwrap_or(1)
                != 0,
            position_offset: props.get("zoidiumFollowPositionOffset").map_or_else(
                || PropertyTrack::vec3([0.0, 0.0, 0.0]),
                |v| parse_property_track(v, PropertyValue::Vec3([0.0, 0.0, 0.0])),
            ),
            rotation_offset: props.get("zoidiumFollowRotationOffset").map_or_else(
                || PropertyTrack::vec3([0.0, 0.0, 0.0]),
                |v| parse_property_track(v, PropertyValue::Vec3([0.0, 0.0, 0.0])),
            ),
            scale_offset: props.get("zoidiumFollowScaleOffset").map_or_else(
                || PropertyTrack::vec3([1.0, 1.0, 1.0]),
                |v| parse_property_track(v, PropertyValue::Vec3([1.0, 1.0, 1.0])),
            ),
            fallback_pose: props.get("zoidiumFollowFallback").cloned(),
        });
    }

    constraints
}

pub(super) fn collect_scene_object_id_maps(
    objects: &[Object3D],
    id_to_uuid: &mut BTreeMap<String, String>,
    zoidium_ids: &mut BTreeSet<String>,
    name_to_uuid: &mut BTreeMap<String, Option<String>>,
    current_uuids: &mut BTreeSet<String>,
) {
    for obj in objects {
        let new_uuid = obj.id.to_string();
        current_uuids.insert(new_uuid.clone());
        if let Some(zid) = obj.constraints.zoidium_id.as_deref()
            && !zid.is_empty()
        {
            id_to_uuid.insert(zid.to_owned(), new_uuid.clone());
            zoidium_ids.insert(zid.to_owned());
        }
        for key in ["id", "property.id"] {
            if let Some(raw_id) = obj.preserved.get(key).and_then(Value::as_str)
                && !raw_id.is_empty()
            {
                id_to_uuid
                    .entry(raw_id.to_owned())
                    .or_insert_with(|| new_uuid.clone());
            }
        }
        if !obj.name.is_empty() {
            name_to_uuid
                .entry(obj.name.clone())
                .and_modify(|slot| *slot = None)
                .or_insert_with(|| Some(new_uuid.clone()));
        }
        if let Object3DKind::Group { children, .. } = &obj.kind {
            collect_scene_object_id_maps(
                children,
                id_to_uuid,
                zoidium_ids,
                name_to_uuid,
                current_uuids,
            );
        }
    }
}

pub(super) fn apply_scene_constraint_target_remap(
    objects: &mut [Object3D],
    id_to_uuid: &BTreeMap<String, String>,
    zoidium_ids: &BTreeSet<String>,
    name_to_uuid: &BTreeMap<String, Option<String>>,
    current_uuids: &BTreeSet<String>,
) {
    let remap_target = |target_id: &mut String| {
        if target_id.is_empty() || current_uuids.contains(target_id.as_str()) {
            return;
        }
        if Uuid::parse_str(target_id).is_ok() {
            if let Some(mapped) = id_to_uuid.get(target_id.as_str()) {
                *target_id = mapped.clone();
                return;
            }
        } else if zoidium_ids.contains(target_id.as_str()) {
            // Preserve explicit non-UUID Zoidium IDs (e.g. `"text-target-id"`) that match
            // a target object's preserved `zoidium_id`.
            return;
        } else if let Some(mapped) = id_to_uuid.get(target_id.as_str()) {
            *target_id = mapped.clone();
            return;
        }
        if let Some(Some(mapped)) = name_to_uuid.get(target_id.as_str()) {
            *target_id = mapped.clone();
        }
    };

    for obj in objects {
        if let Some(tt) = obj.constraints.track_to.as_mut() {
            remap_target(&mut tt.target_id);
        }
        if let Some(fl) = obj.constraints.follow.as_mut() {
            remap_target(&mut fl.target_id);
        }
        if let Some(zid) = obj.constraints.zoidium_id.as_deref()
            && Uuid::parse_str(zid).is_ok()
            && zid != obj.id.to_string()
        {
            obj.constraints.zoidium_id = None;
        }
        if let Object3DKind::Group { children, .. } = &mut obj.kind {
            apply_scene_constraint_target_remap(
                children,
                id_to_uuid,
                zoidium_ids,
                name_to_uuid,
                current_uuids,
            );
        }
    }
}

pub(crate) fn remap_scene_constraint_targets(objects: &mut [Object3D]) {
    let mut id_to_uuid = BTreeMap::new();
    let mut zoidium_ids = BTreeSet::new();
    let mut name_to_uuid = BTreeMap::new();
    let mut current_uuids = BTreeSet::new();
    collect_scene_object_id_maps(
        objects,
        &mut id_to_uuid,
        &mut zoidium_ids,
        &mut name_to_uuid,
        &mut current_uuids,
    );
    apply_scene_constraint_target_remap(
        objects,
        &id_to_uuid,
        &zoidium_ids,
        &name_to_uuid,
        &current_uuids,
    );
}
