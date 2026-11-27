use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use crate::model::{
    CameraData, CameraKind, CompatMode, LightData, LightKind, MaterialSpec, Object3D, Object3DKind,
    ParticlesData, PropertyTrack, Shape3DKind, Text3DAlign, Text3DData,
};

use super::effects::serialize_custom_properties;
use super::tracks::{
    finite_array3, finite_array4, finite_f64, is_non_default_scalar, remap_json_value, remap_str,
    serialize_gradient_stops, serialize_property_track, serialize_static_or_animated_track,
};

#[derive(Default)]
pub(super) struct SceneConstraintExportCtx {
    uuid_to_export_id: BTreeMap<String, String>,
    referenced_export_ids: BTreeSet<String>,
}

impl SceneConstraintExportCtx {
    pub(super) fn from_objects(objects: &[Object3D]) -> Self {
        let mut ctx = Self::default();
        ctx.collect_ids(objects);
        ctx.collect_targets(objects);
        ctx
    }

    fn collect_ids(&mut self, objects: &[Object3D]) {
        for obj in objects {
            let export_id = obj
                .constraints
                .zoidium_id
                .as_deref()
                .filter(|s| !s.is_empty())
                .map_or_else(|| obj.id.to_string(), ToOwned::to_owned);
            self.uuid_to_export_id.insert(obj.id.to_string(), export_id);
            if let Object3DKind::Group { children, .. } = &obj.kind {
                self.collect_ids(children);
            }
        }
    }

    fn collect_targets(&mut self, objects: &[Object3D]) {
        for obj in objects {
            if let Some(tt) = &obj.constraints.track_to
                && !tt.target_id.is_empty()
            {
                let resolved = self
                    .uuid_to_export_id
                    .get(&tt.target_id)
                    .cloned()
                    .unwrap_or_else(|| tt.target_id.clone());
                self.referenced_export_ids.insert(resolved);
            }
            if let Some(fl) = &obj.constraints.follow
                && !fl.target_id.is_empty()
            {
                let resolved = self
                    .uuid_to_export_id
                    .get(&fl.target_id)
                    .cloned()
                    .unwrap_or_else(|| fl.target_id.clone());
                self.referenced_export_ids.insert(resolved);
            }
            if let Object3DKind::Group { children, .. } = &obj.kind {
                self.collect_targets(children);
            }
        }
    }
}

pub(crate) fn serialize_object3d(obj: &Object3D, remap: &BTreeMap<String, String>) -> Value {
    serialize_object3d_with_ctx(obj, remap, None)
}

pub(super) fn serialize_object3d_with_ctx(
    obj: &Object3D,
    remap: &BTreeMap<String, String>,
    constraint_ctx: Option<&SceneConstraintExportCtx>,
) -> Value {
    let plugin_obj_type = obj
        .preserved
        .get("zoidiumPluginObjectType")
        .and_then(Value::as_str);

    let mut out = if let Object3DKind::Dummy { payload, .. } = &obj.kind {
        let mut base = match payload {
            Value::Object(m) => m.clone(),
            _ => Map::new(),
        };
        let mut base_val = Value::Object(base);
        remap_json_value(&mut base_val, remap);
        if let Value::Object(m) = base_val {
            base = m;
        } else {
            base = Map::new();
        }
        base
    } else {
        Map::new()
    };

    for (k, v) in &obj.preserved {
        if k.starts_with("property.") || k == "zoidiumPluginObjectType" {
            continue;
        }
        let mut cloned = v.clone();
        remap_json_value(&mut cloned, remap);
        out.insert(k.clone(), cloned);
    }

    let mut props = out
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (k, v) in &obj.preserved {
        if let Some(pk) = k.strip_prefix("property.") {
            let mut cloned = v.clone();
            remap_json_value(&mut cloned, remap);
            props.insert(pk.to_owned(), cloned);
        }
    }

    props.insert("name".to_owned(), Value::String(obj.name.clone()));
    props.insert(
        "enabled".to_owned(),
        serialize_property_track(&obj.enabled, remap),
    );
    props.insert(
        "position".to_owned(),
        serialize_property_track(&obj.position, remap),
    );
    props.insert(
        "rotation".to_owned(),
        serialize_property_track(&obj.rotation, remap),
    );
    props.insert(
        "scale".to_owned(),
        serialize_property_track(&obj.scale, remap),
    );
    props.insert(
        "eulerOrder".to_owned(),
        Value::String(obj.euler_order.as_str().to_owned()),
    );

    // Constraints
    if let Some(zid) = &obj.constraints.zoidium_id
        && !zid.is_empty()
    {
        props.insert("zoidiumId".to_owned(), Value::String(zid.clone()));
    } else if let Some(ctx) = constraint_ctx {
        let obj_uuid = obj.id.to_string();
        if ctx.referenced_export_ids.contains(&obj_uuid)
            || obj.constraints.track_to.is_some()
            || obj.constraints.follow.is_some()
        {
            props.insert("zoidiumId".to_owned(), Value::String(obj_uuid));
        }
    }
    if let Some(tt) = &obj.constraints.track_to {
        let exported_target = constraint_ctx
            .and_then(|ctx| ctx.uuid_to_export_id.get(&tt.target_id))
            .map_or(tt.target_id.as_str(), String::as_str);
        props.insert(
            "zoidiumTrackTo".to_owned(),
            Value::String(exported_target.to_owned()),
        );
        props.insert("zoidiumTrackAxis".to_owned(), json!(tt.track_axis));
        props.insert("zoidiumUpAxis".to_owned(), json!(tt.up_axis));
        props.insert(
            "zoidiumInfluence".to_owned(),
            serialize_property_track(&tt.influence, remap),
        );
    }
    if let Some(fl) = &obj.constraints.follow {
        let exported_target = constraint_ctx
            .and_then(|ctx| ctx.uuid_to_export_id.get(&fl.target_id))
            .map_or(fl.target_id.as_str(), String::as_str);
        props.insert(
            "zoidiumFollow".to_owned(),
            Value::String(exported_target.to_owned()),
        );
        props.insert("zoidiumFollowSpace".to_owned(), json!(fl.space));
        props.insert(
            "zoidiumFollowPosition".to_owned(),
            json!(u32::from(fl.follow_position)),
        );
        props.insert(
            "zoidiumFollowRotation".to_owned(),
            json!(u32::from(fl.follow_rotation)),
        );
        props.insert(
            "zoidiumFollowScale".to_owned(),
            json!(u32::from(fl.follow_scale)),
        );
        props.insert(
            "zoidiumFollowPositionOffset".to_owned(),
            serialize_property_track(&fl.position_offset, remap),
        );
        props.insert(
            "zoidiumFollowRotationOffset".to_owned(),
            serialize_property_track(&fl.rotation_offset, remap),
        );
        props.insert(
            "zoidiumFollowScaleOffset".to_owned(),
            serialize_property_track(&fl.scale_offset, remap),
        );
        if let Some(fb) = &fl.fallback_pose {
            props.insert("zoidiumFollowFallback".to_owned(), fb.clone());
        }
    }

    // Compatibility flags
    out.insert(
        "zoidiumUvMode".to_owned(),
        Value::String(obj.zoidium_uv_mode.as_str().to_owned()),
    );
    out.insert(
        "zoidiumShadowMode".to_owned(),
        Value::String(obj.zoidium_shadow_mode.as_str().to_owned()),
    );

    // Kind-specific serialization
    match &obj.kind {
        Object3DKind::Group {
            reflection_visibility,
            custom_properties,
            children,
        } => {
            if let Some(pt) = plugin_obj_type {
                out.insert("type".to_owned(), Value::String(pt.to_owned()));
            } else {
                out.insert("type".to_owned(), json!(5));
                props.insert(
                    "reflectionVisibility".to_owned(),
                    json!(*reflection_visibility),
                );
            }
            if !custom_properties.is_empty() {
                out.insert(
                    "customProperties".to_owned(),
                    Value::Array(serialize_custom_properties(custom_properties, remap)),
                );
            }
            let child_arr: Vec<Value> = children
                .iter()
                .map(|c| serialize_object3d_with_ctx(c, remap, constraint_ctx))
                .collect();
            out.insert("objects".to_owned(), Value::Array(child_arr));
        }
        Object3DKind::Text3D(t3d) => {
            out.insert("type".to_owned(), json!(1));
            out.insert(
                "zoidiumTextScaleMode".to_owned(),
                Value::String(t3d.text_scale_mode.as_str().to_owned()),
            );
            serialize_text3d_props(t3d, &mut props, remap);
        }
        Object3DKind::Shape3D(shape) => {
            serialize_shape3d(shape, plugin_obj_type, &mut out, &mut props, remap);
        }
        Object3DKind::Light(light) => {
            serialize_light(light, plugin_obj_type, &mut out, &mut props, remap);
        }
        Object3DKind::Camera(cam) => {
            serialize_camera(cam, plugin_obj_type, &mut out, &mut props, remap);
        }
        Object3DKind::Particles(particles) => {
            out.insert("type".to_owned(), json!(4));
            if let Some(ref sprite) = particles.sprite_preset
                && !sprite.is_empty()
            {
                out.insert(
                    "_zoidiumParticlesPlusSprite".to_owned(),
                    Value::String(sprite.clone()),
                );
            } else {
                out.remove("_zoidiumParticlesPlusSprite");
            }
            serialize_particles_props(particles, &mut props, remap);
        }
        Object3DKind::Model(model) => {
            out.insert("type".to_owned(), json!(2));
            let object_type = match (model.preset_type, model.asset_id.is_some()) {
                (Some(pt), _) => pt,
                (None, true) => 0,
                (None, false) => 1,
            };
            out.insert("objectType".to_owned(), json!(object_type));
            if !model.model_properties.is_empty() {
                let mut mp = Map::new();
                for (k, v) in &model.model_properties {
                    mp.insert(k.clone(), serialize_property_track(v, remap));
                }
                props.insert("modelProperties".to_owned(), Value::Object(mp));
            }
            if let Some(aid) = &model.asset_id {
                props.insert("asset".to_owned(), Value::String(remap_str(aid, remap)));
            }
        }
        Object3DKind::Dummy { original_type, .. } => {
            if !out.contains_key("type") {
                if let Some(orig) = original_type {
                    if let Ok(num) = orig.parse::<u32>() {
                        out.insert("type".to_owned(), json!(num));
                    } else {
                        out.insert("type".to_owned(), Value::String(orig.clone()));
                    }
                } else if let Some(pt) = plugin_obj_type {
                    if pt == "zoidium:optical-flares/flare" {
                        out.insert("type".to_owned(), json!(13));
                    } else {
                        out.insert("type".to_owned(), Value::String(pt.to_owned()));
                    }
                } else {
                    out.insert("type".to_owned(), json!(99));
                }
            }
        }
    }

    out.insert("properties".to_owned(), Value::Object(props));

    if let Some(mat) = &obj.material {
        out.insert(
            "material".to_owned(),
            serialize_material(mat, obj.zoidium_uv_mode, remap),
        );
    }
    if let Some(fm) = obj.material_slots.get("front") {
        out.insert(
            "frontMaterial".to_owned(),
            serialize_material(fm, obj.zoidium_uv_mode, remap),
        );
    }
    if let Some(sm) = obj.material_slots.get("side") {
        out.insert(
            "sideMaterial".to_owned(),
            serialize_material(sm, obj.zoidium_uv_mode, remap),
        );
    }

    Value::Object(out)
}

pub(super) fn serialize_text3d_props(
    t3d: &Text3DData,
    props: &mut Map<String, Value>,
    remap: &BTreeMap<String, String>,
) {
    props.insert("text".to_owned(), Value::String(t3d.text.clone()));
    props.insert(
        "font".to_owned(),
        t3d.font
            .as_ref()
            .map_or(Value::Null, |f| Value::String(remap_str(f, remap))),
    );
    props.insert(
        "size".to_owned(),
        json!([finite_f64(t3d.size[0]), finite_f64(t3d.size[1])]),
    );
    props.insert("detail".to_owned(), json!(t3d.detail));
    props.insert("bevel".to_owned(), json!(u32::from(t3d.bevel)));
    // CRITICAL CM3 MAPPING QUIRK: bevelSize[0] = bevel_thickness, bevelSize[1] = bevel_size.
    props.insert(
        "bevelSize".to_owned(),
        json!([finite_f64(t3d.bevel_thickness), finite_f64(t3d.bevel_size)]),
    );
    props.insert(
        "positionMode".to_owned(),
        json!(text3d_align_to_cm3_id(t3d.position_mode)),
    );

    let has_text_plus = t3d.spacing.abs() > 1e-9
        || is_non_default_scalar(&t3d.character_index_offset, 0.0)
        || is_non_default_scalar(&t3d.character_count_override, 0.0)
        || t3d.bevel_side != 0
        || t3d.bevel_detail != 3
        || t3d.bevel_profile != 1
        || (t3d.bevel_tension - 0.5).abs() > 1e-9;

    if has_text_plus {
        props.insert("spacing".to_owned(), json!(finite_f64(t3d.spacing)));
        props.insert(
            "characterIndexOffset".to_owned(),
            serialize_property_track(&t3d.character_index_offset, remap),
        );
        props.insert(
            "characterCountOverride".to_owned(),
            serialize_property_track(&t3d.character_count_override, remap),
        );
        props.insert("bevelSide".to_owned(), json!(t3d.bevel_side));
        props.insert("bevelDetail".to_owned(), json!(t3d.bevel_detail));
        props.insert("bevelProfile".to_owned(), json!(t3d.bevel_profile));
        props.insert(
            "bevelTension".to_owned(),
            json!(finite_f64(t3d.bevel_tension)),
        );
    }
}

pub(super) const fn text3d_align_to_cm3_id(align: Text3DAlign) -> u32 {
    match align {
        Text3DAlign::Center => 0,
        Text3DAlign::Left => 1,
        Text3DAlign::Right => 2,
        Text3DAlign::Top => 3,
        Text3DAlign::Bottom => 4,
        Text3DAlign::Front => 5,
        Text3DAlign::Back => 6,
    }
}

pub(super) fn serialize_shape3d(
    shape: &Shape3DKind,
    plugin_obj_type: Option<&str>,
    out: &mut Map<String, Value>,
    props: &mut Map<String, Value>,
    remap: &BTreeMap<String, String>,
) {
    match shape {
        Shape3DKind::Box { size } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(1));
            props.insert(
                "geometryProperties".to_owned(),
                json!({ "box_size": [finite_f64(size[0]), finite_f64(size[1]), finite_f64(size[2])] }),
            );
        }
        Shape3DKind::Cylinder {
            top_radius,
            bottom_radius,
            height,
            detail,
            open_ended,
        } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(2));
            // CRITICAL CM3 INVERSION: cylinder_openEnded == 0 => open_ended = true, 1 => closed.
            let raw_open = u32::from(!*open_ended);
            props.insert(
                "geometryProperties".to_owned(),
                json!({
                    "cylinder_size": [finite_f64(*top_radius), finite_f64(*bottom_radius), finite_f64(*height)],
                    "cylinder_detail": *detail,
                    "cylinder_openEnded": raw_open
                }),
            );
        }
        Shape3DKind::Rectangle { size } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(3));
            props.insert(
                "geometryProperties".to_owned(),
                json!({ "rect_size": [finite_f64(size[0]), finite_f64(size[1])] }),
            );
        }
        Shape3DKind::Circle { radius, detail } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(4));
            props.insert(
                "geometryProperties".to_owned(),
                json!({ "circle_size": finite_f64(*radius), "circle_detail": *detail }),
            );
        }
        Shape3DKind::Sphere {
            radius,
            width_segments,
            height_segments,
        } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(5));
            props.insert(
                "geometryProperties".to_owned(),
                json!({
                    "sphere_size": finite_f64(*radius),
                    "sphere_detail": [*width_segments, *height_segments]
                }),
            );
        }
        Shape3DKind::Donut {
            radius,
            tube,
            radial_segments,
            tubular_segments,
        } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(6));
            props.insert(
                "geometryProperties".to_owned(),
                json!({
                    "donut_size": [finite_f64(*radius), finite_f64(*tube)],
                    "donut_detail": [*radial_segments, *tubular_segments]
                }),
            );
        }
        Shape3DKind::Wire {
            size,
            thickness,
            loops,
            inner_segments,
            outer_segments,
        } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(7));
            props.insert(
                "geometryProperties".to_owned(),
                json!({
                    "path_size": [finite_f64(size[0]), finite_f64(size[1]), finite_f64(size[2])],
                    "path_thickness": finite_f64(*thickness),
                    "path_loops": *loops,
                    "path_detail": [*inner_segments, *outer_segments]
                }),
            );
        }
        Shape3DKind::CustomGeometry { asset_id } => {
            out.insert("type".to_owned(), json!(0));
            out.insert("objectType".to_owned(), json!(99));
            props.insert(
                "geometryProperties".to_owned(),
                json!({
                    "customGeometry": asset_id.as_ref().map_or(Value::Null, |id| Value::String(remap_str(id, remap)))
                }),
            );
        }
        // Geometry+ primitives (8/8)
        Shape3DKind::RoundedBox {
            size,
            radius,
            segments,
            corner_radius,
            corner_segments,
            bevel_size,
            bevel_segments,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/rounded-box")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(
                props,
                "size",
                json!([
                    finite_f64(size[0]),
                    finite_f64(size[1]),
                    finite_f64(size[2])
                ]),
            );
            insert_shape_prop_preserving_animated(
                props,
                "cornerRadius",
                json!(finite_f64(corner_radius.unwrap_or(*radius))),
            );
            insert_shape_prop_preserving_animated(
                props,
                "cornerSegments",
                json!(corner_segments.unwrap_or(*segments)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "bevelSize",
                json!(finite_f64(bevel_size.unwrap_or(0.6))),
            );
            insert_shape_prop_preserving_animated(
                props,
                "bevelSegments",
                json!(bevel_segments.unwrap_or(2)),
            );
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::Cone {
            radius,
            height,
            detail,
            open_ended,
            top_radius,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/cone")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(
                props,
                "topRadius",
                json!(finite_f64(*top_radius)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "bottomRadius",
                json!(finite_f64(*radius)),
            );
            insert_shape_prop_preserving_animated(props, "height", json!(finite_f64(*height)));
            insert_shape_prop_preserving_animated(props, "segments", json!(*detail));
            // Zoidium Cone endCaps: 0 = closed, 1 = open.
            insert_shape_prop_preserving_animated(props, "endCaps", json!(u32::from(*open_ended)));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::Tetrahedron {
            radius,
            detail,
            shading,
        }
        | Shape3DKind::Octahedron {
            radius,
            detail,
            shading,
        }
        | Shape3DKind::Icosahedron {
            radius,
            detail,
            shading,
        }
        | Shape3DKind::Dodecahedron {
            radius,
            detail,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/polyhedron")
                        .to_owned(),
                ),
            );
            let shape_idx = match shape {
                Shape3DKind::Tetrahedron { .. } => 0,
                Shape3DKind::Octahedron { .. } => 1,
                Shape3DKind::Icosahedron { .. } => 2,
                Shape3DKind::Dodecahedron { .. } => 3,
                _ => 0,
            };
            insert_shape_prop_preserving_animated(props, "shape", json!(shape_idx));
            insert_shape_prop_preserving_animated(props, "radius", json!(finite_f64(*radius)));
            insert_shape_prop_preserving_animated(props, "detail", json!(*detail));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::Capsule {
            radius,
            length,
            cap_segments,
            radial_segments,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/capsule")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(props, "radius", json!(finite_f64(*radius)));
            insert_shape_prop_preserving_animated(
                props,
                "height",
                json!(finite_f64(*length + 2.0 * *radius)),
            );
            insert_shape_prop_preserving_animated(props, "segments", json!(*cap_segments));
            insert_shape_prop_preserving_animated(props, "radialSegments", json!(*radial_segments));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::TorusKnot {
            radius,
            tube,
            tubular_segments,
            radial_segments,
            p,
            q,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/torus-knot")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(props, "radius", json!(finite_f64(*radius)));
            insert_shape_prop_preserving_animated(props, "tube", json!(finite_f64(*tube)));
            insert_shape_prop_preserving_animated(
                props,
                "tubularSegments",
                json!(*tubular_segments),
            );
            insert_shape_prop_preserving_animated(props, "radialSegments", json!(*radial_segments));
            insert_shape_prop_preserving_animated(props, "p", json!(*p));
            insert_shape_prop_preserving_animated(props, "q", json!(*q));
        }
        Shape3DKind::Tube {
            outer_radius,
            inner_radius,
            height,
            radial_segments,
            open_ended,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/tube")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(
                props,
                "outerRadius",
                json!(finite_f64(*outer_radius)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "innerRadius",
                json!(finite_f64(*inner_radius)),
            );
            insert_shape_prop_preserving_animated(props, "height", json!(finite_f64(*height)));
            insert_shape_prop_preserving_animated(props, "radialSegments", json!(*radial_segments));
            insert_shape_prop_preserving_animated(props, "endCaps", json!(u32::from(*open_ended)));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::Gear {
            teeth,
            inner_radius,
            outer_radius,
            thickness,
            tooth_width,
            bevel_size,
            bevel_segments,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/gear")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(props, "teeth", json!(*teeth));
            insert_shape_prop_preserving_animated(
                props,
                "innerRadius",
                json!(finite_f64(*inner_radius)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "outerRadius",
                json!(finite_f64(*outer_radius)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "thickness",
                json!(finite_f64(*thickness)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "toothWidth",
                json!(finite_f64(*tooth_width)),
            );
            insert_shape_prop_preserving_animated(
                props,
                "bevelSize",
                json!(finite_f64(*bevel_size)),
            );
            insert_shape_prop_preserving_animated(props, "bevelSegments", json!(*bevel_segments));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::Helix {
            radius,
            height,
            turns,
            tube_radius,
            path_segments,
            radial_segments,
            shading,
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/helix")
                        .to_owned(),
                ),
            );
            insert_shape_prop_preserving_animated(props, "radius", json!(finite_f64(*radius)));
            insert_shape_prop_preserving_animated(props, "height", json!(finite_f64(*height)));
            insert_shape_prop_preserving_animated(props, "turns", json!(finite_f64(*turns)));
            insert_shape_prop_preserving_animated(
                props,
                "tubeRadius",
                json!(finite_f64(*tube_radius)),
            );
            insert_shape_prop_preserving_animated(props, "pathSegments", json!(*path_segments));
            insert_shape_prop_preserving_animated(props, "radialSegments", json!(*radial_segments));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
        Shape3DKind::Spline {
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
        } => {
            out.insert(
                "type".to_owned(),
                Value::String(
                    plugin_obj_type
                        .unwrap_or("zoidium:geometry-plus/spline")
                        .to_owned(),
                ),
            );
            let point_count = points.len().clamp(2, 32);
            props.insert("points".to_owned(), json!(point_count));
            for (idx, pt) in points.iter().take(32).enumerate() {
                let key = format!("point{}", idx + 1);
                insert_shape_prop_preserving_animated(
                    props,
                    &key,
                    json!([finite_f64(pt[0]), finite_f64(pt[1]), finite_f64(pt[2])]),
                );
            }
            props.insert(
                "interpolation".to_owned(),
                Value::String(interpolation.clone()),
            );
            insert_shape_prop_preserving_animated(props, "closed", json!(u32::from(*closed)));
            insert_shape_prop_preserving_animated(props, "start", json!(finite_f64(*start)));
            insert_shape_prop_preserving_animated(props, "end", json!(finite_f64(*end)));
            insert_shape_prop_preserving_animated(
                props,
                "thickness",
                json!(finite_f64(*thickness)),
            );
            props.insert(
                "thicknessMap".to_owned(),
                serialize_gradient_stops(thickness_map),
            );
            props.insert(
                "gradientSpace".to_owned(),
                Value::String(gradient_space.clone()),
            );
            insert_shape_prop_preserving_animated(props, "tension", json!(finite_f64(*tension)));
            insert_shape_prop_preserving_animated(props, "pathSegments", json!(*path_segments));
            insert_shape_prop_preserving_animated(props, "radialSegments", json!(*radial_segments));
            props.insert("caps".to_owned(), Value::String(caps.clone()));
            props.insert("capStyle".to_owned(), Value::String(cap_style.clone()));
            insert_shape_prop_preserving_animated(props, "shading", json!(*shading));
        }
    }
}

pub(super) fn is_preserved_animated_track(val: &Value) -> bool {
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
        return channels.iter().any(is_preserved_animated_track);
    }
    false
}

pub(super) fn insert_shape_prop_preserving_animated(
    props: &mut Map<String, Value>,
    key: &str,
    value: Value,
) {
    if props.get(key).is_some_and(is_preserved_animated_track) {
        return;
    }
    props.insert(key.to_owned(), value);
}

pub(super) fn serialize_light(
    light: &LightData,
    plugin_obj_type: Option<&str>,
    out: &mut Map<String, Value>,
    props: &mut Map<String, Value>,
    remap: &BTreeMap<String, String>,
) {
    if let Some(pt) = plugin_obj_type {
        out.insert("type".to_owned(), Value::String(pt.to_owned()));
    } else {
        out.insert("type".to_owned(), json!(3));
        let sub_type = match light.kind {
            LightKind::Spot => 1,
            LightKind::Point => 2,
            LightKind::Directional => 3,
            LightKind::Hemisphere => 4,
            LightKind::Ambient => 5,
        };
        out.insert("objectType".to_owned(), json!(sub_type));
    }

    props.insert(
        "color".to_owned(),
        serialize_property_track(&light.color, remap),
    );
    props.insert(
        "groundColor".to_owned(),
        serialize_property_track(&light.ground_color, remap),
    );
    props.insert(
        "target".to_owned(),
        serialize_property_track(&light.target, remap),
    );
    props.insert(
        "intensity".to_owned(),
        serialize_property_track(&light.intensity, remap),
    );
    props.insert(
        "angle".to_owned(),
        serialize_property_track(&light.angle, remap),
    );
    props.insert(
        "zoidiumCastShadows".to_owned(),
        json!(u32::from(light.cast_shadow)),
    );
    props.insert("penumbra".to_owned(), json!(finite_f64(light.penumbra)));
    props.insert("decay".to_owned(), json!(finite_f64(light.decay)));
    props.insert("distance".to_owned(), json!(finite_f64(light.distance)));
    props.insert("shadowMapSize".to_owned(), json!(light.shadow_map_size));
    props.insert(
        "shadowBias".to_owned(),
        json!(finite_f64(light.shadow_bias)),
    );
}

pub(super) fn serialize_camera(
    cam: &CameraData,
    plugin_obj_type: Option<&str>,
    out: &mut Map<String, Value>,
    props: &mut Map<String, Value>,
    remap: &BTreeMap<String, String>,
) {
    let is_camera_plus = plugin_obj_type.is_some_and(|s| s.starts_with("zoidium:camera-plus/"))
        || cam.dof.is_some()
        || cam.vibrate.is_some()
        || cam.motion_blur.is_some()
        || !cam.active
        || (cam.film_gate - 36.0).abs() > 1e-9
        || is_non_default_scalar(&cam.focal_length, 35.0)
        || is_non_default_scalar(&cam.zoom, 1.0)
        || is_non_default_scalar(&cam.film_offset_x, 0.0)
        || is_non_default_scalar(&cam.film_offset_y, 0.0);

    if is_camera_plus {
        out.insert(
            "type".to_owned(),
            Value::String(
                plugin_obj_type
                    .unwrap_or("zoidium:camera-plus/camera")
                    .to_owned(),
            ),
        );
        props.insert("active".to_owned(), json!(u32::from(cam.active)));
        props.insert(
            "projection".to_owned(),
            Value::String(
                match cam.kind {
                    CameraKind::Perspective => "perspective",
                    CameraKind::Orthographic => "orthographic",
                }
                .to_owned(),
            ),
        );
        props.insert(
            "focalLength".to_owned(),
            serialize_property_track(&cam.focal_length, remap),
        );
        props.insert("filmGate".to_owned(), json!(finite_f64(cam.film_gate)));
        props.insert(
            "zoom".to_owned(),
            serialize_property_track(&cam.zoom, remap),
        );
        props.insert(
            "filmOffsetX".to_owned(),
            serialize_property_track(&cam.film_offset_x, remap),
        );
        props.insert(
            "filmOffsetY".to_owned(),
            serialize_property_track(&cam.film_offset_y, remap),
        );
    } else {
        out.insert("type".to_owned(), json!(6));
        let sub_type = match cam.kind {
            CameraKind::Perspective => 1,
            CameraKind::Orthographic => 2,
        };
        out.insert("objectType".to_owned(), json!(sub_type));
    }

    props.insert("fov".to_owned(), serialize_property_track(&cam.fov, remap));
    props.insert("near".to_owned(), json!(finite_f64(cam.near)));
    props.insert("far".to_owned(), json!(finite_f64(cam.far)));
    props.insert(
        "shakeAmplitude".to_owned(),
        serialize_property_track(&cam.shake_amplitude, remap),
    );
    props.insert(
        "shakeSpeed".to_owned(),
        serialize_property_track(&cam.shake_speed, remap),
    );
    props.insert(
        "shake".to_owned(),
        json!({
            "mode": cam.shake.mode,
            "amplitude": finite_array4(cam.shake.amplitude),
            "frequency": finite_array4(cam.shake.frequency),
            "phase": finite_array4(cam.shake.phase),
            "noise": finite_array4(cam.shake.noise),
            "smooth": finite_array4(cam.shake.smooth),
        }),
    );

    if let Some(dof) = &cam.dof {
        let enabled_val = if let Some(track) = &dof.enabled_track {
            serialize_property_track(track, remap)
        } else {
            serialize_property_track(&PropertyTrack::option(u32::from(dof.enabled)), remap)
        };
        props.insert(
            "depthOfField".to_owned(),
            json!({
                "enabled": enabled_val,
                "focusDistance": serialize_property_track(&dof.focus_distance, remap),
                "aperture": serialize_property_track(&dof.aperture, remap),
                "focusAreaWidth": serialize_property_track(&dof.focus_area_width, remap),
                "nearBlurLevel": serialize_property_track(&dof.near_blur, remap),
                "farBlurLevel": serialize_property_track(&dof.far_blur, remap),
            }),
        );
    }

    if let Some(vib) = &cam.vibrate {
        props.insert(
            "vibrate".to_owned(),
            json!({
                "enabled": u32::from(vib.enabled),
                "regularPulse": u32::from(vib.regular_pulse),
                "relative": u32::from(vib.relative),
                "seed": vib.seed,
                "enablePosition": u32::from(vib.enable_position),
                "positionAmplitude": serialize_static_or_animated_track(&vib.position_amplitude, remap),
                "positionFrequency": serialize_static_or_animated_track(&vib.position_frequency, remap),
                "enableRotation": u32::from(vib.enable_rotation),
                "rotationAmplitude": serialize_static_or_animated_track(&vib.rotation_amplitude, remap),
                "rotationFrequency": serialize_static_or_animated_track(&vib.rotation_frequency, remap),
                "frequency": serialize_static_or_animated_track(&vib.frequency, remap),
            }),
        );
    }

    if let Some(mb) = &cam.motion_blur {
        props.insert(
            "motionBlur".to_owned(),
            json!({
                "enabled": u32::from(mb.enabled),
                "samples": mb.samples,
                "shutter": serialize_property_track(&mb.shutter, remap),
            }),
        );
    }
}

pub(super) fn serialize_particles_props(
    p: &ParticlesData,
    props: &mut Map<String, Value>,
    remap: &BTreeMap<String, String>,
) {
    props.insert("time".to_owned(), serialize_property_track(&p.time, remap));
    props.insert("color".to_owned(), serialize_gradient_stops(&p.color));
    props.insert("size".to_owned(), serialize_gradient_stops(&p.size));
    props.insert("number".to_owned(), json!(p.number));
    props.insert("rate".to_owned(), json!(finite_f64(p.rate)));
    props.insert("lifetime".to_owned(), json!(finite_f64(p.lifetime)));
    props.insert("pdist".to_owned(), json!(p.pdist));
    props.insert("ipos".to_owned(), json!(finite_array3(p.ipos)));
    props.insert("pspread".to_owned(), json!(finite_array3(p.pspread)));
    props.insert("iradius".to_owned(), json!(finite_f64(p.iradius)));
    props.insert("rspread".to_owned(), json!(finite_f64(p.rspread)));
    props.insert("vdist".to_owned(), json!(p.vdist));
    props.insert("ivel".to_owned(), json!(finite_array3(p.ivel)));
    props.insert("vspread".to_owned(), json!(finite_array3(p.vspread)));
    props.insert("irvel".to_owned(), json!(finite_f64(p.irvel)));
    props.insert("rvspread".to_owned(), json!(finite_f64(p.rvspread)));
    props.insert("accel".to_owned(), json!(finite_array3(p.accel)));
    props.insert("iang".to_owned(), json!(finite_f64(p.iang)));
    props.insert("aspread".to_owned(), json!(finite_f64(p.aspread)));
    props.insert("angvel".to_owned(), json!(finite_f64(p.angvel)));
    props.insert("avspread".to_owned(), json!(finite_f64(p.avspread)));
    props.insert("blending".to_owned(), json!(p.blending));
    props.insert(
        "texture".to_owned(),
        p.texture
            .as_ref()
            .map_or(Value::Null, |t| Value::String(remap_str(t, remap))),
    );
}

pub(crate) fn serialize_material(
    mat: &MaterialSpec,
    uv_mode: CompatMode,
    remap: &BTreeMap<String, String>,
) -> Value {
    let mut obj = Map::new();
    obj.insert(
        "zoidiumUvMode".to_owned(),
        Value::String(uv_mode.as_str().to_owned()),
    );

    match mat {
        MaterialSpec::SingleColor { color, side } => {
            obj.insert("type".to_owned(), Value::String("singlecolor".to_owned()));
            obj.insert(
                "properties".to_owned(),
                json!({
                    "color": serialize_property_track(color, remap),
                    "side": *side,
                }),
            );
        }
        MaterialSpec::Texture(m) => {
            obj.insert("type".to_owned(), Value::String("texture".to_owned()));
            obj.insert(
                "properties".to_owned(),
                json!({
                    "texture": m.texture.as_ref().map_or(Value::Null, |t| Value::String(remap_str(t, remap))),
                    "wrap": m.wrap,
                    "repeat": serialize_property_track(&m.repeat, remap),
                    "offset": serialize_property_track(&m.offset, remap),
                    "center": serialize_property_track(&m.center, remap),
                    "rotation": serialize_property_track(&m.rotation, remap),
                    "side": m.side,
                }),
            );
        }
        MaterialSpec::Video(m) => {
            obj.insert("type".to_owned(), Value::String("video".to_owned()));
            obj.insert(
                "properties".to_owned(),
                json!({
                    "video": m.video.as_ref().map_or(Value::Null, |v| Value::String(remap_str(v, remap))),
                    "startOffset": m.start_offset,
                    "offset": m.media_offset,
                    "color": serialize_property_track(&m.color, remap),
                    "transparent": u32::from(m.transparent),
                    "opacity": serialize_property_track(&m.opacity, remap),
                    "blending": m.blending,
                }),
            );
        }
        MaterialSpec::CustomPbr(m) => {
            obj.insert(
                "type".to_owned(),
                Value::String(
                    if m.uv_normalized {
                        "uvcustom"
                    } else {
                        "custom"
                    }
                    .to_owned(),
                ),
            );
            obj.insert(
                "properties".to_owned(),
                json!({
                    "color": serialize_property_track(&m.color, remap),
                    "emissive": serialize_property_track(&m.emissive, remap),
                    "roughness": serialize_static_or_animated_track(&m.roughness, remap),
                    "metalness": serialize_static_or_animated_track(&m.metalness, remap),
                    "texture": m.texture.as_ref().map_or(Value::Null, |t| Value::String(remap_str(t, remap))),
                    "transparent": u32::from(m.transparent),
                    "opacity": serialize_property_track(&m.opacity, remap),
                    "blending": m.blending,
                    "side": m.side,
                    "normalMap": m.normal_map.as_ref().map_or(Value::Null, |n| Value::String(remap_str(n, remap))),
                    "normalScale": finite_f64(m.normal_scale),
                    "wrap": m.wrap,
                    "repeat": serialize_property_track(&m.repeat, remap),
                    "offset": serialize_property_track(&m.offset, remap),
                    "center": serialize_property_track(&m.center, remap),
                    "rotation": serialize_property_track(&m.rotation, remap),
                    "reflection": u32::from(m.reflection),
                }),
            );
        }
        MaterialSpec::CustomShader {
            frag_shader,
            custom_properties,
        } => {
            obj.insert("type".to_owned(), Value::String("shader".to_owned()));
            obj.insert(
                "properties".to_owned(),
                json!({ "fragShader": frag_shader }),
            );
            if !custom_properties.is_empty() {
                obj.insert(
                    "customProperties".to_owned(),
                    Value::Array(serialize_custom_properties(custom_properties, remap)),
                );
            }
        }
        MaterialSpec::PbrPlus(m) => {
            obj.insert("type".to_owned(), Value::String("pbrplus".to_owned()));
            let mut props = Map::new();
            for (k, v) in &m.preserved {
                if matches!(k.as_str(), "baseColor" | "baseColorMap") {
                    continue;
                }
                let mut cloned = v.clone();
                remap_json_value(&mut cloned, remap);
                props.insert(k.clone(), cloned);
            }
            props.insert(
                "color".to_owned(),
                serialize_property_track(&m.base_color, remap),
            );
            props.insert(
                "texture".to_owned(),
                m.base_color_map
                    .as_ref()
                    .map_or(Value::Null, |id| Value::String(remap_str(id, remap))),
            );
            props.insert(
                "emissive".to_owned(),
                serialize_property_track(&m.emissive, remap),
            );
            props.insert(
                "emissiveIntensity".to_owned(),
                serialize_property_track(&m.emissive_intensity, remap),
            );
            if let Some(id) = &m.emissive_map {
                props.insert(
                    "emissiveMap".to_owned(),
                    Value::String(remap_str(id, remap)),
                );
            }
            props.insert(
                "roughness".to_owned(),
                serialize_property_track(&m.roughness, remap),
            );
            if let Some(id) = &m.roughness_map {
                props.insert(
                    "roughnessMap".to_owned(),
                    Value::String(remap_str(id, remap)),
                );
            }
            props.insert(
                "metalness".to_owned(),
                serialize_property_track(&m.metalness, remap),
            );
            if let Some(id) = &m.metalness_map {
                props.insert(
                    "metalnessMap".to_owned(),
                    Value::String(remap_str(id, remap)),
                );
            }
            if let Some(id) = &m.normal_map {
                props.insert("normalMap".to_owned(), Value::String(remap_str(id, remap)));
            }
            props.insert(
                "normalScale".to_owned(),
                m.normal_scale_track.as_ref().map_or_else(
                    || serialize_property_track(&PropertyTrack::number(m.normal_scale), remap),
                    |t| serialize_property_track(t, remap),
                ),
            );
            if let Some(id) = &m.alpha_map {
                props.insert("alphaMap".to_owned(), Value::String(remap_str(id, remap)));
            }
            if let Some(id) = &m.ao_texture {
                props.insert("aoTexture".to_owned(), Value::String(remap_str(id, remap)));
            }
            props.insert("alphaTest".to_owned(), json!(finite_f64(m.alpha_test)));
            props.insert("reflection".to_owned(), json!(u32::from(m.reflection)));
            props.insert(
                "opacity".to_owned(),
                serialize_property_track(&m.opacity, remap),
            );
            props.insert("transparent".to_owned(), json!(u32::from(m.transparent)));
            props.insert("blending".to_owned(), json!(m.blending));
            props.insert("side".to_owned(), json!(m.side));
            props.insert("wrap".to_owned(), json!(m.wrap));
            props.insert(
                "repeat".to_owned(),
                serialize_property_track(&m.repeat, remap),
            );
            props.insert(
                "offset".to_owned(),
                serialize_property_track(&m.offset, remap),
            );
            props.insert(
                "center".to_owned(),
                serialize_property_track(&m.center, remap),
            );
            props.insert(
                "rotation".to_owned(),
                serialize_property_track(&m.rotation, remap),
            );
            props.insert(
                "reflectionIntensity".to_owned(),
                m.reflection_intensity_track.as_ref().map_or_else(
                    || {
                        serialize_property_track(
                            &PropertyTrack::number(m.reflection_intensity),
                            remap,
                        )
                    },
                    |t| serialize_property_track(t, remap),
                ),
            );
            obj.insert("properties".to_owned(), Value::Object(props));
        }
        MaterialSpec::Matcap(m) => {
            obj.insert("type".to_owned(), Value::String("matcap".to_owned()));
            obj.insert(
                "properties".to_owned(),
                json!({
                    "matcap": m.texture.as_ref().map_or(Value::Null, |t| Value::String(remap_str(t, remap))),
                    "color": serialize_property_track(&m.color, remap),
                    "brightness": serialize_property_track(&m.brightness, remap),
                    "opacity": serialize_property_track(&m.opacity, remap),
                    "transparent": u32::from(m.transparent),
                    "blending": m.blending,
                    "side": m.side,
                }),
            );
        }
        MaterialSpec::ImagePlus(m) => {
            obj.insert("type".to_owned(), Value::String("imageplus".to_owned()));
            obj.insert(
                "properties".to_owned(),
                json!({
                    "texture": m.texture.as_ref().map_or(Value::Null, |t| Value::String(remap_str(t, remap))),
                    "opacity": serialize_property_track(&m.opacity, remap),
                    "transparent": u32::from(m.transparent),
                    "blending": m.blending,
                    "wrap": m.wrap,
                    "repeat": serialize_property_track(&m.repeat, remap),
                    "offset": serialize_property_track(&m.offset, remap),
                    "center": serialize_property_track(&m.center, remap),
                    "rotation": serialize_property_track(&m.rotation, remap),
                    "side": m.side,
                }),
            );
        }
        MaterialSpec::NodeMaterial {
            resolution, graph, ..
        } => {
            let mut cloned = graph.clone();
            remap_json_value(&mut cloned, remap);
            let bake_idx = match *resolution {
                128 => 0,
                256 => 1,
                512 => 2,
                1024 => 3,
                2048 => 4,
                other => other,
            };
            let is_already_zoidium_wrapper = cloned.as_object().is_some_and(|m| {
                m.get("properties")
                    .and_then(Value::as_object)
                    .is_some_and(|p| p.contains_key("graph"))
            });
            if is_already_zoidium_wrapper && let Value::Object(mut map) = cloned {
                map.insert("type".to_owned(), Value::String("nodes".to_owned()));
                map.insert(
                    "zoidiumUvMode".to_owned(),
                    Value::String(uv_mode.as_str().to_owned()),
                );
                let props = map
                    .entry("properties".to_owned())
                    .or_insert_with(|| Value::Object(Map::new()));
                if let Some(pmap) = props.as_object_mut() {
                    pmap.insert("bakeResolution".to_owned(), json!(bake_idx));
                }
                return Value::Object(map);
            }
            // Clean native NodeMaterial graph: serialize inside `properties.graph` with
            // `zoidcraftGraphDirect: true` so `.pz` export stays Zoidium-compatible AND
            // re-import recovers the exact clean `graph` without injecting wrapper keys.
            let graph_json_str = serde_json::to_string(&cloned).unwrap_or_else(|_| "{}".to_owned());
            obj.insert("type".to_owned(), Value::String("nodes".to_owned()));
            obj.insert(
                "zoidiumUvMode".to_owned(),
                Value::String(uv_mode.as_str().to_owned()),
            );
            obj.insert(
                "properties".to_owned(),
                json!({
                    "bakeResolution": bake_idx,
                    "graph": graph_json_str,
                    "zoidcraftGraphDirect": true,
                }),
            );
        }
        MaterialSpec::Unknown {
            material_type,
            payload,
        } => {
            let mut cloned = payload.clone();
            remap_json_value(&mut cloned, remap);
            if let Value::Object(mut map) = cloned {
                map.entry("type".to_owned())
                    .or_insert_with(|| Value::String(material_type.clone()));
                map.insert(
                    "zoidiumUvMode".to_owned(),
                    Value::String(uv_mode.as_str().to_owned()),
                );
                return Value::Object(map);
            }
            obj.insert("type".to_owned(), Value::String(material_type.clone()));
        }
    }

    Value::Object(obj)
}
