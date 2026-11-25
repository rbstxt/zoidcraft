use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::Session;
use crate::document::SessionDocument;
use crate::property_conv::prop_f64_static;

impl SessionDocument {
    pub(crate) fn collect_plugin_dependencies(&self) -> BTreeSet<String> {
        let mut deps = BTreeSet::new();
        fn inspect_kind(kind: &str, deps: &mut BTreeSet<String>) {
            if let Some(plugin_id) = extract_namespaced_plugin_id(kind) {
                deps.insert(plugin_id.to_string());
            }
        }

        for track in self.all_tracks() {
            if track.is_3d {
                deps.insert("camera-plus".to_string());
            }
            for clip in &track.clips {
                inspect_kind(&clip.kind, &mut deps);
                for obj in &clip.objects_3d {
                    inspect_kind(&obj.kind, &mut deps);
                    if let Some(mtype) = prop_f64_static(&obj.properties, "material.type")
                        && matches!(mtype.round() as i64, 5..=8)
                    {
                        deps.insert("material-plus".to_string());
                    }
                    if obj.properties.contains_key("dofEnabled")
                        || obj.properties.contains_key("dofFocusDistance")
                        || obj.properties.contains_key("dofAperture")
                        || obj.properties.contains_key("dofFocusAreaWidth")
                        || obj.properties.contains_key("dofNearBlur")
                        || obj.properties.contains_key("dofFarBlur")
                        || obj.properties.keys().any(|k| k.starts_with("vibrate."))
                    {
                        deps.insert("camera-plus".to_string());
                    }
                    if prop_f64_static(&obj.properties, "constraint.trackTo.enabled")
                        .is_some_and(|v| v.round() as i64 != 0)
                        || prop_f64_static(&obj.properties, "constraint.follow.enabled")
                            .is_some_and(|v| v.round() as i64 != 0)
                    {
                        deps.insert("constraints".to_string());
                    }
                    if obj.properties.values().any(|p| p.expression.is_some()) {
                        deps.insert("expressions-plus".to_string());
                    }
                }
                for fx in &clip.effects {
                    inspect_kind(&fx.kind, &mut deps);
                    if let Some(plugin_id) = zc_project::zoidium_effect_plugin_id(&fx.kind)
                        && !plugin_id.is_empty()
                    {
                        deps.insert(plugin_id.to_string());
                    }
                    if fx.properties.values().any(|p| p.expression.is_some()) {
                        deps.insert("expressions-plus".to_string());
                    }
                }
                if clip.properties.values().any(|p| p.expression.is_some()) {
                    deps.insert("expressions-plus".to_string());
                }
            }
        }
        if self
            .sequence_properties
            .keys()
            .any(|k| k.starts_with("ctl_"))
            || self
                .sequence_properties
                .values()
                .any(|p| p.expression.is_some())
        {
            deps.insert("expressions-plus".to_string());
        }
        deps
    }
}

impl Session {
    pub(crate) fn refresh_plugin_requirements_from_document(&mut self) {
        let new_doc_deps = self.document.collect_plugin_dependencies();
        for old_dep in &self.prev_doc_plugin_deps {
            if !new_doc_deps.contains(old_dep)
                && !self.external_plugin_requirements.contains(old_dep)
            {
                self.project.plugin_requirements.remove(old_dep);
            }
        }
        for dep in &new_doc_deps {
            self.project.plugin_requirements.insert(dep.clone());
        }
        self.prev_doc_plugin_deps = new_doc_deps;
    }
}

pub(crate) fn extract_namespaced_plugin_id(s: &str) -> Option<&str> {
    let rest = s
        .strip_prefix("zoidium:")
        .or_else(|| s.strip_prefix("zoidcraft:"))?;
    let (plugin_id, _) = rest.split_once('/')?;
    if plugin_id.is_empty() {
        None
    } else {
        Some(plugin_id)
    }
}

fn track_has_expression(track: &zc_project::PropertyTrack) -> bool {
    track.expression.is_some() || track.channels.iter().any(track_has_expression)
}

fn collect_plugin_deps_from_preserved(
    preserved: &BTreeMap<String, Value>,
    deps: &mut BTreeSet<String>,
) {
    for (k, v) in preserved {
        if matches!(k.as_str(), "required_plugins" | "plugin_requirements")
            && let Some(req) = v.as_array()
        {
            for item in req {
                if let Some(id) = item
                    .as_str()
                    .or_else(|| item.get("id").and_then(Value::as_str))
                    && !id.is_empty()
                {
                    deps.insert(id.to_string());
                }
            }
        }
        collect_plugin_deps_from_json(v, deps);
    }
}

fn collect_plugin_deps_from_material(mat: &zc_project::MaterialSpec, deps: &mut BTreeSet<String>) {
    match mat {
        zc_project::MaterialSpec::SingleColor { color, .. } => {
            if track_has_expression(color) {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::Texture(t) => {
            if track_has_expression(&t.repeat)
                || track_has_expression(&t.offset)
                || track_has_expression(&t.center)
                || track_has_expression(&t.rotation)
            {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::Video(v) => {
            if track_has_expression(&v.color) || track_has_expression(&v.opacity) {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::CustomPbr(p) => {
            if track_has_expression(&p.color)
                || track_has_expression(&p.emissive)
                || track_has_expression(&p.roughness)
                || track_has_expression(&p.metalness)
                || track_has_expression(&p.opacity)
                || track_has_expression(&p.repeat)
                || track_has_expression(&p.offset)
                || track_has_expression(&p.center)
                || track_has_expression(&p.rotation)
            {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::CustomShader {
            custom_properties, ..
        } => {
            for cp in custom_properties {
                if track_has_expression(&cp.track) {
                    deps.insert("expressions-plus".to_string());
                }
                collect_plugin_deps_from_preserved(&cp.metadata, deps);
            }
        }
        zc_project::MaterialSpec::PbrPlus(p) => {
            deps.insert("material-plus".to_string());
            if track_has_expression(&p.base_color)
                || track_has_expression(&p.emissive)
                || track_has_expression(&p.roughness)
                || track_has_expression(&p.metalness)
                || track_has_expression(&p.opacity)
                || track_has_expression(&p.repeat)
                || track_has_expression(&p.offset)
                || track_has_expression(&p.center)
                || track_has_expression(&p.rotation)
            {
                deps.insert("expressions-plus".to_string());
            }
            collect_plugin_deps_from_preserved(&p.preserved, deps);
        }
        zc_project::MaterialSpec::Matcap(m) => {
            deps.insert("material-plus".to_string());
            if track_has_expression(&m.color)
                || track_has_expression(&m.brightness)
                || track_has_expression(&m.opacity)
            {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::ImagePlus(i) => {
            deps.insert("material-plus".to_string());
            if track_has_expression(&i.opacity)
                || track_has_expression(&i.repeat)
                || track_has_expression(&i.offset)
                || track_has_expression(&i.center)
                || track_has_expression(&i.rotation)
            {
                deps.insert("expressions-plus".to_string());
            }
        }
        zc_project::MaterialSpec::NodeMaterial { graph, .. } => {
            deps.insert("material-plus".to_string());
            collect_plugin_deps_from_json(graph, deps);
        }
        zc_project::MaterialSpec::Unknown {
            material_type,
            payload,
        } => {
            if let Some(pid) = extract_namespaced_plugin_id(material_type) {
                deps.insert(pid.to_string());
            }
            collect_plugin_deps_from_json(payload, deps);
        }
    }
}

fn collect_plugin_deps_from_effects(
    effects: &[zc_project::EffectInstance],
    deps: &mut BTreeSet<String>,
) {
    for fx in effects {
        if track_has_expression(&fx.enabled) || fx.params.values().any(track_has_expression) {
            deps.insert("expressions-plus".to_string());
        }
        collect_plugin_deps_from_preserved(&fx.preserved, deps);
        match &fx.kind {
            zc_project::EffectKind::Plugin { plugin_id, .. } => {
                if !plugin_id.is_empty() {
                    deps.insert(plugin_id.clone());
                }
            }
            zc_project::EffectKind::CustomShader {
                custom_properties, ..
            } => {
                for cp in custom_properties {
                    if track_has_expression(&cp.track) {
                        deps.insert("expressions-plus".to_string());
                    }
                    collect_plugin_deps_from_preserved(&cp.metadata, deps);
                }
            }
            zc_project::EffectKind::EffectGroup {
                custom_properties,
                children,
            } => {
                for cp in custom_properties {
                    if track_has_expression(&cp.track) {
                        deps.insert("expressions-plus".to_string());
                    }
                    collect_plugin_deps_from_preserved(&cp.metadata, deps);
                }
                collect_plugin_deps_from_effects(children, deps);
            }
            zc_project::EffectKind::Standard(_) => {}
        }
    }
}

fn collect_plugin_deps_from_objects(objects: &[zc_project::Object3D], deps: &mut BTreeSet<String>) {
    for obj in objects {
        if track_has_expression(&obj.enabled)
            || track_has_expression(&obj.position)
            || track_has_expression(&obj.rotation)
            || track_has_expression(&obj.scale)
        {
            deps.insert("expressions-plus".to_string());
        }
        if let Some(tt) = &obj.constraints.track_to {
            deps.insert("constraints".to_string());
            if track_has_expression(&tt.influence) {
                deps.insert("expressions-plus".to_string());
            }
        }
        if let Some(fl) = &obj.constraints.follow {
            deps.insert("constraints".to_string());
            if track_has_expression(&fl.position_offset)
                || track_has_expression(&fl.rotation_offset)
                || track_has_expression(&fl.scale_offset)
            {
                deps.insert("expressions-plus".to_string());
            }
        }
        if let Some(mat) = &obj.material {
            collect_plugin_deps_from_material(mat, deps);
        }
        for mat in obj.material_slots.values() {
            collect_plugin_deps_from_material(mat, deps);
        }
        collect_plugin_deps_from_preserved(&obj.preserved, deps);
        match &obj.kind {
            zc_project::Object3DKind::Group {
                custom_properties,
                children,
                ..
            } => {
                for cp in custom_properties {
                    if track_has_expression(&cp.track) {
                        deps.insert("expressions-plus".to_string());
                    }
                    collect_plugin_deps_from_preserved(&cp.metadata, deps);
                }
                collect_plugin_deps_from_objects(children, deps);
            }
            zc_project::Object3DKind::Shape3D(shape) => {
                if matches!(
                    shape,
                    zc_project::Shape3DKind::Cone { .. }
                        | zc_project::Shape3DKind::RoundedBox { .. }
                        | zc_project::Shape3DKind::Tetrahedron { .. }
                        | zc_project::Shape3DKind::Octahedron { .. }
                        | zc_project::Shape3DKind::Dodecahedron { .. }
                        | zc_project::Shape3DKind::Icosahedron { .. }
                        | zc_project::Shape3DKind::Capsule { .. }
                        | zc_project::Shape3DKind::TorusKnot { .. }
                        | zc_project::Shape3DKind::Tube { .. }
                        | zc_project::Shape3DKind::Gear { .. }
                        | zc_project::Shape3DKind::Helix { .. }
                        | zc_project::Shape3DKind::Spline { .. }
                ) {
                    deps.insert("geometry-plus".to_string());
                }
            }
            zc_project::Object3DKind::Light(l) => {
                if track_has_expression(&l.color)
                    || track_has_expression(&l.ground_color)
                    || track_has_expression(&l.target)
                    || track_has_expression(&l.intensity)
                    || track_has_expression(&l.angle)
                {
                    deps.insert("expressions-plus".to_string());
                }
            }
            zc_project::Object3DKind::Camera(c) => {
                if track_has_expression(&c.fov)
                    || track_has_expression(&c.zoom)
                    || track_has_expression(&c.shake_amplitude)
                    || track_has_expression(&c.shake_speed)
                    || track_has_expression(&c.focal_length)
                    || track_has_expression(&c.film_offset_x)
                    || track_has_expression(&c.film_offset_y)
                {
                    deps.insert("expressions-plus".to_string());
                }
                if let Some(mb) = &c.motion_blur {
                    deps.insert("camera-plus".to_string());
                    if track_has_expression(&mb.shutter) {
                        deps.insert("expressions-plus".to_string());
                    }
                }
                if let Some(dof) = &c.dof {
                    deps.insert("camera-plus".to_string());
                    if track_has_expression(&dof.focus_distance)
                        || track_has_expression(&dof.aperture)
                        || track_has_expression(&dof.focus_area_width)
                        || track_has_expression(&dof.near_blur)
                        || track_has_expression(&dof.far_blur)
                    {
                        deps.insert("expressions-plus".to_string());
                    }
                }
                if let Some(vib) = &c.vibrate {
                    deps.insert("camera-plus".to_string());
                    if track_has_expression(&vib.position_amplitude)
                        || track_has_expression(&vib.rotation_amplitude)
                        || track_has_expression(&vib.frequency)
                        || track_has_expression(&vib.position_frequency)
                        || track_has_expression(&vib.rotation_frequency)
                    {
                        deps.insert("expressions-plus".to_string());
                    }
                }
            }
            zc_project::Object3DKind::Particles(p) => {
                if p.sprite_preset.is_some() {
                    deps.insert("particles-plus".to_string());
                }
                if track_has_expression(&p.time) {
                    deps.insert("expressions-plus".to_string());
                }
            }
            zc_project::Object3DKind::Model(m) => {
                if m.model_properties.values().any(track_has_expression) {
                    deps.insert("expressions-plus".to_string());
                }
            }
            zc_project::Object3DKind::Dummy {
                original_type,
                payload,
            } => {
                if let Some(t) = original_type
                    && let Some(pid) = extract_namespaced_plugin_id(t)
                {
                    deps.insert(pid.to_string());
                }
                collect_plugin_deps_from_json(payload, deps);
            }
            zc_project::Object3DKind::Text3D(_) => {}
        }
    }
}

fn collect_plugin_deps_from_layer(layer: &zc_project::Layer, deps: &mut BTreeSet<String>) {
    if track_has_expression(&layer.position)
        || track_has_expression(&layer.scale)
        || track_has_expression(&layer.rotation)
        || track_has_expression(&layer.anchor)
        || track_has_expression(&layer.opacity)
    {
        deps.insert("expressions-plus".to_string());
    }
    collect_plugin_deps_from_preserved(&layer.preserved, deps);
    collect_plugin_deps_from_effects(&layer.effects, deps);
    match &layer.kind {
        zc_project::LayerKind::Scene3D { objects, .. } => {
            collect_plugin_deps_from_objects(objects, deps);
        }
        zc_project::LayerKind::Composite { layers } => {
            for child in layers {
                collect_plugin_deps_from_layer(child, deps);
            }
        }
        zc_project::LayerKind::Shape2D { shapes }
        | zc_project::LayerKind::Text2D { shapes, .. }
        | zc_project::LayerKind::PresetShape2D { shapes, .. } => {
            for shape in shapes {
                collect_plugin_deps_from_preserved(&shape.preserved, deps);
            }
        }
        zc_project::LayerKind::CameraLayer { .. } => {
            deps.insert("camera-plus".to_string());
        }
        zc_project::LayerKind::PrecompRef { .. } => {
            deps.insert("precomp-plus".to_string());
        }
        zc_project::LayerKind::Plugin {
            plugin_type,
            payload,
        } => {
            if let Some(pid) = extract_namespaced_plugin_id(plugin_type) {
                deps.insert(pid.to_string());
            }
            collect_plugin_deps_from_json(payload, deps);
        }
        _ => {}
    }
}

fn collect_plugin_deps_from_sequence(seq: &zc_project::Sequence, deps: &mut BTreeSet<String>) {
    if !seq.controls.is_empty()
        || track_has_expression(&seq.motion_blur.samples)
        || track_has_expression(&seq.motion_blur.shutter)
        || seq.controls.iter().any(|c| track_has_expression(&c.track))
    {
        deps.insert("expressions-plus".to_string());
    }
    collect_plugin_deps_from_preserved(&seq.preserved, deps);
    for ctl in &seq.controls {
        collect_plugin_deps_from_preserved(&ctl.preserved, deps);
    }
    for vt in &seq.video_tracks {
        if vt.is_3d {
            deps.insert("camera-plus".to_string());
        }
        collect_plugin_deps_from_preserved(&vt.preserved, deps);
        for vc in &vt.clips {
            if track_has_expression(&vc.time_curve) {
                deps.insert("expressions-plus".to_string());
            }
            collect_plugin_deps_from_preserved(&vc.preserved, deps);
            collect_plugin_deps_from_layer(&vc.layer, deps);
        }
    }
    for at in &seq.audio_tracks {
        collect_plugin_deps_from_preserved(&at.preserved, deps);
        for ac in &at.clips {
            if track_has_expression(&ac.time_curve)
                || track_has_expression(&ac.audio.volume)
                || track_has_expression(&ac.audio.pan)
            {
                deps.insert("expressions-plus".to_string());
            }
            collect_plugin_deps_from_preserved(&ac.preserved, deps);
        }
    }
    for masks in seq.layer_masks.values() {
        for m in masks {
            collect_plugin_deps_from_preserved(&m.preserved, deps);
        }
    }
    for cam_objs in seq.camera_layer_objects.values() {
        collect_plugin_deps_from_objects(cam_objs, deps);
    }
}

pub(crate) fn collect_plugin_deps_from_project(
    project: &zc_project::Project,
    deps: &mut BTreeSet<String>,
) {
    collect_plugin_deps_from_preserved(&project.preserved, deps);
    for asset in project.assets.values() {
        collect_plugin_deps_from_preserved(&asset.preserved, deps);
    }
    for media in &project.media {
        collect_plugin_deps_from_preserved(&media.preserved, deps);
    }
    collect_plugin_deps_from_sequence(&project.sequence, deps);
    for comp in project.compositions.values() {
        collect_plugin_deps_from_sequence(comp, deps);
    }
}

fn collect_plugin_deps_from_json(value: &Value, deps: &mut BTreeSet<String>) {
    match value {
        Value::String(s) => {
            if let Some(plugin_id) = extract_namespaced_plugin_id(s) {
                deps.insert(plugin_id.to_string());
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_plugin_deps_from_json(item, deps);
            }
        }
        Value::Object(map) => {
            for field in ["required_plugins", "plugin_requirements"] {
                if let Some(req) = map.get(field).and_then(Value::as_array) {
                    for item in req {
                        if let Some(id) = item
                            .as_str()
                            .or_else(|| item.get("id").and_then(Value::as_str))
                            && !id.is_empty()
                        {
                            deps.insert(id.to_string());
                        }
                    }
                }
            }
            for child in map.values() {
                collect_plugin_deps_from_json(child, deps);
            }
        }
        _ => {}
    }
}
