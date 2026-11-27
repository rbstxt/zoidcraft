use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde_json::{Value, json};

use crate::document::{SessionObject3D, SessionProperty};
use crate::plugin_defaults::{
    PBR_PLUS_SURFACE_KEYS, canonicalize_object_3d_kind, default_particles_data,
    is_3d_camera_plus_kind, is_3d_container_kind, is_3d_expresstext_plus_kind,
    is_3d_light_plus_kind, is_3d_optical_flare_kind, is_3d_repeater_kind, is_3d_text_plus_kind,
    is_3d_trapcode_kind, is_core_object_3d_property, plugin_light_kind,
    session_property_to_preserved_value, uses_preserved_plugin_properties,
};
use crate::property_conv::{
    euler_order_from_value, parse_preserved_bool_u64, prop_f64_static, prop_str_static,
    prop_vec2_static, prop_vec3_static, prop_vec4_static, property_track_matches_session,
    session_property_to_track,
};
use crate::sync_from_project::{material_type_index, parse_uuid_or_new};

fn sync_material_from_session(
    props: &BTreeMap<String, SessionProperty>,
    mat_slot: &mut Option<zc_project::MaterialSpec>,
) {
    sync_material_from_session_with_prefix(props, "material.", mat_slot);
}

fn sync_material_from_session_with_prefix(
    props: &BTreeMap<String, SessionProperty>,
    prefix: &str,
    mat_slot: &mut Option<zc_project::MaterialSpec>,
) {
    let has_any_mat_prop = props.keys().any(|k| k.starts_with(prefix));
    if !has_any_mat_prop {
        return;
    }

    struct PrefixKeyBuf {
        buf: String,
        prefix_len: usize,
    }
    impl PrefixKeyBuf {
        fn new(prefix: &str) -> Self {
            let mut buf = String::with_capacity(prefix.len() + 32);
            buf.push_str(prefix);
            Self {
                buf,
                prefix_len: prefix.len(),
            }
        }
        fn key(&mut self, suffix: &str) -> &str {
            self.buf.truncate(self.prefix_len);
            self.buf.push_str(suffix);
            &self.buf
        }
        fn key2(&mut self, mid: &str, suffix: &str) -> &str {
            self.buf.truncate(self.prefix_len);
            self.buf.push_str(mid);
            self.buf.push_str(suffix);
            &self.buf
        }
    }

    let mut kbuf = PrefixKeyBuf::new(prefix);
    let desired_type =
        prop_f64_static(props, kbuf.key("type")).map(|v| v.round().clamp(0.0, 8.0) as u32);
    if let Some(t_idx) = desired_type {
        let current_idx = mat_slot.as_ref().and_then(material_type_index);
        if current_idx != Some(t_idx) {
            *mat_slot = Some(match t_idx {
                0 => zc_project::MaterialSpec::single_color_default(),
                1 => zc_project::MaterialSpec::Texture(Box::default()),
                2 => zc_project::MaterialSpec::Video(Box::default()),
                3 => zc_project::MaterialSpec::CustomPbr(Box::default()),
                4 => zc_project::MaterialSpec::CustomShader {
                    frag_shader: String::new(),
                    custom_properties: Vec::new(),
                },
                5 => zc_project::MaterialSpec::PbrPlus(Box::default()),
                6 => zc_project::MaterialSpec::Matcap(Box::default()),
                7 => zc_project::MaterialSpec::ImagePlus(Box::default()),
                8 => zc_project::MaterialSpec::NodeMaterial {
                    schema_version: 1,
                    resolution: 256,
                    graph: json!({}),
                },
                _ => zc_project::MaterialSpec::CustomPbr(Box::default()),
            });
        }
    }
    if mat_slot.is_none() {
        *mat_slot = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
    }
    let Some(mat) = mat_slot.as_mut() else {
        return;
    };
    let has_tex_prop = props.contains_key(kbuf.key("texture"));
    let tex_opt = prop_str_static(props, kbuf.key("texture")).filter(|s| !s.is_empty());
    match mat {
        zc_project::MaterialSpec::SingleColor { color, side } => {
            if let Some(p) = props.get(kbuf.key("color")) {
                *color = session_property_to_track(p, Some(color));
            }
            if let Some(s) = prop_f64_static(props, kbuf.key("side")) {
                *side = s.round().clamp(0.0, 2.0) as u32;
            }
        }
        zc_project::MaterialSpec::CustomPbr(pbr) => {
            if let Some(p) = props.get(kbuf.key("color")) {
                pbr.color = session_property_to_track(p, Some(&pbr.color));
            }
            if let Some(p) = props.get(kbuf.key("emissive")) {
                pbr.emissive = session_property_to_track(p, Some(&pbr.emissive));
            }
            if let Some(p) = props.get(kbuf.key("roughness")) {
                pbr.roughness = session_property_to_track(p, Some(&pbr.roughness));
            }
            if let Some(p) = props.get(kbuf.key("metalness")) {
                pbr.metalness = session_property_to_track(p, Some(&pbr.metalness));
            }
            let prev_transparent = pbr.transparent;
            if let Some(tr) = prop_f64_static(props, kbuf.key("transparent")) {
                pbr.transparent = tr.round() as i64 != 0;
            }
            let transparent_explicitly_changed = pbr.transparent != prev_transparent;
            if let Some(p) = props.get(kbuf.key("opacity")) {
                let opacity_changed = !property_track_matches_session(&pbr.opacity, p);
                pbr.opacity = session_property_to_track(p, Some(&pbr.opacity));
                let op = prop_f64_static(props, kbuf.key("opacity")).unwrap_or(1.0);
                if opacity_changed && !transparent_explicitly_changed && (op < 0.999 || p.animated)
                {
                    pbr.transparent = true;
                }
            }
            if let Some(b) = prop_f64_static(props, kbuf.key("blending")) {
                pbr.blending = b.round().clamp(0.0, 4.0) as u32;
            }
            if let Some(s) = prop_f64_static(props, kbuf.key("side")) {
                pbr.side = s.round().clamp(0.0, 2.0) as u32;
            }
            if has_tex_prop {
                pbr.texture = tex_opt;
            }
            if props.contains_key(kbuf.key("normalMap")) {
                pbr.normal_map =
                    prop_str_static(props, kbuf.key("normalMap")).filter(|s| !s.is_empty());
            }
            if let Some(ns) = prop_f64_static(props, kbuf.key("normalScale"))
                .or_else(|| prop_vec2_static(props, kbuf.key("normalScale")).map(|v| v[0]))
            {
                pbr.normal_scale = ns;
            }
            if let Some(w) = prop_f64_static(props, kbuf.key("wrap")) {
                pbr.wrap = w.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(p) = props.get(kbuf.key("repeat")) {
                pbr.repeat = session_property_to_track(p, Some(&pbr.repeat));
            }
            if let Some(p) = props.get(kbuf.key("offset")) {
                pbr.offset = session_property_to_track(p, Some(&pbr.offset));
            }
            if let Some(p) = props.get(kbuf.key("center")) {
                pbr.center = session_property_to_track(p, Some(&pbr.center));
            }
            if let Some(p) = props.get(kbuf.key("rotation")) {
                pbr.rotation = session_property_to_track(p, Some(&pbr.rotation));
            }
            if let Some(r) = prop_f64_static(props, kbuf.key("reflection")) {
                pbr.reflection = r.round() as i64 != 0;
            }
            if let Some(uv) = prop_f64_static(props, kbuf.key("uvNormalized")) {
                pbr.uv_normalized = uv.round() as i64 != 0;
            }
        }
        zc_project::MaterialSpec::PbrPlus(pbr) => {
            if let Some(p) = props.get(kbuf.key("color")) {
                pbr.base_color = session_property_to_track(p, Some(&pbr.base_color));
            }
            if let Some(p) = props.get(kbuf.key("emissive")) {
                pbr.emissive = session_property_to_track(p, Some(&pbr.emissive));
            }
            if let Some(p) = props.get(kbuf.key("roughness")) {
                pbr.roughness = session_property_to_track(p, Some(&pbr.roughness));
            }
            if let Some(p) = props.get(kbuf.key("metalness")) {
                pbr.metalness = session_property_to_track(p, Some(&pbr.metalness));
            }
            let prev_transparent = pbr.transparent;
            if let Some(tr) = prop_f64_static(props, kbuf.key("transparent")) {
                pbr.transparent = tr.round() as i64 != 0;
            }
            let transparent_explicitly_changed = pbr.transparent != prev_transparent;
            if let Some(p) = props.get(kbuf.key("opacity")) {
                let opacity_changed = !property_track_matches_session(&pbr.opacity, p);
                pbr.opacity = session_property_to_track(p, Some(&pbr.opacity));
                let op = prop_f64_static(props, kbuf.key("opacity")).unwrap_or(1.0);
                if opacity_changed && !transparent_explicitly_changed && (op < 0.999 || p.animated)
                {
                    pbr.transparent = true;
                }
            }
            if let Some(b) = prop_f64_static(props, kbuf.key("blending")) {
                pbr.blending = b.round().clamp(0.0, 4.0) as u32;
            }
            if let Some(s) = prop_f64_static(props, kbuf.key("side")) {
                pbr.side = s.round().clamp(0.0, 2.0) as u32;
            }
            if has_tex_prop {
                pbr.base_color_map = tex_opt;
            } else if props.contains_key(kbuf.key("baseColorMap")) {
                pbr.base_color_map =
                    prop_str_static(props, kbuf.key("baseColorMap")).filter(|s| !s.is_empty());
            }
            if props.contains_key(kbuf.key("emissiveMap")) {
                pbr.emissive_map =
                    prop_str_static(props, kbuf.key("emissiveMap")).filter(|s| !s.is_empty());
            }
            if props.contains_key(kbuf.key("roughnessMap")) {
                pbr.roughness_map =
                    prop_str_static(props, kbuf.key("roughnessMap")).filter(|s| !s.is_empty());
            }
            if props.contains_key(kbuf.key("metalnessMap")) {
                pbr.metalness_map =
                    prop_str_static(props, kbuf.key("metalnessMap")).filter(|s| !s.is_empty());
            }
            if props.contains_key(kbuf.key("normalMap")) {
                pbr.normal_map =
                    prop_str_static(props, kbuf.key("normalMap")).filter(|s| !s.is_empty());
            }
            if let Some(p) = props.get(kbuf.key("normalScale")) {
                if let Some(ns) = prop_f64_static(props, kbuf.key("normalScale")) {
                    pbr.normal_scale = ns;
                }
                if pbr.normal_scale_track.is_some() || p.animated || p.expression.is_some() {
                    pbr.normal_scale_track = Some(session_property_to_track(
                        p,
                        pbr.normal_scale_track.as_ref(),
                    ));
                }
            }
            if props.contains_key(kbuf.key("alphaMap")) {
                pbr.alpha_map =
                    prop_str_static(props, kbuf.key("alphaMap")).filter(|s| !s.is_empty());
            }
            let has_ao_tex = props.contains_key(kbuf.key("aoTexture"));
            let has_ao_map = props.contains_key(kbuf.key("aoMap"));
            if has_ao_tex && has_ao_map {
                let tex_val =
                    prop_str_static(props, kbuf.key("aoTexture")).filter(|s| !s.is_empty());
                let map_val = prop_str_static(props, kbuf.key("aoMap")).filter(|s| !s.is_empty());
                if tex_val != pbr.ao_texture {
                    pbr.ao_texture = tex_val;
                } else if map_val != pbr.ao_texture {
                    pbr.ao_texture = map_val;
                }
            } else if has_ao_tex {
                pbr.ao_texture =
                    prop_str_static(props, kbuf.key("aoTexture")).filter(|s| !s.is_empty());
            } else if has_ao_map {
                pbr.ao_texture =
                    prop_str_static(props, kbuf.key("aoMap")).filter(|s| !s.is_empty());
            }
            if let Some(p) = props.get(kbuf.key("emissiveIntensity")) {
                pbr.emissive_intensity =
                    session_property_to_track(p, Some(&pbr.emissive_intensity));
            }
            if let Some(at) = prop_f64_static(props, kbuf.key("alphaTest")) {
                pbr.alpha_test = at;
            }
            if let Some(r) = prop_f64_static(props, kbuf.key("reflection")) {
                pbr.reflection = r.round() as i64 != 0;
            }
            if let Some(w) = prop_f64_static(props, kbuf.key("wrap")) {
                pbr.wrap = w.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(p) = props.get(kbuf.key("repeat")) {
                pbr.repeat = session_property_to_track(p, Some(&pbr.repeat));
            }
            if let Some(p) = props.get(kbuf.key("offset")) {
                pbr.offset = session_property_to_track(p, Some(&pbr.offset));
            }
            if let Some(p) = props.get(kbuf.key("center")) {
                pbr.center = session_property_to_track(p, Some(&pbr.center));
            }
            if let Some(p) = props.get(kbuf.key("rotation")) {
                pbr.rotation = session_property_to_track(p, Some(&pbr.rotation));
            }
            if let Some(p) = props.get(kbuf.key("reflectionIntensity")) {
                if let Some(ri) = prop_f64_static(props, kbuf.key("reflectionIntensity")) {
                    pbr.reflection_intensity = ri;
                }
                if pbr.reflection_intensity_track.is_some() || p.animated || p.expression.is_some()
                {
                    pbr.reflection_intensity_track = Some(session_property_to_track(
                        p,
                        pbr.reflection_intensity_track.as_ref(),
                    ));
                }
            }
            // Sync Material+ surface props (AO, Cel, Fresnel, Color Ramp, Bump Noise, UV Projection, Env).
            for &sk in PBR_PLUS_SURFACE_KEYS {
                let prop = props
                    .get(kbuf.key(sk))
                    .or_else(|| props.get(kbuf.key2("surface.", sk)));
                if let Some(prop) = prop {
                    let val = session_property_to_preserved_value(prop, |p| {
                        session_property_to_track(p, None)
                    });
                    if val.is_null() {
                        pbr.preserved.remove(sk);
                    } else {
                        pbr.preserved.insert(sk.to_owned(), val);
                    }
                }
            }
        }
        zc_project::MaterialSpec::Texture(tex) => {
            if has_tex_prop {
                tex.texture = tex_opt;
            }
            if let Some(s) = prop_f64_static(props, kbuf.key("side")) {
                tex.side = s.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(w) = prop_f64_static(props, kbuf.key("wrap")) {
                tex.wrap = w.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(p) = props.get(kbuf.key("repeat")) {
                tex.repeat = session_property_to_track(p, Some(&tex.repeat));
            }
            if let Some(p) = props.get(kbuf.key("offset")) {
                tex.offset = session_property_to_track(p, Some(&tex.offset));
            }
            if let Some(p) = props.get(kbuf.key("center")) {
                tex.center = session_property_to_track(p, Some(&tex.center));
            }
            if let Some(p) = props.get(kbuf.key("rotation")) {
                tex.rotation = session_property_to_track(p, Some(&tex.rotation));
            }
        }
        zc_project::MaterialSpec::Video(vid) => {
            if let Some(p) = props.get(kbuf.key("color")) {
                vid.color = session_property_to_track(p, Some(&vid.color));
            }
            let prev_transparent = vid.transparent;
            if let Some(tr) = prop_f64_static(props, kbuf.key("transparent")) {
                vid.transparent = tr.round() as i64 != 0;
            }
            let transparent_explicitly_changed = vid.transparent != prev_transparent;
            if let Some(p) = props.get(kbuf.key("opacity")) {
                let opacity_changed = !property_track_matches_session(&vid.opacity, p);
                vid.opacity = session_property_to_track(p, Some(&vid.opacity));
                let op = prop_f64_static(props, kbuf.key("opacity")).unwrap_or(1.0);
                if opacity_changed && !transparent_explicitly_changed && (op < 0.999 || p.animated)
                {
                    vid.transparent = true;
                }
            }
            if let Some(b) = prop_f64_static(props, kbuf.key("blending")) {
                vid.blending = b.round().clamp(0.0, 4.0) as u32;
            }
            if let Some(so) = prop_f64_static(props, kbuf.key("startOffset")) {
                vid.start_offset = so.round().max(0.0) as u32;
            }
            if let Some(mo) = prop_f64_static(props, kbuf.key("mediaOffset")) {
                vid.media_offset = mo.round().max(0.0) as u32;
            }
            if has_tex_prop {
                vid.video = tex_opt;
            }
        }
        zc_project::MaterialSpec::Matcap(mc) => {
            if let Some(p) = props.get(kbuf.key("color")) {
                mc.color = session_property_to_track(p, Some(&mc.color));
            }
            if let Some(p) = props.get(kbuf.key("brightness")) {
                mc.brightness = session_property_to_track(p, Some(&mc.brightness));
            }
            let prev_transparent = mc.transparent;
            if let Some(tr) = prop_f64_static(props, kbuf.key("transparent")) {
                mc.transparent = tr.round() as i64 != 0;
            }
            let transparent_explicitly_changed = mc.transparent != prev_transparent;
            if let Some(p) = props.get(kbuf.key("opacity")) {
                let opacity_changed = !property_track_matches_session(&mc.opacity, p);
                mc.opacity = session_property_to_track(p, Some(&mc.opacity));
                let op = prop_f64_static(props, kbuf.key("opacity")).unwrap_or(1.0);
                if opacity_changed && !transparent_explicitly_changed && (op < 0.999 || p.animated)
                {
                    mc.transparent = true;
                }
            }
            if let Some(b) = prop_f64_static(props, kbuf.key("blending")) {
                mc.blending = b.round().clamp(0.0, 4.0) as u32;
            }
            if let Some(s) = prop_f64_static(props, kbuf.key("side")) {
                mc.side = s.round().clamp(0.0, 2.0) as u32;
            }
            if has_tex_prop {
                mc.texture = tex_opt;
            }
        }
        zc_project::MaterialSpec::ImagePlus(img) => {
            if let Some(p) = props.get(kbuf.key("opacity")) {
                img.opacity = session_property_to_track(p, Some(&img.opacity));
            }
            if let Some(tr) = prop_f64_static(props, kbuf.key("transparent")) {
                img.transparent = tr.round() as i64 != 0;
            }
            if let Some(b) = prop_f64_static(props, kbuf.key("blending")) {
                img.blending = b.round().clamp(0.0, 4.0) as u32;
            }
            if let Some(s) = prop_f64_static(props, kbuf.key("side")) {
                img.side = s.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(w) = prop_f64_static(props, kbuf.key("wrap")) {
                img.wrap = w.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(p) = props.get(kbuf.key("repeat")) {
                img.repeat = session_property_to_track(p, Some(&img.repeat));
            }
            if let Some(p) = props.get(kbuf.key("offset")) {
                img.offset = session_property_to_track(p, Some(&img.offset));
            }
            if let Some(p) = props.get(kbuf.key("center")) {
                img.center = session_property_to_track(p, Some(&img.center));
            }
            if let Some(p) = props.get(kbuf.key("rotation")) {
                img.rotation = session_property_to_track(p, Some(&img.rotation));
            }
            if has_tex_prop {
                img.texture = tex_opt;
            }
        }
        zc_project::MaterialSpec::CustomShader {
            frag_shader,
            custom_properties,
        } => {
            if let Some(code) = prop_str_static(props, kbuf.key("fragShader")) {
                *frag_shader = code;
            }
            let mut handled = BTreeSet::new();
            for cp in custom_properties.iter_mut() {
                if let Some(prop) = props.get(kbuf.key2("custom.", &cp.name)) {
                    cp.track = session_property_to_track(prop, Some(&cp.track));
                    handled.insert(cp.name.clone());
                }
            }
            let custom_prefix = kbuf.key("custom.");
            for (pk, prop) in props {
                if let Some(name) = pk.strip_prefix(custom_prefix)
                    && !name.is_empty()
                    && !handled.contains(name)
                {
                    custom_properties.push(zc_project::CustomProperty {
                        name: name.to_owned(),
                        prop_type: None,
                        track: session_property_to_track(prop, None),
                        metadata: BTreeMap::new(),
                    });
                }
            }
        }
        zc_project::MaterialSpec::NodeMaterial {
            schema_version,
            resolution,
            graph,
        } => {
            if let Some(sv) = prop_f64_static(props, kbuf.key("schemaVersion")) {
                *schema_version = sv.round().max(1.0) as u32;
            }
            if let Some(br) = prop_f64_static(props, kbuf.key("bakeResolution")) {
                let raw = br.round().max(0.0) as u32;
                *resolution = match raw {
                    0 => 128,
                    1 => 256,
                    2 => 512,
                    3 => 1024,
                    4 => 2048,
                    other if other >= 64 => other.min(4096),
                    _ => 256,
                };
            }
            if let Some(gp) = props
                .get(kbuf.key("graph"))
                .and_then(|p| p.keyframes.first())
            {
                if let Some(s) = gp.value.as_str() {
                    if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                        *graph = parsed;
                    }
                } else {
                    *graph = gp.value.clone();
                }
            }
        }
        zc_project::MaterialSpec::Unknown { .. } => {}
    }
}

fn parse_gradient_stops_from_session(
    props: &BTreeMap<String, SessionProperty>,
    key: &str,
) -> Option<Vec<zc_project::GradientStop>> {
    let raw_val = &props.get(key)?.keyframes.first()?.value;
    let parsed: Value = if let Some(s) = raw_val.as_str() {
        serde_json::from_str::<Value>(s).ok()?
    } else {
        raw_val.clone()
    };
    let arr = parsed.as_array()?;
    let mut stops = Vec::with_capacity(arr.len());
    for item in arr {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let Some(position) = obj
            .get("position")
            .or_else(|| obj.get("t"))
            .or_else(|| obj.get("pos"))
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
        else {
            continue;
        };
        let color_str = if let Some(s) = obj.get("color").and_then(Value::as_str) {
            Some(s.to_owned())
        } else if let Some(c) = obj.get("color").and_then(Value::as_array)
            && c.len() >= 3
            && let (Some(r), Some(g), Some(b)) = (
                c[0].as_f64().filter(|v| v.is_finite()),
                c[1].as_f64().filter(|v| v.is_finite()),
                c[2].as_f64().filter(|v| v.is_finite()),
            )
        {
            let to_u8 = |ch: f64| -> u8 {
                if ch > 1.0 {
                    ch.round().clamp(0.0, 255.0) as u8
                } else {
                    (ch * 255.0).round().clamp(0.0, 255.0) as u8
                }
            };
            let a = c
                .get(3)
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite())
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);
            Some(format!("rgba({},{},{},{a})", to_u8(r), to_u8(g), to_u8(b)))
        } else if let Some(v) = obj.get("size").or_else(|| obj.get("value")) {
            if let Some(n) = v.as_f64().filter(|n| n.is_finite()) {
                Some(format!("{}", n.max(0.0)))
            } else {
                v.as_str().map(ToOwned::to_owned)
            }
        } else {
            None
        };
        if let Some(color) = color_str {
            stops.push(zc_project::GradientStop { position, color });
        }
    }
    (!stops.is_empty()).then_some(stops)
}

pub(crate) fn unique_object_3d_name(existing: &[SessionObject3D], base: &str) -> String {
    if !existing.iter().any(|o| o.name == base) {
        return base.to_owned();
    }
    let mut suffix = 2usize;
    loop {
        let candidate = format!("{base} {suffix}");
        if !existing.iter().any(|o| o.name == candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

fn clamp_track_f64_range(track: &mut zc_project::PropertyTrack, min: f64, max: f64) {
    if let zc_project::PropertyValue::Number(ref mut n) = track.value {
        *n = n.clamp(min, max);
    }
    for kf in &mut track.keyframes {
        if let zc_project::PropertyValue::Number(ref mut n) = kf.value {
            *n = n.clamp(min, max);
        }
    }
}

fn build_object_3d_from_session(
    sobj: &SessionObject3D,
    existing: Option<zc_project::Object3D>,
    children: Vec<zc_project::Object3D>,
) -> zc_project::Object3D {
    let (canonical_kind, poly_shape_alias) = canonicalize_object_3d_kind(&sobj.kind);
    let was_existing = existing.is_some();
    let mut obj = existing.unwrap_or_else(|| {
        let mut default_material: Option<zc_project::MaterialSpec> = None;
        let mut plugin_tag: Option<String> = None;
        let kind = match canonical_kind.as_str() {
            k if is_3d_container_kind(k) => {
                if k != "group" {
                    plugin_tag = Some(k.to_owned());
                }
                zc_project::Object3DKind::Group {
                    reflection_visibility: 0,
                    custom_properties: Vec::new(),
                    children: Vec::new(),
                }
            }
            "text" | "text_3d" => {
                default_material = Some(zc_project::MaterialSpec::single_color_default());
                zc_project::Object3DKind::Text3D(Box::default())
            }
            k if plugin_light_kind(k).is_some() => {
                plugin_tag = Some(k.to_owned());
                let lk = plugin_light_kind(k).unwrap_or(zc_project::LightKind::Spot);
                zc_project::Object3DKind::Light(Box::new(zc_project::LightData::for_kind(lk)))
            }
            "light" | "spot_light" => {
                zc_project::Object3DKind::Light(Box::new(zc_project::LightData::spot()))
            }
            "point_light" => zc_project::Object3DKind::Light(Box::new(
                zc_project::LightData::for_kind(zc_project::LightKind::Point),
            )),
            "directional_light" => zc_project::Object3DKind::Light(Box::new(
                zc_project::LightData::for_kind(zc_project::LightKind::Directional),
            )),
            "hemisphere_light" => zc_project::Object3DKind::Light(Box::new(
                zc_project::LightData::for_kind(zc_project::LightKind::Hemisphere),
            )),
            "ambient_light" => zc_project::Object3DKind::Light(Box::new(
                zc_project::LightData::for_kind(zc_project::LightKind::Ambient),
            )),
            k if is_3d_camera_plus_kind(k) => {
                plugin_tag = Some("zoidium:camera-plus/camera".to_owned());
                zc_project::Object3DKind::Camera(Box::default())
            }
            "camera" => zc_project::Object3DKind::Camera(Box::default()),
            "particles" => zc_project::Object3DKind::Particles(Box::new(default_particles_data())),
            "model" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Model(Box::default())
            }
            "sphere" | "shape.sphere" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Sphere {
                    radius: 5.0,
                    width_segments: 24,
                    height_segments: 16,
                })
            }
            "cylinder" | "shape.cylinder" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Cylinder {
                    top_radius: 5.0,
                    bottom_radius: 5.0,
                    height: 10.0,
                    detail: 24,
                    open_ended: false,
                })
            }
            "rectangle" | "shape.rectangle" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Rectangle {
                    size: [10.0, 10.0],
                })
            }
            "circle" | "shape.circle" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Circle {
                    radius: 5.0,
                    detail: 32,
                })
            }
            "donut" | "shape.donut" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Donut {
                    radius: 5.0,
                    tube: 2.0,
                    radial_segments: 16,
                    tubular_segments: 32,
                })
            }
            "wire" | "shape.wire" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Wire {
                    size: [10.0, 10.0, 10.0],
                    thickness: 0.6,
                    loops: 3,
                    inner_segments: 12,
                    outer_segments: 24,
                })
            }
            "custom" | "shape.custom" => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::CustomGeometry {
                    asset_id: None,
                })
            }
            "zoidium:geometry-plus/rounded-box" => {
                plugin_tag = Some("zoidium:geometry-plus/rounded-box".to_owned());
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::RoundedBox {
                    size: [40.0, 24.0, 6.0],
                    radius: 3.0,
                    segments: 6,
                    corner_radius: Some(3.0),
                    corner_segments: Some(6),
                    bevel_size: Some(0.6),
                    bevel_segments: Some(2),
                    shading: 0,
                })
            }
            "zoidium:geometry-plus/polyhedron" => {
                plugin_tag = Some("zoidium:geometry-plus/polyhedron".to_owned());
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                let shape_idx = poly_shape_alias.unwrap_or_else(|| {
                    prop_f64_static(&sobj.properties, "polyhedron.shape")
                        .map_or(0, |v| v.round().clamp(0.0, 3.0) as u32)
                });
                let shape = match shape_idx {
                    1 => zc_project::Shape3DKind::Octahedron {
                        radius: 12.0,
                        detail: 0,
                        shading: 0,
                    },
                    2 => zc_project::Shape3DKind::Icosahedron {
                        radius: 12.0,
                        detail: 0,
                        shading: 0,
                    },
                    3 => zc_project::Shape3DKind::Dodecahedron {
                        radius: 12.0,
                        detail: 0,
                        shading: 0,
                    },
                    _ => zc_project::Shape3DKind::Tetrahedron {
                        radius: 12.0,
                        detail: 0,
                        shading: 0,
                    },
                };
                zc_project::Object3DKind::Shape3D(shape)
            }
            "zoidium:geometry-plus/cone" => {
                plugin_tag = Some("zoidium:geometry-plus/cone".to_owned());
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Cone {
                    radius: 16.0,
                    height: 32.0,
                    detail: 24,
                    open_ended: false,
                    top_radius: 0.0,
                    shading: 0,
                })
            }
            "zoidium:geometry-plus/capsule" => {
                plugin_tag = Some("zoidium:geometry-plus/capsule".to_owned());
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Capsule {
                    radius: 8.0,
                    length: 16.0,
                    cap_segments: 8,
                    radial_segments: 16,
                    shading: 0,
                })
            }
            "zoidium:geometry-plus/torus-knot" => {
                plugin_tag = Some("zoidium:geometry-plus/torus-knot".to_owned());
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::TorusKnot {
                    radius: 5.0,
                    tube: 1.5,
                    tubular_segments: 64,
                    radial_segments: 12,
                    p: 2,
                    q: 3,
                })
            }
            "zoidium:geometry-plus/tube" => {
                plugin_tag = Some("zoidium:geometry-plus/tube".to_owned());
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Tube {
                    outer_radius: 12.0,
                    inner_radius: 8.0,
                    height: 24.0,
                    radial_segments: 24,
                    open_ended: false,
                    shading: 0,
                })
            }
            "zoidium:geometry-plus/gear" => {
                plugin_tag = Some("zoidium:geometry-plus/gear".to_owned());
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Gear {
                    teeth: 12,
                    inner_radius: 10.0,
                    outer_radius: 16.0,
                    thickness: 6.0,
                    tooth_width: 0.55,
                    bevel_size: 0.3,
                    bevel_segments: 2,
                    shading: 0,
                })
            }
            "zoidium:geometry-plus/helix" => {
                plugin_tag = Some("zoidium:geometry-plus/helix".to_owned());
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Helix {
                    radius: 14.0,
                    height: 32.0,
                    turns: 3.0,
                    tube_radius: 2.0,
                    path_segments: 64,
                    radial_segments: 8,
                    shading: 0,
                })
            }
            "zoidium:geometry-plus/spline" => {
                plugin_tag = Some("zoidium:geometry-plus/spline".to_owned());
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Spline {
                    points: vec![[-20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [20.0, 0.0, 0.0]],
                    interpolation: "Cubic".to_owned(),
                    closed: false,
                    start: 0.0,
                    end: 1.0,
                    thickness: 2.0,
                    thickness_map: zc_project::GradientStop::white_default(),
                    gradient_space: "Curve".to_owned(),
                    tension: 1.0,
                    path_segments: 16,
                    radial_segments: 8,
                    caps: "Both".to_owned(),
                    cap_style: "Flat".to_owned(),
                    shading: 0,
                })
            }
            k if is_3d_optical_flare_kind(k)
                || is_3d_trapcode_kind(k)
                || k.starts_with("zoidium:")
                || k.starts_with("zoidcraft:") =>
            {
                plugin_tag = Some(k.to_owned());
                zc_project::Object3DKind::Dummy {
                    original_type: Some(k.to_owned()),
                    payload: json!({}),
                }
            }
            _ => {
                default_material = Some(zc_project::MaterialSpec::CustomPbr(Box::default()));
                zc_project::Object3DKind::Shape3D(zc_project::Shape3DKind::Box {
                    size: [10.0, 10.0, 10.0],
                })
            }
        };
        let mut created = zc_project::Object3D::new(&sobj.name, kind);
        created.id = parse_uuid_or_new(&sobj.id);
        created.material = default_material;
        if let Some(tag) = plugin_tag {
            created
                .preserved
                .insert("zoidiumPluginObjectType".to_string(), json!(tag));
        }
        created
    });
    if is_3d_repeater_kind(&canonical_kind)
        || is_3d_text_plus_kind(&canonical_kind)
        || is_3d_expresstext_plus_kind(&canonical_kind)
        || is_3d_optical_flare_kind(&canonical_kind)
        || is_3d_trapcode_kind(&canonical_kind)
        || is_3d_light_plus_kind(&canonical_kind)
    {
        obj.preserved
            .insert("zoidiumPluginObjectType".to_string(), json!(canonical_kind));
        obj.preserved.remove("type");
    } else if is_3d_camera_plus_kind(&canonical_kind) {
        obj.preserved.insert(
            "zoidiumPluginObjectType".to_string(),
            json!("zoidium:camera-plus/camera"),
        );
    }
    obj.name = sobj.name.clone();
    if let Some(p) = sobj.properties.get("enabled") {
        obj.enabled = session_property_to_track(p, Some(&obj.enabled));
    }
    if let Some(p) = sobj.properties.get("position") {
        obj.position = session_property_to_track(p, Some(&obj.position));
    }
    if let Some(p) = sobj.properties.get("rotation") {
        obj.rotation = session_property_to_track(p, Some(&obj.rotation));
    }
    if let Some(p) = sobj.properties.get("scale") {
        obj.scale = session_property_to_track(p, Some(&obj.scale));
    }
    if let Some(v) = sobj
        .properties
        .get("eulerOrder")
        .and_then(|p| p.keyframes.first())
        .map(|k| &k.value)
        && let Some(order) = euler_order_from_value(v)
    {
        obj.euler_order = order;
    }

    match &mut obj.kind {
        zc_project::Object3DKind::Text3D(t) => {
            if let Some(s) = prop_str_static(&sobj.properties, "text") {
                t.text = s;
            }
            if sobj.properties.contains_key("font") {
                t.font = prop_str_static(&sobj.properties, "font").filter(|s| !s.is_empty());
            }
            if let Some(h) = prop_f64_static(&sobj.properties, "height")
                && (h - t.size[0]).abs() > 1e-12
            {
                t.size[0] = h.clamp(0.05, 5000.0);
            }
            if let Some(th) = prop_f64_static(&sobj.properties, "thickness")
                && (th - t.size[1]).abs() > 1e-12
            {
                t.size[1] = th.clamp(0.0, 1000.0);
            }
            if let Some(d) = prop_f64_static(&sobj.properties, "detail") {
                let raw = d.round().max(0.0) as u32;
                if raw != t.detail {
                    t.detail = d.round().clamp(1.0, 200.0) as u32;
                }
            }
            if let Some(b) = prop_f64_static(&sobj.properties, "bevelEnabled") {
                t.bevel = b.round() as i64 != 0;
            }
            if let Some(bt) = prop_f64_static(&sobj.properties, "bevelThickness")
                && (bt - t.bevel_thickness).abs() > 1e-12
            {
                t.bevel_thickness = bt.clamp(0.0, 100.0);
            }
            if let Some(bs) = prop_f64_static(&sobj.properties, "bevelSize")
                && (bs - t.bevel_size).abs() > 1e-12
            {
                t.bevel_size = bs.clamp(0.0, 100.0);
            }
            if let Some(pm) = prop_f64_static(&sobj.properties, "positionMode") {
                t.position_mode = zc_project::Text3DAlign::from_cm3_id(pm.round().max(0.0) as u32);
            }
            if let Some(bt) = prop_f64_static(&sobj.properties, "bevelTension")
                && (bt - t.bevel_tension).abs() > 1e-12
            {
                t.bevel_tension = bt.clamp(0.0, 1.0);
            }
            if let Some(sp) = prop_f64_static(&sobj.properties, "spacing")
                && (sp - t.spacing).abs() > 1e-12
            {
                t.spacing = sp.clamp(-200.0, 200.0);
            }
            if let Some(bs) = prop_f64_static(&sobj.properties, "bevelSide") {
                t.bevel_side = bs.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(bd) = prop_f64_static(&sobj.properties, "bevelDetail") {
                let raw = bd.round().max(0.0) as u32;
                if raw != t.bevel_detail {
                    t.bevel_detail = bd.round().clamp(1.0, 100.0) as u32;
                }
            }
            if let Some(bp) = prop_f64_static(&sobj.properties, "bevelProfile") {
                t.bevel_profile = bp.round().clamp(0.0, 3.0) as u32;
            }
            if let Some(p) = sobj.properties.get("characterIndexOffset") {
                t.character_index_offset =
                    session_property_to_track(p, Some(&t.character_index_offset));
            }
            if let Some(p) = sobj.properties.get("characterCountOverride") {
                t.character_count_override =
                    session_property_to_track(p, Some(&t.character_count_override));
            }
            if let Some(tsm_str) = prop_str_static(&sobj.properties, "textScaleMode") {
                t.text_scale_mode = zc_project::CompatMode::from_token(&tsm_str);
            } else if let Some(tsm) = prop_f64_static(&sobj.properties, "textScaleMode") {
                t.text_scale_mode = if tsm.round() as i64 != 0 {
                    zc_project::CompatMode::Legacy
                } else {
                    zc_project::CompatMode::Current
                };
            }
            if let Some(kern) = prop_f64_static(&sobj.properties, "kerning") {
                let new_kern = u64::from(kern.round() as i64 != 0);
                let prev_kern = obj
                    .preserved
                    .get("kerning")
                    .or_else(|| obj.preserved.get("property.kerning"))
                    .and_then(parse_preserved_bool_u64);
                let unchanged_default = was_existing && prev_kern.is_none() && new_kern == 1;
                if prev_kern != Some(new_kern) && !unchanged_default {
                    obj.preserved.remove("property.kerning");
                    obj.preserved.insert("kerning".to_owned(), json!(new_kern));
                }
            }
        }
        zc_project::Object3DKind::Shape3D(s) => match s {
            zc_project::Shape3DKind::Box { size } => {
                if let Some(v) = prop_vec3_static(&sobj.properties, "box_size")
                    && v != *size
                {
                    *size = v;
                }
            }
            zc_project::Shape3DKind::Cylinder {
                top_radius,
                bottom_radius,
                height,
                detail,
                open_ended,
            } => {
                if let Some(v) = prop_vec3_static(&sobj.properties, "cylinder_size") {
                    *top_radius = v[0];
                    *bottom_radius = v[1];
                    *height = v[2];
                }
                if let Some(d) = prop_f64_static(&sobj.properties, "cylinder_detail") {
                    let raw = d.round().max(0.0) as u32;
                    if raw != *detail {
                        *detail = raw.max(1);
                    }
                }
                // CM3 polarity: 0 => open_ended = true, non-zero => closed (open_ended = false).
                if let Some(oe) = prop_f64_static(&sobj.properties, "cylinder_openEnded") {
                    *open_ended = oe.round() as i64 == 0;
                }
            }
            zc_project::Shape3DKind::Rectangle { size } => {
                if let Some(v) = prop_vec2_static(&sobj.properties, "rect_size")
                    && v != *size
                {
                    *size = v;
                }
            }
            zc_project::Shape3DKind::Circle { radius, detail } => {
                if let Some(r) = prop_f64_static(&sobj.properties, "circle_size") {
                    *radius = r;
                }
                if let Some(d) = prop_f64_static(&sobj.properties, "circle_detail") {
                    let raw = d.round().max(0.0) as u32;
                    if raw != *detail {
                        *detail = raw.max(1);
                    }
                }
            }
            zc_project::Shape3DKind::Sphere {
                radius,
                width_segments,
                height_segments,
            } => {
                if let Some(r) = prop_f64_static(&sobj.properties, "sphere_size") {
                    *radius = r;
                }
                if let Some(det) = prop_vec2_static(&sobj.properties, "sphere_detail") {
                    *width_segments = det[0].round().max(1.0) as u32;
                    *height_segments = det[1].round().max(1.0) as u32;
                }
            }
            zc_project::Shape3DKind::Donut {
                radius,
                tube,
                radial_segments,
                tubular_segments,
            } => {
                if let Some(sz) = prop_vec2_static(&sobj.properties, "donut_size") {
                    *radius = sz[0];
                    *tube = sz[1];
                }
                if let Some(det) = prop_vec2_static(&sobj.properties, "donut_detail") {
                    *radial_segments = det[0].round().max(1.0) as u32;
                    *tubular_segments = det[1].round().max(1.0) as u32;
                }
            }
            zc_project::Shape3DKind::Wire {
                size,
                thickness,
                loops,
                inner_segments,
                outer_segments,
            } => {
                if let Some(v) = prop_vec3_static(&sobj.properties, "path_size") {
                    *size = v;
                }
                if let Some(t) = prop_f64_static(&sobj.properties, "path_thickness") {
                    *thickness = t;
                }
                if let Some(l) = prop_f64_static(&sobj.properties, "path_loops") {
                    *loops = l.round().max(1.0) as u32;
                }
                if let Some(det) = prop_vec2_static(&sobj.properties, "path_detail") {
                    *inner_segments = det[0].round().max(1.0) as u32;
                    *outer_segments = det[1].round().max(1.0) as u32;
                } else if let Some(d) = prop_f64_static(&sobj.properties, "path_detail") {
                    *outer_segments = d.round().max(1.0) as u32;
                }
                if let Some(d) = prop_f64_static(&sobj.properties, "path_innerDetail") {
                    *inner_segments = d.round().max(1.0) as u32;
                }
                if let Some(raw_closed) = sobj
                    .properties
                    .get("path_closed")
                    .and_then(|p| p.keyframes.first())
                    .map(|kf| &kf.value)
                {
                    let is_closed = raw_closed
                        .as_bool()
                        .or_else(|| raw_closed.as_f64().map(|n| n.round() as i64 != 0))
                        .unwrap_or(false);
                    if is_closed || obj.preserved.contains_key("property.path_closed") {
                        obj.preserved.insert(
                            "property.path_closed".to_owned(),
                            json!(u32::from(is_closed)),
                        );
                    }
                }
            }
            zc_project::Shape3DKind::CustomGeometry { asset_id } => {
                if sobj.properties.contains_key("customGeometry") {
                    *asset_id = prop_str_static(&sobj.properties, "customGeometry")
                        .filter(|s| !s.is_empty());
                }
            }
            zc_project::Shape3DKind::Cone {
                radius,
                height,
                detail,
                open_ended,
                top_radius,
                shading,
            } => {
                if let Some(r) = prop_f64_static(&sobj.properties, "cone.radius") {
                    *radius = r;
                }
                if let Some(tr) = prop_f64_static(&sobj.properties, "cone.topRadius") {
                    *top_radius = tr.max(0.0);
                }
                if let Some(h) = prop_f64_static(&sobj.properties, "cone.height") {
                    *height = h;
                }
                if let Some(d) = prop_f64_static(&sobj.properties, "cone.detail") {
                    *detail = d.round().max(1.0) as u32;
                }
                if let Some(oe) = prop_f64_static(&sobj.properties, "cone.openEnded") {
                    *open_ended = oe.round() as i64 != 0;
                }
                if let Some(sh) = prop_f64_static(&sobj.properties, "cone.shading") {
                    *shading = sh.round().clamp(0.0, 1.0) as u32;
                }
            }
            zc_project::Shape3DKind::RoundedBox {
                size,
                radius,
                segments,
                corner_radius,
                corner_segments,
                bevel_size,
                bevel_segments,
                shading,
            } => {
                if let Some(v) = prop_vec3_static(&sobj.properties, "box_size") {
                    *size = v;
                }
                if let Some(r) = prop_f64_static(&sobj.properties, "roundedBox.radius") {
                    *radius = r;
                }
                if let Some(seg) = prop_f64_static(&sobj.properties, "roundedBox.segments") {
                    *segments = seg.round().max(1.0) as u32;
                }
                if let Some(cr) = prop_f64_static(&sobj.properties, "roundedBox.cornerRadius")
                    .or_else(|| prop_f64_static(&sobj.properties, "cornerRadius"))
                {
                    *corner_radius = Some(cr.max(0.0));
                }
                if let Some(cs) = prop_f64_static(&sobj.properties, "roundedBox.cornerSegments")
                    .or_else(|| prop_f64_static(&sobj.properties, "cornerSegments"))
                {
                    *corner_segments = Some(cs.round().max(1.0) as u32);
                }
                if let Some(bs) = prop_f64_static(&sobj.properties, "roundedBox.bevelSize")
                    .or_else(|| prop_f64_static(&sobj.properties, "bevelSize"))
                {
                    *bevel_size = Some(bs.max(0.0));
                }
                if let Some(bseg) = prop_f64_static(&sobj.properties, "roundedBox.bevelSegments")
                    .or_else(|| prop_f64_static(&sobj.properties, "bevelSegments"))
                {
                    *bevel_segments = Some(bseg.round().max(0.0) as u32);
                }
                if let Some(sh) = prop_f64_static(&sobj.properties, "roundedBox.shading")
                    .or_else(|| prop_f64_static(&sobj.properties, "shading"))
                {
                    *shading = sh.round().clamp(0.0, 1.0) as u32;
                }
            }
            zc_project::Shape3DKind::Tetrahedron {
                radius,
                detail,
                shading,
            }
            | zc_project::Shape3DKind::Octahedron {
                radius,
                detail,
                shading,
            }
            | zc_project::Shape3DKind::Dodecahedron {
                radius,
                detail,
                shading,
            }
            | zc_project::Shape3DKind::Icosahedron {
                radius,
                detail,
                shading,
            } => {
                if let Some(r) = prop_f64_static(&sobj.properties, "polyhedron.radius") {
                    *radius = r;
                }
                if let Some(d) = prop_f64_static(&sobj.properties, "polyhedron.detail") {
                    *detail = d.round().max(0.0) as u32;
                }
                if let Some(sh) = prop_f64_static(&sobj.properties, "polyhedron.shading") {
                    *shading = sh.round().clamp(0.0, 1.0) as u32;
                }
                if let Some(shape_f) = prop_f64_static(&sobj.properties, "polyhedron.shape") {
                    let cur_r = *radius;
                    let cur_d = *detail;
                    let cur_s = *shading;
                    *s = match shape_f.round() as i64 {
                        1 => zc_project::Shape3DKind::Octahedron {
                            radius: cur_r,
                            detail: cur_d,
                            shading: cur_s,
                        },
                        2 => zc_project::Shape3DKind::Icosahedron {
                            radius: cur_r,
                            detail: cur_d,
                            shading: cur_s,
                        },
                        3 => zc_project::Shape3DKind::Dodecahedron {
                            radius: cur_r,
                            detail: cur_d,
                            shading: cur_s,
                        },
                        _ => zc_project::Shape3DKind::Tetrahedron {
                            radius: cur_r,
                            detail: cur_d,
                            shading: cur_s,
                        },
                    };
                }
            }
            zc_project::Shape3DKind::Capsule {
                radius,
                length,
                cap_segments,
                radial_segments,
                shading,
            } => {
                if let Some(r) = prop_f64_static(&sobj.properties, "capsule.radius") {
                    *radius = r;
                }
                if let Some(l) = prop_f64_static(&sobj.properties, "capsule.length") {
                    *length = l;
                }
                if let Some(cs) = prop_f64_static(&sobj.properties, "capsule.capSegments") {
                    *cap_segments = cs.round().max(1.0) as u32;
                }
                if let Some(rs) = prop_f64_static(&sobj.properties, "capsule.radialSegments") {
                    *radial_segments = rs.round().max(1.0) as u32;
                }
                if let Some(sh) = prop_f64_static(&sobj.properties, "capsule.shading") {
                    *shading = sh.round().clamp(0.0, 1.0) as u32;
                }
            }
            zc_project::Shape3DKind::TorusKnot {
                radius,
                tube,
                tubular_segments,
                radial_segments,
                p,
                q,
            } => {
                if let Some(r) = prop_f64_static(&sobj.properties, "torusKnot.radius") {
                    *radius = r;
                }
                if let Some(t) = prop_f64_static(&sobj.properties, "torusKnot.tube") {
                    *tube = t;
                }
                if let Some(ts) = prop_f64_static(&sobj.properties, "torusKnot.tubularSegments") {
                    *tubular_segments = ts.round().max(1.0) as u32;
                }
                if let Some(rs) = prop_f64_static(&sobj.properties, "torusKnot.radialSegments") {
                    *radial_segments = rs.round().max(1.0) as u32;
                }
                if let Some(pv) = prop_f64_static(&sobj.properties, "torusKnot.p") {
                    *p = pv.round().max(1.0) as u32;
                }
                if let Some(qv) = prop_f64_static(&sobj.properties, "torusKnot.q") {
                    *q = qv.round().max(1.0) as u32;
                }
            }
            zc_project::Shape3DKind::Tube {
                outer_radius,
                inner_radius,
                height,
                radial_segments,
                open_ended,
                shading,
            } => {
                if let Some(v) = prop_f64_static(&sobj.properties, "tube.outerRadius") {
                    *outer_radius = v.max(0.01);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "tube.innerRadius") {
                    *inner_radius = v.max(0.0);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "tube.height") {
                    *height = v.max(0.01);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "tube.radialSegments") {
                    *radial_segments = v.round().max(3.0) as u32;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "tube.openEnded") {
                    *open_ended = v.round() as i64 != 0;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "tube.shading") {
                    *shading = v.round().clamp(0.0, 1.0) as u32;
                }
            }
            zc_project::Shape3DKind::Gear {
                teeth,
                inner_radius,
                outer_radius,
                thickness,
                tooth_width,
                bevel_size,
                bevel_segments,
                shading,
            } => {
                if let Some(v) = prop_f64_static(&sobj.properties, "gear.teeth") {
                    *teeth = v.round().clamp(3.0, 64.0) as u32;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "gear.innerRadius") {
                    *inner_radius = v.max(0.01);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "gear.outerRadius") {
                    *outer_radius = v.max(0.01);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "gear.thickness") {
                    *thickness = v.max(0.01);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "gear.toothWidth") {
                    *tooth_width = v.clamp(0.05, 0.95);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "gear.bevelSize") {
                    *bevel_size = v.max(0.0);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "gear.bevelSegments") {
                    *bevel_segments = v.round().max(0.0) as u32;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "gear.shading") {
                    *shading = v.round().clamp(0.0, 1.0) as u32;
                }
            }
            zc_project::Shape3DKind::Helix {
                radius,
                height,
                turns,
                tube_radius,
                path_segments,
                radial_segments,
                shading,
            } => {
                if let Some(v) = prop_f64_static(&sobj.properties, "helix.radius") {
                    *radius = v.max(0.01);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "helix.height") {
                    *height = v.max(0.01);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "helix.turns") {
                    *turns = v.max(0.05);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "helix.tubeRadius") {
                    *tube_radius = v.max(0.01);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "helix.pathSegments") {
                    *path_segments = v.round().max(8.0) as u32;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "helix.radialSegments") {
                    *radial_segments = v.round().max(3.0) as u32;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "helix.shading") {
                    *shading = v.round().clamp(0.0, 1.0) as u32;
                }
            }
            zc_project::Shape3DKind::Spline {
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
                if let Some(vp) = sobj
                    .properties
                    .get("spline.points")
                    .and_then(|p| p.keyframes.first())
                {
                    let parsed: Option<Vec<[f64; 3]>> = if let Some(s) = vp.value.as_str() {
                        serde_json::from_str(s).ok()
                    } else {
                        serde_json::from_value(vp.value.clone()).ok()
                    };
                    if let Some(pts) = parsed
                        && pts.len() >= 2
                    {
                        *points = pts;
                    }
                }
                if let Some(s) = prop_str_static(&sobj.properties, "spline.interpolation") {
                    *interpolation = s;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "spline.closed") {
                    *closed = v.round() as i64 != 0;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "spline.start") {
                    *start = v.clamp(0.0, 1.0);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "spline.end") {
                    *end = v.clamp(0.0, 1.0);
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "spline.thickness") {
                    *thickness = v.max(0.01);
                }
                if let Some(stops) =
                    parse_gradient_stops_from_session(&sobj.properties, "spline.thicknessMap")
                    && !stops.is_empty()
                {
                    *thickness_map = stops;
                }
                if let Some(s) = prop_str_static(&sobj.properties, "spline.gradientSpace") {
                    *gradient_space = s;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "spline.tension") {
                    *tension = v;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "spline.pathSegments") {
                    *path_segments = v.round().max(1.0) as u32;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "spline.radialSegments") {
                    *radial_segments = v.round().max(3.0) as u32;
                }
                if let Some(s) = prop_str_static(&sobj.properties, "spline.caps") {
                    *caps = s;
                }
                if let Some(s) = prop_str_static(&sobj.properties, "spline.capStyle") {
                    *cap_style = s;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "spline.shading") {
                    *shading = v.round().clamp(0.0, 1.0) as u32;
                }
            }
        },
        zc_project::Object3DKind::Model(m) => {
            if sobj.properties.contains_key("customGeometry") {
                m.asset_id =
                    prop_str_static(&sobj.properties, "customGeometry").filter(|s| !s.is_empty());
            }
        }
        zc_project::Object3DKind::Light(l) => {
            if let Some(lk) = prop_f64_static(&sobj.properties, "light.kind") {
                l.kind = match lk.round() as i64 {
                    1 => zc_project::LightKind::Point,
                    2 => zc_project::LightKind::Directional,
                    3 => zc_project::LightKind::Hemisphere,
                    4 => zc_project::LightKind::Ambient,
                    _ => zc_project::LightKind::Spot,
                };
            }
            if let Some(p) = sobj.properties.get("color") {
                l.color = session_property_to_track(p, Some(&l.color));
            }
            if let Some(p) = sobj.properties.get("groundColor") {
                l.ground_color = session_property_to_track(p, Some(&l.ground_color));
            }
            if let Some(p) = sobj.properties.get("target") {
                l.target = session_property_to_track(p, Some(&l.target));
            }
            if let Some(p) = sobj.properties.get("intensity") {
                l.intensity = session_property_to_track(p, Some(&l.intensity));
            }
            if let Some(p) = sobj.properties.get("angle") {
                l.angle = session_property_to_track(p, Some(&l.angle));
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "penumbra") {
                l.penumbra = v.clamp(0.0, 1.0);
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "decay") {
                l.decay = v.max(0.0);
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "distance") {
                l.distance = v.max(0.0);
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "castShadow") {
                l.cast_shadow = v.round() as i64 != 0;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "shadowMapSize") {
                let raw = v.round().max(0.0) as u32;
                if raw != l.shadow_map_size {
                    l.shadow_map_size = raw.clamp(64, 4096);
                }
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "shadowBias")
                && (v - l.shadow_bias).abs() > 1e-12
            {
                l.shadow_bias = v.clamp(-1.0, 1.0);
            }
            if sobj.properties.contains_key("iesProfile") {
                if let Some(profile) =
                    prop_str_static(&sobj.properties, "iesProfile").filter(|s| !s.is_empty())
                {
                    obj.preserved
                        .insert("property.iesProfile".to_owned(), Value::String(profile));
                } else {
                    obj.preserved.remove("property.iesProfile");
                }
            }
        }
        zc_project::Object3DKind::Camera(c) => {
            if let Some(ck) = prop_f64_static(&sobj.properties, "camera.kind") {
                c.kind = if ck.round() as i64 == 1 {
                    zc_project::CameraKind::Orthographic
                } else {
                    zc_project::CameraKind::Perspective
                };
            }
            if let Some(p) = sobj.properties.get("fov") {
                c.fov = session_property_to_track(p, Some(&c.fov));
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "near") {
                c.near = v.max(1e-4);
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "far") {
                c.far = v.max(c.near + 0.1);
            }
            if let Some(p) = sobj.properties.get("zoom") {
                c.zoom = session_property_to_track(p, Some(&c.zoom));
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "active") {
                c.active = v.round() as i64 != 0;
            }
            if let Some(p) = sobj.properties.get("focalLength") {
                c.focal_length = session_property_to_track(p, Some(&c.focal_length));
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "filmGate")
                && (v - c.film_gate).abs() > 1e-12
            {
                c.film_gate = v.clamp(1.0, 500.0);
            }
            if let Some(p) = sobj.properties.get("filmOffsetX") {
                c.film_offset_x = session_property_to_track(p, Some(&c.film_offset_x));
            }
            if let Some(p) = sobj.properties.get("filmOffsetY") {
                c.film_offset_y = session_property_to_track(p, Some(&c.film_offset_y));
            }
            if sobj.properties.keys().any(|k| k.starts_with("motionBlur.")) {
                let mb = c.motion_blur.get_or_insert_with(Default::default);
                if let Some(v) = prop_f64_static(&sobj.properties, "motionBlur.enabled") {
                    mb.enabled = v.round() as i64 != 0;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "motionBlur.samples") {
                    mb.samples = v.round().clamp(1.0, 64.0) as u32;
                }
                if let Some(p) = sobj.properties.get("motionBlur.shutter") {
                    mb.shutter = session_property_to_track(p, Some(&mb.shutter));
                }
            }
            if let Some(p) = sobj.properties.get("shakeAmplitude") {
                c.shake_amplitude = session_property_to_track(p, Some(&c.shake_amplitude));
            }
            if let Some(p) = sobj.properties.get("shakeSpeed") {
                c.shake_speed = session_property_to_track(p, Some(&c.shake_speed));
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "shake.mode") {
                c.shake.mode = v.round().clamp(0.0, 1.0) as u32;
            }
            if let Some(v) = prop_vec4_static(&sobj.properties, "shake.amplitude") {
                c.shake.amplitude = v;
            }
            if let Some(v) = prop_vec4_static(&sobj.properties, "shake.frequency") {
                c.shake.frequency = v;
            }
            if let Some(v) = prop_vec4_static(&sobj.properties, "shake.phase") {
                c.shake.phase = v;
            }
            if let Some(v) = prop_vec4_static(&sobj.properties, "shake.noise") {
                c.shake.noise = v;
            }
            if let Some(v) = prop_vec4_static(&sobj.properties, "shake.smooth") {
                c.shake.smooth = v;
            }
            if sobj.properties.contains_key("dofEnabled")
                || sobj.properties.contains_key("dofFocusDistance")
                || sobj.properties.contains_key("dofAperture")
                || sobj.properties.contains_key("dofFocusAreaWidth")
                || sobj.properties.contains_key("dofNearBlur")
                || sobj.properties.contains_key("dofFarBlur")
            {
                let was_none = c.dof.is_none();
                let dof = c.dof.get_or_insert_with(Default::default);
                if let Some(p) = sobj.properties.get("dofEnabled") {
                    if let Some(v) = prop_f64_static(&sobj.properties, "dofEnabled") {
                        dof.enabled = v.round() as i64 != 0;
                    }
                    if dof.enabled_track.is_some() || p.animated || p.expression.is_some() {
                        dof.enabled_track =
                            Some(session_property_to_track(p, dof.enabled_track.as_ref()));
                    }
                } else if was_none {
                    // When DOF parameters are edited on a camera without an explicit dofEnabled
                    // toggle yet, enable DOF so inspector adjustments take effect.
                    dof.enabled = true;
                }
                if let Some(p) = sobj.properties.get("dofFocusDistance") {
                    dof.focus_distance = session_property_to_track(p, Some(&dof.focus_distance));
                }
                if let Some(p) = sobj.properties.get("dofAperture") {
                    let changed = !property_track_matches_session(&dof.aperture, p);
                    dof.aperture = session_property_to_track(p, Some(&dof.aperture));
                    if changed {
                        clamp_track_f64_range(&mut dof.aperture, 0.0, 40.0);
                    }
                }
                if let Some(p) = sobj.properties.get("dofFocusAreaWidth") {
                    dof.focus_area_width =
                        session_property_to_track(p, Some(&dof.focus_area_width));
                }
                if let Some(p) = sobj.properties.get("dofNearBlur") {
                    dof.near_blur = session_property_to_track(p, Some(&dof.near_blur));
                }
                if let Some(p) = sobj.properties.get("dofFarBlur") {
                    dof.far_blur = session_property_to_track(p, Some(&dof.far_blur));
                }
            }
            if sobj.properties.keys().any(|k| k.starts_with("vibrate.")) {
                let vib = c.vibrate.get_or_insert_with(Default::default);
                if let Some(v) = prop_f64_static(&sobj.properties, "vibrate.enabled") {
                    vib.enabled = v.round() as i64 != 0;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "vibrate.seed") {
                    vib.seed = v.round().max(0.0) as u32;
                }
                if let Some(p) = sobj.properties.get("vibrate.positionAmplitude") {
                    vib.position_amplitude =
                        session_property_to_track(p, Some(&vib.position_amplitude));
                }
                if let Some(p) = sobj.properties.get("vibrate.rotationAmplitude") {
                    vib.rotation_amplitude =
                        session_property_to_track(p, Some(&vib.rotation_amplitude));
                }
                if let Some(p) = sobj.properties.get("vibrate.frequency") {
                    vib.frequency = session_property_to_track(p, Some(&vib.frequency));
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "vibrate.regularPulse") {
                    vib.regular_pulse = v.round() as i64 != 0;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "vibrate.relative") {
                    vib.relative = v.round() as i64 != 0;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "vibrate.enablePosition") {
                    vib.enable_position = v.round() as i64 != 0;
                }
                if let Some(v) = prop_f64_static(&sobj.properties, "vibrate.enableRotation") {
                    vib.enable_rotation = v.round() as i64 != 0;
                }
                if let Some(p) = sobj.properties.get("vibrate.positionFrequency") {
                    vib.position_frequency =
                        session_property_to_track(p, Some(&vib.position_frequency));
                }
                if let Some(p) = sobj.properties.get("vibrate.rotationFrequency") {
                    vib.rotation_frequency =
                        session_property_to_track(p, Some(&vib.rotation_frequency));
                }
            }
        }
        zc_project::Object3DKind::Particles(pt) => {
            if let Some(p) = sobj.properties.get("particles.time") {
                pt.time = session_property_to_track(p, Some(&pt.time));
            }
            if let Some(stops) =
                parse_gradient_stops_from_session(&sobj.properties, "particles.color")
                && !stops.is_empty()
            {
                pt.color = stops;
            }
            if let Some(stops) =
                parse_gradient_stops_from_session(&sobj.properties, "particles.size")
                && !stops.is_empty()
            {
                pt.size = stops;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.number") {
                let raw = v.round().max(0.0) as u32;
                if raw != pt.number {
                    pt.number = v.round().clamp(0.0, 100_000.0) as u32;
                }
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.rate") {
                pt.rate = v.max(0.0);
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.lifetime") {
                pt.lifetime = v.max(0.0);
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.pdist") {
                pt.pdist = v.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(v) = prop_vec3_static(&sobj.properties, "particles.ipos") {
                pt.ipos = v;
            }
            if let Some(v) = prop_vec3_static(&sobj.properties, "particles.pspread") {
                pt.pspread = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.iradius") {
                pt.iradius = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.rspread") {
                pt.rspread = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.vdist") {
                pt.vdist = v.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(v) = prop_vec3_static(&sobj.properties, "particles.ivel") {
                pt.ivel = v;
            }
            if let Some(v) = prop_vec3_static(&sobj.properties, "particles.vspread") {
                pt.vspread = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.irvel") {
                pt.irvel = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.rvspread") {
                pt.rvspread = v;
            }
            if let Some(v) = prop_vec3_static(&sobj.properties, "particles.accel") {
                pt.accel = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.iang") {
                pt.iang = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.aspread") {
                pt.aspread = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.angvel") {
                pt.angvel = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.avspread") {
                pt.avspread = v;
            }
            if let Some(v) = prop_f64_static(&sobj.properties, "particles.blending") {
                pt.blending = v.round().clamp(0.0, 3.0) as u32;
            }
            if sobj.properties.contains_key("particles.texture") {
                pt.texture = prop_str_static(&sobj.properties, "particles.texture")
                    .filter(|s| !s.is_empty());
            }
            if sobj.properties.contains_key("particles.spritePreset") {
                pt.sprite_preset = prop_str_static(&sobj.properties, "particles.spritePreset")
                    .filter(|s| !s.is_empty());
            }
        }
        zc_project::Object3DKind::Group {
            reflection_visibility,
            custom_properties,
            children: slot,
            ..
        } => {
            if let Some(rv) = prop_f64_static(&sobj.properties, "reflectionVisibility") {
                *reflection_visibility = rv.round().clamp(0.0, 2.0) as u32;
            }
            for cp in custom_properties.iter_mut() {
                if let Some(prop) = sobj.properties.get(&cp.name) {
                    cp.track = session_property_to_track(prop, Some(&cp.track));
                }
            }
            *slot = children;
        }
        zc_project::Object3DKind::Dummy { .. } => {}
    }

    sync_material_from_session(&sobj.properties, &mut obj.material);
    if sobj
        .properties
        .keys()
        .any(|k| k.starts_with("frontMaterial."))
    {
        let mut slot = obj.material_slots.remove("front");
        sync_material_from_session_with_prefix(&sobj.properties, "frontMaterial.", &mut slot);
        if let Some(mat) = slot {
            obj.material_slots.insert("front".to_owned(), mat);
        }
    }
    if sobj
        .properties
        .keys()
        .any(|k| k.starts_with("sideMaterial."))
    {
        let mut slot = obj.material_slots.remove("side");
        sync_material_from_session_with_prefix(&sobj.properties, "sideMaterial.", &mut slot);
        if let Some(mat) = slot {
            obj.material_slots.insert("side".to_owned(), mat);
        }
    }

    if uses_preserved_plugin_properties(&canonical_kind)
        || matches!(obj.kind, zc_project::Object3DKind::Dummy { .. })
    {
        for (pk, prop) in &sobj.properties {
            if is_core_object_3d_property(pk) {
                continue;
            }
            let val =
                session_property_to_preserved_value(prop, |p| session_property_to_track(p, None));
            obj.preserved.insert(format!("property.{pk}"), val.clone());
            if pk == "fieldNoiseEvolution" && !sobj.properties.contains_key("fieldEvolution") {
                obj.preserved
                    .insert("property.fieldEvolution".to_owned(), val);
            }
        }
    } else {
        // Round-trip edits to any imported legacy custom properties stored under
        // `obj.preserved["property.<name>"]` on non-plugin 3D objects as well.
        for (pk, prop) in &sobj.properties {
            if is_core_object_3d_property(pk) || matches!(pk.as_str(), "path_closed" | "iesProfile")
            {
                continue;
            }
            let preserved_key = format!("property.{pk}");
            if obj.preserved.contains_key(&preserved_key) {
                let val = session_property_to_preserved_value(prop, |p| {
                    session_property_to_track(p, None)
                });
                obj.preserved.insert(preserved_key, val);
            }
        }
    }

    let default_track_axis = if matches!(
        obj.kind,
        zc_project::Object3DKind::Camera(_) | zc_project::Object3DKind::Light(_)
    ) {
        5
    } else {
        2
    };
    if let Some(v) = prop_f64_static(&sobj.properties, "constraint.trackTo.enabled") {
        if v.round() as i64 != 0 {
            let tt =
                obj.constraints
                    .track_to
                    .get_or_insert_with(|| zc_project::TrackToConstraint {
                        target_id: String::new(),
                        track_axis: default_track_axis,
                        up_axis: 1,
                        influence: zc_project::PropertyTrack::number(100.0),
                    });
            if let Some(tid) = prop_str_static(&sobj.properties, "constraint.trackTo.targetId") {
                tt.target_id = tid;
            }
            if let Some(ta) = prop_f64_static(&sobj.properties, "constraint.trackTo.trackAxis") {
                tt.track_axis = ta.round().clamp(0.0, 5.0) as u32;
            }
            if let Some(ua) = prop_f64_static(&sobj.properties, "constraint.trackTo.upAxis") {
                tt.up_axis = ua.round().clamp(0.0, 2.0) as u32;
            }
            if let Some(p) = sobj.properties.get("constraint.trackTo.influence") {
                let changed = !property_track_matches_session(&tt.influence, p);
                tt.influence = session_property_to_track(p, Some(&tt.influence));
                if changed {
                    clamp_track_f64_range(&mut tt.influence, 0.0, 100.0);
                }
            }
        } else {
            obj.constraints.track_to = None;
        }
    }
    if let Some(v) = prop_f64_static(&sobj.properties, "constraint.follow.enabled") {
        if v.round() as i64 != 0 {
            let fl = obj
                .constraints
                .follow
                .get_or_insert_with(|| zc_project::FollowConstraint {
                    target_id: String::new(),
                    space: 0,
                    follow_position: true,
                    follow_rotation: true,
                    follow_scale: true,
                    position_offset: zc_project::PropertyTrack::vec3([0.0, 0.0, 0.0]),
                    rotation_offset: zc_project::PropertyTrack::vec3([0.0, 0.0, 0.0]),
                    scale_offset: zc_project::PropertyTrack::vec3([1.0, 1.0, 1.0]),
                    fallback_pose: None,
                });
            if let Some(tid) = prop_str_static(&sobj.properties, "constraint.follow.targetId") {
                fl.target_id = tid;
            }
            let sp_opt = prop_f64_static(&sobj.properties, "constraint.follow.space")
                .map(|v| v.round().clamp(0.0, 1.0) as u32);
            let fsp_opt = prop_f64_static(&sobj.properties, "constraint.follow.followSpace")
                .map(|v| v.round().clamp(0.0, 1.0) as u32);
            match (sp_opt, fsp_opt) {
                (Some(s), Some(f)) => {
                    if f != fl.space && s == fl.space {
                        fl.space = f;
                    } else {
                        fl.space = s;
                    }
                }
                (Some(s), None) => fl.space = s,
                (None, Some(f)) => fl.space = f,
                (None, None) => {}
            }
            if let Some(fp) = prop_f64_static(&sobj.properties, "constraint.follow.followPosition")
            {
                fl.follow_position = fp.round() as i64 != 0;
            }
            if let Some(fr) = prop_f64_static(&sobj.properties, "constraint.follow.followRotation")
            {
                fl.follow_rotation = fr.round() as i64 != 0;
            }
            if let Some(fs) = prop_f64_static(&sobj.properties, "constraint.follow.followScale") {
                fl.follow_scale = fs.round() as i64 != 0;
            }
            if let Some(p) = sobj.properties.get("constraint.follow.positionOffset") {
                fl.position_offset = session_property_to_track(p, Some(&fl.position_offset));
            }
            if let Some(p) = sobj.properties.get("constraint.follow.rotationOffset") {
                fl.rotation_offset = session_property_to_track(p, Some(&fl.rotation_offset));
            }
            if let Some(p) = sobj.properties.get("constraint.follow.scaleOffset") {
                fl.scale_offset = session_property_to_track(p, Some(&fl.scale_offset));
            }
        } else {
            obj.constraints.follow = None;
        }
    }

    obj
}

pub(crate) fn collect_existing_objects(
    objects: Vec<zc_project::Object3D>,
    out: &mut BTreeMap<String, zc_project::Object3D>,
) {
    for mut obj in objects {
        if let zc_project::Object3DKind::Group {
            children: ref mut c,
            ..
        } = obj.kind
        {
            let nested = std::mem::take(c);
            collect_existing_objects(nested, out);
        }
        out.insert(obj.id.to_string(), obj);
    }
}

pub(crate) fn rebuild_object_tree(
    flat: &[SessionObject3D],
    existing: &mut BTreeMap<String, zc_project::Object3D>,
    templates: Option<&BTreeMap<String, zc_project::Object3D>>,
) -> Vec<zc_project::Object3D> {
    let mut children_by_parent: HashMap<Option<&str>, Vec<usize>> =
        HashMap::with_capacity(flat.len().min(64) + 1);
    for (idx, sobj) in flat.iter().enumerate() {
        children_by_parent
            .entry(sobj.parent_id.as_deref())
            .or_default()
            .push(idx);
    }
    rebuild_object_tree_node(flat, &children_by_parent, None, existing, templates, 0)
}

fn rebuild_object_tree_node(
    flat: &[SessionObject3D],
    children_by_parent: &HashMap<Option<&str>, Vec<usize>>,
    parent_id: Option<&str>,
    existing: &mut BTreeMap<String, zc_project::Object3D>,
    templates: Option<&BTreeMap<String, zc_project::Object3D>>,
    depth: usize,
) -> Vec<zc_project::Object3D> {
    if depth > 64 {
        return Vec::new();
    }
    let Some(indices) = children_by_parent.get(&parent_id) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        let sobj = &flat[idx];
        let children = rebuild_object_tree_node(
            flat,
            children_by_parent,
            Some(&sobj.id),
            existing,
            templates,
            depth + 1,
        );
        let prev = existing.remove(&sobj.id).or_else(|| {
            templates.and_then(|tpls| {
                sobj.source_id
                    .as_deref()
                    .and_then(|src_id| tpls.get(src_id))
                    .map(|tpl| {
                        let mut cloned = tpl.clone();
                        cloned.id = parse_uuid_or_new(&sobj.id);
                        cloned
                    })
            })
        });
        out.push(build_object_3d_from_session(sobj, prev, children));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session_obj(kind: &str, props: Vec<(&str, Value)>) -> SessionObject3D {
        let mut properties = BTreeMap::new();
        for (k, v) in props {
            properties.insert(k.to_owned(), SessionProperty::static_value(v));
        }
        SessionObject3D {
            id: "00000000-0000-4000-8000-000000000001".to_owned(),
            parent_id: None,
            source_id: None,
            name: format!("Test {kind}"),
            kind: kind.to_owned(),
            properties,
        }
    }

    #[test]
    fn particles_color_and_size_over_life_accept_ui_and_legacy_schemas() {
        // 1. UI schema: {"t", "color": [r,g,b,a]} and {"t", "size": s}
        let ui_color = json!([
            { "t": 0.0, "color": [1.0, 0.5, 0.0, 1.0] },
            { "t": 1.0, "color": [0.0, 0.2, 1.0, 0.25] }
        ])
        .to_string();
        let ui_size = json!([
            { "t": 0.0, "size": 2.5 },
            { "t": 0.5, "size": 4.0 },
            { "t": 1.0, "size": 0.0 }
        ])
        .to_string();
        let sobj = make_session_obj(
            "particles",
            vec![
                ("particles.color", json!(ui_color)),
                ("particles.size", json!(ui_size)),
            ],
        );
        let obj = build_object_3d_from_session(&sobj, None, Vec::new());
        let zc_project::Object3DKind::Particles(pt) = obj.kind else {
            panic!("expected Particles");
        };
        assert_eq!(pt.color.len(), 2);
        assert!((pt.color[0].position - 0.0).abs() < 1e-9);
        assert_eq!(pt.color[0].color, "rgba(255,128,0,1)");
        assert!((pt.color[1].position - 1.0).abs() < 1e-9);
        assert_eq!(pt.color[1].color, "rgba(0,51,255,0.25)");

        assert_eq!(pt.size.len(), 3);
        assert!((pt.size[1].position - 0.5).abs() < 1e-9);
        assert_eq!(pt.size[1].color, "4");
        assert_eq!(pt.size[2].color, "0");

        // 2. Legacy schema: {"position", "color": "#rrggbb"} and {"position", "value": s}
        let legacy_color = json!([
            { "position": 0.0, "color": "#ff0000" },
            { "position": 1.0, "color": "#00ff00" }
        ])
        .to_string();
        let legacy_size = json!([
            { "position": 0.0, "value": 1.25 },
            { "position": 1.0, "value": 0.5 }
        ])
        .to_string();
        let sobj_legacy = make_session_obj(
            "particles",
            vec![
                ("particles.color", json!(legacy_color)),
                ("particles.size", json!(legacy_size)),
            ],
        );
        let obj_legacy = build_object_3d_from_session(&sobj_legacy, None, Vec::new());
        let zc_project::Object3DKind::Particles(pt2) = obj_legacy.kind else {
            panic!("expected Particles");
        };
        assert_eq!(pt2.color.len(), 2);
        assert_eq!(pt2.color[0].color, "#ff0000");
        assert_eq!(pt2.color[1].color, "#00ff00");
        assert_eq!(pt2.size.len(), 2);
        assert_eq!(pt2.size[0].color, "1.25");
        assert_eq!(pt2.size[1].color, "0.5");
    }

    #[test]
    fn wire_path_closed_and_light_ies_profile_sync_into_preserved() {
        let wire_sobj = make_session_obj("wire", vec![("path_closed", json!(1))]);
        let wire = build_object_3d_from_session(&wire_sobj, None, Vec::new());
        assert_eq!(wire.preserved.get("property.path_closed"), Some(&json!(1)));

        let light_sobj = make_session_obj(
            "zoidium:light-plus/ies-light",
            vec![("iesProfile", json!("ies_asset_sha256"))],
        );
        let light = build_object_3d_from_session(&light_sobj, None, Vec::new());
        assert_eq!(
            light.preserved.get("property.iesProfile"),
            Some(&json!("ies_asset_sha256"))
        );
    }
}
