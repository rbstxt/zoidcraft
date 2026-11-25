use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use uuid::Uuid;

use crate::effects::{
    create_custom_shader, create_effect_group, create_standard_effect, create_zoidium_effect,
};
use crate::model::{
    CustomProperty, EffectInstance, EffectKind, GradientStop, PropertyTrack, PropertyValue,
};

use super::sequence::SequenceSideTables;
use super::tracks::{
    infer_default_value, is_custom_prop_rgb_color, kf_value, parse_property_track, prop_str,
    sanitize_display_name, value_as_bool, value_as_u32,
};

pub(crate) fn parse_effect(
    val: &Value,
    side_tables: &mut SequenceSideTables,
    plugins: &mut BTreeSet<String>,
    warnings: &mut Vec<String>,
) -> EffectInstance {
    let Some(obj) = val.as_object() else {
        return create_effect_group("Effect");
    };
    let props = obj.get("properties").and_then(Value::as_object);
    let eff_type = obj.get("type");

    let is_string_effect_group = eff_type.and_then(Value::as_str).is_some_and(|s| {
        matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "group" | "effectgroup" | "effect_group"
        )
    });

    let custom_props_val = obj
        .get("customProperties")
        .or_else(|| props.and_then(|p| p.get("customProperties")));

    let mut instance = match eff_type.and_then(value_as_u32) {
        Some(0) => {
            let mut grp = create_effect_group("Group");
            let custom_properties = parse_custom_properties(custom_props_val);
            let mut children = Vec::new();
            for arr in [
                obj.get("objects").and_then(Value::as_array),
                obj.get("effects").and_then(Value::as_array),
            ]
            .into_iter()
            .flatten()
            {
                for c in arr {
                    children.push(parse_effect(c, side_tables, plugins, warnings));
                }
            }
            grp.kind = EffectKind::EffectGroup {
                custom_properties,
                children,
            };
            grp
        }
        None if is_string_effect_group => {
            let mut grp = create_effect_group("Group");
            let custom_properties = parse_custom_properties(custom_props_val);
            let mut children = Vec::new();
            for arr in [
                obj.get("objects").and_then(Value::as_array),
                obj.get("effects").and_then(Value::as_array),
            ]
            .into_iter()
            .flatten()
            {
                for c in arr {
                    children.push(parse_effect(c, side_tables, plugins, warnings));
                }
            }
            grp.kind = EffectKind::EffectGroup {
                custom_properties,
                children,
            };
            grp
        }
        Some(1) => {
            let frag_shader = props
                .and_then(|p| p.get("fragShader"))
                .and_then(extract_shader_string)
                .unwrap_or_default();
            let mut shader = create_custom_shader("Custom Shader", frag_shader.clone());
            let custom_properties = parse_custom_properties(custom_props_val);
            shader.kind = EffectKind::CustomShader {
                frag_shader,
                custom_properties,
            };
            shader
        }
        _ => {
            let id_str = eff_type
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_owned();
            if let Some((plugin, bare)) = id_str
                .strip_prefix("zoidium:")
                .and_then(|rest| rest.split_once('/'))
                .filter(|(p, b)| !p.is_empty() && !b.is_empty())
            {
                if let Some(zoid_eff) = create_zoidium_effect(bare) {
                    if let EffectKind::Plugin { ref plugin_id, .. } = zoid_eff.kind {
                        plugins.insert(plugin_id.clone());
                    }
                    zoid_eff
                } else {
                    plugins.insert(plugin.to_owned());
                    warnings.push(format!("preserved unknown plugin effect {id_str:?}"));
                    EffectInstance {
                        id: Uuid::new_v4(),
                        name: bare.to_owned(),
                        kind: EffectKind::Plugin {
                            plugin_id: plugin.to_owned(),
                            effect_id: bare.to_owned(),
                        },
                        enabled: PropertyTrack::option(1),
                        params: BTreeMap::new(),
                        preserved: BTreeMap::new(),
                    }
                }
            } else if let Some(std_eff) = create_standard_effect(&id_str) {
                std_eff
            } else if let Some(zoid_eff) = create_zoidium_effect(&id_str) {
                if let EffectKind::Plugin { ref plugin_id, .. } = zoid_eff.kind {
                    plugins.insert(plugin_id.clone());
                }
                zoid_eff
            } else {
                warnings.push(format!("preserved unknown effect {id_str:?}"));
                EffectInstance {
                    id: Uuid::new_v4(),
                    name: id_str.clone(),
                    kind: EffectKind::Plugin {
                        plugin_id: "unknown".to_owned(),
                        effect_id: id_str,
                    },
                    enabled: PropertyTrack::option(1),
                    params: BTreeMap::new(),
                    preserved: BTreeMap::new(),
                }
            }
        }
    };

    if obj
        .get("zoidiumBypassed")
        .and_then(value_as_bool)
        .unwrap_or(false)
    {
        side_tables.bypassed_effects.insert(instance.id);
    }

    if let Some(props) = props {
        if let Some(n) = prop_str(props, "name") {
            instance.name = n;
        }
        if let Some(en) = props.get("enabled") {
            instance.enabled = parse_property_track(en, PropertyValue::Option(1));
        }

        let has_custom_props = matches!(
            instance.kind,
            EffectKind::CustomShader { .. } | EffectKind::EffectGroup { .. }
        );
        for (k, v) in props {
            if matches!(k.as_str(), "name" | "enabled" | "fragShader")
                || (has_custom_props && k == "customProperties")
            {
                continue;
            }
            if k == "shake"
                && let Some(shake_map) = v.as_object()
            {
                for (sk, sv) in shake_map {
                    let full_key = format!("shake.{sk}");
                    let default_val = instance
                        .params
                        .get(&full_key)
                        .map_or_else(|| infer_default_value(sv), |t| t.value.clone());
                    instance
                        .params
                        .insert(full_key, parse_property_track(sv, default_val));
                }
                continue;
            }

            if let Some(existing) = instance.params.get(k) {
                let parsed = parse_property_track(v, existing.value.clone());
                instance.params.insert(k.clone(), parsed);
            } else {
                let inferred = infer_default_value(v);
                instance
                    .params
                    .insert(k.clone(), parse_property_track(v, inferred));
            }
        }
        // CM3 `vignette` maps to Magic Looks (`magiclooks`): preserve `darkness`/`amount`
        // AND mirror into `vignetteStrength` when `vignetteStrength` was not explicitly set.
        if eff_type
            .and_then(Value::as_str)
            .is_some_and(|s| s.trim().eq_ignore_ascii_case("vignette"))
            && !props.contains_key("vignetteStrength")
            && let Some(src_track) = instance
                .params
                .get("darkness")
                .or_else(|| instance.params.get("amount"))
                .cloned()
        {
            instance
                .params
                .insert("vignetteStrength".to_owned(), src_track);
        }
    }

    let is_effect_group = matches!(instance.kind, EffectKind::EffectGroup { .. });
    for (k, v) in obj {
        if !matches!(
            k.as_str(),
            "type" | "properties" | "objects" | "customProperties"
        ) && !(is_effect_group && k == "effects")
        {
            instance.preserved.insert(k.clone(), v.clone());
        }
    }

    instance
}

pub(super) fn extract_shader_string(val: &Value) -> Option<String> {
    if let Some(s) = val.as_str() {
        return Some(s.to_owned());
    }
    let obj = val.as_object()?;
    if let Some(kfs) = obj.get("keyframes").and_then(Value::as_array) {
        return kfs
            .first()
            .and_then(kf_value)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
    }
    obj.get("value").and_then(extract_shader_string)
}

pub(super) fn parse_custom_properties(val: Option<&Value>) -> Vec<CustomProperty> {
    let Some(arr) = val.and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let props = obj.get("properties").and_then(Value::as_object);
        let type_obj = obj.get("type").and_then(Value::as_object);
        let prop_type = obj
            .get("type")
            .and_then(value_as_u32)
            .or_else(|| type_obj.and_then(|t| t.get("type")).and_then(value_as_u32))
            .or_else(|| props.and_then(|p| p.get("type")).and_then(value_as_u32));
        let name = props
            .and_then(|p| prop_str(p, "name"))
            .or_else(|| {
                obj.get("name")
                    .and_then(Value::as_str)
                    .map(sanitize_display_name)
            })
            .or_else(|| {
                type_obj
                    .and_then(|t| t.get("name"))
                    .and_then(Value::as_str)
                    .map(sanitize_display_name)
            })
            .unwrap_or_else(|| "Property".to_owned());
        let has_top_track = obj.contains_key("value")
            || obj.contains_key("keyframes")
            || obj.contains_key("objects");
        let used_props_value = !has_top_track && props.and_then(|p| p.get("value")).is_some();
        let value_node = if let Some(v) = obj.get("value") {
            v
        } else if used_props_value {
            props.and_then(|p| p.get("value")).unwrap_or(item)
        } else {
            item
        };
        let default_val = match prop_type {
            Some(0) => PropertyValue::Number(0.0),
            Some(1) => PropertyValue::Vec2([0.0, 0.0]),
            Some(2) => PropertyValue::Vec3([0.0, 0.0, 0.0]),
            Some(3) => {
                if is_custom_prop_rgb_color(value_node, type_obj) {
                    PropertyValue::Color([1.0, 1.0, 1.0])
                } else {
                    PropertyValue::Vec4([0.0, 0.0, 0.0, 0.0])
                }
            }
            Some(4) => PropertyValue::Color([1.0, 1.0, 1.0]),
            Some(5) => {
                let is_bool_val = value_node.as_bool().is_some()
                    || value_node
                        .get("keyframes")
                        .and_then(Value::as_array)
                        .and_then(|arr| arr.first())
                        .and_then(|k| k.get("value").or_else(|| k.get("v")))
                        .and_then(Value::as_bool)
                        .is_some();
                if is_bool_val {
                    PropertyValue::Bool(false)
                } else {
                    PropertyValue::Gradient(GradientStop::white_default())
                }
            }
            Some(6) => PropertyValue::Option(0),
            Some(7) => PropertyValue::Text(String::new()),
            Some(8) => PropertyValue::Asset(None),
            Some(9) => PropertyValue::List(String::new()),
            Some(10) => PropertyValue::Curve(Value::Null),
            Some(11) => PropertyValue::Shader(String::new()),
            _ => infer_default_value(value_node),
        };
        let mut track = parse_property_track(value_node, default_val);
        if obj.contains_key("value") || used_props_value {
            if track.expression.is_none()
                && let Some(expr) = obj
                    .get("expression")
                    .or_else(|| props.and_then(|p| p.get("expression")))
                    .and_then(Value::as_str)
            {
                track.expression = Some(expr.to_owned());
            }
            if let Some(anim) = obj
                .get("animated")
                .or_else(|| props.and_then(|p| p.get("animated")))
                .and_then(Value::as_bool)
            {
                track.animated = track.animated || anim;
            }
        }
        let mut metadata = BTreeMap::new();
        for (k, v) in obj {
            let preserve_type_obj = k == "type" && type_obj.is_some();
            if preserve_type_obj
                || !matches!(
                    k.as_str(),
                    "name" | "type" | "value" | "keyframes" | "animated" | "expression" | "objects"
                )
            {
                metadata.insert(k.clone(), v.clone());
            }
        }
        out.push(CustomProperty {
            name,
            prop_type,
            track,
            metadata,
        });
    }
    out
}
