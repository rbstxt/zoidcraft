use super::*;
use crate::model::{EffectKind, LayerKind, MaskMode, Object3DKind, PropertyValue, Shape3DKind};
use serde_json::json;

#[test]
fn parses_bare_cm3_effects_and_effect_group() {
    let bloom_json = r#"{
        "type": "bloom",
        "properties": {
            "name": "Hero Bloom",
            "enabled": { "animated": false, "keyframes": [{ "frame": 0, "value": 1, "tween": 0 }] },
            "intensity": { "animated": true, "keyframes": [
                { "frame": 0, "value": 0.5, "tween": 1 },
                { "frame": 30, "value": 2.0, "tween": 1 }
            ] }
        }
    }"#;

    let res = parse_pz_clipboard_text(bloom_json, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Effects(effects) = res.payload else {
        panic!("expected Effects payload");
    };
    assert_eq!(effects.len(), 1);
    assert_eq!(effects[0].name, "Hero Bloom");
    assert!(matches!(effects[0].kind, EffectKind::Standard(ref id) if id == "bloom"));
    let intensity = effects[0].params.get("intensity").unwrap();
    assert!(intensity.animated);
    assert_eq!(intensity.keyframes.len(), 2);

    // Nested EffectGroup with rgbshift child
    let group_json = r#"{
        "type": 0,
        "properties": { "name": "Glitch Stack" },
        "objects": [
            {
                "type": "rgbshift",
                "properties": { "name": "Chromatic", "amount": 0.025 }
            }
        ]
    }"#;
    let res2 = parse_pz_clipboard_text(group_json, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Effects(grp_fx) = res2.payload else {
        panic!("expected Effects payload for EffectGroup");
    };
    assert_eq!(grp_fx.len(), 1);
    assert_eq!(grp_fx[0].name, "Glitch Stack");
    let EffectKind::EffectGroup { children, .. } = &grp_fx[0].kind else {
        panic!("expected EffectGroup");
    };
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].name, "Chromatic");
    assert!(matches!(children[0].kind, EffectKind::Standard(ref id) if id == "rgbshift"));

    // Repeated pastes must produce distinct UUIDs
    let res3 = parse_pz_clipboard_text(bloom_json, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Effects(effects3) = res3.payload else {
        panic!("expected Effects");
    };
    assert_ne!(effects[0].id, effects3[0].id);
}

#[test]
fn parses_all_panzoid_shader_object_variants_and_raw_glsl() {
    // 1. Canonical CM3 Shader Object (type: 1 + customProperties with frame-0 defaults & sampler2D)
    let cm3_shader = r#"{
        "type": 1,
        "properties": {
            "name": "Neon Plasma",
            "fragShader": "uniform sampler2D tDiffuse; uniform vec3 Color; uniform float Intensity; uniform sampler2D NoiseMap; varying vec2 vUv; void main() { gl_FragColor = vec4(Color * Intensity, 1.0); }"
        },
        "customProperties": [
            {
                "name": "Color",
                "type": 4,
                "animated": false,
                "keyframes": [{ "frame": 0, "value": [0.2, 0.8, 1.0], "tween": 1 }]
            },
            {
                "name": "Intensity",
                "type": 0,
                "animated": false,
                "keyframes": [{ "frame": 0, "value": 1.75, "tween": 1 }]
            },
            {
                "name": "NoiseMap",
                "type": 8,
                "value": null
            }
        ]
    }"#;
    let res = parse_pz_clipboard_text(cm3_shader, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Effects(fxs) = res.payload else {
        panic!("expected Effects payload");
    };
    assert_eq!(fxs.len(), 1);
    assert_eq!(fxs[0].name, "Neon Plasma");
    let EffectKind::CustomShader {
        frag_shader,
        custom_properties,
    } = &fxs[0].kind
    else {
        panic!("expected CustomShader");
    };
    assert!(frag_shader.contains("gl_FragColor"));
    assert_eq!(custom_properties.len(), 3);
    assert_eq!(custom_properties[0].name, "Color");
    assert_eq!(custom_properties[0].prop_type, Some(4));
    assert_eq!(
        custom_properties[0].track.value,
        PropertyValue::Color([0.2, 0.8, 1.0])
    );
    assert_eq!(custom_properties[1].name, "Intensity");
    assert_eq!(
        custom_properties[1].track.value,
        PropertyValue::Number(1.75)
    );
    assert_eq!(custom_properties[2].name, "NoiseMap");
    assert_eq!(custom_properties[2].prop_type, Some(8));
    assert_eq!(custom_properties[2].track.value, PropertyValue::Asset(None));

    // 2. String `type: "shader"` with top-level fragShader
    let string_shader = r#"{
        "type": "shader",
        "name": "Top-Level Shader",
        "fragShader": "void main() { gl_FragColor = vec4(1.0); }",
        "customProperties": [{ "name": "Amount", "type": "float", "value": 0.42 }]
    }"#;
    let res_str = parse_pz_clipboard_text(string_shader, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Effects(fxs_str) = res_str.payload else {
        panic!("expected Effects");
    };
    assert_eq!(fxs_str[0].name, "Top-Level Shader");
    let EffectKind::CustomShader {
        frag_shader: fs2,
        custom_properties: cps2,
    } = &fxs_str[0].kind
    else {
        panic!("expected CustomShader");
    };
    assert!(fs2.contains("gl_FragColor"));
    assert_eq!(cps2.len(), 1);
    assert_eq!(cps2[0].prop_type, Some(0));
    assert_eq!(cps2[0].track.value, PropertyValue::Number(0.42));

    // 3. Top-level `fragmentShader` without `"type"`
    let typeless_shader = r#"{
        "name": "Typeless FX",
        "fragmentShader": "void main() { gl_FragColor = vec4(0.5); }"
    }"#;
    let res_typeless = parse_pz_clipboard_text(typeless_shader, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Effects(fxs_tl) = res_typeless.payload else {
        panic!("expected Effects");
    };
    assert_eq!(fxs_tl[0].name, "Typeless FX");
    assert!(matches!(fxs_tl[0].kind, EffectKind::CustomShader { .. }));

    // 4. Raw GLSL ES 1.00 fragment shader source -> auto-wrapped + uniform extraction!
    let raw_glsl = r#"
        precision highp float;
        uniform sampler2D tDiffuse;
        uniform vec2 resolution;
        uniform float time;
        // Custom uniforms:
        uniform vec3 Color;
        uniform float Opacity;
        uniform vec2 Scale;
        uniform sampler2D MaskTex;
        /* uniform float IgnoredComment; */
        varying vec2 vUv;
        void main() {
            gl_FragColor = vec4(Color * Opacity, 1.0);
        }
    "#;
    let res_raw = parse_pz_clipboard_text(raw_glsl, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Effects(fxs_raw) = res_raw.payload else {
        panic!("expected Effects from raw GLSL");
    };
    assert_eq!(fxs_raw[0].name, "Pasted Shader");
    let EffectKind::CustomShader {
        custom_properties: raw_cps,
        ..
    } = &fxs_raw[0].kind
    else {
        panic!("expected CustomShader");
    };
    let names: Vec<&str> = raw_cps.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, vec!["Color", "Opacity", "Scale", "MaskTex"]);
    assert_eq!(raw_cps[0].prop_type, Some(4)); // vec3 Color -> Color
    assert_eq!(raw_cps[1].track.value, PropertyValue::Number(1.0)); // Opacity default 1.0
    assert_eq!(raw_cps[2].track.value, PropertyValue::Vec2([1.0, 1.0])); // Scale default [1,1]
    assert_eq!(raw_cps[3].prop_type, Some(8)); // sampler2D -> Asset
}

#[test]
fn parses_3d_objects_text_shape_group_light_particles() {
    let objects_json = r#"[
        {
            "type": 1,
            "properties": {
                "name": "Hero Text",
                "text": "ZOIDCRAFT",
                "height": 12.0,
                "bevelEnabled": 1,
                "position": { "objects": [
                    { "animated": true, "keyframes": [{ "frame": 0, "value": 0 }, { "frame": 30, "value": 15 }] },
                    { "animated": false, "keyframes": [{ "frame": 0, "value": 2 }] },
                    { "animated": false, "keyframes": [{ "frame": 0, "value": 0 }] }
                ]}
            },
            "material": {
                "type": "custom",
                "properties": { "color": [1.0, 0.5, 0.2], "roughness": 0.2, "metalness": 0.8 }
            }
        },
        {
            "type": 5,
            "properties": { "name": "Rig Group" },
            "objects": [
                {
                    "type": 0,
                    "objectType": 1,
                    "properties": { "name": "Cube Child", "geometryProperties": { "box_size": [4, 5, 6] } }
                }
            ]
        },
        {
            "type": 3,
            "objectType": 2,
            "properties": { "name": "Key Light", "intensity": 2.5 }
        },
        {
            "type": 4,
            "properties": { "name": "Sparks", "count": 200 }
        }
    ]"#;

    let res = parse_pz_clipboard_text(objects_json, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Objects3D(objs) = res.payload else {
        panic!("expected Objects3D");
    };
    assert_eq!(objs.len(), 4);
    assert_eq!(objs[0].name, "Hero Text");
    assert!(matches!(objs[0].kind, Object3DKind::Text3D(_)));
    assert!(objs[0].position.animated);
    assert!(matches!(objs[0].material, Some(MaterialSpec::CustomPbr(_))));

    assert_eq!(objs[1].name, "Rig Group");
    let Object3DKind::Group { children, .. } = &objs[1].kind else {
        panic!("expected Group");
    };
    assert_eq!(children.len(), 1);
    assert!(matches!(
        children[0].kind,
        Object3DKind::Shape3D(Shape3DKind::Box { .. })
    ));
    assert!(matches!(objs[2].kind, Object3DKind::Light(_)));
    assert!(matches!(objs[3].kind, Object3DKind::Particles(_)));
}

#[test]
fn parses_clips_materials_keyframes_masks_and_shape_plus() {
    // Video Clip + Audio Clip
    let clips_json = r#"[
        {
            "start": 10,
            "length": 90,
            "properties": { "name": "Scene Clip" },
            "object": {
                "type": 4,
                "properties": { "name": "3D Scene" },
                "objects": [
                    { "type": 0, "objectType": 5, "properties": { "name": "Ball" } }
                ],
                "effects": [
                    { "type": "bloom", "properties": { "name": "Glow" } }
                ]
            }
        },
        {
            "start": 10,
            "length": 90,
            "properties": { "name": "Beat" },
            "audio": { "properties": { "name": "Beat", "volume": 0.8, "pan": -0.2 } }
        }
    ]"#;
    let res = parse_pz_clipboard_text(clips_json, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Clips(clips) = res.payload else {
        panic!("expected Clips");
    };
    assert_eq!(clips.len(), 2);
    match &clips[0] {
        PzClipboardClip::Video { clip, .. } => {
            assert_eq!(clip.name, "Scene Clip");
            assert_eq!(clip.layer.effects.len(), 1);
            let LayerKind::Scene3D { objects, .. } = &clip.layer.kind else {
                panic!("expected Scene3D layer");
            };
            assert_eq!(objects.len(), 1);
        }
        _ => panic!("expected Video clip"),
    }
    assert!(matches!(&clips[1], PzClipboardClip::Audio(ac) if ac.name == "Beat"));

    // Node Material raw graph
    let node_mat = r#"{
        "nodes": [{ "id": "n1", "type": "output" }],
        "links": [{ "from": "n0", "to": "n1" }]
    }"#;
    let res_mat = parse_pz_clipboard_text(node_mat, PzClipboardHint::Any).unwrap();
    assert!(res_mat.required_plugins.contains("material-plus"));
    assert!(matches!(
        res_mat.payload,
        PzClipboardPayload::Material(MaterialSpec::NodeMaterial { .. })
    ));

    // Keyframes + expression
    let kf_json = r#"{
        "animated": true,
        "expression": "sin(time * 4.0) * 10.0",
        "keyframes": [
            { "frame": 0, "value": 0.0, "tween": 1 },
            { "frame": 60, "value": 100.0, "tween": 3 }
        ]
    }"#;
    let res_kf = parse_pz_clipboard_text(kf_json, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::PropertyTrack { track } = res_kf.payload else {
        panic!("expected PropertyTrack");
    };
    assert!(track.animated);
    assert_eq!(track.expression.as_deref(), Some("sin(time * 4.0) * 10.0"));
    assert_eq!(track.keyframes.len(), 2);

    // LayerMask
    let mask_json = r#"{
        "name": "Vignette Mask",
        "mode": "subtract",
        "feather": 18.0,
        "invert": true,
        "properties": {
            "closed": true,
            "path": [0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0]
        }
    }"#;
    let res_mask = parse_pz_clipboard_text(mask_json, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::LayerMasks(masks) = res_mask.payload else {
        panic!("expected LayerMasks");
    };
    assert_eq!(masks.len(), 1);
    assert_eq!(masks[0].mode, MaskMode::Subtract);
    assert!(masks[0].invert);
    assert!((masks[0].feather - 18.0).abs() < 1e-6);

    // Shape+ item
    let sp_json = r#"{
        "kind": "rectangle",
        "properties": { "name": "Card", "size": [320, 180], "roundness": 16 }
    }"#;
    let res_sp = parse_pz_clipboard_text(sp_json, PzClipboardHint::Any).unwrap();
    assert!(res_sp.required_plugins.contains("shape-plus"));
    assert!(matches!(
        res_sp.payload,
        PzClipboardPayload::ShapePlusContents(_)
    ));
}

#[test]
fn rejects_oversized_empty_and_malformed_inputs() {
    assert!(parse_pz_clipboard_text("   ", PzClipboardHint::Any).is_err());
    assert!(parse_pz_clipboard_text("{not valid json", PzClipboardHint::Any).is_err());
    let huge = " ".repeat(MAX_CLIPBOARD_JSON_BYTES + 1);
    assert!(parse_pz_clipboard_text(&huge, PzClipboardHint::Any).is_err());
}

#[test]
fn round_trips_serialized_cm3_json() {
    let seq = Sequence::default();
    let fx = crate::effects::create_standard_effect("bloom").unwrap();
    let fx_json = serialize_pz_effects_json(&[fx], &seq);
    let parsed = parse_pz_clipboard_text(&fx_json, PzClipboardHint::Any).unwrap();
    assert!(matches!(parsed.payload, PzClipboardPayload::Effects(v) if v.len() == 1));

    let mat = MaterialSpec::CustomPbr(Box::default());
    let mat_json = serialize_pz_material_json(&mat);
    let parsed_mat = parse_pz_clipboard_text(&mat_json, PzClipboardHint::Any).unwrap();
    assert!(matches!(
        parsed_mat.payload,
        PzClipboardPayload::Material(MaterialSpec::CustomPbr(_))
    ));
}

#[test]
fn parses_effect_group_with_effects_children_alias_and_shape_plus_type_alias() {
    // Community EffectGroup using `"effects": [...]` on `type: 0`
    let grp_num = r#"{
        "type": 0,
        "properties": { "name": "Glow Stack" },
        "effects": [
            { "type": "gaussianblur", "properties": { "name": "Soft Blur", "amount": 6.0 } },
            { "type": "bloom", "properties": { "name": "Outer Bloom" } }
        ]
    }"#;
    let parsed_num = parse_pz_clipboard_text(grp_num, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Effects(fxs_num) = parsed_num.payload else {
        panic!("expected Effects payload for type:0 EffectGroup with `effects` array");
    };
    assert_eq!(fxs_num.len(), 1);
    let EffectKind::EffectGroup { ref children, .. } = fxs_num[0].kind else {
        panic!("expected EffectGroup");
    };
    assert_eq!(children.len(), 2);
    assert_eq!(children[0].name, "Soft Blur");
    assert_eq!(children[1].name, "Outer Bloom");

    // Community EffectGroup using `type: "group"` + `"effects": [...]`
    let grp_str = r#"{
        "type": "group",
        "name": "String Group",
        "effects": [
            { "type": "rgbshift", "properties": { "amount": 0.02 } }
        ]
    }"#;
    let parsed_str = parse_pz_clipboard_text(grp_str, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Effects(fxs_str) = parsed_str.payload else {
        panic!("expected Effects payload for type:\"group\" with `effects` array");
    };
    let EffectKind::EffectGroup { ref children, .. } = fxs_str[0].kind else {
        panic!("expected EffectGroup");
    };
    assert_eq!(children.len(), 1);
    assert_eq!(
        children[0].kind,
        EffectKind::Standard("rgbshift".to_owned())
    );

    // Shape+ item using `"type": "rect"` alias for `"kind": "rectangle"`
    let sp_type_rect =
        r#"{ "type": "rect", "properties": { "name": "Badge", "size": [120, 40] } }"#;
    let parsed_rect = parse_pz_clipboard_text(sp_type_rect, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::ShapePlusContents(val_rect) = parsed_rect.payload else {
        panic!("expected ShapePlusContents for type:rect");
    };
    assert_eq!(val_rect["kind"], "rectangle");
    assert!(val_rect.get("type").is_none());

    // Shape+ item using `"kind": "rect"` alias
    let sp_kind_rect = r#"{ "kind": "rect", "properties": { "name": "Pill" } }"#;
    let parsed_kind = parse_pz_clipboard_text(sp_kind_rect, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::ShapePlusContents(val_kind) = parsed_kind.payload else {
        panic!("expected ShapePlusContents for kind:rect");
    };
    assert_eq!(val_kind["kind"], "rectangle");
}

#[test]
fn qa4_cm3_type0_objects_effect_group_classification() {
    let fx_grp_str_child = r#"{
        "type": 0,
        "properties": { "name": "FX Group" },
        "objects": [{ "type": "bloom", "properties": { "strength": 1.25 } }]
    }"#;
    let res = parse_pz_clipboard_text(fx_grp_str_child, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Effects(fxs) = res.payload else {
        panic!("expected Effects for type:0 + bloom child");
    };
    assert_eq!(fxs.len(), 1);
    assert!(
        matches!(fxs[0].kind, EffectKind::EffectGroup { ref children, .. } if children.len() == 1)
    );

    let fx_grp_num_child = r#"{
        "type": 0,
        "objects": [{ "type": 1, "properties": { "fragShader": "void main(){gl_FragColor=vec4(1.0);}" } }]
    }"#;
    let res2 = parse_pz_clipboard_text(fx_grp_num_child, PzClipboardHint::Any).unwrap();
    assert!(matches!(res2.payload, PzClipboardPayload::Effects(_)));

    let fx_grp_hinted = r#"{
        "type": 0,
        "objects": [{ "type": 15, "properties": { "amount": 0.5 } }]
    }"#;
    let res3 = parse_pz_clipboard_text(fx_grp_hinted, PzClipboardHint::Effects).unwrap();
    assert!(matches!(res3.payload, PzClipboardPayload::Effects(_)));
}

#[test]
fn qa4_compact_cm3_keyframe_aliases_t_v_e() {
    let track_json = r#"{"keyframes":[{"t":0,"v":0,"e":1},{"t":30,"v":12,"e":3}]}"#;
    let res = parse_pz_clipboard_text(track_json, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::PropertyTrack { track } = res.payload else {
        panic!("expected PropertyTrack");
    };
    assert_eq!(track.keyframes.len(), 2);
    assert!((track.keyframes[0].frame - 0.0).abs() < 1e-9);
    assert_eq!(track.keyframes[0].value, PropertyValue::Number(0.0));
    assert!((track.keyframes[1].frame - 30.0).abs() < 1e-9);
    assert_eq!(track.keyframes[1].value, PropertyValue::Number(12.0));
    assert_eq!(track.keyframes[1].tween, 3);

    let raw_arr = r#"[{"f":5,"val":2.5,"e":0},{"f":20,"val":9.5,"e":1}]"#;
    let res2 = parse_pz_clipboard_text(raw_arr, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::PropertyTrack { track: track2 } = res2.payload else {
        panic!("expected PropertyTrack for compact raw array");
    };
    assert_eq!(track2.keyframes.len(), 2);
    assert!((track2.keyframes[0].frame - 5.0).abs() < 1e-9);
    assert_eq!(track2.keyframes[0].value, PropertyValue::Number(2.5));
    assert!((track2.keyframes[1].frame - 20.0).abs() < 1e-9);
    assert_eq!(track2.keyframes[1].value, PropertyValue::Number(9.5));
}

#[test]
fn qa4_shader_object_custom_properties_variants() {
    let shader_json = r#"{
        "type": 1,
        "properties": {
            "name": "Nested CustomProps Shader",
            "fragShader": "uniform sampler2D tDiffuse; uniform float Gain; uniform vec3 Tint; varying vec2 vUv; void main() { gl_FragColor = vec4(Tint * Gain, 1.0); }",
            "customProperties": [
                {
                    "name": "Gain",
                    "type": "float",
                    "properties": {
                        "value": {
                            "animated": true,
                            "keyframes": [
                                { "t": 0, "v": 0.4, "e": 1 },
                                { "t": 30, "v": 2.2, "e": 3 }
                            ]
                        }
                    }
                },
                {
                    "name": "Tint",
                    "type": 4,
                    "properties": {
                        "value": [0.2, 0.8, 1.0]
                    }
                }
            ]
        }
    }"#;
    let res = parse_pz_clipboard_text(shader_json, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Effects(fxs) = res.payload else {
        panic!("expected Effects");
    };
    assert_eq!(fxs.len(), 1);
    assert!(
        !fxs[0].params.contains_key("customProperties"),
        "properties.customProperties must not leak into fx.params"
    );
    let EffectKind::CustomShader {
        ref custom_properties,
        ..
    } = fxs[0].kind
    else {
        panic!("expected CustomShader");
    };
    assert_eq!(custom_properties.len(), 2);
    assert_eq!(custom_properties[0].name, "Gain");
    assert!(custom_properties[0].track.animated);
    assert_eq!(custom_properties[0].track.keyframes.len(), 2);
    assert!((custom_properties[0].track.keyframes[1].frame - 30.0).abs() < 1e-9);
    assert_eq!(
        custom_properties[0].track.keyframes[1].value,
        PropertyValue::Number(2.2)
    );
    assert_eq!(custom_properties[1].name, "Tint");
    assert_eq!(
        custom_properties[1].track.value,
        PropertyValue::Color([0.2, 0.8, 1.0])
    );
}

#[test]
fn qa4_3d_group_paste_hierarchy_and_transforms() {
    // Community variant: `{"type":"group","name":...,"position":...,"children":[...]}`
    let community_group = r#"{
        "type": "group",
        "name": "HeroRig",
        "position": { "keyframes": [{ "t": 0, "v": [0, 0, 0], "e": 1 }, { "t": 30, "v": [15, 5, -2], "e": 3 }] },
        "children": [
            { "type": "box", "name": "ChildCube" }
        ]
    }"#;
    let res = parse_pz_clipboard_text(community_group, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Objects3D(objs) = res.payload else {
        panic!("expected Objects3D");
    };
    assert_eq!(objs.len(), 1);
    assert_eq!(objs[0].name, "HeroRig");
    assert!(objs[0].position.animated);
    assert_eq!(objs[0].position.channels.len(), 3);
    assert!((objs[0].position.channels[0].keyframes[1].frame - 30.0).abs() < 1e-9);
    assert_eq!(
        objs[0].position.channels[0].keyframes[1].value,
        PropertyValue::Number(15.0)
    );
    let Object3DKind::Group { ref children, .. } = objs[0].kind else {
        panic!("expected Group parent, got {:?}", objs[0].kind);
    };
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].name, "ChildCube");
    assert!(matches!(
        children[0].kind,
        Object3DKind::Shape3D(Shape3DKind::Box { size }) if size == [10.0, 10.0, 10.0]
    ));

    // Native CM3 variant: `{"objectType":0,"properties":{...},"objects":[...]}`
    let native_group = r#"{
        "objectType": 0,
        "properties": { "name": "NativeRig", "position": [3, 4, 5] },
        "objects": [
            { "objectType": 1, "properties": { "name": "NativeBox" } }
        ]
    }"#;
    let res2 = parse_pz_clipboard_text(native_group, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Objects3D(objs2) = res2.payload else {
        panic!("expected Objects3D for native CM3 group");
    };
    assert_eq!(objs2.len(), 1);
    assert_eq!(objs2[0].name, "NativeRig");
    assert_eq!(
        objs2[0].position.value,
        PropertyValue::Vec3([3.0, 4.0, 5.0])
    );
    let Object3DKind::Group { ref children, .. } = objs2[0].kind else {
        panic!("expected native CM3 Group parent, got {:?}", objs2[0].kind);
    };
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].name, "NativeBox");
    assert!(matches!(
        children[0].kind,
        Object3DKind::Shape3D(Shape3DKind::Box { size }) if size == [10.0, 10.0, 10.0]
    ));
}

#[test]
fn qa4_clipboard_auto_router_fixes() {
    // 1. Material wrapper
    let mat_json =
        r#"{"type":"material","materialType":"pbr","color":[0.9,0.2,0.1],"roughness":0.3}"#;
    let res_mat = parse_pz_clipboard_text(mat_json, PzClipboardHint::Any).unwrap();
    assert!(matches!(res_mat.payload, PzClipboardPayload::Material(_)));

    // 2. Mask variants (`type: "mask"` and `kind: "path"` with `feather` + `vertices`)
    let mask_type = r#"{"type":"mask","name":"Cutout","mode":"subtract","feather":16,"vertices":[[0,0],[100,0],[100,100]]}"#;
    let res_m1 = parse_pz_clipboard_text(mask_type, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::LayerMasks(m1) = res_m1.payload else {
        panic!("expected LayerMasks for type:mask");
    };
    assert_eq!(m1.len(), 1);
    assert_eq!(m1[0].name, "Cutout");
    assert_eq!(m1[0].mode, MaskMode::Subtract);
    assert!((m1[0].feather - 16.0).abs() < 1e-9);

    let mask_path = r#"{"kind":"path","feather":12,"vertices":[[0,0],[50,0],[50,50]]}"#;
    let res_m2 = parse_pz_clipboard_text(mask_path, PzClipboardHint::Any).unwrap();
    assert!(matches!(res_m2.payload, PzClipboardPayload::LayerMasks(_)));

    // When hint == Masks, NEVER fall back to inserting an Effect!
    assert!(parse_pz_clipboard_text(r#"{"type":"bloom"}"#, PzClipboardHint::Masks).is_err());
    assert!(parse_pz_clipboard_text(r#"{}"#, PzClipboardHint::Masks).is_err());

    // 3. Clip wrapper
    let clip_json = r#"{"type":"clip","clipType":"video","name":"B-Roll","start":10,"length":90}"#;
    let res_clip = parse_pz_clipboard_text(clip_json, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Clips(clips) = res_clip.payload else {
        panic!("expected Clips for type:clip");
    };
    assert_eq!(clips.len(), 1);
    assert!((clips[0].start() - 10.0).abs() < 1e-9);
    assert!((clips[0].length() - 90.0).abs() < 1e-9);

    // 4. `{"type":"rectangle"}` with `Objects3D` hint -> 3D Rectangle, NOT Shape+
    let rect_3d = r#"{"type":"rectangle","name":"Floor","size":[40,20]}"#;
    let res_rect = parse_pz_clipboard_text(rect_3d, PzClipboardHint::Objects3D).unwrap();
    let PzClipboardPayload::Objects3D(objs) = res_rect.payload else {
        panic!("expected Objects3D for rectangle with Objects3D hint");
    };
    assert_eq!(objs.len(), 1);
    assert_eq!(objs[0].name, "Floor");
    assert!(matches!(
        objs[0].kind,
        Object3DKind::Shape3D(Shape3DKind::Rectangle { size }) if size == [40.0, 20.0]
    ));
}

#[test]
fn qa4_shape_plus_wrapper_and_property_fidelity() {
    let sp_json = r##"{
        "type": "shapePlus",
        "items": [
            {
                "type": "rectangle",
                "name": "Hero Card",
                "size": [320, 180],
                "position": [10, 20],
                "roundness": 24,
                "fill": { "color": "#3399ff", "opacity": 0.9 },
                "stroke": { "color": [1, 1, 1], "width": 4, "opacity": 0.8 }
            },
            {
                "kind": "polystar",
                "name": "Badge",
                "points": 6,
                "outerRadius": 120
            }
        ]
    }"##;
    let res = parse_pz_clipboard_text(sp_json, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::ShapePlusContents(doc) = res.payload else {
        panic!("expected ShapePlusContents");
    };
    let contents = doc["contents"].as_array().expect("contents array");
    assert_eq!(contents.len(), 2);

    // First item wrapped in group containing rectangle + fill + stroke
    let grp = &contents[0];
    assert_eq!(grp["kind"], "group");
    assert_eq!(grp["properties"]["name"], "Hero Card");
    let children = grp["contents"].as_array().expect("group children");
    assert_eq!(children.len(), 3);
    assert_eq!(children[0]["kind"], "rectangle");
    assert_eq!(children[0]["properties"]["size"], json!([320, 180]));
    assert_eq!(children[0]["properties"]["position"], json!([10, 20]));
    assert_eq!(children[0]["properties"]["roundness"], json!(24));
    assert_eq!(children[1]["kind"], "fill");
    assert_eq!(children[1]["properties"]["opacity"], json!(0.9));
    assert_eq!(children[2]["kind"], "stroke");
    assert_eq!(children[2]["properties"]["width"], json!(4));

    // Second item preserves polystar properties in `properties`
    let star = &contents[1];
    assert_eq!(star["kind"], "polystar");
    assert_eq!(star["properties"]["name"], "Badge");
    assert_eq!(star["properties"]["points"], json!(6));
    assert_eq!(star["properties"]["outerRadius"], json!(120));
}

#[test]
fn qa4_empty_object_control_char_sanitization_and_oversized_string_cap() {
    // Empty `{}` rejected for every hint
    for hint in [
        PzClipboardHint::Any,
        PzClipboardHint::Effects,
        PzClipboardHint::Objects3D,
        PzClipboardHint::Masks,
        PzClipboardHint::Clips,
        PzClipboardHint::Material,
    ] {
        assert!(
            parse_pz_clipboard_text("{}", hint).is_err(),
            "empty object must be rejected for hint {hint:?}"
        );
    }

    // Control chars `\r`, `\n`, `\0` stripped from display names
    let dirty_name_fx = r#"{"type":"bloom","properties":{"name":"Evil\r\n\u0000Bloom"}}"#;
    let PzClipboardPayload::Effects(fxs) =
        parse_pz_clipboard_text(dirty_name_fx, PzClipboardHint::Any)
            .unwrap()
            .payload
    else {
        panic!("expected Effects");
    };
    assert_eq!(fxs[0].name, "EvilBloom");

    let dirty_name_obj = r#"{"type":"box","name":"Bad\n\r\u0000Cube"}"#;
    let PzClipboardPayload::Objects3D(objs) =
        parse_pz_clipboard_text(dirty_name_obj, PzClipboardHint::Any)
            .unwrap()
            .payload
    else {
        panic!("expected Objects3D");
    };
    assert_eq!(objs[0].name, "BadCube");

    // Oversized junk string in non-shader JSON property rejected
    let junk = "A".repeat(128 * 1024);
    let hostile_json = format!(r#"{{"type":"bloom","properties":{{"junk":"{junk}"}}}}"#);
    assert!(parse_pz_clipboard_text(&hostile_json, PzClipboardHint::Any).is_err());

    // Large legitimate GLSL shader (> 64 KB) preserved cleanly
    let pad_comment = "// ".to_owned() + &"x".repeat(90 * 1024) + "\n";
    let big_glsl = format!(
        "{pad_comment}uniform sampler2D tDiffuse;\nvarying vec2 vUv;\nvoid main() {{ gl_FragColor = texture2D(tDiffuse, vUv); }}"
    );
    let shader_json = serde_json::to_string(&json!({
        "type": 1,
        "properties": { "name": "Big Shader", "fragShader": big_glsl }
    }))
    .unwrap();
    let res = parse_pz_clipboard_text(&shader_json, PzClipboardHint::Any).unwrap();
    assert!(matches!(res.payload, PzClipboardPayload::Effects(_)));
}

#[test]
fn qa6_cm3_corpus_shader_type3_color_phong_normal_text3d_particles_handles_vignette_ca_and_2d() {
    // 1. PanZoid Shader `customProperties` `type: 3` RGB `[r,g,b]` & `{r,g,b}` -> PropertyValue::Color
    //    AND copy -> paste round-trips as PropertyValue::Color (NOT Vec4([0,0,0,0]) or Vec3).
    let shader_t3 = r#"{
        "type": 1,
        "properties": {
            "name": "Type3ColorShader",
            "fragShader": "uniform sampler2D tDiffuse; uniform vec3 ColorA; uniform vec3 ColorB; void main() { gl_FragColor = vec4(ColorA * ColorB, 1.0); }"
        },
        "customProperties": [
            { "name": "ColorA", "type": 3, "value": [0.25, 0.5, 0.75] },
            { "name": "ColorB", "type": 3, "value": { "r": 0.9, "g": 0.2, "b": 0.1 } }
        ]
    }"#;
    let parsed = parse_pz_clipboard_text(shader_t3, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Effects(fxs) = parsed.payload else {
        panic!("expected Effects");
    };
    let EffectKind::CustomShader {
        ref custom_properties,
        ..
    } = fxs[0].kind
    else {
        panic!("expected CustomShader");
    };
    assert_eq!(
        custom_properties[0].track.value,
        PropertyValue::Color([0.25, 0.5, 0.75])
    );
    assert_eq!(
        custom_properties[1].track.value,
        PropertyValue::Color([0.9, 0.2, 0.1])
    );

    let copied = serialize_pz_effects_json(&fxs, &Sequence::default());
    let repasted = parse_pz_clipboard_text(&copied, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Effects(fxs2) = repasted.payload else {
        panic!("expected Effects after round-trip");
    };
    let EffectKind::CustomShader {
        custom_properties: ref cps2,
        ..
    } = fxs2[0].kind
    else {
        panic!("expected CustomShader after round-trip");
    };
    assert_eq!(cps2[0].track.value, PropertyValue::Color([0.25, 0.5, 0.75]));
    assert_eq!(cps2[1].track.value, PropertyValue::Color([0.9, 0.2, 0.1]));

    // 2. Per-axis single-channel `objects` scalar track (AnimatedBloom.strength)
    let anim_bloom = r#"{
        "type": "bloom",
        "properties": {
            "name": "AnimatedBloom",
            "strength": {
                "objects": [
                    {
                        "animated": true,
                        "keyframes": [
                            { "frame": 0, "value": 0.80, "tween": 1 },
                            { "frame": 30, "value": 2.40, "tween": 1 }
                        ]
                    }
                ]
            }
        }
    }"#;
    let PzClipboardPayload::Effects(bloom_fxs) =
        parse_pz_clipboard_text(anim_bloom, PzClipboardHint::Any)
            .unwrap()
            .payload
    else {
        panic!("expected Effects");
    };
    let strength = &bloom_fxs[0].params["strength"];
    assert!(strength.animated);
    assert!(strength.channels.is_empty());
    assert_eq!(strength.keyframes.len(), 2);
    assert_eq!(strength.value, PropertyValue::Number(0.80));

    // 3. CM3 `phong` & `normal` materials classify as Material (NEVER Effects!)
    let phong_json = r#"{"type":"phong","properties":{"color":[0.2,0.4,0.8],"specular":[1.0,1.0,1.0],"shininess":64}}"#;
    let phong_res = parse_pz_clipboard_text(phong_json, PzClipboardHint::Any).unwrap();
    let PzClipboardPayload::Material(MaterialSpec::CustomPbr(pm)) = phong_res.payload else {
        panic!("expected phong Material, never Effects");
    };
    assert_eq!(pm.color.value, PropertyValue::Color([0.2, 0.4, 0.8]));
    assert!(pm.roughness.value.as_f64().unwrap() < 0.25);

    let normal_json = r#"{"type":"normal","properties":{"side":2,"opacity":0.8}}"#;
    let normal_res = parse_pz_clipboard_text(normal_json, PzClipboardHint::Any).unwrap();
    assert!(matches!(
        normal_res.payload,
        PzClipboardPayload::Material(MaterialSpec::CustomPbr(_))
    ));

    // 4. Text3D with frontMaterial/sideMaterial, scalar size+thickness, scalar bevelSize+bevelThickness, curveSegments
    let text3d_json = r#"{
        "type": 1,
        "properties": {
            "name": "Hero3D",
            "text": "ZOID",
            "size": 14.0,
            "thickness": 3.5,
            "bevel": 1,
            "bevelSize": 0.6,
            "bevelThickness": 0.9,
            "curveSegments": 12
        },
        "frontMaterial": { "type": "singlecolor", "properties": { "color": [1.0, 0.0, 0.0] } },
        "sideMaterial": { "type": "phong", "properties": { "color": [0.1, 0.1, 0.1], "shininess": 50 } }
    }"#;
    let PzClipboardPayload::Objects3D(objs) =
        parse_pz_clipboard_text(text3d_json, PzClipboardHint::Objects3D)
            .unwrap()
            .payload
    else {
        panic!("expected Objects3D");
    };
    let Object3DKind::Text3D(ref td) = objs[0].kind else {
        panic!("expected Text3D");
    };
    assert_eq!(td.size, [14.0, 3.5]);
    assert_eq!(td.detail, 12);
    assert!(td.bevel);
    assert!((td.bevel_size - 0.6).abs() < 1e-9);
    assert!((td.bevel_thickness - 0.9).abs() < 1e-9);
    assert!(objs[0].material_slots.contains_key("front"));
    assert!(objs[0].material_slots.contains_key("side"));

    // 5. Legacy & Particles+ aliases (velocity, velocitySpread, gravity, scalar size, RGB color, turbulence/drag)
    let particles_json = r#"{
        "type": 4,
        "properties": {
            "name": "Sparks",
            "number": 250,
            "velocity": [1.0, 5.0, 0.0],
            "velocitySpread": [2.0, 1.0, 2.0],
            "gravity": [0.0, -9.8, 0.0],
            "size": 2.5,
            "color": [1.0, 0.5, 0.0],
            "turbulence": 3.2,
            "drag": 0.4
        }
    }"#;
    let p_res = parse_pz_clipboard_text(particles_json, PzClipboardHint::Objects3D).unwrap();
    assert!(p_res.required_plugins.contains("particles-plus"));
    let PzClipboardPayload::Objects3D(p_objs) = p_res.payload else {
        panic!("expected Objects3D");
    };
    let Object3DKind::Particles(ref pd) = p_objs[0].kind else {
        panic!("expected Particles");
    };
    assert_eq!(pd.number, 250);
    assert_eq!(pd.ivel, [1.0, 5.0, 0.0]);
    assert_eq!(pd.vspread, [2.0, 1.0, 2.0]);
    assert_eq!(pd.accel, [0.0, -9.8, 0.0]);
    assert_eq!(pd.size[0].color, "2.5");
    assert!(pd.color[0].color.starts_with("rgba(255,128,0,"));
    assert_eq!(
        p_objs[0].preserved.get("property.turbulence"),
        Some(&json!(3.2))
    );
    assert_eq!(p_objs[0].preserved.get("property.drag"), Some(&json!(0.4)));

    // 6. bit-8 keyframe custom `handles` preserved (never replaced with [[-10, 0], [10, 0]])
    let track_json = r#"{
        "animated": true,
        "keyframes": [
            { "frame": 0, "value": 0.0, "tween": 256, "handles": [[-4.5, 1.25], [7.5, -2.5]] },
            { "frame": 30, "value": 100.0, "tween": 257, "handles": [[-6.0, 0.5], [6.0, -0.5]] }
        ]
    }"#;
    let PzClipboardPayload::PropertyTrack { track: tr } =
        parse_pz_clipboard_text(track_json, PzClipboardHint::Keyframes)
            .unwrap()
            .payload
    else {
        panic!("expected PropertyTrack");
    };
    assert_eq!(tr.keyframes[0].tween, 256);
    assert_eq!(tr.keyframes[0].control_points, [[-4.5, 1.25], [7.5, -2.5]]);
    assert_eq!(tr.keyframes[1].tween, 257);
    assert!(tr.keyframes[1].is_bezier);
    assert_eq!(tr.keyframes[1].control_points, [[-6.0, 0.5], [6.0, -0.5]]);

    // 7. `vignette` preserves `darkness` and mirrors into `vignetteStrength`; `chromaticaberration` -> `rgbshift` with 0 warnings
    let vig_ca = r#"[
        { "type": "vignette", "properties": { "darkness": 0.82 } },
        { "type": "chromaticaberration", "properties": { "amount": 14.0 } }
    ]"#;
    let vig_res = parse_pz_clipboard_text(vig_ca, PzClipboardHint::Effects).unwrap();
    assert!(
        vig_res.warnings.is_empty(),
        "vignette + chromaticaberration must produce 0 warnings, got {:?}",
        vig_res.warnings
    );
    let PzClipboardPayload::Effects(v_fxs) = vig_res.payload else {
        panic!("expected Effects");
    };
    assert_eq!(
        v_fxs[0].params["darkness"].value,
        PropertyValue::Number(0.82)
    );
    assert_eq!(
        v_fxs[0].params["vignetteStrength"].value,
        PropertyValue::Number(0.82)
    );
    assert_eq!(v_fxs[1].kind, EffectKind::Standard("rgbshift".to_owned()));
    assert_eq!(v_fxs[1].params["amount"].value, PropertyValue::Number(14.0));

    // 8. Text2D (fontSize/color), Shape2D Rect (type: 1), Generator (type: 5 childProperties.color)
    let clips_2d = r#"[
        {
            "start": 0, "length": 60,
            "object": {
                "type": 7,
                "properties": { "name": "Title2D", "childProperties": { "text": "HI", "fontSize": 72, "color": [1.0, 0.2, 0.3] } }
            }
        },
        {
            "start": 0, "length": 60,
            "object": {
                "type": 6,
                "properties": { "name": "Rect2D" },
                "objects": [{ "type": 1, "properties": { "name": "Box", "size": [300, 120], "color": [0.1, 0.8, 0.2] } }]
            }
        },
        {
            "start": 0, "length": 60,
            "object": {
                "type": 5,
                "properties": { "name": "SolidGen", "childProperties": { "color": [0.3, 0.4, 0.9] } }
            }
        }
    ]"#;
    let PzClipboardPayload::Clips(clips) =
        parse_pz_clipboard_text(clips_2d, PzClipboardHint::Clips)
            .unwrap()
            .payload
    else {
        panic!("expected Clips");
    };
    let PzClipboardClip::Video { clip: t2d, .. } = &clips[0] else {
        panic!("expected Text2D clip");
    };
    let LayerKind::Text2D {
        size, ref shapes, ..
    } = t2d.layer.kind
    else {
        panic!("expected Text2D");
    };
    assert_eq!(size, 72);
    assert_eq!(shapes.len(), 1);
    assert!(matches!(
        shapes[0].operations.first(),
        Some(crate::DrawOp::Fill { color }) if color.value == PropertyValue::Color([1.0, 0.2, 0.3])
    ));

    let PzClipboardClip::Video { clip: s2d, .. } = &clips[1] else {
        panic!("expected Shape2D clip");
    };
    let LayerKind::Shape2D {
        shapes: ref s_nodes,
    } = s2d.layer.kind
    else {
        panic!("expected Shape2D");
    };
    let crate::Shape2DKind::Path { ref path } = s_nodes[0].kind else {
        panic!("expected Shape2DKind::Path");
    };
    assert_eq!(path["commands"].as_array().unwrap().len(), 5);
    assert!(matches!(
        s_nodes[0].operations.first(),
        Some(crate::DrawOp::Fill { color }) if color.value == PropertyValue::Color([0.1, 0.8, 0.2])
    ));

    let PzClipboardClip::Video { clip: gen_clip, .. } = &clips[2] else {
        panic!("expected Generator clip");
    };
    assert_eq!(gen_clip.layer.kind, LayerKind::Generator);
    assert_eq!(
        gen_clip.layer.preserved.get("childProperty.color"),
        Some(&json!([0.3, 0.4, 0.9]))
    );
}
