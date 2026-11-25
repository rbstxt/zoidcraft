use super::super::effects::export_zoidium_effect_id;
use super::super::sequence::parse_valid_control_id;
use super::*;
use crate::model::{
    AssetKind, CameraData, CustomProperty, EffectInstance, EffectKind, Layer, LayerKind, LayerMask,
    MaskMode, MaterialSpec, MediaItem, Object3D, Object3DKind, Project, PropertyKeyframe,
    PropertyTrack, PropertyValue, Sequence, SequenceControl, SequenceControlKind, Shape2DKind,
    Shape2DNode, Shape3DKind, VideoClip,
};
use serde_json::{Value, json};
use uuid::Uuid;

#[test]
fn enforces_all_seven_cm3_and_zoidium_export_schema_invariants() {
    use crate::effects::ZOIDIUM_PLUGIN_EFFECT_IDS;
    use crate::model::ImagePlusMaterial;

    let mut project = Project::new_cm3_default();
    project.name = "Schema Invariants".to_owned();

    let tex_bytes = b"\x89PNG\r\n\x1a\nINVARIANT_TEX".to_vec();
    let tex_id = project.insert_binary_asset(AssetKind::Image, "albedo.png", tex_bytes.clone());
    let tex_cm3_key = cm3_asset_key(&tex_bytes);

    // 1. Media item WITHOUT asset_id + Precomp composition entry (both must emit `"assets": []`).
    project.media.push(MediaItem {
        id: Uuid::new_v4(),
        name: "Unbound Media".to_owned(),
        asset_id: None,
        kind: AssetKind::Misc,
        preserved: BTreeMap::new(),
    });
    let comp_uuid = Uuid::new_v4();
    let comp_seq = Sequence {
        id: comp_uuid,
        name: "Empty Comp".to_owned(),
        ..Sequence::default()
    };
    project.compositions.insert(comp_uuid, comp_seq);
    project
        .composition_ids
        .insert("comp-empty".to_owned(), comp_uuid);

    // 7. Sequence controls with invalid, session (`ctl_*`), duplicate IDs and messy/duplicate/blank names.
    project.sequence.controls = vec![
        SequenceControl {
            id: "c2".to_owned(),
            name: "  Speed  ".to_owned(),
            kind: SequenceControlKind::Slider,
            track: PropertyTrack::number(10.0),
            preserved: BTreeMap::new(),
        },
        SequenceControl {
            id: "ctl_session_1".to_owned(),
            name: "Speed".to_owned(),
            kind: SequenceControlKind::Angle,
            track: PropertyTrack::number(45.0),
            preserved: BTreeMap::new(),
        },
        SequenceControl {
            id: "c2".to_owned(),
            name: "   ".to_owned(),
            kind: SequenceControlKind::Checkbox,
            track: PropertyTrack::option(1),
            preserved: BTreeMap::new(),
        },
        SequenceControl {
            id: "c+5".to_owned(),
            name: "Speed".to_owned(),
            kind: SequenceControlKind::Point2D,
            track: PropertyTrack::vec2([1.0, 2.0]),
            preserved: BTreeMap::new(),
        },
        SequenceControl {
            id: "bad_id".to_owned(),
            name: "Checkbox".to_owned(),
            kind: SequenceControlKind::Color,
            track: PropertyTrack::color([0.2, 0.4, 0.6]),
            preserved: BTreeMap::new(),
        },
    ];

    // 4. Material+ objects (PbrPlus, Matcap, ImagePlus, NodeMaterial with legacy prefixed graph.type).
    let mut pbr_box = Object3D::new_shape_3d(
        "PbrBox",
        Shape3DKind::Box {
            size: [10.0, 10.0, 10.0],
        },
    );
    pbr_box.material = Some(MaterialSpec::PbrPlus(Box::new(PbrPlusMaterial {
        base_color: PropertyTrack::color([0.8, 0.4, 0.2]),
        base_color_map: Some(tex_id.clone()),
        normal_scale: 1.5,
        reflection_intensity: 0.75,
        ..PbrPlusMaterial::default()
    })));

    let mut matcap_box = Object3D::new_shape_3d(
        "MatcapBox",
        Shape3DKind::Box {
            size: [10.0, 10.0, 10.0],
        },
    );
    matcap_box.material = Some(MaterialSpec::Matcap(Box::new(MatcapMaterial {
        texture: Some(tex_id.clone()),
        ..MatcapMaterial::default()
    })));

    let mut imgplus_box = Object3D::new_shape_3d(
        "ImagePlusBox",
        Shape3DKind::Box {
            size: [10.0, 10.0, 10.0],
        },
    );
    imgplus_box.material = Some(MaterialSpec::ImagePlus(Box::new(ImagePlusMaterial {
        texture: Some(tex_id.clone()),
        opacity: PropertyTrack::number(0.6),
        transparent: true,
        ..ImagePlusMaterial::default()
    })));

    let mut nodes_box = Object3D::new_shape_3d(
        "NodesBox",
        Shape3DKind::Box {
            size: [10.0, 10.0, 10.0],
        },
    );
    nodes_box.material = Some(MaterialSpec::NodeMaterial {
        schema_version: 1,
        resolution: 512,
        graph: json!({
            "type": "zoidium:material-plus/nodes",
            "properties": { "graph": "{}" }
        }),
    });

    let mut scene_layer = Layer::new_scene_3d(
        [1920, 1080],
        vec![pbr_box, matcap_box, imgplus_box, nodes_box],
    );

    // 3. Custom shader with dynamic scalar, dynamic Vec2, dynamic Color, static Asset, and static Option.
    let mut custom_fx = create_custom_shader("UniformsCheck", "void main() {}");
    if let EffectKind::CustomShader {
        ref mut custom_properties,
        ..
    } = custom_fx.kind
    {
        custom_properties.push(CustomProperty {
            name: "uGain".to_owned(),
            prop_type: Some(0),
            track: PropertyTrack::number(2.5),
            metadata: BTreeMap::new(),
        });
        custom_properties.push(CustomProperty {
            name: "uOffset".to_owned(),
            prop_type: Some(1),
            track: PropertyTrack::vec2([0.25, -0.5]),
            metadata: BTreeMap::new(),
        });
        custom_properties.push(CustomProperty {
            name: "uTint".to_owned(),
            prop_type: Some(4),
            track: PropertyTrack::color([0.3, 0.6, 0.9]),
            metadata: BTreeMap::new(),
        });
        custom_properties.push(CustomProperty {
            name: "uTex".to_owned(),
            prop_type: Some(8),
            track: PropertyTrack::asset(Some(tex_id.clone())),
            metadata: BTreeMap::new(),
        });
        custom_properties.push(CustomProperty {
            name: "uMode".to_owned(),
            prop_type: Some(6),
            track: PropertyTrack::option(2),
            metadata: BTreeMap::new(),
        });
    }

    // 5. All 36 Zoidium plugin effects (35 nativeEffects + nodeeffect) + a standard effect with static fields (`coloroverlay`).
    let mut effects = vec![custom_fx];
    for &id in &ZOIDIUM_PLUGIN_EFFECT_IDS {
        effects.push(create_zoidium_effect(id).expect("zoidium effect"));
    }
    effects.push(create_standard_effect("coloroverlay").expect("coloroverlay"));
    scene_layer.effects = effects;

    // 6. Enabled and disabled zoidiumMasks with unanimated path tracks.
    project.sequence.layer_masks.insert(
        scene_layer.id,
        vec![
            LayerMask {
                id: "m-on".to_owned(),
                name: "Mask On".to_owned(),
                enabled: true,
                mode: MaskMode::Add,
                invert: false,
                feather: 4.0,
                expansion: 0.0,
                opacity: 1.0,
                closed: true,
                locked: false,
                outline_color: String::new(),
                vertex_ids: vec!["v1".to_owned()],
                vertex_modes: BTreeMap::new(),
                path: PropertyTrack::constant(PropertyValue::Json(json!([
                    10.0, 20.0, 0.0, 0.0, 0.0, 0.0
                ]))),
                preserved: BTreeMap::new(),
            },
            LayerMask {
                id: "m-off".to_owned(),
                name: "Mask Off".to_owned(),
                enabled: false,
                mode: MaskMode::Subtract,
                invert: true,
                feather: 0.0,
                expansion: 0.0,
                opacity: 0.5,
                closed: false,
                locked: true,
                outline_color: "#f0883e".to_owned(),
                vertex_ids: vec!["v1".to_owned()],
                vertex_modes: BTreeMap::new(),
                path: PropertyTrack::constant(PropertyValue::Json(json!([
                    5.0, 6.0, 0.0, 0.0, 0.0, 0.0
                ]))),
                preserved: BTreeMap::new(),
            },
        ],
    );

    // 1. Shape2D layer with ZERO effects and a Shape2DNode with ZERO operations.
    let empty_shape_layer = Layer::new(
        "Empty Effects & Ops",
        LayerKind::Shape2D {
            shapes: vec![Shape2DNode {
                id: Uuid::new_v4(),
                name: "BareGroup".to_owned(),
                kind: Shape2DKind::Group {
                    children: Vec::new(),
                },
                position: PropertyTrack::vec2([3.0, 4.0]),
                rotation: PropertyTrack::number(0.0),
                scale: PropertyTrack::vec2([1.0, 1.0]),
                operations: Vec::new(),
                preserved: BTreeMap::new(),
            }],
        },
        [1920, 1080],
    );

    project.sequence.video_tracks[0].clips = vec![
        VideoClip {
            id: Uuid::new_v4(),
            name: "Scene Clip".to_owned(),
            start: 0.0,
            length: 120.0,
            link: None,
            media: None,
            time_curve: PropertyTrack::linear_time_curve(120.0, 30),
            layer: scene_layer,
            preserved: BTreeMap::new(),
        },
        VideoClip {
            id: Uuid::new_v4(),
            name: "Shape Clip".to_owned(),
            start: 0.0,
            length: 120.0,
            link: None,
            media: None,
            time_curve: PropertyTrack::linear_time_curve(120.0, 30),
            layer: empty_shape_layer,
            preserved: BTreeMap::new(),
        },
    ];

    // Export and inspect raw `project` JSON directly for ALL 7 invariants.
    let pz_bytes = export_pz_to_bytes(&project).expect("export_pz_to_bytes");
    let entries = inspect_tar_entries(&pz_bytes);
    let doc: Value = serde_json::from_slice(&entries["project"]).expect("parse project JSON");

    // --- Invariant 1: Required CM3 arrays always emitted even when empty ---
    let shape_layer_json = &doc["sequence"]["videoTracks"][0]["clips"][1]["object"];
    assert_eq!(shape_layer_json["effects"], json!([]));
    assert_eq!(shape_layer_json["objects"][0]["operations"], json!([]));
    let media_arr = doc["media"].as_array().expect("media array");
    assert_eq!(media_arr.len(), 2);
    for m in media_arr {
        assert!(
            m.get("assets").is_some_and(Value::is_array),
            "media entry missing `assets` array: {m}"
        );
        assert_eq!(m["assets"], json!([]));
    }

    // --- Invariant 2: Dynamic properties always wrapped as keyframe/group objects ---
    let scene_json = &doc["sequence"]["videoTracks"][0]["clips"][0]["object"];
    // Scalar dynamic track -> { animated: false, keyframes: [{ frame: 0, value, tween: 1 }] }
    assert_eq!(
        scene_json["properties"]["opacity"],
        json!({
            "animated": false,
            "keyframes": [{ "frame": 0, "value": 1.0, "tween": 1 }]
        })
    );
    // Vec2 dynamic track -> { objects: [{ animated: false, keyframes: [...] }, ...] }
    assert_eq!(
        shape_layer_json["objects"][0]["properties"]["position"],
        json!({
            "objects": [
                { "animated": false, "keyframes": [{ "frame": 0, "value": 3.0, "tween": 1 }] },
                { "animated": false, "keyframes": [{ "frame": 0, "value": 4.0, "tween": 1 }] }
            ]
        })
    );
    // Vec3 dynamic track on Object3D -> { objects: [...] }
    let pbr_obj_json = &scene_json["objects"][0];
    assert_eq!(
        pbr_obj_json["properties"]["scale"]["objects"]
            .as_array()
            .map(Vec::len),
        Some(3)
    );
    // Color dynamic track -> { animated: false, keyframes: [{ frame: 0, value: [...], tween: 1 }] }
    assert_eq!(
        pbr_obj_json["material"]["properties"]["color"],
        json!({
            "animated": false,
            "keyframes": [{ "frame": 0, "value": [0.8, 0.4, 0.2], "tween": 1 }]
        })
    );
    // PbrPlus dynamic normalScale & reflectionIntensity are wrapped (not bare numbers).
    assert_eq!(
        pbr_obj_json["material"]["properties"]["normalScale"],
        json!({
            "animated": false,
            "keyframes": [{ "frame": 0, "value": 1.5, "tween": 1 }]
        })
    );
    assert_eq!(
        pbr_obj_json["material"]["properties"]["reflectionIntensity"],
        json!({
            "animated": false,
            "keyframes": [{ "frame": 0, "value": 0.75, "tween": 1 }]
        })
    );
    // Static-only List field (`coloroverlay.blending`) stays a bare string.
    let coloroverlay_json = scene_json["effects"]
        .as_array()
        .unwrap()
        .last()
        .expect("coloroverlay");
    assert_eq!(
        coloroverlay_json["properties"]["blending"],
        json!("BLEND_SRC_OVER")
    );

    // --- Invariant 3: Custom properties have CM3 `type` definition object & dynamic merge ---
    let cps = scene_json["effects"][0]["customProperties"]
        .as_array()
        .expect("customProperties");
    assert_eq!(cps.len(), 5);
    // uGain (dynamic scalar)
    assert!(cps[0]["type"].is_object());
    assert_eq!(cps[0]["type"]["custom"], json!(true));
    assert_eq!(cps[0]["type"]["dynamic"], json!(true));
    assert_eq!(cps[0]["type"]["type"], json!(0));
    assert_eq!(cps[0]["properties"]["name"], json!("uGain"));
    assert!(cps[0].get("value").is_none());
    assert_eq!(
        cps[0]["keyframes"],
        json!([{ "frame": 0, "value": 2.5, "tween": 1 }])
    );
    // uOffset (dynamic Vec2 group)
    assert_eq!(cps[1]["type"]["custom"], json!(true));
    assert_eq!(cps[1]["type"]["dynamic"], json!(true));
    assert_eq!(cps[1]["type"]["type"], json!(1));
    assert_eq!(cps[1]["type"]["group"], json!(true));
    assert!(cps[1].get("value").is_none());
    assert_eq!(cps[1]["objects"].as_array().map(Vec::len), Some(2));
    // uTint (dynamic Color group)
    assert_eq!(cps[2]["type"]["custom"], json!(true));
    assert_eq!(cps[2]["type"]["dynamic"], json!(true));
    assert_eq!(cps[2]["type"]["type"], json!(4));
    assert_eq!(cps[2]["type"]["group"], json!(true));
    assert!(cps[2].get("value").is_none());
    assert_eq!(cps[2]["objects"].as_array().map(Vec::len), Some(3));
    // uTex (static Asset type 8)
    assert_eq!(cps[3]["type"]["custom"], json!(true));
    assert_eq!(cps[3]["type"]["dynamic"], json!(false));
    assert_eq!(cps[3]["type"]["type"], json!(8));
    assert_eq!(cps[3]["properties"]["name"], json!("uTex"));
    assert_eq!(cps[3]["value"], json!(tex_cm3_key));
    assert!(cps[3].get("keyframes").is_none());
    // uMode (static Option type 6)
    assert_eq!(cps[4]["type"]["custom"], json!(true));
    assert_eq!(cps[4]["type"]["dynamic"], json!(false));
    assert_eq!(cps[4]["type"]["type"], json!(6));
    assert_eq!(cps[4]["value"], json!(2));

    // --- Invariant 4: Material+ bare IDs & PbrPlus `color`/`texture` property names ---
    assert_eq!(
        scene_json["objects"][0]["material"]["type"],
        json!("pbrplus")
    );
    assert_eq!(
        scene_json["objects"][0]["material"]["properties"]["texture"],
        json!(tex_cm3_key)
    );
    assert!(
        scene_json["objects"][0]["material"]["properties"]
            .get("baseColor")
            .is_none()
    );
    assert!(
        scene_json["objects"][0]["material"]["properties"]
            .get("baseColorMap")
            .is_none()
    );
    assert_eq!(
        scene_json["objects"][1]["material"]["type"],
        json!("matcap")
    );
    assert_eq!(
        scene_json["objects"][2]["material"]["type"],
        json!("imageplus")
    );
    assert_eq!(scene_json["objects"][3]["material"]["type"], json!("nodes"));

    // --- Invariant 5: Native Zoidium plugin effects exported as exact bare manifest IDs ---
    let fx_arr = scene_json["effects"].as_array().expect("effects");
    for (i, &internal_id) in ZOIDIUM_PLUGIN_EFFECT_IDS.iter().enumerate() {
        let expected_id = export_zoidium_effect_id(internal_id);
        assert_eq!(
            fx_arr[i + 1]["type"],
            json!(expected_id),
            "expected bare native Zoidium manifest ID {expected_id} for {internal_id}"
        );
    }

    // --- Invariant 6: `zoidiumMasks` `validateMasks()` compliance ---
    let masks_json = scene_json["zoidiumMasks"].as_array().expect("zoidiumMasks");
    assert_eq!(masks_json.len(), 2);
    assert_eq!(masks_json[0]["enabled"], json!(1));
    assert_eq!(masks_json[1]["enabled"], json!(0));
    assert_eq!(
        masks_json[0]["properties"]["path"],
        json!({
            "animated": false,
            "keyframes": [{ "frame": 0, "value": [10.0, 20.0, 0.0, 0.0, 0.0, 0.0], "tween": 1 }]
        })
    );
    assert_eq!(
        masks_json[1]["properties"]["path"],
        json!({
            "animated": false,
            "keyframes": [{ "frame": 0, "value": [5.0, 6.0, 0.0, 0.0, 0.0, 0.0], "tween": 1 }]
        })
    );

    // --- Invariant 7: `zoidiumControls` ID regex & unique trimmed name normalization ---
    let ctl_payload_str = doc["sequence"]["properties"]["zoidiumControls"]
        .as_str()
        .expect("zoidiumControls string");
    let ctl_payload: Value = serde_json::from_str(ctl_payload_str).expect("parse zoidiumControls");
    let ctl_arr = ctl_payload["controls"].as_array().expect("controls array");
    assert_eq!(ctl_arr.len(), 5);

    let mut seen_ids = BTreeSet::new();
    let mut seen_names = BTreeSet::new();
    let mut max_id_num = 0u64;
    for c in ctl_arr {
        let id = c["type"]["zoidiumControlId"]
            .as_str()
            .expect("zoidiumControlId");
        let num = parse_valid_control_id(id)
            .unwrap_or_else(|| panic!("invalid control id {id:?}, must match /^c\\d+$/"));
        max_id_num = max_id_num.max(num);
        assert!(seen_ids.insert(id.to_owned()), "duplicate control id {id}");

        let name = c["properties"]["name"].as_str().expect("control name");
        assert!(!name.is_empty() && name == name.trim(), "bad name {name:?}");
        assert!(
            seen_names.insert(name.to_owned()),
            "duplicate control name {name:?}"
        );
    }
    assert_eq!(ctl_arr[0]["type"]["zoidiumControlId"], json!("c2"));
    assert_eq!(ctl_arr[0]["properties"]["name"], json!("Speed"));
    assert_eq!(ctl_arr[1]["properties"]["name"], json!("Speed 2"));
    assert_eq!(ctl_arr[2]["properties"]["name"], json!("Checkbox"));
    assert_eq!(ctl_arr[3]["properties"]["name"], json!("Speed 3"));
    assert_eq!(ctl_arr[4]["properties"]["name"], json!("Checkbox 2"));
    assert!(
        ctl_payload["nextId"].as_u64().expect("nextId") > max_id_num,
        "nextId must exceed max assigned control number"
    );

    // --- Full round-trip via `import_pz_from_bytes` ---
    // All 36 Zoidium plugin effects (including manifest IDs `"looks"` and `"sglow-hq-v2-preview"`)
    // must re-import with ZERO warnings.
    let reimported = import_pz_from_bytes(&pz_bytes).expect("reimport invariants pz");
    assert!(
        reimported.warnings.is_empty(),
        "unexpected warnings on round-trip: {:?}",
        reimported.warnings
    );
    let rp = reimported.project;

    // Controls round-trip
    assert_eq!(rp.sequence.controls.len(), 5);
    assert_eq!(rp.sequence.controls[0].id, "c2");
    assert_eq!(rp.sequence.controls[0].name, "Speed");
    assert_eq!(rp.sequence.controls[2].name, "Checkbox");
    assert_eq!(rp.sequence.controls[4].name, "Checkbox 2");

    // Masks round-trip
    let rscene = &rp.sequence.video_tracks[0].clips[0].layer;
    let rmasks = &rp.sequence.layer_masks[&rscene.id];
    assert_eq!(rmasks.len(), 2);
    assert!(rmasks[0].enabled);
    assert!(!rmasks[1].enabled);
    assert_eq!(
        rmasks[0].path.value,
        PropertyValue::Json(json!([10.0, 20.0, 0.0, 0.0, 0.0, 0.0]))
    );

    // Custom properties round-trip
    let EffectKind::CustomShader {
        ref custom_properties,
        ..
    } = rscene.effects[0].kind
    else {
        panic!("expected CustomShader");
    };
    assert_eq!(custom_properties.len(), 5);
    assert_eq!(custom_properties[0].track.value, PropertyValue::Number(2.5));
    assert_eq!(
        custom_properties[1].track.value,
        PropertyValue::Vec2([0.25, -0.5])
    );
    assert_eq!(
        custom_properties[2].track.value,
        PropertyValue::Color([0.3, 0.6, 0.9])
    );
    assert_eq!(
        custom_properties[3].track.value,
        PropertyValue::Asset(Some(tex_id.clone()))
    );
    assert_eq!(custom_properties[4].track.value, PropertyValue::Option(2));

    // All 16 native Zoidium plugin effects round-trip back to their canonical `EffectKind::Plugin`
    for (i, &internal_id) in ZOIDIUM_PLUGIN_EFFECT_IDS.iter().enumerate() {
        assert!(
            matches!(
                &rscene.effects[i + 1].kind,
                EffectKind::Plugin { effect_id, .. } if effect_id == internal_id
            ),
            "expected EffectKind::Plugin for {internal_id}, got {:?}",
            rscene.effects[i + 1].kind
        );
    }

    // Material+ round-trip
    let LayerKind::Scene3D { ref objects, .. } = rscene.kind else {
        panic!("expected Scene3D");
    };
    assert!(matches!(
        objects[0].material,
        Some(MaterialSpec::PbrPlus(ref m))
            if m.base_color.value == PropertyValue::Color([0.8, 0.4, 0.2])
                && m.base_color_map.as_deref() == Some(tex_id.as_str())
                && (m.normal_scale - 1.5).abs() < 1e-9
                && (m.reflection_intensity - 0.75).abs() < 1e-9
    ));
    assert!(matches!(
        objects[1].material,
        Some(MaterialSpec::Matcap(ref m)) if m.texture.as_deref() == Some(tex_id.as_str())
    ));
    assert!(matches!(
        objects[2].material,
        Some(MaterialSpec::ImagePlus(ref m)) if m.transparent && m.texture.as_deref() == Some(tex_id.as_str())
    ));
    assert!(matches!(
        objects[3].material,
        Some(MaterialSpec::NodeMaterial {
            resolution: 512,
            ..
        })
    ));
}

#[test]
fn verifies_r2_static_properties_native_effect_ids_dynamic_groups_and_motion_blur_tracks() {
    let mut project = Project::new_cm3_default();
    project.name = "R2 Fixes Verification".to_owned();

    // Bug 4: Dynamic sequence motionBlurTrackFrame and motionBlurSensitivity.
    project.sequence.motion_blur = MotionBlurSettings {
        enabled: true,
        samples: PropertyTrack::number(16.0),
        shutter: PropertyTrack::number(0.5),
        track_frame: 2,
        sensitivity: 145.5,
    };

    // Bug 3: Animated Vec2 / Vec3 / Color SequenceControls with `channels: vec![]` and `keyframes`.
    let vec2_ctl_track = PropertyTrack {
        animated: true,
        expression: Some("[value[0] * 2, value[1]]".to_owned()),
        static_behavior: crate::model::StaticBehavior::Constant,
        value: PropertyValue::Vec2([1.0, 2.0]),
        keyframes: vec![
            PropertyKeyframe::from_cm3_tween(
                0.0,
                PropertyValue::Vec2([1.0, 2.0]),
                257,
                [[-6.0, 0.0], [12.0, 3.0]],
                false,
            ),
            PropertyKeyframe::from_cm3_tween(
                30.0,
                PropertyValue::Vec2([10.0, -5.0]),
                3,
                [[-10.0, 0.0], [10.0, 0.0]],
                true,
            ),
        ],
        channels: Vec::new(),
    };
    let color_ctl_track = PropertyTrack {
        // Even when `animated` wasn't toggled manually, >1 keyframes must mark channels animated.
        animated: false,
        expression: None,
        static_behavior: crate::model::StaticBehavior::Constant,
        value: PropertyValue::Color([0.1, 0.2, 0.3]),
        keyframes: vec![
            PropertyKeyframe::linear(0.0, PropertyValue::Color([0.1, 0.2, 0.3])),
            // Shorter Vec2 keyframe verifies component fallback to base_comps[2] == 0.3.
            PropertyKeyframe::linear(45.0, PropertyValue::Vec2([0.9, 0.8])),
        ],
        channels: Vec::new(),
    };
    project.sequence.controls = vec![
        SequenceControl {
            id: "c1".to_owned(),
            name: "Anim Point".to_owned(),
            kind: SequenceControlKind::Point2D,
            track: vec2_ctl_track.clone(),
            preserved: BTreeMap::new(),
        },
        SequenceControl {
            id: "c2".to_owned(),
            name: "Anim Color".to_owned(),
            kind: SequenceControlKind::Color,
            track: color_ctl_track.clone(),
            preserved: BTreeMap::new(),
        },
    ];

    // Bug 1: CustomPbr (`custom` + `uvcustom`) static roughness/metalness vs PbrPlus dynamic,
    // plus Camera+ static vibrate fields and Scene3D static zoidiumBackdropOpacity.
    let mut custom_box = Object3D::new_shape_3d(
        "CustomPbrBox",
        Shape3DKind::Box {
            size: [10.0, 10.0, 10.0],
        },
    );
    custom_box.material = Some(MaterialSpec::CustomPbr(Box::new(CustomPbrMaterial {
        roughness: PropertyTrack::number(0.32),
        metalness: PropertyTrack::number(0.78),
        uv_normalized: false,
        ..CustomPbrMaterial::default()
    })));

    let mut uvcustom_box = Object3D::new_shape_3d(
        "UvCustomBox",
        Shape3DKind::Box {
            size: [10.0, 10.0, 10.0],
        },
    );
    uvcustom_box.material = Some(MaterialSpec::CustomPbr(Box::new(CustomPbrMaterial {
        roughness: PropertyTrack::number(0.41),
        metalness: PropertyTrack::number(0.64),
        uv_normalized: true,
        ..CustomPbrMaterial::default()
    })));

    let mut pbr_plus_box = Object3D::new_shape_3d(
        "PbrPlusBox",
        Shape3DKind::Box {
            size: [10.0, 10.0, 10.0],
        },
    );
    pbr_plus_box.material = Some(MaterialSpec::PbrPlus(Box::new(PbrPlusMaterial {
        roughness: PropertyTrack::number(0.25),
        metalness: PropertyTrack::number(0.9),
        ..PbrPlusMaterial::default()
    })));

    let vib_cam = Object3D::new(
        "VibCam",
        Object3DKind::Camera(Box::new(CameraData {
            vibrate: Some(CameraVibrateParams {
                enabled: true,
                seed: 12,
                position_amplitude: PropertyTrack::vec3([1.5, 2.5, 3.5]),
                rotation_amplitude: PropertyTrack::vec3([0.5, 1.0, 1.5]),
                frequency: PropertyTrack::number(2.0),
                regular_pulse: false,
                relative: false,
                enable_position: true,
                enable_rotation: true,
                position_frequency: PropertyTrack::number(3.25),
                rotation_frequency: PropertyTrack::number(1.75),
            }),
            ..CameraData::default()
        })),
    );

    let mut scene_layer = Layer::new_scene_3d(
        [1920, 1080],
        vec![custom_box, uvcustom_box, pbr_plus_box, vib_cam],
    );
    project.sequence.scene_backdrops.insert(
        scene_layer.id,
        SceneBackdrop {
            mode: 1,
            color: [0.2, 0.3, 0.4],
            opacity: PropertyTrack::number(0.65),
        },
    );

    // Bug 3: CustomShader with animated Vec3 & Color custom properties (`channels: vec![]`).
    let mut custom_fx = create_custom_shader("AnimUniforms", "void main() {}");
    if let EffectKind::CustomShader {
        ref mut custom_properties,
        ..
    } = custom_fx.kind
    {
        custom_properties.push(CustomProperty {
            name: "uDir".to_owned(),
            prop_type: Some(2),
            track: PropertyTrack {
                animated: true,
                expression: Some("normalize(value)".to_owned()),
                static_behavior: crate::model::StaticBehavior::Constant,
                value: PropertyValue::Vec3([0.0, 1.0, 0.0]),
                keyframes: vec![
                    PropertyKeyframe::from_cm3_tween(
                        0.0,
                        PropertyValue::Vec3([0.0, 1.0, 0.0]),
                        257,
                        [[-4.0, 0.0], [8.0, 0.5]],
                        false,
                    ),
                    PropertyKeyframe::linear(20.0, PropertyValue::Vec3([1.0, 0.0, -1.0])),
                ],
                channels: Vec::new(),
            },
            metadata: BTreeMap::new(),
        });
        custom_properties.push(CustomProperty {
            name: "uColor".to_owned(),
            prop_type: Some(4),
            track: color_ctl_track,
            metadata: BTreeMap::new(),
        });
    }

    // Bug 1: Standard CM3 effects with static params + dynamic params.
    let mut static_fx = create_standard_effect("static").expect("static");
    static_fx
        .params
        .insert("blending".to_owned(), PropertyTrack::option(2));

    let mut pixelsort_fx = create_standard_effect("pixelsort").expect("pixelsort");
    pixelsort_fx
        .params
        .insert("direction".to_owned(), PropertyTrack::option(1));
    pixelsort_fx
        .params
        .insert("reverse".to_owned(), PropertyTrack::option(1));
    pixelsort_fx
        .params
        .insert("affect".to_owned(), PropertyTrack::option(1));

    let mut chromakey_fx = create_standard_effect("chromakey").expect("chromakey");
    chromakey_fx
        .params
        .insert("invert".to_owned(), PropertyTrack::option(1));
    chromakey_fx
        .params
        .insert("mask".to_owned(), PropertyTrack::option(1));

    let mut transform_fx = create_standard_effect("transform").expect("transform");
    transform_fx
        .params
        .insert("cameraType".to_owned(), PropertyTrack::option(1));
    transform_fx
        .params
        .insert("wrap".to_owned(), PropertyTrack::option(1));

    let mut shutter_fx = create_standard_effect("shutter").expect("shutter");
    shutter_fx
        .params
        .insert("mode".to_owned(), PropertyTrack::option(2));

    // Bug 1 & Bug 2: Zoidium native plugin effects (including static params and manifest IDs).
    let mut vhs_fx = create_zoidium_effect("vhs").expect("vhs");
    vhs_fx.params.insert(
        "setup".to_owned(),
        PropertyTrack::constant(PropertyValue::Text("preset-a".to_owned())),
    );
    vhs_fx
        .params
        .insert("timeMode".to_owned(), PropertyTrack::number(1.0));
    vhs_fx
        .params
        .insert("glitchSeed".to_owned(), PropertyTrack::number(42.0));

    let mut datamosh_fx = create_zoidium_effect("datamosh").expect("datamosh");
    datamosh_fx
        .params
        .insert("timeMode".to_owned(), PropertyTrack::number(1.0));
    datamosh_fx
        .params
        .insert("motion".to_owned(), PropertyTrack::option(2));
    datamosh_fx
        .params
        .insert("interval".to_owned(), PropertyTrack::number(30.0));

    let mut jpeg_fx = create_zoidium_effect("jpegdamage").expect("jpegdamage");
    jpeg_fx
        .params
        .insert("timeMode".to_owned(), PropertyTrack::number(1.0));

    let looks_fx = create_zoidium_effect("magiclooks").expect("magiclooks");
    let mut sglow_fx = create_zoidium_effect("sglow").expect("sglow");
    sglow_fx
        .params
        .insert("timeMode".to_owned(), PropertyTrack::option(1));

    let make_native = |plugin_id: &str, effect_id: &str| EffectInstance {
        id: Uuid::new_v4(),
        name: effect_id.to_owned(),
        kind: EffectKind::Plugin {
            plugin_id: plugin_id.to_owned(),
            effect_id: effect_id.to_owned(),
        },
        enabled: PropertyTrack::option(1),
        params: BTreeMap::new(),
        preserved: BTreeMap::new(),
    };

    let mut layerinput_fx = make_native("layer-input", "layerinput");
    layerinput_fx
        .params
        .insert("source".to_owned(), PropertyTrack::list("track-1"));
    let timedisp_fx = make_native("native-fx", "timedisplacement");
    let echo_legacy_fx = make_native("openzoid-legacy", "echo-legacy");
    let smb_lite_fx = make_native("smart-motion-blur-lite", "smart-motion-blur-lite");
    let third_party_fx = make_native("third-party", "custom-fx");

    scene_layer.effects = vec![
        custom_fx,
        static_fx,
        pixelsort_fx,
        chromakey_fx,
        transform_fx,
        shutter_fx,
        vhs_fx,
        datamosh_fx,
        jpeg_fx,
        looks_fx,
        sglow_fx,
        layerinput_fx,
        timedisp_fx,
        echo_legacy_fx,
        smb_lite_fx,
        third_party_fx,
    ];

    project.sequence.video_tracks[0].clips[0].layer = scene_layer;

    let pz_bytes = export_pz_to_bytes(&project).expect("export r2 test pz");
    let entries = inspect_tar_entries(&pz_bytes);
    let doc: Value = serde_json::from_slice(&entries["project"]).expect("parse project json");

    // --- Verify Bug 4 on raw JSON ---
    let seq_props = &doc["sequence"]["properties"];
    assert_eq!(
        seq_props["motionBlurTrackFrame"],
        json!({
            "animated": false,
            "keyframes": [{ "frame": 0, "value": 2, "tween": 0 }]
        })
    );
    assert_eq!(
        seq_props["motionBlurSensitivity"],
        json!({
            "animated": false,
            "keyframes": [{ "frame": 0, "value": 145.5, "tween": 1 }]
        })
    );

    // --- Verify Bug 1 on raw JSON ---
    let scene_json = &doc["sequence"]["videoTracks"][0]["clips"][0]["object"];
    // Backdrop opacity is a raw number, NOT { animated, keyframes }.
    assert_eq!(
        scene_json["properties"]["childProperties"]["zoidiumBackdropOpacity"],
        json!(0.65)
    );
    // CustomPbr (`custom` & `uvcustom`) roughness/metalness are raw numbers.
    assert_eq!(
        scene_json["objects"][0]["material"]["properties"]["roughness"],
        json!(0.32)
    );
    assert_eq!(
        scene_json["objects"][0]["material"]["properties"]["metalness"],
        json!(0.78)
    );
    assert_eq!(
        scene_json["objects"][1]["material"]["properties"]["roughness"],
        json!(0.41)
    );
    assert_eq!(
        scene_json["objects"][1]["material"]["properties"]["metalness"],
        json!(0.64)
    );
    // PbrPlus roughness/metalness stay wrapped as dynamic tracks.
    assert!(
        scene_json["objects"][2]["material"]["properties"]["roughness"]["keyframes"].is_array()
    );
    assert!(
        scene_json["objects"][2]["material"]["properties"]["metalness"]["keyframes"].is_array()
    );
    // Camera+ vibrate static props are raw arrays/numbers.
    let vib_json = &scene_json["objects"][3]["properties"]["vibrate"];
    assert_eq!(vib_json["positionAmplitude"], json!([1.5, 2.5, 3.5]));
    assert_eq!(vib_json["positionFrequency"], json!(3.25));
    assert_eq!(vib_json["rotationAmplitude"], json!([0.5, 1.0, 1.5]));
    assert_eq!(vib_json["rotationFrequency"], json!(1.75));

    // Standard & native static effect params are raw primitives; dynamic params remain wrapped.
    let fx_arr = scene_json["effects"].as_array().expect("effects");
    assert_eq!(fx_arr[1]["properties"]["blending"], json!(2));
    assert!(fx_arr[1]["properties"]["amount"]["keyframes"].is_array());
    assert_eq!(fx_arr[2]["properties"]["direction"], json!(1));
    assert_eq!(fx_arr[2]["properties"]["reverse"], json!(1));
    assert_eq!(fx_arr[2]["properties"]["affect"], json!(1));
    assert!(fx_arr[2]["properties"]["threshold"]["keyframes"].is_array());
    assert_eq!(fx_arr[3]["properties"]["invert"], json!(1));
    assert_eq!(fx_arr[3]["properties"]["mask"], json!(1));
    assert!(fx_arr[3]["properties"]["soften"]["keyframes"].is_array());
    assert_eq!(fx_arr[4]["properties"]["cameraType"], json!(1));
    assert_eq!(fx_arr[4]["properties"]["wrap"], json!(1));
    assert_eq!(fx_arr[5]["properties"]["mode"], json!(2));
    assert_eq!(fx_arr[6]["properties"]["setup"], json!("preset-a"));
    assert_eq!(fx_arr[6]["properties"]["timeMode"], json!(1.0));
    assert_eq!(fx_arr[6]["properties"]["glitchSeed"], json!(42.0));
    assert!(fx_arr[6]["properties"]["noiseOpacity"]["keyframes"].is_array());
    assert_eq!(fx_arr[7]["properties"]["timeMode"], json!(1.0));
    assert_eq!(fx_arr[7]["properties"]["motion"], json!(2));
    assert_eq!(fx_arr[7]["properties"]["interval"], json!(30.0));
    assert!(fx_arr[7]["properties"]["amount"]["keyframes"].is_array());
    assert_eq!(fx_arr[8]["properties"]["timeMode"], json!(1.0));
    assert!(fx_arr[8]["properties"]["amount"]["keyframes"].is_array());
    assert_eq!(fx_arr[10]["properties"]["timeMode"], json!(1));
    assert!(fx_arr[10]["properties"]["glowWidth"]["keyframes"].is_array());
    assert_eq!(fx_arr[11]["properties"]["source"], json!("track-1"));

    // --- Verify Bug 2 on raw JSON ---
    assert_eq!(fx_arr[9]["type"], json!("looks"));
    assert_eq!(fx_arr[10]["type"], json!("sglow-hq-v2-preview"));
    assert_eq!(fx_arr[11]["type"], json!("layerinput"));
    assert_eq!(fx_arr[12]["type"], json!("timedisplacement"));
    assert_eq!(fx_arr[13]["type"], json!("echo-legacy"));
    assert_eq!(fx_arr[14]["type"], json!("smart-motion-blur-lite"));
    assert_eq!(fx_arr[15]["type"], json!("zoidium:third-party/custom-fx"));

    // --- Verify Bug 3 on raw JSON ---
    // CustomProperty uDir (Vec3) & uColor (Color)
    let cps = fx_arr[0]["customProperties"]
        .as_array()
        .expect("customProperties");
    assert_eq!(cps[0]["animated"], json!(true));
    assert_eq!(cps[0]["expression"], json!("normalize(value)"));
    let dir_objs = cps[0]["objects"].as_array().expect("uDir objects");
    assert_eq!(dir_objs.len(), 3);
    assert_eq!(dir_objs[0]["animated"], json!(true));
    assert_eq!(dir_objs[0]["keyframes"][0]["frame"], json!(0.0));
    assert_eq!(dir_objs[0]["keyframes"][0]["value"], json!(0.0));
    assert_eq!(dir_objs[0]["keyframes"][0]["tween"], json!(257));
    assert_eq!(
        dir_objs[0]["keyframes"][0]["continuousTangent"],
        json!(false)
    );
    assert_eq!(dir_objs[0]["keyframes"][1]["frame"], json!(20.0));
    assert_eq!(dir_objs[0]["keyframes"][1]["value"], json!(1.0));
    assert_eq!(dir_objs[2]["keyframes"][1]["value"], json!(-1.0));

    let col_objs = cps[1]["objects"].as_array().expect("uColor objects");
    assert_eq!(col_objs.len(), 3);
    assert_eq!(col_objs[0]["animated"], json!(true));
    assert_eq!(col_objs[0]["keyframes"][1]["value"], json!(0.9));
    assert_eq!(col_objs[1]["keyframes"][1]["value"], json!(0.8));
    // Fallback to base_comps[2] == 0.3 for shorter Vec2 keyframe.
    assert_eq!(col_objs[2]["keyframes"][1]["value"], json!(0.3));

    // SequenceControl Point2D & Color
    let ctls_json: Value = serde_json::from_str(
        seq_props["zoidiumControls"]
            .as_str()
            .expect("zoidiumControls"),
    )
    .expect("parse zoidiumControls");
    let ctl_arr = ctls_json["controls"].as_array().expect("controls");
    assert_eq!(ctl_arr[0]["animated"], json!(true));
    assert_eq!(ctl_arr[0]["expression"], json!("[value[0] * 2, value[1]]"));
    assert_eq!(
        ctl_arr[0]["objects"][0]["keyframes"][0]["tween"],
        json!(257)
    );
    assert_eq!(
        ctl_arr[0]["objects"][0]["keyframes"][0]["controlPoints"],
        json!([[-6.0, 0.0], [12.0, 3.0]])
    );
    assert_eq!(ctl_arr[0]["objects"][0]["keyframes"][1]["tween"], json!(3));
    assert_eq!(
        ctl_arr[0]["objects"][1]["keyframes"][1]["value"],
        json!(-5.0)
    );
    assert_eq!(
        ctl_arr[1]["objects"][2]["keyframes"][1]["value"],
        json!(0.3)
    );

    // --- Verify round-trip through `import_pz_from_bytes` ---
    let reimported = import_pz_from_bytes(&pz_bytes).expect("reimport r2 pz");
    let rp = reimported.project;

    // Animated SequenceControls round-trip with per-channel keyframes intact.
    assert_eq!(rp.sequence.controls.len(), 2);
    let r_pt = &rp.sequence.controls[0].track;
    assert!(r_pt.animated);
    assert_eq!(r_pt.expression.as_deref(), Some("[value[0] * 2, value[1]]"));
    assert_eq!(r_pt.channels.len(), 2);
    assert_eq!(r_pt.channels[0].keyframes.len(), 2);
    assert!(r_pt.channels[0].keyframes[0].is_bezier);
    assert!(!r_pt.channels[0].keyframes[0].continuous_tangent);
    assert_eq!(r_pt.channels[0].keyframes[1].easing_index, 3);
    assert_eq!(
        r_pt.channels[1].keyframes[1].value,
        PropertyValue::Number(-5.0)
    );

    let r_col = &rp.sequence.controls[1].track;
    assert!(r_col.animated);
    assert_eq!(r_col.channels.len(), 3);
    assert_eq!(
        r_col.channels[0].keyframes[1].value,
        PropertyValue::Number(0.9)
    );
    assert_eq!(
        r_col.channels[2].keyframes[1].value,
        PropertyValue::Number(0.3)
    );

    // Scene backdrop, CustomPbr, vibrate, and CustomProperty animated vectors round-trip.
    let rscene = &rp.sequence.video_tracks[0].clips[0].layer;
    assert_eq!(
        rp.sequence.scene_backdrops[&rscene.id].opacity.value,
        PropertyValue::Number(0.65)
    );
    let LayerKind::Scene3D { ref objects, .. } = rscene.kind else {
        panic!("expected Scene3D");
    };
    assert!(matches!(
        objects[0].material,
        Some(MaterialSpec::CustomPbr(ref m))
            if m.roughness.value == PropertyValue::Number(0.32)
                && m.metalness.value == PropertyValue::Number(0.78)
    ));
    let Object3DKind::Camera(ref rvcam) = objects[3].kind else {
        panic!("expected Camera");
    };
    let rvib = rvcam.vibrate.as_ref().expect("vibrate");
    assert_eq!(
        rvib.position_amplitude.value,
        PropertyValue::Vec3([1.5, 2.5, 3.5])
    );
    assert_eq!(rvib.position_frequency.value, PropertyValue::Number(3.25));

    let EffectKind::CustomShader {
        ref custom_properties,
        ..
    } = rscene.effects[0].kind
    else {
        panic!("expected CustomShader");
    };
    assert_eq!(custom_properties[0].track.channels.len(), 3);
    assert_eq!(
        custom_properties[0].track.expression.as_deref(),
        Some("normalize(value)")
    );
    assert_eq!(
        custom_properties[0].track.channels[2].keyframes[1].value,
        PropertyValue::Number(-1.0)
    );
    assert_eq!(custom_properties[1].track.channels.len(), 3);
    assert_eq!(
        custom_properties[1].track.channels[2].keyframes[1].value,
        PropertyValue::Number(0.3)
    );

    // Static effect params round-trip cleanly.
    assert_eq!(
        rscene.effects[1].params["blending"].value,
        PropertyValue::Option(2)
    );
    assert_eq!(
        rscene.effects[2].params["direction"].value,
        PropertyValue::Option(1)
    );
    assert_eq!(
        rscene.effects[3].params["invert"].value,
        PropertyValue::Option(1)
    );
    assert_eq!(
        rscene.effects[6].params["timeMode"].value,
        PropertyValue::Number(1.0)
    );
    // Motion blur dynamic sequence tracks round-trip cleanly.
    assert_eq!(rp.sequence.motion_blur.track_frame, 2);
    assert!((rp.sequence.motion_blur.sensitivity - 145.5).abs() < 1e-9);

    // Zoidium manifest IDs `looks` and `sglow-hq-v2-preview` import as first-class
    // Zoidium plugin effects without "unknown effect" warnings.
    assert!(matches!(
        &rscene.effects[9].kind,
        EffectKind::Plugin { plugin_id, effect_id }
            if plugin_id == "magic-looks" && effect_id == "magiclooks"
    ));
    assert!(matches!(
        &rscene.effects[10].kind,
        EffectKind::Plugin { plugin_id, effect_id }
            if plugin_id == "sglow-hq-v2-preview" && effect_id == "sglow"
    ));
    assert!(
        !reimported
            .warnings
            .iter()
            .any(|w| w.contains("\"looks\"") || w.contains("\"sglow-hq-v2-preview\"")),
        "looks / sglow-hq-v2-preview should not emit unknown-effect warnings: {:?}",
        reimported.warnings
    );
}
