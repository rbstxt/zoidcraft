use super::assets::remap_object_3d;
use super::tracks::rebase_time_curve_for_clipped_prefix;
use super::*;
use crate::model::{
    AssetKind, CompatMode, DrawOp, EffectKind, LayerKind, MaskMode, MaterialSpec, Object3DKind,
    PropertyKeyframe, PropertyTrack, PropertyValue, SequenceControlKind, Shape3DKind,
    StaticBehavior, Text3DAlign, cm3_blend_mode_from_index, normalize_blend_mode_token,
};
use flate2::Compression;
use flate2::write::GzEncoder;

fn build_pz_bytes(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let encoder = GzEncoder::new(Vec::new(), Compression::default());
    let mut builder = tar::Builder::new(encoder);
    for (path, data) in entries {
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder
            .append_data(&mut header, *path, *data)
            .expect("append entry");
    }
    builder
        .into_inner()
        .expect("finish tar")
        .finish()
        .expect("finish gzip")
}

#[test]
fn imports_synthetic_pz_covering_cm3_quirks_and_preservation() {
    let meta = br#"{"version": 5}"#;
    let zoidium = br#"{"name":"demo","plugins":["expressions-plus","constraints"]}"#;
    let present_asset_bytes = b"PNGDATA";

    let project_json = serde_json::json!({
        "customRootField": 42,
        "assets": {
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": {
                "type": 0,
                "source": 1,
                "external": false,
                "filename": "present.png",
                "size": 7,
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            },
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": {
                "type": 0,
                "source": 1,
                "external": false,
                "filename": "missing.png",
                "size": 999,
                "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            }
        },
        "media": [],
        "sequence": {
            "length": 150,
            "properties": {
                "name": "Intro",
                "resolution": [1280, 720],
                "rate": 60,
                "zoidiumControls": "{\"version\":1,\"controls\":[{\"id\":\"ctl_1\",\"name\":\"Master\",\"kind\":\"slider\",\"value\":0.5}]}"
            },
            "videoTracks": [
                {
                    "type": 0,
                    "enabled": true,
                    "clips": [
                        {
                            "start": 0,
                            "length": 150,
                            "properties": { "name": "Comp Clip" },
                            "object": {
                                "type": 2,
                                "properties": { "name": "MyComp", "unknownLayerProp": "keepme" },
                                "objects": [
                                    {
                                        "type": 3,
                                        "properties": {
                                            "name": "TopChild",
                                            "childProperties": { "image": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }
                                        }
                                    },
                                    {
                                        "type": 4,
                                        "properties": { "name": "BottomScene" },
                                        "objects": [
                                            {
                                                "type": 1,
                                                "properties": {
                                                    "name": "Title3D",
                                                    "text": "ZOID",
                                                    "bevel": 1,
                                                    "bevelSize": [0.25, 0.75],
                                                    "positionMode": 1,
                                                    "zoidiumTrackTo": "target-uuid",
                                                    "zoidiumTrackAxis": 5
                                                },
                                                "material": { "type": "singlecolor", "properties": { "color": [1.0, 0.5, 0.25] } }
                                            },
                                            {
                                                "type": 0,
                                                "objectType": 2,
                                                "properties": {
                                                    "name": "OpenCyl",
                                                    "geometryProperties": {
                                                        "cylinder_size": [5, 10, 20],
                                                        "cylinder_detail": 16,
                                                        "cylinder_openEnded": 0
                                                    }
                                                }
                                            },
                                            {
                                                "type": 0,
                                                "objectType": 2,
                                                "properties": {
                                                    "name": "ClosedCyl",
                                                    "geometryProperties": {
                                                        "cylinder_openEnded": 1
                                                    }
                                                }
                                            },
                                            {
                                                "type": 5,
                                                "properties": {
                                                    "name": "LegacyGroup",
                                                    "position": {
                                                        "animated": true,
                                                        "keyframes": [
                                                            { "frame": 0, "value": [1, 2, 3], "tween": 257 },
                                                            { "frame": 30, "value": [4, 5, 6], "tween": 1 }
                                                        ]
                                                    }
                                                },
                                                "objects": []
                                            },
                                            {
                                                "type": 99,
                                                "properties": { "name": "Mystery" },
                                                "mysteryField": true
                                            }
                                        ]
                                    },
                                    {
                                        "type": 6,
                                        "properties": { "name": "VecLayer" },
                                        "objects": [
                                            {
                                                "type": 0,
                                                "properties": { "name": "G" },
                                                "operations": [
                                                    {
                                                        "type": 0,
                                                        "properties": {
                                                            "color": [0.2, 0.4, 0.8]
                                                        }
                                                    },
                                                    {
                                                        "type": 1,
                                                        "properties": {
                                                            "color": [0.1, 0.9, 0.3],
                                                            "size": 3.0,
                                                            "opacity": 0.5
                                                        }
                                                    }
                                                ],
                                                "objects": []
                                            }
                                        ]
                                    }
                                ],
                                "effects": [
                                    {
                                        "type": "bloom",
                                        "properties": { "strength": 1.5, "extraUnknown": "ok" }
                                    },
                                    {
                                        "type": 1,
                                        "properties": { "name": "CustomFX", "fragShader": "void main(){}" }
                                    },
                                    {
                                        "type": 0,
                                        "properties": { "name": "Macro" },
                                        "objects": [
                                            { "type": "negative", "properties": { "enabled": "0" } }
                                        ]
                                    }
                                ]
                            }
                        }
                    ]
                }
            ],
            "audioTracks": [
                {
                    "type": 1,
                    "clips": [
                        {
                            "start": 0,
                            "length": 150,
                            "properties": { "name": "Track Audio" },
                            "object": { "properties": { "volume": 0.8, "pan": -0.25 } }
                        }
                    ]
                }
            ],
            "clipLinks": {
                "seed": 2,
                "links": {
                    "1": { "clips": [[0, 1, 0, 0], [0, 2, 0, 0]] }
                }
            }
        }
    });

    let project_bytes = serde_json::to_vec(&project_json).expect("encode json");
    let archive = build_pz_bytes(&[
        ("meta", meta),
        ("project", &project_bytes),
        ("zoidium.json", zoidium),
        ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", present_asset_bytes),
    ]);

    let imported = import_pz_from_bytes(&archive).expect("import pz");
    let project = imported.project;

    assert_eq!(project.imported_from_pz_version, Some(5));
    assert_eq!(
        project.preserved.get("customRootField"),
        Some(&Value::from(42))
    );
    assert!(project.plugin_requirements.contains("expressions-plus"));
    assert!(project.plugin_requirements.contains("constraints"));

    // Present vs missing assets (normalized to full 64-hex SHA-256 keys per Spec 03 §7.3)
    let expected_present_key = sha256_hex(present_asset_bytes);
    let expected_missing_key =
        sha256_hex(b"pz-missing:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:missing.png");
    assert_eq!(expected_present_key.len(), 64);
    assert!(!project.assets[&expected_present_key].missing);
    assert_eq!(
        project.assets[&expected_present_key].bytes.as_deref(),
        Some(present_asset_bytes.as_slice())
    );
    assert!(project.assets[&expected_missing_key].missing);
    assert!(project.assets[&expected_missing_key].bytes.is_none());
    assert!(imported.warnings.iter().any(|w| w.contains("missing.png")));

    // Sequence & controls
    assert_eq!(project.sequence.resolution, [1280, 720]);
    assert_eq!(project.sequence.fps, 60);
    assert_eq!(project.sequence.controls.len(), 1);
    assert_eq!(project.sequence.controls[0].name, "Master");

    // Clip links resolved to UUIDs
    let v_clip_id = project.sequence.video_tracks[0].clips[0].id;
    let a_clip_id = project.sequence.audio_tracks[0].clips[0].id;
    assert_eq!(project.sequence.clip_links[&1], vec![v_clip_id, a_clip_id]);

    let comp_layer = &project.sequence.video_tracks[0].clips[0].layer;
    assert_eq!(
        comp_layer.preserved.get("property.unknownLayerProp"),
        Some(&Value::from("keepme"))
    );

    // Effects: bloom override + custom shader + group with Shader+ stringified "0" option
    assert_eq!(comp_layer.effects.len(), 3);
    assert_eq!(
        comp_layer.effects[0].params["strength"].value,
        PropertyValue::Number(1.5)
    );
    assert_eq!(
        comp_layer.effects[0].params["radius"].value,
        PropertyValue::Number(0.8)
    );
    assert!(matches!(
        comp_layer.effects[1].kind,
        EffectKind::CustomShader { .. }
    ));
    let EffectKind::EffectGroup { children, .. } = &comp_layer.effects[2].kind else {
        panic!("expected EffectGroup");
    };
    assert_eq!(children[0].enabled.value, PropertyValue::Option(0));

    // Composite reversal: CM3 [TopChild, BottomScene, VecLayer] -> ZoidCraft [VecLayer, BottomScene, TopChild]
    let LayerKind::Composite { layers } = &comp_layer.kind else {
        panic!("expected Composite");
    };
    assert_eq!(layers.len(), 3);
    assert_eq!(layers[0].name, "VecLayer");
    assert_eq!(layers[1].name, "BottomScene");
    assert_eq!(layers[2].name, "TopChild");

    // Verify draw.fill & draw.stroke strict [0]=R, [1]=G, [2]=B index order
    let LayerKind::Shape2D { shapes } = &layers[0].kind else {
        panic!("expected Shape2D");
    };
    assert_eq!(
        shapes[0].operations[0],
        DrawOp::Fill {
            color: PropertyTrack::color([0.2, 0.4, 0.8])
        }
    );
    assert_eq!(
        shapes[0].operations[1],
        DrawOp::Stroke {
            color: PropertyTrack::color([0.1, 0.9, 0.3]),
            size: PropertyTrack::number(3.0),
            opacity: PropertyTrack::number(0.5),
        }
    );

    // Verify Scene3D objects
    let LayerKind::Scene3D {
        objects,
        shadow_mode,
    } = &layers[1].kind
    else {
        panic!("expected Scene3D");
    };
    assert_eq!(*shadow_mode, CompatMode::Legacy);

    // 3D Text bevelSize[0] -> bevel_thickness, bevelSize[1] -> bevel_size
    let Object3DKind::Text3D(ref t3d) = objects[0].kind else {
        panic!("expected Text3D");
    };
    assert_eq!(t3d.text, "ZOID");
    assert!(t3d.bevel);
    assert!((t3d.bevel_thickness - 0.25).abs() < 1e-12);
    assert!((t3d.bevel_size - 0.75).abs() < 1e-12);
    assert_eq!(t3d.position_mode, Text3DAlign::Left);
    assert_eq!(
        objects[0]
            .constraints
            .track_to
            .as_ref()
            .map(|c| c.target_id.as_str()),
        Some("target-uuid")
    );

    // Cylinder openEnded inversion: 0 => open_ended = true; 1 => open_ended = false
    assert!(matches!(
        objects[1].kind,
        Object3DKind::Shape3D(Shape3DKind::Cylinder {
            open_ended: true,
            detail: 16,
            ..
        })
    ));
    assert!(matches!(
        objects[2].kind,
        Object3DKind::Shape3D(Shape3DKind::Cylinder {
            open_ended: false,
            ..
        })
    ));

    // Legacy array-keyframed group split into 3 per-axis channels with Bezier tween (257 = (1<<8)|1)
    assert_eq!(objects[3].position.channels.len(), 3);
    assert_eq!(objects[3].position.channels[0].keyframes.len(), 2);
    assert!(objects[3].position.channels[0].keyframes[0].is_bezier);
    assert_eq!(objects[3].position.channels[0].keyframes[0].easing_index, 1);
    assert_eq!(
        objects[3].position.channels[2].keyframes[1].value,
        PropertyValue::Number(6.0)
    );

    // Unknown / Dummy preserved
    assert!(matches!(objects[4].kind, Object3DKind::Dummy { .. }));
}

#[test]
fn pz_import_rejects_path_traversal() {
    use std::io::Write as _;
    let mut header = tar::Header::new_gnu();
    let name = b"../escape";
    header.as_gnu_mut().expect("gnu header").name[..name.len()].copy_from_slice(name);
    header.set_size(2);
    header.set_mode(0o644);
    header.set_entry_type(tar::EntryType::Regular);
    header.set_cksum();

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(header.as_bytes()).expect("header");
    let mut block = [0u8; 512];
    block[..2].copy_from_slice(b"{}");
    encoder.write_all(&block).expect("data");
    encoder.write_all(&[0u8; 1024]).expect("eof");
    let evil = encoder.finish().expect("gzip");

    assert!(matches!(
        import_pz_from_bytes(&evil),
        Err(ProjectError::InvalidArchivePath(_))
    ));
}

#[test]
fn imports_namespaced_plugin_effects_pbr_plus_defaults_and_preserves_nested_props() {
    let tex_base = b"BASE";
    let tex_emissive = b"EMIS";
    let tex_rough = b"ROUG";
    let tex_metal = b"METL";
    let tex_norm = b"NORM";
    let tex_alpha = b"ALPH";
    let tex_model = b"MODL";

    let project_json = serde_json::json!({
        "assets": {
            "11111111111111111111111111111111": { "type": 0, "source": 1, "external": false, "filename": "base.png" },
            "22222222222222222222222222222222": { "type": 0, "source": 1, "external": false, "filename": "emis.png" },
            "33333333333333333333333333333333": { "type": 0, "source": 1, "external": false, "filename": "roug.png" },
            "44444444444444444444444444444444": { "type": 0, "source": 1, "external": false, "filename": "metl.png" },
            "55555555555555555555555555555555": { "type": 0, "source": 1, "external": false, "filename": "norm.png" },
            "66666666666666666666666666666666": { "type": 0, "source": 1, "external": false, "filename": "alph.png" },
            "77777777777777777777777777777777": { "type": 0, "source": 1, "external": false, "filename": "modl.png" }
        },
        "sequence": {
            "properties": { "name": "EdgeCaseSeq", "rate": 30 },
            "videoTracks": [{
                "type": 0,
                "clips": [
                    {
                        "start": -30,
                        "length": 20,
                        "properties": { "name": "OffTimeline" },
                        "object": { "type": 5, "properties": {} }
                    },
                    {
                        "start": -10,
                        "length": 50,
                        "properties": { "name": "TrimmedClip", "isComp": true, "compLength": 50 },
                        "object": {
                            "type": 4,
                            "properties": {
                                "name": "Scene",
                                "opacity": {
                                    "animated": true,
                                    "keyframes": [
                                        { "frame": 10, "value": 0.25 },
                                        { "frame": 30, "value": 1.0 }
                                    ]
                                }
                            },
                            "objects": [
                                {
                                    "type": 0,
                                    "objectType": 1,
                                    "properties": {
                                        "name": "DefaultMatBox",
                                        "zoidiumTrackTo": "",
                                        "zoidiumFollow": ""
                                    }
                                },
                                {
                                    "type": 0,
                                    "objectType": 1,
                                    "properties": { "name": "PbrPlusBox" },
                                    "material": {
                                        "type": "zoidium:material-plus/pbr-plus",
                                        "properties": {
                                            "baseColor": [0.9, 0.8, 0.7],
                                            "baseColorMap": "11111111111111111111111111111111",
                                            "emissive": [0.1, 0.2, 0.3],
                                            "emissiveMap": "22222222222222222222222222222222",
                                            "roughness": 0.35,
                                            "roughnessMap": "33333333333333333333333333333333",
                                            "metalness": 0.8,
                                            "metalnessMap": "44444444444444444444444444444444",
                                            "normalMap": "55555555555555555555555555555555",
                                            "normalScale": 1.5,
                                            "alphaMap": "66666666666666666666666666666666",
                                            "opacity": 0.9,
                                            "transparent": 1,
                                            "blending": 2,
                                            "side": 2,
                                            "wrap": 0,
                                            "repeat": [2.0, 3.0],
                                            "offset": [0.1, 0.2],
                                            "center": [0.5, 0.5],
                                            "rotation": 0.75,
                                            "reflectionIntensity": 0.4,
                                            "customMatExtra": "kept"
                                        }
                                    }
                                },
                                {
                                    "type": 0,
                                    "objectType": 1,
                                    "properties": { "name": "MatcapBox" },
                                    "material": {
                                        "type": "zoidium:material-plus/matcap",
                                        "properties": { "transparent": 1, "blending": 2, "side": 2 }
                                    }
                                },
                                {
                                    "type": 0,
                                    "objectType": 1,
                                    "properties": { "name": "ImagePlusBox" },
                                    "material": {
                                        "type": "zoidium:material-plus/image-plus",
                                        "properties": {
                                            "transparent": 0,
                                            "blending": 3,
                                            "wrap": 2,
                                            "repeat": [4.0, 5.0],
                                            "offset": [0.2, 0.3],
                                            "center": [0.4, 0.6],
                                            "rotation": 1.25,
                                            "side": 1
                                        }
                                    }
                                },
                                {
                                    "type": 2,
                                    "objectType": 1,
                                    "properties": {
                                        "name": "ModelObj",
                                        "modelProperties": {
                                            "label": "hello",
                                            "texAsset": "77777777777777777777777777777777"
                                        }
                                    }
                                }
                            ],
                            "effects": [
                                {
                                    "type": "zoidium:native-fx/echo",
                                    "properties": { "decay": 0.6 }
                                },
                                {
                                    "type": "zoidium:openzoid-legacy/vhs",
                                    "properties": {}
                                },
                                {
                                    "type": "zoidium:future-pack/superfx",
                                    "properties": { "strength": "NaN" }
                                },
                                {
                                    "type": 1,
                                    "properties": {
                                        "name": "DynShader",
                                        "fragShader": {
                                            "animated": false,
                                            "keyframes": [{ "frame": 0, "value": "void main(){gl_FragColor=vec4(1.0);}" }]
                                        }
                                    },
                                    "customProperties": [
                                        {
                                            "type": { "type": 5, "name": "TintGrad" },
                                            "value": {
                                                "animated": true,
                                                "keyframes": [
                                                    { "frame": 0, "value": [{ "position": 0.0, "color": "rgba(255,0,0,1)" }] }
                                                ]
                                            }
                                        },
                                        {
                                            "name": "MyCurve",
                                            "type": 10,
                                            "value": { "objects": [1, 2], "value": 99, "points": [0, 1] }
                                        },
                                        {
                                            "name": "MyShaderProp",
                                            "type": 11,
                                            "value": "float f() { return 1.0; }"
                                        },
                                        {
                                            "name": "ExprNum",
                                            "type": 0,
                                            "value": { "value": 3.5, "expression": "time * 2.0", "animated": true }
                                        }
                                    ]
                                }
                            ]
                        }
                    },
                    {
                        "start": 10,
                        "length": 60,
                        "properties": { "name": "TextLayerClip" },
                        "object": {
                            "type": 7,
                            "properties": {
                                "name": "Text2D",
                                "scale": { "objects": [{ "keyframes": [] }, { "keyframes": [] }] },
                                "blending": null,
                                "childProperties": {
                                    "text": "Hi",
                                    "zoidiumTextPlus2D": { "tracking": 12 }
                                }
                            },
                            "objects": [{
                                "type": 0,
                                "properties": {
                                    "name": "S",
                                    "customShapeProp": 7,
                                    "position": { "animated": true, "keyframes": [{ "frame": 0, "value": [] }] }
                                }
                            }]
                        }
                    }
                ]
            }],
            "audioTracks": [{
                "type": 1,
                "clips": [
                    { "start": -20, "length": 10, "properties": { "name": "DroppedAudio" } },
                    {
                        "start": -10,
                        "length": 40,
                        "properties": {
                            "name": "TrimmedAudio",
                            "time": {
                                "animated": true,
                                "keyframes": [
                                    { "frame": 0, "value": 0.0 },
                                    { "frame": 5, "value": 0.5 }
                                ]
                            },
                            "audioExtra": "yes"
                        },
                        "object": {
                            "properties": {
                                "volume": {
                                    "animated": true,
                                    "keyframes": [{ "frame": 15, "value": 0.5 }]
                                }
                            }
                        }
                    }
                ]
            }]
        }
    });

    let project_bytes = serde_json::to_vec(&project_json).expect("encode");
    let archive = build_pz_bytes(&[
        ("project", &project_bytes),
        ("11111111111111111111111111111111", tex_base),
        ("22222222222222222222222222222222", tex_emissive),
        ("33333333333333333333333333333333", tex_rough),
        ("44444444444444444444444444444444", tex_metal),
        ("55555555555555555555555555555555", tex_norm),
        ("66666666666666666666666666666666", tex_alpha),
        ("77777777777777777777777777777777", tex_model),
    ]);

    let imported = import_pz_from_bytes(&archive).expect("import pz");
    let project = imported.project;

    // Sequence length computed from max clip end (10 + 60 = 70).
    assert!((project.sequence.length_frames - 70.0).abs() < 1e-9);

    // Off-timeline clips dropped; trimmed clips remain.
    assert_eq!(project.sequence.video_tracks[0].clips.len(), 2);
    assert_eq!(project.sequence.audio_tracks[0].clips.len(), 1);

    let vclip = &project.sequence.video_tracks[0].clips[0];
    assert!((vclip.start - 0.0).abs() < 1e-9);
    assert!((vclip.length - 40.0).abs() < 1e-9);
    // Fallback time_curve was built from raw_length=50 then rebased by 10 frames -> [0..40] => [10/30..50/30].
    assert_eq!(vclip.time_curve.keyframes.len(), 2);
    assert!((vclip.time_curve.keyframes[0].frame - 0.0).abs() < 1e-9);
    assert!((vclip.time_curve.keyframes[1].frame - 40.0).abs() < 1e-9);
    assert!((vclip.time_curve.keyframes[1].value.as_f64().unwrap() - 50.0 / 30.0).abs() < 1e-9);
    // Layer keyframes shifted by raw_start (-10): 10->0, 30->20.
    assert!((vclip.layer.opacity.keyframes[0].frame - 0.0).abs() < 1e-9);
    assert!((vclip.layer.opacity.keyframes[1].frame - 20.0).abs() < 1e-9);
    // Unknown clip.properties preserved.
    assert_eq!(
        vclip.preserved.get("property.isComp"),
        Some(&Value::Bool(true))
    );

    // Audio clip: keyframes shifted by -10 (15->5), and time_curve with all keys < 10 gets a tail keyframe.
    let aclip = &project.sequence.audio_tracks[0].clips[0];
    assert!((aclip.length - 30.0).abs() < 1e-9);
    assert_eq!(aclip.time_curve.keyframes.len(), 2);
    assert!((aclip.time_curve.keyframes[1].frame - 30.0).abs() < 1e-9);
    assert!((aclip.audio.volume.keyframes[0].frame - 5.0).abs() < 1e-9);
    assert_eq!(
        aclip.preserved.get("property.audioExtra"),
        Some(&Value::from("yes"))
    );

    // Namespaced plugin effects registered & recognized.
    assert!(project.plugin_requirements.contains("native-fx"));
    assert!(project.plugin_requirements.contains("openzoid-legacy"));
    assert!(project.plugin_requirements.contains("future-pack"));
    assert!(!project.plugin_requirements.contains("constraints"));

    let fx = &vclip.layer.effects;
    assert_eq!(
        fx[0].kind,
        EffectKind::Plugin {
            plugin_id: "native-fx".into(),
            effect_id: "echo".into(),
        }
    );
    assert_eq!(
        fx[1].kind,
        EffectKind::Plugin {
            plugin_id: "openzoid-legacy".into(),
            effect_id: "vhs".into(),
        }
    );
    assert_eq!(
        fx[2].kind,
        EffectKind::Plugin {
            plugin_id: "future-pack".into(),
            effect_id: "superfx".into(),
        }
    );

    // Dynamic fragShader + custom properties (Gradient, Curve, Shader, {value, expression}).
    let EffectKind::CustomShader {
        ref frag_shader,
        ref custom_properties,
    } = fx[3].kind
    else {
        panic!("expected CustomShader");
    };
    assert_eq!(frag_shader, "void main(){gl_FragColor=vec4(1.0);}");
    assert_eq!(custom_properties[0].name, "TintGrad");
    assert_eq!(custom_properties[0].prop_type, Some(5));
    assert!(matches!(
        custom_properties[0].track.value,
        PropertyValue::Gradient(_)
    ));
    // Non-interpolated types default tween to 0 (Hold).
    assert_eq!(custom_properties[0].track.keyframes[0].tween, 0);
    assert_eq!(custom_properties[0].track.keyframes[0].easing_index, 0);
    // Curve object with "objects" & "value" keys is not mangled.
    assert_eq!(
        custom_properties[1].track.value,
        PropertyValue::Curve(
            serde_json::json!({ "objects": [1, 2], "value": 99, "points": [0, 1] })
        )
    );
    assert_eq!(
        custom_properties[2].track.value,
        PropertyValue::Shader("float f() { return 1.0; }".into())
    );
    assert_eq!(custom_properties[3].track.value, PropertyValue::Number(3.5));
    assert_eq!(
        custom_properties[3].track.expression.as_deref(),
        Some("time * 2.0")
    );
    assert!(custom_properties[3].track.animated);

    // Scene3D objects: default singlecolor material, full PbrPlus/Matcap/ImagePlus & asset remap, ModelData.
    let LayerKind::Scene3D { ref objects, .. } = vclip.layer.kind else {
        panic!("expected Scene3D");
    };
    assert!(matches!(
        objects[0].material,
        Some(MaterialSpec::SingleColor { side: 2, .. })
    ));
    assert!(objects[0].constraints.track_to.is_none());
    assert!(objects[0].constraints.follow.is_none());

    let Some(MaterialSpec::PbrPlus(ref pbr)) = objects[1].material else {
        panic!("expected PbrPlus");
    };
    assert_eq!(
        pbr.base_color_map.as_deref(),
        Some(sha256_hex(tex_base).as_str())
    );
    assert_eq!(
        pbr.emissive_map.as_deref(),
        Some(sha256_hex(tex_emissive).as_str())
    );
    assert_eq!(
        pbr.roughness_map.as_deref(),
        Some(sha256_hex(tex_rough).as_str())
    );
    assert_eq!(
        pbr.metalness_map.as_deref(),
        Some(sha256_hex(tex_metal).as_str())
    );
    assert_eq!(
        pbr.normal_map.as_deref(),
        Some(sha256_hex(tex_norm).as_str())
    );
    assert_eq!(
        pbr.alpha_map.as_deref(),
        Some(sha256_hex(tex_alpha).as_str())
    );
    assert!((pbr.normal_scale - 1.5).abs() < 1e-9);
    assert!((pbr.reflection_intensity - 0.4).abs() < 1e-9);
    assert!(pbr.transparent);
    assert_eq!(pbr.blending, 2);
    assert_eq!(pbr.side, 2);
    assert_eq!(pbr.wrap, 0);
    assert_eq!(pbr.preserved.len(), 1);
    assert_eq!(
        pbr.preserved.get("customMatExtra"),
        Some(&Value::from("kept"))
    );

    let Some(MaterialSpec::Matcap(ref mc)) = objects[2].material else {
        panic!("expected Matcap");
    };
    assert!(mc.transparent);
    assert_eq!(mc.blending, 2);
    assert_eq!(mc.side, 2);

    let Some(MaterialSpec::ImagePlus(ref ip)) = objects[3].material else {
        panic!("expected ImagePlus");
    };
    assert!(!ip.transparent);
    assert_eq!(ip.blending, 3);
    assert_eq!(ip.wrap, 2);
    assert_eq!(ip.side, 1);
    assert_eq!(ip.repeat.value, PropertyValue::Vec2([4.0, 5.0]));

    let Object3DKind::Model(ref mdl) = objects[4].kind else {
        panic!("expected Model");
    };
    assert_eq!(
        mdl.model_properties["label"].value,
        PropertyValue::Text("hello".into())
    );
    // Verify remap_object_3d remaps Asset tracks inside ModelData::model_properties.
    let mut model_obj = objects[4].clone();
    if let Object3DKind::Model(ref mut m) = model_obj.kind {
        m.model_properties.insert(
            "tex".into(),
            PropertyTrack::asset(Some("77777777777777777777777777777777".into())),
        );
    }
    let mut remap = BTreeMap::new();
    remap.insert(
        "77777777777777777777777777777777".into(),
        sha256_hex(tex_model),
    );
    remap_object_3d(&mut model_obj, &remap);
    let Object3DKind::Model(ref remapped_mdl) = model_obj.kind else {
        panic!("expected Model");
    };
    assert_eq!(
        remapped_mdl.model_properties["tex"].value,
        PropertyValue::Asset(Some(sha256_hex(tex_model)))
    );

    // Text2D: empty channel defaults preserve scale [1, 1], childProperty preserved, shape.properties preserved,
    // empty legacy array keyframe doesn't drop keyframes.
    let text_layer = &project.sequence.video_tracks[0].clips[1].layer;
    assert_eq!(text_layer.scale.value, PropertyValue::Vec2([1.0, 1.0]));
    assert_eq!(text_layer.blending, "BLEND_SRC_OVER");
    assert_eq!(
        text_layer.preserved.get("childProperty.zoidiumTextPlus2D"),
        Some(&serde_json::json!({ "tracking": 12 }))
    );
    let LayerKind::Text2D { ref shapes, .. } = text_layer.kind else {
        panic!("expected Text2D");
    };
    assert_eq!(
        shapes[0].preserved.get("property.customShapeProp"),
        Some(&Value::from(7))
    );
    assert_eq!(shapes[0].position.keyframes.len(), 1);

    // Imported project must save & reload cleanly as .zcraft (no NaN, standard/plugin fx serialize).
    let zcraft_bytes = crate::zcraft::save_zcraft_to_bytes(&project).expect("save zcraft");
    let reloaded = crate::zcraft::load_zcraft_from_bytes(&zcraft_bytes).expect("load zcraft");
    assert_eq!(reloaded.sequence.video_tracks[0].clips.len(), 2);
}

#[test]
fn clip_links_preserve_targets_when_earlier_clips_are_dropped() {
    let project_json = serde_json::json!({
        "sequence": {
            "properties": { "name": "DroppedLinkSeq", "rate": 30 },
            "videoTracks": [{
                "type": 0,
                "clips": [
                    {
                        "start": -50.0,
                        "length": 30.0,
                        "properties": { "name": "DroppedVideo" },
                        "object": { "type": 5, "properties": {} }
                    },
                    {
                        "start": 0.0,
                        "length": 60.0,
                        "properties": { "name": "KeptVideoA" },
                        "object": { "type": 5, "properties": {} }
                    },
                    {
                        "start": 60.0,
                        "length": 60.0,
                        "properties": { "name": "KeptVideoB" },
                        "object": { "type": 5, "properties": {} }
                    }
                ]
            }],
            "audioTracks": [{
                "type": 1,
                "clips": [
                    {
                        "start": -50.0,
                        "length": 30.0,
                        "properties": { "name": "DroppedAudio" }
                    },
                    {
                        "start": 0.0,
                        "length": 60.0,
                        "properties": { "name": "KeptAudioA" }
                    }
                ]
            }],
            "clipLinks": {
                "links": {
                    "0": { "clips": [[0, 1, 0, 0], [0, 2, 0, 0]] },
                    "1": { "clips": [[0, 1, 0, 1], [0, 2, 0, 1]] },
                    "2": { "clips": [[0, 1, 0, 0], [0, 1, 0, 2]] }
                }
            }
        }
    });

    let project_bytes = serde_json::to_vec(&project_json).expect("encode json");
    let archive = build_pz_bytes(&[("project", &project_bytes)]);
    let imported = import_pz_from_bytes(&archive).expect("import pz");
    let seq = &imported.project.sequence;

    assert_eq!(seq.video_tracks[0].clips.len(), 2);
    assert_eq!(seq.audio_tracks[0].clips.len(), 1);

    let kept_video_a = &seq.video_tracks[0].clips[0];
    let kept_video_b = &seq.video_tracks[0].clips[1];
    let kept_audio_a = &seq.audio_tracks[0].clips[0];
    assert_eq!(kept_video_a.name, "KeptVideoA");
    assert_eq!(kept_video_b.name, "KeptVideoB");
    assert_eq!(kept_audio_a.name, "KeptAudioA");

    assert!(!seq.clip_links.contains_key(&0));
    assert_eq!(seq.clip_links[&1], vec![kept_video_a.id, kept_audio_a.id]);
    // Group 2 had only 1 surviving member (`kept_video_b`), so singleton pruning removes it.
    assert!(!seq.clip_links.contains_key(&2));
    assert_eq!(kept_video_a.link, Some(1));
    assert_eq!(kept_audio_a.link, Some(1));
    assert_eq!(kept_video_b.link, None);
}

#[test]
fn clip_links_support_5_element_and_legacy_3_element_and_prune_singletons() {
    let project_json = serde_json::json!({
        "sequence": {
            "properties": { "name": "LinksSeq", "rate": 30 },
            "videoTracks": [{
                "type": 0,
                "clips": [
                    { "start": 0, "length": 30, "link": 99, "properties": { "name": "V0" }, "object": { "type": 5, "properties": {} } },
                    { "start": 30, "length": 30, "properties": { "name": "V1" }, "object": { "type": 5, "properties": {} } },
                    { "start": 60, "length": 30, "properties": { "name": "V2" }, "object": { "type": 5, "properties": {} } }
                ]
            }],
            "audioTracks": [{
                "type": 1,
                "clips": [
                    { "start": 0, "length": 30, "properties": { "name": "A0" } },
                    { "start": 30, "length": 30, "properties": { "name": "A1" } },
                    { "start": 60, "length": 30, "properties": { "name": "A2" } }
                ]
            }],
            "clipLinks": {
                "seed": 7,
                "links": {
                    "1": { "clips": [[0, 1, 0, 0, 1], [0, 2, 0, 0, 2]] },
                    "2": { "clips": [[0, 0, 2], [1, 0, 1]] },
                    "3": { "clips": [[0, 1, 0, 0, 0]] }
                }
            }
        }
    });
    let archive = build_pz_bytes(&[("project", &serde_json::to_vec(&project_json).unwrap())]);
    let seq = import_pz_from_bytes(&archive).unwrap().project.sequence;

    let v0 = &seq.video_tracks[0].clips[0];
    let v1 = &seq.video_tracks[0].clips[1];
    let v2 = &seq.video_tracks[0].clips[2];
    let a1 = &seq.audio_tracks[0].clips[1];
    let a2 = &seq.audio_tracks[0].clips[2];

    // 5-element address links clip index 1 (video) <-> clip index 2 (audio), NOT clip 0!
    assert_eq!(seq.clip_links[&1], vec![v1.id, a2.id]);
    assert_eq!(v1.link, Some(1));
    assert_eq!(a2.link, Some(1));

    // Legacy 3-element address [0, 0, 2] (video clip 2) <-> [1, 0, 1] (audio clip 1)
    assert_eq!(seq.clip_links[&2], vec![v2.id, a1.id]);
    assert_eq!(v2.link, Some(2));
    assert_eq!(a1.link, Some(2));

    // Singleton group 3 pruned and stale link=99 cleared on V0
    assert!(!seq.clip_links.contains_key(&3));
    assert_eq!(v0.link, None);
    assert_eq!(seq.preserved.get("clipLinks.seed"), Some(&Value::from(7)));
}

#[test]
fn parses_real_zoidium_controls_schema_with_keyframes_groups_and_expressions() {
    let controls_payload = serde_json::json!({
        "version": 1,
        "nextId": 5,
        "controls": [
            {
                "type": {
                    "custom": true,
                    "dynamic": true,
                    "type": 0,
                    "value": 0,
                    "zoidiumControlKind": "Slider",
                    "zoidiumControlId": "c1",
                    "min": 0,
                    "max": 100,
                    "step": 0.01
                },
                "properties": { "name": "Master Gain" },
                "animated": true,
                "expression": "value * 2",
                "keyframes": [
                    { "frame": 0, "value": 10.0, "tween": 1 },
                    { "frame": 30, "value": 50.0, "tween": 1 }
                ]
            },
            {
                "type": {
                    "custom": true,
                    "dynamic": true,
                    "type": 1,
                    "value": [0, 0],
                    "zoidiumControlKind": "Point 2D",
                    "zoidiumControlId": "c2"
                },
                "properties": { "name": "Aim" },
                "objects": [
                    { "animated": false, "keyframes": [{ "frame": 0, "value": 12.5, "tween": 1 }] },
                    { "animated": false, "keyframes": [{ "frame": 0, "value": -4.0, "tween": 1 }] }
                ]
            },
            {
                "type": {
                    "custom": true,
                    "dynamic": true,
                    "type": 2,
                    "value": [0, 0, 0],
                    "zoidiumControlKind": "Point 3D",
                    "zoidiumControlId": "c3"
                },
                "properties": { "name": "Offset3D" },
                "objects": [
                    { "animated": false, "keyframes": [{ "frame": 0, "value": 1.0, "tween": 1 }] },
                    { "animated": false, "keyframes": [{ "frame": 0, "value": 2.0, "tween": 1 }] },
                    { "animated": false, "keyframes": [{ "frame": 0, "value": 3.0, "tween": 1 }] }
                ]
            },
            {
                "type": {
                    "custom": true,
                    "dynamic": true,
                    "type": 6,
                    "value": 0,
                    "zoidiumControlKind": "Checkbox",
                    "zoidiumControlId": "c4",
                    "items": "off;on"
                },
                "properties": { "name": "Toggle" },
                "animated": false,
                "keyframes": [{ "frame": 0, "value": 1, "tween": 0 }]
            }
        ]
    });

    let project_json = serde_json::json!({
        "sequence": {
            "properties": {
                "name": "CtlSeq",
                "rate": 30,
                "zoidiumControls": { "value": serde_json::to_string(&controls_payload).unwrap() }
            },
            "videoTracks": [],
            "audioTracks": []
        }
    });
    let archive = build_pz_bytes(&[("project", &serde_json::to_vec(&project_json).unwrap())]);
    let project = import_pz_from_bytes(&archive).unwrap().project;

    assert!(project.plugin_requirements.contains("expressions-plus"));
    assert_eq!(
        project.sequence.preserved.get("zoidiumControls.nextId"),
        Some(&Value::from(5))
    );
    assert_eq!(project.sequence.controls.len(), 4);

    let c1 = &project.sequence.controls[0];
    assert_eq!(c1.id, "c1");
    assert_eq!(c1.name, "Master Gain");
    assert_eq!(c1.kind, SequenceControlKind::Slider);
    assert_eq!(c1.track.keyframes.len(), 2);
    assert_eq!(c1.track.expression.as_deref(), Some("value * 2"));
    assert_eq!(c1.preserved.get("max"), Some(&Value::from(100)));

    let c2 = &project.sequence.controls[1];
    assert_eq!(c2.id, "c2");
    assert_eq!(c2.name, "Aim");
    assert_eq!(c2.kind, SequenceControlKind::Point2D);
    assert_eq!(c2.track.value, PropertyValue::Vec2([12.5, -4.0]));
    assert_eq!(c2.track.channels.len(), 2);

    let c3 = &project.sequence.controls[2];
    assert_eq!(c3.id, "c3");
    assert_eq!(c3.kind, SequenceControlKind::Point3D);
    assert_eq!(c3.track.value, PropertyValue::Vec3([1.0, 2.0, 3.0]));

    let c4 = &project.sequence.controls[3];
    assert_eq!(c4.id, "c4");
    assert_eq!(c4.kind, SequenceControlKind::Checkbox);
    assert_eq!(c4.track.value, PropertyValue::Option(1));
}

#[test]
fn imports_tar_with_pax_global_header_pz_prefix_semver_plugins_and_asset_kinds() {
    use std::io::Write as _;

    let png_bytes = b"\x89PNG\r\n\x1a\nFAKEPNG";
    let font_bytes = b"\x00\x01\x00\x00FAKETTF";
    let audio_bytes = b"OggSFAKEOGG";
    let key_png = crate::sha256::cm3_asset_key(png_bytes);
    let key_font = crate::sha256::cm3_asset_key(font_bytes);
    let key_audio = crate::sha256::cm3_asset_key(audio_bytes);

    let meta_bytes = br#"{"version":"1.0.102"}"#;
    let project_json = serde_json::json!({
        "plugins": [
            { "id": "camera-plus", "name": "Camera+", "features": ["dof", "vibrate"] },
            { "id": "material-plus", "materials": ["pbrplus", "matcap"] }
        ],
        "assets": {
            &key_png: { "source": 1, "external": false, "filename": "hero.png", "size": png_bytes.len(), "sha256": &key_png },
            &key_font: { "source": 1, "external": false, "filename": "blob_without_ext", "size": font_bytes.len(), "sha256": &key_font },
            &key_audio: { "source": 1, "external": false, "filename": "track_no_ext", "size": audio_bytes.len(), "sha256": &key_audio },
            "deadbeefdeadbeefdeadbeefdeadbeef": { "source": 1, "external": false, "filename": "gone.png", "size": 10 }
        },
        "media": [
            {
                "properties": { "name": "Hero Image", "icon": "image" },
                "baseType": "image",
                "assets": [&key_png],
                "thumbnail": { "invalid": true }
            }
        ],
        "sequence": {
            "properties": { "name": "Sequence", "rate": 29.97, "markers": { "keyframes": [{ "frame": 10, "value": 5 }] } },
            "videoTracks": [{
                "type": 0,
                "zoidiumSolo": true,
                "zoidiumLocked": true,
                "_zoidiumLayerInputId": "track-1",
                "clips": [{
                    "start": 0,
                    "length": 60,
                    "properties": { "name": "ImgClip" },
                    "object": {
                        "type": 3,
                        "properties": {
                            "name": "Hero",
                            "blending": 25,
                            "childProperties": { "image": &key_png }
                        },
                        "effects": [{
                            "type": "bloom",
                            "zoidiumBypassed": true,
                            "properties": { "enabled": 1 }
                        }]
                    }
                }]
            }],
            "audioTracks": [{
                "type": 1,
                "enabled": false,
                "zoidiumSolo": false,
                "zoidiumLocked": true,
                "clips": [{
                    "start": 0,
                    "length": 60,
                    "properties": { "name": "BGM", "media": &key_audio },
                    "object": { "type": 0, "customTag": "kept", "properties": { "volume": 1.0 } }
                }]
            }]
        }
    });
    let project_bytes = serde_json::to_vec(&project_json).unwrap();
    let zoidium_json = serde_json::json!({
        "version": 1,
        "name": "Real Zoidium Project",
        "missingAssets": ["deadbeefdeadbeefdeadbeefdeadbeef"],
        "integrity": {
            "meta": { "size": meta_bytes.len(), "sha256": sha256_hex(meta_bytes) },
            "project": { "size": project_bytes.len(), "sha256": sha256_hex(&project_bytes) }
        }
    });
    let zoidium_bytes = serde_json::to_vec(&zoidium_json).unwrap();

    // Build TAR with a PAX global header (`g`) and `pz/` prefix on all entries.
    let encoder = GzEncoder::new(Vec::new(), Compression::default());
    let mut builder = tar::Builder::new(encoder);
    let mut pax = tar::Header::new_ustar();
    pax.set_entry_type(tar::EntryType::XGlobalHeader);
    let pax_body = b"20 comment=zoidium\n";
    pax.set_size(pax_body.len() as u64);
    pax.set_mode(0o644);
    pax.set_cksum();
    builder
        .append_data(&mut pax, "pax_global_header", &pax_body[..])
        .unwrap();

    for (path, data) in [
        ("pz/meta", meta_bytes.as_slice()),
        ("pz/project", project_bytes.as_slice()),
        ("pz/zoidium.json", zoidium_bytes.as_slice()),
        (&format!("pz/{key_png}"), png_bytes.as_slice()),
        (&format!("pz/{key_font}"), font_bytes.as_slice()),
        (&format!("pz/{key_audio}"), audio_bytes.as_slice()),
    ] {
        let mut h = tar::Header::new_gnu();
        h.set_size(data.len() as u64);
        h.set_mode(0o644);
        h.set_cksum();
        builder.append_data(&mut h, path, data).unwrap();
    }
    let mut inner = builder.into_inner().unwrap();
    inner.flush().unwrap();
    let archive = inner.finish().unwrap();

    let imported = import_pz_from_bytes(&archive).expect("import pz/ + PAX g archive");
    let proj = imported.project;

    // No integrity mismatch or unexpected missing asset warnings.
    assert!(
        imported.warnings.is_empty(),
        "warnings: {:?}",
        imported.warnings
    );
    assert_eq!(proj.name, "Real Zoidium Project");
    assert_eq!(proj.imported_from_pz_version, Some(102));
    assert_eq!(
        proj.preserved.get("imported_from_pz_version_str"),
        Some(&Value::String("1.0.102".to_owned()))
    );
    assert!(proj.plugin_requirements.contains("camera-plus"));
    assert!(proj.plugin_requirements.contains("material-plus"));

    // AssetKind inferred accurately even though JSON had no `"type"` key.
    let sha_png = sha256_hex(png_bytes);
    let sha_font = sha256_hex(font_bytes);
    let sha_audio = sha256_hex(audio_bytes);
    assert_eq!(proj.assets[&sha_png].kind, AssetKind::Image);
    assert_eq!(proj.assets[&sha_font].kind, AssetKind::Font);
    assert_eq!(proj.assets[&sha_audio].kind, AssetKind::Av);

    // Media item parsed with real CM3 keys, remapped asset ID, and non-string thumbnail stripped.
    assert_eq!(proj.media[0].name, "Hero Image");
    assert_eq!(proj.media[0].asset_id.as_deref(), Some(sha_png.as_str()));
    assert_eq!(proj.media[0].kind, AssetKind::Image);
    assert!(!proj.media[0].preserved.contains_key("thumbnail"));

    // Fractional FPS, numeric marker, track solo/lock, audio enabled=false -> muted=true, layer input id.
    assert_eq!(proj.sequence.fps, 30);
    assert!((proj.sequence.exact_fps.unwrap() - 29.97).abs() < 1e-9);
    assert_eq!(proj.sequence.markers[0].label, "5");
    assert!(proj.sequence.video_tracks[0].solo);
    assert!(proj.sequence.video_tracks[0].locked);
    assert_eq!(
        proj.sequence
            .track_layer_input_ids
            .get(&proj.sequence.video_tracks[0].id)
            .map(String::as_str),
        Some("track-1")
    );
    assert!(!proj.sequence.audio_tracks[0].enabled);
    assert!(proj.sequence.audio_tracks[0].muted);
    assert!(proj.sequence.audio_tracks[0].locked);
    assert_eq!(
        proj.sequence.audio_tracks[0].clips[0]
            .audio
            .preserved
            .get("object.customTag"),
        Some(&Value::from("kept"))
    );

    // Blending index 25 => CM3 BLEND_LINEARDODGE (after 23=INVERT, 24=INVERT_RGB).
    let clip = &proj.sequence.video_tracks[0].clips[0];
    assert_eq!(clip.layer.blending, "BLEND_LINEARDODGE");
    assert!(!clip.time_curve.animated);
    assert!(
        proj.sequence
            .bypassed_effects
            .contains(&clip.layer.effects[0].id)
    );
}

#[test]
fn imports_precomp_plus_shape_plus_zoidium_masks_and_camera_layer() {
    let font_bytes = b"OTTOFAKEFONT";
    let tex_bytes = b"\x89PNG\r\n\x1a\nCOMPIMG";
    let key_font = crate::sha256::cm3_asset_key(font_bytes);
    let key_tex = crate::sha256::cm3_asset_key(tex_bytes);

    let project_json = serde_json::json!({
        "assets": {
            &key_font: { "source": 1, "external": false, "filename": "font.otf" },
            &key_tex: { "source": 1, "external": false, "filename": "comp.png" }
        },
        "media": [{
            "baseType": "track",
            "isComp": true,
            "compLength": 90,
            "properties": { "name": "Nested Comp" },
            "comp": { "id": "comp-hero-1", "clipLinks": null },
            "data": [{
                "type": 0,
                "clips": [{
                    "start": 0,
                    "length": 90,
                    "properties": { "name": "CompImg" },
                    "object": {
                        "type": 3,
                        "properties": { "name": "InnerImg", "childProperties": { "image": &key_tex } }
                    }
                }]
            }]
        }],
        "sequence": {
            "properties": { "name": "Main", "resolution": [1920, 1080], "rate": 30 },
            "videoTracks": [{
                "type": 0,
                "clips": [
                    {
                        "start": 0,
                        "length": 60,
                        "properties": { "name": "PrecompClip" },
                        "object": {
                            "type": 2,
                            "compId": "comp-hero-1",
                            "offset": 15,
                            "properties": { "name": "Precomp Layer", "blending": { "value": "blend_screen" } },
                            "objects": []
                        }
                    },
                    {
                        "start": -10,
                        "length": 50,
                        "properties": { "name": "ShapePlusClip" },
                        "object": {
                            "type": "zoidium:shape-plus/shape-layer",
                            "schemaVersion": 1,
                            "properties": { "name": "Vector+", "depth": { "animated": true, "keyframes": [{ "frame": 15, "value": 25.0 }] } },
                            "contents": [{
                                "id": "t1",
                                "kind": "text",
                                "properties": {
                                    "font": &key_font,
                                    "opacity": { "animated": true, "keyframes": [{ "frame": 20, "value": 0.5 }] }
                                }
                            }],
                            "zoidiumMasks": [{
                                "id": "m1",
                                "kind": "path",
                                "mode": "subtract",
                                "feather": 12.5,
                                "expansion": -4.0,
                                "opacity": 0.8,
                                "invert": 1,
                                "vertexIds": ["v1"],
                                "vertexModes": { "v1": "smooth" },
                                "properties": {
                                    "name": "Mask 1",
                                    "enabled": 1,
                                    "closed": 1,
                                    "path": {
                                        "animated": true,
                                        "keyframes": [{ "frame": 15, "value": [10, 20, 0, 0, 0, 0], "tween": 1 }]
                                    }
                                }
                            }]
                        }
                    },
                    {
                        "start": -5,
                        "length": 45,
                        "properties": { "name": "CamLayerClip" },
                        "object": {
                            "type": 9,
                            "properties": { "name": "Camera Layer" },
                            "objects": [{
                                "type": "zoidium:camera-plus/camera",
                                "schemaVersion": 1,
                                "properties": {
                                    "name": "Hero Cam+",
                                    "active": 1,
                                    "position": { "animated": true, "keyframes": [{ "frame": 10, "value": [1, 2, 90] }] },
                                    "projection": "perspective",
                                    "focalLength": { "animated": true, "keyframes": [{ "frame": 10, "value": 50.0 }] },
                                    "filmGate": 24.89,
                                    "zoom": 1.25,
                                    "filmOffsetX": 5.0,
                                    "filmOffsetY": -3.0,
                                    "depthOfField": {
                                        "enabled": 1,
                                        "focusDistance": 120.0,
                                        "aperture": 6.5,
                                        "focusAreaWidth": 20.0,
                                        "nearBlurLevel": 150.0,
                                        "farBlurLevel": 80.0
                                    },
                                    "vibrate": {
                                        "enabled": 1,
                                        "regularPulse": 1,
                                        "relative": 1,
                                        "seed": 42,
                                        "enablePosition": 1,
                                        "positionAmplitude": [2, 3, 4],
                                        "positionFrequency": 3.5,
                                        "enableRotation": 1,
                                        "rotationAmplitude": [5, 6, 7],
                                        "rotationFrequency": 1.5
                                    },
                                    "motionBlur": { "enabled": 1, "samples": 6, "shutter": 0.75 }
                                }
                            }]
                        }
                    }
                ]
            }]
        }
    });

    let archive = build_pz_bytes(&[
        ("project", &serde_json::to_vec(&project_json).unwrap()),
        (&key_font, font_bytes),
        (&key_tex, tex_bytes),
    ]);
    let proj = import_pz_from_bytes(&archive).unwrap().project;

    assert!(proj.plugin_requirements.contains("precomp-plus"));
    assert!(proj.plugin_requirements.contains("shape-plus"));
    assert!(proj.plugin_requirements.contains("camera-plus"));

    // Precomp+ composition imported, asset key remapped inside composition, and PrecompRef linked + offset stored.
    let comp_uuid = proj.composition_ids["comp-hero-1"];
    let comp_seq = &proj.compositions[&comp_uuid];
    assert_eq!(comp_seq.name, "Nested Comp");
    let LayerKind::Image { ref image } = comp_seq.video_tracks[0].clips[0].layer.kind else {
        panic!("expected Image in comp");
    };
    assert_eq!(image.as_deref(), Some(sha256_hex(tex_bytes).as_str()));

    let precomp_clip = &proj.sequence.video_tracks[0].clips[0];
    assert_eq!(
        precomp_clip.layer.kind,
        LayerKind::PrecompRef {
            composition_id: comp_uuid
        }
    );
    assert_eq!(
        proj.sequence.precomp_offsets.get(&precomp_clip.layer.id),
        Some(&15.0)
    );
    assert_eq!(precomp_clip.layer.blending, "BLEND_SCREEN");

    // Shape+ layer unquoted plugin_type, remapped font inside contents[], keyframes shifted by -10,
    // and zoidiumMasks parsed + shifted by -10!
    let shape_clip = &proj.sequence.video_tracks[0].clips[1];
    let LayerKind::Plugin {
        ref plugin_type,
        ref payload,
    } = shape_clip.layer.kind
    else {
        panic!("expected Shape+ Plugin layer");
    };
    assert_eq!(plugin_type, "zoidium:shape-plus/shape-layer");
    assert_eq!(
        payload["contents"][0]["properties"]["font"].as_str(),
        Some(sha256_hex(font_bytes).as_str())
    );
    assert_eq!(
        payload["contents"][0]["properties"]["opacity"]["keyframes"][0]["frame"].as_f64(),
        Some(10.0)
    );
    let p3d = &proj.sequence.layer_3d_props[&shape_clip.layer.id];
    assert!((p3d.depth.keyframes[0].frame - 5.0).abs() < 1e-9);

    let masks = &proj.sequence.layer_masks[&shape_clip.layer.id];
    assert_eq!(masks.len(), 1);
    assert_eq!(masks[0].id, "m1");
    assert_eq!(masks[0].mode, MaskMode::Subtract);
    assert!(masks[0].invert);
    assert!(masks[0].closed);
    assert!((masks[0].feather - 12.5).abs() < 1e-9);
    assert!((masks[0].expansion - (-4.0)).abs() < 1e-9);
    assert!((masks[0].path.keyframes[0].frame - 5.0).abs() < 1e-9);

    // CameraLayer (type 9) preserved full Object3D in camera_layer_objects, parsed Camera+ fields, shifted by -5!
    let cam_clip = &proj.sequence.video_tracks[0].clips[2];
    let cam_objs = &proj.sequence.camera_layer_objects[&cam_clip.layer.id];
    assert_eq!(cam_objs.len(), 1);
    assert_eq!(cam_objs[0].name, "Hero Cam+");
    assert!((cam_objs[0].position.channels[0].keyframes[0].frame - 5.0).abs() < 1e-9);
    let Object3DKind::Camera(ref cam) = cam_objs[0].kind else {
        panic!("expected Camera+ Object3D");
    };
    assert!((cam.focal_length.keyframes[0].frame - 5.0).abs() < 1e-9);
    assert!((cam.film_gate - 24.89).abs() < 1e-9);
    assert!((cam.zoom.value.as_f64().unwrap() - 1.25).abs() < 1e-9);
    assert!((cam.film_offset_x.value.as_f64().unwrap() - 5.0).abs() < 1e-9);
    let dof = cam.dof.as_ref().expect("dof");
    assert!(dof.enabled);
    assert!((dof.aperture.value.as_f64().unwrap() - 6.5).abs() < 1e-9);
    assert!((dof.near_blur.value.as_f64().unwrap() - 150.0).abs() < 1e-9);
    let vib = cam.vibrate.as_ref().expect("vibrate");
    assert!(
        vib.enabled
            && vib.regular_pulse
            && vib.relative
            && vib.enable_position
            && vib.enable_rotation
    );
    assert_eq!(vib.seed, 42);
    assert!((vib.position_frequency.value.as_f64().unwrap() - 3.5).abs() < 1e-9);
    assert!((vib.rotation_frequency.value.as_f64().unwrap() - 1.5).abs() < 1e-9);
    let mb = cam.motion_blur.as_ref().expect("motion_blur");
    assert!(mb.enabled && mb.samples == 6);
}

#[test]
fn imports_string_typed_3d_objects_compat_flags_matcap_cast_shadows_and_constraints() {
    let matcap_bytes = b"\x89PNG\r\n\x1a\nMATCAP";
    let ao_bytes = b"\x89PNG\r\n\x1a\nAOMAP";
    let front_bytes = b"\x89PNG\r\n\x1a\nFRONT";
    let node_tex_bytes = b"\x89PNG\r\n\x1a\nNODETEX";
    let key_mc = crate::sha256::cm3_asset_key(matcap_bytes);
    let key_ao = crate::sha256::cm3_asset_key(ao_bytes);
    let key_front = crate::sha256::cm3_asset_key(front_bytes);
    let key_node = crate::sha256::cm3_asset_key(node_tex_bytes);

    let project_json = serde_json::json!({
        "assets": {
            &key_mc: { "source": 1, "external": false, "filename": "mc.png" },
            &key_ao: { "source": 1, "external": false, "filename": "ao.png" },
            &key_front: { "source": 1, "external": false, "filename": "front.png" },
            &key_node: { "source": 1, "external": false, "filename": "node.png" }
        },
        "sequence": {
            "properties": { "name": "Scene3DParity", "rate": 30 },
            "videoTracks": [{
                "type": 0,
                "clips": [{
                    "start": 0,
                    "length": 60,
                    "properties": { "name": "SceneClip" },
                    "object": {
                        "type": 4,
                        "zoidiumShadowMode": "current",
                        "properties": {
                            "name": "MainScene",
                            "childProperties": {
                                "zoidiumBackdropMode": 1,
                                "zoidiumBackdropColor": [0.1, 0.2, 0.3],
                                "zoidiumBackdropOpacity": 0.75
                            }
                        },
                        "objects": [
                            {
                                "type": 1,
                                "zoidiumTextScaleMode": "current",
                                "properties": {
                                    "name": "TextPlusProps",
                                    "text": "HELLO",
                                    "spacing": 4.5,
                                    "bevelSide": 1,
                                    "bevelDetail": 6,
                                    "bevelProfile": 0,
                                    "bevelTension": 0.25
                                },
                                "material": {
                                    "type": "pbrplus",
                                    "zoidiumUvMode": "current",
                                    "properties": {
                                        "normalScale": { "animated": true, "keyframes": [{ "frame": 0, "value": 1.8 }] },
                                        "reflectionIntensity": 0.65,
                                        "emissiveIntensity": 2.5,
                                        "alphaTest": 0.4,
                                        "reflection": 1,
                                        "aoTexture": &key_ao
                                    }
                                }
                            },
                            {
                                "type": 3,
                                "objectType": 1,
                                "properties": {
                                    "name": "NoShadowSpot",
                                    "zoidiumCastShadows": 0,
                                    "zoidiumTrackTo": "hero-target"
                                }
                            },
                            {
                                "type": "zoidium:geometry-plus/rounded-box",
                                "properties": { "name": "RBox", "zoidiumTrackTo": "hero-target", "zoidiumFollow": "hero-target" },
                                "material": {
                                    "type": "matcap",
                                    "properties": { "matcap": &key_mc, "brightness": 1.4 }
                                }
                            },
                            {
                                "type": "zoidium:geometry-plus/polyhedron",
                                "properties": { "name": "Ico", "shape": 2, "radius": 15.0, "detail": 1 },
                                "material": {
                                    "type": "uvcustom",
                                    "properties": { "texture": &key_front }
                                }
                            },
                            {
                                "type": "zoidium:geometry-plus/cone",
                                "properties": { "name": "OpenCone", "bottomRadius": 12.0, "height": 20.0, "segments": 18, "endCaps": 1 },
                                "material": {
                                    "type": "nodes",
                                    "properties": {
                                        "bakeResolution": 3,
                                        "graph": serde_json::json!({ "nodes": [{ "params": { "tex": &key_node } }] }).to_string()
                                    }
                                }
                            },
                            {
                                "type": "zoidium:text-plus/character-material",
                                "properties": { "name": "CharMatGrp" },
                                "frontMaterial": { "type": "texture", "properties": { "texture": &key_front } },
                                "objects": [{ "type": 1, "properties": { "name": "ChildText", "text": "A" } }]
                            },
                            {
                                "type": 7,
                                "properties": { "name": "LegacyTwist", "angle": 1.5 },
                                "objects": [{ "type": 0, "objectType": 1, "properties": { "name": "TwistedBox" } }]
                            },
                            {
                                "type": "zoidium:repeater/echo-repeater",
                                "properties": { "name": "EchoRep" },
                                "repeaterProperties": { "count": 4, "delay": 3 },
                                "objects": [{ "type": 0, "objectType": 5, "properties": { "name": "EchoSphere" } }]
                            },
                            {
                                "type": "zoidium:trapcode-suite/area-light",
                                "properties": { "name": "StudioArea", "width": 25.0, "height": 15.0, "distance": 500.0 }
                            },
                            {
                                "type": 4,
                                "_zoidiumParticlesPlusSprite": "spark",
                                "properties": { "name": "Sparks", "number": 100 }
                            }
                        ]
                    }
                }]
            }]
        }
    });

    let archive = build_pz_bytes(&[
        ("project", &serde_json::to_vec(&project_json).unwrap()),
        (&key_mc, matcap_bytes),
        (&key_ao, ao_bytes),
        (&key_front, front_bytes),
        (&key_node, node_tex_bytes),
    ]);
    let imported = import_pz_from_bytes(&archive).unwrap();
    let proj = imported.project;
    assert!(
        imported.warnings.is_empty(),
        "warnings: {:?}",
        imported.warnings
    );

    for req in [
        "text-plus",
        "material-plus",
        "geometry-plus",
        "constraints",
        "repeater",
        "trapcode-suite",
        "particles-plus",
    ] {
        assert!(
            proj.plugin_requirements.contains(req),
            "missing plugin {req}"
        );
    }

    let scene_layer = &proj.sequence.video_tracks[0].clips[0].layer;
    let bd = &proj.sequence.scene_backdrops[&scene_layer.id];
    assert_eq!(bd.mode, 1);
    assert_eq!(bd.color, [0.1, 0.2, 0.3]);

    let LayerKind::Scene3D { ref objects, .. } = scene_layer.kind else {
        panic!("expected Scene3D");
    };

    // 0: Text3D top-level zoidiumTextScaleMode="current" + 7 Text+ props + PBR+ top-level zoidiumUvMode="current" + aoTexture remapped
    assert_eq!(objects[0].zoidium_uv_mode, CompatMode::Current);
    let Object3DKind::Text3D(ref t3d) = objects[0].kind else {
        panic!("expected Text3D");
    };
    assert_eq!(t3d.text_scale_mode, CompatMode::Current);
    assert!((t3d.spacing - 4.5).abs() < 1e-9);
    assert_eq!(t3d.bevel_side, 1);
    assert_eq!(t3d.bevel_detail, 6);
    assert_eq!(t3d.bevel_profile, 0);
    let Some(MaterialSpec::PbrPlus(ref pbr)) = objects[0].material else {
        panic!("expected PbrPlus");
    };
    assert_eq!(
        pbr.ao_texture.as_deref(),
        Some(sha256_hex(ao_bytes).as_str())
    );
    assert!(pbr.normal_scale_track.is_some());
    assert!((pbr.normal_scale - 1.8).abs() < 1e-9);
    assert!((pbr.emissive_intensity.value.as_f64().unwrap() - 2.5).abs() < 1e-9);
    assert!((pbr.alpha_test - 0.4).abs() < 1e-9);
    assert!(pbr.reflection);

    // 1: Light zoidiumCastShadows=0 + Camera/Light default track_axis=5 (-Z) & influence=100.0
    let Object3DKind::Light(ref spot) = objects[1].kind else {
        panic!("expected Light");
    };
    assert!(!spot.cast_shadow);
    let light_tt = objects[1].constraints.track_to.as_ref().unwrap();
    assert_eq!(light_tt.track_axis, 5);
    assert!((light_tt.influence.value.as_f64().unwrap() - 100.0).abs() < 1e-9);

    // 2: Geometry+ RoundedBox + Matcap "matcap" key remapped + non-camera default track_axis=2 & follow_scale=true
    assert!(matches!(
        objects[2].kind,
        Object3DKind::Shape3D(Shape3DKind::RoundedBox {
            size: [40.0, 24.0, 6.0],
            ..
        })
    ));
    let Some(MaterialSpec::Matcap(ref mc)) = objects[2].material else {
        panic!("expected Matcap");
    };
    assert_eq!(
        mc.texture.as_deref(),
        Some(sha256_hex(matcap_bytes).as_str())
    );
    assert_eq!(
        objects[2].constraints.track_to.as_ref().unwrap().track_axis,
        2
    );
    assert!(objects[2].constraints.follow.as_ref().unwrap().follow_scale);

    // 3: Geometry+ Polyhedron shape=2 => Icosahedron + uvcustom material remapped
    assert!(matches!(
        objects[3].kind,
        Object3DKind::Shape3D(Shape3DKind::Icosahedron {
            radius: 15.0,
            detail: 1,
            ..
        })
    ));
    let Some(MaterialSpec::CustomPbr(ref uvc)) = objects[3].material else {
        panic!("expected CustomPbr(uvcustom)");
    };
    assert!(uvc.uv_normalized);
    assert_eq!(
        uvc.texture.as_deref(),
        Some(sha256_hex(front_bytes).as_str())
    );

    // 4: Geometry+ Cone endCaps=1 => open_ended=true + NodeMaterial bakeResolution=3 => 1024 + graph JSON string remapped
    assert!(matches!(
        objects[4].kind,
        Object3DKind::Shape3D(Shape3DKind::Cone {
            open_ended: true,
            detail: 18,
            ..
        })
    ));
    let Some(MaterialSpec::NodeMaterial {
        resolution,
        ref graph,
        ..
    }) = objects[4].material
    else {
        panic!("expected NodeMaterial");
    };
    assert_eq!(resolution, 1024);
    assert!(
        graph["properties"]["graph"]
            .as_str()
            .unwrap()
            .contains(&sha256_hex(node_tex_bytes))
    );

    // 5: Text+ character-material Group preserves child objects AND remapped frontMaterial slot
    let Object3DKind::Group { ref children, .. } = objects[5].kind else {
        panic!("expected Group");
    };
    assert_eq!(children.len(), 1);
    let Some(MaterialSpec::Texture(fm)) = objects[5].material_slots.get("front") else {
        panic!("expected front material slot");
    };
    assert_eq!(
        fm.texture.as_deref(),
        Some(sha256_hex(front_bytes).as_str())
    );

    // 6: Legacy Effector+ type 7 (Twist) preserves children & defaults zoidiumSubdivisionMode="legacy"
    assert_eq!(
        objects[6].preserved.get("zoidiumPluginObjectType"),
        Some(&Value::String("zoidium:repeater/twist".to_owned()))
    );
    assert_eq!(
        objects[6].preserved.get("zoidiumSubdivisionMode"),
        Some(&Value::String("legacy".to_owned()))
    );
    assert!(
        matches!(objects[6].kind, Object3DKind::Group { ref children, .. } if children.len() == 1)
    );

    // 7: Repeater echo-repeater preserves children
    assert!(
        matches!(objects[7].kind, Object3DKind::Group { ref children, .. } if children.len() == 1)
    );

    // 8: Trapcode area-light parsed as Light with distance=500.0
    let Object3DKind::Light(ref area) = objects[8].kind else {
        panic!("expected Trapcode Light");
    };
    assert!((area.distance - 500.0).abs() < 1e-9);
}

#[test]
fn handles_legacy_semver_upgrades_flat_camera_plus_geometry_plus_extras_and_blend_order() {
    // Verify exact CM3 blend mode index order & case-insensitive normalization.
    assert_eq!(cm3_blend_mode_from_index(23), Some("BLEND_INVERT"));
    assert_eq!(cm3_blend_mode_from_index(24), Some("BLEND_INVERT_RGB"));
    assert_eq!(cm3_blend_mode_from_index(25), Some("BLEND_LINEARDODGE"));
    assert_eq!(cm3_blend_mode_from_index(34), Some("BLEND_LUMINOSITY"));
    assert_eq!(cm3_blend_mode_from_index(35), Some("BLEND_PLUS"));
    assert_eq!(
        normalize_blend_mode_token("blend_multiply"),
        "BLEND_MULTIPLY"
    );
    assert_eq!(
        normalize_blend_mode_token("BLEND_COLOR_DODGE"),
        "BLEND_COLORDODGE"
    );
    assert_eq!(normalize_blend_mode_token("blend_add"), "BLEND_PLUS");

    // Legacy CM3 1.0.21 project with pre-core64 clip offset, pre-core43 top-level media,
    // pre-core60 media.title, legacy flat type:6 Camera+, and Geometry+ capsule/tube/gear/helix/spline.
    let project_json = serde_json::json!({
        "media": [{ "title": "Old Media Title", "icon": "film" }],
        "sequence": {
            "properties": { "name": "OldSeq", "rate": 30 },
            "videoTracks": [{
                "clips": [{
                    "start": 0,
                    "length": 60,
                    "offset": 30,
                    "name": "OldClip",
                    "object": {
                        "type": 4,
                        "properties": { "name": "OldScene" },
                        "objects": [
                            {
                                "type": 6,
                                "objectType": 1,
                                "properties": {
                                    "name": "LegacyCamPlus",
                                    "focalLength": 50.0,
                                    "dof": 1,
                                    "dofAperture": 5.0,
                                    "dofFocusDistance": 95.0
                                }
                            },
                            {
                                "type": "zoidium:geometry-plus/capsule",
                                "properties": { "name": "Cap", "radius": 6.0, "height": 30.0, "segments": 10, "radialSegments": 20 }
                            },
                            {
                                "type": "zoidium:geometry-plus/tube",
                                "properties": { "name": "MyTube", "outerRadius": 12.0, "innerRadius": 8.0 },
                                "material": { "type": "singlecolor", "properties": { "color": [0.2, 0.3, 0.4] } }
                            },
                            { "type": "zoidium:geometry-plus/gear", "properties": { "name": "MyGear" } },
                            { "type": "zoidium:geometry-plus/helix", "properties": { "name": "MyHelix" } },
                            { "type": "zoidium:geometry-plus/spline", "properties": { "name": "MySpline" } }
                        ]
                    }
                }]
            }],
            "audioTracks": []
        }
    });
    let archive = build_pz_bytes(&[
        ("meta", br#"{"version":"1.0.21"}"#),
        ("project", &serde_json::to_vec(&project_json).unwrap()),
    ]);
    let imported = import_pz_from_bytes(&archive).unwrap();
    let proj = imported.project;
    assert!(
        imported.warnings.is_empty(),
        "warnings: {:?}",
        imported.warnings
    );
    assert_eq!(proj.imported_from_pz_version, Some(21));
    assert_eq!(proj.media[0].name, "Old Media Title");

    let clip = &proj.sequence.video_tracks[0].clips[0];
    assert_eq!(clip.name, "OldClip");
    // Pre-core64 offset=30 @ 30fps => time_curve starts at 1.0s and ends at 3.0s, animated=false.
    assert!(!clip.time_curve.animated);
    assert!((clip.time_curve.keyframes[0].value.as_f64().unwrap() - 1.0).abs() < 1e-9);
    assert!((clip.time_curve.keyframes[1].value.as_f64().unwrap() - 3.0).abs() < 1e-9);

    let LayerKind::Scene3D { ref objects, .. } = clip.layer.kind else {
        panic!("expected Scene3D");
    };
    let Object3DKind::Camera(ref cam) = objects[0].kind else {
        panic!("expected Camera");
    };
    assert!((cam.focal_length.value.as_f64().unwrap() - 50.0).abs() < 1e-9);
    let dof = cam.dof.as_ref().expect("legacy flat dof migrated");
    assert!(dof.enabled);
    assert!((dof.aperture.value.as_f64().unwrap() - 5.0).abs() < 1e-9);
    assert!((dof.focus_distance.value.as_f64().unwrap() - 95.0).abs() < 1e-9);

    // Capsule total height=30, radius=6 => straight length = 30 - 12 = 18.
    assert!(matches!(
        objects[1].kind,
        Object3DKind::Shape3D(Shape3DKind::Capsule {
            radius: 6.0,
            length: 18.0,
            cap_segments: 10,
            radial_segments: 20,
            ..
        })
    ));
    // Tube, Gear, Helix, Spline all import natively as Shape3D (never Dummy!) and preserve material.
    assert!(matches!(
        objects[2].kind,
        Object3DKind::Shape3D(Shape3DKind::Tube {
            outer_radius: 12.0,
            inner_radius: 8.0,
            ..
        })
    ));
    assert!(matches!(
        objects[2].material,
        Some(MaterialSpec::SingleColor { .. })
    ));
    assert!(matches!(
        objects[3].kind,
        Object3DKind::Shape3D(Shape3DKind::Gear { teeth: 12, .. })
    ));
    assert!(matches!(
        objects[4].kind,
        Object3DKind::Shape3D(Shape3DKind::Helix { .. })
    ));
    assert!(matches!(
        objects[5].kind,
        Object3DKind::Shape3D(Shape3DKind::Spline { ref points, .. }) if points.len() == 3
    ));

    // Verify rebase_time_curve_for_clipped_prefix respects Hold (tween=0) & EaseIn (tween=2).
    let mut hold_curve = PropertyTrack {
        animated: true,
        expression: None,
        static_behavior: StaticBehavior::Linear,
        value: PropertyValue::Number(0.0),
        keyframes: vec![
            PropertyKeyframe::from_cm3_tween(
                0.0,
                PropertyValue::Number(2.0),
                1,
                [[-10.0, 0.0], [10.0, 0.0]],
                true,
            ),
            PropertyKeyframe::from_cm3_tween(
                20.0,
                PropertyValue::Number(8.0),
                0,
                [[-10.0, 0.0], [10.0, 0.0]],
                true,
            ),
        ],
        channels: Vec::new(),
    };
    rebase_time_curve_for_clipped_prefix(&mut hold_curve, 10.0, 10.0, 30);
    // Hold segment [0..20] holds value 2.0 at frame 10 (not linear midpoint 5.0!).
    assert!((hold_curve.keyframes[0].value.as_f64().unwrap() - 2.0).abs() < 1e-9);
}

#[test]
fn qa_pz_import_particles_defaults_and_radial_grid_repeaters() {
    let pz_json = serde_json::json!({
        "version": 3,
        "name": "ParticlesAndRepeaters",
        "sequence": {
            "properties": { "name": "ParticlesAndRepeaters", "rate": 30 },
            "videoTracks": [{
                "clips": [{
                    "start": 0,
                    "length": 90,
                    "properties": { "name": "Scene" },
                    "object": {
                        "type": 4,
                        "properties": { "name": "Scene" },
                        "objects": [
                            {
                                "type": 4,
                                "properties": { "name": "SparseParticles" }
                            },
                            {
                                "type": 4,
                                "properties": {
                                    "name": "ExplicitParticles",
                                    "number": 42,
                                    "rate": 10.0,
                                    "lifetime": 1.5
                                }
                            },
                            {
                                "type": "zoidium:repeater/radial",
                                "properties": { "name": "Radial", "count": 8, "radius": 28.0 },
                                "objects": [
                                    { "type": 0, "objectType": 1, "properties": { "name": "ChildBox" } }
                                ]
                            },
                            {
                                "type": "zoidium:repeater/grid",
                                "properties": { "name": "Grid", "countX": 3, "countY": 3, "countZ": 1 },
                                "objects": [
                                    { "type": 0, "objectType": 3, "properties": { "name": "ChildSphere" } }
                                ]
                            }
                        ]
                    }
                }]
            }]
        }
    });

    let archive = build_pz_bytes(&[
        ("meta", br#"{"version":"1.0.102"}"#),
        ("project", &serde_json::to_vec(&pz_json).unwrap()),
    ]);
    let report = import_pz_from_bytes(&archive).expect("import pz");
    assert!(
        report.warnings.is_empty(),
        "expected zero warnings importing radial/grid repeaters, got {:?}",
        report.warnings
    );
    let LayerKind::Scene3D { ref objects, .. } =
        report.project.sequence.video_tracks[0].clips[0].layer.kind
    else {
        panic!("expected Scene3D");
    };
    let Object3DKind::Particles(ref sparse) = objects[0].kind else {
        panic!("expected Particles");
    };
    assert_eq!(sparse.number, 100);
    assert!((sparse.rate - 25.0).abs() < 1e-9);
    assert!((sparse.lifetime - 2.0).abs() < 1e-9);

    let Object3DKind::Particles(ref explicit) = objects[1].kind else {
        panic!("expected explicit Particles");
    };
    assert_eq!(explicit.number, 42);
    assert!((explicit.rate - 10.0).abs() < 1e-9);
    assert!((explicit.lifetime - 1.5).abs() < 1e-9);

    assert_eq!(
        objects[2].preserved.get("zoidiumPluginObjectType"),
        Some(&Value::String("zoidium:repeater/radial".to_owned()))
    );
    let Object3DKind::Group { ref children, .. } = objects[2].kind else {
        panic!("expected Group for radial repeater");
    };
    assert_eq!(children.len(), 1);

    assert_eq!(
        objects[3].preserved.get("zoidiumPluginObjectType"),
        Some(&Value::String("zoidium:repeater/grid".to_owned()))
    );
    let Object3DKind::Group { ref children, .. } = objects[3].kind else {
        panic!("expected Group for grid repeater");
    };
    assert_eq!(children.len(), 1);
}

#[test]
fn effect_group_parses_effects_children_alias_and_optical_flare_alias_has_zero_warnings() {
    let pz_json = serde_json::json!({
        "version": 3,
        "name": "EffectGroupAndFlare",
        "sequence": {
            "properties": { "name": "EffectGroupAndFlare", "rate": 30 },
            "videoTracks": [{
                "clips": [{
                    "start": 0,
                    "length": 60,
                    "properties": { "name": "Scene" },
                    "object": {
                        "type": 4,
                        "properties": { "name": "Scene" },
                        "objects": [
                            {
                                "type": "zoidium:optical-flares/optical-flare",
                                "properties": { "name": "Hero Flare" }
                            }
                        ],
                        "effects": [
                            {
                                "type": 0,
                                "properties": { "name": "Community Group" },
                                "effects": [
                                    { "type": "gaussianblur", "properties": { "name": "Inner Blur", "amount": 4.0 } },
                                    {
                                        "type": "group",
                                        "properties": { "name": "Nested Group" },
                                        "effects": [
                                            { "type": "bloom", "properties": { "name": "Deep Bloom" } }
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                }]
            }]
        }
    });

    let archive = build_pz_bytes(&[
        ("meta", br#"{"version":"1.0.102"}"#),
        ("project", &serde_json::to_vec(&pz_json).unwrap()),
    ]);
    let report = import_pz_from_bytes(&archive).expect("import pz");
    assert!(
        report.warnings.is_empty(),
        "expected zero warnings, got {:?}",
        report.warnings
    );
    assert!(
        report
            .project
            .plugin_requirements
            .contains("optical-flares")
    );

    let layer = &report.project.sequence.video_tracks[0].clips[0].layer;
    let LayerKind::Scene3D { ref objects, .. } = layer.kind else {
        panic!("expected Scene3D");
    };
    assert_eq!(
        objects[0].preserved.get("zoidiumPluginObjectType"),
        Some(&Value::String("zoidium:optical-flares/flare".to_owned()))
    );

    assert_eq!(layer.effects.len(), 1);
    let EffectKind::EffectGroup { ref children, .. } = layer.effects[0].kind else {
        panic!("expected EffectGroup");
    };
    assert_eq!(children.len(), 2);
    assert_eq!(children[0].name, "Inner Blur");
    assert_eq!(
        children[0].kind,
        EffectKind::Standard("gaussianblur".to_owned())
    );
    let EffectKind::EffectGroup {
        children: ref nested_children,
        ..
    } = children[1].kind
    else {
        panic!("expected nested EffectGroup");
    };
    assert_eq!(nested_children.len(), 1);
    assert_eq!(nested_children[0].name, "Deep Bloom");
    assert_eq!(
        nested_children[0].kind,
        EffectKind::Standard("bloom".to_owned())
    );
}
