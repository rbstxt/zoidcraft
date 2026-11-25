use super::*;
use crate::model::{
    CompatMode, CustomProperty, EffectKind, GradientStop, Layer, LayerKind, LayerMask, LightData,
    LightKind, MaskMode, MaterialSpec, Object3D, Object3DKind, ParticlesData, Project,
    PropertyKeyframe, PropertyTrack, PropertyValue, Sequence, SequenceControl, SequenceControlKind,
    Shape3DKind, VideoClip,
};
use flate2::Compression;
use serde_json::{Value, json};
use uuid::Uuid;

#[test]
fn round_trips_all_8_geometry_plus_primitives_layer_mask_metadata_and_particles_sprite() {
    let mut project = Project::new_cm3_default();
    let clip = &mut project.sequence.video_tracks[0].clips[0];
    let scene_id = clip.layer.id;

    if let LayerKind::Scene3D {
        ref mut objects, ..
    } = clip.layer.kind
    {
        // 1. RoundedBox (with corner/bevel/shading + animated bevelSize track preserved)
        let mut rbox = Object3D::new_shape_3d(
            "RBox",
            Shape3DKind::RoundedBox {
                size: [44.0, 26.0, 8.0],
                radius: 4.0,
                segments: 8,
                corner_radius: Some(4.0),
                corner_segments: Some(8),
                bevel_size: Some(0.9),
                bevel_segments: Some(3),
                shading: 1,
            },
        );
        rbox.preserved.insert(
            "property.bevelSize".to_owned(),
            json!({
                "animated": true,
                "keyframes": [
                    { "frame": 0, "value": 0.9, "tween": 1 },
                    { "frame": 30, "value": 2.4, "tween": 1 }
                ]
            }),
        );
        objects.push(rbox);

        // 2. Polyhedron (Dodecahedron with shading=1)
        objects.push(Object3D::new_shape_3d(
            "Poly",
            Shape3DKind::Dodecahedron {
                radius: 15.0,
                detail: 2,
                shading: 1,
            },
        ));

        // 3. Cone (frustum with top_radius + shading=1)
        objects.push(Object3D::new_shape_3d(
            "FrustumCone",
            Shape3DKind::Cone {
                radius: 18.0,
                height: 36.0,
                detail: 28,
                open_ended: true,
                top_radius: 5.0,
                shading: 1,
            },
        ));

        // 4. Capsule (with shading=1)
        objects.push(Object3D::new_shape_3d(
            "SmoothCapsule",
            Shape3DKind::Capsule {
                radius: 7.0,
                length: 20.0,
                cap_segments: 10,
                radial_segments: 24,
                shading: 1,
            },
        ));

        // 5. Tube
        objects.push(Object3D::new_shape_3d(
            "HollowTube",
            Shape3DKind::Tube {
                outer_radius: 14.0,
                inner_radius: 9.5,
                height: 28.0,
                radial_segments: 32,
                open_ended: true,
                shading: 1,
            },
        ));

        // 6. Gear
        objects.push(Object3D::new_shape_3d(
            "CogGear",
            Shape3DKind::Gear {
                teeth: 16,
                inner_radius: 11.0,
                outer_radius: 19.0,
                thickness: 7.5,
                tooth_width: 0.6,
                bevel_size: 0.4,
                bevel_segments: 3,
                shading: 1,
            },
        ));

        // 7. Helix
        objects.push(Object3D::new_shape_3d(
            "SpringHelix",
            Shape3DKind::Helix {
                radius: 16.0,
                height: 40.0,
                turns: 4.5,
                tube_radius: 2.5,
                path_segments: 96,
                radial_segments: 12,
                shading: 1,
            },
        ));

        // 8. Spline
        objects.push(Object3D::new_shape_3d(
            "ArchSpline",
            Shape3DKind::Spline {
                points: vec![
                    [-25.0, 0.0, 0.0],
                    [-8.0, 18.0, 5.0],
                    [8.0, -12.0, -5.0],
                    [25.0, 4.0, 0.0],
                ],
                interpolation: "Akima".to_owned(),
                closed: true,
                start: 0.1,
                end: 0.9,
                thickness: 3.5,
                thickness_map: vec![
                    GradientStop {
                        position: 0.0,
                        color: "rgba(255,255,255,1.0)".to_owned(),
                    },
                    GradientStop {
                        position: 1.0,
                        color: "rgba(128,128,128,1.0)".to_owned(),
                    },
                ],
                gradient_space: "Trimmed".to_owned(),
                tension: 1.25,
                path_segments: 24,
                radial_segments: 10,
                caps: "Start".to_owned(),
                cap_style: "Round".to_owned(),
                shading: 1,
            },
        ));

        // 9. Particles with Particles+ sprite_preset
        objects.push(Object3D::new(
            "Sparkles",
            Object3DKind::Particles(Box::new(ParticlesData {
                number: 100,
                rate: 20.0,
                lifetime: 1.5,
                sprite_preset: Some("star-4pt".to_owned()),
                ..ParticlesData::default()
            })),
        ));
    }

    // LayerMask with locked + outline_color, and one with default empty outline_color
    project.sequence.layer_masks.insert(
        scene_id,
        vec![
            LayerMask {
                id: "m-locked".to_owned(),
                name: "Locked Mask".to_owned(),
                enabled: true,
                mode: MaskMode::Add,
                invert: false,
                feather: 2.0,
                expansion: 1.5,
                opacity: 0.85,
                closed: true,
                locked: true,
                outline_color: "#ff7b72".to_owned(),
                vertex_ids: vec!["v1".to_owned()],
                vertex_modes: BTreeMap::new(),
                path: PropertyTrack::constant(PropertyValue::Json(json!([
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ]))),
                preserved: BTreeMap::new(),
            },
            LayerMask {
                id: "m-default-outline".to_owned(),
                name: "Default Outline".to_owned(),
                enabled: true,
                mode: MaskMode::Subtract,
                invert: false,
                feather: 0.0,
                expansion: 0.0,
                opacity: 1.0,
                closed: false,
                locked: false,
                outline_color: String::new(),
                vertex_ids: Vec::new(),
                vertex_modes: BTreeMap::new(),
                path: PropertyTrack::constant(PropertyValue::Json(json!([
                    1.0, 2.0, 0.0, 0.0, 0.0, 0.0
                ]))),
                preserved: BTreeMap::new(),
            },
        ],
    );

    let pz_bytes = export_pz_to_bytes(&project).expect("export wave 1a pz");
    let entries = inspect_tar_entries(&pz_bytes);
    let doc: Value = serde_json::from_slice(&entries["project"]).expect("parse project JSON");
    let scene_json = &doc["sequence"]["videoTracks"][0]["clips"][0]["object"];

    // Empty outline_color must NOT emit invalid empty string to Zoidium
    let masks_json = scene_json["zoidiumMasks"].as_array().expect("zoidiumMasks");
    assert_eq!(masks_json[0]["locked"], json!(true));
    assert_eq!(masks_json[0]["outlineColor"], json!("#ff7b72"));
    assert_eq!(masks_json[1]["locked"], json!(false));
    assert!(masks_json[1].get("outlineColor").is_none());

    // Animated bevelSize track on RoundedBox must not be clobbered on export
    assert_eq!(
        scene_json["objects"][1]["properties"]["bevelSize"]["animated"],
        json!(true)
    );
    // Particles+ sprite_preset synced to `_zoidiumParticlesPlusSprite`
    assert_eq!(
        scene_json["objects"][9]["_zoidiumParticlesPlusSprite"],
        json!("star-4pt")
    );

    let reimported = import_pz_from_bytes(&pz_bytes).expect("reimport wave 1a pz");
    assert!(
        reimported.warnings.is_empty(),
        "expected zero warnings, got {:?}",
        reimported.warnings
    );
    let rp = reimported.project;
    assert!(rp.plugin_requirements.contains("geometry-plus"));
    assert!(rp.plugin_requirements.contains("particles-plus"));

    let rscene = &rp.sequence.video_tracks[0].clips[0].layer;
    let rmasks = &rp.sequence.layer_masks[&rscene.id];
    assert_eq!(rmasks.len(), 2);
    assert!(rmasks[0].locked);
    assert_eq!(rmasks[0].outline_color, "#ff7b72");
    assert!(!rmasks[1].locked);
    assert!(rmasks[1].outline_color.is_empty());

    let LayerKind::Scene3D { ref objects, .. } = rscene.kind else {
        panic!("expected Scene3D");
    };
    assert!(matches!(
        objects[1].kind,
        Object3DKind::Shape3D(Shape3DKind::RoundedBox {
            size: [44.0, 26.0, 8.0],
            radius: 4.0,
            segments: 8,
            corner_radius: Some(4.0),
            corner_segments: Some(8),
            bevel_size: Some(0.9),
            bevel_segments: Some(3),
            shading: 1,
        })
    ));
    assert_eq!(
        objects[1]
            .preserved
            .get("property.bevelSize")
            .and_then(|v| v.get("animated")),
        Some(&json!(true))
    );
    assert!(matches!(
        objects[2].kind,
        Object3DKind::Shape3D(Shape3DKind::Dodecahedron {
            radius: 15.0,
            detail: 2,
            shading: 1,
        })
    ));
    assert!(matches!(
        objects[3].kind,
        Object3DKind::Shape3D(Shape3DKind::Cone {
            radius: 18.0,
            height: 36.0,
            detail: 28,
            open_ended: true,
            top_radius: 5.0,
            shading: 1,
        })
    ));
    assert!(matches!(
        objects[4].kind,
        Object3DKind::Shape3D(Shape3DKind::Capsule {
            radius: 7.0,
            length: 20.0,
            cap_segments: 10,
            radial_segments: 24,
            shading: 1,
        })
    ));
    assert!(matches!(
        objects[5].kind,
        Object3DKind::Shape3D(Shape3DKind::Tube {
            outer_radius: 14.0,
            inner_radius: 9.5,
            height: 28.0,
            radial_segments: 32,
            open_ended: true,
            shading: 1,
        })
    ));
    assert!(matches!(
        objects[6].kind,
        Object3DKind::Shape3D(Shape3DKind::Gear {
            teeth: 16,
            inner_radius: 11.0,
            outer_radius: 19.0,
            thickness: 7.5,
            tooth_width: 0.6,
            bevel_size: 0.4,
            bevel_segments: 3,
            shading: 1,
        })
    ));
    assert!(matches!(
        objects[7].kind,
        Object3DKind::Shape3D(Shape3DKind::Helix {
            radius: 16.0,
            height: 40.0,
            turns: 4.5,
            tube_radius: 2.5,
            path_segments: 96,
            radial_segments: 12,
            shading: 1,
        })
    ));
    let Object3DKind::Shape3D(Shape3DKind::Spline {
        ref points,
        ref interpolation,
        closed,
        start,
        end,
        thickness,
        ref thickness_map,
        ref gradient_space,
        tension,
        path_segments,
        radial_segments,
        ref caps,
        ref cap_style,
        shading,
    }) = objects[8].kind
    else {
        panic!("expected Spline, got {:?}", objects[8].kind);
    };
    assert_eq!(points.len(), 4);
    assert_eq!(points[1], [-8.0, 18.0, 5.0]);
    assert_eq!(interpolation, "Akima");
    assert!(closed);
    assert!((start - 0.1).abs() < 1e-9);
    assert!((end - 0.9).abs() < 1e-9);
    assert!((thickness - 3.5).abs() < 1e-9);
    assert_eq!(thickness_map.len(), 2);
    assert_eq!(gradient_space, "Trimmed");
    assert!((tension - 1.25).abs() < 1e-9);
    assert_eq!(path_segments, 24);
    assert_eq!(radial_segments, 10);
    assert_eq!(caps, "Start");
    assert_eq!(cap_style, "Round");
    assert_eq!(shading, 1);

    let Object3DKind::Particles(ref rpart) = objects[9].kind else {
        panic!("expected Particles");
    };
    assert_eq!(rpart.sprite_preset.as_deref(), Some("star-4pt"));
}

#[test]
fn expresstext_plus_optical_flares_and_trapcode_round_trip_with_zero_warnings() {
    let expresstext_types = [
        "zoidium:expresstext-plus/expresstext",
        "zoidium:expresstext-plus/shake",
        "zoidium:expresstext-plus/plain",
        "zoidium:expresstext-plus/random",
        "zoidium:expresstext-plus/formula",
        "zoidium:expresstext-plus/delay",
        "zoidium:expresstext-plus/spline",
        "zoidium:expresstext-plus/orbit",
        "zoidium:expresstext-plus/repeater",
        "zoidium:expresstext-plus/twist",
        "zoidium:expresstext-plus/wrap",
        "zoidium:expresstext-plus/voronoi",
    ];

    let mut scene_objects: Vec<Value> = expresstext_types
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            json!({
                "type": ty,
                "schemaVersion": 1,
                "properties": { "name": format!("ET_{i}") },
                "expressProperties": {
                    "lettersMode": 1,
                    "lettersVersion": 2,
                    "strength": { "animated": false, "keyframes": [{ "frame": 0, "value": 1.0, "tween": 1 }] }
                },
                "objects": [
                    { "type": 1, "properties": { "name": "ChildText", "text": "HI" } }
                ]
            })
        })
        .collect();

    // Optical Flares (numeric type 13 and string type)
    scene_objects.push(json!({
        "type": 13,
        "objectType": 2,
        "properties": { "name": "NumericFlare" },
        "stack": [{ "kind": "glow", "brightness": 1.2 }]
    }));
    scene_objects.push(json!({
        "type": "zoidium:optical-flares/flare",
        "properties": { "name": "StringFlare" }
    }));

    // Trapcode (both `zoidium:trapcode/*` and `zoidium:trapcode-suite/*`)
    for ty in [
        "zoidium:trapcode/particular",
        "zoidium:trapcode/form",
        "zoidium:trapcode/plexus",
        "zoidium:trapcode-suite/particular",
        "zoidium:trapcode-suite/form",
        "zoidium:trapcode-suite/plexus",
    ] {
        scene_objects.push(json!({
            "type": ty,
            "properties": { "name": ty }
        }));
    }

    let project_json = json!({
        "sequence": {
            "properties": { "name": "Seq", "rate": 30 },
            "videoTracks": [{
                "clips": [{
                    "start": 0,
                    "length": 60,
                    "name": "Clip",
                    "object": {
                        "type": 4,
                        "properties": { "name": "Scene" },
                        "objects": scene_objects
                    }
                }]
            }],
            "audioTracks": []
        }
    });

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    {
        let mut builder = tar::Builder::new(&mut encoder);
        let meta = br#"{"version":"1.0.101"}"#;
        let proj = serde_json::to_vec(&project_json).unwrap();
        append_ustar_file(&mut builder, "meta", meta).unwrap();
        append_ustar_file(&mut builder, "project", &proj).unwrap();
        builder.finish().unwrap();
    }
    let pz_bytes = encoder.finish().unwrap();

    let imported = import_pz_from_bytes(&pz_bytes).expect("import plugin objects");
    assert!(
        imported.warnings.is_empty(),
        "expected ZERO warnings for ExpressText+/OpticalFlares/Trapcode, got: {:?}",
        imported.warnings
    );
    assert!(
        imported
            .project
            .plugin_requirements
            .contains("expresstext-plus")
    );
    assert!(
        imported
            .project
            .plugin_requirements
            .contains("optical-flares")
    );
    assert!(
        imported
            .project
            .plugin_requirements
            .contains("trapcode-suite")
    );

    let LayerKind::Scene3D { ref objects, .. } = imported.project.sequence.video_tracks[0].clips[0]
        .layer
        .kind
    else {
        panic!("expected Scene3D");
    };
    assert_eq!(objects.len(), 12 + 2 + 6);

    // Verify all 12 ExpressText+ classes imported as Group with expressProperties & lettersMode preserved
    for (i, &ty) in expresstext_types.iter().enumerate() {
        assert!(
            matches!(objects[i].kind, Object3DKind::Group { ref children, .. } if children.len() == 1),
            "expected Group with 1 child for {ty}"
        );
        assert_eq!(
            objects[i].preserved.get("zoidiumPluginObjectType"),
            Some(&json!(ty))
        );
        assert_eq!(
            objects[i]
                .preserved
                .get("expressProperties")
                .and_then(|v| v.get("lettersMode")),
            Some(&json!(1))
        );
    }

    // Verify Optical Flares
    assert_eq!(
        objects[12].preserved.get("zoidiumPluginObjectType"),
        Some(&json!("zoidium:optical-flares/flare"))
    );
    assert_eq!(
        objects[13].preserved.get("zoidiumPluginObjectType"),
        Some(&json!("zoidium:optical-flares/flare"))
    );

    // Verify Trapcode
    for (offset, ty) in [
        "zoidium:trapcode/particular",
        "zoidium:trapcode/form",
        "zoidium:trapcode/plexus",
        "zoidium:trapcode-suite/particular",
        "zoidium:trapcode-suite/form",
        "zoidium:trapcode-suite/plexus",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(
            objects[14 + offset]
                .preserved
                .get("zoidiumPluginObjectType"),
            Some(&json!(ty))
        );
    }

    // Re-export and re-import to verify lossless round-trip with 0 warnings
    let reexported = export_pz_to_bytes(&imported.project).expect("reexport");
    let roundtripped = import_pz_from_bytes(&reexported).expect("roundtrip");
    assert!(
        roundtripped.warnings.is_empty(),
        "expected ZERO warnings after re-export, got: {:?}",
        roundtripped.warnings
    );
    let LayerKind::Scene3D { ref objects, .. } = roundtripped.project.sequence.video_tracks[0]
        .clips[0]
        .layer
        .kind
    else {
        panic!("expected Scene3D");
    };
    assert_eq!(
        objects[0]
            .preserved
            .get("expressProperties")
            .and_then(|v| v.get("lettersMode")),
        Some(&json!(1))
    );
    assert_eq!(
        objects[12].preserved.get("zoidiumPluginObjectType"),
        Some(&json!("zoidium:optical-flares/flare"))
    );
}

#[test]
fn shape_plus_payload_edits_survive_pz_round_trip() {
    let project_json = json!({
        "version": 3,
        "sequence": {
            "properties": {
                "name": "ShapePlusEditSeq",
                "resolution": [1920, 1080],
                "fps": 30,
                "length": 60
            },
            "videoTracks": [{
                "properties": { "name": "V1" },
                "clips": [{
                    "start": 0,
                    "length": 60,
                    "properties": { "name": "ShapePlusClip" },
                    "object": {
                        "type": "zoidium:shape-plus/shape-layer",
                        "schemaVersion": 1,
                        "renderVersion": 2,
                        "nextId": 7,
                        "zoidiumShapePlus": true,
                        "properties": { "name": "Hero Shape+" },
                        "contents": [{
                            "id": "p1",
                            "kind": "path",
                            "properties": {
                                "name": "Original Path",
                                "closed": 1,
                                "path": { "value": [0, 0, 10, 0, 10, 10] }
                            }
                        }]
                    }
                }]
            }]
        }
    });

    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    {
        let mut builder = tar::Builder::new(&mut encoder);
        let meta = br#"{"version":"1.0.101"}"#;
        let proj = serde_json::to_vec(&project_json).unwrap();
        append_ustar_file(&mut builder, "meta", meta).unwrap();
        append_ustar_file(&mut builder, "project", &proj).unwrap();
        builder.finish().unwrap();
    }
    let pz_bytes = encoder.finish().unwrap();

    let mut imported = import_pz_from_bytes(&pz_bytes).expect("import shape+ pz");
    let layer = &mut imported.project.sequence.video_tracks[0].clips[0].layer;
    assert!(
        !layer.preserved.contains_key("contents"),
        "import should not duplicate plugin payload-owned `contents` into layer.preserved"
    );
    assert!(
        !layer.preserved.contains_key("schemaVersion"),
        "import should not duplicate plugin payload-owned `schemaVersion` into layer.preserved"
    );
    assert!(
        !layer.preserved.contains_key("zoidiumShapePlus"),
        "import should not duplicate plugin payload-owned `zoidiumShapePlus` into layer.preserved"
    );

    // Also inject a stale `contents` entry into `layer.preserved` to prove export treats
    // `LayerKind::Plugin.payload` as authoritative even for legacy/deserialized state.
    layer.preserved.insert(
        "contents".to_owned(),
        json!([{ "id": "stale", "kind": "path" }]),
    );
    layer
        .preserved
        .insert("extraPreservedKey".to_owned(), json!("keep-me"));

    let LayerKind::Plugin {
        ref mut payload, ..
    } = layer.kind
    else {
        panic!("expected Shape+ Plugin layer");
    };
    payload["contents"] = json!([{
        "id": "p1",
        "kind": "path",
        "properties": {
            "name": "Edited Path",
            "closed": 0,
            "path": { "value": [42, 84, 100, 200, 300, 400] }
        }
    }]);

    let exported_bytes = export_pz_to_bytes(&imported.project).expect("export mutated shape+");
    let reimported = import_pz_from_bytes(&exported_bytes).expect("reimport mutated shape+");
    let roundtripped_layer = &reimported.project.sequence.video_tracks[0].clips[0].layer;
    let LayerKind::Plugin {
        ref plugin_type,
        ref payload,
    } = roundtripped_layer.kind
    else {
        panic!("expected Shape+ Plugin layer after round-trip");
    };
    assert_eq!(plugin_type, "zoidium:shape-plus/shape-layer");
    assert_eq!(
        payload["contents"][0]["properties"]["name"].as_str(),
        Some("Edited Path")
    );
    assert_eq!(
        payload["contents"][0]["properties"]["closed"].as_i64(),
        Some(0)
    );
    assert_eq!(
        payload["contents"][0]["properties"]["path"]["value"],
        json!([42, 84, 100, 200, 300, 400])
    );
    assert_eq!(
        payload["extraPreservedKey"].as_str(),
        Some("keep-me"),
        "missing keys in payload should still be filled from layer.preserved"
    );
}

#[test]
fn round_trips_track_to_and_follow_uuid_targets_across_export_and_import() {
    let mut proj = Project::new_cm3_default();
    let target_cube = Object3D::new(
        "Hero Cube",
        Object3DKind::Shape3D(Shape3DKind::Box {
            size: [10.0, 10.0, 10.0],
        }),
    );
    let nested_target = Object3D::new(
        "Nested Target",
        Object3DKind::Shape3D(Shape3DKind::Sphere {
            radius: 4.0,
            width_segments: 16,
            height_segments: 12,
        }),
    );
    let target_cube_uuid = target_cube.id.to_string();
    let nested_target_uuid = nested_target.id.to_string();

    let mut cam = Object3D::new("Camera", Object3DKind::Camera(Box::default()));
    cam.constraints.track_to = Some(TrackToConstraint {
        target_id: target_cube_uuid.clone(),
        track_axis: 5,
        up_axis: 0,
        influence: PropertyTrack::number(100.0),
    });

    let mut nested_follower = Object3D::new(
        "Nested Follower",
        Object3DKind::Shape3D(Shape3DKind::Box {
            size: [2.0, 2.0, 2.0],
        }),
    );
    nested_follower.constraints.follow = Some(FollowConstraint {
        target_id: nested_target_uuid.clone(),
        space: 0,
        follow_position: true,
        follow_rotation: true,
        follow_scale: false,
        position_offset: PropertyTrack::vec3([0.0, 5.0, 0.0]),
        rotation_offset: PropertyTrack::vec3([0.0, 0.0, 0.0]),
        scale_offset: PropertyTrack::vec3([1.0, 1.0, 1.0]),
        fallback_pose: None,
    });

    let group = Object3D::new(
        "Rig Group",
        Object3DKind::Group {
            reflection_visibility: 0,
            custom_properties: Vec::new(),
            children: vec![nested_target, nested_follower],
        },
    );

    let scene_layer = Layer::new(
        "3D Scene",
        LayerKind::Scene3D {
            objects: vec![cam, target_cube, group],
            shadow_mode: CompatMode::Current,
        },
        [1280, 720],
    );
    proj.sequence.video_tracks[0].clips[0].layer = scene_layer;
    proj.plugin_requirements.insert("constraints".to_owned());

    let pz_bytes = export_pz_to_bytes(&proj).expect("export .pz");
    let reimported = import_pz_from_bytes(&pz_bytes).expect("reimport .pz");
    assert!(
        reimported.warnings.is_empty(),
        "expected zero warnings, got {:?}",
        reimported.warnings
    );

    let LayerKind::Scene3D { ref objects, .. } = reimported.project.sequence.video_tracks[0].clips
        [0]
    .layer
    .kind
    else {
        panic!("expected Scene3D");
    };
    assert_eq!(objects.len(), 3);
    let new_cam = &objects[0];
    let new_cube = &objects[1];
    let Object3DKind::Group { ref children, .. } = objects[2].kind else {
        panic!("expected Group");
    };
    let new_nested_target = &children[0];
    let new_nested_follower = &children[1];

    // Fresh UUIDs must have been allocated on import, and constraints must be remapped to them.
    assert_ne!(new_cube.id.to_string(), target_cube_uuid);
    assert_ne!(new_nested_target.id.to_string(), nested_target_uuid);
    assert_eq!(
        new_cam
            .constraints
            .track_to
            .as_ref()
            .map(|t| t.target_id.as_str()),
        Some(new_cube.id.to_string().as_str())
    );
    assert_eq!(
        new_nested_follower
            .constraints
            .follow
            .as_ref()
            .map(|f| f.target_id.as_str()),
        Some(new_nested_target.id.to_string().as_str())
    );
}

#[test]
fn pz_round_trip_preserves_sequence_background_color_and_marker_colors() {
    let mut proj = Project::new_cm3_default();
    proj.name = "BG & Markers".to_owned();
    let bg = [15.0 / 255.0, 20.0 / 255.0, 36.0 / 255.0, 1.0];
    proj.sequence.background_color = bg;
    proj.sequence.markers = vec![
        crate::model::Marker {
            frame: 12.0,
            label: "Intro Hit".to_owned(),
        },
        crate::model::Marker {
            frame: 48.0,
            label: "Drop".to_owned(),
        },
    ];
    proj.sequence
        .preserved
        .insert("markerColors".to_owned(), json!(["#ff3366", "#22cc88"]));

    // Also attach a precomposition with its own background_color & marker colors.
    let sub_bg = [0.25, 0.1, 0.4, 1.0];
    let mut sub = Sequence {
        name: "SubComp".to_owned(),
        background_color: sub_bg,
        ..Sequence::default()
    };
    sub.markers.push(crate::model::Marker {
        frame: 15.0,
        label: "Cue".to_owned(),
    });
    sub.preserved
        .insert("markerColors".to_owned(), json!(["#ffaa00"]));
    let sub_id = sub.id;
    proj.compositions.insert(sub_id, sub);
    proj.composition_ids.insert("comp-sub".to_owned(), sub_id);

    let pz_bytes = export_pz_to_bytes(&proj).expect("export .pz");
    let reimported = import_pz_from_bytes(&pz_bytes).expect("import .pz");

    for i in 0..4 {
        assert!(
            (reimported.project.sequence.background_color[i] - bg[i]).abs() < 1e-5,
            "main sequence background_color[{i}] mismatch: got {:?}, expected {bg:?}",
            reimported.project.sequence.background_color
        );
    }
    assert_eq!(reimported.project.sequence.markers.len(), 2);
    assert_eq!(reimported.project.sequence.markers[0].label, "Intro Hit");
    assert_eq!(reimported.project.sequence.markers[1].label, "Drop");
    assert_eq!(
        reimported.project.sequence.preserved.get("markerColors"),
        Some(&json!(["#ff3366", "#22cc88"]))
    );

    let sub_rt = reimported
        .project
        .compositions
        .values()
        .next()
        .expect("reimported subcomp");
    for i in 0..4 {
        assert!(
            (sub_rt.background_color[i] - sub_bg[i]).abs() < 1e-5,
            "subcomp background_color[{i}] mismatch: got {:?}, expected {sub_bg:?}",
            sub_rt.background_color
        );
    }
    assert_eq!(sub_rt.markers.len(), 1);
    assert_eq!(sub_rt.markers[0].label, "Cue");
    assert_eq!(
        sub_rt.preserved.get("markerColors"),
        Some(&json!(["#ffaa00"]))
    );
}

#[test]
fn qa6_pz_roundtrip_shader_type3_color_ambient_light_precomp_media_control_id_shape_plus_node_graph_and_bit8_256()
 {
    let mut proj = Project::new_cm3_default();
    proj.name = "QA6 Round-Trip".to_owned();

    // 1. CustomShader with Color uniforms (`prop_type: Some(3)` and `Some(4)`)
    let mut shader_fx = create_custom_shader(
        "TintShader",
        "uniform sampler2D tDiffuse; uniform vec3 Tint3; uniform vec3 Tint4; void main() { gl_FragColor = vec4(Tint3 * Tint4, 1.0); }",
    );
    if let EffectKind::CustomShader {
        ref mut custom_properties,
        ..
    } = shader_fx.kind
    {
        custom_properties.push(CustomProperty {
            name: "Tint3".to_owned(),
            prop_type: Some(3),
            track: PropertyTrack::color([0.2, 0.7, 0.9]),
            metadata: BTreeMap::new(),
        });
        custom_properties.push(CustomProperty {
            name: "Tint4".to_owned(),
            prop_type: Some(4),
            track: PropertyTrack::color([0.9, 0.1, 0.4]),
            metadata: BTreeMap::new(),
        });
    }
    proj.sequence.video_tracks[0].clips[0]
        .layer
        .effects
        .push(shader_fx);

    // 2. LightKind::Ambient + clean NodeMaterial graph + bit-8 Hold keyframe (`tween: 256`)
    let ambient_light = Object3D::new(
        "Ambient",
        Object3DKind::Light(Box::new(LightData::for_kind(LightKind::Ambient))),
    );
    let clean_graph = json!({
        "nodes": [
            { "id": "n1", "type": "color", "params": { "color": [0.1, 0.5, 0.9] } },
            { "id": "out", "type": "pbrOutput" }
        ],
        "links": [
            { "from": "n1", "fromPort": "out", "to": "out", "toPort": "baseColor" }
        ]
    });
    let mut node_box = Object3D::new_shape_3d(
        "NodeBox",
        Shape3DKind::Box {
            size: [10.0, 10.0, 10.0],
        },
    );
    node_box.material = Some(MaterialSpec::NodeMaterial {
        schema_version: 1,
        resolution: 512,
        graph: clean_graph.clone(),
    });
    // Keyframe with `tween: 256, easing_index: 0, is_bezier: true` + custom control points.
    proj.sequence.video_tracks[0].clips[0].layer.opacity = PropertyTrack {
        animated: true,
        expression: None,
        static_behavior: crate::StaticBehavior::Constant,
        value: PropertyValue::Number(1.0),
        keyframes: vec![
            PropertyKeyframe {
                frame: 0.0,
                value: PropertyValue::Number(1.0),
                tween: 256,
                easing_index: 0,
                is_bezier: true,
                control_points: [[-4.0, 1.5], [6.0, -2.5]],
                continuous_tangent: true,
            },
            PropertyKeyframe::linear(30.0, PropertyValue::Number(0.25)),
        ],
        channels: Vec::new(),
    };
    if let LayerKind::Scene3D {
        ref mut objects, ..
    } = proj.sequence.video_tracks[0].clips[0].layer.kind
    {
        objects.push(ambient_light);
        objects.push(node_box);
    }

    // 3. SequenceControl with custom ID
    proj.sequence.controls.push(SequenceControl {
        id: "ctl_custom_hero".to_owned(),
        name: "Hero Slider".to_owned(),
        kind: SequenceControlKind::Slider,
        track: PropertyTrack::number(42.0),
        preserved: BTreeMap::new(),
    });

    // 4. Shape+ layer with `payload.version`
    let shape_plus_clip = VideoClip {
        id: Uuid::new_v4(),
        name: "ShapePlus".to_owned(),
        start: 60.0,
        length: 60.0,
        link: None,
        media: None,
        time_curve: PropertyTrack::linear_time_curve(60.0, 30),
        layer: Layer::new(
            "ShapePlus",
            LayerKind::Plugin {
                plugin_type: "zoidium:shape-plus/shape-layer".to_owned(),
                payload: json!({
                    "version": 2,
                    "schemaVersion": 1,
                    "contents": []
                }),
            },
            [1920, 1080],
        ),
        preserved: BTreeMap::new(),
    };
    proj.sequence.video_tracks[0].clips.push(shape_plus_clip);

    // 5. Precomp+ composition (must NOT leak spurious `('PrecompName', 'misc', None)` into `media`)
    let sub = Sequence {
        name: "PrecompName".to_owned(),
        ..Sequence::default()
    };
    let sub_id = sub.id;
    proj.compositions.insert(sub_id, sub);
    proj.composition_ids
        .insert("comp-precomp-1".to_owned(), sub_id);
    proj.media.clear();

    let pz_bytes = export_pz_to_bytes(&proj).expect("export qa6 .pz");
    let reimported = import_pz_from_bytes(&pz_bytes).expect("reimport qa6 .pz");
    let rp = reimported.project;

    // Shader colors preserved as PropertyValue::Color (NOT Vec3/Vec4)
    let fx = &rp.sequence.video_tracks[0].clips[0].layer.effects[0];
    let EffectKind::CustomShader {
        ref custom_properties,
        ..
    } = fx.kind
    else {
        panic!("expected CustomShader");
    };
    assert_eq!(
        custom_properties[0].track.value,
        PropertyValue::Color([0.2, 0.7, 0.9])
    );
    assert_eq!(
        custom_properties[1].track.value,
        PropertyValue::Color([0.9, 0.1, 0.4])
    );

    // Ambient light stays Ambient (never Hemisphere) & NodeMaterial graph stays clean
    let LayerKind::Scene3D { ref objects, .. } = rp.sequence.video_tracks[0].clips[0].layer.kind
    else {
        panic!("expected Scene3D");
    };
    let amb = objects
        .iter()
        .find(|o| o.name == "Ambient")
        .expect("ambient light");
    assert!(matches!(
        amb.kind,
        Object3DKind::Light(ref l) if l.kind == LightKind::Ambient
    ));
    let nb = objects
        .iter()
        .find(|o| o.name == "NodeBox")
        .expect("node box");
    let Some(MaterialSpec::NodeMaterial {
        resolution,
        ref graph,
        ..
    }) = nb.material
    else {
        panic!("expected NodeMaterial");
    };
    assert_eq!(resolution, 512);
    assert_eq!(graph, &clean_graph);
    assert!(graph.get("properties").is_none());
    assert!(graph.get("type").is_none());
    assert!(graph.get("zoidiumUvMode").is_none());

    // Precomp did not leak into project.media
    assert!(
        rp.media.is_empty(),
        "Precomp+ composition must not leak into project.media: {:?}",
        rp.media
    );

    // SequenceControl.id preserved
    assert_eq!(rp.sequence.controls[0].id, "ctl_custom_hero");

    // Shape+ payload.version preserved
    let sp_clip = &rp.sequence.video_tracks[0].clips[1];
    let LayerKind::Plugin { ref payload, .. } = sp_clip.layer.kind else {
        panic!("expected Shape+ Plugin layer");
    };
    assert_eq!(payload.get("version"), Some(&json!(2)));

    // bit-8 Hold keyframe `tween: 256` NOT rewritten to `257`, custom control points preserved
    let kf0 = &rp.sequence.video_tracks[0].clips[0].layer.opacity.keyframes[0];
    assert_eq!(kf0.tween, 256);
    assert_eq!(kf0.easing_index, 0);
    assert!(!kf0.is_bezier);
    assert_eq!(kf0.control_points, [[-4.0, 1.5], [6.0, -2.5]]);
}
