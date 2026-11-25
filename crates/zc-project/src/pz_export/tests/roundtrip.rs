use super::*;
use crate::model::{
    AssetKind, AssetRecord, AudioClip, CameraData, CameraKind, CompatMode, CustomProperty, DrawOp,
    EffectKind, Layer, LayerKind, LayerMask, LightData, LightKind, MaskMode, MaterialSpec,
    MediaItem, Object3D, Object3DKind, ParticlesData, Project, PropertyKeyframe, PropertyTrack,
    PropertyValue, Sequence, SequenceControl, SequenceControlKind, Shape2DKind, Shape2DNode,
    Shape3DKind, Text3DAlign, Text3DData, VideoClip, VideoTrack,
};
use flate2::Compression;
use flate2::write::GzEncoder;
use serde_json::{Value, json};
use uuid::Uuid;

#[test]
fn round_trips_full_featured_project_with_integrity_and_zero_warnings() {
    let mut project = Project::new_cm3_default();
    project.name = "Full Round-Trip Showcase".to_owned();
    project.imported_from_pz_version = Some(102);

    // Embedded binary assets
    let png_bytes = b"\x89PNG\r\n\x1a\nHERO_TEXTURE_PAYLOAD".to_vec();
    let matcap_bytes = b"\x89PNG\r\n\x1a\nMATCAP_TEXTURE_PAYLOAD".to_vec();
    let ao_bytes = b"\x89PNG\r\n\x1a\nAO_MAP_PAYLOAD".to_vec();
    let font_bytes = b"OTTO_FONT_PAYLOAD".to_vec();
    let audio_bytes = b"OggS_AUDIO_PAYLOAD".to_vec();

    let png_id = project.insert_binary_asset(AssetKind::Image, "hero.png", png_bytes.clone());
    let matcap_id =
        project.insert_binary_asset(AssetKind::Image, "matcap.png", matcap_bytes.clone());
    let ao_id = project.insert_binary_asset(AssetKind::Image, "ao.png", ao_bytes.clone());
    let font_id = project.insert_binary_asset(AssetKind::Font, "title.otf", font_bytes.clone());
    let audio_id = project.insert_binary_asset(AssetKind::Av, "bgm.ogg", audio_bytes.clone());

    // Media library entry
    project.media.push(MediaItem {
        id: Uuid::new_v4(),
        name: "Hero Texture".to_owned(),
        asset_id: Some(png_id.clone()),
        kind: AssetKind::Image,
        preserved: BTreeMap::new(),
    });

    // Precomp+ nested composition
    let comp_uuid = Uuid::new_v4();
    let mut comp_seq = Sequence {
        id: comp_uuid,
        name: "Nested Precomp".to_owned(),
        resolution: [1920, 1080],
        fps: 60,
        length_frames: 90.0,
        ..Sequence::default()
    };
    let comp_img_layer = Layer::new(
        "Precomp Image",
        LayerKind::Image {
            image: Some(png_id.clone()),
        },
        [1920, 1080],
    );
    comp_seq.video_tracks.push(VideoTrack {
        id: Uuid::new_v4(),
        name: "Comp V1".to_owned(),
        enabled: true,
        solo: false,
        locked: false,
        is_3d: false,
        clips: vec![VideoClip {
            id: Uuid::new_v4(),
            name: "Comp Img Clip".to_owned(),
            start: 0.0,
            length: 90.0,
            link: None,
            media: None,
            time_curve: PropertyTrack::linear_time_curve(90.0, 60),
            layer: comp_img_layer,
            preserved: BTreeMap::new(),
        }],
        preserved: BTreeMap::new(),
    });
    project.compositions.insert(comp_uuid, comp_seq);
    project
        .composition_ids
        .insert("comp-nested-1".to_owned(), comp_uuid);

    // Main sequence metadata
    project.sequence.name = "Main Timeline".to_owned();
    project.sequence.resolution = [2560, 1440];
    project.sequence.fps = 30;
    project.sequence.exact_fps = Some(29.97);
    project.sequence.length_frames = 240.0;
    project.sequence.background_color = [0.05, 0.1, 0.2, 1.0];
    project.sequence.motion_blur = MotionBlurSettings {
        enabled: true,
        samples: PropertyTrack::number(12.0),
        shutter: PropertyTrack::number(0.65),
        track_frame: 1,
        sensitivity: 125.0,
    };
    project.sequence.markers.push(crate::model::Marker {
        frame: 45.0,
        label: "Beat Drop".to_owned(),
    });

    // Sequence Controls (Expressions+)
    let mut slider_track = PropertyTrack::number(10.0);
    slider_track.animated = true;
    slider_track.expression = Some("value * 1.5".to_owned());
    slider_track.keyframes = vec![
        PropertyKeyframe::from_cm3_tween(
            0.0,
            PropertyValue::Number(10.0),
            257,
            [[-5.0, 0.0], [12.0, 8.0]],
            false,
        ),
        PropertyKeyframe::from_cm3_tween(
            60.0,
            PropertyValue::Number(85.0),
            257,
            [[-15.0, -5.0], [10.0, 0.0]],
            true,
        ),
    ];
    project.sequence.controls.push(SequenceControl {
        id: "c1".to_owned(),
        name: "Master Intensity".to_owned(),
        kind: SequenceControlKind::Slider,
        track: slider_track,
        preserved: BTreeMap::new(),
    });
    project.sequence.controls.push(SequenceControl {
        id: "c2".to_owned(),
        name: "Aim Point".to_owned(),
        kind: SequenceControlKind::Point2D,
        track: PropertyTrack::vec2([12.5, -7.25]),
        preserved: BTreeMap::new(),
    });
    project.sequence.controls.push(SequenceControl {
        id: "c3".to_owned(),
        name: "Accent Tint".to_owned(),
        kind: SequenceControlKind::Color,
        track: PropertyTrack::color([0.2, 0.6, 0.9]),
        preserved: BTreeMap::new(),
    });

    // Scene3D objects: Camera+ (DOF + Vibrate + MotionBlur + filmGate), Light+, Text3D (Text+ props + PbrPlus),
    // Geometry+ RoundedBox (Matcap + TrackTo + Follow), Open Cylinder, Particles
    let mut cam_data = CameraData {
        kind: CameraKind::Perspective,
        fov: PropertyTrack::number(52.0),
        near: 0.25,
        far: 8000.0,
        zoom: PropertyTrack::number(1.2),
        active: true,
        focal_length: PropertyTrack::number(50.0),
        film_gate: 24.89,
        film_offset_x: PropertyTrack::number(4.0),
        film_offset_y: PropertyTrack::number(-2.5),
        dof: Some(CameraDofParams {
            enabled: true,
            enabled_track: None,
            focus_distance: PropertyTrack::number(115.0),
            aperture: PropertyTrack::number(5.5),
            focus_area_width: PropertyTrack::number(18.0),
            near_blur: PropertyTrack::number(140.0),
            far_blur: PropertyTrack::number(90.0),
        }),
        vibrate: Some(CameraVibrateParams {
            enabled: true,
            seed: 77,
            position_amplitude: PropertyTrack::vec3([2.0, 3.0, 4.0]),
            rotation_amplitude: PropertyTrack::vec3([1.0, 2.0, 0.5]),
            frequency: PropertyTrack::number(3.5),
            regular_pulse: true,
            relative: true,
            enable_position: true,
            enable_rotation: true,
            position_frequency: PropertyTrack::number(3.5),
            rotation_frequency: PropertyTrack::number(1.75),
        }),
        motion_blur: Some(CameraMotionBlurParams {
            enabled: true,
            samples: 6,
            shutter: PropertyTrack::number(0.75),
        }),
        ..CameraData::default()
    };
    cam_data.shake.mode = 1;
    let mut cam_obj = Object3D::new("Hero Cam+", Object3DKind::Camera(Box::new(cam_data)));
    cam_obj.position = PropertyTrack::vec3([0.0, 5.0, 95.0]);
    cam_obj.euler_order = EulerOrder::Yxz;
    cam_obj.constraints.zoidium_id = Some("cam-hero-id".to_owned());
    cam_obj.constraints.track_to = Some(TrackToConstraint {
        target_id: "text-target-id".to_owned(),
        track_axis: 5,
        up_axis: 1,
        influence: PropertyTrack::number(85.0),
    });

    let mut light_data = LightData::for_kind(LightKind::Spot);
    light_data.intensity = PropertyTrack::number(2.5);
    light_data.angle = PropertyTrack::number(45.0);
    light_data.penumbra = 0.35;
    light_data.decay = 2.0;
    light_data.distance = 350.0;
    light_data.cast_shadow = false;
    let light_obj = Object3D::new("Key Spot", Object3DKind::Light(Box::new(light_data)));

    let mut text_data = Text3DData {
        text: "ZOIDCRAFT".to_owned(),
        font: Some(font_id.clone()),
        size: [28.0, 5.5],
        detail: 8,
        bevel: true,
        bevel_thickness: 0.35,
        bevel_size: 0.85,
        position_mode: Text3DAlign::Left,
        text_scale_mode: CompatMode::Current,
        spacing: 3.5,
        character_index_offset: PropertyTrack::number(2.0),
        character_count_override: PropertyTrack::number(6.0),
        bevel_side: 1,
        bevel_detail: 6,
        bevel_profile: 0,
        bevel_tension: 0.3,
    };
    text_data.character_index_offset.expression = Some("floor(time * 4)".to_owned());
    let mut text_obj = Object3D::new("Title3D", Object3DKind::Text3D(Box::new(text_data)));
    text_obj.constraints.zoidium_id = Some("text-target-id".to_owned());
    text_obj.zoidium_uv_mode = CompatMode::Current;
    text_obj.material = Some(MaterialSpec::PbrPlus(Box::new(PbrPlusMaterial {
        base_color: PropertyTrack::color([0.9, 0.8, 0.7]),
        base_color_map: Some(png_id.clone()),
        ao_texture: Some(ao_id.clone()),
        emissive_intensity: PropertyTrack::number(2.2),
        alpha_test: 0.25,
        reflection: true,
        reflection_intensity: 0.6,
        transparent: true,
        ..PbrPlusMaterial::default()
    })));

    let mut rbox_obj = Object3D::new(
        "Matcap RBox",
        Object3DKind::Shape3D(Shape3DKind::RoundedBox {
            size: [30.0, 20.0, 8.0],
            radius: 2.5,
            segments: 5,
            corner_radius: Some(2.5),
            corner_segments: Some(5),
            bevel_size: Some(0.6),
            bevel_segments: Some(2),
            shading: 0,
        }),
    );
    rbox_obj.material = Some(MaterialSpec::Matcap(Box::new(MatcapMaterial {
        texture: Some(matcap_id.clone()),
        brightness: PropertyTrack::number(1.35),
        transparent: true,
        blending: 2,
        side: 2,
        ..MatcapMaterial::default()
    })));
    rbox_obj.constraints.follow = Some(FollowConstraint {
        target_id: "text-target-id".to_owned(),
        space: 1,
        follow_position: true,
        follow_rotation: false,
        follow_scale: true,
        position_offset: PropertyTrack::vec3([1.0, 2.0, 3.0]),
        rotation_offset: PropertyTrack::vec3([0.0, 0.5, 0.0]),
        scale_offset: PropertyTrack::vec3([1.0, 1.0, 1.0]),
        fallback_pose: None,
    });

    let open_cyl = Object3D::new_shape_3d(
        "OpenCyl",
        Shape3DKind::Cylinder {
            top_radius: 4.0,
            bottom_radius: 8.0,
            height: 16.0,
            detail: 24,
            open_ended: true,
        },
    );

    let particles_obj = Object3D::new(
        "Sparks",
        Object3DKind::Particles(Box::new(ParticlesData {
            number: 250,
            rate: 45.0,
            lifetime: 2.5,
            pdist: 1,
            texture: Some(png_id.clone()),
            ..ParticlesData::default()
        })),
    );

    // Scene3D layer with backdrop, Layer3DProps, LayerMask, CustomShader + uniforms, bypassed standard effect
    let mut scene_layer = Layer::new_scene_3d(
        [2560, 1440],
        vec![
            cam_obj,
            light_obj,
            text_obj,
            rbox_obj,
            open_cyl,
            particles_obj,
        ],
    );
    scene_layer.blending = "BLEND_SCREEN".to_owned();
    scene_layer.opacity = PropertyTrack {
        animated: true,
        expression: Some("clamp(value, 0, 1)".to_owned()),
        static_behavior: crate::model::StaticBehavior::Constant,
        value: PropertyValue::Number(0.25),
        keyframes: vec![
            PropertyKeyframe::from_cm3_tween(
                0.0,
                PropertyValue::Number(0.25),
                257,
                [[-8.0, 0.0], [12.0, 0.3]],
                false,
            ),
            PropertyKeyframe::from_cm3_tween(
                30.0,
                PropertyValue::Number(1.0),
                257,
                [[-10.0, -0.2], [8.0, 0.0]],
                true,
            ),
        ],
        channels: Vec::new(),
    };

    let mut custom_fx = create_custom_shader(
        "Custom Vignette",
        "void main() { gl_FragColor = vec4(1.0); }",
    );
    if let EffectKind::CustomShader {
        ref mut custom_properties,
        ..
    } = custom_fx.kind
    {
        custom_properties.push(CustomProperty {
            name: "uStrength".to_owned(),
            prop_type: Some(0),
            track: PropertyTrack::number_with_expr(0.75, "sin(time)"),
            metadata: BTreeMap::new(),
        });
        custom_properties.push(CustomProperty {
            name: "uTint".to_owned(),
            prop_type: Some(4),
            track: PropertyTrack::color([0.1, 0.5, 0.9]),
            metadata: BTreeMap::new(),
        });
        custom_properties.push(CustomProperty {
            name: "uMap".to_owned(),
            prop_type: Some(8),
            track: PropertyTrack::asset(Some(png_id.clone())),
            metadata: BTreeMap::new(),
        });
    }
    let bloom_fx = create_standard_effect("bloom").expect("bloom");
    let bloom_id = bloom_fx.id;
    let vhs_fx = create_zoidium_effect("vhs").expect("vhs");
    scene_layer.effects = vec![bloom_fx, custom_fx, vhs_fx];

    project.sequence.bypassed_effects.insert(bloom_id);
    project.sequence.scene_backdrops.insert(
        scene_layer.id,
        SceneBackdrop {
            mode: 1,
            color: [0.1, 0.2, 0.3],
            opacity: PropertyTrack::number(0.8),
        },
    );
    project.sequence.layer_3d_props.insert(
        scene_layer.id,
        Layer3DProps {
            depth: PropertyTrack::number(40.0),
            rotation_x: PropertyTrack::number(0.25),
            rotation_y: PropertyTrack::number(-0.15),
            motion_blur_amount: PropertyTrack::number(0.5),
        },
    );
    project.sequence.layer_masks.insert(
        scene_layer.id,
        vec![LayerMask {
            id: "mask-1".to_owned(),
            name: "Vignette Mask".to_owned(),
            enabled: true,
            mode: MaskMode::Subtract,
            invert: true,
            feather: 14.0,
            expansion: -3.5,
            opacity: 0.9,
            closed: true,
            locked: false,
            outline_color: String::new(),
            vertex_ids: vec!["v1".to_owned(), "v2".to_owned()],
            vertex_modes: BTreeMap::from([("v1".to_owned(), "smooth".to_owned())]),
            path: PropertyTrack::constant(PropertyValue::Json(json!([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]))),
            preserved: BTreeMap::new(),
        }],
    );

    // Composite layer (bottom-to-top: [Shape2D, Text2D, Shape+])
    let shape_2d_layer = Layer::new(
        "Bottom Vec",
        LayerKind::Shape2D {
            shapes: vec![Shape2DNode {
                id: Uuid::new_v4(),
                name: "OutlineGroup".to_owned(),
                kind: Shape2DKind::Group {
                    children: Vec::new(),
                },
                position: PropertyTrack::vec2([10.0, 20.0]),
                rotation: PropertyTrack::number(0.5),
                scale: PropertyTrack::vec2([1.5, 1.5]),
                operations: vec![
                    DrawOp::Fill {
                        color: PropertyTrack::color([0.2, 0.4, 0.8]),
                    },
                    DrawOp::Stroke {
                        color: PropertyTrack::color([0.9, 0.1, 0.3]),
                        size: PropertyTrack::number(4.0),
                        opacity: PropertyTrack::number(0.75),
                    },
                ],
                preserved: BTreeMap::new(),
            }],
        },
        [2560, 1440],
    );
    let text_2d_layer = Layer::new(
        "Mid Text2D",
        LayerKind::Text2D {
            text: "HELLO 2D".to_owned(),
            font: Some(font_id.clone()),
            size: 120,
            shapes: Vec::new(),
        },
        [2560, 1440],
    );
    let shape_plus_layer = Layer::new(
        "Top Shape+",
        LayerKind::Plugin {
            plugin_type: "zoidium:shape-plus/shape-layer".to_owned(),
            payload: json!({
                "type": "zoidium:shape-plus/shape-layer",
                "schemaVersion": 1,
                "contents": [{ "id": "txt1", "kind": "text", "properties": { "font": font_id } }]
            }),
        },
        [2560, 1440],
    );
    let comp_layer = Layer::new(
        "Stack Composite",
        LayerKind::Composite {
            layers: vec![shape_2d_layer, text_2d_layer, shape_plus_layer],
        },
        [2560, 1440],
    );

    // PrecompRef layer
    let precomp_layer = Layer::new(
        "Precomp Instance",
        LayerKind::PrecompRef {
            composition_id: comp_uuid,
        },
        [2560, 1440],
    );
    project
        .sequence
        .precomp_offsets
        .insert(precomp_layer.id, 18.0);

    let scene_clip_id = Uuid::new_v4();
    let audio_clip_id = Uuid::new_v4();

    let vt = &mut project.sequence.video_tracks[0];
    vt.name = "Hero V1".to_owned();
    vt.solo = true;
    vt.locked = true;
    vt.is_3d = true;
    project
        .sequence
        .track_layer_input_ids
        .insert(vt.id, "layer-input-v1".to_owned());
    vt.clips = vec![
        VideoClip {
            id: scene_clip_id,
            name: "Scene Clip".to_owned(),
            start: 0.0,
            length: 180.0,
            link: Some(1),
            media: None,
            time_curve: PropertyTrack::linear_time_curve(180.0, 30),
            layer: scene_layer,
            preserved: BTreeMap::new(),
        },
        VideoClip {
            id: Uuid::new_v4(),
            name: "Composite Clip".to_owned(),
            start: 180.0,
            length: 60.0,
            link: None,
            media: None,
            time_curve: PropertyTrack::linear_time_curve(60.0, 30),
            layer: comp_layer,
            preserved: BTreeMap::new(),
        },
        VideoClip {
            id: Uuid::new_v4(),
            name: "Precomp Clip".to_owned(),
            start: 60.0,
            length: 90.0,
            link: None,
            media: None,
            time_curve: PropertyTrack::linear_time_curve(90.0, 30),
            layer: precomp_layer,
            preserved: BTreeMap::new(),
        },
    ];

    let at = &mut project.sequence.audio_tracks[0];
    at.name = "Music A1".to_owned();
    at.enabled = false;
    at.muted = true;
    at.solo = true;
    at.locked = true;
    at.clips = vec![AudioClip {
        id: audio_clip_id,
        name: "BGM Clip".to_owned(),
        start: 0.0,
        length: 180.0,
        link: Some(1),
        media: Some(audio_id.clone()),
        time_curve: PropertyTrack::linear_time_curve(180.0, 30),
        audio: AudioClipData {
            name: "BGM".to_owned(),
            volume: PropertyTrack::number(0.85),
            pan: PropertyTrack::number(-0.2),
            preserved: BTreeMap::new(),
        },
        preserved: BTreeMap::new(),
    }];

    project
        .sequence
        .clip_links
        .insert(1, vec![scene_clip_id, audio_clip_id]);

    // Export to bytes AND inspect TAR layout + zoidium.json integrity + 5-element clipLinks
    let pz_bytes = export_pz_to_bytes(&project).expect("export_pz_to_bytes");
    let tar_entries = inspect_tar_entries(&pz_bytes);

    assert!(tar_entries.contains_key("meta"));
    assert!(tar_entries.contains_key("project"));
    assert!(tar_entries.contains_key("zoidium.json"));
    assert!(tar_entries.contains_key(&cm3_asset_key(&png_bytes)));
    assert!(tar_entries.contains_key(&cm3_asset_key(&matcap_bytes)));
    assert!(tar_entries.contains_key(&cm3_asset_key(&ao_bytes)));
    assert!(tar_entries.contains_key(&cm3_asset_key(&font_bytes)));
    assert!(tar_entries.contains_key(&cm3_asset_key(&audio_bytes)));

    // Verify zoidium.json covers EVERY non-zoidium.json entry with matching size & SHA-256
    let zoidium_json: Value =
        serde_json::from_slice(&tar_entries["zoidium.json"]).expect("parse zoidium.json");
    assert_eq!(zoidium_json["name"], "Full Round-Trip Showcase");
    let integrity_map = zoidium_json["integrity"]
        .as_object()
        .expect("integrity map");
    for (name, data) in &tar_entries {
        if name == "zoidium.json" {
            continue;
        }
        let rec = integrity_map
            .get(name)
            .unwrap_or_else(|| panic!("missing integrity record for {name}"));
        assert_eq!(rec["size"].as_u64(), Some(data.len() as u64));
        assert_eq!(rec["sha256"].as_str(), Some(sha256_hex(data).as_str()));
    }

    // Verify 5-element clipLinks format and dynamic Camera+ motion-blur tracks in exported JSON
    let exported_doc: Value =
        serde_json::from_slice(&tar_entries["project"]).expect("parse project json");
    assert_eq!(
        exported_doc["sequence"]["clipLinks"]["links"]["1"]["clips"],
        json!([[0, 1, 0, 0, 0], [0, 2, 0, 0, 0]])
    );
    assert_eq!(
        exported_doc["sequence"]["properties"]["motionBlurTrackFrame"],
        json!({
            "animated": false,
            "keyframes": [{ "frame": 0, "value": 1, "tween": 0 }]
        })
    );
    assert_eq!(
        exported_doc["sequence"]["properties"]["motionBlurSensitivity"],
        json!({
            "animated": false,
            "keyframes": [{ "frame": 0, "value": 125.0, "tween": 1 }]
        })
    );

    // Re-import via import_pz_from_bytes and verify zero warnings & lossless fidelity
    let reimported = import_pz_from_bytes(&pz_bytes).expect("reimport pz");
    assert!(
        reimported.warnings.is_empty(),
        "unexpected warnings: {:?}",
        reimported.warnings
    );
    let rp = reimported.project;

    assert_eq!(rp.name, "Full Round-Trip Showcase");
    assert_eq!(rp.imported_from_pz_version, Some(102));
    assert_eq!(rp.sequence.resolution, [2560, 1440]);
    assert_eq!(rp.sequence.fps, 30);
    assert!((rp.sequence.exact_fps.unwrap() - 29.97).abs() < 1e-9);
    assert!((rp.sequence.length_frames - 240.0).abs() < 1e-9);
    assert_eq!(
        rp.sequence.preserved.get("property.backgroundColor"),
        Some(&json!([
            0.05000000074505806,
            0.10000000149011612,
            0.20000000298023224,
            1.0
        ]))
    );
    assert!(rp.sequence.motion_blur.enabled);
    // Note: `pz_import.rs` currently reads `motionBlurTrackFrame` and `motionBlurSensitivity`
    // via primitive-only `value_as_u32` / `prop_f64`, so wrapped dynamic objects are verified
    // on raw exported JSON above until `pz_import.rs` unwraps dynamic sequence motion-blur tracks.
    assert_eq!(rp.sequence.markers.len(), 1);
    assert_eq!(rp.sequence.markers[0].label, "Beat Drop");

    // Controls
    assert_eq!(rp.sequence.controls.len(), 3);
    assert_eq!(rp.sequence.controls[0].id, "c1");
    assert_eq!(rp.sequence.controls[0].name, "Master Intensity");
    assert_eq!(
        rp.sequence.controls[0].track.expression.as_deref(),
        Some("value * 1.5")
    );
    assert!(rp.sequence.controls[0].track.keyframes[0].is_bezier);
    assert!(!rp.sequence.controls[0].track.keyframes[0].continuous_tangent);
    assert_eq!(
        rp.sequence.controls[1].track.value,
        PropertyValue::Vec2([12.5, -7.25])
    );
    assert_eq!(
        rp.sequence.controls[2].track.value,
        PropertyValue::Color([0.2, 0.6, 0.9])
    );

    // Video & Audio tracks
    let rvt = &rp.sequence.video_tracks[0];
    assert!(rvt.solo && rvt.locked && rvt.is_3d);
    assert_eq!(
        rp.sequence
            .track_layer_input_ids
            .get(&rvt.id)
            .map(String::as_str),
        Some("layer-input-v1")
    );
    let rat = &rp.sequence.audio_tracks[0];
    assert!(!rat.enabled && rat.muted && rat.solo && rat.locked);
    assert_eq!(rat.clips[0].media.as_deref(), Some(audio_id.as_str()));

    // Clip links
    assert_eq!(
        rp.sequence.clip_links.get(&1),
        Some(&vec![rvt.clips[0].id, rat.clips[0].id])
    );

    // Scene3D clip checks
    let rscene_layer = &rvt.clips[0].layer;
    assert_eq!(rscene_layer.blending, "BLEND_SCREEN");
    assert!(rscene_layer.opacity.keyframes[0].is_bezier);
    assert!(!rscene_layer.opacity.keyframes[0].continuous_tangent);
    assert_eq!(
        rscene_layer.opacity.expression.as_deref(),
        Some("clamp(value, 0, 1)")
    );
    assert_eq!(rp.sequence.layer_masks[&rscene_layer.id].len(), 1);
    assert_eq!(
        rp.sequence.layer_masks[&rscene_layer.id][0].mode,
        MaskMode::Subtract
    );
    assert!(rp.sequence.layer_masks[&rscene_layer.id][0].invert);
    assert_eq!(rp.sequence.scene_backdrops[&rscene_layer.id].mode, 1);
    assert!(rp.sequence.layer_3d_props.contains_key(&rscene_layer.id));
    assert!(
        rp.sequence
            .bypassed_effects
            .contains(&rscene_layer.effects[0].id)
    );

    // Custom shader + uniforms
    let EffectKind::CustomShader {
        ref frag_shader,
        ref custom_properties,
    } = rscene_layer.effects[1].kind
    else {
        panic!("expected CustomShader");
    };
    assert_eq!(frag_shader, "void main() { gl_FragColor = vec4(1.0); }");
    assert_eq!(custom_properties.len(), 3);
    assert_eq!(
        custom_properties[0].track.expression.as_deref(),
        Some("sin(time)")
    );
    assert_eq!(
        custom_properties[2].track.value,
        PropertyValue::Asset(Some(png_id.clone()))
    );

    // Scene3D objects
    let LayerKind::Scene3D { ref objects, .. } = rscene_layer.kind else {
        panic!("expected Scene3D");
    };
    assert_eq!(objects.len(), 6);
    assert_eq!(objects[0].euler_order, EulerOrder::Yxz);
    assert_eq!(
        objects[0]
            .constraints
            .track_to
            .as_ref()
            .map(|t| t.target_id.as_str()),
        Some("text-target-id")
    );
    let Object3DKind::Camera(ref rcam) = objects[0].kind else {
        panic!("expected Camera");
    };
    assert!((rcam.film_gate - 24.89).abs() < 1e-9);
    assert!(rcam.dof.as_ref().unwrap().enabled);
    assert_eq!(rcam.vibrate.as_ref().unwrap().seed, 77);
    assert_eq!(rcam.motion_blur.as_ref().unwrap().samples, 6);

    let Object3DKind::Light(ref rlight) = objects[1].kind else {
        panic!("expected Light");
    };
    assert!(!rlight.cast_shadow);
    assert!((rlight.distance - 350.0).abs() < 1e-9);

    let Object3DKind::Text3D(ref rtext) = objects[2].kind else {
        panic!("expected Text3D");
    };
    assert_eq!(rtext.text, "ZOIDCRAFT");
    assert_eq!(rtext.font.as_deref(), Some(font_id.as_str()));
    assert!((rtext.bevel_thickness - 0.35).abs() < 1e-9);
    assert!((rtext.bevel_size - 0.85).abs() < 1e-9);
    assert!((rtext.spacing - 3.5).abs() < 1e-9);
    assert_eq!(rtext.bevel_detail, 6);
    assert_eq!(rtext.text_scale_mode, CompatMode::Current);
    let Some(MaterialSpec::PbrPlus(ref rpbr)) = objects[2].material else {
        panic!("expected PbrPlus");
    };
    assert_eq!(rpbr.base_color_map.as_deref(), Some(png_id.as_str()));
    assert_eq!(rpbr.ao_texture.as_deref(), Some(ao_id.as_str()));

    let Some(MaterialSpec::Matcap(ref rmc)) = objects[3].material else {
        panic!("expected Matcap");
    };
    assert_eq!(rmc.texture.as_deref(), Some(matcap_id.as_str()));
    assert_eq!(
        objects[3]
            .constraints
            .follow
            .as_ref()
            .map(|f| f.target_id.as_str()),
        Some("text-target-id")
    );

    assert!(matches!(
        objects[4].kind,
        Object3DKind::Shape3D(Shape3DKind::Cylinder {
            open_ended: true,
            detail: 24,
            ..
        })
    ));
    assert!(matches!(
        objects[5].kind,
        Object3DKind::Particles(ref p) if p.number == 250 && p.texture.as_deref() == Some(png_id.as_str())
    ));

    // Composite order preserved (bottom-to-top: [Bottom Vec, Mid Text2D, Top Shape+])
    let LayerKind::Composite { ref layers } = rvt.clips[1].layer.kind else {
        panic!("expected Composite");
    };
    assert_eq!(layers.len(), 3);
    assert_eq!(layers[0].name, "Bottom Vec");
    assert_eq!(layers[1].name, "Mid Text2D");
    assert_eq!(layers[2].name, "Top Shape+");
    let LayerKind::Plugin {
        ref plugin_type,
        ref payload,
    } = layers[2].kind
    else {
        panic!("expected Shape+ Plugin");
    };
    assert_eq!(plugin_type, "zoidium:shape-plus/shape-layer");
    assert_eq!(
        payload["contents"][0]["properties"]["font"].as_str(),
        Some(font_id.as_str())
    );

    // Precomp+ composition and offset preserved
    let rcomp_uuid = rp.composition_ids["comp-nested-1"];
    assert_eq!(rp.compositions[&rcomp_uuid].name, "Nested Precomp");
    assert_eq!(
        rvt.clips[2].layer.kind,
        LayerKind::PrecompRef {
            composition_id: rcomp_uuid
        }
    );
    assert_eq!(
        rp.sequence.precomp_offsets.get(&rvt.clips[2].layer.id),
        Some(&18.0)
    );

    // Embedded binary assets recovered intact
    assert_eq!(
        rp.assets[&png_id].bytes.as_deref(),
        Some(png_bytes.as_slice())
    );
    assert_eq!(
        rp.assets[&font_id].bytes.as_deref(),
        Some(font_bytes.as_slice())
    );
    assert_eq!(
        rp.assets[&audio_id].bytes.as_deref(),
        Some(audio_bytes.as_slice())
    );

    // Atomic disk export works too
    let dir = TestDir::new();
    let target = dir.0.join("showcase.pz");
    export_pz(&project, &target).expect("export_pz to disk");
    let from_disk = crate::pz_import::import_pz(&target).expect("import_pz from disk");
    assert!(from_disk.warnings.is_empty());
    assert_eq!(from_disk.project.name, "Full Round-Trip Showcase");
}

#[test]
fn reexports_complex_imported_pz_cleanly_with_zero_warnings() {
    let tex_bytes = b"\x89PNG\r\n\x1a\nREEXPORT_TEX";
    let key_tex = cm3_asset_key(tex_bytes);
    let project_json = json!({
        "plugins": [
            { "id": "camera-plus", "features": ["dof"] },
            { "id": "material-plus", "materials": ["uvcustom", "imageplus"] }
        ],
        "assets": {
            &key_tex: {
                "type": 0,
                "source": 1,
                "external": false,
                "filename": "reexport.png",
                "size": tex_bytes.len(),
                "sha256": &key_tex
            }
        },
        "media": [],
        "sequence": {
            "length": 120,
            "properties": {
                "name": "Complex Imported",
                "resolution": [1920, 1080],
                "rate": 60
            },
            "videoTracks": [{
                "type": 0,
                "enabled": true,
                "clips": [
                    {
                        "start": 0,
                        "length": 120,
                        "properties": { "name": "SceneClip" },
                        "object": {
                            "type": 4,
                            "zoidiumShadowMode": "current",
                            "properties": { "name": "MainScene" },
                            "objects": [
                                {
                                    "type": 0,
                                    "objectType": 1,
                                    "properties": { "name": "UvCustomBox" },
                                    "material": {
                                        "type": "uvcustom",
                                        "zoidiumUvMode": "current",
                                        "properties": {
                                            "texture": &key_tex,
                                            "roughness": 0.4,
                                            "metalness": 0.7
                                        }
                                    }
                                },
                                {
                                    "type": 0,
                                    "objectType": 5,
                                    "properties": { "name": "ImagePlusSphere" },
                                    "material": {
                                        "type": "zoidium:material-plus/image-plus",
                                        "properties": {
                                            "texture": &key_tex,
                                            "opacity": 0.85,
                                            "transparent": 1
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    {
                        "start": 0,
                        "length": 120,
                        "properties": { "name": "CamClip" },
                        "object": {
                            "type": 9,
                            "properties": { "name": "Camera Layer" },
                            "objects": [{
                                "type": "zoidium:camera-plus/camera",
                                "properties": {
                                    "name": "Timeline Cam+",
                                    "active": 1,
                                    "projection": "perspective",
                                    "focalLength": 35.0,
                                    "filmGate": 36.0,
                                    "depthOfField": {
                                        "enabled": 1,
                                        "focusDistance": 75.0,
                                        "aperture": 4.0
                                    }
                                }
                            }]
                        }
                    }
                ]
            }],
            "audioTracks": []
        }
    });

    // Build initial .pz archive
    let proj_bytes = serde_json::to_vec(&project_json).unwrap();
    let meta_bytes = br#"{"version":"1.0.102"}"#;
    let enc = GzEncoder::new(Vec::new(), Compression::default());
    let mut tar_b = tar::Builder::new(enc);
    append_ustar_file(&mut tar_b, "meta", meta_bytes).unwrap();
    append_ustar_file(&mut tar_b, "project", &proj_bytes).unwrap();
    append_ustar_file(&mut tar_b, &key_tex, tex_bytes).unwrap();
    let initial_pz = tar_b.into_inner().unwrap().finish().unwrap();

    // Import -> Re-export -> Re-import
    let first = import_pz_from_bytes(&initial_pz).expect("initial import");
    assert!(
        first.warnings.is_empty(),
        "first warnings: {:?}",
        first.warnings
    );

    let reexported_bytes = export_pz_to_bytes(&first.project).expect("re-export");
    let second = import_pz_from_bytes(&reexported_bytes).expect("re-import");
    assert!(
        second.warnings.is_empty(),
        "second warnings: {:?}",
        second.warnings
    );

    let p2 = second.project;
    assert_eq!(p2.sequence.name, "Complex Imported");
    assert_eq!(p2.sequence.fps, 60);
    assert!(p2.plugin_requirements.contains("camera-plus"));
    assert!(p2.plugin_requirements.contains("material-plus"));

    let sha_tex = sha256_hex(tex_bytes);
    let LayerKind::Scene3D { ref objects, .. } = p2.sequence.video_tracks[0].clips[0].layer.kind
    else {
        panic!("expected Scene3D");
    };
    assert!(matches!(
        objects[0].material,
        Some(MaterialSpec::CustomPbr(ref m)) if m.uv_normalized && m.texture.as_deref() == Some(sha_tex.as_str())
    ));
    assert!(matches!(
        objects[1].material,
        Some(MaterialSpec::ImagePlus(ref m)) if m.transparent && m.texture.as_deref() == Some(sha_tex.as_str())
    ));

    let cam_layer = &p2.sequence.video_tracks[0].clips[1].layer;
    assert!(matches!(cam_layer.kind, LayerKind::CameraLayer { .. }));
    let cam_objs = &p2.sequence.camera_layer_objects[&cam_layer.id];
    assert_eq!(cam_objs.len(), 1);
    assert_eq!(cam_objs[0].name, "Timeline Cam+");

    // Ensure CustomPbr (non-normalized) & TextureMaterial round-trip cleanly too
    let mut p3 = p2;
    if let LayerKind::Scene3D {
        ref mut objects, ..
    } = p3.sequence.video_tracks[0].clips[0].layer.kind
    {
        objects[0].material = Some(MaterialSpec::CustomPbr(Box::new(CustomPbrMaterial {
            texture: Some(sha_tex.clone()),
            uv_normalized: false,
            ..CustomPbrMaterial::default()
        })));
        objects[1].material = Some(MaterialSpec::Texture(Box::new(TextureMaterial {
            texture: Some(sha_tex.clone()),
            ..TextureMaterial::default()
        })));
    }
    let third_bytes = export_pz_to_bytes(&p3).expect("third export");
    let third = import_pz_from_bytes(&third_bytes).expect("third import");
    assert!(third.warnings.is_empty());
}

#[test]
fn handles_external_missing_assets_effect_groups_shake_and_geometry_plus() {
    let mut project = Project::new_cm3_default();
    project.name = "Edge Cases".to_owned();

    // External preset asset (no binary bytes, NOT listed in missingAssets)
    let ext_key = sha256_hex(b"pz-external:preset-font:bebas.ttf");
    project.assets.insert(
        ext_key.clone(),
        AssetRecord {
            id: ext_key.clone(),
            kind: AssetKind::Font,
            source: AssetSource::Preset,
            filename: "bebas.ttf".to_owned(),
            url: Some("/assets/fonts/2d/bebas.ttf".to_owned()),
            external: true,
            bytes: None,
            missing: false,
            sha256: ext_key.clone(),
            size: 0,
            preserved: BTreeMap::from([(
                "legacy_pz_asset_key".to_owned(),
                Value::String("preset_bebas".to_owned()),
            )]),
        },
    );

    // Missing non-external asset (listed in missingAssets so re-import emits no warning)
    let miss_key = sha256_hex(b"pz-missing:deadbeef:gone.png");
    project.assets.insert(
        miss_key.clone(),
        AssetRecord {
            id: miss_key.clone(),
            kind: AssetKind::Image,
            source: AssetSource::File,
            filename: "gone.png".to_owned(),
            url: None,
            external: false,
            bytes: None,
            missing: true,
            sha256: miss_key.clone(),
            size: 42,
            preserved: BTreeMap::from([(
                "legacy_pz_asset_key".to_owned(),
                Value::String("deadbeefdeadbeefdeadbeefdeadbeef".to_owned()),
            )]),
        },
    );

    // Add EffectGroup, shake effect with nested `shake.*` params, and Geometry+ primitives
    let mut group_fx = create_effect_group("Color Macro");
    let mut shake_fx = create_standard_effect("shake").expect("shake");
    shake_fx
        .params
        .insert("shake.mode".to_owned(), PropertyTrack::option(1));
    if let EffectKind::EffectGroup {
        ref mut children, ..
    } = group_fx.kind
    {
        children.push(shake_fx);
    }

    let clip = &mut project.sequence.video_tracks[0].clips[0];
    clip.layer.effects.push(group_fx);
    if let LayerKind::Scene3D {
        ref mut objects, ..
    } = clip.layer.kind
    {
        objects.push(Object3D::new_shape_3d(
            "OpenCone",
            Shape3DKind::Cone {
                radius: 10.0,
                height: 24.0,
                detail: 20,
                open_ended: true,
                top_radius: 0.0,
                shading: 0,
            },
        ));
        objects.push(Object3D::new_shape_3d(
            "Cap",
            Shape3DKind::Capsule {
                radius: 5.0,
                length: 14.0,
                cap_segments: 6,
                radial_segments: 12,
                shading: 0,
            },
        ));
        objects.push(Object3D::new_shape_3d(
            "Ico",
            Shape3DKind::Icosahedron {
                radius: 9.0,
                detail: 2,
                shading: 0,
            },
        ));
        let mut vec_expr_box = Object3D::new_shape_3d(
            "ExprBox",
            Shape3DKind::Box {
                size: [5.0, 6.0, 7.0],
            },
        );
        vec_expr_box.position = PropertyTrack {
            animated: true,
            expression: Some("[sin(time), 0, cos(time)]".to_owned()),
            static_behavior: crate::model::StaticBehavior::Constant,
            value: PropertyValue::Vec3([1.0, 2.0, 3.0]),
            keyframes: Vec::new(),
            channels: vec![
                PropertyTrack::number(1.0),
                PropertyTrack::number(2.0),
                PropertyTrack::number(3.0),
            ],
        };
        objects.push(vec_expr_box);
    }

    let pz_bytes = export_pz_to_bytes(&project).expect("export edge cases");
    let entries = inspect_tar_entries(&pz_bytes);
    let zoidium: Value = serde_json::from_slice(&entries["zoidium.json"]).unwrap();
    assert_eq!(
        zoidium["missingAssets"],
        json!(["deadbeefdeadbeefdeadbeefdeadbeef"])
    );

    let reimported = import_pz_from_bytes(&pz_bytes).expect("reimport edge cases");
    assert!(
        reimported.warnings.is_empty(),
        "unexpected warnings: {:?}",
        reimported.warnings
    );

    let rp = reimported.project;
    let rclip = &rp.sequence.video_tracks[0].clips[0];
    let EffectKind::EffectGroup { ref children, .. } = rclip.layer.effects[0].kind else {
        panic!("expected EffectGroup");
    };
    assert_eq!(
        children[0].params["shake.mode"].value,
        PropertyValue::Option(1)
    );
    let LayerKind::Scene3D { ref objects, .. } = rclip.layer.kind else {
        panic!("expected Scene3D");
    };
    assert!(matches!(
        objects[1].kind,
        Object3DKind::Shape3D(Shape3DKind::Cone {
            open_ended: true,
            detail: 20,
            ..
        })
    ));
    assert!(matches!(
        objects[2].kind,
        Object3DKind::Shape3D(Shape3DKind::Capsule {
            cap_segments: 6,
            radial_segments: 12,
            ..
        })
    ));
    assert!(matches!(
        objects[3].kind,
        Object3DKind::Shape3D(Shape3DKind::Icosahedron { detail: 2, .. })
    ));
    assert_eq!(
        objects[4].position.expression.as_deref(),
        Some("[sin(time), 0, cos(time)]")
    );
    assert_eq!(
        objects[4].position.value,
        PropertyValue::Vec3([1.0, 2.0, 3.0])
    );
}
