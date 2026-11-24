#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use serde_json::json;
use zc_plugin::PluginRegistry;
use zc_project::{
    AssetKind, AssetRecord, AssetSource, CameraData, CameraDofParams, CameraMotionBlurParams,
    CameraVibrateParams, CompatMode, CustomPbrMaterial, FollowConstraint, GradientStop, LayerKind,
    LayerMask, LightData, LightKind, MaskMode, MaterialSpec, Object3D, Object3DKind,
    PbrPlusMaterial, PresetShapeKind, Project, PropertyKeyframe, PropertyTrack, PropertyValue,
    SequenceControlKind, Shape2DKind, Shape3DKind, StaticBehavior, Text3DData, TrackToConstraint,
    Uuid, VideoClip,
};
use zc_session::{Command, Session, SessionError};

fn find_obj<'a>(proj: &'a Project, id: &str) -> &'a Object3D {
    fn walk<'a>(objs: &'a [Object3D], id: &str) -> Option<&'a Object3D> {
        for o in objs {
            if o.id.to_string() == id {
                return Some(o);
            }
            if let Object3DKind::Group { children, .. } = &o.kind
                && let Some(found) = walk(children, id)
            {
                return Some(found);
            }
        }
        None
    }
    for vt in &proj.sequence.video_tracks {
        for vc in &vt.clips {
            if let LayerKind::Scene3D { objects, .. } = &vc.layer.kind
                && let Some(found) = walk(objects, id)
            {
                return found;
            }
        }
    }
    panic!("object '{id}' not found in root sequence");
}

fn find_clip<'a>(proj: &'a Project, clip_id: &str) -> &'a VideoClip {
    proj.sequence
        .video_tracks
        .iter()
        .flat_map(|t| &t.clips)
        .find(|c| c.id.to_string() == clip_id)
        .unwrap_or_else(|| panic!("video clip '{clip_id}' not found"))
}

#[test]
fn canonical_plugin_object_creation_and_aliases_never_attach_bogus_material_or_box() {
    let mut session = Session::new();
    let scene_id = session.document().video_tracks[0].clips[0].id.clone();

    // 1. All 9 Repeaters & Effector+ (canonical hyphenated + underscore/zoidcraft aliases)
    let repeater_cases = [
        ("zoidium:repeater/repeater", "zoidium:repeater/repeater"),
        (
            "zoidium:repeater/linear-repeater",
            "zoidium:repeater/linear-repeater",
        ),
        (
            "zoidcraft:repeater/linear_repeater",
            "zoidium:repeater/linear-repeater",
        ),
        (
            "zoidium:repeater/random-repeater",
            "zoidium:repeater/random-repeater",
        ),
        (
            "zoidium:repeater/random_repeater",
            "zoidium:repeater/random-repeater",
        ),
        (
            "zoidium:repeater/echo-repeater",
            "zoidium:repeater/echo-repeater",
        ),
        (
            "zoidium:repeater/echo_repeater",
            "zoidium:repeater/echo-repeater",
        ),
        ("zoidium:repeater/twist", "zoidium:repeater/twist"),
        ("zoidium:repeater/warp", "zoidium:repeater/warp"),
        (
            "zoidium:repeater/voronoi-fracture",
            "zoidium:repeater/voronoi-fracture",
        ),
        (
            "zoidium:repeater/voronoi_fracture",
            "zoidium:repeater/voronoi-fracture",
        ),
        ("zoidium:repeater/plain", "zoidium:repeater/plain"),
        ("zoidium:repeater/delay", "zoidium:repeater/delay"),
    ];
    for (input_kind, expected_canonical) in repeater_cases {
        let rep_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: None,
                kind: input_kind.to_owned(),
                name: None,
            })
            .unwrap_or_else(|e| panic!("add repeater {input_kind}: {e}"))["object_id"]
            .as_str()
            .unwrap()
            .to_owned();

        // Must be a valid 3D container so children can be parented underneath
        let child_id = session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: Some(rep_id.clone()),
                kind: "box".to_owned(),
                name: Some("Child".to_owned()),
            })
            .unwrap_or_else(|e| panic!("parent child under {input_kind}: {e}"))["object_id"]
            .as_str()
            .unwrap()
            .to_owned();

        let proj = session.to_project();
        let rep = find_obj(&proj, &rep_id);
        assert!(
            matches!(rep.kind, Object3DKind::Group { ref children, .. } if children.len() == 1 && children[0].id.to_string() == child_id),
            "{input_kind} must sync as Group containing its child"
        );
        assert!(
            rep.material.is_none(),
            "{input_kind} must NEVER attach bogus PBR material"
        );
        assert_eq!(
            rep.preserved.get("zoidiumPluginObjectType"),
            Some(&json!(expected_canonical))
        );
        if matches!(
            expected_canonical,
            "zoidium:repeater/twist"
                | "zoidium:repeater/warp"
                | "zoidium:repeater/voronoi-fracture"
                | "zoidium:repeater/plain"
        ) {
            assert!(
                rep.preserved.contains_key("property.field"),
                "{input_kind} must initialize default Effector+ field properties"
            );
        } else if expected_canonical == "zoidium:repeater/delay" {
            assert!(
                rep.preserved.contains_key("property.strength")
                    && rep.preserved.contains_key("property.window"),
                "delay effector must initialize strength and window properties"
            );
        } else {
            assert!(
                rep.preserved.contains_key("property.count"),
                "{input_kind} must initialize default repeater count property"
            );
        }
    }

    // 2. All 8 Geometry+ primitives (canonical + aliases) -> real Shape3DKind variants
    let geo_cases: Vec<(&str, &str)> = vec![
        ("zoidium:geometry-plus/rounded-box", "rounded_box"),
        ("zoidium:geometry-plus/rounded_box", "rounded_box"),
        ("zoidium:geometry-plus/polyhedron", "polyhedron"),
        ("zoidium:geometry-plus/icosahedron", "icosahedron"),
        ("zoidium:geometry-plus/octahedron", "octahedron"),
        ("zoidium:geometry-plus/dodecahedron", "dodecahedron"),
        ("zoidium:geometry-plus/tetrahedron", "tetrahedron"),
        ("zoidium:geometry-plus/cone", "cone"),
        ("zoidium:geometry-plus/capsule", "capsule"),
        ("zoidium:geometry-plus/tube", "tube"),
        ("zoidium:geometry-plus/gear", "gear"),
        ("zoidium:geometry-plus/helix", "helix"),
        ("zoidium:geometry-plus/spline", "spline"),
        ("zoidium:geometry-plus/torus_knot", "torus_knot"),
    ];
    for (input_kind, tag) in geo_cases {
        let oid = session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: None,
                kind: input_kind.to_owned(),
                name: None,
            })
            .unwrap_or_else(|e| panic!("add geometry {input_kind}: {e}"))["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let proj = session.to_project();
        let obj = find_obj(&proj, &oid);
        let Object3DKind::Shape3D(ref shape) = obj.kind else {
            panic!("{input_kind} must create Shape3D, got {:?}", obj.kind);
        };
        assert!(
            !matches!(shape, Shape3DKind::Box { .. }),
            "{input_kind} must never fall back to Shape3DKind::Box"
        );
        match tag {
            "rounded_box" => assert!(matches!(shape, Shape3DKind::RoundedBox { .. })),
            "polyhedron" | "tetrahedron" => {
                assert!(matches!(shape, Shape3DKind::Tetrahedron { .. }))
            }
            "icosahedron" => assert!(matches!(shape, Shape3DKind::Icosahedron { .. })),
            "octahedron" => assert!(matches!(shape, Shape3DKind::Octahedron { .. })),
            "dodecahedron" => assert!(matches!(shape, Shape3DKind::Dodecahedron { .. })),
            "cone" => assert!(matches!(shape, Shape3DKind::Cone { .. })),
            "capsule" => assert!(matches!(shape, Shape3DKind::Capsule { .. })),
            "tube" => assert!(matches!(shape, Shape3DKind::Tube { .. })),
            "gear" => assert!(matches!(shape, Shape3DKind::Gear { .. })),
            "helix" => assert!(matches!(shape, Shape3DKind::Helix { .. })),
            "spline" => assert!(matches!(shape, Shape3DKind::Spline { .. })),
            "torus_knot" => assert!(matches!(shape, Shape3DKind::TorusKnot { .. })),
            _ => unreachable!(),
        }
    }

    // 3. Camera+ (canonical + underscore alias)
    for cam_kind in [
        "zoidium:camera-plus/camera",
        "zoidium:camera-plus/camera_plus",
    ] {
        let cid = session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: None,
                kind: cam_kind.to_owned(),
                name: None,
            })
            .expect("add camera+")["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let proj = session.to_project();
        let cam = find_obj(&proj, &cid);
        assert!(matches!(cam.kind, Object3DKind::Camera(_)));
        assert_eq!(
            cam.preserved.get("zoidiumPluginObjectType"),
            Some(&json!("zoidium:camera-plus/camera"))
        );
    }

    // 4. All 7 Text+ classes -> container Group with zoidiumPluginObjectType
    for tp in [
        "zoidium:text-plus/text-plus",
        "zoidium:text-plus/gradient-transform",
        "zoidium:text-plus/character-shake",
        "zoidium:text-plus/selected-transform",
        "zoidium:text-plus/selected-shake",
        "zoidium:text-plus/random-scatter",
        "zoidium:text-plus/character-material",
    ] {
        let id = session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: None,
                kind: tp.to_owned(),
                name: None,
            })
            .expect("add text+")["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: Some(id.clone()),
                kind: "text3d".to_owned(),
                name: None,
            })
            .unwrap_or_else(|e| panic!("parent under {tp}: {e}"));
        let proj = session.to_project();
        let obj = find_obj(&proj, &id);
        assert!(matches!(
            obj.kind,
            Object3DKind::Group { ref children, .. } if children.len() == 1
        ));
        assert!(obj.material.is_none());
        assert_eq!(
            obj.preserved.get("zoidiumPluginObjectType"),
            Some(&json!(tp))
        );
    }

    // 5. All 12 ExpressText+ classes -> container Group with zoidiumPluginObjectType
    for etp in [
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
    ] {
        let id = session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: None,
                kind: etp.to_owned(),
                name: None,
            })
            .expect("add expresstext+")["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: Some(id.clone()),
                kind: "text3d".to_owned(),
                name: None,
            })
            .unwrap_or_else(|e| panic!("parent under {etp}: {e}"));
        let proj = session.to_project();
        let obj = find_obj(&proj, &id);
        assert!(matches!(
            obj.kind,
            Object3DKind::Group { ref children, .. } if children.len() == 1
        ));
        assert!(obj.material.is_none());
        assert_eq!(
            obj.preserved.get("zoidiumPluginObjectType"),
            Some(&json!(etp))
        );
    }

    // 6. Optical Flares, Trapcode Suite, Light+ / Ambient
    for plugin_leaf in [
        "zoidium:optical-flares/flare",
        "zoidium:trapcode-suite/particular",
        "zoidium:trapcode-suite/form",
        "zoidium:trapcode-suite/plexus",
        "zoidium:trapcode/particular",
    ] {
        let id = session
            .execute(Command::AddObject3D {
                clip_id: scene_id.clone(),
                parent_id: None,
                kind: plugin_leaf.to_owned(),
                name: None,
            })
            .expect("add optical flare / trapcode")["object_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let proj = session.to_project();
        let obj = find_obj(&proj, &id);
        assert!(
            matches!(obj.kind, Object3DKind::Dummy { .. }),
            "{plugin_leaf} must sync as Dummy, never Box: {:?}",
            obj.kind
        );
        assert!(obj.material.is_none());
        assert!(obj.preserved.contains_key("zoidiumPluginObjectType"));
    }

    let amb_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "zoidium:light-plus/ambient".to_owned(),
            name: Some("Ambient".to_owned()),
        })
        .expect("add ambient light")["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let proj = session.to_project();
    let amb = find_obj(&proj, &amb_id);
    assert!(
        matches!(amb.kind, Object3DKind::Light(ref l) if l.kind == LightKind::Ambient),
        "expected Ambient light, got {:?}",
        amb.kind
    );

    // 7. Particles defaults match Zoidium particle-defaults.js
    let pt_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_id,
            parent_id: None,
            kind: "particles".to_owned(),
            name: None,
        })
        .expect("add particles")["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let proj = session.to_project();
    let pt_obj = find_obj(&proj, &pt_id);
    let Object3DKind::Particles(ref pt) = pt_obj.kind else {
        panic!("expected Particles, got {:?}", pt_obj.kind);
    };
    assert_eq!(pt.number, 400);
    assert_eq!(pt.rate, 200.0);
    assert_eq!(pt.lifetime, 3.0);
    assert_eq!(pt.pspread, [20.0, 20.0, 20.0]);
    assert_eq!(pt.vspread, [100.0, 100.0, 100.0]);
    assert_eq!(pt.time.expression.as_deref(), Some("time"));
    assert_eq!(pt.sprite_preset.as_deref(), Some("circle_soft"));
}

#[test]
fn clip_creation_defaults_shape_plus_shape2d_and_precomp() {
    let mut session = Session::new();
    let vtrack = session.document().video_tracks[0].id.clone();

    // Shape+ clip -> default_shape_plus_payload() (NOT {})
    let sp_clip_id = session
        .execute(Command::AddClip {
            track_id: vtrack.clone(),
            kind: "zoidium:shape-plus/shape-layer".to_owned(),
            name: Some("Shape+ Layer".to_owned()),
            start: 100.0,
            length: 60.0,
        })
        .expect("add shape+ clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();

    // Shape2D clip -> canonical rectangle path commands (NOT {"preset":"rectangle"})
    let s2d_clip_id = session
        .execute(Command::AddClip {
            track_id: vtrack.clone(),
            kind: "shape2d".to_owned(),
            name: Some("Rect".to_owned()),
            start: 170.0,
            length: 60.0,
        })
        .expect("add shape2d clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();

    // Precomp clip -> real Sequence in project.compositions, non-nil composition_id
    let pc_clip_id = session
        .execute(Command::AddClip {
            track_id: vtrack,
            kind: "precomp".to_owned(),
            name: Some("My Precomp".to_owned()),
            start: 240.0,
            length: 60.0,
        })
        .expect("add precomp clip")["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();

    let proj = session.to_project();

    let sp_clip = find_clip(&proj, &sp_clip_id);
    let LayerKind::Plugin {
        ref plugin_type,
        ref payload,
    } = sp_clip.layer.kind
    else {
        panic!("expected LayerKind::Plugin for Shape+");
    };
    assert_eq!(plugin_type, "zoidium:shape-plus/shape-layer");
    assert_eq!(
        payload,
        &zc_render::shape_plus::default_shape_plus_payload()
    );
    assert!(
        payload
            .get("contents")
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty()),
        "Shape+ payload must contain default rectangle contents"
    );

    let s2d_clip = find_clip(&proj, &s2d_clip_id);
    let LayerKind::Shape2D { ref shapes } = s2d_clip.layer.kind else {
        panic!("expected LayerKind::Shape2D");
    };
    assert_eq!(shapes.len(), 1);
    let Shape2DKind::Path { ref path } = shapes[0].kind else {
        panic!("expected Shape2DKind::Path");
    };
    assert!(path.get("preset").is_none());
    let cmds = path
        .get("commands")
        .and_then(|v| v.as_array())
        .expect("canonical rectangle path commands");
    assert_eq!(cmds.len(), 5);
    assert_eq!(cmds[0].get("type").and_then(|v| v.as_str()), Some("M"));
    assert_eq!(cmds[4].get("type").and_then(|v| v.as_str()), Some("Z"));

    let pc_clip = find_clip(&proj, &pc_clip_id);
    let LayerKind::PrecompRef { composition_id } = pc_clip.layer.kind else {
        panic!("expected LayerKind::PrecompRef");
    };
    assert_ne!(composition_id, Uuid::nil());
    let comp = proj
        .compositions
        .get(&composition_id)
        .expect("allocated composition must exist in project.compositions");
    assert_eq!(comp.name, "My Precomp");
}

#[test]
fn lossless_3d_property_sync_across_all_pro_types() {
    let mut proj = Project::new_cm3_default();

    let mut particles = Object3D::new("FX Particles", Object3DKind::Particles(Box::default()));
    particles.id = Uuid::new_v4();

    let mut rbox = Object3D::new(
        "RBox",
        Object3DKind::Shape3D(Shape3DKind::RoundedBox {
            size: [2.0, 2.0, 2.0],
            radius: 0.25,
            segments: 4,
            corner_radius: Some(0.25),
            corner_segments: Some(4),
            bevel_size: Some(0.1),
            bevel_segments: Some(2),
            shading: 1,
        }),
    );
    rbox.material = Some(MaterialSpec::PbrPlus(Box::new(PbrPlusMaterial {
        ao_texture: Some("asset_ao_0".to_owned()),
        ..PbrPlusMaterial::default()
    })));
    rbox.constraints.track_to = Some(TrackToConstraint {
        target_id: "target_a".to_owned(),
        track_axis: 2,
        up_axis: 1,
        influence: PropertyTrack::number(100.0),
    });
    rbox.constraints.follow = Some(FollowConstraint {
        target_id: "target_b".to_owned(),
        space: 0,
        follow_position: true,
        follow_rotation: false,
        follow_scale: true,
        position_offset: PropertyTrack::vec3([1.0, 2.0, 3.0]),
        rotation_offset: PropertyTrack::vec3([0.0, 0.0, 0.0]),
        scale_offset: PropertyTrack::vec3([1.0, 1.0, 1.0]),
        fallback_pose: None,
    });

    let mut cam = Object3D::new(
        "ProCam",
        Object3DKind::Camera(Box::new(CameraData {
            active: true,
            zoom: PropertyTrack::number(1.2),
            focal_length: PropertyTrack::number(50.0),
            film_gate: 36.0,
            film_offset_x: PropertyTrack::number(0.5),
            film_offset_y: PropertyTrack::number(-0.5),
            motion_blur: Some(CameraMotionBlurParams {
                enabled: true,
                samples: 12,
                shutter: PropertyTrack::number(0.5),
            }),
            dof: Some(CameraDofParams {
                enabled: true,
                enabled_track: None,
                focus_distance: PropertyTrack::number(12.0),
                aperture: PropertyTrack::number(4.0),
                focus_area_width: PropertyTrack::number(2.0),
                near_blur: PropertyTrack::number(110.0),
                far_blur: PropertyTrack::number(120.0),
            }),
            vibrate: Some(CameraVibrateParams {
                enabled: true,
                seed: 42,
                position_amplitude: PropertyTrack::vec3([1.0, 2.0, 3.0]),
                rotation_amplitude: PropertyTrack::vec3([0.1, 0.2, 0.3]),
                frequency: PropertyTrack::number(2.0),
                regular_pulse: true,
                relative: false,
                enable_position: true,
                enable_rotation: true,
                position_frequency: PropertyTrack::number(2.5),
                rotation_frequency: PropertyTrack::number(3.5),
            }),
            ..CameraData::default()
        })),
    );
    cam.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        json!("zoidium:camera-plus/camera"),
    );

    let light = Object3D::new(
        "KeyLight",
        Object3DKind::Light(Box::new(LightData {
            kind: LightKind::Directional,
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            ground_color: PropertyTrack::color([0.2, 0.2, 0.2]),
            target: PropertyTrack::vec3([0.0, 0.0, 0.0]),
            intensity: PropertyTrack::number(2.0),
            angle: PropertyTrack::number(60.0),
            penumbra: 0.5,
            decay: 1.0,
            distance: 0.0,
            cast_shadow: true,
            shadow_map_size: 1024,
            shadow_bias: -0.001,
        })),
    );

    let mut text = Object3D::new(
        "Title3D",
        Object3DKind::Text3D(Box::new(Text3DData {
            text: "HELLO".to_owned(),
            font: Some("Inter".to_owned()),
            size: [20.0, 3.0],
            detail: 5,
            bevel: true,
            bevel_size: 0.4,
            bevel_thickness: 0.2,
            spacing: 0.1,
            bevel_side: 0,
            bevel_detail: 3,
            bevel_profile: 0,
            bevel_tension: 0.5,
            text_scale_mode: CompatMode::Current,
            ..Text3DData::default()
        })),
    );
    text.material_slots.insert(
        "front".to_owned(),
        MaterialSpec::CustomPbr(Box::new(CustomPbrMaterial {
            uv_normalized: true,
            ..CustomPbrMaterial::default()
        })),
    );
    text.material_slots.insert(
        "side".to_owned(),
        MaterialSpec::NodeMaterial {
            graph: json!({"nodes": [{"id": "n1"}]}),
            resolution: 512,
            schema_version: 2,
        },
    );

    let particles_id = particles.id.to_string();
    let rbox_id = rbox.id.to_string();
    let cam_id = cam.id.to_string();
    let light_id = light.id.to_string();
    let text_id = text.id.to_string();

    if let LayerKind::Scene3D {
        ref mut objects, ..
    } = proj.sequence.video_tracks[0].clips[0].layer.kind
    {
        objects.push(particles);
        objects.push(rbox);
        objects.push(cam);
        objects.push(light);
        objects.push(text);
    }

    let mut session = Session::from_project(proj, PluginRegistry::new());
    let scene_id = session.document().video_tracks[0].clips[0].id.clone();

    // Edit Particles (including color gradient JSON ramp & spritePreset)
    session
        .execute(Command::SetPropertyValue {
            target_id: particles_id.clone(),
            property: "particles.number".to_owned(),
            value: json!(750),
        })
        .unwrap();
    session
        .execute(Command::SetPropertyValue {
            target_id: particles_id.clone(),
            property: "particles.spritePreset".to_owned(),
            value: json!("star_glow"),
        })
        .unwrap();
    let color_stops = vec![
        GradientStop {
            position: 0.0,
            color: "#ff0000".to_owned(),
        },
        GradientStop {
            position: 1.0,
            color: "#0000ff".to_owned(),
        },
    ];
    session
        .execute(Command::SetPropertyValue {
            target_id: particles_id.clone(),
            property: "particles.color".to_owned(),
            value: json!(serde_json::to_string(&color_stops).unwrap()),
        })
        .unwrap();

    // Edit RoundedBox, PbrPlus aoMap alias + surface props, TrackTo influence & Follow followSpace
    for (prop, val) in [
        ("roundedBox.cornerRadius", json!(0.45)),
        ("roundedBox.bevelSegments", json!(5)),
        ("roundedBox.shading", json!(0)),
        ("material.aoMap", json!("asset_ao_new")),
        ("material.aoEnabled", json!(1)),
        ("material.celEnabled", json!(1)),
        ("material.fresnelEnabled", json!(1)),
        ("material.projection", json!(2)),
        ("constraint.trackTo.influence", json!(65.0)),
        ("constraint.follow.followSpace", json!(1)),
    ] {
        session
            .execute(Command::SetPropertyValue {
                target_id: rbox_id.clone(),
                property: prop.to_owned(),
                value: val,
            })
            .unwrap();
    }

    // Edit Camera+ (DOF aperture, focalLength, vibrate fields)
    for (prop, val) in [
        ("focalLength", json!(85.0)),
        ("dofAperture", json!(18.5)),
        ("vibrate.positionFrequency", json!(6.25)),
        ("vibrate.rotationFrequency", json!(4.75)),
        ("vibrate.regularPulse", json!(0)),
        ("vibrate.relative", json!(1)),
    ] {
        session
            .execute(Command::SetPropertyValue {
                target_id: cam_id.clone(),
                property: prop.to_owned(),
                value: val,
            })
            .unwrap();
    }

    // Edit Light -> Ambient + shadowMapSize + shadowBias
    for (prop, val) in [
        ("light.kind", json!(4)),
        ("shadowMapSize", json!(2048)),
        ("shadowBias", json!(-0.0025)),
        ("castShadow", json!(0)),
    ] {
        session
            .execute(Command::SetPropertyValue {
                target_id: light_id.clone(),
                property: prop.to_owned(),
                value: val,
            })
            .unwrap();
    }

    // Edit Text3D + frontMaterial & sideMaterial slots + character props
    for (prop, val) in [
        ("spacing", json!(0.35)),
        ("bevelSide", json!(1)),
        ("bevelDetail", json!(6)),
        ("bevelProfile", json!(1)),
        ("bevelTension", json!(0.8)),
        ("kerning", json!(0)),
        ("characterIndexOffset", json!(3.0)),
        ("characterCountOverride", json!(12.0)),
        ("textScaleMode", json!(1)),
        ("frontMaterial.uvNormalized", json!(0)),
        ("sideMaterial.bakeResolution", json!(1024)),
        ("sideMaterial.graph", json!(r#"{"nodes":[{"id":"n2"}]}"#)),
    ] {
        session
            .execute(Command::SetPropertyValue {
                target_id: text_id.clone(),
                property: prop.to_owned(),
                value: val,
            })
            .unwrap();
    }

    // Add a Repeater & edit all 10 field props including fieldNoiseEvolution -> mirrors fieldEvolution
    let rep_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "zoidium:repeater/plain".to_owned(),
            name: Some("PlainEff".to_owned()),
        })
        .unwrap()["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    for (prop, val) in [
        ("field", json!(2)),
        ("fieldPosition", json!([1.0, 2.0, 3.0])),
        ("fieldRotation", json!([0.1, 0.2, 0.3])),
        ("fieldScale", json!([2.0, 2.0, 2.0])),
        ("fieldFalloff", json!(0.4)),
        ("fieldCurve", json!(3)),
        ("fieldInvert", json!(1)),
        ("fieldSweep", json!(1.5)),
        ("fieldNoiseScale", json!(4.2)),
        ("fieldNoiseEvolution", json!(9.75)),
    ] {
        session
            .execute(Command::SetPropertyValue {
                target_id: rep_id.clone(),
                property: prop.to_owned(),
                value: val,
            })
            .unwrap();
    }

    // Add Gear & edit its unique property
    let gear_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_id.clone(),
            parent_id: None,
            kind: "zoidium:geometry-plus/gear".to_owned(),
            name: None,
        })
        .unwrap()["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: gear_id.clone(),
            property: "gear.teeth".to_owned(),
            value: json!(24),
        })
        .unwrap();

    // Add Optical Flare & edit preserved plugin property
    let flare_id = session
        .execute(Command::AddObject3D {
            clip_id: scene_id,
            parent_id: None,
            kind: "zoidium:optical-flares/flare".to_owned(),
            name: None,
        })
        .unwrap()["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: flare_id.clone(),
            property: "brightness".to_owned(),
            value: json!(2.75),
        })
        .unwrap();

    // Verify synced Project AND reload round-trip
    let synced = session.to_project();

    let pt_out = find_obj(&synced, &particles_id);
    if let Object3DKind::Particles(ref pt) = pt_out.kind {
        assert_eq!(pt.number, 750);
        assert_eq!(pt.sprite_preset.as_deref(), Some("star_glow"));
        assert_eq!(pt.color.len(), 2);
    } else {
        panic!("expected Particles");
    }

    let rbox_out = find_obj(&synced, &rbox_id);
    assert!(matches!(
        rbox_out.kind,
        Object3DKind::Shape3D(Shape3DKind::RoundedBox {
            corner_radius: Some(cr),
            bevel_segments: Some(5),
            shading: 0,
            ..
        }) if (cr - 0.45).abs() < 1e-6
    ));
    let Some(MaterialSpec::PbrPlus(ref pbr)) = rbox_out.material else {
        panic!("expected PbrPlus");
    };
    assert_eq!(pbr.ao_texture.as_deref(), Some("asset_ao_new"));
    assert_eq!(pbr.preserved.get("aoEnabled"), Some(&json!(1)));
    assert_eq!(pbr.preserved.get("celEnabled"), Some(&json!(1)));
    assert_eq!(pbr.preserved.get("fresnelEnabled"), Some(&json!(1)));
    assert_eq!(pbr.preserved.get("projection"), Some(&json!(2)));
    assert_eq!(
        rbox_out
            .constraints
            .track_to
            .as_ref()
            .unwrap()
            .influence
            .value,
        PropertyValue::Number(65.0)
    );
    assert_eq!(rbox_out.constraints.follow.as_ref().unwrap().space, 1);

    let cam_out = find_obj(&synced, &cam_id);
    let Object3DKind::Camera(ref c) = cam_out.kind else {
        panic!("expected Camera");
    };
    assert_eq!(c.focal_length.value, PropertyValue::Number(85.0));
    assert_eq!(
        c.dof.as_ref().unwrap().aperture.value,
        PropertyValue::Number(18.5)
    );
    let vib = c.vibrate.as_ref().unwrap();
    assert!(!vib.regular_pulse);
    assert!(vib.relative);
    assert_eq!(vib.position_frequency.value, PropertyValue::Number(6.25));
    assert_eq!(vib.rotation_frequency.value, PropertyValue::Number(4.75));

    let light_out = find_obj(&synced, &light_id);
    let Object3DKind::Light(ref l) = light_out.kind else {
        panic!("expected Light");
    };
    assert_eq!(l.kind, LightKind::Ambient);
    assert_eq!(l.shadow_map_size, 2048);
    assert!((l.shadow_bias - (-0.0025)).abs() < 1e-6);
    assert!(!l.cast_shadow);

    let text_out = find_obj(&synced, &text_id);
    let Object3DKind::Text3D(ref td) = text_out.kind else {
        panic!("expected Text3D");
    };
    assert!((td.spacing - 0.35).abs() < 1e-6);
    assert_eq!(td.bevel_side, 1);
    assert_eq!(td.bevel_detail, 6);
    assert_eq!(td.bevel_profile, 1);
    assert!((td.bevel_tension - 0.8).abs() < 1e-6);
    assert_eq!(td.character_index_offset.value, PropertyValue::Number(3.0));
    assert_eq!(
        td.character_count_override.value,
        PropertyValue::Number(12.0)
    );
    assert_eq!(td.text_scale_mode, CompatMode::Legacy);
    assert_eq!(text_out.preserved.get("kerning"), Some(&json!(0)));
    assert!(matches!(
        text_out.material_slots.get("front"),
        Some(MaterialSpec::CustomPbr(m)) if !m.uv_normalized
    ));
    assert!(matches!(
        text_out.material_slots.get("side"),
        Some(MaterialSpec::NodeMaterial {
            resolution: 1024,
            ..
        })
    ));

    let rep_out = find_obj(&synced, &rep_id);
    assert_eq!(rep_out.preserved.get("property.field"), Some(&json!(2)));
    assert_eq!(
        rep_out.preserved.get("property.fieldNoiseEvolution"),
        Some(&json!(9.75))
    );
    assert_eq!(
        rep_out.preserved.get("property.fieldEvolution"),
        Some(&json!(9.75))
    );

    let gear_out = find_obj(&synced, &gear_id);
    assert!(matches!(
        gear_out.kind,
        Object3DKind::Shape3D(Shape3DKind::Gear { teeth: 24, .. })
    ));

    let flare_out = find_obj(&synced, &flare_id);
    assert_eq!(
        flare_out.preserved.get("property.brightness"),
        Some(&json!(2.75))
    );

    // Reload from synced project and confirm lossless round-trip
    let mut reloaded = Session::from_project(synced.clone(), PluginRegistry::new());
    assert_eq!(
        reloaded.to_project().sequence.video_tracks[0].clips[0]
            .layer
            .kind,
        synced.sequence.video_tracks[0].clips[0].layer.kind
    );
}

#[test]
fn masks_shape_plus_preserved_and_split_duplicate_preserve_side_tables() {
    let mut session = Session::new();
    let vtrack = session.document().video_tracks[0].id.clone();

    let clip_id_str = session
        .execute(Command::AddClip {
            track_id: vtrack.clone(),
            kind: "zoidium:shape-plus/shape-layer".to_owned(),
            name: Some("Masked Shape+".to_owned()),
            start: 10.0,
            length: 80.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip_uuid = Uuid::parse_str(&clip_id_str).unwrap();

    // UpdateShapePlusPayload + undo/redo
    let custom_payload = json!({
        "contents": [{"type": "ellipse", "size": [200, 120]}],
        "version": 2
    });
    session
        .execute(Command::UpdateShapePlusPayload {
            clip_id: clip_uuid,
            payload: custom_payload.clone(),
        })
        .unwrap();
    assert!(matches!(
        find_clip(&session.to_project(), &clip_id_str).layer.kind,
        LayerKind::Plugin { ref payload, .. } if payload == &custom_payload
    ));
    session.execute(Command::Undo).unwrap();
    assert!(matches!(
        find_clip(&session.to_project(), &clip_id_str).layer.kind,
        LayerKind::Plugin { ref payload, .. } if payload == &zc_render::shape_plus::default_shape_plus_payload()
    ));
    session.execute(Command::Redo).unwrap();

    // UpdateLayerPreserved
    session
        .execute(Command::UpdateLayerPreserved {
            clip_id: clip_uuid,
            key: "customMeta".to_owned(),
            value: Some(json!({"tag": "hero"})),
        })
        .unwrap();
    assert_eq!(
        find_clip(&session.to_project(), &clip_id_str)
            .layer
            .preserved
            .get("customMeta"),
        Some(&json!({"tag": "hero"}))
    );

    // AddLayerMask with animated path keyframes at frames 10 and 50
    let mut mask1 = LayerMask {
        id: "m1".to_owned(),
        name: "Primary Mask".to_owned(),
        enabled: true,
        mode: MaskMode::Add,
        invert: false,
        feather: 5.0,
        expansion: 0.0,
        opacity: 1.0,
        closed: true,
        locked: false,
        outline_color: "#58a6ff".to_owned(),
        vertex_ids: Vec::new(),
        vertex_modes: BTreeMap::new(),
        path: PropertyTrack {
            animated: true,
            expression: None,
            static_behavior: StaticBehavior::Constant,
            value: PropertyValue::Json(json!({"commands": []})),
            keyframes: vec![
                PropertyKeyframe::linear(10.0, PropertyValue::Json(json!({"commands": []}))),
                PropertyKeyframe::linear(50.0, PropertyValue::Json(json!({"commands": []}))),
            ],
            channels: Vec::new(),
        },
        preserved: BTreeMap::new(),
    };
    session
        .execute(Command::AddLayerMask {
            clip_id: clip_uuid,
            mask: mask1.clone(),
        })
        .unwrap();

    // DuplicateLayerMask + UpdateLayerMask + ReorderLayerMask + RemoveLayerMask
    session
        .execute(Command::DuplicateLayerMask {
            clip_id: clip_uuid,
            mask_id: "m1".to_owned(),
        })
        .unwrap();
    mask1.name = "Renamed Primary".to_owned();
    mask1.invert = true;
    session
        .execute(Command::UpdateLayerMask {
            clip_id: clip_uuid,
            mask_id: "m1".to_owned(),
            mask: mask1.clone(),
        })
        .unwrap();
    session
        .execute(Command::ReorderLayerMask {
            clip_id: clip_uuid,
            mask_id: "m1".to_owned(),
            to_index: 1,
        })
        .unwrap();
    let proj = session.to_project();
    let lid = find_clip(&proj, &clip_id_str).layer.id;
    let masks = proj.sequence.layer_masks.get(&lid).unwrap();
    assert_eq!(masks.len(), 2);
    assert_eq!(masks[1].name, "Renamed Primary");
    assert!(masks[1].invert);

    // SplitClip at frame 40 (offset = +30 from clip start 10) -> right clip MUST get shifted mask keyframes (-30)
    let split_res = session
        .execute(Command::SplitClip {
            clip_id: clip_id_str.clone(),
            at_frame: 40.0,
        })
        .unwrap();
    let right_clip_id = split_res["right_clip_id"].as_str().unwrap().to_owned();
    let proj_after_split = session.to_project();
    let right_lid = find_clip(&proj_after_split, &right_clip_id).layer.id;
    let right_masks = proj_after_split
        .sequence
        .layer_masks
        .get(&right_lid)
        .expect("split right half must inherit layer_masks");
    assert_eq!(right_masks.len(), 2);
    assert_eq!(right_masks[1].path.keyframes[0].frame, -20.0);
    assert_eq!(right_masks[1].path.keyframes[1].frame, 20.0);

    // DuplicateClip -> duplicated clip MUST inherit layer_masks on its own new Layer.id
    let dup_id = session
        .execute(Command::DuplicateClip {
            clip_id: clip_id_str.clone(),
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let proj_after_dup = session.to_project();
    let dup_lid = find_clip(&proj_after_dup, &dup_id).layer.id;
    assert_ne!(dup_lid, lid);
    assert_eq!(
        proj_after_dup
            .sequence
            .layer_masks
            .get(&dup_lid)
            .map(Vec::len),
        Some(2)
    );

    // RemoveLayerMask + undo
    session
        .execute(Command::RemoveLayerMask {
            clip_id: clip_uuid,
            mask_id: "m1".to_owned(),
        })
        .unwrap();
    assert_eq!(
        session
            .to_project()
            .sequence
            .layer_masks
            .get(&lid)
            .map(Vec::len),
        Some(1)
    );
    session.execute(Command::Undo).unwrap();
    assert_eq!(
        session
            .to_project()
            .sequence
            .layer_masks
            .get(&lid)
            .map(Vec::len),
        Some(2)
    );

    // SetPresetShapeKind on a PresetShape2D clip
    let preset_id = session
        .execute(Command::AddClip {
            track_id: vtrack,
            kind: "preset_shape".to_owned(),
            name: Some("StarClip".to_owned()),
            start: 200.0,
            length: 40.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let preset_uuid = Uuid::parse_str(&preset_id).unwrap();
    session
        .execute(Command::SetPresetShapeKind {
            clip_id: preset_uuid,
            shape_kind: PresetShapeKind::Rectangle,
        })
        .unwrap();
    assert!(matches!(
        find_clip(&session.to_project(), &preset_id).layer.kind,
        LayerKind::PresetShape2D {
            preset: PresetShapeKind::Rectangle,
            ..
        }
    ));

    // Precomp clip split preserves precomp_offsets on right half
    let pc_id = session
        .execute(Command::AddClip {
            track_id: session.document().video_tracks[0].id.clone(),
            kind: "precomp".to_owned(),
            name: Some("OffsetPrecomp".to_owned()),
            start: 260.0,
            length: 60.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    session
        .execute(Command::SetPropertyValue {
            target_id: pc_id.clone(),
            property: "precompOffset".to_owned(),
            value: json!(12.5),
        })
        .unwrap();
    let pc_right_id = session
        .execute(Command::SplitClip {
            clip_id: pc_id,
            at_frame: 290.0,
        })
        .unwrap()["right_clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let proj_pc = session.to_project();
    let pc_right_lid = find_clip(&proj_pc, &pc_right_id).layer.id;
    assert_eq!(
        proj_pc.sequence.precomp_offsets.get(&pc_right_lid),
        Some(&12.5)
    );
}

#[test]
fn precomp_composition_lifecycle_and_active_composition_editing() {
    let mut session = Session::new();
    let vtrack = session.document().video_tracks[0].id.clone();

    // CreateComposition
    let create_res = session
        .execute(Command::CreateComposition {
            name: "Comp A".to_owned(),
            resolution: [1280, 720],
            fps: 60,
            length_frames: 240,
        })
        .unwrap();
    let comp_a_id = Uuid::parse_str(create_res["comp_id"].as_str().unwrap()).unwrap();
    assert_eq!(session.to_project().compositions[&comp_a_id].name, "Comp A");

    // RenameComposition
    session
        .execute(Command::RenameComposition {
            comp_id: comp_a_id,
            name: "Comp Alpha".to_owned(),
        })
        .unwrap();
    assert_eq!(
        session.to_project().compositions[&comp_a_id].name,
        "Comp Alpha"
    );

    // DuplicateComposition
    let dup_res = session
        .execute(Command::DuplicateComposition { comp_id: comp_a_id })
        .unwrap();
    let comp_b_id = Uuid::parse_str(dup_res["comp_id"].as_str().unwrap()).unwrap();
    assert_ne!(comp_b_id, comp_a_id);
    assert!(session.to_project().compositions.contains_key(&comp_b_id));

    // Add a clip in root and PrecomposeClips
    let c1_str = session
        .execute(Command::AddClip {
            track_id: vtrack,
            kind: "text2d".to_owned(),
            name: Some("Nested Title".to_owned()),
            start: 100.0,
            length: 40.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let c1_uuid = Uuid::parse_str(&c1_str).unwrap();
    let precomp_res = session
        .execute(Command::PrecomposeClips {
            clip_ids: vec![c1_uuid],
            name: "Title Precomp".to_owned(),
        })
        .unwrap();
    let precomp_comp_id = Uuid::parse_str(precomp_res["comp_id"].as_str().unwrap()).unwrap();
    let precomp_clip_str = precomp_res["clip_id"].as_str().unwrap().to_owned();
    let precomp_clip_uuid = Uuid::parse_str(&precomp_clip_str).unwrap();

    // DeleteComposition in-use guard MUST fail while referenced by precomp_clip_id
    let err = session
        .execute(Command::DeleteComposition {
            comp_id: precomp_comp_id,
        })
        .unwrap_err();
    assert!(matches!(err, SessionError::InvalidOperation(_)));

    // SetActiveComposition -> edit inside nested comp without corrupting root sequence!
    let root_clip_count_before = session.to_project().sequence.video_tracks[0].clips.len();
    session
        .execute(Command::SetActiveComposition {
            comp_id: Some(precomp_comp_id),
        })
        .unwrap();
    assert_eq!(
        session.document().active_composition_id,
        Some(precomp_comp_id)
    );
    let nested_vtrack = session.document().video_tracks[0].id.clone();
    session
        .execute(Command::AddClip {
            track_id: nested_vtrack,
            kind: "shape2d".to_owned(),
            name: Some("Nested Shape".to_owned()),
            start: 50.0,
            length: 30.0,
        })
        .unwrap();
    let proj_mid = session.to_project();
    assert_eq!(
        proj_mid.sequence.video_tracks[0].clips.len(),
        root_clip_count_before,
        "editing active nested composition must not corrupt root sequence"
    );
    assert_eq!(
        proj_mid.compositions[&precomp_comp_id].video_tracks[0]
            .clips
            .len(),
        2,
        "nested composition edit must sync into project.compositions[&id]"
    );

    // Switch back to root composition
    session
        .execute(Command::SetActiveComposition { comp_id: None })
        .unwrap();
    assert_eq!(session.document().active_composition_id, None);

    // SetPrecompClipTarget -> retarget to comp_a_id, freeing precomp_comp_id for deletion
    session
        .execute(Command::SetPrecompClipTarget {
            clip_id: precomp_clip_uuid,
            composition_id: comp_a_id,
        })
        .unwrap();
    assert!(matches!(
        find_clip(&session.to_project(), &precomp_clip_str).layer.kind,
        LayerKind::PrecompRef { composition_id } if composition_id == comp_a_id
    ));

    // Now DeleteComposition(precomp_comp_id) succeeds and is undoable!
    session
        .execute(Command::DeleteComposition {
            comp_id: precomp_comp_id,
        })
        .unwrap();
    assert!(
        !session
            .to_project()
            .compositions
            .contains_key(&precomp_comp_id)
    );
    session.execute(Command::Undo).unwrap();
    assert!(
        session
            .to_project()
            .compositions
            .contains_key(&precomp_comp_id)
    );
}

#[test]
fn markers_controls_text_split_paste_time_and_assets_are_undoable() {
    let mut session = Session::new();
    let vtrack = session.document().video_tracks[0].id.clone();
    let scene_id_str = session.document().video_tracks[0].clips[0].id.clone();
    let scene_uuid = Uuid::parse_str(&scene_id_str).unwrap();

    // 1. Markers CRUD + color + Undo/Redo
    session
        .execute(Command::AddMarker {
            frame: 30.0,
            name: "Beat 1".to_owned(),
            color: Some("#ff3366".to_owned()),
        })
        .unwrap();
    session
        .execute(Command::UpdateMarker {
            index: 0,
            frame: 36.0,
            name: "Drop".to_owned(),
            color: Some("#00ffcc".to_owned()),
        })
        .unwrap();
    let proj_m = session.to_project();
    assert_eq!(proj_m.sequence.markers.len(), 1);
    assert_eq!(proj_m.sequence.markers[0].frame, 36.0);
    assert_eq!(proj_m.sequence.markers[0].label, "Drop");
    assert_eq!(
        proj_m
            .sequence
            .preserved
            .get("markerColors")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str()),
        Some("#00ffcc")
    );
    session.execute(Command::DeleteMarker { index: 0 }).unwrap();
    assert!(session.to_project().sequence.markers.is_empty());
    session.execute(Command::Undo).unwrap();
    assert_eq!(session.to_project().sequence.markers.len(), 1);

    // 2. Sequence Controls CRUD + Reorder across ALL 6 SequenceControlKind variants
    let all_kinds = [
        ("s", SequenceControlKind::Slider, PropertyValue::Number(0.5)),
        ("a", SequenceControlKind::Angle, PropertyValue::Number(45.0)),
        (
            "b",
            SequenceControlKind::Checkbox,
            PropertyValue::Number(1.0),
        ),
        (
            "c",
            SequenceControlKind::Color,
            PropertyValue::Color([1.0, 0.5, 0.0]),
        ),
        (
            "p2",
            SequenceControlKind::Point2D,
            PropertyValue::Vec2([10.0, 20.0]),
        ),
        (
            "p3",
            SequenceControlKind::Point3D,
            PropertyValue::Vec3([1.0, 2.0, 3.0]),
        ),
    ];
    let mut ctl_ids = Vec::new();
    for (name, kind, val) in all_kinds {
        let res = session
            .execute(Command::AddSequenceControl {
                name: name.to_owned(),
                kind,
                value: val,
                min: Some(0.0),
                max: Some(100.0),
                step: Some(0.5),
            })
            .unwrap();
        ctl_ids.push(res["control_id"].as_str().unwrap().to_owned());
    }
    session
        .execute(Command::UpdateSequenceControl {
            control_id: ctl_ids[0].clone(),
            name: "masterSlider".to_owned(),
            value: PropertyValue::Number(42.0),
            min: Some(-10.0),
            max: Some(200.0),
            step: Some(1.0),
        })
        .unwrap();
    session
        .execute(Command::ReorderSequenceControl {
            control_id: ctl_ids[0].clone(),
            to_index: 5,
        })
        .unwrap();
    let proj_c = session.to_project();
    assert_eq!(proj_c.sequence.controls.len(), 6);
    assert_eq!(proj_c.sequence.controls[5].id, ctl_ids[0]);
    assert_eq!(proj_c.sequence.controls[5].name, "masterSlider");
    assert_eq!(
        proj_c.sequence.controls[5].preserved.get("max"),
        Some(&json!(200.0))
    );
    session
        .execute(Command::DeleteSequenceControl {
            control_id: ctl_ids[1].clone(),
        })
        .unwrap();
    assert_eq!(session.to_project().sequence.controls.len(), 5);
    session.execute(Command::Undo).unwrap();
    assert_eq!(session.to_project().sequence.controls.len(), 6);

    // 3. SplitText3DIntoLetters
    let txt_id_str = session
        .execute(Command::AddObject3D {
            clip_id: scene_id_str,
            parent_id: None,
            kind: "text3d".to_owned(),
            name: Some("Word".to_owned()),
        })
        .unwrap()["object_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let txt_uuid = Uuid::parse_str(&txt_id_str).unwrap();
    session
        .execute(Command::SetPropertyValue {
            target_id: txt_id_str,
            property: "text".to_owned(),
            value: json!("A B"),
        })
        .unwrap();
    let split_txt_res = session
        .execute(Command::SplitText3DIntoLetters {
            clip_id: scene_uuid,
            object_id: txt_uuid,
        })
        .unwrap();
    let group_id = split_txt_res["group_id"].as_str().unwrap().to_owned();
    let letter_ids = split_txt_res["letter_ids"].as_array().unwrap();
    assert_eq!(
        letter_ids.len(),
        2,
        "whitespace skipped; 'A' and 'B' created"
    );
    let proj_t = session.to_project();
    let grp = find_obj(&proj_t, &group_id);
    let Object3DKind::Group { ref children, .. } = grp.kind else {
        panic!("expected Group from SplitText3DIntoLetters");
    };
    assert_eq!(children.len(), 2);
    assert!(matches!(&children[0].kind, Object3DKind::Text3D(td) if td.text == "A"));
    assert!(matches!(&children[1].kind, Object3DKind::Text3D(td) if td.text == "B"));
    let x0 = match children[0].position.value {
        PropertyValue::Vec3(v) => v[0],
        _ => panic!("expected Vec3"),
    };
    let x1 = match children[1].position.value {
        PropertyValue::Vec3(v) => v[0],
        _ => panic!("expected Vec3"),
    };
    assert!(x1 > x0, "letters must be offset by advance width");
    session.execute(Command::Undo).unwrap();

    // 4. copy_clips + PasteClips (preserving masks)
    let src_clip_str = session
        .execute(Command::AddClip {
            track_id: vtrack,
            kind: "video".to_owned(),
            name: Some("ClipToCopy".to_owned()),
            start: 100.0,
            length: 50.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let src_clip_uuid = Uuid::parse_str(&src_clip_str).unwrap();
    session
        .execute(Command::AddLayerMask {
            clip_id: src_clip_uuid,
            mask: LayerMask {
                id: "copy_m".to_owned(),
                name: "CopyMask".to_owned(),
                enabled: true,
                mode: MaskMode::Add,
                invert: false,
                feather: 0.0,
                expansion: 0.0,
                opacity: 1.0,
                closed: true,
                locked: false,
                outline_color: String::new(),
                vertex_ids: Vec::new(),
                vertex_modes: BTreeMap::new(),
                path: PropertyTrack::constant(PropertyValue::Json(json!({"commands": []}))),
                preserved: BTreeMap::new(),
            },
        })
        .unwrap();
    let snapshots = session.copy_clips(&[src_clip_uuid]).unwrap();
    assert_eq!(snapshots.len(), 1);
    let paste_res = session
        .execute(Command::PasteClips {
            clips: snapshots,
            target_frame: 180.0,
        })
        .unwrap();
    let pasted_id = paste_res["clip_ids"][0].as_str().unwrap().to_owned();
    let proj_pasted = session.to_project();
    let pasted_vc = find_clip(&proj_pasted, &pasted_id);
    assert_eq!(pasted_vc.start, 180.0);
    assert_eq!(
        proj_pasted
            .sequence
            .layer_masks
            .get(&pasted_vc.layer.id)
            .map(Vec::len),
        Some(1)
    );

    // 5. FreezeClipFrame + ReverseClip
    session
        .execute(Command::FreezeClipFrame {
            clip_id: src_clip_uuid,
            frame: 115.0,
        })
        .unwrap();
    let proj_frozen = session.to_project();
    let frozen_vc = find_clip(&proj_frozen, &src_clip_str);
    assert!(!frozen_vc.time_curve.animated);
    assert_eq!(frozen_vc.time_curve.value, PropertyValue::Number(0.5));
    session.execute(Command::Undo).unwrap();

    session
        .execute(Command::ReverseClip {
            clip_id: src_clip_uuid,
        })
        .unwrap();
    let proj_rev = session.to_project();
    let rev_vc = find_clip(&proj_rev, &src_clip_str);
    assert_eq!(rev_vc.time_curve.keyframes.len(), 2);
    let PropertyValue::Number(v0) = rev_vc.time_curve.keyframes[0].value else {
        panic!("expected number");
    };
    let PropertyValue::Number(v1) = rev_vc.time_curve.keyframes[1].value else {
        panic!("expected number");
    };
    assert!(v0 > v1, "reversed clip time must descend from end to start");
    session.execute(Command::Undo).unwrap();

    // 6. RelinkAsset + PurgeUnusedAssets
    let mut proj_assets = session.to_project();
    proj_assets.assets.insert(
        "asset_used".to_owned(),
        AssetRecord {
            id: "asset_used".to_owned(),
            kind: AssetKind::Image,
            source: AssetSource::File,
            filename: "used.png".to_owned(),
            url: None,
            external: false,
            bytes: Some(vec![1, 2, 3, 4]),
            missing: false,
            sha256: "asset_used".to_owned(),
            size: 4,
            preserved: BTreeMap::new(),
        },
    );
    proj_assets.assets.insert(
        "asset_orphan".to_owned(),
        AssetRecord {
            id: "asset_orphan".to_owned(),
            kind: AssetKind::Image,
            source: AssetSource::File,
            filename: "orphan.png".to_owned(),
            url: None,
            external: false,
            bytes: Some(vec![5, 6, 7, 8]),
            missing: false,
            sha256: "asset_orphan".to_owned(),
            size: 4,
            preserved: BTreeMap::new(),
        },
    );
    proj_assets.sequence.video_tracks[0].clips.push(VideoClip {
        id: Uuid::new_v4(),
        name: "Img".to_owned(),
        start: 300.0,
        length: 30.0,
        link: None,
        media: None,
        time_curve: PropertyTrack::number(0.0),
        layer: zc_project::Layer::new(
            "Img",
            LayerKind::Image {
                image: Some("asset_used".to_owned()),
            },
            [1920, 1080],
        ),
        preserved: BTreeMap::new(),
    });
    let mut asset_session = Session::from_project(proj_assets, PluginRegistry::new());
    asset_session
        .execute(Command::RelinkAsset {
            asset_id: "asset_used".to_owned(),
            filename: "used_v2.png".to_owned(),
            bytes: vec![9, 10, 11, 12, 13],
        })
        .unwrap();
    let relinked = asset_session.to_project();
    assert_eq!(relinked.assets["asset_used"].filename, "used_v2.png");
    assert_eq!(relinked.assets["asset_used"].size, 5);
    assert!(!relinked.assets["asset_used"].missing);

    let purge_res = asset_session.execute(Command::PurgeUnusedAssets).unwrap();
    assert_eq!(purge_res["purged_count"], json!(1));
    let purged = asset_session.to_project();
    assert!(purged.assets.contains_key("asset_used"));
    assert!(!purged.assets.contains_key("asset_orphan"));

    // Undo restores the purged orphan asset
    asset_session.execute(Command::Undo).unwrap();
    assert!(
        asset_session
            .to_project()
            .assets
            .contains_key("asset_orphan")
    );
}

#[test]
fn precompose_non_adjacent_clips_preserves_unselected_middle_clip_and_undo() {
    let mut session = Session::new();
    let vtrack = session.document().video_tracks[0].id.clone();
    // Remove default scene clip so track 0 contains only our three test clips.
    let default_clip_id = session.document().video_tracks[0].clips[0].id.clone();
    session
        .execute(Command::DeleteClip {
            clip_id: default_clip_id,
        })
        .unwrap();

    let clip0_str = session
        .execute(Command::AddClip {
            track_id: vtrack.clone(),
            kind: "text2d".to_owned(),
            name: Some("Clip 0".to_owned()),
            start: 0.0,
            length: 30.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip1_str = session
        .execute(Command::AddClip {
            track_id: vtrack.clone(),
            kind: "shape2d".to_owned(),
            name: Some("Middle Unselected".to_owned()),
            start: 30.0,
            length: 30.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip2_str = session
        .execute(Command::AddClip {
            track_id: vtrack,
            kind: "text2d".to_owned(),
            name: Some("Clip 2".to_owned()),
            start: 60.0,
            length: 30.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();

    let before_proj = session.to_project();
    assert_eq!(before_proj.sequence.video_tracks.len(), 1);
    assert_eq!(before_proj.sequence.video_tracks[0].clips.len(), 3);

    // Precompose clip 0 + clip 2 while leaving middle clip 1 unselected.
    let clip0_uuid = Uuid::parse_str(&clip0_str).unwrap();
    let clip2_uuid = Uuid::parse_str(&clip2_str).unwrap();
    let res = session
        .execute(Command::PrecomposeClips {
            clip_ids: vec![clip0_uuid, clip2_uuid],
            name: "Bookends Precomp".to_owned(),
        })
        .unwrap();
    let comp_id = Uuid::parse_str(res["comp_id"].as_str().unwrap()).unwrap();
    let precomp_clip_str = res["clip_id"].as_str().unwrap().to_owned();

    let after_proj = session.to_project();
    // Replacement precomp track is inserted directly at target_vtrack_idx (0) at [0..90],
    // while the middle unselected clip 1 remains completely intact at [30..60] on track 1!
    assert_eq!(after_proj.sequence.video_tracks.len(), 2);
    assert_eq!(after_proj.sequence.video_tracks[0].clips.len(), 1);
    let precomp_clip = &after_proj.sequence.video_tracks[0].clips[0];
    assert_eq!(precomp_clip.id.to_string(), precomp_clip_str);
    assert!((precomp_clip.start - 0.0).abs() < 1e-6);
    assert!((precomp_clip.length - 90.0).abs() < 1e-6);

    assert_eq!(after_proj.sequence.video_tracks[1].clips.len(), 1);
    let kept_middle = &after_proj.sequence.video_tracks[1].clips[0];
    assert_eq!(kept_middle.id.to_string(), clip1_str);
    assert!((kept_middle.start - 30.0).abs() < 1e-6);
    assert!((kept_middle.length - 30.0).abs() < 1e-6);

    // Child composition contains only clip 0 and clip 2.
    let child_comp = &after_proj.compositions[&comp_id];
    assert_eq!(child_comp.video_tracks[0].clips.len(), 2);
    assert_eq!(child_comp.video_tracks[0].clips[0].id, clip0_uuid);
    assert_eq!(child_comp.video_tracks[0].clips[1].id, clip2_uuid);

    // Undo MUST restore exact original tracks, clips, and compositions.
    session.execute(Command::Undo).unwrap();
    let undone_proj = session.to_project();
    assert!(!undone_proj.compositions.contains_key(&comp_id));
    assert_eq!(undone_proj.sequence.video_tracks.len(), 1);
    let restored_ids: Vec<String> = undone_proj.sequence.video_tracks[0]
        .clips
        .iter()
        .map(|c| c.id.to_string())
        .collect();
    assert_eq!(restored_ids, vec![clip0_str, clip1_str, clip2_str]);
}

#[test]
fn precompose_clones_sequence_controls_and_shifts_keyframes_to_child_origin() {
    let mut project = Project::new_cm3_default();
    project.sequence.controls.push(zc_project::SequenceControl {
        id: "ctl-drive-01".to_owned(),
        name: "Drive".to_owned(),
        kind: SequenceControlKind::Slider,
        track: PropertyTrack {
            animated: true,
            expression: None,
            static_behavior: StaticBehavior::Constant,
            value: PropertyValue::Number(10.0),
            keyframes: vec![
                PropertyKeyframe::linear(30.0, PropertyValue::Number(10.0)),
                PropertyKeyframe::linear(60.0, PropertyValue::Number(80.0)),
            ],
            channels: Vec::new(),
        },
        preserved: BTreeMap::new(),
    });

    let mut session = Session::from_project(project, PluginRegistry::new());
    let vtrack = session.document().video_tracks[0].id.clone();
    let default_clip_id = session.document().video_tracks[0].clips[0].id.clone();
    session
        .execute(Command::DeleteClip {
            clip_id: default_clip_id,
        })
        .unwrap();

    let clip_str = session
        .execute(Command::AddClip {
            track_id: vtrack,
            kind: "text2d".to_owned(),
            name: Some("Driven Clip".to_owned()),
            start: 30.0,
            length: 40.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip_uuid = Uuid::parse_str(&clip_str).unwrap();

    let res = session
        .execute(Command::PrecomposeClips {
            clip_ids: vec![clip_uuid],
            name: "Driven Precomp".to_owned(),
        })
        .unwrap();
    let comp_id = Uuid::parse_str(res["comp_id"].as_str().unwrap()).unwrap();

    let proj = session.to_project();
    let child = &proj.compositions[&comp_id];
    assert_eq!(child.controls.len(), 1);
    let ctl = &child.controls[0];
    assert_eq!(ctl.id, "ctl-drive-01");
    assert_eq!(ctl.name, "Drive");
    assert_eq!(ctl.track.keyframes.len(), 2);
    assert!((ctl.track.keyframes[0].frame - 0.0).abs() < 1e-6);
    assert!((ctl.track.keyframes[1].frame - 30.0).abs() < 1e-6);
    assert_eq!(ctl.track.keyframes[0].value, PropertyValue::Number(10.0));
    assert_eq!(ctl.track.keyframes[1].value, PropertyValue::Number(80.0));

    // Parent sequence control keyframes remain untouched at 30 and 60.
    assert!((proj.sequence.controls[0].track.keyframes[0].frame - 30.0).abs() < 1e-6);
    assert!((proj.sequence.controls[0].track.keyframes[1].frame - 60.0).abs() < 1e-6);
}

#[test]
fn reverse_clip_migrates_segment_tweens_mirrors_easings_and_mirrors_bezier_handles() {
    let mut session = Session::new();
    let vtrack = session.document().video_tracks[0].id.clone();

    // 1. Mixed-tween curve: [0: Linear, 10: Hold (seg 0->10 is Hold), 20: QuadIn (seg 10->20 is QuadIn)]
    let clip_str = session
        .execute(Command::AddClip {
            track_id: vtrack.clone(),
            kind: "video".to_owned(),
            name: Some("Mixed Tween Clip".to_owned()),
            start: 200.0,
            length: 20.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip_uuid = Uuid::parse_str(&clip_str).unwrap();

    session
        .execute(Command::SetKeyframe {
            target_id: clip_str.clone(),
            property: "time".to_owned(),
            frame: 0.0,
            value: json!(0.0),
            tween: Some(1), // Linear (unused leftmost)
        })
        .unwrap();
    session
        .execute(Command::SetKeyframe {
            target_id: clip_str.clone(),
            property: "time".to_owned(),
            frame: 10.0,
            value: json!(1.0),
            tween: Some(0), // Hold for segment 0->10
        })
        .unwrap();
    session
        .execute(Command::SetKeyframe {
            target_id: clip_str.clone(),
            property: "time".to_owned(),
            frame: 20.0,
            value: json!(2.0),
            tween: Some(2), // QuadIn for segment 10->20
        })
        .unwrap();

    session
        .execute(Command::ReverseClip { clip_id: clip_uuid })
        .unwrap();

    let proj = session.to_project();
    let kfs = &find_clip(&proj, &clip_str).time_curve.keyframes;
    assert_eq!(kfs.len(), 4);
    assert!((kfs[0].frame - 0.0).abs() < 1e-6);
    assert_eq!(kfs[0].value, PropertyValue::Number(2.0));
    assert_eq!(
        kfs[0].tween, 1,
        "leftmost keyframe tween defaults to Linear"
    );

    // Reversed segment 0->10 reverses original 10->20 (QuadIn = 2) -> QuadOut = 3!
    assert!((kfs[1].frame - 10.0).abs() < 1e-6);
    assert_eq!(kfs[1].value, PropertyValue::Number(1.0));
    assert_eq!(
        kfs[1].tween, 3,
        "reversed segment 0->10 must carry mirrored QuadOut (3)"
    );

    // Reversed Hold segment 10->20 inserts a transition step key right after 10.0 so the
    // interior (10..20) holds 0.0 (matching forward [0..10)), while 10.0 still equals 1.0!
    assert!((kfs[2].frame - 10.0001).abs() < 1e-6);
    assert_eq!(kfs[2].value, PropertyValue::Number(0.0));
    assert_eq!(kfs[2].tween, 0);

    assert!((kfs[3].frame - 20.0).abs() < 1e-6);
    assert_eq!(kfs[3].value, PropertyValue::Number(0.0));
    assert_eq!(
        kfs[3].tween, 0,
        "reversed segment 10->20 must carry Hold (0)"
    );

    // 2. Bezier segment curve reversal must mirror curve values identically: f_rev(L - t) == f(t).
    let bez_str = session
        .execute(Command::AddClip {
            track_id: vtrack,
            kind: "video".to_owned(),
            name: Some("Bezier Time Clip".to_owned()),
            start: 240.0,
            length: 20.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let bez_uuid = Uuid::parse_str(&bez_str).unwrap();
    let mut bez_proj = session.to_project();
    {
        let vc = bez_proj.sequence.video_tracks[0]
            .clips
            .iter_mut()
            .find(|c| c.id == bez_uuid)
            .unwrap();
        vc.time_curve.animated = true;
        vc.time_curve.keyframes = vec![
            PropertyKeyframe {
                frame: 0.0,
                value: PropertyValue::Number(0.0),
                tween: 1,
                easing_index: 1,
                is_bezier: false,
                continuous_tangent: false,
                control_points: [[-2.0, 0.0], [4.0, 0.8]],
            },
            PropertyKeyframe {
                frame: 20.0,
                value: PropertyValue::Number(10.0),
                tween: (1 << 8) | 2, // Bezier + QuadIn low byte
                easing_index: 2,
                is_bezier: true,
                continuous_tangent: false,
                control_points: [[-6.0, -0.5], [2.0, 0.0]],
            },
        ];
    }
    let orig_sample = zc_anim::eval_segment_bezier(
        zc_anim::BezierSegment {
            start_frame: 0.0,
            start_value: 0.0,
            out_tangent: [4.0, 0.8],
            end_frame: 20.0,
            end_value: 10.0,
            in_tangent: [-6.0, -0.5],
        },
        6.0,
    );

    let mut bez_session = Session::from_project(bez_proj, PluginRegistry::new());
    bez_session
        .execute(Command::ReverseClip { clip_id: bez_uuid })
        .unwrap();
    let rev_proj = bez_session.to_project();
    let rkfs = &find_clip(&rev_proj, &bez_str).time_curve.keyframes;
    assert_eq!(rkfs.len(), 2);
    assert_eq!(
        rkfs[1].tween,
        (1 << 8) | 3,
        "Bezier flag preserved with mirrored low byte"
    );
    assert_eq!(rkfs[0].control_points[1], [6.0, -0.5]);
    assert_eq!(rkfs[1].control_points[0], [-4.0, 0.8]);

    let rev_sample = zc_anim::eval_segment_bezier(
        zc_anim::BezierSegment {
            start_frame: rkfs[0].frame,
            start_value: 10.0,
            out_tangent: rkfs[0].control_points[1],
            end_frame: rkfs[1].frame,
            end_value: 0.0,
            in_tangent: rkfs[1].control_points[0],
        },
        14.0,
    );
    assert!(
        (rev_sample - orig_sample).abs() < 1e-9,
        "reversed Bezier curve at L - t must match original at t (orig={orig_sample}, rev={rev_sample})"
    );
}

#[test]
fn freeze_clip_frame_evaluates_time_expression_and_sequence_controls() {
    let mut session = Session::new();
    let vtrack = session.document().video_tracks[0].id.clone();

    session
        .execute(Command::AddSequenceControl {
            name: "TimeBoost".to_owned(),
            kind: SequenceControlKind::Slider,
            value: PropertyValue::Number(0.5),
            min: None,
            max: None,
            step: None,
        })
        .unwrap();

    // Clip at start=10, length=60 at 30fps -> linear time [0.0..2.0].
    // At sequence frame 40 -> local_frame = 30 -> base time = 1.0.
    let clip_str = session
        .execute(Command::AddClip {
            track_id: vtrack.clone(),
            kind: "video".to_owned(),
            name: Some("Expr Time Clip".to_owned()),
            start: 10.0,
            length: 60.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip_uuid = Uuid::parse_str(&clip_str).unwrap();

    session
        .execute(Command::SetExpression {
            target_id: clip_str.clone(),
            property: "time".to_owned(),
            expression: Some("time * 2.0 + 1.0 + control('TimeBoost')".to_owned()),
        })
        .unwrap();

    let freeze_res = session
        .execute(Command::FreezeClipFrame {
            clip_id: clip_uuid,
            frame: 40.0,
        })
        .unwrap();
    let frozen_val = freeze_res["frozen_time"].as_f64().unwrap();
    assert!(
        (frozen_val - 3.5).abs() < 1e-9,
        "FreezeClipFrame must freeze expression-evaluated time (3.5), got {frozen_val}"
    );
    let proj = session.to_project();
    let frozen_vc = find_clip(&proj, &clip_str);
    assert!(!frozen_vc.time_curve.animated);
    assert_eq!(frozen_vc.time_curve.expression, None);
    assert_eq!(frozen_vc.time_curve.value, PropertyValue::Number(3.5));

    // Also verify `value + 3.0` expression referencing the base keyframe curve value.
    let clip2_str = session
        .execute(Command::AddClip {
            track_id: vtrack,
            kind: "video".to_owned(),
            name: Some("Value Expr Clip".to_owned()),
            start: 100.0,
            length: 60.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip2_uuid = Uuid::parse_str(&clip2_str).unwrap();
    session
        .execute(Command::SetExpression {
            target_id: clip2_str.clone(),
            property: "time".to_owned(),
            expression: Some("value + 3.0".to_owned()),
        })
        .unwrap();
    let freeze2 = session
        .execute(Command::FreezeClipFrame {
            clip_id: clip2_uuid,
            frame: 130.0,
        })
        .unwrap();
    assert!((freeze2["frozen_time"].as_f64().unwrap() - 4.0).abs() < 1e-9);
    let proj2 = session.to_project();
    assert_eq!(find_clip(&proj2, &clip2_str).time_curve.expression, None);
}

#[test]
fn precompose_fallback_preserves_visibility_solo_stacking_order_and_undo() {
    let mut project = Project::new_cm3_default();
    let v0_id = Uuid::new_v4();
    let v1_id = Uuid::new_v4();
    let v2_id = Uuid::new_v4();
    let v3_id = Uuid::new_v4();

    let clip_a_id = Uuid::new_v4();
    let clip_mid_id = Uuid::new_v4();
    let clip_b_id = Uuid::new_v4();
    let clip_fg_id = Uuid::new_v4();
    let clip_other_solo_id = Uuid::new_v4();

    let make_clip = |id: Uuid, name: &str, start: f64, length: f64| VideoClip {
        id,
        name: name.to_owned(),
        start,
        length,
        link: None,
        media: None,
        time_curve: PropertyTrack::linear_time_curve(length, 30),
        layer: zc_project::Layer::new(
            name,
            LayerKind::Shape2D { shapes: Vec::new() },
            [1920, 1080],
        ),
        preserved: BTreeMap::new(),
    };

    project.sequence.video_tracks = vec![
        // Track 0: empty & DISABLED (must NEVER be picked as fallback!).
        zc_project::VideoTrack {
            id: v0_id,
            name: "V0 Disabled".to_owned(),
            enabled: false,
            solo: false,
            locked: false,
            is_3d: false,
            clips: Vec::new(),
            preserved: BTreeMap::new(),
        },
        // Track 1: enabled & SOLOED, holds non-adjacent selected clips A and B around unselected M.
        zc_project::VideoTrack {
            id: v1_id,
            name: "V1 Solo Target".to_owned(),
            enabled: true,
            solo: true,
            locked: false,
            is_3d: false,
            clips: vec![
                make_clip(clip_a_id, "Selected A", 0.0, 10.0),
                make_clip(clip_mid_id, "Unselected Mid", 10.0, 10.0),
                make_clip(clip_b_id, "Selected B", 20.0, 10.0),
            ],
            preserved: BTreeMap::new(),
        },
        // Track 2: unselected foreground track ABOVE V1.
        zc_project::VideoTrack {
            id: v2_id,
            name: "V2 Foreground".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![make_clip(clip_fg_id, "Foreground", 0.0, 30.0)],
            preserved: BTreeMap::new(),
        },
        // Track 3: another soloed track in parent sequence.
        zc_project::VideoTrack {
            id: v3_id,
            name: "V3 Other Solo".to_owned(),
            enabled: true,
            solo: true,
            locked: false,
            is_3d: false,
            clips: vec![make_clip(clip_other_solo_id, "Other Solo", 40.0, 10.0)],
            preserved: BTreeMap::new(),
        },
    ];

    let mut session = Session::from_project(project, PluginRegistry::new());
    let res = session
        .execute(Command::PrecomposeClips {
            clip_ids: vec![clip_a_id, clip_b_id],
            name: "NonAdjacent Precomp".to_owned(),
        })
        .expect("precompose succeeds");
    let precomp_clip_id = Uuid::parse_str(res["clip_id"].as_str().unwrap()).unwrap();

    let after = session.to_project();
    assert_eq!(
        after.sequence.video_tracks.len(),
        5,
        "must insert a fresh track at target_vtrack_idx instead of reusing disabled V0 or overwriting Mid"
    );

    // Track 0 stays disabled and empty.
    assert_eq!(after.sequence.video_tracks[0].id, v0_id);
    assert!(!after.sequence.video_tracks[0].enabled);
    assert!(after.sequence.video_tracks[0].clips.is_empty());

    // Track 1 is the newly inserted precomp track: enabled, unlocked, soloed, directly at target_vtrack_idx!
    let precomp_track = &after.sequence.video_tracks[1];
    assert!(precomp_track.enabled, "precomp track must be enabled");
    assert!(!precomp_track.locked, "precomp track must be unlocked");
    assert!(
        precomp_track.solo,
        "precomp track must inherit solo visibility when parent has soloed tracks and selected track was soloed"
    );
    assert_eq!(precomp_track.clips.len(), 1);
    assert_eq!(precomp_track.clips[0].id, precomp_clip_id);

    // Original V1 (now index 2) keeps Unselected Mid intact.
    assert_eq!(after.sequence.video_tracks[2].id, v1_id);
    assert_eq!(after.sequence.video_tracks[2].clips.len(), 1);
    assert_eq!(after.sequence.video_tracks[2].clips[0].id, clip_mid_id);

    // Foreground track (now index 3) remains ABOVE the precomp track (preserving z-order!).
    assert_eq!(after.sequence.video_tracks[3].id, v2_id);
    assert_eq!(after.sequence.video_tracks[3].clips[0].id, clip_fg_id);

    // Undo restores the exact original 4 tracks, flags, and clips.
    session.execute(Command::Undo).expect("undo precompose");
    let undone = session.to_project();
    assert_eq!(undone.sequence.video_tracks.len(), 4);
    assert_eq!(undone.sequence.video_tracks[0].id, v0_id);
    assert!(!undone.sequence.video_tracks[0].enabled);
    assert_eq!(undone.sequence.video_tracks[1].id, v1_id);
    assert!(undone.sequence.video_tracks[1].solo);
    let v1_clip_ids: Vec<Uuid> = undone.sequence.video_tracks[1]
        .clips
        .iter()
        .map(|c| c.id)
        .collect();
    assert_eq!(v1_clip_ids, vec![clip_a_id, clip_mid_id, clip_b_id]);
    assert_eq!(undone.sequence.video_tracks[2].id, v2_id);
    assert_eq!(undone.sequence.video_tracks[3].id, v3_id);
}

#[test]
fn freeze_clip_frame_lazy_controls_strict_id_lookup_and_layer_handles() {
    let mut project = Project::new_cm3_default();
    project.sequence.resolution = [1920, 1080];
    project.sequence.fps = 30;

    // Control A (index 0) references Control B (index 1) — requires lazy resolution!
    let track_a = PropertyTrack::number_with_expr(0.0, "control('Base') * 2.0 + 1.0");
    let track_b = PropertyTrack::number_with_expr(0.0, "frame * 0.1 + 0.5");

    project.sequence.controls = vec![
        zc_project::SequenceControl {
            id: "ctl-a".to_owned(),
            name: "Derived".to_owned(),
            kind: SequenceControlKind::Slider,
            track: track_a,
            preserved: BTreeMap::new(),
        },
        zc_project::SequenceControl {
            id: "ctl-b".to_owned(),
            name: "Base".to_owned(),
            kind: SequenceControlKind::Slider,
            track: track_b,
            preserved: BTreeMap::new(),
        },
    ];

    let mut plugins = PluginRegistry::new();
    let _ = plugins.enable("expressions-plus");
    let mut session = Session::from_project(project, plugins);
    let vtrack = session.document().video_tracks[0].id.clone();

    // Clip 1: references lazily-resolved Control A + thisLayer/thisClip/thisComp properties.
    let clip1_str = session
        .execute(Command::AddClip {
            track_id: vtrack.clone(),
            kind: "video".to_owned(),
            name: Some("Lazy Handle Clip".to_owned()),
            start: 10.0,
            length: 60.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip1_uuid = Uuid::parse_str(&clip1_str).unwrap();
    session
        .execute(Command::SetPropertyValue {
            target_id: clip1_str.clone(),
            property: "opacity".to_owned(),
            value: json!(0.5),
        })
        .unwrap();
    session
        .execute(Command::SetExpression {
            target_id: clip1_str.clone(),
            property: "time".to_owned(),
            expression: Some(
                "control('Derived') + thisLayer.property('opacity').value + thisClip.property('start').value * 0.01 + thisComp.property('Resolution').value[0] * 0.001"
                    .to_owned(),
            ),
        })
        .unwrap();

    // At abs frame 40:
    // Control B ('Base') = 40 * 0.1 + 0.5 = 4.5
    // Control A ('Derived') = 4.5 * 2.0 + 1.0 = 10.0
    // opacity = 0.5, clip start * 0.01 = 0.1, seq width * 0.001 = 1.92
    // Expected frozen time = 12.52
    let res1 = session
        .execute(Command::FreezeClipFrame {
            clip_id: clip1_uuid,
            frame: 40.0,
        })
        .unwrap();
    let frozen1 = res1["frozen_time"].as_f64().unwrap();
    assert!(
        (frozen1 - 12.52).abs() < 1e-9,
        "FreezeClipFrame must lazily resolve forward-referenced controls + layer/clip/comp handles (expected 12.52), got {frozen1}"
    );

    // Clip 2: strict ID lookup — nonexistent stable_id MUST NOT fall back to matching by name!
    let clip2_str = session
        .execute(Command::AddClip {
            track_id: vtrack.clone(),
            kind: "video".to_owned(),
            name: Some("Strict Missing ID Clip".to_owned()),
            start: 100.0,
            length: 60.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip2_uuid = Uuid::parse_str(&clip2_str).unwrap();
    session
        .execute(Command::SetExpression {
            target_id: clip2_str.clone(),
            property: "time".to_owned(),
            expression: Some("control('Base', 'nonexistent-id') + 99.0".to_owned()),
        })
        .unwrap();
    // At frame 130 -> local_frame = 30 -> base_time = 1.0; missing ID errors and falls back to 1.0.
    let res2 = session
        .execute(Command::FreezeClipFrame {
            clip_id: clip2_uuid,
            frame: 130.0,
        })
        .unwrap();
    let frozen2 = res2["frozen_time"].as_f64().unwrap();
    assert!(
        (frozen2 - 1.0).abs() < 1e-9,
        "strict ID lookup must NOT fall back to control name when stable_id is missing (expected fallback 1.0), got {frozen2}"
    );

    // Clip 3: valid stable_id resolves even when display name does not match.
    let clip3_str = session
        .execute(Command::AddClip {
            track_id: vtrack,
            kind: "video".to_owned(),
            name: Some("Strict Valid ID Clip".to_owned()),
            start: 200.0,
            length: 60.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip3_uuid = Uuid::parse_str(&clip3_str).unwrap();
    session
        .execute(Command::SetExpression {
            target_id: clip3_str.clone(),
            property: "time".to_owned(),
            expression: Some("control('Renamed', 'ctl-b')".to_owned()),
        })
        .unwrap();
    // At abs frame 230: Control B = 230 * 0.1 + 0.5 = 23.5.
    let res3 = session
        .execute(Command::FreezeClipFrame {
            clip_id: clip3_uuid,
            frame: 230.0,
        })
        .unwrap();
    let frozen3 = res3["frozen_time"].as_f64().unwrap();
    assert!(
        (frozen3 - 23.5).abs() < 1e-9,
        "valid stable_id must resolve control even when name differs (expected 23.5), got {frozen3}"
    );
}

#[test]
fn precompose_preserves_exact_fps_and_shifts_cloned_control_expression_time_origin() {
    let mut project = Project::new_cm3_default();
    project.sequence.fps = 30;
    project.sequence.exact_fps = Some(29.97);

    let orig_expr = "frame * 2.0 + time * 10.0";
    let ctrl_track = PropertyTrack::number_with_expr(0.0, orig_expr);
    project.sequence.controls.push(zc_project::SequenceControl {
        id: "ctl-wave".to_owned(),
        name: "Wave".to_owned(),
        kind: SequenceControlKind::Slider,
        track: ctrl_track,
        preserved: BTreeMap::new(),
    });

    let clip_id = Uuid::new_v4();
    project.sequence.video_tracks[0].clips.push(VideoClip {
        id: clip_id,
        name: "Offset Clip".to_owned(),
        start: 60.0,
        length: 30.0,
        link: None,
        media: None,
        time_curve: PropertyTrack::linear_time_curve(30.0, 30),
        layer: zc_project::Layer::new(
            "Offset Clip",
            LayerKind::Shape2D { shapes: Vec::new() },
            [1920, 1080],
        ),
        preserved: BTreeMap::new(),
    });

    let mut session = Session::from_project(project, PluginRegistry::new());
    let res = session
        .execute(Command::PrecomposeClips {
            clip_ids: vec![clip_id],
            name: "NTSC Child".to_owned(),
        })
        .expect("precompose succeeds");
    let comp_id = Uuid::parse_str(res["comp_id"].as_str().unwrap()).unwrap();

    let after = session.to_project();
    let child_comp = after
        .compositions
        .get(&comp_id)
        .expect("child composition exists");
    assert_eq!(
        child_comp.exact_fps,
        Some(29.97),
        "child composition must preserve parent exact_fps"
    );

    let child_expr = child_comp.controls[0]
        .track
        .expression
        .as_deref()
        .expect("cloned control keeps expression");

    let cfg = zc_expr::EvalConfig::expressions_plus();
    for offset in [0.0_f64, 15.0_f64, 29.0_f64] {
        let parent_frame = 60.0 + offset;
        let parent_ctx = zc_expr::BasicContext {
            frame: parent_frame,
            time: Some(parent_frame / 29.97),
            fps: 29.97,
            ..zc_expr::BasicContext::default()
        };
        let child_ctx = zc_expr::BasicContext {
            frame: offset,
            time: Some(offset / 29.97),
            fps: 29.97,
            ..zc_expr::BasicContext::default()
        };
        let parent_out = zc_expr::eval_with_fallback(orig_expr, &parent_ctx, &cfg);
        let child_out = zc_expr::eval_with_fallback(child_expr, &child_ctx, &cfg);
        assert!(parent_out.error.is_none());
        assert!(
            child_out.error.is_none(),
            "rewritten child control expr `{child_expr}` must evaluate without error: {:?}",
            child_out.error
        );
        let pv = parent_out.value.as_f64().unwrap();
        let cv = child_out.value.as_f64().unwrap();
        assert!(
            (pv - cv).abs() < 1e-9,
            "cloned control at child frame {offset} ({cv}) must match parent frame {parent_frame} ({pv}); expr=`{child_expr}`"
        );
    }
}

#[test]
fn reverse_clip_preserves_hold_segment_interior() {
    let mut session = Session::new();
    let vtrack = session.document().video_tracks[0].id.clone();

    let clip_str = session
        .execute(Command::AddClip {
            track_id: vtrack,
            kind: "video".to_owned(),
            name: Some("Hold Reverse Clip".to_owned()),
            start: 0.0,
            length: 20.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let clip_uuid = Uuid::parse_str(&clip_str).unwrap();

    // Forward Hold curve: [frame 0: 0.0, frame 20: 10.0 (Hold = tween 0)].
    session
        .execute(Command::SetKeyframe {
            target_id: clip_str.clone(),
            property: "time".to_owned(),
            frame: 0.0,
            value: json!(0.0),
            tween: Some(1),
        })
        .unwrap();
    session
        .execute(Command::SetKeyframe {
            target_id: clip_str.clone(),
            property: "time".to_owned(),
            frame: 20.0,
            value: json!(10.0),
            tween: Some(0), // Hold
        })
        .unwrap();

    // Helper: sample the clip's evaluated time at `frame` via FreezeClipFrame on a cloned session.
    let sample_at = |sess: &mut Session, frame: f64| -> f64 {
        let mut clone = Session::from_project(sess.to_project(), sess.plugins.clone());
        clone
            .execute(Command::FreezeClipFrame {
                clip_id: clip_uuid,
                frame,
            })
            .unwrap()["frozen_time"]
            .as_f64()
            .unwrap()
    };

    // Forward sanity check: [0, 20) evaluates to 0.0, and frame 20 evaluates to 10.0.
    assert!((sample_at(&mut session, 0.0) - 0.0).abs() < 1e-9);
    assert!((sample_at(&mut session, 5.0) - 0.0).abs() < 1e-9);
    assert!((sample_at(&mut session, 10.0) - 0.0).abs() < 1e-9);
    assert!((sample_at(&mut session, 20.0) - 10.0).abs() < 1e-9);

    // Reverse the clip across [0, 20].
    session
        .execute(Command::ReverseClip { clip_id: clip_uuid })
        .unwrap();

    // Reversed Hold curve MUST evaluate to 10.0 at frame 0, 0.0 across the interior (5, 10, 15), and 0.0 at frame 20!
    assert!(
        (sample_at(&mut session, 0.0) - 10.0).abs() < 1e-9,
        "reversed Hold start frame 0 must equal 10.0"
    );
    for interior_frame in [5.0, 10.0, 15.0] {
        let v = sample_at(&mut session, interior_frame);
        assert!(
            (v - 0.0).abs() < 1e-9,
            "reversed Hold interior frame {interior_frame} must evaluate to 0.0 (not 10.0!), got {v}"
        );
    }
    assert!(
        (sample_at(&mut session, 20.0) - 0.0).abs() < 1e-9,
        "reversed Hold end frame 20 must equal 0.0"
    );
}

#[test]
fn codex_round3_precompose_solo_exact_fps_scoped_shadow_freeze_coercion_and_short_hold() {
    // 1. Precompose preserves solo state on extracted tracks & uses exact_fps on replacement time_curve
    //    AND scoped `function f(frame) { return frame; } f(1) + frame` still offsets outer `frame`.
    let mut project = Project::new_cm3_default();
    project.sequence.fps = 30;
    project.sequence.exact_fps = Some(29.97);
    project.sequence.controls.push(zc_project::SequenceControl {
        id: "ctl-scoped".to_owned(),
        name: "ScopedExpr".to_owned(),
        kind: SequenceControlKind::Slider,
        track: PropertyTrack::number_with_expr(
            0.0,
            "function f(frame) { return frame; } f(1) + frame",
        ),
        preserved: BTreeMap::new(),
    });
    let c_solo_id = Uuid::new_v4();
    let c_hidden_id = Uuid::new_v4();
    project.sequence.video_tracks = vec![
        zc_project::VideoTrack {
            id: Uuid::new_v4(),
            name: "V1".to_owned(),
            enabled: true,
            solo: true,
            locked: false,
            is_3d: false,
            clips: vec![VideoClip {
                id: c_solo_id,
                name: "SoloClip".to_owned(),
                start: 10.0,
                length: 30.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(30.0, 30),
                layer: zc_project::Layer::new(
                    "SoloClip",
                    LayerKind::Shape2D { shapes: Vec::new() },
                    [1920, 1080],
                ),
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        },
        zc_project::VideoTrack {
            id: Uuid::new_v4(),
            name: "V2".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![VideoClip {
                id: c_hidden_id,
                name: "HiddenClip".to_owned(),
                start: 10.0,
                length: 30.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(30.0, 30),
                layer: zc_project::Layer::new(
                    "HiddenClip",
                    LayerKind::Shape2D { shapes: Vec::new() },
                    [1920, 1080],
                ),
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        },
    ];

    let mut session = Session::from_project(project, PluginRegistry::new());
    let _ = session.plugins.enable("expressions-plus");

    let precomp_res = session
        .execute(Command::PrecomposeClips {
            clip_ids: vec![c_solo_id, c_hidden_id],
            name: "SoloComp".to_owned(),
        })
        .unwrap();
    let comp_id = Uuid::parse_str(precomp_res["comp_id"].as_str().unwrap()).unwrap();
    let precomp_clip_id = Uuid::parse_str(precomp_res["clip_id"].as_str().unwrap()).unwrap();

    let proj = session.to_project();
    let child = proj.compositions.get(&comp_id).expect("child comp exists");
    // Extracted child tracks must preserve their solo flags so hidden non-soloed clip stays hidden.
    assert!(child.video_tracks.iter().any(|t| t.solo));
    assert!(child.video_tracks.iter().any(|t| !t.solo));

    // Replacement clip time_curve end must use exact_fps (30.0 / 29.97), not 30.0 / 30.0.
    let rep_clip = proj
        .sequence
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .find(|c| c.id == precomp_clip_id)
        .expect("replacement clip exists");
    let end_sec = rep_clip
        .time_curve
        .keyframes
        .last()
        .and_then(|k| k.value.as_f64())
        .unwrap();
    assert!(
        (end_sec - (30.0 / 29.97)).abs() < 1e-6,
        "replacement time_curve must use exact_fps 29.97, got {end_sec}"
    );

    // Scoped shadow: `function f(frame) { return frame; } f(1) + frame` must offset ONLY outer `frame`.
    let rewritten = child.controls[0].track.expression.as_deref().unwrap();
    assert!(
        rewritten.contains("function f(frame) { return frame; }")
            && rewritten.contains("(frame + 10.0)"),
        "expected outer frame rewritten while inner parameter untouched, got `{rewritten}`"
    );

    // 2. Freeze coercion: numeric slider control whose expression returns string `"2"` must coerce to number `2`.
    let mut s2 = Session::new();
    let _ = s2.plugins.enable("expressions-plus");
    let vt = s2.document().video_tracks[0].id.clone();
    let cid_str = s2
        .execute(Command::AddClip {
            track_id: vt,
            kind: "video".to_owned(),
            name: Some("CoerceClip".to_owned()),
            start: 0.0,
            length: 20.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let cid = Uuid::parse_str(&cid_str).unwrap();
    let mut p2 = s2.to_project();
    p2.sequence.controls.push(zc_project::SequenceControl {
        id: "ctl-speed".to_owned(),
        name: "Speed".to_owned(),
        kind: SequenceControlKind::Slider,
        track: PropertyTrack::number_with_expr(1.0, "\"2\""),
        preserved: BTreeMap::new(),
    });
    p2.sequence.video_tracks[0].clips[0].time_curve.expression =
        Some("control(\"Speed\") === 2 ? 5 : 1".to_owned());
    let mut s2 = Session::from_project(p2, s2.plugins);
    let frozen = s2
        .execute(Command::FreezeClipFrame {
            clip_id: cid,
            frame: 5.0,
        })
        .unwrap()["frozen_time"]
        .as_f64()
        .unwrap();
    assert!(
        (frozen - 5.0).abs() < 1e-9,
        "FreezeClipFrame must coerce numeric slider expression `\"2\"` to 2, got {frozen}"
    );

    // 3. Short Hold segment (0.00015 frames) still reverses interior to original left value (0.0).
    let mut s3 = Session::new();
    let vt3 = s3.document().video_tracks[0].id.clone();
    let cid3_str = s3
        .execute(Command::AddClip {
            track_id: vt3,
            kind: "video".to_owned(),
            name: Some("ShortHold".to_owned()),
            start: 0.0,
            length: 1.0,
        })
        .unwrap()["clip_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let cid3 = Uuid::parse_str(&cid3_str).unwrap();
    s3.execute(Command::SetKeyframe {
        target_id: cid3_str.clone(),
        property: "time".to_owned(),
        frame: 0.5,
        value: json!(0.0),
        tween: Some(1),
    })
    .unwrap();
    s3.execute(Command::SetKeyframe {
        target_id: cid3_str.clone(),
        property: "time".to_owned(),
        frame: 0.50015,
        value: json!(10.0),
        tween: Some(0),
    })
    .unwrap();
    s3.execute(Command::ReverseClip { clip_id: cid3 }).unwrap();
    // Reversed interval is [1.0 - 0.50015, 1.0 - 0.5] = [0.49985, 0.5]; midpoint 0.499925 must be 0.0.
    let mid_val = s3
        .execute(Command::FreezeClipFrame {
            clip_id: cid3,
            frame: 0.499925,
        })
        .unwrap()["frozen_time"]
        .as_f64()
        .unwrap();
    assert!(
        (mid_val - 0.0).abs() < 1e-9,
        "short Hold segment (0.00015f) reversed midpoint must evaluate to 0.0, got {mid_val}"
    );
}
