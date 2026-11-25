use super::*;

#[test]
fn gpu_regression_text2d_alpha_precomp_gate_and_temporal_effects() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let mut plugins = PluginRegistry::new();
    plugins.enable("precomp-plus").expect("enable precomp-plus");
    plugins.enable("native-fx").expect("enable native-fx");

    // 1. Semi-transparent Text2D must not double-blend over itself (alpha 0.5 stays ~128, not ~191).
    let mut text_proj = Project::new_cm3_default();
    text_proj.sequence.background_color = [0.0, 0.0, 0.0, 0.0];
    text_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Text2D {
        text: "H".to_owned(),
        font: None,
        size: 600,
        shapes: vec![Shape2DNode {
            id: Uuid::new_v4(),
            name: "Fill".to_owned(),
            kind: Shape2DKind::Group {
                children: Vec::new(),
            },
            position: PropertyTrack::vec2([0.0, 0.0]),
            rotation: PropertyTrack::number(0.0),
            scale: PropertyTrack::vec2([1.0, 1.0]),
            operations: vec![DrawOp::Fill {
                color: PropertyTrack::constant(PropertyValue::ColorRgba([1.0, 1.0, 1.0, 0.5])),
            }],
            preserved: BTreeMap::new(),
        }],
    };
    let text_px = engine
        .render_frame_rgba8(&text_proj, &plugins, 0.0, 64, 64)
        .expect("text2d render");
    let max_alpha = text_px
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| p[3])
        .max()
        .unwrap_or(0);
    assert!(
        (115..=140).contains(&max_alpha),
        "expected semi-transparent Text2D alpha ~128 without double-blend, got {max_alpha}"
    );

    // 2. PrecompRef is blocked when precomp-plus plugin is disabled, and renders its
    // child tracks over a transparent nested canvas when enabled (P6 P0).
    let mut precomp_proj = Project::new_cm3_default();
    precomp_proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
    let comp_id = Uuid::new_v4();
    let mut comp_seq = Project::new_cm3_default().sequence;
    comp_seq.video_tracks[0].clips[0].layer.kind = LayerKind::Generator;
    comp_seq.video_tracks[0].clips[0]
        .layer
        .effects
        .push(EffectInstance {
        id: Uuid::new_v4(),
        name: "RedFill".to_owned(),
        enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
        kind: EffectKind::CustomShader {
            frag_shader:
                "precision mediump float; void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }"
                    .to_owned(),
            custom_properties: Vec::new(),
        },
        params: BTreeMap::new(),
        preserved: BTreeMap::new(),
    });
    precomp_proj.compositions.insert(comp_id, comp_seq);
    precomp_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::PrecompRef {
        composition_id: comp_id,
    };
    let mut no_precomp_plugins = PluginRegistry::new();
    no_precomp_plugins.disable_all_optional();
    let disabled_px = engine
        .render_frame_rgba8(&precomp_proj, &no_precomp_plugins, 0.0, 16, 16)
        .expect("precomp disabled render");
    assert_eq!(
        disabled_px[0], 0,
        "PrecompRef must not render when precomp-plus is disabled"
    );
    let enabled_px = engine
        .render_frame_rgba8(&precomp_proj, &plugins, 0.0, 16, 16)
        .expect("precomp enabled render");
    assert!(
        enabled_px[0] > 240,
        "PrecompRef must render when precomp-plus is enabled"
    );

    // 3. Echo and SmartMotionBlur actually sample neighbor frames in ProjectRenderEngine,
    // and Echo with startingIntensity > 1.0 maintains valid premultiplied alpha.
    let make_moving_proj = |fx: Option<EffectInstance>| {
        let mut p = Project::new_cm3_default();
        p.sequence.fps = 30;
        p.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        let mut layer = Layer {
            id: Uuid::new_v4(),
            name: "MovingShader".to_owned(),
            kind: LayerKind::Generator,
            position: PropertyTrack::vec2([0.0, 0.0]),
            scale: PropertyTrack::vec2([1.0, 1.0]),
            rotation: PropertyTrack::number(0.0),
            anchor: PropertyTrack::vec2([0.0, 0.0]),
            opacity: PropertyTrack::number(1.0),
            blending: "BLEND_SRC_OVER".to_owned(),
            effects: vec![EffectInstance {
                id: Uuid::new_v4(),
                name: "MovingBox".to_owned(),
                enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
                kind: EffectKind::CustomShader {
                    frag_shader: "precision mediump float; varying vec2 vUv; uniform float time; void main() { float cx = 0.2 + time * 1.5; float m = step(abs(vUv.x - cx), 0.12) * step(abs(vUv.y - 0.5), 0.25); gl_FragColor = vec4(m, m * 0.5, 0.0, m); }".to_owned(),
                    custom_properties: Vec::new(),
                },
                params: BTreeMap::new(),
                preserved: BTreeMap::new(),
            }],
            resolution: [32, 32],
            preserved: BTreeMap::new(),
        };
        if let Some(extra_fx) = fx {
            layer.effects.push(extra_fx);
        }
        p.sequence.video_tracks[0].clips[0].layer = layer;
        p
    };

    let base_px = engine
        .render_frame_rgba8(&make_moving_proj(None), &plugins, 6.0, 32, 32)
        .expect("base moving render");

    let mut echo_params = BTreeMap::new();
    echo_params.insert("echoTime".into(), PropertyTrack::number(-0.066));
    echo_params.insert("numberOfEchoes".into(), PropertyTrack::number(3.0));
    echo_params.insert("startingIntensity".into(), PropertyTrack::number(2.5));
    echo_params.insert("decay".into(), PropertyTrack::number(0.7));
    echo_params.insert("operator".into(), PropertyTrack::option(4)); // CompositeInFront
    let echo_fx = EffectInstance {
        id: Uuid::new_v4(),
        name: "Echo".to_owned(),
        enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
        kind: EffectKind::Plugin {
            plugin_id: "native-fx".to_owned(),
            effect_id: "echo".to_owned(),
        },
        params: echo_params,
        preserved: BTreeMap::new(),
    };
    let echo_px = engine
        .render_frame_rgba8(&make_moving_proj(Some(echo_fx)), &plugins, 6.0, 32, 32)
        .expect("echo moving render");
    assert_ne!(base_px, echo_px, "echo must modify moving layer frames");

    let mut smb_params = BTreeMap::new();
    smb_params.insert("blurAmount".into(), PropertyTrack::number(1.0));
    smb_params.insert("motionSensitivity".into(), PropertyTrack::number(90.0));
    smb_params.insert("view".into(), PropertyTrack::option(1)); // Motion vectors view
    let smb_fx = EffectInstance {
        id: Uuid::new_v4(),
        name: "SMB".to_owned(),
        enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
        kind: EffectKind::Plugin {
            plugin_id: "native-fx".to_owned(),
            effect_id: "smartmotionblur".to_owned(),
        },
        params: smb_params,
        preserved: BTreeMap::new(),
    };
    let smb_px = engine
        .render_frame_rgba8(&make_moving_proj(Some(smb_fx)), &plugins, 6.0, 32, 32)
        .expect("smb moving render");
    // In Motion Vectors view (mode == 1), without neighbor frames every pixel is (128, 128, 0, 255);
    // with neighbor frames bound, blue channel is 128 (0.5) and active motion pixels deviate.
    assert!(
        smb_px.as_chunks::<4>().0.iter().any(|p| p[2] > 100),
        "smartmotionblur must bind temporal neighbor frames instead of no-oping"
    );
}

#[test]
fn gpu_parity_uvscale_transparent_adjustment_precomp_shapes_text_and_masks() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let mut plugins = PluginRegistry::new();
    plugins.enable("expressions-plus").expect("enable expr+");
    plugins.enable("precomp-plus").expect("enable precomp+");

    // 1. P8 P0: `uvScale` == [1,1], space-normalized property names (`"My Color"` -> `My_Color`),
    //    and `zTime`/`zFrame`/`zFps`/`ctl_*` uniforms in custom GLSL shaders.
    let mut shader_proj = Project::new_cm3_default();
    shader_proj.sequence.fps = 30;
    shader_proj
        .sequence
        .controls
        .push(zc_project::SequenceControl {
            id: "c1".to_owned(),
            name: "Blue Boost".to_owned(),
            kind: SequenceControlKind::Slider,
            track: PropertyTrack::number(0.5),
            preserved: BTreeMap::new(),
        });
    shader_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Generator;
    shader_proj.sequence.video_tracks[0].clips[0]
        .layer
        .effects
        .push(EffectInstance {
            id: Uuid::new_v4(),
            name: "UvScaleAndCtl".to_owned(),
            enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
            kind: EffectKind::CustomShader {
                frag_shader: r#"
                    precision mediump float;
                    varying vec2 vUvScaled;
                    uniform vec2 uvScale;
                    uniform float Style_Amount;
                    uniform float ctl_Blue_Boost;
                    uniform float zFps;
                    void main() {
                        float r = uvScale.x * Style_Amount;
                        float g = (zFps > 29.0 && zFps < 31.0) ? uvScale.y : 0.0;
                        float b = ctl_Blue_Boost;
                        gl_FragColor = vec4(r, g, b, 1.0);
                    }
                "#
                .to_owned(),
                custom_properties: vec![CustomProperty {
                    name: "Style Amount".to_owned(), // space must normalize to Style_Amount!
                    prop_type: Some(0),
                    track: PropertyTrack::number(0.75),
                    metadata: BTreeMap::new(),
                }],
            },
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        });
    let spx = engine
        .render_frame_rgba8(&shader_proj, &plugins, 0.0, 16, 16)
        .expect("custom shader uvScale/ctl render");
    assert!(
        (180..=205).contains(&spx[0]) && spx[1] > 240 && (115..=140).contains(&spx[2]),
        "expected uvScale=[1,1], Style_Amount=0.75, zFps=30, ctl_Blue_Boost=0.5 -> ~(191,255,128), got {:?}",
        &spx[0..4]
    );

    // 2. P9 P0: Transparent background export clears root canvas to alpha == 0.
    let mut empty_proj = Project::new_cm3_default();
    empty_proj.sequence.background_color = [0.2, 0.3, 0.4, 1.0];
    empty_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Generator;
    let opaque_px = engine
        .render_frame_rgba8(&empty_proj, &plugins, 0.0, 16, 16)
        .expect("opaque render");
    assert_eq!(opaque_px[3], 255);
    engine.set_transparent_background(true);
    let transp_px = engine
        .render_frame_rgba8(&empty_proj, &plugins, 0.0, 16, 16)
        .expect("transparent render");
    engine.set_transparent_background(false);
    assert_eq!(
        &transp_px[0..4],
        &[0, 0, 0, 0],
        "transparent_background must clear root canvas to [0,0,0,0]"
    );

    // 3. P6 P0: Adjustment layer with alpha-zeroing mask preserves lower track backdrop!
    let mut adj_proj = Project::new_cm3_default();
    adj_proj.sequence.background_color = [0.0, 0.0, 1.0, 1.0]; // Opaque blue backdrop
    adj_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Adjustment;
    adj_proj.sequence.video_tracks[0].clips[0]
        .layer
        .effects
        .push(EffectInstance {
            id: Uuid::new_v4(),
            name: "HalfMask".to_owned(),
            enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
            kind: EffectKind::CustomShader {
                // Left half outputs red opaque, right half outputs transparent (alpha 0)
                frag_shader: "precision mediump float; varying vec2 vUv; void main() { gl_FragColor = vUv.x < 0.5 ? vec4(1.0, 0.0, 0.0, 1.0) : vec4(0.0); }".to_owned(),
                custom_properties: Vec::new(),
            },
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        });
    let adj_px = engine
        .render_frame_rgba8(&adj_proj, &plugins, 0.0, 16, 16)
        .expect("adjustment alpha render");
    let left_px = &adj_px[(8 * 16 + 2) * 4..(8 * 16 + 2) * 4 + 4];
    let right_px = &adj_px[(8 * 16 + 13) * 4..(8 * 16 + 13) * 4 + 4];
    assert!(
        left_px[0] > 240 && left_px[2] < 15,
        "left half must be red: {left_px:?}"
    );
    assert!(
        right_px[2] > 240 && right_px[3] == 255,
        "right half where Adjustment alpha==0 must reveal blue backdrop, got {right_px:?}"
    );

    // 4. P6 P0 + P7 P0: PrecompRef clears nested canvas to transparent & honors `offset`.
    let mut pre_proj = Project::new_cm3_default();
    pre_proj.sequence.background_color = [0.0, 1.0, 0.0, 1.0]; // Green background below precomp
    let comp_id = Uuid::new_v4();
    // Default opaque black background_color!
    let mut comp_seq = Sequence {
        fps: 30,
        length_frames: 120.0,
        ..Default::default()
    };
    // Put a red clip starting at frame 10 inside nested comp
    let mut nested_clip = Project::new_cm3_default().sequence.video_tracks[0].clips[0].clone();
    nested_clip.start = 10.0;
    nested_clip.length = 20.0;
    nested_clip.layer.kind = LayerKind::Generator;
    nested_clip.layer.effects.push(EffectInstance {
        id: Uuid::new_v4(),
        name: "Red".to_owned(),
        enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
        kind: EffectKind::CustomShader {
            frag_shader:
                "precision mediump float; void main() { gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); }"
                    .to_owned(),
            custom_properties: Vec::new(),
        },
        params: BTreeMap::new(),
        preserved: BTreeMap::new(),
    });
    comp_seq.video_tracks.push(zc_project::VideoTrack {
        id: Uuid::new_v4(),
        name: "V1".to_owned(),
        enabled: true,
        solo: false,
        locked: false,
        is_3d: false,
        clips: vec![nested_clip],
        preserved: BTreeMap::new(),
    });
    pre_proj.compositions.insert(comp_id, comp_seq);
    pre_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::PrecompRef {
        composition_id: comp_id,
    };
    // At frame 0 with offset 0: nested comp has no active clips -> MUST stay transparent -> green bg shows!
    let pre_f0 = engine
        .render_frame_rgba8(&pre_proj, &plugins, 0.0, 16, 16)
        .expect("precomp frame 0");
    assert!(
        pre_f0[1] > 240 && pre_f0[0] < 15,
        "empty PrecompRef frame must be transparent and show green bg, got {:?}",
        &pre_f0[0..4]
    );
    // With offset = 15 frames at parent frame 0: nested comp evaluates frame 15 -> red clip shows!
    pre_proj.sequence.video_tracks[0].clips[0]
        .layer
        .preserved
        .insert("offset".to_owned(), serde_json::json!(15.0));
    let pre_off = engine
        .render_frame_rgba8(&pre_proj, &plugins, 0.0, 16, 16)
        .expect("precomp offset 15");
    assert!(
        pre_off[0] > 240 && pre_off[1] < 15,
        "PrecompRef with offset=15 must show red nested clip at frame 0, got {:?}",
        &pre_off[0..4]
    );

    // 5. P6 P0: Shape2D Bezier path + PresetShape2D canonical 200x200 + Text2D transform/stroke + zoidiumMasks.
    let mut shape_proj = Project::new_cm3_default();
    shape_proj.sequence.resolution = [100, 100];
    shape_proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
    // A small 20x20 square centered at (25, 0) in sequence coords (shifted right of center!)
    shape_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Shape2D {
        shapes: vec![Shape2DNode {
            id: Uuid::new_v4(),
            name: "Square".to_owned(),
            kind: Shape2DKind::Path {
                path: serde_json::json!({
                    "commands": [
                        {"type": "M", "x": -10.0, "y": -10.0},
                        {"type": "L", "x":  10.0, "y": -10.0},
                        {"type": "L", "x":  10.0, "y":  10.0},
                        {"type": "L", "x": -10.0, "y":  10.0},
                        {"type": "Z"}
                    ]
                }),
            },
            position: PropertyTrack::vec2([25.0, 0.0]),
            rotation: PropertyTrack::number(0.0),
            scale: PropertyTrack::vec2([1.0, 1.0]),
            operations: vec![DrawOp::Fill {
                color: PropertyTrack::constant(PropertyValue::ColorRgba([1.0, 0.0, 0.0, 1.0])),
            }],
            preserved: BTreeMap::new(),
        }],
    };
    let sh_px = engine
        .render_frame_rgba8(&shape_proj, &plugins, 0.0, 100, 100)
        .expect("shape2d render");
    let center_sh = &sh_px[(50 * 100 + 50) * 4..(50 * 100 + 50) * 4 + 4];
    let right_sh = &sh_px[(50 * 100 + 75) * 4..(50 * 100 + 75) * 4 + 4];
    assert!(
        center_sh[0] < 10 && right_sh[0] > 240,
        "Shape2D path shifted to x=+25 must be black at center {center_sh:?} and red at x=75 {right_sh:?}"
    );

    // Text2D with DrawOp::Stroke + node position shift
    let mut txt_proj = Project::new_cm3_default();
    txt_proj.sequence.resolution = [128, 128];
    txt_proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
    txt_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Text2D {
        text: "I".to_owned(),
        font: None,
        size: 64,
        shapes: vec![Shape2DNode {
            id: Uuid::new_v4(),
            name: "TextNode".to_owned(),
            kind: Shape2DKind::Group {
                children: Vec::new(),
            },
            position: PropertyTrack::vec2([0.0, 0.0]),
            rotation: PropertyTrack::number(0.0),
            scale: PropertyTrack::vec2([1.0, 1.0]),
            operations: vec![
                DrawOp::Stroke {
                    color: PropertyTrack::color([0.0, 1.0, 0.0]),
                    size: PropertyTrack::number(10.0),
                    opacity: PropertyTrack::number(1.0),
                },
                DrawOp::Fill {
                    color: PropertyTrack::constant(PropertyValue::ColorRgba([1.0, 0.0, 0.0, 1.0])),
                },
            ],
            preserved: BTreeMap::new(),
        }],
    };
    let txt_px = engine
        .render_frame_rgba8(&txt_proj, &plugins, 0.0, 128, 128)
        .expect("text2d stroke render");
    let has_red_fill = txt_px
        .as_chunks::<4>()
        .0
        .iter()
        .any(|p| p[0] > 200 && p[1] < 50);
    let has_green_stroke = txt_px
        .as_chunks::<4>()
        .0
        .iter()
        .any(|p| p[1] > 200 && p[0] < 50);
    assert!(
        has_red_fill && has_green_stroke,
        "Text2D must render both Fill (red={has_red_fill}) and DrawOp::Stroke (green={has_green_stroke})"
    );

    // zoidiumMasks: square mask over left half of a full white Generator
    let mut mask_proj = Project::new_cm3_default();
    mask_proj.sequence.resolution = [64, 64];
    mask_proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
    let mask_layer = &mut mask_proj.sequence.video_tracks[0].clips[0].layer;
    mask_layer.kind = LayerKind::Generator;
    mask_layer.effects.push(EffectInstance {
        id: Uuid::new_v4(),
        name: "White".to_owned(),
        enabled: PropertyTrack::constant(PropertyValue::Bool(true)),
        kind: EffectKind::CustomShader {
            frag_shader: "precision mediump float; void main() { gl_FragColor = vec4(1.0); }"
                .to_owned(),
            custom_properties: Vec::new(),
        },
        params: BTreeMap::new(),
        preserved: BTreeMap::new(),
    });
    mask_layer.preserved.insert(
        "zoidiumMasks".to_owned(),
        serde_json::json!([{
            "enabled": 1,
            "closed": 1,
            "mode": "add",
            "feather": 0,
            "expansion": 0,
            "opacity": 1,
            "invert": 0,
            "vertices": [
                {"point": [0, 0], "in": [0, 0], "out": [0, 0]},
                {"point": [32, 0], "in": [0, 0], "out": [0, 0]},
                {"point": [32, 64], "in": [0, 0], "out": [0, 0]},
                {"point": [0, 64], "in": [0, 0], "out": [0, 0]}
            ]
        }]),
    );
    let m_px = engine
        .render_frame_rgba8(&mask_proj, &plugins, 0.0, 64, 64)
        .expect("zoidiumMasks render");
    let m_left = &m_px[(32 * 64 + 10) * 4..(32 * 64 + 10) * 4 + 4];
    let m_right = &m_px[(32 * 64 + 50) * 4..(32 * 64 + 50) * 4 + 4];
    assert!(
        m_left[0] > 240 && m_right[0] < 10,
        "zoidiumMasks must keep left half white {m_left:?} and mask right half black {m_right:?}"
    );
}

#[test]
fn gpu_c5_layer_input_honors_typed_sequence_layer_masks() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let plugins = PluginRegistry::new();

    let mut project = Project::new_cm3_default();
    project.sequence.resolution = [64, 64];
    project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];

    // Hidden source track 0: solid red layer with a typed LayerMask covering only the left half [0..32, 0..64].
    let src_layer = Layer::new(
        "RedMasked",
        LayerKind::PresetShape2D {
            preset: PresetShapeKind::Rectangle,
            shapes: vec![Shape2DNode {
                id: Uuid::new_v4(),
                name: "FullRect".to_owned(),
                kind: Shape2DKind::Group {
                    children: Vec::new(),
                },
                position: PropertyTrack::vec2([0.0, 0.0]),
                rotation: PropertyTrack::number(0.0),
                scale: PropertyTrack::vec2([1.0, 1.0]),
                operations: vec![DrawOp::Fill {
                    color: PropertyTrack::constant(PropertyValue::ColorRgba([1.0, 0.0, 0.0, 1.0])),
                }],
                preserved: BTreeMap::new(),
            }],
        },
        [64, 64],
    );
    let src_layer_id = src_layer.id;
    let src_track = zc_project::VideoTrack {
        id: Uuid::new_v4(),
        name: "MaskSource".to_owned(),
        enabled: false,
        solo: false,
        locked: false,
        is_3d: false,
        clips: vec![VideoClip {
            id: Uuid::new_v4(),
            name: "SrcClip".to_owned(),
            start: 0.0,
            length: 60.0,
            link: None,
            media: None,
            time_curve: PropertyTrack::linear_time_curve(60.0, 30),
            layer: src_layer,
            preserved: BTreeMap::new(),
        }],
        preserved: BTreeMap::new(),
    };
    let src_track_id = src_track.id;

    // Typed mask in `sequence.layer_masks` (left half x in [0, 32], y in [0, 64] in sequence space).
    project.sequence.layer_masks.insert(
        src_layer_id,
        vec![zc_project::LayerMask {
            id: "mask-left-half".to_owned(),
            name: "LeftHalf".to_owned(),
            enabled: true,
            mode: zc_project::MaskMode::Add,
            invert: false,
            feather: 0.0,
            expansion: 0.0,
            opacity: 1.0,
            closed: true,
            locked: false,
            outline_color: String::new(),
            vertex_ids: vec![
                "v0".to_owned(),
                "v1".to_owned(),
                "v2".to_owned(),
                "v3".to_owned(),
            ],
            vertex_modes: BTreeMap::new(),
            path: PropertyTrack::constant(PropertyValue::Json(serde_json::json!([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 32.0, 64.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 64.0, 0.0, 0.0, 0.0, 0.0
            ]))),
            preserved: BTreeMap::new(),
        }],
    );

    let make_consumer = |stage: &str| {
        let mut p = project.clone();
        let mut dst_layer = Layer::new("Canvas", LayerKind::Generator, [64, 64]);
        let mut params = BTreeMap::new();
        params.insert(
            "source".to_owned(),
            PropertyTrack::text(format!("track:id:{src_track_id}")),
        );
        params.insert("stage".to_owned(), PropertyTrack::text(stage));
        params.insert("opacity".to_owned(), PropertyTrack::number(1.0));
        dst_layer.effects.push(EffectInstance {
            id: Uuid::new_v4(),
            name: "LayerInput".to_owned(),
            kind: EffectKind::Standard("layer-input".to_owned()),
            enabled: PropertyTrack::option(1),
            params,
            preserved: BTreeMap::new(),
        });
        let dst_track = zc_project::VideoTrack {
            id: Uuid::new_v4(),
            name: "Consumer".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![VideoClip {
                id: Uuid::new_v4(),
                name: "DstClip".to_owned(),
                start: 0.0,
                length: 60.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                layer: dst_layer,
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        };
        p.sequence.video_tracks = vec![src_track.clone(), dst_track];
        p
    };

    // Per Zoidium `layer-input/README.md` (Wave 2D B2):
    // - `"effects-masks"` runs all effects AND `zoidiumMasks` / typed `sequence.layer_masks`.
    // - `"masks"` runs ONLY native CM3 `mask` effects and skips `zoidiumMasks`.
    let proj_effects_masks = make_consumer("effects-masks");
    let px_em = engine
        .render_frame_rgba8(&proj_effects_masks, &plugins, 0.0, 64, 64)
        .expect("render layer-input effects-masks");
    let left_r = px_em[(32 * 64 + 12) * 4];
    let right_r = px_em[(32 * 64 + 52) * 4];
    assert!(
        left_r > 200 && right_r < 20,
        "LayerInput stage `effects-masks` must honor typed sequence.layer_masks (left_r={left_r}, right_r={right_r})"
    );

    let proj_masks_only = make_consumer("masks");
    let px_mo = engine
        .render_frame_rgba8(&proj_masks_only, &plugins, 0.0, 64, 64)
        .expect("render layer-input masks-only");
    let mo_left_r = px_mo[(32 * 64 + 12) * 4];
    let mo_right_r = px_mo[(32 * 64 + 52) * 4];
    assert!(
        mo_left_r > 200 && mo_right_r > 200,
        "LayerInput stage `masks` must run only native CM3 mask effects and skip zoidiumMasks (left_r={mo_left_r}, right_r={mo_right_r})"
    );
}

#[test]
fn gpu_c5_shader_plus_curve_range_gradient_cycle_and_text_property() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let plugins = PluginRegistry::new();

    let make_shader_proj = |glsl: &str, custom_properties: Vec<CustomProperty>| {
        let mut project = Project::new_cm3_default();
        project.sequence.resolution = [64, 64];
        project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        let mut layer = Layer::new(
            "Solid",
            LayerKind::PresetShape2D {
                preset: PresetShapeKind::Rectangle,
                shapes: vec![Shape2DNode {
                    id: Uuid::new_v4(),
                    name: "BG".to_owned(),
                    kind: Shape2DKind::Group {
                        children: Vec::new(),
                    },
                    position: PropertyTrack::vec2([0.0, 0.0]),
                    rotation: PropertyTrack::number(0.0),
                    scale: PropertyTrack::vec2([1.0, 1.0]),
                    operations: vec![DrawOp::Fill {
                        color: PropertyTrack::constant(PropertyValue::ColorRgba([
                            0.5, 0.5, 0.5, 1.0,
                        ])),
                    }],
                    preserved: BTreeMap::new(),
                }],
            },
            [64, 64],
        );
        layer.effects.push(EffectInstance {
            id: Uuid::new_v4(),
            name: "CustomShader".to_owned(),
            kind: EffectKind::CustomShader {
                frag_shader: glsl.to_owned(),
                custom_properties,
            },
            enabled: PropertyTrack::option(1),
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        });
        project.sequence.video_tracks = vec![zc_project::VideoTrack {
            id: Uuid::new_v4(),
            name: "Shader".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![VideoClip {
                id: Uuid::new_v4(),
                name: "ShaderClip".to_owned(),
                start: 0.0,
                length: 60.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                layer,
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        }];
        project
    };

    // 1. bind_curve_ranges: two-arg `zCurve(Remap, 0.5)` with `range: [0.4, 0.8]` on identity
    //    curve (0.5 -> 0.5 in [0,1]) must evaluate to mix(0.4, 0.8, 0.5) = 0.6 (~153), NOT 0.5 (~128)!
    let curve_glsl = r#"
        precision highp float;
        // #include <zoidium/curve>
        uniform sampler2D Remap;
        varying vec2 vUv;
        void main() {
            float v = zCurve(Remap, 0.5);
            gl_FragColor = vec4(vec3(v), 1.0);
        }
    "#;
    let mut curve_meta = BTreeMap::new();
    curve_meta.insert(
        "type".to_owned(),
        serde_json::json!({ "zoidiumShader": { "kind": "curve" } }),
    );
    let curve_prop = CustomProperty {
        name: "Remap".to_owned(),
        prop_type: Some(10),
        track: PropertyTrack::constant(PropertyValue::Curve(serde_json::json!({
            "range": [0.4, 0.8],
            "points": [[0.0, 0.0], [1.0, 1.0]]
        }))),
        metadata: curve_meta,
    };
    let px_curve = engine
        .render_frame_rgba8(
            &make_shader_proj(curve_glsl, vec![curve_prop]),
            &plugins,
            0.0,
            64,
            64,
        )
        .expect("render curve range");
    assert!(
        (i32::from(px_curve[0]) - 153).abs() <= 4,
        "bind_curve_ranges must rewrite zCurve(Remap, 0.5) with Remap_range=[0.4,0.8] -> ~153, got {}",
        px_curve[0]
    );

    // 2. bind_gradient_cycles: two-arg `zGradientCycle(Ramp, vUv.x)` with `repetitions = 4.0`
    //    must cycle 4 times across X (so x=0.125 and x=0.375 both hit t=0.5 = green/mid,
    //    and x=0.26 wraps back near red at t~0.04, unlike repetitions=1!).
    let grad_glsl = r#"
        precision highp float;
        // #include <zoidium/gradient>
        uniform sampler2D Ramp;
        varying vec2 vUv;
        void main() {
            gl_FragColor = zGradientCycle(Ramp, vUv.x);
        }
    "#;
    let mut grad_meta = BTreeMap::new();
    grad_meta.insert(
        "type".to_owned(),
        serde_json::json!({ "zoidiumShader": { "kind": "gradient", "cyclic": true, "repetitions": 4.0 } }),
    );
    let grad_prop = CustomProperty {
        name: "Ramp".to_owned(),
        prop_type: Some(5),
        track: PropertyTrack::gradient(vec![
            ProjectGradientStop {
                position: 0.0,
                color: "rgba(255,0,0,1.0)".to_owned(),
            },
            ProjectGradientStop {
                position: 1.0,
                color: "rgba(0,0,255,1.0)".to_owned(),
            },
        ]),
        metadata: grad_meta,
    };
    let px_grad = engine
        .render_frame_rgba8(
            &make_shader_proj(grad_glsl, vec![grad_prop]),
            &plugins,
            0.0,
            64,
            64,
        )
        .expect("render gradient cycle");
    // At x = 17/64 (~0.2656): fract(0.2656 * 4) = 0.0625 -> strongly red (R >> B)!
    // Without bind_gradient_cycles (repetitions=1), t=0.2656 would have B~68; at x=33/64 (~0.5156):
    // fract(0.5156 * 4) = 0.0625 -> strongly red (R > 200, B < 40), whereas without rewrite R~B~128!
    let mid_idx = (32 * 64 + 33) * 4;
    assert!(
        px_grad[mid_idx] > 200 && px_grad[mid_idx + 2] < 45,
        "bind_gradient_cycles with repetitions=4 must wrap at x~0.516 to red, got {:?}",
        &px_grad[mid_idx..mid_idx + 4]
    );

    // 3. Shader+ `text` property: Layout mode (`zText` + `_size` + `_aspect`) AND Atlas mode (`zGlyph` + `_grid` + `_count`).
    let text_layout_glsl = r#"
        precision highp float;
        // #include <zoidium/text>
        uniform sampler2D Title;
        uniform vec2 Title_size;
        uniform vec2 Title_grid;
        uniform float Title_aspect;
        uniform float Title_count;
        varying vec2 vUv;
        void main() {
            float cov = zText(Title, vUv);
            float ok = step(4.0, Title_size.x) * step(4.0, Title_size.y)
                * step(0.1, Title_aspect) * step(0.5, Title_grid.x) * step(0.5, Title_count);
            gl_FragColor = vec4(cov, ok, 0.0, 1.0);
        }
    "#;
    let mut text_meta = BTreeMap::new();
    text_meta.insert(
        "type".to_owned(),
        serde_json::json!({ "zoidiumShader": { "kind": "text" } }),
    );
    let text_layout_prop = CustomProperty {
        name: "Title".to_owned(),
        prop_type: Some(7),
        track: PropertyTrack::text(
            serde_json::json!({ "text": "HI", "size": 64, "mode": "layout", "sdf": false })
                .to_string(),
        ),
        metadata: text_meta.clone(),
    };
    let px_text = engine
        .render_frame_rgba8(
            &make_shader_proj(text_layout_glsl, vec![text_layout_prop]),
            &plugins,
            0.0,
            64,
            64,
        )
        .expect("render shader+ text layout");
    let cov_pixels = px_text
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[0] > 128)
        .count();
    assert!(
        cov_pixels > 40 && px_text[1] > 200,
        "Shader+ text layout must rasterize glyphs and bind _size/_grid/_aspect/_count (cov={cov_pixels}, ok_g={})",
        px_text[1]
    );

    let text_atlas_glsl = r#"
        precision highp float;
        // #include <zoidium/text>
        uniform sampler2D Atlas;
        uniform vec2 Atlas_grid;
        uniform float Atlas_count;
        varying vec2 vUv;
        void main() {
            float g0 = zGlyph(Atlas, Atlas_grid, 0.0, vUv);
            float ok = step(1.5, Atlas_grid.x) * step(3.5, Atlas_count);
            gl_FragColor = vec4(g0, ok, 0.0, 1.0);
        }
    "#;
    let text_atlas_prop = CustomProperty {
        name: "Atlas".to_owned(),
        prop_type: Some(7),
        track: PropertyTrack::text(
            serde_json::json!({ "text": "ABCD", "size": 48, "mode": "atlas", "sdf": true })
                .to_string(),
        ),
        metadata: text_meta,
    };
    let px_atlas = engine
        .render_frame_rgba8(
            &make_shader_proj(text_atlas_glsl, vec![text_atlas_prop]),
            &plugins,
            0.0,
            64,
            64,
        )
        .expect("render shader+ text atlas");
    let atlas_inside = px_atlas
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[0] > 135)
        .count();
    assert!(
        atlas_inside > 40 && px_atlas[1] > 200,
        "Shader+ text atlas+SDF must rasterize grid cells and bind Atlas_grid/Atlas_count (inside={atlas_inside}, ok_g={})",
        px_atlas[1]
    );
}

#[test]
fn gpu_c5_temporal_aliases_echo_legacy_and_smart_motion_blur_lite() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let mut plugins = PluginRegistry::new();
    plugins.enable("native-fx").expect("enable native-fx");

    let make_moving_box_proj = |effect: Option<EffectInstance>| {
        let mut project = Project::new_cm3_default();
        project.sequence.resolution = [64, 64];
        project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        let mut pos_track = PropertyTrack::vec2([-20.0, 0.0]);
        pos_track.animated = true;
        pos_track.keyframes = vec![
            PropertyKeyframe::linear(0.0, PropertyValue::Vec2([-20.0, 0.0])),
            PropertyKeyframe::linear(10.0, PropertyValue::Vec2([20.0, 0.0])),
        ];
        let mut layer = Layer::new(
            "Mover",
            LayerKind::PresetShape2D {
                preset: PresetShapeKind::Rectangle,
                shapes: vec![Shape2DNode {
                    id: Uuid::new_v4(),
                    name: "Square".to_owned(),
                    kind: Shape2DKind::Group {
                        children: Vec::new(),
                    },
                    position: pos_track,
                    rotation: PropertyTrack::number(0.0),
                    scale: PropertyTrack::vec2([0.06, 0.06]),
                    operations: vec![DrawOp::Fill {
                        color: PropertyTrack::constant(PropertyValue::ColorRgba([
                            1.0, 1.0, 1.0, 1.0,
                        ])),
                    }],
                    preserved: BTreeMap::new(),
                }],
            },
            [64, 64],
        );
        if let Some(fx) = effect {
            layer.effects.push(fx);
        }
        project.sequence.video_tracks = vec![zc_project::VideoTrack {
            id: Uuid::new_v4(),
            name: "Moving".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![VideoClip {
                id: Uuid::new_v4(),
                name: "MovingClip".to_owned(),
                start: 0.0,
                length: 60.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                layer,
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        }];
        project
    };

    let px_base = engine
        .render_frame_rgba8(&make_moving_box_proj(None), &plugins, 5.0, 64, 64)
        .expect("render baseline moving box");
    let count_lit =
        |px: &[u8]| -> usize { px.as_chunks::<4>().0.iter().filter(|p| p[0] > 20).count() };
    let base_lit = count_lit(&px_base);

    // 1. `echo-legacy` must run full temporal multi-frame echo sampling (trailing ghost copies).
    let mut echo_params = BTreeMap::new();
    echo_params.insert("echoes".to_owned(), PropertyTrack::number(3.0));
    echo_params.insert("delay".to_owned(), PropertyTrack::number(1.0));
    echo_params.insert("decay".to_owned(), PropertyTrack::number(0.8));
    echo_params.insert("opacity".to_owned(), PropertyTrack::number(0.9));
    let px_echo = engine
        .render_frame_rgba8(
            &make_moving_box_proj(Some(EffectInstance {
                id: Uuid::new_v4(),
                name: "EchoLegacy".to_owned(),
                kind: EffectKind::Plugin {
                    plugin_id: "native-fx".to_owned(),
                    effect_id: "echo-legacy".to_owned(),
                },
                enabled: PropertyTrack::option(1),
                params: echo_params,
                preserved: BTreeMap::new(),
            })),
            &plugins,
            5.0,
            64,
            64,
        )
        .expect("render echo-legacy");
    let echo_lit = count_lit(&px_echo);
    assert!(
        echo_lit > base_lit + 50,
        "echo-legacy must sample temporal echo taps (base_lit={base_lit}, echo_lit={echo_lit})"
    );

    // 2. `smart-motion-blur-lite` must sample prev/next neighbor frames and smear motion!
    let mut smb_params = BTreeMap::new();
    smb_params.insert("strength".to_owned(), PropertyTrack::number(2.0));
    smb_params.insert("samples".to_owned(), PropertyTrack::number(8.0));
    let px_smb = engine
        .render_frame_rgba8(
            &make_moving_box_proj(Some(EffectInstance {
                id: Uuid::new_v4(),
                name: "SMBLite".to_owned(),
                kind: EffectKind::Plugin {
                    plugin_id: "native-fx".to_owned(),
                    effect_id: "smart-motion-blur-lite".to_owned(),
                },
                enabled: PropertyTrack::option(1),
                params: smb_params,
                preserved: BTreeMap::new(),
            })),
            &plugins,
            5.0,
            64,
            64,
        )
        .expect("render smart-motion-blur-lite");
    let smb_diff: usize = px_base
        .iter()
        .zip(px_smb.iter())
        .filter(|(a, b)| a.abs_diff(**b) > 15)
        .count();
    assert!(
        smb_diff > 20,
        "smart-motion-blur-lite must perform temporal frame sampling and differ from static frame (diff={smb_diff})"
    );
}

#[test]
fn gpu_c5_transparent_background_cache_separation() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let plugins = PluginRegistry::new();
    let mut project = Project::new_cm3_default();
    project.sequence.resolution = [32, 32];
    project.sequence.background_color = [0.2, 0.5, 0.8, 1.0];
    project.sequence.video_tracks.clear();

    // Render cached with opaque background first.
    engine.set_transparent_background(false);
    let px_opaque = engine
        .render_frame_rgba8_cached(&project, &plugins, 0.0, 32, 32, 42)
        .expect("cached opaque");
    assert_eq!(px_opaque[3], 255, "opaque render must have alpha=255");
    assert!(px_opaque[2] > 180, "opaque render must have blue bg");

    // Toggle transparent background on the SAME engine, frame, dimensions, and revision.
    engine.set_transparent_background(true);
    let px_transparent = engine
        .render_frame_rgba8_cached(&project, &plugins, 0.0, 32, 32, 42)
        .expect("cached transparent");
    assert_eq!(
        &px_transparent[..4],
        &[0, 0, 0, 0],
        "toggling set_transparent_background(true) must NOT return stale cached opaque pixels"
    );

    // Toggle back to opaque on the SAME revision.
    engine.set_transparent_background(false);
    let px_opaque_again = engine
        .render_frame_rgba8_cached(&project, &plugins, 0.0, 32, 32, 42)
        .expect("cached opaque again");
    assert_eq!(
        px_opaque_again[3], 255,
        "toggling set_transparent_background(false) must NOT return stale cached transparent pixels"
    );
}

#[test]
fn gpu_wave2d_echo_on_adjustment_layer_renders_temporal_trail() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let mut plugins = PluginRegistry::new();
    plugins.enable("native-fx").expect("enable native-fx");

    let make_proj = |with_adj_echo: bool| {
        let mut project = Project::new_cm3_default();
        project.sequence.resolution = [64, 64];
        project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        let mut pos_track = PropertyTrack::vec2([-22.0, 0.0]);
        pos_track.animated = true;
        pos_track.keyframes = vec![
            PropertyKeyframe::linear(0.0, PropertyValue::Vec2([-22.0, 0.0])),
            PropertyKeyframe::linear(10.0, PropertyValue::Vec2([22.0, 0.0])),
        ];
        let moving_layer = Layer::new(
            "MovingSquare",
            LayerKind::PresetShape2D {
                preset: PresetShapeKind::Rectangle,
                shapes: vec![Shape2DNode {
                    id: Uuid::new_v4(),
                    name: "Sq".to_owned(),
                    kind: Shape2DKind::Group {
                        children: Vec::new(),
                    },
                    position: pos_track,
                    rotation: PropertyTrack::number(0.0),
                    scale: PropertyTrack::vec2([0.06, 0.06]),
                    operations: vec![DrawOp::Fill {
                        color: PropertyTrack::constant(PropertyValue::ColorRgba([
                            1.0, 1.0, 1.0, 1.0,
                        ])),
                    }],
                    preserved: BTreeMap::new(),
                }],
            },
            [64, 64],
        );
        let mut tracks = vec![zc_project::VideoTrack {
            id: Uuid::new_v4(),
            name: "BaseTrack".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![VideoClip {
                id: Uuid::new_v4(),
                name: "MoveClip".to_owned(),
                start: 0.0,
                length: 60.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                layer: moving_layer,
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        }];
        if with_adj_echo {
            let mut adj = Layer::new("AdjEcho", LayerKind::Adjustment, [64, 64]);
            let mut echo_params = BTreeMap::new();
            echo_params.insert("echoes".to_owned(), PropertyTrack::number(3.0));
            echo_params.insert("delay".to_owned(), PropertyTrack::number(1.0));
            echo_params.insert("decay".to_owned(), PropertyTrack::number(0.85));
            echo_params.insert("opacity".to_owned(), PropertyTrack::number(0.95));
            adj.effects.push(EffectInstance {
                id: Uuid::new_v4(),
                name: "Echo".to_owned(),
                kind: EffectKind::Plugin {
                    plugin_id: "native-fx".to_owned(),
                    effect_id: "echo".to_owned(),
                },
                enabled: PropertyTrack::option(1),
                params: echo_params,
                preserved: BTreeMap::new(),
            });
            tracks.push(zc_project::VideoTrack {
                id: Uuid::new_v4(),
                name: "AdjTrack".to_owned(),
                enabled: true,
                solo: false,
                locked: false,
                is_3d: false,
                clips: vec![VideoClip {
                    id: Uuid::new_v4(),
                    name: "AdjClip".to_owned(),
                    start: 0.0,
                    length: 60.0,
                    link: None,
                    media: None,
                    time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                    layer: adj,
                    preserved: BTreeMap::new(),
                }],
                preserved: BTreeMap::new(),
            });
        }
        project.sequence.video_tracks = tracks;
        project
    };

    let px_base = engine
        .render_frame_rgba8(&make_proj(false), &plugins, 5.0, 64, 64)
        .expect("render baseline moving square");
    let px_adj_echo = engine
        .render_frame_rgba8(&make_proj(true), &plugins, 5.0, 64, 64)
        .expect("render adjustment echo");
    let count_lit =
        |px: &[u8]| -> usize { px.as_chunks::<4>().0.iter().filter(|p| p[0] > 20).count() };
    let base_lit = count_lit(&px_base);
    let adj_lit = count_lit(&px_adj_echo);
    assert!(
        adj_lit > base_lit + 40,
        "echo on Adjustment layer must re-render lower tracks at historical frames and leave a visible trail (base_lit={base_lit}, adj_lit={adj_lit})"
    );
}

#[test]
fn gpu_wave2d_shape_plus_gradient_fill_and_repeater() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let mut plugins = PluginRegistry::new();
    plugins.enable("shape-plus").expect("enable shape-plus");

    let mut project = Project::new_cm3_default();
    project.sequence.resolution = [128, 64];
    project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
    let payload = serde_json::json!({
        "version": 1,
        "contents": [{
            "id": "g1",
            "kind": "group",
            "properties": {},
            "contents": [
                { "id": "r1", "kind": "rectangle", "properties": { "size": [28.0, 36.0], "position": [-36.0, 0.0] } },
                {
                    "id": "gf1",
                    "kind": "gradientFill",
                    "properties": {
                        "gradientType": "linear",
                        "startPoint": [-50.0, 0.0],
                        "endPoint": [-22.0, 0.0],
                        "stops": [
                            { "position": 0.0, "color": [1.0, 0.0, 0.0, 1.0] },
                            { "position": 1.0, "color": [0.0, 0.0, 1.0, 1.0] }
                        ]
                    }
                },
                {
                    "id": "rep1",
                    "kind": "repeater",
                    "properties": {
                        "copies": 3.0,
                        "offset": 0.0,
                        "position": [36.0, 0.0],
                        "scale": [1.0, 1.0],
                        "rotation": 0.0,
                        "startOpacity": 1.0,
                        "endOpacity": 1.0
                    }
                }
            ]
        }]
    });
    let layer = Layer::new(
        "ShapePlusLayer",
        LayerKind::Plugin {
            plugin_type: "zoidium:shape-plus/shape-layer".to_owned(),
            payload,
        },
        [128, 64],
    );
    project.sequence.video_tracks = vec![zc_project::VideoTrack {
        id: Uuid::new_v4(),
        name: "ShapePlusTrack".to_owned(),
        enabled: true,
        solo: false,
        locked: false,
        is_3d: false,
        clips: vec![VideoClip {
            id: Uuid::new_v4(),
            name: "SPClip".to_owned(),
            start: 0.0,
            length: 60.0,
            link: None,
            media: None,
            time_curve: PropertyTrack::linear_time_curve(60.0, 30),
            layer,
            preserved: BTreeMap::new(),
        }],
        preserved: BTreeMap::new(),
    }];

    let px = engine
        .render_frame_rgba8(&project, &plugins, 0.0, 128, 64)
        .expect("render shape+ gradient+repeater");
    let mut red_dominant = 0usize;
    let mut blue_dominant = 0usize;
    let mut distinct_colors = HashSet::new();
    for p in px.as_chunks::<4>().0 {
        if p[0] > 40 || p[2] > 40 {
            distinct_colors.insert((p[0] / 16, p[1] / 16, p[2] / 16));
        }
        if p[0] > p[2].saturating_add(40) {
            red_dominant += 1;
        }
        if p[2] > p[0].saturating_add(40) {
            blue_dominant += 1;
        }
    }
    assert!(
        red_dominant > 100 && blue_dominant > 100 && distinct_colors.len() >= 4,
        "Shape+ gradientFill + repeater must render distinct non-flat gradient copies (red={red_dominant}, blue={blue_dominant}, distinct={})",
        distinct_colors.len()
    );
}

#[test]
fn gpu_wave2d_node_effect_compiles_and_filters_gpu() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let mut plugins = PluginRegistry::new();
    plugins.enable("node-effect").expect("enable node-effect");

    let make_proj = |with_node_fx: bool| {
        let mut project = Project::new_cm3_default();
        project.sequence.resolution = [64, 64];
        project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        let mut layer = Layer::new(
            "Solid",
            LayerKind::PresetShape2D {
                preset: PresetShapeKind::Rectangle,
                shapes: vec![Shape2DNode {
                    id: Uuid::new_v4(),
                    name: "Rect".to_owned(),
                    kind: Shape2DKind::Group {
                        children: Vec::new(),
                    },
                    position: PropertyTrack::vec2([0.0, 0.0]),
                    rotation: PropertyTrack::number(0.0),
                    scale: PropertyTrack::vec2([0.5, 0.5]),
                    operations: vec![DrawOp::Fill {
                        color: PropertyTrack::constant(PropertyValue::ColorRgba([
                            0.8, 0.8, 0.8, 1.0,
                        ])),
                    }],
                    preserved: BTreeMap::new(),
                }],
            },
            [64, 64],
        );
        if with_node_fx {
            let graph = serde_json::json!({
                "version": 1,
                "nodes": [
                    { "id": "uv", "type": "uv" },
                    { "id": "samp", "type": "sample" },
                    { "id": "vor", "type": "voronoi", "params": { "mode": 2, "scale": 5.0 } },
                    { "id": "ramp", "type": "colorRamp", "params": {
                        "interpolation": 1,
                        "stops": [[0.0, 0.1, 0.0, 0.8, 1.0], [1.0, 1.0, 0.5, 0.0, 1.0]]
                    }},
                    { "id": "mix", "type": "mix", "params": { "mode": 0, "factor": 0.85 } },
                    { "id": "out", "type": "output" }
                ],
                "links": [
                    { "from": { "node": "uv", "port": "out" }, "to": { "node": "samp", "port": "uv" } },
                    { "from": { "node": "vor", "port": "out" }, "to": { "node": "ramp", "port": "factor" } },
                    { "from": { "node": "samp", "port": "color" }, "to": { "node": "mix", "port": "a" } },
                    { "from": { "node": "ramp", "port": "color" }, "to": { "node": "mix", "port": "b" } },
                    { "from": { "node": "mix", "port": "color" }, "to": { "node": "out", "port": "color" } },
                    { "from": { "node": "samp", "port": "alpha" }, "to": { "node": "out", "port": "alpha" } }
                ]
            });
            let mut params = BTreeMap::new();
            params.insert(
                "graph".to_owned(),
                PropertyTrack::constant(PropertyValue::Json(graph)),
            );
            layer.effects.push(EffectInstance {
                id: Uuid::new_v4(),
                name: "NodeFX".to_owned(),
                kind: EffectKind::Plugin {
                    plugin_id: "node-effect".to_owned(),
                    effect_id: "nodeeffect".to_owned(),
                },
                enabled: PropertyTrack::option(1),
                params,
                preserved: BTreeMap::new(),
            });
        }
        project.sequence.video_tracks = vec![zc_project::VideoTrack {
            id: Uuid::new_v4(),
            name: "NodeTrack".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![VideoClip {
                id: Uuid::new_v4(),
                name: "NodeClip".to_owned(),
                start: 0.0,
                length: 60.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                layer,
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        }];
        project
    };

    let px_base = engine
        .render_frame_rgba8(&make_proj(false), &plugins, 0.0, 64, 64)
        .expect("render baseline for nodeeffect");
    let px_fx = engine
        .render_frame_rgba8(&make_proj(true), &plugins, 0.0, 64, 64)
        .expect("render nodeeffect");
    let diff_px = px_base
        .as_chunks::<4>()
        .0
        .iter()
        .zip(px_fx.as_chunks::<4>().0.iter())
        .filter(|(a, b)| {
            a[0].abs_diff(b[0]) > 20 || a[1].abs_diff(b[1]) > 20 || a[2].abs_diff(b[2]) > 20
        })
        .count();
    assert!(
        diff_px > 100,
        "nodeeffect graph must compile to WGSL and produce non-identity filtered GPU output (diff_px={diff_px})"
    );
}
