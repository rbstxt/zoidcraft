use super::*;

#[test]
fn gpu_editing_camera_skips_clip_effects_and_font_changes_render_distinct_glyphs() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let plugins = PluginRegistry::new();

    // 1. Editing Camera MUST bypass clip effects (e.g. RGB Shift) so the editor grid & view stay clean.
    let mut text_obj = Object3D::new_text_3d("ZOID");
    if let Object3DKind::Text3D(ref mut t) = text_obj.kind {
        t.font = Some("Geist-Regular.ttf".to_owned());
    }
    let mut cam = Object3D::new_perspective_camera();
    cam.position = PropertyTrack::vec3([0.0, 0.0, 90.0]);
    let mut project = make_test_scene_project(vec![cam, text_obj], [160, 90]);
    let scene_clip_id = project.sequence.video_tracks[0].clips[0].id;

    let editor_cam = CameraParams::perspective_look_at(
        [0.0, 25.0, 80.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        60.0,
        160.0 / 90.0,
        0.1,
        5000.0,
    );
    let edit_ov = RenderOverrides {
        editing_camera_clip_id: Some(scene_clip_id),
        editing_camera: Some(editor_cam),
        selected_object_ids: Vec::new(),
        is_scene_camera_override: false,
    };

    let edit_no_fx = engine
        .render_frame_rgba8_with_options(
            &project,
            &plugins,
            0.0,
            160,
            90,
            &RenderFrameOptions {
                transparent_background: false,
                overrides: Some(edit_ov.clone()),
            },
        )
        .expect("edit camera without fx");
    let preview_no_fx = engine
        .render_frame_rgba8(&project, &plugins, 0.0, 160, 90)
        .expect("normal preview without fx");

    // Add strong RGB Shift effect on the 3D scene clip.
    let mut rgb = zc_project::create_effect_by_id("rgbshift").expect("rgbshift effect");
    rgb.params
        .insert("amount".to_owned(), PropertyTrack::number(40.0));
    project.sequence.video_tracks[0].clips[0]
        .layer
        .effects
        .push(rgb);
    engine.invalidate_caches();

    let edit_with_fx = engine
        .render_frame_rgba8_with_options(
            &project,
            &plugins,
            0.0,
            160,
            90,
            &RenderFrameOptions {
                transparent_background: false,
                overrides: Some(edit_ov),
            },
        )
        .expect("edit camera with fx");
    let preview_with_fx = engine
        .render_frame_rgba8(&project, &plugins, 0.0, 160, 90)
        .expect("normal preview with fx");

    assert_eq!(
        edit_no_fx, edit_with_fx,
        "Editing Camera view must bypass clip effects like RGB Shift"
    );
    assert_ne!(
        preview_no_fx, preview_with_fx,
        "Normal preview (Editing Camera OFF) must still apply RGB Shift"
    );

    // 2. Switching built-in fonts on Text3D & Text2D must produce visibly distinct pixels.
    project.sequence.video_tracks[0].clips[0]
        .layer
        .effects
        .clear();
    engine.invalidate_caches();
    let text3d_geist = engine
        .render_frame_rgba8(&project, &plugins, 0.0, 160, 90)
        .expect("text3d geist");
    if let LayerKind::Scene3D {
        ref mut objects, ..
    } = project.sequence.video_tracks[0].clips[0].layer.kind
        && let Object3DKind::Text3D(ref mut t) = objects[1].kind
    {
        t.font = Some("SourceCodePro-Regular.ttf".to_owned());
    }
    engine.invalidate_caches();
    let text3d_scp = engine
        .render_frame_rgba8(&project, &plugins, 0.0, 160, 90)
        .expect("text3d source code pro");
    let diff_3d = text3d_geist
        .iter()
        .zip(text3d_scp.iter())
        .filter(|(a, b)| a.abs_diff(**b) > 8)
        .count();
    assert!(
        diff_3d > 80,
        "Text3D font change (Geist -> SourceCodePro) must produce visibly different glyphs, diff={diff_3d}"
    );

    // Text2D font change check.
    let mut p2d = Project::new_cm3_default();
    p2d.sequence.resolution = [160, 90];
    p2d.sequence.video_tracks[0].clips.clear();
    p2d.sequence.video_tracks[0]
        .clips
        .push(zc_project::VideoClip {
            id: Uuid::new_v4(),
            name: "Text2D".to_owned(),
            start: 0.0,
            length: 30.0,
            link: None,
            media: None,
            time_curve: PropertyTrack::linear_time_curve(30.0, 30),
            layer: Layer::new(
                "Text2D",
                LayerKind::Text2D {
                    text: "AgW1".to_owned(),
                    font: Some("Geist-Regular.ttf".to_owned()),
                    size: 42,
                    shapes: Vec::new(),
                },
                [160, 90],
            ),
            preserved: BTreeMap::new(),
        });
    let text2d_geist = engine
        .render_frame_rgba8(&p2d, &plugins, 0.0, 160, 90)
        .expect("text2d geist");
    assert!(
        count_non_black_rgb_pixels(&text2d_geist) > 50,
        "Text2D baseline render must produce visible glyphs"
    );
    if let LayerKind::Text2D { ref mut font, .. } = p2d.sequence.video_tracks[0].clips[0].layer.kind
    {
        *font = Some("SourceCodePro-Regular.ttf".to_owned());
    } else {
        panic!("expected Text2D clip at clips[0]");
    }
    engine.invalidate_caches();
    let text2d_scp = engine
        .render_frame_rgba8(&p2d, &plugins, 0.0, 160, 90)
        .expect("text2d source code pro");
    let diff_2d = text2d_geist
        .iter()
        .zip(text2d_scp.iter())
        .filter(|(a, b)| a.abs_diff(**b) > 8)
        .count();
    assert!(
        diff_2d > 40,
        "Text2D font change (Geist -> SourceCodePro) must produce visibly different glyphs, diff={diff_2d}"
    );
}

#[test]
fn gpu_qa_fixes_zero_light_pbr_particles_presets_particular_repeaters_and_cjk() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let mut plugins = PluginRegistry::new();
    let _ = plugins.enable("particles-plus");
    let _ = plugins.enable("trapcode-suite");
    let _ = plugins.enable("repeater");

    // 1. A non-emissive PBR sphere is black until the scene has a light.
    let mut cam = Object3D::new_perspective_camera();
    cam.position = PropertyTrack::vec3([0.0, 0.0, 45.0]);
    let mut sphere = Object3D::new_shape_3d(
        "PbrSphere",
        Shape3DKind::Sphere {
            radius: 12.0,
            width_segments: 24,
            height_segments: 16,
        },
    );
    sphere.material = Some(MaterialSpec::PbrPlus(Box::new(
        zc_project::PbrPlusMaterial {
            base_color: PropertyTrack::color([0.9, 0.3, 0.2]),
            emissive: PropertyTrack::color([0.0, 0.0, 0.0]),
            roughness: PropertyTrack::number(0.5),
            metalness: PropertyTrack::number(0.1),
            reflection: false,
            ..Default::default()
        },
    )));
    let mut pbr_proj = make_test_scene_project(vec![cam.clone(), sphere], [64, 64]);
    let transparent = RenderFrameOptions {
        transparent_background: true,
        ..Default::default()
    };
    let pbr_px = engine
        .render_frame_rgba8_with_options(&pbr_proj, &plugins, 0.0, 64, 64, &transparent)
        .expect("zero-light pbr sphere");
    let center = (32 * 64 + 32) * 4;
    assert_eq!(&pbr_px[..4], &[0, 0, 0, 0], "transparent background");
    assert_eq!(&pbr_px[center..center + 4], &[0, 0, 0, 255]);
    assert!(
        pbr_px
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p[..3] == [0, 0, 0])
    );

    let mut light = Object3D::new_spot_light();
    light.position = PropertyTrack::vec3([0.0, 0.0, 45.0]);
    if let Object3DKind::Light(data) = &mut light.kind {
        data.kind = ProjectLightKind::Directional;
        data.intensity = PropertyTrack::number(1.0);
        data.cast_shadow = false;
    }
    let LayerKind::Scene3D { objects, .. } =
        &mut pbr_proj.sequence.video_tracks[0].clips[0].layer.kind
    else {
        panic!("expected a 3D scene");
    };
    objects.push(light);
    engine.invalidate_caches();
    let lit_px = engine
        .render_frame_rgba8_with_options(&pbr_proj, &plugins, 0.0, 64, 64, &transparent)
        .expect("explicitly lit pbr sphere");
    assert!(
        lit_px[center] > 60,
        "directional light should illuminate the sphere, got {:?}",
        &lit_px[center..center + 4]
    );

    // 2. Particles+ sprite presets (`star_4` vs `circle_soft` vs `spark`) produce distinct GPU pixels!
    let render_preset = |engine: &mut ProjectRenderEngine, preset: &str| {
        let mut pcam = Object3D::new_perspective_camera();
        pcam.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);
        let pdata = zc_project::ParticlesData {
            texture: None,
            sprite_preset: Some(preset.to_owned()),
            number: 1,
            rate: 60.0,
            lifetime: 10.0,
            pspread: [0.0, 0.0, 0.0],
            iradius: 0.0,
            vspread: [0.0, 0.0, 0.0],
            ..zc_project::ParticlesData::default()
        };
        let pobj = Object3D::new("P", Object3DKind::Particles(Box::new(pdata)));
        let mut proj = make_test_scene_project(vec![pcam, pobj], [64, 64]);
        proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        engine.invalidate_caches();
        engine
            .render_frame_rgba8(&proj, &plugins, 15.0, 64, 64)
            .expect("render preset")
    };
    let px_star4 = render_preset(&mut engine, "star_4");
    let px_soft = render_preset(&mut engine, "circle_soft");
    let px_spark = render_preset(&mut engine, "spark");
    let diff_count = |a: &[u8], b: &[u8]| -> usize {
        a.as_chunks::<4>()
            .0
            .iter()
            .zip(b.as_chunks::<4>().0.iter())
            .filter(|(p, q)| p[0].abs_diff(q[0]) > 12)
            .count()
    };
    assert!(
        diff_count(&px_star4, &px_soft) > 30,
        "star_4 and circle_soft must render visibly distinct particle sprites"
    );
    assert!(
        diff_count(&px_spark, &px_soft) > 30,
        "spark and circle_soft must render visibly distinct particle sprites"
    );
    assert!(
        diff_count(&px_star4, &px_spark) > 30,
        "star_4 and spark must render visibly distinct particle sprites"
    );

    // 3. Trapcode Particular default: colorful cyan-blue particles, NOT pure-white screen blowout!
    let mut tc_obj = Object3D::new(
        "Particular",
        Object3DKind::Group {
            children: Vec::new(),
            custom_properties: Vec::new(),
            reflection_visibility: 0,
        },
    );
    tc_obj.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:trapcode-suite/particular"),
    );
    tc_obj.preserved.insert(
        "properties".to_owned(),
        serde_json::json!({
            "count": 64,
            "particleSize": 3.0,
            "particleColor": [0.2, 0.85, 1.0],
            "opacity": 1.0
        }),
    );
    let mut tc_cam = Object3D::new_perspective_camera();
    tc_cam.position = PropertyTrack::vec3([0.0, 0.0, 100.0]);
    let mut tc_proj = make_test_scene_project(vec![tc_cam, tc_obj], [96, 96]);
    tc_proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
    engine.invalidate_caches();
    let tc_px = engine
        .render_frame_rgba8(&tc_proj, &plugins, 0.0, 96, 96)
        .expect("render particular");
    let total_px = 96 * 96;
    let lit_tc = count_non_black_rgb_pixels(&tc_px);
    let blown_white = tc_px
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[0] > 245 && p[1] > 245 && p[2] > 245)
        .count();
    assert!(
        lit_tc > 20 && blown_white < total_px / 4,
        "Particular must render crisp colorful particles without full-frame white blowout (lit={lit_tc}, blown_white={blown_white}/{total_px})"
    );

    // 4. Radial & Grid Repeaters expand a child Box into multiple distinct spatial regions!
    let make_box_child = || {
        let mut b = Object3D::new_shape_3d(
            "ChildBox",
            Shape3DKind::Box {
                size: [4.0, 4.0, 4.0],
            },
        );
        b.material = Some(MaterialSpec::SingleColor {
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            side: 2,
        });
        b
    };
    let mut rcam = Object3D::new_perspective_camera();
    rcam.position = PropertyTrack::vec3([0.0, 0.0, 90.0]);
    let single_proj = make_test_scene_project(vec![rcam.clone(), make_box_child()], [96, 96]);
    engine.invalidate_caches();
    let px_single = engine
        .render_frame_rgba8(&single_proj, &plugins, 0.0, 96, 96)
        .expect("single box");
    let lit_single = count_non_black_rgb_pixels(&px_single);

    let mut radial_rep = Object3D::new(
        "RadialRep",
        Object3DKind::Group {
            children: vec![make_box_child()],
            custom_properties: Vec::new(),
            reflection_visibility: 0,
        },
    );
    radial_rep.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:repeater/radial"),
    );
    radial_rep.preserved.insert(
        "repeaterProperties".to_owned(),
        serde_json::json!({
            "count": 8,
            "radius": 20.0,
            "axis": 2,
            "alignToCircle": 1
        }),
    );
    let radial_proj = make_test_scene_project(vec![rcam.clone(), radial_rep], [96, 96]);
    engine.invalidate_caches();
    let px_radial = engine
        .render_frame_rgba8(&radial_proj, &plugins, 0.0, 96, 96)
        .expect("radial repeater");
    let lit_radial = count_non_black_rgb_pixels(&px_radial);
    assert!(
        lit_radial > lit_single * 3,
        "zoidium:repeater/radial must instantiate ring copies (single={lit_single}, radial={lit_radial})"
    );

    let mut grid_rep = Object3D::new(
        "GridRep",
        Object3DKind::Group {
            children: vec![make_box_child()],
            custom_properties: Vec::new(),
            reflection_visibility: 0,
        },
    );
    grid_rep.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:repeater/grid"),
    );
    grid_rep.preserved.insert(
        "repeaterProperties".to_owned(),
        serde_json::json!({
            "countX": 3,
            "countY": 3,
            "countZ": 1,
            "spacing": [12.0, 12.0, 0.0]
        }),
    );
    let grid_proj = make_test_scene_project(vec![rcam, grid_rep], [96, 96]);
    engine.invalidate_caches();
    let px_grid = engine
        .render_frame_rgba8(&grid_proj, &plugins, 0.0, 96, 96)
        .expect("grid repeater");
    let lit_grid = count_non_black_rgb_pixels(&px_grid);
    assert!(
        lit_grid > lit_single * 3,
        "zoidium:repeater/grid must instantiate 3x3 grid copies (single={lit_single}, grid={lit_grid})"
    );

    // 5. Automatic OS CJK fallback font renders real Japanese glyphs (`テスト`) distinct from `???`!
    if resolve_system_cjk_fallback_font().is_some() {
        let mut tcam = Object3D::new_perspective_camera();
        tcam.position = PropertyTrack::vec3([0.0, 0.0, 65.0]);
        let make_t3d = |txt: &str| {
            let mut t = Object3D::new_text_3d(txt);
            if let Object3DKind::Text3D(ref mut d) = t.kind {
                d.font = Some("Geist-Regular.ttf".to_owned());
                d.size = [24.0, 2.0];
            }
            t.material = Some(MaterialSpec::SingleColor {
                color: PropertyTrack::color([1.0, 1.0, 1.0]),
                side: 2,
            });
            t
        };
        let proj_jp = make_test_scene_project(vec![tcam.clone(), make_t3d("テスト")], [256, 128]);
        let proj_q = make_test_scene_project(vec![tcam, make_t3d("???")], [256, 128]);
        engine.invalidate_caches();
        let px_jp = engine
            .render_frame_rgba8(&proj_jp, &plugins, 0.0, 256, 128)
            .expect("render jp");
        engine.invalidate_caches();
        let px_q = engine
            .render_frame_rgba8(&proj_q, &plugins, 0.0, 256, 128)
            .expect("render ???");
        let lit_jp = count_non_black_rgb_pixels(&px_jp);
        let diff_jp_q = diff_count(&px_jp, &px_q);
        assert!(
            lit_jp > 80,
            "CJK text must render visible glyphs (lit_jp={lit_jp})"
        );
        assert!(
            diff_jp_q > 80,
            "CJK Japanese text (`テスト`) must use OS CJK fallback glyphs rather than rendering `???` (diff={diff_jp_q}, lit_jp={lit_jp})"
        );
    }
}

#[test]
fn gpu_regression_c5_time_driven_fx_and_timeremap_defaults() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(e) => e,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to init headless engine: {err}"),
    };
    let mut plugins = PluginRegistry::new();
    let _ = plugins.enable("native-fx");

    let make_fx_proj = |fx: EffectInstance, animated_pos: bool| {
        let mut project = Project::new_cm3_default();
        project.sequence.resolution = [128, 72];
        project.sequence.fps = 30;
        project.sequence.length_frames = 90.0;
        let mut shape_layer = Layer::new(
            "Square",
            LayerKind::PresetShape2D {
                preset: PresetShapeKind::Rectangle,
                shapes: vec![Shape2DNode {
                    id: Uuid::new_v4(),
                    name: "Box".to_owned(),
                    kind: Shape2DKind::Group {
                        children: Vec::new(),
                    },
                    position: PropertyTrack::vec2([0.0, 0.0]),
                    rotation: PropertyTrack::number(0.0),
                    scale: PropertyTrack::vec2([0.4, 0.5]),
                    operations: vec![DrawOp::Fill {
                        color: PropertyTrack::constant(PropertyValue::ColorRgba([
                            0.1, 0.9, 1.0, 1.0,
                        ])),
                    }],
                    preserved: BTreeMap::new(),
                }],
            },
            [128, 72],
        );
        if animated_pos {
            let mut pos = PropertyTrack::vec2([-36.0, 0.0]);
            pos.animated = true;
            pos.keyframes = vec![
                zc_project::PropertyKeyframe::linear(0.0, PropertyValue::Vec2([-36.0, 0.0])),
                zc_project::PropertyKeyframe::linear(30.0, PropertyValue::Vec2([36.0, 0.0])),
            ];
            shape_layer.position = pos;
        }
        shape_layer.effects.push(fx);
        project.sequence.video_tracks[0].clips = vec![VideoClip {
            id: Uuid::new_v4(),
            name: "Clip".to_owned(),
            start: 0.0,
            length: 90.0,
            link: None,
            media: None,
            // Deliberately unkeyframed zero placeholder (as produced by QA4-10 projects).
            time_curve: PropertyTrack::number(0.0),
            layer: shape_layer,
            preserved: BTreeMap::new(),
        }];
        project
    };

    // 1. timecode, radiowaves, ccpixelpolly must advance cleanly across frames (not frozen at t=0).
    for (fx_id, frames) in [
        ("timecode", [15.0_f64, 45.0, 67.0]),
        ("radiowaves", [6.0, 15.0, 67.0]),
        ("ccpixelpolly", [15.0, 45.0, 67.0]),
    ] {
        let fx = zc_project::create_zoidium_effect(fx_id).expect("create zoidium fx");
        let proj = make_fx_proj(fx, false);
        engine.invalidate_caches();
        let pa = engine
            .render_frame_rgba8(&proj, &plugins, frames[0], 128, 72)
            .expect("render fa");
        let pb = engine
            .render_frame_rgba8(&proj, &plugins, frames[1], 128, 72)
            .expect("render fb");
        let pc = engine
            .render_frame_rgba8(&proj, &plugins, frames[2], 128, 72)
            .expect("render fc");
        assert_ne!(
            pa, pb,
            "{fx_id} must NOT stay frozen between frame {} and frame {}",
            frames[0], frames[1]
        );
        assert_ne!(
            pb, pc,
            "{fx_id} must NOT stay frozen between frame {} and frame {}",
            frames[1], frames[2]
        );
    }

    // 2. pulse with default expr="time" must stay alive across t > 1 (f15 and f67).
    let pulse_fx = zc_project::create_standard_effect("pulse").expect("pulse");
    let pulse_proj = make_fx_proj(pulse_fx, false);
    let base_proj = make_fx_proj(
        EffectInstance {
            id: Uuid::new_v4(),
            name: "Disabled".to_owned(),
            enabled: PropertyTrack::option(0),
            kind: EffectKind::Standard("pulse".to_owned()),
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        },
        false,
    );
    engine.invalidate_caches();
    let base_px = engine
        .render_frame_rgba8(&base_proj, &plugins, 15.0, 128, 72)
        .expect("base");
    let pulse_f15 = engine
        .render_frame_rgba8(&pulse_proj, &plugins, 15.0, 128, 72)
        .expect("pulse f15");
    let pulse_f67 = engine
        .render_frame_rgba8(&pulse_proj, &plugins, 67.0, 128, 72)
        .expect("pulse f67");
    assert_ne!(
        pulse_f15, base_px,
        "pulse default expr=time must distort frame 15"
    );
    assert_ne!(
        pulse_f67, base_px,
        "pulse default expr=time must loop/wrap and distort frame 67 (t > 1)"
    );

    // 3. timeremap default (remap=1, mode=0, sourceTime=0, speed=1.0) must NOT jump f0/f15 to f30.
    let tr_fx = zc_project::create_zoidium_effect("timeremap").expect("timeremap");
    let tr_proj = make_fx_proj(tr_fx, true);
    let no_tr_proj = make_fx_proj(
        EffectInstance {
            id: Uuid::new_v4(),
            name: "NoTR".to_owned(),
            enabled: PropertyTrack::option(0),
            kind: EffectKind::Standard("exposure".to_owned()),
            params: BTreeMap::new(),
            preserved: BTreeMap::new(),
        },
        true,
    );
    for f in [0.0_f64, 15.0, 30.0] {
        engine.invalidate_caches();
        let got = engine
            .render_frame_rgba8(&tr_proj, &plugins, f, 128, 72)
            .expect("timeremap render");
        let want = engine
            .render_frame_rgba8(&no_tr_proj, &plugins, f, 128, 72)
            .expect("baseline render");
        assert_eq!(
            got, want,
            "timeremap default (speed=1.0, mode=0) must match baseline at frame {f}"
        );
    }
}

#[test]
fn gpu_qa6_metallic_ibl_glsl_3d_bump_ramp_matcap_generator_char_mat_nodemat_aspect_and_expresstext()
{
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let mut plugins = PluginRegistry::new();
    for p in [
        "material-plus",
        "geometry-plus",
        "repeater",
        "text-plus",
        "expresstext-plus",
        "shape-plus",
    ] {
        let _ = plugins.enable(p);
    }

    let mut cam = Object3D::new_perspective_camera();
    cam.position = PropertyTrack::vec3([0.0, 0.0, 42.0]);
    let make_key_light = || {
        let mut l = Object3D::new_spot_light();
        if let Object3DKind::Light(ref mut ld) = l.kind {
            ld.kind = ProjectLightKind::Directional;
            ld.color = PropertyTrack::color([1.0, 1.0, 1.0]);
            ld.intensity = PropertyTrack::number(1.2);
            ld.cast_shadow = false;
        }
        l.position = PropertyTrack::vec3([18.0, 22.0, 26.0]);
        l
    };

    // 1. High-metalness CustomPbr (metalness = 0.95 gold) & Brushed Metal NodeMaterial (0.92)
    // must render bright metallic luster under directional lighting (NOT 97% pitch black).
    let mut gold_sphere = Object3D::new_shape_3d(
        "GoldSphere",
        Shape3DKind::Sphere {
            radius: 11.0,
            width_segments: 24,
            height_segments: 16,
        },
    );
    gold_sphere.material = Some(MaterialSpec::CustomPbr(Box::new(
        zc_project::CustomPbrMaterial {
            color: PropertyTrack::color([1.0, 0.82, 0.25]),
            roughness: PropertyTrack::number(0.25),
            metalness: PropertyTrack::number(0.95),
            ..Default::default()
        },
    )));
    let gold_proj =
        make_test_scene_project(vec![cam.clone(), make_key_light(), gold_sphere], [96, 96]);
    let gold_px = engine
        .render_frame_rgba8(&gold_proj, &plugins, 0.0, 96, 96)
        .expect("render metallic gold");
    let gold_bright_count = gold_px
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[0] > 70 && p[1] > 45 && p[0] > p[2].saturating_add(20))
        .count();
    assert!(
        gold_bright_count > 400,
        "high-metalness gold CustomPbr must render lustrous gold reflection, got only {gold_bright_count} warm metallic pixels"
    );

    // 1b. High-metalness NodeMaterial ("Brushed Metal", metalness=0.92) also renders bright metallic IBL.
    let mut brushed_sphere = Object3D::new_shape_3d(
        "BrushedMetalSphere",
        Shape3DKind::Sphere {
            radius: 12.0,
            width_segments: 24,
            height_segments: 16,
        },
    );
    brushed_sphere.material = Some(MaterialSpec::NodeMaterial {
        schema_version: 1,
        resolution: 64,
        graph: serde_json::json!({
            "nodes": [
                { "id": "n1", "type": "voronoi", "params": { "scale": 10.0 } },
                { "id": "n2", "type": "fresnel", "params": { "ior": 1.8 } },
                { "id": "n3", "type": "float", "params": { "value": 0.92 } },
                { "id": "n4", "type": "mix", "params": { "a": [0.48, 0.56, 0.68], "b": [0.88, 0.92, 0.98] } },
                { "id": "n5", "type": "output" }
            ],
            "links": [
                { "from": "n2", "fromPort": "out", "to": "n4", "toPort": "factor" },
                { "from": "n4", "fromPort": "color", "to": "n5", "toPort": "color" },
                { "from": "n3", "fromPort": "out", "to": "n5", "toPort": "metalness" },
                { "from": "n1", "fromPort": "distance", "to": "n5", "toPort": "roughness" }
            ]
        }),
    });
    let brushed_proj = make_test_scene_project(
        vec![cam.clone(), make_key_light(), brushed_sphere],
        [96, 96],
    );
    let brushed_px = engine
        .render_frame_rgba8(&brushed_proj, &plugins, 0.0, 96, 96)
        .expect("render brushed metal NodeMaterial");
    let brushed_lit_count = brushed_px
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[0] > 55 && p[1] > 55 && p[2] > 55)
        .count();
    assert!(
        brushed_lit_count > 400,
        "high-metalness NodeMaterial (Brushed Metal) must not render pitch-black, got only {brushed_lit_count} lit metallic pixels"
    );

    // 2. 3D MaterialSpec::CustomShader with GLSL ES 1.00 fragment shader renders real GLSL color.
    let mut glsl_box = Object3D::new_shape_3d(
        "GlslBox",
        Shape3DKind::Box {
            size: [18.0, 18.0, 18.0],
        },
    );
    glsl_box.material = Some(MaterialSpec::CustomShader {
        frag_shader:
            "precision mediump float;\nuniform vec3 tint;\nvoid main() {\n    gl_FragColor = vec4(tint, 1.0);\n}\n"
                .to_owned(),
        custom_properties: vec![CustomProperty {
            name: "tint".to_owned(),
            prop_type: None,
            track: PropertyTrack::color([0.1, 0.9, 0.25]),
            metadata: BTreeMap::new(),
        }],
    });
    let glsl_proj = make_test_scene_project(vec![cam.clone(), glsl_box], [96, 96]);
    let glsl_px = engine
        .render_frame_rgba8(&glsl_proj, &plugins, 0.0, 96, 96)
        .expect("render 3D GLSL custom shader");
    let center_idx = (48 * 96 + 48) * 4;
    assert!(
        glsl_px[center_idx] < 50 && glsl_px[center_idx + 1] > 190 && glsl_px[center_idx + 2] < 90,
        "3D GLSL CustomShader must render green tint [~25, ~230, ~64], got {:?}",
        &glsl_px[center_idx..center_idx + 4]
    );

    // 3. PbrPlus BumpNoise (bumpStrength=1.0, bumpNoiseScale=8.0) renders smooth bump relief
    //    without blown-out white TV static (< 5% clipped white pixels).
    let mut bump_sphere = Object3D::new_shape_3d(
        "BumpSphere",
        Shape3DKind::Sphere {
            radius: 11.0,
            width_segments: 24,
            height_segments: 16,
        },
    );
    let mut bump_pbr = zc_project::PbrPlusMaterial {
        base_color: PropertyTrack::color([0.45, 0.55, 0.7]),
        roughness: PropertyTrack::number(0.4),
        metalness: PropertyTrack::number(0.1),
        ..Default::default()
    };
    bump_pbr
        .preserved
        .insert("bumpNoise".to_owned(), serde_json::json!(1));
    bump_pbr
        .preserved
        .insert("bumpStrength".to_owned(), serde_json::json!(1.0));
    bump_pbr
        .preserved
        .insert("bumpNoiseScale".to_owned(), serde_json::json!(8.0));
    bump_sphere.material = Some(MaterialSpec::PbrPlus(Box::new(bump_pbr)));
    let bump_proj =
        make_test_scene_project(vec![cam.clone(), make_key_light(), bump_sphere], [96, 96]);
    let bump_px = engine
        .render_frame_rgba8(&bump_proj, &plugins, 0.0, 96, 96)
        .expect("render bump sphere");
    let mut bump_lit = 0usize;
    let mut bump_white = 0usize;
    for p in bump_px.as_chunks::<4>().0 {
        if p[0] > 5 || p[1] > 5 || p[2] > 5 {
            bump_lit += 1;
            if p[0] > 245 && p[1] > 245 && p[2] > 245 {
                bump_white += 1;
            }
        }
    }
    assert!(bump_lit > 400, "bump sphere should be visible");
    assert!(
        bump_white * 20 < bump_lit,
        "PbrPlus BumpNoise must not blow out into white static ({bump_white}/{bump_lit} white pixels)"
    );

    // 4. PbrPlus ColorRamp (`rampEnabled: 1`, purple -> pink -> yellow) remaps lit shading!
    let mut ramp_sphere = Object3D::new_shape_3d(
        "RampSphere",
        Shape3DKind::Sphere {
            radius: 11.0,
            width_segments: 24,
            height_segments: 16,
        },
    );
    let mut ramp_pbr = zc_project::PbrPlusMaterial {
        base_color: PropertyTrack::color([1.0, 1.0, 1.0]),
        roughness: PropertyTrack::number(0.5),
        metalness: PropertyTrack::number(0.0),
        ..Default::default()
    };
    ramp_pbr
        .preserved
        .insert("rampEnabled".to_owned(), serde_json::json!(1));
    ramp_pbr.preserved.insert(
        "ramp".to_owned(),
        serde_json::json!({
            "stops": [
                { "position": 0.0, "color": [0.35, 0.05, 0.65] },
                { "position": 0.5, "color": [0.95, 0.25, 0.65] },
                { "position": 1.0, "color": [1.0, 0.95, 0.15] }
            ]
        }),
    );
    ramp_sphere.material = Some(MaterialSpec::PbrPlus(Box::new(ramp_pbr)));
    let ramp_proj =
        make_test_scene_project(vec![cam.clone(), make_key_light(), ramp_sphere], [96, 96]);
    let ramp_px = engine
        .render_frame_rgba8(&ramp_proj, &plugins, 0.0, 96, 96)
        .expect("render color ramp sphere");
    let has_purple_shadow = ramp_px
        .as_chunks::<4>()
        .0
        .iter()
        .any(|p| p[2] > 100 && p[2] > p[1].saturating_add(50));
    let has_yellow_highlight = ramp_px
        .as_chunks::<4>()
        .0
        .iter()
        .any(|p| p[0] > 200 && p[1] > 170 && p[2] < 120);
    assert!(
        has_purple_shadow && has_yellow_highlight,
        "PbrPlus ColorRamp must remap shadow->purple ({has_purple_shadow}) and highlight->yellow ({has_yellow_highlight})"
    );

    // 5. Matcap with texture: None renders smooth analytical studio clay shading (std_luma > 15).
    let mut mc_sphere = Object3D::new_shape_3d(
        "MatcapSphere",
        Shape3DKind::Sphere {
            radius: 11.0,
            width_segments: 24,
            height_segments: 16,
        },
    );
    mc_sphere.material = Some(MaterialSpec::Matcap(Box::new(zc_project::MatcapMaterial {
        texture: None,
        ..Default::default()
    })));
    let mc_proj = make_test_scene_project(vec![cam.clone(), mc_sphere], [96, 96]);
    let mc_px = engine
        .render_frame_rgba8(&mc_proj, &plugins, 0.0, 96, 96)
        .expect("render matcap fallback sphere");
    let lumas: Vec<f32> = mc_px
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[0] > 5 || p[1] > 5 || p[2] > 5)
        .map(|p| f32::from(p[0]))
        .collect();
    let mean_l = lumas.iter().sum::<f32>() / (lumas.len().max(1) as f32);
    let var_l = lumas
        .iter()
        .map(|&l| (l - mean_l) * (l - mean_l))
        .sum::<f32>()
        / (lumas.len().max(1) as f32);
    assert!(
        var_l.sqrt() > 15.0,
        "Matcap texture:None must render smooth clay sphere shading (std_luma={})",
        var_l.sqrt()
    );

    // 6. LayerKind::Generator solid fill color honors `childProperty.color` and layer masks.
    let mut gen_proj = Project::new_cm3_default();
    gen_proj.sequence.resolution = [64, 64];
    gen_proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
    let mut gen_layer = Layer::new("ColorGen", LayerKind::Generator, [64, 64]);
    gen_layer.preserved.insert(
        "childProperty.color".to_owned(),
        serde_json::json!([0.1, 0.85, 0.2]),
    );
    let gen_layer_id = gen_layer.id;
    gen_proj.sequence.layer_masks.insert(
        gen_layer_id,
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
    gen_proj.sequence.video_tracks[0].clips[0].layer = gen_layer;
    let gen_px = engine
        .render_frame_rgba8(&gen_proj, &plugins, 0.0, 64, 64)
        .expect("render generator layer with mask");
    let left_px = &gen_px[(32 * 64 + 10) * 4..(32 * 64 + 10) * 4 + 4];
    let right_px = &gen_px[(32 * 64 + 54) * 4..(32 * 64 + 54) * 4 + 4];
    assert!(
        left_px[1] > 180 && left_px[0] < 50 && right_px[1] < 15,
        "Generator layer must fill solid green inside mask ({left_px:?}) and respect mask outside ({right_px:?})"
    );

    // 7. NodeMaterial imported in `.pz` wrapper format with bound texture asset renders texture.
    let mut nm_proj = make_test_scene_project(vec![cam.clone(), make_key_light()], [64, 64]);
    let tex_rgba = [255_u8, 40, 200, 255].repeat(16); // 4x4 bright magenta PNG
    let asset_id = nm_proj.insert_binary_asset(
        zc_project::AssetKind::Image,
        "magenta.png",
        encode_test_png(4, 4, tex_rgba),
    );
    let inner_graph = serde_json::json!({
        "nodes": [
            { "id": "img", "type": "imageTexture", "params": { "assetId": asset_id } },
            { "id": "out", "type": "output" }
        ],
        "links": [
            { "from": "img", "fromPort": "out", "to": "out", "toPort": "color" }
        ]
    });
    let pz_wrapped = serde_json::json!({
        "type": "zoidium:material-plus/nodes",
        "properties": {
            "bakeResolution": 1,
            "graph": inner_graph.to_string()
        }
    });
    let mut nm_quad =
        Object3D::new_shape_3d("NmQuad", Shape3DKind::Rectangle { size: [30.0, 30.0] });
    nm_quad.material = Some(MaterialSpec::NodeMaterial {
        schema_version: 1,
        resolution: 64,
        graph: pz_wrapped,
    });
    if let LayerKind::Scene3D { objects, .. } =
        &mut nm_proj.sequence.video_tracks[0].clips[0].layer.kind
    {
        objects.push(nm_quad);
    }
    let nm_px = engine
        .render_frame_rgba8(&nm_proj, &plugins, 0.0, 64, 64)
        .expect("render .pz wrapped NodeMaterial");
    let nm_c = &nm_px[(32 * 64 + 32) * 4..(32 * 64 + 32) * 4 + 4];
    assert!(
        nm_c[0] > 140 && nm_c[2] > 100 && nm_c[1] < 80,
        ".pz NodeMaterial with bound texture must render magenta texture (not flat white), got {nm_c:?}"
    );

    // 8. 2D Image Layer aspect-ratio preservation on vertical 108x192 (9:16) sequence.
    let mut img_proj = Project::new_cm3_default();
    img_proj.sequence.resolution = [108, 192];
    img_proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
    let sq_rgba = [255_u8, 128, 0, 255].repeat(64 * 64);
    let sq_id = img_proj.insert_binary_asset(
        zc_project::AssetKind::Image,
        "square.png",
        encode_test_png(64, 64, sq_rgba),
    );
    let img_layer = Layer::new("SqImg", LayerKind::Image { image: Some(sq_id) }, [108, 192]);
    img_proj.sequence.video_tracks[0].clips[0].layer = img_layer;
    let img_px = engine
        .render_frame_rgba8(&img_proj, &plugins, 0.0, 108, 192)
        .expect("render 9:16 image");
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (108usize, 0usize, 192usize, 0usize);
    for y in 0..192 {
        for x in 0..108 {
            if img_px[(y * 108 + x) * 4] > 100 {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }
    let span_w = (max_x + 1).saturating_sub(min_x) as f32;
    let span_h = (max_y + 1).saturating_sub(min_y) as f32;
    assert!(
        (span_w - span_h).abs() <= 2.0 && span_w > 100.0,
        "square PNG on 108x192 must stay square (~108x108), got {span_w}x{span_h}"
    );

    // 9. Text+ character-material without characterNumbers applies to ALL glyphs (not just glyph 0).
    let mut text_obj = Object3D::new_text_3d("ABC");
    text_obj.material = Some(MaterialSpec::SingleColor {
        color: PropertyTrack::color([0.1, 0.1, 0.9]),
        side: 0,
    });
    let mut cm_grp = Object3D::new(
        "CharMat",
        Object3DKind::Group {
            reflection_visibility: 0,
            custom_properties: Vec::new(),
            children: vec![text_obj],
        },
    );
    cm_grp.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:text-plus/character-material"),
    );
    cm_grp.material_slots.insert(
        "front".to_owned(),
        MaterialSpec::SingleColor {
            color: PropertyTrack::color([1.0, 0.1, 0.1]),
            side: 0,
        },
    );
    let cm_proj = make_test_scene_project(vec![cam.clone(), cm_grp], [160, 80]);
    let cm_px = engine
        .render_frame_rgba8(&cm_proj, &plugins, 0.0, 160, 80)
        .expect("render character-material all glyphs");
    let right_half_red = (85..155)
        .flat_map(|x| (10..70).map(move |y| (x, y)))
        .filter(|&(x, y)| {
            let o = (y * 160 + x) * 4;
            cm_px[o] > 180 && cm_px[o + 2] < 60
        })
        .count();
    assert!(
        right_half_red > 20,
        "character-material with no filter must color right-side glyphs too, got {right_half_red} red pixels on right half"
    );

    // 10. ExpressText+ Random honors randomAnimationSpeed > 0 across frames,
    //     and Voronoi stays intact at frame 0 when distance == 0 && scatter == 0.
    let mut rand_text = Object3D::new_text_3d("HI");
    rand_text.material = Some(MaterialSpec::SingleColor {
        color: PropertyTrack::color([1.0, 1.0, 1.0]),
        side: 0,
    });
    rand_text.preserved.insert(
        "expressProperties".to_owned(),
        serde_json::json!({
            "effectors": [{
                "kind": "random",
                "properties": {
                    "randomAnimationSpeed": 120.0,
                    "randomPositionMin": [-8.0, -8.0, 0.0],
                    "randomPositionMax": [8.0, 8.0, 0.0]
                }
            }]
        }),
    );
    let rand_proj = make_test_scene_project(vec![cam.clone(), rand_text], [120, 80]);
    let rand_f0 = engine
        .render_frame_rgba8(&rand_proj, &plugins, 0.0, 120, 80)
        .expect("et random f0");
    let rand_f20 = engine
        .render_frame_rgba8(&rand_proj, &plugins, 20.0, 120, 80)
        .expect("et random f20");
    assert_ne!(
        rand_f0, rand_f20,
        "ExpressText+ Random with randomAnimationSpeed > 0 must animate across frames"
    );

    let mut base_v_text = Object3D::new_text_3d("HI");
    base_v_text.material = Some(MaterialSpec::SingleColor {
        color: PropertyTrack::color([1.0, 1.0, 1.0]),
        side: 0,
    });
    let mut vor_v_text = base_v_text.clone();
    vor_v_text.preserved.insert(
        "expressProperties".to_owned(),
        serde_json::json!({
            "effectors": [{
                "kind": "voronoi",
                "properties": {
                    "voronoiChunks": 8.0,
                    "voronoiDistance": 0.0,
                    "voronoiScatter": 0.0,
                    "voronoiSpin": [45.0, 45.0, 45.0]
                }
            }]
        }),
    );
    let base_v_proj = make_test_scene_project(vec![cam.clone(), base_v_text], [120, 80]);
    let vor_v_proj = make_test_scene_project(vec![cam, vor_v_text], [120, 80]);
    let base_v_px = engine
        .render_frame_rgba8(&base_v_proj, &plugins, 0.0, 120, 80)
        .expect("base voronoi text");
    let vor_v_px = engine
        .render_frame_rgba8(&vor_v_proj, &plugins, 0.0, 120, 80)
        .expect("voronoi distance=0 text");
    assert_eq!(
        base_v_px, vor_v_px,
        "ExpressText+ Voronoi must keep text 100% intact when distance == 0 && scatter == 0"
    );
}
