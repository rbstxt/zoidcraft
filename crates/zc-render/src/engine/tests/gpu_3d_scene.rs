use super::*;

#[test]
fn gpu_wave_b_camera_plus_film_gate_offset_and_dof() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let mut plugins = PluginRegistry::new();
    plugins.enable("camera-plus").expect("enable camera-plus");

    // 1. Camera+ filmGate & filmOffsetX shift rendered projection horizontally.
    let mut cam_base = Object3D::new_perspective_camera();
    cam_base.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);
    if let Object3DKind::Camera(ref mut c) = cam_base.kind {
        c.focal_length = PropertyTrack::number(50.0);
        c.film_gate = 36.0;
    }
    let mut box_obj = Object3D::new_shape_3d(
        "CenterBox",
        Shape3DKind::Box {
            size: [12.0, 12.0, 12.0],
        },
    );
    box_obj.material = Some(MaterialSpec::SingleColor {
        color: PropertyTrack::color([1.0, 0.2, 0.2]),
        side: 2,
    });

    let proj_center = make_test_scene_project(vec![cam_base.clone(), box_obj.clone()], [96, 96]);
    let px_center = engine
        .render_frame_rgba8(&proj_center, &plugins, 0.0, 96, 96)
        .expect("render centered camera+");

    let mut cam_shifted = cam_base.clone();
    if let Object3DKind::Camera(ref mut c) = cam_shifted.kind {
        c.film_offset_x = PropertyTrack::number(35.0);
    }
    let proj_shifted = make_test_scene_project(vec![cam_shifted, box_obj.clone()], [96, 96]);
    let px_shifted = engine
        .render_frame_rgba8(&proj_shifted, &plugins, 0.0, 96, 96)
        .expect("render shifted camera+");

    let centroid_x = |pixels: &[u8]| -> f64 {
        let mut sum_x = 0.0;
        let mut weight = 0.0;
        for y in 0..96usize {
            for x in 0..96usize {
                let r = f64::from(pixels[(y * 96 + x) * 4]);
                if r > 40.0 {
                    sum_x += (x as f64) * r;
                    weight += r;
                }
            }
        }
        assert!(weight > 0.0, "expected visible red box pixels");
        sum_x / weight
    };
    let cx_base = centroid_x(&px_center);
    let cx_shift = centroid_x(&px_shifted);
    assert!(
        (cx_shift - cx_base).abs() > 8.0,
        "filmOffsetX must visibly shift rendered projection (base={cx_base:.2}, shifted={cx_shift:.2})"
    );

    // 2. Camera+ 3-pass Bokeh DOF blurs out-of-focus geometry (with enabled_track).
    let mut cam_sharp = Object3D::new_perspective_camera();
    cam_sharp.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);
    let mut rect = Object3D::new_shape_3d("FarQuad", Shape3DKind::Rectangle { size: [26.0, 26.0] });
    rect.position = PropertyTrack::vec3([0.0, 0.0, -40.0]);
    rect.material = Some(MaterialSpec::SingleColor {
        color: PropertyTrack::color([1.0, 1.0, 1.0]),
        side: 2,
    });

    let proj_sharp = make_test_scene_project(vec![cam_sharp.clone(), rect.clone()], [96, 96]);
    let px_sharp = engine
        .render_frame_rgba8(&proj_sharp, &plugins, 0.0, 96, 96)
        .expect("render sharp quad");

    let mut cam_dof = cam_sharp;
    if let Object3DKind::Camera(ref mut c) = cam_dof.kind {
        c.dof = Some(zc_project::CameraDofParams {
            enabled: false,
            enabled_track: Some(PropertyTrack::option(1)),
            focus_distance: PropertyTrack::number(15.0),
            aperture: PropertyTrack::number(16.0),
            focus_area_width: PropertyTrack::number(0.0),
            near_blur: PropertyTrack::number(100.0),
            far_blur: PropertyTrack::number(300.0),
        });
    }
    let proj_dof = make_test_scene_project(vec![cam_dof, rect], [96, 96]);
    let px_dof = engine
        .render_frame_rgba8(&proj_dof, &plugins, 0.0, 96, 96)
        .expect("render dof quad");

    let mid_alpha_or_gray_count = |pixels: &[u8]| -> usize {
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[0] > 10 && p[0] < 245)
            .count()
    };
    let sharp_soft = mid_alpha_or_gray_count(&px_sharp);
    let dof_soft = mid_alpha_or_gray_count(&px_dof);
    assert!(
        dof_soft > sharp_soft + 30,
        "DOF must blur out-of-focus quad edges (sharp soft px={sharp_soft}, dof soft px={dof_soft})"
    );
}

#[test]
fn gpu_wave_b_pbr_plus_extra_maps_wrap_subtractive_and_scene_backdrop() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let mut plugins = PluginRegistry::new();
    plugins
        .enable("material-plus")
        .expect("enable material-plus");

    // 1. PbrPlus emissive_map + emissive_intensity + alpha_map + alpha_test + SceneBackdrop.
    // Build a 4x4 alpha map where left half is transparent (0) and right half is opaque (255).
    let mut alpha_rgba = vec![0u8; 4 * 4 * 4];
    for y in 0..4usize {
        for x in 0..4usize {
            let v = if x < 2 { 0u8 } else { 255u8 };
            let idx = (y * 4 + x) * 4;
            alpha_rgba[idx..idx + 4].copy_from_slice(&[v, v, v, 255]);
        }
    }
    // Build a 2x2 bright green emissive map.
    let emissive_rgba = [0u8, 255, 0, 255].repeat(4);

    let mut cam = Object3D::new_perspective_camera();
    cam.position = PropertyTrack::vec3([0.0, 0.0, 40.0]);
    let mut quad = Object3D::new_shape_3d("PbrQuad", Shape3DKind::Rectangle { size: [40.0, 40.0] });
    let mut proj = make_test_scene_project(vec![cam.clone()], [64, 64]);
    let alpha_id = proj.insert_binary_asset(
        zc_project::AssetKind::Image,
        "alpha.png",
        encode_test_png(4, 4, alpha_rgba),
    );
    let emissive_id = proj.insert_binary_asset(
        zc_project::AssetKind::Image,
        "emissive.png",
        encode_test_png(2, 2, emissive_rgba),
    );

    let pbr = zc_project::PbrPlusMaterial {
        base_color: PropertyTrack::color([0.0, 0.0, 0.0]),
        emissive: PropertyTrack::color([0.0, 1.0, 0.0]),
        emissive_intensity: PropertyTrack::number(1.0),
        emissive_map: Some(emissive_id),
        alpha_map: Some(alpha_id),
        alpha_test: 0.5,
        side: 2,
        ..Default::default()
    };
    quad.material = Some(MaterialSpec::PbrPlus(Box::new(pbr)));
    let scene_layer_id = proj.sequence.video_tracks[0].clips[0].layer.id;
    proj.sequence.scene_backdrops.insert(
        scene_layer_id,
        zc_project::SceneBackdrop {
            mode: 1,
            color: [0.0, 0.0, 1.0],
            opacity: PropertyTrack::number(1.0),
        },
    );
    proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Scene3D {
        objects: vec![cam.clone(), quad],
        shadow_mode: ProjectCompatMode::Current,
    };

    let px = engine
        .render_frame_rgba8(&proj, &plugins, 0.0, 64, 64)
        .expect("render pbrplus + backdrop");
    let left = &px[(32 * 64 + 18) * 4..(32 * 64 + 18) * 4 + 4];
    let right = &px[(32 * 64 + 46) * 4..(32 * 64 + 46) * 4 + 4];
    assert!(
        left[2] > 200 && left[1] < 30,
        "alpha_test must discard left half to reveal blue SceneBackdrop, got {left:?}"
    );
    assert!(
        right[1] > 200 && right[2] < 30,
        "right half must show green PbrPlus emissive_map, got {right:?}"
    );

    // 2. TextureWrap::ClampToEdge vs TextureWrap::Repeat with uv_repeat = [2, 1].
    let mut stripe_rgba = vec![0u8; 4 * 2 * 4];
    for y in 0..2usize {
        for x in 0..4usize {
            let idx = (y * 4 + x) * 4;
            if x < 2 {
                stripe_rgba[idx..idx + 4].copy_from_slice(&[255, 0, 0, 255]);
            } else {
                stripe_rgba[idx..idx + 4].copy_from_slice(&[0, 0, 255, 255]);
            }
        }
    }
    let mut wrap_proj = make_test_scene_project(vec![cam.clone()], [64, 64]);
    let stripe_id = wrap_proj.insert_binary_asset(
        zc_project::AssetKind::Image,
        "stripe.png",
        encode_test_png(4, 2, stripe_rgba),
    );
    let make_wrap_quad = |wrap: u32| {
        let mut q =
            Object3D::new_shape_3d("WrapQuad", Shape3DKind::Rectangle { size: [40.0, 40.0] });
        q.material = Some(MaterialSpec::Texture(Box::new(
            zc_project::TextureMaterial {
                texture: Some(stripe_id.clone()),
                wrap,
                repeat: PropertyTrack::vec2([2.0, 1.0]),
                side: 2,
                ..Default::default()
            },
        )));
        q
    };
    wrap_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Scene3D {
        objects: vec![cam.clone(), make_wrap_quad(0)], // ClampToEdge
        shadow_mode: ProjectCompatMode::Current,
    };
    let px_clamp = engine
        .render_frame_rgba8(&wrap_proj, &plugins, 0.0, 64, 64)
        .expect("render clamp wrap");
    wrap_proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Scene3D {
        objects: vec![cam.clone(), make_wrap_quad(1)], // Repeat
        shadow_mode: ProjectCompatMode::Current,
    };
    let px_repeat = engine
        .render_frame_rgba8(&wrap_proj, &plugins, 0.0, 64, 64)
        .expect("render repeat wrap");
    // At ~62% across the quad (u_tiled ~ 1.25): Clamp stays blue (edge), Repeat wraps to red!
    let sample_idx = (32 * 64 + 38) * 4;
    assert!(
        px_clamp[sample_idx + 2] > 180 && px_repeat[sample_idx] > 180,
        "TextureWrap must distinguish Clamp (blue {:?}) vs Repeat (red {:?})",
        &px_clamp[sample_idx..sample_idx + 4],
        &px_repeat[sample_idx..sample_idx + 4]
    );

    // 3. MaterialBlend::Subtractive (blending == 3) subtracts foreground color from background quad.
    let mut bg_quad =
        Object3D::new_shape_3d("WhiteBg", Shape3DKind::Rectangle { size: [40.0, 40.0] });
    bg_quad.position = PropertyTrack::vec3([0.0, 0.0, -2.0]);
    bg_quad.material = Some(MaterialSpec::SingleColor {
        color: PropertyTrack::color([1.0, 1.0, 1.0]),
        side: 2,
    });
    let mut sub_quad =
        Object3D::new_shape_3d("SubFg", Shape3DKind::Rectangle { size: [24.0, 24.0] });
    sub_quad.position = PropertyTrack::vec3([0.0, 0.0, 0.0]);
    sub_quad.material = Some(MaterialSpec::PbrPlus(Box::new(
        zc_project::PbrPlusMaterial {
            base_color: PropertyTrack::color([0.0, 0.0, 0.0]),
            emissive: PropertyTrack::color([1.0, 0.0, 0.0]),
            emissive_intensity: PropertyTrack::number(1.0),
            transparent: true,
            blending: 3, // Subtractive
            side: 2,
            ..Default::default()
        },
    )));
    let sub_proj = make_test_scene_project(vec![cam, bg_quad, sub_quad], [64, 64]);
    let px_sub = engine
        .render_frame_rgba8(&sub_proj, &plugins, 0.0, 64, 64)
        .expect("render subtractive blend");
    let center_sub = &px_sub[(32 * 64 + 32) * 4..(32 * 64 + 32) * 4 + 4];
    assert!(
        center_sub[0] < 40 && center_sub[1] > 200 && center_sub[2] > 200,
        "Subtractive red over white must yield cyan, got {center_sub:?}"
    );
}

#[test]
fn gpu_wave_b_text3d_extended_options_and_expresstext_plus() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let plugins = PluginRegistry::new(); // text-plus & expresstext-plus enabled by default

    let mut cam = Object3D::new_perspective_camera();
    cam.position = PropertyTrack::vec3([0.0, 0.0, 70.0]);

    let make_text = |spacing: f64, bevel: bool, bevel_side: u32, bevel_detail: u32| {
        let mut t = Object3D::new_text_3d("HI");
        if let Object3DKind::Text3D(ref mut d) = t.kind {
            d.size = [20.0, 4.0];
            d.spacing = spacing;
            d.bevel = bevel;
            d.bevel_thickness = 1.2;
            d.bevel_size = 1.5;
            d.bevel_side = bevel_side;
            d.bevel_detail = bevel_detail;
            d.bevel_profile = 1;
            d.bevel_tension = 0.8;
        }
        t.material = Some(MaterialSpec::SingleColor {
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            side: 2,
        });
        t
    };

    let span_x = |pixels: &[u8]| -> usize {
        let mut min_x = 128usize;
        let mut max_x = 0usize;
        for y in 0..64usize {
            for x in 0..128usize {
                if pixels[(y * 128 + x) * 4] > 40 {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                }
            }
        }
        max_x.saturating_sub(min_x)
    };

    let proj_tight =
        make_test_scene_project(vec![cam.clone(), make_text(0.0, false, 0, 3)], [128, 64]);
    let px_tight = engine
        .render_frame_rgba8(&proj_tight, &plugins, 0.0, 128, 64)
        .expect("render tight text3d");

    let proj_wide =
        make_test_scene_project(vec![cam.clone(), make_text(18.0, true, 0, 6)], [128, 64]);
    let px_wide = engine
        .render_frame_rgba8(&proj_wide, &plugins, 0.0, 128, 64)
        .expect("render spaced + beveled text3d");

    let w_tight = span_x(&px_tight);
    let w_wide = span_x(&px_wide);
    assert!(
        w_wide > w_tight + 15,
        "Text3DExtendedOptions spacing+bevel must widen rendered text (tight={w_tight}, wide={w_wide})"
    );

    // ExpressText+ per-character animation lifts letters upward via Plain effector.
    let mut et_text = make_text(0.0, false, 0, 3);
    et_text.preserved.insert(
        "expressTextEffectors".to_owned(),
        serde_json::json!([
            {
                "kind": "plain",
                "plainPosition": [0.0, 18.0, 0.0]
            }
        ]),
    );
    let proj_et = make_test_scene_project(vec![cam, et_text], [128, 64]);
    let px_et = engine
        .render_frame_rgba8(&proj_et, &plugins, 0.0, 128, 64)
        .expect("render expresstext+");

    let centroid_y = |pixels: &[u8]| -> f64 {
        let mut sum_y = 0.0;
        let mut cnt = 0.0;
        for y in 0..64usize {
            for x in 0..128usize {
                if pixels[(y * 128 + x) * 4] > 40 {
                    sum_y += y as f64;
                    cnt += 1.0;
                }
            }
        }
        assert!(cnt > 0.0, "expected visible text pixels");
        sum_y / cnt
    };
    let cy_base = centroid_y(&px_tight);
    let cy_et = centroid_y(&px_et);
    assert!(
        cy_et < cy_base - 8.0,
        "ExpressText+ Plain +Y must shift letters upward in screen space (base={cy_base:.2}, et={cy_et:.2})"
    );
}

#[test]
fn gpu_wave_b_effector_plus_and_repeater_end_to_end() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let plugins = PluginRegistry::new(); // repeater is enabled by default

    let mut cam = Object3D::new_perspective_camera();
    cam.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);

    let lit_bar = || {
        let mut b = Object3D::new_shape_3d(
            "Bar",
            Shape3DKind::Box {
                size: [36.0, 6.0, 6.0],
            },
        );
        b.material = Some(MaterialSpec::SingleColor {
            color: PropertyTrack::color([1.0, 1.0, 1.0]),
            side: 2,
        });
        b
    };

    let lit_pixel_count = |pixels: &[u8]| -> usize {
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[0] > 40)
            .count()
    };

    // 1. Warp deformer Group bends horizontal bar upward into +Y.
    let proj_plain = make_test_scene_project(vec![cam.clone(), lit_bar()], [96, 96]);
    let px_plain = engine
        .render_frame_rgba8(&proj_plain, &plugins, 0.0, 96, 96)
        .expect("render unwarped bar");

    let mut warp_grp = Object3D::new(
        "Warp",
        Object3DKind::Group {
            reflection_visibility: 0,
            custom_properties: Vec::new(),
            children: vec![lit_bar()],
        },
    );
    warp_grp.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:repeater/warp"),
    );
    warp_grp
        .preserved
        .insert("property.strength".to_owned(), serde_json::json!(2.4));
    warp_grp
        .preserved
        .insert("property.subdivision".to_owned(), serde_json::json!(16));
    let proj_warp = make_test_scene_project(vec![cam.clone(), warp_grp], [96, 96]);
    let px_warp = engine
        .render_frame_rgba8(&proj_warp, &plugins, 0.0, 96, 96)
        .expect("render warped bar");
    let warp_diff: usize = px_plain
        .iter()
        .zip(px_warp.iter())
        .filter(|(a, b)| a.abs_diff(**b) > 30)
        .count();
    assert!(
        warp_diff > 200,
        "Warp deformer must visibly bend child mesh (diff bytes={warp_diff})"
    );

    // 2. Twist deformer Group twists child bar around X.
    let mut twist_grp = Object3D::new(
        "Twist",
        Object3DKind::Group {
            reflection_visibility: 0,
            custom_properties: Vec::new(),
            children: vec![lit_bar()],
        },
    );
    twist_grp.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:repeater/twist"),
    );
    twist_grp
        .preserved
        .insert("property.angle".to_owned(), serde_json::json!(2.0));
    twist_grp
        .preserved
        .insert("property.axis".to_owned(), serde_json::json!(0));
    twist_grp
        .preserved
        .insert("property.subdivision".to_owned(), serde_json::json!(16));
    let proj_twist = make_test_scene_project(vec![cam.clone(), twist_grp], [96, 96]);
    let px_twist = engine
        .render_frame_rgba8(&proj_twist, &plugins, 0.0, 96, 96)
        .expect("render twisted bar");
    assert!(lit_pixel_count(&px_twist) > 50);

    // 3. Voronoi Fracture Group shatters and explodes a box outward.
    let mut small_box = Object3D::new_shape_3d(
        "SmallBox",
        Shape3DKind::Box {
            size: [12.0, 12.0, 12.0],
        },
    );
    small_box.material = Some(MaterialSpec::SingleColor {
        color: PropertyTrack::color([1.0, 1.0, 1.0]),
        side: 2,
    });
    let proj_box = make_test_scene_project(vec![cam.clone(), small_box.clone()], [96, 96]);
    let px_box = engine
        .render_frame_rgba8(&proj_box, &plugins, 0.0, 96, 96)
        .expect("render intact box");

    let mut voronoi_grp = Object3D::new(
        "Voronoi",
        Object3DKind::Group {
            reflection_visibility: 0,
            custom_properties: Vec::new(),
            children: vec![small_box.clone()],
        },
    );
    voronoi_grp.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:repeater/voronoi-fracture"),
    );
    voronoi_grp
        .preserved
        .insert("property.cells".to_owned(), serde_json::json!(8));
    voronoi_grp
        .preserved
        .insert("property.distance".to_owned(), serde_json::json!(10.0));
    let proj_voronoi = make_test_scene_project(vec![cam.clone(), voronoi_grp], [96, 96]);
    let px_voronoi = engine
        .render_frame_rgba8(&proj_voronoi, &plugins, 0.0, 96, 96)
        .expect("render fractured box");
    let voronoi_diff: usize = px_box
        .iter()
        .zip(px_voronoi.iter())
        .filter(|(a, b)| a.abs_diff(**b) > 30)
        .count();
    assert!(
        voronoi_diff > 200,
        "Voronoi Fracture must visibly shatter/explode child mesh (diff bytes={voronoi_diff})"
    );

    // 4. Repeater (Linear & Step) clones a small cube across the viewport.
    let mut tiny_cube = Object3D::new_shape_3d(
        "Cube",
        Shape3DKind::Box {
            size: [6.0, 6.0, 6.0],
        },
    );
    tiny_cube.position = PropertyTrack::vec3([-24.0, 0.0, 0.0]);
    tiny_cube.material = Some(MaterialSpec::SingleColor {
        color: PropertyTrack::color([1.0, 1.0, 1.0]),
        side: 2,
    });
    let proj_one = make_test_scene_project(vec![cam.clone(), tiny_cube.clone()], [96, 96]);
    let px_one = engine
        .render_frame_rgba8(&proj_one, &plugins, 0.0, 96, 96)
        .expect("render single cube");

    let mut linear_rep = Object3D::new(
        "LinearRep",
        Object3DKind::Group {
            reflection_visibility: 0,
            custom_properties: Vec::new(),
            children: vec![tiny_cube.clone()],
        },
    );
    linear_rep.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:repeater/linear-repeater"),
    );
    linear_rep.preserved.insert(
        "repeaterProperties".to_owned(),
        serde_json::json!({
            "count": 5,
            "positionEnd": [48.0, 0.0, 0.0],
            "rotationEnd": [0.0, 0.0, 0.0]
        }),
    );
    let proj_lin = make_test_scene_project(vec![cam.clone(), linear_rep], [96, 96]);
    let px_lin = engine
        .render_frame_rgba8(&proj_lin, &plugins, 0.0, 96, 96)
        .expect("render linear repeater");

    let mut step_rep = Object3D::new(
        "StepRep",
        Object3DKind::Group {
            reflection_visibility: 0,
            custom_properties: Vec::new(),
            children: vec![tiny_cube],
        },
    );
    step_rep.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:repeater/repeater"),
    );
    step_rep.preserved.insert(
        "repeaterProperties".to_owned(),
        serde_json::json!({
            "count": 4,
            "positionStep": [14.0, 0.0, 0.0]
        }),
    );
    let proj_step = make_test_scene_project(vec![cam, step_rep], [96, 96]);
    let px_step = engine
        .render_frame_rgba8(&proj_step, &plugins, 0.0, 96, 96)
        .expect("render step repeater");

    let one_px = lit_pixel_count(&px_one);
    let lin_px = lit_pixel_count(&px_lin);
    let step_px = lit_pixel_count(&px_step);
    assert!(
        lin_px >= one_px * 3 && step_px >= one_px * 3,
        "Linear & Step Repeaters must clone cube across screen (single={one_px}, linear={lin_px}, step={step_px})"
    );
}

#[test]
fn gpu_c5_camera_plus_priority_default_film_gate_and_dof_gating() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };

    // 1. Verify pz_import populates `zoidiumPluginObjectType = "zoidium:camera-plus/camera"`
    //    on a real `.pz` archive containing a Camera+ object after a native CM3 camera.
    let mut cam_plus_export = Object3D::new_perspective_camera();
    cam_plus_export.name = "CamPlus".to_owned();
    cam_plus_export.position = PropertyTrack::vec3([0.0, 0.0, 40.0]);
    cam_plus_export.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:camera-plus/camera"),
    );
    let mut native_export = Object3D::new_perspective_camera();
    native_export.name = "NativeCam".to_owned();
    native_export.position = PropertyTrack::vec3([0.0, 0.0, 200.0]);
    let roundtrip_src = make_test_scene_project(vec![native_export, cam_plus_export], [128, 128]);
    let pz_bytes = zc_project::export_pz_to_bytes(&roundtrip_src).expect("export pz");
    let imported = zc_project::import_pz_from_bytes(&pz_bytes).expect("import pz");
    let LayerKind::Scene3D { ref objects, .. } = imported.project.sequence.video_tracks[0].clips[0]
        .layer
        .kind
    else {
        panic!("expected Scene3D layer");
    };
    assert_eq!(
        objects[1]
            .preserved
            .get("zoidiumPluginObjectType")
            .and_then(serde_json::Value::as_str),
        Some("zoidium:camera-plus/camera"),
        "pz_import must tag Camera+ with zoidiumPluginObjectType"
    );

    // 2. Active Camera+ placed AFTER an active native CM3 camera must override the native camera,
    //    while a second active Camera+ after the first active Camera+ must NOT override the first.
    let mut plugins_on = PluginRegistry::new();
    plugins_on
        .enable("camera-plus")
        .expect("enable camera-plus");
    let mut plugins_off = PluginRegistry::new();
    plugins_off.disable_all_optional();

    let mut cube = Object3D::new_shape_3d(
        "Cube",
        Shape3DKind::Box {
            size: [14.0, 14.0, 14.0],
        },
    );
    cube.material = Some(MaterialSpec::SingleColor {
        color: PropertyTrack::color([1.0, 1.0, 1.0]),
        side: 2,
    });

    let mut native_far = Object3D::new_perspective_camera();
    native_far.position = PropertyTrack::vec3([0.0, 0.0, 200.0]);

    let mut cam_plus_close = Object3D::new_perspective_camera();
    cam_plus_close.position = PropertyTrack::vec3([0.0, 0.0, 35.0]);
    cam_plus_close.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:camera-plus/camera"),
    );

    let mut cam_plus_far = Object3D::new_perspective_camera();
    cam_plus_far.position = PropertyTrack::vec3([0.0, 0.0, 260.0]);
    cam_plus_far.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:camera-plus/camera"),
    );

    let count_lit =
        |px: &[u8]| -> usize { px.as_chunks::<4>().0.iter().filter(|p| p[0] > 200).count() };

    let proj = make_test_scene_project(
        vec![
            native_far,
            cam_plus_close.clone(),
            cam_plus_far,
            cube.clone(),
        ],
        [96, 96],
    );
    let px_on = engine
        .render_frame_rgba8(&proj, &plugins_on, 0.0, 96, 96)
        .expect("render with camera-plus enabled");
    let px_off = engine
        .render_frame_rgba8(&proj, &plugins_off, 0.0, 96, 96)
        .expect("render with camera-plus disabled");
    let lit_on = count_lit(&px_on);
    let lit_off = count_lit(&px_off);
    assert!(
        lit_on > lit_off * 4,
        "Active Camera+ (first active) must override active native camera when plugin is enabled (on={lit_on}, off={lit_off})"
    );

    // 3. Default Camera+ (filmGate=36, focalLength=35) must render at exact 32.27° on 16:9
    //    and 54.43° on 1:1 (NOT fixed 60° native FOV!).
    let controls = RefCell::new(BTreeMap::new());
    let parent_props = BTreeMap::new();
    let stack = RefCell::new(HashSet::new());
    let env_on = test_eval_env(
        &proj,
        &plugins_on,
        &controls,
        &parent_props,
        &stack,
        0.0,
        0.0,
        30.0,
    );
    let Object3DKind::Camera(ref default_cam_plus_data) = cam_plus_close.kind else {
        panic!("expected camera");
    };
    let p_16_9 = build_camera_params_from_node(
        default_cam_plus_data,
        &cam_plus_close,
        Mat4::IDENTITY,
        Vec3::new(0.0, 0.0, 35.0),
        Quat::IDENTITY,
        Vec3::ONE,
        GeomEulerOrder::XYZ,
        &env_on,
        [1920.0, 1080.0],
    );
    let fov_16_9 = 2.0 * (1.0 / p_16_9.proj[1][1]).atan().to_degrees();
    assert!(
        (fov_16_9 - 32.269).abs() < 0.05,
        "Default Camera+ (36/35) on 16:9 must have ~32.27° vertical FOV, got {fov_16_9:.3}°"
    );
    let p_1_1 = build_camera_params_from_node(
        default_cam_plus_data,
        &cam_plus_close,
        Mat4::IDENTITY,
        Vec3::new(0.0, 0.0, 35.0),
        Quat::IDENTITY,
        Vec3::ONE,
        GeomEulerOrder::XYZ,
        &env_on,
        [512.0, 512.0],
    );
    let fov_1_1 = 2.0 * (1.0 / p_1_1.proj[1][1]).atan().to_degrees();
    assert!(
        (fov_1_1 - 54.432).abs() < 0.05,
        "Default Camera+ (36/35) on 1:1 must have ~54.43° vertical FOV, got {fov_1_1:.3}°"
    );

    // GPU check on 1:1 canvas: default Camera+ (54.43° with camera-plus enabled) shows cube
    // larger than native 60° camera (with camera-plus disabled), while standard Camera and
    // Camera+ render identically when camera-plus is enabled.
    let mut native_60 = Object3D::new_perspective_camera();
    native_60.position = PropertyTrack::vec3([0.0, 0.0, 35.0]);
    let proj_nat_1_1 = make_test_scene_project(vec![native_60, cube.clone()], [96, 96]);
    let proj_cp_1_1 = make_test_scene_project(vec![cam_plus_close, cube], [96, 96]);
    let lit_nat_1_1 = count_lit(
        &engine
            .render_frame_rgba8(&proj_nat_1_1, &plugins_off, 0.0, 96, 96)
            .expect("native 1:1"),
    );
    let lit_cp_1_1 = count_lit(
        &engine
            .render_frame_rgba8(&proj_cp_1_1, &plugins_on, 0.0, 96, 96)
            .expect("camera+ 1:1"),
    );
    assert!(
        lit_cp_1_1 > lit_nat_1_1 + 40,
        "Default Camera+ on 1:1 (54.43°) must render larger than native 60° (cp={lit_cp_1_1}, nat={lit_nat_1_1})"
    );

    // 4. DOF must be gated on `camera-plus` plugin enablement.
    let dof_cam = zc_project::CameraData {
        dof: Some(zc_project::CameraDofParams {
            enabled: true,
            enabled_track: None,
            aperture: PropertyTrack::number(12.0),
            focus_distance: PropertyTrack::number(80.0),
            focus_area_width: PropertyTrack::number(5.0),
            near_blur: PropertyTrack::number(100.0),
            far_blur: PropertyTrack::number(100.0),
        }),
        ..Default::default()
    };
    assert!(
        eval_camera_dof(&dof_cam, &env_on).is_some(),
        "eval_camera_dof must return Some when camera-plus plugin is enabled"
    );
    let env_off = test_eval_env(
        &proj,
        &plugins_off,
        &controls,
        &parent_props,
        &stack,
        0.0,
        0.0,
        30.0,
    );
    assert!(
        eval_camera_dof(&dof_cam, &env_off).is_none(),
        "eval_camera_dof must return None when camera-plus plugin is disabled"
    );
}

#[test]
fn gpu_regression_camera_plus_continuous_focal_length_parity_and_plugin_gating() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };

    let mut plugins_on = PluginRegistry::new();
    plugins_on
        .enable("camera-plus")
        .expect("enable camera-plus");
    let mut plugins_off = PluginRegistry::new();
    plugins_off.disable_all_optional();

    let mut cube = Object3D::new_shape_3d(
        "LensCube",
        Shape3DKind::Box {
            size: [10.0, 10.0, 10.0],
        },
    );
    cube.material = Some(MaterialSpec::SingleColor {
        color: PropertyTrack::color([1.0, 1.0, 1.0]),
        side: 2,
    });
    let count_lit =
        |px: &[u8]| -> usize { px.as_chunks::<4>().0.iter().filter(|p| p[0] > 200).count() };

    // 1. Monotonic zoom across 24 -> 30 -> 34.5 -> 35.0 -> 35.5 -> 50 -> 85mm on a standard
    //    Camera with `camera-plus` enabled (zero cliff at 35.0mm!).
    let focal_lengths = [24.0_f64, 30.0, 34.5, 35.0, 35.5, 50.0, 85.0];
    let mut prev_fov_deg = f32::INFINITY;
    let mut prev_lit = 0usize;
    let controls = RefCell::new(BTreeMap::new());
    let parent_props = BTreeMap::new();
    let stack = RefCell::new(HashSet::new());

    for fl in focal_lengths {
        let mut cam = Object3D::new_perspective_camera();
        cam.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);
        if let Object3DKind::Camera(ref mut c) = cam.kind {
            c.fov = PropertyTrack::number(60.0);
            c.focal_length = PropertyTrack::number(fl);
            c.film_gate = 36.0;
        }
        let proj = make_test_scene_project(vec![cam.clone(), cube.clone()], [128, 72]);
        let env_on = test_eval_env(
            &proj,
            &plugins_on,
            &controls,
            &parent_props,
            &stack,
            0.0,
            0.0,
            30.0,
        );
        let Object3DKind::Camera(ref cam_data) = cam.kind else {
            panic!("expected camera");
        };
        let params = build_camera_params_from_node(
            cam_data,
            &cam,
            Mat4::IDENTITY,
            Vec3::new(0.0, 0.0, 80.0),
            Quat::IDENTITY,
            Vec3::ONE,
            GeomEulerOrder::XYZ,
            &env_on,
            [128.0, 72.0],
        );
        let fov_deg = 2.0 * (1.0 / params.proj[1][1]).atan().to_degrees();
        assert!(
            fov_deg < prev_fov_deg,
            "vertical FOV must strictly decrease as focal_length increases (fl={fl}mm, fov={fov_deg:.3}°, prev={prev_fov_deg:.3}°)"
        );
        prev_fov_deg = fov_deg;

        let px = engine
            .render_frame_rgba8(&proj, &plugins_on, 0.0, 128, 72)
            .expect("render focal sweep");
        let lit = count_lit(&px);
        assert!(
            lit >= prev_lit && lit > 0,
            "GPU rendered cube coverage must increase monotonically across focal_length sweep (fl={fl}mm, lit={lit}, prev={prev_lit})"
        );
        prev_lit = lit;
    }

    // 2. Standard Camera and `zoidium:camera-plus/camera` with identical params (including
    //    focal_length=35.0mm) render byte-for-byte identically AND survive `.pz` round-trip
    //    with zero zoom shift!
    let mut std_cam = Object3D::new_perspective_camera();
    std_cam.position = PropertyTrack::vec3([0.0, 0.0, 60.0]);
    if let Object3DKind::Camera(ref mut c) = std_cam.kind {
        c.fov = PropertyTrack::number(60.0);
        c.focal_length = PropertyTrack::number(35.0);
        c.film_gate = 36.0;
        // Attach a Camera+ feature so pz_export upgrades it to `zoidium:camera-plus/camera`
        c.dof = Some(zc_project::CameraDofParams {
            enabled: false,
            enabled_track: None,
            aperture: PropertyTrack::number(2.8),
            focus_distance: PropertyTrack::number(60.0),
            focus_area_width: PropertyTrack::number(5.0),
            near_blur: PropertyTrack::number(100.0),
            far_blur: PropertyTrack::number(100.0),
        });
    }
    let mut cp_cam = std_cam.clone();
    cp_cam.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:camera-plus/camera"),
    );

    let proj_std = make_test_scene_project(vec![std_cam, cube.clone()], [128, 72]);
    let proj_cp = make_test_scene_project(vec![cp_cam, cube.clone()], [128, 72]);
    let px_std = engine
        .render_frame_rgba8(&proj_std, &plugins_on, 0.0, 128, 72)
        .expect("render standard camera");
    let px_cp = engine
        .render_frame_rgba8(&proj_cp, &plugins_on, 0.0, 128, 72)
        .expect("render camera-plus camera");
    assert_eq!(
        px_std, px_cp,
        "Standard Camera and zoidium:camera-plus/camera with identical params must render identically"
    );

    let pz_bytes = zc_project::export_pz_to_bytes(&proj_std).expect("export pz");
    let rt_proj = zc_project::import_pz_from_bytes(&pz_bytes)
        .expect("import pz")
        .project;
    let px_rt = engine
        .render_frame_rgba8(&rt_proj, &plugins_on, 0.0, 128, 72)
        .expect("render round-tripped pz");
    assert_eq!(
        px_std, px_rt,
        "export-pz -> import-pz round-trip must produce zero zoom shift"
    );

    // 3. With `camera-plus` disabled, focal_length=85mm / film_gate / film_offset / zoom
    //    must NOT override native fov=60°.
    let mut native_clean = Object3D::new_perspective_camera();
    native_clean.position = PropertyTrack::vec3([0.0, 0.0, 60.0]);
    if let Object3DKind::Camera(ref mut c) = native_clean.kind {
        c.fov = PropertyTrack::number(60.0);
    }
    let mut cam_85 = native_clean.clone();
    if let Object3DKind::Camera(ref mut c) = cam_85.kind {
        c.focal_length = PropertyTrack::number(85.0);
        c.film_gate = 24.89;
        c.film_offset_x = PropertyTrack::number(35.0);
        c.zoom = PropertyTrack::number(2.0);
    }

    let proj_clean = make_test_scene_project(vec![native_clean, cube.clone()], [128, 72]);
    let proj_85 = make_test_scene_project(vec![cam_85, cube], [128, 72]);
    let px_off_clean = engine
        .render_frame_rgba8(&proj_clean, &plugins_off, 0.0, 128, 72)
        .expect("render clean off");
    let px_off_85 = engine
        .render_frame_rgba8(&proj_85, &plugins_off, 0.0, 128, 72)
        .expect("render 85mm off");
    assert_eq!(
        px_off_clean, px_off_85,
        "With camera-plus disabled, focal_length=85mm must NOT override fov=60°"
    );
}

#[test]
fn gpu_wave2d_node_material_procedural_3d_render() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let plugins = PluginRegistry::new();

    let mat_graph = serde_json::json!({
        "nodes": [
            { "id": "n", "type": "noise", "params": { "scale": 6, "octaves": 4, "seed": 11, "contrast": 2.0 } },
            { "id": "ramp", "type": "colorRamp", "params": {
                "interpolation": 1,
                "stops": [[0.0, 0.05, 0.1, 0.95, 1.0], [1.0, 0.98, 0.45, 0.05, 1.0]]
            }},
            { "id": "out", "type": "pbrOutput" }
        ],
        "links": [
            { "from": "n", "fromPort": "out", "to": "ramp", "toPort": "factor" },
            { "from": "ramp", "fromPort": "out", "to": "out", "toPort": "color" }
        ]
    });
    let mut cube = Object3D::new_shape_3d(
        "NodeMatBox",
        Shape3DKind::Box {
            size: [48.0, 48.0, 48.0],
        },
    );
    cube.material = Some(MaterialSpec::NodeMaterial {
        schema_version: 1,
        resolution: 64,
        graph: mat_graph,
    });
    let mut amb_data = zc_project::LightData::for_kind(zc_project::LightKind::Ambient);
    amb_data.intensity = PropertyTrack::number(1.2);
    let amb = Object3D::new("Ambient", Object3DKind::Light(Box::new(amb_data)));
    let mut cam = Object3D::new_perspective_camera();
    cam.position = PropertyTrack::vec3([0.0, 0.0, 72.0]);
    let project = make_test_scene_project(vec![cam, amb, cube], [64, 64]);

    let px = engine
        .render_frame_rgba8(&project, &plugins, 0.0, 64, 64)
        .expect("render NodeMaterial 3D scene");
    let mut colored_non_white = 0usize;
    let mut distinct_buckets = HashSet::new();
    for p in px.as_chunks::<4>().0 {
        if p[0] > 15 || p[1] > 15 || p[2] > 15 {
            let chroma = p[0].max(p[1]).max(p[2]) - p[0].min(p[1]).min(p[2]);
            if chroma > 30 {
                colored_non_white += 1;
            }
            distinct_buckets.insert((p[0] / 24, p[1] / 24, p[2] / 24));
        }
    }
    assert!(
        colored_non_white > 150 && distinct_buckets.len() >= 3,
        "NodeMaterial must bake procedural maps and render non-white textured 3D pixels (colored={colored_non_white}, distinct={})",
        distinct_buckets.len()
    );
}

#[test]
fn gpu_wave2d_optical_flare_renders_additive_3d_flare() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let mut plugins = PluginRegistry::new();
    plugins
        .enable("optical-flares")
        .expect("enable optical-flares");

    let mut cam = Object3D::new_perspective_camera();
    cam.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);

    let mut flare_obj = Object3D::new(
        "SunFlare",
        Object3DKind::Group {
            reflection_visibility: 0,
            custom_properties: Vec::new(),
            children: Vec::new(),
        },
    );
    flare_obj.position = PropertyTrack::vec3([0.0, 0.0, 0.0]);
    flare_obj.preserved.insert(
        "zoidiumPluginObjectType".to_owned(),
        serde_json::json!("zoidium:optical-flares/optical-flare"),
    );
    flare_obj
        .preserved
        .insert("brightness".to_owned(), serde_json::json!(2.0));
    flare_obj
        .preserved
        .insert("scale".to_owned(), serde_json::json!(1.5));
    flare_obj
        .preserved
        .insert("anamorphic".to_owned(), serde_json::json!(0.9));
    flare_obj
        .preserved
        .insert("color".to_owned(), serde_json::json!([1.0, 0.75, 0.35]));

    let mut proj_dark = make_test_scene_project(vec![cam.clone()], [64, 64]);
    proj_dark.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
    let mut proj_flare = make_test_scene_project(vec![cam, flare_obj], [64, 64]);
    proj_flare.sequence.background_color = [0.0, 0.0, 0.0, 1.0];

    let px_dark = engine
        .render_frame_rgba8(&proj_dark, &plugins, 0.0, 64, 64)
        .expect("render dark scene");
    let px_flare = engine
        .render_frame_rgba8(&proj_flare, &plugins, 0.0, 64, 64)
        .expect("render optical flare scene");
    let dark_lit = count_non_black_rgb_pixels(&px_dark);
    let flare_lit = count_non_black_rgb_pixels(&px_flare);
    let center_idx = (32 * 64 + 32) * 4;
    assert!(
        flare_lit > dark_lit + 100 && px_flare[center_idx] > 180,
        "OpticalFlare must render bright additive flare over 3D scene (dark_lit={dark_lit}, flare_lit={flare_lit}, center_rgb={:?})",
        &px_flare[center_idx..center_idx + 3]
    );
}

#[test]
fn gpu_wave2d_particles_plus_sprite_preset_renders_without_image_asset() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let mut plugins = PluginRegistry::new();
    plugins
        .enable("particles-plus")
        .expect("enable particles-plus");

    let mut cam = Object3D::new_perspective_camera();
    cam.position = PropertyTrack::vec3([0.0, 0.0, 80.0]);

    let pdata = zc_project::ParticlesData {
        texture: None,
        sprite_preset: Some("star_5".to_owned()),
        number: 64,
        rate: 60.0,
        lifetime: 4.0,
        iradius: 18.0,
        ..zc_project::ParticlesData::default()
    };
    let particles_obj = Object3D::new("StarParticles", Object3DKind::Particles(Box::new(pdata)));

    let mut project = make_test_scene_project(vec![cam, particles_obj], [64, 64]);
    project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
    let px = engine
        .render_frame_rgba8(&project, &plugins, 15.0, 64, 64)
        .expect("render Particles+ sprite preset");
    let lit = count_non_black_rgb_pixels(&px);
    assert!(
        lit > 50,
        "Particles+ sprite preset (`star_5`) must render non-blank particles without an external image asset (lit={lit})"
    );
}

#[test]
fn gpu_node_material_constant_opacity_not_applied_twice() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };
    let plugins = PluginRegistry::new();

    let mat_graph = serde_json::json!({
        "nodes": [
            { "id": "col", "type": "color", "params": { "color": [1.0, 1.0, 1.0] } },
            { "id": "op", "type": "float", "params": { "value": 0.5 } },
            { "id": "out", "type": "pbrOutput" }
        ],
        "links": [
            { "from": "col", "fromPort": "out", "to": "out", "toPort": "emissive" },
            { "from": "op", "fromPort": "out", "to": "out", "toPort": "opacity" }
        ]
    });
    let spec = MaterialSpec::NodeMaterial {
        schema_version: 1,
        resolution: 32,
        graph: mat_graph.clone(),
    };

    let mut cube = Object3D::new_shape_3d(
        "HalfOpacityBox",
        Shape3DKind::Box {
            size: [60.0, 60.0, 60.0],
        },
    );
    cube.material = Some(spec.clone());
    let mut cam = Object3D::new_perspective_camera();
    cam.position = PropertyTrack::vec3([0.0, 0.0, 72.0]);
    let mut project = make_test_scene_project(vec![cam, cube], [64, 64]);
    project.sequence.background_color = [0.0, 0.0, 0.0, 1.0];

    let controls = RefCell::new(BTreeMap::new());
    let parent_props = BTreeMap::new();
    let expr_stack = RefCell::new(HashSet::new());
    let env = EvalEnv {
        frame: 0.0,
        abs_frame: 0.0,
        time: 0.0,
        fps: 30.0,
        sequence_resolution: [64, 64],
        project: &project,
        sequence: &project.sequence,
        plugins: &plugins,
        controls: &controls,
        parent_props: &parent_props,
        active_clip_id: None,
        active_layer_id: None,
        active_object_id: None,
        active_effect_id: None,
        active_property_id: None,
        expr_stack: &expr_stack,
    };
    let evaluated = engine.evaluate_material(Some(&spec), &env);
    assert!(
        evaluated.alpha_map.is_some(),
        "wired NodeMaterial opacity should upload alpha_map"
    );
    assert!(
        (evaluated.params.opacity - 1.0).abs() < 1e-6,
        "when alpha_map is bound, params.opacity must be 1.0 to avoid squaring opacity (got {})",
        evaluated.params.opacity
    );
    assert_eq!(evaluated.params.blend, MaterialBlend::Normal);

    let px = engine
        .render_frame_rgba8(&project, &plugins, 0.0, 64, 64)
        .expect("render half-opacity NodeMaterial");
    let center = (32 * 64 + 32) * 4;
    let r = px[center];
    assert!(
        (110..=145).contains(&r),
        "constant 0.5 opacity emissive white box over black must render ~128 (0.5), not ~64 (0.25); got center RGBA={:?}",
        &px[center..center + 4]
    );
}
