//! Effect registry, shader, uniform-layout, and GPU parity regression tests.

use std::collections::BTreeSet;
use std::mem::{align_of, offset_of};

use super::*;

const EXPECTED_44_STANDARD_IDS: [&str; 44] = [
    "anamorphicflare",
    "bloom",
    "boxblur",
    "brightnesscontrast",
    "chromakey",
    "colorcorrection",
    "colorgrading",
    "colorize",
    "coloroverlay",
    "crop",
    "cubelut",
    "directionalblur",
    "displacementmap",
    "duplicate",
    "edgedetection",
    "exposure",
    "fisheye",
    "flip",
    "fxaa",
    "gaussianblur",
    "glitch",
    "gradientoverlay",
    "grayscale",
    "hueshift",
    "lumakey",
    "mask",
    "mirror",
    "negative",
    "overlay",
    "pixelated",
    "pixelsort",
    "posterize",
    "pulse",
    "radialblur",
    "rgbshift",
    "scanlines",
    "shake",
    "sharpen",
    "shutter",
    "static",
    "swirl",
    "technicolor",
    "transform",
    "wavy",
];

const EXPECTED_12_PLUGIN_IDS: [&str; 12] = [
    "colorcurves",
    "radialblurspin",
    "dropshadow",
    "vhs",
    "jpegdamage",
    "datamosh",
    "tracery",
    "ascii",
    "magiclooks",
    "sglow",
    "smartmotionblur",
    "echo",
];

#[test]
fn effect_uniforms_layout_matches_wgsl_byte_for_byte() {
    assert_eq!(size_of::<EffectUniforms>(), 256);
    assert_eq!(size_of::<EffectUniforms>() % 16, 0);
    assert!(align_of::<EffectUniforms>() <= 16);

    assert_eq!(offset_of!(EffectUniforms, resolution), 0);
    assert_eq!(offset_of!(EffectUniforms, time), 8);
    assert_eq!(offset_of!(EffectUniforms, frame), 12);
    assert_eq!(offset_of!(EffectUniforms, mode), 16);
    assert_eq!(offset_of!(EffectUniforms, flags), 20);
    assert_eq!(offset_of!(EffectUniforms, pass), 24);
    assert_eq!(offset_of!(EffectUniforms, extra), 28);
    assert_eq!(offset_of!(EffectUniforms, p0), 32);
    assert_eq!(offset_of!(EffectUniforms, p1), 48);
    assert_eq!(offset_of!(EffectUniforms, p2), 64);
    assert_eq!(offset_of!(EffectUniforms, p3), 80);
    assert_eq!(offset_of!(EffectUniforms, p4), 96);
    assert_eq!(offset_of!(EffectUniforms, p5), 112);
    assert_eq!(offset_of!(EffectUniforms, p6), 128);
    assert_eq!(offset_of!(EffectUniforms, p7), 144);
    assert_eq!(offset_of!(EffectUniforms, c0), 160);
    assert_eq!(offset_of!(EffectUniforms, c1), 176);
    assert_eq!(offset_of!(EffectUniforms, c2), 192);
    assert_eq!(offset_of!(EffectUniforms, c3), 208);
    assert_eq!(offset_of!(EffectUniforms, m0), 224);
    assert_eq!(offset_of!(EffectUniforms, m1), 240);
}

#[test]
fn all_44_standard_and_12_plugin_effect_ids_present_and_unique() {
    assert_eq!(STANDARD_EFFECT_SHADERS.len(), 44);
    assert_eq!(PLUGIN_EFFECT_SHADERS.len(), 12);

    let std_set: BTreeSet<&str> = STANDARD_EFFECT_SHADERS.iter().map(|(id, _)| *id).collect();
    let expected_std: BTreeSet<&str> = EXPECTED_44_STANDARD_IDS.into_iter().collect();
    assert_eq!(std_set, expected_std);

    let plugin_set: BTreeSet<&str> = PLUGIN_EFFECT_SHADERS.iter().map(|(id, _)| *id).collect();
    let expected_plugin: BTreeSet<&str> = EXPECTED_12_PLUGIN_IDS.into_iter().collect();
    assert_eq!(plugin_set, expected_plugin);

    // Disjoint sets
    assert!(std_set.is_disjoint(&plugin_set));

    // Structural sanity check on every assembled WGSL shader
    for &(id, body) in STANDARD_EFFECT_SHADERS
        .iter()
        .chain(PLUGIN_EFFECT_SHADERS.iter())
    {
        let full = build_effect_wgsl(body);
        assert!(full.contains("fn vs_main("), "{id}: missing vs_main");
        assert!(full.contains("fn fs_main("), "{id}: missing fs_main");
        assert!(!body.contains("discard;"), "{id}: must never use discard");
    }
}

#[test]
fn neutral_auto_bypass_rules_match_spec_04() {
    let base = EffectParams::new([1920.0, 1080.0], 1.5, 45.0);

    // Global enabled == false bypasses every effect
    let disabled = base
        .clone()
        .with_param("enabled", EffectParamValue::Bool(false));
    for &(id, _) in &STANDARD_EFFECT_SHADERS {
        assert!(
            should_bypass(id, &disabled),
            "{id} should bypass when disabled"
        );
    }

    // delta == 0
    for id in ["boxblur", "directionalblur", "gaussianblur"] {
        assert!(!should_bypass(id, &base));
        let zero = base
            .clone()
            .with_param("delta", EffectParamValue::Float(0.0));
        assert!(should_bypass(id, &zero), "{id} must bypass at delta == 0");
    }

    // amount == 0
    for id in [
        "colorize",
        "edgedetection",
        "fisheye",
        "glitch",
        "hueshift",
        "rgbshift",
        "static",
        "wavy",
    ] {
        assert!(
            !should_bypass(id, &base),
            "{id} should be active at default"
        );
        let zero = base
            .clone()
            .with_param("amount", EffectParamValue::Float(0.0));
        assert!(should_bypass(id, &zero), "{id} must bypass at amount == 0");
    }

    // saturation == 1
    assert!(should_bypass("grayscale", &base));
    let desat = base
        .clone()
        .with_param("saturation", EffectParamValue::Float(0.5));
    assert!(!should_bypass("grayscale", &desat));

    // size == (1, 1)
    assert!(!should_bypass("pixelated", &base));
    let unit_px = base
        .clone()
        .with_param("size", EffectParamValue::Vec2([1.0, 1.0]));
    assert!(should_bypass("pixelated", &unit_px));

    // opacity == 0
    for id in ["scanlines", "shutter"] {
        assert!(!should_bypass(id, &base));
        let zero = base
            .clone()
            .with_param("opacity", EffectParamValue::Float(0.0));
        assert!(should_bypass(id, &zero));
    }

    // angle == 0
    assert!(!should_bypass("swirl", &base));
    let zero_ang = base
        .clone()
        .with_param("angle", EffectParamValue::Float(0.0));
    assert!(should_bypass("swirl", &zero_ang));

    // time % 1 == 0
    assert!(should_bypass("pulse", &base)); // default time == 0.0
    let int_t = base
        .clone()
        .with_param("time", EffectParamValue::Float(3.0));
    assert!(should_bypass("pulse", &int_t));
    let frac_t = base
        .clone()
        .with_param("time", EffectParamValue::Float(1.25));
    assert!(!should_bypass("pulse", &frac_t));

    // lut == null
    assert!(should_bypass("cubelut", &base));
    let with_lut = base.clone().with_param(
        "lut",
        EffectParamValue::Lut1D(vec![[0, 0, 0, 255], [255, 255, 255, 255]]),
    );
    assert!(!should_bypass("cubelut", &with_lut));
}

#[test]
fn transform_identity_homography_maps_ndc_to_ndc() {
    for cam_type in [0, 1] {
        let params = EffectParams::new([1920.0, 1080.0], 0.0, 0.0)
            .with_param("cameraType", EffectParamValue::Int(cam_type));
        let u = pack_effect_uniforms("transform", &params, false, false);
        assert_eq!(u.flags, 1);
        let ndc = [0.25_f32, -0.5, 1.0];
        let qx = u.p0[0] * ndc[0] + u.p0[1] * ndc[1] + u.p0[2] * ndc[2];
        let qy = u.p1[0] * ndc[0] + u.p1[1] * ndc[1] + u.p1[2] * ndc[2];
        let qw = u.p2[0] * ndc[0] + u.p2[1] * ndc[1] + u.p2[2] * ndc[2];
        assert!(qw > 0.0);
        assert!((qx / qw - ndc[0]).abs() < 1e-5);
        assert!((qy / qw - ndc[1]).abs() < 1e-5);
    }
}

#[test]
fn blend_mode_parser_handles_all_43_cm3_modes() {
    let names = [
        ("BLEND_SRC_OVER", 0),
        ("BLEND_MULTIPLY", 1),
        ("BLEND_SCREEN", 2),
        ("BLEND_OVERLAY", 3),
        ("BLEND_DARKEN", 4),
        ("BLEND_LIGHTEN", 5),
        ("BLEND_COLOR_DODGE", 6),
        ("BLEND_COLOR_BURN", 7),
        ("BLEND_HARD_LIGHT", 8),
        ("BLEND_SOFT_LIGHT", 9),
        ("BLEND_DIFFERENCE", 10),
        ("BLEND_EXCLUSION", 11),
        ("BLEND_HUE", 12),
        ("BLEND_SATURATION", 13),
        ("BLEND_COLOR", 14),
        ("BLEND_LUMINOSITY", 15),
        ("BLEND_ADD", 16),
        ("BLEND_SUBTRACT", 17),
        ("BLEND_LINEARDODGE", 18),
        ("BLEND_LINEARBURN", 19),
        ("BLEND_VIVIDLIGHT", 20),
        ("BLEND_LINEARLIGHT", 21),
        ("BLEND_PINLIGHT", 22),
        ("BLEND_HARDMIX", 23),
        ("BLEND_INVERT", 24),
        ("BLEND_INVERT_RGB", 25),
        ("BLEND_PLUS_DARKER", 26),
        ("BLEND_CONTRAST", 27),
        ("BLEND_INVERT_OVG", 28),
        ("BLEND_RED", 29),
        ("BLEND_GREEN", 30),
        ("BLEND_BLUE", 31),
        ("BLEND_ZERO", 32),
        ("BLEND_SRC", 33),
        ("BLEND_DST", 34),
        ("BLEND_DST_OVER", 35),
        ("BLEND_SRC_IN", 36),
        ("BLEND_DST_IN", 37),
        ("BLEND_SRC_OUT", 38),
        ("BLEND_DST_OUT", 39),
        ("BLEND_SRC_ATOP", 40),
        ("BLEND_DST_ATOP", 41),
        ("BLEND_XOR", 42),
    ];
    assert_eq!(names.len(), 43);
    for (name, expected) in names {
        let val = EffectParamValue::String(name.to_owned());
        assert_eq!(parse_blend_mode(Some(&val)), expected, "failed for {name}");
    }
}

#[test]
fn canonical_plugin_aliases_resolve_to_shaders() {
    assert_eq!(
        lookup_effect_shader("looks").map(|(id, _)| id),
        Some("magiclooks")
    );
    assert_eq!(
        lookup_effect_shader("sglow-hq-v2-preview").map(|(id, _)| id),
        Some("sglow")
    );
    assert_eq!(
        lookup_effect_shader("smart-motion-blur-lite").map(|(id, _)| id),
        Some("smartmotionblur")
    );
    assert_eq!(
        lookup_effect_shader("echo-legacy").map(|(id, _)| id),
        Some("echo")
    );
    // radialblurspin must NOT bypass at density == 0
    let rbs_zero = EffectParams::new([64.0, 64.0], 0.0, 0.0)
        .with_param("density", EffectParamValue::Float(0.0));
    assert!(!should_bypass("radialblurspin", &rbs_zero));
}

#[test]
fn validate_all_56_wgsl_shaders_in_headless_wgpu() {
    let ctx = match crate::GpuContext::new_headless() {
        Ok(ctx) => ctx,
        Err(crate::RenderError::NoAdapter(msg)) if !crate::require_gpu_in_tests() => {
            eprintln!("skipping GPU shader validation test (no adapter): {msg}");
            return;
        }
        Err(err) => panic!("failed to initialize headless GpuContext: {err}"),
    };
    let device = &ctx.device;
    let queue = &ctx.queue;

    let mut registry = EffectRegistry::new(device, queue, wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(registry.compiled_pipeline_count(), 0);

    for &(id, _) in STANDARD_EFFECT_SHADERS
        .iter()
        .chain(PLUGIN_EFFECT_SHADERS.iter())
    {
        registry
            .ensure_pipeline(device, id, false)
            .unwrap_or_else(|e| panic!("failed to compile WGSL shader `{id}`: {e}"));
    }
    assert_eq!(registry.compiled_pipeline_count(), 56);

    // Verify custom WGSL LRU cache stays bounded at MAX_CUSTOM_PIPELINES and evicts cleanly.
    let dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("zc-test-dummy"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let dummy_view = dummy_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let out_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("zc-test-out"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let params = EffectParams::new([4.0, 4.0], 0.0, 0.0);
    let uniforms = EffectUniforms::from_params(&params);
    let targets = EffectTargets {
        input: &dummy_view,
        output: &out_view,
        aux: None,
        aux2: None,
    };
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("zc-test-custom-lru"),
    });
    for i in 0..(MAX_CUSTOM_PIPELINES + 4) {
        let body = format!(
            "@fragment fn fs_main(in_v: VsOut) -> @location(0) vec4<f32> {{ let _k = {i}.0; return sample_diffuse(in_v.v_uv); }}"
        );
        let wgsl = build_effect_wgsl(&body);
        registry
            .run_custom_wgsl(device, &mut encoder, &wgsl, &uniforms, &targets)
            .expect("custom WGSL compile must succeed");
    }
    assert_eq!(registry.custom_pipeline_count(), MAX_CUSTOM_PIPELINES);
    registry.evict_custom_shaders();
    assert_eq!(registry.custom_pipeline_count(), 0);
}

#[test]
fn run_effect_rejects_non_finite_and_oversized_resolution_and_checks_blend_edges() {
    let ctx = match crate::GpuContext::new_headless() {
        Ok(ctx) => ctx,
        Err(crate::RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to init GpuContext: {err}"),
    };
    let mut registry =
        EffectRegistry::new(&ctx.device, &ctx.queue, wgpu::TextureFormat::Rgba8Unorm);

    let src = crate::RenderTarget::new(
        &ctx,
        4,
        4,
        wgpu::TextureFormat::Rgba8Unorm,
        Some("test src"),
    )
    .expect("src target");
    let dst = crate::RenderTarget::new(
        &ctx,
        4,
        4,
        wgpu::TextureFormat::Rgba8Unorm,
        Some("test dst"),
    )
    .expect("dst target");
    let targets = EffectTargets {
        input: &src.view,
        output: &dst.view,
        aux: None,
        aux2: None,
    };

    let too_big = (ctx.device.limits().max_texture_dimension_2d + 1) as f32;
    for bad_res in [
        [f32::INFINITY, 4.0],
        [4.0, f32::NAN],
        [too_big, 4.0],
        [4.0, too_big],
    ] {
        let params = EffectParams::new(bad_res, 0.0, 0.0);
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let res = registry.run_effect(
            &ctx.device,
            &ctx.queue,
            &mut enc,
            "boxblur",
            &params,
            &targets,
        );
        assert!(
            matches!(res, Err(EffectError::InvalidDimensions { .. })),
            "expected InvalidDimensions for {bad_res:?}, got {res:?}"
        );
    }

    // ColorDodge: black backdrop (0) with white overlay (1) must stay black (0).
    let black_src = ctx
        .create_rgba8_texture("zc-test-black", 4, 4, &[0, 0, 0, 255].repeat(16))
        .expect("black src");
    let dodge_params = EffectParams::new([4.0, 4.0], 0.0, 0.0)
        .with_param("color", EffectParamValue::Vec3([1.0, 1.0, 1.0]))
        .with_param("opacity", EffectParamValue::Float(1.0))
        .with_param("blending", EffectParamValue::Int(6));
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    registry
        .run_effect(
            &ctx.device,
            &ctx.queue,
            &mut enc,
            "coloroverlay",
            &dodge_params,
            &EffectTargets {
                input: &black_src.view,
                output: &dst.view,
                aux: None,
                aux2: None,
            },
        )
        .expect("coloroverlay dodge");
    ctx.queue.submit(std::iter::once(enc.finish()));
    let mut dodge_px = Vec::new();
    crate::readback_rgba8(&ctx, &dst, &mut dodge_px).expect("readback dodge");
    assert!(
        dodge_px[0] == 0 && dodge_px[1] == 0 && dodge_px[2] == 0 && dodge_px[3] == 255,
        "ColorDodge(0, 1) must stay black, got {:?}",
        &dodge_px[..4]
    );

    // ColorBurn: white backdrop (1) with black overlay (0) must stay white (255).
    let white_src = ctx
        .create_rgba8_texture("zc-test-white", 4, 4, &[255, 255, 255, 255].repeat(16))
        .expect("white src");
    let burn_params = EffectParams::new([4.0, 4.0], 0.0, 0.0)
        .with_param("color", EffectParamValue::Vec3([0.0, 0.0, 0.0]))
        .with_param("opacity", EffectParamValue::Float(1.0))
        .with_param("blending", EffectParamValue::Int(7));
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    registry
        .run_effect(
            &ctx.device,
            &ctx.queue,
            &mut enc,
            "coloroverlay",
            &burn_params,
            &EffectTargets {
                input: &white_src.view,
                output: &dst.view,
                aux: None,
                aux2: None,
            },
        )
        .expect("coloroverlay burn");
    ctx.queue.submit(std::iter::once(enc.finish()));
    let mut burn_px = Vec::new();
    crate::readback_rgba8(&ctx, &dst, &mut burn_px).expect("readback burn");
    assert!(
        burn_px[0] == 255 && burn_px[1] == 255 && burn_px[2] == 255 && burn_px[3] == 255,
        "ColorBurn(1, 0) must stay white, got {:?}",
        &burn_px[..4]
    );

    // S_Glow Overlay combine on dark non-glowing pixel must preserve source instead of blacking out.
    let dark_src = ctx
        .create_rgba8_texture("zc-test-dark", 4, 4, &[80, 60, 40, 255].repeat(16))
        .expect("dark src");
    let sglow_params = EffectParams::new([4.0, 4.0], 1.5, 45.0)
        .with_param("combine", EffectParamValue::Int(4))
        .with_param("threshold", EffectParamValue::Float(0.95))
        .with_param("brightness", EffectParamValue::Float(2.0));
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    registry
        .run_effect(
            &ctx.device,
            &ctx.queue,
            &mut enc,
            "sglow",
            &sglow_params,
            &EffectTargets {
                input: &dark_src.view,
                output: &dst.view,
                aux: None,
                aux2: None,
            },
        )
        .expect("sglow overlay");
    ctx.queue.submit(std::iter::once(enc.finish()));
    let mut sglow_px = Vec::new();
    crate::readback_rgba8(&ctx, &dst, &mut sglow_px).expect("readback sglow");
    assert!(
        (i32::from(sglow_px[0]) - 80).abs() <= 2
            && (i32::from(sglow_px[1]) - 60).abs() <= 2
            && (i32::from(sglow_px[2]) - 40).abs() <= 2,
        "S_Glow Overlay with zero glow must preserve dark source pixels, got {:?}",
        &sglow_px[..4]
    );
}

#[test]
fn parity_regression_checks_for_fixed_cm3_and_plugin_shaders() {
    let ctx = match crate::GpuContext::new_headless() {
        Ok(ctx) => ctx,
        Err(crate::RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to init GpuContext: {err}"),
    };
    let mut registry =
        EffectRegistry::new(&ctx.device, &ctx.queue, wgpu::TextureFormat::Rgba8Unorm);

    let dst = crate::RenderTarget::new(
        &ctx,
        8,
        8,
        wgpu::TextureFormat::Rgba8Unorm,
        Some("parity dst"),
    )
    .expect("dst");
    let run_and_read = |reg: &mut EffectRegistry,
                        id: &str,
                        src: &crate::RenderTarget,
                        aux: Option<&wgpu::TextureView>,
                        params: &EffectParams|
     -> Vec<u8> {
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        reg.run_effect(
            &ctx.device,
            &ctx.queue,
            &mut enc,
            id,
            params,
            &EffectTargets {
                input: &src.view,
                output: &dst.view,
                aux,
                aux2: None,
            },
        )
        .unwrap_or_else(|e| panic!("effect {id} failed: {e}"));
        ctx.queue.submit(std::iter::once(enc.finish()));
        let mut out = Vec::new();
        crate::readback_rgba8(&ctx, &dst, &mut out).expect("readback");
        out
    };

    // 1. colorcorrection: addRGB applied BEFORE pow: (0.2 + 0.3)^2 * 1.0 = 0.25 (~64)
    // Old bug gave 0.2^2 + 0.3 = 0.34 (~87).
    let src_cc = ctx
        .create_rgba8_texture("cc", 8, 8, &[51, 51, 51, 255].repeat(64))
        .expect("cc");
    let p_cc = EffectParams::new([8.0, 8.0], 0.0, 0.0)
        .with_param("powRGB", EffectParamValue::Vec3([2.0, 2.0, 2.0]))
        .with_param("mulRGB", EffectParamValue::Vec3([1.0, 1.0, 1.0]))
        .with_param("addRGB", EffectParamValue::Vec3([0.3, 0.3, 0.3]));
    let px = run_and_read(&mut registry, "colorcorrection", &src_cc, None, &p_cc);
    assert!(
        (i32::from(px[0]) - 64).abs() <= 3,
        "colorcorrection addRGB must happen before pow (expected ~64, got {})",
        px[0]
    );

    // 2. colorgrading: John Hable mean-subtracted neutral grey [0.5, 0.5, 0.5] must be identity.
    let src_cg = ctx
        .create_rgba8_texture("cg", 8, 8, &[180, 90, 45, 255].repeat(64))
        .expect("cg");
    let p_cg = EffectParams::new([8.0, 8.0], 0.0, 0.0)
        .with_param("shadows", EffectParamValue::Vec3([0.5, 0.5, 0.5]))
        .with_param("midtones", EffectParamValue::Vec3([0.5, 0.5, 0.5]))
        .with_param("highlights", EffectParamValue::Vec3([0.5, 0.5, 0.5]));
    let px = run_and_read(&mut registry, "colorgrading", &src_cg, None, &p_cg);
    assert!(
        (i32::from(px[0]) - 180).abs() <= 2
            && (i32::from(px[1]) - 90).abs() <= 2
            && (i32::from(px[2]) - 45).abs() <= 2,
        "colorgrading with neutral grey [0.5,0.5,0.5] must be identity, got {:?}",
        &px[..4]
    );

    // 3. colorize: must replace ONLY hue and preserve source saturation (grey stays grey!).
    let src_gray = ctx
        .create_rgba8_texture("gray", 8, 8, &[128, 128, 128, 255].repeat(64))
        .expect("gray");
    let p_col = EffectParams::new([8.0, 8.0], 0.0, 0.0)
        .with_param("color", EffectParamValue::Vec3([1.0, 0.0, 0.0]))
        .with_param("amount", EffectParamValue::Float(1.0));
    let px = run_and_read(&mut registry, "colorize", &src_gray, None, &p_col);
    assert!(
        (i32::from(px[0]) - 128).abs() <= 2
            && (i32::from(px[1]) - 128).abs() <= 2
            && (i32::from(px[2]) - 128).abs() <= 2,
        "colorize on desaturated grey must preserve 0 saturation, got {:?}",
        &px[..4]
    );

    // 4. gradientoverlay & overlay: OVERLAP_MODE = 3 must NEVER paint over transparent pixels.
    let src_clear = ctx
        .create_rgba8_texture("clear", 8, 8, &[0, 0, 0, 0].repeat(64))
        .expect("clear");
    let p_grad = EffectParams::new([8.0, 8.0], 0.0, 0.0)
        .with_param(
            "gradient",
            EffectParamValue::Lut1D(vec![[255, 0, 0, 255], [0, 255, 0, 255]]),
        )
        .with_param("opacity", EffectParamValue::Float(1.0));
    let px = run_and_read(&mut registry, "gradientoverlay", &src_clear, None, &p_grad);
    assert_eq!(
        &px[..4],
        &[0, 0, 0, 0],
        "gradientoverlay OVERLAP_MODE=3 must preserve transparent bg"
    );

    let ov_tex = ctx
        .create_rgba8_texture("ov", 8, 8, &[255, 255, 255, 255].repeat(64))
        .expect("ov");
    let p_ov =
        EffectParams::new([8.0, 8.0], 0.0, 0.0).with_param("opacity", EffectParamValue::Float(1.0));
    let px = run_and_read(
        &mut registry,
        "overlay",
        &src_clear,
        Some(&ov_tex.view),
        &p_ov,
    );
    assert_eq!(
        &px[..4],
        &[0, 0, 0, 0],
        "overlay OVERLAP_MODE=3 must preserve transparent bg"
    );

    // 5. mask: invert=true, opacity=0.0 must preserve 100% of image (1 - cov*0 = 1).
    let p_mask = EffectParams::new([8.0, 8.0], 0.0, 0.0)
        .with_param("invert", EffectParamValue::Bool(true))
        .with_param("opacity", EffectParamValue::Float(0.0));
    let px = run_and_read(&mut registry, "mask", &src_cg, None, &p_mask);
    assert!(
        (i32::from(px[0]) - 180).abs() <= 2 && px[3] == 255,
        "mask invert=1 opacity=0 must preserve 100% image, got {:?}",
        &px[..4]
    );

    // 6. transform: quad moved behind near plane (z > d - 1) must be clipped to transparent.
    let p_tf = EffectParams::new([8.0, 8.0], 0.0, 0.0)
        .with_param("imagePosition", EffectParamValue::Vec3([0.0, 0.0, 100.0]));
    let px = run_and_read(&mut registry, "transform", &src_cg, None, &p_tf);
    assert_eq!(
        &px[..4],
        &[0, 0, 0, 0],
        "transform quad behind camera near plane must clip"
    );

    // 7. sglow-hq-v2-preview alias + Mult combine (0) + Show Threshold (1) + legacy engine (0)
    let p_sglow_alias = EffectParams::new([8.0, 8.0], 0.5, 15.0)
        .with_param("engine", EffectParamValue::Int(0))
        .with_param("show", EffectParamValue::Int(1))
        .with_param("threshold", EffectParamValue::Float(0.2));
    let px = run_and_read(
        &mut registry,
        "sglow-hq-v2-preview",
        &src_cg,
        None,
        &p_sglow_alias,
    );
    assert!(
        px[0] > 0 && px[3] == 255,
        "sglow-hq-v2-preview show=1 must output threshold emitter, got {:?}",
        &px[..4]
    );

    // 8. anamorphicflare + bloom + dropshadow + radialblurspin + echo Screen unpremult
    for (id, params) in [
        ("anamorphicflare", EffectParams::new([8.0, 8.0], 0.0, 0.0)),
        ("bloom", EffectParams::new([8.0, 8.0], 0.0, 0.0)),
        ("dropshadow", EffectParams::new([8.0, 8.0], 0.0, 0.0)),
        ("radialblurspin", EffectParams::new([8.0, 8.0], 0.0, 0.0)),
        ("pixelsort", EffectParams::new([8.0, 8.0], 0.0, 0.0)),
        ("shake", EffectParams::new([8.0, 8.0], 0.0, 10.0)),
        ("exposure", EffectParams::new([8.0, 8.0], 0.0, 0.0)),
    ] {
        let out = run_and_read(&mut registry, id, &src_cg, None, &params);
        assert_eq!(out.len(), 8 * 8 * 4, "{id} readback size mismatch");
    }

    // echo Screen on semi-transparent red + semi-transparent green
    let semi_r = ctx
        .create_rgba8_texture("sr", 8, 8, &[128, 0, 0, 128].repeat(64))
        .expect("sr");
    let semi_g = ctx
        .create_rgba8_texture("sg", 8, 8, &[0, 128, 0, 128].repeat(64))
        .expect("sg");
    let p_echo = EffectParams::new([8.0, 8.0], 0.0, 0.0)
        .with_param("composite", EffectParamValue::Int(3))
        .with_param("opacity", EffectParamValue::Float(1.0));
    let px = run_and_read(&mut registry, "echo", &semi_r, Some(&semi_g.view), &p_echo);
    // base=(0.5,0,0,0.5), tap=(0,0.5,0,0.5): straight bs=(1,0,0), ts=(0,1,0), screen=(1,1,0)
    // rgb = (0.5,0,0)*0.5 + (0,0.5,0)*0.5 + (1,1,0)*0.25 = (0.5, 0.5, 0), a = 0.75 (~191)
    assert!(
        (i32::from(px[0]) - 128).abs() <= 3
            && (i32::from(px[1]) - 128).abs() <= 3
            && (i32::from(px[3]) - 191).abs() <= 3,
        "echo Screen unpremultiplied math mismatch: expected ~(128,128,0,191), got {:?}",
        &px[..4]
    );

    // 9. coloroverlay with BLEND_MINUS must preserve bg.a instead of wiping opaque bg to (0,0,0,0).
    let src_minus = ctx
        .create_rgba8_texture("minus_bg", 8, 8, &[204, 153, 102, 255].repeat(64))
        .expect("minus_bg");
    let p_minus = EffectParams::new([8.0, 8.0], 0.0, 0.0)
        .with_param("color", EffectParamValue::Vec3([0.3, 0.2, 0.1]))
        .with_param("opacity", EffectParamValue::Float(1.0))
        .with_param(
            "blending",
            EffectParamValue::String("BLEND_MINUS".to_owned()),
        );
    let px = run_and_read(&mut registry, "coloroverlay", &src_minus, None, &p_minus);
    // bg=(0.8, 0.6, 0.4, 1.0), s=(0.3, 0.2, 0.1, 1.0) -> (0.5, 0.4, 0.3, 1.0) = ~(128, 102, 77, 255)
    assert!(
        (i32::from(px[0]) - 128).abs() <= 3
            && (i32::from(px[1]) - 102).abs() <= 3
            && (i32::from(px[2]) - 77).abs() <= 3
            && px[3] == 255,
        "coloroverlay BLEND_MINUS must preserve bg.a & subtract RGB, got {:?}",
        &px[..4]
    );
    // Over transparent backdrop, BLEND_MINUS stays transparent.
    let px_clear = run_and_read(&mut registry, "coloroverlay", &src_clear, None, &p_minus);
    assert_eq!(
        &px_clear[..4],
        &[0, 0, 0, 0],
        "coloroverlay BLEND_MINUS over transparent bg must remain transparent"
    );
}

#[test]
fn wave2a_all_16_native_fx_upgraded_plugins_and_cm3_fixes_gpu() {
    // 1. Structural & uniqueness check for all 16 native FX + 3 helper shaders
    assert_eq!(NATIVE_FX_EFFECT_SHADERS.len(), 16);
    assert_eq!(HELPER_EFFECT_SHADERS.len(), 3);
    let std_set: BTreeSet<&str> = STANDARD_EFFECT_SHADERS.iter().map(|(id, _)| *id).collect();
    let plg_set: BTreeSet<&str> = PLUGIN_EFFECT_SHADERS.iter().map(|(id, _)| *id).collect();
    let nfx_set: BTreeSet<&str> = NATIVE_FX_EFFECT_SHADERS.iter().map(|(id, _)| *id).collect();
    let hlp_set: BTreeSet<&str> = HELPER_EFFECT_SHADERS.iter().map(|(id, _)| *id).collect();
    assert_eq!(nfx_set.len(), 16);
    assert!(nfx_set.is_disjoint(&std_set));
    assert!(nfx_set.is_disjoint(&plg_set));
    assert!(hlp_set.is_disjoint(&nfx_set));
    for &(id, body) in NATIVE_FX_EFFECT_SHADERS
        .iter()
        .chain(HELPER_EFFECT_SHADERS.iter())
    {
        assert!(!body.contains("discard;"), "{id}: must never use discard");
        assert!(lookup_effect_shader(id).is_some(), "{id}: missing lookup");
    }
    assert!(!SHADER_POLLY_MESH_WGSL.contains("discard;"));

    // 2. Verify 3 CM3 fixes (CPU checks)
    // 2a. bloom amount == 0 bypass
    let bloom_zero = EffectParams::new([32.0, 32.0], 0.0, 0.0)
        .with_param("amount", EffectParamValue::Float(0.0));
    assert!(should_bypass("bloom", &bloom_zero));
    let bloom_active = EffectParams::new([32.0, 32.0], 0.0, 0.0)
        .with_param("amount", EffectParamValue::Float(1.0));
    assert!(!should_bypass("bloom", &bloom_active));

    // 2b. shake exact zoom without arbitrary d * 0.85 clamp
    let shake_p = EffectParams::new([100.0, 100.0], 0.0, 0.0)
        .with_param("amplitude", EffectParamValue::Float(0.0))
        .with_param("zoom", EffectParamValue::Float(95.0));
    let u_shake = pack_effect_uniforms("shake", &shake_p, false, false);
    assert_eq!(u_shake.flags, 1);
    // At zoom=95, d=100, o_cam.z = -100 - (-95) = -5 (old clamp 85 gave -15).
    assert!(
        (u_shake.p4[2] - (-5.0)).abs() < 1e-3,
        "shake must not clamp zoom to d*0.85, got o_cam.z = {}",
        u_shake.p4[2]
    );

    // 2c. transform wrap mapping: 0=tile -> mode 1u, 1=reflect -> mode 2u
    let tf_tile =
        EffectParams::new([64.0, 64.0], 0.0, 0.0).with_param("wrap", EffectParamValue::Int(0));
    let tf_ref =
        EffectParams::new([64.0, 64.0], 0.0, 0.0).with_param("wrap", EffectParamValue::Int(1));
    assert_eq!(
        pack_effect_uniforms("transform", &tf_tile, false, false).mode,
        1
    );
    assert_eq!(
        pack_effect_uniforms("transform", &tf_ref, false, false).mode,
        2
    );

    // 3. Headless GPU compilation + rendering of ALL 16 native FX + upgraded plugins + Wave 2D helpers
    let ctx = match crate::GpuContext::new_headless() {
        Ok(ctx) => ctx,
        Err(crate::RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to init GpuContext: {err}"),
    };
    let mut registry =
        EffectRegistry::new(&ctx.device, &ctx.queue, wgpu::TextureFormat::Rgba8Unorm);

    // Compile every native FX + helper shader in Naga/wgpu
    for &(id, _) in NATIVE_FX_EFFECT_SHADERS
        .iter()
        .chain(HELPER_EFFECT_SHADERS.iter())
    {
        registry
            .ensure_pipeline(&ctx.device, id, false)
            .unwrap_or_else(|e| panic!("failed to compile WGSL `{id}`: {e}"));
    }

    // Build colorful 32x32 test textures (A = gradient/green block, B = shifted pattern, M = luma ramp)
    let (w, h) = (32u32, 32u32);
    let mut px_a = vec![0u8; (w * h * 4) as usize];
    let mut px_b = vec![0u8; (w * h * 4) as usize];
    let mut px_m = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let green_zone = x < w / 2;
            px_a[i] = if green_zone {
                20
            } else {
                (x * 8).min(255) as u8
            };
            px_a[i + 1] = if green_zone {
                230
            } else {
                (y * 8).min(255) as u8
            };
            px_a[i + 2] = if green_zone { 20 } else { 180 };
            px_a[i + 3] = 255;

            px_b[i] = ((x + 6) % w * 8).min(255) as u8;
            px_b[i + 1] = ((y + 4) % h * 8).min(255) as u8;
            px_b[i + 2] = 90;
            px_b[i + 3] = 255;

            let ramp = (x * 255 / (w - 1)) as u8;
            px_m[i] = ramp;
            px_m[i + 1] = ramp;
            px_m[i + 2] = ramp;
            px_m[i + 3] = 255;
        }
    }
    let tex_a = ctx.create_rgba8_texture("w2a-a", w, h, &px_a).expect("a");
    let tex_b = ctx.create_rgba8_texture("w2a-b", w, h, &px_b).expect("b");
    let tex_m = ctx.create_rgba8_texture("w2a-m", w, h, &px_m).expect("m");
    let dst = crate::RenderTarget::new(&ctx, w, h, wgpu::TextureFormat::Rgba8Unorm, Some("dst"))
        .expect("dst");

    let run_fx = |reg: &mut EffectRegistry,
                  id: &str,
                  aux: Option<&wgpu::TextureView>,
                  aux2: Option<&wgpu::TextureView>,
                  params: &EffectParams|
     -> Vec<u8> {
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let ran = reg
            .run_effect(
                &ctx.device,
                &ctx.queue,
                &mut enc,
                id,
                params,
                &EffectTargets {
                    input: &tex_a.view,
                    output: &dst.view,
                    aux,
                    aux2,
                },
            )
            .unwrap_or_else(|e| panic!("effect `{id}` failed: {e}"));
        assert!(ran, "effect `{id}` unexpectedly bypassed");
        ctx.queue.submit(std::iter::once(enc.finish()));
        let mut out = Vec::new();
        crate::readback_rgba8(&ctx, &dst, &mut out).expect("readback");
        assert_eq!(out.len(), (w * h * 4) as usize, "{id}: bad readback len");
        out
    };

    let base_p = EffectParams::new([w as f32, h as f32], 1.25, 37.0);

    // Execute EVERY ONE of the 16 nativeEffects and assert valid functional output
    for (id, aux, aux2, p) in [
        (
            "timedisplacement",
            Some(&tex_b.view),
            Some(&tex_m.view),
            base_p.clone(),
        ),
        ("ccwidetime", Some(&tex_b.view), None, base_p.clone()),
        (
            "timedifference",
            Some(&tex_b.view),
            None,
            base_p
                .clone()
                .with_param("absoluteDifference", EffectParamValue::Bool(true)),
        ),
        (
            "colordiffkey",
            None,
            None,
            base_p
                .clone()
                .with_param("keyColor", EffectParamValue::Vec3([0.0, 1.0, 0.0])),
        ),
        (
            "advspill",
            None,
            None,
            base_p
                .clone()
                .with_param("keyColor", EffectParamValue::Vec3([0.0, 1.0, 0.0])),
        ),
        (
            "keycleaner",
            None,
            None,
            base_p
                .clone()
                .with_param("reduceChatter", EffectParamValue::Bool(true)),
        ),
        (
            "linearcolorkey",
            None,
            None,
            base_p
                .clone()
                .with_param("keyColor", EffectParamValue::Vec3([0.0, 1.0, 0.0])),
        ),
        (
            "radiowaves",
            None,
            None,
            base_p
                .clone()
                .with_param("polygonSides", EffectParamValue::Int(5))
                .with_param("star", EffectParamValue::Bool(true)),
        ),
        ("beam", None, None, base_p.clone()),
        (
            "numbers",
            None,
            None,
            base_p
                .clone()
                .with_param("value", EffectParamValue::Float(42.5))
                .with_param("decimalPlaces", EffectParamValue::Int(1))
                .with_param("size", EffectParamValue::Float(14.0)),
        ),
        (
            "timecode",
            None,
            None,
            base_p
                .clone()
                .with_param("size", EffectParamValue::Float(12.0))
                .with_param("showBox", EffectParamValue::Bool(true)),
        ),
        ("ccpageturn", None, None, base_p.clone()),
        ("cardwipe", None, None, base_p.clone()),
        (
            "ccpixelpolly",
            None,
            None,
            base_p
                .clone()
                .with_param("time", EffectParamValue::Float(0.6))
                .with_param("gridSpacing", EffectParamValue::Float(8.0)),
        ),
        ("radialshadow", None, None, base_p.clone()),
        (
            "layerinputdisplacement",
            Some(&tex_m.view),
            None,
            base_p
                .clone()
                .with_param("amount", EffectParamValue::Float(0.15)),
        ),
    ] {
        let out = run_fx(&mut registry, id, aux, aux2, &p);
        assert!(
            out.iter().any(|&b| b > 0),
            "native FX `{id}` rendered all-zero output"
        );
    }

    // Verify `colordiffkey` keyed out the pure green left half (x=2, y=16 -> lower alpha than right half)
    let cdk_out = run_fx(
        &mut registry,
        "colordiffkey",
        None,
        None,
        &base_p
            .clone()
            .with_param("keyColor", EffectParamValue::Vec3([0.0, 1.0, 0.0])),
    );
    let left_a = cdk_out[((16 * w + 2) * 4 + 3) as usize];
    let right_a = cdk_out[((16 * w + 24) * 4 + 3) as usize];
    assert!(
        left_a < right_a,
        "colordiffkey must key green left half ({left_a}) more than right half ({right_a})"
    );

    // Verify Magic Looks full 29-tool chain execution via `looksInstances` JSON!
    let all_29_tools = serde_json::json!([
        {"toolId": "color-contrast", "ccPivot": 0.18, "ccExposure": 0.05},
        {"toolId": "hsl-colors", "hslRedSat": 0.1, "hslBlueLight": 0.05},
        {"toolId": "color-ranges", "crStrength": 0.3},
        {"toolId": "crush", "crGamma": 1.05},
        {"toolId": "contrast", "ctContrast": 0.1, "ctPivot": 0.5},
        {"toolId": "color-reversal", "rvStrength": 0.0},
        {"toolId": "three-strip", "tsStrength": 0.2},
        {"toolId": "lift-gamma-gain", "lggStrength": 0.25},
        {"toolId": "ranged-saturation", "rsSatMidtone": 1.1},
        {"toolId": "warm-cool", "wcWarmCool": 0.15, "wcTint": -0.05},
        {"toolId": "four-way", "fwStrength": 0.2},
        {"toolId": "mojo", "mjMojo": 0.2},
        {"toolId": "auto-shoulder", "asStrength": 0.15},
        {"toolId": "curves", "curvCurvesJson": "{\"master\":[[0,0],[0.5,0.55],[1,1]]}"},
        {"toolId": "s-curve", "scvScurveJson": "[[0,0],[0.25,0.22],[0.75,0.78],[1,1]]"},
        {"toolId": "lut", "lutMode": 1, "lutStrength": 0.25},
        {"toolId": "lightflex", "lfBoost": 0.3},
        {"toolId": "deflare", "dfStrength": 0.2},
        {"toolId": "vignette", "vgStrength": 0.25, "vgRadius": 0.8},
        {"toolId": "haze-flare", "hfSpillage": 0.1},
        {"toolId": "pop", "popPop": 0.2},
        {"toolId": "diffusion", "diffSize": 0.4, "diffGlow": 0.35},
        {"toolId": "star-filter", "sfSize": 0.3, "sfBoost": 0.4, "sfThreshold": 0.5},
        {"toolId": "anamorphic-flare", "afSize": 0.3, "afBoost": 0.4, "afThreshold": 0.5},
        {"toolId": "lens-distortion", "ldDistortion": 0.05, "ldFlatness": 1.0},
        {"toolId": "shutter-streak", "ssSize": 0.2, "ssBoost": 0.2, "ssFalloff": 0.5},
        {"toolId": "edge-softness", "esBlurSize": 0.01, "esRadius": 0.6},
        {"toolId": "chromatic-aberration", "caRedCyan": 0.2, "caBlueYellow": 0.2},
        {"toolId": "telecine-net", "tnSize": 0.05, "tnStrength": 0.15}
    ])
    .to_string();
    let ml_p = base_p
        .clone()
        .with_param("looksInstances", EffectParamValue::String(all_29_tools));
    let ml_out = run_fx(&mut registry, "looks", None, None, &ml_p);
    assert!(
        ml_out.iter().any(|&b| b > 0),
        "magiclooks 29-tool chain all-zero"
    );

    // Verify upgraded plugins: vhs (OSD + persistence), jpegdamage (DCT + errors), datamosh (anchor),
    // tracery (boxes/markers/lines/arrows/callouts), ascii (katakana + sine/noise),
    // smartmotionblur (quality 0/1/2), echo-legacy (modes 0..3), colorcurves (curves JSON + affectAlpha)
    let vhs_p = base_p
        .clone()
        .with_param("osd", EffectParamValue::Float(1.0))
        .with_param("tracking", EffectParamValue::Float(0.4))
        .with_param("persistence", EffectParamValue::Float(0.5));
    let _ = run_fx(&mut registry, "vhs", Some(&tex_b.view), None, &vhs_p);

    let jd_p = base_p
        .clone()
        .with_param("quality", EffectParamValue::Float(0.15))
        .with_param("errRate", EffectParamValue::Float(2.0))
        .with_param("highFreq", EffectParamValue::Float(0.5));
    let _ = run_fx(&mut registry, "jpegdamage", None, None, &jd_p);

    let dm_p = base_p
        .clone()
        .with_param("algorithm", EffectParamValue::Float(0.0))
        .with_param("blockSize", EffectParamValue::Float(8.0));
    let _ = run_fx(&mut registry, "datamosh", Some(&tex_b.view), None, &dm_p);

    let tr_p = base_p
        .clone()
        .with_param("point1Enable", EffectParamValue::Bool(true))
        .with_param("point2Enable", EffectParamValue::Bool(true))
        .with_param("arrowEnable", EffectParamValue::Bool(true))
        .with_param("labelsEnable", EffectParamValue::Bool(true))
        .with_param("dashEnable", EffectParamValue::Bool(true))
        .with_param("gridEnable", EffectParamValue::Bool(true));
    let _ = run_fx(&mut registry, "tracery", None, None, &tr_p);

    let ascii_p = base_p
        .clone()
        .with_param("charset", EffectParamValue::Int(4))
        .with_param("colorMode", EffectParamValue::Int(2))
        .with_param("sineEnable", EffectParamValue::Bool(true))
        .with_param("noiseEnable", EffectParamValue::Bool(true))
        .with_param("glitchEnable", EffectParamValue::Bool(true));
    let _ = run_fx(&mut registry, "ascii", None, None, &ascii_p);

    for q in [0, 1, 2] {
        let smb_p = base_p
            .clone()
            .with_param("quality", EffectParamValue::Int(q))
            .with_param("shutterAngle", EffectParamValue::Float(270.0));
        let _ = run_fx(
            &mut registry,
            "smart-motion-blur-lite",
            Some(&tex_b.view),
            Some(&tex_m.view),
            &smb_p,
        );
    }

    for m in 0..=3 {
        let echo_leg = base_p
            .clone()
            .with_param("mode", EffectParamValue::Int(m))
            .with_param("strength", EffectParamValue::Float(0.8))
            .with_param("threshold", EffectParamValue::Float(0.1))
            .with_param("wsum", EffectParamValue::Float(2.0))
            .with_param("finalize", EffectParamValue::Bool(true));
        let _ = run_fx(
            &mut registry,
            "echo-legacy",
            Some(&tex_b.view),
            None,
            &echo_leg,
        );
    }

    let cc_json_p = base_p
        .clone()
        .with_param(
            "curves",
            EffectParamValue::String(
                "{\"master\":[[0,0],[0.5,0.75],[1,1]],\"a\":[[0,0],[1,0.5]]}".to_owned(),
            ),
        )
        .with_param("affectAlpha", EffectParamValue::Bool(true));
    let cc_out = run_fx(&mut registry, "colorcurves", None, None, &cc_json_p);
    assert!(
        cc_out[3] < 160,
        "colorcurves affectAlpha with alpha curve [1->0.5] must halve alpha, got {}",
        cc_out[3]
    );

    // Verify Wave 2D helpers: run_frame_mix, run_shutter_accumulate, run_time_displacement_slice, run_wide_time_tap
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    registry
        .run_frame_mix(
            &ctx.device,
            &mut enc,
            &tex_a.view,
            &tex_b.view,
            &dst.view,
            [w as f32, h as f32],
            0.5,
            0,
        )
        .expect("run_frame_mix");
    registry
        .run_shutter_accumulate(
            &ctx.device,
            &mut enc,
            &tex_a.view,
            &tex_b.view,
            &dst.view,
            [w as f32, h as f32],
            1.0,
            1.0,
            true,
            true,
            wgpu::TextureFormat::Rgba8Unorm,
        )
        .expect("run_shutter_accumulate");
    registry
        .run_time_displacement_slice(
            &ctx.device,
            &mut enc,
            &tex_a.view,
            &tex_b.view,
            Some(&tex_m.view),
            &dst.view,
            &base_p,
            1,
            4,
        )
        .expect("run_time_displacement_slice");
    registry
        .run_wide_time_tap(
            &ctx.device,
            &mut enc,
            &tex_a.view,
            &tex_b.view,
            &dst.view,
            &base_p,
            0.5,
            false,
            false,
        )
        .expect("run_wide_time_tap");
    ctx.queue.submit(std::iter::once(enc.finish()));

    // Verify bloom brightTexture luma gating runs cleanly on GPU
    let bloom_p = base_p
        .clone()
        .with_param("amount", EffectParamValue::Float(1.0))
        .with_param("blendThreshold", EffectParamValue::Float(0.4))
        .with_param("blendSoften", EffectParamValue::Float(0.25));
    let bloom_out = run_fx(&mut registry, "bloom", None, None, &bloom_p);
    assert!(bloom_out.iter().any(|&b| b > 0));
}

#[test]
fn gpu_regression_qa05_qa11_alpha_cleanliness_spin_center_timecode_glitch_and_res_scaling() {
    let ctx = match crate::GpuContext::new_headless() {
        Ok(ctx) => ctx,
        Err(crate::RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to init GpuContext: {err}"),
    };
    let mut registry =
        EffectRegistry::new(&ctx.device, &ctx.queue, wgpu::TextureFormat::Rgba8Unorm);

    let (w, h) = (64u32, 64u32);
    // White centered 16x16 square over strictly transparent (0,0,0,0) backdrop.
    let mut sq_px = vec![0u8; (w * h * 4) as usize];
    for y in 24..40u32 {
        for x in 24..40u32 {
            let i = ((y * w + x) * 4) as usize;
            sq_px[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
    }
    let tex_sq = ctx
        .create_rgba8_texture("qa-center-sq", w, h, &sq_px)
        .expect("sq");
    let dst = crate::RenderTarget::new(&ctx, w, h, wgpu::TextureFormat::Rgba8Unorm, Some("qa-dst"))
        .expect("dst");

    let run_one =
        |reg: &mut EffectRegistry, id: &str, src: &crate::RenderTarget, params: &EffectParams| {
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            reg.run_effect(
                &ctx.device,
                &ctx.queue,
                &mut enc,
                id,
                params,
                &EffectTargets {
                    input: &src.view,
                    output: &dst.view,
                    aux: None,
                    aux2: None,
                },
            )
            .unwrap_or_else(|e| panic!("run_effect({id}) failed: {e}"));
            ctx.queue.submit(std::iter::once(enc.finish()));
            let mut out = Vec::new();
            crate::readback_rgba8(&ctx, &dst, &mut out).expect("readback");
            out
        };

    let corners = [(0usize, 0usize), (63, 0), (0, 63), (63, 63)];

    // 1. scanlines: empty transparent corners MUST stay alpha = 0, AND opaque interior MUST show crisp alternating rows.
    let sl_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
        .with_param("opacity", EffectParamValue::Float(1.0))
        .with_param("thickness", EffectParamValue::Float(1.0))
        .with_param("feather", EffectParamValue::Float(0.1));
    let sl_out = run_one(&mut registry, "scanlines", &tex_sq, &sl_p);
    for (cx, cy) in corners {
        let a = sl_out[(cy * (w as usize) + cx) * 4 + 3];
        assert_eq!(
            a, 0,
            "scanlines corrupted transparent corner ({cx},{cy}) alpha={a}"
        );
    }
    let row_even = sl_out[(30 * (w as usize) + 32) * 4];
    let row_odd = sl_out[(31 * (w as usize) + 32) * 4];
    assert!(
        row_even.abs_diff(row_odd) > 150,
        "scanlines thickness=1 on even height must produce crisp alternating rows (even={row_even}, odd={row_odd})"
    );

    // 2. rgbshift: empty corners MUST stay alpha = 0, AND shifted R/B fringes outside center square MUST have non-zero alpha.
    let rgb_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
        .with_param("amount", EffectParamValue::Float(6.0))
        .with_param("angle", EffectParamValue::Float(0.0));
    let rgb_out = run_one(&mut registry, "rgbshift", &tex_sq, &rgb_p);
    for (cx, cy) in corners {
        let a = rgb_out[(cy * (w as usize) + cx) * 4 + 3];
        assert_eq!(
            a, 0,
            "rgbshift corrupted transparent corner ({cx},{cy}) alpha={a}"
        );
    }
    // Left fringe at x = 20 (shifted R from x = 26) must be red with alpha > 200.
    let left_fringe = &rgb_out[(32 * (w as usize) + 20) * 4..(32 * (w as usize) + 20) * 4 + 4];
    assert!(
        left_fringe[0] > 200 && left_fringe[1] == 0 && left_fringe[3] > 200,
        "rgbshift left fringe must preserve shifted R channel with alpha, got {left_fringe:?}"
    );

    // 3. bloom: empty transparent corners MUST stay alpha = 0 even at full bloom amount.
    let bl_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
        .with_param("amount", EffectParamValue::Float(1.0))
        .with_param("strength", EffectParamValue::Float(0.75))
        .with_param("radius", EffectParamValue::Float(0.8));
    let bl_out = run_one(&mut registry, "bloom", &tex_sq, &bl_p);
    for (cx, cy) in corners {
        let a = bl_out[(cy * (w as usize) + cx) * 4 + 3];
        assert_eq!(
            a, 0,
            "bloom bled alpha={a} into empty transparent corner ({cx},{cy})"
        );
    }
    // Glow near square edge (x=22, y=32) must still be visible!
    let near_edge = &bl_out[(32 * (w as usize) + 22) * 4..(32 * (w as usize) + 22) * 4 + 4];
    assert!(
        near_edge[0] > 10 && near_edge[3] > 10,
        "bloom must still produce visible halo right next to bright shape, got {near_edge:?}"
    );

    // 4. radialblurspin: default center [0.5, 0.5] must spin symmetrically around screen center (32, 32).
    let mut bar_px = vec![0u8; (w * h * 4) as usize];
    for y in 28..36u32 {
        for x in 12..52u32 {
            let i = ((y * w + x) * 4) as usize;
            bar_px[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
    }
    let tex_bar = ctx
        .create_rgba8_texture("qa-bar", w, h, &bar_px)
        .expect("bar");
    let rbs_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
        .with_param("center", EffectParamValue::Vec2([0.5, 0.5]))
        .with_param("density", EffectParamValue::Float(2.0))
        .with_param("dither", EffectParamValue::Float(0.0));
    let rbs_out = run_one(&mut registry, "radialblurspin", &tex_bar, &rbs_p);
    let rbs_slice = rbs_out.as_slice();
    let top_left_sum: u64 = (18..32usize)
        .flat_map(|y| (12..32usize).map(move |x| u64::from(rbs_slice[(y * 64 + x) * 4])))
        .sum();
    let bot_right_sum: u64 = (32..46usize)
        .flat_map(|y| (32..52usize).map(move |x| u64::from(rbs_slice[(y * 64 + x) * 4])))
        .sum();
    let sym_diff = top_left_sum.abs_diff(bot_right_sum) as f64 / (top_left_sum.max(1) as f64);
    assert!(
        sym_diff < 0.12,
        "radialblurspin center=[0.5,0.5] must be symmetric around screen center (tl={top_left_sum}, br={bot_right_sum}, diff={sym_diff:.3})"
    );

    // 5. timecode: textPosition moves rendered timecode between top (y=0.15) and bottom (y=0.85).
    let tc_top_p = EffectParams::new([w as f32, h as f32], 1.5, 45.0)
        .with_param("textPosition", EffectParamValue::Vec2([0.5, 0.15]))
        .with_param("textSize", EffectParamValue::Float(120.0))
        .with_param("renderOnOriginal", EffectParamValue::Bool(false));
    let tc_bot_p = EffectParams::new([w as f32, h as f32], 1.5, 45.0)
        .with_param("textPosition", EffectParamValue::Vec2([0.5, 0.85]))
        .with_param("textSize", EffectParamValue::Float(120.0))
        .with_param("renderOnOriginal", EffectParamValue::Bool(false));
    let tc_top = run_one(&mut registry, "timecode", &tex_sq, &tc_top_p);
    let tc_bot = run_one(&mut registry, "timecode", &tex_sq, &tc_bot_p);
    let centroid_y = |px: &[u8]| -> f64 {
        let (mut sy, mut sw) = (0.0_f64, 0.0_f64);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let a = f64::from(px[(y * w as usize + x) * 4 + 3]);
                sy += (y as f64) * a;
                sw += a;
            }
        }
        if sw > 0.0 { sy / sw } else { 32.0 }
    };
    let cy_top = centroid_y(&tc_top);
    let cy_bot = centroid_y(&tc_bot);
    assert!(
        cy_top < 24.0 && cy_bot > 40.0,
        "timecode must honor textPosition (cy_top={cy_top:.1}, cy_bot={cy_bot:.1})"
    );

    // 6. glitch: default parameters at time=0.0 must visibly displace a vertical stripe pattern.
    let mut stripe_px = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let v = if x < w / 2 { 255u8 } else { 20u8 };
            stripe_px[i..i + 4].copy_from_slice(&[v, v, v, 255]);
        }
    }
    let tex_stripe = ctx
        .create_rgba8_texture("qa-stripe", w, h, &stripe_px)
        .expect("stripe");
    let gl_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0);
    let gl_out = run_one(&mut registry, "glitch", &tex_stripe, &gl_p);
    let changed_px = gl_out
        .as_chunks::<4>()
        .0
        .iter()
        .zip(stripe_px.as_chunks::<4>().0.iter())
        .filter(|(a, b)| a[0].abs_diff(b[0]) > 40)
        .count();
    assert!(
        changed_px > 20,
        "default glitch at time=0 must visibly displace pixels (changed={changed_px})"
    );

    // 7. pixelated & ascii resolution scaling via `_sequenceResolution`
    let pix_full = EffectParams::new([1920.0, 1080.0], 0.0, 0.0)
        .with_param(
            "_sequenceResolution",
            EffectParamValue::Vec2([1920.0, 1080.0]),
        )
        .with_param("size", EffectParamValue::Vec2([32.0, 18.0]));
    let pix_half = EffectParams::new([960.0, 540.0], 0.0, 0.0)
        .with_param(
            "_sequenceResolution",
            EffectParamValue::Vec2([1920.0, 1080.0]),
        )
        .with_param("size", EffectParamValue::Vec2([32.0, 18.0]));
    let u_pf = pack_effect_uniforms("pixelated", &pix_full, false, false);
    let u_ph = pack_effect_uniforms("pixelated", &pix_half, false, false);
    assert!(
        (u_pf.p0[0] / u_pf.resolution[0] - u_ph.p0[0] / u_ph.resolution[0]).abs() < 1e-5,
        "pixelated block UV width must match across Full & Half preview"
    );
    let asc_full = EffectParams::new([1920.0, 1080.0], 0.0, 0.0)
        .with_param(
            "_sequenceResolution",
            EffectParamValue::Vec2([1920.0, 1080.0]),
        )
        .with_param("blockSize", EffectParamValue::Float(24.0))
        .with_param("charSize", EffectParamValue::Float(16.0));
    let asc_half = EffectParams::new([960.0, 540.0], 0.0, 0.0)
        .with_param(
            "_sequenceResolution",
            EffectParamValue::Vec2([1920.0, 1080.0]),
        )
        .with_param("blockSize", EffectParamValue::Float(24.0))
        .with_param("charSize", EffectParamValue::Float(16.0));
    let u_af = pack_effect_uniforms("ascii", &asc_full, false, false);
    let u_ah = pack_effect_uniforms("ascii", &asc_half, false, false);
    assert!(
        (u_af.p0[0] / u_af.resolution[0] - u_ah.p0[0] / u_ah.resolution[0]).abs() < 1e-5,
        "ascii cell UV width must match across Full & Half preview"
    );
}

#[test]
fn gpu_regression_effects_contrast_and_rgb_channel_blend_alpha_cleanliness() {
    let ctx = match crate::GpuContext::new_headless() {
        Ok(ctx) => ctx,
        Err(crate::RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to init GpuContext: {err}"),
    };
    let mut registry =
        EffectRegistry::new(&ctx.device, &ctx.queue, wgpu::TextureFormat::Rgba8Unorm);

    let (w, h) = (8u32, 8u32);
    // Opaque sky-blue backdrop: (51, 102, 204, 255) ~= (0.2, 0.4, 0.8, 1.0)
    let bg_px = [51u8, 102, 204, 255].repeat((w * h) as usize);
    let tex_bg = ctx
        .create_rgba8_texture("fx-blend-bg", w, h, &bg_px)
        .expect("bg");
    let dst = crate::RenderTarget::new(
        &ctx,
        w,
        h,
        wgpu::TextureFormat::Rgba8Unorm,
        Some("fx-blend-dst"),
    )
    .expect("dst");

    let run_overlay = |reg: &mut EffectRegistry, mode: &str, opacity: f32| -> [u8; 4] {
        let params = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("color", EffectParamValue::Vec3([1.0, 0.9, 0.1]))
            .with_param("opacity", EffectParamValue::Float(opacity))
            .with_param("blending", EffectParamValue::String(mode.to_owned()));
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        reg.run_effect(
            &ctx.device,
            &ctx.queue,
            &mut enc,
            "coloroverlay",
            &params,
            &EffectTargets {
                input: &tex_bg.view,
                output: &dst.view,
                aux: None,
                aux2: None,
            },
        )
        .expect("run coloroverlay");
        ctx.queue.submit(std::iter::once(enc.finish()));
        let mut out = Vec::new();
        crate::readback_rgba8(&ctx, &dst, &mut out).expect("readback");
        [out[0], out[1], out[2], out[3]]
    };

    let cases: &[(&str, f32, [i32; 4])] = &[
        // opacity == 0.0 must leave backdrop untouched for all 4 modes
        ("BLEND_CONTRAST", 0.0, [51, 102, 204, 255]),
        ("BLEND_RED", 0.0, [51, 102, 204, 255]),
        ("BLEND_GREEN", 0.0, [51, 102, 204, 255]),
        ("BLEND_BLUE", 0.0, [51, 102, 204, 255]),
        // opacity == 0.5 smooth premultiplied blend
        ("BLEND_CONTRAST", 0.5, [70, 110, 150, 255]),
        ("BLEND_RED", 0.5, [153, 102, 204, 255]),
        ("BLEND_GREEN", 0.5, [51, 166, 204, 255]),
        ("BLEND_BLUE", 0.5, [51, 102, 115, 255]),
        // opacity == 1.0 full blend
        ("BLEND_CONTRAST", 1.0, [51, 107, 66, 255]),
        ("BLEND_RED", 1.0, [255, 102, 204, 255]),
        ("BLEND_GREEN", 1.0, [51, 230, 204, 255]),
        ("BLEND_BLUE", 1.0, [51, 102, 26, 255]),
    ];
    for &(mode, opacity, expected) in cases {
        let px = run_overlay(&mut registry, mode, opacity);
        assert!(
            (i32::from(px[0]) - expected[0]).abs() <= 3
                && (i32::from(px[1]) - expected[1]).abs() <= 3
                && (i32::from(px[2]) - expected[2]).abs() <= 3
                && (i32::from(px[3]) - expected[3]).abs() <= 3,
            "coloroverlay {mode} opacity={opacity} expected ~{expected:?}, got {px:?}"
        );
    }
}

#[test]
fn gpu_regression_c5_cardwipe_alpha_blur_fisheye_exposure_pixelsort_keycleaner_smblite_vignette() {
    let ctx = match crate::GpuContext::new_headless() {
        Ok(ctx) => ctx,
        Err(crate::RenderError::NoAdapter(_)) if !crate::require_gpu_in_tests() => return,
        Err(err) => panic!("failed to init GpuContext: {err}"),
    };
    let mut registry =
        EffectRegistry::new(&ctx.device, &ctx.queue, wgpu::TextureFormat::Rgba8Unorm);

    let (w, h) = (64u32, 64u32);
    let dst = crate::RenderTarget::new(&ctx, w, h, wgpu::TextureFormat::Rgba8Unorm, Some("c5-dst"))
        .expect("dst target");

    let run_one = |reg: &mut EffectRegistry,
                   id: &str,
                   src: &crate::RenderTarget,
                   aux: Option<&wgpu::TextureView>,
                   aux2: Option<&wgpu::TextureView>,
                   params: &EffectParams|
     -> Vec<u8> {
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let ran = reg
            .run_effect(
                &ctx.device,
                &ctx.queue,
                &mut enc,
                id,
                params,
                &EffectTargets {
                    input: &src.view,
                    output: &dst.view,
                    aux,
                    aux2,
                },
            )
            .expect("run_effect");
        assert!(ran, "{id} unexpectedly bypassed");
        ctx.queue.submit(std::iter::once(enc.finish()));
        let mut out = Vec::new();
        crate::readback_rgba8(&ctx, &dst, &mut out).expect("readback");
        out
    };

    // Opaque gradient texture for cardwipe / vignette / rgbshift edge / exposure tests.
    let mut grad_px = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            grad_px[i] = ((x * 255) / (w - 1)) as u8;
            grad_px[i + 1] = ((y * 255) / (h - 1)) as u8;
            grad_px[i + 2] = 160;
            grad_px[i + 3] = 255;
        }
    }
    let tex_grad = ctx
        .create_rgba8_texture("c5-grad", w, h, &grad_px)
        .expect("grad");

    // Transparent canvas with centered cyan circle/square + 1px AA ring.
    let mut cyan_px = vec![0u8; (w * h * 4) as usize];
    for y in 20..44u32 {
        for x in 20..44u32 {
            let i = ((y * w + x) * 4) as usize;
            let border = x == 20 || x == 43 || y == 20 || y == 43;
            if border {
                cyan_px[i..i + 4].copy_from_slice(&[0, 100, 100, 100]);
            } else {
                cyan_px[i..i + 4].copy_from_slice(&[0, 255, 255, 255]);
            }
        }
    }
    let tex_cyan = ctx
        .create_rgba8_texture("c5-cyan", w, h, &cyan_px)
        .expect("cyan");
    let corners = [(0usize, 0usize), (63, 0), (0, 63), (63, 63)];

    // 1. cardwipe: completion=0 is 100% pass-through; 0 vs 50 vs 100 all distinct.
    let cw_p = |comp: f32| {
        EffectParams::new([w as f32, h as f32], 0.5, 15.0)
            .with_param("completion", EffectParamValue::Float(comp))
            .with_param("transitionWidth", EffectParamValue::Float(50.0))
            .with_param("backLayer", EffectParamValue::Int(1))
    };
    let cw0 = run_one(&mut registry, "cardwipe", &tex_grad, None, None, &cw_p(0.0));
    let cw50 = run_one(
        &mut registry,
        "cardwipe",
        &tex_grad,
        None,
        None,
        &cw_p(50.0),
    );
    let cw100 = run_one(
        &mut registry,
        "cardwipe",
        &tex_grad,
        None,
        None,
        &cw_p(100.0),
    );
    let max_err_0 = cw0
        .iter()
        .zip(grad_px.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert!(
        max_err_0 <= 2,
        "cardwipe completion=0 must be 100% pass-through (max_err={max_err_0})"
    );
    assert_ne!(cw0, cw50, "cardwipe completion=50 must differ from 0");
    assert_ne!(cw50, cw100, "cardwipe completion=100 must differ from 50");
    assert!(
        cw100.as_chunks::<4>().0.iter().all(|p| p[3] == 0),
        "cardwipe completion=100 with transparent backside must flip all cards"
    );

    // 2. static: transparent corners stay alpha == 0.
    let st_p = EffectParams::new([w as f32, h as f32], 0.5, 15.0)
        .with_param("amount", EffectParamValue::Float(0.5))
        .with_param("blending", EffectParamValue::Int(1));
    let st_out = run_one(&mut registry, "static", &tex_cyan, None, None, &st_p);
    for (cx, cy) in corners {
        assert_eq!(
            st_out[(cy * (w as usize) + cx) * 4 + 3],
            0,
            "static corrupted transparent corner ({cx},{cy})"
        );
    }

    // 3. rgbshift: cyan transparent shape has NO jet-black left crescent halo, AND opaque left edge has no dark stripe.
    let rgb_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
        .with_param("amount", EffectParamValue::Float(6.0))
        .with_param("angle", EffectParamValue::Float(0.0));
    let rgb_cyan = run_one(&mut registry, "rgbshift", &tex_cyan, None, None, &rgb_p);
    let left_crescent = &rgb_cyan[(32 * 64 + 17) * 4..(32 * 64 + 17) * 4 + 4];
    assert_eq!(
        left_crescent[3], 0,
        "rgbshift must NOT create opaque black crescent on cyan left fringe, got {left_crescent:?}"
    );
    let right_fringe = &rgb_cyan[(32 * 64 + 46) * 4..(32 * 64 + 46) * 4 + 4];
    assert!(
        right_fringe[2] > 200 && right_fringe[3] > 200,
        "rgbshift right fringe on cyan must remain pure blue, got {right_fringe:?}"
    );
    let solid_tex = ctx
        .create_rgba8_texture(
            "c5-solid",
            w,
            h,
            &[200, 150, 100, 255].repeat((w * h) as usize),
        )
        .expect("solid");
    let rgb_solid = run_one(&mut registry, "rgbshift", &solid_tex, None, None, &rgb_p);
    let left_edge = &rgb_solid[(32 * 64) * 4..(32 * 64) * 4 + 4];
    assert!(
        left_edge[0] >= 198 && left_edge[1] >= 148 && left_edge[2] >= 98,
        "rgbshift must NOT leave dark left-edge stripe on opaque scene, got {left_edge:?}"
    );

    // 4. datamosh: never stamps opaque black (0,0,0,255) blocks onto transparent cyan layer.
    let dm_p = EffectParams::new([w as f32, h as f32], 0.5, 15.0)
        .with_param("algorithm", EffectParamValue::Int(12))
        .with_param("amount", EffectParamValue::Float(1.0))
        .with_param("intensity", EffectParamValue::Float(0.8))
        .with_param("blend", EffectParamValue::Float(0.9));
    let dm_out = run_one(&mut registry, "datamosh", &tex_cyan, None, None, &dm_p);
    let black_opaque_px = dm_out
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[3] > 200 && p[0] < 10 && p[1] < 10 && p[2] < 10)
        .count();
    assert_eq!(
        black_opaque_px, 0,
        "datamosh stamped {black_opaque_px} opaque black pixels onto transparent cyan layer"
    );

    // 5. bloom: extreme strength=10, radius=5 must NOT produce black ring halo, and corners stay alpha=0.
    let bl_ext_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
        .with_param("threshold", EffectParamValue::Float(0.5))
        .with_param("strength", EffectParamValue::Float(10.0))
        .with_param("radius", EffectParamValue::Float(5.0))
        .with_param("amount", EffectParamValue::Float(3.0))
        .with_param("blendThreshold", EffectParamValue::Float(0.5))
        .with_param("blendSoften", EffectParamValue::Float(0.5));
    let bl_ext = run_one(&mut registry, "bloom", &tex_cyan, None, None, &bl_ext_p);
    let halo_ring = &bl_ext[(32 * 64 + 19) * 4..(32 * 64 + 19) * 4 + 4];
    assert!(
        halo_ring[1] > 50 && halo_ring[2] > 50,
        "bloom strength=10 radius=5 must produce bright halo, not black ring: {halo_ring:?}"
    );
    let bl_def_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
        .with_param("amount", EffectParamValue::Float(1.0))
        .with_param("strength", EffectParamValue::Float(0.75))
        .with_param("radius", EffectParamValue::Float(0.8));
    let bl_def = run_one(&mut registry, "bloom", &tex_cyan, None, None, &bl_def_p);
    for (cx, cy) in corners {
        assert_eq!(
            bl_def[(cy * 64 + cx) * 4 + 3],
            0,
            "bloom corner ({cx},{cy}) must stay alpha=0"
        );
    }

    // 6. boxblur / gaussianblur / directionalblur at delta=200 produce finite smooth blur.
    for (id, extra) in [
        ("boxblur", None),
        ("gaussianblur", None),
        ("directionalblur", Some(("direction", 0.785_f32))),
    ] {
        let mut p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
            .with_param("delta", EffectParamValue::Float(200.0));
        if let Some((k, v)) = extra {
            p.set(k, EffectParamValue::Float(v));
        }
        let out = run_one(&mut registry, id, &tex_cyan, None, None, &p);
        assert!(
            out.as_chunks::<4>().0.iter().any(|px| px[3] > 0),
            "{id} delta=200 must produce non-empty smooth blur"
        );
    }

    // 7. fisheye: extreme positive amount=500 must NOT wipe centered shape to transparent.
    let fe_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
        .with_param("amount", EffectParamValue::Float(500.0));
    let fe_out = run_one(&mut registry, "fisheye", &tex_cyan, None, None, &fe_p);
    let center_fe = &fe_out[(32 * 64 + 32) * 4..(32 * 64 + 32) * 4 + 4];
    assert!(
        center_fe[1] > 200 && center_fe[3] > 200,
        "fisheye amount=500 must preserve centered subject, got {center_fe:?}"
    );

    // 8. exposure: gamma=1.0 brightens smoothly without blowing out to 100% white (255,255,255).
    let mid_tex = ctx
        .create_rgba8_texture(
            "c5-mid",
            w,
            h,
            &[80, 100, 120, 255].repeat((w * h) as usize),
        )
        .expect("mid");
    let exp_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
        .with_param("exposure", EffectParamValue::Float(0.0))
        .with_param("gamma", EffectParamValue::Float(1.0));
    let exp_out = run_one(&mut registry, "exposure", &mid_tex, None, None, &exp_p);
    assert!(
        exp_out[0] > 120 && exp_out[0] < 248 && exp_out[1] < 250,
        "exposure gamma=1.0 must brighten smoothly without whiteout, got {:?}",
        &exp_out[..4]
    );

    // 9. pixelsort: no stray cyan pixels on top row y=0 of transparent canvas.
    let ps_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
        .with_param("threshold", EffectParamValue::Float(0.5))
        .with_param("direction", EffectParamValue::Int(0))
        .with_param("affect", EffectParamValue::Int(0));
    let ps_out = run_one(&mut registry, "pixelsort", &tex_cyan, None, None, &ps_p);
    let stray_top_row = (0..w as usize).filter(|&x| ps_out[x * 4 + 3] > 0).count();
    assert_eq!(
        stray_top_row, 0,
        "pixelsort must not teleport pixels to y=0 on transparent canvas (stray={stray_top_row})"
    );

    // 10. sglow: transparent far corners stay alpha == 0.
    let sg_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
        .with_param("glowWidth", EffectParamValue::Float(96.0))
        .with_param("brightness", EffectParamValue::Float(2.0))
        .with_param("threshold", EffectParamValue::Float(0.5));
    let sg_out = run_one(&mut registry, "sglow", &tex_cyan, None, None, &sg_p);
    for (cx, cy) in corners {
        assert_eq!(
            sg_out[(cy * 64 + cx) * 4 + 3],
            0,
            "sglow leaked haze into transparent corner ({cx},{cy})"
        );
    }

    // 11. smart-motion-blur-lite: default sensitivity=0.7 produces real diff on translated frames.
    let mut shifted_prev = vec![0u8; (w * h * 4) as usize];
    let mut shifted_next = vec![0u8; (w * h * 4) as usize];
    for y in 20..44u32 {
        for x in 14..38u32 {
            let i = ((y * w + x) * 4) as usize;
            shifted_prev[i..i + 4].copy_from_slice(&[0, 255, 255, 255]);
        }
        for x in 26..50u32 {
            let i = ((y * w + x) * 4) as usize;
            shifted_next[i..i + 4].copy_from_slice(&[0, 255, 255, 255]);
        }
    }
    let tex_prev = ctx
        .create_rgba8_texture("c5-prev", w, h, &shifted_prev)
        .expect("prev");
    let tex_next = ctx
        .create_rgba8_texture("c5-next", w, h, &shifted_next)
        .expect("next");
    let smbl_p = EffectParams::new([w as f32, h as f32], 0.5, 15.0)
        .with_param("qualityPresetVersion", EffectParamValue::Float(2.0))
        .with_param("quality", EffectParamValue::Int(1))
        .with_param("amount", EffectParamValue::Float(1.0))
        .with_param("sensitivity", EffectParamValue::Float(0.7));
    let smbl_out = run_one(
        &mut registry,
        "smart-motion-blur-lite",
        &tex_cyan,
        Some(&tex_prev.view),
        Some(&tex_next.view),
        &smbl_p,
    );
    let smbl_changed = smbl_out
        .iter()
        .zip(cyan_px.iter())
        .filter(|(a, b)| a.abs_diff(**b) > 5)
        .count();
    assert!(
        smbl_changed > 20,
        "smart-motion-blur-lite with sensitivity=0.7 must produce motion blur diff (changed={smbl_changed})"
    );

    // 12. keycleaner: default Zoidium alphaContrast=100, strength=100 tightens soft edges.
    let kc_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0)
        .with_param("additionalEdgeRadius", EffectParamValue::Float(3.0))
        .with_param("alphaContrast", EffectParamValue::Float(100.0))
        .with_param("strength", EffectParamValue::Float(100.0));
    let kc_out = run_one(&mut registry, "keycleaner", &tex_cyan, None, None, &kc_p);
    let kc_changed = kc_out
        .iter()
        .zip(cyan_px.iter())
        .filter(|(a, b)| a.abs_diff(**b) > 5)
        .count();
    assert!(
        kc_changed > 20,
        "keycleaner must tighten/decontaminate partial-alpha border (changed={kc_changed})"
    );

    // 13. vignette: darkens corners smoothly relative to center & preserves transparent alpha.
    let vig_p = EffectParams::new([w as f32, h as f32], 0.0, 0.0);
    let vig_out = run_one(&mut registry, "vignette", &solid_tex, None, None, &vig_p);
    let vig_center = vig_out[(32 * 64 + 32) * 4];
    let vig_corner = vig_out[0];
    assert!(
        vig_center >= 195 && vig_corner + 40 < vig_center,
        "vignette must preserve center ({vig_center}) and darken corners ({vig_corner})"
    );
    let vig_trans = run_one(&mut registry, "vignette", &tex_cyan, None, None, &vig_p);
    assert_eq!(
        vig_trans[3], 0,
        "vignette must keep transparent corner alpha=0"
    );

    // 14. beam: _timeIsSeconds=true produces distinct visible beams at t=0.5s vs t=1.5s.
    let beam_p = |t_sec: f32| {
        EffectParams::new([w as f32, h as f32], t_sec, t_sec * 30.0)
            .with_param("length", EffectParamValue::Float(25.0))
            .with_param("time", EffectParamValue::Float(t_sec))
            .with_param("_timeIsSeconds", EffectParamValue::Bool(true))
            .with_param("composite", EffectParamValue::Int(0))
    };
    let b05 = run_one(&mut registry, "beam", &tex_grad, None, None, &beam_p(0.5));
    let b15 = run_one(&mut registry, "beam", &tex_grad, None, None, &beam_p(1.5));
    let b05_lit = b05.as_chunks::<4>().0.iter().filter(|p| p[3] > 30).count();
    let b15_lit = b15.as_chunks::<4>().0.iter().filter(|p| p[3] > 30).count();
    assert!(
        b05_lit > 40 && b15_lit > 40 && b05 != b15,
        "beam with expr=time must render distinct full beams (b05={b05_lit}, b15={b15_lit})"
    );
}
