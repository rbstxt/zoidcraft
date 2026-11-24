//! ZoidCraft Custom Shader translator (`GLSL ES 1.00 -> GLSL 450 -> naga -> WGSL`)
//! and Shader+ custom property / LUT baking subsystem.

#![forbid(unsafe_code)]

pub mod glsl_shim;
pub mod node_graph;
pub mod shader_plus;

pub use glsl_shim::{
    CompileOptions, CompiledCustomShader, MAX_GLSL_SOURCE_BYTES, ShaderError, TextureBinding,
    UniformBlockLayout, UniformField, UniformType, UniformValue, align_up, bind_curve_ranges,
    bind_gradient_cycles, fix_property_name, pack_uniforms, translate_glsl_es100,
    translate_glsl_es100_with_options,
};
pub use node_graph::{
    BakedMap, BakedNodeMaterialMaps, CompiledNodeEffect, NodeEffectParameter,
    bake_node_material_maps, compile_node_effect_to_glsl,
};
pub use shader_plus::{
    CURVE_LUT_WIDTH, ConditionClause, CurveChannel, CurveMode, CurvePoint, CurveSpec,
    GRADIENT_LUT_WIDTH, GradientColorSpace, GradientInterpolation, GradientSpec, GradientStop,
    MAX_PALETTE_COLORS, MAX_TIME_SAMPLES, OpacityStop, PaletteSpec, RenameRule, ShaderPlusError,
    ShaderPropertyDef, ShaderPropertyKind, TextTextureMode, TimeSampleUnit, ToggleDisplayStyle,
    VisibilityCondition, ZoidiumConditionOp, ZoidiumConditionValue, ZoidiumCurveHandles,
    ZoidiumCurvePoint, ZoidiumRenameRule, ZoidiumVisibilityWhen, bake_curve_rgba8,
    bake_curve_rgba16f, bake_curve_rgba32f, bake_gradient_rgba8, bake_gradient_rgba8_with_opacity,
    evaluate_zoidium_curve_points, f32_to_f16_bits, resolve_label,
};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn read_f32_le(buf: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
    }

    fn read_i32_le(buf: &[u8], offset: usize) -> i32 {
        i32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
    }

    fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn compiles_panzoid_shader_with_derivatives_and_standard_uniforms() {
        let src = r#"
            #extension GL_OES_standard_derivatives : require
            precision highp float;

            varying vec2 vUv;
            varying vec2 vUvScaled;

            uniform sampler2D tDiffuse;
            uniform vec2 resolution;
            uniform vec3 Color;
            uniform float Opacity;
            uniform vec2 Position;
            uniform float Rotation;
            uniform vec2 Scale;
            uniform bool Invert;
            uniform int Mode;

            void main() {
                vec4 bg = texture2D(tDiffuse, vUvScaled);
                vec2 p = (vUvScaled * 2.0 - 1.0) - Position;
                p.x *= 16.0 / 9.0;
                float rad = radians(Rotation);
                float c = cos(rad);
                float s = sin(rad);
                p = mat2(c, s, -s, c) * p;
                p /= max(Scale, vec2(0.0001));

                float d = length(p) - 0.5;
                float aa = max(fwidth(d), 0.001);
                float mask = 1.0 - smoothstep(-aa, aa, d);
                if (Invert) {
                    mask = 1.0 - mask;
                }
                if (Mode == 1) {
                    mask *= clamp(abs(dFdx(p.x)) * 100.0, 0.0, 1.0);
                }

                float srcAlpha = clamp(mask * Opacity, 0.0, 1.0);
                vec3 outRgb = Color * srcAlpha + bg.rgb * (1.0 - srcAlpha);
                float outA = srcAlpha + bg.a * (1.0 - srcAlpha);
                gl_FragColor = vec4(outRgb, outA);
            }
        "#;

        let compiled = translate_glsl_es100(src).expect("shader must compile to WGSL");

        assert!(compiled.wgsl.contains("@fragment"));
        assert!(compiled.wgsl.contains("texture_2d<f32>"));
        assert!(compiled.wgsl.contains("sampler"));
        assert!(compiled.wgsl.contains("fwidth"));
        assert!(compiled.wgsl.contains("dpdx"));

        assert_eq!(compiled.texture_bindings.len(), 1);
        assert_eq!(compiled.texture_bindings[0].name, "tDiffuse");
        assert!(compiled.texture_bindings[0].is_diffuse);
        assert_eq!(compiled.texture_bindings[0].group, 0);
        assert_eq!(compiled.texture_bindings[0].texture_binding, 1);
        assert_eq!(compiled.texture_bindings[0].sampler_binding, 2);

        // Verify exact std140 layout offsets
        let layout = &compiled.uniform_layout;
        assert_eq!(layout.set, 0);
        assert_eq!(layout.binding, 0);
        assert_eq!(layout.field("resolution").unwrap().offset, 0);
        assert_eq!(layout.field("Color").unwrap().offset, 16);
        assert_eq!(layout.field("Opacity").unwrap().offset, 28);
        assert_eq!(layout.field("Position").unwrap().offset, 32);
        assert_eq!(layout.field("Rotation").unwrap().offset, 40);
        assert_eq!(layout.field("Scale").unwrap().offset, 48);
        assert_eq!(layout.field("Invert").unwrap().offset, 56);
        assert_eq!(layout.field("Mode").unwrap().offset, 60);
        assert_eq!(layout.byte_size, 64);
    }

    #[test]
    fn compiles_extra_sampler2d_and_zoidium_gradient_include_with_palette() {
        let src = r#"
            // #include <zoidium/gradient>
            precision mediump float;

            varying vec2 vUvScaled;

            uniform sampler2D OverlayTex;
            uniform sampler2D tDiffuse;
            uniform sampler2D MyGrad;
            uniform float MyGrad_repetitions;
            uniform vec4 Colors[16];
            uniform int Colors_count;

            void main() {
                vec4 bg = texture2D(tDiffuse, vUvScaled);
                vec4 ov = texture2D(OverlayTex, vUvScaled);
                vec4 g = zSampleGradient(MyGrad, vUvScaled.x, MyGrad_repetitions);
                vec3 lin = zSrgbToLinear(g.rgb);
                vec3 srgb = zLinearToSrgb(lin);
                vec4 firstSwatch = Colors[0];
                float weight = Colors_count > 0 ? 0.5 : 0.0;
                vec3 fg = mix(srgb, firstSwatch.rgb, weight) * ov.a;
                float a = clamp(g.a * ov.a, 0.0, 1.0);
                gl_FragColor = vec4(fg * a + bg.rgb * (1.0 - a), a + bg.a * (1.0 - a));
            }
        "#;

        let compiled = translate_glsl_es100(src).expect("gradient + palette shader must compile");

        // `tDiffuse` is always assigned bindings 1/2 first, followed by extras in declaration order.
        assert_eq!(compiled.texture_bindings.len(), 3);
        assert_eq!(compiled.texture_bindings[0].name, "tDiffuse");
        assert_eq!(compiled.texture_bindings[0].texture_binding, 1);
        assert_eq!(compiled.texture_bindings[0].sampler_binding, 2);
        assert_eq!(compiled.texture_bindings[1].name, "OverlayTex");
        assert_eq!(compiled.texture_bindings[1].texture_binding, 3);
        assert_eq!(compiled.texture_bindings[1].sampler_binding, 4);
        assert_eq!(compiled.texture_bindings[2].name, "MyGrad");
        assert_eq!(compiled.texture_bindings[2].texture_binding, 5);
        assert_eq!(compiled.texture_bindings[2].sampler_binding, 6);

        let colors_field = compiled.uniform_layout.field("Colors").unwrap();
        assert_eq!(colors_field.offset, 16);
        assert_eq!(colors_field.array_len, Some(16));
        assert_eq!(colors_field.array_stride, 16);
        assert_eq!(colors_field.size, 256);

        let count_field = compiled.uniform_layout.field("Colors_count").unwrap();
        assert_eq!(count_field.offset, 272);
        assert_eq!(compiled.uniform_layout.byte_size, 288);
    }

    #[test]
    fn compiles_curve_and_text_includes_and_static_defines() {
        let src = r#"
            #version 100
            #include <zoidium/curve>
            #include <zoidium/text>
            precision highp float;

            varying vec2 vUvScaled;
            uniform sampler2D tDiffuse;
            uniform sampler2D MyCurve;
            uniform vec2 MyCurve_range;
            uniform sampler2D TitleTex;

            void main() {
                vec4 bg = texture2D(tDiffuse, vUvScaled);
                float c = zSampleCurveValue(MyCurve, vUvScaled.x);
                float mapped = zCurveMapRange(c, MyCurve_range);
                float sdf = zSampleText(TitleTex, vUvScaled).a;
                #if RENDER_MODE == 1
                    float a = zTextSdfAlphaAA(sdf, 0.5);
                #else
                    float a = zTextSdfAlpha(sdf, 0.5, 0.05);
                #endif
                vec3 rgb = vec3(mapped) * a + bg.rgb * (1.0 - a);
                gl_FragColor = vec4(rgb, a + bg.a * (1.0 - a));
            }
        "#;

        let mut opts = CompileOptions::default();
        opts.defines
            .insert("RENDER_MODE".to_string(), "1".to_string());

        let compiled = translate_glsl_es100_with_options(src, &opts)
            .expect("curve + text shader must compile");
        assert!(compiled.wgsl.contains("@fragment"));
        assert_eq!(compiled.texture_bindings.len(), 3);
    }

    #[test]
    fn respects_local_shadowing_and_struct_member_access() {
        let src = r#"
            precision highp float;
            varying vec2 vUvScaled;
            uniform vec3 Color;
            uniform float Opacity;

            struct Surface {
                vec3 Color;
                float Opacity;
            };

            vec3 tint(vec3 Color) {
                return Color * 0.5;
            }

            void main() {
                Surface s;
                s.Color = Color;
                s.Opacity = Opacity;
                float local = s.Opacity;
                vec3 shaded = tint(s.Color);
                gl_FragColor = vec4(shaded * local, local);
            }
        "#;

        let compiled = translate_glsl_es100(src).expect("shadowed shader must compile");
        assert!(compiled.glsl450_source.contains("s.Color = _zc_u.Color;"));
        assert!(
            compiled
                .glsl450_source
                .contains("s.Opacity = _zc_u.Opacity;")
        );
        assert!(compiled.glsl450_source.contains("return Color * 0.5;"));
    }

    #[test]
    fn compiles_pure_generator_without_uniforms_or_textures() {
        let src = r#"
            #version 100
            precision highp float;
            varying vec2 vUv;
            void main() {
                gl_FragColor = vec4(vUv, 0.5, 1.0);
            }
        "#;

        let compiled = translate_glsl_es100(src).expect("uniform-free shader must compile");
        assert_eq!(compiled.uniform_layout.byte_size, 0);
        assert!(compiled.uniform_layout.fields.is_empty());
        assert!(compiled.texture_bindings.is_empty());
        assert!(!compiled.glsl450_source.contains("ZcCustomUniforms"));
    }

    #[test]
    fn rejects_unsupported_webgl1_extensions_and_features() {
        for ext in ["GL_EXT_frag_depth", "GL_EXT_draw_buffers"] {
            let src =
                format!("#extension {ext} : require\nvoid main() {{ gl_FragColor = vec4(1.0); }}");
            let err = translate_glsl_es100(&src).unwrap_err();
            assert_eq!(err, ShaderError::UnsupportedExtension(ext.to_string()));
        }

        let depth_src = "void main() { gl_FragDepthEXT = 0.5; gl_FragColor = vec4(1.0); }";
        assert_eq!(
            translate_glsl_es100(depth_src).unwrap_err(),
            ShaderError::UnsupportedFeature("gl_FragDepthEXT".to_string())
        );

        let frag_data_src = "void main() { gl_FragData[0] = vec4(1.0); }";
        assert_eq!(
            translate_glsl_es100(frag_data_src).unwrap_err(),
            ShaderError::UnsupportedFeature("gl_FragData".to_string())
        );

        // Commented-out unsupported extensions must NOT be rejected.
        let commented = r#"
            /* #extension GL_EXT_frag_depth : require */
            void main() { gl_FragColor = vec4(1.0); }
        "#;
        assert!(translate_glsl_es100(commented).is_ok());
    }

    #[test]
    fn packs_std140_uniforms_with_exact_offsets_and_coercions() {
        let src = r#"
            uniform float Gain;
            uniform vec2 Offset;
            uniform vec3 Tint;
            uniform bool Enabled;
            uniform int Steps;
            uniform float Weights[3];
            uniform vec4 Colors[16];
            void main() {
                float w = Weights[1];
                vec3 c = Enabled ? Tint * Gain : vec3(0.0);
                gl_FragColor = vec4(c * w + Colors[0].rgb + vec3(Offset, float(Steps)), 1.0);
            }
        "#;

        let compiled = translate_glsl_es100(src).expect("packing test shader must compile");
        let layout = &compiled.uniform_layout;

        // Gain: 0 (size 4)
        // Offset: 8 (align 8, size 8)
        // Tint: 16 (align 16, size 12)
        // Enabled: 28 (align 4, size 4 -> packs tightly right after vec3!)
        // Steps: 32 (align 4, size 4)
        // Weights[3]: 48 (align 16, stride 16, size 48)
        // Colors[16]: 96 (align 16, stride 16, size 256)
        // Total: 352
        assert_eq!(layout.field("Gain").unwrap().offset, 0);
        assert_eq!(layout.field("Offset").unwrap().offset, 8);
        assert_eq!(layout.field("Tint").unwrap().offset, 16);
        assert_eq!(layout.field("Enabled").unwrap().offset, 28);
        assert_eq!(layout.field("Steps").unwrap().offset, 32);
        assert_eq!(layout.field("Weights").unwrap().offset, 48);
        assert_eq!(layout.field("Weights").unwrap().array_stride, 16);
        assert_eq!(layout.field("Colors").unwrap().offset, 96);
        assert_eq!(layout.byte_size, 352);

        let palette =
            PaletteSpec::try_new(vec![[1.0, 0.25, 0.5, 1.0], [0.0, 0.75, 1.0, 0.5]]).unwrap();

        let mut values = BTreeMap::new();
        values.insert("Gain".to_string(), UniformValue::Float(2.5));
        values.insert("Offset".to_string(), UniformValue::Vec2([0.25, -0.5]));
        values.insert("Tint".to_string(), UniformValue::Vec3([0.2, 0.4, 0.6]));
        values.insert("Enabled".to_string(), UniformValue::Float(1.0)); // float -> bool coercion
        values.insert("Steps".to_string(), UniformValue::Float(7.0)); // float -> int coercion
        values.insert(
            "Weights".to_string(),
            UniformValue::Array(vec![
                UniformValue::Float(0.1),
                UniformValue::Float(0.5),
                UniformValue::Float(0.9),
            ]),
        );
        for (k, v) in palette.to_uniform_values("Colors") {
            values.insert(k, v);
        }

        let bytes = pack_uniforms(layout, &values);
        assert_eq!(bytes.len(), 352);
        assert!((read_f32_le(&bytes, 0) - 2.5).abs() < 1e-6);
        assert!((read_f32_le(&bytes, 8) - 0.25).abs() < 1e-6);
        assert!((read_f32_le(&bytes, 12) + 0.5).abs() < 1e-6);
        assert!((read_f32_le(&bytes, 16) - 0.2).abs() < 1e-6);
        assert!((read_f32_le(&bytes, 20) - 0.4).abs() < 1e-6);
        assert!((read_f32_le(&bytes, 24) - 0.6).abs() < 1e-6);
        assert_eq!(read_u32_le(&bytes, 28), 1);
        assert_eq!(read_i32_le(&bytes, 32), 7);
        assert!((read_f32_le(&bytes, 48) - 0.1).abs() < 1e-6);
        assert!((read_f32_le(&bytes, 64) - 0.5).abs() < 1e-6);
        assert!((read_f32_le(&bytes, 80) - 0.9).abs() < 1e-6);
        assert!((read_f32_le(&bytes, 96) - 1.0).abs() < 1e-6);
        assert!((read_f32_le(&bytes, 100) - 0.25).abs() < 1e-6);
        assert!((read_f32_le(&bytes, 116) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn parses_and_evaluates_visibility_conditions_and_rename_rules() {
        let cond = VisibilityCondition::parse("Mode", "0; >=3; -5..-2").unwrap();
        assert!(cond.matches_value(0.0));
        assert!(!cond.matches_value(1.0));
        assert!(!cond.matches_value(2.0));
        assert!(cond.matches_value(3.0));
        assert!(cond.matches_value(10.0));
        assert!(cond.matches_value(-4.0));
        assert!(!cond.matches_value(-1.0));

        let ops = VisibilityCondition::parse("Level", "=2; !=5").unwrap();
        assert!(ops.matches_value(2.0));
        assert!(ops.matches_value(4.0));
        assert!(!ops.matches_value(5.0));

        let lt = VisibilityCondition::parse("X", "<1.5; >4.5").unwrap();
        assert!(lt.matches_value(1.0));
        assert!(!lt.matches_value(1.5));
        assert!(!lt.matches_value(3.0));
        assert!(lt.matches_value(5.0));

        let lte = VisibilityCondition::parse("X", "<=1.5").unwrap();
        assert!(lte.matches_value(1.5));
        assert!(!lte.matches_value(1.6));

        let rules = vec![
            RenameRule::parse("Mode", "1", "Radial Strength").unwrap(),
            RenameRule::parse("Mode", "2..4", "Angular Spread").unwrap(),
        ];

        let mut vals = BTreeMap::new();
        vals.insert("Mode".to_string(), 0.0);
        assert_eq!(resolve_label("Amount", &rules, &vals), "Amount");
        vals.insert("Mode".to_string(), 1.0);
        assert_eq!(resolve_label("Amount", &rules, &vals), "Radial Strength");
        vals.insert("Mode".to_string(), 3.0);
        assert_eq!(resolve_label("Amount", &rules, &vals), "Angular Spread");

        assert!(VisibilityCondition::parse("Mode", "5..2").is_err());
        assert!(VisibilityCondition::parse("Mode", "").is_err());
    }

    #[test]
    fn bakes_gradient_rgba8_across_spaces_modes_and_cyclic_wrap() {
        // Linear sRGB black -> white
        let srgb_spec = GradientSpec {
            stops: vec![
                GradientStop::new(0.0, [0.0, 0.0, 0.0, 0.0]),
                GradientStop::new(1.0, [1.0, 1.0, 1.0, 1.0]),
            ],
            interpolation: GradientInterpolation::Linear,
            color_space: GradientColorSpace::Srgb,
            cyclic: false,
            repetitions: 1.0,
        };
        let srgb_lut = bake_gradient_rgba8(&srgb_spec);
        assert_eq!(srgb_lut.len(), GRADIENT_LUT_WIDTH * 4);
        assert_eq!(&srgb_lut[0..4], &[0, 0, 0, 0]);
        assert_eq!(&srgb_lut[1020..1024], &[255, 255, 255, 255]);
        // Straight alpha midpoint: RGB ~ 128, A ~ 128
        assert!((i32::from(srgb_lut[128 * 4]) - 128).abs() <= 1);
        assert!((i32::from(srgb_lut[128 * 4 + 3]) - 128).abs() <= 1);

        // LinearRgb midpoint between 0 and 1 is 0.5^(1/2.4) ~= 0.735 -> ~188 in sRGB
        let linear_spec = GradientSpec {
            color_space: GradientColorSpace::LinearRgb,
            ..srgb_spec.clone()
        };
        let linear_lut = bake_gradient_rgba8(&linear_spec);
        assert!(linear_lut[128 * 4] > 180 && linear_lut[128 * 4] < 195);

        // OkLab perceptual midpoint between red and green is brighter/cleaner than sRGB gray
        let oklab_spec = GradientSpec {
            stops: vec![
                GradientStop::new(0.0, [1.0, 0.0, 0.0, 1.0]),
                GradientStop::new(1.0, [0.0, 1.0, 0.0, 1.0]),
            ],
            interpolation: GradientInterpolation::Linear,
            color_space: GradientColorSpace::OkLab,
            cyclic: false,
            repetitions: 1.0,
        };
        let oklab_lut = bake_gradient_rgba8(&oklab_spec);
        assert_eq!(&oklab_lut[0..4], &[255, 0, 0, 255]);
        assert_eq!(&oklab_lut[1020..1024], &[0, 255, 0, 255]);
        assert!(oklab_lut[128 * 4] > 160 && oklab_lut[128 * 4 + 1] > 160);

        // Constant interpolation holds first stop until second stop
        let const_spec = GradientSpec {
            stops: vec![
                GradientStop::new(0.0, [1.0, 0.0, 0.0, 1.0]),
                GradientStop::new(0.5, [0.0, 0.0, 1.0, 1.0]),
            ],
            interpolation: GradientInterpolation::Constant,
            color_space: GradientColorSpace::Srgb,
            cyclic: false,
            repetitions: 1.0,
        };
        let const_lut = bake_gradient_rgba8(&const_spec);
        assert_eq!(&const_lut[64 * 4..64 * 4 + 4], &[255, 0, 0, 255]);
        assert_eq!(&const_lut[200 * 4..200 * 4 + 4], &[0, 0, 255, 255]);

        // Ease & Cardinal interpolation stay smooth and bounded
        let ease_spec = GradientSpec {
            interpolation: GradientInterpolation::Ease,
            ..srgb_spec.clone()
        };
        let ease_lut = bake_gradient_rgba8(&ease_spec);
        // Smoothstep at t=0.25 is 0.15625 (< 0.25 linear)
        assert!(ease_lut[64 * 4] < srgb_lut[64 * 4]);

        let cardinal_spec = GradientSpec {
            stops: vec![
                GradientStop::new(0.0, [0.0, 0.0, 0.0, 1.0]),
                GradientStop::new(0.33, [0.8, 0.2, 0.1, 1.0]),
                GradientStop::new(0.66, [0.1, 0.8, 0.4, 1.0]),
                GradientStop::new(1.0, [1.0, 1.0, 1.0, 1.0]),
            ],
            interpolation: GradientInterpolation::Cardinal,
            color_space: GradientColorSpace::Srgb,
            cyclic: true,
            repetitions: 1.0,
        };
        let cardinal_lut = bake_gradient_rgba8(&cardinal_spec);
        assert_eq!(cardinal_lut.len(), GRADIENT_LUT_WIDTH * 4);

        // Cyclic wrap connects last stop back to first stop seamlessly
        let cyclic_spec = GradientSpec {
            stops: vec![
                GradientStop::new(0.25, [1.0, 0.0, 0.0, 1.0]),
                GradientStop::new(0.75, [0.0, 0.0, 1.0, 1.0]),
            ],
            interpolation: GradientInterpolation::Linear,
            color_space: GradientColorSpace::Srgb,
            cyclic: true,
            repetitions: 1.0,
        };
        let cyclic_lut = bake_gradient_rgba8(&cyclic_spec);
        // At t = 0.0 (exact midpoint of the wrap segment from 0.75 -> 1.25), R ~ 128, B ~ 128
        assert!((i32::from(cyclic_lut[0]) - 128).abs() <= 1);
        assert!((i32::from(cyclic_lut[2]) - 128).abs() <= 1);
    }

    #[test]
    fn bakes_curve_rgba8_single_and_composite_rgba_channels() {
        // Default RGBA identity LUT
        let id_lut = bake_curve_rgba8(&CurveSpec::default());
        assert_eq!(id_lut.len(), CURVE_LUT_WIDTH * 4);
        assert_eq!(&id_lut[0..4], &[0, 0, 0, 0]);
        assert_eq!(&id_lut[512 * 4..512 * 4 + 4], &[128, 128, 128, 128]);
        assert_eq!(&id_lut[1023 * 4..1023 * 4 + 4], &[255, 255, 255, 255]);

        // Composite master (lift midtone to 0.75) + per-channel Red inversion + independent Alpha
        let rgba_spec = CurveSpec {
            mode: CurveMode::Rgba,
            master: CurveChannel {
                points: vec![
                    CurvePoint::new(0.0, 0.0),
                    CurvePoint::new(0.5, 0.75),
                    CurvePoint::new(1.0, 1.0),
                ],
            },
            red: Some(CurveChannel {
                points: vec![CurvePoint::new(0.0, 1.0), CurvePoint::new(1.0, 0.0)],
            }),
            green: None,
            blue: None,
            alpha: Some(CurveChannel {
                points: vec![CurvePoint::new(0.0, 0.5), CurvePoint::new(1.0, 0.5)],
            }),
            range: [0.0, 1.0],
        };
        let rgba_lut = bake_curve_rgba8(&rgba_spec);
        let mid = 512 * 4;
        // master(0.5) ~= 0.75 -> G,B ~= 191; red(0.75) = 0.25 -> R ~= 64; alpha(0.5) = 0.5 -> A = 128
        assert!((i32::from(rgba_lut[mid]) - 64).abs() <= 2);
        assert!((i32::from(rgba_lut[mid + 1]) - 191).abs() <= 2);
        assert!((i32::from(rgba_lut[mid + 2]) - 191).abs() <= 2);
        assert_eq!(rgba_lut[mid + 3], 128);

        // Single mode with explicit cubic Bezier handles
        let bezier_spec = CurveSpec {
            mode: CurveMode::Single,
            master: CurveChannel {
                points: vec![
                    CurvePoint::with_tangents(0.0, 0.0, None, Some([0.4, 0.0])),
                    CurvePoint::with_tangents(1.0, 1.0, Some([-0.4, 0.0]), None),
                ],
            },
            red: None,
            green: None,
            blue: None,
            alpha: None,
            range: [0.0, 1.0],
        };
        let bezier_lut = bake_curve_rgba8(&bezier_spec);
        // Zoidium CurveMode::Single writes [common, common, common, common]
        assert_eq!(bezier_lut[3], bezier_lut[0]);
        assert_eq!(bezier_lut[512 * 4 + 3], bezier_lut[512 * 4]);
        assert!((i32::from(bezier_lut[512 * 4]) - 128).abs() <= 2);
        // Ease-in-out S-curve is below linear at x = 0.25
        assert!(bezier_lut[256 * 4] < 64);
    }

    #[test]
    fn enforces_palette_limit_of_16_colors() {
        let too_many = vec![[1.0, 1.0, 1.0, 1.0]; 17];
        assert!(matches!(
            PaletteSpec::try_new(too_many.clone()),
            Err(ShaderPlusError::PaletteTooLarge { len: 17, max: 16 })
        ));

        let clamped = PaletteSpec::new(too_many);
        assert_eq!(clamped.len(), MAX_PALETTE_COLORS);
        let (arr, count) = clamped.as_uniform_array();
        assert_eq!(count, 16);
        assert!((arr[15][0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn escapes_glsl450_keywords_and_texture_identifier_without_changing_api_names() {
        let sampler_named_texture = r#"
            precision highp float;
            varying vec2 vUvScaled;
            uniform sampler2D texture;
            uniform float smooth;
            uniform float sample;
            uniform float noperspective;

            struct Data {
                vec4 sample;
                float smooth;
            };

            vec4 sampleHelper(sampler2D tex, vec2 uv) {
                return texture2D(tex, uv);
            }

            void main() {
                vec4 a = texture2D(texture, vUvScaled);
                vec4 b = sampleHelper(texture, vUvScaled + vec2(0.01));
                float patch = smooth + sample + noperspective;
                float buffer = 0.25, shared = 0.5, readonly = 0.75, writeonly = 1.0;
                float coherent = 0.1, volatile = 0.2, restrict = 0.3, subroutine = 0.4;
                Data d;
                d.sample = mix(a, b, 0.5);
                d.smooth = patch + buffer + shared + readonly + writeonly + coherent + volatile + restrict + subroutine;
                gl_FragColor = d.sample * clamp(d.smooth, 0.0, 1.0);
            }
        "#;
        let compiled = translate_glsl_es100(sampler_named_texture)
            .expect("shader with GLSL 450 keyword identifiers and sampler `texture` must compile");
        assert_eq!(compiled.texture_bindings.len(), 1);
        assert_eq!(compiled.texture_bindings[0].name, "texture");
        assert!(compiled.uniform_layout.field("smooth").is_some());
        assert!(compiled.uniform_layout.field("sample").is_some());
        assert!(compiled.uniform_layout.field("noperspective").is_some());

        let local_var_named_texture = r#"
            precision highp float;
            varying vec2 vUvScaled;
            uniform sampler2D tDiffuse;
            void main() {
                vec4 texture = texture2D(tDiffuse, vUvScaled);
                vec4 sample = texture2D(tDiffuse, vUvScaled + vec2(0.01));
                gl_FragColor = mix(texture, sample, 0.5);
            }
        "#;
        translate_glsl_es100(local_var_named_texture)
            .expect("local `vec4 texture` before `texture2D()` must compile");
    }

    #[test]
    fn compiles_non_sampler_uniform_named_texture() {
        let src = "uniform float texture; void main() { gl_FragColor = vec4(texture); }";
        let compiled = translate_glsl_es100(src)
            .expect("non-sampler uniform named `texture` must compile through Naga");
        assert!(compiled.uniform_layout.field("texture").is_some());
        assert!(compiled.glsl450_source.contains("_zc_u._zc_id_texture"));
        assert!(compiled.wgsl.contains("@fragment"));
    }

    #[test]
    fn evaluates_preprocessor_conditionals_before_extraction_and_validation() {
        let src = r#"
            precision highp float;
            varying vec2 vUvScaled;
            uniform sampler2D tDiffuse;

            #if MODE == 1
                uniform vec3 Tint;
                #define ARR_LEN 4
            #else
                uniform vec4 Tint;
                uniform sampler2D GhostTex;
                #define ARR_LEN 8
                #extension GL_EXT_frag_depth : require
                void deadFunc() { gl_FragDepthEXT = 0.5; }
            #endif

            uniform float Weights[ARR_LEN];

            #if defined(MODE) && MODE == 1
            vec3 pickTint() {
            #else
            vec3 pickTint() {
            #endif
                return Tint * Weights[0];
            }

            void main() {
                vec4 bg = texture2D(tDiffuse, vUvScaled);
                gl_FragColor = vec4(bg.rgb * pickTint(), bg.a);
            }
        "#;

        let mut opts = CompileOptions::default();
        opts.defines.insert("MODE".to_string(), "1".to_string());
        let compiled = translate_glsl_es100_with_options(src, &opts)
            .expect("conditional branches must be evaluated before extraction/validation");

        assert_eq!(compiled.texture_bindings.len(), 1);
        assert_eq!(compiled.texture_bindings[0].name, "tDiffuse");
        let tint = compiled.uniform_layout.field("Tint").unwrap();
        assert_eq!(tint.ty, UniformType::Vec3);
        let weights = compiled.uniform_layout.field("Weights").unwrap();
        assert_eq!(weights.array_len, Some(4));
    }

    #[test]
    fn expands_sampling_macros_sampler_aliases_parens_and_include_helpers() {
        let src = r#"
            # include <zoidium/gradient> // trailing comment on spaced # include
            precision highp float;
            varying vec2 vUvScaled;
            uniform sampler2D tDiffuse;
            uniform sampler2D MyGrad;
            uniform float MyGrad_repetitions;

            #define iChannel0 tDiffuse
            #define TEX (tDiffuse)
            #define SAMPLE(tex, uv) texture2D(tex, uv)
            #define MY_GRAD(t) zSampleGradient(MyGrad, t, MyGrad_repetitions)

            void main() {
                vec4 a = SAMPLE(iChannel0, vUvScaled);
                vec4 b = texture2D(TEX, vUvScaled);
                vec4 c = SAMPLE(tDiffuse, SAMPLE(tDiffuse, vUvScaled).xy);
                vec4 g = MY_GRAD(vUvScaled.x);
                gl_FragColor = (a + b + c + g) * 0.25;
            }
        "#;

        let compiled = translate_glsl_es100(src).expect(
            "sampling macros, aliases, parens, and include helpers inside #define must compile",
        );
        assert!(compiled.wgsl.contains("@fragment"));
        assert_eq!(compiled.texture_bindings.len(), 2);
    }

    #[test]
    fn handles_block_comment_reopen_directive_comments_and_extension_behaviors() {
        let src = r#"
            /* header */ #version 100
            /* first
            */ /* reopened block comment
            #extension GL_EXT_frag_depth : require
            */
            #extension all : disable
            #extension GL_EXT_shader_texture_lod : warn
            #extension GL_OES_standard_derivatives : enable
            #define GAIN /* http://example.com/path */ 0.5

            #ifndef GL_OES_standard_derivatives
                #error GL_OES_standard_derivatives must be defined
            #endif

            precision highp float;
            varying vec2 vUv;

            void main() {
                float dy = dFdy(vUv.y);
                float fw = fwidth(vUv.x);
                gl_FragColor = vec4(abs(dy) * GAIN, fw, 0.0, 1.0);
            }
        "#;

        let compiled = translate_glsl_es100(src)
            .expect("block comments, URL in #define comment, dFdy, and disable/warn extensions must compile");
        assert!(
            compiled
                .glsl450_source
                .contains("#define GL_OES_standard_derivatives 1")
        );
        assert!(compiled.glsl450_source.contains("(- dFdy(vUv.y))"));
    }

    #[test]
    fn supports_invariant_varying_array_varying_const_expr_arrays_and_gradient_uv_bounds() {
        let src = r#"
            #include <zoidium/gradient>
            #include <zoidium/curve>
            precision highp float;

            invariant varying vec2 vUv;
            varying vec2 vOffsets[2];

            #define BASE 0x4u//inline comment
            uniform float HexArr[0x4];
            uniform vec4 ExprArr[(BASE) * 2 + 1];
            uniform sampler2D GradTex;
            uniform sampler2D CurveTex;

            void main() {
                vec4 g = zSampleGradient(GradTex, 1.0, 1.0);
                float c = zSampleCurveValue(CurveTex, 1.0);
                gl_FragColor = g * c + ExprArr[0] * HexArr[0] + vec4(vOffsets[0], vOffsets[1]);
            }
        "#;

        let compiled = translate_glsl_es100(src)
            .expect("invariant varying, array varying, hex/expr arrays must compile");
        assert_eq!(
            compiled.uniform_layout.field("HexArr").unwrap().array_len,
            Some(4)
        );
        assert_eq!(
            compiled.uniform_layout.field("ExprArr").unwrap().array_len,
            Some(9)
        );
        assert!(compiled.glsl450_source.contains("in vec2 vOffsets[2];"));
        assert!(
            compiled
                .glsl450_source
                .contains("reps <= 1.0 ? clamp(scaled, 0.0, 1.0) : fract(scaled)")
        );
        assert!(
            compiled
                .glsl450_source
                .contains("(0.5 + clamp(u, 0.0, 1.0) * 255.0) / 256.0")
        );
        assert!(
            compiled
                .glsl450_source
                .contains("(0.5 + clamp(t, 0.0, 1.0) * 1023.0) / 1024.0")
        );
    }

    fn assert_valid_wgsl(compiled: &CompiledCustomShader) {
        let module = naga::front::wgsl::parse_str(&compiled.wgsl).unwrap_or_else(|e| {
            panic!(
                "WGSL re-parse failed: {}\n---\n{}",
                e.emit_to_string(&compiled.wgsl),
                compiled.wgsl
            )
        });
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("WGSL re-validation failed: {e}\n---\n{}", compiled.wgsl));
    }

    #[test]
    fn bake_curve_preserves_normalized_lut_regardless_of_range_and_supports_rgba16f() {
        let spec = CurveSpec {
            mode: CurveMode::Rgba,
            master: CurveChannel::identity(),
            red: Some(CurveChannel {
                points: vec![CurvePoint::new(0.0, 1.0), CurvePoint::new(1.0, 0.0)],
            }),
            green: None,
            blue: None,
            alpha: Some(CurveChannel {
                points: vec![CurvePoint::new(0.0, 0.2), CurvePoint::new(1.0, 0.2)],
            }),
            range: [-10.0, 100.0],
        };
        let lut8 = bake_curve_rgba8(&spec);
        let lut32 = bake_curve_rgba32f(&spec);
        let lut16 = bake_curve_rgba16f(&spec);
        assert_eq!(lut8.len(), CURVE_LUT_WIDTH * 4);
        assert_eq!(lut32.len(), CURVE_LUT_WIDTH * 4);
        assert_eq!(lut16.len(), CURVE_LUT_WIDTH * 4);
        // Range is applied in GLSL via mix(range.x, range.y, texel), NOT during baking!
        assert_eq!(lut8[0], 255);
        assert_eq!(lut8[3], 51);
        assert_eq!(lut8[(CURVE_LUT_WIDTH - 1) * 4], 0);
        assert!((lut32[3] - 0.2).abs() < 1e-6);
        assert_eq!(f32_to_f16_bits(0.0), 0);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(lut16[0], 0x3c00);
    }

    #[test]
    fn compiles_all_nine_shader_plus_and_expressions_plus_templates_to_valid_wgsl() {
        let templates: &[(&str, &str)] = &[
            (
                "passthrough",
                "precision highp float;\nuniform sampler2D tDiffuse;\nvarying vec2 vUvScaled;\nvoid main() {\n  gl_FragColor = texture2D(tDiffuse, vUvScaled);\n}",
            ),
            (
                "color",
                "precision highp float;\nuniform sampler2D tDiffuse;\nuniform float Amount;\nvarying vec2 vUvScaled;\nvoid main() {\n  vec4 color = texture2D(tDiffuse, vUvScaled);\n  gl_FragColor = vec4(color.rgb * (0.5 + Amount), color.a);\n}",
            ),
            (
                "gradient",
                "precision highp float;\n// #include <zoidium/gradient>\nuniform sampler2D tDiffuse;\nuniform sampler2D Ramp;\nvarying vec2 vUvScaled;\nvoid main() {\n  vec4 color = texture2D(tDiffuse, vUvScaled);\n  float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));\n  gl_FragColor = vec4(zGradient(Ramp, luma).rgb, color.a);\n}",
            ),
            (
                "curve",
                "precision highp float;\n// #include <zoidium/curve>\nuniform sampler2D tDiffuse;\nuniform sampler2D Remap;\nvarying vec2 vUvScaled;\nvoid main() {\n  vec4 color = texture2D(tDiffuse, vUvScaled);\n  gl_FragColor = vec4(zCurves(Remap, color.rgb), color.a);\n}",
            ),
            (
                "noise",
                "precision highp float;\n// #include <zoidium/noise>\nuniform float zTime;\nvarying vec2 vUv;\nvoid main() {\n  float n = zNoise(vec3(vUv * 6.0, zTime));\n  gl_FragColor = vec4(vec3(n), 1.0);\n}",
            ),
            (
                "text",
                "precision highp float;\n// #include <zoidium/text>\nuniform sampler2D tDiffuse;\nuniform sampler2D Title;\nvarying vec2 vUvScaled;\nvarying vec2 vUv;\nvoid main() {\n  vec4 color = texture2D(tDiffuse, vUvScaled);\n  float coverage = zText(Title, vUv);\n  gl_FragColor = vec4(mix(color.rgb, vec3(1.0), coverage), max(color.a, coverage));\n}",
            ),
            (
                "velocity",
                "precision highp float;\nuniform sampler2D tDiffuse;\nuniform vec2 Velocity;\nvarying vec2 vUvScaled;\nvoid main() {\n  gl_FragColor = texture2D(tDiffuse, vUvScaled + Velocity * 0.01);\n}",
            ),
            (
                "time-motion",
                "precision highp float;\n// #include <zoidium/temporal>\nuniform sampler2D Before;\nuniform sampler2D After;\nuniform vec2 resolution;\nuniform vec2 uvScale;\nvarying vec2 vUvScaled;\nvoid main() {\n  vec2 motion = zTimeMotion(Before, After, vUvScaled, uvScale / resolution);\n  gl_FragColor = vec4(vec2(0.5) + motion * 20.0, 0.5, 1.0);\n}",
            ),
            (
                "time-jitter",
                "precision highp float;\nuniform sampler2D Jitter;\nvarying vec2 vUvScaled;\nvoid main() {\n  gl_FragColor = texture2D(Jitter, vUvScaled);\n}",
            ),
        ];

        for &(id, src) in templates {
            let compiled = translate_glsl_es100(src)
                .unwrap_or_else(|e| panic!("template `{id}` failed to compile: {e}"));
            assert_valid_wgsl(&compiled);
        }
    }

    #[test]
    fn compiles_all_expressions_plus_and_shader_plus_includes_together() {
        let src = r#"
            precision highp float;
            // #include <zoidium/hash>
            // #include <zoidium/noise>
            // #include <zoidium/color>
            // #include <zoidium/sdf>
            // #include <zoidium/gradient>
            // #include <zoidium/curve>
            // #include <zoidium/text>
            // #include <zoidium/temporal>
            uniform sampler2D tDiffuse;
            uniform sampler2D Ramp;
            uniform sampler2D Remap;
            uniform sampler2D Title;
            uniform sampler2D Before;
            uniform sampler2D After;
            uniform float zTime;
            uniform float zFrame;
            uniform float zFps;
            uniform vec3 ctl_Tint;
            varying vec2 vUv;
            varying vec2 vUvScaled;
            void main() {
                float h = zHash(zFrame) + zHash(vUv) + zHash(vec3(vUv, zTime));
                float n = zNoise(vUv * 4.0) + zNoise(vec3(vUv, zTime / max(zFps, 1.0)));
                vec3 lin = zSrgbToLinear(ctl_Tint);
                vec3 srgb = zLinearToSrgb(lin);
                vec3 hsv = zRgbToHsv(srgb);
                vec3 rgb = zHsvToRgb(hsv);
                float d = zSmoothUnion(
                    zSdCircle(vUv - 0.5, 0.25),
                    zSdBox(vUv - 0.5, vec2(0.2, 0.1)),
                    0.05
                );
                float d3 = zSdSphere(vec3(vUv, 0.0), 0.5) + zSdBox(vec3(vUv, 0.0), vec3(0.2));
                vec4 g = zGradientCycle(Ramp, vUv.x + h * 0.01);
                float c1 = zCurve(Remap, clamp(n * 0.5, 0.0, 1.0));
                vec4 c4 = zCurves(Remap, g);
                float glyph = zGlyph(Title, vec2(4.0, 4.0), 2.0, vUv);
                vec2 mot = zTimeMotion(Before, After, vUvScaled, vec2(1.0 / 1280.0, 1.0 / 720.0));
                gl_FragColor = vec4(rgb * c1 + c4.rgb * glyph + vec3(mot, d + d3) * 0.001, 1.0);
            }
        "#;

        let bound_curve = bind_curve_ranges(src, ["Remap"]);
        assert!(bound_curve.contains("uniform vec2 Remap_range;"));
        assert!(bound_curve.contains("zCurve(Remap, clamp(n * 0.5, 0.0, 1.0), Remap_range)"));
        assert!(bound_curve.contains("zCurves(Remap, g, Remap_range)"));

        let bound_both = bind_gradient_cycles(&bound_curve, ["Ramp"]);
        assert!(bound_both.contains("uniform float Ramp_repetitions;"));
        assert!(bound_both.contains("zGradientCycle(Ramp, vUv.x + h * 0.01, Ramp_repetitions)"));

        let compiled = translate_glsl_es100(&bound_both)
            .expect("combined Shader+ & Expressions+ includes must compile");
        assert_valid_wgsl(&compiled);
    }

    #[test]
    fn compiles_node_effect_vector_math_4_and_out_of_domain_literal_math() {
        let src = r#"
            precision highp float;
            uniform sampler2D tDiffuse;
            uniform vec2 uvScale;
            uniform vec2 resolution;
            varying vec2 vUv;
            const vec3 UNIT_Z = normalize(vec3(0.0, 0.0, 2.0));
            const float ROOT_TWO = sqrt(2.0);
            void main() {
                vec4 n0 = vec4(length((vec4(0.0, 0.0, 0.0, 1.0)).rgb) > 0.0
                    ? normalize((vec4(0.0, 0.0, 0.0, 1.0)).rgb)
                    : vec3(0.0), 1.0);
                float p1 = false ? pow(-2.0, 2.0) : 1.0;
                float p2 = false ? pow(0.0, 0.0) : 1.0;
                vec4 src = texture2D(tDiffuse, vUv * uvScale);
                gl_FragColor = vec4(n0.rgb + UNIT_Z * 0.0 + src.rgb * (p1 * p2 * ROOT_TWO * 0.5), 1.0);
            }
        "#;
        let compiled = translate_glsl_es100(src)
            .expect("dead-branch normalize(vec3(0.0)) and pow literals must compile");
        assert_valid_wgsl(&compiled);
    }

    #[test]
    fn compiles_mat2_widened_array_args_const_int_arrays_bvec_struct_and_sampler_cube() {
        let src = r#"
            precision highp float;
            const int ARR_N = 3;
            struct Light {
                vec3 color;
                float power;
            };
            uniform mat2 Rot;
            uniform mat2 Mats[2];
            uniform float Weights[ARR_N];
            uniform vec2 Offsets[2];
            uniform bool Flags[2];
            uniform bvec2 B2;
            uniform bvec3 B3;
            uniform bvec4 B4;
            uniform Light uLight;
            uniform Light uLights[2];
            uniform samplerCube Env;
            varying vec2 vUv;

            float sum3(float a[3]) {
                return a[0] + a[1] + a[2];
            }
            vec2 sum2v(vec2 a[2]) {
                return a[0] + a[1];
            }
            bool any2(bool a[2]) {
                return a[0] || a[1];
            }
            vec3 evalLight(Light l) {
                return l.color * l.power;
            }
            vec4 sampleEnv(samplerCube c, vec3 dir) {
                return textureCube(c, dir);
            }

            void main() {
                const int LOCAL_N = 4;
                float scratch[LOCAL_N];
                scratch[0] = sum3(Weights);
                vec2 p = Rot * vUv + Mats[1] * sum2v(Offsets);
                vec3 lit = evalLight(uLight) + uLights[0].color;
                vec4 env = sampleEnv(Env, vec3(p, 1.0));
                float mask = (any2(Flags) && B2.x && B3.y && B4.z) ? 1.0 : 0.5;
                gl_FragColor = vec4((lit + env.rgb) * mask + vec3(scratch[0] * 0.0), 1.0);
            }
        "#;
        let compiled = translate_glsl_es100(src).expect("advanced uniform types must compile");
        assert_valid_wgsl(&compiled);

        // Verify mat2 + struct + bvec layout packing
        let mut vals = BTreeMap::new();
        vals.insert("Rot".to_string(), UniformValue::Mat2([1.0, 2.0, 3.0, 4.0]));
        vals.insert("B2".to_string(), UniformValue::Bvec2([true, false]));
        vals.insert(
            "uLight.color".to_string(),
            UniformValue::Vec3([0.2, 0.4, 0.6]),
        );
        vals.insert("uLight.power".to_string(), UniformValue::Float(2.0));
        let bytes = pack_uniforms(&compiled.uniform_layout, &vals);
        let rot = compiled.uniform_layout.field("Rot").unwrap();
        assert!((read_f32_le(&bytes, rot.offset as usize) - 1.0).abs() < 1e-6);
        assert!((read_f32_le(&bytes, rot.offset as usize + 4) - 2.0).abs() < 1e-6);
        assert!((read_f32_le(&bytes, rot.offset as usize + 16) - 3.0).abs() < 1e-6);
        assert!((read_f32_le(&bytes, rot.offset as usize + 20) - 4.0).abs() < 1e-6);
        let b2 = compiled.uniform_layout.field("B2").unwrap();
        assert_eq!(read_u32_le(&bytes, b2.offset as usize), 1);
        assert_eq!(read_u32_le(&bytes, b2.offset as usize + 4), 0);
    }

    #[test]
    fn normalizes_space_containing_property_names_in_pack_uniforms_and_defines() {
        let src = r#"
            precision highp float;
            uniform vec2 Corner_Point;
            uniform float Style_Amount;
            varying vec2 vUv;
            void main() {
                #if Render_Mode == 2
                    gl_FragColor = vec4(Corner_Point * Style_Amount, 0.0, 1.0);
                #else
                    gl_FragColor = vec4(0.0);
                #endif
            }
        "#;
        let mut opts = CompileOptions::default();
        opts.defines
            .insert("Render Mode".to_string(), "2".to_string());
        let compiled = translate_glsl_es100_with_options(src, &opts)
            .expect("space-containing define name must normalize");
        assert_valid_wgsl(&compiled);
        assert_eq!(fix_property_name("Corner Point"), "Corner_Point");

        let mut vals = BTreeMap::new();
        vals.insert("Corner Point".to_string(), UniformValue::Vec2([0.25, 0.75]));
        vals.insert("Style Amount".to_string(), UniformValue::Float(0.5));
        let bytes = pack_uniforms(&compiled.uniform_layout, &vals);
        let cp = compiled.uniform_layout.field("Corner_Point").unwrap();
        let sa = compiled.uniform_layout.field("Style_Amount").unwrap();
        assert!((read_f32_le(&bytes, cp.offset as usize) - 0.25).abs() < 1e-6);
        assert!((read_f32_le(&bytes, cp.offset as usize + 4) - 0.75).abs() < 1e-6);
        assert!((read_f32_le(&bytes, sa.offset as usize) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn supports_texture_lod_ext_point_coord_bg_coord_and_derivative_uniformity_diagnostic() {
        let src = r#"
            #extension GL_EXT_shader_texture_lod : require
            precision highp float;
            uniform sampler2D tDiffuse;
            varying vec2 vUv;
            varying vec2 bgCoord;
            void main() {
                vec4 a = texture2DLodEXT(tDiffuse, bgCoord, 1.0);
                vec4 b = texture2DGradEXT(tDiffuse, vUv, vec2(0.01, 0.0), vec2(0.0, 0.01));
                vec4 c = texture2DProjLodEXT(tDiffuse, vec3(vUv, 1.0), 0.5);
                vec4 d = texture2DProjGradEXT(tDiffuse, vec4(vUv, 0.0, 1.0), vec2(0.01, 0.0), vec2(0.0, 0.01));
                vec2 pc = gl_PointCoord;
                gl_FragColor = (a + b + c + d) * 0.25 + vec4(pc, 0.0, 0.0);
            }
        "#;
        let compiled = translate_glsl_es100(src)
            .expect("GL_EXT_shader_texture_lod, gl_PointCoord, and bgCoord must compile");
        assert!(compiled.glsl450_source.contains("_zc_flip_grad"));
        assert!(!compiled.glsl450_source.contains("gl_PointCoord"));
        assert!(!compiled.glsl450_source.contains("location = 2"));
        assert_valid_wgsl(&compiled);

        // Derivative uniformity diagnostic emitted when implicit derivatives/textureSample are used
        let sample_src = r#"
            uniform sampler2D tDiffuse;
            varying vec2 vUv;
            void main() {
                gl_FragColor = texture2D(tDiffuse, vUv);
            }
        "#;
        let sampled = translate_glsl_es100(sample_src).unwrap();
        assert!(
            sampled
                .wgsl
                .starts_with("diagnostic(off, derivative_uniformity);\n")
        );
        assert_valid_wgsl(&sampled);

        // Dead fallback loop with helper call after unconditional `return;` (AlipFX mosaic.glsl pattern)
        let mosaic_pattern = r#"
            #ifdef GL_EXT_shader_texture_lod
            #extension GL_EXT_shader_texture_lod : enable
            #endif
            precision highp float;
            uniform sampler2D tDiffuse;
            uniform vec2 resolution;
            varying vec2 vUvScaled;
            vec2 snap(vec2 uv) { return (floor(uv * resolution) + 0.5) / resolution; }
            void main() {
                #ifdef GL_EXT_shader_texture_lod
                    gl_FragColor = texture2DGradEXT(tDiffuse, vUvScaled, vec2(0.01, 0.0), vec2(0.0, 0.01));
                    return;
                #endif
                vec4 acc = vec4(0.0);
                for (int i = 0; i < 4; i++) {
                    acc += texture2D(tDiffuse, snap(vUvScaled));
                }
                gl_FragColor = acc * 0.25;
            }
        "#;
        let mosaic_compiled = translate_glsl_es100(mosaic_pattern)
            .expect("mosaic.glsl dead-code-after-return pattern must compile");
        assert_valid_wgsl(&mosaic_compiled);

        // Pure generator without textureSample / derivatives omits diagnostic
        let pure_src = "varying vec2 vUv; void main() { gl_FragColor = vec4(vUv, 0.0, 1.0); }";
        let pure = translate_glsl_es100(pure_src).unwrap();
        assert!(!pure.wgsl.contains("derivative_uniformity"));
        assert_valid_wgsl(&pure);
    }

    #[test]
    fn visibility_gradient_and_curve_match_zoidium_parity_rules() {
        // Visibility strict boundaries, newline split, string tokens, and rejecting `== 4`
        assert!(VisibilityCondition::parse("Mode", "== 4").is_err());
        assert!(VisibilityCondition::parse("Mode", "! 3").is_err());
        assert!(VisibilityCondition::parse("Mode", "1....2").is_err());

        let cond = VisibilityCondition::parse("Mode", "> 5.5\n1..3; = 4; off").unwrap();
        assert!(cond.matches_value(5.5001));
        assert!(!cond.matches_value(5.5));
        assert!(cond.matches_value(1.0));
        assert!(cond.matches_value(4.0));
        assert!(cond.matches_str("off"));
        assert!(!cond.matches_value(0.0));
        assert!(!cond.matches_value(f64::NAN));
        assert!(!cond.matches_value(f64::INFINITY));

        let static_dd = VisibilityCondition::parse("Mode", ">= 1").unwrap();
        assert!(static_dd.matches_static_dropdown("2"));
        assert!(!static_dd.matches_static_dropdown("0"));

        // Zoidium JSON struct round-trip (values array + conditions array)
        let zwhen = cond.to_zoidium();
        let back = VisibilityCondition::from_zoidium(&zwhen).unwrap();
        assert_eq!(back.to_zoidium(), zwhen);
        assert!(back.matches_value(5.5001));
        assert!(!back.matches_value(5.5));
        assert!(back.matches_value(2.0));
        assert!(back.matches_value(4.0));
        assert!(back.matches_str("off"));
        assert!(!back.matches_value(3.5));

        // Gradient empty stops -> opaque white, cyclic seam texel[0] == texel[255], midpoint piecewise-linear, opacityStops
        let empty = bake_gradient_rgba8(&GradientSpec {
            stops: vec![],
            ..GradientSpec::default()
        });
        assert_eq!(&empty[0..4], &[255, 255, 255, 255]);

        let cyclic = bake_gradient_rgba8(&GradientSpec {
            stops: vec![
                GradientStop::new(0.2, [1.0, 0.0, 0.0, 1.0]),
                GradientStop::new(0.8, [0.0, 0.0, 1.0, 1.0]),
            ],
            cyclic: true,
            ..GradientSpec::default()
        });
        assert_eq!(&cyclic[0..4], &cyclic[255 * 4..256 * 4]);

        let mut mid_stop = GradientStop::new(0.0, [0.0, 0.0, 0.0, 1.0]);
        mid_stop.midpoint = 0.25;
        let mid_lut = bake_gradient_rgba8_with_opacity(
            &GradientSpec {
                stops: vec![mid_stop, GradientStop::new(1.0, [1.0, 1.0, 1.0, 1.0])],
                ..GradientSpec::default()
            },
            &[OpacityStop::new(0.0, 0.0), OpacityStop::new(1.0, 1.0)],
        );
        // Piecewise-linear midpoint at t = 0.25 -> u = 0.5 -> R ~= 128; at t = 0.5 -> u = 0.5 + 0.5*(0.25/0.75) = 0.6667 -> R ~= 170
        let at_quarter = mid_lut[64 * 4];
        let at_half = mid_lut[128 * 4];
        assert!((i32::from(at_quarter) - 128).abs() <= 2);
        assert!((i32::from(at_half) - 170).abs() <= 2);
        assert_eq!(mid_lut[3], 0);
        assert_eq!(mid_lut[255 * 4 + 3], 255);

        // Zoidium absolute Bezier handles: ordered vs independent crossed handles
        let ordered = vec![
            ZoidiumCurvePoint::new(0.0, 0.0),
            ZoidiumCurvePoint {
                x: 0.2,
                y: 0.3,
                handles: Some(ZoidiumCurveHandles {
                    in_handle: None,
                    out_handle: Some([0.7, 0.6]),
                    in_independent: false,
                    out_independent: false,
                }),
            },
            ZoidiumCurvePoint {
                x: 0.8,
                y: 0.7,
                handles: Some(ZoidiumCurveHandles {
                    in_handle: Some([0.3, 0.1]),
                    out_handle: None,
                    in_independent: false,
                    out_independent: false,
                }),
            },
            ZoidiumCurvePoint::new(1.0, 1.0),
        ];
        let mut independent = ordered.clone();
        independent[2].handles.as_mut().unwrap().in_independent = true;
        let y_ord = evaluate_zoidium_curve_points(&ordered, 0.5);
        let y_ind = evaluate_zoidium_curve_points(&independent, 0.5);
        assert!((0.0..=1.0).contains(&y_ord));
        assert!((0.0..=1.0).contains(&y_ind));
        assert!((y_ord - y_ind).abs() > 1e-3);
    }
}
