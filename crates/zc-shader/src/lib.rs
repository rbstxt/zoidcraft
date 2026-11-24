//! ZoidCraft Custom Shader translator (`GLSL ES 1.00 -> GLSL 450 -> naga -> WGSL`)
//! and Shader+ custom property / LUT baking subsystem.

#![forbid(unsafe_code)]

pub mod glsl_shim;
pub mod shader_plus;

pub use glsl_shim::{
    CompileOptions, CompiledCustomShader, ShaderError, TextureBinding, UniformBlockLayout,
    UniformField, UniformType, UniformValue, align_up, pack_uniforms, translate_glsl_es100,
    translate_glsl_es100_with_options,
};
pub use shader_plus::{
    CURVE_LUT_WIDTH, ConditionClause, CurveChannel, CurveMode, CurvePoint, CurveSpec,
    GRADIENT_LUT_WIDTH, GradientColorSpace, GradientInterpolation, GradientSpec, GradientStop,
    MAX_PALETTE_COLORS, MAX_TIME_SAMPLES, PaletteSpec, RenameRule, ShaderPlusError,
    ShaderPropertyDef, ShaderPropertyKind, TextTextureMode, TimeSampleUnit, ToggleDisplayStyle,
    VisibilityCondition, bake_curve_rgba8, bake_gradient_rgba8, resolve_label,
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
        for ext in [
            "GL_EXT_shader_texture_lod",
            "GL_EXT_frag_depth",
            "GL_EXT_draw_buffers",
        ] {
            let src =
                format!("#extension {ext} : require\nvoid main() {{ gl_FragColor = vec4(1.0); }}");
            let err = translate_glsl_es100(&src).unwrap_err();
            assert_eq!(err, ShaderError::UnsupportedExtension(ext.to_string()));
        }

        let lod_src = r#"
            uniform sampler2D tDiffuse;
            varying vec2 vUv;
            void main() { gl_FragColor = texture2DLodEXT(tDiffuse, vUv, 0.0); }
        "#;
        assert_eq!(
            translate_glsl_es100(lod_src).unwrap_err(),
            ShaderError::UnsupportedFeature("texture2DLodEXT".to_string())
        );

        let depth_src = "void main() { gl_FragDepthEXT = 0.5; gl_FragColor = vec4(1.0); }";
        assert_eq!(
            translate_glsl_es100(depth_src).unwrap_err(),
            ShaderError::UnsupportedFeature("gl_FragDepthEXT".to_string())
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
        assert_eq!(bezier_lut[3], 255);
        assert_eq!(bezier_lut[512 * 4 + 3], 255);
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
}
