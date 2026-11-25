//! Native GPU plugin effect runners, Magic Looks chain, text/polly helpers, and CPU bake helpers.

use super::*;

impl EffectRegistry {
    /// Multi-pass 4-stage 8x8 DCT/IDCT pipeline for `jpegdamage`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_jpeg_damage(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let hdr_fmt = wgpu::TextureFormat::Rgba16Float;
        let canonical = self.ensure_pipeline(device, "jpegdamage", false)?;
        let _ = self.ensure_pipeline_for_format(device, "jpegdamage", false, hdr_fmt)?;

        let base_u = pack_effect_uniforms("jpegdamage", params, false, false);
        let low_w = (base_u.p5[1].round() as u32).clamp(8, width.max(8));
        let low_h = (base_u.p5[2].round() as u32).clamp(8, height.max(8));
        let dct_w = low_w.div_ceil(8) * 8;
        let dct_h = low_h.div_ceil(8) * 8;

        let stage0 = self.create_temp_target_with_format(device, dct_w, dct_h, hdr_fmt, "zc-jd-s0");
        let stage1 = self.create_temp_target_with_format(device, dct_w, dct_h, hdr_fmt, "zc-jd-s1");
        let stage2 = self.create_temp_target_with_format(device, dct_w, dct_h, hdr_fmt, "zc-jd-s2");

        // Pass 0: Row forward DCT (input -> stage0)
        let mut u0 = base_u;
        u0.pass = 0;
        u0.resolution = [dct_w as f32, dct_h as f32];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, hdr_fmt),
            &u0,
            targets.input,
            &stage0.view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 1: Col forward DCT + freq scaling + JPEG Q50 quant + bitstream errors (stage0 -> stage1)
        let mut u1 = base_u;
        u1.pass = 1;
        u1.resolution = [dct_w as f32, dct_h as f32];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, hdr_fmt),
            &u1,
            &stage0.view,
            &stage1.view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 2: Col inverse DCT (stage1 -> stage2)
        let mut u2 = base_u;
        u2.pass = 2;
        u2.resolution = [dct_w as f32, dct_h as f32];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, hdr_fmt),
            &u2,
            &stage1.view,
            &stage2.view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        // Pass 3: Row inverse DCT + YCbCr->RGB + tone + blend onto unmodified input -> output
        let mut u3 = base_u;
        u3.pass = 3;
        u3.resolution = [width as f32, height as f32];
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u3,
            targets.input,
            targets.output,
            Some(&stage2.view),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        self.release_temp_target(stage0);
        self.release_temp_target(stage1);
        self.release_temp_target(stage2);
        Ok(())
    }

    /// 2-pass macroblock SAD motion field + warp pipeline for `datamosh`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_datamosh(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let hdr_fmt = wgpu::TextureFormat::Rgba16Float;
        let canonical = self.ensure_pipeline(device, "datamosh", false)?;
        let _ = self.ensure_pipeline_for_format(device, "datamosh", false, hdr_fmt)?;

        let mut u_warp = pack_effect_uniforms("datamosh", params, targets.aux.is_some(), false);
        let algorithm = u_warp.p1[0];
        let mut owned_mv: Option<ScratchTarget> = None;

        if algorithm < 0.5 && targets.aux.is_some() {
            let bs = u_warp.p0[2].max(2.0);
            let gw = ((width as f32) / bs).ceil().clamp(1.0, 256.0) as u32;
            let gh = ((height as f32) / bs).ceil().clamp(1.0, 256.0) as u32;
            let mv_tex = self.create_temp_target_with_format(device, gw, gh, hdr_fmt, "zc-dm-mv");

            let mut u0 = u_warp;
            u0.pass = 0;
            u0.p3[0] = gw as f32;
            u0.p3[1] = gh as f32;
            u0.p3[2] = params.float_or("spacing", 4.0).clamp(1.0, 32.0);
            self.encode_pass(
                device,
                encoder,
                (canonical, false, hdr_fmt),
                &u0,
                targets.input,
                &mv_tex.view,
                targets.aux,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            u_warp.flags |= 1u32 << 1;
            owned_mv = Some(mv_tex);
        }

        u_warp.pass = 1;
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u_warp,
            targets.input,
            targets.output,
            targets.aux,
            owned_mv.as_ref().map(|t| &t.view).or(targets.aux2),
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        if let Some(mv) = owned_mv {
            self.release_temp_target(mv);
        }
        Ok(())
    }

    /// 2-pass key-mask extraction + feature overlay pipeline for `tracery`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_tracery(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let canonical = self.ensure_pipeline(device, "tracery", false)?;
        let mask_tex = self.create_temp_target(device, width, height, "zc-tracery-mask");

        let mut u0 = pack_effect_uniforms("tracery", params, false, false);
        u0.pass = 0;
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u0,
            targets.input,
            &mask_tex.view,
            None,
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );

        let mut u1 = u0;
        u1.pass = 1;
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u1,
            targets.input,
            targets.output,
            Some(&mask_tex.view),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        self.release_temp_target(mask_tex);
        Ok(())
    }

    /// Multi-pass tool-chain executor for Zoidium Magic Looks (`magiclooks` / `looks`).
    /// Parses `looksInstances` JSON (or `chainOrder` / legacy `ml_*` flags) and executes every
    /// active tool among all 29 Magic Looks tool types in order on HDR `Rgba16Float` targets,
    /// including multi-pass separable Gaussian glow (`diffusion`), dual-axis streaks
    /// (`star-filter`), horizontal streaks (`anamorphic-flare`), 1D LUT baking (`curves`,
    /// `s-curve`), and 3D `.cube` LUT (`lut`).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_magic_looks_chain(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let tools = parse_magic_looks_chain(params);
        if tools.is_empty() {
            let uniforms = pack_effect_uniforms(
                "magiclooks",
                params,
                targets.aux.is_some(),
                targets.aux2.is_some(),
            );
            let canonical = self.ensure_pipeline(device, "magiclooks", false)?;
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &uniforms,
                targets.input,
                targets.output,
                targets.aux,
                targets.aux2,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            return Ok(());
        }

        let hdr_fmt = wgpu::TextureFormat::Rgba16Float;
        let canonical = self.ensure_pipeline(device, "magiclooks_tool", false)?;
        let _ = self.ensure_pipeline_for_format(device, "magiclooks_tool", false, hdr_fmt)?;

        let ping =
            self.create_temp_target_with_format(device, width, height, hdr_fmt, "zc-ml-ping");
        let pong =
            self.create_temp_target_with_format(device, width, height, hdr_fmt, "zc-ml-pong");
        let mut owned_luts: Vec<wgpu::Texture> = Vec::new();

        let uploaded_3d_lut = if let Some(EffectParamValue::Lut3D { size, data }) =
            params.values.get("lut")
        {
            let n = *size;
            let expected = (n as usize) * (n as usize) * (n as usize) * 4;
            if n >= 2 && data.len() >= expected {
                Some((
                    n as f32,
                    upload_rgba8_texture(device, queue, "zc-ml-lut3d", n * n, n, &data[..expected]),
                ))
            } else {
                None
            }
        } else {
            None
        };
        let lut3d_view = uploaded_3d_lut
            .as_ref()
            .map(|(sz, t)| (*sz, t.create_view(&wgpu::TextureViewDescriptor::default())));

        let total = tools.len();
        let mut read_ping = false;
        let mut first_pass = true;

        for (idx, tool) in tools.iter().enumerate() {
            let is_last = idx + 1 == total;
            let src_view: &wgpu::TextureView = if first_pass {
                targets.input
            } else if read_ping {
                &ping.view
            } else {
                &pong.view
            };
            let in_straight = !first_pass;

            let mut u = EffectUniforms::from_params(params);
            u.mode = tool.mode;
            u.pass = 0;
            u.p0 = tool.p0;
            u.p1 = tool.p1;
            u.p2 = tool.p2;
            u.p3 = tool.p3;
            u.c0 = tool.c0;
            u.c1 = tool.c1;
            u.c2 = tool.c2;
            u.c3 = tool.c3;

            let mut aux_lut_v: Option<wgpu::TextureView> = None;
            let mut aux_scratch: Option<ScratchTarget> = None;
            let mut aux2_scratch: Option<ScratchTarget> = None;
            let mut tool_scratch: Vec<ScratchTarget> = Vec::new();

            match tool.mode {
                // curves (13): 1D RGBA LUT
                13 => {
                    if let Some(ref json_str) = tool.aux_json {
                        let lut = bake_curves_json_rgba8(json_str);
                        let t = upload_rgba8_texture(device, queue, "zc-ml-curves", 256, 1, &lut);
                        aux_lut_v = Some(t.create_view(&wgpu::TextureViewDescriptor::default()));
                        owned_luts.push(t);
                    }
                }
                // s-curve (14): 1D luminance LUT
                14 => {
                    if let Some(ref json_str) = tool.aux_json {
                        let lut = bake_scurve_json_rgba8(json_str);
                        let t = upload_rgba8_texture(device, queue, "zc-ml-scurve", 256, 1, &lut);
                        aux_lut_v = Some(t.create_view(&wgpu::TextureViewDescriptor::default()));
                        owned_luts.push(t);
                    }
                }
                // lut (15): optional 3D LUT
                15 => {
                    if let Some((sz, ref view)) = lut3d_view {
                        u.p0[3] = sz;
                        // Passed directly below via explicit borrow
                        let _ = view;
                    }
                }
                // diffusion (21): 2-pass separable Gaussian glow into temp Rgba16Float
                21 if u.p0[2].abs() > 1e-5 => {
                    let dr = (1.0 + u.p0[0] * 10.0) * (0.5 + u.p0[1] * 0.15);
                    let sigma = (dr * 0.6).clamp(0.75, 16.0);
                    let blur_h = self.create_temp_target_with_format(
                        device,
                        width,
                        height,
                        hdr_fmt,
                        "zc-ml-diff-h",
                    );
                    let blur_v = self.create_temp_target_with_format(
                        device,
                        width,
                        height,
                        hdr_fmt,
                        "zc-ml-diff-v",
                    );

                    let mut u_bh = EffectUniforms::from_params(params);
                    u_bh.pass = 1;
                    u_bh.flags = u32::from(in_straight) | (1u32 << 1);
                    u_bh.p0 = [1.0 / (width as f32), 0.0, sigma, 0.0];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &u_bh,
                        src_view,
                        &blur_h.view,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );

                    let mut u_bv = EffectUniforms::from_params(params);
                    u_bv.pass = 1;
                    u_bv.flags = 1u32 | (1u32 << 1);
                    u_bv.p0 = [0.0, 1.0 / (height as f32), sigma, 0.0];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &u_bv,
                        &blur_h.view,
                        &blur_v.view,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                    tool_scratch.push(blur_h);
                    aux_scratch = Some(blur_v);
                }
                // star-filter (22): dual-axis directional streak passes
                22 if u.p0[1].abs() > 1e-5 => {
                    let ang = u.p1[0].to_radians();
                    let rad = 4.0 + u.p0[0] * 48.0;
                    let thresh = u.p0[2];
                    let s1_t = self.create_temp_target_with_format(
                        device,
                        width,
                        height,
                        hdr_fmt,
                        "zc-ml-star-1",
                    );
                    let s2_t = self.create_temp_target_with_format(
                        device,
                        width,
                        height,
                        hdr_fmt,
                        "zc-ml-star-2",
                    );

                    let mut us1 = EffectUniforms::from_params(params);
                    us1.pass = 2;
                    us1.flags = u32::from(in_straight) | (1u32 << 1);
                    us1.p0 = [ang.cos(), ang.sin(), rad, thresh];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &us1,
                        src_view,
                        &s1_t.view,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );

                    let mut us2 = us1;
                    us2.p0 = [-ang.sin(), ang.cos(), rad, thresh];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &us2,
                        src_view,
                        &s2_t.view,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                    aux_scratch = Some(s1_t);
                    aux2_scratch = Some(s2_t);
                }
                // anamorphic-flare (23): horizontal streak pass
                23 if u.p0[1].abs() > 1e-5 => {
                    let rad = (u.p0[0] * 40.0).max(4.0);
                    let thresh = u.p0[2];
                    let af_t = self
                        .create_temp_target_with_format(device, width, height, hdr_fmt, "zc-ml-af");
                    let mut uaf = EffectUniforms::from_params(params);
                    uaf.pass = 2;
                    uaf.flags = u32::from(in_straight) | (1u32 << 1);
                    uaf.p0 = [1.0, 0.0, rad, thresh];
                    self.encode_pass(
                        device,
                        encoder,
                        (canonical, false, hdr_fmt),
                        &uaf,
                        src_view,
                        &af_t.view,
                        None,
                        None,
                        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    );
                    aux_scratch = Some(af_t);
                }
                _ => {}
            }

            let effective_aux: Option<&wgpu::TextureView> = if tool.mode == 15 {
                lut3d_view.as_ref().map(|(_, v)| v).or(targets.aux)
            } else {
                aux_scratch.as_ref().map(|t| &t.view).or(aux_lut_v.as_ref())
            };
            let effective_aux2: Option<&wgpu::TextureView> = aux2_scratch.as_ref().map(|t| &t.view);

            let out_straight = !is_last;
            u.flags = u32::from(in_straight)
                | (u32::from(out_straight) << 1)
                | (u32::from(effective_aux.is_some()) << 2);

            let (dst_view, dst_fmt) = if is_last {
                (targets.output, self.format)
            } else if read_ping {
                (&pong.view, hdr_fmt)
            } else {
                (&ping.view, hdr_fmt)
            };

            self.encode_pass(
                device,
                encoder,
                (canonical, false, dst_fmt),
                &u,
                src_view,
                dst_view,
                effective_aux,
                effective_aux2,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );

            // Release per-tool scratch targets immediately so subsequent tools can reuse them.
            if let Some(t) = aux_scratch {
                self.release_temp_target(t);
            }
            if let Some(t) = aux2_scratch {
                self.release_temp_target(t);
            }
            self.release_temp_targets(tool_scratch);

            if !is_last {
                read_ping = !read_ping;
                first_pass = false;
            }
        }
        self.release_temp_target(ping);
        self.release_temp_target(pong);
        let _ = owned_luts;
        Ok(())
    }

    /// CPU single-stroke glyph rasterizer + GPU composite for `numbers` and `timecode`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_text_field(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        is_timecode: bool,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let effect_key = if is_timecode { "timecode" } else { "numbers" };
        let canonical = self.ensure_pipeline(device, effect_key, false)?;

        let text = if is_timecode {
            format_timecode_text(params)
        } else {
            format_numbers_text(params)
        };

        let pos = if is_timecode {
            if params.values.contains_key("textPosition") {
                params.vec2_or("textPosition", [0.5, 0.9])
            } else {
                params.vec2_or("position", [0.5, 0.9])
            }
        } else {
            params.vec2_or("position", [0.5, 0.5])
        };
        let raw_size = if is_timecode {
            if params.values.contains_key("textSize") {
                params.float_or("textSize", 36.0)
            } else {
                params.float_or("size", 36.0)
            }
        } else {
            params.float_or("size", 40.0)
        };
        // Scale timecode/numbers glyph size proportionally to render height (1080p reference)
        // so 320x180, 640x360, and 1920x1080 renders match consistently.
        let height_scale = ((height as f32) / 1080.0).max(0.1);
        let size = if is_timecode {
            (raw_size * height_scale).clamp(6.0, 400.0)
        } else {
            raw_size.clamp(6.0, 400.0)
        };
        let tracking = params.float_or("tracking", 0.0);
        let proportional = params.bool_or("proportionalSpacing", false);
        let vertical = !is_timecode && params.int_or("direction", 0) != 0;
        let alignment = if is_timecode {
            1
        } else {
            params.int_or("alignment", 1).clamp(0, 2)
        };
        let stroke_w = if is_timecode {
            0.0
        } else {
            params.float_or("strokeWidth", 1.5).clamp(0.0, 32.0)
        };
        let display_opts = if is_timecode {
            0u32
        } else {
            params.int_or("displayOptions", 0).clamp(0, 3) as u32
        };

        let mut fill_col = if is_timecode {
            params.vec4_or("textColor", [1.0, 1.0, 1.0, 1.0])
        } else {
            params.vec4_or("fillColor", [1.0, 1.0, 1.0, 1.0])
        };
        let tc_opacity = if is_timecode && params.values.contains_key("opacity") {
            let op_raw = params.float_or("opacity", 100.0);
            if op_raw > 1.0 {
                (op_raw / 100.0).clamp(0.0, 1.0)
            } else {
                op_raw.clamp(0.0, 1.0)
            }
        } else {
            1.0
        };
        fill_col[3] = (fill_col[3] * tc_opacity).clamp(0.0, 1.0);
        let stroke_col = params.vec4_or("strokeColor", [0.0, 0.0, 0.0, 1.0]);
        let box_on = is_timecode && params.bool_or("showBox", true);
        let box_rgb = params.vec3_or("boxColor", [0.0, 0.0, 0.0]);
        let box_op_raw = params.float_or("boxOpacity", 60.0);
        let box_op = if box_op_raw > 1.0 {
            (box_op_raw / 100.0).clamp(0.0, 1.0)
        } else {
            box_op_raw.clamp(0.0, 1.0)
        } * tc_opacity;
        let comp_orig = if is_timecode && params.values.contains_key("renderOnOriginal") {
            params.bool_or("renderOnOriginal", true)
        } else {
            params.bool_or("compositeOriginal", true)
        };

        let raster = rasterize_text_field_coverage(
            &text,
            pos[0] * (width as f32),
            pos[1] * (height as f32),
            size,
            tracking,
            proportional,
            vertical,
            alignment,
            stroke_w,
            width,
            height,
        );

        let mut u = EffectUniforms::from_params(params);
        u.mode = display_opts;
        u.c0 = fill_col;
        u.c1 = stroke_col;
        u.c2 = [box_rgb[0], box_rgb[1], box_rgb[2], box_op];

        let mut owned_tex: Option<wgpu::Texture> = None;
        let mut tex_view: Option<wgpu::TextureView> = None;
        let mut has_tex = false;

        if let Some(r) = raster {
            u.p0 = [r.x0 as f32, r.y0 as f32, r.w as f32, r.h as f32];
            let pad = (size * 0.2).max(4.0);
            u.p1 = [
                (r.x0 as f32) - pad,
                (r.x0 as f32) + (r.w as f32) + pad,
                (r.y0 as f32) - pad * 0.6,
                (r.y0 as f32) + (r.h as f32) + pad * 0.6,
            ];
            let tex = upload_rgba8_texture(device, queue, "zc-textfield-cov", r.w, r.h, &r.rgba);
            tex_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            owned_tex = Some(tex);
            has_tex = true;
        }

        u.flags =
            u32::from(comp_orig) | (u32::from(box_on && has_tex) << 1) | (u32::from(has_tex) << 2);
        self.encode_pass(
            device,
            encoder,
            (canonical, false, self.format),
            &u,
            targets.input,
            targets.output,
            tex_view.as_ref(),
            None,
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        );
        let _ = owned_tex;
        Ok(())
    }

    /// 3D projected shattered-polygon mesh runner for `ccpixelpolly`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_pixel_polly(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &EffectParams,
        targets: &EffectTargets<'_>,
        width: u32,
        height: u32,
    ) -> Result<(), EffectError> {
        let u = pack_effect_uniforms("ccpixelpolly", params, false, false);
        let t = u.p0[0];
        if t <= 1e-5 {
            let canonical = self.ensure_pipeline(device, "ccpixelpolly", false)?;
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u,
                targets.input,
                targets.output,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            return Ok(());
        }

        let verts = build_pixel_polly_mesh(params, width as f32, height as f32, t);
        if verts.is_empty() {
            let canonical = self.ensure_pipeline(device, "ccpixelpolly", false)?;
            self.encode_pass(
                device,
                encoder,
                (canonical, false, self.format),
                &u,
                targets.input,
                targets.output,
                None,
                None,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );
            return Ok(());
        }

        if self.polly_mesh_pipeline.is_none() {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zc-effect-ccpixelpolly-mesh"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_POLLY_MESH_WGSL)),
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("zc-effect-ccpixelpolly-mesh"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: size_of::<PollyVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 8,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 16,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 24,
                                shader_location: 3,
                            },
                        ],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            self.polly_mesh_pipeline = Some(pipeline);
        }

        let vb_bytes: &[u8] = bytemuck::cast_slice(&verts);
        let vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zc-polly-vb"),
            size: vb_bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: true,
        });
        {
            let mut mapped = vb
                .slice(..)
                .get_mapped_range_mut()
                .expect("mapped polly vertex buffer");
            mapped.copy_from_slice(vb_bytes);
        }
        vb.unmap();

        let uniform_buf = create_uniform_buffer(device, &u);
        let bind_group = self.create_bind_group(
            device,
            &uniform_buf,
            targets.input,
            &self.fallback_view,
            &self.fallback_view,
        );
        let pipeline = self
            .polly_mesh_pipeline
            .as_ref()
            .expect("polly mesh pipeline compiled");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("zc-ccpixelpolly-mesh"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: targets.output,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.draw(0..(verts.len() as u32), 0..1);
        Ok(())
    }

    pub(crate) fn create_temp_target(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
    ) -> ScratchTarget {
        let fmt = self.format;
        self.create_temp_target_with_format(device, width, height, fmt, label)
    }

    pub(crate) fn create_temp_target_with_format(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> ScratchTarget {
        self.scratch_pool
            .acquire(device, width, height, format, label)
    }

    pub(crate) fn release_temp_target(&mut self, target: ScratchTarget) {
        self.scratch_pool.release(target);
    }

    pub(crate) fn release_temp_targets(&mut self, targets: Vec<ScratchTarget>) {
        for target in targets {
            self.scratch_pool.release(target);
        }
    }

    pub(crate) fn compile_wgsl_pipeline(
        &self,
        device: &wgpu::Device,
        label: &str,
        wgsl_source: &str,
        blend: Option<wgpu::BlendState>,
    ) -> wgpu::RenderPipeline {
        self.compile_wgsl_pipeline_for_format(device, label, wgsl_source, blend, self.format)
    }

    pub(crate) fn compile_wgsl_pipeline_for_format(
        &self,
        device: &wgpu::Device,
        label: &str,
        wgsl_source: &str,
        blend: Option<wgpu::BlendState>,
        format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl_source)),
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&self.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    pub(crate) fn create_bind_group(
        &self,
        device: &wgpu::Device,
        uniform_buf: &wgpu::Buffer,
        input_view: &wgpu::TextureView,
        aux_view: &wgpu::TextureView,
        aux2_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zc-effect-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.linear_clamp_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(aux_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(aux2_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.linear_repeat_sampler),
                },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_pass(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_key: (&'static str, bool, wgpu::TextureFormat),
        uniforms: &EffectUniforms,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        aux_view: Option<&wgpu::TextureView>,
        aux2_view: Option<&wgpu::TextureView>,
        load_op: wgpu::LoadOp<wgpu::Color>,
    ) {
        let uniform_buf = create_uniform_buffer(device, uniforms);
        let bind_group = self.create_bind_group(
            device,
            &uniform_buf,
            input_view,
            aux_view.unwrap_or(&self.fallback_view),
            aux2_view.unwrap_or(&self.fallback_view),
        );
        let pipeline = &self.pipelines[&pipeline_key];
        record_fullscreen_pass(
            encoder,
            pipeline_key.0,
            pipeline,
            &bind_group,
            output_view,
            load_op,
        );
    }
}

// ============================================================================
// CPU Helpers for Native FX & Multi-Pass Plugin Effects
// ============================================================================

pub(crate) fn srgb_to_linear_cpu(x: f32) -> f32 {
    let cx = x.clamp(0.0, 1.0);
    if cx <= 0.04045 {
        cx / 12.92
    } else {
        ((cx + 0.055) / 1.055).powf(2.4)
    }
}

pub(crate) fn dominant_channel_triple(c: [f32; 3]) -> (usize, usize, usize) {
    if c[0] >= c[1] && c[0] >= c[2] {
        (0, 1, 2)
    } else if c[1] >= c[0] && c[1] >= c[2] {
        (1, 0, 2)
    } else {
        (2, 0, 1)
    }
}

pub(crate) fn rgb_hue_cpu(c: [f32; 3]) -> f32 {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    let d = mx - mn;
    if d < 1e-7 {
        return 0.0;
    }
    let h = if (mx - c[0]).abs() < 1e-7 {
        let mut t = (c[1] - c[2]) / d;
        if c[1] < c[2] {
            t += 6.0;
        }
        t / 6.0
    } else if (mx - c[1]).abs() < 1e-7 {
        ((c[2] - c[0]) / d + 2.0) / 6.0
    } else {
        ((c[0] - c[1]) / d + 4.0) / 6.0
    };
    h.rem_euclid(1.0)
}

pub(crate) fn eval_monotone_cubic_curve(pts_in: &[[f32; 2]], x: f32) -> f32 {
    if pts_in.is_empty() {
        return x.clamp(0.0, 1.0);
    }
    if pts_in.len() == 1 {
        return pts_in[0][1].clamp(0.0, 1.0);
    }
    let mut pts: Vec<[f32; 2]> = pts_in
        .iter()
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .map(|p| [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0)])
        .collect();
    if pts.len() < 2 {
        return x.clamp(0.0, 1.0);
    }
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
    // Deduplicate identical x coordinates
    let mut uniq: Vec<[f32; 2]> = Vec::with_capacity(pts.len());
    for p in pts {
        if let Some(last) = uniq.last_mut()
            && (p[0] - last[0]).abs() < 1e-5
        {
            last[1] = p[1];
            continue;
        }
        uniq.push(p);
    }
    if uniq.len() < 2 {
        return uniq[0][1].clamp(0.0, 1.0);
    }
    let cx = x.clamp(0.0, 1.0);
    if cx <= uniq[0][0] {
        return uniq[0][1].clamp(0.0, 1.0);
    }
    let last = uniq.len() - 1;
    if cx >= uniq[last][0] {
        return uniq[last][1].clamp(0.0, 1.0);
    }

    // Fritsch-Carlson monotone cubic Hermite tangents
    let n = uniq.len();
    let mut dx = vec![0.0_f32; n - 1];
    let mut delta = vec![0.0_f32; n - 1];
    for i in 0..(n - 1) {
        dx[i] = (uniq[i + 1][0] - uniq[i][0]).max(1e-6);
        delta[i] = (uniq[i + 1][1] - uniq[i][1]) / dx[i];
    }
    let mut m = vec![0.0_f32; n];
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    for i in 1..(n - 1) {
        if delta[i - 1] * delta[i] <= 0.0 {
            m[i] = 0.0;
        } else {
            m[i] = 0.5 * (delta[i - 1] + delta[i]);
        }
    }
    for i in 0..(n - 1) {
        if delta[i].abs() < 1e-6 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
        } else {
            let a = m[i] / delta[i];
            let b = m[i + 1] / delta[i];
            let h = a.hypot(b);
            if h > 3.0 {
                let s = 3.0 / h;
                m[i] = s * a * delta[i];
                m[i + 1] = s * b * delta[i];
            }
        }
    }

    let mut seg = 0usize;
    for i in 0..(n - 1) {
        if cx >= uniq[i][0] && cx <= uniq[i + 1][0] {
            seg = i;
            break;
        }
    }
    let h = dx[seg];
    let t = ((cx - uniq[seg][0]) / h).clamp(0.0, 1.0);
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    (h00 * uniq[seg][1] + h10 * h * m[seg] + h01 * uniq[seg + 1][1] + h11 * h * m[seg + 1])
        .clamp(0.0, 1.0)
}

pub(crate) fn parse_curve_points_value(v: &serde_json::Value) -> Vec<[f32; 2]> {
    let mut out = Vec::new();
    let Some(arr) = v.as_array() else {
        return out;
    };
    for item in arr {
        if let Some(pair) = item.as_array() {
            if pair.len() >= 2
                && let (Some(x), Some(y)) = (pair[0].as_f64(), pair[1].as_f64())
            {
                out.push([x as f32, y as f32]);
            }
        } else if let Some(obj) = item.as_object()
            && let (Some(x), Some(y)) = (
                obj.get("x").and_then(serde_json::Value::as_f64),
                obj.get("y").and_then(serde_json::Value::as_f64),
            )
        {
            out.push([x as f32, y as f32]);
        }
    }
    out
}

pub(crate) fn bake_curves_json_rgba8(json_str: &str) -> Vec<u8> {
    let mut out = vec![0u8; 256 * 4];
    let parsed: serde_json::Value =
        serde_json::from_str(json_str).unwrap_or(serde_json::Value::Null);
    let empty: Vec<[f32; 2]> = Vec::new();
    let pick = |keys: &[&str]| -> Vec<[f32; 2]> {
        let Some(obj) = parsed.as_object() else {
            return Vec::new();
        };
        for k in keys {
            if let Some(v) = obj.get(*k) {
                let pts = parse_curve_points_value(v);
                if !pts.is_empty() {
                    return pts;
                }
            }
        }
        empty.clone()
    };
    let master = pick(&["master", "rgb", "luma", "c"]);
    let red = pick(&["r", "red"]);
    let green = pick(&["g", "green"]);
    let blue = pick(&["b", "blue"]);
    let alpha = pick(&["a", "alpha"]);

    for i in 0..256usize {
        let x = (i as f32) / 255.0;
        let eval_ch = |ch: &[[f32; 2]]| -> f32 {
            let v1 = if ch.is_empty() {
                x
            } else {
                eval_monotone_cubic_curve(ch, x)
            };
            if master.is_empty() {
                v1
            } else {
                eval_monotone_cubic_curve(&master, v1)
            }
        };
        let r = eval_ch(&red);
        let g = eval_ch(&green);
        let b = eval_ch(&blue);
        let a = if alpha.is_empty() {
            x
        } else {
            eval_monotone_cubic_curve(&alpha, x)
        };
        out[i * 4] = (r.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[i * 4 + 1] = (g.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[i * 4 + 2] = (b.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[i * 4 + 3] = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    out
}

pub(crate) fn bake_scurve_json_rgba8(json_str: &str) -> Vec<u8> {
    let mut out = vec![0u8; 256 * 4];
    let parsed: serde_json::Value =
        serde_json::from_str(json_str).unwrap_or(serde_json::Value::Null);
    let pts = if parsed.is_array() {
        parse_curve_points_value(&parsed)
    } else if let Some(obj) = parsed.as_object() {
        obj.get("points")
            .or_else(|| obj.get("master"))
            .or_else(|| obj.get("curve"))
            .map(parse_curve_points_value)
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    for i in 0..256usize {
        let x = (i as f32) / 255.0;
        let y = if pts.is_empty() {
            x
        } else {
            eval_monotone_cubic_curve(&pts, x)
        };
        let u8v = (y.clamp(0.0, 1.0) * 255.0).round() as u8;
        out[i * 4] = u8v;
        out[i * 4 + 1] = u8v;
        out[i * 4 + 2] = u8v;
        out[i * 4 + 3] = 255;
    }
    out
}

// ============================================================================
// Magic Looks 29-Tool Chain Parser
// ============================================================================

pub(crate) struct MagicLooksToolSpec {
    pub(crate) mode: u32,
    pub(crate) p0: [f32; 4],
    pub(crate) p1: [f32; 4],
    pub(crate) p2: [f32; 4],
    pub(crate) p3: [f32; 4],
    pub(crate) c0: [f32; 4],
    pub(crate) c1: [f32; 4],
    pub(crate) c2: [f32; 4],
    pub(crate) c3: [f32; 4],
    pub(crate) aux_json: Option<String>,
}

pub(crate) fn has_magic_looks_chain(params: &EffectParams) -> bool {
    if let Some(s) = params.string("looksInstances") {
        let trimmed = s.trim();
        if !trimmed.is_empty() && trimmed != "[]" && trimmed != "null" {
            return true;
        }
    }
    if let Some(s) = params.string("chainOrder")
        && !s.trim().is_empty()
    {
        return true;
    }
    params.values.keys().any(|k| k.starts_with("ml_"))
}

pub(crate) fn parse_hex_color(hex: &str) -> Option<[f32; 3]> {
    let s = hex.trim().trim_start_matches('#');
    if s.len() == 6 {
        let r = u8::from_str_radix(&s[0..2], 16).ok()?;
        let g = u8::from_str_radix(&s[2..4], 16).ok()?;
        let b = u8::from_str_radix(&s[4..6], 16).ok()?;
        return Some([
            f32::from(r) / 255.0,
            f32::from(g) / 255.0,
            f32::from(b) / 255.0,
        ]);
    }
    if s.len() == 3 {
        let r = u8::from_str_radix(&s[0..1], 16).ok()? * 17;
        let g = u8::from_str_radix(&s[1..2], 16).ok()? * 17;
        let b = u8::from_str_radix(&s[2..3], 16).ok()? * 17;
        return Some([
            f32::from(r) / 255.0,
            f32::from(g) / 255.0,
            f32::from(b) / 255.0,
        ]);
    }
    None
}

pub(crate) fn json_f32(
    obj: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
    default: f32,
) -> f32 {
    for k in keys {
        if let Some(v) = obj.get(*k) {
            if let Some(f) = v.as_f64() {
                return f as f32;
            }
            if let Some(b) = v.as_bool() {
                return if b { 1.0 } else { 0.0 };
            }
            if let Some(s) = v.as_str()
                && let Ok(f) = s.trim().parse::<f32>()
            {
                return f;
            }
        }
    }
    default
}

pub(crate) fn json_bool(
    obj: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
    default: bool,
) -> bool {
    for k in keys {
        if let Some(v) = obj.get(*k) {
            if let Some(b) = v.as_bool() {
                return b;
            }
            if let Some(f) = v.as_f64() {
                return f.abs() > 1e-6;
            }
        }
    }
    default
}

pub(crate) fn json_color(
    obj: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
    default: [f32; 3],
) -> [f32; 4] {
    for k in keys {
        if let Some(v) = obj.get(*k) {
            if let Some(s) = v.as_str() {
                if let Some(rgb) = parse_hex_color(s) {
                    return [rgb[0], rgb[1], rgb[2], 1.0];
                }
            } else if let Some(arr) = v.as_array() {
                if arr.len() >= 3 {
                    let mut rgb = [
                        arr[0].as_f64().unwrap_or(f64::from(default[0])) as f32,
                        arr[1].as_f64().unwrap_or(f64::from(default[1])) as f32,
                        arr[2].as_f64().unwrap_or(f64::from(default[2])) as f32,
                    ];
                    if rgb[0] > 1.5 || rgb[1] > 1.5 || rgb[2] > 1.5 {
                        rgb = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0];
                    }
                    return [rgb[0], rgb[1], rgb[2], 1.0];
                }
            } else if let Some(m) = v.as_object() {
                let mut rgb = [
                    m.get("r")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(f64::from(default[0])) as f32,
                    m.get("g")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(f64::from(default[1])) as f32,
                    m.get("b")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(f64::from(default[2])) as f32,
                ];
                if rgb[0] > 1.5 || rgb[1] > 1.5 || rgb[2] > 1.5 {
                    rgb = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0];
                }
                return [rgb[0], rgb[1], rgb[2], 1.0];
            }
        }
    }
    [default[0], default[1], default[2], 1.0]
}

pub(crate) fn json_string(
    obj: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<String> {
    for k in keys {
        if let Some(v) = obj.get(*k) {
            if let Some(s) = v.as_str() {
                return Some(s.to_owned());
            }
            if v.is_object() || v.is_array() {
                return Some(v.to_string());
            }
        }
    }
    None
}

pub(crate) fn canonical_magic_looks_tool_mode(raw: &str) -> Option<u32> {
    let norm = raw.trim().to_ascii_lowercase().replace('_', "-");
    match norm.as_str() {
        "color-contrast" | "colorcontrast" | "cc" => Some(0),
        "hsl-colors" | "hslcolors" | "hsl" => Some(1),
        "color-ranges" | "colorranges" => Some(2),
        "crush" => Some(3),
        "contrast" => Some(4),
        "color-reversal" | "colorreversal" => Some(5),
        "three-strip" | "threestrip" | "3-strip" => Some(6),
        "lift-gamma-gain" | "liftgammagain" | "lgg" => Some(7),
        "ranged-saturation" | "rangedsaturation" => Some(8),
        "warm-cool" | "warmcool" => Some(9),
        "four-way" | "fourway" | "4-way" | "4-way-color" => Some(10),
        "mojo" => Some(11),
        "auto-shoulder" | "autoshoulder" => Some(12),
        "curves" => Some(13),
        "s-curve" | "scurve" => Some(14),
        "lut" | "cube-lut" => Some(15),
        "lightflex" => Some(16),
        "deflare" => Some(17),
        "vignette" => Some(18),
        "haze-flare" | "hazeflare" => Some(19),
        "pop" => Some(20),
        "diffusion" => Some(21),
        "star-filter" | "starfilter" => Some(22),
        "anamorphic-flare" | "anamorphicflare" => Some(23),
        "lens-distortion" | "lensdistortion" => Some(24),
        "shutter-streak" | "shutterstreak" => Some(25),
        "edge-softness" | "edgesoftness" => Some(26),
        "chromatic-aberration" | "chromaticaberration" => Some(27),
        "telecine-net" | "telecinenet" => Some(28),
        _ => None,
    }
}

pub(crate) fn build_magic_looks_tool_spec(
    mode: u32,
    cfg: &serde_json::Map<String, serde_json::Value>,
) -> MagicLooksToolSpec {
    let mut spec = MagicLooksToolSpec {
        mode,
        p0: [0.0; 4],
        p1: [0.0; 4],
        p2: [0.0; 4],
        p3: [0.0; 4],
        c0: [1.0, 1.0, 1.0, 1.0],
        c1: [1.0, 1.0, 1.0, 1.0],
        c2: [1.0, 1.0, 1.0, 1.0],
        c3: [1.0, 1.0, 1.0, 1.0],
        aux_json: None,
    };

    match mode {
        0 => {
            // color-contrast
            spec.p0 = [
                json_f32(cfg, &["ccPivot", "pivot"], 0.18),
                json_f32(cfg, &["ccExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
            spec.c0 = json_color(cfg, &["ccColor", "color"], [1.0, 1.0, 1.0]);
        }
        1 => {
            // hsl-colors
            spec.p0 = [
                json_f32(cfg, &["hslRedSat", "redSat"], 0.0),
                json_f32(cfg, &["hslOrangeSat", "orangeSat"], 0.0),
                json_f32(cfg, &["hslYellowSat", "yellowSat"], 0.0),
                json_f32(cfg, &["hslGreenSat", "greenSat"], 0.0),
            ];
            spec.p1 = [
                json_f32(cfg, &["hslCyanSat", "cyanSat"], 0.0),
                json_f32(cfg, &["hslBlueSat", "blueSat"], 0.0),
                json_f32(cfg, &["hslPurpleSat", "purpleSat"], 0.0),
                json_f32(cfg, &["hslMagentaSat", "magentaSat"], 0.0),
            ];
            spec.p2 = [
                json_f32(cfg, &["hslRedLight", "redLight"], 0.0),
                json_f32(cfg, &["hslOrangeLight", "orangeLight"], 0.0),
                json_f32(cfg, &["hslYellowLight", "yellowLight"], 0.0),
                json_f32(cfg, &["hslGreenLight", "greenLight"], 0.0),
            ];
            spec.p3 = [
                json_f32(cfg, &["hslCyanLight", "cyanLight"], 0.0),
                json_f32(cfg, &["hslBlueLight", "blueLight"], 0.0),
                json_f32(cfg, &["hslPurpleLight", "purpleLight"], 0.0),
                json_f32(cfg, &["hslMagentaLight", "magentaLight"], 0.0),
            ];
        }
        2 => {
            // color-ranges
            spec.p0 = [
                json_f32(cfg, &["crStrength", "strength"], 1.0),
                json_f32(cfg, &["crHighlight", "highlight"], 0.75),
                json_f32(cfg, &["crMidtone", "midtone"], 0.5),
                json_f32(cfg, &["crShadow", "shadow"], 0.25),
            ];
            spec.p1 = [
                if json_bool(cfg, &["crShowThreshold", "showThreshold"], false) {
                    1.0
                } else {
                    0.0
                },
                0.0,
                0.0,
                0.0,
            ];
            spec.c0 = json_color(
                cfg,
                &["crHighlightColor", "highlightColor"],
                [1.0, 1.0, 1.0],
            );
            spec.c1 = json_color(cfg, &["crMidtoneColor", "midtoneColor"], [1.0, 1.0, 1.0]);
            spec.c2 = json_color(cfg, &["crShadowColor", "shadowColor"], [1.0, 1.0, 1.0]);
        }
        3 => {
            // crush
            spec.p0 = [
                json_f32(cfg, &["crGamma", "gamma"], 1.2),
                json_f32(cfg, &["crExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
            spec.c0 = json_color(cfg, &["crColor", "color"], [1.0, 1.0, 1.0]);
        }
        4 => {
            // contrast
            spec.p0 = [
                json_f32(cfg, &["ctContrast", "contrast"], 0.25),
                json_f32(cfg, &["ctPivot", "pivot"], 0.18),
                json_f32(cfg, &["ctExposure", "exposure"], 0.0),
                0.0,
            ];
        }
        5 => {
            // color-reversal
            spec.p0 = [
                json_f32(cfg, &["rvStrength", "strength"], 1.0),
                json_f32(cfg, &["rvExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
        }
        6 => {
            // three-strip
            spec.p0 = [
                json_f32(cfg, &["tsStrength", "strength"], 1.0),
                json_f32(cfg, &["tsExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
        }
        7 => {
            // lift-gamma-gain
            spec.p0 = [
                json_f32(cfg, &["lggGammaSpace", "gammaSpace"], 2.2),
                json_f32(cfg, &["lggStrength", "strength"], 1.0),
                json_f32(cfg, &["lggExposure", "exposure"], 0.0),
                0.0,
            ];
            spec.c0 = json_color(cfg, &["lggLift", "lift"], [1.0, 1.0, 1.0]);
            spec.c1 = json_color(cfg, &["lggGamma", "gamma"], [1.0, 1.0, 1.0]);
            spec.c2 = json_color(cfg, &["lggGain", "gain"], [1.0, 1.0, 1.0]);
        }
        8 => {
            // ranged-saturation
            spec.p0 = [
                json_f32(cfg, &["rsSatHighlight", "satHighlight"], 1.0),
                json_f32(cfg, &["rsSatMidtone", "satMidtone"], 1.0),
                json_f32(cfg, &["rsSatShadow", "satShadow"], 1.0),
                json_f32(cfg, &["rsExposure", "exposure"], 0.0),
            ];
            spec.p1 = [
                json_f32(cfg, &["rsThHighlight", "thHighlight"], 0.75),
                json_f32(cfg, &["rsThMidtone", "thMidtone"], 0.5),
                json_f32(cfg, &["rsThShadow", "thShadow"], 0.25),
                if json_bool(cfg, &["rsShowThreshold", "showThreshold"], false) {
                    1.0
                } else {
                    0.0
                },
            ];
            spec.c0 = json_color(cfg, &["rsBalance", "balance"], [1.0, 1.0, 1.0]);
        }
        9 => {
            // warm-cool
            spec.p0 = [
                json_f32(cfg, &["wcWarmCool", "warmCool", "temperature"], 0.0),
                json_f32(cfg, &["wcTint", "tint"], 0.0),
                json_f32(cfg, &["wcExposure", "exposure"], 0.0),
                0.0,
            ];
        }
        10 => {
            // four-way
            spec.p0 = [
                json_f32(cfg, &["fwExposure", "exposure"], 0.0),
                json_f32(cfg, &["fwContrast", "contrast"], 0.0),
                json_f32(cfg, &["fwStrength", "strength"], 1.0),
                if json_bool(cfg, &["fwPreview", "preview"], false) {
                    1.0
                } else {
                    0.0
                },
            ];
            spec.c0 = json_color(cfg, &["fwShadows", "shadows"], [1.0, 1.0, 1.0]);
            spec.c1 = json_color(cfg, &["fwMidtones", "midtones"], [1.0, 1.0, 1.0]);
            spec.c2 = json_color(cfg, &["fwHighlights", "highlights"], [1.0, 1.0, 1.0]);
            spec.c3 = json_color(cfg, &["fwGlobal", "global"], [1.0, 1.0, 1.0]);
        }
        11 => {
            // mojo
            spec.p0 = [
                json_f32(cfg, &["mjMojo", "mojo"], 0.5),
                json_f32(cfg, &["mjPunch", "punch"], 0.2),
                json_f32(cfg, &["mjBleach", "bleach"], 0.0),
                json_f32(cfg, &["mjFade", "fade"], 0.0),
            ];
            spec.p1 = [
                json_f32(cfg, &["mjBlueSqueeze", "blueSqueeze"], 0.0),
                json_f32(cfg, &["mjSkinSqueeze", "skinSqueeze"], 0.0),
                json_f32(cfg, &["mjExposure", "exposure"], 0.0),
                json_f32(cfg, &["mjCoolWarm", "coolWarm"], 0.0),
            ];
            spec.p2 = [
                json_f32(cfg, &["mjGreenMagenta", "greenMagenta"], 0.0),
                json_f32(cfg, &["mjSkinYP", "skinYP"], 0.0),
                json_f32(cfg, &["mjStrength", "strength"], 1.0),
                0.0,
            ];
        }
        12 => {
            // auto-shoulder
            spec.p0 = [
                json_f32(cfg, &["asStrength", "strength"], 0.5),
                0.0,
                0.0,
                0.0,
            ];
        }
        13 => {
            // curves
            spec.aux_json = json_string(cfg, &["curvCurvesJson", "curvesJson", "curves"]);
        }
        14 => {
            // s-curve
            spec.aux_json = json_string(cfg, &["scvScurveJson", "scurveJson", "scurve", "points"]);
        }
        15 => {
            // lut
            spec.p0 = [
                json_f32(cfg, &["lutMode", "mode"], 1.0),
                json_f32(cfg, &["lutStrength", "strength"], 1.0),
                json_f32(cfg, &["lutGamma", "gamma"], 0.0),
                0.0,
            ];
        }
        16 => {
            // lightflex
            spec.p0 = [
                json_f32(cfg, &["lfBoost", "boost"], 1.0),
                json_f32(cfg, &["lfExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
            spec.c0 = json_color(cfg, &["lfColor", "color"], [1.0, 0.95, 0.85]);
        }
        17 => {
            // deflare
            spec.p0 = [
                json_f32(cfg, &["dfStrength", "strength"], 0.5),
                json_f32(cfg, &["dfExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
        }
        18 => {
            // vignette
            spec.p0 = [
                json_f32(cfg, &["vgCenterX", "centerX"], 0.0),
                json_f32(cfg, &["vgCenterY", "centerY"], 0.0),
                json_f32(cfg, &["vgRadius", "radius"], 0.75),
                json_f32(cfg, &["vgAspect", "aspect"], 1.0),
            ];
            spec.p1 = [
                json_f32(cfg, &["vgSpread", "spread"], 0.5),
                json_f32(cfg, &["vgFalloff", "falloff"], 0.5),
                json_f32(cfg, &["vgStrength", "strength"], 0.5),
                json_f32(cfg, &["vgExposure", "exposure"], 0.0),
            ];
            spec.c0 = json_color(cfg, &["vgColor", "color"], [0.0, 0.0, 0.0]);
        }
        19 => {
            // haze-flare
            spec.p0 = [
                json_f32(cfg, &["hfSpillage", "spillage"], 0.35),
                json_f32(cfg, &["hfSoftness", "softness"], 0.5),
                json_f32(cfg, &["hfReach", "reach"], 0.5),
                json_f32(cfg, &["hfExposure", "exposure"], 0.0),
            ];
            spec.p1 = [
                json_f32(cfg, &["hfReflExposure", "reflExposure"], 0.0),
                if json_bool(cfg, &["hfReflection", "reflection"], false) {
                    1.0
                } else {
                    0.0
                },
                json_f32(cfg, &["hfMatteSize", "matteSize"], 0.9),
                json_f32(cfg, &["hfMatteShade", "matteShade"], 0.0),
            ];
            spec.c0 = json_color(cfg, &["hfColor", "color"], [1.0, 0.92, 0.8]);
        }
        20 => {
            // pop
            spec.p0 = [
                json_f32(cfg, &["popPop", "pop", "strength"], 0.5),
                json_f32(cfg, &["popSize", "size"], 0.5),
                json_f32(cfg, &["popPreserveDetail", "preserveDetail"], 0.5),
                0.0,
            ];
        }
        21 => {
            // diffusion
            spec.p0 = [
                json_f32(cfg, &["diffSize", "size"], 0.5),
                json_f32(cfg, &["diffGrade", "grade"], 1.0),
                json_f32(cfg, &["diffGlow", "glow"], 0.5),
                json_f32(cfg, &["diffHighlightsOnly", "highlightsOnly"], 0.3),
            ];
            spec.p1 = [
                json_f32(cfg, &["diffHighlightBias", "highlightBias"], 0.0),
                json_f32(cfg, &["diffExposure", "exposure"], 0.0),
                0.0,
                0.0,
            ];
            spec.c0 = json_color(cfg, &["diffColor", "color"], [1.0, 1.0, 1.0]);
        }
        22 => {
            // star-filter
            spec.p0 = [
                json_f32(cfg, &["sfSize", "size"], 0.5),
                json_f32(cfg, &["sfBoost", "boost"], 1.0),
                json_f32(cfg, &["sfThreshold", "threshold"], 0.65),
                if json_bool(cfg, &["sfShowThreshold", "showThreshold"], false) {
                    1.0
                } else {
                    0.0
                },
            ];
            spec.p1 = [json_f32(cfg, &["sfAngle", "angle"], 45.0), 0.0, 0.0, 0.0];
            spec.c0 = json_color(cfg, &["sfColor", "color"], [1.0, 0.96, 0.85]);
        }
        23 => {
            // anamorphic-flare
            spec.p0 = [
                json_f32(cfg, &["afSize", "size"], 0.5),
                json_f32(cfg, &["afBoost", "boost"], 1.0),
                json_f32(cfg, &["afThreshold", "threshold"], 0.65),
                if json_bool(cfg, &["afShowThreshold", "showThreshold"], false) {
                    1.0
                } else {
                    0.0
                },
            ];
            spec.p1 = [
                if json_bool(cfg, &["afReflection", "reflection"], false) {
                    1.0
                } else {
                    0.0
                },
                json_f32(cfg, &["afReflBoost", "reflBoost"], 0.5),
                0.0,
                0.0,
            ];
            spec.c0 = json_color(cfg, &["afColor", "color"], [0.4, 0.7, 1.0]);
        }
        24 => {
            // lens-distortion
            spec.p0 = [
                json_f32(cfg, &["ldDistortion", "distortion"], 0.1),
                json_f32(cfg, &["ldFlatness", "flatness"], 1.0),
                0.0,
                0.0,
            ];
        }
        25 => {
            // shutter-streak
            spec.p0 = [
                json_f32(cfg, &["ssSize", "size"], 0.5),
                json_f32(cfg, &["ssBoost", "boost"], 0.5),
                json_f32(cfg, &["ssFalloff", "falloff"], 0.5),
                0.0,
            ];
        }
        26 => {
            // edge-softness
            spec.p0 = [
                json_f32(cfg, &["esBlurSize", "blurSize"], 0.02),
                json_f32(cfg, &["esQuality", "quality"], 2.0),
                json_f32(cfg, &["esCenterX", "centerX"], 0.0),
                json_f32(cfg, &["esCenterY", "centerY"], 0.0),
            ];
            spec.p1 = [
                json_f32(cfg, &["esRadius", "radius"], 0.5),
                json_f32(cfg, &["esAspect", "aspect"], 1.0),
                json_f32(cfg, &["esSpread", "spread"], 0.35),
                0.0,
            ];
        }
        27 => {
            // chromatic-aberration
            spec.p0 = [
                json_f32(cfg, &["caRedCyan", "redCyan"], 0.5),
                json_f32(cfg, &["caGreenMagenta", "greenMagenta"], 0.0),
                json_f32(cfg, &["caBlueYellow", "blueYellow"], 0.5),
                0.0,
            ];
        }
        28 => {
            // telecine-net
            spec.p0 = [
                json_f32(cfg, &["tnSize", "size"], 0.04),
                json_f32(cfg, &["tnStrength", "strength"], 0.5),
                json_f32(cfg, &["tnExposure", "exposure"], 0.0),
                0.0,
            ];
        }
        _ => {}
    }

    spec
}

pub(crate) fn parse_magic_looks_chain(params: &EffectParams) -> Vec<MagicLooksToolSpec> {
    let mut out = Vec::new();

    // 1. Primary format: `looksInstances` JSON array of tool instance objects
    if let Some(json_str) = params.string("looksInstances")
        && let Ok(serde_json::Value::Array(items)) =
            serde_json::from_str::<serde_json::Value>(json_str)
    {
        for item in items {
            let Some(obj) = item.as_object() else {
                continue;
            };
            if !json_bool(obj, &["enabled", "on", "active"], true) {
                continue;
            }
            let tool_name = obj
                .get("toolId")
                .or_else(|| obj.get("tool"))
                .or_else(|| obj.get("type"))
                .or_else(|| obj.get("id"))
                .and_then(serde_json::Value::as_str);
            let Some(mode) = tool_name.and_then(canonical_magic_looks_tool_mode) else {
                continue;
            };
            // Merge top-level fields with nested `params` / `values` object if present
            let mut merged = obj.clone();
            if let Some(nested) = obj
                .get("params")
                .or_else(|| obj.get("values"))
                .and_then(serde_json::Value::as_object)
            {
                for (k, v) in nested {
                    merged.insert(k.clone(), v.clone());
                }
            }
            out.push(build_magic_looks_tool_spec(mode, &merged));
        }
        if !out.is_empty() {
            return out;
        }
    }

    // 2. Fallback: `chainOrder` (JSON array or comma-separated tool IDs) reading flat params
    if let Some(order_str) = params.string("chainOrder") {
        let names: Vec<String> =
            if let Ok(serde_json::Value::Array(arr)) = serde_json::from_str(order_str) {
                arr.into_iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            } else {
                order_str
                    .split(',')
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty())
                    .collect()
            };
        let mut flat_map = serde_json::Map::new();
        for (k, v) in &params.values {
            let jv = match v {
                EffectParamValue::Float(f) => serde_json::json!(*f),
                EffectParamValue::Int(i) => serde_json::json!(*i),
                EffectParamValue::Bool(b) => serde_json::json!(*b),
                EffectParamValue::String(s) => serde_json::json!(s),
                EffectParamValue::Vec3(c) => serde_json::json!([c[0], c[1], c[2]]),
                EffectParamValue::Vec4(c) => serde_json::json!([c[0], c[1], c[2], c[3]]),
                _ => continue,
            };
            flat_map.insert(k.clone(), jv);
        }
        for name in names {
            if let Some(mode) = canonical_magic_looks_tool_mode(&name) {
                out.push(build_magic_looks_tool_spec(mode, &flat_map));
            }
        }
    }

    out
}

// ============================================================================
// Text Field Formatter & Single-Stroke Polyline Rasterizer (`numbers`/`timecode`)
// ============================================================================

pub(crate) fn format_numbers_text(params: &EffectParams) -> String {
    let raw_val = if params.bool_or("randomValues", false) {
        let seed = params.float_or("seed", 1.0);
        let h = ((params.frame * 12.9898 + seed * 78.233).sin() * 43_758.547)
            .fract()
            .abs();
        let max_v = params.float_or("value", 100.0).abs().max(1.0);
        h * max_v
    } else {
        params.float_or("value", 0.0)
    };
    let decimals = params.int_or("decimalPlaces", 0).clamp(0, 6) as usize;
    let format_mode = params.int_or("format", 0).clamp(0, 7);
    match format_mode {
        1 => {
            // Number with leading zeroes
            let s = format!("{:.*}", decimals, raw_val.abs());
            let width = if decimals > 0 { 6 + decimals } else { 5 };
            let padded = format!("{s:0>width$}");
            if raw_val < 0.0 {
                format!("-{padded}")
            } else {
                padded
            }
        }
        2..=4 => {
            let fps = match format_mode {
                3 => 24,
                4 => 25,
                _ => 30,
            };
            format_smpte_from_seconds(raw_val.max(0.0), fps, false)
        }
        5 => {
            // HH:MM:SS
            let total = raw_val.max(0.0).floor() as u64;
            let hh = (total / 3600) % 24;
            let mm = (total / 60) % 60;
            let ss = total % 60;
            format!("{hh:02}:{mm:02}:{ss:02}")
        }
        6 => {
            // Short date MM/DD/YY deterministic from value days
            let days = raw_val.max(0.0).floor() as u32;
            let mm = (days % 12) + 1;
            let dd = ((days / 12) % 28) + 1;
            let yy = (26 + (days / 365)) % 100;
            format!("{mm:02}/{dd:02}/{yy:02}")
        }
        7 => {
            // Hexadecimal
            format!("{:X}", raw_val.round().max(0.0) as u64)
        }
        _ => format!("{:.*}", decimals, raw_val),
    }
}

pub(crate) fn format_smpte_from_seconds(sec: f32, fps: u32, drop_frame: bool) -> String {
    let fps_u = fps.max(1);
    let total_frames = (sec.max(0.0) * (fps_u as f32) + 1e-4).floor() as u64;
    let f = total_frames % u64::from(fps_u);
    let total_sec = total_frames / u64::from(fps_u);
    let s = total_sec % 60;
    let m = (total_sec / 60) % 60;
    let h = (total_sec / 3600) % 24;
    let sep = if drop_frame && fps_u == 30 { ';' } else { ':' };
    format!("{h:02}:{m:02}:{s:02}{sep}{f:02}")
}

pub(crate) fn format_timecode_text(params: &EffectParams) -> String {
    let fps = params.int_or("timeUnits", 30).clamp(1, 240) as u32;
    let start_frame = params.int_or("startingFrame", 0).max(0) as u64;
    let time_source = params.int_or("timeSource", 0);
    let t_sec = if params.values.contains_key("time") {
        params.float_or("time", params.time)
    } else if time_source == 0 && params.values.contains_key("_absTime") {
        params.float_or("_absTime", params.time)
    } else {
        params.time
    };
    let total_frames = start_frame + (t_sec.max(0.0) * (fps as f32) + 1e-4).floor() as u64;
    let display_fmt = params.int_or("displayFormat", 0).clamp(0, 3);
    let drop_frame = params.bool_or("dropFrame", false);
    match display_fmt {
        1 => format!("{total_frames:05}"),
        2 => {
            // Feet + Frames 35mm (16 frames/ft)
            let feet = total_frames / 16;
            let rem = total_frames % 16;
            format!("{feet:04}+{rem:02}")
        }
        3 => {
            // Feet + Frames 16mm (40 frames/ft)
            let feet = total_frames / 40;
            let rem = total_frames % 40;
            format!("{feet:04}+{rem:02}")
        }
        _ => format_smpte_from_seconds((total_frames as f32) / (fps as f32), fps, drop_frame),
    }
}

pub(crate) struct TextFieldRaster {
    pub(crate) x0: i32,
    pub(crate) y0: i32,
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) rgba: Vec<u8>,
}

/// Single-stroke glyph segment definitions in normalized `[0, 1] x [0, 1]` cell coordinates
/// (matching Zoidium `fxgen-math.js` `GLYPHS` polyline table).
pub(crate) fn glyph_segments(ch: char) -> &'static [([f32; 2], [f32; 2])] {
    match ch.to_ascii_uppercase() {
        '0' => &[
            ([0.18, 0.08], [0.82, 0.08]),
            ([0.82, 0.08], [0.82, 0.92]),
            ([0.82, 0.92], [0.18, 0.92]),
            ([0.18, 0.92], [0.18, 0.08]),
            ([0.22, 0.78], [0.78, 0.22]),
        ],
        '1' => &[
            ([0.32, 0.22], [0.54, 0.08]),
            ([0.54, 0.08], [0.54, 0.92]),
            ([0.24, 0.92], [0.80, 0.92]),
        ],
        '2' => &[
            ([0.18, 0.22], [0.32, 0.08]),
            ([0.32, 0.08], [0.74, 0.08]),
            ([0.74, 0.08], [0.82, 0.24]),
            ([0.82, 0.24], [0.18, 0.92]),
            ([0.18, 0.92], [0.84, 0.92]),
        ],
        '3' => &[
            ([0.18, 0.14], [0.78, 0.14]),
            ([0.78, 0.14], [0.82, 0.38]),
            ([0.82, 0.38], [0.42, 0.48]),
            ([0.42, 0.48], [0.82, 0.62]),
            ([0.82, 0.62], [0.78, 0.90]),
            ([0.78, 0.90], [0.18, 0.90]),
        ],
        '4' => &[
            ([0.72, 0.92], [0.72, 0.08]),
            ([0.72, 0.08], [0.16, 0.64]),
            ([0.16, 0.64], [0.86, 0.64]),
        ],
        '5' => &[
            ([0.82, 0.08], [0.22, 0.08]),
            ([0.22, 0.08], [0.20, 0.46]),
            ([0.20, 0.46], [0.76, 0.46]),
            ([0.76, 0.46], [0.82, 0.68]),
            ([0.82, 0.68], [0.74, 0.92]),
            ([0.74, 0.92], [0.18, 0.92]),
        ],
        '6' => &[
            ([0.78, 0.12], [0.28, 0.12]),
            ([0.28, 0.12], [0.18, 0.48]),
            ([0.18, 0.48], [0.18, 0.90]),
            ([0.18, 0.90], [0.80, 0.90]),
            ([0.80, 0.90], [0.80, 0.52]),
            ([0.80, 0.52], [0.18, 0.52]),
        ],
        '7' => &[([0.16, 0.08], [0.84, 0.08]), ([0.84, 0.08], [0.38, 0.92])],
        '8' => &[
            ([0.20, 0.08], [0.80, 0.08]),
            ([0.80, 0.08], [0.80, 0.92]),
            ([0.80, 0.92], [0.20, 0.92]),
            ([0.20, 0.92], [0.20, 0.08]),
            ([0.20, 0.48], [0.80, 0.48]),
        ],
        '9' => &[
            ([0.82, 0.48], [0.20, 0.48]),
            ([0.20, 0.48], [0.20, 0.08]),
            ([0.20, 0.08], [0.82, 0.08]),
            ([0.82, 0.08], [0.82, 0.90]),
            ([0.82, 0.90], [0.26, 0.90]),
        ],
        'A' => &[
            ([0.14, 0.92], [0.50, 0.08]),
            ([0.50, 0.08], [0.86, 0.92]),
            ([0.28, 0.62], [0.72, 0.62]),
        ],
        'B' => &[
            ([0.20, 0.08], [0.20, 0.92]),
            ([0.20, 0.08], [0.74, 0.08]),
            ([0.74, 0.08], [0.80, 0.46]),
            ([0.80, 0.46], [0.20, 0.48]),
            ([0.20, 0.48], [0.82, 0.56]),
            ([0.82, 0.56], [0.76, 0.92]),
            ([0.76, 0.92], [0.20, 0.92]),
        ],
        'C' => &[
            ([0.82, 0.14], [0.24, 0.14]),
            ([0.24, 0.14], [0.18, 0.50]),
            ([0.18, 0.50], [0.24, 0.88]),
            ([0.24, 0.88], [0.82, 0.88]),
        ],
        'D' => &[
            ([0.20, 0.08], [0.20, 0.92]),
            ([0.20, 0.08], [0.68, 0.14]),
            ([0.68, 0.14], [0.84, 0.50]),
            ([0.84, 0.50], [0.68, 0.86]),
            ([0.68, 0.86], [0.20, 0.92]),
        ],
        'E' => &[
            ([0.82, 0.08], [0.20, 0.08]),
            ([0.20, 0.08], [0.20, 0.92]),
            ([0.20, 0.92], [0.82, 0.92]),
            ([0.20, 0.48], [0.70, 0.48]),
        ],
        'F' => &[
            ([0.82, 0.08], [0.20, 0.08]),
            ([0.20, 0.08], [0.20, 0.92]),
            ([0.20, 0.48], [0.70, 0.48]),
        ],
        ':' => &[([0.48, 0.28], [0.52, 0.34]), ([0.48, 0.70], [0.52, 0.76])],
        ';' => &[([0.48, 0.28], [0.52, 0.34]), ([0.52, 0.70], [0.42, 0.88])],
        '.' => &[([0.48, 0.84], [0.52, 0.90])],
        ',' => &[([0.52, 0.78], [0.42, 0.94])],
        '-' => &[([0.20, 0.50], [0.80, 0.50])],
        '+' => &[([0.20, 0.50], [0.80, 0.50]), ([0.50, 0.22], [0.50, 0.78])],
        '/' => &[([0.22, 0.92], [0.78, 0.08])],
        '\'' => &[([0.50, 0.10], [0.46, 0.30])],
        _ => &[],
    }
}

pub(crate) fn seg_dist_2d(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let ex = bx - ax;
    let ey = by - ay;
    let l2 = ex * ex + ey * ey;
    if l2 < 1e-8 {
        return (px - ax).hypot(py - ay);
    }
    let t = (((px - ax) * ex + (py - ay) * ey) / l2).clamp(0.0, 1.0);
    (px - (ax + ex * t)).hypot(py - (ay + ey * t))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rasterize_text_field_coverage(
    text: &str,
    anchor_x: f32,
    anchor_y: f32,
    size: f32,
    tracking: f32,
    proportional: bool,
    vertical: bool,
    alignment: i32,
    stroke_w: f32,
    frame_w: u32,
    frame_h: u32,
) -> Option<TextFieldRaster> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return None;
    }
    let char_h = size.max(6.0);
    let base_w = char_h * 0.62;
    let stroke_r = (char_h * 0.065).max(0.8);
    let ring_r = stroke_r + stroke_w.max(0.0);
    let pad = (ring_r + 2.0).ceil();

    let mut segs: Vec<[f32; 4]> = Vec::new();
    let mut advances: Vec<f32> = Vec::with_capacity(chars.len());
    for &ch in &chars {
        let adv = if proportional && matches!(ch, ':' | ';' | '.' | ',' | '\'' | '1') {
            base_w * 0.55
        } else {
            base_w
        };
        advances.push(adv + tracking);
    }

    let total_span: f32 = if vertical {
        (chars.len() as f32) * (char_h * 1.15 + tracking)
    } else {
        advances.iter().sum::<f32>()
    };
    let start_offset = match alignment {
        0 => 0.0,
        2 => -total_span,
        _ => -total_span * 0.5,
    };

    let mut cursor = start_offset;
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for (idx, &ch) in chars.iter().enumerate() {
        let cell_w = base_w;
        let (ox, oy) = if vertical {
            (anchor_x - cell_w * 0.5, anchor_y + cursor)
        } else {
            (anchor_x + cursor, anchor_y - char_h * 0.5)
        };
        for &(a, b) in glyph_segments(ch) {
            let ax = ox + a[0] * cell_w;
            let ay = oy + a[1] * char_h;
            let bx = ox + b[0] * cell_w;
            let by = oy + b[1] * char_h;
            min_x = min_x.min(ax).min(bx);
            min_y = min_y.min(ay).min(by);
            max_x = max_x.max(ax).max(bx);
            max_y = max_y.max(ay).max(by);
            segs.push([ax, ay, bx, by]);
        }
        cursor += if vertical {
            char_h * 1.15 + tracking
        } else {
            advances[idx]
        };
    }

    if segs.is_empty() || !min_x.is_finite() {
        return None;
    }

    let x0 = ((min_x - pad).floor() as i32).clamp(0, frame_w as i32);
    let y0 = ((min_y - pad).floor() as i32).clamp(0, frame_h as i32);
    let x1 = ((max_x + pad).ceil() as i32).clamp(0, frame_w as i32);
    let y1 = ((max_y + pad).ceil() as i32).clamp(0, frame_h as i32);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let w = (x1 - x0) as u32;
    let h = (y1 - y0) as u32;
    let mut rgba = vec![0u8; (w as usize) * (h as usize) * 4];

    for py in 0..h {
        let fy = (y0 as f32) + (py as f32) + 0.5;
        for px in 0..w {
            let fx = (x0 as f32) + (px as f32) + 0.5;
            let mut best = f32::INFINITY;
            for s in &segs {
                let d = seg_dist_2d(fx, fy, s[0], s[1], s[2], s[3]);
                if d < best {
                    best = d;
                }
            }
            let fill_cov = (stroke_r - best + 0.5).clamp(0.0, 1.0);
            let outer_cov = if stroke_w > 1e-3 {
                (ring_r - best + 0.5).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let stroke_ring = (outer_cov - fill_cov).clamp(0.0, 1.0);
            let off = ((py as usize) * (w as usize) + (px as usize)) * 4;
            rgba[off] = (fill_cov * 255.0).round() as u8;
            rgba[off + 1] = (stroke_ring * 255.0).round() as u8;
            rgba[off + 2] = 0;
            rgba[off + 3] = 255;
        }
    }

    Some(TextFieldRaster { x0, y0, w, h, rgba })
}

// ============================================================================
// CC Pixel Polly 3D Shattered Polygon Mesh Builder (`fxtrans-math.js`)
// ============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub(crate) struct PollyVertex {
    pub(crate) pos: [f32; 2],
    pub(crate) luv: [f32; 2],
    pub(crate) cuv: [f32; 2],
    pub(crate) shade: f32,
}

pub(crate) fn polly_hash01(a: u32, b: u32, seed: u32) -> f32 {
    let mut h = (a.wrapping_mul(0x8da6_b343))
        ^ (b.wrapping_mul(0xd816_3841))
        ^ (seed.wrapping_mul(0xcb1a_b31f));
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    ((h & 0x00ff_ffff) as f32) / 16_777_216.0
}

pub(crate) fn polly_rot_axis3(ax: f32, ay: f32, az: f32, angle: f32) -> [[f32; 3]; 3] {
    let len = (ax * ax + ay * ay + az * az).sqrt();
    if !len.is_finite() || len <= 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let x = ax / len;
    let y = ay / len;
    let z = az / len;
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

pub(crate) fn build_pixel_polly_mesh(
    params: &EffectParams,
    width: f32,
    height: f32,
    t: f32,
) -> Vec<PollyVertex> {
    let spacing = params.float_or("gridSpacing", 10.0).clamp(1.0, 500.0);
    let force = params.float_or("force", 100.0);
    let gravity = params.float_or("gravity", 1.0);
    let fc = params.vec2_or("forceCenter", [0.5, 0.5]);
    let spin_rad = params.float_or("spinning", 0.0).to_radians();
    let dir_rand = (params.float_or("directionRandomness", 10.0) / 100.0).clamp(0.0, 1.0);
    let speed_rand = (params.float_or("speedRandomness", 10.0) / 100.0).clamp(0.0, 1.0);
    let obj_mode = params.int_or("object", 1).clamp(0, 3);
    let tris_mode = if params.values.contains_key("object") {
        obj_mode <= 1
    } else {
        params.int_or("sectorShape", 0) != 0
    };
    let depth_sort = params.bool_or("enableDepthSort", true);
    let raw_seed = if params.values.contains_key("randomSeed") {
        params.int_or("randomSeed", 1)
    } else {
        params.int_or("seed", 1)
    };
    let seed = (raw_seed as u32).wrapping_mul(0x632b_e5ab);

    let max_pieces = 8192usize;
    let per = if tris_mode { 2usize } else { 1usize };
    let mut g = spacing;
    let mut cols = (width / g).ceil().max(1.0) as usize;
    let mut rows = (height / g).ceil().max(1.0) as usize;
    while cols * rows * per > max_pieces {
        g *= 1.25;
        cols = (width / g).ceil().max(1.0) as usize;
        rows = (height / g).ceil().max(1.0) as usize;
    }

    let fcx = fc[0] * width;
    let fcy = fc[1] * height;
    let maxd = width.max(height).max(1.0);
    let cam_dist = maxd * 1.5;

    struct Piece {
        index: u32,
        z: f32,
        tris: Vec<PollyVertex>,
    }
    let total = cols * rows * per;
    let mut pieces: Vec<Piece> = Vec::with_capacity(total);

    for n in 0..total {
        let cell = n / per;
        let ci = cell % cols;
        let cj = cell / cols;
        let x0 = (ci as f32) * g;
        let y0 = (cj as f32) * g;
        let corners: Vec<[f32; 2]> = if !tris_mode {
            vec![[x0, y0], [x0 + g, y0], [x0 + g, y0 + g], [x0, y0 + g]]
        } else if n % 2 == 0 {
            vec![[x0, y0], [x0 + g, y0], [x0, y0 + g]]
        } else {
            vec![[x0 + g, y0], [x0 + g, y0 + g], [x0, y0 + g]]
        };

        let inv_n = 1.0 / (corners.len() as f32);
        let cx: f32 = corners.iter().map(|p| p[0]).sum::<f32>() * inv_n;
        let cy: f32 = corners.iter().map(|p| p[1]).sum::<f32>() * inv_n;
        let cuv = [cx / width.max(1.0), cy / height.max(1.0)];

        let nu = n as u32;
        let h1 = polly_hash01(nu, 1, seed) * 2.0 - 1.0;
        let h2 = polly_hash01(nu, 2, seed) * 2.0 - 1.0;
        let h3 = polly_hash01(nu, 3, seed);
        let h4 = polly_hash01(nu, 4, seed) * 2.0 - 1.0;
        let h5 = polly_hash01(nu, 5, seed) * 2.0 - 1.0;
        let h6 = polly_hash01(nu, 6, seed) * 2.0 - 1.0;
        let h7 = polly_hash01(nu, 7, seed);

        let dx = cx - fcx;
        let dy = cy - fcy;
        let dist = dx.hypot(dy).max(1e-6);
        let ang = dy.atan2(dx) + dir_rand * std::f32::consts::PI * h1;
        let speed =
            (force / 100.0) * 0.6 * maxd / (1.0 + dist / (0.5 * maxd)) * (1.0 + speed_rand * h2);
        let vz = -speed * 0.5 * h3;
        let px = cx + ang.cos() * speed * t;
        let py = cy + ang.sin() * speed * t + 0.25 * gravity * height * t * t;
        let pz = vz * t;

        let rot = polly_rot_axis3(h4, h5, h6, t * (std::f32::consts::PI * h7 + spin_rad));
        let shade = (0.35 + 0.65 * rot[2][2]).max(0.0);

        let ox = width * 0.5;
        let oy = height * 0.5;
        let project = |pt: [f32; 2]| -> PollyVertex {
            let lx = pt[0] - cx;
            let ly = pt[1] - cy;
            let wx = rot[0][0] * lx + rot[0][1] * ly + px;
            let wy = rot[1][0] * lx + rot[1][1] * ly + py;
            let wz = rot[2][0] * lx + rot[2][1] * ly + pz;
            let s = cam_dist / (cam_dist - wz).max(cam_dist * 0.05);
            let sx_px = ox + (wx - ox) * s;
            let sy_px = oy + (wy - oy) * s;
            PollyVertex {
                pos: [
                    (sx_px / width.max(1.0)) * 2.0 - 1.0,
                    1.0 - (sy_px / height.max(1.0)) * 2.0,
                ],
                luv: [pt[0] / width.max(1.0), pt[1] / height.max(1.0)],
                cuv,
                shade,
            }
        };

        let pverts: Vec<PollyVertex> = corners.into_iter().map(project).collect();
        let tris = if pverts.len() == 3 {
            pverts
        } else {
            vec![
                pverts[0], pverts[1], pverts[2], pverts[0], pverts[2], pverts[3],
            ]
        };
        pieces.push(Piece {
            index: nu,
            z: pz,
            tris,
        });
    }

    if depth_sort {
        pieces.sort_by(|a, b| a.z.total_cmp(&b.z).then_with(|| a.index.cmp(&b.index)));
    }
    let mut verts = Vec::with_capacity(pieces.len() * 6);
    for p in pieces {
        verts.extend(p.tris);
    }
    verts
}
