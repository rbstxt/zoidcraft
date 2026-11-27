use super::*;

impl ProjectRenderEngine {
    pub(super) fn ensure_shape_mesh(
        &mut self,
        obj: &Object3D,
        shape: &Shape3DKind,
        env: &EvalEnv<'_>,
    ) -> Option<CachedMeshHandle> {
        let uv_norm = object_uses_normalized_uv(obj);
        let content_hash = hash_u64(
            hash_shape_and_effectors(shape, obj, env),
            u64::from(uv_norm),
        );
        let key = MeshCacheKey {
            object_id: obj.id,
            content_hash,
        };
        if let Some(cached) = self.touch_cached_mesh(key) {
            return Some(cached);
        }

        let mut mesh = build_shape_mesh_cpu(shape, env);
        mesh = apply_object_effectors(mesh, obj, env);
        if uv_norm {
            apply_normalized_uv_layout(&mut mesh);
        }
        if mesh.indices.is_empty() {
            return None;
        }
        Some(self.insert_cached_mesh(key, &mesh, "zc-shape3d-mesh"))
    }

    pub(super) fn ensure_text3d_mesh(
        &mut self,
        obj: &Object3D,
        data: &zc_project::Text3DData,
        env: &EvalEnv<'_>,
    ) -> Option<CachedMeshHandle> {
        let uv_norm = object_uses_normalized_uv(obj);
        let content_hash = hash_u64(
            hash_text3d_and_effectors(data, obj, env),
            u64::from(uv_norm),
        );
        let key = MeshCacheKey {
            object_id: obj.id,
            content_hash,
        };
        if let Some(cached) = self.touch_cached_mesh(key) {
            return Some(cached);
        }

        let mut mesh = build_text3d_mesh_cpu(obj, data, env);
        mesh = apply_object_effectors(mesh, obj, env);
        if uv_norm {
            apply_normalized_uv_layout(&mut mesh);
        }
        if mesh.indices.is_empty() {
            return None;
        }
        Some(self.insert_cached_mesh(key, &mesh, "zc-text3d-mesh"))
    }

    pub(super) fn ensure_model_mesh(
        &mut self,
        obj: &Object3D,
        data: &zc_project::ModelData,
        env: &EvalEnv<'_>,
    ) -> Option<CachedMeshHandle> {
        let uv_norm = object_uses_normalized_uv(obj);
        let asset_bytes_hash = data
            .asset_id
            .as_deref()
            .and_then(|id| env.project.assets.get(id))
            .map_or(0u64, hash_asset_record);
        let mut h = fnv1a_hash64(b"model3d_v2");
        h = hash_u64(h, asset_bytes_hash);
        if let Some(p) = data.preset_type {
            h = hash_u64(h, u64::from(p));
        }
        let lid_rot = eval_model_lid_rotation(data, env);
        h = hash_u64(h, u64::from(lid_rot.to_bits()));
        h = hash_u64(h, u64::from(uv_norm));
        h = hash_effectors(h, obj, env);

        let key = MeshCacheKey {
            object_id: obj.id,
            content_hash: h,
        };
        if let Some(cached) = self.touch_cached_mesh(key) {
            return Some(cached);
        }

        let mut mesh = build_model_mesh_cpu(data, env);
        mesh = apply_object_effectors(mesh, obj, env);
        if uv_norm {
            apply_normalized_uv_layout(&mut mesh);
        }
        if mesh.indices.is_empty() {
            return None;
        }
        Some(self.insert_cached_mesh(key, &mesh, "zc-model3d-mesh"))
    }

    pub(super) fn touch_cached_mesh(&mut self, key: MeshCacheKey) -> Option<CachedMeshHandle> {
        if let Some(entry) = self.mesh_cache.get_mut(&key) {
            self.cache_tick = self.cache_tick.wrapping_add(1);
            entry.last_used = self.cache_tick;
            Some(CachedMeshHandle {
                mesh: Arc::clone(&entry.mesh),
                bounds: entry.bounds,
                has_vertex_colors: entry.has_vertex_colors,
                material_slots: entry.material_slots.clone(),
            })
        } else {
            None
        }
    }

    pub(super) fn insert_cached_mesh(
        &mut self,
        key: MeshCacheKey,
        mesh: &MeshData,
        label: &str,
    ) -> CachedMeshHandle {
        self.cache_tick = self.cache_tick.wrapping_add(1);
        let tick = self.cache_tick;
        while self.mesh_cache.len() >= MAX_CACHED_MESHES {
            let Some(&oldest) = self
                .mesh_cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(k, _)| k)
            else {
                break;
            };
            self.mesh_cache.remove(&oldest);
        }

        let has_vertex_colors = !mesh.colors.is_empty();
        let mut slots: Vec<u32> = Vec::new();
        for v in &mesh.vertices {
            if !slots.contains(&v.material_slot) {
                slots.push(v.material_slot);
            }
        }
        slots.sort_unstable();
        if slots.is_empty() {
            slots.push(0);
        }
        let gpu_mesh = Arc::new(GpuMesh::from_mesh_data(&self.ctx, mesh, Some(label)));
        self.mesh_cache.insert(
            key,
            CachedGpuMesh {
                mesh: Arc::clone(&gpu_mesh),
                bounds: mesh.bounds,
                has_vertex_colors,
                material_slots: slots.clone(),
                last_used: tick,
            },
        );
        CachedMeshHandle {
            mesh: gpu_mesh,
            bounds: mesh.bounds,
            has_vertex_colors,
            material_slots: slots,
        }
    }

    /// Decodes an animated GIF asset (`GIF87a` / `GIF89a`) once per asset ID and returns the
    /// frame matching `time_secs` (respecting GIF `< 20ms => 100ms` delay rules and NETSCAPE2.0
    /// loop count metadata) as a cached GPU texture.
    pub(super) fn ensure_gif_frame_texture(
        &mut self,
        project: &Project,
        asset_id: &str,
        time_secs: f64,
        premultiply: bool,
    ) -> Option<Arc<RenderTarget>> {
        let record = project.assets.get(asset_id)?;
        if !peek_gif_magic(record) {
            return None;
        }
        let content_hash = asset_record_effective_hash(record)?;
        let needs_decode = self
            .gif_cache
            .get(asset_id)
            .is_none_or(|c| c.content_hash != content_hash);
        if needs_decode {
            const MAX_GIF_FRAMES: usize = 10_000;
            const MAX_GIF_DECODED_BYTES: usize = 256 * 1024 * 1024;
            let bytes_cow = load_asset_bytes(record)?;
            let bytes = bytes_cow.as_ref();
            if bytes.len() < 6 || (&bytes[..6] != b"GIF87a" && &bytes[..6] != b"GIF89a") {
                return None;
            }
            let loop_count = parse_gif_loop_count(bytes);
            let decoder = image::codecs::gif::GifDecoder::new(std::io::Cursor::new(bytes)).ok()?;
            let mut decoded_frames = Vec::new();
            let mut acc_sec = 0.0_f64;
            let mut total_bytes = 0_usize;
            for frame_res in decoder.into_frames() {
                let Ok(frame) = frame_res else {
                    break;
                };
                let (num, den) = frame.delay().numer_denom_ms();
                let ms = f64::from(num) / f64::from(den.max(1));
                // Standard browser / Zoidium GIF delay rule: delays < 20ms default to 100ms.
                let dur = if ms < 20.0 { 0.1 } else { ms / 1000.0 };
                acc_sec += dur;
                let buf = frame.into_buffer();
                let (w, h) = buf.dimensions();
                if self
                    .ctx
                    .validate_dimensions(w, h, wgpu::TextureFormat::Rgba8Unorm)
                    .is_err()
                {
                    return None;
                }
                let raw = buf.into_raw();
                total_bytes = total_bytes.saturating_add(raw.len());
                decoded_frames.push(CachedGifFrame {
                    width: w,
                    height: h,
                    rgba_straight: raw,
                    end_time_sec: acc_sec,
                });
                if decoded_frames.len() >= MAX_GIF_FRAMES || total_bytes >= MAX_GIF_DECODED_BYTES {
                    break;
                }
            }
            if decoded_frames.is_empty() {
                return None;
            }
            while self.gif_cache.len() >= MAX_CACHED_GIFS {
                let Some(oldest) = self
                    .gif_cache
                    .iter()
                    .min_by_key(|(_, entry)| entry.last_used)
                    .map(|(k, _)| k.clone())
                else {
                    break;
                };
                self.gif_cache.remove(&oldest);
            }
            self.cache_tick = self.cache_tick.wrapping_add(1);
            let tick = self.cache_tick;
            self.gif_cache.insert(
                asset_id.to_owned(),
                CachedGifAnimation {
                    content_hash,
                    total_duration_sec: acc_sec.max(1e-4),
                    loop_count,
                    frames: decoded_frames,
                    last_used: tick,
                },
            );
        }

        let (frame_idx, w, h) = {
            let entry = self.gif_cache.get_mut(asset_id)?;
            self.cache_tick = self.cache_tick.wrapping_add(1);
            entry.last_used = self.cache_tick;
            let t_pos = time_secs.max(0.0);
            let idx = if entry.frames.len() <= 1 {
                0
            } else if let Some(max_repeats) = entry.loop_count.filter(|&n| n > 0)
                && t_pos >= entry.total_duration_sec * f64::from(max_repeats + 1)
            {
                entry.frames.len() - 1
            } else {
                let t_mod = t_pos % entry.total_duration_sec;
                entry
                    .frames
                    .iter()
                    .position(|f| t_mod < f.end_time_sec)
                    .unwrap_or(entry.frames.len() - 1)
            };
            let f = &entry.frames[idx];
            (idx, f.width, f.height)
        };

        let key = TextureCacheKey {
            asset_key: format!("gif:{asset_id}#{frame_idx}"),
            content_hash: hash_u64(content_hash, frame_idx as u64),
            premultiply,
        };
        if let Some(cached) = self.touch_cached_texture(&key) {
            return Some(cached);
        }

        let target = {
            let f = &self.gif_cache.get(asset_id)?.frames[frame_idx];
            if premultiply {
                let mut pixels = f.rgba_straight.clone();
                crate::video::fast_premultiply_rgba8(&mut pixels);
                self.ctx
                    .create_rgba8_texture("zc-asset-gif-frame", w, h, &pixels)
                    .ok()?
            } else {
                self.ctx
                    .create_rgba8_texture("zc-asset-gif-frame", w, h, &f.rgba_straight)
                    .ok()?
            }
        };
        Some(self.insert_cached_texture(key, target))
    }

    /// Samples a video asset at `time_secs` if it is a video container; returns `None` for
    /// static images or when video decode is unavailable (caller falls back to image decode).
    pub(super) fn ensure_video_frame_texture(
        &mut self,
        project: &Project,
        asset_id: &str,
        time_secs: f64,
        premultiply: bool,
        target_hint: Option<(u32, u32)>,
    ) -> Option<Arc<RenderTarget>> {
        let record = project.assets.get(asset_id)?;
        if record.kind != zc_project::AssetKind::Av {
            return None;
        }
        let base_hash = hash_asset_record(record);
        let meta = self.video_pool.probe_asset(
            base_hash,
            record.url.as_deref(),
            &record.filename,
            record.bytes.as_deref(),
        )?;
        let max_dim = self.ctx.device.limits().max_texture_dimension_2d;
        let (dec_w, dec_h) =
            crate::video::compute_decode_dimensions(meta.width, meta.height, target_hint, max_dim);
        let clamped_time = crate::video::clamp_video_time(time_secs, &meta);
        // Quantize cache key by the video's native frame index + decode dimensions so hitting
        // the same frame at slightly different float timestamps achieves 100% GPU cache reuse.
        let frame_idx = (clamped_time * meta.fps).round().max(0.0) as u64;
        let mut content_hash = hash_u64(base_hash, frame_idx);
        content_hash = hash_u64(content_hash, (u64::from(dec_w) << 32) | u64::from(dec_h));
        let key = TextureCacheKey {
            asset_key: format!("vid:{asset_id}@{frame_idx}:{dec_w}x{dec_h}"),
            content_hash,
            premultiply,
        };
        if let Some(cached) = self.touch_cached_texture(&key) {
            return Some(cached);
        }

        let decoded = self.video_pool.decode_frame(
            base_hash,
            record.url.as_deref(),
            &record.filename,
            record.bytes.as_deref(),
            clamped_time,
            target_hint,
            max_dim,
        )?;
        if self
            .ctx
            .validate_dimensions(
                decoded.width,
                decoded.height,
                wgpu::TextureFormat::Rgba8Unorm,
            )
            .is_err()
        {
            return None;
        }

        let target = if premultiply && crate::video::has_non_opaque_alpha(&decoded.rgba) {
            let mut pixels = (*decoded.rgba).clone();
            crate::video::fast_premultiply_rgba8(&mut pixels);
            self.ctx
                .create_rgba8_texture(
                    "zc-asset-video-frame",
                    decoded.width,
                    decoded.height,
                    &pixels,
                )
                .ok()?
        } else {
            self.ctx
                .create_rgba8_texture(
                    "zc-asset-video-frame",
                    decoded.width,
                    decoded.height,
                    &decoded.rgba,
                )
                .ok()?
        };
        #[cfg(target_arch = "wasm32")]
        if !crate::video::web_video_frame_is_exact(base_hash, clamped_time, dec_w, dec_h) {
            return Some(Arc::new(target));
        }
        Some(self.insert_cached_texture(key, target))
    }

    // -----------------------------------------------------------------------
    // Decoded Image Texture Cache
    // -----------------------------------------------------------------------

    pub(super) fn ensure_image_asset_texture(
        &mut self,
        project: &Project,
        asset_id: &str,
        premultiply: bool,
    ) -> Option<Arc<RenderTarget>> {
        let record = project.assets.get(asset_id)?;
        let content_hash = asset_record_effective_hash(record)?;
        let key = TextureCacheKey {
            asset_key: asset_id.to_owned(),
            content_hash,
            premultiply,
        };
        if let Some(cached) = self.touch_cached_texture(&key) {
            return Some(cached);
        }

        let bytes_cow = load_asset_bytes(record)?;
        let bytes = bytes_cow.as_ref();
        let max_dim = self.ctx.device.limits().max_texture_dimension_2d;
        let dim_reader =
            match image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format() {
                Ok(reader) => reader,
                Err(err) => {
                    warn!(asset_id, error = %err, "failed to inspect image asset format");
                    return None;
                }
            };
        let (header_w, header_h) = match dim_reader.into_dimensions() {
            Ok(dims) => dims,
            Err(err) => {
                warn!(asset_id, error = %err, "failed to read image asset dimensions");
                return None;
            }
        };
        if self
            .ctx
            .validate_dimensions(header_w, header_h, wgpu::TextureFormat::Rgba8Unorm)
            .is_err()
        {
            warn!(
                asset_id,
                width = header_w,
                height = header_h,
                max_dim,
                "rejecting image asset exceeding GPU texture dimension limits before decode"
            );
            return None;
        }

        let mut decode_reader =
            match image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format() {
                Ok(reader) => reader,
                Err(err) => {
                    warn!(asset_id, error = %err, "failed to prepare image asset decoder");
                    return None;
                }
            };
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(max_dim);
        limits.max_image_height = Some(max_dim);
        limits.max_alloc = Some(
            u64::from(header_w)
                .saturating_mul(u64::from(header_h))
                .saturating_mul(16)
                .clamp(1024 * 1024, 256 * 1024 * 1024),
        );
        decode_reader.limits(limits);

        let img = match decode_reader.decode() {
            Ok(img) => img.to_rgba8(),
            Err(err) => {
                warn!(asset_id, error = %err, "failed to decode image asset");
                return None;
            }
        };
        let (w, h) = img.dimensions();
        if self
            .ctx
            .validate_dimensions(w, h, wgpu::TextureFormat::Rgba8Unorm)
            .is_err()
        {
            return None;
        }
        let mut pixels = img.into_raw();
        if premultiply {
            // Premultiply RGB by alpha to preserve the renderer's premultiplied-alpha invariant.
            crate::video::fast_premultiply_rgba8(&mut pixels);
        }

        let target = self
            .ctx
            .create_rgba8_texture("zc-asset-image", w, h, &pixels)
            .ok()?;
        Some(self.insert_cached_texture(key, target))
    }

    pub(super) fn get_or_build_text_character_meshes(
        &mut self,
        project: &Project,
        font: Option<&str>,
        params: &Text3DParams,
        options: &Text3DExtendedOptions,
        text_scale: Vec3,
    ) -> Arc<Vec<zc_scene::Text3DCharacterMesh>> {
        let mut font_hash = 0u64;
        if let Some(f) = font {
            font_hash = fnv1a_hash64(f.as_bytes());
            if let Some(rec) = project.assets.get(f) {
                font_hash = hash_u64(font_hash, hash_asset_record(rec));
            }
        }
        let key = TextCharMeshCacheKey {
            font_hash,
            text: params.text.clone(),
            height_bits: params.height.to_bits(),
            thickness_bits: params.thickness.to_bits(),
            curve_segments: params.curve_segments,
            bevel_enabled: params.bevel_enabled,
            bevel_thickness_bits: params.bevel_thickness.to_bits(),
            bevel_size_bits: params.bevel_size.to_bits(),
            bevel_segments: params.bevel_segments,
            align: params.align as u8,
            uv_mode: params.uv_mode as u8,
            letter_spacing_bits: params.letter_spacing.to_bits(),
            line_height_bits: params.line_height.to_bits(),
            bevel_side: options.bevel_side as u8,
            bevel_profile: options.bevel_profile as u8,
            bevel_tension_bits: options.bevel_tension.to_bits(),
            kerning: options.kerning,
            use_font_line_height: options.use_font_line_height,
            text_scale_mode: options.text_scale_mode as u8,
            text_scale_bits: [
                text_scale.x.to_bits(),
                text_scale.y.to_bits(),
                text_scale.z.to_bits(),
            ],
        };
        if let Some(entry) = self.text_char_mesh_cache.get_mut(&key) {
            self.cache_tick = self.cache_tick.wrapping_add(1);
            entry.last_used = self.cache_tick;
            return Arc::clone(&entry.meshes);
        }

        let built = Arc::new(build_text3d_character_meshes_with_fallback(
            project, font, params, options, text_scale,
        ));
        while self.text_char_mesh_cache.len() >= MAX_CACHED_TEXT_CHAR_MESHES {
            let Some(oldest) = self
                .text_char_mesh_cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            self.text_char_mesh_cache.remove(&oldest);
        }
        self.cache_tick = self.cache_tick.wrapping_add(1);
        let tick = self.cache_tick;
        self.text_char_mesh_cache.insert(
            key,
            CachedTextCharMeshes {
                meshes: Arc::clone(&built),
                last_used: tick,
            },
        );
        built
    }

    pub(super) fn get_or_parse_cube_lut(
        &mut self,
        record: &zc_project::AssetRecord,
    ) -> Option<Arc<ParsedCubeLut>> {
        let base_hash = asset_record_effective_hash(record)?;
        let max_dim = self.ctx.device.limits().max_texture_dimension_2d;
        let key = hash_u64(base_hash, u64::from(max_dim));
        if let Some(entry) = self.cube_lut_cache.get_mut(&key) {
            self.cache_tick = self.cache_tick.wrapping_add(1);
            entry.last_used = self.cache_tick;
            return Some(Arc::clone(&entry.parsed));
        }

        let bytes_cow = load_asset_bytes(record)?;
        let (lut, domain_min, domain_max) = parse_cube_lut_bytes(bytes_cow.as_ref(), max_dim)?;
        let parsed = Arc::new(ParsedCubeLut {
            lut,
            domain_min,
            domain_max,
        });
        while self.cube_lut_cache.len() >= MAX_CACHED_CUBE_LUTS {
            let Some(oldest) = self
                .cube_lut_cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(&k, _)| k)
            else {
                break;
            };
            self.cube_lut_cache.remove(&oldest);
        }
        self.cache_tick = self.cache_tick.wrapping_add(1);
        let tick = self.cache_tick;
        self.cube_lut_cache.insert(
            key,
            CachedCubeLut {
                parsed: Arc::clone(&parsed),
                last_used: tick,
            },
        );
        Some(parsed)
    }

    pub(super) fn get_or_compile_node_effect(
        &mut self,
        graph: &serde_json::Value,
    ) -> Result<Arc<CompiledNodeEffect>, String> {
        let key = hash_json_value(fnv1a_hash64(b"node_effect_v1"), graph);
        if let Some(entry) = self.node_effect_cache.get_mut(&key) {
            self.cache_tick = self.cache_tick.wrapping_add(1);
            entry.last_used = self.cache_tick;
            return Ok(Arc::clone(&entry.compiled));
        }

        let compiled = Arc::new(compile_node_effect_to_glsl(graph)?);
        while self.node_effect_cache.len() >= MAX_CACHED_NODE_EFFECTS {
            let Some(oldest) = self
                .node_effect_cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(&k, _)| k)
            else {
                break;
            };
            self.node_effect_cache.remove(&oldest);
        }
        self.cache_tick = self.cache_tick.wrapping_add(1);
        let tick = self.cache_tick;
        self.node_effect_cache.insert(
            key,
            CachedNodeEffect {
                compiled: Arc::clone(&compiled),
                last_used: tick,
            },
        );
        Ok(compiled)
    }

    pub(super) fn touch_cached_texture(
        &mut self,
        key: &TextureCacheKey,
    ) -> Option<Arc<RenderTarget>> {
        if let Some(entry) = self.texture_cache.get_mut(key) {
            self.cache_tick = self.cache_tick.wrapping_add(1);
            entry.last_used = self.cache_tick;
            Some(Arc::clone(&entry.target))
        } else {
            None
        }
    }

    pub(super) fn insert_cached_texture(
        &mut self,
        key: TextureCacheKey,
        target: RenderTarget,
    ) -> Arc<RenderTarget> {
        let bytes = target.byte_size();
        self.cache_tick = self.cache_tick.wrapping_add(1);
        let tick = self.cache_tick;

        if let Some(prev) = self.texture_cache.remove(&key) {
            self.texture_cache_bytes = self.texture_cache_bytes.saturating_sub(prev.bytes);
        }

        while self.texture_cache.len() >= MAX_CACHED_TEXTURES
            || (self.texture_cache_bytes.saturating_add(bytes) > MAX_CACHED_TEXTURE_BYTES
                && !self.texture_cache.is_empty())
        {
            let Some(oldest) = self
                .texture_cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            if let Some(evicted) = self.texture_cache.remove(&oldest) {
                self.texture_cache_bytes = self.texture_cache_bytes.saturating_sub(evicted.bytes);
            }
        }

        let target = Arc::new(target);
        self.texture_cache_bytes = self.texture_cache_bytes.saturating_add(bytes);
        self.texture_cache.insert(
            key,
            CachedGpuTexture {
                target: Arc::clone(&target),
                bytes,
                last_used: tick,
            },
        );
        target
    }
}

pub(super) fn json_vec3(val: Option<&serde_json::Value>, default: Vec3) -> Vec3 {
    let Some(arr) = val.and_then(serde_json::Value::as_array) else {
        return default;
    };
    if arr.len() < 3 {
        return default;
    }
    Vec3::new(
        arr[0].as_f64().unwrap_or(f64::from(default.x)) as f32,
        arr[1].as_f64().unwrap_or(f64::from(default.y)) as f32,
        arr[2].as_f64().unwrap_or(f64::from(default.z)) as f32,
    )
}

pub(super) fn hash_u64(acc: u64, value: u64) -> u64 {
    let mut h = acc ^ value;
    h = h.wrapping_mul(0x0100_0000_01b3);
    h
}

pub(super) fn hash_f64(acc: u64, value: f64) -> u64 {
    hash_u64(acc, value.to_bits())
}

pub(super) fn hash_json_value(mut h: u64, v: &serde_json::Value) -> u64 {
    match v {
        serde_json::Value::Null => hash_u64(h, 0x01),
        serde_json::Value::Bool(b) => hash_u64(hash_u64(h, 0x02), u64::from(*b)),
        serde_json::Value::Number(n) => {
            h = hash_u64(h, 0x03);
            if let Some(u) = n.as_u64() {
                hash_u64(hash_u64(h, 1), u)
            } else if let Some(i) = n.as_i64() {
                hash_u64(hash_u64(h, 2), i as u64)
            } else if let Some(f) = n.as_f64() {
                hash_f64(hash_u64(h, 3), f)
            } else {
                h
            }
        }
        serde_json::Value::String(s) => {
            h = hash_u64(h, 0x04);
            h = hash_u64(h, s.len() as u64);
            hash_u64(h, fnv1a_hash64(s.as_bytes()))
        }
        serde_json::Value::Array(arr) => {
            h = hash_u64(h, 0x05);
            h = hash_u64(h, arr.len() as u64);
            for item in arr {
                h = hash_json_value(h, item);
            }
            h
        }
        serde_json::Value::Object(map) => {
            h = hash_u64(h, 0x06);
            h = hash_u64(h, map.len() as u64);
            for (k, val) in map {
                h = hash_u64(h, k.len() as u64);
                h = hash_u64(h, fnv1a_hash64(k.as_bytes()));
                h = hash_json_value(h, val);
            }
            h
        }
    }
}

pub(super) fn hash_node_material_referenced_assets(
    mut h: u64,
    v: &serde_json::Value,
    project: &Project,
) -> u64 {
    if project.assets.is_empty() {
        return h;
    }
    match v {
        serde_json::Value::String(s) => {
            if let Some(rec) = project.assets.get(s.as_str()) {
                h = hash_u64(h, fnv1a_hash64(s.as_bytes()));
                h = hash_u64(h, hash_asset_record(rec));
            } else if s.as_bytes().first() == Some(&b'{') {
                // Handle `.pz`-wrapped stringified JSON graphs without parsing/decoding on cache hits.
                for (aid, rec) in &project.assets {
                    if s.contains(aid.as_str()) {
                        h = hash_u64(h, fnv1a_hash64(aid.as_bytes()));
                        h = hash_u64(h, hash_asset_record(rec));
                    }
                }
            }
            h
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                h = hash_node_material_referenced_assets(h, item, project);
            }
            h
        }
        serde_json::Value::Object(map) => {
            for val in map.values() {
                h = hash_node_material_referenced_assets(h, val, project);
            }
            h
        }
        _ => h,
    }
}

pub(super) fn shape_plus_is_time_dependent(payload: &serde_json::Value) -> bool {
    match payload {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                if k == "keyframes" && v.as_array().is_some_and(|arr| !arr.is_empty()) {
                    return true;
                }
                if k == "expression" && v.as_str().is_some_and(|s| !s.trim().is_empty()) {
                    return true;
                }
                if matches!(
                    k.as_str(),
                    "wigglePaths" | "wiggleTransform" | "textAnimator"
                ) {
                    return true;
                }
                if k == "type"
                    && v.as_str().is_some_and(|s| {
                        matches!(s, "wigglePaths" | "wiggleTransform" | "textAnimator")
                    })
                {
                    return true;
                }
                if shape_plus_is_time_dependent(v) {
                    return true;
                }
            }
            false
        }
        serde_json::Value::Array(arr) => arr.iter().any(shape_plus_is_time_dependent),
        _ => false,
    }
}

pub(super) fn hash_effectors(mut h: u64, obj: &Object3D, env: &EvalEnv<'_>) -> u64 {
    let active = env.plugins.is_enabled("repeater") || env.plugins.is_enabled("effector-plus");
    h = hash_u64(h, u64::from(active));
    if active && let Some(val) = obj.preserved.get("effectors") {
        h = hash_json_value(h, val);
    }
    h
}

#[cfg(not(target_arch = "wasm32"))]
const MAX_EXTERNAL_ASSET_BYTES: u64 = 256 * 1024 * 1024;

#[cfg(not(target_arch = "wasm32"))]
fn decode_percent_encoded_path(raw: &str) -> Option<String> {
    if !raw.contains('%') {
        return None;
    }
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(hi), Some(lo)) = (
                (bytes[i + 1] as char).to_digit(16),
                (bytes[i + 2] as char).to_digit(16),
            )
        {
            out.push(((hi << 4) | lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_asset_disk_path(url: &str) -> Option<PathBuf> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("data:")
        || lower.starts_with("blob:")
    {
        return None;
    }
    let raw_path = if let Some(rest) = trimmed.strip_prefix("file://") {
        #[cfg(windows)]
        {
            let b = rest.as_bytes();
            if b.len() >= 3 && b[0] == b'/' && b[1].is_ascii_alphabetic() && b[2] == b':' {
                &rest[1..]
            } else {
                rest
            }
        }
        #[cfg(not(windows))]
        {
            rest
        }
    } else {
        trimmed
    };
    let direct = Path::new(raw_path);
    if direct.is_file() {
        return Some(direct.to_path_buf());
    }
    if let Some(decoded) = decode_percent_encoded_path(raw_path) {
        let dec_path = Path::new(&decoded);
        if dec_path.is_file() {
            return Some(dec_path.to_path_buf());
        }
    }
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn disk_file_content_hash(url: &str, path: &Path) -> Option<u64> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() == 0 || meta.len() > MAX_EXTERNAL_ASSET_BYTES {
        return None;
    }
    let mtime_nanos = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0u64, |d| d.as_nanos() as u64);
    let mut h = fnv1a_hash64(b"disk_asset_v1");
    h = hash_u64(h, fnv1a_hash64(url.as_bytes()));
    h = hash_u64(h, meta.len());
    h = hash_u64(h, mtime_nanos);
    Some(h)
}

fn asset_record_effective_hash(record: &zc_project::AssetRecord) -> Option<u64> {
    if let Some(bytes) = record.bytes.as_deref() {
        return Some(if !record.sha256.is_empty() {
            fnv1a_hash64(record.sha256.as_bytes())
        } else {
            fnv1a_hash64(bytes)
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(url) = record.url.as_deref()
        && let Some(path) = resolve_asset_disk_path(url)
    {
        if !record.sha256.is_empty() {
            return Some(fnv1a_hash64(record.sha256.as_bytes()));
        }
        return disk_file_content_hash(url, &path);
    }
    None
}

fn load_asset_bytes<'a>(record: &'a zc_project::AssetRecord) -> Option<Cow<'a, [u8]>> {
    if let Some(bytes) = record.bytes.as_deref() {
        return Some(Cow::Borrowed(bytes));
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(url) = record.url.as_deref()
        && let Some(path) = resolve_asset_disk_path(url)
    {
        let meta = std::fs::metadata(&path).ok()?;
        if !meta.is_file() || meta.len() == 0 || meta.len() > MAX_EXTERNAL_ASSET_BYTES {
            return None;
        }
        let data = std::fs::read(&path).ok()?;
        return Some(Cow::Owned(data));
    }
    None
}

fn peek_gif_magic(record: &zc_project::AssetRecord) -> bool {
    if let Some(bytes) = record.bytes.as_deref() {
        return bytes.len() >= 6 && (&bytes[..6] == b"GIF87a" || &bytes[..6] == b"GIF89a");
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(url) = record.url.as_deref()
        && let Some(path) = resolve_asset_disk_path(url)
    {
        use std::io::Read;
        let Ok(mut file) = std::fs::File::open(&path) else {
            return false;
        };
        let mut header = [0u8; 6];
        if file.read_exact(&mut header).is_ok() {
            return &header == b"GIF87a" || &header == b"GIF89a";
        }
    }
    false
}

pub(super) fn hash_asset_record(rec: &zc_project::AssetRecord) -> u64 {
    if !rec.sha256.is_empty() {
        fnv1a_hash64(rec.sha256.as_bytes())
    } else if let Some(bytes) = rec.bytes.as_deref() {
        if bytes.len() <= 1_048_576 {
            fnv1a_hash64(bytes)
        } else {
            let sample = 8192usize;
            let mid = bytes.len() / 2;
            let mut h = hash_u64(fnv1a_hash64(b"large_asset_v1"), bytes.len() as u64);
            h = hash_u64(h, fnv1a_hash64(&bytes[..sample]));
            h = hash_u64(h, fnv1a_hash64(&bytes[mid..mid + sample]));
            h = hash_u64(h, fnv1a_hash64(&bytes[bytes.len() - sample..]));
            h
        }
    } else if let Some(url) = rec.url.as_deref() {
        hash_u64(fnv1a_hash64(url.as_bytes()), rec.size)
    } else if !rec.filename.is_empty() {
        hash_u64(fnv1a_hash64(rec.filename.as_bytes()), rec.size)
    } else {
        0
    }
}

pub(super) fn hash_shape_and_effectors(
    shape: &Shape3DKind,
    obj: &Object3D,
    env: &EvalEnv<'_>,
) -> u64 {
    let mut h = fnv1a_hash64(b"shape3d");
    if let Ok(bytes) = serde_json::to_vec(shape) {
        h = hash_u64(h, fnv1a_hash64(&bytes));
    }
    if let Shape3DKind::CustomGeometry { asset_id: Some(id) } = shape
        && let Some(rec) = env.project.assets.get(id)
    {
        h = hash_u64(h, hash_asset_record(rec));
    }
    hash_effectors(h, obj, env)
}

pub(super) fn hash_text3d_and_effectors(
    data: &zc_project::Text3DData,
    obj: &Object3D,
    env: &EvalEnv<'_>,
) -> u64 {
    let mut h = fnv1a_hash64(b"text3d_v2");
    h = hash_u64(h, fnv1a_hash64(data.text.as_bytes()));
    if let Some(font) = data.font.as_deref() {
        h = hash_u64(h, fnv1a_hash64(font.as_bytes()));
        if let Some(rec) = env.project.assets.get(font) {
            h = hash_u64(h, hash_asset_record(rec));
        }
    }
    h = hash_f64(h, data.size[0]);
    h = hash_f64(h, data.size[1]);
    h = hash_u64(h, u64::from(data.detail));
    h = hash_u64(h, u64::from(data.bevel));
    h = hash_f64(h, data.bevel_thickness);
    h = hash_f64(h, data.bevel_size);
    h = hash_u64(h, u64::from(data.bevel_side));
    h = hash_u64(h, u64::from(data.bevel_detail));
    h = hash_u64(h, u64::from(data.bevel_profile));
    h = hash_f64(h, data.bevel_tension);
    h = hash_f64(h, data.spacing);
    h = hash_u64(h, data.text_scale_mode as u64);
    h = hash_u64(h, data.position_mode as u64);
    h = hash_u64(h, obj.zoidium_uv_mode as u64);
    let kerning = parse_preserved_bool_flag(
        obj.preserved
            .get("kerning")
            .or_else(|| obj.preserved.get("property.kerning")),
        false,
    );
    h = hash_u64(h, u64::from(kerning));
    hash_effectors(h, obj, env)
}

pub(super) enum ResolvedFont<'a> {
    Borrowed(&'a [u8]),
    #[cfg(not(target_arch = "wasm32"))]
    Owned(Arc<[u8]>),
}

impl ResolvedFont<'_> {
    pub(super) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(b) => b,
            #[cfg(not(target_arch = "wasm32"))]
            Self::Owned(b) => b,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) const MAX_SYSTEM_FONT_BYTES: u64 = 64 * 1024 * 1024;

#[cfg(not(target_arch = "wasm32"))]
pub(super) type DiskFontCache = Mutex<HashMap<PathBuf, Option<Arc<[u8]>>>>;

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn load_font_file_cached(path: &Path) -> Option<Arc<[u8]>> {
    static DISK_FONT_CACHE: OnceLock<DiskFontCache> = OnceLock::new();
    let cache = DISK_FONT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(guard) = cache.lock()
        && let Some(entry) = guard.get(path)
    {
        return entry.clone();
    }
    let loaded: Option<Arc<[u8]>> = (|| {
        let meta = std::fs::metadata(path).ok()?;
        if !meta.is_file() || meta.len() == 0 || meta.len() > MAX_SYSTEM_FONT_BYTES {
            return None;
        }
        let bytes = std::fs::read(path).ok()?;
        Some(Arc::<[u8]>::from(bytes))
    })();
    if let Ok(mut guard) = cache.lock() {
        guard.insert(path.to_path_buf(), loaded.clone());
    }
    loaded
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn normalize_font_token(raw: &str) -> String {
    raw.trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| !matches!(c, ' ' | '-' | '_' | '.'))
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn system_font_index() -> &'static [(String, PathBuf)] {
    static INDEX: OnceLock<Vec<(String, PathBuf)>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut dirs: Vec<PathBuf> = Vec::new();
        #[cfg(target_os = "macos")]
        {
            dirs.push(PathBuf::from("/System/Library/Fonts"));
            dirs.push(PathBuf::from("/System/Library/Fonts/Supplemental"));
            dirs.push(PathBuf::from("/Library/Fonts"));
            if let Some(home) = std::env::var_os("HOME") {
                dirs.push(PathBuf::from(home).join("Library/Fonts"));
            }
        }
        #[cfg(target_os = "windows")]
        {
            dirs.push(PathBuf::from(r"C:\Windows\Fonts"));
        }
        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        {
            dirs.push(PathBuf::from("/usr/share/fonts"));
            dirs.push(PathBuf::from("/usr/local/share/fonts"));
            if let Some(home) = std::env::var_os("HOME") {
                let h = PathBuf::from(home);
                dirs.push(h.join(".local/share/fonts"));
                dirs.push(h.join(".fonts"));
            }
        }

        let mut out = Vec::new();
        let mut stack = dirs;
        let mut visited_dirs = 0usize;
        while let Some(dir) = stack.pop() {
            if visited_dirs >= 64 {
                break;
            }
            visited_dirs += 1;
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_ascii_lowercase);
                if matches!(ext.as_deref(), Some("ttf" | "otf" | "ttc"))
                    && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                {
                    out.push((normalize_font_token(stem), path));
                }
            }
        }
        out
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn resolve_system_font_by_name(font_key: &str) -> Option<Arc<[u8]>> {
    let trimmed = font_key.trim();
    if trimmed.is_empty() {
        return None;
    }
    // 1. Direct path on disk (supports `.ttf`, `.otf`, `.ttc`)
    let as_path = Path::new(trimmed);
    if as_path.is_file() {
        return load_font_file_cached(as_path);
    }

    // 2. Common family aliases (e.g. "Hiragino Sans" -> macOS CJK TTC, "Helvetica" -> Helvetica.ttc)
    let norm = normalize_font_token(trimmed);
    #[cfg(target_os = "macos")]
    {
        let alias_candidates: &[&str] = match norm.as_str() {
            "hiraginosans" | "hiraginokakugothicpron" | "hiraginokakugothicstd" => &[
                "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
            ],
            "hiraginosansgb" => &["/System/Library/Fonts/Hiragino Sans GB.ttc"],
            "helvetica" => &["/System/Library/Fonts/Helvetica.ttc"],
            "helveticaneue" => &["/System/Library/Fonts/HelveticaNeue.ttc"],
            "arial" => &[
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            ],
            "arialunicode" | "arialunicodems" => {
                &["/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]
            }
            _ => &[],
        };
        for cand in alias_candidates {
            if let Some(bytes) = load_font_file_cached(Path::new(cand)) {
                return Some(bytes);
            }
        }
    }

    // 3. Index lookup across OS font directories by normalized stem (exact match first, then prefix).
    let index = system_font_index();
    if let Some((_, path)) = index.iter().find(|(stem, _)| stem == &norm) {
        return load_font_file_cached(path);
    }
    if norm.len() >= 4
        && let Some((_, path)) = index.iter().find(|(stem, _)| stem.starts_with(&norm))
    {
        return load_font_file_cached(path);
    }
    None
}

#[cfg(any(test, not(target_arch = "wasm32")))]
pub(super) fn font_supports_cjk(bytes: &[u8]) -> bool {
    if font_has_missing_glyphs(bytes, "あテ中") {
        return false;
    }
    let params = zc_scene::Text3DParams {
        text: "テ".to_owned(),
        height: 16.0,
        thickness: 1.0,
        curve_segments: 2,
        ..Default::default()
    };
    zc_scene::build_text3d_mesh(bytes, &params)
        .is_ok_and(|mesh| !mesh.vertices.is_empty() && !mesh.indices.is_empty())
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn resolve_system_cjk_fallback_font() -> Option<Arc<[u8]>> {
    static CJK_FALLBACK: OnceLock<Option<Arc<[u8]>>> = OnceLock::new();
    CJK_FALLBACK
        .get_or_init(|| {
            let candidates: &[&str] = {
                #[cfg(target_os = "macos")]
                {
                    &[
                        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
                        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                        "/System/Library/Fonts/Hiragino Sans GB.ttc",
                        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                        "/System/Library/Fonts/STHeiti Medium.ttc",
                    ]
                }
                #[cfg(target_os = "windows")]
                {
                    &[
                        r"C:\Windows\Fonts\YuGothM.ttc",
                        r"C:\Windows\Fonts\YuGothR.ttc",
                        r"C:\Windows\Fonts\meiryo.ttc",
                        r"C:\Windows\Fonts\msgothic.ttc",
                        r"C:\Windows\Fonts\msyh.ttc",
                    ]
                }
                #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
                {
                    &[
                        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
                        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
                    ]
                }
            };
            for path_str in candidates {
                if let Some(bytes) = load_font_file_cached(Path::new(path_str))
                    && font_supports_cjk(&bytes)
                {
                    return Some(bytes);
                }
            }
            None
        })
        .clone()
}

#[cfg(target_arch = "wasm32")]
pub(super) fn resolve_system_cjk_fallback_font() -> Option<Arc<[u8]>> {
    None
}

pub(super) fn resolve_font_bytes<'a>(
    project: &'a Project,
    font: Option<&str>,
) -> Option<ResolvedFont<'a>> {
    let font_key = font?.trim();
    if font_key.is_empty() {
        return None;
    }
    if let Some(rec) = project.assets.get(font_key) {
        if let Some(bytes) = rec.bytes.as_deref() {
            return Some(ResolvedFont::Borrowed(bytes));
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(url) = rec.url.as_deref()
            && let Some(path) = resolve_asset_disk_path(url)
            && let Some(bytes) = load_font_file_cached(&path)
        {
            return Some(ResolvedFont::Owned(bytes));
        }
    }
    // Fallback lookup by URL, filename, or legacy `.pz` asset key in `project.assets`.
    for rec in project.assets.values() {
        let matches = rec.url.as_deref() == Some(font_key)
            || rec.filename == font_key
            || rec
                .preserved
                .get("legacy_pz_asset_key")
                .and_then(serde_json::Value::as_str)
                == Some(font_key);
        if matches {
            if let Some(bytes) = rec.bytes.as_deref() {
                return Some(ResolvedFont::Borrowed(bytes));
            }
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(url) = rec.url.as_deref()
                && let Some(path) = resolve_asset_disk_path(url)
                && let Some(bytes) = load_font_file_cached(&path)
            {
                return Some(ResolvedFont::Owned(bytes));
            }
        }
    }
    // Built-in bundled OFL fonts available offline without embedding.
    match font_key.to_ascii_lowercase().as_str() {
        "geist-regular.ttf" | "builtin:geist" | "geist" => {
            return Some(ResolvedFont::Borrowed(DEFAULT_TTF_FONT_BYTES));
        }
        "geistmono-regular.ttf" | "builtin:geist-mono" | "geist mono" => {
            return Some(ResolvedFont::Borrowed(GEIST_MONO_TTF_FONT_BYTES));
        }
        "sourcecodepro-regular.ttf" | "builtin:source-code-pro" | "source code pro" => {
            return Some(ResolvedFont::Borrowed(SOURCE_CODE_PRO_TTF_FONT_BYTES));
        }
        _ => {}
    }
    // System font path (`.ttf`/`.otf`/`.ttc`) or installed OS font family name (cached).
    #[cfg(not(target_arch = "wasm32"))]
    {
        resolve_system_font_by_name(font_key).map(ResolvedFont::Owned)
    }
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
}
