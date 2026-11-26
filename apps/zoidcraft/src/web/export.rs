use std::sync::mpsc::Sender;

use js_sys::Object;

use super::*;

pub(crate) fn frame_export_settings(settings: &UiExportSettings, frame: f64) -> UiExportSettings {
    let mut settings = settings.clone();
    settings.container = UiExportContainer::PngSequence;
    settings.start_frame = frame;
    settings.end_frame = frame + 1.0;
    settings
}

impl ZoidCraftApp {
    pub(crate) fn validate_web_export(&self, settings: &CoreExportSettings) -> Result<(), String> {
        if self.web.project_videos_pending > 0 {
            return Err("Wait for the project videos to finish loading before exporting.".into());
        }
        if settings.container == CoreExportContainer::Gif {
            return Err(
                "GIF export is unavailable in the web edition. Choose a PNG sequence or WebM."
                    .into(),
            );
        }
        if !matches!(
            settings.container,
            CoreExportContainer::PngSequence | CoreExportContainer::Wav
        ) && !self.workspace.panels.export_status.ffmpeg_available
        {
            return Err("This browser does not expose WebCodecs encoding.".into());
        }
        Ok(())
    }

    pub(crate) fn prompt_export_output_path(
        &mut self,
        settings: &UiExportSettings,
    ) -> Option<PathBuf> {
        self.pause_playback_for_modal_dialog();
        let base = sanitize_filename(&self.session.document().project_name);
        if settings.mode == ExportMode::Frame {
            let frame = self.session.playback.current_frame.round() as i64;
            return Some(PathBuf::from(format!("{base}_frame_{frame}.png")));
        }
        let ext = match settings.container {
            UiExportContainer::Mp4 => "mp4",
            UiExportContainer::Mkv
                if matches!(
                    settings.video_codec,
                    ExportVideoCodec::H264 | ExportVideoCodec::H265
                ) =>
            {
                "mp4"
            }
            UiExportContainer::WebM | UiExportContainer::Mkv => "webm",
            UiExportContainer::PngSequence => "zip",
            UiExportContainer::Wav => "wav",
            UiExportContainer::M4a => "m4a",
            UiExportContainer::Gif => return None,
        };
        Some(PathBuf::from(format!("{base}.{ext}")))
    }
}

pub(crate) struct BrowserExport {
    pub(crate) settings: CoreExportSettings,
    pub(crate) project: Project,
    pub(crate) plugins: PluginRegistry,
    pub(crate) audio_buffers: HashMap<String, Arc<AudioBuffer>>,
    pub(crate) seq_fps: f64,
    pub(crate) start_seq_frame: u32,
    pub(crate) end_seq_frame: u32,
    pub(crate) single_frame: Option<f64>,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) events: Sender<ExportWorkerEvent>,
}

pub(crate) fn launch_export(
    renderer: Option<&ViewportGpuPresenter>,
    job: BrowserExport,
) -> Result<(), String> {
    let gpu = if job.settings.container.is_audio_only() {
        None
    } else {
        let renderer = renderer.ok_or("The project renderer is unavailable")?;
        Some((
            Arc::clone(&renderer.engine.ctx.device),
            Arc::clone(&renderer.engine.ctx.queue),
        ))
    };
    spawn_local(async move {
        let events = job.events.clone();
        let result = run_export(job, gpu).await;
        let _ = events.send(ExportWorkerEvent::Finished(result));
    });
    Ok(())
}

fn check_cancel(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Acquire) {
        return Err("Export cancelled".into());
    }
    Ok(())
}

async fn run_export(
    job: BrowserExport,
    gpu: Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)>,
) -> Result<ExportReport, String> {
    // Give the editor a chance to paint the progress state before mixing.
    call_async(FILES, "yieldToBrowser", &[]).await?;
    check_cancel(&job.cancel)?;
    let BrowserExport {
        settings,
        project,
        plugins,
        audio_buffers,
        seq_fps,
        start_seq_frame,
        end_seq_frame,
        single_frame,
        cancel,
        events,
    } = job;
    let mut settings = settings;
    if settings.audio_codec == AudioCodec::Opus
        || (settings.container == CoreExportContainer::Mkv
            && !matches!(settings.video_codec, VideoCodec::H264 | VideoCodec::H265)
            && settings.audio_codec != AudioCodec::None)
    {
        settings.sample_rate = STANDARD_SAMPLE_RATE;
    }
    let settings = &settings;
    let session = Session::from_project(project, plugins);
    let mut audio = build_offline_export_audio(
        &session,
        &audio_buffers,
        settings,
        seq_fps,
        start_seq_frame,
        end_seq_frame,
    )?;
    let total_frames = settings.total_frames();
    if settings.container.is_audio_only() && audio.is_none() {
        let count = zc_audio::try_offline_total_samples(
            f64::from(total_frames),
            settings.fps,
            settings.sample_rate,
        )
        .map_err(|err| err.to_string())?;
        audio = Some(vec![[0.0; 2]; count]);
    }
    let (output_width, output_height) = if settings.container.is_audio_only() {
        (0, 0)
    } else {
        settings.padded_dimensions()
    };
    let mut output_path = settings.output_path.clone();
    if settings.container == CoreExportContainer::PngSequence && total_frames == 1 {
        output_path.set_extension("png");
    }
    let filename = output_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("Missing export filename")?;
    let audio_frames = audio.as_ref().map_or(0, Vec::len);
    let report = ExportReport {
        frames_exported: if settings.container.is_audio_only() {
            0
        } else {
            total_frames
        },
        output_width,
        output_height,
        audio_samples_written: audio_frames,
        output_path: output_path.clone(),
    };
    if settings.container == CoreExportContainer::Wav {
        let wav = zc_audio::encode_wav_pcm16_stereo(
            audio.as_deref().unwrap_or_default(),
            settings.sample_rate,
        );
        check_cancel(&cancel)?;
        download(filename, &wav, "audio/wav")?;
        return Ok(report);
    }

    let mut engine = match gpu {
        Some((device, queue)) => {
            let mut engine = ProjectRenderEngine::from_device_queue(device, queue)
                .map_err(|err| err.to_string())?;
            engine.set_transparent_background(settings.transparent_alpha);
            Some(engine)
        }
        None => None,
    };
    let may_have_video = session.project.assets.values().any(|asset| {
        Path::new(&asset.filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(is_video_extension)
    });
    let export_id = start_export(settings, filename, audio.as_deref()).await?;
    drop(audio);
    let result = async {
        check_cancel(&cancel)?;
        if let Some(engine) = &mut engine {
            let mut padded = Vec::new();
            for offset in 0..total_frames {
                check_cancel(&cancel)?;
                let sequence_frame = single_frame.unwrap_or_else(|| {
                    f64::from(start_seq_frame) + f64::from(offset) * (seq_fps / settings.fps)
                });
                if may_have_video {
                    call(CODECS, "beginFrameRequests", &[])?;
                    let result = engine
                        .render_frame_to_texture(
                            &session.project,
                            &session.plugins,
                            sequence_frame,
                            settings.width,
                            settings.height,
                        )
                        .map_err(|err| err.to_string());
                    let requests = call(CODECS, "endFrameRequests", &[])?;
                    let frames = JsFuture::from(Promise::resolve(&requests))
                        .await
                        .map_err(js_error)?;
                    result?;
                    check_cancel(&cancel)?;
                    call(CODECS, "usePreparedFrames", &[frames])?;
                }
                let pixels = engine
                    .render_frame_rgba8_async(
                        &session.project,
                        &session.plugins,
                        sequence_frame,
                        settings.width,
                        settings.height,
                    )
                    .await;
                if may_have_video {
                    call(CODECS, "usePreparedFrames", &[JsValue::NULL])?;
                }
                let mut rgba = pixels.map_err(|err| err.to_string())?;
                check_cancel(&cancel)?;
                if settings.transparent_alpha {
                    zc_render::unpremultiply_rgba8_in_place(&mut rgba);
                } else {
                    for pixel in rgba.as_chunks_mut::<4>().0 {
                        pixel[3] = 255;
                    }
                }
                let (method, bytes) = if settings.container == CoreExportContainer::PngSequence {
                    let png = encode_png(&rgba, settings.width, settings.height)
                        .map_err(|err| err.to_string())?;
                    ("encodeExportPng", Uint8Array::from(png.as_slice()))
                } else if (settings.width, settings.height) != (output_width, output_height) {
                    zc_export::pad_rgba_frame_into(
                        &rgba,
                        (settings.width, settings.height),
                        (output_width, output_height),
                        settings.transparent_alpha,
                        &mut padded,
                    )
                    .map_err(|err| err.to_string())?;
                    ("encodeExportFrame", Uint8Array::from(padded.as_slice()))
                } else {
                    ("encodeExportFrame", Uint8Array::from(rgba.as_slice()))
                };
                call_async(
                    CODECS,
                    method,
                    &[export_id.clone(), bytes.into(), offset.into()],
                )
                .await?;
                let _ = events.send(ExportWorkerEvent::Progress(ExportProgress {
                    completed_frames: offset + 1,
                    total_frames,
                    frame_index: offset,
                }));
                call_async(FILES, "yieldToBrowser", &[]).await?;
            }
        }
        check_cancel(&cancel)?;
        // Let the encoder flush before checking for cancellation and handing off the download.
        let output = call_async(CODECS, "finishExport", &[export_id.clone(), false.into()]).await?;
        check_cancel(&cancel)?;
        call(
            FILES,
            "downloadBlob",
            &[filename.into(), output, JsValue::UNDEFINED],
        )?;
        Ok(report)
    }
    .await;
    if result.is_err() {
        let _ = call(CODECS, "cancelExport", &[export_id]);
    }
    result
}

async fn start_export(
    settings: &CoreExportSettings,
    filename: &str,
    audio: Option<&[[f32; 2]]>,
) -> Result<JsValue, String> {
    let video_codec = match settings.video_codec {
        VideoCodec::H264 => "h264",
        VideoCodec::H265 => "h265",
        VideoCodec::Vp9 => "vp9",
        VideoCodec::Vp8 => "vp8",
        VideoCodec::Av1 => "av1",
    };
    let container = match settings.container {
        CoreExportContainer::Mp4 => "mp4",
        CoreExportContainer::Webm => "webm",
        CoreExportContainer::Mkv
            if matches!(settings.video_codec, VideoCodec::H264 | VideoCodec::H265) =>
        {
            "mp4"
        }
        CoreExportContainer::Mkv => "webm",
        CoreExportContainer::M4a => "m4a",
        CoreExportContainer::PngSequence => "png",
        CoreExportContainer::Wav | CoreExportContainer::Gif => {
            return Err("Unsupported WebCodecs container".into());
        }
    };
    let audio_codec = match (settings.audio_codec, container) {
        (AudioCodec::None, _) => "none",
        (_, "webm") => "opus",
        (AudioCodec::Opus, _) => "opus",
        _ => "aac",
    };
    let bitrate_kbps = match settings.quality {
        QualityMode::BitrateKbps(kbps) => kbps,
        QualityMode::Crf(crf) => {
            // WebCodecs exposes bitrate instead of CRF. Scale the default by pixel rate and quality.
            let pixels_per_second =
                f64::from(settings.width) * f64::from(settings.height) * settings.fps;
            (pixels_per_second * 0.1 * 2.0_f64.powf((23.0 - f64::from(crf)) / 12.0) / 1000.0)
                .round()
                .clamp(100.0, 150_000.0) as u32
        }
    };
    let js_audio = audio.map_or(JsValue::NULL, |samples| {
        Float32Array::from(samples.as_flattened()).into()
    });
    let (width, height) = settings.padded_dimensions();
    let options = Object::new();
    for (key, value) in [
        ("filename", filename.into()),
        ("width", width.into()),
        ("height", height.into()),
        ("fps", settings.fps.into()),
        ("totalFrames", settings.total_frames().into()),
        ("container", container.into()),
        ("videoCodec", video_codec.into()),
        ("audioCodec", audio_codec.into()),
        ("bitrateKbps", bitrate_kbps.into()),
        ("audioBitrateKbps", settings.audio_bitrate_kbps.into()),
        ("sampleRate", settings.sample_rate.into()),
        ("transparentAlpha", settings.transparent_alpha.into()),
        ("audioSamples", js_audio),
    ] {
        Reflect::set(&options, &key.into(), &value).map_err(js_error)?;
    }
    call_async(CODECS, "startExport", &[options.into()]).await
}
