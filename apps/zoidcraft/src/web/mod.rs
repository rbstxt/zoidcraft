mod export;
pub(crate) use export::{BrowserExport, frame_export_settings, launch_export};

use std::future::Future;

use js_sys::{Array, Float32Array, Function, Promise, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};
use wasm_bindgen_futures::{JsFuture, spawn_local};

use super::*;

pub(crate) const CODECS: &str = "__zcWebCodecs";
const FILES: &str = "__zcFiles";
const MEDIA_ACCEPT: &str = ".png,.jpg,.jpeg,.webp,.gif,.wav,.mp3,.ogg,.flac,.aac,.mp4,.mov,.webm,.mkv,.m4v,.avi,.obj,.gltf,.glb,.ttf,.otf,.cube";

thread_local! {
    static RUNNER: eframe::WebRunner = eframe::WebRunner::new();
}

#[wasm_bindgen]
pub async fn start_zoidcraft_web(canvas_id: &str, webgl: bool) -> Result<(), JsValue> {
    let canvas = web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id(canvas_id))
        .and_then(|element| element.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        .ok_or_else(|| JsValue::from_str("ZoidCraft canvas is missing"))?;
    let mut options = eframe::WebOptions::default();
    if let egui_wgpu::WgpuSetup::CreateNew(setup) = &mut options.wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends = if webgl {
            wgpu::Backends::GL
        } else {
            wgpu::Backends::BROWSER_WEBGPU
        };
    }
    let runner = RUNNER.with(Clone::clone);
    runner
        .start(
            canvas,
            options,
            Box::new(|cc| {
                let app = ZoidCraftApp::new(cc);
                if app.renderer.is_none() {
                    return Err("Could not initialize ZoidCraft's project renderer".into());
                }
                Ok(Box::new(app))
            }),
        )
        .await
}

pub(crate) fn call(bridge: &str, method: &str, args: &[JsValue]) -> Result<JsValue, String> {
    let invoke = || {
        let host = Reflect::get(&js_sys::global(), &bridge.into())?;
        let function = Reflect::get(&host, &method.into())?.dyn_into::<Function>()?;
        function.apply(&host, &args.iter().collect::<Array>())
    };
    invoke().map_err(js_error)
}

fn js_error(error: JsValue) -> String {
    error
        .as_string()
        .or_else(|| Reflect::get(&error, &"message".into()).ok()?.as_string())
        .unwrap_or_else(|| format!("{error:?}"))
}

async fn call_async(bridge: &str, method: &str, args: &[JsValue]) -> Result<JsValue, String> {
    JsFuture::from(Promise::resolve(&call(bridge, method, args)?))
        .await
        .map_err(js_error)
}

pub(crate) fn configure_export_status(status: &mut ExportStatus) {
    status.ffmpeg_available = call(CODECS, "isSupported", &[])
        .ok()
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    status.ffmpeg_version = if status.ffmpeg_available {
        "WebCodecs API (Browser Hardware H.264/VP9/AV1)".into()
    } else {
        "WebCodecs is unavailable".into()
    };
}

fn download(filename: &str, bytes: &[u8], mime: &str) -> Result<(), String> {
    call(
        FILES,
        "downloadBlob",
        &[filename.into(), Uint8Array::from(bytes).into(), mime.into()],
    )?;
    Ok(())
}

pub(crate) fn download_project(
    project: &Project,
    path: &Path,
    serialize: impl FnOnce(&Project) -> Result<Vec<u8>, zc_project::ProjectError>,
) -> Result<(), String> {
    let bytes = serialize(project).map_err(|err| err.to_string())?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("project.zcraft");
    download(filename, &bytes, "application/octet-stream")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct File {
    name: String,
    bytes: Arc<Vec<u8>>,
}

pub(crate) enum FileAction {
    OpenProject,
    ImportMedia,
    Drop,
    Assign { target_id: String, property: String },
    Relink { asset_id: String },
    ImportSvg { clip_id: Uuid },
}

type Completion = Box<dyn FnOnce(&mut ZoidCraftApp)>;
type TaskResult = Result<Completion, String>;

#[derive(Default)]
pub(crate) struct AppState {
    tasks: Vec<Receiver<TaskResult>>,
    picker_pending: bool,
    project_videos_pending: usize,
    generation: u64,
}

impl AppState {
    pub(crate) fn clear_pending(&mut self) {
        self.tasks.clear();
        self.picker_pending = false;
        self.project_videos_pending = 0;
        self.generation = self.generation.wrapping_add(1);
    }

    pub(super) fn enqueue(&mut self, work: impl Future<Output = TaskResult> + 'static) {
        let (tx, rx) = mpsc::channel();
        self.tasks.push(rx);
        spawn_local(async move {
            let _ = tx.send(work.await);
        });
    }
}

pub(super) fn finish_with(f: impl FnOnce(&mut ZoidCraftApp) + 'static) -> TaskResult {
    Ok(Box::new(f))
}

async fn picked_file(value: JsValue) -> Result<Option<File>, String> {
    let value = JsFuture::from(Promise::resolve(&value))
        .await
        .map_err(js_error)?;
    if value.is_null() || value.is_undefined() {
        return Ok(None);
    }
    let name = Reflect::get(&value, &"name".into())
        .map_err(js_error)?
        .as_string()
        .ok_or("Browser did not provide a filename")?;
    let bytes = Reflect::get(&value, &"bytes".into())
        .map_err(js_error)?
        .dyn_into::<Uint8Array>()
        .map_err(js_error)?;
    Ok(Some(File {
        name,
        bytes: Arc::new(bytes.to_vec()),
    }))
}

fn embedded_asset(filename: &str, bytes: Vec<u8>, kind: AssetKind, digest: String) -> AssetRecord {
    AssetRecord {
        id: digest.clone(),
        kind,
        source: AssetSource::File,
        filename: filename.to_owned(),
        url: None,
        external: false,
        size: bytes.len() as u64,
        bytes: Some(bytes),
        missing: false,
        sha256: digest,
        preserved: BTreeMap::new(),
    }
}

type DecodedAudio = Option<(Arc<AudioBuffer>, AudioWaveformPeaks)>;

async fn register_video(
    hash: u64,
    filename: &str,
    bytes: Uint8Array,
) -> Result<(zc_render::video::VideoMetadata, DecodedAudio), String> {
    let result = call_async(
        CODECS,
        "importVideo",
        &[format!("{hash:016x}").into(), filename.into(), bytes.into()],
    )
    .await?;
    let field = |name: &str| Reflect::get(&result, &name.into()).map_err(js_error);
    let number = |name: &str| {
        field(name)?
            .as_f64()
            .ok_or_else(|| format!("Video is missing {name}"))
    };
    let meta = zc_render::video::VideoMetadata {
        width: number("width")? as u32,
        height: number("height")? as u32,
        fps: number("fps")?,
        duration_secs: number("durationSecs")?,
        has_audio: field("hasAudio")?.as_bool().unwrap_or(false),
    };
    if meta.width == 0 || meta.height == 0 || !meta.fps.is_finite() || meta.fps <= 0.0 {
        return Err("Invalid video dimensions or frame rate".into());
    }
    zc_render::video::register_web_video_metadata(hash, meta.clone());
    let audio = if let Ok(samples) = field("audioSamples")?.dyn_into::<Float32Array>() {
        let buf = AudioBuffer {
            sample_rate: number("sampleRate")? as u32,
            channels: 2,
            samples_interleaved_f32: samples.to_vec(),
        };
        let peaks = WaveformPeaks::from_audio_buffer(&buf);
        Some((
            Arc::new(buf),
            AudioWaveformPeaks {
                points_per_sec: f64::from(peaks.points_per_sec()),
                min_buf: peaks.min_peaks().to_vec(),
                max_buf: peaks.max_peaks().to_vec(),
            },
        ))
    } else {
        None
    };
    Ok((meta, audio))
}

impl ZoidCraftApp {
    pub(crate) fn perform_open_project(&mut self) {
        self.pick_web_file(".zcraft,.pz", FileAction::OpenProject);
    }

    pub(crate) fn handle_import_media(&mut self) {
        self.pick_web_file(MEDIA_ACCEPT, FileAction::ImportMedia);
    }

    pub(crate) fn handle_pick_asset_for_property(
        &mut self,
        target_id: String,
        property: String,
        is_lut: bool,
    ) {
        let accept = if is_lut {
            ".cube,.png,.jpg,.jpeg,.webp,.gif"
        } else {
            MEDIA_ACCEPT
        };
        self.pick_web_file(
            accept,
            FileAction::Assign {
                target_id,
                property,
            },
        );
    }

    pub(crate) fn pick_web_file(&mut self, accept: &str, action: FileAction) {
        if self.web.picker_pending {
            return;
        }
        self.pause_playback_for_modal_dialog();
        // Invoke click within the UI event, then await the selected file off the UI tick.
        match call(FILES, "pickFile", &[accept.into()]) {
            Ok(picker) => {
                self.web.picker_pending = true;
                self.web.enqueue(async move {
                    let result = picked_file(picker).await;
                    finish_with(move |app| {
                        app.web.picker_pending = false;
                        match result {
                            Ok(Some(file)) => app.apply_web_file(action, file),
                            Ok(None) => {}
                            Err(err) => app.show_toast(err),
                        }
                    })
                });
            }
            Err(err) => self.show_toast(err),
        }
    }

    pub(crate) fn handle_web_drops(&mut self, ctx: &egui::Context) {
        let mut files = ctx.input_mut(|input| std::mem::take(&mut input.raw.dropped_files));
        if files.is_empty() {
            return;
        }
        if let Some(index) = files
            .iter()
            .position(|file| classify_dropped_path(file.path()) == DroppedFileKind::Project)
        {
            files = vec![files.swap_remove(index)];
        }
        self.web.enqueue(async move {
            let mut results = Vec::with_capacity(files.len());
            for file in files {
                let name = file.path().to_string_lossy().into_owned();
                results.push(file.bytes_async().await.map(|bytes| File {
                    name,
                    bytes: Arc::new(bytes),
                }));
            }
            finish_with(move |app| {
                for result in results {
                    match result {
                        Ok(file) => app.apply_web_file(FileAction::Drop, file),
                        Err(err) => app.show_toast(err),
                    }
                }
            })
        });
    }

    fn apply_web_file(&mut self, action: FileAction, file: File) {
        let path = Path::new(&file.name);
        if matches!(action, FileAction::OpenProject) {
            self.open_web_project(file);
            return;
        }
        if matches!(action, FileAction::Drop)
            && classify_dropped_path(path) == DroppedFileKind::Project
        {
            if self.needs_confirmation_before_replace_or_close() {
                self.pending_unsaved_dialog =
                    Some(PendingUnsavedAction::OpenDroppedWebProject(file));
            } else {
                self.open_web_project(file);
            }
            return;
        }
        if matches!(action, FileAction::Drop)
            && classify_dropped_path(path) == DroppedFileKind::Unsupported
        {
            self.show_toast(format!("Unsupported file: {}", file.name));
            return;
        }
        let (filename, ext, kind) = asset_filename_and_kind(path);
        let bytes = Arc::unwrap_or_clone(file.bytes);
        if let FileAction::ImportSvg { clip_id } = action {
            let result = std::str::from_utf8(&bytes)
                .map_err(|err| err.to_string())
                .and_then(zc_render::shape_plus::import_svg_to_shape_plus)
                .and_then(|payload| {
                    self.session
                        .execute(Command::UpdateShapePlusPayload { clip_id, payload })
                        .map_err(|err| err.to_string())
                });
            if let Err(err) = result {
                self.show_toast(format!("Could not import SVG: {err}"));
            }
            return;
        }
        let digest = sha256_hex(&bytes);
        let replaced_id = match &action {
            FileAction::Relink { asset_id } => asset_id,
            _ => &digest,
        };
        if !self.web_asset_fits(replaced_id, bytes.len() as u64) {
            return;
        }
        if let FileAction::Relink { asset_id } = action {
            if let Err(err) = self.session.execute(Command::RelinkAsset {
                asset_id,
                filename,
                bytes,
            }) {
                self.show_toast(err.to_string());
                return;
            }
            self.rebuild_audio_caches();
            self.register_web_project_videos();
            self.force_single_render = true;
            return;
        }

        if is_video_extension(&ext) {
            let record = embedded_asset(&filename, bytes, kind, digest);
            let hash = zc_render::web_video_asset_hash(&record);
            let start = self.session.playback.current_frame;
            let typed_bytes = Uint8Array::from(record.bytes.as_deref().unwrap_or_default());
            self.show_toast(format!("Importing {filename}…"));
            self.web.enqueue(async move {
                let (meta, decoded_audio) = register_video(hash, &filename, typed_bytes)
                    .await
                    .map_err(|err| format!("Could not import {filename}: {err}"))?;
                finish_with(move |app| {
                    if !app.web_asset_fits(&record.id, record.size) {
                        return;
                    }
                    match action {
                        FileAction::Assign {
                            target_id,
                            property,
                        } => {
                            let id = record.id.clone();
                            app.assign_media_to_property(target_id, property, record, &ext);
                            app.cache_web_video_audio(id, decoded_audio);
                        }
                        _ => {
                            let payload = VideoImportPayload {
                                filename,
                                asset_record: record,
                                probed: ProbedVideoInfo {
                                    width: Some(meta.width),
                                    height: Some(meta.height),
                                    fps: Some(meta.fps),
                                    duration_secs: Some(meta.duration_secs),
                                },
                                decoded_audio,
                                desired_start: start,
                            };
                            match app.finalize_video_import_payload(payload) {
                                Ok(message) | Err(message) => app.show_toast(message),
                            }
                        }
                    }
                    app.force_single_render = true;
                })
            });
        } else if let FileAction::Assign {
            target_id,
            property,
        } = action
        {
            let record = embedded_asset(&filename, bytes, kind, digest);
            self.assign_media_to_property(target_id, property, record, &ext);
        } else {
            match self.import_media_bytes(filename, ext, kind, bytes, None) {
                Ok(message) | Err(message) => self.show_toast(message),
            }
        }
    }

    fn web_asset_fits(&mut self, id: &str, size: u64) -> bool {
        let previous_size = self
            .session
            .project
            .assets
            .get(id)
            .and_then(|asset| asset.bytes.as_ref())
            .map_or(0, |bytes| bytes.len() as u64);
        let remaining = zc_project::MAX_RETAINED_ASSET_BYTES.saturating_sub(
            self.existing_embedded_asset_bytes()
                .saturating_sub(previous_size),
        );
        if size > remaining {
            self.show_toast("The file exceeds the remaining embedded asset budget");
            return false;
        }
        true
    }

    pub(crate) fn open_web_project(&mut self, file: File) {
        let path = Path::new(&file.name);
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pz"))
        {
            self.finish_open_pz(path, zc_project::import_pz_from_bytes(&file.bytes));
        } else {
            self.finish_open_zcraft(path, zc_project::load_zcraft_from_bytes(&file.bytes));
        }
    }

    fn cache_web_video_audio(&mut self, id: String, audio: DecodedAudio) {
        if let Some((buffer, peaks)) = audio {
            self.decoded_audio.insert(id.clone(), buffer);
            self.asset_waveform_peaks.insert(id, peaks);
            self.cached_audio_mix_hash = None;
            self.last_audio_peaks_sync_rev = None;
        }
    }

    pub(crate) fn register_web_project_videos(&mut self) {
        for (id, record) in &self.session.project.assets {
            let (_, ext, _) = asset_filename_and_kind(Path::new(&record.filename));
            let Some(bytes) = record.bytes.as_deref().filter(|_| is_video_extension(&ext)) else {
                continue;
            };
            let hash = zc_render::web_video_asset_hash(record);
            let bytes = Uint8Array::from(bytes);
            let filename = record.filename.clone();
            let id = id.clone();
            self.web.project_videos_pending += 1;
            self.web.enqueue(async move {
                let result = register_video(hash, &filename, bytes).await;
                finish_with(move |app| {
                    app.web.project_videos_pending -= 1;
                    match result {
                        Ok((_, audio)) => {
                            app.cache_web_video_audio(id, audio);
                            app.force_single_render = true;
                        }
                        Err(err) => {
                            app.show_toast(format!("Could not decode {filename}: {err}"));
                        }
                    }
                })
            });
        }
    }

    pub(crate) fn poll_web_tasks(&mut self, ctx: &egui::Context) {
        let mut ready = Vec::new();
        self.web.tasks.retain(|task| match task.try_recv() {
            Ok(result) => {
                ready.push(result);
                false
            }
            Err(TryRecvError::Empty) => true,
            Err(TryRecvError::Disconnected) => false,
        });
        let generation = self.web.generation;
        for result in ready {
            if self.web.generation != generation {
                break;
            }
            match result {
                Ok(complete) => complete(self),
                Err(err) => self.show_toast(err),
            }
        }
        let dirty = call(CODECS, "consumeDirty", &[])
            .ok()
            .and_then(|value| value.as_bool());
        if dirty == Some(true) {
            self.force_single_render = true;
            if let Some(renderer) = self.renderer.as_mut() {
                renderer.last_render_key = None;
                renderer.preview_cache.clear();
            }
        }
        if !self.web.tasks.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }
}

pub(crate) fn poll_video_repaint(ctx: &egui::Context) {
    if call(CODECS, "isDecoding", &[])
        .ok()
        .and_then(|value| value.as_bool())
        == Some(true)
    {
        ctx.request_repaint_after(Duration::from_millis(16));
    }
}
