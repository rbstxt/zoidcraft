#![forbid(unsafe_code)]

//! ZoidCraft headless CLI, batch renderer, `.pz` converter, effect self-test runner,
//! and JSON-Lines session control loop (L5 headless — zero GUI dependencies).

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use clap::{Parser, Subcommand};
use image::RgbaImage;
use zc_anim::{DynamicProperty, InterpolationMode, Keyframe, StaticBehavior};
use zc_audio::{
    AudioBuffer, AudioClipScheduleItem, ClipAudioSampleParams, STANDARD_SAMPLE_RATE,
    decode_audio_bytes, mix_offline_stereo_animated,
};
use zc_export::{
    AudioCodec, ExportContainer, ExportError, ExportSettings, QualityMode, VideoCodec,
    export_video_with_progress,
};
use zc_plugin::PluginRegistry;
use zc_project::{
    Project, PropertyTrack, PropertyValue, StaticBehavior as ProjectStaticBehavior, import_pz,
    load_zcraft, save_zcraft,
};
use zc_render::ProjectRenderEngine;
use zc_session::{Session, dispatch_json_lines};

#[derive(Debug, Parser)]
#[command(
    name = "zoidcraft-cli",
    version,
    about = "ZoidCraft headless motion-graphics renderer, validator, importer, and control CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Print detailed metadata for a `.zcraft` project archive.
    Info {
        /// Path to the `.zcraft` project file.
        project: PathBuf,
    },

    /// Load and validate archive structure, JSON bounds, and SHA-256 asset integrity.
    Validate {
        /// Path to the `.zcraft` project file.
        project: PathBuf,
    },

    /// Import a ClipMaker 3 `.pz` archive and atomically save it as `.zcraft`.
    ImportPz {
        /// Path to the input `.pz` file.
        input: PathBuf,
        /// Destination `.zcraft` output path.
        #[arg(long, short = 'o')]
        out: PathBuf,
    },

    /// Render a single frame headlessly on the GPU and save it as a PNG image.
    Render {
        /// Optional `.zcraft` (or `.pz`) project file; defaults to the CM3 starter project.
        #[arg(long, short = 'p')]
        project: Option<PathBuf>,
        /// Fractional frame number to render.
        #[arg(long, short = 'f', default_value_t = 0.0)]
        frame: f64,
        /// Output PNG file path.
        #[arg(long, short = 'o')]
        out: PathBuf,
        /// Optional output width override in pixels.
        #[arg(long)]
        width: Option<u32>,
        /// Optional output height override in pixels.
        #[arg(long)]
        height: Option<u32>,
    },

    /// Render a frame range headlessly on the GPU, mix project audio tracks, and export
    /// video (MP4/WebM/MKV via FFmpeg) or a PNG image sequence.
    Export {
        /// Optional `.zcraft` (or `.pz`) project file; defaults to the CM3 starter project.
        #[arg(long, short = 'p')]
        project: Option<PathBuf>,
        /// Output video file path (or directory for `--format png`).
        #[arg(long, short = 'o')]
        out: PathBuf,
        /// Container format (`mp4`, `webm`, `mkv`, `png`); inferred from `--out` extension if omitted.
        #[arg(long)]
        format: Option<String>,
        /// Video codec (`h264`, `h265`, `vp9`, `vp8`, `av1`).
        #[arg(long)]
        video_codec: Option<String>,
        /// Audio codec (`aac`, `opus`, `pcm`, `none`).
        #[arg(long)]
        audio_codec: Option<String>,
        /// Constant Rate Factor quality (default 18). Ignored if `--bitrate-kbps` is set.
        #[arg(long, default_value_t = 18)]
        crf: u8,
        /// Target video bitrate in kbps (overrides `--crf`).
        #[arg(long)]
        bitrate_kbps: Option<u32>,
        /// Audio bitrate in kbps for lossy audio codecs (default 192).
        #[arg(long, default_value_t = 192)]
        audio_bitrate_kbps: u32,
        /// Optional output width override in pixels.
        #[arg(long)]
        width: Option<u32>,
        /// Optional output height override in pixels.
        #[arg(long)]
        height: Option<u32>,
        /// Optional output frame rate override in fps.
        #[arg(long)]
        fps: Option<f64>,
        /// Inclusive starting frame index (default 0).
        #[arg(long, default_value_t = 0)]
        start_frame: u32,
        /// Exclusive ending frame index (defaults to sequence length).
        #[arg(long)]
        end_frame: Option<u32>,
        /// Preserve transparent alpha channel (`yuva420p` for VP9, RGBA8 for PNG sequence).
        #[arg(long, default_value_t = false)]
        transparent: bool,
        /// Optional path to the `ffmpeg` binary (defaults to `ffmpeg` on PATH).
        #[arg(long)]
        ffmpeg: Option<PathBuf>,
    },

    /// Run a headless GPU self-test across all 44 standard effects, 12 plugin effects,
    /// and the default 3D scene.
    SelftestEffects {
        /// Test frame width in pixels.
        #[arg(long, default_value_t = 128)]
        width: u32,
        /// Test frame height in pixels.
        #[arg(long, default_value_t = 128)]
        height: u32,
    },

    /// Run an interactive JSON-Lines stdin/stdout control loop backed by `zc_session::Session`.
    Control {
        /// Optional `.zcraft` (or `.pz`) project file to load into the session.
        #[arg(long, short = 'p')]
        project: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Commands::Info { project } => cmd_info(&project),
        Commands::Validate { project } => cmd_validate(&project),
        Commands::ImportPz { input, out } => cmd_import_pz(&input, &out),
        Commands::Render {
            project,
            frame,
            out,
            width,
            height,
        } => cmd_render(project.as_deref(), frame, &out, width, height),
        Commands::Export {
            project,
            out,
            format,
            video_codec,
            audio_codec,
            crf,
            bitrate_kbps,
            audio_bitrate_kbps,
            width,
            height,
            fps,
            start_frame,
            end_frame,
            transparent,
            ffmpeg,
        } => cmd_export(ExportCliArgs {
            project: project.as_deref(),
            out: &out,
            format: format.as_deref(),
            video_codec: video_codec.as_deref(),
            audio_codec: audio_codec.as_deref(),
            crf,
            bitrate_kbps,
            audio_bitrate_kbps,
            width,
            height,
            fps,
            start_frame,
            end_frame,
            transparent,
            ffmpeg,
        }),
        Commands::SelftestEffects { width, height } => cmd_selftest_effects(width, height),
        Commands::Control { project } => cmd_control(project.as_deref()),
    }
}

fn load_project_any(path: &Path) -> Result<(Project, Vec<String>), String> {
    let is_pz = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("pz"));
    if is_pz {
        let imported = import_pz(path).map_err(|e| format!("failed to import .pz: {e}"))?;
        Ok((imported.project, imported.warnings))
    } else {
        let project = load_zcraft(path).map_err(|e| format!("failed to load .zcraft: {e}"))?;
        Ok((project, Vec::new()))
    }
}

fn build_plugins_for_project(project: &Project) -> PluginRegistry {
    let mut plugins = PluginRegistry::new();
    for req in &project.plugin_requirements {
        let _ = plugins.enable(req);
    }
    plugins
}

fn cmd_info(path: &Path) -> Result<(), String> {
    let project =
        load_zcraft(path).map_err(|e| format!("failed to load {}: {e}", path.display()))?;
    let seq = &project.sequence;
    let fps = f64::from(seq.fps.max(1));
    let duration_secs = seq.length_frames / fps;
    let video_clips: usize = seq.video_tracks.iter().map(|t| t.clips.len()).sum();
    let audio_clips: usize = seq.audio_tracks.iter().map(|t| t.clips.len()).sum();
    let missing_assets: usize = project.assets.values().filter(|a| a.missing).count();

    println!("Project: {}", project.name);
    println!("Schema Version: {}", project.version);
    println!("Resolution: {}x{}", seq.resolution[0], seq.resolution[1]);
    println!("FPS: {}", seq.fps);
    println!(
        "Length: {:.2} frames ({:.3} s)",
        seq.length_frames, duration_secs
    );
    println!(
        "Video Tracks: {} ({} clips)",
        seq.video_tracks.len(),
        video_clips
    );
    println!(
        "Audio Tracks: {} ({} clips)",
        seq.audio_tracks.len(),
        audio_clips
    );
    println!("Compositions: {}", project.compositions.len());
    println!(
        "Assets: {} ({} missing)",
        project.assets.len(),
        missing_assets
    );
    if project.plugin_requirements.is_empty() {
        println!("Required Plugins: none (Vanilla CM3 compatible)");
    } else {
        let list: Vec<&str> = project
            .plugin_requirements
            .iter()
            .map(String::as_str)
            .collect();
        println!("Required Plugins: {}", list.join(", "));
    }
    if let Some(pz_ver) = project.imported_from_pz_version {
        println!("Imported From CM3 .pz Version: {pz_ver}");
    }
    Ok(())
}

fn cmd_validate(path: &Path) -> Result<(), String> {
    let project =
        load_zcraft(path).map_err(|e| format!("validation failed for {}: {e}", path.display()))?;
    let seq = &project.sequence;
    if seq.resolution[0] == 0 || seq.resolution[1] == 0 {
        return Err(format!(
            "invalid sequence resolution: {}x{}",
            seq.resolution[0], seq.resolution[1]
        ));
    }
    if seq.fps == 0 {
        return Err("invalid sequence fps: 0".to_owned());
    }

    let missing_assets: Vec<&str> = project
        .assets
        .iter()
        .filter(|(_, rec)| rec.missing || (!rec.external && rec.bytes.is_none()))
        .map(|(id, _)| id.as_str())
        .collect();

    let registry = PluginRegistry::new();
    let unknown_plugins: Vec<&str> = project
        .plugin_requirements
        .iter()
        .filter(|id| registry.get(id).is_none())
        .map(String::as_str)
        .collect();

    println!(
        "OK: {} ('{}', {}x{} @ {}fps, {} video tracks, {} audio tracks, {} assets)",
        path.display(),
        project.name,
        seq.resolution[0],
        seq.resolution[1],
        seq.fps,
        seq.video_tracks.len(),
        seq.audio_tracks.len(),
        project.assets.len(),
    );
    if !missing_assets.is_empty() {
        eprintln!(
            "warning: {} missing/unembedded asset(s): {}",
            missing_assets.len(),
            missing_assets.join(", ")
        );
    }
    if !unknown_plugins.is_empty() {
        eprintln!(
            "warning: {} unregistered plugin requirement(s): {}",
            unknown_plugins.len(),
            unknown_plugins.join(", ")
        );
    }
    Ok(())
}

fn cmd_import_pz(input: &Path, out: &Path) -> Result<(), String> {
    let result =
        import_pz(input).map_err(|e| format!("failed to import {}: {e}", input.display()))?;
    for warning in &result.warnings {
        eprintln!("warning: {warning}");
    }
    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create output directory {}: {e}",
                parent.display()
            )
        })?;
    }
    save_zcraft(&result.project, out)
        .map_err(|e| format!("failed to save {}: {e}", out.display()))?;
    println!(
        "Imported '{}' -> {} ({} video tracks, {} assets, {} warnings)",
        result.project.name,
        out.display(),
        result.project.sequence.video_tracks.len(),
        result.project.assets.len(),
        result.warnings.len(),
    );
    Ok(())
}

fn cmd_render(
    project_path: Option<&Path>,
    frame: f64,
    out: &Path,
    width_override: Option<u32>,
    height_override: Option<u32>,
) -> Result<(), String> {
    let (project, warnings) = match project_path {
        Some(path) => load_project_any(path)?,
        None => (Project::new_cm3_default(), Vec::new()),
    };
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }

    let width = width_override
        .unwrap_or(project.sequence.resolution[0])
        .max(1);
    let height = height_override
        .unwrap_or(project.sequence.resolution[1])
        .max(1);
    let plugins = build_plugins_for_project(&project);

    let mut engine = ProjectRenderEngine::new_headless()
        .map_err(|e| format!("failed to initialize headless GPU renderer: {e}"))?;
    let premul_rgba = engine
        .render_frame_rgba8(&project, &plugins, frame, width, height)
        .map_err(|e| format!("render failed at frame {frame}: {e}"))?;
    let straight_rgba = zc_render::unpremultiply_rgba8(&premul_rgba);

    let img = RgbaImage::from_raw(width, height, straight_rgba)
        .ok_or_else(|| format!("rendered pixel buffer size mismatch for {width}x{height}"))?;

    if let Some(parent) = out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create output directory {}: {e}",
                parent.display()
            )
        })?;
    }
    img.save(out)
        .map_err(|e| format!("failed to save PNG {}: {e}", out.display()))?;

    let non_black = img
        .pixels()
        .filter(|px| px[0] > 2 || px[1] > 2 || px[2] > 2)
        .count();
    println!(
        "Rendered frame {frame} ({width}x{height}) -> {} ({non_black} non-black pixels)",
        out.display()
    );
    Ok(())
}

struct ExportCliArgs<'a> {
    project: Option<&'a Path>,
    out: &'a Path,
    format: Option<&'a str>,
    video_codec: Option<&'a str>,
    audio_codec: Option<&'a str>,
    crf: u8,
    bitrate_kbps: Option<u32>,
    audio_bitrate_kbps: u32,
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<f64>,
    start_frame: u32,
    end_frame: Option<u32>,
    transparent: bool,
    ffmpeg: Option<PathBuf>,
}

fn parse_container(format: Option<&str>, out: &Path) -> Result<ExportContainer, String> {
    let token = match format {
        Some(f) => f.to_ascii_lowercase(),
        None => out
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| "mp4".to_owned()),
    };
    match token.as_str() {
        "mp4" => Ok(ExportContainer::Mp4),
        "webm" => Ok(ExportContainer::Webm),
        "mkv" => Ok(ExportContainer::Mkv),
        "png" | "png-sequence" => Ok(ExportContainer::PngSequence),
        other => Err(format!(
            "unsupported export format `{other}` (expected mp4, webm, mkv, or png)"
        )),
    }
}

fn parse_video_codec(
    codec: Option<&str>,
    container: ExportContainer,
) -> Result<VideoCodec, String> {
    let Some(raw) = codec else {
        return Ok(match container {
            ExportContainer::Webm => VideoCodec::Vp9,
            ExportContainer::Mp4 | ExportContainer::Mkv | ExportContainer::PngSequence => {
                VideoCodec::H264
            }
        });
    };
    match raw.to_ascii_lowercase().as_str() {
        "h264" | "avc" => Ok(VideoCodec::H264),
        "h265" | "hevc" => Ok(VideoCodec::H265),
        "vp9" => Ok(VideoCodec::Vp9),
        "vp8" => Ok(VideoCodec::Vp8),
        "av1" => Ok(VideoCodec::Av1),
        other => Err(format!(
            "unsupported video codec `{other}` (expected h264, h265, vp9, vp8, or av1)"
        )),
    }
}

fn parse_audio_codec(
    codec: Option<&str>,
    container: ExportContainer,
    has_audio: bool,
) -> Result<AudioCodec, String> {
    let Some(raw) = codec else {
        if !has_audio || matches!(container, ExportContainer::PngSequence) {
            return Ok(AudioCodec::None);
        }
        return Ok(match container {
            ExportContainer::Webm => AudioCodec::Opus,
            ExportContainer::Mp4 | ExportContainer::Mkv => AudioCodec::Aac,
            ExportContainer::PngSequence => AudioCodec::None,
        });
    };
    match raw.to_ascii_lowercase().as_str() {
        "aac" => Ok(AudioCodec::Aac),
        "opus" => Ok(AudioCodec::Opus),
        "pcm" => Ok(AudioCodec::Pcm),
        "none" => Ok(AudioCodec::None),
        other => Err(format!(
            "unsupported audio codec `{other}` (expected aac, opus, pcm, or none)"
        )),
    }
}

fn eval_scalar_track(track: &PropertyTrack, frame: f64, default: f64) -> f64 {
    let base = match &track.value {
        PropertyValue::Number(v) => *v,
        _ => default,
    };
    let static_behavior = match track.static_behavior {
        ProjectStaticBehavior::Linear => StaticBehavior::Linear,
        ProjectStaticBehavior::Constant => StaticBehavior::Constant,
    };
    if (!track.animated && static_behavior == StaticBehavior::Constant)
        || track.keyframes.is_empty()
    {
        return base;
    }
    let mut prop = DynamicProperty::new_static(base);
    prop.static_behavior = static_behavior;
    prop.animated = track.animated;
    for kf in &track.keyframes {
        if let PropertyValue::Number(v) = kf.value {
            let tween_code = if kf.tween != 0 {
                kf.tween as u16
            } else if kf.is_bezier {
                (1u16 << 8) | u16::from(kf.easing_index.max(1))
            } else {
                u16::from(kf.easing_index)
            };
            let (mode, easing) = InterpolationMode::from_cm3_tween_code(tween_code);
            let mut key = Keyframe::with_easing(kf.frame, v, easing);
            key.interpolated = mode != InterpolationMode::Discrete;
            if kf.is_bezier {
                key.in_tangent = Some(kf.control_points[0]);
                key.out_tangent = Some(kf.control_points[1]);
            }
            key.continuous_tangent = kf.continuous_tangent;
            prop.set_keyframe(key);
        }
    }
    prop.evaluate(frame)
}

fn mix_project_audio(
    project: &Project,
    start_frame: u32,
    end_frame: u32,
    fps: f64,
    sample_rate: u32,
) -> Result<Option<Vec<[f32; 2]>>, String> {
    let seq = &project.sequence;
    let any_solo = seq
        .audio_tracks
        .iter()
        .any(|t| t.enabled && !t.muted && t.solo);
    let mut buffers: HashMap<String, AudioBuffer> = HashMap::new();
    let mut scheduled: Vec<AudioClipScheduleItem> = Vec::new();
    let mut source_clips: Vec<&zc_project::AudioClip> = Vec::new();

    for track in &seq.audio_tracks {
        if !track.enabled || track.muted || (any_solo && !track.solo) {
            continue;
        }
        for clip in &track.clips {
            if clip.length <= 0.0 {
                continue;
            }
            let Some(asset_id) = clip.media.as_deref() else {
                continue;
            };
            if !buffers.contains_key(asset_id) {
                let Some(record) = project.assets.get(asset_id) else {
                    continue;
                };
                let Some(bytes) = record.bytes.as_deref() else {
                    continue;
                };
                let ext_hint = Path::new(&record.filename)
                    .extension()
                    .and_then(|e| e.to_str());
                let decoded = decode_audio_bytes(bytes, ext_hint)
                    .map_err(|e| format!("failed to decode audio asset `{asset_id}`: {e}"))?;
                buffers.insert(asset_id.to_owned(), decoded);
            }

            let t0 = eval_scalar_track(&clip.time_curve, 0.0, 0.0);
            let t1 = eval_scalar_track(&clip.time_curve, clip.length, clip.length / fps);
            let rate = zc_audio::effective_playback_rate(t0, t1, clip.length, fps);
            scheduled.push(AudioClipScheduleItem {
                clip_start_frame: clip.start - f64::from(start_frame),
                clip_length_frames: clip.length,
                media_start_sec: t0,
                playback_rate: rate,
                volume: eval_scalar_track(&clip.audio.volume, 0.0, 1.0) as f32,
                pan: eval_scalar_track(&clip.audio.pan, 0.0, 0.0) as f32,
                buffer_id: asset_id.to_owned(),
            });
            source_clips.push(clip);
        }
    }

    if scheduled.is_empty() {
        return Ok(None);
    }

    let total_frames = f64::from(end_frame.saturating_sub(start_frame));
    let mixed = mix_offline_stereo_animated(
        total_frames,
        fps,
        sample_rate,
        &scheduled,
        &buffers,
        |idx, _item, local_frame| {
            let clip = source_clips[idx];
            ClipAudioSampleParams {
                media_time_sec: eval_scalar_track(&clip.time_curve, local_frame, local_frame / fps),
                volume: eval_scalar_track(&clip.audio.volume, local_frame, 1.0) as f32,
                pan: eval_scalar_track(&clip.audio.pan, local_frame, 0.0) as f32,
            }
        },
    );
    Ok(Some(mixed))
}

fn cmd_export(args: ExportCliArgs<'_>) -> Result<(), String> {
    let (project, warnings) = match args.project {
        Some(path) => load_project_any(path)?,
        None => (Project::new_cm3_default(), Vec::new()),
    };
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }

    let width = args.width.unwrap_or(project.sequence.resolution[0]).max(1);
    let height = args.height.unwrap_or(project.sequence.resolution[1]).max(1);
    let fps = args.fps.unwrap_or(f64::from(project.sequence.fps.max(1)));
    let default_end = project.sequence.length_frames.round().max(1.0) as u32;
    let end_frame = args.end_frame.unwrap_or(default_end);
    if end_frame <= args.start_frame {
        return Err(format!(
            "end_frame ({end_frame}) must be greater than start_frame ({})",
            args.start_frame
        ));
    }

    let container = parse_container(args.format, args.out)?;
    let video_codec = parse_video_codec(args.video_codec, container)?;
    let sample_rate = STANDARD_SAMPLE_RATE;
    let mixed_audio = mix_project_audio(&project, args.start_frame, end_frame, fps, sample_rate)?;
    let audio_codec = parse_audio_codec(args.audio_codec, container, mixed_audio.is_some())?;
    let quality = match args.bitrate_kbps {
        Some(kbps) => QualityMode::BitrateKbps(kbps),
        None => QualityMode::Crf(args.crf),
    };

    let settings = ExportSettings {
        width,
        height,
        fps,
        start_frame: args.start_frame,
        end_frame,
        container,
        video_codec,
        audio_codec,
        quality,
        audio_bitrate_kbps: args.audio_bitrate_kbps,
        sample_rate,
        transparent_alpha: args.transparent,
        output_path: args.out.to_path_buf(),
        ffmpeg_binary: args.ffmpeg,
    };
    settings
        .validate()
        .map_err(|e| format!("invalid export settings: {e}"))?;

    let plugins = build_plugins_for_project(&project);
    let mut engine = ProjectRenderEngine::new_headless()
        .map_err(|e| format!("failed to initialize headless GPU renderer: {e}"))?;
    let cancel = Arc::new(AtomicBool::new(false));
    let audio_slice = if matches!(audio_codec, AudioCodec::None) {
        None
    } else {
        mixed_audio.as_deref()
    };

    let report = export_video_with_progress(
        &settings,
        audio_slice,
        &cancel,
        |frame_idx: u32, rgba_out: &mut Vec<u8>| {
            let premul = engine
                .render_frame_rgba8(&project, &plugins, f64::from(frame_idx), width, height)
                .map_err(|e| ExportError::Render(e.to_string()))?;
            let straight = zc_render::unpremultiply_rgba8(&premul);
            rgba_out.clear();
            rgba_out.extend_from_slice(&straight);
            Ok(())
        },
        |_progress| {},
    )
    .map_err(|e| format!("export failed: {e}"))?;

    println!(
        "Exported {} frames ({}x{}, audio_samples={}) -> {}",
        report.frames_exported,
        report.output_width,
        report.output_height,
        report.audio_samples_written,
        report.output_path.display()
    );
    Ok(())
}

fn cmd_selftest_effects(width: u32, height: u32) -> Result<(), String> {
    let mut engine = ProjectRenderEngine::new_headless()
        .map_err(|e| format!("failed to initialize headless GPU renderer: {e}"))?;
    let report = engine
        .selftest_all_effects(width, height)
        .map_err(|e| format!("selftest failed: {e}"))?;

    for entry in &report.entries {
        let kind = if entry.is_plugin { "plugin" } else { "std" };
        if entry.passed {
            println!(
                "PASS [{kind:6}] {:<20} non_zero_pixels={}",
                entry.effect_id, entry.non_zero_pixels
            );
        } else {
            eprintln!(
                "FAIL [{kind:6}] {:<20} bypassed={} error={}",
                entry.effect_id,
                entry.bypassed,
                entry.error.as_deref().unwrap_or("unknown"),
            );
        }
    }

    println!(
        "{} [scene ] {:<20} non_black_pixels={}",
        if report.default_scene_passed {
            "PASS"
        } else {
            "FAIL"
        },
        "cm3_default_3d",
        report.default_scene_non_black_pixels,
    );
    println!(
        "Summary ({}x{}): standard {}/{}, plugin {}/{}, default_3d={}",
        report.width,
        report.height,
        report.standard_passed,
        report.standard_total,
        report.plugin_passed,
        report.plugin_total,
        report.default_scene_passed,
    );

    if report.all_passed() {
        Ok(())
    } else {
        Err("one or more effects or default 3D scene failed GPU self-test".to_owned())
    }
}

fn cmd_control(project_path: Option<&Path>) -> Result<(), String> {
    let mut session = match project_path {
        Some(path) => {
            let (project, warnings) = load_project_any(path)?;
            for warning in &warnings {
                eprintln!("warning: {warning}");
            }
            let plugins = build_plugins_for_project(&project);
            Session::from_project(project, plugins)
        }
        None => Session::new(),
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    dispatch_json_lines(&mut session, stdin.lock(), stdout.lock())
        .map_err(|e| format!("control loop I/O error: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_all_subcommands() {
        let cli = Cli::try_parse_from([
            "zoidcraft-cli",
            "render",
            "--frame",
            "12.5",
            "--out",
            "frame.png",
            "--width",
            "320",
            "--height",
            "180",
        ])
        .expect("render subcommand must parse");
        assert!(matches!(
            cli.command,
            Commands::Render {
                project: None,
                frame,
                width: Some(320),
                height: Some(180),
                ..
            } if (frame - 12.5).abs() < 1e-9
        ));

        let st = Cli::try_parse_from(["zoidcraft-cli", "selftest-effects"])
            .expect("selftest-effects must parse");
        assert!(matches!(
            st.command,
            Commands::SelftestEffects {
                width: 128,
                height: 128
            }
        ));

        let ctrl = Cli::try_parse_from(["zoidcraft-cli", "control"]).expect("control must parse");
        assert!(matches!(ctrl.command, Commands::Control { project: None }));

        let exp = Cli::try_parse_from([
            "zoidcraft-cli",
            "export",
            "--out",
            "out.mp4",
            "--width",
            "320",
            "--height",
            "180",
            "--start-frame",
            "0",
            "--end-frame",
            "15",
        ])
        .expect("export subcommand must parse");
        assert!(matches!(
            exp.command,
            Commands::Export {
                width: Some(320),
                height: Some(180),
                start_frame: 0,
                end_frame: Some(15),
                ..
            }
        ));
    }

    #[test]
    fn mix_project_audio_decodes_and_mixes_embedded_wav_clip() {
        let mut project = Project::new_cm3_default();
        let fps = f64::from(project.sequence.fps.max(1));
        let sr = STANDARD_SAMPLE_RATE;
        let tone_samples = sr as usize / 2;
        let pcm: Vec<[f32; 2]> = (0..tone_samples)
            .map(|i| {
                let s = (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / (sr as f32)).sin() * 0.5;
                [s, s]
            })
            .collect();
        let wav_bytes = zc_audio::encode_wav_pcm16_stereo(&pcm, sr);
        let asset_id = "test_tone_wav".to_owned();
        project.assets.insert(
            asset_id.clone(),
            zc_project::AssetRecord {
                id: asset_id.clone(),
                kind: zc_project::AssetKind::Av,
                source: zc_project::AssetSource::File,
                filename: "tone.wav".to_owned(),
                url: None,
                external: false,
                size: wav_bytes.len() as u64,
                sha256: zc_project::sha256_hex(&wav_bytes),
                missing: false,
                bytes: Some(wav_bytes),
                preserved: std::collections::BTreeMap::new(),
            },
        );
        project.sequence.audio_tracks.push(zc_project::AudioTrack {
            id: zc_project::Uuid::new_v4(),
            name: "Audio 1".to_owned(),
            enabled: true,
            muted: false,
            solo: false,
            clips: vec![zc_project::AudioClip {
                id: zc_project::Uuid::new_v4(),
                name: "Tone".to_owned(),
                start: 0.0,
                length: 15.0,
                link: None,
                media: Some(asset_id),
                time_curve: PropertyTrack::number(0.0),
                audio: zc_project::AudioClipData::default(),
                preserved: std::collections::BTreeMap::new(),
            }],
            preserved: std::collections::BTreeMap::new(),
        });
        // Configure linear time curve 0 -> 15 frames = 0.0 -> 0.5s at 30fps.
        let tc = &mut project.sequence.audio_tracks[0].clips[0].time_curve;
        tc.static_behavior = ProjectStaticBehavior::Linear;
        tc.keyframes = vec![
            zc_project::PropertyKeyframe {
                frame: 0.0,
                value: PropertyValue::Number(0.0),
                tween: 0,
                easing_index: 0,
                is_bezier: false,
                control_points: [[0.0, 0.0], [0.0, 0.0]],
                continuous_tangent: true,
            },
            zc_project::PropertyKeyframe {
                frame: 15.0,
                value: PropertyValue::Number(15.0 / fps),
                tween: 0,
                easing_index: 0,
                is_bezier: false,
                control_points: [[0.0, 0.0], [0.0, 0.0]],
                continuous_tangent: true,
            },
        ];

        let mixed = mix_project_audio(&project, 0, 15, fps, sr)
            .expect("audio mixing must succeed")
            .expect("mixed buffer must be Some");
        assert_eq!(
            mixed.len(),
            zc_audio::mixer::offline_total_samples(15.0, fps, sr)
        );
        let peak = mixed
            .iter()
            .fold(0.0_f32, |acc, &[l, r]| acc.max(l.abs()).max(r.abs()));
        assert!(peak > 0.3, "expected audible sine wave peak, got {peak}");
    }
}
