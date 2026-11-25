#![forbid(unsafe_code)]

//! ZoidCraft headless CLI, batch renderer, `.pz` converter, effect self-test runner,
//! and JSON-Lines session control loop (L5 headless — zero GUI dependencies).

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use clap::{Parser, Subcommand};
use image::{RgbImage, RgbaImage};
#[cfg(test)]
use zc_audio::project_has_active_audio_in_range;
use zc_audio::{
    STANDARD_SAMPLE_RATE, mix_project_audio_with_precomps_filtered,
    project_has_active_audio_in_range_with_precomp,
};
use zc_export::{
    AudioCodec, ExportContainer, ExportError, ExportSettings, QualityMode, VideoCodec,
    export_video_with_progress,
};
use zc_plugin::PluginRegistry;
use zc_project::{
    AssetKind, Layer, LayerKind, Project, PropertyKeyframe, PropertyTrack, PropertyValue, Uuid,
    VideoClip, VideoTrack, create_effect_by_id, export_pz, import_pz, load_zcraft, save_zcraft,
    standard_effect_catalog, zoidium_effect_catalog,
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
    /// Print detailed metadata for a `.zcraft` or `.pz` project archive.
    Info {
        /// Path to the `.zcraft` or `.pz` project file.
        project: PathBuf,
    },

    /// Load and validate `.zcraft` or `.pz` archive structure, JSON bounds, and SHA-256 asset integrity.
    Validate {
        /// Path to the `.zcraft` or `.pz` project file.
        project: PathBuf,
    },

    /// Import a `.pz` archive and atomically save it as `.zcraft`.
    ImportPz {
        /// Path to the input `.pz` file.
        input: PathBuf,
        /// Destination `.zcraft` output path.
        #[arg(long, short = 'o')]
        out: PathBuf,
    },

    /// Export a `.zcraft` or `.pz` project and atomically save it as a `.pz` archive.
    ExportPz {
        /// Path to the input `.zcraft` or `.pz` file.
        input: PathBuf,
        /// Destination `.pz` output path.
        output: PathBuf,
    },

    /// Render a single frame headlessly on the GPU and save it as PNG, JPEG, or WebP.
    Render {
        /// Optional `.zcraft` (or `.pz`) project file; defaults to the CM3 starter project.
        #[arg(long, short = 'p')]
        project: Option<PathBuf>,
        /// Optional child composition name (case-insensitive) or UUID to render instead of the root sequence.
        #[arg(long)]
        comp: Option<String>,
        /// Fractional frame number to render (must be finite and >= 0).
        #[arg(long, short = 'f', default_value_t = 0.0, allow_hyphen_values = true)]
        frame: f64,
        /// Output image file path (`.png`, `.jpg`/`.jpeg`, or `.webp`; defaults to PNG).
        #[arg(long, short = 'o')]
        out: PathBuf,
        /// Optional output width override in pixels.
        #[arg(long)]
        width: Option<u32>,
        /// Optional output height override in pixels.
        #[arg(long)]
        height: Option<u32>,
        /// Render with a transparent root background (`rgba(0, 0, 0, 0)`).
        #[arg(long, default_value_t = false)]
        transparent: bool,
    },

    /// Render a frame range headlessly on the GPU, mix project audio tracks, and export
    /// video (MP4/WebM/MKV/GIF via FFmpeg), PNG image sequence, or audio-only (WAV/M4A).
    Export {
        /// Optional `.zcraft` (or `.pz`) project file; defaults to the CM3 starter project.
        #[arg(long, short = 'p')]
        project: Option<PathBuf>,
        /// Optional child composition name (case-insensitive) or UUID to export instead of the root sequence.
        #[arg(long)]
        comp: Option<String>,
        /// Output file path (or directory for `--format png`).
        #[arg(long, short = 'o')]
        out: PathBuf,
        /// Container format (`mp4`, `webm`, `mkv`, `png`, `gif`, `wav`, `m4a`); inferred from `--out` extension if omitted.
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
        /// Audio sample rate in Hz (default 48000).
        #[arg(long, default_value_t = STANDARD_SAMPLE_RATE)]
        sample_rate: u32,
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
        /// Exclusive ending frame index (defaults to `ceil(sequence.length_frames)`).
        #[arg(long)]
        end_frame: Option<u32>,
        /// Preserve transparent alpha channel (`yuva420p` for VP9, RGBA8 for PNG sequence, GIF transparency).
        #[arg(long, default_value_t = false)]
        transparent: bool,
        /// Optional path to the `ffmpeg` binary (defaults to `ffmpeg` on PATH).
        #[arg(long)]
        ffmpeg: Option<PathBuf>,
    },

    /// Run a headless GPU self-test across all 44 standard effects, all 36 plugin effects
    /// (80 total), and the default 3D scene.
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
        Commands::ExportPz { input, output } => cmd_export_pz(&input, &output),
        Commands::Render {
            project,
            comp,
            frame,
            out,
            width,
            height,
            transparent,
        } => cmd_render(
            project.as_deref(),
            comp.as_deref(),
            frame,
            &out,
            width,
            height,
            transparent,
        ),
        Commands::Export {
            project,
            comp,
            out,
            format,
            video_codec,
            audio_codec,
            crf,
            bitrate_kbps,
            audio_bitrate_kbps,
            sample_rate,
            width,
            height,
            fps,
            start_frame,
            end_frame,
            transparent,
            ffmpeg,
        } => cmd_export(ExportCliArgs {
            project: project.as_deref(),
            comp: comp.as_deref(),
            out: &out,
            format: format.as_deref(),
            video_codec: video_codec.as_deref(),
            audio_codec: audio_codec.as_deref(),
            crf,
            bitrate_kbps,
            audio_bitrate_kbps,
            sample_rate,
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
    let (project, warnings) =
        load_project_any(path).map_err(|e| format!("failed to load {}: {e}", path.display()))?;
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }
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
    let (project, warnings) = load_project_any(path)
        .map_err(|e| format!("validation failed for {}: {e}", path.display()))?;
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }
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

fn cmd_export_pz(input: &Path, output: &Path) -> Result<(), String> {
    let (project, warnings) =
        load_project_any(input).map_err(|e| format!("failed to load {}: {e}", input.display()))?;
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create output directory {}: {e}",
                parent.display()
            )
        })?;
    }
    export_pz(&project, output)
        .map_err(|e| format!("failed to export {}: {e}", output.display()))?;
    println!(
        "Exported '{}' -> {} ({} video tracks, {} audio tracks, {} assets)",
        project.name,
        output.display(),
        project.sequence.video_tracks.len(),
        project.sequence.audio_tracks.len(),
        project.assets.len(),
    );
    Ok(())
}

fn select_composition_for_render(project: &mut Project, comp_spec: &str) -> Result<(), String> {
    let query = comp_spec.trim();
    if query.is_empty() {
        return Err("composition selector `--comp` cannot be empty".to_owned());
    }

    // 1. Exact UUID match against child compositions
    let matched_id = Uuid::parse_str(query)
        .ok()
        .filter(|id| project.compositions.contains_key(id))
        // 2. Legacy string ID match (`composition_ids`)
        .or_else(|| {
            project
                .composition_ids
                .iter()
                .find(|(legacy_id, _)| legacy_id.eq_ignore_ascii_case(query))
                .map(|(_, &uuid)| uuid)
                .filter(|id| project.compositions.contains_key(id))
        })
        // 3. Case-insensitive composition name match
        .or_else(|| {
            project
                .compositions
                .iter()
                .find(|(_, seq)| seq.name.eq_ignore_ascii_case(query))
                .map(|(&id, _)| id)
        });

    if let Some(target_id) = matched_id {
        let mut child = project
            .compositions
            .remove(&target_id)
            .ok_or_else(|| format!("composition '{query}' not found"))?;
        // Keep root sequence controls accessible to child expressions if not already overridden.
        for root_ctl in &project.sequence.controls {
            let exists = child
                .controls
                .iter()
                .any(|c| c.id == root_ctl.id || c.name == root_ctl.name);
            if !exists {
                child.controls.push(root_ctl.clone());
            }
        }
        project.sequence = child;
        project
            .plugin_requirements
            .insert("precomp-plus".to_owned());
        return Ok(());
    }

    // Allow selecting the root sequence explicitly by name, UUID, or `"root"`.
    if query.eq_ignore_ascii_case("root")
        || query.eq_ignore_ascii_case(&project.sequence.name)
        || Uuid::parse_str(query).ok() == Some(project.sequence.id)
    {
        return Ok(());
    }

    let available: Vec<String> = project
        .compositions
        .iter()
        .map(|(id, seq)| format!("'{}' ({id})", seq.name))
        .collect();
    if available.is_empty() {
        Err(format!(
            "composition '{query}' not found (project has no child compositions)"
        ))
    } else {
        Err(format!(
            "composition '{query}' not found (available: {})",
            available.join(", ")
        ))
    }
}

fn cmd_render(
    project_path: Option<&Path>,
    comp: Option<&str>,
    frame: f64,
    out: &Path,
    width_override: Option<u32>,
    height_override: Option<u32>,
    transparent: bool,
) -> Result<(), String> {
    if !frame.is_finite() || frame < 0.0 {
        return Err(format!(
            "invalid frame {frame}: frame must be a finite non-negative number"
        ));
    }

    let (mut project, warnings) = match project_path {
        Some(path) => load_project_any(path)?,
        None => (Project::new_cm3_default(), Vec::new()),
    };
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }
    if let Some(comp_spec) = comp {
        select_composition_for_render(&mut project, comp_spec)?;
    }

    let width = width_override.unwrap_or(project.sequence.resolution[0]);
    let height = height_override.unwrap_or(project.sequence.resolution[1]);
    if width == 0 || height == 0 {
        return Err(format!(
            "invalid render dimensions: {width}x{height} (width and height must be greater than zero)"
        ));
    }
    let plugins = build_plugins_for_project(&project);

    let mut engine = ProjectRenderEngine::new_headless()
        .map_err(|e| format!("failed to initialize headless GPU renderer: {e}"))?;
    engine.set_transparent_background(transparent);
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

    let ext = out
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("jpg" | "jpeg") => {
            // JPEG has no alpha channel: flatten premultiplied pixels directly over black.
            let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
            for px in premul_rgba.as_chunks::<4>().0 {
                rgb.extend_from_slice(&px[..3]);
            }
            let rgb_img = RgbImage::from_raw(width, height, rgb)
                .ok_or_else(|| format!("RGB pixel buffer size mismatch for {width}x{height}"))?;
            rgb_img
                .save_with_format(out, image::ImageFormat::Jpeg)
                .map_err(|e| format!("failed to save JPEG {}: {e}", out.display()))?;
        }
        Some("webp") => {
            img.save_with_format(out, image::ImageFormat::WebP)
                .map_err(|e| format!("failed to save WebP {}: {e}", out.display()))?;
        }
        _ => {
            img.save_with_format(out, image::ImageFormat::Png)
                .map_err(|e| format!("failed to save PNG {}: {e}", out.display()))?;
        }
    }

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
    comp: Option<&'a str>,
    out: &'a Path,
    format: Option<&'a str>,
    video_codec: Option<&'a str>,
    audio_codec: Option<&'a str>,
    crf: u8,
    bitrate_kbps: Option<u32>,
    audio_bitrate_kbps: u32,
    sample_rate: u32,
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
        "gif" => Ok(ExportContainer::Gif),
        "wav" | "audio_wav" => Ok(ExportContainer::Wav),
        "m4a" | "audio_m4a" => Ok(ExportContainer::M4a),
        other => Err(format!(
            "unsupported export format `{other}` (expected mp4, webm, mkv, png, gif, wav, or m4a)"
        )),
    }
}

fn parse_video_codec(
    codec: Option<&str>,
    container: ExportContainer,
    transparent: bool,
) -> Result<VideoCodec, String> {
    let Some(raw) = codec else {
        return Ok(match container {
            ExportContainer::Webm => VideoCodec::Vp9,
            ExportContainer::Mkv if transparent => VideoCodec::Vp9,
            ExportContainer::Mp4
            | ExportContainer::Mkv
            | ExportContainer::PngSequence
            | ExportContainer::Gif
            | ExportContainer::Wav
            | ExportContainer::M4a => VideoCodec::H264,
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
        return Ok(match container {
            ExportContainer::Wav => AudioCodec::Pcm,
            ExportContainer::M4a => AudioCodec::Aac,
            ExportContainer::PngSequence | ExportContainer::Gif => AudioCodec::None,
            ExportContainer::Webm => {
                if has_audio {
                    AudioCodec::Opus
                } else {
                    AudioCodec::None
                }
            }
            ExportContainer::Mp4 | ExportContainer::Mkv => {
                if has_audio {
                    AudioCodec::Aac
                } else {
                    AudioCodec::None
                }
            }
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

#[cfg(test)]
fn build_scalar_property(track: &PropertyTrack, default: f64) -> zc_anim::DynamicProperty<f64> {
    use zc_anim::{DynamicProperty, InterpolationMode, Keyframe, StaticBehavior};
    use zc_project::StaticBehavior as ProjectStaticBehavior;

    let base = match &track.value {
        PropertyValue::Number(v) => *v,
        _ => default,
    };
    let static_behavior = match track.static_behavior {
        ProjectStaticBehavior::Linear => StaticBehavior::Linear,
        ProjectStaticBehavior::Constant => StaticBehavior::Constant,
    };
    let use_keyframes = !track.keyframes.is_empty()
        && (track.animated
            || static_behavior == StaticBehavior::Linear
            || track.keyframes.len() > 1);
    if !use_keyframes {
        return DynamicProperty::new_static(base);
    }
    let mut prop = DynamicProperty::new_static(base);
    prop.static_behavior = static_behavior;
    prop.animated = track.animated;
    prop.keyframes.clear();
    for kf in &track.keyframes {
        if let PropertyValue::Number(v) = kf.value {
            let ease_idx = (kf.tween & 0xFF) as u8;
            let is_bezier = ease_idx != 0 && (kf.tween >> 8) == 1;
            let tween_code = if is_bezier {
                (1u16 << 8) | u16::from(ease_idx)
            } else {
                u16::from(ease_idx)
            };
            let (mode, easing) = InterpolationMode::from_cm3_tween_code(tween_code);
            let mut key = Keyframe::with_easing(kf.frame, v, easing);
            key.interpolated = mode != InterpolationMode::Discrete;
            if mode == InterpolationMode::Bezier {
                key.in_tangent = Some(kf.control_points[0]);
            }
            key.out_tangent = Some(kf.control_points[1]);
            key.continuous_tangent = kf.continuous_tangent;
            prop.set_keyframe(key);
        }
    }
    if prop.keyframes.is_empty() {
        return DynamicProperty::new_static(base);
    }
    prop
}

#[cfg(test)]
fn eval_scalar_track(track: &PropertyTrack, frame: f64, default: f64) -> f64 {
    build_scalar_property(track, default).evaluate(frame)
}

fn mix_project_audio(
    project: &Project,
    start_seq_frame: u32,
    total_seq_frames: f64,
    _seq_fps: f64,
    sample_rate: u32,
) -> Result<Option<Vec<[f32; 2]>>, String> {
    let plugins = build_plugins_for_project(project);
    mix_project_audio_with_precomps_filtered(
        project,
        f64::from(start_seq_frame),
        total_seq_frames,
        sample_rate,
        plugins.is_enabled("precomp-plus"),
    )
    .map_err(|e| format!("failed to mix project audio: {e}"))
}

/// Standard NTSC rational frame rates (`N * 1000 / 1001`), matching `zc-export::ffmpeg::format_fps`.
const NTSC_RATIONAL_FPS: &[f64] = &[
    24_000.0 / 1001.0,
    30_000.0 / 1001.0,
    60_000.0 / 1001.0,
    120_000.0 / 1001.0,
];

/// Snaps shorthand NTSC decimal frame rates (`23.976`, `29.97`, `59.94`, `119.88`) to their exact
/// rational values (`24000/1001`, `30000/1001`, `60000/1001`, `120000/1001`) so frame sampling
/// matches FFmpeg's `-r` flag with zero cumulative A/V drift.
fn canonical_export_fps(fps: f64) -> f64 {
    for &target in NTSC_RATIONAL_FPS {
        if (fps - target).abs() < 5e-4 {
            return target;
        }
    }
    fps
}

/// Returns the effective frame rate of a sequence (`exact_fps` when set and valid, else `fps`),
/// matching `zc-audio::mixer::sequence_effective_fps` and `zc-render::ProjectRenderEngine`.
fn sequence_effective_fps(seq: &zc_project::Sequence) -> f64 {
    seq.exact_fps
        .filter(|f| f.is_finite() && *f > 0.0)
        .unwrap_or_else(|| f64::from(seq.fps.max(1)))
}

#[derive(Debug, Clone, Copy)]
struct ExportTiming {
    src_fps: f64,
    exact_export_fps: f64,
    export_frame_count: u32,
    audio_seq_frames: f64,
}

impl ExportTiming {
    /// Computes the source sequence frame corresponding to output video frame `output_frame_idx`
    /// so video frame `n` at PTS `n / exact_export_fps` samples exact project time `t = n / exact_export_fps`.
    fn scene_frame_for_output(&self, start_seq_frame: u32, output_frame_idx: u32) -> f64 {
        f64::from(start_seq_frame)
            + (f64::from(output_frame_idx) / self.exact_export_fps) * self.src_fps
    }
}

fn compute_export_timing(
    seq: &zc_project::Sequence,
    start_seq_frame: u32,
    end_seq_frame: u32,
    fps_override: Option<f64>,
) -> Result<ExportTiming, String> {
    if end_seq_frame <= start_seq_frame {
        return Err(format!(
            "end_frame ({end_seq_frame}) must be greater than start_frame ({start_seq_frame})"
        ));
    }
    let src_fps = sequence_effective_fps(seq);
    let raw_export_fps = fps_override.unwrap_or(src_fps);
    if !raw_export_fps.is_finite() || raw_export_fps <= 0.0 {
        return Err(format!("invalid export fps: {raw_export_fps}"));
    }
    let exact_export_fps = canonical_export_fps(raw_export_fps);
    let duration_sec = f64::from(end_seq_frame - start_seq_frame) / src_fps;
    let raw_export_frames = duration_sec * exact_export_fps;
    if !raw_export_frames.is_finite() {
        return Err(format!(
            "export frame count out of range for duration {duration_sec}s at {exact_export_fps} fps"
        ));
    }
    let export_frames_f64 = raw_export_frames.round().max(1.0);
    if export_frames_f64 > f64::from(u32::MAX) {
        return Err(format!(
            "export frame count out of range for duration {duration_sec}s at {exact_export_fps} fps"
        ));
    }
    let export_frame_count = export_frames_f64 as u32;
    let audio_seq_frames = (f64::from(export_frame_count) / exact_export_fps) * src_fps;
    Ok(ExportTiming {
        src_fps,
        exact_export_fps,
        export_frame_count,
        audio_seq_frames,
    })
}

fn cmd_export(args: ExportCliArgs<'_>) -> Result<(), String> {
    let (mut project, warnings) = match args.project {
        Some(path) => load_project_any(path)?,
        None => (Project::new_cm3_default(), Vec::new()),
    };
    for warning in &warnings {
        eprintln!("warning: {warning}");
    }
    if let Some(comp_spec) = args.comp {
        select_composition_for_render(&mut project, comp_spec)?;
    }

    let width = args.width.unwrap_or(project.sequence.resolution[0]);
    let height = args.height.unwrap_or(project.sequence.resolution[1]);
    // Match Zoidium export-math.js: `Math.ceil(length)` so fractional trailing frames are not dropped.
    let default_end = project.sequence.length_frames.ceil().max(1.0) as u32;
    let end_seq_frame = args.end_frame.unwrap_or(default_end);
    let timing =
        compute_export_timing(&project.sequence, args.start_frame, end_seq_frame, args.fps)?;
    let seq_fps = timing.src_fps;
    let export_fps = timing.exact_export_fps;
    let export_frame_count = timing.export_frame_count;
    let audio_seq_frames = timing.audio_seq_frames;

    let container = parse_container(args.format, args.out)?;
    let video_codec = parse_video_codec(args.video_codec, container, args.transparent)?;
    let sample_rate = args.sample_rate;
    let plugins = build_plugins_for_project(&project);
    let has_active_audio = if container.is_audio_only() {
        true
    } else if matches!(
        container,
        ExportContainer::PngSequence | ExportContainer::Gif
    ) {
        false
    } else {
        project_has_active_audio_in_range_with_precomp(
            &project,
            f64::from(args.start_frame),
            f64::from(args.start_frame) + audio_seq_frames,
            plugins.is_enabled("precomp-plus"),
        )
    };
    let audio_codec = parse_audio_codec(args.audio_codec, container, has_active_audio)?;
    let quality = match args.bitrate_kbps {
        Some(kbps) => QualityMode::BitrateKbps(kbps),
        None => QualityMode::Crf(args.crf),
    };

    let settings = ExportSettings {
        width,
        height,
        fps: export_fps,
        start_frame: 0,
        end_frame: export_frame_count,
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
    // Validate ALL settings (dimensions, fps bounds, frame count, container/codec compatibility)
    // BEFORE decoding or allocating audio buffers.
    settings.validate().map_err(|e| e.to_string())?;

    let mixed_audio = if matches!(
        container,
        ExportContainer::PngSequence | ExportContainer::Gif
    ) || (!container.is_audio_only()
        && matches!(audio_codec, AudioCodec::None))
    {
        None
    } else {
        mix_project_audio(
            &project,
            args.start_frame,
            audio_seq_frames,
            seq_fps,
            sample_rate,
        )?
    };

    let cancel = Arc::new(AtomicBool::new(false));

    if container.is_audio_only() {
        let report = export_video_with_progress(
            &settings,
            mixed_audio.as_deref(),
            &cancel,
            |_output_frame_idx: u32, _rgba_out: &mut Vec<u8>| Ok(()),
            |_progress| {},
        )
        .map_err(|e| format!("audio export failed: {e}"))?;
        println!(
            "Exported audio ({} samples @ {} Hz) -> {}",
            report.audio_samples_written,
            sample_rate,
            report.output_path.display()
        );
        return Ok(());
    }

    let mut engine = ProjectRenderEngine::new_headless()
        .map_err(|e| format!("failed to initialize headless GPU renderer: {e}"))?;
    engine.set_transparent_background(args.transparent);
    let start_seq_frame = args.start_frame;
    let transparent = args.transparent;

    let report = export_video_with_progress(
        &settings,
        mixed_audio.as_deref(),
        &cancel,
        |output_frame_idx: u32, rgba_out: &mut Vec<u8>| {
            let scene_frame = timing.scene_frame_for_output(start_seq_frame, output_frame_idx);
            let premul = engine
                .render_frame_rgba8(&project, &plugins, scene_frame, width, height)
                .map_err(|e| ExportError::Render(e.to_string()))?;
            rgba_out.clear();
            if transparent {
                let straight = zc_render::unpremultiply_rgba8(&premul);
                rgba_out.extend_from_slice(&straight);
            } else {
                rgba_out.extend_from_slice(&premul);
                for px in rgba_out.as_chunks_mut::<4>().0 {
                    px[3] = 255;
                }
            }
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

fn build_selftest_gradient_png(width: u32, height: u32) -> Result<Vec<u8>, String> {
    let mut rgba = Vec::with_capacity((width as usize) * (height as usize) * 4);
    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / width.max(1) as f32;
            let v = y as f32 / height.max(1) as f32;
            // Vibrant red/amber/blue diagonal pattern (avoiding pure chroma-green so keying
            // effects retain visible foreground pixels).
            let r = (40.0 + u * 200.0).clamp(0.0, 255.0) as u8;
            let g = (30.0 + (u * v) * 140.0).clamp(0.0, 255.0) as u8;
            let b = (60.0 + (1.0 - v) * 180.0).clamp(0.0, 255.0) as u8;
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
    }
    zc_render::encode_png(&rgba, width, height)
        .map_err(|e| format!("failed to encode selftest gradient PNG: {e}"))
}

fn cmd_selftest_effects(width: u32, height: u32) -> Result<(), String> {
    let mut engine = ProjectRenderEngine::new_headless()
        .map_err(|e| format!("failed to initialize headless GPU renderer: {e}"))?;
    let report = engine
        .selftest_all_effects(width, height)
        .map_err(|e| format!("selftest failed: {e}"))?;

    let mut tested_std: HashSet<String> = HashSet::new();
    let mut tested_plugin: HashSet<String> = HashSet::new();
    let mut std_passed = 0usize;
    let mut plugin_passed = 0usize;

    for entry in &report.entries {
        let kind = if entry.is_plugin { "plugin" } else { "std" };
        if entry.is_plugin {
            tested_plugin.insert(entry.effect_id.to_ascii_lowercase());
            if entry.passed {
                plugin_passed += 1;
            }
        } else {
            tested_std.insert(entry.effect_id.to_ascii_lowercase());
            if entry.passed {
                std_passed += 1;
            }
        }
        if entry.passed {
            println!(
                "PASS [{kind:6}] {:<24} non_zero_pixels={}",
                entry.effect_id, entry.non_zero_pixels
            );
        } else {
            eprintln!(
                "FAIL [{kind:6}] {:<24} bypassed={} error={}",
                entry.effect_id,
                entry.bypassed,
                entry.error.as_deref().unwrap_or("unknown"),
            );
        }
    }

    // Verify all remaining catalog entries across standard_effect_catalog() (44) and
    // zoidium_effect_catalog() (36) — 80 total — through the full ProjectRenderEngine pipeline.
    let mut all_plugins = PluginRegistry::new();
    for desc in zoidium_effect_catalog() {
        if let Some(pid) = desc.plugin_id {
            let _ = all_plugins.enable(pid);
        }
    }
    let test_png = build_selftest_gradient_png(width, height)?;
    let expected_bytes = (width as usize) * (height as usize) * 4;

    let mut run_catalog_effect = |effect_id: &str, is_plugin: bool, plugin_id: Option<&str>| {
        let mut proj = Project::new_cm3_default();
        proj.sequence.resolution = [width, height];
        proj.sequence.fps = 30;
        proj.sequence.length_frames = 30.0;
        if let Some(pid) = plugin_id {
            proj.plugin_requirements.insert(pid.to_owned());
        }
        let asset_id = proj.insert_binary_asset(
            AssetKind::Image,
            "selftest_grad.png".to_owned(),
            test_png.clone(),
        );

        let mut base_layer = Layer::new(
            "Base",
            LayerKind::Image {
                image: Some(asset_id.clone()),
            },
            [width, height],
        );
        // Animate position across frames 0..10 so temporal/motion effects observe real motion.
        base_layer.position = PropertyTrack {
            animated: true,
            expression: None,
            static_behavior: zc_project::StaticBehavior::Constant,
            value: PropertyValue::Vec2([0.0, 0.0]),
            keyframes: vec![
                PropertyKeyframe::linear(0.0, PropertyValue::Vec2([-12.0, -6.0])),
                PropertyKeyframe::linear(10.0, PropertyValue::Vec2([12.0, 6.0])),
            ],
            channels: Vec::new(),
        };

        let mut fx_layer = base_layer.clone();
        fx_layer.id = Uuid::new_v4();
        fx_layer.name = format!("FX {effect_id}");

        let fx_res = create_effect_by_id(effect_id).ok_or_else(|| {
            format!(
                "catalog effect `{effect_id}` could not be instantiated via create_effect_by_id"
            )
        });

        let kind = if is_plugin { "plugin" } else { "std" };
        match fx_res.and_then(|fx| {
            fx_layer.effects.push(fx);
            proj.sequence.video_tracks = vec![
                VideoTrack {
                    id: Uuid::new_v4(),
                    name: "Track 1".to_owned(),
                    enabled: true,
                    solo: false,
                    locked: false,
                    is_3d: false,
                    clips: vec![VideoClip {
                        id: Uuid::new_v4(),
                        name: "Base Clip".to_owned(),
                        start: 0.0,
                        length: 30.0,
                        link: None,
                        media: None,
                        time_curve: PropertyTrack::linear_time_curve(30.0, 30),
                        layer: base_layer,
                        preserved: BTreeMap::new(),
                    }],
                    preserved: BTreeMap::new(),
                },
                VideoTrack {
                    id: Uuid::new_v4(),
                    name: "Track 2".to_owned(),
                    enabled: true,
                    solo: false,
                    locked: false,
                    is_3d: false,
                    clips: vec![VideoClip {
                        id: Uuid::new_v4(),
                        name: format!("FX Clip {effect_id}"),
                        start: 0.0,
                        length: 30.0,
                        link: None,
                        media: None,
                        time_curve: PropertyTrack::linear_time_curve(30.0, 30),
                        layer: fx_layer,
                        preserved: BTreeMap::new(),
                    }],
                    preserved: BTreeMap::new(),
                },
            ];
            engine
                .render_frame_rgba8(&proj, &all_plugins, 5.0, width, height)
                .map_err(|e| e.to_string())
        }) {
            Ok(pixels) => {
                let non_zero = pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .filter(|px| px[0] > 0 || px[1] > 0 || px[2] > 0 || px[3] > 0)
                    .count();
                if pixels.len() == expected_bytes && non_zero > 0 {
                    println!("PASS [{kind:6}] {effect_id:<24} non_zero_pixels={non_zero}");
                    true
                } else {
                    eprintln!(
                        "FAIL [{kind:6}] {effect_id:<24} error=empty output (bytes={}, non_zero_pixels={non_zero})",
                        pixels.len()
                    );
                    false
                }
            }
            Err(err) => {
                eprintln!("FAIL [{kind:6}] {effect_id:<24} error={err}");
                false
            }
        }
    };

    for desc in standard_effect_catalog() {
        let lower = desc.id.to_ascii_lowercase();
        if tested_std.insert(lower) && run_catalog_effect(desc.id, false, desc.plugin_id) {
            std_passed += 1;
        }
    }
    for desc in zoidium_effect_catalog() {
        let lower = desc.id.to_ascii_lowercase();
        if tested_plugin.insert(lower) && run_catalog_effect(desc.id, true, desc.plugin_id) {
            plugin_passed += 1;
        }
    }

    let std_total = standard_effect_catalog().len().max(report.standard_total);
    let plugin_total = zoidium_effect_catalog().len().max(tested_plugin.len());

    println!(
        "{} [scene ] {:<24} non_black_pixels={}",
        if report.default_scene_passed {
            "PASS"
        } else {
            "FAIL"
        },
        "cm3_default_3d",
        report.default_scene_non_black_pixels,
    );
    println!(
        "Summary ({}x{}): standard {}/{}, plugin {}/{}, total {}/{}, default_3d={}",
        report.width,
        report.height,
        std_passed,
        std_total,
        plugin_passed,
        plugin_total,
        std_passed + plugin_passed,
        std_total + plugin_total,
        report.default_scene_passed,
    );

    if std_passed == std_total && plugin_passed == plugin_total && report.default_scene_passed {
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

        let exp_pz =
            Cli::try_parse_from(["zoidcraft-cli", "export-pz", "source.zcraft", "archive.pz"])
                .expect("export-pz subcommand must parse");
        assert!(matches!(
            exp_pz.command,
            Commands::ExportPz { ref input, ref output }
                if input == Path::new("source.zcraft") && output == Path::new("archive.pz")
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
        project.sequence.audio_tracks = vec![zc_project::AudioTrack {
            id: zc_project::Uuid::new_v4(),
            name: "Audio 1".to_owned(),
            enabled: true,
            muted: false,
            solo: false,
            locked: false,
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
        }];
        // Configure linear time curve 0 -> 15 frames = 0.0 -> 0.5s at 30fps.
        project.sequence.audio_tracks[0].clips[0].time_curve =
            PropertyTrack::linear_time_curve(15.0, project.sequence.fps);

        let mixed = mix_project_audio(&project, 0, 15.0, fps, sr)
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

    #[test]
    fn parse_video_codec_defaults_mkv_transparent_to_vp9() {
        assert_eq!(
            parse_video_codec(None, ExportContainer::Mkv, true).expect("mkv transparent"),
            VideoCodec::Vp9
        );
        assert_eq!(
            parse_video_codec(None, ExportContainer::Mkv, false).expect("mkv opaque"),
            VideoCodec::H264
        );
    }

    #[test]
    fn eval_scalar_track_holds_first_keyframe_after_zero_and_uses_bezier_mode() {
        let mut track = PropertyTrack::number(0.0);
        track.animated = true;
        track.keyframes = vec![
            zc_project::PropertyKeyframe {
                frame: 10.0,
                value: PropertyValue::Number(2.0),
                tween: 0,
                easing_index: 0,
                is_bezier: false,
                control_points: [[0.0, 0.0], [3.0, 2.0]],
                continuous_tangent: true,
            },
            zc_project::PropertyKeyframe {
                frame: 20.0,
                value: PropertyValue::Number(4.0),
                tween: (1 << 8) | 1, // Bezier mode via tween code even if is_bezier is false
                easing_index: 0,
                is_bezier: false,
                control_points: [[-3.0, -2.0], [0.0, 0.0]],
                continuous_tangent: true,
            },
        ];
        assert!((eval_scalar_track(&track, 0.0, 0.0) - 2.0).abs() < 1e-9);
        assert!((eval_scalar_track(&track, 5.0, 0.0) - 2.0).abs() < 1e-9);

        let prop = build_scalar_property(&track, 0.0);
        assert_eq!(prop.keyframes.len(), 2);
        assert_eq!(prop.keyframes[1].in_tangent, Some([-3.0, -2.0]));
    }

    #[test]
    fn hold_keyframe_with_stale_easing_index_stays_hold() {
        let mut track = PropertyTrack::number(0.0);
        track.animated = true;
        track.keyframes = vec![
            zc_project::PropertyKeyframe {
                frame: 0.0,
                value: PropertyValue::Number(1.0),
                tween: 1,
                easing_index: 1,
                is_bezier: false,
                control_points: [[0.0, 0.0], [0.0, 0.0]],
                continuous_tangent: true,
            },
            zc_project::PropertyKeyframe {
                frame: 10.0,
                value: PropertyValue::Number(5.0),
                tween: 0, // Hold must win over stale easing_index / is_bezier!
                easing_index: 1,
                is_bezier: true,
                control_points: [[-5.0, 0.0], [5.0, 0.0]],
                continuous_tangent: true,
            },
        ];
        // Mid-segment at frame 5.0 must hold k0 (1.0), NOT interpolate to 3.0.
        assert!((eval_scalar_track(&track, 5.0, 0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn soloed_muted_track_silences_mix_instead_of_unmuting_others() {
        let mut project = Project::new_cm3_default();
        let fps = f64::from(project.sequence.fps.max(1));
        let sr = STANDARD_SAMPLE_RATE;
        let wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.6, 0.6]; sr as usize / 2], sr);
        let asset_id = "dc_tone".to_owned();
        project.assets.insert(
            asset_id.clone(),
            zc_project::AssetRecord {
                id: asset_id.clone(),
                kind: zc_project::AssetKind::Av,
                source: zc_project::AssetSource::File,
                filename: "dc.wav".to_owned(),
                url: None,
                external: false,
                size: wav.len() as u64,
                sha256: zc_project::sha256_hex(&wav),
                missing: false,
                bytes: Some(wav),
                preserved: std::collections::BTreeMap::new(),
            },
        );
        let mk_clip = || zc_project::AudioClip {
            id: Uuid::new_v4(),
            name: "Tone".to_owned(),
            start: 0.0,
            length: 10.0,
            link: None,
            media: Some(asset_id.clone()),
            time_curve: PropertyTrack::linear_time_curve(10.0, project.sequence.fps),
            audio: zc_project::AudioClipData::default(),
            preserved: std::collections::BTreeMap::new(),
        };
        // Track 1 is soloed AND muted; Track 2 is normal (not soloed).
        // Per Zoidium timeline-track-state.js, `hasSolo` is still true -> output must be silent!
        project.sequence.audio_tracks = vec![
            zc_project::AudioTrack {
                id: Uuid::new_v4(),
                name: "A1".to_owned(),
                enabled: true,
                muted: true,
                solo: true,
                locked: false,
                clips: vec![mk_clip()],
                preserved: std::collections::BTreeMap::new(),
            },
            zc_project::AudioTrack {
                id: Uuid::new_v4(),
                name: "A2".to_owned(),
                enabled: true,
                muted: false,
                solo: false,
                locked: false,
                clips: vec![mk_clip()],
                preserved: std::collections::BTreeMap::new(),
            },
        ];

        assert!(!project_has_active_audio_in_range(&project, 0.0, 10.0));
        let mixed = mix_project_audio(&project, 0, 10.0, fps, sr).expect("mix ok");
        assert!(mixed.is_none());
    }

    #[test]
    fn audio_expressions_and_nested_precomp_audio_are_mixed() {
        let mut project = Project::new_cm3_default();
        let fps = f64::from(project.sequence.fps.max(1));
        let sr = STANDARD_SAMPLE_RATE;
        // Constant 0.5 stereo buffer (1.0s)
        let wav = zc_audio::encode_wav_pcm16_stereo(&vec![[0.5, 0.5]; sr as usize], sr);
        let asset_id = "const_half".to_owned();
        project.assets.insert(
            asset_id.clone(),
            zc_project::AssetRecord {
                id: asset_id.clone(),
                kind: zc_project::AssetKind::Av,
                source: zc_project::AssetSource::File,
                filename: "const.wav".to_owned(),
                url: None,
                external: false,
                size: wav.len() as u64,
                sha256: zc_project::sha256_hex(&wav),
                missing: false,
                bytes: Some(wav),
                preserved: std::collections::BTreeMap::new(),
            },
        );

        // Create a nested composition with an audio clip driven by volume & pan expressions:
        // volume = "0.4", pan = "1.0" (hard right -> L=0, R = (0.5+0.5)*0.4 = 0.4)
        let comp_id = Uuid::new_v4();
        let mut comp_seq = zc_project::Sequence {
            id: comp_id,
            fps: project.sequence.fps,
            length_frames: 30.0,
            ..zc_project::Sequence::default()
        };
        let mut audio_data = zc_project::AudioClipData::default();
        audio_data.volume.expression = Some("0.4".to_owned());
        audio_data.pan.expression = Some("1.0".to_owned());
        comp_seq.audio_tracks.push(zc_project::AudioTrack {
            id: Uuid::new_v4(),
            name: "Comp Audio".to_owned(),
            enabled: true,
            muted: false,
            solo: false,
            locked: false,
            clips: vec![zc_project::AudioClip {
                id: Uuid::new_v4(),
                name: "Nested".to_owned(),
                start: 0.0,
                length: 15.0,
                link: None,
                media: Some(asset_id),
                time_curve: PropertyTrack::linear_time_curve(15.0, comp_seq.fps),
                audio: audio_data,
                preserved: std::collections::BTreeMap::new(),
            }],
            preserved: std::collections::BTreeMap::new(),
        });
        project.compositions.insert(comp_id, comp_seq);

        // Reference that nested composition from root sequence video track (no top-level audio tracks).
        project.sequence.audio_tracks.clear();
        project.sequence.video_tracks = vec![zc_project::VideoTrack {
            id: Uuid::new_v4(),
            name: "V1".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![zc_project::VideoClip {
                id: Uuid::new_v4(),
                name: "Precomp Clip".to_owned(),
                start: 0.0,
                length: 15.0,
                link: None,
                media: None,
                time_curve: PropertyTrack::linear_time_curve(15.0, project.sequence.fps),
                layer: zc_project::Layer::new(
                    "Precomp",
                    LayerKind::PrecompRef {
                        composition_id: comp_id,
                    },
                    project.sequence.resolution,
                ),
                preserved: std::collections::BTreeMap::new(),
            }],
            preserved: std::collections::BTreeMap::new(),
        }];

        // Without `precomp-plus` enabled, CLI audio matches video by skipping PrecompRef clips.
        assert!(
            !project_has_active_audio_in_range_with_precomp(&project, 0.0, 15.0, false),
            "nested Precomp+ audio must be skipped when precomp-plus is disabled"
        );
        assert!(
            mix_project_audio(&project, 0, 15.0, fps, sr)
                .expect("disabled precomp mix ok")
                .is_none(),
            "CLI mixer must skip nested Precomp+ audio when precomp-plus is not required"
        );

        project
            .plugin_requirements
            .insert("precomp-plus".to_owned());
        assert!(project_has_active_audio_in_range(&project, 0.0, 15.0));
        let mixed = mix_project_audio(&project, 0, 15.0, fps, sr)
            .expect("nested audio mix ok")
            .expect("nested audio must produce samples");
        let mid = mixed[mixed.len() / 2];
        assert!(
            mid[0].abs() < 1e-3,
            "hard-right pan L must be ~0, got {}",
            mid[0]
        );
        assert!(
            (mid[1] - 0.4).abs() < 1e-2,
            "hard-right pan R must be ~0.4, got {}",
            mid[1]
        );
    }

    fn build_minimal_pz_archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut tar_bytes = Vec::new();
        for &(name, data) in entries {
            let mut header = [0_u8; 512];
            header[..name.len()].copy_from_slice(name.as_bytes());
            header[100..108].copy_from_slice(b"0000644\0");
            header[108..116].copy_from_slice(b"0000000\0");
            header[116..124].copy_from_slice(b"0000000\0");
            let size_oct = format!("{:011o}\0", data.len());
            header[124..136].copy_from_slice(size_oct.as_bytes());
            header[136..148].copy_from_slice(b"00000000000\0");
            header[148..156].copy_from_slice(b"        ");
            header[156] = b'0';
            header[257..263].copy_from_slice(b"ustar\0");
            header[263..265].copy_from_slice(b"00");
            let cksum: u32 = header.iter().map(|&b| u32::from(b)).sum();
            let cksum_oct = format!("{cksum:06o}\0 ");
            header[148..156].copy_from_slice(cksum_oct.as_bytes());
            tar_bytes.extend_from_slice(&header);
            tar_bytes.extend_from_slice(data);
            let pad = (512 - (data.len() % 512)) % 512;
            tar_bytes.resize(tar_bytes.len() + pad, 0);
        }
        tar_bytes.resize(tar_bytes.len() + 1024, 0);

        // Wrap `tar_bytes` in a valid RFC 1952 gzip stream using uncompressed DEFLATE stored blocks.
        let mut gz = vec![0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0xff];
        let chunks: Vec<&[u8]> = tar_bytes.chunks(65_535).collect();
        for (i, chunk) in chunks.iter().enumerate() {
            let is_final = u8::from(i + 1 == chunks.len());
            gz.push(is_final);
            let len = chunk.len() as u16;
            gz.extend_from_slice(&len.to_le_bytes());
            gz.extend_from_slice(&(!len).to_le_bytes());
            gz.extend_from_slice(chunk);
        }
        // IEEE 802.3 CRC-32
        let mut crc: u32 = 0xFFFF_FFFF;
        for &b in &tar_bytes {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = if (crc & 1) != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        crc = !crc;
        gz.extend_from_slice(&crc.to_le_bytes());
        gz.extend_from_slice(&(tar_bytes.len() as u32).to_le_bytes());
        gz
    }

    #[test]
    fn cli_info_and_validate_accept_pz_archives_and_export_parses_new_flags() {
        let tmp_dir =
            std::env::temp_dir().join(format!("zoidcraft_cli_pz_test_{}", std::process::id()));
        let _ = fs::create_dir_all(&tmp_dir);
        let pz_path = tmp_dir.join("minimal.pz");
        let meta_json = br#"{"version": 5}"#;
        let project_json = br#"{"sequence":{"properties":{"resolution":{"value":[640,360]},"rate":{"value":30},"length":{"value":30}},"videoTracks":[],"audioTracks":[]}}"#;
        let pz_bytes =
            build_minimal_pz_archive(&[("meta.json", meta_json), ("project.json", project_json)]);
        fs::write(&pz_path, &pz_bytes).expect("write test .pz");

        cmd_info(&pz_path).expect("cmd_info must accept .pz");
        cmd_validate(&pz_path).expect("cmd_validate must accept .pz");
        let _ = fs::remove_dir_all(&tmp_dir);

        let cli = Cli::try_parse_from([
            "zoidcraft-cli",
            "export",
            "--out",
            "out.wav",
            "--sample-rate",
            "44100",
        ])
        .expect("export with --sample-rate and wav must parse");
        assert!(matches!(
            cli.command,
            Commands::Export {
                sample_rate: 44_100,
                ..
            }
        ));

        let ren = Cli::try_parse_from([
            "zoidcraft-cli",
            "render",
            "--out",
            "frame.png",
            "--transparent",
        ])
        .expect("render --transparent must parse");
        assert!(matches!(
            ren.command,
            Commands::Render {
                transparent: true,
                ..
            }
        ));

        assert_eq!(
            parse_container(None, Path::new("clip.gif")).unwrap(),
            ExportContainer::Gif
        );
        assert_eq!(
            parse_container(None, Path::new("mix.wav")).unwrap(),
            ExportContainer::Wav
        );
        assert_eq!(
            parse_container(None, Path::new("mix.m4a")).unwrap(),
            ExportContainer::M4a
        );
    }

    #[test]
    fn video_only_mp4_on_audio_clip_is_treated_as_silent_not_error() {
        if zc_export::probe_ffmpeg(None).is_err() {
            return;
        }
        let tmp_mp4 = std::env::temp_dir().join(format!(
            "zoidcraft_cli_video_only_{}.mp4",
            std::process::id()
        ));
        let settings = ExportSettings {
            width: 16,
            height: 16,
            fps: 30.0,
            start_frame: 0,
            end_frame: 2,
            container: ExportContainer::Mp4,
            video_codec: VideoCodec::H264,
            audio_codec: AudioCodec::None,
            quality: QualityMode::Crf(23),
            audio_bitrate_kbps: 0,
            sample_rate: STANDARD_SAMPLE_RATE,
            transparent_alpha: false,
            output_path: tmp_mp4.clone(),
            ffmpeg_binary: None,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        zc_export::export_video(&settings, None, &cancel, |_idx, buf: &mut Vec<u8>| {
            buf.fill(255);
            Ok(())
        })
        .expect("create synthetic video-only MP4");
        let mp4_bytes = fs::read(&tmp_mp4).expect("read video-only MP4");
        let _ = fs::remove_file(&tmp_mp4);

        let mut project = Project::new_cm3_default();
        let fps = f64::from(project.sequence.fps.max(1));
        let asset_id = "video_only_mp4".to_owned();
        project.assets.insert(
            asset_id.clone(),
            zc_project::AssetRecord {
                id: asset_id.clone(),
                kind: zc_project::AssetKind::Av,
                source: zc_project::AssetSource::File,
                filename: "silent_video.mp4".to_owned(),
                url: None,
                external: false,
                size: mp4_bytes.len() as u64,
                sha256: zc_project::sha256_hex(&mp4_bytes),
                missing: false,
                bytes: Some(mp4_bytes),
                preserved: std::collections::BTreeMap::new(),
            },
        );
        project.sequence.audio_tracks = vec![zc_project::AudioTrack {
            id: Uuid::new_v4(),
            name: "Audio 1".to_owned(),
            enabled: true,
            muted: false,
            solo: false,
            locked: false,
            clips: vec![zc_project::AudioClip {
                id: Uuid::new_v4(),
                name: "Silent Video".to_owned(),
                start: 0.0,
                length: 10.0,
                link: None,
                media: Some(asset_id),
                time_curve: PropertyTrack::linear_time_curve(10.0, project.sequence.fps),
                audio: zc_project::AudioClipData::default(),
                preserved: std::collections::BTreeMap::new(),
            }],
            preserved: std::collections::BTreeMap::new(),
        }];

        assert!(
            !project_has_active_audio_in_range(&project, 0.0, 10.0),
            "video-only MP4 clip must not advertise an active audio stream"
        );
        let mixed = mix_project_audio(&project, 0, 10.0, fps, STANDARD_SAMPLE_RATE)
            .expect("video-only MP4 must not abort audio mixing");
        assert!(mixed.is_none());
    }

    #[test]
    fn cli_export_pz_round_trips_zcraft_and_pz_through_import_pz() {
        let tmp_dir =
            std::env::temp_dir().join(format!("zoidcraft_cli_export_pz_rt_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&tmp_dir).expect("create temp test dir");

        let mut project = Project::new_cm3_default();
        project.name = "CLI Roundtrip Demo".to_owned();
        project.sequence.resolution = [1280, 720];
        project.sequence.fps = 60;
        project.sequence.length_frames = 120.0;
        let payload = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3, 4];
        let asset_id = project.insert_binary_asset(
            zc_project::AssetKind::Image,
            "badge.png".to_owned(),
            payload.clone(),
        );

        let zcraft_path = tmp_dir.join("input.zcraft");
        save_zcraft(&project, &zcraft_path).expect("save .zcraft input");

        // 1. Export `.zcraft` -> `.pz` (also verifying nested output directory creation).
        let pz_out = tmp_dir.join("nested").join("exported.pz");
        cmd_export_pz(&zcraft_path, &pz_out).expect("cmd_export_pz from .zcraft must succeed");
        assert!(pz_out.is_file(), "exported .pz file must exist");

        let imported = import_pz(&pz_out).expect("import_pz must read exported .pz");
        assert!(
            imported.warnings.is_empty(),
            "expected zero warnings on round-trip import, got: {:?}",
            imported.warnings
        );
        assert_eq!(imported.project.name, "CLI Roundtrip Demo");
        assert_eq!(imported.project.sequence.resolution, [1280, 720]);
        assert_eq!(imported.project.sequence.fps, 60);
        assert!((imported.project.sequence.length_frames - 120.0).abs() < 1e-6);
        assert_eq!(
            imported.project.sequence.video_tracks.len(),
            project.sequence.video_tracks.len()
        );
        assert!(
            imported.project.assets.values().any(
                |a| a.filename == "badge.png" && a.bytes.as_deref() == Some(payload.as_slice())
            ),
            "embedded asset ({asset_id}) must survive .zcraft -> export-pz -> import_pz"
        );

        // 2. Re-export `.pz` -> `.pz` via CLI `run(...)` and re-import cleanly.
        let pz_reout = tmp_dir.join("reexported.pz");
        let cli = Cli::try_parse_from([
            "zoidcraft-cli",
            "export-pz",
            pz_out.to_str().unwrap(),
            pz_reout.to_str().unwrap(),
        ])
        .expect("parse export-pz with real paths");
        run(cli).expect("run export-pz from .pz must succeed");

        let reimported = import_pz(&pz_reout).expect("import_pz must read re-exported .pz");
        assert!(
            reimported.warnings.is_empty(),
            "expected zero warnings on .pz -> .pz re-import, got: {:?}",
            reimported.warnings
        );
        assert_eq!(reimported.project.name, "CLI Roundtrip Demo");
        assert_eq!(reimported.project.sequence.resolution, [1280, 720]);
        assert_eq!(reimported.project.sequence.fps, 60);

        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn build_scalar_property_evaluates_unanimated_multi_keyframes_and_preserves_k0_out_tangent() {
        // 1. Multi-keyframe track with `animated: false` and `StaticBehavior::Constant`
        //    must still evaluate all keyframes (matching CM3).
        let mut unanimated = PropertyTrack::number(0.0);
        unanimated.animated = false;
        unanimated.static_behavior = zc_project::StaticBehavior::Constant;
        unanimated.keyframes = vec![
            zc_project::PropertyKeyframe::linear(0.0, PropertyValue::Number(0.0)),
            zc_project::PropertyKeyframe::linear(10.0, PropertyValue::Number(1.0)),
        ];
        assert!(
            (eval_scalar_track(&unanimated, 5.0, 0.0) - 0.5).abs() < 1e-9,
            "multi-keyframe track with animated=false must interpolate at midpoint"
        );
        assert!(
            (eval_scalar_track(&unanimated, 10.0, 0.0) - 1.0).abs() < 1e-9,
            "multi-keyframe track with animated=false must reach last keyframe"
        );

        // 2. Non-Bezier `k0` carrying a custom outgoing tangent `control_points[1]` into a
        //    Bezier `k1` (`(tween >> 8) == 1`) must preserve `k0.out_tangent`.
        let mut bezier_track = PropertyTrack::number(0.0);
        bezier_track.animated = true;
        bezier_track.keyframes = vec![
            zc_project::PropertyKeyframe {
                frame: 0.0,
                value: PropertyValue::Number(0.0),
                tween: 1, // Linear (NOT Bezier) on k0
                easing_index: 1,
                is_bezier: false,
                control_points: [[0.0, 0.0], [2.0, 8.0]],
                continuous_tangent: false,
            },
            zc_project::PropertyKeyframe {
                frame: 10.0,
                value: PropertyValue::Number(1.0),
                tween: (1 << 8) | 1, // Bezier mode on segment end keyframe k1
                easing_index: 1,
                is_bezier: true,
                control_points: [[-2.0, 0.0], [0.0, 0.0]],
                continuous_tangent: false,
            },
        ];

        let prop = build_scalar_property(&bezier_track, 0.0);
        assert_eq!(prop.keyframes.len(), 2);
        assert_eq!(prop.keyframes[0].in_tangent, None);
        assert_eq!(prop.keyframes[0].out_tangent, Some([2.0, 8.0]));
        assert_eq!(prop.keyframes[1].in_tangent, Some([-2.0, 0.0]));

        let actual = prop.evaluate(5.0);
        let expected = zc_anim::eval_segment_bezier(
            zc_anim::BezierSegment {
                start_frame: 0.0,
                start_value: 0.0,
                out_tangent: [2.0, 8.0],
                end_frame: 10.0,
                end_value: 1.0,
                in_tangent: [-2.0, 0.0],
            },
            5.0,
        );
        let fallback_default_tangent = zc_anim::eval_segment_bezier(
            zc_anim::BezierSegment {
                start_frame: 0.0,
                start_value: 0.0,
                out_tangent: [10.0, 0.0],
                end_frame: 10.0,
                end_value: 1.0,
                in_tangent: [-2.0, 0.0],
            },
            5.0,
        );
        assert!(
            (actual - expected).abs() < 1e-9,
            "Bezier evaluation must use k0.control_points[1] (got {actual}, expected {expected})"
        );
        assert!(
            (actual - fallback_default_tangent).abs() > 1.0,
            "k0 custom out_tangent must meaningfully differ from default [10.0, 0.0] tangent"
        );
    }

    #[test]
    fn cli_selftest_effects_covers_all_80_catalog_effects_and_default_scene() {
        assert_eq!(standard_effect_catalog().len(), 44);
        assert_eq!(zoidium_effect_catalog().len(), 36);
        let require_gpu = std::env::var("ZOIDCRAFT_REQUIRE_GPU")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        if !require_gpu && ProjectRenderEngine::new_headless().is_err() {
            return;
        }
        cmd_selftest_effects(64, 64).expect(
            "all 44 standard + 36 plugin effects + default 3D scene must pass GPU selftest",
        );
    }

    #[test]
    fn cli_render_rejects_negative_nan_and_infinite_frames() {
        let out = PathBuf::from("unused_frame.png");
        for bad_frame in [-5.0_f64, -0.001, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = cmd_render(None, None, bad_frame, &out, Some(64), Some(64), false)
                .expect_err("invalid frame must be rejected");
            assert!(
                err.contains("invalid frame"),
                "expected clear invalid frame message, got: {err}"
            );
        }

        // Verify clap accepts negative & non-finite frame flags so `run()` returns our clean Err.
        for args in [
            vec!["zoidcraft-cli", "render", "-f=-5", "-o", "out.png"],
            vec!["zoidcraft-cli", "render", "--frame", "-5", "-o", "out.png"],
            vec!["zoidcraft-cli", "render", "--frame", "NaN", "-o", "out.png"],
            vec!["zoidcraft-cli", "render", "--frame", "inf", "-o", "out.png"],
            vec![
                "zoidcraft-cli",
                "render",
                "--frame",
                "-inf",
                "-o",
                "out.png",
            ],
        ] {
            let cli = Cli::try_parse_from(&args).expect("clap should parse negative/special float");
            let err = run(cli).expect_err("run() must reject invalid frame");
            assert!(err.contains("invalid frame"), "unexpected error: {err}");
        }
    }

    #[test]
    fn cli_validate_rejects_truncated_gzip_zcraft() {
        let tmp = std::env::temp_dir().join(format!("zc_cli_trunc_{}.zcraft", Uuid::new_v4()));
        let proj = Project::new_cm3_default();
        let bytes = zc_project::save_zcraft_to_bytes(&proj).expect("save zcraft bytes");
        // Truncate inside the gzip trailer so TAR entries decompress, yet gzip stream is incomplete.
        fs::write(&tmp, &bytes[..bytes.len().saturating_sub(4)]).expect("write truncated file");

        let res = cmd_validate(&tmp);
        let _ = fs::remove_file(&tmp);
        assert!(
            res.is_err(),
            "validate must fail on truncated gzip .zcraft file"
        );
    }

    #[test]
    fn cli_comp_flag_parses_selects_and_renders_child_composition() {
        let cli = Cli::try_parse_from([
            "zoidcraft-cli",
            "render",
            "--comp",
            "Precomp 1",
            "--frame",
            "0",
            "--out",
            "comp.png",
            "--width",
            "64",
            "--height",
            "64",
        ])
        .expect("render --comp must parse");
        assert!(matches!(
            cli.command,
            Commands::Render {
                comp: Some(ref c),
                ..
            } if c == "Precomp 1"
        ));

        let exp = Cli::try_parse_from([
            "zoidcraft-cli",
            "export",
            "--comp",
            "Precomp 1",
            "--out",
            "comp.mp4",
        ])
        .expect("export --comp must parse");
        assert!(matches!(
            exp.command,
            Commands::Export {
                comp: Some(ref c),
                ..
            } if c == "Precomp 1"
        ));

        // Build a project with a black root sequence and a bright red child composition `Hero Comp`.
        let mut proj = Project::new_cm3_default();
        proj.sequence.resolution = [64, 64];
        proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
        proj.sequence.video_tracks.clear();

        let child_id = Uuid::new_v4();
        let child_seq = zc_project::Sequence {
            id: child_id,
            name: "Hero Comp".to_owned(),
            resolution: [64, 64],
            fps: 30,
            length_frames: 30.0,
            background_color: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        };
        proj.compositions.insert(child_id, child_seq);

        // Selection by case-insensitive name & by UUID must promote child sequence.
        let mut by_name = proj.clone();
        select_composition_for_render(&mut by_name, "hero comp").expect("select by name");
        assert_eq!(by_name.sequence.id, child_id);
        assert_eq!(by_name.sequence.background_color, [1.0, 0.0, 0.0, 1.0]);

        let mut by_uuid = proj.clone();
        select_composition_for_render(&mut by_uuid, &child_id.to_string()).expect("select by uuid");
        assert_eq!(by_uuid.sequence.id, child_id);

        let mut missing = proj.clone();
        assert!(select_composition_for_render(&mut missing, "DoesNotExist").is_err());

        // End-to-end render test (when headless GPU is available).
        if ProjectRenderEngine::new_headless().is_ok() {
            let proj_path =
                std::env::temp_dir().join(format!("zc_cli_comp_{}.zcraft", Uuid::new_v4()));
            let out_path = std::env::temp_dir().join(format!("zc_cli_comp_{}.png", Uuid::new_v4()));
            save_zcraft(&proj, &proj_path).expect("save project with child comp");

            cmd_render(
                Some(&proj_path),
                Some("hero comp"),
                0.0,
                &out_path,
                Some(64),
                Some(64),
                false,
            )
            .expect("render child composition via --comp");

            let img = image::open(&out_path)
                .expect("open rendered child PNG")
                .to_rgba8();
            let px = img.get_pixel(32, 32);
            assert!(
                px[0] > 200 && px[1] < 20 && px[2] < 20,
                "expected red child composition background, got rgba={px:?}"
            );
            let _ = fs::remove_file(&proj_path);
            let _ = fs::remove_file(&out_path);
        }
    }

    #[test]
    fn cli_export_invalid_settings_error_has_no_duplicate_prefix() {
        let out = std::env::temp_dir().join(format!("zc_cli_bad_exp_{}.mp4", Uuid::new_v4()));
        let err = cmd_export(ExportCliArgs {
            project: None,
            comp: None,
            out: &out,
            format: Some("mp4"),
            video_codec: Some("h264"),
            audio_codec: Some("none"),
            crf: 18,
            bitrate_kbps: None,
            audio_bitrate_kbps: 192,
            sample_rate: STANDARD_SAMPLE_RATE,
            width: Some(0),
            height: Some(64),
            fps: Some(30.0),
            start_frame: 0,
            end_frame: Some(5),
            transparent: false,
            ffmpeg: None,
        })
        .expect_err("zero width export must fail validation");
        assert!(
            !err.contains("invalid export settings: invalid export settings:"),
            "error message must not duplicate prefix: {err}"
        );
    }

    #[test]
    fn qa6_ntsc_fractional_fps_export_timing_zero_drift() {
        // Case 1: 30fps source exported at --fps 29.97 and --fps 23.976
        let seq_30 = zc_project::Sequence {
            fps: 30,
            exact_fps: None,
            length_frames: 300.0, // 10.0s
            ..zc_project::Sequence::default()
        };
        for (fps_arg, expected_rational, expected_frames) in [
            (29.97, 30_000.0 / 1001.0, 300u32),
            (23.976, 24_000.0 / 1001.0, 240u32),
        ] {
            let t = compute_export_timing(&seq_30, 0, 300, Some(fps_arg)).expect("timing");
            assert!((t.exact_export_fps - expected_rational).abs() < 1e-12);
            assert_eq!(t.export_frame_count, expected_frames);
            // Video duration and audio duration must match identically (0.0ms drift)
            let video_dur = f64::from(t.export_frame_count) / t.exact_export_fps;
            let audio_dur = t.audio_seq_frames / t.src_fps;
            assert!((video_dur - audio_dur).abs() < 1e-12);

            // At 1.0s, 5.0s, 9.0s: sampled project time == video PTS with 0.0ms cumulative drift
            for target_sec in [1.0, 5.0, 9.0] {
                let n = (target_sec * t.exact_export_fps).round() as u32;
                let video_pts = f64::from(n) / t.exact_export_fps;
                let scene_frame = t.scene_frame_for_output(0, n);
                let sampled_project_time = scene_frame / t.src_fps;
                assert!(
                    (video_pts - sampled_project_time).abs() < 1e-12,
                    "fps={fps_arg} t={target_sec}s: drift={:.6}ms",
                    (video_pts - sampled_project_time).abs() * 1000.0
                );
            }
        }

        // Case 2: Sequence with `exact_fps = Some(29.97)` exported without `--fps` or with `--fps 29.97`
        let seq_ntsc = zc_project::Sequence {
            fps: 30,
            exact_fps: Some(30_000.0 / 1001.0),
            length_frames: 300.0,
            ..zc_project::Sequence::default()
        };
        for override_opt in [None, Some(29.97)] {
            let t = compute_export_timing(&seq_ntsc, 0, 300, override_opt).expect("ntsc timing");
            assert_eq!(t.export_frame_count, 300);
            for target_sec in [1.0, 5.0, 9.0] {
                let n = (target_sec * t.exact_export_fps).round() as u32;
                let video_pts = f64::from(n) / t.exact_export_fps;
                let sampled_time = t.scene_frame_for_output(0, n) / t.src_fps;
                assert!((video_pts - sampled_time).abs() < 1e-12);
            }
        }
    }
}
