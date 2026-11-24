#![forbid(unsafe_code)]

//! ZoidCraft headless CLI, batch renderer, `.pz` converter, effect self-test runner,
//! and JSON-Lines session control loop (L5 headless — zero GUI dependencies).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use image::RgbaImage;
use zc_plugin::PluginRegistry;
use zc_project::{Project, import_pz, load_zcraft, save_zcraft};
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
    }
}
