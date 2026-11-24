use std::collections::{BTreeMap, BTreeSet};
use std::process::{Command, ExitCode};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Metadata {
    packages: Vec<Package>,
    workspace_members: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Package {
    id: String,
    name: String,
    dependencies: Vec<Dependency>,
}

#[derive(Debug, Deserialize)]
struct Dependency {
    name: String,
    kind: Option<String>,
}

/// Strictly enforced downward-only layer table inspired by ArtCraft / EffectCraft.
/// Crates may only depend on workspace crates in strictly lower layers,
/// and core crates (L0..L4) may never depend on UI/windowing/OS-dialog crates.
fn layer_of(name: &str) -> Option<u8> {
    match name {
        // L0: math, time, color primitives
        "zc-time" | "zc-geom" | "zc-color" => Some(0),
        // L1: animation, expressions, plugin contracts
        "zc-anim" | "zc-expr" | "zc-plugin" => Some(1),
        // L2: project model, 3D scene, shader translator
        "zc-project" | "zc-scene" | "zc-shader" => Some(2),
        // L3: GPU renderer, audio engine, FFmpeg exporter
        "zc-render" | "zc-audio" | "zc-export" => Some(3),
        // L4: session, undo/redo, command registry, headless control
        "zc-session" => Some(4),
        // L5: end-user applications
        "zoidcraft" | "zoidcraft-cli" => Some(5),
        "xtask" => None,
        _ => None,
    }
}

const FORBIDDEN_IN_CORE: &[&str] = &[
    "egui",
    "eframe",
    "egui-wgpu",
    "egui_wgpu",
    "winit",
    "rfd",
    "muda",
];

fn check_layers() -> Result<(), Vec<String>> {
    let output = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .output()
        .map_err(|e| vec![format!("failed to execute cargo metadata: {e}")])?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(vec![format!("cargo metadata failed:\n{stderr}")]);
    }

    let meta: Metadata = serde_json::from_slice(&output.stdout)
        .map_err(|e| vec![format!("failed to parse cargo metadata: {e}")])?;

    let ws_ids: BTreeSet<&str> = meta.workspace_members.iter().map(String::as_str).collect();
    let mut violations = Vec::new();
    let mut seen_layers: BTreeMap<u8, Vec<String>> = BTreeMap::new();

    for pkg in &meta.packages {
        if !ws_ids.contains(pkg.id.as_str()) {
            continue;
        }
        let Some(pkg_layer) = layer_of(&pkg.name) else {
            if pkg.name != "xtask" {
                violations.push(format!(
                    "workspace crate `{}` is not registered in xtask layer table",
                    pkg.name
                ));
            }
            continue;
        };
        seen_layers
            .entry(pkg_layer)
            .or_default()
            .push(pkg.name.clone());

        for dep in &pkg.dependencies {
            // Dev-dependencies are excluded from runtime layer direction only if explicitly needed,
            // but UI crates are forbidden in core crates even as dev-dependencies.
            let is_dev = dep.kind.as_deref() == Some("dev");

            if pkg_layer <= 4 && FORBIDDEN_IN_CORE.contains(&dep.name.as_str()) {
                violations.push(format!(
                    "core crate `{}` (L{}) must not depend on UI/windowing crate `{}`",
                    pkg.name, pkg_layer, dep.name
                ));
            }

            if pkg.name == "zoidcraft-cli" && FORBIDDEN_IN_CORE.contains(&dep.name.as_str()) {
                violations.push(format!(
                    "headless CLI `{}` must not depend on GUI crate `{}`",
                    pkg.name, dep.name
                ));
            }

            if !is_dev && let Some(dep_layer) = layer_of(&dep.name) {
                // Allow same-layer only within L2 where scene/project/shader share clean data models?
                // Keep it strict: strictly downward (dep_layer < pkg_layer),
                // except L2 (zc-project may depend on nothing in L2; zc-scene & zc-shader may depend on zc-project).
                let allowed = if pkg_layer == 2 && dep_layer == 2 {
                    dep.name == "zc-project" && pkg.name != "zc-project"
                } else if pkg_layer == 3 && dep_layer == 3 {
                    // zc-export may depend on zc-render and zc-audio
                    pkg.name == "zc-export" && matches!(dep.name.as_str(), "zc-render" | "zc-audio")
                } else {
                    dep_layer < pkg_layer
                };

                if !allowed {
                    violations.push(format!(
                        "invalid layer dependency: `{}` (L{}) -> `{}` (L{})",
                        pkg.name, pkg_layer, dep.name, dep_layer
                    ));
                }
            }
        }
    }

    if violations.is_empty() {
        println!("Layer check PASSED:");
        for (layer, mut crates) in seen_layers {
            crates.sort();
            println!("  L{layer}: {}", crates.join(", "));
        }
        Ok(())
    } else {
        Err(violations)
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("layers") => match check_layers() {
            Ok(()) => ExitCode::SUCCESS,
            Err(errors) => {
                eprintln!("Layer check FAILED ({} violation(s)):", errors.len());
                for err in errors {
                    eprintln!("  - {err}");
                }
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("Usage: cargo run -p xtask -- <command>");
            eprintln!("Commands:");
            eprintln!("  layers    Verify strict Engine-First crate layer boundaries");
            ExitCode::FAILURE
        }
    }
}
