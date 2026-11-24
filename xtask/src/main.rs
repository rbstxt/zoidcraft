use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::process::{Command, ExitCode};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Metadata {
    packages: Vec<Package>,
    workspace_members: Vec<String>,
    resolve: Option<Resolve>,
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
}

#[derive(Debug, Deserialize)]
struct Resolve {
    nodes: Vec<ResolveNode>,
}

#[derive(Debug, Deserialize)]
struct ResolveNode {
    id: String,
    deps: Vec<NodeDep>,
}

#[derive(Debug, Deserialize)]
struct NodeDep {
    pkg: String,
    dep_kinds: Vec<DepKindInfo>,
}

#[derive(Debug, Deserialize)]
struct DepKindInfo {
    kind: Option<String>,
}

/// Strictly enforced downward-only layer table inspired by ArtCraft / EffectCraft.
/// Core crates (L0..L4) and the headless CLI may never depend (directly OR transitively)
/// on GUI/windowing/OS-dialog crates.
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
        _ => None,
    }
}

const FORBIDDEN_IN_CORE: &[&str] = &[
    "egui",
    "eframe",
    "egui-wgpu",
    "egui_wgpu",
    "egui-winit",
    "epaint",
    "winit",
    "rfd",
    "muda",
    "accesskit",
];

fn check_layers() -> Result<(), Vec<String>> {
    // Full resolved metadata with --all-features and --locked so non-default
    // features and transitive dependencies are checked against the committed lockfile.
    let output = Command::new("cargo")
        .args([
            "metadata",
            "--locked",
            "--all-features",
            "--format-version",
            "1",
        ])
        .output()
        .map_err(|e| vec![format!("failed to execute cargo metadata: {e}")])?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(vec![format!("cargo metadata failed:\n{stderr}")]);
    }

    let meta: Metadata = serde_json::from_slice(&output.stdout)
        .map_err(|e| vec![format!("failed to parse cargo metadata: {e}")])?;

    let resolve = meta
        .resolve
        .ok_or_else(|| vec!["cargo metadata returned no resolve graph".to_string()])?;

    let ws_ids: BTreeSet<&str> = meta.workspace_members.iter().map(String::as_str).collect();
    let id_to_name: BTreeMap<&str, &str> = meta
        .packages
        .iter()
        .map(|p| (p.id.as_str(), p.name.as_str()))
        .collect();
    let node_map: BTreeMap<&str, &ResolveNode> =
        resolve.nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    let mut violations = Vec::new();
    let mut seen_layers: BTreeMap<u8, Vec<String>> = BTreeMap::new();

    for pkg in &meta.packages {
        if !ws_ids.contains(pkg.id.as_str()) {
            continue;
        }
        if pkg.name == "xtask" {
            continue;
        }
        let Some(pkg_layer) = layer_of(&pkg.name) else {
            violations.push(format!(
                "workspace crate `{}` is not registered in xtask layer table",
                pkg.name
            ));
            continue;
        };
        seen_layers
            .entry(pkg_layer)
            .or_default()
            .push(pkg.name.clone());

        // 1. Direct dependency checks (including optional, dev, and xtask bridge guard).
        let forbid_gui = pkg_layer <= 4 || pkg.name == "zoidcraft-cli";
        for dep in &pkg.dependencies {
            if dep.name == "xtask" {
                violations.push(format!(
                    "workspace crate `{}` must not depend on `xtask`",
                    pkg.name
                ));
                continue;
            }
            if forbid_gui && FORBIDDEN_IN_CORE.contains(&dep.name.as_str()) {
                violations.push(format!(
                    "forbidden direct GUI/windowing dependency in `{}` (L{}): `{}`",
                    pkg.name, pkg_layer, dep.name
                ));
            }
            if let Some(dep_layer) = layer_of(&dep.name) {
                let allowed = if pkg_layer == 2 && dep_layer == 2 {
                    dep.name == "zc-project" && pkg.name != "zc-project"
                } else if pkg_layer == 3 && dep_layer == 3 {
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

        // 2. Full transitive dependency walk for all core crates (L0..L4) and headless CLI.
        if forbid_gui {
            let mut visited: BTreeSet<&str> = BTreeSet::new();
            let mut queue: VecDeque<(&str, Vec<String>)> = VecDeque::new();
            visited.insert(pkg.id.as_str());
            queue.push_back((pkg.id.as_str(), vec![pkg.name.clone()]));

            while let Some((cur_id, path)) = queue.pop_front() {
                let Some(node) = node_map.get(cur_id) else {
                    continue;
                };
                for dep in &node.deps {
                    // Only follow runtime/build dependencies for non-root nodes; for root check all kinds including dev.
                    let is_root = cur_id == pkg.id.as_str();
                    let is_dev_only = !dep.dep_kinds.is_empty()
                        && dep
                            .dep_kinds
                            .iter()
                            .all(|k| k.kind.as_deref() == Some("dev"));
                    if !is_root && is_dev_only {
                        continue;
                    }
                    let dep_name = id_to_name.get(dep.pkg.as_str()).copied().unwrap_or("?");
                    if FORBIDDEN_IN_CORE.contains(&dep_name) {
                        let mut chain = path.clone();
                        chain.push(dep_name.to_string());
                        violations.push(format!(
                            "forbidden GUI/windowing dependency in `{}` (L{}): {}",
                            pkg.name,
                            pkg_layer,
                            chain.join(" -> ")
                        ));
                    }
                    if visited.insert(dep.pkg.as_str()) {
                        let mut next_path = path.clone();
                        next_path.push(dep_name.to_string());
                        queue.push_back((dep.pkg.as_str(), next_path));
                    }
                }
            }
        }
    }

    if violations.is_empty() {
        println!("Layer & transitive dependency check PASSED:");
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
            eprintln!("Usage: cargo xtask <command>");
            eprintln!("Commands:");
            eprintln!(
                "  layers    Verify strict Engine-First crate layer & transitive GUI boundaries"
            );
            ExitCode::FAILURE
        }
    }
}
