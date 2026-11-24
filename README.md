# ZoidCraft

ZoidCraft is a clean-room, native Rust desktop and CLI motion-graphics editor reimplementing Panzoid ClipMaker 3 (CM3) with an Engine-First architecture and toggleable Zoidium-style extensions.

## Highlights

- **Engine-First layer isolation (`L0`–`L5`)**: Core crates under `crates/*` contain zero GUI or windowing dependencies (`egui`, `eframe`, `winit`), enforced in CI via `cargo run --locked -p xtask -- layers`.
- **ClipMaker 3 workspace (`zoidcraft`)**: Native `eframe` + `egui-wgpu` desktop application with zero-copy GPU viewport composition, Elevator rail navigation, Scene / Objects / Effects / Timeline / Graph Editor / Export / Plugins / Settings panels, and CM3 keyboard shortcuts.
- **Toggleable native plugins & Vanilla CM3 mode**: Built-in extensions (Easing+, Expressions+, Shader+, Player+, Precomp+, Light+, Geometry+, Effector, Camera+, 12 GPU shader plugins, and more) can be toggled individually or disabled in one click for Vanilla ClipMaker 3 behavior while preserving project data.
- **GPU rendering & custom shaders (`zc-render`, `zc-shader`)**: `wgpu` 3D PBR/shadow/particle renderer, 43-blend-mode 2D compositor, 44 standard CM3 effects, 12 GPU plugin effects, and WebGL 1 / GLSL ES 1.00 custom shader compilation via `naga`.
- **Audio & FFmpeg export (`zc-audio`, `zc-export`)**: Symphonia decoding, zero-copy native-rate stereo mixing, `cpal` playback synced to a sample-accurate master clock, PNG sequence export, and FFmpeg video export (MP4/H.264/H.265/AV1, WebM/VP9/VP8 with alpha, MKV).
- **Headless CLI (`zoidcraft-cli`)**: Render single frames, export video/PNG sequences, import `.pz` projects, validate `.zcraft` archives, run GPU effect self-tests, and drive sessions over JSON-Lines stdin/stdout.

## Workspace layout

| Layer | Crates | Responsibility |
| --- | --- | --- |
| L0 Foundation | `zc-time`, `zc-geom`, `zc-color` | Rational ticks, 3D math, color spaces |
| L1 Evaluation | `zc-anim`, `zc-expr`, `zc-plugin` | Keyframes/Bezier/easing, deterministic expressions, plugin catalog |
| L2 Document & Geometry | `zc-project`, `zc-scene`, `zc-shader` | `.zcraft` / `.pz` schema, 3D/text/particle meshes, GLSL->WGSL compiler |
| L3 Media & GPU | `zc-audio`, `zc-export`, `zc-render` | Audio decoder/mixer/clock, FFmpeg/PNG export, `wgpu` renderer |
| L4 Session | `zc-session` | Commands, bounded undo/redo transactions, selection, JSON-Lines control |
| L5 Apps | `apps/zoidcraft`, `apps/zoidcraft-cli`, `xtask` | Desktop GUI, headless CLI, architecture layer checker |

## Requirements

- Rust stable toolchain (Edition 2024)
- On Linux: `libasound2-dev`, `libwayland-dev`, `libxkbcommon-dev`, `libgtk-3-dev`, `pkg-config`
- Optional for video export (MP4/WebM/MKV): `ffmpeg` available on `PATH` (PNG sequence export works without `ffmpeg`)

## Building and testing

```bash
# Format, check architecture layer isolation, and lint
cargo fmt --all -- --check
cargo run --locked -p xtask -- layers
cargo clippy --locked --workspace --all-targets -- -D warnings

# Run all workspace unit and integration tests (require a GPU or software wgpu adapter)
ZOIDCRAFT_REQUIRE_GPU=1 cargo test --locked --workspace

# Launch the desktop application
cargo run --release --bin zoidcraft
```

## CLI usage (`zoidcraft-cli`)

```bash
# Render frame 0 of the default CM3 scene to PNG
zoidcraft-cli render --frame 0 --out frame.png

# Export a 2-second 1280x720 MP4 (requires ffmpeg on PATH)
zoidcraft-cli export --out out.mp4 --width 1280 --height 720 --fps 30 --start-frame 0 --end-frame 60

# Import a ClipMaker 3 .pz archive to .zcraft and validate it
zoidcraft-cli import-pz input.pz --out project.zcraft
zoidcraft-cli validate project.zcraft
zoidcraft-cli info project.zcraft

# Run headless GPU verification across all 44 standard + 12 plugin effects
zoidcraft-cli selftest-effects --width 128 --height 128

# Start the JSON-Lines session control loop
zoidcraft-cli control
```

## Licensing & third-party notices

ZoidCraft is authored from clean-room specifications and licensed under the MIT License (`LICENSE`). Bundled third-party assets (including Geist Mono under the SIL Open Font License 1.1) are documented in `THIRD_PARTY_LICENSES.md`.
