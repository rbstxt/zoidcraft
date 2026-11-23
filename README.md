# ZoidCraft

Clean-room native Rust reimplementation of Panzoid ClipMaker 3, combining pixel-faithful ClipMaker 3 UI and interaction design with an Engine-First Rust architecture inspired by ArtCraft (`EffectCraft` / `PhotoCraft`) and toggleable Zoidium plugins.

## Design principles

- **Engine-First layer separation**: Core crates (`crates/*`) have zero dependency on UI or windowing crates (`egui`, `eframe`, `winit`), enforced in CI by `cargo xtask layers`.
- **Pixel- and interaction-faithful CM3 UI**: Built natively on `egui` + `egui-wgpu` with zero-copy GPU viewport composition.
- **Clean-room & copyright-safe**: All Rust code, WGSL shaders, procedural textures, and presets are authored from clean specifications.
- **Lightweight & leak-free**: Bounded GPU/frame caches, deterministic offline audio/video rendering, and explicit resource lifecycle management.
- **FFmpeg export**: Pipes raw RGBA frames and PCM audio to an external `ffmpeg` process.
- **Toggleable Zoidium plugins**: Native plugin registry with enable/disable toggles, in-use locks, and preserved data for disabled plugins.

## Building

```bash
cargo check --workspace
cargo test --workspace
cargo xtask layers
```
