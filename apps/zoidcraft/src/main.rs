#![forbid(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    zoidcraft::run_desktop()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
