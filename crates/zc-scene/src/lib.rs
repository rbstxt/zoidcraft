//! L2 3D scene graph, procedural meshes, 3D text extrusion, lights, cameras, particles, and 2D layers.

/// Returns the crate identifier for workspace smoke verification.
pub const fn crate_name() -> &'static str {
    "zc-scene"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_crate_name() {
        assert_eq!(crate_name(), "zc-scene");
    }
}
