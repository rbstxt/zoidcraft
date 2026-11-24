//! L3 wgpu 3D renderer, 2D compositor, 44 standard effects, and bounded frame cache.

/// Returns the crate identifier for workspace smoke verification.
pub const fn crate_name() -> &'static str {
    "zc-render"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_crate_name() {
        assert_eq!(crate_name(), "zc-render");
    }
}
