//! L3 deterministic offline FFmpeg video/audio export pipeline.

/// Returns the crate identifier for workspace smoke verification.
pub const fn crate_name() -> &'static str {
    "zc-export"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_crate_name() {
        assert_eq!(crate_name(), "zc-export");
    }
}
