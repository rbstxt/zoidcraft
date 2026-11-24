//! L3 sample-accurate audio decoding, offline mixing, and realtime playback clock.

/// Returns the crate identifier for workspace smoke verification.
pub const fn crate_name() -> &'static str {
    "zc-audio"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_crate_name() {
        assert_eq!(crate_name(), "zc-audio");
    }
}
