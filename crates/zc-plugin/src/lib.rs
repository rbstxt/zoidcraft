//! L1 Zoidium-compatible toggleable native plugin registry and lifecycle contracts.

/// Returns the crate identifier for workspace smoke verification.
pub const fn crate_name() -> &'static str {
    "zc-plugin"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_crate_name() {
        assert_eq!(crate_name(), "zc-plugin");
    }
}
