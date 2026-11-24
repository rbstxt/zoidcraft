//! L1 deterministic sandboxed expression parser and evaluator.

/// Returns the crate identifier for workspace smoke verification.
pub const fn crate_name() -> &'static str {
    "zc-expr"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_crate_name() {
        assert_eq!(crate_name(), "zc-expr");
    }
}
