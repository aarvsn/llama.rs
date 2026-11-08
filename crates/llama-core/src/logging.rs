//! Logging and diagnostics helpers built on `tracing`.

use std::sync::OnceLock;
use tracing_subscriber::EnvFilter;

static INIT: OnceLock<()> = OnceLock::new();

/// Initializes the global `tracing` subscriber exactly once.
///
/// The filter level is taken from the `LLAMA_LOG` environment variable and
/// defaults to `info`. Callers may invoke this from every binary; only the
/// first call has an effect.
pub fn init_tracing() {
    INIT.get_or_init(|| {
        let filter =
            EnvFilter::try_from_env("LLAMA_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(false)
            .with_thread_ids(false)
            // Logs go to stderr so piped stdout stays clean generated text.
            .with_writer(std::io::stderr)
            .try_init();
    });
}

/// Initializes tracing for tests ( quieter by default).
pub fn init_tracing_for_tests() {
    INIT.get_or_init(|| {
        let filter =
            EnvFilter::try_from_env("LLAMA_LOG").unwrap_or_else(|_| EnvFilter::new("warn"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    });
}
