//! Engine-wide configuration options.

use std::num::NonZeroUsize;

/// Default number of threads used when the caller does not specify one.
pub const DEFAULT_THREADS: usize = 4;

/// Default context window (tokens) allocated for the KV cache.
pub const DEFAULT_CONTEXT_SIZE: usize = 2048;

/// Runtime options for loading and running a model.
///
/// # Examples
///
/// ```
/// use llama_core::EngineOptions;
///
/// let opts = EngineOptions::default()
///     .with_threads(8)
///     .with_context_size(4096);
/// assert_eq!(opts.threads.get(), 8);
/// assert_eq!(opts.context_size, 4096);
/// ```
#[derive(Debug, Clone)]
pub struct EngineOptions {
    /// Number of worker threads for CPU compute. Defaults to the number of
    /// logical CPUs, capped at 16.
    pub threads: NonZeroUsize,
    /// Maximum context window (KV cache capacity) in tokens.
    pub context_size: usize,
    /// Store K/V cache entries as f16 instead of f32 (halves cache memory).
    pub kv_cache_f16: bool,
    /// Print load timing information via `tracing` at INFO level.
    pub log_load_info: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(DEFAULT_THREADS)
            .min(16);
        Self {
            threads: NonZeroUsize::new(cpus.max(1))
                .unwrap_or_else(|| NonZeroUsize::new(1).unwrap()),
            context_size: DEFAULT_CONTEXT_SIZE,
            kv_cache_f16: false,
            log_load_info: true,
        }
    }
}

impl EngineOptions {
    /// Builder: sets the number of compute threads.
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads =
            NonZeroUsize::new(threads.max(1)).unwrap_or_else(|| NonZeroUsize::new(1).unwrap());
        self
    }

    /// Builder: sets the context window size in tokens.
    #[must_use]
    pub fn with_context_size(mut self, ctx: usize) -> Self {
        self.context_size = ctx.max(16);
        self
    }

    /// Builder: enables or disables the f16 KV cache.
    #[must_use]
    pub fn with_kv_cache_f16(mut self, enabled: bool) -> Self {
        self.kv_cache_f16 = enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let d = EngineOptions::default();
        assert!(d.threads.get() >= 1);
        assert!(d.context_size >= 16);
    }

    #[test]
    fn builder_clamps() {
        let opts = EngineOptions::default()
            .with_threads(0)
            .with_context_size(1);
        assert_eq!(opts.threads.get(), 1);
        assert_eq!(opts.context_size, 16);
    }
}
