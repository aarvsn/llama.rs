//! Performance and memory statistics.

use std::time::Duration;

/// A measured time span.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timing {
    /// Wall-clock duration.
    pub duration: Duration,
    /// Number of items processed in that duration.
    pub count: u64,
}

impl Timing {
    /// Items per second (0 when nothing was measured).
    #[must_use]
    pub fn per_second(&self) -> f64 {
        if self.duration.as_secs_f64() > 0.0 {
            f64::from(self.count as u32) / self.duration.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// Engine-level statistics.
#[derive(Debug, Clone)]
pub struct EngineStats {
    /// Time spent loading and validating the model.
    pub load_time: Duration,
    /// Serialized tensor bytes of the model.
    pub model_bytes: u64,
    /// GGUF file size.
    pub file_bytes: u64,
}

/// Session-level performance counters (prefill vs decode).
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// Prompt tokens processed and the time it took.
    pub prefill: Timing,
    /// Generated tokens and the time it took.
    pub decode: Timing,
    /// Number of times the KV cache was reused via prefix matching.
    pub cache_reuses: u64,
    /// Tokens currently in the session context.
    pub context_tokens: usize,
    /// Context window capacity.
    pub context_capacity: usize,
    /// KV cache memory in bytes.
    pub kv_cache_bytes: usize,
}

impl SessionStats {
    /// Prompt processing speed in tokens/s.
    #[must_use]
    pub fn prefill_tps(&self) -> f64 {
        self.prefill.per_second()
    }

    /// Generation speed in tokens/s.
    #[must_use]
    pub fn decode_tps(&self) -> f64 {
        self.decode.per_second()
    }
}
