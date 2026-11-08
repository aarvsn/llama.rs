//! Llama-family hyper-parameters, read and validated from GGUF metadata.

use llama_core::{LlamaError, Result};
use llama_gguf::GgufMetadata;

use crate::arch::Architecture;

/// Llama transformer hyper-parameters.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Model context length (training context).
    pub n_ctx_train: u64,
    /// Embedding / hidden dimension.
    pub n_embd: usize,
    /// Number of transformer blocks (layers).
    pub n_layers: usize,
    /// Feed-forward hidden dimension.
    pub n_ff: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of key/value heads (GQA; may be smaller than `n_heads`).
    pub n_kv_heads: usize,
    /// Per-head key dimension.
    pub head_dim_k: usize,
    /// Per-head value dimension.
    pub head_dim_v: usize,
    /// Number of rotary dimensions per head.
    pub n_rot: usize,
    /// `RoPE` frequency base.
    pub rope_freq_base: f32,
    /// `RMSNorm` epsilon.
    pub rms_eps: f32,
    /// Vocabulary size (filled in during weight loading).
    pub n_vocab: usize,
}

impl LlamaConfig {
    /// Reads and validates the configuration from GGUF metadata.
    ///
    /// Defaults follow llama.cpp (`key_length`/`value_length` fall back to
    /// the head dimension, `head_count_kv` to `head_count`, rope base to
    /// 10000, `RMSNorm` eps to 1e-5).
    pub fn from_gguf(meta: &GgufMetadata) -> Result<Self> {
        let arch = Architecture::detect(meta)?;
        debug_assert_eq!(arch, Architecture::Llama);
        let prefix = "llama.";

        let n_ctx_train = meta.get_u64_or(&format!("{prefix}context_length"), 2048);
        let n_embd = meta.get_u64(&format!("{prefix}embedding_length"))? as usize;
        let n_layers = meta.get_u64(&format!("{prefix}block_count"))? as usize;
        let n_ff = meta.get_u64(&format!("{prefix}feed_forward_length"))? as usize;
        let n_heads = meta.get_u64(&format!("{prefix}attention.head_count"))? as usize;
        let n_kv_heads =
            meta.get_u64_or(&format!("{prefix}attention.head_count_kv"), n_heads as u64) as usize;
        let head_dim = n_embd / n_heads;
        let head_dim_k =
            meta.get_u64_or(&format!("{prefix}attention.key_length"), head_dim as u64) as usize;
        let head_dim_v = meta.get_u64_or(
            &format!("{prefix}attention.value_length"),
            head_dim_k as u64,
        ) as usize;
        let n_rot =
            meta.get_u64_or(&format!("{prefix}rope.dimension_count"), head_dim_k as u64) as usize;
        let rope_freq_base = meta.get_f32_or(&format!("{prefix}rope.freq_base"), 10_000.0);
        let rms_eps = meta.get_f32_or(&format!("{prefix}attention.layer_norm_rms_epsilon"), 1e-5);

        Self::validate(LlamaConfig {
            n_ctx_train,
            n_embd,
            n_layers,
            n_ff,
            n_heads,
            n_kv_heads,
            head_dim_k,
            head_dim_v,
            n_rot,
            rope_freq_base,
            rms_eps,
            n_vocab: 0,
        })
    }

    fn validate(cfg: LlamaConfig) -> Result<Self> {
        let problems = [
            (cfg.n_embd == 0, "embedding_length is zero"),
            (
                cfg.n_embd % cfg.n_heads != 0,
                "embedding_length is not divisible by head_count",
            ),
            (
                cfg.n_heads % cfg.n_kv_heads != 0,
                "head_count is not divisible by head_count_kv (invalid GQA)",
            ),
            (cfg.n_ff == 0, "feed_forward_length is zero"),
            (cfg.n_rot % 2 != 0, "rope.dimension_count must be even"),
            (
                cfg.n_rot > cfg.head_dim_k,
                "rope.dimension_count exceeds the key head dimension",
            ),
            (cfg.n_layers == 0, "block_count is zero"),
            (
                cfg.rms_eps <= 0.0,
                "layer_norm_rms_epsilon must be positive",
            ),
            (cfg.rope_freq_base <= 0.0, "rope.freq_base must be positive"),
        ];
        for (bad, msg) in problems {
            if bad {
                return Err(LlamaError::InvalidMetadata(format!(
                    "invalid llama configuration: {msg}"
                )));
            }
        }
        Ok(cfg)
    }

    /// GQA repetition factor: query heads per KV head.
    #[must_use]
    pub fn n_gqa(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Human-readable summary.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "n_embd={} n_layers={} n_heads={} n_kv_heads={} head_dim={} n_ff={} n_ctx_train={} n_rot={} rope_base={}",
            self.n_embd,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim_k,
            self.n_ff,
            self.n_ctx_train,
            self.n_rot,
            self.rope_freq_base
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_gguf::metadata::{GgufMetadata, MetadataValue};

    fn meta_with(entries: &[(&str, MetadataValue)]) -> GgufMetadata {
        let mut m = GgufMetadata::new();
        m.insert(
            "general.architecture".into(),
            MetadataValue::String("llama".into()),
        );
        for (k, v) in entries {
            m.insert((*k).to_string(), v.clone());
        }
        m
    }

    #[test]
    fn parses_full_config() {
        let meta = meta_with(&[
            ("llama.context_length", MetadataValue::U32(4096)),
            ("llama.embedding_length", MetadataValue::U32(512)),
            ("llama.block_count", MetadataValue::U32(4)),
            ("llama.feed_forward_length", MetadataValue::U32(1408)),
            ("llama.attention.head_count", MetadataValue::U32(8)),
            ("llama.attention.head_count_kv", MetadataValue::U32(2)),
            ("llama.rope.freq_base", MetadataValue::F32(500_000.0)),
        ]);
        let cfg = LlamaConfig::from_gguf(&meta).unwrap();
        assert_eq!(cfg.n_embd, 512);
        assert_eq!(cfg.head_dim_k, 64);
        assert_eq!(cfg.head_dim_v, 64);
        assert_eq!(cfg.n_rot, 64);
        assert_eq!(cfg.n_gqa(), 4);
        assert_eq!(cfg.rope_freq_base, 500_000.0);
    }

    #[test]
    fn defaults_apply() {
        let meta = meta_with(&[
            ("llama.embedding_length", MetadataValue::U32(256)),
            ("llama.block_count", MetadataValue::U32(2)),
            ("llama.feed_forward_length", MetadataValue::U32(704)),
            ("llama.attention.head_count", MetadataValue::U32(4)),
        ]);
        let cfg = LlamaConfig::from_gguf(&meta).unwrap();
        assert_eq!(cfg.n_kv_heads, 4);
        assert_eq!(cfg.rope_freq_base, 10_000.0);
        assert_eq!(cfg.n_ctx_train, 2048);
    }

    #[test]
    fn invalid_gqa_rejected() {
        let meta = meta_with(&[
            ("llama.embedding_length", MetadataValue::U32(256)),
            ("llama.block_count", MetadataValue::U32(2)),
            ("llama.feed_forward_length", MetadataValue::U32(704)),
            ("llama.attention.head_count", MetadataValue::U32(4)),
            ("llama.attention.head_count_kv", MetadataValue::U32(3)),
        ]);
        assert!(LlamaConfig::from_gguf(&meta).is_err());
    }
}
