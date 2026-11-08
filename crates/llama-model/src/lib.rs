//! # llama-model
//!
//! Model architecture definitions and the Llama-family decoder-only
//! transformer:
//!
//! * [`Architecture`] — architecture detection from `general.architecture`.
//! * [`LlamaConfig`] — hyper-parameters validated against GGUF metadata.
//! * [`LlamaModel`] — zero-copy weight views plus the forward pass
//!   (embeddings, `RMSNorm`, `RoPE`, GQA attention with a KV cache, `SwiGLU` FFN).
//!
//! Architectures other than `llama` are detected and rejected with explicit
//! errors until they are implemented.

pub mod arch;
pub mod config;
pub mod kv;
pub mod llama;

pub use arch::Architecture;
pub use config::LlamaConfig;
pub use kv::{KvCache, KvPrecision};
pub use llama::{ForwardScratch, LlamaModel};
