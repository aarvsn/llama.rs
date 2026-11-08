//! # llama
//!
//! Facade crate for the llama.rs inference engine.
//!
//! It re-exports the primary crates so users can depend on `llama` alone:
//!
//! ```no_run
//! use llama::llama_runtime::{Engine, GenerationConfig};
//!
//! # fn main() -> Result<(), llama::llama_core::LlamaError> {
//! let engine = Engine::load("model.gguf")?;
//! let output = engine.generate(
//!     "Explain how Rust works.",
//!     &GenerationConfig::default().max_tokens(128).temperature(0.7),
//! )?;
//! println!("{}", output.text);
//! # Ok(())
//! # }
//! ```

pub use llama_core;
pub use llama_gguf;
pub use llama_kernels;
pub use llama_model;
pub use llama_runtime;
pub use llama_sampling;
pub use llama_tensor;
pub use llama_tokenizer;
