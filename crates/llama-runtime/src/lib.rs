//! # llama-runtime
//!
//! Inference orchestration: loading, sessions, generation and statistics.
//!
//! ```no_run
//! use llama_runtime::{Engine, GenerationConfig};
//!
//! # fn main() -> Result<(), llama_core::LlamaError> {
//! let engine = Engine::load("model.gguf")?;
//! let config = GenerationConfig::default()
//!     .max_tokens(128)
//!     .temperature(0.7);
//! let output = engine.generate("Explain how Rust works.", &config)?;
//! println!("{}", output.text);
//! # Ok(())
//! # }
//! ```

pub mod engine;
pub mod info;
pub mod session;
pub mod stats;

pub use engine::{Engine, EngineOptions};
pub use info::ModelInfo;
pub use session::{Flow, GenerateOutput, GenerationConfig, Session, StopReason};
pub use stats::{EngineStats, Timing};
