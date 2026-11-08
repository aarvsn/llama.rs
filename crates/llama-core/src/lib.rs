//! # llama-core
//!
//! Shared foundation for the llama.rs workspace: error types, configuration,
//! common data structures, half-precision floats, GGML tensor type geometry,
//! the device/backend abstraction and diagnostics helpers.
//!
//! This crate has no dependency on any other llama.rs crate.

pub mod config;
pub mod device;
pub mod error;
pub mod f16;
pub mod ggml_type;
pub mod logging;

pub use config::EngineOptions;
pub use device::{Backend, DeviceInfo};
pub use error::{LlamaError, Result};
pub use f16::F16;
pub use ggml_type::{GgmlType, QK_K};

/// Semantic token identifier used throughout the engine.
pub type TokenId = u32;
