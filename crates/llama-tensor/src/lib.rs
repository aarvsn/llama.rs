//! # llama-tensor
//!
//! Tensor abstractions for llama.rs:
//!
//! * [`Shape`] — dimension/stride bookkeeping with GGML ordering conventions.
//! * [`Tensor`] — an owned dense `f32` tensor used for activations.
//! * [`WeightMatrix`] — a zero-copy view over serialized GGUF weight bytes
//!   with row indexing; matrix products are delegated to `llama-kernels`.

pub mod matmul;
pub mod shape;
pub mod tensor;
pub mod weight;

pub use matmul::{matmul_row, matmul_rows};
pub use shape::Shape;
pub use tensor::Tensor;
pub use weight::WeightMatrix;
