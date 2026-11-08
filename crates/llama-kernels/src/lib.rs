//! # llama-kernels
//!
//! CPU compute kernels for the llama.rs inference engine.
//!
//! * Scalar reference implementations for every supported GGML tensor type.
//! * SIMD implementations (AVX2/FMA/F16C on x86-64) selected at runtime.
//! * Parallel quantized matrix multiplication via `rayon`.
//! * Transformer pointwise operations: `RMSNorm`, `RoPE`, softmax, `SwiGLU`.
//!
//! All kernels operate on raw byte slices (`&[u8]`) for weights so that
//! memory-mapped GGUF data can be used zero-copy, and on `f32` buffers for
//! activations. Block layouts match ggml/llama.cpp bit-for-bit.

// SIMD kernels call CPU intrinsics (`std::arch`) behind runtime feature
// detection, which requires `unsafe` blocks. This crate is the single place
// in the workspace where that is permitted; every block carries a `SAFETY:`
// comment, and all other crates keep the workspace `unsafe_code = warn` lint.
#![allow(unsafe_code)]

pub mod dispatch;
pub mod matmul;
pub mod ops;
pub mod quant;
pub mod vecdot;

#[cfg(target_arch = "x86_64")]
pub mod simd;

pub use dispatch::vec_dot_row;
pub use quant::{BlockQ8_0, QuantizedRow, dequantize_row};
pub use vecdot::{vec_dot_f16, vec_dot_f32};
