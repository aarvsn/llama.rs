//! Unified `vec_dot` dispatch by tensor type with runtime SIMD selection.

use llama_core::{GgmlType, LlamaError, Result};

use crate::quant::{QuantizedRow, quantize_row_q8_0};

/// An activation row in the format a matrix multiply needs.
///
/// Unquantized weights consume the raw `f32` row; quantized weights consume
/// the Q8_0-quantized row (quantize once per row, reuse across every output
/// channel).
pub enum Activation {
    /// Plain f32 activations (for F32/F16 weight rows).
    F32(Vec<f32>),
    /// Q8_0-quantized activations (for quantized weight rows).
    Q8(QuantizedRow),
}

impl Activation {
    /// Prepares an activation row for a weight of type `ty`.
    ///
    /// This is where activations are quantized exactly once so that the dot
    /// products against every output channel reuse the same `Q8_0` blocks.
    #[must_use]
    pub fn prepare(ty: GgmlType, x: &[f32]) -> Self {
        match ty {
            GgmlType::F32 | GgmlType::F16 => Self::F32(x.to_vec()),
            _ => Self::Q8(quantize_row_q8_0(x)),
        }
    }

    /// Number of elements.
    #[must_use]
    pub fn n_elements(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::Q8(q) => q.n_elements(),
        }
    }
}

/// Computes the dot product of a weight row with an activation row.
///
/// `w` is the raw serialized weight row (GGUF byte layout); the SIMD variant
/// is selected at runtime when the CPU supports it.
pub fn vec_dot_row(ty: GgmlType, w: &[u8], act: &Activation) -> Result<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        // LLAMA_RS_NO_SIMD=1 forces the portable scalar kernels (debugging).
        let simd_disabled = std::env::var_os("LLAMA_RS_NO_SIMD").is_some_and(|v| v == "1");
        let avx2 = !simd_disabled && std::arch::is_x86_feature_detected!("avx2");
        let fma = !simd_disabled && std::arch::is_x86_feature_detected!("fma");
        let f16c = !simd_disabled && std::arch::is_x86_feature_detected!("f16c");
        if avx2 {
            let value = match (ty, act) {
                (GgmlType::F32, Activation::F32(x)) if fma => {
                    // SAFETY: AVX2+FMA were detected at runtime.
                    Some(unsafe { crate::simd::vec_dot_f32_avx2(w, x) })
                }
                (GgmlType::F16, Activation::F32(x)) if f16c => {
                    // SAFETY: AVX2+F16C were detected at runtime.
                    Some(unsafe { crate::simd::vec_dot_f16_avx2(w, x) })
                }
                (GgmlType::Q4_0, Activation::Q8(x)) => {
                    // SAFETY: AVX2 was detected at runtime.
                    Some(unsafe { crate::simd::vec_dot_q4_0_avx2(w, x) })
                }
                (GgmlType::Q4_1, Activation::Q8(x)) => {
                    // SAFETY: AVX2 was detected at runtime.
                    Some(unsafe { crate::simd::vec_dot_q4_1_avx2(w, x) })
                }
                (GgmlType::Q5_0, Activation::Q8(x)) => {
                    // SAFETY: AVX2 was detected at runtime.
                    Some(unsafe { crate::simd::vec_dot_q5_0_avx2(w, x) })
                }
                (GgmlType::Q5_1, Activation::Q8(x)) => {
                    // SAFETY: AVX2 was detected at runtime.
                    Some(unsafe { crate::simd::vec_dot_q5_1_avx2(w, x) })
                }
                (GgmlType::Q8_0, Activation::Q8(x)) => {
                    // SAFETY: AVX2 was detected at runtime.
                    Some(unsafe { crate::simd::vec_dot_q8_0_avx2(w, x) })
                }
                _ => None,
            };
            if let Some(v) = value {
                return Ok(v);
            }
        }
    }

    let value = match (ty, act) {
        (GgmlType::F32, Activation::F32(x)) => crate::vecdot::vec_dot_f32(w, x),
        (GgmlType::F16, Activation::F32(x)) => crate::vecdot::vec_dot_f16(w, x),
        (GgmlType::Q4_0, Activation::Q8(x)) => crate::vecdot::vec_dot_q4_0(w, x),
        (GgmlType::Q4_1, Activation::Q8(x)) => crate::vecdot::vec_dot_q4_1(w, x),
        (GgmlType::Q5_0, Activation::Q8(x)) => crate::vecdot::vec_dot_q5_0(w, x),
        (GgmlType::Q5_1, Activation::Q8(x)) => crate::vecdot::vec_dot_q5_1(w, x),
        (GgmlType::Q8_0, Activation::Q8(x)) => crate::vecdot::vec_dot_q8_0(w, x),
        (GgmlType::Q4_K, Activation::Q8(x)) => crate::vecdot::vec_dot_q4_k(w, x),
        (GgmlType::Q5_K, Activation::Q8(x)) => crate::vecdot::vec_dot_q5_k(w, x),
        (GgmlType::Q6_K, Activation::Q8(x)) => crate::vecdot::vec_dot_q6_k(w, x),
        (ty, act) => {
            return Err(LlamaError::kernel(format!(
                "vec_dot_row: type {ty} with {} activation is not supported",
                match act {
                    Activation::F32(_) => "f32",
                    Activation::Q8(_) => "q8_0",
                }
            )));
        }
    };
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_prepare_selects_format() {
        let x = vec![0.5f32; 64];
        assert!(matches!(
            Activation::prepare(GgmlType::F32, &x),
            Activation::F32(_)
        ));
        assert!(matches!(
            Activation::prepare(GgmlType::F16, &x),
            Activation::F32(_)
        ));
        let q = Activation::prepare(GgmlType::Q4_0, &x);
        assert!(matches!(&q, Activation::Q8(r) if r.n_elements() == 64));
    }
}
