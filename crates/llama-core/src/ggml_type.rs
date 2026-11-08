//! GGML tensor type enumeration and block geometry.
//!
//! The numeric values are the stable GGUF wire values used by ggml/llama.cpp.
//! Each type knows its block size (elements per quantization block), its
//! serialized block size in bytes, and whether it is supported by the llama.rs
//! compute kernels.

use crate::error::{LlamaError, Result};

/// A GGML tensor data type as stored in GGUF files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    /// 32-bit float.
    F32,
    /// 16-bit IEEE float.
    F16,
    /// 4-bit quantization (32 elements per block, symmetric).
    Q4_0,
    /// 4-bit quantization (32 elements per block, min+delta).
    Q4_1,
    /// 5-bit quantization (32 elements per block, symmetric).
    Q5_0,
    /// 5-bit quantization (32 elements per block, min+delta).
    Q5_1,
    /// 8-bit quantization (32 elements per block, symmetric).
    Q8_0,
    /// 8-bit quantization with sum (only used internally by ggml).
    Q8_1,
    /// K-quant, 2-bit, 256-element super-blocks.
    Q2_K,
    /// K-quant, 3-bit, 256-element super-blocks.
    Q3_K,
    /// K-quant, 4-bit, 256-element super-blocks.
    Q4_K,
    /// K-quant, 5-bit, 256-element super-blocks.
    Q5_K,
    /// K-quant, 6-bit, 256-element super-blocks.
    Q6_K,
    /// K-quant, 8-bit (internal).
    Q8_K,
    /// 16-bit brain float.
    BF16,
    /// Any value llama.rs does not know about.
    Unknown(u32),
}

/// Number of elements in a K-quant super-block.
pub const QK_K: usize = 256;

impl GgmlType {
    /// Parses the raw GGUF type identifier.
    #[must_use]
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            30 => Self::BF16,
            other => Self::Unknown(other),
        }
    }

    /// Returns the raw GGUF identifier.
    #[must_use]
    pub fn to_raw(self) -> u32 {
        match self {
            Self::Unknown(v) => v,
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
            Self::BF16 => 30,
        }
    }

    /// Short lowercase name, e.g. `q4_0`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2_K => "q2_k",
            Self::Q3_K => "q3_k",
            Self::Q4_K => "q4_k",
            Self::Q5_K => "q5_k",
            Self::Q6_K => "q6_k",
            Self::Q8_K => "q8_k",
            Self::BF16 => "bf16",
            Self::Unknown(v) => {
                // Not a leak-prone pattern: the value is formatted on demand.
                // We keep the API const-friendly by returning a static label.
                let _ = v;
                "unknown"
            }
        }
    }

    /// Number of elements per quantization block (1 for unquantized types).
    #[must_use]
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::Unknown(_) => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => QK_K,
        }
    }

    /// Serialized size in bytes of a single block.
    #[must_use]
    pub fn block_byte_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::Q4_0 => 2 + 16,                // 18
            Self::Q4_1 => 2 + 2 + 16,            // 20
            Self::Q5_0 => 2 + 4 + 16,            // 22
            Self::Q5_1 => 2 + 2 + 4 + 16,        // 24
            Self::Q8_0 => 2 + 32,                // 34
            Self::Q8_1 => 2 + 2 + 32,            // 36
            Self::Q2_K => 84,                    // scales(16) + qs(64) + d(2) + dmin(2)
            Self::Q3_K => 110,                   // hmask(32) + qs(64) + scales(12) + d(2)
            Self::Q4_K => 2 + 2 + 12 + 128,      // 144
            Self::Q5_K => 2 + 2 + 12 + 32 + 128, // 176
            Self::Q6_K => 2 + 16 + 128 + 64,     // 210
            Self::Q8_K => 4 + 256 + 32,          // 292 (float d + qs + bsums)
            Self::Unknown(_) => 0,
        }
    }

    /// Whether llama.rs can dequantize and compute with this type.
    #[must_use]
    pub fn is_supported(self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F16
                | Self::Q4_0
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q4_K
                | Self::Q5_K
                | Self::Q6_K
        )
    }

    /// Whether this type is a quantized (blocked) format.
    #[must_use]
    pub fn is_quantized(self) -> bool {
        self.block_size() > 1
    }

    /// Number of bytes needed to store `n_elements` of this type.
    ///
    /// Returns `None` when `n_elements` is not a multiple of the block size
    /// (which indicates a malformed tensor).
    #[must_use]
    pub fn row_size_bytes(self, n_elements: u64) -> Option<u64> {
        let bs = self.block_size() as u64;
        if n_elements % bs != 0 {
            return None;
        }
        Some(n_elements / bs * self.block_byte_size() as u64)
    }

    /// Validates that the type can be used, returning a descriptive error otherwise.
    pub fn ensure_supported(self) -> Result<()> {
        if self.is_supported() {
            Ok(())
        } else {
            Err(LlamaError::unsupported_gguf(format!(
                "tensor type '{}' (GGUF id {}) is not supported by this build of llama.rs; \
                 supported types: f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0, q4_k, q5_k, q6_k",
                self.name(),
                self.to_raw()
            )))
        }
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_values_match_ggml() {
        // Values taken from the ggml headers (GGML_TYPE_*).
        assert_eq!(GgmlType::from_raw(0), GgmlType::F32);
        assert_eq!(GgmlType::from_raw(2), GgmlType::Q4_0);
        assert_eq!(GgmlType::from_raw(6), GgmlType::Q5_0);
        assert_eq!(GgmlType::from_raw(8), GgmlType::Q8_0);
        assert_eq!(GgmlType::from_raw(14), GgmlType::Q6_K);
        assert_eq!(GgmlType::from_raw(30), GgmlType::BF16);
        assert!(matches!(GgmlType::from_raw(42), GgmlType::Unknown(42)));
    }

    #[test]
    fn block_geometry() {
        // Sizes from ggml-common.h static_asserts.
        assert_eq!(GgmlType::Q4_0.block_byte_size(), 18);
        assert_eq!(GgmlType::Q5_0.block_byte_size(), 22);
        assert_eq!(GgmlType::Q5_1.block_byte_size(), 24);
        assert_eq!(GgmlType::Q8_0.block_byte_size(), 34);
        assert_eq!(GgmlType::Q4_K.block_byte_size(), 144);
        assert_eq!(GgmlType::Q5_K.block_byte_size(), 176);
        assert_eq!(GgmlType::Q6_K.block_byte_size(), 210);
    }

    #[test]
    fn row_sizes() {
        assert_eq!(GgmlType::Q4_0.row_size_bytes(256), Some(256 / 32 * 18));
        assert_eq!(GgmlType::Q4_0.row_size_bytes(33), None);
        assert_eq!(GgmlType::F32.row_size_bytes(8), Some(32));
    }

    #[test]
    fn support_matrix() {
        assert!(GgmlType::Q4_K.is_supported());
        assert!(!GgmlType::Q3_K.is_supported());
        assert!(!GgmlType::BF16.is_supported());
        assert!(GgmlType::Q3_K.ensure_supported().is_err());
    }
}
