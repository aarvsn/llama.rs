//! Tensor directory entries parsed from a GGUF file.

use llama_core::{GgmlType, Result};

/// A parsed tensor directory entry.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name, e.g. `blk.0.attn_q.weight`.
    pub name: String,
    /// Dimensions. `dims[0]` is the fastest-varying (innermost) dimension,
    /// matching the GGUF/ggml convention.
    pub dims: Vec<u64>,
    /// Element/quantization type.
    pub ty: GgmlType,
    /// Byte offset of the tensor data relative to the start of the data
    /// section. Guaranteed to be a multiple of the file alignment.
    pub offset: u64,
}

impl TensorInfo {
    /// Number of elements in the tensor (product of dims).
    #[must_use]
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Number of rows: product of all dims except `dims[0]`.
    /// For a 2-D weight matrix `[n_in, n_out]` this is `n_out`.
    #[must_use]
    pub fn n_rows(&self) -> u64 {
        if self.dims.is_empty() {
            0
        } else {
            self.dims[1..].iter().product::<u64>().max(1)
        }
    }

    /// Length of the fastest-varying dimension (`dims[0]`).
    #[must_use]
    pub fn row_elements(&self) -> u64 {
        self.dims.first().copied().unwrap_or(0)
    }

    /// Total serialized size in bytes.
    ///
    /// Errors when the element count is not a multiple of the block size.
    pub fn size_bytes(&self) -> Result<u64> {
        self.ty.row_size_bytes(self.n_elements()).ok_or_else(|| {
            llama_core::LlamaError::InvalidTensor(format!(
                "tensor '{}' has {} elements, not a multiple of the block size {} for type {}",
                self.name,
                self.n_elements(),
                self.ty.block_size(),
                self.ty
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry() {
        let info = TensorInfo {
            name: "w".into(),
            dims: vec![64, 32],
            ty: GgmlType::Q4_0,
            offset: 0,
        };
        assert_eq!(info.n_elements(), 64 * 32);
        assert_eq!(info.n_rows(), 32);
        assert_eq!(info.row_elements(), 64);
        assert_eq!(info.size_bytes().unwrap(), 2 * 32 * 18); // 2 blocks per row, 32 rows
    }
}
