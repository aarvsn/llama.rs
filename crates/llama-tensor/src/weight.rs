//! Zero-copy weight matrix views over serialized GGUF tensor bytes.

use llama_core::{GgmlType, LlamaError, Result};

use crate::shape::Shape;

/// A immutable view of a GGUF tensor: raw serialized bytes plus type/shape.
///
/// Nothing is copied or dequantized on construction; kernels consume the raw
/// bytes directly.
#[derive(Debug, Clone)]
pub struct WeightMatrix<'a> {
    bytes: &'a [u8],
    ty: GgmlType,
    shape: Shape,
    row_bytes: usize,
}

impl<'a> WeightMatrix<'a> {
    /// Wraps serialized tensor bytes.
    ///
    /// Validates that the buffer length matches the shape/type geometry.
    pub fn new(bytes: &'a [u8], ty: GgmlType, dims: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = Shape::new(dims);
        shape.validate_buffer(ty, bytes.len())?;
        ty.ensure_supported()?;
        let row_bytes = ty.row_size_bytes(shape.row_len() as u64).ok_or_else(|| {
            LlamaError::InvalidTensor(format!(
                "row length {} invalid for type {ty}",
                shape.row_len()
            ))
        })? as usize;
        Ok(Self {
            bytes,
            ty,
            shape,
            row_bytes,
        })
    }

    /// Tensor type.
    #[must_use]
    pub fn ty(&self) -> GgmlType {
        self.ty
    }

    /// Shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Number of rows (output channels for a 2-D weight).
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.shape.n_rows()
    }

    /// Row length (input channels).
    #[must_use]
    pub fn n_in(&self) -> usize {
        self.shape.row_len()
    }

    /// Serialized bytes of output channel `o`.
    #[must_use]
    pub fn row_bytes(&self, o: usize) -> &[u8] {
        debug_assert!(o < self.n_rows());
        &self.bytes[o * self.row_bytes..(o + 1) * self.row_bytes]
    }

    /// All serialized bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes
    }

    /// Number of bytes per row.
    #[must_use]
    pub fn row_byte_len(&self) -> usize {
        self.row_bytes
    }

    /// Dequantizes output channel `o` into `out` (len = `n_in`).
    pub fn dequantize_row_into(&self, o: usize, out: &mut [f32]) -> Result<()> {
        llama_kernels::quant::dequantize_row(self.ty, self.row_bytes(o), out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_and_validates() {
        let bytes = vec![0u8; 4 * 8 * 4]; // [n_in=4, n_out=8] f32
        assert!(WeightMatrix::new(&bytes, GgmlType::F32, [4, 8]).is_ok());
        assert!(WeightMatrix::new(&bytes, GgmlType::F32, [4, 9]).is_err());
        let w = WeightMatrix::new(&bytes, GgmlType::F32, [4, 8]).unwrap();
        assert_eq!(w.n_in(), 4);
        assert_eq!(w.n_rows(), 8);
        assert_eq!(w.row_bytes(3).len(), 16);
    }
}
