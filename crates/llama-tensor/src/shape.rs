//! Tensor shapes and strides with GGML dimension ordering.
//!
//! GGUF/ggml order dimensions fastest-first: `dims[0]` is the innermost
//! (contiguous) dimension. A 2-D weight `[n_in, n_out]` is stored as `n_out`
//! contiguous rows of `n_in` elements.

use llama_core::{GgmlType, LlamaError, Result};

/// A tensor shape in GGML (fastest-first) order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creates a shape from dimensions.
    #[must_use]
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    /// Dimensions (fastest-first).
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Total element count.
    #[must_use]
    pub fn n_elements(&self) -> usize {
        self.dims
            .iter()
            .product::<usize>()
            .max(usize::from(!self.dims.is_empty()))
    }

    /// For a 2-D shape `[n_in, n_out]`: the row length.
    #[must_use]
    pub fn row_len(&self) -> usize {
        self.dims.first().copied().unwrap_or(0)
    }

    /// For a 2-D shape `[n_in, n_out]`: the number of rows.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        if self.dims.len() < 2 {
            1
        } else {
            self.dims[1..].iter().product()
        }
    }

    /// Row-major (C-order, output-channel-major) strides matching the GGML
    /// layout: `stride[0] = 1`, `stride[k] = prod(dims[..k])`.
    #[must_use]
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = Vec::with_capacity(self.dims.len());
        let mut acc = 1usize;
        for &d in &self.dims {
            strides.push(acc);
            acc *= d;
        }
        strides
    }

    /// Validates that a serialized buffer matches this shape for `ty`.
    pub fn validate_buffer(&self, ty: GgmlType, byte_len: usize) -> Result<()> {
        let expected = ty.row_size_bytes(self.n_elements() as u64).ok_or_else(|| {
            LlamaError::InvalidTensor(format!(
                "shape {self:?} has {} elements, not a multiple of the {} block size",
                self.n_elements(),
                ty.block_size()
            ))
        })? as usize;
        if byte_len != expected {
            return Err(LlamaError::InvalidTensor(format!(
                "serialized tensor is {byte_len} bytes, expected {expected} for shape {self:?} type {ty}"
            )));
        }
        Ok(())
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry() {
        let s = Shape::new([64, 32]);
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.n_elements(), 2048);
        assert_eq!(s.row_len(), 64);
        assert_eq!(s.n_rows(), 32);
        assert_eq!(s.strides(), vec![1, 64]);
    }

    #[test]
    fn buffer_validation() {
        let s = Shape::new([64, 32]);
        assert!(s.validate_buffer(GgmlType::F32, 64 * 32 * 4).is_ok());
        assert!(s.validate_buffer(GgmlType::F32, 100).is_err());
        assert!(s.validate_buffer(GgmlType::Q4_0, 64 / 32 * 18 * 32).is_ok());
        // 64 elements not divisible into 256-element super-blocks.
        assert!(s.validate_buffer(GgmlType::Q6_K, 1024).is_err());
    }
}
