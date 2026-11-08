//! Owned dense f32 tensors used for activations and small parameters.

use llama_core::Result;

use crate::shape::Shape;

/// An owned, dense, row-major `f32` tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
}

impl Tensor {
    /// Allocates a zeroed tensor.
    #[must_use]
    pub fn zeros(dims: impl Into<Vec<usize>>) -> Self {
        let shape = Shape::new(dims);
        let n = shape.n_elements();
        Self {
            data: vec![0.0; n],
            shape,
        }
    }

    /// Creates a tensor from raw data (validates the element count).
    pub fn from_data(dims: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Self> {
        let shape = Shape::new(dims);
        if data.len() != shape.n_elements() {
            return Err(llama_core::LlamaError::InvalidTensor(format!(
                "tensor data has {} elements, shape {shape} needs {}",
                data.len(),
                shape.n_elements()
            )));
        }
        Ok(Self { data, shape })
    }

    /// Shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Flat data view.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable flat data view.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Row `i` (for 2-D tensors: contiguous slice of `row_len` elements).
    #[must_use]
    pub fn row(&self, i: usize) -> &[f32] {
        let rl = self.shape.row_len();
        &self.data[i * rl..(i + 1) * rl]
    }

    /// Mutable row `i`.
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let rl = self.shape.row_len();
        &mut self.data[i * rl..(i + 1) * rl]
    }

    /// Number of rows.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.shape.n_rows()
    }

    /// Row length.
    #[must_use]
    pub fn row_len(&self) -> usize {
        self.shape.row_len()
    }

    /// Total element count.
    #[must_use]
    pub fn n_elements(&self) -> usize {
        self.data.len()
    }

    /// Fills with a value.
    pub fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    /// Copies a row into `out`.
    pub fn copy_row_into(&self, i: usize, out: &mut [f32]) {
        out.copy_from_slice(self.row(i));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_and_rows() {
        let mut t = Tensor::zeros([4, 3]);
        assert_eq!(t.n_elements(), 12);
        assert_eq!(t.n_rows(), 3);
        assert_eq!(t.row_len(), 4);
        t.row_mut(1).copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.row(1), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rejects_mismatched_data() {
        assert!(Tensor::from_data([2, 2], vec![1.0; 5]).is_err());
        assert!(Tensor::from_data([2, 2], vec![1.0; 4]).is_ok());
    }
}
