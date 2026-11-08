//! Matrix multiplication entry points combining `WeightMatrix` views with
//! the llama-kernels dispatcher and rayon parallelism.

use llama_core::Result;
use llama_kernels::dispatch::Activation;

use crate::weight::WeightMatrix;

/// Multiplies a weight matrix with a batch of activation rows.
///
/// `out` must hold `inputs.len() * weight.n_rows()` floats, laid out
/// row-major over the input rows (`out[t * n_out + o]`).
pub fn matmul_rows<'a>(
    weight: &WeightMatrix<'_>,
    inputs: &'a [Activation],
    out: &'a mut [f32],
) -> Result<()> {
    llama_kernels::matmul::matmul(
        weight.ty(),
        weight.as_bytes(),
        weight.n_rows(),
        weight.n_in(),
        inputs,
        out,
    )
}

/// Multiplies a weight matrix with a single prepared activation row.
pub fn matmul_row(weight: &WeightMatrix<'_>, input: &Activation, out: &mut [f32]) -> Result<()> {
    llama_kernels::matmul::matmul_row(
        weight.ty(),
        weight.as_bytes(),
        weight.n_rows(),
        weight.n_in(),
        input,
        out,
    )
}

/// Prepares an activation row for a weight matrix of the given type.
#[must_use]
pub fn prepare_activation(weight: &WeightMatrix<'_>, x: &[f32]) -> Activation {
    Activation::prepare(weight.ty(), x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_core::GgmlType;

    #[test]
    fn f32_matmul_end_to_end() {
        let n_in = 16;
        let n_out = 8;
        let w: Vec<f32> = (0..n_in * n_out)
            .map(|i| (i % 7) as f32 * 0.25 - 0.75)
            .collect();
        let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let weight = WeightMatrix::new(&bytes, GgmlType::F32, [n_in, n_out]).unwrap();

        let x: Vec<f32> = (0..n_in).map(|i| (i % 5) as f32 * 0.2 - 0.4).collect();
        let act = prepare_activation(&weight, &x);
        let mut out = vec![0.0f32; n_out];
        matmul_row(&weight, &act, &mut out).unwrap();

        for o in 0..n_out {
            let expected: f32 = (0..n_in).map(|i| w[o * n_in + i] * x[i]).sum();
            assert!(
                (out[o] - expected).abs() < 1e-4,
                "{o}: {} vs {expected}",
                out[o]
            );
        }
    }
}
