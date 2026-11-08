//! Parallel matrix multiplication: `out[t, o] = sum_i w[o, i] * x[t, i]`.
//!
//! `w` is a GGUF weight matrix laid out as `n_out` contiguous rows of `n_in`
//! elements each (the natural GGUF `[ne0 = n_in, ne1 = n_out]` layout).
//! Work is split over output channels so each thread streams its weight rows
//! once while reusing the (small) activation block.

use llama_core::{GgmlType, LlamaError, Result};
use rayon::prelude::*;

use crate::dispatch::{Activation, vec_dot_row};

/// Computes a batched matrix-vector product for a quantized/f16/f32 weight
/// matrix.
///
/// * `w` — serialized weight bytes (`n_out * row_size` bytes).
/// * `n_out`, `n_in` — matrix shape.
/// * `inputs` — `T` prepared activation rows.
/// * `out` — output buffer of `T * n_out` floats (row-major over `t`).
pub fn matmul(
    ty: GgmlType,
    w: &[u8],
    n_out: usize,
    n_in: usize,
    inputs: &[Activation],
    out: &mut [f32],
) -> Result<()> {
    let t = inputs.len();
    if out.len() != t * n_out {
        return Err(LlamaError::kernel(format!(
            "matmul: output buffer is {} floats, expected {}x{}",
            out.len(),
            t,
            n_out
        )));
    }
    let row_bytes = ty.row_size_bytes(n_in as u64).ok_or_else(|| {
        LlamaError::kernel(format!(
            "matmul: n_in {n_in} is not a multiple of the {} block size",
            ty.block_size()
        ))
    })? as usize;
    if w.len() != n_out * row_bytes {
        return Err(LlamaError::kernel(format!(
            "matmul: weight buffer is {} bytes, expected {} rows x {} bytes",
            w.len(),
            n_out,
            row_bytes
        )));
    }
    for act in inputs {
        if act.n_elements() != n_in {
            return Err(LlamaError::kernel(format!(
                "matmul: activation has {} elements, weight expects {n_in}",
                act.n_elements()
            )));
        }
    }

    // Parallelize over output channels; sequential over the T input rows so
    // the activation block stays hot in cache.
    let chunk = out.chunks_exact_mut(n_out);
    let n_chunks = t.max(1);
    debug_assert_eq!(chunk.len(), n_chunks);

    out.par_chunks_exact_mut(n_out)
        .enumerate()
        .try_for_each(|(t_idx, out_row)| {
            let act = &inputs[t_idx];
            let mut o = 0usize;
            while o < n_out {
                let wrow = &w[o * row_bytes..(o + 1) * row_bytes];
                out_row[o] = vec_dot_row(ty, wrow, act)?;
                o += 1;
            }
            Ok::<(), LlamaError>(())
        })
}

/// Convenience single-row matmul (token decoding hot path).
pub fn matmul_row(
    ty: GgmlType,
    w: &[u8],
    n_out: usize,
    n_in: usize,
    act: &Activation,
    out: &mut [f32],
) -> Result<()> {
    matmul(ty, w, n_out, n_in, std::slice::from_ref(act), out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::quantize_row_q8_0;

    #[test]
    fn f32_matmul_matches_naive() {
        let n_in = 64;
        let n_out = 48;
        let w: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i * 37) % 101) as f32 / 51.0 - 1.0)
            .collect();
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i * 7) % 13) as f32 / 6.5 - 1.0)
            .collect();
        let wb: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();

        let act = Activation::F32(x.clone());
        let mut out = vec![0.0f32; n_out];
        matmul_row(GgmlType::F32, &wb, n_out, n_in, &act, &mut out).unwrap();

        for o in 0..n_out {
            let expected: f32 = (0..n_in).map(|i| w[o * n_in + i] * x[i]).sum();
            assert!(
                (out[o] - expected).abs() < 1e-3 * expected.abs().max(1.0),
                "row {o}: {} vs {expected}",
                out[o]
            );
        }
    }

    #[test]
    fn q8_0_matmul_matches_scalar_dot() {
        let n_in = 128;
        let n_out = 16;
        let w: Vec<f32> = (0..n_in * n_out)
            .map(|i| ((i * 29) % 71) as f32 / 35.0 - 1.0)
            .collect();
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i * 11) % 19) as f32 / 9.5 - 1.0)
            .collect();
        let wq = quantize_row_q8_0(&w);
        let wb: Vec<u8> = wq
            .blocks
            .iter()
            .flat_map(|b| {
                let mut v = b.d.to_le_bytes().to_vec();
                v.extend(b.qs.iter().map(|&q| q as u8));
                v
            })
            .collect();
        let act = Activation::prepare(GgmlType::Q8_0, &x);
        let mut out = vec![0.0f32; n_out];
        matmul_row(GgmlType::Q8_0, &wb, n_out, n_in, &act, &mut out).unwrap();

        let xq = quantize_row_q8_0(&x);
        for o in 0..n_out {
            let wrow = &wb[o * (n_in / 32 * 34)..(o + 1) * (n_in / 32 * 34)];
            let expected = crate::vecdot::vec_dot_q8_0(wrow, &xq);
            assert!((out[o] - expected).abs() < 1e-4, "row {o}");
        }
    }

    #[test]
    fn rejects_bad_shapes() {
        let wb = vec![0u8; 64 * 4];
        let act = Activation::F32(vec![0.0; 63]);
        let mut out = vec![0.0f32; 64];
        let err = matmul_row(GgmlType::F32, &wb, 64, 64, &act, &mut out);
        assert!(err.is_err());
    }
}
