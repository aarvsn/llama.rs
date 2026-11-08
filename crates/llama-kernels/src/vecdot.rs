//! Scalar dot products between a weight row (raw GGUF bytes) and an
//! activation vector (`f32` for unquantized weights, `Q8_0` for quantized
//! weights).
//!
//! These are the portable reference implementations; x86-64 builds override
//! the hot paths with AVX2 kernels at runtime (see [`crate::simd`]).
//!
//! Integer accumulation follows ggml: products of small integers are
//! accumulated exactly in `i32`/`i64` and only combined with the block scales
//! at the end, which keeps quantized dot products precise.

// Kernel bodies follow ggml naming (d, q, qs, b, g): single-letter block
// and loop variables are the established convention for quantized math.
#![allow(clippy::many_single_char_names)]

use crate::quant::{QuantizedRow, block_f16};

/// Dot product of an f32 weight row against an f32 activation row.
#[must_use]
pub fn vec_dot_f32(w: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len() * 4);
    let n = x.len();
    let mut acc = 0.0f32;
    for i in 0..n {
        let b = [w[i * 4], w[i * 4 + 1], w[i * 4 + 2], w[i * 4 + 3]];
        acc += f32::from_le_bytes(b) * x[i];
    }
    acc
}

/// Dot product of an f16 weight row against an f32 activation row.
#[must_use]
pub fn vec_dot_f16(w: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len() * 2);
    let mut acc = 0.0f32;
    for i in 0..x.len() {
        let h = u16::from_le_bytes([w[i * 2], w[i * 2 + 1]]);
        acc += llama_core::F16::from_bits(h).to_f32() * x[i];
    }
    acc
}

/// `Q8_0` weight row against a `Q8_0` activation row.
#[must_use]
pub fn vec_dot_q8_0(w: &[u8], x: &QuantizedRow) -> f32 {
    let nb = x.n_blocks();
    debug_assert_eq!(w.len(), nb * 34);
    let mut acc = 0.0f32;
    for b in 0..nb {
        let dw = block_f16(w, b * 34);
        let wq = &w[b * 34 + 2..b * 34 + 34];
        let xb = &x.blocks[b];
        let dx = llama_core::F16::from_bits(xb.d).to_f32();
        let mut sum: i32 = 0;
        for (&wj, &xj) in wq.iter().zip(xb.qs.iter()) {
            sum += i32::from(wj as i8) * i32::from(xj);
        }
        acc += dw * dx * sum as f32;
    }
    acc
}

/// `Q4_0` weight row against a `Q8_0` activation row.
#[must_use]
pub fn vec_dot_q4_0(w: &[u8], x: &QuantizedRow) -> f32 {
    let nb = x.n_blocks();
    debug_assert_eq!(w.len(), nb * 18);
    let mut acc = 0.0f32;
    for b in 0..nb {
        let dw = block_f16(w, b * 18);
        let qs = &w[b * 18 + 2..b * 18 + 18];
        let xb = &x.blocks[b];
        let dx = llama_core::F16::from_bits(xb.d).to_f32();
        let mut sum: i32 = 0;
        let mut sum_quants: i32 = 0;
        let (x_lo, x_hi) = xb.qs.split_at(16);
        for ((&qs_j, &x_lo), &x_hi) in qs.iter().zip(x_lo).zip(x_hi) {
            let lo = i32::from(qs_j & 0x0F);
            let hi = i32::from(qs_j >> 4);
            sum += lo * i32::from(x_lo) + hi * i32::from(x_hi);
            sum_quants += i32::from(x_lo) + i32::from(x_hi);
        }
        // value_i = (nibble_i - 8) * d
        acc += dw * dx * (sum as f32 - 8.0 * sum_quants as f32);
    }
    acc
}

/// `Q4_1` weight row against a `Q8_0` activation row.
#[must_use]
pub fn vec_dot_q4_1(w: &[u8], x: &QuantizedRow) -> f32 {
    let nb = x.n_blocks();
    debug_assert_eq!(w.len(), nb * 20);
    let mut acc = 0.0f32;
    for b in 0..nb {
        let dw = block_f16(w, b * 20);
        let mw = block_f16(w, b * 20 + 2);
        let qs = &w[b * 20 + 4..b * 20 + 20];
        let xb = &x.blocks[b];
        let dx = llama_core::F16::from_bits(xb.d).to_f32();
        let mut sum: i32 = 0;
        let mut sum_quants: i32 = 0;
        let (x_lo, x_hi) = xb.qs.split_at(16);
        for ((&qs_j, &x_lo), &x_hi) in qs.iter().zip(x_lo).zip(x_hi) {
            let lo = i32::from(qs_j & 0x0F);
            let hi = i32::from(qs_j >> 4);
            sum += lo * i32::from(x_lo) + hi * i32::from(x_hi);
            sum_quants += i32::from(x_lo) + i32::from(x_hi);
        }
        // value_i = nibble_i * d + m
        acc += dx * (dw * sum as f32 + mw * sum_quants as f32);
    }
    acc
}

/// `Q5_0` weight row against a `Q8_0` activation row.
#[must_use]
pub fn vec_dot_q5_0(w: &[u8], x: &QuantizedRow) -> f32 {
    let nb = x.n_blocks();
    debug_assert_eq!(w.len(), nb * 22);
    let mut acc = 0.0f32;
    for b in 0..nb {
        let dw = block_f16(w, b * 22);
        let qh = u32::from_le_bytes([w[b * 22 + 2], w[b * 22 + 3], w[b * 22 + 4], w[b * 22 + 5]]);
        let qs = &w[b * 22 + 6..b * 22 + 22];
        let xb = &x.blocks[b];
        let dx = llama_core::F16::from_bits(xb.d).to_f32();
        let mut sum: i32 = 0;
        let mut sum_quants: i32 = 0;
        let mut sum_high: i32 = 0;
        let (x_lo, x_hi) = xb.qs.split_at(16);
        for (j, ((&qs_j, &x_lo), &x_hi)) in qs.iter().zip(x_lo).zip(x_hi).enumerate() {
            let lo = i32::from(qs_j & 0x0F);
            let hi = i32::from(qs_j >> 4);
            sum += lo * i32::from(x_lo) + hi * i32::from(x_hi);
            sum_quants += i32::from(x_lo) + i32::from(x_hi);
            if qh & (1 << j) != 0 {
                sum_high += 16 * i32::from(x_lo);
            }
            if qh & (1 << (j + 16)) != 0 {
                sum_high += 16 * i32::from(x_hi);
            }
        }
        // value_i = ((nibble_i | high_i) - 16) * d
        acc += dw * dx * ((sum + sum_high) as f32 - 16.0 * sum_quants as f32);
    }
    acc
}

/// `Q5_1` weight row against a `Q8_0` activation row.
#[must_use]
pub fn vec_dot_q5_1(w: &[u8], x: &QuantizedRow) -> f32 {
    let nb = x.n_blocks();
    debug_assert_eq!(w.len(), nb * 24);
    let mut acc = 0.0f32;
    for b in 0..nb {
        let dw = block_f16(w, b * 24);
        let mw = block_f16(w, b * 24 + 2);
        let qh = u32::from_le_bytes([w[b * 24 + 4], w[b * 24 + 5], w[b * 24 + 6], w[b * 24 + 7]]);
        let qs = &w[b * 24 + 8..b * 24 + 24];
        let xb = &x.blocks[b];
        let dx = llama_core::F16::from_bits(xb.d).to_f32();
        let mut sum: i32 = 0;
        let mut sum_quants: i32 = 0;
        let mut sum_high: i32 = 0;
        let (x_lo, x_hi) = xb.qs.split_at(16);
        for (j, ((&qs_j, &x_lo), &x_hi)) in qs.iter().zip(x_lo).zip(x_hi).enumerate() {
            let lo = i32::from(qs_j & 0x0F);
            let hi = i32::from(qs_j >> 4);
            sum += lo * i32::from(x_lo) + hi * i32::from(x_hi);
            sum_quants += i32::from(x_lo) + i32::from(x_hi);
            if qh & (1 << j) != 0 {
                sum_high += 16 * i32::from(x_lo);
            }
            if qh & (1 << (j + 16)) != 0 {
                sum_high += 16 * i32::from(x_hi);
            }
        }
        // value_i = (nibble_i | high_i) * d + m
        acc += dx * (dw * (sum + sum_high) as f32 + mw * sum_quants as f32);
    }
    acc
}

/// `Q4_K` weight row against a `Q8_0` activation row.
///
/// Each 256-element super-block contains 8 sub-blocks of 32 elements; the
/// s-th sub-block pairs with the s-th `Q8_0` activation block. Sub-block 2g
/// reads the low nibbles of `qs[32g..32g+32]` and sub-block 2g+1 reads the
/// high nibbles of the same bytes.
#[must_use]
pub fn vec_dot_q4_k(w: &[u8], x: &QuantizedRow) -> f32 {
    const BLK: usize = 144;
    const NB32: usize = 8; // sub-blocks per super-block
    let nsb = x.n_blocks(); // Q8_0 blocks == 32-element sub-blocks
    let nsuper = nsb / NB32;
    debug_assert_eq!(w.len(), nsuper * BLK);
    let mut acc = 0.0f32;
    for s in 0..nsuper {
        let off = s * BLK;
        let d = block_f16(w, off);
        let dmin = block_f16(w, off + 2);
        let scales = &w[off + 4..off + 16];
        let qs = &w[off + 16..off + 144];
        for sub in 0..NB32 {
            let (sc, m) = crate::quant::scale_min_k4_public(scales, sub);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let xb = &x.blocks[s * NB32 + sub];
            let dx = llama_core::F16::from_bits(xb.d).to_f32();
            let qbase = (sub / 2) * 32;
            let low = sub % 2 == 0;
            let mut sum: i32 = 0;
            let mut sumq: i32 = 0;
            for l in 0..32 {
                let nib = i32::from(if low {
                    qs[qbase + l] & 0x0F
                } else {
                    qs[qbase + l] >> 4
                });
                sum += nib * i32::from(xb.qs[l]);
                sumq += i32::from(xb.qs[l]);
            }
            // elements = d1 * nibble - m1
            acc += dx * (d1 * sum as f32 - m1 * sumq as f32);
        }
    }
    acc
}

/// `Q5_K` weight row against a `Q8_0` activation row.
///
/// Same sub-block structure as `Q4_K`, plus a 5th bit per element: element
/// (64g + l) reads bit 2g of `qh[l]` and element (64g + 32 + l) reads bit
/// (2g + 1) of `qh[l]`.
#[must_use]
pub fn vec_dot_q5_k(w: &[u8], x: &QuantizedRow) -> f32 {
    const BLK: usize = 176;
    const NB32: usize = 8;
    let nsb = x.n_blocks();
    let nsuper = nsb / NB32;
    debug_assert_eq!(w.len(), nsuper * BLK);
    let mut acc = 0.0f32;
    for s in 0..nsuper {
        let off = s * BLK;
        let d = block_f16(w, off);
        let dmin = block_f16(w, off + 2);
        let scales = &w[off + 4..off + 16];
        let qh = &w[off + 16..off + 48];
        let ql = &w[off + 48..off + 176];
        for sub in 0..NB32 {
            let (sc, m) = crate::quant::scale_min_k4_public(scales, sub);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            let xb = &x.blocks[s * NB32 + sub];
            let dx = llama_core::F16::from_bits(xb.d).to_f32();
            let g = sub / 2;
            let qbase = g * 32;
            let low = sub % 2 == 0;
            // Bit 2g selects elements [64g, 64g+32); bit 2g+1 selects
            // elements [64g+32, 64g+64).
            let u: u8 = if low { 1 << (2 * g) } else { 2 << (2 * g) };
            let mut sum: i32 = 0;
            let mut sumq: i32 = 0;
            for l in 0..32 {
                let mut nib = i32::from(if low {
                    ql[qbase + l] & 0x0F
                } else {
                    ql[qbase + l] >> 4
                });
                if qh[l] & u != 0 {
                    nib += 16;
                }
                sum += nib * i32::from(xb.qs[l]);
                sumq += i32::from(xb.qs[l]);
            }
            acc += dx * (d1 * sum as f32 - m1 * sumq as f32);
        }
    }
    acc
}

/// `Q6_K` weight row against a `Q8_0` activation row.
///
/// Within a super-block, 16-element group `g` (0..16) reads:
/// `p = (g % 8) / 2` selects the ql half (32-byte stride) and 2-bit shift in
/// qh; `(g % 8) % 2` selects the 16-byte lane inside the half; `g / 8`
/// selects the 128-element half.
#[must_use]
pub fn vec_dot_q6_k(w: &[u8], x: &QuantizedRow) -> f32 {
    const BLK: usize = 210;
    let nsb = x.n_blocks(); // 32-element Q8_0 blocks; 8 per super-block
    let nsuper = nsb / 8;
    debug_assert_eq!(w.len(), nsuper * BLK);
    let mut acc = 0.0f32;
    for s in 0..nsuper {
        let off = s * BLK;
        let d = block_f16(w, off);
        let ql = &w[off + 2..off + 130];
        let qh = &w[off + 130..off + 194];
        let scales = &w[off + 194..off + 210];
        for (g, &scale_byte) in scales.iter().enumerate() {
            let scale = f32::from(scale_byte as i8);
            let xb = &x.blocks[s * 8 + g / 2];
            let dx = llama_core::F16::from_bits(xb.d).to_f32();
            let half = g % 2;
            let qs = &xb.qs[half * 16..half * 16 + 16];

            let h128 = g / 8; // which 128-element half
            let g8 = g % 8;
            let p = g8 / 2; // pattern within the half
            let lbase = (g8 % 2) * 16;
            let q_low_off = h128 * 64 + (p % 2) * 32;
            let q_high_off = h128 * 32 + lbase;
            let high_nibble = p >= 2;
            let shift = (p * 2) as u32;

            let mut sum: i32 = 0;
            for l in 0..16 {
                let byte = ql[q_low_off + lbase + l];
                let nib = i32::from(if high_nibble { byte >> 4 } else { byte & 0x0F });
                let q = (nib | (i32::from((qh[q_high_off + l] >> shift) & 3) << 4)) - 32;
                sum += q * i32::from(qs[l]);
            }
            acc += d * dx * scale * sum as f32;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{
        dequantize_row_q4_0, dequantize_row_q4_k, dequantize_row_q6_k, quantize_row_q8_0,
    };
    use llama_core::F16;

    fn reference_dot_f32(w: &[f32], x: &[f32]) -> f32 {
        w.iter().zip(x).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn f32_dot_matches_reference() {
        let w: Vec<f32> = (0..100).map(|i| (i as f32 * 0.13) - 6.5).collect();
        let x: Vec<f32> = (0..100).map(|i| ((i * 7) % 23) as f32 - 11.0).collect();
        let wbytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let got = vec_dot_f32(&wbytes, &x);
        let want = reference_dot_f32(&w, &x);
        assert!(
            (got - want).abs() < 1e-3 * want.abs().max(1.0),
            "{got} vs {want}"
        );
    }

    #[test]
    fn f16_dot_matches_reference() {
        let w: Vec<f32> = [0.0, 1.0, -2.0, 0.5, 3.75, -0.25].to_vec();
        let x: Vec<f32> = [1.0, 2.0, 0.5, -1.5, 0.0, 4.0].to_vec();
        let wbytes: Vec<u8> = w
            .iter()
            .flat_map(|v| F16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        let got = vec_dot_f16(&wbytes, &x);
        let want = reference_dot_f32(&w, &x);
        assert!((got - want).abs() < 1e-3, "{got} vs {want}");
    }

    /// Builds a `q4_0` weight row from f32 values with the ggml quantizer.
    fn quantize_row_q4_0(values: &[f32]) -> Vec<u8> {
        assert_eq!(values.len() % 32, 0);
        let mut out = Vec::new();
        for chunk in values.chunks(32) {
            let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 15.0;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
            let mut qs = vec![0u8; 16];
            for (j, &v) in chunk.iter().enumerate() {
                let q = ((v * id) + 8.5) as u32; // ggml rounds like this
                let q = q.clamp(0, 15) as u8;
                if j < 16 {
                    qs[j] |= q & 0x0F;
                } else {
                    qs[j - 16] |= q << 4;
                }
            }
            out.extend_from_slice(&qs);
        }
        out
    }

    #[test]
    fn q4_0_dot_matches_dequantized_reference() {
        let w: Vec<f32> = (0..64)
            .map(|i| ((i * 37) % 41) as f32 / 7.0 - 3.0)
            .collect();
        let x: Vec<f32> = (0..64)
            .map(|i| ((i * 11) % 17) as f32 / 5.0 - 1.5)
            .collect();
        let wbytes = quantize_row_q4_0(&w);
        let xq = quantize_row_q8_0(&x);

        let mut w_deq = vec![0.0f32; 64];
        dequantize_row_q4_0(&wbytes, &mut w_deq);
        let mut x_deq = vec![0.0f32; 64];
        let xbytes: Vec<u8> = xq
            .blocks
            .iter()
            .flat_map(|b| {
                let mut v = b.d.to_le_bytes().to_vec();
                v.extend(b.qs.iter().map(|&q| q as u8));
                v
            })
            .collect();
        crate::quant::dequantize_row_q8_0(&xbytes, &mut x_deq);

        let got = vec_dot_q4_0(&wbytes, &xq);
        let want = reference_dot_f32(&w_deq, &x_deq);
        assert!(
            (got - want).abs() < 0.01 * want.abs().max(1.0),
            "q4_0 dot {got} vs dequantized reference {want}"
        );
    }

    #[test]
    fn q8_0_dot_is_exact_for_integer_scales() {
        // Both rows scaled so all quants are exact small integers.
        let w: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
        let x: Vec<f32> = (0..32).map(|i| (i % 5) as f32 - 2.0).collect();
        let wb: Vec<u8> = {
            let q = quantize_row_q8_0(&w);
            q.blocks
                .iter()
                .flat_map(|b| {
                    let mut v = b.d.to_le_bytes().to_vec();
                    v.extend(b.qs.iter().map(|&qq| qq as u8));
                    v
                })
                .collect()
        };
        let xq = quantize_row_q8_0(&x);
        let got = vec_dot_q8_0(&wb, &xq);
        let want = reference_dot_f32(&w, &x);
        assert!(
            (got - want).abs() < 1e-2 * want.abs().max(1.0),
            "{got} vs {want}"
        );
    }

    #[test]
    fn q6_k_dot_matches_dequantized_reference() {
        // Build a deterministic q6_K row: random-ish quants via LCG.
        let mut seed = 0x1234_5678u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as u8
        };
        let n_super = 2;
        let mut wbytes = Vec::with_capacity(n_super * 210);
        for _ in 0..n_super {
            wbytes.extend_from_slice(&F16::from_f32(0.05).to_bits().to_le_bytes()); // d
            wbytes.extend((0..128).map(|_| next()));
            wbytes.extend((0..64).map(|_| next()));
            wbytes.extend((0..16).map(|_| next()));
        }
        let x: Vec<f32> = (0..512)
            .map(|i| ((i * 13) % 29) as f32 / 9.0 - 1.6)
            .collect();
        let xq = quantize_row_q8_0(&x);

        let mut w_deq = vec![0.0f32; 512];
        dequantize_row_q6_k(&wbytes, &mut w_deq);
        let mut x_deq = vec![0.0f32; 512];
        let xbytes: Vec<u8> = xq
            .blocks
            .iter()
            .flat_map(|b| {
                let mut v = b.d.to_le_bytes().to_vec();
                v.extend(b.qs.iter().map(|&qq| qq as u8));
                v
            })
            .collect();
        crate::quant::dequantize_row_q8_0(&xbytes, &mut x_deq);

        let got = vec_dot_q6_k(&wbytes, &xq);
        let want = reference_dot_f32(&w_deq, &x_deq);
        assert!(
            (got - want).abs() < 0.02 * want.abs().max(1.0),
            "q6_K dot {got} vs dequantized reference {want}"
        );
    }

    #[test]
    fn q4_k_dot_matches_dequantized_reference() {
        let mut seed = 0x00C0_FFEE_u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as u8
        };
        let n_super = 2;
        let mut wbytes = Vec::with_capacity(n_super * 144);
        for _ in 0..n_super {
            wbytes.extend_from_slice(&F16::from_f32(0.03).to_bits().to_le_bytes()); // d
            wbytes.extend_from_slice(&F16::from_f32(0.01).to_bits().to_le_bytes()); // dmin
            wbytes.extend((0..12).map(|_| next())); // scales
            wbytes.extend((0..128).map(|_| next())); // qs
        }
        let x: Vec<f32> = (0..512)
            .map(|i| ((i * 17) % 31) as f32 / 11.0 - 1.4)
            .collect();
        let xq = quantize_row_q8_0(&x);

        let mut w_deq = vec![0.0f32; 512];
        dequantize_row_q4_k(&wbytes, &mut w_deq);
        let mut x_deq = vec![0.0f32; 512];
        let xbytes: Vec<u8> = xq
            .blocks
            .iter()
            .flat_map(|b| {
                let mut v = b.d.to_le_bytes().to_vec();
                v.extend(b.qs.iter().map(|&qq| qq as u8));
                v
            })
            .collect();
        crate::quant::dequantize_row_q8_0(&xbytes, &mut x_deq);

        let got = vec_dot_q4_k(&wbytes, &xq);
        let want = reference_dot_f32(&w_deq, &x_deq);
        assert!(
            (got - want).abs() < 0.02 * want.abs().max(1.0),
            "q4_K dot {got} vs dequantized reference {want}"
        );
    }
}
