//! Quantization primitives: the `Q8_0` activation format, quantization and
//! dequantization of every supported GGML type.
//!
//! Block layouts are the ggml wire layouts (see `ggml-common.h`):

// Block layouts follow ggml naming (d, ql, qh, sc): single-letter block
// and loop variables are the established convention for quantized math.
#![allow(clippy::many_single_char_names)]

use llama_core::ggml_type::QK_K;
use llama_core::{F16, GgmlType, LlamaError, Result};

/// A `Q8_0` block: 32 symmetric int8 quants with an f16 scale.
///
/// `repr(C)` gives the exact ggml layout (2 + 32 bytes).
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BlockQ8_0 {
    /// Scale (f16 bits).
    pub d: u16,
    /// Quantized values.
    pub qs: [i8; 32],
}

impl core::fmt::Debug for BlockQ8_0 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BlockQ8_0")
            .field("d", &F16::from_bits(self.d).to_f32())
            .field("qs", &self.qs)
            .finish()
    }
}

/// A row of activations quantized to `Q8_0` blocks.
///
/// This is the standard ggml "activation quantization": matrix products
/// against any quantized weight type are computed as blockwise integer dot
/// products scaled by the block scales.
#[derive(Clone, Debug)]
pub struct QuantizedRow {
    /// One block per 32 elements.
    pub blocks: Vec<BlockQ8_0>,
}

impl QuantizedRow {
    /// Number of elements represented.
    #[must_use]
    pub fn n_elements(&self) -> usize {
        self.blocks.len() * 32
    }

    /// Number of blocks.
    #[must_use]
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }
}

/// Quantizes a row of `f32` values to `Q8_0` blocks.
///
/// Mirrors `quantize_row_q8_0_ref` from ggml (round-half-away-from-zero).
///
/// # Panics
///
/// Panics in debug builds when `x.len()` is not a multiple of 32. In release
/// builds the trailing partial block is ignored.
#[must_use]
pub fn quantize_row_q8_0(x: &[f32]) -> QuantizedRow {
    let n_blocks = x.len() / 32;
    let mut blocks = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let chunk = &x[b * 32..b * 32 + 32];
        let mut amax = 0.0f32;
        for &v in chunk {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let mut qs = [0i8; 32];
        for (j, &v) in chunk.iter().enumerate() {
            // roundf: round half away from zero, matching the ggml reference.
            qs[j] = (v * id).round() as i8;
        }
        blocks.push(BlockQ8_0 {
            d: F16::from_f32(d).to_bits(),
            qs,
        });
    }
    QuantizedRow { blocks }
}

/// Reads an f16 scale from the first two bytes of a block.
#[inline]
#[must_use]
pub fn block_f16(bytes: &[u8], offset: usize) -> f32 {
    F16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

/// Returns the number of blocks for `n` elements of type `ty`.
#[must_use]
pub fn n_blocks(ty: GgmlType, n: usize) -> usize {
    n / ty.block_size()
}

// ---------------------------------------------------------------------------
// Dequantization (scalar reference implementations, ggml-equivalent)
// ---------------------------------------------------------------------------

/// Dequantizes a `Q4_0` row (`n` must be a multiple of 32).
pub fn dequantize_row_q4_0(bytes: &[u8], out: &mut [f32]) {
    let nb = bytes.len() / 18;
    debug_assert_eq!(out.len(), nb * 32);
    for b in 0..nb {
        let d = block_f16(bytes, b * 18);
        let qs = &bytes[b * 18 + 2..b * 18 + 18];
        let base = b * 32;
        for j in 0..16 {
            out[base + j] = (i32::from(qs[j] & 0x0F) - 8) as f32 * d;
            out[base + j + 16] = (i32::from(qs[j] >> 4) - 8) as f32 * d;
        }
    }
}

/// Dequantizes a `Q4_1` row.
pub fn dequantize_row_q4_1(bytes: &[u8], out: &mut [f32]) {
    let nb = bytes.len() / 20;
    debug_assert_eq!(out.len(), nb * 32);
    for b in 0..nb {
        let d = block_f16(bytes, b * 20);
        let m = block_f16(bytes, b * 20 + 2);
        let qs = &bytes[b * 20 + 4..b * 20 + 20];
        let base = b * 32;
        for j in 0..16 {
            out[base + j] = f32::from(qs[j] & 0x0F) * d + m;
            out[base + j + 16] = f32::from(qs[j] >> 4) * d + m;
        }
    }
}

/// Dequantizes a `Q5_0` row.
pub fn dequantize_row_q5_0(bytes: &[u8], out: &mut [f32]) {
    let nb = bytes.len() / 22;
    debug_assert_eq!(out.len(), nb * 32);
    for b in 0..nb {
        let d = block_f16(bytes, b * 22);
        let qh = u32::from_le_bytes([
            bytes[b * 22 + 2],
            bytes[b * 22 + 3],
            bytes[b * 22 + 4],
            bytes[b * 22 + 5],
        ]);
        let qs = &bytes[b * 22 + 6..b * 22 + 22];
        let base = b * 32;
        for j in 0..16 {
            // Bit j is the 5th bit of element j; bit (j+16) of element j+16.
            let xh_0 = (((qh >> j) & 1) << 4) as u8;
            let xh_1 = (((qh >> (j + 16)) & 1) << 4) as u8;
            out[base + j] = (i32::from((qs[j] & 0x0F) | xh_0) - 16) as f32 * d;
            out[base + j + 16] = (i32::from((qs[j] >> 4) | xh_1) - 16) as f32 * d;
        }
    }
}

/// Dequantizes a `Q5_1` row.
pub fn dequantize_row_q5_1(bytes: &[u8], out: &mut [f32]) {
    let nb = bytes.len() / 24;
    debug_assert_eq!(out.len(), nb * 32);
    for b in 0..nb {
        let d = block_f16(bytes, b * 24);
        let m = block_f16(bytes, b * 24 + 2);
        let qh = u32::from_le_bytes([
            bytes[b * 24 + 4],
            bytes[b * 24 + 5],
            bytes[b * 24 + 6],
            bytes[b * 24 + 7],
        ]);
        let qs = &bytes[b * 24 + 8..b * 24 + 24];
        let base = b * 32;
        for j in 0..16 {
            let xh_0 = (((qh >> j) & 1) << 4) as u8;
            let xh_1 = (((qh >> (j + 16)) & 1) << 4) as u8;
            out[base + j] = f32::from((qs[j] & 0x0F) | xh_0) * d + m;
            out[base + j + 16] = f32::from((qs[j] >> 4) | xh_1) * d + m;
        }
    }
}

/// Dequantizes a `Q8_0` row.
pub fn dequantize_row_q8_0(bytes: &[u8], out: &mut [f32]) {
    let nb = bytes.len() / 34;
    debug_assert_eq!(out.len(), nb * 32);
    for b in 0..nb {
        let d = block_f16(bytes, b * 34);
        let qs = &bytes[b * 34 + 2..b * 34 + 34];
        let base = b * 32;
        for j in 0..32 {
            out[base + j] = f32::from(qs[j] as i8) * d;
        }
    }
}

/// Extracts the j-th (scale, min) pair from a `Q4_K/Q5_K` 12-byte scales field.
///
/// This is `get_scale_min_k4` from ggml. Public so the `vec_dot` kernels can
/// reuse it.
#[inline]
#[must_use]
pub fn scale_min_k4_public(scales: &[u8], j: usize) -> (u32, u32) {
    scale_min_k4(scales, j)
}

/// Extracts the j-th (scale, min) pair from a `Q4_K/Q5_K` 12-byte scales field.
///
/// This is `get_scale_min_k4` from ggml.
#[inline]
fn scale_min_k4(scales: &[u8], j: usize) -> (u32, u32) {
    if j < 4 {
        (u32::from(scales[j] & 63), u32::from(scales[j + 4] & 63))
    } else {
        let d = (u32::from(scales[j + 4] & 0x0F)) | ((u32::from(scales[j - 4] >> 6)) << 4);
        let m = (u32::from(scales[j + 4] >> 4)) | ((u32::from(scales[j] >> 6)) << 4);
        (d, m)
    }
}

/// Dequantizes a `Q4_K` row (256-element super-blocks).
pub fn dequantize_row_q4_k(bytes: &[u8], out: &mut [f32]) {
    const BLK: usize = 144; // block_q4_K size
    let nb = bytes.len() / BLK;
    debug_assert_eq!(out.len(), nb * QK_K);
    for b in 0..nb {
        let off = b * BLK;
        let d = block_f16(bytes, off);
        let dmin = block_f16(bytes, off + 2);
        let scales = &bytes[off + 4..off + 16];
        let qs = &bytes[off + 16..off + 16 + 128];
        let base = b * QK_K;
        let mut is = 0;
        for j in (0..QK_K).step_by(64) {
            let (sc0, m0) = scale_min_k4(scales, is);
            let (sc1, m1) = scale_min_k4(scales, is + 1);
            let d1 = d * sc0 as f32;
            let m1v = dmin * m0 as f32;
            let d2 = d * sc1 as f32;
            let m2v = dmin * m1 as f32;
            let q = &qs[(j / 2)..(j / 2) + 32];
            for l in 0..32 {
                out[base + j + l] = d1 * f32::from(q[l] & 0x0F) - m1v;
            }
            for l in 0..32 {
                out[base + j + 32 + l] = d2 * f32::from(q[l] >> 4) - m2v;
            }
            is += 2;
        }
    }
}

/// Dequantizes a `Q5_K` row.
///
/// Layout note: `ql` advances by 32 bytes per 64-element group, while the
/// 5th-bit masks `u1`/`u2` shift left by 2 per group (the `qh` pointer stays
/// fixed), exactly matching the ggml reference.
pub fn dequantize_row_q5_k(bytes: &[u8], out: &mut [f32]) {
    const BLK: usize = 176; // block_q5_K size
    let nb = bytes.len() / BLK;
    debug_assert_eq!(out.len(), nb * QK_K);
    for b in 0..nb {
        let off = b * BLK;
        let d = block_f16(bytes, off);
        let dmin = block_f16(bytes, off + 2);
        let scales = &bytes[off + 4..off + 16];
        let qh = &bytes[off + 16..off + 16 + 32];
        let ql = &bytes[off + 48..off + 48 + 128];
        let base = b * QK_K;
        let mut is = 0;
        // u1 = 1 << 2g selects the 5th bit of elements [64g, 64g+32),
        // u2 = 2 << 2g selects the bit of elements [64g+32, 64g+64).
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..QK_K).step_by(64) {
            let (sc0, m0) = scale_min_k4(scales, is);
            let (sc1, m1) = scale_min_k4(scales, is + 1);
            let d1 = d * sc0 as f32;
            let m1v = dmin * m0 as f32;
            let d2 = d * sc1 as f32;
            let m2v = dmin * m1 as f32;
            let g = j / 64;
            let lo = &ql[g * 32..g * 32 + 32];
            for l in 0..32 {
                let h = u32::from(u1 & qh[l] != 0) * 16;
                out[base + j + l] = d1 * (u32::from(lo[l] & 0x0F) + h) as f32 - m1v;
            }
            for l in 0..32 {
                let h = u32::from(u2 & qh[l] != 0) * 16;
                out[base + j + 32 + l] = d2 * (u32::from(lo[l] >> 4) + h) as f32 - m2v;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// Dequantizes a `Q6_K` row.
pub fn dequantize_row_q6_k(bytes: &[u8], out: &mut [f32]) {
    const BLK: usize = 210; // block_q6_K size
    let nb = bytes.len() / BLK;
    debug_assert_eq!(out.len(), nb * QK_K);
    for b in 0..nb {
        let off = b * BLK;
        let d = block_f16(bytes, off);
        let ql = &bytes[off + 2..off + 2 + 128];
        let qh = &bytes[off + 130..off + 130 + 64];
        let scales = &bytes[off + 194..off + 194 + 16];
        let base = b * QK_K;
        let mut sc = 0usize;
        let mut q_low_off = 0usize;
        let mut q_high_off = 0usize;
        for n in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let is_ = l / 16;
                let q1 =
                    i32::from((ql[q_low_off + l] & 0x0F) | ((qh[q_high_off + l] & 3) << 4)) - 32;
                let q2 = i32::from(
                    (ql[q_low_off + l + 32] & 0x0F) | (((qh[q_high_off + l] >> 2) & 3) << 4),
                ) - 32;
                let q3 =
                    i32::from((ql[q_low_off + l] >> 4) | (((qh[q_high_off + l] >> 4) & 3) << 4))
                        - 32;
                let q4 = i32::from(
                    (ql[q_low_off + l + 32] >> 4) | (((qh[q_high_off + l] >> 6) & 3) << 4),
                ) - 32;
                let s0 = f32::from(scales[sc + is_] as i8);
                let s2 = f32::from(scales[sc + is_ + 2] as i8);
                let s4 = f32::from(scales[sc + is_ + 4] as i8);
                let s6 = f32::from(scales[sc + is_ + 6] as i8);
                out[base + n + l] = d * s0 * q1 as f32;
                out[base + n + l + 32] = d * s2 * q2 as f32;
                out[base + n + l + 64] = d * s4 * q3 as f32;
                out[base + n + l + 96] = d * s6 * q4 as f32;
            }
            q_low_off += 64;
            q_high_off += 32;
            sc += 8;
        }
    }
}

/// Dequantizes an F16 row.
pub fn dequantize_row_f16(bytes: &[u8], out: &mut [f32]) {
    debug_assert_eq!(bytes.len(), out.len() * 2);
    for (i, v) in out.iter_mut().enumerate() {
        *v = F16::from_bits(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]])).to_f32();
    }
}

/// Dequantizes any supported tensor type into `out`.
pub fn dequantize_row(ty: GgmlType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    match ty {
        GgmlType::F32 => {
            debug_assert_eq!(bytes.len(), out.len() * 4);
            for (i, v) in out.iter_mut().enumerate() {
                *v = f32::from_le_bytes([
                    bytes[i * 4],
                    bytes[i * 4 + 1],
                    bytes[i * 4 + 2],
                    bytes[i * 4 + 3],
                ]);
            }
            Ok(())
        }
        GgmlType::F16 => {
            dequantize_row_f16(bytes, out);
            Ok(())
        }
        GgmlType::Q4_0 => {
            dequantize_row_q4_0(bytes, out);
            Ok(())
        }
        GgmlType::Q4_1 => {
            dequantize_row_q4_1(bytes, out);
            Ok(())
        }
        GgmlType::Q5_0 => {
            dequantize_row_q5_0(bytes, out);
            Ok(())
        }
        GgmlType::Q5_1 => {
            dequantize_row_q5_1(bytes, out);
            Ok(())
        }
        GgmlType::Q8_0 => {
            dequantize_row_q8_0(bytes, out);
            Ok(())
        }
        GgmlType::Q4_K => {
            dequantize_row_q4_k(bytes, out);
            Ok(())
        }
        GgmlType::Q5_K => {
            dequantize_row_q5_k(bytes, out);
            Ok(())
        }
        GgmlType::Q6_K => {
            dequantize_row_q6_k(bytes, out);
            Ok(())
        }
        other => Err(LlamaError::unsupported_gguf(format!(
            "cannot dequantize tensor type {other}"
        ))),
    }
}

/// Quantizes `x` into an existing [`QuantizedRow`], reusing its allocation.
pub fn quantize_row_q8_0_into(dest: &mut QuantizedRow, x: &[f32]) {
    let n_blocks = x.len() / 32;
    dest.blocks.clear();
    dest.blocks.reserve(n_blocks);
    for chunk in x.chunks(32).take(n_blocks) {
        let mut amax = 0.0f32;
        for &v in chunk {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let mut qs = [0i8; 32];
        for (j, &v) in chunk.iter().enumerate() {
            qs[j] = (v * id).round() as i8;
        }
        dest.blocks.push(BlockQ8_0 {
            d: llama_core::F16::from_f32(d).to_bits(),
            qs,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn q8_0_round_trip() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.03).collect();
        let q = quantize_row_q8_0(&x);
        assert_eq!(q.n_elements(), 64);
        let mut back = vec![0.0f32; 64];
        let bytes: Vec<u8> = q
            .blocks
            .iter()
            .flat_map(|b| {
                let mut v = b.d.to_le_bytes().to_vec();
                v.extend(b.qs.iter().map(|&q| q as u8));
                v
            })
            .collect();
        dequantize_row_q8_0(&bytes, &mut back);
        for (a, b) in x.iter().zip(back.iter()) {
            assert!((a - b).abs() < 0.01, "{a} vs {b}");
        }
    }

    #[test]
    fn q4_0_layout_matches_ggml_reference() {
        // Hand-built block: d = 1.0 (0x3C00), nibbles known.
        let mut bytes = vec![0x00, 0x3C]; // f16 1.0 LE
        for j in 0..16 {
            bytes.push((j | ((15 - j) << 4)) as u8);
        }
        let mut out = vec![0.0f32; 32];
        dequantize_row_q4_0(&bytes, &mut out);
        // Element j (j<16) = lo(j) - 8 = j - 8; element j+16 = hi(j) - 8 = (15-j) - 8.
        for j in 0..16 {
            assert_eq!(out[j], (j as f32) - 8.0);
            assert_eq!(out[j + 16], (15.0 - j as f32) - 8.0);
        }
    }

    #[test]
    fn q5_0_layout_matches_ggml_reference() {
        // d = 0.5 (0x3800); qh: bit 0 and bit 16 set; qs[0] = 0x12 (lo=2, hi=1).
        let mut bytes = vec![0x00, 0x38]; // f16 0.5
        let qh: u32 = (1 << 0) | (1 << 16); // bits for element 0 and element 16
        bytes.extend_from_slice(&qh.to_le_bytes());
        bytes.push(0x12);
        bytes.extend(std::iter::repeat_n(0, 15));
        let mut out = vec![0.0f32; 32];
        dequantize_row_q5_0(&bytes, &mut out);
        // element 0 = (2 | 16) - 16 = 2; element 16 = (1 | 16) - 16 = 1; scale 0.5.
        assert_eq!(out[0], 1.0);
        assert_eq!(out[16], 0.5);
        assert_eq!(out[1], (0.0 - 16.0) * 0.5);
    }

    #[test]
    fn f16_dequant() {
        let bytes = f32_bytes(&[1.0]).len();
        let _ = bytes;
        let mut bytes = Vec::new();
        for v in [0.0, 1.0, -2.0, 0.5] {
            bytes.extend_from_slice(&F16::from_f32(v).to_bits().to_le_bytes());
        }
        let mut out = vec![0.0f32; 4];
        dequantize_row_f16(&bytes, &mut out);
        assert_eq!(out, vec![0.0, 1.0, -2.0, 0.5]);
    }

    #[test]
    fn scale_min_k4_reference() {
        // Reproduce ggml get_scale_min_k4 semantics on a crafted table.
        let mut q = [0u8; 12];
        q[0] = 5; // scale 0 = 5
        q[4] = 9; // min 0 = 9
        q[3] = 63; // scale 3 = 63
        q[7] = 10; // min 3 = 10
        q[8] = 0x21; // j=4: scale = lo(q[8]) | (q[0]>>6)<<4 = 1 | 0 = 1
        q[4] |= 0x40; // min 4 = (q[8]>>4) | (q[4]>>6)<<4 = 2 | 1<<4 = 18
        assert_eq!(scale_min_k4(&q, 0), (5, 9));
        assert_eq!(scale_min_k4(&q, 3), (63, 10));
        assert_eq!(scale_min_k4(&q, 4), (1, 18));
    }

    #[test]
    fn q6_k_dequant_reference_block() {
        // One super-block with d = 1.0, all-zero ql/qh and known scales.
        let mut bytes = Vec::with_capacity(210);
        bytes.extend_from_slice(&F16::from_f32(1.0).to_bits().to_le_bytes());
        bytes.extend(std::iter::repeat_n(0u8, 128)); // ql
        bytes.extend(std::iter::repeat_n(0u8, 64)); // qh
        let scales: Vec<i8> = (0..16).map(|i| i as i8 - 8).collect();
        bytes.extend(scales.iter().map(|&s| s as u8));
        assert_eq!(bytes.len(), 210);
        let mut out = vec![0.0f32; 256];
        dequantize_row_q6_k(&bytes, &mut out);
        // Element i uses scales[i / 16]; all quants are (0 - 32) = -32.
        for (i, v) in out.iter().enumerate() {
            let expected = f32::from(scales[i / 16]) * (-32.0);
            assert_eq!(*v, expected, "element {i}");
        }
    }
}
