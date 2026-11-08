//! x86-64 SIMD kernels (AVX2/FMA/F16C) with runtime feature detection.
//!
//! Every function here has a scalar counterpart in [`crate::vecdot`]; the
//! dispatch layer picks the widest implementation the CPU supports. The
//! binary itself is compiled for the baseline x86-64 ISA, so it stays
//! runnable on older processors.
//!
//! Nibble expansion note: GGML Q4/Q5 tensors store element `j` in the low
//! nibble of byte `j` and element `j+16` in its high nibble. The kernels
//! below expand to the vector `[lo(b0)..lo(b15), hi(b0)..hi(b15)]` so the
//! lanes line up with the 32 `Q8_0` activation quants.

// `_mm*_loadu_*`/`_mm*_storeu_*` intrinsics take unaligned pointers by
// design; casting `*const u8` to `*const __m256i` for them is safe and does
// not actually require 32-byte alignment.
#![allow(clippy::cast_ptr_alignment)]
// Kernel bodies follow ggml naming (d, q, qs, b): single-letter loop and
// block variables are the established convention for quantized math.
#![allow(clippy::many_single_char_names)]

use crate::quant::{QuantizedRow, block_f16};
use llama_core::F16;

#[cfg(target_arch = "x86_64")]
// The stdarch glob is the idiomatic way to bring in CPU intrinsics;
// listing them individually is unmaintainable.
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;

/// Horizontal sum of the 8 i32 lanes of a `__m256i`.
#[must_use]
pub fn hsum_i32(v: __m256i) -> i32 {
    unsafe {
        // SAFETY: all intrinsics used are plain register shuffles/adds with
        // no memory access or preconditions.
        let hi = _mm256_extracti128_si256(v, 1);
        let lo = _mm256_castsi256_si128(v);
        let s = _mm_add_epi32(lo, hi); // 4 lanes
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x0E)); // [a+c, b+d, ...]
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x01));
        _mm_cvtsi128_si32(s)
    }
}

/// Horizontal sum of the 8 f32 lanes of a `__m256`.
#[must_use]
pub fn hsum_f32(v: __m256) -> f32 {
    unsafe {
        // SAFETY: register-only shuffles and adds.
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }
}

/// Expands 16 packed nibble bytes into 32 unsigned lanes in dequant order
/// `[lo(b0)..lo(b15), hi(b0)..hi(b15)]`.
#[inline]
#[must_use]
fn nibbles_32(qs: &[u8]) -> __m256i {
    debug_assert!(qs.len() >= 16);
    unsafe {
        // SAFETY: `qs` has at least 16 readable bytes (checked above); all
        // loads are unaligned.
        let b = _mm_loadu_si128(qs.as_ptr().cast::<__m128i>());
        let lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
        let hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
        _mm256_set_m128i(hi, lo)
    }
}

/// Sums 32 signed i8 lanes (as i32).
#[inline]
#[must_use]
fn sum_i8_lanes(q: &[u8]) -> i32 {
    unsafe {
        // SAFETY: `q` has at least 32 readable bytes; maddubs with all-ones
        // computes pairwise products 1*q in i16, then madd collapses to i32.
        let qv = _mm256_loadu_si256(q.as_ptr().cast::<__m256i>());
        let ones = _mm256_set1_epi8(1);
        let pairs = _mm256_maddubs_epi16(ones, qv); // unsigned 1 * signed q
        hsum_i32(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)))
    }
}

/// Sums pairwise products of 32 unsigned lanes (`n`) and 32 signed lanes (`q`).
#[inline]
#[must_use]
fn maddubs_hsum(n: __m256i, q: &[u8]) -> i32 {
    unsafe {
        // SAFETY: `q` has at least 32 readable bytes.
        let qv = _mm256_loadu_si256(q.as_ptr().cast::<__m256i>());
        let pairs = _mm256_maddubs_epi16(n, qv);
        hsum_i32(_mm256_madd_epi16(pairs, _mm256_set1_epi16(1)))
    }
}

/// AVX2+FMA f32 dot product.
///
/// # Safety
///
/// Requires AVX2 and FMA at runtime.
#[target_feature(enable = "avx2", enable = "fma")]
#[must_use]
pub unsafe fn vec_dot_f32_avx2(w: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    debug_assert_eq!(w.len(), n * 4);
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i < chunks {
        // SAFETY: caller guarantees AVX2+FMA; indices bounded by chunks*8 <= n.
        unsafe {
            let wv = _mm256_loadu_ps(w[i * 32..].as_ptr().cast::<f32>());
            let xv = _mm256_loadu_ps(x[i * 8..].as_ptr());
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        i += 1;
    }
    let mut sum = hsum_f32(acc);
    let mut j = chunks * 8;
    while j < n {
        let b = [w[j * 4], w[j * 4 + 1], w[j * 4 + 2], w[j * 4 + 3]];
        sum += f32::from_le_bytes(b) * x[j];
        j += 1;
    }
    sum
}

/// AVX2+F16C f16 × f32 dot product.
///
/// # Safety
///
/// Requires AVX2 and F16C at runtime.
#[target_feature(enable = "avx2", enable = "f16c")]
#[must_use]
pub unsafe fn vec_dot_f16_avx2(w: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    debug_assert_eq!(w.len(), n * 2);
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i < chunks {
        // SAFETY: caller guarantees AVX2+F16C; bounded indices.
        unsafe {
            let hv = _mm_loadu_si128(w[i * 16..].as_ptr().cast::<__m128i>());
            let wv = _mm256_cvtph_ps(hv);
            let xv = _mm256_loadu_ps(x[i * 8..].as_ptr());
            acc = _mm256_add_ps(acc, _mm256_mul_ps(wv, xv));
        }
        i += 1;
    }
    let mut sum = hsum_f32(acc);
    let mut j = chunks * 8;
    while j < n {
        let h = u16::from_le_bytes([w[j * 2], w[j * 2 + 1]]);
        sum += F16::from_bits(h).to_f32() * x[j];
        j += 1;
    }
    sum
}

/// AVX2 `Q8_0` × `Q8_0` dot product.
///
/// # Safety
///
/// Requires AVX2 at runtime.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn vec_dot_q8_0_avx2(w: &[u8], x: &QuantizedRow) -> f32 {
    let nb = x.n_blocks();
    debug_assert_eq!(w.len(), nb * 34);
    let mut acc = 0.0f32;
    for b in 0..nb {
        let dw = block_f16(w, b * 34);
        let wq = &w[b * 34 + 2..b * 34 + 34];
        let xb = &x.blocks[b];
        let dx = F16::from_bits(xb.d).to_f32();
        let qbytes: [u8; 32] = core::array::from_fn(|i| xb.qs[i] as u8);

        // SAFETY: caller guarantees AVX2; slices have 32 readable bytes.
        let mut acc32 = _mm256_setzero_si256();
        for half in 0..2 {
            // SAFETY: caller guarantees AVX2; both slices have 16 readable bytes.
            unsafe {
                let a16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    wq[half * 16..].as_ptr().cast::<__m128i>(),
                ));
                let b16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                    qbytes[half * 16..].as_ptr().cast::<__m128i>(),
                ));
                acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(a16, b16));
            }
        }
        acc += dw * dx * hsum_i32(acc32) as f32;
    }
    acc
}

/// AVX2 `Q4_0` × `Q8_0` dot product.
///
/// # Safety
///
/// Requires AVX2 at runtime.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn vec_dot_q4_0_avx2(w: &[u8], x: &QuantizedRow) -> f32 {
    let nb = x.n_blocks();
    debug_assert_eq!(w.len(), nb * 18);
    let mut acc = 0.0f32;
    for b in 0..nb {
        let dw = block_f16(w, b * 18);
        let qs = &w[b * 18 + 2..b * 18 + 18];
        let xb = &x.blocks[b];
        let dx = F16::from_bits(xb.d).to_f32();
        let qbytes: [u8; 32] = core::array::from_fn(|i| xb.qs[i] as u8);

        let n = nibbles_32(qs);
        let sum = maddubs_hsum(n, &qbytes); // Σ nibble * q
        let sumq = sum_i8_lanes(&qbytes); // Σ q
        // value_i = (nibble_i - 8) * d
        acc += dw * dx * (sum as f32 - 8.0 * sumq as f32);
    }
    acc
}

/// AVX2 `Q4_1` × `Q8_0` dot product.
///
/// # Safety
///
/// Requires AVX2 at runtime.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn vec_dot_q4_1_avx2(w: &[u8], x: &QuantizedRow) -> f32 {
    let nb = x.n_blocks();
    debug_assert_eq!(w.len(), nb * 20);
    let mut acc = 0.0f32;
    for b in 0..nb {
        let dw = block_f16(w, b * 20);
        let mw = block_f16(w, b * 20 + 2);
        let qs = &w[b * 20 + 4..b * 20 + 20];
        let xb = &x.blocks[b];
        let dx = F16::from_bits(xb.d).to_f32();
        let qbytes: [u8; 32] = core::array::from_fn(|i| xb.qs[i] as u8);

        let n = nibbles_32(qs);
        let sum = maddubs_hsum(n, &qbytes);
        let sumq = sum_i8_lanes(&qbytes);
        // value_i = nibble_i * d + m
        acc += dx * (dw * sum as f32 + mw * sumq as f32);
    }
    acc
}

/// Expands the 32 one-bit flags of `qh` (bit i -> element i) into a 32-lane
/// vector of 0/16 bytes.
#[inline]
#[must_use]
fn high_bits_32(qh: u32) -> __m256i {
    let mut buf = [0u8; 32];
    for (i, slot) in buf.iter_mut().enumerate() {
        *slot = (((qh >> i) as u8) & 1) << 4;
    }
    unsafe {
        // SAFETY: `buf` is a local 32-byte array.
        _mm256_loadu_si256(buf.as_ptr().cast::<__m256i>())
    }
}

/// AVX2 `Q5_0` × `Q8_0` dot product.
///
/// # Safety
///
/// Requires AVX2 at runtime.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn vec_dot_q5_0_avx2(w: &[u8], x: &QuantizedRow) -> f32 {
    let nb = x.n_blocks();
    debug_assert_eq!(w.len(), nb * 22);
    let mut acc = 0.0f32;
    for b in 0..nb {
        let dw = block_f16(w, b * 22);
        let qh = u32::from_le_bytes([w[b * 22 + 2], w[b * 22 + 3], w[b * 22 + 4], w[b * 22 + 5]]);
        let qs = &w[b * 22 + 6..b * 22 + 22];
        let xb = &x.blocks[b];
        let dx = F16::from_bits(xb.d).to_f32();
        let qbytes: [u8; 32] = core::array::from_fn(|i| xb.qs[i] as u8);

        let n = nibbles_32(qs);
        let h = high_bits_32(qh);
        // 5-bit values = nibble | high (both unsigned, 0..31).
        let v = _mm256_or_si256(n, h);
        let sum = maddubs_hsum(v, &qbytes);
        let sumq = sum_i8_lanes(&qbytes);
        // value_i = (v_i - 16) * d
        acc += dw * dx * (sum as f32 - 16.0 * sumq as f32);
    }
    acc
}

/// AVX2 `Q5_1` × `Q8_0` dot product.
///
/// # Safety
///
/// Requires AVX2 at runtime.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn vec_dot_q5_1_avx2(w: &[u8], x: &QuantizedRow) -> f32 {
    let nb = x.n_blocks();
    debug_assert_eq!(w.len(), nb * 24);
    let mut acc = 0.0f32;
    for b in 0..nb {
        let dw = block_f16(w, b * 24);
        let mw = block_f16(w, b * 24 + 2);
        let qh = u32::from_le_bytes([w[b * 24 + 4], w[b * 24 + 5], w[b * 24 + 6], w[b * 24 + 7]]);
        let qs = &w[b * 24 + 8..b * 24 + 24];
        let xb = &x.blocks[b];
        let dx = F16::from_bits(xb.d).to_f32();
        let qbytes: [u8; 32] = core::array::from_fn(|i| xb.qs[i] as u8);

        let n = nibbles_32(qs);
        let h = high_bits_32(qh);
        let v = _mm256_or_si256(n, h);
        let sum = maddubs_hsum(v, &qbytes);
        let sumq = sum_i8_lanes(&qbytes);
        // value_i = v_i * d + m
        acc += dx * (dw * sum as f32 + mw * sumq as f32);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::quantize_row_q8_0;
    use crate::vecdot::{
        vec_dot_f32, vec_dot_q4_0, vec_dot_q4_1, vec_dot_q5_0, vec_dot_q5_1, vec_dot_q8_0,
    };
    use llama_core::F16;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*seed >> 40) as f32 / 8_388_608.0) - 1.0
    }

    fn quantize_row_q4_0(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for chunk in values.chunks(32) {
            let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 15.0;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
            let mut qs = vec![0u8; 16];
            for (j, &v) in chunk.iter().enumerate() {
                let q = (((v * id) + 8.5) as u32).clamp(0, 15) as u8;
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

    fn quantize_row_q4_1(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for chunk in values.chunks(32) {
            let mut amax = 0.0f32;
            let mut amin = f32::MAX;
            for &v in chunk {
                amax = amax.max(v);
                amin = amin.min(v);
            }
            let d = (amax - amin) / 15.0;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
            out.extend_from_slice(&F16::from_f32(amin).to_bits().to_le_bytes());
            let mut qs = vec![0u8; 16];
            for (j, &v) in chunk.iter().enumerate() {
                let q = (((v - amin) * id + 0.5) as u32).clamp(0, 15) as u8;
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

    fn q5_pack(values: &[f32], symmetric: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for chunk in values.chunks(32) {
            let mut amax = 0.0f32;
            let mut amin = f32::MAX;
            for &v in chunk {
                amax = amax.max(v);
                amin = amin.min(v);
            }
            let (d, m) = if symmetric {
                (amax / 15.0, 0.0)
            } else {
                ((amax - amin) / 31.0, amin)
            };
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            let off = if symmetric { 16.0 } else { 0.0 };
            out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
            if !symmetric {
                out.extend_from_slice(&F16::from_f32(m).to_bits().to_le_bytes());
            }
            let mut qh: u32 = 0;
            let mut qs = vec![0u8; 16];
            for (j, &v) in chunk.iter().enumerate() {
                let q = (((v - m) * id + off + 0.5) as u32).clamp(0, 31);
                if j < 16 {
                    qs[j] |= (q & 0x0F) as u8;
                    if q & 0x10 != 0 {
                        qh |= 1 << j;
                    }
                } else {
                    qs[j - 16] |= ((q & 0x0F) << 4) as u8;
                    if q & 0x10 != 0 {
                        qh |= 1 << j;
                    }
                }
            }
            out.extend_from_slice(&qh.to_le_bytes());
            out.extend_from_slice(&qs);
        }
        out
    }

    #[test]
    fn avx2_f32_dot_matches_scalar() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return;
        }
        let mut seed = 42u64;
        let w: Vec<f32> = (0..1000).map(|_| lcg(&mut seed)).collect();
        let x: Vec<f32> = (0..1000).map(|_| lcg(&mut seed)).collect();
        let wb: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let scalar = vec_dot_f32(&wb, &x);
        // SAFETY: AVX2+FMA detected above.
        let simd = unsafe { vec_dot_f32_avx2(&wb, &x) };
        assert!(
            (scalar - simd).abs() < 1e-2 * scalar.abs().max(1.0),
            "scalar {scalar} vs simd {simd}"
        );
    }

    #[test]
    fn avx2_q8_0_dot_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut seed = 7u64;
        let w: Vec<f32> = (0..256).map(|_| lcg(&mut seed)).collect();
        let x: Vec<f32> = (0..256).map(|_| lcg(&mut seed)).collect();
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
        let xq = quantize_row_q8_0(&x);
        let scalar = vec_dot_q8_0(&wb, &xq);
        // SAFETY: AVX2 detected above.
        let simd = unsafe { vec_dot_q8_0_avx2(&wb, &xq) };
        assert!(
            (scalar - simd).abs() < 1e-2 * scalar.abs().max(1.0),
            "scalar {scalar} vs simd {simd}"
        );
    }

    #[test]
    fn avx2_q4_0_dot_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut seed = 9u64;
        let w: Vec<f32> = (0..512).map(|_| lcg(&mut seed)).collect();
        let x: Vec<f32> = (0..512).map(|_| lcg(&mut seed)).collect();
        let wb = quantize_row_q4_0(&w);
        let xq = quantize_row_q8_0(&x);
        let scalar = vec_dot_q4_0(&wb, &xq);
        // SAFETY: AVX2 detected above.
        let simd = unsafe { vec_dot_q4_0_avx2(&wb, &xq) };
        assert!(
            (scalar - simd).abs() < 1e-2 * scalar.abs().max(1.0),
            "scalar {scalar} vs simd {simd}"
        );
    }

    #[test]
    fn avx2_q4_1_dot_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut seed = 11u64;
        let w: Vec<f32> = (0..512).map(|_| lcg(&mut seed)).collect();
        let x: Vec<f32> = (0..512).map(|_| lcg(&mut seed)).collect();
        let wb = quantize_row_q4_1(&w);
        let xq = quantize_row_q8_0(&x);
        let scalar = vec_dot_q4_1(&wb, &xq);
        // SAFETY: AVX2 detected above.
        let simd = unsafe { vec_dot_q4_1_avx2(&wb, &xq) };
        assert!(
            (scalar - simd).abs() < 1e-2 * scalar.abs().max(1.0),
            "scalar {scalar} vs simd {simd}"
        );
    }

    #[test]
    fn avx2_q5_0_q5_1_dot_match_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut seed = 13u64;
        let w: Vec<f32> = (0..512).map(|_| lcg(&mut seed)).collect();
        let x: Vec<f32> = (0..512).map(|_| lcg(&mut seed)).collect();

        let wb = q5_pack(&w, true);
        let xq = quantize_row_q8_0(&x);
        let scalar = vec_dot_q5_0(&wb, &xq);
        // SAFETY: AVX2 detected above.
        let simd = unsafe { vec_dot_q5_0_avx2(&wb, &xq) };
        assert!(
            (scalar - simd).abs() < 1e-2 * scalar.abs().max(1.0),
            "q5_0 scalar {scalar} vs simd {simd}"
        );

        let wb = q5_pack(&w, false);
        let scalar = vec_dot_q5_1(&wb, &xq);
        // SAFETY: AVX2 detected above.
        let simd = unsafe { vec_dot_q5_1_avx2(&wb, &xq) };
        assert!(
            (scalar - simd).abs() < 1e-2 * scalar.abs().max(1.0),
            "q5_1 scalar {scalar} vs simd {simd}"
        );
    }

    #[test]
    fn nibble_expansion_order() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let qs: Vec<u8> = (0..16)
            .collect::<Vec<u8>>()
            .iter()
            .map(|&b| (b & 0x0F) | ((15 - (b & 0x0F)) << 4))
            .collect();
        let n = nibbles_32(&qs);
        let mut lanes = [0u8; 32];
        unsafe {
            // SAFETY: `lanes` is a 32-byte local array.
            _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), n);
        }
        for j in 0..16 {
            assert_eq!(lanes[j], qs[j] & 0x0F, "low nibble lane {j}");
            assert_eq!(lanes[j + 16], qs[j] >> 4, "high nibble lane {}", j + 16);
        }
    }
}
