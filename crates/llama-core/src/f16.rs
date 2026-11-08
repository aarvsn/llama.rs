//! IEEE 754 half-precision (`binary16`) support implemented natively.
//!
//! GGUF quantized blocks store their per-block scales as `f16`, so the engine
//! needs fast and exact conversions. Instead of pulling in an external crate
//! we implement the conversion here with well-known bit manipulation:
//!
//! * `f16 -> f32` is lossless (a `binary16` value is exactly representable in
//!   `binary32`).
//! * `f32 -> f16` rounds to nearest, ties to even, matching the behaviour of
//!   hardware F16C instructions and the reference `ggml` tables.

use core::num::FpCategory;
use std::fmt;

/// An IEEE 754 `binary16` value stored as raw bits.
///
/// The bits are always kept in native (little-endian layout on all supported
/// platforms) form; GGUF files are little-endian and the loaders convert
/// explicitly, so the type simply wraps a `u16`.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct F16(u16);

impl F16 {
    /// Positive zero.
    pub const ZERO: Self = Self(0x0000);
    /// Negative zero.
    pub const NEG_ZERO: Self = Self(0x8000);
    /// Positive infinity.
    pub const INFINITY: Self = Self(0x7C00);
    /// Not a number (quiet).
    pub const NAN: Self = Self(0x7E00);

    /// Constructs from raw bits.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bits.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Lossless conversion to `f32`.
    #[inline]
    #[must_use]
    pub fn to_f32(self) -> f32 {
        let sign = ((self.0 & 0x8000) as u32) << 16;
        let exp = ((self.0 >> 10) & 0x1F) as u32;
        let mant = (self.0 & 0x03FF) as u32;

        let bits = match exp {
            0 => {
                // Subnormal or zero: normalize the 10-bit significand.
                if mant == 0 {
                    sign
                } else {
                    // k = leading zeros of `mant` inside the 10-bit field.
                    let k = 10 - (32 - mant.leading_zeros());
                    // Normalized value = (mant << k) * 2^(-24-k) = 1.f * 2^(-15-k),
                    // so the f32 biased exponent is 112 - k.
                    let norm_exp = 112 - k;
                    debug_assert!(
                        (1..=112).contains(&norm_exp),
                        "f16 subnormal exponent out of range"
                    );
                    // Keep the 9 fraction bits below the (implicit) leading 1.
                    let m = ((mant << k) & 0x01FF) << 14;
                    sign | (norm_exp << 23) | m
                }
            }
            0x1F => {
                // Inf / NaN.
                sign | 0x7F80_0000 | (mant << 13)
            }
            _ => sign | ((exp + 112) << 23) | (mant << 13),
        };
        f32::from_bits(bits)
    }

    /// Converts `f32` to `f16` with round-to-nearest, ties-to-even.
    ///
    /// Overflow (including infinity) saturates to infinity; NaN maps to a
    /// quiet NaN.
    #[inline]
    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        let x = value.to_bits();

        let sign = ((x >> 16) & 0x8000) as u16;
        let exp = ((x >> 23) & 0xFF) as i32;
        let mant = x & 0x007F_FFFF;

        if exp == 0xFF {
            // Inf or NaN: preserve the top payload bits of NaNs.
            return Self(sign | 0x7C00 | if mant != 0 { (mant >> 13) as u16 } else { 0 });
        }

        // Unbiased exponent with f16 bias applied.
        let half_exp = exp - 127 + 15;

        if half_exp >= 0x1F {
            // Overflow -> infinity.
            return Self(sign | 0x7C00);
        }
        if half_exp <= 0 {
            // Subnormal or zero in f16.
            if half_exp < -10 {
                // Smaller than half of the smallest subnormal quantum.
                return Self(sign);
            }
            // value / 2^-24 = m >> (14 - half_exp) with the usual sticky bits
            // for round-to-nearest-even, where m is the 24-bit f32 significand.
            let m = mant | 0x0080_0000;
            let shift = (14 - half_exp) as u32; // in 14..=24
            let mut units = m >> shift;
            let rem = m & ((1u32 << shift) - 1);
            let half = 1u32 << (shift - 1);
            if rem > half || (rem == half && (units & 1) == 1) {
                units += 1;
            }
            // Rounding may carry into the exponent field, which is exactly the
            // correct encoding for the smallest normal number (units == 1024).
            return Self(sign | units as u16);
        }

        // Normal f16.
        let mut m = mant >> 13;
        let rem = mant & 0x1FFF;
        if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
            m += 1;
        }
        let mut out = sign | ((half_exp as u16) << 10) | m as u16;
        if m == 0x400 {
            // Rounded up to the next exponent.
            out = sign | (((half_exp + 1) as u16) << 10);
        }
        Self(out)
    }

    /// Classifies the value.
    #[inline]
    #[must_use]
    pub fn classify(self) -> FpCategory {
        match (self.0 >> 10) & 0x1F {
            0 => {
                if self.0 & 0x03FF == 0 {
                    FpCategory::Zero
                } else {
                    FpCategory::Subnormal
                }
            }
            0x1F => {
                if self.0 & 0x03FF == 0 {
                    FpCategory::Infinite
                } else {
                    FpCategory::Nan
                }
            }
            _ => FpCategory::Normal,
        }
    }

    /// Returns true for NaN.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        matches!(self.classify(), FpCategory::Nan)
    }
}

impl From<F16> for f32 {
    #[inline]
    fn from(v: F16) -> f32 {
        v.to_f32()
    }
}

impl From<f32> for F16 {
    #[inline]
    fn from(v: f32) -> F16 {
        F16::from_f32(v)
    }
}

impl From<F16> for f64 {
    #[inline]
    fn from(v: F16) -> f64 {
        f64::from(v.to_f32())
    }
}

impl fmt::Debug for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F16({})", self.to_f32())
    }
}

impl PartialOrd for F16 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        let a = self.to_f32();
        let b = other.to_f32();
        a.partial_cmp(&b)
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_inf() {
        assert_eq!(F16::ZERO.to_f32(), 0.0);
        assert_eq!(F16::NEG_ZERO.to_f32(), -0.0);
        assert_eq!(F16::INFINITY.to_f32(), f32::INFINITY);
        assert!(F16::NAN.to_f32().is_nan());
    }

    #[test]
    fn round_trip_common_values() {
        let values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            2.0,
            0.333_251_953_125, // closest f16 to 1/3
            65504.0,           // largest normal f16
            6.1035156e-5,      // smallest normal f16
            5.9604645e-8,      // smallest subnormal f16
            -42.75,
            2048.0,
        ];
        for v in values {
            let h = F16::from_f32(v);
            let back = h.to_f32();
            assert!(
                (back - v).abs() <= f32::EPSILON * v.abs().max(1e-30),
                "roundtrip failed for {v} (got {back})"
            );
        }
    }

    #[test]
    fn known_bit_patterns() {
        assert_eq!(F16::from_f32(1.0).to_bits(), 0x3C00);
        assert_eq!(F16::from_f32(-2.0).to_bits(), 0xC000);
        assert_eq!(F16::from_f32(0.5).to_bits(), 0x3800);
        assert_eq!(F16::from_f32(65504.0).to_bits(), 0x7BFF);
        assert!((F16::from_bits(0x3555).to_f32() - 0.333_251_95).abs() < 1e-9);
    }

    #[test]
    fn rounding_ties_to_even() {
        // At [2048, 4096) the f16 mantissa quantum is 2: odd integers are
        // halfway cases.
        assert_eq!(F16::from_f32(2049.0).to_f32(), 2048.0);
        assert_eq!(F16::from_f32(2051.0).to_f32(), 2052.0);
        // Even integers are exactly representable here.
        assert_eq!(F16::from_f32(2050.0).to_f32(), 2050.0);
        assert_eq!(F16::from_f32(2052.0).to_f32(), 2052.0);
    }

    #[test]
    fn overflow_saturates_to_infinity() {
        assert_eq!(F16::from_f32(65520.0).to_f32(), f32::INFINITY);
        assert_eq!(F16::from_f32(1e30).to_f32(), f32::INFINITY);
        assert_eq!(F16::from_f32(-1e30).to_f32(), f32::NEG_INFINITY);
    }

    #[test]
    fn exhaustively_compare_subnormal_and_normal_round_trip() {
        // Every f16 -> f32 is exact; every such f32 -> f16 must be the identity.
        let mut bits = 0u16;
        loop {
            let h = F16::from_bits(bits);
            let back = F16::from_f32(h.to_f32());
            assert_eq!(h, back, "mismatch at bits {bits:#06x}");
            bits = bits.wrapping_add(1);
            if bits == 0 {
                break;
            }
        }
    }

    #[test]
    fn subnormal_conversion_matches_expectation() {
        // 0x0400 is the smallest normal (6.103515625e-5).
        assert_eq!(F16::from_bits(0x0400).to_f32(), 6.103_515_625e-5);
        // 0x0001 is the smallest subnormal (5.960464477539063e-8).
        assert_eq!(F16::from_bits(0x0001).to_f32(), 5.960_464_477_539_063e-8);
        // 0x03FF is the largest subnormal.
        assert_eq!(F16::from_bits(0x03FF).to_f32(), 6.097_555_16e-5);
    }
}
