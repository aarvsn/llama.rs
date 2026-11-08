//! Transformer pointwise operations: `RMSNorm`, `RoPE`, softmax, `SwiGLU` and
//! element-wise helpers.

/// Rotary positional embedding pairing style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RotaryStyle {
    /// Pairs `(x[2i], x[2i+1])` — the ggml `GGML_ROPE_TYPE_NORMAL` layout
    /// used by Llama-family GGUF models (HF converters permute Q/K into this
    /// order). This is the default.
    #[default]
    Interleaved,
    /// Pairs `(x[i], x[i + n_rot/2])` — the ggml `GGML_ROPE_TYPE_NEOX`
    /// layout used by e.g. GPT-NeoX style checkpoints.
    HalfSplit,
}

/// Applies `RMSNorm` in-place style: `y = x / rms(x) * w`.
///
/// `x` is read, `y` is written (`y = normalize(x) * weight`). They may be the
/// same buffer.
pub fn rms_norm(x: &[f32], weight: &[f32], y: &mut [f32], eps: f32) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), weight.len());
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    ss /= n as f32;
    ss += eps;
    let scale = 1.0 / ss.sqrt();
    for i in 0..n {
        y[i] = x[i] * scale * weight[i];
    }
}

/// Applies rotary positional embeddings to a single attention head in place.
///
/// Only the first `n_rot` dimensions of `head` are rotated; any remainder is
/// left untouched (partial `RoPE`).
///
/// * `position` is the absolute token position.
/// * `freq_base` is the `RoPE` theta (typically 10000).
pub fn rope_head(
    head: &mut [f32],
    n_rot: usize,
    position: f32,
    freq_base: f32,
    style: RotaryStyle,
) {
    let n = n_rot.min(head.len());
    debug_assert!(n % 2 == 0, "rotary dimensions must be even");
    let inv_base = -2.0 / n as f32;
    match style {
        RotaryStyle::Interleaved => {
            let mut theta = 1.0f32; // freq_base^0
            let step = freq_base.powf(inv_base);
            let mut i = 0;
            while i < n {
                let angle = position * theta;
                let (s, c) = angle.sin_cos();
                let x0 = head[i];
                let x1 = head[i + 1];
                head[i] = x0 * c - x1 * s;
                head[i + 1] = x0 * s + x1 * c;
                theta *= step;
                i += 2;
            }
        }
        RotaryStyle::HalfSplit => {
            let half = n / 2;
            let mut theta = 1.0f32;
            let step = freq_base.powf(inv_base);
            for i in 0..half {
                let angle = position * theta;
                let (s, c) = angle.sin_cos();
                let x0 = head[i];
                let x1 = head[i + half];
                head[i] = x0 * c - x1 * s;
                head[i + half] = x0 * s + x1 * c;
                theta *= step;
            }
        }
    }
}

/// Numerically stable in-place softmax over `v`.
pub fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &x in v.iter() {
        max = max.max(x);
    }
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// `SiLU` activation in place: `x * sigmoid(x)`.
pub fn silu_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = *x / (1.0 + (-*x).exp());
    }
}

/// Element-wise multiply: `y[i] *= x[i]`.
pub fn mul_inplace(y: &mut [f32], x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    for (a, b) in y.iter_mut().zip(x) {
        *a *= b;
    }
}

/// Element-wise add: `y[i] += x[i]`.
pub fn add_inplace(y: &mut [f32], x: &[f32]) {
    debug_assert_eq!(y.len(), x.len());
    for (a, b) in y.iter_mut().zip(x) {
        *a += b;
    }
}

/// Scales a vector: `y[i] *= s`.
pub fn scale_inplace(y: &mut [f32], s: f32) {
    for v in y.iter_mut() {
        *v *= s;
    }
}

/// Index of the maximum element.
#[must_use]
pub fn argmax(v: &[f32]) -> usize {
    debug_assert_ne!(v, []);
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_reference() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0; 4];
        let mut y = [0.0; 4];
        rms_norm(&x, &w, &mut y, 1e-6);
        let ms = (x.iter().map(|v| v * v).sum::<f32>() / 4.0 + 1e-6).sqrt();
        for i in 0..4 {
            let expected = x[i] / ms;
            assert!((y[i] - expected).abs() < 1e-5, "{} vs {}", y[i], expected);
        }
    }

    #[test]
    fn rms_norm_with_weights() {
        let x = [3.0, 4.0];
        let w = [2.0, 0.5];
        let mut y = [0.0; 2];
        rms_norm(&x, &w, &mut y, 0.0);
        let rms = 5.0 / 2.0f32.sqrt(); // sqrt((9+16)/2)
        assert!((y[0] - 3.0 / rms * 2.0).abs() < 1e-5);
        assert!((y[1] - 4.0 / rms * 0.5).abs() < 1e-5);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = [1.0, 2.0, 3.0, 100.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(v[3] > 0.999);
    }

    #[test]
    fn softmax_handles_extremes() {
        let mut v = [-1e30, -1e30, -1e30];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        let mut big = [1e30, 0.0];
        softmax_inplace(&mut big);
        assert!((big[0] - 1.0).abs() < 1e-6);
        assert!(big[1].abs() < 1e-6);
    }

    #[test]
    fn silu_values() {
        let mut v = [0.0, 1.0, -1.0];
        silu_inplace(&mut v);
        assert!(v[0].abs() < 1e-7);
        assert!((v[1] - 0.731_058_6).abs() < 1e-5);
        assert!((v[2] + 0.268_941_43).abs() < 1e-5);
    }

    #[test]
    fn rope_interleaved_reference() {
        // head = [1, 0, 1, 0], position 1, freq_base huge so only pair 0
        // rotates (theta_0 = 1, theta_1 ~ 0).
        let mut head = [1.0, 0.0, 1.0, 0.0];
        rope_head(&mut head, 4, 1.0, 1e12, RotaryStyle::Interleaved);
        let (s, c) = (1.0f32.sin(), 1.0f32.cos());
        assert!((head[0] - c).abs() < 1e-4);
        assert!((head[1] - s).abs() < 1e-4);
        assert!((head[2] - 1.0).abs() < 1e-4);
        assert!(head[3].abs() < 1e-4);
    }

    #[test]
    fn rope_half_split_reference() {
        // head = [1, 1, 0, 0]: pair (0, 2) rotates by 1 rad; pair (1, 3) has
        // theta ~ 1e-6 so it barely moves.
        let mut head = [1.0, 1.0, 0.0, 0.0];
        rope_head(&mut head, 4, 1.0, 1e12, RotaryStyle::HalfSplit);
        let (s, c) = (1.0f32.sin(), 1.0f32.cos());
        assert!((head[0] - c).abs() < 1e-4, "head[0] = {}", head[0]);
        assert!((head[2] - s).abs() < 1e-4, "head[2] = {}", head[2]);
        assert!((head[1] - 1.0).abs() < 1e-4, "head[1] = {}", head[1]);
        assert!(head[3].abs() < 1e-4, "head[3] = {}", head[3]);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let mut head = [0.5, -1.25, 3.0, 0.75, -0.1, 2.2];
        let orig = head;
        rope_head(&mut head, 6, 0.0, 10000.0, RotaryStyle::Interleaved);
        for (a, b) in head.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_preserves_norm() {
        let mut head = [0.3, -1.7, 2.4, 0.9, -0.6, 1.1, 0.2, -0.8];
        let norm: f32 = head.iter().map(|v| v * v).sum::<f32>().sqrt();
        rope_head(&mut head, 8, 123.0, 10000.0, RotaryStyle::Interleaved);
        let norm2: f32 = head.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - norm2).abs() < 1e-4, "{norm} vs {norm2}");
    }

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 5.0, 2.0]), 1);
        assert_eq!(argmax(&[-1.0, -5.0, -2.0]), 0);
    }
}
