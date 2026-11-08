//! The sampling pipeline: penalties -> temperature -> top-k -> top-p -> min-p
//! -> sample (or argmax for greedy decoding).

use llama_core::TokenId;

use crate::rng::Xoshiro256;

/// Sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplerParams {
    /// Greedy decoding (argmax) — ignores every other knob.
    pub greedy: bool,
    /// Softmax temperature (1.0 = unchanged; 0.x sharpens).
    pub temperature: f32,
    /// Keep only the `k` highest-probability tokens (0 = disabled).
    pub top_k: usize,
    /// Keep the smallest prefix with cumulative probability >= `top_p`
    /// (1.0 = disabled).
    pub top_p: f32,
    /// Keep tokens with probability >= `min_p` * `max_probability`
    /// (0.0 = disabled).
    pub min_p: f32,
    /// Repetition penalty (1.0 = disabled): divides logits of tokens that
    /// already appeared (negative logits are multiplied instead).
    pub repeat_penalty: f32,
    /// Frequency penalty: subtracted once per prior occurrence.
    pub frequency_penalty: f32,
    /// Presence penalty: subtracted once if the token occurred at all.
    pub presence_penalty: f32,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            greedy: false,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

impl SamplerParams {
    /// Validates the parameters.
    ///
    /// # Errors
    ///
    /// Returns an error for out-of-range values instead of silently clamping.
    pub fn validate(&self) -> Result<(), String> {
        if self.temperature < 0.0 {
            return Err("temperature must be >= 0".into());
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err("top_p must be in (0, 1]".into());
        }
        if self.min_p < 0.0 || self.min_p > 1.0 {
            return Err("min_p must be in [0, 1]".into());
        }
        if self.repeat_penalty <= 0.0 {
            return Err("repeat_penalty must be > 0".into());
        }
        if self.frequency_penalty < -2.0 || self.frequency_penalty > 2.0 {
            return Err("frequency_penalty must be in [-2, 2]".into());
        }
        if self.presence_penalty < -2.0 || self.presence_penalty > 2.0 {
            return Err("presence_penalty must be in [-2, 2]".into());
        }
        Ok(())
    }
}

/// A configured sampler with its own RNG state.
pub struct Sampler {
    params: SamplerParams,
    rng: Xoshiro256,
}

/// The outcome of one sampling step.
#[derive(Debug, Clone, Copy)]
pub struct SampleResult {
    /// The selected token.
    pub token: TokenId,
}

impl Sampler {
    /// Creates a sampler with the given parameters and seed.
    #[must_use]
    pub fn new(params: SamplerParams, seed: u64) -> Self {
        Self {
            params,
            rng: Xoshiro256::seed_from_u64(seed),
        }
    }

    /// Creates a sampler with entropy seeding.
    #[must_use]
    pub fn from_entropy(params: SamplerParams) -> Self {
        Self {
            params,
            rng: Xoshiro256::from_entropy(),
        }
    }

    /// The parameters in use.
    #[must_use]
    pub fn params(&self) -> &SamplerParams {
        &self.params
    }

    /// Applies the repetition/frequency/presence penalties to a copy of the
    /// logits, using the token counts of the current context.
    ///
    /// Only tokens present in `counts` are touched, so the cost is
    /// proportional to the context, not the vocabulary.
    // 1.0 / 0.0 are exact "penalty disabled" sentinels, so `==` is intended.
    #[allow(clippy::float_cmp)]
    #[must_use]
    pub fn apply_penalties(
        &self,
        logits: &[f32],
        counts: &std::collections::HashMap<TokenId, usize>,
    ) -> Vec<f32> {
        let p = &self.params;
        if p.repeat_penalty == 1.0 && p.frequency_penalty == 0.0 && p.presence_penalty == 0.0 {
            return logits.to_vec();
        }
        let mut out = logits.to_vec();
        for (&tok, &freq) in counts {
            let idx = tok as usize;
            if idx >= out.len() {
                continue;
            }
            // Repetition penalty: divide positive logits, multiply negative.
            if p.repeat_penalty != 1.0 {
                out[idx] = if out[idx] > 0.0 {
                    out[idx] / p.repeat_penalty
                } else {
                    out[idx] * p.repeat_penalty
                };
            }
            out[idx] -= p.frequency_penalty * freq as f32;
            out[idx] -= p.presence_penalty;
        }
        out
    }

    /// Samples the next token from (possibly penalized) logits.
    #[must_use]
    pub fn sample(&mut self, logits: &[f32]) -> SampleResult {
        let p = &self.params;
        if p.greedy || p.temperature == 0.0 {
            let best = argmax(logits);
            return SampleResult {
                token: best as TokenId,
            };
        }

        // Temperature.
        let scaled: Vec<f32> = logits.iter().map(|&l| l / p.temperature).collect();

        // Softmax (stable).
        let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let mut probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

        // Build the candidate list sorted by descending probability.
        let mut order: Vec<u32> = (0..probs.len() as u32).collect();
        order.sort_by(|&a, &b| {
            probs[b as usize]
                .partial_cmp(&probs[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Top-k.
        let mut keep = order.len();
        if p.top_k > 0 && p.top_k < keep {
            keep = p.top_k;
        }

        // Top-p (nucleus) over the sorted candidates.
        let mut cumulative = 0.0f32;
        let mut top_p_cut = keep;
        for (rank, &tok) in order.iter().take(keep).enumerate() {
            cumulative += probs[tok as usize];
            if cumulative >= p.top_p {
                top_p_cut = rank + 1;
                break;
            }
        }
        keep = top_p_cut.min(keep);

        // Min-p relative to the best candidate.
        if p.min_p > 0.0 {
            let best_prob = probs[order[0] as usize];
            let threshold = p.min_p * best_prob;
            let mut min_p_cut = 0;
            for (rank, &tok) in order.iter().take(keep).enumerate() {
                if probs[tok as usize] >= threshold {
                    min_p_cut = rank + 1;
                } else {
                    break;
                }
            }
            keep = min_p_cut.max(1).min(keep);
        }

        // Renormalize the kept candidates and sample.
        let kept: Vec<u32> = order[..keep].to_vec();
        let total: f32 = kept.iter().map(|&t| probs[t as usize]).sum();
        if total <= 0.0 || !total.is_finite() {
            return SampleResult {
                token: argmax(logits) as TokenId,
            };
        }
        let mut pick = self.rng.next_f32() * total;
        let mut chosen = kept[0];
        for &tok in &kept {
            pick -= probs[tok as usize];
            if pick <= 0.0 {
                chosen = tok;
                break;
            }
        }

        // Zero out the rest (kept for potential debug output).
        let _ = &mut probs;
        SampleResult { token: chosen }
    }
}

fn argmax(v: &[f32]) -> usize {
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

    fn flat_logits(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.1).collect()
    }

    #[test]
    fn greedy_picks_max() {
        let mut s = Sampler::new(
            SamplerParams {
                greedy: true,
                ..SamplerParams::default()
            },
            1,
        );
        let logits = vec![0.0, 3.0, 1.0, 2.0];
        assert_eq!(s.sample(&logits).token, 1);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let mut s = Sampler::new(
            SamplerParams {
                temperature: 0.0,
                ..SamplerParams::default()
            },
            1,
        );
        assert_eq!(s.sample(&[0.0, 5.0, 1.0]).token, 1);
    }

    #[test]
    fn same_seed_same_sequence() {
        let params = SamplerParams::default();
        let logits = flat_logits(100);
        let mut a = Sampler::new(params.clone(), 123);
        let mut b = Sampler::new(params, 123);
        for _ in 0..20 {
            assert_eq!(a.sample(&logits).token, b.sample(&logits).token);
        }
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s = Sampler::new(
            SamplerParams {
                top_k: 1,
                temperature: 1.0,
                ..SamplerParams::default()
            },
            5,
        );
        let logits = vec![10.0, 9.0, 0.0];
        assert_eq!(s.sample(&logits).token, 0);
    }

    #[test]
    fn top_p_one_keeps_all() {
        let mut s = Sampler::new(
            SamplerParams {
                top_p: 1.0,
                top_k: 0,
                min_p: 0.0,
                temperature: 1.0,
                ..SamplerParams::default()
            },
            5,
        );
        let logits = vec![0.0; 10];
        // Uniform: every token is possible; just check it does not panic and
        // returns a valid id.
        assert!(s.sample(&logits).token < 10);
    }

    #[test]
    fn min_p_filters_tail() {
        // Construct logits where one token dominates.
        let mut s = Sampler::new(
            SamplerParams {
                min_p: 0.5,
                top_k: 0,
                top_p: 1.0,
                temperature: 1.0,
                ..SamplerParams::default()
            },
            5,
        );
        let logits = vec![10.0, 0.0, 0.0];
        for _ in 0..50 {
            assert_eq!(s.sample(&logits).token, 0);
        }
    }

    #[test]
    fn repetition_penalty_divides_positive_logits() {
        let s = Sampler::new(SamplerParams::default(), 1);
        let mut counts = std::collections::HashMap::new();
        counts.insert(1u32, 3usize);
        let penalized = s.apply_penalties(&[2.0, 2.0], &counts);
        let p = SamplerParams::default();
        let expected = 2.0 / p.repeat_penalty - 3.0 * p.frequency_penalty - p.presence_penalty;
        assert!((penalized[1] - expected).abs() < 1e-5);
        assert_eq!(penalized[0], 2.0);
    }

    #[test]
    fn params_validation() {
        assert!(SamplerParams::default().validate().is_ok());
        assert!(
            SamplerParams {
                temperature: -1.0,
                ..SamplerParams::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            SamplerParams {
                top_p: 1.5,
                ..SamplerParams::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn penalties_noop_when_disabled() {
        let s = Sampler::new(
            SamplerParams {
                repeat_penalty: 1.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                ..SamplerParams::default()
            },
            1,
        );
        let mut counts = std::collections::HashMap::new();
        counts.insert(0u32, 5usize);
        let out = s.apply_penalties(&[1.0, -2.0], &counts);
        assert_eq!(out, vec![1.0, -2.0]);
    }

    #[test]
    fn extreme_logits_do_not_panic() {
        let mut s = Sampler::new(SamplerParams::default(), 3);
        let _ = s.sample(&[f32::NEG_INFINITY; 8]);
        let _ = s.sample(&[f32::INFINITY, 0.0, 0.0]);
        let _ = s.sample(&[f32::NAN, 1.0]);
    }
}
