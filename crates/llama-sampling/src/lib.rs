//! # llama-sampling
//!
//! Token sampling strategies with deterministic, seeded randomness.
//!
//! Supported knobs (all composable):
//!
//! * greedy (argmax)
//! * temperature
//! * top-k
//! * top-p (nucleus)
//! * min-p
//! * repetition penalty
//! * frequency & presence penalties
//! * fixed random seed for reproducible sampling
//!
//! The RNG is a self-contained xoshiro256** generator so identical seeds
//! produce identical generations on every platform.

pub mod rng;
pub mod sampler;

pub use rng::Xoshiro256;
pub use sampler::{SampleResult, Sampler, SamplerParams};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_with_same_seed() {
        let mut s1 = Xoshiro256::seed_from_u64(42);
        let mut s2 = Xoshiro256::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(s1.next_u64(), s2.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut s1 = Xoshiro256::seed_from_u64(1);
        let mut s2 = Xoshiro256::seed_from_u64(2);
        let differ = (0..16).any(|_| s1.next_u64() != s2.next_u64());
        assert!(differ);
    }
}
