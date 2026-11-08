//! Per-layer key/value cache for a single generation sequence.

use llama_core::F16;

/// KV cache storage precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    /// 32-bit floats (default).
    F32,
    /// 16-bit floats (half the memory).
    F16,
}

/// The key/value cache for all layers of one sequence.
///
/// Layout: `[layer][position][kv_head][dim]`, allocated once for the full
/// context window so that decoding never reallocates.
pub struct KvCache {
    n_layers: usize,
    n_ctx: usize,
    n_kv_heads: usize,
    head_dim: usize,
    k_f32: Vec<f32>,
    v_f32: Vec<f32>,
    k_f16: Vec<u16>,
    v_f16: Vec<u16>,
    precision: KvPrecision,
    len: usize,
}

impl KvCache {
    /// Allocates a cache.
    #[must_use]
    pub fn new(
        n_layers: usize,
        n_ctx: usize,
        n_kv_heads: usize,
        head_dim: usize,
        precision: KvPrecision,
    ) -> Self {
        let total = n_layers * n_ctx * n_kv_heads * head_dim;
        let (k_f32, v_f32, k_f16, v_f16) = match precision {
            KvPrecision::F32 => (vec![0.0; total], vec![0.0; total], Vec::new(), Vec::new()),
            KvPrecision::F16 => (Vec::new(), Vec::new(), vec![0; total], vec![0; total]),
        };
        Self {
            n_layers,
            n_ctx,
            n_kv_heads,
            head_dim,
            k_f32,
            v_f32,
            k_f16,
            v_f16,
            precision,
            len: 0,
        }
    }

    /// Number of cached positions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Context capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.n_ctx
    }

    /// Truncates the cache to `n` positions (prefix reuse).
    pub fn truncate(&mut self, n: usize) {
        self.len = n.min(self.len);
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Bytes of memory used by the cache.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let bytes_per = match self.precision {
            KvPrecision::F32 => 8,
            KvPrecision::F16 => 4,
        };
        self.n_layers * self.n_ctx * self.n_kv_heads * self.head_dim * bytes_per
    }
}

impl KvCache {
    fn offset(&self, layer: usize, pos: usize, head: usize) -> usize {
        ((layer * self.n_ctx + pos) * self.n_kv_heads + head) * self.head_dim
    }

    /// Reads the key vector of (layer, pos, head).
    ///
    /// For f16 storage this materializes into `scratch` and returns it.
    #[must_use]
    pub fn k(&self, layer: usize, pos: usize, head: usize) -> CowKey<'_> {
        let off = self.offset(layer, pos, head);
        match self.precision {
            KvPrecision::F32 => CowKey::F32(&self.k_f32[off..off + self.head_dim]),
            KvPrecision::F16 => CowKey::F16(&self.k_f16[off..off + self.head_dim]),
        }
    }

    /// Reads the value vector of (layer, pos, head).
    #[must_use]
    pub fn v(&self, layer: usize, pos: usize, head: usize) -> CowKey<'_> {
        let off = self.offset(layer, pos, head);
        match self.precision {
            KvPrecision::F32 => CowKey::F32(&self.v_f32[off..off + self.head_dim]),
            KvPrecision::F16 => CowKey::F16(&self.v_f16[off..off + self.head_dim]),
        }
    }

    /// Writes the key vector for (layer, pos, head).
    pub fn set_k(&mut self, layer: usize, pos: usize, head: usize, value: &[f32]) {
        debug_assert_eq!(value.len(), self.head_dim);
        let off = self.offset(layer, pos, head);
        match self.precision {
            KvPrecision::F32 => self.k_f32[off..off + self.head_dim].copy_from_slice(value),
            KvPrecision::F16 => {
                for (slot, &v) in self.k_f16[off..off + self.head_dim].iter_mut().zip(value) {
                    *slot = F16::from_f32(v).to_bits();
                }
            }
        }
    }

    /// Writes the value vector for (layer, pos, head).
    pub fn set_v(&mut self, layer: usize, pos: usize, head: usize, value: &[f32]) {
        debug_assert_eq!(value.len(), self.head_dim);
        let off = self.offset(layer, pos, head);
        match self.precision {
            KvPrecision::F32 => self.v_f32[off..off + self.head_dim].copy_from_slice(value),
            KvPrecision::F16 => {
                for (slot, &v) in self.v_f16[off..off + self.head_dim].iter_mut().zip(value) {
                    *slot = F16::from_f32(v).to_bits();
                }
            }
        }
    }

    /// Marks positions `[len, len + n)` as used after the caller wrote them.
    pub fn advance(&mut self, n: usize) {
        debug_assert!(
            self.len + n <= self.n_ctx,
            "kv overflow: len={} + n={} > ctx={}",
            self.len,
            n,
            self.n_ctx
        );
        self.len += n;
    }
}

/// A borrowed key/value vector that is either f32 storage or f16 storage.
pub enum CowKey<'a> {
    /// Direct f32 slice.
    F32(&'a [f32]),
    /// f16-bit slice (convert on demand).
    F16(&'a [u16]),
}

impl CowKey<'_> {
    /// The i-th component.
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize) -> f32 {
        match self {
            Self::F32(s) => s[i],
            Self::F16(s) => F16::from_bits(s[i]).to_f32(),
        }
    }

    /// Vector length.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::F32(s) => s.len(),
            Self::F16(s) => s.len(),
        }
    }

    /// Whether empty (always false for real caches).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_round_trip() {
        let mut kv = KvCache::new(2, 8, 2, 4, KvPrecision::F32);
        let vals = [1.0, 2.0, 3.0, 4.0];
        kv.set_k(1, 3, 0, &vals);
        kv.set_v(1, 3, 1, &vals);
        kv.advance(4);
        assert_eq!(kv.len(), 4);
        assert_eq!(kv.k(1, 3, 0).get(2), 3.0);
        assert_eq!(kv.v(1, 3, 1).get(0), 1.0);
    }

    #[test]
    fn f16_round_trip() {
        let mut kv = KvCache::new(1, 4, 1, 2, KvPrecision::F16);
        kv.set_k(0, 0, 0, &[0.5, -1.25]);
        kv.advance(1);
        assert_eq!(kv.k(0, 0, 0).get(0), 0.5);
        assert_eq!(kv.k(0, 0, 0).get(1), -1.25);
        assert!(kv.memory_bytes() < KvCache::new(1, 4, 1, 2, KvPrecision::F32).memory_bytes());
    }

    #[test]
    fn truncate_for_prefix_reuse() {
        let mut kv = KvCache::new(1, 8, 1, 2, KvPrecision::F32);
        kv.advance(6);
        kv.truncate(3);
        assert_eq!(kv.len(), 3);
        kv.truncate(10); // never grows
        assert_eq!(kv.len(), 3);
        kv.clear();
        assert!(kv.is_empty());
    }
}
