//! The Llama-family decoder-only transformer forward pass.
//!
//! Layout notes (validated against llama.cpp):
//!
//! * Weight matrices are GGUF `[ne0 = n_in, ne1 = n_out]`: `n_out` contiguous
//!   rows of `n_in` elements; the product is `y = x · Wᵀ`.
//! * `RoPE` uses the *interleaved* pairing `(x[2i], x[2i+1])` — the
//!   `GGML_ROPE_TYPE_NORMAL` layout that HF→GGUF converters permute Q/K into.
//! * GQA: query head `h` attends with KV head `h / n_gqa`.
//!
//! One-D norm weights are dequantized once at load time; everything else is
//! consumed zero-copy from the memory map.

use std::collections::HashMap;
use std::sync::Arc;

use llama_core::{LlamaError, Result, TokenId};
use llama_gguf::{GgufFile, TensorInfo};
use llama_kernels::dispatch::Activation;
use llama_kernels::ops::{self, RotaryStyle};
use llama_kernels::quant::quantize_row_q8_0_into;
use llama_tensor::{WeightMatrix, matmul_rows};

use crate::config::LlamaConfig;
use crate::kv::KvCache;

/// A loaded Llama model: configuration plus zero-copy weight views over the
/// memory-mapped GGUF file.
pub struct LlamaModel {
    config: LlamaConfig,
    file: Arc<GgufFile>,
    /// tensor name -> index into `infos`
    index: HashMap<String, usize>,
    infos: Vec<TensorInfo>,
    /// Per-layer attention norm weights (f32, dequantized at load).
    attn_norms: Vec<Vec<f32>>,
    /// Per-layer FFN norm weights (f32).
    ffn_norms: Vec<Vec<f32>>,
    /// Final norm weight (f32).
    output_norm: Vec<f32>,
}

impl LlamaModel {
    /// Loads a Llama model from a GGUF file, validating every tensor the
    /// forward pass needs.
    pub fn load(file: Arc<GgufFile>, mut config: LlamaConfig) -> Result<Self> {
        let mut index = HashMap::new();
        let mut infos = Vec::new();
        for (i, t) in file.tensors().iter().enumerate() {
            index.insert(t.name.clone(), i);
            infos.push(t.clone());
        }
        config.n_vocab = infos
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .map_or(0, |t| t.dims[1] as usize);

        let mut model = Self {
            config,
            file,
            index,
            infos,
            attn_norms: Vec::new(),
            ffn_norms: Vec::new(),
            output_norm: Vec::new(),
        };
        model.validate()?;
        model.load_norms()?;
        Ok(model)
    }

    fn validate(&self) -> Result<()> {
        let c = &self.config;

        if c.n_heads * c.head_dim_v != c.n_embd {
            return Err(LlamaError::InvalidMetadata(format!(
                "n_heads ({}) * value head dim ({}) must equal n_embd ({})",
                c.n_heads, c.head_dim_v, c.n_embd
            )));
        }

        let emb = self.tensor_info("token_embd.weight")?;
        if emb.dims.len() != 2 || emb.dims[0] as usize != c.n_embd {
            return Err(LlamaError::InvalidTensor(format!(
                "token_embd.weight has shape {:?}, expected [n_embd={}, vocab]",
                emb.dims, c.n_embd
            )));
        }

        let check = |name: &str, n_in: usize, n_out: usize| -> Result<()> {
            let info = self.tensor_info(name)?;
            let expected: Vec<u64> = if n_out == 1 {
                vec![n_in as u64] // 1-D vector (norm weights)
            } else {
                vec![n_in as u64, n_out as u64]
            };
            if info.dims != expected {
                return Err(LlamaError::InvalidTensor(format!(
                    "{name} has shape {:?}, expected {expected:?}",
                    info.dims
                )));
            }
            Ok(())
        };

        let q_len = c.n_heads * c.head_dim_k;
        let kv_len = c.n_kv_heads * c.head_dim_k;
        let v_len = c.n_kv_heads * c.head_dim_v;
        for layer in 0..c.n_layers {
            let p = format!("blk.{layer}.");
            check(&format!("{p}attn_norm.weight"), c.n_embd, 1)?;
            check(&format!("{p}attn_q.weight"), c.n_embd, q_len)?;
            check(&format!("{p}attn_k.weight"), c.n_embd, kv_len)?;
            check(&format!("{p}attn_v.weight"), c.n_embd, v_len)?;
            check(&format!("{p}attn_output.weight"), c.n_embd, c.n_embd)?;
            check(&format!("{p}ffn_norm.weight"), c.n_embd, 1)?;
            check(&format!("{p}ffn_gate.weight"), c.n_embd, c.n_ff)?;
            check(&format!("{p}ffn_up.weight"), c.n_embd, c.n_ff)?;
            check(&format!("{p}ffn_down.weight"), c.n_ff, c.n_embd)?;
        }
        check("output_norm.weight", c.n_embd, 1)?;
        let _ = &self.index;
        if let Some(out) = self.index.get("output.weight") {
            let info = &self.infos[*out];
            if info.dims.len() != 2
                || info.dims[0] as usize != c.n_embd
                || info.dims[1] as usize != c.n_vocab
            {
                return Err(LlamaError::InvalidTensor(format!(
                    "output.weight has shape {:?}, expected [{}, {}]",
                    info.dims, c.n_embd, c.n_vocab
                )));
            }
        }
        Ok(())
    }

    /// Dequantizes the 1-D norm weights once.
    fn load_norms(&mut self) -> Result<()> {
        let n_layers = self.config.n_layers;
        let n_embd = self.config.n_embd;
        let mut attn_norms = vec![Vec::new(); n_layers];
        let mut ffn_norms = vec![Vec::new(); n_layers];
        for layer in 0..n_layers {
            self.load_norm_into(
                &format!("blk.{layer}.attn_norm.weight"),
                &mut attn_norms[layer],
                n_embd,
            )?;
            self.load_norm_into(
                &format!("blk.{layer}.ffn_norm.weight"),
                &mut ffn_norms[layer],
                n_embd,
            )?;
        }
        let mut output_norm = Vec::new();
        self.load_norm_into("output_norm.weight", &mut output_norm, n_embd)?;
        self.attn_norms = attn_norms;
        self.ffn_norms = ffn_norms;
        self.output_norm = output_norm;
        Ok(())
    }

    fn load_norm_into(&self, name: &str, out: &mut Vec<f32>, n: usize) -> Result<()> {
        let w = self.weight(name)?;
        out.clear();
        out.resize(n, 0.0);
        llama_kernels::quant::dequantize_row(w.ty(), w.as_bytes(), out)
    }

    fn tensor_info(&self, name: &str) -> Result<&TensorInfo> {
        let idx = self
            .index
            .get(name)
            .ok_or_else(|| LlamaError::MissingTensor(name.to_string()))?;
        Ok(&self.infos[*idx])
    }

    /// Builds a validated weight view for a tensor.
    fn weight(&self, name: &str) -> Result<WeightMatrix<'_>> {
        let info = self.tensor_info(name)?;
        let bytes = self.file.tensor_bytes(info)?;
        WeightMatrix::new(
            bytes,
            info.ty,
            info.dims.iter().map(|&d| d as usize).collect::<Vec<_>>(),
        )
    }

    /// The validated configuration.
    #[must_use]
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Vocabulary size.
    #[must_use]
    pub fn n_vocab(&self) -> usize {
        self.config.n_vocab
    }

    /// Total parameter count across all tensors.
    #[must_use]
    pub fn n_parameters(&self) -> u64 {
        self.infos
            .iter()
            .map(llama_gguf::TensorInfo::n_elements)
            .sum()
    }

    /// Total serialized tensor bytes.
    #[must_use]
    pub fn tensor_bytes_total(&self) -> u64 {
        self.infos.iter().filter_map(|t| t.size_bytes().ok()).sum()
    }

    /// Runs the transformer over `tokens` starting at KV position
    /// `start_pos`, appending keys/values to the cache.
    ///
    /// When `logits` is `Some`, it receives the final-token logits
    /// (`n_vocab` floats, resized if necessary).
    // Q/K/V projections and per-head loops deliberately use parallel names
    // (wq/wk/wv, q/k/v, h/d): the symmetry mirrors the attention math. The
    // body is one linear dataflow (embed -> blocks -> norm -> head) kept in
    // a single function so each stage's buffer flow is visible in order.
    #[allow(
        clippy::similar_names,
        clippy::many_single_char_names,
        clippy::too_many_lines
    )]
    pub fn forward(
        &self,
        tokens: &[TokenId],
        start_pos: usize,
        kv: &mut KvCache,
        scratch: &mut ForwardScratch,
        logits: Option<&mut Vec<f32>>,
    ) -> Result<()> {
        let c = &self.config;
        let n_embd = c.n_embd;
        let n_vocab = c.n_vocab;
        let t = tokens.len();
        if t == 0 {
            return Ok(());
        }
        if start_pos + t > kv.capacity() {
            return Err(LlamaError::runtime(format!(
                "sequence length {} exceeds the context window of {} tokens",
                start_pos + t,
                kv.capacity()
            )));
        }
        for &tok in tokens {
            if tok as usize >= n_vocab {
                return Err(LlamaError::runtime(format!(
                    "token id {tok} is outside the vocabulary (size {n_vocab})"
                )));
            }
        }

        scratch.begin_batch(t, c);

        // --- Embeddings --------------------------------------------------
        {
            let emb = self.weight("token_embd.weight")?;
            for (ti, &tok) in tokens.iter().enumerate() {
                emb.dequantize_row_into(
                    tok as usize,
                    &mut scratch.x[ti * n_embd..(ti + 1) * n_embd],
                )?;
            }
        }

        // --- Transformer blocks -------------------------------------------
        for layer in 0..c.n_layers {
            let p = format!("blk.{layer}.");
            let attn_norm = &self.attn_norms[layer];

            scratch.residual[..t * n_embd].copy_from_slice(&scratch.x[..t * n_embd]);

            // Attention norm into the separate `normed` buffer (the
            // residual stream `x` must stay intact).
            for ti in 0..t {
                let (a, b) = (ti * n_embd, (ti + 1) * n_embd);
                ops::rms_norm(
                    &scratch.x[a..b],
                    attn_norm,
                    &mut scratch.normed[a..b],
                    c.rms_eps,
                );
            }

            // QKV projections (batched over tokens).
            let wq = self.weight(&format!("{p}attn_q.weight"))?;
            let wk = self.weight(&format!("{p}attn_k.weight"))?;
            let wv = self.weight(&format!("{p}attn_v.weight"))?;
            let wq_n_out = c.n_heads * c.head_dim_k;
            let wk_n_out = c.n_kv_heads * c.head_dim_k;
            let wv_n_out = c.n_kv_heads * c.head_dim_v;
            {
                let ForwardScratch {
                    act,
                    normed,
                    q,
                    k,
                    v,
                    ..
                } = scratch;
                prepare_rows(act, wq.ty(), &normed[..t * n_embd], n_embd, t);
                matmul_rows(&wq, act, &mut q[..t * wq_n_out])?;
                prepare_rows(act, wk.ty(), &normed[..t * n_embd], n_embd, t);
                matmul_rows(&wk, act, &mut k[..t * wk_n_out])?;
                prepare_rows(act, wv.ty(), &normed[..t * n_embd], n_embd, t);
                matmul_rows(&wv, act, &mut v[..t * wv_n_out])?;
            }

            // Per-token attention (positions are sequential and causal).
            let scale = 1.0 / (c.head_dim_k as f32).sqrt();
            for ti in 0..t {
                let pos = start_pos + ti;
                let q_base = ti * c.n_heads * c.head_dim_k;
                let k_base = ti * c.n_kv_heads * c.head_dim_k;
                let v_base = ti * c.n_kv_heads * c.head_dim_v;

                for h in 0..c.n_heads {
                    let head =
                        &mut scratch.q[q_base + h * c.head_dim_k..q_base + (h + 1) * c.head_dim_k];
                    ops::rope_head(
                        head,
                        c.n_rot,
                        pos as f32,
                        c.rope_freq_base,
                        RotaryStyle::Interleaved,
                    );
                }
                for h in 0..c.n_kv_heads {
                    let head =
                        &mut scratch.k[k_base + h * c.head_dim_k..k_base + (h + 1) * c.head_dim_k];
                    ops::rope_head(
                        head,
                        c.n_rot,
                        pos as f32,
                        c.rope_freq_base,
                        RotaryStyle::Interleaved,
                    );
                }
                for h in 0..c.n_kv_heads {
                    kv.set_k(
                        layer,
                        pos,
                        h,
                        &scratch.k[k_base + h * c.head_dim_k..k_base + (h + 1) * c.head_dim_k],
                    );
                    kv.set_v(
                        layer,
                        pos,
                        h,
                        &scratch.v[v_base + h * c.head_dim_v..v_base + (h + 1) * c.head_dim_v],
                    );
                }

                for h in 0..c.n_heads {
                    let kv_head = h / c.n_gqa();
                    let q_head =
                        &scratch.q[q_base + h * c.head_dim_k..q_base + (h + 1) * c.head_dim_k];

                    scratch.scores.clear();
                    scratch.scores.resize(pos + 1, 0.0);
                    for (j, slot) in scratch.scores.iter_mut().enumerate() {
                        let k_vec = kv.k(layer, j, kv_head);
                        let mut dot = 0.0f32;
                        for (d, &qv) in q_head.iter().enumerate() {
                            dot += qv * k_vec.get(d);
                        }
                        *slot = dot * scale;
                    }
                    ops::softmax_inplace(&mut scratch.scores);

                    let out_base = ti * n_embd + h * c.head_dim_v;
                    scratch.attn[out_base..out_base + c.head_dim_v].fill(0.0);
                    for (j, &w) in scratch.scores.iter().enumerate() {
                        let v_vec = kv.v(layer, j, kv_head);
                        let out_head = &mut scratch.attn[out_base..out_base + c.head_dim_v];
                        for (d, slot) in out_head.iter_mut().enumerate() {
                            *slot += w * v_vec.get(d);
                        }
                    }
                }
            }

            // Attention output projection + residual.
            {
                let wo = self.weight(&format!("{p}attn_output.weight"))?;
                {
                    let ForwardScratch {
                        act, attn, proj, ..
                    } = scratch;
                    prepare_rows(act, wo.ty(), &attn[..t * n_embd], n_embd, t);
                    matmul_rows(&wo, act, &mut proj[..t * n_embd])?;
                }
                for i in 0..t * n_embd {
                    scratch.x[i] = scratch.residual[i] + scratch.proj[i];
                }
            }

            // FFN.
            {
                let ffn_norm = &self.ffn_norms[layer];
                for ti in 0..t {
                    let (a, b) = (ti * n_embd, (ti + 1) * n_embd);
                    ops::rms_norm(
                        &scratch.x[a..b],
                        ffn_norm,
                        &mut scratch.normed[a..b],
                        c.rms_eps,
                    );
                }

                let gate = self.weight(&format!("{p}ffn_gate.weight"))?;
                let up = self.weight(&format!("{p}ffn_up.weight"))?;
                let down = self.weight(&format!("{p}ffn_down.weight"))?;

                {
                    let ForwardScratch {
                        act,
                        normed,
                        ffn_gate,
                        ffn_up,
                        ..
                    } = scratch;
                    prepare_rows(act, gate.ty(), &normed[..t * n_embd], n_embd, t);
                    matmul_rows(&gate, act, &mut ffn_gate[..t * c.n_ff])?;
                    prepare_rows(act, up.ty(), &normed[..t * n_embd], n_embd, t);
                    matmul_rows(&up, act, &mut ffn_up[..t * c.n_ff])?;

                    ops::silu_inplace(&mut ffn_gate[..t * c.n_ff]);
                    ops::mul_inplace(&mut ffn_gate[..t * c.n_ff], &ffn_up[..t * c.n_ff]);
                }
                {
                    let ForwardScratch {
                        act,
                        ffn_gate,
                        proj,
                        ..
                    } = scratch;
                    prepare_rows(act, down.ty(), &ffn_gate[..t * c.n_ff], c.n_ff, t);
                    matmul_rows(&down, act, &mut proj[..t * n_embd])?;
                }
                // FFN residual: ADD to the current hidden state (which
                // already contains the attention contribution).
                for i in 0..t * n_embd {
                    scratch.x[i] += scratch.proj[i];
                }
            }
        }

        // The KV positions are shared across layers: advance once per
        // forward call.
        kv.advance(t);

        // --- Final norm + logits -------------------------------------------
        let last = (t - 1) * n_embd;
        ops::rms_norm(
            &scratch.x[last..last + n_embd],
            &self.output_norm,
            &mut scratch.normed[last..last + n_embd],
            c.rms_eps,
        );

        if let Some(logits_out) = logits {
            let out_name = if self.index.contains_key("output.weight") {
                "output.weight"
            } else {
                "token_embd.weight"
            };
            let w = self.weight(out_name)?;
            let act = Activation::prepare(w.ty(), &scratch.normed[last..last + n_embd]);
            logits_out.clear();
            logits_out.resize(w.n_rows(), 0.0);
            llama_tensor::matmul_row(&w, &act, logits_out)?;
        }
        Ok(())
    }

    /// Weight tensor inventory for inspection (name, type, element count).
    #[must_use]
    pub fn tensor_types(&self) -> Vec<(String, llama_core::GgmlType, u64)> {
        self.infos
            .iter()
            .map(|t| (t.name.clone(), t.ty, t.n_elements()))
            .collect()
    }
}

/// Reusable computation buffers for the forward pass.
///
/// Owned by the caller (a runtime session) so that decoding performs no
/// allocations after warm-up.
pub struct ForwardScratch {
    /// Hidden states `[t, n_embd]`.
    x: Vec<f32>,
    /// Residual stream copy.
    residual: Vec<f32>,
    /// Attention output `[t, n_embd]`.
    attn: Vec<f32>,
    /// Projection scratch `[t, n_embd]`.
    proj: Vec<f32>,
    /// Query projections `[t, n_heads * head_dim_k]`.
    q: Vec<f32>,
    /// Key projections `[t, n_kv_heads * head_dim_k]`.
    k: Vec<f32>,
    /// Value projections `[t, n_kv_heads * head_dim_v]`.
    v: Vec<f32>,
    /// FFN gate `[t, n_ff]`.
    ffn_gate: Vec<f32>,
    /// FFN up `[t, n_ff]`.
    ffn_up: Vec<f32>,
    /// Normed activations `[t, n_embd]` (kept separate from the residual
    /// stream `x`, which must survive the norm for the residual adds).
    normed: Vec<f32>,
    /// Softmax scores (reused per head).
    scores: Vec<f32>,
    /// Prepared activation rows for the current matmul.
    act: Vec<Activation>,
}

impl ForwardScratch {
    /// Allocates scratch for a model configuration and a maximum batch size.
    #[must_use]
    pub fn new(config: &LlamaConfig, max_batch: usize) -> Self {
        let t = max_batch;
        Self {
            x: vec![0.0; t * config.n_embd],
            residual: vec![0.0; t * config.n_embd],
            attn: vec![0.0; t * config.n_embd],
            normed: vec![0.0; t * config.n_embd],
            proj: vec![0.0; t * config.n_embd],
            q: vec![0.0; t * config.n_heads * config.head_dim_k],
            k: vec![0.0; t * config.n_kv_heads * config.head_dim_k],
            v: vec![0.0; t * config.n_kv_heads * config.head_dim_v],
            ffn_gate: vec![0.0; t * config.n_ff],
            ffn_up: vec![0.0; t * config.n_ff],
            scores: Vec::with_capacity(4096),
            act: (0..t).map(|_| Activation::F32(Vec::new())).collect(),
        }
    }

    fn begin_batch(&mut self, t: usize, config: &LlamaConfig) {
        let need = |v: &mut Vec<f32>, per_row: usize| {
            if t * per_row > v.len() {
                v.resize(t * per_row, 0.0);
            }
        };
        need(&mut self.x, config.n_embd);
        need(&mut self.residual, config.n_embd);
        need(&mut self.attn, config.n_embd);
        need(&mut self.normed, config.n_embd);
        need(&mut self.proj, config.n_embd);
        need(&mut self.q, config.n_heads * config.head_dim_k);
        need(&mut self.k, config.n_kv_heads * config.head_dim_k);
        need(&mut self.v, config.n_kv_heads * config.head_dim_v);
        need(&mut self.ffn_gate, config.n_ff);
        need(&mut self.ffn_up, config.n_ff);
        while self.act.len() < t {
            self.act.push(Activation::F32(Vec::new()));
        }
        self.act.truncate(t);
    }
}

/// (Re)prepares `n_rows` activation rows from `src` (row-major, `n_in`
/// per row) for a weight of type `ty`, reusing allocations.
///
/// Quantized weights switch the row to the `Q8_0` representation (quantize
/// once, reuse for every output channel); F32/F16 weights keep raw rows.
fn prepare_rows(
    act: &mut [Activation],
    ty: llama_core::GgmlType,
    src: &[f32],
    n_in: usize,
    n_rows: usize,
) {
    debug_assert!(src.len() >= n_rows * n_in);
    let unquantized = matches!(ty, llama_core::GgmlType::F32 | llama_core::GgmlType::F16);
    for ti in 0..n_rows {
        let row = &src[ti * n_in..(ti + 1) * n_in];
        match &mut act[ti] {
            Activation::F32(v) if unquantized => {
                v.clear();
                v.extend_from_slice(row);
            }
            Activation::Q8(q) if !unquantized => quantize_row_q8_0_into(q, row),
            slot @ Activation::Q8(_) => *slot = Activation::F32(row.to_vec()),
            slot @ Activation::F32(_) => {
                let mut q = llama_kernels::quant::QuantizedRow {
                    blocks: Vec::with_capacity(row.len() / 32),
                };
                quantize_row_q8_0_into(&mut q, row);
                *slot = Activation::Q8(q);
            }
        }
    }
}
