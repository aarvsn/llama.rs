//! # llama-fixtures
//!
//! Deterministic synthetic GGUF model generators used by tests, benchmarks
//! and examples. Nothing here is part of the inference engine.
//!
//! Every fixture is built from a fixed seed, so CI never needs to download
//! model files and failures are reproducible byte-for-byte.

use std::path::Path;

use llama_core::{F16, GgmlType, QK_K, Result};
use llama_gguf::{GgufWriter, TensorPayload};

/// Quantization scheme applied to the 2-D weight matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// All matrices F32.
    F32,
    /// All matrices F16.
    F16,
    /// All matrices `Q8_0`.
    Q8_0,
    /// All matrices `Q4_0`.
    Q4_0,
    /// Embedding/output `Q6_K`, attention `Q4_K`, FFN `Q5_K` (K-quant mix).
    KQuants,
}

/// Tiny Llama model geometry.
#[derive(Debug, Clone)]
pub struct FixtureSpec {
    /// Embedding dimension (multiple of 256 for K-quant fixtures).
    pub n_embd: usize,
    /// Layer count.
    pub n_layers: usize,
    /// Query heads.
    pub n_heads: usize,
    /// KV heads (GQA).
    pub n_kv_heads: usize,
    /// Feed-forward dimension.
    pub n_ff: usize,
    /// Vocabulary size (>= 259: unk/bos/eos + 256 byte tokens).
    pub n_vocab: usize,
    /// Training context length.
    pub n_ctx: usize,
    /// Deterministic seed.
    pub seed: u64,
    /// Quantization scheme.
    pub quant: QuantScheme,
    /// When true, all weights are exactly zero instead of pseudo-random.
    /// Every hidden state and logit is then 0, so greedy argmax always picks
    /// token 0 — used to test the EOS stop path deterministically.
    pub zero_weights: bool,
    /// Overrides `tokenizer.ggml.eos_token_id` (use with `zero_weights` and
    /// `Some(0)` to make the model emit EOS on the first decode step).
    pub eos_token_id: Option<u64>,
}

impl Default for FixtureSpec {
    fn default() -> Self {
        Self {
            n_embd: 64,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            n_ff: 192,
            n_vocab: 512,
            n_ctx: 512,
            seed: 0x5EED_1A7A,
            quant: QuantScheme::F32,
            zero_weights: false,
            eos_token_id: None,
        }
    }
}

impl FixtureSpec {
    /// A small fixture using K-quants (requires `n_embd` divisible by 256).
    #[must_use]
    pub fn kquant() -> Self {
        Self {
            n_embd: 256,
            n_layers: 2,
            n_heads: 8,
            n_kv_heads: 4,
            n_ff: 256,
            n_vocab: 512,
            n_ctx: 256,
            seed: 0x5EED_2B2B,
            quant: QuantScheme::KQuants,
            zero_weights: false,
            eos_token_id: None,
        }
    }

    fn head_dim(&self) -> usize {
        self.n_embd / self.n_heads
    }
}

/// Deterministic LCG random floats in `(-scale, scale)`.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / 8_388_608.0 - 0.5) * 2.0
    }

    fn next_u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 56) as u8
    }
}

/// Serializes an f32 matrix (row-major, `n_out` rows of `n_in`) into the
/// target tensor format.
fn serialize_matrix(data: &[f32], quant: GgmlType) -> Vec<u8> {
    match quant {
        GgmlType::F32 => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        GgmlType::F16 => data
            .iter()
            .flat_map(|v| F16::from_f32(*v).to_bits().to_le_bytes())
            .collect(),
        GgmlType::Q8_0 => {
            let mut out = Vec::with_capacity(data.len() / 32 * 34);
            for chunk in data.chunks(32) {
                let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let id = if d == 0.0 { 0.0 } else { 1.0 / d };
                out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
                for &v in chunk {
                    out.push((v * id).round() as i8 as u8);
                }
            }
            out
        }
        GgmlType::Q4_0 => {
            let mut out = Vec::with_capacity(data.len() / 32 * 18);
            for chunk in data.chunks(32) {
                let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = amax / 15.0;
                let id = if d == 0.0 { 0.0 } else { 1.0 / d };
                out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
                let mut qs = [0u8; 16];
                for (j, &v) in chunk.iter().enumerate() {
                    let q = (((v * id) + 8.5) as i32).clamp(0, 15) as u8;
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
        // K-quants in fixtures are crafted directly from random bytes; the
        // dequantization correctness is validated separately against
        // independent reference values (see tests/reference_values).
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => {
            let block_bytes = quant.block_byte_size();
            let n_blocks = data.len() / quant.block_size();
            if data.iter().all(|&v| v == 0.0) {
                // Zero weights: d = dmin = 0 and zero payload dequantize to
                // exactly zero for every K-quant format.
                return vec![0u8; n_blocks * block_bytes];
            }
            let mut rng = Lcg(0xABCD_EF01);
            let mut out = Vec::with_capacity(n_blocks * block_bytes);
            for _ in 0..n_blocks {
                // Scale fields with benign values so activations stay finite.
                let d = F16::from_f32(0.01).to_bits().to_le_bytes();
                match quant {
                    GgmlType::Q4_K | GgmlType::Q5_K => {
                        out.extend_from_slice(&d);
                        out.extend_from_slice(&F16::from_f32(0.002).to_bits().to_le_bytes());
                    }
                    _ => out.extend_from_slice(&d),
                }
                for _ in 0..block_bytes
                    - 2
                    - if matches!(quant, GgmlType::Q4_K | GgmlType::Q5_K) {
                        2
                    } else {
                        0
                    }
                {
                    out.push(rng.next_u8());
                }
            }
            out
        }
        other => unimplemented!("fixture quantization for {other}"),
    }
}

fn matrix_type(scheme: QuantScheme, name: &str) -> GgmlType {
    match scheme {
        QuantScheme::F32 => GgmlType::F32,
        QuantScheme::F16 => GgmlType::F16,
        QuantScheme::Q8_0 => GgmlType::Q8_0,
        QuantScheme::Q4_0 => GgmlType::Q4_0,
        QuantScheme::KQuants => {
            if name.starts_with("token_embd") || name.starts_with("output") {
                GgmlType::Q6_K
            } else if name.contains("attn_") {
                GgmlType::Q4_K
            } else {
                GgmlType::Q5_K
            }
        }
    }
}

/// Common `SentencePiece` pieces (metaspace `▁`-prefixed words plus a few
/// bare characters) used before the filler tail of fixture vocabularies.
const FIXTURE_WORDS: &[&str] = &[
    "▁",
    "▁the",
    "▁a",
    "▁is",
    "▁was",
    "▁and",
    "▁to",
    "▁of",
    "▁in",
    "▁that",
    "▁it",
    "▁hello",
    "▁world",
    "▁rust",
    "▁llama",
    "▁model",
    "▁test",
    "▁quick",
    "▁brown",
    "▁fox",
    "▁jumps",
    "▁over",
    "▁lazy",
    "▁dog",
    "▁once",
    "▁upon",
    "▁time",
    "▁there",
    "▁were",
    "▁three",
    "▁little",
    "▁pigs",
    "▁who",
    "▁went",
    "▁out",
    "▁into",
    "▁worldd",
    "▁find",
    "▁their",
    "▁fortune",
    "▁story",
    "▁end",
    "▁new",
    "▁day",
    "▁night",
    "▁good",
    "▁bad",
    "▁big",
    "▁small",
    "▁one",
    "▁two",
    "▁three",
    "▁four",
    "▁five",
    "s",
    "▁said",
    "▁they",
    "▁with",
    "▁for",
    "▁but",
    "▁not",
    "▁you",
    "▁i",
    "▁he",
    "▁she",
    "▁we",
    "▁they2",
];

/// Builds the SentencePiece-style vocabulary for fixtures.
///
/// Layout: `<unk>` 0, `<s>` 1, `</s>` 2, `<|fixture|>` 3, byte tokens
/// `<0x00>`..`<0xFF>` 4..259, then `▁`-prefixed words and single characters.
fn build_vocab(n_vocab: usize) -> (Vec<String>, Vec<i32>, Vec<f32>) {
    let mut pieces = Vec::with_capacity(n_vocab);
    let mut kinds = Vec::with_capacity(n_vocab);
    let mut scores = Vec::with_capacity(n_vocab);

    macro_rules! push {
        ($piece:expr, $kind:expr, $score:expr) => {{
            pieces.push($piece);
            kinds.push($kind);
            scores.push($score);
        }};
    }

    push!("<unk>".to_string(), 2, 0.0);
    push!("<s>".to_string(), 3, 0.0);
    push!("</s>".to_string(), 3, 0.0);
    push!("<|fixture|>".to_string(), 4, 0.0);
    for b in 0..=255u8 {
        push!(format!("<0x{b:02X}>"), 6, 0.0);
    }

    // Common words (with ▁), then single characters, then filler pieces.
    let mut score = -1.0f32;
    for w in FIXTURE_WORDS {
        if pieces.len() >= n_vocab {
            break;
        }
        push!(w.to_string(), 1, score);
        score -= 0.01;
    }
    // Single ASCII characters (letters/digits/punct).
    for c in b'a'..=b'z' {
        if pieces.len() >= n_vocab {
            break;
        }
        push!((c as char).to_string(), 1, score);
        score -= 0.01;
    }
    for c in b'A'..=b'Z' {
        if pieces.len() >= n_vocab {
            break;
        }
        push!((c as char).to_string(), 1, score);
        score -= 0.01;
    }
    while pieces.len() < n_vocab {
        push!(format!("▁filler{}", pieces.len()), 1, score);
        score -= 0.01;
    }
    (pieces, kinds, scores)
}

/// Writes a synthetic llama GGUF model to `path`.
///
/// Returns the number of tensors written.
///
/// # Errors
///
/// Returns an error when the file cannot be written or the spec is invalid.
/// Appends a randomized (or all-zero) matrix tensor in the scheme's format.
fn add_matrix(
    w: &mut GgufWriter,
    rng: &mut Lcg,
    quant: QuantScheme,
    scale: f32,
    zero_weights: bool,
    n_tensors: &mut usize,
    name: &str,
    n_in: usize,
    n_out: usize,
) {
    let ty = matrix_type(quant, name);
    let count = n_in * n_out;
    let data: Vec<f32> = if zero_weights {
        vec![0.0; count]
    } else {
        (0..count).map(|_| rng.next_f32() * scale).collect()
    };
    let bytes = serialize_matrix(&data, ty);
    w.add_tensor(TensorPayload {
        name: name.to_string(),
        dims: vec![n_in as u64, n_out as u64],
        ty,
        data: bytes,
    });
    *n_tensors += 1;
}

/// Appends a norm-weight vector (values near 1.0, or all-zero).
fn add_vector(
    w: &mut GgufWriter,
    rng: &mut Lcg,
    zero_weights: bool,
    n_tensors: &mut usize,
    name: &str,
    n: usize,
) {
    let data: Vec<f32> = if zero_weights {
        vec![0.0; n]
    } else {
        (0..n).map(|_| 1.0 + rng.next_f32() * 0.1).collect()
    };
    w.add_tensor(TensorPayload {
        name: name.to_string(),
        dims: vec![n as u64],
        ty: GgmlType::F32,
        data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    });
    *n_tensors += 1;
}

/// Checks the geometry constraints a fixture needs to be loadable.
fn validate_spec(spec: &FixtureSpec) -> Result<()> {
    if spec.n_embd % spec.n_heads != 0 || spec.n_heads % spec.n_kv_heads != 0 {
        return Err(llama_core::LlamaError::InvalidArgument(
            "invalid fixture geometry".into(),
        ));
    }
    for (name, rows_len) in [
        ("n_embd", spec.n_embd),
        ("n_ff", spec.n_ff),
        ("vocab*embd rows", spec.n_vocab),
    ] {
        let divisible = match spec.quant {
            QuantScheme::KQuants => rows_len % QK_K == 0,
            _ => true,
        };
        if !divisible {
            return Err(llama_core::LlamaError::InvalidArgument(format!(
                "K-quant fixtures require {name} divisible by {QK_K}"
            )));
        }
    }
    Ok(())
}

/// Writes the GGUF metadata (architecture hyper-parameters, tokenizer
/// arrays and chat template) for a fixture.
fn write_fixture_metadata(w: &mut GgufWriter, spec: &FixtureSpec, head_dim: usize) {
    w.set_str("general.architecture", "llama");
    w.set_str("general.name", "llama.rs synthetic fixture");
    w.set_str("general.version", "0.1");
    w.set_u64(
        "general.file_type",
        match spec.quant {
            QuantScheme::F32 => 0,
            QuantScheme::F16 => 1,
            QuantScheme::Q8_0 => 7,
            QuantScheme::Q4_0 => 2,
            QuantScheme::KQuants => 15,
        },
    );
    w.set_u64("llama.context_length", spec.n_ctx as u64);
    w.set_u64("llama.embedding_length", spec.n_embd as u64);
    w.set_u64("llama.block_count", spec.n_layers as u64);
    w.set_u64("llama.feed_forward_length", spec.n_ff as u64);
    w.set_u64("llama.attention.head_count", spec.n_heads as u64);
    w.set_u64("llama.attention.head_count_kv", spec.n_kv_heads as u64);
    w.set_u64("llama.rope.dimension_count", head_dim as u64);
    w.set_f32("llama.rope.freq_base", 10_000.0);
    w.set_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
    w.set_str(
        "tokenizer.chat_template",
        "{% for message in messages %}[INST] {{ message['content'] }} [/INST] {% endfor %}",
    );

    // Tokenizer metadata.
    let (pieces, kinds, scores) = build_vocab(spec.n_vocab);
    w.set_str("tokenizer.ggml.model", "llama");
    w.set_str_vec("tokenizer.ggml.tokens", pieces);
    w.set_i32_vec("tokenizer.ggml.token_type", kinds);
    w.set_f32_vec("tokenizer.ggml.scores", scores);
    w.set_u64("tokenizer.ggml.bos_token_id", 1);
    w.set_u64(
        "tokenizer.ggml.eos_token_id",
        spec.eos_token_id.unwrap_or(2),
    );
    w.set_u64("tokenizer.ggml.unk_token_id", 0);
    w.set_bool("tokenizer.ggml.add_bos_token", true);
}

/// Writes the attention tensors (norm, Q, K, V, output projection) of one
/// transformer block.
fn write_attention_tensors(
    w: &mut GgufWriter,
    rng: &mut Lcg,
    spec: &FixtureSpec,
    zw: bool,
    layer: usize,
    head_dim: usize,
    n_tensors: &mut usize,
) {
    let p = format!("blk.{layer}.");
    let scale = 1.0 / (spec.n_embd as f32).sqrt();
    add_vector(
        w,
        rng,
        zw,
        n_tensors,
        &format!("{p}attn_norm.weight"),
        spec.n_embd,
    );
    add_matrix(
        w,
        rng,
        spec.quant,
        scale,
        zw,
        n_tensors,
        &format!("{p}attn_q.weight"),
        spec.n_embd,
        spec.n_heads * head_dim,
    );
    add_matrix(
        w,
        rng,
        spec.quant,
        scale,
        zw,
        n_tensors,
        &format!("{p}attn_k.weight"),
        spec.n_embd,
        spec.n_kv_heads * head_dim,
    );
    add_matrix(
        w,
        rng,
        spec.quant,
        scale,
        zw,
        n_tensors,
        &format!("{p}attn_v.weight"),
        spec.n_embd,
        spec.n_kv_heads * head_dim,
    );
    add_matrix(
        w,
        rng,
        spec.quant,
        scale,
        zw,
        n_tensors,
        &format!("{p}attn_output.weight"),
        spec.n_embd,
        spec.n_embd,
    );
}

/// Writes the feed-forward tensors (norm, gate, up, down) of one block.
fn write_ffn_tensors(
    w: &mut GgufWriter,
    rng: &mut Lcg,
    spec: &FixtureSpec,
    zw: bool,
    layer: usize,
    n_tensors: &mut usize,
) {
    let p = format!("blk.{layer}.");
    let scale = 1.0 / (spec.n_embd as f32).sqrt();
    add_vector(
        w,
        rng,
        zw,
        n_tensors,
        &format!("{p}ffn_norm.weight"),
        spec.n_embd,
    );
    add_matrix(
        w,
        rng,
        spec.quant,
        scale,
        zw,
        n_tensors,
        &format!("{p}ffn_gate.weight"),
        spec.n_embd,
        spec.n_ff,
    );
    add_matrix(
        w,
        rng,
        spec.quant,
        scale,
        zw,
        n_tensors,
        &format!("{p}ffn_up.weight"),
        spec.n_embd,
        spec.n_ff,
    );
    add_matrix(
        w,
        rng,
        spec.quant,
        scale,
        zw,
        n_tensors,
        &format!("{p}ffn_down.weight"),
        spec.n_ff,
        spec.n_embd,
    );
}

pub fn write_llama_fixture(path: &Path, spec: &FixtureSpec) -> Result<usize> {
    validate_spec(spec)?;

    let head_dim = spec.head_dim();
    let mut rng = Lcg(spec.seed);
    let scale = 1.0 / (spec.n_embd as f32).sqrt();

    let mut w = GgufWriter::new();
    write_fixture_metadata(&mut w, spec, head_dim);

    let mut n_tensors = 0usize;

    let zw = spec.zero_weights;
    add_matrix(
        &mut w,
        &mut rng,
        spec.quant,
        scale,
        zw,
        &mut n_tensors,
        "token_embd.weight",
        spec.n_embd,
        spec.n_vocab,
    );
    add_vector(
        &mut w,
        &mut rng,
        zw,
        &mut n_tensors,
        "output_norm.weight",
        spec.n_embd,
    );
    for layer in 0..spec.n_layers {
        write_attention_tensors(&mut w, &mut rng, spec, zw, layer, head_dim, &mut n_tensors);
        write_ffn_tensors(&mut w, &mut rng, spec, zw, layer, &mut n_tensors);
    }
    // No output.weight -> tied embeddings.

    w.write_to_file(path, 32)?;
    Ok(n_tensors)
}
