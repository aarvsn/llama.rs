//! Converts llama2.c `.bin` checkpoints to GGUF.
//!
//! Usage:
//! ```text
//! cargo run --release -p llama-fixtures --example convert_llama2c -- \
//!     stories15M.bin tokenizer.bin stories15M.gguf [quant]
//! ```
//!
//! `quant` is one of `f32` (default), `f16`, `q8_0`, `q4_0`.
//!
//! The llama2.c layout matches the GGUF llama conventions directly:
//! every weight matrix is `[n_in, n_out]` with contiguous output rows, and
//! llama2.c's rotary embedding uses consecutive pairs `(x[2i], x[2i+1])`,
//! which is exactly the pairing llama.rs applies for the llama architecture
//! (no permutation needed).

use std::path::PathBuf;

use llama_core::{F16, GgmlType, Result};
use llama_gguf::{GgufWriter, TensorPayload};

struct Config {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
    shared_weights: bool,
}

fn read_i32(bytes: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn read_f32(bytes: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn serialize_matrix(data: &[u8], ty: GgmlType) -> Vec<u8> {
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    match ty {
        GgmlType::F32 => data.to_vec(),
        GgmlType::F16 => floats
            .iter()
            .flat_map(|v| F16::from_f32(*v).to_bits().to_le_bytes())
            .collect(),
        GgmlType::Q8_0 => quantize_q8_0(&floats),
        GgmlType::Q4_0 => quantize_q4_0(&floats),
        other => unimplemented!("conversion to {other}"),
    }
}

fn quantize_q8_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 34);
    for chunk in x.chunks(32) {
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

fn quantize_q4_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 18);
    for chunk in x.chunks(32) {
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: convert_llama2c <model.bin> <tokenizer.bin> <out.gguf> [f32|f16|q8_0|q4_0]"
        );
        std::process::exit(2);
    }
    let model_path = PathBuf::from(&args[1]);
    let tok_path = PathBuf::from(&args[2]);
    let out_path = PathBuf::from(&args[3]);
    let quant = match args.get(4).map(String::as_str) {
        Some("f16") => GgmlType::F16,
        Some("q8_0") => GgmlType::Q8_0,
        Some("q4_0") => GgmlType::Q4_0,
        _ => GgmlType::F32,
    };

    match run(&model_path, &tok_path, &out_path, quant) {
        Ok(info) => println!("wrote {} ({info})", out_path.display()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

/// Reads the llama2.c 28-byte header into a [`Config`].
fn parse_config(model: &[u8]) -> Result<Config> {
    if model.len() < 28 {
        return Err(llama_core::LlamaError::InvalidArgument(
            "model file too small".into(),
        ));
    }
    let raw_vocab = read_i32(model, 5 * 4);
    Ok(Config {
        dim: read_i32(model, 0).max(0) as usize,
        hidden_dim: read_i32(model, 4).max(0) as usize,
        n_layers: read_i32(model, 8).max(0) as usize,
        n_heads: read_i32(model, 12).max(0) as usize,
        n_kv_heads: read_i32(model, 16).max(0) as usize,
        vocab_size: raw_vocab.unsigned_abs() as usize,
        seq_len: read_i32(model, 24).max(0) as usize,
        shared_weights: raw_vocab > 0,
    })
}

/// Parses `tokenizer.bin`: `max_token_length` (i32), then per token a score
/// (f32), a length (i32) and the raw bytes.
///
/// Karpathy's `tokenizer.bin` applies: `<0xNN>` -> raw byte char, `\u{2581}`
/// -> space, BOS/EOS wrapped in newlines. The pieces are rebuilt using the
/// fixed `SentencePiece` id layout (0 unk, 1 bos, 2 eos, 3..=258 bytes).
#[allow(clippy::type_complexity)]
fn parse_tokenizer(tok: &[u8], vocab_size: usize) -> Result<(Vec<String>, Vec<i32>, Vec<f32>)> {
    let mut tok_off = 0usize;
    let max_token_length = read_i32(tok, tok_off);
    tok_off += 4;
    let mut pieces = Vec::with_capacity(vocab_size);
    let mut kinds = Vec::with_capacity(vocab_size);
    let mut scores = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        if tok_off + 8 > tok.len() {
            return Err(llama_core::LlamaError::InvalidArgument(
                "tokenizer.bin truncated".into(),
            ));
        }
        let score = read_f32(tok, tok_off);
        tok_off += 4;
        let len = read_i32(tok, tok_off).max(0) as usize;
        tok_off += 4;
        if tok_off + len > tok.len() {
            return Err(llama_core::LlamaError::InvalidArgument(
                "tokenizer.bin truncated piece".into(),
            ));
        }
        let raw = &tok[tok_off..tok_off + len];
        tok_off += len;
        let (piece, kind): (String, i32) = match i {
            0 => ("<unk>".to_string(), 2),
            1 => ("<s>".to_string(), 3),
            2 => ("</s>".to_string(), 3),
            3..=258 => (format!("<0x{:02X}>", i - 3), 6),
            _ => {
                // Spaces in the raw pieces are SentencePiece metaspaces.
                let text = String::from_utf8_lossy(raw).replace(' ', "\u{2581}");
                (text, 1)
            }
        };
        pieces.push(piece);
        scores.push(score);
        kinds.push(kind);
    }
    let _ = max_token_length;
    Ok((pieces, kinds, scores))
}

/// Reads `count` little-endian f32 values from `model` at `*off`, advancing it.
fn cursor(model: &[u8], count: usize, off: &mut usize) -> Result<Vec<u8>> {
    let bytes = count * 4;
    if *off + bytes > model.len() {
        return Err(llama_core::LlamaError::InvalidArgument(
            "model file truncated".into(),
        ));
    }
    let slice = model[*off..*off + bytes].to_vec();
    *off += bytes;
    Ok(slice)
}

fn add_quantized(
    w: &mut GgufWriter,
    name: String,
    data: &[u8],
    n_in: usize,
    n_out: usize,
    ty: GgmlType,
) {
    w.add_tensor(TensorPayload {
        name,
        dims: if n_out == 1 {
            vec![n_in as u64]
        } else {
            vec![n_in as u64, n_out as u64]
        },
        ty,
        data: serialize_matrix(data, ty),
    });
}

fn add_norm(w: &mut GgufWriter, name: String, data: Vec<u8>, n: usize) {
    w.add_tensor(TensorPayload {
        name,
        dims: vec![n as u64],
        ty: GgmlType::F32,
        data,
    });
}

fn layer_slice(blob: &[u8], l: usize, per_layer_floats: usize) -> Vec<u8> {
    let start = l * per_layer_floats * 4;
    blob[start..start + per_layer_floats * 4].to_vec()
}

/// Writes all model tensors, walking the llama2.c weight groups.
// llama2.c groups weights by type across all layers; the walk below mirrors
// that ordering directly, which reads best as one flat sequence.
#[allow(clippy::too_many_lines)]
fn write_weights(
    w: &mut GgufWriter,
    model: &[u8],
    config: &Config,
    head_size: usize,
    quant: GgmlType,
) -> Result<()> {
    let mut off = 28usize;

    let emb = cursor(model, config.vocab_size * config.dim, &mut off)?;
    add_quantized(
        &mut *w,
        "token_embd.weight".into(),
        &emb,
        config.dim,
        config.vocab_size,
        quant,
    );

    let rms_att = cursor(model, config.n_layers * config.dim, &mut off)?;
    let wq = cursor(
        model,
        config.n_layers * config.dim * config.n_heads * head_size,
        &mut off,
    )?;
    let wk = cursor(
        model,
        config.n_layers * config.dim * config.n_kv_heads * head_size,
        &mut off,
    )?;
    let wv = cursor(
        model,
        config.n_layers * config.dim * config.n_kv_heads * head_size,
        &mut off,
    )?;
    let wo = cursor(
        model,
        config.n_layers * config.n_heads * head_size * config.dim,
        &mut off,
    )?;
    let rms_ffn = cursor(model, config.n_layers * config.dim, &mut off)?;
    let w1 = cursor(
        model,
        config.n_layers * config.dim * config.hidden_dim,
        &mut off,
    )?;
    let w2 = cursor(
        model,
        config.n_layers * config.hidden_dim * config.dim,
        &mut off,
    )?;
    let w3 = cursor(
        model,
        config.n_layers * config.dim * config.hidden_dim,
        &mut off,
    )?;

    for l in 0..config.n_layers {
        add_norm(
            w,
            format!("blk.{l}.attn_norm.weight"),
            layer_slice(&rms_att, l, config.dim),
            config.dim,
        );
        add_quantized(
            w,
            format!("blk.{l}.attn_q.weight"),
            &layer_slice(&wq, l, config.dim * config.n_heads * head_size),
            config.dim,
            config.n_heads * head_size,
            quant,
        );
        add_quantized(
            w,
            format!("blk.{l}.attn_k.weight"),
            &layer_slice(&wk, l, config.dim * config.n_kv_heads * head_size),
            config.dim,
            config.n_kv_heads * head_size,
            quant,
        );
        add_quantized(
            w,
            format!("blk.{l}.attn_v.weight"),
            &layer_slice(&wv, l, config.dim * config.n_kv_heads * head_size),
            config.dim,
            config.n_kv_heads * head_size,
            quant,
        );
        add_quantized(
            w,
            format!("blk.{l}.attn_output.weight"),
            &layer_slice(&wo, l, config.n_heads * head_size * config.dim),
            config.dim,
            config.dim,
            quant,
        );
        add_norm(
            w,
            format!("blk.{l}.ffn_norm.weight"),
            layer_slice(&rms_ffn, l, config.dim),
            config.dim,
        );
        add_quantized(
            w,
            format!("blk.{l}.ffn_gate.weight"),
            &layer_slice(&w1, l, config.dim * config.hidden_dim),
            config.dim,
            config.hidden_dim,
            quant,
        );
        add_quantized(
            w,
            format!("blk.{l}.ffn_down.weight"),
            &layer_slice(&w2, l, config.hidden_dim * config.dim),
            config.hidden_dim,
            config.dim,
            quant,
        );
        add_quantized(
            w,
            format!("blk.{l}.ffn_up.weight"),
            &layer_slice(&w3, l, config.dim * config.hidden_dim),
            config.dim,
            config.hidden_dim,
            quant,
        );
    }

    let rms_final = cursor(model, config.dim, &mut off)?;
    add_norm(w, "output_norm.weight".into(), rms_final, config.dim);

    // Skip the legacy RoPE tables.
    let rope_floats = config.seq_len * head_size / 2;
    let _ = cursor(model, rope_floats, &mut off)?;
    let _ = cursor(model, rope_floats, &mut off)?;

    if !config.shared_weights {
        let wcls = cursor(model, config.vocab_size * config.dim, &mut off)?;
        add_quantized(
            &mut *w,
            "output.weight".into(),
            &wcls,
            config.dim,
            config.vocab_size,
            quant,
        );
    }
    // With shared weights the output tensor is omitted -> tied embeddings.
    Ok(())
}

fn run(
    model_path: &std::path::Path,
    tok_path: &std::path::Path,
    out_path: &std::path::Path,
    quant: GgmlType,
) -> Result<String> {
    let model = std::fs::read(model_path)?;
    let tok = std::fs::read(tok_path)?;

    let config = parse_config(&model)?;
    let head_size = config.dim / config.n_heads;
    eprintln!(
        "config: dim={} hidden={} layers={} heads={} kv_heads={} vocab={} seq_len={} shared={}",
        config.dim,
        config.hidden_dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.vocab_size,
        config.seq_len,
        config.shared_weights
    );

    let (pieces, kinds, scores) = parse_tokenizer(&tok, config.vocab_size)?;

    let mut w = GgufWriter::new();
    w.set_str("general.architecture", "llama");
    w.set_str(
        "general.name",
        model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("llama2.c model"),
    );
    w.set_u64("llama.context_length", config.seq_len as u64);
    w.set_u64("llama.embedding_length", config.dim as u64);
    w.set_u64("llama.block_count", config.n_layers as u64);
    w.set_u64("llama.feed_forward_length", config.hidden_dim as u64);
    w.set_u64("llama.attention.head_count", config.n_heads as u64);
    w.set_u64("llama.attention.head_count_kv", config.n_kv_heads as u64);
    w.set_u64("llama.rope.dimension_count", head_size as u64);
    w.set_f32("llama.rope.freq_base", 10_000.0);
    w.set_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
    w.set_str("tokenizer.ggml.model", "llama");
    w.set_str_vec("tokenizer.ggml.tokens", pieces);
    w.set_i32_vec("tokenizer.ggml.token_type", kinds);
    w.set_f32_vec("tokenizer.ggml.scores", scores);
    w.set_u64("tokenizer.ggml.bos_token_id", 1);
    w.set_u64("tokenizer.ggml.eos_token_id", 2);
    w.set_u64("tokenizer.ggml.unk_token_id", 0);
    w.set_bool("tokenizer.ggml.add_bos_token", true);

    write_weights(&mut w, &model, &config, head_size, quant)?;

    let n_tensors = w.tensor_count();
    w.write_to_file(out_path, 32)?;
    let out_len = std::fs::metadata(out_path)?.len();
    Ok(format!(
        "{n_tensors} tensors, {:.1} MiB, quant={quant}",
        out_len as f64 / 1024.0 / 1024.0
    ))
}
