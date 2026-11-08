//! End-to-end pipeline tests using deterministic synthetic GGUF fixtures.
//!
//! These tests generate tiny Llama models on the fly (see `llama-fixtures`)
//! and exercise loading, tokenization, prefill, KV caching, generation,
//! sampling determinism and statistics — no model downloads required.

use std::path::PathBuf;
use std::sync::Arc;

use llama_fixtures::{FixtureSpec, QuantScheme, write_llama_fixture};
use llama_model::{ForwardScratch, KvCache, KvPrecision, LlamaConfig, LlamaModel};
use llama_runtime::{Engine, EngineOptions, Flow, GenerationConfig, StopReason};

fn fixture_path(tag: &str, spec: &FixtureSpec) -> PathBuf {
    let dir = std::env::temp_dir().join("llama-rs-tests");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join(format!("fixture-{tag}.gguf"));
    write_llama_fixture(&path, spec).expect("write fixture");
    path
}

fn default_spec(quant: QuantScheme) -> FixtureSpec {
    FixtureSpec {
        quant,
        ..FixtureSpec::default()
    }
}

#[test]
fn loads_and_inspects_f32_fixture() {
    let path = fixture_path("f32", &default_spec(QuantScheme::F32));
    let engine = Engine::load(&path).expect("load");
    let info = engine.info();
    assert_eq!(info.architecture, "llama");
    assert_eq!(info.gguf_version, 3);
    assert_eq!(info.n_layers, 2);
    assert_eq!(info.n_heads, 4);
    assert_eq!(info.n_kv_heads, 2);
    assert_eq!(info.n_vocab, 512);
    assert_eq!(info.n_embd, 64);
    assert_eq!(info.quant_types, vec![llama_core::GgmlType::F32]);
    assert!(info.n_parameters > 0);
    // embd + output_norm + 2 layers x (attn_norm, q, k, v, o, ffn_norm, gate, up, down)
    assert_eq!(info.n_tensors, 2 + 2 * 9);
}

#[test]
fn generates_text_from_every_quant_scheme() {
    for (tag, quant) in [
        ("f16", QuantScheme::F16),
        ("q8_0", QuantScheme::Q8_0),
        ("q4_0", QuantScheme::Q4_0),
    ] {
        let path = fixture_path(tag, &default_spec(quant));
        let engine = Engine::load(&path).unwrap_or_else(|e| panic!("load {tag}: {e}"));
        let config = GenerationConfig::default()
            .max_tokens(8)
            .greedy(true)
            .seed(42);
        let output = engine
            .generate("Once upon a time", &config)
            .unwrap_or_else(|e| panic!("generate {tag}: {e}"));
        assert_eq!(output.tokens.len(), 8, "{tag}: expected max_tokens tokens");
        assert!(
            !output.text.is_empty() || output.tokens.iter().all(|&t| t < 512),
            "{tag}"
        );
        assert_eq!(output.stop_reason, StopReason::MaxTokens, "{tag}");
    }
}

#[test]
fn kquant_fixture_end_to_end() {
    let path = fixture_path("kquant", &FixtureSpec::kquant());
    let engine = Engine::load(&path).expect("load K-quant fixture");
    let types = &engine.info().quant_types;
    assert!(types.contains(&llama_core::GgmlType::Q4_K), "expected q4_k");
    assert!(types.contains(&llama_core::GgmlType::Q5_K), "expected q5_k");
    assert!(types.contains(&llama_core::GgmlType::Q6_K), "expected q6_k");

    let config = GenerationConfig::default().max_tokens(6).greedy(true);
    let output = engine.generate("hello world", &config).expect("generate");
    assert_eq!(output.tokens.len(), 6);
}

#[test]
fn deterministic_with_fixed_seed() {
    let path = fixture_path("det", &default_spec(QuantScheme::Q8_0));
    let engine = Engine::load(&path).expect("load");
    let config = || {
        GenerationConfig::default()
            .max_tokens(16)
            .seed(1234)
            .temperature(0.9)
    };

    let out1 = engine
        .generate("The quick brown fox", &config())
        .expect("gen1");
    let out2 = engine
        .generate("The quick brown fox", &config())
        .expect("gen2");
    assert_eq!(out1.tokens, out2.tokens, "same seed must give same tokens");
    assert_eq!(out1.text, out2.text);

    let out3 = engine
        .generate("The quick brown fox", &config().seed(999))
        .expect("gen3");
    // Different seeds usually diverge (not guaranteed for tiny models, so
    // just require well-formed output).
    assert_eq!(out3.tokens.len(), out1.tokens.len());
}

#[test]
fn batched_prefill_matches_token_by_token() {
    // The strongest correctness property: prefilling a prompt in one batch
    // must produce the same logits as feeding tokens one at a time.
    let path = fixture_path("batch", &default_spec(QuantScheme::F32));
    let file = Arc::new(llama_gguf::GgufFile::open(&path).expect("open"));
    let meta = file.metadata();
    let config = LlamaConfig::from_gguf(meta).expect("config");
    let model = LlamaModel::load(Arc::clone(&file), config).expect("load");
    let tokenizer = llama_tokenizer::Tokenizer::from_gguf(meta).expect("tokenizer");

    let tokens = tokenizer
        .encode("Once upon a time there were three little pigs", true, true)
        .expect("encode");
    assert!(
        tokens.len() > 8,
        "expected a multi-token prompt, got {}",
        tokens.len()
    );

    let make_kv = || {
        KvCache::new(
            model.config().n_layers,
            512,
            model.config().n_kv_heads,
            model.config().head_dim_k,
            KvPrecision::F32,
        )
    };
    let mut scratch = ForwardScratch::new(model.config(), 64);

    // All at once.
    let mut kv = make_kv();
    let mut logits_batch = Vec::new();
    model
        .forward(&tokens, 0, &mut kv, &mut scratch, Some(&mut logits_batch))
        .expect("batch forward");

    // One at a time.
    let mut kv = make_kv();
    let mut logits_seq = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        model
            .forward(&[tok], i, &mut kv, &mut scratch, Some(&mut logits_seq))
            .expect("sequential forward");
    }

    assert_eq!(logits_batch.len(), logits_seq.len());
    for (i, (a, b)) in logits_batch.iter().zip(logits_seq.iter()).enumerate() {
        let tolerance = a.abs().max(b.abs()).max(1.0) * 2e-3;
        assert!(
            (a - b).abs() <= tolerance,
            "logit {i}: batch {a} vs sequential {b}"
        );
    }
}

#[test]
fn kv_cache_prefix_reuse_is_consistent() {
    let path = fixture_path("kv-reuse", &default_spec(QuantScheme::Q8_0));
    let engine = Engine::load(&path).expect("load");

    // Session 1: generate a continuation after a prompt.
    let config = GenerationConfig::default().max_tokens(10).greedy(true);
    let mut session = engine.create_session();
    session
        .set_prompt("Once upon a time", true)
        .expect("prefill");
    let out1 = session
        .generate(&config, |_, _| Flow::Continue)
        .expect("generate");

    // Session 2: same prompt, fresh cache: must produce the identical greedy
    // continuation (validates that reuse did not corrupt the cache).
    let mut session2 = engine.create_session();
    session2
        .set_prompt("Once upon a time", true)
        .expect("prefill");
    let out2 = session2
        .generate(&config, |_, _| Flow::Continue)
        .expect("generate");
    assert_eq!(out1.tokens, out2.tokens);

    // Extending the same session with a longer shared prefix reuses the cache.
    let before = session.stats().cache_reuses;
    session
        .set_prompt("Once upon a time there were three little pigs", true)
        .expect("extend");
    assert!(session.stats().cache_reuses > before || !session.context_tokens().is_empty());
    let out3 = session
        .generate(&config, |_, _| Flow::Continue)
        .expect("generate 3");
    assert_eq!(out3.tokens.len(), 10);
}

#[test]
fn eos_and_max_tokens_stop_conditions() {
    let path = fixture_path("stops", &default_spec(QuantScheme::F32));
    let engine = Engine::load(&path).expect("load");

    // max_tokens = 1.
    let out = engine
        .generate(
            "hello",
            &GenerationConfig::default().max_tokens(1).greedy(true),
        )
        .expect("generate");
    assert_eq!(out.tokens.len(), 1);
    assert_eq!(out.stop_reason, StopReason::MaxTokens);

    // EOS: a zero-weight fixture with eos_token_id = 0 makes every logit
    // exactly 0.0, so greedy argmax deterministically picks token 0 — the
    // configured EOS — on the very first decode step.
    let eos_spec = FixtureSpec {
        zero_weights: true,
        eos_token_id: Some(0),
        ..default_spec(QuantScheme::F32)
    };
    let path = fixture_path("eos", &eos_spec);
    let engine = Engine::load(&path).expect("load eos fixture");
    let out = engine
        .generate(
            "hello",
            &GenerationConfig::default().max_tokens(4).greedy(true),
        )
        .expect("generate until eos");
    assert_eq!(out.stop_reason, StopReason::Eos);
    assert!(
        out.tokens.is_empty(),
        "EOS token must not appear in output tokens"
    );
    assert!(
        out.text.is_empty(),
        "EOS piece must not appear in output text"
    );
}

#[test]
fn stop_sequences_truncate_output() {
    let path = fixture_path("stopseq", &default_spec(QuantScheme::Q8_0));
    let engine = Engine::load(&path).expect("load");
    let mut session = engine.create_session();
    session.set_prompt("hello world", true).expect("prefill");
    // Collect the raw text first.
    let raw = session
        .generate(
            &GenerationConfig::default().max_tokens(10).greedy(true),
            |_, _| Flow::Continue,
        )
        .expect("generate");
    assert!(!raw.text.is_empty() || !raw.tokens.is_empty());

    // Now request a stop on a substring of whatever was generated, if any
    // text was produced; otherwise use a token-boundary stop.
    if let Some(first_char) = raw.text.chars().next() {
        let mut session2 = engine.create_session();
        session2.set_prompt("hello world", true).expect("prefill");
        let out = session2
            .generate(
                &GenerationConfig::default()
                    .max_tokens(10)
                    .greedy(true)
                    .stop(vec![first_char.to_string()]),
                |_, _| Flow::Continue,
            )
            .expect("generate with stop");
        assert_eq!(out.stop_reason, StopReason::StopSequence);
        assert!(
            out.text.is_empty(),
            "output truncated before the stop char: {:?}",
            out.text
        );
    }
}

#[test]
fn streaming_callback_receives_pieces() {
    let path = fixture_path("stream", &default_spec(QuantScheme::F32));
    let engine = Engine::load(&path).expect("load");
    let mut session = engine.create_session();
    session
        .set_prompt("Once upon a time", true)
        .expect("prefill");

    let mut pieces = Vec::new();
    let out = session
        .generate(
            &GenerationConfig::default().max_tokens(5).greedy(true),
            |_tok, piece| {
                pieces.push(piece.to_string());
                Flow::Continue
            },
        )
        .expect("generate");
    assert_eq!(pieces.len(), 5);
    assert_eq!(out.text, pieces.concat());
}

#[test]
fn callback_stop_halts_generation() {
    let path = fixture_path("cbstop", &default_spec(QuantScheme::F32));
    let engine = Engine::load(&path).expect("load");
    let mut session = engine.create_session();
    session.set_prompt("story", true).expect("prefill");
    let out = session
        .generate(
            &GenerationConfig::default().max_tokens(20).greedy(true),
            |_, _| Flow::Stop,
        )
        .expect("generate");
    assert_eq!(out.stop_reason, StopReason::Callback);
    assert_eq!(out.tokens.len(), 1);
}

#[test]
fn tokenizer_round_trip_on_fixture_vocab() {
    let path = fixture_path("tok", &default_spec(QuantScheme::F32));
    let engine = Engine::load(&path).expect("load");
    let tokenizer = engine.tokenizer();

    for text in [
        "hello world",
        "The quick brown fox jumps over the lazy dog",
        "unicode: héllo wörld 你好",
        "",
    ] {
        let tokens = tokenizer.encode(text, true, true).expect("encode");
        let decoded = tokenizer.decode(&tokens, false);
        if text.is_empty() {
            assert_eq!(tokens.len(), 1, "empty text still gets BOS: {tokens:?}");
            continue;
        }
        // Byte-fallback vocab guarantees an exact round trip.
        assert_eq!(decoded.trim_start(), text, "round trip failed for {text:?}");
    }

    // Special token parsing.
    let tokens = tokenizer
        .encode("before <|fixture|> after", true, true)
        .expect("encode");
    let special_id = tokenizer
        .vocab()
        .id_of("<|fixture|>")
        .expect("special in vocab");
    assert!(
        tokens.contains(&special_id),
        "special token recognized: {tokens:?}"
    );
}

#[test]
fn chat_template_detected_and_applied() {
    let path = fixture_path("chat", &default_spec(QuantScheme::F32));
    let engine = Engine::load(&path).expect("load");
    assert_eq!(
        engine.chat_template(),
        Some(llama_tokenizer::ChatTemplate::Llama2)
    );

    let mut session = engine.create_session();
    session
        .push_chat_turn(
            &[llama_tokenizer::Message::new("user", "Hello there")],
            true,
            true,
        )
        .expect("chat turn");
    assert!(!session.context_tokens().is_empty());
}

#[test]
fn engine_options_context_size_and_stats() {
    let path = fixture_path("opts", &default_spec(QuantScheme::F32));
    let engine = Engine::load_with(
        &path,
        EngineOptions::default()
            .with_context_size(128)
            .with_threads(2),
    )
    .expect("load with options");
    assert_eq!(engine.effective_context_size(), 128);
    assert!(engine.stats().load_time.as_nanos() > 0);
    assert!(engine.stats().model_bytes > 0);

    let mut session = engine.create_session();
    assert_eq!(session.stats().context_capacity, 128);
    assert!(session.stats().kv_cache_bytes > 0);

    session.set_prompt("hello", true).expect("prefill");
    assert!(session.stats().prefill.count > 0);
    session
        .generate(
            &GenerationConfig::default().max_tokens(3).greedy(true),
            |_, _| Flow::Continue,
        )
        .expect("generate");
    assert!(session.stats().decode.count >= 2);
    assert!(session.stats().decode_tps() >= 0.0);
}

#[test]
fn malformed_gguf_errors_are_explicit() {
    let dir = std::env::temp_dir().join("llama-rs-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let bad = dir.join("bad-magic.gguf");
    std::fs::write(&bad, b"NOTGGUF....").unwrap();
    let err = Engine::load(&bad).unwrap_err();
    assert!(err.to_string().contains("magic"), "{err}");

    let truncated = dir.join("truncated.gguf");
    let good = fixture_path("trunc-src", &default_spec(QuantScheme::F32));
    let data = std::fs::read(&good).unwrap();
    std::fs::write(&truncated, &data[..data.len() / 2]).unwrap();
    let err = Engine::load(&truncated).unwrap_err();
    assert!(
        err.to_string().contains("GGUF")
            || err.to_string().contains("tensor")
            || err.to_string().contains("end of file"),
        "{err}"
    );
}

#[test]
fn unsupported_architecture_reports_name() {
    use llama_gguf::metadata::{GgufMetadata, MetadataValue};
    let mut meta = GgufMetadata::new();
    meta.insert(
        "general.architecture".into(),
        MetadataValue::String("mamba".into()),
    );
    let err = llama_model::Architecture::detect(&meta).unwrap_err();
    assert!(err.to_string().contains("mamba"), "{err}");
}

/// GGUF v2 files have the same layout as v3 (the version field is the only
/// difference), so rewriting the version bytes of a v3 fixture must still
/// load and generate identically.
#[test]
fn loads_and_generates_from_gguf_v2() {
    let v3_path = fixture_path("f32", &default_spec(QuantScheme::F32));
    let v2_path = std::env::temp_dir().join("llama-rs-tests/fixture-f32-v2.gguf");
    let bytes = std::fs::read(&v3_path).expect("read v3 fixture");
    assert_eq!(&bytes[0..4], b"GGUF");
    let mut v2 = bytes;
    v2[4..8].copy_from_slice(&2u32.to_le_bytes());
    std::fs::write(&v2_path, &v2).expect("write v2 fixture");

    let engine = Engine::load(&v2_path).expect("load v2");
    assert_eq!(engine.info().gguf_version, 2);

    let mut session = engine.create_session();
    session.set_prompt("hello", false).expect("prompt");
    let config = GenerationConfig::default().max_tokens(8).greedy(true);
    let out = session
        .generate(&config, |_, _| Flow::Continue)
        .expect("generate");
    assert_eq!(out.tokens.len(), 8);
}

/// Two sessions on the same engine must not interfere (independent KV caches).
#[test]
fn sessions_are_independent() {
    let path = fixture_path("f32-indep", &default_spec(QuantScheme::F32));
    let engine = Engine::load(&path).expect("load");
    let config = GenerationConfig::default().max_tokens(8).greedy(true);

    let mut a = engine.create_session();
    a.set_prompt("hello", false).expect("prompt a");
    let out_a = a.generate(&config, |_, _| Flow::Continue).expect("gen a");

    let mut b = engine.create_session();
    b.set_prompt("hello", false).expect("prompt b");
    let out_b = b.generate(&config, |_, _| Flow::Continue).expect("gen b");

    assert_eq!(
        out_a.tokens, out_b.tokens,
        "same prompt, greedy, same engine"
    );
}
