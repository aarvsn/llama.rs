//! Optional integration test against a REAL GGUF model.
//!
//! Skipped unless `LLAMA_RS_TEST_MODEL` points to a GGUF file. The sandbox /
//! release workflow can download a small model (e.g. the 15M-parameter
//! `stories15M` in Q8_0) and set the variable to validate true end-to-end
//! quality: coherent text, real tokenizer, llama.cpp-compatible layouts.

use llama_runtime::{Engine, GenerationConfig, StopReason};

#[test]
#[ignore = "set LLAMA_RS_TEST_MODEL=/path/to/model.gguf to enable"]
fn generates_coherent_text_from_real_model() {
    let model_path = match std::env::var("LLAMA_RS_TEST_MODEL") {
        Ok(p) => p,
        Err(_) => return,
    };

    let engine = Engine::load(&model_path).expect("load real model");
    eprintln!(
        "model: {} ({} params, {} tensors)",
        engine.info().name.as_deref().unwrap_or("?"),
        engine.info().n_parameters,
        engine.info().n_tensors
    );

    let config = GenerationConfig::default()
        .max_tokens(48)
        .temperature(0.0) // greedy for reproducibility
        .greedy(true);
    let output = engine
        .generate("Once upon a time", &config)
        .expect("generate");

    eprintln!("--- generated ---\n{}\n-----------------", output.text);
    assert!(!output.text.is_empty(), "real model produced no text");
    assert!(
        output.tokens.len() == 48 || output.stop_reason == StopReason::Eos,
        "unexpected stop: {:?} after {} tokens",
        output.stop_reason,
        output.tokens.len()
    );

    // Coherence heuristic for story models: mostly letters/spaces.
    let letters = output
        .text
        .chars()
        .filter(|c| c.is_alphabetic() || c.is_whitespace())
        .count();
    let ratio = letters as f32 / output.text.chars().count().max(1) as f32;
    assert!(
        ratio > 0.7,
        "output does not look like text: {:?}",
        output.text
    );
}

#[test]
#[ignore = "set LLAMA_RS_TEST_MODEL to enable"]
fn real_model_tokenizer_round_trip() {
    let model_path = match std::env::var("LLAMA_RS_TEST_MODEL") {
        Ok(p) => p,
        Err(_) => return,
    };
    let engine = Engine::load(&model_path).expect("load");
    let tokenizer = engine.tokenizer();
    for text in ["Once upon a time", "Hello, world! 123"] {
        let tokens = tokenizer.encode(text, true, true).expect("encode");
        let decoded = tokenizer.decode(&tokens, false);
        assert_eq!(decoded.trim_start(), text, "round trip failed for {text:?}");
    }
}
