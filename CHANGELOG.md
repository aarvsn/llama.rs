# Changelog

All notable changes to llama.rs are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses semantic versioning.


## [0.1.0] - 2026-11-07

First public release: a native Rust inference engine for GGUF models.

### Added
- GGUF v2/v3 reader with full metadata parsing, tensor directory
  validation, alignment handling and memory-mapped zero-copy weights;
  GGUF v3 writer for fixtures and converters.
- Tensor types: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K
  with ggml-compatible block layouts, reference dequantization and
  quantized dot products.
- AVX2/FMA/F16C kernels with runtime CPU detection and portable scalar
  fallbacks (`LLAMA_RS_NO_SIMD=1` forces scalar).
- Parallel batched matmul (rayon) and batched prompt prefill.
- SentencePiece (llama) tokenizer with metaspace handling, score-priority
  unigram merges, byte fallback and special-token parsing; byte-level BPE
  (gpt2/llama3/qwen2 pre-tokenizers); incremental UTF-8 detokenizer;
  chat-template families (llama2, llama3, chatml, gemma, plain).
- Llama-family transformer: RMSNorm, RoPE (interleaved pairing, validated
  bit-exactly against llama2.c), multi-head and grouped-query attention,
  causal masking, SwiGLU FFN, residual connections, tied/untied output.
- Per-session KV cache (f32/f16) with prefix reuse across turns.
- Sampling: temperature, top-k, top-p, min-p, repetition/frequency/
  presence penalties, greedy, seeded reproducible RNG, stop sequences,
  EOS detection, streaming callbacks.
- `llama` CLI: `inspect`, `run` (with `--stats`), `chat`.
- `llama-server`: OpenAI-compatible `/v1/completions`,
  `/v1/chat/completions` (SSE streaming), `/v1/models`, `/health`,
  localhost-by-default binding, graceful shutdown.
- Rust API (`llama-runtime`): Engine/Session/GenerationConfig, model
  metadata, performance statistics.
- Test suite: unit tests, independently generated kernel reference values
  (Python translation of the ggml reference), synthetic-fixture pipeline
  tests, optional real-model integration tests; 150+ tests total.
- Criterion benchmark suite; llama2.c → GGUF converter example; synthetic
  fixture generators; CI (fmt/check/test/clippy/doc on Linux, macOS,
  Windows), release and benchmark workflows.

### Fixed
- `llama run --no-stream` now works as a valueless flag
  (`ArgAction::SetFalse`) instead of demanding `--no-stream <true|false>`.
- Logs go to stderr; piped stdout contains only generated text.
- Documentation performance claims now match measured numbers on the
  documented runner (stories15M Q8_0, 2 threads: ~240 tok/s greedy decode,
  ~245 tok/s prefill); the README example shows actual greedy output.

### Added
- Regression tests: GGUF v2 loading + generation (version-field rewrite of
  a v3 fixture) and multi-session independence on a shared engine.
- Integration tests pass against the real `stories15M` Q8_0 GGUF
  (`LLAMA_RS_TEST_MODEL=... cargo test --test real_model -- --ignored`):
  coherent greedy text and tokenizer round-trip on a 32k-vocab SentencePiece
  model converted by llama.cpp conventions.

### Fixed
- Generation no longer commits the EOS token to the session context or the
  completion output: EOS now stops the turn before anything is recorded,
  keeping the KV cache exactly in sync with the token history (llama.cpp
  semantics). A deterministic zero-weight fixture with a custom EOS id
  exercises the full stop path in CI.
- `llama-server` reports truthful per-request `usage` numbers on the shared
  session (the cumulative prefill counter no longer leaks into
  `prompt_tokens`/`total_tokens`), and `finish_reason` now reflects the real
  stop reason (`length` vs `stop`) instead of a hard-coded value.
- Rust 2024 `unsafe_op_in_unsafe_fn` violations in three AVX2 kernels; the
  workspace `unsafe_code` lint policy is now enforced with targeted,
  documented allows in the kernels crate (CPU intrinsics) and the GGUF mmap
  loader.
- Clippy is clean under `-D warnings` across the workspace (all targets,
  all features): loop rewrites, literal separators, dead-field removal,
  import hygiene, and justified allows where domain naming (ggml block
  math, Q/K/V projections) or wire formats (OpenAI `*_tokens` fields)
  demand it. CI's clippy gate now passes for real.
- Removed a stray debug print hook from the transformer forward pass.

### Changed
- `llama-fixtures` supports `zero_weights` and custom `eos_token_id`
  fixture options; fixture builders are split into testable helpers.

