# Contributing to llama.rs

Thanks for your interest.

## Ground rules

- **Native implementation**: the inference path (GGUF parsing, kernels,
  tokenizer, transformer, generation) must remain implemented in this
  repository. Wrapping llama.cpp/Candle/Burn as the compute backend is out
  of scope; supporting libraries (serde, memmap2, rayon, axum, ...) are
  fine.
- **No fake features**: don't add a flag that claims GPU/quant/arch
  support without a real implementation and tests. Unsupported paths must
  error explicitly.
- **No silent degradation**: errors are surfaced, never swallowed.
- **Minimal unsafe**: every `unsafe` block needs a `SAFETY` comment.

## Workflow

1. Fork, create a feature branch.
2. Make your change with tests.
3. Run the full gate locally:
   ```sh
   cargo fmt
   cargo test --workspace
   cargo clippy --all-targets --all-features -- -D warnings
   cargo doc --no-deps
   ```
4. Open a PR describing what changed and how it was validated.

CI runs the same gate on Linux, macOS and Windows.

## Adding a quantization type

1. Block geometry in `llama-core::ggml_type` (size constants from the
   ggml reference).
2. `dequantize_row_*` + `vec_dot_*_q8_0` in `llama-kernels` (transcribe
   from the ggml reference, cite it in a comment).
3. Regenerate reference values:
   `python3 scripts/generate_reference_values.py > tests/reference_values.json`
   and extend the generator for the new type.
4. Fixture support in `llama-fixtures`, pipeline test, docs table.

## Adding a model architecture

1. `Architecture` detection entry + explicit error for anything else.
2. Config keys and validation in a `config.rs`.
3. Weight names/shapes validation and forward pass in the model crate.
4. Tokenizer quirks if any; chat template family.
5. Synthetic fixture + tests; a real-model validation is strongly
   preferred (see docs/development.md for the llama precedent).

## Reporting issues

Include: OS, CPU, Rust version, the exact command, the full error message,
and (if relevant) how the GGUF was produced. For numerical issues, a
minimal repro with a small model is gold.

## Security

See SECURITY.md — report vulnerabilities privately, not via public issues.
