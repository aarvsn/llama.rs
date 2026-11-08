# Benchmarking

llama.rs distinguishes two kinds of measurements, and never mixes them:

1. **Kernel micro-benchmarks** — Criterion suites over synthetic data.
2. **Full-model benchmarks** — real models through the CLI/server.

No benchmark result in this repository is invented; everything quoted in
the README/docs was produced by the commands below on the development
machine (2 vCPU x86-64 runner with AVX-512, Rust stable). Rerun them on
your hardware.

## Kernel micro-benchmarks

```sh
cargo bench --bench engine
```

Benchmarks included:

| Group | Benchmark | What it measures |
| --- | --- | --- |
| `vec_dot` | `f32_dispatched`, `f32_scalar_reference`, `f16_dispatched`, `q8_0_q8_0`, `q4_0_q8_0`, `q6_k_q8_0` | 4096-element dot products (SIMD vs scalar) |
| `dequantize_row` | `q4_0` | block → f32 dequantization throughput |
| `quantize` | `q8_0` | activation quantization |
| `ops` | `rmsnorm`, `rope_head_64`, `softmax_2048` | transformer pointwise ops |
| `tokenizer` | `spm_encode`, `spm_decode` | encode/decode throughput (bytes/s) |
| `engine` | `load_fixture`, `generate_32_tokens` | end-to-end on the synthetic fixture |

These run on **synthetic fixtures generated on the fly** (the
`llama-fixtures` crate); no model downloads are needed and results are
reproducible (fixed seeds).

Results from the development runner (see docs/cpu-backend.md for the
table): the AVX2 f32 dot is ~7.6× the scalar reference; the SIMD kernels
are selected at runtime (`LLAMA_RS_NO_SIMD=1` disables them to measure the
portable path).

## Full-model benchmarks

Use a real GGUF file with the CLI's built-in statistics:

```sh
llama run model.gguf --prompt "Once upon a time" \
  --max-tokens 128 --greedy --stats --threads 4
```

Reported: prompt tokens and tokens/s (prefill), generated tokens and
tokens/s (decode), stop reason, context usage, KV-cache and model memory.
The HTTP server reports the same fields in `usage`/timings where
applicable.

For a rough apples-to-apples comparison against llama.cpp on the same
machine, use the same model file, thread count, greedy decoding and prompt.

### Reference results (stories15M, Q8_0, 2 threads)

- Prompt processing: ~245 tokens/s
- Greedy decode: ~240 tokens/s
- Model load: ~0.01 s (mmap + validation of a 25 MiB file)

These figures are from the CI-class runner described above; larger models
scale with parameter count and memory bandwidth.

## What is deliberately *not* claimed

- No GPU numbers (no GPU backend exists).
- No comparisons on hardware we did not run.
- No tokens/s promises for models we did not measure.
