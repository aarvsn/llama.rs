# CPU backend

The CPU backend is the first and currently only execution backend. It
is complete, validated, and portable.

## Kernel layers

1. **Scalar reference kernels** (`llama-kernels::vecdot`) for every
   supported weight type. Integer products accumulate exactly in `i32`
   before the block scales are applied, matching ggml numerics.
2. **SIMD kernels** (`llama-kernels::simd`), compiled for the baseline ISA
   and enabled at runtime:
   - x86-64: **AVX2** (+ **FMA** for the f32 dot, **F16C** for f16→f32
     conversion).
   - aarch64: NEON kernels are planned; today aarch64 runs the scalar
     path (still fully functional).
3. **Dispatch** (`vec_dot_row`): picks the best kernel for the
   (weight type, activation) pair. `LLAMA_RS_NO_SIMD=1` forces the scalar
   path — useful for debugging numerical differences.

## Parallelism

Matrix products split work over output channels (`rayon`
`par_chunks_exact_mut`); each thread streams its weight rows once while
reusing the shared activation block. The thread count comes from
`EngineOptions::threads` (default: logical CPUs, capped at 16) and
configures the global rayon pool at first engine load.

Prompt prefill is additionally batched (up to 256 tokens per forward
call), which amortizes the per-call overhead across the QKV/FFN matmuls.

## Quantized execution path

For a weight row of type T and an activation row `x`:

1. `x` is quantized **once** to Q8_0 blocks (32 elements per block,
   per-block f16 scale) — reused for every output channel of that matmul.
2. Each output channel computes a blockwise integer dot:
   `Q4_0`: `Σ (nibble−8)·q`; `Q4_1`: `(Σ nibble·q)·d_w·d_x + m_w·d_x·Σq`;
   `Q5_*`: adds the high-bit term; `Q8_0`: plain `Σ w·q`; K-quants:
   per-sub-block affine/scale forms.
3. F32/F16 weights skip activation quantization entirely and dot directly
   against the f32 row.

Whole-weight dequantization never happens.

## Measured performance

Runner: 2 vCPU x86-64 (AVX-512 available to the runtime detector), Rust
stable, `--release`. Kernel micro-benchmarks (4096-element dots,
`cargo bench`):

| Kernel | Time | Relative |
| --- | --- | --- |
| f32 scalar reference | 4.67 µs | 1× |
| f32 AVX2+FMA | 0.61 µs | 7.6× |
| f16 AVX2+F16C | 0.34 µs | 13.7× |
| q8_0 × q8_0 AVX2 | 0.53 µs | — |
| q4_0 × q8_0 AVX2 | 1.63 µs | — |
| q6_K × q8_0 (scalar) | 2.85 µs | — |

Full-model numbers (stories15M, Q8_0, greedy, 2 threads): ~240 tokens/s
decode, ~245 tokens/s prefill. These are **measured** figures from the
development machine, not marketing numbers — rerun `cargo bench` and
`llama run --stats` on your own hardware.

## Portability rules

- No `#[target_feature]` requirements at the crate level: binaries run on
  any x86-64/aarch64 CPU.
- Every `unsafe` block is a documented SIMD intrinsic call with a SAFETY
  comment; there is no `unsafe` slice transmutation of mapped weights
  (quantized blocks are parsed from `&[u8]` with unaligned-safe loads).
- aarch64/Android: the code avoids desktop-only dependencies; mmap and
  threading go through memmap2/rayon which support those targets.
