# GGUF format support

llama.rs reads GGUF **v2** and **v3** (v1 is rejected with a clear error).
This document describes what is parsed, how validation works, and the
tensor types that are supported.

## File layout

```
+----------------+
| header         |  magic "GGUF" (u32 LE), version (u32), tensor_count (u64),
+----------------+  metadata_kv_count (u64)
| metadata KVs   |  key (string), value_type (u32), value
+----------------+
| tensor infos   |  name (string), n_dims (u32), dims[n_dims] (u64),
|                |  type (u32), offset (u64, relative to data section)
+----------------+  (padding to `general.alignment`, default 32)
| tensor data    |  raw serialized blocks, zero-copy accessible
+----------------+
```

- Dimensions are ordered **fastest-first**: a 2-D weight is
  `[ne0 = n_in, ne1 = n_out]` with `n_out` contiguous rows of `n_in`
  elements.
- Offsets are relative to the start of the (aligned) data section and must
  be multiples of the alignment.
- The reader validates every tensor's byte size against its shape/type
  geometry and rejects anything extending beyond the file.

## Metadata

All 13 GGUF value types are parsed (u8/i8/u16/i16/u32/i32/u64/i64/f32/f64/
bool/string/array) into a typed tree with checked accessors. Keys llama.rs
consumes:

| Key | Use |
| --- | --- |
| `general.architecture` | architecture detection (must be `llama`) |
| `general.alignment` | data-section alignment (validated, 8..=1 MiB) |
| `llama.context_length` | default context window |
| `llama.embedding_length` | hidden size (required) |
| `llama.block_count` | layer count (required) |
| `llama.feed_forward_length` | FFN size (required) |
| `llama.attention.head_count` | query heads (required) |
| `llama.attention.head_count_kv` | KV heads (default: head_count) |
| `llama.attention.key_length` | per-head key dim (default: n_embd/n_heads) |
| `llama.attention.value_length` | per-head value dim (default: key_length) |
| `llama.attention.layer_norm_rms_epsilon` | RMSNorm eps (default 1e-5) |
| `llama.rope.freq_base` | RoPE theta (default 10000) |
| `llama.rope.dimension_count` | rotary dims (default: head dim) |
| `tokenizer.ggml.model` | `llama` (SPM) or `gpt2` (BPE); anything else errors |
| `tokenizer.ggml.pre` | BPE pre-tokenizer family |
| `tokenizer.ggml.tokens` / `scores` / `token_type` | vocabulary |
| `tokenizer.ggml.merges` | BPE merge ranks |
| `tokenizer.ggml.*_token_id`, `add_bos_token`, `add_eos_token` | specials |
| `tokenizer.chat_template` | chat template (llama2/llama3/chatml/gemma) |
| `split.count` | detected; split models are rejected with a message |

Missing required keys and wrong-typed values produce specific errors.

## Tensor types

Block geometry (identical to ggml):

| Type | Id | Block | Block bytes | Layout |
| --- | --- | --- | --- | --- |
| F32 | 0 | 1 | 4 | — |
| F16 | 1 | 1 | 2 | — |
| Q4_0 | 2 | 32 | 18 | f16 d + packed nibbles (elem j = lo(j), elem j+16 = hi(j)) |
| Q4_1 | 3 | 32 | 20 | f16 d, f16 m + nibbles |
| Q5_0 | 6 | 32 | 22 | f16 d + u32 qh (bit j ↔ elem j) + nibbles |
| Q5_1 | 7 | 32 | 24 | f16 d, m + u32 qh + nibbles |
| Q8_0 | 8 | 32 | 34 | f16 d + 32 × i8 |
| Q4_K | 12 | 256 | 144 | f16 d, dmin + 12 B packed 6-bit scales + 128 B nibbles |
| Q5_K | 13 | 256 | 176 | Q4_K + 32 B high bits (mask bit 2g / 2g+1 of qh[l]) |
| Q6_K | 14 | 256 | 210 | f16 d + 128 B ql + 64 B qh + 16 × i8 scales |

The dequantization routines are transcriptions of the ggml reference
(`ggml-quants.c`) and are verified against independently generated reference
values — see `tests/reference_kernels.rs` and
`scripts/generate_reference_values.py`.

Unsupported types (Q2_K, Q3_K, Q8_K, IQ*, BF16, ...) fail at load with the
type name and id in the error message.

## Memory mapping

`GgufFile::open` mmaps the file read-only and validates the full directory
against the file length before returning. Tensor bytes are handed out as
`&[u8]` slices borrowed from the mapping; the engine never copies whole
tensors. `GgufFile::from_bytes` provides the same parsing for in-memory
buffers (used by tests).

## Writer

The GGUF v3 writer (`llama_gguf::GgufWriter`) powers the test fixtures, the
synthetic model generator and `examples/convert_llama2c.rs`, which converts
Karpathy llama2.c checkpoints (`.bin` + `tokenizer.bin`) to GGUF.
