# Model support

## Implemented: the Llama family

`general.architecture = "llama"` — the decoder-only transformer with
RMSNorm, rotary embeddings, grouped-query attention, SwiGLU FFN and tied or
untied output embeddings.

Validated configurations include:

- **Llama 1 / 2 / 3 style checkpoints** (via any HF→GGUF converter).
- **Mistral v0.1-style** 7B layouts (same operator set).
- **TinyLlama-1.1B** (GQA, 2048 context).
- **Karpathy stories15M / stories260K** (via `convert_llama2c`).
- Any llama-arch GGUF with:
  - `head_count_kv < head_count` (GQA) or equal (MHA),
  - custom `rope.freq_base` (e.g. 500000 for Llama 3),
  - partial rotary dimensions (`rope.dimension_count < head_dim`),
  - tied embeddings (no `output.weight`) or separate output head,
  - any mix of the supported quantization types per tensor.

The forward pass was validated **bit-exactly against llama2.c** (same
model, same prompt, identical logits to 4 decimal places) and produces the
same tokenizations as the reference tokenizer for the tested prompts.

### RoPE convention

llama.rs applies rotary embeddings with **consecutive pairing**
`(x[2i], x[2i+1])`, frequency `base^(-2i/n_rot)` — the ggml
`GGML_ROPE_TYPE_NORMAL` layout. HF→GGUF converters permute Q/K weights into
this order, so GGUF llama models expect it. (The NeoX half-split style is
implemented too, for future architectures.)

## Not implemented (explicit errors)

Any other architecture is rejected at load with its name:

- Qwen2 / Qwen2.5, Gemma, Phi-3, StableLM, Falcon, Baichuan, OLMo, ...
- Encoder-decoder and non-transformer models (Mamba, RWKV).

The error message lists what is supported. Adding an architecture means:
arch detection entry, config keys, weight names/shapes, tokenizer quirks
and a forward pass — plus tests with synthetic fixtures and, ideally, a
real-model validation like the llama one.

## Tokenizer support

| `tokenizer.ggml.model` | Status |
| --- | --- |
| `llama` (SentencePiece unigram) | Supported: ▁ metaspace, dummy prefix, score-priority merges, byte fallback (`<0xXX>`), special-token parsing |
| `gpt2` (byte-level BPE) | Supported: GPT-2 byte encoder, merge ranks, pre-tokenizers `default`, `llama3`, `qwen2` (and their aliases) |

Other models (`wpm`, `ugm`, `rwkv`, ...) produce explicit errors.

Known approximation: letter/number/whitespace classes in the pre-tokenizer
scanners use Rust's Unicode predicates (`is_alphabetic` etc.) rather than
the exact `\p{L}`/`\p{N}` regex classes; the difference affects a few
combining marks only.

## Chat templates

Templates are matched by family: **llama2** (`[INST]`), **llama3**
(`<|start_header_id|>`), **ChatML** (`<|im_start|>`), **Gemma**
(`<start_of_turn>`) and **plain**. An unrecognized template errors with the
supported options; `llama chat --chat-template <name>` overrides the
detected family.

## Test models

- **Synthetic fixtures** (`llama-fixtures` crate): deterministic tiny llama
  models in F32/F16/Q8_0/Q4_0/K-quant mixes. Generated on the fly; CI never
  downloads weights.
- **stories15M** (TinyStories, 15M params, Apache-2.0 via the llama2.c
  project): used by the optional `real_model` integration tests
  (`LLAMA_RS_TEST_MODEL=/path/to/model.gguf cargo test --test real_model
  -- --ignored`).
