# Rust API

The public API lives in `llama-runtime` (re-exported through the `llama`
facade crate).

## Loading and one-shot generation

```rust
use llama_runtime::{Engine, GenerationConfig};

let engine = Engine::load("model.gguf")?;

let config = GenerationConfig::default()
    .max_tokens(128)
    .temperature(0.7)
    .seed(42);

let output = engine.generate("Explain how Rust works.", &config)?;
println!("{}", output.text);
println!("stopped: {:?}", output.stop_reason);
```

`Engine::load_with` accepts options:

```rust
use llama_runtime::EngineOptions;

let engine = Engine::load_with(
    "model.gguf",
    EngineOptions::default()
        .with_threads(8)
        .with_context_size(4096)
        .with_kv_cache_f16(true),
)?;
```

## Sessions: reuse, streaming, chat

`Engine::generate` allocates a fresh session each call. For interactive
use, keep a `Session` — it owns the KV cache and reuses the longest common
prefix when the prompt grows:

```rust
use llama_runtime::{Engine, Flow, GenerationConfig};

let engine = Engine::load("model.gguf")?;
let mut session = engine.create_session();

session.set_prompt("You are a helpful assistant.", true)?;

// Stream tokens as they are generated.
let out = session.generate(
    &GenerationConfig::default().max_tokens(64),
    |token, piece| {
        print!("{piece}");
        Flow::Continue // or Flow::Stop to end early
    },
)?;

// Extend the conversation: only the new suffix is prefilled.
session.set_prompt("You are a helpful assistant. Hello!", true)?;
```

Chat with the model's template (detected from GGUF metadata):

```rust
use llama_tokenizer::Message;

session.push_chat_turn(
    &[Message::new("user", "Hello!")],
    /* add_generation_prompt */ true,
    /* parse_special */ true,
)?;
let out = session.generate(&GenerationConfig::default(), |_, _| Flow::Continue)?;
```

## Configuration reference

`GenerationConfig`:

| Field | Default | Meaning |
| --- | --- | --- |
| `max_tokens` | 256 | token budget |
| `temperature` | 0.8 | 0 = greedy |
| `top_k` | 40 | 0 disables |
| `top_p` | 0.95 | nucleus sampling |
| `min_p` | 0.05 | relative probability floor |
| `repeat_penalty` | 1.1 | 1.0 disables |
| `frequency_penalty` | 0.0 | per-occurrence logit penalty |
| `presence_penalty` | 0.0 | one-time logit penalty |
| `seed` | entropy | fixed seed = reproducible sampling |
| `stop` | — | stop sequences (output truncated before them) |
| `greedy` | false | argmax decoding |
| `parse_special` | true | recognize special tokens in prompts |
| `emit_special` | false | render control-token text in output |

All values are validated; out-of-range settings return
`LlamaError::InvalidArgument`.

## Metadata, model inspection

```rust
let info = engine.info();
println!("{}: {} params, {} tensors, quant {:?}",
    info.name.as_deref().unwrap_or("?"),
    info.n_parameters, info.n_tensors, info.quant_types);

let tokenizer = engine.tokenizer();
let ids = tokenizer.encode("hello", /* add_special */ true, /* parse_special */ true)?;
let text = tokenizer.decode(&ids, /* special */ false);
assert_eq!(text.trim_start(), "hello");
```

## Performance statistics

```rust
let out = session.generate(&config, |_, _| Flow::Continue)?;
let stats = &out.stats;
println!("prefill: {:.1} tok/s", stats.prefill_tps());
println!("decode:  {:.1} tok/s", stats.decode_tps());
println!("kv cache: {:.1} MiB", stats.kv_cache_bytes as f64 / 1048576.0);
println!("context: {}/{}", stats.context_tokens, stats.context_capacity);
```

Engine-level: `engine.stats()` reports load time and model bytes.

## Error handling

Every fallible operation returns `llama_core::Result<T>` (= `Result<T,
LlamaError>`). Variants distinguish invalid GGUF files, unsupported
formats/architectures/tokenizers, missing tensors, runtime limits and
invalid arguments — messages are specific and actionable. Unsupported
features never degrade silently.
