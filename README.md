### llama.rs

llama.rs is a local LLM inference engine written entirely in Rust. It loads GGUF models, runs inference on the CPU, and provides both a command-line interface and an OpenAI-compatible HTTP API.

The project implements its own GGUF loader, tokenizer, tensor operations, quantized matrix multiplication, sampling, and text generation. It does not depend on Python, llama.cpp, or another inference runtime.

```sh
$ llama run stories15M-q8_0.gguf --prompt "Once upon a time" --greedy
```

### Features

* GGUF support: Reads GGUF v2 and v3 files, including metadata, tensor directories, alignment, and memory-mapped weights.

* Quantized inference: Supports Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, and Q6_K.

* CPU acceleration: Uses AVX2, FMA, and F16C where available, with a scalar fallback for other CPUs.

* Multithreading: Parallelizes matrix multiplication across CPU threads using Rayon.

* Memory management: Maps model weights into memory and allocates the KV cache once per session. An optional F16 cache mode is available.

* Tokenization: Supports SentencePiece and byte-level BPE tokenization using GGUF metadata.

* Text generation: Includes temperature, top-k, top-p, min-p, repetition penalties, and seeded sampling.

* Chat support: Uses model chat templates and reuses cached prompt prefixes across turns.

* CLI: Provides commands for inspecting models, generating text, and interactive chat.

* HTTP API: Exposes an OpenAI-compatible API with streaming support.

### Installation

Prebuilt binaries are available on the [GitHub Releases](https://github.com/aarvsn/llama.rs/releases?utm_source=chatgpt.com) page.

### Build from source

Install stable Rust 1.85 or newer using [rustup](https://rustup.rs?utm_source=chatgpt.com).

```sh
git clone https://github.com/aarvsn/llama.rs
cd llama.rs

cargo build --release
```

The compiled binaries are available at:

```text
target/release/llama
target/release/llama-server
```

No additional system libraries or GPU SDKs are required.

Run the test suite:

```sh
cargo test --workspace
```

### Usage

### Command-line interface

Inspect a GGUF model:

```sh
llama inspect model.gguf
```

List its tensors:

```sh
llama inspect model.gguf --tensors
```

Generate text:

```sh
llama run model.gguf \
  --prompt "Explain Rust ownership" \
  --max-tokens 256
```

Configure sampling:

```sh
llama run model.gguf \
  --prompt "Hello" \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --max-tokens 256
```

Set the context size and thread count:

```sh
llama run model.gguf \
  --prompt "Hello" \
  --context-size 4096 \
  --threads 4
```

Display generation statistics:

```sh
llama run model.gguf --prompt "Hello" --stats
```

Start an interactive chat session:

```sh
llama chat model.gguf
```

Set a system prompt:

```sh
llama chat model.gguf --system "You are a concise assistant."
```

### HTTP server

Start the server:

```sh
llama-server --model model.gguf --port 8080
```

The server listens on `127.0.0.1` by default.

Available endpoints:

| Method | Endpoint             | Description              |
| ------ | -------------------- | ------------------------ |
| GET    | /health              | Server health            |
| GET    | /v1/models           | Loaded model information |
| POST   | /v1/completions      | Text completion          |
| POST   | /v1/chat/completions | Chat completion          |

Streaming responses are supported through Server-Sent Events (SSE).

Example request:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 64
  }'
```

### Rust library

The inference engine can also be used directly from Rust applications.

```rust
use llama_runtime::{Engine, GenerationConfig};

fn main() -> anyhow::Result<()> {
    let engine = Engine::load("model.gguf")?;

    let config = GenerationConfig::default()
        .max_tokens(128)
        .temperature(0.7);

    let output = engine.generate(
        "Explain how Rust works.",
        &config,
    )?;

    println!("{}", output.text);

    Ok(())
}
```

See [docs/api.md](https://docs/api.md?utm_source=chatgpt.com) for the library API, sessions, streaming, and generation statistics.

### Model support

The current implementation supports decoder-only models using the Llama architecture.

Tested model families include:

* Llama 1, 2, and 3

* Mistral v0.1-style models

* TinyLlama

* Stories15M and other small stories models

The implementation also supports grouped-query attention (GQA), tied embeddings, configurable RoPE parameters, and partial rotary dimensions.

Other architectures, including Qwen, Gemma, and Phi, are not supported yet. Attempting to load an unsupported architecture returns an error.

See [docs/model-support.md](https://docs/model-support.md?utm_source=chatgpt.com) for details.

### Quantization support

| Format           | Status        |
| ---------------- | ------------- |
| F32              | Supported     |
| F16              | Supported     |
| Q4_0             | Supported     |
| Q4_1             | Supported     |
| Q5_0             | Supported     |
| Q5_1             | Supported     |
| Q8_0             | Supported     |
| Q4_K             | Supported     |
| Q5_K             | Supported     |
| Q6_K             | Supported     |
| Q2_K, Q3_K, Q8_K | Not supported |
| IQ formats       | Not supported |
| BF16             | Not supported |

Q4_0, Q4_1, Q5_0, Q5_1, and Q8_0 have AVX2-accelerated kernels on supported x86-64 CPUs. K-quant formats currently use scalar kernels.

Quantization implementations are tested against reference values generated from the corresponding GGML routines.

### Platform support

| Platform       | Status                                          |
| -------------- | ----------------------------------------------- |
| Linux x86-64   | Tested in CI                                    |
| Windows x86-64 | Built and tested in CI                          |
| macOS x86-64   | Built and tested in CI                          |
| macOS ARM64    | Built and tested in CI                          |
| Android ARM64  | Portable code prepared; JNI integration pending |

The current inference backend is CPU-only. GPU support for Vulkan, CUDA, and Metal has not been implemented.

See [docs/android.md](https://docs/android.md?utm_source=chatgpt.com) and [docs/gpu-backends.md](https://docs/gpu-backends.md?utm_source=chatgpt.com).

### Performance

Benchmarks were run on a 2-vCPU x86-64 environment with AVX-512 available, using two threads and the Stories15M Q8_0 model.

| Measurement       | Result        |
| ----------------- | ------------- |
| Greedy generation | ~240 tokens/s |
| Prompt processing | ~245 tokens/s |

Criterion microbenchmarks for 4096-element dot products:

| Operation | Time    |
| --------- | ------- |
| F32       | 0.61 µs |
| F16       | 0.34 µs |
| Q8_0      | 0.52 µs |
| Q4_0      | 1.6 µs  |

Results depend on the CPU, model, quantization format, and runtime configuration.

See [docs/benchmarking.md](docs/benchmarking.md) for benchmark details.

See [docs/architecture.md](docs/architecture.md) for the internal design.

### Limitations

The current implementation has a few limitations:

* Only the Llama architecture is supported.

* Inference runs on the CPU; GPU backends are not available.

* K-quant formats do not yet have SIMD-optimized kernels.

* Each session processes one sequence at a time.

* The server does not support continuous batching.

* Custom Jinja chat templates are not supported.

* Split GGUF models are not supported.

### Roadmap

Planned development includes:

* Support for Qwen2, Gemma, Phi-3, and additional architectures.

* SIMD-optimized K-quant kernels.

* ARM NEON acceleration.

* Vulkan, CUDA, and Metal backends.

* Continuous batching and parallel sequence processing.

* Android JNI integration.

### Contributing

Contributions are welcome.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [docs/development.md](https://docs/development.md?utm_source=chatgpt.com) for development instructions.

### License

llama.rs is released under the MIT License.
