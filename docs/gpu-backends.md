# GPU backends

**Status: not implemented. Planned.**

llama.rs is CPU-only today. This document describes the backend
architecture that GPU work will plug into, and the rules that govern how
backends may be advertised.

## Design

The device abstraction lives in `llama-core::device`:

- `Backend` — an execution target (Cpu, Vulkan, Cuda, Metal).
- `DeviceInfo` — what a loaded engine runs on (backend, threads, detected
  CPU features, description).
- `CpuFeatures` — runtime CPU capability probe used by the kernel dispatch.

Backend selection will be per-feature-gated crates
(`llama-vulkan`, `llama-cuda`, `llama-metal`, ...) exposing a common
interface for:

- buffer allocation and GGUF uploads,
- the primitive ops the Llama forward pass needs (matmul, rmsnorm, rope,
  softmax, swiglu),
- KV cache storage and position updates.

## Rules (non-negotiable)

1. **A backend is only reported as available if it can actually execute the
   supported operations.** Initialization failure means the engine falls
   back to CPU *and says so* — it never claims GPU execution.
2. No fake GPU implementations, no stub kernels that return zeros, no
   silent CPU fallback behind a GPU flag.
3. Optional hardware features must not break ordinary CI: GPU crates are
   feature-gated and their tests are skipped when no device/SDK is present.
4. Numerical parity: a GPU backend must pass the same reference-value test
   suite (with appropriate tolerances) as the CPU kernels.

## Target order

1. **Vulkan** — cross-platform (Linux/Windows/Android), the first target;
   the interface below is designed with mobile Vulkan in mind.
2. **CUDA** — NVIDIA-specific path.
3. **Metal** — Apple platforms.

The backend interface deliberately avoids desktop-only assumptions
(windowing, WSI, display) so an Android Vulkan backend remains possible.

## What exists today

- The `Backend`/`DeviceInfo` abstraction with honest availability
  reporting (`Backend::is_available()` returns false for every GPU backend).
- CPU feature detection and SIMD dispatch — the pattern GPU dispatch will
  follow.
- The engine surfaces `DeviceInfo` in `ModelInfo` and the CLI/server
  reports it, so "what am I running on" is always observable.
