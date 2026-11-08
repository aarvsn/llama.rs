# Android

**Status: architecture prepared, execution not yet validated on devices.**

llama.rs is written to cross-compile for Android arm64, but no on-device
testing has been performed yet — so Android is *ready by construction*, not
*verified*. This page documents what makes the code Android-friendly and
how to try it.

## What the codebase already does

- **No desktop-only dependencies** in the core path: model loading uses
  `memmap2` (plain `mmap(2)`), threading uses `rayon` (pthreads), and there
  is no use of desktop windowing, filesystem APIs beyond std, or process
  APIs that Android lacks.
- **aarch64 support**: all kernels have portable scalar implementations;
  NEON kernels are planned (see docs/cpu-backend.md). Everything runs
  correctly on aarch64 today, just without NEON acceleration.
- **Memory-mapped weights**: models can be read straight from APK assets or
  app storage without loading copies into RAM.
- **Runtime resource detection**: `EngineOptions::threads` lets callers
  match the device's big/little core budget.
- **No global state** beyond the rayon pool: the engine is embeddable in a
  library context.

## Cross-compiling

With the Android NDK installed (API level 24+):

```sh
rustup target add aarch64-linux-android

# Point cargo at the NDK linker (or use cargo-ndk):
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/<version>
export PATH=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=aarch64-linux-android24-clang

cargo build --release -p llama-cli --target aarch64-linux-android
# (also: -p llama-server for an on-device HTTP endpoint)
```

For a library to bind from Kotlin/Java, build the runtime crate as a
`cdylib` and expose a small C ABI — the planned `llama-jni` crate will
provide ready-made JNI bindings.

## Planned

1. **NEON kernels** for the dot products and dequantization (aarch64
   baseline ISA includes NEON, so runtime detection is trivial).
2. **`llama-jni`**: a feature-gated crate with Kotlin-friendly bindings
   (load model, generate, stream callbacks, stats).
3. **On-device CI** (GitHub Actions arm64 runners or a physical-device
   lane) before Android is advertised as supported.

## Honest limitation summary

Until NEON kernels and on-device validation land, Android remains
"compiles and should run correctly (scalar)" rather than "supported".
The desktop implementation is not compromised for it.
