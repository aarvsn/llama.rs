//! Criterion benchmarks for the llama.rs engine.
//!
//! Run with `cargo bench`. Synthetic fixtures are generated on the fly; the
//! real-model benchmarks (prefill/decode throughput) are exercised through
//! the CLI against a real GGUF file, not here — see docs/benchmarking.md.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use llama_core::{F16, GgmlType};
use llama_kernels::quant::{dequantize_row, quantize_row_q8_0};
use llama_kernels::vecdot;

const N: usize = 4096; // row length (must be a multiple of 256 for K-quants)

fn lcg_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / 8_388_608.0) - 0.5
        })
        .collect()
}

fn quantize_q4_0(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 18);
    for chunk in x.chunks(32) {
        let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&F16::from_f32(d).to_bits().to_le_bytes());
        let mut qs = [0u8; 16];
        for (j, &v) in chunk.iter().enumerate() {
            let q = (((v * id) + 8.5) as i32).clamp(0, 15) as u8;
            if j < 16 {
                qs[j] |= q & 0x0F;
            } else {
                qs[j - 16] |= q << 4;
            }
        }
        out.extend_from_slice(&qs);
    }
    out
}

fn quantize_q6_k_random(x: &[f32]) -> Vec<u8> {
    // Deterministic pseudo-random bytes with a benign scale.
    let mut s = 0xDEAD_BEEFu64;
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 56) as u8
    };
    let mut out = Vec::with_capacity(x.len() / 256 * 210);
    for _ in 0..x.len() / 256 {
        out.extend_from_slice(&F16::from_f32(0.01).to_bits().to_le_bytes());
        for _ in 0..208 {
            out.push(next());
        }
    }
    out
}

fn bench_vec_dots(c: &mut Criterion) {
    let x = lcg_vec(42, N);
    let xq = quantize_row_q8_0(&x);

    let f32_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let f16_bytes: Vec<u8> = x
        .iter()
        .flat_map(|v| F16::from_f32(*v).to_bits().to_le_bytes())
        .collect();
    let q4_0 = quantize_q4_0(&x);
    let q6_k = quantize_q6_k_random(&x);
    let q8_w = quantize_row_q8_0(&lcg_vec(7, N));
    let q8_bytes: Vec<u8> = q8_w
        .blocks
        .iter()
        .flat_map(|b| {
            let mut v = b.d.to_le_bytes().to_vec();
            v.extend(b.qs.iter().map(|&q| q as u8));
            v
        })
        .collect();

    let mut g = c.benchmark_group("vec_dot");
    g.throughput(Throughput::Elements(N as u64));

    // Through the runtime dispatcher (SIMD when available) — the path the
    // engine actually takes.
    let act_f32 = llama_kernels::dispatch::Activation::F32(x.clone());
    g.bench_function("f32_dispatched", |b| {
        b.iter(|| {
            llama_kernels::dispatch::vec_dot_row(
                GgmlType::F32,
                black_box(&f32_bytes),
                black_box(&act_f32),
            )
            .unwrap()
        })
    });
    g.bench_function("f32_scalar_reference", |b| {
        b.iter(|| vecdot::vec_dot_f32(black_box(&f32_bytes), black_box(&x)))
    });
    g.bench_function("f16_dispatched", |b| {
        b.iter(|| {
            llama_kernels::dispatch::vec_dot_row(
                GgmlType::F16,
                black_box(&f16_bytes),
                black_box(&act_f32),
            )
            .unwrap()
        })
    });
    g.bench_function("q8_0_q8_0", |b| {
        b.iter(|| vecdot::vec_dot_q8_0(black_box(&q8_bytes), black_box(&xq)))
    });
    g.bench_function("q4_0_q8_0", |b| {
        b.iter(|| vecdot::vec_dot_q4_0(black_box(&q4_0), black_box(&xq)))
    });
    g.bench_function("q6_k_q8_0", |b| {
        b.iter(|| vecdot::vec_dot_q6_k(black_box(&q6_k), black_box(&xq)))
    });
    g.finish();
}

fn bench_dequantize(c: &mut Criterion) {
    let x = lcg_vec(43, N);
    let q4_0 = quantize_q4_0(&x);
    let mut out = vec![0.0f32; N];

    let mut g = c.benchmark_group("dequantize_row");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("q4_0", |b| {
        b.iter(|| dequantize_row(GgmlType::Q4_0, black_box(&q4_0), black_box(&mut out)))
    });
    g.finish();
}

fn bench_quantize_q8(c: &mut Criterion) {
    let x = lcg_vec(44, N);
    let mut g = c.benchmark_group("quantize");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("q8_0", |b| b.iter(|| quantize_row_q8_0(black_box(&x))));
    g.finish();
}

fn bench_ops(c: &mut Criterion) {
    let n = 2048usize;
    let x = lcg_vec(45, n);
    let w = vec![1.0f32; n];
    let mut y = x.clone();
    let mut head = lcg_vec(46, 64);

    let mut g = c.benchmark_group("ops");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("rmsnorm", |b| {
        b.iter(|| {
            llama_kernels::ops::rms_norm(black_box(&x), black_box(&w), black_box(&mut y), 1e-5)
        })
    });
    g.bench_function("rope_head_64", |b| {
        b.iter(|| {
            llama_kernels::ops::rope_head(
                black_box(&mut head),
                64,
                1234.0,
                10_000.0,
                llama_kernels::ops::RotaryStyle::Interleaved,
            )
        })
    });
    g.bench_function("softmax_2048", |b| {
        b.iter(|| llama_kernels::ops::softmax_inplace(black_box(&mut y)))
    });
    g.finish();
}

fn bench_tokenizer(c: &mut Criterion) {
    // Build a fixture model (SPM vocab) and measure encode throughput.
    let dir = std::env::temp_dir().join("llama-rs-bench");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bench-fixture.gguf");
    llama_fixtures::write_llama_fixture(
        &path,
        &llama_fixtures::FixtureSpec {
            quant: llama_fixtures::QuantScheme::Q8_0,
            ..llama_fixtures::FixtureSpec::default()
        },
    )
    .unwrap();
    let engine = llama_runtime::Engine::load(&path).unwrap();
    let text = "Once upon a time there was a little girl who loved to play \
                outside in the sunshine and pick flowers by the river.";

    let mut g = c.benchmark_group("tokenizer");
    g.throughput(Throughput::Bytes(text.len() as u64));
    g.bench_function("spm_encode", |b| {
        b.iter(|| {
            engine
                .tokenizer()
                .encode(black_box(text), true, true)
                .unwrap()
        })
    });
    let tokens = engine.tokenizer().encode(text, true, true).unwrap();
    g.bench_function("spm_decode", |b| {
        b.iter(|| engine.tokenizer().decode(black_box(&tokens), false))
    });
    g.finish();
}

fn bench_engine(c: &mut Criterion) {
    // Full-engine generation on the synthetic fixture.
    let dir = std::env::temp_dir().join("llama-rs-bench");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bench-fixture.gguf");
    if !path.exists() {
        llama_fixtures::write_llama_fixture(
            &path,
            &llama_fixtures::FixtureSpec {
                quant: llama_fixtures::QuantScheme::Q8_0,
                ..llama_fixtures::FixtureSpec::default()
            },
        )
        .unwrap();
    }
    let engine = llama_runtime::Engine::load(&path).unwrap();

    let mut g = c.benchmark_group("engine");
    g.bench_function("load_fixture", |b| {
        b.iter(|| llama_runtime::Engine::load(black_box(&path)).unwrap())
    });
    g.bench_function("generate_32_tokens", |b| {
        b.iter(|| {
            let config = llama_runtime::GenerationConfig::default()
                .max_tokens(32)
                .greedy(true);
            engine
                .generate(black_box("Once upon a time"), black_box(&config))
                .unwrap()
        })
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_vec_dots,
    bench_dequantize,
    bench_quantize_q8,
    bench_ops,
    bench_tokenizer,
    bench_engine
);
criterion_main!(benches);
