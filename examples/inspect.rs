//! Model inspection example: prints GGUF metadata and tensor inventory.

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "model.gguf".to_string());

    let engine = llama_runtime::Engine::load(&model)?;
    let info = engine.info();
    println!(
        "GGUF v{}  architecture: {}",
        info.gguf_version, info.architecture
    );
    println!(
        "name:         {}",
        info.name.as_deref().unwrap_or("<unnamed>")
    );
    println!("vocab:        {}", info.n_vocab);
    println!("context:      {}", info.n_ctx_train);
    println!("layers:       {}", info.n_layers);
    println!("n_embd:       {}", info.n_embd);
    println!("heads:        {} ({} kv)", info.n_heads, info.n_kv_heads);
    println!("tensors:      {}", info.n_tensors);
    println!("parameters:   {}", info.n_parameters);
    println!("quant types:  {:?}", info.quant_types);
    println!(
        "model size:   {:.2} MiB",
        info.tensor_bytes as f64 / 1024.0 / 1024.0
    );
    println!("device:       {}", info.device);
    Ok(())
}
