//! Minimal generation example: loads a GGUF model and prints a completion.

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "model.gguf".to_string());
    let prompt = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("Once upon a time");

    let engine = llama_runtime::Engine::load(&model)?;
    let config = llama_runtime::GenerationConfig::default()
        .max_tokens(128)
        .temperature(0.7)
        .seed(42);

    let output = engine.generate(prompt, &config)?;
    println!("{}", output.text);
    eprintln!(
        "\n[{} tokens in {:.2}s ({:.1} tok/s), stopped: {:?}]",
        output.tokens.len(),
        output.stats.decode.duration.as_secs_f64(),
        output.stats.decode_tps(),
        output.stop_reason
    );
    Ok(())
}
