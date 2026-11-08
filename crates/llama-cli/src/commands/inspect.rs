//! `llama inspect` — model metadata report.

use anyhow::{Context, Result};
use llama_core::GgmlType;
use llama_runtime::Engine;

use super::InspectArgs;

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

/// Runs the inspect command.
///
/// # Errors
///
/// Returns an error when the model cannot be opened or parsed.
pub fn run(args: &InspectArgs) -> Result<()> {
    let engine =
        Engine::load(&args.model).with_context(|| format!("cannot load '{}'", args.model))?;
    let info = engine.info();

    println!("GGUF version      : v{}", info.gguf_version);
    println!("architecture      : {}", info.architecture);
    println!(
        "name              : {}",
        info.name.as_deref().unwrap_or("<unnamed>")
    );
    println!("vocab size        : {}", info.n_vocab);
    println!("context length    : {}", info.n_ctx_train);
    println!("layers            : {}", info.n_layers);
    println!("n_embd            : {}", info.n_embd);
    println!("n_ff              : {}", info.n_ff);
    println!(
        "attention heads   : {} ({} kv heads)",
        info.n_heads, info.n_kv_heads
    );
    println!("tensor count      : {}", info.n_tensors);
    println!("parameters        : {}", info.n_parameters);
    let types: Vec<String> = info
        .quant_types
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    println!("quantization      : {}", types.join(", "));
    println!("tensor bytes      : {}", format_bytes(info.tensor_bytes));
    println!("file size         : {}", format_bytes(info.file_bytes));
    println!(
        "chat template     : {}",
        info.chat_template.as_deref().unwrap_or("<none>")
    );
    println!("device            : {}", info.device);
    println!("config            : {}", info.config_summary);

    if args.tensors {
        println!();
        println!("{:<40} {:<8} {:<14} SHAPE", "TENSOR", "TYPE", "ELEMENTS");
        let file = llama_gguf::GgufFile::open(&args.model)?;
        for t in file.tensors() {
            println!(
                "{:<40} {:<8} {:<14} {:?}",
                t.name,
                t.ty,
                t.n_elements(),
                t.dims
            );
        }
        let _ = GgmlType::F32; // keep the import meaningful for future filters
    }

    Ok(())
}
