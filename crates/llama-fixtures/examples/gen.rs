//! Generates a synthetic GGUF fixture: cargo run -p llama-fixtures --example gen -- <path> [quant]
//! (kept as a script example; see scripts/README.md)

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/tmp/fixture.gguf".into());
    let quant = match args.get(2).map(String::as_str) {
        Some("q8_0") => llama_fixtures::QuantScheme::Q8_0,
        Some("q4_0") => llama_fixtures::QuantScheme::Q4_0,
        Some("f16") => llama_fixtures::QuantScheme::F16,
        Some("kquant") => llama_fixtures::QuantScheme::KQuants,
        _ => llama_fixtures::QuantScheme::F32,
    };
    let spec = if quant == llama_fixtures::QuantScheme::KQuants {
        llama_fixtures::FixtureSpec::kquant()
    } else {
        llama_fixtures::FixtureSpec {
            quant,
            ..Default::default()
        }
    };
    match llama_fixtures::write_llama_fixture(std::path::Path::new(&path), &spec) {
        Ok(n) => println!("wrote {path} with {n} tensors"),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
