//! `llama` — the llama.rs command-line interface.

mod commands;

use clap::Parser;
use commands::{Cli, Command};

fn main() {
    llama_core::logging::init_tracing();
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Inspect(args) => commands::inspect::run(&args),
        Command::Run(args) => commands::run::run(&args),
        Command::Chat(args) => commands::chat::run(&args),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
