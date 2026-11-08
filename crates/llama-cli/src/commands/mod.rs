//! Command definitions and shared argument types.

use clap::{Parser, Subcommand};
use llama_runtime::EngineOptions;

pub mod chat;
pub mod inspect;
pub mod run;

#[derive(Parser)]
#[command(
    name = "llama",
    version,
    about = "llama.rs — a native Rust LLM inference engine for GGUF models",
    long_about = "Runs quantized Llama-family language models locally.\n\n\
                  Supported tensor formats: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K.\n\
                  Supported architectures: llama (Llama 1/2/3, Mistral v0.1, TinyLlama, ...).\n\
                  CPU execution with runtime SIMD (AVX2/FMA/F16C, NEON on ARM64)."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Print model metadata without running inference.
    Inspect(InspectArgs),
    /// Generate text from a prompt.
    Run(RunArgs),
    /// Interactive chat with the model.
    Chat(ChatArgs),
}

#[derive(clap::Args)]
pub struct InspectArgs {
    /// Path to the GGUF model.
    pub model: String,
    /// Print every tensor with its type and shape.
    #[arg(long)]
    pub tensors: bool,
}

/// Sampling and generation options shared by `run` and `chat`.
#[derive(clap::Args)]
pub struct GenerationArgs {
    /// Maximum number of tokens to generate.
    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,
    /// Softmax temperature (0 = greedy).
    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,
    /// Top-k sampling (0 disables).
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,
    /// Nucleus (top-p) sampling.
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    /// Min-p sampling.
    #[arg(long, default_value_t = 0.05)]
    pub min_p: f32,
    /// Repetition penalty (1.0 disables).
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,
    /// Frequency penalty.
    #[arg(long, default_value_t = 0.0)]
    pub frequency_penalty: f32,
    /// Presence penalty.
    #[arg(long, default_value_t = 0.0)]
    pub presence_penalty: f32,
    /// Random seed for sampling (deterministic when fixed).
    #[arg(long)]
    pub seed: Option<u64>,
    /// Greedy decoding (argmax); overrides sampling options.
    #[arg(long, default_value_t = false)]
    pub greedy: bool,
    /// Stop sequences (comma separated).
    #[arg(long, value_delimiter = ',')]
    pub stop: Vec<String>,
}

/// Engine options shared by `run` and `chat`.
#[derive(clap::Args)]
pub struct EngineArgs {
    /// Context window size in tokens.
    #[arg(long, default_value_t = 2048)]
    pub context_size: usize,
    /// Number of CPU compute threads.
    #[arg(long)]
    pub threads: Option<usize>,
    /// Store the KV cache as f16 (halves cache memory).
    #[arg(long, default_value_t = false)]
    pub kv_f16: bool,
}

impl EngineArgs {
    /// Converts to engine options.
    #[must_use]
    pub fn to_options(&self) -> EngineOptions {
        let mut opts = EngineOptions::default()
            .with_context_size(self.context_size)
            .with_kv_cache_f16(self.kv_f16);
        if let Some(threads) = self.threads {
            opts = opts.with_threads(threads);
        }
        opts
    }
}

#[derive(clap::Args)]
pub struct RunArgs {
    /// Path to the GGUF model.
    pub model: String,
    /// The prompt. Omit to read from stdin.
    #[arg(long)]
    pub prompt: Option<String>,
    #[command(flatten)]
    pub generation: GenerationArgs,
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Print performance statistics after generation.
    #[arg(long)]
    pub stats: bool,
    /// Stream tokens to stdout as they are generated (default; disable with --no-stream).
    #[arg(long = "no-stream", default_value_t = true, action = clap::ArgAction::SetFalse)]
    pub stream: bool,
}

#[derive(clap::Args)]
pub struct ChatArgs {
    /// Path to the GGUF model.
    pub model: String,
    #[command(flatten)]
    pub generation: GenerationArgs,
    #[command(flatten)]
    pub engine: EngineArgs,
    /// Override the chat template (llama2, llama3, chatml, gemma, plain).
    #[arg(long)]
    pub chat_template: Option<String>,
    /// Print performance statistics after each reply.
    #[arg(long)]
    pub stats: bool,
    /// System prompt for the conversation.
    #[arg(long)]
    pub system: Option<String>,
}

/// Converts CLI generation arguments into a runtime configuration.
#[must_use]
pub fn to_generation_config(args: &GenerationArgs) -> llama_runtime::GenerationConfig {
    llama_runtime::GenerationConfig {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        repeat_penalty: args.repeat_penalty,
        frequency_penalty: args.frequency_penalty,
        presence_penalty: args.presence_penalty,
        seed: args.seed,
        stop: args.stop.clone(),
        greedy: args.greedy,
        parse_special: true,
        emit_special: false,
    }
}
