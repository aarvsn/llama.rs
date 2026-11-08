//! `llama run` — one-shot generation.

use std::io::Write as _;
use std::io::{IsTerminal, Read};

use anyhow::{Context, Result};
use llama_runtime::{Engine, EngineOptions, Flow, StopReason};

use super::{RunArgs, to_generation_config};

/// Runs the run command.
///
/// # Errors
///
/// Returns an error for invalid arguments or inference failures.
pub fn run(args: &RunArgs) -> Result<()> {
    if args.generation.temperature < 0.0 {
        anyhow::bail!("--temperature must be >= 0");
    }
    if !(0.0..=1.0).contains(&args.generation.top_p) {
        anyhow::bail!("--top-p must be in (0, 1]");
    }

    let prompt = if let Some(p) = &args.prompt {
        p.clone()
    } else {
        if std::io::stdin().is_terminal() {
            anyhow::bail!("no --prompt given and stdin is a terminal");
        }
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("reading prompt from stdin")?;
        buf
    };

    let options: EngineOptions = args.engine.to_options();
    let engine = Engine::load_with(&args.model, options)
        .with_context(|| format!("cannot load '{}'", args.model))?;
    let config = to_generation_config(&args.generation);

    let mut session = engine.create_session();
    session
        .set_prompt(&prompt, config.parse_special)
        .context("processing the prompt")?;

    let mut printed = String::new();
    let result = session.generate(&config, |_token, piece| {
        if args.stream {
            print!("{piece}");
            let _ = std::io::stdout().flush();
        }
        printed.push_str(piece);
        Flow::Continue
    })?;

    if args.stream {
        println!();
    } else {
        println!("{}", result.text);
    }

    if args.stats {
        let stats = &result.stats;
        eprintln!();
        eprintln!("prompt tokens    : {}", stats.prefill.count);
        eprintln!(
            "prompt eval      : {:.3}s ({:.1} tokens/s)",
            stats.prefill.duration.as_secs_f64(),
            stats.prefill_tps()
        );
        eprintln!("generated tokens : {}", result.tokens.len());
        eprintln!(
            "generation       : {:.3}s ({:.1} tokens/s)",
            stats.decode.duration.as_secs_f64(),
            stats.decode_tps()
        );
        eprintln!(
            "stop reason      : {}",
            stop_reason_text(result.stop_reason)
        );
        eprintln!(
            "context          : {}/{} tokens",
            stats.context_tokens, stats.context_capacity
        );
        eprintln!(
            "kv cache         : {:.1} MiB",
            stats.kv_cache_bytes as f64 / 1024.0 / 1024.0
        );
        eprintln!(
            "model            : {:.1} MiB",
            engine.stats().model_bytes as f64 / 1024.0 / 1024.0
        );
    }

    Ok(())
}

/// Human-readable stop reason.
#[must_use]
pub fn stop_reason_text(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Eos => "end of sequence",
        StopReason::StopSequence => "stop sequence",
        StopReason::MaxTokens => "token limit reached",
        StopReason::ContextFull => "context window full",
        StopReason::Callback => "stopped by caller",
    }
}
