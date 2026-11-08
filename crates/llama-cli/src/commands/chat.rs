//! `llama chat` — interactive conversation with prefix-reuse caching.

use std::io::{BufRead, Write};

use anyhow::{Context, Result};
use llama_runtime::{Engine, EngineOptions, Flow};
use llama_tokenizer::Message;

use super::{ChatArgs, run::stop_reason_text, to_generation_config};

/// Runs the chat command.
///
/// # Errors
///
/// Returns an error when the model cannot load or a turn fails.
pub fn run(args: &ChatArgs) -> Result<()> {
    let mut options: EngineOptions = args.engine.to_options();
    if let Some(template_name) = &args.chat_template {
        let template = llama_tokenizer::ChatTemplate::from_name(template_name)
            .with_context(|| format!("--chat-template {template_name}"))?;
        options.chat_template_override = Some(template);
    }

    let engine = Engine::load_with(&args.model, options)
        .with_context(|| format!("cannot load '{}'", args.model))?;
    let config = to_generation_config(&args.generation);

    if engine.chat_template().is_none() {
        anyhow::bail!(
            "the model has no recognizable chat template; pass --chat-template \
             (llama2, llama3, chatml, gemma or plain) to choose one"
        );
    }

    println!(
        "llama.rs chat — {} ({} vocab, ctx {}). Type /quit to exit, /reset to clear.",
        engine.info().name.as_deref().unwrap_or("model"),
        engine.info().n_vocab,
        engine.effective_context_size()
    );

    let mut session = engine.create_session();
    let mut history: Vec<Message> = Vec::new();
    if let Some(system) = &args.system {
        history.push(Message::new("system", system));
    }

    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        print!("> ");
        let _ = std::io::stdout().flush();
        line.clear();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break; // EOF
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        match input {
            "/quit" | "/exit" => break,
            "/reset" => {
                session.clear();
                history.clear();
                if let Some(system) = &args.system {
                    history.push(Message::new("system", system));
                }
                println!("(context cleared)");
                continue;
            }
            _ => {}
        }

        history.push(Message::new("user", input));
        session
            .push_chat_turn(&history, true, true)
            .context("preparing the conversation")?;

        print!("  ");
        let _ = std::io::stdout().flush();
        let result = session.generate(&config, |_token, piece| {
            print!("{piece}");
            let _ = std::io::stdout().flush();
            Flow::Continue
        })?;
        println!();
        let _ = stop_reason_text(result.stop_reason);

        // Append the model reply for the next turn.
        let reply = result.text.trim();
        if !reply.is_empty() {
            history.push(Message::new("assistant", reply));
        }

        if args.stats {
            let stats = &result.stats;
            eprintln!(
                "  [{} tokens, {:.1} tok/s, ctx {}/{}]",
                result.tokens.len(),
                stats.decode_tps(),
                stats.context_tokens,
                stats.context_capacity
            );
        }
    }
    Ok(())
}
