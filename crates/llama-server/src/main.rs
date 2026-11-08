//! `llama-server` — an OpenAI-compatible HTTP API around the llama.rs engine.

mod api;

use anyhow::{Context, Result};
use clap::Parser;

use crate::api::AppState;

#[derive(Parser)]
#[command(
    name = "llama-server",
    version,
    about = "OpenAI-compatible HTTP server for the llama.rs inference engine",
    long_about = "Serves a GGUF model over an OpenAI-compatible API.\n\n\
                  Listens on localhost by default. Pass --host 0.0.0.0 to expose it on all \
                  interfaces (only on networks you trust)."
)]
struct Args {
    /// Path to the GGUF model.
    #[arg(long, short)]
    model: String,
    /// Bind address. Defaults to 127.0.0.1 for safety.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Listen port.
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Context window size in tokens.
    #[arg(long, default_value_t = 2048)]
    context_size: usize,
    /// Number of CPU compute threads.
    #[arg(long)]
    threads: Option<usize>,
    /// Store the KV cache as f16.
    #[arg(long, default_value_t = false)]
    kv_f16: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    llama_core::logging::init_tracing();

    let args = Args::parse();

    let mut options = llama_runtime::EngineOptions::default()
        .with_context_size(args.context_size)
        .with_kv_cache_f16(args.kv_f16);
    if let Some(threads) = args.threads {
        options = options.with_threads(threads);
    }

    let engine = llama_runtime::Engine::load_with(&args.model, options)
        .with_context(|| format!("cannot load model '{}'", args.model))?;

    let state = std::sync::Arc::new(AppState::new(engine));
    let app = api::router(state.clone());

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("cannot bind {addr}"))?;
    tracing::info!("listening on http://{addr} (model: {})", state.model_id());

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let ctrl_c = async {
                let _ = tokio::signal::ctrl_c().await;
            };
            #[cfg(unix)]
            let terminate = async {
                let _ = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("install SIGTERM handler")
                    .recv()
                    .await;
            };
            #[cfg(not(unix))]
            let terminate = std::future::pending::<()>();
            tokio::select! {
                () = ctrl_c => {},
                () = terminate => {},
            }
            tracing::info!("shutting down");
        })
        .await
        .context("server error")?;

    Ok(())
}
