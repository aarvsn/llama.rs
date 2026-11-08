//! The [`Engine`]: loads a GGUF model once, hands out sessions.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use llama_core::{DeviceInfo, EngineOptions as CoreOptions, LlamaError, Result};
use llama_gguf::GgufFile;
use llama_model::{Architecture, LlamaConfig, LlamaModel};
use llama_tokenizer::Tokenizer;
use llama_tokenizer::template::ChatTemplate;

use crate::info::ModelInfo;
use crate::session::Session;
use crate::stats::EngineStats;

/// Runtime options for the engine.
///
/// Re-exports [`llama_core::EngineOptions`] plus engine-specific knobs.
#[derive(Debug, Clone, Default)]
pub struct EngineOptions {
    /// Core options (threads, context size, KV precision).
    pub core: CoreOptions,
    /// Override the chat template family detected from the GGUF file.
    pub chat_template_override: Option<ChatTemplate>,
}

impl EngineOptions {
    /// Builder: sets the number of compute threads.
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.core = self.core.with_threads(threads);
        self
    }

    /// Builder: sets the context window size.
    #[must_use]
    pub fn with_context_size(mut self, ctx: usize) -> Self {
        self.core = self.core.with_context_size(ctx);
        self
    }

    /// Builder: enables the f16 KV cache.
    #[must_use]
    pub fn with_kv_cache_f16(mut self, enabled: bool) -> Self {
        self.core = self.core.with_kv_cache_f16(enabled);
        self
    }
}

/// A loaded inference engine.
///
/// The engine owns the memory-mapped model, the tokenizer and configuration.
/// It is cheap to clone ([`Arc`] internally) and shared immutably across
/// sessions.
#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    model: LlamaModel,
    tokenizer: Tokenizer,
    chat_template: Option<ChatTemplate>,
    options: CoreOptions,
    info: ModelInfo,
    stats: EngineStats,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("architecture", &self.inner.info.architecture)
            .field("name", &self.inner.info.name)
            .field("n_tensors", &self.inner.info.n_tensors)
            .finish_non_exhaustive()
    }
}

impl Engine {
    /// Loads a GGUF model with default options.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with(path, EngineOptions::default())
    }

    /// Loads a GGUF model with explicit options.
    pub fn load_with(path: impl AsRef<Path>, options: EngineOptions) -> Result<Self> {
        let start = Instant::now();

        // Configure the global rayon pool once (the compute kernels use it).
        let threads = options.core.threads.get();
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("llama-compute-{i}"))
            .build_global();

        let file = Arc::new(GgufFile::open(path)?);

        // Architecture detection and configuration.
        let meta = file.metadata();
        let arch = Architecture::detect(meta)?;
        let config = match arch {
            Architecture::Llama => LlamaConfig::from_gguf(meta)?,
            _ => {
                return Err(LlamaError::UnsupportedArchitecture(format!(
                    "{} is not implemented",
                    arch.name()
                )));
            }
        };

        // Reject GGUF splits (multi-part models) with a clear message.
        if meta.get("split.count").is_some() {
            return Err(LlamaError::unsupported_gguf(
                "this file is part of a split GGUF; llama.rs does not support split models yet — \
                 merge the shards first",
            ));
        }

        let model = LlamaModel::load(Arc::clone(&file), config)?;
        let tokenizer = Tokenizer::from_gguf(meta)?;

        let chat_template = match options.chat_template_override {
            Some(t) => Some(t),
            None => match meta.get_string_opt("tokenizer.chat_template") {
                Some(t) if !t.is_empty() => Some(ChatTemplate::detect(t)?),
                _ => None,
            },
        };

        let device = DeviceInfo::cpu(threads);
        let info = ModelInfo::collect(&file, &model, chat_template.map(ChatTemplate::name), device);
        let stats = EngineStats {
            load_time: start.elapsed(),
            model_bytes: model.tensor_bytes_total(),
            file_bytes: file.file_size(),
        };

        if options.core.log_load_info {
            tracing::info!(
                "loaded '{}' ({}, {} tensors, {} params, {:.1} MiB) in {:.2}s on {}",
                info.name.as_deref().unwrap_or("<unnamed>"),
                info.architecture,
                info.n_tensors,
                info.n_parameters,
                stats.model_bytes as f64 / (1024.0 * 1024.0),
                stats.load_time.as_secs_f64(),
                info.device,
            );
        }

        Ok(Self {
            inner: Arc::new(EngineInner {
                model,
                tokenizer,
                chat_template,
                options: options.core,
                info,
                stats,
            }),
        })
    }

    /// The model summary.
    #[must_use]
    pub fn info(&self) -> &ModelInfo {
        &self.inner.info
    }

    /// The tokenizer.
    #[must_use]
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.inner.tokenizer
    }

    /// The detected chat template (if any).
    #[must_use]
    pub fn chat_template(&self) -> Option<ChatTemplate> {
        self.inner.chat_template
    }

    /// Engine options in use.
    #[must_use]
    pub fn options(&self) -> &CoreOptions {
        &self.inner.options
    }

    /// Engine load statistics.
    #[must_use]
    pub fn stats(&self) -> &EngineStats {
        &self.inner.stats
    }

    /// The underlying model (for advanced callers).
    #[must_use]
    pub fn model(&self) -> &LlamaModel {
        &self.inner.model
    }

    /// The effective context size: the minimum of the configured context and
    /// the model's training context.
    #[must_use]
    pub fn effective_context_size(&self) -> usize {
        let model_ctx = self.inner.model.config().n_ctx_train as usize;
        self.inner.options.context_size.min(model_ctx).max(16)
    }

    /// Creates a new generation session with its own KV cache.
    #[must_use]
    pub fn create_session(&self) -> Session {
        Session::new(self.clone())
    }

    /// One-shot generation: prompt in, text out.
    ///
    /// This allocates a fresh session (no cross-call cache reuse). For
    /// interactive use, prefer [`Engine::create_session`].
    pub fn generate(
        &self,
        prompt: &str,
        config: &crate::session::GenerationConfig,
    ) -> Result<crate::session::GenerateOutput> {
        let mut session = self.create_session();
        session.set_prompt(prompt, config.parse_special)?;
        session.generate(config, |_token, _piece| crate::session::Flow::Continue)
    }
}
