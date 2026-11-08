//! Model metadata exposed to callers (inspection, `/v1/models`, CLI).

use llama_core::{DeviceInfo, GgmlType};
use llama_gguf::GgufFile;

/// A summary of a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// GGUF version (2 or 3).
    pub gguf_version: u32,
    /// Architecture name (`general.architecture`).
    pub architecture: String,
    /// Model name (`general.name`), if present.
    pub name: Option<String>,
    /// Vocabulary size (from the embedding table).
    pub n_vocab: usize,
    /// Embedding dimension.
    pub n_embd: usize,
    /// Number of layers.
    pub n_layers: usize,
    /// Feed-forward dimension.
    pub n_ff: usize,
    /// Attention head count.
    pub n_heads: usize,
    /// KV head count.
    pub n_kv_heads: usize,
    /// Training context length.
    pub n_ctx_train: u64,
    /// Number of tensors.
    pub n_tensors: usize,
    /// Distinct quantization types in use.
    pub quant_types: Vec<GgmlType>,
    /// Total parameter count.
    pub n_parameters: u64,
    /// Total serialized tensor bytes.
    pub tensor_bytes: u64,
    /// Size of the GGUF file on disk.
    pub file_bytes: u64,
    /// Chat template family (if a template was present and recognized).
    pub chat_template: Option<String>,
    /// Execution device description.
    pub device: DeviceInfo,
    /// Human-readable hyper-parameter summary.
    pub config_summary: String,
}

impl ModelInfo {
    /// Collects the information from an open GGUF file and loaded model.
    #[must_use]
    pub fn collect(
        file: &GgufFile,
        model: &llama_model::LlamaModel,
        chat_template: Option<&str>,
        device: DeviceInfo,
    ) -> Self {
        let meta = file.metadata();
        let cfg = model.config();
        let mut quant_types: Vec<GgmlType> = Vec::new();
        for (_, ty, _) in model.tensor_types() {
            if !quant_types.contains(&ty) {
                quant_types.push(ty);
            }
        }
        quant_types.sort_by_key(|t| t.to_raw());
        Self {
            gguf_version: file.version(),
            architecture: meta
                .get_string("general.architecture")
                .unwrap_or("unknown")
                .to_string(),
            name: meta.get_string_opt("general.name").map(str::to_owned),
            n_vocab: model.n_vocab(),
            n_embd: cfg.n_embd,
            n_layers: cfg.n_layers,
            n_ff: cfg.n_ff,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            n_ctx_train: cfg.n_ctx_train,
            n_tensors: file.tensors().len(),
            quant_types,
            n_parameters: model.n_parameters(),
            tensor_bytes: model.tensor_bytes_total(),
            file_bytes: file.file_size(),
            chat_template: chat_template.map(str::to_owned),
            device,
            config_summary: cfg.describe(),
        }
    }
}
