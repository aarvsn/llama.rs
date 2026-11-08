//! Model architecture detection.

use llama_core::{LlamaError, Result};
use llama_gguf::GgufMetadata;

/// The model architectures llama.rs knows about.
///
/// Only [`Architecture::Llama`] is implemented; every other detected
/// architecture is reported as unsupported with its exact name so users get
/// an actionable message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Architecture {
    /// The Llama family (Llama 1/2/3, Mistral v0.1, `TinyLlama`, ...).
    Llama,
}

impl Architecture {
    /// Reads `general.architecture` from GGUF metadata.
    pub fn detect(meta: &GgufMetadata) -> Result<Self> {
        match meta.get_string("general.architecture") {
            Ok(name) => Self::from_name(name),
            Err(LlamaError::MissingMetadata(key)) => Err(LlamaError::MissingMetadata(key)),
            Err(e) => Err(e),
        }
    }

    /// Maps an architecture name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "llama" => Ok(Self::Llama),
            other => Err(LlamaError::UnsupportedArchitecture(format!(
                "'{other}' is not implemented yet; llama.rs currently supports the 'llama' \
                 architecture family (Llama 1/2/3, Mistral v0.1, TinyLlama, stories models, ...)"
            ))),
        }
    }

    /// Canonical architecture name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_gguf::metadata::{GgufMetadata, MetadataValue};

    #[test]
    fn detects_llama() {
        let mut meta = GgufMetadata::new();
        meta.insert(
            "general.architecture".into(),
            MetadataValue::String("llama".into()),
        );
        assert_eq!(Architecture::detect(&meta).unwrap(), Architecture::Llama);
    }

    #[test]
    fn rejects_unknown_and_missing() {
        let mut meta = GgufMetadata::new();
        meta.insert(
            "general.architecture".into(),
            MetadataValue::String("qwen2".into()),
        );
        let err = Architecture::detect(&meta).unwrap_err();
        assert!(err.to_string().contains("qwen2"));

        let empty = GgufMetadata::new();
        assert!(matches!(
            Architecture::detect(&empty),
            Err(LlamaError::MissingMetadata(_))
        ));
    }
}
