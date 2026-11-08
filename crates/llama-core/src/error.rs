//! Error types shared across the llama.rs workspace.

use std::fmt;

/// Convenient result alias used throughout llama.rs.
pub type Result<T, E = LlamaError> = std::result::Result<T, E>;

/// The single top-level error type for the llama.rs engine.
///
/// Every subsystem reports failures through this enum so that callers get a
/// uniform error surface. Messages are specific and actionable: unsupported
/// tensor formats, architectures, tokenizer kinds, and malformed GGUF files
/// each have their own variant.
#[derive(Debug, thiserror::Error)]
pub enum LlamaError {
    /// A file could not be opened or read.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A malformed GGUF file was detected. Details in the message.
    #[error("invalid GGUF file: {0}")]
    InvalidGguf(String),

    /// The GGUF file is valid but uses a feature llama.rs does not support yet.
    #[error("unsupported GGUF feature: {0}")]
    UnsupportedGguf(String),

    /// A required metadata key is missing.
    #[error("missing GGUF metadata: {0}")]
    MissingMetadata(String),

    /// A metadata key exists but has the wrong type or an invalid value.
    #[error("invalid GGUF metadata: {0}")]
    InvalidMetadata(String),

    /// A tensor referenced by name does not exist in the model.
    #[error("missing tensor: {0}")]
    MissingTensor(String),

    /// A tensor exists but has an unexpected shape or type.
    #[error("invalid tensor: {0}")]
    InvalidTensor(String),

    /// The model architecture is recognized but not implemented.
    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    /// The tokenizer kind is recognized but not implemented.
    #[error("unsupported tokenizer: {0}")]
    UnsupportedTokenizer(String),

    /// The requested operation does not fit the runtime configuration
    /// (for example, exceeding the context window).
    #[error("runtime error: {0}")]
    Runtime(String),

    /// A tokenizer operation failed (unknown piece, bad UTF-8, ...).
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Invalid user-supplied arguments.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// A numerical kernel received incompatible operands.
    #[error("kernel error: {0}")]
    Kernel(String),

    /// Errors bubbling up from the HTTP server.
    #[error("server error: {0}")]
    Server(String),
}

impl LlamaError {
    /// Builds an [`LlamaError::InvalidGguf`] quickly.
    pub fn invalid_gguf(msg: impl Into<String>) -> Self {
        Self::InvalidGguf(msg.into())
    }

    /// Builds an [`LlamaError::UnsupportedGguf`] quickly.
    pub fn unsupported_gguf(msg: impl Into<String>) -> Self {
        Self::UnsupportedGguf(msg.into())
    }

    /// Builds an [`LlamaError::Runtime`] quickly.
    pub fn runtime(msg: impl Into<String>) -> Self {
        Self::Runtime(msg.into())
    }

    /// Builds an [`LlamaError::Kernel`] quickly.
    pub fn kernel(msg: impl Into<String>) -> Self {
        Self::Kernel(msg.into())
    }
}

/// A simple, allocation-free location descriptor used in diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLocation {
    /// File name.
    pub file: &'static str,
    /// Line number.
    pub line: u32,
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)
    }
}

/// Captures the current source location (for diagnostics).
#[macro_export]
macro_rules! here {
    () => {
        $crate::error::SourceLocation {
            file: std::file!(),
            line: std::line!(),
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_are_distinct() {
        assert!(
            LlamaError::invalid_gguf("bad magic")
                .to_string()
                .contains("invalid GGUF")
        );
        assert!(
            LlamaError::UnsupportedArchitecture("mamba".into())
                .to_string()
                .contains("mamba")
        );
    }
}
