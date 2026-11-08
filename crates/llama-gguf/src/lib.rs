//! # llama-gguf
//!
//! A self-contained GGUF v2/v3 reader and writer.
//!
//! * [`GgufReader`] parses headers, metadata and tensor directories with full
//!   bounds checking and validation.
//! * [`GgufFile`] owns a memory map of the model and hands out zero-copy
//!   tensor byte slices.
//! * [`GgufWriter`] serializes GGUF files; it powers the test-fixture
//!   generators and can be used by conversion tooling.
//!
//! The reader is deliberately strict: malformed files produce
//! [`LlamaError::InvalidGguf`] with a descriptive message instead of panics.

pub mod metadata;
pub mod reader;
pub mod tensor_info;
pub mod writer;

pub use metadata::{GgufMetadata, MetadataValue};
pub use reader::{GgufFile, GgufReader};
pub use tensor_info::TensorInfo;
pub use writer::{GgufWriter, TensorPayload};

/// Magic bytes at the start of every GGUF file (`"GGUF"` little-endian).
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Supported GGUF versions.
pub const MIN_SUPPORTED_VERSION: u32 = 2;
/// Highest GGUF version understood by this reader.
pub const MAX_SUPPORTED_VERSION: u32 = 3;
