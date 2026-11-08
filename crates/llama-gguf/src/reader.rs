//! GGUF file reader: header, metadata, tensor directory and memory-mapped
//! tensor data with full validation.

use std::fs::File;
use std::path::Path;

use llama_core::{GgmlType, LlamaError, Result};
use memmap2::Mmap;

use crate::metadata::{GgufMetadata, MetadataValue};
use crate::tensor_info::TensorInfo;
use crate::{GGUF_MAGIC, MAX_SUPPORTED_VERSION, MIN_SUPPORTED_VERSION};

/// Default alignment of the GGUF data section when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// A bounds-checked cursor over immutable bytes.
struct ByteCursor<'a> {
    data: &'a [u8],
    pos: u64,
}

impl<'a> ByteCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn position(&self) -> u64 {
        self.pos
    }

    fn remaining(&self) -> u64 {
        self.data.len() as u64 - self.pos
    }

    fn err(&self, msg: impl Into<String>) -> LlamaError {
        LlamaError::invalid_gguf(format!("at byte offset {}: {}", self.pos, msg.into()))
    }

    fn take(&mut self, n: u64) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(self.err(format!(
                "unexpected end of file: needed {} bytes, only {} remain",
                n,
                self.remaining()
            )));
        }
        let start = self.pos as usize;
        let out = &self.data[start..start + n as usize];
        self.pos += n;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(self.take(1)?[0] as i8)
    }

    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn i16(&mut self) -> Result<i16> {
        let b = self.take(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self) -> Result<i32> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(b);
        Ok(u64::from_le_bytes(arr))
    }

    fn i64(&mut self) -> Result<i64> {
        let b = self.take(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(b);
        Ok(i64::from_le_bytes(arr))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn bool(&mut self) -> Result<bool> {
        let v = self.u8()?;
        if v > 1 {
            return Err(self.err(format!("invalid bool value {v} (expected 0 or 1)")));
        }
        Ok(v == 1)
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        if len > (64 << 20) {
            return Err(self.err(format!("string length {len} exceeds 64 MiB safety limit")));
        }
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| self.err("metadata string is not valid UTF-8"))
    }

    fn value(&mut self, type_tag: u32) -> Result<MetadataValue> {
        Ok(match type_tag {
            0 => MetadataValue::U8(self.u8()?),
            1 => MetadataValue::I8(self.i8()?),
            2 => MetadataValue::U16(self.u16()?),
            3 => MetadataValue::I16(self.i16()?),
            4 => MetadataValue::U32(self.u32()?),
            5 => MetadataValue::I32(self.i32()?),
            6 => MetadataValue::F32(self.f32()?),
            7 => MetadataValue::Bool(self.bool()?),
            8 => MetadataValue::String(self.string()?),
            9 => {
                // Array: element type tag + count + elements.
                let elem_tag = self.u32()?;
                let count = self.u64()?;
                // Sanity limit: a valid GGUF never has more metadata elements
                // than file bytes.
                if count > self.data.len() as u64 {
                    return Err(self.err(format!(
                        "metadata array claims {count} elements, larger than the file"
                    )));
                }
                let mut items = Vec::with_capacity(count.min(1 << 20) as usize);
                for _ in 0..count {
                    items.push(self.value(elem_tag)?);
                }
                MetadataValue::Array(items)
            }
            10 => MetadataValue::U64(self.u64()?),
            11 => MetadataValue::I64(self.i64()?),
            12 => MetadataValue::F64(self.f64()?),
            other => {
                return Err(self.err(format!(
                    "unknown metadata value type {other} (GGUF v2/v3 defines types 0..=12)"
                )));
            }
        })
    }
}

/// Parsed GGUF header and directory.
#[derive(Debug)]
pub struct GgufReader {
    version: u32,
    metadata: GgufMetadata,
    tensors: Vec<TensorInfo>,
    alignment: u64,
    /// Byte offset of the tensor data section within the file.
    data_start: u64,
    /// Total size in bytes of the mapped file (from the caller).
    file_len: u64,
}

impl GgufReader {
    /// Parses the header and directories from an in-memory buffer.
    ///
    /// The buffer must start at offset 0 of the GGUF file. `file_len` is used
    /// for tensor bounds validation.
    pub fn parse(data: &[u8], file_len: u64) -> Result<Self> {
        let mut cur = ByteCursor::new(data);

        let magic = cur.u32()?;
        if magic != GGUF_MAGIC {
            let shown = u32::to_le_bytes(magic);
            return Err(LlamaError::invalid_gguf(format!(
                "bad magic {shown:?}; this does not look like a GGUF file"
            )));
        }

        let version = cur.u32()?;
        if !(MIN_SUPPORTED_VERSION..=MAX_SUPPORTED_VERSION).contains(&version) {
            return Err(LlamaError::unsupported_gguf(format!(
                "GGUF version {version} is not supported (supported: v{MIN_SUPPORTED_VERSION} and v{MAX_SUPPORTED_VERSION})"
            )));
        }

        let tensor_count = cur.u64()?;
        let metadata_kv_count = cur.u64()?;

        if tensor_count > 100_000 {
            return Err(cur.err(format!("implausible tensor count {tensor_count}")));
        }

        // Metadata.
        let mut metadata = GgufMetadata::new();
        for _ in 0..metadata_kv_count {
            let key = cur.string()?;
            let type_tag = cur.u32()?;
            let value = cur.value(type_tag)?;
            metadata.insert(key, value);
        }

        // Alignment.
        let alignment = metadata
            .get("general.alignment")
            .and_then(MetadataValue::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment % 8 != 0 || alignment == 0 || alignment > (1 << 20) {
            return Err(LlamaError::InvalidMetadata(format!(
                "general.alignment = {alignment} is out of the supported range"
            )));
        }

        // Tensor directory.
        let mut tensors = Vec::with_capacity(tensor_count.min(65_536) as usize);
        for _ in 0..tensor_count {
            let name = cur.string()?;
            let n_dims = cur.u32()?;
            if n_dims == 0 || n_dims > 4 {
                return Err(cur.err(format!(
                    "tensor '{name}' has {n_dims} dimensions (supported: 1..=4)"
                )));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let d = cur.u64()?;
                if d == 0 || d > (1 << 34) {
                    return Err(cur.err(format!("tensor '{name}' has an invalid dimension {d}")));
                }
                dims.push(d);
            }
            let ty_raw = cur.u32()?;
            let ty = GgmlType::from_raw(ty_raw);
            let offset = cur.u64()?;

            let info = TensorInfo {
                name,
                dims,
                ty,
                offset,
            };
            // Validate byte size consistency early.
            info.size_bytes()?;
            tensors.push(info);
        }

        // Data section starts after the directory, aligned up.
        let directory_end = cur.position();
        let data_start = directory_end.div_ceil(alignment) * alignment;

        let reader = Self {
            version,
            metadata,
            tensors,
            alignment,
            data_start,
            file_len,
        };

        reader.validate_tensor_offsets()?;
        Ok(reader)
    }

    fn validate_tensor_offsets(&self) -> Result<()> {
        let mut max_end = 0u64;
        for t in &self.tensors {
            if t.offset % self.alignment != 0 {
                return Err(LlamaError::invalid_gguf(format!(
                    "tensor '{}' offset {} is not aligned to the file alignment {}",
                    t.name, t.offset, self.alignment
                )));
            }
            let size = t.size_bytes()?;
            let end = t.offset.checked_add(size).ok_or_else(|| {
                LlamaError::invalid_gguf(format!("tensor '{}' size overflows", t.name))
            })?;
            if self.data_start + end > self.file_len {
                return Err(LlamaError::invalid_gguf(format!(
                    "tensor '{}' data [{}, {}) extends beyond the end of the file (len {})",
                    t.name,
                    self.data_start + t.offset,
                    self.data_start + end,
                    self.file_len
                )));
            }
            max_end = max_end.max(end);
        }
        let _ = max_end;
        Ok(())
    }

    /// GGUF version (2 or 3).
    #[must_use]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Parsed metadata table.
    #[must_use]
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Tensor directory.
    #[must_use]
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Finds a tensor by name.
    #[must_use]
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Data-section alignment.
    #[must_use]
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Byte offset of the data section in the file.
    #[must_use]
    pub fn data_start(&self) -> u64 {
        self.data_start
    }
}

/// Backing storage of a [`GgufFile`].
enum Backing {
    /// A read-only memory map of a file on disk.
    Mmap(Mmap),
    /// An owned in-memory buffer.
    Bytes(Vec<u8>),
}

impl Backing {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(m) => m,
            Self::Bytes(b) => b,
        }
    }
}

/// A memory-mapped GGUF model file.
///
/// Owns the mapping (or buffer); tensor bytes are borrowed zero-copy from it.
pub struct GgufFile {
    reader: GgufReader,
    backing: Backing,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("version", &self.reader.version)
            .field("tensors", &self.reader.tensors.len())
            .field("metadata_entries", &self.reader.metadata.len())
            .finish_non_exhaustive()
    }
}

impl GgufFile {
    /// Opens and maps a GGUF file from disk.
    // The `unsafe` block below is the mmap call; see its SAFETY comment. It is
    // the only `unsafe` in this crate, so we allow the lint right here rather
    // than relaxing the workspace policy crate-wide.
    #[allow(unsafe_code)]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| {
            LlamaError::Io(std::io::Error::other(format!(
                "cannot open model file '{}': {e}",
                path.display()
            )))
        })?;
        let mmap = unsafe {
            // SAFETY: the file is opened read-only and we never hand out
            // mutable access to the mapping. Between creating and using the
            // map, the file could be truncated externally by another process;
            // this is the standard risk profile of every mmap-based model
            // loader (including llama.cpp) and is mitigated by the immediate
            // bounds validation below.
            Mmap::map(&file)?
        };
        let file_len = file.metadata()?.len();
        if mmap.len() as u64 != file_len {
            return Err(LlamaError::invalid_gguf(
                "file changed size while mapping (was it truncated?)",
            ));
        }
        let reader = GgufReader::parse(&mmap, file_len)?;
        Ok(Self {
            reader,
            backing: Backing::Mmap(mmap),
        })
    }

    /// Parses a GGUF model from an in-memory buffer (no mmap).
    ///
    /// Useful for tests and for loading from non-filesystem sources.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        let file_len = data.len() as u64;
        let reader = GgufReader::parse(&data, file_len)?;
        Ok(Self {
            reader,
            backing: Backing::Bytes(data),
        })
    }

    /// The parsed reader view.
    #[must_use]
    pub fn reader(&self) -> &GgufReader {
        &self.reader
    }

    /// Metadata table.
    #[must_use]
    pub fn metadata(&self) -> &GgufMetadata {
        self.reader.metadata()
    }

    /// Tensor directory.
    #[must_use]
    pub fn tensors(&self) -> &[TensorInfo] {
        self.reader.tensors()
    }

    /// Finds a tensor by name.
    #[must_use]
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.reader.find_tensor(name)
    }

    /// GGUF version.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.reader.version
    }

    /// Byte offset of the data section.
    #[must_use]
    pub fn data_start(&self) -> u64 {
        self.reader.data_start
    }

    /// Alignment.
    #[must_use]
    pub fn alignment(&self) -> u64 {
        self.reader.alignment
    }

    /// Returns the raw tensor bytes for a tensor directory entry.
    ///
    /// Zero-copy: the slice borrows from the memory map.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> Result<&[u8]> {
        let start = self.reader.data_start + info.offset;
        let len = info.size_bytes()?;
        let end = start + len;
        let data = self.backing.as_slice();
        if end > data.len() as u64 {
            // Defensive: the directory was already validated against the file
            // length, so this indicates a logic error or a truncated mmap.
            return Err(LlamaError::invalid_gguf(format!(
                "tensor '{}' is out of bounds after validation",
                info.name
            )));
        }
        Ok(&data[start as usize..end as usize])
    }

    /// Total size of the backing storage in bytes.
    #[must_use]
    pub fn file_size(&self) -> u64 {
        self.backing.as_slice().len() as u64
    }
}
