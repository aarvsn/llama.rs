//! GGUF v3 writer.
//!
//! Used by the test-fixture generators and available to conversion tooling.
//! Produces byte-identical structure to what llama.cpp writes (v3, default
//! alignment 32).

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use llama_core::{GgmlType, Result};

use crate::metadata::MetadataValue;

/// Metadata key/value pairs awaiting serialization.
type MetaMap = BTreeMap<String, MetadataValue>;

/// A tensor being written: directory info plus its serialized payload.
pub struct TensorPayload {
    /// Tensor name.
    pub name: String,
    /// Dimensions (fastest-varying first).
    pub dims: Vec<u64>,
    /// Tensor type.
    pub ty: GgmlType,
    /// Serialized tensor bytes (must match `dims`/`ty` geometry).
    pub data: Vec<u8>,
}

/// Incremental GGUF v3 serializer.
#[derive(Default)]
pub struct GgufWriter {
    meta: MetaMap,
    tensors: Vec<TensorPayload>,
}

impl GgufWriter {
    /// Creates an empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds/overwrites a metadata entry.
    pub fn set(&mut self, key: impl Into<String>, value: MetadataValue) -> &mut Self {
        self.meta.insert(key.into(), value);
        self
    }

    /// Adds a u32 metadata entry.
    pub fn set_u32(&mut self, key: impl Into<String>, value: u32) -> &mut Self {
        self.set(key, MetadataValue::U32(value))
    }

    /// Adds a u64 metadata entry.
    pub fn set_u64(&mut self, key: impl Into<String>, value: u64) -> &mut Self {
        self.set(key, MetadataValue::U64(value))
    }

    /// Adds an f32 metadata entry.
    pub fn set_f32(&mut self, key: impl Into<String>, value: f32) -> &mut Self {
        self.set(key, MetadataValue::F32(value))
    }

    /// Adds a bool metadata entry.
    pub fn set_bool(&mut self, key: impl Into<String>, value: bool) -> &mut Self {
        self.set(key, MetadataValue::Bool(value))
    }

    /// Adds a string metadata entry.
    pub fn set_str(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.set(key, MetadataValue::String(value.into()))
    }

    /// Adds a string-array metadata entry.
    pub fn set_str_vec(&mut self, key: impl Into<String>, values: Vec<String>) -> &mut Self {
        self.set(
            key,
            MetadataValue::Array(values.into_iter().map(MetadataValue::String).collect()),
        )
    }

    /// Adds an f32-array metadata entry.
    pub fn set_f32_vec(&mut self, key: impl Into<String>, values: Vec<f32>) -> &mut Self {
        self.set(
            key,
            MetadataValue::Array(values.into_iter().map(MetadataValue::F32).collect()),
        )
    }

    /// Adds an i32-array metadata entry.
    pub fn set_i32_vec(&mut self, key: impl Into<String>, values: Vec<i32>) -> &mut Self {
        self.set(
            key,
            MetadataValue::Array(values.into_iter().map(MetadataValue::I32).collect()),
        )
    }

    /// Registers a tensor payload. Offsets are assigned at write time.
    pub fn add_tensor(&mut self, payload: TensorPayload) -> &mut Self {
        self.tensors.push(payload);
        self
    }

    /// Number of registered tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Serializes the file to a byte vector.
    pub fn to_bytes(&self, alignment: u64) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(4096);
        self.write_into(&mut out, alignment)?;
        Ok(out)
    }

    /// Serializes the file to a path.
    pub fn write_to_file(&self, path: impl AsRef<Path>, alignment: u64) -> Result<()> {
        let bytes = self.to_bytes(alignment)?;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&bytes)?;
        Ok(())
    }

    fn write_into(&self, out: &mut Vec<u8>, alignment: u64) -> Result<()> {
        // Header.
        out.extend_from_slice(&crate::GGUF_MAGIC.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes()); // version 3
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.meta.len() as u64).to_le_bytes());

        // Metadata.
        for (key, value) in &self.meta {
            write_str(out, key);
            out.extend_from_slice(&value.type_tag().to_le_bytes());
            write_value(out, value);
        }

        // Tensor directory. Offsets are relative to the (aligned) data section.
        let mut offset: u64 = 0;
        let mut entries = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            let expected =
                t.ty.row_size_bytes(t.dims.iter().product())
                    .ok_or_else(|| {
                        llama_core::LlamaError::InvalidTensor(format!(
                            "tensor '{}' dims are not a multiple of the {} block size",
                            t.name,
                            t.ty.block_size()
                        ))
                    })?;
            if t.data.len() as u64 != expected {
                return Err(llama_core::LlamaError::InvalidTensor(format!(
                    "tensor '{}' payload is {} bytes, expected {} for dims {:?} type {}",
                    t.name,
                    t.data.len(),
                    expected,
                    t.dims,
                    t.ty
                )));
            }
            entries.push((offset, t));
            offset += expected;
        }
        for (off, t) in &entries {
            write_str(out, &t.name);
            out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&t.ty.to_raw().to_le_bytes());
            out.extend_from_slice(&off.to_le_bytes());
        }

        // Align the data section.
        while out.len() % alignment as usize != 0 {
            out.push(0);
        }

        // Data.
        for (_, t) in &entries {
            out.extend_from_slice(&t.data);
        }
        Ok(())
    }
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_value(out: &mut Vec<u8>, value: &MetadataValue) {
    match value {
        MetadataValue::U8(v) => out.push(*v),
        MetadataValue::I8(v) => out.push(*v as u8),
        MetadataValue::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        MetadataValue::F32(v) => out.extend_from_slice(&v.to_bits().to_le_bytes()),
        MetadataValue::F64(v) => out.extend_from_slice(&v.to_bits().to_le_bytes()),
        MetadataValue::Bool(v) => out.push(u8::from(*v)),
        MetadataValue::String(s) => write_str(out, s),
        MetadataValue::Array(items) => {
            if let Some(first) = items.first() {
                out.extend_from_slice(&first.type_tag().to_le_bytes());
            } else {
                // Empty arrays are written with a u8 element type (GGUF
                // readers accept any tag for zero elements).
                out.extend_from_slice(&0u32.to_le_bytes());
            }
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                write_value(out, item);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgufFile;
    use llama_core::LlamaError;

    #[test]
    fn round_trip_minimal_file() {
        let mut w = GgufWriter::new();
        w.set_str("general.architecture", "llama");
        w.set_u64("llama.context_length", 128);
        w.add_tensor(TensorPayload {
            name: "token_embd.weight".into(),
            dims: vec![64, 32],
            ty: GgmlType::F32,
            data: vec![0.25f32; 64 * 32]
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect(),
        });

        let bytes = w.to_bytes(32).unwrap();
        let file = GgufFile::from_bytes(bytes).unwrap();
        assert_eq!(file.version(), 3);
        assert_eq!(
            file.metadata().get_string("general.architecture").unwrap(),
            "llama"
        );
        assert_eq!(
            file.metadata().get_u64("llama.context_length").unwrap(),
            128
        );
        let info = file.find_tensor("token_embd.weight").unwrap();
        let data = file.tensor_bytes(info).unwrap();
        assert_eq!(data.len(), 64 * 32 * 4);
        assert_eq!(
            f32::from_le_bytes([data[0], data[1], data[2], data[3]]),
            0.25
        );
    }

    #[test]
    fn rejects_bad_payload_size() {
        let mut w = GgufWriter::new();
        w.add_tensor(TensorPayload {
            name: "x".into(),
            dims: vec![33],
            ty: GgmlType::Q4_0,
            data: vec![0; 10],
        });
        match w.to_bytes(32) {
            Err(LlamaError::InvalidTensor(_)) => {}
            other => panic!("expected InvalidTensor, got {other:?}"),
        }
    }
}
