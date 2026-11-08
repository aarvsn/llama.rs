//! GGUF metadata values: a typed tree with safe, checked accessors.

use std::collections::BTreeMap;

use llama_core::{LlamaError, Result};

/// A GGUF metadata value.
///
/// Mirrors the GGUF value types (u8, i8, u16, i16, u32, i32, f32, u64, i64,
/// f64, bool, string, array).
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Signed 8-bit integer.
    I8(i8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Signed 16-bit integer.
    I16(i16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Signed 32-bit integer.
    I32(i32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 64-bit integer.
    I64(i64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Homogeneous array.
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    /// GGUF type tag for this value.
    #[must_use]
    pub fn type_tag(&self) -> u32 {
        match self {
            Self::U8(_) => 0,
            Self::I8(_) => 1,
            Self::U16(_) => 2,
            Self::I16(_) => 3,
            Self::U32(_) => 4,
            Self::I32(_) => 5,
            Self::F32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array(_) => 9,
            Self::U64(_) => 10,
            Self::I64(_) => 11,
            Self::F64(_) => 12,
        }
    }

    /// Human-readable type name (used in error messages).
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
            Self::Bool(_) => "bool",
            Self::String(_) => "string",
            Self::Array(_) => "array",
        }
    }

    /// Interprets the value as `u64` when the raw type is an integer kind.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(u64::from(v)),
            Self::I8(v) => Some(u64::from(v.unsigned_abs())),
            Self::U16(v) => Some(u64::from(v)),
            Self::I16(v) => Some(u64::from(v.unsigned_abs())),
            Self::U32(v) => Some(u64::from(v)),
            Self::I32(v) => Some(u64::from(v.unsigned_abs())),
            Self::U64(v) => Some(v),
            Self::I64(v) => Some(v.unsigned_abs()),
            _ => None,
        }
    }

    /// Interprets the value as `f64` when the raw type is numeric.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(f64::from(v)),
            Self::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    /// Interprets the value as `bool`.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Interprets the value as a string slice.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Interprets the value as an array slice.
    #[must_use]
    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }
}

impl std::fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => write!(f, "{s:?}"),
            Self::Array(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().take(4).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                if items.len() > 4 {
                    write!(f, ", ... ({} items)", items.len())?;
                }
                write!(f, "]")
            }
            other => match other.as_f64() {
                Some(v) => write!(f, "{v}"),
                None => write!(f, "{other:?}"),
            },
        }
    }
}

/// The parsed metadata table of a GGUF file.
#[derive(Debug, Clone, Default)]
pub struct GgufMetadata {
    entries: BTreeMap<String, MetadataValue>,
}

impl GgufMetadata {
    /// Creates an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Inserts a key (used by the reader and writer).
    pub fn insert(&mut self, key: String, value: MetadataValue) {
        self.entries.insert(key, value);
    }

    /// Raw lookup.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    /// Iterates over all entries in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetadataValue)> {
        self.entries.iter()
    }

    /// Requires a value, erroring with the missing-key variant.
    pub fn require(&self, key: &str) -> Result<&MetadataValue> {
        self.entries
            .get(key)
            .ok_or_else(|| LlamaError::MissingMetadata(key.to_string()))
    }

    /// Reads an unsigned integer (any GGUF integer width).
    pub fn get_u64(&self, key: &str) -> Result<u64> {
        let value = self.require(key)?;
        value.as_u64().ok_or_else(|| {
            LlamaError::InvalidMetadata(format!(
                "'{key}' expected an integer, found {}",
                value.type_name()
            ))
        })
    }

    /// Reads an `f32`/`f64` (or integer, cast) value.
    pub fn get_f32(&self, key: &str) -> Result<f32> {
        let value = self.require(key)?;
        value.as_f64().map(|v| v as f32).ok_or_else(|| {
            LlamaError::InvalidMetadata(format!(
                "'{key}' expected a number, found {}",
                value.type_name()
            ))
        })
    }

    /// Reads a boolean.
    pub fn get_bool(&self, key: &str) -> Result<bool> {
        let value = self.require(key)?;
        value.as_bool().ok_or_else(|| {
            LlamaError::InvalidMetadata(format!(
                "'{key}' expected bool, found {}",
                value.type_name()
            ))
        })
    }

    /// Reads a string.
    pub fn get_string(&self, key: &str) -> Result<&str> {
        let value = self.require(key)?;
        value.as_str().ok_or_else(|| {
            LlamaError::InvalidMetadata(format!(
                "'{key}' expected string, found {}",
                value.type_name()
            ))
        })
    }

    /// Reads an optional string.
    #[must_use]
    pub fn get_string_opt(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(MetadataValue::as_str)
    }

    /// Reads an array.
    pub fn get_array(&self, key: &str) -> Result<&[MetadataValue]> {
        let value = self.require(key)?;
        value.as_array().ok_or_else(|| {
            LlamaError::InvalidMetadata(format!(
                "'{key}' expected array, found {}",
                value.type_name()
            ))
        })
    }

    /// Reads an unsigned integer with a default when the key is absent.
    #[must_use]
    pub fn get_u64_or(&self, key: &str, default: u64) -> u64 {
        self.get(key)
            .and_then(MetadataValue::as_u64)
            .unwrap_or(default)
    }

    /// Reads a float with a default when the key is absent.
    #[must_use]
    pub fn get_f32_or(&self, key: &str, default: f32) -> f32 {
        self.get(key)
            .and_then(|v| v.as_f64().map(|x| x as f32))
            .unwrap_or(default)
    }

    /// Reads a string array as `Vec<String>` (errors on non-string items).
    pub fn get_string_vec(&self, key: &str) -> Result<Vec<String>> {
        let array = self.get_array(key)?;
        array
            .iter()
            .map(|item| {
                item.as_str().map(str::to_owned).ok_or_else(|| {
                    LlamaError::InvalidMetadata(format!("'{key}' array contains a non-string item"))
                })
            })
            .collect()
    }

    /// Reads an f32 array (errors on non-numeric items).
    pub fn get_f32_vec(&self, key: &str) -> Result<Vec<f32>> {
        let array = self.get_array(key)?;
        array
            .iter()
            .map(|item| {
                item.as_f64().map(|v| v as f32).ok_or_else(|| {
                    LlamaError::InvalidMetadata(format!(
                        "'{key}' array contains a non-numeric item"
                    ))
                })
            })
            .collect()
    }

    /// Reads an i32 array (used for tokenizer token types).
    pub fn get_i32_vec(&self, key: &str) -> Result<Vec<i32>> {
        let array = self.get_array(key)?;
        array
            .iter()
            .map(|item| {
                // GGUF stores token types as i32; tolerate wider ints.
                match item.as_u64().map(|v| v as i64) {
                    Some(v) if i32::try_from(v).is_ok() => Ok(v as i32),
                    _ => Err(LlamaError::InvalidMetadata(format!(
                        "'{key}' array contains a non-integer item"
                    ))),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessors_type_check() {
        let mut meta = GgufMetadata::new();
        meta.insert("a.u32".into(), MetadataValue::U32(7));
        meta.insert("a.str".into(), MetadataValue::String("llama".into()));
        meta.insert("a.bool".into(), MetadataValue::Bool(true));
        meta.insert("a.f32".into(), MetadataValue::F32(1.5));
        meta.insert(
            "a.arr".into(),
            MetadataValue::Array(vec![MetadataValue::U8(1), MetadataValue::U8(2)]),
        );

        assert_eq!(meta.get_u64("a.u32").unwrap(), 7);
        assert_eq!(meta.get_string("a.str").unwrap(), "llama");
        assert!(meta.get_bool("a.bool").unwrap());
        assert!((meta.get_f32("a.f32").unwrap() - 1.5).abs() < 1e-6);
        assert_eq!(meta.get_array("a.arr").unwrap().len(), 2);

        assert!(matches!(
            meta.get_string("a.u32"),
            Err(LlamaError::InvalidMetadata(_))
        ));
        assert!(matches!(
            meta.get_u64("missing"),
            Err(LlamaError::MissingMetadata(_))
        ));
    }

    #[test]
    fn integer_widening_is_forgiving() {
        let mut meta = GgufMetadata::new();
        meta.insert("v".into(), MetadataValue::U8(3));
        assert_eq!(meta.get_u64("v").unwrap(), 3);
        meta.insert("v".into(), MetadataValue::I32(-1));
        // Negative integers surface as their magnitude; documented GGUF keys we
        // use as u64 are always non-negative.
        assert_eq!(meta.get_u64("v").unwrap(), 1);
    }
}
