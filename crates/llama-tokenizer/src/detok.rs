//! Incremental detokenization with UTF-8 buffering.
//!
//! Tokens can split multi-byte UTF-8 sequences (byte fallback in SPM,
//! byte-level BPE), so decoded bytes are buffered until they form complete
//! characters.

use llama_core::TokenId;

use crate::vocab::{TokenKind, Vocabulary};

/// Streaming detokenizer.
pub struct Detokenizer {
    /// Pending bytes that may not form complete UTF-8 yet.
    buffer: Vec<u8>,
    /// Byte-level BPE vocabulary (pieces map through the GPT-2 table).
    byte_level: bool,
}

impl Detokenizer {
    /// Creates a detokenizer.
    #[must_use]
    pub fn new(byte_level: bool) -> Self {
        Self {
            buffer: Vec::new(),
            byte_level,
        }
    }

    /// Pushes a token and returns the text that is safe to emit now.
    #[must_use]
    pub fn push(&mut self, vocab: &Vocabulary, id: TokenId, special: bool) -> String {
        let kind = vocab.kind(id);
        match kind {
            TokenKind::Control | TokenKind::Unknown | TokenKind::Unused => {
                if special {
                    vocab.piece(id).to_string()
                } else {
                    String::new()
                }
            }
            TokenKind::UserDefined | TokenKind::Normal | TokenKind::Byte => {
                self.push_piece_text(vocab, id, kind)
            }
        }
    }

    fn push_piece_text(&mut self, vocab: &Vocabulary, id: TokenId, kind: TokenKind) -> String {
        let piece = vocab.piece(id);
        if kind == TokenKind::Byte || vocab.token_byte(id).is_some() {
            // Byte token: append the raw byte.
            if let Some(byte) = vocab.token_byte(id) {
                self.buffer.push(byte);
            }
        } else if self.byte_level {
            // Byte-level BPE: each character of the piece maps back to a byte.
            for c in piece.chars() {
                self.buffer.push(reverse_byte(c));
            }
        } else {
            // SPM: replace ▁ with space.
            for c in piece.chars() {
                if c == '\u{2581}' {
                    self.buffer.push(b' ');
                } else {
                    let mut buf = [0u8; 4];
                    self.buffer
                        .extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        self.drain_valid_utf8()
    }

    /// Emits the longest valid UTF-8 prefix of the buffer, keeping any
    /// incomplete trailing sequence for later.
    fn drain_valid_utf8(&mut self) -> String {
        let mut valid_end = 0usize;
        let mut i = 0usize;
        while i < self.buffer.len() {
            let b = self.buffer[i];
            let len = if b < 0x80 {
                1
            } else if b >= 0xF0 {
                4
            } else if b >= 0xE0 {
                3
            } else if b >= 0xC0 {
                2
            } else {
                // Stray continuation byte: treat as invalid, emit as lossy.
                1
            };
            if i + len > self.buffer.len() {
                break; // incomplete sequence: keep for later
            }
            if std::str::from_utf8(&self.buffer[i..i + len]).is_ok() {
                valid_end = i + len;
                i += len;
            } else {
                // Invalid sequence: replace this byte lossily and continue.
                valid_end = i + 1;
                i += 1;
            }
        }
        let text = String::from_utf8_lossy(&self.buffer[..valid_end]).into_owned();
        self.buffer.drain(..valid_end);
        text
    }

    /// Flushes any remaining buffered bytes (lossy).
    #[must_use]
    pub fn flush(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        text
    }
}

/// Reverses the GPT-2 byte-to-unicode mapping for one character.
///
/// Characters outside the mapping (possible only with malformed
/// vocabularies) map to `?` so the output stays valid UTF-8.
fn reverse_byte(c: char) -> u8 {
    static REVERSE: std::sync::OnceLock<std::collections::HashMap<char, u8>> =
        std::sync::OnceLock::new();
    let reverse = REVERSE.get_or_init(|| {
        let mut map = std::collections::HashMap::with_capacity(256);
        for b in 0..=255u8 {
            map.insert(crate::bpe::byte_to_unicode_char(b), b);
        }
        map
    });
    reverse.get(&c).copied().unwrap_or(b'?')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spm_vocab() -> Vocabulary {
        let pieces = vec![
            "<unk>".into(),
            "<s>".into(),
            "</s>".into(),
            "▁hello".into(),
            "▁w".into(),
            "orld".into(),
            "<0xE2>".into(),
            "<0x99>".into(),
            "<0xA5>".into(),
            "<0x41>".into(),
        ];
        let kinds = vec![2, 3, 3, 1, 1, 1, 1, 1, 1, 1];
        let scores = vec![0.0; 10];
        Vocabulary::new(pieces, &kinds, scores).unwrap()
    }

    #[test]
    fn simple_pieces() {
        let v = spm_vocab();
        let mut d = Detokenizer::new(false);
        assert_eq!(d.push(&v, 3, false), " hello");
    }

    #[test]
    fn control_tokens_hidden_unless_special() {
        let v = spm_vocab();
        let mut d = Detokenizer::new(false);
        assert_eq!(d.push(&v, 1, false), "");
        assert_eq!(d.push(&v, 1, true), "<s>");
    }

    #[test]
    fn multibyte_split_across_byte_tokens() {
        let v = spm_vocab();
        let mut d = Detokenizer::new(false);
        assert_eq!(d.push(&v, 6, false), ""); // E2 incomplete
        assert_eq!(d.push(&v, 7, false), ""); // E2 99 still incomplete
        assert_eq!(d.push(&v, 8, false), "♥"); // complete
    }

    #[test]
    fn byte_level_bpe_pieces() {
        // Pieces in encoded space: "Ġ" = space, "hello" plain.
        let pieces = vec!["<unk>".into(), "Ġhello".into(), "Ċ".into()];
        let kinds = vec![2, 1, 1];
        let v = Vocabulary::new(pieces, &kinds, vec![0.0; 3]).unwrap();
        let mut d = Detokenizer::new(true);
        assert_eq!(d.push(&v, 1, false), " hello");
        assert_eq!(d.push(&v, 2, false), "\n");
    }
}
