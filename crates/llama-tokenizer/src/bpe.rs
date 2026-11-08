//! Byte-level BPE encoding (GPT-2 style), matching HF `tokenizers`:
//!
//! 1. Pre-tokenize into word fragments (see [`crate::pre`]).
//! 2. Map each fragment's UTF-8 bytes through the GPT-2 `bytes_to_unicode`
//!    table so every byte is a visible character.
//! 3. Greedily apply merges in rank order.
//! 4. Look up the resulting pieces in the vocabulary.

use llama_core::{LlamaError, Result, TokenId};

use crate::vocab::Vocabulary;

/// The GPT-2 byte-to-unicode table (computed once).
#[must_use]
pub fn byte_to_unicode_char(byte: u8) -> char {
    // Bytes 33..=126, 161..=172, 174..=255 map to themselves; the remaining
    // bytes map to 256 + n in order.
    match byte {
        33..=126 | 161..=172 | 174..=255 => char::from(byte),
        _ => {
            static TABLE: std::sync::OnceLock<[u8; 256]> = std::sync::OnceLock::new();
            let table = TABLE.get_or_init(build_reverse_table);
            // table[b] = the unicode code point (as u32 truncated to u8 only
            // for the identity range) — we store the mapped code point
            // directly for the non-identity range.
            let _ = table;
            non_identity_char(byte)
        }
    }
}

/// Computes the mapped code point for bytes outside the identity range.
fn non_identity_char(byte: u8) -> char {
    static OFFSETS: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        let mut map = [0u32; 256];
        let mut n = 0u32;
        for b in 0..=255u8 {
            let identity = matches!(b, 33..=126 | 161..=172 | 174..=255);
            if !identity {
                map[b as usize] = 256 + n;
                n += 1;
            }
        }
        map
    });
    char::from_u32(offsets[byte as usize]).expect("mapped code point is always valid")
}

fn build_reverse_table() -> [u8; 256] {
    // Only used for documentation of the identity range; not a hot path.
    let mut t = [0u8; 256];
    for (i, v) in t.iter_mut().enumerate() {
        *v = i as u8;
    }
    t
}

/// Byte-encodes a fragment into the BPE symbol string.
#[must_use]
pub fn byte_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for &b in text.as_bytes() {
        out.push(byte_to_unicode_char(b));
    }
    out
}

/// Encodes one pre-tokenized word and appends token ids to `out`.
pub fn encode_word(vocab: &Vocabulary, word: &str, out: &mut Vec<TokenId>) -> Result<()> {
    let encoded = byte_encode(word);
    let mut symbols: Vec<String> = encoded.chars().map(String::from).collect();
    if symbols.is_empty() {
        return Ok(());
    }

    // Standard BPE merge loop: repeatedly merge the pair with the best rank.
    loop {
        let mut best_rank: Option<u32> = None;
        let mut best_idx = usize::MAX;
        for i in 0..symbols.len() - 1 {
            if let Some(rank) = vocab.merge_rank(&symbols[i], &symbols[i + 1]) {
                if best_rank.is_none() || rank < best_rank.expect("checked") {
                    best_rank = Some(rank);
                    best_idx = i;
                }
            }
        }
        let Some(_rank) = best_rank else {
            break;
        };
        let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
        symbols[best_idx] = merged;
        symbols.remove(best_idx + 1);
        if symbols.len() == 1 {
            break;
        }
    }

    for symbol in &symbols {
        match vocab.id_of(symbol) {
            Some(id) => out.push(id),
            None => {
                return Err(LlamaError::Tokenizer(format!(
                    "BPE symbol {symbol:?} is not in the vocabulary (word {word:?}); \
                     the GGUF vocabulary does not cover its bytes"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_identity_range() {
        assert_eq!(byte_to_unicode_char(b'a'), 'a');
        assert_eq!(byte_to_unicode_char(b'Z'), 'Z');
        assert_eq!(byte_to_unicode_char(b' '), 'Ġ'); // 0x20 -> 256
        assert_eq!(byte_to_unicode_char(b'\n'), 'Ċ'); // 0x0A -> 258
    }

    #[test]
    fn byte_encode_word() {
        assert_eq!(byte_encode(" hello"), "Ġhello");
        assert_eq!(byte_encode("a\n"), "aĊ");
    }

    #[test]
    fn bpe_merges_apply_in_rank_order() {
        let pieces = vec![
            "<unk>".into(),
            "a".into(),
            "b".into(),
            "c".into(),
            "aĠ".into(),
            "aĠb".into(),
            "c!".into(),
        ];
        let kinds = vec![2, 1, 1, 1, 1, 1, 1];
        let scores = vec![0.0; 7];
        let mut vocab = crate::vocab::Vocabulary::new(pieces, &kinds, scores).unwrap();
        vocab
            .build_merge_ranks(&["a Ġ".to_string(), "aĠ b".to_string(), "c !".to_string()])
            .unwrap();

        let mut out = Vec::new();
        encode_word(&vocab, "a b", &mut out).unwrap();
        // "a b" byte-encodes to "aĠb"; merges: "a Ġ" (rank 0) then "aĠ b" (rank 1).
        assert_eq!(out, vec![5]);

        let mut out = Vec::new();
        encode_word(&vocab, "c!", &mut out).unwrap();
        assert_eq!(out, vec![6]);
    }

    #[test]
    fn bpe_unknown_symbol_errors() {
        let pieces = vec!["<unk>".to_string(), "a".to_string()];
        let kinds = vec![2, 1];
        let vocab = crate::vocab::Vocabulary::new(pieces, &kinds, vec![0.0; 2]).unwrap();
        let mut out = Vec::new();
        assert!(encode_word(&vocab, "z", &mut out).is_err());
    }
}
