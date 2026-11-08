//! `SentencePiece` tokenizer (the `tokenizer.ggml.model = "llama"` path),
//! faithful to the llama.cpp reference implementation:
//!
//! 1. Prepend a dummy space when the fragment starts a sequence and escape
//!    `' '` to `'▁'` (U+2581).
//! 2. Split into UTF-8 characters as merge symbols.
//! 3. Repeatedly merge the highest-scoring adjacent pair (ties: leftmost).
//!    Pairs only enter the queue when their concatenation is a vocabulary
//!    piece, so long pieces like `▁Once` form through chains of merges
//!    (`▁`+`O` → `▁O`, `▁O`+`n` → `▁On`, `c`+`e` → `ce`, `▁On`+`ce` →
//!    `▁Once`).
//! 4. Resegment: emit vocab pieces, or fall back to `<0xXX>` byte tokens.

use std::collections::{BinaryHeap, HashMap};

use llama_core::{LlamaError, Result, TokenId};

use crate::vocab::Vocabulary;

/// A pending merge candidate.
#[derive(Clone)]
struct Bigram {
    left: usize,
    right: usize,
    score: f32,
    size: usize,
}

// BinaryHeap pops the greatest element. We want the highest score first and,
// on ties, the smallest left index — so `Ord` treats higher scores and
// smaller indices as "greater".
impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.left == other.left && self.size == other.size
    }
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(other.left.cmp(&self.left))
    }
}

#[derive(Clone, Copy)]
struct Symbol {
    start: usize,
    len: usize,
    prev: i32,
    next: i32,
}

/// Encodes one raw text fragment (no special tokens inside).
///
/// `at_sequence_start` controls the dummy-prefix behaviour (llama.cpp emits
/// the leading ▁ only for the first fragment after a special token).
pub fn encode(
    vocab: &Vocabulary,
    text: &str,
    at_sequence_start: bool,
    out: &mut Vec<TokenId>,
) -> Result<()> {
    // Dummy prefix + whitespace escape.
    let mut normalized = String::with_capacity(text.len() + 4);
    if at_sequence_start {
        normalized.push('\u{2581}');
    }
    for c in text.chars() {
        if c == ' ' {
            normalized.push('\u{2581}');
        } else {
            normalized.push(c);
        }
    }
    if normalized.is_empty() {
        return Ok(());
    }

    // Symbols over UTF-8 characters, chained with prev/next indices.
    let mut symbols: Vec<Symbol> = Vec::new();
    let mut off = 0usize;
    for ch in normalized.chars() {
        symbols.push(Symbol {
            start: off,
            len: ch.len_utf8(),
            prev: symbols.len() as i32 - 1,
            next: symbols.len() as i32 + 1,
        });
        off += ch.len_utf8();
    }
    let n = symbols.len();
    symbols[n - 1].next = -1;

    let mut rev_merge: HashMap<String, (usize, usize)> = HashMap::new();
    let mut queue: BinaryHeap<Bigram> = BinaryHeap::new();

    let try_add = |symbols: &[Symbol],
                   i: i32,
                   queue: &mut BinaryHeap<Bigram>,
                   rev_merge: &mut HashMap<String, (usize, usize)>| {
        if i < 0 {
            return;
        }
        let left = i as usize;
        let right = symbols[left].next;
        if right < 0 {
            return;
        }
        let right = right as usize;
        let text = &normalized
            [symbols[left].start..symbols[left].start + symbols[left].len + symbols[right].len];
        let Some(id) = vocab.id_of(text) else {
            return;
        };
        queue.push(Bigram {
            left,
            right,
            score: vocab.score(id),
            size: symbols[left].len + symbols[right].len,
        });
        rev_merge.insert(text.to_string(), (left, right));
    };

    for i in 0..n - 1 {
        try_add(&symbols, i as i32, &mut queue, &mut rev_merge);
    }

    while let Some(bigram) = queue.pop() {
        let (left, right) = (bigram.left, bigram.right);
        if symbols[left].len == 0
            || symbols[right].len == 0
            || symbols[left].len + symbols[right].len != bigram.size
        {
            continue; // stale queue entry
        }
        // Merge `right` into `left`.
        let (next, prev) = (symbols[right].next, symbols[left].prev);
        symbols[left].len += symbols[right].len;
        symbols[right].len = 0;
        symbols[left].next = next;
        if next >= 0 {
            symbols[next as usize].prev = left as i32;
        }
        if prev >= 0 {
            symbols[prev as usize].next = left as i32;
        }
        try_add(&symbols, symbols[left].prev, &mut queue, &mut rev_merge);
        try_add(&symbols, left as i32, &mut queue, &mut rev_merge);
    }

    // Walk the chain and resegment each merged span.
    let mut i = 0i32;
    while i != -1 {
        let sym = symbols[i as usize];
        resegment(vocab, &normalized, &symbols, i as usize, &rev_merge, out)?;
        i = sym.next;
    }
    Ok(())
}

/// Emits tokens for the span of symbol `idx`, recursing into the two merge
/// parts when the merged text is not itself a vocabulary piece.
fn resegment(
    vocab: &Vocabulary,
    normalized: &str,
    symbols: &[Symbol],
    idx: usize,
    rev_merge: &HashMap<String, (usize, usize)>,
    out: &mut Vec<TokenId>,
) -> Result<()> {
    let sym = symbols[idx];
    let text = &normalized[sym.start..sym.start + sym.len];

    if let Some(id) = vocab.id_of(text) {
        if vocab.kind(id) != crate::vocab::TokenKind::Unused {
            out.push(id);
            return Ok(());
        }
    }

    if let Some(&(left, right)) = rev_merge.get(text) {
        resegment(vocab, normalized, symbols, left, rev_merge, out)?;
        resegment(vocab, normalized, symbols, right, rev_merge, out)?;
        return Ok(());
    }

    // Byte fallback.
    for &byte in text.as_bytes() {
        match vocab.byte_token(byte) {
            Some(id) => out.push(id),
            None => match vocab.unk() {
                Some(unk) => out.push(unk),
                None => {
                    return Err(LlamaError::Tokenizer(format!(
                        "cannot encode byte {byte:#04x}: vocabulary has neither a byte token nor an unk token"
                    )));
                }
            },
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Realistic vocabulary with multi-character pieces that only form via
    /// merge chains (mirrors the llama tokenizer behaviour).
    fn test_vocab() -> Vocabulary {
        let pieces = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "▁".to_string(),
            "▁O".to_string(),
            "▁On".to_string(),
            "▁Once".to_string(),
            "n".to_string(),
            "nc".to_string(),
            "ce".to_string(),
            "e".to_string(),
            "▁hello".to_string(),
            "▁h".to_string(),
            "el".to_string(),
            "lo".to_string(),
            "▁he".to_string(),
            "▁hel".to_string(),
            "▁hell".to_string(),
            "ello".to_string(),
            "▁upon".to_string(),
            "▁a".to_string(),
            "▁time".to_string(),
            "<0x41>".to_string(), // 'A'
            "<0xE2>".to_string(), // UTF-8 lead byte for ♥
            "<0x99>".to_string(),
            "<0xA5>".to_string(),
        ];
        let kinds = vec![
            2, 3, 3, // unk, bos, eos
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // normal pieces
            1, 1, 1, 1, // byte tokens
        ];
        let scores = vec![
            0.0, 0.0, 0.0, -1.0, -179.0,   // ▁O
            -1292.0,  // ▁On
            -8779.0,  // ▁Once
            -29617.0, // n
            -17349.0, // nc
            -87.0,    // ce
            -29613.0, // e
            -21913.0, // ▁hello
            -39.0,    // ▁h
            -36.0,    // el
            -158.0,   // lo
            -900.0,   // ▁he
            -3000.0,  // ▁hel
            -12000.0, // ▁hell
            -500.0,   // ello
            -2.0, -2.0, -2.0, // ▁upon, ▁a, ▁time
            0.0, 0.0, 0.0, 0.0,
        ];
        let mut v = Vocabulary::new(pieces, &kinds, scores).unwrap();
        v.set_bos(Some(1));
        v.set_eos(Some(2));
        v.set_unk(Some(0));
        v.set_add_bos(true);
        v
    }

    #[test]
    fn merges_reach_multi_character_pieces() {
        let v = test_vocab();
        let mut out = Vec::new();
        encode(&v, "Once", true, &mut out).unwrap();
        // The merge chain: ▁+O→▁O, ▁O+n→▁On, c+e→ce, ▁On+ce→▁Once.
        assert_eq!(out, vec![v.id_of("▁Once").unwrap()], "must reach ▁Once");
    }

    #[test]
    fn highest_score_merges_first() {
        let v = test_vocab();
        let mut out = Vec::new();
        encode(&v, "hello", true, &mut out).unwrap();
        // ▁h(-39) + el(-36) + lo(-158) chain up to ▁hello(-21913).
        assert_eq!(out, vec![v.id_of("▁hello").unwrap()]);
    }

    #[test]
    fn encodes_known_words() {
        let v = test_vocab();
        // "▓a" is directly reachable (adjacent characters); "▓hello" forms
        // through the merge chain ▓h → ▓he → ▓hel → ▓hell + ello.
        let mut out = Vec::new();
        encode(&v, "a hello", true, &mut out).unwrap();
        assert_eq!(
            out,
            vec![v.id_of("▁a").unwrap(), v.id_of("▁hello").unwrap()]
        );
    }

    #[test]
    fn unreachable_pieces_fall_back_to_bytes() {
        let v = test_vocab();
        // "▁upon" exists in the vocabulary but cannot be reached by pairwise
        // merges (no "▁u"/"up"/"pon" intermediates), so its characters fall
        // back to unk/byte tokens — exactly like llama.cpp.
        let mut out = Vec::new();
        encode(&v, "upon", true, &mut out).unwrap();
        assert!(
            out.contains(&0),
            "expected unk for unreachable chars: {out:?}"
        );
        assert!(!out.contains(&v.id_of("▁upon").unwrap()));
    }

    #[test]
    fn byte_fallback_for_missing_piece() {
        let v = test_vocab();
        let mut out = Vec::new();
        encode(&v, "A", true, &mut out).unwrap();
        assert_eq!(out, vec![v.id_of("▁").unwrap(), v.id_of("<0x41>").unwrap()]);
    }

    #[test]
    fn multibyte_byte_fallback() {
        let v = test_vocab();
        let mut out = Vec::new();
        encode(&v, "♥", true, &mut out).unwrap();
        // "♥" = E2 99 A5 after the dummy ▁.
        assert_eq!(
            out,
            vec![
                v.id_of("▁").unwrap(),
                v.id_of("<0xE2>").unwrap(),
                v.id_of("<0x99>").unwrap(),
                v.id_of("<0xA5>").unwrap()
            ]
        );
    }

    #[test]
    fn dummy_prefix_only_at_sequence_start() {
        let v = test_vocab();
        let mut out = Vec::new();
        encode(&v, "hello", false, &mut out).unwrap();
        // No dummy space: ▁hello cannot match; expect char-level fallbacks.
        assert!(!out.contains(&v.id_of("▁hello").unwrap()));
    }
}
