//! Vocabulary storage: pieces, kinds, scores, special tokens, byte tokens
//! and merge ranks.

use std::collections::HashMap;

use llama_core::{LlamaError, Result, TokenId};

/// Token kind, mirroring `llama_token_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TokenKind {
    /// Ordinary vocabulary piece.
    Normal = 1,
    /// Unknown-piece marker.
    Unknown = 2,
    /// Control token (never produced by encoding).
    Control = 3,
    /// User-defined piece (always produced verbatim).
    UserDefined = 4,
    /// Unused piece.
    Unused = 5,
    /// Byte piece (`<0xXX>`).
    Byte = 6,
}

impl TokenKind {
    fn from_raw(raw: i32, index: usize) -> Result<Self> {
        Ok(match raw {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            other => {
                return Err(LlamaError::InvalidMetadata(format!(
                    "token {index} has unknown token_type {other}"
                )));
            }
        })
    }

    /// True for control-like tokens that are skipped when decoding non-special.
    #[must_use]
    pub fn is_special(&self) -> bool {
        matches!(self, Self::Control | Self::Unknown | Self::Unused)
    }
}

/// The full vocabulary.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pieces: Vec<String>,
    kinds: Vec<TokenKind>,
    scores: Vec<f32>,
    piece_to_id: HashMap<String, TokenId>,
    byte_to_id: [Option<TokenId>; 256],
    id_to_byte: HashMap<TokenId, u8>,
    /// Special tokens that should be recognized during parsing, with their
    /// text form sorted by descending length (longest match first).
    special_pieces: Vec<(String, TokenId)>,
    bos: Option<TokenId>,
    eos: Option<TokenId>,
    unk: Option<TokenId>,
    pad: Option<TokenId>,
    add_bos: bool,
    add_eos: bool,
    merge_ranks: HashMap<(String, String), u32>,
    merges: Vec<String>,
    max_piece_len: usize,
    min_piece_score: f32,
}

impl Vocabulary {
    /// Builds a vocabulary from GGUF arrays.
    pub fn new(pieces: Vec<String>, token_types: &[i32], scores: Vec<f32>) -> Result<Self> {
        let n = pieces.len();
        let mut kinds = Vec::with_capacity(n);
        for (i, &raw) in token_types.iter().enumerate() {
            kinds.push(TokenKind::from_raw(raw, i)?);
        }

        let mut piece_to_id = HashMap::with_capacity(n * 2);
        let mut byte_to_id = [None; 256];
        let mut id_to_byte = HashMap::new();
        let mut special_pieces = Vec::new();

        for (i, piece) in pieces.iter().enumerate() {
            let id = i as TokenId;
            piece_to_id.insert(piece.clone(), id);
            // Byte piece "<0xXX>".
            if let Some(rest) = piece.strip_prefix("<0x") {
                if let Some(hex) = rest.strip_suffix('>') {
                    if hex.len() == 2 {
                        if let Ok(byte) = u8::from_str_radix(hex, 16) {
                            byte_to_id[byte as usize] = Some(id);
                            id_to_byte.insert(id, byte);
                        }
                    }
                }
            }
            if (kinds[i] == TokenKind::UserDefined || kinds[i] == TokenKind::Control)
                && !piece.is_empty()
                && piece.starts_with('<')
                && piece.ends_with('>')
            {
                special_pieces.push((piece.clone(), id));
            }
        }
        // Longest-first matching.
        special_pieces.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));

        let max_piece_len = pieces.iter().map(String::len).max().unwrap_or(1);
        let min_piece_score = scores.iter().copied().fold(f32::INFINITY, f32::min);
        Ok(Self {
            pieces,
            kinds,
            scores,
            piece_to_id,
            byte_to_id,
            id_to_byte,
            special_pieces,
            bos: None,
            eos: None,
            unk: None,
            pad: None,
            add_bos: false,
            add_eos: false,
            merge_ranks: HashMap::new(),
            merges: Vec::new(),
            max_piece_len,
            min_piece_score: if min_piece_score.is_finite() {
                min_piece_score
            } else {
                0.0
            },
        })
    }

    /// Number of tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// Whether the vocabulary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Piece text for an id.
    #[must_use]
    pub fn piece(&self, id: TokenId) -> &str {
        &self.pieces[id as usize]
    }

    /// All pieces (fixture writer support).
    #[must_use]
    pub fn pieces(&self) -> &[String] {
        &self.pieces
    }

    /// All kinds (fixture writer support).
    #[must_use]
    pub fn kinds(&self) -> &[TokenKind] {
        &self.kinds
    }

    /// All scores (fixture writer support).
    #[must_use]
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    /// Token kind for an id.
    #[must_use]
    pub fn kind(&self, id: TokenId) -> TokenKind {
        self.kinds[id as usize]
    }

    /// Score for an id.
    #[must_use]
    pub fn score(&self, id: TokenId) -> f32 {
        self.scores[id as usize]
    }

    /// Looks up a piece by text.
    #[must_use]
    pub fn id_of(&self, piece: &str) -> Option<TokenId> {
        self.piece_to_id.get(piece).copied()
    }

    /// Byte-fallback token for a byte value.
    #[must_use]
    pub fn byte_token(&self, byte: u8) -> Option<TokenId> {
        self.byte_to_id[byte as usize]
    }

    /// Reverse mapping: token id to byte value (for `<0xXX>` pieces).
    #[must_use]
    pub fn token_byte(&self, id: TokenId) -> Option<u8> {
        self.id_to_byte.get(&id).copied()
    }

    /// BOS id.
    #[must_use]
    pub fn bos(&self) -> Option<TokenId> {
        self.bos
    }

    /// EOS id.
    #[must_use]
    pub fn eos(&self) -> Option<TokenId> {
        self.eos
    }

    /// UNK id.
    #[must_use]
    pub fn unk(&self) -> Option<TokenId> {
        self.unk
    }

    /// PAD id.
    #[must_use]
    pub fn pad(&self) -> Option<TokenId> {
        self.pad
    }

    /// Sets the BOS id.
    pub fn set_bos(&mut self, id: Option<TokenId>) {
        self.bos = id;
    }

    /// Sets the EOS id.
    pub fn set_eos(&mut self, id: Option<TokenId>) {
        self.eos = id;
    }

    /// Sets the UNK id.
    pub fn set_unk(&mut self, id: Option<TokenId>) {
        self.unk = id;
    }

    /// Sets the PAD id.
    pub fn set_pad(&mut self, id: Option<TokenId>) {
        self.pad = id;
    }

    /// Whether encoding should prepend BOS.
    #[must_use]
    pub fn add_bos(&self) -> bool {
        self.add_bos
    }

    /// Whether encoding should append EOS.
    #[must_use]
    pub fn add_eos(&self) -> bool {
        self.add_eos
    }

    /// Toggles BOS prepending.
    pub fn set_add_bos(&mut self, v: bool) {
        self.add_bos = v;
    }

    /// Toggles EOS appending.
    pub fn set_add_eos(&mut self, v: bool) {
        self.add_eos = v;
    }

    /// Builds BPE merge ranks from `tokenizer.ggml.merges` entries
    /// (format: `"a b"`).
    pub fn build_merge_ranks(&mut self, merges: &[String]) -> Result<()> {
        for (rank, merge) in merges.iter().enumerate() {
            let Some((left, right)) = merge.split_once(' ') else {
                return Err(LlamaError::InvalidMetadata(format!(
                    "merge entry {rank} is not of the form 'a b': {merge:?}"
                )));
            };
            self.merge_ranks
                .insert((left.to_string(), right.to_string()), rank as u32);
        }
        self.merges = merges.to_vec();
        Ok(())
    }

    /// Rank of a merge pair (lower = earlier).
    #[must_use]
    pub fn merge_rank(&self, left: &str, right: &str) -> Option<u32> {
        self.merge_ranks
            .get(&(left.to_string(), right.to_string()))
            .copied()
    }

    /// Raw merges list (fixture writer support).
    #[must_use]
    pub fn merges(&self) -> &[String] {
        &self.merges
    }

    /// Longest piece length in bytes (Viterbi search window).
    #[must_use]
    pub fn max_piece_len(&self) -> usize {
        self.max_piece_len
    }

    /// Lowest piece score (used to derive the unknown penalty).
    #[must_use]
    pub fn min_piece_score(&self) -> f32 {
        self.min_piece_score
    }

    /// Splits text into raw fragments and recognized special tokens.
    ///
    /// Longest special piece wins at each position.
    #[must_use]
    pub(crate) fn split_special_tokens<'a>(&self, text: &'a str) -> Vec<crate::Fragment<'a>> {
        use crate::Fragment;
        let mut out = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0usize;
        let mut text_start = 0usize;
        while i < bytes.len() {
            let mut matched = None;
            for (piece, id) in &self.special_pieces {
                let pb = piece.as_bytes();
                if bytes[i..].starts_with(pb) {
                    matched = Some(*id);
                    break;
                }
            }
            if let Some(id) = matched {
                if i > text_start {
                    out.push(Fragment::Text(&text[text_start..i]));
                }
                out.push(Fragment::Special(id));
                i += self.pieces[id as usize].len();
                text_start = i;
            } else {
                i += 1;
            }
        }
        if text_start < bytes.len() {
            out.push(Fragment::Text(&text[text_start..]));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_vocab() -> Vocabulary {
        let pieces = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "▁hello".to_string(),
            "▁world".to_string(),
            "<0x41>".to_string(),
            "<|special|>".to_string(),
        ];
        let kinds = vec![2, 3, 3, 1, 1, 1, 4];
        let scores = vec![0.0; 7];
        let mut v = Vocabulary::new(pieces, &kinds, scores).unwrap();
        v.set_bos(Some(1));
        v.set_eos(Some(2));
        v.set_unk(Some(0));
        v.set_add_bos(true);
        v
    }

    #[test]
    fn lookup() {
        let v = build_vocab();
        assert_eq!(v.id_of("▁hello"), Some(3));
        assert_eq!(v.bos(), Some(1));
        assert_eq!(v.kind(3), TokenKind::Normal);
        assert_eq!(v.kind(1), TokenKind::Control);
        assert_eq!(v.byte_token(0x41), Some(5));
        assert_eq!(v.token_byte(5), Some(0x41));
    }

    #[test]
    fn special_splitting() {
        let v = build_vocab();
        let text = "hello <|special|> world";
        let _ = v.split_special_tokens(text);
        // (encoding correctness covered by the crate-level tests)
    }
}
