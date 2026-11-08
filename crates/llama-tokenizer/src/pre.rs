//! Pre-tokenization scanners for byte-level BPE models.
//!
//! The scanners are hand-written finite-state approximations of the HF
//! tokenizer regexes used by the `tokenizer.ggml.pre` types llama.rs
//! supports. Letter/number/whitespace classes use the Rust `char` predicates
//! (`is_alphabetic`, `is_numeric`, `is_whitespace`), which match the Unicode
//! `Alphabetic`/`N`/`White_Space` properties; the underlying regexes use the
//! slightly narrower `\p{L}`. The difference only affects a few combining
//! marks, which is documented in `docs/model-support.md`.

use llama_core::{LlamaError, Result};

/// Supported pre-tokenizer kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizerKind {
    /// `SentencePiece`: no regex pre-split (only used for SPM vocabularies).
    Spm,
    /// GPT-2 / `tokenizer.ggml.pre = "default"`.
    Gpt2,
    /// Llama 3 style (`pre = "llama3"`): 3-digit runs, contractions
    /// case-insensitive, trailing newlines attach to punctuation runs.
    Llama3,
    /// Qwen2 style (`pre = "qwen2"`): single digits.
    Qwen2,
}

impl PreTokenizerKind {
    /// Maps the GGUF `tokenizer.ggml.pre` string.
    pub fn from_name(name: &str) -> Result<Self> {
        Ok(match name {
            "default" | "gpt2" | "palm" | "jina" | "scbond" | "poro" | "chatglm" | "megrez" => {
                Self::Gpt2
            }
            "llama3" | "llama4" | "dbrx" | "smaug" | "jais2" | "gemma3" | "bailing" => Self::Llama3,
            "qwen2" | "hunyuan" | "solar" | "qwen35" => Self::Qwen2,
            other => {
                return Err(LlamaError::UnsupportedTokenizer(format!(
                    "pre-tokenizer '{other}' is not supported (supported: default, llama3, qwen2)"
                )));
            }
        })
    }

    /// Canonical name written back into GGUF metadata.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Spm => "spm",
            Self::Gpt2 => "default",
            Self::Llama3 => "llama3",
            Self::Qwen2 => "qwen2",
        }
    }
}

/// Case-insensitive contraction set: 's 't 're 've 'm 'll 'd.
const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_number(c: char) -> bool {
    c.is_numeric()
}

fn is_space(c: char) -> bool {
    c.is_whitespace()
}

fn is_other(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}

fn matches_contraction(rest: &[char]) -> Option<usize> {
    for c in CONTRACTIONS {
        let chars: Vec<char> = c.chars().collect();
        if rest.len() >= chars.len() {
            let matches = chars
                .iter()
                .zip(&rest[..chars.len()])
                .all(|(a, b)| a.eq_ignore_ascii_case(b));
            if matches {
                return Some(chars.len());
            }
        }
    }
    None
}

/// Splits text into pre-tokens for the given pre-tokenizer kind.
#[must_use]
pub fn split(kind: PreTokenizerKind, text: &str) -> Vec<String> {
    match kind {
        PreTokenizerKind::Spm => vec![text.to_string()],
        PreTokenizerKind::Gpt2 => split_gpt2_like(text, None, false),
        PreTokenizerKind::Llama3 => split_gpt2_like(text, Some(3), true),
        PreTokenizerKind::Qwen2 => split_gpt2_like(text, Some(1), false),
    }
}

/// GPT-2-family scanner.
///
/// Implements the pattern
/// `(?:'s|'t|'re|'ve|'m|'ll|'d) | ?\p{L}+ | ?\p{N}{1,max} | ?[^\s\p{L}\p{N}]+[\r\n]*? | \s+(?!\S) | \s+`
/// with `max = None` meaning unbounded digits and `attach_newlines`
/// controlling the llama3 trailing-`\r\n` behaviour.
fn push_run(out: &mut Vec<String>, chars: &[char], start: usize, end: usize) {
    if end > start {
        out.push(chars[start..end].iter().collect());
    }
}

/// Consumes the whitespace run starting at `start` plus the word, number or
/// punctuation run that follows it, following the GPT-2 rules: a multi-space
/// run keeps its final space as the prefix of the next piece, and a lone
/// space joins the following run. Returns the index after the region.
fn split_space_run(
    chars: &[char],
    start: usize,
    max_digits: Option<usize>,
    attach_newlines: bool,
    out: &mut Vec<String>,
) -> usize {
    let n = chars.len();
    let mut i = start;
    let mut j = i;
    while j < n && is_space(chars[j]) {
        j += 1;
    }
    if j == n {
        // Run reaches the end: emit it whole.
        push_run(out, chars, i, j);
        return j;
    }
    if j - i > 1 {
        // Emit all but the final whitespace char; the final one will join
        // the next token (space prefix) or stand alone.
        push_run(out, chars, i, j - 1);
        i = j - 1;
    }
    // Now a single whitespace char at i, followed by non-space.
    if chars[i] == ' ' {
        let next = chars[i + 1];
        if is_letter(next) {
            let mut j = i + 1;
            while j < n && is_letter(chars[j]) {
                j += 1;
            }
            push_run(out, chars, i, j);
            return j;
        }
        if is_number(next) {
            let mut j = i + 1;
            let mut count = 0usize;
            while j < n && is_number(chars[j]) && max_digits.is_none_or(|m| count < m) {
                j += 1;
                count += 1;
            }
            push_run(out, chars, i, j);
            return j;
        }
        if is_other(next) {
            let mut j = i + 1;
            while j < n && is_other(chars[j]) {
                j += 1;
            }
            if attach_newlines {
                while j < n && (chars[j] == '\r' || chars[j] == '\n') {
                    j += 1;
                }
            }
            push_run(out, chars, i, j);
            return j;
        }
        // next is a space — impossible (the run collapsed above).
        push_run(out, chars, i, i + 1);
        return i + 1;
    }
    // Tab/newline: single token.
    push_run(out, chars, i, i + 1);
    i + 1
}

fn split_gpt2_like(text: &str, max_digits: Option<usize>, attach_newlines: bool) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let n = chars.len();
    let mut i = 0usize;

    while i < n {
        let c = chars[i];

        // Contraction (case-insensitive for llama3/qwen2, exact for gpt2 —
        // we accept the case-insensitive form for all, matching HF
        // (?i:...) behaviour).
        if c == '\'' {
            if let Some(len) = matches_contraction(&chars[i..]) {
                push_run(&mut out, &chars, i, i + len);
                i += len;
                continue;
            }
        }

        if is_space(c) {
            i = split_space_run(&chars, i, max_digits, attach_newlines, &mut out);
            continue;
        }

        if is_letter(c) {
            let mut j = i;
            while j < n && is_letter(chars[j]) {
                j += 1;
            }
            push_run(&mut out, &chars, i, j);
            i = j;
        } else if is_number(c) {
            let mut j = i;
            let mut count = 0usize;
            while j < n && is_number(chars[j]) && max_digits.is_none_or(|m| count < m) {
                j += 1;
                count += 1;
            }
            push_run(&mut out, &chars, i, j);
            i = j;
        } else {
            // Other run.
            let mut j = i;
            while j < n && is_other(chars[j]) {
                j += 1;
            }
            if attach_newlines {
                while j < n && (chars[j] == '\r' || chars[j] == '\n') {
                    j += 1;
                }
            }
            push_run(&mut out, &chars, i, j);
            i = j;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(kind: PreTokenizerKind, text: &str) -> Vec<String> {
        split(kind, text)
    }

    #[test]
    fn gpt2_basic_words() {
        let words = run(PreTokenizerKind::Gpt2, "Hello world");
        assert_eq!(words, vec!["Hello", " world"]);
    }

    #[test]
    fn gpt2_numbers_and_punct() {
        let words = run(PreTokenizerKind::Gpt2, "abc123 def45!");
        assert_eq!(words, vec!["abc", "123", " def", "45", "!"]);
    }

    #[test]
    fn gpt2_multiple_spaces() {
        let words = run(PreTokenizerKind::Gpt2, "a  b   c");
        assert_eq!(words, vec!["a", " ", " b", "  ", " c"]);
    }

    #[test]
    fn gpt2_trailing_whitespace() {
        let words = run(PreTokenizerKind::Gpt2, "hello  ");
        assert_eq!(words, vec!["hello", "  "]);
    }

    #[test]
    fn gpt2_contractions() {
        let words = run(PreTokenizerKind::Gpt2, "don't we're I'll");
        assert_eq!(words, vec!["don", "'t", " we", "'re", " I", "'ll"]);
    }

    #[test]
    fn gpt2_newline_handling() {
        let words = run(PreTokenizerKind::Gpt2, "a\nb");
        assert_eq!(words, vec!["a", "\n", "b"]);
        let words = run(PreTokenizerKind::Gpt2, "a \nb");
        assert_eq!(words, vec!["a", " ", "\n", "b"]);
    }

    #[test]
    fn llama3_digit_groups() {
        let words = run(PreTokenizerKind::Llama3, "12345678");
        assert_eq!(words, vec!["123", "456", "78"]);
    }

    #[test]
    fn qwen2_single_digits() {
        let words = run(PreTokenizerKind::Qwen2, "12345");
        assert_eq!(words, vec!["1", "2", "3", "4", "5"]);
    }

    #[test]
    fn llama3_newlines_attach_to_punct() {
        let words = run(PreTokenizerKind::Llama3, "hi!\n\nnext");
        assert_eq!(words, vec!["hi", "!\n\n", "next"]);
    }

    #[test]
    fn unicode_letters() {
        let words = run(PreTokenizerKind::Gpt2, "héllo wörld");
        assert_eq!(words, vec!["héllo", " wörld"]);
    }

    #[test]
    fn empty_and_whitespace_only() {
        assert_eq!(
            run(PreTokenizerKind::Gpt2, ""),
            [] as [std::string::String; 0]
        );
        // A whitespace run that reaches the end of the text is one token.
        assert_eq!(run(PreTokenizerKind::Gpt2, "   "), vec!["   "]);
        assert_eq!(run(PreTokenizerKind::Gpt2, "  \t"), vec!["  \t"]);
        // A whitespace run followed by text keeps its last char for the
        // next token's optional space prefix.
        assert_eq!(run(PreTokenizerKind::Gpt2, "  \tx"), vec!["  ", "\t", "x"]);
    }
}
