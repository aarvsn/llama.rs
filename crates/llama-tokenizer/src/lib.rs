//! # llama-tokenizer
//!
//! Tokenizer subsystem driven entirely by GGUF metadata.
//!
//! * [`TokenizerKind::Spm`] — unigram `SentencePiece` (Llama/Mistral style)
//!   with the ▁ space escape, score-priority merging and byte fallback.
//! * [`TokenizerKind::Bpe`] — byte-level BPE (Qwen/llama3 style) with
//!   GPT-2/llama3/qwen2 pre-tokenization.
//!
//! Unsupported tokenizer models (WPM, UGM, RWKV, ...) produce explicit
//! [`LlamaError::UnsupportedTokenizer`] errors — never silently wrong output.

pub mod bpe;
pub mod detok;
pub mod pre;
pub mod spm;
pub mod template;
pub mod vocab;

pub use detok::Detokenizer;
pub use template::{ChatTemplate, Message};
pub use vocab::{TokenKind, Vocabulary};

use llama_core::{LlamaError, Result, TokenId};
use llama_gguf::{GgufMetadata, GgufWriter};

/// Which tokenizer algorithm to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    /// `SentencePiece` unigram (Llama family, `tokenizer.ggml.model = "llama"`).
    Spm,
    /// Byte-level BPE (`tokenizer.ggml.model = "gpt2"`).
    Bpe,
}

/// A ready-to-use tokenizer.
pub struct Tokenizer {
    kind: TokenizerKind,
    vocab: Vocabulary,
    pre: pre::PreTokenizerKind,
}

impl Tokenizer {
    /// Builds a tokenizer from a parsed GGUF metadata table.
    pub fn from_gguf(meta: &GgufMetadata) -> Result<Self> {
        let model = meta.get_string("tokenizer.ggml.model").unwrap_or("");
        let kind = match model {
            "llama" => TokenizerKind::Spm,
            "gpt2" | "bpe" => TokenizerKind::Bpe,
            other => {
                return Err(LlamaError::UnsupportedTokenizer(format!(
                    "tokenizer model '{other}' is not supported (supported: llama [SPM], gpt2 [BPE])"
                )));
            }
        };

        let tokens = meta.get_string_vec("tokenizer.ggml.tokens")?;
        let token_types = meta.get_i32_vec("tokenizer.ggml.token_type")?;
        if tokens.len() != token_types.len() {
            return Err(LlamaError::InvalidMetadata(format!(
                "tokenizer.ggml.tokens has {} entries but token_type has {}",
                tokens.len(),
                token_types.len()
            )));
        }
        let scores = match meta.get_f32_vec("tokenizer.ggml.scores") {
            Ok(s) if s.len() == tokens.len() => s,
            Ok(s) => {
                return Err(LlamaError::InvalidMetadata(format!(
                    "tokenizer.ggml.scores has {} entries, expected {}",
                    s.len(),
                    tokens.len()
                )));
            }
            Err(LlamaError::MissingMetadata(_)) => vec![0.0; tokens.len()],
            Err(e) => return Err(e),
        };

        let mut vocab = Vocabulary::new(tokens, &token_types, scores)?;

        // Special token ids.
        if let Ok(id) = meta.get_u64("tokenizer.ggml.bos_token_id") {
            vocab.set_bos(Some(id as TokenId));
        }
        if let Ok(id) = meta.get_u64("tokenizer.ggml.eos_token_id") {
            vocab.set_eos(Some(id as TokenId));
        }
        if let Ok(id) = meta.get_u64("tokenizer.ggml.unk_token_id") {
            vocab.set_unk(Some(id as TokenId));
        }
        if let Ok(id) = meta.get_u64("tokenizer.ggml.pad_token_id") {
            vocab.set_pad(Some(id as TokenId));
        }
        if let Ok(v) = meta.get_bool("tokenizer.ggml.add_bos_token") {
            vocab.set_add_bos(v);
        }
        if let Ok(v) = meta.get_bool("tokenizer.ggml.add_eos_token") {
            vocab.set_add_eos(v);
        }

        let pre_kind = if kind == TokenizerKind::Spm {
            pre::PreTokenizerKind::Spm
        } else {
            let pre_name = meta.get_string("tokenizer.ggml.pre").unwrap_or("default");
            pre::PreTokenizerKind::from_name(pre_name)?
        };

        let mut tokenizer = Self {
            kind,
            vocab,
            pre: pre_kind,
        };

        if kind == TokenizerKind::Bpe {
            let merges = meta.get_string_vec("tokenizer.ggml.merges")?;
            tokenizer.vocab.build_merge_ranks(&merges)?;
        }

        Ok(tokenizer)
    }

    /// The vocabulary.
    #[must_use]
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Mutable vocabulary access (used during construction).
    pub fn vocab_mut(&mut self) -> &mut Vocabulary {
        &mut self.vocab
    }

    /// The tokenizer kind.
    #[must_use]
    pub fn kind(&self) -> TokenizerKind {
        self.kind
    }

    /// Encodes text to token ids.
    ///
    /// * `add_special` — prepend BOS / append EOS according to the model
    ///   metadata.
    /// * `parse_special` — recognize special token strings (e.g. `<|im_start|>`)
    ///   inside the text instead of tokenizing them as plain text.
    pub fn encode(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<TokenId>> {
        let mut out = Vec::new();
        if text.is_empty() {
            if add_special && self.vocab.add_bos() {
                if let Some(bos) = self.vocab.bos() {
                    out.push(bos);
                }
            }
            if add_special && self.vocab.add_eos() {
                if let Some(eos) = self.vocab.eos() {
                    out.push(eos);
                }
            }
            return Ok(out);
        }

        let fragments = if parse_special {
            self.vocab.split_special_tokens(text)
        } else {
            vec![Fragment::Text(text)]
        };

        if add_special && self.vocab.add_bos() {
            if let Some(bos) = self.vocab.bos() {
                out.push(bos);
            }
        }

        let mut first_fragment = true;
        for fragment in fragments {
            match fragment {
                Fragment::Text(t) => {
                    match self.kind {
                        TokenizerKind::Spm => {
                            spm::encode(&self.vocab, t, first_fragment, &mut out)?;
                        }
                        TokenizerKind::Bpe => {
                            let words = pre::split(self.pre, t);
                            for word in words {
                                bpe::encode_word(&self.vocab, &word, &mut out)?;
                            }
                        }
                    }
                    first_fragment = false;
                }
                Fragment::Special(id) => {
                    out.push(id);
                    first_fragment = true;
                }
            }
        }

        if add_special && self.vocab.add_eos() {
            if let Some(eos) = self.vocab.eos() {
                out.push(eos);
            }
        }
        Ok(out)
    }

    /// Decodes tokens to text.
    ///
    /// Control tokens are only rendered when `special` is true.
    #[must_use]
    pub fn decode(&self, tokens: &[TokenId], special: bool) -> String {
        let mut detok = Detokenizer::new(self.kind == TokenizerKind::Bpe);
        let mut out = String::new();
        for &t in tokens {
            out.push_str(&detok.push(&self.vocab, t, special));
        }
        out.push_str(&detok.flush());
        out
    }

    /// Renders a chat (message list) into a prompt string.
    #[must_use]
    pub fn apply_chat_template(
        &self,
        template: &template::ChatTemplate,
        messages: &[template::Message],
        add_generation_prompt: bool,
    ) -> String {
        template.render(messages, self.vocab(), add_generation_prompt)
    }

    /// Writes the standard tokenizer metadata keys into a GGUF writer
    /// (used by the fixture generators).
    ///
    /// # Panics
    ///
    /// Panics when a GGUF writer error occurs; fixture generation only.
    pub fn write_gguf_metadata(&self, w: &mut GgufWriter, chat_template: Option<&str>) {
        let v = &self.vocab;
        w.set_str(
            "tokenizer.ggml.model",
            match self.kind {
                TokenizerKind::Spm => "llama",
                TokenizerKind::Bpe => "gpt2",
            },
        );
        if self.kind == TokenizerKind::Bpe {
            w.set_str("tokenizer.ggml.pre", self.pre.name());
        }
        w.set_str_vec("tokenizer.ggml.tokens", v.pieces().to_vec());
        w.set_i32_vec(
            "tokenizer.ggml.token_type",
            v.kinds().iter().map(|k| *k as i32).collect(),
        );
        w.set_f32_vec("tokenizer.ggml.scores", v.scores().to_vec());
        if self.kind == TokenizerKind::Bpe {
            w.set_str_vec("tokenizer.ggml.merges", v.merges().to_vec());
        }
        if let Some(id) = v.bos() {
            w.set_u64("tokenizer.ggml.bos_token_id", u64::from(id));
        }
        if let Some(id) = v.eos() {
            w.set_u64("tokenizer.ggml.eos_token_id", u64::from(id));
        }
        if let Some(id) = v.unk() {
            w.set_u64("tokenizer.ggml.unk_token_id", u64::from(id));
        }
        if v.add_bos() {
            w.set_bool("tokenizer.ggml.add_bos_token", true);
        }
        if v.add_eos() {
            w.set_bool("tokenizer.ggml.add_eos_token", true);
        }
        if let Some(t) = chat_template {
            w.set_str("tokenizer.chat_template", t);
        }
    }
}

/// A fragment of input text: either raw text or a recognized special token.
pub(crate) enum Fragment<'a> {
    /// Raw text to tokenize.
    Text(&'a str),
    /// A special token that appears verbatim in the input.
    Special(TokenId),
}
