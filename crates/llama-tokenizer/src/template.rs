//! Chat template integration.
//!
//! GGUF files store `tokenizer.chat_template` as a Jinja template. llama.rs
//! implements rendering for the widely used template families (Llama-2
//! `[INST]`, Llama-3 header style, `ChatML`, Mistral, Gemma turn style) by
//! pattern-matching the stored template and rendering with the exact
//! conventions those families use.
//!
//! Templates that do not match a known family produce an explicit error
//! naming the supported options — never silently wrong prompts.

use llama_core::{LlamaError, Result, TokenId};
use std::fmt::Write as _;

use crate::vocab::Vocabulary;

/// A chat message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// `system`, `user` or `assistant`.
    pub role: String,
    /// Message text.
    pub content: String,
}

impl Message {
    /// Creates a message.
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// The recognized template families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Llama-2 / Mistral style:
    /// `[INST] <<sys>>...<</sys>> prompt [/INST] answer </s>`.
    Llama2,
    /// Llama-3 style with `<|start_header_id|>role<|end_header_id|>`.
    Llama3,
    /// `ChatML` (`<|im_start|>role\n...<|im_end|>\n`).
    ChatMl,
    /// Gemma turn style (`<start_of_turn>role\n...<end_of_turn>\n`).
    Gemma,
    /// No template: messages are joined with newlines.
    Plain,
}

impl ChatTemplate {
    /// Detects the template family from a GGUF `tokenizer.chat_template`
    /// string.
    pub fn detect(template: &str) -> Result<Self> {
        let t = template;
        if t.is_empty() {
            return Ok(Self::Plain);
        }
        if t.contains("<|im_start|>") {
            return Ok(Self::ChatMl);
        }
        if t.contains("<|start_header_id|>") {
            return Ok(Self::Llama3);
        }
        if t.contains("<start_of_turn>") {
            return Ok(Self::Gemma);
        }
        if t.contains("[INST]") {
            return Ok(Self::Llama2);
        }
        Err(LlamaError::UnsupportedTokenizer(format!(
            "the stored chat template uses a format llama.rs cannot render; \
             supported families: llama2 ([INST]), llama3, chatml, gemma; \
             the template begins with: {:?}",
            t.chars().take(120).collect::<String>()
        )))
    }

    /// Canonical template name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Llama2 => "llama2",
            Self::Llama3 => "llama3",
            Self::ChatMl => "chatml",
            Self::Gemma => "gemma",
            Self::Plain => "plain",
        }
    }

    /// Parses a template selection by name (CLI flag).
    pub fn from_name(name: &str) -> Result<Self> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "llama2" | "mistral" => Self::Llama2,
            "llama3" => Self::Llama3,
            "chatml" => Self::ChatMl,
            "gemma" => Self::Gemma,
            "plain" | "none" => Self::Plain,
            other => {
                return Err(LlamaError::InvalidArgument(format!(
                    "unknown chat template '{other}' (supported: llama2, llama3, chatml, gemma, plain)"
                )));
            }
        })
    }

    /// Renders a conversation into a prompt.
    #[must_use]
    pub fn render(
        &self,
        messages: &[Message],
        vocab: &Vocabulary,
        add_generation_prompt: bool,
    ) -> String {
        let bos = vocab
            .bos()
            .map(|id| vocab.piece(id).to_string())
            .unwrap_or_default();
        let eos = vocab
            .eos()
            .map(|id| vocab.piece(id).to_string())
            .unwrap_or_default();
        match self {
            Self::Plain => {
                let mut out = String::new();
                for (i, m) in messages.iter().enumerate() {
                    if i > 0 {
                        out.push('\n');
                    }
                    out.push_str(&m.content);
                }
                out
            }
            Self::Llama2 => {
                let mut out = String::new();
                let mut in_inst = false;
                for m in messages {
                    match m.role.as_str() {
                        "system" => {
                            let _ = write!(out, "[INST] <<sys>> {} <</sys>> ", m.content);
                            in_inst = true;
                        }
                        "user" => {
                            if !in_inst {
                                out.push_str("[INST] ");
                            }
                            out.push_str(&m.content);
                            out.push_str(" [/INST] ");
                            in_inst = false;
                        }
                        "assistant" => {
                            out.push_str(&m.content);
                            out.push_str(&eos);
                        }
                        _ => {}
                    }
                }
                if add_generation_prompt && !in_inst {
                    // The trailing [/INST] above already opens the assistant
                    // turn for llama2.
                }
                format!("{bos}{out}")
            }
            Self::Llama3 => {
                let mut out = String::new();
                out.push_str(&bos);
                for m in messages {
                    let _ = write!(
                        out,
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        m.role, m.content
                    );
                }
                if add_generation_prompt {
                    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                }
                out
            }
            Self::ChatMl => {
                let mut out = String::new();
                for m in messages {
                    let _ = write!(out, "<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content);
                }
                if add_generation_prompt {
                    out.push_str("<|im_start|>assistant\n");
                }
                out
            }
            Self::Gemma => {
                let mut out = String::new();
                out.push_str("<bos>");
                for m in messages {
                    if m.role == "system" {
                        // Gemma folds the system prompt into the first turn.
                        let _ = write!(out, "<start_of_turn>user\n{} ", m.content);
                    } else {
                        let _ = write!(out, "<start_of_turn>{}\n{}", m.role, m.content);
                    }
                    out.push_str("<end_of_turn>\n");
                }
                if add_generation_prompt {
                    out.push_str("<start_of_turn>model\n");
                }
                out
            }
        }
    }
}

/// Convenience: tokenizes the rendered prompt (helper for callers).
pub fn encode_prompt(
    tokenizer: &crate::Tokenizer,
    template: &ChatTemplate,
    messages: &[Message],
    add_generation_prompt: bool,
    parse_special: bool,
) -> Result<Vec<TokenId>> {
    let prompt = template.render(messages, tokenizer.vocab(), add_generation_prompt);
    tokenizer.encode(&prompt, true, parse_special)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA2_TEMPLATE: &str =
        "{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token }}[INST] ";
    const LLAMA3_TEMPLATE: &str = "{{- bos_token }}{%- for message in messages %}{{- '<|start_header_id|>' + message['role'] }}";
    const CHATML_TEMPLATE: &str =
        "{%- for message in messages %}{{- '<|im_start|>' + message['role'] }}";
    const GEMMA_TEMPLATE: &str =
        "{{ bos_token}}{% for message in messages %}{{'<start_of_turn>' + message['role']}}";

    fn vocab() -> Vocabulary {
        let pieces = vec!["<unk>".into(), "<s>".into(), "</s>".into(), "▁x".into()];
        let kinds = vec![2, 3, 3, 1];
        let mut v = Vocabulary::new(pieces, &kinds, vec![0.0; 4]).unwrap();
        v.set_bos(Some(1));
        v.set_eos(Some(2));
        v
    }

    #[test]
    fn detection() {
        assert_eq!(
            ChatTemplate::detect(LLAMA2_TEMPLATE).unwrap(),
            ChatTemplate::Llama2
        );
        assert_eq!(
            ChatTemplate::detect(LLAMA3_TEMPLATE).unwrap(),
            ChatTemplate::Llama3
        );
        assert_eq!(
            ChatTemplate::detect(CHATML_TEMPLATE).unwrap(),
            ChatTemplate::ChatMl
        );
        assert_eq!(
            ChatTemplate::detect(GEMMA_TEMPLATE).unwrap(),
            ChatTemplate::Gemma
        );
        assert_eq!(ChatTemplate::detect("").unwrap(), ChatTemplate::Plain);
        assert!(ChatTemplate::detect("some unknown jinja").is_err());
    }

    #[test]
    fn llama2_render() {
        let v = vocab();
        let msgs = vec![
            Message::new("user", "Hello"),
            Message::new("assistant", "Hi there"),
            Message::new("user", "Bye"),
        ];
        let out = ChatTemplate::Llama2.render(&msgs, &v, true);
        assert_eq!(
            out,
            "<s>[INST] Hello [/INST] Hi there</s>[INST] Bye [/INST] "
        );
    }

    #[test]
    fn llama3_render() {
        let v = vocab();
        let msgs = vec![
            Message::new("system", "You are helpful"),
            Message::new("user", "Hello"),
        ];
        let out = ChatTemplate::Llama3.render(&msgs, &v, true);
        assert_eq!(
            out,
            "<s><|start_header_id|>system<|end_header_id|>\n\nYou are helpful<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn chatml_render() {
        let v = vocab();
        let msgs = vec![Message::new("user", "Hello")];
        let out = ChatTemplate::ChatMl.render(&msgs, &v, true);
        assert_eq!(
            out,
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
