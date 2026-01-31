//! Model Specification Types
//!
//! Declarative model definitions that describe how to load and configure models.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Complete specification for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Unique model identifier (e.g., "mistral-7b-q4")
    pub name: String,

    /// Architecture family
    pub family: ModelFamily,

    /// Where to get the model weights
    pub source: ModelSource,

    /// Weight format
    pub format: WeightFormat,

    /// Chat template style
    pub chat_template: ChatTemplate,

    /// Maximum context length
    pub context_length: usize,

    /// Tokenizer source (usually same repo or specific)
    #[serde(default)]
    pub tokenizer: TokenizerSource,

    /// Human-readable description
    #[serde(default)]
    pub description: String,

    /// Model size category for display
    #[serde(default)]
    pub size: String,

    /// Whether auth token is required
    #[serde(default)]
    pub requires_auth: bool,
}

/// Model architecture family
///
/// Models in the same family share the same forward pass implementation.
/// Many models are Llama-compatible (Mistral, Qwen, Yi, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFamily {
    /// Llama architecture (also: Mistral, Qwen, Yi, Vicuna, CodeLlama, etc.)
    Llama,
    /// Phi architecture (Phi-1, Phi-2, Phi-3)
    Phi,
    /// Gemma architecture (Gemma, Gemma2)
    Gemma,
    /// Mamba state-space models
    Mamba,
    /// StableLM architecture
    StableLM,
    /// Falcon architecture
    Falcon,
    /// MPT architecture
    MPT,
}

impl ModelFamily {
    /// Get the GGUF architecture string for this family
    pub fn gguf_arch(&self) -> &'static [&'static str] {
        match self {
            Self::Llama => &["llama", "mistral", "qwen", "qwen2", "yi", "internlm", "internlm2"],
            Self::Phi => &["phi", "phi2", "phi3"],
            Self::Gemma => &["gemma", "gemma2"],
            Self::Mamba => &["mamba"],
            Self::StableLM => &["stablelm"],
            Self::Falcon => &["falcon"],
            Self::MPT => &["mpt"],
        }
    }

    /// Detect family from GGUF architecture string
    pub fn from_gguf_arch(arch: &str) -> Option<Self> {
        let arch_lower = arch.to_lowercase();

        for family in [Self::Llama, Self::Phi, Self::Gemma, Self::Mamba,
                       Self::StableLM, Self::Falcon, Self::MPT] {
            if family.gguf_arch().iter().any(|a| arch_lower.contains(a)) {
                return Some(family);
            }
        }
        None
    }
}

/// Where to get model weights
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSource {
    /// HuggingFace Hub
    HuggingFace {
        repo: String,
        #[serde(default)]
        file: Option<String>,
        #[serde(default)]
        revision: Option<String>,
    },
    /// Local file path
    Local {
        path: PathBuf,
    },
    /// Direct URL
    Url {
        url: String,
    },
}

impl ModelSource {
    /// Create a HuggingFace source
    pub fn hf(repo: impl Into<String>, file: impl Into<String>) -> Self {
        Self::HuggingFace {
            repo: repo.into(),
            file: Some(file.into()),
            revision: None,
        }
    }

    /// Create a local file source
    pub fn local(path: impl Into<PathBuf>) -> Self {
        Self::Local { path: path.into() }
    }
}

/// Weight format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum WeightFormat {
    /// GGUF quantized format (llama.cpp compatible)
    #[default]
    GGUF,
    /// SafeTensors format
    SafeTensors,
    /// PyTorch .bin format
    PyTorch,
}

/// Chat template format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ChatTemplate {
    /// Mistral format: [INST] user [/INST] assistant
    #[default]
    Mistral,
    /// Llama 2 format: [INST] <<SYS>> system <</SYS>> user [/INST]
    Llama2,
    /// Llama 3 format: <|begin_of_text|><|start_header_id|>...
    Llama3,
    /// ChatML format: <|im_start|>role\ncontent<|im_end|>
    ChatML,
    /// Phi-3 format
    Phi3,
    /// Gemma format
    Gemma,
    /// Alpaca format
    Alpaca,
    /// Vicuna format
    Vicuna,
    /// Zephyr format
    Zephyr,
    /// Raw (no template, just concatenate)
    Raw,
}

impl ChatTemplate {
    /// Format a conversation using this template
    pub fn format(&self, messages: &[ChatMessage]) -> String {
        match self {
            Self::Mistral => self.format_mistral(messages),
            Self::Llama2 => self.format_llama2(messages),
            Self::Llama3 => self.format_llama3(messages),
            Self::ChatML => self.format_chatml(messages),
            Self::Phi3 => self.format_phi3(messages),
            Self::Gemma => self.format_gemma(messages),
            Self::Alpaca => self.format_alpaca(messages),
            Self::Vicuna => self.format_vicuna(messages),
            Self::Zephyr => self.format_zephyr(messages),
            Self::Raw => self.format_raw(messages),
        }
    }

    fn format_mistral(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();
        for msg in messages {
            match msg.role {
                Role::System => {
                    result.push_str(&format!("[INST] {}\n", msg.content));
                }
                Role::User => {
                    if result.is_empty() {
                        result.push_str(&format!("[INST] {} [/INST]", msg.content));
                    } else {
                        result.push_str(&format!("[INST] {} [/INST]", msg.content));
                    }
                }
                Role::Assistant => {
                    result.push_str(&format!("{}</s>", msg.content));
                }
            }
        }
        result
    }

    fn format_llama2(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();
        let mut system_msg = String::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    system_msg = format!("<<SYS>>\n{}\n<</SYS>>\n\n", msg.content);
                }
                Role::User => {
                    if result.is_empty() && !system_msg.is_empty() {
                        result.push_str(&format!("<s>[INST] {}{} [/INST]", system_msg, msg.content));
                        system_msg.clear();
                    } else {
                        result.push_str(&format!("<s>[INST] {} [/INST]", msg.content));
                    }
                }
                Role::Assistant => {
                    result.push_str(&format!(" {} </s>", msg.content));
                }
            }
        }
        result
    }

    fn format_llama3(&self, messages: &[ChatMessage]) -> String {
        let mut result = "<|begin_of_text|>".to_string();
        for msg in messages {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            result.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                role, msg.content
            ));
        }
        result.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        result
    }

    fn format_chatml(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();
        for msg in messages {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            result.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, msg.content));
        }
        result.push_str("<|im_start|>assistant\n");
        result
    }

    fn format_phi3(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();
        for msg in messages {
            match msg.role {
                Role::System => {
                    result.push_str(&format!("<|system|>\n{}<|end|>\n", msg.content));
                }
                Role::User => {
                    result.push_str(&format!("<|user|>\n{}<|end|>\n", msg.content));
                }
                Role::Assistant => {
                    result.push_str(&format!("<|assistant|>\n{}<|end|>\n", msg.content));
                }
            }
        }
        result.push_str("<|assistant|>\n");
        result
    }

    fn format_gemma(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();
        for msg in messages {
            match msg.role {
                Role::System | Role::User => {
                    result.push_str(&format!("<start_of_turn>user\n{}<end_of_turn>\n", msg.content));
                }
                Role::Assistant => {
                    result.push_str(&format!("<start_of_turn>model\n{}<end_of_turn>\n", msg.content));
                }
            }
        }
        result.push_str("<start_of_turn>model\n");
        result
    }

    fn format_alpaca(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();
        for msg in messages {
            match msg.role {
                Role::System => {
                    result.push_str(&format!("### System:\n{}\n\n", msg.content));
                }
                Role::User => {
                    result.push_str(&format!("### Instruction:\n{}\n\n", msg.content));
                }
                Role::Assistant => {
                    result.push_str(&format!("### Response:\n{}\n\n", msg.content));
                }
            }
        }
        result.push_str("### Response:\n");
        result
    }

    fn format_vicuna(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();
        for msg in messages {
            match msg.role {
                Role::System => {
                    result.push_str(&format!("{}\n\n", msg.content));
                }
                Role::User => {
                    result.push_str(&format!("USER: {}\n", msg.content));
                }
                Role::Assistant => {
                    result.push_str(&format!("ASSISTANT: {}\n", msg.content));
                }
            }
        }
        result.push_str("ASSISTANT:");
        result
    }

    fn format_zephyr(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();
        for msg in messages {
            match msg.role {
                Role::System => {
                    result.push_str(&format!("<|system|>\n{}</s>\n", msg.content));
                }
                Role::User => {
                    result.push_str(&format!("<|user|>\n{}</s>\n", msg.content));
                }
                Role::Assistant => {
                    result.push_str(&format!("<|assistant|>\n{}</s>\n", msg.content));
                }
            }
        }
        result.push_str("<|assistant|>\n");
        result
    }

    fn format_raw(&self, messages: &[ChatMessage]) -> String {
        messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n")
    }
}

/// Where to get the tokenizer
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TokenizerSource {
    /// Same repo as the model
    #[default]
    SameAsModel,
    /// Different HuggingFace repo
    HuggingFace {
        repo: String,
    },
    /// Local path
    Local {
        path: PathBuf,
    },
}

/// Chat message for template formatting
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

/// Message role
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Model info for listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub family: String,
    pub size: String,
    pub format: String,
    pub description: String,
}

impl From<&ModelSpec> for ModelInfo {
    fn from(spec: &ModelSpec) -> Self {
        Self {
            name: spec.name.clone(),
            family: format!("{:?}", spec.family),
            size: spec.size.clone(),
            format: format!("{:?}", spec.format),
            description: spec.description.clone(),
        }
    }
}
