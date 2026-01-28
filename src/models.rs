//! Model configuration and registry
//!
//! Pre-configured model settings for popular LLMs.

use serde::{Deserialize, Serialize};

/// Model family
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFamily {
    Llama,
    Mistral,
    Qwen,
    Gemma,
    Phi,
}

impl ModelFamily {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "llama" | "tinyllama" => Some(Self::Llama),
            "mistral" => Some(Self::Mistral),
            "qwen" | "qwen2" => Some(Self::Qwen),
            "gemma" => Some(Self::Gemma),
            "phi" | "phi2" | "phi-2" => Some(Self::Phi),
            _ => None,
        }
    }

    pub fn chat_format(&self) -> ChatFormat {
        match self {
            Self::Llama => ChatFormat::Llama,
            Self::Mistral => ChatFormat::Mistral,
            Self::Qwen => ChatFormat::ChatML,
            Self::Gemma => ChatFormat::Gemma,
            Self::Phi => ChatFormat::Phi,
        }
    }
}

/// Chat formatting style
#[derive(Debug, Clone, Copy)]
pub enum ChatFormat {
    /// Llama 2/3 style: [INST] ... [/INST]
    Llama,
    /// Mistral style: [INST] ... [/INST]
    Mistral,
    /// ChatML style: <|im_start|>role\n...<|im_end|>
    ChatML,
    /// Gemma style: <start_of_turn>role\n...<end_of_turn>
    Gemma,
    /// Phi style: Instruct: ...\nOutput:
    Phi,
}

impl ChatFormat {
    /// Format a simple user message (fallback when no chat template)
    pub fn format_simple(&self, user_input: &str) -> String {
        match self {
            ChatFormat::Llama | ChatFormat::Mistral => {
                format!("[INST] {} [/INST]", user_input)
            }
            ChatFormat::ChatML => {
                format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    user_input
                )
            }
            ChatFormat::Gemma => {
                format!(
                    "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                    user_input
                )
            }
            ChatFormat::Phi => {
                format!("Instruct: {}\nOutput:", user_input)
            }
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_id: String,
    pub family: ModelFamily,
    pub name: String,
    pub parameters: String,
    pub context_length: usize,
    pub quantized: bool,
    pub requires_auth: bool,
}

impl ModelConfig {
    /// Get pre-configured models
    pub fn registry() -> Vec<ModelConfig> {
        vec![
            ModelConfig {
                model_id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".into(),
                family: ModelFamily::Llama,
                name: "TinyLlama 1.1B".into(),
                parameters: "1.1B".into(),
                context_length: 2048,
                quantized: false,
                requires_auth: false,
            },
            ModelConfig {
                model_id: "mistralai/Mistral-7B-Instruct-v0.2".into(),
                family: ModelFamily::Mistral,
                name: "Mistral 7B Instruct".into(),
                parameters: "7B".into(),
                context_length: 32768,
                quantized: false,
                requires_auth: false,
            },
            ModelConfig {
                model_id: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF".into(),
                family: ModelFamily::Mistral,
                name: "Mistral 7B Q4".into(),
                parameters: "7B (4-bit)".into(),
                context_length: 32768,
                quantized: true,
                requires_auth: false,
            },
            ModelConfig {
                model_id: "Qwen/Qwen2-0.5B-Instruct".into(),
                family: ModelFamily::Qwen,
                name: "Qwen2 0.5B".into(),
                parameters: "0.5B".into(),
                context_length: 32768,
                quantized: false,
                requires_auth: false,
            },
            ModelConfig {
                model_id: "google/gemma-2b-it".into(),
                family: ModelFamily::Gemma,
                name: "Gemma 2B".into(),
                parameters: "2B".into(),
                context_length: 8192,
                quantized: false,
                requires_auth: true,
            },
            ModelConfig {
                model_id: "microsoft/phi-2".into(),
                family: ModelFamily::Phi,
                name: "Phi-2".into(),
                parameters: "2.7B".into(),
                context_length: 2048,
                quantized: false,
                requires_auth: false,
            },
        ]
    }

    /// Find model by ID or name
    pub fn find(query: &str) -> Option<ModelConfig> {
        let query_lower = query.to_lowercase();
        Self::registry()
            .into_iter()
            .find(|m| {
                m.model_id.to_lowercase().contains(&query_lower)
                    || m.name.to_lowercase().contains(&query_lower)
            })
    }
}
