// Model registry and metadata

use crate::config::ModelFamily;

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: &'static str,
    pub family: ModelFamily,
    pub name: &'static str,
    pub size: &'static str,
    pub quantized: bool,
    pub description: &'static str,
}

/// All available models
pub const MODELS: &[ModelInfo] = &[
    // Llama family
    ModelInfo {
        id: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        family: ModelFamily::Llama,
        name: "TinyLlama 1.1B Chat",
        size: "1.1B",
        quantized: false,
        description: "Small, fast Llama-based chat model",
    },

    // Mistral family
    ModelInfo {
        id: "mistralai/Mistral-7B-Instruct-v0.2",
        family: ModelFamily::Mistral,
        name: "Mistral 7B Instruct",
        size: "7B",
        quantized: false,
        description: "High-quality instruction-tuned model",
    },
    ModelInfo {
        id: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        family: ModelFamily::Mistral,
        name: "Mistral 7B Q4",
        size: "7B (4-bit)",
        quantized: true,
        description: "Quantized Mistral, ~4GB VRAM",
    },

    // Qwen family
    ModelInfo {
        id: "Qwen/Qwen2-0.5B-Instruct",
        family: ModelFamily::Qwen,
        name: "Qwen2 0.5B",
        size: "0.5B",
        quantized: false,
        description: "Tiny but capable multilingual model",
    },
    ModelInfo {
        id: "Qwen/Qwen2-1.5B-Instruct",
        family: ModelFamily::Qwen,
        name: "Qwen2 1.5B",
        size: "1.5B",
        quantized: false,
        description: "Small multilingual model",
    },

    // Gemma family
    ModelInfo {
        id: "google/gemma-2b-it",
        family: ModelFamily::Gemma,
        name: "Gemma 2B",
        size: "2B",
        quantized: false,
        description: "Google's efficient instruction model (requires license)",
    },

    // Phi family
    ModelInfo {
        id: "microsoft/phi-2",
        family: ModelFamily::Phi,
        name: "Phi-2",
        size: "2.7B",
        quantized: false,
        description: "Strong reasoning and code capabilities",
    },
];

impl ModelInfo {
    /// Find a model by ID or name
    pub fn find(query: &str) -> Option<&'static ModelInfo> {
        let query_lower = query.to_lowercase();
        MODELS.iter().find(|m| {
            m.id.to_lowercase() == query_lower
                || m.name.to_lowercase().contains(&query_lower)
        })
    }

    /// List all models for a family
    pub fn for_family(family: ModelFamily) -> Vec<&'static ModelInfo> {
        MODELS.iter().filter(|m| m.family == family).collect()
    }

    /// List all available models
    pub fn all() -> &'static [ModelInfo] {
        MODELS
    }
}
