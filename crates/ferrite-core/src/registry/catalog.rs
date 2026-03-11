//! Model Catalog
//!
//! Built-in model definitions and catalog loading.

use super::spec::*;
use std::collections::HashMap;

/// Model catalog containing all available model specs
#[derive(Debug, Default)]
pub struct Catalog {
    models: HashMap<String, ModelSpec>,
}

impl Catalog {
    /// Create a new catalog with built-in models
    pub fn new() -> Self {
        let mut catalog = Self::default();
        catalog.register_builtins();
        catalog
    }

    /// Register the built-in model catalog
    fn register_builtins(&mut self) {
        // ═══════════════════════════════════════════════════════════════
        // MISTRAL FAMILY
        // ═══════════════════════════════════════════════════════════════

        self.register(ModelSpec {
            name: "mistral-7b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF".into(),
                file: Some("mistral-7b-instruct-v0.2.Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Mistral,
            context_length: 32768,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "mistralai/Mistral-7B-Instruct-v0.2".into(),
            },
            description: "Mistral 7B Instruct v0.2 (4-bit quantized)".into(),
            size: "7B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "mistral-7b-q8".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF".into(),
                file: Some("mistral-7b-instruct-v0.2.Q8_0.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Mistral,
            context_length: 32768,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "mistralai/Mistral-7B-Instruct-v0.2".into(),
            },
            description: "Mistral 7B Instruct v0.2 (8-bit quantized)".into(),
            size: "7B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "openhermes-7b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF".into(),
                file: Some("openhermes-2.5-mistral-7b.Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::ChatML,
            context_length: 32768,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "mistralai/Mistral-7B-Instruct-v0.2".into(),
            },
            description: "OpenHermes 2.5 Mistral 7B (4-bit quantized)".into(),
            size: "7B".into(),
            requires_auth: false,
        });

        // ═══════════════════════════════════════════════════════════════
        // LLAMA 3 FAMILY
        // ═══════════════════════════════════════════════════════════════

        self.register(ModelSpec {
            name: "llama3-8b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF".into(),
                file: Some("Meta-Llama-3-8B-Instruct.Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Llama3,
            context_length: 8192,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "meta-llama/Meta-Llama-3-8B-Instruct".into(),
            },
            description: "Llama 3 8B Instruct (4-bit quantized)".into(),
            size: "8B".into(),
            requires_auth: true,
        });

        self.register(ModelSpec {
            name: "llama3.1-8b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF".into(),
                file: Some("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Llama3,
            context_length: 131072,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "meta-llama/Llama-3.1-8B-Instruct".into(),
            },
            description: "Llama 3.1 8B Instruct (4-bit quantized)".into(),
            size: "8B".into(),
            requires_auth: true,
        });

        // ═══════════════════════════════════════════════════════════════
        // QWEN FAMILY
        // ═══════════════════════════════════════════════════════════════

        self.register(ModelSpec {
            name: "qwen2-0.5b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "Qwen/Qwen2-0.5B-Instruct-GGUF".into(),
                file: Some("qwen2-0_5b-instruct-q4_k_m.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::ChatML,
            context_length: 32768,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "Qwen/Qwen2-0.5B-Instruct".into(),
            },
            description: "Qwen2 0.5B Instruct (4-bit quantized)".into(),
            size: "0.5B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "qwen2-1.5b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "Qwen/Qwen2-1.5B-Instruct-GGUF".into(),
                file: Some("qwen2-1_5b-instruct-q4_k_m.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::ChatML,
            context_length: 32768,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "Qwen/Qwen2-1.5B-Instruct".into(),
            },
            description: "Qwen2 1.5B Instruct (4-bit quantized)".into(),
            size: "1.5B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "qwen2-7b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "Qwen/Qwen2-7B-Instruct-GGUF".into(),
                file: Some("qwen2-7b-instruct-q4_k_m.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::ChatML,
            context_length: 131072,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "Qwen/Qwen2-7B-Instruct".into(),
            },
            description: "Qwen2 7B Instruct (4-bit quantized)".into(),
            size: "7B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "qwen2.5-7b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "Qwen/Qwen2.5-7B-Instruct-GGUF".into(),
                file: Some("qwen2.5-7b-instruct-q4_k_m.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::ChatML,
            context_length: 131072,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "Qwen/Qwen2.5-7B-Instruct".into(),
            },
            description: "Qwen2.5 7B Instruct (4-bit quantized)".into(),
            size: "7B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "qwen3-8b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "Qwen/Qwen3-8B-GGUF".into(),
                file: Some("Qwen3-8B-Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::ChatML,
            context_length: 32768,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "Qwen/Qwen3-8B".into(),
            },
            description: "Qwen3 8B Instruct (4-bit quantized)".into(),
            size: "8B".into(),
            requires_auth: false,
        });

        // ═══════════════════════════════════════════════════════════════
        // PHI FAMILY
        // ═══════════════════════════════════════════════════════════════

        self.register(ModelSpec {
            name: "phi3-mini-q4".into(),
            family: ModelFamily::Phi,
            source: ModelSource::HuggingFace {
                repo: "microsoft/Phi-3-mini-4k-instruct-gguf".into(),
                file: Some("Phi-3-mini-4k-instruct-q4.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Phi3,
            context_length: 4096,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "microsoft/Phi-3-mini-4k-instruct".into(),
            },
            description: "Phi-3 Mini 4K Instruct (4-bit quantized)".into(),
            size: "3.8B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "phi3-medium-q4".into(),
            family: ModelFamily::Phi,
            source: ModelSource::HuggingFace {
                repo: "microsoft/Phi-3-medium-4k-instruct-gguf".into(),
                file: Some("Phi-3-medium-4k-instruct-Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Phi3,
            context_length: 4096,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "microsoft/Phi-3-medium-4k-instruct".into(),
            },
            description: "Phi-3 Medium 4K Instruct (4-bit quantized)".into(),
            size: "14B".into(),
            requires_auth: false,
        });

        // ═══════════════════════════════════════════════════════════════
        // GEMMA FAMILY
        // ═══════════════════════════════════════════════════════════════

        self.register(ModelSpec {
            name: "gemma-2b-q4".into(),
            family: ModelFamily::Gemma,
            source: ModelSource::HuggingFace {
                repo: "google/gemma-2b-it-GGUF".into(),
                file: Some("gemma-2b-it.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Gemma,
            context_length: 8192,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "google/gemma-2b-it".into(),
            },
            description: "Gemma 2B Instruct".into(),
            size: "2B".into(),
            requires_auth: true,
        });

        self.register(ModelSpec {
            name: "gemma2-9b-q4".into(),
            family: ModelFamily::Gemma,
            source: ModelSource::HuggingFace {
                repo: "bartowski/gemma-2-9b-it-GGUF".into(),
                file: Some("gemma-2-9b-it-Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Gemma,
            context_length: 8192,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "google/gemma-2-9b-it".into(),
            },
            description: "Gemma 2 9B Instruct (4-bit quantized)".into(),
            size: "9B".into(),
            requires_auth: true,
        });

        // ═══════════════════════════════════════════════════════════════
        // EMBEDDED/ORIN-FRIENDLY MODELS (fit in 4GB TLSF pool)
        // ═══════════════════════════════════════════════════════════════

        self.register(ModelSpec {
            name: "tinyllama-1.1b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".into(),
                file: Some("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Zephyr,
            context_length: 2048,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0".into(),
            },
            description: "TinyLlama 1.1B Chat (4-bit quantized)".into(),
            size: "1.1B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "openllama-3b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "TheBloke/open_llama_3b_v2-GGUF".into(),
                file: Some("open_llama_3b_v2.Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Raw,
            context_length: 2048,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "openlm-research/open_llama_3b_v2".into(),
            },
            description: "OpenLlama 3B v2 (4-bit quantized, Orin-friendly)".into(),
            size: "3B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "stablelm-zephyr-3b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "TheBloke/stablelm-zephyr-3b-GGUF".into(),
                file: Some("stablelm-zephyr-3b.Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Zephyr,
            context_length: 4096,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "stabilityai/stablelm-zephyr-3b".into(),
            },
            description: "StableLM Zephyr 3B (4-bit quantized, Orin-friendly)".into(),
            size: "3B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "rocket-3b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "TheBloke/rocket-3B-GGUF".into(),
                file: Some("rocket-3b.Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Zephyr,
            context_length: 4096,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "pansophic/rocket-3B".into(),
            },
            description: "Rocket 3B (4-bit quantized, Orin-friendly)".into(),
            size: "3B".into(),
            requires_auth: false,
        });

        // ═══════════════════════════════════════════════════════════════
        // CODE MODELS
        // ═══════════════════════════════════════════════════════════════

        self.register(ModelSpec {
            name: "codellama-7b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "TheBloke/CodeLlama-7B-Instruct-GGUF".into(),
                file: Some("codellama-7b-instruct.Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Llama2,
            context_length: 16384,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "codellama/CodeLlama-7b-Instruct-hf".into(),
            },
            description: "Code Llama 7B Instruct (4-bit quantized)".into(),
            size: "7B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "deepseek-coder-6.7b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "TheBloke/deepseek-coder-6.7B-instruct-GGUF".into(),
                file: Some("deepseek-coder-6.7b-instruct.Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::ChatML,
            context_length: 16384,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "deepseek-ai/deepseek-coder-6.7b-instruct".into(),
            },
            description: "DeepSeek Coder 6.7B Instruct (4-bit quantized)".into(),
            size: "6.7B".into(),
            requires_auth: false,
        });

        self.register(ModelSpec {
            name: "starcoder2-7b-q4".into(),
            family: ModelFamily::Llama,
            source: ModelSource::HuggingFace {
                repo: "second-state/StarCoder2-7B-GGUF".into(),
                file: Some("starcoder2-7b-Q4_K_M.gguf".into()),
                revision: None,
            },
            format: WeightFormat::GGUF,
            chat_template: ChatTemplate::Raw,
            context_length: 16384,
            tokenizer: TokenizerSource::HuggingFace {
                repo: "bigcode/starcoder2-7b".into(),
            },
            description: "StarCoder2 7B (4-bit quantized)".into(),
            size: "7B".into(),
            requires_auth: false,
        });
    }

    /// Register a model spec
    pub fn register(&mut self, spec: ModelSpec) {
        self.models.insert(spec.name.clone(), spec);
    }

    /// Get a model spec by name
    pub fn get(&self, name: &str) -> Option<&ModelSpec> {
        // Try exact match first
        if let Some(spec) = self.models.get(name) {
            return Some(spec);
        }

        // Try case-insensitive match
        let name_lower = name.to_lowercase();
        self.models.values().find(|s| s.name.to_lowercase() == name_lower)
    }

    /// List all available models
    pub fn list(&self) -> Vec<&ModelSpec> {
        let mut models: Vec<_> = self.models.values().collect();
        models.sort_by(|a, b| a.name.cmp(&b.name));
        models
    }

    /// Search models by name pattern
    pub fn search(&self, pattern: &str) -> Vec<&ModelSpec> {
        let pattern_lower = pattern.to_lowercase();
        self.models
            .values()
            .filter(|s| {
                s.name.to_lowercase().contains(&pattern_lower)
                    || s.description.to_lowercase().contains(&pattern_lower)
            })
            .collect()
    }

    /// Get models by family
    pub fn by_family(&self, family: ModelFamily) -> Vec<&ModelSpec> {
        self.models.values().filter(|s| s.family == family).collect()
    }

    /// Get model count
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Check if catalog is empty
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_has_models() {
        let catalog = Catalog::new();
        assert!(catalog.len() > 0);
    }

    #[test]
    fn test_get_model() {
        let catalog = Catalog::new();
        let spec = catalog.get("mistral-7b-q4");
        assert!(spec.is_some());
        assert_eq!(spec.unwrap().family, ModelFamily::Llama);
    }

    #[test]
    fn test_search() {
        let catalog = Catalog::new();
        let results = catalog.search("qwen");
        assert!(results.len() >= 2);
    }
}
