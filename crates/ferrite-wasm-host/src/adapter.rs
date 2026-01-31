//! Type Adapter Layer - Bridge between WIT types and Ferrite engine
//!
//! This is the TRAIT SIGNATURE ENGINE that connects WASM to native Ferrite inference.
//! Now uses the model registry for dynamic model loading.

use candle_core::{quantized::gguf_file, Device, Tensor};
use crate::bindings::WitGenConfig;
use ferrite_core::registry::{
    Catalog, ChatTemplate, DownloadedModel, ModelFamily, ModelLoader, ModelSpec,
};
use std::path::PathBuf;
use std::sync::Arc;

/// Convert WIT GenerationConfig to ferrite's GenerationConfig
pub fn wit_to_ferrite_config(wit_config: &WitGenConfig) -> ferrite_core::GenerationConfig {
    ferrite_core::GenerationConfig {
        max_tokens: wit_config.max_tokens as usize,
        temperature: wit_config.temperature as f64,
        top_p: wit_config.top_p as f64,
        top_k: wit_config.top_k as usize,
        min_p: 0.05,
        repetition_penalty: 1.0,
        stop_sequences: vec![],
        seed: wit_config.seed.unwrap_or(42),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUANTIZED MODEL IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Generic quantized Llama-family model (Llama, Mistral, Qwen, Yi, etc.)
struct QuantizedLlama {
    model: candle_transformers::models::quantized_llama::ModelWeights,
    device: Device,
}

impl QuantizedLlama {
    fn new(gguf_path: &PathBuf, device: Device) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(gguf_path)?;
        let gguf_content = gguf_file::Content::read(&mut file)?;
        let model = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
            gguf_content,
            &mut file,
            &device,
        )?;

        Ok(Self { model, device })
    }
}

impl ferrite_core::InferenceModel for QuantizedLlama {
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<Tensor, Box<dyn std::error::Error>> {
        if tokens.is_empty() {
            return Ok(Tensor::zeros((1, 32000), candle_core::DType::F32, &self.device)?);
        }

        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, pos)?;
        let logits = logits.squeeze(0)?;
        Ok(logits)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut pos = 0;
        for &token in tokens.iter().take(tokens.len().saturating_sub(1)) {
            let input = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
            let _ = self.model.forward(&input, pos)?;
            pos += 1;
        }

        if let Some(&last_token) = tokens.last() {
            let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            Ok(logits)
        } else {
            Ok(Tensor::zeros((1, 32000), candle_core::DType::F32, &self.device)?)
        }
    }

    fn clear_cache(&mut self) {
        // Quantized models don't have explicit cache clear
    }
}

/// Quantized Phi model
struct QuantizedPhi {
    model: candle_transformers::models::quantized_llama::ModelWeights,
    device: Device,
}

impl QuantizedPhi {
    fn new(gguf_path: &PathBuf, device: Device) -> Result<Self, Box<dyn std::error::Error>> {
        // Phi models use the same GGUF format as Llama
        let mut file = std::fs::File::open(gguf_path)?;
        let gguf_content = gguf_file::Content::read(&mut file)?;
        let model = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
            gguf_content,
            &mut file,
            &device,
        )?;

        Ok(Self { model, device })
    }
}

impl ferrite_core::InferenceModel for QuantizedPhi {
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<Tensor, Box<dyn std::error::Error>> {
        if tokens.is_empty() {
            return Ok(Tensor::zeros((1, 51200), candle_core::DType::F32, &self.device)?);
        }
        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, pos)?;
        Ok(logits.squeeze(0)?)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut pos = 0;
        for &token in tokens.iter().take(tokens.len().saturating_sub(1)) {
            let input = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
            let _ = self.model.forward(&input, pos)?;
            pos += 1;
        }
        if let Some(&last) = tokens.last() {
            let input = Tensor::new(&[last], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?;
            Ok(logits.squeeze(0)?)
        } else {
            Ok(Tensor::zeros((1, 51200), candle_core::DType::F32, &self.device)?)
        }
    }

    fn clear_cache(&mut self) {}
}

// ═══════════════════════════════════════════════════════════════════════════════
// DYNAMIC MODEL WRAPPER
// ═══════════════════════════════════════════════════════════════════════════════

/// Enum to hold different model types
enum DynamicModel {
    Llama(QuantizedLlama),
    Phi(QuantizedPhi),
    // Add more as needed: Gemma, Mamba, etc.
}

impl ferrite_core::InferenceModel for DynamicModel {
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<Tensor, Box<dyn std::error::Error>> {
        match self {
            Self::Llama(m) => m.forward(tokens, pos),
            Self::Phi(m) => m.forward(tokens, pos),
        }
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor, Box<dyn std::error::Error>> {
        match self {
            Self::Llama(m) => m.prefill(tokens),
            Self::Phi(m) => m.prefill(tokens),
        }
    }

    fn clear_cache(&mut self) {
        match self {
            Self::Llama(m) => m.clear_cache(),
            Self::Phi(m) => m.clear_cache(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MODEL ADAPTER (uses registry)
// ═══════════════════════════════════════════════════════════════════════════════

/// Model adapter - bridges WIT interface to ferrite engine via registry
pub struct ModelAdapter {
    pub name: String,
    pub spec: ModelSpec,
    session: ferrite_core::ChatSession<DynamicModel>,
}

impl ModelAdapter {
    /// Create a new model adapter using the registry
    pub fn new(model_name: String, auth_token: Option<String>) -> Result<Self, String> {
        tracing::info!("🔄 Initializing model: {}", model_name);

        // Create model loader with auth
        let loader = ModelLoader::new("./models")
            .with_auth(auth_token.clone());

        // Look up model in catalog
        let spec = loader.get_spec(&model_name)
            .ok_or_else(|| {
                let available = loader.catalog().list()
                    .iter()
                    .take(5)
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "Unknown model: '{}'. Available: {} (and {} more). Use 'list-models' to see all.",
                    model_name,
                    available,
                    loader.catalog().len().saturating_sub(5)
                )
            })?
            .clone();

        tracing::info!("📦 Found model: {} ({:?})", spec.name, spec.family);
        tracing::info!("📝 Description: {}", spec.description);

        // Download model
        let downloaded = loader.download_spec(&spec)
            .map_err(|e| format!("Download failed: {}", e))?;

        // Load model based on family
        Self::load_from_downloaded(downloaded, auth_token)
    }

    /// Load a GGUF file directly with auto-detection
    pub fn from_gguf(path: &std::path::Path, auth_token: Option<String>) -> Result<Self, String> {
        tracing::info!("🔄 Auto-loading GGUF: {}", path.display());

        let loader = ModelLoader::new("./models")
            .with_auth(auth_token.clone());

        let downloaded = loader.load_gguf_auto(path)
            .map_err(|e| format!("Auto-detection failed: {}", e))?;

        tracing::info!("✅ Detected: {:?} model", downloaded.family());

        Self::load_from_downloaded(downloaded, auth_token)
    }

    /// Load from a downloaded model
    fn load_from_downloaded(downloaded: DownloadedModel, auth_token: Option<String>) -> Result<Self, String> {
        let device = Device::cuda_if_available(0)
            .map_err(|e| format!("Device error: {}", e))?;

        tracing::info!("🖥️  Device: {:?}", device);

        // Load the model based on family
        tracing::info!("🧠 Loading {:?} model...", downloaded.family());

        let model: DynamicModel = match downloaded.family() {
            ModelFamily::Llama => {
                let m = QuantizedLlama::new(&downloaded.weights_path, device)
                    .map_err(|e| format!("Model load failed: {}", e))?;
                DynamicModel::Llama(m)
            }
            ModelFamily::Phi => {
                let m = QuantizedPhi::new(&downloaded.weights_path, device)
                    .map_err(|e| format!("Model load failed: {}", e))?;
                DynamicModel::Phi(m)
            }
            ModelFamily::Gemma => {
                // Gemma uses Llama-compatible GGUF loader
                let m = QuantizedLlama::new(&downloaded.weights_path, device)
                    .map_err(|e| format!("Model load failed: {}", e))?;
                DynamicModel::Llama(m)
            }
            _ => {
                return Err(format!("Architecture {:?} not yet implemented", downloaded.family()));
            }
        };

        tracing::info!("✓ Model loaded!");

        // Load tokenizer
        tracing::info!("🔤 Loading tokenizer...");
        let tokenizer = Self::load_tokenizer(&downloaded, auth_token)
            .map_err(|e| format!("Tokenizer failed: {}", e))?;

        tracing::info!("✓ Tokenizer ready");

        // Create session config based on chat template
        let config = Self::create_session_config(&downloaded.spec);

        let session = ferrite_core::ChatSession::new(
            model,
            Arc::new(tokenizer),
            Some("You are a helpful AI assistant."),
            config,
        )
        .map_err(|e| format!("Session creation failed: {}", e))?;

        tracing::info!("✅ Ready for inference!");

        Ok(Self {
            name: downloaded.spec.name.clone(),
            spec: downloaded.spec,
            session,
        })
    }

    /// Create session config from model spec
    fn create_session_config(spec: &ModelSpec) -> ferrite_core::ChatSessionConfig {
        let chat_format = match spec.chat_template {
            ChatTemplate::Mistral => ferrite_core::ChatFormat::Mistral,
            ChatTemplate::Llama2 | ChatTemplate::Llama3 => ferrite_core::ChatFormat::Llama,
            ChatTemplate::ChatML | ChatTemplate::Phi3 => ferrite_core::ChatFormat::ChatML,
            ChatTemplate::Gemma => ferrite_core::ChatFormat::Gemma,
            _ => ferrite_core::ChatFormat::Mistral,
        };

        ferrite_core::ChatSessionConfig::default()
            .with_context_length(spec.context_length)
            .with_chat_format(chat_format)
    }

    /// Load tokenizer from downloaded model
    fn load_tokenizer(
        downloaded: &DownloadedModel,
        auth_token: Option<String>,
    ) -> Result<ferrite_core::Tokenizer, Box<dyn std::error::Error>> {
        // Try to load from the tokenizer path first
        if downloaded.tokenizer_path.join("tokenizer.json").exists() {
            return Ok(ferrite_core::Tokenizer::from_dir(&downloaded.tokenizer_path)?);
        }

        // Fall back to downloading from HuggingFace
        let tokenizer_repo = match &downloaded.spec.tokenizer {
            ferrite_core::registry::TokenizerSource::HuggingFace { repo } => repo.clone(),
            ferrite_core::registry::TokenizerSource::SameAsModel => {
                match &downloaded.spec.source {
                    ferrite_core::registry::ModelSource::HuggingFace { repo, .. } => repo.clone(),
                    _ => return Err("Cannot determine tokenizer repo".into()),
                }
            }
            ferrite_core::registry::TokenizerSource::Local { path } => {
                return Ok(ferrite_core::Tokenizer::from_dir(path)?);
            }
        };

        tracing::info!("📥 Downloading tokenizer from {}", tokenizer_repo);

        let mut api_builder = hf_hub::api::sync::ApiBuilder::new();
        if let Some(t) = auth_token {
            api_builder = api_builder.with_token(Some(t));
        }
        let api = api_builder.build()?;
        let repo = api.model(tokenizer_repo);

        let tokenizer_file = repo.get("tokenizer.json")?;
        let _ = repo.get("tokenizer_config.json").ok();

        let model_dir = tokenizer_file.parent()
            .ok_or("No parent directory")?
            .to_path_buf();

        Ok(ferrite_core::Tokenizer::from_dir(&model_dir)?)
    }

    /// Generate text from a prompt
    pub fn generate(&mut self, prompt: &str, _config: &ferrite_core::GenerationConfig) -> Result<String, String> {
        tracing::debug!("🤖 Generating for: {}", prompt);

        match self.session.user_turn(prompt) {
            Ok(response) => Ok(response),
            Err(e) => Err(format!("Generation failed: {}", e)),
        }
    }

    /// Generate text as a stream of tokens
    pub fn generate_stream(&mut self, prompt: &str, config: &ferrite_core::GenerationConfig)
        -> Result<Vec<String>, String>
    {
        // For prototype, call generate and split into words
        let response = self.generate(prompt, config)?;
        let tokens: Vec<String> = response
            .split_whitespace()
            .map(|s| s.to_string() + " ")
            .collect();
        Ok(tokens)
    }

    /// List all available models
    pub fn list_available() -> Vec<ferrite_core::ModelInfo> {
        ModelLoader::new("./models").list_models()
    }

    /// Search for models
    pub fn search(query: &str) -> Vec<ferrite_core::ModelInfo> {
        ModelLoader::new("./models").search(query)
    }
}

/// Tokenizer adapter (for direct tokenization without full model)
pub struct TokenizerAdapter;

impl TokenizerAdapter {
    pub fn encode(text: &str) -> Vec<u32> {
        text.split_whitespace()
            .enumerate()
            .map(|(i, _)| (i + 1000) as u32)
            .collect()
    }

    pub fn decode(tokens: &[u32]) -> String {
        format!("[decoded {} tokens]", tokens.len())
    }

    pub fn decode_token(token: u32) -> String {
        format!("tok_{}", token)
    }
}
