//! Type Adapter Layer - Bridge between WIT types and Ferrite engine
//!
//! This is the TRAIT SIGNATURE ENGINE that connects WASM to native Ferrite inference.

use candle_core::{quantized::gguf_file, Device, Tensor};
use std::path::PathBuf;
use std::sync::Arc;

use crate::ferrite::inference::inference::GenerationConfig as WitGenConfig;

/// Convert WIT GenerationConfig to ferrite's GenerationConfig
pub fn wit_to_ferrite_config(wit_config: &WitGenConfig) -> ferrite::GenerationConfig {
    ferrite::GenerationConfig {
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

/// Quantized model wrapper
struct QuantizedMistral {
    model: candle_transformers::models::quantized_llama::ModelWeights,
    device: Device,
}

impl QuantizedMistral {
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

impl ferrite::InferenceModel for QuantizedMistral {
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

/// Model adapter - bridges WIT interface to ferrite engine
pub struct ModelAdapter {
    pub name: String,
    session: ferrite::ChatSession<QuantizedMistral>,
}

impl ModelAdapter {
    pub fn new(model_name: String, auth_token: Option<String>) -> Result<Self, String> {
        tracing::info!("🔄 Initializing model: {}", model_name);

        // Map model name
        let (model_id, filename) = match model_name.as_str() {
            "mistral-7b-q4" | "mistral-7b" | "mistral" => (
                "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            ),
            _ => return Err(format!("Unknown model: {}. Try 'mistral-7b-q4'", model_name)),
        };

        tracing::info!("📦 Model ID: {}", model_id);

        // Get device
        let device = Device::cuda_if_available(0)
            .map_err(|e| format!("Device error: {}", e))?;

        tracing::info!("🖥️  Device: {:?}", device);

        // Download GGUF (blocking)
        tracing::info!("⬇️  Downloading model...");
        let gguf_path = Self::download_gguf_sync(model_id, filename, auth_token.clone())
            .map_err(|e| format!("Download failed: {}", e))?;

        tracing::info!("✓ Model file: {}", gguf_path.display());

        // Load tokenizer (blocking)
        tracing::info!("🔤 Loading tokenizer...");
        let tokenizer = Self::load_tokenizer_sync("mistralai/Mistral-7B-Instruct-v0.2", auth_token)
            .map_err(|e| format!("Tokenizer failed: {}", e))?;

        tracing::info!("✓ Tokenizer ready");

        // Load model
        tracing::info!("🧠 Loading quantized model...");
        let model = QuantizedMistral::new(&gguf_path, device)
            .map_err(|e| format!("Model load failed: {}", e))?;

        tracing::info!("✓ Model loaded!");

        // Create session with generation config that will be updated per request
        let config = ferrite::ChatSessionConfig::mistral()
            .with_context_length(32768)
            .with_generation(ferrite::GenerationConfig::default());

        let session = ferrite::ChatSession::new(
            model,
            Arc::new(tokenizer),
            Some("You are a helpful AI assistant."),
            config,
        )
        .map_err(|e| format!("Session creation failed: {}", e))?;

        tracing::info!("✅ Ready for inference!");

        Ok(Self {
            name: model_name,
            session,
        })
    }

    fn download_gguf_sync(model_id: &str, filename: &str, token: Option<String>) -> Result<PathBuf, Box<dyn std::error::Error>> {
        if let Some(ref t) = token {
            tracing::info!("🔑 Using provided authentication token (length: {})", t.len());
        } else {
            tracing::info!("🔓 No authentication token provided");
        }

        let mut api_builder = hf_hub::api::sync::ApiBuilder::new();
        if let Some(t) = token {
            api_builder = api_builder.with_token(Some(t));
        }
        
        let api = api_builder.build()?;
        let repo = api.model(model_id.to_string());

        tracing::info!("📂 Checking hf-hub cache...");
        let gguf_path = repo.get(filename)?;

        tracing::info!("✓ Model located at: {}", gguf_path.display());
        Ok(gguf_path)
    }

    fn load_tokenizer_sync(model_id: &str, token: Option<String>) -> Result<ferrite::Tokenizer, Box<dyn std::error::Error>> {
        if let Some(ref t) = token {
            tracing::info!("🔑 Using provided authentication token for tokenizer (length: {})", t.len());
        } else {
            tracing::info!("🔓 No authentication token provided for tokenizer");
        }

        let mut api_builder = hf_hub::api::sync::ApiBuilder::new();
        if let Some(t) = token {
            api_builder = api_builder.with_token(Some(t));
        }
        
        let api = api_builder.build()?;
        let repo = api.model(model_id.to_string());

        // Download tokenizer files
        let tokenizer_file = repo.get("tokenizer.json")?;
        let _config_file = repo.get("tokenizer_config.json").ok();

        // Get the directory (should be the cache dir for this model)
        let model_dir = tokenizer_file.parent()
            .ok_or("No parent directory")?
            .to_path_buf();

        Ok(ferrite::Tokenizer::from_dir(&model_dir)?)
    }

    pub fn generate(&mut self, prompt: &str, _config: &ferrite::GenerationConfig) -> Result<String, String> {
        tracing::debug!("🤖 Generating for: {}", prompt);

        // Note: ChatSession uses the config from initialization
        // In a more complete impl, we'd recreate the session or make config mutable

        match self.session.user_turn(prompt) {
            Ok(response) => Ok(response),
            Err(e) => Err(format!("Generation failed: {}", e)),
        }
    }

    pub fn generate_stream(&mut self, prompt: &str, _config: &ferrite::GenerationConfig)
        -> Result<Vec<String>, String>
    {
        // For prototype, call generate and split into words
        let response = self.generate(prompt, _config)?;
        let tokens: Vec<String> = response
            .split_whitespace()
            .map(|s| s.to_string() + " ")
            .collect();
        Ok(tokens)
    }
}

/// Tokenizer adapter
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
