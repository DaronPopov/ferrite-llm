//! Type Adapter Layer - Bridge between WIT types and Ferrite engine
//!
//! This is the TRAIT SIGNATURE ENGINE that connects WASM to native Ferrite inference.
//! Now uses the model registry for dynamic model loading.
use candle_core::{quantized::gguf_file, Device, Tensor};
use crate::bindings::WitGenConfig;
use ferrite_core::registry::{
    ChatTemplate, DownloadedModel, ModelFamily, ModelLoader, ModelSpec,
};
use mistralrs::{
    ChatCompletionChunkResponse, ChunkChoice, Delta, GgufModelBuilder, RequestBuilder,
    Response as MistralResponse, TextMessageRole, Usage,
};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendKind {
    Candle,
    MistralRs,
}

impl BackendKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Candle => "candle",
            Self::MistralRs => "mistralrs",
        }
    }
}

#[derive(Clone, Debug)]
struct ChatTurn {
    role: TextMessageRole,
    content: String,
}

struct MistralRsSession {
    model: Arc<Mutex<mistralrs::Model>>,
    runtime: Arc<tokio::runtime::Runtime>,
    history: Arc<Mutex<Vec<ChatTurn>>>,
}

impl MistralRsSession {
    fn new(downloaded: &DownloadedModel, force_cpu: bool) -> Result<Self, String> {
        let model_root = downloaded
            .weights_path
            .parent()
            .unwrap_or(&downloaded.weights_path)
            .to_string_lossy()
            .to_string();
        let model_file = downloaded
            .weights_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| "Model weights path is missing a file name".to_string())?
            .to_string();
        let tokenizer_json = downloaded.tokenizer_path.join("tokenizer.json");
        if !tokenizer_json.exists() {
            return Err(format!(
                "mistralrs backend requires tokenizer.json, missing at {}",
                tokenizer_json.display()
            ));
        }

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("Failed to create Tokio runtime: {e}"))?;

        let mut builder = GgufModelBuilder::new(model_root, vec![model_file])
            .with_tokenizer_json(tokenizer_json.display().to_string())
            .with_logging();

        let tokenizer_config = downloaded.tokenizer_path.join("tokenizer_config.json");
        if tokenizer_config.exists() {
            builder = builder.with_chat_template(tokenizer_config.display().to_string());
        }
        if force_cpu {
            builder = builder.with_force_cpu();
        }

        let model = runtime
            .block_on(builder.build())
            .map_err(|e| format!("mistralrs model load failed: {e}"))?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            runtime: Arc::new(runtime),
            history: Arc::new(Mutex::new(vec![ChatTurn {
                role: TextMessageRole::System,
                content: "You are a helpful AI assistant.".to_string(),
            }])),
        })
    }

    fn request_from_turns(
        turns: &[ChatTurn],
        config: &ferrite_core::GenerationConfig,
    ) -> RequestBuilder {
        let mut request = RequestBuilder::new();
        for turn in turns {
            request = request.add_message(turn.role.clone(), &turn.content);
        }

        request
            .set_sampler_max_len(config.max_tokens)
            .set_sampler_temperature(config.temperature)
            .set_sampler_topp(config.top_p)
            .set_sampler_topk(config.top_k)
            .set_sampler_minp(config.min_p as f64)
            .with_truncate_sequence(true)
    }

    fn generate(
        &mut self,
        prompt: &str,
        config: &ferrite_core::GenerationConfig,
    ) -> Result<String, String> {
        let mut turns = self
            .history
            .lock()
            .map_err(|_| "mistralrs history lock poisoned".to_string())?
            .clone();
        turns.push(ChatTurn {
            role: TextMessageRole::User,
            content: prompt.to_string(),
        });
        let request = Self::request_from_turns(&turns, config);

        let response = match self.runtime.block_on(
            self.model
                .lock()
                .map_err(|_| "mistralrs model lock poisoned".to_string())?
                .send_chat_request(request),
        ) {
            Ok(response) => response,
            Err(err) => return Err(format!("mistralrs generation failed: {err}")),
        };

        let content = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .ok_or_else(|| "mistralrs returned no assistant content".to_string())?;

        print!("{content}");
        print_usage_stats(response.usage);

        let mut history = self
            .history
            .lock()
            .map_err(|_| "mistralrs history lock poisoned".to_string())?;
        history.push(ChatTurn {
            role: TextMessageRole::User,
            content: prompt.to_string(),
        });
        history.push(ChatTurn {
            role: TextMessageRole::Assistant,
            content: content.clone(),
        });

        Ok(content)
    }

    fn start_generate_stream(
        &mut self,
        prompt: &str,
        config: &ferrite_core::GenerationConfig,
    ) -> Result<ActiveGeneration, String> {
        let mut turns = self
            .history
            .lock()
            .map_err(|_| "mistralrs history lock poisoned".to_string())?
            .clone();
        turns.push(ChatTurn {
            role: TextMessageRole::User,
            content: prompt.to_string(),
        });
        let request = Self::request_from_turns(&turns, config);
        let (tx, rx) = mpsc::channel();
        let runtime = Arc::clone(&self.runtime);
        let model = Arc::clone(&self.model);
        let history = Arc::clone(&self.history);
        let prompt_text = prompt.to_string();

        std::thread::spawn(move || {
            let mut response_text = String::new();
            let send_result: Result<(), String> = (|| {
                let model_guard = model
                    .lock()
                    .map_err(|_| "mistralrs model lock poisoned".to_string())?;
                let mut stream = runtime
                    .block_on(model_guard.stream_chat_request(request))
                    .map_err(|e| format!("mistralrs streaming generation failed: {e}"))?;

                loop {
                    let next = runtime.block_on(stream.next());
                    let Some(response) = next else {
                        break;
                    };

                    match response {
                        MistralResponse::Chunk(ChatCompletionChunkResponse { choices, .. }) => {
                            if let Some(ChunkChoice {
                                delta:
                                    Delta {
                                        content: Some(content),
                                        ..
                                    },
                                ..
                            }) = choices.first()
                            {
                                response_text.push_str(content);
                                tx.send(StreamEvent::Chunk(content.clone())).ok();
                            }
                        }
                        MistralResponse::ModelError(message, _) => {
                            return Err(message);
                        }
                        MistralResponse::CompletionDone(response) => {
                            if response_text.is_empty() {
                                if let Some(choice) = response.choices.first() {
                                    response_text = choice.text.clone();
                                }
                            }
                            tx.send(StreamEvent::Stats(response.usage.avg_compl_tok_per_sec))
                                .ok();
                        }
                        _ => {}
                    }
                }

                Ok(())
            })();

            match send_result {
                Ok(()) => {
                    if let Ok(mut turns) = history.lock() {
                        turns.push(ChatTurn {
                            role: TextMessageRole::User,
                            content: prompt_text,
                        });
                        turns.push(ChatTurn {
                            role: TextMessageRole::Assistant,
                            content: response_text,
                        });
                    }
                    tx.send(StreamEvent::Done).ok();
                }
                Err(err) => {
                    tx.send(StreamEvent::Error(err)).ok();
                }
            }
        });

        Ok(ActiveGeneration::MistralRs(MistralRsActiveGeneration { rx }))
    }
}

fn print_usage_stats(usage: Usage) {
    println!("\n[Stats] {:.1} tok/s", usage.avg_compl_tok_per_sec);
}

enum StreamEvent {
    Chunk(String),
    Stats(f32),
    Done,
    Error(String),
}

pub(crate) struct MistralRsActiveGeneration {
    rx: mpsc::Receiver<StreamEvent>,
}

impl MistralRsActiveGeneration {
    fn next_chunk(&mut self) -> Result<Option<String>, String> {
        match self.rx.recv() {
            Ok(StreamEvent::Chunk(chunk)) => Ok(Some(chunk)),
            Ok(StreamEvent::Stats(tps)) => Ok(Some(format!("\n[Stats] {:.1} tok/s\n", tps))),
            Ok(StreamEvent::Done) => Ok(None),
            Ok(StreamEvent::Error(err)) => Err(err),
            Err(_) => Ok(None),
        }
    }
}

pub(crate) enum BufferedGeneration {
    Chunks(VecDeque<String>),
}

impl BufferedGeneration {
    fn next_chunk(&mut self) -> Option<String> {
        match self {
            Self::Chunks(chunks) => chunks.pop_front(),
        }
    }
}

pub(crate) enum ActiveGeneration {
    Buffered(BufferedGeneration),
    MistralRs(MistralRsActiveGeneration),
}

impl ActiveGeneration {
    pub fn next_chunk(&mut self) -> Result<Option<String>, String> {
        match self {
            Self::Buffered(buffered) => Ok(buffered.next_chunk()),
            Self::MistralRs(stream) => stream.next_chunk(),
        }
    }
}

enum BackendSession {
    Candle(ferrite_core::ChatSession<DynamicModel>),
    MistralRs(MistralRsSession),
}

// ═══════════════════════════════════════════════════════════════════════════════
// MODEL ADAPTER (uses registry)
// ═══════════════════════════════════════════════════════════════════════════════

/// Model adapter - bridges WIT interface to ferrite engine via registry
pub struct ModelAdapter {
    pub name: String,
    pub spec: ModelSpec,
    tokenizer: Arc<ferrite_core::Tokenizer>,
    session: BackendSession,
}

impl ModelAdapter {
    /// Create a new model adapter using the registry
    pub fn new(
        model_name: String,
        model_cache: PathBuf,
        auth_token: Option<String>,
    ) -> Result<Self, String> {
        tracing::info!("🔄 Initializing model: {}", model_name);

        // Create model loader with auth
        let loader = ModelLoader::new(&model_cache)
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
    pub fn from_gguf(
        path: &Path,
        model_cache: PathBuf,
        auth_token: Option<String>,
    ) -> Result<Self, String> {
        tracing::info!("🔄 Auto-loading GGUF: {}", path.display());

        let loader = ModelLoader::new(&model_cache)
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
        let require_cuda = std::env::var("FERRITE_REQUIRE_CUDA")
            .map(|value| value != "0")
            .unwrap_or(false);

        let backend_kind = Self::select_backend(&downloaded);
        let device_name = match &device {
            Device::Cuda(_) => "cuda",
            Device::Cpu => "cpu",
            _ => "other",
        };

        match &device {
            Device::Cuda(_) => tracing::info!("Using CUDA device 0 for model execution"),
            Device::Cpu if require_cuda => {
                return Err(
                    "CUDA device unavailable or ferrite was built without the `cuda` feature".into(),
                );
            }
            Device::Cpu => tracing::warn!(
                "CUDA unavailable; falling back to CPU execution. Set FERRITE_REQUIRE_CUDA=1 to fail instead."
            ),
            _ => {}
        }

        tracing::info!(
            "Ferrite backend selection: backend={} device={} model={}",
            backend_kind.as_str(),
            device_name,
            downloaded.spec.name
        );
        tracing::info!("🖥️  Device: {:?}", device);

        // Load the model based on family
        tracing::info!("🧠 Loading {:?} model with {:?} backend...", downloaded.family(), backend_kind);

        // Load tokenizer
        tracing::info!("🔤 Loading tokenizer...");
        let tokenizer = Self::load_tokenizer(&downloaded, auth_token)
            .map_err(|e| format!("Tokenizer failed: {}", e))?;

        tracing::info!("✓ Tokenizer ready");

        // Create session config based on chat template
        let config = Self::create_session_config(&downloaded.spec);

        let tokenizer = Arc::new(tokenizer);
        let force_cpu = matches!(device, Device::Cpu);

        let session = match backend_kind {
            BackendKind::Candle => {
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
                        let m = QuantizedLlama::new(&downloaded.weights_path, device)
                            .map_err(|e| format!("Model load failed: {}", e))?;
                        DynamicModel::Llama(m)
                    }
                    _ => {
                        return Err(format!("Architecture {:?} not yet implemented", downloaded.family()));
                    }
                };

                BackendSession::Candle(
                    ferrite_core::ChatSession::new(
                        model,
                        Arc::clone(&tokenizer),
                        Some("You are a helpful AI assistant."),
                        config,
                    )
                    .map_err(|e| format!("Session creation failed: {}", e))?,
                )
            }
            BackendKind::MistralRs => BackendSession::MistralRs(
                MistralRsSession::new(&downloaded, force_cpu)?,
            ),
        };

        tracing::info!("✅ Ready for inference!");

        Ok(Self {
            name: downloaded.spec.name.clone(),
            spec: downloaded.spec,
            tokenizer,
            session,
        })
    }

    fn select_backend(downloaded: &DownloadedModel) -> BackendKind {
        let backend_env = std::env::var("FERRITE_BACKEND")
            .ok()
            .or_else(|| std::env::var("FERRITE_INFERENCE_BACKEND").ok());

        match backend_env {
            Some(value) if value.eq_ignore_ascii_case("mistralrs") => BackendKind::MistralRs,
            Some(value) if value.eq_ignore_ascii_case("candle") => BackendKind::Candle,
            _ => {
                if downloaded.spec.family == ModelFamily::Llama {
                    BackendKind::MistralRs
                } else {
                    BackendKind::Candle
                }
            }
        }
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
    pub fn generate(
        &mut self,
        prompt: &str,
        config: &ferrite_core::GenerationConfig,
    ) -> Result<String, String> {
        tracing::debug!("🤖 Generating for: {}", prompt);
        match &mut self.session {
            BackendSession::Candle(session) => {
                session.set_generation_config(config.clone());
                session
                    .user_turn(prompt)
                    .map_err(|e| format!("Generation failed: {}", e))
            }
            BackendSession::MistralRs(session) => session.generate(prompt, config),
        }
    }

    /// Generate text as a stream of tokens
    pub fn generate_stream(&mut self, prompt: &str, config: &ferrite_core::GenerationConfig)
        -> Result<Vec<String>, String>
    {
        match &mut self.session {
            BackendSession::Candle(session) => {
                session.set_generation_config(config.clone());

                let mut stream = session
                    .user_turn_streaming(prompt)
                    .map_err(|e| format!("Streaming generation failed: {}", e))?;

                let mut chunks = Vec::new();
                while let Some(chunk) = stream
                    .next_chunk()
                    .map_err(|e| format!("Streaming generation failed: {}", e))?
                {
                    if !chunk.is_empty() {
                        chunks.push(chunk);
                    }
                }

                Ok(chunks)
            }
            BackendSession::MistralRs(session) => {
                let mut active = session.start_generate_stream(prompt, config)?;
                let mut chunks = Vec::new();
                while let Some(chunk) = active.next_chunk()? {
                    chunks.push(chunk);
                }
                Ok(chunks)
            }
        }
    }

    pub(crate) fn start_generate_stream(
        &mut self,
        prompt: &str,
        config: &ferrite_core::GenerationConfig,
    ) -> Result<ActiveGeneration, String> {
        match &mut self.session {
            BackendSession::Candle(session) => {
                session.set_generation_config(config.clone());
                let mut stream = session
                    .user_turn_streaming(prompt)
                    .map_err(|e| format!("Streaming generation failed: {}", e))?;
                let mut chunks = VecDeque::new();
                while let Some(chunk) = stream
                    .next_chunk()
                    .map_err(|e| format!("Streaming generation failed: {}", e))?
                {
                    if !chunk.is_empty() {
                        chunks.push_back(chunk);
                    }
                }
                Ok(ActiveGeneration::Buffered(BufferedGeneration::Chunks(chunks)))
            }
            BackendSession::MistralRs(session) => session.start_generate_stream(prompt, config),
        }
    }

    /// List all available models
    pub fn list_available(model_cache: &Path) -> Vec<ferrite_core::ModelInfo> {
        ModelLoader::new(model_cache).list_models()
    }

    /// Search for models
    pub fn search(model_cache: &Path, query: &str) -> Vec<ferrite_core::ModelInfo> {
        ModelLoader::new(model_cache).search(query)
    }

    pub fn tokenizer(&self) -> Arc<ferrite_core::Tokenizer> {
        Arc::clone(&self.tokenizer)
    }
}

/// Tokenizer adapter (for direct tokenization without full model)
pub struct TokenizerAdapter<'a> {
    tokenizer: &'a ferrite_core::Tokenizer,
}

impl<'a> TokenizerAdapter<'a> {
    pub fn new(tokenizer: &'a ferrite_core::Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer
            .encode(text)
            .map(|encoding| encoding.ids)
            .unwrap_or_default()
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens, true).unwrap_or_default()
    }

    pub fn decode_token(&self, token: u32) -> String {
        self.decode(&[token])
    }
}
