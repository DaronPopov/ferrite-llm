//! Model Registry
//!
//! Declarative model catalog with HuggingFace integration and GGUF auto-detection.
//!
//! # Architecture
//!
//! The registry provides a clean abstraction over model loading:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      ModelLoader                             │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │   Catalog   │  │  Downloader │  │   Auto-Detector     │  │
//! │  │ (built-in   │  │ (HF Hub,    │  │ (GGUF metadata      │  │
//! │  │  specs)     │  │  local)     │  │  parsing)           │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//!                    ┌─────────────────┐
//!                    │ DownloadedModel │
//!                    │  - weights_path │
//!                    │  - tokenizer    │
//!                    │  - spec         │
//!                    └─────────────────┘
//! ```
//!
//! # Usage
//!
//! ## List Available Models
//!
//! ```rust,ignore
//! use ferrite_core::registry::ModelLoader;
//!
//! let loader = ModelLoader::new("./models");
//!
//! // List all models
//! for model in loader.list_models() {
//!     println!("{}: {} ({})", model.name, model.description, model.size);
//! }
//!
//! // Search for specific models
//! let results = loader.search("llama");
//! ```
//!
//! ## Download and Load a Model
//!
//! ```rust,ignore
//! let loader = ModelLoader::new("./models")
//!     .with_auth(Some("hf_xxx".into()));
//!
//! // Download by name
//! let model = loader.download("mistral-7b-q4")?;
//! println!("Downloaded to: {}", model.weights_path.display());
//!
//! // Auto-detect from GGUF file
//! let model = loader.load_gguf_auto(Path::new("./my-model.gguf"))?;
//! println!("Detected: {:?}", model.family());
//! ```
//!
//! # Adding New Models
//!
//! New models can be added by modifying `catalog.rs`:
//!
//! ```rust,ignore
//! self.register(ModelSpec {
//!     name: "my-model-7b-q4".into(),
//!     family: ModelFamily::Llama,
//!     source: ModelSource::HuggingFace {
//!         repo: "org/my-model-gguf".into(),
//!         file: Some("my-model.Q4_K_M.gguf".into()),
//!         revision: None,
//!     },
//!     format: WeightFormat::GGUF,
//!     chat_template: ChatTemplate::ChatML,
//!     context_length: 32768,
//!     tokenizer: TokenizerSource::HuggingFace {
//!         repo: "org/my-model".into(),
//!     },
//!     description: "My Model 7B (4-bit quantized)".into(),
//!     size: "7B".into(),
//!     requires_auth: false,
//! });
//! ```

mod catalog;
mod loader;
mod spec;

pub use catalog::Catalog;
pub use loader::{DownloadedModel, ModelLoader};
pub use spec::{
    ChatMessage, ChatTemplate, ModelFamily, ModelInfo, ModelSource, ModelSpec,
    Role, TokenizerSource, WeightFormat,
};
