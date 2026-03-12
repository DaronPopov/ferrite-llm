//! Ferrite - Lightweight LLM Inference Utilities
//!
//! A companion library for Candle that provides:
//! - Tokenizer with chat template support
//! - Model configuration registry
//! - Generation utilities (sampling, streaming)
//! - Performance tracking (TPS metrics)
//!
//! # Example
//! ```ignore
//! use ferrite::{Tokenizer, ChatMessage, GenerationConfig};
//!
//! let tokenizer = Tokenizer::from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").await?;
//!
//! let messages = vec![
//!     ChatMessage::system("You are a helpful assistant."),
//!     ChatMessage::user("Hello!"),
//! ];
//!
//! let prompt = tokenizer.apply_chat_template(&messages, true)?;
//! ```

pub mod tokenizer;
pub mod generation;
pub mod models;
pub mod sampling;
pub mod chat_session;
pub mod hooks;
pub mod flash_attention;
pub mod cuda_kernel;
pub mod registry;
pub mod tlsf_alloc;
#[cfg(feature = "cuda")]
pub mod ug_cuda_support;

// Re-exports for convenience
pub use tokenizer::{Tokenizer, ChatMessage, ChatRole, Encoding, TokenizerError};
pub use generation::{GenerationConfig, StopCondition, GenerationStats, InferenceModel, StreamingInference, SpeculativeInference};
pub use models::{ModelConfig, ModelFamily};
pub use sampling::{Sampler, SamplerConfig};
pub use hooks::{LogitsCandidate, LogitsHook};
pub use chat_session::{ChatSession, ChatSessionConfig, ChatSessionError, CachedMessage, MessageRole, StreamingResponse};
pub use models::ChatFormat;
pub use flash_attention::{flash_attention, FlashAttentionConfig};

#[cfg(feature = "cuda")]
pub use cuda_kernel::{CudaKernel, CudaKernelConfig};
#[cfg(feature = "cuda")]
pub use ug_cuda_support::{create_device as create_ug_cuda_device, ug, ug_cuda};
pub use tlsf_alloc::maybe_enable_tlsf_allocator;

// Registry re-exports
pub use registry::{
    Catalog, ChatTemplate, DownloadedModel, ModelFamily as RegistryModelFamily,
    ModelInfo, ModelLoader, ModelSource, ModelSpec, TokenizerSource, WeightFormat,
};
