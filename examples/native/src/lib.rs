// Ferrite Inference Platform
//
// A unified inference platform for running LLM chatbots and other ML models.

pub mod config;
pub mod models;

// Shared utilities for chatbot examples
pub mod precision;
pub mod cli;
pub mod utils;

// Re-export commonly used items
pub use precision::Precision;
pub use utils::{
    get_device, download_model, load_safetensors, load_config_json,
    download_gguf, print_tps_stats, GenerationTimer
};
