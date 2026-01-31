//! Common utilities for chatbot examples
//!
//! Provides helpers for model loading, statistics, and device management.

use candle_core::Device;
use candle_nn::VarBuilder;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Print tokens-per-second statistics
pub fn print_tps_stats(generated_tokens: usize, elapsed: Duration) {
    let tps = generated_tokens as f64 / elapsed.as_secs_f64();
    println!(
        "[Stats] {} tokens in {:.2}s ({:.1} tok/s)",
        generated_tokens,
        elapsed.as_secs_f64(),
        tps
    );
}

/// Get CUDA device if available, fallback to CPU
pub fn get_device() -> Result<Device, Box<dyn std::error::Error>> {
    Ok(Device::cuda_if_available(0)?)
}

/// Load safetensor files from a directory
pub fn load_safetensors<'a>(
    model_dir: &Path,
    dtype: candle_core::DType,
    device: &'a Device,
) -> Result<VarBuilder<'a>, Box<dyn std::error::Error>> {
    println!("[Init] Loading model weights ({:?})...", dtype);

    let safetensor_files: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "safetensors")
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();

    if safetensor_files.is_empty() {
        return Err("No safetensor files found in model directory".into());
    }

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)?
    };

    Ok(vb)
}

/// Download model from HuggingFace Hub
pub async fn download_model(
    model_id: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    println!("[Download] Fetching from HuggingFace...");
    let model_dir = llm_tokenizer::hub::from_hf(model_id, false).await?;
    println!("[Download] Cached at: {:?}\n", model_dir);
    Ok(model_dir)
}

/// Load model config from JSON file
pub fn load_config_json<T: serde::de::DeserializeOwned>(
    model_dir: &Path,
) -> Result<T, Box<dyn std::error::Error>> {
    let config_path = model_dir.join("config.json");
    let config = serde_json::from_reader(std::fs::File::open(&config_path)?)?;
    Ok(config)
}

/// Download GGUF file from HuggingFace
pub async fn download_gguf(
    repo_id: &str,
    filename: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("huggingface")
        .join("gguf");

    std::fs::create_dir_all(&cache_dir)?;

    let local_path = cache_dir.join(filename);

    if local_path.exists() {
        println!("[Download] Using cached: {:?}", local_path);
        return Ok(local_path);
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, filename
    );

    println!("[Download] Fetching {} ...", filename);
    println!("[Download] URL: {}", url);

    let response = reqwest::get(&url).await?;

    if !response.status().is_success() {
        return Err(format!("Failed to download: HTTP {}", response.status()).into());
    }

    let bytes = response.bytes().await?;
    std::fs::write(&local_path, &bytes)?;

    println!("[Download] Saved to: {:?}", local_path);
    Ok(local_path)
}

/// Simple performance tracker for generation
pub struct GenerationTimer {
    start: Instant,
}

impl GenerationTimer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn print_stats(&self, generated_tokens: usize) {
        print_tps_stats(generated_tokens, self.elapsed());
    }
}

impl Default for GenerationTimer {
    fn default() -> Self {
        Self::new()
    }
}
