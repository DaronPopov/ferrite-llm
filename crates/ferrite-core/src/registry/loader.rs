//! Model Loader
//!
//! Unified model loading with HuggingFace integration and GGUF auto-detection.

use super::catalog::Catalog;
use super::spec::*;
use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Model loader with caching and HuggingFace integration
pub struct ModelLoader {
    catalog: Catalog,
    #[allow(dead_code)]
    cache_dir: PathBuf,
    auth_token: Option<String>,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            catalog: Catalog::new(),
            cache_dir: cache_dir.into(),
            auth_token: None,
        }
    }

    /// Set authentication token
    pub fn with_auth(mut self, token: Option<String>) -> Self {
        self.auth_token = token;
        self
    }

    /// Get the model catalog
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// List all available models
    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.catalog.list().iter().map(|s| ModelInfo::from(*s)).collect()
    }

    /// Search for models
    pub fn search(&self, query: &str) -> Vec<ModelInfo> {
        self.catalog.search(query).iter().map(|s| ModelInfo::from(*s)).collect()
    }

    /// Get a model spec by name
    pub fn get_spec(&self, name: &str) -> Option<&ModelSpec> {
        self.catalog.get(name)
    }

    /// Download a model and return the local path
    pub fn download(&self, name: &str) -> Result<DownloadedModel> {
        let spec = self.catalog.get(name)
            .ok_or_else(|| anyhow!("Unknown model: {}. Use 'list' to see available models.", name))?;

        self.download_spec(spec)
    }

    /// Download a model from its spec
    pub fn download_spec(&self, spec: &ModelSpec) -> Result<DownloadedModel> {
        info!("Downloading model: {} ({:?})", spec.name, spec.family);

        // Check if auth is required
        if spec.requires_auth && self.auth_token.is_none() {
            warn!("Model {} requires authentication. Set HF_TOKEN environment variable.", spec.name);
        }

        // Download weights
        let weights_path = self.download_weights(spec)?;

        // Download tokenizer
        let tokenizer_path = self.download_tokenizer(spec)?;

        Ok(DownloadedModel {
            spec: spec.clone(),
            weights_path,
            tokenizer_path,
        })
    }

    /// Download model weights
    fn download_weights(&self, spec: &ModelSpec) -> Result<PathBuf> {
        match &spec.source {
            ModelSource::HuggingFace { repo, file, revision } => {
                let api = self.create_hf_api()?;
                let repo_api = match revision {
                    Some(rev) => api.repo(hf_hub::Repo::with_revision(
                        repo.clone(),
                        hf_hub::RepoType::Model,
                        rev.clone(),
                    )),
                    None => api.model(repo.clone()),
                };

                let filename = file.as_ref()
                    .ok_or_else(|| anyhow!("No filename specified for HuggingFace model"))?;

                info!("Downloading {} from {}", filename, repo);
                let path = repo_api.get(filename)
                    .with_context(|| format!("Failed to download {} from {}", filename, repo))?;

                Ok(path)
            }
            ModelSource::Local { path } => {
                if path.exists() {
                    Ok(path.clone())
                } else {
                    Err(anyhow!("Local model not found: {}", path.display()))
                }
            }
            ModelSource::Url { url } => {
                Err(anyhow!("URL download not yet implemented: {}", url))
            }
        }
    }

    /// Download tokenizer files
    fn download_tokenizer(&self, spec: &ModelSpec) -> Result<PathBuf> {
        let (repo, is_local) = match &spec.tokenizer {
            TokenizerSource::SameAsModel => {
                match &spec.source {
                    ModelSource::HuggingFace { repo, .. } => (repo.clone(), false),
                    ModelSource::Local { path } => {
                        // Assume tokenizer is in same directory
                        return Ok(path.parent().unwrap_or(path).to_path_buf());
                    }
                    _ => return Err(anyhow!("Cannot determine tokenizer location")),
                }
            }
            TokenizerSource::HuggingFace { repo } => (repo.clone(), false),
            TokenizerSource::Local { path } => return Ok(path.clone()),
        };

        if is_local {
            return Err(anyhow!("Local tokenizer not found"));
        }

        let api = self.create_hf_api()?;
        let repo_api = api.model(repo.clone());

        // Download tokenizer.json (required)
        info!("Downloading tokenizer from {}", repo);
        let tokenizer_path = repo_api.get("tokenizer.json")
            .with_context(|| format!("Failed to download tokenizer.json from {}", repo))?;

        // Try to download tokenizer_config.json (optional)
        if let Ok(_) = repo_api.get("tokenizer_config.json") {
            debug!("Downloaded tokenizer_config.json from {}", repo);
        }

        // Return the directory containing the tokenizer
        Ok(tokenizer_path.parent().unwrap_or(&tokenizer_path).to_path_buf())
    }

    /// Create HuggingFace API client
    fn create_hf_api(&self) -> Result<hf_hub::api::sync::Api> {
        let mut builder = hf_hub::api::sync::ApiBuilder::new();

        if let Some(ref token) = self.auth_token {
            builder = builder.with_token(Some(token.clone()));
            debug!("Using HuggingFace authentication token");
        }

        builder.build().context("Failed to create HuggingFace API client")
    }

    /// Load a GGUF file directly with auto-detection
    pub fn load_gguf_auto(&self, path: &Path) -> Result<DownloadedModel> {
        info!("Auto-detecting GGUF model: {}", path.display());

        // Read GGUF metadata to detect architecture
        let metadata = read_gguf_metadata(path)?;
        let arch = metadata.get("general.architecture")
            .ok_or_else(|| anyhow!("GGUF file missing architecture metadata"))?;

        let family = ModelFamily::from_gguf_arch(arch)
            .ok_or_else(|| anyhow!("Unknown architecture: {}", arch))?;

        info!("Detected architecture: {:?} (from '{}')", family, arch);

        // Get model name from metadata or filename
        let name = metadata.get("general.name")
            .cloned()
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            });

        // Detect context length
        let context_length = metadata.get("llama.context_length")
            .or_else(|| metadata.get("phi.context_length"))
            .or_else(|| metadata.get("gemma.context_length"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);

        // Create a dynamic spec
        let spec = ModelSpec {
            name: name.clone(),
            family,
            source: ModelSource::Local { path: path.to_path_buf() },
            format: WeightFormat::GGUF,
            chat_template: detect_chat_template(&metadata, family),
            context_length,
            tokenizer: TokenizerSource::SameAsModel,
            description: format!("Auto-detected {:?} model", family),
            size: detect_size(&metadata),
            requires_auth: false,
        };

        Ok(DownloadedModel {
            spec,
            weights_path: path.to_path_buf(),
            tokenizer_path: path.parent().unwrap_or(path).to_path_buf(),
        })
    }
}

/// Downloaded model ready to be loaded
#[derive(Debug, Clone)]
pub struct DownloadedModel {
    pub spec: ModelSpec,
    pub weights_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

impl DownloadedModel {
    pub fn name(&self) -> &str {
        &self.spec.name
    }

    pub fn family(&self) -> ModelFamily {
        self.spec.family
    }

    pub fn chat_template(&self) -> ChatTemplate {
        self.spec.chat_template
    }
}

/// Read GGUF metadata as key-value pairs
fn read_gguf_metadata(path: &Path) -> Result<std::collections::HashMap<String, String>> {
    use std::fs::File;
    use std::io::{BufReader, Read};

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read magic number
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(anyhow!("Invalid GGUF file: bad magic number"));
    }

    // Read version
    let mut version = [0u8; 4];
    reader.read_exact(&mut version)?;
    let version = u32::from_le_bytes(version);
    debug!("GGUF version: {}", version);

    // Read tensor count
    let mut tensor_count = [0u8; 8];
    reader.read_exact(&mut tensor_count)?;

    // Read metadata count
    let mut metadata_count = [0u8; 8];
    reader.read_exact(&mut metadata_count)?;
    let metadata_count = u64::from_le_bytes(metadata_count);
    debug!("GGUF metadata entries: {}", metadata_count);

    let mut metadata = std::collections::HashMap::new();

    // Read metadata entries (simplified - only string values for now)
    for _ in 0..metadata_count.min(100) {
        // Read key
        let key = match read_gguf_string(&mut reader) {
            Ok(k) => k,
            Err(_) => break,
        };

        // Read value type
        let mut value_type = [0u8; 4];
        if reader.read_exact(&mut value_type).is_err() {
            break;
        }
        let value_type = u32::from_le_bytes(value_type);

        // Read value based on type
        let value = match value_type {
            // GGUF_TYPE_STRING = 8
            8 => read_gguf_string(&mut reader).unwrap_or_default(),
            // GGUF_TYPE_UINT32 = 4
            4 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf).ok();
                u32::from_le_bytes(buf).to_string()
            }
            // GGUF_TYPE_UINT64 = 10
            10 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf).ok();
                u64::from_le_bytes(buf).to_string()
            }
            // Skip other types for now
            _ => {
                // Skip unknown type - this is a simplification
                continue;
            }
        };

        if !key.is_empty() {
            metadata.insert(key, value);
        }
    }

    Ok(metadata)
}

/// Read a GGUF string
fn read_gguf_string<R: std::io::Read>(reader: &mut R) -> Result<String> {
    let mut len_buf = [0u8; 8];
    reader.read_exact(&mut len_buf)?;
    let len = u64::from_le_bytes(len_buf) as usize;

    if len > 1024 * 1024 {
        return Err(anyhow!("String too long: {} bytes", len));
    }

    let mut string_buf = vec![0u8; len];
    reader.read_exact(&mut string_buf)?;

    String::from_utf8(string_buf).context("Invalid UTF-8 in GGUF string")
}

/// Detect chat template from metadata
fn detect_chat_template(metadata: &std::collections::HashMap<String, String>, family: ModelFamily) -> ChatTemplate {
    // Check for explicit chat template in metadata
    if let Some(template) = metadata.get("tokenizer.chat_template") {
        if template.contains("[INST]") {
            return ChatTemplate::Mistral;
        }
        if template.contains("<|im_start|>") {
            return ChatTemplate::ChatML;
        }
        if template.contains("<|begin_of_text|>") {
            return ChatTemplate::Llama3;
        }
        if template.contains("<start_of_turn>") {
            return ChatTemplate::Gemma;
        }
        if template.contains("<|system|>") {
            return ChatTemplate::Phi3;
        }
    }

    // Check model name
    if let Some(name) = metadata.get("general.name") {
        let name_lower = name.to_lowercase();
        if name_lower.contains("mistral") {
            return ChatTemplate::Mistral;
        }
        if name_lower.contains("llama-3") || name_lower.contains("llama3") {
            return ChatTemplate::Llama3;
        }
        if name_lower.contains("qwen") {
            return ChatTemplate::ChatML;
        }
        if name_lower.contains("phi-3") || name_lower.contains("phi3") {
            return ChatTemplate::Phi3;
        }
        if name_lower.contains("gemma") {
            return ChatTemplate::Gemma;
        }
    }

    // Fall back to family default
    match family {
        ModelFamily::Llama => ChatTemplate::Mistral,
        ModelFamily::Phi => ChatTemplate::Phi3,
        ModelFamily::Gemma => ChatTemplate::Gemma,
        _ => ChatTemplate::Raw,
    }
}

/// Detect model size from metadata
fn detect_size(metadata: &std::collections::HashMap<String, String>) -> String {
    // Try to get from file metadata
    if let Some(size) = metadata.get("general.size_label") {
        return size.clone();
    }

    // Estimate from parameter count or layer info
    if let Some(layers) = metadata.get("llama.block_count")
        .or_else(|| metadata.get("phi.block_count")) {
        if let Ok(n) = layers.parse::<u32>() {
            return match n {
                0..=16 => "~1B".into(),
                17..=24 => "~3B".into(),
                25..=32 => "~7B".into(),
                33..=48 => "~13B".into(),
                _ => "~70B".into(),
            };
        }
    }

    "Unknown".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_creation() {
        let loader = ModelLoader::new("./models");
        assert!(loader.catalog().len() > 0);
    }

    #[test]
    fn test_list_models() {
        let loader = ModelLoader::new("./models");
        let models = loader.list_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_search() {
        let loader = ModelLoader::new("./models");
        let results = loader.search("mistral");
        assert!(!results.is_empty());
    }
}
