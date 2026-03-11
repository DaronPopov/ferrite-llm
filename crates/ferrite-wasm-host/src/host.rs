//! Host implementation - WASM runtime side
//!
//! Implements wasmtime Host traits and bridges to ferrite engine via adapter layer.

use std::path::PathBuf;
use std::sync::Arc;
use wasmtime::component::{Resource, ResourceTable};

use super::adapter::{ModelAdapter, TokenizerAdapter, wit_to_ferrite_config};
use crate::bindings::{
    HostGeneration,
    WitGenConfig,
    InferenceHost,
    WitGeneration,
    HostModel,
    WitModel,
    TokenizerHost,
};

/// Host state - bridges WASM to ferrite engine
pub struct HostState {
    model_cache: PathBuf,
    active_tokenizer: Option<Arc<ferrite_core::Tokenizer>>,
    table: ResourceTable,
}

impl HostState {
    pub fn new(model_cache: PathBuf) -> anyhow::Result<Self> {
        Ok(Self {
            model_cache,
            active_tokenizer: None,
            table: ResourceTable::new(),
        })
    }

    pub fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }
}

/// Implement the inference interface host functions
impl InferenceHost for HostState {
    fn load_model(
        &mut self,
        model_name: String,
        auth_token: Option<String>,
    ) -> Result<Resource<WitModel>, String> {
        tracing::info!("🔄 Loading model: {}", model_name);

        // Create model adapter
        let model_adapter = ModelAdapter::new(
            model_name.clone(),
            self.model_cache.clone(),
            auth_token,
        )?;
        self.active_tokenizer = Some(model_adapter.tokenizer());

        let resource = self
            .table
            .push(model_adapter)
            .map_err(|e| format!("Resource creation failed: {}", e))?;

        tracing::info!("✓ Model '{}' loaded", model_name);

        Ok(Resource::new_own(resource.rep()))
    }
}

/// Implement model resource methods
impl HostModel for HostState {
    fn generate(
        &mut self,
        model: Resource<WitModel>,
        prompt: String,
        config: WitGenConfig,
    ) -> Result<String, String> {
        let model_resource = Resource::<ModelAdapter>::new_borrow(model.rep());

        let model_adapter = self
            .table
            .get_mut(&model_resource)
            .map_err(|e| format!("Model not found: {}", e))?;

        tracing::debug!("Generating text");
        tracing::debug!("Prompt: {}", prompt);

        // Convert config and call ferrite engine
        let ferrite_config = wit_to_ferrite_config(&config);
        model_adapter.generate(&prompt, &ferrite_config)
    }

    fn generate_stream(
        &mut self,
        model: Resource<WitModel>,
        prompt: String,
        config: WitGenConfig,
    ) -> Result<Vec<String>, String> {
        let model_resource = Resource::<ModelAdapter>::new_borrow(model.rep());

        let model_adapter = self
            .table
            .get_mut(&model_resource)
            .map_err(|e| format!("Model not found: {}", e))?;

        let ferrite_config = wit_to_ferrite_config(&config);
        model_adapter.generate_stream(&prompt, &ferrite_config)
    }

    fn start_generate_stream(
        &mut self,
        model: Resource<WitModel>,
        prompt: String,
        config: WitGenConfig,
    ) -> Result<Resource<WitGeneration>, String> {
        let model_resource = Resource::<ModelAdapter>::new_borrow(model.rep());

        let model_adapter = self
            .table
            .get_mut(&model_resource)
            .map_err(|e| format!("Model not found: {}", e))?;

        let ferrite_config = wit_to_ferrite_config(&config);
        let generation = model_adapter.start_generate_stream(&prompt, &ferrite_config)?;

        let resource = self
            .table
            .push(generation)
            .map_err(|e| format!("Generation handle creation failed: {}", e))?;

        Ok(Resource::new_own(resource.rep()))
    }

    fn drop(&mut self, model: Resource<WitModel>) -> wasmtime::Result<()> {
        let model_resource = Resource::<ModelAdapter>::new_own(model.rep());
        self.table.delete(model_resource)?;
        tracing::info!("Model resource dropped");
        Ok(())
    }
}

impl HostGeneration for HostState {
    fn next_chunk(
        &mut self,
        generation: Resource<WitGeneration>,
    ) -> Result<Option<String>, String> {
        let generation_resource = Resource::<super::adapter::ActiveGeneration>::new_borrow(generation.rep());
        let generation_handle = self
            .table
            .get_mut(&generation_resource)
            .map_err(|e| format!("Generation handle not found: {}", e))?;
        generation_handle.next_chunk()
    }

    fn drop(&mut self, generation: Resource<WitGeneration>) -> wasmtime::Result<()> {
        let generation_resource = Resource::<super::adapter::ActiveGeneration>::new_own(generation.rep());
        self.table.delete(generation_resource)?;
        tracing::info!("Generation handle dropped");
        Ok(())
    }
}

/// Implement tokenizer interface
impl TokenizerHost for HostState {
    fn encode(&mut self, text: String) -> Vec<u32> {
        self.active_tokenizer
            .as_deref()
            .map(|tokenizer| TokenizerAdapter::new(tokenizer).encode(&text))
            .unwrap_or_else(|| {
                tracing::warn!("Tokenizer requested before a model was loaded");
                Vec::new()
            })
    }

    fn decode(&mut self, tokens: Vec<u32>) -> String {
        self.active_tokenizer
            .as_deref()
            .map(|tokenizer| TokenizerAdapter::new(tokenizer).decode(&tokens))
            .unwrap_or_else(|| {
                tracing::warn!("Tokenizer requested before a model was loaded");
                String::new()
            })
    }

    fn decode_token(&mut self, token: u32) -> String {
        self.active_tokenizer
            .as_deref()
            .map(|tokenizer| TokenizerAdapter::new(tokenizer).decode_token(token))
            .unwrap_or_else(|| {
                tracing::warn!("Tokenizer requested before a model was loaded");
                String::new()
            })
    }
}
