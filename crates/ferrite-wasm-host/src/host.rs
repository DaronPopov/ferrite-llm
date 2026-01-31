//! Host implementation - WASM runtime side
//!
//! Implements wasmtime Host traits and bridges to ferrite engine via adapter layer.

use std::path::PathBuf;
use wasmtime::component::{Resource, ResourceTable};

use super::adapter::{ModelAdapter, TokenizerAdapter, wit_to_ferrite_config};
use crate::bindings::{
    WitGenConfig,
    InferenceHost,
    HostModel,
    WitModel,
    TokenizerHost,
};

/// Host state - bridges WASM to ferrite engine
pub struct HostState {
    #[allow(dead_code)]
    model_cache: PathBuf,
    table: ResourceTable,
}

impl HostState {
    pub fn new(model_cache: PathBuf) -> anyhow::Result<Self> {
        Ok(Self {
            model_cache,
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
        let model_adapter = ModelAdapter::new(model_name.clone(), auth_token)?;

        // Push the model adapter into the resource table
        // This returns a Resource<ModelAdapter>, but we need Resource<WitModel>
        // We'll use rep() to get the underlying representation
        let resource = self
            .table
            .push(model_adapter)
            .map_err(|e| format!("Resource creation failed: {}", e))?;

        tracing::info!("✓ Model '{}' loaded", model_name);

        // Convert Resource<ModelAdapter> to Resource<WitModel> using unsafe transmute
        // This is safe because Resource<T> is just a newtype wrapper around u32
        let wit_resource: Resource<WitModel> = unsafe { std::mem::transmute(resource) };

        Ok(wit_resource)
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
        // Transmute back to Resource<ModelAdapter>
        let model_resource: Resource<ModelAdapter> = unsafe { std::mem::transmute(model) };

        // Get the model from the table
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
        let model_resource: Resource<ModelAdapter> = unsafe { std::mem::transmute(model) };

        let model_adapter = self
            .table
            .get_mut(&model_resource)
            .map_err(|e| format!("Model not found: {}", e))?;

        let ferrite_config = wit_to_ferrite_config(&config);
        model_adapter.generate_stream(&prompt, &ferrite_config)
    }

    fn drop(&mut self, model: Resource<WitModel>) -> wasmtime::Result<()> {
        let model_resource: Resource<ModelAdapter> = unsafe { std::mem::transmute(model) };
        self.table.delete(model_resource)?;
        tracing::info!("Model resource dropped");
        Ok(())
    }
}

/// Implement tokenizer interface
impl TokenizerHost for HostState {
    fn encode(&mut self, text: String) -> Vec<u32> {
        TokenizerAdapter::encode(&text)
    }

    fn decode(&mut self, tokens: Vec<u32>) -> String {
        TokenizerAdapter::decode(&tokens)
    }

    fn decode_token(&mut self, token: u32) -> String {
        TokenizerAdapter::decode_token(token)
    }
}
