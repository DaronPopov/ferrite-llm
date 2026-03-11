use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct ConnectorConfig {
    #[serde(default)]
    pub sources: Vec<SourceConfig>,
    #[serde(default)]
    pub sink: SinkConfig,
    pub server: Option<ServerSectionConfig>,
    #[serde(default = "default_queue_capacity")]
    pub queue_capacity: usize,
    #[serde(default = "default_shutdown_drain_timeout_ms")]
    pub shutdown_drain_timeout_ms: u64,
}

fn default_queue_capacity() -> usize {
    4096
}

fn default_shutdown_drain_timeout_ms() -> u64 {
    5000
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SourceConfig {
    Rest {
        name: String,
        url: String,
        #[serde(default = "default_interval_ms")]
        interval_ms: u64,
        #[serde(default)]
        headers: HashMap<String, String>,
        jq_path: Option<String>,
        #[serde(default = "default_max_retries")]
        max_retries: u32,
    },
    Ws {
        name: String,
        url: String,
        #[serde(default = "default_reconnect_ms")]
        reconnect_ms: u64,
        #[serde(default = "default_max_retries")]
        max_retries: u32,
    },
}

fn default_interval_ms() -> u64 {
    5000
}

fn default_reconnect_ms() -> u64 {
    3000
}

fn default_max_retries() -> u32 {
    5
}

#[derive(Debug, Deserialize)]
pub struct SinkConfig {
    pub socket_path: Option<String>,
    #[serde(default = "default_command_template")]
    pub command_template: String,
    pub output_file: Option<String>,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_flush_interval_ms")]
    pub flush_interval_ms: u64,
    #[serde(default = "default_ipc_enabled")]
    pub ipc_enabled: bool,
}

fn default_command_template() -> String {
    "ping".to_string()
}

fn default_batch_size() -> usize {
    1
}

fn default_flush_interval_ms() -> u64 {
    1000
}

fn default_ipc_enabled() -> bool {
    true
}

impl Default for SinkConfig {
    fn default() -> Self {
        Self {
            socket_path: None,
            command_template: default_command_template(),
            output_file: None,
            batch_size: default_batch_size(),
            flush_interval_ms: default_flush_interval_ms(),
            ipc_enabled: default_ipc_enabled(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ServerSectionConfig {
    #[serde(default = "default_server_bind")]
    pub bind: String,
    #[serde(default = "default_server_port")]
    pub port: u16,
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    #[serde(default)]
    pub feed_queue: bool,
}

fn default_server_bind() -> String {
    "127.0.0.1".to_string()
}

fn default_server_port() -> u16 {
    8080
}

fn default_max_connections() -> usize {
    64
}

impl SourceConfig {
    pub fn name(&self) -> &str {
        match self {
            SourceConfig::Rest { name, .. } => name,
            SourceConfig::Ws { name, .. } => name,
        }
    }
}

pub fn load_config(path: &Path) -> anyhow::Result<ConnectorConfig> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read config {}: {}", path.display(), e))?;
    let config: ConnectorConfig = toml::from_str(&text)
        .map_err(|e| anyhow::anyhow!("failed to parse config: {}", e))?;
    Ok(config)
}
