use std::time::Instant;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cmd")]
pub enum AgentCommand {
    Read { path: String },
    Write { path: String, content: String },
    Edit { path: String, line: usize, delete_count: usize, insert: Vec<String> },
    List { path: Option<String>, recursive: Option<bool> },
    Mkdir { path: String },
    Touch { path: String },
    Mv { src: String, dst: String },
    Cp { src: String, dst: String },
    Rm { path: String, confirmed: Option<bool> },
    Run { file: Option<String>, profile: Option<String>, args: Option<Vec<String>> },
    Stop,
    Status,
    OpenBuffer { path: String },
    SaveBuffer,
    BufferInfo,
    Checkpoint { label: String },
    Rollback { label: String },
    Lock { owner: String },
    Unlock,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub ok: bool,
    pub command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl AgentResponse {
    pub fn success(command: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            ok: true,
            command: command.into(),
            message: Some(message.into()),
            data: None,
            error: None,
        }
    }

    pub fn success_data(command: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            ok: true,
            command: command.into(),
            message: None,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(command: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            ok: false,
            command: command.into(),
            message: None,
            data: None,
            error: Some(error.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AuditEntry {
    #[allow(dead_code)]
    pub timestamp: Instant,
    pub command: String,
    pub success: bool,
    pub message: String,
    pub duration_us: u64,
}
