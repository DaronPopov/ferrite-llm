use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Event {
    pub source: String,
    pub ts_unix_ms: u64,
    pub event_type: String,
    pub idempotency_key: String,
    pub payload: serde_json::Value,
}
