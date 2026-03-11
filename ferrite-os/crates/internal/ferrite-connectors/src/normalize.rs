use crate::model::Event;
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn normalize(source_name: &str, event_type: &str, raw: serde_json::Value) -> Event {
    let ts_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let payload_bytes = serde_json::to_vec(&raw).unwrap_or_default();
    let idempotency_key = compute_idempotency_key(source_name, &payload_bytes);

    Event {
        source: source_name.to_string(),
        ts_unix_ms,
        event_type: event_type.to_string(),
        idempotency_key,
        payload: raw,
    }
}

pub fn compute_idempotency_key(source: &str, payload_bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    hasher.update(payload_bytes);
    let result = hasher.finalize();
    hex::encode(result)
}

mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes
            .as_ref()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idempotency_key_is_deterministic() {
        let payload = serde_json::json!({"price": 42});
        let bytes = serde_json::to_vec(&payload).unwrap();
        let k1 = compute_idempotency_key("src1", &bytes);
        let k2 = compute_idempotency_key("src1", &bytes);
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn different_sources_yield_different_keys() {
        let payload = serde_json::json!({"price": 42});
        let bytes = serde_json::to_vec(&payload).unwrap();
        let k1 = compute_idempotency_key("src1", &bytes);
        let k2 = compute_idempotency_key("src2", &bytes);
        assert_ne!(k1, k2);
    }

    #[test]
    fn normalize_sets_timestamp() {
        let event = normalize("test", "data", serde_json::json!({"v": 1}));
        assert!(event.ts_unix_ms > 0);
        assert_eq!(event.source, "test");
        assert_eq!(event.event_type, "data");
    }
}
