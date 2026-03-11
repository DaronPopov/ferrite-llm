use ferrite_connectors::normalize::{compute_idempotency_key, normalize};

#[test]
fn idempotency_key_deterministic() {
    let payload = serde_json::json!({"price": 42.5, "symbol": "BTC"});
    let bytes = serde_json::to_vec(&payload).unwrap();
    let k1 = compute_idempotency_key("source-a", &bytes);
    let k2 = compute_idempotency_key("source-a", &bytes);
    assert_eq!(k1, k2, "same inputs must produce same key");
}

#[test]
fn idempotency_key_is_sha256_hex() {
    let payload = serde_json::json!({"x": 1});
    let bytes = serde_json::to_vec(&payload).unwrap();
    let key = compute_idempotency_key("src", &bytes);
    assert_eq!(key.len(), 64, "SHA-256 hex digest must be 64 chars");
    assert!(
        key.chars().all(|c| c.is_ascii_hexdigit()),
        "key must be hex"
    );
}

#[test]
fn different_sources_different_keys() {
    let payload = serde_json::json!({"x": 1});
    let bytes = serde_json::to_vec(&payload).unwrap();
    let k1 = compute_idempotency_key("alpha", &bytes);
    let k2 = compute_idempotency_key("beta", &bytes);
    assert_ne!(k1, k2, "different sources must produce different keys");
}

#[test]
fn different_payloads_different_keys() {
    let b1 = serde_json::to_vec(&serde_json::json!({"a": 1})).unwrap();
    let b2 = serde_json::to_vec(&serde_json::json!({"a": 2})).unwrap();
    let k1 = compute_idempotency_key("src", &b1);
    let k2 = compute_idempotency_key("src", &b2);
    assert_ne!(k1, k2);
}

#[test]
fn normalize_populates_all_fields() {
    let event = normalize("my-source", "test_event", serde_json::json!({"val": 99}));
    assert_eq!(event.source, "my-source");
    assert_eq!(event.event_type, "test_event");
    assert!(event.ts_unix_ms > 0);
    assert_eq!(event.idempotency_key.len(), 64);
    assert_eq!(event.payload, serde_json::json!({"val": 99}));
}

#[test]
fn normalize_timestamp_is_recent() {
    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let event = normalize("src", "t", serde_json::json!(null));
    let after = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    assert!(event.ts_unix_ms >= before);
    assert!(event.ts_unix_ms <= after);
}
