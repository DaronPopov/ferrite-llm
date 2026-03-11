use ferrite_connectors::http::client::HttpClient;
use ferrite_connectors::http::router::{Response, Router};
use ferrite_connectors::http::server::HttpServer;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn start_mock_server() -> (HttpServer, String, Arc<AtomicBool>) {
    let shutdown = Arc::new(AtomicBool::new(false));
    let router = Arc::new(
        Router::new()
            .get("/hello", |_, _| Response::ok("world"))
            .get("/json", |_, _| {
                Response::json(&serde_json::json!({"key": "value"}))
            })
            .post("/echo", |req, _| Response::ok(&req.body))
            .put("/data", |req, _| Response::ok(&req.body))
            .delete("/item", |_, _| Response::ok("deleted"))
            .get("/users/:id", |_, params| {
                Response::json(&serde_json::json!({"id": params["id"]}))
            })
            .get("/headers", |req, _| {
                let auth = req
                    .headers
                    .get("authorization")
                    .cloned()
                    .unwrap_or_default();
                Response::ok(&auth)
            }),
    );
    let server = HttpServer::start(
        "127.0.0.1:0",
        router,
        None,
        10,
        shutdown.clone(),
    )
    .unwrap();
    let addr = format!("http://{}", server.addr());
    thread::sleep(Duration::from_millis(50));
    (server, addr, shutdown)
}

#[test]
fn get_request() {
    let (mut server, addr, _) = start_mock_server();
    let client = HttpClient::new();
    let resp = client.get(&format!("{}/hello", addr)).unwrap();
    assert_eq!(resp.status, 200);
    assert_eq!(resp.body, "world");
    server.stop();
}

#[test]
fn get_json() {
    let (mut server, addr, _) = start_mock_server();
    let client = HttpClient::new();
    let resp = client.get(&format!("{}/json", addr)).unwrap();
    assert_eq!(resp.status, 200);
    let val: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert_eq!(val["key"], "value");
    server.stop();
}

#[test]
fn post_request() {
    let (mut server, addr, _) = start_mock_server();
    let client = HttpClient::new();
    let body = serde_json::json!({"hello": "world"});
    let resp = client.post(&format!("{}/echo", addr), &body).unwrap();
    assert_eq!(resp.status, 200);
    let parsed: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert_eq!(parsed["hello"], "world");
    server.stop();
}

#[test]
fn put_request() {
    let (mut server, addr, _) = start_mock_server();
    let client = HttpClient::new();
    let body = serde_json::json!({"updated": true});
    let resp = client.put(&format!("{}/data", addr), &body).unwrap();
    assert_eq!(resp.status, 200);
    server.stop();
}

#[test]
fn delete_request() {
    let (mut server, addr, _) = start_mock_server();
    let client = HttpClient::new();
    let resp = client.delete(&format!("{}/item", addr)).unwrap();
    assert_eq!(resp.status, 200);
    assert_eq!(resp.body, "deleted");
    server.stop();
}

#[test]
fn bearer_auth() {
    let (mut server, addr, _) = start_mock_server();
    let client = HttpClient::new().with_bearer_token("my-secret-token");
    let resp = client.get(&format!("{}/headers", addr)).unwrap();
    assert_eq!(resp.status, 200);
    assert_eq!(resp.body, "Bearer my-secret-token");
    server.stop();
}

#[test]
fn basic_auth() {
    let (mut server, addr, _) = start_mock_server();
    let client = HttpClient::new().with_basic_auth("user", "pass");
    let resp = client.get(&format!("{}/headers", addr)).unwrap();
    assert_eq!(resp.status, 200);
    assert!(resp.body.starts_with("Basic "));
    // Verify it contains the correct base64: "user:pass" = "dXNlcjpwYXNz"
    assert_eq!(resp.body, "Basic dXNlcjpwYXNz");
    server.stop();
}

#[test]
fn not_found_response() {
    let (mut server, addr, _) = start_mock_server();
    let client = HttpClient::new();
    let resp = client.get(&format!("{}/nonexistent", addr)).unwrap();
    assert_eq!(resp.status, 404);
    server.stop();
}

#[test]
fn custom_header() {
    let (mut server, addr, _) = start_mock_server();
    let client =
        HttpClient::new().with_header("Authorization", "Custom xyz");
    let resp = client.get(&format!("{}/headers", addr)).unwrap();
    assert_eq!(resp.status, 200);
    assert_eq!(resp.body, "Custom xyz");
    server.stop();
}

#[test]
fn download_file() {
    let (mut server, addr, _) = start_mock_server();
    let client = HttpClient::new();
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();
    let bytes = client
        .download(&format!("{}/hello", addr), &path)
        .unwrap();
    assert!(bytes > 0);
    let content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(content, "world");
    server.stop();
}

#[test]
fn with_timeout() {
    let (mut server, addr, _) = start_mock_server();
    let client = HttpClient::new().with_timeout_ms(5000);
    let resp = client.get(&format!("{}/hello", addr)).unwrap();
    assert_eq!(resp.status, 200);
    server.stop();
}

#[test]
fn param_route_via_client() {
    let (mut server, addr, _) = start_mock_server();
    let client = HttpClient::new();
    let resp = client.get(&format!("{}/users/42", addr)).unwrap();
    assert_eq!(resp.status, 200);
    let val: serde_json::Value = serde_json::from_str(&resp.body).unwrap();
    assert_eq!(val["id"], "42");
    server.stop();
}

#[test]
fn client_default_trait() {
    let client = HttpClient::default();
    // Should construct without panicking
    let _ = client;
}

#[test]
fn chained_builder() {
    let client = HttpClient::new()
        .with_header("X-Custom", "value")
        .with_max_retries(3)
        .with_timeout_ms(10_000);
    // Should construct without panicking
    let _ = client;
}
