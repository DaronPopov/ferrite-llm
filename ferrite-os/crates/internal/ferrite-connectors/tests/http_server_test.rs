use ferrite_connectors::http::router::{Response, Router};
use ferrite_connectors::http::server::HttpServer;
use ferrite_connectors::queue::BoundedQueue;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn simple_get(addr: &str, path: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(addr).unwrap();
    write!(
        stream,
        "GET {} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        path
    )
    .unwrap();
    stream.flush().unwrap();

    read_http_response(stream)
}

fn simple_post(addr: &str, path: &str, body: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(addr).unwrap();
    write!(
        stream,
        "POST {} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        path,
        body.len(),
        body
    )
    .unwrap();
    stream.flush().unwrap();

    read_http_response(stream)
}

fn read_http_response(stream: TcpStream) -> (u16, String) {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .ok();
    let mut reader = BufReader::new(stream);

    let mut status_line = String::new();
    reader.read_line(&mut status_line).unwrap();
    let status: u16 = status_line
        .split(' ')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line.trim().is_empty() {
            break;
        }
        if line.to_lowercase().starts_with("content-length:") {
            content_length = line
                .split(':')
                .nth(1)
                .unwrap()
                .trim()
                .parse()
                .unwrap();
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).unwrap();
    }

    (status, String::from_utf8_lossy(&body).to_string())
}

#[test]
fn basic_get() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let router = Arc::new(Router::new().get("/hello", |_, _| Response::ok("world")));
    let mut server =
        HttpServer::start("127.0.0.1:0", router, None, 10, shutdown.clone())
            .unwrap();
    let addr = server.addr().to_string();
    thread::sleep(Duration::from_millis(50));

    let (status, body) = simple_get(&addr, "/hello");
    assert_eq!(status, 200);
    assert_eq!(body, "world");

    server.stop();
}

#[test]
fn not_found_route() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let router = Arc::new(Router::new());
    let mut server =
        HttpServer::start("127.0.0.1:0", router, None, 10, shutdown.clone())
            .unwrap();
    let addr = server.addr().to_string();
    thread::sleep(Duration::from_millis(50));

    let (status, _) = simple_get(&addr, "/missing");
    assert_eq!(status, 404);

    server.stop();
}

#[test]
fn post_with_body() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let router =
        Arc::new(Router::new().post("/echo", |req, _| Response::ok(&req.body)));
    let mut server =
        HttpServer::start("127.0.0.1:0", router, None, 10, shutdown.clone())
            .unwrap();
    let addr = server.addr().to_string();
    thread::sleep(Duration::from_millis(50));

    let (status, body) = simple_post(&addr, "/echo", "test-body");
    assert_eq!(status, 200);
    assert_eq!(body, "test-body");

    server.stop();
}

#[test]
fn queue_integration() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let queue = Arc::new(BoundedQueue::new(64));
    let router = Arc::new(
        Router::new().post("/events", |_, _| Response::ok("accepted")),
    );
    let mut server = HttpServer::start(
        "127.0.0.1:0",
        router,
        Some(queue.clone()),
        10,
        shutdown.clone(),
    )
    .unwrap();
    let addr = server.addr().to_string();
    thread::sleep(Duration::from_millis(50));

    let (status, _) = simple_post(&addr, "/events", r#"{"key":"value"}"#);
    assert_eq!(status, 200);

    // Give the server a moment to push to queue
    thread::sleep(Duration::from_millis(200));

    let events = queue.drain_batch(10);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event_type, "http_request");
    assert_eq!(events[0].source, "http-server");
    // Verify the payload contains request details
    assert_eq!(events[0].payload["method"], "POST");
    assert_eq!(events[0].payload["path"], "/events");

    server.stop();
}

#[test]
fn json_response() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let router = Arc::new(Router::new().get("/status", |_, _| {
        Response::json(&serde_json::json!({"status": "ok"}))
    }));
    let mut server =
        HttpServer::start("127.0.0.1:0", router, None, 10, shutdown.clone())
            .unwrap();
    let addr = server.addr().to_string();
    thread::sleep(Duration::from_millis(50));

    let (status, body) = simple_get(&addr, "/status");
    assert_eq!(status, 200);
    let parsed: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(parsed["status"], "ok");

    server.stop();
}

#[test]
fn param_in_route() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let router = Arc::new(Router::new().get("/items/:id", |_, params| {
        Response::ok(params.get("id").unwrap())
    }));
    let mut server =
        HttpServer::start("127.0.0.1:0", router, None, 10, shutdown.clone())
            .unwrap();
    let addr = server.addr().to_string();
    thread::sleep(Duration::from_millis(50));

    let (status, body) = simple_get(&addr, "/items/abc");
    assert_eq!(status, 200);
    assert_eq!(body, "abc");

    server.stop();
}

#[test]
fn query_string_parsing() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let router = Arc::new(Router::new().get("/search", |req, _| {
        let q = req.query.get("q").cloned().unwrap_or_default();
        Response::ok(&q)
    }));
    let mut server =
        HttpServer::start("127.0.0.1:0", router, None, 10, shutdown.clone())
            .unwrap();
    let addr = server.addr().to_string();
    thread::sleep(Duration::from_millis(50));

    let (status, body) = simple_get(&addr, "/search?q=hello&page=1");
    assert_eq!(status, 200);
    assert_eq!(body, "hello");

    server.stop();
}

#[test]
fn multiple_requests() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let router = Arc::new(
        Router::new()
            .get("/a", |_, _| Response::ok("a"))
            .get("/b", |_, _| Response::ok("b")),
    );
    let mut server =
        HttpServer::start("127.0.0.1:0", router, None, 10, shutdown.clone())
            .unwrap();
    let addr = server.addr().to_string();
    thread::sleep(Duration::from_millis(50));

    for _ in 0..5 {
        let (s1, b1) = simple_get(&addr, "/a");
        assert_eq!(s1, 200);
        assert_eq!(b1, "a");

        let (s2, b2) = simple_get(&addr, "/b");
        assert_eq!(s2, 200);
        assert_eq!(b2, "b");
    }

    server.stop();
}
