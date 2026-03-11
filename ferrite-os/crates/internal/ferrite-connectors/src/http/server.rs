use crate::http::router::{Request, Response, Router};
use crate::model::Event;
use crate::normalize::normalize;
use crate::queue::BoundedQueue;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Minimal HTTP/1.1 server built on std::net::TcpListener.
///
/// Thread-per-connection model matching the rest of ferrite-connectors.
/// Optionally feeds incoming requests into a BoundedQueue as Events.
pub struct HttpServer {
    addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl HttpServer {
    pub fn start(
        bind: &str,
        router: Arc<Router>,
        queue: Option<Arc<BoundedQueue<Event>>>,
        max_connections: usize,
        shutdown: Arc<AtomicBool>,
    ) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(bind)
            .map_err(|e| anyhow::anyhow!("HTTP bind {}: {}", bind, e))?;
        let addr = listener.local_addr()?;
        listener.set_nonblocking(true)?;

        let shutdown_clone = shutdown.clone();
        let handle = thread::Builder::new()
            .name(format!("http-server-{}", addr))
            .spawn(move || {
                tracing::info!(addr = %addr, "HTTP server started");
                let active = Arc::new(AtomicUsize::new(0));

                while !shutdown_clone.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, peer)) => {
                            if active.load(Ordering::Relaxed) >= max_connections {
                                tracing::warn!(peer = %peer, "max connections reached, rejecting");
                                drop(stream);
                                continue;
                            }
                            let r = router.clone();
                            let q = queue.clone();
                            let a = active.clone();
                            a.fetch_add(1, Ordering::Relaxed);
                            thread::Builder::new()
                                .name(format!("http-conn-{}", peer))
                                .spawn(move || {
                                    handle_connection(stream, &r, q.as_ref());
                                    a.fetch_sub(1, Ordering::Relaxed);
                                })
                                .ok();
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(50));
                        }
                        Err(e) => {
                            if !shutdown_clone.load(Ordering::Relaxed) {
                                tracing::error!(error = %e, "HTTP accept error");
                            }
                            break;
                        }
                    }
                }
                tracing::info!("HTTP server stopped");
            })
            .map_err(|e| anyhow::anyhow!("spawn HTTP server: {}", e))?;

        Ok(Self {
            addr,
            shutdown,
            handle: Some(handle),
        })
    }

    /// Returns the local address the server is bound to.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Signal shutdown and wait for the server thread to finish.
    pub fn stop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            // Brief connect to unblock the accept loop
            let _ = TcpStream::connect(self.addr);
            let _ = handle.join();
        }
    }
}

fn handle_connection(
    stream: TcpStream,
    router: &Router,
    queue: Option<&Arc<BoundedQueue<Event>>>,
) {
    stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(10))).ok();

    let mut write_stream = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut reader = BufReader::new(stream);
    let request = match parse_request(&mut reader) {
        Ok(req) => req,
        Err(e) => {
            tracing::debug!(error = %e, "failed to parse HTTP request");
            let resp = Response::bad_request("malformed request");
            let _ = write_response(&mut write_stream, &resp);
            return;
        }
    };

    // Optionally feed into queue as an Event
    if let Some(q) = queue {
        let payload = serde_json::json!({
            "method": request.method,
            "path": request.path,
            "headers": request.headers,
            "body": request.body,
            "query": request.query,
        });
        let event = normalize("http-server", "http_request", payload);
        if q.push(event) {
            tracing::warn!("queue full, dropped oldest event");
        }
    }

    let response = router.dispatch(&request);
    if let Err(e) = write_response(&mut write_stream, &response) {
        tracing::debug!(error = %e, "failed to write HTTP response");
    }
}

fn parse_request(reader: &mut BufReader<TcpStream>) -> anyhow::Result<Request> {
    // Read request line
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let parts: Vec<&str> = request_line.trim().splitn(3, ' ').collect();
    if parts.len() < 2 {
        anyhow::bail!("invalid request line");
    }
    let method = parts[0].to_string();
    let full_path = parts[1].to_string();

    // Split path and query string
    let (path, query) = if let Some(idx) = full_path.find('?') {
        let p = full_path[..idx].to_string();
        let q = parse_query(&full_path[idx + 1..]);
        (p, q)
    } else {
        (full_path, HashMap::new())
    };

    // Read headers
    let mut headers = HashMap::new();
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some((key, val)) = trimmed.split_once(':') {
            let key = key.trim().to_lowercase();
            let val = val.trim().to_string();
            if key == "content-length" {
                content_length = val.parse().unwrap_or(0);
            }
            headers.insert(key, val);
        }
    }

    // Read body based on Content-Length
    let body = if content_length > 0 {
        let mut buf = vec![0u8; content_length];
        reader.read_exact(&mut buf)?;
        String::from_utf8_lossy(&buf).to_string()
    } else {
        String::new()
    };

    Ok(Request {
        method,
        path,
        headers,
        body,
        query,
    })
}

fn parse_query(query: &str) -> HashMap<String, String> {
    query
        .split('&')
        .filter(|s| !s.is_empty())
        .filter_map(|pair| {
            pair.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
        })
        .collect()
}

fn write_response(stream: &mut TcpStream, response: &Response) -> anyhow::Result<()> {
    let status_text = match response.status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "OK",
    };
    write!(stream, "HTTP/1.1 {} {}\r\n", response.status, status_text)?;
    write!(stream, "Content-Length: {}\r\n", response.body.len())?;
    write!(stream, "Connection: close\r\n")?;
    for (k, v) in &response.headers {
        write!(stream, "{}: {}\r\n", k, v)?;
    }
    write!(stream, "\r\n")?;
    stream.write_all(response.body.as_bytes())?;
    stream.flush()?;
    Ok(())
}
