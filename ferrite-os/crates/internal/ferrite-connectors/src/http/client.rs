use crate::retry::retry_with_backoff;
use std::collections::HashMap;
use std::io::Write;

/// Minimal Base64 encoder for HTTP Basic auth (no external dependency).
fn base64_encode(input: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// HTTP response from the client.
#[derive(Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
}

/// Full-featured HTTP client wrapping ureq.
///
/// Supports all standard methods, auth (Basic/Bearer), JSON bodies,
/// file downloads, custom headers, and retry with backoff.
pub struct HttpClient {
    headers: HashMap<String, String>,
    max_retries: u32,
    timeout_ms: u64,
}

impl HttpClient {
    pub fn new() -> Self {
        Self {
            headers: HashMap::new(),
            max_retries: 0,
            timeout_ms: 30_000,
        }
    }

    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_bearer_token(self, token: &str) -> Self {
        self.with_header("Authorization", &format!("Bearer {}", token))
    }

    pub fn with_basic_auth(self, user: &str, pass: &str) -> Self {
        let encoded = base64_encode(format!("{}:{}", user, pass).as_bytes());
        self.with_header("Authorization", &format!("Basic {}", encoded))
    }

    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    pub fn get(&self, url: &str) -> anyhow::Result<HttpResponse> {
        self.request("GET", url, None)
    }

    pub fn post(&self, url: &str, body: &serde_json::Value) -> anyhow::Result<HttpResponse> {
        self.request("POST", url, Some(body))
    }

    pub fn put(&self, url: &str, body: &serde_json::Value) -> anyhow::Result<HttpResponse> {
        self.request("PUT", url, Some(body))
    }

    pub fn delete(&self, url: &str) -> anyhow::Result<HttpResponse> {
        self.request("DELETE", url, None)
    }

    pub fn patch(&self, url: &str, body: &serde_json::Value) -> anyhow::Result<HttpResponse> {
        self.request("PATCH", url, Some(body))
    }

    pub fn head(&self, url: &str) -> anyhow::Result<HttpResponse> {
        self.request("HEAD", url, None)
    }

    /// Download a URL to a file, returning the number of bytes written.
    pub fn download(&self, url: &str, path: &std::path::Path) -> anyhow::Result<u64> {
        let headers = self.headers.clone();
        let url = url.to_string();
        let timeout_ms = self.timeout_ms;

        let resp = retry_with_backoff(self.max_retries, 500, || {
            let agent = ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_millis(timeout_ms))
                .build();
            let mut req = agent.get(&url);
            for (k, v) in &headers {
                req = req.set(k, v);
            }
            req.call().map_err(|e| anyhow::anyhow!("{}", e))
        })?;

        let mut file = std::fs::File::create(path)
            .map_err(|e| anyhow::anyhow!("create file {}: {}", path.display(), e))?;
        let mut reader = resp.into_reader();
        let bytes = std::io::copy(&mut reader, &mut file)
            .map_err(|e| anyhow::anyhow!("download write: {}", e))?;
        file.flush()?;
        Ok(bytes)
    }

    fn request(
        &self,
        method: &str,
        url: &str,
        body: Option<&serde_json::Value>,
    ) -> anyhow::Result<HttpResponse> {
        let headers = self.headers.clone();
        let url = url.to_string();
        let body_str = body.map(|b| serde_json::to_string(b).unwrap_or_default());
        let method = method.to_string();
        let timeout_ms = self.timeout_ms;

        retry_with_backoff(self.max_retries, 500, || {
            let agent = ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_millis(timeout_ms))
                .build();
            let mut req = agent.request(&method, &url);
            for (k, v) in &headers {
                req = req.set(k, v);
            }
            if body_str.is_some() {
                req = req.set("Content-Type", "application/json");
            }

            let resp = if let Some(ref b) = body_str {
                req.send_string(b)
            } else {
                req.call()
            };

            match resp {
                Ok(resp) => {
                    let status = resp.status();
                    let resp_headers = extract_headers(&resp);
                    let body = resp.into_string().unwrap_or_default();
                    Ok(HttpResponse {
                        status,
                        headers: resp_headers,
                        body,
                    })
                }
                Err(ureq::Error::Status(code, resp)) => {
                    // Non-2xx status — still a valid HTTP response, don't retry
                    let resp_headers = extract_headers(&resp);
                    let body = resp.into_string().unwrap_or_default();
                    Ok(HttpResponse {
                        status: code,
                        headers: resp_headers,
                        body,
                    })
                }
                Err(e) => Err(anyhow::anyhow!("{}", e)),
            }
        })
    }
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

fn extract_headers(resp: &ureq::Response) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    for name in resp.headers_names() {
        if let Some(val) = resp.header(&name) {
            headers.insert(name, val.to_string());
        }
    }
    headers
}

#[cfg(test)]
mod tests {
    use super::base64_encode;

    #[test]
    fn base64_empty() {
        assert_eq!(base64_encode(b""), "");
    }

    #[test]
    fn base64_standard_vectors() {
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_auth_string() {
        let encoded = base64_encode(b"user:pass");
        assert_eq!(encoded, "dXNlcjpwYXNz");
    }
}
