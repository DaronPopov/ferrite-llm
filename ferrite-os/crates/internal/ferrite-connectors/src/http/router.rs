use std::collections::HashMap;

/// An incoming HTTP request.
#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: String,
    pub query: HashMap<String, String>,
}

/// An HTTP response to send back.
#[derive(Debug, Clone)]
pub struct Response {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl Response {
    pub fn ok(body: &str) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "text/plain".to_string());
        Self {
            status: 200,
            headers,
            body: body.to_string(),
        }
    }

    pub fn json(value: &serde_json::Value) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        Self {
            status: 200,
            headers,
            body: serde_json::to_string(value).unwrap_or_default(),
        }
    }

    pub fn not_found() -> Self {
        Self {
            status: 404,
            headers: HashMap::new(),
            body: "Not Found".to_string(),
        }
    }

    pub fn bad_request(msg: &str) -> Self {
        Self {
            status: 400,
            headers: HashMap::new(),
            body: msg.to_string(),
        }
    }

    pub fn internal_error(msg: &str) -> Self {
        Self {
            status: 500,
            headers: HashMap::new(),
            body: msg.to_string(),
        }
    }
}

enum Segment {
    Literal(String),
    Param(String),
}

struct Route {
    method: String,
    segments: Vec<Segment>,
    handler: Box<dyn Fn(&Request, &HashMap<String, String>) -> Response + Send + Sync>,
}

/// Path-based request router with `:param` extraction.
pub struct Router {
    routes: Vec<Route>,
}

impl Router {
    pub fn new() -> Self {
        Self { routes: Vec::new() }
    }

    /// Register a route with a method, path pattern, and handler.
    /// Path segments starting with `:` are extracted as named parameters.
    pub fn route<F>(mut self, method: &str, path: &str, handler: F) -> Self
    where
        F: Fn(&Request, &HashMap<String, String>) -> Response + Send + Sync + 'static,
    {
        let segments = parse_segments(path);
        self.routes.push(Route {
            method: method.to_uppercase(),
            segments,
            handler: Box::new(handler),
        });
        self
    }

    pub fn get<F>(self, path: &str, handler: F) -> Self
    where
        F: Fn(&Request, &HashMap<String, String>) -> Response + Send + Sync + 'static,
    {
        self.route("GET", path, handler)
    }

    pub fn post<F>(self, path: &str, handler: F) -> Self
    where
        F: Fn(&Request, &HashMap<String, String>) -> Response + Send + Sync + 'static,
    {
        self.route("POST", path, handler)
    }

    pub fn put<F>(self, path: &str, handler: F) -> Self
    where
        F: Fn(&Request, &HashMap<String, String>) -> Response + Send + Sync + 'static,
    {
        self.route("PUT", path, handler)
    }

    pub fn delete<F>(self, path: &str, handler: F) -> Self
    where
        F: Fn(&Request, &HashMap<String, String>) -> Response + Send + Sync + 'static,
    {
        self.route("DELETE", path, handler)
    }

    /// Dispatch a request to the first matching route, or return 404.
    pub fn dispatch(&self, request: &Request) -> Response {
        for route in &self.routes {
            if route.method != request.method {
                continue;
            }
            if let Some(params) = match_segments(&route.segments, &request.path) {
                return (route.handler)(request, &params);
            }
        }
        Response::not_found()
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_segments(path: &str) -> Vec<Segment> {
    path.trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|s| {
            if let Some(name) = s.strip_prefix(':') {
                Segment::Param(name.to_string())
            } else {
                Segment::Literal(s.to_string())
            }
        })
        .collect()
}

fn match_segments(segments: &[Segment], path: &str) -> Option<HashMap<String, String>> {
    let path_parts: Vec<&str> = path
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    if path_parts.len() != segments.len() {
        return None;
    }
    let mut params = HashMap::new();
    for (seg, part) in segments.iter().zip(path_parts.iter()) {
        match seg {
            Segment::Literal(lit) => {
                if lit != part {
                    return None;
                }
            }
            Segment::Param(name) => {
                params.insert(name.clone(), part.to_string());
            }
        }
    }
    Some(params)
}
