use ferrite_connectors::http::router::{Request, Response, Router};
use std::collections::HashMap;

fn make_request(method: &str, path: &str) -> Request {
    Request {
        method: method.to_string(),
        path: path.to_string(),
        headers: HashMap::new(),
        body: String::new(),
        query: HashMap::new(),
    }
}

fn make_request_with_body(method: &str, path: &str, body: &str) -> Request {
    Request {
        method: method.to_string(),
        path: path.to_string(),
        headers: HashMap::new(),
        body: body.to_string(),
        query: HashMap::new(),
    }
}

#[test]
fn simple_get_route() {
    let router = Router::new().get("/hello", |_, _| Response::ok("world"));

    let resp = router.dispatch(&make_request("GET", "/hello"));
    assert_eq!(resp.status, 200);
    assert_eq!(resp.body, "world");
}

#[test]
fn param_extraction() {
    let router = Router::new().get("/users/:id", |_, params| {
        Response::ok(params.get("id").unwrap())
    });

    let resp = router.dispatch(&make_request("GET", "/users/42"));
    assert_eq!(resp.status, 200);
    assert_eq!(resp.body, "42");
}

#[test]
fn multiple_params() {
    let router =
        Router::new().get("/users/:user_id/posts/:post_id", |_, params| {
            let body = format!("{}:{}", params["user_id"], params["post_id"]);
            Response::ok(&body)
        });

    let resp = router.dispatch(&make_request("GET", "/users/5/posts/10"));
    assert_eq!(resp.status, 200);
    assert_eq!(resp.body, "5:10");
}

#[test]
fn not_found_for_missing_route() {
    let router = Router::new().get("/exists", |_, _| Response::ok("yes"));

    let resp = router.dispatch(&make_request("GET", "/missing"));
    assert_eq!(resp.status, 404);
}

#[test]
fn method_mismatch() {
    let router = Router::new()
        .get("/data", |_, _| Response::ok("get"))
        .post("/data", |_, _| Response::ok("post"));

    let get_resp = router.dispatch(&make_request("GET", "/data"));
    assert_eq!(get_resp.body, "get");

    let post_resp = router.dispatch(&make_request("POST", "/data"));
    assert_eq!(post_resp.body, "post");

    // DELETE not registered, should 404
    let del_resp = router.dispatch(&make_request("DELETE", "/data"));
    assert_eq!(del_resp.status, 404);
}

#[test]
fn trailing_slash_handling() {
    let router = Router::new().get("/api/data", |_, _| Response::ok("ok"));

    let resp = router.dispatch(&make_request("GET", "/api/data/"));
    assert_eq!(resp.status, 200);
}

#[test]
fn leading_slash_normalization() {
    let router = Router::new().get("api/data", |_, _| Response::ok("ok"));

    let resp = router.dispatch(&make_request("GET", "/api/data"));
    assert_eq!(resp.status, 200);
}

#[test]
fn put_and_delete_routes() {
    let router = Router::new()
        .put("/items/:id", |req, params| {
            Response::ok(&format!("put-{}:{}", params["id"], req.body))
        })
        .delete("/items/:id", |_, params| {
            Response::ok(&format!("del-{}", params["id"]))
        });

    let put_resp = router.dispatch(&make_request_with_body("PUT", "/items/7", "data"));
    assert_eq!(put_resp.body, "put-7:data");

    let del_resp = router.dispatch(&make_request("DELETE", "/items/7"));
    assert_eq!(del_resp.body, "del-7");
}

#[test]
fn response_helpers() {
    let json_resp = Response::json(&serde_json::json!({"a": 1}));
    assert_eq!(json_resp.status, 200);
    assert!(json_resp
        .headers
        .get("Content-Type")
        .unwrap()
        .contains("json"));

    let bad = Response::bad_request("oops");
    assert_eq!(bad.status, 400);
    assert_eq!(bad.body, "oops");

    let err = Response::internal_error("fail");
    assert_eq!(err.status, 500);
    assert_eq!(err.body, "fail");

    let nf = Response::not_found();
    assert_eq!(nf.status, 404);
}

#[test]
fn empty_path_matches_root() {
    let router = Router::new().get("/", |_, _| Response::ok("root"));

    // "/" has no non-empty segments, so segments list is empty
    let resp = router.dispatch(&make_request("GET", "/"));
    assert_eq!(resp.status, 200);
    assert_eq!(resp.body, "root");
}

#[test]
fn segment_count_mismatch_returns_404() {
    let router = Router::new().get("/a/b", |_, _| Response::ok("ok"));

    // Too few
    let resp = router.dispatch(&make_request("GET", "/a"));
    assert_eq!(resp.status, 404);

    // Too many
    let resp = router.dispatch(&make_request("GET", "/a/b/c"));
    assert_eq!(resp.status, 404);
}
