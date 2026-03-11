use std::io::{Read, Write};
use std::net::Shutdown;
use std::os::unix::net::{UnixListener, UnixStream};
use std::thread;

/// Test that our IPC framing matches the daemon protocol:
/// write command + newline, shutdown(Write) for EOF, read response.
#[test]
fn ipc_framing_write_shutdown_read() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("test.sock");
    let sock_path_str = sock_path.to_str().unwrap().to_string();

    let listener = UnixListener::bind(&sock_path).unwrap();

    // Server thread: accept, read command, send response
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = String::new();
        stream.read_to_string(&mut buf).unwrap();
        assert_eq!(buf.trim(), "ping");

        let response = r#"{"ok":true,"message":"pong"}"#;
        stream.write_all(response.as_bytes()).unwrap();
        stream.write_all(b"\n").unwrap();
    });

    // Client: use the same framing as our ipc_send
    let result = ferrite_connectors::sink::ferrite_ipc::ipc_send(&sock_path_str, "ping");
    assert!(result.is_ok(), "ipc_send failed: {:?}", result.err());
    let resp = result.unwrap();
    assert!(resp.contains("pong"), "response should contain pong: {}", resp);

    server.join().unwrap();
}

/// Test that server reads complete command before responding.
#[test]
fn ipc_framing_multi_word_command() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("test2.sock");
    let sock_path_str = sock_path.to_str().unwrap().to_string();

    let listener = UnixListener::bind(&sock_path).unwrap();

    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = String::new();
        stream.read_to_string(&mut buf).unwrap();
        assert_eq!(buf.trim(), "app-start my-app --flag");

        let response = r#"{"ok":true,"message":"app started"}"#;
        stream.write_all(response.as_bytes()).unwrap();
        stream.write_all(b"\n").unwrap();
    });

    let result =
        ferrite_connectors::sink::ferrite_ipc::ipc_send(&sock_path_str, "app-start my-app --flag");
    assert!(result.is_ok());
    let resp = result.unwrap();
    assert!(resp.contains("app started"));

    server.join().unwrap();
}

/// Test that manual client framing matches expected protocol.
#[test]
fn manual_framing_matches_ipc_send() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("test3.sock");

    let listener = UnixListener::bind(&sock_path).unwrap();

    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = String::new();
        stream.read_to_string(&mut buf).unwrap();
        // Verify the framing: command + newline
        assert_eq!(buf, "status\n");
        stream
            .write_all(b"{\"ok\":true,\"healthy\":true}\n")
            .unwrap();
    });

    // Manual client: replicate framing
    let mut stream = UnixStream::connect(&sock_path).unwrap();
    stream.write_all(b"status\n").unwrap();
    stream.shutdown(Shutdown::Write).unwrap();
    let mut resp = String::new();
    stream.read_to_string(&mut resp).unwrap();
    assert!(resp.contains("healthy"));

    server.join().unwrap();
}
