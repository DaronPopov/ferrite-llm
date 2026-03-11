use ferrite_connectors::net::tcp::{tcp_connect, tcp_recv, tcp_send, TcpServer};
use std::io::{Read, Write};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn echo_server() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let mut server = TcpServer::start(
        "127.0.0.1:0",
        |mut stream| {
            let mut buf = [0u8; 1024];
            while let Ok(n) = stream.read(&mut buf) {
                if n == 0 {
                    break;
                }
                stream.write_all(&buf[..n]).ok();
            }
        },
        shutdown.clone(),
    )
    .unwrap();

    let addr = server.addr().to_string();
    let mut client = tcp_connect(&addr, 5000).unwrap();
    tcp_send(&mut client, b"hello").unwrap();
    thread::sleep(Duration::from_millis(100));
    let data = tcp_recv(&mut client, 1024).unwrap();
    assert_eq!(&data, b"hello");

    server.stop();
}

#[test]
fn multiple_clients() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let mut server = TcpServer::start(
        "127.0.0.1:0",
        |mut stream| {
            let mut buf = [0u8; 1024];
            if let Ok(n) = stream.read(&mut buf) {
                stream.write_all(&buf[..n]).ok();
            }
        },
        shutdown.clone(),
    )
    .unwrap();

    let addr = server.addr().to_string();
    for i in 0..5 {
        let msg = format!("msg{}", i);
        let mut client = tcp_connect(&addr, 5000).unwrap();
        tcp_send(&mut client, msg.as_bytes()).unwrap();
        thread::sleep(Duration::from_millis(100));
        let data = tcp_recv(&mut client, 1024).unwrap();
        assert_eq!(data, msg.as_bytes());
    }

    server.stop();
}

#[test]
fn server_addr_is_bound() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let mut server =
        TcpServer::start("127.0.0.1:0", |_| {}, shutdown.clone()).unwrap();
    assert_ne!(server.addr().port(), 0);
    server.stop();
}

#[test]
fn server_shutdown() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let mut server =
        TcpServer::start("127.0.0.1:0", |_| {}, shutdown.clone()).unwrap();
    let addr = server.addr();

    server.stop();

    // After stop, connecting should fail (port is released)
    thread::sleep(Duration::from_millis(100));
    let result = std::net::TcpStream::connect_timeout(
        &addr,
        Duration::from_millis(200),
    );
    assert!(result.is_err());
}
