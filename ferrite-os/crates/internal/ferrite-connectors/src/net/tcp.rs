use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// A TCP server that spawns a thread per connection.
pub struct TcpServer {
    addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl TcpServer {
    /// Start a TCP server on the given bind address.
    /// The handler is called once per accepted connection.
    pub fn start<F>(bind: &str, handler: F, shutdown: Arc<AtomicBool>) -> anyhow::Result<Self>
    where
        F: Fn(TcpStream) + Send + Sync + 'static,
    {
        let listener = TcpListener::bind(bind)
            .map_err(|e| anyhow::anyhow!("TCP bind {}: {}", bind, e))?;
        let addr = listener.local_addr()?;
        listener.set_nonblocking(true)?;

        let shutdown_clone = shutdown.clone();
        let handler = Arc::new(handler);

        let handle = thread::Builder::new()
            .name(format!("tcp-server-{}", addr))
            .spawn(move || {
                tracing::info!(addr = %addr, "TCP server started");
                while !shutdown_clone.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, peer)) => {
                            tracing::debug!(peer = %peer, "TCP connection accepted");
                            let h = handler.clone();
                            thread::Builder::new()
                                .name(format!("tcp-conn-{}", peer))
                                .spawn(move || h(stream))
                                .ok();
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(50));
                        }
                        Err(e) => {
                            if !shutdown_clone.load(Ordering::Relaxed) {
                                tracing::error!(error = %e, "TCP accept error");
                            }
                            break;
                        }
                    }
                }
                tracing::info!("TCP server stopped");
            })
            .map_err(|e| anyhow::anyhow!("spawn TCP server: {}", e))?;

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

/// Connect to a TCP server with a timeout.
pub fn tcp_connect(addr: &str, timeout_ms: u64) -> anyhow::Result<TcpStream> {
    let socket_addr = addr
        .to_socket_addrs()
        .map_err(|e| anyhow::anyhow!("resolve {}: {}", addr, e))?
        .next()
        .ok_or_else(|| anyhow::anyhow!("no addresses for {}", addr))?;
    let stream = TcpStream::connect_timeout(&socket_addr, Duration::from_millis(timeout_ms))
        .map_err(|e| anyhow::anyhow!("TCP connect {}: {}", addr, e))?;
    Ok(stream)
}

/// Send data over a TCP stream.
pub fn tcp_send(stream: &mut TcpStream, data: &[u8]) -> anyhow::Result<()> {
    stream.write_all(data)?;
    stream.flush()?;
    Ok(())
}

/// Read data from a TCP stream into a buffer.
pub fn tcp_recv(stream: &mut TcpStream, buf_size: usize) -> anyhow::Result<Vec<u8>> {
    let mut buf = vec![0u8; buf_size];
    let n = stream.read(&mut buf)?;
    buf.truncate(n);
    Ok(buf)
}
