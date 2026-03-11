//! Platform-neutral IPC abstraction over Unix domain sockets / Windows named pipes.

use std::io::{self, Read, Write};
use std::net::Shutdown;
use std::path::Path;
use std::time::Duration;

// ── Endpoint ────────────────────────────────────────────────────────────

/// A platform-neutral IPC endpoint address.
///
/// On Unix this is a filesystem path to a domain socket.
/// On Windows this will be a named-pipe path (e.g. `\\.\pipe\ferrite-daemon`).
#[derive(Debug, Clone)]
pub struct Endpoint {
    /// Raw address string — socket path on Unix, pipe name on Windows.
    addr: String,
}

impl Endpoint {
    pub fn new(addr: impl Into<String>) -> Self {
        Self { addr: addr.into() }
    }

    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Return as a filesystem [`Path`] (meaningful on Unix; on Windows the
    /// named-pipe path is not a real filesystem entry, but callers can still
    /// use this for `exists()` checks where appropriate).
    pub fn as_path(&self) -> &Path {
        Path::new(&self.addr)
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.addr)
    }
}

// ── IpcStream ───────────────────────────────────────────────────────────

/// A connected IPC stream (client-side or accepted from a listener).
pub struct IpcStream {
    #[cfg(unix)]
    inner: std::os::unix::net::UnixStream,
    #[cfg(windows)]
    inner: std::net::TcpStream, // placeholder — real impl uses named pipes
}

impl IpcStream {
    /// Connect to a daemon endpoint.
    pub fn connect(endpoint: &Endpoint) -> io::Result<Self> {
        #[cfg(unix)]
        {
            let stream = std::os::unix::net::UnixStream::connect(&endpoint.addr)?;
            Ok(Self { inner: stream })
        }
        #[cfg(windows)]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Windows named-pipe connect not yet implemented",
            ))
        }
    }

    pub fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.inner.set_read_timeout(dur)
    }

    pub fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> {
        self.inner.set_write_timeout(dur)
    }

    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        self.inner.set_nonblocking(nonblocking)
    }

    pub fn shutdown(&self, how: Shutdown) -> io::Result<()> {
        #[cfg(unix)]
        {
            self.inner.shutdown(how)
        }
        #[cfg(windows)]
        {
            let _ = how;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Windows named-pipe shutdown not yet implemented",
            ))
        }
    }
}

impl Read for IpcStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl Write for IpcStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

// ── IpcListener ─────────────────────────────────────────────────────────

/// A platform-neutral IPC listener (server side).
pub struct IpcListener {
    #[cfg(unix)]
    inner: std::os::unix::net::UnixListener,
    #[cfg(windows)]
    _placeholder: (),
}

impl IpcListener {
    /// Bind a new listener to the given endpoint.
    ///
    /// On Unix, removes any stale socket file and creates directories as needed.
    pub fn bind(endpoint: &Endpoint) -> io::Result<Self> {
        #[cfg(unix)]
        {
            let path = Path::new(&endpoint.addr);

            // Check for live daemon before removing stale socket.
            if path.exists() {
                if std::os::unix::net::UnixStream::connect(path).is_ok() {
                    return Err(io::Error::new(
                        io::ErrorKind::AddrInUse,
                        "Daemon already running",
                    ));
                }
                std::fs::remove_file(path)?;
            }

            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let listener = std::os::unix::net::UnixListener::bind(&endpoint.addr)?;
            Ok(Self { inner: listener })
        }
        #[cfg(windows)]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Windows named-pipe listener not yet implemented",
            ))
        }
    }

    /// Set the listener to non-blocking mode.
    pub fn set_nonblocking(&self, nonblocking: bool) -> io::Result<()> {
        #[cfg(unix)]
        {
            self.inner.set_nonblocking(nonblocking)
        }
        #[cfg(windows)]
        {
            let _ = nonblocking;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Windows named-pipe set_nonblocking not yet implemented",
            ))
        }
    }

    /// Accept a new connection.  Returns `WouldBlock` in non-blocking mode
    /// when no connection is pending.
    pub fn accept(&self) -> io::Result<IpcStream> {
        #[cfg(unix)]
        {
            let (stream, _addr) = self.inner.accept()?;
            Ok(IpcStream { inner: stream })
        }
        #[cfg(windows)]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Windows named-pipe accept not yet implemented",
            ))
        }
    }
}

/// Remove the socket/pipe endpoint from the filesystem (Unix only; no-op on Windows).
pub fn remove_endpoint(endpoint: &Endpoint) {
    let _ = std::fs::remove_file(&endpoint.addr);
}

/// Check whether an endpoint address exists on disk (Unix socket file check).
pub fn endpoint_exists(endpoint: &Endpoint) -> bool {
    Path::new(&endpoint.addr).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_display_and_path() {
        let ep = Endpoint::new("/tmp/test.sock");
        assert_eq!(ep.addr(), "/tmp/test.sock");
        assert_eq!(ep.as_path(), Path::new("/tmp/test.sock"));
        assert_eq!(format!("{}", ep), "/tmp/test.sock");
    }

    #[cfg(unix)]
    #[test]
    fn bind_connect_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("test.sock");
        let ep = Endpoint::new(sock.to_str().unwrap());

        let listener = IpcListener::bind(&ep).unwrap();
        listener.set_nonblocking(true).unwrap();

        let mut client = IpcStream::connect(&ep).unwrap();
        client.write_all(b"hello").unwrap();
        client.shutdown(Shutdown::Write).unwrap();

        let mut accepted = listener.accept().unwrap();
        let mut buf = String::new();
        accepted.read_to_string(&mut buf).unwrap();
        assert_eq!(buf, "hello");
    }
}
