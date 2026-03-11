use std::net::{SocketAddr, ToSocketAddrs};

/// Resolve a hostname:port string to all socket addresses.
pub fn resolve(host: &str) -> anyhow::Result<Vec<SocketAddr>> {
    let addrs: Vec<SocketAddr> = host
        .to_socket_addrs()
        .map_err(|e| anyhow::anyhow!("DNS resolution failed for {}: {}", host, e))?
        .collect();
    if addrs.is_empty() {
        anyhow::bail!("DNS resolution returned no addresses for {}", host);
    }
    Ok(addrs)
}

/// Resolve and return just the first address.
pub fn resolve_one(host: &str) -> anyhow::Result<SocketAddr> {
    resolve(host).map(|mut addrs| addrs.remove(0))
}
