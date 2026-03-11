use ferrite_connectors::net::dns;

#[test]
fn resolve_localhost() {
    let addrs = dns::resolve("localhost:80").unwrap();
    assert!(!addrs.is_empty());
}

#[test]
fn resolve_one_localhost() {
    let addr = dns::resolve_one("localhost:80").unwrap();
    assert!(addr.ip().is_loopback());
}

#[test]
fn resolve_ip_literal() {
    let addrs = dns::resolve("127.0.0.1:9090").unwrap();
    assert_eq!(addrs.len(), 1);
    assert_eq!(addrs[0].port(), 9090);
    assert!(addrs[0].ip().is_loopback());
}

#[test]
fn resolve_one_ip_literal() {
    let addr = dns::resolve_one("127.0.0.1:443").unwrap();
    assert_eq!(addr.port(), 443);
}

#[test]
fn resolve_invalid_fails() {
    // Empty string should fail
    assert!(dns::resolve("").is_err());
}

#[test]
fn resolve_no_port_fails() {
    // Missing port should fail for ToSocketAddrs
    assert!(dns::resolve("localhost").is_err());
}
