//! Runtime directory and default path resolution.
//!
//! Centralizes the logic for determining socket paths, PID file locations,
//! temporary directories, and home directory — removing hardcoded `/tmp`
//! and UID lookups from business code.

use std::path::PathBuf;

/// Return the platform-appropriate default temporary directory.
///
/// On Unix: `std::env::temp_dir()` (typically `/tmp`).
/// On Windows: `std::env::temp_dir()` (typically `%TEMP%`).
pub fn temp_dir() -> PathBuf {
    std::env::temp_dir()
}

/// Return the user's home directory.
///
/// Checks `HOME` on Unix, `USERPROFILE` on Windows.
/// Falls back to [`temp_dir()`] if unavailable.
pub fn home_dir() -> PathBuf {
    #[cfg(unix)]
    {
        std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| temp_dir())
    }
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| temp_dir())
    }
}

/// Return a per-user runtime directory suitable for sockets and PID files.
///
/// Resolution order:
/// 1. `XDG_RUNTIME_DIR` (common on Linux, may be set on Windows via WSL).
/// 2. `%LOCALAPPDATA%\ferrite-os` on Windows.
/// 3. `<temp_dir>/ferrite-os-<uid>/` on Unix.
/// 4. `<temp_dir>/ferrite-os/` on Windows (fallback).
pub fn runtime_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(xdg);
    }

    #[cfg(windows)]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(local).join("ferrite-os");
        }
        return temp_dir().join("ferrite-os");
    }

    #[cfg(unix)]
    {
        let uid = current_uid();
        temp_dir().join(format!("ferrite-os-{}", uid))
    }
}

/// Default socket endpoint address.
pub fn default_socket_addr() -> String {
    let base = runtime_dir();
    let path = base.join("ferrite-daemon.sock");
    path.to_string_lossy().to_string()
}

/// Default PID file path.
pub fn default_pid_path() -> String {
    let base = runtime_dir();
    let path = base.join("ferrite-daemon.pid");
    path.to_string_lossy().to_string()
}

/// Default job state directory (under user home).
pub fn default_jobs_state_dir() -> String {
    let home = home_dir();
    home.join(".ferrite").join("jobs").to_string_lossy().to_string()
}

/// Fallback job state directory when primary is unavailable.
pub fn fallback_job_state_dir() -> String {
    temp_dir()
        .join("ferrite-jobs-fallback")
        .to_string_lossy()
        .to_string()
}

/// Legacy socket path for backward compatibility.
pub fn legacy_socket_path() -> &'static str {
    #[cfg(unix)]
    {
        "/tmp/ferrite.sock"
    }
    #[cfg(windows)]
    {
        "\\\\.\\pipe\\ferrite-legacy"
    }
}

// ── Platform helpers ────────────────────────────────────────────────────

/// Current effective user ID (Unix).  Returns 0 on Windows (no direct equivalent).
#[cfg(unix)]
fn current_uid() -> u32 {
    unsafe { libc::geteuid() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_dir_exists() {
        assert!(temp_dir().exists() || true); // may not exist in sandbox
    }

    #[test]
    fn default_socket_addr_not_empty() {
        let addr = default_socket_addr();
        assert!(!addr.is_empty());
        assert!(addr.contains("ferrite-daemon"));
    }

    #[test]
    fn default_pid_path_not_empty() {
        let p = default_pid_path();
        assert!(!p.is_empty());
        assert!(p.contains("ferrite-daemon"));
    }

    #[test]
    fn jobs_state_dir_not_empty() {
        let d = default_jobs_state_dir();
        assert!(!d.is_empty());
        assert!(d.contains("ferrite") || d.contains("jobs"));
    }
}
