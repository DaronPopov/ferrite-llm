//! Dynamic library environment variable abstraction.
//!
//! Centralizes the platform-specific env var name used for the dynamic
//! linker search path (`LD_LIBRARY_PATH` on Linux, `PATH` on Windows).

use std::env;
use std::path::Path;
use std::process::Command;

/// Environment variable name for the dynamic library search path.
///
/// - Linux: `LD_LIBRARY_PATH`
/// - macOS: `DYLD_LIBRARY_PATH`
/// - Windows: `PATH`
pub const DYLIB_PATH_VAR: &str = {
    #[cfg(target_os = "linux")]
    {
        "LD_LIBRARY_PATH"
    }
    #[cfg(target_os = "macos")]
    {
        "DYLD_LIBRARY_PATH"
    }
    #[cfg(target_os = "windows")]
    {
        "PATH"
    }
};

/// Path separator for the dynamic library search path variable.
pub const DYLIB_PATH_SEP: char = {
    #[cfg(unix)]
    {
        ':'
    }
    #[cfg(windows)]
    {
        ';'
    }
};

/// Read the current value of the dynamic library search path variable.
pub fn get_dylib_path() -> String {
    env::var(DYLIB_PATH_VAR).unwrap_or_default()
}

/// Set the dynamic library search path on a [`Command`], prepending `dirs`
/// to whatever is already in the environment.
///
/// Deduplicates and skips non-existent directories.
pub fn apply_dylib_path(cmd: &mut Command, dirs: &[impl AsRef<Path>]) {
    let existing = get_dylib_path();
    let mut parts: Vec<String> = Vec::new();

    for dir in dirs {
        push_unique_existing(&mut parts, dir.as_ref());
    }

    for segment in existing.split(DYLIB_PATH_SEP) {
        let s = segment.trim();
        if !s.is_empty() && !parts.iter().any(|v| v == s) {
            parts.push(s.to_string());
        }
    }

    if !parts.is_empty() {
        let sep = String::from(DYLIB_PATH_SEP);
        cmd.env(DYLIB_PATH_VAR, parts.join(&sep));
    }
}

/// Propagate the current dynamic library env to a [`Command`] (pass-through).
pub fn propagate_dylib_path(cmd: &mut Command) {
    if let Ok(val) = env::var(DYLIB_PATH_VAR) {
        cmd.env(DYLIB_PATH_VAR, val);
    }
}

fn push_unique_existing(out: &mut Vec<String>, path: &Path) {
    if !path.exists() {
        return;
    }
    let val = std::fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .to_string();
    if !out.iter().any(|v| v == &val) {
        out.push(val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dylib_path_var_is_not_empty() {
        assert!(!DYLIB_PATH_VAR.is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_uses_ld_library_path() {
        assert_eq!(DYLIB_PATH_VAR, "LD_LIBRARY_PATH");
        assert_eq!(DYLIB_PATH_SEP, ':');
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_uses_path() {
        assert_eq!(DYLIB_PATH_VAR, "PATH");
        assert_eq!(DYLIB_PATH_SEP, ';');
    }
}
