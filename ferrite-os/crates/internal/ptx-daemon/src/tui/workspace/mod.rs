pub mod fs_ops;
pub mod tree;

use std::path::{Path, PathBuf};
use std::time::Instant;

/// Confirmation state for destructive operations.
#[derive(Debug, Clone)]
pub struct PendingConfirm {
    #[allow(dead_code)]
    pub action: String,
    pub target: PathBuf,
    pub timestamp: Instant,
}

impl PendingConfirm {
    pub fn new(action: impl Into<String>, target: PathBuf) -> Self {
        Self {
            action: action.into(),
            target,
            timestamp: Instant::now(),
        }
    }

    pub fn is_expired(&self) -> bool {
        self.timestamp.elapsed().as_secs() >= 30
    }
}

/// Canonicalize and guard a path so it never escapes the workspace root.
///
/// - Resolves `..`, `.`, and symlinks where possible.
/// - For paths that don't yet exist on disk, walks existing ancestors
///   and resolves the remainder lexically.
/// - Returns `Err(String)` with a human-readable message if the
///   resolved path is outside `workspace_root`.
pub fn guard_path(workspace_root: &Path, target: &str, cwd: &Path) -> Result<PathBuf, String> {
    let raw = Path::new(target);
    let base = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        cwd.join(raw)
    };

    // Try full canonicalization first (works if path exists).
    let resolved = if base.exists() {
        base.canonicalize()
            .map_err(|e| format!("cannot resolve path: {}", e))?
    } else {
        // Walk from the deepest existing ancestor upward.
        resolve_nonexistent(&base)?
    };

    let canon_root = workspace_root
        .canonicalize()
        .map_err(|e| format!("cannot resolve workspace root: {}", e))?;

    if resolved.starts_with(&canon_root) {
        Ok(resolved)
    } else {
        Err(format!(
            "path escapes workspace: {} is outside {}",
            resolved.display(),
            canon_root.display()
        ))
    }
}

/// Resolve a path that may not fully exist yet by canonicalizing the
/// longest existing prefix, then appending the remaining components
/// with `..` collapsed lexically.
fn resolve_nonexistent(path: &Path) -> Result<PathBuf, String> {
    let mut existing = path.to_path_buf();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();

    loop {
        if existing.exists() {
            break;
        }
        match existing.file_name() {
            Some(name) => {
                tail.push(name.to_os_string());
                existing.pop();
            }
            None => break,
        }
    }

    let mut resolved = if existing.exists() {
        existing
            .canonicalize()
            .map_err(|e| format!("cannot resolve path: {}", e))?
    } else {
        existing
    };

    for component in tail.into_iter().rev() {
        let s = component.to_string_lossy();
        if s == ".." {
            resolved.pop();
        } else if s != "." {
            resolved.push(component);
        }
    }

    Ok(resolved)
}
