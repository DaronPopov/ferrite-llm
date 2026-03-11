use std::path::{Path, PathBuf};

use super::guard_path;

/// Create a directory (and parents) within the workspace.
pub fn ws_mkdir(root: &Path, cwd: &Path, target: &str) -> Result<String, String> {
    let path = guard_path(root, target, cwd)?;
    std::fs::create_dir_all(&path).map_err(|e| format!("mkdir failed: {}", e))?;
    Ok(format!("created {}", short(root, &path)))
}

/// Create an empty file (touch) within the workspace.
pub fn ws_touch(root: &Path, cwd: &Path, target: &str) -> Result<String, String> {
    let path = guard_path(root, target, cwd)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("touch failed: {}", e))?;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| format!("touch failed: {}", e))?;
    Ok(format!("touched {}", short(root, &path)))
}

/// Move/rename a file or directory within the workspace.
pub fn ws_mv(root: &Path, cwd: &Path, src: &str, dst: &str) -> Result<String, String> {
    let src_path = guard_path(root, src, cwd)?;
    let dst_path = guard_path(root, dst, cwd)?;
    if !src_path.exists() {
        return Err(format!("source not found: {}", src));
    }
    if let Some(parent) = dst_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mv failed: {}", e))?;
    }
    std::fs::rename(&src_path, &dst_path).map_err(|e| format!("mv failed: {}", e))?;
    Ok(format!(
        "moved {} -> {}",
        short(root, &src_path),
        short(root, &dst_path)
    ))
}

/// Copy a file within the workspace.
pub fn ws_cp(root: &Path, cwd: &Path, src: &str, dst: &str) -> Result<String, String> {
    let src_path = guard_path(root, src, cwd)?;
    let dst_path = guard_path(root, dst, cwd)?;
    if !src_path.exists() {
        return Err(format!("source not found: {}", src));
    }
    if src_path.is_dir() {
        return Err("cp: directory copy not supported (use mkdir + cp for files)".into());
    }
    if let Some(parent) = dst_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("cp failed: {}", e))?;
    }
    std::fs::copy(&src_path, &dst_path).map_err(|e| format!("cp failed: {}", e))?;
    Ok(format!(
        "copied {} -> {}",
        short(root, &src_path),
        short(root, &dst_path)
    ))
}

/// Remove a file or directory. Requires prior confirmation via PendingConfirm.
pub fn ws_rm(root: &Path, cwd: &Path, target: &str, confirmed: bool) -> Result<RmResult, String> {
    let path = guard_path(root, target, cwd)?;
    if !path.exists() {
        return Err(format!("not found: {}", target));
    }

    if !confirmed {
        let desc = if path.is_dir() {
            format!("remove directory '{}' and all contents?", short(root, &path))
        } else {
            format!("remove '{}'?", short(root, &path))
        };
        return Ok(RmResult::NeedsConfirm(desc, path));
    }

    if path.is_dir() {
        std::fs::remove_dir_all(&path).map_err(|e| format!("rm failed: {}", e))?;
    } else {
        std::fs::remove_file(&path).map_err(|e| format!("rm failed: {}", e))?;
    }
    Ok(RmResult::Removed(format!("removed {}", short(root, &path))))
}

pub enum RmResult {
    NeedsConfirm(String, PathBuf),
    Removed(String),
}

/// List directory contents within the workspace.
pub fn ws_ls(root: &Path, cwd: &Path, target: Option<&str>) -> Result<Vec<String>, String> {
    let path = match target {
        Some(t) => guard_path(root, t, cwd)?,
        None => guard_path(root, ".", cwd)?,
    };
    if !path.is_dir() {
        return Err(format!("not a directory: {}", short(root, &path)));
    }
    let mut entries = Vec::new();
    let rd = std::fs::read_dir(&path).map_err(|e| format!("ls failed: {}", e))?;
    for entry in rd.flatten() {
        let name = entry
            .file_name()
            .to_string_lossy()
            .to_string();
        let is_dir = entry.path().is_dir();
        entries.push(if is_dir {
            format!("{}/", name)
        } else {
            name
        });
    }
    entries.sort();
    Ok(entries)
}

/// Change current working directory within the workspace.
pub fn ws_cd(root: &Path, cwd: &Path, target: &str) -> Result<PathBuf, String> {
    let path = guard_path(root, target, cwd)?;
    if !path.is_dir() {
        return Err(format!("not a directory: {}", target));
    }
    Ok(path)
}

/// Return current working directory relative to workspace root.
pub fn ws_pwd(root: &Path, cwd: &Path) -> String {
    cwd.strip_prefix(root)
        .map(|p| {
            if p.as_os_str().is_empty() {
                "/".to_string()
            } else {
                format!("/{}", p.display())
            }
        })
        .unwrap_or_else(|_| cwd.display().to_string())
}

fn short(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}
