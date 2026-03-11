use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAX_CHECKPOINTS: usize = 10;

#[derive(Debug, Clone)]
pub struct CheckpointData {
    #[allow(dead_code)]
    pub label: String,
    #[allow(dead_code)]
    pub timestamp: Instant,
    pub file_snapshots: HashMap<PathBuf, Vec<u8>>,
}

/// Create a checkpoint of all agent-modified files.
pub fn create_checkpoint(
    label: &str,
    modified_files: &std::collections::HashSet<PathBuf>,
    checkpoints: &mut HashMap<String, CheckpointData>,
) -> Result<String, String> {
    if checkpoints.len() >= MAX_CHECKPOINTS && !checkpoints.contains_key(label) {
        return Err(format!(
            "max {} checkpoints reached — rollback or drop one first",
            MAX_CHECKPOINTS
        ));
    }

    let mut snapshots = HashMap::new();
    for path in modified_files {
        if path.exists() {
            let data = std::fs::read(path)
                .map_err(|e| format!("checkpoint read failed for {}: {}", path.display(), e))?;
            snapshots.insert(path.clone(), data);
        }
    }

    let count = snapshots.len();
    checkpoints.insert(
        label.to_string(),
        CheckpointData {
            label: label.to_string(),
            timestamp: Instant::now(),
            file_snapshots: snapshots,
        },
    );

    Ok(format!(
        "checkpoint '{}' created ({} files snapshotted)",
        label, count
    ))
}

/// Rollback to a previously created checkpoint.
pub fn rollback_checkpoint(
    label: &str,
    checkpoints: &mut HashMap<String, CheckpointData>,
    open_file: Option<&Path>,
) -> Result<(String, bool), String> {
    let data = checkpoints
        .remove(label)
        .ok_or_else(|| format!("no checkpoint named '{}'", label))?;

    let mut reload_needed = false;
    let mut restored = 0;

    for (path, content) in &data.file_snapshots {
        std::fs::write(path, content)
            .map_err(|e| format!("rollback write failed for {}: {}", path.display(), e))?;
        restored += 1;
        if open_file == Some(path.as_path()) {
            reload_needed = true;
        }
    }

    Ok((
        format!(
            "rolled back to '{}' ({} files restored)",
            label, restored
        ),
        reload_needed,
    ))
}
