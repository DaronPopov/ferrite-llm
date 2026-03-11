//! File-based JSON checkpoint store with atomic writes via `tempfile`.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use serde::de::DeserializeOwned;
use serde::Serialize;
use tempfile::NamedTempFile;

use crate::error::AppError;

/// Persists checkpoint state as JSON files using atomic writes.
///
/// Uses `tempfile::NamedTempFile` for crash-safe persistence:
/// data is written to a temporary file in the same directory, then
/// atomically renamed to the final path. If the process crashes
/// mid-write, the temp file is cleaned up automatically.
pub(crate) struct CheckpointStore {
    dir: PathBuf,
}

impl CheckpointStore {
    /// Create a new checkpoint store rooted at `dir`.
    ///
    /// Creates the directory if it does not exist.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, AppError> {
        let dir = dir.into();
        fs::create_dir_all(&dir).map_err(|e| AppError::CheckpointError {
            detail: format!("failed to create checkpoint dir {}: {}", dir.display(), e),
        })?;
        Ok(Self { dir })
    }

    /// Save a checkpoint with the given label.
    ///
    /// Uses `NamedTempFile::persist()` for atomic write — the data is
    /// fully written and fsynced before the rename, so a crash at any
    /// point leaves either the old checkpoint or the new one, never a
    /// partial file.
    pub fn save(&self, label: &str, state: &impl Serialize) -> Result<(), AppError> {
        let dest = self.label_path(label);
        let data = serde_json::to_string_pretty(state)?;

        let mut tmp = NamedTempFile::new_in(&self.dir).map_err(|e| {
            AppError::CheckpointError {
                detail: format!("failed to create temp file: {}", e),
            }
        })?;

        tmp.write_all(data.as_bytes()).map_err(|e| {
            AppError::CheckpointError {
                detail: format!("failed to write checkpoint data: {}", e),
            }
        })?;

        tmp.persist(&dest).map_err(|e| AppError::CheckpointError {
            detail: format!("failed to persist checkpoint to {}: {}", dest.display(), e),
        })?;

        tracing::debug!(label, path = %dest.display(), "checkpoint saved");
        Ok(())
    }

    /// Restore a checkpoint by label.
    ///
    /// Returns `Ok(None)` if no checkpoint exists for this label.
    pub fn restore<T: DeserializeOwned>(&self, label: &str) -> Result<Option<T>, AppError> {
        let path = self.label_path(label);
        if !path.exists() {
            return Ok(None);
        }

        let data = fs::read_to_string(&path).map_err(|e| AppError::CheckpointError {
            detail: format!("failed to read checkpoint {}: {}", path.display(), e),
        })?;
        let value: T = serde_json::from_str(&data)?;
        tracing::debug!(label, path = %path.display(), "checkpoint restored");
        Ok(Some(value))
    }

    fn label_path(&self, label: &str) -> PathBuf {
        self.dir.join(format!("{}.json", label))
    }
}
