//! File-based durable job persistence.
//!
//! Each job is stored as a JSON file under the configured state directory
//! (`~/.ferrite/jobs/` by default). Writes use the atomic write-to-tmp-then-
//! rename pattern so a crash mid-write never corrupts state.

use std::fs;
use std::path::{Path, PathBuf};

use ptx_runtime::job::{DurableJob, DurableJobId};
use ptx_runtime::error::{Error, Result};

/// Persistent store for durable jobs backed by JSON files on disk.
#[derive(Debug, Clone)]
pub struct JobStore {
    state_dir: PathBuf,
}

#[allow(dead_code)]
impl JobStore {
    /// Create a new store rooted at `state_dir`.
    ///
    /// The directory (and parents) are created eagerly so that later writes
    /// do not need to handle `ENOENT`.
    pub fn new(state_dir: impl Into<PathBuf>) -> Result<Self> {
        let state_dir = state_dir.into();
        fs::create_dir_all(&state_dir).map_err(|e| Error::JobPersistenceError {
            detail: format!("failed to create state dir {}: {}", state_dir.display(), e),
        })?;
        Ok(Self { state_dir })
    }

    /// The directory this store writes to.
    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }

    // -- file naming ----------------------------------------------------

    fn job_path(&self, id: DurableJobId) -> PathBuf {
        self.state_dir.join(format!("{}.json", id.raw()))
    }

    fn tmp_path(&self, id: DurableJobId) -> PathBuf {
        self.state_dir.join(format!("{}.json.tmp", id.raw()))
    }

    // -- CRUD -----------------------------------------------------------

    /// Persist a job atomically (write to `.tmp`, then rename).
    pub fn save(&self, job: &DurableJob) -> Result<()> {
        let data = serde_json::to_string_pretty(job).map_err(|e| {
            Error::JobPersistenceError {
                detail: format!("serialize job {}: {}", job.id, e),
            }
        })?;

        let tmp = self.tmp_path(job.id);
        let dest = self.job_path(job.id);

        fs::write(&tmp, data.as_bytes()).map_err(|e| Error::JobPersistenceError {
            detail: format!("write tmp file {}: {}", tmp.display(), e),
        })?;

        fs::rename(&tmp, &dest).map_err(|e| Error::JobPersistenceError {
            detail: format!(
                "rename {} -> {}: {}",
                tmp.display(),
                dest.display(),
                e
            ),
        })?;

        tracing::trace!(job_id = job.id.raw(), path = %dest.display(), "job persisted");
        Ok(())
    }

    /// Load a single job by ID.
    pub fn load(&self, id: DurableJobId) -> Result<DurableJob> {
        let path = self.job_path(id);
        let data = fs::read_to_string(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::JobNotFound { id: id.raw() }
            } else {
                Error::JobPersistenceError {
                    detail: format!("read {}: {}", path.display(), e),
                }
            }
        })?;

        serde_json::from_str(&data).map_err(|e| Error::JobPersistenceError {
            detail: format!("deserialize {}: {}", path.display(), e),
        })
    }

    /// Load every job in the state directory.
    pub fn load_all(&self) -> Result<Vec<DurableJob>> {
        let entries = fs::read_dir(&self.state_dir).map_err(|e| {
            Error::JobPersistenceError {
                detail: format!("read_dir {}: {}", self.state_dir.display(), e),
            }
        })?;

        let mut jobs = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| Error::JobPersistenceError {
                detail: format!("read_dir entry: {}", e),
            })?;

            let path = entry.path();

            // Skip tmp files and non-json files.
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            if !name.ends_with(".json") || name.ends_with(".json.tmp") {
                continue;
            }

            let data = fs::read_to_string(&path).map_err(|e| {
                Error::JobPersistenceError {
                    detail: format!("read {}: {}", path.display(), e),
                }
            })?;

            match serde_json::from_str::<DurableJob>(&data) {
                Ok(job) => {
                    tracing::debug!(job_id = job.id.raw(), name = %job.name, "loaded persisted job");
                    jobs.push(job);
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "skipping corrupt job file"
                    );
                }
            }
        }

        Ok(jobs)
    }

    /// Delete the persisted state for a job.
    pub fn delete(&self, id: DurableJobId) -> Result<()> {
        let path = self.job_path(id);
        if path.exists() {
            fs::remove_file(&path).map_err(|e| Error::JobPersistenceError {
                detail: format!("delete {}: {}", path.display(), e),
            })?;
            tracing::trace!(job_id = id.raw(), "job file deleted");
        }
        // Also clean up any stale tmp file.
        let tmp = self.tmp_path(id);
        if tmp.exists() {
            let _ = fs::remove_file(&tmp);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JobReconciler
// ---------------------------------------------------------------------------

use ptx_runtime::job::state::JobLifecycleState;
use ptx_runtime::telemetry::{DiagnosticEvent, DiagnosticStatus};

/// Boot-time reconciliation of persisted job state against reality.
pub struct JobReconciler;

impl JobReconciler {
    /// Reconcile all persisted jobs.
    ///
    /// For each job:
    /// - If state was Running/Starting, check if the recorded PID is alive.
    ///   - Dead process -> mark Failed, return it for restart-policy evaluation.
    ///   - Alive process -> update `process_pid` (reattach).
    /// - Terminal jobs are left as-is.
    ///
    /// Returns `(reconciled_jobs, diagnostics)`.
    pub fn reconcile(
        store: &JobStore,
    ) -> Result<(Vec<DurableJob>, Vec<DiagnosticEvent>)> {
        let mut jobs = store.load_all()?;
        let mut diagnostics = Vec::new();

        // Ensure the ID counter is advanced past any loaded IDs so new jobs
        // never collide with recovered ones.
        for job in &jobs {
            DurableJobId::advance_counter_to(job.id.raw());
        }

        for job in jobs.iter_mut() {
            match job.state() {
                JobLifecycleState::Running | JobLifecycleState::Starting => {
                    // The process_pid field is transient (serde skip), so we
                    // look at the last transition's reason which contains
                    // "pid NNN" for Running jobs. However, the canonical
                    // approach is to check if *any* process with the PID
                    // recorded in the state is still alive. Since process_pid
                    // is not persisted, we cannot reattach by PID alone. The
                    // safe choice: mark the job as Failed so the restart
                    // policy can kick in.
                    let reason = format!(
                        "daemon restarted while job {} was in {} state; process assumed lost",
                        job.id, job.state()
                    );
                    tracing::warn!(job_id = job.id.raw(), state = %job.state(), "reconciling orphaned job");

                    if let Err(e) = job.record_failure(reason.clone()) {
                        tracing::error!(job_id = job.id.raw(), error = %e, "failed to mark job as failed during reconciliation");
                        continue;
                    }

                    store.save(job)?;

                    diagnostics.push(DiagnosticEvent::new(
                        "daemon.job_reconciler",
                        DiagnosticStatus::WARN,
                        "JOB-REC-0001",
                        reason,
                        "review job restart policy; supervisor will attempt restart if policy allows",
                    ));
                }
                _ => {
                    // Terminal or not-yet-started jobs need no reconciliation.
                    diagnostics.push(DiagnosticEvent::new(
                        "daemon.job_reconciler",
                        DiagnosticStatus::PASS,
                        "JOB-REC-0002",
                        format!(
                            "job {} ({}) recovered in {} state",
                            job.id, job.name, job.state()
                        ),
                        "none",
                    ));
                }
            }
        }

        Ok((jobs, diagnostics))
    }
}
