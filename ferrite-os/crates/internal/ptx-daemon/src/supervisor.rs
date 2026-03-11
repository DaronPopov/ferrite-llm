//! Job supervisor -- owns the durable job store and manages active jobs.
//!
//! The supervisor is the single authority for job lifecycle management inside
//! the daemon. It exposes a `tick()` method that the daemon event loop calls
//! periodically to drive restart logic and health checks.

use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use ptx_runtime::error::{Error, Result};
use ptx_runtime::job::{
    DurableJob, DurableJobId, JobLifecycleState, RestartDecision,
};
use ptx_runtime::telemetry::{DiagnosticEvent, DiagnosticStatus};

use crate::job_store::{JobReconciler, JobStore};

// ---------------------------------------------------------------------------
// SupervisorAction
// ---------------------------------------------------------------------------

/// Actions emitted by a supervisor tick for the daemon to act on.
#[allow(dead_code)]
#[derive(Debug)]
pub enum SupervisorAction {
    /// A job should be started (first time or retry).
    StartJob(DurableJobId),
    /// A job should be restarted after the given delay.
    RestartJob(DurableJobId, Duration),
    /// A job has been marked as permanently failed.
    MarkFailed(DurableJobId),
    /// A job has been marked as succeeded.
    MarkSucceeded(DurableJobId),
    /// Emit a diagnostic event to the TUI / logs.
    EmitDiagnostic(DiagnosticEvent),
}

// ---------------------------------------------------------------------------
// PendingRestart
// ---------------------------------------------------------------------------

/// Tracks a deferred restart (waiting for backoff delay to elapse).
#[allow(dead_code)]
struct PendingRestart {
    ready_at: Instant,
}

// ---------------------------------------------------------------------------
// JobSupervisor
// ---------------------------------------------------------------------------

/// Central supervisor for durable jobs.
pub struct JobSupervisor {
    store: JobStore,
    jobs: HashMap<DurableJobId, DurableJob>,
    pending_restarts: HashMap<DurableJobId, PendingRestart>,
}

#[allow(dead_code)]
impl JobSupervisor {
    /// Create an empty supervisor with no recovered jobs (fallback path).
    pub fn empty(store: JobStore) -> Self {
        Self {
            store,
            jobs: HashMap::new(),
            pending_restarts: HashMap::new(),
        }
    }

    /// Create a new supervisor and run boot-time reconciliation.
    ///
    /// Returns the supervisor and a list of diagnostic events produced
    /// during reconciliation.
    pub fn new(store: JobStore) -> Result<(Self, Vec<DiagnosticEvent>)> {
        let (recovered_jobs, diagnostics) = JobReconciler::reconcile(&store)?;

        let mut jobs = HashMap::new();
        let mut pending_restarts: HashMap<DurableJobId, PendingRestart> = HashMap::new();

        for mut job in recovered_jobs {
            // If a recovered job is in Failed state and has a restart policy,
            // evaluate whether it should be restarted.
            if job.state() == JobLifecycleState::Failed {
                // Create a synthetic error for the restart policy evaluation.
                let synth_error = Error::Internal {
                    message: job
                        .last_failure
                        .clone()
                        .unwrap_or_else(|| "recovered after daemon restart".into()),
                };
                match job.should_restart(&synth_error) {
                    Ok(RestartDecision::Restart { delay }) => {
                        tracing::info!(
                            job_id = job.id.raw(),
                            delay_ms = delay.as_millis(),
                            "scheduling recovered job for restart"
                        );
                        pending_restarts.insert(
                            job.id,
                            PendingRestart {
                                ready_at: Instant::now() + delay,
                            },
                        );
                        store.save(&job)?;
                    }
                    Ok(RestartDecision::GiveUp { reason }) => {
                        tracing::info!(
                            job_id = job.id.raw(),
                            reason = %reason,
                            "recovered job will not be restarted"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            job_id = job.id.raw(),
                            error = %e,
                            "failed to evaluate restart for recovered job"
                        );
                    }
                }
            }
            jobs.insert(job.id, job);
        }

        Ok((
            Self {
                store,
                jobs,
                pending_restarts,
            },
            diagnostics,
        ))
    }

    // -- public API -----------------------------------------------------

    /// Submit a new job. It is immediately enqueued and persisted.
    pub fn submit(&mut self, mut job: DurableJob) -> Result<DurableJobId> {
        let id = job.id;
        job.enqueue()?;
        self.store.save(&job)?;
        self.jobs.insert(id, job);

        tracing::info!(job_id = id.raw(), "job submitted");
        Ok(id)
    }

    /// Stop (cancel) a running or pending job.
    pub fn stop(&mut self, id: DurableJobId, reason: String) -> Result<()> {
        let job = self
            .jobs
            .get_mut(&id)
            .ok_or(Error::JobNotFound { id: id.raw() })?;

        // If the job has a running process, kill it.
        if let Some(pid) = job.process_pid {
            tracing::info!(job_id = id.raw(), pid, "killing job process");
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
        }

        job.cancel(reason)?;
        self.store.save(job)?;
        self.pending_restarts.remove(&id);

        tracing::info!(job_id = id.raw(), "job stopped");
        Ok(())
    }

    /// Stop all active (non-terminal) jobs. Returns the number of jobs stopped.
    pub fn stop_all(&mut self, reason: &str) -> usize {
        let active_ids: Vec<DurableJobId> = self
            .jobs
            .iter()
            .filter(|(_, job)| !job.state().is_terminal())
            .map(|(id, _)| *id)
            .collect();
        let count = active_ids.len();
        for id in active_ids {
            if let Err(e) = self.stop(id, reason.to_string()) {
                tracing::warn!(job_id = id.raw(), error = %e, "failed to stop job in stop_all");
            }
        }
        count
    }

    /// Look up a job by ID.
    pub fn status(&self, id: DurableJobId) -> Option<&DurableJob> {
        self.jobs.get(&id)
    }

    /// List all known jobs (active and terminal).
    pub fn list(&self) -> Vec<&DurableJob> {
        self.jobs.values().collect()
    }

    /// The main tick function. Called periodically by the daemon event loop.
    ///
    /// 1. Check for pending restarts whose delay has elapsed -> start them.
    /// 2. For Running jobs, poll process liveness.
    /// 3. For Queued jobs, start them.
    pub fn tick(&mut self) -> Vec<SupervisorAction> {
        let mut actions = Vec::new();

        // Collect IDs to avoid borrowing conflicts.
        let job_ids: Vec<DurableJobId> = self.jobs.keys().copied().collect();

        for id in job_ids {
            // -- handle pending restarts --
            if let Some(pending) = self.pending_restarts.get(&id) {
                if Instant::now() >= pending.ready_at {
                    self.pending_restarts.remove(&id);
                    match self.start_job(id) {
                        Ok(()) => {
                            actions.push(SupervisorAction::StartJob(id));
                            actions.push(SupervisorAction::EmitDiagnostic(
                                DiagnosticEvent::new(
                                    "daemon.supervisor",
                                    DiagnosticStatus::PASS,
                                    "JOB-SUP-0001",
                                    format!("restarted job {}", id),
                                    "none",
                                ),
                            ));
                        }
                        Err(e) => {
                            tracing::error!(job_id = id.raw(), error = %e, "failed to restart job");
                            actions.push(SupervisorAction::EmitDiagnostic(
                                DiagnosticEvent::new(
                                    "daemon.supervisor",
                                    DiagnosticStatus::FAIL,
                                    "JOB-SUP-0002",
                                    format!("failed to restart job {}: {}", id, e),
                                    "check job command and system resources",
                                ),
                            ));
                        }
                    }
                }
                continue;
            }

            let job = match self.jobs.get(&id) {
                Some(j) => j,
                None => continue,
            };

            match job.state() {
                // -- start queued jobs --
                JobLifecycleState::Queued => {
                    match self.start_job(id) {
                        Ok(()) => {
                            actions.push(SupervisorAction::StartJob(id));
                            actions.push(SupervisorAction::EmitDiagnostic(
                                DiagnosticEvent::new(
                                    "daemon.supervisor",
                                    DiagnosticStatus::PASS,
                                    "JOB-SUP-0003",
                                    format!("started job {}", id),
                                    "none",
                                ),
                            ));
                        }
                        Err(e) => {
                            tracing::error!(job_id = id.raw(), error = %e, "failed to start queued job");
                        }
                    }
                }

                // -- poll running jobs --
                JobLifecycleState::Running => {
                    if let Some(pid) = job.process_pid {
                        if !Self::is_process_alive(pid) {
                            // Process died. Determine exit status.
                            self.handle_process_exit(id, &mut actions);
                        }
                    }
                }

                // -- retrying jobs that are not yet in pending_restarts --
                JobLifecycleState::Retrying => {
                    // If we somehow ended up in Retrying without a pending
                    // restart entry (e.g. after deserialization), schedule
                    // an immediate restart.
                    if !self.pending_restarts.contains_key(&id) {
                        self.pending_restarts.insert(
                            id,
                            PendingRestart {
                                ready_at: Instant::now(),
                            },
                        );
                    }
                }

                _ => {}
            }
        }

        actions
    }

    // -- internal helpers -----------------------------------------------

    /// Start or restart a job process.
    ///
    /// Enforces the no-duplicate-running invariant: if the job already has
    /// a live process, we refuse to start another.
    fn start_job(&mut self, id: DurableJobId) -> Result<()> {
        let job = self
            .jobs
            .get_mut(&id)
            .ok_or(Error::JobNotFound { id: id.raw() })?;

        // No-duplicate-running invariant.
        if let Some(pid) = job.process_pid {
            if Self::is_process_alive(pid) {
                tracing::warn!(
                    job_id = id.raw(),
                    pid,
                    "refusing to double-start job (process still alive)"
                );
                return Ok(());
            }
        }

        job.mark_starting()?;
        self.store.save(job)?;

        let child = Command::new(&job.command)
            .args(&job.args)
            .env("FERRITE_JOB_ID", id.raw().to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();

        match child {
            Ok(child) => {
                let pid = child.id();
                // We intentionally drop the `Child` handle -- the process
                // becomes an orphan managed by PID only. This avoids keeping
                // an open file descriptor per job.
                std::mem::forget(child);

                let job = self.jobs.get_mut(&id).ok_or_else(|| Error::Internal {
                    message: format!(
                        "job {} missing from registry during running transition",
                        id.raw()
                    ),
                })?;
                job.mark_running(pid)?;
                self.store.save(job)?;

                tracing::info!(job_id = id.raw(), pid, "job process started");
                Ok(())
            }
            Err(e) => {
                let msg = format!("spawn failed: {}", e);
                let job = self.jobs.get_mut(&id).ok_or_else(|| Error::Internal {
                    message: format!(
                        "job {} missing from registry during failure transition",
                        id.raw()
                    ),
                })?;
                job.record_failure(msg)?;
                self.store.save(job)?;
                Ok(())
            }
        }
    }

    /// Handle a Running job whose process has exited.
    fn handle_process_exit(
        &mut self,
        id: DurableJobId,
        actions: &mut Vec<SupervisorAction>,
    ) {
        let job = match self.jobs.get_mut(&id) {
            Some(j) => j,
            None => return,
        };

        // We cannot retrieve the exit code because we forgot the Child
        // handle. Treat any unexpected exit as a failure.
        let msg = "process exited unexpectedly".to_string();
        if let Err(e) = job.record_failure(msg) {
            tracing::error!(job_id = id.raw(), error = %e, "failed to record failure");
            return;
        }
        let _ = self.store.save(job);

        // Evaluate restart policy.
        let synth_error = Error::Internal {
            message: job
                .last_failure
                .clone()
                .unwrap_or_else(|| "process exited".into()),
        };
        match job.should_restart(&synth_error) {
            Ok(RestartDecision::Restart { delay }) => {
                tracing::info!(
                    job_id = id.raw(),
                    delay_ms = delay.as_millis(),
                    "scheduling job restart"
                );
                let _ = self.store.save(job);
                self.pending_restarts.insert(
                    id,
                    PendingRestart {
                        ready_at: Instant::now() + delay,
                    },
                );
                actions.push(SupervisorAction::RestartJob(id, delay));
            }
            Ok(RestartDecision::GiveUp { reason }) => {
                tracing::info!(
                    job_id = id.raw(),
                    reason = %reason,
                    "job restart policy exhausted"
                );
                actions.push(SupervisorAction::MarkFailed(id));
                actions.push(SupervisorAction::EmitDiagnostic(DiagnosticEvent::new(
                    "daemon.supervisor",
                    DiagnosticStatus::FAIL,
                    "JOB-SUP-0004",
                    format!("job {} permanently failed: {}", id, reason),
                    "review job configuration and restart policy",
                )));
            }
            Err(e) => {
                tracing::error!(
                    job_id = id.raw(),
                    error = %e,
                    "error evaluating restart policy"
                );
            }
        }
    }

    /// Check if a process is alive using `kill(pid, 0)`.
    fn is_process_alive(pid: u32) -> bool {
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }
}
