use std::collections::HashMap;
use std::process::Child;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::Instant;

use ptx_runtime::PtxRuntime;
use ptx_runtime::scheduler::Scheduler;

use crate::config::DaemonConfig;
use crate::event_stream::SchedulerEventStream;
use crate::policy::engine::PolicyEngine;
use crate::supervisor::JobSupervisor;

pub const MANAGED_APPS: &[&str] = &[
    "vram_database",
    "neural_fabric",
    "stream_compute",
    "checkpoint_engine",
    "gpu_nas",
];

pub struct DaemonState {
    pub runtime: Arc<PtxRuntime>,
    pub config: DaemonConfig,
    pub start_time: Instant,
    pub active_clients: AtomicU64,
    pub total_requests: AtomicU64,
    pub failed_requests: AtomicU64,
    pub running: AtomicBool,
    pub apps: parking_lot::Mutex<HashMap<u64, ManagedApp>>,
    pub next_app_id: AtomicU64,
    pub next_run_id: AtomicU64,

    // ── control plane (Plan-C) ──────────────────────────────────
    /// Policy enforcement engine (protected by Mutex for interior mutability).
    pub policy_engine: parking_lot::Mutex<PolicyEngine>,
    /// Machine-readable event stream.
    pub event_stream: parking_lot::Mutex<SchedulerEventStream>,
    /// Whether the scheduler queue is paused.
    pub scheduler_paused: AtomicBool,

    // ── durable job supervisor (Plan-B) ─────────────────────────
    /// Job supervisor managing durable jobs.
    pub supervisor: parking_lot::Mutex<JobSupervisor>,

    // ── multi-tenant scheduler (Plan-A) ─────────────────────────
    /// Multi-tenant GPU scheduler with fair dispatch and quota enforcement.
    pub scheduler: parking_lot::Mutex<Scheduler>,
}

pub struct ManagedApp {
    pub id: u64,
    pub name: String,
    pub args: Vec<String>,
    pub started_at: Instant,
    pub child: Child,
}

#[derive(Debug, Clone)]
pub struct DaemonHealthDiagnostic {
    pub component: &'static str,
    pub status: &'static str,
    pub code: &'static str,
    pub summary: String,
    pub remediation: &'static str,
}

impl DaemonState {
    pub fn new(runtime: Arc<PtxRuntime>, config: DaemonConfig, supervisor: JobSupervisor) -> Self {
        let cp = &config.control_plane;
        let policy_engine = PolicyEngine::with_mode(cp.audit_max_entries, &cp.default_policy);
        let mut event_stream = SchedulerEventStream::new(cp.event_stream_buffer);
        let sched_config = config.scheduler.to_runtime_config();
        let scheduler = Scheduler::new(sched_config);

        // Emit pool init marker to establish single authoritative pool ownership.
        {
            let tlsf = runtime.tlsf_stats();
            let stats = runtime.stats();
            event_stream.emit(crate::event_stream::SchedulerEvent::DaemonPoolInit {
                pool_size_bytes: tlsf.total_pool_size as u64,
                pool_fraction: config.pool_fraction,
                max_streams: config.max_streams,
                device_id: config.device_id,
            });
            let _ = stats; // used only for the emit above
        }

        Self {
            runtime,
            config,
            start_time: Instant::now(),
            active_clients: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            running: AtomicBool::new(true),
            apps: parking_lot::Mutex::new(HashMap::new()),
            next_app_id: AtomicU64::new(1),
            next_run_id: AtomicU64::new(1),
            policy_engine: parking_lot::Mutex::new(policy_engine),
            event_stream: parking_lot::Mutex::new(event_stream),
            scheduler_paused: AtomicBool::new(false),
            supervisor: parking_lot::Mutex::new(supervisor),
            scheduler: parking_lot::Mutex::new(scheduler),
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub fn shutdown(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    pub fn health_diagnostic(&self) -> DaemonHealthDiagnostic {
        let failed = self.failed_requests.load(Ordering::Relaxed);
        let total = self.total_requests.load(Ordering::Relaxed);
        if failed > 0 {
            DaemonHealthDiagnostic {
                component: "daemon.state",
                status: "WARN",
                code: "DMN-STATE-0001",
                summary: format!("failed_requests={} total_requests={}", failed, total),
                remediation: "inspect recent daemon logs and client command failures",
            }
        } else {
            DaemonHealthDiagnostic {
                component: "daemon.state",
                status: "PASS",
                code: "DMN-STATE-0002",
                summary: format!("failed_requests=0 total_requests={}", total),
                remediation: "none",
            }
        }
    }
}

pub struct ClientGuard<'a> {
    state: &'a DaemonState,
}

impl<'a> ClientGuard<'a> {
    pub fn new(state: &'a DaemonState) -> Self {
        Self { state }
    }
}

impl<'a> Drop for ClientGuard<'a> {
    fn drop(&mut self) {
        self.state.active_clients.fetch_sub(1, Ordering::Relaxed);
    }
}
