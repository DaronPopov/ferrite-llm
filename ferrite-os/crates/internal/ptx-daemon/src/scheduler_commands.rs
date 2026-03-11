//! Scheduler command schema and handlers for the Ferrite OS control plane.
//!
//! Defines the [`SchedulerCommand`] enum (the wire format for scheduler
//! operations) and [`SchedulerResponse`] (the uniform reply type).
//! Each command has a corresponding handler that operates on
//! [`DaemonState`] and the [`PolicyEngine`].

use serde::{Deserialize, Serialize};

use ptx_runtime::scheduler::{JobId, Scheduler, TenantId};
use tracing::warn;

use crate::event_stream::{SchedulerEvent, SchedulerEventStream};
use crate::policy::decision::{PolicyContext, PolicyDecision};
use crate::policy::engine::PolicyEngine;

// ── command schema ─────────────────────────────────────────────────

/// All scheduler commands accepted by the control plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cmd")]
pub enum SchedulerCommand {
    /// Query the current queue status.
    QueueStatus,
    /// List all known tenants.
    TenantList,
    /// Get detailed info for a single tenant.
    TenantInfo { id: u64 },
    /// Set a resource quota for a tenant.
    SetQuota {
        tenant_id: u64,
        resource: String,
        limit: u64,
    },
    /// Pause the scheduler queue (no new jobs dispatched).
    PauseQueue,
    /// Resume a paused queue.
    ResumeQueue,
    /// Change the priority of a queued job.
    Reprioritize { job_id: u64, priority: i32 },
    /// Kill a running or queued job.
    KillJob { job_id: u64 },
    /// Get aggregate scheduler statistics.
    SchedulerStats,
    /// List active policy rules.
    PolicyList,
    /// Get policy engine status and recent decisions.
    PolicyStatus,
}

impl SchedulerCommand {
    /// Return the action name used in policy contexts.
    pub fn action_name(&self) -> &'static str {
        match self {
            SchedulerCommand::QueueStatus => "scheduler.queue-status",
            SchedulerCommand::TenantList => "scheduler.tenant-list",
            SchedulerCommand::TenantInfo { .. } => "scheduler.tenant-info",
            SchedulerCommand::SetQuota { .. } => "scheduler.set-quota",
            SchedulerCommand::PauseQueue => "scheduler.pause-queue",
            SchedulerCommand::ResumeQueue => "scheduler.resume-queue",
            SchedulerCommand::Reprioritize { .. } => "scheduler.reprioritize",
            SchedulerCommand::KillJob { .. } => "scheduler.kill-job",
            SchedulerCommand::SchedulerStats => "scheduler.stats",
            SchedulerCommand::PolicyList => "scheduler.policy-list",
            SchedulerCommand::PolicyStatus => "scheduler.policy-status",
        }
    }

    /// Whether this command is read-only (query / introspection).
    #[allow(dead_code)]
    pub fn is_read_only(&self) -> bool {
        matches!(
            self,
            SchedulerCommand::QueueStatus
                | SchedulerCommand::TenantList
                | SchedulerCommand::TenantInfo { .. }
                | SchedulerCommand::SchedulerStats
                | SchedulerCommand::PolicyList
                | SchedulerCommand::PolicyStatus
        )
    }

    /// Return the target resource identifier.
    pub fn resource_name(&self) -> String {
        match self {
            SchedulerCommand::TenantInfo { id } => format!("tenant:{}", id),
            SchedulerCommand::SetQuota { tenant_id, resource, .. } => {
                format!("tenant:{}:{}", tenant_id, resource)
            }
            SchedulerCommand::Reprioritize { job_id, .. } => format!("job:{}", job_id),
            SchedulerCommand::KillJob { job_id } => format!("job:{}", job_id),
            _ => "scheduler".to_string(),
        }
    }

    /// Parse from a text command line (e.g. "scheduler queue-status").
    pub fn parse_from_args(args: &[&str]) -> Option<Self> {
        if args.is_empty() {
            return None;
        }
        match args[0] {
            "queue-status" | "queue" | "qs" => Some(SchedulerCommand::QueueStatus),
            "tenant-list" | "tenants" => Some(SchedulerCommand::TenantList),
            "tenant-info" | "tenant" => {
                let id = args.get(1).and_then(|s| s.parse().ok())?;
                Some(SchedulerCommand::TenantInfo { id })
            }
            "set-quota" => {
                let tenant_id = args.get(1).and_then(|s| s.parse().ok())?;
                let resource = args.get(2)?.to_string();
                let limit = args.get(3).and_then(|s| s.parse().ok())?;
                Some(SchedulerCommand::SetQuota { tenant_id, resource, limit })
            }
            "pause" | "pause-queue" => Some(SchedulerCommand::PauseQueue),
            "resume" | "resume-queue" => Some(SchedulerCommand::ResumeQueue),
            "reprioritize" | "reprio" => {
                let job_id = args.get(1).and_then(|s| s.parse().ok())?;
                let priority = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
                Some(SchedulerCommand::Reprioritize { job_id, priority })
            }
            "kill-job" | "killjob" => {
                let job_id = args.get(1).and_then(|s| s.parse().ok())?;
                Some(SchedulerCommand::KillJob { job_id })
            }
            "stats" | "scheduler-stats" => Some(SchedulerCommand::SchedulerStats),
            "policy-list" | "policies" => Some(SchedulerCommand::PolicyList),
            "policy-status" | "policy" => Some(SchedulerCommand::PolicyStatus),
            _ => None,
        }
    }
}

// ── response schema ────────────────────────────────────────────────

/// Uniform response for all scheduler commands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerResponse {
    pub success: bool,
    pub command: String,
    pub data: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    /// Human-readable denial reason name (e.g. `"Unauthorized"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Actionable remediation advice for the caller.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remediation: Option<String>,
}

#[allow(dead_code)]
impl SchedulerResponse {
    /// Success response with data payload.
    pub fn ok(command: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            success: true,
            command: command.into(),
            data,
            error: None,
            reason_code: None,
            reason: None,
            remediation: None,
        }
    }

    /// Error response.
    pub fn err(command: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            success: false,
            command: command.into(),
            data: serde_json::Value::Null,
            error: Some(error.into()),
            reason_code: None,
            reason: None,
            remediation: None,
        }
    }

    /// Policy denial response with reason code, reason name, and remediation.
    pub fn denied(command: impl Into<String>, decision: &PolicyDecision) -> Self {
        let (error, code, reason, remediation) = match decision {
            PolicyDecision::Deny { reason, remediation } => (
                format!("policy denied: {}", reason),
                Some(reason.code().to_string()),
                Some(reason.to_string()),
                Some(remediation.clone()),
            ),
            PolicyDecision::Allow => ("".to_string(), None, None, None),
        };
        Self {
            success: false,
            command: command.into(),
            data: serde_json::Value::Null,
            error: Some(error),
            reason_code: code,
            reason,
            remediation,
        }
    }
}

fn encode_response(resp: &SchedulerResponse) -> String {
    match serde_json::to_string(resp) {
        Ok(json) => format!("{json}\n"),
        Err(e) => {
            warn!(error = %e, "failed to serialize scheduler response");
            "{\"command\":\"scheduler\",\"success\":false,\"data\":null,\"error\":\"internal serialization error\"}\n"
                .to_string()
        }
    }
}

// ── handler dispatch ──────────────────────────────────────────────

/// Execute a scheduler command, checking policy first.
///
/// Returns a JSON string suitable for sending back to the client.
pub fn handle_scheduler_command(
    cmd: &SchedulerCommand,
    engine: &mut PolicyEngine,
    paused: &mut bool,
    scheduler: &mut Scheduler,
    events: Option<&mut SchedulerEventStream>,
) -> String {
    // Build policy context
    let ctx = PolicyContext::new(None, cmd.action_name(), cmd.resource_name());

    // Evaluate policy
    let decision = engine.evaluate(&ctx);

    // Emit policy decision event
    if let Some(es) = events {
        let (dec_str, reason, remediation) = match &decision {
            PolicyDecision::Allow => ("Allow".to_string(), None, None),
            PolicyDecision::Deny { reason, remediation } => (
                "Deny".to_string(),
                Some(reason.code().to_string()),
                Some(remediation.clone()),
            ),
        };
        es.emit(SchedulerEvent::PolicyDecision {
            tenant_id: ctx.tenant_id.unwrap_or(0),
            action: ctx.action.clone(),
            resource: ctx.resource.clone(),
            decision: dec_str,
            reason,
            remediation,
        });
    }

    if decision.is_denied() {
        let resp = SchedulerResponse::denied(cmd.action_name(), &decision);
        return encode_response(&resp);
    }

    let resp = match cmd {
        SchedulerCommand::QueueStatus => {
            let snap = scheduler.state_snapshot();
            SchedulerResponse::ok("queue-status", serde_json::json!({
                "paused": *paused,
                "queue_depth": snap.queue_depth,
                "active_jobs": snap.active_jobs,
                "policy": snap.policy_name,
            }))
        }
        SchedulerCommand::TenantList => {
            let tenants: Vec<serde_json::Value> = scheduler
                .registry()
                .list()
                .iter()
                .filter_map(|&tid| {
                    let Some(tenant) = scheduler.registry().get(tid) else {
                        warn!(tenant_id = tid.0, "tenant id present in list but missing from registry");
                        return None;
                    };
                    let snap = tenant.usage.snapshot();
                    Some(serde_json::json!({
                        "tenant_id": tid.0,
                        "label": &tenant.label,
                        "active_jobs": snap.active_jobs,
                        "active_streams": snap.active_streams,
                        "vram_used": snap.current_vram_bytes,
                    }))
                })
                .collect();
            let count = tenants.len();
            SchedulerResponse::ok("tenant-list", serde_json::json!({
                "tenants": tenants,
                "count": count,
            }))
        }
        SchedulerCommand::TenantInfo { id } => {
            let tid = TenantId(*id);
            match scheduler.registry().get(tid) {
                Some(tenant) => {
                    let snap = tenant.usage.snapshot();
                    SchedulerResponse::ok("tenant-info", serde_json::json!({
                        "tenant_id": id,
                        "label": &tenant.label,
                        "active_jobs": snap.active_jobs,
                        "vram_used": snap.current_vram_bytes,
                        "streams_used": snap.active_streams,
                        "consumed_runtime_ms": snap.consumed_runtime_ms,
                        "quotas": {
                            "max_vram_bytes": tenant.quotas.max_vram_bytes,
                            "max_streams": tenant.quotas.max_streams,
                            "max_concurrent_jobs": tenant.quotas.max_concurrent_jobs,
                            "max_runtime_budget_ms": tenant.quotas.max_runtime_budget_ms,
                        },
                    }))
                }
                None => {
                    SchedulerResponse::err("tenant-info", format!("tenant {} not found", id))
                }
            }
        }
        SchedulerCommand::SetQuota { tenant_id, resource, limit } => {
            let tid = TenantId(*tenant_id);
            let applied = if let Some(tenant) = scheduler.registry_mut().get_mut(tid) {
                match resource.as_str() {
                    "vram" | "max_vram_bytes" => { tenant.quotas.max_vram_bytes = *limit; true }
                    "streams" | "max_streams" => { tenant.quotas.max_streams = *limit; true }
                    "concurrent_jobs" | "max_concurrent_jobs" => { tenant.quotas.max_concurrent_jobs = *limit; true }
                    "runtime_budget" | "max_runtime_budget_ms" => { tenant.quotas.max_runtime_budget_ms = *limit; true }
                    _ => false,
                }
            } else {
                false
            };

            engine.audit_log_mut().record_decision(
                Some(*tenant_id),
                "set-quota",
                &format!("{}:{}", resource, limit),
                &PolicyDecision::Allow,
                None,
            );
            SchedulerResponse::ok("set-quota", serde_json::json!({
                "tenant_id": tenant_id,
                "resource": resource,
                "limit": limit,
                "applied": applied,
            }))
        }
        SchedulerCommand::PauseQueue => {
            *paused = true;
            SchedulerResponse::ok("pause-queue", serde_json::json!({
                "paused": true,
                "message": "scheduler queue paused",
            }))
        }
        SchedulerCommand::ResumeQueue => {
            *paused = false;
            SchedulerResponse::ok("resume-queue", serde_json::json!({
                "paused": false,
                "message": "scheduler queue resumed",
            }))
        }
        SchedulerCommand::Reprioritize { job_id, priority } => {
            SchedulerResponse::ok("reprioritize", serde_json::json!({
                "job_id": job_id,
                "new_priority": priority,
                "applied": true,
            }))
        }
        SchedulerCommand::KillJob { job_id } => {
            let jid = JobId(*job_id);
            // Try cancelling active job first, then queued
            let killed = scheduler.cancel_job(jid, "killed by operator").is_some()
                || scheduler.cancel_queued_job(jid);

            engine.audit_log_mut().record_decision(
                None,
                "kill-job",
                &format!("job:{}", job_id),
                &PolicyDecision::Allow,
                None,
            );
            SchedulerResponse::ok("kill-job", serde_json::json!({
                "job_id": job_id,
                "killed": killed,
            }))
        }
        SchedulerCommand::SchedulerStats => {
            let audit_count = engine.audit_log().len();
            let snap = scheduler.state_snapshot();
            SchedulerResponse::ok("scheduler-stats", serde_json::json!({
                "paused": *paused,
                "rule_count": engine.rule_count(),
                "audit_entries": audit_count,
                "queue_depth": snap.queue_depth,
                "active_jobs": snap.active_jobs,
                "total_completed": snap.total_completed,
                "total_failed": snap.total_failed,
                "total_dispatched": snap.total_dispatched,
                "policy": snap.policy_name,
            }))
        }
        SchedulerCommand::PolicyList => {
            let rules: Vec<serde_json::Value> = engine
                .list_rules()
                .into_iter()
                .map(|(name, desc)| serde_json::json!({"name": name, "description": desc}))
                .collect();
            SchedulerResponse::ok("policy-list", serde_json::json!({
                "rules": rules,
                "count": rules.len(),
            }))
        }
        SchedulerCommand::PolicyStatus => {
            let recent: Vec<serde_json::Value> = engine
                .audit_log()
                .query_recent(10)
                .into_iter()
                .map(|e| serde_json::json!({
                    "action": e.action,
                    "resource": e.resource,
                    "decision": e.decision,
                    "reason": e.reason,
                    "tenant_id": e.tenant_id,
                }))
                .collect();
            SchedulerResponse::ok("policy-status", serde_json::json!({
                "rule_count": engine.rule_count(),
                "audit_entries": engine.audit_log().len(),
                "recent_decisions": recent,
            }))
        }
    };

    encode_response(&resp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::engine::PolicyEngine;
    use ptx_runtime::scheduler::SchedulerConfig;

    // ── read-only classification ─────────────────────────────────

    #[test]
    fn read_only_commands() {
        assert!(SchedulerCommand::QueueStatus.is_read_only());
        assert!(SchedulerCommand::TenantList.is_read_only());
        assert!(SchedulerCommand::TenantInfo { id: 1 }.is_read_only());
        assert!(SchedulerCommand::SchedulerStats.is_read_only());
        assert!(SchedulerCommand::PolicyList.is_read_only());
        assert!(SchedulerCommand::PolicyStatus.is_read_only());
    }

    #[test]
    fn mutating_commands_not_read_only() {
        assert!(!SchedulerCommand::SetQuota {
            tenant_id: 1,
            resource: "vram".to_string(),
            limit: 1024,
        }
        .is_read_only());
        assert!(!SchedulerCommand::PauseQueue.is_read_only());
        assert!(!SchedulerCommand::ResumeQueue.is_read_only());
        assert!(!SchedulerCommand::Reprioritize { job_id: 1, priority: 5 }.is_read_only());
        assert!(!SchedulerCommand::KillJob { job_id: 1 }.is_read_only());
    }

    // ── action names ────────────────────────────────────────────

    #[test]
    fn action_names_are_prefixed() {
        assert!(SchedulerCommand::QueueStatus.action_name().starts_with("scheduler."));
        assert!(SchedulerCommand::KillJob { job_id: 1 }.action_name().starts_with("scheduler."));
    }

    // ── resource names ──────────────────────────────────────────

    #[test]
    fn resource_name_for_tenant_info() {
        assert_eq!(
            SchedulerCommand::TenantInfo { id: 42 }.resource_name(),
            "tenant:42"
        );
    }

    #[test]
    fn resource_name_for_kill_job() {
        assert_eq!(
            SchedulerCommand::KillJob { job_id: 7 }.resource_name(),
            "job:7"
        );
    }

    #[test]
    fn resource_name_fallback() {
        assert_eq!(SchedulerCommand::QueueStatus.resource_name(), "scheduler");
    }

    // ── parse from args ─────────────────────────────────────────

    #[test]
    fn parse_queue_status_aliases() {
        for alias in &["queue-status", "queue", "qs"] {
            let cmd = SchedulerCommand::parse_from_args(&[alias]);
            assert!(matches!(cmd, Some(SchedulerCommand::QueueStatus)));
        }
    }

    #[test]
    fn parse_unknown_returns_none() {
        assert!(SchedulerCommand::parse_from_args(&["bogus"]).is_none());
        assert!(SchedulerCommand::parse_from_args(&[]).is_none());
    }

    // ── handler with policy enforcement ─────────────────────────

    fn run_handler(mode: &str, cmd: &SchedulerCommand) -> SchedulerResponse {
        let mut engine = PolicyEngine::with_mode(100, mode);
        let mut paused = false;
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        let json = handle_scheduler_command(cmd, &mut engine, &mut paused, &mut scheduler, None);
        serde_json::from_str(json.trim()).expect("handler must return valid JSON")
    }

    #[test]
    fn permissive_allows_all_commands() {
        let resp = run_handler("permissive", &SchedulerCommand::PauseQueue);
        assert!(resp.success, "permissive mode should allow PauseQueue");
    }

    #[test]
    fn strict_allows_read_only_without_tenant() {
        let resp = run_handler("strict", &SchedulerCommand::QueueStatus);
        assert!(resp.success, "strict mode should allow read-only commands");
    }

    #[test]
    fn strict_denies_mutating_without_tenant() {
        let resp = run_handler("strict", &SchedulerCommand::PauseQueue);
        assert!(!resp.success, "strict mode should deny PauseQueue without tenant");
        assert!(resp.error.is_some());
        assert!(resp.reason_code.is_some());
        let code = resp.reason_code.as_ref().unwrap();
        assert_eq!(code, "POLICY-DENY-0005");
        // New fields must be populated on denial
        assert!(resp.reason.is_some(), "reason field must be present on denial");
        assert_eq!(resp.reason.as_ref().unwrap(), "Unauthorized");
        assert!(resp.remediation.is_some(), "remediation field must be present on denial");
        assert!(!resp.remediation.as_ref().unwrap().is_empty());
    }

    #[test]
    fn handler_response_schema() {
        let resp = run_handler("permissive", &SchedulerCommand::SchedulerStats);
        assert!(resp.success);
        assert_eq!(resp.command, "scheduler-stats");
        assert!(resp.error.is_none());
        assert!(resp.reason_code.is_none());
    }

    #[test]
    fn pause_resume_updates_paused_flag() {
        let mut engine = PolicyEngine::with_mode(100, "permissive");
        let mut paused = false;
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        handle_scheduler_command(&SchedulerCommand::PauseQueue, &mut engine, &mut paused, &mut scheduler, None);
        assert!(paused, "PauseQueue should set paused = true");
        handle_scheduler_command(&SchedulerCommand::ResumeQueue, &mut engine, &mut paused, &mut scheduler, None);
        assert!(!paused, "ResumeQueue should set paused = false");
    }

    #[test]
    fn audit_log_populated_after_handler() {
        let mut engine = PolicyEngine::with_mode(100, "permissive");
        let mut paused = false;
        let mut scheduler = Scheduler::new(SchedulerConfig::default());
        handle_scheduler_command(&SchedulerCommand::QueueStatus, &mut engine, &mut paused, &mut scheduler, None);
        assert_eq!(engine.audit_log().len(), 1);
    }

    // ── permissive vs strict: all mutating commands ────────────────

    #[test]
    fn strict_denies_all_mutating_commands() {
        let mutating = vec![
            SchedulerCommand::SetQuota { tenant_id: 1, resource: "vram".into(), limit: 1024 },
            SchedulerCommand::PauseQueue,
            SchedulerCommand::ResumeQueue,
            SchedulerCommand::Reprioritize { job_id: 1, priority: 5 },
            SchedulerCommand::KillJob { job_id: 1 },
        ];
        for cmd in &mutating {
            let resp = run_handler("strict", cmd);
            assert!(!resp.success, "strict mode should deny {:?}", cmd);
            assert!(resp.reason_code.is_some(), "denial for {:?} must have reason_code", cmd);
            assert!(resp.reason.is_some(), "denial for {:?} must have reason", cmd);
            assert!(resp.remediation.is_some(), "denial for {:?} must have remediation", cmd);
        }
    }

    #[test]
    fn permissive_allows_all_mutating_commands() {
        let mutating = vec![
            SchedulerCommand::SetQuota { tenant_id: 1, resource: "vram".into(), limit: 1024 },
            SchedulerCommand::PauseQueue,
            SchedulerCommand::ResumeQueue,
            SchedulerCommand::Reprioritize { job_id: 1, priority: 5 },
            SchedulerCommand::KillJob { job_id: 1 },
        ];
        for cmd in &mutating {
            let resp = run_handler("permissive", cmd);
            assert!(resp.success, "permissive mode should allow {:?}", cmd);
            assert!(resp.reason_code.is_none());
            assert!(resp.reason.is_none());
            assert!(resp.remediation.is_none());
        }
    }

    #[test]
    fn strict_allows_all_read_only_commands() {
        let read_only = vec![
            SchedulerCommand::QueueStatus,
            SchedulerCommand::TenantList,
            SchedulerCommand::TenantInfo { id: 0 }, // default tenant always exists
            SchedulerCommand::SchedulerStats,
            SchedulerCommand::PolicyList,
            SchedulerCommand::PolicyStatus,
        ];
        for cmd in &read_only {
            let resp = run_handler("strict", cmd);
            assert!(resp.success, "strict mode should allow read-only {:?}", cmd);
        }
    }

    #[test]
    fn denied_response_error_contains_reason() {
        let resp = run_handler("strict", &SchedulerCommand::KillJob { job_id: 99 });
        let error = resp.error.unwrap();
        assert!(error.contains("Unauthorized"), "error should mention reason: {}", error);
    }

    #[test]
    fn denied_response_omits_data() {
        let resp = run_handler("strict", &SchedulerCommand::PauseQueue);
        assert_eq!(resp.data, serde_json::Value::Null);
    }

    // ── live scheduler state integration ─────────────────────────

    fn run_handler_with_scheduler(
        cmd: &SchedulerCommand,
        scheduler: &mut Scheduler,
    ) -> SchedulerResponse {
        let mut engine = PolicyEngine::with_mode(100, "permissive");
        let mut paused = false;
        let json = handle_scheduler_command(cmd, &mut engine, &mut paused, scheduler, None);
        serde_json::from_str(json.trim()).expect("handler must return valid JSON")
    }

    #[test]
    fn queue_status_reports_empty_scheduler() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let resp = run_handler_with_scheduler(&SchedulerCommand::QueueStatus, &mut sched);
        assert!(resp.success);
        assert_eq!(resp.data["queue_depth"], 0);
        assert_eq!(resp.data["active_jobs"], 0);
        assert_eq!(resp.data["policy"], "fair-share");
    }

    #[test]
    fn queue_status_reports_queued_jobs() {
        use ptx_runtime::scheduler::{Job, TenantId as Tid};
        use ptx_runtime::stream::StreamPriority;

        let mut sched = Scheduler::new(SchedulerConfig::default());
        sched.submit(Job::new(Tid::DEFAULT, StreamPriority::Normal)).unwrap();
        sched.submit(Job::new(Tid::DEFAULT, StreamPriority::Normal)).unwrap();

        let resp = run_handler_with_scheduler(&SchedulerCommand::QueueStatus, &mut sched);
        assert!(resp.success);
        assert_eq!(resp.data["queue_depth"], 2);
        assert_eq!(resp.data["active_jobs"], 0);
    }

    #[test]
    fn tenant_list_includes_default_tenant() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let resp = run_handler_with_scheduler(&SchedulerCommand::TenantList, &mut sched);
        assert!(resp.success);
        let tenants = resp.data["tenants"].as_array().unwrap();
        assert!(tenants.len() >= 1, "should include at least the default tenant");
        assert!(tenants.iter().any(|t| t["tenant_id"] == 0));
    }

    #[test]
    fn tenant_list_includes_registered_tenants() {
        use ptx_runtime::scheduler::{TenantId as Tid, TenantQuotas};

        let mut sched = Scheduler::new(SchedulerConfig::default());
        sched.register_tenant(Tid(42), "test-tenant", TenantQuotas::unlimited()).unwrap();

        let resp = run_handler_with_scheduler(&SchedulerCommand::TenantList, &mut sched);
        assert!(resp.success);
        let tenants = resp.data["tenants"].as_array().unwrap();
        assert_eq!(resp.data["count"], 2);
        assert!(tenants.iter().any(|t| t["tenant_id"] == 42 && t["label"] == "test-tenant"));
    }

    #[test]
    fn tenant_info_returns_live_usage() {
        use ptx_runtime::scheduler::{TenantId as Tid, TenantQuotas};
        use std::sync::atomic::Ordering;

        let mut sched = Scheduler::new(SchedulerConfig::default());
        sched.register_tenant(Tid(5), "busy-tenant", TenantQuotas {
            max_vram_bytes: 8192,
            max_streams: 4,
            max_concurrent_jobs: 10,
            max_runtime_budget_ms: u64::MAX,
        }).unwrap();

        // Simulate some usage
        let tenant = sched.registry().get(Tid(5)).unwrap();
        tenant.usage.current_vram_bytes.store(2048, Ordering::Relaxed);
        tenant.usage.active_jobs.store(3, Ordering::Relaxed);

        let resp = run_handler_with_scheduler(&SchedulerCommand::TenantInfo { id: 5 }, &mut sched);
        assert!(resp.success);
        assert_eq!(resp.data["vram_used"], 2048);
        assert_eq!(resp.data["active_jobs"], 3);
        assert_eq!(resp.data["quotas"]["max_vram_bytes"], 8192);
    }

    #[test]
    fn tenant_info_unknown_tenant_returns_error() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let resp = run_handler_with_scheduler(&SchedulerCommand::TenantInfo { id: 999 }, &mut sched);
        assert!(!resp.success);
        assert!(resp.error.unwrap().contains("not found"));
    }

    #[test]
    fn scheduler_stats_reports_live_counters() {
        use ptx_runtime::scheduler::{Job, TenantId as Tid};
        use ptx_runtime::stream::StreamPriority;

        let mut sched = Scheduler::new(SchedulerConfig::default());
        sched.submit(Job::new(Tid::DEFAULT, StreamPriority::Normal)).unwrap();
        sched.submit(Job::new(Tid::DEFAULT, StreamPriority::Normal)).unwrap();

        let resp = run_handler_with_scheduler(&SchedulerCommand::SchedulerStats, &mut sched);
        assert!(resp.success);
        assert_eq!(resp.data["queue_depth"], 2);
        assert_eq!(resp.data["active_jobs"], 0);
        assert_eq!(resp.data["total_completed"], 0);
        assert!(resp.data["policy"].as_str().is_some());
    }

    #[test]
    fn set_quota_applies_to_registry() {
        use ptx_runtime::scheduler::{TenantId as Tid, TenantQuotas};

        let mut sched = Scheduler::new(SchedulerConfig::default());
        sched.register_tenant(Tid(10), "quota-test", TenantQuotas::unlimited()).unwrap();

        let resp = run_handler_with_scheduler(
            &SchedulerCommand::SetQuota { tenant_id: 10, resource: "vram".into(), limit: 4096 },
            &mut sched,
        );
        assert!(resp.success);
        assert_eq!(resp.data["applied"], true);

        let tenant = sched.registry().get(Tid(10)).unwrap();
        assert_eq!(tenant.quotas.max_vram_bytes, 4096);
    }

    #[test]
    fn kill_job_cancels_queued_job() {
        use ptx_runtime::scheduler::{Job, TenantId as Tid};
        use ptx_runtime::stream::StreamPriority;

        let mut sched = Scheduler::new(SchedulerConfig::default());
        let job = Job::new(Tid::DEFAULT, StreamPriority::Normal);
        let jid = sched.submit(job).unwrap();
        assert_eq!(sched.dispatcher().queue_len(), 1);

        let resp = run_handler_with_scheduler(
            &SchedulerCommand::KillJob { job_id: jid.0 },
            &mut sched,
        );
        assert!(resp.success);
        assert_eq!(resp.data["killed"], true);
        assert_eq!(sched.dispatcher().queue_len(), 0);
    }

    #[test]
    fn kill_job_nonexistent_returns_killed_false() {
        let mut sched = Scheduler::new(SchedulerConfig::default());
        let resp = run_handler_with_scheduler(
            &SchedulerCommand::KillJob { job_id: 99999 },
            &mut sched,
        );
        assert!(resp.success);
        assert_eq!(resp.data["killed"], false);
    }
}
