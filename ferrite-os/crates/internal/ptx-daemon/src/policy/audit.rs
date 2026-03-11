//! Audit trail for control-plane actions.
//!
//! Every policy decision and scheduler command is recorded as an
//! [`AuditEntry`] in a bounded ring buffer.  Entries can be queried by
//! tenant, exported as JSON, and are also emitted as tracing events for
//! external log aggregation.

use std::collections::VecDeque;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tracing::info;

use super::decision::PolicyDecision;

/// A single audit record.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Wall-clock time (serialized as elapsed-since-boot seconds for JSON).
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    /// Elapsed seconds since daemon start (populated on serialization).
    #[serde(rename = "timestamp_secs")]
    pub elapsed_secs: f64,
    /// Tenant that initiated the action (0 = system / unauthenticated).
    pub tenant_id: u64,
    /// Human-readable action name.
    pub action: String,
    /// Target resource identifier.
    pub resource: String,
    /// Whether the action was allowed or denied.
    pub decision: String,
    /// Denial reason code (None when allowed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Source IP of the client (when available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_ip: Option<String>,
}

impl AuditEntry {
    /// Create a new entry from a policy decision.
    pub fn from_decision(
        tenant_id: Option<u64>,
        action: impl Into<String>,
        resource: impl Into<String>,
        decision: &PolicyDecision,
        source_ip: Option<String>,
        boot_time: Instant,
    ) -> Self {
        let now = Instant::now();
        let (dec_str, reason) = match decision {
            PolicyDecision::Allow => ("Allow".to_string(), None),
            PolicyDecision::Deny { reason, .. } => {
                ("Deny".to_string(), Some(reason.to_string()))
            }
        };
        Self {
            timestamp: now,
            elapsed_secs: now.duration_since(boot_time).as_secs_f64(),
            tenant_id: tenant_id.unwrap_or(0),
            action: action.into(),
            resource: resource.into(),
            decision: dec_str,
            reason,
            source_ip,
        }
    }

    /// Create an "Allow" entry directly.
    #[allow(dead_code)]
    pub fn allowed(
        tenant_id: Option<u64>,
        action: impl Into<String>,
        resource: impl Into<String>,
        boot_time: Instant,
    ) -> Self {
        Self::from_decision(
            tenant_id,
            action,
            resource,
            &PolicyDecision::Allow,
            None,
            boot_time,
        )
    }
}

/// Bounded audit log with query support.
pub struct AuditLog {
    entries: VecDeque<AuditEntry>,
    max_entries: usize,
    boot_time: Instant,
}

#[allow(dead_code)]
impl AuditLog {
    /// Create a new audit log with the specified capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_entries.min(100_000)),
            max_entries,
            boot_time: Instant::now(),
        }
    }

    /// Boot time reference for elapsed calculations.
    pub fn boot_time(&self) -> Instant {
        self.boot_time
    }

    /// Record an audit entry and emit a tracing event.
    pub fn record(&mut self, entry: AuditEntry) {
        // Emit as a structured tracing event for log aggregation.
        info!(
            target: "ferrite::audit",
            tenant_id = entry.tenant_id,
            action = %entry.action,
            resource = %entry.resource,
            decision = %entry.decision,
            reason = entry.reason.as_deref().unwrap_or(""),
            source_ip = entry.source_ip.as_deref().unwrap_or(""),
            "audit"
        );

        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Record a policy decision as an audit entry.
    pub fn record_decision(
        &mut self,
        tenant_id: Option<u64>,
        action: impl Into<String>,
        resource: impl Into<String>,
        decision: &PolicyDecision,
        source_ip: Option<String>,
    ) {
        let entry = AuditEntry::from_decision(
            tenant_id,
            action,
            resource,
            decision,
            source_ip,
            self.boot_time,
        );
        self.record(entry);
    }

    /// Return the most recent N entries.
    pub fn query_recent(&self, n: usize) -> Vec<&AuditEntry> {
        self.entries.iter().rev().take(n).collect::<Vec<_>>().into_iter().rev().collect()
    }

    /// Return entries for a specific tenant.
    pub fn query_by_tenant(&self, tenant_id: u64, limit: usize) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .rev()
            .filter(|e| e.tenant_id == tenant_id)
            .take(limit)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Export all entries as a JSON string.
    pub fn export_json(&self) -> String {
        serde_json::to_string_pretty(&self.entries.iter().collect::<Vec<_>>())
            .unwrap_or_else(|_| "[]".to_string())
    }

    /// Total number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the audit log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all entries (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &AuditEntry> {
        self.entries.iter()
    }
}
