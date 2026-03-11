//! Policy decision model for the Ferrite OS control plane.
//!
//! Every policy evaluation produces a [`PolicyDecision`]: either [`Allow`] or
//! [`Deny`] with a machine-readable reason code and human-friendly remediation
//! text.  Consumers never need to guess why an action was rejected.

use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Outcome of a policy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyDecision {
    /// The action is permitted.
    Allow,
    /// The action is denied with a structured reason and remediation advice.
    Deny {
        reason: DenialReason,
        remediation: String,
    },
}

#[allow(dead_code)]
impl PolicyDecision {
    /// Returns `true` when the decision permits the action.
    pub fn is_allowed(&self) -> bool {
        matches!(self, PolicyDecision::Allow)
    }

    /// Returns `true` when the decision blocks the action.
    pub fn is_denied(&self) -> bool {
        matches!(self, PolicyDecision::Deny { .. })
    }

    /// Convenience constructor for a denial.
    pub fn deny(reason: DenialReason, remediation: impl Into<String>) -> Self {
        PolicyDecision::Deny {
            reason,
            remediation: remediation.into(),
        }
    }

    /// Build a standardized [`DenialPayload`].  Returns `None` for `Allow`.
    pub fn to_denial_payload(&self, action: &str, resource: &str) -> Option<DenialPayload> {
        match self {
            PolicyDecision::Allow => None,
            PolicyDecision::Deny { reason, remediation } => Some(DenialPayload {
                ok: false,
                error: format!("policy denied: {}", reason),
                reason: reason.to_string(),
                reason_code: reason.code().to_string(),
                remediation: remediation.clone(),
                action: action.to_string(),
                resource: resource.to_string(),
            }),
        }
    }

    /// Build a denial as a JSON string (with trailing newline), or `None`
    /// for `Allow`.
    pub fn to_denial_json(&self, action: &str, resource: &str) -> Option<String> {
        self.to_denial_payload(action, resource)
            .map(|p| match serde_json::to_string(&p) {
                Ok(json) => format!("{json}\n"),
                Err(_) => "{\"ok\":false,\"error\":\"policy denied\"}\n".to_string(),
            })
    }
}

impl fmt::Display for PolicyDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolicyDecision::Allow => write!(f, "ALLOW"),
            PolicyDecision::Deny { reason, remediation } => {
                write!(f, "DENY [{}]: {}", reason, remediation)
            }
        }
    }
}

/// Machine-readable reason for a policy denial.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DenialReason {
    /// Tenant has exceeded their resource quota (VRAM, streams, etc.).
    QuotaExceeded,
    /// Request rate limit has been hit.
    RateLimited,
    /// The tenant account is suspended or disabled.
    TenantSuspended,
    /// The requested resource is temporarily unavailable.
    ResourceUnavailable,
    /// The caller lacks authorization for this action.
    Unauthorized,
    /// A custom policy rule was violated.
    PolicyViolation(String),
}

impl DenialReason {
    /// Return a stable diagnostic code for this denial reason.
    pub fn code(&self) -> &'static str {
        match self {
            DenialReason::QuotaExceeded => "POLICY-DENY-0001",
            DenialReason::RateLimited => "POLICY-DENY-0002",
            DenialReason::TenantSuspended => "POLICY-DENY-0003",
            DenialReason::ResourceUnavailable => "POLICY-DENY-0004",
            DenialReason::Unauthorized => "POLICY-DENY-0005",
            DenialReason::PolicyViolation(_) => "POLICY-DENY-0006",
        }
    }
}

impl fmt::Display for DenialReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DenialReason::QuotaExceeded => write!(f, "QuotaExceeded"),
            DenialReason::RateLimited => write!(f, "RateLimited"),
            DenialReason::TenantSuspended => write!(f, "TenantSuspended"),
            DenialReason::ResourceUnavailable => write!(f, "ResourceUnavailable"),
            DenialReason::Unauthorized => write!(f, "Unauthorized"),
            DenialReason::PolicyViolation(detail) => write!(f, "PolicyViolation({})", detail),
        }
    }
}

/// Context supplied to every policy evaluation.
///
/// Describes the tenant, the action being attempted, the target resource,
/// and the wall-clock time of the request.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PolicyContext {
    /// Tenant performing the action (None for unauthenticated/system).
    pub tenant_id: Option<u64>,
    /// Human-readable action name (e.g. "app-start", "kill-job").
    pub action: String,
    /// Target resource identifier (e.g. app name, job id).
    pub resource: String,
    /// When the request was received.
    pub timestamp: Instant,
}

impl PolicyContext {
    /// Create a new context with the current timestamp.
    pub fn new(tenant_id: Option<u64>, action: impl Into<String>, resource: impl Into<String>) -> Self {
        Self {
            tenant_id,
            action: action.into(),
            resource: resource.into(),
            timestamp: Instant::now(),
        }
    }
}

/// Standardized, machine-readable denial payload for wire responses.
///
/// Every policy denial—whether from the scheduler command path or the
/// top-level command path—serializes to this shape so that clients can
/// parse denials uniformly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenialPayload {
    /// Always `false` for a denial.
    pub ok: bool,
    /// Short human-readable error summary.
    pub error: String,
    /// Human-readable denial reason name (e.g. `"Unauthorized"`).
    pub reason: String,
    /// Stable machine-readable code (e.g. `"POLICY-DENY-0005"`).
    pub reason_code: String,
    /// Actionable remediation advice.
    pub remediation: String,
    /// The action that was attempted.
    pub action: String,
    /// The target resource.
    pub resource: String,
}
