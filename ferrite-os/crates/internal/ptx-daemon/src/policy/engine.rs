//! Policy enforcement engine.
//!
//! The engine evaluates an ordered list of [`PolicyRule`]s against a
//! [`PolicyContext`].  First denial wins.  Every evaluation is recorded in
//! the embedded [`AuditLog`].

use super::audit::AuditLog;
use super::decision::{DenialReason, PolicyContext, PolicyDecision};

/// A named policy rule with a check function.
pub struct PolicyRule {
    /// Human-readable rule name (e.g. "max-concurrent-jobs").
    pub name: String,
    /// Description of what this rule enforces.
    pub description: String,
    /// The check function.  Returns `Allow` or `Deny`.
    pub check: Box<dyn Fn(&PolicyContext) -> PolicyDecision + Send + Sync>,
}

impl PolicyRule {
    /// Create a new policy rule.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        check: impl Fn(&PolicyContext) -> PolicyDecision + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            check: Box::new(check),
        }
    }
}

impl std::fmt::Debug for PolicyRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyRule")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish()
    }
}

/// The policy enforcement engine.
///
/// Holds an ordered list of rules and an audit log.  Call [`evaluate`] to
/// run all rules against a context.
pub struct PolicyEngine {
    rules: Vec<PolicyRule>,
    audit_log: AuditLog,
    strict: bool,
}

#[allow(dead_code)]
impl PolicyEngine {
    /// Create a new engine with the given audit log capacity.
    pub fn new(audit_max_entries: usize) -> Self {
        Self {
            rules: Vec::new(),
            audit_log: AuditLog::new(audit_max_entries),
            strict: false,
        }
    }

    /// Create an engine pre-loaded with the default rule set.
    ///
    /// In `"permissive"` mode (the default), anonymous requests are allowed.
    /// In `"strict"` mode, mutating actions require a tenant ID.
    pub fn with_defaults(audit_max_entries: usize) -> Self {
        Self::with_mode(audit_max_entries, "permissive")
    }

    /// Create an engine for the given policy mode (`"permissive"` or `"strict"`).
    pub fn with_mode(audit_max_entries: usize, mode: &str) -> Self {
        let mut engine = Self::new(audit_max_entries);
        engine.strict = mode == "strict";
        match mode {
            "strict" => engine.add_rule(strict_require_tenant_rule()),
            _ => engine.add_rule(permissive_require_tenant_rule()),
        }
        engine
    }

    /// Return the current policy mode label (for diagnostics / status).
    pub fn mode_label(&self) -> &'static str {
        if self.strict {
            "strict"
        } else {
            "permissive"
        }
    }

    /// Whether the engine is in strict mode.
    pub fn is_strict(&self) -> bool {
        self.strict
    }

    /// Set the strict-mode flag.
    pub fn set_strict(&mut self, strict: bool) {
        self.strict = strict;
    }

    /// Add a rule to the evaluation chain.
    pub fn add_rule(&mut self, rule: PolicyRule) {
        self.rules.push(rule);
    }

    /// Evaluate all rules.  First `Deny` wins; if all pass the result
    /// is `Allow`.  The decision is recorded in the audit log.
    ///
    /// In strict mode an additional safety net applies: when the rule
    /// chain is empty **and** the action is mutating, the engine
    /// returns a default denial rather than silently allowing.
    pub fn evaluate(&mut self, ctx: &PolicyContext) -> PolicyDecision {
        // Strict-mode safety net: an empty rule chain must never
        // silently allow mutating actions.
        if self.strict && self.rules.is_empty() && !is_read_only_action(&ctx.action) {
            let decision = PolicyDecision::deny(
                DenialReason::PolicyViolation("default-deny".to_string()),
                "No policy rules are configured in strict mode. \
                 All mutating actions are denied by default. \
                 Add an explicit allow-rule or switch to permissive mode.",
            );
            self.audit_log.record_decision(
                ctx.tenant_id,
                &ctx.action,
                &ctx.resource,
                &decision,
                None,
            );
            return decision;
        }

        let mut decision = PolicyDecision::Allow;

        for rule in &self.rules {
            let result = (rule.check)(ctx);
            if result.is_denied() {
                decision = result;
                break;
            }
        }

        // Record in audit log.
        self.audit_log.record_decision(
            ctx.tenant_id,
            &ctx.action,
            &ctx.resource,
            &decision,
            None,
        );

        decision
    }

    /// Evaluate without recording (useful for dry-run / preview).
    #[allow(dead_code)]
    pub fn evaluate_dry(&self, ctx: &PolicyContext) -> PolicyDecision {
        for rule in &self.rules {
            let result = (rule.check)(ctx);
            if result.is_denied() {
                return result;
            }
        }
        PolicyDecision::Allow
    }

    /// Access the audit log.
    pub fn audit_log(&self) -> &AuditLog {
        &self.audit_log
    }

    /// Mutable access to the audit log (for recording external events).
    pub fn audit_log_mut(&mut self) -> &mut AuditLog {
        &mut self.audit_log
    }

    /// List all registered rules (name + description).
    pub fn list_rules(&self) -> Vec<(&str, &str)> {
        self.rules
            .iter()
            .map(|r| (r.name.as_str(), r.description.as_str()))
            .collect()
    }

    /// Number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

// ── built-in rules ─────────────────────────────────────────────────

/// Permissive tenant rule: anonymous (None) tenants are allowed for all actions.
pub fn permissive_require_tenant_rule() -> PolicyRule {
    PolicyRule::new(
        "require-tenant-permissive",
        "Allows all actions regardless of tenant context (permissive mode)",
        |_ctx| PolicyDecision::Allow,
    )
}

/// Strict tenant rule: anonymous (None) tenants are denied for mutating actions.
///
/// Read-only actions (those whose action name contains ".list", ".info",
/// ".status", ".stats", ".policy-list", ".policy-status", or starts with
/// "scheduler.queue-status") are always allowed.  All other actions require
/// a valid tenant ID.
pub fn strict_require_tenant_rule() -> PolicyRule {
    PolicyRule::new(
        "require-tenant-strict",
        "Denies mutating actions when no tenant ID is provided (strict mode)",
        |ctx| {
            if ctx.tenant_id.is_some() {
                return PolicyDecision::Allow;
            }
            // Allow read-only / introspection actions without a tenant
            if is_read_only_action(&ctx.action) {
                return PolicyDecision::Allow;
            }
            PolicyDecision::deny(
                DenialReason::Unauthorized,
                "A tenant ID is required for mutating actions in strict mode. \
                 Supply a tenant context or switch to permissive mode.",
            )
        },
    )
}

/// Classify an action name as read-only (introspection / query).
pub fn is_read_only_action(action: &str) -> bool {
    matches!(
        action,
        "scheduler.queue-status"
            | "scheduler.tenant-list"
            | "scheduler.tenant-info"
            | "scheduler.stats"
            | "scheduler.policy-list"
            | "scheduler.policy-status"
            | "ping"
            | "status"
            | "stats"
            | "metrics"
            | "snapshot"
            | "health"
            | "keepalive"
            | "help"
            | "apps"
            | "scheduler-status"
            | "scheduler-queue"
            | "scheduler-policy"
            | "audit-query"
            | "events-stream"
            | "run-list"
            | "job-status"
            | "job-list"
            | "job-history"
    )
}

/// Create a quota-check rule that denies when a callback says the
/// tenant is over quota.
#[allow(dead_code)]
pub fn quota_check_rule(
    is_over_quota: impl Fn(u64) -> bool + Send + Sync + 'static,
) -> PolicyRule {
    PolicyRule::new(
        "quota-check",
        "Denies actions when the tenant has exceeded their resource quota",
        move |ctx| {
            if let Some(tid) = ctx.tenant_id {
                if is_over_quota(tid) {
                    return PolicyDecision::deny(
                        DenialReason::QuotaExceeded,
                        "Reduce resource usage or request a quota increase from the administrator.",
                    );
                }
            }
            PolicyDecision::Allow
        },
    )
}

/// Create a rate-limit rule that denies when a callback says the
/// tenant has exceeded request rate.
#[allow(dead_code)]
pub fn rate_limit_rule(
    is_rate_limited: impl Fn(u64) -> bool + Send + Sync + 'static,
) -> PolicyRule {
    PolicyRule::new(
        "rate-limit",
        "Denies actions when the tenant has exceeded the request rate limit",
        move |ctx| {
            if let Some(tid) = ctx.tenant_id {
                if is_rate_limited(tid) {
                    return PolicyDecision::deny(
                        DenialReason::RateLimited,
                        "Wait before retrying. Consider batching operations to reduce request rate.",
                    );
                }
            }
            PolicyDecision::Allow
        },
    )
}

/// Create a max-concurrent-jobs rule.
#[allow(dead_code)]
pub fn max_concurrent_jobs_rule(
    current_jobs: impl Fn() -> usize + Send + Sync + 'static,
    max_jobs: usize,
) -> PolicyRule {
    PolicyRule::new(
        "max-concurrent-jobs",
        "Limits the total number of concurrently running jobs",
        move |_ctx| {
            if current_jobs() >= max_jobs {
                return PolicyDecision::deny(
                    DenialReason::ResourceUnavailable,
                    format!(
                        "Maximum concurrent job limit ({}) reached. Wait for running jobs to complete.",
                        max_jobs
                    ),
                );
            }
            PolicyDecision::Allow
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(tenant: Option<u64>, action: &str) -> PolicyContext {
        PolicyContext::new(tenant, action, "test-resource")
    }

    // ── mode factory tests ───────────────────────────────────────

    #[test]
    fn with_mode_permissive_allows_anonymous_mutating() {
        let mut engine = PolicyEngine::with_mode(100, "permissive");
        assert_eq!(engine.mode_label(), "permissive");
        let decision = engine.evaluate(&ctx(None, "app-start"));
        assert!(decision.is_allowed());
    }

    #[test]
    fn with_mode_strict_denies_anonymous_mutating() {
        let mut engine = PolicyEngine::with_mode(100, "strict");
        assert_eq!(engine.mode_label(), "strict");
        let decision = engine.evaluate(&ctx(None, "app-start"));
        assert!(decision.is_denied());
    }

    #[test]
    fn with_mode_strict_allows_anonymous_read_only() {
        let mut engine = PolicyEngine::with_mode(100, "strict");
        let decision = engine.evaluate(&ctx(None, "scheduler.queue-status"));
        assert!(decision.is_allowed());
    }

    #[test]
    fn with_mode_strict_allows_tenant_mutating() {
        let mut engine = PolicyEngine::with_mode(100, "strict");
        let decision = engine.evaluate(&ctx(Some(42), "app-start"));
        assert!(decision.is_allowed());
    }

    #[test]
    fn with_defaults_is_permissive() {
        let engine = PolicyEngine::with_defaults(100);
        assert_eq!(engine.mode_label(), "permissive");
    }

    // ── denial payload tests ─────────────────────────────────────

    #[test]
    fn strict_denial_has_unauthorized_reason_code() {
        let mut engine = PolicyEngine::with_mode(100, "strict");
        let decision = engine.evaluate(&ctx(None, "job-submit"));
        match decision {
            PolicyDecision::Deny { reason, remediation } => {
                assert_eq!(reason.code(), "POLICY-DENY-0005");
                assert_eq!(reason, DenialReason::Unauthorized);
                assert!(!remediation.is_empty(), "remediation text must be present");
            }
            PolicyDecision::Allow => panic!("expected denial"),
        }
    }

    // ── audit log coverage ───────────────────────────────────────

    #[test]
    fn evaluate_records_allow_in_audit_log() {
        let mut engine = PolicyEngine::with_mode(100, "permissive");
        engine.evaluate(&ctx(Some(7), "app-start"));
        assert_eq!(engine.audit_log().len(), 1);
        let entry = &engine.audit_log().query_recent(1)[0];
        assert_eq!(entry.decision, "Allow");
        assert_eq!(entry.tenant_id, 7);
        assert_eq!(entry.action, "app-start");
    }

    #[test]
    fn evaluate_records_deny_in_audit_log() {
        let mut engine = PolicyEngine::with_mode(100, "strict");
        engine.evaluate(&ctx(None, "kill-job"));
        assert_eq!(engine.audit_log().len(), 1);
        let entry = &engine.audit_log().query_recent(1)[0];
        assert_eq!(entry.decision, "Deny");
        assert!(entry.reason.is_some());
    }

    #[test]
    fn evaluate_dry_does_not_record() {
        let engine = PolicyEngine::with_mode(100, "strict");
        let decision = engine.evaluate_dry(&ctx(None, "app-stop"));
        assert!(decision.is_denied());
        assert_eq!(engine.audit_log().len(), 0, "dry-run must not touch the audit log");
    }

    // ── read-only action classification ──────────────────────────

    #[test]
    fn is_read_only_known_actions() {
        assert!(is_read_only_action("ping"));
        assert!(is_read_only_action("status"));
        assert!(is_read_only_action("scheduler.queue-status"));
        assert!(is_read_only_action("scheduler.tenant-list"));
        assert!(is_read_only_action("scheduler.stats"));
        assert!(is_read_only_action("scheduler.policy-list"));
        assert!(is_read_only_action("scheduler.policy-status"));
        assert!(is_read_only_action("health"));
        assert!(is_read_only_action("metrics"));
        assert!(is_read_only_action("audit-query"));
        assert!(is_read_only_action("events-stream"));
        assert!(is_read_only_action("run-list"));
    }

    #[test]
    fn is_read_only_rejects_mutating() {
        assert!(!is_read_only_action("app-start"));
        assert!(!is_read_only_action("app-stop"));
        assert!(!is_read_only_action("job-submit"));
        assert!(!is_read_only_action("job-stop"));
        assert!(!is_read_only_action("run-file"));
        assert!(!is_read_only_action("run-entry"));
        assert!(!is_read_only_action("scheduler.set-quota"));
        assert!(!is_read_only_action("scheduler.kill-job"));
        assert!(!is_read_only_action("scheduler.pause-queue"));
    }

    // ── rule listing ────────────────────────────────────────────

    #[test]
    fn list_rules_includes_default_rule() {
        let engine = PolicyEngine::with_mode(100, "strict");
        let rules = engine.list_rules();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].0, "require-tenant-strict");
    }

    #[test]
    fn list_rules_permissive() {
        let engine = PolicyEngine::with_mode(100, "permissive");
        let rules = engine.list_rules();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].0, "require-tenant-permissive");
    }

    // ── quota / rate-limit built-in rules ────────────────────────

    #[test]
    fn quota_check_denies_over_quota_tenant() {
        let mut engine = PolicyEngine::new(100);
        engine.add_rule(quota_check_rule(|tid| tid == 99));
        let decision = engine.evaluate(&ctx(Some(99), "job-submit"));
        assert!(decision.is_denied());
        match decision {
            PolicyDecision::Deny { reason, .. } => {
                assert_eq!(reason, DenialReason::QuotaExceeded);
            }
            _ => panic!("expected denial"),
        }
    }

    #[test]
    fn quota_check_allows_under_quota_tenant() {
        let mut engine = PolicyEngine::new(100);
        engine.add_rule(quota_check_rule(|tid| tid == 99));
        let decision = engine.evaluate(&ctx(Some(1), "job-submit"));
        assert!(decision.is_allowed());
    }

    #[test]
    fn rate_limit_denies_when_limited() {
        let mut engine = PolicyEngine::new(100);
        engine.add_rule(rate_limit_rule(|_| true));
        let decision = engine.evaluate(&ctx(Some(1), "app-start"));
        assert!(decision.is_denied());
        match decision {
            PolicyDecision::Deny { reason, .. } => {
                assert_eq!(reason, DenialReason::RateLimited);
            }
            _ => panic!("expected denial"),
        }
    }

    #[test]
    fn first_deny_wins() {
        let mut engine = PolicyEngine::new(100);
        engine.add_rule(quota_check_rule(|_| true)); // always over quota
        engine.add_rule(rate_limit_rule(|_| true));   // also rate limited
        let decision = engine.evaluate(&ctx(Some(1), "app-start"));
        // quota check comes first
        match decision {
            PolicyDecision::Deny { reason, .. } => {
                assert_eq!(reason, DenialReason::QuotaExceeded);
            }
            _ => panic!("expected denial"),
        }
    }

    // ── strict default-deny safety net ──────────────────────────────

    #[test]
    fn strict_empty_rules_denies_mutating() {
        let mut engine = PolicyEngine::new(100);
        engine.set_strict(true);
        // No rules added — safety net must fire
        let decision = engine.evaluate(&ctx(None, "app-start"));
        assert!(decision.is_denied());
        match decision {
            PolicyDecision::Deny { reason, .. } => {
                assert_eq!(reason, DenialReason::PolicyViolation("default-deny".to_string()));
            }
            _ => panic!("expected default-deny"),
        }
    }

    #[test]
    fn strict_empty_rules_allows_read_only() {
        let mut engine = PolicyEngine::new(100);
        engine.set_strict(true);
        let decision = engine.evaluate(&ctx(None, "ping"));
        assert!(decision.is_allowed());
    }

    #[test]
    fn strict_empty_rules_records_denial_in_audit() {
        let mut engine = PolicyEngine::new(100);
        engine.set_strict(true);
        engine.evaluate(&ctx(None, "job-submit"));
        assert_eq!(engine.audit_log().len(), 1);
        let entry = &engine.audit_log().query_recent(1)[0];
        assert_eq!(entry.decision, "Deny");
        assert!(entry.reason.as_ref().unwrap().contains("PolicyViolation"));
    }

    #[test]
    fn permissive_empty_rules_allows_mutating() {
        let mut engine = PolicyEngine::new(100);
        // permissive (default), no rules
        let decision = engine.evaluate(&ctx(None, "app-start"));
        assert!(decision.is_allowed());
    }

    #[test]
    fn is_strict_reflects_mode() {
        let engine = PolicyEngine::with_mode(100, "strict");
        assert!(engine.is_strict());
        let engine = PolicyEngine::with_mode(100, "permissive");
        assert!(!engine.is_strict());
    }

    // ── denial payload tests ────────────────────────────────────────

    #[test]
    fn denial_payload_has_all_fields() {
        let decision = PolicyDecision::deny(
            DenialReason::Unauthorized,
            "Supply a tenant ID",
        );
        let payload = decision.to_denial_payload("app-start", "stream_compute");
        assert!(payload.is_some());
        let p = payload.unwrap();
        assert!(!p.ok);
        assert_eq!(p.reason, "Unauthorized");
        assert_eq!(p.reason_code, "POLICY-DENY-0005");
        assert_eq!(p.remediation, "Supply a tenant ID");
        assert_eq!(p.action, "app-start");
        assert_eq!(p.resource, "stream_compute");
        assert!(p.error.contains("Unauthorized"));
    }

    #[test]
    fn denial_payload_none_for_allow() {
        let decision = PolicyDecision::Allow;
        assert!(decision.to_denial_payload("ping", "test").is_none());
        assert!(decision.to_denial_json("ping", "test").is_none());
    }

    #[test]
    fn denial_json_is_valid() {
        let decision = PolicyDecision::deny(
            DenialReason::QuotaExceeded,
            "Reduce usage",
        );
        let json = decision.to_denial_json("job-submit", "gpu:0").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(json.trim()).unwrap();
        assert_eq!(parsed["ok"], false);
        assert_eq!(parsed["reason_code"], "POLICY-DENY-0001");
        assert_eq!(parsed["reason"], "QuotaExceeded");
        assert_eq!(parsed["remediation"], "Reduce usage");
        assert_eq!(parsed["action"], "job-submit");
        assert_eq!(parsed["resource"], "gpu:0");
    }
}
