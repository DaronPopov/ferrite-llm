//! Tenant model for multi-tenant GPU scheduling.
//!
//! Each tenant represents an isolated workload owner with its own resource quotas
//! and usage tracking. The default tenant (id=0) provides backward compatibility
//! for single-user deployments.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Unique identifier for a tenant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TenantId(pub u64);

impl TenantId {
    /// The default tenant ID for backward-compatible single-user mode.
    pub const DEFAULT: TenantId = TenantId(0);
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tenant-{}", self.0)
    }
}

impl From<u64> for TenantId {
    fn from(id: u64) -> Self {
        TenantId(id)
    }
}

/// Resource quotas for a tenant.
///
/// A quota value of `u64::MAX` means unlimited.
#[derive(Debug, Clone)]
pub struct TenantQuotas {
    /// Maximum VRAM in bytes this tenant may consume.
    pub max_vram_bytes: u64,
    /// Maximum number of streams this tenant may hold concurrently.
    pub max_streams: u64,
    /// Maximum number of concurrently running jobs.
    pub max_concurrent_jobs: u64,
    /// Total runtime budget in milliseconds (cumulative wall-clock).
    pub max_runtime_budget_ms: u64,
}

impl TenantQuotas {
    /// Create unlimited quotas (used by the default tenant).
    pub fn unlimited() -> Self {
        Self {
            max_vram_bytes: u64::MAX,
            max_streams: u64::MAX,
            max_concurrent_jobs: u64::MAX,
            max_runtime_budget_ms: u64::MAX,
        }
    }
}

impl Default for TenantQuotas {
    fn default() -> Self {
        Self::unlimited()
    }
}

/// Live resource usage counters for a tenant.
///
/// All fields use atomics for lock-free concurrent updates from multiple streams.
pub struct TenantUsage {
    /// Current VRAM bytes consumed by this tenant.
    pub current_vram_bytes: AtomicU64,
    /// Number of streams currently held by this tenant.
    pub active_streams: AtomicU64,
    /// Number of jobs currently in the Running state for this tenant.
    pub active_jobs: AtomicU64,
    /// Cumulative runtime consumed in milliseconds.
    pub consumed_runtime_ms: AtomicU64,
}

impl TenantUsage {
    /// Create zeroed usage counters.
    pub fn new() -> Self {
        Self {
            current_vram_bytes: AtomicU64::new(0),
            active_streams: AtomicU64::new(0),
            active_jobs: AtomicU64::new(0),
            consumed_runtime_ms: AtomicU64::new(0),
        }
    }

    /// Take a point-in-time snapshot of the usage counters.
    pub fn snapshot(&self) -> TenantUsageSnapshot {
        TenantUsageSnapshot {
            current_vram_bytes: self.current_vram_bytes.load(Ordering::Relaxed),
            active_streams: self.active_streams.load(Ordering::Relaxed),
            active_jobs: self.active_jobs.load(Ordering::Relaxed),
            consumed_runtime_ms: self.consumed_runtime_ms.load(Ordering::Relaxed),
        }
    }
}

impl Default for TenantUsage {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TenantUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenantUsage")
            .field("current_vram_bytes", &self.current_vram_bytes.load(Ordering::Relaxed))
            .field("active_streams", &self.active_streams.load(Ordering::Relaxed))
            .field("active_jobs", &self.active_jobs.load(Ordering::Relaxed))
            .field("consumed_runtime_ms", &self.consumed_runtime_ms.load(Ordering::Relaxed))
            .finish()
    }
}

/// Immutable snapshot of tenant usage at a point in time.
#[derive(Debug, Clone)]
pub struct TenantUsageSnapshot {
    pub current_vram_bytes: u64,
    pub active_streams: u64,
    pub active_jobs: u64,
    pub consumed_runtime_ms: u64,
}

/// A tenant with its identity, quotas, and live usage tracking.
pub struct Tenant {
    /// Unique tenant identifier.
    pub id: TenantId,
    /// Human-readable label (e.g., "inference-prod", "training-nightly").
    pub label: String,
    /// When this tenant was registered.
    pub created_at: Instant,
    /// Resource quotas governing this tenant.
    pub quotas: TenantQuotas,
    /// Live resource usage counters.
    pub usage: TenantUsage,
}

impl Tenant {
    /// Create a new tenant with the given ID, label, and quotas.
    pub fn new(id: TenantId, label: impl Into<String>, quotas: TenantQuotas) -> Self {
        Self {
            id,
            label: label.into(),
            created_at: Instant::now(),
            quotas,
            usage: TenantUsage::new(),
        }
    }

    /// Create the default tenant (id=0) with unlimited quotas.
    ///
    /// This ensures backward compatibility: all legacy workloads run under the
    /// default tenant with no resource restrictions.
    pub fn default_tenant() -> Self {
        Self::new(TenantId::DEFAULT, "default", TenantQuotas::unlimited())
    }

    /// Check whether the tenant's VRAM usage is within quota.
    pub fn vram_within_quota(&self) -> bool {
        self.usage.current_vram_bytes.load(Ordering::Relaxed) < self.quotas.max_vram_bytes
    }

    /// Check whether the tenant's stream usage is within quota.
    pub fn streams_within_quota(&self) -> bool {
        self.usage.active_streams.load(Ordering::Relaxed) < self.quotas.max_streams
    }

    /// Check whether the tenant's concurrent job count is within quota.
    pub fn jobs_within_quota(&self) -> bool {
        self.usage.active_jobs.load(Ordering::Relaxed) < self.quotas.max_concurrent_jobs
    }

    /// Check whether the tenant's runtime budget has not been exhausted.
    pub fn runtime_within_budget(&self) -> bool {
        self.usage.consumed_runtime_ms.load(Ordering::Relaxed) < self.quotas.max_runtime_budget_ms
    }

    /// Return the VRAM utilization as a percentage (0.0..=100.0).
    pub fn vram_utilization_pct(&self) -> f64 {
        if self.quotas.max_vram_bytes == u64::MAX {
            return 0.0;
        }
        let used = self.usage.current_vram_bytes.load(Ordering::Relaxed) as f64;
        let quota = self.quotas.max_vram_bytes as f64;
        (used / quota) * 100.0
    }
}

impl std::fmt::Debug for Tenant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tenant")
            .field("id", &self.id)
            .field("label", &self.label)
            .field("quotas", &self.quotas)
            .field("usage", &self.usage)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_id_default() {
        assert_eq!(TenantId::DEFAULT, TenantId(0));
        assert_eq!(TenantId::DEFAULT.to_string(), "tenant-0");
    }

    #[test]
    fn test_tenant_id_from_u64() {
        let id: TenantId = 42u64.into();
        assert_eq!(id, TenantId(42));
    }

    #[test]
    fn test_unlimited_quotas() {
        let q = TenantQuotas::unlimited();
        assert_eq!(q.max_vram_bytes, u64::MAX);
        assert_eq!(q.max_streams, u64::MAX);
        assert_eq!(q.max_concurrent_jobs, u64::MAX);
        assert_eq!(q.max_runtime_budget_ms, u64::MAX);
    }

    #[test]
    fn test_default_tenant() {
        let t = Tenant::default_tenant();
        assert_eq!(t.id, TenantId::DEFAULT);
        assert_eq!(t.label, "default");
        assert!(t.vram_within_quota());
        assert!(t.streams_within_quota());
        assert!(t.jobs_within_quota());
        assert!(t.runtime_within_budget());
    }

    #[test]
    fn test_quota_enforcement() {
        let quotas = TenantQuotas {
            max_vram_bytes: 1024,
            max_streams: 2,
            max_concurrent_jobs: 4,
            max_runtime_budget_ms: 10_000,
        };
        let t = Tenant::new(TenantId(1), "test-tenant", quotas);

        // Initially within all quotas
        assert!(t.vram_within_quota());
        assert!(t.streams_within_quota());
        assert!(t.jobs_within_quota());
        assert!(t.runtime_within_budget());

        // Exceed VRAM quota
        t.usage.current_vram_bytes.store(2048, Ordering::Relaxed);
        assert!(!t.vram_within_quota());

        // Exceed stream quota
        t.usage.active_streams.store(3, Ordering::Relaxed);
        assert!(!t.streams_within_quota());

        // Exceed job quota
        t.usage.active_jobs.store(5, Ordering::Relaxed);
        assert!(!t.jobs_within_quota());

        // Exhaust runtime budget
        t.usage.consumed_runtime_ms.store(15_000, Ordering::Relaxed);
        assert!(!t.runtime_within_budget());
    }

    #[test]
    fn test_usage_snapshot() {
        let usage = TenantUsage::new();
        usage.current_vram_bytes.store(512, Ordering::Relaxed);
        usage.active_streams.store(2, Ordering::Relaxed);
        usage.active_jobs.store(1, Ordering::Relaxed);
        usage.consumed_runtime_ms.store(5000, Ordering::Relaxed);

        let snap = usage.snapshot();
        assert_eq!(snap.current_vram_bytes, 512);
        assert_eq!(snap.active_streams, 2);
        assert_eq!(snap.active_jobs, 1);
        assert_eq!(snap.consumed_runtime_ms, 5000);
    }

    #[test]
    fn test_vram_utilization_pct() {
        let quotas = TenantQuotas {
            max_vram_bytes: 1000,
            max_streams: u64::MAX,
            max_concurrent_jobs: u64::MAX,
            max_runtime_budget_ms: u64::MAX,
        };
        let t = Tenant::new(TenantId(1), "pct-test", quotas);
        t.usage.current_vram_bytes.store(500, Ordering::Relaxed);
        assert!((t.vram_utilization_pct() - 50.0).abs() < f64::EPSILON);

        // Unlimited quota returns 0%
        let t2 = Tenant::default_tenant();
        assert!((t2.vram_utilization_pct() - 0.0).abs() < f64::EPSILON);
    }
}
