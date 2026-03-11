//! Tenant registry for managing registered tenants.
//!
//! The registry maintains the canonical set of tenants and provides
//! lookup, registration, and usage-update operations.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use super::tenant::{Tenant, TenantId, TenantQuotas};

/// Central registry of all known tenants.
///
/// The registry always contains a default tenant (id=0) for backward compatibility.
pub struct TenantRegistry {
    tenants: HashMap<u64, Tenant>,
}

impl TenantRegistry {
    /// Create a new registry containing only the default tenant.
    pub fn new() -> Self {
        let mut tenants = HashMap::new();
        tenants.insert(TenantId::DEFAULT.0, Tenant::default_tenant());
        Self { tenants }
    }

    /// Register a new tenant. Returns an error if the tenant ID already exists.
    pub fn register(
        &mut self,
        id: TenantId,
        label: impl Into<String>,
        quotas: TenantQuotas,
    ) -> Result<&Tenant, RegistryError> {
        if self.tenants.contains_key(&id.0) {
            return Err(RegistryError::AlreadyExists(id));
        }
        let tenant = Tenant::new(id, label, quotas);
        self.tenants.insert(id.0, tenant);
        Ok(self.tenants.get(&id.0).unwrap())
    }

    /// Get an immutable reference to a tenant by ID.
    pub fn get(&self, id: TenantId) -> Option<&Tenant> {
        self.tenants.get(&id.0)
    }

    /// Get a mutable reference to a tenant by ID.
    pub fn get_mut(&mut self, id: TenantId) -> Option<&mut Tenant> {
        self.tenants.get_mut(&id.0)
    }

    /// Get the default tenant.
    pub fn default_tenant(&self) -> &Tenant {
        self.tenants
            .get(&TenantId::DEFAULT.0)
            .expect("default tenant must always exist in registry")
    }

    /// Update VRAM usage for a tenant by a signed delta.
    ///
    /// Positive delta adds bytes, negative delta subtracts.
    /// Returns the new VRAM usage, or None if the tenant is not found.
    pub fn update_vram_usage(&self, id: TenantId, delta: i64) -> Option<u64> {
        let tenant = self.tenants.get(&id.0)?;
        if delta >= 0 {
            let new_val = tenant
                .usage
                .current_vram_bytes
                .fetch_add(delta as u64, Ordering::Relaxed)
                + delta as u64;
            Some(new_val)
        } else {
            let abs_delta = delta.unsigned_abs();
            let prev = tenant
                .usage
                .current_vram_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    Some(current.saturating_sub(abs_delta))
                })
                .unwrap_or(0);
            Some(prev.saturating_sub(abs_delta))
        }
    }

    /// Increment the active streams counter for a tenant.
    pub fn acquire_stream(&self, id: TenantId) -> Option<u64> {
        let tenant = self.tenants.get(&id.0)?;
        let new_val = tenant.usage.active_streams.fetch_add(1, Ordering::Relaxed) + 1;
        Some(new_val)
    }

    /// Decrement the active streams counter for a tenant.
    pub fn release_stream(&self, id: TenantId) -> Option<u64> {
        let tenant = self.tenants.get(&id.0)?;
        let prev = tenant
            .usage
            .active_streams
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(1))
            })
            .unwrap_or(0);
        Some(prev.saturating_sub(1))
    }

    /// Increment the active jobs counter for a tenant.
    pub fn increment_active_jobs(&self, id: TenantId) -> Option<u64> {
        let tenant = self.tenants.get(&id.0)?;
        let new_val = tenant.usage.active_jobs.fetch_add(1, Ordering::Relaxed) + 1;
        Some(new_val)
    }

    /// Decrement the active jobs counter for a tenant.
    pub fn decrement_active_jobs(&self, id: TenantId) -> Option<u64> {
        let tenant = self.tenants.get(&id.0)?;
        let prev = tenant
            .usage
            .active_jobs
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(1))
            })
            .unwrap_or(0);
        Some(prev.saturating_sub(1))
    }

    /// Add consumed runtime milliseconds for a tenant.
    pub fn add_runtime_ms(&self, id: TenantId, ms: u64) -> Option<u64> {
        let tenant = self.tenants.get(&id.0)?;
        let new_val = tenant
            .usage
            .consumed_runtime_ms
            .fetch_add(ms, Ordering::Relaxed)
            + ms;
        Some(new_val)
    }

    /// Check whether a tenant is within all its quotas.
    pub fn check_quota(&self, id: TenantId) -> Option<bool> {
        let tenant = self.tenants.get(&id.0)?;
        Some(
            tenant.vram_within_quota()
                && tenant.streams_within_quota()
                && tenant.jobs_within_quota()
                && tenant.runtime_within_budget(),
        )
    }

    /// List all registered tenant IDs.
    pub fn list(&self) -> Vec<TenantId> {
        self.tenants.keys().map(|&k| TenantId(k)).collect()
    }

    /// Return the number of registered tenants (including default).
    pub fn len(&self) -> usize {
        self.tenants.len()
    }

    /// Returns true if the registry contains only the default tenant.
    pub fn is_empty(&self) -> bool {
        self.tenants.len() <= 1
    }

    /// Remove a tenant from the registry.
    ///
    /// The default tenant (id=0) cannot be removed.
    pub fn unregister(&mut self, id: TenantId) -> Result<Tenant, RegistryError> {
        if id == TenantId::DEFAULT {
            return Err(RegistryError::CannotRemoveDefault);
        }
        self.tenants
            .remove(&id.0)
            .ok_or(RegistryError::NotFound(id))
    }
}

impl Default for TenantRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for TenantRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenantRegistry")
            .field("num_tenants", &self.tenants.len())
            .field("tenant_ids", &self.list())
            .finish()
    }
}

/// Errors that can occur during registry operations.
#[derive(Debug, Clone)]
pub enum RegistryError {
    /// A tenant with this ID already exists.
    AlreadyExists(TenantId),
    /// No tenant with this ID was found.
    NotFound(TenantId),
    /// The default tenant cannot be removed.
    CannotRemoveDefault,
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::AlreadyExists(id) => write!(f, "tenant {} already exists", id),
            RegistryError::NotFound(id) => write!(f, "tenant {} not found", id),
            RegistryError::CannotRemoveDefault => write!(f, "cannot remove the default tenant"),
        }
    }
}

impl std::error::Error for RegistryError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_registry_has_default() {
        let reg = TenantRegistry::new();
        assert!(reg.get(TenantId::DEFAULT).is_some());
        assert_eq!(reg.default_tenant().id, TenantId::DEFAULT);
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_register_and_get() {
        let mut reg = TenantRegistry::new();
        let quotas = TenantQuotas {
            max_vram_bytes: 1024,
            max_streams: 4,
            max_concurrent_jobs: 8,
            max_runtime_budget_ms: 60_000,
        };
        reg.register(TenantId(1), "tenant-alpha", quotas).unwrap();

        let t = reg.get(TenantId(1)).unwrap();
        assert_eq!(t.label, "tenant-alpha");
        assert_eq!(t.quotas.max_vram_bytes, 1024);
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn test_register_duplicate() {
        let mut reg = TenantRegistry::new();
        reg.register(TenantId(1), "first", TenantQuotas::unlimited())
            .unwrap();
        let err = reg
            .register(TenantId(1), "second", TenantQuotas::unlimited())
            .unwrap_err();
        assert!(matches!(err, RegistryError::AlreadyExists(TenantId(1))));
    }

    #[test]
    fn test_unregister() {
        let mut reg = TenantRegistry::new();
        reg.register(TenantId(5), "temp", TenantQuotas::unlimited())
            .unwrap();
        assert_eq!(reg.len(), 2);

        let removed = reg.unregister(TenantId(5)).unwrap();
        assert_eq!(removed.id, TenantId(5));
        assert_eq!(reg.len(), 1);
        assert!(reg.get(TenantId(5)).is_none());
    }

    #[test]
    fn test_cannot_remove_default() {
        let mut reg = TenantRegistry::new();
        let err = reg.unregister(TenantId::DEFAULT).unwrap_err();
        assert!(matches!(err, RegistryError::CannotRemoveDefault));
    }

    #[test]
    fn test_update_vram_usage() {
        let reg = TenantRegistry::new();
        let new_usage = reg.update_vram_usage(TenantId::DEFAULT, 512).unwrap();
        assert_eq!(new_usage, 512);

        let new_usage = reg.update_vram_usage(TenantId::DEFAULT, 256).unwrap();
        assert_eq!(new_usage, 768);

        let new_usage = reg.update_vram_usage(TenantId::DEFAULT, -200).unwrap();
        assert_eq!(new_usage, 568);

        // Cannot go negative
        let new_usage = reg.update_vram_usage(TenantId::DEFAULT, -10000).unwrap();
        assert_eq!(new_usage, 0);
    }

    #[test]
    fn test_stream_acquire_release() {
        let reg = TenantRegistry::new();
        assert_eq!(reg.acquire_stream(TenantId::DEFAULT), Some(1));
        assert_eq!(reg.acquire_stream(TenantId::DEFAULT), Some(2));
        assert_eq!(reg.release_stream(TenantId::DEFAULT), Some(1));
        assert_eq!(reg.release_stream(TenantId::DEFAULT), Some(0));

        // Cannot go negative
        assert_eq!(reg.release_stream(TenantId::DEFAULT), Some(0));
    }

    #[test]
    fn test_active_jobs_tracking() {
        let reg = TenantRegistry::new();
        assert_eq!(reg.increment_active_jobs(TenantId::DEFAULT), Some(1));
        assert_eq!(reg.increment_active_jobs(TenantId::DEFAULT), Some(2));
        assert_eq!(reg.decrement_active_jobs(TenantId::DEFAULT), Some(1));
    }

    #[test]
    fn test_runtime_tracking() {
        let reg = TenantRegistry::new();
        assert_eq!(reg.add_runtime_ms(TenantId::DEFAULT, 500), Some(500));
        assert_eq!(reg.add_runtime_ms(TenantId::DEFAULT, 300), Some(800));
    }

    #[test]
    fn test_check_quota() {
        let mut reg = TenantRegistry::new();
        let quotas = TenantQuotas {
            max_vram_bytes: 1000,
            max_streams: u64::MAX,
            max_concurrent_jobs: u64::MAX,
            max_runtime_budget_ms: u64::MAX,
        };
        reg.register(TenantId(10), "limited", quotas).unwrap();

        assert_eq!(reg.check_quota(TenantId(10)), Some(true));

        // Exceed VRAM
        reg.update_vram_usage(TenantId(10), 2000);
        assert_eq!(reg.check_quota(TenantId(10)), Some(false));
    }

    #[test]
    fn test_list_tenants() {
        let mut reg = TenantRegistry::new();
        reg.register(TenantId(3), "t3", TenantQuotas::unlimited())
            .unwrap();
        reg.register(TenantId(7), "t7", TenantQuotas::unlimited())
            .unwrap();

        let mut ids = reg.list();
        ids.sort_by_key(|t| t.0);
        assert_eq!(ids, vec![TenantId(0), TenantId(3), TenantId(7)]);
    }

    #[test]
    fn test_unknown_tenant_operations() {
        let reg = TenantRegistry::new();
        assert!(reg.get(TenantId(999)).is_none());
        assert_eq!(reg.update_vram_usage(TenantId(999), 100), None);
        assert_eq!(reg.acquire_stream(TenantId(999)), None);
        assert_eq!(reg.check_quota(TenantId(999)), None);
    }
}
