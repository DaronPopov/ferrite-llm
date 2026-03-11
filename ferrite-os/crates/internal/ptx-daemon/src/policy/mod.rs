//! Policy enforcement and audit trail for the Ferrite OS control plane.
//!
//! This module provides:
//!
//! - [`decision`] -- policy decision types (`Allow` / `Deny` with reason codes)
//! - [`engine`]   -- rule evaluation engine with first-deny-wins semantics
//! - [`audit`]    -- bounded audit log with query and JSON export

pub mod audit;
pub mod decision;
pub mod engine;

#[allow(unused_imports)]
pub use audit::{AuditEntry, AuditLog};
#[allow(unused_imports)]
pub use decision::{DenialPayload, DenialReason, PolicyContext, PolicyDecision};
#[allow(unused_imports)]
pub use engine::{PolicyEngine, PolicyRule};
