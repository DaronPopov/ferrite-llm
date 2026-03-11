//! `ptx-render` — hierarchical GPU render stack over TLSF allocator.
//!
//! Four-layer architecture:
//!
//! | Level | Module     | Responsibility                        |
//! |-------|------------|---------------------------------------|
//! | 0     | `arena`    | GPU scratch arena (bump over TLSF)    |
//! | 1     | `dispatch` | Typed kernel dispatch wrappers        |
//! | 2     | `state`    | Pipeline state (MVP + lights on GPU)  |
//! | 3     | `context`  | Frame-level render orchestrator       |
//!
//! **Zero per-frame TLSF allocations.** The scratch arena bump-resets
//! each frame; persistent buffers are allocated once at init.

pub mod arena;
pub mod dispatch;
pub mod state;
pub mod context;

pub use arena::{GpuSlice, ScratchArena};
pub use context::{RenderConfig, RenderContext};
pub use state::PipelineState;
