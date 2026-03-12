//! Backward-compatible re-exports from the generic `pipeline` module.
//!
//! The Tensor-typed `RingBuffer` is a type alias to `pipeline::RingBuffer<tch::Tensor>`.
//! `StageMetrics` and `PipelineStats` are re-exported directly.

use tch::Tensor;

/// Tensor-typed ring buffer (backward compat alias).
pub type RingBuffer = crate::pipeline::RingBuffer<Tensor>;

pub use crate::pipeline::stage::{StageMetrics, PipelineStats};
