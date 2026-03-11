//! Computation graph container.

use std::sync::Arc;

use indexmap::IndexMap;
use smallvec::SmallVec;

use ptx_runtime::{PtxRuntime, Result};
use ptx_tensor::DType;

use crate::backend::CompiledGraph;
use crate::ir::node::{Node, NodeId, OpAttrs, OpCode};
use crate::ir::tensor::{TensorId, TensorMeta};
use crate::passes;

mod model_ops;
mod analysis;
mod pipeline;

/// A computation graph representing tensor operations.
#[derive(Debug)]
pub struct Graph {
    /// Tensor metadata indexed by ID.
    tensors: IndexMap<TensorId, TensorMeta>,
    /// Operation nodes indexed by ID.
    nodes: IndexMap<NodeId, Node>,
    /// Input tensor IDs.
    inputs: Vec<TensorId>,
    /// Output tensor IDs.
    outputs: Vec<TensorId>,
    /// Next tensor ID.
    next_tensor_id: u32,
    /// Next node ID.
    next_node_id: u32,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
