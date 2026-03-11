//! Computation tape for recording operations.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use smallvec::SmallVec;

use ptx_tensor::{Tensor, DType};
use ptx_compiler::{OpCode, OpAttrs};
use crate::grad_fn::GradFn;

/// Unique identifier for tensors in the tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub u64);

impl TensorId {
    /// Generate a new unique tensor ID.
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        TensorId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for TensorId {
    fn default() -> Self {
        Self::new()
    }
}

/// A node in the computation tape.
pub struct TapeNode {
    /// The output tensor ID.
    pub output_id: TensorId,
    /// Operation code (for compiler integration).
    pub op: OpCode,
    /// Operation attributes (scalars, dimensions, etc.).
    pub attrs: OpAttrs,
    /// The gradient function.
    pub grad_fn: Box<dyn GradFn>,
    /// Input tensor IDs.
    pub inputs: SmallVec<[TensorId; 4]>,
    /// Saved tensors for backward pass.
    pub saved_tensors: SmallVec<[Tensor; 4]>,
}

impl TapeNode {
    /// Create a new tape node.
    pub fn new(
        output_id: TensorId,
        op: OpCode,
        attrs: OpAttrs,
        grad_fn: Box<dyn GradFn>,
        inputs: SmallVec<[TensorId; 4]>,
        saved_tensors: SmallVec<[Tensor; 4]>,
    ) -> Self {
        Self {
            output_id,
            op,
            attrs,
            grad_fn,
            inputs,
            saved_tensors,
        }
    }
}

/// Lightweight tensor metadata for compiler tracing.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// The computation tape that records operations.
pub struct Tape {
    /// Recorded operations.
    nodes: Vec<TapeNode>,
    /// Accumulated gradients.
    grads: HashMap<TensorId, Tensor>,
    /// Tensor metadata (shape/dtype) for compiler integration.
    tensor_info: HashMap<TensorId, TensorInfo>,
}

impl Tape {
    /// Create a new empty tape.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            grads: HashMap::new(),
            tensor_info: HashMap::new(),
        }
    }

    /// Record an operation on the tape.
    pub fn record(&mut self, node: TapeNode) {
        self.nodes.push(node);
    }

    /// Get the number of recorded operations.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tape is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear the tape.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.grads.clear();
        self.tensor_info.clear();
    }

    /// Get the accumulated gradient for a tensor.
    pub fn get_grad(&self, id: TensorId) -> Option<&Tensor> {
        self.grads.get(&id)
    }

    /// Set the gradient for a tensor.
    pub fn set_grad(&mut self, id: TensorId, grad: Tensor) {
        self.grads.insert(id, grad);
    }

    /// Accumulate gradient for a tensor.
    pub fn accumulate_grad(&mut self, id: TensorId, grad: Tensor) -> ptx_runtime::Result<()> {
        if let Some(existing) = self.grads.get_mut(&id) {
            // Add to existing gradient
            *existing = existing.add(&grad)?;
        } else {
            self.grads.insert(id, grad);
        }
        Ok(())
    }

    /// Get access to all nodes (for backward pass).
    pub fn nodes(&self) -> &[TapeNode] {
        &self.nodes
    }

    /// Get mutable access to all nodes.
    pub fn nodes_mut(&mut self) -> &mut Vec<TapeNode> {
        &mut self.nodes
    }

    /// Take ownership of the gradients.
    pub fn take_grads(&mut self) -> HashMap<TensorId, Tensor> {
        std::mem::take(&mut self.grads)
    }

    /// Register tensor metadata (shape/dtype).
    pub fn register_tensor(&mut self, id: TensorId, tensor: &Tensor) {
        self.tensor_info.insert(
            id,
            TensorInfo {
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype(),
            },
        );
    }

    /// Get tensor metadata by ID.
    pub fn tensor_info(&self, id: TensorId) -> Option<&TensorInfo> {
        self.tensor_info.get(&id)
    }

    /// Iterate over nodes in reverse order (for backward pass).
    pub fn iter_reverse(&self) -> impl Iterator<Item = &TapeNode> {
        self.nodes.iter().rev()
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

// Global thread-local tape
thread_local! {
    /// The global computation tape for the current thread.
    pub static TAPE: Mutex<Tape> = Mutex::new(Tape::new());
}

/// Record an operation on the global tape.
pub fn record_op(node: TapeNode) {
    TAPE.with(|tape| {
        tape.lock().record(node);
    });
}

/// Register tensor metadata on the global tape.
pub fn register_tensor(id: TensorId, tensor: &Tensor) {
    TAPE.with(|tape| {
        tape.lock().register_tensor(id, tensor);
    });
}

/// Clear the global tape.
pub fn clear_tape() {
    TAPE.with(|tape| {
        tape.lock().clear();
    });
}

/// Get the gradient for a tensor from the global tape.
pub fn get_grad(id: TensorId) -> Option<Tensor> {
    TAPE.with(|tape| {
        tape.lock().get_grad(id).cloned()
    })
}

/// Set the gradient for a tensor on the global tape.
pub fn set_grad(id: TensorId, grad: Tensor) {
    TAPE.with(|tape| {
        tape.lock().set_grad(id, grad);
    });
}
