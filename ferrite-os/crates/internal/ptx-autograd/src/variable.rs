//! Variable type that wraps Tensor with gradient tracking.

use std::sync::Arc;

use parking_lot::Mutex;
use smallvec::{smallvec, SmallVec};

use ptx_compiler::{OpAttrs, OpCode};
use ptx_runtime::{Error, PtxRuntime, Result};
use ptx_tensor::{DType, Tensor};

use crate::grad_fn::*;
use crate::is_grad_enabled;
use crate::tape::{get_grad, record_op, register_tensor, TapeNode, TensorId};

mod model;
mod ops;
mod backward;

/// A tensor that tracks gradients for automatic differentiation.
///
/// Variables wrap tensors and record operations for backward pass.
#[derive(Clone)]
pub struct Variable {
    /// The underlying tensor.
    tensor: Tensor,
    /// Unique ID for this tensor in the tape.
    id: TensorId,
    /// Whether this variable requires gradient.
    requires_grad: bool,
    /// Accumulated gradient (populated after backward).
    grad: Arc<Mutex<Option<Tensor>>>,
}

impl std::fmt::Debug for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Variable({:?}, requires_grad={})",
            self.tensor, self.requires_grad
        )
    }
}
