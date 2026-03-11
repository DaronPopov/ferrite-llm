//! Automatic differentiation for PTX-OS tensor library.
//!
//! This crate provides reverse-mode automatic differentiation (autograd) for
//! training neural networks. It records operations in a computation tape and
//! computes gradients via backpropagation.
//!
//! # Example
//!
//! ```no_run
//! use ptx_autograd::{Variable, no_grad};
//! use ptx_tensor::{Tensor, DType};
//! use ptx_runtime::PtxRuntime;
//! use std::sync::Arc;
//!
//! let runtime = Arc::new(PtxRuntime::new(0).unwrap());
//!
//! // Create variables that track gradients
//! let x = Variable::new(Tensor::ones(&[2, 3], DType::F32, &runtime).unwrap(), true);
//! let w = Variable::new(Tensor::zeros(&[2, 3], DType::F32, &runtime).unwrap(), true);
//!
//! // Forward pass
//! let y = x.add(&w).unwrap();
//! let loss = y.sum_all().unwrap();
//!
//! // Backward pass
//! loss.backward();
//!
//! // Access gradients
//! let x_grad = x.grad();
//! let w_grad = w.grad();
//! ```

pub mod tape;
pub mod backward;
pub mod grad_fn;
pub mod variable;
pub mod ops;
pub mod compiler;

pub use tape::{Tape, TapeNode, TAPE};
pub use backward::{backward, backward_with_create_graph};
pub use grad_fn::GradFn;
pub use variable::Variable;
pub use compiler::{compile_from_tape, compile_from_tape_with_inputs};

use std::cell::RefCell;

thread_local! {
    /// Flag to disable gradient computation.
    static GRAD_ENABLED: RefCell<bool> = RefCell::new(true);
}

/// Check if gradient computation is enabled.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|f| *f.borrow())
}

/// Set whether gradient computation is enabled.
pub fn set_grad_enabled(enabled: bool) {
    GRAD_ENABLED.with(|f| *f.borrow_mut() = enabled);
}

/// Context manager for disabling gradient computation.
pub struct NoGrad {
    prev: bool,
}

impl NoGrad {
    /// Create a new no-grad context.
    pub fn new() -> Self {
        let prev = is_grad_enabled();
        set_grad_enabled(false);
        Self { prev }
    }
}

impl Default for NoGrad {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGrad {
    fn drop(&mut self) {
        set_grad_enabled(self.prev);
    }
}

/// Execute a closure with gradient computation disabled.
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGrad::new();
    f()
}

/// Execute a closure with gradient computation enabled.
pub fn enable_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let prev = is_grad_enabled();
    set_grad_enabled(true);
    let result = f();
    set_grad_enabled(prev);
    result
}
