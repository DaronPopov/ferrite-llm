//! Tensor library backed by PTX-OS GPU runtime.
//!
//! This crate provides a high-level tensor API similar to PyTorch or NumPy,
//! with GPU acceleration through PTX-OS.
//!
//! # Example
//!
//! ```no_run
//! use ptx_tensor::{Tensor, DType};
//! use ptx_runtime::PtxRuntime;
//! use std::sync::Arc;
//!
//! let runtime = Arc::new(PtxRuntime::new(0).unwrap());
//! let a = Tensor::zeros(&[2, 3], DType::F32, &runtime).unwrap();
//! let b = Tensor::ones(&[2, 3], DType::F32, &runtime).unwrap();
//! let c = a.add(&b).unwrap();  // Element-wise addition
//! ```

pub mod dtype;
pub mod shape;
pub mod storage;
pub mod tensor;
pub mod ops;

pub use dtype::DType;
pub use shape::{Shape, Strides};
pub use storage::Storage;
pub use tensor::Tensor;
pub use ops::loss::Reduction;

// Re-export runtime for convenience
pub use ptx_runtime::{PtxRuntime, Error, Result};

// Re-export bytemuck so downstream crates can use Pod/Zeroable bounds
pub use bytemuck;
