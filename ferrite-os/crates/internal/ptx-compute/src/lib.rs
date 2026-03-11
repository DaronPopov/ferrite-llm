//! High-level compute operations for PTX-OS.
//!
//! This crate provides ergonomic, high-level APIs for common GPU compute patterns,
//! built on top of the low-level `ptx-runtime` wrapper.
//!
//! # Modules
//!
//! - [`gemm`] - Matrix multiplication operations
//! - [`neural`] - Neural network forward pass primitives
//! - [`reduction`] - Parallel reduction operations (sum, mean, max, min)
//! - [`monte_carlo`] - Monte Carlo simulation utilities
//! - [`tiling`] - Custom compute tiling (like CuTe) for optimal performance
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use ptx_runtime::PtxRuntime;
//! use ptx_compute::gemm::Matmul;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let runtime = Arc::new(PtxRuntime::new(0)?);
//! let matmul = Matmul::new(&runtime)?;
//!
//! // Allocate matrices
//! let (m, n, k) = (1024, 1024, 1024);
//! let a = runtime.alloc(m * k * 4)?;
//! let b = runtime.alloc(k * n * 4)?;
//! let c = runtime.alloc(m * n * 4)?;
//!
//! // Perform matrix multiplication: C = A @ B
//! unsafe {
//!     matmul.multiply_f32(
//!         a.as_ptr() as *const f32,
//!         b.as_ptr() as *const f32,
//!         c.as_ptr() as *mut f32,
//!         m, n, k
//!     )?;
//! }
//! # Ok(())
//! # }
//! ```

pub mod gemm;
pub mod neural;
pub mod reduction;
pub mod monte_carlo;
pub mod tiling;

pub use ptx_runtime::{Error, Result};

/// Calculate TFLOPS from operation count and elapsed time.
pub fn calculate_tflops(flops: f64, elapsed_secs: f64) -> f64 {
    flops / elapsed_secs / 1e12
}

/// Calculate FLOPS for matrix multiplication: C = A @ B
/// where A is (m, k), B is (k, n), C is (m, n)
pub fn matmul_flops(m: usize, n: usize, k: usize) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64
}

/// Calculate memory bandwidth from bytes transferred and elapsed time.
pub fn calculate_bandwidth_gbps(bytes: usize, elapsed_secs: f64) -> f64 {
    bytes as f64 / elapsed_secs / 1e9
}
