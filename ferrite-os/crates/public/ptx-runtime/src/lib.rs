//! Safe Rust wrappers for PTX-OS GPU runtime.
//!
//! This crate provides safe, idiomatic Rust abstractions over the PTX-OS GPU runtime,
//! including:
//! - GPU device management
//! - Memory allocation with RAII
//! - Stream pool management
//! - CUDA graph capture and replay
//! - cuBLAS handle management
//!
//! # Example
//!
//! ```no_run
//! use ptx_runtime::PtxRuntime;
//!
//! let runtime = PtxRuntime::new(0).expect("Failed to initialize runtime");
//! let ptr = runtime.alloc(1024).expect("Failed to allocate");
//! // Memory is automatically freed when ptr goes out of scope
//! ```

pub mod device;
pub mod runtime;
pub mod stream;
pub mod memory;
pub mod graph;
pub mod cublas;
pub mod error;
pub mod stats;
pub mod telemetry;
pub mod resilience;
pub mod scheduler;
pub mod job;
pub mod multi_gpu;

pub use device::Device;
pub use runtime::PtxRuntime;
pub use runtime::{global_runtime, init_global_runtime, install_global_runtime};
pub use stream::Stream;
pub use memory::GpuPtr;
pub use graph::CudaGraph;
pub use cublas::{CublasHandle, GemmOp, Gemm};
pub use error::{Error, Result};
pub use stats::{increment_ops, get_ops_count, reset_ops_count};
pub use multi_gpu::GpuCluster;

// Re-export commonly used types from ptx-sys
pub use ptx_sys::{
    GPUHotConfig, GPUHotStats, TLSFPoolStats, TLSFHealthReport,
    PTXStableConfig, PTXStableStats, PTXStableStatus,
    PTXDType as DType, PTXTensorOpcode as OpCode,
    GPU_HOT_MAX_STREAMS, PTX_STABLE_ABI_VERSION,
    GPU_HOT_MAX_GPUS, GPUCommType, MultiTensorDtype, GPUMultiStats,
};
