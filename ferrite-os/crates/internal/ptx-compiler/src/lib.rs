//! Graph compiler for PTX-OS tensor computations.
//!
//! This crate provides:
//! - IR (Intermediate Representation) for tensor computations
//! - Optimization passes (dead code elimination, constant folding, fusion)
//! - Backend code generation (CUDA graphs, PTX task dispatch)
//!
//! # Example
//!
//! ```no_run
//! use ptx_compiler::{Graph, OpCode, DType};
//! use ptx_runtime::PtxRuntime;
//! use std::sync::Arc;
//!
//! let runtime = Arc::new(PtxRuntime::new(0).unwrap());
//!
//! // Build a computation graph
//! let mut graph = Graph::new();
//! let a = graph.input(&[2, 3], DType::F32);
//! let b = graph.input(&[2, 3], DType::F32);
//! let c = graph.add(a, b);
//! let d = graph.relu(c);
//! graph.mark_output(d);
//!
//! // Compile and execute
//! let compiled = graph.compile(&runtime).unwrap();
//! // compiled.execute(inputs) ...
//! ```

pub mod ir;
pub mod passes;
pub mod backend;

pub use ir::{Graph, Node, NodeId, TensorId, TensorMeta, OpAttrs};
pub use ir::node::OpCode;
pub use ptx_tensor::DType;
pub use backend::CompiledGraph;

// Re-export runtime
pub use ptx_runtime::{PtxRuntime, Result, Error};
