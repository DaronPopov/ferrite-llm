//! CUDA graph backend.
//!
//! Compiles the IR graph to a CUDA graph for efficient replay.

use std::collections::HashMap;
use std::sync::Arc;

use ptx_runtime::stream::Stream;
use ptx_runtime::{Error, GpuPtr, PtxRuntime, Result};
use ptx_tensor::DType;

use crate::backend::CompiledGraph;
use crate::ir::{Graph, Node, OpCode, TensorId, TensorMeta};
use crate::passes::MemoryPlan;

mod compile;
mod emit;

pub use compile::compile;
