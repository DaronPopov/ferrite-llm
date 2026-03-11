//! PTX task dispatch backend.
//!
//! Alternative backend that dispatches operations to the PTX-OS task queue
//! instead of capturing to a CUDA graph.

use std::sync::Arc;

use ptx_runtime::{PtxRuntime, Result, Error};

use crate::ir::{Graph, Node, OpCode};
use crate::passes::MemoryPlan;

/// Dispatch a graph to the PTX-OS task queue.
///
/// This is an alternative to CUDA graph capture that uses the persistent
/// GPU kernel's task queue for scheduling.
pub fn dispatch(
    graph: &Graph,
    _memory_plan: &MemoryPlan,
    _runtime: &Arc<PtxRuntime>,
) -> Result<()> {
    // Build PTXTensorOp descriptors for each node
    for node_id in graph.topo_order() {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => continue,
        };

        if node.is_input() || node.is_constant() {
            continue;
        }

        // Create operation descriptor
        let op = build_tensor_op(graph, node)?;

        // Dispatch to task queue
        let err = unsafe { ptx_sys::ptx_tensor_dispatch(&op) };
        if err != 0 {
            return Err(Error::CudaError {
                code: err,
                message: "Task dispatch failed".to_string(),
            });
        }
    }

    Ok(())
}

/// Build a PTXTensorOp from a graph node.
fn build_tensor_op(graph: &Graph, node: &Node) -> Result<ptx_sys::PTXTensorOp> {
    let output_meta = graph.tensor(node.output).ok_or_else(|| Error::Internal {
        message: "Missing output metadata".to_string(),
    })?;

    let opcode = match node.op {
        OpCode::Add => ptx_sys::PTXTensorOpcode::Add,
        OpCode::Sub => ptx_sys::PTXTensorOpcode::Sub,
        OpCode::Mul => ptx_sys::PTXTensorOpcode::Mul,
        OpCode::Div => ptx_sys::PTXTensorOpcode::Div,
        OpCode::Neg => ptx_sys::PTXTensorOpcode::Neg,
        OpCode::Exp => ptx_sys::PTXTensorOpcode::Exp,
        OpCode::Log => ptx_sys::PTXTensorOpcode::Log,
        OpCode::Sqrt => ptx_sys::PTXTensorOpcode::Sqrt,
        OpCode::Tanh => ptx_sys::PTXTensorOpcode::Tanh,
        OpCode::Relu => ptx_sys::PTXTensorOpcode::Relu,
        OpCode::Gelu => ptx_sys::PTXTensorOpcode::Gelu,
        OpCode::Sigmoid => ptx_sys::PTXTensorOpcode::Sigmoid,
        OpCode::Silu => ptx_sys::PTXTensorOpcode::Silu,
        OpCode::ReduceSum => ptx_sys::PTXTensorOpcode::ReduceSum,
        OpCode::ReduceMean => ptx_sys::PTXTensorOpcode::ReduceMean,
        OpCode::Softmax => ptx_sys::PTXTensorOpcode::Softmax,
        _ => return Err(Error::NotSupported {
            message: format!("Opcode {:?} not supported in dispatch", node.op),
        }),
    };

    let dtype = output_meta.dtype.to_ptx();

    let op = ptx_sys::PTXTensorOp {
        opcode,
        dtype,
        input_a: std::ptr::null_mut(),
        input_b: std::ptr::null_mut(),
        output: std::ptr::null_mut(),
        elem_count: output_meta.elem_count(),
        scalar_a: node.attrs.scalar_a.unwrap_or(0.0),
        scalar_b: node.attrs.scalar_b.unwrap_or(0.0),
        reduce_dim: node.attrs.reduce_dim.unwrap_or(0) as u32,
        reduce_size: 0,
        outer_size: 0,
        inner_size: 0,
        stream: std::ptr::null_mut(),
    };

    // Note: Actual pointers would be set at execution time
    // This is a template for the operation

    Ok(op)
}
