//! Backend code generation.

pub mod cuda_graph;
pub mod ptx_dispatch;

use std::sync::Arc;

use ptx_runtime::{PtxRuntime, GpuPtr, Result, Error};
use ptx_tensor::{Tensor, Shape};
use ptx_tensor::shape::contiguous_strides;
use ptx_tensor::storage::Storage;

use crate::ir::TensorId;
use crate::ir::TensorMeta;
use crate::passes::MemoryPlan;

/// A compiled computation graph ready for execution.
pub struct CompiledGraph {
    /// CUDA graph ID (if using graph capture).
    pub graph_id: Option<i32>,
    /// Memory plan.
    pub memory_plan: MemoryPlan,
    /// Allocated buffers.
    pub buffers: Vec<Arc<GpuPtr>>,
    /// Input buffers (one per graph input, in input order).
    pub input_buffers: Vec<Arc<GpuPtr>>,
    /// Input tensor IDs.
    pub input_ids: Vec<TensorId>,
    /// Input tensor metadata.
    pub input_metas: Vec<TensorMeta>,
    /// Output tensor IDs.
    pub output_ids: Vec<TensorId>,
    /// Output tensor metadata.
    pub output_metas: Vec<TensorMeta>,
    /// Runtime reference.
    pub runtime: Arc<PtxRuntime>,
}

impl CompiledGraph {
    /// Execute the compiled graph with the given inputs.
    pub fn execute(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.len() != self.input_metas.len() {
            return Err(Error::Internal {
                message: format!(
                    "Expected {} inputs, got {}",
                    self.input_metas.len(),
                    inputs.len()
                ),
            });
        }

        // Copy inputs into internal input buffers
        for (i, input) in inputs.iter().enumerate() {
            let meta = &self.input_metas[i];
            if input.dtype() != meta.dtype {
                return Err(Error::DTypeMismatch {
                    expected: meta.dtype.to_ptx(),
                    actual: input.dtype().to_ptx(),
                });
            }
            if input.shape() != meta.shape.as_slice() {
                return Err(Error::ShapeMismatch {
                    expected: meta.shape.to_vec(),
                    actual: input.shape().to_vec(),
                });
            }
            if !input.is_contiguous() {
                return Err(Error::NotSupported {
                    message: "execute requires contiguous inputs".to_string(),
                });
            }

            let dst = self.input_buffers[i].as_ptr();
            let src = input.data_ptr();
            let bytes = meta.size_bytes();
            let err = unsafe {
                ptx_sys::cudaMemcpy(
                    dst,
                    src,
                    bytes,
                    ptx_sys::cudaMemcpyDeviceToDevice,
                )
            };
            Error::check_cuda(err)?;
        }

        // Launch the CUDA graph
        if let Some(graph_id) = self.graph_id {
            let stream = self.runtime.stream(0)?;
            unsafe {
                ptx_sys::gpu_hot_launch_graph(
                    self.runtime.raw(),
                    graph_id,
                    stream.raw(),
                );
            }
        }

        // Synchronize
        self.runtime.sync_all()?;

        // Extract outputs as tensor views into output buffers
        let mut outputs = Vec::with_capacity(self.output_ids.len());
        for (i, output_id) in self.output_ids.iter().enumerate() {
            let meta = &self.output_metas[i];
            let buf = if let Some(buffer_id) = self.memory_plan.buffer_for(*output_id) {
                Arc::clone(&self.buffers[buffer_id])
            } else if let Some(idx) = self.input_ids.iter().position(|id| id == output_id) {
                Arc::clone(&self.input_buffers[idx])
            } else {
                return Err(Error::Internal {
                    message: "No buffer assigned for output".to_string(),
                });
            };
            let storage = Storage::from_gpu_ptr(
                buf,
                meta.elem_count(),
                meta.dtype,
                &self.runtime,
            );
            let shape = Shape::from_slice(&meta.shape);
            let strides = contiguous_strides(&shape);
            let tensor = Tensor::from_storage(storage, shape, strides, 0);
            outputs.push(tensor);
        }

        Ok(outputs)
    }

    /// Get the total memory used by this graph.
    pub fn memory_usage(&self) -> usize {
        let buffer_bytes: usize = self.buffers.iter().map(|b| b.size()).sum();
        let input_bytes: usize = self.input_buffers.iter().map(|b| b.size()).sum();
        buffer_bytes + input_bytes
    }
}

impl Drop for CompiledGraph {
    fn drop(&mut self) {
        // Destroy the CUDA graph if it exists
        if let Some(graph_id) = self.graph_id {
            unsafe {
                ptx_sys::gpu_hot_destroy_graph(self.runtime.raw(), graph_id);
            }
        }
    }
}
