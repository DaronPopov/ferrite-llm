//! Memory planning pass.
//!
//! Assigns memory buffers to tensors, reusing memory where lifetimes don't overlap.

use std::collections::HashMap;

use crate::ir::{Graph, TensorId};
use ptx_runtime::Result;

/// A memory buffer that can be shared by multiple tensors.
#[derive(Debug, Clone)]
pub struct MemoryBuffer {
    /// Buffer ID.
    pub id: usize,
    /// Size in bytes.
    pub size: usize,
    /// Tensors assigned to this buffer.
    pub tensors: Vec<TensorId>,
}

/// Memory allocation plan for the graph.
#[derive(Debug)]
pub struct MemoryPlan {
    /// Memory buffers.
    pub buffers: Vec<MemoryBuffer>,
    /// Tensor to buffer mapping.
    pub tensor_to_buffer: HashMap<TensorId, usize>,
    /// Tensor to offset within buffer.
    pub tensor_to_offset: HashMap<TensorId, usize>,
    /// Total memory required.
    pub total_memory: usize,
}

impl MemoryPlan {
    /// Get the buffer ID for a tensor.
    pub fn buffer_for(&self, tensor_id: TensorId) -> Option<usize> {
        self.tensor_to_buffer.get(&tensor_id).copied()
    }

    /// Get the offset within a buffer for a tensor.
    pub fn offset_for(&self, tensor_id: TensorId) -> Option<usize> {
        self.tensor_to_offset.get(&tensor_id).copied()
    }
}

/// Compute the liveness interval for each tensor.
fn compute_liveness(graph: &Graph) -> HashMap<TensorId, (usize, usize)> {
    let mut liveness: HashMap<TensorId, (usize, usize)> = HashMap::new();
    let topo_order = graph.topo_order();

    for (step, &node_id) in topo_order.iter().enumerate() {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => continue,
        };

        // Update liveness for output
        liveness.insert(node.output, (step, step));

        // Update end time for inputs
        for &input_id in &node.inputs {
            if let Some(entry) = liveness.get_mut(&input_id) {
                entry.1 = step;
            }
        }
    }

    // Extend liveness for output tensors to the end
    let last_step = topo_order.len();
    for &output_id in graph.outputs() {
        if let Some(entry) = liveness.get_mut(&output_id) {
            entry.1 = last_step;
        }
    }

    liveness
}

/// Check if two liveness intervals overlap.
fn intervals_overlap(a: (usize, usize), b: (usize, usize)) -> bool {
    !(a.1 < b.0 || b.1 < a.0)
}

/// Plan memory allocations for the graph.
pub fn plan(graph: &Graph) -> Result<MemoryPlan> {
    let liveness = compute_liveness(graph);

    // Simple greedy allocation strategy:
    // For each tensor, try to reuse an existing buffer that doesn't overlap
    // Otherwise, allocate a new buffer

    let mut buffers: Vec<MemoryBuffer> = Vec::new();
    let mut tensor_to_buffer: HashMap<TensorId, usize> = HashMap::new();
    let mut tensor_to_offset: HashMap<TensorId, usize> = HashMap::new();

    // Sort tensors by start time
    let mut tensors: Vec<_> = graph.tensors().keys().copied().collect();
    tensors.sort_by_key(|&tid| liveness.get(&tid).map(|l| l.0).unwrap_or(0));

    for tensor_id in tensors {
        let tensor_meta = match graph.tensor(tensor_id) {
            Some(m) => m,
            None => continue,
        };

        // Skip input tensors (they have external storage)
        if tensor_meta.is_input {
            continue;
        }

        let size = tensor_meta.size_bytes();
        let interval = liveness.get(&tensor_id).copied().unwrap_or((0, 0));

        // Try to find an existing buffer we can reuse
        let mut found_buffer = None;
        for (buf_id, buffer) in buffers.iter_mut().enumerate() {
            // Check if any tensor in this buffer overlaps
            let overlaps = buffer.tensors.iter().any(|&other_id| {
                let other_interval = liveness.get(&other_id).copied().unwrap_or((0, 0));
                intervals_overlap(interval, other_interval)
            });

            if !overlaps && buffer.size >= size {
                found_buffer = Some(buf_id);
                break;
            }
        }

        let buffer_id = match found_buffer {
            Some(id) => {
                buffers[id].tensors.push(tensor_id);
                // Grow buffer if needed
                if size > buffers[id].size {
                    buffers[id].size = size;
                }
                id
            }
            None => {
                // Allocate new buffer
                let id = buffers.len();
                buffers.push(MemoryBuffer {
                    id,
                    size,
                    tensors: vec![tensor_id],
                });
                id
            }
        };

        tensor_to_buffer.insert(tensor_id, buffer_id);
        tensor_to_offset.insert(tensor_id, 0);  // Simple: no offset within buffer
    }

    let total_memory: usize = buffers.iter().map(|b| b.size).sum();

    Ok(MemoryPlan {
        buffers,
        tensor_to_buffer,
        tensor_to_offset,
        total_memory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptx_tensor::DType;

    #[test]
    fn test_memory_planning() {
        let mut graph = Graph::new();
        let a = graph.input(&[2, 3], DType::F32);
        let b = graph.input(&[2, 3], DType::F32);
        let c = graph.add(a, b);
        let d = graph.relu(c);
        graph.mark_output(d);

        let plan = plan(&graph).unwrap();

        // Should have allocated buffers for c and d
        // c and d can potentially share a buffer since c is not used after d
        assert!(plan.buffers.len() <= 2);
    }
}
