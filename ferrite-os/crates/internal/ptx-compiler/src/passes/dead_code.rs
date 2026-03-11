//! Dead code elimination pass.
//!
//! Removes nodes whose outputs are not used by any other node or marked as
//! graph outputs. Works backward from outputs to identify live tensors, then
//! strips everything else.

use std::collections::HashSet;

use crate::ir::{Graph, TensorId};
use ptx_runtime::Result;

/// Eliminate dead code from the graph.
///
/// A node is considered dead if its output tensor is not:
/// 1. Used as input to another live node
/// 2. Marked as a graph output
///
/// Graph inputs are always preserved to keep the backend contract stable.
pub fn eliminate_dead_code(mut graph: Graph) -> Result<Graph> {
    let mut live_tensors: HashSet<TensorId> = HashSet::new();

    // Graph outputs are roots of liveness.
    for &output_id in graph.outputs() {
        live_tensors.insert(output_id);
    }

    // Graph inputs are unconditionally live (backend stability).
    for &input_id in graph.inputs() {
        live_tensors.insert(input_id);
    }

    // Propagate liveness backward through the graph until fixed-point.
    let nodes: Vec<_> = graph
        .nodes()
        .iter()
        .map(|(id, n)| (*id, n.clone()))
        .collect();

    loop {
        let prev_size = live_tensors.len();

        for (_, node) in &nodes {
            if live_tensors.contains(&node.output) {
                for input_id in &node.inputs {
                    live_tensors.insert(*input_id);
                }
            }
        }

        if live_tensors.len() == prev_size {
            break;
        }
    }

    // Remove dead nodes and orphaned tensors.
    graph.remove_dead_nodes(&live_tensors);

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptx_tensor::DType;

    #[test]
    fn test_dce_removes_dead_node() {
        let mut graph = Graph::new();
        let a = graph.input(&[2, 3], DType::F32);
        let b = graph.input(&[2, 3], DType::F32);
        let c = graph.add(a, b);
        let _d = graph.mul(a, b); // Dead — not used
        graph.mark_output(c);

        assert_eq!(graph.num_nodes(), 4); // input(a), input(b), add(c), mul(d)
        let optimized = eliminate_dead_code(graph).unwrap();
        assert_eq!(optimized.outputs().len(), 1);

        // The mul node should have been removed.
        // Remaining: 2 input nodes + 1 add node = 3.
        assert_eq!(optimized.num_nodes(), 3);
    }

    #[test]
    fn test_dce_preserves_all_live() {
        let mut graph = Graph::new();
        let a = graph.input(&[4], DType::F32);
        let b = graph.relu(a);
        graph.mark_output(b);

        let before = graph.num_nodes();
        let optimized = eliminate_dead_code(graph).unwrap();
        assert_eq!(optimized.num_nodes(), before);
    }

    #[test]
    fn test_dce_chain_dead() {
        // a(input) -> b=neg -> c=exp (dead chain, only a is output)
        let mut graph = Graph::new();
        let a = graph.input(&[2], DType::F32);
        let b = graph.neg(a);
        let _c = graph.exp(b); // dead
        graph.mark_output(a);

        let optimized = eliminate_dead_code(graph).unwrap();
        // Only the input node should remain.
        assert_eq!(optimized.num_nodes(), 1);
    }

    #[test]
    fn test_dce_idempotent() {
        let mut graph = Graph::new();
        let a = graph.input(&[2, 3], DType::F32);
        let b = graph.input(&[2, 3], DType::F32);
        let c = graph.add(a, b);
        let _d = graph.mul(a, b); // dead
        graph.mark_output(c);

        let pass1 = eliminate_dead_code(graph).unwrap();
        let n1 = pass1.num_nodes();
        let pass2 = eliminate_dead_code(pass1).unwrap();
        let n2 = pass2.num_nodes();

        assert_eq!(n1, n2, "second pass should be a no-op");
    }

    #[test]
    fn test_dce_removes_dead_constants() {
        let mut graph = Graph::new();
        let a = graph.input(&[4], DType::F32);
        let _unused_const = graph.constant(&[4], DType::F32, Some(42.0)); // dead
        let b = graph.relu(a);
        graph.mark_output(b);

        let optimized = eliminate_dead_code(graph).unwrap();
        // input(a) + relu(b) = 2 nodes; the constant node is removed.
        assert_eq!(optimized.num_nodes(), 2);
    }
}
