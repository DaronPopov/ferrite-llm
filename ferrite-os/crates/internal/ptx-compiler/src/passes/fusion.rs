//! Operator fusion pass.
//!
//! Fuses chains of elementwise operations into single kernels.

use std::collections::HashSet;

use crate::ir::{Graph, NodeId, TensorId};
use ptx_runtime::Result;

/// A group of nodes that can be fused into a single kernel.
#[derive(Debug)]
pub struct FusionGroup {
    /// Node IDs in the group (in execution order).
    pub nodes: Vec<NodeId>,
    /// Input tensor IDs (from outside the group).
    pub inputs: Vec<TensorId>,
    /// Output tensor ID.
    pub output: TensorId,
}

/// Fuse chains of elementwise operations.
///
/// Patterns like `relu(add(a, b))` can be fused into a single kernel,
/// avoiding intermediate memory allocations and improving cache locality.
pub fn fuse_elementwise(graph: Graph) -> Result<Graph> {
    // Find fusion opportunities
    let groups = find_fusion_groups(&graph);

    // For now, we just identify groups but don't actually fuse
    // A full implementation would:
    // 1. Generate fused kernel code
    // 2. Replace the group of nodes with a single fused node
    // 3. Update tensor references

    if !groups.is_empty() {
        // Log fusion opportunities (in a real implementation)
        // println!("Found {} fusion groups", groups.len());
    }

    Ok(graph)
}

/// Find groups of nodes that can be fused.
fn find_fusion_groups(graph: &Graph) -> Vec<FusionGroup> {
    let mut groups = Vec::new();
    let mut visited: HashSet<NodeId> = HashSet::new();

    // Process nodes in topological order
    for node_id in graph.topo_order() {
        if visited.contains(&node_id) {
            continue;
        }

        let node = match graph.node(node_id) {
            Some(n) => n,
            None => continue,
        };

        // Skip non-elementwise ops
        if !node.op.is_elementwise() {
            continue;
        }

        // Try to build a fusion group starting from this node
        if let Some(group) = try_build_group(graph, node_id, &visited) {
            for &nid in &group.nodes {
                visited.insert(nid);
            }
            groups.push(group);
        }
    }

    groups
}

/// Try to build a fusion group starting from a node.
fn try_build_group(
    graph: &Graph,
    start_id: NodeId,
    visited: &HashSet<NodeId>,
) -> Option<FusionGroup> {
    let mut group_nodes = vec![start_id];
    let mut external_inputs: HashSet<TensorId> = HashSet::new();

    let start_node = graph.node(start_id)?;

    // Collect external inputs
    for &input_id in &start_node.inputs {
        external_inputs.insert(input_id);
    }

    // Try to extend the group forward (to consumers)
    let mut current = start_id;
    loop {
        let node = graph.node(current)?;
        let consumers = graph.consumers(node.output);

        // Can only extend if there's exactly one consumer
        if consumers.len() != 1 {
            break;
        }

        let consumer_id = consumers[0];
        if visited.contains(&consumer_id) {
            break;
        }

        let consumer = graph.node(consumer_id)?;

        // Consumer must be elementwise and have same shape
        if !consumer.op.is_elementwise() {
            break;
        }

        // Check shape compatibility
        let node_shape = &graph.tensor(node.output)?.shape;
        let consumer_out_shape = &graph.tensor(consumer.output)?.shape;
        if node_shape != consumer_out_shape {
            break;
        }

        // Add consumer's other inputs to external inputs
        for &input_id in &consumer.inputs {
            if input_id != node.output {
                external_inputs.insert(input_id);
            }
        }

        group_nodes.push(consumer_id);
        current = consumer_id;
    }

    // Only return if we found a group of at least 2 nodes
    if group_nodes.len() >= 2 {
        let last_node = graph.node(*group_nodes.last()?)?;
        Some(FusionGroup {
            nodes: group_nodes,
            inputs: external_inputs.into_iter().collect(),
            output: last_node.output,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptx_tensor::DType;

    #[test]
    fn test_fusion_detection() {
        let mut graph = Graph::new();
        let a = graph.input(&[2, 3], DType::F32);
        let b = graph.input(&[2, 3], DType::F32);
        let c = graph.add(a, b);      // Fusable
        let d = graph.relu(c);        // Fusable with add
        let e = graph.mul(d, b);      // Fusable with relu
        graph.mark_output(e);

        let groups = find_fusion_groups(&graph);
        // Should find a group: add -> relu -> mul
        assert!(!groups.is_empty());
    }
}
