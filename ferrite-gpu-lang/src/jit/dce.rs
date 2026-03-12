/// Dead Code Elimination pass on the HIR graph.
///
/// Walks backwards from the output, marks reachable nodes, and rebuilds
/// the graph with only live nodes.  IDs are compacted and remapped.

use super::hir::*;

/// Remove unreachable nodes from the graph.
pub fn dce(graph: HirGraph) -> HirGraph {
    let output = match graph.output {
        Some(o) => o,
        None => return graph,
    };

    // Mark reachable nodes by walking backwards from output.
    let mut reachable = vec![false; graph.nodes.len()];
    let mut worklist = vec![output];

    // Also mark all input nodes as reachable (they must stay).
    for &inp in &graph.inputs {
        worklist.push(inp);
    }

    while let Some(id) = worklist.pop() {
        if reachable[id.index()] {
            continue;
        }
        reachable[id.index()] = true;
        for dep in HirGraph::deps(&graph.nodes[id.index()].op) {
            if !reachable[dep.index()] {
                worklist.push(dep);
            }
        }
    }

    // Rebuild with only reachable nodes, compacting IDs.
    let mut new_graph = HirGraph::new();
    let mut id_remap: Vec<Option<HirId>> = vec![None; graph.nodes.len()];

    for (old_idx, node) in graph.nodes.iter().enumerate() {
        if !reachable[old_idx] {
            continue;
        }
        let new_op = HirGraph::remap_op(&node.op, &id_remap);
        let new_id = new_graph.push(new_op, node.ty.clone(), node.span);
        id_remap[old_idx] = Some(new_id);

        // Track inputs in the new graph
        if matches!(node.op, HirOp::Input { .. }) {
            new_graph.inputs.push(new_id);
        }
    }

    // Remap output
    new_graph.output = id_remap[output.index()];

    new_graph
}
