/// Elementwise fusion pass on the HIR computation graph.
///
/// Detects patterns where a unary activation immediately follows a binary
/// operation and replaces the two nodes with a single fused node, eliminating
/// one intermediate VRAM allocation.
///
/// Supported patterns:
///   relu(add(a, b))    → FusedReluAdd(a, b)
///   relu(mul(a, b))    → FusedReluMul(a, b)
///   sigmoid(add(a, b)) → FusedSigmoidAdd(a, b)
///   tanh(add(a, b))    → FusedTanhAdd(a, b)
///   gelu(add(a, b))    → FusedGeluAdd(a, b)
///   silu(add(a, b))    → FusedSiluAdd(a, b)
///   silu(mul(a, b))    → FusedSiluMul(a, b)

use std::collections::HashSet;

use super::hir::*;

/// Run fusion on a `HirGraph`, returning a new graph with fused ops.
pub fn fuse(graph: HirGraph) -> HirGraph {
    let nodes = &graph.nodes;
    let output = graph.output;

    if nodes.len() < 2 {
        return graph;
    }

    // Build use-count
    let use_count = graph.use_counts();

    // Track which nodes have been absorbed into a fused node.
    let mut absorbed: HashSet<usize> = HashSet::new();

    // Build new node list, attempting fusion for each unary activation.
    let mut new_nodes: Vec<HirNode> = Vec::with_capacity(nodes.len());
    let mut remap: Vec<Option<HirId>> = vec![None; nodes.len()];

    for (i, node) in nodes.iter().enumerate() {
        if absorbed.contains(&i) {
            // This node was fused into a prior node — emit as placeholder
            // that preserves index mapping.
            new_nodes.push(node.clone());
            remap[i] = Some(HirId(new_nodes.len() - 1));
            continue;
        }

        // Don't fuse across reduction barriers
        if HirGraph::is_reduction(&node.op) {
            let remapped = HirGraph::remap_op(&node.op, &remap);
            new_nodes.push(HirNode {
                id: HirId(new_nodes.len()),
                op: remapped,
                ty: node.ty.clone(),
                span: node.span,
            });
            remap[i] = Some(HirId(new_nodes.len() - 1));
            continue;
        }

        if let Some((fused_op, binary_idx)) = try_fuse(i, nodes, &use_count) {
            absorbed.insert(binary_idx);
            let remapped = HirGraph::remap_op(&fused_op, &remap);
            new_nodes.push(HirNode {
                id: HirId(new_nodes.len()),
                op: remapped,
                ty: node.ty.clone(),
                span: node.span,
            });
            remap[i] = Some(HirId(new_nodes.len() - 1));
        } else {
            let remapped = HirGraph::remap_op(&node.op, &remap);
            new_nodes.push(HirNode {
                id: HirId(new_nodes.len()),
                op: remapped,
                ty: node.ty.clone(),
                span: node.span,
            });
            remap[i] = Some(HirId(new_nodes.len() - 1));
        }
    }

    // Rebuild inputs list
    let mut new_inputs = Vec::new();
    for &inp in &graph.inputs {
        if let Some(new_id) = remap[inp.index()] {
            new_inputs.push(new_id);
        }
    }

    // Remap output
    let new_output = output.and_then(|o| remap[o.index()]);

    HirGraph {
        nodes: new_nodes,
        inputs: new_inputs,
        output: new_output,
    }
}

/// Try to fuse node `i` (a unary activation) with its input (a binary op).
fn try_fuse(i: usize, nodes: &[HirNode], use_count: &[u32]) -> Option<(HirOp, usize)> {
    let node = &nodes[i];

    // Identify unary activations and their input.
    let (activation, input_id) = match &node.op {
        HirOp::Relu(x) => ("relu", *x),
        HirOp::Sigmoid(x) => ("sigmoid", *x),
        HirOp::Tanh(x) => ("tanh", *x),
        HirOp::Gelu(x) => ("gelu", *x),
        HirOp::Silu(x) => ("silu", *x),
        _ => return None,
    };

    let bin_idx = input_id.index();
    if bin_idx >= nodes.len() {
        return None;
    }

    // Only fuse if the binary op has exactly one consumer (this unary).
    if use_count[bin_idx] != 1 {
        return None;
    }

    // Don't fuse across reduction barriers.
    if HirGraph::is_reduction(&nodes[bin_idx].op) {
        return None;
    }

    let binary = &nodes[bin_idx];
    match (activation, &binary.op) {
        ("relu", HirOp::Add(a, b)) => Some((HirOp::FusedReluAdd(*a, *b), bin_idx)),
        ("relu", HirOp::Mul(a, b)) => Some((HirOp::FusedReluMul(*a, *b), bin_idx)),
        ("sigmoid", HirOp::Add(a, b)) => Some((HirOp::FusedSigmoidAdd(*a, *b), bin_idx)),
        ("tanh", HirOp::Add(a, b)) => Some((HirOp::FusedTanhAdd(*a, *b), bin_idx)),
        ("gelu", HirOp::Add(a, b)) => Some((HirOp::FusedGeluAdd(*a, *b), bin_idx)),
        ("silu", HirOp::Add(a, b)) => Some((HirOp::FusedSiluAdd(*a, *b), bin_idx)),
        ("silu", HirOp::Mul(a, b)) => Some((HirOp::FusedSiluMul(*a, *b), bin_idx)),
        _ => None,
    }
}
