/// Constant Folding pass on the HIR graph.
///
/// Forward pass: if all inputs to a node are constants (`Constant`),
/// evaluate the operation and replace with a new `Constant` node.

use super::hir::*;

/// Fold constant expressions in the graph.
pub fn const_fold(mut graph: HirGraph) -> HirGraph {
    // We iterate forward, checking if operands are constants.
    // For nodes whose operands are all constant, we evaluate
    // and replace the node.

    let n = graph.nodes.len();
    for i in 0..n {
        let new_op = try_fold(i, &graph.nodes);
        if let Some(value) = new_op {
            graph.nodes[i].op = HirOp::Constant { value };
            graph.nodes[i].ty = HirType::Scalar(ElemType::F32);
        }
    }

    graph
}

fn get_const(nodes: &[HirNode], id: HirId) -> Option<f64> {
    match &nodes[id.index()].op {
        HirOp::Constant { value } => Some(*value),
        HirOp::FillLike { value, .. } => Some(*value),
        _ => None,
    }
}

fn try_fold(idx: usize, nodes: &[HirNode]) -> Option<f64> {
    let op = &nodes[idx].op;
    match op {
        // Unary on constant
        HirOp::Relu(x) => {
            let v = get_const(nodes, *x)?;
            Some(v.max(0.0))
        }
        HirOp::Neg(x) => {
            let v = get_const(nodes, *x)?;
            Some(-v)
        }
        HirOp::Abs(x) => {
            let v = get_const(nodes, *x)?;
            Some(v.abs())
        }
        HirOp::Exp(x) => {
            let v = get_const(nodes, *x)?;
            Some(v.exp())
        }
        HirOp::Log(x) => {
            let v = get_const(nodes, *x)?;
            Some(v.ln())
        }
        HirOp::Sqrt(x) => {
            let v = get_const(nodes, *x)?;
            Some(v.sqrt())
        }
        HirOp::Sigmoid(x) => {
            let v = get_const(nodes, *x)?;
            Some(1.0 / (1.0 + (-v).exp()))
        }
        HirOp::Tanh(x) => {
            let v = get_const(nodes, *x)?;
            Some(v.tanh())
        }

        // Binary on two constants
        HirOp::Add(a, b) => {
            let va = get_const(nodes, *a)?;
            let vb = get_const(nodes, *b)?;
            Some(va + vb)
        }
        HirOp::Sub(a, b) => {
            let va = get_const(nodes, *a)?;
            let vb = get_const(nodes, *b)?;
            Some(va - vb)
        }
        HirOp::Mul(a, b) => {
            let va = get_const(nodes, *a)?;
            let vb = get_const(nodes, *b)?;
            Some(va * vb)
        }
        HirOp::Div(a, b) => {
            let va = get_const(nodes, *a)?;
            let vb = get_const(nodes, *b)?;
            if vb == 0.0 { None } else { Some(va / vb) }
        }

        _ => None,
    }
}
