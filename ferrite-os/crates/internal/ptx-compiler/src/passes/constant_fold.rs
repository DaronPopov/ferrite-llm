//! Constant folding pass.
//!
//! Evaluates operations on constant tensors at compile time. When every input
//! to a node carries a known scalar value the result is computed on the CPU and
//! the node is replaced with a `Constant`. The pass iterates to a fixed point
//! so that chains of constant expressions are fully collapsed.

use crate::ir::{Graph, Node, NodeId, OpCode};
use ptx_runtime::Result;

/// Fold constant expressions in the graph.
///
/// Iterates until no more nodes can be folded.
pub fn fold_constants(mut graph: Graph) -> Result<Graph> {
    loop {
        let foldable: Vec<(NodeId, f32)> = graph
            .nodes()
            .iter()
            .filter_map(|(&node_id, node)| {
                if !can_fold(node, &graph) {
                    return None;
                }
                evaluate_constant(node, &graph).map(|val| (node_id, val))
            })
            .collect();

        if foldable.is_empty() {
            break;
        }

        for (node_id, value) in foldable {
            graph.replace_with_constant(node_id, value);
        }
    }

    Ok(graph)
}

/// Check if a node can be constant folded.
///
/// A node is foldable when it is not an `Input`/`Constant` itself and every
/// input tensor is a constant with a known scalar value.
fn can_fold(node: &Node, graph: &Graph) -> bool {
    if node.is_input() || node.is_constant() {
        return false;
    }

    node.inputs.iter().all(|input_id| {
        graph
            .tensor(*input_id)
            .map(|meta| meta.is_constant && meta.constant_value.is_some())
            .unwrap_or(false)
    })
}

/// Evaluate a node with all-constant inputs and return the scalar result.
///
/// Returns `None` for operations we choose not to fold (e.g. domain errors,
/// unsupported ops).
fn evaluate_constant(node: &Node, graph: &Graph) -> Option<f32> {
    let vals: Vec<f32> = node
        .inputs
        .iter()
        .filter_map(|id| graph.tensor(*id)?.constant_value)
        .collect();

    if vals.len() != node.inputs.len() {
        return None;
    }

    match node.op {
        // ----- Binary ops -----
        OpCode::Add => Some(vals[0] + vals[1]),
        OpCode::Sub => Some(vals[0] - vals[1]),
        OpCode::Mul => Some(vals[0] * vals[1]),
        OpCode::Div => {
            if vals[1] == 0.0 {
                None
            } else {
                Some(vals[0] / vals[1])
            }
        }
        OpCode::Max => Some(vals[0].max(vals[1])),
        OpCode::Min => Some(vals[0].min(vals[1])),

        // ----- Unary ops -----
        OpCode::Neg => Some(-vals[0]),
        OpCode::Abs => Some(vals[0].abs()),
        OpCode::Exp => Some(vals[0].exp()),
        OpCode::Log => {
            if vals[0] <= 0.0 {
                None
            } else {
                Some(vals[0].ln())
            }
        }
        OpCode::Sqrt => {
            if vals[0] < 0.0 {
                None
            } else {
                Some(vals[0].sqrt())
            }
        }
        OpCode::Rsqrt => {
            if vals[0] <= 0.0 {
                None
            } else {
                Some(1.0 / vals[0].sqrt())
            }
        }
        OpCode::Ceil => Some(vals[0].ceil()),
        OpCode::Floor => Some(vals[0].floor()),
        OpCode::Round => Some(vals[0].round()),
        OpCode::Sqr => Some(vals[0] * vals[0]),
        OpCode::Recip => {
            if vals[0] == 0.0 {
                None
            } else {
                Some(1.0 / vals[0])
            }
        }
        OpCode::Sin => Some(vals[0].sin()),
        OpCode::Cos => Some(vals[0].cos()),
        OpCode::Tanh => Some(vals[0].tanh()),

        // ----- Activations -----
        OpCode::Relu => Some(vals[0].max(0.0)),
        OpCode::Relu6 => Some(vals[0].max(0.0).min(6.0)),
        OpCode::Sigmoid => Some(1.0 / (1.0 + (-vals[0]).exp())),

        // ----- Reductions on uniform tensors -----
        // A uniform tensor reduced along a dimension keeps the value for
        // mean/max/min and multiplies by dim-size for sum.
        OpCode::ReduceSum => {
            let input_meta = graph.tensor(node.inputs[0])?;
            let dim = node.attrs.reduce_dim? as usize;
            if dim >= input_meta.shape.len() {
                return None;
            }
            Some(vals[0] * input_meta.shape[dim] as f32)
        }
        OpCode::ReduceMean | OpCode::ReduceMax | OpCode::ReduceMin => Some(vals[0]),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ptx_tensor::DType;

    #[test]
    fn test_constant_fold_binary_add() {
        let mut graph = Graph::new();
        let a = graph.constant(&[2, 3], DType::F32, Some(1.0));
        let b = graph.constant(&[2, 3], DType::F32, Some(2.0));
        let c = graph.add(a, b);
        graph.mark_output(c);

        let optimized = fold_constants(graph).unwrap();

        // The add node should now be a constant with value 3.0
        let out_id = optimized.outputs()[0];
        let out_meta = optimized.tensor(out_id).unwrap();
        assert!(out_meta.is_constant);
        assert_eq!(out_meta.constant_value, Some(3.0));
    }

    #[test]
    fn test_constant_fold_chain() {
        // c1=1.0, c2=2.0 => add=3.0 => neg=-3.0
        let mut graph = Graph::new();
        let a = graph.constant(&[4], DType::F32, Some(1.0));
        let b = graph.constant(&[4], DType::F32, Some(2.0));
        let c = graph.add(a, b);
        let d = graph.neg(c);
        graph.mark_output(d);

        let optimized = fold_constants(graph).unwrap();

        let out_meta = optimized.tensor(optimized.outputs()[0]).unwrap();
        assert!(out_meta.is_constant);
        assert_eq!(out_meta.constant_value, Some(-3.0));
    }

    #[test]
    fn test_constant_fold_mixed_graph() {
        // input + constant should NOT be folded
        let mut graph = Graph::new();
        let a = graph.input(&[2, 3], DType::F32);
        let b = graph.constant(&[2, 3], DType::F32, Some(2.0));
        let c = graph.add(a, b);
        graph.mark_output(c);

        let optimized = fold_constants(graph).unwrap();

        // The add should remain (not folded)
        let out_meta = optimized.tensor(optimized.outputs()[0]).unwrap();
        assert!(!out_meta.is_constant);
    }

    #[test]
    fn test_constant_fold_relu() {
        let mut graph = Graph::new();
        let a = graph.constant(&[4], DType::F32, Some(-5.0));
        let b = graph.relu(a);
        graph.mark_output(b);

        let optimized = fold_constants(graph).unwrap();

        let out_meta = optimized.tensor(optimized.outputs()[0]).unwrap();
        assert!(out_meta.is_constant);
        assert_eq!(out_meta.constant_value, Some(0.0));
    }

    #[test]
    fn test_constant_fold_idempotent() {
        let mut graph = Graph::new();
        let a = graph.constant(&[2], DType::F32, Some(3.0));
        let b = graph.constant(&[2], DType::F32, Some(4.0));
        let c = graph.mul(a, b);
        graph.mark_output(c);

        let pass1 = fold_constants(graph).unwrap();
        let n1 = pass1.num_nodes();
        let pass2 = fold_constants(pass1).unwrap();
        let n2 = pass2.num_nodes();

        assert_eq!(n1, n2, "second pass should be a no-op");
    }

    #[test]
    fn test_constant_fold_reduce_sum() {
        // constant(shape=[2,3], val=2.0) -> reduce_sum(dim=1) => 2.0 * 3 = 6.0
        let mut graph = Graph::new();
        let a = graph.constant(&[2, 3], DType::F32, Some(2.0));
        let b = graph.reduce_sum(a, 1, false);
        graph.mark_output(b);

        let optimized = fold_constants(graph).unwrap();

        let out_meta = optimized.tensor(optimized.outputs()[0]).unwrap();
        assert!(out_meta.is_constant);
        assert_eq!(out_meta.constant_value, Some(6.0));
    }
}
