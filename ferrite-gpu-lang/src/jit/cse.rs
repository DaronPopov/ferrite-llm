/// Common Subexpression Elimination on the HIR graph.
///
/// Hashes `(op_tag, operand_ids)` for each pure op.  If a duplicate
/// is found, remap all uses of the duplicate to the original.
/// Run DCE afterwards to clean up dead duplicates.

use std::collections::HashMap;

use super::hir::*;

/// Eliminate common subexpressions in the graph.
pub fn cse(mut graph: HirGraph) -> HirGraph {
    let n = graph.nodes.len();
    let mut seen: HashMap<OpKey, HirId> = HashMap::new();
    let mut remap: Vec<Option<HirId>> = vec![None; n];

    for i in 0..n {
        let node = &graph.nodes[i];

        // Skip non-pure ops (Input, etc.)
        if !HirGraph::is_pure(&node.op) {
            remap[i] = Some(HirId(i));
            continue;
        }

        // Build a canonical key from the (already-remapped) op
        let remapped_op = HirGraph::remap_op(&node.op, &remap);
        let key = op_key(&remapped_op);

        if let Some(existing_id) = seen.get(&key) {
            // This is a duplicate — remap to existing
            remap[i] = Some(*existing_id);
        } else {
            // First occurrence — record it
            remap[i] = Some(HirId(i));
            seen.insert(key, HirId(i));
        }
    }

    // Apply remap to all ops
    for i in 0..n {
        graph.nodes[i].op = HirGraph::remap_op(&graph.nodes[i].op, &remap);
    }

    // Remap output
    if let Some(out) = graph.output {
        graph.output = remap[out.index()].or(graph.output);
    }

    graph
}

/// A hashable key representing an op and its operands.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[allow(dead_code)]
enum OpKey {
    Unary(OpTag, usize),
    Binary(OpTag, usize, usize),
    Ternary(OpTag, usize, usize, usize),
    FillLike(u64, usize), // value bits + like id
    Other(usize), // non-CSE-able, use node index as unique key
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum OpTag {
    Relu, Tanh, Sigmoid, Gelu, Silu, Abs, Sqrt, Exp, Log, Neg,
    Add, Sub, Mul, Div,
    CmpLt, CmpGt, CmpLe, CmpGe, CmpEq, CmpNe,
    FusedReluAdd, FusedReluMul, FusedSigmoidAdd, FusedTanhAdd,
    FusedGeluAdd, FusedSiluAdd, FusedSiluMul,
}

fn op_key(op: &HirOp) -> OpKey {
    match op {
        HirOp::Relu(x) => OpKey::Unary(OpTag::Relu, x.index()),
        HirOp::Tanh(x) => OpKey::Unary(OpTag::Tanh, x.index()),
        HirOp::Sigmoid(x) => OpKey::Unary(OpTag::Sigmoid, x.index()),
        HirOp::Gelu(x) => OpKey::Unary(OpTag::Gelu, x.index()),
        HirOp::Silu(x) => OpKey::Unary(OpTag::Silu, x.index()),
        HirOp::Abs(x) => OpKey::Unary(OpTag::Abs, x.index()),
        HirOp::Sqrt(x) => OpKey::Unary(OpTag::Sqrt, x.index()),
        HirOp::Exp(x) => OpKey::Unary(OpTag::Exp, x.index()),
        HirOp::Log(x) => OpKey::Unary(OpTag::Log, x.index()),
        HirOp::Neg(x) => OpKey::Unary(OpTag::Neg, x.index()),

        HirOp::Add(a, b) => OpKey::Binary(OpTag::Add, a.index(), b.index()),
        HirOp::Sub(a, b) => OpKey::Binary(OpTag::Sub, a.index(), b.index()),
        HirOp::Mul(a, b) => OpKey::Binary(OpTag::Mul, a.index(), b.index()),
        HirOp::Div(a, b) => OpKey::Binary(OpTag::Div, a.index(), b.index()),

        HirOp::CmpLt(a, b) => OpKey::Binary(OpTag::CmpLt, a.index(), b.index()),
        HirOp::CmpGt(a, b) => OpKey::Binary(OpTag::CmpGt, a.index(), b.index()),
        HirOp::CmpLe(a, b) => OpKey::Binary(OpTag::CmpLe, a.index(), b.index()),
        HirOp::CmpGe(a, b) => OpKey::Binary(OpTag::CmpGe, a.index(), b.index()),
        HirOp::CmpEq(a, b) => OpKey::Binary(OpTag::CmpEq, a.index(), b.index()),
        HirOp::CmpNe(a, b) => OpKey::Binary(OpTag::CmpNe, a.index(), b.index()),

        HirOp::FillLike { value, like } => OpKey::FillLike(value.to_bits(), like.index()),

        HirOp::FusedReluAdd(a, b) => OpKey::Binary(OpTag::FusedReluAdd, a.index(), b.index()),
        HirOp::FusedReluMul(a, b) => OpKey::Binary(OpTag::FusedReluMul, a.index(), b.index()),
        HirOp::FusedSigmoidAdd(a, b) => OpKey::Binary(OpTag::FusedSigmoidAdd, a.index(), b.index()),
        HirOp::FusedTanhAdd(a, b) => OpKey::Binary(OpTag::FusedTanhAdd, a.index(), b.index()),
        HirOp::FusedGeluAdd(a, b) => OpKey::Binary(OpTag::FusedGeluAdd, a.index(), b.index()),
        HirOp::FusedSiluAdd(a, b) => OpKey::Binary(OpTag::FusedSiluAdd, a.index(), b.index()),
        HirOp::FusedSiluMul(a, b) => OpKey::Binary(OpTag::FusedSiluMul, a.index(), b.index()),

        // Non-CSE-able ops: CumSum, TopK, Where, Gather, IndexSelect,
        // ScatterAdd, Argsort, reductions, matmul, Input, Constant
        // These have extra parameters or are not safe to deduplicate.
        _ => {
            // Use a dummy key that won't match anything else
            static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            OpKey::Other(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
        }
    }
}
