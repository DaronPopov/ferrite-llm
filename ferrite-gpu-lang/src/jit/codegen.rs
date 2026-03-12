/// Code generation: translates an optimized `HirGraph` into the
/// existing `Program`/`Op` representation for compilation and execution.
///
/// This is a straightforward traversal mapping `HirOp` variants to `Op`
/// variants with `HirId` -> `ValueId` remapping.

use super::hir::*;
use super::JitError;
use crate::{Program, ValueId};

/// Translate an optimized HirGraph into a Program.
pub fn codegen(graph: &HirGraph) -> Result<Program, JitError> {
    let mut program = Program::new();

    // Map HirId -> ValueId
    let mut id_map: Vec<Option<ValueId>> = vec![None; graph.nodes.len()];

    for node in &graph.nodes {
        let r = |hid: HirId| -> Result<ValueId, JitError> {
            id_map[hid.index()].ok_or_else(|| {
                JitError::Lower(format!("codegen: unresolved HirId({})", hid.index()))
            })
        };

        let vid = match &node.op {
            HirOp::Input { shape } => {
                program.input(shape).map_err(JitError::from)?
            }
            HirOp::Constant { value: _ } => {
                // Constants without a tensor reference can't be directly represented
                // as Op nodes.  Skip them — they should have been folded into
                // FillLike by the lowering pass.  If a constant reaches codegen,
                // it means it was unused or only consumed by other constants.
                // We emit a dummy FillLike(value, self) which will be cleaned up
                // by DCE if truly unused. For now, skip and mark as None.
                id_map[node.id.index()] = None;
                continue;
            }

            // ── unary ops ──
            HirOp::Relu(x) => program.relu(r(*x)?),
            HirOp::Tanh(x) => program.tanh(r(*x)?),
            HirOp::Sigmoid(x) => program.sigmoid(r(*x)?),
            HirOp::Gelu(x) => program.gelu(r(*x)?),
            HirOp::Silu(x) => program.silu(r(*x)?),
            HirOp::Abs(x) => program.abs(r(*x)?),
            HirOp::Sqrt(x) => program.sqrt(r(*x)?),
            HirOp::Exp(x) => program.exp(r(*x)?),
            HirOp::Log(x) => program.log(r(*x)?),
            HirOp::Neg(x) => {
                let xv = r(*x)?;
                let zero = program.fill_like(0.0, xv);
                program.sub(zero, xv)
            }

            // ── binary ops ──
            HirOp::Add(a, b) => program.add(r(*a)?, r(*b)?),
            HirOp::Sub(a, b) => program.sub(r(*a)?, r(*b)?),
            HirOp::Mul(a, b) => program.mul(r(*a)?, r(*b)?),
            HirOp::Div(a, b) => program.div(r(*a)?, r(*b)?),

            // ── fill ──
            HirOp::FillLike { value, like } => program.fill_like(*value, r(*like)?),

            // ── scan / topk ──
            HirOp::CumSum { input, dim } => program.cumsum(r(*input)?, *dim),
            HirOp::TopK { input, k, dim, largest } => program.topk(r(*input)?, *k, *dim, *largest),

            // ── ternary ──
            HirOp::Where { cond, true_val, false_val } => {
                program.where_cond(r(*cond)?, r(*true_val)?, r(*false_val)?)
            }

            // ── indexing ──
            HirOp::Gather { input, indices, dim } => {
                program.gather(r(*input)?, r(*indices)?, *dim)
            }
            HirOp::IndexSelect { input, indices, dim } => {
                program.index_select(r(*input)?, r(*indices)?, *dim)
            }
            HirOp::ScatterAdd { input, indices, src, dim } => {
                program.scatter_add(r(*input)?, r(*indices)?, r(*src)?, *dim)
            }
            HirOp::Argsort { input, dim, ascending } => {
                program.argsort(r(*input)?, *dim, *ascending)
            }

            // ── comparison ops ──
            HirOp::CmpLt(a, b) => program.cmp_lt(r(*a)?, r(*b)?),
            HirOp::CmpGt(a, b) => program.cmp_gt(r(*a)?, r(*b)?),
            HirOp::CmpLe(a, b) => program.cmp_le(r(*a)?, r(*b)?),
            HirOp::CmpGe(a, b) => program.cmp_ge(r(*a)?, r(*b)?),
            HirOp::CmpEq(a, b) => program.cmp_eq(r(*a)?, r(*b)?),
            HirOp::CmpNe(a, b) => program.cmp_ne(r(*a)?, r(*b)?),

            // ── reductions ──
            HirOp::ReduceSum { input, dim } => program.reduce_sum(r(*input)?, *dim),
            HirOp::ReduceMean { input, dim } => program.reduce_mean(r(*input)?, *dim),
            HirOp::ReduceMax { input, dim } => program.reduce_max(r(*input)?, *dim),
            HirOp::ReduceMin { input, dim } => program.reduce_min(r(*input)?, *dim),
            HirOp::Argmax { input, dim } => program.argmax(r(*input)?, *dim),
            HirOp::Argmin { input, dim } => program.argmin(r(*input)?, *dim),
            HirOp::Softmax { input, dim } => program.softmax(r(*input)?, *dim),

            // ── matmul ──
            HirOp::Matmul { lhs, rhs } => program.matmul(r(*lhs)?, r(*rhs)?),

            // ── fused ops ──
            HirOp::FusedReluAdd(a, b) => program.fused_relu_add(r(*a)?, r(*b)?),
            HirOp::FusedReluMul(a, b) => program.fused_relu_mul(r(*a)?, r(*b)?),
            HirOp::FusedSigmoidAdd(a, b) => program.fused_sigmoid_add(r(*a)?, r(*b)?),
            HirOp::FusedTanhAdd(a, b) => program.fused_tanh_add(r(*a)?, r(*b)?),
            HirOp::FusedGeluAdd(a, b) => program.fused_gelu_add(r(*a)?, r(*b)?),
            HirOp::FusedSiluAdd(a, b) => program.fused_silu_add(r(*a)?, r(*b)?),
            HirOp::FusedSiluMul(a, b) => program.fused_silu_mul(r(*a)?, r(*b)?),
        };

        id_map[node.id.index()] = Some(vid);
    }

    // Set output
    if let Some(out_hid) = graph.output {
        if let Some(out_vid) = id_map[out_hid.index()] {
            program.set_output(out_vid);
        } else {
            return Err(JitError::Lower(
                "codegen: output node was not generated (possibly a bare constant)".into(),
            ));
        }
    }

    Ok(program)
}
