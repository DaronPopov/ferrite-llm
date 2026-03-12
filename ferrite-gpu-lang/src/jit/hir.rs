/// Typed High-Level Intermediate Representation for ferrite JIT.
///
/// Sits between the AST and the final `Program`/`Op` graph.  Every node
/// carries a computed type and an optional source span, enabling
/// optimization passes and better error messages.

use super::lexer::Span;

// ── IDs ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HirId(pub usize);

impl HirId {
    pub fn index(self) -> usize {
        self.0
    }
}

// ── Types ──────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ElemType {
    F32,
    I32,
    U8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HirType {
    Tensor { shape: Vec<usize>, elem: ElemType },
    Scalar(ElemType),
}

impl HirType {
    /// Shorthand for an f32 tensor type.
    pub fn f32_tensor(shape: Vec<usize>) -> Self {
        HirType::Tensor { shape, elem: ElemType::F32 }
    }

    pub fn shape(&self) -> Option<&[usize]> {
        match self {
            HirType::Tensor { shape, .. } => Some(shape),
            HirType::Scalar(_) => None,
        }
    }

    pub fn shape_vec(&self) -> Vec<usize> {
        match self {
            HirType::Tensor { shape, .. } => shape.clone(),
            HirType::Scalar(_) => vec![1],
        }
    }
}

// ── Operations ─────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum HirOp {
    // ── sources ──
    Input { shape: Vec<usize> },
    Constant { value: f64 },

    // ── unary activations ──
    Relu(HirId),
    Tanh(HirId),
    Sigmoid(HirId),
    Gelu(HirId),
    Silu(HirId),
    Abs(HirId),
    Sqrt(HirId),
    Exp(HirId),
    Log(HirId),
    Neg(HirId),

    // ── binary elementwise ──
    Add(HirId, HirId),
    Sub(HirId, HirId),
    Mul(HirId, HirId),
    Div(HirId, HirId),

    // ── broadcast / fill ──
    FillLike { value: f64, like: HirId },

    // ── scan / topk ──
    CumSum { input: HirId, dim: usize },
    TopK { input: HirId, k: usize, dim: usize, largest: bool },

    // ── ternary ──
    Where { cond: HirId, true_val: HirId, false_val: HirId },

    // ── indexing ──
    Gather { input: HirId, indices: HirId, dim: usize },
    IndexSelect { input: HirId, indices: HirId, dim: usize },
    ScatterAdd { input: HirId, indices: HirId, src: HirId, dim: usize },
    Argsort { input: HirId, dim: usize, ascending: bool },

    // ── comparison (produce f32 0.0/1.0 masks) ──
    CmpLt(HirId, HirId),
    CmpGt(HirId, HirId),
    CmpLe(HirId, HirId),
    CmpGe(HirId, HirId),
    CmpEq(HirId, HirId),
    CmpNe(HirId, HirId),

    // ── reductions ──
    ReduceSum { input: HirId, dim: usize },
    ReduceMean { input: HirId, dim: usize },
    ReduceMax { input: HirId, dim: usize },
    ReduceMin { input: HirId, dim: usize },
    Argmax { input: HirId, dim: usize },
    Argmin { input: HirId, dim: usize },
    Softmax { input: HirId, dim: usize },

    // ── matmul ──
    Matmul { lhs: HirId, rhs: HirId },

    // ── fused ops (produced by fusion pass, never by lowering) ──
    FusedReluAdd(HirId, HirId),
    FusedReluMul(HirId, HirId),
    FusedSigmoidAdd(HirId, HirId),
    FusedTanhAdd(HirId, HirId),
    FusedGeluAdd(HirId, HirId),
    FusedSiluAdd(HirId, HirId),
    FusedSiluMul(HirId, HirId),
}

// ── Graph ──────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct HirNode {
    pub id: HirId,
    pub op: HirOp,
    pub ty: HirType,
    pub span: Option<Span>,
}

#[derive(Clone, Debug)]
pub struct HirGraph {
    pub nodes: Vec<HirNode>,
    pub inputs: Vec<HirId>,
    pub output: Option<HirId>,
}

impl HirGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            output: None,
        }
    }

    /// Add a node to the graph, returning its ID.
    pub fn push(&mut self, op: HirOp, ty: HirType, span: Option<Span>) -> HirId {
        let id = HirId(self.nodes.len());
        self.nodes.push(HirNode { id, op, ty, span });
        id
    }

    /// Get a node by ID.
    pub fn node(&self, id: HirId) -> &HirNode {
        &self.nodes[id.index()]
    }

    /// Get the type of a node.
    pub fn type_of(&self, id: HirId) -> &HirType {
        &self.nodes[id.index()].ty
    }

    /// Get the shape of a node's output.
    pub fn shape_of(&self, id: HirId) -> Vec<usize> {
        self.type_of(id).shape_vec()
    }

    /// Get all direct dependencies of an op.
    pub fn deps(op: &HirOp) -> Vec<HirId> {
        match op {
            HirOp::Input { .. } | HirOp::Constant { .. } => vec![],
            HirOp::Relu(x) | HirOp::Tanh(x) | HirOp::Sigmoid(x)
            | HirOp::Gelu(x) | HirOp::Silu(x) | HirOp::Abs(x)
            | HirOp::Sqrt(x) | HirOp::Exp(x) | HirOp::Log(x) | HirOp::Neg(x) => vec![*x],
            HirOp::Add(a, b) | HirOp::Sub(a, b) | HirOp::Mul(a, b) | HirOp::Div(a, b) => vec![*a, *b],
            HirOp::FillLike { like, .. } => vec![*like],
            HirOp::CumSum { input, .. } | HirOp::TopK { input, .. } => vec![*input],
            HirOp::Where { cond, true_val, false_val } => vec![*cond, *true_val, *false_val],
            HirOp::Gather { input, indices, .. } | HirOp::IndexSelect { input, indices, .. } => vec![*input, *indices],
            HirOp::ScatterAdd { input, indices, src, .. } => vec![*input, *indices, *src],
            HirOp::Argsort { input, .. } => vec![*input],
            HirOp::CmpLt(a, b) | HirOp::CmpGt(a, b)
            | HirOp::CmpLe(a, b) | HirOp::CmpGe(a, b)
            | HirOp::CmpEq(a, b) | HirOp::CmpNe(a, b) => vec![*a, *b],
            HirOp::ReduceSum { input, .. } | HirOp::ReduceMean { input, .. }
            | HirOp::ReduceMax { input, .. } | HirOp::ReduceMin { input, .. }
            | HirOp::Argmax { input, .. } | HirOp::Argmin { input, .. }
            | HirOp::Softmax { input, .. } => vec![*input],
            HirOp::Matmul { lhs, rhs } => vec![*lhs, *rhs],
            HirOp::FusedReluAdd(a, b) | HirOp::FusedReluMul(a, b)
            | HirOp::FusedSigmoidAdd(a, b) | HirOp::FusedTanhAdd(a, b)
            | HirOp::FusedGeluAdd(a, b) | HirOp::FusedSiluAdd(a, b)
            | HirOp::FusedSiluMul(a, b) => vec![*a, *b],
        }
    }

    /// Compute use counts for all nodes.
    pub fn use_counts(&self) -> Vec<u32> {
        let mut counts = vec![0u32; self.nodes.len()];
        for node in &self.nodes {
            for dep in Self::deps(&node.op) {
                if dep.index() < counts.len() {
                    counts[dep.index()] += 1;
                }
            }
        }
        if let Some(out) = self.output {
            if out.index() < counts.len() {
                counts[out.index()] += 1;
            }
        }
        counts
    }

    /// Returns true if the op is pure (side-effect free, deterministic).
    pub fn is_pure(op: &HirOp) -> bool {
        !matches!(op, HirOp::Input { .. })
    }

    /// Returns true if this op is a reduction (fusion barrier).
    pub fn is_reduction(op: &HirOp) -> bool {
        matches!(op,
            HirOp::ReduceSum { .. } | HirOp::ReduceMean { .. }
            | HirOp::ReduceMax { .. } | HirOp::ReduceMin { .. }
            | HirOp::Argmax { .. } | HirOp::Argmin { .. }
            | HirOp::Softmax { .. } | HirOp::Matmul { .. }
        )
    }

    /// Remap all HirId references in an op.
    pub fn remap_op(op: &HirOp, remap: &[Option<HirId>]) -> HirOp {
        let r = |id: HirId| -> HirId {
            remap.get(id.index()).and_then(|m| *m).unwrap_or(id)
        };
        match op {
            HirOp::Input { shape } => HirOp::Input { shape: shape.clone() },
            HirOp::Constant { value } => HirOp::Constant { value: *value },
            HirOp::Relu(x) => HirOp::Relu(r(*x)),
            HirOp::Tanh(x) => HirOp::Tanh(r(*x)),
            HirOp::Sigmoid(x) => HirOp::Sigmoid(r(*x)),
            HirOp::Gelu(x) => HirOp::Gelu(r(*x)),
            HirOp::Silu(x) => HirOp::Silu(r(*x)),
            HirOp::Abs(x) => HirOp::Abs(r(*x)),
            HirOp::Sqrt(x) => HirOp::Sqrt(r(*x)),
            HirOp::Exp(x) => HirOp::Exp(r(*x)),
            HirOp::Log(x) => HirOp::Log(r(*x)),
            HirOp::Neg(x) => HirOp::Neg(r(*x)),
            HirOp::Add(a, b) => HirOp::Add(r(*a), r(*b)),
            HirOp::Sub(a, b) => HirOp::Sub(r(*a), r(*b)),
            HirOp::Mul(a, b) => HirOp::Mul(r(*a), r(*b)),
            HirOp::Div(a, b) => HirOp::Div(r(*a), r(*b)),
            HirOp::FillLike { value, like } => HirOp::FillLike { value: *value, like: r(*like) },
            HirOp::CumSum { input, dim } => HirOp::CumSum { input: r(*input), dim: *dim },
            HirOp::TopK { input, k, dim, largest } => HirOp::TopK { input: r(*input), k: *k, dim: *dim, largest: *largest },
            HirOp::Where { cond, true_val, false_val } => HirOp::Where { cond: r(*cond), true_val: r(*true_val), false_val: r(*false_val) },
            HirOp::Gather { input, indices, dim } => HirOp::Gather { input: r(*input), indices: r(*indices), dim: *dim },
            HirOp::IndexSelect { input, indices, dim } => HirOp::IndexSelect { input: r(*input), indices: r(*indices), dim: *dim },
            HirOp::ScatterAdd { input, indices, src, dim } => HirOp::ScatterAdd { input: r(*input), indices: r(*indices), src: r(*src), dim: *dim },
            HirOp::Argsort { input, dim, ascending } => HirOp::Argsort { input: r(*input), dim: *dim, ascending: *ascending },
            HirOp::CmpLt(a, b) => HirOp::CmpLt(r(*a), r(*b)),
            HirOp::CmpGt(a, b) => HirOp::CmpGt(r(*a), r(*b)),
            HirOp::CmpLe(a, b) => HirOp::CmpLe(r(*a), r(*b)),
            HirOp::CmpGe(a, b) => HirOp::CmpGe(r(*a), r(*b)),
            HirOp::CmpEq(a, b) => HirOp::CmpEq(r(*a), r(*b)),
            HirOp::CmpNe(a, b) => HirOp::CmpNe(r(*a), r(*b)),
            HirOp::ReduceSum { input, dim } => HirOp::ReduceSum { input: r(*input), dim: *dim },
            HirOp::ReduceMean { input, dim } => HirOp::ReduceMean { input: r(*input), dim: *dim },
            HirOp::ReduceMax { input, dim } => HirOp::ReduceMax { input: r(*input), dim: *dim },
            HirOp::ReduceMin { input, dim } => HirOp::ReduceMin { input: r(*input), dim: *dim },
            HirOp::Argmax { input, dim } => HirOp::Argmax { input: r(*input), dim: *dim },
            HirOp::Argmin { input, dim } => HirOp::Argmin { input: r(*input), dim: *dim },
            HirOp::Softmax { input, dim } => HirOp::Softmax { input: r(*input), dim: *dim },
            HirOp::Matmul { lhs, rhs } => HirOp::Matmul { lhs: r(*lhs), rhs: r(*rhs) },
            HirOp::FusedReluAdd(a, b) => HirOp::FusedReluAdd(r(*a), r(*b)),
            HirOp::FusedReluMul(a, b) => HirOp::FusedReluMul(r(*a), r(*b)),
            HirOp::FusedSigmoidAdd(a, b) => HirOp::FusedSigmoidAdd(r(*a), r(*b)),
            HirOp::FusedTanhAdd(a, b) => HirOp::FusedTanhAdd(r(*a), r(*b)),
            HirOp::FusedGeluAdd(a, b) => HirOp::FusedGeluAdd(r(*a), r(*b)),
            HirOp::FusedSiluAdd(a, b) => HirOp::FusedSiluAdd(r(*a), r(*b)),
            HirOp::FusedSiluMul(a, b) => HirOp::FusedSiluMul(r(*a), r(*b)),
        }
    }
}

impl Default for HirGraph {
    fn default() -> Self {
        Self::new()
    }
}
