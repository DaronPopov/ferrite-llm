/// Abstract syntax tree for ferrite JIT scripts.
///
/// Scripts are flat sequences of statements that map 1:1 to the
/// `Program` builder ops.  Functions (`fn ... end`) are graph
/// templates that are inlined during lowering — no runtime call
/// overhead.

use super::lexer::Span;

// ── Spanned wrapper ────────────────────────────────────────────

/// A value paired with its source span.
#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

// ── Operators ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOper {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Or,
}

// ── Expressions ────────────────────────────────────────────────

/// Convenience alias for a spanned expression.
pub type SExpr = Spanned<Expr>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeDim {
    Const(usize),
    Symbol(String),
}

#[derive(Debug, Clone)]
pub enum Expr {
    /// Variable reference: `x`
    Var(String),
    /// Integer literal: `42`
    Int(i64),
    /// Float literal: `2.0`, `0.5`, `3.14`
    Float(f64),
    /// Boolean literal: `true` / `false`
    Bool(bool),
    /// Shape literal: `[1, 1, 1, 4096]` or symbolic `[B, T, H]`
    Shape(Vec<ShapeDim>),
    /// Function / builtin call: `relu(x)` or `topk(x, k=10, dim=3)`
    Call { func: String, args: Vec<Arg> },
    /// Binary operation: `x + y`, `a * b`
    BinOp { op: BinOper, left: Box<SExpr>, right: Box<SExpr> },
    /// Unary negation: `-x`
    Neg(Box<SExpr>),
    /// Comparison: `a < b`, `x == y`
    Cmp { op: CmpOp, left: Box<SExpr>, right: Box<SExpr> },
    /// Logical: `a and b`, `x or y`
    Logic { op: LogicOp, left: Box<SExpr>, right: Box<SExpr> },
    /// Logical negation: `not x`
    LogicNot(Box<SExpr>),
}

#[derive(Debug, Clone)]
pub enum Arg {
    /// Positional argument: `x` or `[1, 2, 3]`
    Positional(SExpr),
    /// Named argument: `dim=3`
    Named { name: String, value: SExpr },
}

// ── Statements ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilePrecision {
    F32,
    F16,
    Bf16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileQuant {
    None,
    Int8,
    Nf4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileDistribution {
    None,
    Replicate,
    Shard,
    ReduceScatter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileLayout {
    RowMajor,
    ColMajor,
    Blocked32x8,
    Blocked64x4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileAccum {
    F32,
    Bf16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileCollective {
    None,
    AllReduce,
    ReduceScatter,
    AllGather,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TileAnnotations {
    pub tile_m: Option<usize>,
    pub tile_n: Option<usize>,
    pub tile_k: Option<usize>,
    pub unroll: Option<usize>,
    pub pipeline_stages: Option<usize>,
    pub precision: Option<TilePrecision>,
    pub quant: Option<TileQuant>,
    pub distribution: Option<TileDistribution>,
    pub replicas: Option<usize>,
    pub mesh_axis: Option<usize>,
    pub layout: Option<TileLayout>,
    pub accum: Option<TileAccum>,
    pub collective: Option<TileCollective>,
}

impl TileAnnotations {
    pub fn inherit_from(&self, parent: &TileAnnotations) -> TileAnnotations {
        TileAnnotations {
            tile_m: self.tile_m.or(parent.tile_m),
            tile_n: self.tile_n.or(parent.tile_n),
            tile_k: self.tile_k.or(parent.tile_k),
            unroll: self.unroll.or(parent.unroll),
            pipeline_stages: self.pipeline_stages.or(parent.pipeline_stages),
            precision: self.precision.or(parent.precision),
            quant: self.quant.or(parent.quant),
            distribution: self.distribution.or(parent.distribution),
            replicas: self.replicas.or(parent.replicas),
            mesh_axis: self.mesh_axis.or(parent.mesh_axis),
            layout: self.layout.or(parent.layout),
            accum: self.accum.or(parent.accum),
            collective: self.collective.or(parent.collective),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Stmt {
    /// Variable binding: `x = relu(y)`
    Let { name: String, value: SExpr, where_clauses: Vec<SExpr>, span: Span },
    /// Return statement: `return x`
    Return { name: String, span: Span },
    /// Function definition: `fn name(params): body end`
    FnDef {
        name: String,
        params: Vec<String>,
        body: Vec<Stmt>,
        span: Span,
    },
    /// Tile block: `tile output over (inputs): body end`
    Tile {
        output: String,
        inputs: Vec<String>,
        annotations: TileAnnotations,
        body: Vec<Stmt>,
        span: Span,
    },
    /// Conditional: `if cond then body (elif cond then body)* (else body)? end`
    If {
        branches: Vec<(SExpr, Vec<Stmt>)>,
        else_body: Option<Vec<Stmt>>,
        span: Span,
    },
    /// Bounded loop: `for var in start..end: body end`
    For {
        var: String,
        start: i64,
        end: i64,
        body: Vec<Stmt>,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub struct Script {
    pub stmts: Vec<Stmt>,
}
