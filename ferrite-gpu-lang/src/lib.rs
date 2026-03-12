use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use ptx_kernels::{GuardedBuffer, KernelContext, GuardError};
use ptx_kernels::safe_api::{unary, binary, scan, topk, ternary, gather, indexing, sort};
use ptx_runtime::{GpuPtr, PtxRuntime};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, LangError>;

#[derive(Debug, Error)]
pub enum LangError {
    #[error("invalid shape {0:?}: must be non-empty and have no zero dimensions")]
    InvalidShape(Vec<usize>),
    #[error("invalid node reference: {0}")]
    InvalidNodeRef(usize),
    #[error("program has no output")]
    MissingOutput,
    #[error("shape mismatch for op {op}: left={left:?}, right={right:?}")]
    ShapeMismatch {
        op: &'static str,
        left: Vec<usize>,
        right: Vec<usize>,
    },
    #[error("expected {expected} input tensors, got {got}")]
    InputCountMismatch { expected: usize, got: usize },
    #[error("input shape mismatch at slot {index}: expected {expected:?}, got {got:?}")]
    InputShapeMismatch {
        index: usize,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("input data len mismatch at slot {index}: expected {expected}, got {got}")]
    InputLenMismatch {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("transfer error: {message}")]
    Transfer { message: String },
    #[cfg(feature = "capture")]
    #[error("capture error: {message}")]
    Capture { message: String },
    #[cfg(feature = "capture")]
    #[error("camera not opened: {reason}")]
    CameraNotOpened { reason: String },
    #[error(transparent)]
    Runtime(#[from] ptx_runtime::Error),
    #[error(transparent)]
    Guard(#[from] GuardError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ValueId(pub(crate) usize);

impl ValueId {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Op {
    Input { shape: Vec<usize> },
    // ── unary activations ──
    Relu(ValueId),
    Tanh(ValueId),
    Sigmoid(ValueId),
    Gelu(ValueId),
    Silu(ValueId),
    Abs(ValueId),
    Sqrt(ValueId),
    Exp(ValueId),
    Log(ValueId),
    // ── binary elementwise ──
    Add { lhs: ValueId, rhs: ValueId },
    Mul { lhs: ValueId, rhs: ValueId },
    Sub { lhs: ValueId, rhs: ValueId },
    Div { lhs: ValueId, rhs: ValueId },
    // ── fill / broadcast ──
    FillLike { value: f64, like: ValueId },
    // ── scan / topk ──
    CumSum { input: ValueId, dim: usize },
    TopK { input: ValueId, k: usize, dim: usize, largest: bool },
    // ── ternary ──
    Where { cond: ValueId, true_val: ValueId, false_val: ValueId },
    // ── indexing ──
    Gather { input: ValueId, indices: ValueId, dim: usize },
    IndexSelect { input: ValueId, indices: ValueId, dim: usize },
    ScatterAdd { input: ValueId, indices: ValueId, src: ValueId, dim: usize },
    Argsort { input: ValueId, dim: usize, ascending: bool },
    // ── comparison (produce f32 0.0/1.0 masks) ──
    CmpLt { lhs: ValueId, rhs: ValueId },
    CmpGt { lhs: ValueId, rhs: ValueId },
    CmpLe { lhs: ValueId, rhs: ValueId },
    CmpGe { lhs: ValueId, rhs: ValueId },
    CmpEq { lhs: ValueId, rhs: ValueId },
    CmpNe { lhs: ValueId, rhs: ValueId },
    // ── reductions ──
    ReduceSum { input: ValueId, dim: usize },
    ReduceMean { input: ValueId, dim: usize },
    ReduceMax { input: ValueId, dim: usize },
    ReduceMin { input: ValueId, dim: usize },
    Argmax { input: ValueId, dim: usize },
    Argmin { input: ValueId, dim: usize },
    Softmax { input: ValueId, dim: usize },
    // ── matmul ──
    Matmul { lhs: ValueId, rhs: ValueId },
    // ── fused ops (graph optimizer output) ─────────────────
    FusedReluAdd { lhs: ValueId, rhs: ValueId },
    FusedReluMul { lhs: ValueId, rhs: ValueId },
    FusedSigmoidAdd { lhs: ValueId, rhs: ValueId },
    FusedTanhAdd { lhs: ValueId, rhs: ValueId },
    FusedGeluAdd { lhs: ValueId, rhs: ValueId },
    FusedSiluAdd { lhs: ValueId, rhs: ValueId },
    FusedSiluMul { lhs: ValueId, rhs: ValueId },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Node {
    pub(crate) op: Op,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub(crate) nodes: Vec<Node>,
    pub(crate) output: Option<ValueId>,
}

impl Program {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            output: None,
        }
    }

    pub fn input(&mut self, shape: &[usize]) -> Result<ValueId> {
        validate_shape(shape)?;
        Ok(self.push(Op::Input {
            shape: shape.to_vec(),
        }))
    }

    // ── unary ops ──

    pub fn relu(&mut self, x: ValueId) -> ValueId { self.push(Op::Relu(x)) }
    pub fn tanh(&mut self, x: ValueId) -> ValueId { self.push(Op::Tanh(x)) }
    pub fn sigmoid(&mut self, x: ValueId) -> ValueId { self.push(Op::Sigmoid(x)) }
    pub fn gelu(&mut self, x: ValueId) -> ValueId { self.push(Op::Gelu(x)) }
    pub fn silu(&mut self, x: ValueId) -> ValueId { self.push(Op::Silu(x)) }
    pub fn abs(&mut self, x: ValueId) -> ValueId { self.push(Op::Abs(x)) }
    pub fn sqrt(&mut self, x: ValueId) -> ValueId { self.push(Op::Sqrt(x)) }
    pub fn exp(&mut self, x: ValueId) -> ValueId { self.push(Op::Exp(x)) }
    pub fn log(&mut self, x: ValueId) -> ValueId { self.push(Op::Log(x)) }

    // ── binary ops ──

    pub fn add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::Add { lhs, rhs }) }
    pub fn mul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::Mul { lhs, rhs }) }
    pub fn sub(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::Sub { lhs, rhs }) }
    pub fn div(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::Div { lhs, rhs }) }

    pub fn fill_like(&mut self, value: f64, like: ValueId) -> ValueId {
        self.push(Op::FillLike { value, like })
    }

    pub fn cumsum(&mut self, input: ValueId, dim: usize) -> ValueId {
        self.push(Op::CumSum { input, dim })
    }

    pub fn topk(&mut self, input: ValueId, k: usize, dim: usize, largest: bool) -> ValueId {
        self.push(Op::TopK { input, k, dim, largest })
    }

    // ── ternary ──

    pub fn where_cond(&mut self, cond: ValueId, true_val: ValueId, false_val: ValueId) -> ValueId {
        self.push(Op::Where { cond, true_val, false_val })
    }

    // ── indexing ──

    pub fn gather(&mut self, input: ValueId, indices: ValueId, dim: usize) -> ValueId {
        self.push(Op::Gather { input, indices, dim })
    }

    pub fn index_select(&mut self, input: ValueId, indices: ValueId, dim: usize) -> ValueId {
        self.push(Op::IndexSelect { input, indices, dim })
    }

    pub fn scatter_add(&mut self, input: ValueId, indices: ValueId, src: ValueId, dim: usize) -> ValueId {
        self.push(Op::ScatterAdd { input, indices, src, dim })
    }

    pub fn argsort(&mut self, input: ValueId, dim: usize, ascending: bool) -> ValueId {
        self.push(Op::Argsort { input, dim, ascending })
    }

    // ── comparison ──

    pub fn cmp_lt(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::CmpLt { lhs, rhs }) }
    pub fn cmp_gt(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::CmpGt { lhs, rhs }) }
    pub fn cmp_le(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::CmpLe { lhs, rhs }) }
    pub fn cmp_ge(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::CmpGe { lhs, rhs }) }
    pub fn cmp_eq(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::CmpEq { lhs, rhs }) }
    pub fn cmp_ne(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::CmpNe { lhs, rhs }) }

    // ── reductions ──

    pub fn reduce_sum(&mut self, input: ValueId, dim: usize) -> ValueId { self.push(Op::ReduceSum { input, dim }) }
    pub fn reduce_mean(&mut self, input: ValueId, dim: usize) -> ValueId { self.push(Op::ReduceMean { input, dim }) }
    pub fn reduce_max(&mut self, input: ValueId, dim: usize) -> ValueId { self.push(Op::ReduceMax { input, dim }) }
    pub fn reduce_min(&mut self, input: ValueId, dim: usize) -> ValueId { self.push(Op::ReduceMin { input, dim }) }
    pub fn argmax(&mut self, input: ValueId, dim: usize) -> ValueId { self.push(Op::Argmax { input, dim }) }
    pub fn argmin(&mut self, input: ValueId, dim: usize) -> ValueId { self.push(Op::Argmin { input, dim }) }
    pub fn softmax(&mut self, input: ValueId, dim: usize) -> ValueId { self.push(Op::Softmax { input, dim }) }

    // ── matmul ──

    pub fn matmul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::Matmul { lhs, rhs }) }

    // ── fused ops ──

    pub fn fused_relu_add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::FusedReluAdd { lhs, rhs }) }
    pub fn fused_relu_mul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::FusedReluMul { lhs, rhs }) }
    pub fn fused_sigmoid_add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::FusedSigmoidAdd { lhs, rhs }) }
    pub fn fused_tanh_add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::FusedTanhAdd { lhs, rhs }) }
    pub fn fused_gelu_add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::FusedGeluAdd { lhs, rhs }) }
    pub fn fused_silu_add(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::FusedSiluAdd { lhs, rhs }) }
    pub fn fused_silu_mul(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId { self.push(Op::FusedSiluMul { lhs, rhs }) }

    pub fn set_output(&mut self, output: ValueId) {
        self.output = Some(output);
    }

    pub fn compile(self) -> Result<CompiledProgram> {
        let output = self.output.ok_or(LangError::MissingOutput)?;
        if output.index() >= self.nodes.len() {
            return Err(LangError::InvalidNodeRef(output.index()));
        }

        let mut shapes: Vec<Vec<usize>> = Vec::with_capacity(self.nodes.len());
        let mut numels: Vec<usize> = Vec::with_capacity(self.nodes.len());
        let mut input_ids = Vec::new();

        for (i, node) in self.nodes.iter().enumerate() {
            let (shape, n) = match &node.op {
                Op::Input { shape } => {
                    validate_shape(shape)?;
                    input_ids.push(ValueId(i));
                    (shape.clone(), numel(shape))
                }
                // ── unary ops (same shape as input) ──
                Op::Relu(x) | Op::Tanh(x) | Op::Sigmoid(x)
                | Op::Gelu(x) | Op::Silu(x) | Op::Abs(x)
                | Op::Sqrt(x) | Op::Exp(x) | Op::Log(x) => {
                    let shape = get_shape(&shapes, *x)?;
                    let n = numel(&shape);
                    (shape, n)
                }
                Op::CumSum { input, dim } => {
                    let shape = get_shape(&shapes, *input)?;
                    if *dim >= shape.len() {
                        return Err(LangError::InvalidNodeRef(*dim));
                    }
                    let n = numel(&shape);
                    (shape, n)
                }
                Op::TopK { input, k, dim, .. } => {
                    let mut shape = get_shape(&shapes, *input)?;
                    if *dim >= shape.len() {
                        return Err(LangError::InvalidNodeRef(*dim));
                    }
                    shape[*dim] = *k;
                    let n = numel(&shape);
                    (shape, n)
                }
                // ── binary ops (shapes must match) ──
                Op::Add { lhs, rhs } | Op::Mul { lhs, rhs }
                | Op::Sub { lhs, rhs } | Op::Div { lhs, rhs }
                | Op::CmpLt { lhs, rhs } | Op::CmpGt { lhs, rhs }
                | Op::CmpLe { lhs, rhs } | Op::CmpGe { lhs, rhs }
                | Op::CmpEq { lhs, rhs } | Op::CmpNe { lhs, rhs }
                | Op::FusedReluAdd { lhs, rhs } | Op::FusedReluMul { lhs, rhs }
                | Op::FusedSigmoidAdd { lhs, rhs } | Op::FusedTanhAdd { lhs, rhs }
                | Op::FusedGeluAdd { lhs, rhs } | Op::FusedSiluAdd { lhs, rhs }
                | Op::FusedSiluMul { lhs, rhs } => {
                    let lhs_shape = get_shape(&shapes, *lhs)?;
                    let rhs_shape = get_shape(&shapes, *rhs)?;
                    if lhs_shape != rhs_shape {
                        let op_name = match &node.op {
                            Op::Add { .. } | Op::FusedReluAdd { .. }
                            | Op::FusedSigmoidAdd { .. } | Op::FusedTanhAdd { .. }
                            | Op::FusedGeluAdd { .. } | Op::FusedSiluAdd { .. } => "add",
                            Op::Sub { .. } => "sub",
                            Op::Div { .. } => "div",
                            Op::CmpLt { .. } => "cmp_lt",
                            Op::CmpGt { .. } => "cmp_gt",
                            Op::CmpLe { .. } => "cmp_le",
                            Op::CmpGe { .. } => "cmp_ge",
                            Op::CmpEq { .. } => "cmp_eq",
                            Op::CmpNe { .. } => "cmp_ne",
                            _ => "mul",
                        };
                        return Err(LangError::ShapeMismatch {
                            op: op_name,
                            left: lhs_shape,
                            right: rhs_shape,
                        });
                    }
                    let n = numel(&lhs_shape);
                    (lhs_shape, n)
                }
                Op::FillLike { like, .. } => {
                    let shape = get_shape(&shapes, *like)?;
                    let n = numel(&shape);
                    (shape, n)
                }
                // ── ternary ──
                Op::Where { cond, true_val, false_val } => {
                    let cs = get_shape(&shapes, *cond)?;
                    let ts = get_shape(&shapes, *true_val)?;
                    let fs = get_shape(&shapes, *false_val)?;
                    if cs != ts || ts != fs {
                        return Err(LangError::ShapeMismatch {
                            op: "where",
                            left: ts,
                            right: fs,
                        });
                    }
                    let n = numel(&cs);
                    (cs, n)
                }
                // ── indexing ──
                Op::Gather { indices, .. } => {
                    // Output shape = indices shape
                    let shape = get_shape(&shapes, *indices)?;
                    let n = numel(&shape);
                    (shape, n)
                }
                Op::IndexSelect { input, indices, dim } => {
                    let mut shape = get_shape(&shapes, *input)?;
                    let idx_shape = get_shape(&shapes, *indices)?;
                    if *dim < shape.len() {
                        shape[*dim] = idx_shape.iter().product();
                    }
                    let n = numel(&shape);
                    (shape, n)
                }
                Op::ScatterAdd { input, .. } => {
                    // Output shape = input shape
                    let shape = get_shape(&shapes, *input)?;
                    let n = numel(&shape);
                    (shape, n)
                }
                Op::Argsort { input, .. } => {
                    // Output shape = input shape (contains indices)
                    let shape = get_shape(&shapes, *input)?;
                    let n = numel(&shape);
                    (shape, n)
                }
                // ── reductions ──
                Op::ReduceSum { input, dim } | Op::ReduceMean { input, dim }
                | Op::ReduceMax { input, dim } | Op::ReduceMin { input, dim }
                | Op::Argmax { input, dim } | Op::Argmin { input, dim } => {
                    let inp_shape = get_shape(&shapes, *input)?;
                    if *dim >= inp_shape.len() {
                        return Err(LangError::InvalidNodeRef(*dim));
                    }
                    let mut shape: Vec<usize> = inp_shape.iter().enumerate()
                        .filter(|(idx, _)| *idx != *dim)
                        .map(|(_, &d)| d)
                        .collect();
                    if shape.is_empty() {
                        shape.push(1);
                    }
                    let n = numel(&shape);
                    (shape, n)
                }
                Op::Softmax { input, .. } => {
                    // Softmax preserves shape
                    let shape = get_shape(&shapes, *input)?;
                    let n = numel(&shape);
                    (shape, n)
                }
                // ── matmul ──
                Op::Matmul { lhs, rhs } => {
                    let l = get_shape(&shapes, *lhs)?;
                    let r = get_shape(&shapes, *rhs)?;
                    if l.len() != 2 || r.len() != 2 {
                        return Err(LangError::ShapeMismatch {
                            op: "matmul",
                            left: l,
                            right: r,
                        });
                    }
                    if l[1] != r[0] {
                        return Err(LangError::ShapeMismatch {
                            op: "matmul",
                            left: l,
                            right: r,
                        });
                    }
                    let shape = vec![l[0], r[1]];
                    let n = numel(&shape);
                    (shape, n)
                }
            };

            shapes.push(shape);
            numels.push(n);
        }

        Ok(CompiledProgram {
            nodes: self.nodes,
            shapes,
            numels,
            input_ids,
            output,
        })
    }

    pub(crate) fn push(&mut self, op: Op) -> ValueId {
        let id = ValueId(self.nodes.len());
        self.nodes.push(Node { op });
        id
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompiledProgram {
    nodes: Vec<Node>,
    shapes: Vec<Vec<usize>>,
    numels: Vec<usize>,
    input_ids: Vec<ValueId>,
    output: ValueId,
}

impl CompiledProgram {
    pub fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.input_ids
            .iter()
            .map(|id| self.shapes[id.index()].clone())
            .collect()
    }

    pub fn output_shape(&self) -> &[usize] {
        &self.shapes[self.output.index()]
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn total_elements(&self) -> usize {
        self.numels.iter().sum()
    }

    pub fn input_count(&self) -> usize {
        self.input_ids.len()
    }

    pub fn op_summary(&self) -> Vec<(&'static str, usize)> {
        let mut counts = std::collections::HashMap::new();
        for node in &self.nodes {
            let name = match &node.op {
                Op::Input { .. } => "input",
                Op::Relu(_) => "relu",
                Op::Tanh(_) => "tanh",
                Op::Sigmoid(_) => "sigmoid",
                Op::Gelu(_) => "gelu",
                Op::Silu(_) => "silu",
                Op::Abs(_) => "abs",
                Op::Sqrt(_) => "sqrt",
                Op::Exp(_) => "exp",
                Op::Log(_) => "log",
                Op::Add { .. } => "add",
                Op::Mul { .. } => "mul",
                Op::Sub { .. } => "sub",
                Op::Div { .. } => "div",
                Op::FillLike { .. } => "fill_like",
                Op::CumSum { .. } => "cumsum",
                Op::TopK { .. } => "topk",
                Op::Where { .. } => "where",
                Op::Gather { .. } => "gather",
                Op::IndexSelect { .. } => "index_select",
                Op::ScatterAdd { .. } => "scatter_add",
                Op::Argsort { .. } => "argsort",
                Op::CmpLt { .. } => "cmp_lt",
                Op::CmpGt { .. } => "cmp_gt",
                Op::CmpLe { .. } => "cmp_le",
                Op::CmpGe { .. } => "cmp_ge",
                Op::CmpEq { .. } => "cmp_eq",
                Op::CmpNe { .. } => "cmp_ne",
                Op::ReduceSum { .. } => "reduce_sum",
                Op::ReduceMean { .. } => "reduce_mean",
                Op::ReduceMax { .. } => "reduce_max",
                Op::ReduceMin { .. } => "reduce_min",
                Op::Argmax { .. } => "argmax",
                Op::Argmin { .. } => "argmin",
                Op::Softmax { .. } => "softmax",
                Op::Matmul { .. } => "matmul",
                Op::FusedReluAdd { .. } => "fused_relu_add",
                Op::FusedReluMul { .. } => "fused_relu_mul",
                Op::FusedSigmoidAdd { .. } => "fused_sigmoid_add",
                Op::FusedTanhAdd { .. } => "fused_tanh_add",
                Op::FusedGeluAdd { .. } => "fused_gelu_add",
                Op::FusedSiluAdd { .. } => "fused_silu_add",
                Op::FusedSiluMul { .. } => "fused_silu_mul",
            };
            *counts.entry(name).or_insert(0usize) += 1;
        }
        let mut out: Vec<_> = counts.into_iter().collect();
        out.sort_by(|a, b| b.1.cmp(&a.1));
        out
    }
}

#[derive(Clone, Debug)]
pub struct HostTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl HostTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        validate_shape(&shape)?;
        let expected = numel(&shape);
        if expected != data.len() {
            return Err(LangError::InputLenMismatch {
                index: 0,
                expected,
                got: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

pub struct GpuLangRuntime {
    runtime: Arc<PtxRuntime>,
}

impl GpuLangRuntime {
    pub fn new(device_id: i32) -> Result<Self> {
        let config = ptx_runtime::PTXStableConfig {
            struct_size: std::mem::size_of::<ptx_runtime::PTXStableConfig>() as u32,
            abi_version: ptx_runtime::PTX_STABLE_ABI_VERSION,
            flags: 0,
            device_id,
            pool_fraction: 0.70,
            fixed_pool_size: 0,
            reserve_vram: 256 * 1024 * 1024,
            max_streams: 16,
            quiet_init: 0,
            enable_leak_detection: 1,
            enable_pool_health: 1,
            _reserved0: 0,
        };
        let runtime = Arc::new(PtxRuntime::with_stable_config(device_id, Some(config))?);
        runtime.export_for_hook();
        runtime.export_context();
        Ok(Self { runtime })
    }

    pub fn from_runtime(runtime: Arc<PtxRuntime>) -> Self {
        Self { runtime }
    }

    pub fn with_max_streams(device_id: i32, max_streams: u32) -> Result<Self> {
        let config = ptx_runtime::PTXStableConfig {
            struct_size: std::mem::size_of::<ptx_runtime::PTXStableConfig>() as u32,
            abi_version: ptx_runtime::PTX_STABLE_ABI_VERSION,
            flags: 0,
            device_id,
            pool_fraction: 0.70,
            fixed_pool_size: 0,
            reserve_vram: 256 * 1024 * 1024,
            max_streams,
            quiet_init: 0,
            enable_leak_detection: 1,
            enable_pool_health: 1,
            _reserved0: 0,
        };
        let runtime = Arc::new(PtxRuntime::with_stable_config(device_id, Some(config))?);
        runtime.export_for_hook();
        runtime.export_context();
        Ok(Self { runtime })
    }

    pub fn num_streams(&self) -> usize {
        self.runtime.num_streams()
    }

    pub fn runtime(&self) -> &Arc<PtxRuntime> {
        &self.runtime
    }

    pub fn execute(&self, program: &CompiledProgram, inputs: &[HostTensor]) -> Result<HostTensor> {
        if inputs.len() != program.input_ids.len() {
            return Err(LangError::InputCountMismatch {
                expected: program.input_ids.len(),
                got: inputs.len(),
            });
        }

        for (i, (id, input)) in program.input_ids.iter().zip(inputs.iter()).enumerate() {
            let expected_shape = &program.shapes[id.index()];
            if input.shape() != expected_shape {
                return Err(LangError::InputShapeMismatch {
                    index: i,
                    expected: expected_shape.clone(),
                    got: input.shape().to_vec(),
                });
            }
            let expected_numel = program.numels[id.index()];
            if input.data().len() != expected_numel {
                return Err(LangError::InputLenMismatch {
                    index: i,
                    expected: expected_numel,
                    got: input.data().len(),
                });
            }
        }

        let stream = self.runtime.next_stream();
        let runtime_ptr = self.runtime.raw();
        let ctx = KernelContext::new(runtime_ptr, stream.raw())?;

        let mut slots: Vec<Option<GpuPtr>> = (0..program.nodes.len()).map(|_| None).collect();
        let mut scratch: Vec<GpuPtr> = Vec::new();
        let mut input_data_map: HashMap<usize, &HostTensor> = HashMap::new();
        for (i, id) in program.input_ids.iter().enumerate() {
            input_data_map.insert(id.index(), &inputs[i]);
        }

        for (i, node) in program.nodes.iter().enumerate() {
            let numel = program.numels[i];
            let bytes = numel * std::mem::size_of::<f32>();

            let out_ptr = match &node.op {
                Op::Input { .. } => {
                    let tensor = input_data_map
                        .get(&i)
                        .expect("input id must exist during execution");
                    let gpu = self.runtime.alloc(bytes)?;
                    unsafe {
                        gpu.copy_from_host(tensor.data().as_ptr() as *const c_void, bytes)?;
                    }
                    gpu
                }

                // ── unary activations ──
                Op::Relu(x) => {
                    let inp = get_slot(&slots, *x)?;
                    let out = self.runtime.alloc(bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    unary::relu(&ig, &og, numel, &ctx)?;
                    out
                }
                Op::Tanh(x) => {
                    let inp = get_slot(&slots, *x)?;
                    let out = self.runtime.alloc(bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    unary::tanh(&ig, &og, numel, &ctx)?;
                    out
                }
                Op::Sigmoid(x) => {
                    let inp = get_slot(&slots, *x)?;
                    let out = self.runtime.alloc(bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    unary::sigmoid(&ig, &og, numel, &ctx)?;
                    out
                }
                Op::Gelu(x) => {
                    let inp = get_slot(&slots, *x)?;
                    let out = self.runtime.alloc(bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    unary::gelu(&ig, &og, numel, &ctx)?;
                    out
                }
                Op::Silu(x) => {
                    let inp = get_slot(&slots, *x)?;
                    let out = self.runtime.alloc(bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    unary::silu(&ig, &og, numel, &ctx)?;
                    out
                }
                Op::Abs(x) => {
                    let inp = get_slot(&slots, *x)?;
                    let out = self.runtime.alloc(bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    unary::abs(&ig, &og, numel, &ctx)?;
                    out
                }
                Op::Sqrt(x) => {
                    let inp = get_slot(&slots, *x)?;
                    let out = self.runtime.alloc(bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    unary::sqrt(&ig, &og, numel, &ctx)?;
                    out
                }
                Op::Exp(x) => {
                    let inp = get_slot(&slots, *x)?;
                    let out = self.runtime.alloc(bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    unary::exp(&ig, &og, numel, &ctx)?;
                    out
                }
                Op::Log(x) => {
                    let inp = get_slot(&slots, *x)?;
                    let out = self.runtime.alloc(bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    unary::log(&ig, &og, numel, &ctx)?;
                    out
                }

                // ── binary elementwise ──
                Op::Add { lhs, rhs } => {
                    let (l, r, og, out) = alloc_binary(&self.runtime, &slots, *lhs, *rhs, bytes, runtime_ptr)?;
                    binary::add(&l, &r, &og, numel, &ctx)?;
                    out
                }
                Op::Mul { lhs, rhs } => {
                    let (l, r, og, out) = alloc_binary(&self.runtime, &slots, *lhs, *rhs, bytes, runtime_ptr)?;
                    binary::mul(&l, &r, &og, numel, &ctx)?;
                    out
                }
                Op::Sub { lhs, rhs } => {
                    let (l, r, og, out) = alloc_binary(&self.runtime, &slots, *lhs, *rhs, bytes, runtime_ptr)?;
                    binary::sub(&l, &r, &og, numel, &ctx)?;
                    out
                }
                Op::Div { lhs, rhs } => {
                    let (l, r, og, out) = alloc_binary(&self.runtime, &slots, *lhs, *rhs, bytes, runtime_ptr)?;
                    binary::div(&l, &r, &og, numel, &ctx)?;
                    out
                }

                Op::FillLike { value, .. } => {
                    let out = self.runtime.alloc(bytes)?;
                    let host = vec![*value as f32; numel];
                    unsafe {
                        out.copy_from_host(host.as_ptr() as *const c_void, bytes)?;
                    }
                    out
                }

                Op::CumSum { input, dim } => {
                    let inp = get_slot(&slots, *input)?;
                    let shape = &program.shapes[i];
                    let outer: usize = shape[..*dim].iter().product();
                    let dim_size = shape[*dim];
                    let inner: usize = shape[*dim + 1..].iter().product();
                    let out = self.runtime.alloc(bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    scan::cumsum(&ig, &og, outer, dim_size, inner, &ctx)?;
                    out
                }
                Op::TopK { input, k, dim, largest } => {
                    let inp = get_slot(&slots, *input)?;
                    let inp_shape = &program.shapes[input.index()];
                    let outer: usize = inp_shape[..*dim].iter().product();
                    let dim_size = inp_shape[*dim];
                    let inner: usize = inp_shape[*dim + 1..].iter().product();
                    let out = self.runtime.alloc(bytes)?;
                    let idx_bytes = numel * std::mem::size_of::<i32>();
                    let idx_buf = self.runtime.alloc(idx_bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    let idxg = unsafe { GuardedBuffer::new(idx_buf.as_ptr(), idx_buf.size(), runtime_ptr)? };
                    topk::topk(&ig, &og, &idxg, outer, dim_size, inner, *k, *largest, &ctx)?;
                    scratch.push(idx_buf);
                    out
                }

                // ── ternary: where ──
                Op::Where { cond, true_val, false_val } => {
                    let cond_ptr = get_slot(&slots, *cond)?;
                    let true_ptr = get_slot(&slots, *true_val)?;
                    let false_ptr = get_slot(&slots, *false_val)?;
                    let out = self.runtime.alloc(bytes)?;

                    // Convert f32 cond to u8: non-zero -> 255, zero -> 0
                    let cond_u8_bytes = numel;
                    let cond_u8 = self.runtime.alloc(cond_u8_bytes)?;
                    {
                        let mut host_f32 = vec![0.0f32; numel];
                        unsafe {
                            cond_ptr.copy_to_host(host_f32.as_mut_ptr() as *mut c_void, bytes)?;
                        }
                        let host_u8: Vec<u8> = host_f32.iter().map(|&v| if v != 0.0 { 255u8 } else { 0u8 }).collect();
                        unsafe {
                            cond_u8.copy_from_host(host_u8.as_ptr() as *const c_void, cond_u8_bytes)?;
                        }
                    }

                    let cg = unsafe { GuardedBuffer::new(cond_u8.as_ptr(), cond_u8.size(), runtime_ptr)? };
                    let tg = unsafe { GuardedBuffer::new(true_ptr.as_ptr(), true_ptr.size(), runtime_ptr)? };
                    let fg = unsafe { GuardedBuffer::new(false_ptr.as_ptr(), false_ptr.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    ternary::where_cond(&cg, &tg, &fg, &og, numel, &ctx)?;
                    scratch.push(cond_u8);
                    out
                }

                // ── indexing ──
                Op::Gather { input, indices, dim } => {
                    let inp = get_slot(&slots, *input)?;
                    let idx = get_slot(&slots, *indices)?;
                    let inp_shape = &program.shapes[input.index()];
                    let idx_shape = &program.shapes[indices.index()];
                    let outer: usize = inp_shape[..*dim].iter().product();
                    let input_dim_size = inp_shape[*dim];
                    let idx_dim_size = idx_shape[*dim];
                    let inner: usize = inp_shape[*dim + 1..].iter().product();
                    let out = self.runtime.alloc(bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let idxg = unsafe { GuardedBuffer::new(idx.as_ptr(), idx.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    gather::gather(&ig, &idxg, &og, outer, input_dim_size, idx_dim_size, inner, &ctx)?;
                    out
                }
                Op::IndexSelect { input, indices, dim } => {
                    let inp = get_slot(&slots, *input)?;
                    let idx = get_slot(&slots, *indices)?;
                    let inp_shape = &program.shapes[input.index()];
                    let idx_numel: usize = program.shapes[indices.index()].iter().product();
                    let left_size: usize = inp_shape[..*dim].iter().product();
                    let src_dim_size = inp_shape[*dim];
                    let right_size: usize = inp_shape[*dim + 1..].iter().product();
                    let out = self.runtime.alloc(bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let idxg = unsafe { GuardedBuffer::new(idx.as_ptr(), idx.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    indexing::index_select(&ig, &idxg, &og, left_size, src_dim_size, idx_numel, right_size, &ctx)?;
                    out
                }
                Op::ScatterAdd { input, indices, src, dim } => {
                    let inp = get_slot(&slots, *input)?;
                    let idx = get_slot(&slots, *indices)?;
                    let src_ptr = get_slot(&slots, *src)?;
                    let inp_shape = &program.shapes[input.index()];
                    let src_shape = &program.shapes[src.index()];
                    let left_size: usize = src_shape[..*dim].iter().product();
                    let src_dim_size = src_shape[*dim];
                    let dst_dim_size = inp_shape[*dim];
                    let right_size: usize = src_shape[*dim + 1..].iter().product();

                    // Copy input to output first (scatter_add accumulates into output)
                    let out = self.runtime.alloc(bytes)?;
                    out.copy_from_device(inp)?;
                    let idxg = unsafe { GuardedBuffer::new(idx.as_ptr(), idx.size(), runtime_ptr)? };
                    let sg = unsafe { GuardedBuffer::new(src_ptr.as_ptr(), src_ptr.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    indexing::scatter_add(&idxg, &sg, &og, left_size, src_dim_size, dst_dim_size, right_size, &ctx)?;
                    out
                }
                Op::Argsort { input, dim, ascending } => {
                    let inp = get_slot(&slots, *input)?;
                    let inp_shape = &program.shapes[input.index()];
                    let nrows: usize = inp_shape[..*dim].iter().product();
                    let ncols = inp_shape[*dim];
                    let out = self.runtime.alloc(bytes)?;
                    let ig = unsafe { GuardedBuffer::new(inp.as_ptr(), inp.size(), runtime_ptr)? };
                    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
                    sort::argsort(&ig, &og, nrows, ncols, *ascending, &ctx)?;
                    out
                }

                // ── comparison ops: TLSF staging → cpu_kernels ──
                Op::CmpLt { lhs, rhs } => {
                    let l = download_to_tlsf(get_slot(&slots, *lhs)?, numel)?;
                    let r = download_to_tlsf(get_slot(&slots, *rhs)?, numel)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::cmp_lt(l.as_f32_slice(numel), r.as_f32_slice(numel), out_buf.as_f32_mut_slice(numel));
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(numel))?
                }
                Op::CmpGt { lhs, rhs } => {
                    let l = download_to_tlsf(get_slot(&slots, *lhs)?, numel)?;
                    let r = download_to_tlsf(get_slot(&slots, *rhs)?, numel)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::cmp_gt(l.as_f32_slice(numel), r.as_f32_slice(numel), out_buf.as_f32_mut_slice(numel));
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(numel))?
                }
                Op::CmpLe { lhs, rhs } => {
                    let l = download_to_tlsf(get_slot(&slots, *lhs)?, numel)?;
                    let r = download_to_tlsf(get_slot(&slots, *rhs)?, numel)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::cmp_le(l.as_f32_slice(numel), r.as_f32_slice(numel), out_buf.as_f32_mut_slice(numel));
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(numel))?
                }
                Op::CmpGe { lhs, rhs } => {
                    let l = download_to_tlsf(get_slot(&slots, *lhs)?, numel)?;
                    let r = download_to_tlsf(get_slot(&slots, *rhs)?, numel)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::cmp_ge(l.as_f32_slice(numel), r.as_f32_slice(numel), out_buf.as_f32_mut_slice(numel));
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(numel))?
                }
                Op::CmpEq { lhs, rhs } => {
                    let l = download_to_tlsf(get_slot(&slots, *lhs)?, numel)?;
                    let r = download_to_tlsf(get_slot(&slots, *rhs)?, numel)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::cmp_eq(l.as_f32_slice(numel), r.as_f32_slice(numel), out_buf.as_f32_mut_slice(numel));
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(numel))?
                }
                Op::CmpNe { lhs, rhs } => {
                    let l = download_to_tlsf(get_slot(&slots, *lhs)?, numel)?;
                    let r = download_to_tlsf(get_slot(&slots, *rhs)?, numel)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::cmp_ne(l.as_f32_slice(numel), r.as_f32_slice(numel), out_buf.as_f32_mut_slice(numel));
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(numel))?
                }

                // ── reductions: TLSF staging → cpu_kernels ──
                // With `torch` feature: libtorch MKL/OpenBLAS vectorized ops.
                // Without: simple Rust loops. Either way, all staging via TLSF.
                Op::ReduceSum { input, dim } => {
                    let inp_shape = &program.shapes[input.index()];
                    let inp_numel = program.numels[input.index()];
                    let inp_buf = download_to_tlsf(get_slot(&slots, *input)?, inp_numel)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::reduce_sum(inp_buf.as_f32_slice(inp_numel), out_buf.as_f32_mut_slice(numel), inp_shape, *dim);
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(numel))?
                }
                Op::ReduceMean { input, dim } => {
                    let inp_shape = &program.shapes[input.index()];
                    let inp_numel = program.numels[input.index()];
                    let inp_buf = download_to_tlsf(get_slot(&slots, *input)?, inp_numel)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::reduce_mean(inp_buf.as_f32_slice(inp_numel), out_buf.as_f32_mut_slice(numel), inp_shape, *dim);
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(numel))?
                }
                Op::ReduceMax { input, dim } => {
                    let inp_shape = &program.shapes[input.index()];
                    let inp_numel = program.numels[input.index()];
                    let inp_buf = download_to_tlsf(get_slot(&slots, *input)?, inp_numel)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::reduce_max(inp_buf.as_f32_slice(inp_numel), out_buf.as_f32_mut_slice(numel), inp_shape, *dim);
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(numel))?
                }
                Op::ReduceMin { input, dim } => {
                    let inp_shape = &program.shapes[input.index()];
                    let inp_numel = program.numels[input.index()];
                    let inp_buf = download_to_tlsf(get_slot(&slots, *input)?, inp_numel)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::reduce_min(inp_buf.as_f32_slice(inp_numel), out_buf.as_f32_mut_slice(numel), inp_shape, *dim);
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(numel))?
                }
                Op::Argmax { input, dim } => {
                    let inp_shape = &program.shapes[input.index()];
                    let inp_numel = program.numels[input.index()];
                    let inp_buf = download_to_tlsf(get_slot(&slots, *input)?, inp_numel)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::argmax(inp_buf.as_f32_slice(inp_numel), out_buf.as_f32_mut_slice(numel), inp_shape, *dim);
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(numel))?
                }
                Op::Argmin { input, dim } => {
                    let inp_shape = &program.shapes[input.index()];
                    let inp_numel = program.numels[input.index()];
                    let inp_buf = download_to_tlsf(get_slot(&slots, *input)?, inp_numel)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::argmin(inp_buf.as_f32_slice(inp_numel), out_buf.as_f32_mut_slice(numel), inp_shape, *dim);
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(numel))?
                }
                Op::Softmax { input, dim } => {
                    let inp_shape = &program.shapes[input.index()];
                    let inp_numel = program.numels[input.index()];
                    let inp_buf = download_to_tlsf(get_slot(&slots, *input)?, inp_numel)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::softmax(inp_buf.as_f32_slice(inp_numel), out_buf.as_f32_mut_slice(numel), inp_shape, *dim);
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(numel))?
                }

                // ── matmul: TLSF staging → cpu_kernels ──
                Op::Matmul { lhs, rhs } => {
                    let l_shape = &program.shapes[lhs.index()];
                    let r_shape = &program.shapes[rhs.index()];
                    let m = l_shape[0];
                    let k = l_shape[1];
                    let n = r_shape[1];

                    let l_buf = download_to_tlsf(get_slot(&slots, *lhs)?, m * k)?;
                    let r_buf = download_to_tlsf(get_slot(&slots, *rhs)?, k * n)?;
                    let mut out_buf = TlsfBuf::new(bytes)?;
                    cpu_kernels::matmul(l_buf.as_f32_slice(m * k), r_buf.as_f32_slice(k * n), out_buf.as_f32_mut_slice(m * n), m, k, n);
                    upload_from_slice(&self.runtime, out_buf.as_f32_slice(m * n))?
                }

                // ── fused ops ──
                Op::FusedReluAdd { lhs, rhs } => {
                    let (l, r, og, out) = alloc_binary(&self.runtime, &slots, *lhs, *rhs, bytes, runtime_ptr)?;
                    binary::add(&l, &r, &og, numel, &ctx)?;
                    unary::relu(&og, &og, numel, &ctx)?;
                    out
                }
                Op::FusedReluMul { lhs, rhs } => {
                    let (l, r, og, out) = alloc_binary(&self.runtime, &slots, *lhs, *rhs, bytes, runtime_ptr)?;
                    binary::mul(&l, &r, &og, numel, &ctx)?;
                    unary::relu(&og, &og, numel, &ctx)?;
                    out
                }
                Op::FusedSigmoidAdd { lhs, rhs } => {
                    let (l, r, og, out) = alloc_binary(&self.runtime, &slots, *lhs, *rhs, bytes, runtime_ptr)?;
                    binary::add(&l, &r, &og, numel, &ctx)?;
                    unary::sigmoid(&og, &og, numel, &ctx)?;
                    out
                }
                Op::FusedTanhAdd { lhs, rhs } => {
                    let (l, r, og, out) = alloc_binary(&self.runtime, &slots, *lhs, *rhs, bytes, runtime_ptr)?;
                    binary::add(&l, &r, &og, numel, &ctx)?;
                    unary::tanh(&og, &og, numel, &ctx)?;
                    out
                }
                Op::FusedGeluAdd { lhs, rhs } => {
                    let (l, r, og, out) = alloc_binary(&self.runtime, &slots, *lhs, *rhs, bytes, runtime_ptr)?;
                    binary::add(&l, &r, &og, numel, &ctx)?;
                    unary::gelu(&og, &og, numel, &ctx)?;
                    out
                }
                Op::FusedSiluAdd { lhs, rhs } => {
                    let (l, r, og, out) = alloc_binary(&self.runtime, &slots, *lhs, *rhs, bytes, runtime_ptr)?;
                    binary::add(&l, &r, &og, numel, &ctx)?;
                    unary::silu(&og, &og, numel, &ctx)?;
                    out
                }
                Op::FusedSiluMul { lhs, rhs } => {
                    let (l, r, og, out) = alloc_binary(&self.runtime, &slots, *lhs, *rhs, bytes, runtime_ptr)?;
                    binary::mul(&l, &r, &og, numel, &ctx)?;
                    unary::silu(&og, &og, numel, &ctx)?;
                    out
                }
            };

            slots[i] = Some(out_ptr);
        }

        stream.synchronize()?;
        drop(scratch);

        let out_id = program.output.index();
        let out_shape = program.shapes[out_id].clone();
        let out_numel = program.numels[out_id];
        let out_bytes = out_numel * std::mem::size_of::<f32>();
        let mut host = vec![0.0f32; out_numel];
        let out_gpu = slots[out_id]
            .as_ref()
            .ok_or(LangError::InvalidNodeRef(out_id))?;

        unsafe {
            out_gpu.copy_to_host(host.as_mut_ptr() as *mut c_void, out_bytes)?;
        }

        Ok(HostTensor {
            shape: out_shape,
            data: host,
        })
    }
}

// ── Execution helpers ──────────────────────────────────────────

fn get_slot(slots: &[Option<GpuPtr>], id: ValueId) -> Result<&GpuPtr> {
    slots[id.index()].as_ref().ok_or(LangError::InvalidNodeRef(id.index()))
}

fn alloc_binary(
    runtime: &PtxRuntime,
    slots: &[Option<GpuPtr>],
    lhs: ValueId,
    rhs: ValueId,
    bytes: usize,
    runtime_ptr: *mut ptx_sys::GPUHotRuntime,
) -> Result<(GuardedBuffer, GuardedBuffer, GuardedBuffer, GpuPtr)> {
    let l = get_slot(slots, lhs)?;
    let r = get_slot(slots, rhs)?;
    let out = runtime.alloc(bytes)?;
    let lg = unsafe { GuardedBuffer::new(l.as_ptr(), l.size(), runtime_ptr)? };
    let rg = unsafe { GuardedBuffer::new(r.as_ptr(), r.size(), runtime_ptr)? };
    let og = unsafe { GuardedBuffer::new(out.as_ptr(), out.size(), runtime_ptr)? };
    Ok((lg, rg, og, out))
}

// ── TLSF scratch buffer (RAII) ────────────────────────────────

/// A CPU scratch buffer backed by the global TLSF pool.
/// Automatically returns memory to the pool on drop.
struct TlsfBuf {
    ptr: std::ptr::NonNull<u8>,
    bytes: usize,
    align: usize,
}

impl TlsfBuf {
    fn new(bytes: usize) -> Result<Self> {
        let align = std::mem::align_of::<f32>();
        let ptr = runtime::cpu_tlsf::global_cpu_tlsf()
            .allocate(bytes.max(4), align)?;
        Ok(Self { ptr, bytes, align })
    }

    fn as_f32_slice(&self, numel: usize) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<f32>().as_ptr(), numel) }
    }

    fn as_f32_mut_slice(&mut self, numel: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<f32>().as_ptr(), numel) }
    }

    fn as_void_ptr(&self) -> *mut c_void {
        self.ptr.as_ptr() as *mut c_void
    }
}

impl Drop for TlsfBuf {
    fn drop(&mut self) {
        let _ = unsafe {
            runtime::cpu_tlsf::global_cpu_tlsf()
                .deallocate(self.ptr, self.bytes, self.align)
        };
    }
}

/// Download GPU slot to a TLSF-backed host buffer.
fn download_to_tlsf(gpu: &GpuPtr, numel: usize) -> Result<TlsfBuf> {
    let bytes = numel * std::mem::size_of::<f32>();
    let buf = TlsfBuf::new(bytes)?;
    unsafe { gpu.copy_to_host(buf.as_void_ptr(), bytes)?; }
    Ok(buf)
}

/// Upload a host f32 slice to a new GPU allocation.
fn upload_from_slice(runtime: &PtxRuntime, data: &[f32]) -> Result<GpuPtr> {
    let bytes = data.len() * std::mem::size_of::<f32>();
    let out = runtime.alloc(bytes)?;
    unsafe { out.copy_from_host(data.as_ptr() as *const c_void, bytes)?; }
    Ok(out)
}

fn get_shape(shapes: &[Vec<usize>], id: ValueId) -> Result<Vec<usize>> {
    shapes
        .get(id.index())
        .cloned()
        .ok_or(LangError::InvalidNodeRef(id.index()))
}

fn validate_shape(shape: &[usize]) -> Result<()> {
    if shape.is_empty() || shape.iter().any(|&d| d == 0) {
        return Err(LangError::InvalidShape(shape.to_vec()));
    }
    Ok(())
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

// ── Modules ──────────────────────────────────────────────────────

pub use ferrite_apps as apps;

pub mod runtime;
pub mod jit;
mod cpu_kernels;

// Always available (no feature gate for traits/types):
pub mod sensor;
pub mod vision;
pub mod pipeline;

#[cfg(feature = "torch")]
pub mod torch;

#[cfg(feature = "capture")]
pub mod capture;

// ── Backwards-compatible module re-exports ───────────────────────

pub use runtime::context;
pub use runtime::cpu_tlsf;
pub use runtime::tensor;

#[cfg(feature = "torch")]
pub use torch::cv;
#[cfg(feature = "torch")]
pub use torch::dataflow;
#[cfg(feature = "torch")]
pub use torch::bridge as torch_bridge;

// ── Type-level re-exports ────────────────────────────────────────

pub use runtime::context::{cpu, gpu, fer, CpuCtx, FerCtx, FerStats, GpuCtx, HasAllocator, HasRuntime};
#[cfg(feature = "torch")]
pub use runtime::context::{gpu_anyhow, fer_anyhow};
pub use runtime::cpu_tlsf::CpuTlsfStats;
pub use runtime::tensor::{CpuTensor, GpuTensor, ToCpu, ToGpu};

// ── Sensor re-exports ────────────────────────────────────────────

pub use sensor::{SensorStream, Stamped, SensorInfo, SensorClock, CaptureThread};
#[cfg(feature = "capture")]
pub use sensor::CameraAdapter;

// ── Vision re-exports ────────────────────────────────────────────

pub use vision::{BoundingBox, Detection, Track, Tracker, TrackerConfig, nms};
#[cfg(feature = "capture")]
pub use vision::ops::{resize, crop, letterbox, convert_format};
#[cfg(feature = "capture")]
pub use vision::draw::{draw_bbox, draw_detections, draw_tracks};

// ── Pipeline re-exports ──────────────────────────────────────────

pub use pipeline::{RingBuffer, SharedRing, Stage, StageMetrics, PipelineStats};

// ── Capture re-exports ───────────────────────────────────────────

#[cfg(feature = "capture")]
pub use capture::camera::{Camera, CameraConfig, CameraSource};
#[cfg(feature = "capture")]
pub use capture::frame::{Frame, FrameMeta, FramePool, PixelFormat};
#[cfg(feature = "capture")]
pub use capture::convert::{frame_to_host_tensor, Normalize};
#[cfg(all(feature = "capture", feature = "torch"))]
pub use capture::bridge::FrameToTensor;
