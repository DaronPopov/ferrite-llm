//! Operation nodes in the computation graph.

use smallvec::SmallVec;
use crate::ir::tensor::TensorId;

/// Unique identifier for a node in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

/// Operation codes matching PTXTensorOpcode.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCode {
    // Special
    Input = 0x00,
    Constant = 0x01,

    // Binary Operations (0x10 - 0x1F)
    Add = 0x10,
    Sub = 0x11,
    Mul = 0x12,
    Div = 0x13,
    Max = 0x14,
    Min = 0x15,
    Pow = 0x16,
    Mod = 0x17,

    // Unary Operations (0x20 - 0x3F)
    Neg = 0x20,
    Abs = 0x21,
    Exp = 0x22,
    Log = 0x23,
    Log2 = 0x24,
    Log10 = 0x25,
    Sqrt = 0x26,
    Rsqrt = 0x27,
    Sin = 0x28,
    Cos = 0x29,
    Tan = 0x2A,
    Tanh = 0x2B,
    Sinh = 0x2C,
    Cosh = 0x2D,
    Ceil = 0x2E,
    Floor = 0x2F,
    Round = 0x30,
    Sign = 0x31,
    Recip = 0x32,
    Sqr = 0x33,
    Erf = 0x34,

    // Activation Functions (0x40 - 0x4F)
    Relu = 0x40,
    Relu6 = 0x41,
    LeakyRelu = 0x42,
    Elu = 0x43,
    Selu = 0x44,
    Gelu = 0x45,
    GeluTanh = 0x46,
    Sigmoid = 0x47,
    Silu = 0x48,
    Softplus = 0x49,
    Mish = 0x4A,
    HardSwish = 0x4B,
    HardSigmoid = 0x4C,

    // Reduction Operations (0x50 - 0x5F)
    ReduceSum = 0x50,
    ReduceMean = 0x51,
    ReduceMax = 0x52,
    ReduceMin = 0x53,
    ReduceProd = 0x54,
    ReduceArgmax = 0x55,
    ReduceArgmin = 0x56,

    // Softmax Operations (0x60 - 0x6F)
    Softmax = 0x60,
    LogSoftmax = 0x61,

    // Comparison Operations (0x70 - 0x7F)
    CmpEq = 0x70,
    CmpNe = 0x71,
    CmpLt = 0x72,
    CmpLe = 0x73,
    CmpGt = 0x74,
    CmpGe = 0x75,

    // Transform Operations (0x80 - 0x8F)
    Affine = 0x80,
    Clamp = 0x81,
    Where = 0x82,

    // Copy Operations (0x90 - 0x9F)
    Copy = 0x90,
    Cast = 0x91,
    Fill = 0x92,

    // Matrix Operations (0xA0 - 0xAF)
    Matmul = 0xA0,
    BatchMatmul = 0xA1,
    Transpose = 0xA2,

    // Gather/Scatter Operations (0xB0 - 0xB5)
    Gather = 0xB0,
    ScatterAdd = 0xB1,
    IndexSelect = 0xB2,
    IndexAdd = 0xB3,

    // Scan/Prefix Operations (0xC0 - 0xC5)
    CumSum = 0xC0,

    // Sort/Select Operations (0xE0 - 0xEF)
    Argsort = 0xE1,
    TopK = 0xE2,
}

impl OpCode {
    /// Check if this is a binary operation.
    pub fn is_binary(&self) -> bool {
        let v = *self as u32;
        (0x10..0x20).contains(&v)
    }

    /// Check if this is a unary operation.
    pub fn is_unary(&self) -> bool {
        let v = *self as u32;
        (0x20..0x40).contains(&v)
    }

    /// Check if this is an activation function.
    pub fn is_activation(&self) -> bool {
        let v = *self as u32;
        (0x40..0x50).contains(&v)
    }

    /// Check if this is a reduction operation.
    pub fn is_reduction(&self) -> bool {
        let v = *self as u32;
        (0x50..0x60).contains(&v)
    }

    /// Check if this is a gather/scatter operation.
    pub fn is_gather(&self) -> bool {
        let v = *self as u32;
        (0xB0..0xC0).contains(&v)
    }

    /// Check if this is a scan/prefix operation.
    pub fn is_scan(&self) -> bool {
        let v = *self as u32;
        (0xC0..0xD0).contains(&v)
    }

    /// Check if this is a sort/select operation.
    pub fn is_selection(&self) -> bool {
        let v = *self as u32;
        (0xE0..0xF0).contains(&v)
    }

    /// Check if this is an elementwise operation.
    pub fn is_elementwise(&self) -> bool {
        self.is_binary() || self.is_unary() || self.is_activation()
    }

    /// Convert to PTX opcode value.
    pub fn to_ptx(&self) -> u32 {
        *self as u32
    }
}

/// Operation attributes (scalars, dimensions, etc.)
#[derive(Debug, Clone, Default)]
pub struct OpAttrs {
    /// Scalar parameter A (for affine, leaky_relu alpha, etc.)
    pub scalar_a: Option<f32>,
    /// Scalar parameter B (for affine, clamp max, etc.)
    pub scalar_b: Option<f32>,
    /// Reduction dimension.
    pub reduce_dim: Option<i32>,
    /// Keep dimension after reduction.
    pub keepdim: bool,
    /// Target dtype for cast operations.
    pub target_dtype: Option<ptx_tensor::DType>,
    /// Transpose dimensions.
    pub transpose_dims: Option<(usize, usize)>,
    /// K value for top-k selection.
    pub k: Option<usize>,
}

impl OpAttrs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_scalar_a(mut self, a: f32) -> Self {
        self.scalar_a = Some(a);
        self
    }

    pub fn with_scalar_b(mut self, b: f32) -> Self {
        self.scalar_b = Some(b);
        self
    }

    pub fn with_reduce_dim(mut self, dim: i32) -> Self {
        self.reduce_dim = Some(dim);
        self
    }

    pub fn with_keepdim(mut self, keepdim: bool) -> Self {
        self.keepdim = keepdim;
        self
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }
}

/// A node in the computation graph.
#[derive(Debug, Clone)]
pub struct Node {
    /// Node ID.
    pub id: NodeId,
    /// Operation code.
    pub op: OpCode,
    /// Input tensor IDs.
    pub inputs: SmallVec<[TensorId; 4]>,
    /// Output tensor ID.
    pub output: TensorId,
    /// Operation attributes.
    pub attrs: OpAttrs,
}

impl Node {
    /// Create a new node.
    pub fn new(
        id: NodeId,
        op: OpCode,
        inputs: SmallVec<[TensorId; 4]>,
        output: TensorId,
        attrs: OpAttrs,
    ) -> Self {
        Self {
            id,
            op,
            inputs,
            output,
            attrs,
        }
    }

    /// Check if this is an input node.
    pub fn is_input(&self) -> bool {
        self.op == OpCode::Input
    }

    /// Check if this is a constant node.
    pub fn is_constant(&self) -> bool {
        self.op == OpCode::Constant
    }

    /// Get the number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }
}
