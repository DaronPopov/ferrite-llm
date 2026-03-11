//! Tensor descriptors in the IR.

use smallvec::SmallVec;
use ptx_tensor::DType;

/// Unique identifier for a tensor in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub u32);

impl TensorId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

/// Device identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DeviceId(pub i32);

impl DeviceId {
    pub fn gpu(id: i32) -> Self {
        Self(id)
    }

    pub fn cpu() -> Self {
        Self(-1)
    }

    pub fn is_gpu(&self) -> bool {
        self.0 >= 0
    }
}

/// Metadata for a tensor in the graph.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Tensor ID.
    pub id: TensorId,
    /// Shape.
    pub shape: SmallVec<[usize; 8]>,
    /// Data type.
    pub dtype: DType,
    /// Device.
    pub device: DeviceId,
    /// Whether this tensor is an input.
    pub is_input: bool,
    /// Whether this tensor is an output.
    pub is_output: bool,
    /// Whether this is a constant.
    pub is_constant: bool,
    /// Constant value (if is_constant and small).
    pub constant_value: Option<f32>,
}

impl TensorMeta {
    /// Create a new tensor metadata.
    pub fn new(id: TensorId, shape: &[usize], dtype: DType) -> Self {
        Self {
            id,
            shape: SmallVec::from_slice(shape),
            dtype,
            device: DeviceId::default(),
            is_input: false,
            is_output: false,
            is_constant: false,
            constant_value: None,
        }
    }

    /// Create an input tensor metadata.
    pub fn input(id: TensorId, shape: &[usize], dtype: DType) -> Self {
        let mut meta = Self::new(id, shape, dtype);
        meta.is_input = true;
        meta
    }

    /// Create a constant tensor metadata.
    pub fn constant(id: TensorId, shape: &[usize], dtype: DType, value: Option<f32>) -> Self {
        let mut meta = Self::new(id, shape, dtype);
        meta.is_constant = true;
        meta.constant_value = value;
        meta
    }

    /// Get the number of elements.
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.elem_count() * self.dtype.size_bytes()
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Check if shapes are compatible for elementwise operations.
    pub fn shape_compatible(&self, other: &TensorMeta) -> bool {
        self.shape == other.shape
    }

    /// Mark as output.
    pub fn mark_output(&mut self) {
        self.is_output = true;
    }
}
