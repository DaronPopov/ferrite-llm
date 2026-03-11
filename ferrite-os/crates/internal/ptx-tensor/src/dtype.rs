//! Data types for tensors.

use std::fmt;

/// Data types supported by tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DType {
    /// 32-bit floating point (default)
    #[default]
    F32,
    /// 64-bit floating point
    F64,
    /// 16-bit floating point (half precision)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 32-bit unsigned integer
    U32,
}

impl DType {
    /// Get the size in bytes for this data type.
    pub const fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
            DType::U32 => 4,
        }
    }

    /// Check if this is a floating point type.
    pub const fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
    }

    /// Check if this is a signed integer type.
    pub const fn is_signed(&self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64)
    }

    /// Check if this is an unsigned integer type.
    pub const fn is_unsigned(&self) -> bool {
        matches!(self, DType::U8 | DType::U32)
    }

    /// Check if this is an integer type (signed or unsigned).
    pub const fn is_integer(&self) -> bool {
        self.is_signed() || self.is_unsigned()
    }

    /// Convert to PTX-OS dtype.
    pub fn to_ptx(&self) -> ptx_sys::PTXDType {
        match self {
            DType::F32 => ptx_sys::PTXDType::F32,
            DType::F64 => ptx_sys::PTXDType::F64,
            DType::F16 => ptx_sys::PTXDType::F16,
            DType::BF16 => ptx_sys::PTXDType::BF16,
            DType::I8 => ptx_sys::PTXDType::I8,
            DType::I16 => ptx_sys::PTXDType::I16,
            DType::I32 => ptx_sys::PTXDType::I32,
            DType::I64 => ptx_sys::PTXDType::I64,
            DType::U8 => ptx_sys::PTXDType::U8,
            DType::U32 => ptx_sys::PTXDType::U32,
        }
    }

    /// Convert from PTX-OS dtype.
    pub fn from_ptx(dtype: ptx_sys::PTXDType) -> Self {
        match dtype {
            ptx_sys::PTXDType::F32 => DType::F32,
            ptx_sys::PTXDType::F64 => DType::F64,
            ptx_sys::PTXDType::F16 => DType::F16,
            ptx_sys::PTXDType::BF16 => DType::BF16,
            ptx_sys::PTXDType::I8 => DType::I8,
            ptx_sys::PTXDType::I16 => DType::I16,
            ptx_sys::PTXDType::I32 => DType::I32,
            ptx_sys::PTXDType::I64 => DType::I64,
            ptx_sys::PTXDType::U8 => DType::U8,
            ptx_sys::PTXDType::U32 => DType::U32,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::I8 => write!(f, "i8"),
            DType::I16 => write!(f, "i16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "u8"),
            DType::U32 => write!(f, "u32"),
        }
    }
}

/// Trait for types that can be used as tensor elements.
///
/// Requires `bytemuck::Pod` to guarantee safe reinterpretation between
/// byte slices and typed slices during GPU memory transfers.
pub trait TensorDType: bytemuck::Pod + Default + Send + Sync + 'static {
    /// The corresponding DType variant.
    const DTYPE: DType;

    /// Zero value.
    fn zero() -> Self;

    /// One value.
    fn one() -> Self;
}

impl TensorDType for f32 {
    const DTYPE: DType = DType::F32;
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl TensorDType for f64 {
    const DTYPE: DType = DType::F64;
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}

impl TensorDType for half::f16 {
    const DTYPE: DType = DType::F16;
    fn zero() -> Self { half::f16::from_f32(0.0) }
    fn one() -> Self { half::f16::from_f32(1.0) }
}

impl TensorDType for i8 {
    const DTYPE: DType = DType::I8;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl TensorDType for i16 {
    const DTYPE: DType = DType::I16;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl TensorDType for i32 {
    const DTYPE: DType = DType::I32;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl TensorDType for i64 {
    const DTYPE: DType = DType::I64;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl TensorDType for u8 {
    const DTYPE: DType = DType::U8;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}

impl TensorDType for u32 {
    const DTYPE: DType = DType::U32;
    fn zero() -> Self { 0 }
    fn one() -> Self { 1 }
}
