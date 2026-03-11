//! Binary element-wise operations.

use crate::tensor::Tensor;
use crate::dtype::DType;
use ptx_runtime::{Result, Error, increment_ops};

/// Binary operation type.
#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Mod,
}

impl Tensor {
    /// Generic binary operation.
    fn binary_op(&self, other: &Tensor, op: BinaryOp) -> Result<Tensor> {
        // Check shapes match
        if self.shape() != other.shape() {
            return Err(Error::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: other.shape().to_vec(),
            });
        }

        // Check dtypes match
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                expected: self.dtype().to_ptx(),
                actual: other.dtype().to_ptx(),
            });
        }

        let lhs = self.require_contiguous()?;
        let rhs = other.require_contiguous()?;

        let output = lhs.empty_like()?;
        let n = lhs.elem_count();
        if n == 0 {
            return Ok(output);
        }
        let stream = lhs.runtime().next_stream();

        match lhs.dtype() {
            DType::F32 => unsafe {
                let a = lhs.data_ptr_typed::<f32>();
                let b = rhs.data_ptr_typed::<f32>();
                let out = output.data_ptr_typed::<f32>();

                match op {
                    BinaryOp::Add => ptx_sys::ptx_tensor_add_f32(a, b, out, n, stream.raw()),
                    BinaryOp::Sub => ptx_sys::ptx_tensor_sub_f32(a, b, out, n, stream.raw()),
                    BinaryOp::Mul => ptx_sys::ptx_tensor_mul_f32(a, b, out, n, stream.raw()),
                    BinaryOp::Div => ptx_sys::ptx_tensor_div_f32(a, b, out, n, stream.raw()),
                    BinaryOp::Max => ptx_sys::ptx_tensor_max_f32(a, b, out, n, stream.raw()),
                    BinaryOp::Min => ptx_sys::ptx_tensor_min_f32(a, b, out, n, stream.raw()),
                    BinaryOp::Mod => ptx_sys::ptx_tensor_mod_f32(a, b, out, n, stream.raw()),
                }
            },
            DType::F64 => unsafe {
                let a = lhs.data_ptr_typed::<f64>();
                let b = rhs.data_ptr_typed::<f64>();
                let out = output.data_ptr_typed::<f64>();

                match op {
                    BinaryOp::Add => ptx_sys::ptx_tensor_add_f64(a, b, out, n, stream.raw()),
                    BinaryOp::Sub => ptx_sys::ptx_tensor_sub_f64(a, b, out, n, stream.raw()),
                    BinaryOp::Mul => ptx_sys::ptx_tensor_mul_f64(a, b, out, n, stream.raw()),
                    BinaryOp::Div => ptx_sys::ptx_tensor_div_f64(a, b, out, n, stream.raw()),
                    _ => return Err(Error::NotSupported {
                        message: format!("{:?} not supported for F64", op),
                    }),
                }
            },
            DType::F16 => unsafe {
                let a = lhs.data_ptr_typed::<ptx_sys::__half>();
                let b = rhs.data_ptr_typed::<ptx_sys::__half>();
                let out = output.data_ptr_typed::<ptx_sys::__half>();

                match op {
                    BinaryOp::Add => ptx_sys::ptx_tensor_add_f16(a, b, out, n, stream.raw()),
                    BinaryOp::Sub => ptx_sys::ptx_tensor_sub_f16(a, b, out, n, stream.raw()),
                    BinaryOp::Mul => ptx_sys::ptx_tensor_mul_f16(a, b, out, n, stream.raw()),
                    BinaryOp::Div => ptx_sys::ptx_tensor_div_f16(a, b, out, n, stream.raw()),
                    _ => return Err(Error::NotSupported {
                        message: format!("{:?} not supported for F16", op),
                    }),
                }
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Binary ops not supported for {:?}", dtype),
            }),
        }

        increment_ops();
        Ok(output)
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, BinaryOp::Add)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, BinaryOp::Sub)
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, BinaryOp::Mul)
    }

    /// Element-wise division.
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, BinaryOp::Div)
    }

    /// Element-wise maximum.
    pub fn maximum(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, BinaryOp::Max)
    }

    /// Element-wise minimum.
    pub fn minimum(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, BinaryOp::Min)
    }

    /// Element-wise modulo.
    pub fn fmod(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, BinaryOp::Mod)
    }

    // ========================================================================
    // Scalar operations
    // ========================================================================

    /// Add a scalar to all elements.
    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "add_scalar only supported for F32".to_string(),
            });
        }

        let input = self.require_contiguous()?;
        let output = input.empty_like()?;
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_add_scalar_f32(
                input.data_ptr_typed::<f32>(),
                scalar,
                output.data_ptr_typed::<f32>(),
                n,
                stream.raw(),
            );
        }

        Ok(output)
    }

    /// Subtract a scalar from all elements.
    pub fn sub_scalar(&self, scalar: f32) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "sub_scalar only supported for F32".to_string(),
            });
        }

        let input = self.require_contiguous()?;
        let output = input.empty_like()?;
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_sub_scalar_f32(
                input.data_ptr_typed::<f32>(),
                scalar,
                output.data_ptr_typed::<f32>(),
                n,
                stream.raw(),
            );
        }

        Ok(output)
    }

    /// Multiply all elements by a scalar.
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "mul_scalar only supported for F32".to_string(),
            });
        }

        let input = self.require_contiguous()?;
        let output = input.empty_like()?;
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_mul_scalar_f32(
                input.data_ptr_typed::<f32>(),
                scalar,
                output.data_ptr_typed::<f32>(),
                n,
                stream.raw(),
            );
        }

        Ok(output)
    }

    /// Divide all elements by a scalar.
    pub fn div_scalar(&self, scalar: f32) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "div_scalar only supported for F32".to_string(),
            });
        }

        let input = self.require_contiguous()?;
        let output = input.empty_like()?;
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_div_scalar_f32(
                input.data_ptr_typed::<f32>(),
                scalar,
                output.data_ptr_typed::<f32>(),
                n,
                stream.raw(),
            );
        }

        Ok(output)
    }

    /// Affine transformation: out = mul * x + add
    pub fn affine(&self, mul: f32, add: f32) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "affine only supported for F32".to_string(),
            });
        }

        let input = self.require_contiguous()?;
        let output = input.empty_like()?;
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_affine_f32(
                input.data_ptr_typed::<f32>(),
                output.data_ptr_typed::<f32>(),
                n,
                mul,
                add,
                stream.raw(),
            );
        }

        Ok(output)
    }

    /// Clamp values to a range.
    pub fn clamp(&self, min_val: f32, max_val: f32) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "clamp only supported for F32".to_string(),
            });
        }

        let input = self.require_contiguous()?;
        let output = input.empty_like()?;
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_clamp_f32(
                input.data_ptr_typed::<f32>(),
                output.data_ptr_typed::<f32>(),
                n,
                min_val,
                max_val,
                stream.raw(),
            );
        }

        Ok(output)
    }

    /// Element-wise power with scalar exponent.
    pub fn pow(&self, exponent: f32) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "pow only supported for F32".to_string(),
            });
        }

        let input = self.require_contiguous()?;
        let output = input.empty_like()?;
        let n = input.elem_count();
        let stream = input.runtime().next_stream();

        unsafe {
            ptx_sys::ptx_tensor_powf_f32(
                input.data_ptr_typed::<f32>(),
                output.data_ptr_typed::<f32>(),
                n,
                exponent,
                stream.raw(),
            );
        }

        Ok(output)
    }
}

// ============================================================================
// Operator overloads
// ============================================================================

use std::ops;

impl ops::Add<&Tensor> for &Tensor {
    type Output = Result<Tensor>;

    fn add(self, rhs: &Tensor) -> Self::Output {
        Tensor::add(self, rhs)
    }
}

impl ops::Sub<&Tensor> for &Tensor {
    type Output = Result<Tensor>;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        Tensor::sub(self, rhs)
    }
}

impl ops::Mul<&Tensor> for &Tensor {
    type Output = Result<Tensor>;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        Tensor::mul(self, rhs)
    }
}

impl ops::Div<&Tensor> for &Tensor {
    type Output = Result<Tensor>;

    fn div(self, rhs: &Tensor) -> Self::Output {
        Tensor::div(self, rhs)
    }
}
