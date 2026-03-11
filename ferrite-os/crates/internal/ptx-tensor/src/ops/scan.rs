//! Scan/prefix operations (cumulative sum, product, max, min).

use crate::tensor::Tensor;
use crate::dtype::DType;
use crate::shape::{contiguous_strides, reduction_sizes, checked_elem_count};
use crate::storage::Storage;
use ptx_runtime::{Result, Error, increment_ops};

/// Scan operation type.
#[derive(Debug, Clone, Copy)]
enum ScanOp {
    Sum,
    Prod,
    Max,
    Min,
}

impl Tensor {
    /// Generic scan operation along a dimension.
    fn scan_op(&self, dim: i32, op: ScanOp) -> Result<Tensor> {
        let input = self.require_contiguous()?;
        let ndim = input.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim };
        if dim < 0 || dim >= ndim {
            return Err(Error::Internal {
                message: format!("Invalid dimension {} for tensor with {} dims", dim, ndim),
            });
        }
        let dim = dim as usize;

        let out_shape: smallvec::SmallVec<[usize; 8]> = input.shape().iter().copied().collect();
        let (outer, dim_size, inner) = reduction_sizes(input.shape(), dim);

        let out_elems = checked_elem_count(&out_shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        let out_storage = Storage::new(
            out_elems,
            input.dtype(),
            input.runtime(),
        )?;
        let output = Tensor::from_storage(
            out_storage,
            out_shape.clone(),
            contiguous_strides(&out_shape),
            0,
        );

        let stream = input.runtime().next_stream();

        match input.dtype() {
            DType::F32 => unsafe {
                let inp = input.data_ptr_typed::<f32>();
                let out = output.data_ptr_typed::<f32>();
                match op {
                    ScanOp::Sum => ptx_sys::ptx_tensor_cumsum_f32(inp, out, outer, dim_size, inner, stream.raw()),
                    ScanOp::Prod => ptx_sys::ptx_tensor_cumprod_f32(inp, out, outer, dim_size, inner, stream.raw()),
                    ScanOp::Max => ptx_sys::ptx_tensor_cummax_f32(inp, out, outer, dim_size, inner, stream.raw()),
                    ScanOp::Min => ptx_sys::ptx_tensor_cummin_f32(inp, out, outer, dim_size, inner, stream.raw()),
                }
            },
            dtype => return Err(Error::NotSupported {
                message: format!("Scan ops not supported for {:?}", dtype),
            }),
        }

        stream.synchronize()?;
        increment_ops();
        Ok(output)
    }

    /// Inclusive cumulative sum along a dimension.
    pub fn cumsum(&self, dim: i32) -> Result<Tensor> {
        self.scan_op(dim, ScanOp::Sum)
    }

    /// Inclusive cumulative product along a dimension.
    pub fn cumprod(&self, dim: i32) -> Result<Tensor> {
        self.scan_op(dim, ScanOp::Prod)
    }

    /// Inclusive cumulative max along a dimension.
    pub fn cummax(&self, dim: i32) -> Result<Tensor> {
        self.scan_op(dim, ScanOp::Max)
    }

    /// Inclusive cumulative min along a dimension.
    pub fn cummin(&self, dim: i32) -> Result<Tensor> {
        self.scan_op(dim, ScanOp::Min)
    }
}
