use super::*;

impl Tensor {
    /// Copy tensor data to host.
    ///
    /// If the tensor is non-contiguous (e.g. from a transpose or slice),
    /// a contiguous GPU copy is materialized first, then copied to host.
    pub fn to_vec<T: bytemuck::Pod>(&self) -> Result<Vec<T>> {
        if !self.is_contiguous() {
            let contig = self.contiguous()?;
            return contig.storage.to_host();
        }
        self.storage.to_host()
    }

    /// Copy tensor data to host as f32.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(Error::DTypeMismatch {
                expected: DType::F32.to_ptx(),
                actual: self.dtype.to_ptx(),
            });
        }
        self.to_vec::<f32>()
    }

    /// Copy tensor data to host as i32.
    pub fn to_vec_i32(&self) -> Result<Vec<i32>> {
        if self.dtype != DType::I32 {
            return Err(Error::DTypeMismatch {
                expected: DType::I32.to_ptx(),
                actual: self.dtype.to_ptx(),
            });
        }
        self.to_vec::<i32>()
    }

    /// Copy tensor data to host as u32.
    pub fn to_vec_u32(&self) -> Result<Vec<u32>> {
        if self.dtype != DType::U32 {
            return Err(Error::DTypeMismatch {
                expected: DType::U32.to_ptx(),
                actual: self.dtype.to_ptx(),
            });
        }
        self.to_vec::<u32>()
    }

    /// Copy tensor data to host as u8.
    pub fn to_vec_u8(&self) -> Result<Vec<u8>> {
        if self.dtype != DType::U8 {
            return Err(Error::DTypeMismatch {
                expected: DType::U8.to_ptx(),
                actual: self.dtype.to_ptx(),
            });
        }
        self.to_vec::<u8>()
    }

    // ========================================================================
    // Reshaping
    // ========================================================================

    /// Reshape the tensor (returns a view if possible).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let new_elem_count: usize = checked_elem_count(new_shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        if new_elem_count != self.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: self.shape.to_vec(),
                actual: new_shape.to_vec(),
            });
        }

        if self.is_contiguous() {
            // Can return a view
            Ok(Self {
                storage: self.storage.clone(),
                shape: Shape::from_slice(new_shape),
                strides: contiguous_strides(new_shape),
                offset: self.offset,
                dtype: self.dtype,
            })
        } else {
            // Need to make contiguous first
            let contiguous = self.contiguous()?;
            Ok(Self {
                storage: contiguous.storage,
                shape: Shape::from_slice(new_shape),
                strides: contiguous_strides(new_shape),
                offset: 0,
                dtype: self.dtype,
            })
        }
    }

    /// Flatten the tensor to 1D.
    pub fn flatten(&self) -> Result<Self> {
        self.reshape(&[self.elem_count()])
    }

    /// Squeeze dimensions of size 1.
    pub fn squeeze(&self) -> Result<Self> {
        let new_shape: Shape = self.shape.iter()
            .copied()
            .filter(|&d| d != 1)
            .collect();
        self.reshape(&new_shape)
    }

    /// Unsqueeze: add a dimension of size 1 at the given position.
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let mut new_shape = self.shape.clone();
        new_shape.insert(dim, 1);
        self.reshape(&new_shape)
    }

    /// Transpose two dimensions (returns a view — no data copy).
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return Err(Error::Internal {
                message: format!("transpose dims ({}, {}) out of range for {} dims", dim0, dim1, self.ndim()),
            });
        }
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);
        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
        })
    }

    /// Transpose a 2D matrix (shorthand for transpose(0, 1)).
    pub fn t(&self) -> Result<Self> {
        if self.ndim() != 2 {
            return Err(Error::Internal {
                message: format!("t() requires 2D tensor, got {}D", self.ndim()),
            });
        }
        self.transpose(0, 1)
    }

    /// Permute dimensions (returns a view — no data copy).
    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        if dims.len() != self.ndim() {
            return Err(Error::Internal {
                message: format!("permute requires {} dims, got {}", self.ndim(), dims.len()),
            });
        }
        let mut new_shape = Shape::with_capacity(self.ndim());
        let mut new_strides = Strides::with_capacity(self.ndim());
        for &d in dims {
            if d >= self.ndim() {
                return Err(Error::Internal {
                    message: format!("permute dim {} out of range for {} dims", d, self.ndim()),
                });
            }
            new_shape.push(self.shape[d]);
            new_strides.push(self.strides[d]);
        }
        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
        })
    }

    /// Make the tensor contiguous (copy if necessary).
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let n = self.elem_count();
        let out = Self::new(self.shape(), self.dtype, self.runtime())?;
        let stream = self.runtime().next_stream();

        // Upload shape and strides to GPU for the kernel
        let shape_bytes = self.ndim() * std::mem::size_of::<usize>();
        let shape_gpu = self.runtime().alloc(shape_bytes)?;
        let strides_gpu = self.runtime().alloc(shape_bytes)?;
        unsafe {
            shape_gpu.copy_from_host(
                self.shape().as_ptr() as *const libc::c_void,
                shape_bytes,
            )?;
            strides_gpu.copy_from_host(
                self.strides().as_ptr() as *const libc::c_void,
                shape_bytes,
            )?;
        }

        match self.dtype {
            DType::F32 => unsafe {
                ptx_sys::ptx_tensor_strided_copy_f32(
                    self.data_ptr_typed::<f32>(),
                    out.data_ptr_typed::<f32>(),
                    shape_gpu.as_ptr() as *const usize,
                    strides_gpu.as_ptr() as *const usize,
                    self.ndim() as i32,
                    n,
                    stream.raw(),
                );
            },
            DType::F64 => unsafe {
                ptx_sys::ptx_tensor_strided_copy_f64(
                    self.data_ptr_typed::<f64>(),
                    out.data_ptr_typed::<f64>(),
                    shape_gpu.as_ptr() as *const usize,
                    strides_gpu.as_ptr() as *const usize,
                    self.ndim() as i32,
                    n,
                    stream.raw(),
                );
            },
            DType::U8 => unsafe {
                ptx_sys::ptx_tensor_strided_copy_u8(
                    self.data_ptr_typed::<u8>(),
                    out.data_ptr_typed::<u8>(),
                    shape_gpu.as_ptr() as *const usize,
                    strides_gpu.as_ptr() as *const usize,
                    self.ndim() as i32,
                    n,
                    stream.raw(),
                );
            },
            _ => {
                return Err(Error::NotSupported {
                    message: format!("contiguous() not yet supported for {:?}", self.dtype),
                });
            }
        }

        stream.synchronize()?;
        Ok(out)
    }

    /// Return self if contiguous, else make a contiguous copy.
    pub fn require_contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            self.contiguous()
        }
    }

    /// Create a deep copy of the tensor.
    pub fn clone_tensor(&self) -> Result<Self> {
        let new_storage = self.storage.deep_clone()?;
        Ok(Self {
            storage: new_storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            dtype: self.dtype,
        })
    }

    // ========================================================================
    // Synchronization
    // ========================================================================

    /// Synchronize all pending operations on this tensor.
    pub fn sync(&self) {
        self.runtime().sync_all().expect("sync_all failed");
    }
}
