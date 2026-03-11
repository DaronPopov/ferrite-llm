use super::*;

impl Tensor {
    /// Create a new tensor with the given storage and shape.
    pub fn from_storage(
        storage: Storage,
        shape: Shape,
        strides: Strides,
        offset: usize,
    ) -> Self {
        let dtype = storage.dtype();
        Self {
            storage,
            shape,
            strides,
            offset,
            dtype,
        }
    }

    /// Create a new contiguous tensor with the given shape.
    pub fn new(shape: &[usize], dtype: DType, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let shape = Shape::from_slice(shape);
        let strides = contiguous_strides(&shape);
        let len = checked_elem_count(&shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        let storage = Storage::new(len, dtype, runtime)?;
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
            dtype,
        })
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[usize], dtype: DType, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let shape = Shape::from_slice(shape);
        let strides = contiguous_strides(&shape);
        let len = checked_elem_count(&shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;
        let storage = Storage::zeros(len, dtype, runtime)?;
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
            dtype,
        })
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &[usize], dtype: DType, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let tensor = Self::zeros(shape, dtype, runtime)?;
        if dtype == DType::F32 {
            tensor.storage.fill_f32(1.0)?;
        } else {
            return Err(Error::NotSupported {
                message: format!("ones not implemented for {:?}", dtype),
            });
        }
        Ok(tensor)
    }

    /// Create a tensor filled with a specific value.
    pub fn full(shape: &[usize], value: f32, dtype: DType, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let tensor = Self::new(shape, dtype, runtime)?;
        if dtype == DType::F32 {
            tensor.storage.fill_f32(value)?;
        } else {
            return Err(Error::NotSupported {
                message: format!("full not implemented for {:?}", dtype),
            });
        }
        Ok(tensor)
    }

    /// Create a tensor from host data.
    pub fn from_slice<T: bytemuck::Pod>(
        data: &[T],
        shape: &[usize],
        dtype: DType,
        runtime: &Arc<PtxRuntime>,
    ) -> Result<Self> {
        let shape = Shape::from_slice(shape);
        let strides = contiguous_strides(&shape);
        let len = checked_elem_count(&shape).map_err(|msg| Error::Internal {
            message: msg.to_string(),
        })?;

        if len != data.len() {
            return Err(Error::ShapeMismatch {
                expected: vec![len],
                actual: vec![data.len()],
            });
        }

        let storage = unsafe { Storage::from_host(data, dtype, runtime)? };
        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
            dtype,
        })
    }

    /// Create a 1D tensor with evenly spaced values: [start, start+step, start+2*step, ...].
    pub fn arange(start: f32, end: f32, step: f32, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        let n = ((end - start) / step).ceil() as usize;
        if n == 0 {
            return Self::new(&[0], DType::F32, runtime);
        }
        let tensor = Self::new(&[n], DType::F32, runtime)?;
        let stream = runtime.next_stream();
        unsafe {
            ptx_sys::ptx_tensor_arange_f32(
                tensor.data_ptr_typed::<f32>(),
                n,
                start,
                step,
                stream.raw(),
            );
        }
        stream.synchronize()?;
        Ok(tensor)
    }

    /// Create a 1D tensor with n evenly spaced values from start to end (inclusive).
    pub fn linspace(start: f32, end: f32, steps: usize, runtime: &Arc<PtxRuntime>) -> Result<Self> {
        if steps == 0 {
            return Self::new(&[0], DType::F32, runtime);
        }
        if steps == 1 {
            return Self::from_slice(&[start], &[1], DType::F32, runtime);
        }
        let step = (end - start) / (steps - 1) as f32;
        let tensor = Self::new(&[steps], DType::F32, runtime)?;
        let stream = runtime.next_stream();
        unsafe {
            ptx_sys::ptx_tensor_arange_f32(
                tensor.data_ptr_typed::<f32>(),
                steps,
                start,
                step,
                stream.raw(),
            );
        }
        stream.synchronize()?;
        Ok(tensor)
    }

    /// Create an empty tensor with the same shape and dtype as another.
    pub fn empty_like(&self) -> Result<Self> {
        Self::new(&self.shape, self.dtype, self.runtime())
    }

    /// Create a zeros tensor with the same shape and dtype as another.
    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(&self.shape, self.dtype, self.runtime())
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get the shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get a specific dimension size.
    pub fn dim(&self, d: usize) -> usize {
        self.shape[d]
    }

    /// Get the total number of elements.
    pub fn elem_count(&self) -> usize {
        elem_count(&self.shape)
    }

    /// Get the size in bytes.
    pub fn size_bytes(&self) -> usize {
        size_bytes(&self.shape, self.dtype.size_bytes())
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the runtime.
    pub fn runtime(&self) -> &Arc<PtxRuntime> {
        self.storage.runtime()
    }

    /// Get the raw data pointer (with offset applied).
    pub fn data_ptr(&self) -> *mut libc::c_void {
        let base = self.storage.as_ptr() as *mut u8;
        unsafe { base.add(self.offset * self.dtype.size_bytes()) as *mut libc::c_void }
    }

    /// Get the typed data pointer.
    pub fn data_ptr_typed<T>(&self) -> *mut T {
        self.data_ptr() as *mut T
    }

    /// Check if the tensor is contiguous.
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(&self.shape, &self.strides)
    }

    /// Check if this tensor shares storage with another.
    pub fn shares_storage(&self, other: &Tensor) -> bool {
        self.storage.same_storage(&other.storage)
    }

}
