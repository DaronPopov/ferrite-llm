#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

#[derive(Debug)]
pub enum InteropError {
    AlreadyRegistered,
    NotRegistered,
    HostCopyForbidden,
    Backend(String),
}

impl fmt::Display for InteropError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyRegistered => write!(f, "buffer already registered"),
            Self::NotRegistered => write!(f, "buffer not registered"),
            Self::HostCopyForbidden => write!(f, "host copy forbidden in zero-copy mode"),
            Self::Backend(msg) => write!(f, "backend error: {}", msg),
        }
    }
}

impl std::error::Error for InteropError {}

/// Strongly typed device pointer view. Constructing this is unsafe and must
/// only be done for valid CUDA device memory.
#[derive(Clone, Copy)]
pub struct DeviceSliceF32 {
    ptr: *const f32,
    len: usize,
}

impl DeviceSliceF32 {
    /// # Safety
    /// Caller must ensure `ptr..ptr+len` is valid device memory.
    pub unsafe fn from_raw(ptr: *const f32, len: usize) -> Self {
        Self { ptr, len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn as_ptr(self) -> *const f32 {
        self.ptr
    }
}

pub trait InteropBackend {
    type Handle: Copy + Eq + std::hash::Hash;

    fn register_gl_buffer(&mut self, gl_buffer_id: u32) -> Result<Self::Handle, InteropError>;
    fn write_host_bytes(&mut self, handle: Self::Handle, bytes: &[u8]) -> Result<(), InteropError>;
    fn copy_from_device_f32(
        &mut self,
        handle: Self::Handle,
        src: DeviceSliceF32,
    ) -> Result<(), InteropError>;
    fn unregister(&mut self, handle: Self::Handle) -> Result<(), InteropError>;
}

enum BufferState<H> {
    Unregistered,
    Registered(H),
}

pub struct CudaGlBuffer<B: InteropBackend> {
    gl_buffer_id: u32,
    state: BufferState<B::Handle>,
    backend: B,
}

impl<B: InteropBackend> CudaGlBuffer<B> {
    pub fn new(gl_buffer_id: u32, backend: B) -> Self {
        Self {
            gl_buffer_id,
            state: BufferState::Unregistered,
            backend,
        }
    }

    pub fn register(&mut self) -> Result<(), InteropError> {
        match self.state {
            BufferState::Unregistered => {
                let handle = self.backend.register_gl_buffer(self.gl_buffer_id)?;
                self.state = BufferState::Registered(handle);
                Ok(())
            }
            BufferState::Registered(_) => Err(InteropError::AlreadyRegistered),
        }
    }

    pub fn write_host(&mut self, bytes: &[u8]) -> Result<(), InteropError> {
        let BufferState::Registered(handle) = self.state else {
            return Err(InteropError::NotRegistered);
        };
        self.backend.write_host_bytes(handle, bytes)
    }

    pub fn upload_device_f32(&mut self, src: DeviceSliceF32) -> Result<(), InteropError> {
        let BufferState::Registered(handle) = self.state else {
            return Err(InteropError::NotRegistered);
        };
        self.backend.copy_from_device_f32(handle, src)
    }

    pub fn unregister(&mut self) -> Result<(), InteropError> {
        let handle = match self.state {
            BufferState::Registered(h) => h,
            BufferState::Unregistered => return Err(InteropError::NotRegistered),
        };
        self.backend.unregister(handle)?;
        self.state = BufferState::Unregistered;
        Ok(())
    }
}

pub struct MockInteropBackend {
    next_handle: u64,
    storage: HashMap<u64, Vec<u8>>,
}

impl Default for MockInteropBackend {
    fn default() -> Self {
        Self {
            next_handle: 1,
            storage: HashMap::new(),
        }
    }
}

impl InteropBackend for MockInteropBackend {
    type Handle = u64;

    fn register_gl_buffer(&mut self, _gl_buffer_id: u32) -> Result<Self::Handle, InteropError> {
        let id = self.next_handle;
        self.next_handle = self.next_handle.saturating_add(1);
        self.storage.insert(id, Vec::new());
        Ok(id)
    }

    fn write_host_bytes(&mut self, handle: Self::Handle, bytes: &[u8]) -> Result<(), InteropError> {
        let Some(buf) = self.storage.get_mut(&handle) else {
            return Err(InteropError::NotRegistered);
        };
        buf.clear();
        buf.extend_from_slice(bytes);
        Ok(())
    }

    fn copy_from_device_f32(
        &mut self,
        _handle: Self::Handle,
        _src: DeviceSliceF32,
    ) -> Result<(), InteropError> {
        Err(InteropError::HostCopyForbidden)
    }

    fn unregister(&mut self, handle: Self::Handle) -> Result<(), InteropError> {
        self.storage.remove(&handle);
        Ok(())
    }
}

#[cfg(feature = "cuda-gl-interop")]
pub struct CudaInteropBackend {
    resources: HashMap<u64, ptx_sys::cudaGraphicsResource_t>,
    next_id: u64,
    stream: ptx_sys::cudaStream_t,
}

#[cfg(feature = "cuda-gl-interop")]
impl Default for CudaInteropBackend {
    fn default() -> Self {
        Self {
            resources: HashMap::new(),
            next_id: 1,
            stream: std::ptr::null_mut(),
        }
    }
}

#[cfg(feature = "cuda-gl-interop")]
impl CudaInteropBackend {
    pub fn with_stream(stream: ptx_sys::cudaStream_t) -> Self {
        Self {
            resources: HashMap::new(),
            next_id: 1,
            stream,
        }
    }
}

#[cfg(feature = "cuda-gl-interop")]
impl InteropBackend for CudaInteropBackend {
    type Handle = u64;

    fn register_gl_buffer(&mut self, gl_buffer_id: u32) -> Result<Self::Handle, InteropError> {
        let mut res: ptx_sys::cudaGraphicsResource_t = std::ptr::null_mut();
        cuda_check(
            unsafe {
                ptx_sys::cudaGraphicsGLRegisterBuffer(
                    &mut res as *mut _,
                    gl_buffer_id,
                    ptx_sys::cudaGraphicsRegisterFlagsWriteDiscard,
                )
            },
            "cudaGraphicsGLRegisterBuffer",
        )?;
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.resources.insert(id, res);
        Ok(id)
    }

    fn write_host_bytes(
        &mut self,
        _handle: Self::Handle,
        _bytes: &[u8],
    ) -> Result<(), InteropError> {
        Err(InteropError::HostCopyForbidden)
    }

    fn copy_from_device_f32(
        &mut self,
        handle: Self::Handle,
        src: DeviceSliceF32,
    ) -> Result<(), InteropError> {
        let Some(resource) = self.resources.get_mut(&handle) else {
            return Err(InteropError::NotRegistered);
        };

        let mut list = [*resource];
        cuda_check(
            unsafe { ptx_sys::cudaGraphicsMapResources(1, list.as_mut_ptr(), self.stream) },
            "cudaGraphicsMapResources",
        )?;

        let mut dst_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut mapped_size: usize = 0;
        let mapped = cuda_check(
            unsafe {
                ptx_sys::cudaGraphicsResourceGetMappedPointer(
                    &mut dst_ptr as *mut _,
                    &mut mapped_size as *mut _,
                    *resource,
                )
            },
            "cudaGraphicsResourceGetMappedPointer",
        );
        if let Err(e) = mapped {
            let _ = unsafe { ptx_sys::cudaGraphicsUnmapResources(1, list.as_mut_ptr(), self.stream) };
            return Err(e);
        }

        let bytes = src
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| InteropError::Backend("byte count overflow".to_string()))?;
        if mapped_size < bytes {
            let _ = unsafe { ptx_sys::cudaGraphicsUnmapResources(1, list.as_mut_ptr(), self.stream) };
            return Err(InteropError::Backend(format!(
                "mapped buffer too small: {} < {}",
                mapped_size, bytes
            )));
        }

        let copy = cuda_check(
            unsafe {
                ptx_sys::cudaMemcpyAsync(
                    dst_ptr,
                    src.as_ptr() as *const std::ffi::c_void,
                    bytes,
                    ptx_sys::cudaMemcpyDeviceToDevice,
                    self.stream,
                )
            },
            "cudaMemcpyAsync(D2D)",
        );
        let unmap = cuda_check(
            unsafe { ptx_sys::cudaGraphicsUnmapResources(1, list.as_mut_ptr(), self.stream) },
            "cudaGraphicsUnmapResources",
        );

        copy?;
        unmap?;
        Ok(())
    }

    fn unregister(&mut self, handle: Self::Handle) -> Result<(), InteropError> {
        let Some(resource) = self.resources.remove(&handle) else {
            return Err(InteropError::NotRegistered);
        };
        cuda_check(
            unsafe { ptx_sys::cudaGraphicsUnregisterResource(resource) },
            "cudaGraphicsUnregisterResource",
        )
    }
}

#[cfg(feature = "cuda-gl-interop")]
fn cuda_check(code: ptx_sys::cudaError_t, op: &str) -> Result<(), InteropError> {
    if code == ptx_sys::cudaSuccess {
        return Ok(());
    }
    let msg = unsafe {
        let s = ptx_sys::cudaGetErrorString(code);
        if s.is_null() {
            format!("{} failed with code {}", op, code)
        } else {
            let c = std::ffi::CStr::from_ptr(s);
            format!("{}: {} ({})", op, c.to_string_lossy(), code)
        }
    };
    Err(InteropError::Backend(msg))
}

pub struct InteropPipeline {
    upload: CudaGlBuffer<MockInteropBackend>,
}

impl InteropPipeline {
    pub fn new_mock(gl_buffer_id: u32) -> Result<Self, InteropError> {
        let mut upload = CudaGlBuffer::new(gl_buffer_id, MockInteropBackend::default());
        upload.register()?;
        Ok(Self { upload })
    }

    pub fn upload_f32(&mut self, values: &[f32]) -> Result<(), InteropError> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_ne_bytes());
        }
        self.upload.write_host(&bytes)
    }
}

#[cfg(feature = "cuda-gl-interop")]
pub struct ZeroCopyInteropPipeline {
    upload: CudaGlBuffer<CudaInteropBackend>,
}

#[cfg(feature = "cuda-gl-interop")]
impl ZeroCopyInteropPipeline {
    pub fn new(gl_buffer_id: u32, stream: ptx_sys::cudaStream_t) -> Result<Self, InteropError> {
        let mut upload = CudaGlBuffer::new(gl_buffer_id, CudaInteropBackend::with_stream(stream));
        upload.register()?;
        Ok(Self { upload })
    }

    pub fn upload_from_device(&mut self, src: DeviceSliceF32) -> Result<(), InteropError> {
        self.upload.upload_device_f32(src)
    }

    pub fn upload_from_ipc_hex(
        &mut self,
        ipc_handle_hex: &str,
        len_f32: usize,
    ) -> Result<(), InteropError> {
        let handle = decode_ipc_handle(ipc_handle_hex)?;
        let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        cuda_check(
            unsafe {
                ptx_sys::cudaIpcOpenMemHandle(
                    &mut dev_ptr as *mut _,
                    handle,
                    ptx_sys::cudaIpcMemLazyEnablePeerAccess,
                )
            },
            "cudaIpcOpenMemHandle",
        )?;
        if dev_ptr.is_null() {
            return Err(InteropError::Backend(
                "cudaIpcOpenMemHandle returned null pointer".to_string(),
            ));
        }
        let src = unsafe { DeviceSliceF32::from_raw(dev_ptr as *const f32, len_f32) };
        let copy = self.upload_from_device(src);
        let close = cuda_check(
            unsafe { ptx_sys::cudaIpcCloseMemHandle(dev_ptr) },
            "cudaIpcCloseMemHandle",
        );
        copy?;
        close
    }

    pub fn upload_host_via_device(&mut self, values: &[f32]) -> Result<(), InteropError> {
        let bytes = values
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| InteropError::Backend("byte count overflow".to_string()))?;
        let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        cuda_check(
            unsafe { ptx_sys::cudaMalloc(&mut dev_ptr as *mut _, bytes) },
            "cudaMalloc",
        )?;
        let copy = cuda_check(
            unsafe {
                ptx_sys::cudaMemcpy(
                    dev_ptr,
                    values.as_ptr() as *const std::ffi::c_void,
                    bytes,
                    ptx_sys::cudaMemcpyHostToDevice,
                )
            },
            "cudaMemcpy(H2D)",
        );
        if let Err(e) = copy {
            let _ = unsafe { ptx_sys::cudaFree(dev_ptr) };
            return Err(e);
        }
        let device = unsafe { DeviceSliceF32::from_raw(dev_ptr as *const f32, values.len()) };
        let upload = self.upload_from_device(device);
        let free = cuda_check(unsafe { ptx_sys::cudaFree(dev_ptr) }, "cudaFree");
        upload?;
        free
    }
}

#[cfg(feature = "cuda-gl-interop")]
fn decode_ipc_handle(hex: &str) -> Result<ptx_sys::cudaIpcMemHandle_t, InteropError> {
    if hex.len() != 128 {
        return Err(InteropError::Backend(format!(
            "invalid ipc hex length: {}",
            hex.len()
        )));
    }
    let mut out = ptx_sys::cudaIpcMemHandle_t::default();
    let mut idx = 0usize;
    while idx < 64 {
        let start = idx * 2;
        let end = start + 2;
        let byte = u8::from_str_radix(&hex[start..end], 16)
            .map_err(|e| InteropError::Backend(format!("invalid ipc hex: {}", e)))?;
        out.reserved[idx] = byte as i8;
        idx += 1;
    }
    Ok(out)
}
