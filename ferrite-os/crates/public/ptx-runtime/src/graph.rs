//! CUDA graph capture and replay.

use crate::stream::Stream;
use crate::error::{Error, Result};

/// A captured CUDA graph that can be replayed efficiently.
pub struct CudaGraph {
    graph_id: i32,
    runtime: *mut ptx_sys::GPUHotRuntime,
    name: String,
}

impl CudaGraph {
    /// Create a new graph wrapper (internal use).
    pub(crate) fn new(
        graph_id: i32,
        runtime: *mut ptx_sys::GPUHotRuntime,
        name: String,
    ) -> Self {
        Self {
            graph_id,
            runtime,
            name,
        }
    }

    /// Get the graph ID.
    pub fn id(&self) -> i32 {
        self.graph_id
    }

    /// Get the graph name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Launch the graph on the given stream.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        unsafe {
            ptx_sys::gpu_hot_launch_graph(self.runtime, self.graph_id, stream.raw());
            let err = ptx_sys::cudaGetLastError();
            Error::check_cuda(err)?;
        }
        Ok(())
    }

    /// Launch the graph on the default stream.
    pub fn launch_default(&self) -> Result<()> {
        unsafe {
            ptx_sys::gpu_hot_launch_graph(self.runtime, self.graph_id, std::ptr::null_mut());
            let err = ptx_sys::cudaGetLastError();
            Error::check_cuda(err)?;
        }
        Ok(())
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe {
            ptx_sys::gpu_hot_destroy_graph(self.runtime, self.graph_id);
        }
    }
}

// Safety: CUDA graphs can be used from multiple threads
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

/// Builder for capturing CUDA graphs.
pub struct GraphCapture {
    runtime: *mut ptx_sys::GPUHotRuntime,
    stream_id: i32,
    name: String,
    capturing: bool,
}

impl GraphCapture {
    /// Start capturing a new graph.
    pub(crate) fn begin(
        runtime: *mut ptx_sys::GPUHotRuntime,
        stream_id: i32,
        name: &str,
    ) -> Result<Self> {
        let c_name = std::ffi::CString::new(name).map_err(|_| Error::Internal {
            message: "Invalid graph name".to_string(),
        })?;

        let result = unsafe {
            ptx_sys::gpu_hot_begin_capture(runtime, stream_id, c_name.as_ptr())
        };

        if result < 0 {
            return Err(Error::GraphError {
                message: "Failed to begin graph capture".to_string(),
            });
        }

        Ok(Self {
            runtime,
            stream_id,
            name: name.to_string(),
            capturing: true,
        })
    }

    /// Get the stream ID being captured.
    pub fn stream_id(&self) -> i32 {
        self.stream_id
    }

    /// Check if capture is active.
    pub fn is_capturing(&self) -> bool {
        self.capturing
    }

    /// End capture and create the graph.
    pub fn end(mut self) -> Result<CudaGraph> {
        if !self.capturing {
            return Err(Error::GraphError {
                message: "Graph capture already ended".to_string(),
            });
        }

        let graph_id = unsafe {
            ptx_sys::gpu_hot_end_capture(self.runtime, self.stream_id)
        };

        self.capturing = false;

        if graph_id < 0 {
            return Err(Error::GraphError {
                message: "Failed to end graph capture".to_string(),
            });
        }

        Ok(CudaGraph::new(graph_id, self.runtime, self.name.clone()))
    }

    /// Abort capture without creating a graph.
    pub fn abort(mut self) {
        self.capturing = false;
        // The graph is automatically cleaned up when capture is abandoned
    }
}

impl Drop for GraphCapture {
    fn drop(&mut self) {
        if self.capturing {
            // If dropped while capturing, end the capture but ignore the result
            unsafe {
                ptx_sys::gpu_hot_end_capture(self.runtime, self.stream_id);
            }
        }
    }
}

/// Low-level CUDA graph wrapper using raw CUDA API.
pub struct RawCudaGraph {
    graph: ptx_sys::cudaGraph_t,
    exec: ptx_sys::cudaGraphExec_t,
}

impl RawCudaGraph {
    /// Create from raw handles (takes ownership).
    pub unsafe fn from_raw(
        graph: ptx_sys::cudaGraph_t,
        exec: ptx_sys::cudaGraphExec_t,
    ) -> Self {
        Self { graph, exec }
    }

    /// Get the raw graph handle.
    pub fn graph(&self) -> ptx_sys::cudaGraph_t {
        self.graph
    }

    /// Get the executable graph handle.
    pub fn exec(&self) -> ptx_sys::cudaGraphExec_t {
        self.exec
    }

    /// Launch the graph on a stream.
    pub fn launch(&self, stream: ptx_sys::cudaStream_t) -> Result<()> {
        let err = unsafe { ptx_sys::ptx_tensor_graph_launch(self.exec, stream) };
        Error::check_cuda(err)
    }
}
