use super::*;

impl PtxRuntime {
    /// Get a stream by ID.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidStreamId` if `id` is negative or >= `num_streams()`.
    pub fn stream(&self, id: i32) -> Result<Stream> {
        if id < 0 || id >= self.streams.len() as i32 {
            return Err(Error::InvalidStreamId {
                id,
                pool_size: self.streams.len(),
            });
        }

        // SAFETY: bounds check above guarantees id is a valid index
        self.streams
            .get(id as usize)
            .ok_or(Error::InvalidStreamId {
                id,
                pool_size: self.streams.len(),
            })
    }

    /// Get the next stream in round-robin order.
    pub fn next_stream(&self) -> Stream {
        self.streams.next()
    }

    /// Get the number of streams in the pool.
    pub fn num_streams(&self) -> usize {
        self.streams.len()
    }

    /// Get a stream by priority.
    ///
    /// # Errors
    ///
    /// Returns `Error::StreamError` if the underlying FFI call returns a null stream.
    pub fn priority_stream(&self, priority: StreamPriority) -> Result<Stream> {
        let raw = unsafe {
            ptx_sys::gpu_hot_get_priority_stream(self.inner.raw, priority as i32)
        };
        if raw.is_null() {
            return Err(Error::StreamError {
                message: format!("priority stream request returned null for priority {:?}", priority),
            });
        }
        Ok(Stream::new(raw, priority as i32))
    }

    /// Get access to the stream pool.
    pub fn stream_pool(&self) -> &StreamPool {
        &self.streams
    }

    /// Synchronize all streams.
    ///
    /// # Errors
    ///
    /// Returns `Error::CudaError` if any stream fails to synchronize.
    pub fn sync_all(&self) -> Result<()> {
        let _timer = crate::telemetry::OpTimer::new("sync_all");
        crate::telemetry::metrics().record_stream_sync();

        tracing::trace!("Synchronizing all streams");

        self.streams.synchronize_all()
    }

    // ========================================================================
    // CUDA Graphs
    // ========================================================================

    /// Begin capturing a CUDA graph.
    ///
    /// All operations on the returned stream will be captured until `end_capture` is called.
    pub fn begin_capture(&self, stream_id: i32, name: &str) -> Result<GraphCapture> {
        GraphCapture::begin(self.inner.raw, stream_id, name)
    }

    /// Launch a previously captured graph.
    pub fn launch_graph(&self, graph: &CudaGraph, stream: &Stream) -> Result<()> {
        graph.launch(stream)
    }

    // ========================================================================
    // cuBLAS
    // ========================================================================

    /// Get or create the cuBLAS handle.
    pub fn cublas(&self) -> Result<parking_lot::MutexGuard<'_, Option<CublasHandle>>> {
        let mut guard = self.cublas.lock();
        if guard.is_none() {
            *guard = Some(CublasHandle::new()?);
        }
        Ok(guard)
    }
}
