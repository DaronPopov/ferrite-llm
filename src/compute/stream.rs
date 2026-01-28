// Stream - Zero-copy streaming execution pipeline
//
// Keeps data on GPU, uses CUDA streams for async execution,
// only copies to host when explicitly requested.

use std::collections::HashMap;
use std::sync::Arc;
use crate::dynamics::allocator::TlsfAllocator;
use crate::compute::spcpp;

/// GPU memory handle - just an offset into the pool, no copies
#[derive(Debug, Clone, Copy)]
pub struct GpuHandle {
    pub offset: usize,
    pub size: usize,  // in bytes
}

/// Operation in the execution graph
#[derive(Debug, Clone)]
pub enum StreamOp {
    // Memory
    Alloc { size: usize, out: &'static str },
    Free { name: &'static str },

    // Data movement (only when needed)
    HostToDevice { name: &'static str },
    DeviceToHost { name: &'static str },

    // Compute (zero-copy, operates on GPU handles)
    MatMul { a: &'static str, b: &'static str, c: &'static str, m: i32, k: i32, n: i32 },
    Gemm { a: &'static str, b: &'static str, c: &'static str, m: i32, k: i32, n: i32, alpha: f32, beta: f32, trans_a: bool, trans_b: bool },

    // Unary (in-place, zero-copy)
    ReLU { x: &'static str, n: i32 },
    Sigmoid { x: &'static str, n: i32 },
    Tanh { x: &'static str, n: i32 },
    GeLU { x: &'static str, n: i32 },
    SiLU { x: &'static str, n: i32 },
    Softmax { x: &'static str, rows: i32, cols: i32 },
    Scale { x: &'static str, scalar: f32, n: i32 },

    // Binary (zero-copy)
    Add { a: &'static str, b: &'static str, c: &'static str, n: i32 },
    Mul { a: &'static str, b: &'static str, c: &'static str, n: i32 },

    // Linear layer (fused, zero-copy)
    Linear { input: &'static str, weights: &'static str, bias: Option<&'static str>, output: &'static str, batch: i32, in_f: i32, out_f: i32 },

    // Optimizer (in-place on GPU)
    AdamStep { name: &'static str, weights: &'static str, grad: &'static str, n: i32 },
    SGDStep { name: &'static str, weights: &'static str, grad: &'static str, n: i32, lr: f32, momentum: f32, wd: f32 },
    ZeroGrad { grad: &'static str, n: i32 },

    // Sync point
    Sync,
}

/// Streaming execution context - zero-copy pipeline
pub struct Stream {
    allocator: Arc<TlsfAllocator>,
    handles: HashMap<String, GpuHandle>,
    host_buffers: HashMap<String, Vec<f32>>,
    pool_base: u64,
    initialized: bool,
}

impl Stream {
    pub fn new(allocator: Arc<TlsfAllocator>) -> Self {
        let pool_base = allocator.pool_ptr();
        Stream {
            allocator,
            handles: HashMap::new(),
            host_buffers: HashMap::new(),
            pool_base,
            initialized: false,
        }
    }

    /// Initialize the spcpp backend
    pub fn init(&mut self) -> Result<(), String> {
        if self.initialized {
            return Ok(());
        }

        if !spcpp::load_spcpp() {
            return Err("Failed to load spcpp".to_string());
        }
        spcpp::init("kernels/external/spcpp/ops.cu")?;
        self.initialized = true;
        Ok(())
    }

    /// Get raw GPU pointer for a handle (zero-copy access)
    #[inline(always)]
    fn ptr(&self, name: &str) -> *mut f32 {
        let handle = self.handles.get(name).expect(&format!("Unknown tensor: {}", name));
        (self.pool_base + handle.offset as u64) as *mut f32
    }

    /// Allocate GPU memory (no initialization, no copy)
    pub fn alloc(&mut self, name: &str, num_floats: usize) -> GpuHandle {
        let size = num_floats * 4;
        let offset = self.allocator.alloc(size).expect("TLSF alloc failed");
        let handle = GpuHandle { offset, size };
        self.handles.insert(name.to_string(), handle);
        handle
    }

    /// Stage host data for later transfer (doesn't copy yet)
    pub fn stage_host(&mut self, name: &str, data: Vec<f32>) {
        self.host_buffers.insert(name.to_string(), data);
    }

    /// Transfer staged host data to GPU (only call when needed)
    pub fn upload(&mut self, name: &str) {
        if let Some(data) = self.host_buffers.get(name) {
            let handle = self.handles.get(name).expect("Must alloc before upload");
            self.allocator.copy_to_offset(handle.offset, data);
        }
    }

    /// Download GPU data to host buffer (only call when needed)
    pub fn download(&mut self, name: &str) -> Vec<f32> {
        let handle = self.handles.get(name).expect("Unknown tensor");
        let num_floats = handle.size / 4;
        let mut data = vec![0.0f32; num_floats];
        self.allocator.copy_from_offset(handle.offset, &mut data);
        data
    }

    // =========================================================================
    // ZERO-COPY OPERATIONS (all operate directly on GPU memory)
    // =========================================================================

    /// Matrix multiply: C = A @ B (zero-copy)
    #[inline(always)]
    pub fn matmul(&self, a: &str, b: &str, c: &str, m: i32, k: i32, n: i32) {
        spcpp::matmul(self.ptr(a), self.ptr(b), self.ptr(c), m, k, n);
    }

    /// GEMM: C = alpha * op(A) @ op(B) + beta * C (zero-copy)
    #[inline(always)]
    pub fn gemm(&self, a: &str, b: &str, c: &str, m: i32, k: i32, n: i32,
                alpha: f32, beta: f32, trans_a: bool, trans_b: bool) {
        spcpp::gemm(self.ptr(a), self.ptr(b), self.ptr(c), m, k, n, alpha, beta, trans_a, trans_b);
    }

    /// ReLU in-place (zero-copy)
    #[inline(always)]
    pub fn relu(&self, x: &str, n: i32) {
        spcpp::unary("relu", self.ptr(x), n);
    }

    /// Sigmoid in-place (zero-copy)
    #[inline(always)]
    pub fn sigmoid(&self, x: &str, n: i32) {
        spcpp::unary("sigmoid", self.ptr(x), n);
    }

    /// Tanh in-place (zero-copy)
    #[inline(always)]
    pub fn tanh(&self, x: &str, n: i32) {
        spcpp::unary("tanh", self.ptr(x), n);
    }

    /// GELU in-place (zero-copy)
    #[inline(always)]
    pub fn gelu(&self, x: &str, n: i32) {
        spcpp::unary("gelu", self.ptr(x), n);
    }

    /// SiLU in-place (zero-copy)
    #[inline(always)]
    pub fn silu(&self, x: &str, n: i32) {
        spcpp::unary("silu", self.ptr(x), n);
    }

    /// Softmax in-place (zero-copy)
    #[inline(always)]
    pub fn softmax(&self, x: &str, rows: i32, cols: i32) {
        spcpp::softmax(self.ptr(x), rows, cols);
    }

    /// Scale in-place (zero-copy)
    #[inline(always)]
    pub fn scale(&self, x: &str, scalar: f32, n: i32) {
        spcpp::scale(self.ptr(x), scalar, n);
    }

    /// Fill with value (zero-copy)
    #[inline(always)]
    pub fn fill(&self, x: &str, value: f32, n: i32) {
        spcpp::fill(self.ptr(x), value, n);
    }

    /// Fill at offset within buffer (zero-copy)
    #[inline(always)]
    pub fn fill_at(&self, x: &str, offset: i32, value: f32, n: i32) {
        let ptr = unsafe { self.ptr(x).add(offset as usize) };
        spcpp::fill(ptr, value, n);
    }

    /// Copy: dst = src (zero-copy memcpy on GPU)
    #[inline(always)]
    pub fn copy(&self, src: &str, dst: &str, n: i32) {
        spcpp::copy(self.ptr(src), self.ptr(dst), n);
    }

    /// Add: C = A + B (zero-copy)
    #[inline(always)]
    pub fn add(&self, a: &str, b: &str, c: &str, n: i32) {
        spcpp::binary("add", self.ptr(a), self.ptr(b), self.ptr(c), n);
    }

    /// Mul: C = A * B (zero-copy)
    #[inline(always)]
    pub fn mul(&self, a: &str, b: &str, c: &str, n: i32) {
        spcpp::binary("mul", self.ptr(a), self.ptr(b), self.ptr(c), n);
    }

    /// Linear: output = input @ weights + bias (zero-copy, fused)
    #[inline(always)]
    pub fn linear(&self, input: &str, weights: &str, bias: Option<&str>, output: &str,
                  batch: i32, in_f: i32, out_f: i32) {
        let bias_ptr = bias.map(|b| self.ptr(b)).unwrap_or(std::ptr::null_mut());
        spcpp::linear_forward(
            self.ptr(input), self.ptr(weights), bias_ptr, self.ptr(output),
            batch, in_f, out_f, bias.is_some()
        );
    }

    /// Initialize weights (GPU-side, no host copy)
    #[inline(always)]
    pub fn init_weights(&self, name: &str, rows: i32, cols: i32, init_type: &str) {
        spcpp::init_weights(name, self.ptr(name), rows, cols, init_type);
    }

    /// Adam step in-place (zero-copy)
    #[inline(always)]
    pub fn adam_step(&self, name: &str, weights: &str, grad: &str, n: i32) {
        spcpp::adam_step(name, self.ptr(weights), self.ptr(grad), n);
    }

    /// SGD step in-place (zero-copy)
    #[inline(always)]
    pub fn sgd_step(&self, name: &str, weights: &str, grad: &str, n: i32,
                    lr: f32, momentum: f32, wd: f32) {
        spcpp::sgd_step(name, self.ptr(weights), self.ptr(grad), n, lr, momentum, wd);
    }

    /// Zero gradients in-place (zero-copy)
    #[inline(always)]
    pub fn zero_grad(&self, grad: &str, n: i32) {
        spcpp::zero_grad(self.ptr(grad), n);
    }

    /// Set optimizer params
    pub fn set_optimizer(&self, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32) {
        spcpp::set_optimizer_params(lr, beta1, beta2, eps, wd);
    }

    /// Sync (wait for all GPU ops to complete)
    #[inline(always)]
    pub fn sync(&self) {
        spcpp::sync();
    }

    /// AXPY: y = alpha * x + y (zero-copy)
    #[inline(always)]
    pub fn axpy(&self, x: &str, y: &str, n: i32, alpha: f32) {
        spcpp::axpy(self.ptr(x), self.ptr(y), n, alpha);
    }

    /// Dot product (returns scalar, minimal copy)
    #[inline(always)]
    pub fn dot(&self, x: &str, y: &str, n: i32) -> f32 {
        spcpp::dot(self.ptr(x), self.ptr(y), n)
    }

    /// L2 norm (returns scalar, minimal copy)
    #[inline(always)]
    pub fn nrm2(&self, x: &str, n: i32) -> f32 {
        spcpp::nrm2(self.ptr(x), n)
    }

    /// MSE loss (returns scalar, minimal copy)
    #[inline(always)]
    pub fn mse_loss(&self, pred: &str, target: &str, n: i32) -> f32 {
        spcpp::mse_loss(self.ptr(pred), self.ptr(target), n)
    }

    /// Cross entropy loss (returns scalar)
    #[inline(always)]
    pub fn cross_entropy(&self, logits: &str, labels: *mut i32, batch: i32, classes: i32) -> f32 {
        spcpp::cross_entropy(self.ptr(logits), labels, batch, classes)
    }

    // =========================================================================
    // PIPELINE EXECUTION
    // =========================================================================

    /// Execute a sequence of operations (streaming, zero-copy)
    pub fn execute(&mut self, ops: &[StreamOp]) {
        for op in ops {
            match op {
                StreamOp::Alloc { size, out } => {
                    self.alloc(out, *size);
                }
                StreamOp::Free { name: _ } => {
                    // TLSF doesn't support individual free in current impl
                    // Memory is freed when allocator is dropped
                }
                StreamOp::HostToDevice { name } => {
                    self.upload(name);
                }
                StreamOp::DeviceToHost { name: _ } => {
                    // Handled by explicit download() call
                }
                StreamOp::MatMul { a, b, c, m, k, n } => {
                    self.matmul(a, b, c, *m, *k, *n);
                }
                StreamOp::Gemm { a, b, c, m, k, n, alpha, beta, trans_a, trans_b } => {
                    self.gemm(a, b, c, *m, *k, *n, *alpha, *beta, *trans_a, *trans_b);
                }
                StreamOp::ReLU { x, n } => {
                    self.relu(x, *n);
                }
                StreamOp::Sigmoid { x, n } => {
                    self.sigmoid(x, *n);
                }
                StreamOp::Tanh { x, n } => {
                    self.tanh(x, *n);
                }
                StreamOp::GeLU { x, n } => {
                    self.gelu(x, *n);
                }
                StreamOp::SiLU { x, n } => {
                    self.silu(x, *n);
                }
                StreamOp::Softmax { x, rows, cols } => {
                    self.softmax(x, *rows, *cols);
                }
                StreamOp::Scale { x, scalar, n } => {
                    self.scale(x, *scalar, *n);
                }
                StreamOp::Add { a, b, c, n } => {
                    self.add(a, b, c, *n);
                }
                StreamOp::Mul { a, b, c, n } => {
                    self.mul(a, b, c, *n);
                }
                StreamOp::Linear { input, weights, bias, output, batch, in_f, out_f } => {
                    self.linear(input, weights, *bias, output, *batch, *in_f, *out_f);
                }
                StreamOp::AdamStep { name, weights, grad, n } => {
                    self.adam_step(name, weights, grad, *n);
                }
                StreamOp::SGDStep { name, weights, grad, n, lr, momentum, wd } => {
                    self.sgd_step(name, weights, grad, *n, *lr, *momentum, *wd);
                }
                StreamOp::ZeroGrad { grad, n } => {
                    self.zero_grad(grad, *n);
                }
                StreamOp::Sync => {
                    self.sync();
                }
            }
        }
    }
}

/// Builder for constructing streaming pipelines
pub struct Pipeline {
    ops: Vec<StreamOp>,
}

impl Pipeline {
    pub fn new() -> Self {
        Pipeline { ops: Vec::new() }
    }

    pub fn alloc(mut self, name: &'static str, size: usize) -> Self {
        self.ops.push(StreamOp::Alloc { size, out: name });
        self
    }

    pub fn matmul(mut self, a: &'static str, b: &'static str, c: &'static str, m: i32, k: i32, n: i32) -> Self {
        self.ops.push(StreamOp::MatMul { a, b, c, m, k, n });
        self
    }

    pub fn relu(mut self, x: &'static str, n: i32) -> Self {
        self.ops.push(StreamOp::ReLU { x, n });
        self
    }

    pub fn sigmoid(mut self, x: &'static str, n: i32) -> Self {
        self.ops.push(StreamOp::Sigmoid { x, n });
        self
    }

    pub fn softmax(mut self, x: &'static str, rows: i32, cols: i32) -> Self {
        self.ops.push(StreamOp::Softmax { x, rows, cols });
        self
    }

    pub fn linear(mut self, input: &'static str, weights: &'static str, bias: Option<&'static str>,
                  output: &'static str, batch: i32, in_f: i32, out_f: i32) -> Self {
        self.ops.push(StreamOp::Linear { input, weights, bias, output, batch, in_f, out_f });
        self
    }

    pub fn adam_step(mut self, name: &'static str, weights: &'static str, grad: &'static str, n: i32) -> Self {
        self.ops.push(StreamOp::AdamStep { name, weights, grad, n });
        self
    }

    pub fn sync(mut self) -> Self {
        self.ops.push(StreamOp::Sync);
        self
    }

    pub fn build(self) -> Vec<StreamOp> {
        self.ops
    }

    /// Execute on a stream
    pub fn run(self, stream: &mut Stream) {
        stream.execute(&self.ops);
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;

    #[test]
    #[ignore] // Requires GPU
    fn test_zero_copy_matmul() {
        let device = CudaDevice::new(0).unwrap();
        let allocator = Arc::new(TlsfAllocator::new(device, 64 * 1024 * 1024));

        let mut stream = Stream::new(allocator);
        stream.init().unwrap();

        // Allocate on GPU (no copy)
        stream.alloc("a", 4);
        stream.alloc("b", 4);
        stream.alloc("c", 4);

        // Initialize on GPU (no host involvement)
        stream.fill("a", 1.0, 4);
        stream.fill("b", 2.0, 4);

        // Compute (zero-copy)
        stream.matmul("a", "b", "c", 2, 2, 2);
        stream.sync();

        // Only copy to host when we need the result
        let result = stream.download("c");
        assert!(result.iter().all(|&x| (x - 4.0).abs() < 0.01));
    }
}
