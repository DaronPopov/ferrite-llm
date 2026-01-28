// Backend - Primary GPU execution engine using spcpp/cuBLAS
//
// Rust orchestrates memory safety, spcpp provides raw GPU performance.
// All GPU memory is managed through the TLSF allocator for O(1) alloc/free.

use crate::compute::spcpp;
use crate::dynamics::allocator::TlsfAllocator;
use crate::dynamics::Precision;
use std::sync::Arc;

/// GPU tensor handle - memory-safe wrapper around device pointer
#[derive(Debug)]
pub struct GpuTensor {
    /// Offset into the TLSF pool
    pub offset: usize,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Precision type
    pub precision: Precision,
    /// Reference to allocator (keeps memory valid)
    allocator: Arc<TlsfAllocator>,
}

impl GpuTensor {
    /// Create a new GPU tensor with given shape
    pub fn new(allocator: Arc<TlsfAllocator>, shape: Vec<usize>, precision: Precision) -> Option<Self> {
        let numel: usize = shape.iter().product();
        let bytes = precision.storage_bytes(numel);
        let offset = allocator.alloc(bytes)?;

        Some(GpuTensor {
            offset,
            shape,
            precision,
            allocator,
        })
    }

    /// Create tensor from host data
    pub fn from_data(allocator: Arc<TlsfAllocator>, data: &[f32], shape: Vec<usize>) -> Option<Self> {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "Data length must match shape");

        let bytes = numel * 4; // f32
        let offset = allocator.alloc(bytes)?;
        allocator.copy_to_offset(offset, data);

        Some(GpuTensor {
            offset,
            shape,
            precision: Precision::F32,
            allocator,
        })
    }

    /// Get raw device pointer
    pub fn ptr(&self) -> *mut f32 {
        (self.allocator.pool_ptr() + self.offset as u64) as *mut f32
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.precision.storage_bytes(self.numel())
    }

    /// Copy data to host
    pub fn to_host(&self) -> Vec<f32> {
        let mut data = vec![0.0f32; self.numel()];
        self.allocator.copy_from_offset(self.offset, &mut data);
        data
    }

    /// Copy data from host
    pub fn from_host(&self, data: &[f32]) {
        assert_eq!(data.len(), self.numel());
        self.allocator.copy_to_offset(self.offset, data);
    }

    /// Get a reference to the allocator
    pub fn allocator(&self) -> &Arc<TlsfAllocator> {
        &self.allocator
    }
}

// No Drop impl - memory stays allocated until pool is destroyed
// This is intentional for performance (avoid fragmentation)

/// Backend execution context
pub struct Backend {
    allocator: Arc<TlsfAllocator>,
    initialized: bool,
}

impl Backend {
    /// Create a new backend with the given allocator
    pub fn new(allocator: Arc<TlsfAllocator>) -> Self {
        Backend {
            allocator,
            initialized: false,
        }
    }

    /// Initialize the spcpp backend
    pub fn init(&mut self) -> Result<(), String> {
        if self.initialized {
            return Ok(());
        }

        // Load spcpp library
        if !spcpp::load_spcpp() {
            return Err("Failed to load spcpp library. Run ./build_spcpp.sh first.".to_string());
        }

        // Initialize with kernel path
        let kernel_path = "kernels/external/spcpp/ops.cu";
        spcpp::init(kernel_path)?;

        self.initialized = true;
        Ok(())
    }

    /// Check if backend is ready
    pub fn is_ready(&self) -> bool {
        self.initialized
    }

    /// Get the allocator
    pub fn allocator(&self) -> &Arc<TlsfAllocator> {
        &self.allocator
    }

    // =========================================================================
    // TENSOR OPERATIONS
    // =========================================================================

    /// Create a new tensor
    pub fn tensor(&self, shape: Vec<usize>) -> Option<GpuTensor> {
        GpuTensor::new(self.allocator.clone(), shape, Precision::F32)
    }

    /// Create tensor from data
    pub fn tensor_from_data(&self, data: &[f32], shape: Vec<usize>) -> Option<GpuTensor> {
        GpuTensor::from_data(self.allocator.clone(), data, shape)
    }

    /// Create zeros tensor
    pub fn zeros(&self, shape: Vec<usize>) -> Option<GpuTensor> {
        let t = self.tensor(shape)?;
        spcpp::fill(t.ptr(), 0.0, t.numel() as i32);
        Some(t)
    }

    /// Create ones tensor
    pub fn ones(&self, shape: Vec<usize>) -> Option<GpuTensor> {
        let t = self.tensor(shape)?;
        spcpp::fill(t.ptr(), 1.0, t.numel() as i32);
        Some(t)
    }

    // =========================================================================
    // MATMUL
    // =========================================================================

    /// Matrix multiplication: C = A @ B
    /// A: [m, k], B: [k, n] -> C: [m, n]
    pub fn matmul(&self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) {
        assert_eq!(a.shape.len(), 2);
        assert_eq!(b.shape.len(), 2);
        assert_eq!(c.shape.len(), 2);
        assert_eq!(a.shape[1], b.shape[0], "Inner dimensions must match");
        assert_eq!(c.shape[0], a.shape[0]);
        assert_eq!(c.shape[1], b.shape[1]);

        let m = a.shape[0] as i32;
        let k = a.shape[1] as i32;
        let n = b.shape[1] as i32;

        spcpp::matmul(a.ptr(), b.ptr(), c.ptr(), m, k, n);
    }

    /// General matrix multiply: C = alpha * op(A) @ op(B) + beta * C
    pub fn gemm(&self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor,
                alpha: f32, beta: f32, trans_a: bool, trans_b: bool) {
        let m = if trans_a { a.shape[1] } else { a.shape[0] } as i32;
        let k = if trans_a { a.shape[0] } else { a.shape[1] } as i32;
        let n = if trans_b { b.shape[0] } else { b.shape[1] } as i32;

        spcpp::gemm(a.ptr(), b.ptr(), c.ptr(), m, k, n, alpha, beta, trans_a, trans_b);
    }

    // =========================================================================
    // ACTIVATIONS
    // =========================================================================

    /// ReLU in-place
    pub fn relu(&self, x: &GpuTensor) {
        spcpp::unary("relu", x.ptr(), x.numel() as i32);
    }

    /// Sigmoid in-place
    pub fn sigmoid(&self, x: &GpuTensor) {
        spcpp::unary("sigmoid", x.ptr(), x.numel() as i32);
    }

    /// Tanh in-place
    pub fn tanh(&self, x: &GpuTensor) {
        spcpp::unary("tanh", x.ptr(), x.numel() as i32);
    }

    /// GELU in-place
    pub fn gelu(&self, x: &GpuTensor) {
        spcpp::unary("gelu", x.ptr(), x.numel() as i32);
    }

    /// SiLU/Swish in-place
    pub fn silu(&self, x: &GpuTensor) {
        spcpp::unary("silu", x.ptr(), x.numel() as i32);
    }

    /// Softmax over last dimension
    pub fn softmax(&self, x: &GpuTensor) {
        let rows = x.shape[..x.shape.len()-1].iter().product::<usize>() as i32;
        let cols = *x.shape.last().unwrap() as i32;
        spcpp::softmax(x.ptr(), rows, cols);
    }

    /// Leaky ReLU in-place
    pub fn leaky_relu(&self, x: &GpuTensor, alpha: f32) {
        spcpp::leaky_relu(x.ptr(), x.numel() as i32, alpha);
    }

    // =========================================================================
    // ELEMENT-WISE MATH
    // =========================================================================

    /// Element-wise add: c = a + b
    pub fn add(&self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) {
        assert_eq!(a.numel(), b.numel());
        assert_eq!(a.numel(), c.numel());
        spcpp::binary("add", a.ptr(), b.ptr(), c.ptr(), a.numel() as i32);
    }

    /// Element-wise subtract: c = a - b
    pub fn sub(&self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) {
        assert_eq!(a.numel(), b.numel());
        assert_eq!(a.numel(), c.numel());
        spcpp::binary("sub", a.ptr(), b.ptr(), c.ptr(), a.numel() as i32);
    }

    /// Element-wise multiply: c = a * b
    pub fn mul(&self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) {
        assert_eq!(a.numel(), b.numel());
        assert_eq!(a.numel(), c.numel());
        spcpp::binary("mul", a.ptr(), b.ptr(), c.ptr(), a.numel() as i32);
    }

    /// Element-wise divide: c = a / b
    pub fn div(&self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) {
        assert_eq!(a.numel(), b.numel());
        assert_eq!(a.numel(), c.numel());
        spcpp::binary("div", a.ptr(), b.ptr(), c.ptr(), a.numel() as i32);
    }

    /// Scale tensor in-place: x = x * scalar
    pub fn scale(&self, x: &GpuTensor, scalar: f32) {
        spcpp::scale(x.ptr(), scalar, x.numel() as i32);
    }

    /// Fill tensor with value
    pub fn fill(&self, x: &GpuTensor, value: f32) {
        spcpp::fill(x.ptr(), value, x.numel() as i32);
    }

    /// Exp in-place
    pub fn exp(&self, x: &GpuTensor) {
        spcpp::unary("exp", x.ptr(), x.numel() as i32);
    }

    /// Log in-place
    pub fn log(&self, x: &GpuTensor) {
        spcpp::unary("log", x.ptr(), x.numel() as i32);
    }

    /// Sqrt in-place
    pub fn sqrt(&self, x: &GpuTensor) {
        spcpp::unary("sqrt", x.ptr(), x.numel() as i32);
    }

    /// Negate in-place
    pub fn neg(&self, x: &GpuTensor) {
        spcpp::unary("neg", x.ptr(), x.numel() as i32);
    }

    /// Absolute value in-place
    pub fn abs(&self, x: &GpuTensor) {
        spcpp::unary("abs", x.ptr(), x.numel() as i32);
    }

    // =========================================================================
    // LINEAR LAYERS
    // =========================================================================

    /// Linear forward: output = input @ weights + bias
    pub fn linear(&self, input: &GpuTensor, weights: &GpuTensor,
                  bias: Option<&GpuTensor>, output: &GpuTensor) {
        let batch = input.shape[0] as i32;
        let in_features = input.shape[1] as i32;
        let out_features = weights.shape[1] as i32;

        let bias_ptr = bias.map(|b| b.ptr()).unwrap_or(std::ptr::null_mut());
        let has_bias = bias.is_some();

        spcpp::linear_forward(
            input.ptr(), weights.ptr(), bias_ptr, output.ptr(),
            batch, in_features, out_features, has_bias
        );
    }

    /// Linear backward pass
    pub fn linear_backward(&self, input: &GpuTensor, weights: &GpuTensor,
                           grad_output: &GpuTensor,
                           grad_input: Option<&GpuTensor>,
                           grad_weights: Option<&GpuTensor>,
                           grad_bias: Option<&GpuTensor>) {
        let batch = input.shape[0] as i32;
        let in_f = input.shape[1] as i32;
        let out_f = weights.shape[1] as i32;

        let gi_ptr = grad_input.map(|t| t.ptr()).unwrap_or(std::ptr::null_mut());
        let gw_ptr = grad_weights.map(|t| t.ptr()).unwrap_or(std::ptr::null_mut());
        let gb_ptr = grad_bias.map(|t| t.ptr()).unwrap_or(std::ptr::null_mut());

        spcpp::linear_backward(
            input.ptr(), weights.ptr(), grad_output.ptr(),
            gi_ptr, gw_ptr, gb_ptr,
            batch, in_f, out_f, grad_bias.is_some()
        );
    }

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    /// Initialize weights with Xavier/Glorot
    pub fn init_xavier(&self, name: &str, weights: &GpuTensor) {
        let rows = weights.shape[0] as i32;
        let cols = weights.shape[1] as i32;
        spcpp::init_weights(name, weights.ptr(), rows, cols, "xavier");
    }

    /// Initialize weights with Kaiming/He
    pub fn init_kaiming(&self, name: &str, weights: &GpuTensor) {
        let rows = weights.shape[0] as i32;
        let cols = weights.shape[1] as i32;
        spcpp::init_weights(name, weights.ptr(), rows, cols, "kaiming");
    }

    /// Initialize with normal distribution
    pub fn init_normal(&self, name: &str, weights: &GpuTensor) {
        let rows = weights.shape[0] as i32;
        let cols = weights.shape[1] as i32;
        spcpp::init_weights(name, weights.ptr(), rows, cols, "normal");
    }

    // =========================================================================
    // OPTIMIZER
    // =========================================================================

    /// Set optimizer hyperparameters
    pub fn set_optimizer(&self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        spcpp::set_optimizer_params(lr, beta1, beta2, eps, weight_decay);
    }

    /// SGD step with momentum
    pub fn sgd_step(&self, name: &str, weights: &GpuTensor, grad: &GpuTensor,
                    lr: f32, momentum: f32, weight_decay: f32) {
        spcpp::sgd_step(name, weights.ptr(), grad.ptr(), weights.numel() as i32,
                        lr, momentum, weight_decay);
    }

    /// Adam step
    pub fn adam_step(&self, name: &str, weights: &GpuTensor, grad: &GpuTensor) {
        spcpp::adam_step(name, weights.ptr(), grad.ptr(), weights.numel() as i32);
    }

    /// Zero gradients
    pub fn zero_grad(&self, grad: &GpuTensor) {
        spcpp::zero_grad(grad.ptr(), grad.numel() as i32);
    }

    // =========================================================================
    // NORMALIZATION
    // =========================================================================

    /// Layer normalization
    pub fn layer_norm(&self, x: &GpuTensor, gamma: &GpuTensor, beta: &GpuTensor,
                      out: &GpuTensor, eps: f32) {
        let batch = x.shape[0] as i32;
        let dim = x.shape[1] as i32;
        spcpp::layer_norm(x.ptr(), gamma.ptr(), beta.ptr(), out.ptr(), batch, dim, eps);
    }

    /// Dropout (training mode)
    pub fn dropout(&self, x: &GpuTensor, mask: &GpuTensor, p: f32, seed: u32) {
        spcpp::dropout(x.ptr(), mask.ptr(), x.numel() as i32, p, seed);
    }

    // =========================================================================
    // EMBEDDING
    // =========================================================================

    /// Embedding lookup
    pub fn embedding(&self, table: &GpuTensor, indices: *mut i32, out: &GpuTensor, batch: i32) {
        let dim = table.shape[1] as i32;
        spcpp::embedding(table.ptr(), indices, out.ptr(), batch, dim);
    }

    // =========================================================================
    // LOSS FUNCTIONS
    // =========================================================================

    /// Cross entropy loss
    pub fn cross_entropy(&self, logits: &GpuTensor, labels: *mut i32) -> f32 {
        let batch = logits.shape[0] as i32;
        let classes = logits.shape[1] as i32;
        spcpp::cross_entropy(logits.ptr(), labels, batch, classes)
    }

    /// MSE loss
    pub fn mse_loss(&self, pred: &GpuTensor, target: &GpuTensor) -> f32 {
        assert_eq!(pred.numel(), target.numel());
        spcpp::mse_loss(pred.ptr(), target.ptr(), pred.numel() as i32)
    }

    // =========================================================================
    // BLAS PRIMITIVES
    // =========================================================================

    /// y = alpha * x + y (AXPY)
    pub fn axpy(&self, x: &GpuTensor, y: &GpuTensor, alpha: f32) {
        assert_eq!(x.numel(), y.numel());
        spcpp::axpy(x.ptr(), y.ptr(), x.numel() as i32, alpha);
    }

    /// x = alpha * x (SCAL)
    pub fn scal(&self, x: &GpuTensor, alpha: f32) {
        spcpp::scal(x.ptr(), x.numel() as i32, alpha);
    }

    /// Dot product
    pub fn dot(&self, x: &GpuTensor, y: &GpuTensor) -> f32 {
        assert_eq!(x.numel(), y.numel());
        spcpp::dot(x.ptr(), y.ptr(), x.numel() as i32)
    }

    /// L2 norm
    pub fn nrm2(&self, x: &GpuTensor) -> f32 {
        spcpp::nrm2(x.ptr(), x.numel() as i32)
    }

    // =========================================================================
    // SYNC
    // =========================================================================

    /// Synchronize GPU
    pub fn sync(&self) {
        spcpp::sync();
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        if self.initialized {
            spcpp::shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaDevice;

    fn create_test_backend() -> Option<Backend> {
        let device = CudaDevice::new(0).ok()?;
        let allocator = Arc::new(TlsfAllocator::new(device, 64 * 1024 * 1024));
        let mut backend = Backend::new(allocator);
        backend.init().ok()?;
        Some(backend)
    }

    #[test]
    #[ignore] // Requires GPU and spcpp
    fn test_backend_init() {
        let backend = create_test_backend();
        assert!(backend.is_some());
    }

    #[test]
    #[ignore] // Requires GPU and spcpp
    fn test_tensor_creation() {
        let backend = create_test_backend().unwrap();
        let t = backend.tensor(vec![2, 3]).unwrap();
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    #[ignore] // Requires GPU and spcpp
    fn test_matmul() {
        let backend = create_test_backend().unwrap();

        let a = backend.tensor_from_data(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = backend.tensor_from_data(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c = backend.zeros(vec![2, 2]).unwrap();

        backend.matmul(&a, &b, &c);
        backend.sync();

        let result = c.to_host();
        // A @ I = A
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
        assert!((result[2] - 3.0).abs() < 0.01);
        assert!((result[3] - 4.0).abs() < 0.01);
    }
}
