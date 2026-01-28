// Device - Unified interface for GPU and CPU backends
//
// Provides automatic device selection and a common API.

use crate::compute::backend::{Backend as GpuBackend, GpuTensor};
use crate::compute::cpu::{CpuBackend, CpuTensor};
use crate::dynamics::allocator::TlsfAllocator;
use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// Device type selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceType {
    Cpu,
    Cuda(usize), // GPU index
}

impl DeviceType {
    /// Auto-select best available device
    pub fn auto() -> Self {
        if CudaDevice::new(0).is_ok() {
            DeviceType::Cuda(0)
        } else {
            DeviceType::Cpu
        }
    }
}

/// Unified tensor that can live on CPU or GPU
pub enum Tensor {
    Cpu(CpuTensor),
    Gpu(GpuTensor),
}

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::Cpu(t) => &t.shape,
            Tensor::Gpu(t) => &t.shape,
        }
    }

    pub fn numel(&self) -> usize {
        match self {
            Tensor::Cpu(t) => t.numel(),
            Tensor::Gpu(t) => t.numel(),
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        match self {
            Tensor::Cpu(t) => t.to_vec(),
            Tensor::Gpu(t) => t.to_host(),
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Tensor::Cpu(_))
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Tensor::Gpu(_))
    }
}

/// Unified device that wraps CPU or GPU backend
pub enum Runtime {
    Cpu(CpuBackend),
    Gpu {
        backend: GpuBackend,
        allocator: Arc<TlsfAllocator>,
    },
}

impl Runtime {
    /// Create CPU device
    pub fn cpu() -> Self {
        Runtime::Cpu(CpuBackend::new())
    }

    /// Create GPU device with specified memory pool size
    pub fn cuda(gpu_index: usize, pool_size_mb: usize) -> Result<Self, String> {
        let cuda_dev = CudaDevice::new(gpu_index)
            .map_err(|e| format!("Failed to init CUDA device {}: {:?}", gpu_index, e))?;

        let pool_bytes = pool_size_mb * 1024 * 1024;
        let allocator = Arc::new(TlsfAllocator::new(cuda_dev, pool_bytes));

        let mut backend = GpuBackend::new(allocator.clone());
        backend.init()?;

        Ok(Runtime::Gpu { backend, allocator })
    }

    /// Auto-select best device
    pub fn auto(pool_size_mb: usize) -> Self {
        match Self::cuda(0, pool_size_mb) {
            Ok(dev) => {
                println!("[ferrite] Using CUDA GPU");
                dev
            }
            Err(_) => {
                println!("[ferrite] Using CPU fallback");
                Self::cpu()
            }
        }
    }

    /// Get device type
    pub fn device_type(&self) -> DeviceType {
        match self {
            Runtime::Cpu(_) => DeviceType::Cpu,
            Runtime::Gpu { .. } => DeviceType::Cuda(0),
        }
    }

    // =========================================================================
    // TENSOR CREATION
    // =========================================================================

    pub fn tensor(&self, shape: Vec<usize>) -> Tensor {
        match self {
            Runtime::Cpu(backend) => Tensor::Cpu(backend.tensor(shape)),
            Runtime::Gpu { backend, .. } => {
                Tensor::Gpu(backend.tensor(shape).expect("GPU alloc failed"))
            }
        }
    }

    pub fn tensor_from_data(&self, data: &[f32], shape: Vec<usize>) -> Tensor {
        match self {
            Runtime::Cpu(backend) => Tensor::Cpu(backend.tensor_from_data(data.to_vec(), shape)),
            Runtime::Gpu { backend, .. } => {
                Tensor::Gpu(backend.tensor_from_data(data, shape).expect("GPU alloc failed"))
            }
        }
    }

    pub fn zeros(&self, shape: Vec<usize>) -> Tensor {
        match self {
            Runtime::Cpu(backend) => Tensor::Cpu(backend.zeros(shape)),
            Runtime::Gpu { backend, .. } => {
                Tensor::Gpu(backend.zeros(shape).expect("GPU alloc failed"))
            }
        }
    }

    pub fn ones(&self, shape: Vec<usize>) -> Tensor {
        match self {
            Runtime::Cpu(backend) => Tensor::Cpu(backend.ones(shape)),
            Runtime::Gpu { backend, .. } => {
                Tensor::Gpu(backend.ones(shape).expect("GPU alloc failed"))
            }
        }
    }

    // =========================================================================
    // MATMUL
    // =========================================================================

    pub fn matmul(&self, a: &Tensor, b: &Tensor, c: &Tensor) {
        match (self, a, b, c) {
            (Runtime::Cpu(backend), Tensor::Cpu(a), Tensor::Cpu(b), Tensor::Cpu(c)) => {
                backend.matmul_blocked(a, b, c);
            }
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(a), Tensor::Gpu(b), Tensor::Gpu(c)) => {
                backend.matmul(a, b, c);
            }
            _ => panic!("Tensor device mismatch"),
        }
    }

    // =========================================================================
    // ACTIVATIONS
    // =========================================================================

    pub fn relu(&self, x: &Tensor) {
        match (self, x) {
            (Runtime::Cpu(backend), Tensor::Cpu(t)) => backend.relu(t),
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(t)) => backend.relu(t),
            _ => panic!("Tensor device mismatch"),
        }
    }

    pub fn sigmoid(&self, x: &Tensor) {
        match (self, x) {
            (Runtime::Cpu(backend), Tensor::Cpu(t)) => backend.sigmoid(t),
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(t)) => backend.sigmoid(t),
            _ => panic!("Tensor device mismatch"),
        }
    }

    pub fn tanh(&self, x: &Tensor) {
        match (self, x) {
            (Runtime::Cpu(backend), Tensor::Cpu(t)) => backend.tanh(t),
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(t)) => backend.tanh(t),
            _ => panic!("Tensor device mismatch"),
        }
    }

    pub fn gelu(&self, x: &Tensor) {
        match (self, x) {
            (Runtime::Cpu(backend), Tensor::Cpu(t)) => backend.gelu(t),
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(t)) => backend.gelu(t),
            _ => panic!("Tensor device mismatch"),
        }
    }

    pub fn softmax(&self, x: &Tensor) {
        match (self, x) {
            (Runtime::Cpu(backend), Tensor::Cpu(t)) => backend.softmax(t),
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(t)) => backend.softmax(t),
            _ => panic!("Tensor device mismatch"),
        }
    }

    // =========================================================================
    // ELEMENT-WISE
    // =========================================================================

    pub fn add(&self, a: &Tensor, b: &Tensor, c: &Tensor) {
        match (self, a, b, c) {
            (Runtime::Cpu(backend), Tensor::Cpu(a), Tensor::Cpu(b), Tensor::Cpu(c)) => {
                backend.add(a, b, c);
            }
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(a), Tensor::Gpu(b), Tensor::Gpu(c)) => {
                backend.add(a, b, c);
            }
            _ => panic!("Tensor device mismatch"),
        }
    }

    pub fn mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) {
        match (self, a, b, c) {
            (Runtime::Cpu(backend), Tensor::Cpu(a), Tensor::Cpu(b), Tensor::Cpu(c)) => {
                backend.mul(a, b, c);
            }
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(a), Tensor::Gpu(b), Tensor::Gpu(c)) => {
                backend.mul(a, b, c);
            }
            _ => panic!("Tensor device mismatch"),
        }
    }

    pub fn scale(&self, x: &Tensor, scalar: f32) {
        match (self, x) {
            (Runtime::Cpu(backend), Tensor::Cpu(t)) => backend.scale(t, scalar),
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(t)) => backend.scale(t, scalar),
            _ => panic!("Tensor device mismatch"),
        }
    }

    pub fn fill(&self, x: &Tensor, value: f32) {
        match (self, x) {
            (Runtime::Cpu(backend), Tensor::Cpu(t)) => backend.fill(t, value),
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(t)) => backend.fill(t, value),
            _ => panic!("Tensor device mismatch"),
        }
    }

    // =========================================================================
    // LINEAR
    // =========================================================================

    pub fn linear(&self, input: &Tensor, weights: &Tensor, bias: Option<&Tensor>, output: &Tensor) {
        match self {
            Runtime::Cpu(backend) => {
                if let (Tensor::Cpu(i), Tensor::Cpu(w), Tensor::Cpu(o)) = (input, weights, output) {
                    let b = bias.map(|t| match t { Tensor::Cpu(b) => b, _ => panic!("bias mismatch") });
                    backend.linear(i, w, b, o);
                }
            }
            Runtime::Gpu { backend, .. } => {
                if let (Tensor::Gpu(i), Tensor::Gpu(w), Tensor::Gpu(o)) = (input, weights, output) {
                    let b = bias.map(|t| match t { Tensor::Gpu(b) => b, _ => panic!("bias mismatch") });
                    backend.linear(i, w, b, o);
                }
            }
        }
    }

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    pub fn init_xavier(&self, name: &str, weights: &Tensor) {
        match (self, weights) {
            (Runtime::Cpu(backend), Tensor::Cpu(t)) => backend.init_xavier(t),
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(t)) => backend.init_xavier(name, t),
            _ => panic!("Tensor device mismatch"),
        }
    }

    pub fn init_kaiming(&self, name: &str, weights: &Tensor) {
        match (self, weights) {
            (Runtime::Cpu(backend), Tensor::Cpu(t)) => backend.init_kaiming(t),
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(t)) => backend.init_kaiming(name, t),
            _ => panic!("Tensor device mismatch"),
        }
    }

    // =========================================================================
    // OPTIMIZER
    // =========================================================================

    pub fn set_optimizer(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        match self {
            Runtime::Cpu(backend) => backend.set_optimizer(lr, beta1, beta2, eps, weight_decay),
            Runtime::Gpu { backend, .. } => backend.set_optimizer(lr, beta1, beta2, eps, weight_decay),
        }
    }

    pub fn adam_step(&mut self, name: &str, weights: &Tensor, grad: &Tensor) {
        match self {
            Runtime::Cpu(backend) => {
                if let (Tensor::Cpu(w), Tensor::Cpu(g)) = (weights, grad) {
                    backend.adam_step(name, w, g);
                }
            }
            Runtime::Gpu { backend, .. } => {
                if let (Tensor::Gpu(w), Tensor::Gpu(g)) = (weights, grad) {
                    backend.adam_step(name, w, g);
                }
            }
        }
    }

    pub fn zero_grad(&self, grad: &Tensor) {
        match (self, grad) {
            (Runtime::Cpu(backend), Tensor::Cpu(t)) => backend.zero_grad(t),
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(t)) => backend.zero_grad(t),
            _ => panic!("Tensor device mismatch"),
        }
    }

    // =========================================================================
    // LOSS
    // =========================================================================

    pub fn mse_loss(&self, pred: &Tensor, target: &Tensor) -> f32 {
        match (self, pred, target) {
            (Runtime::Cpu(backend), Tensor::Cpu(p), Tensor::Cpu(t)) => backend.mse_loss(p, t),
            (Runtime::Gpu { backend, .. }, Tensor::Gpu(p), Tensor::Gpu(t)) => backend.mse_loss(p, t),
            _ => panic!("Tensor device mismatch"),
        }
    }

    // =========================================================================
    // SYNC
    // =========================================================================

    pub fn sync(&self) {
        match self {
            Runtime::Cpu(backend) => backend.sync(),
            Runtime::Gpu { backend, .. } => backend.sync(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device() {
        let device = Runtime::cpu();
        assert_eq!(device.device_type(), DeviceType::Cpu);

        let a = device.tensor_from_data(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = device.tensor_from_data(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let c = device.zeros(vec![2, 2]);

        device.matmul(&a, &b, &c);

        let result = c.to_vec();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[3] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_auto_device() {
        let device = Runtime::auto(64);
        println!("Auto-selected: {:?}", device.device_type());
    }
}
