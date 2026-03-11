//! Optimizers — SGD and Adam with fused CUDA kernels.

use crate::tensor::Tensor;
use crate::dtype::DType;
use ptx_runtime::{Result, Error, increment_ops};

/// Stochastic Gradient Descent with optional momentum and weight decay.
pub struct SGD {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    velocities: Vec<Option<Tensor>>,
    initialized: bool,
}

impl SGD {
    pub fn new(lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            velocities: Vec::new(),
            initialized: false,
        }
    }

    /// Perform one optimization step: update params in-place using gradients.
    pub fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(Error::Internal {
                message: format!(
                    "SGD::step: params.len()={} != grads.len()={}",
                    params.len(),
                    grads.len()
                ),
            });
        }

        // Initialize velocity buffers on first call
        if !self.initialized {
            self.velocities.clear();
            for p in params.iter() {
                if self.momentum != 0.0 {
                    self.velocities.push(Some(Tensor::zeros(p.shape(), DType::F32, p.runtime())?));
                } else {
                    self.velocities.push(None);
                }
            }
            self.initialized = true;
        }

        for i in 0..params.len() {
            let p = &params[i];
            let g = &grads[i];

            if p.dtype() != DType::F32 || g.dtype() != DType::F32 {
                return Err(Error::NotSupported {
                    message: "SGD only supports F32 parameters".to_string(),
                });
            }

            let p_c = p.require_contiguous()?;
            let g_c = g.require_contiguous()?;
            let n = p_c.elem_count();

            let vel_ptr = match &self.velocities[i] {
                Some(v) => v.data_ptr_typed::<f32>(),
                None => std::ptr::null_mut(),
            };

            let stream = p_c.runtime().next_stream();
            unsafe {
                ptx_sys::ptx_tensor_sgd_step_f32(
                    p_c.data_ptr_typed::<f32>(),
                    g_c.data_ptr_typed::<f32>(),
                    vel_ptr,
                    n,
                    self.lr,
                    self.momentum,
                    self.weight_decay,
                    stream.raw(),
                );
            }
            increment_ops();
        }

        Ok(())
    }
}

/// Adam optimizer with bias correction.
pub struct Adam {
    pub lr: f32,
    pub betas: (f32, f32),
    pub eps: f32,
    pub weight_decay: f32,
    pub step_count: usize,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    initialized: bool,
}

impl Adam {
    pub fn new(lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            betas,
            eps,
            weight_decay,
            step_count: 0,
            m: Vec::new(),
            v: Vec::new(),
            initialized: false,
        }
    }

    /// Perform one optimization step: update params in-place using gradients.
    pub fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(Error::Internal {
                message: format!(
                    "Adam::step: params.len()={} != grads.len()={}",
                    params.len(),
                    grads.len()
                ),
            });
        }

        // Initialize moment buffers on first call
        if !self.initialized {
            self.m.clear();
            self.v.clear();
            for p in params.iter() {
                self.m.push(Tensor::zeros(p.shape(), DType::F32, p.runtime())?);
                self.v.push(Tensor::zeros(p.shape(), DType::F32, p.runtime())?);
            }
            self.initialized = true;
        }

        self.step_count += 1;
        let (beta1, beta2) = self.betas;

        // Bias correction factors
        let bc1 = 1.0 - beta1.powi(self.step_count as i32);
        let bc2 = 1.0 - beta2.powi(self.step_count as i32);

        for i in 0..params.len() {
            let p = &params[i];
            let g = &grads[i];

            if p.dtype() != DType::F32 || g.dtype() != DType::F32 {
                return Err(Error::NotSupported {
                    message: "Adam only supports F32 parameters".to_string(),
                });
            }

            let p_c = p.require_contiguous()?;
            let g_c = g.require_contiguous()?;
            let n = p_c.elem_count();

            let stream = p_c.runtime().next_stream();
            unsafe {
                ptx_sys::ptx_tensor_adam_step_f32(
                    p_c.data_ptr_typed::<f32>(),
                    g_c.data_ptr_typed::<f32>(),
                    self.m[i].data_ptr_typed::<f32>(),
                    self.v[i].data_ptr_typed::<f32>(),
                    n,
                    self.lr,
                    beta1,
                    beta2,
                    self.eps,
                    self.weight_decay,
                    bc1,
                    bc2,
                    stream.raw(),
                );
            }
            increment_ops();
        }

        Ok(())
    }
}
