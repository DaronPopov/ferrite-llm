// CPU Backend - Fallback when no GPU available
//
// Uses OpenBLAS for matmul, native Rust for element-wise ops.
// Not as fast as GPU but fully portable.

use std::sync::{Arc, Mutex};

/// CPU tensor - just a Vec<f32> with shape
#[derive(Debug, Clone)]
pub struct CpuTensor {
    pub data: Arc<Mutex<Vec<f32>>>,
    pub shape: Vec<usize>,
}

impl CpuTensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        CpuTensor {
            data: Arc::new(Mutex::new(vec![0.0; numel])),
            shape,
        }
    }

    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "Data length must match shape");
        CpuTensor {
            data: Arc::new(Mutex::new(data)),
            shape,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.lock().unwrap().clone()
    }

    pub fn from_slice(&self, data: &[f32]) {
        let mut guard = self.data.lock().unwrap();
        guard.copy_from_slice(data);
    }
}

/// CPU Backend
pub struct CpuBackend {
    // Optimizer state
    momentum: std::collections::HashMap<String, Vec<f32>>,
    adam_m: std::collections::HashMap<String, Vec<f32>>,
    adam_v: std::collections::HashMap<String, Vec<f32>>,
    adam_t: i32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend {
            momentum: std::collections::HashMap::new(),
            adam_m: std::collections::HashMap::new(),
            adam_v: std::collections::HashMap::new(),
            adam_t: 0,
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }

    // =========================================================================
    // TENSOR CREATION
    // =========================================================================

    pub fn tensor(&self, shape: Vec<usize>) -> CpuTensor {
        CpuTensor::new(shape)
    }

    pub fn tensor_from_data(&self, data: Vec<f32>, shape: Vec<usize>) -> CpuTensor {
        CpuTensor::from_data(data, shape)
    }

    pub fn zeros(&self, shape: Vec<usize>) -> CpuTensor {
        CpuTensor::new(shape)
    }

    pub fn ones(&self, shape: Vec<usize>) -> CpuTensor {
        let numel: usize = shape.iter().product();
        CpuTensor::from_data(vec![1.0; numel], shape)
    }

    // =========================================================================
    // MATMUL - Uses OpenBLAS via cblas if available, fallback to naive
    // =========================================================================

    /// Matrix multiplication: C = A @ B
    /// A: [m, k], B: [k, n] -> C: [m, n]
    pub fn matmul(&self, a: &CpuTensor, b: &CpuTensor, c: &CpuTensor) {
        assert_eq!(a.shape.len(), 2);
        assert_eq!(b.shape.len(), 2);
        assert_eq!(a.shape[1], b.shape[0], "Inner dimensions must match");

        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];

        let a_data = a.data.lock().unwrap();
        let b_data = b.data.lock().unwrap();
        let mut c_data = c.data.lock().unwrap();

        // Naive matmul (TODO: link OpenBLAS for performance)
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a_data[i * k + p] * b_data[p * n + j];
                }
                c_data[i * n + j] = sum;
            }
        }
    }

    /// Optimized matmul with cache blocking
    pub fn matmul_blocked(&self, a: &CpuTensor, b: &CpuTensor, c: &CpuTensor) {
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];

        let a_data = a.data.lock().unwrap();
        let b_data = b.data.lock().unwrap();
        let mut c_data = c.data.lock().unwrap();

        // Zero output
        c_data.fill(0.0);

        // Block size tuned for L1 cache (~32KB)
        const BLOCK: usize = 64;

        for ii in (0..m).step_by(BLOCK) {
            for jj in (0..n).step_by(BLOCK) {
                for kk in (0..k).step_by(BLOCK) {
                    let i_end = (ii + BLOCK).min(m);
                    let j_end = (jj + BLOCK).min(n);
                    let k_end = (kk + BLOCK).min(k);

                    for i in ii..i_end {
                        for p in kk..k_end {
                            let a_val = a_data[i * k + p];
                            for j in jj..j_end {
                                c_data[i * n + j] += a_val * b_data[p * n + j];
                            }
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // ACTIVATIONS
    // =========================================================================

    pub fn relu(&self, x: &CpuTensor) {
        let mut data = x.data.lock().unwrap();
        for v in data.iter_mut() {
            *v = v.max(0.0);
        }
    }

    pub fn sigmoid(&self, x: &CpuTensor) {
        let mut data = x.data.lock().unwrap();
        for v in data.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    }

    pub fn tanh(&self, x: &CpuTensor) {
        let mut data = x.data.lock().unwrap();
        for v in data.iter_mut() {
            *v = v.tanh();
        }
    }

    pub fn gelu(&self, x: &CpuTensor) {
        let mut data = x.data.lock().unwrap();
        for v in data.iter_mut() {
            // GELU approximation
            *v = 0.5 * *v * (1.0 + (0.7978845608 * (*v + 0.044715 * *v * *v * *v)).tanh());
        }
    }

    pub fn silu(&self, x: &CpuTensor) {
        let mut data = x.data.lock().unwrap();
        for v in data.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
    }

    pub fn leaky_relu(&self, x: &CpuTensor, alpha: f32) {
        let mut data = x.data.lock().unwrap();
        for v in data.iter_mut() {
            *v = if *v > 0.0 { *v } else { alpha * *v };
        }
    }

    pub fn softmax(&self, x: &CpuTensor) {
        let mut data = x.data.lock().unwrap();
        let rows = x.shape[0];
        let cols = x.shape[1];

        for i in 0..rows {
            let row_start = i * cols;
            let row_end = row_start + cols;
            let row = &mut data[row_start..row_end];

            // Numerical stability: subtract max
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }

    // =========================================================================
    // ELEMENT-WISE MATH
    // =========================================================================

    pub fn add(&self, a: &CpuTensor, b: &CpuTensor, c: &CpuTensor) {
        let a_data = a.data.lock().unwrap();
        let b_data = b.data.lock().unwrap();
        let mut c_data = c.data.lock().unwrap();
        for i in 0..a_data.len() {
            c_data[i] = a_data[i] + b_data[i];
        }
    }

    pub fn sub(&self, a: &CpuTensor, b: &CpuTensor, c: &CpuTensor) {
        let a_data = a.data.lock().unwrap();
        let b_data = b.data.lock().unwrap();
        let mut c_data = c.data.lock().unwrap();
        for i in 0..a_data.len() {
            c_data[i] = a_data[i] - b_data[i];
        }
    }

    pub fn mul(&self, a: &CpuTensor, b: &CpuTensor, c: &CpuTensor) {
        let a_data = a.data.lock().unwrap();
        let b_data = b.data.lock().unwrap();
        let mut c_data = c.data.lock().unwrap();
        for i in 0..a_data.len() {
            c_data[i] = a_data[i] * b_data[i];
        }
    }

    pub fn div(&self, a: &CpuTensor, b: &CpuTensor, c: &CpuTensor) {
        let a_data = a.data.lock().unwrap();
        let b_data = b.data.lock().unwrap();
        let mut c_data = c.data.lock().unwrap();
        for i in 0..a_data.len() {
            c_data[i] = a_data[i] / b_data[i];
        }
    }

    pub fn scale(&self, x: &CpuTensor, scalar: f32) {
        let mut data = x.data.lock().unwrap();
        for v in data.iter_mut() {
            *v *= scalar;
        }
    }

    pub fn fill(&self, x: &CpuTensor, value: f32) {
        let mut data = x.data.lock().unwrap();
        data.fill(value);
    }

    pub fn exp(&self, x: &CpuTensor) {
        let mut data = x.data.lock().unwrap();
        for v in data.iter_mut() {
            *v = v.exp();
        }
    }

    pub fn log(&self, x: &CpuTensor) {
        let mut data = x.data.lock().unwrap();
        for v in data.iter_mut() {
            *v = v.ln();
        }
    }

    pub fn sqrt(&self, x: &CpuTensor) {
        let mut data = x.data.lock().unwrap();
        for v in data.iter_mut() {
            *v = v.sqrt();
        }
    }

    pub fn neg(&self, x: &CpuTensor) {
        let mut data = x.data.lock().unwrap();
        for v in data.iter_mut() {
            *v = -*v;
        }
    }

    pub fn abs(&self, x: &CpuTensor) {
        let mut data = x.data.lock().unwrap();
        for v in data.iter_mut() {
            *v = v.abs();
        }
    }

    // =========================================================================
    // LINEAR LAYER
    // =========================================================================

    pub fn linear(&self, input: &CpuTensor, weights: &CpuTensor,
                  bias: Option<&CpuTensor>, output: &CpuTensor) {
        // output = input @ weights + bias
        self.matmul_blocked(input, weights, output);

        if let Some(b) = bias {
            let bias_data = b.data.lock().unwrap();
            let mut out_data = output.data.lock().unwrap();
            let batch = input.shape[0];
            let out_features = weights.shape[1];

            for i in 0..batch {
                for j in 0..out_features {
                    out_data[i * out_features + j] += bias_data[j];
                }
            }
        }
    }

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    pub fn init_xavier(&self, weights: &CpuTensor) {
        let rows = weights.shape[0];
        let cols = weights.shape[1];
        let limit = (6.0 / (rows + cols) as f32).sqrt();

        let mut data = weights.data.lock().unwrap();
        let mut rng = SimpleRng::new(42);
        for v in data.iter_mut() {
            *v = rng.uniform(-limit, limit);
        }
    }

    pub fn init_kaiming(&self, weights: &CpuTensor) {
        let rows = weights.shape[0];
        let std = (2.0 / rows as f32).sqrt();

        let mut data = weights.data.lock().unwrap();
        let mut rng = SimpleRng::new(42);
        for v in data.iter_mut() {
            *v = rng.normal(0.0, std);
        }
    }

    // =========================================================================
    // OPTIMIZER
    // =========================================================================

    pub fn set_optimizer(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) {
        self.lr = lr;
        self.beta1 = beta1;
        self.beta2 = beta2;
        self.eps = eps;
        self.weight_decay = weight_decay;
    }

    pub fn sgd_step(&mut self, name: &str, weights: &CpuTensor, grad: &CpuTensor,
                    lr: f32, momentum: f32, weight_decay: f32) {
        let n = weights.numel();
        let key = name.to_string();

        if !self.momentum.contains_key(&key) {
            self.momentum.insert(key.clone(), vec![0.0; n]);
        }

        let mom = self.momentum.get_mut(&key).unwrap();
        let mut w_data = weights.data.lock().unwrap();
        let g_data = grad.data.lock().unwrap();

        for i in 0..n {
            let g = g_data[i] + weight_decay * w_data[i];
            mom[i] = momentum * mom[i] + g;
            w_data[i] -= lr * mom[i];
        }
    }

    pub fn adam_step(&mut self, name: &str, weights: &CpuTensor, grad: &CpuTensor) {
        let n = weights.numel();
        let key = name.to_string();

        if !self.adam_m.contains_key(&key) {
            self.adam_m.insert(key.clone(), vec![0.0; n]);
            self.adam_v.insert(key.clone(), vec![0.0; n]);
        }

        self.adam_t += 1;
        let t = self.adam_t as f32;

        let m = self.adam_m.get_mut(&key).unwrap();
        let v = self.adam_v.get_mut(&key).unwrap();
        let mut w_data = weights.data.lock().unwrap();
        let g_data = grad.data.lock().unwrap();

        for i in 0..n {
            let g = g_data[i] + self.weight_decay * w_data[i];
            m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;
            v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = m[i] / (1.0 - self.beta1.powf(t));
            let v_hat = v[i] / (1.0 - self.beta2.powf(t));

            w_data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    pub fn zero_grad(&self, grad: &CpuTensor) {
        let mut data = grad.data.lock().unwrap();
        data.fill(0.0);
    }

    // =========================================================================
    // LOSS
    // =========================================================================

    pub fn mse_loss(&self, pred: &CpuTensor, target: &CpuTensor) -> f32 {
        let pred_data = pred.data.lock().unwrap();
        let target_data = target.data.lock().unwrap();
        let n = pred_data.len();

        let mut sum = 0.0;
        for i in 0..n {
            let diff = pred_data[i] - target_data[i];
            sum += diff * diff;
        }
        sum / n as f32
    }

    pub fn cross_entropy(&self, logits: &CpuTensor, labels: &[i32]) -> f32 {
        let data = logits.data.lock().unwrap();
        let batch = logits.shape[0];
        let classes = logits.shape[1];

        let mut total_loss = 0.0;
        for i in 0..batch {
            let row_start = i * classes;
            let row = &data[row_start..row_start + classes];

            // Softmax + cross entropy
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|x| (x - max_val).exp()).sum();
            let log_prob = (row[labels[i] as usize] - max_val) - sum.ln();
            total_loss -= log_prob;
        }
        total_loss / batch as f32
    }

    // =========================================================================
    // BLAS-LIKE
    // =========================================================================

    pub fn axpy(&self, x: &CpuTensor, y: &CpuTensor, alpha: f32) {
        let x_data = x.data.lock().unwrap();
        let mut y_data = y.data.lock().unwrap();
        for i in 0..x_data.len() {
            y_data[i] += alpha * x_data[i];
        }
    }

    pub fn dot(&self, x: &CpuTensor, y: &CpuTensor) -> f32 {
        let x_data = x.data.lock().unwrap();
        let y_data = y.data.lock().unwrap();
        x_data.iter().zip(y_data.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn nrm2(&self, x: &CpuTensor) -> f32 {
        let data = x.data.lock().unwrap();
        data.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    // No sync needed on CPU
    pub fn sync(&self) {}
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

// Simple RNG for initialization (no external deps)
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        let r = (self.next() as f64) / (u64::MAX as f64);
        low + (high - low) * r as f32
    }

    fn normal(&mut self, mean: f32, std: f32) -> f32 {
        // Box-Muller transform
        let u1 = (self.next() as f64 / u64::MAX as f64).max(1e-10);
        let u2 = self.next() as f64 / u64::MAX as f64;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_tensor() {
        let t = CpuTensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.numel(), 4);
    }

    #[test]
    fn test_cpu_matmul() {
        let backend = CpuBackend::new();

        // 2x2 identity matmul
        let a = backend.tensor_from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = backend.tensor_from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let c = backend.zeros(vec![2, 2]);

        backend.matmul(&a, &b, &c);

        let result = c.to_vec();
        assert!((result[0] - 1.0).abs() < 0.001);
        assert!((result[1] - 2.0).abs() < 0.001);
        assert!((result[2] - 3.0).abs() < 0.001);
        assert!((result[3] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_cpu_relu() {
        let backend = CpuBackend::new();
        let t = backend.tensor_from_data(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        backend.relu(&t);

        let result = t.to_vec();
        assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_cpu_softmax() {
        let backend = CpuBackend::new();
        let t = backend.tensor_from_data(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]);
        backend.softmax(&t);

        let result = t.to_vec();
        // Each row should sum to 1
        let sum1: f32 = result[0..3].iter().sum();
        let sum2: f32 = result[3..6].iter().sum();
        assert!((sum1 - 1.0).abs() < 0.001);
        assert!((sum2 - 1.0).abs() < 0.001);
    }
}
