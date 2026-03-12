//! CPU compute kernels for host-fallback operations.
//!
//! When the `torch` feature is enabled, these use libtorch CPU ops
//! (MKL/OpenBLAS matmul, vectorized reductions, SIMD comparisons).
//! Without `torch`, they use simple Rust implementations.
//!
//! All functions take `&[f32]` inputs and write into `&mut [f32]` outputs.
//! The caller manages buffer allocation (via CPU TLSF) and GPU transfers.

// ── Torch-backed implementations ────────────────────────────────────

#[cfg(feature = "torch")]
mod imp {
    use tch::{Device, Kind, Tensor};

    /// Wrap a host f32 slice as a zero-copy tch CPU tensor.
    ///
    /// # Safety
    ///
    /// The returned tensor borrows the data — the slice must outlive it.
    unsafe fn wrap(data: &[f32], shape: &[i64]) -> Tensor {
        Tensor::from_blob(
            data.as_ptr() as *const u8,
            shape,
            &[],         // default contiguous strides
            Kind::Float,
            Device::Cpu,
        )
    }

    /// Copy a tch tensor's data into an output slice.
    fn extract(t: &Tensor, out: &mut [f32]) {
        let n = t.numel() as usize;
        assert!(
            out.len() >= n,
            "cpu_kernels: output buffer too small ({} < {})",
            out.len(),
            n
        );
        if n > 0 {
            t.copy_data(out, n);
        }
    }

    // ── matmul ──

    pub fn matmul(lhs: &[f32], rhs: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        let l = unsafe { wrap(lhs, &[m as i64, k as i64]) };
        let r = unsafe { wrap(rhs, &[k as i64, n as i64]) };
        let result = l.matmul(&r);
        extract(&result, out);
    }

    // ── reductions ──

    pub fn reduce_sum(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        let s: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let t = unsafe { wrap(inp, &s) };
        let result = t.sum_dim_intlist([dim as i64].as_slice(), false, Kind::Float);
        extract(&result, out);
    }

    pub fn reduce_mean(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        let s: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let t = unsafe { wrap(inp, &s) };
        let result = t.mean_dim([dim as i64].as_slice(), false, Kind::Float);
        extract(&result, out);
    }

    pub fn reduce_max(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        let s: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let t = unsafe { wrap(inp, &s) };
        let (values, _indices) = t.max_dim(dim as i64, false);
        extract(&values, out);
    }

    pub fn reduce_min(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        let s: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let t = unsafe { wrap(inp, &s) };
        let (values, _indices) = t.min_dim(dim as i64, false);
        extract(&values, out);
    }

    pub fn argmax(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        let s: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let t = unsafe { wrap(inp, &s) };
        let result = t.argmax(dim as i64, false).to_kind(Kind::Float);
        extract(&result, out);
    }

    pub fn argmin(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        let s: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let t = unsafe { wrap(inp, &s) };
        let result = t.argmin(dim as i64, false).to_kind(Kind::Float);
        extract(&result, out);
    }

    pub fn softmax(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        let s: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let t = unsafe { wrap(inp, &s) };
        let result = t.softmax(dim as i64, Kind::Float);
        extract(&result, out);
    }

    // ── comparisons ──

    fn cmp_op(lhs: &[f32], rhs: &[f32], out: &mut [f32], f: fn(&Tensor, &Tensor) -> Tensor) {
        let n = lhs.len() as i64;
        let l = unsafe { wrap(lhs, &[n]) };
        let r = unsafe { wrap(rhs, &[n]) };
        let mask = f(&l, &r).to_kind(Kind::Float);
        extract(&mask, out);
    }

    pub fn cmp_lt(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        cmp_op(lhs, rhs, out, |l, r| l.lt_tensor(r));
    }

    pub fn cmp_gt(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        cmp_op(lhs, rhs, out, |l, r| l.gt_tensor(r));
    }

    pub fn cmp_le(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        cmp_op(lhs, rhs, out, |l, r| l.le_tensor(r));
    }

    pub fn cmp_ge(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        cmp_op(lhs, rhs, out, |l, r| l.ge_tensor(r));
    }

    pub fn cmp_eq(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        cmp_op(lhs, rhs, out, |l, r| l.eq_tensor(r));
    }

    pub fn cmp_ne(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        cmp_op(lhs, rhs, out, |l, r| l.ne_tensor(r));
    }
}

// ── Naive Rust implementations (no torch dependency) ────────────────

#[cfg(not(feature = "torch"))]
mod imp {
    // ── matmul ──

    pub fn matmul(lhs: &[f32], rhs: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for ki in 0..k {
                    sum += lhs[row * k + ki] * rhs[ki * n + col];
                }
                out[row * n + col] = sum;
            }
        }
    }

    // ── reductions ──

    fn reduce_generic(
        inp: &[f32],
        out: &mut [f32],
        shape: &[usize],
        dim: usize,
        f: impl Fn(&[f32]) -> f32,
    ) {
        let outer: usize = shape[..dim].iter().product();
        let dim_size = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();

        let mut slice = vec![0.0f32; dim_size];
        for o in 0..outer {
            for k in 0..inner {
                for d in 0..dim_size {
                    slice[d] = inp[o * dim_size * inner + d * inner + k];
                }
                out[o * inner + k] = f(&slice);
            }
        }
    }

    pub fn reduce_sum(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        reduce_generic(inp, out, shape, dim, |s| s.iter().sum());
    }

    pub fn reduce_mean(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        reduce_generic(inp, out, shape, dim, |s| {
            let sum: f32 = s.iter().sum();
            sum / s.len() as f32
        });
    }

    pub fn reduce_max(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        reduce_generic(inp, out, shape, dim, |s| {
            s.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        });
    }

    pub fn reduce_min(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        reduce_generic(inp, out, shape, dim, |s| {
            s.iter().cloned().fold(f32::INFINITY, f32::min)
        });
    }

    pub fn argmax(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        reduce_generic(inp, out, shape, dim, |s| {
            s.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as f32)
                .unwrap_or(0.0)
        });
    }

    pub fn argmin(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        reduce_generic(inp, out, shape, dim, |s| {
            s.iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as f32)
                .unwrap_or(0.0)
        });
    }

    pub fn softmax(inp: &[f32], out: &mut [f32], shape: &[usize], dim: usize) {
        let outer: usize = shape[..dim].iter().product();
        let dim_size = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();

        // Copy input to output first
        out[..inp.len()].copy_from_slice(inp);

        for o in 0..outer {
            for k in 0..inner {
                // Find max for numerical stability
                let mut max_val = f32::NEG_INFINITY;
                for d in 0..dim_size {
                    let idx = o * dim_size * inner + d * inner + k;
                    max_val = max_val.max(out[idx]);
                }
                // Compute exp and sum
                let mut sum = 0.0f32;
                for d in 0..dim_size {
                    let idx = o * dim_size * inner + d * inner + k;
                    out[idx] = (out[idx] - max_val).exp();
                    sum += out[idx];
                }
                // Normalize
                for d in 0..dim_size {
                    let idx = o * dim_size * inner + d * inner + k;
                    out[idx] /= sum;
                }
            }
        }
    }

    // ── comparisons ──

    fn cmp_generic(lhs: &[f32], rhs: &[f32], out: &mut [f32], f: fn(f32, f32) -> f32) {
        for i in 0..lhs.len() {
            out[i] = f(lhs[i], rhs[i]);
        }
    }

    pub fn cmp_lt(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        cmp_generic(lhs, rhs, out, |a, b| if a < b { 1.0 } else { 0.0 });
    }

    pub fn cmp_gt(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        cmp_generic(lhs, rhs, out, |a, b| if a > b { 1.0 } else { 0.0 });
    }

    pub fn cmp_le(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        cmp_generic(lhs, rhs, out, |a, b| if a <= b { 1.0 } else { 0.0 });
    }

    pub fn cmp_ge(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        cmp_generic(lhs, rhs, out, |a, b| if a >= b { 1.0 } else { 0.0 });
    }

    pub fn cmp_eq(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        cmp_generic(lhs, rhs, out, |a, b| {
            if (a - b).abs() < 1e-7 { 1.0 } else { 0.0 }
        });
    }

    pub fn cmp_ne(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        cmp_generic(lhs, rhs, out, |a, b| {
            if (a - b).abs() >= 1e-7 { 1.0 } else { 0.0 }
        });
    }
}

pub use imp::*;
