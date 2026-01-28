// SPCPP Backend - Rust FFI bindings to the spcpp/cuBLAS backend
//
// This provides an alternative GPU backend using cuBLAS for matmul
// and JIT-compiled CUDA kernels via spcpp for element-wise ops.

use libloading::{Library, Symbol};
use std::ffi::CString;
use std::sync::OnceLock;

static SPCPP_LIB: OnceLock<Option<Library>> = OnceLock::new();

/// Load the spcpp launcher library
pub fn load_spcpp() -> bool {
    SPCPP_LIB.get_or_init(|| {
        // Try to load the spcpp launcher
        let paths = [
            "kernels/external/spcpp/build/lib/spcpp_launcher.so",
            "./build/lib/spcpp_launcher.so",
            "libspcpp_launcher.so",
        ];

        for path in &paths {
            if let Ok(lib) = unsafe { Library::new(path) } {
                println!("[spcpp] Loaded from {}", path);
                return Some(lib);
            }
        }

        eprintln!("[spcpp] Could not load spcpp_launcher.so");
        None
    });

    SPCPP_LIB.get().map(|o| o.is_some()).unwrap_or(false)
}

/// Check if spcpp is available
pub fn is_available() -> bool {
    SPCPP_LIB.get().map(|o| o.is_some()).unwrap_or(false)
}

fn get_lib() -> Option<&'static Library> {
    SPCPP_LIB.get().and_then(|o| o.as_ref())
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/// Initialize spcpp with path to CUDA kernels
pub fn init(kernel_path: &str) -> Result<(), String> {
    let lib = get_lib().ok_or("spcpp not loaded")?;
    let c_path = CString::new(kernel_path).map_err(|e| e.to_string())?;

    unsafe {
        let func: Symbol<unsafe extern "C" fn(*const i8) -> i32> =
            lib.get(b"spcpp_init").map_err(|e| e.to_string())?;
        let result = func(c_path.as_ptr());
        if result != 0 {
            return Err(format!("spcpp_init failed with code {}", result));
        }
    }
    Ok(())
}

/// Shutdown spcpp and release resources
pub fn shutdown() {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn()>(b"spcpp_shutdown") {
                func();
            }
        }
    }
}

// ============================================================================
// MATMUL
// ============================================================================

/// Matrix multiplication: C = A @ B
/// A: [m, k], B: [k, n], C: [m, n]
pub fn matmul(a: *mut f32, b: *mut f32, c: *mut f32, m: i32, k: i32, n: i32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, *mut f32, *mut f32, i32, i32, i32)>(b"spcpp_matmul") {
                func(a, b, c, m, k, n);
            }
        }
    }
}

/// General matrix multiplication: C = alpha * op(A) @ op(B) + beta * C
pub fn gemm(
    a: *mut f32, b: *mut f32, c: *mut f32,
    m: i32, k: i32, n: i32,
    alpha: f32, beta: f32,
    trans_a: bool, trans_b: bool,
) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, *mut f32, *mut f32, i32, i32, i32, f32, f32, i32, i32)>(b"spcpp_gemm") {
                func(a, b, c, m, k, n, alpha, beta, trans_a as i32, trans_b as i32);
            }
        }
    }
}

// ============================================================================
// UNARY OPERATIONS
// ============================================================================

/// Apply unary operation in-place
pub fn unary(op: &str, data: *mut f32, n: i32) {
    if let Some(lib) = get_lib() {
        if let Ok(c_op) = CString::new(op) {
            unsafe {
                if let Ok(func) = lib.get::<unsafe extern "C" fn(*const i8, *mut f32, i32)>(b"spcpp_unary") {
                    func(c_op.as_ptr(), data, n);
                }
            }
        }
    }
}

/// Softmax over rows
pub fn softmax(data: *mut f32, rows: i32, cols: i32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, i32, i32)>(b"spcpp_softmax") {
                func(data, rows, cols);
            }
        }
    }
}

/// Leaky ReLU
pub fn leaky_relu(data: *mut f32, n: i32, alpha: f32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, i32, f32)>(b"spcpp_leaky_relu") {
                func(data, n, alpha);
            }
        }
    }
}

// ============================================================================
// BINARY OPERATIONS
// ============================================================================

/// Apply binary operation: c = op(a, b)
pub fn binary(op: &str, a: *mut f32, b: *mut f32, c: *mut f32, n: i32) {
    if let Some(lib) = get_lib() {
        if let Ok(c_op) = CString::new(op) {
            unsafe {
                if let Ok(func) = lib.get::<unsafe extern "C" fn(*const i8, *mut f32, *mut f32, *mut f32, i32)>(b"spcpp_binary") {
                    func(c_op.as_ptr(), a, b, c, n);
                }
            }
        }
    }
}

/// Scale data by scalar
pub fn scale(data: *mut f32, scalar: f32, n: i32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, f32, i32)>(b"spcpp_scale") {
                func(data, scalar, n);
            }
        }
    }
}

/// Fill with value
pub fn fill(data: *mut f32, val: f32, n: i32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, f32, i32)>(b"spcpp_fill") {
                func(data, val, n);
            }
        }
    }
}

// ============================================================================
// LINEAR LAYER
// ============================================================================

/// Linear forward: output = input @ weights + bias
pub fn linear_forward(
    input: *mut f32, weights: *mut f32, bias: *mut f32, output: *mut f32,
    batch: i32, in_features: i32, out_features: i32, has_bias: bool,
) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, *mut f32, *mut f32, *mut f32, i32, i32, i32, i32)>(b"spcpp_linear_forward") {
                func(input, weights, bias, output, batch, in_features, out_features, has_bias as i32);
            }
        }
    }
}

/// Linear backward
pub fn linear_backward(
    input: *mut f32, weights: *mut f32, grad_out: *mut f32,
    grad_input: *mut f32, grad_weights: *mut f32, grad_bias: *mut f32,
    batch: i32, in_f: i32, out_f: i32, has_bias: bool,
) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, *mut f32, *mut f32, *mut f32, *mut f32, *mut f32, i32, i32, i32, i32)>(b"spcpp_linear_backward") {
                func(input, weights, grad_out, grad_input, grad_weights, grad_bias, batch, in_f, out_f, has_bias as i32);
            }
        }
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/// Initialize weights
pub fn init_weights(name: &str, data: *mut f32, rows: i32, cols: i32, init_type: &str) {
    if let Some(lib) = get_lib() {
        if let (Ok(c_name), Ok(c_init)) = (CString::new(name), CString::new(init_type)) {
            unsafe {
                if let Ok(func) = lib.get::<unsafe extern "C" fn(*const i8, *mut f32, i32, i32, *const i8)>(b"spcpp_init_weights") {
                    func(c_name.as_ptr(), data, rows, cols, c_init.as_ptr());
                }
            }
        }
    }
}

// ============================================================================
// OPTIMIZER
// ============================================================================

/// Set optimizer parameters
pub fn set_optimizer_params(lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(f32, f32, f32, f32, f32)>(b"spcpp_set_optimizer_params") {
                func(lr, beta1, beta2, epsilon, weight_decay);
            }
        }
    }
}

/// SGD step with momentum
pub fn sgd_step(name: &str, weights: *mut f32, grad: *mut f32, n: i32, lr: f32, momentum: f32, wd: f32) {
    if let Some(lib) = get_lib() {
        if let Ok(c_name) = CString::new(name) {
            unsafe {
                if let Ok(func) = lib.get::<unsafe extern "C" fn(*const i8, *mut f32, *mut f32, i32, f32, f32, f32)>(b"spcpp_sgd_step") {
                    func(c_name.as_ptr(), weights, grad, n, lr, momentum, wd);
                }
            }
        }
    }
}

/// Adam step
pub fn adam_step(name: &str, weights: *mut f32, grad: *mut f32, n: i32) {
    if let Some(lib) = get_lib() {
        if let Ok(c_name) = CString::new(name) {
            unsafe {
                if let Ok(func) = lib.get::<unsafe extern "C" fn(*const i8, *mut f32, *mut f32, i32)>(b"spcpp_adam_step") {
                    func(c_name.as_ptr(), weights, grad, n);
                }
            }
        }
    }
}

/// Zero gradients
pub fn zero_grad(grad: *mut f32, n: i32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, i32)>(b"spcpp_zero_grad") {
                func(grad, n);
            }
        }
    }
}

// ============================================================================
// NORMALIZATION
// ============================================================================

/// Layer normalization
pub fn layer_norm(x: *mut f32, gamma: *mut f32, beta: *mut f32, out: *mut f32, batch: i32, dim: i32, eps: f32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, *mut f32, *mut f32, *mut f32, i32, i32, f32)>(b"spcpp_layer_norm") {
                func(x, gamma, beta, out, batch, dim, eps);
            }
        }
    }
}

/// Dropout
pub fn dropout(data: *mut f32, mask: *mut f32, n: i32, p: f32, seed: u32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, *mut f32, i32, f32, u32)>(b"spcpp_dropout") {
                func(data, mask, n, p, seed);
            }
        }
    }
}

// ============================================================================
// EMBEDDING
// ============================================================================

/// Embedding lookup
pub fn embedding(table: *mut f32, indices: *mut i32, out: *mut f32, batch: i32, dim: i32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, *mut i32, *mut f32, i32, i32)>(b"spcpp_embedding") {
                func(table, indices, out, batch, dim);
            }
        }
    }
}

// ============================================================================
// LOSS
// ============================================================================

/// Cross entropy loss
pub fn cross_entropy(logits: *mut f32, labels: *mut i32, batch: i32, classes: i32) -> f32 {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, *mut i32, i32, i32) -> f32>(b"spcpp_cross_entropy") {
                return func(logits, labels, batch, classes);
            }
        }
    }
    0.0
}

/// MSE loss
pub fn mse_loss(pred: *mut f32, target: *mut f32, n: i32) -> f32 {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, *mut f32, i32) -> f32>(b"spcpp_mse_loss") {
                return func(pred, target, n);
            }
        }
    }
    0.0
}

// ============================================================================
// QUANTIZATION
// ============================================================================

/// Quantize f32 to int8
pub fn quantize_int8(input: *mut f32, output: *mut i8, n: i32, scale: f32, zero_point: i32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, *mut i8, i32, f32, i32)>(b"spcpp_quantize_int8") {
                func(input, output, n, scale, zero_point);
            }
        }
    }
}

/// Dequantize int8 to f32
pub fn dequantize_int8(input: *mut i8, output: *mut f32, n: i32, scale: f32, zero_point: i32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut i8, *mut f32, i32, f32, i32)>(b"spcpp_dequantize_int8") {
                func(input, output, n, scale, zero_point);
            }
        }
    }
}

// ============================================================================
// MEMORY
// ============================================================================

/// Allocate GPU memory
pub fn alloc(bytes: usize) -> *mut u8 {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(usize) -> *mut u8>(b"spcpp_alloc") {
                return func(bytes);
            }
        }
    }
    std::ptr::null_mut()
}

/// Free GPU memory
pub fn free(ptr: *mut u8) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut u8)>(b"spcpp_free") {
                func(ptr);
            }
        }
    }
}

/// Copy host to device
pub fn memcpy_h2d(dst: *mut u8, src: *const u8, bytes: usize) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut u8, *const u8, usize)>(b"spcpp_h2d") {
                func(dst, src, bytes);
            }
        }
    }
}

/// Copy device to host
pub fn memcpy_d2h(dst: *mut u8, src: *const u8, bytes: usize) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut u8, *const u8, usize)>(b"spcpp_d2h") {
                func(dst, src, bytes);
            }
        }
    }
}

/// Copy device to device
pub fn memcpy_d2d(dst: *mut u8, src: *const u8, bytes: usize) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut u8, *const u8, usize)>(b"spcpp_d2d") {
                func(dst, src, bytes);
            }
        }
    }
}

/// Copy float buffers device to device
pub fn copy(src: *mut f32, dst: *mut f32, n: i32) {
    memcpy_d2d(dst as *mut u8, src as *const u8, (n as usize) * 4);
}

/// Synchronize device
pub fn sync() {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn()>(b"spcpp_sync") {
                func();
            }
        }
    }
}

// ============================================================================
// BLAS PRIMITIVES
// ============================================================================

/// y = alpha * x + y
pub fn axpy(x: *mut f32, y: *mut f32, n: i32, alpha: f32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, *mut f32, i32, f32)>(b"spcpp_axpy") {
                func(x, y, n, alpha);
            }
        }
    }
}

/// x = alpha * x
pub fn scal(x: *mut f32, n: i32, alpha: f32) {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, i32, f32)>(b"spcpp_scal") {
                func(x, n, alpha);
            }
        }
    }
}

/// dot product
pub fn dot(x: *mut f32, y: *mut f32, n: i32) -> f32 {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, *mut f32, i32) -> f32>(b"spcpp_dot") {
                return func(x, y, n);
            }
        }
    }
    0.0
}

/// L2 norm
pub fn nrm2(x: *mut f32, n: i32) -> f32 {
    if let Some(lib) = get_lib() {
        unsafe {
            if let Ok(func) = lib.get::<unsafe extern "C" fn(*mut f32, i32) -> f32>(b"spcpp_nrm2") {
                return func(x, n);
            }
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_spcpp() {
        // This test checks if the library can be loaded
        // Will pass even if library doesn't exist (returns false)
        let _ = load_spcpp();
    }
}
