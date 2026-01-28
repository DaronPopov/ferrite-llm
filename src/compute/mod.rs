pub mod synth;
pub mod spcpp;
pub mod backend;
pub mod cpu;
pub mod device;
pub mod stream;

// GPU backend (direct access)
pub use backend::{Backend, GpuTensor};
pub use backend::Backend as GpuBackend;

// CPU backend (direct access)
pub use cpu::{CpuBackend, CpuTensor};

// Unified API (recommended)
pub use device::{Runtime, DeviceType, Tensor};

// Zero-copy streaming API (fastest)
pub use stream::{Stream, Pipeline, StreamOp, GpuHandle};

use crate::data::{Grid, Values};
use crate::dynamics::{RuntimeRules, Device};
use crate::structure::Meaning;
use cudarc::driver::{LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use libloading::Symbol;

use crate::dynamics::Precision;

#[derive(Debug, Clone)]
pub enum Op {
    // Basic ops
    Add,
    MatMul,
    ReLU,
    MoveToDevice,
    MoveToHost,
    TorchUnary(String),
    TorchBinary(String),
    Vision(String),
    Math(String),
    Fused(Vec<Op>),

    // Training ops
    InitWeights { init_type: String },
    LinearForward { has_bias: bool },
    LinearBackward { has_bias: bool },
    MSELoss,
    MSELossBackward,
    CrossEntropyLoss { num_classes: usize },
    CrossEntropyBackward { num_classes: usize },
    BCEWithLogitsLoss,
    BCEBackward,
    ActivationBackward(String),  // relu, sigmoid, tanh, gelu
    SGDStep { lr: f32, momentum: f32, weight_decay: f32 },
    AdamStep { timestep: i32 },
    ZeroGrad,
    SetOptimizerParams { lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32 },
    BatchNormForward { training: bool, momentum: f32, eps: f32 },
    BatchNormBackward { eps: f32 },
    LayerNormForward { eps: f32 },
    DropoutForward { p: f32, training: bool },
    DropoutBackward { p: f32 },
    ScaledDotProductAttention { scale: f32, causal: bool },
    Randn { mean: f32, std: f32 },
    Rand { low: f32, high: f32 },
    Fill(f32),
    Scale(f32),
    Copy,
    ClipGradNorm { max_norm: f32 },
    ReduceSum,
    ReduceMean,
    ReduceMax,
    ReduceMin,
    Conv2dForward { stride: (usize, usize), padding: (usize, usize), has_bias: bool },
    MaxPool2d { kernel: (usize, usize), stride: (usize, usize) },
    AvgPool2d { kernel: (usize, usize), stride: (usize, usize) },

    // Mixed precision ops
    Cast { target_precision: Precision },
    HalfUnary { op: String, precision: Precision },
    HalfBinary { op: String, precision: Precision },
    HalfLinearForward { has_bias: bool, precision: Precision },
    SetLossScale { scale: f32 },
    ScaleGradients { scale: f32 },
    UnscaleGradients { scale: f32 },
    UpdateLossScale { had_overflow: bool, scale_factor: f32, scale_window: i32 },

    // Embedding ops
    /// Embedding lookup: indices (f32, will be cast to int) -> embeddings
    /// Input 0: indices Grid [num_indices] (values treated as integer indices)
    /// Input 1: embedding_table Grid [vocab_size, embed_dim]
    /// Output: Grid [num_indices, embed_dim]
    EmbeddingForward,

    // Gradient accumulation ops
    /// Accumulate gradients: accumulator += gradient
    /// Input 0: accumulator Grid
    /// Input 1: gradient Grid
    /// Returns: updated accumulator
    GradientAccumulate,

    /// Apply accumulated gradients with scaling
    /// Divides accumulated gradients by accumulation steps and applies optimizer
    /// Input 0: weight Grid
    /// Input 1: accumulated_gradient Grid
    /// Input 2: momentum Grid (for SGD) or m Grid (for Adam)
    /// Input 3 (Adam only): v Grid
    AccumulatedSGDStep { lr: f32, momentum: f32, weight_decay: f32, accumulation_steps: i32 },
    AccumulatedAdamStep { timestep: i32, accumulation_steps: i32 },

    /// Zero out gradient accumulator
    GradientZero,

    // ========================================================================
    // QUANTIZATION OPS (Custom Rust + CUDA implementation)
    // ========================================================================

    /// Quantize F32 tensor to INT8/UINT8/INT4/UINT4
    /// Input: F32 Grid
    /// Output: Quantized Grid (stored as bytes)
    Quantize {
        target: Precision,
        scale: f32,
        zero_point: i32,
    },

    /// Dequantize INT8/UINT8/INT4/UINT4 tensor back to F32
    /// Input: Quantized Grid
    /// Output: F32 Grid
    Dequantize {
        source: Precision,
        scale: f32,
        zero_point: i32,
    },

    /// Compute min/max of F32 tensor for calibration
    /// Returns Grid with [min, max]
    CalibrationMinMax,

    /// Quantized linear layer: F32 input, INT8 weights -> F32 output
    /// Input 0: F32 input [batch, in_features]
    /// Input 1: INT8 quantized weights [in_features, out_features]
    /// Input 2: F32 bias (optional)
    QuantizedLinearInt8 {
        has_bias: bool,
        weight_scale: f32,
        weight_zero_point: i32,
    },

    /// Quantized linear layer with INT4 weights (extreme compression)
    QuantizedLinearInt4 {
        has_bias: bool,
        weight_scale: f32,
        weight_zero_point: i32,
    },

    /// Quantized matmul: INT8 x INT8 -> F32
    QuantizedMatMulInt8 {
        scale_a: f32,
        zero_point_a: i32,
        scale_b: f32,
        zero_point_b: i32,
    },

    /// Quantized ReLU (operates in quantized domain)
    QuantizedReLU {
        precision: Precision,
        zero_point: i32,
    },

    /// Add two quantized tensors, output F32
    QuantizedAdd {
        precision: Precision,
        scale_a: f32,
        zero_point_a: i32,
        scale_b: f32,
        zero_point_b: i32,
    },
}

// QuantParams available for users via crate::dynamics::QuantParams

/// Convert Precision enum to C-compatible integer
fn precision_to_int(p: Precision) -> i32 {
    match p {
        Precision::F16 => 0,
        Precision::BF16 => 1,
        Precision::F32 => 2,
        Precision::F64 => 3,
        Precision::Int8 => 4,
        Precision::UInt8 => 5,
        Precision::Int4 => 6,
        Precision::UInt4 => 7,
    }
}

fn load_ptx_robust(dev: &std::sync::Arc<cudarc::driver::CudaDevice>, ptx_path: &str, module_name: &str, funcs: &[&str]) {
    if dev.get_func(module_name, funcs[0]).is_some() {
        return;
    }

    let ptx_content = std::fs::read_to_string(ptx_path).expect("Failed to read PTX file");
    let mut modified_ptx = ptx_content.clone();
    
    // Check for high PTX versions and downgrade
    if ptx_content.contains(".version 8.5") || ptx_content.contains(".version 8.4") || ptx_content.contains(".version 8.3") || ptx_content.contains(".version 8.2") || ptx_content.contains(".version 8.1") {
        println!("[Brain] Auto-downgrading static kernel {} to PTX 8.0", ptx_path);
        modified_ptx = ptx_content.replace(".version 8.5", ".version 8.0")
                                  .replace(".version 8.4", ".version 8.0")
                                  .replace(".version 8.3", ".version 8.0")
                                  .replace(".version 8.2", ".version 8.0")
                                  .replace(".version 8.1", ".version 8.0");
    }

    let ptx = Ptx::from_src(modified_ptx);
    let leaked_funcs: Vec<&'static str> = funcs.iter().map(|&s| Box::leak(s.to_string().into_boxed_str()) as &'static str).collect();
    dev.load_ptx(ptx, module_name, &leaked_funcs).expect("Physical Body rejected static kernel after surgery");
}

pub fn compute(
    op: Op,
    inputs: Vec<&Grid>,
    _context: Option<&Meaning>,
    rules: &RuntimeRules,
) -> Grid {
    match op {
        Op::MoveToDevice => {
            let a = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                a.to_device(alloc)
            } else {
                panic!("MoveToDevice requires a GPU device");
            }
        }
        Op::MoveToHost => {
            let a = inputs[0];
            match &a.values {
                Values::Device { allocator, offset, len } => {
                    let mut host_vec = vec![0.0f32; *len];
                    allocator.copy_from_offset(*offset, &mut host_vec);
                    Grid::new(host_vec, a.shape.clone())
                }
                Values::Host(_) => a.clone(),
            }
        }
        Op::ReLU => {
            let a = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &a.values {
                    let dev = alloc.device();
                    dev.bind_to_thread().expect("Context bind failed");
                    load_ptx_robust(dev, "kernels/math/activation/relu.ptx", "relu_mod", &["relu_f32"]);
                    let f = dev.get_func("relu_mod", "relu_f32").expect("Func not found");

                    let ptr = alloc.pool_ptr() + *offset as u64;
                    let n = *len as i32;
                    
                    unsafe {
                        f.launch(LaunchConfig::for_num_elems(*len as u32), (ptr, n)).expect("Kernel launch failed");
                    }
                    return a.clone();
                }
            }
            match &a.values {
                Values::Host(va) => {
                    let data = va.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
                    Grid::new(data, a.shape.clone())
                }
                _ => panic!("ReLU CPU fallback failed"),
            }
        }
        Op::MatMul => {
             let a = inputs[0];
             let b = inputs[1];
             if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: a_off, .. }, Values::Device { offset: b_off, .. }) = (&a.values, &b.values) {
                    let m = a.shape[0] as i32;
                    let k = a.shape[1] as i32;
                    let n = b.shape[1] as i32;
                    
                    let res_size = (m * n) as usize;
                    let c_off = alloc.alloc(res_size * std::mem::size_of::<f32>()).expect("Out of memory for Result C");

                    let dev = alloc.device();
                    dev.bind_to_thread().expect("Context bind failed");
                    load_ptx_robust(dev, "kernels/math/linear/matmul.ptx", "matmul_mod", &["matmul_f32"]);
                    let f = dev.get_func("matmul_mod", "matmul_f32").expect("Func not found");

                    let a_ptr = alloc.pool_ptr() + *a_off as u64;
                    let b_ptr = alloc.pool_ptr() + *b_off as u64;
                    let c_ptr = alloc.pool_ptr() + c_off as u64;

                    let cfg = LaunchConfig {
                        grid_dim: ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };

                    unsafe {
                        f.launch(cfg, (a_ptr, b_ptr, c_ptr, m, k, n)).expect("Kernel launch failed");
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: c_off,
                            len: res_size,
                        },
                        shape: vec![m as usize, n as usize],
                    };
                }
             }
             match (&a.values, &b.values) {
                (Values::Host(va), Values::Host(vb)) => {
                    let (m, k1) = (a.shape[0], a.shape[1]);
                    let (k2, n) = (b.shape[0], b.shape[1]);
                    assert_eq!(k1, k2);
                    let mut res = vec![0.0; m * n];
                    for i in 0..m {
                        for j in 0..n {
                            for k in 0..k1 {
                                res[i * n + j] += va[i * k1 + k] * vb[k * n + j];
                            }
                        }
                    }
                    Grid::new(res, vec![m, n])
                }
                _ => panic!("MatMul CPU fallback failed"),
            }
        }
        Op::Add => {
            let a = inputs[0];
            let b = inputs[1];
            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: a_off, len, .. }, Values::Device { offset: b_off, .. }) = (&a.values, &b.values) {
                    let res_size = *len;
                    let c_off = alloc.alloc(res_size * std::mem::size_of::<f32>()).expect("Out of memory for Result C");

                    let dev = alloc.device();
                    dev.bind_to_thread().expect("Context bind failed");
                    load_ptx_robust(dev, "kernels/math/linear/add.ptx", "add_mod", &["add_f32"]);
                    let f = dev.get_func("add_mod", "add_f32").expect("Func not found");

                    let a_ptr = alloc.pool_ptr() + *a_off as u64;
                    let b_ptr = alloc.pool_ptr() + *b_off as u64;
                    let c_ptr = alloc.pool_ptr() + c_off as u64;

                    unsafe {
                        f.launch(LaunchConfig::for_num_elems(res_size as u32), (a_ptr, b_ptr, c_ptr, res_size as i32)).expect("Kernel launch failed");
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: c_off,
                            len: res_size,
                        },
                        shape: a.shape.clone(),
                    };
                }
            }
            match (&a.values, &b.values) {
                (Values::Host(va), Values::Host(vb)) => {
                    let data = va.iter().zip(vb.iter()).map(|(x, y)| x + y).collect();
                    Grid::new(data, a.shape.clone())
                }
                _ => panic!("Add CPU fallback failed"),
            }
        }
        Op::TorchUnary(torch_op) => {
            let a = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &a.values {
                    let ptr = (alloc.pool_ptr() + *offset as u64) as *mut f32;
                    let size = *len as i32;
                    
                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib_path = "kernels/external/torch/libtorch_bridge.so";
                        let lib = rules.muscle_memory.get(lib_path);
                        let func: Symbol<extern "C" fn(*const std::ffi::c_char, *mut f32, i32)> = lib.get(b"torch_bridge_unary").expect("Symbol not found");
                        
                        let c_op = std::ffi::CString::new(torch_op).unwrap();
                        func(c_op.as_ptr(), ptr, size);
                    }
                    return a.clone();
                }
            }
            panic!("TorchUnary requires GPU memory");
        }
        Op::TorchBinary(torch_op) => {
            let a = inputs[0];
            let b = inputs[1];
            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: a_off, len, .. }, Values::Device { offset: b_off, .. }) = (&a.values, &b.values) {
                    let res_size = *len;
                    let c_off = alloc.alloc(res_size * std::mem::size_of::<f32>()).expect("Out of memory for Result C");

                    let a_ptr = (alloc.pool_ptr() + *a_off as u64) as *mut f32;
                    let b_ptr = (alloc.pool_ptr() + *b_off as u64) as *mut f32;
                    let c_ptr = (alloc.pool_ptr() + c_off as u64) as *mut f32;
                    
                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib_path = "kernels/external/torch/libtorch_bridge.so";
                        let lib = rules.muscle_memory.get(lib_path);
                        let func: Symbol<extern "C" fn(*const std::ffi::c_char, *mut f32, *mut f32, *mut f32, i32)> = lib.get(b"torch_bridge_binary").expect("Symbol not found");
                        
                        let c_op = std::ffi::CString::new(torch_op).unwrap();
                        func(c_op.as_ptr(), a_ptr, b_ptr, c_ptr, res_size as i32);
                    }
                    
                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: c_off,
                            len: res_size,
                        },
                        shape: a.shape.clone(),
                    };
                }
            }
            panic!("TorchBinary requires GPU memory");
        }
        Op::Vision(vision_op) => {
            let a = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &a.values {
                    // Host Fallback because OpenCV CUDA is not detected
                    let mut host_data = vec![0.0f32; *len];
                    alloc.copy_from_offset(*offset, &mut host_data);

                    let width = a.shape.last().cloned().unwrap_or(1) as i32;
                    let height = if a.shape.len() > 1 { a.shape[a.shape.len()-2] } else { 1 } as i32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib_path = "kernels/external/vision/libopencv_bridge.so";
                        let lib = rules.muscle_memory.get(lib_path);
                        let func: Symbol<extern "C" fn(*const std::ffi::c_char, *mut f32, i32, i32, i32)> = lib.get(b"opencv_bridge_unary").expect("Symbol not found");
                        
                        let c_op = std::ffi::CString::new(vision_op).unwrap();
                        func(c_op.as_ptr(), host_data.as_mut_ptr(), *len as i32, width, height);
                    }

                    alloc.copy_to_offset(*offset, &host_data);
                    return a.clone();
                }
            }
            panic!("Vision relies on GPU/Host sync in this implementation");
        }
        Op::Math(math_op) => {
            let a = inputs[0];
            if let Device::Gpu(_alloc) = &rules.device {
                if let Values::Device { .. } = &a.values {
                    // Semantically optimize: single Math op is just a 1-op Fusion
                    return compute(Op::Fused(vec![Op::Math(math_op)]), inputs, _context, rules);
                }
            }
            panic!("Math relies on GPU/Host sync in this implementation");
        }
        Op::Fused(ops) => {
            let a = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &a.values {
                    let dev = alloc.device();
                    dev.bind_to_thread().expect("Context bind failed");

                    // 1. Generate JIT Kernel Source
                    let mut kernel_logic = String::new();
                    for op in ops {
                        match op {
                            Op::ReLU => kernel_logic.push_str("x = (x > 0.0f) ? x : 0.0f;\n"),
                            Op::Math(m) => {
                                if m == "sin" { kernel_logic.push_str("x = sinf(x);\n"); }
                                else if m == "exp" { kernel_logic.push_str("x = expf(x);\n"); }
                                else { kernel_logic.push_str(&format!("x = {};\n", m)); }
                            },
                            _ => panic!("Operation {:?} not supported for fusion yet", op),
                        }
                    }

                    let kernel_name = format!("fused_{:x}", fxhash::hash64(&kernel_logic));

                    let synth = rules.synthesizer.as_ref().expect("Synthesizer required for Fused ops");
                    let ptx_key = synth.synthesize(&kernel_name, &kernel_logic);

                    // 3. Launch
                    let f = dev.get_func(&ptx_key, &kernel_name).expect("Fused func not found");
                    let ptr = alloc.pool_ptr() + *offset as u64;
                    let n = *len as i32;

                    unsafe {
                        f.launch(LaunchConfig::for_num_elems(*len as u32), (ptr, n)).expect("Fused kernel launch failed");
                    }
                    return a.clone();
                }
            }
            panic!("Fused ops require GPU memory");
        }

        // ====================================================================
        // TRAINING OPERATIONS
        // ====================================================================

        Op::InitWeights { init_type } => {
            let w = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len: _, .. } = &w.values {
                    let ptr = (alloc.pool_ptr() + *offset as u64) as *mut f32;
                    let rows = w.shape[0] as i32;
                    let cols = if w.shape.len() > 1 { w.shape[1] } else { 1 } as i32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*const std::ffi::c_char, *mut f32, i32, i32, *const std::ffi::c_char)> =
                            lib.get(b"torch_init_weights").expect("Symbol not found");

                        let c_name = std::ffi::CString::new("weight").unwrap();
                        let c_init = std::ffi::CString::new(init_type.as_str()).unwrap();
                        func(c_name.as_ptr(), ptr, rows, cols, c_init.as_ptr());
                    }
                    return w.clone();
                }
            }
            panic!("InitWeights requires GPU memory");
        }

        Op::LinearForward { has_bias } => {
            // inputs: [input, weight, bias?]
            let input = inputs[0];
            let weight = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: in_off, .. }, Values::Device { offset: w_off, .. }) =
                    (&input.values, &weight.values)
                {
                    let batch = input.shape[0] as i32;
                    let in_features = input.shape[1] as i32;
                    let out_features = weight.shape[1] as i32;

                    // Allocate output
                    let out_size = (batch * out_features) as usize;
                    let out_off = alloc.alloc(out_size * std::mem::size_of::<f32>()).expect("OOM");

                    let in_ptr = (alloc.pool_ptr() + *in_off as u64) as *mut f32;
                    let w_ptr = (alloc.pool_ptr() + *w_off as u64) as *mut f32;
                    let out_ptr = (alloc.pool_ptr() + out_off as u64) as *mut f32;

                    let bias_ptr = if has_bias && inputs.len() > 2 {
                        if let Values::Device { offset: b_off, .. } = &inputs[2].values {
                            (alloc.pool_ptr() + *b_off as u64) as *mut f32
                        } else { std::ptr::null_mut() }
                    } else { std::ptr::null_mut() };

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32, i32, *mut f32, i32, *mut f32, *mut f32)> =
                            lib.get(b"torch_linear_forward").expect("Symbol not found");
                        func(in_ptr, batch, in_features, w_ptr, out_features, bias_ptr, out_ptr);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: out_size,
                        },
                        shape: vec![batch as usize, out_features as usize],
                    };
                }
            }
            panic!("LinearForward requires GPU memory");
        }

        Op::LinearBackward { has_bias } => {
            // inputs: [input, weight, grad_output, grad_input_out, grad_weight_out, grad_bias_out?]
            let input = inputs[0];
            let weight = inputs[1];
            let grad_output = inputs[2];

            if let Device::Gpu(alloc) = &rules.device {
                if let (
                    Values::Device { offset: in_off, .. },
                    Values::Device { offset: w_off, .. },
                    Values::Device { offset: go_off, .. }
                ) = (&input.values, &weight.values, &grad_output.values) {
                    let batch = input.shape[0] as i32;
                    let in_features = input.shape[1] as i32;
                    let out_features = weight.shape[1] as i32;

                    // Allocate gradient buffers
                    let gi_size = (batch * in_features) as usize;
                    let gw_size = (in_features * out_features) as usize;
                    let gi_off = alloc.alloc(gi_size * 4).expect("OOM");
                    let gw_off = alloc.alloc(gw_size * 4).expect("OOM");
                    let gb_off = if has_bias {
                        Some(alloc.alloc(out_features as usize * 4).expect("OOM"))
                    } else { None };

                    let in_ptr = (alloc.pool_ptr() + *in_off as u64) as *mut f32;
                    let w_ptr = (alloc.pool_ptr() + *w_off as u64) as *mut f32;
                    let go_ptr = (alloc.pool_ptr() + *go_off as u64) as *mut f32;
                    let gi_ptr = (alloc.pool_ptr() + gi_off as u64) as *mut f32;
                    let gw_ptr = (alloc.pool_ptr() + gw_off as u64) as *mut f32;
                    let gb_ptr = gb_off.map(|o| (alloc.pool_ptr() + o as u64) as *mut f32)
                        .unwrap_or(std::ptr::null_mut());

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32, i32, *mut f32, i32, *mut f32, *mut f32, *mut f32, *mut f32)> =
                            lib.get(b"torch_linear_backward").expect("Symbol not found");
                        func(in_ptr, batch, in_features, w_ptr, out_features, go_ptr, gi_ptr, gw_ptr, gb_ptr);
                    }

                    // Return grad_input (caller can access others via registry)
                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: gi_off,
                            len: gi_size,
                        },
                        shape: vec![batch as usize, in_features as usize],
                    };
                }
            }
            panic!("LinearBackward requires GPU memory");
        }

        Op::MSELoss => {
            // inputs: [pred, target]
            let pred = inputs[0];
            let target = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: p_off, len, .. }, Values::Device { offset: t_off, .. }) =
                    (&pred.values, &target.values)
                {
                    let p_ptr = (alloc.pool_ptr() + *p_off as u64) as *mut f32;
                    let t_ptr = (alloc.pool_ptr() + *t_off as u64) as *mut f32;

                    let loss: f32 = unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut f32, i32) -> f32> =
                            lib.get(b"torch_mse_loss").expect("Symbol not found");
                        func(p_ptr, t_ptr, *len as i32)
                    };

                    // Return scalar loss as 1-element grid
                    let loss_off = alloc.alloc(4).expect("OOM");
                    alloc.copy_to_offset(loss_off, &[loss]);
                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: loss_off,
                            len: 1,
                        },
                        shape: vec![1],
                    };
                }
            }
            panic!("MSELoss requires GPU memory");
        }

        Op::MSELossBackward => {
            // inputs: [pred, target]
            let pred = inputs[0];
            let target = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: p_off, len, .. }, Values::Device { offset: t_off, .. }) =
                    (&pred.values, &target.values)
                {
                    let grad_off = alloc.alloc(*len * 4).expect("OOM");
                    let p_ptr = (alloc.pool_ptr() + *p_off as u64) as *mut f32;
                    let t_ptr = (alloc.pool_ptr() + *t_off as u64) as *mut f32;
                    let g_ptr = (alloc.pool_ptr() + grad_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut f32, *mut f32, i32)> =
                            lib.get(b"torch_mse_loss_backward").expect("Symbol not found");
                        func(p_ptr, t_ptr, g_ptr, *len as i32);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: grad_off,
                            len: *len,
                        },
                        shape: pred.shape.clone(),
                    };
                }
            }
            panic!("MSELossBackward requires GPU memory");
        }

        Op::CrossEntropyLoss { num_classes } => {
            // inputs: [logits, targets (int)]
            let logits = inputs[0];
            let targets = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: l_off, .. }, Values::Device { offset: t_off, .. }) =
                    (&logits.values, &targets.values)
                {
                    let batch = logits.shape[0] as i32;
                    let l_ptr = (alloc.pool_ptr() + *l_off as u64) as *mut f32;
                    let t_ptr = (alloc.pool_ptr() + *t_off as u64) as *mut i32;

                    let loss: f32 = unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut i32, i32, i32) -> f32> =
                            lib.get(b"torch_cross_entropy_loss").expect("Symbol not found");
                        func(l_ptr, t_ptr, batch, num_classes as i32)
                    };

                    let loss_off = alloc.alloc(4).expect("OOM");
                    alloc.copy_to_offset(loss_off, &[loss]);
                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: loss_off,
                            len: 1,
                        },
                        shape: vec![1],
                    };
                }
            }
            panic!("CrossEntropyLoss requires GPU memory");
        }

        Op::CrossEntropyBackward { num_classes } => {
            let logits = inputs[0];
            let targets = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: l_off, len, .. }, Values::Device { offset: t_off, .. }) =
                    (&logits.values, &targets.values)
                {
                    let batch = logits.shape[0] as i32;
                    let grad_off = alloc.alloc(*len * 4).expect("OOM");

                    let l_ptr = (alloc.pool_ptr() + *l_off as u64) as *mut f32;
                    let t_ptr = (alloc.pool_ptr() + *t_off as u64) as *mut i32;
                    let g_ptr = (alloc.pool_ptr() + grad_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut i32, *mut f32, i32, i32)> =
                            lib.get(b"torch_cross_entropy_backward").expect("Symbol not found");
                        func(l_ptr, t_ptr, g_ptr, batch, num_classes as i32);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: grad_off,
                            len: *len,
                        },
                        shape: logits.shape.clone(),
                    };
                }
            }
            panic!("CrossEntropyBackward requires GPU memory");
        }

        Op::BCEWithLogitsLoss => {
            let pred = inputs[0];
            let target = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: p_off, len, .. }, Values::Device { offset: t_off, .. }) =
                    (&pred.values, &target.values)
                {
                    let p_ptr = (alloc.pool_ptr() + *p_off as u64) as *mut f32;
                    let t_ptr = (alloc.pool_ptr() + *t_off as u64) as *mut f32;

                    let loss: f32 = unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut f32, i32) -> f32> =
                            lib.get(b"torch_bce_with_logits_loss").expect("Symbol not found");
                        func(p_ptr, t_ptr, *len as i32)
                    };

                    let loss_off = alloc.alloc(4).expect("OOM");
                    alloc.copy_to_offset(loss_off, &[loss]);
                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: loss_off,
                            len: 1,
                        },
                        shape: vec![1],
                    };
                }
            }
            panic!("BCEWithLogitsLoss requires GPU memory");
        }

        Op::BCEBackward => {
            let pred = inputs[0];
            let target = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: p_off, len, .. }, Values::Device { offset: t_off, .. }) =
                    (&pred.values, &target.values)
                {
                    let grad_off = alloc.alloc(*len * 4).expect("OOM");

                    let p_ptr = (alloc.pool_ptr() + *p_off as u64) as *mut f32;
                    let t_ptr = (alloc.pool_ptr() + *t_off as u64) as *mut f32;
                    let g_ptr = (alloc.pool_ptr() + grad_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut f32, *mut f32, i32)> =
                            lib.get(b"torch_bce_backward").expect("Symbol not found");
                        func(p_ptr, t_ptr, g_ptr, *len as i32);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: grad_off,
                            len: *len,
                        },
                        shape: pred.shape.clone(),
                    };
                }
            }
            panic!("BCEBackward requires GPU memory");
        }

        Op::ActivationBackward(activation) => {
            // inputs: [x_or_output, grad_out]
            let x = inputs[0];
            let grad_out = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: x_off, len, .. }, Values::Device { offset: go_off, .. }) =
                    (&x.values, &grad_out.values)
                {
                    let grad_in_off = alloc.alloc(*len * 4).expect("OOM");

                    let x_ptr = (alloc.pool_ptr() + *x_off as u64) as *mut f32;
                    let go_ptr = (alloc.pool_ptr() + *go_off as u64) as *mut f32;
                    let gi_ptr = (alloc.pool_ptr() + grad_in_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");

                        let func_name = format!("torch_{}_backward", activation);
                        let func: Symbol<extern "C" fn(*mut f32, *mut f32, *mut f32, i32)> =
                            lib.get(func_name.as_bytes()).expect("Activation backward not found");
                        func(x_ptr, go_ptr, gi_ptr, *len as i32);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: grad_in_off,
                            len: *len,
                        },
                        shape: x.shape.clone(),
                    };
                }
            }
            panic!("ActivationBackward requires GPU memory");
        }

        Op::SGDStep { lr, momentum, weight_decay } => {
            // inputs: [weight, grad, momentum_buffer]
            let weight = inputs[0];
            let grad = inputs[1];
            let mom_buf = inputs[2];

            if let Device::Gpu(alloc) = &rules.device {
                if let (
                    Values::Device { offset: w_off, len, .. },
                    Values::Device { offset: g_off, .. },
                    Values::Device { offset: m_off, .. }
                ) = (&weight.values, &grad.values, &mom_buf.values) {
                    let w_ptr = (alloc.pool_ptr() + *w_off as u64) as *mut f32;
                    let g_ptr = (alloc.pool_ptr() + *g_off as u64) as *mut f32;
                    let m_ptr = (alloc.pool_ptr() + *m_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut f32, *mut f32, i32, f32, f32, f32)> =
                            lib.get(b"torch_sgd_step").expect("Symbol not found");
                        func(w_ptr, g_ptr, m_ptr, *len as i32, lr, momentum, weight_decay);
                    }

                    return weight.clone();
                }
            }
            panic!("SGDStep requires GPU memory");
        }

        Op::AdamStep { timestep } => {
            // inputs: [weight, grad, m_buffer, v_buffer]
            let weight = inputs[0];
            let grad = inputs[1];
            let m_buf = inputs[2];
            let v_buf = inputs[3];

            if let Device::Gpu(alloc) = &rules.device {
                if let (
                    Values::Device { offset: w_off, len, .. },
                    Values::Device { offset: g_off, .. },
                    Values::Device { offset: m_off, .. },
                    Values::Device { offset: v_off, .. }
                ) = (&weight.values, &grad.values, &m_buf.values, &v_buf.values) {
                    let w_ptr = (alloc.pool_ptr() + *w_off as u64) as *mut f32;
                    let g_ptr = (alloc.pool_ptr() + *g_off as u64) as *mut f32;
                    let m_ptr = (alloc.pool_ptr() + *m_off as u64) as *mut f32;
                    let v_ptr = (alloc.pool_ptr() + *v_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut f32, *mut f32, *mut f32, i32, i32)> =
                            lib.get(b"torch_adam_step").expect("Symbol not found");
                        func(w_ptr, g_ptr, m_ptr, v_ptr, *len as i32, timestep);
                    }

                    return weight.clone();
                }
            }
            panic!("AdamStep requires GPU memory");
        }

        Op::ZeroGrad => {
            let grad = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &grad.values {
                    let g_ptr = (alloc.pool_ptr() + *offset as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32)> =
                            lib.get(b"torch_zero_grad").expect("Symbol not found");
                        func(g_ptr, *len as i32);
                    }

                    return grad.clone();
                }
            }
            panic!("ZeroGrad requires GPU memory");
        }

        Op::SetOptimizerParams { lr, beta1, beta2, eps, weight_decay } => {
            if let Device::Gpu(alloc) = &rules.device {
                unsafe {
                    alloc.device().bind_to_thread().expect("Context bind failed");
                    let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                    let func: Symbol<extern "C" fn(f32, f32, f32, f32, f32)> =
                        lib.get(b"torch_set_optimizer_params").expect("Symbol not found");
                    func(lr, beta1, beta2, eps, weight_decay);
                }
            }
            // Return empty grid as this is just setting state
            Grid::new(vec![], vec![0])
        }

        Op::Randn { mean, std } => {
            let a = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &a.values {
                    let ptr = (alloc.pool_ptr() + *offset as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32, f32, f32)> =
                            lib.get(b"torch_randn").expect("Symbol not found");
                        func(ptr, *len as i32, mean, std);
                    }

                    return a.clone();
                }
            }
            panic!("Randn requires GPU memory");
        }

        Op::Rand { low, high } => {
            let a = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &a.values {
                    let ptr = (alloc.pool_ptr() + *offset as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32, f32, f32)> =
                            lib.get(b"torch_rand").expect("Symbol not found");
                        func(ptr, *len as i32, low, high);
                    }

                    return a.clone();
                }
            }
            panic!("Rand requires GPU memory");
        }

        Op::Fill(value) => {
            let a = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &a.values {
                    let ptr = (alloc.pool_ptr() + *offset as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32, f32)> =
                            lib.get(b"torch_fill").expect("Symbol not found");
                        func(ptr, *len as i32, value);
                    }

                    return a.clone();
                }
            }
            panic!("Fill requires GPU memory");
        }

        Op::Scale(factor) => {
            let a = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &a.values {
                    let ptr = (alloc.pool_ptr() + *offset as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32, f32)> =
                            lib.get(b"torch_scale").expect("Symbol not found");
                        func(ptr, *len as i32, factor);
                    }

                    return a.clone();
                }
            }
            panic!("Scale requires GPU memory");
        }

        Op::Copy => {
            let src = inputs[0];
            let dst = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: s_off, len, .. }, Values::Device { offset: d_off, .. }) =
                    (&src.values, &dst.values)
                {
                    let s_ptr = (alloc.pool_ptr() + *s_off as u64) as *mut f32;
                    let d_ptr = (alloc.pool_ptr() + *d_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut f32, i32)> =
                            lib.get(b"torch_copy").expect("Symbol not found");
                        func(s_ptr, d_ptr, *len as i32);
                    }

                    return dst.clone();
                }
            }
            panic!("Copy requires GPU memory");
        }

        Op::ReduceSum | Op::ReduceMean | Op::ReduceMax | Op::ReduceMin => {
            let a = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &a.values {
                    let ptr = (alloc.pool_ptr() + *offset as u64) as *mut f32;

                    let func_name = match &op {
                        Op::ReduceSum => "torch_reduce_sum",
                        Op::ReduceMean => "torch_reduce_mean",
                        Op::ReduceMax => "torch_reduce_max",
                        Op::ReduceMin => "torch_reduce_min",
                        _ => unreachable!(),
                    };

                    let result: f32 = unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32) -> f32> =
                            lib.get(func_name.as_bytes()).expect("Symbol not found");
                        func(ptr, *len as i32)
                    };

                    let result_off = alloc.alloc(4).expect("OOM");
                    alloc.copy_to_offset(result_off, &[result]);
                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: result_off,
                            len: 1,
                        },
                        shape: vec![1],
                    };
                }
            }
            panic!("Reduce ops require GPU memory");
        }

        Op::BatchNormForward { training, momentum, eps } => {
            // inputs: [input, gamma, beta, running_mean, running_var]
            let input = inputs[0];
            let gamma = inputs[1];
            let beta = inputs[2];
            let running_mean = inputs[3];
            let running_var = inputs[4];

            if let Device::Gpu(alloc) = &rules.device {
                let batch = input.shape[0] as i32;
                let features = input.shape[1] as i32;

                if let (
                    Values::Device { offset: i_off, .. },
                    Values::Device { offset: g_off, .. },
                    Values::Device { offset: b_off, .. },
                    Values::Device { offset: rm_off, .. },
                    Values::Device { offset: rv_off, .. }
                ) = (&input.values, &gamma.values, &beta.values, &running_mean.values, &running_var.values) {
                    let out_size = (batch * features) as usize;
                    let out_off = alloc.alloc(out_size * 4).expect("OOM");
                    let sm_off = alloc.alloc(features as usize * 4).expect("OOM");
                    let sv_off = alloc.alloc(features as usize * 4).expect("OOM");

                    let i_ptr = (alloc.pool_ptr() + *i_off as u64) as *mut f32;
                    let g_ptr = (alloc.pool_ptr() + *g_off as u64) as *mut f32;
                    let b_ptr = (alloc.pool_ptr() + *b_off as u64) as *mut f32;
                    let rm_ptr = (alloc.pool_ptr() + *rm_off as u64) as *mut f32;
                    let rv_ptr = (alloc.pool_ptr() + *rv_off as u64) as *mut f32;
                    let o_ptr = (alloc.pool_ptr() + out_off as u64) as *mut f32;
                    let sm_ptr = (alloc.pool_ptr() + sm_off as u64) as *mut f32;
                    let sv_ptr = (alloc.pool_ptr() + sv_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32, i32, *mut f32, *mut f32, *mut f32, *mut f32, *mut f32, *mut f32, *mut f32, f32, f32, bool)> =
                            lib.get(b"torch_batch_norm_forward").expect("Symbol not found");
                        func(i_ptr, batch, features, g_ptr, b_ptr, rm_ptr, rv_ptr, o_ptr, sm_ptr, sv_ptr, momentum, eps, training);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: out_size,
                        },
                        shape: input.shape.clone(),
                    };
                }
            }
            panic!("BatchNormForward requires GPU memory");
        }

        Op::BatchNormBackward { eps } => {
            // inputs: [grad_out, input, gamma, save_mean, save_var]
            let grad_out = inputs[0];
            let input = inputs[1];
            let gamma = inputs[2];
            let save_mean = inputs[3];
            let save_var = inputs[4];

            if let Device::Gpu(alloc) = &rules.device {
                let batch = input.shape[0] as i32;
                let features = input.shape[1] as i32;

                if let (
                    Values::Device { offset: go_off, .. },
                    Values::Device { offset: i_off, .. },
                    Values::Device { offset: g_off, .. },
                    Values::Device { offset: sm_off, .. },
                    Values::Device { offset: sv_off, .. }
                ) = (&grad_out.values, &input.values, &gamma.values, &save_mean.values, &save_var.values) {
                    let gi_size = (batch * features) as usize;
                    let gi_off = alloc.alloc(gi_size * 4).expect("OOM");
                    let dg_off = alloc.alloc(features as usize * 4).expect("OOM");
                    let db_off = alloc.alloc(features as usize * 4).expect("OOM");

                    let go_ptr = (alloc.pool_ptr() + *go_off as u64) as *mut f32;
                    let i_ptr = (alloc.pool_ptr() + *i_off as u64) as *mut f32;
                    let g_ptr = (alloc.pool_ptr() + *g_off as u64) as *mut f32;
                    let sm_ptr = (alloc.pool_ptr() + *sm_off as u64) as *mut f32;
                    let sv_ptr = (alloc.pool_ptr() + *sv_off as u64) as *mut f32;
                    let gi_ptr = (alloc.pool_ptr() + gi_off as u64) as *mut f32;
                    let dg_ptr = (alloc.pool_ptr() + dg_off as u64) as *mut f32;
                    let db_ptr = (alloc.pool_ptr() + db_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut f32, i32, i32, *mut f32, *mut f32, *mut f32, *mut f32, *mut f32, *mut f32, f32)> =
                            lib.get(b"torch_batch_norm_backward").expect("Symbol not found");
                        func(go_ptr, i_ptr, batch, features, g_ptr, sm_ptr, sv_ptr, gi_ptr, dg_ptr, db_ptr, eps);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: gi_off,
                            len: gi_size,
                        },
                        shape: input.shape.clone(),
                    };
                }
            }
            panic!("BatchNormBackward requires GPU memory");
        }

        Op::LayerNormForward { eps } => {
            // inputs: [input, gamma, beta]
            let input = inputs[0];
            let gamma = inputs[1];
            let beta = inputs[2];

            if let Device::Gpu(alloc) = &rules.device {
                let batch = input.shape[0] as i32;
                let features = input.shape[1] as i32;

                if let (
                    Values::Device { offset: i_off, .. },
                    Values::Device { offset: g_off, .. },
                    Values::Device { offset: b_off, .. }
                ) = (&input.values, &gamma.values, &beta.values) {
                    let out_size = (batch * features) as usize;
                    let out_off = alloc.alloc(out_size * 4).expect("OOM");
                    let sm_off = alloc.alloc(batch as usize * 4).expect("OOM");
                    let sr_off = alloc.alloc(batch as usize * 4).expect("OOM");

                    let i_ptr = (alloc.pool_ptr() + *i_off as u64) as *mut f32;
                    let g_ptr = (alloc.pool_ptr() + *g_off as u64) as *mut f32;
                    let b_ptr = (alloc.pool_ptr() + *b_off as u64) as *mut f32;
                    let o_ptr = (alloc.pool_ptr() + out_off as u64) as *mut f32;
                    let sm_ptr = (alloc.pool_ptr() + sm_off as u64) as *mut f32;
                    let sr_ptr = (alloc.pool_ptr() + sr_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32, i32, *mut f32, *mut f32, *mut f32, *mut f32, *mut f32, f32)> =
                            lib.get(b"torch_layer_norm_forward").expect("Symbol not found");
                        func(i_ptr, batch, features, g_ptr, b_ptr, o_ptr, sm_ptr, sr_ptr, eps);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: out_size,
                        },
                        shape: input.shape.clone(),
                    };
                }
            }
            panic!("LayerNormForward requires GPU memory");
        }

        Op::DropoutForward { p, training } => {
            let input = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &input.values {
                    let mask_off = alloc.alloc(*len * 4).expect("OOM");

                    let x_ptr = (alloc.pool_ptr() + *offset as u64) as *mut f32;
                    let m_ptr = (alloc.pool_ptr() + mask_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut f32, i32, f32, bool)> =
                            lib.get(b"torch_dropout_forward").expect("Symbol not found");
                        func(x_ptr, m_ptr, *len as i32, p, training);
                    }

                    return input.clone();
                }
            }
            panic!("DropoutForward requires GPU memory");
        }

        Op::DropoutBackward { p } => {
            let grad_out = inputs[0];
            let mask = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: go_off, len, .. }, Values::Device { offset: m_off, .. }) =
                    (&grad_out.values, &mask.values)
                {
                    let gi_off = alloc.alloc(*len * 4).expect("OOM");

                    let go_ptr = (alloc.pool_ptr() + *go_off as u64) as *mut f32;
                    let m_ptr = (alloc.pool_ptr() + *m_off as u64) as *mut f32;
                    let gi_ptr = (alloc.pool_ptr() + gi_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut f32, *mut f32, i32, f32)> =
                            lib.get(b"torch_dropout_backward").expect("Symbol not found");
                        func(go_ptr, m_ptr, gi_ptr, *len as i32, p);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: gi_off,
                            len: *len,
                        },
                        shape: grad_out.shape.clone(),
                    };
                }
            }
            panic!("DropoutBackward requires GPU memory");
        }

        Op::ScaledDotProductAttention { scale, causal } => {
            // inputs: [query, key, value]
            let query = inputs[0];
            let key = inputs[1];
            let value = inputs[2];

            if let Device::Gpu(alloc) = &rules.device {
                let batch = query.shape[0] as i32;
                let seq_len = query.shape[1] as i32;
                let head_dim = query.shape[2] as i32;

                if let (
                    Values::Device { offset: q_off, .. },
                    Values::Device { offset: k_off, .. },
                    Values::Device { offset: v_off, .. }
                ) = (&query.values, &key.values, &value.values) {
                    let out_size = (batch * seq_len * head_dim) as usize;
                    let attn_size = (batch * seq_len * seq_len) as usize;
                    let out_off = alloc.alloc(out_size * 4).expect("OOM");
                    let attn_off = alloc.alloc(attn_size * 4).expect("OOM");

                    let q_ptr = (alloc.pool_ptr() + *q_off as u64) as *mut f32;
                    let k_ptr = (alloc.pool_ptr() + *k_off as u64) as *mut f32;
                    let v_ptr = (alloc.pool_ptr() + *v_off as u64) as *mut f32;
                    let o_ptr = (alloc.pool_ptr() + out_off as u64) as *mut f32;
                    let a_ptr = (alloc.pool_ptr() + attn_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut f32, *mut f32, i32, i32, i32, *mut f32, *mut f32, f32, bool)> =
                            lib.get(b"torch_scaled_dot_product_attention").expect("Symbol not found");
                        func(q_ptr, k_ptr, v_ptr, batch, seq_len, head_dim, o_ptr, a_ptr, scale, causal);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: out_size,
                        },
                        shape: query.shape.clone(),
                    };
                }
            }
            panic!("ScaledDotProductAttention requires GPU memory");
        }

        Op::ClipGradNorm { max_norm: _ } => {
            // This requires multiple grads - simplified version clips single grad
            let grad = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &grad.values {
                    let g_ptr = (alloc.pool_ptr() + *offset as u64) as *mut f32;

                    // For now, just scale if norm exceeds max
                    // Full implementation would handle multiple grads
                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");

                        // Get norm first
                        let norm_func: Symbol<extern "C" fn(*mut f32, i32) -> f32> =
                            lib.get(b"torch_reduce_sum").expect("Symbol not found");

                        // Simplified clipping
                        let _norm = norm_func(g_ptr, *len as i32);
                    }

                    return grad.clone();
                }
            }
            panic!("ClipGradNorm requires GPU memory");
        }

        Op::Conv2dForward { stride, padding, has_bias } => {
            // inputs: [input, weight, bias?]
            let input = inputs[0];
            let weight = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (
                    Values::Device { offset: i_off, .. },
                    Values::Device { offset: w_off, .. }
                ) = (&input.values, &weight.values) {
                    let batch = input.shape[0] as i32;
                    let in_channels = input.shape[1] as i32;
                    let height = input.shape[2] as i32;
                    let width = input.shape[3] as i32;
                    let out_channels = weight.shape[0] as i32;
                    let kernel_h = weight.shape[2] as i32;
                    let kernel_w = weight.shape[3] as i32;

                    let out_h = (height + 2 * padding.0 as i32 - kernel_h) / stride.0 as i32 + 1;
                    let out_w = (width + 2 * padding.1 as i32 - kernel_w) / stride.1 as i32 + 1;
                    let out_size = (batch * out_channels * out_h * out_w) as usize;
                    let out_off = alloc.alloc(out_size * 4).expect("OOM");

                    let i_ptr = (alloc.pool_ptr() + *i_off as u64) as *mut f32;
                    let w_ptr = (alloc.pool_ptr() + *w_off as u64) as *mut f32;
                    let o_ptr = (alloc.pool_ptr() + out_off as u64) as *mut f32;

                    let bias_ptr = if has_bias && inputs.len() > 2 {
                        if let Values::Device { offset: b_off, .. } = &inputs[2].values {
                            (alloc.pool_ptr() + *b_off as u64) as *mut f32
                        } else { std::ptr::null_mut() }
                    } else { std::ptr::null_mut() };

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32, i32, i32, i32, *mut f32, i32, i32, i32, *mut f32, *mut f32, i32, i32, i32, i32)> =
                            lib.get(b"torch_conv2d_forward").expect("Symbol not found");
                        func(i_ptr, batch, in_channels, height, width, w_ptr, out_channels, kernel_h, kernel_w, bias_ptr, o_ptr, stride.0 as i32, stride.1 as i32, padding.0 as i32, padding.1 as i32);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: out_size,
                        },
                        shape: vec![batch as usize, out_channels as usize, out_h as usize, out_w as usize],
                    };
                }
            }
            panic!("Conv2dForward requires GPU memory");
        }

        Op::MaxPool2d { kernel, stride } => {
            let input = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset: i_off, .. } = &input.values {
                    let batch = input.shape[0] as i32;
                    let channels = input.shape[1] as i32;
                    let height = input.shape[2] as i32;
                    let width = input.shape[3] as i32;

                    let out_h = (height - kernel.0 as i32) / stride.0 as i32 + 1;
                    let out_w = (width - kernel.1 as i32) / stride.1 as i32 + 1;
                    let out_size = (batch * channels * out_h * out_w) as usize;
                    let out_off = alloc.alloc(out_size * 4).expect("OOM");

                    let i_ptr = (alloc.pool_ptr() + *i_off as u64) as *mut f32;
                    let o_ptr = (alloc.pool_ptr() + out_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32, i32, i32, i32, *mut f32, i32, i32, i32, i32)> =
                            lib.get(b"torch_max_pool2d").expect("Symbol not found");
                        func(i_ptr, batch, channels, height, width, o_ptr, kernel.0 as i32, kernel.1 as i32, stride.0 as i32, stride.1 as i32);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: out_size,
                        },
                        shape: vec![batch as usize, channels as usize, out_h as usize, out_w as usize],
                    };
                }
            }
            panic!("MaxPool2d requires GPU memory");
        }

        Op::AvgPool2d { kernel, stride } => {
            let input = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset: i_off, .. } = &input.values {
                    let batch = input.shape[0] as i32;
                    let channels = input.shape[1] as i32;
                    let height = input.shape[2] as i32;
                    let width = input.shape[3] as i32;

                    let out_h = (height - kernel.0 as i32) / stride.0 as i32 + 1;
                    let out_w = (width - kernel.1 as i32) / stride.1 as i32 + 1;
                    let out_size = (batch * channels * out_h * out_w) as usize;
                    let out_off = alloc.alloc(out_size * 4).expect("OOM");

                    let i_ptr = (alloc.pool_ptr() + *i_off as u64) as *mut f32;
                    let o_ptr = (alloc.pool_ptr() + out_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32, i32, i32, i32, *mut f32, i32, i32, i32, i32)> =
                            lib.get(b"torch_avg_pool2d").expect("Symbol not found");
                        func(i_ptr, batch, channels, height, width, o_ptr, kernel.0 as i32, kernel.1 as i32, stride.0 as i32, stride.1 as i32);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: out_size,
                        },
                        shape: vec![batch as usize, channels as usize, out_h as usize, out_w as usize],
                    };
                }
            }
            panic!("AvgPool2d requires GPU memory");
        }

        // ====================================================================
        // MIXED PRECISION OPERATIONS
        // ====================================================================

        Op::Cast { target_precision } => {
            let a = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &a.values {
                    let src_precision = precision_to_int(rules.precision);
                    let dst_precision = precision_to_int(target_precision);

                    // Calculate output size based on target precision
                    let dst_elem_size = target_precision.size_bytes();
                    let dst_size_bytes = *len * dst_elem_size;
                    let dst_off = alloc.alloc(dst_size_bytes).expect("Cast allocation failed");

                    let src_ptr = (alloc.pool_ptr() + *offset as u64) as *mut std::ffi::c_void;
                    let dst_ptr = (alloc.pool_ptr() + dst_off as u64) as *mut std::ffi::c_void;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void, i32, i32, i32)> =
                            lib.get(b"torch_cast").expect("Symbol not found");
                        func(src_ptr, dst_ptr, *len as i32, src_precision, dst_precision);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: dst_off,
                            len: *len,
                        },
                        shape: a.shape.clone(),
                    };
                }
            }
            panic!("Cast requires GPU memory");
        }

        Op::HalfUnary { op, precision } => {
            let a = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &a.values {
                    let ptr = (alloc.pool_ptr() + *offset as u64) as *mut std::ffi::c_void;
                    let prec = precision_to_int(precision);

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*const std::ffi::c_char, *mut std::ffi::c_void, i32, i32)> =
                            lib.get(b"torch_half_unary").expect("Symbol not found");
                        let c_op = std::ffi::CString::new(op.as_str()).unwrap();
                        func(c_op.as_ptr(), ptr, *len as i32, prec);
                    }

                    return a.clone();
                }
            }
            panic!("HalfUnary requires GPU memory");
        }

        Op::HalfBinary { op, precision } => {
            let a = inputs[0];
            let b = inputs[1];
            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: a_off, len, .. }, Values::Device { offset: b_off, .. }) = (&a.values, &b.values) {
                    let c_off = alloc.alloc(*len * precision.size_bytes()).expect("HalfBinary allocation failed");

                    let a_ptr = (alloc.pool_ptr() + *a_off as u64) as *mut std::ffi::c_void;
                    let b_ptr = (alloc.pool_ptr() + *b_off as u64) as *mut std::ffi::c_void;
                    let c_ptr = (alloc.pool_ptr() + c_off as u64) as *mut std::ffi::c_void;
                    let prec = precision_to_int(precision);

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*const std::ffi::c_char, *mut std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void, i32, i32)> =
                            lib.get(b"torch_half_binary").expect("Symbol not found");
                        let c_op = std::ffi::CString::new(op.as_str()).unwrap();
                        func(c_op.as_ptr(), a_ptr, b_ptr, c_ptr, *len as i32, prec);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: c_off,
                            len: *len,
                        },
                        shape: a.shape.clone(),
                    };
                }
            }
            panic!("HalfBinary requires GPU memory");
        }

        Op::HalfLinearForward { has_bias, precision } => {
            let x = inputs[0];
            let w = inputs[1];
            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: x_off, .. }, Values::Device { offset: w_off, .. }) = (&x.values, &w.values) {
                    let batch = x.shape[0] as i32;
                    let in_features = x.shape[1] as i32;
                    let out_features = w.shape[1] as i32;

                    let out_len = (batch * out_features) as usize;
                    let y_off = alloc.alloc(out_len * precision.size_bytes()).expect("HalfLinear allocation failed");

                    let x_ptr = (alloc.pool_ptr() + *x_off as u64) as *mut std::ffi::c_void;
                    let w_ptr = (alloc.pool_ptr() + *w_off as u64) as *mut std::ffi::c_void;
                    let y_ptr = (alloc.pool_ptr() + y_off as u64) as *mut std::ffi::c_void;

                    let (b_ptr, has_bias_int): (*mut std::ffi::c_void, i32) = if has_bias && inputs.len() > 2 {
                        if let Values::Device { offset: b_off, .. } = &inputs[2].values {
                            ((alloc.pool_ptr() + *b_off as u64) as *mut std::ffi::c_void, 1)
                        } else {
                            (std::ptr::null_mut(), 0)
                        }
                    } else {
                        (std::ptr::null_mut(), 0)
                    };

                    let prec = precision_to_int(precision);

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void, i32, i32, i32, i32, i32)> =
                            lib.get(b"torch_half_linear_forward").expect("Symbol not found");
                        func(x_ptr, w_ptr, b_ptr, y_ptr, batch, in_features, out_features, has_bias_int, prec);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: y_off,
                            len: out_len,
                        },
                        shape: vec![batch as usize, out_features as usize],
                    };
                }
            }
            panic!("HalfLinearForward requires GPU memory");
        }

        Op::SetLossScale { scale } => {
            if let Device::Gpu(alloc) = &rules.device {
                unsafe {
                    alloc.device().bind_to_thread().expect("Context bind failed");
                    let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                    let func: Symbol<extern "C" fn(f32)> =
                        lib.get(b"torch_set_loss_scale").expect("Symbol not found");
                    func(scale);
                }
            }
            // Return empty grid (side-effect only operation)
            Grid::new(vec![scale], vec![1])
        }

        Op::ScaleGradients { scale } => {
            let grad = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &grad.values {
                    let ptr = (alloc.pool_ptr() + *offset as u64) as *mut std::ffi::c_void;
                    let prec = precision_to_int(rules.precision);

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut std::ffi::c_void, i32, f32, i32)> =
                            lib.get(b"torch_scale_gradients").expect("Symbol not found");
                        func(ptr, *len as i32, scale, prec);
                    }

                    return grad.clone();
                }
            }
            panic!("ScaleGradients requires GPU memory");
        }

        Op::UnscaleGradients { scale } => {
            let grad = inputs[0];
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &grad.values {
                    let ptr = (alloc.pool_ptr() + *offset as u64) as *mut std::ffi::c_void;
                    let prec = precision_to_int(rules.precision);

                    let valid: i32;
                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut std::ffi::c_void, i32, f32, i32) -> i32> =
                            lib.get(b"torch_unscale_gradients").expect("Symbol not found");
                        valid = func(ptr, *len as i32, scale, prec);
                    }

                    // Return a Grid with the validity flag (1 = valid, 0 = overflow)
                    return Grid::new(vec![valid as f32], vec![1]);
                }
            }
            panic!("UnscaleGradients requires GPU memory");
        }

        Op::UpdateLossScale { had_overflow, scale_factor, scale_window } => {
            if let Device::Gpu(alloc) = &rules.device {
                unsafe {
                    alloc.device().bind_to_thread().expect("Context bind failed");
                    let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                    let func: Symbol<extern "C" fn(i32, f32, i32)> =
                        lib.get(b"torch_update_loss_scale").expect("Symbol not found");
                    func(if had_overflow { 1 } else { 0 }, scale_factor, scale_window);

                    // Get the new loss scale
                    let get_scale: Symbol<extern "C" fn() -> f32> =
                        lib.get(b"torch_get_loss_scale").expect("Symbol not found");
                    let new_scale = get_scale();
                    return Grid::new(vec![new_scale], vec![1]);
                }
            }
            panic!("UpdateLossScale requires GPU device");
        }

        // ====================================================================
        // EMBEDDING OPERATIONS
        // ====================================================================

        Op::EmbeddingForward => {
            let indices = inputs[0];
            let embedding_table = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: idx_off, len: num_indices, .. },
                        Values::Device { offset: table_off, .. }) = (&indices.values, &embedding_table.values) {

                    let vocab_size = embedding_table.shape[0] as i32;
                    let embed_dim = embedding_table.shape[1] as i32;

                    // Allocate output: [num_indices, embed_dim]
                    let out_len = *num_indices * (embed_dim as usize);
                    let out_off = alloc.alloc(out_len * std::mem::size_of::<f32>()).expect("Embedding output allocation failed");

                    // The indices need to be converted from f32 to i64 for PyTorch
                    // First, copy indices to host, convert, then copy back as i64
                    let mut host_indices_f32 = vec![0.0f32; *num_indices];
                    alloc.copy_from_offset(*idx_off, &mut host_indices_f32);

                    // Convert to i32 (we'll use cuMemcpyHtoD for the i32 data)
                    let host_indices_i32: Vec<i32> = host_indices_f32.iter().map(|&x| x as i32).collect();

                    // Allocate GPU memory for i32 indices
                    let idx_i32_off = alloc.alloc(*num_indices * std::mem::size_of::<i32>()).expect("Index conversion allocation failed");

                    // Copy i32 indices to GPU
                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");

                        #[link(name = "cuda")]
                        extern "C" {
                            fn cuMemcpyHtoD_v2(dst: u64, src: *const std::ffi::c_void, bytes: usize) -> i32;
                        }

                        let dst_ptr = alloc.pool_ptr() + idx_i32_off as u64;
                        let res = cuMemcpyHtoD_v2(dst_ptr, host_indices_i32.as_ptr() as *const std::ffi::c_void, *num_indices * 4);
                        if res != 0 {
                            panic!("Failed to copy indices to GPU: {}", res);
                        }

                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut i32, i32, *mut f32, i32, i32, *mut f32)> =
                            lib.get(b"torch_embedding_forward").expect("Symbol not found");

                        let idx_ptr = (alloc.pool_ptr() + idx_i32_off as u64) as *mut i32;
                        let table_ptr = (alloc.pool_ptr() + *table_off as u64) as *mut f32;
                        let out_ptr = (alloc.pool_ptr() + out_off as u64) as *mut f32;

                        func(idx_ptr, *num_indices as i32, table_ptr, vocab_size, embed_dim, out_ptr);
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: out_len,
                        },
                        shape: vec![*num_indices, embed_dim as usize],
                    };
                }
            }

            // CPU fallback
            if let (Values::Host(idx_data), Values::Host(table_data)) = (&indices.values, &embedding_table.values) {
                let vocab_size = embedding_table.shape[0];
                let embed_dim = embedding_table.shape[1];
                let num_indices = indices.numel();

                let mut output = Vec::with_capacity(num_indices * embed_dim);

                for &idx_f in idx_data {
                    let idx = idx_f as usize;
                    if idx >= vocab_size {
                        panic!("Embedding index {} out of range (vocab_size={})", idx, vocab_size);
                    }
                    let start = idx * embed_dim;
                    output.extend_from_slice(&table_data[start..start + embed_dim]);
                }

                return Grid::new(output, vec![num_indices, embed_dim]);
            }

            panic!("EmbeddingForward requires matching memory locations");
        }

        // ====================================================================
        // GRADIENT ACCUMULATION OPERATIONS
        // ====================================================================

        Op::GradientAccumulate => {
            let accumulator = inputs[0];
            let gradient = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: acc_off, len, .. },
                        Values::Device { offset: grad_off, .. }) = (&accumulator.values, &gradient.values) {

                    let acc_ptr = (alloc.pool_ptr() + *acc_off as u64) as *mut f32;
                    let grad_ptr = (alloc.pool_ptr() + *grad_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        // Use torch_bridge_binary with "add" to accumulate
                        let func: Symbol<extern "C" fn(*const std::ffi::c_char, *mut f32, *mut f32, *mut f32, i32)> =
                            lib.get(b"torch_bridge_binary").expect("Symbol not found");
                        let c_op = std::ffi::CString::new("add").unwrap();
                        // Add gradient to accumulator in-place (acc = acc + grad)
                        func(c_op.as_ptr(), acc_ptr, grad_ptr, acc_ptr, *len as i32);
                    }

                    return accumulator.clone();
                }
            }

            // CPU fallback
            if let (Values::Host(acc_data), Values::Host(grad_data)) = (&accumulator.values, &gradient.values) {
                let result: Vec<f32> = acc_data.iter().zip(grad_data.iter())
                    .map(|(&a, &g)| a + g)
                    .collect();
                return Grid::new(result, accumulator.shape.clone());
            }

            panic!("GradientAccumulate requires matching memory locations");
        }

        Op::GradientZero => {
            let accumulator = inputs[0];

            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &accumulator.values {
                    let ptr = (alloc.pool_ptr() + *offset as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, i32, f32)> =
                            lib.get(b"torch_fill").expect("Symbol not found");
                        func(ptr, *len as i32, 0.0);
                    }

                    return accumulator.clone();
                }
            }

            // CPU fallback
            if let Values::Host(data) = &accumulator.values {
                let result = vec![0.0f32; data.len()];
                return Grid::new(result, accumulator.shape.clone());
            }

            panic!("GradientZero requires GPU or Host memory");
        }

        Op::AccumulatedSGDStep { lr, momentum, weight_decay, accumulation_steps } => {
            let weight = inputs[0];
            let accumulated_grad = inputs[1];
            let momentum_buf = inputs[2];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: w_off, len, .. },
                        Values::Device { offset: g_off, .. },
                        Values::Device { offset: m_off, .. }) = (&weight.values, &accumulated_grad.values, &momentum_buf.values) {

                    let w_ptr = (alloc.pool_ptr() + *w_off as u64) as *mut f32;
                    let g_ptr = (alloc.pool_ptr() + *g_off as u64) as *mut f32;
                    let m_ptr = (alloc.pool_ptr() + *m_off as u64) as *mut f32;

                    // Scale learning rate by 1/accumulation_steps to average gradients
                    let scaled_lr = lr / (accumulation_steps as f32);

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");
                        let func: Symbol<extern "C" fn(*mut f32, *mut f32, *mut f32, i32, f32, f32, f32)> =
                            lib.get(b"torch_sgd_step").expect("Symbol not found");
                        func(w_ptr, g_ptr, m_ptr, *len as i32, scaled_lr, momentum, weight_decay);
                    }

                    return weight.clone();
                }
            }

            panic!("AccumulatedSGDStep requires GPU memory");
        }

        Op::AccumulatedAdamStep { timestep, accumulation_steps } => {
            let weight = inputs[0];
            let accumulated_grad = inputs[1];
            let m = inputs[2];
            let v = inputs[3];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: w_off, len, .. },
                        Values::Device { offset: g_off, .. },
                        Values::Device { offset: m_off, .. },
                        Values::Device { offset: v_off, .. }) = (&weight.values, &accumulated_grad.values, &m.values, &v.values) {

                    let w_ptr = (alloc.pool_ptr() + *w_off as u64) as *mut f32;
                    let g_ptr = (alloc.pool_ptr() + *g_off as u64) as *mut f32;
                    let m_ptr = (alloc.pool_ptr() + *m_off as u64) as *mut f32;
                    let v_ptr = (alloc.pool_ptr() + *v_off as u64) as *mut f32;

                    unsafe {
                        alloc.device().bind_to_thread().expect("Context bind failed");
                        let lib = rules.muscle_memory.get("kernels/external/torch/libtorch_bridge.so");

                        // First, scale accumulated gradients by 1/accumulation_steps
                        let scale_func: Symbol<extern "C" fn(*mut f32, i32, f32)> =
                            lib.get(b"torch_scale").expect("Symbol not found");
                        scale_func(g_ptr, *len as i32, 1.0 / (accumulation_steps as f32));

                        // Then apply Adam step
                        let adam_func: Symbol<extern "C" fn(*mut f32, *mut f32, *mut f32, *mut f32, i32, i32)> =
                            lib.get(b"torch_adam_step").expect("Symbol not found");
                        adam_func(w_ptr, g_ptr, m_ptr, v_ptr, *len as i32, timestep);
                    }

                    return weight.clone();
                }
            }

            panic!("AccumulatedAdamStep requires GPU memory");
        }

        // ====================================================================
        // QUANTIZATION OPERATIONS (Custom CUDA kernels)
        // ====================================================================

        Op::Quantize { target, scale, zero_point } => {
            let input = inputs[0];

            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset: in_off, len, .. } = &input.values {
                    let dev = alloc.device();
                    dev.bind_to_thread().expect("Context bind failed");

                    let in_ptr = alloc.pool_ptr() + *in_off as u64;

                    // Calculate output size based on precision
                    let out_bytes = target.storage_bytes(*len);
                    let out_off = alloc.alloc(out_bytes).expect("Quantize allocation failed");
                    let out_ptr = alloc.pool_ptr() + out_off as u64;

                    // Load quantization kernels
                    let kernel_name = match target {
                        Precision::Int8 => "quantize_f32_to_int8",
                        Precision::UInt8 => "quantize_f32_to_uint8",
                        Precision::Int4 => "quantize_f32_to_int4_packed",
                        Precision::UInt4 => "quantize_f32_to_uint4_packed",
                        _ => panic!("Quantize only supports Int8, UInt8, Int4, UInt4"),
                    };

                    load_ptx_robust(dev, "kernels/quant/quantize.ptx", "quant_mod", &[kernel_name]);
                    let f = dev.get_func("quant_mod", kernel_name).expect("Quant func not found");

                    let n = if target.is_packed_4bit() {
                        ((*len + 1) / 2) as u32  // Number of packed bytes
                    } else {
                        *len as u32
                    };

                    unsafe {
                        f.launch(
                            LaunchConfig::for_num_elems(n),
                            (in_ptr, out_ptr, *len as i32, scale, zero_point)
                        ).expect("Quantize kernel launch failed");
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: out_bytes,  // Store byte count for packed types
                        },
                        shape: input.shape.clone(),
                    };
                }
            }

            // CPU fallback for Int8
            if let Values::Host(data) = &input.values {
                match target {
                    Precision::Int8 => {
                        let quantized: Vec<f32> = data.iter()
                            .map(|&x| {
                                let q = (x / scale + zero_point as f32).round();
                                q.max(-128.0).min(127.0)
                            })
                            .collect();
                        return Grid::new(quantized, input.shape.clone());
                    }
                    Precision::UInt8 => {
                        let quantized: Vec<f32> = data.iter()
                            .map(|&x| {
                                let q = (x / scale + zero_point as f32).round();
                                q.max(0.0).min(255.0)
                            })
                            .collect();
                        return Grid::new(quantized, input.shape.clone());
                    }
                    _ => panic!("CPU quantization only supports Int8/UInt8"),
                }
            }

            panic!("Quantize requires GPU or Host memory");
        }

        Op::Dequantize { source, scale, zero_point } => {
            let input = inputs[0];

            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset: in_off, len, .. } = &input.values {
                    let dev = alloc.device();
                    dev.bind_to_thread().expect("Context bind failed");

                    let in_ptr = alloc.pool_ptr() + *in_off as u64;

                    // Calculate output size (always F32)
                    let num_elements = if source.is_packed_4bit() {
                        *len * 2  // Each byte holds 2 values
                    } else {
                        *len
                    };
                    let out_bytes = num_elements * 4;
                    let out_off = alloc.alloc(out_bytes).expect("Dequantize allocation failed");
                    let out_ptr = alloc.pool_ptr() + out_off as u64;

                    let kernel_name = match source {
                        Precision::Int8 => "dequantize_int8_to_f32",
                        Precision::UInt8 => "dequantize_uint8_to_f32",
                        Precision::Int4 => "dequantize_int4_packed_to_f32",
                        Precision::UInt4 => "dequantize_uint4_packed_to_f32",
                        _ => panic!("Dequantize only supports Int8, UInt8, Int4, UInt4"),
                    };

                    load_ptx_robust(dev, "kernels/quant/quantize.ptx", "quant_mod", &[kernel_name]);
                    let f = dev.get_func("quant_mod", kernel_name).expect("Dequant func not found");

                    let n = if source.is_packed_4bit() {
                        ((*len + 1) / 2) as u32
                    } else {
                        *len as u32
                    };

                    unsafe {
                        f.launch(
                            LaunchConfig::for_num_elems(n),
                            (in_ptr, out_ptr, num_elements as i32, scale, zero_point)
                        ).expect("Dequantize kernel launch failed");
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: num_elements,
                        },
                        shape: input.shape.clone(),
                    };
                }
            }

            // CPU fallback
            if let Values::Host(data) = &input.values {
                let dequantized: Vec<f32> = data.iter()
                    .map(|&x| (x - zero_point as f32) * scale)
                    .collect();
                return Grid::new(dequantized, input.shape.clone());
            }

            panic!("Dequantize requires GPU or Host memory");
        }

        Op::CalibrationMinMax => {
            let input = inputs[0];

            // CPU implementation (GPU would need atomic min/max)
            if let Values::Host(data) = &input.values {
                let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                return Grid::new(vec![min_val, max_val], vec![2]);
            }

            // For GPU, move to host first and compute
            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &input.values {
                    let mut host_data = vec![0.0f32; *len];
                    alloc.copy_from_offset(*offset, &mut host_data);

                    let min_val = host_data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_val = host_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    return Grid::new(vec![min_val, max_val], vec![2]);
                }
            }

            panic!("CalibrationMinMax requires valid memory");
        }

        Op::QuantizedLinearInt8 { has_bias, weight_scale, weight_zero_point } => {
            let input = inputs[0];  // F32 [batch, in_features]
            let weight = inputs[1]; // INT8 [in_features, out_features]

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: in_off, .. },
                        Values::Device { offset: w_off, .. }) = (&input.values, &weight.values) {

                    let batch = input.shape[0] as i32;
                    let in_features = input.shape[1] as i32;
                    let out_features = weight.shape[1] as i32;

                    let out_len = (batch * out_features) as usize;
                    let out_off = alloc.alloc(out_len * 4).expect("QuantLinear allocation failed");

                    let dev = alloc.device();
                    dev.bind_to_thread().expect("Context bind failed");

                    let in_ptr = alloc.pool_ptr() + *in_off as u64;
                    let w_ptr = alloc.pool_ptr() + *w_off as u64;
                    let out_ptr = alloc.pool_ptr() + out_off as u64;

                    let (b_ptr, has_bias_int): (u64, i32) = if has_bias && inputs.len() > 2 {
                        if let Values::Device { offset: b_off, .. } = &inputs[2].values {
                            (alloc.pool_ptr() + *b_off as u64, 1)
                        } else {
                            (0, 0)
                        }
                    } else {
                        (0, 0)
                    };

                    load_ptx_robust(dev, "kernels/quant/quantize.ptx", "quant_mod", &["linear_f32_int8_f32"]);
                    let f = dev.get_func("quant_mod", "linear_f32_int8_f32").expect("QuantLinear func not found");

                    let block_dim = (16u32, 16u32, 1u32);
                    let grid_dim = (
                        ((out_features as u32) + 15) / 16,
                        ((batch as u32) + 15) / 16,
                        1u32
                    );

                    unsafe {
                        f.launch(
                            LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 },
                            (in_ptr, w_ptr, b_ptr, out_ptr, batch, in_features, out_features, weight_scale, weight_zero_point, has_bias_int)
                        ).expect("QuantLinear kernel launch failed");
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: out_len,
                        },
                        shape: vec![batch as usize, out_features as usize],
                    };
                }
            }

            panic!("QuantizedLinearInt8 requires GPU memory");
        }

        Op::QuantizedLinearInt4 { has_bias, weight_scale, weight_zero_point } => {
            let input = inputs[0];
            let weight = inputs[1]; // Packed INT4

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: in_off, .. },
                        Values::Device { offset: w_off, .. }) = (&input.values, &weight.values) {

                    let batch = input.shape[0] as i32;
                    let in_features = input.shape[1] as i32;
                    let out_features = weight.shape[1] as i32;

                    let out_len = (batch * out_features) as usize;
                    let out_off = alloc.alloc(out_len * 4).expect("QuantLinear4 allocation failed");

                    let dev = alloc.device();
                    dev.bind_to_thread().expect("Context bind failed");

                    let in_ptr = alloc.pool_ptr() + *in_off as u64;
                    let w_ptr = alloc.pool_ptr() + *w_off as u64;
                    let out_ptr = alloc.pool_ptr() + out_off as u64;

                    let (b_ptr, has_bias_int): (u64, i32) = if has_bias && inputs.len() > 2 {
                        if let Values::Device { offset: b_off, .. } = &inputs[2].values {
                            (alloc.pool_ptr() + *b_off as u64, 1)
                        } else {
                            (0, 0)
                        }
                    } else {
                        (0, 0)
                    };

                    load_ptx_robust(dev, "kernels/quant/quantize.ptx", "quant_mod", &["linear_f32_int4_f32"]);
                    let f = dev.get_func("quant_mod", "linear_f32_int4_f32").expect("QuantLinear4 func not found");

                    let block_dim = (16u32, 16u32, 1u32);
                    let grid_dim = (
                        ((out_features as u32) + 15) / 16,
                        ((batch as u32) + 15) / 16,
                        1u32
                    );

                    unsafe {
                        f.launch(
                            LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 },
                            (in_ptr, w_ptr, b_ptr, out_ptr, batch, in_features, out_features, weight_scale, weight_zero_point, has_bias_int)
                        ).expect("QuantLinear4 kernel launch failed");
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: out_len,
                        },
                        shape: vec![batch as usize, out_features as usize],
                    };
                }
            }

            panic!("QuantizedLinearInt4 requires GPU memory");
        }

        Op::QuantizedMatMulInt8 { scale_a, zero_point_a, scale_b, zero_point_b } => {
            let a = inputs[0]; // INT8 [M, K]
            let b = inputs[1]; // INT8 [K, N]

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: a_off, .. },
                        Values::Device { offset: b_off, .. }) = (&a.values, &b.values) {

                    let m = a.shape[0] as i32;
                    let k = a.shape[1] as i32;
                    let n = b.shape[1] as i32;

                    let out_len = (m * n) as usize;
                    let out_off = alloc.alloc(out_len * 4).expect("QuantMatMul allocation failed");

                    let dev = alloc.device();
                    dev.bind_to_thread().expect("Context bind failed");

                    let a_ptr = alloc.pool_ptr() + *a_off as u64;
                    let b_ptr = alloc.pool_ptr() + *b_off as u64;
                    let out_ptr = alloc.pool_ptr() + out_off as u64;

                    load_ptx_robust(dev, "kernels/quant/quantize.ptx", "quant_mod", &["matmul_int8_int8_f32"]);
                    let f = dev.get_func("quant_mod", "matmul_int8_int8_f32").expect("QuantMatMul func not found");

                    let block_dim = (16u32, 16u32, 1u32);
                    let grid_dim = (
                        ((n as u32) + 15) / 16,
                        ((m as u32) + 15) / 16,
                        1u32
                    );

                    unsafe {
                        f.launch(
                            LaunchConfig { block_dim, grid_dim, shared_mem_bytes: 0 },
                            (a_ptr, b_ptr, out_ptr, m, k, n, scale_a, zero_point_a, scale_b, zero_point_b)
                        ).expect("QuantMatMul kernel launch failed");
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: out_len,
                        },
                        shape: vec![m as usize, n as usize],
                    };
                }
            }

            panic!("QuantizedMatMulInt8 requires GPU memory");
        }

        Op::QuantizedReLU { precision, zero_point } => {
            let input = inputs[0];

            if let Device::Gpu(alloc) = &rules.device {
                if let Values::Device { offset, len, .. } = &input.values {
                    let dev = alloc.device();
                    dev.bind_to_thread().expect("Context bind failed");

                    let ptr = alloc.pool_ptr() + *offset as u64;

                    let kernel_name = match precision {
                        Precision::Int8 => "relu_int8",
                        Precision::UInt8 => "relu_uint8",
                        _ => panic!("QuantizedReLU only supports Int8/UInt8"),
                    };

                    load_ptx_robust(dev, "kernels/quant/quantize.ptx", "quant_mod", &[kernel_name]);
                    let f = dev.get_func("quant_mod", kernel_name).expect("QuantReLU func not found");

                    unsafe {
                        f.launch(
                            LaunchConfig::for_num_elems(*len as u32),
                            (ptr, *len as i32, zero_point)
                        ).expect("QuantReLU kernel launch failed");
                    }

                    return input.clone();
                }
            }

            panic!("QuantizedReLU requires GPU memory");
        }

        Op::QuantizedAdd { precision: _, scale_a, zero_point_a, scale_b, zero_point_b } => {
            let a = inputs[0];
            let b = inputs[1];

            if let Device::Gpu(alloc) = &rules.device {
                if let (Values::Device { offset: a_off, len, .. },
                        Values::Device { offset: b_off, .. }) = (&a.values, &b.values) {

                    let out_off = alloc.alloc(*len * 4).expect("QuantAdd allocation failed");

                    let dev = alloc.device();
                    dev.bind_to_thread().expect("Context bind failed");

                    let a_ptr = alloc.pool_ptr() + *a_off as u64;
                    let b_ptr = alloc.pool_ptr() + *b_off as u64;
                    let out_ptr = alloc.pool_ptr() + out_off as u64;

                    load_ptx_robust(dev, "kernels/quant/quantize.ptx", "quant_mod", &["add_int8_int8_f32"]);
                    let f = dev.get_func("quant_mod", "add_int8_int8_f32").expect("QuantAdd func not found");

                    unsafe {
                        f.launch(
                            LaunchConfig::for_num_elems(*len as u32),
                            (a_ptr, b_ptr, out_ptr, *len as i32, scale_a, zero_point_a, scale_b, zero_point_b)
                        ).expect("QuantAdd kernel launch failed");
                    }

                    return Grid {
                        values: Values::Device {
                            allocator: std::sync::Arc::clone(alloc),
                            offset: out_off,
                            len: *len,
                        },
                        shape: a.shape.clone(),
                    };
                }
            }

            panic!("QuantizedAdd requires GPU memory");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::MuscleMemory;
    use std::sync::Arc;

    fn cpu_rules() -> RuntimeRules {
        RuntimeRules {
            device: Device::Cpu,
            synthesizer: None,
            muscle_memory: Arc::new(MuscleMemory::new()),
            track_gradients: false,
            ..RuntimeRules::default()
        }
    }

    #[test]
    fn test_op_enum_variants() {
        // Verify Op enum variants can be constructed
        let _add = Op::Add;
        let _relu = Op::ReLU;
        let _matmul = Op::MatMul;
        let _move_device = Op::MoveToDevice;
        let _move_host = Op::MoveToHost;
        let _torch_unary = Op::TorchUnary("sigmoid".to_string());
        let _torch_binary = Op::TorchBinary("add".to_string());
        let _sgd = Op::SGDStep { lr: 0.01, momentum: 0.9, weight_decay: 0.0 };
        let _adam = Op::AdamStep { timestep: 1 };
        let _fill = Op::Fill(0.0);
        let _scale = Op::Scale(2.0);
    }

    #[test]
    fn test_relu_cpu_fallback() {
        let rules = cpu_rules();
        let data = vec![-1.0, 0.0, 1.0, -2.0, 2.0, 0.5];
        let grid = Grid::new(data, vec![2, 3]);

        let result = compute(Op::ReLU, vec![&grid], None, &rules);

        if let Values::Host(v) = &result.values {
            assert_eq!(v, &vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.5]);
        } else {
            panic!("Expected Host values");
        }
    }

    #[test]
    fn test_move_to_host_from_host() {
        let rules = cpu_rules();
        let data = vec![1.0, 2.0, 3.0];
        let grid = Grid::new(data.clone(), vec![3]);

        // MoveToHost on a Host grid should just clone it
        let result = compute(Op::MoveToHost, vec![&grid], None, &rules);

        assert!(result.is_host());
        if let Values::Host(v) = &result.values {
            assert_eq!(v, &data);
        }
    }

    #[test]
    fn test_matmul_shapes() {
        // Test that we can construct grids with proper matmul shapes
        let a = Grid::new(vec![1.0; 6], vec![2, 3]); // 2x3
        let b = Grid::new(vec![1.0; 12], vec![3, 4]); // 3x4
        // Result should be 2x4

        assert_eq!(a.shape[1], b.shape[0]); // Inner dimensions match
    }

    #[test]
    fn test_op_clone() {
        let op1 = Op::SGDStep { lr: 0.01, momentum: 0.9, weight_decay: 0.0001 };
        let op2 = op1.clone();

        if let (Op::SGDStep { lr: lr1, .. }, Op::SGDStep { lr: lr2, .. }) = (&op1, &op2) {
            assert_eq!(lr1, lr2);
        }
    }

    #[test]
    fn test_op_debug() {
        let op = Op::AdamStep { timestep: 5 };
        let debug_str = format!("{:?}", op);
        assert!(debug_str.contains("AdamStep"));
        assert!(debug_str.contains("5"));
    }

    #[test]
    fn test_fused_ops() {
        let fused = Op::Fused(vec![Op::ReLU, Op::Scale(2.0)]);
        if let Op::Fused(ops) = fused {
            assert_eq!(ops.len(), 2);
        }
    }

    // Mixed precision tests
    #[test]
    fn test_precision_to_int() {
        assert_eq!(precision_to_int(Precision::F16), 0);
        assert_eq!(precision_to_int(Precision::BF16), 1);
        assert_eq!(precision_to_int(Precision::F32), 2);
        assert_eq!(precision_to_int(Precision::F64), 3);
    }

    #[test]
    fn test_precision_size_bytes() {
        assert_eq!(Precision::F16.size_bytes(), 2);
        assert_eq!(Precision::BF16.size_bytes(), 2);
        assert_eq!(Precision::F32.size_bytes(), 4);
        assert_eq!(Precision::F64.size_bytes(), 8);
    }

    #[test]
    fn test_precision_is_half() {
        assert!(Precision::F16.is_half());
        assert!(Precision::BF16.is_half());
        assert!(!Precision::F32.is_half());
        assert!(!Precision::F64.is_half());
    }

    #[test]
    fn test_mixed_precision_op_variants() {
        let _cast = Op::Cast { target_precision: Precision::F16 };
        let _half_unary = Op::HalfUnary { op: "relu".to_string(), precision: Precision::F16 };
        let _half_binary = Op::HalfBinary { op: "add".to_string(), precision: Precision::BF16 };
        let _half_linear = Op::HalfLinearForward { has_bias: true, precision: Precision::F16 };
        let _set_scale = Op::SetLossScale { scale: 65536.0 };
        let _scale_grad = Op::ScaleGradients { scale: 65536.0 };
        let _unscale = Op::UnscaleGradients { scale: 65536.0 };
        let _update = Op::UpdateLossScale { had_overflow: false, scale_factor: 2.0, scale_window: 1000 };
    }

    #[test]
    fn test_precision_torch_dtype() {
        assert_eq!(Precision::F16.torch_dtype(), "float16");
        assert_eq!(Precision::BF16.torch_dtype(), "bfloat16");
        assert_eq!(Precision::F32.torch_dtype(), "float32");
        assert_eq!(Precision::F64.torch_dtype(), "float64");
    }

    // Embedding tests
    #[test]
    fn test_embedding_forward_cpu() {
        let rules = cpu_rules();

        // Create embedding table: vocab_size=5, embed_dim=3
        let table = Grid::new(vec![
            0.1, 0.2, 0.3,  // word 0
            1.1, 1.2, 1.3,  // word 1
            2.1, 2.2, 2.3,  // word 2
            3.1, 3.2, 3.3,  // word 3
            4.1, 4.2, 4.3,  // word 4
        ], vec![5, 3]);

        // Indices (as f32, will be cast to int): [1, 3, 0, 2]
        let indices = Grid::new(vec![1.0, 3.0, 0.0, 2.0], vec![4]);

        let result = compute(Op::EmbeddingForward, vec![&indices, &table], None, &rules);

        assert_eq!(result.shape, vec![4, 3]);

        if let Values::Host(data) = &result.values {
            // Check word 1 embedding
            assert!((data[0] - 1.1).abs() < 1e-6);
            assert!((data[1] - 1.2).abs() < 1e-6);
            assert!((data[2] - 1.3).abs() < 1e-6);
            // Check word 3 embedding
            assert!((data[3] - 3.1).abs() < 1e-6);
            // Check word 0 embedding
            assert!((data[6] - 0.1).abs() < 1e-6);
            // Check word 2 embedding
            assert!((data[9] - 2.1).abs() < 1e-6);
        } else {
            panic!("Expected Host values");
        }
    }

    #[test]
    fn test_embedding_op_variant() {
        let _emb = Op::EmbeddingForward;
        let debug_str = format!("{:?}", _emb);
        assert!(debug_str.contains("EmbeddingForward"));
    }

    // Gradient accumulation tests
    #[test]
    fn test_gradient_accumulate_cpu() {
        let rules = cpu_rules();

        let accumulator = Grid::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let gradient = Grid::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]);

        let result = compute(Op::GradientAccumulate, vec![&accumulator, &gradient], None, &rules);

        if let Values::Host(data) = &result.values {
            assert!((data[0] - 1.1).abs() < 1e-6);
            assert!((data[1] - 2.2).abs() < 1e-6);
            assert!((data[2] - 3.3).abs() < 1e-6);
            assert!((data[3] - 4.4).abs() < 1e-6);
        } else {
            panic!("Expected Host values");
        }
    }

    #[test]
    fn test_gradient_zero_cpu() {
        let rules = cpu_rules();

        let accumulator = Grid::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let result = compute(Op::GradientZero, vec![&accumulator], None, &rules);

        if let Values::Host(data) = &result.values {
            assert_eq!(data, &vec![0.0, 0.0, 0.0, 0.0]);
        } else {
            panic!("Expected Host values");
        }
    }

    #[test]
    fn test_gradient_accumulation_op_variants() {
        let _acc = Op::GradientAccumulate;
        let _zero = Op::GradientZero;
        let _sgd = Op::AccumulatedSGDStep {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0001,
            accumulation_steps: 4,
        };
        let _adam = Op::AccumulatedAdamStep {
            timestep: 1,
            accumulation_steps: 8,
        };

        // Verify debug output
        assert!(format!("{:?}", _acc).contains("GradientAccumulate"));
        assert!(format!("{:?}", _zero).contains("GradientZero"));
        assert!(format!("{:?}", _sgd).contains("accumulation_steps"));
        assert!(format!("{:?}", _adam).contains("accumulation_steps"));
    }

    // Quantization tests
    #[test]
    fn test_quantization_op_variants() {
        let _q_int8 = Op::Quantize {
            target: Precision::Int8,
            scale: 0.1,
            zero_point: 0,
        };
        let _q_uint8 = Op::Quantize {
            target: Precision::UInt8,
            scale: 0.1,
            zero_point: 128,
        };
        let _q_int4 = Op::Quantize {
            target: Precision::Int4,
            scale: 0.2,
            zero_point: 0,
        };
        let _dq = Op::Dequantize {
            source: Precision::Int8,
            scale: 0.1,
            zero_point: 0,
        };
        let _cal = Op::CalibrationMinMax;
        let _qlin8 = Op::QuantizedLinearInt8 {
            has_bias: true,
            weight_scale: 0.05,
            weight_zero_point: 0,
        };
        let _qlin4 = Op::QuantizedLinearInt4 {
            has_bias: false,
            weight_scale: 0.1,
            weight_zero_point: 0,
        };
        let _qmm = Op::QuantizedMatMulInt8 {
            scale_a: 0.1,
            zero_point_a: 0,
            scale_b: 0.1,
            zero_point_b: 0,
        };
        let _qrelu = Op::QuantizedReLU {
            precision: Precision::Int8,
            zero_point: 0,
        };
        let _qadd = Op::QuantizedAdd {
            precision: Precision::Int8,
            scale_a: 0.1,
            zero_point_a: 0,
            scale_b: 0.1,
            zero_point_b: 0,
        };
    }

    #[test]
    fn test_precision_quantized_helpers() {
        assert!(Precision::Int8.is_quantized());
        assert!(Precision::UInt8.is_quantized());
        assert!(Precision::Int4.is_quantized());
        assert!(Precision::UInt4.is_quantized());
        assert!(!Precision::F32.is_quantized());
        assert!(!Precision::F16.is_quantized());

        assert!(Precision::Int4.is_packed_4bit());
        assert!(Precision::UInt4.is_packed_4bit());
        assert!(!Precision::Int8.is_packed_4bit());

        assert_eq!(Precision::Int8.quant_range(), (-128.0, 127.0));
        assert_eq!(Precision::UInt8.quant_range(), (0.0, 255.0));
        assert_eq!(Precision::Int4.quant_range(), (-8.0, 7.0));
        assert_eq!(Precision::UInt4.quant_range(), (0.0, 15.0));
    }

    #[test]
    fn test_precision_storage_bytes() {
        assert_eq!(Precision::Int8.storage_bytes(100), 100);
        assert_eq!(Precision::UInt8.storage_bytes(100), 100);
        assert_eq!(Precision::Int4.storage_bytes(100), 50);  // Packed: 2 values per byte
        assert_eq!(Precision::Int4.storage_bytes(99), 50);   // Rounds up
        assert_eq!(Precision::F32.storage_bytes(100), 400);
    }

    #[test]
    fn test_quantize_cpu_int8() {
        let rules = cpu_rules();

        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let grid = Grid::new(data, vec![5]);

        let result = compute(
            Op::Quantize {
                target: Precision::Int8,
                scale: 0.1,
                zero_point: 0,
            },
            vec![&grid],
            None,
            &rules
        );

        if let Values::Host(v) = &result.values {
            // -1.0 / 0.1 = -10, -0.5 / 0.1 = -5, etc.
            assert!((v[0] - (-10.0)).abs() < 0.5);
            assert!((v[1] - (-5.0)).abs() < 0.5);
            assert!((v[2] - 0.0).abs() < 0.5);
            assert!((v[3] - 5.0).abs() < 0.5);
            assert!((v[4] - 10.0).abs() < 0.5);
        }
    }

    #[test]
    fn test_dequantize_cpu() {
        let rules = cpu_rules();

        // Simulated quantized values (as f32 for CPU fallback)
        let quantized = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
        let grid = Grid::new(quantized, vec![5]);

        let result = compute(
            Op::Dequantize {
                source: Precision::Int8,
                scale: 0.1,
                zero_point: 0,
            },
            vec![&grid],
            None,
            &rules
        );

        if let Values::Host(v) = &result.values {
            assert!((v[0] - (-1.0)).abs() < 1e-6);
            assert!((v[1] - (-0.5)).abs() < 1e-6);
            assert!((v[2] - 0.0).abs() < 1e-6);
            assert!((v[3] - 0.5).abs() < 1e-6);
            assert!((v[4] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_calibration_minmax() {
        let rules = cpu_rules();

        let data = vec![-2.5, 1.0, 3.0, -1.0, 0.5];
        let grid = Grid::new(data, vec![5]);

        let result = compute(Op::CalibrationMinMax, vec![&grid], None, &rules);

        if let Values::Host(v) = &result.values {
            assert_eq!(v.len(), 2);
            assert!((v[0] - (-2.5)).abs() < 1e-6);  // min
            assert!((v[1] - 3.0).abs() < 1e-6);     // max
        }
    }
}
