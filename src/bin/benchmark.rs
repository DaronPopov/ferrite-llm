// Benchmark: Compare torch bridge vs spcpp/cuBLAS backend
//
// Tests matmul performance at various sizes

use ferrite::compute::spcpp;
use ferrite::dynamics::allocator::TlsfAllocator;
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("=== Ferrite Backend Benchmark ===\n");

    // Initialize CUDA
    let device = CudaDevice::new(0).expect("No CUDA device");
    let allocator = Arc::new(TlsfAllocator::new(device.clone(), 512 * 1024 * 1024));
    let pool_base = allocator.pool_ptr();

    // Test sizes
    let sizes = vec![
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ];

    let warmup_iters = 5;
    let bench_iters = 20;

    // Try to load spcpp
    let spcpp_available = spcpp::load_spcpp();
    if spcpp_available {
        let kernel_path = "kernels/external/spcpp/ops.cu";
        if let Err(e) = spcpp::init(kernel_path) {
            println!("Warning: spcpp init failed: {}", e);
        } else {
            println!("spcpp backend: LOADED\n");
        }
    } else {
        println!("spcpp backend: NOT AVAILABLE\n");
    }

    // Load torch bridge
    let torch_lib = unsafe { libloading::Library::new("kernels/external/torch/libtorch_bridge.so") };
    let torch_available = torch_lib.is_ok();
    println!("torch backend: {}\n", if torch_available { "LOADED" } else { "NOT AVAILABLE" });

    if !torch_available && !spcpp_available {
        println!("No backends available! Build at least one backend first.");
        return;
    }

    println!("{:>10} {:>10} {:>10} | {:>12} {:>12} | {:>10}",
             "M", "K", "N", "Torch (ms)", "SPCPP (ms)", "Speedup");
    println!("{}", "-".repeat(75));

    for (m, k, n) in sizes {
        let size_a = m * k;
        let size_b = k * n;
        let size_c = m * n;

        // Allocate GPU memory
        let offset_a = allocator.alloc(size_a * 4).unwrap();
        let offset_b = allocator.alloc(size_b * 4).unwrap();
        let offset_c = allocator.alloc(size_c * 4).unwrap();

        let ptr_a = (pool_base + offset_a as u64) as *mut f32;
        let ptr_b = (pool_base + offset_b as u64) as *mut f32;
        let ptr_c = (pool_base + offset_c as u64) as *mut f32;

        // Initialize with data
        let data_a: Vec<f32> = (0..size_a).map(|i| (i as f32 * 0.001) % 1.0).collect();
        let data_b: Vec<f32> = (0..size_b).map(|i| (i as f32 * 0.002) % 1.0).collect();
        allocator.copy_to_offset(offset_a, &data_a);
        allocator.copy_to_offset(offset_b, &data_b);

        let mut torch_time = f64::NAN;
        let mut spcpp_time = f64::NAN;

        // Benchmark torch bridge
        // torch_matmul(a, a_rows, a_cols, b, b_rows, b_cols, c, trans_a, trans_b)
        if torch_available {
            if let Ok(ref lib) = torch_lib {
                unsafe {
                    type MatMulFn = unsafe extern "C" fn(*mut f32, i32, i32, *mut f32, i32, i32, *mut f32, bool, bool);
                    if let Ok(matmul) = lib.get::<MatMulFn>(b"torch_matmul") {
                        // Warmup
                        for _ in 0..warmup_iters {
                            matmul(ptr_a, m as i32, k as i32, ptr_b, k as i32, n as i32, ptr_c, false, false);
                        }
                        spcpp::sync();

                        // Benchmark
                        let start = Instant::now();
                        for _ in 0..bench_iters {
                            matmul(ptr_a, m as i32, k as i32, ptr_b, k as i32, n as i32, ptr_c, false, false);
                        }
                        spcpp::sync();
                        torch_time = start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
                    }
                }
            }
        }

        // Benchmark spcpp/cuBLAS
        if spcpp_available {
            // Warmup
            for _ in 0..warmup_iters {
                spcpp::matmul(ptr_a, ptr_b, ptr_c, m as i32, k as i32, n as i32);
            }
            spcpp::sync();

            // Benchmark
            let start = Instant::now();
            for _ in 0..bench_iters {
                spcpp::matmul(ptr_a, ptr_b, ptr_c, m as i32, k as i32, n as i32);
            }
            spcpp::sync();
            spcpp_time = start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
        }

        // Calculate speedup
        let speedup = if !torch_time.is_nan() && !spcpp_time.is_nan() && spcpp_time > 0.0 {
            torch_time / spcpp_time
        } else {
            f64::NAN
        };

        println!("{:>10} {:>10} {:>10} | {:>12} {:>12} | {:>10}",
                 m, k, n,
                 if torch_time.is_nan() { "N/A".to_string() } else { format!("{:.3}", torch_time) },
                 if spcpp_time.is_nan() { "N/A".to_string() } else { format!("{:.3}", spcpp_time) },
                 if speedup.is_nan() { "N/A".to_string() } else { format!("{:.2}x", speedup) });
    }

    println!("\n=== Element-wise Operations (ReLU) ===\n");

    let elem_sizes = vec![100_000, 1_000_000, 10_000_000];

    println!("{:>12} | {:>12} {:>12} | {:>10}",
             "Elements", "Torch (ms)", "SPCPP (ms)", "Speedup");
    println!("{}", "-".repeat(55));

    for size in elem_sizes {
        let offset = allocator.alloc(size * 4).unwrap();
        let ptr = (pool_base + offset as u64) as *mut f32;

        let data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.001) - 0.5).collect();
        allocator.copy_to_offset(offset, &data);

        let mut torch_time = f64::NAN;
        let mut spcpp_time = f64::NAN;

        // Benchmark torch unary (relu)
        if torch_available {
            if let Ok(ref lib) = torch_lib {
                unsafe {
                    type UnaryFn = unsafe extern "C" fn(*const i8, *mut f32, i32);
                    if let Ok(unary) = lib.get::<UnaryFn>(b"torch_bridge_unary") {
                        let op = std::ffi::CString::new("relu").unwrap();

                        // Reset data and warmup
                        for _ in 0..warmup_iters {
                            allocator.copy_to_offset(offset, &data);
                            unary(op.as_ptr(), ptr, size as i32);
                        }
                        spcpp::sync();

                        // Benchmark
                        let start = Instant::now();
                        for _ in 0..bench_iters {
                            allocator.copy_to_offset(offset, &data);
                            unary(op.as_ptr(), ptr, size as i32);
                        }
                        spcpp::sync();
                        torch_time = start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
                    }
                }
            }
        }

        // Benchmark spcpp unary (relu)
        if spcpp_available {
            // Warmup
            for _ in 0..warmup_iters {
                allocator.copy_to_offset(offset, &data);
                spcpp::unary("relu", ptr, size as i32);
            }
            spcpp::sync();

            // Benchmark
            let start = Instant::now();
            for _ in 0..bench_iters {
                allocator.copy_to_offset(offset, &data);
                spcpp::unary("relu", ptr, size as i32);
            }
            spcpp::sync();
            spcpp_time = start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
        }

        let speedup = if !torch_time.is_nan() && !spcpp_time.is_nan() && spcpp_time > 0.0 {
            torch_time / spcpp_time
        } else {
            f64::NAN
        };

        println!("{:>12} | {:>12} {:>12} | {:>10}",
                 size,
                 if torch_time.is_nan() { "N/A".to_string() } else { format!("{:.3}", torch_time) },
                 if spcpp_time.is_nan() { "N/A".to_string() } else { format!("{:.3}", spcpp_time) },
                 if speedup.is_nan() { "N/A".to_string() } else { format!("{:.2}x", speedup) });
    }

    // Cleanup
    if spcpp_available {
        spcpp::shutdown();
    }

    println!("\nBenchmark complete.");
}
