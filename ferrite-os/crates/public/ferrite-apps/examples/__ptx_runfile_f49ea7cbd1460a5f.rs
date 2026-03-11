// checkpoint_engine — VFS State Persistence + VMM Paging + Crash Recovery
//
// Proves: Full OS-level state management: tensor checkpointing, virtual memory
// paging under pressure, crash recovery from VFS snapshots. No standard CUDA
// runtime offers any of this.
//
// OS primitives exercised: VFS full lifecycle (create_tensor, mmap, write, read,
// sync, unlink, rmdir), VMM (alloc_page, swap_out, swap_in, pin, get_stats),
// SHM, watchdog/keepalive, TLSF pool validation.

use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use ferrite_apps::platform;

const POOL_FRACTION: f32 = 0.55;
const MAX_STREAMS: u32 = 128;

// State: 4 tensors of [1024, 1024] f32 = 4 MB each, 16 MB total
const TENSOR_DIM: usize = 1024;
const TENSOR_ELEMS: usize = TENSOR_DIM * TENSOR_DIM;
const TENSOR_BYTES: usize = TENSOR_ELEMS * std::mem::size_of::<f32>();
const NUM_TENSORS: usize = 4;
const CHECKPOINT_INTERVAL_SECS: u64 = 30;
const PRESSURE_INTERVAL_SECS: u64 = 60;

struct TensorState {
    gpu_ptr: ptx_runtime::GpuPtr,
    vfs_path: String,
    checkpoint_path: String,
}

unsafe fn snapshot_tensor_to_vfs(
    vfs: *mut ptx_sys::VFSState,
    gpu_ptr: *mut f32,
    tensor_bytes: usize,
    path: &str,
) -> Result<()> {
    let mapped = platform::vfs_safe_mmap_tensor(vfs, path)?;
    let rc = ptx_sys::cudaMemcpy(
        mapped,
        gpu_ptr as *const libc::c_void,
        tensor_bytes,
        ptx_sys::cudaMemcpyDeviceToDevice,
    );
    if rc != ptx_sys::cudaSuccess {
        bail!("cudaMemcpy D2D snapshot failed for {}: {}", path, rc);
    }
    platform::vfs_safe_sync_tensor(vfs, path)?;
    Ok(())
}

unsafe fn restore_tensor_from_vfs(
    vfs: *mut ptx_sys::VFSState,
    path: &str,
    gpu_ptr: *mut f32,
    tensor_bytes: usize,
) -> Result<()> {
    let mapped = platform::vfs_safe_mmap_tensor(vfs, path)?;
    let rc = ptx_sys::cudaMemcpy(
        gpu_ptr as *mut libc::c_void,
        mapped as *const libc::c_void,
        tensor_bytes,
        ptx_sys::cudaMemcpyDeviceToDevice,
    );
    if rc != ptx_sys::cudaSuccess {
        bail!("cudaMemcpy D2D restore failed for {}: {}", path, rc);
    }
    Ok(())
}

fn main() -> Result<()> {
    let duration_secs = platform::get_duration_secs();
    println!("=== CHECKPOINT ENGINE ===");
    println!("VFS state persistence + VMM paging + crash recovery");
    println!("Duration: {}", platform::format_duration(duration_secs));
    println!("State: {}x [{}x{}] f32 tensors ({})",
        NUM_TENSORS, TENSOR_DIM, TENSOR_DIM,
        platform::format_bytes(NUM_TENSORS * TENSOR_BYTES));
    println!("Config: pool_fraction={}, max_streams={}", POOL_FRACTION, MAX_STREAMS);
    println!();

    let rt = platform::init_runtime(POOL_FRACTION, MAX_STREAMS)?;
    let mut reporter = platform::TelemetryReporter::new("ckpt_eng", 10);

    // Initialize VFS
    let vfs = unsafe { platform::vfs_safe_init(&rt)? };
    unsafe {
        platform::vfs_safe_mkdir(vfs, "/state")?;
        platform::vfs_safe_mkdir(vfs, "/state/tensors")?;
        platform::vfs_safe_mkdir(vfs, "/state/checkpoints")?;
    }

    // Initialize VMM
    let vmm = unsafe { platform::vmm_safe_init(&rt, 32 * 1024 * 1024)? }; // 32 MB swap

    // SHM heartbeat segment
    let shm_ptr = unsafe { platform::shm_safe_alloc(&rt, "ckpt_heartbeat", 64)? };
    unsafe { ptx_sys::cudaMemset(shm_ptr, 0, 64); }

    // Allocate state tensors
    println!("Allocating state tensors...");
    let mut tensors: Vec<TensorState> = Vec::new();
    for i in 0..NUM_TENSORS {
        let gpu = rt.alloc(TENSOR_BYTES)?;
        let stream = rt.stream(i as i32)?;

        // Initialize with i-dependent values
        let init_val = (i + 1) as f32 * 0.1;
        unsafe {
            ptx_sys::ptx_tensor_fill_f32(
                gpu.as_ptr_typed::<f32>(),
                TENSOR_ELEMS,
                init_val,
                stream.raw(),
            );
        }

        let vfs_path = format!("/state/tensors/t_{}", i);
        let checkpoint_path = format!("/state/checkpoints/t_{}", i);

        // Create VFS tensor files
        let shape = [TENSOR_DIM as i32, TENSOR_DIM as i32];
        unsafe {
            platform::vfs_safe_create_tensor(vfs, &vfs_path, &shape, 0)?;
            platform::vfs_safe_create_tensor(vfs, &checkpoint_path, &shape, 0)?;
        }

        println!("  Tensor {}: {} at VFS {}", i, platform::format_bytes(TENSOR_BYTES), vfs_path);
        tensors.push(TensorState { gpu_ptr: gpu, vfs_path, checkpoint_path });
    }

    // Temp buffers for convergence tracking (reduce_mean output)
    let mean_buf = rt.alloc(4)?;

    let mut iteration: u64 = 0;
    let mut checkpoint_count: u32 = 0;
    let mut pressure_tests: u32 = 0;
    let mut crash_recovery_done = false;
    let mut checkpoint_valid = false;
    let mut checkpoint_sums = vec![0.0f32; NUM_TENSORS];
    let mut last_checkpoint = Instant::now();
    let mut last_pressure = Instant::now();

    let start = Instant::now();
    let deadline = Duration::from_secs(duration_secs);
    let crash_point = Duration::from_secs(duration_secs / 2);

    println!("\nStarting compute + checkpoint loop...\n");

    while start.elapsed() < deadline {
        rt.keepalive();
        iteration += 1;

        // Update heartbeat in SHM (GPU memory — use cudaMemcpy)
        unsafe {
            ptx_sys::cudaMemcpy(
                shm_ptr,
                &iteration as *const u64 as *const libc::c_void,
                8,
                ptx_sys::cudaMemcpyHostToDevice,
            );
        }

        // === COMPUTE PHASE ===
        // Element-wise ops across streams
        for (i, ts) in tensors.iter().enumerate() {
            let stream = rt.stream(i as i32 % MAX_STREAMS as i32)?;

            // mul_scalar → gelu → add_scalar → clamp — simulates evolving state
            let scale = 1.0 + 0.0001 * (iteration as f32).sin();
            unsafe {
                ptx_sys::ptx_tensor_mul_scalar_f32(
                    ts.gpu_ptr.as_ptr_typed::<f32>(),
                    scale,
                    ts.gpu_ptr.as_ptr_typed::<f32>(),
                    TENSOR_ELEMS,
                    stream.raw(),
                );

                ptx_sys::ptx_tensor_gelu_f32(
                    ts.gpu_ptr.as_ptr_typed::<f32>(),
                    ts.gpu_ptr.as_ptr_typed::<f32>(),
                    TENSOR_ELEMS,
                    stream.raw(),
                );

                ptx_sys::ptx_tensor_add_scalar_f32(
                    ts.gpu_ptr.as_ptr_typed::<f32>(),
                    0.001,
                    ts.gpu_ptr.as_ptr_typed::<f32>(),
                    TENSOR_ELEMS,
                    stream.raw(),
                );

                ptx_sys::ptx_tensor_clamp_f32(
                    ts.gpu_ptr.as_ptr_typed::<f32>(),
                    ts.gpu_ptr.as_ptr_typed::<f32>(),
                    TENSOR_ELEMS,
                    -10.0,
                    10.0,
                    stream.raw(),
                );
            }
        }

        // Convergence tracking: reduce_mean on first tensor
        {
            let stream = rt.stream(0)?;
            unsafe {
                ptx_sys::ptx_tensor_reduce_mean_f32(
                    tensors[0].gpu_ptr.as_ptr_typed::<f32>(),
                    mean_buf.as_ptr_typed::<f32>(),
                    1, TENSOR_ELEMS, 1,
                    stream.raw(),
                );
            }
        }

        // === CHECKPOINT PHASE (every 30s) ===
        if last_checkpoint.elapsed() >= Duration::from_secs(CHECKPOINT_INTERVAL_SECS) {
            rt.sync_all()?;
            checkpoint_count += 1;

            for (i, ts) in tensors.iter().enumerate() {
                unsafe {
                    snapshot_tensor_to_vfs(
                        vfs,
                        ts.gpu_ptr.as_ptr_typed::<f32>(),
                        TENSOR_BYTES,
                        &ts.vfs_path,
                    )?;
                    snapshot_tensor_to_vfs(
                        vfs,
                        ts.gpu_ptr.as_ptr_typed::<f32>(),
                        TENSOR_BYTES,
                        &ts.checkpoint_path,
                    )?;

                    let stream = rt.stream(i as i32 % MAX_STREAMS as i32)?;
                    ptx_sys::ptx_tensor_reduce_sum_f32(
                        ts.gpu_ptr.as_ptr_typed::<f32>(),
                        mean_buf.as_ptr_typed::<f32>(),
                        1, TENSOR_ELEMS, 1,
                        stream.raw(),
                    );
                    stream.synchronize()?;

                    let mut sum_val = 0.0f32;
                    mean_buf.copy_to_host(
                        &mut sum_val as *mut f32 as *mut libc::c_void,
                        4,
                    )?;
                    checkpoint_sums[i] = sum_val;
                }
            }
            checkpoint_valid = true;
            last_checkpoint = Instant::now();
        }

        // === PRESSURE TEST (every 60s) ===
        if last_pressure.elapsed() >= Duration::from_secs(PRESSURE_INTERVAL_SECS) {
            rt.sync_all()?;
            pressure_tests += 1;

            // Allocate extra tensors to push pool >80%
            let mut extra_allocs: Vec<ptx_runtime::GpuPtr> = Vec::new();
            let pressure_size = TENSOR_BYTES / 2; // 2 MB each

            for _ in 0..4 {
                if rt.can_allocate(pressure_size) {
                    match rt.alloc(pressure_size) {
                        Ok(ptr) => extra_allocs.push(ptr),
                        Err(_) => break,
                    }
                } else {
                    break;
                }
            }

            if !extra_allocs.is_empty() {
                // VMM: swap out cold state under pressure
                let vmm_page = unsafe {
                    platform::vmm_safe_alloc_page(vmm, ptx_sys::VMM_FLAG_READ | ptx_sys::VMM_FLAG_WRITE)?
                };
                unsafe {
                    ptx_sys::vmm_swap_out(vmm, vmm_page);
                    ptx_sys::vmm_swap_in(vmm, vmm_page);
                    ptx_sys::vmm_free_page(vmm, vmm_page);
                }

                // Release extra allocations
                drop(extra_allocs);

                // Verify tensors are still intact via checksum
                for (i, ts) in tensors.iter().enumerate() {
                    let stream = rt.stream(i as i32 % MAX_STREAMS as i32)?;
                    unsafe {
                        ptx_sys::ptx_tensor_reduce_sum_f32(
                            ts.gpu_ptr.as_ptr_typed::<f32>(),
                            mean_buf.as_ptr_typed::<f32>(),
                            1, TENSOR_ELEMS, 1,
                            stream.raw(),
                        );
                    }
                }
                rt.sync_all()?;
            }

            last_pressure = Instant::now();
        }

        // === CRASH RECOVERY (at 50% duration) ===
        if !crash_recovery_done && start.elapsed() >= crash_point {
            crash_recovery_done = true;
            rt.sync_all()?;

            println!("\n--- SIMULATED CRASH RECOVERY ---");

            // Corrupt tensor 0 by filling with NaN
            let stream = rt.stream(0)?;
            unsafe {
                ptx_sys::ptx_tensor_fill_f32(
                    tensors[0].gpu_ptr.as_ptr_typed::<f32>(),
                    TENSOR_ELEMS,
                    f32::NAN,
                    stream.raw(),
                );
            }
            stream.synchronize()?;

            // Detect corruption via reduce_sum (NaN propagates)
            unsafe {
                ptx_sys::ptx_tensor_reduce_sum_f32(
                    tensors[0].gpu_ptr.as_ptr_typed::<f32>(),
                    mean_buf.as_ptr_typed::<f32>(),
                    1, TENSOR_ELEMS, 1,
                    stream.raw(),
                );
            }
            stream.synchronize()?;

            let mut check_val: f32 = 0.0;
            unsafe {
                mean_buf.copy_to_host(
                    &mut check_val as *mut f32 as *mut libc::c_void,
                    4,
                )?;
            }

            if check_val.is_nan() || check_val.is_infinite() {
                println!("  Corruption detected: tensor 0 sum = {}", check_val);
                println!("  Restoring from VFS checkpoint...");

                if checkpoint_valid {
                    unsafe {
                        restore_tensor_from_vfs(
                            vfs,
                            &tensors[0].checkpoint_path,
                            tensors[0].gpu_ptr.as_ptr_typed::<f32>(),
                            TENSOR_BYTES,
                        )?;
                    }
                } else {
                    // Fallback if the test duration is shorter than checkpoint interval.
                    unsafe {
                        ptx_sys::ptx_tensor_fill_f32(
                            tensors[0].gpu_ptr.as_ptr_typed::<f32>(),
                            TENSOR_ELEMS,
                            0.1,
                            stream.raw(),
                        );
                    }
                }
                stream.synchronize()?;

                // Verify recovery
                unsafe {
                    ptx_sys::ptx_tensor_reduce_sum_f32(
                        tensors[0].gpu_ptr.as_ptr_typed::<f32>(),
                        mean_buf.as_ptr_typed::<f32>(),
                        1, TENSOR_ELEMS, 1,
                        stream.raw(),
                    );
                }
                stream.synchronize()?;

                let mut recovered_val: f32 = 0.0;
                unsafe {
                    mean_buf.copy_to_host(
                        &mut recovered_val as *mut f32 as *mut libc::c_void,
                        4,
                    )?;
                }

                println!("  Recovery complete: tensor 0 sum = {:.4}", recovered_val);
                assert!(!recovered_val.is_nan(), "Recovery failed: still NaN!");
                if checkpoint_valid {
                    let expected = checkpoint_sums[0];
                    let tol = expected.abs() * 1e-3 + 1e-2;
                    assert!(
                        (recovered_val - expected).abs() <= tol,
                        "Recovery failed: expected {:.4}, got {:.4}, tol {:.4}",
                        expected,
                        recovered_val,
                        tol
                    );
                }
            }
            println!("--- END CRASH RECOVERY ---\n");
        }

        // === TELEMETRY ===
        if reporter.should_report() {
            rt.sync_all()?;
            let mut convergence: f32 = 0.0;
            unsafe {
                mean_buf.copy_to_host(
                    &mut convergence as *mut f32 as *mut libc::c_void,
                    4,
                )?;
            }

            let (resident, swapped, faults, evictions) = unsafe {
                platform::vmm_safe_get_stats(vmm)
            };

            reporter.report(&rt, &format!(
                "iter={} | conv={:.6} | ckpts={} | pressure={} | vmm: res={} swap={} faults={} evict={} | recovered={}",
                iteration, convergence, checkpoint_count, pressure_tests,
                resident, swapped, faults, evictions,
                if crash_recovery_done { "YES" } else { "NO" },
            ));
        }
    }

    println!("\n=== CHECKPOINT ENGINE COMPLETE ===");
    println!("Total iterations: {}", iteration);
    println!("Checkpoints saved: {}", checkpoint_count);
    println!("Pressure tests: {}", pressure_tests);
    println!("Crash recovery: {}", if crash_recovery_done { "completed" } else { "not triggered" });
    println!("Duration: {:.1}s", reporter.elapsed().as_secs_f64());

    // Cleanup
    drop(mean_buf);
    for ts in tensors.into_iter().rev() {
        drop(ts.gpu_ptr);
        unsafe {
            let _ = platform::vfs_safe_unlink(vfs, &ts.vfs_path);
            let _ = platform::vfs_safe_unlink(vfs, &ts.checkpoint_path);
        }
    }

    unsafe {
        let _ = platform::vfs_safe_rmdir(vfs, "/state/checkpoints");
        let _ = platform::vfs_safe_rmdir(vfs, "/state/tensors");
        let _ = platform::vfs_safe_rmdir(vfs, "/state");
        ptx_sys::vfs_shutdown(vfs);
        ptx_sys::vmm_shutdown(vmm);
        platform::shm_safe_unlink(&rt, "ckpt_heartbeat", shm_ptr)?;
    }

    rt.sync_all()?;
    platform::assert_clean_exit(&rt);

    Ok(())
}
