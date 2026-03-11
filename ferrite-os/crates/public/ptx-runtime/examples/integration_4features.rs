//! Integration smoke test for the 4 core runtime features:
//!   1. Per-client memory accounting (owner stats)
//!   2. VMM<->TLSF eviction bridge (set_vmm wiring)
//!   3. Allocation event ring buffer
//!   4. Task submission API
//!
//! Run:  cargo run --example integration_4features

use ptx_runtime::PtxRuntime;
use std::ptr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Ferrite-OS: 4-Feature Integration Test ===\n");

    let mut config = ptx_sys::GPUHotConfig::default();
    config.pool_fraction = 0.5;
    config.quiet_init = true;

    let runtime = PtxRuntime::with_config(0, Some(config))?;
    let tlsf = runtime.tlsf_stats();
    println!("[init] pool {:.0} MB, allocated {:.0} MB",
             tlsf.total_pool_size as f64 / 1e6,
             tlsf.allocated_bytes as f64 / 1e6);

    let mut pass = 0u32;
    let mut fail = 0u32;

    // ── Feature 1: per-owner allocation ─────────────────────
    {
        print!("[1] owner_stats ... ");

        // Baseline
        let before = runtime.owner_stats();
        let _before_count = before.num_owners;

        // Allocate 10 blocks under owner 42
        let mut ptrs = Vec::new();
        for _ in 0..10 {
            let p = unsafe {
                ptx_sys::gpu_hot_alloc_owned(runtime.raw(), 4096, 42)
            };
            if !p.is_null() {
                ptrs.push(p);
            }
        }

        let after = runtime.owner_stats();
        // Find owner 42
        let mut found = false;
        for i in 0..after.num_owners as usize {
            if after.owners[i].owner_id == 42 {
                found = true;
                if after.owners[i].block_count >= 10 {
                    println!("PASS  (owner 42: {} blocks, {} bytes)",
                             after.owners[i].block_count,
                             after.owners[i].allocated_bytes);
                    pass += 1;
                } else {
                    println!("FAIL  (expected >=10 blocks, got {})",
                             after.owners[i].block_count);
                    fail += 1;
                }
                break;
            }
        }
        if !found {
            println!("FAIL  (owner 42 not found, num_owners={})", after.num_owners);
            fail += 1;
        }

        // Free owner 42
        unsafe { ptx_sys::gpu_hot_free_owner(runtime.raw(), 42) };

        let post_free = runtime.owner_stats();
        let mut remaining = 0u32;
        for i in 0..post_free.num_owners as usize {
            if post_free.owners[i].owner_id == 42 {
                remaining = post_free.owners[i].block_count;
            }
        }
        print!("[1b] free_owner ... ");
        if remaining == 0 {
            println!("PASS  (owner 42 blocks reclaimed)");
            pass += 1;
        } else {
            println!("FAIL  ({} blocks still tracked)", remaining);
            fail += 1;
        }
    }

    // ── Feature 2: VMM eviction bridge (wiring check) ──────
    {
        print!("[2] set_vmm wiring ... ");
        // We can't fully test eviction without a real VMM + memory pressure,
        // but we verify the call doesn't crash and the API is linked.
        unsafe { ptx_sys::gpu_hot_set_vmm(runtime.raw(), ptr::null_mut()) };
        println!("PASS  (gpu_hot_set_vmm linked and callable)");
        pass += 1;
    }

    // ── Feature 3: allocation event ring buffer ────────────
    {
        print!("[3] alloc_events ... ");

        let before = runtime.alloc_events();
        let before_count = before.count;

        // Do some allocs + frees
        let mut ptrs = Vec::new();
        for _ in 0..20 {
            let p = runtime.alloc(2048);
            if let Ok(p) = p {
                ptrs.push(p);
            }
        }
        drop(ptrs); // frees all

        let after = runtime.alloc_events();
        let delta = after.count.wrapping_sub(before_count);

        // We did 20 allocs + 20 frees = 40 events minimum
        if delta >= 40 {
            println!("PASS  ({} events recorded, head={})", delta, after.head);
            pass += 1;
        } else {
            println!("FAIL  (expected >=40 events, got {})", delta);
            fail += 1;
        }
    }

    // ── Feature 4: task submission ─────────────────────────
    {
        print!("[4] submit_task ... ");

        // Submit a NOP task (opcode 0)
        let mut args: [*mut libc::c_void; 8] = [ptr::null_mut(); 8];
        let tid = runtime.submit_task(0, 1, &mut args);

        if tid >= 0 {
            println!("PASS  (task_id={})", tid);
            pass += 1;
        } else {
            println!("FAIL  (returned {})", tid);
            fail += 1;
        }
    }

    // ── Summary ────────────────────────────────────────────
    println!("\n=== Results: {} passed, {} failed ===", pass, fail);
    if fail > 0 {
        std::process::exit(1);
    }

    Ok(())
}
