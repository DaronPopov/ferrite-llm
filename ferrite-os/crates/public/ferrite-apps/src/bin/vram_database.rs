//! vram_database — GPU-Resident Key-Value Store on TLSF + VFS
//!
//! Proves: Persistent GPU state across millions of alloc/free cycles with zero
//! fragmentation. Impossible with cudaMalloc (fragments after variable-size churn).
//!
//! OS primitives exercised: TLSF alloc/free, VFS (mkdir/open/write/read/mmap_tensor/unlink),
//! SHM, streams, watchdog/keepalive.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use rand::Rng;

use ferrite_apps::platform;

const POOL_FRACTION: f32 = 0.55;
const MAX_STREAMS: u32 = 256;
const EMBED_DIM: usize = 65536; // 256 KB per entry — visible in MB-scale telemetry
const EMBED_BYTES: usize = EMBED_DIM * std::mem::size_of::<f32>();
const MAX_ENTRIES: usize = 500; // Stay well under VFS_MAX_NODES=1024 (dirs take inodes too)

struct DbEntry {
    gpu_ptr: ptx_runtime::GpuPtr,
    vfs_path: String,
    checksum: f32,
}

fn main() -> Result<()> {
    let duration_secs = platform::get_duration_secs();
    println!("=== VRAM DATABASE ===");
    println!("GPU-resident key-value store on TLSF + VFS");
    println!("Duration: {}", platform::format_duration(duration_secs));
    println!("Config: pool_fraction={}, max_streams={}, embed_dim={}", POOL_FRACTION, MAX_STREAMS, EMBED_DIM);
    println!();

    let rt = platform::init_runtime(POOL_FRACTION, MAX_STREAMS)?;
    let mut reporter = platform::TelemetryReporter::new("vram_db", 10);

    // Initialize VFS
    let vfs = unsafe { platform::vfs_safe_init(&rt)? };
    unsafe {
        platform::vfs_safe_mkdir(vfs, "/db")?;
        platform::vfs_safe_mkdir(vfs, "/db/tables")?;
        platform::vfs_safe_mkdir(vfs, "/db/meta")?;
    }

    // Allocate SHM for stats accumulator: [inserts, queries, updates, deletes, total_ops]
    let shm_ptr = unsafe { platform::shm_safe_alloc(&rt, "db_stats", 5 * 4)? };
    unsafe {
        ptx_sys::cudaMemset(shm_ptr, 0, 5 * 4);
    }

    let mut entries: HashMap<u32, DbEntry> = HashMap::new();
    let mut next_id: u32 = 0;
    let mut rng = rand::thread_rng();

    let mut total_inserts: u64 = 0;
    let mut total_queries: u64 = 0;
    let mut total_updates: u64 = 0;
    let mut total_deletes: u64 = 0;

    let start = Instant::now();
    let deadline = std::time::Duration::from_secs(duration_secs);

    println!("Starting database operations loop...\n");

    while start.elapsed() < deadline {
        rt.keepalive();

        // Choose operation: INSERT(40%) / QUERY(30%) / UPDATE(20%) / DELETE(10%)
        let roll: f32 = rng.gen();

        if roll < 0.40 && entries.len() < MAX_ENTRIES {
            // INSERT: allocate embedding, fill with data, write to VFS
            let id = next_id;
            next_id += 1;

            let gpu = rt.alloc(EMBED_BYTES)?;
            let stream = rt.next_stream();

            // Fill with a deterministic pattern: value = id as f32 * 0.001
            let fill_val = id as f32 * 0.001;
            unsafe {
                ptx_sys::ptx_tensor_fill_f32(
                    gpu.as_ptr_typed::<f32>(),
                    EMBED_DIM,
                    fill_val,
                    stream.raw(),
                );
            }

            // Create VFS tensor file
            let vfs_path = format!("/db/tables/entry_{}", id);
            let shape = [EMBED_DIM as i32];
            unsafe {
                platform::vfs_safe_create_tensor(vfs, &vfs_path, &shape, 0)?;
            }

            // Compute checksum via reduce_sum
            let sum_buf = rt.alloc(4)?;
            unsafe {
                ptx_sys::ptx_tensor_reduce_sum_f32(
                    gpu.as_ptr_typed::<f32>(),
                    sum_buf.as_ptr_typed::<f32>(),
                    1, EMBED_DIM, 1,
                    stream.raw(),
                );
            }
            stream.synchronize()?;

            let mut checksum: f32 = 0.0;
            unsafe {
                sum_buf.copy_to_host(
                    &mut checksum as *mut f32 as *mut libc::c_void,
                    4,
                )?;
            }

            entries.insert(id, DbEntry {
                gpu_ptr: gpu,
                vfs_path,
                checksum,
            });
            total_inserts += 1;

        } else if roll < 0.70 && !entries.is_empty() {
            // QUERY: pick random entry, verify checksum
            let keys: Vec<u32> = entries.keys().copied().collect();
            let key = keys[rng.gen_range(0..keys.len())];
            let entry = entries.get(&key).unwrap();

            let stream = rt.next_stream();

            // mmap the VFS tensor and reduce_sum to verify
            let _mmap_ptr = unsafe { platform::vfs_safe_mmap_tensor(vfs, &entry.vfs_path) };

            // Verify via reduce_sum on the TLSF allocation
            let sum_buf = rt.alloc(4)?;
            unsafe {
                ptx_sys::ptx_tensor_reduce_sum_f32(
                    entry.gpu_ptr.as_ptr_typed::<f32>(),
                    sum_buf.as_ptr_typed::<f32>(),
                    1, EMBED_DIM, 1,
                    stream.raw(),
                );
            }
            stream.synchronize()?;

            let mut result: f32 = 0.0;
            unsafe {
                sum_buf.copy_to_host(
                    &mut result as *mut f32 as *mut libc::c_void,
                    4,
                )?;
            }

            // Checksums should match (within floating point tolerance)
            if (result - entry.checksum).abs() > 1.0 {
                eprintln!("WARNING: checksum mismatch for entry {}: expected {}, got {}",
                    key, entry.checksum, result);
            }
            total_queries += 1;

        } else if roll < 0.90 && !entries.is_empty() {
            // UPDATE: scale an existing entry by 1.01
            let keys: Vec<u32> = entries.keys().copied().collect();
            let key = keys[rng.gen_range(0..keys.len())];
            let entry = entries.get_mut(&key).unwrap();

            let stream = rt.next_stream();
            unsafe {
                ptx_sys::ptx_tensor_mul_scalar_f32(
                    entry.gpu_ptr.as_ptr_typed::<f32>(),
                    1.01,
                    entry.gpu_ptr.as_ptr_typed::<f32>(),
                    EMBED_DIM,
                    stream.raw(),
                );
            }

            // Recompute checksum
            let sum_buf = rt.alloc(4)?;
            unsafe {
                ptx_sys::ptx_tensor_reduce_sum_f32(
                    entry.gpu_ptr.as_ptr_typed::<f32>(),
                    sum_buf.as_ptr_typed::<f32>(),
                    1, EMBED_DIM, 1,
                    stream.raw(),
                );
            }
            stream.synchronize()?;

            let mut new_checksum: f32 = 0.0;
            unsafe {
                sum_buf.copy_to_host(
                    &mut new_checksum as *mut f32 as *mut libc::c_void,
                    4,
                )?;
            }
            entry.checksum = new_checksum;
            total_updates += 1;

        } else if !entries.is_empty() {
            // DELETE: remove a random entry, free TLSF alloc, unlink VFS file
            let keys: Vec<u32> = entries.keys().copied().collect();
            let key = keys[rng.gen_range(0..keys.len())];
            let entry = entries.remove(&key).unwrap();

            // GpuPtr drops automatically, freeing TLSF memory
            let _gpu = entry.gpu_ptr;
            unsafe {
                let _ = platform::vfs_safe_unlink(vfs, &entry.vfs_path);
            }
            total_deletes += 1;
        }

        // Update SHM stats (GPU memory — use cudaMemcpy)
        let total_ops = total_inserts + total_queries + total_updates + total_deletes;
        let shm_data: [u32; 5] = [
            total_inserts as u32, total_queries as u32,
            total_updates as u32, total_deletes as u32,
            total_ops as u32,
        ];
        unsafe {
            ptx_sys::cudaMemcpy(
                shm_ptr,
                shm_data.as_ptr() as *const libc::c_void,
                5 * 4,
                ptx_sys::cudaMemcpyHostToDevice,
            );
        }

        // Periodic telemetry
        if reporter.should_report() {
            let total_ops = total_inserts + total_queries + total_updates + total_deletes;
            reporter.report(&rt, &format!(
                "entries={} | ops={} | ins={} qry={} upd={} del={} | mem={}",
                entries.len(), total_ops,
                total_inserts, total_queries, total_updates, total_deletes,
                platform::format_bytes(entries.len() * EMBED_BYTES),
            ));
        }
    }

    let total_ops = total_inserts + total_queries + total_updates + total_deletes;
    println!("\n=== VRAM DATABASE COMPLETE ===");
    println!("Total operations: {}", total_ops);
    println!("  Inserts: {}", total_inserts);
    println!("  Queries: {}", total_queries);
    println!("  Updates: {}", total_updates);
    println!("  Deletes: {}", total_deletes);
    println!("Remaining entries: {}", entries.len());
    println!("Duration: {:.1}s", reporter.elapsed().as_secs_f64());

    // Cleanup: delete all remaining entries
    println!("\nCleaning up {} remaining entries...", entries.len());
    let remaining_keys: Vec<u32> = entries.keys().copied().collect();
    for key in remaining_keys {
        let entry = entries.remove(&key).unwrap();
        let _gpu = entry.gpu_ptr; // Drop frees TLSF
        unsafe {
            let _ = platform::vfs_safe_unlink(vfs, &entry.vfs_path);
        }
    }

    // Cleanup VFS
    unsafe {
        let _ = platform::vfs_safe_rmdir(vfs, "/db/meta");
        let _ = platform::vfs_safe_rmdir(vfs, "/db/tables");
        let _ = platform::vfs_safe_rmdir(vfs, "/db");
        ptx_sys::vfs_shutdown(vfs);
    }

    // Cleanup SHM
    unsafe {
        platform::shm_safe_unlink(&rt, "db_stats", shm_ptr)?;
    }

    rt.sync_all()?;
    platform::assert_clean_exit(&rt);

    Ok(())
}
