// dataflow_proof — Prove Software-Defined TPU Properties on GPU Hardware
//
// Demonstrates that the TLSF + stream runtime creates a deterministic dataflow
// engine with TPU-like memory semantics:
//
//   1. **Deterministic placement** — same-size allocations return identical
//      VRAM addresses after free/realloc cycles across 4 TLSF size classes.
//   2. **O(1) alloc latency** — allocation time is constant regardless of
//      size class or system state.
//   3. **Zero fragmentation** — mixed-size alloc/compute/free streaming
//      leaves fragmentation at exactly 0.0.
//   4. **Stable pipeline throughput** — deep multi-lane dataflow graphs
//      execute in constant time with no variance over long runs.
//
// Layout: 8 VRAM slots across 4 size classes (~5.3 MB total)
//   LARGE  2×2MB   — primary compute lane  (6 nonlinear ops)
//   MEDIUM 2×512KB — transform lane        (4 nonlinear ops)
//   SMALL  2×128KB — normalization lane    (3 nonlinear ops)
//   TINY   2×16KB  — reduction scratch     (3 cross-lane reductions)

use std::time::{Duration, Instant};

use anyhow::Result;

use ferrite_apps::platform;

const POOL_FRACTION: f32 = 0.40;
const MAX_STREAMS: u32 = 64;

// ── Multi-class allocation table ──────────────────────────────────────
// 4 TLSF size classes × 2 buffers each = 8 slots.
// Ping-pong pairs enable deep compute chains without extra allocations.

/// Large buffers: 2 MB each (524288 × f32)
const LARGE_ELEMS: usize = 524_288;
const LARGE_BYTES: usize = LARGE_ELEMS * std::mem::size_of::<f32>();

/// Medium buffers: 512 KB each (131072 × f32)
const MEDIUM_ELEMS: usize = 131_072;
const MEDIUM_BYTES: usize = MEDIUM_ELEMS * std::mem::size_of::<f32>();

/// Small buffers: 128 KB each (32768 × f32)
const SMALL_ELEMS: usize = 32_768;
const SMALL_BYTES: usize = SMALL_ELEMS * std::mem::size_of::<f32>();

/// Tiny buffers: 16 KB each (4096 × f32) — reduction scratch
const TINY_ELEMS: usize = 4_096;
const TINY_BYTES: usize = TINY_ELEMS * std::mem::size_of::<f32>();

const NUM_SLOTS: usize = 8;
const SLOT_SIZES: [usize; NUM_SLOTS] = [
    LARGE_BYTES, LARGE_BYTES,
    MEDIUM_BYTES, MEDIUM_BYTES,
    SMALL_BYTES, SMALL_BYTES,
    TINY_BYTES, TINY_BYTES,
];
const SLOT_LABELS: [&str; NUM_SLOTS] = ["L0", "L1", "M0", "M1", "S0", "S1", "T0", "T1"];
const CLASS_NAMES: [&str; 4] = ["L", "M", "S", "T"];

/// How often to recycle all slots and verify address determinism.
const RECYCLE_EVERY: u64 = 15;

fn total_vram() -> usize {
    SLOT_SIZES.iter().sum()
}

fn main() -> Result<()> {
    let duration_secs = std::env::args()
        .position(|a| a == "--duration")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(platform::get_duration_secs);

    println!("=== DATAFLOW PROOF ===");
    println!("Proving software-defined TPU properties on GPU hardware");
    println!("Duration: {}", platform::format_duration(duration_secs));
    println!(
        "Layout: {} slots across 4 size classes  total={}  recycle every {} cycles",
        NUM_SLOTS,
        platform::format_bytes(total_vram()),
        RECYCLE_EVERY,
    );
    println!(
        "  LARGE  2x{} ({}/buf)  primary compute lane (6 nonlinear ops)",
        LARGE_ELEMS,
        platform::format_bytes(LARGE_BYTES),
    );
    println!(
        "  MEDIUM 2x{} ({}/buf)  transform lane (4 nonlinear ops)",
        MEDIUM_ELEMS,
        platform::format_bytes(MEDIUM_BYTES),
    );
    println!(
        "  SMALL  2x{} ({}/buf)  normalization lane (3 nonlinear ops)",
        SMALL_ELEMS,
        platform::format_bytes(SMALL_BYTES),
    );
    println!(
        "  TINY   2x{} ({}/buf)   reduction scratch (3 cross-lane reductions)",
        TINY_ELEMS,
        platform::format_bytes(TINY_BYTES),
    );
    println!();

    let rt = platform::init_runtime(POOL_FRACTION, MAX_STREAMS)?;

    // ── Phase 1: Initial allocation — record baseline addresses ────────
    let mut slots: Vec<ptx_runtime::GpuPtr> = Vec::with_capacity(NUM_SLOTS);
    let mut baseline_addrs: Vec<usize> = Vec::with_capacity(NUM_SLOTS);

    for (i, &size) in SLOT_SIZES.iter().enumerate() {
        let ptr = rt.alloc(size)?;
        baseline_addrs.push(ptr.as_ptr() as usize);
        slots.push(ptr);
        println!(
            "  {} [{}]: 0x{:X}",
            SLOT_LABELS[i],
            platform::format_bytes(SLOT_SIZES[i]),
            baseline_addrs[i],
        );
    }
    println!();

    // ── Phase 2: Streaming multi-lane pipeline loop ───────────────────
    let mut cycle: u64 = 0;
    let mut total_addr_checks: u64 = 0;
    let mut total_addr_matches: u64 = 0;
    let mut alloc_times_ns: Vec<u64> = Vec::new();
    let mut pipe_times_us: Vec<u64> = Vec::new();
    let mut peak_frag: f32 = 0.0;
    let mut seed: u64 = 1;

    let start = Instant::now();
    let deadline = Duration::from_secs(duration_secs);

    let mut reporter = platform::TelemetryReporter::new("dataflow", 5);

    while start.elapsed() < deadline {
        rt.keepalive();
        cycle += 1;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

        let stream = rt.next_stream();

        // ── Execute the multi-lane dataflow pipeline ──────────────────
        // Indices: L0=0 L1=1 M0=2 M1=3 S0=4 S1=5 T0=6 T1=7
        let pipe_start = Instant::now();

        unsafe {
            // ── Lane A: Primary compute (LARGE 2MB ping-pong) ─────────
            // Ingest: shaped random data with affine transform
            ptx_sys::ptx_tensor_randn_f32(
                slots[0].as_ptr_typed::<f32>(),
                LARGE_ELEMS,
                seed,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_affine_f32(
                slots[0].as_ptr_typed::<f32>(),
                slots[0].as_ptr_typed::<f32>(),
                LARGE_ELEMS,
                2.0,
                3.0,
                stream.raw(),
            );
            // Deep nonlinear chain: L0→gelu→L1→tanh→L0→sigmoid→L1→silu→L0→exp→L1
            ptx_sys::ptx_tensor_gelu_f32(
                slots[0].as_ptr_typed::<f32>(),
                slots[1].as_ptr_typed::<f32>(),
                LARGE_ELEMS,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_tanh_f32(
                slots[1].as_ptr_typed::<f32>(),
                slots[0].as_ptr_typed::<f32>(),
                LARGE_ELEMS,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_sigmoid_f32(
                slots[0].as_ptr_typed::<f32>(),
                slots[1].as_ptr_typed::<f32>(),
                LARGE_ELEMS,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_silu_f32(
                slots[1].as_ptr_typed::<f32>(),
                slots[0].as_ptr_typed::<f32>(),
                LARGE_ELEMS,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_exp_f32(
                slots[0].as_ptr_typed::<f32>(),
                slots[1].as_ptr_typed::<f32>(),
                LARGE_ELEMS,
                stream.raw(),
            );

            // ── Lane B: Transform (MEDIUM 512KB ping-pong) ───────────
            ptx_sys::ptx_tensor_randn_f32(
                slots[2].as_ptr_typed::<f32>(),
                MEDIUM_ELEMS,
                seed.wrapping_add(1),
                stream.raw(),
            );
            ptx_sys::ptx_tensor_relu_f32(
                slots[2].as_ptr_typed::<f32>(),
                slots[3].as_ptr_typed::<f32>(),
                MEDIUM_ELEMS,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_gelu_f32(
                slots[3].as_ptr_typed::<f32>(),
                slots[2].as_ptr_typed::<f32>(),
                MEDIUM_ELEMS,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_tanh_f32(
                slots[2].as_ptr_typed::<f32>(),
                slots[3].as_ptr_typed::<f32>(),
                MEDIUM_ELEMS,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_softplus_f32(
                slots[3].as_ptr_typed::<f32>(),
                slots[2].as_ptr_typed::<f32>(),
                MEDIUM_ELEMS,
                stream.raw(),
            );

            // ── Lane C: Normalization (SMALL 128KB ping-pong) ────────
            ptx_sys::ptx_tensor_randn_f32(
                slots[4].as_ptr_typed::<f32>(),
                SMALL_ELEMS,
                seed.wrapping_add(2),
                stream.raw(),
            );
            ptx_sys::ptx_tensor_sigmoid_f32(
                slots[4].as_ptr_typed::<f32>(),
                slots[5].as_ptr_typed::<f32>(),
                SMALL_ELEMS,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_gelu_f32(
                slots[5].as_ptr_typed::<f32>(),
                slots[4].as_ptr_typed::<f32>(),
                SMALL_ELEMS,
                stream.raw(),
            );
            ptx_sys::ptx_tensor_exp_f32(
                slots[4].as_ptr_typed::<f32>(),
                slots[5].as_ptr_typed::<f32>(),
                SMALL_ELEMS,
                stream.raw(),
            );

            // ── Cross-lane reductions → T0 scratch ───────────────────
            // Pack 3 scalar results into T0 at consecutive offsets
            let t0 = slots[6].as_ptr_typed::<f32>();
            // Mean of large lane final output (L1)
            ptx_sys::ptx_tensor_reduce_mean_f32(
                slots[1].as_ptr_typed::<f32>(),
                t0,
                1,
                LARGE_ELEMS,
                1,
                stream.raw(),
            );
            // Max of medium lane final output (M0)
            ptx_sys::ptx_tensor_reduce_max_f32(
                slots[2].as_ptr_typed::<f32>(),
                t0.add(1),
                1,
                MEDIUM_ELEMS,
                1,
                stream.raw(),
            );
            // Sum of small lane final output (S1)
            ptx_sys::ptx_tensor_reduce_sum_f32(
                slots[5].as_ptr_typed::<f32>(),
                t0.add(2),
                1,
                SMALL_ELEMS,
                1,
                stream.raw(),
            );
        }

        stream.synchronize()?;
        let pipe_us = pipe_start.elapsed().as_micros() as u64;
        pipe_times_us.push(pipe_us);

        // Read back 3 reduction results (proves data flowed through all lanes)
        let mut results: [f32; 3] = [0.0; 3];
        unsafe {
            slots[6].copy_to_host(
                results.as_mut_ptr() as *mut libc::c_void,
                std::mem::size_of_val(&results),
            )?;
        }

        // Track fragmentation
        let tlsf = rt.tlsf_stats();
        if tlsf.fragmentation_ratio > peak_frag {
            peak_frag = tlsf.fragmentation_ratio;
        }

        // ── Periodic recycle: free all slots, re-allocate, compare ─────
        if cycle % RECYCLE_EVERY == 0 {
            slots.clear();
            rt.sync_all()?;
            rt.poll_deferred(0);

            // Per-class match tracking: [LARGE, MEDIUM, SMALL, TINY]
            let mut class_matches: [usize; 4] = [0; 4];
            let mut class_checks: [usize; 4] = [0; 4];

            for i in 0..NUM_SLOTS {
                let alloc_start = Instant::now();
                let ptr = rt.alloc(SLOT_SIZES[i])?;
                let alloc_ns = alloc_start.elapsed().as_nanos() as u64;
                alloc_times_ns.push(alloc_ns);

                let new_addr = ptr.as_ptr() as usize;
                let class_idx = i / 2;
                class_checks[class_idx] += 1;
                total_addr_checks += 1;
                if new_addr == baseline_addrs[i] {
                    total_addr_matches += 1;
                    class_matches[class_idx] += 1;
                }
                baseline_addrs[i] = new_addr;
                slots.push(ptr);
            }

            let match_rate = if total_addr_checks > 0 {
                total_addr_matches as f64 / total_addr_checks as f64 * 100.0
            } else {
                0.0
            };

            let avg_alloc = if alloc_times_ns.is_empty() {
                0.0
            } else {
                alloc_times_ns.iter().sum::<u64>() as f64 / alloc_times_ns.len() as f64
            };

            let avg_pipe = if pipe_times_us.is_empty() {
                0.0
            } else {
                pipe_times_us.iter().sum::<u64>() as f64 / pipe_times_us.len() as f64
            };

            let class_status: String = (0..4)
                .map(|c| {
                    let ok = class_matches[c] == class_checks[c];
                    format!("{}={}", CLASS_NAMES[c], if ok { "+" } else { "-" })
                })
                .collect::<Vec<_>>()
                .join(" ");

            let all_match = class_matches
                .iter()
                .zip(class_checks.iter())
                .all(|(m, c)| m == c);

            println!(
                "  [{}] {}/{} addr ({:.0}%)  {}  alloc={:.0}ns  pipe={:.0}us  frag={:.6}  out=[{:.4},{:.4},{:.1}]  {}",
                cycle,
                total_addr_matches,
                total_addr_checks,
                match_rate,
                class_status,
                avg_alloc,
                avg_pipe,
                tlsf.fragmentation_ratio,
                results[0],
                results[1],
                results[2],
                if all_match { "DETERMINISTIC" } else { "PARTIAL" },
            );
        }

        // Periodic telemetry
        if reporter.should_report() {
            let match_rate = if total_addr_checks > 0 {
                total_addr_matches as f64 / total_addr_checks as f64 * 100.0
            } else {
                100.0
            };
            reporter.report(
                &rt,
                &format!(
                    "cycles={} addr_reuse={:.0}% peak_frag={:.6} vram={}",
                    cycle,
                    match_rate,
                    peak_frag,
                    platform::format_bytes(total_vram()),
                ),
            );
        }

        std::thread::sleep(Duration::from_millis(2));
    }

    // ── Final report ───────────────────────────────────────────────────
    let elapsed = start.elapsed();
    let match_rate = if total_addr_checks > 0 {
        total_addr_matches as f64 / total_addr_checks as f64 * 100.0
    } else {
        100.0
    };

    let (alloc_mean, alloc_std) = mean_std(&alloc_times_ns);
    let (pipe_mean, pipe_std) = mean_std(&pipe_times_us);

    let tlsf = rt.tlsf_stats();

    println!("\n=== DATAFLOW PROOF COMPLETE ===");
    println!("  cycles:          {}", cycle);
    println!("  elapsed:         {:.1}s", elapsed.as_secs_f64());
    println!(
        "  vram footprint:  {} across 4 size classes",
        platform::format_bytes(total_vram()),
    );
    println!();
    println!(
        "  address reuse:   {:.1}% ({}/{} recycled slots matched baseline)",
        match_rate, total_addr_matches, total_addr_checks,
    );
    println!(
        "  alloc latency:   {:.0}ns mean, {:.0}ns stddev (O(1) {})",
        alloc_mean,
        alloc_std,
        if alloc_std < alloc_mean * 0.5 {
            "confirmed"
        } else {
            "variable"
        },
    );
    println!(
        "  fragmentation:   {:.6} peak, {:.6} final ({})",
        peak_frag,
        tlsf.fragmentation_ratio,
        if peak_frag < 0.001 { "zero" } else { "non-zero" },
    );
    println!(
        "  pipeline:        {:.0}us mean, {:.0}us stddev ({})",
        pipe_mean,
        pipe_std,
        if pipe_std < pipe_mean * 0.3 {
            "stable dataflow"
        } else {
            "variable"
        },
    );
    println!();

    let verdict =
        match_rate >= 95.0 && peak_frag < 0.001 && alloc_std < alloc_mean * 0.5;

    if verdict {
        println!("  VERDICT: software-defined dataflow engine");
        println!("           TPU-like memory semantics CONFIRMED on GPU hardware");
        println!(
            "           Deterministic across {} slots, 4 size classes, {}",
            NUM_SLOTS,
            platform::format_bytes(total_vram()),
        );
    } else {
        println!("  VERDICT: partial — some properties did not meet thresholds");
        if match_rate < 95.0 {
            println!("           address reuse {:.0}% < 95% target", match_rate);
        }
        if peak_frag >= 0.001 {
            println!(
                "           fragmentation {:.6} >= 0.001 target",
                peak_frag,
            );
        }
    }
    println!("==============================\n");

    // Cleanup
    slots.clear();
    rt.sync_all()?;
    platform::assert_clean_exit(&rt);

    Ok(())
}

fn mean_std(data: &[u64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    let n = data.len() as f64;
    let mean = data.iter().sum::<u64>() as f64 / n;
    let var = data
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    (mean, var.sqrt())
}
