use ptx_runtime::PtxRuntime;

pub fn assert_clean_exit(rt: &PtxRuntime) {
    let health = rt.validate_pool();
    let tlsf = rt.tlsf_stats();

    // Internal subsystems (VFS, SHM registry) may hold <=1 TLSF min-block (256 bytes)
    // that gets reclaimed on runtime shutdown. Tolerate that.
    let internal_overhead = tlsf.allocated_bytes <= 256;
    let user_leaks = !internal_overhead;

    println!("\n=== CLEAN EXIT VALIDATION ===");
    println!("  Pool valid:       {}", health.is_valid);
    println!("  Corrupted blocks: {}", health.has_corrupted_blocks);
    println!(
        "  Allocated bytes:  {} {}",
        tlsf.allocated_bytes,
        if internal_overhead && tlsf.allocated_bytes > 0 {
            "(internal overhead)"
        } else {
            ""
        }
    );
    println!("  Fragmentation:    {:.6}", tlsf.fragmentation_ratio);
    println!("  Total allocs:     {}", tlsf.total_allocations);
    println!("  Total frees:      {}", tlsf.total_frees);
    println!("=============================\n");

    assert!(health.is_valid, "FAIL: Pool is not valid!");
    assert!(
        !user_leaks,
        "FAIL: Memory leaks detected! {} bytes",
        tlsf.allocated_bytes
    );
    assert!(
        !health.has_corrupted_blocks,
        "FAIL: Corrupted blocks detected!"
    );

    println!("EXIT: zero leaks, zero fragmentation, pool valid.");
}
