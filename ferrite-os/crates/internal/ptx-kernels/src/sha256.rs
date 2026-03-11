//! SHA-256 Brute Force Kernels
//!
//! High-performance parallel hash search for cryptographic research.

use ptx_sys::cudaStream_t;

extern "C" {
    pub fn launch_sha256_bruteforce(
        base_message: *const u32,      // 16 words (64 bytes)
        nonce_start: u64,              // Starting nonce
        total_nonces: u64,             // How many nonces to test
        target_zeros: i32,             // Target leading zero bits
        found_nonce: *mut u64,         // Output: winning nonce
        found_hash: *mut u32,          // Output: winning hash (8 words)
        attempts: *mut u64,            // Output: attempts counter
        stream: cudaStream_t,
    );
}
