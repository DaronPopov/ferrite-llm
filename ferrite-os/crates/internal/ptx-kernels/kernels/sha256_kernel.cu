// SHA-256 Brute Force Kernel
// Each thread tests a different nonce to find hashes with leading zeros

#include <cstdint>
#include <cstdio>

// SHA-256 constants
__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 functions
__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Simplified SHA-256 for fixed 64-byte input (message + nonce)
__device__ void sha256_transform(const uint32_t* message, uint32_t* hash) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    // Prepare message schedule
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = message[i];
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    // Initialize working variables
    a = 0x6a09e667;
    b = 0xbb67ae85;
    c = 0x3c6ef372;
    d = 0xa54ff53a;
    e = 0x510e527f;
    f = 0x9b05688c;
    g = 0x1f83d9ab;
    h = 0x5be0cd19;

    // Main loop
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Output hash
    hash[0] = 0x6a09e667 + a;
    hash[1] = 0xbb67ae85 + b;
    hash[2] = 0x3c6ef372 + c;
    hash[3] = 0xa54ff53a + d;
    hash[4] = 0x510e527f + e;
    hash[5] = 0x9b05688c + f;
    hash[6] = 0x1f83d9ab + g;
    hash[7] = 0x5be0cd19 + h;
}

// Count leading zero bits in hash
__device__ int count_leading_zeros(const uint32_t* hash) {
    int zeros = 0;
    for (int i = 0; i < 8; i++) {
        uint32_t word = hash[i];
        if (word == 0) {
            zeros += 32;
        } else {
            zeros += __clz(word);  // Count leading zeros in word
            break;
        }
    }
    return zeros;
}

// Brute force kernel - each thread tests nonces in its range
__global__ void sha256_bruteforce(
    const uint32_t* base_message,  // 16 words (64 bytes) - base message
    uint64_t nonce_start,          // Starting nonce for this kernel
    uint64_t nonces_per_thread,    // How many nonces each thread tests
    int target_zeros,              // How many leading zero bits we want
    uint64_t* found_nonce,         // Output: nonce that meets criteria (atomic)
    uint32_t* found_hash,          // Output: the hash that was found
    uint64_t* attempts             // Output: total attempts made
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t thread_nonce_start = nonce_start + (idx * nonces_per_thread);

    uint32_t message[16];
    uint32_t hash[8];

    // Copy base message to local memory
    #pragma unroll
    for (int i = 0; i < 14; i++) {
        message[i] = base_message[i];
    }

    // Try nonces in this thread's range
    for (uint64_t i = 0; i < nonces_per_thread; i++) {
        uint64_t nonce = thread_nonce_start + i;

        // Embed nonce in message (last 2 words)
        message[14] = (uint32_t)(nonce >> 32);
        message[15] = (uint32_t)(nonce & 0xFFFFFFFF);

        // Compute hash
        sha256_transform(message, hash);

        // Check if we found a solution
        int zeros = count_leading_zeros(hash);

        if (zeros >= target_zeros) {
            // Found it! Use atomic CAS to claim victory
            unsigned long long old = atomicCAS(
                (unsigned long long*)found_nonce,
                0xFFFFFFFFFFFFFFFFULL,
                (unsigned long long)nonce
            );
            if (old == 0xFFFFFFFFFFFFFFFFULL) {
                // We were first! Store the hash
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    found_hash[j] = hash[j];
                }
            }
            // Early exit - someone found it
            break;
        }
    }

    // Update attempt counter
    atomicAdd((unsigned long long*)attempts, nonces_per_thread);
}

// C wrapper for Rust FFI
extern "C" void launch_sha256_bruteforce(
    const uint32_t* base_message,
    uint64_t nonce_start,
    uint64_t total_nonces,
    int target_zeros,
    uint64_t* found_nonce,
    uint32_t* found_hash,
    uint64_t* attempts,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = 1024;  // Each kernel gets 256K threads
    const uint64_t threads_total = (uint64_t)threads * blocks;
    const uint64_t nonces_per_thread = (total_nonces + threads_total - 1) / threads_total;

    sha256_bruteforce<<<blocks, threads, 0, stream>>>(
        base_message,
        nonce_start,
        nonces_per_thread,
        target_zeros,
        found_nonce,
        found_hash,
        attempts
    );
}
