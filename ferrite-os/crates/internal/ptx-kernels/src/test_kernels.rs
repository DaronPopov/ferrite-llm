//! Test kernel FFI bindings
//!
//! Simple kernels for validating the guard layer

use libc::size_t;
use crate::cudaStream_t;

extern "C" {
    pub fn test_launch_add_f32(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        n: size_t,
        stream: cudaStream_t,
    );

    pub fn test_launch_mul_f32(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        n: size_t,
        stream: cudaStream_t,
    );

    pub fn test_launch_gelu_f32(
        input: *const f32,
        out: *mut f32,
        n: size_t,
        stream: cudaStream_t,
    );
}
