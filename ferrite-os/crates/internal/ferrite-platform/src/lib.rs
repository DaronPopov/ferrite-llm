//! Thin platform boundary for Ferrite-OS.
//!
//! All OS-specific logic lives here behind small, concrete function APIs.
//! Core business crates call into this module instead of using `std::os::unix`,
//! `libc`, or conditional compilation directly.

pub mod dylib_env;
pub mod ipc;
pub mod paths;
pub mod pid;
pub mod signals;
pub mod tty;
